Preprint. 

# - **BRIGHT** : A REALISTIC AND CHALLENGING BENCH MARK FOR REASONING-INTENSIVE RETRIEVAL 

**Hongjin Su** _[∗]_ _**[h]**_ **Howard Yen** _[∗]_ _**[p]**_ **Mengzhou Xia** _[∗]_ _**[p]**_ **Weijia Shi** _**[w]**_ **Niklas Muennighoff** _**[s]**_ **Han-yu Wang** _**[h]**_ **Haisu Liu** _**[h]**_ **Quan Shi** _**[p]**_ **Zachary S. Siegel** _**[p]**_ **Michael Tang** _**[p]**_ **Ruoxi Sun** _**[g]**_ **Jinsung Yoon** _**[g]**_ **Sercan Ö. Arık** _**[g]**_ **Danqi Chen** _**[p]**_ **Tao Yu** _**[h]**_ 

> _**h**_ The University of Hong Kong _**p**_ Princeton University _**s**_ Stanford University 

> _**w**_ University of Washington _**g**_ Google Cloud AI Research 

{hjsu,tyu}@cs.hku.hk {hyen,mengzhou,danqic}@cs.princeton.edu 

## ABSTRACT 

Existing retrieval benchmarks primarily consist of information-seeking queries (e.g., aggregated questions from search engines) where keyword or semantic-based retrieval is usually sufficient. However, many complex real-world queries require in-depth reasoning to identify relevant documents that go beyond surface form matching. For example, finding documentation for a coding question requires understanding the logic and syntax of the functions involved. To better benchmark retrieval on such challenging queries, we introduce **BRIGHT** , the first text retrieval benchmark that requires _intensive reasoning_ to retrieve relevant documents. Our dataset consists of 1,384 real-world queries spanning diverse domains, such as economics, psychology, mathematics, and coding. These queries are drawn from naturally occurring and carefully curated human data. Extensive evaluation reveals that even state-of-the-art retrieval models perform poorly on **BRIGHT** . The leading model on the MTEB leaderboard (Muennighoff et al., 2023), which achieves a score of 59.0 nDCG@10,[1] produces a score of nDCG@10 of 18.3 on **BRIGHT** . We show that incorporating explicit reasoning about the query improves retrieval performance by up to 12.2 points. Moreover, incorporating retrieved documents from the top-performing retriever boosts question-answering performance by over 6.6 points. We believe that **BRIGHT** paves the way for future research on retrieval systems in more realistic and challenging settings.[2] 

## 1 INTRODUCTION 

Information retrieval is a widely employed technology that assists users in locating relevant information from extensive corpora, containing documents, web pages, and logging records (Bajaj et al., 2016; Thakur et al., 2021). Relevant information can relate to user queries in different ways—sometimes through straightforward matching patterns like shared keywords or semantic similarities, and other times through deeper, more nuanced connections such as analogous underlying principles. In many real-world scenarios, user queries can be highly complex, and finding the relevant documents requires intensive reasoning. For instance, an economist might want to find a story explained by the same economic theory as another story, or a programmer might want to use an error message to locate the corresponding syntax documentation. For these applications, relevant documents cannot be directly retrieved through lexical or semantic matching alone, but instead require additional reasoning steps to identify. 

In this work, we study the problem of reasoning-intensive retrieval with **BRIGHT** , a new benchmark that requires intensive reasoning to retrieve relevant documents. Existing retrieval benchmarks, such as BEIR (Thakur et al., 2021) and MTEB (Muennighoff et al., 2023), primarily focus on fact-based queries typically derived from search engines, where the relevance between queries and documents 

> _∗_ Equal contribution. 

> 1Retrieved from the MTEB leaderboard on 2024-05-28. 

> 2 Our code and data are available at https://brightbenchmark.github.io. 

1 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval_images/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Level 1: Keyword-based Retrieval<br>Positive Document<br>Query<br>What is the widest highway in North America? Relevance: 
 The part of highway 401 that passes through Toronto is North America's busiest highway, and one of the<br>Keyword matching widest.<br>Level 2: Semantic-based Retrieval Natural Question, Kwiatkowski et al. (2019)<br>Positive Document<br>QueHowry human activities influence climate system? Relevance: 
 Deforestation and urbanization result in increased emissions, urban heat island effects and changes in<br>Semantic matching<br>natural water cycle.<br>Level 3: Reasoning-based Retrieval - BRIGHT MS MARCO, Bajaj et al. (2018)<br>Query Positive Document<br>Sustainable Living - post Relevance: 
 Sustainable Living - post<br>At home, after I water my plants, the Risk of using recycled  Soluble salts are commonly found in soils.<br>water goes to plates below the pots. plant water. When they build up, they destroy the soil<br>Can I reuse it for my plants next time? structure and cause direct damage to roots ..<br>Code - issue Code - issue<br>I have this table and need to transform it Relevance: 
 The function FLATTEN flattens (explodes)<br>to ... I don’t like UNPIVOT. Is there a Alternative function. compound values into multiple rows ...<br>better function in snowflake for this? FLATTEN( INPUT => <expr> ...<br>MATH - question MATH - question<br>Let k=2008^2+2^2008. What is the units Relevance: 
 Determine all positive integers relatively<br>digit of k^2+2^k? Uses the same theorem. prime to all the terms of the infinite<br>sequence a_n=2^n+3^n+6^n -1...<br>**----- End of picture text -----**<br>


Figure 1: Existing retrieval benchmarks focus on keyword-based retrieval (level 1), or semantic-based retrieval (level 2), e.g., NQ, MS MARCO datasets (Kwiatkowski et al., 2019; Bajaj et al., 2018). **BRIGHT introduces level 3 retrieval, where the relevance between queries and documents requires intensive reasoning to determine.** Our data consists of natural user queries from diverse domains (e.g., economics, math, earth sciences, etc.). **BRIGHT** corpora also span across web data, such as blogs, syntax documentation, and STEM problem-solutions. 

is often straightforward and can be detected through simple keyword or semantic matching (Lee et al., 2019; Karpukhin et al., 2020). These datasets focus on retrieving specific pieces of information (e.g., “the widest highway in North America”), which leads to the relevant documents often having high lexical or semantic overlap with the queries. In contrast, the relevance between queries and documents in **BRIGHT** is not easily detectable through simple keyword or semantic matching, and requires deliberate reasoning due to the complex nature of our domains and queries (Figure 1). 

**BRIGHT** consists of 12 datasets from diverse and advanced domains, sourced from naturally occurring human data and meticulously curated sources. The benchmark comprises two main components: 1) seven datasets are constructed from StackExchange, where relevance is defined by whether a document is cited in the answer, and further validated through multiple annotators to ensure that the positive documents effectively support addressing queries. Given the inherent subjectivity of this assessment, we only include documents unanimously agreed upon by the annotators. 2) The remaining five datasets focus on coding and math problems, where the queries are inherently linked to the positive documents due to their shared underlying algorithms or theorems. 

We evaluate 13 representative retrieval models of varying sizes and architectures. Comprehensive experiments highlight the challenges posed by **BRIGHT** , as the best-performing model, SFR-EmbeddingMistral (Meng et al., 2024), which scores 59.0 on the MTEB retrieval subset, BEIR (Thakur et al., 2021), achieves only an nDCG@10 score of 18.3 on **BRIGHT** . To identify promising directions for improving **BRIGHT** , we explore various strategies—one effective approach leverages LLMs to generate chain of thought reasoning steps (Wei et al., 2022) about the query before retrieval, resulting in average improvements of up to 12.2 points. Furthermore, augmenting the downstream model with retrieved documents from the top-performing retriever, Qwen, significantly enhances the question-answering performance by up to 6.6 points compared to the closed-book setting. However, using the oracle documents results in a 17.2-point improvement, highlighting the potential benefits of improving retriever models. Moreover, our results demonstrate the robustness of **BRIGHT** against potential data leakage during large-scale pre-training, as no substantial performance gains are observed even when models are further trained on documents from the benchmark dataset. We hope that our findings inspire research directions to advance the state of the art in reasoning-intensive retrieval. 

2 

Preprint. 

## 2 RELATED WORK 

**Benchmarking retrieval.** Existing information retrieval (IR) datasets are typically constructed for information-seeking tasks, such as question-answering (Voorhees & Tice, 2000; Craswell et al., 2020; Kwiatkowski et al., 2019; Chen et al., 2017; Maia et al., 2018), claim verification (Thorne et al., 2018; Diggelmann et al., 2020; Wadden et al., 2020), or entity retrieval (Hasibi et al., 2017; Petroni et al., 2021). Recent works expand retrieval benchmarks with more scenarios, such as instruction-following (Su et al., 2023; Weller et al., 2024; Oh et al., 2024; Li et al., 2024), multi-hop (Yang et al., 2018), and long-context retrieval (Saad-Falcon et al., 2024; Zhu et al., 2024). Comprehensive benchmarks like BEIR (Thakur et al., 2021) evaluate retrieval systems on diverse domains and tasks, with relevant documents sharing high semantic overlap with the query. Closest to our work, BIRCO (Wang et al., 2024) is designed to evaluate retrieval systems based on multifaceted objectives by leveraging existing datasets. However, it is limited to the LLM reranking setting and uses only a small candidate pool ( _∼_ 100 documents) for each query. RAR-b (Xiao et al., 2024) adapts existing commonsense, math, and code datasets into a retrieval setting to test whether models can directly retrieve answers to reasoning problems. However, we focus on a more realistic scenario where the answers are unlikely to be found as a substring in documents. While both of the benchmarks focus on reasoning-intensive retrieval, **BRIGHT** is the first benchmark to collect realistic user queries and align them with relevant natural documents from large corpora, requiring deep reasoning to identify the correct matches. 

**Dense retrieval models and retrieval augmented generation.** State-of-the-art retrieval systems often use dense models to encode text with rich representation. These models are trained on unsupervised data (Lee et al., 2019; Izacard et al., 2022), supervised data (Su et al., 2023; Asai et al., 2023a; Muennighoff, 2022), as well as LLM-generated data (Lee et al., 2024; Wang et al., 2023a; Muennighoff et al., 2024). In this work, we benchmark a diverse set of models across different axes: sparse and dense; small and large; open-source and proprietary. Additionally, as dense generative models continue to improve, retrieval-augmented generation (RAG; Asai et al., 2020; Borgeaud et al., 2022; Asai et al., 2023b; Muennighoff et al., 2024; Asai et al., 2024; Gao et al., 2023; Shi et al., 2024), which retrieves relevant documents to help generate coherent answers, has become an important application. Although we mainly focus on retrieval in this work, we conduct initial analyses and demonstrate that using stronger retrievers improves model generation for reasoning-intensive tasks. 

**Benchmarking reasoning.** Many benchmarks aim to evaluate the reasoning abilities of LLMs, especially focused on mathematics and coding. As for mathematics, for example, datasets include GSM8K (Cobbe et al., 2021) and its extensions GSM1K (Zhang et al., 2024), TheoremQA (Chen et al., 2023a), MATH (Hendrycks et al., 2021), and LeanDojo (Yang et al., 2023). As for coding, HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and LiveCodeBench (Jain et al., 2024) are often used. These benchmarks contain question-answer pairs and are usually sourced from textbooks, online resources, competitions, or domain experts. We source queries from selected high-quality datasets and construct **BRIGHT** through additional annotations, creating a realistic reasoning-intensive retrieval benchmark. 

## 3 CONSTRUCTING **BRIGHT** 

We introduce **BRIGHT** , a retrieval benchmark that tests whether retrieval systems can match queries and documents whose relevance requires intensive reasoning to solve, beyond just lexical and semantic similarities. In this section, we first formulate the task of reasoning-intensive retrieval (§3.1). Then, we detail the data collection process for the data from StackExchange (§3.2), coding datasets (§3.3), and theorem-based questions (§3.4). In Table 1, we present the benchmark statistics. 

## 3.1 TASK FORMULATION 

Given a query _Q_ and the retrieval corpus _D_ = _{D_ 1 _, . . . , Dn}_ , retrievers are tasked to find relevant documents _DQ_[+][=] _[{][D] Q,_[+] 1 _[, . . . , D] Q,m_[+] _[}][⊂D]_[,][where] _[m][≪][n]_[(positive).][Negative][documents][are] defined as _DQ[−]_[=] _[D][\][D] Q_[+][.][In][reasoning-intensive][retrieval,][the][relevant][document][set] _[D] Q_[+][is] connected to the query _Q_ through specific reasoning traces or explanations (e.g., underlying principles, algorithms, or theorems) related to the query. For instance, common reasoning traces might involve identifying the query’s intent, analyzing and modeling the problem, and drawing sub-conclusions 

3 

Preprint. 

Table 1: **Data statistics of BRIGHT.** For each dataset, we show the number of queries ( **Q** ) and documents ( _**D**_ ), the average number of positive documents ( _**D**_ **[+]** ) per example, the average length of queries and documents (measured by the GPT-2 tokenizer Radford et al. (2019)), and sources of queries and documents. Q&Sol refers to demonstration examples of question-solution pairs. TheoremQA-Q and TheoremQA-T refer to question retrieval and theorem retrieval based on TheoremQA respectively. Examples for each dataset can be found in Appendix C. 

|**Dataset**|**Total Number**|**Avg. Length**|**Source**|**Source**|
|---|---|---|---|---|
||**Q**<br>**_D_**<br>**_D_+**<br>**Q**<br>**_D_**<br>**Q**|||**_D_**|
||**_StackExchange_**||||
|Biology<br>Earth Science<br>Economics<br>Psychology<br>Robotics<br>Stack Overfow<br>Sustainable Living|103<br>57,359<br>3.6<br>115.2<br>83.6<br>Tab.20<br>116<br>121,249<br>5.3<br>109.5<br>132.6<br>Web pages:<br>article,<br>tutorial,<br>news, blog,<br>report ...<br>Tab.21<br>103<br>50,220<br>8.0<br>181.5<br>120.2<br>StackExchange<br>post<br>Tab.22<br>101<br>52,835<br>7.3<br>149.6<br>118.2<br>Tab.23<br>101<br>61,961<br>5.5<br>818.9<br>121.0<br>Tab.24<br>117<br>107,081<br>7.0<br>478.3<br>704.7<br>Tab.25<br>108<br>60,792<br>5.6<br>148.5<br>107.9<br>Tab.26||||
||**_Coding_**||||
|LeetCode<br>Pony|142<br>413,932<br>1.8<br>497.5<br>482.6<br>Coding question<br>Coding Q&Sol<br>Tab.27<br>112<br>7,894<br>22.5<br>102.6<br>98.3<br>Coding question<br>Syntax Doc<br>Tab.28||||
||**_Theorems_**||||
|AoPS<br>TheoremQA-Q<br>TheoremQA-T|111<br>188,002<br>4.7<br>117.1<br>250.5<br>Math Olympiad Q<br>STEM Q&Sol<br>Tab.29<br>194<br>188,002<br>3.2<br>93.4<br>250.5<br>Theorem-based Q<br>STEM Q&Sol<br>Tab.30<br>76<br>23,839<br>2.0<br>91.7<br>354.8<br>Theorem-based Q<br>Theorems<br>Tab.31||||



based on the provided descriptions. Such reasoning traces are typically absent from the query itself, making direct retrieval using only the query very challenging. 

- 3.2 STACKEXCHANGE: RETRIEVING WEB PAGES THAT HELP ANSWER QUESTIONS 

**Relevance** : A document is considered relevant to a query only if it is cited in an accepted or highly voted answer and unanimously confirmed by annotators and domain experts that it helps reason through the query with critical concepts or theories. 

StackExchange[3] is a popular community-driven platform where users ask questions and receive answers from other users. Among its 170+ sites, we select 7 diverse and knowledge-intensive domains: Sustainable Living, Economics, Psychology, Robotics, Earth Science, Biology, and coding in Stack Overflow. Unlike the short questions in traditional retrieval benchmarks, questions on StackExchange often contain long and technical descriptions of the problems and end with a logically complex question, such as fixing a bug. Responses often link to external web pages that contain relevant information to address the question. We construct query-document pairs based on user posts and documents referenced in the answers (see Figure 2). 

**Selecting posts.** Human annotators[4] browse posts from newest to oldest and select a post with at least one answer that (1) is accepted by the user or receives _>_ 5 votes, and (2) contains one or more URL links. This process ensures that each dataset has a sufficient number of high-quality examples. 

**Constructing query and positive documents.** For each selected post, we construct the query and positive documents as follows: 

Step 1: The annotator combines the title and content of the post to form the query _Q_ . 

- Step 2: The annotator visits web pages linked in the answers and includes them as positive documents if they are relevant to queries or discards them otherwise. 

> 3https://stackexchange.com/ 

> 4authors and college students from corresponding fields. 

4 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval_images/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Query<br>title, keywords<br>Search<br>How good is it to reuse water from plant pots? generated by LLM<br>Humans/LLMs select<br>I'm living in an apartment, and after I water my plants the water goes to  irrelevant results<br>plates below the pots The pots are in a metallic structure above the<br>plates, so I can take the plates to reuse the water (throwing it at the plants again) ... Water garden or aquatic garden, is a Wikipedia Self-watering pots are convenient to Blog Water reuse can provide alternatives Article<br>term sometimes  use and provide  to existing water<br>2 Answers used for gardens ... water supply ... supplies ...<br>In my experience plants suffer in the long term from accumulation of salts in the soil, so fresh water would be better than reusing the water. Even better would be to get hold of fresh rain water ... for watering  Soluble salts are commonly found in Report split Negative<br>them, as that won't contain the salts that tap water does.

 soils. When they  split<br>build up, they des-<br>More detail here. troy the soil struc- Humans/LLMs  Positive<br>collect linked documents ture and cause  select relevant<br>... direct damage to roots ... passages<br>**----- End of picture text -----**<br>


Figure 2: **An overview of the data annotation process for StackExchange data.** The post content is used as the query. Positive documents are selected passages from web pages linked in the answer, while the remaining passages, results from Google, and passages filtered by annotators are considered negatives. The web pages can include content from Wikipedia, blogs, articles, reports, and more. 

- Step 3: If no web page is considered positive, the post itself is discarded. For each collected web page, the annotator splits the content into passages and selects positives _DQ,i_[+][following the] relevance definition. 

**Constructing negative documents.** To prevent models from relying purely on lexical or semantic similarities, we ensure that the negative documents for each query also address similar topics. Specifically, annotators gather search results from Google on the same topic and select documents that, while topically related, do not meet the specific requirements of the query (for more examples, see Appendix C). The collection procedure is as follows: 

- Step 1: Annotators search Google using the posts’ title or LLM-summarized post keywords, and identify web pages that are semantically similar but not relevant to answering the question. 

- Step 2: Annotators collect up to 5 negative web pages for each query and split them into hard negative passages, which consist of _DQ[−]_[.] 

For each dataset, we compile all the collected passages into a unified retrieval corpus _D_ . For any given query, all passages in the corpus—excluding its positive passages—are treated as negatives, including positive and negative passages associated with other queries. In contrast to traditional retrieval tasks such as open-domain QA (Fan et al., 2019; Kwiatkowski et al., 2019), where the retrieval pool typically includes documents that directly answer the query, we simulate a realistic scenario where positive documents only provide useful information to help users derive an answer. 

The annotation process involves a single computer science student performing the initial annotations, which are then verified by two PhD students specializing in the corresponding fields. Only annotations that receive unanimous approval from all involved parties, the original annotator and two expert reviewers, are retained in the final dataset. This approval process applies to both the positive and negative document annotations, ensuring a high standard of accuracy and reliability in the annotated examples. See more guidelines to annotators in Appendix E. 

- 3.3 CODING: RETRIEVING DOCUMENTATION OR SIMILAR SOLVED PROBLEMS 

**Relevance** : The relevance between queries and positive documents is defined by whether the coding problem (i.e., query) either requires the corresponding syntax documentation or involves the same algorithm and/or data structure. 

To solve a coding problem, programmers often need to refer to the documentation or find similar problems that share the same algorithmic design. However, given only a problem description, it is difficult to find relevant documentation or similar problems via simple keyword or semantic matching. We construct two retrieval datasets on coding, where the relevance between queries and documents is grounded in the syntax usage and algorithm design. 

5 

Preprint. 

**Pony.** When working with a rare programming language, consulting the manual can be invaluable for understanding its syntax and function usage. However, in such cases, the problem description would likely have low semantic similarity and lexical overlap with the relevant documentation. This discrepancy necessitates intensive reasoning in identifying the particular syntax or function that is relevant to the problem at hand. We adopt a code generation dataset featuring Pony (Su et al., 2024), a rare programming language, and construct a retrieval dataset. We use the instructions of coding problems as queries _Q_ , the annotated documentation about the required syntax as the positive documents _D_[+] _Q_[, and the complete language manual as the retrieval pool of documents] _[ D]_[, where each] _Di_ contains descriptions about syntax usage of Pony, such as conditionals, loops, and classes. 

**LeetCode.** We also explore coding problems that deal with algorithms and data structures, where the goal is to retrieve problems and solutions that share the same algorithmic design. We source coding problems and solutions from LeetCode.[5] The problem descriptions are used as queries _Q_ , and the positive documents _DQ_[+][are solved problems (with solutions) that were annotated as similar problems] by LeetCode. Each document _Di_ = ( _Qi, Ai_ ) from LeetCode contains a problem statement _Qi_ and a Python solution _Ai_ . To increase difficulty, we only keep questions that are grounded in real-world scenarios, where arriving at the key algorithm or data structure requires intensive reasoning. We construct a large corpus _D_ by combining questions and solutions from LeetCode and Python code from CodeSearchNet (Husain et al., 2019). See Appendix B.5 for more details on the dataset. 

- 3.4 THEOREM-BASED QUESTIONS: RETRIEVE SOLVED PROBLEMS USING THE SAME THEOREMS OR RELEVANT THEOREM STATEMENTS 

**Relevance** : A query (i.e., a solved problem) is relevant to a document if the document references the same theorem used in the query. 

When tackling a new math or physics problem, users often reference previously solved problems or directly consult relevant theorem statements to guide their reasoning. Retrieving such similar problems or theorems can be challenging, as problems that share the same underlying logic may differ significantly in surface form, as shown in Table 30. In this setting, the query _Q_ is a theorembased question, and the corpus _D_ consists of solved STEM problems _Di_ = ( _Qi, Ai_ ), where _Qi_ is the problem statement and _Ai_ is its solution, or _Di_ are theorem documents from formal theorem collections such as ProofWiki. We consider _Di_ as a positive document if it shares the same theorem as the query’s solution. The dataset is built from high-quality STEM sources (Cobbe et al., 2021; Yuan et al., 2023; Hendrycks et al., 2021; Ling et al., 2017; Chen et al., 2023a; Li et al., 2023a), and for details on corpus construction, refer to Appendix B.1. 

**TheoremQA.** Derived from textbooks, online resources, and experts, TheoremQA (Chen et al., 2023b) contains questions that are based on specific mathematical or scientific theorems (e.g., the binomial theorems), and represent problems that students and other users might encounter in their studies. To ensure that the model does not simply rely on the surface-level wording of the questions, we use GPT-4[6] to rephrase the question into more concrete, applied scenarios while maintaining the same required theorem. The prompts used for rewriting the questions and an example are shown in Table 13. Human annotators carefully review the rewritten questions and make necessary revisions to ensure that they are valid and consistent with the original questions. A document _Di_ = ( _Qi, Ai_ ) is positive if _Ai_ uses the same theorem as the query’s solution. Additional details are in Appendix B.2. 

**AoPS.** Math competition problems have been widely used to evaluate the problem-solving skills of students and LLMs (Hendrycks et al., 2021). Sourced from American and International Math Olympiads, these problems often require the application of advanced mathematical theorems and techniques, such as Fermat’s Little Theorem or Ptolemy’s theorem. To practice for the competitions, students often learn by solving other problems that require the same problem-solving skills. To this end, we collect a new dataset of math competition problems, called AoPS, annotated with their respective problem-solving skills from AoPS Wiki[7] . The collected problem-solving skills are shown in Table 18. Similar to TheoremQA, we consider a solved math problem _Di_ = ( _Qi, Ai_ ) positive if its solution uses the same problem-solving skill as the query’s solution. From preliminary qualitative 

> 5https://leetcode.com/ 

> 6GPT-4 refers to the version gpt-4-0125-preview throughout this work. 

> 7https://artofproblemsolving.com/wiki/ 

6 

Preprint. 

Table 2: **The performance of retrieval models in BRIGHT.** We report nDCG@10 for all datasets: Biology (Bio.), Earth Science (Earth.), Economics (Econ.), Psychology (Psy.), Robotics (Rob.), Stack Overflow (Stack.), Sustainable Living (Sus.), LeetCode (Leet.), Pony, AoPS, TheoremQA with question retrieval (TheoQ.) and with theorem retrieval (TheoT.). Avg. denotes the average score across 12 datasets. The best score on each dataset is shown in bold and the second best is underlined. We show that reasoning-intensive retrieval is challenging for current retrievers, where the best model only achieves an nDCG@10 score of 22 _._ 1 on average. Model details are in Appendix A.1 

||.||||
|---|---|---|---|---|
||||||
||**StackExchange**<br>**Bio. Earth. Econ. Psy. Rob. Stack. Sus.**|**Coding**<br>**Leet. Pony**|**Theorem-based**<br>**Avg.**<br>**AoPS TheoQ. TheoT.**||
||_Sparse model_||||
|BM25|19.2<br>27.1<br>14.9<br>12.5 13.5<br>16.5<br>15.2|24.4<br>7.9|6.0<br>13.0<br>6.9|14.8|
||_Open-sourced models (<1B)_||||
|BGE<br>Inst-L<br>SBERT|12.0<br>24.2<br>16.6<br>17.4 12.2<br>9.5<br>13.3<br>15.6<br>21.5<br>16.0<br>21.9 11.5<br>11.2<br>13.2<br>15.5<br>20.1<br>16.6<br>22.6<br>8.4<br>9.5<br>15.3|26.7<br>5.6<br>20.0<br>1.3<br>26.4<br>6.9|6.0<br>13.0<br>6.9<br>8.1<br>20.9<br>9.1<br>5.3<br>20.0<br>10.8|13.6<br>14.2<br>14.8|
||_Open-sourced models (>1B)_||||
|E5<br>SFR<br>Inst-XL<br>GritLM<br>Qwen|18.8<br>26.0<br>15.5<br>15.8 16.4<br>9.8<br>18.5<br>19.5<br>26.6<br>17.8<br>19.0 16.7<br>12.7<br>19.8<br>21.9<br>34.4<br>**22.8**<br>27.4 **17.4**<br>19.1<br>18.8<br>25.0<br>32.8<br>19.0<br>19.9 17.3<br>11.6<br>18.0<br>**30.9**<br>**36.2**<br>17.7<br>24.6 13.5<br>**19.9**<br>14.9|28.7<br>4.8<br>27.4<br>2.0<br>27.5<br>5.0<br>29.8<br>**22.0**<br>25.5<br>14.4|7.1<br>26.1<br>26.8<br>7.4<br>24.3<br>26.0<br>8.5<br>15.6<br>5.9<br>8.8<br>25.1<br>21.1<br>**27.8**<br>**32.9**<br>**32.9**|17.9<br>18.3<br>18.7<br>20.9<br>**24.3**|
||_Proprietary models_||||
|Cohere<br>OpenAI<br>Voyage<br>Google|19.0<br>27.5<br>20.2<br>21.8 16.2<br>16.5<br>17.7<br>23.7<br>26.3<br>20.0<br>27.5<br>12.9<br>12.5<br>**20.3**<br>23.6<br>25.1<br>19.8<br>24.8 11.2<br>15.0<br>15.6<br>23.0<br>34.4<br>19.5<br>**27.9** 16.0<br>17.9<br>17.3|26.8<br>1.8<br>23.6<br>2.5<br>**30.6**<br>1.5<br>29.6<br>3.6|6.5<br>15.1<br>7.1<br>8.5<br>23.8<br>12.3<br>7.4<br>26.1<br>11.1<br>9.3<br>21.5<br>14.3|16.3<br>17.8<br>17.7<br>19.5|



analysis, we find that competition problems are deliberately written in diverse ways such that it is challenging to identify the required techniques; thus, we do not rephrase the problem statements. 

**Theorem retrieval.** Besides similar problems, retrieving relevant theorem definitions is also helpful. We use the queries from the aforementioned TheoremQA dataset with a different corpus _D_ , where each document _Di_ is a theorem statement from ProofWiki[8] . We align theorems in theoremQA with ProofWiki documents using title matching, followed by GPT-4 verification to ensure a candidate theorem is used in the solution. We retain only queries with at least one relevant theorem statement. Manual annotation of relevance between problems and documents also showed substantial agreement (Cohen’s _κ_ = 0 _._ 62) between human annotators and GPT-4. Details are in §B.3. 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETUP 

We evaluate 13 representative retrieval models, ranging from traditional bag-of-words models to large dense retrieval models, including the top performers from the retrieval set of the MTEB leaderboard (Muennighoff et al., 2023), BEIR (Thakur et al., 2021). First, we employ BM25 (Robertson et al., 2009) as our primary sparse, lexical-based retrieval model, which demonstrates strong performance on BEIR (Thakur et al., 2021), comparable to that of larger trained dense retrieval models. We also evaluate a diverse set of open-source dense retrieval models: the small (<1B) models are SentenceBERT (109M; Reimers & Gurevych, 2019), BGE (335M; Xiao et al., 2023), and Instructor-Large (335M Su et al., 2022), and the large (>1B) models are Instructor-XL (1.5B; Su et al., 2022), E5-Mistral (7.1B; Wang et al., 2023a), SFR-Embedding-Mistral (7.1B; Meng et al., 

> 8proofwiki.org; a collection of over 20K formal definitions and proofs of mathematical theorems. 

7 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval_images/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval.pdf-0008-01.png)


**----- Start of picture text -----**<br>
27 Original<br>GritLM<br>25<br>Llama3-70B<br>23 GPT-4<br>21<br>19<br>17<br>15<br>13<br>BM25 SBERT Inst-xl Qwen Google<br>Models<br>NDCG@10<br>**----- End of picture text -----**<br>


Figure 3: **Average nDCG@10 score on BRIGHT when using the original query vs. reasoning steps generated by GritLM, Llama3-70B and GPT-4 for retrieval.** Searching with LLM reasoning steps significantly improves performance. Surprisingly, BM25 achieves the best performance in the leaderboard using reasoning steps written by GPT-4 as new queries. Detailed scores are in Table 34 to 38. 

|**Retriever **|**Reranker**|**_k_**|**nDCG@10**|
|---|---|---|---|
||None|-|14.3|
||MiniLM|10|13.1|
|BM25|MiniLM<br>Gemini|100<br>10|8.3<br>15.7|
||GPT-4|10|17.4|
||GPT-4|100|17.0|
||None|-|19.5|
||MiniLM|10|16.0|
|Google|MiniLM<br>Gemini|100<br>10|9.4<br>20.1|
||GPT-4|10|21.5|
||GPT-4|100|22.6|



Table 3: **Average reranking performance on BRIGHT.** We also include the retrieval results (reranker=None) for comparison. Detailed scores can be found in Table 41 and 42. 

2024), GritLM (7.1B; Muennighoff et al., 2024), and gte-Qwen1.5 (7.7B; Li et al., 2023b). Notably, all large dense models and Instructor-Large are instruction-tuned. Lastly, we include proprietary models from Cohere (Cohere), Voyage (Voyage AI), OpenAI (OpenAI), and Google(1.2B) (Lee et al., 2024). We provide details of each model in § A.1. Following prior work (Thakur et al., 2021; Bajaj et al., 2018; Voorhees & Tice, 2000), we use nDCG@10 as the main metric. 

## 4.2 MAIN RESULTS 

**Existing retrieval systems perform poorly on BRIGHT.** Results in Table 2 show that **BRIGHT** is very challenging, with the best model achieving only 22 _._ 1 nDCG@10. Although BM25 matches the _<_ 1B models, it significantly underperforms larger models. This suggests that traditional keyword matching (“level 1 search”) is insufficient for **BRIGHT** . Although larger models that have been trained on semantic-based retrieval datasets like MS MARCO (Figure 1), such as GritLM (Muennighoff et al., 2024), perform better than BM25, they are still unable to solve **BRIGHT** . Proprietary models perform similarly to large open-source ones. Overall, the low performance indicates that the existing retrieval system cannot perform reasoning-intensive retrieval, and new methods are required to solve “level 3 search”. 

**Querying with LLM reasoning steps improves retrieval performance.** Considering the strong reasoning abilities of LLMs, we propose that leveraging LLM-generated reasoning steps as queries may enhance retrieval performance. To validate this hypothesis, we prompt LLMs to write reasoning traces given a query with the following prompt: _“(1) Identify the essential problem in the post. (2) Think step by step to reason about what should be included in the relevant documents. (3) Draft an answer.”_ . We encourage LLMs to first understand the question by summarizing it, then use chain-of-thought reasoning (Wei et al., 2022) to identify relevant content, and finally write a candidate answer. We use then use these reasoning-enhanced as new queries to evaluate all retrieval models. We use GPT-4[9] , GritLM (Muennighoff et al., 2024) and Llama-3-70B-Instruct[10] to generate reasoning steps. Figure 3 shows that using Llama-3-70B or GPT-4 reasoning steps as queries significantly improves performance compared to the original query (the detailed scores are in Tables 34 to 38). GritLM-generated reasoning steps improve BM25 performance but are less effective for other models likely due to having fewer parameters. Overall, BM25 improves the most, possibly because BM25 can adapt to different queries, while LLM-generated queries are out-of-distribution for trained models. With the best score still being below 30, significant room remains for improvement on **BRIGHT** . 

> 9gpt-4-0125-preview 

> 10https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct 

8 

Preprint. 

Table 4: **Question-answering results with different retrievers.** We use Claude-3.5-sonnet as the generation model and evaluate the answers with Claude-3.5-sonnet. We find that stronger retrieval typically results in better QA results, indicating the helpfulness of the annotated documents for addressing the posts in StackExchange. Performance with GPT-4 in Table 47 shows similar trends. 

|**Retriever**|**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Average**|
|---|---|---|
|None|61.3<br>63.8<br>56.9<br>56.1<br>51.7<br>64.1<br>50.0|57.7|
|BM25<br>SBERT<br>Qwen|63.4<br>64.7<br>56.2<br>53.2<br>59.3<br>65.7<br>56.3<br>60.8<br>61.2<br>59.0<br>64.4<br>54.5<br>59.6<br>**60.8**<br>**68.2**<br>**67.3**<br>**60.8**<br>**65.6**<br>**63.4**<br>**68.9**<br>55.7|59.8<br>60.0<br>**64.3**|
|Oracle|_78.8_<br>_79.4_<br>_71.7_<br>_76.1_<br>_69.7_<br>_79.0_<br>_69.7_|_74.9_|



**Retrieval augmentation boosts performance in question-answering.** An important application of retrieval is to ultimately improve the question-answering (QA) results. In addressing a StackExchange post, one typical pipeline is to first retrieve useful documents, and then answer questions leveraging the retrieved content. While the second step could involve reasoning processes to derive an answer, we emphasize intensive reasoning in the first step to find documents. We evaluate the end-to-end QA performance of Claude-3.5-sonnet when augmented with documents retrieved by different models, and use Claude-3.5-sonnet to evaluate the answer correctness. As shown in Table 4, stronger retrievers generally result in better QA performance, with the top-performing retriever, Qwen, achieving a 6.6-point gain. However, using the oracle documents boosts performance by 17.2 points, highlighting substantial room for improvement in reasoning-intensive retrieval to enhance downstream QA results. See §D for similar experiments on the TheoremQA and AoPS datasets. 

Although better retrieval quality correlates with higher QA performance, the QA results may not always accurately capture retrieval performance due to the following reasons: 1) the generator model may not effectively comprehend or integrate the retrieved documents into its responses, and 2) the evaluator may fail to fully recognize the similarities and differences between two open-ended answers. Please check out Table 45 and 46 for inference and evaluation prompts used for LLMs. 

## 5 ANALYSIS 

## 5.1 RERANKING WITH LLMS ENHANCES RETRIEVAL PERFORMANCE 

A common approach for improving retrieval results is to utilize powerful rerankers capable of performing joint computation over both the query and the documents. To this end, we investigate if performance on **BRIGHT** can be improved through reranking. We test this with a classical cross-encoder, MiniLM[11] , and LLMs to rerank the top _k_ = _{_ 10 _,_ 100 _}_ retrieved documents. The cross-encoder is trained on the MS MARCO reranking task (Bajaj et al., 2016) and outputs a relevance score for each pair of query and document ( _Q, Di_ ). Following Sun et al. (2023), we also rerank with LLMs by including the query and top-k documents in the prompt and asking the LLMs to order the documents based on their relevance to the query (detailed prompts can be found in Table 43). Table 3 shows that the traditional cross-encoder negatively impacts retrieval quality, with performance declining as more documents are reranked, suggesting that training rerankers on MS MARCO does not transfer well to **BRIGHT** . On the other hand, reranking by LLMs generally enhances performance. Stronger LLMs provide more significant improvements; for instance, based on BM25 retrieval results, Gemini-1.0 reranking increases the score by 1.4, and GPT-4 reranking enhances by 3.1 and continues to improve with higher _k_ . LLMs can serve as an effective tool for reasoning-intensive retrieval, but the final results still highly depend on the underlying retrieval system. 

## 5.2 ROBUSTNESS AGAINST DATA LEAKAGE FROM PRETRAINING 

Many existing benchmarks lead to overestimated performance improvements due to large-scale pretraining that accidentally includes benchmark data (Zhou et al., 2023; Xu et al., 2024). In this subsection, we demonstrate that **BRIGHT** is robust to such leakage, even when the retrieving 

> 11https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2 

9 

Preprint. 

documents are fully seen during pretraining. We simulate a scenario where language models are trained on data crawled from the internet, which may include StackExchange data. Specifically, we continue training GritLM (Muennighoff et al., 2024) on the data in the StackExchange retrieval pool from **BRIGHT** using language modeling loss. To maintain the retrieval ability of GritLM, we also train it with a contrastive learning objective on StackExchange question and answer pairs (more training details in Appendix A.3). This exposes all the StackExchange data from **BRIGHT** to the model, but omits direct training on the mappings between queries and documents, which require intensive reasoning to resolve and do not naturally occur during pretraining. Table 5 shows a slight decrease in the average results of fine-tuned GritLM, indicating that the current data formats and training procedures may not significantly impact performance in **BRIGHT** . This indicates that **BRIGHT** is robust to data leakage from large-scale pretraining and calls for novel approaches to improve reasoning-intensive retrieval. 

Table 5: **BRIGHT is robust to massive pre-training.** By continuing training GritLM on StackExchange data without showing the mapping between queries and documents, the model does not improve the average performance after learning the in-domain knowledge, indicating the importance of reasoning capabilities in the retrieval process. 

||**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Avg.**|
|---|---|---|
|GritLM<br>Fine-tuned GritLM|25.0<br>32.8<br>19.0<br>19.9<br>17.3<br>11.6<br>18.0<br>21.1<br>25.5<br>18.8<br>30.7<br>12.7<br>12.1<br>21.9|20.5<br>20.4|



## 5.3 LONG-CONTEXT RETRIEVAL WITH A REDUCED SEARCH SPACE IS CHALLENGING 

Retrieving information from long documents is crucial for applications such as legal contracts, company financial documents, and patient notes (Saad-Falcon et al., 2024; Zhu et al., 2024). To evaluate retrieval models on reasoning-intensive tasks involving lengthy documents, we convert the StackExchange datasets to a long-context retrieval setting, where documents are complete web pages with significantly more tokens but fewer total number of documents (Table 40). With most datasets containing only a few hundred documents, nDCG@10, which evaluates the top 10 results, becomes more susceptible to randomness. Moreover, processing 10 long documents with an average length of up to 40,000 tokens is challenging for both humans and LLMs. Therefore, we decide to use recall@1 metric to provide a more reliable measure in this setting. Table 6 presents the average scores for 8 datasets from StackExchange and Pony. The highest recall achieved is 27 _._ 8, indicating that even with significantly reduced retrieval pools, the combination of long-context documents and intensive reasoning remains challenging for existing retrieval models. 

Table 6: **Long-context retrieval performance where retrievers retrieve from unsplit web pages.** The results are reported as the average recall@1 score of StackExchange and Pony datasets. More detailed numbers can be found in Table 39. 

|**BM25**|**BGE**|**Inst-L**|**SBERT**|**E5**|**SFR**|**Inst-XL**|**GritLM**|**Qwen**|**Cohere**|**OpenAI**|**Voyage**|**Google**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|11.4|14.8|18.2|17.4|25.5|26.0|17.8|26.0|27.8|18.4|21.9|24.6|22.4|



## 6 CONCLUSION 

We introduce **BRIGHT** , the first retrieval benchmark that encompasses realistic retrieval scenarios requiring intensive reasoning steps to identify relevant documents. We utilize existing online document structures and dedicate substantial human effort to curate **BRIGHT** and verify its correctness. Through extensive evaluation, we find that existing retrieval models perform extremely poorly on **BRIGHT** , with a maximum nDCG@10 score of only 22 _._ 1. Augmenting retrieval queries with reasoning steps generated by LLMs improves performance, but even the best model still achieves a score below 30. Furthermore, strong retrieval results can significantly improve downstream performance on reasoning tasks, highlighting a practical application of reasoning-intensive retrieval. We hope that **BRIGHT** will contribute to future research investigations into pushing the state-of-the-art in this direction. 

10 

Preprint. 

## 7 CODE OF ETHICS AND ETHICS STATEMENT 

In the process of collecting datasets for BRIGHT, we ensure that all sources come from public data, used solely for academic research and not for commercial purposes, in full compliance with the copyright rights granted by the sources. We guarantee that none of the datasets contain harmful information to society, such as racial discrimination, violence, or any private data. Our work aims to contribute to the welfare of society and humanity, and any researcher is free to use our dataset for research purposes. All the data and experiments presented in our paper adhere to the highest standards of scientific excellence, ensuring the authenticity and accuracy of the data. 

## 8 REPRODUCIBILITY 

Our datasets and annotation process are introduced in Sec. 3, and the experimental settings are described in Sec. 4. Specific implementation details can be found in App. A. To facilitate the reproduction of our experiments, the code and data are provided in https://drive.google. com/file/d/1reppp1sMjl7pfV-nrivpvb4tJg4pjOsm/view?usp=sharing. 

## REFERENCES 

- Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hanna Hajishirzi, and Wen tau Yih. Reliable, adaptable, and attributable language models with retrieval. 2024. url https://api. semanticscholar. org/corpusid: 268248911. oprea, and colin raffel. extracting training data from large language models. In _USENIX Security Symposium (USENIX)_ , 2020. 

- Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. Task-aware retrieval with instructions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 3650–3675, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.225. URL https://aclanthology.org/ 2023.findings-acl.225. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2023b. 

- Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and Wen tau Yih. Reliable, adaptable, and attributable language models with retrieval, 2024. 

- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. Ms marco: A human generated machine reading comprehension dataset. _arXiv preprint arXiv:1611.09268_ , 2016. 

- Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. Ms marco: A human generated machine reading comprehension dataset, 2018. 

- Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. Improving language models by retrieving from trillions of tokens. In _International Conference on Machine Learning (ICML)_ , volume 162, pp. 2206–2240, 2022. URL https://proceedings.mlr. press/v162/borgeaud22a.html. 

11 

Preprint. 

- Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading Wikipedia to answer open-domain questions. In Regina Barzilay and Min-Yen Kan (eds.), _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 1870–1879, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1171. URL https://aclanthology.org/P17-1171. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. TheoremQA: A theorem-driven question answering dataset. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 7889–7901, Singapore, December 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.489. URL https://aclanthology.org/ 2023.emnlp-main.489. 

- Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset. In _The 2023 Conference on Empirical Methods in Natural Language Processing_ , 2023b. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. 

- Cohere. Cohere-embed-english-light-v3.0, 2022. URL https://huggingface.co/Cohere/ Cohere-embed-english-light-v3.0. 

- Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. Overview of the trec 2019 deep learning track, 2020. 

- Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International Conference on Learning Representations (ICLR)_ , 2024. 

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bulian, Massimiliano Ciaramita, and Markus Leippold. Climate-fever: A dataset for verification of real-world climate claims. _arXiv preprint arXiv:2012.00614_ , 2020. 

- Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5: Long form question answering. _arXiv preprint arXiv:1907.09190_ , 2019. 

- Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In _Empirical Methods in Natural Language Processing (EMNLP)_ , 2023. 

- Faegheh Hasibi, Fedor Nikolaev, Chenyan Xiong, Krisztian Balog, Svein Erik Bratsberg, Alexander Kotov, and Jamie Callan. Dbpedia-entity v2: a test collection for entity search. In _Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pp. 1265–1268, 2017. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe. 

- Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. CodeSearchNet challenge: Evaluating the state of semantic code search. _arXiv preprint arXiv:1909.09436_ , 2019. 

12 

Preprint. 

- Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. _Transactions on Machine Learning Research_ , 2022. ISSN 2835-8856. URL https: //openreview.net/forum?id=jKN1pXi7b0. 

- Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. _arXiv preprint arXiv:2403.07974_ , 2024. 

- Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. _arXiv preprint arXiv:2004.04906_ , 2020. 

- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:453–466, 2019. 

- Jinhyuk Lee, Zhuyun Dai, Xiaoqi Ren, Blair Chen, Daniel Cer, Jeremy R Cole, Kai Hui, Michael Boratko, Rajvi Kapadia, Wen Ding, et al. Gecko: Versatile text embeddings distilled from large language models. _arXiv preprint arXiv:2403.20327_ , 2024. 

- Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In _Association for Computational Linguistics (ACL)_ , pp. 6086–6096, 2019. doi: 10.18653/v1/P19-1612. URL https://aclanthology.org/P19-1612. 

- Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society. In _Thirtyseventh Conference on Neural Information Processing Systems_ , 2023a. 

- Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. _arXiv preprint arXiv:2308.03281_ , 2023b. 

- Zhuoqun Li, Hongyu Lin, Tianshu Wang, Boxi Cao, Yaojie Lu, Weixiang Zhou, Hao Wang, Zhenyu Zeng, Le Sun, and Xianpei Han. Url: Universal referential knowledge linking via task-instructed representation compression. _arXiv preprint arXiv:2404.16248_ , 2024. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Regina Barzilay and Min-Yen Kan (eds.), _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015. 

- Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. Www’18 open challenge: financial opinion mining and question answering. In _Companion proceedings of the the web conference 2018_ , pp. 1941–1942, 2018. 

- Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. Sfrembedding-mistral: enhance text retrieval with transfer learning. _Salesforce AI Research Blog_ , 3, 2024. 

- Niklas Muennighoff. Sgpt: Gpt sentence embeddings for semantic search. _arXiv preprint arXiv:2202.08904_ , 2022. 

- Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. MTEB: Massive text embedding benchmark. In Andreas Vlachos and Isabelle Augenstein (eds.), _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pp. 2014– 2037, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/ v1/2023.eacl-main.148. URL https://aclanthology.org/2023.eacl-main.148. 

13 

Preprint. 

- Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. Generative representational instruction tuning. _arXiv preprint arXiv:2402.09906_ , 2024. 

- Hanseok Oh, Hyunji Lee, Seonghyeon Ye, Haebin Shin, Hansol Jang, Changwook Jun, and Minjoon Seo. Instructir: A benchmark for instruction following of information retrieval models, 2024. 

OpenAI. New embedding models and api updates, 2024. URL https://openai.com/index/ new-embedding-models-and-api-updates/. 

- Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. KILT: a benchmark for knowledge intensive language tasks. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2523–2544, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.200. URL https://aclanthology.org/ 2021.naacl-main.200. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. _arXiv preprint arXiv:1908.10084_ , 2019. 

- Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. _Foundations and Trends® in Information Retrieval_ , 3(4):333–389, 2009. 

- Jon Saad-Falcon, Daniel Y Fu, Simran Arora, Neel Guha, and Christopher Ré. Benchmarking and building long-context retrieval models with loco and m2-bert. _arXiv preprint arXiv:2402.07440_ , 2024. 

- Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. REPLUG: Retrieval-augmented black-box language models, 2024. 

- Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instruction-finetuned text embeddings. _arXiv preprint arXiv:2212.09741_ , 2022. 

- Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instruction-finetuned text embeddings. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 1102–1121, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.71. URL https://aclanthology.org/2023.findings-acl.71. 

- Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao Shi, Che Liu, Qian Liu, and Tao Yu. Arks: Active retrieval in knowledge soup for code generation. _arXiv preprint arXiv:2402.12317_ , 2024. 

- Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. Is chatgpt good at search? investigating large language models as re-ranking agent. _arXiv preprint arXiv:2304.09542_ , 2023. 

- Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. _arXiv preprint arXiv:2104.08663_ , 2021. 

- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for fact extraction and verification. _arXiv preprint arXiv:1803.05355_ , 2018. 

14 

Preprint. 

- Ellen M. Voorhees and Dawn M. Tice. Building a question answering test collection. In _Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval_ , SIGIR ’00, pp. 200–207, New York, NY, USA, 2000. Association for Computing Machinery. ISBN 1581132263. doi: 10.1145/345508.345577. URL https://doi. org/10.1145/345508.345577. 

Voyage AI. Embeddings, 2024. URL https://docs.voyageai.com/docs/embeddings. 

- David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. _arXiv preprint arXiv:2004.14974_ , 2020. 

- Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. _arXiv preprint arXiv:2401.00368_ , 2023a. 

- Xiaoyue Wang, Jianyou Wang, Weili Cao, Kaicheng Wang, Ramamohan Paturi, and Leon Bergen. Birco: A benchmark of information retrieval tasks with complex objectives. _arXiv preprint arXiv:2402.14151_ , 2024. 

- Zengzhi Wang, Rui Xia, and Pengfei Liu. Generative ai for math: Part i – mathpile: A billion-tokenscale pretraining corpus for math, 2023b. URL https://arxiv.org/abs/2312.17120. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, and Luca Soldaini. Followir: Evaluating and teaching information retrieval models to follow instructions. _arXiv preprint arXiv:2403.15246_ , 2024. 

- Chenghao Xiao, G Thomas Hudson, and Noura Al Moubayed. Rar-b: Reasoning as retrieval benchmark, 2024. 

- Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to advance general chinese embedding, 2023. 

- Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. _arXiv preprint arXiv:2404.18824_ , 2024. 

- Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models, 2023. 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. _arXiv preprint arXiv:1809.09600_ , 2018. 

- Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023. 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning, 2023. 

- Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, et al. A careful examination of large language model performance on grade school arithmetic. _arXiv preprint arXiv:2405.00332_ , 2024. 

- Kun Zhou, Yutao Zhu, Zhipeng Chen, Wentong Chen, Wayne Xin Zhao, Xu Chen, Yankai Lin, Ji-Rong Wen, and Jiawei Han. Don’t make your llm an evaluation benchmark cheater. _arXiv preprint arXiv:2311.01964_ , 2023. 

- Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. Longembed: Extending embedding models for long context retrieval, 2024. 

15 

Preprint. 

## A EXPERIMENT DETAILS 

## A.1 MODELS AND INSTRUCTIONS 

For each model used in this paper, Table 7 provides information on the size, architecture, maximum context length of queries and documents, whether we include instructions and the specific version we use in the experiments. All parameters are set by following the official tutorial. The only exceptions are Inst-L and Inst-XL, where we empirically find that extending the maximum context length to 2048 significantly enhances the performance. In Table 8, 9, 10 and 11, We specify the instructions used for BGE, Inst-L, Inst-XL, E5, GritLM, Qwen and SFR in each dataset. For the embedding model from Google, we use the parameter "task" with the values "RETRIEVAL_QUERY" and "RETRIEVAL_DOCUMENT" to distinguish queries from documents and use the parameter "input_type" with the values "query" and "document" for the embedding model from Voyage. 

Table 7: **All 13 models benchmarked in experiments.** We report the number of parameters of each model except the sparse model BM25 and proprietary models without public information. Regarding the model architecture, we distinguish between sparse and dense models and further classify dense models as encoders or decoders if known. Max _|Q|_ and Max _|D|_ denote the maximum context length we use for each model in the experiments. The instruction column indicates whether we include instructions in the retrieval. The version column denotes the specific checkpoint or implementation. 

||**Size**<br>**Architecture**<br>**Max**_|Q|_<br>**Max**_|D|_<br>**Instruction**<br>**Version**<br>**License**|
|---|---|
|Sparse model||
|BM25 (Robertson et al.,2009)|N/A<br>Sparse<br>_∞_<br>_∞_<br>No<br>gensim12<br>LGPL-2.1-only|
|_Open-sourced models (<1B)_||
|SBERT (Reimers & Gurevych,2019)<br>BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)|109M<br>Encoder<br>512<br>512<br>No<br>all-mpnet-base-v2<br>Apache-2.0<br>335M<br>Encoder<br>512<br>512<br>No<br>bge-large-en-v1.5<br>MIT<br>335M<br>Encoder<br>2048<br>2048<br>Yes<br>instructor-large<br>Apache-2.0|
|_Open-sourced models (>1B)_||
|Inst-XL (Su et al.,2022)<br>E5 (Wang et al.,2023a)<br>GritLM (Muennighoff et al.,2024)<br>SFR (Meng et al.,2024)<br>Qwen (Li et al.,2023b)|1.5B<br>Encoder<br>2048<br>2048<br>Yes<br>instructor-xl<br>Apache-2.0<br>7.1B<br>Decoder<br>4096<br>4096<br>Yes<br>e5-mistral-7b-instruct<br>MIT<br>7.1B<br>Decoder<br>256<br>2048<br>Yes<br>GritLM-7B<br>Apache-2.0<br>7.1B<br>Decoder<br>4096<br>4096<br>Yes<br>SFR-Embedding-Mistral<br>CC-BY-NC-4.0<br>7.7B<br>Decoder<br>8192<br>8192<br>Yes<br>gte-Qwen1.5-7B-instruct<br>Apache-2.0|
|_Proprietary models_||
|Cohere (Cohere)<br>Google (Lee et al.,2024)<br>OpenAI (OpenAI)<br>Voyage (Voyage AI)|N/A<br>Dense<br>512<br>512<br>No<br>Cohere-embed-english-v3.0<br>Company<br>1.2B<br>Dense<br>2000<br>2000<br>Yes<br>text-embedding-preview-<br>0409, dimension=768<br>Company<br>N/A<br>Dense<br>8191<br>8191<br>No<br>text-embedding-3-large<br>Company<br>N/A<br>Dense<br>16000<br>16000<br>Yes<br>voyage-large-2-instruct<br>Company|



Table 8: **Instructions used for benchmarking StackExchange datasets.** {domain} is one of Biology, Earth Science, Economics, Psychology, Robotics, Stack Overflow and Sustainable Living. 

|Models|Instructions|
|---|---|
|BGE<br>**Query:** Represent this {domain} post for searching relevant passages:||
|||
|Inst-L,<br>Inst-XL|**Query:** Represent the {domain} post for retrieving relevant paragraphs:<br>**Doc:** Represent the {domain} paragraph for retrieval:|
|||
|E5, GritLM,<br>Qwen, SFR|**Query:** Given a {domain} post, retrieve relevant passages that help answer the post|



## A.2 MACHINES 

We run all experiments on NVIDIA V100, A100, or H100 GPUs. The amount of time that it takes to complete one round of experiments is dependent on the model. For the sparse model, BM25, the evaluation takes less than 1 hour on CPU-only machines. For the open-sourced dense models ( _<_ 1 _B_ ), the evaluation requires about 8 hours on one H100 GPU. For the open-sourced dense models ( _>_ 1 _B_ ), the evaluation takes up to 36 hours on one H100 GPU. We leverage FlashAttention (Dao et al., 2022; Dao, 2024) for speedup when evaluating the dense models. For the proprietary models, 

16 

Preprint. 

Table 9: **Instructions used for benchmarking the LeetCode dataset.** 

|Models|Instructions|
|---|---|
|||
|BGE|**Query:** Represent this Coding problem for searching relevant examples:|
|||
|Inst-L,<br>Inst-XL|**Query:** Represent the Coding problem for retrieving relevant examples:<br>**Doc:** Represent the Coding example for retrieval:|
|||
|E5, GritLM,<br>Qwen, SFR|**Query:** Given a Coding problem, retrieve relevant examples that help answer the<br>problem|



Table 10: **Instructions used for benchmarking the Pony dataset.** 

|Models|Instructions|
|---|---|
|||
|BGE|**Query:** Represent this Pony question for searching relevant passages:|
|||
|Inst-L,<br>Inst-XL|**Query:** Represent the Pony question for retrieving relevant paragraphs:<br>**Doc:** Represent the Pony paragraph for retrieval:|
|||
|E5, GritLM,<br>Qwen, SFR|**Query:** Given a Pony question, retrieve relevant passages that help answer the<br>question|



Table 11: **Instructions used for benchmarking Math datasets (AoPS and TheoremQA).** 

|Models|Instructions|
|---|---|
|||
|BGE|**Query:** Represent this Math problem for searching relevant examples:|
|||
|Inst-L,<br>Inst-XL|**Query:** Represent the Math problem for retrieving relevant examples:<br>**Doc:** Represent the Math example for retrieval:|
|||
|E5, GritLM,<br>Qwen, SFR|**Query:** Given a Math problem, retrieve relevant examples that help answer the<br>problem|



the evaluation speed is dependent on the API bandwidth, but we found that one round of experiments can be completed within 2 days. 

## A.3 CONTINUAL TRAINING 

In Section 5.2, we introduce the continual training method GritLM on StackExchange data to evaluate whether training on in-domain data enhances the performance of **BRIGHT** . Detailed experimental settings are described in this section. Specifically, we follow GritLM to train models with two distinct objectives: a contrastive loss to maintain the model’s retrieval capability and a language modeling loss to preserve the model’s language generation ability. For training with the contrastive loss, we collect 3,200 (post, answer) pairs from the Biology, Earth Science, Economics, Psychology, Robotics, and Stack Overflow sections of StackExchange, and 1,538 pairs from Sustainable Living. Each post’s answer is used as a positive example, with other answers serving as in-batch negatives. For training with the language modeling loss, we use both positive and negative documents from each domain within the StackExchange subsection of **BRIGHT** . These documents are split into chunks of 2048 tokens, and we sample up to 3,200 chunks for training. We use a small batch size of 64 to ensure sufficient learning steps, while following the other hyperparameters as outlined in Muennighoff et al. (2024). We continue training GritLM for 10 epochs, benchmarking the checkpoint from each epoch on the StackExchange datasets of **BRIGHT** . The detailed scores are in Table 12, where we copy the scores of GritLM to epoch=0 for easier reference. The results indicate no significant improvement across the 10 epochs, suggesting that even with intensive inclusion of StackExchange data or relevant domain knowledge in the training data of language models or retrievers, performance may not increase substantially without enhancing incorporating reasoning into the retrieval process. 

> 12Specifically, we use the LuceneBM25Model from gensim and the text analyzer from pyserini 

17 

Preprint. 

Table 12: **Scores of finetuned GritLM of every epoch on StackExchange datasets of BRIGHT.** Epoch=0 indicates the performance of GritLM without further training. 

|**Epoch**|**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Avg.**|
|---|---|---|
||||
|0 (GritLM)<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10|25.0<br>32.8<br>19.0<br>19.9<br>17.3<br>11.6<br>18.0<br>22.2<br>25.4<br>17.6<br>28.1<br>11.1<br>9.8<br>19.6<br>18.7<br>23.8<br>13.5<br>19.3<br>10.7<br>10.2<br>16.5<br>20.9<br>23.6<br>16.9<br>25.2<br>11.1<br>8.5<br>16.6<br>24.3<br>28.0<br>18.3<br>26.9<br>13.4<br>13.3<br>20.0<br>23.1<br>28.5<br>18.4<br>26.1<br>14.6<br>11.7<br>21.6<br>19.9<br>26.4<br>16.0<br>27.9<br>9.6<br>9.3<br>19.3<br>24.3<br>25.4<br>16.5<br>28.1<br>11.0<br>9.8<br>17.0<br>21.6<br>28.7<br>19.2<br>28.7<br>11.1<br>11.8<br>22.4<br>21.3<br>29.0<br>20.0<br>28.7<br>11.4<br>14.3<br>22.0<br>21.1<br>25.5<br>18.8<br>30.7<br>12.7<br>12.1<br>21.9|20.5<br>19.1<br>16.1<br>17.5<br>20.6<br>20.6<br>18.3<br>18.9<br>20.5<br>21.0<br>20.4|



## B DATASET CONSTRUCTION 

## B.1 STEM QUESTION AND SOLUTION CORPUS FOR THEOREMQA AND AOPS 

In this subsection, we describe the construction of the STEM question and solution corpus, which is used for both TheoremQA Questions and AoPS. We source the documents (pairs of problem statements and solutions) _Di_ = ( _Qi, Ai_ ) from existing datasets—GSM8K (Cobbe et al., 2021), GSM8K-RFT (Yuan et al., 2023), MATH (Hendrycks et al., 2021), AQuA-RAT (Ling et al., 2017), TheoremQA (Chen et al., 2023a), and CAMEL-Math (Li et al., 2023a). To reduce the likelihood of false negatives among the STEM corpus, we leverage the metadata from the original datasets to exclude specific documents from the corpus for each test query. For example, CAMEL-Math contains problem-solution pairs labeled with the category “Calculus”, which covers different questions that involve derivatives and integrals. Therefore, for queries in TheoremQA that uses “derivative chain rule” or “integral rules”, we excluded CAMEL-Math pairs in the category “Calculus” to reduce possible false negatives. Thus, for each test query in TheoremQA-Q and AoPS, we manually decide which labels in the other datasets to exclude based on the metadata. We do not exclude any problemsolution pairs from GS8K, GSM8K-RFT, or AQuA-RAT due to the relative elementary difficulty (mostly basic algebra questions) in comparison to our test queries, which leverages more advanced theorems and techniques. The mapping from the test query category to the excluded problem-solution categories can be found in Table 14, 15, and 16. 

An alternative approach to excluding false negatives from the corpus for each test query is to inspect every problem-solution pair and annotate if they are relevant to test query. Although this would yield harder negatives and additional positives, we opt to not to use this approach due to its expensive cost to conduct annotation between every test query and possible candidates. 

## B.2 THEOREMQA: REPHRASING QUESTIONS INTO SPECIFIC SCENARIOS 

TheoremQA is a dataset consisting of theorem-driven questions in mathematics, physics, finance, and computer science and electrical engineering (Chen et al., 2023a). For each question in TheoremQA, we refer it to MathInstruct dataset[13] (Yue et al., 2023), as each question in this dataset is annotated with the reasoning steps and final answers. 

From preliminary analysis, we found that TheoremQA questions are often written in a way such that the theorem used to solve the problem is explicitly mentioned in the question. As a result, questions that use the same theorem can have high keyword overlap, which means retrievers can easily retrieve the correct document by matching the keywords. Thus, we rewrite the questions in TheoremQA by grounding them in real-world scenarios or applications, which makes the reasoning steps less explicit and provides more diverse questions. We leverage GPT-4 with manually written instructions and in-context demonstrations to rewrite the queries _Q_ . We provide the prompt used to rewrite 

> 13https://huggingface.co/datasets/TIGER-Lab/MathInstruct 

18 

Preprint. 

TheoremQA questions and an example in Table 13. After rewriting the question, the authors manually inspect each rewritten question to ensure that the question is solvable and consistent with the original question (i.e., the reasoning steps and final answer still hold). When applicable, we manually edit the rewritten question to improve the fluency and coherence of the question, and discard the query if the rewritten question is not solvable or consistent with the original question. Consequently, we obtain 206 rewritten questions in TheoremQA from the original set of 800 questions. 

## B.3 THEOREMQA: ANNOTATING RELEVANT THEOREMS 

For TheoremQA-Theorems, we use the test queries from TheoremQA-Questions and annotate them with useful theorem proofs and definitions. We source mathematical theorem proofs and definitions from ProofWiki, which is community-driven effort with more than 20K formal definitions and proofs of mathematical theorems. ProofWiki is preprocessed and provided by MathPile (Wang et al., 2023b). We opt to map the original theorem names to the documents in ProofWiki so that the gold documents have consistent forms with other documents in the corpus. 

For each test query, we first construct a candidate set of useful documents from ProofWiki using the theorem name and definition provided by the original TheoremQA dataset. Specifically, we construct the candidate set with the following steps: 

1. Find documents where the theorem name exists as a substring. We discard this set if there are more than 100 such documents, which typically means that the theorem name is too common. 

2. Using the theorem name and definition from the original dataset as the query, we use BM25 to retrieve the top _k_ = 10 documents from ProofWiki. 

Then, we prompt GPT-4 (gpt-4-0125-preview) to check if each document’s described theorem are used in the problem solutions, which labels each candidate as either a positive or negative document. The prompt for this step can be found in Table 17. The authors manually annotated 50 instances and found a substantial agreement of Cohen’s _κ_ = 0 _._ 62 with the model judgments. Finally, we keep test queries with at least one positive document. 

## B.4 AOPS: CONNECTING AOPS PROBLEMS TO THE MATH DATASET 

AoPS Wiki is a community-driven platform where users can post problems and solutions to math competition problems. These math competitions include, but are not limited to the American Mathematics Competitions (AMC), the American Invitational Mathematics Examination (AIME), and the International Mathematical Olympiad (IMO). In addition to problems, the AoPS Wiki also contains articles on various topics in mathematics, such as Fermat’s Little Theorem and Ball and Urns. These articles not only describe the theorem or technique but also link to problems that can be solved by them. We browse the AoPS Wiki and collect the topics and the linked problems. The topics are listed in Table 18. 

Although math competition problems are used in previous datasets, such as MATH (Hendrycks et al., 2021), they lack the necessary annotations on the problem-solving skills to construct positive documents. Thus, we opt to collect these annotations from AoPS Wiki instead. 

Furthermore, since MATH examples are used in the STEM corpus, we deduplicate them by matching the collected problems with the MATH instances. Specifically, for each question _Q_ we collected from AoPS, we find the closest problem statement in MATH using n-gram overlap, and manually check if they are the same problem. If the same problems are found, we merge them into one instance, otherwise, we create a new instance and insert it into the corpus. 

## B.5 LEETCODE 

We first obtain the publicly available LeetCode[14] questions from HuggingFace[15] . Our retrieval pool is sourced from a combination of LeetCode and CodeSearchNet, including a problem description 

> 14https://leetcode.com/ 

> 15https://huggingface.co/datasets/greengerong/leetcode 

19 

Preprint. 

and a solution in each example. In the following sections, we outline the process for constructing positive examples and performing additional checks to minimize the likelihood of false negatives while ensuring the use of a large retrieval pool. 

**Using similar questions as positive examples.** For each question, we obtain the gold pair annotations from the “Similar Questions” field, which contains links to other LeetCode questions that are similar to the problem. While the website does not explicitly describe the guidelines behind how this field is populated, our qualitative analysis showed that these questions have a high overlap in terms of the data structure, algorithms, and/or logical reasoning used to solve the problem. 

## **Select problems based on real-world scenarios to avoid false negatives from CodeSearchNet.** 

From a preliminary qualitative analysis, we discovered that some questions in LeetCode are frequently found in CodeSearchNet due to the popularity of certain algorithms and the simplicity of their problem statements. Examples include implementing sorting algorithms or merging two linked lists, which could unexpectedly introduce false positives into this retrieval setup. Thus, we use an additional filtering step—we only keep questions that are grounded in real-world concepts that are not as commonly used in the context of coding problems. The intuition behind this is similar to the TheoremQA annotations process: the reasoning steps (i.e., the algorithms and data structures used to solve the problem) cannot be as easily deciphered as if the problem statement clearly describes the algorithm and data structure used. To this end, we first manually write instructions with six-shot incontext learning demonstrations, and use GPT-4 (gpt-4-0125-preview) to classify all LeetCode questions. Then, we validate the GPT-4 judgment with 80 annotations by the authors. Authors’ and GPT-4’s annotations have a Cohen’s kappa of 0 _._ 73, which suggests substantial agreement. The prompt used in this step and examples can be found in Table 19. GPT-4 judged a total of 291 samples of being grounded in real-world concepts, and we randomly sample 142 questions from this set to construct our test set. 

**Remove examples with high topic overlap from the pool to avoid false negatives in LeetCode.** To avoid potential false negatives that are not annotated by the LeetCode website, we leverage the “Topics” field from the website, which contains information about the algorithms and data structure used in the problem, such as “stack” and “breath first search”. For each question _Q_ , we collect its topics _T_ ( _Q_ ) = _{t_[1] _q[, . . . , t][m] q[}]_[ from the LeetCode website, where] _[ t][q]_[denotes the] _[ m][ ≥]_[1][ different topics] assigned to the question by the website. Since each question may have multiple tags, we exclude other questions that have a high overlap with test questions from the corpus for that specific question.[(] _[Q]_[)] _[∩][T]_[(] _[Q][′]_[)] _[|]_ Specifically, we exclude question _Q[′]_ from the corpus for test query _Q_ if _[|][T] |T_ ( _Q_ ) _| ≥_ 0 _._ 5, because more than half of the topics used in _Q[′]_ are also used in _Q_ . This means that the two questions can be highly related in reasoning steps. Overlap smaller than the 0 _._ 5 threshold means that _Q[′]_ is unlikely to be related, and thus a false negative, to the test question _Q_ . Finally, we construct the rest of the corpus from CodeSearchNet (Husain et al., 2019)[16] . We only consider the Python functions as the solutions to the LeetCode questions are all in Python. 

## C DATA EXAMPLES 

In Table 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, and 31, we show more examples in **BRIGHT** . 

## D DOWNSTREAM PERFORMANCE 

In this work, we evaluated the effect of retrieval on downstream tasks. In this subsection, we continue the investigation into the downstream performance on the theorem-based tasks—TheoremQA Questions, TheoremQA Theorems, and AoPS—by evaluating gpt-4o-2024-08-06 on these tasks. We run with with three types of documents in context: none (closed-book), random (randomly sampled document from the corpus), and oracle (the annotated corpus). We use a temperature of 0.2 and top-p of 0.9, and average the results across 5 different random seeds. We show the results in Table 48, and find that the oracle document can consistently improve the performance across all datasets. This improvement is statistically significant in TheoremQA Questions and TheoremQA Theorems, suggesting the benefits of using strong retrieval results. 

> 16https://huggingface.co/datasets/code_search_net 

20 

Preprint. 

## E ANNOTATOR INSTRUCTIONS 

In this section, we describe the instructions for annotators to collect data in **BRIGHT** . 

## E.1 STACKEXCHANGE 

1. Browse posts from the newest to the oldest. 

2. Discard posts without an answer accepted by the user or obtains more than 5 votes 

3. Discard answers of posts without URL links. 

4. For each link in the answer, write down the answers to: (1). why are the document and the post relevant; (2). what is the reasoning required to understand the relevance between the post and the document. If these are not possible, discard the link. 

5. Check whether the linked documents provide critical information to understand the post or address the questions. The relevance could include explaining critical concepts or details or providing theorems, lemmas or code pieces that would contribute to solving the problems. Refer to examples for a better understanding on the relevance (Examples in Table 20 to 26) 

6. Use LLMs (e.g., ChatGPT, Claude, etc.) to generate post keywords, or use the post title to search for web pages with large keyword or semantic overlap in Google. Search for at most 5 negative web pages per query. Ensure that hard negatives are totally unhelpful in addressing the post. Common hard negatives include descriptions or documentation of similar contexts but different problems, or code pieces that share the same variable names but are unrelated to the post. 

7. Check all the other negative passages, which include positive and negative documents in previously annotated examples in the same dataset. Ensure that all those documents and current problems focus on different sub-topics, and thus totally irrelevant. 

8. Split every web page into small passages either by two newline symbols, "#" in markdown files or fixed-length tokens, and fine-grain positive ones following criteria in step 4 and 5. 

## E.2 THEOREMQA 

In TheoremQA, the main task for the annotator is to check if the GPT-4 rewritten questions are valid. The specific instructions are as follows: 

1. Read the rewritten question and determine if it is solvable. 

2. If it is solvable, read the original question and solution, and determine if the rewritten question is consistent with the original question. That is, the same reasoning steps and the final answer should hold. 

3. If it is also consistent, mark the question as valid, and make any minor edits to the problem statement (e.g., to improve grammar or fluency) as you see fit. 

4. If it is not solvable or not consistent, read the original question and solution, and correct the rewritten question if possible. If not, then discard the problem. 

## E.3 AOPS 

In AoPS, annotators are tasked to find questions from the AoPS Wiki and record the problems: 

1. Browse through the AoPS Wiki and find topic/category pages (example 1, example 2). 

2. Look through each page and find pages specific theorems or techniques that can be used to solve problems. The page should link to at least two competition problems (example 1, example 2). 

3. Record the links of both the theorem/technique as well as the problem pages. 

The annotators are assigned a category to look for theorems in to avoid overlaps, and the categories are _{_ algebra _,_ geometry _,_ calculus _,_ probability _,_ number theory _,_ other _}_ . After all links are collected, we use a web scraper to collect the problem statement and solutions, and we manually check the quality of the scraped data. 

21 

Preprint. 

## E.4 LEETCODE 

In LeetCode, annotators determine whether a question is grounded in real-world concepts. We give a similar instruction to the annotator as to GPT-4: 

1. Read the problem statement carefully. 

2. Categorize the question into one of three categories: 

   - 0: The question is not grounded in any real-world concepts. The description only uses coding-specific terms, such as "linked list", "binary search", "palindrome", "sorting", etc.. 

   - 1: The question is not grounded in any real-world concepts or real-world concepts that are commonly used in the context of coding, such as needle in a haystack, strings/words, or a spiral matrix. 

   - 2: The question is grounded in real-world concepts that are not commonly used in the context of coding, such as building height, planting trees, or games. It may still use some code-specific terms to specify the data structure involved. 

## E.5 LLM USAGE 

There is no specific procedure for the annotators to use LLMs. The LLMs serve a tool to help annotators understand queries and documents, i.e., whenever they fail to understand something, they ask LLMs for clarification, explanation, etc. They are also used to summarize the content of StackExchange posts for searching negative documents in Google for StackExchange. To diversify the search results, the annotator prompts LLMs with role-playing scenarios (e.g., as a biology student) to generate keywords of the posts. The responses from LLMs are not included in any of **BRIGHT** datasets. Since the responses from LLMs are not guaranteed to be correct, the annotators always search for trustworthy sources to verify the information. The LLMs used in the annotation include ChatGPT[17] , GPT-4, and Claude-3. 

## E.6 PASSAGE SPLIT 

To split web pages or long documents into smaller pieces of passages, we employ simple heuristics with separators like two new line symbols, "#" in markdown files without additional assumptions on file structures. Although this split may not be optimal for every document, it simulates the realistic setting where long documents are automatically processed without human or expert intervention. 

## E.7 FALSE POSITIVE AND FALSE NEGATIVE 

we discuss the rationale for avoiding false positives and false negatives in the data collection process. In StackExchange, the positive relevance between queries and documents is manually verified by annotators, with detailed reasoning traces to illustrate their thinking process. To avoid false negatives, the annotators only select a StackExchange post that clearly distinguishes from previously annotated examples in the same domain, e.g., different entities and semantic meanings in both posts and answers, etc. This ensures that no pair of examples share the same positive documents. 

## E.8 SENSITIVE INFORMATION 

All the data in **BRIGHT** are manually collected, carefully verified, and reviewed to remove any personally identifiable information or offensive content. 

## F LIMITATIONS AND FUTURE WORK 

One limitation of this work is that the judgment about relevance between queries and documents is subjective. Even if the StackExchange answers are accepted by the users or obtain high votes, it is not guaranteed that everyone will agree the referenced documents are relevant. We may not expect 

> 17https://chatgpt.com/ 

22 

Preprint. 

human retrieval results to be exactly the same as our annotation. However, we mitigate this issues by using multiple annotators and only retain the queries in which all annotators, including domain experts, agree on the relevance. Therefore, we believe that the relevance judgment in **BRIGHT** is reliable and consistent. 

Aside from retrieval, other embedding tasks such as clustering may also require reasoning. We have not addressed multi-modal settings in this paper, but they represent intriguing avenues for future exploration. 

## G POTENTIAL SOCIAL IMPACTS 

This paper presents a new retrieval benchmark that features relevance beyond lexical and semantic similarity and requires intensive reasoning to solve. There are many potential societal consequences of our work, e.g., improving search algorithms, fostering better information access, developing more advanced retrieval models, etc. It could also promote collaboration among researchers and facilitate the development of more effective search engines, ultimately benefiting society by enhancing the way people find and access information. 

23 

Preprint. 

Table 13: **Prompt used to rewrite TheoremQA questions.** Blue text denote instance-specific inputs. Violet text denotes the rewritten question and answer outputted by GPT-4. For each question, we provide the original question and answer, as well as the theorem name and theorem definition from the original TheoremQA dataset. The instruction prompts the model to rewrite the question with a different surface form without changing the reasoning steps or the final answer. We hand-write three examples to illustrate the rewriting process. We also allow the model to skip the question. 

Rewrite the following question such that the logical steps and final answer are still the same, only change the surface form of the question. Avoid using the words related to the theorem used in the question. You can do this by using real-world examples and applications to illustrate the concepts, in a way that is easier to understand for a layman. Try to ground the question in a real-world context such as finance and engineering problems, but be creative and feel free to use any domain you like! However, the question should be standalone and solvable. Rewrite the question in a json format, with the fields "question" and "answer". The "theorem" field indicates the theorem used in the question, but you should not use the words related to the theorem in the rewritten question. If you do not think that the question cannot be rewritten in layman’s terms in a standalone fashion, then you can write a json object with the field "skip" set to True. 

{ "question": "In a group of 10 people, each of whom has one of 3 different eye colors, at least how many people must have the same eye color?", "answer": "4", "theorem": "pigeonhole principle", "theorem_definition": "The Pigeonhole Principle is a fundamental concept in combinatorics, a branch of mathematics that deals with counting and arranging objects. _[additional text omitted...]_ " } one demonstration was omitted... { "question": "dy/dt = \sqrt{t}, y(1) = 1. What is y(4)?", "answer": "5.667", "theorem": "integral rules", "theorem_definition": "Integral rules in calculus are a set of techniques _[additional text omitted...]_ " } { "question": "You’re tracking the growth of a plant from a seed. The rate at which the plant grows in height is equal to the square root of the number of days since you planted it. One day after the seed was first planted, the plant was 1 inch tall. How tall will the plant be after 4 days?", "answer": "5.667" } 

{ "question": "Given image \begin{tabular}{|llll|} \hline 7 & 1 & 6 & 0 \\ 3 & 3 & 7 & 6 \\ 6 & 6 & 5 & 7 \\ \hline \end{tabular} , and the bit -depth of the image is 4. Suppose you want to use the thresholding technique to segment the image. What is the appropriate threshold value based on the histogram of the image? Follow the following rule when you do thresholding or grouping: pixel $(i, j)\in$ Group A pixels if $g(i, j )\leq$ current threshold $\mathrm{T}$; pixel $(i, j)\in$ Group B pixels otherwise, where $g(i, j)$ is the intensity value of pixel $(i, j)$.", "answer": "4", "theorem": "image contrast ", "theorem_definition": "Image contrast in signal processing refers to the difference in intensity or color between various elements or regions within an image. _[additional text omitted...]_ " } 

{ "question": "Imagine you have a collection of 12 unique stickers, each with a number from 0 to 7 representing its rarity level. You’re planning to divide these stickers into two groups based on their rarity to make it easier for collectors to understand. You decide to use a method where stickers with a rarity level at or below a certain number go into the ’Common’ group, and those above this number go into the ’Rare’ group. Given the distribution of stickers’ rarity levels as follows: two stickers each of levels 7, 6, and 5; three stickers of level 3; and one sticker each of levels 1 and 0. What is the rarity level that should be used as the cutoff to divide the stickers into the ’Common’ and ’Rare’ groups, ensuring a balanced understanding of rarity?", "answer": "4" } 

24 

Preprint. 

Table 14: **TheoremQA-Q Theorem names and the excluded topic names from CAMEL-Math.** 

|TheoremQA-Q Theorem|CAMEL-Math Topics|
|---|---|
|acyclic graph|Graph theory, Combinatorics|
|binomial theorem|Combinatorics, Algebra|
|catalan-mingantu number|Combinatorics|
|cauchy’s integral theorem|Complex analysis|
|cayley’s formula|Graph theory, Combinatorics|
|convexity|Optimization|
|cramer rao lower bound|Statistics|
|defnite matrix criteria|Algebra, Linear algebra|
|derivative chain rule|Calculus|
|double integral theorem|Calculus|
|eigenvalues and eigenvectors|Linear algebra|
|euler’s method|Calculus, Numerical analysis, Differential equations|
|euler’s theory|Graph theory, Combinatorics|
|expected utility|Game theory|
|fsher information|Statistics|
|fourier’s theorem|Fourier analysis|
|gauss’s lemma|Number theory, Algebra|
|integral rules|Calculus|
|intermediate value theorem|Calculus|
|isomorphisms|Group theory, Algebra|
|jensen’s inequality|Statistics, Probability|
|l’hôpital’s rule|Calculus|
|limit laws for sequences|Calculus|
|limiting theorem|Calculus|
|linear independence|Algebra, Linear algebra|
|linear subspaces|Algebra, Linear algebra|
|linear systems|Algebra, Linear algebra|
|liouville’s theorem|Complex analysis|
|martingale|Statistics, Probability|
|matrix determinant formula|Algebra, Linear algebra|
|maximal planar graph|Graph theory, Combinatorics|
|maximum entropy|Statistics, Probability|
|message passing algorithm|Graph theory, Combinatorics|
|multinomial theorem|Combinatorics|
|newton-raphson method|Calculus, Numerical analysis|
|order|Group theory, Algebra|
|ordinary differential equation|Calculus, Differential equations|
|p-value|Statistics|
|pigeonhole principle|Combinatorics|
|poisson process|Statistics, Probability|
|projection theory|Algebra, Linear algebra|
|ramsey’s theorem|Graph theory, Combinatorics|
|rolle’s theorem|Calculus|
|series convergence|Calculus|
|shortest path|Graph theory, Optimization, Combinatorics|
|similarity|Geometry|
|squeeze theorem|Calculus|
|stirling number of the frst kind|Combinatorics|
|stirling number of the second kind|Combinatorics|
|t-test|Statistics|
|taylor’s approximation theorem|Calculus|
|trapezoidal rule|Calculus, Numerical analysis|
|vertex cover|Graph theory, Combinatorics|
|viterbi algorithm|Statistics|
|wave theorem|Differential equations|



25 

Preprint. 

Table 15: **Subfield name in the original TheoremQA dataset and their corresponding categories in BRIGHT. Each category is used to specify the excluded problem-solution pairs from MATH and AoPS.** 

|Name|Category|
|---|---|
|Algebra|algebra|
|Atomic physics|others|
|Calculus|calculus|
|Celestial mechanics|others|
|Classic mechanics|others|
|Combinatorics|number theory|
|Complex analysis|calculus|
|Computer networking|others|
|Condensed matter physics|others|
|Derivatives|calculus|
|Economics|others|
|Electromagnetism|others|
|Equity investments|others|
|Fixed income|others|
|Fluid mechanics|others|
|Functional analysis|calculus|
|Geometry|geometry|
|Graph theory|others|
|Group theory|algebra|
|Information theory|others|
|Kinetics|others|
|Machine learning|others|
|Mathematical analysis|calculus|
|Number theory|number theory|
|Numerical analysis|calculus|
|Optics|others|
|Particle|others|
|Portfolio management|others|
|Probability theory|probability|
|Quantitive methods|others|
|Quantum|others|
|Real analysis|calculus|
|Relativity|others|
|Signal processing|others|
|Statistical physics|others|
|Statistics|others|
|Stochastic process|probability|
|Thermodynamics|others|
|Wave|others|



26 

Preprint. 

Table 16: **AoPS theorems and techniques and the excluded topic names from CAMEL-Math and TheoremQA.** 

|AoPS Theorem|CAMEL-Math Topics|TheoremQA Theorems|
|---|---|---|
|Binomial Theorem|Combinatorics, Algebra|binomial theorem, multino-|
|||mial theorem|
|Vietas Formulas|Algebra|vieta’s formula, birg-vieta’s|
|||theorem|
|Ptolemys theorem|Geometry|properties of kites, similarity,|
|||rhombus, rectangle, quadri-|
|||lateral, triangle, isosceles tri-|
|||angle, parallelogram|
|Recursive Series|Calculus, Algebra||
|Power of a Point|Geometry|similarity|
|Ball and urn|Combinatorics||
|Newtons Sums|Algebra|vieta’s formula|
|Probability|Probability, Statistics|probability|
|Fibonacci sequence|Combinatorics, Number the-||
||ory||
|Chicken McNugget Theorem|Combinatorics, Number the-||
||ory||
|Central Tendency|Statistics||
|Principle of Inclusion Exclusion|Combinatorics, Set theory|inclusion-exclusion principle|
|Factorial|Combinatorics||
|Picks Theorem|Geometry|similarity, triangle, isosce-|
|||les triangle, parallelogram,|
|||rhombus, quadrilateral, rect-|
|||angle, triangle midsegment|
|||theorem|
|Shoelace Theorem|Geometry|rhombus, rectangle, quadri-|
|||lateral, triangle, isosceles tri-|
|||angle, parallelogram|
|Geometric probability|Geometry, Probability||
|Euclidean algorithm|Number theory, Cryptogra-|euclidean algorithm|
||phy, Algebra||
|Mass Points|Geometry|similarity, triangle, isosce-|
|||les triangle, parallelogram,|
|||rhombus, quadrilateral, rect-|
|||angle, triangle midsegment|
|||theorem|
|Geometric Series|Algebra||
|Triangle Inequality|Geometry|inequalities, triangle, isosce-|
|||les triangle|
|Simons Favorite Factoring Trick|Number theory, Algebra||
|Properties of Logarithms|Algebra||
|Fermats Little Theorem|Number theory, Cryptogra-|fermat’s little theorem, eu-|
||phy|ler’s totient theorem|



27 

Preprint. 

Table 17: **Prompt used to check TheoremQA-Theorem documents.** Blue text denote instancespecific inputs. For each question, we provide the problem statement, the solution, the theorem name, the theorem definition, and the document text from a ProofWiki document. 

Instruction: Determine if the given text can be helpful in understanding and solving the given problem. Use the theorem, its definition, and the problem solution to help you make a decision. The text can be helpful if it uses very similar reasoning steps as the solution and applies the theorem in a related way as the solution applies the theorem to solve the problem. The text should be able to assist a student who is not familiar with the theorem in ultimately solving the problem. The input is given to you in a json format with the following keys: Problem statement: The problem statement that the student is trying to solve. Solution: The solution to the problem. Theorem: The theorem that is used in the solution. Theorem definition: The definition of the theorem. Text: The text that you need to evaluate. Think step by step and reason about the theorem and the text first before finally making a decision. Output True if the text is helpful, and False if it is not. { "Problem statement": "", "Solution": "", "Theorem": "", "Theorem definition": "", "Text": "", } Now, write your answer in the following format: Reasoning: [your reasoning here] Answer: [True/False] 

Table 18: **Theorems and techniques used in the AoPS dataset, and their corresponding subfield categories.** 

|Name|Category|
|---|---|
|Ball and urn|number theory|
|Binomial Theorem|algebra|
|Central Tendency|others|
|Chicken McNugget Theorem|algebra|
|Euclidean algorithm|number theory|
|Factorial|number theory|
|Fermat’s Little Theorem|number theory|
|Fibonacci sequence|number theory|
|Geometric Series|algebra|
|Geometric probability|probability|
|Mass Points|geometry|
|Newtons Sums|algebra|
|Picks Theorem|geometry|
|Power of a Point|geometry|
|Principle of Inclusion Exclusion|number theory|
|Probability|probability|
|Properties of Logarithms|algebra|
|Ptolemy’s theorem|geometry|
|Recursive Series|algebra|
|Shoelace Theorem|geometry|
|Simon’s Favorite Factoring Trick|number theory|
|Triangle Inequality|algebra|
|Vieta’s Formulas|algebra|



28 

Preprint. 

Table 19: **Prompt used to check LeetCode questions.** Blue text denote instance-specific inputs. Violet text denotes the output from GPT-4. For each question, we provide the title and the problem statement. We use six in-context demonstrations, and the instruction prompts the model to categorize the question based on the criteria provided. Although there are three possible labels, we only keep questions that were labeled as 2, which are grounded in real-world concepts. For sake of brevity, we only show one example, but other examples can be found on the code repo. 

Read the coding question and categorize it using the following criteria: 0: The question is not grounded in any real-world concepts. The description only uses coding-specific terms, such as "linked list", "binary search", "palindrome", "sorting", etc.. 1: The question is not grounded in any real-world concepts or real-world concepts that are commonly used in the context of coding, such as needle in a haystack, strings/words, or a spiral matrix. 2: The question is grounded in real-world concepts that are not commonly used in the context of coding, such as building height, planting trees, or games. It may still uses some code-specific terms to specify the data structure involved. 

You should only consider the initial problem statement and problem title, not the examples or constraints. Use the following examples to help you categorize the question: 

Example 1: {{ "title": "Merge Two Sorted Lists", "question": "You are given the heads of two sorted linked lists ‘list1‘ and ‘list2‘. Merge the two lists in a one **sorted** list. The list should be made by splicing together the nodes of the first two lists. Return _the head of the merged linked list_. **[Example][1:] ** _rest of the example omitted..._ }} {{ "label": 0 }} _Examples 2-6 are omitted for brevity..._ Now, consider the question below and categorize it using the criteria above. Output your answer in a json format: {{ "title": "Container With Most Water", "question": "You are given an integer array ‘height‘ of length ‘n‘. There are ‘n‘ vertical lines drawn such that the two endpoints of the ‘ith‘ line are ‘(i, 0)‘ and ‘(i, height[i])‘. Find two lines that together with the x-axis form a container, such that the container contains the most water. Return _the maximum amount of water a container can store_. _rest of the question omitted..._ " }} { "label": 2 } 

29 

Preprint. 

Table 20: **Biology example.** The positive document explains the concept of meristem, which is the mechanism by which plant cells develop into other tissues and organs. This provides critical support for why a tree chunk sprouts and grows after being cut. 

## _**Query**_ 

After I cut trees into logs and remove the branches in winter, they start growing. They sprout out and grow completely normal looking stems and leaves and maintain them all summer. The sprouts mostly appear around the cut branches. Sometimes they last all winter and grow for another year. How does a tree trunk sprout and grow after being cut? 

## _**Chain-of-thought reasoning to find documents**_ 

The essential question is to figure out the growing mechanism of trees after being cut. It is usually related to cell division, which later develops into tissues and organs. We need to find relevant information of these cells in plants and their growing capabilities after being cut. 

## _**Example positive document**_ 

## Meristem 

Tunica-corpus model of the apical meristem (growing tip). The epidermal (L1) and subepidermal (L2) layers form the outer layers called the tunica. The inner L3 layer is called the corpus. Cells in the L1 and L2 layers divide in a sideways fashion, which keeps these layers distinct, whereas the L3 layer divides in a more random fashion. 

In cell biology, the meristem is a type of tissue found in plants. It consists of undifferentiated cells (meristematic cells) capable of cell division. Cells in the meristem can develop into all the other tissues and organs that occur in plants. These cells continue to divide until they become differentiated and lose the ability to divide. 

There are three types of meristematic tissues: apical (at the tips), intercalary or basal (in the middle), and lateral (at the sides also known as cambium). At the meristem summit, there is a small group of slowly dividing cells, which is commonly called the central zone. Cells of this zone have a stem cell function and are essential for meristem maintenance. The proliferation and growth rates at the meristem summit usually differ considerably from those at the periphery. 

... Under appropriate conditions, each shoot meristem can develop into a complete, new plant or clone. Such new plants can be grown from shoot cuttings that contain an apical meristem. 

... 

## _**Example negative document**_ 

Cutting firewood from felled trees safely and easily (bucking) 

There are loads of articles on the web showing how (and sometimes how not to!) cut down trees, but not so much about actually cutting firewood after the tree has been felled and stripped of its branches (called snedding or limbing in arborist speak). This ‘how to’ shows the easiest way to produce firewood by cutting many logs at the same time. I actually use my firewood cutting operation as part of my fitness regime, much more useful than going to the gym and the same muscle burn the day after! 

I start by cutting the trees into 10’ (3m) logs so that I can easily drag them to my cleared working area close to the pick up point. 

You can make a homemade ‘cradle’ that holds a whole bunch of logs from sturdy construction lumber (2 by 4). This enables you to safely cut many at once, as long as you put some of the heavy logs both at the bottom and on the top. Don’t forget to make the cradle narrow enough for the size of chainsaw bar you have. Usually not more than 12” or 300mm inside measurement. Make sure that it is constructed in such a way that you cannot hit any nails etc with the chainsaw. 

Place the logs onto the cradle keeping the ends off the logs flush one side ( or at least staggered in 12” steps). If you are fussy like me you can mark one of the top logs at 12” (300mm) intervals with chalk to ensure even log lengths. Then cut alternate sides (so that it doesn’t topple over to one side), finishing off with a couple of cuts over the cradle itself, don’t forget to stop before you saw the thing in half lol! (See upgraded version below...) 

... 

30 

Preprint. 

Table 21: **Earth Science example** . The positive document describes the Airy’s isostasy model, which is needed in calculating the hydrostatic equilibrium. 

## _**Query**_ 

How to calculate hydrostatic equilibrium? 

I’m trying to solve the following problem. The sea level in the past was 200 m higher than today. The seawater became in isostatic equilibrium with the ocean basin. what is the increase in the depth x of the ocean basins? Water density is _pw_ = 1000 kg _m[−]_[3] , and mantle density is 3300 kg _m[−]_[3] 

Using the compensation column, I reach: _x_ = ( _pw ∗_ 200 _m_ )/3300 = 60.60 m but normally I expected to find 290 m. Can someone explain to me what’s wrong? 

## _**Chain-of-thought reasoning to find documents**_ 

This problem is about the hydrostatic equilibrium. This may be related to Pascal’s law, which gives insights on the pressure change in fluid’s mechanics. To understand hydrostatic equilibrium or Pascal’s law, it would be helpful to provide a detailed explanation of the concept and formula used in calculations. Other content on the application of the theorems, or related lemmas that also focus on the discussion of isotatic equilibrium with the ocean basin would also be useful. 

## _**Example positive document**_ 

Airy 

The basis of the model is Pascal’s law, and particularly its consequence that, within a fluid in static equilibrium, the hydrostatic pressure is the same on every point at the same elevation (surface of hydrostatic compensation): 

_h_ 1 _· p_ 1 = _h_ 2 _· p_ 2 = _h_ 3 _· p_ 3 = _...hn · pn_ 

For the simplified picture shown, the depth of the mountain belt roots (b1) is calculated as follows: 

( _h_ 1 + _c_ + _b_ 1) _pc_ = ( _cpc_ ) + ( _b_ 1 _pm_ ) 

_b_ 1( _pm − pc_ ) = _h_ 1 _pc b_ 1 = _h_ 1 _pc/_ ( _pm − pc_ ) 

where _pm_ is the density of the mantle (ca. 3,300 kg _m[−]_[3] ) and _pc_ is the density of the crust (ca. 2,750 kg _m[−]_[3] ). Thus, generally: _b_ 1 = 5 _h_ 1 

... 

## _**Example negative document**_ 

Global or eustatic sea level has fluctuated significantly over Earth’s history. The main factors affecting sea level are the amount and volume of available water and the shape and volume of the ocean basins. The primary influences on water volume are the temperature of the seawater, which affects density, and the amounts of water retained in other reservoirs like rivers, aquifers, lakes, glaciers, polar ice caps and sea ice. Over geological timescales, changes in the shape of the oceanic basins and in land/sea distribution affect sea level. In addition to eustatic changes, local changes in sea level are caused by the earth’s crust uplift and subsidence. Over geologic time sea level has fluctuated by more than 300 metres, possibly more than 400 metres. The main reasons for sea level fluctuations in the last 15 million years are the Antarctic ice sheet and Antarctic post-glacial rebound during warm periods. 

The current sea level is about 130 metres higher than the historical minimum. Historically low levels were reached during the Last Glacial Maximum (LGM), about 20,000 years ago. The last time the sea level was higher than today was during the Eemian, about 130,000 years ago. ... 

31 

Preprint. 

Table 22: **Economics example** . The positive document describes the stochastic dominance, which is the underlying concept used to solve the problem. 

## _**Query**_ 

Which of these two lotteries, a consumer with Von-Neumann Morgenstern preferences will choose under exponential distribution? Consider two lotteries each having an exponential distribution. The function of cumulative distribution of an exponential distribution is: _F_ ( _x_ ; _λ_ ) = 1 _− e[−][λx] ∀x ∈ R_ + _._ The expected gain given by this distribution is _E_ [ _X_ ] = 1 _/λ_ . Suppose that the first lottery, _F_ 1 has a parameter _λ_ 1 and the second, _F_ 2 has a parameter _λ_ 2. Suppose that _λ_ 1< _λ_ 2. Which of these two lotteries, a consumer with Von-Neumann Morgenstern preferences will choose? I am confused on this question. _**Chain-of-thought reasoning to find documents**_ 

The expected value is different from the expected utility. The preferences under uncertainty in this case should be linked to first-order stochastic dominance. We need to look up the detailed definition to understand how the problem could be solved by using that framework. _**Example positive document**_ 

Statewise dominance implies first-order stochastic dominance (FSD), which is defined as: Random variable A has first-order stochastic dominance over random variable B if for any outcome x, A gives at least as high a probability of receiving at least x as does B, and for some x, A gives a higher probability of receiving at least x. In notation form, _P_ [ _A ≥ x_ ] _≥ P_ [ _B ≥ x_ ] for all x, and for some x, _P_ [ _A ≥ x_ ] _> P_ [ _B ≥ x_ ]. In terms of the cumulative distribution functions of the two random variables, A dominating B means that _FA_ ( _x_ ) _≤ FB_ ( _x_ ) for all x, with strict inequality at some x. In the case of non-intersecting distribution functions, the Wilcoxon rank-sum test tests for first-order stochastic dominance. 

... Extended example Consider three gambles over a single toss of a fair six-sided die: State (die result) 1 2 3 4 5 6 gamble A wins $ 1 1 2 2 2 2 gamble B wins $ 1 1 1 2 2 2 gamble C wins $ 3 3 3 1 1 1 Gamble A statewise dominates gamble B because A gives at least as good a yield in all possible states (outcomes of the die roll) and gives a strictly better yield in one of them (state 3). Since A statewise dominates B, it also first-order dominates B. 

... 

## _**Example negative document**_ 

Which Lottery Is Easiest To Win Within the U.S.? 

Winning the lottery is one of the least likely things to happen in your life. However, some lotteries are much easier to win than others. 

It’s common for a lot of Americans to daydream about winning the lottery. These dreams became a reality for a few lucky winners as they became millionaires overnight. The cold truth is that the overwhelming majority of lottery dreams will remain dreams. Most people know that the odds of winning the lottery are not in your favor. If it were easy to win, everyone would play, and the prizes would only be a few cents. 

You’re likelier to be struck by lightning, live to be 110 years old, and be declared a saint by the Catholic Church. 

The combination of high jackpots and low odds has caused many people to try and cheat in the lottery. The attempts might be successful for a short period, but they always end with prison sentences. 

Fortunately, there are a few ways that you can legally improve your odds of winning. It will start by choosing games that have the best odds. 

... 

32 

Preprint. 

Table 23: **Psychology example** . The positive document describes the critical equation needed to explain the confusion in the post. 

_**Query**_ Why do fNIRS devices commonly use two different frequencies? One of the most common techniques used for functional neuroimaging nowadays is functional near infra-red spectroscopy (fun fact: IIRC Natalie Portman worked on a research paper involving fNIRS as the modality), which shines near infrared light into the brain from a source to a detector (both called optodes) in a "banana" shape. It’s not uncommon to read that most of these devices, be they continuous wave (CW) or one of the two kinds that involve fast modulation, frequency domain (FD) or time domain (TD), require two separate frequencies to be emitted. For instance, NIRx explains it as follows on their website: "For neuro-imaging applications it is by far most common to illuminate with two discrete wavelength, which is the minimum requirement to assess relative variations of both oxygenation states of the hemoglobin molecule independently." Why is that the case? I haven’t delved into the intricacies of it, but no reason immediately jumps out at me. For instance, in the case of CW, the relative difference in intensity is all that matters, so why do we need two frequencies? 

## _**Chain-of-thought reasoning**_ 

In order to understand why we need two frequencies in fNIRS, we will need to check the measurements and calculation in continuous wave systems. This may be related to oxygenated (HbO) and deoxygenated (HbR) hemoglobin in the brain. We need to refer to the dual-wavelength approach, which is fundamental to fNIRS’s ability to provide meaningful information about local brain activity. 

## _**Example positive document**_ 

... Where OD is the optical density or attenuation, _I_ 0 is emitted light intensity, _I_ is measured light intensity, _ϵ_ is the attenuation coefficient, [ _X_ ] is the chromophore concentration, _l_ is the distance between source and detector and _DPF_ is the differential path length factor, and _G_ is a geometric factor associated with scattering. 

When the attenuation coefficients _ϵ_ are known, constant scattering loss is assumed, and the measurements are treated differentially in time, the equation reduces to: ∆[ _X_ ] = ∆ _OD/_ ( _ϵd_ ) Where d is the total corrected photon path-length. Using a dual wavelength system, measurements for HbO2 and Hb can be solved from the matrix equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval_images/iclr_2025_BRIGHT_A_Realistic_and_Challenging_Benchmark_for_Reasoning-Intensive_Retrieval.pdf-0033-08.png)


... 

## _**Example negative document**_ 

What Is Wave Frequency? 

The number of waves that pass a fixed point in a given amount of time is wave frequency. Wave frequency can be measured by counting the number of crests (high points) of waves that pass the fixed point in 1 second or some other time period. The higher the number is, the greater the frequency of the waves. The SI unit for wave frequency is the hertz (Hz), where 1 hertz equals 1 wave passing a fixed point in 1 second. The Figure below shows high-frequency and low-frequency transverse waves. 

Q: The wavelength of a wave is the distance between corresponding points on adjacent waves. For example, it is the distance between two adjacent crests in the transverse waves in the diagram. Infer how wave frequency is related to wavelength. 

A: Waves with a higher frequency have crests that are closer together, so higher frequency waves have shorter wavelengths. 

... 

33 

Preprint. 

Table 24: **Robotics example** . The positive document provides the code pieces to modify. 

|**_Query_**|
|---|
|Can’t see debug messages using RCLCPP_DEBUG|
|I can’t see messages using RCLCPP_DEBUG by terminal and rqt, but I can using other levels|
|of verbosity( INFO, ERROR, FATAL. . . ).Selecting debug in rqt to see those messages doesn’t|
|work either.|
|I’m using rolling,<br>working in C++ in a plugin of a controller and launching|
|tb3_simulation_launch.py from nav2_bringup.|
|I also saw a post here where they recommended to set the environment variable:|
|RCLCPP_LOG_MIN_SEVERITY=RCLCPP_LOG_MIN_SEVERITY_DEBUG|
|but that didn’t work either. It must be something silly that I’m missing. Has this ever happened|
|to you?|
|Thank you|
|**_Chain-of-thought reasoning_**|
|The problem looks like to be related to the setting of log level. If selecting debug in rqt does|
|not work, we can check the code about potential arguments that could affect the logging|
|messages and verbosity,|
|**_Example positive document_**|
|...|
|declare_log_level_cmd = DeclareLaunchArgument(|
|’log_level’, default_value=’info’, description=’log level’|
|)|
|load_nodes = GroupAction(|
|condition=IfCondition(PythonExpression([´not ´, use_composition])),|
|actions=[|
|SetParameter(’use_sim_time’, use_sim_time),|
|Node(|
|package=’nav2_controller’,|
|executable=’controller_server’,|
|output=’screen’,|
|respawn=use_respawn,|
|respawn_delay=2.0,|
|parameters=[confgured_params],|
|arguments=[’–ros-args’, ’–log-level’, log_level],|
|remappings=remappings + [(’cmd_vel’, ’cmd_vel_nav’)],|
|),|
|...|
|**_Example negative document_**|



The logger methods are named after the level or severity of the events they are used to track. The standard levels and their applicability are described below (in increasing order of severity): DEBUG Detailed information, typically of interest only when diagnosing problems. INFO Confirmation that things are working as expected. WARNING An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected. ERROR Due to a more serious problem, the software has not been able to perform some function. CRITICAL A serious error, indicating that the program itself may be unable to continue running. ... 

34 

Preprint. 

Table 25: **Stack Overflow example** . The positive document provides the documentation that explains the method to solve the problem. 

## _**Query**_ 

Is there a Snowflake command that can do the following: 

a,b,c 1,10,0.1 2,11,0.12 3,12,0.13 to a table like this: key,value a,1 a,2 a,3 b,10 b,11 b,13 c,0.1 c,0.12 c,0.13 ? 

This operation is often called melt in other tabular systems, but the basic idea is to convert the table into a list of key value pairs. 

There is an UNPIVOT in SnowSQL, but as I understand it UNPIVOT requires to manually specify every single column. This doesn’t seem practical for a large number of columns. 

_**Chain-of-thought reasoning steps to find documents**_ This operation is a kind of table transformation like reshaping, serialization, or flattening. We are looking for an operation in Snowflake that can take 2-dimensional values, and transform them into an view that presents one-one correlation mapping between key and value. 

## _**Example positive document**_ 

FLATTEN Flattens (explodes) compound values into multiple rows. FLATTEN is a table function that takes a VARIANT, OBJECT, or ARRAY column and produces a lateral view (i.e. an inline view that contains correlation referring to other tables that precede it in the FROM clause). FLATTEN can be used to convert semi-structured data to a relational representation. Syntax FLATTEN( INPUT => <expr> [ , PATH => <constant_expr> ] [ , OUTER => TRUE | FALSE ] [ , RECURSIVE => TRUE | FALSE ] [ , MODE => ’OBJECT’ | ’ARRAY’ | ’BOTH’ ] ) ... 

## _**Example negative document**_ 

SYSTEM$EXTERNAL_TABLE_PIPE_STATUS 

Retrieves a JSON representation of the current refresh status for the internal (hidden) pipe object associated with an external table. 

Automatically refreshing the metadata for an external table relies internally on Snowpipe, which receives event notifications when changes occur in the monitored cloud storage. For more information, see Introduction to external tables. Syntax SYSTEM$EXTERNAL_TABLE_PIPE_STATUS( ’<external_table_name>’ ) ... 

35 

Preprint. 

Table 26: **Sustainable Living example** . The positive document explains the critical concept of soluble salts, which indicate that they contain dissolved minerals and could be harmful to plants. 

_**Query**_ How good is it to reuse water from plant pots? I’m living in an apartment, and after I water my plants the water goes to plates below the pots. The pots are in a metallic structure above the plates, so I can take the plates to reuse the water (throwing it at the plants again). This reuse seems beneficial, because I think I can get rid of mosquitoes that would reproduce in the stagnated water. And also some nutrients of the soil (as well as earthworms) can return to the vase. Is there some negative points in doing that? _**Chain-of-thought reasoning to find documents**_ 

The water that accumulates in the plates as a result of watering is likely to contain minerals, soluble salts, and other materials that exist in soil or plants. To figure out whether there is any negative point in reusing that water, we thus need to understand whether the components in that water will result in any adverse effects. 

_**Example positive document**_ Soluble Salts Soluble salts may accumulate on the top of the soil, forming a yellow or white crust. A ring of salt deposits may form around the pot at the soil line or around the drainage hole. Salts may also build up on the outside of clay pots. In house plants, signs of excess soluble salts include reduced growth, brown leaf tips, dropping of lower leaves, small new growth, dead root tips, and wilting. Soluble salts are minerals dissolved in water. Fertilizer dissolved in water becomes a soluble salt. When water evaporates from the soil, the minerals or salts stay behind. As the salts in the soil become more and more concentrated, it becomes more difficult for plants to take up water. If salts build up to an extremely high level, water can be taken out of the root tips, causing them to die. High levels of soluble salts damage the roots directly, weakening the plant and making it more susceptible to attack from insects and diseases. One of the most common problems associated with high salt levels is root rot. 

The best way to prevent soluble salt injury is to stop the salts from building up. When watering, allow some water to drain through the container and then empty the saucer. Do not allow the pot to sit in water. If the drained water is absorbed by the soil, the salts that were washed out are reabsorbed through the drainage hole or directly through a clay pot. 

... 

## _**Example negative document**_ 

Water reuse in California 

Water reuse in California is the use of reclaimed water for beneficial use. As a heavily populated state in the drought-prone arid west, water reuse is developing as an integral part of water in California enabling both the economy and population to grow. 

Wastewater Reclaimed water is treated wastewater that comes from homes and businesses, such as sink water, shower water, and toilet water including everything dumped into wastewater drains from laundry soap to bleach to oil to human waste. Wastewater can be divided into greywater and blackwater, with the first being defined as water that had been used for laundry, bathing, sink washing, and dishwashers. Blackwater is defined as sewage that includes feces from toilets.[1] Due to the low amounts of physical pollutants in greywater, most of its contaminants are dissolved organic matter, which can be physically filtered and cleaned through various membranes, as well as through biological treatment methods. 

... 

36 

Preprint. 

Table 27: **LeetCode example.** Both the query and the positive document uses two pointers in the solution. The negative example is retrieved by BM25. 

## _**Query**_ 

Given ‘n’ non-negative integers representing an elevation map where the width of each bar is ‘1’, compute how much water it can trap after raining. **Example 1:** **Input:** height =[0 _,_ 1 _,_ 0 _,_ 2 _,_ 1 _,_ 0 _,_ 1 _,_ 3 _,_ 2 _,_ 1 _,_ 2 _,_ 1] **Output:** 6 

**Explanation:** The above elevation map (black section) is represented by array [0 _,_ 1 _,_ 0 _,_ 2 _,_ 1 _,_ 0 _,_ 1 _,_ 3 _,_ 2 _,_ 1 _,_ 2 _,_ 1]. In this case, 6 units of rain water (blue section) are being trapped. **Example 2:** **Input:** height = [4 _,_ 2 _,_ 0 _,_ 3 _,_ 2 _,_ 5] **Output:** 9 **Constraints:** * ‘n == height.length’ * ‘1 <= n <= 2 * 104’ * ‘0 <= height[i] <= 105’ 

## _**Chain-of-thought reasoning to find documents**_ 

This problem can be solved using a two-pointer approach, and uses ideas from dynamic programming to keep track of the maximum height to the left and right of each bar. We can find other example code that also use these techniques. 

## _**Example positive document**_ 

def max_area(height): 

"""You are given an integer array ‘height’ of length ‘n’. There are ‘n’ vertical lines drawn such that the two endpoints of the ‘ith’ line are ‘(i, 0)’ and ‘(i, height[i])’. Find two lines that together with the x-axis form a container, such that the container contains the most water. Return the maximum amount of water a container can store. **Notice** that you may not slant the container. **Example 1:** **Input:** height = [1 _,_ 8 _,_ 6 _,_ 2 _,_ 5 _,_ 4 _,_ 8 _,_ 3 _,_ 7] **Output:** 49 **Constraints:** * ‘n == height.length’ * ‘2 <= n <= 105’ * ‘0 <= height[i] <= 104’""" max_area, left, right = 0, 0, len(height) - 1 while left < right: max_area = max(max_area, min(height[left], height[right]) * (right - left)) if height[left] < height[right]: left += 1 else: right -= 1 return max_area 

## _**Example negative document**_ 

def get_power(x): 

"""Your country has an infinite number of lakes. Initially, all the lakes are empty, but when it rains over the ‘nth’ lake, the ‘nth’ lake becomes full of water. If it rains over a lake that is **full of water**, there will be a **flood**. Your goal is to avoid floods in any lake. Given an integer array ‘rains‘ where: _...rest of the document omitted_ 

37 

Preprint. 

## Table 28: **Pony example** 

## _**Query**_ 

Given the lengths of a triangle’s sides, write a pony program to classify it as equilateral, isosceles or scalene. 

## _**Chain-of-thought reasoning to find documents**_ 

To determine the category of the triangle, we need to compare the side lengths. A common syntax for conditions and comparison used in many other programming languages is the ‘if’ or ‘switch’ related to control structure. We need to find the syntax that allows to specify such conditions. 

## _**Example positive document**_ 

# Control Structures 

To do real work in a program you have to be able to make decisions, iterate through collections of items and perform actions repeatedly. For this, you need control structures. Pony has control structures that will be familiar to programmers who have used most languages, 

such as ‘if’, ‘while’ and ‘for’, but in Pony, they work slightly differently. 

## Conditionals 

The simplest control structure is the good old ‘if‘. It allows you to perform some action only when a condition is true. In Pony it looks like this: 

if condition then control_body end 

Here is a simple example: 

if a > b then env.out.print("a is bigger") end 

Often the condition may be composed of many sub conditions connected by ‘and’ and ‘or’. 

... 

## _**Example negative document**_ 

# Classes 

Just like other object-oriented languages, Pony has __classes__. A class is declared with the keyword ‘class‘, and it has to have a name that starts with a capital letter, like this: 

class Wombat 

Do all types start with a capital letter? Yes! And nothing else starts with a capital letter. So when you see a name in Pony code, you will instantly know whether it’s a type or not. 

## What goes in a class? 

A class is composed of: 

1. Fields. 2. Constructors. 3. Functions. 

... 

~~38~~ 

Preprint. 

Table 29: **AoPS example.** The negative example is retrieved by BM25, and it uses stars and bars to solve the problem, whereas the query and gold document use the Chicken McNugget Theorem. 

_**Query** (from 2015 AMC 10B Problem 15)_ The town of Hamlet has 3 people for each horse, 4 sheep for each cow, and 3 ducks for each person. Which of the following could not possibly be the total number of people, horses, sheep, cows, and ducks in Hamlet? **(A)** 41 **(B)** 47 **(C)** 59 **(D)** 61 **(E)** 66 

## _**Chain-of-thought reasoning to find documents**_ 

We can use the Chicken McNugget Theorem to solve this problem. We can find other solutions that also apply this theorem. _**Example positive document**_ Find the sum of all positive integers _n_ such that, given an unlimited supply of stamps of denominations 5 _, n,_ and _n_ + 1 cents, 91 cents is the greatest postage that cannot be formed. By the Chicken McNugget theorem, the least possible value of _n_ such that 91 cents cannot be formed satisfies 5 _n −_ (5 + _n_ ) = 91 = _⇒ n_ = 24, so _n_ must be at least 24. For a value of _n_ to work, we must not only be unable to form the value 91, but we must also be able to form the values 92 through 96, as with these five values, we can form any value greater than 96 by using additional 5 cent stamps. Notice that we must form the value 96 without forming the value 91. If we use any 5 cent stamps when forming 96, we could simply remove one to get 91. This means that we must obtain the value 96 using only stamps of denominations _n_ and _n_ + 1. Recalling that _n ≥_ 24, we can easily figure out the working ( _n, n_ + 1) pairs that can used to obtain 96, as we can use at most[96] 24[=][4][ stamps without going over.][The potential sets are] (24 _,_ 25) _,_ (31 _,_ 32) _,_ (32 _,_ 33) _,_ (47 _,_ 48) _,_ (48 _,_ 49) _,_ (95 _,_ 96), and (96 _,_ 97). The last two obviously do not work, since they are too large to form the values 92 through 94, and by a little testing, only (24 _,_ 25) and (47 _,_ 48) can form the necessary values, so _n ∈{_ 24 _,_ 47 _}_ . 24 + 47 = 71 . _**Example negative document**_ Alice has 24 apples. In how many ways can she share them with Becky and Chris so that each of the three people has at least two apples? **(A)** 105 **(B)** 114 **(C)** 190 **(D)** 210 **(E)** 380 Note: This solution uses the non-negative version for stars and bars. A solution using the positive version of stars is similar (first removing an apple from each person instead of 2). This method uses the counting method of stars and bars (non-negative version). Since each person must have at least 2 apples, we can remove 2 _∗_ 3 apples from the total that need to be sorted. With the remaining 18 apples, we can use stars and bars to determine the number of possibilities. Assume there are 18 stars in a row, and 2 bars, which will be placed to separate the stars into groups of 3. In total, there are 18 spaces for stars +2 spaces for bars, for a total of 20 spaces. We can now do �202 �. This is because if we choose distinct 2 spots for the bars to be placed, each combo of 3 groups will be different, and all apples will add up to 18. We can also do this because the apples are indistinguishable. �202 � is 190, therefore the answer is **(C)** 190 . 

39 

Preprint. 

Table 30: **TheoremQA Questions example.** Both the query and the positive document uses the Pigeonhole Principle to solve the problem. The negative document is retrieved by BM25, and it only requires simple arithmetics to solve. 

## _**Query**_ 

Mary is planning to bake exactly 10 cookies, and each cookie may be one of three different shapes – triangle, circle, and square. Mary wants the cookie shapes to be a diverse as possible. What is the smallest possible count for the most common shape across the ten cookies? 

## _**Chain-of-thought reasoning to find documents**_ 

This problem is similar to the formulation of Pigeonhole Principle. We can find solutions to other examples that can also be solved using this principle. 

## _**Example positive document**_ 

Arbitrarily place 19 points in a unit square and cover as many of these points as possible with ~~_√_~~ 2 a circle of diameter[Question:][At least how many points can be guaranteed to be covered?] 3[.] We can divide the unit square into 9 smaller squares, each with side length 1/3. Since there are 19 points and 9 smaller squares, by the Pigeonhole Principle, at least one of these smaller squares must contain at least 3 points. ~~_√_~~ 2 ~~_√_~~ 2 Now, consider a circle with diameter 3[.][The diagonal of each smaller square has length] 3[,] so the circle can be placed inside the smaller square in such a way that it covers the entire square. Therefore, the circle can cover at least 3 points in that square. Therefore, the answer is 3. 

## _**Example negative document**_ 

Yanna baked twenty butter cookies and forty biscuits in the morning. In the afternoon, she baked ten butter cookies and twenty biscuits. How many more biscuits did she bake than butter cookies? She baked 20 + 10 = 30 butter cookies. And, she baked 40 + 20 = 60 biscuits. Therefore, she baked 60 - 30 = 30 more biscuits than butter cookies. The answer is 30 

40 

Preprint. 

Table 31: **TheoremQA Theorems example.** The query uses the Pigeonhole Principle, which is the gold document from ProofWiki. The negative document is retrieved by BM25, which is not relevant to the question 

## _**Query**_ 

Mary is planning to bake exactly 10 cookies, and each cookie may be one of three different shapes – triangle, circle, and square. Mary wants the cookie shapes to be a diverse as possible. What is the smallest possible count for the most common shape across the ten cookies? 

## _**Chain-of-thought reasoning to find documents**_ 

This problem is similar to the formulation of Pigeonhole Principle. We may look into the principle and find hints about how this problem may be solve using that principle. _**Example positive document**_ 

Pigeonhole Principle Tags: Pigeonhole Principle, Named Theorems, Combinatorics begin{theorem} Let _S_ be a finite set whose cardinality is _n_ . Let _S_ 1 _, S_ 2 _, . . . , Sk_ be a partition of _S_ into _k_ subsets. Then: :at least one subset _Si_ of _S_ contains at least _⌊[n] k[⌋]_[elements where] _⌊· ⌋_ denotes the ceiling function. end{theorem} begin{proof} AimForCont no subset _Si_ of _S_ has as many as _⌊[n] k[⌋]_[elements. Then the maximum] number of elements of any _Si_ would be _⌊[n]_[So the total number of elements of] _[ S]_[would] _k[⌋−]_[1][.] be no more than _k_ � _⌊[n] k[⌋−]_[1] � = _k⌊[n] k[⌋−][k]_[.][There are two cases:][:] _[n]_[ is divisible by] _[ k]_[ :] _[n]_[ is not] divisible by _k_ . Suppose _k\n_ . Then _⌊[n][n]_[:] _[k][⌊][n]_[=] _[n][ −][k]_[Thus:] _k[⌋]_[=] _k_[is an integer and:] _k[⌋−][k]_ : _|S|_ =[�] _[k] i_ = 1 _[|][S][i][| ≤][n][ −][k][< n]_[ This contradicts the fact that] _[ |][S][|]_[ =] _[ n]_[.][Hence our assumption] that no subset _Si_ of _S_ has as many as _⌊[n]_[Next, suppose that] _[ kmidn]_[.] _k[⌋]_[elements was false.] Then: : _|S|_ = _k⌊[n] k[⌋−][k][<][k]_[(] _[n]_[ +] _k[ k]_[)] _− k_ = _n_ and again this contradicts the fact that _|S|_ = _n_ . In the same way, our assumption that no subset _Si_ of _S_ has as many as _⌊[n] k[⌋]_[elements was] false. Hence, by Proof by Contradiction, there has to be at least _⌊[n] k[⌋]_[elements in at least one] _Si ⊆ S_ . qed end{proof} 

## _**Example negative document**_ 

begin{definition}[Definition:Conditional/Semantics of Conditional]Let _p_ = _⇒ q_ where = _⇒_ denotes the conditional operator. _p_ = _⇒ q_ can be stated thus:* ””’If” _p_ is true ”then” _q_ is true.”’* ”’ _q_ is true ”if” _p_ is true.”’* ”’(The truth of) _p_ ”implies” (the truth of) _q_ .”’* ”’(The truth of) _q_ ”is implied by” (the truth of) _p_ .”’* ”’ _q_ ”follows from” _p_ .”’* ”’ _p_ is true ”only if” _q_ is true.”’The latter one may need some explanation. _p_ can be either true or false, as can _q_ . But if _q_ is false, and _p_ = _⇒ q_ , then _p_ can not be true. Therefore, _p_ can be true ”only if” _q_ is also true, which leads us to our assertion.* ”’ _p_ is true ”therefore” _q_ is true.”’* ”’ _p_ is true ”entails” that _q_ is true.”’* ”’ _q_ is true ”because” _p_ is true.”’* ”’ _p_ ”may” be true ”unless” _q_ is false.”’* ””’Given that” _p_ is true, _q_ is true.”’* ”’ _q_ is true ”whenever” _p_ is true.”’* ”’ _q_ is true ”provided that” _p_ is true.”’* ”’ _q_ is true ”in case” _p_ is true.”’* ”’ _q_ is true ”assuming that” _p_ is true.”’* ”’ _q_ is true ”on the condition that” _p_ is true.”’Further colloquial interpretations can often be found in natural language whose meaning can be reduced down ”’ _p_ only if _q_ ”’, for example:* ”’ _p_ is true ”as long as” _q_ is true”’::Example::::”"Mummy, can I go to the pictures?"”:::”"”’As long as”’ you’ve done your homework. Have you done your homework? No? Then you cannot go to the pictures."”::In other words::::”"You can go to the pictures ”’only if”’ you have done your homework."”::Using the full language of logic::::”"If it is true that you are going to the pictures, it is true that you must have done your homework."”* ”’ _p_ is true ”as soon as” _q_ is true”’:::”"Are we going to this party, then?"”:::”"”’As soon as”’ I’ve finished putting on my makeup.”"::The analysis is the same as for the above example of ”’as long as”’. end{definition} 

41 

Preprint. 

## Table 32: **Example of Gemini reasoning on Sustainable Living** 

_**Query**_ How good is it to reuse water from plant pots? I’m living in an apartment, and after I water my plants the water goes to plates below the pots. The pots are in a metallic structure above the plates, so I can take the plates to reuse the water (throwing it at the plants again). This reuse seems beneficial, because I think I can get rid of mosquitoes that would reproduce in the stagnated water. And also some nutrients of the soil (as well as earthworms) can return to the vase. Is there some negative points in doing that? 

## _**LLM Reasoning steps**_ 

## Plant Pot Water Reuse 

**(1) Essential Problem:** 

The user wants to know if it is beneficial or detrimental to reuse the water collected from plant pot drainage plates. 

They are concerned about potential mosquito breeding and the impact on plant growth. 

**(2) Reasoning:** 

**Reused Water Composition:** 

- Potentially contains dissolved nutrients, minerals and salts from the soil. - May contain microorganisms. 

- Could harbor mosquito larvae if stagnant. 

**Negative Impact on Plant Growth:** 

- Stagnant water can attract mosquitoes. 

- High salt concentrations from fertilizers can harm plants. 

- Waterborne pathogens could potentially harm plants. 

**(3) Draft Answer:** 

## Reusing Water from Plant Pots: A Balancing Act 

It’s commendable that you’re looking for ways to conserve water and potentially benefit your plants. 

Reusing water from plant pot drainage plates can be a good idea, but there are some drawbacks to consider: 

* **Mosquito Breeding:** Stagnant water in the plates can become a breeding ground for mosquitoes, 

posing a health risk. Regularly empty and clean the plates to prevent this. 

* **Salt Accumulation:** Repeatedly using the same water can lead to a buildup of salts from fertilizers, which can harm plants. 

* **Pathogen Risk:** While less common, waterborne pathogens could potentially harm your plants. 

Using filtered or boiled water can minimize this risk. 

42 

Preprint. 

## Table 33: **Example of Gemini reasoning on Sustainable Living, continue** 

## _**Example positive document**_ 

## Soluble Salts 

Soluble salts may accumulate on the top of the soil, forming a yellow or white crust. A ring of salt deposits may form around the pot at the soil line or around the drainage hole. Salts may also build up on the outside of clay pots. In house plants, signs of excess soluble salts include reduced growth, brown leaf tips, dropping of lower leaves, small new growth, dead root tips, and wilting. 

Soluble salts are minerals dissolved in water. Fertilizer dissolved in water becomes a soluble salt. When water evaporates from the soil, the minerals or salts stay behind. As the salts in the soil become more and more concentrated, it becomes more difficult for plants to take up water. If salts build up to an extremely high level, water can be taken out of the root tips, causing them to die. High levels of soluble salts damage the roots directly, weakening the plant and making it more susceptible to attack from insects and diseases. One of the most common problems associated with high salt levels is root rot. 

The best way to prevent soluble salt injury is to stop the salts from building up. When watering, allow some water to drain through the container and then empty the saucer. Do not allow the pot to sit in water. If the drained water is absorbed by the soil, the salts that were washed out are reabsorbed through the drainage hole or directly through a clay pot. 

... 

Table 34: **The performance of retrieval models when using reasoning steps generated by GritLM as queries** 

||**StackExchange**|**Coding**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|
|---|---|---|---|---|
||**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Leet.**<br>**Pony**|||
||_Sparse model_||||
|BM25 (Robertson et al.,2009)|33.1<br>38.5<br>19.2<br>28.1<br>16.7<br>16.2<br>20.7|19.7<br>13.3|3.3<br>12.1<br>8.7|19.1|
||_Open-sourced models (<1B)_||||
|BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)<br>SBERT (Reimers & Gurevych,2019)|17.7<br>29.8<br>18.7<br>22.9<br>12.6<br>11.6<br>17.6<br>22.3<br>27.6<br>19.3<br>26.2<br>13.1<br>11.9<br>17.7<br>17.0<br>22.0<br>15.2<br>23.9<br>9.7<br>9.3<br>13.1|23.5<br>3.2<br>16.5<br>0.7<br>24.2<br>1.7|4.2<br>15.4<br>11.3<br>4.7<br>19.3<br>9.0<br>3.8<br>15.4<br>9.4|15.7<br>15.7<br>13.7|
||_Open-sourced models (>1B)_||||
|E5 (Wang et al.,2023a)<br>SFR (Meng et al.,2024)<br>Inst-XLSu et al.(2022)<br>GritLM (Muennighoff et al.,2024)<br>Qwen (Li et al.,2023b)|24.3<br>36.5<br>18.5<br>21.2<br>10.6<br>14.9<br>15.0<br>22.1<br>31.3<br>20.0<br>23.0<br>12.2<br>15.0<br>16.8<br>34.5<br>44.7<br>26.5<br>36.5<br>19.3<br>21.7<br>24.8<br>24.6<br>31.6<br>18.8<br>21.0<br>14.2<br>10.9<br>21.5<br>25.7<br>35.8<br>19.5<br>29.7<br>12.1<br>18.1<br>22.0|27.0<br>0.3<br>26.2<br>0.6<br>24.7<br>1.6<br>28.9<br>5.0<br>22.9<br>7.8|4.0<br>20.0<br>17.4<br>5.3<br>19.2<br>15.2<br>6.0<br>16.0<br>8.4<br>5.5<br>18.6<br>17.0<br>2.6<br>17.9<br>21.8|17.5<br>17.2<br>22.1<br>18.1<br>19.7|
||_Proprietary models_||||
|Cohere (Cohere)<br>OpenAI (OpenAI)<br>Voyage (Voyage AI)<br>Google (Lee et al.,2024)|21.8<br>26.4<br>22.2<br>21.5<br>15.1<br>15.8<br>17.1<br>26.4<br>29.8<br>21.5<br>29.4<br>11.4<br>11.6<br>21.1<br>28.4<br>29.0<br>19.9<br>28.3<br>9.6<br>17.7<br>17.2<br>25.6<br>34.4<br>22.1<br>30.0<br>14.8<br>19.3<br>16.5|21.3<br>1.9<br>22.6<br>2.6<br>29.2<br>1.7<br>28.9<br>1.6|4.7<br>14.4<br>9.9<br>7.5<br>18.3<br>11.9<br>7.2<br>23.1<br>10.6<br>8.5<br>16.9<br>12.5|16.0<br>17.8<br>18.5<br>19.3|



43 

Preprint. 

Table 35: **The performance of retrieval models when using reasoning steps generated by Geminipro as queries** 

||**StackExchange**|**Coding**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|
|---|---|---|---|---|
||**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Leet.**<br>**Pony**|||
||_Sparse model_||||
|BM25 (Robertson et al.,2009)|54.3<br>49.1<br>20.7<br>33.2<br>19.8<br>18.3<br>23.5|20.9<br>12.7|2.7<br>17.2<br>9.5|23.5|
||_Open-sourced models (<1B)_||||
|BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)<br>SBERT (Reimers & Gurevych,2019)|28.4<br>34.7<br>21.8<br>25.1<br>13.5<br>13.4<br>19.1<br>35.9<br>41.3<br>22.8<br>34.0<br>17.8<br>16.7<br>16.4<br>20.3<br>24.2<br>15.5<br>24.7<br>11.5<br>9.8<br>16.6|24.7<br>5.3<br>17.4<br>0.9<br>25.1<br>2.3|4.0<br>17.7<br>13.8<br>6.1<br>23.2<br>12.1<br>4.1<br>18.2<br>10.9|18.4<br>20.4<br>15.3|
||_Open-sourced models (>1B)_||||
|E5 (Wang et al.,2023a)<br>SFR (Meng et al.,2024)<br>Inst-XLSu et al.(2022)<br>GritLM (Muennighoff et al.,2024)<br>Qwen (Li et al.,2023b)|29.7<br>42.0<br>18.1<br>19.2<br>13.8<br>17.1<br>14.1<br>29.3<br>38.2<br>20.4<br>23.8<br>16.3<br>16.0<br>16.8<br>44.5<br>48.7<br>26.8<br>34.7<br>22.3<br>22.0<br>26.5<br>29.0<br>33.9<br>20.8<br>24.6<br>17.0<br>12.9<br>19.2<br>35.0<br>41.6<br>20.0<br>29.0<br>15.9<br>17.4<br>21.3|27.6<br>0.2<br>26.8<br>0.3<br>24.8<br>1.5<br>32.1<br>4.3<br>23.6<br>4.4|5.9<br>20.5<br>23.9<br>6.9<br>21.6<br>22.0<br>7.0<br>17.4<br>11.6<br>6.7<br>22.8<br>22.9<br>6.0<br>25.6<br>27.6|19.3<br>19.9<br>24.0<br>20.5<br>22.3|
||_Proprietary models_||||
|Cohere (Cohere)<br>OpenAI (OpenAI)<br>Voyage (Voyage AI)<br>Google (Lee et al.,2024)|30.2<br>33.2<br>22.9<br>27.1<br>16.6<br>19.3<br>21.6<br>35.1<br>38.3<br>24.7<br>33.2<br>15.1<br>14.4<br>17.6<br>37.5<br>40.0<br>22.8<br>29.7<br>13.6<br>17.9<br>19.6<br>35.6<br>38.4<br>23.6<br>33.1<br>16.6<br>22.5<br>16.2|20.7<br>3.4<br>24.3<br>4.7<br>32.3<br>2.3<br>32.3<br>2.6|5.2<br>18.2<br>15.6<br>6.7<br>21.4<br>19.3<br>5.3<br>26.7<br>17.6<br>8.0<br>23.8<br>17.4|19.5<br>21.2<br>22.1<br>22.5|



Table 36: **The performance of retrieval models when using reasoning steps generated by Llama370B as queries** 

||**StackExchange**|**Coding**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|
|---|---|---|---|---|
||**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Leet.**<br>**Pony**|||
||_Sparse model_||||
|BM25 (Robertson et al.,2009)|54.0<br>51.2<br>24.1<br>35.2<br>19.6<br>21.6<br>25.3|21.1<br>14.0|4.9<br>16.2<br>17.1|25.3|
||_Open-sourced models (<1B)_||||
|BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)<br>SBERT (Reimers & Gurevych,2019)|28.8<br>38.6<br>22.7<br>23.5<br>15.0<br>16.5<br>22.2<br>39.8<br>45.4<br>24.2<br>34.4<br>19.5<br>20.3<br>19.4<br>20.2<br>25.5<br>16.9<br>24.1<br>10.3<br>11.7<br>16.6|25.4<br>7.5<br>19.2<br>1.4<br>24.7<br>6.6|5.7<br>21.7<br>16.7<br>4.4<br>24.1<br>15.4<br>3.8<br>19.7<br>13.8|20.3<br>22.3<br>16.1|
||_Open-sourced models (>1B)_||||
|E5 (Wang et al.,2023a)<br>SFR (Meng et al.,2024)<br>Inst-XLSu et al.(2022)<br>GritLM (Muennighoff et al.,2024)<br>Qwen (Li et al.,2023b)|30.2<br>43.7<br>18.2<br>23.4<br>13.4<br>15.8<br>15.2<br>26.1<br>38.4<br>21.4<br>25.3<br>14.7<br>15.8<br>17.7<br>45.2<br>51.2<br>32.1<br>39.8<br>21.1<br>26.2<br>28.0<br>28.2<br>35.0<br>22.4<br>24.6<br>17.0<br>14.4<br>20.9<br>33.1<br>43.0<br>21.7<br>30.4<br>13.7<br>20.0<br>22.3|26.9<br>0.3<br>27.1<br>0.5<br>25.3<br>1.7<br>32.0<br>4.3<br>25.8<br>8.0|5.9<br>19.0<br>23.3<br>6.9<br>20.3<br>21.5<br>6.8<br>19.5<br>13.1<br>4.8<br>22.1<br>20.8<br>5.2<br>25.3<br>28.9|19.6<br>19.7<br>25.8<br>20.5<br>23.1|
||_Proprietary models_||||
|Cohere (Cohere)<br>OpenAI (OpenAI)<br>Voyage (Voyage AI)<br>Google (Lee et al.,2024)|34.8<br>35.1<br>26.4<br>27.6<br>17.9<br>22.4<br>24.0<br>34.1<br>41.0<br>26.1<br>35.5<br>12.3<br>14.5<br>22.4<br>37.1<br>41.1<br>23.8<br>26.9<br>13.6<br>18.8<br>24.2<br>39.5<br>44.0<br>26.4<br>37.0<br>18.7<br>25.0<br>17.5|24.5<br>4.2<br>24.3<br>5.9<br>31.9<br>1.4<br>31.1<br>3.1|6.6<br>19.7<br>19.7<br>7.7<br>20.8<br>18.9<br>6.9<br>26.0<br>21.4<br>8.8<br>21.2<br>21.4|21.9<br>22.0<br>22.8<br>24.5|



Table 37: **The performance of retrieval models when using reasoning steps generated by Claudeopus as queries** 

||**StackExchange**|**Coding**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|
|---|---|---|---|---|
||**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Leet.**<br>**Pony**|||
||_Sparse model_||||
|BM25 (Robertson et al.,2009)|54.2<br>51.7<br>23.2<br>38.3<br>22.4<br>20.3<br>25.8|20.0<br>19.8|4.0<br>18.5<br>17.6|26.3|
||_Open-sourced models (<1B)_||||
|BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)<br>SBERT (Reimers & Gurevych,2019)|30.2<br>38.0<br>23.4<br>29.3<br>14.7<br>15.0<br>22.0<br>37.8<br>40.0<br>24.9<br>34.2<br>19.6<br>17.6<br>18.9<br>19.1<br>24.4<br>18.5<br>24.9<br>11.5<br>11.0<br>14.5|22.4<br>6.0<br>17.8<br>1.5<br>23.0<br>5.7|4.5<br>24.2<br>19.4<br>4.0<br>24.5<br>18.2<br>3.1<br>19.3<br>18.5|20.7<br>21.6<br>16.1|
||_Open-sourced models (>1B)_||||
|E5 (Wang et al.,2023a)<br>SFR (Meng et al.,2024)<br>Inst-XLSu et al.(2022)<br>GritLM (Muennighoff et al.,2024)<br>Qwen (Li et al.,2023b)|31.6<br>43.3<br>20.9<br>25.8<br>13.6<br>17.4<br>14.6<br>29.9<br>38.8<br>24.3<br>28.4<br>15.6<br>16.8<br>18.8<br>45.7<br>50.0<br>29.0<br>43.0<br>22.5<br>23.0<br>27.0<br>32.4<br>36.4<br>22.5<br>28.8<br>19.5<br>15.8<br>24.1<br>35.8<br>45.1<br>22.0<br>34.4<br>15.8<br>19.5<br>26.4|29.1<br>0.3<br>28.5<br>0.6<br>23.8<br>2.0<br>31.1<br>6.3<br>24.6<br>5.7|6.8<br>23.1<br>26.4<br>7.3<br>24.3<br>24.9<br>6.6<br>21.2<br>15.4<br>5.7<br>25.0<br>26.5<br>6.2<br>27.0<br>30.9|21.1<br>21.5<br>25.8<br>22.8<br>24.5|
||_Proprietary models_||||
|Cohere (Cohere)<br>OpenAI (OpenAI)<br>Voyage (Voyage AI)<br>Google (Lee et al.,2024)|33.3<br>36.4<br>24.8<br>28.7<br>18.1<br>20.4<br>21.2<br>36.5<br>39.6<br>25.6<br>36.7<br>13.2<br>14.8<br>22.6<br>37.2<br>40.2<br>24.8<br>31.1<br>12.6<br>18.4<br>20.5<br>37.3<br>43.7<br>27.1<br>38.8<br>21.8<br>23.7<br>17.6|21.6<br>4.4<br>25.5<br>5.4<br>30.9<br>1.9<br>31.4<br>2.9|6.9<br>21.6<br>20.2<br>7.7<br>22.5<br>21.2<br>6.9<br>27.9<br>21.3<br>9.1<br>24.6<br>22.0|21.5<br>22.6<br>22.8<br>25.0|



44 

Preprint. 

Table 38: **The performance of retrieval models when using reasoning steps generated by GPT-4 as queries** 

||**StackExchange**|**Coding**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|**Theorem-based**<br>**Avg.**<br>**AoPS**<br>**TheoQ.**<br>**TheoT.**|
|---|---|---|---|---|
||**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Leet.**<br>**Pony**|||
||_Sparse model_||||
|BM25 (Robertson et al.,2009)|53.6<br>53.6<br>24.3<br>38.6<br>18.8<br>22.7<br>25.9|19.3<br>17.7|3.9<br>18.9<br>20.2|26.5|
||_Open-sourced models (<1B)_||||
|BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)<br>SBERT (Reimers & Gurevych,2019)|29.3<br>41.8<br>23.4<br>27.8<br>11.7<br>14.4<br>21.6<br>40.0<br>44.9<br>24.2<br>34.4<br>18.2<br>21.1<br>19.0<br>18.9<br>26.0<br>17.5<br>27.1<br>9.1<br>9.9<br>17.3|24.2<br>8.2<br>17.8<br>1.3<br>24.3<br>10.2|7.5<br>24.3<br>25.1<br>6.8<br>27.1<br>19.9<br>5.0<br>21.4<br>22.9|21.6<br>22.9<br>17.5|
||_Open-sourced models (>1B)_||||
|E5 (Wang et al.,2023a)<br>SFR (Meng et al.,2024)<br>Inst-XLSu et al.(2022)<br>GritLM (Muennighoff et al.,2024)<br>Qwen (Li et al.,2023b)|29.6<br>43.6<br>20.1<br>26.7<br>11.8<br>17.7<br>15.6<br>26.2<br>39.1<br>21.5<br>28.3<br>13.4<br>16.8<br>19.5<br>47.3<br>51.4<br>29.7<br>40.5<br>20.5<br>25.3<br>27.0<br>33.1<br>38.9<br>22.3<br>28.8<br>17.4<br>17.7<br>24.1<br>35.8<br>43.0<br>24.3<br>34.3<br>15.6<br>19.7<br>24.4|29.1<br>0.9<br>28.4<br>1.5<br>24.4<br>2.1<br>31.8<br>11.7<br>25.4<br>5.2|5.3<br>25.6<br>35.7<br>7.1<br>25.9<br>33.2<br>8.4<br>21.8<br>16.4<br>6.7<br>26.3<br>29.5<br>4.6<br>28.0<br>33.7|21.8<br>21.7<br>26.2<br>24.0<br>24.5|
||_Proprietary models_||||
|Cohere (Cohere)<br>OpenAI (OpenAI)<br>Voyage (Voyage AI)<br>Google (Lee et al.,2024)|31.8<br>36.6<br>25.6<br>29.3<br>16.1<br>22.0<br>23.2<br>35.5<br>40.0<br>24.8<br>38.0<br>13.7<br>17.0<br>24.2<br>36.9<br>42.7<br>24.5<br>34.3<br>13.9<br>21.2<br>22.1<br>36.5<br>45.3<br>25.6<br>38.1<br>19.1<br>25.0<br>18.0|20.8<br>7.0<br>24.4<br>6.5<br>31.4<br>2.1<br>31.1<br>3.6|7.0<br>23.0<br>25.16<br>7.7<br>22.1<br>23.5<br>6.6<br>29.0<br>27.4<br>10.1<br>27.4<br>29.6|22.3<br>23.1<br>24.4<br>25.8|



Table 39: **Long-context retrieval performance on unsplit web pages of StackExchange data.** The scores are reported in recall@1 

||Bio.<br>Earth.<br>Econ.<br>Psy.<br>Rob.<br>Stack.<br>Sus.<br>Pony|Avg.|
|---|---|---|
|Sparse models|||
|BM25<br>10.7<br>15.4<br>10.7<br>8.4<br>7.4<br>22.2<br>10.7<br>5.4<br>11.4|||
|Open-sourced models (<1B)|||
||||
|BGE<br>Inst-L<br>SBERT|16.4<br>27.7<br>20.9<br>11.6<br>10.9<br>13.2<br>16.9<br>0.4<br>24.1<br>29.5<br>14.1<br>21.3<br>12.9<br>14.5<br>25.4<br>3.9<br>25.6<br>34.1<br>18.9<br>15.8<br>10.9<br>15.0<br>18.0<br>1.2|14.8<br>18.2<br>17.4|
|Open-sourced models (>1B)|||
||||
|E5<br>SFR<br>Inst-XL<br>GritLM<br>Qwen|29.9<br>36.3<br>26.2<br>46.7<br>17.3<br>14.5<br>32.2<br>1.1<br>30.3<br>37.0<br>24.3<br>47.7<br>17.3<br>14.5<br>35.0<br>2.0<br>22.0<br>32.0<br>14.1<br>20.5<br>12.9<br>15.0<br>20.1<br>5.9<br>37.5<br>39.4<br>25.7<br>35.3<br>17.8<br>20.1<br>32.4<br>0.0<br>39.2<br>36.1<br>25.7<br>42.3<br>21.3<br>23.5<br>33.1<br>1.3|25.5<br>26.0<br>17.8<br>26.0<br>27.8|
|Proprietary models|||
||||
|Cohere<br>OpenAI<br>Voyage<br>Google|31.5<br>35.5<br>18.0<br>20.5<br>9.9<br>15.8<br>15.2<br>0.8<br>33.1<br>32.3<br>24.8<br>35.1<br>11.9<br>11.5<br>26.3<br>0.0<br>34.4<br>35.3<br>26.7<br>41.6<br>12.9<br>13.7<br>31.1<br>1.3<br>30.9<br>38.0<br>21.8<br>30.7<br>12.9<br>19.2<br>25.6<br>0.3|18.4<br>21.9<br>24.6<br>22.4|



Table 40: **Statistics of StackExchange and Pony data before web pages and documentation are split.** For each dataset, we show the number of queries and documents, the average length of queries and documents, and the average number of gold documents. 

||# Query|# Doc|Avg Query Len|Avg Doc Len|# Avg Gold Doc|
|---|---|---|---|---|---|
|Biology|103|524|115.2|9422.4|1.3|
|Earth Science|116|601|109.5|27312.3|1.6|
|Economics|103|516|181.5|11896.4|1.1|
|Psychology|101|512|149.6|12411.66|1.1|
|Robotics|101|508|818.9|14998.2|1.1|
|Stack Overfow|117|1858|478.3|40759.7|1.1|
|Sustainable Living|108|554|148.5|12077.7|1.2|
|Pony|112|577|102.6|1361.0|6.9|



45 

Preprint. 

Table 41: **MiniLM (cross-encoder), Gemini and GPT-4 reranking scores based on BM25 top-10 or top-100 retrieval results.** 

|Reranker<br>top-k|StackExchange<br>Bio.<br>Earth.<br>Econ.<br>Psy.<br>Rob.<br>Stack.<br>Sus.|Code<br>Leet.<br>Pony|Math<br>Avg.<br>AoPS<br>TheoQ.<br>TheoT.|Math<br>Avg.<br>AoPS<br>TheoQ.<br>TheoT.|
|---|---|---|---|---|
|None<br>-|19.2<br>27.1<br>14.9<br>12.5<br>13.5<br>16.5<br>15.2|24.4<br>7.9|6.2<br>9.8<br>4.8|14.3|
|MiniLM<br>10<br>100|15.4<br>26.6<br>13.0<br>11.8<br>14.3<br>15.4<br>13.6<br>8.5<br>18.9<br>6.0<br>5.4<br>7.6<br>7.9<br>8.9|21.8<br>8.7<br>15.0<br>11.3|6.1<br>6.5<br>4.2<br>6.1<br>3.6<br>0.5|13.1<br>8.3|
|Gemini<br>10|21.9<br>29.7<br>16.9<br>14.2<br>16.1<br>16.7<br>16.7|24.5<br>8.0|6.2<br>9.5<br>8.2|15.7|
|GPT-4<br>10<br>100|23.8<br>33.7<br>18.4<br>16.4<br>18.4<br>20.3<br>17.2<br>33.8<br>34.2<br>16.7<br>27.0<br>22.3<br>27.7<br>11.1|22.6<br>10.2<br>3.4<br>15.6|6.5<br>11.3<br>9.6<br>1.2<br>2.0<br>8.6|17.4<br>17.0|



Table 42: **MiniLM (cross-encoder), Gemini and GPT-4 reranking scores based on Google retrieval top-10 or top-100 results.** 

|Reranker<br>top-k|StackExchange<br>Bio.<br>Earth.<br>Econ.<br>Psy.<br>Rob.<br>Stack.<br>Sus.|Code<br>Leet.<br>Pony|Math<br>Avg.<br>AoPS<br>TheoQ.<br>TheoT.|Math<br>Avg.<br>AoPS<br>TheoQ.<br>TheoT.|
|---|---|---|---|---|
|None<br>-|23.0<br>34.4<br>19.5<br>27.9<br>16.0<br>17.9<br>17.3|29.6<br>3.6|9.3<br>21.5<br>14.3|19.5|
|MiniLM<br>10<br>100|17.0<br>30.6<br>15.8<br>20.3<br>12.3<br>15.0<br>14.6<br>7.5<br>21.7<br>6.4<br>6.2<br>7.0<br>7.1<br>8.3|24.0<br>6.0<br>16.0<br>17.2|9.8<br>14.2<br>11.9<br>8.1<br>4.2<br>2.9|16.0<br>9.4|
|Gemini<br>10|23.8<br>35.8<br>19.6<br>29.0<br>16.4<br>17.2<br>18.6|29.1<br>5.0|9.4<br>20.8<br>16.3|20.1|
|GPT-4<br>10<br>100|26.1<br>36.5<br>20.9<br>32.6<br>16.8<br>22.6<br>20.8<br>42.5<br>40.9<br>25.9<br>42.1<br>23.2<br>35.1<br>17.2|24.5<br>5.5<br>5.6<br>10.8|8.9<br>22.9<br>19.8<br>2.4<br>6.6<br>19.3|21.5<br>22.6|



Table 43: **Gemini prompt to rerank documents.** 

## Gemini 

The following passages are related to the query: {query} [1]. {doc 1} [2]. {doc 2} ... 

First identify the essential problem in the query. Think step by step to reason about why each document is relevant or irrelevant. Rank these passages based on their relevance to the query. Please output the ranking result of the most relevant {k} passages as a list, where the first element is the id of the most relevant passage, the second element is the id of the second most element, etc. Please strictly follow the format to output a list of {k} ids: _′′′_ [ _..._ ] _′′′_ 

46 

Preprint. 

Table 44: **Results of retrieval models copied from MTEB Muennighoff et al. (2023) for easier reference.** Argu. refers to ArguAna, Climate. refers to ClimateFEVER, CQA. refers to CQADupstackRetrieval, FIQA. refers to FIQA2018, Hot. refers to HotpotQA, MS. refers to MSMARCO, NF. refers to NFCorpus, Quora refers to QuoraRetrieval, SCI. refers to SCIDOCS, Sci. refers to SciFact, Touche. refers to Touche2020, TREC. refers to TRECCOVID. Except BM25, whose results are from Thakur et al. (2021), all other results are from Muennighoff et al. (2023). 

||**Argu.**<br>**Climate.**<br>**CQA.**<br>**DBPedia**<br>**FEVER**<br>**FIQA.**<br>**Hot.**<br>**MS.**<br>**NF.**<br>**NQ**<br>**Quora.**<br>**SCI.**<br>**Sci.**<br>**Touche.**<br>**TREC.**<br>Avg.|
|---|---|
|_Sparse model_||
|BM25 (Robertson et al.,2009)|31.5<br>21.3<br>29.9<br>31.3<br>75.3<br>23.6<br>60.3<br>22.8<br>32.5<br>32.9<br>78.9<br>15.8<br>66.5<br>36.7<br>65.6<br>41.6|
|_Open-sourced models (<1B)_||
|BGE (Xiao et al.,2023)<br>Inst-LSu et al.(2022)<br>SBERT (Reimers & Gurevych,2019)|63.5<br>36.6<br>42.2<br>44.1<br>87.2<br>45.0<br>74.1<br>42.6<br>38.1<br>55.0<br>89.1<br>22.6<br>74.6<br>24.8<br>74.8<br>54.3<br>57.1<br>27.7<br>43.8<br>36.7<br>72.7<br>45.5<br>55.2<br>39.7<br>34.1<br>50.1<br>88.4<br>18.6<br>64.3<br>21.6<br>58.1<br>47.6<br>46.5<br>22.0<br>45.0<br>32.1<br>50.9<br>50.0<br>39.3<br>39.8<br>33.3<br>50.5<br>87.5<br>23.8<br>65.6<br>19.9<br>51.3<br>43.8|
|_Open-sourced models (>1B)_||
|E5 (Wang et al.,2023a)<br>SFR (Meng et al.,2024)<br>Inst-XLSu et al.(2022)<br>GritLM (Muennighoff et al.,2024)<br>Qwen (Li et al.,2023b)|61.9<br>38.4<br>43.0<br>48.9<br>87.8<br>56.6<br>75.7<br>43.1<br>38.6<br>63.5<br>89.6<br>16.3<br>76.4<br>26.4<br>87.3<br>56.9<br>67.2<br>36.4<br>46.5<br>49.1<br>89.4<br>60.4<br>77.0<br>43.4<br>41.9<br>69.9<br>89.8<br>19.9<br>77.7<br>29.0<br>87.6<br>59.0<br>55.7<br>26.5<br>43.1<br>40.2<br>70.0<br>47.0<br>55.9<br>41.6<br>36.0<br>57.2<br>88.9<br>17.4<br>64.6<br>23.4<br>71.4<br>49.3<br>63.2<br>30.9<br>49.4<br>46.6<br>82.7<br>60.0<br>79.4<br>42.0<br>40.9<br>70.3<br>89.5<br>24.4<br>79.2<br>27.9<br>74.8<br>57.4<br>62.7<br>44.0<br>40.6<br>48.0<br>93.4<br>55.3<br>72.3<br>41.7<br>38.3<br>61.8<br>89.6<br>27.7<br>75.3<br>20.3<br>72.7<br>56.2|
|_Proprietary models_||
|Cohere (Cohere)<br>Voyage (Voyage AI)<br>OpenAI (OpenAI)<br>Google (Lee et al.,2024)|61.5<br>38.4<br>41.5<br>43.4<br>89.0<br>42.2<br>70.7<br>42.9<br>38.6<br>61.6<br>88.7<br>20.3<br>71.8<br>32.4<br>81.9<br>55.0<br>64.1<br>32.7<br>46.6<br>46.0<br>91.5<br>59.8<br>70.9<br>40.6<br>40.3<br>65.9<br>87.4<br>24.3<br>80.0<br>39.2<br>85.1<br>58.3<br>58.1<br>30.3<br>47.5<br>44.8<br>87.9<br>55.0<br>71.6<br>40.2<br>42.1<br>61.3<br>89.1<br>23.1<br>77.8<br>23.4<br>79.6<br>55.4<br>62.2<br>33.2<br>48.9<br>47.1<br>87.0<br>59.2<br>71.3<br>32.6<br>40.3<br>61.3<br>88.2<br>20.3<br>75.4<br>25.9<br>82.6<br>55.7|



Table 45: QA inference prompt 

Problem: {problem_description} Documents: {retrieved_doc} 

Based on the provided documents, write an answer to the problem. 

47 

Preprint. 

Table 46: QA evaluation prompt 

- ———- PROBLEM START ———{predicted_answer} ———- PROBLEM END ——————- STUDENT ANSWER START ———{predicted_answer} ———- STUDENT ANSWER END ——————- REFERENCE ANSWER START ———{gold_answer} ———- REFERENCE ANSWER END ———Criteria: 

- 0 - The student’s answer is completely irrelevant or blank. 

- 10 - The student’s answer addresses about 10% of the reference content. 

- 20 - The student’s answer addresses about 20% of the reference content. 

- 30 - The student’s answer addresses about 30% of the reference content. 

- 40 - The student’s answer addresses about 40% of the reference content. 50 - The student’s answer addresses about 50% of the reference content. 60 - The student’s answer addresses about 60% of the reference content. 70 - The student’s answer addresses about 70% of the reference content. 80 - The student’s answer addresses about 80% of the reference content. 90 - The student’s answer addresses about 90% of the reference content. 100 - The student’s answer addresses about 100% of the reference content. Use the following format to give a score: REASON: Describe why you give a specific score SCORE: The score you give, e.g., 60 Do not say anything after the score. 

Table 47: **Question answering results with different retrievers.** We use the GPT-4 as the generation models, and evaluate the answers with Claude-3.5-sonnet. Results show that stronger retrieval typically result in better QA results, indicating the helpfulness of the annotated documents for addressing the posts in StackExchange. 

|**Retriever**|**Bio.**<br>**Earth.**<br>**Econ.**<br>**Psy.**<br>**Rob.**<br>**Stack.**<br>**Sus.**|**Average**|
|---|---|---|
|None|62.6<br>64.9<br>57.8<br>58.2<br>53.5<br>64.4<br>49.8|58.7|
|BM25<br>SBERT<br>Qwen|64.3<br>65.2<br>57.8<br>54.6<br>60.2<br>65.9<br>56.5<br>62.2<br>63.2<br>59.7<br>65.6<br>55.7<br>59.9<br>**61.0**<br>**69.4**<br>**68.3**<br>**62.0**<br>**66.3**<br>**64.2**<br>**69.5**<br>55.5|60.6<br>61.0<br>**65.0**|
|Oracle|_80.0_<br>_81.2_<br>_73.6_<br>_77.8_<br>_70.5_<br>_80.4_<br>_70.6_|_76.3_|



|**Model**|**Setting**|**TheoQ.**|**TheoT.**|**AoPS**|
|---|---|---|---|---|
||None|76_._31_._6|82_._13_._0|36_._62_._6|
|GPT-4o|Random<br>Qwen|76_._51_._3<br>76_._42_._2|85_._01_._8<br>82_._62_._2|37_._42_._6<br>36_._11_._9|
||Oracle|79_._30_._4|89_._71_._7|37_._23_._8|



Table 48: Downstream results of gpt-4o-2024-08-06 on TheoremQA Questions, TheoremQA Theorems, and AoPS. The results are averaged across 5 random seeds and we report the standard deviation in the subscript. 

48 

