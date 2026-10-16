# **DOCMATH-EVAL: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and Specialized Documents** 

**Yilun Zhao** _[∗]_[1] **Yitao Long** _[∗]_[2] **Hongjun Liu**[2] **Ryo Kamoi**[3] **Linyong Nan**[1] **Lyuhao Chen**[4] **Yixin Liu**[1] **Xiangru Tang**[1] **Rui Zhang**[3] **Arman Cohan**[1] _[,]_[5] 

1Yale University 2New York University 3Penn State University 4Carnegie Mellon University 5Allen Institute for AI 

� https://github.com/yale-nlp/DocMath-Eval 

- https://docmath-eval.github.io/ 

## **Abstract** 

Recent LLMs have demonstrated remarkable performance in solving exam-like math word problems. However, the degree to which these numerical reasoning skills are effective in real-world scenarios, particularly in expert domains, is still largely unexplored. This paper introduces DOCMATH-EVAL, a comprehensive benchmark specifically designed to evaluate the numerical reasoning capabilities of LLMs in the context of understanding and analyzing specialized documents containing both text and tables. We conduct an extensive evaluation of 48 LLMs using Chainof-Thought and Program-of-Thought prompting techniques, aiming to comprehensively assess the capabilities and limitations of existing LLMs in DOCMATH-EVAL. We found that even the current best-performing system ( _i.e.,_ GPT-4o) still significantly lags behind human experts in solving complex numerical reasoning problems grounded in long contexts. We believe that DOCMATH-EVAL can serve as a valuable benchmark for evaluating LLMs’ capabilities in solving challenging numerical reasoning problems within expert domains. 

## **1 Introduction** 

Recent advancements in large language models (LLMs) have attracted significant attention due to their capabilities in solving a broad range of tasks (OpenAI, 2023; AI@Meta, 2024), including math word problems (MWPs) commonly found in academic exams (Wang et al., 2017; Miao et al., 2020; Amini et al., 2019; Cobbe et al., 2021; Hendrycks et al., 2021; Cobbe et al., 2021; Lu et al., 2023; Chen et al., 2023b). These MWPs vary from basic arithmetic to advanced algebra, showcasing LLMs’ proficiency in numerical reasoning — a crucial skill for interpreting and manipulating numerical data across various contexts. Despite 

- _∗_ Equal Contributions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents_images/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents.pdf-0001-10.png)


**----- Start of picture text -----**<br>
What is the rate of increase in debt<br>from 2021 to 2022?<br>….<br>Model Output with  Chain-of-Thought Prompting:<br>First, we know from the table that  the values of debt in 2021<br>and 2022 are xxx and xxx, respectively. We then calculate the<br>(...abbreviate…)<br>Therefore, the final answer is 17.3%<br>Model Output with  Program-of-Thought Prompting:<br>def solution():<br>debt_2021 = 125<br>    debt_2022 = 278<br>(…abbreviate)<br>return answer<br>**----- End of picture text -----**<br>


Figure 1: The overview of DOCMATH-EVAL and the prompting methods explored. DOCMATH-EVAL evaluates the LLMs’ performance in the context of understanding and analyzing financial documents containing both text and tables. The models are required to first locate question-relevant data points within lengthy documents, and then apply numerical reasoning and specialized financial knowledge to answer the question. 

this progress, there is still a significant gap in understanding the practicality of LLMs’ numerical reasoning in real-world scenarios, particularly in specialized fields such as finance, medicine, and science. As illustrated in Figure 1, these expert domains necessitate LLMs to interpret complex, domain-specific documents, applying numerical 

16103 

_Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 16103–16115 August 11-16, 2024 ©2024 Association for Computational Linguistics 

reasoning to complex problem-solving (Chen et al., 2021; Zhu et al., 2021; Zhao et al., 2022; Li et al., 2022b). Recognizing this gap, our research focuses on the finance domain (Li et al., 2022a; Wu et al., 2023a; Yang et al., 2023; Callanan et al., 2023; Xie et al., 2024). The finance industry often deals with lengthy and data-intensive documents that demand advanced numerical reasoning skills for accurate analysis and decision-making. 

We introduce **DOCMATH-EVAL** , a comprehensive and standardized benchmark that systematically evaluates the numerical reasoning capabilities of LLMs in understanding and interpreting specialized documents containing both textual and tabular data. DOCMATH-EVAL encompasses four evaluation sets, each with varying levels of difficulty in _numerical reasoning_ and _document understanding_ . Specifically, We construct a new evaluation set, **DMCompLong** , from scratch, to examine the LLM’s capabilities in performing _complex_ numerical reasoning over _extreme long_ documents containing _multiple_ tables. We also adapt and re-annotate four existing finance QA benchmarks to develop three additional, less challenging evaluation sets: 1) **DMSimpShort** based on TAT-QA (Zhu et al., 2021) and FinQA (Chen et al., 2021), necessitates _simple_ numerical reasoning over _short_ document with _one_ table; 2) **DMSimpLong** based on MultiHiertt (Zhao et al., 2022), necessitates _simple_ numerical reasoning over _long_ document with _multiple_ tables; and 3) **DMCompShort** based on TAT-HQA (Li et al., 2022b), necessitates _complex_ numerical reasoning over _short_ document with _one_ table. 

We conduct an extensive evaluation on DOCMATH-EVAL, covering a total of 48 proprietary and open-source LLMs from 17 organizations. Two prompting methods, Chain-of-Thought (CoT) (Wei et al., 2022) and Program-of-Thought (PoT) (Chen et al., 2023a), are applied for result analysis. Our experimental results indicate that while the existing best-performing LLM on average ( _i.e.,_ GPT-4o) can achieve high performance in simple settings ( _e.g.,_ DMSimpShort), it still falls short of human experts in more challenging ones, _i.e.,_ , DMCompLong. Moreover, Claude-3.5-Sonnet outperforms other LLMs, achieving an accuracy of 40.0% on the DMCompLong set when applying CoT prompting. However, it still lags far behind human expert performance, which stands at 76%. This significant gap between LLMs and human experts underscores the challenges presented by DOCMATH-EVAL. It underscores the importance 

of advancing LLMs’ numerical reasoning and document understanding abilities to effectively apply them in the real-world specialized domains. We conclude our main contributions as follows: 

- We introduce DOCMATH-EVAL, a comprehensive benchmark designed to systematically evaluate LLMs’ numerical reasoning ability to understand and interpret long and specialized documents. This includes a newly developed, challenging evaluation set and three adapted evaluation sets for varying difficulty levels. 

- We conduct an extensive evaluation encompassing a wide range of LLMs, including those specialized in math and coding. We also incorporate different prompting methods ( _i.e.,_ CoT and PoT) to comprehensively assess the capabilities and limitations of existing LLMs in our task. 

- Our experimental results reveal a noticeable performance gap compared to human experts in more complex scenarios ( _i.e.,_ problems requiring complex numerical reasoning over long documents). This highlights the limitations of current LLMs in complex real-world applications and the need for continued advancements. 

## **2 Related Work** 

**Math Word Problems** The research community has shown significant interest in the vital role of numerical reasoning skills in LLMs. These skills are vital for models to effectively engage in complex problem-solving. To this end, a wide variety of MWP datasets have been proposed in recent years (Hosseini et al., 2014; Koncel-Kedziorski et al., 2016; Wang et al., 2017; Ling et al., 2017; Cobbe et al., 2021). More challenging datasets have recently been introduced to enhance diversity (Miao et al., 2020), difficulty (Chen et al., 2023b; Hendrycks et al., 2021), and adversarial robustness (Patel et al., 2021). However, existing MWP datasets predominantly focus on problems akin to academic exams, with a limited emphasis on real-world scenarios. Addressing this gap, our paper introduces a novel and comprehensive benchmark designed to evaluate LLMs’ abilities in understanding and interpreting long and specialized documents through numerical reasoning. 

**Numerical Reasoning over Documents** Numerical reasoning over documents requires models to 

16104 

|**Property**(Median/Avg)|**DMSimpShort**|**DMSimpLong**|**DMCompShort **|**DMCompLong (new)**|
|---|---|---|---|---|
|Data Source|TAT-QA (Zhu et al.,2021)<br>FinQA (Chen et al.,2021)|MultiHiertt<br>TAT-HQA<br> (Zhao et al.,2022) (Li et al.,2022b)||expert annotated<br>from scratch|
|Question Length|19 / 20.0|21 / 21.6|29 / 30.1|34 / 37.7|
|# Sentences in Text|14 / 16.9|64 / 66.9|6 / 7.8|535 / 752.3|
|# Words in Text|504 / 506.6|2,216 / 2,334.0|251 / 314.2|25,149 / 34,589.0|
|# Table|1 / 1.0|4 / 3.9|1 / 1.0|46 / 72.5|
|# Rows per Table|6 / 7.0|9 / 11.6|7 / 8.2|3 / 7.5|
|# Columns per Table|5 / 4.7|4 / 4.5|5 / 5.0|3 / 3.1|
|# Text Evidence|0 / 0.4|1 / 0.9|0 / 0.4|1 / 1.0|
|# Table Evidence|1 / 0.9|1 / 1.1|1 / 1.0|1 / 1.0|
|% Questions_w._Table Evidence|92.9%|86.4%|97.8%|76.3%|
|# Math Operations in Python Solution|2 / 2.1|2 / 2.3|2 / 2.3|4 / 4.9|
|# Code Lines in Python Solution|5 / 5.3|6 / 5.9|5 / 5.3|8 / 8.2|
|# Comment Lines in Python Solution|2 / 2.0|2 / 2.0|2 / 2.0|2 / 3.4|
|Development set|200|100|200|300|
|Test set|800|400|800|1,200|
|Total Size|1,000|500|1,000|1,500|



Table 1: Basic statistics of DOCMATH-EVAL dataset. Our newly constructed evaluation set, DMCompLong, poses unique challenges in both numerical reasoning and financial document understanding. 

have a deep understanding of context and the ability to derive answers through numerical reasoning (Dua et al., 2019). Applying these models in the finance domain (Xie et al., 2023; Wu et al., 2023a; Yang et al., 2023) presents additional challenges in terms of interpreting hybrid data (Zhu et al., 2021) and utilizing domain-specific expertise (Chen et al., 2021; Zhao et al., 2024). Numerous datasets focusing on numerical reasoning over specialized documents have been proposed recently. Two notable benchmarks are TAT-QA (Zhu et al., 2021) and FinQA (Chen et al., 2021), which represent pioneering efforts in studying numerical reasoning in finance, particularly requiring the fusion of tabular and textual content. Building upon TAT-QA, a more challenging dataset named TAT-HQA (Li et al., 2022b) was developed, focusing on counterfactual questions in relation to the provided context. Additionally, MultiHiertt (Zhao et al., 2022) focuses on numerical reasoning over longer financial documents containing multiple tables. However, as illustrated in Table 1, these four datasets focus on less challenging scenarios, where either simple numerical reasoning ( _e.g.,_ calculating the increasing rate or average value) is sufficient, or the input context is short. Furthermore, there is a lack of a standardized benchmark for systematically evaluating models’ performance across varying difficulty levels in terms of numerical reasoning and document understanding. 

## **3 DOCMATH-EVAL** 

In this section, we first offer a formal definition of the DOCMATH-EVAL task. We then explain the rationale and methodology for adopting Python program as the standardized solution format for DOCMATH-EVAL. Subsequently, we detail the data annotation process used to construct the challenging DMCompLong evaluation set, as well as the data re-annotation process for compiling the other three evaluation sets. Table 7 in the Appendix presents the profiles of the seven annotators involved. Finally, we present human-level performance on each evaluation set in DOCMATH-EVAL. 

## **3.1 Task Formulation** 

We formally define the task of DOCMATH-EVAL in the context of LLMs as follows: Presented with a numerical reasoning question _q_ and a financial document consisting of textual contents _E_ and structured tables _T_ , the task is to generate the numericvalue answer _a_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents_images/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents.pdf-0003-07.png)


To obtain the best candidate answer _a_ ˆ, we use greedy decoding in all our LLM evaluations. 

## **3.2 Solution Format Standardization** 

We observe that existing finance QA datasets feature solutions in various formats. Specifically, TATQA (Zhu et al., 2021) and TAT-HQA (Li et al., 

16105 

2022b) utilize text, while MultiHiertt (Zhao et al., 2022) employs mathematical expressions, such as 100/3, and FinQA (Chen et al., 2021) uses math programs, such as divide(100,3), for solution annotations. This diversity in annotation formats hinders the development of a unified evaluation framework to assess LLM performance across different benchmarks. Additionally, text-based solutions often fall short in precision and clarity, making them less suitable for computational problemsolving; and the solutions presented as mathematical equations or programs can be less descriptive, with the intended semantic meaning of the equations sometimes being unclear. 

To overcome the aforementioned limitations, in DOCMATH-EVAL, we represent solutions using Python programs (Zhao et al., 2024). Such a unified Python program format supports a standardized and effective evaluation framework for LLM assessment. Specifically, annotators are instructed to initially define variables at the start of the Python function, beginning with “def solution():”. These variables should align with the primary elements or quantities referenced in the question or relevant content in the documents. They then write a Python program that methodically address the problem, solving it step by step. Additionally, annotators receive a bonus for writing detailed comments, thereby enhancing the code’s readability and understandability. To verify the correctness and performance of the solutions, our annotation interface automatically runs the Python function. This process checks that the output is either a float or int and ensures that the execution finishes without any errors. 

## **3.3 Data Re-Annotation From Public Datasets** 

We re-annotate four existing datasets and incorporate them into DOCMATH-EVAL. Specifically, we re-annotate TAT-QA (Zhu et al., 2021) and FinQA (Chen et al., 2021) for DMSimpShort, MultiHiertt (Zhao et al., 2022) for DMSimpLong, and TAT-HQA (Li et al., 2022b) for DMCompShort. 

**Question Validation and Re-annotation** We instruct the annotators to identify and remove questions with incorrect annotations or those whose answers are not numerical. Annotators are then asked to enhance each question by adding a scale descriptor to ensure clarity and specificity. For example, _"Question: What is the average payment volume per transaction for American Express?_ _**(in**_ 

_**billions)** "_ . They were also asked to correct any identified errors in the original questions. 

**Solution Validation and Re-annotation** As outlined in Section 3.2, we require annotators to rewrite the original solutions into a unified Python format, standardizing variable names and adding comments to enhance the readability of the solutions. Regarding the supporting evidence annotation, we initially convert the original evidence annotations to our format. We then highlight these evidences in the annotation interface, and direct annotators to verify their correctness. 

## **3.4 Data Annotation From Scratch** 

In real-world scenarios, financial professionals typically need to handle documents spanning tens of pages, along with problems that require more complex numerical reasoning combined with financial knowledge. However, as previously discussed, existing benchmarks (Zhu et al., 2021; Chen et al., 2021; Zhao et al., 2022; Li et al., 2022b) focus on less challenging scenarios, where either simple numerical reasoning is sufficient, or the input context is short. To bridge this gap, we have developed a new, challenging evaluation set, **DMCompLong** , from scratch. This set focuses on settings that more closely align with real-world scenarios, where models are required to perform complex numerical reasoning over long financial documents for problem solving. The annotation process is as follows: 

**Source Document Collection** Following previous work (Zhu et al., 2021; Chen et al., 2021; Zhao et al., 2022), we use the quarterly (i.e., Form 10-Q) and annual reports (i.e., Form 10-K) of companies as our source documents, which are publicly available at the open-source database[1] of U.S. Securities and Exchange Commission. After collecting all the source documents, we utilize a commercial API[2] to extract their textual and tabular content. Subsequently, we apply a heuristic-based method to preprocess these two formats of content. The preprocessed documents are then passed to expert annotators for question annotation. 

**Data Annotation** Given a financial document, annotators are first required to briefly read its content and determine the data points to be used in the question. They must then compose the question and highlight the selected paragraphs or ta- 

1https://www.sec.gov/edgar/search/ 

> 2https://sec-api.io/ 

16106 

bles as evidence supporting it. Finally, the annotators are required to write down the solution to the question in Python program format, as discussed in Section 3.2. We set up a _bonus payment system_ for complex annotations that involve difficult document comprehension and numerical reasoning. Specifically, to increase the difficulty of document understanding, we award bonuses to annotators for questions that necessitate information from: 1) multiple tables, 2) multiple sections, or 3) a combination of tables and textual content. To enhance the challenge in numerical reasoning, we provide bonuses for questions requiring financial expertise or involving complex mathematical operations. If such annotations are validated during the quality validation stage, a bonus payment will be added. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents_images/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents.pdf-0005-01.png)


Figure 2: Example of _zero_ -shot CoT prompt used. 

**Quality Validation** We implement a comprehensive quality validation protocol to ensure that each annotated example meets the required standards. For every question annotation, we assign it to another annotator, recognized for their high performance in annotation, to verify its accuracy. This process involves manually locating the question-relevant evidence in the documents using our retrieval-based search toolkits. They then compare this evidence with the original annotations and correct any errors found. Additionally, validators are tasked with confirming the accuracy of the annotated solutions. We offer bonus payments to annotators for identifying erroneous annotations. Ultimately, 232 of the annotated questions are flagged as erroneous and are subsequently revised. Table 6 in the Appendix presents the human evaluation scores and inter-evaluator agreements for a subset of 200 sampled examples. DOCMATHEVAL exhibits superior annotation quality and a high degree of inter-annotator agreement. 

## **3.5 Expert-level Performance Evaluation** 

To give a general yet insightful estimate of the performance on each of the DOCMATH-EVAL sets, we enlisted two professionals who hold Chartered Financial Analyst licenses to conduct the evaluation. Regarding human expert performance on DMSimpShort and DMSimpLong, we report the same results as those in the original papers, with accuracy of 91% and 87%, respectively. For DMCompShort and DMCompLong, We randomly sample 25 examples from each set, asking the expert evaluators to answer the questions individually within a fourhour period. They achieve accuracy of 88% and 

80% on DMCompShort (average 84%); and accuracy of 72% and 80% on DMCompLong (average 76%). 

## **3.6 Dataset Release** 

Table 1 presents the data statistics of four developed evaluation sets. DOCMATH-EVAL contains a total of 4,000 questions with high-quality annotations, featuring varying difficulty levels in numerical reasoning and document understanding. We randomly partitioned the dataset into two subsets: _testmini_ and _test_ . The _testmini_ subset includes 800 examples and is intended for model development and validation. The _test_ subset consists of the remaining 3,200 examples, which are reserved for standard evaluation. To avoid data contamination (Deng et al., 2024), the features directly related to the ground truth for the _test_ set are kept private. Instead, we have developed and manage an online evaluation platform, where researchers can assess models and participate in a leaderboard. 

## **4 Experiment Setup** 

This section discusses the experiment setup, including the evaluated LLMs, prompting methods, and our implementation details. 

## **4.1 Evaluated Large Language Models** 

Our goal is to investigate the capabilities of current state-of-the-art LLMs on DOCMATH-EVAL to better understand their strengths and limitations. To this end, we evaluate a wide range of models, including 32 general-purpose LLMs, 4 math-specific LLMs, 6 code-based LLMs, and 7 mixture of ex- 

16107 

perts (MoE) models. The specific details of each evaluated LLM, including the exact version used, can be found in Table 8 in the Appendix. 

## **4.2 Prompting Methods** 

Following recent works on LLM reasoning benchmarks (Lu et al., 2024; Chen et al., 2023b), we evaluate two commonly used prompting methods for math reasoning: 

**Chain-of-Thought** The CoT method (Wei et al., 2022) instructs the LLMs to explicitly outline their reasoning process step by step before arriving at the final answer. Figure 2 presents the CoT prompt used in our experiment. 

**Program-of-Thought** The PoT method (Chen et al., 2023a) separates computation from the reasoning process by instructing the LLMs to produce a structured program that encapsulates the reasoning steps. The final answer is obtained by executing the generated program. Figure 3 in Appendix presents the PoT prompt we used. 

of evaluated LLMs. Therefore, in our main experiments with DMCompLong, we evaluate models using the retrieval-augmented generation (RAG) setting. In this setting, external retrievers are employed to extract the top- _n_ most relevant textual and tabular evidence from the source document. We maintain the original relative order of the evidence and input it into the LLMs to answer the given question. We experiment with commonly-used sparse retriever, _i.e.,_ BM25 (Robertson et al., 1995), and three dense retrievers, including OpenAI Embedding 3 small & large versions (Neelakantan et al., 2022) and Contriever (Izacard et al., 2022). 

**Final Answer Extraction** For LLMs using CoT prompting, we adopt the answer extraction process from Chen et al. (2023b) and Lu et al. (2024) to extract the final answer from the model’s output. For LLMs employing PoT prompting, we first develop a heuristic method to extract the generated python solution from the model response. We then execute it to obtain the final answer. 

## **5 Results and Analysis** 

## **4.3 Implementation Details** 

**LLM Experiment** The experiments involving open-sourced LLMs were conducted using the vLLM framework (Kwon et al., 2023). In all the experiments, we used a temperature setting of 1.0 and maximum output length of 512. Given the extensive context length of input document, the main evaluation of DOCMATH-EVAL is conducted under a _zero-shot_ setting, aiming to assess LLMs’ capabilities to generate accurate answers without few-shot demonstrations or additional training. 

**Input Tabular Data Serialization** Building on previous work that evaluated LLMs on tablerelevant tasks (Chen, 2023; Zhao et al., 2023a,b), we present our method for processing tabular data in documents. Specifically, we separate headers or cells in different columns using a vertical bar (|), and rows using a newline. This approach allows for the direct feeding of flattened table input into LLMs. In our preliminary study, we found that most LLMs can comprehend these table formats well. Nevertheless, we believe that future research could explore more effective methods for encoding tabular data (Fang et al., 2024). 

**RAG-based Setting for DMCompLong** For the DMCompLong subset, the input document length is extremely long and exceeds the context length limit 

We next discuss our main findings from the experiments and our analysis of the DMCompLong subset. 

## **5.1 Main Results** 

Table 2 and Table 9 in the Appendix present the LLM performance on the DOCMATH-EVAL testmini and test sets, respectively. 

While the current best-performing LLM, GPT4o, achieves performance comparable to human experts in simple problem settings ( _i.e.,_ DMSimpShort and DMCompShort), we find significant performance gaps in more challenging settings. Specifically, GPT-4o achieves an accuracy of 41.0% on DMCompLong with PoT, which is far behind the human expert performance of 76.0%. This underscores the need for ongoing LLM development, particularly in complex problem-solving over long and specialized documents. Most open-source LLMs still lag behind the proprietary LLMs. However, the two DeepSeek-V2-* models come close to matching the performance of the leading proprietary models. The DeepSeek-V2 even outperforms GPT-4o on the DMCompLong subset. This suggests that opensource LLMs have the potential to bridge the performance gap with the leading proprietary models in the near future. 

The code-specific and proprietary LLMs generally perform as well as or better with PoT prompt- 

16108 

|**Model**<br>**Size**<br>**Notes**|**DMSimpShort**<br>PoT<br>CoT|**DMCompShort**<br>PoT<br>CoT|**DMSimpLong**<br>PoT<br>CoT|**DMCompLong**<br>PoT<br>CoT|**Avg. Acc**<br>PoT<br>CoT|
|---|---|---|---|---|---|
|Human Expert|91.0|87.0|84.0|76.0||
|GPT-4o<br>GPT-4-Turbo<br>Claude-3-Opus<br>Claude-3.5-Sonnet<br>Claude-3-Sonnet<br>Gemini-1.5-Flash<br>Gemini-1.5-Pro<br>Claude-3-Haiku<br>GPT-4o-Mini<br>GPT-3.5-Turbo|**_Proprietary LLMs_**<br>84.0<br>86.0<br>69.5<br>76.5<br>85.5<br>82.5<br>80.0<br>81.0<br>80.5<br>79.5<br>73.5<br>77.5<br>78.0<br>77.0<br>76.0<br>69.5<br>82.5<br>80.0<br>80.5<br>73.0<br>85.0<br>78.0<br>78.5<br>69.5<br>85.5<br>80.5<br>80.0<br>58.0<br>74.5<br>79.0<br>71.5<br>58.5<br>88.5<br>69.5<br>77.0<br>69.5<br>71.0<br>60.5<br>52.5<br>39.0||56.0<br>64.0<br>56.0<br>53.0<br>51.0<br>61.0<br>54.0<br>61.0<br>55.0<br>56.0<br>55.0<br>46.0<br>58.0<br>55.0<br>55.0<br>50.0<br>53.0<br>56.0<br>41.0<br>28.0|41.0<br>36.7<br>38.7<br>38.3<br>42.0<br>39.7<br>44.0<br>40.0<br>40.3<br>35.3<br>40.0<br>31.7<br>40.3<br>30.0<br>36.7<br>31.7<br>38.7<br>28.0<br>28.7<br>15.0|60.8<br>62.4|
||||||62.9<br>61.9|
||||||60.6<br>61.8<br>61.8<br>59.2<br>62.7<br>58.5<br>62.8<br>54.5<br>63.7<br>52.8<br>57.1<br>52.5<br>62.5<br>52.2<br>46.8<br>34.0|
|DeepSeek-V2<br>236B<br>MoE<br>Mistral-Large<br>123B<br>DeepSeek-Coder-V2<br>236B<br>Code, MoE<br>Llama-3.1<br>70B<br>Qwen2<br>72B<br>Llama-3<br>70B<br>Mixtral-8x22B<br>141B<br>MoE<br>Gemma-2<br>9B<br>DeepSeek-Coder-V2-Lite<br>16B<br>Code<br>WizardLM-2<br>141B<br>MoE<br>C4AI Command R+<br>104B<br>Yi-1.5<br>9B<br>Yi-1.5<br>34B<br>Mistral-Nemo<br>12B<br>Llama-3.1<br>8B<br>DBRX<br>132B<br>MoE<br>Codestral<br>22B<br>Code<br>Llama-3<br>8B<br>Qwen2<br>7B<br>Mathstral<br>7B<br>Math<br>GLM-4<br>9B<br>Aya-23<br>35B<br>DeepSeek-V2-Lite<br>16B<br>MoE<br>Mixtral-8x7B-v0.1<br>46B<br>MoE<br>DeepSeek-Math<br>7B<br>Math<br>Llama-2<br>70B<br>WizardLM-2<br>7B<br>Mistral-v0.3<br>7B<br>WizardMath<br>7B<br>Math<br>InternLM2-Math-Plus<br>7B<br>Math<br>StarCoder2<br>15B<br>Code<br>InternLM2<br>7B<br>Gemma-1<br>7B<br>Llama-2<br>7B<br>DeepSeek-Coder-V1<br>33B<br>Code<br>WizardCoder<br>33B<br>Code<br>Aya-23<br>8B<br>Gemma-1<br>2B|**_Open-source LLMs_**<br>87.0<br>82.0<br>75.5<br>69.5<br>85.0<br>83.5<br>76.5<br>81.0<br>85.0<br>79.0<br>78.0<br>66.5<br>74.5<br>76.5<br>68.0<br>71.0<br>26.5<br>74.0<br>24.5<br>72.5<br>84.5<br>73.5<br>64.0<br>63.5<br>30.0<br>74.0<br>21.5<br>57.0<br>79.0<br>66.5<br>65.0<br>54.5<br>66.0<br>67.5<br>51.0<br>53.5<br>62.5<br>60.5<br>56.5<br>55.5<br>35.5<br>65.5<br>39.0<br>51.0<br>18.0<br>68.5<br>24.5<br>56.0<br>0.5<br>64.5<br>1.0<br>53.0<br>52.5<br>59.5<br>37.5<br>44.0<br>62.0<br>60.0<br>44.0<br>42.5<br>41.0<br>57.0<br>29.5<br>43.0<br>39.0<br>51.5<br>38.5<br>41.5<br>49.5<br>56.5<br>21.5<br>31.0<br>13.0<br>56.0<br>9.5<br>33.0<br>43.5<br>55.0<br>32.5<br>35.0<br>69.5<br>44.0<br>53.5<br>34.0<br>1.5<br>44.0<br>1.0<br>25.5<br>7.0<br>45.5<br>3.5<br>18.0<br>0.5<br>39.0<br>2.0<br>17.0<br>2.0<br>46.0<br>1.0<br>27.0<br>32.5<br>43.5<br>16.5<br>25.0<br>47.0<br>42.0<br>30.5<br>28.5<br>49.5<br>40.0<br>40.5<br>28.0<br>22.5<br>32.0<br>12.0<br>22.5<br>28.5<br>27.5<br>15.0<br>14.0<br>47.5<br>21.0<br>34.0<br>15.5<br>18.0<br>20.0<br>4.5<br>11.0<br>1.0<br>20.0<br>0.0<br>7.5<br>4.0<br>17.0<br>4.0<br>11.5<br>19.0<br>18.5<br>8.5<br>8.5<br>32.5<br>16.0<br>17.5<br>8.0<br>1.0<br>13.0<br>0.0<br>9.0<br>4.0<br>8.0<br>1.5<br>7.5||61.0<br>56.0<br>56.0<br>55.0<br>56.0<br>54.0<br>53.0<br>50.0<br>8.0<br>45.0<br>52.0<br>42.0<br>25.0<br>47.0<br>50.0<br>39.0<br>27.0<br>30.0<br>25.0<br>34.0<br>19.0<br>31.0<br>2.0<br>14.0<br>0.0<br>14.0<br>28.0<br>37.0<br>32.0<br>33.0<br>32.0<br>30.0<br>18.0<br>23.0<br>24.0<br>29.0<br>4.0<br>31.0<br>10.0<br>23.0<br>33.0<br>20.0<br>0.0<br>20.0<br>1.0<br>17.0<br>0.0<br>25.0<br>1.0<br>4.0<br>1.0<br>8.0<br>5.0<br>6.0<br>25.0<br>9.0<br>6.0<br>7.0<br>7.0<br>9.0<br>11.0<br>6.0<br>9.0<br>10.0<br>0.0<br>7.0<br>0.0<br>2.0<br>2.0<br>2.0<br>5.0<br>2.0<br>0.0<br>2.0<br>0.0<br>2.0|43.0<br>39.7<br>41.0<br>31.3<br>41.0<br>37.7<br>34.7<br>29.3<br>7.0<br>27.0<br>41.0<br>28.3<br>14.7<br>24.0<br>24.3<br>17.7<br>22.0<br>20.3<br>17.7<br>18.0<br>8.7<br>18.3<br>4.0<br>14.0<br>0.0<br>15.3<br>15.3<br>16.7<br>19.0<br>14.3<br>12.0<br>16.3<br>17.3<br>13.0<br>10.0<br>12.3<br>2.3<br>10.0<br>11.3<br>11.7<br>17.7<br>8.7<br>0.0<br>11.7<br>1.0<br>10.3<br>0.0<br>12.7<br>0.3<br>8.0<br>2.0<br>7.0<br>7.3<br>5.7<br>11.3<br>5.7<br>3.7<br>3.3<br>4.7<br>4.0<br>8.3<br>4.3<br>2.7<br>2.3<br>0.0<br>3.3<br>1.3<br>2.7<br>3.7<br>1.7<br>5.0<br>1.0<br>0.3<br>2.3<br>0.0<br>0.0||
||||||64.4<br>59.8|
||||||62.8<br>59.7|
||||||63.1<br>57.3|
||||||55.3<br>54.1<br>16.4<br>52.4<br>59.0<br>50.1<br>21.5<br>47.6<br>51.4<br>41.8<br>40.9<br>41.6<br>39.5<br>40.0<br>24.3<br>39.9<br>12.4<br>38.1<br>0.4<br>36.9<br>31.7<br>36.8<br>37.6<br>35.1<br>26.1<br>34.9<br>28.1<br>31.0<br>24.5<br>30.1<br>7.0<br>29.9<br>24.5<br>29.8<br>41.5<br>25.3<br>0.6<br>24.3<br>3.1<br>21.9<br>0.6<br>21.9<br>1.0<br>21.8<br>13.1<br>20.8<br>22.7<br>20.5<br>29.9<br>20.3<br>10.8<br>15.7<br>13.5<br>13.0<br>24.9<br>11.5<br>7.8<br>9.9<br>0.2<br>9.0<br>2.5<br>8.4<br>8.5<br>7.6<br>15.0<br>6.6<br>0.4<br>6.6<br>1.4<br>4.1|



Table 2: LLM performance on the _testmini_ set of DOCMATH-EVAL. We utilize the average accuracy achieved through CoT prompting as the metric for ranking model performance. For DMCompLong, we use the OpenAI Embedding 3 Large retriever to retrieve top-10 evidence as input document. Numbers underlined indicate that models using PoT prompting outperform those using CoT prompting. 

ing compared to CoT prompting. This is likely because LLMs are prone to making errors during complex mathematical computations, as revealed in concurrent work (Zhao et al., 2024). Additionally, for math-specific LLMs, InternLM2-MathPlus outperforms its base model in CoT performance, with average accuracy rising from 9.9% to 13.0%. This highlights the impact of instructiontuning in improving math reasoning abilities. 

## **5.2 Analysis on DMCompLong Set** 

We next conduct a detailed analysis of the RAG setting, long-context LLMs, and model failure cases. 

**RAG Analysis** We analyze the impact of retriever performance on the final accuracy of RAGbased LLM systems by selecting the Llama-3-70B and GPT-4o models for our study. As demonstrated in Table 3, the OpenAI Embedding-3 significantly 

16109 

|top-_n_|Retriever|R@_n_|Llama-3|GPT-4o|
|---|---|---|---|---|
||Contriever|22.3|13.7|16.0|
|3|BM25<br>Embedding-3-Small|29.5<br>44.7|13.7<br>19.0|15.7<br>24.0|
|5|Embedding-3-Large<br>Contriever<br>BM25<br>Embedding-3-Small|48.2<br>32.0<br>38.0<br>57.1|22.0<br>15.3<br>15.3<br>21.0|27.0<br>22.0<br>20.7<br>29.0|
|10|Embedding-3-Large<br>Contriever<br>BM25<br>Embedding-3-Small|62.0<br>45.3<br>47.9<br>71.2|24.0<br>18.3<br>20.3<br>25.3|32.7<br>25.7<br>23.0<br>31.7|
|–|Embedding-3-Large<br>Oracle|75.8|26.3<br>35.3|36.7<br>42.0|



Table 3: Results of the Llama-3-70B and GPT-4o with CoT prompting approaches under various retrieval settings on the DMCompLong testmini set. A correlation is observed between LLM performance and the questionrelevance of the retrieved evidence. 

|Model|RAG|Long Context|
|---|---|---|
|GPT-4o|36.7|**40.3**|
|Gemini-1.5-Pro<br>Claude-3-Sonnet<br>Gemini-1.5-Flash|30.0<br>**35.3**<br>31.7|**37.3**<br>34.7<br>**34.3**|
|Claude-3-Haiku|**31.7**|31.0|
|DeepSeek-V2<br>DeepSeek-Coder-V2|**39.7**<br>**37.7**|38.7<br>36.0|
|Llama-3.1-70B|**29.3**|26.3|
|Llama-3.1-8B|**14.3**|9.0|
|Mistral-Nemo|**16.7**|4.7|
|Phi-3-Medium|12.7|**13.0**|
|GLM-4-9B|8.7|**9.7**|



Table 4: Results of the CoT prompting approach under various retrieval settings on DMCompLong testmini set. 

outperforms Contriever and BM25. Additionally, improved retriever performance consistently boosts the final accuracy of the models in our task. These results highlight the need for future work to develop more advanced information retrieval techniques for enhancing complex problem-solving over long and specialized documents. 

**Long-Context LLM Analysis** In addition to using RAG for analyzing long specialized documents, recent advancements have extended the input length of LLMs to handle lengthy documents (Su et al., 2023). We compare models with a context length limit of over 100K under both the RAG (as used in the main results) and LongContext settings, where the entire document is input. As illustrated in Table 4, the evaluated models 

||**Error Type**||**Explanation**|
|---|---|---|---|
||Inaccurate|Evi-|The challenge lies in fnding accu-|
||dence Retrieval||rate evidence, especially in situa-|
||(39 / 100)||tions where the values needed for|
||||intermediate reasoning steps are|
||||not explicitly stated. This makes it|
||||diffcult for the retriever to identify|
||||the correct evidence.|
||Calculation Error||The reasoning process is accurate,|
||(28 / 100)||but there are errors in the interme-|
||||diate or fnal computations.|
||Table<br>Misunder-||The model faces challenges in|
||standing||comprehending and parsing cell|
||(16 / 100)||values, particularly in complex ta-|
||||bles.|
||Exceeding Context||The input document exceeds the|
||Length (8 / 100)||context length limit.|
||Others|||



Table 5: Error types and explanations of GPT-3.5-turbo failure cases on the DMCompLong testmini set. 

generally achieve close performance under RAG and long-context settings. This indicates that models with extended context lengths can effectively process lengthy inputs without a significant drop in performance compared to the RAG setting. 

**Error Analysis** To better understand the strengths and weaknesses of LLMs, we conduct an extensive error analysis. This analysis focuses on 100 randomly selected examples from the DM testmini set where GPT-3.5-turbo CompLong failed. We identify four common types of errors in current LLMs: inaccurate evidence retrieval, calculation errors, table misunderstandings, and exceeding context length. A detailed explanation for each type is provided in Table 5. 

## **6 Conclusion** 

This paper introduces DOCMATH-EVAL, a comprehensive benchmark designed to evaluate the capabilities of LLMs in numerical reasoning over long and specialized documents. Our experiments show that even the best-performing current models still fall short of human expert performance on problems requiring complex reasoning over extended contexts. This highlights the need for future research to improve LLMs’ proficiency in complex numerical reasoning tasks within expert domains. 

16110 

## **Limitations** 

There are some limitations in our study that we believe can be addressed in future work. First, our approach to extracting the final answer from the model’s output is not yet flawless. In certain instances, this method fails to accurately identify the answer, causing the reported accuracy to be an approximate lower limit. Additionally, we suggest that future research could investigate training large language models (LLMs) on finance-specific data to improve their performance on the DOCMATHEVAL benchmark (Wu et al., 2023b; Luukkonen et al., 2023; Xie et al., 2023). 

## **Acknowledgement** 

We are grateful for the compute support provided by Microsoft Research’s Accelerate Foundation Models Research (AFMR) program. We extend our gratitude to the anonymous reviewers and area chairs for their valuable discussions and feedback. 

## **References** 

AI@Meta. 2024. Llama 3 model card. 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 2357–2367, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Ethan Callanan, Amarachi Mbakwe, Antony Papadimitriou, Yulong Pei, Mathieu Sibue, Xiaodan Zhu, Zhiqiang Ma, Xiaomo Liu, and Sameena Shah. 2023. Can gpt models be financial analysts? an evaluation of chatgpt and gpt-4 on mock cfa exams. 

- Wenhu Chen. 2023. Large language models are few(1)shot table reasoners. In _Findings of the Association for Computational Linguistics: EACL 2023_ , pages 1120–1130, Dubrovnik, Croatia. Association for Computational Linguistics. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023a. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _Transactions on Machine Learning Research_ . 

- Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. 2023b. TheoremQA: A theorem-driven question answering dataset. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language_ 

_Processing_ , pages 7889–7901, Singapore. Association for Computational Linguistics. 

- Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A dataset of numerical reasoning over financial data. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 3697–3711, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ . 

- Chunyuan Deng, Yilun Zhao, Yuzhao Heng, Yitong Li, Jiannan Cao, Xiangru Tang, and Arman Cohan. 2024. Unveiling the spectrum of data contamination in language models: A survey from detection to remediation. 

- Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Xi Fang, Weijie Xu, Fiona Anting Tan, Ziqing Hu, Jiani Zhang, Yanjun Qi, Srinivasan H. Sengamedu, and Christos Faloutsos. 2024. Large language models (LLMs) on tabular data: Prediction, generation, and understanding - a survey. _Transactions on Machine Learning Research_ . 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In _Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ . 

- Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. 2014. Learning to solve arithmetic word problems with verb categorization. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 523–533, Doha, Qatar. Association for Computational Linguistics. 

- Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. _Transactions on Machine Learning Research_ . 

- Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. MAWPS: 

16111 

A math word problem repository. In _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 1152–1157, San Diego, California. Association for Computational Linguistics. 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_ . 

- Chenying Li, Wenbo Ye, and Yilun Zhao. 2022a. FinMath: Injecting a tree-structured solver for question answering over financial reports. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ , pages 6147–6152, Marseille, France. European Language Resources Association. 

- Moxin Li, Fuli Feng, Hanwang Zhang, Xiangnan He, Fengbin Zhu, and Tat-Seng Chua. 2022b. Learning to imagine: Integrating counterfactual thinking in neural discrete reasoning. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 57–69, Dublin, Ireland. Association for Computational Linguistics. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 158–167, Vancouver, Canada. Association for Computational Linguistics. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2024. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _The Twelfth International Conference on Learning Representations_ . 

- Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2023. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In _The Eleventh International Conference on Learning Representations_ . 

- Risto Luukkonen, Ville Komulainen, Jouni Luoma, Anni Eskelinen, Jenna Kanerva, Hanna-Mari Kupari, Filip Ginter, Veronika Laippala, Niklas Muennighoff, Aleksandra Piktus, Thomas Wang, Nouamane Tazi, Teven Scao, Thomas Wolf, Osma Suominen, Samuli Sairanen, Mikko Merioksa, Jyrki Heinonen, Aija Vahtola, Samuel Antao, and Sampo Pyysalo. 2023. FinGPT: Large generative models for a small language. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 2710–2726, Singapore. Association for Computational Linguistics. 

- Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing English math word problem solvers. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 975–984, Online. Association for Computational Linguistics. 

- Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. 2022. Text and code embeddings by contrastive pre-training. 

- OpenAI. 2023. Gpt-4 technical report. _ArXiv_ , abs/2303.08774. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2080–2094, Online. Association for Computational Linguistics. 

- Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. 1995. Okapi at trec-3. _Nist Special Publication Sp_ , 109:109. 

- Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2023. Roformer: Enhanced transformer with rotary position embedding. 

- Yan Wang, Xiaojiang Liu, and Shuming Shi. 2017. Deep neural solver for math word problems. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 845–854, Copenhagen, Denmark. Association for Computational Linguistics. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ . 

- Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023a. Bloomberggpt: A large language model for finance. 

- Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023b. Bloomberggpt: A large language model for finance. _ArXiv_ , abs/2303.17564. 

- Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, 

16112 

Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, Zhiyang Deng, Yuechen Jiang, Zhiyuan Yao, Haohang Li, Yangyang Yu, Gang Hu, Jiajia Huang, Xiao-Yang Liu, Alejandro Lopez-Lira, Benyou Wang, Yanzhao Lai, Hao Wang, Min Peng, Sophia Ananiadou, and Jimin Huang. 2024. Finben: A holistic financial benchmark for large language models. 

- Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. PIXIU: A comprehensive benchmark, instruction dataset and large language model for finance. In _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ . 

- Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang. 2023. Fingpt: Open-source financial large language models. _FinLLM Symposium at IJCAI 2023_ . 

- Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. 2022. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6588–6600, Dublin, Ireland. Association for Computational Linguistics. 

- Yilun Zhao, Hongjun Liu, Yitao Long, Rui Zhang, Chen Zhao, and Arman Cohan. 2024. Financemath: Knowledge-intensive math reasoning in finance domains. 

- Yilun Zhao, Zhenting Qi, Linyong Nan, Boyu Mi, Yixin Liu, Weijin Zou, Simeng Han, Ruizhe Chen, Xiangru Tang, Yumo Xu, Dragomir Radev, and Arman Cohan. 2023a. QTSumm: Query-focused summarization over tabular data. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 1157–1172, Singapore. Association for Computational Linguistics. 

- Yilun Zhao, Haowei Zhang, Shengyun Si, Linyong Nan, Xiangru Tang, and Arman Cohan. 2023b. Investigating table-to-text generation capabilities of large language models in real-world information seeking scenarios. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track_ , pages 160–175, Singapore. Association for Computational Linguistics. 

- Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and TatSeng Chua. 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 3277–3287, Online. Association for Computational Linguistics. 

## **A Appendix** 

|**Annotation Quality**|**%S**_≥_**4**|
|---|---|
|Question Fluency|97.4|
|Question Correctness|96.0|
|Evidence Relevance|88.5|
|Evidence Completeness|91.3|
|Final Answer Correctness|97.9|
|Python Solution Correctness|97.6|
|Variable Value Correctness|98.5|
|Python Solution Conciseness|89.1|
|Variable Name Meaningfulness|95.4|



Table 6: Human evaluation was conducted on 200 samples from DOCMATH-EVAL, with three internal reviewers asked to rate each sample on a scale from 1 to 5. We present the percentage of samples that received an average score of 4 or higher, as an indicator of the annotation quality of DOCMATH-EVAL. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents_images/acl_2024_DocMath-Eval_Evaluating_Math_Reasoning_Capabilities_of_LLMs_in_Understanding_Financial_Documents.pdf-0011-11.png)


Figure 3: Example of _zero_ -shot PoT prompt used. 

16113 

|**Annotator ID**|**Finance Industry Experience**|**Annotation Sets**|
|---|---|---|
|1|1 working and 1 internship at US|New subset, Annotation validation|
|2|>= 2 internship at US|New subset, Annotation validation|
|3|1 working at Singapore and 2 internship at US|New subset|
|4|2 working and >= 1 internship at US|New subset|
|5|1 internship at US, 2 internship at China|Re-annotation on three subsets, Annotation validation|
|6|Graduate student majored in computer science|Re-annotation on three subsets, Annotation validation|
|7|Graduate student majored in statistics|Re-annotation on three subsets|



Table 7: Details of annotators involved in dataset construction. 

|Organization|Model|Size|Notes|Source|
|---|---|---|---|---|
||GPT-4-Turbo|–||gpt-4o-2024-05-13|
|OpenAI|GPT-4o|–||gpt-4-turbo-2024-04-09|
||GPT-3.5-Turbo|–||gpt-3.5-turbo-0125|
||Claude-3.5-Sonnet|–||claude-3-5-sonnet-20240620|
|Anthropic|Claude-3-Opus<br>Claude-3-Sonnet|–<br>–||claude-3-opus-20240229<br>claude-3-sonnet-20240229|
||Claude-3-Haiku|–||claude-3-haiku-20240307|
|Google|Gemini-1.5-Pro<br>Gemini-1.5-Flash|–<br>–||gemini-1.5-pro<br>gemini-1.5-flash|
|Alibaba|Qwen2|7 & 72B||Qwen/Qwen2-*B-Instruct|
||Llama-2|7 & 70B||meta-llama/Llama-2-*b-chat-hf|
|Meta|Llama-3|8 & 70B||meta-llama/Meta-Llama-3-*B-Instruct|
||Llama-3.1|8 & 70B & 405B||meta-llama/Meta-Llama-3.1-*B-Instruct|
|Google|Gemma-1<br>Gemma-2|2 & 7B<br>9B||google/gemma-b-it<br>google/gemma-2-9b-it|
||Mistral-v0.3|7B||mistralai/Mistral-7B-Instruct-v0.3|
||Mistral-Nemo|12B||mistralai/Mistral-Nemo-Instruct-2407|
|Mistral AI|Mistral-Large|123B||mistralai/Mistral-Large-Instruct-2407|
||Mathstral|7B|Math-Specifc|mistralai/Mathstral-7B-v0.1|
||Mixtral|46 & 141B|MoE|mistralai/Mixtral-Instruct-v0.1|
||Codestral|22B|Code-Specifc|mistralai/Codestral-22B-v0.1|
||DeepSeek-Math|7B|Math-Specifc|deepseek-ai/deepseek-math-7b-instruct|
|DeepSeek|DeepSeek-Coder-V1<br>DeepSeek-V2|33B<br>16 & 236B|Code-Specifc<br>MoE|deepseek-ai/deepseek-coder-33b-instruct<br>deepseek-ai/DeepSeek-V2-*-Chat|
||DeepSeek-Coder-V2|16 & 236B|Code-Specifc, MoE|deepseek-ai/DeepSeek-Coder-V2-*-Instruct|
|01 AI|Yi-1.5|9 & 34B||01-ai/Yi-1.5-34B-Chat|
|Microsoft|Phi-3-Medium|14B||microsoft/Phi-3-medium-4k-instruct|
||Phi-3-Mini|3B||microsoft/Phi-3-mini-4k-instruct|
|THUDM|GLM-4|9B||THUDM/glm-4-9b-chat|
|Databricks|DBRX|132B|MoE|databricks/dbrx-instruct|
|Cohere|C4AI Command R+|104B||CohereForAI/c4ai-command-r-plus|
||Aya-23|8 & 35B||CohereForAI/aya-23-*B|
|InternLM|InternLM2|7B||internlm/internlm2-chat-7b|
||InternLM2-Math-Plus|7B|Math-Specifc|internlm/internlm2-math-plus-7b|
||WizardLM-2|7B||lucyknada/microsoft_WizardLM-2-7B|
|WizardLM Team|WizardMath|7B|Math-Specifc|WizardLMTeam/WizardMath-7B-V1.1|
||WizardCoder|33B|Code-Specifc|WizardLMTeam/WizardCoder-33B-V1.1|
||WizardLM-2 (MoE)|141B|MoE|alpindale/WizardLM-2-8x22B|
|BigCode|StarCoder2|15B|Code-Specifc|bigcode/starcoder2-15b-instruct-v0.1|



Table 8: Details of the LLMs evaluated in this study. 

16114 

|**Model**<br>**Size**<br>**Notes**|**DMSimpShort**<br>PoT<br>CoT|**DMCompShort**<br>PoT<br>CoT|**DMSimpLong**<br>PoT<br>CoT|**DMCompLong**<br>PoT<br>CoT|**Avg. Acc**<br>PoT<br>CoT|
|---|---|---|---|---|---|
|Human Expert|91.0|87.0|84.0|76.0||
|GPT-4-Turbo<br>GPT-4o<br>Claude-3.5-Sonnet<br>GPT-4o-Mini<br>Gemini-1.5-Pro<br>Claude-3-Haiku<br>Gemini-1.5-Flash<br>GPT-3.5-Turbo|**_Proprietary LLMs_**<br>88.9<br>86.2<br>78.6<br>77.8<br>87.0<br>86.4<br>69.6<br>75.9<br>87.2<br>81.8<br>71.0<br>70.0<br>88.8<br>76.0<br>77.2<br>72.4<br>87.9<br>82.5<br>80.4<br>58.4<br>77.8<br>82.2<br>72.4<br>61.5<br>87.9<br>80.5<br>79.9<br>67.0<br>75.6<br>64.0<br>47.6<br>34.2||31.5<br>61.2<br>62.2<br>66.2<br>65.0<br>61.5<br>56.8<br>50.2<br>63.0<br>58.0<br>55.0<br>53.2<br>60.5<br>53.8<br>46.2<br>38.8|20.2<br>35.6<br>41.5<br>31.7<br>40.8<br>31.5<br>36.9<br>29.2<br>41.8<br>31.2<br>34.6<br>30.8<br>38.9<br>28.1<br>23.2<br>13.2|53.4<br>62.0<br>62.5<br>60.7|
||||||63.0<br>57.4|
||||||62.4<br>54.3<br>65.6<br>54.2<br>57.4<br>54.2<br>64.1<br>54.1<br>45.3<br>34.3|
|Mistral-Large<br>123B<br>DeepSeek-V2<br>236B<br>MoE<br>DeepSeek-Coder-V2<br>236B<br>Code, MoE<br>Llama-3.1<br>70B<br>Qwen2<br>72B<br>Llama-3<br>70B<br>Mixtral-8x22B<br>141B<br>MoE<br>DeepSeek-Coder-V2-Lite<br>16B<br>Code<br>Gemma-2<br>9B<br>Yi-1.5<br>34B<br>C4AI Command R+<br>104B<br>WizardLM-2<br>141B<br>MoE<br>DBRX<br>132B<br>MoE<br>Mistral-Nemo<br>12B<br>Yi-1.5<br>9B<br>Llama-3.1<br>8B<br>Codestral<br>22B<br>Code<br>Llama-3<br>8B<br>Mathstral<br>7B<br>Math<br>Qwen2<br>7B<br>GLM-4<br>9B<br>Aya-23<br>35B<br>Mixtral-8x7B-v0.1<br>46B<br>MoE<br>DeepSeek-Math<br>7B<br>Math<br>Mistral-v0.3<br>7B<br>DeepSeek-V2-Lite<br>16B<br>MoE<br>Llama-2<br>70B<br>WizardLM-2<br>7B<br>WizardMath<br>7B<br>Math<br>InternLM2-Math-Plus<br>7B<br>Math<br>StarCoder2<br>15B<br>Code<br>InternLM2<br>7B<br>Gemma-1<br>7B<br>Llama-2<br>7B<br>WizardCoder<br>33B<br>Code<br>Aya-23<br>8B<br>DeepSeek-Coder-V1<br>33B<br>Code<br>Gemma-1<br>2B|**_Open-source LLMs_**<br>87.9<br>85.0<br>74.4<br>79.8<br>88.9<br>86.1<br>76.9<br>66.6<br>88.2<br>80.9<br>74.9<br>67.2<br>81.0<br>80.1<br>65.8<br>70.8<br>27.8<br>77.5<br>25.0<br>70.1<br>86.2<br>80.9<br>64.8<br>62.0<br>27.4<br>74.9<br>23.1<br>59.4<br>68.8<br>71.4<br>52.2<br>44.9<br>82.0<br>70.1<br>62.4<br>49.2<br>1.0<br>71.2<br>0.9<br>58.2<br>37.6<br>67.5<br>36.5<br>50.6<br>62.1<br>59.4<br>49.5<br>47.6<br>46.8<br>64.0<br>33.1<br>41.5<br>51.1<br>66.0<br>32.1<br>40.5<br>23.2<br>69.5<br>17.4<br>45.0<br>66.5<br>63.0<br>41.9<br>34.1<br>43.1<br>58.0<br>36.8<br>37.2<br>51.6<br>57.8<br>22.8<br>31.2<br>45.8<br>54.4<br>31.1<br>34.0<br>15.4<br>52.6<br>6.6<br>33.2<br>68.0<br>48.8<br>46.5<br>32.1<br>0.9<br>46.9<br>0.5<br>26.5<br>1.1<br>42.5<br>0.4<br>19.6<br>1.4<br>47.1<br>0.4<br>28.0<br>48.6<br>40.8<br>28.5<br>24.8<br>7.6<br>49.1<br>4.6<br>16.9<br>27.1<br>45.1<br>16.8<br>26.0<br>47.0<br>42.6<br>31.2<br>31.2<br>22.1<br>34.2<br>14.4<br>24.8<br>30.0<br>30.2<br>15.0<br>16.2<br>51.0<br>29.8<br>32.0<br>16.8<br>17.4<br>24.5<br>9.4<br>11.8<br>0.5<br>23.2<br>0.2<br>6.0<br>5.6<br>20.4<br>2.4<br>11.8<br>38.4<br>19.9<br>17.5<br>8.4<br>0.5<br>14.2<br>0.1<br>8.5<br>19.4<br>16.2<br>8.5<br>6.6<br>5.9<br>9.1<br>2.5<br>6.0||58.8<br>58.8<br>58.2<br>57.2<br>56.8<br>54.8<br>49.2<br>53.8<br>16.8<br>49.0<br>51.5<br>45.0<br>21.5<br>46.8<br>28.8<br>35.0<br>45.5<br>33.2<br>0.2<br>17.8<br>19.2<br>35.0<br>31.2<br>36.0<br>26.5<br>29.8<br>29.0<br>37.0<br>1.2<br>15.0<br>35.5<br>35.5<br>25.0<br>29.5<br>22.5<br>25.0<br>11.0<br>24.0<br>4.2<br>29.0<br>32.5<br>22.8<br>0.0<br>19.5<br>0.2<br>24.2<br>1.2<br>12.5<br>19.5<br>18.0<br>2.5<br>15.2<br>2.0<br>8.0<br>9.2<br>8.5<br>5.8<br>6.2<br>10.8<br>11.2<br>9.5<br>5.2<br>9.0<br>6.8<br>0.0<br>7.8<br>1.0<br>3.2<br>7.2<br>3.2<br>0.0<br>2.8<br>3.2<br>2.2<br>0.8<br>2.8|37.2<br>30.6<br>40.2<br>32.0<br>37.5<br>31.5<br>32.1<br>26.6<br>5.7<br>25.1<br>35.0<br>26.3<br>14.8<br>22.3<br>18.0<br>19.8<br>24.4<br>17.7<br>0.2<br>13.7<br>7.6<br>15.1<br>19.0<br>16.1<br>10.7<br>18.4<br>15.7<br>15.0<br>2.2<br>11.9<br>14.7<br>13.8<br>14.2<br>14.0<br>9.3<br>11.5<br>9.1<br>11.7<br>2.6<br>11.4<br>18.2<br>11.1<br>0.5<br>10.1<br>0.2<br>13.0<br>0.5<br>7.9<br>12.8<br>7.6<br>1.0<br>8.3<br>1.1<br>6.9<br>7.2<br>4.8<br>3.4<br>4.6<br>4.2<br>3.3<br>8.4<br>3.0<br>2.9<br>4.3<br>0.2<br>3.2<br>0.4<br>2.2<br>6.1<br>1.8<br>0.0<br>2.2<br>3.6<br>1.2<br>0.2<br>1.0||
||||||61.8<br>60.0|
||||||63.8<br>57.3|
||||||61.9<br>55.7|
||||||54.9<br>54.4<br>17.4<br>52.4<br>57.3<br>51.2<br>20.9<br>47.8<br>40.6<br>40.9<br>50.9<br>40.6<br>0.6<br>39.7<br>23.8<br>39.6<br>38.9<br>37.3<br>27.3<br>37.0<br>30.3<br>36.9<br>11.1<br>35.0<br>37.0<br>33.9<br>28.4<br>32.8<br>24.9<br>29.7<br>24.0<br>29.5<br>7.0<br>29.4<br>39.5<br>27.2<br>0.5<br>24.6<br>0.5<br>23.4<br>0.8<br>23.3<br>26.5<br>21.5<br>3.7<br>21.5<br>11.6<br>21.4<br>23.4<br>21.3<br>11.1<br>17.2<br>14.2<br>14.3<br>25.1<br>13.4<br>8.9<br>11.5<br>0.3<br>9.5<br>2.3<br>9.3<br>17.2<br>8.2<br>0.2<br>6.8<br>8.7<br>6.4<br>2.3<br>4.5|



Table 9: Results of Chain-of-Thought and Program-of-Thought prompting on the _test_ set of DOCMATH-EVAL. We use average Accuracy using CoT prompting as the ranking indicator of model performance. For DMCompLong, we use the OpenAI Embedding 3 Large retriever to retrieve top-10 evidence as input document. Numbers underscored indicate that models with PoT prompting achieves better results than with CoT prompting. 

16115 

