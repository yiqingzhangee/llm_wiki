Published as a conference paper at ICLR 2025 

# - UGMATHBENCH: A DIVERSE AND DYNAMIC BENCH MARK FOR UNDERGRADUATE-LEVEL MATHEMATICAL REASONING WITH LARGE LANGUAGE MODELS 

Xin Xu _[∗]_ , Jiaxin Zhang _[∗]_ , Tianhao Chen _[∗]_ , Zitong Chao, Jishan Hu, Can Yang 

Department of Mathematics The Hong Kong University of Science and Technology 

{xxuca, jzhangat, tchenbb, zchaoaa}@connect.ust.hk 

{majhu, macyang}@ust.hk 

## ABSTRACT 

Large Language Models (LLMs) have made significant strides in mathematical reasoning, underscoring the need for a comprehensive and fair evaluation of their capabilities. However, existing benchmarks often fall short, either lacking extensive coverage of undergraduate-level mathematical problems or probably suffering from test-set contamination. To address these issues, we introduce UGMathBench, a diverse and dynamic benchmark specifically designed for evaluating undergraduatelevel mathematical reasoning with LLMs. UGMathBench comprises 5,062 problems across 16 subjects and 111 topics, featuring 10 distinct answer types. Each problem includes three randomized versions, with additional versions planned for release as leading open-source LLMs become saturated in UGMathBench. Furthermore, we propose two key metrics: effective accuracy (EAcc), which measures the percentage of correctly solved problems across all three versions, and reasoning gap (∆), which assesses reasoning robustness by calculating the difference between the average accuracy across all versions and EAcc. Our extensive evaluation of 23 leading LLMs reveals that the highest EAcc achieved is 56.3% by OpenAIo1-mini, with large ∆ values observed across different models. This highlights the need for future research aimed at developing "large reasoning models" with high EAcc and ∆= 0. We anticipate that the release of UGMathBench, along with its detailed evaluation codes, will serve as a valuable resource to advance the development of LLMs in solving mathematical problems. Codes and data are available at https://github.com/YangLabHKUST/UGMathBench. 

## 1 INTRODUCTION 

Mathematical reasoning and problem-solving are critical components of human intelligence, and the ability of machines to understand and address mathematical challenges is crucial for their deployment (Ahn et al., 2024; Liu et al., 2024; He et al., 2024a). Solving mathematical problems with machines has been a significant research topic in natural language processing since the 1960s (Bobrow et al., 1964), initially focusing on elementary math word problems (Patel et al., 2021; Wang et al., 2017; Ling et al., 2017; Welbl et al., 2017; Cobbe et al., 2021). With the advent of Large Language Models (LLMs) (Brown et al., 2020; Ouyang et al., 2022; OpenAI, 2023; Team et al., 2023; Anthropic, 2024), interest in using these advanced technologies to solve math problems has continued to grow. Researchers are exploring various approaches to improve the mathematical reasoning capabilities of LLMs, including prompting (Wei et al., 2022; Wang et al., 2023b; Kojima et al., 2022; Zhang et al., 2023; Zheng et al., 2023; Xue et al., 2024), supervised fine-tuning (Yue et al., 2023; Yu et al., 2023; Gou et al., 2023; Li et al., 2024a; Xu et al., 2024b; Tong et al., 2024; Yan et al., 2024; Xu et al., 2025b; Yan et al., 2025), and continued pretraining (Lewkowycz et al., 2022; Shao et al., 2024; 

* Equal contribution. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Undergraduate-Level Mathematics  Evaluating True Reasoning<br>16 Subjects 111 Topics 583 Sub-Topics i-th example𝑒𝑖 𝑞𝑖1 𝑞𝑖2 𝑞𝑖3 EAcc<br>Answer Types 8 Atomic Types Diverse Randomized Versionsmore versions  𝐴𝑐𝑐𝑣, 𝑣= 1,2,3 AAcc<br>2 Compound Types in the future Reasoning Gap 𝝙<br>Distribution across different  Data Example<br>subjects and levels Subject : single-variable calculus<br>Topic : application of differentiation<br>Dynamic KP : Linear approximation and differentials<br>Keywords : Calculus, Derivatives, Linear<br>Approximation, Trigonometric Functions<br>Problem_v1 : Find the linear approximation at $x=0$ to<br>$\\frac{1}{\\sqrt{8-x}}$. Write your answer in the form $y=Ax+B$.<br>Answer_v1 : y-0.0220971*x = 0.353553  Type : EQ<br>Problem_v2 : Find the linear approximation at $x=0$ to<br>$\\frac{1}{\\sqrt{2-x}}$. Write your answer in the form $y=Ax+B$.<br>Answer_v2 : y-0.176777*x = 0.707107  Type : EQ<br>Problem_v3 : Find the linear approximation at $x=0$ to<br>$\\frac{1}{\\sqrt{4-x}}$. Write your answer in the form $y=Ax+B$.<br>Answer_v3 : y-0.0625*x = 0.5  Type : EQ<br>Difficulty Level : 3<br>**----- End of picture text -----**<br>


Figure 1: Overview of UGMathBench. UGMathBench is a diverse and dynamic benchmark specifically designed for evaluating undergraduate-level mathematics with LLMs, covering 16 distinct subjects and featuring 10 different answer types. Each problem contains three randomized versions, with EAcc and ∆ rigorously assessing LLMs’ true reasoning skills. 

Azerbayev et al., 2023). Consequently, LLMs have become increasingly capable of solving complex mathematical problems (Hendrycks et al., 2021b; Ahn et al., 2024). 

With the rapid advancements in LLMs, evaluating their reasoning capabilities has become increasingly important (He et al., 2024a; Xu et al., 2025a; Phan et al., 2025), especially in mathematics (Liu et al., 2024; Gao et al., 2024). Although benchmarks such as GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) are commonly used to assess these abilities, they are becoming insufficient due to rapid progress in model performance, as evidenced by accuracy exceeding 97% in GSM8K (Zhou et al., 2023) and 94.8% in MATH (OpenAI, 2024b). While more challenging benchmarks are being introduced (Tang et al., 2024; Liu et al., 2024), they often remain limited in size and scope regarding undergraduate-level mathematics, which is essential due to its breadth and complexity (see Table 1). Moreover, there are growing concerns about test set contamination in these static benchmarks (Srivastava et al., 2024; Zhang et al., 2024; Qian et al., 2024; White et al., 2024). Recent efforts Qian et al. (2024); Srivastava et al. (2024) have introduced dynamic benchmarks by functionalizing the original problems in GSM8K and MATH to generate randomized variations through variable disturbance. However, these initiatives primarily focus on elementary and competition-level mathematics (see Table 1). These limitations highlight the pressing need for a comprehensive and dynamic benchmark specifically designed to assess undergraduate-level mathematical reasoning. 

In this paper, we present UGMathBench, a diverse and dynamic benchmark designed to evaluate the mathematical reasoning capabilities of LLMs across a wide range of undergraduate-level mathematical topics, as illustrated in Figure 1. We meticulously collect, clean, and format undergraduate-level mathematical problems from our online homework grading system (see Appendix B.1), resulting in a benchmark comprising 5,062 problems in 16 subjects, categorized into eight atomic answer types and two compound answer types. A key feature of UGMathBench is the inclusion of multiple randomized versions for each problem, which aids in assessing the true reasoning abilities of LLMs through the **E** ffective **Acc** uracy (EAcc) and reasoning gap (∆) (see Section 3.3). EAcc represents the percentage of problems correctly solved across all versions, providing insights into intrinsic reasoning skills. It operates on the premise that a model capable of solving a problem through reasoning should also be able to solve all its variants under variable disturbance (Srivastava et al., 2024; Qian et al., 2024). The reasoning gap, ∆, is defined as the difference between the average accuracy across all versions and the EAcc, quantifying the robustness of reasoning when the original problems undergo slight 

2 

Published as a conference paper at ICLR 2025 

modifications. These metrics help mitigate the impact of potential test set contamination (Deng et al., 2024; Dong et al., 2024; Golchin & Surdeanu, 2023; Roberts et al., 2023) and ensure a more rigorous evaluation of LLMs’ mathematical reasoning abilities. These features are summarized more clearly in Figure 1 and Table 1. 

We conducted an extensive evaluation of the leading LLMs, including proprietary models such as OpenAI-o1 (OpenAI, 2024b) and open-source models like LLaMA-3-Instruct (AI@Meta, 2024). Despite their advanced capabilities, the best EAcc achieved is 56.3% by OpenAI-o1 (OpenAI, 2024b) and all LLMs exhibit a large reasoning gap. These results highlight the considerable challenges that UGMathBench presents to current LLms in terms of mathematical reasoning, underscoring the need for future research focused on developing "large reasoning models" characterized by high EAcc and a reasoning gap ∆= 0. To summarize our key findings: 

1. Even LLMs with the most advanced reasoning ability, OpenAI-o1-mini, achieves a 56.30% EAcc on UGMathBench, much lower on other text-only mathematical benchmarks. 2. All LLMs evaluated exhibit high reasoning gap with Robustness Efficiency (RE, the ratio between ∆ and EAcc) ranging from 20.78% to 196.6%, pinpointing the inconsistencies of current LLMs in solving problems with variable disturbance. 3. There remains a significant discrepancy among closed-source LLMs and open-source LLMs (even specialized mathematical LLMs). Among open-source LLMs, only Qwen-2-Math-72B-Instruct and Mistral-Large-Instruct have comparable performance with GPT-4o. 4. The average EAcc varies by subject, with Arithmetic scoring 62.8%. In contrast, Abstract Algebra, Differential Equations, and Financial Mathematics have average EAccs of less than 10%. 5. An error analysis of OpenAI-o1-mini’s performance reveals that calculation errors are a major concern. Even the same problem presented in different randomized versions can lead to varying types of errors. 

Table 1: Comparison of various benchmarks. "#Types" indicates the number of answer types in the dataset. "#Subjects" specifies the number of mathematical subjects covered. "Dynamic" denotes whether the dataset is dynamic or static. "#Test" shows the number of test examples in the dataset, while "#College" refers to the number of test examples at the college level. 

|**Dataset**|**Level**|**#Types**|**#Subjects**|**Dynamic**|**#Test**|**#College**|
|---|---|---|---|---|---|---|
|GSM8K|Elementary|1|-||1,319|0|
|MATH|Competition|3|7||5,000|0|
|MMLU-Math|All|1|-||844|116|
|TAL-SCQ|K12 Math|1|-||1,496|0|
|AGIEval-SAT-Math|High School|2|-||102|0|
|AGIEval-Math|Competition|2|-||938|0|
|CollegeMath|College|3|7||2,818|2,818|
|MathBench|All|1|5||1781|466|
|GSM1K|Elementary|1|-||1,250|0|
|FN-EVAL|Competition|3|7||2,060|0|
|VarBench-Math|Elementary|1|-||1,319|0|
|LiveBench-Math|Competition|2|-||232|0|
|UGMathBench|College|**10**|**16**||**5,062**|**5,062**|



## 2 RELATED WORK 

**Mathematical Benchmarks** . Mathematical reasoning is increasingly vital for assessing the fundamental reasoning capabilities of LLMs (Ahn et al., 2024). Several math-related datasets have been proposed in this area (Koncel-Kedziorski et al., 2016; Amini et al., 2019; Hendrycks et al., 2021a; Cobbe et al., 2021; Hendrycks et al., 2021b; Chen et al., 2022). Among these, GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) are the most representative datasets for elementary and high school-level math reasoning, respectively. However, as modern LLMs become increasingly powerful, these benchmarks lack sufficient challenge forlatest LLMs. Notably, o1 (OpenAI, 2024b) achieves 94.8% accuracy on MATH, which was previously considered highly complex. To better assess the mathematical reasoning abilities of current LLMs, some researchers create variants of existing benchmarks (Shi et al., 2023; Chen et al., 2024; Li et al., 2024b; Xu et al., 2024b), while others propose new, more challenging math reasoning benchmarks (Chen et al., 2023; Wang et al., 

3 

Published as a conference paper at ICLR 2025 

2023a; Collins et al., 2024; Tang et al., 2024; Liu et al., 2024; Wang et al., 2024). CollegeMath (Tang et al., 2024) covers several college-level mathematics subjects with limited answer types. In contrast, our UGMathBench encompasses a broader range of subjects, answer types, and test examples. In addition, there are also several cross-modality math-related datasets (Chen et al., 2021; Lu et al., 2023; Xu et al., 2024a; Yue et al., 2024a; He et al., 2024b;a; Huang et al., 2024; Yue et al., 2024b). 

**Dynamic Benchmarks for Mathematical Reasoning** . Test set contamination, wherein benchmark test data appear in a newer model’s training set, significantly challenges fair LLM evaluation by artificially inflating performance (Deng et al., 2024; Dong et al., 2024; Golchin & Surdeanu, 2023; Roberts et al., 2023). Since pretraining data often involve large corpora scraped from the Internet, any static benchmark risks data contamination (Zhang et al., 2024; Qian et al., 2024). To mitigate this, recent benchmarks maintain private test sets (Zhang et al., 2024; Huang et al., 2024), requiring anyone who wishes to evaluate their models to submit predictions for centralized processing before publishing results on their leaderboards. However, this process can be inefficient and lacks transparency for error analysis (Qian et al., 2024). An alternative is releasing dynamic benchmarks that are periodically updated (Srivastava et al., 2024; Qian et al., 2024; White et al., 2024). For example, Srivastava et al. (2024) have functionalized a subset of the MATH dataset to regenerate new versions of the test set by reassigning variable values. In this vein, our UGMathBench is a dynamic benchmark featuring different sampled values for variables by setting distinct random seeds. Currently, we release three snapshots for each question in UGMathBench and plan to release new versions if leading open-source LLMs reach accuracy saturation. 

## 3 THE UGMATHBENCH BENCHMARK 

## 3.1 UGMATHBENCH OVERVIEW 

We introduce the UGMathBench, a dynamic undergraduate-level mathematical reasoning benchmark designed to thoroughly and robustly assess the mathematical reasoning ability of LLMs. UGMathBench enables fair evaluation through randomized versions of single problems. Unlike GSM1K (Zhang et al., 2024), our test set labels are publicly available, facilitating efficient evaluation and effective error analysis. UGMathBench covers fifteen core subject areas in undergraduate-level mathematics, including single-variable calculus, multivariable calculus, differential equations, probability, and more, encompassing a total of 111 specific topics (details in Appendix A.2). UGMathBench comprises a set of 5,062 problems 

Table 2: Benchmark Statistics 

||**Statistic**|**Number**|
|---|---|---|
||Total Problems|5562|
||Number of Versions<br>Total Subjects/topics<br>Total Answer Types|x 3<br>16/111<br>10|
||Total Diffculty Level|6|
||Average Problem Tokens<br>Average Number of Answers|122.63<br>2.77|



in 3 different randomized snapshots with 10 different answer types (see Appendix A.3). These answer types range from atomic types (e.g., numerical value, expression) to compound types (e.g., multiple answers in ordered or unordered lists), setting UGMathBench apart from many other math-related benchmarks that focus primarily on a single answer with an atomic type. We randomly select 100 problems to examine student performance using our grading system’ records, with each problem being completed by varying numbers of students ranging from 99 to 1,537. The average accuracy on the first attempt is 56.5%, while the average accuracy on the final attempt increased to 96.1%. 

## 3.2 UGMATHBENCH CREATION 

Our UGMathBench creation process has three distinct phases: data collection, data cleaning & deduplication, and answer type annotation. 

**Data Collection** . The dataset for UGMathBench is carefully compiled from the online grading system of our institute’s undergraduate courses (see Appendix B.1). All problems in our system are generated by programs that specify particular variable values to ensure correctness and maintain the same solution (see Appendix A.1). We gather all mathematics-related problems, resulting in 16 subjects and 111 topics in total. To prevent student cheating, our grading system offers randomized versions of most problems (see Figure 1), similar to the variable disturbance approach in Qian et al. (2024). To create a dynamic benchmark, we exclude static problems without randomized versions, as well as those containing images, ensuring a text-only reasoning benchmark. The collected problems are originally in HTML format. 

4 

Published as a conference paper at ICLR 2025 

**Data Cleaning and Deduplication** . After collecting problems in the HTML format, we utilize the bs4[1] and re[2] Python packages to convert them into Latex. Since no conversion process is flawless, we manually verify the converted LaTeX files against the original HTML files. The latex files are then further organized into the format shown in Figure 1. After converting and cleaning all the problems, we perform deduplication within each subject based on embeddings generated by text-embedding-ada-002 to remove duplicated problems (see Appendix B.2). The thresholds and the number of questions that are filtered out are given in Table 10. 

**Answer Type Annotation** . Meta-information (e.g. subject, topic, subtopic, difficulty level as shown in Figure 1) is stored in our grading system and is easily extracted along with the LaTeX files. The primary task is determining the answer types. Problems requiring definitive responses are largely categorized into two main classes: atomic and compound. Questions with a single required answer fall into the atomic type, while questions with multiple answers are classified as compound, represented by a list of atomic answers separated by commas. The atomic type can be further classified into eight types, and the compound answer lists can be either ordered or unordered, with each atomic answer fitting one of the aforementioned eight types. Simple examples of each type are provided in Table 3, and detailed definitions are available in Appendix A.3. 

Table 3: Examples of eight atomic answer types. 

||**Type**|**Example**|
|---|---|---|
||Numerical Value<br>Expression<br>Equation|_π/_4<br>_x_2 + 1<br>_x_2 +_y_2 = 1|
||Interval<br>True/False<br>MC with single answer|(_−∞, −_1]<br>Yes<br>A|
||MC with multiple answers<br>Open-Ended|ACF<br>h(1-x)|



## 3.3 EVALUATION METRICS 

We denote the set of test examples in UGMathBench by _D_ with a specific test example denoted as _ei_ , where _i_ represents the index of the example. Each example _ei_ consists of questions presented in different randomized versions: _qi_[1] _[, q] i_[2] _[, ..., q] i[V]_[,][where] _[V]_[is][the][total][number][of][versions][3][.][The] corresponding ground-truth answers for these versions _qi_[1] _[, q] i_[2] _[, ..., q] i[V]_[are denoted by] _[ a]_[1] _i[, a]_[2] _i[, ..., a][V] i_[.] The answer generated by an LLM _M_ for a specific version of the question in the _i_ -th test example is denoted by _M_ ( _qi[v]_[)][.][Inspired by Srivastava et al. (2024), we define the following metrics to evaluate] the true mathematical reasoning ability of LLM _M_ in UGMathBench. 

**Accuracy of Version v Accv** is defined as the average accuracy of model _M_ on the set of questions with version _v_ in _D_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0005-08.png)


where I is an indicator function and _|D|_ denotes the number of examples in UGMathBench. It assesses the performance of an LLM on the specific version _v_ from UGMathBench. 

**Average Accuracy AAcc** is defined as the mean of all Acc _v_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0005-11.png)


This metric evaluates the performance across all versions of the questions. 

**Effective Accuracy EAcc** is defined as the accuracy in solving a test example _ei_ across all its _V_ versions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0005-14.png)


If a model is able to solve a test case using proper reasoning, it should correctly solve this problem for all randomized versions. Thus, effective accuracy measures the fraction of test cases correctly solved across all versions _V_ . It measures true reasoning of test cases in UGMathBench. 

**Reasoning Gap** ∆ is defined as the percentage decrease between AAcc and EAcc. It provides a measure of the robustness of reasoning, with ∆= 0 being true reasoning with high robustness. 

> 1https://pypi.org/project/beautifulsoup4/ 

> 2https://docs.python.org/3/library/re.html 

> 3Currently, V=3 and we plan to release more versions in the future. 

5 

Published as a conference paper at ICLR 2025 

**Robustness Efficiency (RE)** is defined as the ratio of the Reasoning Gap (∆) to the EAcc, expressed as RE = ∆ _/_ EAcc. This metric evaluates the extent of the reasoning gap relative to the model’s effective reasoning ability (i.e., EAcc). RE captures robustness by taking the effectiveness of mathematical reasoning into account , with lower values indicating superior performance in adapting to variations across different versions of problems in UGMathBench. Achieving a higher EAcc and a lower ∆ results in a more favorable (lower) RE, reflecting improved robustness relative to "true" reasoning ablility of LLMs. 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETUP 

**Evaluated LLMs** . Our evaluation covers 23 leading LLMs, including closed-source commercial LLMs and open-source LLMs. Based on our UGMathBench, we provide a thorough evaluation of the mathematical reasoning capabilities of current LLMs. The evaluated LLMs are listed below: 

- For proprietary LLMs, we select OpenAI-o1-mini (OpenAI, 2024b), GPT4o (OpenAI, 2024a), GPT4o-mini (OpenAI, 2024a), and Claude-3-Opus (Anthropic, 2024). 

- For open-source general-purpose LLMs, we evaluated the LLaMA-3-Instruct series (8B, 70B) (AI@Meta, 2024), Qwen2-Instruct (7B, 72B)(Yang et al., 2024a), Yi-1.5-Chat (6B, 9B, 34B) (AI et al., 2024), Mistral-7B-Instruct (Jiang et al., 2023), Mistral-Nemo-Instruct-2407 (Mistral, 2024c), Mistral-Small-Instruct-2409 (Mistral, 2024d), Mistral-Large-Instruct-2407 (Mistral, 2024b), DeepSeek-MOE-16B-Chat (Dai et al., 2024), and DeepSeek-V2-Lite-Chat (DeepSeek-AI, 2024). 

- We also include some specialized math LLMs: DeepSeekMath-7B (-RL, -Instruct) (Shao et al., 2024), Qwen2-Math (7B, 72B)(Yang et al., 2024b), Mathstral-7B (Mistral, 2023), and NuminaMath-7B-CoT (Beeching et al., 2024). 

Details of these LLMs are provided in Appendix C.1. 

**Evaluation Settings** . We employ Acc _v_ to evaluate the average performance of version _v_ , AAcc to measure the average performance across all versions, EAcc to quantify true reasoning, and reasoning gap ∆ to assess the robustness of reasoning (see Section 3.3). To remove the effect of sensitivity of few-shot prompts (Lu et al., 2022; Ma et al., 2023), all our experiments use zero-shot prompts, tailored to different answer types for better answer extraction and rule-based matching. Detailed prompts are given in Appendix C.2. We use vLLM[4] to speed up the evaluation process. To maintain consistency in evaluations and facilitate reproduction, we set the maximum output length to 2,048 tokens and employ a greedy decoding strategy with temperature 0. 

## 4.2 MAIN RESULTS 

The overall experiment results are shown in table 4. We have the following key observations: 

**UGMathBench is a challenging benchmark for evaluating the mathematical reasoning capabilities of LLMs.** Even LLMs with the most advanced reasoning abilities, OpenAI-o1 (mini version), achieve only 56.3% EAcc on UGMathBench, while most open-source LLMs, including most specialized mathematical models, struggle to reach a 30% EAcc. Compared to commonly used mathematics benchmarks like MATH (Hendrycks et al., 2021b), UGMathBench proves to be more challenging. For instance, OpenAI-o1-mini achieves 90% on MATH (v.s. 56.3% on UGMathBench). 

**Even leading LLMs still have inconsistencies when solving problems with multiple versions** . LLMs with an AAcc greater than 20% display a reasoning gap ∆ exceeding (or near) 10%. All LLMs demonstrate extremely high RE on UGMathBench, with values ranging from 20.78% to 196.6%. Among the five models with the lowest RE, three of them are from OpenAI (OpenAI-o1-mini: 20.78%; GPT-4o: 20.89%; Mistral-Large-Instruct: 24.36%; Qwen2-Math-72B-Instruct: 24.39%; GPT-4o-mini: 27.87%). These results pinpoint the limitaions of current LLMs and urge us to develop ture "large reasoning models" with high EAcc and ∆= 0. 

> 4https://github.com/vllm-project/vllm 

6 

Published as a conference paper at ICLR 2025 

Table 4: **Main Results on UGMathBench** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**68.02**<br>**67.95**<br>**68.04**<br>GPT-4o-2024-08-06<br>59.92<br>60.79<br>60.41<br>GPT-4o-mini-2024-07-18<br>51.58<br>53.14<br>52.61<br>Claude-3-Opus-20240229<br>48.62<br>50.32<br>49.47||||**68.00**<br>60.37<br>52.44<br>49.47|**56.30**<br>49.94<br>41.01<br>37.00|11.70<br>10.43<br>11.43<br>12.47|**20.78**<br>20.89<br>27.87<br>33.69|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>25.33<br>26.83<br>26.59<br>Mistral-7B-Instruct<br>10.19<br>11.16<br>10.33<br>Qwen2-7B-Instruct<br>35.60<br>37.30<br>36.23<br>LLaMA3-8B-Instruct<br>16.00<br>17.01<br>16.63<br>Yi-1.5-9B-Chat<br>33.72<br>34.29<br>34.85||||26.25<br>10.56<br>36.38<br>16.55<br>34.29|15.23<br>4.44<br>25.15<br>8.91<br>21.12|11.02<br>6.12<br>11.23<br>7.64<br>13.17|72.33<br>137.6<br>44.65<br>85.74<br>62.36|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|24.53<br>5.59<br>12.82<br>40.10<br>37.08|25.62<br>5.85<br>13.67<br>40.52<br>38.11|25.09<br>5.97<br>12.76<br>40.04<br>37.65|25.08<br>5.80<br>13.08<br>40.22<br>37.61|15.43<br>1.96<br>5.69<br>28.84<br>24.34|9.65<br>**3.85**<br>7.39<br>11.38<br>13.28|62.57<br>196.6<br>130.0<br>39.45<br>54.55|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|33.25<br>47.49<br>55.91|34.35<br>48.56<br>56.16|33.27<br>47.23<br>55.97|33.62<br>47.76<br>56.01|23.27<br>35.78<br>45.04|10.35<br>11.98<br>10.97|44.48<br>33.50<br>24.36|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>23.61<br>24.87<br>23.19<br>DeepSeek-Math-7B-RL<br>28.66<br>29.97<br>28.94<br>NuminaMath-7B-CoT<br>29.32<br>30.07<br>30.01<br>Mathstral-7B-v0.1<br>28.57<br>28.47<br>28.51<br>Qwen2-Math-7B-Instruct<br>43.01<br>44.13<br>44.05<br>Qwen2-Math-72B-Instruct<br>56.95<br>57.05<br>57.09||||23.89<br>29.19<br>29.80<br>28.51<br>43.73<br>57.03|13.61<br>19.24<br>18.81<br>17.94<br>32.46<br>45.85|10.28<br>9.95<br>10.99<br>10.58<br>11.27<br>11.18|75.52<br>51.71<br>58.44<br>58.96<br>34.73<br>24.39|



**There remains a significant discrepancy between closed-source models and open-source LLMs.** OpenAI-o1-mini achieves the best results across Acc _i, i_ = 1 _,_ 2 _,_ 3, as well as in AAcc and EAcc. However, the most powerful open-source LLM evaluated, Qwen2-Math-72B-Instruct, still falls short: it shows a 10.97% lower AAcc and a 10.45% lower EAcc compared to OpenAI-o1-mini. The best-performing open-source chat model is Mistral-Large-Instruct-2407, which ranks second among all open-source LLMs. Only 2 out of 19 open-source LLMs exceed GPT-4o-mini in terms of EAcc, and only 3 out of 19 have an EAcc comparable to Claude-3-Opus. Additionally, more than half of the open-source LLMs (10 out of 19) have an EAcc smaller than 20%. 

## 5 ANALYSIS 

In this section, we conduct an in-depth analysis of the performance of the 23 LLMs evaluated on UGMathBench by investigating the following research questions: 1. What is the relationship between EAcc and Acc _v_ ? (Section 5.1) 2. How do model size and model series influence performance on UGMathBench? (Section 5.2) 3. How do LLMs perform across different subjects, difficulty levels, and, different topics on UGMathBench? (Section 5.3) 4. What are the typical response errors made by the best-performing LLM (OpenAI-o1-mini), and how are they distributed? (Section 5.4) 

## 5.1 RELATIONSHIP BETWEEN EACC AND ACC _v_ 

To investigate the relationship between EAcc and Acc _v_ , scatter plots of EAcc against each Acc _v_ are shown in Figure 2. From Figure 2, we have the following conclusions: 

**All LLMs fall below the diagonal lines** . Each LLM evaluated is represented as a point in the subfigures of Figure 2, plotted on the axes of (Acc _v,_ EAcc). Although LLMs exhibit small variations in accuracy across different versions, they consistently demonstrate a lower EAcc than Acc _v_ , which suggests that the accuracy of individual versions is insufficient for assessing the reasoning capabilities of LLMs. By considering EAcc alongside accuracy, we can gain a better understanding of how LLMs 

7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0008-01.png)


**----- Start of picture text -----**<br>
OpenAI-o1-mini-2024-09-12 Yi-1.5-34B-Chat Qwen2-72B-Instruct DeepSeek-V2-Lite-Chat<br>GPT-4o-2024-08-06 Mistral-7B-Instruct Qwen2-Math-7B-Instruct DeepSeek-Math-7B-Instruct<br>GPT-4o-mini-2024-07-18 Mistral-Nemo-Instruct-2407 Qwen2-Math-72B-Instruct DeepSeek-Math-7B-RL<br>Claude-3-Opus-20240229 Mistral-Small-Instruct-2409 LLaMA3-8B-Instruct NuminaMath-7B-CoT<br>Yi-1.5-6B-Chat Mistral-Large-Instruct-2407 LLaMA3-70B-Instruct Mathstral-7B-v0.1<br>Yi-1.5-9B-Chat Qwen2-7B-Instruct DeepSeek-MOE-16B-Chat<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>0 10 20 30 40 50 60 70 0 10 20 30 40 50 60 70 0 10 20 30 40 50 60 70<br>Acc1 Acc2 Acc3<br>EAcc EAcc EAcc<br>**----- End of picture text -----**<br>


Figure 2: EAcc v.s. Acc _v_ on UGMathBench. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0008-03.png)


**----- Start of picture text -----**<br>
OpenAI-o1-mini-2024-09-12 GPT-4o-2024-08-06 GPT-4o-mini-2024-07-18 Claude-3-Opus-20240229<br>Yi Qwen-Chat DeepSeek-Chat Numina Qwen-Math<br>Mistral LLaMA DeepSeek-Math Mathstral<br>1.75<br>50 1.50 DeepSeek-Math-7B-RL<br>40 1.25 DeepSeek-Math-7B-Instruct<br>1.00<br>30<br>0.75 DeepSeek-V2-Lite-Chat<br>20 DeepSeek-Math-7B-RL<br>0.50<br>DeepSeek-Math-7B-Instruct<br>10 0.25<br>DeepSeek-V2-Lite-Chat<br>DeepSeek-MOE-16B-Chat DeepSeek-MOE-16B-Chat<br>0 0.00<br>7 22 70 123 7 22 70 123<br>Model Size (in billions of parameters) Model Size (in billions of parameters)<br>EAcc 2-RE<br>**----- End of picture text -----**<br>


Figure 3: Left: EAcc v.s. Model Size. Right: 2-RE v.s. Model Size. The comparison chart of performance versus performance (EAcc and RE) on UGMathBench for all LLMs evaluated, with models from the same series connected by lines of the same color. The horizontal dotted lines represent the score of close-source LLMs. 

perform when solving problems that have different randomized versions. The discrepancy between EAcc and Acc _v_ highlights a new inconsistency mode (Ahn et al., 2024) of current LLMs: they may become inconsistent in their answers when the problem is slightly altered. 

**There is an apparent trend that a high EAcc consistently leads to a high Acc** _v_ . A model with high EAcc is more effective in handling variable disturbances, resulting in high accuracy for each version. However, as EAcc becomes increasingly large, the difference between Acc _v_ and EAcc tends to increase until it stabilizes around 10%. 

## 5.2 THE EFFECT OF MODEL SIZE AND MODEL SERIES 

Figure 3 has shown how EAcc and 2-RE changes with the parameter size.We can observe that: 

**LLMs within the same series have shown steady improvement as the parameter size increases** . When the model size increases from 7B to around 100B, EAcc substantially improve and RE steadily decrease for Qwen-Chat, Qwen-Math, Mistral, Deepseek-Chat, LLaMA-3-Instruct, and Yi-Chat series, indicating a steady improvement in performance in effectiveness and robustness of mathematical reasoning. 

8 

Published as a conference paper at ICLR 2025 

**Specialized mathematical LLMs typically outperform their general-purpose counterparts** . For example, the Qwen2-Math series achieves significantly higher EAcc and lower RE than its generalpurpose chat LLMs with the same model size. Among all 7B specialized mathematical LLMs, Qwen2-Math-7B-Instruct ranks first surpassing DeepSeek-Math-RL (second best) by a large margin. 

## 5.3 PERFORMANCE ACROSS DIFFERENT SUBJECTS AND DIFFICULTY LEVELS 

Figure 4a shows the average EAcc of different models in each subject. Figure 4b shows the averaged EAcc of all subjects with respect to different levels of difficulty. The detailed performances across different subjects and topics for each model can be found in Appendix D and F. 

**The average EAcc varies across different subjects** . LLMs are effective at solving Arithmetic problems, achieving 62.8% EAcc. In addition to Arithmetic, LLMs are also adept at Algebra, Combinatorics, and Complex Analysis (over 30% average EAcc). The three least effective areas are Abstract Algebra, Differential Equations, and Financial Mathematics, which typically require challenging domain knowledge (less than 10% average EAcc). 

**In general, the averaged EAcc decreases as the level increases.** OpenAI’s o1-mini is the strongest among all models and wins by a larger margin as the level increases. Mistral-Large/Small-Instruct and Qwen2 series are the most competitive open-source models on our benchmark, with EAcc comparable to leading commercial models such as GPT-4o. However, as the level of difficulty increases, they still lag behind GPT-4o, suggesting the gap between the leading open-source LLMs and proprietary LLMs in solving difficult math problems. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0009-06.png)


**----- Start of picture text -----**<br>
Arithmetic 62.8% 1<br>Algebra 39.3% 70%<br>Combinatorics 34.1% 60% o1-mini-2024-09-12<br>Calculus-Single-VariableComplex AnalysisNumber TheoryLinear AlgebraTrigonometryGeometry 24.3%25.1%25.5%26.0%27.8%31.6% 2 50%40%30%20% 5 & 6 gpt-4o-2024-08-06Mistral-Large-Instruct-2407Qwen2-Math-72B-Instructgpt-4o-mini-2024-07-18<br>Calculus-MultivariableSet Theory and LogicAbstract AlgebraProbabilityStatistics 9.2% 17.3%19.2%21.7%23.7% Level 1Level 2Level 3Level 4 claude-3-opus-20240229Qwen2-72B-InstructQwen2-Math-7B-InstructQwen2-7B-Instruct<br>Differential Equations 8.7% Level 5 Mistral-Small-Instruct-2409<br>Financial Mathematics 6.4% Level 6 3 4<br>0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7<br>EAcc<br>(a) EAcc across subjects (b) EAcc across levels<br>**----- End of picture text -----**<br>


Figure 4: Relationship between EAcc, subject, and level of difficulty. (a) EAcc of different subjects, averaged across all models. Each bar consists of several segments with colors indicating their corresponding difficulty level. Notice that the length of each color segment only indicates its proportion within all problems of all levels within that subject, and is not comparable between levels or subjects. (b) EAcc of different levels, averaged across all subjects. Only models with top-10 EAcc are included for brevity. Levels 5 and 6 are combined since level 6 has few samples. 

## 5.4 ERROR ANALYSIS 

We perform a comprehensive error analysis on OpenAI-o1-mini by randomly selecting 100 problems, each having at least one incorrectly solved version (yielding a total of 300 versions). As shown in Figure 5a, there are 231 incorrect versions, and OpenAI-o1-mini failed to solve 56% of the problems across all versions. We then categorize these errors into six types, as illustrated in Figure 5b. Calculation errors, including both numerical and expression errors, represent the largest category, with several examples provided in Appendix E. We find that OpenAI-o1-mini tends to streamline its outputs to avoid generating too long responses, sometimes leading to erroneous results. Additionally, we encounter some "bad questions" that primarily arise due to overly complex structures (e.g. containing long tables) or inadequately described (e.g. undefined variables in previous problems). This is because our homework grading system (see Appendix B.1) is designed for students with a userfriendly interface, and some problems may not be suitable for LLM to solve. In our sample of 300 versions, 19 were identified as "bad problems," giving us an estimated occurrence of approximately 2.7% in our UGMathBench. These problems do not impact our main claims, as no LLMs are able 

9 

Published as a conference paper at ICLR 2025 

to solve these "bad questions." We will refine these types of problems to make them more suitable for LLM evaluation in the future. No answer cleaning process is perfect, and improved evaluation codes continue to be released in MATH (Hendrycks et al., 2021b). We will also actively update our evaluation repository to improve its quality. Inspired by recent model-based evaluation efforts (Gao et al., 2024; Xu et al., 2025a), we have updated our code repository to integrate the model-based evaluation introduced by Xu et al. (2025a). Updated results will be made available there. 

Notably, we have found that even when OpenAI-o1-mini solves a problem incorrectly among all its randomized versions, the error types can be different. As shown in Figure 5a, there are around 16.1% such inconsistent errors among the 100 problems sampled. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0010-03.png)


**----- Start of picture text -----**<br>
56.0%<br>Total Count<br>50 Consistent Error<br>Inconsistent Error<br>34.2% expression calcluation error<br>40 numeric calculation error<br>30 25.0% 27.7% 5.6%2.6% flawed resoningbad questions<br>20 19.0% 8.2% instruction following<br>21.6% misunderstanding<br>10<br>0<br>1 2 3<br>Number of Wrong Versions<br>(a) Count vs Number of Wrong Versions (b) Error Type Distribution<br>Count<br>**----- End of picture text -----**<br>


Figure 5: Error Analysis of OpenAI-o1-mini on UGMathBench. 

## 5.5 FURTHER ANALYSIS 

**About Self-Improvement** . To examine how LLMs perform with refinement on UGMathBench, we conducted experiments using Progressive-Hint Prompting (PHP) (Zheng et al., 2023) (see Appendix G). Detailed results can be found in Table 32 in Appendix G. Although PHP improves EAcc and AAcc for most LLMs, the enhancements are not significant, indicating considerable room for future development. The fine-grained results in Table 33 suggest that the impact of refinement for GPT-4o varies across different subjects. For instance, PHP improves GPT-4o’s performance in abstract algebra by 7.14%, yet reduces its performance in probability by 2.08% in terms of EAcc. Our UGMathBench serves as an excellent testing ground for future research into refinement methods for solving undergraduate-level mathematics with LLMs. 

**Reasoning Gap and Test Set Contamination** . To explore how models specifically overfitting to a particular variation affect the Table 5: Effects of Overfitting. reasoning gap, we mixed a portion of the test set from one version with MetaMathQA (Yu et al., 2023) and then conducted supervised Proportion 5% 10% 15% 20% fine-tuning (SFT) Llama-3-8B on this data. Details of the SFT ∆ 3.43 3.80 4.50 4.64 process are provided in Appendix H, and the results are presented in Table 5. As the proportion of the test set included in the training data increases, the reasoning gap (∆) also becomes more pronounced. This study serves as an initial investigation into test set contamination during the SFT stage. It’s worth noting that contamination at the pre-training stage is also a significant area of interest (Razeghi et al., 2022; Jiang et al., 2024). 

## 6 CONCLUSION 

Current mathematical benchmarks are often inadequate, lacking comprehensive coverage of undergraduate-level math problems or being susceptible to test-set contamination. To fill these gaps, we propose UGMathBench, a diverse and dynamic benchmark for undergraduate-level mathematical reasoning. Our fine-grained analysis has pointed out the potential inconsistencies when LLMs encounter problems with slightly different versions. We hope that our UGMathBench can contribute to future development of "true" reasoning LLMs. 

10 

Published as a conference paper at ICLR 2025 

## ACKNOWLEDGMENTS 

This work was partially supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project Reference Number: AoE/E-601/24-N). We would like to thank all reviewers for their helpful suggestions in improving this paper. 

## REFERENCES 

- Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. In Neele Falk, Sara Papi, and Mike Zhang (eds.), _Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop_ , pp. 225–237, St. Julian’s, Malta, 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024. eacl-srw.17. 

01. AI, :, Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, Kaidong Yu, Peng Liu, Qiang Liu, Shawn Yue, Senbin Yang, Shiming Yang, Tao Yu, Wen Xie, Wenhao Huang, Xiaohui Hu, Xiaoyi Ren, Xinyao Niu, Pengcheng Nie, Yuchi Xu, Yudong Liu, Yue Wang, Yuxuan Cai, Zhenyu Gu, Zhiyuan Liu, and Zonghong Dai. Yi: Open foundation models by 01.ai, 2024. 

- AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/ blob/main/MODEL_CARD.md. 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 2357–2367, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1245. URL https://aclanthology.org/N19-1245. 

- Anthropic. Claude 3 family. https://www.anthropic.com/news/claude-3-family, 2024. Accessed: 2024-09-23. 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. _ArXiv preprint_ , abs/2310.10631, 2023. URL https://arxiv.org/abs/ 2310.10631. 

- Edward Beeching, Shengyi Costa Huang, Albert Jiang, Jia Li, Benjamin Lipkin, Zihan Qina, Kashif Rasul, Ziju Shen, Roman Soletskyi, and Lewis Tunstall. Numinamath 7b cot. https:// huggingface.co/AI-MO/NuminaMath-7B-CoT, 2024. 

- Daniel Bobrow et al. Natural language input for a computer problem solving system. 1964. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. 

- Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. GeoQA: A geometric question answering benchmark towards multimodal numerical reasoning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), _Findings of the_ 

11 

Published as a conference paper at ICLR 2025 

_Association for Computational Linguistics: ACL-IJCNLP 2021_ , pp. 513–523, Online, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.46. URL https://aclanthology.org/2021.findings-acl.46. 

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _ArXiv preprint_ , abs/2211.12588, 2022. URL https://arxiv.org/abs/2211.12588. 

- Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. TheoremQA: A theorem-driven question answering dataset. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 7889–7901, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.489. URL https://aclanthology.org/ 2023.emnlp-main.489. 

- Xinyun Chen, Ryan A Chi, Xuezhi Wang, and Denny Zhou. Premise order matters in reasoning with large language models. _ArXiv preprint_ , abs/2402.08939, 2024. URL https://arxiv.org/ abs/2402.08939. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _ArXiv preprint_ , abs/2110.14168, 2021. URL https://arxiv.org/ abs/2110.14168. 

- Katherine M Collins, Albert Q Jiang, Simon Frieder, Lionel Wong, Miri Zilka, Umang Bhatt, Thomas Lukasiewicz, Yuhuai Wu, Joshua B Tenenbaum, William Hart, et al. Evaluating language models for mathematics through interactions. _Proceedings of the National Academy of Sciences_ , 121(24): e2318124121, 2024. 

- Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-ofexperts language models. _ArXiv preprint_ , abs/2401.06066, 2024. URL https://arxiv.org/ abs/2401.06066. 

- DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. 

- Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. Investigating data contamination in modern benchmarks for large language models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 8706–8719, Mexico City, Mexico, 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.naacl-long.482. 

- Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, and Ge Li. Generalization or memorization: Data contamination and trustworthy evaluation for large language models. _ArXiv preprint_ , abs/2402.15938, 2024. URL https://arxiv.org/abs/2402.15938. 

- Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. _ArXiv preprint_ , abs/2410.07985, 2024. URL https://arxiv.org/ abs/2410.07985. 

- Shahriar Golchin and Mihai Surdeanu. Time travel in llms: Tracing data contamination in large language models. _ArXiv preprint_ , abs/2308.08493, 2023. URL https://arxiv.org/abs/ 2308.08493. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. _ArXiv preprint_ , abs/2309.17452, 2023. URL https://arxiv.org/abs/2309.17452. 

12 

Published as a conference paper at ICLR 2025 

- Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. _ArXiv preprint_ , abs/2402.14008, 2024a. URL https://arxiv.org/abs/2402.14008. 

- Zheqi He, Xinya Wu, Pengfei Zhou, Richeng Xuan, Guang Liu, Xi Yang, Qiannan Zhu, and Hua Huang. Cmmu: A benchmark for chinese multi-modal multi-type question understanding and reasoning. _ArXiv preprint_ , abs/2401.14011, 2024b. URL https://arxiv.org/abs/2401. 14011. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _ArXiv preprint_ , abs/2103.03874, 2021b. URL https://arxiv.org/abs/2103.03874. 

- Zhen Huang, Zengzhi Wang, Shijie Xia, Xuefeng Li, Haoyang Zou, Ruijie Xu, Run-Ze Fan, Lyumanshan Ye, Ethan Chern, Yixin Ye, et al. Olympicarena: Benchmarking multi-discipline cognitive reasoning for superintelligent ai. _ArXiv preprint_ , abs/2406.12753, 2024. URL https://arxiv.org/abs/2406.12753. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _ArXiv preprint_ , abs/2310.06825, 2023. URL https://arxiv.org/abs/2310. 06825. 

- Minhao Jiang, Ken Ziyu Liu, Ming Zhong, Rylan Schaeffer, Siru Ouyang, Jiawei Han, and Sanmi Koyejo. Investigating data contamination for pre-training language models. _ArXiv preprint_ , abs/2401.06059, 2024. URL https://arxiv.org/abs/2401.06059. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html. 

- Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In Kevin Knight, Ani Nenkova, and Owen Rambow (eds.), _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 1152–1157, San Diego, California, 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-1136. URL https://aclanthology.org/N16-1136. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html. 

- Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng. Common 7b language models already possess strong math capabilities. _ArXiv preprint_ , abs/2403.04706, 2024a. URL https://arxiv.org/abs/2403.04706. 

13 

Published as a conference paper at ICLR 2025 

Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, and Wei Bi. Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers. _ArXiv preprint_ , abs/2402.19255, 2024b. URL https://arxiv.org/abs/2402.19255. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Regina Barzilay and Min-Yen Kan (eds.), _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, Vancouver, Canada, 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015. 

- Hongwei Liu, Zilong Zheng, Yuxuan Qiao, Haodong Duan, Zhiwei Fei, Fengzhe Zhou, Wenwei Zhang, Songyang Zhang, Dahua Lin, and Kai Chen. Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark. _ArXiv preprint_ , abs/2405.12209, 2024. URL https://arxiv.org/abs/2405.12209. 

- Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _ArXiv preprint_ , abs/2310.02255, 2023. URL https: //arxiv.org/abs/2310.02255. 

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8086–8098, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.556. URL https://aclanthology.org/2022.acl-long.556. 

- Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, and Bingzhe Wu. Fairness-guided few-shot prompting for large language models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 8678da90126aa58326b2fc0254b33a8c-Abstract-Conference.html. 

- Mistral. Mathstral. https://mistral.ai/news/mathstral/, 2023. Accessed: 2024-0923. 

Mistral. Mistral-7b-instruct-v0.3. https://huggingface.co/mistralai/ Mistral-7B-Instruct-v0.3, 2024a. 

Mistral. Mistral large 2. =https://mistral.ai/news/mistral-large-2407/, 2024b. 

Mistral. The future of ai: Trends and predictions. https://mistral.ai/news/ mistral-nemo/, 2024c. Accessed: September 29, 2024. 

Mistral. mistralai/mistral-small-instruct-2409. https://huggingface.co/mistralai/ Mistral-Small-Instruct-2409, 2024d. 

OpenAI. Gpt-4 technical report. _ArXiv preprint_ , abs/2303.08774, 2023. URL https://arxiv. org/abs/2303.08774. 

OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024a. 

OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, 2024b. Accessed: 2024-09-23. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information_ 

14 

Published as a conference paper at ICLR 2025 

_Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2080–2094, Online, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.168. URL https://aclanthology.org/2021.naacl-main.168. 

- Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, et al. Humanity’s last exam. _ArXiv preprint_ , abs/2501.14249, 2025. URL https://arxiv.org/abs/2501.14249. 

- Kun Qian, Shunji Wan, Claudia Tang, Youzhi Wang, Xuanming Zhang, Maximillian Chen, and Zhou Yu. Varbench: Robust language model benchmarking through dynamic variable perturbation. _ArXiv preprint_ , abs/2406.17681, 2024. URL https://arxiv.org/abs/2406.17681. 

- Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot reasoning. _ArXiv preprint_ , abs/2202.07206, 2022. URL https: //arxiv.org/abs/2202.07206. 

- Manley Roberts, Himanshu Thakur, Christine Herlihy, Colin White, and Samuel Dooley. To the cutoff... and beyond? a longitudinal perspective on llm data contamination. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Y Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _ArXiv preprint_ , abs/2402.03300, 2024. URL https://arxiv.org/abs/2402. 03300. 

- Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 31210– 31227. PMLR, 2023. URL https://proceedings.mlr.press/v202/shi23a.html. 

- Saurabh Srivastava, Anto PV, Shashank Menon, Ajay Sukumar, Alan Philipose, Stevin Prince, Sooraj Thomas, et al. Functional benchmarks for robust evaluation of reasoning performance, and the reasoning gap. _ArXiv preprint_ , abs/2402.19450, 2024. URL https://arxiv.org/abs/ 2402.19450. 

- Zhengyang Tang, Xingxing Zhang, Benyou Wan, and Furu Wei. Mathscale: Scaling instruction tuning for mathematical reasoning. _ArXiv preprint_ , abs/2403.02884, 2024. URL https:// arxiv.org/abs/2403.02884. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _ArXiv preprint_ , abs/2312.11805, 2023. URL https://arxiv. org/abs/2312.11805. 

- Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. _ArXiv preprint_ , abs/2407.13690, 2024. URL https://arxiv.org/abs/2407.13690. 

- Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. _ArXiv preprint_ , abs/2307.10635, 2023a. URL https://arxiv.org/abs/2307.10635. 

15 

Published as a conference paper at ICLR 2025 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023b. URL https://openreview.net/ pdf?id=1PL1NIMMrw. 

- Yan Wang, Xiaojiang Liu, and Shuming Shi. Deep neural solver for math word problems. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pp. 845–854, Copenhagen, Denmark, 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1088. URL https: //aclanthology.org/D17-1088. 

- Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. _ArXiv preprint_ , abs/2406.01574, 2024. URL https://arxiv.org/abs/2406.01574. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html. 

- Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin (eds.), _Proceedings of the 3rd Workshop on Noisy User-generated Text_ , pp. 94–106, Copenhagen, Denmark, 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4413. URL https://aclanthology. org/W17-4413. 

- Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid ShwartzZiv, Neel Jain, Khalid Saifullah, Siddartha Naidu, et al. Livebench: A challenging, contaminationfree llm benchmark. _ArXiv preprint_ , abs/2406.19314, 2024. URL https://arxiv.org/abs/ 2406.19314. 

- Xin Xu, Shizhe Diao, Can Yang, and Yang Wang. Can we verify step by step for incorrect answer detection? _ArXiv preprint_ , abs/2402.10528, 2024a. URL https://arxiv.org/abs/2402. 10528. 

- Xin Xu, Tong Xiao, Zitong Chao, Zhenya Huang, Can Yang, and Yang Wang. Can llms solve longer math word problems better? _ArXiv preprint_ , abs/2405.14804, 2024b. URL https: //arxiv.org/abs/2405.14804. 

- Xin Xu, Qiyun Xu, Tong Xiao, Tianhao Chen, Yuchen Yan, Jiaxin Zhang, Shizhe Diao, Can Yang, and Yang Wang. Ugphysics: A comprehensive benchmark for undergraduate physics reasoning with large language models. _ArXiv preprint_ , abs/2502.00334, 2025a. URL https: //arxiv.org/abs/2502.00334. 

- Xin Xu, Yan Xu, Tianhao Chen, Yuchen Yan, Chengwu Liu, Zaoyu Chen, Yufei Wang, Yichun Yin, Yasheng Wang, Lifeng Shang, et al. Teaching llms according to their aptitude: Adaptive reasoning for mathematical problem solving. _ArXiv preprint_ , abs/2502.12022, 2025b. URL https://arxiv.org/abs/2502.12022. 

- Shangzi Xue, Zhenya Huang, Jiayu Liu, Xin Lin, Yuting Ning, Binbin Jin, Xin Li, and Qi Liu. Decompose, analyze and rethink: Solving intricate problems with human-like reasoning cycle. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. URL https://openreview.net/forum?id=NPKZF1WDjZ. 

- Yuchen Yan, Jin Jiang, Yang Liu, Yixin Cao, Xin Xu, Xunliang Cai, Jian Shao, et al. S[3] c-math: Spontaneous step-level self-correction makes large language models better mathematical reasoners. _ArXiv preprint_ , abs/2409.01524, 2024. URL https://arxiv.org/abs/2409.01524. 

16 

Published as a conference paper at ICLR 2025 

- Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Xin Xu, Mengdi Zhang, Jian Shao, and Yueting Zhuang. Mathfimer: Enhancing mathematical reasoning by expanding reasoning steps through fill-in-the-middle task. _ArXiv preprint_ , abs/2502.11684, 2025. URL https://arxiv.org/ abs/2502.11684. 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _ArXiv preprint_ , abs/2407.10671, 2024a. URL https://arxiv.org/abs/2407.10671. 

- An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. _ArXiv preprint_ , abs/2409.12122, 2024b. URL https://arxiv.org/abs/2409.12122. 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _ArXiv preprint_ , abs/2309.12284, 2023. URL https: //arxiv.org/abs/2309.12284. 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _ArXiv preprint_ , abs/2309.05653, 2023. URL https://arxiv.org/abs/2309.05653. 

- Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9556–9567, 2024a. 

- Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. _ArXiv preprint_ , abs/2409.02813, 2024b. URL https://arxiv. org/abs/2409.02813. 

- Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, et al. A careful examination of large language model performance on grade school arithmetic. _ArXiv preprint_ , abs/2405.00332, 2024. URL https://arxiv.org/ abs/2405.00332. 

- Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. _ArXiv preprint_ , abs/2308.04371, 2023. URL https://arxiv.org/abs/ 2308.04371. 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. _ArXiv preprint_ , abs/2304.09797, 2023. URL https://arxiv.org/abs/2304.09797. 

- Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. _ArXiv preprint_ , abs/2308.07921, 2023. URL https://arxiv.org/abs/2308.07921. 

17 

Published as a conference paper at ICLR 2025 

## LIMITATIONS 

This work has several limitations. First, UGMathBench focuses on text-only reasoning, whereas some undergraduate-level math problems require images for their solutions. Developing a multimodal benchmark for undergraduate-level mathematics will be future work. Second, UGMathBench is designed as an English-language benchmark. Extending UGMathBench to support multiple languages could be an interesting avenue for future research. Third, the number of problems in certain subjects is limited. Expanding these subjects would be valuable. 

## A DETAILED STATISTICS OF UGMATHBENCH 

## A.1 INTRINSIC MECHANISM OF DYNAMIC PROBLEMS 

A key feature of our UGMathBench is its dynamic nature. In this appendix, we detail how this is achieved. All problems in our homework grading system (see Appendix B.1) are stored as programs written in Program Generation, an established programming language for mathematics. These programs strictly specify conditions to ensure that the generated variations of each problem do not fundamentally alter their nature, required solution approach, difficulty level, or underlying knowledge points. One such program is shown in Listing 1, and two different versions of this problem it generates are given in Figure 7. The relationship between the first and second variables is defined by _$expnt_ = _−_ 1 + 2 _∗ a_ ;, which maintains the consistency of the concepts, techniques, and solutions involved in different versions of each problem. 

## A.2 DISTRIBUTION OF PROBLEMS 

Our UGMathBench covers various subjects in undergraduate-level mathematics. The detailed topics and the number of subtopics of each topic across different subjects are listed in Table 6 and 7. There are 111 topics and 583 subtopics across 16 subjects in total. Furthermore, the distribution information of our benchmark on different subjects and difficulty level is presented in Table 8. Note that there are problems with missing difficulty level in our online homework grading system (see Appendix B.1) and we remain as is for consistency. The keywords of our UGMathBench are shown in Figure 6. The detailed results across different subjects and topics are discussed in Appendix D and F. 

## A.3 ANSWER TYPES 

By carefully reviewing a large collection of problems and referring to various past benchmarks (He et al., 2024a; Huang et al., 2024), we classify all answers to be two main categories: atomic and compound. There are 8 atomic types and 2 compound types. Each compound type is composed of a list of atomic ones. These types are designed to encompass a wide range of problems. Detailed definitions for each answer type can be found in Table 9. 

## B UGMATHBENCH CREATION 

## B.1 DATA SOURCE 

UGMathBench originates from questions in the online homework grading system of our institute, utilizing WebWork[5] , an open-source online platform licensed under GNU. Widely employed for assigning mathematics and science homework in educational settings, WebWork benefits from collaborative contributions by educators across various institutions. 

Each question in WebWork is tagged with keywords related to concepts and difficulty level based on Bloom’s taxonomy[6] , which helps simplify statistical analysis and cognitive assessment. To prevent cheating from each other, WebWork is able to generate tailored problem sets with different random seeds, making it popular among educational institutions. 

> 5https://www.webwork.org 

> 6https://webwork.maa.org/wiki/Problem_Levels 

18 

Published as a conference paper at ICLR 2025 

Table 6: Topics of each subject and corresponding number of subtopics included in UGMathBench. 

|**Subject**|**Topics**|**# Sub-Topics**|
|---|---|---|
|Arithmetic|Integers|11|
||Fractions/rational numbers|12|
||Decimals|9|
||Percents|3|
||Irrational numbers|1|
||Other bases|3|
||Units|2|
|Algebra|Algebra of real numbers and simplifying expressions|8|
||Absolute value expressions and functions|3|
||Properties of exponents, rational exponents and radicals|2|
||Cartesian coordinate system|4|
||Factoring|5|
||Functions|8|
||Transformations of functions and graphs|6|
||Linear equations and functions|9|
||Quadratic equations and functions|8|
||Operations on polynomial and rational expressions|7|
||Polynomial equations and functions|7|
||Variation and power functions|5|
||Systems of equations and inequalities|1|
||Functions with fractional exponents and radical func-|1|
||tions||
||Rational equations and functions|6|
||Inverse functions|3|
||Exponential and logarithmic expressions and functions|8|
||Finite sequences and series|4|
||Conic sections|3|
|Set theory and logic|Operations on sets|5|
||Relations between sets|2|
||Functions|2|
||Propositional logic|4|
||First order logic|3|
||Pattern matching|1|
|Trigonometry|Geometric and algebraic foundations for trigonometry|3|
||Trigonometric functions|8|
||Triangle trigonometry|4|
||Analytic trigonometry|7|
||Polar coordinates & vectors|2|
|Combinatorics|Counting|8|
||Recurrence relations|3|
|Geometry|Shapes|4|
||Circle geometry|1|
||Vectorgeometry|7|
|Calculus single-variable|Calculus of vector valued functions|6|
||Concepts for multivariable functions|6|
||Differentiation of multivariable functions|7|
||Integration of multivariable functions|9|
||Vector felds|1|
||Vector calculus|6|
||Fundamental theorems|4|
|Calculus multivariable|Limits and continuity|14|
||Differentiation|13|
||Applications of differentiation|18|
||Integrals|4|
||Techniques of integration|8|
||Applications of integration|16|
||Infnite sequences and series|18|
||Parametric|6|
||Polar|3|
|Linear Algebra|Systems of linear equations|7|
||Matrices|8|
||Matrix factorizations|5|
||Euclidean spaces|8|
||Abstract vector spaces|7|
||Eigenvalues and eigenvectors|5|
||Inner products|6|
||Linear transformations|6|
||Determinants|3|
|Number Theory|Divisibility|4|
||Congruences|5|
||Diophantine equations|1|
|Financial Mathematics|Annuities|5|
||Bonds|3|
||Equations of value|2|
||Interest|6|
||Options|5|
||Expected and contingent payments|2|
||Equities|2|



19 

Published as a conference paper at ICLR 2025 

Table 7: Topics of each subject and corresponding number of subtopics included in UGMathBench (Con’t). 

|**Subject**|**Topics**|**# Sub-Topics**|
|---|---|---|
|Probability|Sample Space|6|
||Random variables|5|
||Discrete distributions|6|
||Continuous distributions|5|
||Laws, theory|2|
||Several variables|2|
|Statistics|Experimental design|1|
||Sample survey methods|2|
||Exploratory data analysis/descriptive statistics|5|
||Sampling distributions|3|
||Confdence intervals|9|
||Hypothesis tests|13|
||Simple linear regression|7|
||Time series|2|
||Point estimation|1|
||Multiple regression|5|
||Generalized linear methods|1|
|Complex Analysis|Arithmetic|8|
||Complex equations|1|
||Complex plane|1|
||Complex functions|2|
||Analytic functions|3|
||Series and residues|4|
|Differential Equations|Introductory concepts|2|
||First order differential equations|13|
||Higher order differential equations|9|
||Laplace transforms|6|
||Systems of differential equations|4|
||Numerical methods|1|
||Series solutions|3|
||Partial differential equations|7|
|Abstract Algebra|Groups|7|
||Rings|4|
||Fields andpolynomials|1|
|Total count|111|583|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0020-03.png)


**----- Start of picture text -----**<br>
systems of linear equations Polynomial vector function<br>compound intereststatisticGradient Differential Equationdivergence theorem Equationsannuitiesvectorsubgroupscombinatoricsdecimal Triple IntegralMultivariable permutationminimaSurface Integralnormal linear approximationlinear equationstaylor series<br>metricexponential functionsCalculusdirectional derivativesurface integraldescriptive statisticsmedian trigonometryMultiple Integral linear system of equationspopulation growthQuotient Line Integralvector fieldcorrelation generatorslimit unionunit circle<br>3DPDEproof’countingDouble IntegralDerivativearccossubtracteconomicsalgebraunitsystemyield ratesidentitydot productlinear algebratangent lineinferences tangentcontinuity differentiationloans hypothesis testingIntegralfundamentaltrigonometric functionsestimatesmeanvarying paymentstheoremdegreesof calculusamplitudedivisionorderLaplace transformiterated integralofapproximationoperationsmatrix operationHypothesis Testingangle measurelogicsystemsSeriesprobabilityJoint DistributionANOVARandom Variableconversionabsolutecyclic groupsratemaximumeffective and nominal ratesdiscrete mathematicsexponentprecalculusvector spacedivergenceperiodproblem solving solvefunction<br>force<br>conditionaldifferential equation' 'linear Test Statistic matrixDirectional Derivative<br>Statistics integrating factorbasisseparablecomplexpredicatenumber sine calculusconvergent determinantModdifferentialAlgebra' 'Linear Equations' 'Matrix' 'Matricesmixed numberOptimizationcosinescalar product algebraic expressionradians double integral<br>maximum numerical heat equation Sequences<br>differential equationsthree dimensionssubspacescritical pointsorder of elements definite integralsLength negative intersectionminimuminference continuousoptimization samplesrelated ratesline integralmidlineevaluate inner product workset theory<br>confidenceintegralInferencesimple interestVector Fieldsinterval Complexrate producingof change datastatisticssample proj integers arcsin number theorydivergent gradientcomplexdistributionflux integral<br>mathematics for business first order regressioninverse trigonometric functionsModeling proportionarctan complex numbers<br>Vector lines<br>Derivatives<br>Inference about a population differentials right triangle<br>expected value<br>theory<br>differential equation' 'separable' 'initial<br>Differentiation<br>Modular Arithmetic<br>equation<br>Imaginary<br>absolute minimum<br>partial parametric<br>population real numbers<br>differential equation<br>maxima inclusion exclusion<br>**----- End of picture text -----**<br>


Figure 6: Word Cloud of Keywords in UGMathBench. 

20 

Published as a conference paper at ICLR 2025 

Table 8: Statistics of UGMathBench across different subjects and difficulty levels. 

||**Level 1**|**Level 2**|**Level 3**|**Level 4**|**Level 5**|**Level 6**|**All**|
|---|---|---|---|---|---|---|---|
|Arithmetic|22|179|47|21|53|0|322|
|Algebra|25|233|176|80|73|0|583|
|Set theory and logic|0|35|11|12|9|0|69|
|Trigonometry|5|86|44|17|25|0|178|
|Combinatorics|3|28|47|4|5|1|88|
|Geometry|5|48|86|13|6|0|161|
|Calculus single-variable|4|297|317|175|176|1|982|
|Calculus multivariable|13|279|244|80|37|0|654|
|Linear Algebra|2|172|232|53|26|0|498|
|Number Theory|2|33|7|1|1|2|46|
|Financial Mathematics|7|52|105|72|89|0|346|
|Probability|11|167|139|16|2|0|336|
|Statistics|44|151|158|46|0|0|401|
|Complex Analysis|0|40|7|2|1|1|51|
|Differential Equations|2|47|115|56|84|0|305|
|Abstract Algebra|0|8|16|9|1|7|42|
|Grand Total|145|1,855|1,751|657|588|12|5,062|



Table 9: Answer Types and Definitions 

|**Answer Type**|**Defnition**|
|---|---|
|_Atomic Types_||
|Numerical Value (NV)|Problems where the answer is a numerical value, including special values|
||such as_π_,_e_,<br>~~_√_~~<br>7,sin_π/_8, etc., represented in LaTeX.|
|Expression (EX)|Problems requiring an expression containing variables, e.g.,8_x_2 +_x_+1,|
||represented in LaTeX.|
|Equation (EQ)|Problems requiring an equation containing variables, e.g.,_y_ = 2_x_+ 1|
||represented in LaTeX.|
|Interval (INT)|Problems where the answer is a range of values, e.g.,(_−∞,_2)_∪_(3_, ∞_)|
||represented as an interval in LaTeX.|
|True/False (TF)|Problems where the answer is either True or False, Yes or No, T or F, Y|
||or N, etc.|
|MC with Single answer (MCS)|Multiple-Choice (MC) problems with only one correct option (e.g., one|
||out of four, one out of fve, etc.). The options can be captical letters|
||(ABCD) or any other string according to the problems (independent or|
||dependent, etc.).|
|MC with multiple answers (MCM)|Multiple-Choice (MC) problems with multiple correct options (e.g., two|
||out of four, two out of fve, two out of six, etc.). The options can be|
||captical letters (ABCD) or any other string according to the problems|
||(independent or dependent, etc.).|
|Open-Ended (OE)|Problems whose answers can be a term, name, or any other string that|
||satisfes the description of the problem, for example, the name of the|
||variable or function that occurs in the problem (which should be treated|
||differently with EX).|
|_Compound Types_||
|Ordered List (OL)|Problems where the answer is an ordered list, e.g. a coordinate ((1_,_2_,_3),<br>(2_t, t_2), etc.).|
|Unordered List (UOL)|Problems where the answer is an unordered list, e.g., a set or multiple|
||solutions for an equation.|



As mentioned in Appendix A.1, our problem generation programs will remain the main problem structure and knowledge points. One such example is illustrated in Figure 7 and the corresponding program is given in Listing 1. The dashed red box highlights the differences between randomized 

21 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0022-01.png)


Figure 7: An Example of Our Online Homework Grading System Based on WebWork. The dashed red box highlights the differences between randomized versions of the problem. 

versions of the problem. For this specific example, the second random variable is initialized through _a_ = _random_ (2 _,_ 10). The relationship between the first and second variables is defined by _expnt_ = _−_ 1+2 _∗ a_ . This meticulous setup maintains the consistency of the concepts, techniques, and solutions involved in different versions of each problem. Another example is demonstrated in Figure 8, where ’n’ is initialized to a randomly chosen exponent from 2 to 10 in increments of 0.1. 

All problems within WebWork are stored in the Problem Generation language and are presented in HTML format with JavaScript and external resources. This poses challenges for human interpretation and LLMs analysis, urging us to clean and re-format them in Latex (see Section 3.2). 

$a=random (2 , 10); # all-inclusive integers between 2 and 10 $expnt = −1+2*$a ; $ans = " 1/ $a*x^$a* sin ( x^$a )+1/ $a*cos ( x^$a ) " ; #Right answer TEXT( beginproblem ( ) ) ; BEGIN_TEXT _First make a substitution and then use integration by parts to evaluate the integral._ $BR \ [ \ **int** x^{ $expnt } \ **cos** ( x^$a ) dx \ ] $BR Answer : \{ ans_rule ( 4 0 )\ } \ ( + \ ) \ (C\ ) END_TEXT #Compare the right answer to the input ANS( fun_cmp ( $ans , mode=> ’ a n t i d e r ’ ) ) ; 

Listing 1: Problem Generator Code for the Problem in Figure 7 

## B.2 DEDUPLICATION 

After converting and cleaning all the problems, we perform deduplication within each subject. More specifically, we adhere to the following steps: First, we transform each question into a vector with dimension 1536 using the embedding model text-embedding-ada-002, which is the most capable 2nd generation embedding model of OpenAI[7] . We then calculate pairwise cosine similarities using the embeddings in the previous step. Finally, a threshold is selected based on manual inspection within each subject, and problems that have a cosine similarity higher than that threshold with existing problems are excluded. The thresholds and the number of questions filtered out for different subjects are presented in Table 10. We use subject-agnostic thresholds and filter out 9,382 questions in total. 

> 7The information can be found at https://openai.com/index/new-embedding-models-and-api-updates/ 

22 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0023-01.png)


Figure 8: Another Example of Our Online Homework Grading System Based on WebWork. 

Table 10: Thresholds of deduplication and the number of questions that filtered out in UGMathBench. 

||**Threshold**|**Before**|**After**|**# Filter Out**|
|---|---|---|---|---|
|Arithmetic|0.92|845|322|523|
|Algebra|0.86|4663|583|4080|
|Set theory and logic|0.96|94|69|25|
|Trigonometry|0.91|399|178|221|
|Combinatorics|0.89|140|88|52|
|Geometry|0.92|267|161|106|
|Calculus single-variable|0.90|3234|982|2252|
|Calculus multivariable|0.94|866|654|212|
|Linear Algebra|0.93|1235|498|737|
|Number Theory|0.92|68|46|22|
|Financial Mathematics|0.92|653|346|307|
|Probability|0.91|453|336|117|
|Statistics|0.92|599|401|198|
|Complex Analysis|0.95|129|51|78|
|Differential Equations|0.92|741|305|436|
|Abstract Algebra|0.94|58|42|16|
|Grand Total|-|14444|5,062|9382|



- C DETAILED EXPERIMENTAL SETUP 

## C.1 EVALUATED LLMS 

A variety of LLMs are covered in our evaluation, including closed-source commercial models and open-source models, general-purpose models and models dedicated for math problem solving. Closed-source LLMs are as follows: 

- **o1-preview** (OpenAI, 2024b): An early preview of OpenAI’s o1 model, designed to reason about hard problems using broad general knowledge about the world. We used o1-preview2024-09-12 for our evaluation. 

- **GPT-4o** (OpenAI, 2024a): GPT-4o is multimodal, and has the same high intelligence as GPT-4 Turbo but is much more efficient. 

- **Claude-3-Opus** (Anthropic, 2024): Anthropic’s most intelligent model, claimed to outperform its peers on most of the common evaluation benchmarks for AI systems. 

We evaluated the following open-source general-purpose LLMs on our benchmark: 

- **Llama-3-Instruct** (AI@Meta, 2024): LLaMA 3 Community License. 

23 

Published as a conference paper at ICLR 2025 

- **Mistral-7B-Instruct-v0.3** (Mistral, 2024a): Apache 2.0 

- **Mistral-Nemo-Instruct-2407** (Mistral, 2024c): Apache 2.0 

- **Mistral-Small-Instruct-2409** (Mistral, 2024d): MRL License. 

- **Mistral-Large-Instruct-2407** (Mistral, 2024b): MRL License. 

- **Qwen2-Instruct** (Yang et al., 2024a): Qwen2 series are developed with dedication to math and coding. We used 7B and 72B models. 7B models are licensed under Apache 2.0, while 72B models are under Tongyi Qianwen License. 

- **Yi-1.5-Chat** (AI et al., 2024): Yi-1.5 delivers stronger performance in coding, math, reasoning, and instruction-following capability compared to its predecessor. We used 6B, 9B, 34B variants. Yi-1.5 series are licensed under Apache 2.0. 

- **DeepSeek-V2-Lite-Chat** (DeepSeek-AI, 2024): model under Model License code under MIT License. 

- **deepseek-moe-16b-chat** (Dai et al., 2024): model under Model License, code under MIT License. 

The following specialized math LLMs are evaluated in our study: 

- **DeepSeekMath-7B** (Shao et al., 2024): DeepSeekMath is initialized with DeepSeek-Coderv1.5 7B and continues pre-training on math-related tokens. We tested both DeepSeekMath7B-RL and DeepSeekMath-7B-Instruct variants. Models are under Model License while code is under MIT License. 

- **Qwen2-Math** (Yang et al., 2024b): Qwen2-Math is a series of specialized math language models built upon the Qwen2 LLMs. We evaluated 7B and 72B variants. They are under the same license as Qwen2-Instruct series. 

- **Mathstral-7B** (Mistral, 2023): Mathstral stands on the shoulders of Mistral 7B and specializes in STEM subjects. This model is published under Apache 2.0. 

- **Numinamath-7B-CoT** (Beeching et al., 2024): This model is finetuned from DeepSeekMath-7B-base with two stages of supervised fine-tuning to solve math problems using chain of thought (CoT). It is licensed under Apache 2.0. 

## C.2 EVALUATION PROMPTS 

The evaluation prompts in our experiments are given in Table 11, where detailed answer type descriptions are given in Table 12. Following He et al. (2024a); Huang et al. (2024), these prompts are specially designed for different subjects and answer types for better evaluation. Note that, for chat models, we will apply chat template for better evaluation. 

## D RESULTS ACROSS DIFFERENT SUBJECTS 

The detailed results across different subjects are given in Table 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, and 28. From the results, we have the following observations: 

- OpenAI-o1-mini achieves the best results across nearly all subjects, although GPT-4o sometimes excels in terms of RE. 

- For open-source LLMs, Qwen-2-Math-72B-instruct achieves the best results in almost all subjects. However, Mistral-Large-instruct-2407 outperforms Qwen-2-Math-72B-instruct in Algebra. 

- Some LLMs even achieve zero EAcc in certain subjects. For instance, Mistral-7B-Instruct get zero EAcc in Set Theory and Logic, and seven LLMs exhibit zero EAcc in Abstract Algebra. 

- The variation in the reasoning gap differs significantly across subjects, providing more fine-grained information on how different LLMs perform across various domains. 

24 

Published as a conference paper at ICLR 2025 

Table 11: Evaluation prompts for problems with single answer or multiple answers. {problem} is the specific problem to evaluate. {subject} denotes the subject this problem belongs to and all subjects are given in Table 6. {answer_type_description} are specified in Table 12. {number_of_answers} stands for the number of answers in the problem evaluated. 

Evaluation Prompt for Single Answer 

The following is an undergraduate-level mathematical problem in {subject}. You need to solve the problem by completing all placeholders [ANS]. 

This problem involves only one placeholders [ANS] to be completed. The answer type is {answer_type_description}. 

Problem: {problem} 

All mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: "The final answer is ANSWER , where ANSWER should be your final answer. 

Evaluation Prompt for Multiple Answers 

The following is an undergraduate-level mathematical problem in {subject}. You need to solve the problem by completing all placeholders [ANS]. 

This problem involves {number_of_answers} placeholders [ANS] to be completed. Their answer types are, in order, {answer_type_description}. 

Problem: 

{problem} 

All mathematical formulas and symbols you output should be represented with LaTeX. Please end your response with: "The final answer is ANSWER , where ANSWER should be the sequence of your final answers, separated by commas. 

Table 12: Descriptions of answer types included in evaluation prompts, where {options} is the specific options from the multiple choice question evaluated. 

|**Answer Type**|**Answer Type Description**|
|---|---|
|NV|a numerical value without units|
|EX|an expression|
|EQ|an equation|
|INT|a range interval|
|TF|either True or False|
|MCS|one option for a multiple choice question with options {options}|
|MCM|more than one option concatenated without space or commas of a multi-|
||ple choice question with options {options}, for example: BD|
|OE|a word, phrase, term or string that satisfes the requirements of the|
||problem|
|OL|an ordered list of answers surrounded by parentheses with any answer|
||types, for example(1_, x_2_, True_), where "ordered list" means changing|
||the order of elements results in different answers|
|UOL|an unordered list of answers surrounded by parentheses with any answer|
||types, for example,(1_, x_2_, True_), where "unordered list" means chang-|
||ing the order of elements results in the same answer|



25 

Published as a conference paper at ICLR 2025 

Table 13: **Results on Abstract Algebra** (all figures are in %). The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆<br>**RE**|
|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**76.19**<br>**73.81**<br>**71.42**<br>GPT-4o-2024-08-06<br>42.86<br>50.00<br>52.38<br>GPT-4o-mini-2024-07-18<br>26.19<br>35.71<br>38.09<br>Claude-3-Opus-20240229<br>23.81<br>38.10<br>26.19||||**73.81**<br>48.41<br>33.33<br>29.37|**57.14**<br>28.57<br>14.29<br>11.90|16.67<br>**29.17**<br>19.84<br>69.44<br>19.04<br>133.2<br>17.47<br>146.8|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>4.76<br>2.38<br>9.52<br>Mistral-7B-Instruct<br>2.38<br>2.38<br>0.00<br>Qwen2-7B-Instruct<br>4.76<br>16.67<br>19.05<br>LLaMA3-8B-Instruct<br>0.00<br>7.14<br>9.52<br>Yi-1.5-9B-Chat<br>7.14<br>14.29<br>11.90||||5.56<br>1.59<br>13.49<br>5.56<br>11.11|0.00<br>0.00<br>4.76<br>0.00<br>0.00|5.56<br>_∞_<br>**1.59**<br>_∞_<br>8.73<br>183.4<br>5.56<br>_∞_<br>11.11<br>_∞_|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|0.00<br>0.00<br>0.00<br>9.52<br>19.05|9.52<br>2.38<br>2.38<br>19.05<br>9.52|4.76<br>2.38<br>2.38<br>21.43<br>9.52|4.76<br>1.59<br>1.59<br>16.67<br>12.70|0.00<br>0.00<br>0.00<br>7.14<br>0.00|4.76<br>_∞_<br>**1.59**<br>_∞_<br>**1.59**<br>_∞_<br>9.53<br>133.5<br>12.70<br>_∞_|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|19.05<br>28.57<br>35.71|16.67<br>47.62<br>35.71|19.04<br>30.95<br>45.24|18.25<br>35.71<br>38.89|4.76<br>14.29<br>21.43|13.49<br>283.4<br>21.42<br>149.9<br>17.46<br>81.47|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>7.14<br>9.52<br>4.76<br>DeepSeek-Math-7B-RL<br>4.76<br>19.05<br>11.90<br>NuminaMath-7B-CoT<br>7.14<br>16.67<br>7.14<br>Mathstral-7B-v0.1<br>9.52<br>4.76<br>7.14<br>Qwen2-Math-7B-Instruct<br>23.81<br>33.33<br>28.57<br>Qwen2-Math-72B-Instruct<br>45.24<br>40.48<br>42.86||||7.14<br>11.90<br>10.32<br>7.14<br>28.57<br>42.86|0.00<br>2.38<br>0.00<br>0.00<br>14.29<br>26.19|7.14<br>_∞_<br>9.52<br>400.0<br>10.32<br>_∞_<br>7.14<br>_∞_<br>14.28<br>99.93<br>16.67<br>63.65|



## E ERROR ANALYSIS 

We perform error analysis in Section 5.4, and here we showcase several examples of various error types in Table 29, 30, and 31. 

The distribution of the relative error for OpenAI-o1-mini numerical values is illustrated in Figure 9. We excluded numerical answers identical to the ground truth, as their logarithmic relative error would be negative infinity. 

## F RESULTS ACROSS DIFFERENT TOPICS 

The performances of differerent LLMs on 20 topics are shown in Figure 10, 11, 12 and 13. We observe that different LLMs exhibit varying performance patterns across these topics, and even models within the same family show differences in their rankings. 

## G REFINEMENT RESULTS 

Progressive-Hint Prompting (PHP) (Zheng et al., 2023) is a technique designed to enhance automatic, iterative interactions with LLMs. PHP uses previously generated answers as hints to progressively guide users toward the correct solutions. In our experiments, we employ the zero-shot manner to ensure a fair comparison with the primary experiments in Table 4, which helps to assess the impact of refinement on the performance of LLMs in solving undergraduate-level mathematical problems. We have set the maximum number of interaction rounds to five. To save cost, we only experiment with GPT-4o for closed-source LLMs. The results are presented in Table 32. While PHP can improve AAcc and EAcc in most cases, the improvements are not substantial. There remains considerable potential for enhancing the mathematical reasoning abilities of LLMs in solving undergraduate-level 

26 

Published as a conference paper at ICLR 2025 

Table 14: **Results on Algebra** (all figures are in %). The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**<br>∆<br>**RE**|
|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**76.16**<br>**75.47**<br>**74.44**<br>GPT-4o-2024-08-06<br>70.50<br>71.53<br>72.04<br>GPT-4o-mini-2024-07-18<br>67.41<br>67.24<br>66.38<br>Claude-3-Opus-20240229<br>63.29<br>66.38<br>63.81||||**75.36**<br>71.36<br>67.01<br>64.49|**66.72**<br>8.64<br>12.95<br>64.67<br>**6.69**<br>**10.34**<br>59.52<br>7.49<br>12.58<br>54.20<br>10.29<br>18.99|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>38.77<br>37.22<br>40.48<br>Mistral-7B-Instruct<br>19.73<br>21.96<br>19.55<br>Qwen2-7B-Instruct<br>51.63<br>53.00<br>54.03<br>LLaMA3-8B-Instruct<br>30.53<br>33.28<br>30.19<br>Yi-1.5-9B-Chat<br>46.83<br>48.89<br>48.37||||38.82<br>20.41<br>52.89<br>31.33<br>48.03|25.21<br>13.61<br>53.99<br>11.32<br>9.09<br>80.30<br>41.85<br>11.04<br>26.38<br>18.35<br>12.98<br>70.74<br>33.79<br>14.24<br>42.14|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|39.79<br>12.01<br>23.33<br>58.32<br>53.34|43.05<br>12.35<br>24.01<br>57.46<br>53.00|39.79<br>10.98<br>24.53<br>56.95<br>52.66|40.88<br>11.78<br>23.96<br>57.58<br>53.00|28.82<br>12.06<br>41.85<br>4.97<br>6.81<br>137.0<br>11.32<br>12.64<br>111.7<br>47.86<br>9.72<br>20.31<br>40.31<br>12.69<br>31.48|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|50.09<br>63.81<br>68.44|51.80<br>63.46<br>68.10|51.11<br>64.49<br>67.92|51.00<br>63.92<br>68.15|40.48<br>10.52<br>25.99<br>54.72<br>9.20<br>16.81<br>59.86<br>8.29<br>13.85|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>40.31<br>38.77<br>38.25<br>DeepSeek-Math-7B-RL<br>44.43<br>47.51<br>44.94<br>NuminaMath-7B-CoT<br>45.80<br>45.45<br>46.14<br>Mathstral-7B-v0.1<br>45.97<br>43.91<br>43.57<br>Qwen2-Math-7B-Instruct<br>56.09<br>57.46<br>57.63<br>Qwen2-Math-72B-Instruct<br>67.58<br>67.75<br>67.24||||39.11<br>45.63<br>45.80<br>44.48<br>57.06<br>67.52|26.07<br>13.04<br>50.02<br>34.31<br>11.32<br>32.99<br>34.65<br>11.15<br>32.18<br>32.25<br>12.23<br>37.92<br>47.86<br>9.20<br>19.22<br>58.83<br>8.69<br>14.77|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0027-03.png)


**----- Start of picture text -----**<br>
35%<br>Threshold: 0.001<br>30%<br>25%<br>20%<br>15%<br>10%<br>5%<br>0%<br>15 10 5 0 5<br>Log Relative Error<br>Percentage<br>**----- End of picture text -----**<br>


Figure 9: Relative Error of Numeric Answers excluding ones that are identical to the ground truth. Red dotted line indicates the tolerance threshold when we evaluate model answers. 

mathematics. The results for PHP across different subjects for GPT-4o (see Table 33) indicate that the impact of PHP is subject-agnostic. 

27 

Published as a conference paper at ICLR 2025 

Table 15: **Main Results on Arithmetic** (all figures are in %). The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**<br>∆<br>**RE**|
|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**93.48**<br>**92.86**<br>**94.10**<br>GPT-4o-2024-08-06<br>91.93<br>91.30<br>92.24<br>GPT-4o-mini-2024-07-18<br>89.44<br>87.27<br>90.06<br>Claude-3-Opus-20240229<br>84.78<br>85.71<br>86.96||||**93.48**<br>91.82<br>88.92<br>85.82|**90.37**<br>**3.11**<br>**3.44**<br>87.27<br>4.55<br>5.21<br>83.23<br>5.69<br>6.84<br>78.26<br>7.56<br>9.66|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>63.98<br>64.91<br>67.70<br>Mistral-7B-Instruct<br>30.43<br>34.78<br>33.54<br>Qwen2-7B-Instruct<br>77.64<br>78.26<br>78.57<br>LLaMA3-8B-Instruct<br>51.24<br>53.42<br>54.97<br>Yi-1.5-9B-Chat<br>73.60<br>73.60<br>70.81||||65.53<br>32.92<br>78.16<br>53.21<br>72.67|50.00<br>15.53<br>31.06<br>15.22<br>17.70<br>116.3<br>68.94<br>9.22<br>13.37<br>33.54<br>19.67<br>58.65<br>59.32<br>13.35<br>22.51|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|69.88<br>26.09<br>50.31<br>82.61<br>75.47|70.19<br>30.43<br>53.42<br>84.78<br>74.84|72.67<br>31.68<br>49.07<br>81.68<br>75.47|70.91<br>29.40<br>50.93<br>83.02<br>75.26|55.28<br>15.63<br>28.27<br>12.42<br>16.98<br>136.7<br>32.92<br>18.01<br>54.71<br>73.29<br>9.73<br>13.28<br>62.42<br>12.84<br>20.57|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|79.81<br>88.51<br>90.37|82.61<br>88.20<br>89.13|83.85<br>89.44<br>90.06|82.09<br>88.72<br>89.86|72.36<br>9.73<br>13.45<br>82.61<br>6.11<br>7.40<br>82.61<br>7.25<br>8.78|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>62.73<br>68.63<br>65.22<br>DeepSeek-Math-7B-RL<br>74.53<br>76.71<br>76.71<br>NuminaMath-7B-CoT<br>74.22<br>75.78<br>77.33<br>Mathstral-7B-v0.1<br>71.74<br>72.67<br>72.36<br>Qwen2-Math-7B-Instruct<br>88.51<br>84.78<br>86.65<br>Qwen2-Math-72B-Instruct<br>90.68<br>90.68<br>90.99||||65.53<br>75.98<br>75.78<br>72.26<br>86.65<br>90.79|49.69<br>15.84<br>31.88<br>67.08<br>8.90<br>13.27<br>62.42<br>13.36<br>21.40<br>59.63<br>12.63<br>21.18<br>78.88<br>7.77<br>9.85<br>86.96<br>3.83<br>4.40|



## H REASONING GAP AND TEST SET CONTAMINATION 

Given the limited number of examples in the test set for one version (in terms of SFT), it is necessary to mix the test set with general mathematical SFT data. For our experiments, we adopt MetaMathQA (Yu et al., 2023), a high-quality SFT dataset for math word problems, which includes 395,000 training examples. We use Llama-3-8B as our base model and set the maximum output token length to 4096. Following Tong et al. (2024), we set the learning rate to 5e-5, use a warmup ratio of 0.03, adopt cosine decay, and train the model for one epoch. Training one model takes approximately three hours on four A100 GPUs. The results are presented in Table 5. We set the proportion of the test set (from one version) for SFT to 5%, 10%, 15% and, 20% and see how the reasoning gap varies. As the proportion of the test set included in the training data increases, the reasoning gap (∆))also becomes more pronounced. This study serves as an initial investigation into test set contamination during the SFT stage. It is important to note that contamination at the pre-training stage is also a significant area of interest (Razeghi et al., 2022; Jiang et al., 2024). 

28 

Published as a conference paper at ICLR 2025 

Table 16: **Main Results on Calculus - multivariable** (all figures are in %). The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>**AAcc**<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**59.33**<br>**60.09**<br>**59.94**<br>**59.79**<br>GPT-4o-2024-08-06<br>50.00<br>49.69<br>49.24<br>49.64<br>GPT-4o-mini-2024-07-18<br>42.81<br>44.80<br>43.27<br>43.63<br>Claude-3-Opus-20240229<br>35.93<br>39.14<br>35.93<br>37.00<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>20.34<br>22.48<br>20.03<br>20.95<br>Mistral-7B-Instruct<br>5.05<br>3.98<br>4.59<br>4.54<br>Qwen2-7B-Instruct<br>25.38<br>25.23<br>26.15<br>25.59<br>LLaMA3-8B-Instruct<br>6.27<br>7.80<br>7.03<br>7.03<br>Yi-1.5-9B-Chat<br>28.75<br>28.75<br>29.51<br>29.00<br>Mistral-Nemo-Instruct-2407<br>13.61<br>12.39<br>10.55<br>12.18<br>DeepSeek-MOE-16B-Chat<br>1.68<br>0.46<br>1.22<br>1.12<br>DeepSeek-V2-Lite-Chat<br>5.05<br>4.89<br>3.52<br>4.49<br>Mistral-Small-Instruct-2409<br>27.98<br>29.51<br>27.68<br>28.39<br>Yi-1.5-34B-Chat<br>26.45<br>27.98<br>26.15<br>26.86<br>LLaMA3-70B-Instruct<br>20.79<br>23.39<br>20.95<br>21.71<br>Qwen2-72B-Instruct<br>35.02<br>37.92<br>35.32<br>36.09<br>Mistral-Large-Instruct-2407<br>46.94<br>49.08<br>47.71<br>47.91<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>11.16<br>12.54<br>12.23<br>11.98<br>DeepSeek-Math-7B-RL<br>18.81<br>17.58<br>16.06<br>17.48<br>NuminaMath-7B-CoT<br>20.49<br>20.80<br>22.02<br>21.10<br>Mathstral-7B-v0.1<br>16.36<br>17.89<br>17.89<br>17.38<br>Qwen2-Math-7B-Instruct<br>35.78<br>36.09<br>33.33<br>35.07<br>Qwen2-Math-72B-Instruct<br>48.47<br>49.39<br>51.22<br>49.69|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>**AAcc**<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**59.33**<br>**60.09**<br>**59.94**<br>**59.79**<br>GPT-4o-2024-08-06<br>50.00<br>49.69<br>49.24<br>49.64<br>GPT-4o-mini-2024-07-18<br>42.81<br>44.80<br>43.27<br>43.63<br>Claude-3-Opus-20240229<br>35.93<br>39.14<br>35.93<br>37.00<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>20.34<br>22.48<br>20.03<br>20.95<br>Mistral-7B-Instruct<br>5.05<br>3.98<br>4.59<br>4.54<br>Qwen2-7B-Instruct<br>25.38<br>25.23<br>26.15<br>25.59<br>LLaMA3-8B-Instruct<br>6.27<br>7.80<br>7.03<br>7.03<br>Yi-1.5-9B-Chat<br>28.75<br>28.75<br>29.51<br>29.00<br>Mistral-Nemo-Instruct-2407<br>13.61<br>12.39<br>10.55<br>12.18<br>DeepSeek-MOE-16B-Chat<br>1.68<br>0.46<br>1.22<br>1.12<br>DeepSeek-V2-Lite-Chat<br>5.05<br>4.89<br>3.52<br>4.49<br>Mistral-Small-Instruct-2409<br>27.98<br>29.51<br>27.68<br>28.39<br>Yi-1.5-34B-Chat<br>26.45<br>27.98<br>26.15<br>26.86<br>LLaMA3-70B-Instruct<br>20.79<br>23.39<br>20.95<br>21.71<br>Qwen2-72B-Instruct<br>35.02<br>37.92<br>35.32<br>36.09<br>Mistral-Large-Instruct-2407<br>46.94<br>49.08<br>47.71<br>47.91<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>11.16<br>12.54<br>12.23<br>11.98<br>DeepSeek-Math-7B-RL<br>18.81<br>17.58<br>16.06<br>17.48<br>NuminaMath-7B-CoT<br>20.49<br>20.80<br>22.02<br>21.10<br>Mathstral-7B-v0.1<br>16.36<br>17.89<br>17.89<br>17.38<br>Qwen2-Math-7B-Instruct<br>35.78<br>36.09<br>33.33<br>35.07<br>Qwen2-Math-72B-Instruct<br>48.47<br>49.39<br>51.22<br>49.69|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>**AAcc**<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**59.33**<br>**60.09**<br>**59.94**<br>**59.79**<br>GPT-4o-2024-08-06<br>50.00<br>49.69<br>49.24<br>49.64<br>GPT-4o-mini-2024-07-18<br>42.81<br>44.80<br>43.27<br>43.63<br>Claude-3-Opus-20240229<br>35.93<br>39.14<br>35.93<br>37.00<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>20.34<br>22.48<br>20.03<br>20.95<br>Mistral-7B-Instruct<br>5.05<br>3.98<br>4.59<br>4.54<br>Qwen2-7B-Instruct<br>25.38<br>25.23<br>26.15<br>25.59<br>LLaMA3-8B-Instruct<br>6.27<br>7.80<br>7.03<br>7.03<br>Yi-1.5-9B-Chat<br>28.75<br>28.75<br>29.51<br>29.00<br>Mistral-Nemo-Instruct-2407<br>13.61<br>12.39<br>10.55<br>12.18<br>DeepSeek-MOE-16B-Chat<br>1.68<br>0.46<br>1.22<br>1.12<br>DeepSeek-V2-Lite-Chat<br>5.05<br>4.89<br>3.52<br>4.49<br>Mistral-Small-Instruct-2409<br>27.98<br>29.51<br>27.68<br>28.39<br>Yi-1.5-34B-Chat<br>26.45<br>27.98<br>26.15<br>26.86<br>LLaMA3-70B-Instruct<br>20.79<br>23.39<br>20.95<br>21.71<br>Qwen2-72B-Instruct<br>35.02<br>37.92<br>35.32<br>36.09<br>Mistral-Large-Instruct-2407<br>46.94<br>49.08<br>47.71<br>47.91<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>11.16<br>12.54<br>12.23<br>11.98<br>DeepSeek-Math-7B-RL<br>18.81<br>17.58<br>16.06<br>17.48<br>NuminaMath-7B-CoT<br>20.49<br>20.80<br>22.02<br>21.10<br>Mathstral-7B-v0.1<br>16.36<br>17.89<br>17.89<br>17.38<br>Qwen2-Math-7B-Instruct<br>35.78<br>36.09<br>33.33<br>35.07<br>Qwen2-Math-72B-Instruct<br>48.47<br>49.39<br>51.22<br>49.69|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>**AAcc**<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**59.33**<br>**60.09**<br>**59.94**<br>**59.79**<br>GPT-4o-2024-08-06<br>50.00<br>49.69<br>49.24<br>49.64<br>GPT-4o-mini-2024-07-18<br>42.81<br>44.80<br>43.27<br>43.63<br>Claude-3-Opus-20240229<br>35.93<br>39.14<br>35.93<br>37.00<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>20.34<br>22.48<br>20.03<br>20.95<br>Mistral-7B-Instruct<br>5.05<br>3.98<br>4.59<br>4.54<br>Qwen2-7B-Instruct<br>25.38<br>25.23<br>26.15<br>25.59<br>LLaMA3-8B-Instruct<br>6.27<br>7.80<br>7.03<br>7.03<br>Yi-1.5-9B-Chat<br>28.75<br>28.75<br>29.51<br>29.00<br>Mistral-Nemo-Instruct-2407<br>13.61<br>12.39<br>10.55<br>12.18<br>DeepSeek-MOE-16B-Chat<br>1.68<br>0.46<br>1.22<br>1.12<br>DeepSeek-V2-Lite-Chat<br>5.05<br>4.89<br>3.52<br>4.49<br>Mistral-Small-Instruct-2409<br>27.98<br>29.51<br>27.68<br>28.39<br>Yi-1.5-34B-Chat<br>26.45<br>27.98<br>26.15<br>26.86<br>LLaMA3-70B-Instruct<br>20.79<br>23.39<br>20.95<br>21.71<br>Qwen2-72B-Instruct<br>35.02<br>37.92<br>35.32<br>36.09<br>Mistral-Large-Instruct-2407<br>46.94<br>49.08<br>47.71<br>47.91<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>11.16<br>12.54<br>12.23<br>11.98<br>DeepSeek-Math-7B-RL<br>18.81<br>17.58<br>16.06<br>17.48<br>NuminaMath-7B-CoT<br>20.49<br>20.80<br>22.02<br>21.10<br>Mathstral-7B-v0.1<br>16.36<br>17.89<br>17.89<br>17.38<br>Qwen2-Math-7B-Instruct<br>35.78<br>36.09<br>33.33<br>35.07<br>Qwen2-Math-72B-Instruct<br>48.47<br>49.39<br>51.22<br>49.69|**EAcc**<br>∆|**RE**|
|---|---|---|---|---|---|
|||||**48.32**<br>11.47<br>38.23<br>11.41<br>32.11<br>11.52<br>24.01<br>12.99|**23.74**<br>29.85<br>35.88<br>54.10|
|||||10.40<br>10.55<br>1.22<br>3.32<br>16.06<br>9.53<br>2.45<br>4.58<br>16.67<br>12.33|101.4<br>272.1<br>59.34<br>186.9<br>73.97|
||13.61<br>1.68<br>5.05<br>27.98<br>26.45|12.39<br>0.46<br>4.89<br>29.51<br>27.98|10.55<br>12.18<br>1.22<br>1.12<br>3.52<br>4.49<br>27.68<br>28.39<br>26.15<br>26.86|5.81<br>6.37<br>0.00<br>**1.12**<br>1.38<br>3.11<br>17.28<br>11.11<br>14.22<br>12.64|109.6<br>_∞_<br>225.4<br>64.29<br>88.89|
||20.79<br>35.02<br>46.94|23.39<br>37.92<br>49.08|20.95<br>21.71<br>35.32<br>36.09<br>47.71<br>47.91|12.23<br>9.48<br>23.55<br>12.54<br>36.70<br>11.21|77.51<br>53.25<br>30.54|
|||||5.35<br>6.63<br>8.56<br>8.92<br>11.47<br>9.63<br>9.63<br>7.75<br>24.77<br>10.30<br>39.45<br>10.24|123.9<br>104.2<br>83.96<br>80.48<br>41.58<br>25.96|



Table 17: **Main Results on Calculus - single variable** (all figures are in %). The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆<br>**RE**|
|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**72.81**<br>**73.32**<br>**72.61**<br>GPT-4o-2024-08-06<br>63.65<br>65.48<br>65.48<br>GPT-4o-mini-2024-07-18<br>54.68<br>57.43<br>55.91<br>Claude-3-Opus-20240229<br>47.96<br>51.32<br>50.61|||**72.91**<br>64.87<br>56.01<br>49.97|**59.37**<br>53.05<br>42.26<br>34.73|13.54<br>22.81<br>11.82<br>**22.28**<br>13.75<br>32.53<br>15.24<br>43.88|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>27.39<br>31.57<br>30.96<br>Mistral-7B-Instruct<br>8.86<br>10.29<br>8.96<br>Qwen2-7B-Instruct<br>39.51<br>41.96<br>39.41<br>LLaMA3-8B-Instruct<br>13.14<br>13.95<br>13.44<br>Yi-1.5-9B-Chat<br>37.27<br>37.88<br>38.19|||29.97<br>9.37<br>40.29<br>13.51<br>37.78|16.80<br>4.38<br>27.70<br>7.03<br>23.42|13.17<br>78.39<br>4.99<br>113.9<br>12.59<br>45.45<br>6.48<br>92.18<br>14.36<br>61.32|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|21.28<br>4.58<br>11.30<br>39.41<br>39.41|23.22<br>22.20<br>4.48<br>4.48<br>12.53<br>12.02<br>40.73<br>39.51<br>41.34<br>39.21|22.23<br>4.51<br>11.95<br>39.88<br>39.99|12.32<br>1.22<br>4.38<br>26.37<br>24.95|9.91<br>80.44<br>**3.29**<br>269.67<br>7.57<br>172.83<br>13.51<br>51.23<br>15.04<br>60.28|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|32.89<br>47.86<br>57.33|36.46<br>34.21<br>50.20<br>48.88<br>59.17<br>59.57|34.52<br>48.98<br>58.69|22.81<br>33.50<br>45.93|11.71<br>51.34<br>15.48<br>46.21<br>12.76<br>27.78|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>23.83<br>27.39<br>23.83<br>DeepSeek-Math-7B-RL<br>30.04<br>30.65<br>30.65<br>NuminaMath-7B-CoT<br>32.08<br>32.89<br>32.79<br>Mathstral-7B-v0.1<br>27.70<br>27.19<br>29.53<br>Qwen2-Math-7B-Instruct<br>46.03<br>49.49<br>49.49<br>Qwen2-Math-72B-Instruct<br>61.71<br>61.81<br>60.08|||25.02<br>30.45<br>32.59<br>28.14<br>48.34<br>61.20|13.75<br>20.06<br>20.16<br>16.29<br>35.74<br>49.29|11.27<br>81.96<br>10.39<br>51.79<br>12.43<br>61.66<br>11.85<br>72.74<br>12.60<br>35.25<br>11.91<br>24.16|



29 

Published as a conference paper at ICLR 2025 

Table 18: **Main Results on Combinatorics** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**<br>**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**90.91**<br>**92.05**<br>**95.45**<br>GPT-4o-2024-08-06<br>77.27<br>77.27<br>78.41<br>GPT-4o-mini-2024-07-18<br>70.45<br>69.32<br>73.86<br>Claude-3-Opus-20240229<br>56.82<br>65.91<br>68.18||||**92.80**<br>**88.64**<br>77.65<br>61.36<br>71.21<br>56.82<br>63.64<br>50.00|**4.16**<br>16.29<br>14.39<br>13.64|**4.69**<br>26.55<br>25.33<br>27.28|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>25.00<br>34.09<br>29.55<br>Mistral-7B-Instruct<br>10.23<br>12.50<br>11.36<br>Qwen2-7B-Instruct<br>39.77<br>45.45<br>44.32<br>LLaMA3-8B-Instruct<br>22.73<br>23.86<br>23.86<br>Yi-1.5-9B-Chat<br>32.95<br>43.18<br>44.32||||29.55<br>18.18<br>11.36<br>2.27<br>43.18<br>28.41<br>23.48<br>14.77<br>40.15<br>25.00|11.37<br>9.09<br>14.77<br>8.71<br>15.15|62.54<br>400.4<br>51.99<br>58.97<br>60.60|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|26.14<br>6.82<br>17.05<br>46.59<br>37.50|30.68<br>11.36<br>22.73<br>59.09<br>48.86|32.95<br>7.95<br>12.50<br>50.00<br>52.27|29.92<br>18.18<br>8.71<br>2.27<br>17.42<br>6.82<br>51.89<br>38.64<br>46.21<br>27.27|11.74<br>64.58<br>6.44<br>283.7<br>10.60<br>155.43<br>13.25<br>34.29<br>18.94<br>69.45||
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|47.73<br>60.23<br>69.32|52.27<br>63.64<br>72.73|45.45<br>60.23<br>71.59|48.48<br>37.50<br>61.36<br>47.73<br>71.21<br>57.95|10.98<br>13.63<br>13.26|29.28<br>28.56<br>22.88|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>26.14<br>37.50<br>27.27<br>DeepSeek-Math-7B-RL<br>30.68<br>42.05<br>39.77<br>NuminaMath-7B-CoT<br>30.68<br>35.23<br>37.50<br>Mathstral-7B-v0.1<br>34.09<br>38.64<br>35.23<br>Qwen2-Math-7B-Instruct<br>62.50<br>62.50<br>64.77<br>Qwen2-Math-72B-Instruct<br>76.14<br>76.14<br>75.00||||30.30<br>17.05<br>37.50<br>26.14<br>34.47<br>23.86<br>35.98<br>26.14<br>63.26<br>47.73<br>75.76<br>62.50|13.25<br>11.36<br>10.61<br>9.84<br>15.53<br>13.26|77.71<br>43.46<br>44.47<br>37.64<br>32.54<br>21.22|



Table 19: **Main Results on Complex analysis** (all figures are in %). The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**74.51**<br>**74.51**<br>**76.47**<br>GPT-4o-2024-08-06<br>66.67<br>70.59<br>70.59<br>GPT-4o-mini-2024-07-18<br>62.75<br>62.75<br>62.75<br>Claude-3-Opus-20240229<br>47.06<br>49.02<br>50.98<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>35.29<br>37.25<br>31.37<br>Mistral-7B-Instruct<br>11.76<br>11.76<br>9.80<br>Qwen2-7B-Instruct<br>45.10<br>50.98<br>43.14<br>LLaMA3-8B-Instruct<br>19.61<br>17.65<br>17.65<br>Yi-1.5-9B-Chat<br>39.22<br>39.22<br>39.22<br>Mistral-Nemo-Instruct-2407<br>29.41<br>25.49<br>25.49<br>DeepSeek-MOE-16B-Chat<br>1.96<br>3.92<br>1.96<br>DeepSeek-V2-Lite-Chat<br>17.65<br>13.73<br>15.69<br>Mistral-Small-Instruct-2409<br>47.06<br>49.02<br>43.14<br>Yi-1.5-34B-Chat<br>47.06<br>45.10<br>39.22<br>LLaMA3-70B-Instruct<br>43.14<br>33.33<br>37.25<br>Qwen2-72B-Instruct<br>52.94<br>62.75<br>62.75<br>Mistral-Large-Instruct-2407<br>64.71<br>64.71<br>60.78<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>35.29<br>29.41<br>27.45<br>DeepSeek-Math-7B-RL<br>35.29<br>41.18<br>47.06<br>NuminaMath-7B-CoT<br>33.33<br>35.29<br>37.25<br>Mathstral-7B-v0.1<br>37.25<br>29.41<br>31.37<br>Qwen2-Math-7B-Instruct<br>56.86<br>49.02<br>52.94<br>Qwen2-Math-72B-Instruct<br>70.59<br>72.55<br>72.55|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**74.51**<br>**74.51**<br>**76.47**<br>GPT-4o-2024-08-06<br>66.67<br>70.59<br>70.59<br>GPT-4o-mini-2024-07-18<br>62.75<br>62.75<br>62.75<br>Claude-3-Opus-20240229<br>47.06<br>49.02<br>50.98<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>35.29<br>37.25<br>31.37<br>Mistral-7B-Instruct<br>11.76<br>11.76<br>9.80<br>Qwen2-7B-Instruct<br>45.10<br>50.98<br>43.14<br>LLaMA3-8B-Instruct<br>19.61<br>17.65<br>17.65<br>Yi-1.5-9B-Chat<br>39.22<br>39.22<br>39.22<br>Mistral-Nemo-Instruct-2407<br>29.41<br>25.49<br>25.49<br>DeepSeek-MOE-16B-Chat<br>1.96<br>3.92<br>1.96<br>DeepSeek-V2-Lite-Chat<br>17.65<br>13.73<br>15.69<br>Mistral-Small-Instruct-2409<br>47.06<br>49.02<br>43.14<br>Yi-1.5-34B-Chat<br>47.06<br>45.10<br>39.22<br>LLaMA3-70B-Instruct<br>43.14<br>33.33<br>37.25<br>Qwen2-72B-Instruct<br>52.94<br>62.75<br>62.75<br>Mistral-Large-Instruct-2407<br>64.71<br>64.71<br>60.78<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>35.29<br>29.41<br>27.45<br>DeepSeek-Math-7B-RL<br>35.29<br>41.18<br>47.06<br>NuminaMath-7B-CoT<br>33.33<br>35.29<br>37.25<br>Mathstral-7B-v0.1<br>37.25<br>29.41<br>31.37<br>Qwen2-Math-7B-Instruct<br>56.86<br>49.02<br>52.94<br>Qwen2-Math-72B-Instruct<br>70.59<br>72.55<br>72.55|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**74.51**<br>**74.51**<br>**76.47**<br>GPT-4o-2024-08-06<br>66.67<br>70.59<br>70.59<br>GPT-4o-mini-2024-07-18<br>62.75<br>62.75<br>62.75<br>Claude-3-Opus-20240229<br>47.06<br>49.02<br>50.98<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>35.29<br>37.25<br>31.37<br>Mistral-7B-Instruct<br>11.76<br>11.76<br>9.80<br>Qwen2-7B-Instruct<br>45.10<br>50.98<br>43.14<br>LLaMA3-8B-Instruct<br>19.61<br>17.65<br>17.65<br>Yi-1.5-9B-Chat<br>39.22<br>39.22<br>39.22<br>Mistral-Nemo-Instruct-2407<br>29.41<br>25.49<br>25.49<br>DeepSeek-MOE-16B-Chat<br>1.96<br>3.92<br>1.96<br>DeepSeek-V2-Lite-Chat<br>17.65<br>13.73<br>15.69<br>Mistral-Small-Instruct-2409<br>47.06<br>49.02<br>43.14<br>Yi-1.5-34B-Chat<br>47.06<br>45.10<br>39.22<br>LLaMA3-70B-Instruct<br>43.14<br>33.33<br>37.25<br>Qwen2-72B-Instruct<br>52.94<br>62.75<br>62.75<br>Mistral-Large-Instruct-2407<br>64.71<br>64.71<br>60.78<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>35.29<br>29.41<br>27.45<br>DeepSeek-Math-7B-RL<br>35.29<br>41.18<br>47.06<br>NuminaMath-7B-CoT<br>33.33<br>35.29<br>37.25<br>Mathstral-7B-v0.1<br>37.25<br>29.41<br>31.37<br>Qwen2-Math-7B-Instruct<br>56.86<br>49.02<br>52.94<br>Qwen2-Math-72B-Instruct<br>70.59<br>72.55<br>72.55|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3<br>_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**74.51**<br>**74.51**<br>**76.47**<br>GPT-4o-2024-08-06<br>66.67<br>70.59<br>70.59<br>GPT-4o-mini-2024-07-18<br>62.75<br>62.75<br>62.75<br>Claude-3-Opus-20240229<br>47.06<br>49.02<br>50.98<br>_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>35.29<br>37.25<br>31.37<br>Mistral-7B-Instruct<br>11.76<br>11.76<br>9.80<br>Qwen2-7B-Instruct<br>45.10<br>50.98<br>43.14<br>LLaMA3-8B-Instruct<br>19.61<br>17.65<br>17.65<br>Yi-1.5-9B-Chat<br>39.22<br>39.22<br>39.22<br>Mistral-Nemo-Instruct-2407<br>29.41<br>25.49<br>25.49<br>DeepSeek-MOE-16B-Chat<br>1.96<br>3.92<br>1.96<br>DeepSeek-V2-Lite-Chat<br>17.65<br>13.73<br>15.69<br>Mistral-Small-Instruct-2409<br>47.06<br>49.02<br>43.14<br>Yi-1.5-34B-Chat<br>47.06<br>45.10<br>39.22<br>LLaMA3-70B-Instruct<br>43.14<br>33.33<br>37.25<br>Qwen2-72B-Instruct<br>52.94<br>62.75<br>62.75<br>Mistral-Large-Instruct-2407<br>64.71<br>64.71<br>60.78<br>_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>35.29<br>29.41<br>27.45<br>DeepSeek-Math-7B-RL<br>35.29<br>41.18<br>47.06<br>NuminaMath-7B-CoT<br>33.33<br>35.29<br>37.25<br>Mathstral-7B-v0.1<br>37.25<br>29.41<br>31.37<br>Qwen2-Math-7B-Instruct<br>56.86<br>49.02<br>52.94<br>Qwen2-Math-72B-Instruct<br>70.59<br>72.55<br>72.55|**AAcc**|**EAcc**<br>∆|**RE**|
|---|---|---|---|---|---|---|
|||||**75.16**<br>69.28<br>62.75<br>49.02|**64.71**<br>10.45<br>58.82<br>10.46<br>50.98<br>11.77<br>37.25<br>11.77|16.15<br>17.78<br>23.09<br>31.60|
|||||34.64<br>11.11<br>46.41<br>18.30<br>39.22|23.53<br>11.11<br>5.88<br>5.23<br>35.29<br>11.12<br>13.73<br>4.57<br>27.45<br>11.77|47.22<br>88.95<br>31.51<br>33.28<br>42.88|
||29.41<br>1.96<br>17.65<br>47.06<br>47.06|25.49<br>3.92<br>13.73<br>49.02<br>45.10|25.49<br>1.96<br>15.69<br>43.14<br>39.22|26.80<br>2.61<br>15.69<br>46.41<br>43.79|19.61<br>7.19<br>0.00<br>**2.61**<br>7.84<br>7.85<br>37.25<br>9.16<br>25.49<br>18.30|36.66<br>_∞_<br>100.1<br>24.59<br>71.79|
||43.14<br>52.94<br>64.71|33.33<br>62.75<br>64.71|37.25<br>62.75<br>60.78|37.91<br>59.48<br>63.40|25.49<br>12.42<br>47.06<br>12.42<br>52.94<br>10.46|48.72<br>26.39<br>19.76|
|||||30.72<br>41.18<br>35.29<br>32.68<br>52.94<br>71.90|23.53<br>7.19<br>27.45<br>13.73<br>21.57<br>13.72<br>15.69<br>16.99<br>41.18<br>11.76<br>64.71<br>7.19|30.56<br>50.02<br>63.61<br>108.3<br>28.56<br>**11.11**|



30 

Published as a conference paper at ICLR 2025 

Table 20: **Main Results on Differential equations** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**<br>**EAcc**<br>∆|**RE**|
|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**29.84**<br>**29.18**<br>**29.51**<br>GPT-4o-2024-08-06<br>23.93<br>24.59<br>25.57<br>GPT-4o-mini-2024-07-18<br>20.00<br>22.30<br>19.67<br>Claude-3-Opus-20240229<br>18.69<br>18.03<br>20.33||||**29.51**<br>**22.62**<br>6.89<br>24.70<br>19.34<br>5.36<br>20.66<br>16.07<br>4.59<br>19.02<br>13.11<br>5.91|30.46<br>**27.71**<br>28.56<br>45.08|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>6.23<br>9.51<br>4.59<br>Mistral-7B-Instruct<br>3.28<br>3.93<br>4.59<br>Qwen2-7B-Instruct<br>11.80<br>12.13<br>11.48<br>LLaMA3-8B-Instruct<br>4.26<br>5.90<br>5.57<br>Yi-1.5-9B-Chat<br>11.80<br>13.44<br>12.13||||6.78<br>2.95<br>3.83<br>3.93<br>1.64<br>2.29<br>11.80<br>7.54<br>4.26<br>5.25<br>2.62<br>2.63<br>12.46<br>6.23<br>6.23|129.8<br>139.6<br>56.50<br>100.4<br>100.0|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|8.52<br>0.66<br>2.62<br>15.74<br>12.13|7.87<br>1.31<br>2.62<br>15.74<br>15.08|9.18<br>0.00<br>1.97<br>14.43<br>12.13|8.52<br>4.26<br>4.26<br>0.66<br>0.00<br>**0.66**<br>2.40<br>0.33<br>2.07<br>15.30<br>9.84<br>5.46<br>13.11<br>7.21<br>5.90|100.0<br>_∞_<br>627.3<br>55.49<br>81.83|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|12.79<br>19.34<br>22.95|13.44<br>20.00<br>25.90|12.13<br>18.36<br>24.92|12.79<br>7.54<br>5.25<br>19.23<br>13.77<br>5.46<br>24.59<br>18.69<br>5.90|69.63<br>39.65<br>31.57|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>6.89<br>7.54<br>5.90<br>DeepSeek-Math-7B-RL<br>8.52<br>10.82<br>9.84<br>NuminaMath-7B-CoT<br>10.82<br>10.16<br>7.87<br>Mathstral-7B-v0.1<br>10.49<br>10.49<br>9.51<br>Qwen2-Math-7B-Instruct<br>15.41<br>18.69<br>18.03<br>Qwen2-Math-72B-Instruct<br>22.95<br>22.95<br>22.95||||6.78<br>1.97<br>4.81<br>9.73<br>5.25<br>4.48<br>9.62<br>4.26<br>5.36<br>10.16<br>4.92<br>5.24<br>17.38<br>12.13<br>5.25<br>22.95<br>17.38<br>5.57|244.2<br>85.33<br>125.8<br>106.5<br>43.28<br>32.05|



31 

Published as a conference paper at ICLR 2025 

Table 21: **Main Results on Financial mathematics** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**<br>∆|**RE**|
|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**46.53**<br>**49.13**<br>**51.16**<br>GPT-4o-2024-08-06<br>28.61<br>29.77<br>29.77<br>GPT-4o-mini-2024-07-18<br>16.19<br>17.63<br>19.65<br>Claude-3-Opus-20240229<br>19.65<br>21.39<br>20.52||||**48.94**<br>29.38<br>17.82<br>20.52|**28.32**<br>20.62<br>18.21<br>11.17<br>9.54<br>8.28<br>10.98<br>9.54|72.81<br>61.34<br>86.79<br>86.89|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>5.49<br>6.94<br>6.94<br>Mistral-7B-Instruct<br>1.16<br>3.47<br>2.02<br>Qwen2-7B-Instruct<br>8.67<br>12.43<br>8.96<br>LLaMA3-8B-Instruct<br>2.60<br>2.02<br>3.18<br>Yi-1.5-9B-Chat<br>9.25<br>14.45<br>12.43||||6.45<br>2.22<br>10.02<br>2.60<br>12.04|3.18<br>3.27<br>0.87<br>1.35<br>3.76<br>6.26<br>1.16<br>1.44<br>4.05<br>7.99|102.8<br>155.2<br>166.5<br>124.1<br>197.2|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|4.62<br>0.58<br>3.76<br>12.43<br>11.85|5.49<br>0.87<br>3.76<br>11.85<br>14.45|6.07<br>1.16<br>3.76<br>13.01<br>13.58|5.39<br>0.87<br>3.76<br>12.43<br>13.29|1.73<br>3.66<br>0.00<br>**0.87**<br>1.45<br>2.31<br>5.78<br>6.65<br>5.20<br>8.09|211.6<br>_∞_<br>159.3<br>115.1<br>155.6|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|8.09<br>15.03<br>22.54|10.12<br>18.50<br>26.59|9.25<br>16.47<br>21.97|9.15<br>16.67<br>23.70|4.62<br>4.53<br>8.96<br>7.71<br>15.32<br>8.38|98.05<br>86.05<br>**54.70**|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>4.91<br>6.07<br>4.62<br>DeepSeek-Math-7B-RL<br>5.49<br>7.51<br>5.78<br>NuminaMath-7B-CoT<br>8.67<br>9.25<br>9.25<br>Mathstral-7B-v0.1<br>4.91<br>9.83<br>7.23<br>Qwen2-Math-7B-Instruct<br>10.69<br>17.05<br>15.32<br>Qwen2-Math-72B-Instruct<br>28.90<br>26.88<br>29.19||||5.20<br>6.26<br>9.06<br>7.32<br>14.35<br>28.32|2.31<br>2.89<br>125.1<br>2.02<br>4.24<br>209.9<br>3.76<br>5.30<br>141.0<br>3.18<br>4.14<br>130.2<br>5.49<br>8.86<br>161.38<br>16.47<br>11.85<br>71.95||



32 

Published as a conference paper at ICLR 2025 

Table 22: **Main Results on Geometry** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**<br>∆<br>**RE**|
|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**71.43**<br>**67.08**<br>**71.43**<br>GPT-4o-2024-08-06<br>64.60<br>63.98<br>63.98<br>GPT-4o-mini-2024-07-18<br>60.87<br>55.90<br>59.63<br>Claude-3-Opus-20240229<br>57.76<br>49.69<br>49.07||||**69.98**<br>64.18<br>58.80<br>52.17|**58.39**<br>11.59<br>19.85<br>51.55<br>12.63<br>24.50<br>44.10<br>14.70<br>33.33<br>37.89<br>14.28<br>37.69|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>32.92<br>27.33<br>30.43<br>Mistral-7B-Instruct<br>8.70<br>8.70<br>6.83<br>Qwen2-7B-Instruct<br>50.31<br>39.75<br>42.86<br>LLaMA3-8B-Instruct<br>24.84<br>19.88<br>19.88<br>Yi-1.5-9B-Chat<br>38.51<br>37.89<br>36.02||||30.23<br>8.07<br>44.31<br>21.53<br>37.47|18.01<br>12.22<br>67.85<br>3.11<br>4.96<br>159.5<br>29.19<br>15.12<br>51.80<br>9.94<br>11.59<br>116.6<br>22.36<br>15.11<br>67.58|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|31.68<br>4.97<br>11.18<br>44.72<br>43.48|27.95<br>3.73<br>14.29<br>47.21<br>37.89|28.57<br>6.21<br>14.91<br>46.58<br>44.72|29.40<br>4.97<br>13.46<br>46.17<br>42.03|16.77<br>12.63<br>75.31<br>1.86<br>**3.11**<br>167.2<br>4.97<br>8.49<br>170.8<br>32.92<br>13.25<br>40.25<br>27.33<br>14.70<br>53.79|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|39.13<br>59.01<br>61.49|38.51<br>49.69<br>57.76|39.75<br>53.42<br>60.25|39.13<br>54.04<br>59.83|25.47<br>13.66<br>53.63<br>40.99<br>13.05<br>31.84<br>50.31<br>9.52<br>**18.92**|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>29.19<br>24.22<br>24.22<br>DeepSeek-Math-7B-RL<br>34.78<br>29.19<br>29.81<br>NuminaMath-7B-CoT<br>33.54<br>31.68<br>28.57<br>Mathstral-7B-v0.1<br>39.13<br>32.92<br>37.89<br>Qwen2-Math-7B-Instruct<br>52.80<br>42.86<br>45.96<br>Qwen2-Math-72B-Instruct<br>67.08<br>61.49<br>64.60||||25.88<br>31.26<br>31.26<br>36.65<br>47.20<br>64.39|14.29<br>11.59<br>81.11<br>19.25<br>12.01<br>62.39<br>19.25<br>12.01<br>62.39<br>24.22<br>12.43<br>51.32<br>34.16<br>13.04<br>38.17<br>53.42<br>10.97<br>20.54|



Table 23: **Main Results on Linear algebra** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**74.90**<br>**74.90**<br>**75.90**<br>GPT-4o-2024-08-06<br>63.86<br>61.65<br>61.65<br>GPT-4o-mini-2024-07-18<br>57.63<br>57.03<br>56.22<br>Claude-3-Opus-20240229<br>51.20<br>48.80<br>49.00||||**75.23**<br>62.38<br>56.96<br>49.67|**65.86**<br>51.61<br>43.37<br>37.15|9.37<br>10.77<br>13.59<br>12.52|**14.23**<br>20.87<br>31.34<br>33.70|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>23.69<br>23.69<br>21.49<br>Mistral-7B-Instruct<br>8.03<br>7.43<br>7.23<br>Qwen2-7B-Instruct<br>36.35<br>34.54<br>32.73<br>LLaMA3-8B-Instruct<br>13.25<br>12.45<br>12.65<br>Yi-1.5-9B-Chat<br>34.14<br>31.93<br>31.93||||22.96<br>7.56<br>34.54<br>12.78<br>32.66|12.45<br>2.61<br>22.29<br>6.43<br>18.47|10.51<br>4.95<br>12.25<br>6.35<br>14.19|84.42<br>189.7<br>54.96<br>98.76<br>76.83|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|23.69<br>2.01<br>9.24<br>43.17<br>35.34|20.48<br>2.41<br>8.23<br>38.35<br>31.93|23.90<br>3.61<br>7.43<br>41.57<br>34.94|22.69<br>2.68<br>8.30<br>41.03<br>34.07|11.24<br>0.80<br>1.81<br>28.51<br>20.08|11.45<br>**1.88**<br>6.49<br>12.52<br>13.99|101.9<br>235.0<br>358.6<br>43.91<br>69.67|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|34.34<br>52.81<br>62.25|30.72<br>51.00<br>59.04|29.32<br>49.80<br>58.43|31.46<br>51.20<br>59.91|19.28<br>38.15<br>48.59|12.18<br>13.05<br>11.32|63.17<br>34.21<br>23.30|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>20.08<br>21.08<br>18.07<br>DeepSeek-Math-7B-RL<br>26.91<br>26.51<br>23.09<br>NuminaMath-7B-CoT<br>25.70<br>25.50<br>27.11<br>Mathstral-7B-v0.1<br>29.92<br>23.90<br>24.70<br>Qwen2-Math-7B-Instruct<br>46.39<br>43.17<br>43.57<br>Qwen2-Math-72B-Instruct<br>61.45<br>59.24<br>62.85||||19.75<br>25.50<br>26.10<br>26.17<br>44.38<br>61.18|9.44<br>15.46<br>14.66<br>13.86<br>32.73<br>47.59|10.31<br>10.04<br>11.44<br>12.31<br>11.65<br>13.59|109.2<br>64.94<br>78.04<br>88.82<br>35.59<br>28.56|



33 

Published as a conference paper at ICLR 2025 

Table 24: **Main Results on Number theory** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**80.43**<br>**76.09**<br>**73.91**<br>GPT-4o-2024-08-06<br>60.87<br>63.04<br>67.39<br>GPT-4o-mini-2024-07-18<br>60.87<br>58.70<br>58.70<br>Claude-3-Opus-20240229<br>47.83<br>54.35<br>56.52||||**76.81**<br>63.77<br>59.42<br>52.90|**58.70**<br>56.52<br>52.17<br>39.13|18.11<br>7.25<br>7.25<br>13.77|30.85<br>**12.83**<br>13.90<br>35.19|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>26.09<br>26.09<br>19.57<br>Mistral-7B-Instruct<br>6.52<br>6.52<br>8.70<br>Qwen2-7B-Instruct<br>34.78<br>39.13<br>41.30<br>LLaMA3-8B-Instruct<br>19.57<br>21.74<br>15.22<br>Yi-1.5-9B-Chat<br>34.78<br>32.61<br>34.78||||23.91<br>7.25<br>38.41<br>18.84<br>34.06|13.04<br>2.17<br>23.91<br>8.70<br>23.91|10.87<br>5.08<br>14.50<br>10.14<br>10.15|83.36<br>234.1<br>60.64<br>116.6<br>42.45|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|17.39<br>6.52<br>15.22<br>45.65<br>36.96|28.26<br>6.52<br>13.04<br>34.78<br>34.78|28.26<br>2.17<br>15.22<br>41.30<br>47.83|24.64<br>5.07<br>14.49<br>40.58<br>39.86|8.70<br>2.17<br>4.35<br>30.43<br>26.09|15.94<br>**2.90**<br>10.14<br>10.15<br>13.77|183.2<br>133.6<br>233.1<br>33.36<br>52.78|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|30.43<br>47.83<br>58.70|23.91<br>58.70<br>58.70|36.96<br>43.48<br>58.70|30.43<br>50.00<br>58.70|15.22<br>36.96<br>45.65|15.21<br>13.04<br>13.05|99.93<br>35.28<br>28.59|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>23.91<br>28.26<br>30.43<br>DeepSeek-Math-7B-RL<br>21.74<br>26.09<br>26.09<br>NuminaMath-7B-CoT<br>17.39<br>32.61<br>32.61<br>Mathstral-7B-v0.1<br>28.26<br>30.43<br>30.43<br>Qwen2-Math-7B-Instruct<br>39.13<br>43.48<br>43.48<br>Qwen2-Math-72B-Instruct<br>60.87<br>54.35<br>63.04||||27.54<br>24.64<br>27.54<br>29.71<br>42.03<br>59.42|15.22<br>13.04<br>13.04<br>17.39<br>30.43<br>50.00|12.32<br>11.60<br>14.50<br>12.32<br>11.60<br>9.42|80.95<br>88.96<br>111.2<br>70.85<br>38.12<br>18.84|



Table 25: **Main Results on Probability** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**66.07**<br>**68.45**<br>**68.15**<br>GPT-4o-2024-08-06<br>66.07<br>66.07<br>65.77<br>GPT-4o-mini-2024-07-18<br>50.89<br>56.85<br>52.38<br>Claude-3-Opus-20240229<br>56.55<br>59.23<br>59.82||||**67.56**<br>65.97<br>53.37<br>58.53|51.49<br>**54.76**<br>39.88<br>44.94|16.07<br>11.21<br>13.49<br>13.59|31.21<br>**20.47**<br>33.83<br>30.24|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>20.24<br>22.62<br>21.43<br>Mistral-7B-Instruct<br>10.12<br>11.01<br>10.42<br>Qwen2-7B-Instruct<br>31.85<br>39.58<br>33.63<br>LLaMA3-8B-Instruct<br>16.37<br>16.96<br>15.48<br>Yi-1.5-9B-Chat<br>28.27<br>30.06<br>31.85||||21.43<br>10.52<br>35.02<br>16.27<br>30.06|10.42<br>3.57<br>20.54<br>9.23<br>13.69|11.01<br>6.95<br>14.48<br>7.04<br>16.37|105.7<br>194.7<br>70.50<br>76.27<br>119.6|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|23.81<br>5.65<br>9.82<br>39.58<br>37.20|26.79<br>4.76<br>11.61<br>43.15<br>41.67|25.00<br>4.17<br>10.42<br>43.45<br>38.39|25.20<br>4.86<br>10.62<br>42.06<br>39.09|15.18<br>1.19<br>3.27<br>27.68<br>23.51|10.02<br>**3.67**<br>7.35<br>14.38<br>15.58|66.01<br>308.4<br>224.8<br>51.95<br>66.27|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|35.12<br>48.51<br>59.82|36.31<br>48.51<br>60.12|34.23<br>44.05<br>58.33|35.22<br>47.02<br>59.42|22.02<br>33.04<br>45.54|13.20<br>13.98<br>13.88|59.95<br>42.31<br>30.48|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>21.73<br>22.92<br>21.43<br>DeepSeek-Math-7B-RL<br>24.40<br>25.89<br>26.19<br>NuminaMath-7B-CoT<br>25.60<br>26.49<br>25.30<br>Mathstral-7B-v0.1<br>27.38<br>29.46<br>25.30<br>Qwen2-Math-7B-Instruct<br>36.61<br>44.35<br>39.88<br>Qwen2-Math-72B-Instruct<br>58.33<br>62.20<br>59.52||||22.02<br>25.50<br>25.79<br>27.38<br>40.28<br>60.02|10.71<br>14.88<br>13.39<br>14.88<br>25.89<br>44.05|11.31<br>10.62<br>12.40<br>12.50<br>14.39<br>15.97|105.6<br>71.37<br>92.61<br>84.01<br>55.58<br>36.25|



34 

Published as a conference paper at ICLR 2025 

Table 26: **Main Results on Set theory and logic** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**78.26**<br>73.91<br>**79.71**<br>GPT-4o-2024-08-06<br>75.36<br>**76.81**<br>68.12<br>GPT-4o-mini-2024-07-18<br>59.42<br>57.97<br>59.42<br>Claude-3-Opus-20240229<br>56.52<br>60.87<br>62.32||||**77.29**<br>73.43<br>58.94<br>59.90|**66.67**<br>59.42<br>40.58<br>43.48|10.62<br>14.01<br>18.36<br>16.42|**15.93**<br>23.58<br>45.24<br>37.76|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>14.49<br>14.49<br>18.84<br>Mistral-7B-Instruct<br>2.90<br>5.80<br>5.80<br>Qwen2-7B-Instruct<br>20.29<br>36.23<br>24.64<br>LLaMA3-8B-Instruct<br>8.70<br>15.94<br>8.70<br>Yi-1.5-9B-Chat<br>24.64<br>18.84<br>26.09||||15.94<br>4.83<br>27.05<br>11.11<br>23.19|7.25<br>0.00<br>13.04<br>5.80<br>5.80|8.69<br>4.83<br>14.01<br>5.31<br>17.39|119.9<br>_∞_<br>107.4<br>91.55<br>299.8|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|24.64<br>2.90<br>4.35<br>47.83<br>30.43|33.33<br>1.45<br>7.25<br>44.93<br>40.58|26.09<br>1.45<br>5.80<br>40.58<br>43.48|28.02<br>1.93<br>5.80<br>44.44<br>38.16|18.84<br>0.00<br>1.45<br>30.43<br>21.74|9.18<br>**1.93**<br>4.35<br>14.01<br>16.42|48.73<br>_∞_<br>300.0<br>46.04<br>75.53|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|28.99<br>52.17<br>59.42|26.09<br>55.07<br>60.87|24.64<br>49.28<br>57.97|26.57<br>52.17<br>59.42|15.94<br>36.23<br>44.93|10.63<br>15.94<br>14.49|66.69<br>44.00<br>32.25|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>21.74<br>18.84<br>13.04<br>DeepSeek-Math-7B-RL<br>23.19<br>23.19<br>24.64<br>NuminaMath-7B-CoT<br>17.39<br>26.09<br>18.84<br>Mathstral-7B-v0.1<br>23.19<br>26.09<br>20.29<br>Qwen2-Math-7B-Instruct<br>31.88<br>39.13<br>27.54<br>Qwen2-Math-72B-Instruct<br>60.87<br>56.52<br>55.07||||17.87<br>23.67<br>20.77<br>23.19<br>32.85<br>57.49|8.70<br>10.14<br>8.70<br>14.49<br>18.84<br>40.58|9.17<br>13.53<br>12.07<br>8.70<br>14.01<br>16.91|105.4<br>133.4<br>138.7<br>60.04<br>74.36<br>41.67|



Table 27: **Main Results on Statistics** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**<br>∆|**RE**|
|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**64.09**<br>**62.34**<br>**61.85**<br>GPT-4o-2024-08-06<br>59.10<br>62.09<br>60.10<br>GPT-4o-mini-2024-07-18<br>43.14<br>46.38<br>48.63<br>Claude-3-Opus-20240229<br>52.12<br>53.12<br>50.37||||**62.76**<br>60.43<br>46.05<br>51.87|**49.88**<br>12.88<br>49.38<br>11.05<br>32.92<br>13.13<br>37.16<br>14.71|25.82<br>**22.38**<br>39.88<br>39.59|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>15.96<br>16.96<br>21.45<br>Mistral-7B-Instruct<br>11.97<br>11.72<br>11.47<br>Qwen2-7B-Instruct<br>26.68<br>27.93<br>29.18<br>LLaMA3-8B-Instruct<br>10.72<br>13.47<br>16.21<br>Yi-1.5-9B-Chat<br>26.43<br>23.94<br>31.67||||18.12<br>11.72<br>27.93<br>13.47<br>27.35|6.23<br>11.89<br>2.99<br>8.73<br>13.97<br>13.96<br>5.99<br>7.48<br>13.97<br>13.38|190.9<br>291.0<br>99.93<br>124.9<br>95.78|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|22.94<br>3.24<br>8.98<br>37.66<br>32.92|27.18<br>3.49<br>8.48<br>35.91<br>37.41|24.94<br>4.74<br>9.23<br>35.66<br>37.91|25.02<br>3.82<br>8.89<br>36.41<br>36.08|14.71<br>10.31<br>1.00<br>**2.82**<br>2.74<br>6.15<br>23.69<br>12.72<br>21.45<br>14.63|70.09<br>282.0<br>224.5<br>53.69<br>68.21|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|22.44<br>43.14<br>54.36|20.70<br>44.39<br>50.12|21.45<br>44.89<br>53.62|21.53<br>44.14<br>52.70|11.97<br>9.56<br>31.42<br>12.72<br>39.90<br>12.80|79.87<br>40.48<br>32.08|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>18.95<br>16.96<br>18.95<br>DeepSeek-Math-7B-RL<br>21.70<br>24.44<br>22.69<br>NuminaMath-7B-CoT<br>17.71<br>18.20<br>18.95<br>Mathstral-7B-v0.1<br>19.70<br>22.69<br>21.70<br>Qwen2-Math-7B-Instruct<br>31.42<br>31.17<br>37.66<br>Qwen2-Math-72B-Instruct<br>47.38<br>50.87<br>47.63||||18.29<br>22.94<br>18.29<br>21.36<br>33.42<br>48.63|6.73<br>11.56<br>9.23<br>13.71<br>6.48<br>11.81<br>9.73<br>11.63<br>17.71<br>15.71<br>33.92<br>14.71|171.8<br>148.5<br>182.3<br>119.5<br>88.71<br>43.37|



35 

Published as a conference paper at ICLR 2025 

Table 28: **Main Results on Trigonometry** (all figures are in %). Models are classified into three categories according to their purpose and origin. The best results within each column are **bolded** and the best results of open-source Chat LLMs within a similar parameter size group are underlined. 

|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**Models**<br>**Acc**1<br>**Acc**2<br>**Acc**3|**AAcc**|**EAcc**|∆|**RE**|
|---|---|---|---|---|---|---|---|
|_Closed-source LLMs_<br>OpenAI-o1-mini-2024-09-12<br>**75.84**<br>**74.16**<br>**69.10**<br>GPT-4o-2024-08-06<br>67.98<br>74.16<br>66.29<br>GPT-4o-mini-2024-07-18<br>52.25<br>58.99<br>55.06<br>Claude-3-Opus-20240229<br>53.93<br>52.81<br>53.37||||**73.03**<br>69.48<br>55.43<br>53.37|**56.74**<br>51.69<br>37.64<br>37.64|16.29<br>17.79<br>17.79<br>15.73|28.71<br>34.42<br>47.26<br>41.79|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>24.16<br>24.72<br>20.79<br>Mistral-7B-Instruct<br>6.74<br>7.87<br>6.18<br>Qwen2-7B-Instruct<br>36.52<br>41.01<br>42.13<br>LLaMA3-8B-Instruct<br>14.61<br>12.92<br>13.48<br>Yi-1.5-9B-Chat<br>32.02<br>30.34<br>32.02||||23.22<br>6.93<br>39.89<br>13.67<br>31.46|11.24<br>1.69<br>25.84<br>4.49<br>15.73|11.98<br>5.24<br>14.05<br>9.18<br>15.73|106.6<br>310.0<br>54.37<br>204.5<br>100.0|
|Mistral-Nemo-Instruct-2407<br>DeepSeek-MOE-16B-Chat<br>DeepSeek-V2-Lite-Chat<br>Mistral-Small-Instruct-2409<br>Yi-1.5-34B-Chat|23.03<br>3.93<br>10.67<br>38.76<br>44.38|23.60<br>3.93<br>15.73<br>41.01<br>39.33|24.72<br>4.49<br>11.80<br>45.51<br>37.64|23.78<br>4.12<br>12.73<br>41.76<br>40.45|11.80<br>0.00<br>3.37<br>27.53<br>25.28|11.98<br>**4.12**<br>9.36<br>14.23<br>15.17|101.5<br>_∞_<br>277.7<br>51.69<br>60.01|
|LLaMA3-70B-Instruct<br>Qwen2-72B-Instruct<br>Mistral-Large-Instruct-2407|33.71<br>52.25<br>65.73|36.52<br>50.56<br>64.61|34.83<br>50.00<br>66.85|35.02<br>50.94<br>65.73|23.03<br>35.39<br>50.00|11.99<br>15.55<br>15.73|52.06<br>43.94<br>31.46|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>26.40<br>28.09<br>29.78<br>DeepSeek-Math-7B-RL<br>32.02<br>33.71<br>36.52<br>NuminaMath-7B-CoT<br>33.71<br>34.83<br>30.34<br>Mathstral-7B-v0.1<br>30.34<br>31.46<br>34.27<br>Qwen2-Math-7B-Instruct<br>53.93<br>50.00<br>51.69<br>Qwen2-Math-72B-Instruct<br>62.92<br>65.17<br>63.48||||28.09<br>34.08<br>32.96<br>32.02<br>51.87<br>63.86|11.24<br>20.22<br>17.42<br>18.54<br>38.76<br>50.00|16.85<br>13.86<br>15.54<br>13.48<br>13.11<br>13.86|149.9<br>68.55<br>89.21<br>72.71<br>33.82<br>**27.72**|



36 

Published as a conference paper at ICLR 2025 

Table 29: An example of Linear algebra with expression calculation error. 

**Problem** : You have boxes of five types. They are characterized by their length _L_ , their width _W_ , their height _H_ , their age _A_ , and their price _P_ , as given in the following table: 

type _L W H A P I_ 1 1 1 1 1 _II_ 1 1 2 3 2 _III_ 1 2 3 2 2 _IV_ 1 2 4 1 2 _V_ 1 2 5 1 3 

So for example, boxes of type IV measure 1 by 2 by 4 feet, are a year old, and cost $2 each. Suppose the sum of the lengths of your boxes is 25, the sum of their widths is 40, the sum of their heights is 75, the sum of their ages is 43, and the total price of those boxes is 51. You have [ANS] boxes of type I, [ANS] boxes of type II, [ANS] boxes of type III, [ANS] boxes of type IV, [ANS] boxes of type V. 

**o1-mini Completion** : To solve the system of equations based on the given constraints, we define the variables as follows: 

 _xx_ 12 = Number= Number ofof TypeType IIIboxesboxes _x_ 3 = Number of Type III boxes  _xx_ 45 = Number= Number ofof TypeType IVV boxesboxes 

Using the provided data, we set up the following equations: 

1 _. x_ 1 + _x_ 2 + _x_ 3 + _x_ 4 + _x_ 5 = 25 2 _. x_ 1 + _x_ 2 + 2 _x_ 3 + 2 _x_ 4 + 2 _x_ 5 = 40 3 _. x_ 1 + 2 _x_ 2 + 3 _x_ 3 + 4 _x_ 4 + 5 _x_ 5 = 75 4 _. x_ 1 + 3 _x_ 2 + 2 _x_ 3 + _x_ 4 + _x_ 5 = 43 5 _. x_ 1 + 2 _x_ 2 + 2 _x_ 3 + 2 _x_ 4 + 3 _x_ 5 = 51 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0037-09.png)


[It skipped detailed steps and obtained incorrect results in equation manipulation] **The final answers are 1 _._ 5 _,_ 8 _._ 5 _,_ 3 _._ 5 _,_ 9 _,_ 2 _._ 5 ** 

**Ground-Truth Answer** : 4, 6, 6, 4, 5 

**Model Output** : 1.5, 8.5, 3.5, 9, 2.5 

37 

Published as a conference paper at ICLR 2025 

Table 30: An example of Financial mathematics with numeric calculation error. 

**Problem** : How long will it take for $3100 compounded semiannually at an annual rate of 1% to amount to $7000? [ANS] years 

**o1-mini Completion** : To determine how long it will take for $3100 to grow to $7000 with semiannual compounding at an annual rate of 1%, we can use the compound interest formula: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0038-04.png)


Where: - _A_ = 7000 (the final amount) - _P_ = 3100 (the principal amount) - _r_ = 0 _._ 01 (the annual interest rate) - _n_ = 2 (the number of compounding periods per year) - _t_ is the time in years Plugging in the known values: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0038-06.png)


Taking the natural logarithm of both sides: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0038-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0038-09.png)


Calculating the values: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0038-11.png)


**Ground-Truth Answer** : 81.6542614995239 

**Model Output** : 81.56 

38 

Published as a conference paper at ICLR 2025 

Table 31: An example of Financial mathematics with flawed reasoning. 

**Problem** : If you make a deposit into a bank account, at what interest rate (compounded weekly) should you invest if you would like to double your investment in 78 weeks? [ANS] % (Note: Your answer should be accurate to two decimal places) 

**o1-mini Completion** : To determine the interest rate _r_ (compounded weekly) required to double your investment in 78 weeks, we can use the compound interest formula: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0039-04.png)


Where: - _FV_ is the future value - _PV_ is the present value - _r_ is the weekly interest rate (in percentage) - _n_ is the number of weeks Given that you want to double your investment: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0039-06.png)


To solve for _r_ , take the natural logarithm of both sides: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0039-08.png)


Calculating the above: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0039-10.png)


The final answer is 0 _._ 89 . 

**Ground-Truth Answer** : 46.4157 **Model Output** : 0.89 

39 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0040-01.png)


**----- Start of picture text -----**<br>
o1-mini-2024-09-12 gpt-4o-mini-2024-07-18<br>Integers 95.0% Integers 92.5%<br>Limits and continuity 82.4% Limits and continuity 64.7%<br>Systems of linear equations 75.7% Linear equations and functions 61.7%<br>Linear equations and functions 67.8% Differentiation 57.5%<br>Differentiation 64.7% Systems of linear equations 55.6%<br>Integration of multivariable functions 63.8% Sample Space 50.3%<br>Sample Space 60.1% Vector geometry 42.5%<br>Applications of differentiation 58.9% Applications of differentiation 40.8%<br>Infinite sequences and series 57.2% Infinite sequences and series 39.9%<br>Vector geometry 54.8% Differentiation of multivariable functions 35.2%<br>Applications of integration 54.7% Continuous distributions 34.7%<br>Vector calculus 51.9% Vector calculus 32.1%<br>Confidence intervals 49.4% Calculus of vector valued functions 32.0%<br>Continuous distributions 44.9% Confidence intervals 31.3%<br>Differentiation of multivariable functions 44.7% Level 1 Integration of multivariable functions 30.4% Level 1<br>Interest 39.3% Level 2 Applications of integration 28.4% Level 2<br>Hypothesis tests 34.5% Level 3 Hypothesis tests 19.8% Level 3<br>Calculus of vector valued functions 28.9% Level 4 First order differential equations 15.1% Level 4<br>First order differential equations 19.9% Level 5 Interest 13.5% Level 5<br>Annuities 10.8% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>gpt-4o-2024-08-06 Mistral-Large-Instruct-2407<br>Integers 91.7% Integers 89.2%<br>Limits and continuity 82.4% Limits and continuity 67.1%<br>Sample Space 74.1% Linear equations and functions 66.1%<br>Linear equations and functions 67.0% Systems of linear equations 61.1%<br>Systems of linear equations 66.0% Differentiation 60.8%<br>Differentiation 64.7% Sample Space 57.3%<br>Applications of differentiation 52.8% Vector geometry 47.3%<br>Vector geometry 48.6% Applications of differentiation 43.8%<br>Infinite sequences and series 47.1% Infinite sequences and series 40.6%<br>Confidence intervals 47.0% Integration of multivariable functions 39.9%<br>Applications of integration 45.3% Applications of integration 37.2%<br>Continuous distributions 42.9% Vector calculus 36.8%<br>Integration of multivariable functions 41.3% Confidence intervals 34.9%<br>Hypothesis tests 39.7% Continuous distributions 34.7%<br>Differentiation of multivariable functions 36.3% Level 1 Differentiation of multivariable functions 33.5% Level 1<br>Vector calculus 35.8% Level 2 Hypothesis tests 31.9% Level 2<br>Calculus of vector valued functions 32.0% Level 3 Calculus of vector valued functions 28.9% Level 3<br>Interest 28.8% Level 4 Interest 23.3% Level 4<br>First order differential equations 18.5% Level 5 First order differential equations 18.5% Level 5<br>Annuities 0.8% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>Qwen2-Math-72B-Instruct Qwen2-72B-Instruct<br>Integers 87.5% Integers 87.5%<br>Differentiation 65.4% Linear equations and functions 58.3%<br>Limits and continuity 64.7% Differentiation 54.9%<br>Linear equations and functions 62.6% Systems of linear equations 51.4%<br>Sample Space 61.5% Limits and continuity 48.2%<br>Systems of linear equations 57.6% Sample Space 46.2%<br>Vector geometry 51.4% Vector geometry 39.7%<br>Infinite sequences and series 48.6% Applications of differentiation 34.0%<br>Applications of differentiation 48.3% Infinite sequences and series 28.3%<br>Vector calculus 44.3% Differentiation of multivariable functions 26.8%<br>Integration of multivariable functions 41.3% Calculus of vector valued functions 25.8%<br>Differentiation of multivariable functions 38.0% Hypothesis tests 23.3%<br>Applications of integration 37.2% Vector calculus 21.7%<br>Confidence intervals 32.5% Confidence intervals 20.5%<br>Continuous distributions 28.6% Level 1 Continuous distributions 19.4% Level 1<br>Calculus of vector valued functions 27.8% Level 2 Integration of multivariable functions 18.8% Level 2<br>Interest 27.6% Level 3 Applications of integration 15.5% Level 3<br>Hypothesis tests 25.0% Level 4 Interest 14.7% Level 4<br>First order differential equations 17.1% Level 5 First order differential equations 11.0% Level 5<br>Annuities 1.7% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>**----- End of picture text -----**<br>


Figure 10: Model accuracy across topics 

40 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0041-01.png)


**----- Start of picture text -----**<br>
claude-3-opus-20240229 Qwen2-Math-7B-Instruct<br>Integers 85.0% Integers 80.8%<br>Sample Space 61.5% Limits and continuity 60.0%<br>Linear equations and functions 56.5% Linear equations and functions 51.3%<br>Limits and continuity 56.5% Differentiation 49.7%<br>Systems of linear equations 47.2% Systems of linear equations 46.5%<br>Differentiation 46.4% Sample Space 37.8%<br>Applications of differentiation 37.0% Applications of differentiation 35.8%<br>Vector geometry 34.2% Integration of multivariable functions 31.9%<br>Continuous distributions 29.6% Vector geometry 30.1%<br>Infinite sequences and series 29.0% Infinite sequences and series 28.3%<br>Differentiation of multivariable functions 26.3% Vector calculus 26.4%<br>Calculus of vector valued functions 24.7% Applications of integration 26.4%<br>Hypothesis tests 24.1% Calculus of vector valued functions 22.7%<br>Confidence intervals 24.1% Differentiation of multivariable functions 21.8%<br>Integration of multivariable functions 21.0% Level 1 Continuous distributions 19.4% Level 1<br>Applications of integration 20.9% Level 2 Confidence intervals 18.1% Level 2<br>Vector calculus 17.9% Level 3 First order differential equations 12.3% Level 3<br>Interest 17.8% Level 4 Interest 9.8% Level 4<br>First order differential equations 10.3% Level 5 Hypothesis tests 8.6% Level 5<br>Annuities 0.8% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>Meta-Llama-3-70B-Instruct Mistral-Small-Instruct-2409<br>Integers 80.0% Integers 76.7%<br>Differentiation 40.5% Linear equations and functions 55.7%<br>Linear equations and functions 39.1% Limits and continuity 45.9%<br>Sample Space 33.6% Sample Space 41.3%<br>Limits and continuity 32.9% Differentiation 39.2%<br>Applications of differentiation 22.6% Systems of linear equations 37.5%<br>Systems of linear equations 22.2% Applications of differentiation 30.6%<br>Vector geometry 21.9% Vector geometry 30.1%<br>Infinite sequences and series 18.1% Calculus of vector valued functions 23.7%<br>Calculus of vector valued functions 14.4% Differentiation of multivariable functions 22.9%<br>Differentiation of multivariable functions 13.4% Continuous distributions 19.4%<br>Continuous distributions 11.2% Infinite sequences and series 18.8%<br>Integration of multivariable functions 10.9% Hypothesis tests 15.5%<br>Applications of integration 9.5% Confidence intervals 14.5%<br>Hypothesis tests 8.6% Level 1 Integration of multivariable functions 12.3% Level 1<br>Vector calculus 8.5% Level 2 Vector calculus 11.3% Level 2<br>First order differential equations 8.2% Level 3 Applications of integration 10.8% Level 3<br>Interest 6.1% Level 4 Interest 9.8% Level 4<br>Confidence intervals 4.8% Level 5 First order differential equations 9.6% Level 5<br>Annuities 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>deepseek-math-7b-rl Qwen2-7B-Instruct<br>Integers 73.3% Integers 70.8%<br>Limits and continuity 41.2% Limits and continuity 44.7%<br>Differentiation 41.2% Differentiation 43.1%<br>Linear equations and functions 33.9% Linear equations and functions 42.6%<br>Systems of linear equations 31.9% Systems of linear equations 38.2%<br>Sample Space 21.0% Applications of differentiation 32.1%<br>Applications of differentiation 18.1% Sample Space 28.7%<br>Vector geometry 16.4% Vector geometry 25.3%<br>Calculus of vector valued functions 12.4% Applications of integration 20.3%<br>Infinite sequences and series 12.3% Infinite sequences and series 18.1%<br>Continuous distributions 10.2% Differentiation of multivariable functions 17.9%<br>Differentiation of multivariable functions 8.9% Calculus of vector valued functions 17.5%<br>Applications of integration 8.8% Integration of multivariable functions 16.7%<br>Integration of multivariable functions 8.7% Continuous distributions 14.3%<br>First order differential equations 6.2% Level 1 Confidence intervals 13.3% Level 1<br>Vector calculus 5.7% Level 2 Vector calculus 11.3% Level 2<br>Hypothesis tests 4.3% Level 3 Hypothesis tests 9.5% Level 3<br>Interest 3.7% Level 4 First order differential equations 8.9% Level 4<br>Confidence intervals 3.6% Level 5 Interest 7.4% Level 5<br>Annuities 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>**----- End of picture text -----**<br>


Figure 11: Model accuracy across topics 

41 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0042-01.png)


**----- Start of picture text -----**<br>
NuminaMath-7B-CoT Yi-1.5-9B-Chat<br>Integers 69.2% Integers 69.2%<br>Limits and continuity 38.8% Differentiation 37.9%<br>Linear equations and functions 37.4% Linear equations and functions 36.5%<br>Differentiation 33.3% Limits and continuity 35.3%<br>Systems of linear equations 27.1% Systems of linear equations 25.7%<br>Applications of differentiation 18.5% Applications of differentiation 24.5%<br>Sample Space 16.8% Vector geometry 19.9%<br>Vector geometry 16.4% Applications of integration 18.2%<br>Applications of integration 16.2% Integration of multivariable functions 18.1%<br>Integration of multivariable functions 13.0% Differentiation of multivariable functions 17.3%<br>Differentiation of multivariable functions 12.8% Confidence intervals 16.9%<br>Infinite sequences and series 12.3% Sample Space 16.8%<br>Calculus of vector valued functions 10.3% Vector calculus 16.0%<br>Continuous distributions 9.2% Calculus of vector valued functions 15.5%<br>Vector calculus 8.5% Level 1 Infinite sequences and series 12.3% Level 1<br>Interest 6.7% Level 2 Continuous distributions 12.2% Level 2<br>First order differential equations 4.1% Level 3 Hypothesis tests 9.5% Level 3<br>Hypothesis tests 3.4% Level 4 First order differential equations 8.2% Level 4<br>Confidence intervals 2.4% Level 5 Interest 6.7% Level 5<br>Annuities 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>Mathstral-7B-v0.1 Yi-1.5-34B-Chat<br>Integers 66.7% Integers 66.7%<br>Linear equations and functions 39.1% Linear equations and functions 45.2%<br>Limits and continuity 29.4% Differentiation 42.5%<br>Differentiation 28.1% Limits and continuity 42.4%<br>Vector geometry 22.6% Sample Space 31.5%<br>Sample Space 22.4% Systems of linear equations 30.6%<br>Applications of differentiation 18.1% Applications of differentiation 24.2%<br>Systems of linear equations 16.0% Vector geometry 23.3%<br>Calculus of vector valued functions 13.4% Confidence intervals 22.9%<br>Infinite sequences and series 9.4% Applications of integration 17.6%<br>Integration of multivariable functions 9.4% Continuous distributions 17.3%<br>Differentiation of multivariable functions 8.9% Infinite sequences and series 15.9%<br>Continuous distributions 8.2% Integration of multivariable functions 15.9%<br>Hypothesis tests 6.9% Differentiation of multivariable functions 15.6%<br>Vector calculus 6.6% Level 1 Hypothesis tests 13.8% Level 1<br>Interest 6.1% Level 2 Calculus of vector valued functions 13.4% Level 2<br>Applications of integration 6.1% Level 3 Vector calculus 12.3% Level 3<br>First order differential equations 4.1% Level 4 First order differential equations 9.6% Level 4<br>Confidence intervals 2.4% Level 5 Interest 8.0% Level 5<br>Annuities 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>Mistral-Nemo-Instruct-2407 deepseek-math-7b-instruct<br>Integers 63.3% Integers 57.5%<br>Linear equations and functions 33.9% Linear equations and functions 24.3%<br>Limits and continuity 29.4% Differentiation 24.2%<br>Differentiation 22.9% Limits and continuity 23.5%<br>Sample Space 20.3% Sample Space 16.8%<br>Vector geometry 15.8% Applications of differentiation 15.8%<br>Systems of linear equations 15.3% Systems of linear equations 15.3%<br>Applications of differentiation 13.6% Vector geometry 11.6%<br>Hypothesis tests 9.5% Differentiation of multivariable functions 7.3%<br>Calculus of vector valued functions 9.3% Infinite sequences and series 7.2%<br>Continuous distributions 9.2% Applications of integration 6.8%<br>Infinite sequences and series 8.0% Calculus of vector valued functions 5.2%<br>Differentiation of multivariable functions 6.1% Hypothesis tests 4.3%<br>Confidence intervals 4.8% Vector calculus 3.8%<br>First order differential equations 4.8% Level 1 Interest 3.7% Level 1<br>Integration of multivariable functions 4.3% Level 2 Integration of multivariable functions 3.6% Level 2<br>Applications of integration 4.1% Level 3 Confidence intervals 3.6% Level 3<br>Vector calculus 3.8% Level 4 Continuous distributions 3.1% Level 4<br>Interest 3.7% Level 5 First order differential equations 2.1% Level 5<br>Annuities 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>**----- End of picture text -----**<br>


Figure 12: Model accuracy across topics 

42 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models_images/iclr_2025_UGMathBench_A_Diverse_and_Dynamic_Benchmark_for_Undergraduate-Level_Mathematical_Reasoning_with_Large_Language_Models.pdf-0043-01.png)


**----- Start of picture text -----**<br>
Yi-1.5-6B-Chat Meta-Llama-3-8B-Instruct<br>Integers 54.2% Integers 41.7%<br>Limits and continuity 30.6% Linear equations and functions 22.6%<br>Differentiation 30.1% Differentiation 17.0%<br>Linear equations and functions 25.2% Sample Space 14.0%<br>Systems of linear equations 22.9% Limits and continuity 11.8%<br>Applications of differentiation 17.4% Vector geometry 8.2%<br>Vector geometry 16.4% Applications of differentiation 7.5%<br>Sample Space 14.0% Systems of linear equations 6.9%<br>Applications of integration 12.2% Differentiation of multivariable functions 5.0%<br>Differentiation of multivariable functions 11.2% Hypothesis tests 4.3%<br>Calculus of vector valued functions 10.3% First order differential equations 4.1%<br>Confidence intervals 9.6% Continuous distributions 4.1%<br>Integration of multivariable functions 9.4% Confidence intervals 2.4%<br>Infinite sequences and series 9.4% Infinite sequences and series 2.2%<br>Continuous distributions 8.2% Level 1 Vector calculus 1.9% Level 1<br>Vector calculus 7.5% Level 2 Interest 1.8% Level 2<br>Interest 6.1% Level 3 Integration of multivariable functions 1.4% Level 3<br>Hypothesis tests 3.4% Level 4 Applications of integration 1.4% Level 4<br>First order differential equations 3.4% Level 5 Calculus of vector valued functions 1.0% Level 5<br>Annuities 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>DeepSeek-V2-Lite-Chat Mistral-7B-Instruct-v0.3<br>Integers 40.8% Integers 24.2%<br>Linear equations and functions 13.0% Linear equations and functions 13.9%<br>Differentiation 11.8% Differentiation 10.5%<br>Limits and continuity 5.9% Limits and continuity 9.4%<br>Applications of differentiation 5.3% Sample Space 6.3%<br>Sample Space 4.9% Systems of linear equations 3.5%<br>Systems of linear equations 3.5% Vector geometry 2.7%<br>Vector geometry 3.4% Applications of differentiation 2.6%<br>Applications of integration 2.7% Differentiation of multivariable functions 2.2%<br>Interest 2.5% Calculus of vector valued functions 2.1%<br>Confidence intervals 2.4% Applications of integration 2.0%<br>Differentiation of multivariable functions 2.2% Interest 1.8%<br>Hypothesis tests 1.7% Infinite sequences and series 1.4%<br>Calculus of vector valued functions 1.0% First order differential equations 1.4%<br>Continuous distributions 1.0% Level 1 Continuous distributions 1.0% Level 1<br>Vector calculus 0.9% Level 2 Hypothesis tests 0.9% Level 2<br>Infinite sequences and series 0.7% Level 3 Confidence intervals 0.0% Level 3<br>Integration of multivariable functions 0.7% Level 4 Integration of multivariable functions 0.0% Level 4<br>Annuities 0.0% Level 5 Vector calculus 0.0% Level 5<br>First order differential equations 0.0% Level 6 Annuities 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8 0.0 0.2 0.4 0.6 0.8<br>EAcc EAcc<br>deepseek-moe-16b-chat<br>Integers 17.5%<br>Linear equations and functions 7.8%<br>Limits and continuity 3.5%<br>Differentiation 3.3%<br>Sample Space 1.4%<br>Vector geometry 1.4%<br>Applications of differentiation 1.1%<br>Continuous distributions 1.0%<br>Infinite sequences and series 0.7%<br>Systems of linear equations 0.7%<br>Calculus of vector valued functions 0.0%<br>Confidence intervals 0.0%<br>Integration of multivariable functions 0.0%<br>Annuities 0.0%<br>Hypothesis tests 0.0% Level 1<br>Vector calculus 0.0% Level 2<br>First order differential equations 0.0% Level 3<br>Applications of integration 0.0% Level 4<br>Interest 0.0% Level 5<br>Differentiation of multivariable functions 0.0% Level 6<br>0.0 0.2 0.4 0.6 0.8<br>EAcc<br>**----- End of picture text -----**<br>


Figure 13: Model accuracy across topics 

43 

Published as a conference paper at ICLR 2025 

Table 32: **Results of PHP on UGMathBench** (all figures are in %). "diff." refers to the improvement over results in Table 4. The best results within column with "diff." are **bolded** and the worst results are underlined. 

|**Models**<br>**AAcc**<br>diff. **AAcc**<br>**EAcc**<br>diff. **EAcc**<br>∆<br>diff. ∆|**Models**<br>**AAcc**<br>diff. **AAcc**<br>**EAcc**<br>diff. **EAcc**<br>∆<br>diff. ∆|
|---|---|
|_Closed-source LLMs_<br>GPT-4o-2024-08-06<br>60.73<br>+0.36<br>50.44<br>+0.50<br>10.29|-0.14|
|_Open-source Chat LLMs_<br>Yi-1.5-6B-Chat<br>26.11<br>-0.14<br>15.57<br>+0.24<br>10.54<br>Mistral-7B-Instruct<br>10.76<br>+0.20<br>4.41<br>-0.03<br>6.35<br>Qwen2-7B-Instruct<br>36.38<br>0.0<br>25.29<br>0.14<br>11.09<br>LLaMA3-8B-Instruct<br>16.41<br>-0.14<br>8.61<br>-0.3<br>7.8<br>Yi-1.5-9B-Chat<br>34.50<br>+0.21<br>21.75<br>+0.63<br>12.75|-0.38<br>+0.23<br>-0.14<br>+0.16<br>-0.42|
|Mistral-Nemo-Instruct-2407<br>25.46<br>+0.38<br>15.19<br>-0.24<br>10.27<br>DeepSeek-MOE-16B-Chat<br>6.34<br>+0.54<br>2.07<br>+0.11<br>4.27<br>DeepSeek-V2-Lite-Chat<br>13.98<br>+0.9<br>6.44<br>+0.75<br>7.54<br>Mistral-Small-Instruct-2409<br>39.66<br>-0.56<br>27.90<br>-0.94<br>11.76<br>Yi-1.5-34B-Chat<br>37.57<br>+0.04<br>23.83<br>-0.51<br>13.74|+0.62<br>+0.44<br>+0.25<br>+0.38<br>+0.47|
|LLaMA3-70B-Instruct<br>33.93<br>+0.31<br>23.24<br>-0.03<br>10.69<br>Qwen2-72B-Instruct<br>47.89<br>+0.13<br>36.08<br>+0.30<br>11.81<br>Mistral-Large-Instruct-2407<br>55.97<br>+0.04<br>45.17<br>+0.13<br>10.8|+0.34<br>-0.17<br>-0.09|
|_Specialized Mathematical LLMs_<br>DeepSeek-Math-7B-Instruct<br>24.57<br>+0.68<br>14.90<br>+1.29<br>9.67<br>DeepSeek-Math-7B-RL<br>29.42<br>-0.23<br>19.58<br>+0.24<br>9.84<br>NuminaMath-7B-CoT<br>27.81<br>-1.99<br>16.62<br>-2.19<br>11.19<br>Mathstral-7B-v0.1<br>29.75<br>**+1.24**<br>19.26<br>**+1.32**<br>10.49<br>Qwen2-Math-7B-Instruct<br>42.65<br>-1.08<br>31.18<br>-1.08<br>11.47<br>Qwen2-Math-72B-Instruct<br>57.35<br>+ 0.32<br>46.04<br>+0.19<br>11.31|**-0.61**<br>-0.47<br>+0.20<br>-0.08<br>0.0<br>+0.13|



Table 33: **Results across different subjects of PHP for GPT-4o** . (all figures are in %). "diff." refers to the improvement over results in Table 4. The best results within column with "diff." are **bolded** and the worst results are underlined. 

|**Subject**<br>**AAcc**<br>diff. **AAcc**<br>**EAcc**<br>diff. **EAcc**<br>∆<br>diff. ∆|**Subject**<br>**AAcc**<br>diff. **AAcc**<br>**EAcc**<br>diff. **EAcc**<br>∆<br>diff. ∆|
|---|---|
|Arithmetic<br>92.11<br>+0.29<br>88.47<br>+1.20<br>3.64<br>Algebra<br>72.33<br>+0.97<br>65.35<br>+0.68<br>6.98<br>Set theory and logic<br>71.98<br>-1.45<br>57.97<br>-1.45<br>14.01<br>Trigonometry<br>67.79<br>-1.69<br>53.37<br>+1.68<br>14.42<br>Combinatorics<br>78.79<br>+1.14<br>65.91<br>+4.55<br>12.88<br>Geometry<br>67.49<br>+3.31<br>54.04<br>+2.49<br>13.45<br>Calculus single-variable<br>65.11<br>+0.24<br>53.26<br>+0.21<br>11.85<br>Calculus multivariable<br>49.85<br>+0.21<br>38.53<br>+0.30<br>11.32<br>Linear Algebra<br>62.85<br>+0.47<br>52.81<br>+1.20<br>10.04<br>Number Theory<br>65.22<br>+1.45<br>54.35<br>+2.18<br>10.87<br>Financial Mathematics<br>29.87<br>+0.49<br>20.81<br>+2.60<br>9.06<br>Probability<br>65.38<br>-0.59<br>52.68<br>-2.08<br>12.7<br>Statistics<br>60.27<br>-0.16<br>48.88<br>-0.50<br>11.39<br>Complex Analysis<br>70.59<br>+1.31<br>58.82<br>0.0<br>11.77<br>Differential Equations<br>25.03<br>0.33<br>18.03<br>-1.31<br>7.00<br>Abstract Algebra<br>53.17<br>**+4.76**<br>35.71<br>**+7.14**<br>7.46|-0.91<br>+ 0.29<br>0.0<br>-3.37<br>**-3.41**<br>+1.82<br>+0.03<br>-0.09<br>-0.73<br>-0.73<br>-2.11<br>+1.49<br>+0.34<br>+1.31<br>+1.64<br>-2.38|



44 

