# **MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation** 

**Zhongshen Zeng** Chinese University of HongKong `zszeng23@cse.cuhk.edu.hk` 

**Pengguang Chen** Chinese University of Hong Kong `pgchen@cse.cuhk.edu.hk` 

**Shu Liu Haiyun Jiang** Smartmore Co.Ltd Tencent AI Lab `sliu@smartmore.com haiyunjiang@tencent.com` 

**Jiaya Jia** Chinese University of Hong Kong `leojia@cse.cuhk.edu.hk` 

## **Abstract** 

In this work, we introduce a novel evaluation paradigm for Large Language Models (LLMs) that compels them to transition from a traditional question-answering role, akin to a student, to a solution-scoring role, akin to a teacher. This paradigm, focusing on "reasoning about reasoning," hence termed meta-reasoning, shifts the emphasis from result-oriented assessments, which often neglect the reasoning process, to a more comprehensive evaluation that effectively distinguishes between the cognitive capabilities of different models. By applying this paradigm in the GSM8K dataset, we have developed the MR-GSM8K benchmark. Our extensive analysis includes several state-of-the-art models from both open-source and commercial domains, uncovering fundamental deficiencies in their training and evaluation methodologies. Notably, while models like Deepseek-v2 and Claude3-Sonnet closely competed with GPT-4 in GSM8K, their performance disparities expanded dramatically in MR-GSM8K, with differences widening to over 20 absolute points, underscoring the significant challenge posed by our meta-reasoning approach. 

## **1 Introduction** 

Pretrained on trillions of tokens and equipped with billions of parameters, today’s large language models [25, 1, 33] are capable of generating coherent texts and achieving super-human performances in many tasks [8, 15]. With the aim of differentiating cognitive abilities among models, math questions are often chosen as a proxy evaluation task. However, despite the complexity and diversity of these math problems, recent state-of-the-art LLMs [25, 38, 13] have achieved accuracy rates exceeding 80% [20] on multi-step math reasoning datasets like GSM8K [10]. 

Upon detailed examination of the design principles and objectives of current math datasets, we identified several key shortcomings. Firstly, the majority of these datasets focus on result-oriented metrics, scoring accuracy based solely on the final answer, without considering the underlying reasoning process. With the emergence of the chain of thought methodology [34] and its derivative techniques [9, 36] as the de facto standard for evaluating reasoning processes, we argue that the resultdriven evaluation method may be insufficient for a comprehensive assessment of intended cognitive and reasoning capabilities. Secondly, a recent study [26] suggests that some LLMs, which achieved state-of-the-art performances in GSM8K and MATH [15] benchmarks, demonstrate unexpectedly low performance when facing newly released Hungarian high school exams. This raises concerns about data contamination and potential overfitting to the benchmarks, and it challenges the efficacy of these benchmarks in differentiating model capabilities. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
100 Comparison of Model Performance on GSM8K vs MR_GSM8K<br>GSM8K<br>MR_GSM8K<br>80<br>60<br>40<br>20<br>0<br>GPT-4-turboLlama3-70B-InstructDeepseek-v2-236BClaude3-sonnetPhi-3-3.8BGPT-3.5-turboQwen-v1.5-72BLlama3-8B-InstructClaude3-haikuDeepseek-Math-7B-RLMAmmoTH-70BWizardMath-v1.1-7BQwen-v1.5-1.8BMetaMath-70B<br>Scores<br>**----- End of picture text -----**<br>


Figure 1: Model performances on GSM8K measured in accuracy versus performances on MRGSM8K measured in MR-Score. Note that some models score near zero in our benchmark, highlighting the limitations of current training and evaluation paradigms. 

In response to these identified limitations, we introduced a novel paradigm that shifts the role of the evaluated model from a question-answering student to a solution-scoring teacher. Specifically, instead of delivering potential solutions to given questions, which may be subject to data contamination issues, the evaluated models are now presented with question-solution pairs and tasked with determining solution correctness, identifying potential first error steps, and providing reasons for errors. This paradigm shift challenges the evaluated models to engage in meta-reasoning about different reasoning processes, a concept we term "meta-reasoning" in this paper. 

Following this design principle, we have developed a new benchmark named **Meta-ReasoningGSM8k** ( **MR-GSM8k** ) and proposed a novel metric called MR-Score. Our benchmark, characterized by instances manually labeled by experts and rigorously reviewed, serves as a robust tool for both qualitative and quantitative assessments of language models. Our findings indicate that most stateof-the-art models demonstrate a significant performance decline in this more nuanced assessment. As demonstrated in Figure-1, although state-of-the-art models exhibit comparable performance in GSM8K, there is considerable variance in their effectiveness on our benchmark, with discrepancies up to more than tenfold. 

We argue that our evaluation paradigm not only introduces a metric that more effectively differentiates based on the reasoning process over mere computational outcomes, but it also exposes fundamental deficiencies within current evaluation and training methodologies. First, as detailed in Section-5, our experiments revealed that specialized math models struggle to generalize their reasoning abilities to our new paradigm, regardless of whether they are directed by specific instructions or through few-shot in-context learning. Second, we observed that contrary to the common belief that larger models inherently perform better in reasoning tasks, smaller chat models such as Phi-3-Mini [14] can surpass those twenty times their size. This underscores the significance of data quality and techniques like knowledge distillation in enhancing model performance. Third, our empirical findings indicate that current models still engage in superficial mathematical reasoning, exhibiting flaws such as a lack of ontological understanding [32], susceptibility to the reversal curse [6], and inadequate global comprehension of solution spaces. 

In conclusion, our paper significantly contributes to the field in the following ways: 

- We introduce a novel evaluation principle, the accompanying open-source benchmark MRGSM8k, and the metric MR-Score. 

2 

- We demonstrate the effective transformation of an existing benchmark (e.g., GSM8K) and how such modification can lead to robust evaluation against potential overfitting and data contamination. 

- We conduct comprehensive experiments on an array of state-of-the-art models using the MRGSM8k benchmark, highlighting critical shortcomings in current training and evaluation paradigms. 

- Through analysis of cognitive levels and examination of holistic coverage on the solution space, we emphasize the need for benchmarks that go beyond surface-level evaluations, fostering more sophisticated and nuanced AI development. 

## **2 Related Works** 

Complex reasoning tasks, such as math problems, have long been recognized as effective proxies for gauging the cognitive abilities of language models [29, 17, 30, 27, 23, 15, 10]. These tasks require the ability to understand symbols and text, dissect problems into logically connected sub-problems, combine results, and synthesize final solutions. They engage cognitive functions such as pattern induction, formula recall, deductive rule application, and abstract symbolic reasoning. 

GSM8K [10] and MATH [15] have been prominent benchmarks for evaluating the math reasoning capabilities of large language models (LLMs) in recent years. The chain of thought approach, proposed by [34], addresses multi-step reasoning tasks by breaking them down into manageable steps. Stanford Alpaca [31] has popularized the knowledge distillation method of cloning abilities from ChatGPT [24] by generating related QA pairs. WizardMath [20] further refined this distillation by specifying the difficulties of the QA pairs in the generation process. Mammoth [39] combines chain of thought and program of thought, finetuning its models with answers generated by GPT-4 [25] in either natural or code language. MetaMath [38] broadens the variety of question types by introducing forward/backward reasoning variations. 

Despite significant advancements in math reasoning, evidence suggests that large language models may not fully master reasoning or even understand their own outputs. For example, [12] found that LLMs fail to generalize to questions of varying complexity from their training data. [2] demonstrated that, despite occasional analytical brilliance, GPT-4 is still severely limited in its reasoning capabilities. Similarly, [16] and [37] have shown that ChatGPT struggles to judge the correctness of math problem solutions. However, our work focuses on constructing a qualitative and quantitative evaluation framework and discusses the evaluation principles and deficiencies of the current training paradigm in greater depth. 

## **3 Dataset Construction** 

## **3.1 Construction** 

**Question Types:** The MR-GSM8k consists of three distinct types of questions. The first type includes original GSM8K instances, sampled directly from the dataset [10]. The second type modifies GSM8K questions to include prompts requesting code solutions, as explored in [39]. The third type, termed reversed reasoning [38], involves concealing one of the inputs and asking for solutions that compute the missing input using the provided original answer. These variations were selected for their potential to broaden the scope of reasoning methodologies in LLMs. The "Program of Thought" approach, which includes code solutions, was proposed by [21] and has been empirically validated as an effective framework for math reasoning [39, 13]. Reversed reasoning, a method that tests whether models can effectively learn backward relationships, has been recently emphasized in studies such as [5], presenting additional challenges for these systems. 

**Solution:** For each question collected, we used MetaMath-7B [38] with a temperature setting of 1 to generate step-by-step solutions. We found this setting results in outputs with decent accuracy and nuanced mistakes that we would like the annotators and evaluated models to figure out. Intentionally, we targeted a result accuracy of approximately 50% for the collected question-solution pairs, ensuring a balance between correct outcomes and those arising from flawed reasoning processes. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0004-00.png)


**----- Start of picture text -----**<br>
POT GSM8K Original GSM8K REVERSED GSM8K<br>### Instruction: ### Instruction: ### Instruction:<br>• An electronics seller bought 5 phones  • An electronics seller bought 5 phones  • An electronics seller bought 5 phones for<br>for $700 each and gives the seller  for $700 each and gives the seller  $700 each and gives the seller  $X  in dollar<br>$4000 in dollar bills. How much will the  $4000 in dollar bills. How much will the  bills. How much will the seller give back in<br>seller give back in change?  Let's write a  seller give back in change?  change? If we know the answer to the<br>Input  program. above question is 500,  what is the value of<br>Question unknown variable 'X'?<br>### Response: ### Response: ### Response:<br>• "Step 1: total_amount_paid = 5 * 700", • "Step 1: The seller bought 5 phones for  • "Step 1: The seller bought 5 phones for<br>• "Step 2: total_amount_given = 4000", 5*700 = $<<5*700=3500>>3500.", $700 each, so…….",<br>• "Step 3: change = total_amount_given -  • "Step 2: …….", • "Step 2: The seller gives the seller $X in<br>total_amount_paid", • "Step 3:……", dollar bills, so ……",<br>• "Step 4: print(change)" • "Step 4: The seller will give back $3500- • "Step 3: ……",<br>$4000 = $<<3500-4000=500>>500 in  • "Step 4: Solving for X, we get: X = 4000.",<br>Output change." • "Step 5: The value of X is 4000.",<br>Solution<br>• Final Judgement : Wrong • Final Judgement : Correct • Final Judgement :  Correct<br>• First Error Step : Step 3 • First Error Step : N/A • First Error Step : N/A<br>• Error Analysis : The student has  • Error Analysis : N/A • Error Analysis : N/A<br>misunderstood the concept of<br>calculating change. It is not possible to<br>get a negative change. Instead, the<br>student should have checked if the<br>total amount given was less than the<br>GPT4-Grading  total amount required and then calculated the additional amount<br>Output required.<br>**----- End of picture text -----**<br>


Figure 2: Structure of the MR-GSM8k benchmark and its evaluation paradigm. 

## **3.2 Annotation** 

For each collected question-solution pair, a panel of selected annotators was recruited to review each question-solution pair for its reasoning process and determine the following fields: 

**Solution Correctness** : Solutions that yield a final output differing from the established ground truth are automatically marked as incorrect. However, in cases where the solution’s final output aligns with the ground truth, annotators are tasked with reviewing the entire reasoning path. Their objective is to ascertain whether the correct output is the result of a logical and sensible reasoning process. 

**First Error Step** : This attribute is applicable for solutions with either an unmatched final output or a matched final output underpinned by flawed reasoning. Annotators identify the initial step in the reasoning process where the logic deviates from correctness. In line with the approach of [18], we dissected GSM8K solutions into discrete steps, each marked by a newline character, and indexed them accordingly. Each step is then categorized as positive, neutral, or negative. Positive and neutral steps represent stages in the reasoning process where the correct final output remains attainable, whereas negative steps indicate a divergence from the path leading to the correct solution. 

**Error Analysis** : Beyond identifying the first erroneous step, annotators are also responsible for conducting an in-depth analysis of the reasoning that led to the error. This involves an examination of the solution’s reasoning flow, focusing on the cause of the initial error and what the correct line of reasoning should have been at that juncture. This error analysis is subsequently compared against the reasoning errors identified by the evaluated models during testing, to assess their accuracy and validity. 

## **3.3 Quality Control** 

**Annotators:** Our panel of annotators is both well-trained and highly selective. Each candidate is required to thoroughly review our annotation manual (see Appendix-D for details) and watch a demonstration video. Following this, candidates undergo a trial labeling process using a small, 

4 

Table 1: MR-GSM8K statistics: The first two columns are the numbers of correct and incorrect solutions. The following two columns hold the lengths of questions and solutions, which are measured in number of words. The last two column shows the averaged solution steps and first error steps. 

|**Question Types**|**Correct**|**Incorrect**|**Q Length**|**S Length**|**S Steps**|**First Error Steps**|
|---|---|---|---|---|---|---|
|Original|693|725|46.9|100.5|7.|3.4|
|POT|113|109|45.1|34.9|6.5|3.9|
|Reverse|622|737|62.4|157.0|11.3|4.5|
|Total|1428|1571|51.5|97.5|8.3|3.9|



hold-out dataset. Recruitment decisions are based on performance during this trial. Additionally, four quality control supervisors, including the first author, are designated to address queries and monitor the labeling quality throughout the project. 

**Annotation Procedure:** Every question in the MR-GSM8K undergoes multiple levels of scrutiny before being used in the evaluation process. Initially, each question is labeled twice by different annotators to ensure consistency. Any discrepancies in labeling, particularly regarding solution correctness or the identification of the first error step, are flagged and reviewed by a quality control supervisor. Furthermore, 50 percent of the labeled questions are randomly selected for a second round of quality control to identify and eliminate any inconsistent error steps or reasons. In the final stage of quality control, approximately 10 percent of the questions are manually inspected by the authors to ensure accuracy. This rigorous process ensures that any questions with incorrect error steps or reasoning are rectified before inclusion in the final dataset. 

## **3.4 Dataset Statistics** 

Table-1 presents the statistics of MR-GSM8K, illustrating the distribution of correct and incorrect solutions across the three different types of questions. It is noteworthy that the reversed question type exhibits a significantly longer average question length due to its construction methodology. This type of question, due to its complex nature, also tends to have longer solution lengths as indicated in the table. Conversely, questions classified under the Program-of-Thought category, which typically require code solutions, have the shortest solution lengths, reflecting the concise and succinct nature of coding language. Despite these variations in average solution lengths and step counts, the average position of the first error step remains remarkably consistent across all question types. 

## **4 Evaluation Process & Metric** 

**Evaluation** As illustrated in Figure-2, instead of simply solving a given question, the evaluated models are now presented with a question and a paired potential solution to score (e.g., the upper light blue part in the figure). Specifically, the evaluated model (e.g., the lower white part) is required to predict the correctness of the solution. If the solution is deemed incorrect, the model must further identify the first-error-step and explain the error-reason. The solution-correctness and firsterror-step are scored automatically based on comparison with manual annotations. Only when the evaluated model correctly identifies an incorrect solution and its first-error-step will its error-reason be further examined, either manually or automatically by models (see our discussions on error reason examination in the limitation section of Appendix-A). 

**Metrics** To provide a unified and normalized score reflecting the overall competence of the evaluated model, we propose a novel metric named **MR-Score** , consisting of three sub-metrics. The first sub-metric is the Matthews Correlation Coefficient (MCC, [22]) for binary classification of solutioncorrectness. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0005-09.png)


Here, TP, TN, FP, and FN represent true positive, true negative, false positive, and false negative, respectively. The MCC score ranges from -1 to +1, where -1 indicates total disagreement between prediction and observation, 0 suggests performance near random, and +1 denotes perfect prediction. 

5 

Table 2: Evaluation results on MR-GSM8K 

|**Model**|**Task1-TPR**<br>_k_=0<br>_k_=3|**Task1-TNR**<br>_k_=0<br>_k_=3|**Task1-MCC**<br>_k_=0<br>_k_=3|**Task2-Accy**<br>_k_=0<br>_k_=3|**Task3-Accy**<br>_k_=0<br>_k_=3|**MR-Score**|
|---|---|---|---|---|---|---|
|||||||_k_=0<br>_k_=3|
|||**Open-Source Small**|||||
|Qwen-1.8B<br>Phi3-3.8B|21.8<br>33.3<br>11.3<br>62.6|0.1<br>3.9<br>0.<br>0.<br>98.5<br>72.6<br>20.4<br>35.4||0.<br>0.4<br>32.9<br>26.3|0.<br>0.<br>18.0<br>13.9|0.<br>0.1<br>**22.9**<br>21.9|
|||**Open-Source Medium**|||||
|Deepseek-Math-7B-RL <br>WizardMath-v1.1-7B<br>Llama3-8B|77.3<br>2.4<br>99.3<br>6.7<br>3.2<br>40.9|52.3<br>0.4<br>30.4<br>0.<br>0.5<br>0.6<br>0.0<br>0.0<br>98.3<br>80.3<br>5.1<br>23.1||9.8<br>0.1<br>0.3<br>0.2<br>29.1<br>23.3|5.1<br>0.1<br>0.3<br>0.1<br>15.0<br>11.6|11.6<br>0.1<br>0.2<br>0.1<br>17.2<br>**17.4**|
|||**Open-Source Large**|||||
|MAmmoTH-70B<br>MetaMath-70B<br>Llama3-70B<br>Qwen1.5-72B<br>Deepseek-v2-236B|88.0<br>89.8<br>7.8<br>0.0<br>67.6<br>89.3<br>83.7<br>87.7<br>60.1<br>88.2|23.1<br>2.8<br>14.6<br>0.0<br>0.3<br>0.0<br>0.0<br>0.0<br>83.0<br>66.0<br>51.3<br>56.4<br>57.1<br>52.4<br>42.0<br>42.5<br>87.2<br>61.5<br>49.4<br>51.2||3.9<br>0.3<br>0.1<br>0.0<br>38.9<br>33.5<br>19.1<br>23.1<br>26.8<br>32.4|1.8<br>0.3<br>0.0<br>0.0<br>32.7<br>25.7<br>13.5<br>15.8<br>23.8<br>28.3|5.0<br>0.2<br>0.0<br>0.0<br>**38.3**<br>34.2<br>20.9<br>23.3<br>29.8<br>34.1|
|||**Closed-Source LLMs**|||||
|Claude3-Haiku<br>GPT-3.5-Turbo<br>Claude3-Sonnet<br>GPT-4-Turbo|70.4<br>99.0<br>16.3<br>59.7<br>35.1<br>88.4<br>69.5<br>83.0|51.7<br>8.1<br>22.5<br>16.7<br>93.8<br>65.7<br>16.2<br>25.5<br>89.8<br>44.8<br>30.0<br>36.5<br>91.8<br>84.2<br>63.3<br>67.2||17.2<br>2.3<br>30.6<br>21.0<br>25.2<br>18.8<br>48.8<br>51.7|11.3<br>1.8<br>20.3<br>13.0<br>19.9<br>15.6<br>46.3<br>48.1|15.3<br>4.9<br>22.6<br>17.9<br>23.5<br>20.8<br>50.5<br>**53.0**|



In the context of this paper, negative values are interpreted as no better than random guesses, and 0 is set as the cut-off threshold for normalization purposes. 

The second metric is the accuracy of the first-error-step prediction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0006-04.png)


The third metric calculates the accuracy of identifying both the correct first-error-step and the error-reason: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0006-06.png)


**MR-Score** is a weighted combination of these three metrics: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0006-08.png)


The weights _w_ 1 _, w_ 2 _,_ and _w_ 3 are chosen empirically to maximize differentiation between model performances by taking the difficulties of each task into account. For an extended discussion on the design of MR-Score, please refer to Appendix-C. 

## **5 Experiments** 

## **5.1 Experiment Setup** 

To evaluate the performance of different language models on our benchmark, we selected models from diverse backgrounds. These models vary greatly in size, ranging from a few billion parameters, such as Qwen-v1.5-1.8B [3], to 70 billion parameters like Llama3-70B [33], and up to 236 billion parameters as seen in Deepseek-v2-236B [7]. Additionally, to contrast performances between models fine-tuned from general instructions and those specialized in math problems, we included representative math models from the open-source community, such as WizardMath-v1.1-7B [20], MAmmoTH-70B [39], DeepseekMath-7B-RL [28], and MetaMath-70B [38]. Furthermore, to explore the differences between commercial and open-source models, we included models from the OpenAI GPT family [24] and the Anthropic Claude-3 series [1]. 

6 

Each model was evaluated under a zero-shot setting to assess their ability to follow instructions and their mathematical reasoning capabilities. Given that some evaluated models are not fine-tuned for general instruction following, we also tested each model under a few-shot setting to leverage their in-context learning abilities for understanding mathematical reasoning (see our prompts in Figure-4 and 5 in the Appendix). To ensure reproducibility and minimize variance, the inference temperature was set to zero across all models. 

## **5.2 Experiment Results** 

Our evaluation results are presented in Table-2, where tasks 1, 2, and 3 correspond to determining solution correctness, the first error step, and the error reason, respectively. For Task 1, we also report the true positive rate and true negative rate. Key observations from our study are as follows: 

**Overall Performance:** As depicted in the table, GPT-4-Turbo significantly outperforms all other models across both open-source and closed-source domains. Among the open-source models, Llama370B exhibits the closest performance to GPT-4, yet it still lags by more than 12 absolute points in MR-Score. In the small to medium model size category, Phi3-3.8B outshines others, even surpassing the scores of Claude3-Haiku. Notably, most specialized models we evaluated failed to adapt to our evaluation paradigm and systematically underperformed compared to the generalized chat models. Despite similar levels of success on GSM8K, as illustrated in Figure-1, all models tested show a significant drop in performance in our benchmark, resulting in a much wider differentiation in scores. 

**Performance by Model Size:** Contrary to the common belief that larger models inherently possess greater capabilities, our findings challenge this notion. Specifically, Phi3-3.8B performed substantially better than other models at the 7B level and achieved comparable performance with Qwen1.5-72B, which is approximately twenty times larger. A similar trend is observed between Llama3-70B and Deepseek-v2-236B, with Llama3-70B outperforming the latter in both zero-shot and few-shot settings. These results suggest that while model size is an important factor in reasoning ability, the quality of pretraining data and the application of data synthesis techniques such as knowledge distillation may also play crucial roles. 

**Specialized Math Models:** Within the open-source community, a multitude of models are dedicated to math reasoning, employing various fine-tuning techniques and datasets. Despite this, most models failed significantly in our benchmark. Specifically, WizardMath-v1.1-7B and MetaMath-70B appeared overfitted to the GSM8K response format and were unable to adhere to our evaluation instructions, both with and without few-shot demonstrations. Conversely, Deepseek-Math-7B-RL and MAmmoTH-70B managed to comprehend our complex evaluation instructions and achieved decent performance. 

**Few Shot Demonstrations:** Given that the MR-GSM8K benchmark poses significant challenges in complex instruction-following in addition to mathematical reasoning, we explored whether providing few-shot demonstrations could enhance the performance of specialized models. However, as indicated in Table-2, the few-shot setting proved detrimental to all tested models. Although Deepseek-Math-7BRL and MAmmoTH-70B demonstrated decent performance in the zero-shot setting, they struggled to adhere to the desired task instructions in the few-shot setting, reverting to the question-answering paradigm when presented with extended context and demonstrations. Similarly, few-shot examples did not aid WizardMath-v1.1-7B and MetaMath-70B in adhering to the expected scoring and reasoning format. These specialized models exhibited a strong tendency to revert to the question-answering paradigm, an issue not as prevalent in general chat models. This tendency suggests that fine-tuning on a narrowly focused dataset, often sampled or augmented from specific math datasets like GSM8K, may lead to overfitting to a particular input/output data distribution, resulting in only a superficial mastery of mathematical reasoning. Despite the close similarity between MR-GSM8K and the datasets these models were exposed to, their underperformance highlights a critical shortcoming in the generalization capabilities of reasoning abilities developed through specialized fine-tuning. 

For general chat models, the impact of few-shot demonstrations varied significantly across models, even within the same family, with outcomes ranging from slight improvements to notable deteriorations. No consistent pattern emerged, indicating that improvements do not uniformly correlate with model size or initial benchmark performance. 

**In Context Learning Bias:** An intriguing outcome from our few-shot experiments is the significant impact this setting had on the models’ propensity to score solutions as correct or incorrect. Analyzing 

7 

the true positive rate (TPR) and true negative rate (TNR) in the solution correctness task, we empirically observed that most models exhibited an increase in TPR while concurrently showing a reduction in TNR under the three-shot setting. This trend can likely be attributed to the composition of our few-shot examples, which included two correct solutions and one incorrect solution, suggesting that the distribution of correctness within these examples may influence model predictions. To verify our conjecture, we conducted an ablation study on the number of correct solutions and the result shows strong correlation with our proposal (see Figure-6 in Appendix for more details). This susceptibility of language models to the distribution of few-shot examples highlights a fundamental flaw in the current reasoning paradigms: rather than making scoring decisions based purely on reasoning, the models appear to be swayed by the few-shot examples. This influence might also account for the generally worse performance observed in the few-shot experiments compared to the zero-shot settings, where such biases are absent. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0008-01.png)


Figure 3: This figure aims to illustrate the fundamental limitations of the current training and evaluation paradigms for LLMs. The left side of the figure demonstrates that utilizing only correct solutions limits coverage of the solution space despite various data augmentation techniques. The right side depicts potential solutions that might contain incorrect steps or reach the final computation result through an incorrect reasoning path. Models trained exclusively on correct reasoning paths often fail to assess the validity or discern nuanced differences of alternative reasoning paths for the same problem. This highlights a critical gap in the training paradigm, where models exhibit basic imitation skills but lack a deep understanding of the underlying logical rules, leading to a superficial grasp of reasoning processes. 

## **6 Discussion** 

## **6.1 Case Studies on Reasoning Failures** 

In Appendix-D, we provide several case studies of responses generated by evaluated models. These examples help illuminate various reasoning errors that are often obscured in simpler benchmarks such as GSM8K. First, models frequently exhibit what is known as the "reversal curse," where they mistakenly claim that "A equals to B" is not equivalent to "B equals to A" [5]. Second, inconsistencies and errors in quantity unit usage by the models suggest a fundamental lack of ontological understanding regarding the properties of the quantities they manipulate. Third, many models display insensitivity to numerical computation errors, highlighting an inherent weakness in arithmetic processing by language models. These observations underscore the critical gaps in model training and evaluation, pointing to an over-reliance on correct solution paths that neglects the broader context and complexity of real-world reasoning. 

## **6.2 What Is the Significance of Reason About Reasoning?** 

In this paper, we have demonstrated that simply observing computation results is insufficient to uncover the cognitive depth of evaluated models. Equally important is the validity and logic of the reasoning process employed by these models. For a model to successfully diagnose solution 

8 

correctness, it must infer the correct result and also engage in counterfactual reasoning along different reasoning paths, actively examining the conditions and assumptions made at various steps. Success in this paradigm is unlikely without a holistic understanding and robust mastery of the underlying concepts. Thus, the "reason about reasoning" paradigm emerges as a critical meta-evaluative tool. 

Another key significance of this paradigm is its capability to transform any existing benchmark into a more robust and holistic assessment tool. As highlighted by [4] and [35], data contamination issues are becoming increasingly prevalent and elusive to detect. Our paradigm not only facilitates modifications to existing benchmarks but also demonstrates robustness against potential data contamination, as evidenced by our experiments across a wide array of state-of-the-art LLMs. 

## **7 Conclusion** 

Throughout this paper, we have explored the inadequacies of prevalent math reasoning benchmarks and introduced a pioneering evaluation paradigm that compels models to engage in meta-reasoning. Our empirical findings demonstrate that this novel paradigm enables our benchmark to effectively differentiate between models and uncover their various deficiencies. This differentiation has been particularly evident in the performance struggles of state-of-the-art language models when confronted with our benchmark, revealing significant shortcomings in current training methodologies. 

These revelations underscore the need for a critical reevaluation of existing training and evaluation practices in the realm of large language models. By advocating for the widespread adoption of our "reason about reasoning" evaluation paradigm, we encourage researchers to adapt and broaden other reasoning benchmarks similarly. Such transformation is vital not only for a more rigorous assessment of LLMs but also for fostering a deeper and more holistic understanding of these models’ capabilities. 

9 

## **References** 

- [1] Anthropic. Introducing claude, 2023. URL `https://www.anthropic.com/index/ introducing-claude` . 

- [2] Konstantine Arkoudas. GPT-4 can’t reason. _CoRR_ , abs/2308.03762, 2023. doi: 10.48550/ ARXIV.2308.03762. URL `https://doi.org/10.48550/arXiv.2308.03762` . 

- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _CoRR_ , abs/2309.16609, 2023. doi: 10.48550/ARXIV.2309.16609. URL `https: //doi.org/10.48550/arXiv.2309.16609` . 

- [4] Simone Balloccu, Patr’icia Schmidtov’a, Mateusz Lango, and Ondvrej Duvsek. Leak, cheat, repeat: Data contamination and evaluation malpractices in closed-source llms. In _Annual Meeting of the Association for Computational Linguistics_ , 2024. URL `https: //api.semanticscholar.org/CorpusID:267499939` . 

- [5] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on "a is b" fail to learn "b is a". _ArXiv_ , abs/2309.12288, 2023. URL `https://api.semanticscholar.org/CorpusID: 262083829` . 

- [6] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on "a is b" fail to learn "b is a". _CoRR_ , abs/2309.12288, 2023. doi: 10.48550/ARXIV.2309.12288. URL `https://doi.org/ 10.48550/arXiv.2309.12288` . 

- [7] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, Alex X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. Deepseek LLM: scaling open-source language models with longtermism. _CoRR_ , abs/2401.02954, 2024. doi: 10.48550/ARXIV.2401.02954. URL `https: //doi.org/10.48550/arXiv.2401.02954` . 

- [8] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, John A. Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuan-Fang Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4. _ArXiv_ , abs/2303.12712, 2023. URL `https://api.semanticscholar.org/ CorpusID:257663729` . 

- [9] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _ArXiv_ , abs/2211.12588, 2022. URL `https://api.semanticscholar.org/CorpusID: 253801709` . 

- [10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. 

10 

- [11] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. _ArXiv_ , abs/2305.14314, 2023. URL `https://api.semanticscholar. org/CorpusID:258841328` . 

- [12] Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaïd Harchaoui, and Yejin Choi. Faith and fate: Limits of transformers on compositionality. _ArXiv_ , abs/2305.18654, 2023. URL `https://api. semanticscholar.org/CorpusID:258967391` . 

- [13] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. _ArXiv_ , abs/2309.17452, 2023. URL `https://api.semanticscholar.org/CorpusID: 263310365` . 

- [14] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need. _CoRR_ , abs/2306.11644, 2023. doi: 10.48550/ARXIV.2306.11644. URL `https://doi.org/10.48550/arXiv.2306. 11644` . 

- [15] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. 

- [16] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. _ArXiv_ , abs/2310.01798, 2023. URL `https://api.semanticscholar.org/CorpusID: 263609132` . 

- [17] Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In _North American Chapter of the Association for Computational Linguistics_ , 2016. URL `https://api.semanticscholar.org/CorpusID: 2228719` . 

- [18] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _ArXiv_ , abs/2305.20050, 2023. URL `https://api.semanticscholar.org/CorpusID: 258987659` . 

- [19] Xiao Liu, Xuanyu Lei, Sheng-Ping Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Tam, Xiaohan Zhang, Lichao Sun, Hongning Wang, Jing Zhang, Minlie Huang, Yuxiao Dong, and Jie Tang. Alignbench: Benchmarking chinese alignment of large language models. _ArXiv_ , abs/2311.18743, 2023. URL `https://api.semanticscholar. org/CorpusID:265506600` . 

- [20] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ , 2023. 

- [21] Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. Language models of code are few-shot commonsense learners. _ArXiv_ , abs/2210.07128, 2022. URL `https://api.semanticscholar.org/CorpusID:252873120` . 

- [22] Brian W. Matthews. Comparison of the predicted and observed secondary structure of t4 phage lysozyme. _Biochimica et biophysica acta_ , 405 2:442–51, 1975. URL `https://api. semanticscholar.org/CorpusID:44596673` . 

- [23] Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. In _Annual Meeting of the Association for Computational Linguistics_ , 2020. URL `https://api.semanticscholar.org/CorpusID: 220047831` . 

11 

- [24] OpenAI. Chatgpt: Optimizing language models for dialogue, 2022. URL `https://openai. com/blog/chatgpt` . 

- [25] OpenAI. Gpt-4 technical report. _ArXiv_ , abs/2303.08774, 2023. URL `https://api. semanticscholar.org/CorpusID:257532815` . 

- [26] Keiran Paster. Testing language models on a held-out high school national finals exam. `https: //huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam` , 2023. 

- [27] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. _ArXiv_ , abs/2009.03393, 2020. URL `https://api.semanticscholar.org/CorpusID: 221535103` . 

- [28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _CoRR_ , abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL `https://doi.org/10.48550/arXiv.2402.03300` . 

- [29] Mike Sharples, David C. Hogg, Chris Hutchinson, Steve Torrance, and David J. Young. Computers and thought: A practical introduction to artificial intelligence. In _Proceedings of the Conference on Innovative Applications of Artificial Intelligence_ , 1989. URL `https://api.semanticscholar.org/CorpusID:61039423` . 

- [30] Christian Szegedy. A promising path towards autoformalization and general artificial intelligence. In _International Conference on Intelligent Computer Mathematics_ , 2020. URL `https://api.semanticscholar.org/CorpusID:220729524` . 

- [31] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. `https://github.com/tatsu-lab/stanford_alpaca` , 2023. 

- [32] Vernon Toh, Ratish Puduppully, and Nancy F. Chen. Veritymath: Advancing mathematical reasoning by self-verification through unit consistency. _ArXiv_ , abs/2311.07172, 2023. URL `https://api.semanticscholar.org/CorpusID:265149887` . 

- [33] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- [34] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _ArXiv_ , abs/2201.11903, 2022. URL `https://api.semanticscholar.org/CorpusID: 246411621` . 

- [35] Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E. Gonzalez, and Ion Stoica. Rethinking benchmark and contamination for language models with rephrased samples. _ArXiv_ , abs/2311.04850, 2023. URL `https://api.semanticscholar.org/CorpusID: 265050721` . 

- [36] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _ArXiv_ , abs/2305.10601, 2023. URL `https://api.semanticscholar.org/CorpusID: 258762525` . 

- [37] An-Zi Yen and Wei-Ling Hsu. Three questions concerning the use of large language models to facilitate mathematics learning. _ArXiv_ , abs/2310.13615, 2023. URL `https: //api.semanticscholar.org/CorpusID:264405766` . 

- [38] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

12 

- [39] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _ArXiv_ , abs/2309.05653, 2023. URL `https://api.semanticscholar.org/CorpusID: 261696697` . 

- [40] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Haotong Zhang, Joseph Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. _ArXiv_ , abs/2306.05685, 2023. URL `https://api.semanticscholar.org/CorpusID:259129398` . 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0014-00.png)


**----- Start of picture text -----**<br>
Pos Neg<br>Pred-Pos 960/1042 218/626<br>Pred-Neg 82/1042 408/626<br>**----- End of picture text -----**<br>


Table 3: This confusion matrix represents the accuracy of GPT4-Turbo-1106 in assessing 1668 incorrect solutions that were correctly identified with the right error step. The task for GPT4Turbo-1106 was to evaluate the correctness of the error reason provided by the evaluated model, in comparison with the actual ground truth labelled by expert. ’Pos’ and ’Neg’ represent the ground truth correctness of the provided explanation, while ’Pred-Pos’ and ’Pred-Neg’ indicate GPT4’s prediction about the correctness. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0014-02.png)


**----- Start of picture text -----**<br>
Prompt for Zero-Shot Scoring<br>Act as a grade school math teacher and score the following problem solution.<br>Question:<br>{data['question']}<br>Student Solution:<br>{sol_steps}<br>Your task involves three parts:<br>1. **Step-by-step Evaluation:** Go through the student solution carefully and identify<br>key errors and potential misunderstandings that led to the incorrect solution.<br>2. **Final Judgement:** Provide an overall judgement on the correctness of the<br>student's solution.<br>3. **First Error Step:** If the solution is incorrect, generate the step number where<br>the first error occurs, otherwise generate N/A here<br>4. **Error Analysis:** If the solution is incorrect, analyse the cause and reasons for<br>the first error step, otherwise generate N/A here<br>Here's the format I want:<br>Step-by-step Evaluation: [Provide a step by step examination of the student solution and<br>identify key errors and misunderstandings here.]<br>Final Judgement: [Insert only **correct** or **wrong** here]<br>First Error Step: [Insert either N/A or the step number where the first error occurs]<br>Error Analysis: [Insert either N/A or the analysis of error in the first error among<br>solution steps]<br>Please follow this format without any additional introductory or concluding statements.<br>**----- End of picture text -----**<br>


Figure 4: This is the zero shot prompt we used to evaluate all the models 

## **A Limitations** 

## **Does MR-GSM8k Require Human Labeling?** 

In this study, we proposed the MR-Score as a unified and normalized metric to evaluate the language models. The MR-Score consists of three sub-metrics, with the error reason being just one of the evaluation criteria. Similar to translation tasks, where expressions in one language may correspond to many variations in another, it is challenging to develop an automatic evaluator that scores the 

14 

```
Act as a grade school math teacher and score the following problem solution.
```


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0015-01.png)


**----- Start of picture text -----**<br>
Your task involves three parts:<br>1. **Step-by-step Evaluation:** Go through the student solution carefully and identify key errors<br>and potential misunderstandings that led to the incorrect solution.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0015-02.png)


**----- Start of picture text -----**<br>
2. **Final Judgement:** Provide an overall judgement on the correctness of the student's<br>solution.<br>**----- End of picture text -----**<br>


`3. **First Error Step:** If the solution is incorrect, generate the step number where the first error occurs, otherwise generate N/A here` 

`4. **Error Analysis:** If the solution is incorrect, analyse the cause and reasons for the first error step, otherwise generate N/A here` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0015-05.png)


**----- Start of picture text -----**<br>
Here's the format I want:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0015-06.png)


**----- Start of picture text -----**<br>
Step-by-step Evaluation: [Provide a step by step examination of the student solution and identify<br>key errors and misunderstandings here.]<br>Final Judgement: [Insert only **correct** or **wrong** here]<br>First Error Step: [Insert either N/A or the step number where the first error occurs]<br>Error Analysis: [Insert either N/A or the analysis of error in the first error among solution<br>steps]<br>Here is some demo examples:<br>{k_shot_demos}<br>Here is the question and solution for you to solve, please follow the desired format without any<br>additional introductory or concluding statements<br>Question:<br>{data['question']}<br>Student Solution:<br>{sol_steps}<br>**----- End of picture text -----**<br>


Figure 5: This is the few shot prompt we used to evaluate all the models 

error reason perfectly. Despite these limitations, this does not undermine the arguments we present, nor does it affect the cognitive deficiencies unveiled by this metric. To the best of our knowledge, GPT-4 has been the most popular choice for serving as an automatic evaluator across different metrics [40, 19]. In Appendix-C, we empirically demonstrate that GPT-4 can serve as a decent automatic evaluator, with the final MR-Score based on its labeling results closely matching those of manual labeling. 

## **Limitations of the Meta Reasoning Evaluation Paradigm and MR-GSM8k Dataset** 

Reflecting on Goodhart’s law, which states that "When a measure becomes a target, it ceases to be a good measure," it’s evident that the "reason about reasoning" paradigm is not immune to this phenomenon. This paradigm, like any other, can be targeted for optimization. However, our evaluation paradigm presents a greater challenge to overfitting compared to others, due to its demand for a comprehensive understanding within a broad error space, as illustrated in Section-B. 

On the other hand, the meta-reasoning evaluation framework in MR-GSM8K, while innovative, is not without its limitations. Firstly, its applicability may be restricted when it comes to subjects that are inherently holistic or creative in nature, such as humanities or sociology. These subjects often require a comprehensive understanding and modification (e.g. essay writing), which can be challenging to break down into specific, sequential reasoning steps and corrections. Secondly, MR-GSM8K is currently confined to questions in English. This could potentially limit the scope of reasoning challenges that can be explored, as different languages may present unique cognitive and linguistic hurdles. Lastly, the analysis and correction of errors in the reasoning steps are currently based on solutions generated by MetaMath-7B model only. It’s important to note that different LLMs and different individuals, may exhibit distinct reasoning and error patterns. Therefore, it would be beneficial to broaden the spectrum of solutions analyzed, incorporating a more diverse range of LLMs and even human responses. This would not only enhance the robustness of the evaluation framework but also provide a more nuanced understanding of the reasoning processes at play. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0016-00.png)


**----- Start of picture text -----**<br>
TRR VS Number of Correct Examples by Model TNR VS Number of Correct Examples by Model<br>100<br>2-correct 2-correct<br>1-correct 1-correct<br>0-correct 0-correct<br>80<br>60<br>40<br>20<br>0<br>Phi3-3.8B Llama3-8B Llama3-70B Phi3-3.8B Llama3-8B Llama3-70B<br>True Positive Rate (%) True Negative Rate (%)<br>**----- End of picture text -----**<br>


Figure 6: This is how the true positive and true negative rates change with the number of correct solutions in the 3-shot demonstrations. Note for the smaller models like Phi-3 and Llama3-8B, their susceptibility trend is opposite with that of Llama3-70B. 

|**Models**|**Step**|**Step+Reason/M**|**Step+Reason/A**|**MR-Score/M**|**MR-Score/A**|
|---|---|---|---|---|---|
|GPT4|823/1573|677/1573|732/1573|0.495|0.512|
|Claude2|331/1573|185/1573|224/1573|0.191|0.203|
|llama2-70B-MR|327/1573|99/1573|139/1573|0.105|0.118|
|GPT3.5|179/1573|73/1573|73/1573|0.097|0.097|
|MetaMath-70B|22/1573|6/1573|7/1573|0.013|0.013|
|Mammoth-70B|4/1573|1/1573|2/1573|0.012|0.012|
|WizardMath-70B|6/1573|1/1573|1/1573|0.001|0.001|



Table 4: Comparison of the manual labelling results and GPT4-Turbo-1106 labelling results. Step column shows the number that each evaluated models successfully located the first error steps among incorrect solutions. Step+Reason/M stands for the manual labelling results of the error reasons where its first error step is correct. Step+Reason/A corresponds to the labelling results of GPT4-Turbo-1106. llama2-70B-MR are llama2-70B model finetuned on the GSM8k training set and its meta-reasoning augmentation by GPT4. 

## **B In Domain Finetuning** 

Given the challenges posed by the novel "reason about reasoning" task paradigm, we explored how much targeted task-specific training data could enhance the performance of current state-of-the-art models on this task. We considered augmenting the GSM8K training set with diagnostics data in a similar format. However, due to the labor-intensive nature of manual annotation, we opted for a more feasible approach using an expert-designed procedure where GPT-4 generates the training data based solely on the original GSM8K problems, excluding any Program of Thought (POT) or reversed transformations. 

This process involved presenting GPT-4 with a question and its correct solution, then instructing it to introduce an error at a randomly chosen step and complete the solution accordingly. The step-by-step analysis was subsequently generated, focusing on the fabricated error. Despite GPT-4’s modest 40+% accuracy in correctly identifying incorrect solutions in the test set, this procedure successfully generated accurate diagnostic training data with over 90 percent accuracy, as verified by expert annotation on a sub-sampled set. This is largely due to the expert designed procedure (Figure-7) that greatly lowered the difficulties for instruction following. Note, occasionally GPT4 will fail to fabricate a valid error due to the lack of true understanding of errors (e.g., switch the fraction 8/3 to 2 and 2/3 then claims this is an error). 

16 

```
Hi ChatGPT, I need your assistance in creating a math dataset that focuses on annotated incorrect
solutions. The dataset is intended for grade school math.
Provided Information:
Question: {question}
Correct Solution Steps: {steps}
```

## `Procedure:` 

```
Understand the Question and Solution: Ensure you comprehend the question and the correct solution.
Fabricate Error Step and Reason: Randomly select a step and come up with an error that is both
realistic and commonly observed among students such as calculation mistake, conceptual
misunderstanding, or incorrect formula application etc.
Develop the Incorrect Solution: Base on the error step and error reason, rewrite the solution,
ensuring they are internally consistent but incorrect due to the initial error. **Do not mention
anything about incorrectness or deliberate error in the error step in your solution**.
Independent Analysis: Provide a step-by-step analysis of the solution, including the error, as if
encountering the problem for the first time. Do not reference the fact that the error was
intentionally created.
Summarize the Error: Based on your analysis, summarize the error reason for the first incorrect step.
Format:
Error Step: [Identify a step number where the first error in the correct solution should occur. Any
step from either the beginning, middle or the end will work.]
Error Reason: [Describe the nature of the error]
Incorrect Solution: [Present the solution, starting correctly but deviating post error step. DO NOT
MENTION ANYTHING ABOUT THE INCORRECTNESS.]
Independent Step-by-Step Analysis: [Analyse the solution independently, as if unaware of the
intentional error]
Independent Error Reason: [Conclude with a summary of the perceived error reason]
Please adhere strictly to this format, avoiding any additional introductory or concluding remarks.
```

Figure 7: This is the prompt we used for GPT4 to create in-domain training data. 

For our base model, we used llama-2-70B-base, consistent with the approach of other open-source SOTA math models. We merged the GSM8K training set with the GPT-4 generated diagnostic data, consisting of 5k incorrect solutions and 4k correct solutions. For fine-tuning, we employed the Qlora method [11], maintaining the same hyperparameters as used for MetaMath-70B. The evaluated results indicate a 31.74% true positive rate and 73.49% true negative rate, which lead to a 5.8% score in MCC. The accuracy for first-error-step and error-reason is 20.79% and 6.29% respectively. The MR-Score for this finetuned model is 10.5%. 

Notably, the fine-tuned Llama2 model demonstrated a distinct tendency from that of GPT3.5 and other open-source models; it was less inclined to accept solutions uncritically, tending instead to over-reject solutions regardless of their correctness. As depicted in Figure-8, of the 99 questions where the model accurately predicted both correctness and the first error step, a significant portion involved questions with POT and reversed reasoning types. This is particularly noteworthy given that the model was trained exclusively on original questions. 

Caution is necessary when interpreting the outcomes of in-domain fine-tuning. Although the finetuned model achieved improved results, it is important to recognize that the overall number of correct diagnoses for incorrect solutions remains relatively low (e.g., 6.29%). This underscores the challenging nature of our MR-GSM8k benchmark, where effective diagnosis across diverse solution spaces requires a comprehensive understanding of the problem. Consequently, simple fine-tuning strategies may not yield substantial improvements in performance. 

## **C Design thinking of MR-Score** 

The MR-Score is consist of three sub-metrics corresponds to the three sequential reasoning subtasks. For the first solution correctness prediction, we empirically noticed that most of the evaluated language models tend to either blindly classify the given solution as correct or incorrect, exemplified by the low true-positive/true-negative rate in Table-2. Therefore, we chose the MCC score instead of metrics like F1 or Balanced-Accuracy due to its value range. The models that have high true-positive 

17 

rate but low true negative rate will have near zero score under the MCC metric and vice versa. For the second and third tasks of locating first error step and elucidating error reason, we chose the simple accuracy metric. One of the reason is that locating the first error step is a multi-class classification problem and it is difficult to have large prediction bias while at the same time scores high accuracy. Similarly, the explaining error reason task is a free-form generation task that requires substantial understanding and a simple accuracy metric is enough to categorize the model behavior. 

As to the weights given to the three metrics, they are crafted by considering the task complexity and the difference between manual labelling and auto labelling results of the error reason. As discussed in Section-A, we chose the GPT-4-Turbo as our proxy evaluator and Table-4 is the results of autolabelling vs our expert manual-labelling. It is clear that the final MR-Score calculated from manual labelling VS auto labelling are very close to each other, exhibiting the potential of GPT4 to serve as a delegate evaluator for our task. 

Table-3 displays the confusion matrix based on GPT4’s labelling of all the error reasons. Notably, GPT4 is able to achieve 82% of overall accuracy despite a substantial higher false positive rate than false negative rate. However, we still encourage large tech companies, who have the resources to bear manual labelling costs, to release open-source manual labelling results when publishing findings using MR-GSM8k for the best of rigorousness. 

## **D Annotation Manuals, Examples and Case Studies** 

18 

**1 MR-GSM8K Annotation Manual** 

**Annotation Manual** This annotation task involves annotating math problems in grade school level. Specifically, for each math problem and its solution process, it is necessary to judge whether it is correct or not. If there is an error in the solution process, the erroneous step number and the reason for the error must be annotated. 

Below is the format description of the annotation data. 

**Problem:** The ”problem” part provides the math problem. 

**ground-truth-solution:** ”ground-truth-solution” refers to the reference standard answer for the math problem. For some problems, the solution process in this reference answer may be incorrect, but their final answers are correct. Nevertheless, please read this reference answer carefully and ensure you understand the problem before judging the annotated solution process. Report any problem if you believe the question is incorrect or missing any condition. 

**model-output-steps** ”model-output-steps” is the answer output by the language model and the object we need to annotate. Read each step of this solution process carefully and judge whether the thought process is correct. If there is an error, identify at which step the error begins. Note each step can be categorized as positive, neutral, or negative. Positive and neutral steps represent stages in the reasoning process where the correct final output remains attainable, whereas negative steps indicate a divergence from the path leading to the correct solution. We need to find the first step where an error occurs. Reasons for errors include, but are not limited to, calculation errors (e.g., 11*11=133 is a calculation error), formula errors (such as multiplying speed by days instead of hours, which is a unit error), misunderstanding errors (such as not understanding the problem and making nonsensical statements), or missing positive/negative signs. If a step is unnecessary, redundant, or repetitive, but does not directly cause an error or prevent reaching the final result, do not judge this step as an error. Note: (It was found that the solution steps of ten problems are not correctly segmented. If encountered, please manually segment them into Step 1, Step 2, etc., in the same format before annotating.) 

**model-output-answer-correctness** ”model-output-answer-correctness” compares the language model’s output with the standard answer. Its value can be ”correct” or ”wrong.” Even if some answers are marked as ”correct,” the solution process may still be incorrect, so the solution steps must still be judged step by step. Generally, this option does not need modification. If modification is necessary, please report the UUID number and keep a record. 

**model-output-solution-correctness** ”model-output-solution-correctness” is the first field we need to annotate. If ”model-output-answer-correctness” is ”correct,” you need to judge the solution process. If it is confirmed that the correct result is reached through the correct steps, this column should be marked ”correct;” otherwise, mark ”wrong.” If ”model-output-answer-correctness” is ”wrong,” this column defaults to ”wrong.” 

**model-output-solution-first–error-step** ”model-output-solution-first-error-step” is the second field we need to annotate. If the solution process in the first column is correct, then this column and the next column for the reason of the error should be filled with ”N/A.” If the solution process is incorrect, identify the first erroneous step and fill in this column with Arabic numerals ”1”, ”2”, ”3”, etc. 

**model-output-solution-first-error-reason** ”model-output-solution-first-error-reason” is the third column that needs annotation. If the solution process is correct, fill in ”N/A;” if incorrect, specify the reason for the first error in detail. Describe the erroneous process, what misunderstanding led to the incorrect inference, then provide the correct answer, rather than just stating the misunderstanding followed by the correct answer. The purpose of annotating reasons for errors is mainly for later evaluation of some open-source and closed-source models, like ChatGPT, GPT4, to assess their accuracy in identifying erroneous steps and reasons, and to compare their provided reasons for errors with those marked by humans. So, when explaining the reason for an error, assume you are explaining to another person why it is wrong, and the explanation should be as comprehensive as possible. Since there might be multiple perspectives to judge the model’s reasons for errors. All possible viewpoints should be mentioned. The ultimate goal is for another annotator, seeing your reasons for error and the model’s reasons, to be able to quickly compare and understand whether the model has correctly identified the reasons for the error. 

- Some examples of how to write reasons for errors: 

”According to the problem, John was stuck in traffic for the first two hours and did not immediately turn around to head home. A misunderstanding of the total time allocation for the return journey affected the calculation for each subsequent time segment.” ”Because Raymond is 6 years older than Samantha, and Samantha is currently 31 years old, Raymond should be 37 years old (31+6), not 25 years old (31-6). The model misunderstood the age difference.” ”The step incorrectly set up the equation. If the running speed is 300/x miles per hour, then the walking speed is one-quarter of the running speed, so the total distance traveled in six hours would be (300/x)*2 for running plus (300/x)*(1/4)*4 for walking, and their sum should be 18 miles.” 

This is the partial annotation manual we provide to the annotators. In the full annotation manual, we have an extra section including several examples detailing frequent asked questions, as well as cases for desired and undesired annotation results. 

**2 MR-GSM8K Examples** 

**uuid:** f25b796f-bc99-49c7-a6cd-ac5bb98412ce **question:** There are 6 girls in the park. If there are twice the number of boys in the park, how many kids are in the park? **ground truth solution** : There are 6 girls x 2 boys/girl = 6*2=12, 12 boys in the park. In total there are 6 girls + 12 boys = 6+12=18,18 kids in the park #### 18”, **ground truth answer** : 18 **model output steps:** Step 1: There are twice the number of boys in the park, so there are 2 boys/girl * 6 girls = 2*6=12, 12 boys in the park. Step 2: In total there are 12 boys + 6 girls = 12+6=18,18 kids in the park. Step 3: #### 18 **model output answer correctness:** correct **model output solution correctness** : correct **model output solution first error step** : N/A **model output solution first error reason** : N/A **question type** : original 

**——————————————————————————————————— uuid:** d4ec7076-6dcb-48e3-ba5b-b29d2adc483f 

**question:** If Chester eats 3 eggs a day for 30 days and then increases it to 5 eggs a day for 30 days, how many dozens of eggs will Chester need for 60 days? **ground truth solution** : 

He starts off eating 3 eggs a day for 30 days for a total of 3*30=90 eggs.Then he increases it to 5 eggs a day for 30 days for a total of 5*30 = 150 eggs. All total he will eat 90+150 = 240 eggs. There are 12 eggs in 1 dozen and he will 240 eggs which is 240/12 = 20 dozen eggs. #### 20”, **ground truth answer** : 20 **model output steps:** Step 1: months ~~e~~ ating ~~3~~ eggs=2. Step 2: increased ~~e~~ ggs=5 Step 3: days ~~e~~ ating ~~5~~ eggs=30 Step 4: months ~~e~~ ating ~~5~~ eggs=6 Step 5: total ~~d~~ ays=60 Step 6: total ~~m~~ onths=6+2 Step 7: total ~~e~~ ggs ~~m~~ onths=months ~~e~~ ating ~~3 e~~ ggs*30+months ~~e~~ ating ~~5 e~~ ggs*days ~~e~~ ating ~~5 e~~ ggs+total ~~m~~ onths Step 8: total ~~d~~ ozen eggs=total ~~e~~ ggs ~~m~~ onths/12 Step 9: print(total ~~d~~ ozen ~~e~~ ggs) 

**model output answer correctness:** wrong 

**model output solution correctness** : wrong 

**model output solution first error step** : 1 

**model output solution first error reason** : The solution assumes the months eating three eggs are 2 which is unsupported since there is only 30 days for eating three eggs per day. **question type** : Program of Thought 

**——————————————————————————————————— uuid:** 18eb1c66-ede5-4504-8831-a3154eae6938 **question:** Tom gets X car washes a month. If each car wash costs 15 how much does he pay in a year? If we know the answer to the above question is 720, what is the value of unknown variable ’X’? **ground truth solution** : 

Since there are 12 months a year and the total amount Tom paid for is known to be 720, then the monthly pay is 720/12=60. Given that each car wash costs 15 dollars, we know that the car Tom washes a month equals to 60/15=4. #### 4”, 

**ground truth answer** : 4 

**model output steps:** Step 1: We know that Tom gets X car washes a month. Step 2: Each car wash costs 15. 

Step 3: To find out how much Tom pays in a year, we need to multiply the monthly cost by the number of months in a year. Step 4: So, the total cost for a year is 15*X * 12 = 180X. Step 5: We are given that the total cost for a year is 720, so we can write: 180X = 720. Step 6: Dividing both sides of the equation by 180, we get: X = 4. Step 7: The value of X is 4. Step 8: 4 Step 9: The answer is: 4 **model output answer correctness:** correct **model output solution correctness** : correct **model output solution first error step** : N/A **model output solution first error reason** : N/A **question type** : Reversed 

Above are the annotated samples extracted from the MR-GSM8K dataset, with one question from each type. The same problems can be found in the MR-GSM8K.json file in our open sourced repository. 

## **3 Case Studies of Response from Evaluated Models** 

**question:** two sports coaches went shopping together. The baseball coach bought x new baseballs for $3 each. The basketball coach bought 8 new basketballs for $14 each. How much more did the basketball coach spend than the baseball coach? If we know the answer of the above question is 85, what is the value of unknown variable x? **model output steps:** Step 1: The baseball coach bought x new baseballs for $3 each, so the total cost for the baseballs is 3x. Step 2: The basketball coach bought 8 new basketballs for $14 each, so the total cost for the basketballs is 8 * $14 = $112. Step 3: To find out how much more the basketball coach spent than the baseball coach, we need to subtract the cost of the baseballs from the cost of the basketballs: $112 - 3x. Step 4: We are given that the difference in cost is $85, so we can write: $112 - 3x = $85. Step 5: Simplifying the left side, we get: $112 - $85 = 3x. Step 6: Subtracting $85 from $112, we get: $27 = 3x. Step 7: Dividing both sides by 3, we get: $9 = x. **Final Judgement by GPT-4-Turbo:** wrong **First Error Step by GPT-4-Turbo:** step 4 

**Error Analysis by GPT-4-Turbo:** The student incorrectly simplified the equation in Step 4. The correct equation should be 3x = $112 - $85, not $112 - 3x = $85. This error led to the incorrect calculation in Step 5. 

**——————————————————————————————————— question:** Trinity sells magazines at 11/8 of the price she bought the magazines. If she bought the magazines at $72, what is her profit? **model output steps:** Step 1: If Trinity bought the magazines at $72, then she sells them at 11/8 * $72 = $90 Step 2: Her profit is the selling price minus the buying price, so her profit is $90 - $72 = $18 Step 3: #### 18 

**Final Judgement by GPT-4-Turbo:** Correct **First Error Step by GPT-4-Turbo:** N/A **Error Analysis by GPT-4-Turbo:** N/A 

**——————————————————————————————————— question:** Christina is planning a birthday party and needs .75 gift bags per invited guest, because 1/4 of attendees don’t show up. She invited 16 friends. Gift bags are $2 each. How much will she spend? **model output steps:** Step 1: no show = 1/4 Step 2: invited ~~g~~ uests = 16 Step 3: gift ~~b~~ ags ~~n~~ eeded = invited ~~g~~ uests * 0.75 Step 4: total ~~g~~ ift ~~b~~ ags = no ~~s~~ how + gift ~~b~~ ags ~~n~~ eeded Step 5: total ~~c~~ ost = total ~~g~~ ift ~~b~~ ags * 2 Step 6: print(total ~~c~~ ost) 

**Final Judgement by Claude-3-Sonnet:** Wrong **First Error Step by Claude-3-Sonnet:** Step 4 **Error Analysis by Claude-3-Sonnet:** The student incorrectly adds the no-show fraction (1/4) to the gift ~~b~~ ags ~~n~~ eeded instead of subtracting it from the invited ~~g~~ uests before multiplying by the required gift bags per guest (0.75). This error likely stems from a misunderstanding of how to account for the no-show guests in the calculation. 

Above are the response from the evaluated models. A number of errors can be observed in their responses. For the first response we sampled, GPT-4-Turbo fail to realize that the two equations it is referring to are actually equivalent, thus demonstrating the typical ’reversal curse’. For the second example, GPT-4-Turbo fail to capture the obvious calculation error despite the simplicity of the problem. For the last response, Claude-3-Sonnet hallucinate on the error reason and provide a correction with yet another unit error, demonstrating a lack of ontological understanding on the objects involved in the calculation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation_images/iclr_2025_MR-GSM8K_A_Meta-Reasoning_Benchmark_for_Large_Language_Model_Evaluation.pdf-0022-00.png)


**----- Start of picture text -----**<br>
Problem Types of Solutions Llama2-70B Correctly<br>Identified<br>Backward<br>23%<br>POT<br>12% Original<br>65%<br>Original POT Backward<br>Figure 8: Problem types of incorrect solutions for which the llama2-70B-MR model has successfully<br>identified both the error step and error reason. Note that the training set only included solutions from<br>original problems.<br>**----- End of picture text -----**<br>


22 

