## MIRAGE: EVALUATING AND EXPLAINING INDUCTIVE REASONING PROCESS IN LANGUAGE MODELS 

**Jiachun Li[1,2] , Pengfei Cao[1,2] , Zhuoran Jin[1,2] , Yubo Chen[1,2] , Kang Liu[1,2] , Jun Zhao[1,2]** 

1School of Artificial Intelligence, University of Chinese Academy of Sciences 

2The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences jiachun.li@nlpr.ia.ac.cn 

## ABSTRACT 

Inductive reasoning is an essential capability for large language models (LLMs) to achieve higher intelligence, which requires the model to generalize rules from observed facts and then apply them to unseen examples. We present MIRAGE, a synthetic dataset that addresses the limitations of previous work, specifically the lack of comprehensive evaluation and flexible test data. In it, we evaluate LLMs’ capabilities in both the inductive and deductive stages, allowing for flexible variation in input distribution, task scenario, and task difficulty to analyze the factors influencing LLMs’ inductive reasoning. Based on these multi-faceted evaluations, we demonstrate that the LLM is a poor rule-based reasoner. In many cases, when conducting inductive reasoning, they do not rely on a correct rule to answer the unseen case. From the perspectives of different prompting methods, observation numbers, and task forms, models tend to consistently conduct correct deduction without correct inductive rules. Besides, we find that LLMs are good neighborbased reasoners. In the inductive reasoning process, the model tends to focus on observed facts that are close to the current test example in feature space. By leveraging these similar examples, the model maintains strong inductive capabilities within a localized region, significantly improving its deductive performance. 

## 1 INTRODUCTION 

Inductive reasoning, known as the ability of an intelligent agent to infer abstract rules from limited observations and apply them to new examples, is crucial for large language model (LLMs) progressing toward artificial general intelligence (AGI) (Xu et al., 2024b; Sun et al., 2024; Wang et al., 2024b). As illustrated in Figure 1, given a set of observed facts, inductive reasoning process expect the model to generate abstract rules from the provided facts (i.e. [A,B,C] _→_ [B+C,B+C,C] in the inductive task) and apply these rules to answer specific new questions (i.e. [3,4,7] _→_ [11,11,7] in the deductive task). Despite its significant research value, it has been relatively neglected compared to other types of reasoning (e.g., math reasoning, multi-hop reasoning, etc.). 

Recently, some works have started to explore this problem. They primarily evaluate the model’s inductive reasoning capabilities using various datasets (Shao et al., 2024; Cheng et al., 2024; Qiu et al., 2024; Jiang et al., 2024). Though they have made great progress, their works still have two main limitations: **(1) Previous works lack comprehensive evaluation.** Most works have only one evaluation task: the inductive task on collected rules (Yang et al., 2024b; Shao et al., 2024) or the deductive task on specific test samples (Chollet, 2019; Xu et al., 2024a; Qiu et al., 2024). Therefore, they can only evaluate the rule induction performance or final results of inductive reasoning, instead of comprehensively analyzing the whole process (i.e. inductive + deductive). **(2) Previous works lack flexible test data.** Most former datasets evaluate the overall performance of models by collecting observation and test examples under the same rules (Rule, 2020; Kim et al., 2022; Lake et al., 2019). However, due to the absence of transformation rules, it is impossible to extend these examples, resulting in a fixed test set. This limitation makes it challenging to assess the impact of factors such as distribution, quantity, and form of input examples on the model’s inductive reasoning, thereby hindering a deeper analysis of the model’s reasoning mechanisms. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Observed Facts<br>[0,1,3] → [4,4,3]     [5,2,5] → [7,7,5]     [2,1,8] → [9,9,8]     [9,6,3] → [9,9,3]      [3,3,7] → [10,10,7]<br>Inductive Task<br>What is the rule? What is the rule?<br>[A,B,C]  →  [B+C,B+C,C] [A,B,C] → [A+4,B+1,C]<br>Deductive Task<br>[3,4,7] → ? [3,4,7] → ?<br>Rule: [A,B,C] → [B+C,B+C,C] Case: [3,3,7] → [10,10,7]<br>Answer: [11,11,7] Answer: [11,11,7]<br>Rule-based Neighbor-based<br>**----- End of picture text -----**<br>


Figure 1: An overview of two paradigms (i.e. rule-based and neighbor-based) in inductive reasoning. 

In this paper, we present **MIRAGE** (Meta Inductive ReAsoning Evaluation), a dataset designed to address the two aforementioned limitations. It includes both inductive and deductive evaluation tasks, while offering flexibility to construct test data with various forms, arbitrary input distributions, and controllable difficulties. In detail, we first construct a rule library based on various vector operations (e.g., [A,B,C] _→_ [B+C,B+C,C] as shown in Figure 1). Using the automatically synthesized rules, we can generate facts arbitrarily through instantiation, ensuring the flexibility and scalability of the test data. Next, we filter out the noise data (e.g. duplicated facts) to further improve the effectiveness and quality of our dataset. Finally, to comprehensively evaluate the inductive reasoning process, we not only design inductive and deductive questions based on the synthesized data but also construct diverse application scenarios for these tasks, including list transformations, real-world problems, code generations, and string transformations (as shown in Figure 2). 

Based on our dataset, we perform a deeper analysis of the model’s inductive reasoning process, from which we draw two new conclusions about the inductive reasoning mechanisms of LLMs: **(1) Language models are poor rule-based reasoners.** As shown in the left column of Figure 1, in the rule-based reasoning paradigm, inductive reasoning involves first deriving the correct rule through the observation of examples and then using the inductive rule to answer new questions (like what humans do). However, we find that LLMs perform poorly in this paradigm: In many cases, though they can not induce a correct rule, they can still perform well on deductive tasks. Through experimentation, we observe this performance gap between induction and deduction across different prompting methods, models, observed example numbers, and scenarios. This indicates that the final performance of the LLM’s inductive reasoning rarely relies on the intermediate inductive rules. **(2) Language models are good neighbor-based reasoners.** Furthermore, we identify an important mechanism behind LLM’s inductive reasoning, which we refer to as “ **neighbor-based reasoning** ”: If some observed facts are close to the test examples in feature space, the model tends to leverage this similarity to improve performance on deductive tasks. For example, as shown in the right column of Figure 1, even when the model cannot generate the correct rule, it can rely on the neighbor fact [3,3,7] _→_ [10,10,7] (here the distance between [3,3,7] and [3,4,7] is small, so we refer to them as neighbors) to successfully performs deductive tasks. We demonstrate that this paradigm persists across different scenarios, models, and observed example numbers. However, it can only enhance the performance within a localized scope. 

To sum up, the main contributions of our work are as follows: **(1) We present a new dataset MIRAGE** , through it, we can comprehensively evaluate the LLM’s inductive reasoning process under more flexible settings. **(2) We prove that LLM is a poor rule-based inductive reasoner** . In many cases, it does not rely on inductive rules to make correct deductions. **(3) We prove that LLM is a neighbor-based inductive reasoner** . When performing inductive reasoning, models rely on the neighbor facts in the observed fact set to get better performance. 

2 

## 2 DATA CONSTRUCTION 

In this section, we describe the whole pipeline to build MIRAGE. We start by constructing rules based on five basic operations ( _§_ 2.1). Next, we substitute the instantiate vectors into the rules to generate facts ( _§_ 2.2) and apply filtering to them ( _§_ 2.3). Finally, we transform the facts into different scenarios, creating questions to evaluate the LLM’s inductive reasoning performance ( _§_ 2.4). 

## 2.1 RULE GENERATION 

According to previous work and relevant definitions (Huber, 2017; Han et al., 2024), in inductive reasoning, for each observed fact X _k_ = ( _**x** ,_ _**y**_ ), the input vector _**x**_ is transformed into the output vector _**y**_ according to a certain rule _f_ , i.e.: 

_f_ ( _**x**_ ) = _**y** , ∀_ ( _**x** ,_ _**y**_ ) _∈_ X (1) 

where X is the observed fact set under the rule _f_ . We believe that _f_ is the core of the problem, as it allows us to generate additional facts for X based on the rule automatically. Conversely, inferring _f_ from X requires significantly more effort due to the vast range of possible rules. Therefore, we first consider automating these rules’ large-scale synthesis. 

Based on previous representative datasets (Chollet, 2019; Rule, 2020; Xu et al., 2024a), we summarize the main types of rules, resulting in five atomic operations in this dataset: 

- **Add:** The operation adds certain components together. For example: [ _x, y, z_ ] _→_ [ _x, x_ + _y, z_ ]. 

- **Copy:** The operation copies some components to others. For example: [ _x, y, z_ ] _→_ [ _x, x, z_ ]. 

- **Map:** The operation applies a linear transformation to some components. For example: [ _x, y, z_ ] _→_ [ _x, ky_ + _b, z_ ]. Here, to avoid the interference of complex mathematical calculations, we have _k ∈_ [1 _,_ 9] and _b ∈_ [0 _,_ 9]. 

- **Pad:** The operation fills certain components with constant values. For example: [ _x, y, z_ ] _→_ [ _x, c, c_ ], where _c ∈_ [0 _,_ 9]. 

- **Swap:** The operation swaps certain components. For example: [ _x, y, z_ ] _→_ [ _z, y, x_ ]. 

For each operation _O_ , we randomly initialize the set index vector _**d**_ on which the operation applies and the index vector _**r**_ where the result is output. Specifically, for _x ∈_ _**x**_ , _y ∈_ _**y**_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0003-13.png)


where _ri_ = _j_ and [ _·_ ] _i_ represents the _i_ -th component. Therefore, we can generate a meta-rule _f_ = ( _O,_ _**d** ,_ _**r**_ ). Through sampling ( _O,_ _**d** ,_ _**r**_ ) randomly, we can construct a meta-rule library F. 

## 2.2 FACT GENERATION 

After generating the rule library, we can randomly initialize _**x**_ , and apply a specific rule _f ∈_ F to get _**y**_ . We repeat this process to generate the fact set X under the rule _f_ . All the ( _**x** ,_ _**y**_ ) _∈_ X are used for the LLM to induce the rule _f_ . It is worth noting that we can control the inductive difficulty by adjusting two factors: the dimension _D_ of _**x** ,_ _**y**_ and the fact number _N_ of X. As an example, in Figure 1, _D_ is 3 and _N_ is 5. Empirically, a higher _D_ and a smaller _N_ tend to increase the task difficulty. Additionally, to avoid the interference of complex mathematical calculations in evaluating inductive reasoning ability, we restrict the elements in each _**x**_ to integers between 0 and 9.[1] Since we can synthesize any _D_ -dimensional vector _**x**_ to construct a fact, we can flexibly control the input distribution. 

## 2.3 DATA FILTERING 

To ensure the quality of the dataset and the effectiveness of the evaluation, we need to filter out some noisy data. The following filtering steps are applied: **(1) Filtering out duplicate facts.** For any two facts in X, if their input vectors _**x**_ are identical, one of them is removed and resampled. This ensures that for each rule, all observed facts are unique. **(2) Filtering out duplicate rules.** To ensure diversity in the evaluation, we also remove duplicate rules, which have the same ( _O,_ _**d** ,_ _**r**_ ). **(3) Filtering out trivial facts.** After random sampling, X may include some trivial facts that provide little value for model induction, such as facts like _**x**_ = _**y**_ , _**x**_ = 0, or _**y**_ = 0. We filter the data to ensure that each X contains at most one trivial fact, thereby limiting the noise that could affect the model’s inductive reasoning process. 

> 1Our pilot experiments indicate that, under these constraints, most of the models can achieve an accuracy of nearly 100% in performing purely mathematical operations. See Appendix A.2 for details. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0004-00.png)


**----- Start of picture text -----**<br>
List Transformation Real-world Problem Code Generation String Transformation<br>Rule: Rule: Rule: Rule:<br>Meta Rule Jack went to trade items. He<br>[A,B,C] -> [B+C, B+C, C] [A,B,C] -> [B+C,B+C,C] originally had bags,  C chairs. After the A beds,  B def f(A,B,C):       A,B,C = B+C, B+C, C ABC -> BCBCC<br>trade, he has  B+C beds,       return A,B,C<br>B+C bags, C chairs.<br>Fact: Fact: Fact: Fact:<br>Meta Fact Jack went to trade items. He<br>[3,3,6] -> [9, 9, 6] [3,3,6] -> [9,9,6] originally had 3 beds, 3 bags,  f(3,3,6) = 9,9,6 ddg -> dgdgg<br>6 chairs. After the trade, he<br>has 9 beds, 9 bags, 6 chairs.<br>**----- End of picture text -----**<br>


Figure 2: Examples in four different scenarios of MIRAGE. 

## 2.4 QUESTION GENERATION 

So far, we have constructed all the metadata that we need to generate specific questions. It is worth noting that both F and X contain only abstract rules and facts, without any specific context. Therefore, they represent the fundamental inductive reasoning test data, which is why we refer to them as meta-rules and meta-facts. As shown in Figure 2, to evaluate the practical inductive reasoning capability of models, we apply these metadata to various scenarios to generate concrete problems. Specifically, we have: 

- **List Transformation (LT):** List transformation is the primary format used in previous inductive reasoning tasks (Rule, 2020; Xu et al., 2024a; Chollet, 2019), and here we adopt this approach as well. We transform all fact vectors into one-dimensional lists and require the model to inductively infer the transformation rules applied to these lists. 

- **Real-world Problem (RP):** Previous datasets lack tests for inductive reasoning capabilities in realworld scenarios (Rule, 2020; Xu et al., 2024a; Qiu et al., 2024).[2] To mitigate this gap, we populate the metadata into different natural language templates across five real-life scenarios. The example in Figure 2 describes a trading scenario, where we use different items to represent different dimensions of the vector. All item transactions follow the same rule. 

- **Code Generation (CG):** For each fact, we use _**x**_ as the input and _**y**_ as the output of a function. The model is then tasked with predicting the corresponding Python function. 

- **String Transformation (ST):** The former three scenarios are related to numbers. Here, we replace the basic elements in the fact vectors with characters to conduct a new test. Notably, we modify the operations as follows: addition in the Add and Map operations is replaced with string concatenation, multiplication in Map is replaced with character replication, zero-padding in Pad becomes character deletion, and the numbers 0-9 are replaced with the characters _a_ - _j_ . 

For each scenario, we design both inductive (Ind) and deductive (Ded) tasks. In the inductive task, given a set of observed facts X, we ask the model to generate the rule _f_ in a specified format (e.g. a Python Function for CG) and then evaluate the **accuracy** of the generation. In the deductive task, using the same observed facts, we provide an unseen example input _**x** t_ as input and measure the **accuracy** of the predicted _**y** t_ . We provide all the prompts used for these tasks in Appendix A.3. 

## 3 LANGUAGE MODELS ARE POOR RULE-BASED REASONERS 

## 3.1 OVERALL PERFORMANCES ON MIRAGE 

**Setup** We first evaluate the overall performance of various LLMs on MIRAGE. Here, we select GPT-4 (OpenAI, 2023), GPT-4o, Claude-3.5, Llama3-8B (Dubey et al., 2024), and Llama2-13B (Touvron et al., 2023) as representative models.[3] For the first three models, given their strong instruction-following capabilities, we provide only the instruction and allow them to answer the questions in a zero-shot setting. For the latter two models, to improve the format accuracy of the response, we additionally provide five examples before they answer the questions. Unless otherwise specified, we continue to use this setup to prompt the model in the subsequent experiments. For the dataset setting, we fix the size _N_ at 5 and measure performance across four scenarios when the dimension _D_ = 3, 5, 8. We sample 500 questions for each test. More implementation details can be found in Appendix B.1. 

> 2Here, real-world scenarios refer to mathematical inductive reasoning within natural language contexts. 

> 3Due to the frequency limitations of API calls, we can not conduct our evaluation on the latest o1 model. 

4 

|**Model**<br>**Task**|**D=3**<br>LT<br>RP<br>CG<br>ST|**D=5**<br>LT<br>RP<br>CG<br>ST|**D=8**<br>LT<br>RP<br>CG<br>ST|
|---|---|---|---|
|Llama2-13B<br>Ind<br>Ded|0.01<br>0.00<br>0.00<br>0.03<br>0.26<br>0.11<br>0.25<br>0.22|0.01<br>0.01<br>0.00<br>0.21<br>0.13<br>0.03<br>0.14<br>0.25|0.00<br>0.01<br>0.00<br>0.10<br>0.06<br>0.01<br>0.06<br>0.19|
|Llama3-8B<br>Ind<br>Ded|0.15<br>0.11<br>0.19<br>0.19<br>0.30<br>0.15<br>0.25<br>0.25|0.23<br>0.04<br>0.14<br>0.22<br>0.20<br>0.12<br>0.25<br>0.29|0.16<br>0.02<br>0.08<br>0.21<br>0.09<br>0.11<br>0.16<br>0.24|
|GPT-4o<br>Ind<br>Ded|0.41<br>0.32<br>0.38<br>0.32<br>0.68<br>0.37<br>0.61<br>0.56|0.35<br>0.21<br>0.44<br>0.30<br>0.58<br>0.25<br>0.64<br>0.39|0.33<br>**0.16**<br>0.41<br>0.24<br>0.42<br>0.17<br>0.49<br>0.29|
|GPT-4<br>Ind<br>Ded|**0.47**<br>0.29<br>**0.41**<br>0.28<br>0.68<br>0.37<br>0.61<br>0.57|**0.58**<br>**0.22**<br>**0.56**<br>0.27<br>0.63<br>0.29<br>0.71<br>0.44|**0.46**<br>0.15<br>**0.45**<br>0.23<br>0.42<br>0.21<br>0.64<br>0.30|
|Claude-3.5<br>Ind<br>Ded|0.44<br>**0.35**<br>0.34<br>**0.46**<br>0.79<br>0.45<br>0.62<br>0.58|0.22<br>0.20<br>0.38<br>**0.33**<br>0.65<br>0.33<br>0.76<br>0.45|0.24<br>0.13<br>0.38<br>**0.26**<br>0.46<br>0.24<br>0.59<br>0.30|



Table 1: Overall performance of different models on MIRAGE. The best results in the inductive task (Ind) are in **bold** , while the best results in the deductive task (Ded) are underlined. 

**Results** The results are shown in Table 1, from which we can draw the following conclusions: **(1) LLMs’ deductive reasoning does not rely on rule induction.** Given the same set of observed facts, the model’s performance on inductive tasks is noticeably worse than on deductive tasks in almost all cases. This suggests that most of the model’s correct deductions do not depend on inducing a correct rule. **(2) LLMs face difficulties in handling inductive reasoning in real-world problems.** When comparing different scenarios, all models perform the worst on the RP tasks. For example, GPT-4o only achieves **0.16** and **0.17** accuracy when the dimension is 8. This indicates that, compared to purely symbolic forms (LT, CG, ST), natural language forms pose a greater challenge for the models’ inductive reasoning abilities. 

**Supplementary Experiments** In the main experiments, we find that there is a significant performance gap between inductive and deductive tasks for LLMs. However, this gap may be caused by the difference in difficulty between the two tasks. When the model is unable to perform correct inductive reasoning, it is likely to guess the correct answers 

for the deductive task more easily compared to the inductive task, resulting in **Model Inductive Deductive** a higher accuracy. We conduct this adBF AF CR BF AF CR ditional experiment to eliminate the inGPT-4o 0.50 0.13 0.74 0.66 0.15 0.77 terference of this factor. Specifically, we Claude-3.5 0.37 0.07 0.81 0.65 0.22 0.66 randomly perturb one fact in X to violate rule _f_ . Then, we observe the performance of both tasks and calculate the change rate Table 2: Comparison of CR on two tasks ( _D_ = 3, _N_ = 3). BF (CR) of accuracy before and after the perand AF indicate the accuracy before and after perturbation. turbation. CR represents the sensitivity of the model’s performance to the input. If 

CR is high, it indicates a strong correlation between task performance and input, making it difficult to answer the question correctly through random guessing, Therefore, CR can serve as an indicator of the reasoning difficulty for the task. We randomly choose 100 pieces of test data from the dataset and generate questions under the LF scenario.[4] The experimental results on different models are demonstrated in Table 2. We can observe that the two tasks have comparable CR for both models, indicating that the reasoning difficulty of the deductive task is not lower than the inductive task. The tasks themselves do not cause such a large performance gap. 

## 3.2 PERFORMANCES OF ADVANCED METHODS 

In _§_ 3.1, we observe that LLMs perform poorly on our dataset, especially in inductive tasks. Considering previous work has proposed numerous methods to elicit the model’s reasoning abilities (Wei et al., 2022; Wang et al., 2023b; Madaan et al., 2023), we wonder whether they can boost models’ performance on MIRAGE. 

**Setup** Since we focus on exploring the model’s intrinsic capabilities, we only consider methods that do not introduce any external tools or knowledge. Specifically, the methods are as follows: **Input-Output (IO):** We prompt models to generate answers directly under different shots. **Inductive-Deductive (ID):** We prompt models to generate rules for inductive tasks and apply them to answer questions in deductive tasks. **Chain-ofThought (CoT)** (Wei et al., 2022): We prompt models to generate rationales and answers for the two tasks. **Self-Consistency (SC)** (Wang et al., 2023b): Based on CoT, we sample _n_ rationales and use the major voting 

> 4Unless otherwise specified, this configuration will be maintained for all subsequent experiments. 

5 

|**Method**|**LT**<br>Ind<br>Ded<br>(∆)|**RP**<br>Ind<br>Ded<br>(∆)|**CG**<br>Ind<br>Ded<br>(∆)|**ST**<br>Ind<br>Ded<br>(∆)|
|---|---|---|---|---|
|IO(0-shot)<br>IO(5-shot)|0.46<br>0.76<br>**0.30**<br>0.63<br>0.76<br>0.13|0.43<br>0.72<br>**0.28**<br>**0.59**<br>0.77<br>0.17|0.39<br>0.46<br>0.08<br>**0.55**<br>0.54<br>-0.02|0.47<br>0.70<br>0.23<br>0.52<br>**0.78**<br>0.26|
|ID(0-shot)<br>ID(5-shot)|0.46<br>0.56<br>0.11<br>0.59<br>0.68<br>0.08|0.46<br>0.57<br>0.11<br>0.57<br>0.66<br>0.08|0.33<br>0.42<br>0.09<br>0.47<br>0.54<br>0.08|0.22<br>0.65<br>**0.43**<br>0.48<br>0.69<br>0.21|
|CoT(0-shot)<br>CoT(5-shot)|0.50<br>0.57<br>0.07<br>0.56<br>0.59<br>0.04|0.47<br>0.55<br>0.08<br>0.41<br>0.55<br>0.13|0.34<br>0.39<br>0.05<br>0.37<br>0.40<br>0.03|0.52<br>0.62<br>0.10<br>0.45<br>0.64<br>0.20|
|SC(n=5)|0.59<br>0.74<br>0.15|0.49<br>0.62<br>0.14|0.38<br>0.45<br>0.07|**0.57**<br>0.68<br>0.10|
|SR(t=3)|0.48<br>0.64<br>0.16|0.42<br>0.67<br>0.25|0.36<br>0.49<br>**0.13**|0.53<br>0.67<br>0.14|
|HR(t=3, n=1)<br>HR(t=3, n=5)|0.56<br>0.68<br>0.12<br>**0.66**<br>**0.79**<br>0.13|0.45<br>0.71<br>0.26<br>**0.59**<br>**0.80**<br>0.21|0.41<br>0.53<br>0.11<br>**0.55**<br>**0.60**<br>0.05|0.43<br>0.71<br>0.27<br>0.49<br>0.67<br>0.18|



Table 3: Performance of different methods on MIRAGE using GPT-4o. The best results in each column are highlighted in **bold** , while the second best results are underlined. 

strategy to predict the final answer. **Self-Refine (SR)** (Madaan et al., 2023): We prompt the model to provide feedback on the generated rules, and then refine the rules based on that feedback (with a maximum of _t_ iterations). After the iteration stops, we use the latest rule to answer the inductive task and apply it to answer the deductive task. **Hypothesis Refinement (HR)** (Qiu et al., 2024): HR is an optimized version of SR, which first generates _n_ rules. In each iteration, we apply the current rules to all observed examples, compare the actual output with the expected output, and get the number of correct predictions along with the information about incorrect examples. If a candidate rule is correct for all observed facts, it is immediately returned. Otherwise, the rule with the highest number of correct predictions and the associated error information is used as input for the model to refine, generating _n_ rules for the next round, until the maximum number of iterations _t_ is reached. We sample 200 questions for each test. 

**Results** We demonstrate the experimental results on GPT-4o in Table 3 and report other results in Appendix B.2. We can conclude that: **(1) Advanced methods provide limited improvement to the model’s inductive reasoning ability or may even have negative effects.** For both tasks, directly answering with few-shot settings can consistently achieve the highest or second-highest accuracy in most cases. After applying prompting methods like CoT, the model’s accuracy decreases by up to **18%** and **22%** on two tasks, respectively. It indicates that the key to optimizing inductive reasoning does not lie in refining the intermediate inductive process (as CoT-like methods do). **(2) The model’s disregard for abstract rules during inductive reasoning is methodagnostic.** Although some methods use instructions to guide the model to focus more on the previously induced rules during reasoning (e.g. ID, SR, HR), there remains a significant gap in the model’s inductive and deductive performance. For example, in the case of SR, the model’s deductive accuracy outperforms its inductive accuracy by an average of **16%** . 

## 3.3 IMPACT OF INCREASING FACT SIZE 

In the previous experiments, we consistently fix the observed fact numbers _N_ . Therefore, as a supplement, we explore the impact of _N_ on the model’s inductive reasoning process in this section. Theoretically, as the number of observed facts increases, the scope of the candidate rules narrows, which can lead to the incorrect inductive process becoming correct. If the reasoning process is rule-based, the model is likely to generate the correct rule (inductive) before applying it correctly (deductive). In other words, the time when the LLM induces the correct rule is no later than the time it performs the correct deduction. Thus, the cumulative number of observations required for the inductive rule to change from incorrect to correct should not exceed the number required for the test case to become correct. We design this experiment to validate whether it holds on the LLM. 

**Setup** Given a fact set X _[k]_ of size _k_ , and a fixed test input _**x** t_ , we define the inductive correction threshold (ICT) and deductive correction threshold (DCT) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0006-07.png)


Here, _M_ ( _·|I_ ) _, M_ ( _·|D_ ) are the model’s outputs in inductive and deductive tasks. We set _D_ = 5 and vary _N_ , then analyze the distribution of these two thresholds across 100 samples, reporting the results in Figure 3. 

**Results** Based on the result, we further prove that **LLM’s deduction does not rely on an inductive rule** . From both of the two figures, we can observe that most points are distributed in the upper left region of the line 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0007-00.png)


**----- Start of picture text -----**<br>
8 14 8 12 0.8 0.8<br>7 12 7 10 0.77 0.38 0.65 0.7 0.79 0.46 0.63<br>0.7<br>6 10 6<br>5 8 5 8 0.4 0.46 0.34 0.6 0.44 0.51 0.41 0.6<br>4 6 4 6 0.5<br>32 4 32 4 0.71 0.37 0.73 0.4 0.71 0.41 0.73 0.5<br>1 1 2 3 4DCT5 6 7 8 2 1 1 2 DCT3 4 5 2 LF ObservedRP CG 0.3 LF ObservedRP CG 0.4<br>(a) GPT-4o (b) Claude-3.5 (a) GPT-4o (b) Claude-3.5<br>LF LF<br>ICT ICT Test RP Test RP<br>Frequency Frequency<br>CG CG<br>**----- End of picture text -----**<br>


Figure 3: The distribution of ICT and DCT for the examples across different models. 

Figure 4: Performance on deductive tasks under different scenarios of observed and test facts. 

_x_ = _y_ , indicating that for the vast majority of cases, DCT is smaller than ICT. Therefore, the fact numbers _N_ does not affect the conclusion we stated earlier. LLM requires fewer facts to successfully perform a deductive task compared to correct induction. 

## 3.4 TRANSFERABILITY TEST OF INDUCTIVE RULES 

Finally, we investigate the impact of different scenarios on the inductive reasoning process. For rule-based reasoning, once a rule is formed through induction, it should be transferable. That is, a rule induced in one scenario should be applicable to another scenario with the same underlying transformation. We experiment to explore whether LLMs possess this ability when performing inductive reasoning. 

**Setup** Specifically, we exclude ST in this experiment since its basic transformations differ from the other three scenarios (see _§_ 2.1). For the remaining three scenarios, we generate the observed facts in one scenario, and then transform the test case into another scenario. Since our dataset can generate questions in different scenarios based on the same meta-rule, we can easily ensure that they share the same underlying transformation. 

**Results** From the results shown in Figure 4, we can get that: **(1) LLMs lack transferability in inductive reasoning.** Across different cases, the highest performance occurs when the scenarios of the observed and test facts are consistent (i.e., the diagonal from the top left to the bottom right in the figure). **(2) The inductive reasoning process of the LLM is form-related.** Compared to the transfer between LT and RP (or CG and RP), the transfer between LT and CG demonstrates better performance. We infer that this is because the forms of LT and CG are more similar (see Figure 2). Based on the above two observations, we further confirm that LLMs do not rely on abstract rules when performing inductive reasoning. So, what is the underlying mechanism behind it? In the following section, we focus on addressing this question. 

## 4 LANGUAGE MODELS ARE GOOD NEIGHBOR-BASED REASONERS 

## 4.1 MOTIVATION 

From _§_ 3.4, we know that closer forms between the observed facts and the test case can enable the model to perform inductive reasoning more effectively. However, is the positive impact brought by the similarity limited only to the form? The answer to this question is likely “No”. Upon reviewing related works, we find that models tend to match various similar patterns in the context and use them to predict the next token (Olsson et al., 2022; Wang et al., 2023a; Hu et al., 2024b). Therefore, we aim to identify a metric to measure some other similarities between the observed facts and the test input. Since all of our facts are transformed from vectors, we associate this similarity with the distance between these facts in feature space. 

In topology, if _f_ : _X → Y_ is a continuous function between two Euclidean spaces and _N_ ( _**x**_ 0 _, ϵ_ ) is a _ϵ_ - neighborhood of the point _**x**_ 0 in _X_ , then we have: 

_∃η >_ 0 _,_ s.t. _∀_ _**x** ∈N_ ( _**x**_ 0 _, ϵ_ ) _, f_ ( _**x**_ ) _∈N_ ( _f_ ( _**x**_ 0) _, η_ ) (5) 

In other words, continuous functions preserve the neighborhood property. If a fact input vector _**x**_ closes to the test input _**x** t_ , then their output vectors _**y**_ and _**y** t_ will remain close.[5] Therefore, **the close distance between** _**y** t_ **and** _**y**_ **may allow LLM to predict** _yt_ **based on** _y_ **in observed facts without the need for correct rule generation** . In the following sections, we demonstrate through experiments that the model’s inductive reasoning relies on this paradigm, which we refer to as **neighbor-based reasoning** . 

> 5We rigorously prove in Appendix C.1 that the rules _f_ in our dataset are all continuous functions. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0008-00.png)


**----- Start of picture text -----**<br>
IF IF 0.7 IF<br>0.8 CF 0.8 CF 0.6 CF<br>OF OF OF<br>0.6 0.5<br>0.6<br>0.4<br>0.4 0.4 0.3<br>0.2<br>0.2 0.2<br>0.1<br>0.0 0.0 0.0<br>1 2 3 4 1 2 3 4 1 2 3 4<br>(a) GPT-4o (b) Claude-3.5 (c) Llama3-8B<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Performance of different fact types on our dataset ( _D_ = 3, _N_ = 5). The dashed line represents the baseline accuracy. 

## 4.2 NEIGHBOR FACTS IN INDUCTIVE REASONING 

Before conducting the experiments, we first define some key concepts in our work: the distance _d_ and neighborhood _N_ . In our setup, the components at corresponding positions in the vectors follow the same transformation rules, while non-corresponding components may undergo different transformations (see Equation 2). Hence, we consider using the distance based on the corresponding components: Chebyshev distance.[6] Given observed fact X _i_ = ( _**x** i,_ _**y** i_ ) and test input _**x** t_ , we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0008-04.png)


where _xik_ and _xtk_ are the _k_ -th component of two input vectors. Then we can define the _ϵ_ -neighborhood of _**x** t_ based on the distance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0008-06.png)


**Setup** According above definitions, we can divide an observed fact X _k_ into three categories based on the distance between _**x**_ and the test input _**x** t_ : **(1) In-neighborhood Fact (IF):** If X _k ∈N_ ( _**x** t, ϵ_ ), we call X _k_ is a inneighborhood fact. **(2) Cross-neighborhood Fact (CF):** If X _k ∈N/_ ( _**x** t, ϵ_ ), but _∃i ∈_ [1 _, D_ ] _,_ s.t. _|xi −xti| ≤ ϵ_ , we consider it a suboptimal neighbor fact because some of its components can still contribute to the model’s inductive reasoning process. In this case, we call X _k_ is a cross-neighborhood fact. **(3) Out-neighborhood Fact (OF):** If _∀i ∈_ [1 _, D_ ] _, |xi − xti| > ϵ_ , we call X _k_ is an out-neighborhood fact. By substitution, we can make the fact set X contain only one type of the fact, while keeping the size _N_ fixed. After constructing different fact sets, we compare the model’s performance on deductive tasks under these settings. Besides, we use the performance under the default fact set as the baseline, where all facts are randomly sampled without any constraints. 

**Results** We report the experimental results in Figure 5. It demonstrates that: **(1) LLM’s inductive reasoning is neighbor-based.** By comparing the three settings, we find that observed facts closer to the test case result in better performance (IF _>_ CF _>_ OF) across all of the models. Besides, compared to the baseline (the dashed line in figures), the accuracy significantly drops after we remove all the neighbor cases in X (i.e. OF). These phenomena indicate that the model heavily relies on neighbor facts during reasoning. **(2) LLMs have a strong ability to capture neighboring patterns.** When we set the neighborhood radius _ϵ_ to 4, both IF and CF still contribute to high reasoning performances for the model. Besides, OF continues to show a significant decline (compared to _ϵ_ = 3). These observations indicate that LLMs can still learn similar patterns even when the observed facts are relatively distant. 

## 4.3 UNIVERSALITY OF NEIGHBOR-BASED REASONING 

We consider whether LLM’s inductive reasoning universally relies on neighbor cases, hence, we set _ϵ_ to 1 and repeat the experiment under more different settings, where the baseline is the same as we set in _§_ 4.2. The results on GPT-4o are reported in Table 4, from which we can prove that the **neighbor-based paradigm is universal in LLMs’ inductive reasoning process** . Across different scenarios and fact numbers, IF consistently gets the highest accuracy, while OF gets the lowest accuracy. The reliance of LLMs’ inductive reasoning on neighbor facts is independent of the specific task scenarios, models, or fact numbers. For more results on other models, you can see Appendix C.3. 

- 4.4 EFFECTIVE SCOPE OF NEIGHBOR-BASED REASONING 

> 6We demonstrate through experiments that this distance is more suitable for constructing neighborhoods compared to other distances. For details, see Appendix C.2. 

8 

|**Type**|**N=3**<br>LT<br>RP<br>CG<br>ST|**N=5**<br>LT<br>RP<br>CG<br>ST|**N=8**<br>LT<br>RP<br>CG<br>ST|
|---|---|---|---|
|Baseline<br>IF Only<br>CF Only<br>OF Only|0.52<br>0.19<br>0.78<br>0.42<br>**0.78**<br>**0.46**<br>**0.82**<br>**0.59**<br>0.48<br>0.25<br>0.59<br>0.43<br>0.46<br>0.18<br>0.50<br>0.38|0.66<br>0.36<br>0.71<br>0.46<br>**0.84**<br>**0.52**<br>**0.84**<br>**0.63**<br>0.69<br>0.35<br>0.72<br>0.50<br>0.49<br>0.23<br>0.57<br>0.36|0.76<br>0.34<br>0.80<br>0.54<br>**0.86**<br>**0.54**<br>**0.91**<br>**0.70**<br>0.75<br>0.34<br>0.82<br>0.53<br>0.61<br>0.23<br>0.67<br>0.38|



Table 4: Performance of different fact types under various settings ( _D_ = 5). The best results in each column are highlighted in **bold** , while the worst results are underlined. 

We have proved that neighbor examples in obTest Space [1,0,4] Test Space [1,0,4] Test Space [1,0,4] served facts can significantly affect the model’s performance on the test case.the effective scope of it? Is it only effective onHowever, what is [4,3,8]Neighborhoodη [4,3,8]Neighborhoodη [4,3,8]Neighborhoodη a single test example or does it generate an im[3,4,7] [3,4,7] [3,4,7] plicit rule that affects more examples? To an[3,3,6] [3,3,6] [3,3,6] swer the question, we first make three assump[9,9,1] [9,9,1] [9,9,1] tions about its possible scope and show them in **Individual Localized Global** Figure 6. For individual scope, the model can only answer the test case _**x** t_ (e.g. [3,4,7] in the Figure 6: Examples for three different effective scopes. figure), for all other cases, the accuracy of the prediction is very low. For localized scope, the model can also successfully answer cases close to _**x** t_ (i.e. the neighbor facts of _**x** t_ ). For global scope, the model can answer all cases with high accuracy. 

**Setup** In our experiment, we sample _n_ test cases X _t_ (here _n_ = 5) in each deductive task _τ_ and define the accuracy _aτ_ for this particular task as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0009-04.png)


Here X _τ_ is the observed fact set of the task. Let _T_ denote the set of all deductive tasks (we set _|T |_ = 100), we define **deductive density** _Id_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0009-06.png)


where _**x** t_ is the origin test input in task _τ_ . We use this metric to indicate the impact of a successful deduction (i.e. [3, 4, 7] in Figure 6) on reasoning over other examples in the test region _N_ ( _**x** t, η_ ). A high _Id_ indicates that the model performs well in most cases within this region, while a low _Id_ suggests that the model’s reasoning is more localized or even individual. For comparison, we set the test radius _η_ to 1, 2, 3, and infinity (i.e. the full test space), and calculate the corresponding _Id_ for the model. Besides, we also vary the neighborhood radius _ϵ_ to examine the impact of different distributions of neighbor facts on their effective scope (here we set the test region to the full space). We repeat the experiment five times to eliminate the interference of random errors, and the results are illustrated in Figure 7. 

**Results** We can draw conclusions as follows: **(1) LLM conducts localized reasoning through the neighbor-based paradigm.** From Figure 7a, 7c, we observe that the _Id_ of IF and CF decreases continuously as the radius of the test domain expands. These neighbor cases are highly effective within the neighborhood of _**x** t_ . For example, in Figure 7a, when _η_ = 1, the model can achieve over 0.9 _Id_ . However, this impact diminishes for test cases that are farther from _**x** t_ . As an example, in Figure 7c, the model only gets around 0.5 _Id_ in full test space. **(2) The effective scope of neighbor facts is proportional to their distance from the test case.** According to Figure 7b,7d, the _Id_ of IF and CF (particularly IF) increases as _ϵ_ becomes large. When the neighborhood radius increases, the distribution of these facts becomes more dispersed. We can infer that a more dispersed distribution of neighbor facts tends to make the effective scope more global. 

## 5 LIMITATIONS AND DISCUSSIONS 

**Interpretation Methods.** Most model interpretation studies delve into the internal of models (e.g. neurons, attention layers), providing a comprehensive explanation of the working mechanisms (Romera-Paredes 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0010-00.png)


**----- Start of picture text -----**<br>
1.0 0.9 0.9 0.9<br>BL BL BL BL<br>0.9 IFCF 0.8 IF CF 0.8 IF CF 0.8 IF CF<br>OF OF OF OF<br>0.8<br>0.7 0.7 0.7<br>0.7<br>0.6 0.6 0.6<br>0.6<br>0.5 1 2 3 inf 0.5 1 2 3 0.5 1 2 3 inf 0.5 1 2 3<br>(a)  D  = 3,  ϵ  = 1 (b)  D  = 3,  η  = inf (c)  D  = 5,  ϵ  = 1 (d)  D  = 5,  η  = inf<br>Deductive Density Deductive Density Deductive Density Deductive Density<br>**----- End of picture text -----**<br>


Figure 7: Deductive Density ( _Id_ ) of various fact types on GPT-4o under different test radius _η_ and neighborhood radius _ϵ_ ( _N_ = 5). BL represents the performance of the baseline, where we use default fact sets with no substitution. 

et al., 2024; Li et al., 2024). However, our work does not conduct internal analysis but instead relies on performance comparisons under different settings. There are two main considerations for this: On one hand, this work aims to identify mechanisms that are applicable to black-box models. Since we do not have access to the internal parameters of these models, we are unable to use previous methods. On the other hand, whitebox models exhibit poor inductive reasoning capabilities according to Table 1. Therefore, conducting in-depth interpretations based on white-box models may introduce noise to the conclusions. 

**Experimental Settings.** The goal of this paper is to evaluate and explain the inductive reasoning process in LLMs, rather than to improve the task performance. Therefore, we do not meticulously design the prompts used in the experiments, nor do we use the best-performing inductive reasoning methods throughout the analysis. We believe that the experimental setup of 0-shot IO with simple instructions is more aligned with real-world application scenarios, making our evaluation and explanation results more meaningful. 

**Future Directions.** Our study demonstrates that LLMs perform poorly in rule-based reasoning but excel at using neighbor facts for reasoning. Future work could explore methods to encourage the model to follow rules more closely during reasoning or to further optimize the model’s inductive reasoning abilities based on this neighbor-matching finding. 

## 6 RELATED WORK 

**Evaluating Inductive Reasoning Abilities of LLMs.** Existing studies on evaluating LLM’s inductive reasoning capabilities mainly use only a single task. On one hand, some works assess the model’s rule induction ability by evaluating the accuracy on unseen examples (Moskvichev et al., 2023; Tang et al., 2023; Gendron et al., 2023; Xu et al., 2024b; Qiu et al., 2024). However, since the model’s deduction does not always rely on inducing the correct rule, this indirect evaluation method can introduce some inaccuracies. On the other hand, some studies directly evaluate the correctness of the generated rules to assess inductive reasoning ability (Shao et al., 2024; Cheng et al., 2024; Yang et al., 2024b). These studies lack evaluation on test examples, making it difficult to confirm the model’s mastery of the inductive rules. Our work evaluates both aspects, providing a comprehensive analysis of the model’s inductive reasoning process. 

**Mechanism Analysis on LLM’s Reasoning.** A growing body of interpretability research has begun analyzing the reasoning mechanisms of LLMs, aiming to deepen our understanding of how these models function. Some studies explore the mechanisms behind mathematical reasoning (Zhang et al., 2024; Hu et al., 2024b; Romera-Paredes et al., 2024; Stolfo et al., 2023), some works investigate multi-hop reasoning (Wang et al., 2024a; Hou et al., 2023; Yang et al., 2024a; Biran et al., 2024), and some focus on other types of reasoning (Li et al., 2024; Hu et al., 2024a). However, there is currently a lack of analysis on the mechanisms of inductive reasoning. Our work mitigates this gap and uncovers the neighbor-based paradigms LLMs follow when performing inductive reasoning. 

## 7 CONCLUSION 

In this paper, we focus on evaluating and explaining the inductive reasoning process of LLMs. First, we construct a dataset MIRAGE, which provides both inductive and deductive evaluation tasks, with the flexibility to generate test examples in any distribution, different difficulties, and various forms. Based on it, we prove that LLM is a poor rule-based reasoner, it does not need to rely on inductive rules when performing deductive tasks. Compared to correct induction, the model can perform successful deduction with fewer observations, and this deduction is closely related to the form of the input. Furthermore, we identify a key paradigm of LLM 

10 

inductive reasoning: neighbor-based reasoning. The model tends to leverage observed facts that are close to the test examples in feature space for inductive reasoning. Through it, the model can achieve strong inductive reasoning capabilities within a localized scope and apply this ability to make inferences on unseen examples. 

## REFERENCES 

- Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, and Amir Globerson. Hopping too late: Exploring the limitations of large language models on multi-hop queries. CoRR, abs/2406.12775, 2024. doi: 10.48550/ ARXIV.2406.12775. URL https://doi.org/10.48550/arXiv.2406.12775. 

- Kewei Cheng, Jingfeng Yang, Haoming Jiang, Zhengyang Wang, Binxuan Huang, Ruirui Li, Shiyang Li, Zheng Li, Yifan Gao, Xian Li, Bing Yin, and Yizhou Sun. Inductive or deductive? rethinking the fundamental reasoning abilities of llms. CoRR, abs/2408.00114, 2024. doi: 10.48550/ARXIV.2408.00114. URL https://doi.org/10.48550/arXiv.2408.00114. 

- Franc¸ois Chollet. On the measure of intelligence. CoRR, abs/1911.01547, 2019. URL http://arxiv. org/abs/1911.01547. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aur´elien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozi`ere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego GarciaOlano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gr´egoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL https://doi.org/10.48550/arXiv.2407.21783. 

- Ga¨el Gendron, Qiming Bao, Michael Witbrock, and Gillian Dobbie. Large language models are not abstract reasoners. CoRR, abs/2305.19555, 2023. doi: 10.48550/ARXIV.2305.19555. URL https://doi.org/ 10.48550/arXiv.2305.19555. 

- Simon Jerome Han, Keith J Ransom, Andrew Perfors, and Charles Kemp. Inductive reasoning in humans and large language models. Cognitive Systems Research, 83:101155, 2024. 

- Yifan Hou, Jiaoda Li, Yu Fei, Alessandro Stolfo, Wangchunshu Zhou, Guangtao Zeng, Antoine Bosselut, and Mrinmaya Sachan. Towards a mechanistic interpretation of multi-step reasoning capabilities of language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 4902–4919. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.299. URL https://doi.org/10.18653/v1/2023.emnlp-main.299. 

- Yebowen Hu, Kaiqiang Song, Sangwoo Cho, Xiaoyang Wang, Wenlin Yao, Hassan Foroosh, Dong Yu, and Fei Liu. When reasoning meets information aggregation: A case study with sports narratives. CoRR, abs/2406.12084, 2024a. doi: 10.48550/ARXIV.2406.12084. URL https://doi.org/10.48550/ arXiv.2406.12084. 

- Yi Hu, Xiaojuan Tang, Haotong Yang, and Muhan Zhang. Case-based or rule-based: How do transformers do the math? In Forty-frst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024b. URL https://openreview.net/forum?id=4Vqr8SRfyX. 

- Franz Huber. On the justification of deduction and induction. European Journal for Philosophy of Science, 7: 507–534, 2017. 

- Yifan Jiang, Jiarui Zhang, Kexuan Sun, Zhivar Sourati, Kian Ahrabian, Kaixin Ma, Filip Ilievski, and Jay Pujara. MARVEL: multidimensional abstraction and reasoning through visual evaluation and learning. CoRR, abs/2404.13591, 2024. doi: 10.48550/ARXIV.2404.13591. URL https://doi.org/10.48550/ arXiv.2404.13591. 

- Subin Kim, Prin Phunyaphibarn, Donghyun Ahn, and Sundong Kim. Playgrounds for abstraction and reasoning. In NeurIPS 2022 Workshop on Neuro Causal and Symbolic AI (nCSI), 2022. 

- Brenden M. Lake, Tal Linzen, and Marco Baroni. Human few-shot learning of compositional instructions. In Ashok K. Goel, Colleen M. Seifert, and Christian Freksa (eds.), Proceedings of the 41th Annual Meeting of 

11 

the Cognitive Science Society, CogSci 2019: Creativity + Cognition + Computation, Montreal, Canada, July 24-27, 2019, pp. 611–617. cognitivesciencesociety.org, 2019. URL https://mindmodeling.org/ cogsci2019/papers/0123/index.html. 

- Jiachun Li, Pengfei Cao, Chenhao Wang, Zhuoran Jin, Yubo Chen, Daojian Zeng, Kang Liu, and Jun Zhao. Focus on your question! interpreting and mitigating toxic cot problems in commonsense reasoning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 9206–9230. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024. ACL-LONG.499. URL https://doi.org/10.18653/v1/2024.acl-long.499. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. CoRR, abs/2303.17651, 2023. doi: 10.48550/ARXIV.2303.17651. URL https://doi.org/10.48550/ arXiv.2303.17651. 

- Arsenii Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptarc benchmark: Evaluating understanding and generalization in the ARC domain. Trans. Mach. Learn. Res., 2023, 2023. URL https: //openreview.net/forum?id=8ykyGbtt2q. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. CoRR, abs/2209.11895, 2022. doi: 10.48550/ARXIV.2209.11895. URL https: //doi.org/10.48550/arXiv.2209.11895. 

- OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https://doi.org/10.48550/arXiv.2303.08774. 

- Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, and Xiang Ren. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=bNt7oajl2a. 

- Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nat., 625(7995):468–475, 2024. doi: 10.1038/S41586-023-06924-6. URL https://doi.org/10. 1038/s41586-023-06924-6. 

- Joshua Stewart Rule. The child as hacker: building more human-like models of learning. PhD thesis, Massachusetts Institute of Technology, 2020. 

- Yunfan Shao, Linyang Li, Yichuan Ma, Peiji Li, Demin Song, Qinyuan Cheng, Shimin Li, Xiaonan Li, Pengyu Wang, Qipeng Guo, Hang Yan, Xipeng Qiu, Xuanjing Huang, and Dahua Lin. Case2code: Learning inductive reasoning with synthetic data. CoRR, abs/2407.12504, 2024. doi: 10.48550/ARXIV.2407.12504. URL https://doi.org/10.48550/arXiv.2407.12504. 

- Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 7035–7052. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.435. URL https://doi.org/10.18653/v1/ 2023.emnlp-main.435. 

- Wangtao Sun, Haotian Xu, Xuanqing Yu, Pei Chen, Shizhu He, Jun Zhao, and Kang Liu. Itd: Large language models can teach themselves induction through deduction. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 2719–2731. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.150. URL https://doi. org/10.18653/v1/2024.acl-long.150. 

- Xiaojuan Tang, Zilong Zheng, Jiaqi Li, Fanxu Meng, Song-Chun Zhu, Yitao Liang, and Muhan Zhang. Large language models are in-context semantic reasoners rather than symbolic reasoners. CoRR, abs/2305.14825, 2023. doi: 10.48550/ARXIV.2305.14825. URL https://doi.org/10.48550/arXiv.2305. 14825. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, 

12 

Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/ arXiv.2307.09288. 

- Boshi Wang, Xiang Yue, Yu Su, and Huan Sun. Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization. CoRR, abs/2405.15071, 2024a. doi: 10.48550/ARXIV.2405.15071. URL https://doi.org/10.48550/arXiv.2405.15071. 

- Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 9840–9855. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.EMNLP-MAIN.609. URL https://doi. org/10.18653/v1/2023.emnlp-main.609. 

- Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D. Goodman. Hypothesis search: Inductive reasoning with language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024b. URL https: //openreview.net/forum?id=G7UtIGQmjm. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023b. URL https://openreview.net/pdf?id=1PL1NIMMrw. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html. 

- Yudong Xu, Wenhao Li, Pashootan Vaezipoor, Scott Sanner, and Elias Boutros Khalil. Llms and the abstraction and reasoning corpus: Successes, failures, and the importance of object-based representations. Trans. Mach. Learn. Res., 2024, 2024a. URL https://openreview.net/forum?id=E8m8oySvPJ. 

- Yudong Xu, Wenhao Li, Pashootan Vaezipoor, Scott Sanner, and Elias Boutros Khalil. Llms and the abstraction and reasoning corpus: Successes, failures, and the importance of object-based representations. Trans. Mach. Learn. Res., 2024, 2024b. URL https://openreview.net/forum?id=E8m8oySvPJ. 

- Sohee Yang, Elena Gribovskaya, Nora Kassner, Mor Geva, and Sebastian Riedel. Do large language models latently perform multi-hop reasoning? In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 10210–10229. Association for Computational Linguistics, 2024a. doi: 10.18653/V1/2024.ACL-LONG.550. URL https://doi.org/10. 18653/v1/2024.acl-long.550. 

- Zonglin Yang, Li Dong, Xinya Du, Hao Cheng, Erik Cambria, Xiaodong Liu, Jianfeng Gao, and Furu Wei. Language models as inductive reasoners. In Yvette Graham and Matthew Purver (eds.), Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2024 - Volume 1: Long Papers, St. Julian’s, Malta, March 17-22, 2024, pp. 209–225. Association for Computational Linguistics, 2024b. URL https://aclanthology.org/2024.eacl-long.13. 

- Wei Zhang, Chaoqun Wan, Yonggang Zhang, Yiu-ming Cheung, Xinmei Tian, Xu Shen, and Jieping Ye. Interpreting and improving large language models in arithmetic calculation. In Forty-frst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=CfOtiepP8s. 

13 

## A MORE DETAILS FOR DATASET CONSTRUCTION 

## A.1 COMPARISON OF DATASETS BETWEEN RELATED WORK AND OUR STUDY 

Since our work is conducted entirely on our MIRAGE dataset, we aim to provide a detailed comparison with representative datasets from related studies to demonstrate its effectiveness. Specifically, we report the comparison in Table 5. From the results, we can see that our dataset can cover most of the operations and forms in previous datasets. For example, the transformation examples in the 1D-ARC dataset shown in the table are equivalent to our PAD operation. Therefore, we demonstrate that our dataset is an effective dataset for inductive reasoning. Moreover, based on its coverage, conducting experiments solely on it is sufficient. 

|**Dataset**|**Fact Form**|**Main Operation**|**Example**|
|---|---|---|---|
||||input: [[0,1,0],[1,1,0],|
||||[0,1,0],[0,1,1],[0,1,0],|
||||[1,1,0]]|
|ARC (Chollet, 2019)|List (2D)|Fill, Move, Pile|output: [[0,2,0],[2,2,0],|
||||[0,2,0],[0,2,2],[0,2,0],|
||||[2,2,0],[0,2,0],[0,2,2],|
||||[0,2,0]|
||||input: her sneury voirk|
|MiniSCAN (Lake et al., 2019)|String|String Translation||
||||output: GREEN BLUE|
||||input: [4,7,6,9,0]|
|ListFunctions (Rule, 2020)|List|List Operation||
||||output:[4,8,6,9,0]|
||||input: [[1,1,5,6,8],|
||||[0,1,5,6,6],[5,5,5,5,5],|
||||[7,7,5,4,4],[7,7,5,0,4]]|
|MiniARC (Kim et al., 2022)|List (2D)|Fill, Move, Pile||
||||output: [[1,6,0,0,0],|
||||[7,4,0,0,0],[0,0,0,0,0],|
||||[0,0,0,0,0], [0,0,0,0,0]]|
||||input: [0,0,2,0,0,0,0,2,|
||||0,0,0]|
|1D-ARC (Xu et al., 2024a)|List|Fill, Move, Pile||
||||output: [0,0,2,2,2,2,2,2,|
||||0,0,0]|
||||input: dict(no=2)|
|Case2Code (Shao et al., 2024)|Code|Python Function||
||||output: [2]|
|MIRAGE|All above + RP|All above|See Figure 2|



Table 5: Comparison between some representative datasets and ours. 

## A.2 EVALUATION OF MATHEMATICAL OPERATION DIFFICULTY IN MIRAGE 

Our data introduces some mathematical operations in Add and Map, here we aim to demonstrate that these calculations are inherently simple for LLMs, ensuring that they do not interfere with our evaluation of the model’s inductive reasoning performance. Specifically, we randomly construct linear operations in Add and Map with single-digit operands (cover all of the operations included in this paper) and observe the accuracy of each model on 100 questions. The results are reported in Table 6. We can observe that most of the modes can achieve very high accuracy on these math operations. Specifically, all closed-source models achieve 100% accuracy. This indicates that our dataset construction effectively eliminates noise introduced by mathematical calculations in most cases. 

14 

|**Operation**|Llama2-13B|Llama3-8B|GPT-4o|GPT-4|Claude-3.5|
|---|---|---|---|---|---|
|Map|0.96|0.93|1.00|1.00|1.00|
|Add|0.51|0.99|1.00|1.00|1.00|



Table 6: Accuracy of basic mathematical computations across different models in our dataset. 

## A.3 EVALUATION TEMPLATES FOR DIFFERENT TASKS AND SCENARIOS 

In Tables 13 and 14, we report the evaluation prompts used to evaluate the models in our work. In Table 15, we provide the templates used for constructing different scenarios in RP. 

## B MORE DETAILS FOR RULE-BASED REASONING EVALUATION 

## B.1 IMPLEMENT DETAILS FOR MAIN EXPERIMENTS 

For model version, we select Llama-2-13b-chat-hf, Meta-Llama-3-8B-Instruct, gpt-4-0613, gpt-4o-2024-05-13 and claude-3-5-sonnet-20240620. All experiments are conducted on 4 NVIDIA GeForce RTX 3090 GPUs. For the sake of simplicity, we include all the prompts used in this work in the supplementary materials. 

## B.2 MORE EXPERIMENTS ON OTHER MODELS 

In this section, we repeat the experiments in _§_ 3.2 on the Llama3-8B model and Llama2-13B model, the results are shown in Table 7, 8. Here we set _D_ = 3, _N_ = 5. We can observe that the results in our main text still hold on these models. We do not apply HR on them, since these two models have difficulty in evaluating the rule based on given templates under 0-shot settings. 

|**Method**|**LT**<br>Ind<br>Ded<br>(∆)|**RP**<br>Ind<br>Ded<br>(∆)|**CG**<br>Ind<br>Ded<br>(∆)|**ST**<br>Ind<br>Ded<br>(∆)|
|---|---|---|---|---|
|IO(0-shot)<br>IO(5-shot)|0.21<br>0.54<br>0.33<br>0.32<br>0.41<br>0.09|0.03<br>0.48<br>0.45<br>0.35<br>0.40<br>0.05|0.17<br>0.22<br>0.05<br>0.14<br>0.24<br>0.10|0.07<br>0.45<br>0.38<br>0.25<br>0.35<br>0.10|
|ID(0-shot)<br>ID(5-shot)|0.10<br>0.24<br>0.14<br>0.35<br>0.44<br>0.09|0.12<br>0.30<br>0.18<br>0.30<br>0.40<br>0.10|0.10<br>0.17<br>0.07<br>0.01<br>0.04<br>0.03|0.01<br>0.35<br>0.34<br>0.31<br>0.33<br>0.02|
|CoT(0-shot)<br>CoT(5-shot)|0.25<br>0.39<br>0.14<br>0.54<br>0.59<br>0.05|0.13<br>0.39<br>0.26<br>0.36<br>0.40<br>0.04|0.16<br>0.14<br>-0.02<br>0.41<br>0.55<br>0.14|0.16<br>0.42<br>0.26<br>0.47<br>0.66<br>0.19|
|SC(n=5)|0.47<br>0.54<br>0.07|0.35<br>0.37<br>0.02|0.45<br>0.55<br>0.10|0.51<br>0.62<br>0.11|
|SR(t=3)|0.16<br>0.26<br>0.10|0.09<br>0.27<br>0.18|0.08<br>0.20<br>0.12|0.05<br>0.39<br>0.34|



Table 7: Performance of different methods on MIRAGE using Llama3-8B (100 examples). 

|**Method**|**LT**<br>Ind<br>Ded<br>(∆)|**RP**<br>Ind<br>Ded<br>(∆)|**CG**<br>Ind<br>Ded<br>(∆)|**ST**<br>Ind<br>Ded<br>(∆)|
|---|---|---|---|---|
|IO(0-shot)<br>IO(5-shot)|0.01<br>0.29<br>0.28<br>0.02<br>0.37<br>0.35|0.01<br>0.41<br>0.40<br>0.01<br>0.34<br>0.33|0.01<br>0.15<br>0.15<br>0.01<br>0.14<br>0.14|0.18<br>0.40<br>0.22<br>0.05<br>0.30<br>0.25|
|ID(0-shot)<br>ID(5-shot)|0.00<br>0.02<br>0.02<br>0.02<br>0.12<br>0.10|0.00<br>0.03<br>0.03<br>0.01<br>0.14<br>0.14|0.00<br>0.06<br>0.06<br>0.00<br>0.01<br>0.01|0.17<br>0.20<br>0.02<br>0.01<br>0.30<br>0.29|
|CoT(0-shot)<br>CoT(5-shot)|0.03<br>0.10<br>0.08<br>0.01<br>0.24<br>0.23|0.03<br>0.20<br>0.17<br>0.02<br>0.14<br>0.12|0.00<br>0.13<br>0.13<br>0.00<br>0.14<br>0.14|0.06<br>0.13<br>0.08<br>0.07<br>0.10<br>0.03|
|SC(n=5)|0.07<br>0.24<br>0.17|0.08<br>0.49<br>0.41|0.01<br>0.34<br>0.33|0.17<br>0.37<br>0.20|
|SR(t=3)|0.00<br>0.03<br>0.03|0.01<br>0.07<br>0.06|0.00<br>0.06<br>0.06|0.01<br>0.10<br>0.09|



Table 8: Performance of different methods on MIRAGE using Llama2-13B (200 examples). 

15 

## C MORE DETAILS FOR NEIGHBOR-BASED REASONING EVALUATION 

## C.1 PROOF OF CONTINUES FUNCTIONS 

Here, we prove that the five basic vector operations in MIRAGE are all continuous functions: **Theorem 1** (Add Operation Continuity) **.** _Let_ **A** = ( _a_ 1 _, a_ 2 _, . . . , an_ ) _∈_ R _[n] . Define a mapping f_ : R _[n] →_ R _[n] such that for a fixed index k ∈{_ 1 _,_ 2 _, . . . , n} and a fixed subset I ⊆{_ 1 _,_ 2 _, . . . , n}, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0016-03.png)


_where k ∈/ I. Then f is a continuous function._ 

_Proof._ Consider two vectors **A** _,_ **B** _∈_ R _[n]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0016-06.png)


The mapping _f_ replaces the _k_ -th element of the vector with the sum of elements indexed by the subset _I_ . Thus, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0016-08.png)


The distance between the images of **A** and **B** under _f_ is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0016-10.png)


Let us focus on the term involving the sums: 

By the triangle inequality, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0016-13.png)


Therefore, 

Using the Cauchy-Schwarz inequality, we get 

where _|I|_ is the cardinality of the set _I_ . Therefore, 

This can be bounded as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0016-18.png)


where _C_ is a constant depending on _n_ and _|I|_ . Therefore, for any _ϵ >_ 0, choose _δ_ = _Cϵ_[.][If] _[ ∥]_ **[A]** _[ −]_ **[B]** _[∥][< δ]_[, then] _∥f_ ( **A** ) _− f_ ( **B** ) _∥ < Cδ_ = _ϵ._ 

Hence, _f_ is continuous. 

16 

**Theorem 2** (Copy Operation Continuity) **.** _Let_ **A** = ( _a_ 1 _, a_ 2 _, . . . , an_ ) _∈_ R _[n] . Define a mapping f_ : R _[n] →_ R _[n] such that for fixed indices J ⊆{_ 1 _,_ 2 _, . . . , n} and a fixed index k ∈{_ 1 _,_ 2 _, . . . , n}, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-01.png)


_where_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-03.png)


_Then f is a continuous function._ 

_Proof._ Consider two vectors **A** _,_ **B** _∈_ R _[n]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-06.png)


The mapping _f_ replaces each element of **A** at the positions indexed by _J_ with the value of the element at index _k_ . Specifically, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-08.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-10.png)


Similarly, 

where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-13.png)


The distance between the images of **A** and **B** under _f_ is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-15.png)


By the definition of _f_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-17.png)


Therefore, the distance can be rewritten as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-19.png)


Since the sum over _J_ has _|J|_ terms that are all equal to ( _ak − bk_ )[2] , this simplifies to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-21.png)


This can be bounded as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0017-24.png)


Hence, _f_ is continuous. 

17 

**Theorem 3** (Map Operation Continuity) **.** _Let_ **A** = ( _a_ 1 _, a_ 2 _, . . . , an_ ) _∈_ R _[n] . Define a mapping f_ : R _[n] →_ R _[n] such that for fixed indices J ⊆{_ 1 _,_ 2 _, . . . , n} and fixed scalars k, b ∈_ R _, we have_ 

_where_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-02.png)


_Then f is a continuous function._ 

_Proof._ Consider two vectors **A** _,_ **B** _∈_ R _[n]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-05.png)


The mapping _f_ applies the linear transformation _kx_ + _b_ to the elements of **A** indexed by _J_ and leaves the other elements unchanged: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-07.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-09.png)


Similarly, 

where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-12.png)


The distance between the images of **A** and **B** under _f_ is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-14.png)


By the definition of _f_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-16.png)


Therefore, the distance can be rewritten as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0018-18.png)


This simplifies to 

Let _C_ = max(1 _, |k|_ ). Then 

Therefore, for any _ϵ >_ 0, choose _δ_ = _Cϵ_[.][If] _[ ∥]_ **[A]** _[ −]_ **[B]** _[∥][< δ]_[, then] _∥f_ ( **A** ) _− f_ ( **B** ) _∥ < Cδ_ = _ϵ._ 

Hence, _f_ is continuous. 

18 

**Theorem 4** (Pad Operation Continuity) **.** _Let_ **A** = ( _a_ 1 _, a_ 2 _, . . . , an_ ) _∈_ R _[n] . Define a mapping f_ : R _[n] →_ R _[n] such that for a fixed subset J ⊆{_ 1 _,_ 2 _, . . . , n} and a fixed constant C ∈_ R _, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-01.png)


_where_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-03.png)


_Then f is a continuous function._ 

_Proof._ Consider two vectors **A** _,_ **B** _∈_ R _[n]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-06.png)


The mapping _f_ replaces each element of **A** at the positions indexed by _J_ with the constant _C_ , and leaves the other elements unchanged: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-08.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-10.png)


Similarly, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-12.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-14.png)


The distance between the images of **A** and **B** under _f_ is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-16.png)


By the definition of _f_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-18.png)


Therefore, the distance can be rewritten as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-20.png)


Note that the sum is only over the indices not in _J_ . This is because the elements in _J_ are replaced by the constant _C_ , and thus their difference is zero. 

Since 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-23.png)


for any _ϵ >_ 0, choose _δ_ = _ϵ_ . If _∥_ **A** _−_ **B** _∥ < δ_ , then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0019-25.png)


Therefore, _f_ is continuous. 

**Theorem 5** (Swap Operation Continuity) **.** _Let_ **A** = ( _a_ 1 _, a_ 2 _, . . . , an_ ) _∈_ R _[n] . Define a mapping f_ : R _[n] →_ R _[n] such that for fixed disjoint subsets I, J ⊆{_ 1 _,_ 2 _, . . . , n} with I ∩ J_ = _∅ and |I|_ = _|J|, the elements of_ **A** _indexed by I are swapped with the elements indexed by J. Then f is a continuous function._ 

19 

_Proof._ Let **A** = ( _a_ 1 _, a_ 2 _, . . . , an_ ) _∈_ R _[n]_ , and let _I_ = _{i_ 1 _, i_ 2 _, . . . , im}_ and _J_ = _{j_ 1 _, j_ 2 _, . . . , jm}_ be two disjoint subsets of indices with _|I|_ = _|J|_ = _m_ . Define the mapping _f_ such that it swaps the elements of **A** indexed by _I_ and _J_ . Specifically, for any **A** , the mapping _f_ produces a vector **B** = _f_ ( **A** ) given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-01.png)


Consider two vectors **A** _,_ **C** _∈_ R _[n]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-03.png)


Applying the mapping _f_ to both vectors, we obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-05.png)


where 

and similarly, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-08.png)


The distance between _f_ ( **A** ) and _f_ ( **C** ) is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-10.png)


Since _f_ only swaps the elements indexed by _I_ and _J_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-12.png)


Therefore, the norm becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-14.png)


Rearranging the terms, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-16.png)


Therefore, for any _ϵ >_ 0, choose _δ_ = _ϵ_ . If _∥_ **A** _−_ **C** _∥ < δ_ , then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0020-18.png)


Hence, _f_ is continuous. 

## C.2 COMPARISON WITH OTHER DISTANCE METRIC 

We aim to explore whether using different distance metrics to define neighbor facts would also influence the model’s inductive reasoning. Therefore, we additionally introduce three other distance metrics: Euclidean distance _deuc_ , Manhattan distance _dman_ , and Minkowski distance _dmin_ . Like Equation 6, we have: 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0021-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0021-01.png)


where we set _p_ = 3. We can generate three distinct new neighborhoods _N_ ( _**x** t, ϵ_ ) by incorporating these distances into Equation 7, thereby constructing three new kinds of OF. Therefore, we compare the model’s performance on deductive tasks when using only these different OFs, and the results are shown in Figure 8. From the figure, we can see that **our neighborhood construction outperforms those constructed using other distance metrics** . The deductive performance of the other three OFs across different radii is similar to the baseline, indicating that removing neighbor facts constructed using these methods does not influence the model’s inductive reasoning ability. In contrast, our constructed OF leads to a significant decline in accuracy, proving the validity of our neighborhood construction. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0021-03.png)


**----- Start of picture text -----**<br>
0.8<br>Euc 0.8 Euc<br>0.7 Man Man<br>Min 0.7 Min<br>0.6 Our Our<br>0.6<br>0.5<br>0.5<br>0.4<br>0.4<br>0.3 0.3<br>0.2 0.2<br>0.1 0.1<br>0.0 0.0<br>1 2 3 4 1 2 3 4<br>Radius Radius<br>(a) GPT-4o (b) Claude-3.5<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 8: Performance comparison of the impact of different OFs. The dashed line represents the baseline accuracy using default fact sets. Euc represents Euclidean distance, Man represents Manhattan distance and Min represents Minkowski distance. 

## C.3 MORE EXPERIMENTS ON OTHER MODELS 

We repeat the experiments in _§_ 4.3 on Llama2-13B, Claude-3.5 and Llama3-8B. The results are shown in Table 9, 10, 11. Besides, we also repeat the experiments in _§_ 4.4 on Claude-3.5 and report the results in Figure 9. The results of all these additional experiments are consistent with those in the main text. 

## C.4 SUPPLEMENTARY EXPERIMENT FOR MAIN EXPERIMENT 

We observe that, in the experiment of _§_ 4.2, though the performance of OF significantly decreases compared to the baseline, some models are still able to maintain around 40% accuracy, even with only distant observed facts. We infer that models are likely to conduct rule-based reasoning in these cases. Hence, we design an extra experiment for supplementary. In it, we prompt LLMs to induct rules and finish deductive tasks (i.e. ID in _§_ 3.2) on these cases in Table 12. From the table, we can observe that the model’s deductive accuracy using the rule exceeds 70% when there are fewer neighbor facts in the context. This demonstrates that the model tends to rely more on rule-based induction if there is less neighbor-based matching. 

21 

|**Type**|**N=3**<br>LT<br>RP<br>CG<br>ST|**N=5**<br>LT<br>RP<br>CG<br>ST|**N=8**<br>LT<br>RP<br>CG<br>ST|
|---|---|---|---|
|Baseline<br>IF Only<br>CF Only<br>OF Only|0.18<br>0.05<br>0.14<br>0.30<br>0.43<br>0.14<br>0.35<br>0.34<br>0.17<br>0.06<br>0.14<br>0.28<br>0.16<br>0.04<br>0.13<br>0.27|0.15<br>0.05<br>0.17<br>0.23<br>0.48<br>0.03<br>0.49<br>0.36<br>0.15<br>0.04<br>0.22<br>0.25<br>0.09<br>0.02<br>0.09<br>0.26|0.14<br>0.00<br>0.17<br>0.31<br>0.46<br>0.02<br>0.52<br>0.36<br>0.14<br>0.00<br>0.17<br>0.31<br>0.10<br>0.01<br>0.10<br>0.22|



Table 9: Performance of different fact types under various settings on Llama2-13B ( _D_ = 5). 

|**Type**|**N=3**<br>LT<br>RP<br>CG<br>ST|**N=5**<br>LT<br>RP<br>CG<br>ST|**N=8**<br>LT<br>RP<br>CG<br>ST|
|---|---|---|---|
|Baseline<br>IF Only<br>CF Only<br>OF Only|0.49<br>0.23<br>0.49<br>0.38<br>0.76<br>0.42<br>0.84<br>0.61<br>0.54<br>0.23<br>0.59<br>0.36<br>0.50<br>0.24<br>0.49<br>0.45|0.68<br>0.39<br>0.80<br>0.53<br>0.90<br>0.46<br>0.87<br>0.61<br>0.67<br>0.36<br>0.81<br>0.56<br>0.60<br>0.30<br>0.71<br>0.39|0.78<br>0.51<br>0.81<br>0.56<br>0.89<br>0.62<br>0.93<br>0.76<br>0.76<br>0.42<br>0.85<br>0.53<br>0.66<br>0.26<br>0.77<br>0.49|



Table 10: Performance of different fact types under various settings on Claude-3.5 ( _D_ = 5). 

|**Type**|**N=3**<br>LT<br>RP<br>CG<br>ST|**N=5**<br>LT<br>RP<br>CG<br>ST|**N=8**<br>LT<br>RP<br>CG<br>ST|
|---|---|---|---|
|Baseline<br>IF Only<br>CF Only<br>OF Only|0.19<br>0.12<br>0.26<br>0.29<br>0.55<br>0.22<br>0.55<br>0.38<br>0.21<br>0.07<br>0.29<br>0.29<br>0.24<br>0.06<br>0.34<br>0.26|0.27<br>0.17<br>0.28<br>0.32<br>0.65<br>0.40<br>0.65<br>0.46<br>0.27<br>0.14<br>0.31<br>0.31<br>0.23<br>0.08<br>0.24<br>0.25|0.24<br>0.26<br>0.37<br>0.20<br>0.68<br>0.41<br>0.75<br>0.41<br>0.27<br>0.22<br>0.39<br>0.19<br>0.18<br>0.11<br>0.23<br>0.15|



Table 11: Performance of different fact types under various settings on Llama3-8B ( _D_ = 5). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models_images/iclr_2025_MIRAGE_Evaluating_and_Explaining_Inductive_Reasoning_Process_in_Language_Models.pdf-0022-06.png)


**----- Start of picture text -----**<br>
0.9 0.9<br>BL BL<br>IF IF<br>0.8 CF 0.8 CF<br>OF OF<br>0.7 0.7<br>0.6 0.6<br>0.5 0.5<br>1 2 3 inf 1 2 3<br>(a)  ϵ  = 1 (b)  η  = inf<br>Deductive Density Deductive Density<br>**----- End of picture text -----**<br>


Figure 9: Deductive Density of various fact types on Claude-3.5 under different test radius _η_ and neighborhood radius _ϵ_ ( _D_ = 5, _N_ = 5). 

|**Model**|_ϵ_=1|_ϵ_=2|_ϵ_=3|Avg|
|---|---|---|---|---|
|GPT-4o|0.74|0.72|0.76|0.74|
|Claude-3.5|0.76|0.72|0.73|0.74|



Table 12: Performance on the correct case of OF. We use the 0-shot setting and vary the radius _ϵ_ . 

22 

|**Scenario**|**Prompt**||
|---|---|---|
||Please summarize the rules of the list transformation based on the given facts.||
||Your reply should strictly follow the following format:||
||Rule: [A, B, C]_→_[<<expression>>,<<expression>>,<<expression>>]||
|**LT**|Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Please generate the rule of list transformation based on the former facts.||
||Please summarize the rules of the_{_TASK<br>~~T~~YPE_}_based on the given facts.||
||Your reply should strictly follow the following format:||
||Rule: If there are A_{_OBJ1_}_, B_{_OBJ2_}_, C_{_OBJ3_}_. After the_{_TASK|TYPE_}_,|
|**RP**|there are<<expression>>_{_OBJ1_}_,<<expression>>_{_OBJ2_}_,<<expression>>_{_OBJ3_}_.||
||Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Please generate the rule of_{_TASK<br>~~T~~YPE_}_based on the former facts.||
||Please summarize the rules of the function based on the given facts.||
||Your reply should strictly follow the following format:||
||Rule:||
||def f(A, B, C):||
|**CG**|A, B, C =<<expression>>,<<expression>>,<<expression>>||
||return A, B, C||
||Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Please generate the rule of function based on the former facts.||
||Please summarize the rules of the string transformation based on the given facts.||
||Your reply should strictly follow the following format:||
||Rule: ABC_→_...||
|**ST**|Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Please generate the rule of string transformation based on the former facts.||



Table 13: Prompts for inductive tasks ( _D_ = 3). 

23 

|**Scenario**|**Prompt**||
|---|---|---|
||Please answer the question based on rules of the list transformation in the given facts.||
||Your reply should strictly follow the following format:||
||Answer: [<<expression>>,<<expression>>,<<expression>>]||
|**LT**|Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Question: Input: _{_TEST<br>~~I~~NPUT_}_||
||Please answer the question based on rules of the_{_TASK|TYPE_}_in the given facts.|
||Your reply should strictly follow the following format:||
||Answer: <<expression>>_{_OBJ1_}_,<<expression>>_{_OBJ2_}_,<<expression>>_{_OBJ3_}_.||
|**RP**|Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Question: Input: _{_TEST<br>~~I~~NPUT_}_||
||Please answer the question based on rules of the functioon in the given facts.||
||Your reply should strictly follow the following format:||
||Answer: <<expression>>,<<expression>>,<<expression>>||
|**CG**|Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Question: Input: _{_TEST<br>~~I~~NPUT_}_||
||Please answer the question based on rules of the string transformation in the given facts.||
||Your reply should strictly follow the following format:||
||Answer: ...||
|**ST**|Fact 1: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||...||
||Fact n: Input: _{_INPUT_}_<br>Output: _{_OUTPUT_}_||
||Question: Input: _{_TEST<br>~~I~~NPUT_}_||



Table 14: Prompts for deductive tasks ( _D_ = 3). 

24 

|**Scenario**|**Template**|||**Objects**|
|---|---|---|---|---|
||_{_NAME_}_went to the market to trade items based on the rule.||||
|**Trade**|He originally had_{_obj<br>~~e~~xpression_}_|||chairs, tables, pens ...|
||After the trade, he had_{_obj<br>~~e~~xpression_}_||||
||_{_NAME_}_adjusted his diet plan according to the expert’s advice.||||
|**Diet**|He originally planned to take in_{_obj<br>expression_}_|||apples, bananas, oranges ...|
||After the adjustment, he had_{_obj<br>expression_}_||||
||_{_NAME_}_was performing a card magic trick.||||
|**Magic**|Initially, he had_{_obj<br>~~e~~xpression_}_|||Spade 5s, Jokers, Hearts 6s ...|
||After performing the magic, he ended up with_{_obj|~~e~~xpression_}_|||
||_{_NAME_}_adjusted the investment amount of each asset based on criteria.||||
|**Invest**|Initially, he invested_{_obj<br>~~e~~xpression_}_|||stocks, bonds, funds ...|
||After the adjustment, he invested_{_obj<br>~~e~~xpression_}_||||
||_{_NAME_}_adjusted the students’ courses according|to certain rules.|||
|**Course**|Initially, the weekly course schedule was: _{_obj<br>~~e~~xpression_}_|||math, science, history ...|
||After the adjustment, the weekly course schedule was: _{_obj||~~e~~xpression_}_||



Table 15: Prompts for real-world problems construction. 

25 

