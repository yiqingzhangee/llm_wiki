# **StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving** _[∗]_ 

**Chang Gao** _[♣†]_ **, Haiyun Jiang** _[♢‡]_ **, Deng Cai** _[♡]_ **, Shuming Shi** _[♡]_ **, Wai Lam** _[♣]_ 

> _♣_ The Chinese University of Hong Kong _♡_ Tencent AI Lab 

> _♢_ School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University {gaochang,wlam}@se.cuhk.edu.hk haiyunjiangnlp@gmail.com {jcykcai,shumingshi}@tencent.com 

## **Abstract** 

Most existing prompting methods suffer from the issues of generalizability and consistency, as they often rely on instance-specific solutions that may not be applicable to other instances and lack task-level consistency across the selected few-shot examples. To address these limitations, we propose a comprehensive framework, StrategyLLM, allowing LLMs to perform inductive reasoning, deriving general strategies from specific task instances, and deductive reasoning, applying these general strategies to particular task examples, for constructing generalizable and consistent few-shot prompts. It employs four LLM-based agents: strategy generator, executor, optimizer, and evaluator, working together to generate, evaluate, and select promising strategies for a given task. Experimental results demonstrate that StrategyLLM outperforms the competitive baseline CoT-SC that requires human-annotated solutions on 13 datasets across 4 challenging tasks without human involvement, including math reasoning (34.2% _→_ 38.8%), commonsense reasoning (70.3% _→_ 72.5%), algorithmic reasoning (73.7% _→_ 85.0%), and symbolic reasoning (30.0% _→_ 79.2%). Further analysis reveals that StrategyLLM is applicable to various LLMs and demonstrates advantages across numerous scenarios. Our code is available at https://github.com/gao-xiao-bai/StrategyLLM. 

## **1 Introduction** 

Recent advances in large language models (LLMs) have facilitated the development of prompting techniques [26, 43, 20, 8]. In particular, chain-of-thought (CoT) prompting methods [43, 6, 12, 41], which condition LLMs on a few task examples with step-by-step solutions, guide LLMs to break down complex reasoning processes into simpler steps. These approaches have markedly improved performance compared to standard few-shot prompting across a variety of tasks. 

Despite their potential, current CoT approaches employing few-shot prompts with instance-specific solutions may face challenges in terms of _generalizability_ and _consistency_ . Concerning _generalizability_ , the solution can be highly specific to the question in each instance, limiting its applicability to other instances. For example, as illustrated in the left part of Figure 1, a solution for a particular system of linear equations with two variables may not provide valuable insights for addressing another system with three variables. Furthermore, the solutions in different instances within the few-shot prompt may exhibit a lack of task-level _consistency_ , which complicates the process for LLMs to 

> _∗_ This research was partially supported by the Center for Perceptual and Interactive Intelligence (CPII) Ltd. under the Innovation and Technology Commission’s InnoHK scheme. 

> _†_ Work done during an internship at Tencent AI Lab. 

> _‡_ Haiyun Jiang is the corresponding author. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Task Strategy (Gaussian Elimination Method)<br>1. Write the augmented matrix of the system of linear equations.<br>Solving a system of  2. Perform row operations to transform the matrix into an upper<br>linear equations triangular matrix. The row operations include: …<br>3. Perform back-substitution to find the solution.<br>Specific Solutions Questions Strategy-based Solutions<br>Solution 1:  equation: 𝑦= (5 −𝑥)/2First, solve for . Then substitute this 𝑦 from the first  Question 1:  Solution 1:  2. Row operations:1. Augmented matrix: subtracting 3 times the first row from the 13 −12 58<br>expression into the second equation: (5 −𝑥)/2 = 8substitute 𝑥 back into the first equation: . Solve for  𝑥 : 𝑥= 3. Now 3𝑥−𝑦= /3𝑥−𝑦= 8 [𝑥+ 2𝑦= 5] second row3. Back-substitution: 10 −72 −7𝑦= −7/−7 = 15 [. Now we have an upper triangular matrix.] 𝑥= (5 −2 ∗1)/1 = 3<br>5 −3 /2 = 1.<br>Solution 2:  Subtracting the second equation  Question 2:  Solution 2:  1. Augmented matrix: 11 −12 82<br>from the first equation: Now substitute 𝑥= 2 + 2 = 4.  𝑦 into the second equation: 𝑦= (8 −2)/3 = 2.  / [𝑥+ 2𝑦= 8] 𝑥−𝑦= 2 2. Row operations:3. Back-substitution: 10 −32 −68 [. Now we have an upper triangular matrix.]  subtracting the first row from the second row𝑦= −6/−3 = 2 𝑥= (8 −2 ∗2)/1 = 4<br>Specific Solution New Question Strategy-based Solution<br>How to solve? Applying the strategy!<br>𝑥+ 𝑦+ 𝑧= 7<br>…… 2𝑥−𝑦+ 𝑧= 1 1. Augmented matrix: …<br>43𝑥+ 2𝑦−𝑧= 1 2. Row operations: …<br>3. Back-substitution: …<br>**----- End of picture text -----**<br>


Figure 1: Comparison of specific solutions and strategy-based solutions. 

develop effective solutions for tackling new instances. As demonstrated in the left part of Figure 1, the two specific solutions are based on different approaches: Solution 1 employs expression substitution, while Solution 2 utilizes equation subtraction, which may not provide consistent guidance for LLMs to solve new instances. To address these limitations, it is crucial to incorporate _effective problem-solving strategies_ and develop _consistent strategy-based solutions_ within few-shot prompts. The right part of Figure 1 presents an effective strategy, i.e., Gaussian Elimination Method, offering generalizable steps applicable to any system of linear equations. By providing this strategy and consistently applying it across various instances in the few-shot prompt, LLMs can be better equipped to generate effective solutions for new task instances. 

This paper aims to _construct generalizable and consistent strategy-based few-shot prompts for various tasks automatically, while being highly cost-efficient_ . Our proposed framework, StrategyLLM, draws inspiration from human cognitive processes to derive general problem-solving strategies. This approach enables LLMs to reason inductively, i.e., deriving general strategies from specific task instances, and deductively, i.e., applying general strategies to particular task examples, to formulate prompts. An example of strategy-based prompts can be seen in Figure 5. The inductive reasoning process enhances _generalizability_ by formulating general problem-solving strategies, while the deductive reasoning process improves _consistency_ by producing consistent solutions using a given strategy. Developing effective problem-solving strategies is crucial to the success of our framework. To achieve this, we design StrategyLLM as a multi-agent collaboration framework comprising four LLM-based agents—strategy generator, executor, optimizer, and evaluator, as shown in Figure 2. The strategy generator initially creates a pool of strategies that are executed on task examples to assess accuracy, with qualified strategies cached based on a threshold and further evaluated. Unqualified ones may be optimized and re-evaluated iteratively. Through the collaboration of these intelligent agents, our framework is capable of autonomously generating, evaluating, and selecting effective strategies for various tasks and eliminating the need for human involvement. 

Crucially, the strategy-based few-shot prompt generation phase is applied once for a given task, after which the learned prompt can be employed for inference on the entire test set. This inference process does not require any additional input beyond the standard few-shot prompting settings. The prompt generation process is highly cost-effective as it necessitates only a few task examples. In particular, StrategyLLM expends less than $0.24 to develop a strategy-based prompt for a variety of tasks using the latest version of GPT-3.5-Turbo. 

We conduct comprehensive evaluations of StrategyLLM on 13 datasets across 4 challenging tasks: math reasoning, commonsense reasoning, algorithmic reasoning, and symbolic reasoning. The experimental results reveal the following key findings: (1) StrategyLLM outperforms competitive baselines on all tasks without using any human-annotated reasoning processes; (2) StrategyLLM can be applied to various LLMs and is robust to different groups of task examples; (3) StrategyLLM can generate generalizable and consistent prompts in a cost-effective manner. These findings demonstrate the potential of StrategyLLM as an effective, efficient, and reliable problem-solving framework. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Task Definition<br>Qualified Strategies<br>The task is to solve diverse math word problems. Execution Accuracy >= Threshold<br>Task Examples Strategy 1:Strategy 3: ……  Execution Result 1: Execution Result 3: …… Cache Top  k EvaluatorStrategy<br>Question 1:  …   Answer 1:  … ……<br>…… Question 8: …    Answer 8:  … Strategy<br>Executor Unqualified Strategies<br>Execution Accuracy < Threshold<br>Strategy 2: … Execution Result 2: … If |Cache| <  k Strategy<br>Strategy Generator Strategy 5: …  Execution Result 5: … Optimizer<br>……<br>Strategies<br>Strategy 1:  1. Identify the given information … Updated Strategies<br>…… Strategy 2: …   Strategy 5: …<br>Strategy n:  1. Understand the problem by … ……<br>**----- End of picture text -----**<br>


Figure 2: Overview of StrategyLLM. Initially, the strategy generator creates a pool of strategies, which are then applied by the strategy executor to task examples to calculate execution accuracy. Qualified strategies meeting a pre-defined threshold are cached, and if necessary, unqualified strategies are optimized and re-evaluated in iterative cycles. Once a sufficient number of qualified strategies are obtained or the maximum iteration number is reached, the top _k_ strategies are ranked by execution accuracy and evaluated using a validation set. 

## **2 StrategyLLM** 

Our StrategyLLM framework is designed to efficiently create strategy-based few-shot prompts for a wide range of tasks. Subsequently, these prompts can be utilized for inference. In this section, we will introduce our framework in detail. The inference procedure will be discussed in Section 3. 

**Overview of StrategyLLM** As presented in Figure 2, our framework consists of four key agents: strategy generator, executor, optimizer, and evaluator. The prompts for the strategy generator and executor are presented in Figure 3 and Figure 4, respectively. The prompts of the strategy optimizer are in Appendix C. Typically, only a few task examples are used in the collaboration process, making our framework highly efficient. 

The collaboration process begins with the strategy generator formulating a pool of strategies based on its understanding of the target task. These strategies then undergo two rounds of validation and selection. In the first round, the strategy executor applies each strategy to a set of task examples to yield its execution result and compute its execution accuracy. Strategies that meet or exceed a pre-set threshold of execution accuracy are deemed qualified and are cached with their corresponding execution results and accuracy. If the number of qualified strategies is less than a pre-defined number _k_ , the optimizer refines the unqualified strategies using their execution results. These enhanced strategies are then sent back to the strategy executor for the next iteration. This cycle may repeat until a sufficient number of qualified strategies are achieved or the maximum iteration limit is reached. Following this, all cached strategies are ranked based on their execution accuracy, and the top _k_ strategies are selected. In the second round, the strategy evaluator constructs strategy-based few-shot prompts for each candidate strategy using itself and its execution result and assesses all candidate strategies using their corresponding prompts for inference on a validation set. 

**Notations** We use _p_ , _q_ , _st_ , _so_ , and _a_ to denote the prompt, question, strategy, solution, and answer, respectively. During inference, given a question _q_ , the language model _M_ : ( _p, q_ ) _→_ ( _so, a_ ) generates a solution _so_ and an answer _a_ for it conditioned on the prompt _p_ . We denote the target task as _t_ , its definition as _d_ , and the set of task examples as _E_ . Each example in _E_ is a ( _q, a_ ) pair. 

**Strategy Definition** In this paper, a task-solving strategy _st_ is defined as a systematic approach designed for universal application across task examples, comprising a series of subtasks that encode task knowledge to address the target task _t_ . It is characterized by the following properties: (1) _Task-Level Applicability_ : The strategy is formulated in a manner that allows for its application across all task instances, ensuring universality and consistency in its implementation. (2) _Structured Organization_ : The strategy comprises a sequence of subtasks that are organized in a logical order to collectively tackle the target task. These subtasks are interconnected and contribute to the overall achievement of the task objective. (3) _Task Knowledge Encoding_ : The strategy encapsulates general 

3 

Strategy Generator Task: {task definition} Some examples of the task are as follows: {task examples} Let’s understand the task and write a strategy that consists of a sequence of subtasks to solve the task. For writing, you must satisfy the following requirements: - Include all necessary subtasks. - All subtasks are easy to solve. - Subtasks are in an appropriate order. - Do not include specific information in the examples. - Make sure the strategy is general and concise. The result must be a numbered list in the following format: 1. First subtask 2. Second subtask 

Figure 3: Prompt of the strategy generator. 

Strategy Executor Task: {task definition} Example of the task: {example} Strategy: {strategy} The strategy consists of a sequence of subtasks for solving the task. Please execute the strategy on the provided example. For executing, you need to write a step-by-step solution to the example based on the subtasks. The solution must satisfy the following requirements: - Adjust and execute these subtasks for this example. - Compute as many intermediate results as possible. - The answer obtained from the solution must be the same as the original answer. The result must be in the following format: Question: Question in the provided example Solution: Solution obtained based on the subtasks in the strategy Answer: Answer in the provided example, which must include the string ‘The answer is ’ 

Figure 4: Prompt of the strategy executor. 

task knowledge and principles, avoiding any specific details unique to individual task examples. These properties collectively contribute to the effectiveness and efficiency of a strategy by promoting consistency, clarity, and informed decision-making in addressing task-level challenges. By embodying these properties, a strategy can serve as a valuable tool to navigate complex tasks and achieve optimal outcomes. An example of the strategy is presented in Figure 5. 

**Strategy Generator** _G_ The strategy generator, represented as _G_ : ( _E, d, n_ ) _→{stj}[n] j_ =1[,][aims] to generate _n_ diverse strategies for the target task _t_ based on a set of task examples _E_ and the task definition _d_ using temperature sampling. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Standard Prompt Strategy-based Prompt<br>Strategy:<br>Question:  Jason had 20 lollipops. He gave  1. Identify the given information in the word problem<br>Denny some lollipops. Now Jason has 12 lollipops.  2. Determine the desired information or what needs to be calculated<br>How many lollipops did Jason give to Denny? 3. Identify the mathematical operation needed to solve the problem<br>Answer:  The answer is 8. 4. Calculate the solution using the given information and the appropriate mathematical operation<br>…… 5. Verify the solution by checking if it satisfies the conditions stated in the problem<br>CoT Prompt Question:  Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops.<br>Question:  Jason had 20 lollipops. He gave  How many lollipops did Jason give to Denny?<br>Denny some lollipops. Now Jason has 12 lollipops.  Solution:  1. Given information: Jason had 20 lollipops and now he has 12 lollipops.<br>How many lollipops did Jason give to Denny? 2. Desired information: Number of lollipops Jason gave to Denny.<br>Solution:  Jason started with 20 lollipops. Then he  3. Mathematical operation: Subtraction.<br>had 12 after giving some to Denny. So he gave  4. Calculation: find the difference between the initial and final number of lollipops: 20 - 12 = 8.<br>Denny 20 - 12 = 8. 5. Verification: If Jason had 20 lollipops and gave 8 to Denny, he would have 12 lollipops left.<br>Answer:  The answer is 8. Answer:  The answer is 8.<br>…… ……<br>**----- End of picture text -----**<br>


Figure 5: Comparison of the strategy-based, standard, and chain-of-thought (CoT) [43] prompt. 

**Strategy Executor** _X_ The strategy executor, denoted as _X_ : ( _E, d, st_ ) _→_ ( _Rst, eaccst_ ), writes solutions to a set of task examples _E_ following the strategy _st_ to obtain the execution result _Rst_ = _{_ ( _q, so, a_ ) _}[|E|] i_ =1[of] _[st]_[.][The][execution][accuracy] _[eacc][st]_[is][calculated][as][the][proportion][of][examples] whose solutions yield correct answers, reflecting the degree of alignment between the strategy and task. Therefore, we select strategies with high execution accuracy as qualified strategies. 

**Strategy Optimizer** _O_ The strategy optimizer, represented as _O_ : ( _E, d, st, Rst_ ) _→ st[o]_ , optimize the strategy _st_ according to its execution result _Rst_ to obtain the updated strategy _st[o]_ . Firstly, the strategy optimizer _O_ analyzes why some solutions in _Rst_ are not correct and provides suggestions for improving _st_ . Secondly, it modifies _st_ to obtain _st[o]_ based on the analysis and suggestions. 

**Strategy Evaluator** _E_ We select top _k_ candidate strategies according to the execution accuracy. However, to ensure efficiency, we use a limited number of task examples for execution, making the execution accuracy not a very informative metric for choosing strategies. Therefore, we introduce a strategy evaluator to further evaluate the candidate strategies on a validation set _V_ . This process only requires to perform inference once for each candidate strategy and is efficient. The strategy evaluator, denoted as _E_ : ( _st, Rst, V_ ) _→ vaccst_ , computes the validation accuracy _vaccst_ of the strategy _st_ on _V_ . To achieve this, it constructs the strategy-based few-shot prompt _pst_ = ( _st, Rst_ ) and conducts inference on _V_ using _pst_ . An example of strategy-based prompts is presented in Figure 5. The validation accuracy _vaccst_ is calculated as the percentage of validation examples whose answers are correct, reflecting the effectiveness of _st_ in real-world scenarios. Strategies with high validation accuracy can be used for inference. 

## **3 Inference** 

Through collaborative efforts among multiple agents, we have obtained multiple candidate strategies, each with its few-shot prompt and validation accuracy. Depending on the task at hand, we can select one or more strategies with high validation accuracy for inference. For simple or specific tasks, a single optimal strategy that solves all task examples effectively may exist, making it sufficient to use only one strategy. However, for complex or diverse tasks, it is unlikely to find a strategy with absolute superiority. In such cases, adopting multiple strategies for inference is more appropriate, as they may excel in different task examples. To harness the strengths of multiple strategies, we employ two methods. The first method involves taking a majority vote on all answers obtained by multiple strategies, akin to the self-consistency (SC) method [41]. The second method requires LLMs to determine the final answer by considering the solutions derived from multiple strategies in a zero-shot (ZS) manner, making it more proper for complex and diverse tasks. We denote the first and second methods as StrategyLLM-SC and StrategyLLM-ZS, respectively. The prompt for the second approach is provided in Appendix D. 

## **4 Experiments** 

## **4.1 Experimental Setup** 

**Evaluation Tasks and Datasets** We evaluate StrategyLLM on a variety of tasks: 

5 

Table 1: Experimental results on the math reasoning task. The numbers in parentheses represent the relative improvement compared to CoT-SC. 

|**Methods**|AL|PA|IA|CP|NT|GE|PC|Avg||
|---|---|---|---|---|---|---|---|---|---|
|SP|32.0|50.0|17.5|27.0|20.5|21.0|20.5|26.9||
|SolutionLLM|58.5|56.5|13.5|33.0|32.0|28.0|19.5|34.4||
|CoT|57.0|57.5|15.0|33.5|28.0|23.0|20.0|33.4||
|CoT-SC|59.0|62.0|16.5|34.5|28.0|24.5|15.0|34.2||
|StrategyLLM|58.5|57.5|18.0|35.0|29.5|24.5|22.5|35.1||
|StrategyLLM-SC|60.0|61.5|18.0|38.5|30.5|28.0|**24.0**|37.2|(+8.8%)|
|StrategyLLM-ZS|**64.5**|**65.5**|**19.0**|**39.0**|**32.5**|**28.5**|22.5|**38.8**|**(+13.4%)**|



Table 2: Experimental results on the commonsense, algorithmic, and symbolic reasoning tasks. The numbers in parentheses represent the relative improvement compared to CoT-SC. 

|**Methods**|Commonsense|Algorithmic|Symbolic|
|---|---|---|---|
||StrategyQA DU Avg|WS<br>MA Avg|LLC-4 LLC-8 LLC-16 Avg|
|SP<br>SolutionLLM<br>CoT<br>CoT-SC|56.5<br>48.5 52.5<br>59.5<br>52.0 55.8<br>64.0<br>70.5 67.3<br>70.0<br>70.5 70.3|73.3 2.0 37.7<br>74.7 55.3 65.0<br>67.2 84.0 75.6<br>61.3 86.0 73.7|0<br>0<br>0<br>0<br>81.5<br>25.5<br>0<br>35.7<br>68.5<br>22.0<br>0<br>30.2<br>68.0<br>22.0<br>0<br>30.0|
|StrategyLLM<br>67.5<br>68.5 68.0<br>StrategyLLM-SC<br>**71.0**<br>**74.0 72.5 (+3.1%)**<br>StrategyLLM-ZS<br>70.0<br>72.5 71.3 (+1.4%)||**80.0** 86.7 83.4<br>79.3 **90.7 85.0 (+15.4%)**<br>78.7 89.3 84.0 (+14.1%)|**98.0**<br>86.5<br>51.5<br>78.7<br>**98.0**<br>**87.0**<br>**52.5**<br>**79.2 (+164.0%)**<br>**98.0**<br>86.0<br>44.0<br>76.0 (+153.3%)|



- **Math Reasoning** : We use the challenging MATH benchmark [16] which comprises problems from mathematics competitions that require more than standard K-12 mathematics tools. It consists of seven datasets of different subjects, namely, Algebra (AL), Prealgebra (PA), Intermediate Algebra (IA), Counting and Probability (CP), Number Theory (NT), Geometry (GE), and Precalculus (PC). 

- **Commonsense Reasoning** : We employ StrategyQA [14] and the Date Understanding (DU) task from Big-Bench Hard [38, 9]. StrategyQA necessitates inferring a multi-hop strategy to answer questions, while the DU task involves deducing a date from a given context. 

- **Algorithmic Reasoning** : We adopt the Word Sorting (WS) task and the Multi-step Arithmetic (MA) task from Big-Bench Hard [38, 9]. The WS task involves sorting a list of words lexicographically, and the MA task requires solving multi-step equations with basic arithmetic operations. 

- **Symbolic Reasoning** : We utilize the Last Letter Concatenation (LLC) task from [43], which requires concatenating the last letters of words in a sequence. In the few-shot prompt, the model only sees examples with two words. To evaluate the generalization abilities of different methods, we construct three out-of-distribution test sets (LLC-4, LLC-8, and LLC-16) with 4, 8, and 16 words in a sequence, respectively. 

**Baselines** We conduct experiments in the few-shot setting and compare StrategyLLM with the following baselines: 

- **Standard Prompting (SP)** : SP is the most direct approach for problem-solving. In SP, the prompt _p_ contains a set of question-answer pairs without intermediate reasoning steps. 

- **Chain-of-Thought (CoT) Prompting** [43]: CoT incorporates step-by-step solutions for questions in the prompt _p_ to elicit the multi-step reasoning capabilities of LLMs. We use few-shot CoT prompts from [43] for StrategyQA, DU, and LLC, and prompts from [38] for WS and MA. For MATH datasets, we create few-shot CoT prompts by randomly sampling 4 examples from each dataset’s training set since these datasets contain human-annotated solutions. The CoT prompts for these datasets are in Appendix H. 

- **Self-Consistency with CoT (CoT-SC)** [41]: CoT-SC generates a set of solutions using CoT via temperature sampling to obtain multiple answers. Subsequently, it takes a majority vote over these answers to determine the final answer. For experiments, we sample 3 reasoning paths using temperature sampling with a temperature of 0.7. 

6 

Table 3: Experimental results on two math reasoning datasets, namely AL and CP, with different groups of examples. 

|**Methods**|AL-dev|AL-random|CP-dev|CP-random|
|---|---|---|---|---|
|SP|36.0|29.1_±_3.9|25.5|26.8_±_2.5|
|SolutionLLM|58.0|56.5_±_2.2|31.0|32.2_±_2.8|
|CoT|57.5|55.1_±_1.5|34.0|33.4_±_1.2|
|CoT-SC|59.5|58.3_±_1.2|31.5|33.0_±_1.2|
|StrategyLLM|57.0|54.7_±_2.5|34.5|35.6_±_2.3|
|StrategyLLM-SC|**64.0**|58.9_±_1.1|36.5|38.4_±_1.3|
|StrategyLLM-ZS|62.5|**60.8**_±_**2.6**|**38.5**|**38.8**_±_**1.7**|



- **SolutionLLM** : We construct this baseline to leverage LLMs to directly write the solution for each example in the few-shot prompts using greedy decoding, without using any strategies. The prompt of SolutionLLM is in Appendix E. Since both SolutionLLM and StrategyLLM generate prompts using LLMs, we can eliminate the potential effect of human expertise in the comparison, isolating the impact of incorporating effective strategies. 

**Implementation Details** We employ GPT-3.5 (gpt-3.5-turbo-16k-0613) [32] as the language model for our main experiments, serving as the backend for the strategy generator, executor, optimizer, and evaluator. For a fair comparison with baselines such as CoT, we use the same examples in their few-shot prompts for strategy generation, execution, and optimization. We select the top 1 or 3 strategies with the highest validation accuracy for inference. This allows us to demonstrate the performance of the optimal strategy and the benefits of using multiple strategies. We adopt greedy decoding for inference. Details of the strategies for each dataset can be found in Appendix G. The validation set size is 100 for all the datasets. For datasets with over 200 test examples, we randomly sample 200 examples for testing to reduce API costs. More details can be found in Appendix B. 

## **4.2 Main Results** 

Tables 1 and 2 present the experimental results of StrategyLLM and several baselines across four reasoning tasks. We have the following observations: 

- _StrategyLLM is an effective and efficient framework for problem-solving_ . StrategyLLM using multiple strategies, i.e., StrategyLLM-SC and StrategyLLM-ZS, outperforms all baselines across the four reasoning tasks. Furthermore, StrategyLLM employing the best discovered strategy consistently outperforms CoT and SolutionLLM. Notably, StrategyLLM automatically constructs generalizable and consistent few-shot prompts for tackling various tasks without human expertise, while CoT relies on human-annotated examples for each task. 

- _Explicitly incorporating effective strategies significantly enhance the complex reasoning and out-ofdistribution (OOD) generalization abilities of LLMs_ . For example, our framework demonstrates more considerable improvements on the MATH benchmark compared to the simpler commonsense reasoning datasets. Furthermore, StrategyLLM substantially surpasses CoT and SolutionLLM on the three OOD test sets of the LLC task, showcasing the generalizability of effective strategies. 

- _Adopting multiple strategies brings obvious benefits on complex or diverse tasks_ . The performance of StrategyLLM is significantly improved by using multiple strategies on the math and commonsense reasoning tasks. The benefits of leveraging multiple strategies on simpler or more specific tasks, i.e., symbolic and algorithmic reasoning, is less significant. These observations indicate that our framework is capable of creating multiple complementary strategies for diverse or complex tasks. Furthermore, StrategyLLM-ZS outperforms StrategyLLM-SC on the math reasoning task, showing that allowing LLMs to determine the answer is more appropriate for intricate tasks. 

## **5 Analysis** 

**Evaluating the robustness of StrategyLLM** We conduct an investigation to assess the robustness of our StrategyLLM framework with respect to varying groups of examples. For this purpose, we 

7 

Table 4: Experimental results of closed-source models on the CP, StrategyQA, and MA datasets. The numbers in parentheses represent the relative improvement compared to CoT-SC. 

|**Methods**|GPT-4<br>CP StrategyQA MA Avg|Claude-3-Sonnet|
|---|---|---|
|||CP StrategyQA MA Avg|
|SolutionLLM<br>CoT<br>CoT-SC|52.0<br>75.5<br>96.7 74.7<br>49.5<br>80.5<br>92.7 74.2<br>54.5<br>**83.5**<br>94.7 77.6|21.0<br>73.5<br>69.3 54.6<br>26.0<br>69.0<br>72.7 55.9<br>26.0<br>75.0<br>76.7 59.2|
|StrategyLLM<br>StrategyLLM-SC|52.5<br>81.5<br>**98.7** 77.2<br> **56.0**<br>**83.5**<br>**98.7 79.4 (+2.4%)**|**28.0**<br>75.0<br>83.3 62.1<br>**28.0**<br>**77.0**<br>**88.0 64.3 (+8.6%)**|



Table 5: Experimental results of open-source models on the CP, StrategyQA, and MA datasets. The numbers in parentheses represent the relative improvement compared to CoT-SC. 

|**Methods**|Meta-Llama-3-8B-Instruct|Meta-Llama-3-70B-Instruct|
|---|---|---|
||CP StrategyQA MA Avg|CP StrategyQA MA Avg|
|SolutionLLM<br>CoT<br>CoT-SC|20.5<br>64.0<br>43.3 42.6<br>16.0<br>61.0<br>44.7 40.6<br>19.5<br>71.0<br>45.3 45.3|51.5<br>79.0<br>72.0 67.5<br>48.5<br>80.5<br>81.3 70.1<br>47.0<br>81.5<br>82.0 70.2|
|StrategyLLM<br>StrategyLLM-SC|24.5<br>**74.0**<br>64.7 54.4<br> **25.0**<br>**74.0**<br>**66.0 55.0 (+21.5%)**|51.5<br>82.0<br>88.0 73.8<br>**54.0**<br>**83.5**<br>**91.3 76.3 (+8.7%)**|
|**Methods**|Mixtral-8x7B-Instruct-v0.1|Mixtral-8x22B-Instruct-v0.1|
||CP StrategyQA MA Avg|CP StrategyQA MA Avg|
|SolutionLLM<br>CoT<br>CoT-SC|22.5<br>61.0<br>34.7 39.4<br>24.5<br>63.0<br>59.3 48.9<br>26.5<br>73.5<br>62.7 54.2|44.5<br>72.0<br>60.7 59.1<br>41.0<br>72.0<br>80.0 64.3<br>40.5<br>75.0<br>80.7 65.4|
|StrategyLLM<br>StrategyLLM-SC|28.5<br>73.5<br>76.0 59.3<br> **32.0**<br>**75.0**<br>**78.0 61.7 (+13.7%)**|44.5<br>76.5<br>84.0 68.3<br>**47.5**<br>**77.0**<br>**89.3 71.3 (+9.0%)**|



select two math reasoning datasets with diverse examples, namely AL and CP, and randomly sample 5 distinct groups of examples from their respective training sets. We then report the mean and standard deviation of the results. Additionally, we employ the validation set to identify a group of 4 examples from the training set. Specifically, we use the OpenAI embedding model API (the text-embedding3-large model) to map training and validation questions to embeddings and subsequently select the 4 training examples with the highest cosine similarities to all validation examples. We designate these groups of examples as AL-dev and CP-dev, respectively. The results, as presented in Table 3, demonstrate that StrategyLLM consistently delivers satisfactory performance on both datasets, suggesting that StrategyLLM is a robust and reliable framework for problem-solving. 

**Exploring the universality of StrategyLLM** To investigate the universality of our StrategyLLM framework, we apply it to a variety of LLMs to evaluate its effectiveness. For closed-source models, we utilize GPT-4 (gpt-4-0613) [31] and Claude-3-Sonnet (claude-3-sonnet-20240229) [2]. For opensource models, we employ Meta-Llama-3-8B-Instruct, Meta-Llama-3-70B-Instruct [1], Mixtral-8x7BInstruct-v0.1, and Mixtral-8x22B-Instruct-v0.1 [18]. We conduct experiments on the CP, StrategyQA, and MA datasets, which represent three distinct reasoning tasks. The results, summarized in Tables 4 and 5, reveal that integrating effective strategies for constructing generalizable and consistent fewshot prompts yields significant benefits across a range of model capabilities and task complexities, underscoring the framework’s universality. StrategyLLM notably enhances performance in opensource models such as Meta-Llama-3-8B-Instruct and Mixtral-8x7B-Instruct-v0.1, particularly on the CP and MA datasets which demand complex reasoning, indicating the effectiveness of our framework in scenarios requiring sophisticated problem-solving. These findings further corroborate StrategyLLM’s robustness and reliability as a problem-solving framework. 

**Comparing reasoning via task-level strategy and instance-specific planning** Our framework facilitates generalizable and consistent reasoning by developing task-level strategies. To evaluate 

8 

Table 6: Comparison of Plan-and-Solve, CoT+Strategy, and StrategyLLM. 

|**Methods**|CP|StrategyQA|MA|Avg|
|---|---|---|---|---|
|Plan-and-Solve|26.0|54.0|69.3|49.8|
|Plan-and-Solve-SC|27.5|64.5|70.0|54.0|
|CoT+Strategy|30.5|63.0|62.7|52.1|
|CoT+Strategy-SC|36.5|70.0|70.0|58.8|
|StrategyLLM|35.0|67.5|86.7|63.1|
|StrategyLLM-SC|**38.5**|**71.0**|**90.7**|**66.7**|



the necessity of effective task-level strategies, we compare our framework against two baselines: (1) Plan-and-Solve Prompting [40], which directs LLMs to formulate specific plans for each test instance at inference time and execute these plans to solve the instances; (2) CoT+Strategy, which combines the CoT prompt with instructions that guide LLMs to devise a task-solving strategy and apply it to a specific test example at inference time. The prompt for CoT+Strategy is detailed in Appendix F. 

The performance of GPT-3.5 on the CP, StrategyQA, and MA datasets, representing three distinct reasoning tasks, is presented in Table 6. Our observations are as follows: (1) StrategyLLM significantly outperforms both Plan-and-Solve Prompting and CoT+Strategy across all three datasets. This highlights the superiority of generalizable task-level strategies over instance-specific plans in enhancing performance across various problem-solving contexts. This improvement can be attributed to two key factors: (a) our task-level strategies encapsulate essential task-level knowledge, thereby providing professional and high-level guidance; (b) generating high-quality, specific plans for each test example at inference time is inherently challenging, making it difficult to ensure the quality of these plans. (2) Even when explicitly encouraged to devise a general task-solving strategy in the CoT+Strategy method, the LLM tends to produce strategies that are highly specific to the test example and encode limited task-level knowledge. This underscores the necessity of creating generalizable strategy-based few-shot prompts. 

**Analyzing the cost of strategy-based prompt generation** In this analysis, we evaluate the cost of the strategy-based prompt generation process. The process includes the strategy generator, executor, optimizer, and evaluator, each contributing to the overall cost for each reasoning task. Table 7 details the average cost incurred by our StrategyLLM framework in generating a candidate strategy-based prompt, calculated by dividing the total cost of the process by the number of candidate strategies _k_ . The costs are presented in terms of input and output tokens and the money associated with using GPT-3.5-Turbo. The results indicate that our framework is economically efficient. The average cost for gpt-3.5-turbo-16k-0613 ranges from $0.33 to $1.12 across the four reasoning tasks. For the latest version of GPT-3.5-Turbo, specifically gpt-3.5-turbo-0125, the cost is considerably lower, ranging from $0.08 to $0.24. Generally, tasks of higher complexity consume more tokens due to their inherently longer solutions. 

## **Examining results across various difficulty** 

**levels** The problems in the MATH benchmark are classified by difficulty on a scale of 1 to 5. The easiest problems are assigned a difficulty level of 1, while the most challenging problems are given a difficulty level of 5. Figure 6 illustrates the performance of CoT-SC and StrategyLLM-SC on the seven datasets within the MATH benchmark, considering different difficulty levels. It is evident that the enhanced performance of StrategyLLM-SC over CoT-SC stems from its ability to tackle more complex problems, underscoring the significance of generalizable strategies in augmenting intricate reasoning. 

More analysis can be found in Appendix A. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0009-08.png)


**----- Start of picture text -----**<br>
72.65 70.94 CoT-SC StrategyLLM-SC<br>70<br>60<br>52.06<br>49.81<br>50<br>45.21<br>42.90<br>40<br>33.33<br>30.38<br>30<br>20<br>13.10<br>11.50<br>10<br>1 2 3 4 5<br>Difficulty Level<br>Performance (%)<br>**----- End of picture text -----**<br>


Figure 6: Comparison of CoT-SC and StrategyLLM-SC performance on the MATH benchmark across various difficulty levels. 

9 

Table 7: Average cost of prompt generation across four reasoning tasks. 

||Math|Commonsense|Algorithmic|Symbolic|
|---|---|---|---|---|
|# Input Tokens|287.83K|228.67K|107.27K|70.94K|
|# Output Tokens|63.14K|33.15K|32.95K|28.48K|
|Cost ofgpt-3.5-turbo-16k-0613|$1.12|$0.82|$0.45|$0.33|
|Cost ofgpt-3.5-turbo-0125|$0.24|$0.16|$0.10|$0.08|



## **6 Related Work** 

**Prompting LLMs for Problem Solving** The prominent chain-of-thought (CoT) prompting approach [43] has inspired a variety of prompting methods aimed at enhancing the problem-solving abilities of LLMs. These methods include using programming languages to describe the reasoning process [6, 13, 28], representing the reasoning process with complex structures such as trees or graphs [46, 3, 36, 47], applying task decomposition [49, 19, 34, 4], implementing self-correction with automatic feedback [22, 29, 30, 5, 7], and combining different prompting techniques [27, 50]. However, most of these approaches require manual annotation of reasoning processes, limiting their generalizability and flexibility. By comparison, our StrategyLLM framework can automatically construct strategy-based few-shot prompts for any task, ensuring generalizable and consistent solutions following effective strategies. This approach sets it apart from existing automatic prompt construction methods [48, 37, 45], which may generate inconsistent solutions within the prompt. The plan-and-solve prompting method [40] aims to address missing-step errors by requesting LLMs to generate a plan before solving a specific example in a zero-shot manner. The plan is instance-specific and significantly different from the task-solving strategy which can be applied to all task examples. The learning-to-plan approach [15] learns a text plan for each task to assist LLMs in problem-solving. The plan, which is not necessarily a strategy, can be any instruction helpful for solving the task. Moreover, it demands a large training and validation set during the learning process, resulting in high costs. In contrast, our framework is efficient and cost-effective. 

**LLM-based Autonomous Agents** The adoption of autonomous agents driven by LLMs across various disciplines is revolutionizing our methodologies for tackling problems, making decisions, and fostering innovation [39, 44]. These agents have been utilized to enhance the reasoning capabilities of LLMs [42, 24, 11], contribute to social simulation [33, 23, 25, 21], and advance software development [35, 17, 10]. In this paper, we employ multiple LLM-based agents to collaborate in the generation, execution, optimization, and evaluation of problem-solving strategies. 

## **7 Discussion** 

**Limitation and Impact** The key idea behind StrategyLLM is to harness the knowledge and reasoning capabilities of LLMs to develop and refine task-solving strategies tailored to specific tasks. By utilizing the extensive knowledge embedded in these LLMs, which are trained on diverse data sources spanning multiple domains, StrategyLLM is able to generate generalizable strategies that incorporate domain-specific expertise. However, if the model possesses limited knowledge in a particular domain, it is unlikely to create effective strategies for that domain. In such cases, merely optimizing the prompt may not significantly improve performance, and domain-specific continual training may be necessary. As LLMs continue to expand their knowledge bases and enhance their reasoning capabilities, their ability to generate generalizable strategies for diverse tasks is expected to improve, implying the potential of our StrategyLLM framework. 

**Conclusion** This paper proposes StrategyLLM, harnessing the power of LLMs to construct generalizable and consistent few-shot prompts for various tasks efficiently. Our framework’s effectiveness and reliability are substantiated through extensive evaluations on four challenging tasks: mathematical reasoning, commonsense reasoning, algorithmic reasoning, and symbolic reasoning. Further analysis reveals that our framework exhibits robustness across different task example groups, application to various LLMs, cost-efficiency in prompt generation, and effectiveness in complex reasoning. 

10 

## **References** 

- [1] Meta AI. Introducing meta llama 3: The most capable openly available llm to date. 2024. 

- [2] Anthropic. Introducing the next generation of claude. 2024. 

- [3] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 17682–17690, 2024. 

- [4] Jiaao Chen, Xiaoman Pan, Dian Yu, Kaiqiang Song, Xiaoyang Wang, Dong Yu, and Jianshu Chen. Skills-in-context prompting: Unlocking compositionality in large language models. _arXiv preprint arXiv:2308.00304_ , 2023. 

- [5] Pinzhen Chen, Zhicheng Guo, Barry Haddow, and Kenneth Heafield. Iterative translation refinement with large language models. _arXiv preprint arXiv:2306.03856_ , 2023. 

- [6] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _Transactions on Machine Learning Research_ , 2023. 

- [7] Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [8] Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. A survey of chain of thought reasoning: Advances, frontiers and future. _arXiv preprint arXiv:2309.15402_ , 2023. 

- [9] BIG-Bench collaboration. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions on Machine Learning Research_ , 2023. 

- [10] Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via chatgpt. _arXiv preprint arXiv:2304.07590_ , 2023. 

- [11] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. _arXiv preprint arXiv:2305.14325_ , 2023. 

- [12] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [13] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 10764–10799. PMLR, 23–29 Jul 2023. 

- [14] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , 9:346–361, 2021. 

- [15] Yiduo Guo, Yaobo Liang, Chenfei Wu, Wenshan Wu, Dongyan Zhao, and Nan Duan. Learning to plan with natural language. _arXiv preprint arXiv:2304.10464_ , 2023. 

- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021. 

- [17] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [18] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

11 

- [19] Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [20] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. 

- [21] Grgur Kovaˇc, Rémy Portelas, Peter Ford Dominey, and Pierre-Yves Oudeyer. The socialai school: Insights from developmental psychology towards artificial socio-cultural agents. _arXiv preprint arXiv:2307.07871_ , 2023. 

- [22] Miaoran Li, Baolin Peng, and Zhu Zhang. Self-checker: Plug-and-play modules for factchecking with large language models. _arXiv preprint arXiv:2305.14623_ , 2023. 

- [23] Siyu Li, Jin Yang, and Kui Zhao. Are you in a masquerade? exploring the behavior and impact of large language model driven social bots in online social networks. _arXiv preprint arXiv:2307.10337_ , 2023. 

- [24] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multi-agent debate. _arXiv preprint arXiv:2305.19118_ , 2023. 

- [25] Jiaju Lin, Haoran Zhao, Aochi Zhang, Yiting Wu, Huqiuyue Ping, and Qin Chen. Agentsims: An open-source sandbox for large language model evaluation. _arXiv preprint arXiv:2308.04026_ , 2023. 

- [26] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. _ACM Comput. Surv._ , 55(9), jan 2023. 

- [27] Tengxiao Liu, Qipeng Guo, Yuqing Yang, Xiangkun Hu, Yue Zhang, Xipeng Qiu, and Zheng Zhang. Plan, verify and switch: Integrated reasoning with diverse X-of-thoughts. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 2807–2822, Singapore, December 2023. Association for Computational Linguistics. 

- [28] Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. In Jong C. Park, Yuki Arase, Baotian Hu, Wei Lu, Derry Wijaya, Ayu Purwarianti, and Adila Alfa Krisnadhi, editors, _Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 305–329, Nusa Dua, Bali, November 2023. Association for Computational Linguistics. 

- [29] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- [30] Ning Miao, Yee Whye Teh, and Tom Rainforth. Selfcheck: Using LLMs to zero-shot check their own step-by-step reasoning. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [31] OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [32] OpenAI. Introducing chatgpt. 2023. 

- [33] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology_ , UIST ’23, New York, NY, USA, 2023. Association for Computing Machinery. 

- [34] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 5687–5711, Singapore, December 2023. Association for Computational Linguistics. 

12 

- [35] Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. _arXiv preprint arXiv:2307.07924_ , 2023. 

- [36] Bilgehan Sel, Ahmad Al-Tawaha, Vanshaj Khattar, Lu Wang, Ruoxi Jia, and Ming Jin. Algorithm of thoughts: Enhancing exploration of ideas in large language models. _arXiv preprint arXiv:2308.10379_ , 2023. 

- [37] Kashun Shum, Shizhe Diao, and Tong Zhang. Automatic prompt augmentation and selection with chain-of-thought from labeled data. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 12113–12139, Singapore, December 2023. Association for Computational Linguistics. 

- [38] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics. 

- [39] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. _arXiv preprint arXiv:2308.11432_ , 2023. 

- [40] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and EePeng Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2609–2634, Toronto, Canada, July 2023. Association for Computational Linguistics. 

- [41] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [42] Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration. _arXiv preprint arXiv:2307.05300_ , 2023. 

- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. 

- [44] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey. _arXiv preprint arXiv:2309.07864_ , 2023. 

- [45] Weijia Xu, Andrzej Banburski-Fahey, and Nebojsa Jojic. Reprompting: Automated chain-ofthought prompt inference through gibbs sampling. _arXiv preprint arXiv:2305.09993_ , 2023. 

- [46] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- [47] Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. _arXiv preprint arXiv:2308.04371_ , 2023. 

- [48] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [49] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [50] Jianpeng Zhou, Wanjun Zhong, Yanlin Wang, and Jiahai Wang. Adaptive-solver framework for dynamic strategy selection in large language model reasoning. _arXiv preprint arXiv:2310.01446_ , 2023. 

13 

## **A Additional Analysis** 

**Examining the optimization process** In our primary experiments, the strategy generator initially produces 30 strategies, and we aim to obtain 10 qualified strategies for validation. For complex datasets, it is difficult to directly acquire 10 qualified strategies without optimization. Table 8 presents the optimization process for three challenging datasets: CP from the mathematical reasoning task, StrategyQA from the commonsense reasoning task, and MA from the algorithmic reasoning task, along with the iteration in which the optimal strategy (i.e., the strategy with the highest validation accuracy) is obtained. It is evident that the strategy optimizer plays a vital role in obtaining more qualified strategies and superior strategies, indicating its significance in our framework. 

Table 8: Total count of qualified strategies achieved in each iteration of the optimization process and the specific iteration when the optimal strategy is attained. 

|**Iteration**|CP|StrategyQA|MA|
|---|---|---|---|
|1|7|9|6|
|2|12|10|9|
|3|-|-|10|
|**Optimal**|2|1|2|



**Analyzing the inference cost** In this section, we assess the inference cost associated with the optimal strategy-based prompt generated by our StrategyLLM framework, as well as the inference costs of the SolutionLLM and CoT baselines. The costs are represented by the average input and output tokens required for each test example, which are displayed in Table 9. Generally, StrategyLLM consumes more tokens during inference compared to CoT and SolutionLLM, as its prompt encompasses both general strategies and step-by-step solutions adhering to these strategies. In this paper, our primary objective is to develop a framework capable of generating generalizable and consistent prompts for various tasks without human intervention. To reduce inference costs, we may encourage LLMs to create more succinct strategies by imposing additional constraints during the prompt generation process, which will be explored in our future work. 

In our main experiments, we utilize 4 examples in the few-shot prompts of datasets within the MATH benchmark. To compare StrategyLLM with baselines of comparable costs, we construct CoT baselines comprising 8 few-shot examples, with average input and output tokens on the datasets of the MATH benchmark amounting to 2697 and 288, respectively. This baseline is referred as CoT-8, and its first 4 examples are the same as CoT in Table 1. The comparison results between CoT-8 and StrategyLLM are presented in Table 10. Our framework achieves a marked improvement over CoT-8, showcasing its effectiveness. 

**Examining the impact of prompt consistency** In our primary experiments, we have demonstrated that StrategyLLM outperforms inconsistent CoT prompts. To further examine the impact of prompt consistency, we intentionally create inconsistent prompts by employing multiple strategies. For each test example within a specific dataset, we generate an inconsistent few-shot prompt for it by randomly selecting examples from different strategy-based prompts. Specifically, we apply the top 3 strategies randomly and uniformly to the examples in the prompt. As a result, the prompt fails to 

Table 9: Average inference cost of each test example across four reasoning tasks. # I and # O denote the number of input and output tokens, respectively. 

||Math|Commonsense|Algorithmic|Symbolic|
|---|---|---|---|---|
|SolutionLLM (# I)|1693|740|373|301|
|SolutionLLM (# O)|373|90|103|48|
|CoT (# I)|1332|487|830|261|
|CoT (# O)|304|50|331|65|
|StrategyLLM (# I)|2649|2139|888|842|
|StrategyLLM (# O)|511|279|227|244|



14 

Table 10: Experimental results on the math reasoning task. The numbers in parentheses represent the relative improvement compared to CoT-8-SC. 

|**Methods**|AL|PA|IA|CP|NT|GE|PC|Avg||
|---|---|---|---|---|---|---|---|---|---|
|CoT-8|56.5|59.5|**20.5**|28.5|27.0|21.0|22.0|33.6||
|CoT-8-SC|61.0|57.5|17.0|38.5|29.0|23.5|18.0|34.9||
|StrategyLLM|58.5|57.5|18.0|35.0|29.5|24.5|22.5|35.1||
|StrategyLLM-SC|60.0|61.5|18.0|38.5|30.5|28.0|**24.0**|37.2|(+6.5%)|
|StrategyLLM-ZS|**64.5**|**65.5**|19.0|**39.0**|**32.5**|**28.5**|22.5|**38.8**|**(+11.1%)**|



Table 11: Results of StrategyLLM using the best discovered strategy and the method employing inconsistent prompts. 

|**Methods**|CP|StrategyQA|MA|Avg|
|---|---|---|---|---|
|Inconsistent Prompt|29.0|56.5|77.3|54.3|
|StrategyLLM|**35.0**|**67.5**|**86.7**|**63.1**|



offer consistent guidance for LLMs, requiring them to choose the most appropriate strategy for each test example based on their understanding of the strategies and the test example itself. We evaluate the effect of prompt consistency on the CP, StrategyQA, and MA datasets. Table 11 presents the comparison between this method and StrategyLLM using the best discovered strategy. It is evident that the approach employing inconsistent prompts performs considerably worse than StrategyLLM, indicating that automatically identifying the most suitable strategy for each test example is quite difficult. Therefore, consistently applying an effective strategy to various examples within the prompt is advantageous. 

**Assessing the complementarity of strategies** To this end, we employ multiple strategies to derive various solutions and ascertain the answer by majority voting. Specifically, we employ the top 1, 3, 5, 7, and 9 strategies for the CP, StrategyQA, and MA datasets. Figure 7 unveils the following observations: (1) StrategyLLM-SC consistently surpasses CoT-SC on all three datasets when employing multiple solutions, suggesting that explicitly introducing effective strategies to obtain solutions is beneficial; (2) Leveraging multiple strategies outperforms the utilization of a single strategy across all three datasets, implying that multiple complementary strategies exist in most scenarios; (3) Incorporating additional strategies generally demonstrates advantageous. However, this does not guarantee enhancement in performance, as it relies on the effectiveness of the newly introduced strategies and their complementarity with pre-existing strategies. 

**Upper limit of accuracy with multiple strategies** The upper limit of utilizing multiple strategies can be determined by calculating the _coverage_ , which is defined as the percentage of examples that can be accurately solved by at least one strategy. The coverage represents the maximum potential accuracy achievable with multiple strategies. Figure 8 illustrates the coverage and accuracy (i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0015-07.png)


**----- Start of picture text -----**<br>
72<br>40 70<br>90<br>68<br>66<br>35 StrategyLLM-SC StrategyLLM-SC StrategyLLM-SC<br>85<br>CoT-SC CoT-SC CoT-SC<br>64<br>1 3 5 7 9 1 3 5 7 9 1 3 5 7 9<br>Number of solutions Number of solutions Number of solutions<br>(a) CP (b) StrategyQA (c) MA<br>**----- End of picture text -----**<br>


Figure 7: Performance of StrategyLLM-SC and CoT-SC on the CP, StrategyQA, and MA datasets. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0016-00.png)


**----- Start of picture text -----**<br>
85<br>100<br>60<br>80<br>95<br>50<br>75<br>90<br>40<br>70<br>Coverage Coverage Coverage<br>85<br>Accuracy Accuracy Accuracy<br>1 3 5 7 9 1 3 5 7 9 1 3 5 7 9<br>Number of solutions Number of solutions Number of solutions<br>(a) CP (b) StrategyQA (c) MA<br>**----- End of picture text -----**<br>


Figure 8: Coverage and accuracy of StrategyLLM using multiple strategies on the CP, StrategyQA, and MA datasets. 

through a majority vote) of the StrategyLLM using multiple strategies on the CP, StrategyQA, and MA datasets. We can observe that there is a significant gap between accuracy and coverage, indicating that substantial performance improvements could be realized if the optimal strategy for each test example is selected rather than relying on a simple majority vote. 

**Case study** In this paragraph, we present a case study to demonstrate the advantages of StrategyLLM over CoT reasoning. We utilize an example with a difficulty level of 5 from the CP dataset, as illustrated in Figure 9, which includes the example and solutions provided by both StrategyLLM and CoT. While CoT offers a flexible reasoning process, it can sometimes lead to inconsistencies or omissions in understanding and solving problems. Specifically, in the provided example, CoT mishandles rotations and reflections and fails to provide adequate details on how to address them. In contrast, StrategyLLM is designed to reduce the likelihood of misinterpretations and enhance reasoning through a structured approach. The generalizable strategy employed by StrategyLLM facilitates a comprehensive understanding of the problem, effective utilization of relevant concepts, and a systematic breakdown of the task. This structured approach ensures that critical details are not overlooked, thereby leading to the correct solution. This case study clearly highlights the superiority of StrategyLLM in handling complex reasoning tasks. Beyond accuracy, the solutions derived from StrategyLLM are more detailed, interpretable, and accessible, making them particularly suitable for contexts that demand clarity and comprehensibility, such as educational settings. 

## **B Implementation Details** 

The strategy generator creates _n_ = 30 strategies using temperature sampling with a temperature of 1. The strategy executor, optimizer, and evaluator employ greedy decoding with a temperature of 0. The threshold of the execution accuracy for all datasets is set to 0.75, except for the CP and GE datasets. Their thresholds are set to 0.5 due to the challenging examples in their few-shot prompts. The maximum iteration number for strategy optimization is 3. We retain _k_ = 10 candidate strategies for validation, selecting the top 1 or 3 strategies with the highest validation accuracy for inference. 

## **C Prompts of Strategy Optimizer** 

For optimizing a strategy, the strategy optimizer conducts the following tasks: (1) constructing the examination result for each task example by comparing the answer extracted from its execution result and the gold answer, which is demonstrated in Figure 10; (2) analyzing the examination results of all provided task examples and providing suggestions for improving the strategy using the prompt in Figure 11; (3) modifying the strategy to obtain an updated strategy based on the feedback, i.e., the analysis and suggestions, leveraging the prompt in Figure 12. 

16 

## **D Prompt of StrategyLLM-ZS** 

The prompt is shown in Figure 13. 

## **E Prompt of SolutionLLM** 

The prompt is presented in Figure 14. 

## **F Prompt of CoT+Strategy** 

The prompt is demonstrated in Figure 15. 

## **G Strategies and Execution Results** 

Figures 16-27 provide the best strategies discovered for various datasets and their execution results on task examples obtained by GPT-3.5. Figures 28-30 present the best strategies discovered for various datasets and their execution results on task examples obtained by GPT-4. 

## **H Chain-of-Thought Prompts for MATH Datasets** 

Figures 31-39 demonstrate the chain-of-thought prompts for the seven datasets from the MATH benchmark. 

17 

## Case Study 

**Question:** How many ways are there to arrange 6 beads of distinct colors in a 2 _×_ 3 grid if reflections and rotations are considered the same? (In other words, two arrangements are considered the same if I can rotate and/or reflect one arrangement to get the other.) **Gold Answer:** 180 

## **CoT Solution:** 

We can consider the 2 _×_ 3 grid as a 6-element cycle. There are 6! ways to arrange the 6 beads in a line, but we must divide by 6 to account for rotations. However, this overcounts the number of arrangements by a factor of 2 because we are also counting reflections. Therefore, 6! the final answer is 6 _·_ 2[= 60][.] **Answer:** The answer is 60. 

## **StrategyLLM Solution:** 

1. Understand the problem statement: - We have 6 beads of distinct colors. 

- We need to arrange these beads in a 2x3 grid. 

- Reflections and rotations are considered the same, meaning two arrangements are considered the same if one can be obtained from the other by rotating or reflecting. 

2. Identify the counting or probability concept: - We need to use the concept of combinations to solve this problem. 3. Break down the problem: 

- We have 6 beads to arrange in a 2x3 grid. 

- We need to consider the symmetries of the grid to eliminate redundant cases. 

- We need to determine the number of distinct arrangements considering reflections and 

- rotations. 

4. Calculate the required counting or probability: - To calculate the number of distinct arrangements, we need to count the number of combinations that satisfy the given conditions. - We can use combinations to calculate this. 

- Let’s consider the possible arrangements: 

- We can choose any of the 6 beads for the first position in the grid. 

- We can choose any of the remaining 5 beads for the second position. - We can choose any of the remaining 4 beads for the third position. 

- We can choose any of the remaining 3 beads for the fourth position. - We can choose any of the remaining 2 beads for the fifth position. 

- We can choose the last remaining bead for the sixth position. - Total number of distinct arrangements = 6 x 5 x 4 x 3 x 2 x 1 = 720. 

5. Simplify or reduce the problem: - We need to consider the symmetries of the grid to eliminate redundant cases. - The 2x3 grid has rotational symmetry of order 2, meaning it can be rotated 180 degrees to look the same. - The 2x3 grid also has reflectional symmetry, meaning it can be reflected horizontally or vertically to look the same. - Therefore, we need to divide the total number of distinct arrangements by 2 to account for the rotational symmetry. - We also need to divide the total number of distinct arrangements by 2 to account for the reflectional symmetry. 6. Calculate the total number of distinct arrangements: - Total number of distinct arrangements = 720 / 2 / 2 = 180. **Answer:** The answer is 180. 

18 Figure 9: Case study. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0019-00.png)


**----- Start of picture text -----**<br>
Strategy Optimizer - Construction of Examination Result<br>Example:<br>{example}<br>Execution result obtained by executing the strategy on the example:<br>{execution result}<br>The answer extracted from the execution result is {predicted answer}, and the correct answer<br>is {gold answer}. They are {different or the same}.<br>**----- End of picture text -----**<br>


Figure 10: Construction of the examination result for a task example. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0019-02.png)


**----- Start of picture text -----**<br>
Strategy Optimizer - Analysis & Suggestion<br>Task:<br>{task definition}<br>Strategy:<br>{strategy}<br>Examination results obtained by executing the strategy on the provided examples of the task<br>and examining the execution results:<br>{examination results}<br>We can see that we do not get the correct answer after executing this strategy on some of the<br>provided examples. Please carefully analyze why the answers extracted from the execution<br>results of these examples are incorrect and provide suggestions for improving the strategy.<br>**----- End of picture text -----**<br>


Figure 11: Analysis & Suggestion Prompt of the strategy optimizer. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0019-04.png)


**----- Start of picture text -----**<br>
Strategy Optimizer - Modification<br>Task:<br>{task definition}<br>Some examples of the task are as follows:<br>{task examples}<br>Original strategy to solve the task:<br>{original strategy}<br>Feedback:<br>{feedback}<br>You need to modify the original strategy based on the feedback to obtain a better strategy.<br>The newly obtained strategy must be a numbered list in the following format:<br>1. First subtask<br>2. Second subtask<br>**----- End of picture text -----**<br>


Figure 12: Modification Prompt of the strategy optimizer. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0020-00.png)


**----- Start of picture text -----**<br>
StrategyLLM-ZS<br>Question:<br>{question}<br>Candidate Solutions:<br>{candidate solutions}<br>Please determine the final answer for the question based on the candidate solutions and their<br>corresponding answers. The output should be in the following format:<br>Final Answer: The final answer, which must include the string ‘The final answer is ’.<br>**----- End of picture text -----**<br>


Figure 13: Prompt of StrategyLLM-ZS. 

SolutionLLM Task: {task definition} Example of the task: {example} Please write a solution to the provided example. The answer obtained from the solution must be the same as the original answer. The result must be in the following format: Question: Question in the provided example Solution: Solution to the question Answer: Answer in the provided example, which must include the string ‘The answer is ’ 

Figure 14: Prompt of SolutionLLM. 

## CoT+Strategy 

{CoT Prompt} 

We show you several task examples of how to generate solutions. Please develop a general strategy that consists of a sequence of subtasks to solve the task and apply the strategy to the following question to obtain its solution and answer. Question: {question} The result must be in the following format: Strategy: Strategy employed to solve the task Question: Provided question Solution: Solution obtained by applying the strategy to the question Answer: Answer to the question, which must include the string ‘The answer is ’ 

Figure 15: Prompt of CoT+Strategy. 

20 

## MATH-Algebra 

## **Strategy** : 

1. Simplify expressions involving radicals by multiplying and combining under a single radical if possible. 

2. Rationalize the denominators of fractions with radicals by multiplying the numerator and denominator by the conjugate of the denominator. 

3. Use the quadratic formula or factoring to find the roots of quadratic equations. 4. Find the coordinates of points of intersection between two curves by solving the system of equations formed by setting the two equations equal to each other. 

5. Find the product or sum of the roots of a quadratic equation by applying Vieta’s formulas. 

## **Executing the strategy on a task example** : 

- **Question:** The quadratic 8 _x_[2] + 12 _x −_ 14 has two real roots. What is the sum of the squares of these roots? Express your answer as a common fraction in lowest terms. **Solution:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0021-08.png)


**----- Start of picture text -----**<br>
1. Simplify expressions involving radicals by multiplying and combining under a single<br>radical if possible.<br>- There are no radicals in the given quadratic equation, so no simplification is needed.<br>2. Rationalize the denominators of fractions with radicals by multiplying the numerator and<br>denominator by the conjugate of the denominator.<br>- There are no fractions with radicals in the given quadratic equation, so no rationalization is<br>needed.<br>3. Use the quadratic formula or factoring to find the roots of quadratic equations.<br>√ b [2] − 4 ac<br>- The quadratic formula is given by: x  = [−][b][±] 2 a , where  ax [2] +  bx  +  c  = 0.<br>- For the given quadratic equation 8 x [2] + 12 x − 14, we have  a  = 8,  b  = 12, and  c  =  − 14.<br>- Plugging these values into the quadratic formula, we get:<br>− 12 ± [√] 12 [2] − 4(8)( − 14)<br>x  =<br>2(8)<br>144+448<br>x  = [−] [12] [±] [√]<br>16<br>√ 592<br>x  = [−] [12] [±]<br>16<br>√ 37<br>x  = [−] [12] [±] [4]<br>16<br>√ 37<br>x  = [−] [3] [±]<br>4<br>√ 37 √ 37<br>- Therefore, the two real roots of the quadratic equation are [−] [3+] 4 and [−] [3] [−] 4 .<br>4. Find the sum of the squares of the roots.<br>- Let  r 1 and  r 2 be the roots of the quadratic equation.<br>- The sum of the squares of the roots is given by  r 1 [2] [+] [ r] 2 [2][.]<br>- Plugging in the values of the roots, we get:<br>2 2<br>− 3+ √ 37 − 3 − √ 37<br>r 1 [2] [+] [ r] 2 [2] [=] 4 + 4<br>� � � �<br>r 1 [2] [+] [ r] 2 [2] [=] [(] [−] [3][+] 4 √ [2] 37) [2] + [(] [−] [3] [−] 4 √ [2] 37) [2]<br>r 1 [2] [+] [ r] 2 [2] [=] [9] [−] [6] √ 1637+37 + [9+6] √ 1637+37<br>r 1 [2] [+] [ r] 2 [2] [=] [2][(][9] 16 [+][37][)]<br>r 1 [2] [+] [ r] 2 [2] [=] [2][(] 16 [46][)]<br>r 1 [2] [+] [ r] 2 [2] [=] 16 [92]<br>r 1 [2] [+] [ r] 2 [2] [=] [23] 4<br>Answer: The answer is [23]<br>4 [.]<br>**----- End of picture text -----**<br>


Figure 16: Strategy generated by GPT-3.5 for MATH-Algebra and its execution on a task example. 

21 

## MATH-Prealgebra 

## **Strategy** : 

1. Understand the problem: Read and carefully analyze the given problem to understand the requirements and constraints. 

2. Identify the known and unknown information: Determine what information is provided and what needs to be calculated or determined. 

3. Translate the problem into mathematical equations or statements: Use appropriate mathematical symbols, equations, or statements to represent the problem. 

4. Solve the equations or statements: Use algebraic techniques to solve the mathematical equations or statements. 

5. Verify the Solution: Substitute the found values back into the original problem to check if they satisfy the given conditions. 

6. Express the solution in the required format: Answer the question or provide the solution in the appropriate form, such as a fraction, decimal, or percentage. 

## **Executing the strategy on a task example** : 

**Question:** Pete thinks of a number. He doubles it, adds 10, multiplies by 4, and ends up with 120. What was his original number? 

## **Solution:** 

1. Understand the problem: Pete starts with a number, performs a series of operations on it, and ends up with a final result. We need to determine the original number. 

2. Identify the known and unknown information: 

Known information: Pete ends up with 120 after doubling the number, adding 10, and multiplying by 4. 

Unknown information: The original number Pete started with. 

3. Translate the problem into mathematical equations or statements: Let x be the original number Pete started with. 

Pete doubles the number: 2x 

Pete adds 10: 2x + 10 Pete multiplies by 4: 4(2x + 10) Pete ends up with 120: 4(2x + 10) = 120 

4. Solve the equations or statements: 

4(2x + 10) = 120 

Distribute the 4: 8x + 40 = 120 Subtract 40 from both sides: 8x = 80 Divide both sides by 8: x = 10 The original number Pete started with is 10. 

5. Verify the Solution: Substitute the found value back into the original problem. Pete doubles 10: 2(10) = 20 Pete adds 10: 20 + 10 = 30 Pete multiplies by 4: 4(30) = 120 The final result is indeed 120, which matches the given information. 6. Express the solution in the required format: **Answer:** The answer is 10. 

Figure 17: Strategy generated by GPT-3.5 for MATH-Prealgebra and its execution on a task example. 

22 

## MATH-Intermediate Algebra 

## **Strategy** : 

1. Begin by identifying the type of problem you are given: quadratic polynomial, algebraic equation, geometric problem, etc. 2. If the problem involves a polynomial, start by determining the degree of the polynomial and then proceed to solve for the unknown coefficients using the given conditions or given equations. 3. If the problem involves algebraic equations, simplify the expressions and solve for the unknowns by isolating the variable. Make use of algebraic rules such as distributive property, solving equations with fractions, and solving systems of equations if necessary. 

4. If the problem involves geometric concepts, analyze the given information and understand the geometric relationships involved. Apply the appropriate formulas or theorems and solve for the unknown quantities. 5. If the problem involves inequalities, solve for the range of possible values by analyzing the conditions or constraints given. Make use of algebraic manipulation and reasoning skills to determine the range or the maximum/minimum values. 6. If the problem involves functions, analyze the properties of the function and use any given information to determine the unknown variables, such as finding the maximum/minimum values, solving for x-intercepts or y-intercepts, or finding asymptotes. 7. Check your solutions by substituting them back into the original equation or problem statement to ensure they satisfy the given conditions. 

## **Executing the strategy on a task example** : 

- **Question:** Find the quadratic polynomial _p_ ( _x_ ) such that _p_ ( _−_ 3) = 10 _, p_ (0) = 1 _,_ and _p_ (2) = 5 _._ . **Solution:** 

1. Identify the type of problem: quadratic polynomial. 2. Determine the degree of the polynomial and solve for the unknown coefficients using the given conditions. Let the quadratic polynomial be _p_ ( _x_ ) = _ax_[2] + _bx_ + _c_ . Using the given conditions: _p_ ( _−_ 3) = 10 _⇒ a_ ( _−_ 3)[2] + _b_ ( _−_ 3) + _c_ = 10 _p_ (0) = 1 _⇒ a_ (0)[2] + _b_ (0) + _c_ = 1 _p_ (2) = 5 _⇒ a_ (2)[2] + _b_ (2) + _c_ = 5 Simplifying the equations: 9 _a −_ 3 _b_ + _c_ = 10 _c_ = 1 4 _a_ + 2 _b_ + _c_ = 5 Substituting _c_ = 1 into the equations: 9 _a −_ 3 _b_ + 1 = 10 4 _a_ + 2 _b_ + 1 = 5 Simplifying further: 9 _a −_ 3 _b_ = 9 4 _a_ + 2 _b_ = 4 3. Solve the system of equations using algebraic manipulation. Multiplying the second equation by 3: 9 _a −_ 3 _b_ = 9 12 _a_ + 6 _b_ = 12 Adding the equations together: 21 _a_ = 21 Dividing both sides by 21: _a_ = 1 Substituting _a_ = 1 into the second equation: 4(1) + 2 _b_ = 4 4 + 2 _b_ = 4 2 _b_ = 0 Dividing both sides by 2: _b_ = 0 Substituting _a_ = 1 and _b_ = 0 into the first equation: 9(1) _−_ 3(0) = 9 9 = 9 Therefore, _a_ = 1, _b_ = 0, and _c_ = 1. 4. Write the quadratic polynomial _p_ ( _x_ ). _p_ ( _x_ ) = 1 _x_[2] + 0 _x_ + 1 Simplifying: _p_ ( _x_ ) = _x_[2] + 1 **Answer:** The answer is _x_[2] + 1. 

Figure 18: Strategy generated by GPT-3.5 for MATH-Intermediate Algebra and its execution on a task example. 23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0024-00.png)


**----- Start of picture text -----**<br>
MATH-Counting and Probability<br>Strategy :<br>1. Understand the problem statement:<br>- Read and understand the given problem.<br>- Identify what needs to be calculated or determined.<br>- Make note of any special conditions or restrictions mentioned.<br>2. Identify the counting or probability concept:<br>- Determine which counting or probability concept is needed to solve the problem.<br>- Common concepts include permutations, combinations, probability rules, etc.<br>3. Break down the problem:<br>- Break down the problem into smaller subproblems or cases if needed.<br>- Identify any patterns or symmetries that can simplify the calculations.<br>- Consider any specific conditions or restrictions mentioned in the problem.<br>4. Calculate the required counting or probability:<br>- Apply the appropriate counting or probability formula or rule to calculate the desired quantity.<br>- Use combinations or permutations for counting problems.<br>- Apply probability rules like the multiplication rule, addition rule, conditional probability, etc., for<br>probability problems.<br>5. Simplify or reduce the problem:<br>- Look for ways to simplify or reduce the problem.<br>- Utilize symmetry or equivalence to eliminate redundant cases.<br>- Consider any combinatorial identities or properties that can simplify calculations.<br>6. Verify and check the Solution:<br>- Double-check the calculations to ensure accuracy.<br>- Check if the solution satisfies any specific conditions or restrictions mentioned in the problem.<br>- Compare the solution with any provided answers or results.<br>7. Communicate the solution clearly:<br>- Present the final solution in a clear and concise manner.<br>- Include any necessary explanations or steps taken to arrive at the solution.<br>- Express the final answer in the required units or as a fraction, decimal, or whole number, as specified<br>by the problem.<br>8. Practice and review:<br>- Solve additional problems on counting and probability to reinforce the concepts.<br>- Review the solved problems to identify any common patterns or strategies that can be applied to<br>similar problems.<br>Executing the strategy on a task example :<br>Question: Given that �177 � = 19448, �178 � = 24310 and �179 � = 24310, calculate �199 �.<br>Solution:<br>1. Understand the problem statement:<br>17 17 17<br>- We are given three values of binomial coefficients: � 7 � = 19448, � 8 � = 24310, and � 9 � = 24310.<br>19<br>- We need to calculate � 9 �.<br>2. Identify the counting or probability concept:<br>- We need to use the concept of combinations to solve this problem.<br>- We can use the identity3. Break down the problem:� nk � = � nk−− 11� + � n−k 1� to break down the problem.<br>- Using this identity, we can express �199 � in terms of the given values: �199 � = �188 � + �189 �.<br>4. Calculate the required counting or probability:<br>- Using the given values, we can calculate �188 � and �189 �.<br>- �188 � = �177 � + �178 � = 19448 + 24310 = 43758<br>- �189 � = �178 � + �179 � = 24310 + 24310 = 48620<br>5. Simplify or reduce the problem:<br>- We have calculated the values of �188 � and �189 �.<br>- We can now substitute these values into the expression for �199 �: �199 � = �188 � + �189 � =<br>43758 + 48620 = 92378<br>6. Verify and check the Solution:<br>- The solution obtained is 92378, which matches the provided answer.<br>7. Communicate the solution clearly:<br>- The value of �199 � is 92378.<br>Answer: The answer is 92378.<br>**----- End of picture text -----**<br>


Figure 19: Strategy generated by GPT-3.5 for MATH-Counting and Probability and its execution on a task example. 

24 

## MATH-Number Theory 

## **Strategy** : 

1. Understand the problem: Read the problem carefully and identify what is being asked and what information is given. 2. Apply basic number theory concepts: Use basic number theory principles such as modular arithmetic, divisibility rules, prime factorization, etc., to simplify the problem or find relevant information. 3. Simplify expressions: Simplify any given expressions or equations to make them easier to work with. 4. Solve congruences: If the problem involves solving congruences (e.g., finding the remainder when a number is divided by another number), use the Chinese Remainder Theorem or specific methods for solving modular equations to find the solution. 

5. Study patterns and properties: Look for patterns or properties in the given problem that might help in finding the solution. This could involve using modular arithmetic properties, Euler’s totient function, or other relevant principles. 6. Use prime factorization: If the problem involves finding factors or multiples of numbers, use the prime factorization of the numbers involved to simplify the calculations. 

7. Apply number theory theorems: If the problem relates to specific number theory theorems, such as Fermat’s Little Theorem or Wilson’s Theorem, apply those theorems to find the solution. 

8. Check for divisibility: Use divisibility rules to check if a number is divisible by another number, or find other relevant information that can help in solving the problem. 9. Solve step-by-step: Break down the problem into smaller steps and solve them one by one to find the final solution. 

10. Check the Answer: Always double-check your final answer and ensure that it satisfies the given conditions or requirements of the problem. 

## **Executing the strategy on a task example** : 

**Question:** What is the remainder when 2007 is divided by 81? **Solution:** 

1. Understand the problem: We are asked to find the remainder when 2007 is divided by 81. 2. Apply basic number theory concepts: We can use the concept of modular arithmetic to solve this problem. In modular arithmetic, we find the remainder when a number is divided by another number. 

3. Simplify expressions: There are no expressions to simplify in this problem. 

4. Solve congruences: To find the remainder when 2007 is divided by 81, we can write it as a congruence: 2007 _≡ x_ (mod 81), where _x_ is the remainder we want to find. 5. Study patterns and properties: There are no specific patterns or properties to consider in this problem. 6. Use prime factorization: We can use the prime factorization of 81 to simplify the calculations. 81 = 3[4] . 7. Apply number theory theorems: There are no specific number theory theorems to apply in this problem. 

8. Check for divisibility: We can check if 2007 is divisible by 81. If it is divisible, then the remainder will be 0. Otherwise, we need to find the remainder. 9. Solve step-by-step: a. Divide 2007 by 81: 2007 _÷_ 81 = 24 with a remainder of 63. b. The remainder is 63. 

10. Check the Answer: The remainder we found, 63, is the same as the original answer. **Answer:** The answer is 63. 

Figure 20: Strategy generated by GPT-3.5 for MATH-Number Theory and its execution on a task example. 

25 

## MATH-Geometry 

## **Strategy** : 

1. To find the area of a shaded region in a circle sector overlap problem: a. Calculate the area of each sector using the formula _A_ sector =[1] 2 _[r]_[2] _[θ]_[, where] _[ r]_[ is the radius and] _[ θ]_[ is the] central angle. b. Determine the area of the overlapping region by subtracting the area of the triangle formed by the two radii from the sum of the areas of the two sectors. c. Double-check the calculations and verify the obtained answer against the correct answer. 2. To calculate the perimeter of a quadrilateral formed by cutting a square with a line: a. Find the coordinates of the intersection points between the line and the square. b. Calculate the lengths of the sides of the quadrilateral using the distance formula. c. Add up the lengths of the sides to find the perimeter. d. Double-check the calculations and verify the obtained answer against the correct answer. 3. To determine the height of a cylinder inscribed in a hemisphere: a. Use the Pythagorean theorem to find the radius of the cylinder’s base by subtracting the height of the cylinder from the radius of the hemisphere. b. The height of the cylinder is equal to the radius of the hemisphere minus the radius of the cylinder’s base. c. Double-check the calculations and verify the obtained answer against the correct answer. 4. To reflect a parallelogram across the x-axis: a. Change the sign of the y-coordinate of each vertex to obtain the reflected image of the parallelogram. b. Double-check the coordinates of the reflected vertices and verify against the correct answer. 5. To reflect a parallelogram across a given line: a. Find the equation of the given line. b. Calculate the distance between each vertex of the parallelogram and the given line. c. Reflect each vertex across the given line by moving it the same distance on the opposite side of the line. 

- d. Connect the reflected vertices to form the reflected parallelogram. e. Double-check the coordinates of the reflected vertices and verify against the correct answer. 6. To find the ordered pair of a reflected point: a. Determine the equation of the line of reflection. 

- b. Use the formula for the reflection of a point across a line to find the coordinates of the reflected point. c. Double-check the coordinates of the reflected point and verify against the correct answer. 7. Double-check all calculations and results to ensure accuracy. a. Verify the obtained answers against the correct answers. 

- b. Review all steps and calculations to identify any errors or discrepancies. 

- c. Make necessary corrections and adjustments to improve the accuracy of the solution. 

## **Executing the strategy on a task example** : 

- **Question:** Parallelogram _ABCD_ with _A_ (2 _,_ 5), _B_ (4 _,_ 9), _C_ (6 _,_ 5), and _D_ (4 _,_ 1) is reflected across the _x_ -axis to _A[′] B[′] C[′] D[′]_ and then _A[′] B[′] C[′] D[′]_ is reflected across the line _y_ = _x_ + 1 to _A[′′] B[′′] C[′′] D[′′]_ . This is done such that _D[′]_ is the image of _D_ , and _D[′′]_ is the image of _D[′]_ . What is the ordered pair of _D[′′]_ in the coordinate plane? 

## **Solution:** 

1. Reflect parallelogram _ABCD_ across the x-axis to obtain parallelogram _A[′] B[′] C[′] D[′]_ : a. Change the sign of the y-coordinate of each vertex: - _A[′]_ (2 _, −_ 5) - _B[′]_ (4 _, −_ 9) - _C[′]_ (6 _, −_ 5) - _D[′]_ (4 _, −_ 1) 2. Reflect parallelogram _A[′] B[′] C[′] D[′]_ across the line _y_ = _x_ + 1 to obtain parallelogram _A[′′] B[′′] C[′′] D[′′]_ : a. Find the equation of the line _y_ = _x_ + 1: - The slope of the line is 1, and the y-intercept is 1. - The equation of the line is _y_ = _x_ + 1. 

- b. Calculate the distance between each vertex of parallelogram _A[′] B[′] C[′] D[′]_ and the line _y_ = _x_ + 1: - Distance from _A[′]_ to the line _y_ = _x_ + 1: - Substitute the coordinates of _A[′]_ into the equation of the line: - _−_ 5 = 2 + 1 - _−_ 5 = 3 - The distance is _| −_ 5 _−_ 3 _|_ = 8. 

- Distance from _B[′]_ to the line _y_ = _x_ + 1: - Substitute the coordinates of _B[′]_ into the equation of the line: - _−_ 9 = 4 + 1 - _−_ 9 = 5 - The distance is _| −_ 9 _−_ 5 _|_ = 14. - Distance from _C[′]_ to the line _y_ = _x_ + 1: - Substitute the coordinates of _C[′]_ into the equation of the line: - _−_ 5 = 6 + 1 - _−_ 5 = 7 - The distance is _| −_ 5 _−_ 7 _|_ = 12. - Distance from _D[′]_ to the line _y_ = _x_ + 1: - Substitute the coordinates of _D[′]_ into the equation of the line: - _−_ 1 = 4 + 1 - _−_ 1 = 5 - The distance is _| −_ 1 _−_ 5 _|_ = 6. c. Reflect each vertex of parallelogram _A[′] B[′] C[′] D[′]_ across the line _y_ = _x_ + 1 by moving it the same distance on the opposite side of the line: - Reflect _A[′]_ across the line _y_ = _x_ + 1: - Move _A[′]_ 8 units in the opposite direction of the line’s normal vector: - _A[′′]_ (2 _−_ 8 _, −_ 5 + 8) = ( _−_ 6 _,_ 3) 

- Reflect _B[′]_ across the line _y_ = _x_ + 1: - Move _B[′]_ 14 units in the opposite direction of the line’s normal vector: - _B[′′]_ (4 _−_ 14 _, −_ 9 + 14) = ( _−_ 10 _,_ 5) - Reflect _C[′]_ across the line _y_ = _x_ + 1: - Move _C[′]_ 12 units in the opposite direction of the line’s normal vector: - _C[′′]_ (6 _−_ 12 _, −_ 5 + 12) = ( _−_ 6 _,_ 7) 

- Reflect _D[′]_ across the line _y_ = _x_ + 1: - Move _D[′]_ 6 units in the opposite direction of the line’s normal vector: - _D[′′]_ (4 _−_ 6 _, −_ 1 + 6) = ( _−_ 2 _,_ 5) 26 

3. The ordered pair of _D[′′]_ is ( _−_ 2 _,_ 5). **Answer:** The answer is ( _−_ 2 _,_ 5). 

## MATH-Precalculus 

## **Strategy** : 

1. Identify the problem or concept being tested in the given question. 

2. Define any necessary terms or formulas related to the problem. 

3. Simplify or manipulate any given expressions or equations to solve the problem. 

4. Solve the problem step by step with the help of appropriate formulas or techniques. 

5. Verify the solution or answer to ensure its correctness. 

6. Repeat steps 1-5 for each problem in the precalculus subject. 

## **Executing the strategy on a task example** : 

**Question:** The function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0027-10.png)


represents a rotation around some complex number _c_ . Find _c_ . 

## **Solution:** 

1. Identify the problem or concept being tested in the given question. 

- The problem is asking us to find the complex number _c_ that represents the center of rotation for the given function. 

2. Define any necessary terms or formulas related to the problem. 

- The formula for a rotation around a complex number _c_ is given by _f_ ( _z_ ) = _e[iθ]_ ( _z − c_ ) + _c_ , where _θ_ is the angle of rotation. 

3. Simplify or manipulate any given expressions or equations to solve the problem. ~~_√_~~ 3 _z−_ 2 ~~_√_~~ 3 _−_ 18 _i_ - The given function can be rewritten as _f_ ( _z_ ) = _[−][z]_[+] _[i]_ 2 . 4. Solve the problem step by step with the help of appropriate formulas or techniques. - Comparing the given function with the formula for rotation, we can see that _e[iθ]_ = _[−]_[1+] 2 _[i]_ ~~_√_~~ 3 ~~_√_~~ 3 _−_ 18 _i −_ 1+ _i_ ~~_√_~~ 3 and _c_ = _[−]_[2] 2 . - To find _θ_ , we can take the argument of _e[iθ]_ , which is _θ_ = arg 2 . � � - Simplifying the expression inside the argument, we get _[−]_[1+] 2 _[i]_ ~~_√_~~ 3 = _[−]_[2+2] 4 _[i]_ ~~_√_~~ 3 = _[−]_ 2[1][+] _[i]_ ~~_√_~~ 2 3[.] - The argument of this complex number is _θ_ = arctan ~~_√_~~ _−_ 13 = arctan( _−√_ 3). - Using the � � unit circle, we can determine that _θ_ = _−[π]_ 3[.][- Therefore, the angle of rotation is] _[ −][π]_ 3[.] 5. Verify the solution or answer to ensure its correctness. - Substituting _θ_ = _−[π]_ 3[into][the][formula][for][rotation,][we][get] _[f]_[(] _[z]_[)][=] _[e][−][i][π]_ 3 ( _z − c_ ) + _c_ . - Simplifying this expression, we have _f_ ( _z_ ) = 21 �cos � _−[π]_ 3 � + _i_ sin � _−[π]_ 3 �� ( _z − c_ ) + _c_ . - Using the trigonometric identities cos( _−[π]_ 3[)][=] 12[and][sin(] _[−][π]_ 3[)][=] _[−]_ ~~_√_~~ 23[,][we][can][further] simplify the expression to _f_ ( _z_ ) =[1] 2 12 _[−]_ ~~_√_~~ 23 _[i]_ ( _z − c_ ) + _c_ . - Distributing and simplifying, � � we get _f_ ( _z_ ) = 14[(] _[z][−][c]_[)] _[−]_ ~~_√_~~ 43 _[i]_[(] _[z][−][c]_[)][+] _[ c]_[.][-][Combining][like][terms,][we][have] _[f]_[(] _[z]_[)][=] 1 ~~_√_~~ 3 3 ~~_√_~~ 3 4 _[−]_ 4 _[i] z_ + 4[+] 4 _[i] c_ . - Comparing this expression with the given function, we can � � � � see that[1] 4 _[−]_ ~~_√_~~ 43 _[i]_[ =] _[−]_[1+] 2 _[i]_ ~~_√_~~ 3 and[3] 4[+] ~~_√_~~ 43 _[i]_[ =] _[−]_[2] ~~_√_~~ 32 _−_ 18 _i_ . - Therefore, our solution is correct. **Answer:** The answer is _√_ 3 _−_ 5 _i_ . 

Figure 22: Strategy generated by GPT-3.5 for MATH-Precalculus and its execution on a task example. 

27 

## StrategyQA 

## **Strategy** : 

1. Read and understand the question fully by carefully analyzing the given information and requirements. 2. Identify the type of question being asked, whether it is a simple "yes or no" question or if there is a need to provide a specific answer. 3. Determine the key elements and concepts mentioned in the question that need to be addressed. 4. Break down the question into smaller parts, if necessary, to identify any hidden information or assumptions. 5. Look for any clues or hints within the question that can help in finding the answer. 6. Decide on the approach to be used, whether it requires logical reasoning, knowledge of specific facts, comparison, or analysis. 7. Use prior knowledge and general understanding to answer the question, if applicable. 8. If additional information is required, either use external sources or gather relevant data from memory. 9. Organize the information and answer the question clearly and concisely. 10. Review the answer to ensure it accurately addresses the question and provides a correct response. 11. If necessary, double-check the answer to ensure it aligns with the requirements and request. 12. Finally, present the answer in a clear and understandable manner, either as a "yes or no" response or as a specific answer as required by the question. 

## **Executing the strategy on a task example** : 

## **Question:** Could Brooke Shields succeed at University of Pennsylvania? **Solution:** 

1. Read and understand the question fully by carefully analyzing the given information and requirements. - The question is asking if Brooke Shields could succeed at the University of Pennsylvania. 2. Identify the type of question being asked, whether it is a simple "yes or no" question or if there is a need to provide a specific answer. 

- The question requires a simple "yes or no" answer. 3. Determine the key elements and concepts mentioned in the question that need to be addressed. - Key elements: Brooke Shields, University of Pennsylvania, succeed. 4. Break down the question into smaller parts, if necessary, to identify any hidden information or assumptions. - No need to break down the question further. 5. Look for any clues or hints within the question that can help in finding the answer. - The question does not provide any specific clues or hints. 6. Decide on the approach to be used, whether it requires logical reasoning, knowledge of specific facts, comparison, or analysis. - The approach requires logical reasoning and knowledge of the University of Pennsylvania. 7. Use prior knowledge and general understanding to answer the question, if applicable. - The University of Pennsylvania is a prestigious university known for its rigorous academic programs. Brooke Shields, as a successful actress and model, may have the potential to succeed at the university. 8. If additional information is required, either use external sources or gather relevant data from memory. - No additional information is required. 9. Organize the information and answer the question clearly and concisely. - The answer is yes, Brooke Shields could succeed at the University of Pennsylvania. 10. Review the answer to ensure it accurately addresses the question and provides a correct response. - The answer accurately addresses the question and provides a correct response. 11. If necessary, double-check the answer to ensure it aligns with the requirements and request. - The answer aligns with the requirements and request. 12. Finally, present the answer in a clear and understandable manner, either as a "yes or no" response or as a specific answer as required by the question. **Answer:** The answer is yes. 

Figure 23: Strategy generated by GPT-3.5 for StrategyQA and its execution on a task example. 

28 

Date Understanding **Strategy** : 1. Determine the current date based on the given information. 2. Calculate the number of hours or days given in the question. 3. Determine if the time period given is in the past or future. 4. Adjust the current date based on the time period and direction (past or future). 5. Convert the adjusted date to the required format (MM/DD/YYYY) as specified in the question. 6. Provide the final date as the answer to the question. **Executing the strategy on a task example** : **Question:** The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY? **Solution:** 1. Determine the current date based on the given information. - The concert was delayed by one day, so the current date is 06/02/1943. 2. Calculate the number of days given in the question. - The question asks for the date 10 days ago. 3. Determine if the time period given is in the past or future. - The time period is in the past. 4. Adjust the current date based on the time period and direction (past or future). - Subtract 10 days from the current date: 06/02/1943 - 10 days = 05/23/1943. 5. Convert the adjusted date to the required format (MM/DD/YYYY) as specified in the question. - The adjusted date is already in the required format: 05/23/1943. 6. Provide the final date as the answer to the question. **Answer:** The answer is 05/23/1943. 

Figure 24: Strategy generated by GPT-3.5 for Date Understanding and its execution on a task example. 

Word Sorting **Strategy** : 1. Identify the list of words that needs to be sorted. 2. Split the list of words into individual words. 3. Sort the individual words alphabetically. 4. Combine the sorted words back into a single list. 5. Present the sorted list of words as the answer. **Executing the strategy on a task example** : **Question:** Sort the following words alphabetically: List: oven costume counterpart **Solution:** 1. Identify the list of words that needs to be sorted. - The list of words is "oven costume counterpart". 2. Split the list of words into individual words. - The individual words are "oven", "costume", and "counterpart". 3. Sort the individual words alphabetically. - The sorted words are "costume", "counterpart", and "oven". 4. Combine the sorted words back into a single list. - The combined sorted list is "costume counterpart oven". 5. Present the sorted list of words as the answer. **Answer:** The answer is costume counterpart oven. 

Figure 25: Strategy generated by GPT-3.5 for Word Sorting and its execution on a task example. 

29 

## Multi-step Arithmetic 

- **Strategy** : 1. Identify the innermost operations enclosed in parentheses. 2. Evaluate the innermost operations first. 3. Replace the evaluated expression with the result. 4. Repeat steps 1-3 until all parentheses are removed. 5. Evaluate the remaining arithmetic operations according to the order of operations (multiplication and division before addition and subtraction). 6. Evaluate all multiplication and division operations from left to right. 7. Replace the evaluated expression with the result. 8. Repeat steps 5-7 until there are no more multiplication and division operations to be performed. 9. Evaluate all addition and subtraction operations from left to right. 10. Replace the evaluated expression with the result. 11. Repeat steps 9-10 until there are no more addition and subtraction operations to be performed. 12. The final result is the solution to the multi-step arithmetic problem. **Executing the strategy on a task example** : **Question:** ((-9 * 7 * 7 * -9) + (4 * -9 - 8 - -4)) = **Solution:** 

Step 1: Identify the innermost operations enclosed in parentheses. The innermost operation is (4 * -9 - 8 - -4). Step 2: Evaluate the innermost operations first. 4 * -9 = -36 -36 - 8 = -44 -44 - -4 = -40 Step 3: Replace the evaluated expression with the result. (((-9 * 7 * 7 * -9) + -40) Step 4: Repeat steps 1-3 until all parentheses are removed. The innermost operation is (-9 * 7 * 7 * -9). Step 5: Evaluate the innermost operations first. -9 * 7 = -63 -63 * 7 = -441 -441 * -9 = 3969 Step 6: Replace the evaluated expression with the result. (3969 + -40) Step 7: Repeat steps 5-6 until there are no more multiplication and division operations to be performed. There are no more multiplication and division operations to be performed. Step 8: Evaluate all addition and subtraction operations from left to right. 3969 + -40 = 3929 Step 9: Replace the evaluated expression with the result. **Answer:** The answer is 3929. 

Figure 26: Strategy generated by GPT-3.5 for Multi-step Arithmetic and its execution on a task example. 

30 

Last Letter Concatenation **Strategy** : 1. Identify the given input, which will be a string containing multiple words separated by spaces. 2. Split the input string into individual words. 3. Initialize an empty string to store the last letters of the words. 4. Iterate through each word in the list of words. 5. Get the last letter of each word. 6. Concatenate the last letter to the string from step 3. 7. Continue steps 4-6 until all words have been processed. 8. Print or return the final concatenated string. **Executing the strategy on a task example** : **Question:** Take the last letters of the words in "Sergey Brin" and concatenate them. **Solution:** 1. Identify the given input: "Sergey Brin" 2. Split the input string into individual words: ["Sergey", "Brin"] 3. Initialize an empty string: last_letters = "" 4. Iterate through each word in the list of words: - For the word "Sergey": - Get the last letter: last_letter = "y" - Concatenate the last letter to the string: last_letters = "y" - For the word "Brin": - Get the last letter: last_letter = "n" - Concatenate the last letter to the string: last_letters = "yn" 5. Print or return the final concatenated string. **Answer:** The answer is yn. 

Figure 27: Strategy generated by GPT-3.5 for Last Letter Concatenation and its execution on a task example. 

31 

## MATH-Counting and Probability 

## **Strategy** : 

1. Identify the type of the counting or probability problem presented. 

2. If the problem is regarding permutations or combinations, identify all possible outcomes and arrange or select the required number of objects accordingly. 

3. If the problem is about probability, assess the sample space, the number of possible outcomes and the number of favorable outcomes. 

4. For probability questions, use the formula of probability, which is the number of favorable outcomes divided by the number of total outcomes. 

5. If the problem involves conditions or restrictions, ensure to factor in these when calculating probability or counting the possible arrangements or selections. 

6. If the problem involves complex calculations, break them down into smaller manageable calculations. 

7. Apply the suitable rule or formula according to the type of counting or probability problem, such as multiplication rule, addition rule, arrangement rules, rules of permutations and combinations, or probability rules. 

8. Compute the result using arithmetic calculation or mathematical formulas, and write down the answer. 

9. Review the process and the answer to ensure accuracy. 

**Executing the strategy on a task example** : **Question:** Given that �177 � = 19448, �178 � = 24310 and �179 � = 24310, calculate �199 �. **Solution:** 1. The problem is about combinations, which is a type of counting problem. 2. The problem is asking for the number of ways to choose 9 items from a set of 19, which is 19 represented by � 9 �. 17 17 17 19 3. We are given the values of � 7 �, � 8 �, and � 9 �, which can be used to calculate � 9 �. 4. The formula for combinations is � _nr_ � = _r_ !( _nn−_ ! _r_ )![, where n is the total number of items, r is] the number of items to choose, and ’!’ denotes factorial. _n n_ 5. However, we can use the property of combinations that � _r_ � = � _n−r_ �, and the relationship between � _n_ +1 _r_ � and � _r−n_ 1� and � _nr_ �, which is � _n_ +1 _r_ � = � _r−n_ 1� + � _nr_ �. 19 18 18 6. Using these properties, we can express � 9 � as � 8 � + � 9 �. 7. Similarly, �188 � can be expressed as �177 � + �178 �, and �189 � can be expressed as �178 � + �179 �. 19 8. Substituting the given values, we get � 9 � = (19448 + 24310) + (24310 + 24310). 19 9. Performing the arithmetic calculation, we get � 9 � = 43758 + 48620 = 92378. **Answer:** The answer is 92378. 

1. The problem is about combinations, which is a type of counting problem. 

Figure 28: Strategy generated by GPT-4 for MATH-Counting and Probability and its execution on a task example. 

32 

## StrategyQA 

## **Strategy** : 

1. Identify the main subject(s) or concept(s) in the question. 

2. Identify the relationship(s) or action(s) associated with the main subject(s) or concept(s). 3. Break down the question into sub-questions based on the identified subjects or concepts and their associated relationships or actions, ensuring each sub-question leads to additional information required to answer the main question. 

4. For each sub-question, infer or retrieve relevant information. This can come from provided information, general knowledge, or external resources. 

5. Compare and synthesize the information gathered from each sub-question. 

6. Based on the synthesis, determine if the information satisfies or contradicts the main question. 

7. Formulate a ’yes’ or ’no’ answer to the main question based on the evaluation. 8. Confirm that the answer accurately addresses the main question. If not, revisit the sub-questions and retrieval process. 

9. Record the answer. 

## **Executing the strategy on a task example** : 

- **Question:** Could Brooke Shields succeed at University of Pennsylvania? **Solution:** 

1. The main subject in the question is Brooke Shields and the main concept is University of Pennsylvania. 

2. The relationship associated with the main subject and concept is Brooke Shields’ potential success at the University of Pennsylvania. 3. Sub-questions based on the identified subjects or concepts and their associated relationships or actions could be: 

- Who is Brooke Shields? 

- What is the University of Pennsylvania? 

- What does it mean to succeed at the University of Pennsylvania? 4. For each sub-Question: 

- Brooke Shields is a well-known American actress and model. 

- The University of Pennsylvania is an Ivy League research university in Philadelphia. It is one of the most prestigious universities in the world. - To succeed at the University of Pennsylvania would likely mean to graduate with a degree, possibly with honors or high academic achievement. 5. Comparing and synthesizing the information gathered from each sub-question, we know that Brooke Shields is a successful actress and model, and the University of Pennsylvania is a prestigious university. Success at the university would likely require dedication, intelligence, and hard work. 

6. Based on this synthesis, it is possible that Brooke Shields could succeed at the University of Pennsylvania, given her demonstrated success in her career. However, without specific information about her academic abilities or aspirations, we cannot definitively say. 7. Given the information available, we can infer a ’yes’ answer to the main question. 8. The answer ’yes’ accurately addresses the main question, given the information available. 9. The answer is recorded. **Answer:** The answer is yes. 

Figure 29: Strategy generated by GPT-4 for StrategyQA and its execution on a task example. 

33 

## Multi-step Arithmetic 

## **Strategy** : 

1. Begin by identifying all the numbers and operations in the given problem. 2. Apply the mathematical rules of order operations. The correct order of operations is Parentheses, Exponents, Multiplication and Division (from left to right), Addition and Subtraction (from left to right). This rule is often abbreviated as PEMDAS. 3. Start solving the problem from the inside of the parentheses first, keeping PEMDAS rule in mind. 4. While doing multiplication, division, addition, and subtraction (in the order), make sure to change the sign of the number if there is a negative sign before it. 5. After solving the operations inside the parentheses, proceed solving the operations outside the parentheses. Remember to still follow the rules and order of operations. 6. Continue to calculate step by step until you reach the final answer. **Executing the strategy on a task example** : **Question:** ((-9 * 7 * 7 * -9) + (4 * -9 - 8 - -4)) = **Solution:** 1. Identify all the numbers and operations in the given problem. The numbers are -9, 7, -9, 4, -9, 8, -4. The operations are multiplication (*), addition (+), and subtraction (-). 2. Apply the mathematical rules of order operations (PEMDAS). 3. Start solving the problem from the inside of the parentheses first. - The first parentheses is (-9 * 7 * 7 * -9). According to the order of operations, we should do multiplication first. So, -9 * 7 = -63, -63 * 7 = -441, -441 * -9 = 3969. - The second parentheses is (4 * -9 - 8 - -4). According to the order of operations, we should do multiplication first and then subtraction. So, 4 * -9 = -36, -36 - 8 = -44, -44 - -4 = -40. 4. After solving the operations inside the parentheses, proceed solving the operations outside the parentheses. So, 3969 + -40 = 3929. 5. Continue to calculate step by step until you reach the final answer. **Answer:** The answer is 3929. 

Figure 30: Strategy generated by GPT-4 for Multi-step Arithmetic and its execution on a task example. 

34 

## CoT Prompt for MATH-Algebra 

**Question:** Calculate ~~_√_~~ 75 _x ·_ ~~_√_~~ 2 _x ·_ ~~_√_~~ 14 _x_ . Express your answer in simplest radical form in terms of _x_ . **Solution:** Writing everything in terms of prime factorizations, the given expression is ~~_√_~~ 3 _·_ 5[2] _·_ 2 _·_ 2 _·_ 7 _· x_[3] = ~~�~~ (2[2] _·_ 5[2] _· x_[2] ) _·_ (3 _·_ 7 _· x_ ) = 10 _x_ ~~_√_~~ 21 _x_ . **Answer:** The answer is 10 _x_ ~~_√_~~ 21 _x_ 

**Question:** Rationalize the denominator of ~~_√_~~ 16 _√−_ 32 ~~_√_~~ 2[.][The answer can be written as] _[A]_ ~~_√_~~ _DB_ + _C_ , where _A_ , _B_ , _C_ , and _D_ are integers, _D_ is positive, and _B_ is not divisible by the square of any prime. Find the minimum possible value of _A_ + _B_ + _C_ + _D_ . **Solution:** We can write the numerator as 4 ~~_√_~~ 2, the denominator as 4 _−_ ~~_√_~~ 2. Then, we multiply the numerator and denominator by the conjugate of the denominator. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0035-03.png)


Therefore, _A_ + _B_ + _C_ + _D_ = 8 + 2 + 4 + 7 = 21. **Answer:** The answer is 21 

**Question:** The quadratic 8 _x_[2] + 12 _x −_ 14 has two real roots. What is the sum of the squares of these roots? Express your answer as a common fraction in lowest terms. 

**Solution:** Let _x_ 1 and _x_ 2 be the roots of the equation 8 _x_[2] + 12 _x −_ 14. We want to find _x_[2] 1[+] _[ x]_[2] 2[.][Note] that _x_[2] 1[+] _[ x]_[2] 2[=][(] _[x]_ 1[+] _[ x]_ 2[)][2] _[−]_[2] _[x]_ 1 _[x]_ 2[.][We know that] _[ x]_ 1[+] _[ x]_ 2[, the sum of the roots, is equal to] _[−] a[b]_[,] which for this equation is _[−]_ 8[12] = _[−]_ 2[3][.][Likewise, we know that] _[ x]_[1] _[x]_[2][, the product of the roots, is equal] to _a[c]_[, which for this equation is] _[−]_ 8[14] = _[−]_ 4[7][.][Thus,] _[ x]_[2] 1[+] _[ x]_[2] 2[=] � _−_ 23 �2 _−_ 2 � _−_ 47 � =[9] 4[+][14] 4[=][23] 4[.] **Answer:** The answer is[23] 4 

**Question:** What is the product of all the coordinates of all the points of intersection of the two circles defined by _x_[2] _−_ 2 _x_ + _y_[2] _−_ 10 _y_ + 25 = 0 and _x_[2] _−_ 8 _x_ + _y_[2] _−_ 10 _y_ + 37 = 0? **Solution:** Add ( _−_ 2 _/_ 2)[2] and ( _−_ 10 _/_ 2)[2] to the first equation and ( _−_ 8 _/_ 2)[2] and ( _−_ 10 _/_ 2)[2] to the second equation to find that the given equations are equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0035-08.png)


which are equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0035-10.png)


respectively. Hence, the two circles have centers (1 _,_ 5) and (4 _,_ 5) respectively and radii 1 and 2 respectively. Since the centers of the circles are 3 units apart and the sum of their radii is 3, the two circles intersect at only one point. We can see that (2 _,_ 5) is the desired intersection point, so our product is 2 _·_ 5 = 10. 

**Answer:** The answer is 10 

Figure 31: Chain-of-thought (CoT) prompt for MATH-Algebra. 

35 

## CoT Prompt for MATH-Prealgebra 

**Question:** Pete thinks of a number. He doubles it, adds 10, multiplies by 4, and ends up with 120. What was his original number? 

**Solution:** Let Pete’s original number be _x_ . If he double it and adds 10, he will have 2 _x_ + 10 _._ After multiplying by 4, Pete ended up with 120. From this information, we have the equation: 4(2 _x_ + 10) = 120 _._ Expanding the left side and solving, we find: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0036-03.png)


**Answer:** The answer is 10 

**Question:** A standard deck of 52 cards has 13 ranks (Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King) and 4 suits ( _♠_ , _♡_ , _♢_ , and _♣_ ), such that there is exactly one card for any given rank and suit. Two of the suits ( _♠_ and _♣_ ) are black and the other two suits ( _♡_ and _♢_ ) are red. The deck is randomly arranged. What is the probability that the top card is a _♡_ ? **Solution:** There are 13 _♡_ ’s and 52 cards total, so the probability that the top card is a _♡_ is 13[1] 52[=] 4[.] 

**Answer:** The answer is[1] 4 

**Question:** A number in the set _{_ 50 _,_ 51 _,_ 52 _,_ 53 _, ...,_ 999 _}_ is randomly selected. What is the probability that it is a two-digit number? Express your answer as a common fraction. **Solution:** To count the number of numbers in this set, we subtract 49 from all of the numbers, giving the set _{_ 1 _,_ 2 _,_ 3 _, . . . ,_ 950 _}_ , making it obvious that there are 950 numbers total. Furthermore, the set _{_ 50 _,_ 51 _,_ 52 _, . . . ,_ 98 _,_ 99 _}_ corresponds to the more easily counted _{_ 1 _,_ 2 _,_ 3 _, . . . ,_ 49 _,_ 50 _}_ by subtracting 49. So, the probability of selecting a two-digit number is 50 1 950[=] 19[.] 

**Answer:** The answer is 1 19 

**Question:** A pentagon is drawn by placing an isosceles right triangle on top of a square as pictured. What percent of the area of the pentagon is the area of the right triangle? [asy] size(50); draw((0,0)–(0,-1)–(1,-1)–(1,0)–(0,0)–(.5,.5)–(1,0)); [/asy] 

**Solution:** Let the leg length of the isosceles right triangle be _x_ , so the hypotenuse of the triangle has length _x√_ 2. The hypotenuse of the triangle is a side of the square, so the area of the square is ( _x√_ 2)[2] = 2 _x_[2] . The area of the triangle is ( _x_ )( _x_ ) _/_ 2 = _x_[2] _/_ 2. So, the area of the pentagon is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0036-11.png)


Therefore, the fraction of the pentagon’s area that is inside the triangle is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0036-13.png)


**Answer:** The answer is 20 

Figure 32: Chain-of-thought (CoT) prompt for MATH-Prealgebra. 

36 

CoT Prompt for MATH-Intermediate Algebra 

**Question:** Find the quadratic polynomial _p_ ( _x_ ) such that _p_ ( _−_ 3) = 10 _, p_ (0) = 1 _,_ and _p_ (2) = 5 _._ **Solution:** Let _p_ ( _x_ ) = _ax_[2] + _bx_ + _c._ Then from the given information, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0037-02.png)


Then 9 _a −_ 3 _b_ = 9 and 4 _a_ + 2 _b_ = 4 _,_ which reduce to 3 _a − b_ = 3 and 2 _a_ + _b_ = 2 _._ Adding, we get 5 _a_ = 5 _,_ so _a_ = 1 _._ Then 4 + 2 _b_ = 4 _,_ so _b_ = 0 _._ Therefore, _p_ ( _x_ ) = _x_[2] + 1 _._ **Answer:** The answer is _x_[2] + 1 

**Question:** Calculate: 3[5] _[×]_ 106 _[×]_[15] 9 _[×]_[12] 20 _[×]_ 15[25] _[×]_[18] 30 _[×]_ 21[35] _[×]_ 40[24] **Solution:** Each of the fractions[5] 3 _[,]_[15] 9 _[,]_[25] 15 _[,]_[35] 21[reduce to][5] 3 _[,]_[ and each of the fractions] 106 _[,]_[12] 20 _[,]_ 30[18] _[,]_[24] 40 reduce to[3] 5 _[.]_[ Therefore, the product of all eight fractions is 1.] **Answer:** The answer is 1 

**Question:** For _−_ 25 _≤ x ≤_ 25 _,_ find the maximum value of ~~_[√]_~~ 25 + _x_ + ~~_[√]_~~ 25 _− x._ **Solution:** By QM-AM, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0037-06.png)


so ~~_[√]_~~ 25 + _x_ + ~~_[√]_~~ 25 _− x ≤_ 10 _._ Equality occurs at _x_ = 0 _,_ so the maximum value is 10. **Answer:** The answer is 10 

**Question:** The hyperbolas 

and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0037-10.png)


have the same asymptotes. Find _N._ **Solution:** In general, for the hyperbola _[x] a_[2][2] _[−][y] b_[2][2][=][1] _[,]_[ the asymptotes are] _[x] a_[=] _[±][y] b[,]_[ or] _[ y]_[=] _[±] a[b][x.]_ Therefore, the asymptotes of the first hyperbola are _y_ = _±_ 2[3] _[x.]_ For the hyperbola _a[y]_[2][2] _[−][x] b_[2][2][= 1] _[,]_[ the asymptotes are] _a[y]_[=] _[ ±][x] b[,]_[ or] _[ y]_[=] _[ ±][a] b[x.]_[ Therefore, the asymptotes] of the second hyperbola are _y_ = _±_[3] ~~_√_~~ _√N_ 2 _[x.]_ For the two hyperbolas to have the same asymptotes, we must have[3] 2[=][3] ~~_√_~~ _√N_ 2 _[.]_[ Solving for] _[ N]_[gives] _N_ = 8 _._ 

[asy] void axes(real x0, real x1, real y0, real y1) { draw((x0,0)–(x1,0),EndArrow); draw((0,y0)– (0,y1),EndArrow); label(" _x_ ",(x1,0),E); label(" _y_ ",(0,y1),N); for (int i=floor(x0)+1; i<x1; ++i) draw((i,.1)–(i,-.1)); for (int i=floor(y0)+1; i<y1; ++i) draw((.1,i)–(-.1,i)); } path[] yh(real a, real b, real h, real k, real x0, real x1, bool upper=true, bool lower=true, pen color=black) { real f(real x) return _k_ + _a/b ∗ sqrt_ ( _b_[2] + ( _x − h_ )[2] ); real g(real x) return _k − a/b ∗ sqrt_ ( _b_[2] + ( _x − h_ )[2] ); if (upper) draw(graph(f, x0, x1),color, Arrows); if (lower) draw(graph(g, x0, x1),color, Arrows); path [] arr = graph(f, x0, x1), graph(g, x0, x1); return arr; } void xh(real a, real b, real h, real k, real y0, real y1, bool right=true, bool left=true, pen color=black) { path [] arr = yh(a, b, k, h, y0, y1, false, false); if (right) draw(reflect((0,0),(1,1))*arr[0],color, Arrows); if (left) draw(reflect((0,0),(1,1))*arr[1],color, Arrows); } void e(real a, real b, real h, real k) { draw(shift((h,k))*scale(a,b)*unitcircle); } size(8cm); axes(-8,8, -10, 10); xh(2, 3, 0, 0, -8, 8); yh(3*sqrt(2),sqrt(8),0,0,-5,5); draw((-6,9)–(6,-9)h6,9)–(-6,-9),dotted); [/asy] **Answer:** The answer is 8 

Figure 33: Chain-of-thought (CoT) prompt for MATH-Intermediate Algebra. 

37 

## CoT Prompt for MATH-Counting and Probability 

**Question:** The digits 2, 3, 5 and 7 are arranged randomly to form a four-digit number. What is the probability that the number is odd? Express your answer as a common fraction. 

**Solution:** The number formed is odd if and only if its units digit is not 2. Since the digits 2, 3, 5, and 7 are arranged randomly, the probability that 2 is the units digit is 1/4. Therefore, the probability that the number is odd is 1 _−_ 1 _/_ 4 = 4[3][.] 

**Answer:** The answer is[3] 

4 

**Question:** Eight congruent equilateral triangles, each of a different color, are used to construct a regular octahedron. How many distinguishable ways are there to construct the octahedron? (Two colored octahedrons are distinguishable if neither can be rotated to look just like the other.) 

[asy] import three; import math; unitsize(1.5cm); currentprojection=orthographic(2,0.2,1); triple A=(0,0,1); triple B=(sqrt(2)/2,sqrt(2)/2,0); triple C=(sqrt(2)/2,-sqrt(2)/2,0); triple D=(-sqrt(2)/2,sqrt(2)/2,0); triple E=(-sqrt(2)/2,sqrt(2)/2,0); triple F=(0,0,-1); draw(A–B–E–cycle); draw(A–C–D– cycle); draw(F–C–B–cycle); draw(F–D–E–cycle,dotted+linewidth(0.7)); [/asy] **(A)** 210 **(B)** 560 **(C)** 840 **(D)** 1260 **(E)** 1680 **Solution:** Since the octahedron is indistinguishable by rotations, without loss of generality fix a face to be red. 

[asy] size(8cm); defaultpen(0.5); import three; import math; currentprojection=orthographic(2,0.2,1); triple A=(0,0,1); triple B=(sqrt(2)/2,sqrt(2)/2,0); triple C=(sqrt(2)/2,-sqrt(2)/2,0); triple D=(-sqrt(2)/2,sqrt(2)/2,0); triple E=(-sqrt(2)/2,sqrt(2)/2,0); triple F=(0,0,-1); draw(A–B–E–cycle); draw(A–C– D–cycle); draw(F–C–B–cycle); draw(F–D–E–cycle,dotted+linewidth(0.7)); draw(surface(A–B–C– cycle),rgb(1,.6,.6),nolight);[/asy] There are 7! ways to arrange the remaining seven colors, but there still are three possible rotations about the fixed face, so the answer is 7! _/_ 3 = 1680. 

[asy] size(8cm); defaultpen(0.5); import three; import math; currentprojection=orthographic(2,0,1); triple A=(0,0,1); triple B=(sqrt(2)/2,sqrt(2)/2,0); triple C=(sqrt(2)/2,-sqrt(2)/2,0); triple D=(sqrt(2)/2,-sqrt(2)/2,0); triple E=(-sqrt(2)/2,sqrt(2)/2,0); triple F=(0,0,-1); triple right=(0,1,0); picture p = new picture, r = new picture, s = new picture; draw(p,A–B–E–cycle); draw(p,A–C–D–cycle); draw(p,F–C–B–cycle); draw(p,F–D–E–cycle,dotted+linewidth(0.7)); draw(p,surface(A–B–C–cycle),rgb(1,.6,.6),nolight); draw(p,surface(A–B–E–cycle),rgb(1,1,.6),nolight); add(scale3(2.2)*p); draw(r,A–B–E–cycle); draw(r,A–C–D–cycle); draw(r,F–C–B–cycle); draw(r,F–D–E–cycle,dotted+linewidth(0.7)); draw(r,surface(A–B–C–cycle),rgb(1,.6,.6),nolight); draw(r,surface(A–C–D–cycle),rgb(1,1,.6),nolight); add(scale3(2.2)*shift(2*right)*r); draw(s,A–B–E– cycle); draw(s,A–C–D–cycle); draw(s,F–C–B–cycle); draw(s,F–D–E–cycle,dotted+linewidth(0.7)); draw(s,surface(A–B–C–cycle),rgb(1,.6,.6),nolight); draw(s,surface(B–C–F–cycle),rgb(1,1,.6),nolight); add(scale3(2.2)*shift(4*right)*s); [/asy] **Answer:** The answer is 1680 

**Question:** The Gnollish language consists of 3 words, “splargh,” “glumph,” and “amr.” In a sentence, “splargh” cannot come directly before “glumph”; all other sentences are grammatically correct (including sentences with repeated words). How many valid 3-word sentences are there in Gnollish? **Solution:** We proceed by counting the complement, or the number of invalid 3-word sentences. A sentence is invalid precisely when it is of the form “(word) splargh glumph” or “splargh glumph (word).” There are 3 choices for the missing word in each sentence, and since each case is exclusive, we have a total of 6 invalid sentences. Since there are 3 _·_ 3 _·_ 3 = 27 possible 3-word sentences with no restrictions, there are 27 _−_ 6 = 21 that satisfy the restrictions of the problem. **Answer:** The answer is 21 

**Question:** Given that �177 � = 19448, �178 � = 24310 and �179 � = 24310, calculate �199 �. **Solution:** We can apply Pascal’s identity to get that �199 � = �188 � + �189 �. From here, we can apply it twice more to get that �199 � = �188 � + �189 � = ��177 � + �178 �� + ��178 � + �179 ��. Substituting the provided 17 17 17 19 values of � 7 �, � 8 �, and � 9 � gives us � 9 � = 19448 + 2(24310) + 24310 = 92378. **Answer:** The answer is 92378 

Figure 34: Chain-of-thought (CoT) prompt for MATH-Counting and Probability. 

38 

CoT Prompt for MATH-Number Theory 

**Question:** Let _x_ be a positive integer such that 9 _x ≡_ 1 (mod 25). What is the remainder when 11 + _x_ is divided by 25? 

**Solution:** The given information can be expressed by writing _x ≡_ 9 _[−]_[1] (mod 25). Thus we wish to compute 11 + 9 _[−]_[1] (mod 25). Modulo 25, we can write 11 as 11 _·_ (9 _·_ 9 _[−]_[1] ) _≡_ (11 _·_ 9) _·_ 9 _[−]_[1] _≡_ 99 _·_ 9 _[−]_[1] . Thus 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0039-03.png)


so the remainder when 11 + _x_ is divided by 25 is 0. 

Notice that the trick we used here is analogous to using a common denominator to add fractions. 

**Answer:** The answer is 0 

**Question:** Let _m_ be the product of all positive integers less than 4! which are invertible modulo 4!. Find the remainder when _m_ is divided by 4!.(Here _n_ ! denotes 1 _× · · · × n_ for each positive integer _n_ .) 

**Solution:** We compute that 4! = 1 _×_ 2 _×_ 3 _×_ 4 = 2[3] _×_ 3 = 24. So we want exactly the numbers in the set _{_ 1 _, . . . ,_ 24 _}_ which are divisible by neither 2 nor 3, since an integer _a_ is invertible modulo _n_ for some positive integer _n_ if and only if gcd( _a, n_ ) = 1. These turn out to be _{_ 1 _,_ 5 _,_ 7 _,_ 11 _,_ 13 _,_ 17 _,_ 19 _,_ 23 _}_ . Then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0039-09.png)


**Answer:** The answer is 1 

**Question:** Find 2 _[−]_[1] (mod 185), as a residue modulo 185. (Give an answer between 0 and 184, inclusive.) 

**Solution:** Since 2 _·_ 93 _≡_ 186 _≡_ 1 (mod 185), 2 _[−]_[1] _≡_ 93 (mod 185). **Answer:** The answer is 93 

**Question:** What is the remainder when 2007 is divided by 81? 

**Solution:** Dividing using long division, we find that 2007 = 81 _·_ 24 + 63, so the remainder is 63. 

**Answer:** The answer is 63 

Figure 35: Chain-of-thought (CoT) prompt for MATH-Number Theory. 

39 

## CoT Prompt for MATH-Geometry (Part 1) 

**Question:** Two sectors of a circle of radius 12 overlap as shown, with _P_ and _R_ as the centers of the respective circles. Determine the area of the shaded region. 

[asy] draw((0,0)–(10.3923,-6)–(20.7846,0)–(10.3923,6)–cycle,black+linewidth(1)); filldraw((10.3923,6)..(12,0)..(10.3923,-6)–cycle,gray,black+linewidth(1)); filldraw((10.3923,6)..(8.7846,0)..(10.3923,-6)–cycle,gray,black+linewidth(1)); label(" _P_ ",(0,0),W); label(" _Q_ ",(10.3923,6),N); label(" _R_ ",(20.7846,0),E); label(" _S_ ",(10.3923,-6),S); label("60 _[◦]_ ",(0,0),2E); label("60 _[◦]_ ",(20.7846,0),2W); [/asy] 

**Solution:** By symmetry, the areas of the two parts of the shaded region are equal. Consider the right part of the shaded region and the left triangle. [asy] draw((0,0)–(10.3923,-6)–(10.3923,6)–cycle,black+linewidth(1)); filldraw((10.3923,6)..(12,0)..(10.3923,-6)–cycle,gray,black+linewidth(1)); draw((0,0)– (10.3923,0),black+linewidth(1)); draw((10.3923,0)–(9.3923,0)–(9.3923,1)– 

(10.3923,1),black+linewidth(1)); label(" _P_ ",(0,0),W); label(" _Q_ ",(10.3923,6),N); label(" _S_ ",(10.3923,6),S); label(" _Z_ ",(10.3923,0),SW); [/asy] 

The shaded area is equal to the area of sector _PQS_ minus the area of triangle _PQS._ Since ∠ _PQS_ = 60 _[◦]_ and _PQ_ = 12 _,_ the area of sector _PQS_ is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0040-06.png)


Also, triangle _PQS_ is equilateral with side length 12, so its area is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0040-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0040-09.png)


**Answer:** The answer is 48 _π −_ 72 ~~_√_~~ 3 

**Question:** The square with vertices ( _−a, −a_ ) _,_ ( _a, −a_ ) _,_ ( _−a, a_ ) _,_ ( _a, a_ ) is cut by the line _y_ = _x/_ 2 into congruent quadrilaterals. The perimeter of one of these congruent quadrilaterals divided by _a_ equals what? Express your answer in simplified radical form. 

**Solution:** The line _y_ = _[x]_ 2[will intersect the two vertical sides of the square, as shown below:] [asy] real f(real x) { return x/2; } import graph; size(6cm); real a = 8; pair A=(a,a), B=(a,a), C=(a,-a), D=(-a,-a); draw(A–B–C–D–cycle); draw(graph(f,-11,11),Arrows); axes(Arrows(4)); dot("( _−a, a_ )",A,N); dot("( _a, a_ )",B,N); dot("( _a, −a_ )",C,S); dot("( _−a, −a_ )",D,S); real eps=0.2; dot((8,4)h-8,-4)); draw(shift((10,0))*"2 _a_ ",(-a+eps,-a/2-.5)–(a-eps,-a/2-.5),Arrows); draw(shift((0,10))*" _a_ ",(a+2*eps,-a/2)–(a+2*eps,a/2),Arrows);[/asy] The equation of the right side of the square is _x_ = _a,_ so we have _y_ = _[x]_ 2[=] _[a]_ 2 _[,]_[ which means that the intersection point with the right] side of the square is � _a,[a]_ 2 � _._ Similarly, the equation of the left side of the square is _x_ = _−a,_ so we have _y_ = _[x]_ 2[=] _[ −][a]_ 2 _[,]_[ which means that the intersection point with the left side of the square is] � _−a, −[a]_ 2 � _._ It follows that the sides of each quadrilateral have lengths _[a]_ 2 _[,]_[ 2] _[a,]_[3] 2 _[a][,]_[ and] ~~�~~ _a_[2] + (2 _a_ )[2] = _a_ ~~_√_~~ 5 _,_ by the Pythagorean theorem. Hence, the perimeter of the quadrilateral is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0040-13.png)


Figure 36: Chain-of-thought (CoT) prompt for MATH-Geometry (Part 1). 

40 

## CoT Prompt for MATH-Geometry (Part 2) 

**Question:** A right circular cylinder with radius 2 is inscribed in a hemisphere with radius 5 so that its bases are parallel to the base of the hemisphere. What is the height of this cylinder? **Solution:** We draw and label a diagram as follows: 

[asy] size(110); pair O = (0,0); pair A = (.3,.94); pair B = (.3,.075); draw(O–A–B– cycle,heavycyan); label(" _O_ ",O,W); label(" _A_ ",A,N); label(" _B_ ",B,S); import solids; import three; defaultpen(linewidth(0.8)); currentprojection = orthographic(5,0,1.3); revolution c = cylinder((0,0,0), .4, .91); draw(c,black); draw(scale(1,.25)*arc((0,0),1,0,180),dashed); draw(scale(1,.25)*arc((0,0),1,180,360)); draw(Arc((0,0),1,0,180)); [/asy] 

Let the center of the hemisphere be _O_ , and let _A_ be a point on the circumference of the top circle of the cylinder. Since the cylinder is inscribed in the hemisphere, _A_ lies on the hemisphere as well, so _OA_ = 5. We drop a perpendicular from _A_ to the base of the hemisphere and let it intersect the base of the hemisphere at _B_ . Since the cylinder is right and _AB_ is a height of the cylinder, ∠ _OBA_ is a right angle, and _B_ lies on the circumference of the bottom circle of the cylinder. Thus, _OB_ is a radius of the cylinder, so _OB_ = 2. We have that _△OBA_ is right, so by the Pythagorean theorem, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0041-04.png)


**Question:** Parallelogram _ABCD_ with _A_ (2 _,_ 5), _B_ (4 _,_ 9), _C_ (6 _,_ 5), and _D_ (4 _,_ 1) is reflected across the _x_ -axis to _A[′] B[′] C[′] D[′]_ and then _A[′] B[′] C[′] D[′]_ is reflected across the line _y_ = _x_ + 1 to _A[′′] B[′′] C[′′] D[′′]_ . This is done such that _D[′]_ is the image of _D_ , and _D[′′]_ is the image of _D[′]_ . What is the ordered pair of _D[′′]_ in the coordinate plane? 

**Solution:** Reflecting a point across the _x_ -axis multiplies its _y_ -coordinate by _−_ 1. Therefore, _D[′]_ = (4 _, −_ 1). To reflect _D[′]_ across the line _y_ = _x_ + 1, we first translate both the line and the point down one unit so that the equation of the translated line is _y_ = _x_ and the coordinates of the translated point are (4 _, −_ 2). To reflect across _y_ = _x_ , we switch the _x_ -coordinate and _y_ -coordinate to obtain ( _−_ 2 _,_ 4). Translating this point one unit up, we find that _D[′′]_ = ( _−_ 2 _,_ 5). **Answer:** The answer is (-2,5) 

Figure 37: Chain-of-thought (CoT) prompt for MATH-Geometry (Part 2). 

41 

CoT Prompt for MATH-Precalculus (Part 1) **Question:** The function _f_ ( _z_ ) =[(] _[−]_[1 +] _[ i] √_ 3) _z_ + ( _−_ 2 _√_ 3 _−_ 18 _i_ ) 2 represents a rotation around some complex number _c_ . Find _c_ . **Solution:** Since a rotation around _c_ fixes _c_ , the complex number _c_ must satisfy _f_ ( _c_ ) = _c_ . In other words, _√_ 3) _c_ + ( _−_ 2 _√_ 3 _−_ 18 _i_ ) _c_ =[(] _[−]_[1 +] _[ i]_ 2 Then 2 _c_ = ( _−_ 1 + _i√_ 3) _c_ + ( _−_ 2 _√_ 3 _−_ 18 _i_ ), so (3 _− i√_ 3) _c_ = _−_ 2 _√_ 3 _−_ 18 _i._ Then _√_ 3 _−_ 18 _i c_ = _[−]_[2] 3 _− i_ ~~_√_~~ 3 =[(] _[−]_[2] _√_ 3 _−_ 18 _i_ )(3 + _i√_ 3) (3 _− i_ ~~_√_~~ 3)(3 + _i_ ~~_√_~~ 3) _√_ 3 _−_ 6 _i −_ 54 _i_ + 18 _√_ 3 = _[−]_[6] 12 _√_ 3 _−_ 60 _i_ =[12] 12 = _√_ 3 _−_ 5 _i._ **Answer:** The answer is _√_ 3 _−_ 5 _i_ 

**Question:** Convert the point �8 _,_[7] 6 _[π]_ � in polar coordinates to rectangular coordinates. **Solution:** In rectangular coordinates, �8 _,_[7] 6 _[π]_ � becomes 8 cos[7] _[π]_[7] _[π]_ = ( _−_ 4 _√_ 3 _, −_ 4) _._ � 6 _[,]_[ 8 sin] 6 � **Answer:** The answer is ( _−_ 4 _√_ 3 _, −_ 4) 

Figure 38: Chain-of-thought (CoT) prompt for MATH-Precalculus (Part 1). 

42 

## CoT Prompt for MATH-Precalculus (Part 2) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving_images/nips_2024_StrategyLLM_Large_Language_Models_as_Strategy_Generators,_Executors,_Optimizers,_and_Evaluators_for_Problem_Solving.pdf-0043-01.png)


Figure 39: Chain-of-thought (CoT) prompt for MATH-Precalculus (Part 2). 

43 

