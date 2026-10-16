# **Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning** 

**Beichen Zhang**[13] _[∗]_ **, Kun Zhou**[23] _[∗]_ **, Xilin Wei**[4] **, Wayne Xin Zhao**[13] _[†]_ **, Jing Sha**[5] **, Shijin Wang**[56] **, Ji-Rong Wen**[123] 

1Gaoling School of Artificial Intelligence, Renmin University of China. 

2School of Information, Renmin University of China. 

3Beijing Key Laboratory of Big Data Management and Analysis Methods. 

4College of Computer Science, Sichuan University. 

5iFLYTEK Research. 

6iFLYTEK AI Research (Central China). 

`{zhangbeichen724,wiselnn570,batmanfly}@gmail.com` , `francis_kun_zhou@163.com {jingsha,sjwang3}@iflytek.com` , `jrwen@ruc.edu.cn` 

## **Abstract** 

Chain-of-thought prompting (CoT) and tool augmentation have been validated in recent work as effective practices for improving large language models (LLMs) to perform step-by-step reasoning on complex math-related tasks. However, most existing math reasoning datasets may be not able to fully evaluate and analyze the ability of LLMs in manipulating tools and performing reasoning, as they may only require very few invocations of tools or miss annotations for evaluating intermediate reasoning steps. To address the issue, we construct **CARP** , a new Chinese dataset consisting of 4,886 computation-intensive algebra problems with formulated annotations on intermediate steps. In CARP, we test four LLMs with CoT prompting, and find that they are all prone to make mistakes at the early steps of the solution, leading to wrong answers. Based on this finding, we propose a new approach that can deliberate the reasoning steps with tool interfaces, namely **DELI** . In DELI, we first initialize a step-by-step solution based on retrieved exemplars, then iterate two deliberation procedures that check and refine the intermediate steps of the generated solution, from the perspectives of tool manipulation and natural language reasoning, until obtaining converged solutions or reaching the maximum turn. Experimental results on CARP and six other datasets show that the proposed DELI mostly outperforms competitive baselines, and can further boost the performance of existing CoT methods. Our data and code are available in `https://github.com/RUCAIBox/CARP` . 

## **1 Introduction** 

Recently, large language models (LLMs) ( _e.g.,_ GPT-3 and ChatGPT) have shown remarkable zeroshot and few-shot performance on various tasks [1–3], including language generation and question answering. As LLMs have been pre-trained on a large amount of text data, covering broad types of world knowledge, existing work also shows that LLMs can solve complex tasks, _e.g.,_ math reasoning [4] and college entrance exam [5, 6]. 

To evaluate the capacity of LLMs for solving complex tasks, math reasoning datasets have been widely used as testbeds, _e.g.,_ GSM8K [7] and MATH [8], where the math problems can not be 

> _∗_ Equal contributions. 

> _†_ Corresponding author. 

Preprint. Under review. 

directly answered but require multi-step reasoning. To elicit LLMs for step-by-step reasoning, chainof-thought (CoT) [4, 9] has become the de facto prompting strategy, where LLMs can be guided to generate a solution consisting of a series of intermediate steps for reaching the answer. However, previous work also reveals that LLMs are prone to make mistake at intermediate steps, especially for numerical computation [10–14], and yet a minor mistake would lead to a totally wrong answer. To alleviate it, a line of work [15–25] employs external tools to make up for the weakness of LLMs, and can greatly improve the answer accuracy on math reasoning tasks. With the rapidly evolving LLMs and tool-augmented methods, it is necessary to adopt a suitable math reasoning dataset for evaluating them systematically and differentially. Whereas, the problems in most existing math reasoning datasets may only require the one-off utilization of tools [16, 17, 19, 20], which are not adequate to fully measure the ability of tool manipulation in existing methods. Besides, although the wrong answers mostly derive from the incorrect intermediate steps in step-by-step reasoning, most existing datasets can not be utilized for testing that, due to lack of formal annotations of the intermediate steps in the solution text. The two issues limit existing math reasoning datasets to systemically evaluate and analyze LLMs and tool-augmented methods. 

To address them, we construct a new Chinese dataset that consists of 4,886 **C** omputation-intensive **A** lgeb **R** a **P** roblems associated with formulated annotations of all the intermediate steps, namely **CARP** . In CARP, all the problems require deriving multiple intermediate math expressions based on math knowledge, and solving them based on arithmetical knowledge, which make it a complex and difficult dataset to evaluate the computation-intensive math reasoning ability. In addition, the formulated annotations also enable researchers to test the accuracy of intermediate reasoning steps for analyzing the errors of LLMs. As shown in Table 3, four popular LLMs with CoT prompting can not solve over half of the problems in our CARP, indicating the difficulty of CARP. Furthermore, we also find that all LLMs are more likely to make mistakes in the first step (over 69%), leading to totally wrong solutions and answers. It reveals that LLMs mostly fail in performing early reasoning steps, and can not correct the errors in the latter steps. Based on CARP, we also devise a variety of fine-grained interfaces based on available tools, to provide practical functionalities for handling complicated calculations. These interfaces can also be applied to other math reasoning datasets to improve the tool manipulation capacity of LLMs. 

Considering that LLMs can not fix the errors in early steps by themselves, we propose a new approach that can **deli** berate the reasoning steps of LLMs with interfaces of tools, namely **DELI** . In DELI, we first initialize a step-by-step solution for the given question based on retrieved relevant exemplars, then iterate two deliberation procedures that check and refine the generated step-by-step solution from the perspectives of tool manipulation and natural language reasoning, until reaching the stop condition, _e.g.,_ solution has converged or iterations reach the maximum number. Such a way is similar to the solution checking process of humans, and can elicit LLMs to deliberate and correct the possible errors in intermediate steps of the solution. We evaluate our proposed DELI and existing prompting methods on CARP and six other computation-intensive datasets. Experimental results show that the proposed DELI mostly outperforms competitive baselines ( _e.g.,_ 9.35% accuracy improvement over the best baseline on CARP), and can further boost the performance of existing CoT prompting methods. 

To summarize, our major contributions are: 

_•_ We construct a new dataset named CARP with formulated annotation of intermediate reasoning steps for systematically evaluating LLMs in solving computation-intensive math problems, and devise interfaces with practical functionalities to help LLMs. 

_•_ We propose DELI, a new approach that can deliberate and correct the reasoning steps of LLMs with interfaces of tools. 

_•_ We conduct extensive experiments to show the superiority of our DELI over existing prompting methods on 7 computation-intensive math reasoning datasets. 

## **2 CARP Dataset** 

Computation-intensive math reasoning task aims to solve complex math problems that require performing multi-step arithmetical computation and reasoning based on mathematical knowledge. Typically, to solve a computation-intensive math problem, humans or models need to iterate the 

2 

|Table 1: Statistics for CARP dataset.|Table 1: Statistics for CARP dataset.||**Given Conditions**|
|---|---|---|---|
|**Statistic**<br># of training samples|**Number**<br>3,410|𝑎𝑥+ 2𝑏𝑥=<br>substituted expr|3<br>𝑥= 1<br>𝑎<br>𝑦−1 + 2𝑏<br>𝑦−1 = 3<br>factored expr<br>substitution|
|# of development samples<br># of testing samples|500<br>976|𝑎+ 2𝑏=|𝑎+ 2𝑏𝑦−<br>𝑎+ 2𝑏= 3<br>3<br>common factor|
|# of nodes (Avg./Max)|6.0/18|substitution<br>substituted expr||
|# of edges (Avg./Max)<br># of expression nodes (Avg./Max)<br>Problem length (Avg./Max)|5.7/25<br>4.7/15<br>52.1/257||𝑦= 1<br>3𝑦−3 = 0<br>equation|
|Solution length (Avg./Max)|71.3/278|Figure 1:An|EFG annotation example for CAR|



Figure 1: An EFG annotation example for CARP. 

process multiple times that derives the formula and then compute the result (via calculators or other tools), until obtaining the final answer. In this way, the accuracy of intermediate reasoning and computation steps is crucial, where a subtle error would lead to totally wrong answers. In this paper, we construct a new dataset CARP (Computation-intensive AlgebRa Problems) that provides the formulated annotations of all the intermediate steps for the computation-intensive middle school math problems. Based on the annotations, we also design a set of interfaces with fine-grained computation functions, to help LLMs manipulate commonly-used tools for solving these problems. 

## **2.1 Dataset Construction** 

Although there are a number of computation-intensive math problems in available datasets, their solutions are generally in not well-formulated natural language and may omit intermediate steps [26– 30, 7, 8]. To construct a well-formulated dataset, we first collect real-world computation-intensive math problems, then invite crowd-sourced workers to extract and annotate their expression flow graph. 

**Data Collection.** We collect the math problems and their step-by-step solutions from a Chinese education website Zhixue[3] , which contains vast problems to provide education assistance for students. We mainly crawl middle school math problems, since they are of moderate difficulty and require basic arithmetical computations ( _e.g.,_ quadratic equation) and mathematical knowledge ( _e.g.,_ Veda’s theorem), making them a good testbed for computation-intensive math reasoning. We first crawl about 1,000,000 problems with solutions. Then, to obtain computation-intensive problems, we design hand-crafted rules based on SymPy to roughly extract and count the computation steps in solutions, and only select the ones with both over one computation step and over two reasoning steps. Finally, we invite math teachers to select about 10,000 high-quality examples for annotation. 

**Expression Flow Graph Annotation.** In a math problem, the natural language solution can be generally formulated as a directed acyclic graph (DAG), where the nodes and edges refer to the intermediate results and derivation steps, respectively [31]. For computation-intensive problems, we consider a special DAG format that adopts intermediate math expressions as nodes. We name it _expression flow graphs_ (EFG), as it can explicitly show how to derive new math expressions based on existing ones in the step-by-step reasoning process. In this way, a solution can be formulated as: starting from initial condition nodes within the problem, we continue deriving new nodes ( _i.e.,_ intermediate math expressions) from existing nodes, until reaching the final expression that can obtain the answer, where the computation results of intermediate expressions can be utilized for evaluation. Whereas, a math problem may involve special initial conditions that are hard to be converted into readable expressions, _e.g.,_ Equations have rational solutions. Thus, we add a special type of node to store these conditions in natural language, while guaranteeing that all the derived new nodes are math expressions. As an example, the EFG annotation of the Problem in Table 2 is shown in Figure 1. 

Based on the above definition, we invite five middle school math teachers as crowd-sourced workers to annotate the formulated EFGs of our collected problems. The annotation process is similar to the information extraction process [32], where we first extract the nodes and then link them to compose 

> 3https://www.zhixue.com/ 

3 

Table 2: An example from the CARP dataset, which is translated into English. Errors are annotated with red color. 

|ith red color.||
|---|---|
|**Problem**|The solution to the equation_ax_+ 2_bx_ = 3is_x_ = 1, then the solution to the|
||equation_a_(_y −_1) + 2_b_(_y −_1) = 3is ?|
|**Solution**|From the question we have :_a_+2_b_= 3,_a_(_y −_1)+2_b_(_y −_1) = 3. Rectifying|
||gives(_a_+ 2_b_)_y −_(_a_+ 2_b_) = 3,_i.e.,_3_y −_3 = 3, therefore_y_ = 2.|
|**ChatGPT**|Substituting_x_= 1into_ax_+ 2_bx_= 3gives_a_+ 2_b_= 3, andsubstituting_y −_1|
||gives_a_(_y −_1) + 2_b_(_y −_1) = 3, which simplifes to_ay_+_by_ = 3. _· · · · · ·_ The<br>answer is_y_ =<br>3<br>3_−a −_1|
|**Error Type**|Reasoning error.|



the graph. Concretely, we first rely on hand-crafted rules to automatically extract the math expressions and text conditions from the solution texts as node candidates. Then, we ask math teachers to link the related node candidates and annotate their corresponding relations. To reduce the difficulty, we utilize heuristic rules to select the most possible related nodes and relations as references. Consequently, we can collect a set of edges with special relations connecting several nodes from the node candidates, which compose the EFG of a problem. After annotation, we further design an automatic verification program to verify the completeness of the EFG and the validity of relations, and filter improper ones. Besides, we also ask teachers to check the annotated EFGs from each other, to judge if the EFG has fully covered the whole problem-solving process of the problem, and refine the incomplete ones. 

## **2.2 Dataset Details** 

**Dataset Description.** The statistics of the CARP dataset are shown in Table 1. CARP consists of 4,886 middle school computation-intensive algebra problems, and each problem is associated with a natural language solution and an annotated EFG. Our annotated EFG explicitly depicts the step-by-step reasoning process of a math problem in a readable and concise format. On average, an EFG contains 6.0 nodes and 5.7 edges, as we only keep the expressions and conditions that lead to the final answer in the EFG. Besides, an EFG has 4.7 expression nodes on average, which are the main stem of the whole reasoning process and can be used for evaluating the accuracy of intermediate steps. 

To solve the problems in CARP, LLMs require to iteratively perform reasoning based on math knowledge to correctly derive the intermediate math expressions, and solve it accurately. As the example in Table 2, given the conditions, a reasonable solving process should first deduce the intermediate equation _a_ + 2 _b_ = 3 by substituting _x_ = 1 into _ax_ + 2 _bx_ = 3, and then reformulate the equation _a_ ( _y −_ 1) + 2 _b_ ( _y −_ 1) = 3 to support plugging _a_ + 2 _b_ = 3 into it. Such a reformulation step is not easy to reason out, and ChatGPT has made a mistake there, leading to a wrong answer. 

**Evaluation Metrics.** Based on EFGs, we can evaluate the intermediate step-by-step reasoning process of LLMs. Specifically, we propose two new metrics, _i.e., ExpAcc_ and _Fail@where_ . ExpAcc measures the accuracy rate of the intermediate expressions of a problem in the generated output. Considering that a math problem may have different ways to solve it, we also regard the ancestors of a correct intermediate expression as true ones, as they derive the right expression. In this way, ExpAcc can be obtained by first finding accurately matched expression nodes and then counting their ancestors and themselves as accurate ones for computing the rate. We leverage SymPy to judge if two math expressions are matched. Fail@where is another type of metric for analyzing where are the causes of incorrect answers, and we define three implementations, _i.e.,_ Fail@first, Fail@middle, and Fail@last. The three metrics refer to the rates of making the first mistakes in the first step, middle steps, and last step (before the answer) within all generated incorrect solutions, respectively. 

As shown in Table 3, we evaluate competitive LLMs on CARP with chain-of-thought prompt [4] and report the answer accuracy, ExpAcc, and Fail@where. First, all LLMs can not solve over half of the problems in CARP (under 50.0), and the accuracy of intermediate steps is relatively lower (under 57.0), indicating the difficulty of computation-intensive math reasoning. Second, all LLMs are more likely to make mistakes in the first step, while less likely in the last step. It demonstrates that LLMs 

4 

Table 3: Evaluation results of different LLMs with CoT prompting on CARP. 

|**Models**<br>**Acc.**<br>**ExpAcc**|**Fail@where**|
|---|---|
||**Fail@frst**<br>**Fail@middle**<br>**Fail@last**|
|`text-davinci-002`<br>31.15<br>37.45<br>`text-davinci-003`<br>37.50<br>44.89<br>`claude-v1.3`<br>40.78<br>46.89<br>`gpt-3.5-turbo`<br>**49.39**<br>**56.48**|79.04<br>11.29<br>9.65<br>73.61<br>15.41<br>10.98<br>76.85<br>12.08<br>11.05<br>69.69<br>16.36<br>13.94|



are prone to fail in early steps, due to misuse of improper math knowledge or wrong calculations. Thus, careful deliberations on early steps might be promising to reduce errors of the model. 

## **2.3 Tool Interfaces** 

As the results in Table 3 and existing work [8, 11–13]., it is hard for LLMs to solve computationintensive math problems, especially for numerical calculation. In the real world, humans can utilize tools ( _e.g.,_ calculator) to avoid errors in manual work. Inspired by it, we consider augmenting LLMs with tools for handling complicated calculations. Considering the complexity of math calculation, we devise multiple interfaces based on available tools, to provide specific and practical functionalities. All the interfaces are formulated into a unified format with detailed descriptions, to support convenient manipulation of LLMs. Concretely, we mainly utilize SymPy [33] as the tool, which is a Python library including various basic and advanced arithmetic operators. Based on it, we encapsulate three types of interfaces to help the computation of LLMs: (1) **Numerical Computation** : compute the value _v_ of an expression _e_ by calculating directly or substituting existing conditions. (2) **Equation Solving** : solve an equation or inequation _e_ , or solve the system of equations or inequalities _{e}_ . (3) **Expression Transformation** : transform an expression _e_ into the desired format _e[′]_ . 

Based on them, we devise fine-grained interfaces covering commonly-used functionalities in math calculation. We set the name, arguments, and output formats of each interface, associated with a docstring that provides a natural language explanation for its usage. These interfaces are general to various computation-intensive math reasoning tasks, and can help LLMs perform complex computations. In addition, we also add a special interface, _think_ , which can utilize the LLM to analyze existing conditions, deduce new conclusions, and create new math expressions, before or after tool manipulation. It can also help handle the cases that fail to invoke computation interfaces, where LLMs _think_ to produce an output instead, to prevent the solving process from being interrupted. 

## **2.4 Dataset Discussion** 

Our proposed CARP dataset focuses on systematically evaluating LLMs in solving computationintensive math problems. CARP exhibits three key characteristics: First, solving problems in CARP involves multi-step reasoning with math domain knowledge, math expression understanding, and complex computations; Second, fine-grained interfaces are provided in CARP for LLMs to evaluate the tool manipulation ability in complex reasoning. In this scenario, LLMs should understand the usage of fine-grained interfaces, and invoke them reasonably multiple times based on reasoning and mathematical knowledge in the solving process. Third, evaluation metrics for intermediate reasoning steps based on formulated annotations are employed to better analyze the multi-step reasoning performance of LLMs, while existing datasets mainly focus on evaluating the outcome accuracy of solutions [6–8]. Via those metrics, researchers can quantify the models’ mastery of the solution process and thus acquire more clues for improving models. 

## **3 Approach** 

According to the results in Section 2.2, it is hard for LLMs to solely solve computation-intensive math problems, and they often make mistakes in early reasoning steps. Inspired by humans that often review and check the solutions, we propose a new approach that can **deli** berate the reasoning steps of LLMs with interfaces of tools, namely **DELI** . The overview of DELI is shown in Figure 2. In DELI, we leverage a retrieval-augmented chain-of-though prompting strategy to initialize a step-by-step 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning_images/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Question:  Knowing that the result of (𝑥−2)(1 − Deliberation with Chain of Thought Tool Interfaces<br>𝑘𝑥) −(2𝑥−3)(2𝑥+ 3) does not contain 𝑥 in the<br>primary form , then 𝑘 = ____ ? Answer:  Expanding (𝑥−2)(1 −𝑘𝑥) −(2𝑥−3)(2𝑥+ calculate (expr)<br>Retrieve QA pairs 3which gives gives 𝑥 [!] −𝑘−4 𝑘= − "!+ 𝑥 [.] [ The answer is: ] 2𝑘+ 1 + 7… … we get  [−] "! 2𝑘+ 1 = 0， ► solve_eq new_expr(expr)<br>► new_exprs<br>Question:  If 𝑥+ 1 𝑥+ 𝑞 ……<br>Answer:  (𝑥+ 1)(𝑥+ 𝑞) ……. expand (expr)<br>Question:  …… Deliberation with Tool Interfaces ► new_expr<br>…… Action:  expand( 𝑥−2 1 −𝑘𝑥− 2𝑥−3 2𝑥+ 3 ) collect (expr, symbol)<br>Serve as exemplars OutputAction:  : −𝑘𝑥collect( [!] + 2𝑘𝑥−4𝑥−𝑘𝑥 [!] + 2𝑘𝑥−4𝑥 [!] + 𝑥+ 7 [!] + 𝑥+ 7, 𝑥) ► new_expr<br>Answer:  −4𝑥 [!] −𝑘𝑥𝑥−2 [!] + 5𝑥−151 −𝑘𝑥− , since the result does not 2𝑥−3 2𝑥+ 3 = Output: Action: Output : [solve_eq(𝑥𝑘= − [!] (−𝑘−4) + 𝑥(2𝑘+ 1) + 7"! []] 2𝑘+ 1 = 0) InvokeReturn …… think (thought)<br>contain the primary term of 0. Hence no solution. 𝑥 , we get 5 = Final Answer:  − "! ► conclusion<br>Retrieval-Augmented Solution Initialization Iterative Deliberation<br>**----- End of picture text -----**<br>


Figure 2: The overview of our DELI. DELI first initializes the step-by-step solution via retrievalaugmented strategy, and then performs iterative deliberation with tool manipulation and chain of thought, respectively. 

natural language solution. Then, we iterate the two-stage deliberation method that checks and refines the solution from the perspectives of natural language reasoning and tool manipulation. After multiple iterations, we can finally obtain a more reasonable solution with the answer. 

## **3.1 Retrieval-Augmented Solution Initialization** 

As our approach focuses on performing deliberation on the generated solution of LLMs, we aim to first initialize a high-quality step-by-step solution for the given question that covers useful mathematical knowledge and arithmetic operators. Therefore, we propose to retrieve relevant problems and solutions as the exemplars, and then utilize the chain-of-thought prompting method [4] to generate the initial solution based on them. Concretely, given a math problem _p_ , we first utilize a retriever to select the top- _k_ relevant problems _C_ = _{⟨pi, si⟩}[k] i_ =1[from the candidate pool based on question-question] matching, where the retriever can be either lexicon-based [34] or dense retrieval models [35]. Then, the retrieved problems with their associated step-by-step solutions, will be employed to compose the input prompt, to elicit LLMs for performing chain-of-thought reasoning. The pattern of the input “ prompt is denoted as: _You are a helpful assistant for solving math problems in LaTeX format: [p_ 1 _], [s_ 1 _], · · · , [pk], [sk], [p]_ ”. In this way, LLMs would follow the in-context exemplars to perform step-by-step reasoning, and can also refer to useful mathematical knowledge and arithmetic operators from them, leading to more high-quality initial solutions for deliberation. Note that such a way also supports other prompting methods to initialize solutions. 

## **3.2 Iterative Deliberation** 

Based on the initial solution of the given question, we iterate two types of deliberation procedures, _i.e.,_ deliberation with tool interfaces and deliberation with chain of thought, until reaching the stop condition. In the two deliberation procedures, we adopt specific in-context exemplars to guide LLMs, for checking and correcting the errors in the current solution. Next, we first introduce the details of the two deliberation procedures, and then present the stop condition. The algorithm of the iterative framework is illustrated in the supplemental materials. 

**Deliberation with Tool Manipulation.** Since LLMs are prone to make mistakes in numerical calculation, we design the procedure of deliberation with tool manipulation, for seeking help from external tools to address it. Based on our devised interfaces in Section 2.3, we aim to rewrite the current solution into a process that orderly invokes the interfaces to produce the result. In this way, the deliberation procedure is divided into a sequence of steps, where the LLM should select the interface and then invoke it to produce the intermediate result in each step. 

Concretely, first, we construct an instruction that introduces the goal and formats of this procedure, and the details of all available interfaces. For each interface, we not only list its name, arguments and description, but also provide an example to exhibit the way to use it, _e.g.,_ “ _expand(expression:_ 

6 

_str)→ new expression: str: Expand the expression into a polynomial_ . _e.g.,_ expand(( _x_ + 1)[2] ) -> _x_[2] + 2 _x_ + 1”. Then, we demonstrate several exemplars to guide LLMs to invoke the interfaces. Each exemplar consists of four parts, _i.e.,_ a question, a trial, multiple actions, and their outputs. The trial is the initial step-by-step solution to the given question, which may contain a few errors requiring correction. Actions are a series of interface invocation operations derived from the trial, and outputs are the intermediate results by executing the actions, _e.g.,_ “ _Action: solve_eq(_ 2 _k_ + 1 = 0 _). Output: [k_ = _−_[1] 2 _[]]_[”.][Based on the instruction and exemplars, the LLM would be elicited to generate the action] in formal language iteratively ( _i.e.,_ selecting the interface and setting its arguments), then execute it to obtain the intermediate result, until reaching the answer. To guarantee the continuity of the deliberation procedure, we set a special token after the generated action, for pausing the generation process and waiting for the result of interface invocation. In the iterative selection-then-execution process, we can deliberate the intermediate steps of the generated solution, and benefit from tool manipulation for accurate numerical computation. 

**Deliberation with Chain of Thought.** After deliberation with tools, we can obtain the solution in formal language consisting of a series of actions to invoke interfaces and their outputs. Next, we further deliberate the solution with chain of thought to reorganize it into the natural language format, which can better make use of the learned textual knowledge from LLMs to recheck it and also improve the readability. 

Similarly, we also leverage an instruction with in-context exemplars to compose the input prompt. The instruction is “ _You have access to both natural language problem solving processes and formal problem solving processes, but there may be errors within them. You need to learn the correct methods in order to better solve problems._ ”, to introduce the goal of the deliberation procedure. All the exemplars are composed of four components, _i.e.,_ a question, a given solution, the verification, and the revised solution. The given solution is the last natural language solution that is either the initial solution or the solution from the last deliberation iteration with chain of thought, and the verification is the formal language solution from the last deliberation procedure with tool interfaces. The revised solution is the result of integrating the two types of solutions into the chain-of-thought manner, where the errors and unreasonable steps have been corrected. Guided by the exemplars, LLMs would also deliberate the intermediate steps from in-context solutions, and generate a new natural language solution to answer the given problem. Besides, as there are often inconsistent intermediate computation results in the in-context solutions, we also add an instruction to elicit LLMs to trust more on the result from tool manipulation, _i.e.,_ “ _If the computed result in the verification differs from the computed result in the given solution, the computed result in the verification must be used as the standard_ ”. 

**Stop Conditions of Iteration.** The devised two deliberation procedures would be alternated multiple times, where the solution might be iteratively revised and improved. To control the cost, we set the stop conditions of the iteration process. First, once the solution of the new iteration is the same as the last one, the iteration will be stopped, since it reflects that the iteration has converged. Second, if the answers of the two deliberation procedures are consistent, we will also stop the iteration. Third, if we have reached the maximum number of iterations, the answer from the last deliberation procedure with tool manipulation will be regarded as the final answer, as such a procedure can better solve computation subproblems, leading to a more accurate answer. 

Table 4: Basic information about datasets in evaluated datasets. MS and HS refer to “middle school” and “high school”, respectively. 

|and “high school”,|respectively.||||||
|---|---|---|---|---|---|---|
|**Dataset**|**Source**|**Language**|**Domain**|**Diffculty**|**Train**|**Test**|
|CARP|Ours|Chinese|Algebra|MS|3,410|976|
|Algebra|MATH|English|Algebra|HS|1,744|1,187|
|Prealgebra|MATH|English|Algebra|HS|1,205|871|
|Count. & Prob.|MATH|English|Probability|HS|771|474|
|Num. Theory|MATH|English|Number Theory|HS|869|540|
|GK-Cloze|AGIEval|Chinese|Mixture|HS|-|220|
|SAT-Math|AGIEval|English|Mixture|HS|-|351|



7 

Table 5: Results on 7 computation-intensive math reasoning datasets. We copy results of LP from Guo et al. [36]. The best and second-best methods are marked in bold and underlined respectively. 

|e a. . e es an secon-es meos are mare n o an une<br>.|e a. . e es an secon-es meos are mare n o an une<br>.|e a. . e es an secon-es meos are mare n o an une<br>.|e a. . e es an secon-es meos are mare n o an une<br>.|rne re|pecvey.|
|---|---|---|---|---|---|
|**Methods**<br>**CARP**<br>**Algebra**<br>**Prealgebra**<br>**CP**<br>**NT**||||**GKC**|**SAT**<br>**Avg.**|
|Random CoT<br>Complex CoT<br>Retrieval CoT|49.39<br>48.06<br>63.93|49.37<br>51.64<br>53.75|55.57<br>32.91<br>29.81<br>53.73<br>32.91<br>32.22<br>56.72<br>33.12<br>30.00|14.41<br>-<br>-|65.91<br>42.48<br>-<br>-<br>-<br>-|
|PAL<br>ReAct|40.00<br>64.11|34.29<br>54.51|50.52<br>35.86<br>31.30<br>54.53<br>**41.77**<br>31.67|5.93<br>16.94|47.73<br>35.09<br>72.27<br>48.07|
|LP<br>PHP|-<br>61.68|49.60<br>54.42|52.30<br>30.20<br>29.80<br>57.86<br>36.71<br>**35.37**|-<br>16.94|-<br>-<br>71.82<br>47.82|
|Iterative CoT<br>Iterative ReAct|61.27<br>61.17|52.74<br>53.92|55.34<br>33.97<br>29.81<br>52.12<br>37.34<br>32.22|14.41<br>15.25|69.55<br>45.30<br>70.00<br>46.00|
|DELI|**73.46**|**59.65**|**58.32**<br>39.03<br>33.15|**17.80**|**74.54**<br>**50.85**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning_images/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning.pdf-0008-02.png)


**----- Start of picture text -----**<br>
CMA Algebra Count. & Prob.<br>80 65 42<br>CoT + DELI<br>70 CoT 60<br>38<br>60 55<br>34<br>50 50<br>40 45 30<br>Random CoTComplex CoT PHPRetrieval CoT Random CoTComplex CoT PHPRetrieval CoT Random CoTComplex CoT PHPRetrieval CoT<br>**----- End of picture text -----**<br>


Figure 3: The results of combining DELI with existing CoT methods. 

## **4 Experiment** 

## **4.1 Main Experiments** 

**Settings.** In addition to our CARP dataset, we also collect six existing computation-intensive math problem datasets for evaluation, including Algebra, Prealgebra, Counting and Probability (CP), and Number Theory (NT) from MATH benchmark [8] and GK-Math-Cloze (GKC) and SAT-Math from AGIEval [6]. The statistics of these datasets are shown in Table 4 These datasets need multistep reasoning and computation to solve the problem, and the required knowledge varies from middle school to math competitions. We also show the details of baselines and implementation in supplementary material. 

**Main Results.** Table 5 compares DELI with other baselines on the 7 datasets. For the comparison between CoT prompting methods, Retrieval CoT outperforms Random CoT on average, indicating that reviewing relevant knowledge and problem-solving idea benefit answering complex math problems. Augmented with our provided interfaces, ReAct achieves better average performance than CoT prompting methods, demonstrating the effectiveness of wrapping tools to aid math reasoning through interfaces. Besides, in the comparison between tool-augmented prompting, the performance of ReAct is better than PAL on the benchmark. It indicates that reasoning with intermediate results is important for solving computation-intensive math problems. Finally, our proposed DELI performs better than competitive baselines in most cases. DELI improves upon the basic components of deliberations, including CoT prompting methods and ReAct, while iterative variants ( _i.e.,_ Iterative CoT and Iterative ReAct) without our designed deliberations underperform basic solutions. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning_images/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning.pdf-0009-00.png)


**----- Start of picture text -----**<br>
CMA Algebra Prealgebra<br>74 60<br>72 DELI 58<br>Iter. CoT 58<br>70<br>Iter. ReAct 56<br>68 Ret. CoT 56<br>66 ReAct 54<br>64 54<br>62 52<br>60 52<br>50<br>58<br>50<br>0 1 2 3 0 1 2 3 0 1 2 3<br>**----- End of picture text -----**<br>


Figure 4: Accuracy of different methods w.r.t. the maximum number of iteration turns. 

## **4.2 Analysis** 

**Combining with Existing CoT methods.** In DELI, we initialize a step-by-step solution by retrieving problems and solutions as exemplars. However, it is noted that DELI also supports other prompting methods to initialize solutions. We report the performance of DELI combined with different CoT methods on CMA, Algebra, and Count. & Prob. datasets. As shown in Figure 3, DELI brings improvement upon all CoT methods, which demonstrates our framework can be applied to various CoT prompting methods. In particular, DELI can further boost the performance of the existing iterative method PHP, which shows that incorporating fine-grained interfaces to assist reasoning can find and fix errors that are difficult to correct with CoT, such as complex calculation errors. 

**Impact of Iterative Deliberation Turns.** We evaluate the accuracy of DELI at different maximum numbers of iteration turns on the CMA, Algebra, and Prealgebra. To validate the effectiveness of our designed two-stage deliberation, we also evaluate two iterative variants, _i.e.,_ Iterative CoT and Iterative ReAct. As shown in Figure 4, the accuracy of DELI overall increases and eventually converges as the maximum number of iteration turns increases. As a comparison, iterative variants overall do not lead to accuracy gains, even leading to a drop, which indicates that is difficult for LLM to directly discover and correct errors based on existing solutions by eliciting their comments. In DELI, the two-stage deliberation effectively reduces model calculation errors, and considers two different perspectives of solutions to a refined solution, thus gradually improving accuracy as iterations proceed. 

## **Evaluating Intermediate Reasoning Steps.** 

We evaluate intermediate reasoning steps from different prompting methods with our proposed metrics ExpAcc and Fail@where on 128 challenging problems from CARP that are incorrectly answered by all evaluated methods. As shown in Table 6, DELI achieves better ExpAcc than CoT and ReAct, which indicates that the method derives more correct intermediate results on the challenging subset. Besides, Fail@where shows that DELI is less declined to generate completely wrong solutions, and has a larger 

Table 6: ExpAcc and Fail@where on a challenging subset of CARP. 

|**Methods ExpAcc**|**Fail@where**|
|---|---|
||**frst**<br>**middle**<br>**last**|
|CoT<br>13.91<br>ReAct<br>12.58<br>DELI<br>**18.90**|67.97<br>22.65<br>9.38<br>66.41<br>29.69<br>3.91<br>60.16<br>25.78<br>14.06|



percentage of near-correct solutions. Instead of completing solutions in one go, DELI can fix errors as much as possible in the reasoning process by iterative deliberation, leading to better intermediate reasoning progress even in incorrect solutions. 

## **5 Conclusion** 

In this paper, we proposed CARP, a computation-intensive algebra problem dataset with formulated annotation of intermediate reasoning steps for systematically evaluating LLMs in tools manipulation and math reasoning. Based on the experiments in CARP, we found that popular LLMs with chain-ofthough prompting can not solve over half of the problems in CARP, and they are more likely to make 

9 

mistakes in early steps, leading to wrong answers. To alleviate it, we proposed DELI, a new approach that can deliberate the intermediate reasoning steps with interfaces of tools. DELI incorporated two iterative deliberation procedures to check and refine the intermediate reasoning steps of the generated step-by-step solution, from the perspectives of tool manipulation and natural language reasoning. To verify the effectiveness of our approach, we conducted extensive experiments on CARP and 6 other computation-intensive math reasoning datasets. Experimental results have shown that DELI outperforms baselines and can boost the performance of various CoT prompting methods. 

## **References** 

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _NeurIPS_ , 2020. 

- [2] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. _ArXiv_ , abs/2203.02155, 2022. 

- [3] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Z. Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jianyun Nie, and Ji rong Wen. A survey of large language models. _ArXiv_ , abs/2303.18223, 2023. 

- [4] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _ArXiv_ , abs/2201.11903, 2022. 

- [5] OpenAI. Gpt-4 technical report. _ArXiv_ , abs/2303.08774, 2023. 

- [6] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied Sanosi Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _ArXiv_ , abs/2304.06364, 2023. 

- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- [8] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _ArXiv_ , abs/2103.03874, 2021. 

- [9] Z. Chen, Kun Zhou, Beichen Zhang, Zheng Gong, Wayne Xin Zhao, and Ji rong Wen. Chatcot: Tool-augmented chain-of-thought reasoning on chat-based large language models. 2023. 

- [10] Jing Qian, Hong Wang, Zekun Li, Shiyang Li, and Xifeng Yan. Limitations of language models in arithmetic and symbolic induction. _ArXiv_ , abs/2208.05051, 2022. 

- [11] Pan Lu, Liang Qiu, Wenhao Yu, Sean Welleck, and Kai-Wei Chang. A survey of deep learning for mathematical reasoning. _ArXiv_ , abs/2212.10535, 2022. 

- [12] Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Christian Petersen, Alexis Chevalier, and Julius Berner. Mathematical capabilities of chatgpt. _ArXiv_ , abs/2301.13867, 2023. 

- [13] Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, and Songfang Huang. How well do large language models perform in arithmetic tasks? _ArXiv_ , abs/2304.02015, 2023. 

- [14] Wayne Xin Zhao, Kun Zhou, Zheng Gong, Beichen Zhang, Yuanhang Zhou, Jing Sha, Zhigang Chen, Shijin Wang, Cong Liu, and Ji rong Wen. Jiuzhang: A chinese pre-trained language model for mathematical problem understanding. _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , 2022. 

10 

- [15] Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. _arXiv preprint arXiv:2205.12255_ , 2022. 

- [16] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. _arXiv preprint arXiv:2211.10435_ , 2022. 

- [17] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _ArXiv_ , abs/2211.12588, 2022. 

- [18] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. _arXiv preprint arXiv:2302.04761_ , 2023. 

- [19] QING LYU, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. _ArXiv_ , abs/2301.13379, 2023. 

- [20] Joy He-Yueya, Gabriel Poesia, Rose E. Wang, and Noah D. Goodman. Solving math word problems by combining language models with symbolic solvers. _ArXiv_ , abs/2304.09102, 2023. 

- [21] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. _ArXiv_ , abs/2210.03350, 2022. 

- [22] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _ArXiv_ , abs/2210.03629, 2022. 

- [23] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji rong Wen. Structgpt: A general framework for large language model to reason over structured data. _ArXiv_ , abs/2305.09645, 2023. 

- [24] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. _ArXiv_ , abs/2305.11738, 2023. 

- [25] Shima Imani, Liang Du, and H. Shrivastava. Mathprompter: Mathematical reasoning using large language models. _ArXiv_ , abs/2303.05398, 2023. 

- [26] Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In _EMNLP_ , pages 523–533, 2014. 

- [27] Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena Dumas Ang. Parsing algebraic word problems into equations. _Transactions of the Association for Computational Linguistics_ , 3:585–597, 2015. 

- [28] Subhro Roy and Dan Roth. Solving general arithmetic word problems. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pages 1743– 1752, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1202. URL `https://aclanthology.org/D15-1202` . 

- [29] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 158–167, 2017. 

- [30] Arkil Patel, S. Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In _North American Chapter of the Association for Computational Linguistics_ , 2021. 

- [31] Jahring Jahring, N. Nasruddin, and Ida Farida. The effectiveness of mind mapping learning models based on contextual learning on mathematical problem solving ability. 2020. 

11 

- [32] Chia-Hui Chang, Mohammed Kayed, Moheb R. Girgis, and Khaled F. Shaalan. A survey of web information extraction systems. _IEEE Transactions on Knowledge and Data Engineering_ , 18:1411–1428, 2006. 

- [33] Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondrej Certík, Sergey B. Kirpichev, Matthew Rocklin, Amit Kumar, Sergiu Ivanov, Jason Keith Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger, Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J. Curry, Andy R. Terrel, Stˇepán Rouka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, and Anthony M. Scopatz. Sympy: Symbolic computing in python. _PeerJ Prepr._ , 4:e2083, 2017. 

- [34] Stephen E. Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond. _Found. Trends Inf. Retr._ , 3:333–389, 2009. 

- [35] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. _ArXiv_ , abs/2104.08821, 2021. 

- [36] Yiduo Guo, Yaobo Liang, Chenfei Wu, Wenshan Wu, Dongyan Zhao, and Nan Duan. Learning to program with natural language. _ArXiv_ , abs/2304.10464, 2023. 

- [37] Yao Fu, Hao-Chun Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. _ArXiv_ , abs/2210.00720, 2022. 

- [38] Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. _ArXiv_ , abs/2304.09797, 2023. 

12 

## **A Interface Definition** 

We provide interface definitions of our tools in Table 7. 

Table 7: Interface definitions of tools. Num. Comp., Eq. Solving, and Expr. Trans. refer to numerical computation, equation solving, and expression transformation, respectively. 

|**Category**|**Interface**|**Description**|
|---|---|---|
|Num. Comp.|_calculate_(_e_)_→v_<br>_substitute_(_e_, _{c}_)_→v_|Calculate the value_v_of_e_.<br>Substitute the contextual conditions _{c}_into_e_.|
||_solve_eq_(_e_)_→, {e′}_|Solve the equation_e_to get the solution set_{e′}_.|
||_solve_ineq_(_e_)_→{e′}_|Solve the inequation_e_to get the solution set_{e′}_.|
|Eq. Solving|_solve_multi_eq_(_{e}_)_→{e′}_|Solve the system of equations to get the solution|
|||set_{e′}_.|
||_solve_multi_ineq_(_{e}_)|Solve the system of inequations to get the solution|
||_→{e′}_|set_{e′}_.|
||_partial_solve_(_e_,_u_)_→{e′}_|Solve the equation_e_assuming that_u_is an unknown|
|||toget the solution set _{e′}_.|
||_expand_(_e_)_→e′_|Expand_e_to get_e′_.|
|Expr. Trans.|_factor_(_e_)_→e′_<br>_collect_(_e_,_x_)_→e′_|Factorize_e_to get_e′_.<br>Collect_e_based on the symbol_x_.|
||_complete_the_square_(_e_)_→e′_|Complete the square of_e_toget_e′_|
|Thinking|_think_(_l_)_→l′_|Draw a conclusion_l′_ based on the thought_l_.|



## **B DELI Algorithem** 

The process of DELI is illustrated in Algorithm 1. 

**Algorithm 1** DELI algorithm **Require:** input problem _p_ , retrieval module _R_ (optional), interfaces of tools _{T }_ 1: _CR ← R_ ( _p_ ) _▷_ Retrieval-augmented solution initialization 2: _s_[(0)] _n ←_ init( _CR, p_ ) 3: **for** iteration _i ∈_ 1 _...M_ **do** 4: _st_[(] _[i]_[)] _←_ invoke( _p, sn_[(] _[i][−]_[1)] _, {T }_ ) _▷_ Deliberation with tool interfaces 5: _sn_[(] _[i]_[)] _[←]_[integrate][(] _[p, s] t_[(] _[i]_[)] _[, s] n_[(] _[i][−]_[1)] ) _▷_ Deliberation with chain of thought 6: **if** equal_answer( _s_[(] _t[i]_[)] _[, s] n_[(] _[i]_[)][)] **[ then]** _▷_ Early stopping 7: break 8: **else if** _i ̸_ = 0 **and** equal( _s_[(] _t[i][−]_[1)] _, s_[(] _t[i]_[)][)] **[ and]**[ equal][(] _[s]_[(] _n[i][−]_[1)] _, sn_[(] _[i]_[)][)] **[then]** 9: break 10: **end if** 11: **end for** 12: **return** Last _st_[(] _[i]_[)] 

## **C Baselines** 

We compare our proposed DELI with several competitive prompting methods for LLMs. For all the methods, we implement them on ChatGPT. 

_• CoT prompting methods._ We test three variants of the CoT prompting with different in-context exemplars. **Random CoT** [4] randomly selects exemplars from the training set. **Complex CoT** [37] samples the most complex problems and their solutions as exemplars. **Retrieval CoT** retrieves the most relevant problems and solutions from the training set as exemplars. 

_• Tool-augmented prompting methods._ We select two tool-augmented prompting methods and implement them with our proposed interfaces on tools. **PAL** [16] converts the reasoning process into 

13 

a Python program and executes it to get the answer. **ReAct** [22] invokes interfaces immediately in the reasoning process when necessary. 

_• Iterative prompting methods._ We also compare our approach with existing iterative prompting methods and our variants. **Learning to Program (LP)** [36] aims to iteratively learn solutions from training sets to guide LLMs in solving similar problems based on in-context learning. **ProgressiveHint Prompting (PHP)** [38] iteratively utilizes previous answers as hints to progressive guide LLMs generating CoT solutions. As a variant of our framework, **Iterative CoT** integrates the existing CoT solution and self-generated judgment into a refined CoT solution. Similarly, **Iterative ReAct** aims to generate better interface invocations by reviewing existing interface invocations and self-generated judgment. 

## **D Implementation Details** 

We employ OpenAI `gpt-3.5-turbo` as the solver and reasoning tool and implement the computation tool based on SymPy [33]. We set the temperature to 0 and top_p to 1 for determined outputs. To retrieve similar problems, we train a sentence embedding model following SimCSE [35] to index MATH datasets and employ BM25 algorithm for the CARP dataset. The maximum number of iteration turns is set to 3 for all datasets. For each dataset, we specify the descriptions of interfaces that may be useful to solve the problems in prompts. 

We initialize the solution with Retrieval CoT in most datasets. For GK-Cloze and SAT-Math, we initialize the solution with Random CoT, since these datasets only provide few-shot exemplars but not training sets. Following the settings in Zheng et al. [38], the initial solution of PHP is from Complex CoT in subsets of MATH (Algebra, Prealgebra, CP, NT), while using the same initial solution as DELI in other datasets. 

## **E Case Study** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning_images/nips_2023_Evaluating_and_Improving_Tool-Augmented_Computation-Intensive_Math_Reasoning.pdf-0014-06.png)


**----- Start of picture text -----**<br>
⓪ Retrieval-Augmented Solution Intialization Baseline: ReAct<br>Question:  (Examples)If (𝑥+ 1)(𝑥+ 𝑞) …… ……. You can use the following functions:……<br>Answer:  (𝑥+ 1)(𝑥+ 𝑞) = 𝑥 [!] + 𝑞𝑥+ 𝑥+ 𝑞= 𝑥 [!] + (𝑞+ 1)𝑥+ 𝑞 Question:  Knowing that the result of (𝑥−2)(1 −𝑘𝑥) −(2𝑥−<br>……. 3)(2𝑥+ 3)<br>Question:  3)(2𝑥+ 3) does not contain Knowing that the result of 𝑥 in the primary form , then (𝑥−2)(1 −𝑘𝑥) −(2𝑥−𝑘 = ____ ? …… Action: Output: collect( 2𝑘𝑥+ 𝑥 [!] 𝑥−2−𝑘−41 −𝑘𝑥−+ 𝑥+ 7 2𝑥−3 2𝑥+ 3 , x [!] )<br>Answer:  since the result does not contain the primary term of 𝑥−2 1 −𝑘𝑥− 2𝑥−3 2𝑥+ 3 = −4𝑥 [!] −𝑘𝑥𝑥 [!] , we get + 5𝑥−15 ,  Action:Output:  solve_eq( [𝑘= − #! []] 2𝑘𝑥+ 𝑥 [!] −𝑘−4 + 𝑥+ 7)<br>5 = 0. Hence no solution. Final Answer:  − #!<br>① Deliberation with Tool Interfaces ② Integration with Chain of Thought<br>……. You can use the following functions:<br>expand …… (expression: str) -> new_expression: str: Expand the  Question:  Knowing that the result of (𝑥−2)(1 −𝑘𝑥) −(2𝑥−3)(2𝑥+ 3)<br>expression into a polynomial  does not contain 𝑥 in the primary form , then 𝑘 = ____ ?<br>collect (expression: str, symbol: str) -> new_expression: str ……<br>…… Given Solution:<br>𝑥−2 1 −𝑘𝑥− 2𝑥−3 2𝑥+ 3 = −4𝑥 [!] −𝑘𝑥 [!] + 5𝑥−15 … …<br>Question:  Knowing that the result of (𝑥−2)(1 −𝑘𝑥) −(2𝑥−<br>3)(2𝑥+ 3) does not contain 𝑥 in the primary form , then 𝑘 =  Verification:<br>____ ? … …<br>Trial:  since the result does not contain the primary term of 𝑥−2 1 −𝑘𝑥− 2𝑥−3 2𝑥+ 3 = −4𝑥 [!] −𝑘𝑥 [!] + 5𝑥−15𝑥 , we get  ,  Action: solve_eq(Output: 𝑘= − "! 2𝑘+ 1 = 0)<br>5 = 0 Action: . Hence  expand(no solution𝑥−2 1 −𝑘𝑥−.  2𝑥−3 2𝑥+ 3 ) Final Answer: − "!<br>Output : −𝑘𝑥 [!] + 2𝑘𝑥−4𝑥 [!] + 𝑥+ 7<br>Action: Output:  collect(𝑥 [!] (−𝑘−4) + 𝑥(2𝑘+ 1) + 7−𝑘𝑥 [!] + 2𝑘𝑥−4𝑥 [!] + 𝑥+ 7, 𝑥) Revised Solution: Expanding (𝑥−2)(1 −𝑘𝑥) −(2𝑥−3)(2𝑥+ 3) gives 𝑥 [!] (−𝑘−4) +<br>Action:  solve_eq(2𝑘+ 1 = 0) 𝑥(2𝑘+ 1) + 7, since the result does not contain the primary<br>OutputFinal Answer:  : 𝑘= − "!− "! term of , since the result does not contain the primary term of 𝑥, we get 2𝑘+ 1 = 0,which gives 𝑘= − "! [.] [ The answer is: ][−] "!<br>Iterate  ①② when the answers are inconsistent<br>**----- End of picture text -----**<br>


Figure 5: Case study of our method with baselines on the CARP dataset. The case is translated into English. 

14 

To better present the process of DELI, we provide a case study that shows the solving process of DELI on CARP, which is shown in Figure 5. We also report the solution of Retrieval CoT and ReAct in the figure. It is noted that the solution of Retrieval CoT is also the initial solution in our framework. 

First, both Retrieval CoT and ReAct make minor mistakes in the solving process. Although following the correct solving idea from relevant solutions, Retrieval CoT struggles with expanding the expression, leading to an incorrect intermediate result. ReAct fails at understanding the condition _the expression does not contain the primary form of x_ , thus collecting the expression according to a wrong term _x_[2] . Therefore, both CoT and ReAct can not solve the case individually due to the challenges of computations and reasoning. 

Our method iterates over the existing solutions. In deliberation with tool interfaces, the model reviews the existing solution, and invokes interfaces based on the ideas therein. In this case, the model invokes interfaces _expand_ and _collect_ in a row to get the correct expanded expression with the help of tools. Then, the model solves the equation derived from the expanded expression and gets the correct answer. 

In deliberation with chain of thought, the model reviews both natural language and tool manipulation solutions from the previous iteration and generates a revised CoT solution, which fixes the computation error in the original CoT solution according to interface invocations. In this case, due to the consistent answers between the revised CoT solution and the previous ReAct solution, the iteration terminates. In general cases, the iteration continues until solutions converge or the answer is consistent, or the maximum number of iterations is reached. 

## **F Prompts for Two-Stage Deliberation** 

We list the prompts for two-stage deliberation on CARP. The prompts are translated into English. 

Listing 1: Prompt for Deliberation with Tool Interfaces. 

- `You are ChatGPT, a math problem solver equipped with multiple functions to tackle various math problems. While you may have access to existing problem-solving processes, there is a possibility of errors. Therefore, you need to learn the correct approaches to solve problems more efficiently. You can use the following functions:` 

- `calculate(expression: str) -> new_expression: str: Calculate the value of the expression and return it as a string. For example, calculate("34 * 2") -> "68".` 

- `solve_eq(expression: str) -> new_expressions: list: Solve the equation expression and return the result as a list. For example, solve_eq("3 x + 4 = 1") -> ["x = -1"].` 

- `solve_ineq(expression: str) -> new_expression: str: Solve the inequality expression and return the result as a string. For example, solve_ineq("3 x + 4 < 1") -> "x < -1".` 

- `solve_multi_eq(expressions: list) -> new_expressions: dict: Solve the system of equations given by the list of expressions and return the result as a dictionary. For example, solve_multi_eq(["x + y = 2", "x - 2 y = -7"]) -> {"x": ["x = -1"], "y": ["y = 3"]}.` 

- `solve_multi_ineq(expressions: list) -> new_expression: str: Solve the system of inequalities given by the list of expressions and return the result as a string. For example, solve_multi_ineq(["x \le 2", "x \le -7"]) -> "x \le -7".` 

- `substitute(expression: str, conditions: list[str]) -> new_expression: str: Substitute the contextual conditions in the list into the expression and return the result. For example, substitute("3 x + 4", ["x = 1"]) -> "7".` 

- `expand(expression: str) -> new_expression: str: Expand the expression into a polynomial. For example, expand("(x + 1) ^ 2") -> "x ^ 2 + 2x + 1"` 

- `factor(expression: str) -> new_expression: str: Factorize the polynomial. For example, factor("x ^ 2 + 2x + 1") -> "(x + 1) ^ 2"` 

- `collect(expression: str, symbol: str) -> new_expression: str: Collect the coefficients of the corresponding powers according to the given symbol. For example, collect("a x - 5 a + x ^ { 2 } - 5 x", "x") -> "- 5 a + x ^ { 2 } + x ( a - 5 )"` 

- `partial_solve(expression: str, symbol: str) -> new_expression: str: Let the given symbol be the unknown variable and solve the linear equation expression with` 

15 

   - `one variable. For example, partial_solve("x + 3 y - 3 = 0", "x) -> "x = - 3 y + 3"` 

- `think(thought: str) -> conclusion: str: Generate new conclusions based on natural language description thought. Think should only be called when the above functions are not applicable. For example, think("\sqrt{x-8} the expression inside the root is always greater than or equal to 0") -> "x-8\\ge0"` 

- `To use ChatGPT, simply provide a mathematical problem or question in LaTeX format. You can use any of the above functions to help solve the problem. Please follow the following format:` 

- `Question: The input question you must answer. This appears only once.` 

- `Trial: The problem-solving approach that can be referred to. It may contain errors, you can refer to the correct part in it.` 

- `Action: A function call, which should be one of the mentioned functions with arguments. You must only call one function in one Action.` 

- `Output: The result of the action. Every Action must be immediately followed by one and only one Output.` 

- `... (This Action/Output cycle can repeat N times.)` 

- `Final Answer: The final answer to the original input question. The answer should be numerical or LaTeX math expression. Do not use natural language in the answer` 

## Listing 2: Prompt for Deliberation with Chain of Thought. 

- `You are ChatGPT, a mathematical problem solver equipped with multiple functions for solving mathematical problems. You have access to both natural language problem solving processes and formal problem solving processes, but there may be` 

- `errors within them. You need to learn the correct methods in order to better solve problems. You can use the following functions:` 

- `calculate(expression: str) -> new_expression: str: Calculate the value of the expression and return it as a string. For example, calculate("34 * 2") -> "68".` 

- `solve_eq(expression: str) -> new_expressions: list: Solve the equation expression and return the result as a list. For example, solve_eq("3 x + 4 = 1") -> ["x = -1"].` 

- `solve_ineq(expression: str) -> new_expression: str: Solve the inequality expression and return the result as a string. For example, solve_ineq("3 x + 4 < 1") -> "x < -1".` 

- `solve_multi_eq(expressions: list) -> new_expressions: dict: Solve the system of equations given by the list of expressions and return the result as a dictionary. For example, solve_multi_eq(["x + y = 2", "x - 2 y = -7"]) -> {"x": ["x = -1"], "y": ["y = 3"]}.` 

- `solve_multi_ineq(expressions: list) -> new_expression: str: Solve the system of inequalities given by the list of expressions and return the result as a string. For example, solve_multi_ineq(["x \le 2", "x \le -7"]) -> "x \le -7".` 

- `substitute(expression: str, conditions: list[str]) -> new_expression: str: Substitute the contextual conditions in the list into the expression and return the result. For example, substitute("3 x + 4", ["x = 1"]) -> "7".` 

- `expand(expression: str) -> new_expression: str: Expand the expression into a polynomial. For example, expand("(x + 1) ^ 2") -> "x ^ 2 + 2x + 1"` 

- `factor(expression: str) -> new_expression: str: Factorize the polynomial. For example, factor("x ^ 2 + 2x + 1") -> "(x + 1) ^ 2"` 

- `collect(expression: str, symbol: str) -> new_expression: str: Collect the coefficients of the corresponding powers according to the given symbol. For example, collect("a x - 5 a + x ^ { 2 } - 5 x", "x") -> "- 5 a + x ^ { 2 } + x ( a - 5 )"` 

- `partial_solve(expression: str, symbol: str) -> new_expression: str: Let the given symbol be the unknown variable and solve the linear equation expression with one variable. For example, partial_solve("x + 3 y - 3 = 0", "x) -> "x = - 3 y + 3"` 

- `think(thought: str) -> conclusion: str: Generate new conclusions based on natural language description thought. Think should only be called when the above functions are not applicable. For example, think("\sqrt{x-8} the expression inside the root is always greater than or equal to 0") -> "x-8\\ge0"` 

16 

```
Followthisformat:
```

```
‘‘‘
```

```
Question:
```

```
Theinputquestionthatyoumustanswer.Itappearsonlyonce.
```

```
GivenSolution:
```

```
Anaturallanguagesolutionthatcanbeusedasareference,whichmaycontain
errors.Youcanrefertothecorrectideasinit.
```

```
Verification:Transformtheoriginalsolutionintoaverificationprocessthatuses
functions,correctsanycomputationalerrors,andsimplifiestheprocess.
Action:Afunctioncall,whichmustbeoneofthefunctionsmentionedaboveand
includeparameters.YoucanonlycallonefunctioninanAction.
```

```
Output:TheresultofanAction.EachActionmusthaveoneandonlyoneOutput
followingit.
```

```
(Action/Outputcanberepeatedanynumberoftimes...)
```

```
FinalAnswer:Theultimatesolutiontotheoriginalinputproblem.
```

```
Revisethegivensolutionbasedontheverificationprocess:
```

```
Revisetheoriginalsolutionbasedonthecomputedresultintheverification
process.Ifthecomputedresultintheverificationprocessdiffersfromthe
computedresultintheoriginalsolution,thecomputedresultinthe
verificationprocessmustbeusedasthestandard.
```

```
‘‘‘
```

17 

