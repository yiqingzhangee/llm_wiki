Published as a conference paper at ICLR 2025 

# UNVEILING THE MAGIC OF CODE REASONING THROUGH REFLECTIVE HYPOTHESIS DECOMPOSITION AND AMENDMENT 

**Yuze Zhao**[1] **, Tianyun Ji**[1] _[,][∗]_ **, Wenjun Feng**[1] _[,][∗]_ **, Zhenya Huang**[1] _[,]_[2] _[,][†]_ **, Qi Liu**[1] _[,]_[2] **, Zhiding Liu**[1] **, Yixiao Ma**[1] **, Kai Zhang**[1] **, Enhong Chen**[1] 

1State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China 

2Institute of Artificial Intelligence, Hefei Comprehensive National Science Center yuzezhao@mail.ustc.edu.cn huangzhy@ustc.edu.cn 

## ABSTRACT 

The reasoning abilities are one of the most enigmatic and captivating aspects of large language models (LLMs). Numerous studies are dedicated to exploring and expanding the boundaries of this reasoning capability. However, tasks that embody both reasoning and recall characteristics are often overlooked. In this paper, we introduce such a novel task, **code reasoning** , to provide a new perspective for the reasoning abilities of LLMs. We summarize three meta-benchmarks based on established forms of logical reasoning, and instantiate these into eight specific benchmark tasks. Our testing on these benchmarks reveals that LLMs continue to struggle with identifying satisfactory reasoning pathways. Additionally, we present a new pathway exploration pipeline inspired by human intricate problemsolving methods. This **R** eflective **H** ypothesis **D** ecomposition and **A** mendment ( **RHDA** ) pipeline consists of the following iterative steps: (1) Proposing potential hypotheses based on observations and decomposing them; (2) Utilizing tools to validate hypotheses and reflection outcomes; (3) Revising hypothesis in light of observations. Our approach effectively mitigates logical chain collapses arising from forgetting or hallucination issues in multi-step reasoning, resulting in performance gains of up to 3 _×_ . Finally, we expand this pipeline by applying it to simulate complex household tasks in real-world scenarios, specifically in VirtualHome, enhancing the handling of failure cases. We release our code and all of results at https://github.com/TnTWoW/code_reasoning. 

## 1 INTRODUCTION 

Large Language Models (LLMs), which are trained on billions of tokens, have demonstrated impressive reasoning abilities in complex tasks (Brown et al., 2020; Wei et al., 2022; Kojima et al., 2022; OpenAI, 2023). However, it is evident that as potential fuzzy retrieval systems or parameterized knowledge compression systems (Xie et al., 2021), LLMs perform better on System 1 tasks than on System 2 tasks (Kahneman, 2011; Bengio, 2019; Yao et al., 2023a; Weston & Sukhbaatar, 2023; Liu et al., 2023a). Specifically, LLMs excel in intuitive memory retrieval tasks, but continue to face significant challenges with tasks requiring rational reasoning (Kambhampati, 2024). 

From the perspective of human cognitive psychology, **reasoning can be viewed as a process of memory retrieval** , in which people retrieve relevant information from memory and use it to make inferences (Kyllonen & Christal, 1990; S¨uß et al., 2002; Hayes et al., 2014; Feeney & Thompson, 2014; Hardman & Cowan, 2015). For example, Haidt (2001) proposed that when individuals engage in moral reasoning, they typically draw on their prior knowledge from social and cultural contexts. Similarly, studies involving animal lesions and human neuroimaging have confirmed that the hippocampus, which is primarily associated with memory, also plays a crucial role in reasoning 

> *Equal contribution 

> †Corresponding author 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0002-01.png)


**----- Start of picture text -----**<br>
SYSTEM 1 SYSTEM 2 LOGICAL<br>EMOTIONAL<br>REFLECTIVE<br>INTUITIVE<br>RATIONAL<br>INSTINCTUAL<br>DELIBERATE<br>UNCONSCIOUS<br>FAST / AUTOMATIC SLOW / EFFORTFUL<br>Recall Reasoning<br>Input Output '''python mccl<br>文 def fn():```Two groups of blocks  szeafioct<br>A move towards the center…``` szeamccl binbin<br>'''python fioct bin<br>Translation Summarization Code Reasoning Symbolic Arithmetic<br>**----- End of picture text -----**<br>


Figure 1: Code reasoning is a category of tasks that incorporates logical reasoning into code, aiming to solve programming problems through logical reasoning. These tasks require a balance between background knowledge and thinking span, placing greater emphasis on the collaborative functioning of both System 1 and System 2 thinking. 

abilities (Zeithamova et al., 2012). Therefore, memory and reasoning are interdependent, with considerable overlap between the two, rendering the distinction between them somewhat arbitrary (Heit et al., 2012; Liu et al., 2023b). 

We believe that, similar to humans (Strachan et al., 2024; Liu et al., 2024b; Lin et al., 2024), LLMs do not exhibit a clear boundary between memory and reasoning (Schaeffer et al., 2024; Razeghi et al., 2022). However, tasks that lie at this intersection are often overlooked in research. Here, we propose a novel task to explore the capability boundaries of LLMs: **Code Reasoning** . Code reasoning encompasses a category of tasks that demonstrate logical reasoning through code and address problems in a systematic manner. As illustrated in Figure 1, we position some tasks along an axis that reflects 1) the degree of reliance on prior knowledge (Recall) and 2) the extent to which prior knowledge is applied to the current context (Reasoning). We position the code reasoning task between memory and reasoning. On one hand, the highly structured nature of code requires the model to learn syntax from pre-training data, enabling it to recall relevant information during solving a problem. On the other hand, generating code solutions necessitates the model’s understanding of the problem and context, involving reasoning to produce appropriate solutions. 

In this paper, we introduce code reasoning, a task that formalizes reasoning steps into a programming language and offloads the computation process to the compiler. To explore different aspects of code reasoning, we summarize three meta-benchmarks based on existing forms of logical reasoning: inductive code reasoning, deductive code reasoning, and abductive code reasoning. 

Inductive code reasoning involves deriving broad generalizations from a series of observations, demonstrating the ability to infer rules from examples and generate programs to meet input-output mapping. Deductive code reasoning starts from premises and derives valid conclusions, focusing on the model’s capacity to understand a program’s intermediate states and reasoning step by step. Abductive code reasoning seeks the simplest and most likely explanation based on a set of observations, highlighting the model’s ability to abstractly understand a function’s purpose. 

We concretize these three meta-benchmarks into eight specific benchmarks. Based on these eight benchmarks, we evaluate the performance of existing models in code reasoning. Due to data sparsity, we find that current state-of-the-art LLMs still struggle to achieve satisfactory results in solving such problems. To enhance the reasoning process, we implement a **R** eflective **H** ypothesis **D** ecomposition and **A** mendment ( **RHDA** ) pipeline. This pipeline is iterative, encompassing hypothesis decomposition, execution verification, and amendment submission. Specifically, we first guide the LLM to formulate initial hypotheses based on complex observations and decompose these into sub-hypotheses. These sub-hypotheses are then compiled into executable functions through a translator, enabling direct application to the observations, followed by validation using external tools. Subsequently, based on the execution results and observations, the LLM submits amendments to reflect on and refine the issues within the sub-hypotheses. 

2 

Published as a conference paper at ICLR 2025 

Our experimental results indicate that RHDA methods effectively mitigate reasoning failures caused by data sparsity. With the same or even lower overhead, this method achieved performance improvements of up to three times compared to baseline methods. Finally, we extend this pipeline to complex, simulated real-world household tasks VirtualHome (Puig et al., 2018; 2020), guiding the LLM to complete a series of intricate operations. 

## 2 META-BENCHMARK 

We describe the general process of code reasoning as the transformation from Input _I_ and Program _P P_ to Output _O_ , represented as _I −→O_ . Inductive code reasoning is concretized as the Programming by Example (PBE) task. In this task, a neural program synthesis model _M_ searches the execution space to find a program that best satisfies all given input-output specifications. We donate this metabenchmark as _M_ ( _I, O_ ) _→ P_[�] . Deductive code reasoning is exemplified in tasks that simulate the program execution process. In this task, a neural simulation compiler model _M_ tracks the program’s execution and records intermediate states, gradually deriving the final valid output. We denote this meta-benchmark as _M_ ( _I, P_ ) _→ O_[�] . Abductive code reasoning is concretized as input prediction tasks. This task requires the neural understanding model _M_ to form an abstract-level understanding of function’s behavior and perform abductive inference based on the given program and output. We represent this meta-benchmark as _M_ ( _O, P_ ) _→ I_[�] . The details of the benchmarks are provided in the Appendix C. 

## 2.1 INDUCTIVE CODE REASONING 

Inductive code reasoning can be represented as _M_ ( _I, O_ ) _→ P_[�] and is concretized as a PBE task (Qiu et al., 2024; Shi et al., 2024). PBE is a program synthesis task designed to help end-users, particularly non-programmers, create scripts to automate repetitive tasks (Gulwani, 2016). Based on inputoutput specifications, PBE systems can synthesize a program in either a general-purpose language (GPL) or a domain-specific language (DSL). Inductive code reasoning encompasses four challenging PBE tasks, two of which are GPL tasks: List Function (Rule, 2020) and MiniARC (Kim et al., 2022), while the other two are DSL tasks: RobustFill (Devlin et al., 2017) and DeepCoder (Balog et al., 2016). GPL tasks are relatively complex, allowing the model to solve problems in a more flexible manner. In contrast, DSL tasks require the model to quickly learn the syntax of DSL through few-shot learning and address relatively simpler problems. 

**List Function.** The List Function task was originally designed to investigate how humans learn the concept of computable functions that map lists to lists. Given input and output specifications in the form of lists, the model generates GPL rules that conform to these specifications. For example, with an input specification of [2, 4, 8, 10] and an output specification of [3, 5, 9, 11], we expect the resulting rule to be lambda x : x + 1[1] . 

**MiniARC.** MiniARC is a compressed 5x5 version of the Abstraction and Reasoning Corpus (Chollet, 2019; Moskvichev et al., 2023), designed to assess imaginative and reasoning abilities. MiniARC balances the length of the input-output pairs with the difficulty of the problems. The specifications are 5x5 2D grids, where the numbers represent blocks of specific colors. The model must find valid problem-solving paths (such as color swapping, row flipping) to achieve the transformation from input to output. 

**RobustFill.** RobustFill is a string manipulation task where the model is expected to perform a combination of atomic operations, such as extracting a substring from position _k_ 1 to _k_ 2 using _SubString_ ( _k_ 1 _, k_ 1), to achieve generalization. As an example, a program ToCase(Lower, SubStr(1,3)) converts full month names (January, April) to their abbreviations (jan, apr). 

**DeepCoder.** The DeepCoder task involves using DSL to perform operations on integer lists. In DeepCoder, each line represents a subroutine that performs atomic operations on previous variables and assigns the results to new variables. The result of the final line is the program’s out- 

> 1For conciseness while maintaining generality, we will use lambda expressions to represent a program. 

3 

Published as a conference paper at ICLR 2025 

put. For example, program a _←_ [int] | b _←_ FILTER(<0) a | c _←_ MAP(*4) b | d _←_ SORT c | e _←_ REVERSE b (where “|” denotes subroutine separator.) transforms the input [-17, -3, 4, 11, 0, -5, -9, 13, 6, 6, -8, 11] into the output [-12, -20, -32, -36, -68]. We provide detailed RobustFill and Deepcoder DSLs in Appendix A. 

## 2.2 DEDUCTIVE CODE REASONING 

Deductive code reasoning refers to the process of deriving a sound inference _O_ by reasoning from the given premise _I_ , assuming the validity of the argument _P_ . Deductive code reasoning can be instantiated as an output prediction task (Gu et al., 2024). Based on the given premise, the output prediction requires the LLM to simulate a compiler (Kim et al., 2024b), executing step by step until it arrives at a valid conclusion. For example, given a program P = lambda text, value: ’’.join(list(text) + [value]) and inputs text = ‘bcksrut’, b = ‘q’, the output prediction from LLM should be ‘bcksrutq’. 

## 2.3 ABDUCTIVE CODE REASONING 

Starting from existing facts _P_ and _O_ , deriving the most reasonable and optimal explanation _I_ is referred to as abductive code reasoning. This meta-benchmark can be framed as an input prediction task. Given the provided facts, the input prediction requires the LLM to backtrack through the program’s execution process to recover the potential inputs. In cases where multiple possible inputs exist, the model should apply Occam’s Razor and return the simplest input. For example, given a program P = lambda nums: nums + [nums[i % 2] for i in range(len(nums))] and outputs [-1, 0, 0, 1, 1, -1, 0, -1, 0, -1], the input prediction from LLM should be [-1, 0, 0, 1, 1]. 

Deductive code and abductive code reasoning can be regarded as opposite processes; therefore, we selected two identical and representative datasets, CRUXEval (Gu et al., 2024) and LiveCodeBench (Jain et al., 2024), as benchmarks to validate these two capabilities. 

**CRUXEval.** CRUXEval is a benchmark designed to evaluate code understanding and execution. Many models that achieve high scores on HumanEval (Chen et al., 2021) do not show the same level of improvement on the CRUXEval benchmark. This benchmark includes 800 functions along with their corresponding inputs and outputs. 

**LiveCodeBench.** LiveCodeBench is a dynamically updated benchmark sourced from competition platforms. Each problem is timestamped, and we selected data from October 2023 (later than GPT-4o training) to March 2024 (the most recent), ensuring there is no data leakage and thereby guaranteeing the model’s generalization performance. 

## 3 CODE REASONING WITH HYPOTHESIS DECOMPOSITION AND AMENDMENT 

We aim to generate a reliable reasoning process for problem-solving by establishing a problemsolving pathway _f_ : _X →Y_ . For a given task _τ_ and the seen specifications/observations _Xτ[s]_[,][the] pathway _f_ , should lead to a seen valid solution _Yτ[s]_[through][a][chain][of][reasoning.][We][expect][this] pathway _f_ to have sufficient generalization capabilities to handle unseen specifications/observations _Xτ[u]_[.][To this aim, we employ a process involving hypothesis decomposition, execution verification,] and amendment submission to iteratively explore and refine the reasoning pathway. We first establish an initial hypothesis _h_[0] _∈_ Σ _[∗]_ based on observations _x[s] τ[∈X] τ[ s]_[,][where][Σ] _[∗]_[is][the][closure][form] of LLM’s vocabulary. This initial hypothesis _h_[0] serves as a preliminary solution pathway to the problem. Given the complexity of many problems, we decompose the hypothesis _h_[0] into simpler sub-hypotheses _h_[0] _⇐⇒ {h_[0] _s_ 0 _[, h] s_[0] 1 _[, h] s_[0] 2 _[, ...][}]_[.][A][translator][function] _[g]_[:][Σ] _[∗][→]_[Σ] _[∗] E_[,][which][maps] the hypothesis space Σ _[∗]_ into an executable function space Σ _[∗] E_[,][is then used to ‘compiled’ the sub-] hypotheses _h_[0] into an executable function _e_[0] . This executable function is directly applicable to the observations _x[s] τ_[, allowing for the derivation of conclusions][ �] _[y] τ[s]_[, that is:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0004-11.png)


(1) 

4 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Task: Inductive  (a) Hypothesis Decomposition (b) Execute & Verify<br>Example 1 Hypothesis  Example 1<br>Executable function  𝒆<br>Decomposition<br>[ Step 1 ]: Identify the position  Translator  𝑔 #  Step 3 : Adjust the column count<br>of '5' in the input matrix. values<br>Example 2 [of '5's in each column. Step 2 ]: Count the number  adjusted_counts = []for count in column_counts: Example 2<br>[ Step 3 ]: Adjust the column  if count == 2:<br>count values. adjusted_counts.append(4)<br>[ Step 4 ]: Fill each row of the  elif adjusted_counts.append(count == 3: 2)<br>Seen Observation output grid with the adjusted column counts. else:adjusted_counts.append(0) Actual  Expected<br>Output Output<br>(c) Amendment Submission (d) Feedback * N (f) Validation<br>Polished Executable function  𝒆 Example 1 Example 3<br>Hypothesis #  Step 3 : Adjust the column count values<br>[ Step 1 ]: Identify the position  adjusted_counts = []<br>of '5' in the input matrix. for count in column_counts:<br>[ Step 2 ]: Count the number  if count == 5:<br>of '5's in each column.[ Step 3 ]: Adjust the column  𝑔 elif adjusted_counts.append(count == 3: 6) Example 2 Example 4<br>count values,  possibly by  adjusted_counts.append(2)<br>increasing or decreasing by specific amounts [output grid with the adjusted column counts. Step 4 ]: Fill each row of the . elif elif elseadjusted_counts.append(adjusted_counts.append(:adjusted_counts.append(count == count == 2:1: 140))) OutputActual  Expected Output Unseen Observation<br>**----- End of picture text -----**<br>


Figure 2: An overview of pipeline to solve code reasoning task. We decompose the hypothesis and generate executable functions step by step. After comparing the results with the seen observations and receiving feedback, we propose amendments, reflect on potential errors at each step, and generate revised hypotheses. This process is repeated until a valid problem-solving pathway is discovered. For concise expression, we show partial code snippets. 

Feedback _F_ ( _yτ[s][,]_[ �] _[y] τ[s]_[)][ is used to evaluate the conclusions drawn from the current hypothesis, guiding] the LLM to reflect on its sub-hypotheses. Through this iterative process of reflection, the model generates a new hypothesis _h_[1] for the next iteration. Finally, the problem-solving pathway _f_ is applied to unseen observations _Xτ[u]_[,][and][the][model’s][generalization][performance][is][assessed][by][measuring] its accuracy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0005-04.png)


The preceding section presents a unified framework for the hypothesis decomposition and amendment method. However, the implementation specifics differ across various tasks. In the following sections, we will introduce these task-specific variations in detail. 

**Hypothesis Decomposition.** We recognize that complex logical reasoning problems are difficult to encapsulate in a single reasonable hypothesis, which can adversely affect the performance of LLMs. Therefore, we require the LLM to decompose its hypotheses. Specifically, given an observation _x[s] τ_[,][the LLM gradually presents corresponding hypotheses step by step.][For inductive code] reasoning, _h_ 0 represents the step-by-step hypothesis of the input-to-output transformation rules. For deductive and abductive code reasoning, _h_ 0 refers to the step-by-step hypothesis regarding the functionality of the program. 

**Execution Verification.** After obtaining the hypothesis, we need to apply it to the observations. However, hypotheses are often not directly usable, so we need to convert the decomposed hypothesis into an executable function _e_ through a translator _g_ . For inductive code reasoning, the executable function is a program; for deductive and abductive code reasoning, the executable function is the predicted output and input, respectively. These three types of task are then sent to a compiler to obtain the actual execution results, and the feedback generated by the compiler is provided to the LLM to help it further refine and adjust the sub-hypotheses. 

**Amendment Submission.** During the amendment submission stage, there are no significant differences in handling the three tasks. The LLM receives validation feedback from the tools and generates amendments based on this feedback, reflecting on possible issues in the previous hypothe- 

5 

Published as a conference paper at ICLR 2025 

Table 1: RHDA method on inductive code reasoning task. _T_ refers to the maximum number of iterations. _N_ refers to the number of candidates. 

|Method|**Accuracy**<br>List Func<br>MiniARC<br>RobustFill<br>Deepcoder|**Task Accuracy**<br>List Func<br>MiniARC<br>RobustFill<br>Deepcoder|
|---|---|---|
|IO<br>PoT<br>CoC<br>SC (N=3)<br>SR (T=2)|**64.85**<br>**28.21**<br>**61.74**<br>23.78<br>44.90<br>10.90<br>37.39<br>30.90<br>42.45<br>10.90<br>31.30<br>26.39<br>52.95<br>12.31<br>46.09<br>37.85<br>51.10<br>10.26<br>41.74<br>36.81|38.00<br>13.08<br>21.74<br>10.42<br>33.60<br>8.46<br>21.74<br>19.79<br>34.40<br>4.62<br>13.04<br>13.54<br>41.20<br>9.23<br>26.09<br>26.04<br>41.60<br>8.46<br>21.74<br>25.00|
|w/o Sub-Hyp<br>w/o Amend|42.45<br>7.95<br>40.87<br>18.05<br>47.10<br>8.46<br>35.65<br>30.21|33.20<br>4.62<br>21.74<br>9.37<br>36.40<br>6.92<br>17.39<br>19.79|
|T=2, N=1<br>T=3, N=1<br>T=2, N=3|51.05<br>12.56<br>43.48<br>38.89<br>53.20<br>14.10<br>47.83<br>38.19<br>58.35<br>19.74<br>54.78<br>**43.06**|41.20<br>10.77<br>30.43<br>23.96<br>44.00<br>11.54<br>30.43<br>26.04<br>**48.80**<br>**13.85**<br>**34.78**<br>**29.17**|



ses. The reflection process involves revising each sub-hypothesis individually, forming an updated hypothesis _h_ 1 _⇐⇒{h_[1] _s_ 0 _[, h] s_[1] 1 _[, h] s_[1] 2 _[, ...][}]_[.][This process ensures that each sub-hypothesis is adjusted] to better align with the observations and validation results, gradually improving the reasoning pathway’s coherence and accuracy. 

## 4 EXPERIMENTS 

**Experimental Setup.** We utilize the latest and most advanced model, gpt-4o-2024-08-06, as the backbone LLM for all our experiments. We report the results using Llama-3.1-70B-Instruct, Qwenmax (qwen-max-2024-09-19) (Bai et al., 2023), Claude 3.5 (claude-3-5-sonnet-20240620) in Appendix B. Following the methodology of Qiu et al. (2024), we set the temperature to 0.7. We report results using several methods: input-output (IO) prompting, standard prompting, Chain of Thought (CoT) (Wei et al., 2023), Program of Thought (PoT) (Chen et al., 2023), Chain of Code (CoC) (Li et al., 2024), Self-Consistency (SC) (Wang et al., 2023c) and Self-Refine (SR) (Madaan et al., 2024), all implemented with 2-shot learning.[2] For our proposed process, we employ 0-shot prompts, allowing the LLM to explore problem-solving pathways in a more flexible manner. We provide detailed prompt templates in Appendix H. 

## 4.1 INDUCTIVE CODE REASONING 

For inductive code reasoning, we establish four baseline methods. The Input-Output (IO) prompting requires the LLM to predict outputs based on all seen observations and an unseen input. The Program of Thought (PoT) method generates and executes programs to derive outputs. The CoC method prompts the LLM to utilize pseudocode for reasoning in output prediction. The SC method builds upon PoT by sampling multiple programs and selecting the one that demonstrates optimal performance on seen observations. Furthermore, since each example may contain multiple unseen observations, we adopt the approach from (Qiu et al., 2024) to define task accuracy externally. An example is deemed passed only when all unseen observations within it pass; thus, the proportion of passed examples reflects the task accuracy. The experimental results are presented in Table 1. 

The results demonstrate that the RHDA method achieves optimal performance across four benchmarks, with task accuracy exceeding that of the second-best methods by 18.45%, 5.89%, 33.31%, and 12.02%, respectively. However, we observe that RHDA appears to underperform compared to IO prompting. This is because the IO prompt does not generate a hypothesis that satisfies all observations but instead predicts the output for a single input. A successful prediction for a single instance does not generate a hypothesis that satisfies all observations, resulting in a high prediction accuracy but a relatively low task accuracy. 

**Ablation Study.** We introduce two variants to separately validate the effectiveness of hypothesis decomposition and amendment submission. The first variant does not require the LLM to decompose 

> 2Not all methods are suitable for these three meta-benchmarks, thus we selected the most appropriate methods for each benchmark. 

6 

Published as a conference paper at ICLR 2025 

hypotheses, referred to as w/o Sub-Hyp. The second variant, termed w/o Amend, indicates that the model no longer modifies its hypotheses through reflection. The experimental results presented in Table 1 show that the performance of these two variants declined by 25.39% to 67.88% and 19.28% to 57.14%, respectively. This finding suggests that the introduction of sub-hypotheses is a critical step, as it simplifies complex problems, reducing the workload for the subsequent translator _g_ while also enabling individual amendments to each sub-hypothesis. Nonetheless, the reflection process is equally important. Our results align with previous research (Zhao et al., 2024; Olausson et al., 2024; Peng et al., 2023) indicating that rational reflection can significantly enhance performance. 

## 4.2 DEDUCTIVE CODE REASONING 

For deductive code reasoning, we select standard prompting, CoT, SC, SR and CoC as benchmark methods. The experimental results are presented in Table 2. These results indicate that the CoT and CoC methods significantly enhanced the accuracy of reasoning outcomes by guiding the model to think step-by-step about function capabilities. Our proposed method advances this further, achieving optimal performance with a single round of amendments, resulting in an improvement of up to 104.37% compared with baseline method. A horizontal comparison of the two datasets revealed that, due to the absence of LiveCodeBench data in internet corpora, the performance with standard 

Table 2: RHDA method on deductive code reasoning task. _T_ refers to the maximum number of iterations. _N_ refers to the number of candidates. 

||Standard<br>CoT<br>SC (N=3)<br>SR (T=2)<br>CoC<br>w/o Amend<br>T=2, N=1|CRUXEval<br>68.75<br>89.12<br>71.12<br>80.38<br>85.62<br>86.62<br>**90.62**|LiveCodeBench<br>41.18<br>83.14<br>36.27<br>63.73<br>81.37<br>71.29<br>**84.16**|
|---|---|---|---|



prompts showed a marked advantage, with the SC method amplifying this gap. Notably, the combination of CoT, CoC, and hypothesis decomposition and amendment enabled the LLM to exhibit a substantial degree of reasoning and generalization ability, nearly solving all presented problems. 

## 4.3 ABDUCTIVE CODE REASONING 

For abductive code reasoning, we employ the same baseline methods as those used for deductive reasoning. The experimental results are presented in Figure 3. Compared to deductive reasoning, abductive reasoning involves a reverse thinking process, which presents significant challenges. The LLM cannot derive the program’s intermediate states through deduction and must first establish an abstractlevel understanding of the function’s behavior before proceeding with abduction. On the CRUXEval dataset, the performance decline for abductive reasoning ranged from 8.20% to 25.52%. However, the hypothesis decomposition and amendment approach demonstrate robustness, as the change in reasoning modes resulted in only minimal performance degradation (8.20%) while still outperforming baseline methods by 10.02% to 31.89% on the CRUX- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0007-09.png)


**----- Start of picture text -----**<br>
83.75<br>CRUXEval<br>80 LiveCodeBench 74.50 76.12 77.12<br>71.00 71.57<br>70 63.50 66.67 64.25 65.69 61.76 60.78<br>60<br>54.90<br>50.98<br>50<br>40<br>30<br>20<br>10<br>0<br>Standard CoT SC (N=3) SR (T=2) CoC Sub-Hyp T=2, N=1<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 3: RHDA method on abductive code reasoning task. _T_ refers to the maximum number of iterations. _N_ refers to the number of candidates. 

Eval dataset and 7.35% to 40.39% on the LiveCodeBench dataset. A horizontal comparison of the two datasets revealed a trend similar to that observed in deductive reasoning, with an overall performance decline on the LiveCodeBench dataset, suggesting a complex relationship between reasoning and recall. 

## 4.4 QUALITATIVE ANALYZE 

We select some cases to conduct an in-depth exploration of the quality of RHDA. 

7 

Published as a conference paper at ICLR 2025 

Table 3: We compare the results obtained using the sub-hypothesis decomposition method with those obtained without it. The results without hypothesis decomposition are presented at the top of the table, while those with hypothesis decomposition are shown below. Benchmark: MiniARC-ID26. 

||**Observations**|**Observations**|**Observations**|**Observations**|**Observations**|**Observations**|**Observations**|**Observations**|**Observations**|**Hypothesis**<br>**Executable Function**|
|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||**No Sub Hypothesis:**<br>For a iven 5x5 matrix inut<br>**def** fn(matrix):<br>**iflen**(matrix)!=5**orany**(**len**(row)!=5**for**row**in**matrix):|
|||||||||||g   p,<br>shift the frst row to the last row,<br>the second row to the fourth row,<br>the third row to the third row<br>(unchanged), the fourth row to the<br>second row, and the ffth<br>row to the frst row. This rotates<br>the rows up by two positions.<br>           <br>**raise** ValueError("Input must be a 5x5 matrix")<br>_# Create a new matrix to store the result_<br>result = [[0]*5 **for** _ **in range**(5)]<br>_# Apply the row shifting rule_<br>result[0] = matrix[2]<br>_# Third row to first row_<br>result[1] = matrix[3]<br>_# Fourth row to second row_<br>result[2] = matrix[4]<br>_# Fifth row to third row_<br>result[3] = matrix[0]<br>_# First row to fourth row_<br>result[4] = matrix[1]<br>_# Second row to fifth row_<br>**return** result|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
|||||||||||**Sub Hypothesis:**<br>Step 1: Identify the non-zero rows.<br>Step 2: Move non-zero rows to the bottom.<br>Step 3: Shift all rows down to fll the grid.<br>Step 4: Repeat steps for next input.<br>**def** fn(grid):<br>_# Step 1: Identify non-zero rows_<br>non_zero_rows = [row **for** row **in** grid **if any**(row)]<br>_# Step 2: Reverse the order of non-zero rows_<br>non_zero_rows.reverse()<br>_# Step 3: Calculate the number of zero rows needed_<br>num_zero_rows = **len**(grid) - **len**(non_zero_rows)<br>_# Step 4: Create the new grid with zero rows at the top_<br>zero_rows = [[0] * **len**(grid[0]) **for** _ **in range**(num_zero_rows)]<br>_# Step 5: Combine zero rows and reversed non-zero rows_<br>new_grid = zero_rows + non_zero_rows<br>**return** new_grid|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||



**Hypothesis Decomposition Reduces Task Complexity.** In Table 3, we present a qualitative analysis of the MiniARC benchmark, comparing the effects of using hypothesis decomposition versus not using it. As illustrated in the examples above, without hypothesis decomposition, the descriptions require a substantial amount of language, which ultimately leads to severe hallucinations in the program and results in failure. In contrast, the examples below, which utilize hypothesis decomposition, present clear and executable objectives, enabling the translator to easily write executable functions and achieve successful solutions as instructed. 

Table 4: We present the additional improvements resulting from the use of amendments. The results shown above are those obtained after the initial hypotheses, while the results displayed below reflect the outcomes following the submission of amendments and subsequent reflections. Benchmark: List Function-ID29. 

||**Observations**|**Rounds**|**Executable Function**|
|---|---|---|---|
|||**Round 1:**|**def** fn(lst):<br>_# Check if the list has fewer than two elements_|
|||Step 1: Remove elements from the end|**if len**(lst)<2:|
|||of the input list until a ’9’ is encountered.<br>Step 2: If ’9’ is present, remove all|**return** []<br>_# Check if ’9’ is present in the list_<br>**if** 9 **in** lst:|
|[0,|8, 3, 9]_→_[0, 8]<br>[6, 1]_→_[]|elements after the last occurrence of ’9’.<br>Step 3: If ’9’ is not present, remove the|_# Find the last occurrence of ’9’_<br>last_nine_index=**len**(lst)-1-lst[::-1].index(9)<br>_# Return the list up to and including the last ’9’_|
||[4, 8, 7]_→_[4]<br>. ..|last two elements of the list.<br>Step 4: If the list has fewer than two|**return** lst[:last_nine_index+1]<br>**else**:<br>_# If ’9’ is not present,_|
|||elements, return an empty list.|_# remove the last two elements_|
||||**return** lst[:-2]|
|||**Round 2:**|**def** fn(input_list):|
|||Step 1: Remove the last two elements<br>from the input list.|_# Remove the last two elements from the input list_<br>modified_list = input_list[:-2]<br>_# Check if the list has fewer than two elements_|
|||Step 2: If the resulting list has fewer than<br>two elements, return an empty list.|**if len**(modified_list) < 2:<br>**return** []<br>**return** modified_list|



**Amendments Guide LLM Towards Correct Pathway.** We present a qualitative analysis of the use of amendments in the List Function benchmark in Table 4. The upper section displays the initialization of the hypothesis, where the LLM generates a potential guess based on the observations and translates it into an executable program. After offloading the execution to the tool (e.g., Python executor) and receiving feedback, amendments are proposed to modify the initial hypothesis. Following this reflection, the LLM re-optimizes the rules, ultimately yielding the correct execution results. More qualitative analyse examples please refer to Appendix E.1. 

**Failure Analyse.** We also conduct an in-depth analysis of the reasons behind process failures in RHDA, detailed in Appendix E.2. Our findings reveal that the primary limitation arises from the restricted intrinsic reasoning capabilities of LLMs, which continue to face challenges in understanding and addressing complex problems. These limitations are primarily reflected in two aspects: 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Task Initial decomposed hypothesis  𝒉𝟎 :  The Generated Scripts  Executor<br>𝑔<br>Store 1. I need to take the pie.  [grab the pie]<br>the 2. I need to store the pie in fridge. [go to fridge, open fridge,  store the pie]<br>Piein  Updated hypothesis  𝒉𝟏 :  𝑔<br>1. I need to get to the pie, and take the pie.  [walk to the pie, grab the pie]<br>Fridge.<br>2. I need to store the pie in fridge. [go to fridge, open fridge,  store the pie]<br>Walk to the Pie Grab the Pie Open Fridge Store the Pie<br>**----- End of picture text -----**<br>


Figure 4: We demonstrate how RHDA can be extended to the VirtualHome framework to successfully complete the task of storing the pie in fridge. 

   - Difficulty in Generating Accurate Sub-Hypotheses: The generation of sub-hypotheses during the reasoning process often proves inaccurate, leading to subsequent breakdowns in reasoning chains. 

   - Sensitivity to Initial Hypotheses: The model exhibits a pronounced dependency on its initial hypotheses. Even when feedback is provided through amendment submissions, the model struggles to break free from its original thought framework, constraining its reasoning capabilities. 

- 4.5 RHDA IS A FLEXIBLE AND SCALABLE PROBLEM-SOLVING PATHWAY 

We consider extending the RHDA pipeline to more complex scenarios. To this end, we select VirtualHome (Puig et al., 2018; 2020), a sophisticated multi-agent platform for simulating household activities, as our new exploration subject. VirtualHome comprises a set of predefined atomic actions and objects that can be combined into high-level instructions. For example, ‘〈char0〉 [walk] 〈salmon〉’ describes character 0 walking to the salmon. Given a specific scenario, the LLM is tasked with completing concrete housework using a series of high-level instructions. As depicted in Figure 4, and guided by the RHDA process, we demonstrate how the LLM successfully accomplishes the task of storing pie in the fridge through the methods of hypothesis decomposition, execution verification (offloading to VirtualHome engine), and reflection. we show another example in Appneidx D. 

## 5 LIMITATION AND DISCUSSIONS 

**Benchmark Selection.** This paper represents the first systematic exploration of the code reasoning task, focusing on the analysis of three forms of logical reasoning: inductive, deductive, and abductive. Due to time and cognitive constraints, we were unable to collect all benchmarks for testing. Our aim is to stimulate in-depth discussion on this topic and inspire meaningful follow-up research. While several excellent studies utilize code to address logical reasoning tasks (Zelikman et al., 2023; Hu et al., 2023; Srivastava et al., 2024; Liu et al., 2024a), we did not include them here due to their differing starting points from this paper. 

**Hyperparameters.** The goal of this paper is to explore the potential of LLMs in code reasoning, rather than solely improving the performance of a specific code reasoning task. The RHDA framework serves as a preliminary exploration process; therefore, we didn’t fully optimized the prompt templates or specific hyperparameters (such as temperature, _T_ , and _N_ ) utilized. In the inductive code reasoning task, we examined a broader range of hyperparameter settings to illustrate that exploring multiple pathways aids in more effectively solving problems. 

9 

Published as a conference paper at ICLR 2025 

**Task Assessment.** We propose a novel code reasoning task, and experimental results indicate that current state-of-the-art LLMs exhibit limitations in tackling this task. In the future, we aim to further explore this challenging area and investigate the boundaries of human capabilities in similar tasks. 

## 6 RELATED WORK 

**Reasoning with LLMs.** LLMs such as GPT (OpenAI, 2023), LLaMA (Touvron et al., 2023), and Claude (Anthropic, 2024), demonstrate impressive reasoning capabilities across various NLP tasks (Zhang et al., 2024). However, due to the problems of direct reasoning with LLMs such as hallucinations (Ji et al., 2023), researchers have proposed several methods to enhance the reasoning power of LLMs. For example, (Zhou et al., 2023; Xue et al., 2025) decompose complex tasks into sequential subproblems, while (Sun et al., 2024) refine reasoning through environment feedback. Moreover, intermediate representations, such as graphs (Jiang et al., 2024), planning domain definition languages (PDDL) (Guan et al., 2023), and triples (Wang et al., 2023a), have been employed to enhance LLM’s reasoning. Most recently, OpenAI o1 (OpenAI, 2024) demonstrates strong reasoning capabilities and broad world knowledge. Upon further contemplation, it is capable of reasoning through complex tasks and addressing challenges that exceed those faced by previous scientific, coding, and mathematical models. 

Simultaneously, domain-specific reasoning with LLMs has gained attention. (Kim et al., 2024a) enhance reasoning outputs in computer tasks through recursive critique. In a case study using Minecraft, (Wang et al., 2023d) introduce a Describe, Interpret, Plan, and Select framework for open-world multitasking. In computer vision, (Gupta & Kembhavi, 2023) employ Python-like modular programs to tackle complex tasks. Nonetheless, reasoning in code remains an area yet to be thoroughly explored. 

**Improvement with Reflection.** Reflective ability is regarded as a crucial metric for evaluating LLMs as agents. Reflection can be categorized into internal and external based on its feedback source (Pan et al., 2024). Internal reflection relies feedback from the model’s own knowledge and parameters (Huang et al., 2022), while external feedback comes from various sources, including humans (Wang et al., 2023b), other models (Paul et al., 2024), external tools (Gou et al., 2024; Chen et al., 2024), or knowledge bases (Yao et al., 2023b; Asai et al., 2024). (Huang et al., 2024) find that LLMs struggle to self-correct their responses without external feedback, and in some cases, their performance may even decline following self-correction. Our work focuses on leveraging external tools, such as compilers, to generate feedback and enhance the performance of LLMs. 

## 7 CONCLUSION 

In this paper, we emphasized that the reasoning capabilities of LLMs still depend on recalling prior knowledge and highlighted that code reasoning has not been sufficiently explored as a novel perspective for examining the boundaries of LLM capabilities. Based on this consideration, we designed three meta-benchmarks—inductive code reasoning, deductive code reasoning, and abductive code reasoning—drawing on established forms of logical reasoning, and instantiated these benchmarks into eight specific tasks. Experimental results indicated that these benchmarks present significant challenges for current state-of-the-art LLMs. To initially explore code reasoning tasks, we proposed a method involving **R** eflective **H** ypothesis **D** ecomposition and **A** mendment ( **RHDA** ). This method was iterative: LLMs need to generate decomposed initial hypotheses based on observations and employ a translator to interpret these into executable functions that can be directly applied to the observations. After obtaining the executable functions, we performed execution verification and submit amendments, allowing for reflection and refinement of the sub-hypotheses. Experimental results demonstrated that this approach, which integrated the principles of divide-and-conquer and reflection, can flexibly solve complex code reasoning problems, achieving performance improvements of 2 to 3 times compared to baseline methods. Finally, we extended this process to simulate household tasks in real-world complex scenarios to validate its scalability and transferability. 

10 

Published as a conference paper at ICLR 2025 

## 8 ACKNOWLEDGMENT 

This research was partially supported by the Key Technologies R&D Program of Anhui Province (No.202423k09020039), the National Natural Science Foundation of China (Grants No.62477044, 62406303), Anhui Provincial Natural Science Foundation (No. 2308085QF229), the Fundamental Research Funds for the Central Universities (No.WK2150110038, WK2150110034). 

## 9 REPRODUCIBILITY STATEMENT 

Our code, datasets and experimental results are available at https://github.com/TnTWoW/ code_reasoning. Additionally, Appendix H contains details about pipeline and prompts used in method. 

## REFERENCES 

Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/ claude-3-5-sonnet. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id= hSyW5go0v8. 

- Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023. 

- Matej Balog, Alexander L Gaunt, Marc Brockschmidt, Sebastian Nowozin, and Daniel Tarlow. Deepcoder: Learning to write programs. _arXiv preprint arXiv:1611.01989_ , 2016. 

- Yoshua Bengio. From system 1 deep learning to system 2 deep learning., 2019. URL https: //nips.cc/Conferences/2019/Schedule?showEvent=15488. 

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

Mark Chen et al. Evaluating large language models trained on code, 2021. URL https:// arxiv.org/abs/2107.03374. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks, 2023. URL https: //arxiv.org/abs/2211.12588. 

- Xinyun Chen, Maxwell Lin, Nathanael Sch¨arli, and Denny Zhou. Teaching large language models to self-debug. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=KuPixIqPiq. 

Franc¸ois Chollet. On the measure of intelligence, 2019. URL https://arxiv.org/abs/ 1911.01547. 

- Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, and Pushmeet Kohli. Robustfill: Neural program learning under noisy i/o. In _International conference on machine learning_ , pp. 990–998. PMLR, 2017. 

- A. Feeney and V.A. Thompson. _Reasoning as Memory_ . Current Issues in Thinking and Reasoning. Taylor & Francis, 2014. ISBN 9781317820130. URL https://books.google.com.hk/ books?id=gEuvBAAAQBAJ. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing, 2024. URL https://arxiv.org/abs/2305.11738. 

11 

Published as a conference paper at ICLR 2025 

- Alex Gu, Baptiste Rozi`ere, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I. Wang. Cruxeval: A benchmark for code reasoning, understanding and execution, 2024. URL https://arxiv.org/abs/2401.03065. 

- Lin Guan, Karthik Valmeekam, Sarath Sreedharan, and Subbarao Kambhampati. Leveraging pretrained large language models to construct and utilize world models for model-based task planning. _Advances in Neural Information Processing Systems_ , 36:79081–79094, 2023. 

- Sumit Gulwani. Programming by examples-and its applications in data wrangling. In _Dependable Software Systems Engineering_ , pp. 137–158. IOS Press, 2016. 

- Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14953–14962, 2023. 

- Jonathan Haidt. The emotional dog and its rational tail: a social intuitionist approach to moral judgment. _Psychological review_ , 108(4):814, 2001. 

- Kyle Hardman and Nelson Cowan. Reasoning and memory: People make varied use of the information available in working memory. _Journal of experimental psychology. Learning, memory, and cognition_ , 42, 11 2015. doi: 10.1037/xlm0000197. 

- Brett K Hayes, Evan Heit, and Caren M Rotello. Memory, reasoning, and categorization: Parallels and common mechanisms. _Frontiers in psychology_ , 5:529, 2014. 

- Evan Heit, Caren M. Rotello, and Brett K. Hayes. Chapter two - relations between memory and reasoning. In Brian H. Ross (ed.), _The Psychology of Learning and Motivation_ , volume 57 of _Psychology of Learning and Motivation_ , pp. 57–101. Academic Press, 2012. doi: https://doi. org/10.1016/B978-0-12-394293-7.00002-9. URL https://www.sciencedirect.com/ science/article/pii/B9780123942937000029. 

- Yi Hu, Haotong Yang, Zhouchen Lin, and Muhan Zhang. Code prompting: a neural symbolic method for complex reasoning in large language models. _arXiv preprint arXiv:2305.18507_ , 2023. 

- Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve, 2022. URL https://arxiv.org/abs/ 2210.11610. 

- Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview. net/forum?id=IkmD3fKBPQ. 

- Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. _arXiv preprint_ , 2024. 

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Computing Surveys_ , 55(12):1–38, 2023. 

- Song Jiang, Zahra Shakeri, Aaron Chan, Maziar Sanjabi, Hamed Firooz, Yinglong Xia, Bugra Akyildiz, Yizhou Sun, Jinchao Li, Qifan Wang, et al. Resprompt: Residual connection prompting advances multi-step reasoning in large language models. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 5784–5809, 2024. 

- D. Kahneman. _Thinking, Fast and Slow_ . Penguin Books Limited, 2011. ISBN 9780141918921. URL https://books.google.com.hk/books?id=oV1tXT3HigoC. 

- Subbarao Kambhampati. Can large language models reason and plan? _Annals of the New York Academy of Sciences_ , 1534(1):15–18, 2024. 

12 

Published as a conference paper at ICLR 2025 

- Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. _Advances in Neural Information Processing Systems_ , 36, 2024a. 

- Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. An llm compiler for parallel function calling, 2024b. URL https://arxiv. org/abs/2312.04511. 

- Subin Kim, Prin Phunyaphibarn, Donghyun Ahn, and Sundong Kim. Playgrounds for abstraction and reasoning. In _NeurIPS 2022 Workshop on Neuro Causal and Symbolic AI (nCSI)_ , 2022. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- Patrick C. Kyllonen and Raymond E. Christal. Reasoning ability is (little more than) workingmemory capacity?! _Intelligence_ , 14(4):389–433, 1990. ISSN 0160-2896. doi: https://doi.org/10. 1016/S0160-2896(05)80012-1. URL https://www.sciencedirect.com/science/ article/pii/S0160289605800121. 

- Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, and Brian Ichter. Chain of code: Reasoning with a language modelaugmented code emulator, 2024. URL https://arxiv.org/abs/2312.04474. 

- Xin Lin, Zhenya Huang, Hongke Zhao, Enhong Chen, Qi Liu, Defu Lian, Xin Li, and Hao Wang. Learning relation-enhanced hierarchical solver for math word problems. _IEEE Transactions on Neural Networks and Learning Systems_ , 35(10):13830–13844, 2024. 

- Changshu Liu, Shizhuo Dylan Zhang, Ali Reza Ibrahimzada, and Reyhaneh Jabbarvand. Codemind: A framework to challenge large language models for code reasoning. _arXiv preprint arXiv:2402.09664_ , 2024a. 

- Jiayu Liu, Zhenya Huang, Zhiyuan Ma, Qi Liu, Enhong Chen, Tianhuang Su, and Haifeng Liu. Guiding mathematical reasoning via mastering commonsense formula knowledge. In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , KDD ’23, pp. 1477–1488. Association for Computing Machinery, 2023a. ISBN 9798400701030. doi: 10.1145/ 3580305.3599375. URL https://doi.org/10.1145/3580305.3599375. 

- Jiayu Liu, Zhenya Huang, Chengxiang Zhai, and Qi Liu. Learning by applying: A general framework for mathematical reasoning via enhancing explicit knowledge learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pp. 4497–4506, 2023b. 

- Jiayu Liu, Zhenya Huang, Tong Xiao, Jing Sha, Jinze Wu, Qi Liu, Shijin Wang, and Enhong Chen. SocraticLM: Exploring socratic personalized teaching with large language models. In _The Thirtyeighth Annual Conference on Neural Information Processing Systems_ , 2024b. URL https: //openreview.net/forum?id=qkoZgJhxsA. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptarc benchmark: Evaluating understanding and generalization in the arc domain, 2023. URL https://arxiv. org/abs/2305.07141. 

- Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and Armando SolarLezama. Is self-repair a silver bullet for code generation? In _International Conference on Learning Representations (ICLR)_ , 2024. 

OpenAI. Gpt-4 technical report, 2023. 

- OpenAI. Introducing openai o1, 2024. URL https://openai.com/o1/. 

13 

Published as a conference paper at ICLR 2025 

- Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. Automatically correcting large language models: Surveying the landscape of diverse automated correction strategies. _Transactions of the Association for Computational Linguistics_ , 12:484–506, 2024. doi: 10.1162/tacl ~~a 0~~ 0660. URL https://aclanthology.org/2024.tacl-1. 27. 

- Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boi Faltings. REFINER: Reasoning feedback on intermediate representations. In Yvette Graham and Matthew Purver (eds.), _Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 1100–1126, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. URL https: //aclanthology.org/2024.eacl-long.67. 

- Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al. Check your facts and try again: Improving large language models with external knowledge and automated feedback. _arXiv preprint arXiv:2302.12813_ , 2023. 

- Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 8494–8502, 2018. 

- Xavier Puig, Tianmin Shu, Shuang Li, Zilin Wang, Joshua B. Tenenbaum, Sanja Fidler, and Antonio Torralba. Watch-and-help: A challenge for social perception and human-ai collaboration, 2020. 

- Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, and Xiang Ren. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement, 2024. URL https://arxiv.org/abs/2310.08559. 

- Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot numerical reasoning. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pp. 840–854, 2022. 

- Joshua Stewart Rule. _The child as hacker: building more human-like models of learning_ . PhD thesis, Massachusetts Institute of Technology, 2020. 

- Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Kensen Shi, Joey Hong, Yinlin Deng, Pengcheng Yin, Manzil Zaheer, and Charles Sutton. ExeDec: Execution decomposition for compositional generalization in neural program synthesis. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Saurabh Srivastava, Annarose M B, Anto P V au2, Shashank Menon, Ajay Sukumar, Adwaith Samod T, Alan Philipose, Stevin Prince, and Sooraj Thomas. Functional benchmarks for robust evaluation of reasoning performance, and the reasoning gap, 2024. 

- James WA Strachan, Dalila Albergo, Giulia Borghini, Oriana Pansardi, Eugenio Scaliti, Saurabh Gupta, Krati Saxena, Alessandro Rufo, Stefano Panzeri, Guido Manzi, et al. Testing theory of mind in large language models and humans. _Nature Human Behaviour_ , pp. 1–11, 2024. 

- Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. Adaplanner: Adaptive planning from feedback with language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Heinz-Martin S¨uß, Klaus Oberauer, Werner W Wittmann, Oliver Wilhelm, and Ralf Schulze. Working-memory capacity explains reasoning ability—and a little bit more. _Intelligence_ , 30(3):261–288, 2002. ISSN 0160-2896. doi: https://doi.org/10.1016/S0160-2896(01) 00100-3. URL https://www.sciencedirect.com/science/article/pii/ S0160289601001003. 

14 

Published as a conference paper at ICLR 2025 

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- Jianing Wang, Qiushi Sun, Xiang Li, and Ming Gao. Boosting language models reasoning with chain-of-knowledge prompting. _arXiv preprint arXiv:2306.06427_ , 2023a. 

- Tianlu Wang, Ping Yu, Xiaoqing Ellen Tan, Sean O’Brien, Ramakanth Pasunuru, Jane Dwivedi-Yu, Olga Golovneva, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. Shepherd: A critic for language model generation, 2023b. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023c. URL https://arxiv.org/abs/2203.11171. 

- Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang, and Team CraftJarvis. Describe, explain, plan and select: interactive planning with large language models enables open-world multi-task agents. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , pp. 34153–34189, 2023d. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903. 

- Jason Weston and Sainbayar Sukhbaatar. System 2 attention (is something you might need too), 2023. URL https://arxiv.org/abs/2311.11829. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_ , 2021. 

- Shangzi Xue, Zhenya Huang, Jiayu Liu, Xin Lin, Yuting Ning, Binbin Jin, Xin Li, and Qi Liu. Decompose, analyze and rethink: Solving intricate problems with human-like reasoning cycle. _Advances in Neural Information Processing Systems_ , 37:357–385, 2025. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of Thoughts: Deliberate problem solving with large language models, 2023a. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In _International Conference on Learning Representations (ICLR)_ , 2023b. 

- Dagmar Zeithamova, Margaret L Schlichting, and Alison R Preston. The hippocampus and inferential reasoning: building memories to navigate future decisions. _Frontiers in human neuroscience_ , 6:70, 2012. 

- Eric Zelikman, Qian Huang, Gabriel Poesia, Noah Goodman, and Nick Haber. Parsel: Algorithmic reasoning with language models by composing decompositions. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum? id=qd9qcbVAwQ. 

- Yadong Zhang, Shaoguang Mao, Tao Ge, Xun Wang, Adrian de Wynter, Yan Xia, Wenshan Wu, Ting Song, Man Lan, and Furu Wei. Llm as a mastermind: A survey of strategic reasoning with large language models. _arXiv preprint arXiv:2404.01230_ , 2024. 

- Yuze Zhao, Zhenya Huang, Yixiao Ma, Rui Li, Kai Zhang, Hao Jiang, Qi Liu, Linbo Zhu, and Yu Su. RePair: Automated program repair with process-based feedback. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Findings of the Association for Computational Linguistics ACL 2024_ , pp. 16415–16429, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.973. URL https://aclanthology.org/2024.findings-acl.973. 

15 

Published as a conference paper at ICLR 2025 

Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

16 

Published as a conference paper at ICLR 2025 

## A DSL GRAMMARS 

RobustFill is a string manipulation task using the DSL. Figure 5 illustrates the DSL syntax for RobustFill. Our implementation is based on the works of ExeDec (Shi et al., 2024) and RobustFill (Devlin et al., 2017). 

Deepcoder is a list transformation task using the DSL. Figure 6. This implementation is based on the works of ExeDec (Shi et al., 2024) and DeepCoder (Balog et al., 2016). 

Program _P_ := Concat( _e_ 1 _, e_ 2 _, . . ._ ) Expression _e_ := _s | m | o |_ ConstStr( _c_ ) Compose _o_ := _m_ 1( _m_ 2) _| m_ ( _s_ ) Substring _s_ := SubStr( _k_ 1 _, k_ 2) _|_ GetSpan( _r_ 1 _, i_ 1 _, b_ 1 _, r_ 2 _, i_ 2 _, b_ 2) _|_ GetToken( _r, i_ ) _|_ GetUpto( _r_ ) _|_ GetFrom( _r_ ) Modification _m_ := ToCase( _a_ ) _|_ Replace( _δ_ 1 _, δ_ 2) _|_ Trim() _|_ GetFirst( _r, i_ ) _|_ GetAll( _r_ ) _|_ Substitute( _r, i, c_ ) _|_ SubstituteAll( _r, c_ ) _|_ Remove( _r, i_ ) _|_ RemoveAll( _r_ ) Regex _r_ := NUMBER _|_ WORD _|_ ALPHANUM _|_ ALL ~~C~~ APS _|_ PROPER ~~C~~ ASE _|_ LOWER _|_ DIGIT _|_ CHAR _| δ_ Case _a_ := ALL ~~C~~ APS _|_ PROPER ~~C~~ ASE _|_ LOWER Position _k_ := _−_ 100 _| −_ 99 _| . . . | −_ 1 _|_ 0 _|_ 1 _|_ 2 _| . . . |_ 100 Index _i_ := _−_ 5 _| −_ 4 _| . . . | −_ 1 _|_ 1 _|_ 2 _| . . . |_ 5 Boundary _b_ := START _|_ END Character _c_ := _A | . . . | Z | a | . . . | z |_ 0 _| . . . |_ 9 _| δ_ Delimiter _δ_ := &,.?!@()[]%{}/:;$# "’ 

Figure 5: The DSL syntax for string manipulation tasks in the RobustFill domain. 

Program _P_ := _i_ 1; _i_ 2; _. . ._ ; _a_ 1; _a_ 2; _. . ._ Initialization _i_ := _v ←_ INPUT Assignment _a_ := _v ← f | v ← h_ First-Order Operation _f_ := Head( _l_ ) _|_ Last( _l_ ) _|_ Access( _n, l_ ) _|_ Minimum( _l_ ) _|_ Maximum( _l_ ) _|_ Sum( _l_ ) _|_ Take( _n, l_ ) _|_ Drop( _n, l_ ) _|_ Reverse( _l_ ) _|_ Sort( _l_ ) Higher-Order Operation _h_ := Map( _λ, l_ ) _|_ Filter( _β, l_ ) _|_ Count( _β, l_ ) _|_ ZipWith(Σ _, l, l_ ) _|_ Scanl1(Σ _, l_ ) int _→_ int Lambda _λ_ := (+1) _|_ ( _−_ 1) _|_ ( _∗_ 2) _|_ ( _/_ 2) _|_ ( _∗_ ( _−_ 1)) _|_ ( _∗∗_ 2) _|_ ( _∗_ 3) _|_ ( _/_ 3) _|_ ( _∗_ 4) _|_ ( _/_ 4) int _→_ bool Lambda _β_ := ( _>_ 0) _|_ ( _<_ 0) _|_ (%2 == 0) _|_ (%2 == 1) (int, int) _→_ int Lambda Σ := (+) _|_ ( _−_ ) _|_ ( _∗_ ) _|_ (min) _|_ (max) Integer Variable _n_ := _v_ List Variable _l_ := _v_ Variable Name _v_ := _x_ 1 _| x_ 2 _| . . ._ 

Figure 6: The DSL for integer and list manipulation tasks in the DeepCoder domain. 

## B EXPERIMENTAL RESULTS USING MORE LLMS 

We report the performance of Llama3.1-70B-Instruct, Qwen-max (qwen-max-2024-09-19), Claude 3.5 (claude-3-5-sonnet-20240620) using the RHDA method and compare them with GPT-4o (gpt4o-2024-0806). The results for inductive code reasoning are shown in Table 5. The experimental results indicate that GPT-4o performs better in solving DSL problems, while Claude 3.5 excels in General Propose Language (GPL) tasks. Compared to closed-source models, the open-source model Llama still exhibits relatively limited reasoning capabilities. However, in list manipulation tasks (List Function and Deepcoder), Llama demonstrates stronger programming abilities. In Table 6, we 

17 

Published as a conference paper at ICLR 2025 

Table 5: Performance comparison of Llama3.1-70B-Instruct, Qwen-max, Claude 3.5 and GPT-4o on the PoT and RHDA methods in inductive code reasoning task. _T_ refers to the maximum number of iterations. _N_ refers to the number of candidates. 

|Model<br>Method|Accuracy<br>MiniARC<br>List Func<br>RobustFill<br>DeepCoder|Task Accuracy<br>MiniARC<br>List Func<br>RobustFill<br>DeepCoder|
|---|---|---|
|Llama3.1<br>PoT<br>Sub-Hyp<br>T=2, N=1|3.08<br>35.25<br>14.78<br>22.92<br>3.33<br>26.45<br>13.04<br>18.06<br>3.85<br>32.35<br>20.87<br>11.46|1.54<br>26.80<br>8.70<br>11.46<br>3.08<br>20.40<br>4.35<br>6.25<br>3.85<br>26.40<br>13.04<br>7.29|
|Qwen-max<br>PoT<br>Sub-Hyp<br>T=2, N=1|6.41<br>41.75<br>36.52<br>25.35<br>5.90<br>46.25<br>26.09<br>17.36<br>6.41<br>46.60<br>33.91<br>24.64|3.85<br>30.00<br>21.74<br>14.58<br>3.08<br>36.40<br>8.70<br>5.21<br>3.08<br>41.60<br>13.04<br>10.42|
|Claude-3.5<br>PoT<br>Sub-Hyp<br>T=2, N=1|11.79<br>51.30<br>30.43<br>25.69<br>12.56<br>53.55<br>22.61<br>33.33<br>**18.21**<br>**57.95**<br>33.91<br>29.86|8.46<br>39.20<br>27.14<br>13.54<br>9.23<br>42.40<br>8.70<br>16.67<br>**13.85**<br>**48.40**<br>17.39<br>20.83|
|GPT-4o<br>PoT<br>Sub-Hyp<br>T=2, N=1|10.90<br>44.90<br>37.39<br>30.90<br>8.46<br>47.10<br>35.65<br>24.65<br>12.56<br>51.05<br>**43.48**<br>**38.89**|8.46<br>33.60<br>26.09<br>19.79<br>6.92<br>36.40<br>17.39<br>12.50<br>10.77<br>41.20<br>**40.43**<br>**23.96**|



Table 6: Performance comparison of Llama3.1-70B-Instruct, Qwen-max, Claude 3.5 and GPT-4o on the CoT and RHDA methods in deductive and abductive code reasoning tasks. _T_ refers to the maximum number of iterations. _N_ refers to the number of candidates. 

|Model<br>Method|Deductive<br>CRUXEval<br>LiveCodeBench|Abductive<br>CRUXEval<br>LiveCodeBench|
|---|---|---|
|Llama3.1<br>CoT<br>Sub-Hyp<br>T=2, N=1|40.25<br>7.84<br>30.75<br>6.86<br>45.62<br>10.78|53.12<br>38.24<br>50.88<br>8.82<br>59.62<br>40.20|
|Qwen-max<br>CoT<br>Sub-Hyp<br>T=2, N=1|81.12<br>86.27<br>78.25<br>81.37<br>81.62<br>**88.24**|75.12<br>58.82<br>72.25<br>59.80<br>79.38<br>66.67|
|Claude-3.5<br>CoT<br>Sub-Hyp<br>T=2, N=1|82.75<br>77.45<br>77.75<br>65.69<br>86.88<br>80.39|73.62<br>61.76<br>74.75<br>53.92<br>83.38<br>61.76|
|GPT-4o<br>CoT<br>Sub-Hyp<br>T=2, N=1|89.12<br>83.14<br>86.62<br>71.29<br>**90.62**<br>84.16|71.00<br>66.67<br>77.12<br>60.78<br>**83.75**<br>**71.57**|



report the performance of the models in deductive and abductive code reasoning together. The experimental results show that GPT-4o outperforms Claude 3.5 in terms of program understanding and execution capabilities. These results suggest that RHDA is a framework-agnostic general process that can achieve optimal performance through a single reflection, applicable to both Llama, Qwen, Claude and GPT series models. 

## C BENCHMARK DETAILS 

**List Function.** We use the original dataset (Rule, 2020), which consists of a total of 250 tasks. Due to the limited context lengths of LMs, we only use the first 16 examples from BIG-Bench (bench authors, 2023): 8 for seen examples and 8 for unseen examples. We manually examined the exemplars and found 8 examples are generally sufficient to describe the pattern. 

**MiniARC.** We use the data from (Qiu et al., 2024). Such tasks are typically difficult to describe in natural language at an abstract level. Therefore, 

Table 7: The number of tasks per dataset, the numbers of seen examples per task, and unseen examples per task. 

||Dataset<br>List Function<br>MiniARC<br>RobustFill<br>Deepcoder|# Tasks<br>250<br>130<br>22<br>96|# Seen<br>8<br>3<br>5<br>3|# Unseen<br>8<br>3<br>5<br>3|
|---|---|---|---|---|
||CRUXEval|800|1|1|
||LiveCodeBench|102|1|1|



18 

Published as a conference paper at ICLR 2025 

we did not consider them for our evaluations. As we only evaluate textonly models, we use textual representations of the original visual grids by mapping each cell to a corresponding integer. 

**RobustFill.** RobustFill is a string manipulation task where the model is expected to perform a combination of atomic operations, such as extracting a substring from position _k_ 1 to _k_ 2 using _SubString_ ( _k_ 1 _, k_ 1), to achieve generalization. As an example, a program ToCase(Lower, SubStr(1,3)) converts full month names (January, April) to their abbreviations (jan, apr). 

**DeepCoder.** The DeepCoder task involves using DSL to perform operations on integer lists. In DeepCoder, each line represents a subroutine that performs atomic operations on previous variables and assigns the results to new variables. The result of the final line is the program’s output. For example, program a _←_ [int] | b _←_ FILTER(<0) a | c _←_ MAP(*4) b | d _←_ SORT c | e _←_ REVERSE b (where “|” denotes subroutine separator.) transforms the input [-17, -3, 4, 11, 0, -5, -9, 13, 6, 6, -8, 11] into the output [-12, -20, -32, -36, -68] 

## D RHDA ACTING AS AN AGENT IN VIRTUALHOME 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0019-05.png)


**----- Start of picture text -----**<br>
Task Initial decomposed hypothesis  𝒉𝟎 :  The Generated Scripts  Executor<br>1. I need to get the towel.  𝑔 [open the cabinet, grab towel]<br>2. I need to open the faucet for water. [walk to facuet, open facuet]<br>Clean<br>3. I need to get shampoo. [walk to shampoo, grab shampoo]<br>the<br>Bathr Updated hypothesis  𝒉𝟏 :<br>oom. 1. I need to get the towel.  𝑔 [open the cabinet, grab towel]<br>2. I need to switch on the faucet for water. [walk to facuet, switch on facuet]<br>3. I need to get shampoo. [walk to shampoo, grab shampoo]<br>Open Bathroom Cabinet Grab Towel Switch on Faucet Get Shampoo<br>**----- End of picture text -----**<br>


Figure 7: We illustrate how the RHDA framework can be extended to the VirtualHome environment to effectively accomplish the task of cleaning the bathroom. 

We utilized the RHDA framework to drive agent actions in the VirtualHome environment powered by LLMs. Figure 7 illustrates a task of cleaning the bathroom. 

Table 8: Execution Error Rate on VirtualHome 

||native GPT-4o|w/o Sub-Hyp|w/o Amend|RHDA|
|---|---|---|---|---|
|# Error Action_↓_|92|84|84|**52**|
|Avg. Err per Step_↓_|0.84|0.35|0.20|**0.16**|
|Avg. Err per Task_↓_|2.09|1.83|1.75|**1.08**|



We also provided some quantitative metrics to validate the potential of RHDA as a agent in VirtualHome. Specifically, we selected a total of 52 tasks across two scenarios in VirtualHome and manually tested their execution error rates. The test results are shown in Table 8, which indicate that native GPT-4o struggles to handle simulated real-world scenarios effectively. The primary cause of failure lies in generating scripts that, while semantically similar to correct actions, are not executable within the environment (e.g., ‘open the tap’ is invalid action, whereas ‘touch the tap’ is valid action). By employing the RHDA method, which incorporates step-by-step solutions and effective feedback mechanisms, the error rate was significantly reduced. 

19 

Published as a conference paper at ICLR 2025 

## E EXAMPLES ANALYSES 

## E.1 EFFECTIVE CASE STUDY 

Table 9: We compare the results obtained using the sub-hypothesis decomposition method with those obtained without it. The results without hypothesis decomposition are presented at the top of the table, while those with hypothesis decomposition are shown below. Benchmark ARC-ID37. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0020-04.png)


**----- Start of picture text -----**<br>
Observations Rounds Executable Function<br>def fn(grid):<br># Step 1: Identify the third row<br>third_row = grid[2]<br># Step 2: Check for a sequence of three or more identical non-zero numbers<br>n = len (third_row)<br>i = 0<br>while i < n:<br>if third_row[i] != 0:<br>Round 1: # Find the length of the sequence of identical numbers<br>Step 1:Step 2: Identify the third rowCheck for a non-zero start while i +==i1i< n and third_row[i] == third_row[start]:<br>numbers in the third row. length = i - start<br>Step 3: Replace the number at # if If length the sequence >= 3: is three or more, replace it<br>the center position of the sequence.Step 4: Change all numbers # center_indexreplacement_number Step 3: Replace = start the = +grid[1][center_index] sequence length // with 2 the number from the second row<br>in rows 4 and 5 to zero. third_row[start:i] = [0] * n # Set the entire row to zero first<br>third_row[center_index] = replacement_number<br>break # Assuming only one sequence needs to be replaced<br>else :<br>i += 1<br># Step 4: Set all numbers in rows 4 and 5 to zero<br>grid[3] = [0] * n<br>grid[4] = [0] * n<br>return grid<br>def fn(grid):<br># Step 1: Identify the third row<br>third_row = grid[2]<br># Step 2: Check for a sequence of three or more identical non-zero numbers<br>n = len (third_row)<br>i = 0<br>while i < n:<br>Round 2: if third_row[i] != 0:<br># Find the length of the sequence of identical numbers<br>Step 1: Identify the third row start = i<br>Step 2:numbers in the third row.Check for a non-zero length while i +=i=1<in- and startthird_row[i] == third_row[start]:<br>Step 3:in the third row with theReplace identical numbers # if If# length Stepthe sequence3: >= Replace 3: isthethreesequenceor more, replace it<br>corresponding non-zero number # Find the number from the first row at the same column<br>from the first row. replacement_number # Replace the sequence = grid[0][start] with the replacement number<br>third_row[start:i] = [replacement_number] + [0] * (length - 1)<br>else :<br>i += 1<br># Step 4: Change all numbers in rows 4 and 5 to zero<br>for row in range (3, 5):<br>grid[row] = [0] * n<br>return grid<br>**----- End of picture text -----**<br>


Table 10: We compare the results obtained using the sub-hypothesis decomposition method with those obtained without it. The results without hypothesis decomposition are presented at the top of the table, while those with hypothesis decomposition are shown below. Benchmark List FunctionID2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0020-06.png)


**----- Start of picture text -----**<br>
Observations Hypothesis Executable Function<br>No Sub Hypothesis: def # fn(input_list): Check if the list is empty or has only one element<br>The output is the element if len (input_list) <= 1:<br>[]  → [] from the input list that # return Calculate [] the middle index<br>[6, 9]  → [] is exactly in the middle middle_index = ( len (input_list) - 1) // 2<br>[1, 5, 0, 6, 2, 9, 3]  → [0] of the list. # return Return [input_list[middle_index]] the middle element as a list<br>[6, 3, 4, 1, 7, 2, 9, 8, 0]  → [4]<br>.. . Step 1: Sub Hypothesis: If the list has fewer<br>than three elements, def # fn(input_list): Check if the list has fewer than three elements<br>the output is an empty list. if len (input_list) < 3:<br>Step 2:If the list has three elsereturn : []<br>or more elements, # Return a list containing the integer<br>the output is a list containingonly the integer at the third # return at the third [input_list[2]] position (index 2)<br>position.<br>**----- End of picture text -----**<br>


We validated the effectiveness of the proposed method using examples from various benchmarks. For instance, as shown in Table 9, the MiniARC task example with ID 37 highlights how the LLM, after receiving feedback, successfully reflects on its errors and submits a revised solution. 

20 

Published as a conference paper at ICLR 2025 

Table 11: We compare the results obtained using the sub-hypothesis decomposition method with those obtained without it. The results without hypothesis decomposition are presented at the top of the table, while those with hypothesis decomposition are shown below. Benchmark Livecodebench Input-ID37. 

|Input-ID37.||
|---|---|
|**Observations**|**Hypothesis**<br>**Executable Function**|
|**def** minOperations(a, b):<br>**def** f():<br>ret = 0<br>aa, bb = a[:], b[:]<br>**for** i **in range**(**len**(a)):<br>**if** a[i] > a[-1] **or** b[i] > b[-1]:<br>a[i], b[i] = b[i], a[i]<br>ret += 1<br>**if** a[i] > a[-1] **or** b[i] > b[-1]:<br>**return** inf<br>a[:] = aa<br>b[:] = bb<br>**return** ret<br>ans = f()<br>a[-1], b[-1] = b[-1], a[-1]<br>ans = **min**(ans, f() + 1)<br>**return** -1 **if** ans > **len**(a) **else** ans<br>_# assert f(??) == 1_|**No Sub Hypothesis:**<br>The function ‘minOperations’ attempts<br>to determine the minimum number<br>of swaps needed to ensure<br>that for each index ‘i’,<br>neither ‘a[i]’ nor ‘b[i]’<br>are greater than the last<br>elements of their respective lists.<br>assert minOperations([3, 1, 2], [1, 3, 2]) == 1|
||**Sub Hypothesis:**<br>Step 1: **Function Purpose**<br>The goal of ‘minOperations‘ is to ...<br>Step 2: **Inner Function**<br>It attempts to ensure<br>that for each ‘i’, ...<br>Step 3: **Main Logic**<br>It calculates ‘ans‘ by calling ‘f’, ...<br>Step 4: **Objective**<br>To fnd an input such that...<br>assert minOperations([2, 3], [3, 2]) == 1|



Table 12: We present the additional improvements resulting from the use of amendments. The results shown above are those obtained after the initial hypotheses, while the results displayed below reflect the outcomes following the submission of amendments and subsequent reflections. Benchmark CruxEval Output-ID328. 

|**Observations**|**Rounds**<br>**Executable Function**|
|---|---|
|**def** f(array, L):<br>**if** L <= 0:<br>**return** array<br>**if len**(array) < L:<br>array.extend(f(array, L - **len**(array)))<br>**return** array<br>_# assert f([1, 2, 3], 4) == ??_|**Round 1:**<br>Step 1 Base Case Check:<br>The function checks if ‘L’ is<br>less than or equal to 0.<br>If true, it simply returns<br>the array as is<br>Step 2 Recursive Extension:<br>If the length of ‘array’<br>is less than L, it then<br>calls itself recursively<br>Step 3 Return Array:<br>If the array is already of<br>length ‘L‘ or longer, the<br>function simply returns the<br>array without any modifcations.<br>assert f([1, 2, 3], 4) == [1, 2, 3, 1]|
||**Round 2:**<br>Step 1 Base Case Check: ...<br>Step 2 Recursive Extension: ...<br>Step 3 Return Array: ...<br>Step 4 Example Check:<br>The original call then extends<br>‘[1, 2, 3]’ by ‘[1, 2, 3]’,<br>resulting in ‘[1, 2, 3, 1, 2, 3]’.<br>assert f([1, 2, 3], 4) == [1, 2, 3, 1, 2, 3]|



In Table 10, hypothesis decomposition reveals that the output number is determined not only by its position at the middle of the input array but also by being the third character. 

In Table 11, compared to models without hypothesis decomposition, those utilizing this approach progressively analyze the function’s behavior, ultimately achieving an abstract understanding of the program and making accurate assertions. In Table 12, for a complex recursive function, while the LLM accurately grasped the overall functionality of the function through hypothesis decomposition, it encountered difficulties during the detailed analysis of specific instances. Following the submission of a revised solution, the LLM reflected on its errors and successfully resolved the issue, addressing the collapse of the overall logical chain caused by a failure in a single step during multi-step reasoning. 

21 

Published as a conference paper at ICLR 2025 

Table 13: Failure case study. The observations indicate that the hypotheses proposed by the LLM are overly simplistic, rendering it difficult to account for all possible cases. Additionally, manual inspection and efforts to summarize these hypotheses proved challenging. This limitation is one of the primary factors contributing to the LLM’s failure in this task. Benchmark: List Function-ID9. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0022-02.png)


**----- Start of picture text -----**<br>
Observations Rounds Executable Function<br>def fn(input_list):<br>"""<br>Removes the first element from the<br>input list of integers.<br>Round 1: """<br>[2, 8]  → [8] Remove the first element # if Check not input_list: if the input list is empty<br>[7, 5, ..., 8, 4]  → [5, ..., 8] from the input list. return []<br>[8, 2, ..., 9]  → [2, ..., 9] # Return the list excluding the<br>[3, 2, 1, 0, 7, 8]  → [2, 1, 0] # first element<br>. .. return input_list[1:]<br>. .. def fn(input_list):<br>"""<br>Remove the first and last elements<br>from the input list.<br>Round 2: """<br>Remove the first and the last ## Check2 elementsif the list has less than<br>elements from the input list. if len (input_list) < 2:<br>return []<br># Return the list excluding the<br># first and last elements<br>return input_list[1:-1]<br>**----- End of picture text -----**<br>


Table 14: Failure case study. The LLM make some redundant hypotheses, which led to failure. Even after amendments, it was unable to break free from its original thought framework, resulting in a failure to reflect effectively. Benchmark: ARC-ID5. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0022-04.png)


**----- Start of picture text -----**<br>
Observations Rounds Executable Function<br>def fn(grid):<br>if not grid or not grid[0]:<br>return grid<br>num_rows = len (grid)<br>num_cols = len (grid[0])<br># Create a new grid initialized with zeros<br>Round 1: new_grid for col in = range [[0] *(num_cols):num_cols for _ in range (num_rows)]<br>Step 1: Identify non-zero elements # non_zero_elements Collect non-zero = elements [grid[row][col] in the current for row column in range (num_rows) \\<br>in the input grid. if grid[row][col] != 0]<br>Step 2: Shift all non-zero elementsdown one row. # if #If non_zero_elements: Shifttheredownare bynon-zeroone rowelements,with wrap-aroundshift them down by one row<br>Step 3: Repeat the process for each shifted_elements = [0] * num_rows<br>input-output pair to verify for new_rowi, elem= (i in + enumerate 1) % num_rows(non_zero_elements):<br>consistency. shifted_elements[new_row] = elem<br># Find the maximum value among the shifted elements<br>max_value = max (shifted_elements)<br># Fill the entire column in the new grid with the maximum value<br>for row in range (num_rows):<br>new_grid[row][col] = max_value<br>return new_grid<br>def fn(grid):<br>if not grid or not grid[0]:<br>return grid<br>rows, cols = len (grid), len (grid[0])<br># Step 1: Identify non-zero elements and their positions<br>Round 2: non_zero_positions = []<br>Step 1:Identify non-zero elements forfor r in c inrangerange (rows):(cols):<br>in the input grid. if grid[r][c] != 0:<br>Step 2: Shift all non-zero non_zero_positions.append((r, c))<br>elements down one row. # Step 2: Shift non-zero elements down by one row<br>Step 3: In the resulting grid, fill each shifted_positions = [( (r + 1) % rows, c) for r, c in \\<br>row with the maximum value non_zero_positions]<br>from its respective column, # Step 3: Determine the maximum value for each column<br>considering only the shifted # max_values from the shifted = [0] * positions cols<br>non-zero elements. for r, c in shifted_positions:<br>Step 4: Repeat the process for each max_values[c] = max (max_values[c], grid[r][c])<br>input-output pair to verify consistency. # Step 4: Construct the new grid<br>new_grid = [[0] * cols for _ in range (rows)]<br>for r, c in shifted_positions:<br>new_grid[r][c] = max_values[c]<br>return new_grid<br>**----- End of picture text -----**<br>


## E.2 FAILURE ANALYZE 

We analyze RHDA’s performance in numerous failure cases and summarize the underlying causes of these failures. Our findings suggest that the primary reason can be attributed to the insufficient 

22 

Published as a conference paper at ICLR 2025 

intrinsic capability of LLMs in code reasoning tasks. This limitation is specifically reflected in two aspects: 

- **Sub-hypotheses fail to address the problem** : For tasks that are overly complex or abstract (e.g., cases shown in Table 13), even though hypothesis decomposition attempts to reduce task complexity, LLMs still struggle to handle them effectively. 

- **Amendments fail to correct sub-hypotheses** : While amendments leverage external feedback to help LLMs reflect on their mistakes, the models often remain confined to their existing thought framework, even after recognizing errors (e.g., cases shown in Table 14). This results in the correction failing to resolve the issue. 

These observations indicate that for tasks exceeding the intrinsic capabilities of LLMs, relying solely on reflective hypothesis decomposition and amendment may not be sufficient to improve the model’s performance. 

## F COSTS 

Table 15: Avg. API calls and Total Cost using GPT-4o. 

|Method|Avg. A|PI Calls<br>RobustFill<br>Deepcoder|Total Co|st (cent)<br>RobustFill<br>Deepcoder|
|---|---|---|---|---|
||List Func<br>MiniARC||List Func<br>MiniARC||
|IO<br>PoT<br>CoC<br>SC (N=3)<br>SR (T=2)<br>T=2, N=3|8.0<br>4.0<br>1.0<br>1.0<br>1.0<br>1.0<br>3.0<br>24.0<br>1.4<br>1.9<br>5.4<br>5.9|5.0<br>3.0<br>1.0<br>1.0<br>1.0<br>1.0<br>15.0<br>9.0<br>1.5<br>1.6<br>5.5<br>5.6|10.2<br>4.6<br>5.0<br>3.7<br>11.0<br>9.0<br>5.3<br>3.7<br>4.6<br>3.3<br>8.6<br>4.0|2.0<br>3.3<br>0.6<br>1.2<br>1.1<br>1.4<br>0.6<br>1.2<br>0.5<br>1.1<br>3.1<br>4.7|
|Method|Deductive<br>CRUXEval<br>LiveCodeBench|Abductive<br>CRUXEval<br>LiveCodeBench|Deductive<br>CRUXEval<br>LiveCodeBench|Abductive<br>CRUXEval<br>LiveCodeBench|
|Standard<br>CoT<br>SC (N=3)<br>SR (T=2)<br>CoC<br>T=2, N=1|1.0<br>1.0<br>1.0<br>1.0<br>3.0<br>3.0<br>1.6<br>1.7<br>1.0<br>1.0<br>1.6<br>1.7|1.0<br>1.0<br>1.0<br>1.0<br>3.0<br>3.0<br>1.4<br>1.5<br>1.0<br>1.0<br>1.4<br>1.5|2.9<br>0.5<br>19.4<br>3.7<br>2.9<br>0.5<br>3.8<br>0.6<br>18.3<br>4.1<br>19.0<br>4.4|3.1<br>1.5<br>19.5<br>3.3<br>3.3<br>1.5<br>3.4<br>1.6<br>19.0<br>3.4<br>18.8<br>4.4|



In Table 15, we present the average number of API calls and the total cost for each task. We used GPT-4o, with an input cost of $0.0025/1K tokens and an output cost of $0.01/1K tokens. The results indicate that our approach still demonstrates high cost-effectiveness for certain tasks. 

- G TRADE OFF BETWEEN NUMBER OF ITERATIONS AND PERFORMANCE GAIN 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0023-10.png)


**----- Start of picture text -----**<br>
(a) List Function (b) MiniARC (c) RobustFill (d) DeepCoder<br>52.5 14 45 37.5<br>50.0 13 35.0<br>47.5 12 40 32.5<br>45.0 11 35 30.0<br>42.5 10 30 27.5<br>40.0 9 25 25.0<br>37.5 AccuracyTask Accuracy 87 20 22.520.0<br>1 2 3 1 2 3 1 2 3 1 2 3<br>Iteration Iteration Iteration Iteration<br>Accuracy(%)<br>**----- End of picture text -----**<br>


Figure 8: In the inductive code reasoning tasks, as the number of iterations increased, the performance continued to improve. 

In this section, we investigate the impact of iteration count on the performance of three types of reasoning tasks, with experimental results illustrated in Figure 8 and Figure 9. For inductive and abductive code reasoning tasks, performance consistently improved as the number of iterations increased. However, the rate of improvement diminished, with marginal gains becoming less significant at higher iteration counts. Conversely, for deductive code reasoning tasks, performance followed a rise-and-fall trend, initially improving but declining with excessive iterations. These findings suggest that while increasing the number of iterations can enhance performance for general code reasoning tasks, it is crucial to balance iterative gains against potential performance instability. 

23 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment_images/iclr_2025_Unveiling_the_Magic_of_Code_Reasoning_through_Hypothesis_Decomposition_and_Amendment.pdf-0024-01.png)


**----- Start of picture text -----**<br>
(a) CRUXEval (b) LiveCodeBench<br>85<br>90<br>88 80<br>86<br>75<br>84<br>82 70<br>80 Deductive 65<br>78 Abductive<br>60<br>1 2 3 1 2 3<br>Iteration Iteration<br>Accuracy(%)<br>**----- End of picture text -----**<br>


Figure 9: In the deductive code reasoning tasks, the performance slightly decreased as the number of iterations increased. Conversely, in the abductive code reasoning tasks, the performance consistently improved with an increasing number of iterations. 

## H PROMPTS 

Table 16: Prompts used in our study. _{}_ refers to a placeholder. 

|Type|Prompt||
|---|---|---|
||Generate a rule that maps the following inputs to their||
||corresponding outputs step by steps.<br>**_{_Task description****_}_**||
|Sub Hypothesis||**_{_Examples****_}_**|
|Generation|||
|||Please format your rule as follows:|
|||**_{_Rule format****_}_**|
|||Your rule:<br>**_{_Rule****_}_**|
||This rule does not work for the following examples.||
|||**_{_Feedback****_}_**|
|Amendment<br>Submission|Please|carefully reconsider each of your steps to ensure<br>that the rules are correct.<br>Systematically|
|||generate new rules, step by step.|
|||**_{_Feedback description****_}_** Please|
|||format your rule as follows:|
|||**_{_Rule format****_}_**|
||You|are an expert Python programmer.<br>Write a Python|
|Hypothesis<br>Translation|function ‘fn‘ for the following rule.<br>**_{_Translation**<br>**Example description****_}_**||
|||Rule:<br>**_{_Rule****_}_**|
||Generate an output corresponding to the given input based||
||on the rule.<br>**_{_Application Example description****_}_**||
|Rule<br>Application||Rule:<br>**_{_Rule****_}_**|
|||Input:<br>**_{_Test input****_}_**|
|||Output:|



24 

