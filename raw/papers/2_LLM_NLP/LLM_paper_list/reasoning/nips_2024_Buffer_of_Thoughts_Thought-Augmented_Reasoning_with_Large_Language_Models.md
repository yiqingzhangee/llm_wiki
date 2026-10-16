# **Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models** 

**Ling Yang**[1] _[∗]_[�] **, Zhaochen Yu**[1] _[∗]_ **, Tianjun Zhang**[2] **, Shiyi Cao**[2] **, Minkai Xu**[3] **, Wentao Zhang**[1] **, Joseph E. Gonzalez**[2] **, Bin Cui**[1] 1Peking University, 2UC Berkeley, 3Stanford University Project: https://github.com/YangLing0818/buffer-of-thought-llm 

## **Abstract** 

We introduce Buffer of Thoughts (BoT), a novel and versatile thought-augmented reasoning approach for enhancing accuracy, efficiency and robustness of large language models (LLMs). Specifically, we propose _meta-buffer_ to store a series of informative high-level thoughts, namely _thought-template_ , distilled from the problem-solving processes across various tasks. Then for each problem, we retrieve a relevant thought-template and adaptively instantiate it with specific reasoning structures to conduct efficient reasoning. To guarantee the scalability and stability, we further propose _buffer-manager_ to dynamically update the meta-buffer, thus enhancing the capacity of meta-buffer as more tasks are solved. We conduct extensive experiments on 10 challenging reasoning-intensive tasks, and achieve significant performance improvements over previous SOTA methods: 11% on Game of 24, 20% on Geometric Shapes and 51% on Checkmate-in-One. Further analysis demonstrate the superior generalization ability and model robustness of our BoT, while requiring only 12% of the cost of multi-query prompting methods (e.g., tree/graph of thoughts) on average. Notably, we find that our Llama3-8B + BoT has the potential to surpass Llama3-70B model. Our project is available at https://github.com/YangLing0818/buffer-of-thought-llm 

## **1 Introduction** 

A series of Large Language Models (LLMs) [1–5] like GPT-4 [3], PaLM [2] and LLaMA [6, 7] have showcased the impressive performance in various reasoning tasks. In addition to scaling up the model size to improve the reasoning performance, there are more effective prompting methods that further enhance the functionality and performance of LLMs. We divide these methods into two categories: (i) **single-query reasoning:** these methods [8–10] usually focus on prompt engineering and their reasoning process can be finished within a single query, such as CoT [8] that appends the input query with ’Let’s think step by step’ to produce rationales for increasing reasoning accuracy, and Few-shot Prompting [11, 12, 9, 13] which provides task-relevant exemplars to assist the answer generation; (ii) **multi-query reasoning:** these methods [14, 15] focus on leveraging multiple LLM queries to elicit different plausible reasoning paths, thus decomposing a complex problem into a series of simpler sub-problems, such as Least-to-Most [16], ToT [14] and GoT [17]. 

However, both kinds of methods face some limitations: (1) single-query reasoning usually requires prior assumption or relevant exemplars of reasoning process, which makes it impractical to manually design them task by task, thus lacking universality and generalization; (2) Due to the recursive expansion of reasoning paths, multi-query reasoning is usually computationally-intensive when finding a unique intrinsic structure underlying the reasoning process for each specific task; (3) 

> _∗_ Equal Contribution. � yangling0818@163.com 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Single-Query<br>Input query LLM Reasoning Output<br>Manual Prompt for Specific Task<br>(e.g., CoT, Few-shot  Prompting) Accuracy<br>N-hop Iteration<br>Input query LLM Multi-Query Thought  Reasoning Output<br>Pre-defined Query  Expansion<br>Structure Efficiency<br>(e.g., ToT, GoT)<br>Problem Instantiated<br>Input query LLM Distiller Reasoning Output<br>High-level  Thought Distillation  Accuracy<br>Thought Retrieval Thoughts  and Update &<br>Efficiency<br>Meta Buffer<br>Buffer of Thoughts (BoT)<br>**----- End of picture text -----**<br>


Figure 1: Comparison between single-query [8, 11], multi-query [14, 17], and (c) our BoT methods. 

Both single-query and multi-query reasoning processes are limited by their designed exemplars and reasoning structures, and they neglect to derive general and high-level guidelines or thoughts from previously-completed tasks, which are informative for improving efficiency and accuracy when solving similar problems. 

To address these limitations, we propose Buffer of Thoughts (BoT), a novel and versatile thoughtaugmented reasoning framework aimed at enhancing reasoning accuracy, efficiency and robustness of LLMs across various tasks. Specifically, we design _meta-buffer_ , a lightweight library housing a series of universal high-level thoughts ( _thought-template_ ), which are distilled from different problem-solving processes and can be shared across tasks. Then, for each problem, we retrieve a relevant thoughttemplate and instantiate it with specific reasoning structure for efficient thought-augmented reasoning. In order to guarantee the scalability and stability of our BoT, we further propose _buffer-manager_ to dynamically update the meta-buffer, which effectively enhances the capacity of meta-buffer as more tasks are solved. 

Our method has three critical advantages: (i) **Accuracy Improvement:** With the shared thoughttemplates, we can adaptively instantiate high-level thoughts for addressing different tasks, eliminating the need to build reasoning structures from scratch, thereby improving reasoning accuracy. (ii) **Reasoning Efficiency:** Our thought-augmented reasoning could directly leverage informative historical reasoning structures to conduct reasoning without complex multi-query processes, thus improving reasoning efficiency. (iii) **Model Robustness:** The procedure from thought retrieval to thought instantiation is just like the human thought process, enabling LLMs to address similar problems in a consistent way, thus significantly enhancing the model robustness of our method. Our empirical studies demonstrate that Buffer of Thoughts significantly improves precision, efficiency, and robustness over a diverse array of tasks. Here, we summarize our contributions as follows: 

1. We propose a novel thought-augmented reasoning framework Buffer of Thoughts (BoT) for improving the accuracy, efficiency and robustness of LLM-based reasoning. 

2. We propose meta-buffer for store informative high-level thoughts distilled from different problems, and adaptively instantiate each thought template to address each specific task. 

3. We design buffer-manager to distill thought-templates from various solutions, and is continually improves the capacity of meta-buffer as more tasks are solved. 

4. We conduct extensive experiments on 10 challenging reasoning-intensive tasks. Our BoT achieves significant performance improvements over previous SOTA methods: **11% on Game of 24, 20% on Geometric Shapes and 51% on Checkmate-in-One** , while requiring **only 12% of the cost** of multi-query prompting methods on average. 

2 

## **2 Related Work and Discussions** 

**Retrieval-Augmented Language Models** The retrieval-augmented (Large) Language Model is introduced as a solution to mitigate the phenomenon of hallucination and enhance the output quality of language models [18–22]. When presented with an input question, the retrieval-augmented LLM first queries an external database with billion-level tokens [23] for retrieving a subset of the text corpus to help generating the final answer. Notably, the retrieval-augmented LLM achieves superior question-answering performance using fewer parameters compared to conventional LLMs [19], and it has found application across various downstream tasks [24–26], including multi-modal generation [24, 22, 23, 25] and biomedical applications [26, 27]. In this paper, we construct a novel category of retrieval database, termed _meta-buffer_ , which contains a series of high-level thoughts rather than specific instances, aiming to universally address various tasks for LLM-based reasoning. 

**Prompt-based Reasoning with Large Language Models** Prompting techniques have significantly enahnced the arithmetic and commonsense reasoning capabilities of LLMs. Chain-of-Thought (CoT) prompting [8] and its variants [28–30], such as Least-to-Most [16], Decomposed Prompting [31], and Auto-CoT [13]—prompt LLMs to break down complex questions into simpler subtasks and systematically solve them before summarizing a final answer. Numerous studies [32–37] have demonstrated the effectiveness of these prompting methods across a wide range of tasks and benchmarks. Innovations like Tree-of-Thought [14] and Graph-of-Thought [17], have further advanced this field by exploring dynamic, non-linear reasoning pathways to expand heuristic capabilities of LLMs [38, 39]. However, they suffer from increased resource demands and greater time complexity, depend on manual prompt crafting, and are often tailored to specific task types. Recent meta prompting methods [15, 40] utilize a same task-agnostic form of prompting for various tasks and recursively guide a single LLM to adaptively addressing different input queries. Nevertheless, such a long meta prompt may require a considerable context window, and these methods fail to leverage historical informative guidelines or thoughts for potential similar tasks. 

**Analogical Reasoning** Analogical reasoning is a useful technique for natural language reasoning [41–45]. Recent works demonstrate that LLMs can perform analogical reasoning just like humans [46, 47, 12, 48, 49]. For example, Analogical Prompting [12] and Thought Propagation [48] prompt LLMs to self-generate a set of analogous problems, and then utilize the results of analogous problems to produce a solution for input problem. However, the specific solutions for self-explored problems may introduce additional noise and cause error accumulation. Recent Thought-Retriever [49] uses the intermediate thoughts generated when solving past user to address analogous queries, but it only focuses on textual comprehension/generation instead of general reasoning problems. Thus, a more high-level and general analogical approach for LLM complex reasoning is still lacking. 

## **3 Buffer of Thoughts** 

**Overview of Buffer of Thoughts** In this section, we introduce our Buffer of Thoughts in detail and we also illustrate our core thought-augmented reasoning process in Figure 2. Given a specific task, we utilize our _problem-distiller_ (Section 3.1) to extract critical task-specific information along with relevant constraints. Based on the distilled information, we search in _meta-buffer_ (Section 3.2) that contains a series of high-level thoughts ( _thought-template_ ) and retrieve a most relevant thoughttemplate for the task. Subsequently, we instantiate the retrieved thought-template with more taskspecific reasoning structures and conduct reasoning process. Finally, we employs a _buffer-manager_ (Section 3.3) for summarizing the whole problem-solving process and distilling high-level thoughts for imcreasing the capacity of meta-buffer. 

## **3.1 Problem Distiller** 

Most of complex tasks contain implicit constraints, complex object relationships, and intricate variables and parameters within their contexts. Consequently, during the reasoning stage, LLMs need to overcome three main challenges: extracting vital information, recognizing potential constraints, and performing accurate reasoning. These challenges would impose a significant burden on a single LLM. Therefore, we separate the extraction and comprehension stages of task information from the final reasoning stage, through prepending a _problem distiller_ to the reasoning process. More 

3 

Input Problem A certain shopping mall sells a batch of branded shirts, with an average daily sales of 20 pieces and a profit of 40 yuan per piece. In order to expand sales, increase profits, and reduce inventory as soon as possible, the mall has decided to take appropriate price reduction measures. After investigation, it was found that for every 1 yuan decrease in the price of this shirt, an average of 2 more shirts are sold per day. If the mall wants to make an average profit of 1200 yuan per day, how much price should each shirt be reduced? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Buffer of Thoughts<br>Chain-of-Thought<br>Thought Template  𝑻𝟏 Thought Template  𝑻𝑵<br>1.Calculate the current daily profit:Current daily profit = 20 * 40 = 800 yuan To solve any quadratic equation of the form  𝑎𝑥 [!] +  𝑏𝑥 +  𝑐 =  0, we can follow a general  def process_element(element): e.g., applying a filter or transformation.# Define how to process each individual element,<br>2.Calculate additional daily profit from selling more shirts:Additional daily profit = 2 * 40 = 80 yuan5. Calculate Price reduction:Price reduction needed = 320 / 2 = 160 yuan…… approach based on the method described. Here is the structured template for solving such equations: Step 1: Calculate the Discriminant •𝐷 Step 2: Determine the Nature of the Roots Compute the discriminant=𝑏 [!] −4𝑎𝑐. ……  𝐷 using the formula …… def combine_elements(element1, element2): summing numbers, concatenating strings, etc. def check_condition(accumulated_result): result must meet. def solve_problem(input_list):. # Define how to combine elements, e.g., # Define the condition that the accumulated  Meta Buffer<br>Therefore, each shirt should be reduced by 160 yuan in order for the mall to make an average profit of 1200  Problem Distillation& Instantiated Reasoning<br>yuan per day. Thought Template  𝑻𝟏 Thought Retrieval  Solution: •𝑝 be the original profit per shirt (40 yuan).<br>To devise a plan to solve the problem, let's break it down into steps:1.Determine the current daily profit: Calculate the current daily profit using the average daily Plan-and-Solve To solve any quadratic equation of the form  𝑎𝑥approach based on the method described. Here is the structured template for solving such equations: Step 1: Calculate the Discriminant •𝐷 Step 2: Determine the Nature of the Roots Compute the discriminant=𝑏 [!][!] +  𝑏𝑥 +  𝑐 =  0−4𝑎𝑐. , we can follow a general  𝐷 using the formula •reduced.••The mall wants to maintain an average daily profit of 1200 yuan, so we have:After simplification𝒙𝑥𝑞𝑟 [𝟐] be the rate at which the quantity of shirts sold increases.be the amount by which the price of each shirt is be the original quantity of shirts sold daily (20 pieces).−𝟑𝟎𝒙+ 𝟐𝟎𝟎= 𝟎： ……<br>sales and profit per shirt •If 𝐷≥0, the equation has two distinct real roots. Now we can solve this quadratic equation using the<br>……. •If 𝐷=0, the equation has exactly one real root  steps outlined in the template:<br>3. Set up the equation for the desired daily profit Create an equation that relates new profit per shirt,  increased number of shirts sold, and desired daily profitStep 4: Solve the equation …… (also known as a repeated or double root).• Step 3: Compute the Roots •𝑥=IfFor 𝐷 𝐷<0, the equation has two complex roots. ["#±] ≥0, calculate the roots using the formula!& % Step 1: Calculate the Discriminant 𝐷 𝑫= (−𝟑𝟎) Step 2: Determine the Nature of the RootsSinceStep 3: Compute the Roots= 𝒃 [𝟐] 𝐷 −𝟒𝒂𝒄 >=0, the equation has two distinct real roots. [𝟐] −𝟒∗𝟏∗𝟐𝟎𝟎 =100<br>Expand the equation and collect like terms:1200=(20×40)+(2𝑝×40)−(20×𝑝)−(2𝑝×𝑝) Forof the complex roots using the formula"#± 𝐷"%'<0, calculate the real and imaginary parts 𝑥= 𝒙= The two possible solutions are: "𝒃±𝟐𝒂 𝑫 = 𝟑𝟎±𝟏𝟎𝟐<br>Divide by 2 to simplify: !& 𝑥 =​20 or x =10<br>𝒑 [𝟐] −30 𝑝 −200 = 0  where 𝑖 is the imaginary unit. Due to reducing inventory as soon as possible, x=20 is<br>p≈35.6155 ………. taken<br>**----- End of picture text -----**<br>


Figure 2: Illustration of different reasoning process. Buffer of Thoughts enables large language models to tackle complex reasoning tasks through our thought-augmented reasoning process. Thought template is marked in orange and instantiated thought is marked in blue. 

concretely, we design a meta prompt _ϕ_ to first distill and formalize the task information. The distilled task information could be denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0004-04.png)


where _x_ is the task statement. Due to the page limit, we put the detailed meta prompt for problemdistiller in Appendix A.2. 

**Problem Condensation and Translation** We use the problem distiller to extract key elements from input tasks, focusing on: (1). Essential parameters and variables for problem-solving; (2). The objectives of the input tasks and their corresponding constraints. We then re-organize this distilled information into a clear, comprehensible format for the subsequent reasoning stage. We then translate the specific problems into high-level concepts and structures. This translation procedure decomposes complex real-world problems, like intricate mathematical application scenarios, into simpler, multi-step calculations, making it easier for later retrieval of high-level thought. 

## **3.2 Thought-Augmented Reasoning with Meta Buffer** 

**Motivation** Human often summarize and induce higher-level guidelines when solving problems and then apply them to relevant problems. Motivated by this, we propose _meta-buffer_ , a lightweight library that contains a series of high-level thoughts ( _thought-template_ ) for addressing various types of problems. Unlike traditional methods [11, 46, 12, 36, 9] that require specific instructions or exemplars, our high-level thought-templates can be adaptively instantiated when solving different problems, thereby enhancing LLMs with superior precision and flexibility. 

**Thought Template** As a kind of high-level guideline, our thought-template is stored in metabuffer , and is obtained from various problem-solving processes by our _buffer-manager_ . The details about acquiring thought-templates would be introduced in Section 3.3. Since our BoT aims to provide a general reasoning approach for various tasks, we correspondingly classify the thoughttemplates into six categories: Text Comprehension, Creative Language Generation, Common Sense Reasoning, Mathematical Reasoning, Code Programming and Application Scheduling. We provide some example thought-templates in Appendix A.1. Such classification of thought-templates can 

4 

facilitate the template retrieval for finding most suitable solutions to different problems. Here we denote thought template, template description and its corresponding category as ( _Ti, DTi, Ck_ ), where _i_ denotes the index of meta-template, _k ∈_ Z[+] and 1 _≤ k ≤_ 6, which means _Ck_ is in one of the six categories, and _DTi_ is the description of thought template. 

**Template Retrieval** For each task, our BoT retrieves a thought-template _Ti_ that is highly similar to the distilled problem _xd_ by calculating the embedding similarity between the description _DTi_ and _xd_ . The retrieval process can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0005-02.png)


_N_ is the size of the meta-buffer, _f_ ( _·_ ) is a normal text embedding model, and _Tj_ denotes the retrieved thought template. We set a threshold _δ_ (0.5 _∼_ 0.7 is recommended) to determine whether the current task is new. Therefore, if Sim( _f_ ( _xd_ ) _, {f_ ( _DTi_ ) _}[n] i_ =0[)] _[ < δ]_[, we identify the task] _[ x]_[ as a new task.] 

**Instantiated Reasoning** For each specific task, we discuss two situations for the instantiated reasoning, depending on whether the current task is new: The first situation is that we successfully retrieve a thought-template _Tj_ for the task. In this case, as presented in Figure 2, our thoughtaugmented reasoning will be adaptively instantiated to suitable reasoning structures with our designed instantiation prompt (in Appendix A.3). For example, in a Checkmate-in-One problem, we instantiate the template of updating chess board state to solve the problem step by step. Thus we conduct the instantiated reasoning for task _x_ using the distilled information _xd_ and the retrieved template _Tj_ , and produce its solution _Sx_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0005-05.png)


where _LLM_ instantiation denotes the instantiated reasoner with a LLM. 

In the second situation, the task is identified as a new task. To enable proper instantiated reasoning, we prepare three general coarse-grained thought-templates for utilization. Based on the distilled task information _xd_ , our BoT would automatically assign a suitable thought-template to the reasoning process. The detailed pre-defined thought-templates are included in Appendix A.3). 

## **3.3 Buffer Manager** 

We propose _buffer-manager_ to summarize the high-level guidelines and thoughts that are gained from each problem-solving process. It can generalize each specific solution to more problems, storing the critical distilled knowledge in the form of thought-templates within the meta buffer. In contrast to methods that **temporarily** generate exemplars or instructions for each problem, our buffer-manager can ensure **permanent** advancements in accuracy, efficiency, and robustness for LLM-based reasoning. 

**Template Distillation** To extract a general though-template, we propose a three-step approach: (1) Core task summarization: identifying and describing basic types and core challenges of problems; (2) Solution steps description: summarize the general steps for solving a problem; (3) General answering template: based on the above analysis, propose a solution template or approach that can be widely applied to similar problems. Additionally, to boost the generalization ability and stability of template distillation, we carefully design two types of in-context examples of how to generate thoughttemplate— _in-task_ and _cross-task_ examples. _Cross-task_ means we choose the template distilled from one task to tackle the problem of other tasks, such as addressing a mathematical problem with a code-related thought-template. The new template distilled from input task _x_ can be denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0005-12.png)


where _LLM_ distill is the LLM-based template distiller initialized with the following prompt: 

5 

## _**Prompt for Template Distillation:**_ 

## _**User: [Problem Description] + [Solution Steps or Code]**_ 

_To extract and summarize the high-level paradigms and general approaches for solving such problems, please follow these steps in your response:_ 

## _**1. Core task summarization:**_ 

_Identify and describe the basic type and core challenges of the problem, such as classifying it as a mathematical problem (e.g., solving a quadratic equation), a data structure problem (e.g., array sorting), an algorithm problem (e.g., search algorithms), etc. And analyze the most efficient way to solve the problem._ 

## _**2. Solution Steps Description:**_ 

_Outline the general solution steps, including how to define the problem, determine variables, list key equations or constraints, choose appropriate solving strategies and methods, and how to verify the correctness of the results._ 

## _**3. General Answer Template:**_ 

_Based on the above analysis, propose a template or approach that can be widely applied to this type of problem, including possible variables, functions, class definitions, etc. If it is a programming problem, provide a set of base classes and interfaces that can be used to construct solutions to specific problems._ 

_Please ensure that your response is highly concise and structured, so that specific solutions can be transformed into generalizable methods._ 

_**[Optional] Here are some exemplars of the thought-template:** (Choose cross-task or in-task exemplars based on the analysis of the_ _**Core task summarization** .)_ 

**Dynamic Update of Meta-Buffer** After template distillation, we need to consider whether the distilled template should be updated into the meta-buffer. If we initialize an empty meta-buffer or encounter a problem without a proper thought-template, the distilled thought-templates will be directly stored in the meta-buffer. If we solve problem with a retrieved thought-template, new insights may arise during the instantiation of a certain thought-template. Therefore, to avoid the redundancy of the meta-buffer while maintaining newly-generated informative thoughts, we will calculate the similarity between the embedding vectors of _DTnew_ and _{DTi}[n] i_ =0[and update the meta-buffer with] the following rule: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0006-12.png)


Otherwise, it means the meta-buffer has already possessed the necessary knowledge to solve this task and does not need to perform the update. Our dynamic update strategy effectively reduces the computational burden of template retrieval while ensuring the lightweight property of our meta-buffer. We further conduct ablation study to analyze it in Section 6. 

## **4 Experiments** 

**Datasets and Tasks** To evaluate the efficacy of our proposed Buffer of Thoughts and compare with previous methods, we consider a diverse set of tasks and datasets that require varying degrees of mathematical and algorithmic reasoning, domain-specific knowledge, and literary creativity: (a). The **Game of 24** from ToT [14], where the objective is to form an arithmetic expression that equals 24 using each of four given numbers exactly once; (b). Three BIG-Bench Hard (BBH) [35] tasks: **Geometric Shapes** , **Multi-Step Arithmetic Two** , and **Word Sorting** ; (c). Three reasoning tasks directly obtained from the BIG-Bench suite [50]: **Checkmate-in-One** , **Penguins** —where the task is to answer questions about penguins’ attributes based on a given table and additional natural language information, and **DateUnderstanding** —a task that involves inferring dates from natural language descriptions, performing arithmetic operations on dates, and utilizing global knowledge such as the number of days in February; (d). **Python Programming Puzzles** (P3) [51, 52], a collection of challenging programming puzzles written in Python with varying difficulty levels; (e). **Multilingual Grade School Math** (MGSM) [33], a multilingual version of the GSM8K dataset [53] featuring translations of a subset of examples into ten typologically diverse languages, including Bengali, Japanese, and Swahili; (f). **Shakespearean Sonnet Writing** from meta-prompting [15], a novel task where the goal is to write a sonnet following the strict rhyme scheme "ABAB CDCD EFEF GG" and incorporating three provided words verbatim. 

6 

Table 1: Comparing BoT with previous methods across various tasks. We denote the best score in **blue** , and the second-best score in **green** . Our BoT significantly outperforms other methods on all tasks, especially on general reasoning problems. 

|Task|Standard|Single-Query|Multi-Query<br>**BoT (Ours)**<br>ToT [14]<br>GoT [17]<br>Meta Prompting [15]|
|---|---|---|---|
||GPT4 [3]|GPT4+CoT [8]<br>Expert [9]<br>PAL [10]||
|Game of 24<br>MGSM (avg)<br>Multi-Step Arithmetic<br>WordSorting<br>Python Puzzles<br>Geometric Shapes<br>Checkmate-in-One<br>Date Understanding<br>Penguins<br>Sonnet Writing|3.0<br>84.4<br>84.0<br>80.4<br>31.1<br>52.6<br>36.4<br>68.4<br>71.1<br>62.0|11.0<br>3.0<br>64.0<br>85.5<br>85.0<br>72.0<br>83.2<br>83.2<br>87.4<br>83.6<br>85.2<br>93.2<br>36.3<br>33.8<br>47.3<br>69.2<br>55.2<br>51.2<br>32.8<br>39. 6<br>10.8<br>69.6<br>68.4<br>76.2<br>73.6<br>75.8<br>**93.3**<br>71.2<br>74.0<br>36.2|**74.0**<br>73.2<br>67.0<br>**82.4**<br>86.4<br>**87.0**<br>84.8<br>**89.2**<br>88.2<br>89.2<br>**90.0**<br>**99.8**<br>96.4<br>98.4<br>**99.6**<br>**100.0**<br>43.5<br>41.9<br>**45.8**<br>**52.4**<br>56.8<br>54.2<br>**78.2**<br>**93.6**<br>49.2<br>51.4<br>**57.2**<br>**86.4**<br>78.6<br>77.4<br>**79.2**<br>**88.2**<br>84.2<br>85.4<br>88.6<br>**94.7**<br>68.4<br>62.8<br>**79.6**<br>**80.0**|



**Implementation and Baselines** For the fair comparisons with previous methods, we use GPT-4 as the base model of our BoT, including the main experiment and the ablation study (in Section 6). We also use Llama3-8B and Llama3-70B in our analysis part on NVIDIA A100-PCIE-40GB GPU. We compare our Buffer of Thoughts with the following prompting methods: **1. Standard Prompting** : This is our most basic baseline, where an LLM is asked to generate a response directly from the input query, without any specific guiding input-output examples or additional instructions beyond the task description included in the query. 

**2. Single-query Method** : This includes Zero-shot CoT [8] and PAL [10], which use the LLM to analyze natural language problems and generate intermediate reasoning steps. We also include Expert Prompting [9], which creates an expert identity tailored to the specific context of the input query, and then integrates this expert profile into the input to generate a well-informed response. 

**3. Multi-query Method** : This includes ToT [14] and GoT [17], which enable LLMs to make deliberate decisions by considering multiple reasoning paths and self-evaluating choices to determine the next course of action. These methods also allow for looking ahead or backtracking when necessary to make global decisions. Additionally, we include Meta Prompting [15], which employs an effective scaffolding technique designed to enhance the functionality of LLMs. 

## **4.1 BoT Achieves Better Accuracy, Efficiency and Robustness** 

**Reasoning Accuracy** As shown in Table 1, our BoT consistently outperforms all previous prompting methods across multiple kinds of challenging benchmarks, particularly demonstrated in complicated reasoning tasks such as Game of 24 and Checkmate-in-One. Taking GPT-4 as a baseline, our method achieves an astonishing 79.4% accuracy improvement in Game of 24, and compared to ToT, which has a good performance on this task, we also achieve an 8.4% accuracy improvement. What’s more, compared to recent Meta-prompting method [15], we see **significant accuracy improvements: 23% on Game of 24, 20% on Geometric Shapes and 51% on Checkmate-in-One** . Existing methods need complex, iterative, and heuristic search strategies to address these problems on a case-by-case basis. Conversely, our BoT leverages the historical insights and informative guidelines from thought-templates, and further adaptively instantiate a more optimal reasoning structure for addressing these complex problems. 

**Reasoning Efficiency** In addition to significant improvements in accuracy, as a multi-query method, our BoT can achieve comparable reasoning time to single-query method across various tasks, while being considerably less than conventional multi-query method like ToT [14] as shown in Figure 3. For example, in Game of 24, both single-query and multi-query methods necessitate iterative and heuristic searches to identify feasible solutions. This process is particularly time-consuming and inefficient, especially for the multi-query method, which involves conducting multi-query search and backtrace phases. In contrast, our BoT directly retrieves a thought-template in code format, thus a program is instantiated to traverse combinations of numbers and symbols, thereby eliminating the need to build the reasoning structure from scratch. This allows for solving the problem with just one query after invoking the problem-distiller, significantly reducing the time required for complex reasoning. Notably, our **BoT requires only 12% of the cost of multi-query methods** (e.g., tree of thoughts and meta-prompting) on average. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Comparison of the inference time<br>Expert PAL ToT Meta-prompting Ours<br>10<br>9.03<br>9 8.73 8.47 8.34 8.43<br>8.04<br>8<br>7<br>6.39<br>6 5.5<br>5 4.64 5.17 4.81 5 4.81 5.21<br>4.16<br>4<br>3<br>2<br>1<br>0<br>Game of 24 MGSM Checkmate-in-One<br>Logarithmic time (s)<br>**----- End of picture text -----**<br>


Figure 3: Comparison of **logarithmic inference time** between our Buffer of Thoughts and GPT4 [3], GPT4+CoT [8], Expert-prompting [9], PAL [10], ToT [14] across different benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0008-02.png)


**----- Start of picture text -----**<br>
Success rate<br>GPT4 Expert PAL ToT Ours<br>100 98 96.8 93.4 95.15<br>90 85 87 84 84.57<br>80 76 78.4 71.82<br>71 70.12<br>70 67.13<br>61<br>60<br>53.4<br>50 48.2<br>40 36 36.4<br>30 27<br>20<br>10<br>0<br>Game of 24 MGSM Checkmate-in-One Average<br>Average accuracy (%)<br>**----- End of picture text -----**<br>


Figure 4: Comparison of reasoning robustness between our Buffer of Thoughts and GPT4 [3], GPT4+CoT [8], Expert-prompting [9], PAL [10], ToT [14] across different benchmarks. 

**Reasoning Robustness** To better evaluate our BoT, we devise a new evaluation metric: _success rate_ , which is used to assess the reasoning robustness. We randomly sample 1000 examples from various benchmarks as a test subset and evaluate different methods on this subset. As shown in Figure 4, we repeat this evaluation process 10 times and take the average accuracy as the success rate of different methods on each benchmark. Compared with other methods, our BoT consistently maintains a higher success rate across various tasks, surpassing the second-best by 10% in average success rate. We attribute our outstanding robustness to the great generalization ability of our distilled thought-templates during reasoning across different tasks. By offering high-level thought from the suitable thought-templates, the stability of our method across different tasks is greatly enhanced. 

## **5 Model Analysis** 

**Distribution Analysis of Thought-Templates** As depicted in the left figure of Figure 5, we choose six different benchmarks, each sampled with 100 distinct tasks. We update the meta-buffer from scratch, and after completing all sampled tasks, we display the number of derived thought-templates. 

8 

We can observe that our BoT generates a greater number of thought-templates in the MGSM tasks that contain more diverse scenarios. In tasks with relatively simple requirements, such as Checkmatein-One and Penguins, BoT produces more fixed thought-templates tailored for those specific issues. The distribution of templates indicates that our BoT can effectively discover appropriate thought templates for different benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0009-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0009-02.png)


Figure 5: Distribution Analysis of Thought-Templates and Time. _Left_ : Distribution Analysis of Thought-Templates. _Right_ : Distribution Analysis of Thought-Templates. 

**Distribution Analysis of Time Cost** As illustrated in Figure 5, we measured the average time cost for each component of BoT’s reasoning framework across different tasks. The time required for distilling task information and template retrieval is relatively short, whereas instantiated reasoning takes longer. Overall, considering the complexity of different components, our BoT achieves a relatively balanced distribution of time cost, demonstrating the efficiency of our BoT framework. 

**Better Trade-off between Model Size and Performance** As depicted in Figure 6, on Game of 24, word list sorting and Checkmate-in-One, Llama3-8B and Llama-70B models [6] may result in poor outcomes. However, equipped with our BoT, both models demonstrate a substantial accuracy improvement. Notably, **BoT+Llama3-8B has the potential to surpass single Llama3-70B model** . Our BoT enables smaller models to exhibit the capabilities that approximate or even surpass larger models, significantly bridging the gap between their reasoning abilities. Furthermore, it greatly diminishes the inference cost required by large language models when tackling complex problems. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0009-06.png)


**----- Start of picture text -----**<br>
Trade-off between model size and performance<br>BoT+Llama-3-70B BoT+Llama-3-8B Llama3-70B Llama3-8B<br>75.6<br>56.7<br>Checkmate-in-One<br>15<br>0.8<br>92.3<br>73.4<br>Word list sorting 79<br>48.4<br>78.4<br>73.4<br>Game of 24<br>2.4<br>1.2<br>0 10 20 30 40 50 60 70 80 90 100<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 6: We evaluate the trade-off between model size and performance with Llama3-8B and Llama3-70B models on three challenging benchmarks. 

9 

## **6 Ablation Study** 

**Impact of Problem-Distiller** As illustrated in Figure 7, when the problem-distiller is disabled, both Llama3-70B and GPT-4 experience a certain degree of accuracy decline. More complex problems, such as Game of 24 and Checkmate-in-One, show a more significant accuracy reduction, whereas relatively simpler problems like word list sorting and MGSM exhibit smaller decreases. This is because LLMs can more easily extract key information in simpler tasks, making the impact of the problem-distiller less noticeable. In contrast, extracting key information and potential constraints in complex problems is more challenging, making the role of our problem-distiller more prominent, thereby explaining the differences depicted in the figure. 

## **Ablation study of problem-distiller** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0010-03.png)


**----- Start of picture text -----**<br>
BoT+Llama-3-70B (w/o problem-distiller) BoT+Llama-3-70B BoT+GPT-4 (w/o problem-distiller) BoT+GPT-4<br>100 97.3 99.6<br>92.3<br>90 89.5 86.4 85.6 86.8 87.4 89.2<br>82.4<br>80 78.4 76.5 75.6 78.9<br>71.2<br>70 64.3<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>Game of 24 Word list sorting Checkmate-in-One MGSM<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 7: We conduct ablation study on problem-distiller across four benchmarks, employing Llama370B and GPT-4 as the base models. 

**Impact of Meta-Buffer** As illustrated in Figure 8, when the meta-buffer is disabled, both Llama370B and GPT-4 models exhibit a noticeable decline in performance, particularly in benchmarks requiring complex reasoning, such as Game of 24 and Checkmate-in-One. This further underscores the superiority of our meta-buffer in addressing complex problems. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0010-06.png)


**----- Start of picture text -----**<br>
Ablation study of meta-buffer<br>BoT + Llama-3-70B (w/o meta-buffer) BoT+Llama-3-70B BoT+GPT-4 (w/o meta-buffer) BoT+GPT-4<br>99.6<br>100 95.4<br>92.3<br>89.2<br>90 86.4 86.8 85.4<br>82.4 81.7<br>78.4 79.6<br>80 75.2 75.6<br>70 65.6<br>60 56.7<br>50<br>40<br>30 27.4<br>20<br>10<br>0<br>Game of 24 Word list sorting Checkmate-in-One MGSM<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 8: We conduct ablation study on meta-buffer across four benchmarks, employing Llama3-70B and GPT-4 as the base models. 

10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0011-00.png)


**----- Start of picture text -----**<br>
Ablation study of buffer-manager -- Accuracy<br>100<br>87.4 88.5<br>90<br>78.5<br>80<br>70<br>60 56.8<br>57.4<br>50 52.8 53.6 54.1<br>40<br>30<br>20<br>10<br>0<br>Round 1 Round 2 Round 3 Round 4<br>BoT+GPT4 BoT+GPT4 (w/o buffer-manager)<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 9: We conduct ablation study on buffer-manager regarding reasoning accuracy across four tasks, employing Llama3-70B and GPT-4 as the base models. 

**Impact of Buffer-Manager** In this ablation study, we divide the entire process into four rounds. In each round, we randomly sample 50 questions from each benchmark and conduct reasoning. In the subsequent round, we continue to randomly sample another 50 questions from each benchmark. As depicted in Figure 9, with the increase of the number of rounds, the model with the buffermanager continually expands the meta-buffer while also utilizing the thought-templates obtained from previously solved problems to help addressing subsequent similar problems. Therefore, we can observe that the accuracy of BoT steadily improves with each round. In contrast, the model without the buffer-manager fails to exhibit an upward trend. Additionally, we have also measured the reasoning time as depicted in Figure 10. when the number of rounds increases, the model with the buffer-manager will experience a continual improvement in reasoning efficiency. This is because, with the continual expansion of the meta-buffer, the likelihood of retrieving suitable thought-templates also increases. Consequently, models can avoid constructing reasoning structures from scratch, thereby enhancing the inference efficiency accordingly. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models_images/nips_2024_Buffer_of_Thoughts_Thought-Augmented_Reasoning_with_Large_Language_Models.pdf-0011-03.png)


**----- Start of picture text -----**<br>
Ablation study of buffer-manager -- Time<br>350<br>308<br>300 317<br>304 306<br>297<br>250<br>205<br>200<br>150 128<br>100 78.5<br>50<br>0<br>Round 1 Round 2 Round 3 Round 4<br>BoT+GPT4 BoT+GPT4 (w/o buffer-manager)<br>Average inference time per problem (s)<br>**----- End of picture text -----**<br>


Figure 10: We conduct ablation study on buffer-manager regarding reasoning efficiency across four tasks, employing Llama3-70B and GPT-4 as the base models. 

11 

## **7 Discussion** 

**Limitations and Future Directions** Despite our method’s significant improvement in accuracy while maintaining reasoning efficiency and robustness, our method’s enhancements are limited when addressing problems requiring human-like creativity, as this issue often does not rely on a specific thought-template. Besides, if our BoT initializes the meta-buffer with a weaker model, the quality of the derived thought-templates may be suboptimal due to the weaker model’s limited reasoning ability and instruction-following capability. Overall, our BoT brings out a set of future directions: 1. integrating external resources with BoT to build a open-domain system like agent models [54, 55]. 2. making the distillation of thought-templates optimizable, which may significantly enhance their template qualities for more complex tasks. 

**Conclusion** In this work, we introduce Buffer of Thoughts, a novel beffered reasoning framework that employs LLMs to utilize pre-accumulated experiences and methodologies from prior tasks as thought-templates stored within a meta-buffer. We further design buffer-manager to continuously refine the problem-solving processes and dynamically distill thought-templates, thereby progressively raising the LLM’s reasoning capacity. Our BoT demonstrates SOTA performance on 10 challenging tasks, and offers promising prospects for future research and application. 

## **References** 

- [1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, _et al._ , “Language models are few-shot learners,” _Advances in neural information processing systems_ , vol. 33, pp. 1877–1901, 2020. 

- [2] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, _et al._ , “Palm 2 technical report,” _arXiv preprint arXiv:2305.10403_ , 2023. 

- [3] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, _et al._ , “Gpt-4 technical report,” _arXiv preprint arXiv:2303.08774_ , 2023. 

- [4] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, “Glm: General language model pretraining with autoregressive blank infilling,” in _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 320–335, 2022. 

- [5] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, _et al._ , “Mixtral of experts,” _arXiv preprint arXiv:2401.04088_ , 2024. 

- [6] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, _et al._ , “Llama: Open and efficient foundation language models,” _arXiv preprint arXiv:2302.13971_ , 2023. 

- [7] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, _et al._ , “Llama 2: Open foundation and fine-tuned chat models,” _arXiv preprint arXiv:2307.09288_ , 2023. 

- [8] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, _et al._ , “Chain-of-thought prompting elicits reasoning in large language models,” _Advances in neural information processing systems_ , vol. 35, pp. 24824–24837, 2022. 

- [9] B. Xu, A. Yang, J. Lin, Q. Wang, C. Zhou, Y. Zhang, and Z. Mao, “Expertprompting: Instructing large language models to be distinguished experts,” _arXiv preprint arXiv:2305.14688_ , 2023. 

- [10] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig, “Pal: Program-aided language models,” in _International Conference on Machine Learning_ , pp. 10764–10799, PMLR, 2023. 

- [11] X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou, “Selfconsistency improves chain of thought reasoning in language models,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [12] M. Yasunaga, X. Chen, Y. Li, P. Pasupat, J. Leskovec, P. Liang, E. H. Chi, and D. Zhou, “Large language models as analogical reasoners,” _International Conference on Learning Representations_ , 2024. 

- [13] Z. Zhang, A. Zhang, M. Li, and A. Smola, “Automatic chain of thought prompting in large language models,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

12 

- [14] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan, “Tree of thoughts: Deliberate problem solving with large language models,” _Advances in Neural Information Processing Systems_ , vol. 36, 2024. 

- [15] M. Suzgun and A. T. Kalai, “Meta-prompting: Enhancing language models with task-agnostic scaffolding,” _arXiv preprint arXiv:2401.12954_ , 2024. 

- [16] D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. V. Le, _et al._ , “Least-to-most prompting enables complex reasoning in large language models,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [17] M. Besta, N. Blach, A. Kubicek, R. Gerstenberger, M. Podstawski, L. Gianinazzi, J. Gajda, T. Lehmann, H. Niewiadomski, P. Nyczyk, _et al._ , “Graph of thoughts: Solving elaborate problems with large language models,” in _Proceedings of the AAAI Conference on Artificial Intelligence_ , vol. 38, pp. 17682–17690, 2024. 

- [18] A. Asai, S. Min, Z. Zhong, and D. Chen, “Retrieval-based language models and applications,” in _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pp. 41–46, 2023. 

- [19] G. Mialon, R. Dessi, M. Lomeli, C. Nalmpantis, R. Pasunuru, R. Raileanu, B. Roziere, T. Schick, J. Dwivedi-Yu, A. Celikyilmaz, _et al._ , “Augmented language models: a survey,” _Transactions on Machine Learning Research_ , 2023. 

- [20] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W.-t. Yih, “Replug: Retrieval-augmented black-box language models,” _arXiv preprint arXiv:2301.12652_ , 2023. 

- [21] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang, “Retrieval-augmented generation for large language models: A survey,” _arXiv preprint arXiv:2312.10997_ , 2023. 

- [22] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y. Geng, F. Fu, L. Yang, W. Zhang, and B. Cui, “Retrieval-augmented generation for ai-generated content: A survey,” _arXiv preprint arXiv:2402.19473_ , 2024. 

- [23] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark, _et al._ , “Improving language models by retrieving from trillions of tokens,” in _International conference on machine learning_ , pp. 2206–2240, PMLR, 2022. 

- [24] M. Yasunaga, A. Aghajanyan, W. Shi, R. James, J. Leskovec, P. Liang, M. Lewis, L. Zettlemoyer, and W.-T. Yih, “Retrieval-augmented multimodal language modeling,” in _International Conference on Machine Learning_ , pp. 39755–39769, PMLR, 2023. 

- [25] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave, “Atlas: Few-shot learning with retrieval augmented language models,” _Journal of Machine Learning Research_ , vol. 24, no. 251, pp. 1–43, 2023. 

- [26] Z. Wang, W. Nie, Z. Qiao, C. Xiao, R. Baraniuk, and A. Anandkumar, “Retrieval-based controllable molecule generation,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [27] L. Yang, Z. Huang, X. Zhou, M. Xu, W. Zhang, Y. Wang, X. Zheng, W. Yang, R. O. Dror, S. Hong, _et al._ , “Prompt-based 3d molecular diffusion models for structure-based drug design,” 2023. 

- [28] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” _Advances in neural information processing systems_ , vol. 35, pp. 22199–22213, 2022. 

- [29] O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis, “Measuring and narrowing the compositionality gap in language models,” in _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 5687–5711, 2023. 

- [30] S. Arora, A. Narayan, M. F. Chen, L. Orr, N. Guha, K. Bhatia, I. Chami, and C. Re, “Ask me anything: A simple strategy for prompting language models,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [31] T. Khot, H. Trivedi, M. Finlayson, Y. Fu, K. Richardson, P. Clark, and A. Sabharwal, “Decomposed prompting: A modular approach for solving complex tasks,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [32] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, _et al._ , “Emergent abilities of large language models,” _Transactions on Machine Learning Research_ , 2022. 

13 

- [33] F. Shi, M. Suzgun, M. Freitag, X. Wang, S. Srivats, S. Vosoughi, H. W. Chung, Y. Tay, S. Ruder, D. Zhou, _et al._ , “Language models are multilingual chain-of-thought reasoners,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [34] Y. Fu, H. Peng, A. Sabharwal, P. Clark, and T. Khot, “Complexity-based prompting for multi-step reasoning,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [35] M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. Le, E. Chi, D. Zhou, _et al._ , “Challenging big-bench tasks and whether chain-of-thought can solve them,” in _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 13003–13051, 2023. 

- [36] H. S. Zheng, S. Mishra, X. Chen, H.-T. Cheng, E. H. Chi, Q. V. Le, and D. Zhou, “Take a step back: Evoking reasoning via abstraction in large language models,” _arXiv preprint arXiv:2310.06117_ , 2023. 

- [37] P. Zhou, J. Pujara, X. Ren, X. Chen, H.-T. Cheng, Q. V. Le, E. H. Chi, D. Zhou, S. Mishra, and H. S. Zheng, “Self-discover: Large language models self-compose reasoning structures,” _arXiv preprint arXiv:2402.03620_ , 2024. 

- [38] W. Chen, X. Ma, X. Wang, and W. W. Cohen, “Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks,” _Transactions on Machine Learning Research_ , 2023. 

- [39] X. Ning, Z. Lin, Z. Zhou, Z. Wang, H. Yang, and Y. Wang, “Skeleton-of-thought: Large language models can do parallel decoding,” in _The Twelfth International Conference on Learning Representations_ , 2023. 

- [40] Y. Zhang, “Meta prompting for agi systems,” _arXiv preprint arXiv:2311.11482_ , 2023. 

- [41] J. Chen, R. Xu, Z. Fu, W. Shi, Z. Li, X. Zhang, C. Sun, L. Li, Y. Xiao, and H. Zhou, “E-kar: A benchmark for rationalizing natural language analogical reasoning,” in _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 3941–3955, 2022. 

- [42] O. Sultan and D. Shahaf, “Life is a circus and we are the clowns: Automatically finding analogies between situations and processes,” in _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 3547–3562, 2022. 

- [43] N. Zhang, L. Li, X. Chen, X. Liang, S. Deng, and H. Chen, “Multimodal analogical reasoning over knowledge graphs,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [44] B. Bhavya, J. Xiong, and C. Zhai, “Analogy generation by prompting large language models: A case study of instructgpt,” in _Proceedings of the 15th International Conference on Natural Language Generation_ , pp. 298–312, 2022. 

- [45] B. Bhavya, J. Xiong, and C. Zhai, “Cam: A large language model-based creative analogy mining framework,” in _Proceedings of the ACM Web Conference 2023_ , pp. 3903–3914, 2023. 

- [46] Z. Zhang, A. Zhang, M. Li, and A. Smola, “Automatic chain of thought prompting in large language models,” in _The Eleventh International Conference on Learning Representations_ , 2022. 

- [47] T. Webb, K. J. Holyoak, and H. Lu, “Emergent analogical reasoning in large language models,” _Nature Human Behaviour_ , vol. 7, no. 9, pp. 1526–1541, 2023. 

- [48] J. Yu, R. He, and Z. Ying, “Thought propagation: An analogical approach to complex reasoning with large language models,” in _International Conference on Learning Representations_ , 2024. 

- [49] T. Feng, P. Han, G. Lin, G. Liu, and J. You, “Thought-retriever: Don’t just retrieve raw data, retrieve thoughts,” in _ICLR 2024 Workshop: How Far Are We From AGI_ . 

- [50] B. bench authors, “Beyond the imitation game: Quantifying and extrapolating the capabilities of language models,” _Transactions on Machine Learning Research_ , 2023. 

- [51] T. Schuster, A. Kalyan, A. Polozov, and A. T. Kalai, “Programming puzzles,” in _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2021. 

- [52] A. T. K. Patrick Haluptzok, Matthew Bowers, “Language models can teach themselves to program better,” in _Eleventh International Conference on Learning Representations (ICLR)_ , 2023. 

- [53] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, “Training verifiers to solve math word problems,” _arXiv preprint arXiv:2110.14168_ , 2021. 

14 

- [54] G. Chen, S. Dong, Y. Shu, G. Zhang, J. Sesay, B. F. Karlsson, J. Fu, and Y. Shi, “Autoagents: A framework for automatic agent generation,” _arXiv preprint arXiv:2309.17288_ , 2023. 

- [55] Q. Wu, G. Bansal, J. Zhang, Y. Wu, S. Zhang, E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang, “Autogen: Enabling next-gen llm applications via multi-agent conversation framework,” _arXiv preprint arXiv:2308.08155_ , 2023. 

15 

## **A Additional Method Details** 

## **A.1 Detailed Thought-Templates** 

Here we show six example thought-templates in six different categories: 

## **A.1.1 Text Comprehension** 

## **Task Description:** 

The task involves analyzing a table with various attributes of penguins, such as name, age, height, and weight, and answering questions about these attributes. The table may be updated with new entries, and additional context or comparisons may be provided in natural language. 

## **Solution Description:** 

To accurately answer questions about the penguins’ attributes, one must be able to interpret the data presented in tabular form, understand any additional information provided in natural language, and apply logical reasoning to identify the correct attribute based on the question asked. 

## **Thought Template:** 

Step 1: Parse the initial table, extracting the header information and each penguin’s attributes into a structured format (e.g., a list of dictionaries). 

Step 2: Read and integrate any additional natural language information that updates or adds to the table, ensuring the data remains consistent. 

Step 3: Identify the attribute in question (e.g., oldest penguin, heaviest penguin) and the corresponding column in the table. 

Step 4: Apply logical reasoning to compare the relevant attribute across all entries to find the correct answer (e.g., the highest age for the oldest penguin). 

Step 5: Select the answer from the provided options that matches the result of the logical comparison. 

## **A.1.2 Creative Language Generation** 

## **Task Description:** 

The task is to generate a sonnet that adheres to the traditional English sonnet rhyme scheme of "ABAB CDCD EFEF GG" and includes three specific words verbatim in the text. 

## **Solution Description:** 

Writing a sonnet involves crafting 14 lines of poetry that follow a specific rhyme pattern. The lines are typically in iambic pentameter, though flexibility in rhythm can be allowed for creative reasons. The given rhyme scheme dictates the end sounds of each line, ensuring a structured poetic form. Incorporating the three provided words verbatim requires strategic placement within the lines to maintain the poem’s coherence and thematic unity. **Thought Template:** 

Step 1: Identify the three words that must be included in the sonnet. 

Step 2: Understand the rhyme scheme "ABAB CDCD EFEF GG" and prepare a list of rhyming words that could be used. 

Step 3: Develop a theme or story for the sonnet that can naturally incorporate the three provided words. 

Step 4: Begin drafting the sonnet by writing the first quatrain (four lines) following the "ABAB" rhyme scheme, ensuring one or more of the provided words are included. Step 5: Continue with the second quatrain "CDCD," the third quatrain "EFEF," and finally the closing couplet "GG," each time incorporating the provided words as needed. Step 6: Review the sonnet for coherence, flow, and adherence to the rhyme scheme, making adjustments as necessary. 

16 

## **A.1.3 Common Sense Reasoning** 

## **Task Description:** 

Given a specific date and an event, such as a holiday or historical event, determine the following date. 

## **Solution Description:** 

To determine the next date, we need to consider the structure of the calendar, the number of days in each month, and whether it’s a leap year. Typically, the number of days in a month is fixed, except February may vary due to leap years. The next day in a year is usually the date increased by one day unless it’s the end of the month, then the next day will be the first day of the following month. For the end of the year, the next day will be January 1st of the following year. 

## **Thought Template:** 

Step 1: Identify the given date’s month and day number. 

Step 2: Check if it’s the end of the month; if so, confirm the start date of the next month. Step 3: If it’s not the end of the month, simply add one to the day number. Step 4: Pay special attention to the end of the year, ensuring the year increments. 

## **A.1.4 Mathematical Reasoning** 

## **Task Description:** 

Solve an quadratic equation of the form _ax_[2] + _bx_ + _c_ = 0 considering any situations. 

## **Solution Description:** 

To solve any quadratic equation of the form _ax_[2] + _bx_ + _c_ = 0, we can follow a general approach based on the method described. Here is the structured template for solving such equations: 

## **Thought Template:** 

- Step 1: Calculate the Discriminant - Compute the discriminant _D_ using the formula _D_ = _b_[2] _−_ 4 _ac_ . Step 2: Determine the Nature of the Roots - If _D >_ 0, the equation has two distinct real roots. 

- If _D_ = 0, the equation has exactly one real root (also known as a repeated or double root). - If _D <_ 0, the equation has two complex roots. ~~_√_~~ _D_ Step 3: Compute the Roots - For _D ≥_ 0, calculate the roots using the formula _x_ = _[−][b][±]_ 2 _a_ . - For _D <_ 0, calculate the real and imaginary parts of the complex roots using the formula ~~_√_~~ _−D x_ = _[−][b]_ 2 _a[±]_ 2 _a[i]_[, where] _[ i]_[ is the imaginary unit.] 

17 

18 

## **A.1.5 Code Programming** 

**Task Description:** When given a list of numbers, try to utilize 4 basic mathematical operations (+-*/) to get a target number. 

## **Thought Template:** 

Listing 1: Python template `from itertools import permutations , product def perform_operation (a, b, operation): # Define the operation logic (e.g., addition , subtraction , etc.). pass def evaluate_sequence (sequence , operations): # Apply operations to the sequence and check if the result meets the criteria. pass def generate_combinations (elements , operations): # Generate all possible combinations of elements and operations. pass def format_solution (sequence , operations): # Format the sequence and operations into a human -readable string. pass def find_solution(input_elements , target_result ): # Data Input Handling # Validate and preprocess input data if necessary. # Core Algorithm Logic for sequence in permutations( input_elements ): for operation_combination in generate_combinations ( sequence , operations): try : if evaluate_sequence (sequence , operation_combination ) == target_result : # Data Output Formatting return format_solution (sequence , operation_combination ) except Exception as e: # Error Handling # Handle specific exceptions that may occur during evaluation. continue # If no solution is found after all iterations , return a default message. # return No solution found message return # Example usage: input_elements = [1, 7, 10, 3] target_result = 24 print (find_solution(input_elements , target_result ))` 

19 

## **A.1.6 Application Scheduling** 

**Task Description:** Given some Chess moves in SAN, update the chess board state. Listing 2: Python template `import chess def find_checkmate_move (moves_san): # Initialize a new chess board board = chess.Board () # Apply the moves to the board for move_san in moves_san: # Remove move numbers and periods (e.g., "1." or "2.") if len (move_san.split( ’.␣’)) > 1: move_san = move_san.split( ’.␣’)[1] # Skip empty strings resulting from the removal if move_san: # Apply each move in SAN format to the board move = board.parse_san(move_san) board.push(move) # Generate all possible legal moves from the current position for move in board.legal_moves: # Make the move on a copy of the board to test the result board_copy = board.copy () board_copy.push(move) # Check if the move results in a checkmate if board_copy.is_checkmate (): # Return the move that results in checkmate in SAN format return board.san(move) # return No solution found message return #Example usage: input = ’...... ’ # Check input format and transform the input into legal format # Remove move numbers and periods (e.g., "1." or "2.") checkmate_move = find_checkmate_move (moves_san) print (checkmate_move)` 

20 

## **A.2 Prompt for Problem Distiller** 

## _**[Problem Distiller]** :_ 

_As a highly professional and intelligent expert in information distillation, you excel at extracting essential information to solve problems from user input queries. You adeptly transform this extracted information into a suitable format based on the respective type of the issue._ 

_Please categorize and extract the crucial information required to solve the problem from the user’s input query, the distilled information should include._ 

_1. Key information:_ 

_Values and information of key variables extracted from user input, which will be handed over to the respective expert for task resolution, ensuring all essential information required to solve the problem is provided._ 

_2. Restrictions:_ 

_The objective of the problem and corresponding constraints._ 

_3. Distilled task:_ 

_Extend the problem based on 1 and 2, summarize a meta problem that can address the user query and handle more input and output variations. Incorporate the real-world scenario of the extended problem along with the types of key variables and information constraints from the original problem to restrict the key variables in the extended problem. After that, use the user query input key information as input to solve the problem as an example._ 

## **A.3 Prompt for Instantiated Reasoning** 

## _**[Meta Reasoner]**_ 

_You are a Meta Reasoner who are extremely knowledgeable in all kinds of fields including Computer Science, Math, Physics, Literature, History, Chemistry, Logical reasoning, Culture, Language..... You are also able to find different high-level thought for different tasks. Here are three reasoning sturctures:_ 

## _**i) Prompt-based structure:**_ 

_It has a good performance when dealing with problems like Common Sense Reasoning, Application Scheduling_ 

## _**ii) Procedure-based structure**_ 

_It has a good performance when dealing with creative tasks like Creative Language Generation, and Text Comprehension_ 

## _**iii) Programming-based:**_ 

_It has a good performance when dealing with Mathematical Reasoning and Code Programming, it can also transform real-world problems into programming problem which could be solved efficiently._ 

## _(_ _**Reasoning instantiation** )_ 

_Your task is:_ 

_1. Deliberately consider the context and the problem within the distilled respond from problem distiller and use your understanding of the question within the distilled respond to find a domain expert who are suitable to solve the problem._ 

_2. Consider the distilled information, choose one reasoning structures for the problem. 3. If the thought-template is provided, directly follow the thought-template to instantiate for the given problem._ 

21 

