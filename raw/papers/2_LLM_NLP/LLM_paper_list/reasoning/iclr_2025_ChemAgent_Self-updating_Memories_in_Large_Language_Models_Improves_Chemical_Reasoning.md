# - CHEMAGENT: SELF UPDATING LIBRARY IN LARGE - LANGUAGE MODELS IMPROVES CHEMICAL REASON ING 

**Xiangru Tang**[1] _[,]_ **, Tianyu Hu** _[∗]_[1] _[,][∗]_ **, Muyang Ye**[1] _[,][∗]_ **, Yanjun Shao**[1] _[,][∗]_ **, Xunjian Yin**[1] **, Siru Ouyang**[2] **, Wangchunshu Zhou, Pan Lu**[3] **, Zhuosheng Zhang**[4] **, Yilun Zhao**[1] **, Arman Cohan**[1] **, Mark Gerstein**[1] 

1Yale University 2UIUC 3Stanford University 4Shanghai Jiao Tong University 

xiangru.tang@yale.edu 

## ABSTRACT 

Chemical reasoning usually involves complex, multi-step processes that demand precise calculations, where even minor errors can lead to cascading failures. Furthermore, large language models (LLMs) encounter difficulties handling domainspecific formulas, executing reasoning steps accurately, and integrating code effectively when tackling chemical reasoning tasks. To address these challenges, we present ChemAgent, a novel framework designed to improve the performance of LLMs through a dynamic, self-updating library. This library is developed by decomposing chemical tasks into sub-tasks and compiling these sub-tasks into a structured collection that can be referenced for future queries. Then, when presented with a new problem, ChemAgent retrieves and refines pertinent information from the library, which we call memory, facilitating effective task decomposition and the generation of solutions. Our method designs three types of memory and a library-enhanced reasoning component, enabling LLMs to improve over time through experience. Experimental results on four chemical reasoning datasets from SciBench demonstrate that ChemAgent achieves performance gains of up to **46%** (GPT-4), significantly outperforming existing methods. Our findings suggest substantial potential for future applications, including tasks such as drug discovery and materials science. Our code can be found at https://github.com/gersteinlab/chemagent. 

## 1 INTRODUCTION 

Chemical reasoning presents unique challenges in the realm of artificial intelligence, demanding sophisticated reasoning and precise calculations beyond typical reasoning tasks (McQuarrie, 2008; Atkins et al., 2014; Talanquer, 2022; Guo et al., 2023b; Cao et al., 2024). For example, the GPT solution with CoT prompting in Figure 1 contains numerous errors in both the calculation process and the chemical constants used. Even in short reasoning chains, a single error can cascade, reducing answer quality and escalating the probability of additional errors (Liao et al., 2024; Sun et al., 2024b). 

Recent advancements in large language models (LLMs) have demonstrated capabilities in simpler scientific tasks or chemical scenarios that do not require complex reasoning (Boiko et al., 2023; Atkins et al., 2023; Wang et al., 2024a; Hu & Shu, 2023; Xiao et al., 2024; Darvish et al., 2024; Skreta et al., 2024). However, their application to complex chemical reasoning reveals significant limitations Pei et al. (2024); Li et al. (2024a). LLMs often (1) struggle to effectively utilize domainspecific formulas, (2) exhibit incorrect reasoning steps, and (3) produce errors when combining textual reasoning with Python code for calculations (Zhong et al., 2024b); here, syntax errors may arise, causing the code to fail to compile. As shown in Figure 1 (StructChem output), errors often arise from determining a constant’s incorrect form or unit. 

> _∗_ Equal contribution. The work was done when Mr. Hu and Mrs. Ye were interns at Yale University. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0002-00.png)


Figure 1: **Comparison of problem-solving approaches for a hydrogen atom energy transition problem.** The figure illustrates three different methods: **(a)** shows a standard Chain-of-Thought approach with calculation errors (in steps 3 and 4) in Wang et al. (2024a). **(b)** demonstrates the StructChem (Ouyang et al., 2024) method with formula generation and step-by-step reasoning but fails due to an incorrect constant and incorrect unit conversion (in steps 1 and 4). **(c)** presents the ChemAgent solution, featuring task decomposition, memory retrieval from the library, and reasoning, leading to the accurate final answer. 

Some Previous approaches to address these challenges in chemical reasoning tasks have focused on decomposing the reasoning process (Zhong et al., 2023), e.g., formatting the reasoning steps through techniques such as self-reflection (Shinn et al., 2023) and StructChem prompting (Ouyang et al., 2024). However, these methods often rely heavily on human-curated knowledge (Chan et al., 2023), or fixed workflows (Fu et al., 2022; Wang et al., 2023; Zhou et al., 2023). Unlike human learners who utilize a library system to retain and apply previous experiences, these methods lack the ability to remember and learn from past analogous problems. For instance, humans can abstract and store theorems or solution strategies from previous tasks and utilize this memory for future problem-solving.[1] 

Building on these insights, ChemAgent introduces a dynamic “LIBRARY” system that facilitates iterative problem-solving by continuously updating and refining its content based on task decomposition. The LIBRARY in ChemAgent serves as a comprehensive repository where decomposed chemical tasks are stored. These tasks are broken down into sub-tasks, and their solutions are compiled in the library for future reference. As new tasks are encountered, the library is updated with new sub-tasks and corresponding solutions, ensuring its content remains relevant and progressively improves over time. In parallel, inspired by human cognitive mechanisms (Osman, 2004; Kaufman, 2011), our system incorporates three main memory components: _Planning Memory_ for high-level strategies, _Execution Memory_ for specific task solutions, and _Knowledge Memory_ for fundamental chemistry principles. These memories are stored externally, allowing the model to retrieve them efficiently during the problem-solving process. Unlike previous works relevant to LLM’s external memory system (Huang et al., 2024a; Zhong et al., 2024a; Zhang et al., 2023; Li et al., 2024b; Guo et al., 2023a; Zhang et al., 2024b), we carefully integrate all these components in a complete agentic framework and allow them to update dynamically. 

These memories are stored in a structured tree-based format that allows for efficient retrieval during the problem-solving process. When ChemAgent encounters a new problem, it first decomposes the task into manageable sub-tasks, then leverages the library to retrieve relevant sub-tasks and solutions stored in its memory components. The retrieved information is validated and refined using analogous sub-tasks previously encountered, optimizing both task decomposition and solution generation. The library is dynamically updated by adding new sub-tasks and solutions as they are encountered 

> 1For a more detailed discussion of the related work, please refer to Appendix A. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
(a) Library-enhanced Reasoning (b) Library Construction<br>Task:  Given that the work function for sodium metal is<br>Task:  Calculate the de Broglie wavelength for (a) an electron with a kinetic energy of 100 eV .            2.28 eV , what is the threshold frequency  v 0 for sodium?<br>_                    The unit of the answer should be  test set nm . Solution:  First, we need to convert the work function 휙 from<br>electron volts (eV) to joules (J). This conversion can be done<br>Knowledge Memory  M k using the relation: 1 eV  = 1.602 × 10 [−19] J . …… dev set<br>Decomposition<br>LLM      Plan Memory        M p LLM Split and verify<br>Condition 1:  The work function for sodium metal is 2.28 eV .<br>formula to calculate velocity of  Sub-task 1: electron Use kinetic energy  Sub-task 2: wavelength formula to compute wavelength  Use de Broglie  …… Sub-task n:  …… Sub-task and sub-solutionCondition 2:  ……<br>LLM<br>retrieve retrieve retrieve correct<br>Find relevant  Library LLM Find relevant  (Updated) Library …… LLM (Updated) Library wrong { condition Execution Memory  1,  sub _ task 1,  sub _ solution M1 e }<br>memories memories …… Discard! { condition 2,  sub _ task 2,  sub _ solution 2}<br>[Formula 1]Sub-solution 1:    p = m * v where p is ….  [Formula 1] Sub-solution 2:   The kinetic energy can  …… Sub-solution n:  … ……<br>[Step 1]  Identify the given values:  be calculated:  KE  = [1] 2 [mv] [2] , where…<br>Summarize and generate Plan Memory   M p<br>Final  (relevant task, relevant knowledge) Library<br>If memory not found, then generate a task as imagination. answer<br>**----- End of picture text -----**<br>


Figure 2: **The diagram of our overall framework.** It contains (a) library-enhanced reasoning and (b) library construction. **(a)** illustrates how ChemAgent utilizes the library to address a new task for the test set. And **(b)** demonstrates the construction of the library over the dev set, including Plan Memory _Mp_ and Execution Memory _Me_ ). 

and validated. This iterative process ensures that the memory is continuously enriched with new strategies and solutions, mimicking human problem-solving improvements through practice. By maintaining an evolving domain knowledge base, ChemAgent enables LLMs to autonomously navigate the problem-solving process, thereby enhancing their performance on similar tasks over time. 

Our experiments are conducted on four chemical reasoning datasets from SciBench (Wang et al., 2024a) with GPT-3.5, GPT-4 (OpenAI et al., 2024), and open-source models like Llama3 (Llama Team, 2024). Experimental results indicate that ChemAgent significantly enhances the accuracy, though the degree of improvement achieved by our model varies accordingly due to the varying sizes of these datasets. Compared to the base model (take GPT-4, for example), the application of memory self-improving leads to an average increase in accuracy by 37%, with a maximum improvement of 46%. In comparison with the current state-of-the-art model, StructChem (Ouyang et al., 2024), our approach achieves an average improvement of 10% and a maximum improvement of 15% across different datasets. Furthermore, the improvement is more pronounced for stronger base models: GPT-4 exhibits greater enhancements compared to GPT-3.5 when augmented with our framework. Additionally, we analyze the role of our library system and examine the benefits of different memory updates across various stages. This analysis provides insights into how our iterative library updates contributed to performance improvements in sub-task resolution and optimal task decomposition. By continuously enriching the library with structured memory components, ChemAgent ensures that the problem-solving process is progressively refined and optimized, leading to substantial improvements in accuracy. 

## 2 METHOD 

## 2.1 PRELIMINARIES 

Analogous to how students organize and reference their problem-solving approaches for exams, our motivation for developing a library system in ChemAgent is to enhance LLMs’ ability to tackle complex problems by providing structured access to a repository of previous sub-tasks and solutions. 

The overall reasoning framework is shown in Figure 2 **(a)** . Formally, in a simplified form, given a complex and open-ended chemical problem _PS_ as input, our method aims to generate a solution _OS_ . The problem _PS_ , which comprises problem descriptions _TS_ and initial conditions _CS_ , is firstly decomposed into a series of sub-tasks _PSi_ . The solution _OS_ is then synthesized from the subsolutions _OSi_ of the sub-tasks. Each _OSi_ includes intermediate steps, such as formulae, reasoning steps, code, and calculations. Then, a final answer _AS_ is derived from the overall solution _OS_ . We evaluate the performance via the accuracy of _AS_ against a ground truth _ASg_ . 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0004-00.png)


Figure 3: Given a task _P_ , the relevant memory examples are provided in the library. Specifically, while Execution Memory ( _Me_ ) and Plan Memory ( _Mp_ ) are derived from prior experiences, Knowledge memory ( _Mk_ ) is generated by LLM based on the problem prompt. The conditions _C_ are not explicitly presented here but are embedded within _P_ and the [GOAL] of _Me_ . 

## 2.2 COMPOSITION OF THE LIBRARY 

We divide _library_ into three main memory components: planning memory, execution memory, and knowledge memory. Figure 3 provides detailed examples. The exact context of memory and corresponding prompts are in Appendix G. 

- **Planning Memory** ( _Mp_ ): This component stores high-level strategies and methodologies for approaching complex problems. 

- **Execution Memory** ( _Me_ ): This contains highly structured descriptions of specific problem contexts and their corresponding solutions, serving as detailed execution plans. 

- **Knowledge Memory** ( _Mk_ ): This houses fundamental chemistry principles and formulas, acting as a ready reference. This component is generated temporarily during the solution of a specific task and is not intended for permanent retention. 

Thus, ChemAgent enables LLMs to tackle problems proactively, form new memories from these attempts, utilize existing memories to solve complex problems, and continuously refine their solutions. Built upon the _library_ of three types of memories, ChemAgent operates in two main stages: 

1. **Initial Library Construction:** Complex problems in the development set are decomposed into structured atomic sub-tasks, which serve as building blocks of the library, consisting of static long-term memory ( _Mp_ and _Me_ ). 

2. **Inference Phase and Dynamic Memory Updating:** 

   - (a) **Memory-Enhanced Reasoning:** The memory is dynamically enriched and improved during runtime. 

   - (b) **Evaluation & refinement:** Each sub-task’s solution is evaluated on their veracity and relevance to the main task, which leads to a correction of the solution or a refinement of the overall plan. 

Figure 2 and Figure 4 illustrates this workflow. In the following sections, we explain how each part is constructed and utilized. All prompts used in our framework can be found in Appendix G. 

## 2.3 DECOMPOSITION AS ATOMIC BLOCKS 

Human problem-solvers naturally break down complex chemistry problems into smaller, manageable sub-tasks (Zhou et al., 2023). This decomposition enhances the understanding of individual components and their interactions, facilitating the resolution of the original problem in a structured manner (Johnson et al., 2017). These sub-tasks not only improve the reasoning process but also function as atomic building blocks within the execution memory, each paired with a corresponding sub-solution. Key characteristics of this decomposition include: 

4 

**Hierarchical Decomposition** : Using the strategy stored in Planning Memory, we break down complex chemistry problems into hierarchical sub-tasks. If a sub-task is too difficult to complete, it is further decomposed until it can be executed in a single step by the LLM. We store each decomposed sub-task and its intermediate sub-solutions, enabling direct retrieval of solutions for similar future problems. This hierarchical approach ensures each sub-task acts as an independent atomic block, aiding future problem-solving. 

**Structured Sub-tasks** : To maximize the efficiency of decomposition, sub-tasks are structured into four distinct parts: (i) _Task Query_ defining the specific question, (ii) _Sub-task Objectives_ outlining clear goals, (iii) _Step-by-step Solution_ detailing the method to address the sub-task, and (iv) _Guidance for Solutions_ suggesting potential approaches. 

**Atomic Blocks as Memory** : Decomposed sub-tasks serve as building blocks for memory. Identifying and recalling similar problems can be challenging, but sub-tasks often share commonalities that make it easier to leverage past experiences. These atomic blocks also function as examples in few-shot prompting, standardizing response formats and bolstering the overall problem-solving process. 

Building on the concept of sub-tasks as fundamental units, we outline the processes involved in our framework, which includes both a dynamic library system and the structured memory inside our library. This section details how we initially construct the library and memory (§2.4), utilize and update them for new tasks (§2.5), and evaluate and refine the generated solutions (§2.6). Specifically, our approach leverages the development set to build a library, which is designed to dynamically selfimprove during runtime. 

## 2.4 LIBRARY CONSTRUCTION 

The Library is constructed using the development set. As outlined in §2.3, we leverage sub-tasks decomposed from each problem as the execution memory units. Each execution memory unit _Ui_ is defined as follows, where _C_ represents the conditions of a given problem _P_ , and _Ti_ and _Oi_ denote the sub-task and its corresponding sub-solution, respectively: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0005-06.png)


Given a problem _P_ and its corresponding solution _S_ in the development set, our method begins by identifying and extracting the conditions from _P_ . We then verify these conditions for accuracy to ensure that the subsequent steps operate with precise and correctly parsed data. Based on the identified conditions, we instruct the LLMs to generate detailed 

## **Algorithm 1:** Library Construction 

**Input:** Development set _D_ , LLM _F_ , prompts _{p_ split _, p_ ref _, p_ rank _}_ **Output:** Static memory _M_ consisting of units _U_ = _{_ condition _,_ question _,_ solution _}_ **for** _(P, S) in D_ **do** Conditions _C ←F_ ( _p_ split _∥P_ ) // Verify conditions and refine if necessary _C ←F_ ( _p_ ref _∥P∥C_ ) Sub-tasks _T ←F_ ( _p_ task _∥P∥C_ ) Sub-solutions _O ←F_ ( _p_ sol _∥C∥P∥S_ ) **Assert** _len_ ( _C_ ) = _len_ ( _T_ ) = _len_ ( _O_ ) **for** _i in len_ ( _C_ ) **do** Add ( _Ci, Ti, Oi_ ) into _U U ←F_ ( _p_ rank _∥U_ ) **return** _U_ 

sub-tasks. For each identified sub-task _Ti_ , the corresponding sub-solutions _Oi_ are parsed from _S_ and assigned accordingly. Inspired by curriculum learning (Bengio et al., 2009), we then rank the memory units _U_ based on their difficulty. In addition to ranking, we discard any memory units that do not meet a predefined confidence threshold, as evaluated by the LLMs. This ensures that the memory utilized for future problem-solving is both relevant and reliable, enhancing the LLM’s ability to tackle increasingly complex chemistry problems. The detailed memory construction process is further described in Algorithm 1. 

## 2.5 LIBRARY-ENHANCED REASONING 

During testing, we first decompose a given problem into several sub-tasks. For each sub-task, we retrieve related memory units _Ur_ to aid in solving it. Specifically, we compute the similarity between the given sub-task and units stored in the memory. Memory units with similarity above a predefined threshold _θ_ are used to assist the model in determining the answer for the sub-task. 

Formally, let ( _Cj, Tj_ ) represent a sub-task decomposed from a new problem. We retrieve memory units _Ur_ that satisfy 

Similarity( _Tj, TUi_ ) _≥ θ_ for _Ui ∈Me._ 

5 

Specifically, the similarity between tasks is calculated using Llama3’s embeddings: 

Embed( _Ta_ ) _·_ Embed( _Tb_ ) Similarity( _Ta, Tb_ ) = _||_ Embed( _Ta_ ) _|| × ||_ Embed( _Tb_ ) _||[.]_ 

In scenarios where the _execution memory_ does not contain similar sub-tasks, we incorporate a selfimproving mechanism. Initially, we enrich the memory with the required information by leveraging the internal parametric knowledge of LLMs. The LLM is instructed to identify the _topic_ of the given sub-task (e.g., quantum chemistry or thermodynamics) and generate self-created chemistry problems related to that topic, adhering to specific guidelines. This forms a kind of “synthetic” execution memory. We encourage the LLMs to generate diverse problems and solutions beyond the provided examples, thereby enhancing the robustness and adaptability of the memory system. 

Moreover, the memory is continuously updated with newly solved sub-tasks and their solutions during runtime. For each newly solved sub-task _Tj_ , sub-answers from previous sub-problems are incorporated into the conditions as _Cj_ , and the execution memory is updated based on solution _Oj_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0006-04.png)


The plan memory will be updated by a summary of the overall strategy knowledge used, _Kj_ (e.g. formulas, conception, order of resolution): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0006-06.png)


This dynamic updating ensures that the memory evolves and improves over time, progressively enhancing the problem-solving capabilities. More details on self-evolution are in §2.9. 

## 2.6 EVALUATE & REFINE MODULE 

To enhance the flexibility and reliability of ChemAgent, we propose an evaluation & refinement module, to correct the planning trajectory and verify the response to each sub-task. 

As illustrated in Figure 4, after addressing a specific sub-task _Pi_ , ChemAgent examines the sub-solution _Oi_ . The system evaluates whether the sub-solution conflicts with fundamental knowledge in _Mk_ or contains common errors, such as incorrect units. If discrepancies are identified, a new sub-solution _O[′] i_ is generated by refining the original _Oi_ based on the relevant knowledge in _Mk_ and the identified mistakes. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0006-11.png)


Figure 4: **Overall framework of the evaluation & refinement module.** ChemAgent continuously modifies the solution or the comprehensive strategy until it either reaches the maximum number of trials or meets the evaluator’s criteria. 

Furthermore, if a sub-task fails due to insufficient conditions or if the evaluation determines that the sub-task’s question does not align with the main task (e.g., calculating energy using wavelength instead of calculating wavelength using given energy), the sub-tasks from _Pi_ to _Pn_ will be restructured as _Pi[′]_[to] _[ P] m[′]_[, taking into account the main task and all preceding sub-tasks.] 

The evaluation & refinement module can use a different LLM than the one used for the base reasoning process. For instance, if GPT-3.5 is the base model, GPT-4 can handle evaluation & refinement. The evaluation component judges solutions without modifying them, while the refinement component adjusts solutions based on these evaluations. This separation clarifies error identification, helping humans understand where and why mistakes occur. 

## 2.7 SETUP 

We use four chemistry datasets from SciBench (Wang et al., 2024a), and the detailed distribution of the specific fields covered by each problem in the four datasets is shown in Figure 10. Each dataset is divided into a development set ( _Dd_ ) and a test set ( _Dt_ ), with exact sizes provided in Table 6. 

6 

Table 1: Performance on the test sets of four datasets: QUAN, CHEMMC, ATKINS, and MATTER. Results are compared with baselines under two different setups: a zero-shot setting with no demonstrations and a few-shot setting with three demonstrations. Accuracy scores are computed using the approximation detailed in Section 4.3. The best results for each setup are highlighted in **bold** , and the second-best results are underlined. In some settings, the two modules described in Sections 2.4 (Memory) and 2.6 (Evaluation & refinement) are disabled, helping illustrate each module’s impact on the overall performance. 

|Models<br>CHEMMC<br>MATTER<br>ATKINS|Models<br>CHEMMC<br>MATTER<br>ATKINS|Models<br>CHEMMC<br>MATTER<br>ATKINS|QUAN<br>Avg.|
|---|---|---|---|
|_Baselines, all based on GPT-4 (gpt-4-1106-preview)_||||
|Few-shot + Direct reasoning<br>28.21<br>14.29<br>Few-shot + Python<br>38.46<br>34.69<br>StructChem<br>58.97<br>30.67||20.69<br>57.01<br>59.81|14.71<br>19.48<br>**44.12**<br>43.57<br>41.18<br>47.66|
|_ChemAgent, all based on GPT-4 (gpt-4-1106-preview)_||||
|ChemAgent<br>**74.36**<br>w/oEvaluation & Refinement<br>61.54<br>w/oMemory,Evaluation & Refinement<br>58.97|**48.98**<br>**61.18**<br>44.89<br>57.94<br>38.78<br>57.94||**44.12**<br>**57.16**<br>**44.12**<br>52.12<br>41.18<br>49.22|
|_all based on Llama 3.1-7b_||||
|Direct reasoning<br>28.20<br>ChemAgent<br>**56.41**|10.20<br>**22.43**<br>**12.24**<br>19.63||8.82<br>17.41<br>**14.71**<br>**25.75**|
|_all based on Llama 3.1-70b_||||
|Direct reasoning<br>33.33<br>ChemAgent<br>**64.10**|30.61<br>30.84<br>**32.65**<br>**43.93**||23.53<br>29.48<br>**29.41**<br>**42.52**|
|_all based on Qwen 2.5-72b_||||
|Direct reasoning<br>48.72<br>ChemAgent<br>**69.23**|32.65<br>**57.01**<br>**44.90**<br>56.07||35.50<br>43.47<br>**44.12**<br>**53.58**|



According to previous research (Ouyang et al., 2024), SciBench chemical sets are representative of chemical reasoning tasks, addressing a broad range of queries typically encountered in this field. 

During the reasoning stage, we configure the planning memory ( _Mp_ ) to provide a maximum of two related memory instances (2-shot) for each query, and the execution memory ( _Me_ ) to provide up to four related instances (4-shot). However, during the construction of the library, only the knowledge memory ( _Mk_ ) is used, as the standard solutions are already available in the development set ( _Dd_ ). We evaluate the accuracy by comparing their outputs with the correct answers, using a relative tolerance of 0.01. 

We consider three baselines in alignment with the evaluation paradigm in SciBench (detailed instructions are provided in Appendix G): **(1) Few-shot + Direct reasoning** involves directly feeding the problem into the LLM without any additional instructions, the source of the data is StructChem (Ouyang et al., 2024). **(2) Few-shot + Python** combines few-shot examples with a tool-augmented strategy. Here, we provide six examples for every query, each consisting of a problem, its corresponding solution, and Python code. These examples are taken from _Dd_ . The results are from the original benchmark, SciBench (Wang et al., 2024a). **(3) StructChem** (Ouyang et al., 2024) employs structured instruction, confidence-based review, and refinement to guide LLMs through precise stepby-step reasoning. 

## 2.8 RESULTS 

We report the performance of all methods regarding accuracy score for each sub-dataset and an average score across the entire dataset. The results are summarized in Tables 1, 4 and 5. Additional results and analysis of experiments conducted on other LLMs can be found in Appendix B. 

Firstly, ChemAgent consistently outperforms the other baselines across various datasets and settings. Specifically, in terms of the average score, ChemAgent improves by **9.50%** (47.66 vs. 57.16) over StructChem, which is a 2.93 times increase and by **37%** (19.48 vs. 57.16) over direct reasoning, which is a 2.93 times increase. Notably, the performance gain varies across different datasets. In the CHEMMC dataset, our method exhibits the largest improvement, with a **46%** increase (28.21 vs. 74.36) compared to the direct reasoning setup. Secondly, the results also highlight the crucial role of memory in our library. The version of our framework equipped with memory consistently outperforms the version without memory across all cases. Specifically, there is an absolute improvement 

7 

of **2.90%** in terms of the average score. This underscores the importance of memory in retaining and leveraging past information to improve the accuracy and reliability of solving chemistry problems. Additionally, the Evaluation & Refinement module plays a significant role when using stronger LLMs. For instance, incorporating Evaluation & Refinement with GPT-4 increases ChemAgent’s performance by **5.04%** compared to ChemAgent with memory alone. In summary, ChemAgent demonstrates superior performance across various LLM backbones, highlighting the critical importance of the library system, memory design, and evaluation modules in enhancing problem-solving capabilities. Appendix C details the specific aspects and reasons for our method’s improved performance. 

## 2.9 SELF-EVOLUTION DURING RUNTIME 

Moreover, we aim to show that library systems with evolving memory perform better when exposed to an increasing number of problems. Much like humans improve their skills through practice, these systems benefit from continuous exposure to new tasks. 

We allow ChemAgent to dynamically update and enrich its library during the test stage to analyze this self-evolution process. Specifically, in iteration _Ii_ , ChemAgent uses all accumulated long-term memory from iterations _I_ 1 to _Ii−_ 1 as its library. When solving a new problem _P_ , all related responses and knowledge from that process are added to the library if the solution is correct. This means that in subsequent iterations, the system can leverage the newly acquired information (updated _Mp_ and _Me_ ) to improve its performance. We employ 2-shot _Mp_ and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0008-04.png)


Figure 5: **Self-evolving analysis.** We test ChemAgent twice for each iteration, and the difference between the two results serves as the error margin. All the experiments here are done on MATTER dataset. 

4-shot _Me_ during reasoning but simplify the setup by removing the evaluation and refinement modules. To ensure accuracy and prevent target leakage, the memory derived from problem _Pi_ in iteration _Ij_ is excluded when solving _Pi_ again in _Ik_ for any _j < k_ . 

This iterative process demonstrates that as the memory pool grows with each new example, ChemAgent’s problem-solving performance improves. Figure 5 shows that as the number of iterations increases, the agent’s performance gradually improves and converges to a score higher than the baseline (44 _._ 89%). This improvement indicates that ChemAgent can enhance its performance through a simple correct-or-not evaluation of past solutions instead of human-written high-quality trajectories. 

## 2.10 COST ANALYSIS 

On average, each problem requires 0.012 million tokens without the Evaluation & Refinement module, equivalent to around $0.09 per example. When this module is applied, the average token consumption per problem increases to approximately 0.023 million, costing about $0.1725 per example. Note that the initial library construction is not included in these calculations as it is a one-time setup. Based on the results in Figure 6, the computational time required by our method is reasonable. While the cost and resource consumption are slightly higher than StructChem, the improvements justify the additional expense. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0008-10.png)


Figure 6: **Cost Analysis.** The size of each bubble corresponds to the average number of inferences for each method, while the _y-axis_ indicates the average accuracy across the four datasets. 

## 2.11 ERROR ANALYSIS 

We conduct an analysis of the trajectories for the failed examples and find three types of errors, shown in Figure 7. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0009-00.png)


Figure 7: **Error analysis.** This example highlights a typical incorrect solution, which can be attributed to three main types of errors. Specifically, this problem pertains to an adiabatic process. 

**(1) Lack of Understanding of the Question.** We observe that plans often fail when the problem text contains critical hidden information (e.g., in Figure 7, the terms ‘reversibly’ and ’adiabatically’ are key) or include excessive, redundant details (e.g., the exact conditions such as “250K” are irrelevant to the solution). This challenge is understandable, as even human solvers can be misled by such information and err in their approach. This issue is prevalent and independent of whether the task is tackled by our model or directly queried to the LLM, suggesting that these errors may be inherent limitations of the LLM’s capacity. Hence, improving the ability of the foundation model to handle such nuances could significantly enhance performance. 

**(2) Inaccurate Reasoning.** As indicated before (Ouyang et al., 2024), the planning capabilities of LLMs remain insufficient for handling complex problems. Incorrect planning for the reasoning chains exacerbates the problem-solving process, as subsequent decisions and actions are based on initial problem decompositions. This issue persists until the Evaluation & Refinement module detects an error, which may not happen promptly enough to correct the trajectory. 

**(3) Incorrect Memory Selection.** While ChemAgent with memory demonstrates superior performance compared to setups without memory, it sometimes invokes misleading information, even when the similarity between the invoked memory and the problem is high. This indicates a need for more sophisticated memory retrieval and utilization strategies. 

In Figure 7, the invoked memory and sub-task 1.1 share considerable similarities—both concern entropy change during a compression process. However, the subtle difference is that the current problem involves an adiabatic process, whereas the examples in memory do not. This seemingly minor distinction can lead to significant changes in the problem-solving strategy. Distinguishing between misleading and beneficial memories remains a challenging issue, as invoked memories and problem texts may be semantically similar yet differ in critical aspects. 

## 3 ABLATION STUDY 

## 3.1 MEMORY COMPONENT ANALYSIS 

To understand why our method performs particularly well and which memory component contributes the most, we conduct an ablation study by independently removing each memory component. We test these different settings on all four sub-datasets using GPT-4 as the base LLM. The results are shown in Table 2. As mentioned in §2.3, the execution memory also serves as a few-shot prompting strategy. Therefore, when we remove _Me_ , we add two fixed human-written few-shot examples (provided with our code) into each query of each sub-task. These examples are selected from 

9 

Table 2: The best performing score is highlighted in **bold** and second-best is underlined. The number of problems in each textbook weighs the average score. We also provide the results of ChemAgent without any of these memories as a baseline. Evaluation & refinement module is removed. 

|Memory component<br>CHEMMC<br>MATTER<br>ATKINS<br>QUAN|Memory component<br>CHEMMC<br>MATTER<br>ATKINS<br>QUAN|Memory component<br>CHEMMC<br>MATTER<br>ATKINS<br>QUAN|Memory component<br>CHEMMC<br>MATTER<br>ATKINS<br>QUAN|Avg.|
|---|---|---|---|---|
|None<br>+_Mp Me Mk_<br>+_Mp Me_<br>+_Me Mk_<br>+_Mp Mk_ few-shot|58.97<br>61.54<br>66.66<br>58.97<br>**74.36**|38.78<br>44.89<br>44.89<br>**46.94**<br>42.86|57.94<br>41.18<br>57.94<br>**44.12**<br>52.34<br>41.18<br>**61.68**<br>29.41<br>57.94<br>32.35|51.53<br>**53.71**<br>51.53<br>53.27<br>**53.71**|



the development set ( _Dd_ ) corresponding to each dataset to ensure they do not provide misleading information. 

Firstly, removing any memory component results in an overall performance drop. And notably, the relevant knowledge in LLM itself ( _Mk_ ) significantly impacts the overall performance of ChemAgent. Delving into why _Mk_ contributes the most, we find that this might be due to the insufficient _Mp_ and _Me_ in the Atkins dataset. As shown in Table 6, ATKINS has the lowest ratio of the development set to the test set, while it also has the largest test set. This imbalance may result in a small and sometimes irrelevant memory pool for _Mp_ and _Me_ to search from. We hypothesize that this issue can be mitigated when the model is exposed to more chemical questions over time. However, due to limited computational resources, we cannot scale up to verify this hypothesis in this research. 

It is also worth noting that although the _Mp, Mk, fewshot_ setting achieves the same accuracy as the complete memory setting, most of its improvement is evident in the CHEMMC dataset, showing surprisingly higher improvements than others. Since Table 6 and Figure 10 demonstrate that the CHEMMC dataset has a higher ratio of _Dd_ to _Dt_ and a relatively narrow distribution in subfields, and since human experts carefully wrote the few-shot examples, these high-quality, expert-written examples may serve as a superior memory pool in such circumstances. In the next section, we show that better memory leads to better results, which might explain this phenomenon. 

## 3.2 MEMORY QUALITY INFLUENCE 

We investigate the impact of memory quality through a series of experiments. Specifically, we compare the performance of memory generated by GPT-4 against that generated by GPT3.5 on the MATTER dataset. Additionally, we create a ”hybrid memory” by mixing memories generated by both GPT-3.5 and GPT-4 to observe its performance. These experiments were conducted without including the relevant knowledge within the LLM itself ( _Mk_ ). As mentioned in §2.1, unlike _Mp_ and _Me_ , _Mk_ 

Table 3: Results of memory quality analysis, with the best scores highlighting in **bold** . Here, memory refers to _Mp_ and _Me_ . The relevant knowledge in LLM itself ( _Mk_ ) is removed from the framework. Test on MATTER dataset. 

||GPT-3.5 Memory<br>GPT-4 Memory<br>Hybrid Memory|**GPT-3.5**<br>11.22<br>**13.26**<br>10.20|**GPT-4**<br>36.73<br>**44.89**<br>28.57|
|---|---|---|---|



is not preserved in the memory pool shared with other LLMs. Including this type of memory in the ablation study would result in the pollution of provided memory by the _Mk_ generated during use. 

As shown in Table 3, our data analysis indicates that memory quality significantly impacts the accuracy of responses from ChemAgent. Whether GPT-3.5 or GPT-4 is used, the memory produced by GPT-4 consistently outperforms that generated by GPT-3.5, showing an 8% variation. This underscores the substantial impact of memory quality on the accuracy of generated responses. This observation also explains why the improvements seen with GPT-3.5 are less pronounced compared to those with GPT-4; the lower quality of memory produced by GPT-3.5 is the limiting factor. 

Interestingly, hybrid memory performs the worst among the three settings. This can be attributed to the simultaneous invocation of different memories for the same question—one generated by GPT3.5 and the other by GPT-4—which may confuse the LLM, increasing the likelihood of producing irrelevant or incorrect answers. Additionally, invoking an excessive number of memory tracks during resolution may also contribute to this issue. A more detailed analysis regarding the number of memory tracks invoked is provided in Appendix D. 

10 

## 4 CONCLUSION 

Our research presents a novel approach to enhancing large language models for solving complex chemical problems through self-exploration and memory formation. This method enables models to construct and utilize a library, significantly improving response accuracy. Experiments using datasets and models like GPT-3.5, GPT-4, and Llama3 demonstrate substantial performance gains, with the ChemAgent architecture achieving up to a 36% improvement. The structured library, built through memory decomposition into planning, execution, and knowledge, enhances problemsolving capabilities, which holds promise for generalization to other domains. 

## REFERENCES 

Peter Atkins, Julio De Paula, and Ronald Friedman. _Physical chemistry: quanta, matter, and change_ . Oxford University Press, USA, 2014. 

Peter Atkins, Julio De Paula, and James Keeler. _Atkins’ physical chemistry_ . Oxford university press, 2023. 

Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In _Proceedings of the 26th annual international conference on machine learning_ , pp. 41–48, 2009. 

- Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38,16, pp. 17682–17690, 2024. 

- Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. _Nature_ , 624:570 – 578, 2023. 

- Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. ChemCrow: Augmenting large-language models with chemistry tools. _arXiv preprint arXiv:2304.05376_ , 2023. 

- He Cao, Yanjun Shao, Zhiyuan Liu, Zijing Liu, Xiangru Tang, Yuan Yao, and Yu Li. PRESTO: Progressive pretraining enhances synthetic chemistry outcomes. _arXiv preprint arXiv:2406.13193_ , 2024. 

- Aaron Chan, Zhiyuan Zeng, Wyatt Lake, Brihi Joshi, Hanjie Chen, and Xiang Ren. KNIFE: Distilling meta-reasoning knowledge with free-text rationales. In _ICLR 2023 Workshop on Pitfalls of limited data and computation for Trustworthy ML_ , 2023. 

Kourosh Darvish, Marta Skreta, Yuchi Zhao, Naruki Yoshikawa, Sagnik Som, Miroslav Bogdanovic, Yang Cao, Han Hao, Haoping Xu, Al´an Aspuru-Guzik, Animesh Garg, and Florian Shkurti. Organa: A robotic assistant for automated chemistry experimentation and characterization. _ArXiv_ , abs/2401.06949, 2024. 

Ernest Davis and Scott Aaronson. Testing GPT-4 with wolfram alpha and code interpreter plug-ins on math and science problems. _arXiv preprint arXiv:2308.05713_ , 2023. 

- Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen. SciKnowEval: Evaluating multi-level scientific knowledge of large language models. _arXiv preprint arXiv:2406.09098_ , 2024. 

- Chrisantha Fernando, Dylan Sunil Banarse, Henryk Michalewski, Simon Osindero, and Tim Rockt¨aschel. Promptbreeder: Self-referential self-improvement via prompt evolution. In _Fortyfirst International Conference on Machine Learning_ , 2023. 

- Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. In _The Eleventh International Conference on Learning Representations_ , 2022. 

11 

- Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. Rarr: Researching and revising what language models say, using language models. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 16477–16508, 2023. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Nan Duan, Weizhu Chen, et al. Critic: Large language models can self-correct with tool-interactive critiquing. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Jing Guo, Nan Li, Jian Qiang Qi, Hang Yang, Ruiqiao Li, Yuzhen Feng, Si Zhang, and Ming Xu. Empowering working memory for large language model agents. _ArXiv_ , abs/2312.17259, 2023a. 

- Taicheng Guo, Bozhao Nan, Zhenwen Liang, Zhichun Guo, Nitesh Chawla, Olaf Wiest, Xiangliang Zhang, et al. What can large language models do in chemistry? a comprehensive benchmark on eight tasks. _Advances in Neural Information Processing Systems_ , 36:59662–59688, 2023b. 

- Joseph F Hair Jr, Wiliam C Black, Barry J Babin, and Rolph E Anderson. Multivariate data analysis. In _Multivariate data analysis_ , pp. 785–785. Prentice Hall, 2010. 

- Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. _arXiv preprint arXiv:2408.08435_ , 2024. 

- Zhiting Hu and Tianmin Shu. Language models, agent models, and world models: The law for machine reasoning and planning. _arXiv preprint arXiv:2312.05230_ , 2023. 

- Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. Understanding the planning of llm agents: A survey. _ArXiv_ , abs/2402.02716, 2024a. 

- Yuqing Huang, Rongyang Zhang, Xuesong He, Xuyang Zhi, Hao Wang, Xin Li, Feiyang Xu, Deguang Liu, Huadong Liang, Yi Li, Jian Cui, Zimu Liu, Shijin Wang, Guoping Hu, Guiquan Liu, Qi Liu, Defu Lian, and Enhong Chen. Chemeval: A comprehensive multi-level chemical evaluation for large language models. _arXiv preprint arXiv:2409.13989_ , 2024b. 

- Shuyang Jiang, Yuhao Wang, and Yu Wang. Selfevolve: A code evolution framework via large language models. _arXiv preprint arXiv:2306.02907_ , 2023. 

- Zhanming Jie, Trung Quoc Luong, Xinbo Zhang, Xiaoran Jin, and Hang Li. Design of chain-ofthought in math problem solving. _arXiv preprint arXiv:2309.11054_ , 2023. 

- Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 2901–2910, 2017. 

- Scott Barry Kaufman. Intelligence and the cognitive unconscious. _The Cambridge handbook of intelligence_ , pp. 442–467, 2011. 

- Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Jiawei Li, Yizhe Yang, Yu Bai, Xiaofeng Zhou, Yinghao Li, Huashan Sun, Yuhang Liu, Xingpeng Si, Yuhao Ye, Yixiao Wu, et al. Fundamental capabilities of large language models and their applications in domain scenarios: A survey. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 11116–11141, 2024a. 

- Kun Li, Xin Jing, and Chengang Jing. Vector storage based long-term memory research on llm. _International Journal of Advanced Network, Monitoring and Controls_ , 2024b. 

12 

- Zekun Li, Xianjun Yang, Kyuri Choi, Wanrong Zhu, Ryan Hsieh, HyeonJung Kim, Jin Hyuk Lim, Sungyoung Ji, Byungju Lee, Xifeng Yan, et al. MMSci: A multimodal multi-discipline dataset for PhD-level scientific comprehension. _arXiv preprint arXiv:2407.04903_ , 2024c. 

- Chang Liao, Yemin Yu, Yu Mei, and Ying Wei. From words to molecules: A survey of large language models in chemistry. _arXiv preprint arXiv:2402.01439_ , 2024. 

- AI@Meta Llama Team. The llama 3 herd of models, July 2024. URL https://llama.meta. com/. 

- Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_ , 35:2507–2521, 2022. 

Donald A McQuarrie. _Quantum chemistry_ . University Science Books, 2008. 

- Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Benedict Emoekabu, Aswanth Krishnan, Mara Wilhelmi, Macjonathan Okereke, Juliane Eberhardt, Amir Mohammad Elahi, Maximilian Greiner, Caroline T. Holick, Tanya Gupta, Mehrdad Asgari, Christina Glaubitz, Lea C. Klepsch, Yannik K¨oster, Jakob Meyer, Santiago Miret, Tim Hoffmann, Fabian Alexander Kreth, Michael Ringleb, Nicole Roesner, Ulrich S. Schubert, Leanne M. Stafast, Dinga Wonanke, Michael Pieler, Philippe Schwaller, and Kevin Maik Jablonka. Are large language models superhuman chemists?, 2024. URL https://arxiv.org/abs/2404.01475. 

- OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, et al. GPT-4 technical report, 2024. 

- Magda Osman. An evaluation of dual-process theories of reasoning. _Psychonomic bulletin & review_ , 11(6):988–1010, 2004. 

- Siru Ouyang, Zhuosheng Zhang, Bing Yan, Xuan Liu, Yejin Choi, Jiawei Han, and Lianhui Qin. Structured chemistry reasoning with large language models. In _Forty-first International Conference on Machine Learning_ , 2024. 

- Pruthvi Patel, Swaroop Mishra, Mihir Parmar, and Chitta Baral. Is a question decomposition unit all we need? In _2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022_ , 2022. 

- Qizhi Pei, Lijun Wu, Kaiyuan Gao, Jinhua Zhu, Yue Wang, Zun Wang, Tao Qin, and Rui Yan. Leveraging biomolecule and natural language through multi-modal learning: A survey. _arXiv preprint arXiv:2403.01528_ , 2024. 

- Cheng Qian, Shihao Liang, Yujia Qin, Yining Ye, Xin Cong, Yankai Lin, Yesai Wu, Zhiyuan Liu, and Maosong Sun. Investigate-consolidate-exploit: A general strategy for inter-task agent selfevolution. _arXiv preprint arXiv:2401.13996_ , 2024. 

- Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. Recursive introspection: Teaching language model agents how to self-improve. _arXiv preprint arXiv:2407.18219_ , 2024. 

Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. _arXiv preprint arXiv:2303.11366_ , 2023. 

- Marta Skreta, Zihan Zhou, Jia Lin Yuan, Kourosh Darvish, Al´an Aspuru-Guzik, and Animesh Garg. Replan: Robotic replanning with perception and language models. _ArXiv_ , abs/2401.04157, 2024. 

- Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. SciEval: A multi-level large language model evaluation benchmark for scientific research. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38,17, pp. 19053–19061, 2024a. 

- Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, et al. Trustllm: Trustworthiness in large language models. _arXiv preprint arXiv:2401.05561_ , 2024b. 

13 

Vicente Talanquer. The complexity of reasoning about and with chemical representations. _JACS Au_ , 2(12):2658–2669, 2022. doi: 10.1021/jacsau.2c00498. 

- David Wadden, Kejian Shi, Jacob Morrison, Aakanksha Naik, Shruti Singh, Nitzan Barzilay, Kyle Lo, Tom Hope, Luca Soldaini, Shannon Zejiang Shen, et al. SciRIFF: A resource to enhance language model instruction-following over scientific literature. _arXiv preprint arXiv:2406.07835_ , 2024. 

- Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. SciBench: Evaluating college-level scientific problem-solving abilities of large language models. In _Forty-first International Conference on Machine Learning_ , 2024a. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. _arXiv preprint arXiv:2409.07429_ , 2024b. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Kangda Wei, Dawn Lawrie, Benjamin Van Durme, Yunmo Chen, and Orion Weller. When do decompositions help for machine reading? In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 3599–3606, 2023. 

- Yixuan Weng, Minjun Zhu, Shizhu He, Kang Liu, and Jun Zhao. Large language models are reasoners with self-verification. _arXiv preprint arXiv:2212.09561_ , 2, 2022. 

- Yi Xiao, Xiangxin Zhou, Qiang Liu, and Liang Wang. Bridging text and molecule: A survey on multimodal frameworks for molecule. _arXiv preprint arXiv:2403.13830_ , 2024. 

- Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. Buffer of thoughts: Thought-augmented reasoning with large language models. _arXiv preprint arXiv:2406.04271_ , 2024. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Yuan Zhang, Chao Wang, Juntong Qi, and Yan Peng. Leave it to large language models! correction and planning with memory integration. _Cyborg and Bionic Systems_ , 5, 2023. 

- Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, and Lu Wang. Small language models need strong verifiers to self-correct reasoning. _arXiv preprint arXiv:2404.17140_ , 2024a. 

- Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. _ArXiv_ , abs/2404.13501, 2024b. 

- Ming Zhong, Siru Ouyang, Minhao Jiang, Vivian Hu, Yizhu Jiao, Xuan Wang, and Jiawei Han. ReactIE: Enhancing chemical reaction extraction with weak supervision. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 12120–12130, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.767. URL https://aclanthology.org/ 2023.findings-acl.767. 

14 

- Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 38(17):19724–19731, Mar. 2024a. doi: 10.1609/aaai.v38i17.29946. URL https: //ojs.aaai.org/index.php/AAAI/article/view/29946. 

- Xianrui Zhong, Yufeng Du, Siru Ouyang, Ming Zhong, Tingfeng Luo, Qirong Ho, Hao Peng, Heng Ji, and Jiawei Han. Actionie: Action extraction from scientific literature with programming languages. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 12656–12671, 2024b. 

- Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id= WZH7099tgfM. 

- Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, et al. Symbolic learning enables self-evolving agents. _arXiv preprint arXiv:2406.18532_ , 2024. 

15 

## A RELATED WORKS 

## A.1 CHEMICAL REASONING 

Recent advances in LLMs have shown promise in scientific reasoning, yet chemical reasoning remains particularly challenging. Benchmarks like SciBench (Wang et al., 2024a) have revealed that current LLMs struggle significantly with complex chemical calculations and multi-step reasoning tasks. SciBench includes 869 college-level problems across mathematics, chemistry, and physics, providing a rigorous evaluation of LLM capabilities in these domains. Other datasets (Wadden et al., 2024; Li et al., 2024c; Sun et al., 2024a; Feng et al., 2024; Huang et al., 2024b) have also contributed to advancing the evaluation of LLMs in scientific problem-solving. 

In response to these challenges, several approaches have been proposed. StructChem (Ouyang et al., 2024) provides structured guidance by decomposing chemical reasoning into phases such as formula generation, detailed step-by-step reasoning, and confidence-based review. While showing improvements, it still faces limitations with highly complex problems. Other researchers have explored enhancing LLM performance through various prompting strategies (Yang et al., 2024; Yao et al., 2024; Besta et al., 2024). 

Some studies have focused on using Python code for reasoning tasks. Jie et al. (2023) demonstrated that self-describing programs, which transform reasoning processes into executable code, can significantly outperform natural language Chain-of-Thought methods (Wei et al., 2022) in certain scenarios. 

Specialized tools like ChemCrow (Bran et al., 2023) utilize function calling and precise code generation to tackle chemistry problems. Davis & Aaronson (2023) highlight that combining GPT-4 with external tools like Wolfram Alpha or Code Interpreter can significantly improve problem-solving in science. However, reliably integrating these tools for complex chemical calculations remains a challenge. 

## A.2 PROBLEM DECOMPOSITION IN SCIENTIFIC REASONING 

Decomposing complex problems into smaller sub-tasks has shown to enhance model understanding and accuracy across various domains. In the context of chemical reasoning, this approach is particularly relevant due to the multi-step nature of many chemical problems. Patel et al. (2022) highlights the efficacy of question decomposition by breaking down complex queries into manageable subtasks. Similarly, Khot et al. (2022) demonstrates the benefits of modular task decomposition. Other studies (Lu et al., 2022; Wei et al., 2023) further underscore the advantages of decomposition in complex question answering and reading comprehension. Our work builds on these insights, specifically focusing on how breaking down complex chemical problems can improve the performance of self-evolving agents in this domain. 

## A.3 SELF-EVOLUTION AND SELF-CORRECTION IN AGENT REASONING 

Recent research has explored the self-evolution and optimization of LLMs, which is particularly relevant for tackling the complexities of chemical reasoning. Yang et al. (2023) explore methods for enhancing LLM performance through self-improvement techniques, while Fernando et al. (2023) investigate self-referential self-improvement via prompt evolution. Additionally, Zhou et al. (2024), Jiang et al. (2023), Hu et al. (2024) and Qian et al. (2024) present frameworks for agent self-evolution, aligning with our approach of enabling self-exploration and continuous learning in complex chemical problem-solving. While some of these frameworks also incorporate a memory system, they predominantly emphasize the reuse of past workflows in daily tasks, as demonstrated by Wang et al. (2024b) and Qian et al. (2024). 

Furthermore, in chemical reasoning, where even minor errors can lead to significant discrepancies in results, the ability of LLMs to self-correct and refine their outputs is crucial. Huang et al. (2023) investigate the limitations of self-correcting mechanisms in LLMs, while Weng et al. (2022) explore the potential of LLMs as reasoners with self-verification capabilities. In terms of tool-assisted methods, Gou et al. (2023) propose a framework for LLMs to self-correct with tool-interactive critiquing, and Gao et al. (2023) offer insights into researching and revising LLM outputs using LLMs them- 

16 

selves. Other studies like Qu et al. (2024) also inform our approach to developing a self-updating memory system capable of refining its chemical reasoning processes. 

## B EXPERIMENTAL RESULTS OF MODELS OTHER THAN GPT-4 

We also evaluated ChemAgent thoroughly on earlier and less powerful models, such as GPT-3.5 (gpt-3.5-turbo-16k). Specifically, ChemAgent achieves an absolute improvement of +0.17% in terms of the average score compared with the previous SOTA. Also, our framework significantly performs better when equipped with a memory system (+7.13% absolute improvement), which underscores the importance of memory. 

However, when it comes to the evaluation and refinement module, an interesting phenomenon is observed: even when outputs generated by GPT-3.5 are evaluated by more capable models like GPT-4, the agent often fails to correct its mistakes. This indicates that GPT-3.5 has a relatively weaker self-correction ability than GPT-4, explaining why evaluation and refinement provide little benefit when GPT-3.5 is used. This finding aligns with previous research (Zhang et al., 2024a). 

Table 4: Performance on the test sets of four datasets: QUAN, CHEMMC, ATKINS, and MATTER. We compare baselines under two different settings: a zero-shot setting with no demonstrations and a few-shot setting with three demonstrations. The accuracy scores are computed with the approximation detailed in Section 4.3. The best results under each setup are highlighted in **bold** , and the second-best results are underlined. This experiment is done using GPT-3.5 (gpt-3.5-turbo-16k) 

|GPT-3.5<br>CHEMMC<br>MATTER<br>ATKINS|GPT-3.5<br>CHEMMC<br>MATTER<br>ATKINS|GPT-3.5<br>CHEMMC<br>MATTER<br>ATKINS|GPT-3.5<br>CHEMMC<br>MATTER<br>ATKINS|QUAN|Avg.|
|---|---|---|---|---|---|
|_Baselines, all based on GPT-3.5_||||||
|Few-shot+Direct reasoning<br>23.08<br>Fewshot+Python<br>33.33<br>StructChem<br>43.59||8.16<br>16.33<br>**24.49**|9.35<br>13.08<br>26.17|5.88<br>8.82<br>**32.35**|11.61<br>17.89<br>31.65|
|_ChemAgent, all based on_||_GPT-3.5_||||
|ChemAgent w/o memory<br>ChemAgent w/ memory<br>+ evaluate and refne<br>+ evaluate by GPT-4 and refne|38.46<br>**56.41**<br>41.03<br>46.15|16.33<br>18.37<br>16.33<br>22.45|23.36<br>**28.97**<br>28.04<br>**28.97**|20.59<br>23.53<br>20.59<br>26.47|24.69<br>**31.82**<br>26.50<br>31.01|



For open-source models, we choose Llama3-7b-Instruct, Llama3-70b-Instruct, and Qwen2.5-72bInstruct as the experimental model here. The baseline direct reasoning is to directly query the model without adding other instructions. The evaluation and refinement modules are removed from the ChemAgent configuration, and only the Execution Memory ( _Me_ ) is added due to the model’s relatively lower ability on instruction following. Each experiment is repeated at least 3 times, and the results are averaged. On llama3-7b, the average increase across four datasets is 8.34%. On llama3-70b, the average increase across four datasets is 13.04%. The experiment demonstrates that the stronger the self-capabilities of large models, the more pronounced the performance gains using our framework. 

Reference to Appendix F, the _atkins_ dataset, and the _matter_ dataset involve a lot of specialized knowledge in chemistry. The questions in _atkins_ involve the theoretical aspects of thermodynamics, kinetics, and other chemical processes, while _matter_ focuses on studying chemical reaction mechanisms and kinetic processes. This demands a high level of chemical expertise from the model itself. However, the Llama3-7b model, compared to GPT-4 and Llama3-70b, has a lower reservoir of chemical knowledge, thus resulting in average performance on _atkins_ and _matter_ . On the other hand, the challenges in the _quan_ and _chemmc_ datasets lie in complex computations, prediction of molecular structures, and chemical bond reactions, emphasizing the logical reasoning ability of large models. The Llama3 experiment demonstrates that the ChemAgent framework significantly enhances the model’s capability in handling logically complex chemical problems, highlighting the requirement for a certain level of subject matter expertise within the framework itself. 

17 

However, the significantly lower performance on the _atkins_ dataset by Llama3-7b requires further explanation. The development set in _atkins_ is much more challenging than other datasets. The solutions tend to require more reasoning steps in their decomposition, and the questions in the development set focus on a much narrower sub-topic than those in the test set. This results in a higher rate of mistakes during the development stage, leading to an initially lower-quality static memory pool. Consequently, this greatly influences and even misleads the agent’s performance during the test stage. 

In general, it can be seen that our approach generalizes well across different LLMs, showing consistent improvement in performance. The results indicate that our self-updating memory mechanism enhances the problem-solving capabilities of LLM-powered agents regardless of the backbone used. Notably, ChemAgent performs significantly better on GPT-4 compared to other models, suggesting that our framework becomes even more effective as base LLM models become more powerful. 

Table 5: Experimental results on Llama 3.1-70b, Llama 3.1-70b and Qwen 2.5-72b. From the average improvement across four datasets, the stronger the model, the greater the improvement. Compared to the Llama 3.1-7b model, the overall enhancement using Llama 3.1-70b is better, particularly noticeable in the CHEMMC and ATKINS datasets. 

|Models|CHEMMC|MATTER|ATKINS|QUAN|Avg.|
|---|---|---|---|---|---|
||_Llama 3.1-7b_|||||
|Direct reasoning|28.20|10.20|**22.43**|8.82|17.41|
|ChemAgent|**56.41**|**12.24**|19.63|**14.71**|**25.75**|
||_Llama 3.1-70b_|||||
|Direct reasoning|33.33|30.61|30.84|23.53|29.48|
|ChemAgent|**64.10**|**32.65**|**43.93**|**29.41**|**42.52**|
||_Qwen 2.5-72b_|||||
|Direct reasoning|48.72|32.65|**57.01**|35.50|43.47|
|ChemAgent|**69.23**|**44.90**|56.07|**44.12**|**53.58**|



- C PERFORMANCE BOOST: WHERE AND WHY OUR METHOD IMPROVES 

## C.1 CALCULATION AND UNIT CONVERSION ARE MORE PRECISE 

ChemAgent achieves notably higher accuracy in calculations and unit conversions. Two key factors contribute to this: (1) Python code is demonstrated alongside each corresponding sub-task in memory; (2) During development, ChemAgent adopted a strategy to save unit conversion steps in the long-term plan memory pool. This allows the agent to reference correct conversion steps when necessary, ensuring accurate unit conversion. 

## C.2 HIGHER MEMORY’S SIMILARITY HELPS THE SOLUTION 

When solving a given problem _P_ , a series of memories [ _U_ 1 _, . . . , Un_ ] may be invoked during the process. Let their similarity to the problem be represented as [ _S_ 1 _, . . . , Sn_ ], and the mean similarity value is denoted as _S_ mean _,P_ . In Figure 8, we visualize the distribution of _S_ mean _,P_ , categorized by whether _P_ is correctly solved. It is evident that problems with higher _S_ mean _,P_ are more likely to be solved correctly. 

We also conducted a Chi-Square Test of Independence to assess the relationship between a similarity threshold (i.e., whether the similarity is greater than 0.805) and the correctness of the solution. The Chi-Square statistic is 8 _._ 77 with a p-value of 0 _._ 003, indicating a statistically significant relationship. Thus, future work could focus on improving the similarity between invoked memories and the problem at hand to further enhance problem-solving performance. 

## D THE INFLUENCE OF THE NUMBER OF MEMORY INVOKED 

We analyze the impact of the number of memory instances used on our agent’s performance. In this analysis, we preserve the evaluation and refinement module since they also utilize memory. Given 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0019-00.png)


Figure 8: **Memory Similarity Analysis.** The probability density functions of invoked memory similarity for solved and failed tasks are visualized using Kernel Density Estimation. We use a reference value of 0 _._ 805 to do the Mann-Whitney U Test and get a p-value of 0 _._ 008. The similarity values are generally higher in the trajectories of solved tasks. 

that the quantities of _Mp_ and _Mk_ are not fixed, we focus on examining how the number of _Me_ impacts performance. The results are presented in Figure 9. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0019-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0019-04.png)


Figure 9: Test on MATTER dataset. “shots” represents the number of memory tracks given to the LLM. It demonstrates that as the number of demonstrations increases, the agent’s average performance improves, but this comes at the expense of its stability. 

As the number of invoked memories increases, the average accuracy improves, but the variance also grows. This indicates that although the agent’s performance benefits from more memory, it also becomes more unstable. The increase in accuracy suggests that with more memory, ChemAgent adheres more strictly to the required format and acquires more useful knowledge. However, the increase in variance indicates that as the number of memory instances used increases, so too does the potential for encountering misleading or irrelevant information. Our case studies in the 4-shot setting reveal that many memory instances carry some degree of irrelevant or _trash_ information, which accumulates and can harm the agent’s abilities. 

Interestingly, some seemingly irrelevant information in _Me_ may enhance the creativity of the LLM when solving specific sub-tasks, resulting in a higher maximum accuracy in the 4-shot setting. This suggests that a certain level of seemingly irrelevant information can be beneficial, allowing the agent to explore hard and unknown tasks better. 

19 

## E LIMITATION 

Despite our advances, limitations include the computational intensity and time required for selfexploration, as well as the need for further refinement of memory mechanisms for tackling more complex problems. Future research should focus on understanding the specific mechanisms by which memory formation benefits reasoning, exploring how different types of memory contribute to problem-solving, and identifying optimal strategies for memory utilization. Additionally, due to limitations in budget and computational resources, we demonstrate our approach solely in the context of chemistry and have not conducted comprehensive research across the entire scientific domain, such as considering mathematics or physics. 

However, we believe our proposed method has strong potential for generalization across scientific fields. We have made our code open-source and encourage future researchers to apply it to their own datasets. Specifically, adapting our method to a new domain (e.g., the CLASS sub-dataset in SciBench, related to Physics) involves modifying a small portion of the prompts (such as replacing chemistry-related sentences, e.g., ”You are a Chemistry expert”) and then re-starting the Library Construction phase using a development set from the new domain. 

20 

## F TASK DOMAINS OF THE DATASETS 

The Chemistry domain of SciBench has four datasets, each hand-collected from four college chemistry textbooks. _Quantum chemistry (quan)_ (Hair Jr et al., 2010) provides an exploration of equilibrium, structure, and reactions. _Chemistry kinetics (matter)_ (Atkins et al., 2014) combines physics and mathematics, spanning through quantum mechanics and atomic structure. _Quantum mechanics (chemmc)_ (McQuarrie, 2008) covers quantum mechanics and the applications in chemical bonding. _Physical chemistry (atkins)_ (Atkins et al., 2023) provides explorations of equilibrium, structure, and reactions. We leverage GPT-4 to annotate each data sample in these datasets for the specific subfields. Statistically, the four datasets cover 15 chemical subfields, and the division of chemical subfields also helps us make associations in related fields to enrich the memory pool. The subfields of the datasets are shown in the figure 10. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0021-02.png)


Figure 10: Chemical subfields covered by the four datasets. The subfields involved in each dataset are different, and a task may require knowledge from multiple subfields. 

Table 6: Detailed statistics and information of the four datasets we experiment with. _Dt_ and _Dd_ refer to the number of data samples with and without solutions (development set and test set). “ _Rd/t_ ” means the ratio of the development set to the test set. 

|Datasets|Topics|_Dt_(_Dd_)|_Rd/t_|
|---|---|---|---|
|CHEMMC|Quantum mechanics|39 (9)|0.231|
|MATTER|Chemistry kinetics|49 (10)|0.204|
|ATKINS|Physical chemistry|107 (16)|0.150|
|QUAN|Quantum chemistry|34 (8)|0.235|



The problems in this dataset are challenging, with an average of 2 formulas and 5 steps of reasoning required to solve the problems in the experiment (Ouyang et al., 2024). Meanwhile, each dataset provides detailed step-by-step solutions for example problems, which fit well in our framework for the initial construction of memory pools, shown in Table 6. 

21 

## G PROMPTS 

Here is the list of prompts we used in our study. 

## G.1 **INSTRUCTIONS FOR BUILDING MEMORY POOLS.** 

(1) Split prompts direct ChatGPT to decompose a given task into condition and problem parts. Reflect prompts are used to double-check the results of the initial decomposition to confirm that the decomposition is complete and correct and to return corrected results in the event of errors. See Figure 11. 

Task Split Prompt for Memory Pool Development 

## **SPLIT** ~~**P**~~ **ROMPT** ~~**S**~~ **YS** = 

You are ChatGPT, a large language model trained by OpenAI, also an excellent prompt evaluator, who is capable of splitting tasks and evaluating the difficulty of tasks which will be solved by other autonomous agents powered by LLM. 

## **SPLIT** ~~**P**~~ **ROMPT** = 

Now, I have a target task and its solution here: task: task solution: solution And I also have a Python environment where agents can write code. (It is also okay not to use them and just use your knowledge): Then, split or separate the sentence of the original task into two parts: one is conditions, and the other is questions. 

Below is the format you should follow when you give the answer: 

*** 

CONDITIONS: QUESTIONS: 

*** 

## **REFLECT** ~~**P**~~ **ROMPT** ~~**S**~~ **YS** = 

You are ChatGPT, a large language model trained by OpenAI, and also an excellent answer checker who is capable of figuring out whether some conditions and questions separated from the original task are complete and correct. 

## **REFLECT** ~~**P**~~ **ROMPT** = 

Now, I have a task and its corresponding conditions and questions listed below. I need your help to check whether the conditions and questions are correct and exactly fit the original task. 

_{{_ tasks _}} {{_ conditions _}} {{_ questions _}}_ 

You should give me back the conditions and questions refined according to the original task; if you think there is nothing that needs to be changed, just output the original conditions and questions. 

You do NOT need to provide solutions. 

All the specified numerical data should be included in the conditions part instead of the question part. 

Also, the questions should NOT contain something that is not given in the condition. Your answer should strictly follow the format below. 

*** 

I think the given conditions and questions fit the original task well / not well. Because ... REFINED ~~C~~ ONDITIONS: 

22 

REFINED ~~Q~~ UESTIONS: *** 

Figure 11: A prompt for task splitting. This prompt is mainly used to initialize the construction of the memory pool by dividing the answers and solutions given in the sample into multiple subproblems and corresponding solutions. 

(2) As a collection of problems and solutions already exists in the dataset ( _Dd_ ), we can decompose a problem and its solution at the same time. Subtask prompts decompose a complex problem into multiple sub-problems, each of which must be clearly described. The Sub ~~S~~ olution prompt directs ChatGPT to decompose the solution into sub-steps corresponding to each of the sub-problems based on the set of sub-problems obtained in the previous step. Sort prompts sort the obtained subproblems from easy to difficult and provide a rationale for the sorting. See Figure 12 

## Task Difficulty Ranking Prompt for Memory Pool Development 

## **SUBTASK** ~~**P**~~ **ROMPT** ~~**S**~~ **YS** = 

You are an excellent expert who thinks step by step and splits a complex question into several steps. 

## **SUBTASK** ~~**P**~~ **ROMPT** = 

Given a task’s background and its condition: condition Now the question is: question And here is the given solution to this question: solution 

Please think generally and step by step to divide the question into N subtasks according to the given solution. You must give each task a clear description and every subtask must be designed as a necessary step to the final question. You should strictly follow the format below: 

*** 

I think the best way to solve the question and my solution is: ... So I set up subtasks as described below. 

TASKS= _>_ 

_<_ st _>_ SUBTASK 1: ... _<_ ed _>_ 

_<_ st _>_ SUBTASK 2: ... _<_ ed _>_ 

... _<_ st _>_ SUBTASK N: ... _<_ ed _>_ 

*** 

## **SUB** ~~**S**~~ **OL** ~~**P**~~ **ROMPT** ~~**S**~~ **YS** = 

You are an excellent expert who can think step by step and split a given complex solution of a certain question into several sub-steps according to the given subtasks split from the original question. 

## **SUB** ~~**S**~~ **OL** ~~**P**~~ **ROMPT** = 

Given a task’s background and its condition: _{{_ condition _}}_ Now the question is: _{{_ question _}}_ And I have already split the question into the subtasks listed below: _{{_ subtask _}}_ And the whole solution to the original whole question is: _{{_ solution _}}_ 

Now, Please think generally and step by step to divide the solution into sub-solutions, which must correspond one-to-one with subtasks. You must give each sub-solution a clear description and every sub-solution should be specific and exactly solve the corresponding subtask. 

23 

For example, if there are N subtasks, you must generate N sub-solutions. And sub-solution 1 should exactly solve subtask 1. 

You must generate sub-solutions according to the whole solution given. If the whole solution is “Not provided”, just generate your own suggested solution for each subtask. Else if the whole solution is provided, your every sub-solution should be a part of the whole solution. You should strictly follow the format below: 

*** 

I think the relations between subtasks and the whole solution are: ... 

So, I set up sub-solutions as described below. SOLUTIONS= _> <_ st _>_ SUBSOLUTION 1: ... _<_ ed _> <_ st _>_ SUBSOLUTION 2: ... _<_ ed _> <_ st _>_ SUBSOLUTION N: ... _<_ ed _>_ 

*** 

## **SORT** ~~**P**~~ **ROMPT** ~~**S**~~ **YS** = 

You are ChatGPT, a large language model trained by OpenAI, and also an excellent prompt difficulty evaluator who is capable of sorting a series of tasks according to their difficulty. 

## **SORT** ~~**P**~~ **ROMPT** = 

Now, I have some tasks (each containing conditions, questions, and corresponding solutions) listed below. I need your help to sort these tasks from the easiest one to the most difficult one for an agent to solve, which means you need to put the simpler subtasks in front. _{{_ tasks _}}_ 

Think carefully and tell me the reason why you think one is easier and another is harder. You do not take the temporal relationship into consideration when sorting. You do NOT need to refine the given solutions. 

You do NOT need to follow the original order of the tasks. 

You are FORBIDDEN to change the description of the tasks given. Just change their order. Your answer should be a permutation of the n tasks. And the id of easier tasks should be placed in front of the harder ones. For example, if there are 3 tasks, and task 2 is the simplest, task 3 is the hardest. 

Then your final output should be “RESULT= _> <_ st _>_ 2 1 3 _<_ ed _>_ ” 

Please strictly follow the format below. 

*** 

I think the difficult relationship between the tasks is: ... SO MY ANSWER IS: RESULT= _> <_ st _>_ [a permutation] _<_ ed _>_ *** 

If there is just 1 subtask, just list it and follow this format: *** SO MY ANSWER IS: RESULT= _> <_ st _>_ 1 _<_ ed _>_ *** 

Figure 12: A prompt for ranking task difficulty. The subproblems are sorted in order of difficulty and solved one by one, from easy to hard, by solving the easier subproblems and learning from the experience to solve the harder ones better. This is the idea of using course learning to help agent self-evolution. 

## G.2 **INSTRUCTIONS FOR SOLVING PROBLEMS.** 

(1) Decomposition prompts (Figure 13) are used to decompose a given chemistry problem into 1-3 subtasks, each with specific goals, criticism, and milestones. 

24 

## Task Decomposition Prompt 

## **SYSTEM** ~~**P**~~ **ROMPT** = 

You are a Chemistry expert and an efficient plan-generation agent. 

Now, you are doing an exam; you must decompose a problem into several subtasks that describe what the goals for the problem. 

— Background Information — PLAN AND SUBTASK: 

A plan has a tree manner of subtasks: task 1 contains subtasks task 1.1, task 1.2, task 1.3, ... and task 1.2 contains subtasks 1.2.1, 1.2.2, ... 

A subtask structure has the following JSON component: 

## _{_ 

“subtask name”: string, name of the subtask 

“goal.goal”: string, the main purpose of the subtask, and what will you do to reach this goal? 

“goal.criticism”: string, what potential problems may the current subtask and goal have? “milestones”: list[string], what milestones should be achieved to ensure the subtask is done? And What formulas might the current subtask use? Make it detailed and specific. _}_ 

## SUBTASK HANDLE: 

A task-handling agent will handle all the subtasks as the inorder-traversal. For example: 1. it will handle subtask 1 first. 

2. if solved, handle subtask 2. If failed, split subtask 1 as subtask 1.1 1.2 1.3... Then handle subtask 1.1 1.2 1.3... 3. Handle subtasks recursively until all subtasks are solved. Do not make the task queue too complex, make it efficiently solve the original task. 

## RESOURCES: 

A task-handling agent can write and execute Python code. 

## — Task Description — 

Generate the plan for query with operation SUBTASK ~~S~~ PLIT, and make sure all must-reach goals are included in the plan. 

— Important Notice — 

- Always make feasible and efficient plans that can lead to successful task-solving. Never create new subtasks that are similar or the same as the existing subtasks. 

- For subtasks with similar goals, try to do them together in one subtask with a list of subgoals rather than split them into multiple subtasks. 

- Do not waste time on making irrelevant or unnecessary plans. 

- The task handler is powered by sota LLM, which can directly answer many questions. So make sure your plan can fully utilize its ability and reduce the complexity of the subtasks tree. 

- You can plan multiple subtasks if you want. 

- Minimize the number of subtasks, but make sure all must-reach goals are included in the plan. 

- Don’t generate tasks that are aimed at understanding a concept, such as “understanding the problem”, the LLM who answers the task already knows the underlying concept. Check the generated subtask objectives and milestones for understanding, and regenerate the subtasks if so. 

- After the subtask is generated, check to see if the answer for the task has been given in the task’s known conditions. If the task has already been resolved, delete the subtask. 

## **USER** ~~**P**~~ **ROMPT** = 

This is the first time you are handling the task (query), so you should give an initial plan. 

25 

_{{_ similar ~~t~~ ask ~~a~~ nd ~~p~~ lan _}}_ Now try to use SUBTASK ~~S~~ PLIT to split the following problem, here is the query which you should give an initial plan to solve: 

—Your task— _{{_ query _}}_ 

You will use operation SUBTASK ~~S~~ PLIT to split the query into 1-3 subtasks and then commit. 

Figure 13: A prompt for task decomposition. Splitting the problem reduces the difficulty of solving each sub-problem. Atomized simple tasks make it easier to find trajectories of experience that can be drawn upon in the memory pool. 

(2) The execution instructions (Figure 14) direct ChatGPT to solve the problem based on the current subtask and known conditions, standardize the return format that must contain the formulae used to solve the problem, a step-by-step reasoning process, and finally give a piece of Python code to arrive at the answer to the problem. 

## Prompt for Task Execution 

## **SYSTEM** ~~**P**~~ **ROMPT** = 

A question is divided into many steps, and you will complete one of them. Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should be included in the answer. 

You have been solving a complex task by following a given plan listed below. — Plan Overview — The complex task has already been split into a tree-based plan as follows: _{{_ all ~~p~~ lan _}}_ You have already performed some of the subtasks. 

## **USER** ~~**P**~~ **ROMPT** = 

Now, you continue to complete subtasks. Please combine the results of previous tasks to complete the current goal of processing subtasks. Please complete all milestones and give a concise answer that can be efficiently used by subsequent subtasks. — Status — Current Subtask: _{{_ subtask id _}}_ The query: _{{_ subtask ~~g~~ oal _}}_ Milestones: _{{_ milestones _}}_ Please respond strictly to the format provided. For each instance, you need to do three things. Firstly, for “formulae retrieval”, you need to identify the formulae explicitly and implicitly entailed in the problem context. Then there is a “reasoning/calculation process” where you are required to reason step by step based on the identified formulae and problem context. Finally, conclude the answer by writing a piece of corresponding Python code; you MUST use the International System of Units in this stage. For each problem, the output format should incorporate the following components in the corresponding format: **Formulae retrieval: ** [Formula 1] (the formula required to solve the problem) 

26 

[Formula 2] (the second formula required to solve the problem, if any) 

... 

[Formula n] (the n-th formula required to solve the problem, if any) **Reasoning/calculation process:** [step 1] (the first step for solving this problem) ..... [step n] (the n-th step for solving the problem, if any) 

**Answer conclusion:** 

[answer] (Python code that can be executed independently) 

Your answer should be a piece of Python code that solves the current question. You must end your code by printing all the result and their units. Make sure the code can be successfully run without any input. Be precise. The answer should be accurate, choose the appropriate units, and prohibit the use of round functions to round off and lose the results. Make sure the code can be successfully run without any input. And import all the modules that you need to use. 

for example, you could respond to the Answer conclusion part like this: **Answer conclusion:** [answer]: 

import numpy as np # Value of 2 pi c omega_obs omega_obs = 1.8133708490380042e+23 # Hz # Value of D for H35Cl D = 440.2 # kJ/mol # Calculate beta beta = omega_obs * (2 * D)**0.5 

# Print the result print("The value of beta is:", beta, "cmˆ(-1)") 

— Similar tasks — 

The following are the trajectories of tasks that have been dealt with in the past that are similar to this goal, including the successful tasks and their action lists. You can learn from them and use relevant knowledge in their procedure. _{{_ success ~~p~~ rompt _}}_ 

Figure 14: A prompt for executing tasks. The format of the specified output consists of formulas and reasoning processes. The relevant knowledge in Knowledge Memory improves the accuracy of the formula part, and the reasoning processes make the output Python code more logical. 

(3) If the similarity between the retrieved memory trace and the current problem is too low, the Imagination prompts (Figure 15) direct ChatGPT to generate a problem and a solution in the specified format based on the current topic. 

Prompt for Similar Task Association 

## **IMG** ~~**P**~~ **ROMPT** = 

Please create _{{_ top ~~k~~ _}}_ advanced chemistry questions suitable for in-depth understanding and application of chemical formulas and principles. Each question should focus on the deeper aspects of the [topic] provided to understand the principles of chemistry and the 

27 

reasoning process. [topic]: _{{_ topic _}}_ Guidelines for Problem Creation: 

- Use of Samples: You are provided with sample questions for reference. Feel free to use these to guide the style and depth of the problems. 

- Beyond the Examples: You are encouraged to use your background and expertise to create problems that go beyond the provided examples, ensuring the problems are as diverse and comprehensive as possible. 

Requirements for Each Problem: 

a. Problem Statement: Clearly define the challenge or task. 

b. Solution: Provide a detailed solution that includes: 

I. Formulas and Knowledge Needed: List the equations and concepts required to understand and solve the problem. 

II. Reasoning Steps: Outline the logical or mathematical steps to solve the problem. III. Python code: Executable Python code is generated to solve problems. At the end of each problem, please include Python code that can be used to confirm and verify the correctness of the provided solution. The Python solutions should illustrate the entire solution process, from the initial step to the final answer, rather than merely validating the result. Develop these solutions such that each step of the mathematical process is explicitly demonstrated and calculated in Python. Additionally, ensure that you run your Python code to confirm it is free from any errors. 

d. Diversity: Ensure a wide range of problems, each focusing on different elements from the subtopic list. 

e. Presentation: Please output your problem statement, solution, detailed explanation, and a self-contained Python code for verification below in the specified format. 

For each generated question, the output is required to be in the following format: [Task Start] 

[Problem Statement]: (your problem) 

**Formulae retrieval: ** 

[Formula 1] (the formula required to solve the problem) 

[Formula 2] (the second formula required to solve the problem, if any) ... 

[Formula n] (the n-th formula required to solve the problem, if any) 

**Reasoning/calculation process:** 

[step 1] (the first step for solving this problem) 

..... [step n] (the n-th step for solving the problem, if any) 

**Answer conclusion:** [answer]: (Python code that can be executed independently) [Task End] 

You have to generate _{{_ topk _}}_ tasks about _{{_ topic _}}_ . Sample demonstration example: _{{_ example ~~s~~ hots _}}_ 

Figure 15: A prompt for associating similar tasks. The dataset covered 15 sub-domains, and the question was modeled by interrogation to determine which sub-domain the question belonged to. The associatively generated trajectories must be in the same format as the trajectories in Execution Memory. 

28 

(4) The evaluation prompt (Figure 16) is used to assess the correctness of the current solution, including judging the correctness of the given formulas, the rigor of the reasoning process, and whether the output of the Python code meets the task goals and completes all task milestones. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0029-01.png)


**----- Start of picture text -----**<br>
Prompt for Task Evaluation<br>SCORE P ROMPT  =<br>You tackled a sub-problem in a chemistry problem; the format of the solution to the<br>problem is **Formulae retrieval: ** and **Reasoning/calculation process:** and **Answer<br>conclusion:**(which includes a piece of Python code and its corresponding output).<br>—Subtask—<br>The question: { subtask g oal }<br>Milestones: { milestones }<br>[Response Start]<br>{ response }<br>[Response End]<br>For each instance, you need to do four things:<br>- First, judge whether the given formula is correct and whether the constants are correct.<br>- Second, judge whether the reasoning process is rigorous and correct.<br>-Third, determine whether the Python function outputs the parameters required by the task<br>goal and milestone.<br>- Finally, score the degree of completion and correctness of the whole task. You should give<br>the “confidence score” on the scale of [0,1]. Please be very strict about your ratings.<br>The output format should incorporate these components in the following format:<br>**Judgement of the retrieved formulae:**<br>[judgement] (Your assessment of whether the retrieved formulae are correct or not.)<br>**Judgement of the reasoning process:**<br>[judgement] (Your assessment of whether the reasoning process is correct or not.)<br>**Judgement of the Answer conclusion:**<br>[judgement] (Your assessment of whether the Python code outputs the parameters required<br>in the task objective and whether the Python code correctly infers according to the analysis<br>in reason.)<br>**Confidence score:**<br>[score] (float number in [0,1], A very strict score is given to the correctness of the solution<br>of the entire task)<br>**----- End of picture text -----**<br>


Figure 16: A prompt for evaluating tasks. Evaluate the current answer, generate answers consecutively, and select the answer with the highest score. 

(5) Finally, the final answer and summary of posterior knowledge are obtained by summary prompts (Figure 17). Summarize the total process of solving the problem, the relevant knowledge used, and the formulas that can be used to enrich the memory pool. At the same time, the answers to all sub-problems are combined to produce the final answer to the whole problem. 

29 

Prompt for Task Summary 

## **SYSTEM** ~~**P**~~ **ROMPT** = 

You are a posterior ~~k~~ nowledge ~~o~~ btainer. 

Now that you’ve completed a parent task, the task is made up of many subtasks, each of which has been completed. 

Your plan for the parent task is as follows: 

—Parent Goal — 

_{{_ parent ~~g~~ oal _}}_ 

— Sub-task division — 

_{{_ all ~~p~~ lan _}}_ 

The flow of your actions for handling subtasks is: — Action lists for Subtasks — 

_{{_ sub ~~p~~ lan _}}_ 

Now, you have to learn some posterior knowledge from this process by doing the following things: 

1. Summary: Summarize the ideas of all subtasks to the parent task, and summarize a total process to the parent task according to the action process of each subtask. Explicitly include all the formulas used during performing the subtasks in this summary. Specific numbers and numerical results are NOT needed in this part. 

2. Reflection of knowledge: After performing this task, you get some knowledge of generating plans for similar tasks. Only conclude the used knowledge and formulas used in this whole task, do NOT contain numerical calculation process and results. 

3. Final Answer: Give the final answer to the task according to the course of the task, and ask the answer to be very short, without explaining the reason and adding unnecessary punctuation. If it’s a math problem, only the last value is given. 

Figure 17: A prompt for summarizing tasks. Summarizing tasks is crucial for the self-evolution of ChemAgent. Summarize the formulas and principles of this task and add them to the Memory Library. 

(6) When a task fails to execute, it may be because the task decomposition is unreasonable. The framework can perform adjustment operations such as adding/splitting/deleting the current task tree to adjust the overall task framework. See Figure 18 for the exact prompt. 

Prompt for Task Refinement 

## **SYSTEM** ~~**P**~~ **ROMPT** = 

You are a plan-rectify agent, your task is to iteratively rectify a plan of a query. — Background Information — PLAN AND SUBTASK: A plan has a tree manner of subtasks: task 1 contains subtasks task 1.1, task 1.2, task 1.3, ... and task 1.2 contains subtasks 1.2.1, 1.2.2, ... A subtask structure has the following JSON component: _{_ “subtask name”: string, name of the subtask 

- “goal.goal”: string, the main purpose of the subtask, and what will you do to reach this goal? 

- “goal.criticism”: string, what potential problems may the current subtask and goal have? 

30 

“milestones”: list[string], what milestones should be achieved to ensure the subtask is done? And What formulas might the current subtask use? Make it detailed and specific. _}_ SUBTASK HANDLE: A task-handling agent will handle all the subtasks as the inorder-traversal. For example: 1. it will handle subtask 1 first. 2. if solved, handle subtask 2. If failed, split subtask 1 as subtask 1.1 1.2 1.3... Then handle subtask 1.1 1.2 1.3... 3. Handle subtasks recursively until all subtasks are solved. Do not make the task queue too complex, make it efficiently solve the original task. RESOURCES: A task-handling agent can write and execute Python code. — Task Description — Your task is iteratively to rectify a given plan based on the goals, suggestions, and now handling positions. In this mode, you will use the given operations to rectify the plan. At each time, use one operation. SUBTASK OPERATION: 1. split: Split an already handled but failed subtask into subtasks because it is still so hard. The “target ~~s~~ ubtask ~~i~~ d” for this operation must be a leaf task node that has no children subtasks, and should provide new split “subtasks” of length 2-4. You must ensure the “target ~~s~~ ubtask ~~i~~ d” exists, and the depth of new split subtasks _< {{_ max ~~p~~ lan ~~t~~ ree ~~d~~ epth _}}_ . - split 1.2 with 2 subtasks will result in create new 1.2.1, 1.2.2 subtasks. 2. add: Add new subtasks as brother nodes of the ‘target ~~s~~ ubtask ~~i~~ d‘. This operation will expand the width of the plan tree. The ‘target ~~s~~ ubtask ~~i~~ d‘ should point to a now-handling subtask or future subtask. - add 1.1 with two subtasks will result in creating new 1.2, 1.3 subtasks. - add 1.2.1 with 3 subtasks wil result in create new 1.2.2, 1.2.3, 1.2.4 subtasks. 3. delete: Delete a subtask. The ‘target ~~s~~ ubtask ~~i~~ d‘ should point to a future/TODO subtask. Don’t delete the now handling or done subtask. - delete 1.2.1 will result in delete 1.2.1 subtask. 

— Note — The user is busy, so make efficient plans that can lead to successful task-solving. Do not waste time making irrelevant or unnecessary plans. Don’t use search engines since you know about planning. Don’t divide trivial tasks into multiple steps. If the task is unsolvable, give up and submit the tas k. *** Important Notice *** - Never change the subtasks before the handling positions, you can compare them in lexicographical order. 

- Never create (with add or split action) new subtasks that are similar or the same as the existing subtasks. - For subtasks with similar goals, try to do them together in one subtask with a list of subgoals rather than split them into multiple subtasks. - Every time you use an operation, make sure the hierarchy structure of the subtasks remains, e.g., if subtask 1.2 is to “find A,B,C” , then the newly added plan directly related to this plan (like “find A”, “find B”, “find C”) should always be added as 1.2.1, 1.2.2, 1.2.3... - You are restricted to give operations in at most 4 times, so the plan refinement is not so much. - The task handler is powered by sota LLM, which can directly answer many questions. So 

31 

make sure your plan can fully utilize its ability and reduce the complexity of the subtasks tree. 

## **USER** ~~**P**~~ **ROMPT** = 

Your task is to choose one of the operators of SUBTASK OPERATION, note that 1.You can only modify the subtask with subtask ~~i~~ d _> {{_ subtask ~~i~~ d _}}_ (not included). 

2. Please use a function call to respond to me (remember this!!!). 

Figure 18: A prompt for refining tasks. Incorrect answers may result from an error in a sub-question, leading to subsequent inaccuracies, or from a misjudgment in the decomposition of the question. In the former case, the Task Evaluation module is used to enhance the accuracy of problem-solving, while the Task Refinement module is employed to modify the question decomposition tree in the latter case. 

## G.3 **INSTRUCTIONS FOR BASELINE** 

Both direct-reasoning and fewshot+Python setups are used as baselines in the main experiment, mainly referring to the experimental setup in SciBench Wang et al. (2024a). The following instruction (Figure 19) is used in the fewshot+Python setup, which asks for a piece of Python code to be output and is followed by three demonstrations. 

## Prompt for Few-Shot Python 

## **SYS** ~~**P**~~ **ROMPT** = 

Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement. For clarity, please translate the solution steps into Python code and encase the Python code within triple backticks. 

Figure 19: A prompt for Few-shot+Python. The baseline setup directly requires LLMs to generate Python code, and through few-shot learning, the responses are standardized. This approach, compared to the direct-reasoning setup, has shown an improvement in correctness. 

32 

## H SPECIFIC CASE TRAJECTORIES 

To concretely demonstrate the problem-solving process of our framework, we provide a complete execution trajectory of a successfully answered question to illustrate the procedural workflow of our framework. 

## **Task Description** 

At 300 K and 20 atm, the compression factor of a gas is 0.86. Calculate the volume occupied by 8 _._ 2 mmol of the gas under these conditions. The unit of the answer should be cm[3] . **Task Decomposition Task 1: Calculate the volume in liters** Use the definition of the compression factor and the ideal gas law to calculate the volume occupied by 8 _._ 2 mmol of the gas under the given conditions in liters. **Task 2: Convert the volume to** cm[3] Convert the calculated volume from liters to cubic centimeters. **Execution of Task 1 - Calculate the Volume in Liters Formulas:** - Ideal Gas Law: _PV_ = _nRT_ - Compression Factor: _Z_ = _[PV] nRT_ **Given:** - Compression factor, _Z_ = 0 _._ 86 - Pressure, _P_ = 20 atm (convert to Pa) - Number of moles, _n_ = 8 _._ 2 mmol = 8 _._ 2 _×_ 10 _[−]_[3] mol - Gas constant, _R_ = 8 _._ 3145 J K _[−]_[1] mol _[−]_[1] - Temperature, _T_ = 300 K **Calculations:** 1. Convert pressure from atm to Pa: _P_ = 20 atm _×_ 101325 Pa _/_ atm = 2 _,_ 026 _,_ 500 Pa 2. Rearrange the compression factor formula to solve for _V_ : _Z_ = _[PV]_ = _⇒ V_ = _[ZnRT] nRT P_ 3. Calculate the volume in cubic meters: _V_ =[0] _[.]_[86] _[ ×]_[ 8] _[.]_[2] _[ ×]_[ 10] _[−]_[3][ mol] _[ ×]_[ 8] _[.]_[3145 J K] _[−]_[1][ mol] _[−]_[1] _[ ×]_[ 300 K] 2 _,_ 026 _,_ 500 Pa 4. Compute _V_ : _V_ =[0] _[.]_[86] _[ ×]_[ 8] _[.]_[2] _[ ×]_[ 10] _[−]_[3] _[ ×]_[ 8] _[.]_[3145] _[ ×]_[ 300] 2 _,_ 026 _,_ 500 _V ≈_ 8 _._ 6801 _×_ 10 _[−]_[6] m[3] 5. Convert volume to liters (since 1 m[3] = 1000 L): _V_ = 8 _._ 6801 _×_ 10 _[−]_[6] m[3] _×_ 1000 L _/_ m[3] = 0 _._ 0086801 L **Execution of Task 2 - Convert the Volume to** cm[3] **Formula:** - Conversion from liters to cubic centimeters: 1 L = 1000 cm[3] **Calculation:** 1. Convert volume from liters to cubic centimeters: _V_ = 0 _._ 0086801 L _×_ 1000 cm[3] _/_ L = 8 _._ 6801 cm[3] **Summary Results** The volume occupied by 8 _._ 2 mmol of the gas under the given conditions is approximately **8.6801** cm[3] . 

We also provide an execution trajectory of a successfully answered question in another benchmark, ChemBench (Mirza et al., 2024) which direct-reasoning method fails to solve. The backbone LLM used here is gpt-4o (2024-08-01-preview). 

## **Task Description** 

33 

Statement 1: Addition of a non-reacting gas at constant volume and constant temperature to a container containing gas (X) does not change the partial pressure of gas (X). Statement 2: The addition of a non-reacting gas to a container at constant volume and constant temperature results in an increase in the total pressure of the container, but not in the partial pressure of the gases in the container. Which of the following statements is the most correct option for the given two statements? A.Statement 1 is False, Statement 2 is True. B.Statement 1 is True, Statement 2 is False. C.Statements 1 and 2 are True. Statement 2 is NOT the correct explanation for Statement 1. D.Statements 1 and 2 are True. Statement 2 is the correct explanation for Statement 1. Choose one correct choice. 

**Task Decomposition** 

**Task 1: Verify the validity of Statement 1** Determine effect of adding a non-reacting gas on partial pressure of gas (X) at constant volume and temperature. **Task 2: Verify the validity of Statement 2** Determine effect of adding a non-reacting gas on total pressure at constant volume and temperature. **Task 3: Compare Statements and Choose the Correct Option** Compare the findings from the analysis of Statements 1 and 2 and determine the most correct option from the provided choices. **Execution of Task 1 - Verify the validity of Statement 1 Formulas:** 

- Ideal Gas Law: _PV_ = _nRT_ 

- _P_ = _[n]_[g][as] _[RT]_ partial _V_ **Given:** 

- _P_ partial is the partial pressure of gas (X) 

- _n_ gas is the number of moles of gas (X) 

- _T_ is the temperature, _V_ is the volume **Calculations:** 

1. The partial pressure of a gas in a container is given by the ideal gas law, which states that the partial pressure is directly proportional to the number of moles of the gas, the temperature, and inversely proportional to the volume. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0034-09.png)


2. When a non-reacting gas is added to the container at constant volume and constant temperature, the number of moles of gas (X) does not change. Therefore, the partial pressure of gas (X) remains the same because _n_ gas, _R_ , _T_ , and _V_ are constant. The initial and final partial pressures of gas (X) are the same, verifying that Statement 1 is true. **Execution of Task 2 - Verify the validity of Statement 2 Formula:** 

- The total pressure of a gas mixture is the sum of the partial pressures of the individual gases: 

_P_ total = _P_ 1 + _P_ 2 + _. . ._ + _Pn_ 

## **Calculation:** 

1. Consider a container with gas (X) at constant volume and temperature. The initial total pressure is the partial pressure of gas (X), _PX_ . 2. When a non-reacting gas (Y) is added to the container at constant volume and temperature, the partial pressure of gas (X) remains the same, _PX_ . 

3. The total pressure of the container after adding gas (Y) is the sum of the partial pressures of gas (X) and gas (Y), _P_ total = _PX_ + _PY_ . **Execution of Task 3 - Compare Statements and Choose the Correct Option Formula:** No specific formula is needed for this subtask. 

34 

## **Reasoning:** 

1. Analyze the results from the previous subtasks: - From Task 1.1: The partial pressure of gas (X) remains unchanged after adding a non-reacting gas at constant volume and temperature. - From Task 1.2: The total pressure increases after adding a non-reacting gas at constant volume and temperature, but the partial pressure of gas (X) remains unchanged. 

2. Compare the findings from the analysis of Statements 1 and 2. 

3. Determine the most correct option based on the analysis: - Both statements are true, and Statement 2 correctly explains Statement 1. **Summary Results** D. Statements 1 and 2 are True. Statement 2 is the correct explanation for Statement 1. 

An example result of the decomposition step of the library construction phase, which help understand how the sub-tasks are organized. 

## **Original Task** 

The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ mol _[−]_[1] . Calculate the difference between the molar enthalpy and internal energy changes when the pressure is 1.0 bar given that the densities of the polymorphs are 2 _._ 71 g cm _[−]_[3] and 2 _._ 93 g cm _[−]_[3] , respectively. **Original Solution** The change in enthalpy when the transition occurs is ∆ _H_ m = _H_ m( aragonite ) _− H_ m( calcite ) = _{U_ m(a) + _pV_ m(a) _} −{U_ m(c) + _pV_ m(c) _}_ = ∆ _U_ m + _p {V_ m(a) _− V_ m(c) _}_ where a denotes aragonite and c calcite. It follows by substituting _V_ m = _M/ρ_ that 1 1 ∆ _H_ m _−_ ∆ _U_ m = _pM_ � _ρ_ (a) _[−] ρ_ (c) � Substitution of the data, using _M_ = 100 g mol _[−]_[1] , gives 1 1 ∆ _H_ m _−_ ∆ _U_ m = �1 _._ 0 _×_ 10[5] Pa� _×_ �100 g mol _[−]_[1][�] _×_ � 2 _._ 93 g cm _[−]_[3] _[−]_ 2 _._ 71 g cm _[−]_[3] � = _−_ 2 _._ 8 _×_ 10[5] Pa cm[3] mol _[−]_[1] = _−_ 0 _._ 28 Pa m[3] mol _[−]_[1] 

**Decomposition Results SUB-TASK 1** _Condition 1_ The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 1_ Identify the given data and the required physical quantities (molar internal energy change, pressure, densities, etc.). _Solution 1 ρ_ calcite = 2 _._ 71 g _/_ cm[3] _, ρ_ aragonite = 2 _._ 93 g _/_ cm[3] **SUB-TASK 2** _Condition 2_ The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 2_ 

35 

Express the molar volumes of calcite and aragonite in terms of their densities and the molar mass of CaCO3. _Solution 2_ 

_M M Vm_ (calcite) = _, Vm_ (aragonite) = _ρ_ calcite _ρ_ aragonite 

where _M_ is the molar mass of CaCO3 (100 g/mol). 

**SUB-TASK 3** _Condition 3_ 

The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 3_ Substitute the given densities and molar mass into the molar volume expressions to find the specific volumes. _Solution 3_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0036-05.png)


## **SUB-TASK 4** 

_Condition 4_ 

The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 4_ Calculate the change in volume between aragonite and calcite using the specific volumes. _Solution 4_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning_images/iclr_2025_ChemAgent_Self-updating_Memories_in_Large_Language_Models_Improves_Chemical_Reasoning.pdf-0036-09.png)


## **SUB-TASK 5** 

_Condition 5_ 

The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 5_ Write down the equation relating the change in molar enthalpy to the change in molar internal energy and the pressure-volume work term. _Solution 5_ 

∆ _Hm_ = ∆ _Um_ + _p_ ∆ _Vm_ where ∆ _Vm_ = _Vm_ (aragonite) _− Vm_ (calcite). 

**SUB-TASK 6** _Condition 6_ 

The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 6_ 

Substitute the values of pressure, molar internal energy change, and the change in volume into the enthalpy change equation to find the difference between the molar enthalpy and internal energy changes. _Solution 6_ 

36 

∆ _Hm −_ ∆ _Um_ = _p_ ∆ _Vm_ ∆ _Hm −_ ∆ _Um_ = (1 _._ 0 _×_ 10[5] Pa) _×_ ( _−_ 2 _._ 77 _×_ 10 _[−]_[6] m[3] _/_ mol) ∆ _Hm −_ ∆ _Um_ = _−_ 0 _._ 277 kJ _/_ mol 

**SUB-TASK 7** _Condition 7_ 

The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 7_ 

Verify the units of the calculated result to ensure consistency with the required units (e.g., kJ _/_ mol). _Solution 7_ The units of the result _−_ 0 _._ 277 kJ _/_ mol are correct and consistent with the required units. **SUB-TASK 8** _Condition 8_ The change in molar internal energy when CaCO3( s) as calcite converts to another form, aragonite, is +0 _._ 21 kJ _/_ mol. The pressure is 1.0 bar. The densities of the polymorphs are 2 _._ 71 g _/_ cm[3] and 2 _._ 93 g _/_ cm[3] , respectively. _Question 8_ Summarize the final answer with the correct sign and units. _Solution 8_ The difference between the molar enthalpy change and the molar internal energy change is ∆ _Hm −_ ∆ _Um_ = _−_ 0 _._ 28 kJ _/_ mol _._ 

37 

