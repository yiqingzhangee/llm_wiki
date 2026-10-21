**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

**Chengshu Li**[1] **Jacky Liang**[2] **Andy Zeng**[2] **Xinyun Chen**[2] **Karol Hausman**[1 2] **Dorsa Sadigh**[1 2] **Sergey Levine**[2 3] **Li Fei-Fei**[1] **Fei Xia**[* 2] **Brian Ichter**[* 2] 

https://chain-of-code.github.io/ 

## **Abstract** 

Code provides a general syntactic structure to build complex programs and perform precise computations when paired with a code interpreter - we hypothesize that language models (LMs) can leverage code-writing to improve Chain of Thought reasoning not only for logic and arithmetic tasks (Chen et al., 2022; Nye et al., 2021; Austin et al., 2021), but also for _semantic_ ones (and in particular, those that are a mix of both). For example, consider prompting an LM to write code that counts the number of times it detects sarcasm in an essay: the LM may struggle to write an implementation for “detect_sarcasm(string)” that can be executed by the interpreter (handling the edge cases would be insurmountable). However, LMs may still produce a valid solution if they not only write code, but also selectively “emulate” the interpreter by generating the expected output of “detect_sarcasm(string)”. In this work, we propose Chain of Code (CoC), a simple yet surprisingly effective extension that improves LM codedriven reasoning. The key idea is to encourage LMs to format semantic sub-tasks in a program as flexible pseudocode that the interpreter can explicitly catch undefined behaviors and hand off to simulate with an LM (as an “LMulator"). Experiments demonstrate that Chain of Code outperforms Chain of Thought and other baselines across a variety of benchmarks; on BIG-Bench Hard, Chain of Code achieves 84%, a gain of 12% over Chain of Thought. In a nutshell, CoC broadens the scope of reasoning questions that LMs can answer by “thinking in code". 

> *Equal contribution 1Department of Computer Science, Stanford University, California, USA[2] Google DeepMind, California, USA 3Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, California, USA. Correspondence to: Chengshu Li <chengshu@stanford.edu>. 

> _Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

Direct answer only Q: How many countries have I been to? I’ve been to Mumbai, London, Washington, Grand Canyon, ... **A: 32 (20%,** ✗ **), 29 (10%,** ✗ **), 52 (10%,** ✓ **), ...** 

## Chain of Thought 

Q: Let’s think step by step. How many countries have I been to? I’ve been to Mumbai, London, ... We’ll group by countries and count: 1. India: Mumbai, Delhi, Agra 2. UK: London, Dover, Edinburgh, Skye 3. USA: Washington, Grand Canyon, ... **A: 61 (20%,** ✗ **), 60 (20%,** ✗ **), 52 (10%,** ✓ **), ...** 

## Chain of Code (Ours) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0001-12.png)


**----- Start of picture text -----**<br>
Q: How many countries have I been to? I’ve been to Mum-<br>bai, London, Washington, Grand Canyon, Baltimore, ...<br>1 places, countries = ["Mumbai", ...], set()<br>delta state: {places = [‘Mumbai’, ...], countries =<br>set()}<br>2 for place in places:<br>delta state: {place = ‘Mumbai’}<br>3 country = get_country(place)<br>delta state: {country = ‘India’)}<br>4 countries.add(country)<br>delta state: {countries = {‘India’}}<br>5 answer = len(countries) delta state: {answer =<br>52}<br>A: 52 (100%, ✓ )<br>**----- End of picture text -----**<br>


_Figure 1._ Chain of Code generates code and reasons through an LMaugmented code emulator. Lines evaluated with Python are in red and with an LM are in purple. The full query is in Fig. A5. 

## **1. Introduction** 

Language models (LMs) at certain scale exhibit the profound ability to solve complex reasoning questions (Brown et al., 2020; Wei et al., 2022a) – from writing math programs (Drori et al., 2022) to solving science problems (Lewkowycz et al., 2022). Notably, these capabilities have shown to improve with Chain of Thought (CoT) prompting (Wei et al., 2022b), whereby complex problems are decomposed into a sequence of intermediate reasoning steps. CoT excels at semantic reasoning tasks, but tends to struggle with questions that involve numeric or symbolic reasoning (Suzgun et al., 2022; Mirchandani et al., 2023). Subsequent work addresses this by prompting LMs (e.g., trained on Github (Chen et al., 2021)) to write and 

1 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0002-01.png)


**----- Start of picture text -----**<br>
(a) Direct answer only (b) Chain of Thought (c) Chain of Code (Ours)<br>**----- End of picture text -----**<br>


_Figure 2._ Overall results on BIG-Bench Hard compared to human performance (Srivastava et al., 2022). 

execute code (Chen et al., 2022; Nye et al., 2021; Austin et al., 2021). Code in particular is advantageous because it provides both (i) a general syntactic structure to build and encode complex programs (Liang et al., 2023) (e.g., logic structures, functional vocabularies – in ways that are Turing complete), and (ii) an interface by which existing APIs paired together with an interpreter can be used to perform precise algorithmic computations (e.g., from multiplication of large numbers to sorting an array of size 10,000) that a language model trained only to mimic the statistically most likely next token would otherwise struggle to produce. 

While writing and executing code may improve LM reasoning performance across a wide range of arithmetic tasks, this particular approach contends with the fact that many semantic tasks are rather difficult (and at times, nearly impossible) to express in code. For example, it remains unclear how to write a function that returns a boolean when it detects sarcasm in a string (Suzgun et al., 2022) (handling the edge cases would be insurmountable). Perhaps fundamentally, using LMs to write programs in lieu of multi-step textual reasoning inherently assumes that the intermediate reasoning traces (expressed in lines of code) all need to be _executable_ by an interpreter. Is it possible to lift these restrictions to get the best of both reasoning in code and reasoning in language? 

In this work, we propose Chain of Code (CoC), a simple yet surprisingly effective extension to improve LM code-driven reasoning – where the LM not only writes a program, but also selectively “simulates” the interpreter by generating the expected output of certain lines of code (that the interpreter could not execute). The key idea is to encourage LMs to format semantic sub-tasks in a program as flexible pseudocode that at runtime can be explicitly caught and handed off to emulate with an LM – we term this an LMulator (a portmanteau of LM and emulator). For example, given the task “ _in the above paragraph, count how many times the person was sarcastic_ ,” we can incontext prompt the LM to write a program that may call helper functions such as is_sarcastic(sentence), to which the LM makes a linguistic prediction and returns the result as a boolean output, that then gets processed with the rest of the program. Specifically, we formulate LM reasoning as the following process (illustrated in Figure 1): the LM writes code, the interpreter steps through to execute each line of code (in red), or if it 

fails, simulates the result with the LM (in purple) and updates the program state (in green). CoC inherits the benefits of both (i) writing executable code (where precise algorithmic compututations are left to an interpreter), and (ii) writing pseudocode for semantic problems, and generating their outputs (which can be thought of as a simple formatting change, to which LMs are robust (Min et al., 2022)) – enabling the LM to “think in code”. 

Extensive experiments demonstrate that CoC is applicable to a wide variety of challenging numerical and semantic reasoning questions, and outperforms a number of popular baselines. In particular, we find that it achieves high performance on BIG-Bench Hard tasks (Suzgun et al., 2022), outperforming average human raters overall and outperforming even the best human raters on an algorithmic subset of tasks, and to the best of our knowledge setting a new state of the art. We further show that _both_ code interpreter execution and language model execution simulation are necessary for this performance, and that the approach scales well with large and small models alike – contrary to prompting techniques like Chain of Thought that only emerge at scale. We then demonstrate how Chain of Code can serve as a general purpose reasoner via _cross-task prompting_ benchmark, which in contrast to prior work, uses prompts from different families of problems as context – providing only the structure of the response (as opposed to the solution itself). Finally, we show CoC is complementary to more advanced instruction tuned chat models, robust against prompt variation, and applicable beyond language reasoning domain like robotics. This work underscores how one may leverage the structure and computational power of code and the reasoning abilities of language models to enable a “best of both worlds” reasoner. 

## **2. Chain of Code: Reasoning with an LMulator** 

In this section, we describe Chain of Code (CoC), an approach that leverages the ability of language models to code, to reason, and to leverage an LM-augmented code emulator (an LMulator) to simulate running code. We start with background in Section 2.1, then overview the method in Section 2.2, its implementation in Section 2.3, and finally its capabilities in Section 2.4. 

2 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

## **2.1. Preliminaries** 

Briefly, we overview some background on LM reasoning. Many of these reasoning techniques have been enabled by incontext learning (Brown et al., 2020), which provides the model with a few demonstrative examples at inference time, rather than updating any weights with gradients. These examples serve to provide context and format for the setting, enabling the model to emulate these examples while adapting to a new query. This property has been instrumental in easily applying LMs to new tasks as it can be rapidly adapted and requires minimal data. 

Through in-context learning, approaches have been developed to leverage human thought processes and use tools to improve performance of language models. We outline three such approaches that provide the foundations for Chain of Code. Chain of Thought (CoT) (Wei et al., 2022b), ScratchPad (Nye et al., 2021), and Program of Thoughts (Chen et al., 2022) demonstrated the efficacy of breaking problems down into substeps. For CoT these substeps are in natural language, mirroring one’s thought process when stepping through a complicated problem. ScratchPad, on the other hand, maintains a program state of intermediate steps when simulating the output of code – resulting in an LM acting as a code interpreter. Program of Thoughts (Chen et al., 2022) focused on generating the code itself, which is then executed by a code interpreter to solve reasoning problems. Each of these is visualized in Figure 3. 

## **2.2. Chain of Code** 

Inspired by how a human may reason through a particularly complex problem with a mix of natural language, pseudocode, and runnable code or how a researcher may develop a new general algorithm through a code-based formalism then apply it to a problem, Chain of Code proceeds in two steps: (1) Generation, which, given the question to solve, an LM generates code to reason through the problem, and (2) Execution, which executes the code via a code interpreter when possible and via an LM when not. See Section 2.3 for more details on the specific implementation. 

**Chain of Code Generation** Given a problem to solve, CoC generates reasoning substeps in the structure of code. This code provides the framework of reasoning through the problem, and may be in the form of explicit code, pseudocode, or natural language. Figure 3d walks through a potential generation to solve an object counting problem from BIG-Bench. 

**Chain of Code Execution** A core contribution of CoC is not just the generation of reasoning code, but the manner in which it is executed. Once the code is written, the code is attempted to be run by a code interpreter – in this work we consider Python, but the approach is general to any interpreter. If the code is successfully executed, the program state is updated and the execution continues. If the code is not executable or raises any exception, the language model instead is used to simulate the 

execution. The program state is subsequently updated by the language model’s outputs and the execution continues. Herein, we refer to this as an LMulator, a portmanteau of LM and code emulator. This relatively simple change enables a variety of new applications for code which mix semantics and numerics. Figure 3e shows how the generated code is run, maintaining the program state and switching between the Python executor and the LMulator. 

## **2.3. Chain of Code Implementation** 

While the generation implementation is straightforward prompting and language model generation, the execution implementation is slightly more complex. Our implementation is based on using Python’s try and except and maintaining a program state. Line by line CoC steps through the code. If the line is executable by a code interpreter, it is executed, the program state is updated, and the program continues. If it is not executable by a code interpreter, a language model is given the context of the program (the question, the prior lines, and the history of the program state) and generates the next program state. This emulation can also leverage chain of thought to determine how to respond. That generated program state is then updated for the code interpreter as well. This sharing of program state interweaves the code interpreter and the language model simulator in a manner applicable to arbitrary interweaving, even control flow like for-loops and if-statements. This continues until the entire code is run, and the answer is retrieved as the value of the variable named answer, or in case of irrecoverable errors, with the language model outputting A: answer. 

To illustrate with a brief example, the code answer = 0; answer += is_sarcastic(‘you don’t say’); answer += 1; would be executed as follows: (1) Python would execute the first line answer = 0; and update the program state to {answer = 0}, (2) Python would attempt to execute the second line and fail, and thus the LMulator would simulate the code answer += is_sarcastic(‘you don’t say’); by generating the program state {answer = 1}, which would be updated in the program, (3) Python would execute the last line answer += 1; and update the program state to {answer = 2}, (4) the answer would be retrieved as 2. 

## **2.4. Chain of Code Abilities** 

Chain of Code has several attractive properties: 

1. It enables code use in entirely new regimes, by combining the advantages of code with the powerful semantic and commonsense knowledge of language models, which can easily express rules that are challenging to express in code (e.g., which foods are fruits?). Such an ability may have benefits beyond reasoning problems and its flexibility enables executing expressive language, such as pseudocode. 

3 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0004-01.png)


**----- Start of picture text -----**<br>
(a) Chain of Thought (b) Program of Thoughts (c) ScratchPad<br>Q: Roger has 5 balls. He buys Q: Roger has 5 balls. He buys Q: Roger has 5 balls. He buys 2 more packs,<br>2 more packs, each with 3. How 2 more packs, each with 3. How each with 3. How many balls does he have now?<br>many balls does he have now? many balls does he have now?<br>Roger starts with 5 balls. num_balls = 5 num_balls = 5 state: {num_balls = 5}<br>2 packs of 3 balls is 6. num_balls += 2 * 3 num_balls += 2 * 3 state: {num_balls = 11}<br>5 + 6 = 11. answer = num_balls answer = num_balls state: {answer = 11}<br>A: 11 A: 11 A: 11<br>(d) Chain of Code Generation (Ours) (e) Chain of Code Execution (Ours)<br>Q: I have an orange, a violin, two peaches, Q: I have an orange, a violin, two peaches, an apple, a pepper,<br>an apple, a pepper, and three plums. How and three plums. How many fruits do I have?<br>many fruits do I have?<br>1 objects = {"orange": 1, "violin": 1, 1"apple":objects1, ="pepper":{"orange":1, 1,"plum":"violin":3} 1, "peaches": 2,<br>"peaches": 2, "apple": 1, "pepper": 1, delta state: {objects = {‘orange’: 1, ‘violin’: 1, ...}}<br>"plum": 3}<br>2 num_fruits = 0<br>2 num_fruits = 0 delta state: {num_fruits = 0}<br>3 for object in objects:<br>3 for object in objects: delta state: {object = ‘orange’} # updated for each loop<br>4 object_is_fruit = is_fruit(object) 4delta state:object_is_fruit{object_is_fruit= is_fruit(object)= True}<br>5 if object_is_fruit: 5 if object_is_fruit:<br>delta state: {}<br>6 num_fruits += objects[object] 6 num_fruits += objects[object]<br>delta state: {num_fruits = 1}<br>7 answer = num_fruits 7 answer = num_fruits<br>delta state: {answer = 7}<br>A: 7<br>**----- End of picture text -----**<br>


_Figure 3._ **Previous reasoning methods:** To solve advanced problems, (3a) Chain of Thought prompting breaks the problem down into intermediate steps, (3b) Program of Thoughts prompting writes and executes code, and (3c) ScratchPad prompting simulates running already written code by tracking intermediate steps through a program state. **Our reasoning method:** Chain of Code first (3d) generates code or psuedocode to solve the question and then (3e) executes the code with a code interpreter if possible, and with an LMulator (language model emulating code) otherwise. Blue highlight indicates LM generation, red highlight indicates LM generated code being executed, and purple highlight indicates LMulator simulating the code via a program state in green. 

2. It leverages the ability of language models to code, a particular strength of recent language models due to the high quality data available. 

3. It inherits many of the benefits of reasoning code, both the formal yet expressive structure of code (e.g., Turing completeness) and powerful computational tools available to code (whether simply multiplying two numbers, calculating ~~_[√]_~~ 5 12121, or simulating physics). 

## **3. Experimental Evaluation** 

We select challenging problems requiring varied types of reasoning, whether arithmetic, commonsense, or symbolic reasoning tasks, to answer the following questions: 

   1. How well does CoC perform across a variety of tasks? 

   2. Which types of problems does CoC perform best? 

   3. How does each aspect of CoC affect performance? 

   4. How does CoC scale with model size? 

4. It inherits many of the benefits of techniques that reason via intermediate steps, such as Chain of Thought. These techniques enable the language model to use more computation when necessary to solve a problem as well as provide more interpretability. 

5. How does CoC perform as a general-purpose reasoner, with prompt examples from different problems rather than the same problem (which we term cross-task prompting)? 

6. How can CoC be used with instruction tuned chat models? 

7. How robust CoC is against prompt variation? 

8. Can CoC be applied beyond language reasoning tasks? 

Empirically, we observe in Section 3 that these benefits results in significant improvements in reasoning performance over a variety of challenging tasks. 

We first discuss the approaches, ablations, and baselines considered in Section 3.1, then the tasks considered in Section 3.2, and finally the results in Section 3.3. 

4 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

## **3.1. Baselines and Ablations** 

We consider our main method to be **CoC (Interweave)** , also referred to as **CoC (Ours)** , though we also propose two variants with simpler implementation and modestly lower performance: **CoC (try Python except LM)** and **CoC (try Python except LM state)** . These two variants attempt to run the entire generated code with Python (rather than line by line) and if it fails, simulate the code execution with the LMulator, outputting a final answer or an intermediate state trace, respectively. We also perform the following ablations, some of which are comparable to previous work as noted. In **CoC (Python)** Python is used to run the entire generated code and if the code is not executable, it is marked as failure – this can be thought of as a comparison to Program of Thoughts (Chen et al., 2022) or Program-aided language models (Gao et al., 2023). We note that in many cases this baseline is particularly challenged, as writing executable code for some of the reasoning problems becomes nearly impossible (e.g., writing code to judge if a phrase is sarcastic), but one may focus on the results for Algorithmic only tasks for a more fair comparison. In **CoC (LM)** the code is interpreted by an LMulator outputting the final answer, and in **CoC (LM state)** the code is interpreted by an LMulator outputting a state trace of intermediate steps – this can be thought of as ScratchPad prompting for reasoning (Nye et al., 2021). Note, the last two ablations do not leverage the Python interpreter. 

We also compare against the following baselines. In **Direct** question answering the LM simply responds to the question with a final answer. In Chain of Thought prompting **(CoT)** the LM uses intermediate steps to solve the task; we use CoT as our standard prompt technique for the field of substep prompting (Kojima et al., 2022; Zhou et al., 2022a) as prompts are readily available. 

## **3.2. Tasks** 

We consider a subset of challenging tasks from BIG-Bench (Srivastava et al., 2022) called BIG-Bench Hard (BBH) (Suzgun et al., 2022) to ensure we are solving the most challenging tasks. These tasks were specifically selected for their difficulty for language models and the datasets provides human-rater baselines and a set of Chain of Thought prompts. The 23 tasks require semantic reasoning (e.g., “Movie Recommendation”), numerical reasoning (e.g., “Multi-Step Arithmetic”), and a combination of both (e.g., “Object Counting”). As such they enable us to study the efficacy of CoC across varied problems, not just those that coding is a natural fit for. Several prompts are shown in Figure A1. We also show results for the grade-school math (GSM8K) benchmark (Cobbe et al., 2021) in Section A.2, although we find that these problems are primarily solved algorithmically alone through code. 

These tasks are evaluated with **few-shot prompting** , whereby three examples from the same problem family are provided as context. We also introduce a new evaluation setting, **cross-task** 

**prompting** , whereby three examples of _different_ problems are provided as context. As such, the language model has in-context examples of the _format_ of reasoning, but isn’t provided explicit instructions on _how_ to reason. We see this as an indicative signal for a general-purpose reasoner, which in many real-world applications (e.g., chatbots) would be asked to reason across a wide variety of tasks. 

The models used herein include the OpenAI family of models: text-ada-001, text-baggage-001, text-curie-001, and text-davinci-003 (in plots we denote these as a-1, b-1, c-1, and d-3). We also consider PaLM-2’s code finetuned variant (Chowdhery et al., 2022; Google et al., 2023). For instruction tuned models, we compare to recent variants of GPT (gpt-3.5-turbo and gpt-4) with the chat completion mode run in October 2023 and January 2024. The results below are using the text-davinci-003 model unless otherwise stated. 

## **3.3. Results** 

**Question 1: Overall Performance.** The overall performance of CoC is shown in Figure 2 and Table 1 (with full results in Table A1). We see that CoC outperforms other approaches, both in the number of tasks it exceeds the human baseline and in the overall amount that it exceeds the baseline. Indeed, CoC’s 84% is SoTA to the best of our knowledge (Gemini Team, 2023). In fact, when combined with gpt-4, CoC achieves 91% (see Table A4). In several tasks CoC vastly outperforms the human baseline and other methods, achieving nearly 100% – generally for these tasks the result is complicated in language but trivial in code (e.g., a task from multi-step arithmetic Q: (( _−_ 3+5 _×_ 8 _×−_ 4) _−_ (9 _−_ 8 _×−_ 7)) =). We also observe that CoT outperforms the human baseline on a number of tasks, while the Direct answer fares poorly. 

**Question 2: Problem Type.** Figure 4 breaks the results down by problem type; the task labels are shown in Table A1. First, we isolate problems that are primarily algorithmic or primarily natural language (these categories were identified in (Suzgun et al., 2022)). We see that on algorithmic tasks, CoC performs particularly well, while on natural language tasks CoC performs on par with CoT. This is particularly encouraging, because one may expect these language oriented tasks to be a worse fit for code. The key is that our method offers the flexibility of using a LMulator to simulate the output of code execution, retaining the semantic reasoning capabilities of LMs for natural language problems. 

Figure 4 additionally breaks the tasks down into categories that capture how different each question’s response is and whether the code can be fully executed by Python (denoted Python only vs. Python + LM). For some tasks within the benchmark, each question has the same code or Chain of Thought, with the only variation being the inputs – in this case we say the code is (repeated code), and if not then it is denoted (new code). 

5 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

_Table 1._ Overall performance (%) on BIG-Bench Hard with both few-shot prompting with a single task and cross-task. The delta compared to direct prompting is shown in parenthesis. 

|Prompt<br>Human|text-davinci-003<br>Direct<br>CoT<br>CoC (Ours)|PaLM 2-S* (code variant (Google et al.,2023))<br>Direct<br>CoT<br>CoC (Ours)|
|---|---|---|
|Single task<br>68<br>Cross task<br>-|55<br>72 (+17)<br>84 (+29)<br>50<br>55 (+5)<br>61 (+11)|49<br>61 (+12)<br>78 (+29)<br>45<br>47 (+2)<br>47 (+2)|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0006-03.png)


_Figure 4._ Average performance across different baselines grouped by task type, indicating the problem type and how CoC is generated & executed. 

As expected, we see that when the code is repeated and run by Python, CoC gets nearly 100%, though these tasks (e.g., multi-step arithmetic) seem to be among the most challenging for the other baselines, including human raters. The other categories are more challenging for CoC; however in each, we still see a benefit over baselines. 

**Question 3: Ablations.** Figures 5 and 6, and Table 2 show the ablations performed to motivate each aspect of Chain of Code prompting. As one may expect, the approaches that execute Python (CoC (Interweave, Python, try Python except LM, try Python except LM state)) achieve 100% performance on several tasks – if the code is correct, then the model will be correct every time. However, the approach that relies on Python alone (CoC (Python)) performs poorly when applied to non-algorithmic tasks, failing almost all. The CoC (Python) ablation is similar to recent works (Gao et al., 2023; Chen et al., 2022), which show that if applied to numerical problems then code reasoning performs well. CoC without the Python interpreter (CoC (LM, LM state)) too fares poorly, though we see that the step-by-step approach proposed in ScratchPad prompting (Nye et al., 2021) improves in each task. 

We also show that ablations CoC (try Python except LM, try Python except LM state), in which CoC first tries to run the entire code with Python and if it fails simulates the code with an LM, perform quite well. Again we see that maintaining a program state provides an improvement in performance. With only minor degradations in performance observed, they are reasonable alternatives to the fully interweaved CoC for their simplicity. Though we note, these ablations’ performance 

would be much worse in cases where interweaving code and semantics is truly necessary – for example, if we imagine a case where code is necessary to parse image inputs or to access an external database, but language is necessary to parse the results (see the robotics applications in Section A.6). 

**Question 4: Scaling.** Figure 7 shows the performance of CoC across various model sizes. We observe that, similar to Chain of Thought prompting, the improvements of CoC increases as model size increases. In fact, for some of the algorithmic tasks, Chain of Code even outperforms the best human raters (whom admittedly did not have access to code). Unlike Chain of Thought prompting, however, which only brings performance benefits for the largest model (d-3), CoC outperforms the direct question answering baseline also for smaller models (a-1, b-1, c- 1), suggesting that it’s easier for smaller models to output structured code as intermediate steps rather than natural languages. 

**Question 5: Cross-task Prompting.** For cross-task prompting, we prompt the language models with a few examples from different problems. We see the performance drops for all methods in Figure 7 and Table 1. Despite this drop, CoC outperforms CoT and direct prompting at scale, nearly achieving human average performance. This is a promising indication towards general purpose reasoning, in which a model does not expect to receive examples of similar problems in its prompt. 

**Question 6: Instruction Tuned Models.** The reason why we chose text-davinci-003, a completion model, as our primary evaluation model, over more advanced instruction tuned models (gpt-3.5-turbo and gpt-4) is that the 

6 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-01.png)


_Figure 5._ Chain of Code ablations on average performance grouped by task type. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0007-08.png)


_Figure 6._ Results across all BIG-Bench Hard tasks compared to human baseline (Srivastava et al., 2022). The tasks (x-axis) in each plot are sorted individually by performance. See Table A1 and Figure 5 for a breakdown by task type. 

_Table 2._ Ablation overall performance (%) with both few-shot prompting with a single task and cross-task. The delta compared to the full model (Interweave) is shown in parenthesis. 

|Prompt|Chain of Code<br>Interweave<br>try Python<br>try Python<br>Python<br>LM state<br>LM<br>except LM state<br>except LM|
|---|---|
|Single task<br>Cross task|84<br>82 (-2)<br>80 (-4)<br>48 (-36)<br>63 (-21)<br>57 (-27)<br>61<br>57 (-4)<br>60 (-1)<br>35 (-26)<br>49 (-12)<br>50 (-11)|



former is more amenable to few-shot prompting with examples, which is the main evaluation paradigm for BIG-Bench Hard. However, we still made our best attempt to evaluate our method with the instruction tuned models using two different setups. The first is zero-shot prompting, where we directly prompt the models via the system message to output direct answers, chain of thoughts, or pseudocode/code (which we optionally execute with the python interpreter and feed back the results). The second is few-shot prompting, where we coerce the models to behave like completion models via the system message, and feed the few-shot examples as usual. In both cases, we demonstrated that CoC brings noticeable benefits with little 

modification needed. See Sec. A.4 for more details. 

**Question 7: Robustness of Chain of Code** We showed that CoC is generally robust against prompt variation by evaluating with different prompts independently written by three annotators on the same set of problems. Specifically, we select four representative tasks from BIG-Bench Hard that require generation of new code (as opposed to repeated code). While the performance of individual tasks has some variance, the average performance across the four tasks only vary within a few percentage points. See Sec. A.5 for more details. 

**Question 8: Beyond Language Reasoning** We showed that 

7 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0008-01.png)


_Figure 7._ Average performance with model scaling, from text-ada-001 (smallest) to text-davinci-003 (largest). 

CoC is well-suited for tasks that require both semantic and algorithmic reasoning beyond language reasoning, such as robotics. The unique advantage of CoC in robotics is that it interact seamlessly with the robot perception and control APIs via python code such as running object detectors or invoking parameterized robot skills, while performing semantic subtasks in an “inline” fashion (e.g. classifying what trash is compostable before picking them). When equipped with the necessary robot APIs, and a single example in the prompt to teach LMs the format, CoC can solve seven different robot manipulation tasks in the real world, showcasing generalization to new objects, languages and task domains. See Sec. A.6 for more details. 

## **4. Related Work** 

**Language Model Reasoning** The abilities and applications of language models have seen significant progress, due to their overall performance (Chowdhery et al., 2022; Touvron et al., 2023; Radford et al., 2019; Gemini Team, 2023) and emergent capabilities (Wei et al., 2022a), such as few-shot prompting (Brown et al., 2020) and abstract reasoning (Wei et al., 2022b). Perhaps most related to this work, a number of works have leveraged prompting to improve reasoning (Dohan et al., 2022): Chain of Thought (Wei et al., 2022b) proposes to break a task down into intermediate reasoning steps, leastto-most (Zhou et al., 2022a) proposes a series of increasingly simpler problems, and ScratchPad (Nye et al., 2021) proposes to maintain a trace of intermediate results for interpreting code (this first demonstrated the code simulation ability of LMs required for our LMulator). Along these lines “let’s think step by step” (Kojima et al., 2022) uses a few key words to elicit such break downs (words that were later refined to “Take a deep breath and work on this problem step-by-step” in (Yang et al., 2023)). Beyond these, other approaches structure such step-by-step solutions into graphical structures (Yao et al., 2023; Besta et al., 2023), plans (Wang et al., 2023b; Ning et al., 2023), or mixture of expert-based sampling (Wang et al., 2022; Zhou et al., 2022b). CoC builds upon the intuition of these works, with the observation that _code_ is a formal, structured approach to breaking a problem down into sub-steps with many advantages beyond natural language alone. 

**Language Model Tool Use** Many recent works have proposed 

techniques for language models to use tools to respond to queries (Mialon et al., 2023). These tools have often been provided to the language model through prompting (Cobbe et al., 2021; Khot et al., 2022; Chowdhery et al., 2022; Drori et al., 2022; Yao et al., 2022), enabling tools like calculators for math problems, code interpreters, databases, or more. These tools too can provide feedback on novel modalities (Surís et al., 2023; Zeng et al., 2022). To expand the range of tools available, others have used external tool databases or finetuned language models (Schick et al., 2023; Qin et al., 2023; Parisi et al., 2022; Paranjape et al., 2023). As tool interfaces vary, feedback from the tool too can improve performance (Gou et al., 2023; Zhou et al., 2023). In this work we leverage the expressibility and generality of full code as well as its structure, by treating it both as a tool and as a framework. 

**Language Model Program Synthesis** The ability of language models to code is well known and they have been applied as programming assistants (Chen et al., 2021) and shown to be capable programmers on their own (Austin et al., 2021; Li et al., 2022; Nijkamp et al., 2022). This ability has been applied to a variety of tasks outside of language alone, leveraging their ability to reason through code in new settings, such as robotics (Liang et al., 2023; Singh et al., 2023), embodied agents (Wang et al., 2023a), or vision (Surís et al., 2023). Others have specifically done so for reasoning, such as Program of Thoughts (Chen et al., 2022) and Program-aided Language Models (Gao et al., 2023), which generate code to solve numerical reasoning problems. Herein, we focus on the interplay between writing code, running code, and language models simulating code, thus enabling new regimes of language model code applications, such as semantic reasoning. 

## **5. Conclusions, Limitations, and Future Work** 

We have proposed Chain of Code, an approach towards reasoning with language models through writing code, and executing code either with an interpreter or with a language model that simulates the execution (termed herein an LMulator) if the code is not executable. As such, CoC can leverage both the expressive structure of code and the powerful tools available to it. Beyond this, by simulating the execution of non-executable code, CoC can apply to problems nominally 

8 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

outside the scope of code (e.g., semantic reasoning problems). We have demonstrated that this approach outperforms baselines, and for some tasks even the best human raters, in a range of challenging language and numeric reasoning problems. 

This work is not without its limitations. First, generating and executing in two steps as well as interweaving code and language execution requires additional context length and computation time. Second, though we have not seen any loss of performance for semantic tasks in aggregate, there are few tasks in which code doesn’t help, e.g., the task Ruin Names, which asks whether an edit for a name is humorous. Finally, our implementation to interweave LM and code is quite simple, tracking the program state in strings and parsing the strings into Python’s built-in data types (e.g., dict, tuple). As our method stands now, the LM cannot modify custom Python objects while simulating code execution. In theory, however, it is doable as long as each of these Python objects have a serialization and deserialization method, e.g., using techniques like Protocol Buffers. 

There are many avenues for future work with CoC. First, we believe that a unified code and language interpreter well combines the commonsense of language models with the analytical abilities, structure, and interpretability of code. Such a technology can thus enable applications of code and code-like reasoning to novel problem regimes, beyond simple reasoning. Second, we are interested in investigating the degree to which finetuning a language model to be an LMulator can benefit semantic code reasoning. Third, we see evidence that reasoning through many pathways yields improvements, which is a promising step forward. Finally, we believe this integration with code enables access to external modalities, such as vision or databases, and represents a interesting path for new applications (e.g., robotics, augmented reality). 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, most of which are related to the usage of large language models (LLMs). One aspect of Chain of Code that warrants further discussion is that CoC executes the output of LLMs using the Python interpreter as if they are always benign code. If deployed in the wild, however, Chain of Code will need to install additional safeguards against potentially harmful code from LLMs that might be maliciously prompted, before running the code. 

## **References** 

- Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gianinazzi, L., Gajda, J., Lehmann, T., Podstawski, M., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. _arXiv preprint arXiv:2308.09687_ , 2023. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _arXiv preprint arXiv:2211.12588_ , 2022. 

- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Dohan, D., Xu, W., Lewkowycz, A., Austin, J., Bieber, D., Lopes, R. G., Wu, Y., Michalewski, H., Saurous, R. A., Sohl-Dickstein, J., et al. Language model cascades. _arXiv preprint arXiv:2207.10342_ , 2022. 

9 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

- Drori, I., Zhang, S., Shuttleworth, R., Tang, L., Lu, A., Ke, E., Liu, K., Chen, L., Tran, S., Cheng, N., et al. A neural network solves, explains, and generates university math problems by program synthesis and few-shot learning at human level. _Proceedings of the National Academy of Sciences_ , 119(32):e2123433119, 2022. 

- Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. 

- Gemini Team, G. Gemini: A family of highly capable multimodal models. Technical report, Google, 2023. URL https://storage.googleapis.com/ deepmind-media/gemini/gemini_1_report. pdf. 

- Google, Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

- Gou, Z., Shao, Z., Gong, Y., Shen, Y., Yang, Y., Duan, N., and Chen, W. Critic: Large language models can self-correct with tool-interactive critiquing. _arXiv preprint arXiv:2305.11738_ , 2023. 

- Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., and Sabharwal, A. Decomposed prompting: A modular approach for solving complex tasks. _arXiv preprint arXiv:2210.02406_ , 2022. 

- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., and Girshick, R. Segment anything. _arXiv:2304.02643_ , 2023. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models, 2022. _arXiv preprint arXiv:2206.14858_ , 2022. URL https://arxiv.org/abs/2206.14858. 

- Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., et al. Competition-level code generation with alphacode. _Science_ , 378(6624):1092–1097, 2022. 

- Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control. In _2023 IEEE_ 

_International Conference on Robotics and Automation (ICRA)_ , pp. 9493–9500. IEEE, 2023. 

- Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. _arXiv preprint arXiv:2303.05499_ , 2023. 

- Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., Dwivedi-Yu, J., Celikyilmaz, A., et al. Augmented language models: a survey. _arXiv preprint arXiv:2302.07842_ , 2023. 

- Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., and Zettlemoyer, L. Rethinking the role of demonstrations: What makes in-context learning work? _arXiv preprint arXiv:2202.12837_ , 2022. 

- Mirchandani, S., Xia, F., Florence, P., Ichter, B., Driess, D., Arenas, M. G., Rao, K., Sadigh, D., and Zeng, A. Large language models as general pattern machines. _arXiv preprint arXiv:2307.04721_ , 2023. 

- Nijkamp, E., Pang, B., Hayashi, H., Tu, L., Wang, H., Zhou, Y., Savarese, S., and Xiong, C. Codegen: An open large language model for code with multi-turn program synthesis. _arXiv preprint arXiv:2203.13474_ , 2022. 

- Ning, X., Lin, Z., Zhou, Z., Yang, H., and Wang, Y. Skeletonof-thought: Large language models can do parallel decoding. _arXiv preprint arXiv:2307.15337_ , 2023. 

- Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

OpenAI. Gpt-4 technical report, 2023. 

- Paranjape, B., Lundberg, S., Singh, S., Hajishirzi, H., Zettlemoyer, L., and Ribeiro, M. T. Art: Automatic multi-step reasoning and tool-use for large language models. _arXiv preprint arXiv:2303.09014_ , 2023. 

- Parisi, A., Zhao, Y., and Fiedel, N. Talm: Tool augmented language models. _arXiv preprint arXiv:2205.12255_ , 2022. 

- Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. _arXiv preprint arXiv:2307.16789_ , 2023. 

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

10 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. _arXiv preprint arXiv:2302.04761_ , 2023. 

- Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J., and Garg, A. Progprompt: Generating situated robot task plans using large language models. In _2023 IEEE International Conference on Robotics and Automation (ICRA)_ , pp. 11523–11530. IEEE, 2023. 

- Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ , 2022. 

- Surís, D., Menon, S., and Vondrick, C. Vipergpt: Visual inference via python execution for reasoning. _arXiv preprint arXiv:2303.08128_ , 2023. 

- Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging big-bench tasks and whether chain-ofthought can solve them. _arXiv preprint arXiv:2210.09261_ , 2022. 

- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

   - Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., and Chen, X. Large language models as optimizers. _arXiv preprint arXiv:2309.03409_ , 2023. 

   - Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

   - Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. _arXiv preprint arXiv:2305.10601_ , 2023. 

   - Zeng, A., Attarian, M., Ichter, B., Choromanski, K., Wong, A., Welker, S., Tombari, F., Purohit, A., Ryoo, M., Sindhwani, V., et al. Socratic models: Composing zero-shot multimodal reasoning with language. _arXiv preprint arXiv:2204.00598_ , 2022. 

   - Zhou, A., Wang, K., Lu, Z., Shi, W., Luo, S., Qin, Z., Lu, S., Jia, A., Song, L., Zhan, M., et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. _arXiv preprint arXiv:2308.07921_ , 2023. 

   - Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., et al. Least-to-most prompting enables complex reasoning in large language models. _arXiv preprint arXiv:2205.10625_ , 2022a. 

   - Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A. M., Le, Q. V., Laudon, J., et al. Mixture-of-experts with expert choice routing. _Advances in Neural Information Processing Systems_ , 35:7103–7114, 2022b. 

- Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An open-ended embodied agent with large language models. _arXiv preprint arXiv:2305.16291_ , 2023a. 

- Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., and Lim, E.-P. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. _arXiv preprint arXiv:2305.04091_ , 2023b. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_ , 2022a. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022b. 

11 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

## **A. Appendix** 

## **A.1. Quantitative results on language reasoning tasks** 

Table A1 shows the full per-task results across ablations on BIG-Bench Hard (BBH) tasks, as well as broken down by task type and execution type. 

_Table A1._ Full results across ablations on BIG-Bench Hard (BBH) tasks. 

|BIG-Bench Hard Task|Srivastava et al.(2022)<br>Suzgun et al.(2022)<br>Rand.<br>Human<br>(Avg.)<br>Human<br>(Max)<br>Direct<br>CoT|Chain of Code<br>Inter-<br>weave<br>try<br>Python<br>except<br>LM<br>state<br>try<br>Python<br>except<br>LM<br>Python<br>LM<br>state<br>LM|
|---|---|---|
|Boolean Expressions_λ_+<br>Causal Judgement_κ∗_<br>Date Understanding_κ−_<br>Disambiguation QA_κ/_<br>Dyck Languages_λ_+<br>Formal Fallacies_κ∗_<br>Geometric Shapes_λ_+<br>Hyperbaton_κ/_<br>Logical Deduction_λ∗_<br>Movie Recommendation_κ/_<br>Multi-Step Arithmetic_λ_+<br>Navigate_λ∗_<br>Object Counting_λ−_<br>Penguins in a Table_κ−_<br>Reasoning about Colored Objects_κ−_<br>Ruin Names_κ/_<br>Salient Translation Error Detection_κ/_<br>Snarks_κ/_<br>Sports Understanding_κ/_<br>Temporal Sequences_λ∗_<br>Tracking Shuffled Objects_λ−_<br>Web of Lies_λ−_<br>Word Sorting_λ_+|50<br>79<br>100<br>88<br>89<br>50<br>70<br>100<br>64<br>64<br>17<br>77<br>100<br>61<br>84<br>33<br>67<br>93<br>70<br>68<br>1<br>48<br>100<br>6<br>50<br>25<br>91<br>100<br>56<br>56<br>12<br>54<br>100<br>48<br>66<br>50<br>75<br>100<br>63<br>64<br>23<br>40<br>89<br>49<br>66<br>25<br>61<br>90<br>85<br>81<br>0<br>10<br>25<br>0<br>48<br>50<br>82<br>100<br>58<br>94<br>0<br>86<br>100<br>30<br>82<br>0<br>78<br>100<br>62<br>82<br>12<br>75<br>100<br>64<br>87<br>25<br>78<br>100<br>76<br>70<br>17<br>37<br>80<br>66<br>61<br>50<br>77<br>100<br>70<br>71<br>50<br>71<br>100<br>72<br>96<br>25<br>91<br>100<br>38<br>60<br>23<br>65<br>100<br>25<br>72<br>50<br>81<br>100<br>54<br>100<br>0<br>63<br>100<br>51<br>50|100<br>100<br>100<br>100<br>95<br>90<br>56<br>57<br>63<br>0<br>57<br>60<br>75<br>72<br>74<br>59<br>66<br>57<br>71<br>67<br>68<br>0<br>67<br>68<br>100<br>100<br>99<br>99<br>1<br>7<br>55<br>54<br>55<br>0<br>54<br>56<br>100<br>100<br>100<br>100<br>13<br>44<br>98<br>62<br>55<br>0<br>62<br>55<br>68<br>79<br>57<br>0<br>79<br>58<br>80<br>83<br>80<br>0<br>83<br>79<br>100<br>100<br>100<br>100<br>0<br>1<br>86<br>84<br>68<br>0<br>84<br>68<br>96<br>98<br>98<br>98<br>57<br>50<br>90<br>88<br>90<br>88<br>71<br>59<br>78<br>74<br>78<br>64<br>64<br>70<br>55<br>56<br>46<br>0<br>56<br>47<br>58<br>63<br>64<br>0<br>63<br>64<br>76<br>76<br>66<br>0<br>76<br>66<br>91<br>93<br>75<br>0<br>93<br>74<br>98<br>93<br>99<br>93<br>93<br>99<br>100<br>96<br>96<br>96<br>71<br>24<br>97<br>96<br>96<br>97<br>96<br>50<br>99<br>100<br>99<br>100<br>54<br>54|
|Task Averages<br>NLP Task (avg)_κ_<br>Algorithmic Task (avg)_λ_<br>All Tasks (avg)|30<br>71<br>97<br>67<br>74<br>21<br>64<br>92<br>41<br>71<br>26<br>68<br>95<br>55<br>72|74<br>70<br>68<br>18<br>68<br>63<br>95<br>95<br>92<br>80<br>58<br>50<br>84<br>82<br>80<br>48<br>63<br>57|
|Execution Type<br>Python exec (same program)+<br>Python exec (different program)_−_<br>LM exec (same program)_/_<br>LM exec (different program)_∗_|13<br>51<br>85<br>38<br>61<br>17<br>77<br>100<br>49<br>84<br>36<br>66<br>95<br>72<br>73<br>35<br>75<br>98<br>53<br>68|100<br>100<br>100<br>100<br>33<br>39<br>89<br>87<br>89<br>84<br>71<br>52<br>76<br>71<br>65<br>0<br>71<br>65<br>72<br>73<br>68<br>19<br>73<br>68|



_λ_ denotes an algorithmic task and _κ_ denotes an NLP task (with categories outlined in Suzgun et al. (2022)). + denotes a task where the code between prompts is repeated and can be executed by Python, _−_ denotes a task where the code between prompts must change and can be executed by Python, _/_ denotes a task where the code between prompts is repeated and must be executed by the LM, and _∗_ denotes a task where the code between prompts must change and must be executed by the LM. 

## **A.2. Quantitative results on the GSM8K Benchmark** 

Table A2 shows results on the the grade-school math benchmark (GSM8K) (Cobbe et al., 2021) with direct prompting, Chain of Thought, and Chain of Code. We find that CoC generally outperforms CoT and Direct prompting. Since these tasks are primarily algorithmic and are solved by Python alone, all Chain of Code variants that use Python achieve the same performance – also the same performance shown in Program of Thoughts (Chen et al., 2022). 

## **A.3. Qualitative results on language reasoning tasks** 

Figure A1 shows the model outputs for a few reasoning tasks from BIG-Bench Hard (BBH) and Figure A2 shows a demonstrative example of date reasoning. These examples are selected to highlight the interweaving execution of the Python interpreter and the LMulator. 

12 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

_Table A2._ GSM8K (Cobbe et al., 2021) performance (%) with both few-shot prompting with a single task and cross-task. The delta compared to direct prompting is shown in parenthesis. 

|Prompt<br>Direct<br>CoT|Chain of Code<br>Interweave<br>try Python<br>try Python<br>Python only<br>LM state<br>LM only<br>except LM state<br>except LM|
|---|---|
|Single task<br>16<br>63 (47)<br>Cross task<br>14<br>55 (41)|71 (55)<br>72 (56)<br>71 (55)<br>71 (55)<br>45 (29)<br>22 (6)<br>60 (46)<br>60 (46)<br>60 (46)<br>60 (46)<br>41 (27)<br>16 (2)|



## **A.4. Instruction Tuned Models** 

Since most of the results presented in our main paper are using text-davinci-003, a completion model that is particularly amenable to few-shot prompting, one may wonder how CoC can be used with instruction tuned models, like gpt-4 (OpenAI, 2023). Figuring out ways to elicit the desired behavior of CoC from these instruction tuned models (i.e. writing code/pseudocode to solve problems) is non-trivial. We conduct two additional experiments below as our best effort to shed some light on this subject. 

**Zero-shot prompting.** We directly prompt the models with instructions to elicit the desired reasoning approaches. Note that we do not provide few-shot examples in the prompt (hence “zero-shot”). For the baselines, we ask the model to “directly answer” (Direct) or “think step by step” (CoT). For CoC variants, we ask the model to “write python code to help solve the problem, if it’s helpful”. If a program is written, we either run the code with a Python interpreter and then feed the result (or the error message if execution fails) back to the model to determine a final answer (CoC (Python)), or ask the model to simulate the output of code execution as a LMulator (CoC (LM)). The CoC (Python) baseline can be thought of as a comparison to an LM with Python tool use. 

Table A3 shows the performance of each. With gpt-3.5-turbo, both CoT and CoC (Python) show benefits over direct prompting, although both are strongly outperformed by CoC (Interweave). With gpt-4, despite the considerable model strength advantage over text-davinci-003, CoC (Interweave) still outperforms, though the gap is narrower. Admittedly, CoC (Interweave) is prompted with three examples whereas the other two are not. 

_Table A3._ Comparisons with instruction tuned models in the chat interface, with and without tool use, denoted as CoC (Python) and CoC (LLM) respectively. The delta compared to CoC with text-davinci-003 is shown in parenthesis. In this experiment, the prompts for gpt-3.5-turbo and gpt-4 only contain a generic, shared system message and do not contain few-shot examples. 

||text-davinci-003<br>CoC<br>(Interweave)|gpt-3.5-turbo<br>Direct<br>CoT<br>CoC<br>CoC<br>(Python)<br>(LM)|gpt-4<br>Direct<br>CoT<br>CoC<br>CoC<br>(Python)<br>(LM)|
|---|---|---|---|
||84|51 (-33)<br>56 (-28)<br>56 (-28)<br>45 (-39)|70 (-14)<br>78 (-6)<br>82 (-2)<br>75 (-9)|



**Few-shot prompting.** We attempt to coerce the instruction tuned models to behave like completion models by using the following system message: “Pretend you are a completion model that is prompted with three examples. You should follow the pattern of these examples strictly. At the end of your reply, you should always output an answer”. In the user message, we include three examples from the same (single task) or different (cross task) task domains, as well as the query question, exactly the same as how we evaluated the completion models. 

Table A4 shows that CoC still brings a sizable performance gain over the Direct and CoT baselines. With gpt-4, the gap is again narrower mainly because the base model already performs quite well and leaves little room for improvement. This experiment suggests a viable way to combine the strength of CoC with that of more advanced instruction tuned models like gpt-4. 

_Table A4._ Applying CoC to instruction tuned models in the chat interface, while coercing them to behave like completion models. The delta compared to direct prompting is shown in parenthesis. In this experiment, the prompts contains a generic, shared system message that asks LMs to behave like completion models, and also three examples from the same or different task domains at the beginning of the user message. as before. 

|Prompt|gpt-3.5-turbo<br>Direct<br>CoT<br>CoC|gpt-4<br>Direct<br>CoT<br>CoC|
|---|---|---|
|Single task<br>Cross task|47<br>73 (+26)<br>79 (+32)<br>47<br>60 (+13)<br>61 (+14)|69<br>88 (+19)<br>91 (+22)<br>67<br>81 (+14)<br>84 (+17)|



13 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

## **A.5. Robustness of Chain of Code** 

Similar to Chain of Thought prompts, Chain of Code prompts can also come with various forms: e.g. different ways of function decomposition, coding styles, variable names, reasoning pathways, and so on. In this section, we want to analyze whether CoC is robust against variation across prompts. 

We invite three annotators that are familiar with Python to write CoC prompts for four representative tasks in BIG-Bench Hard. We select these four tasks because they all require generation of new code (as opposed to repeated code) during test time. As before, for single task evaluation, we prompt LMs with three examples from the same task domain, whereas for cross task evaluation, we prompt LMs with three examples from different task domains (one from each of the other three tasks). 

Results in Table A5 show that our method is robust against prompt variation and doesn’t require extensive prompt engineering. 

_Table A5._ Performance variation across prompts written independently by different authors for four representative tasks in BIG-Bench Hard. Our results that CoC is generally robust against prompt variation, allowing for different coding styles and reasoning logic in the few-shot prompts. 

|Prompt|Prompt Annotator|Date Understanding|Logical Deduction|Object Counting|Penguins in a Table|Average|
|---|---|---|---|---|---|---|
|Single task|A|73|64|92|78|77|
||B|68|54|95|88|76|
||C|69|43|90|89|73|
|Cross task|A|41|33|67|76|54|
||B|48|29|78|88|61|
||C|60|30|76|64|57|



## **A.6. Robotics Applications** 

Downstream applications such as robotics are well fit for CoC as robotics tasks require semantic reasoning and algorithmic reasoning, as well as interfacing with other APIs through code (such as control or perception APIs (Liang et al., 2023)) and with users through natural language. For example, given a task like “sort the fruits by size”, the robot must reason over which items are fruits, sort them by size, and then connect those decisions to actions executable on the robot. CoC (Interweave) is able to solve these challenges with the Python interpreter and the LMulator at runtime, while allowing for more interpretability and fine-grained control of the robot policies. 

**Environment and Robot Setup.** Our environment is a tabletop with small objects (containers, toys, etc) and a UR5 robot arm equipped with a vacuum gripper and a wrist-mounted RGB-D camera. For the purpose of our experiments, the available perception API is detect_objects(), which returns a list of detected objects (probabilities, labels, bounding boxes and segmentation masks) from the wrist camera. This API is implemented with first querying GPT-4V (OpenAI, 2023) for a list of objects, and then using Grounding-SAM (Kirillov et al., 2023; Liu et al., 2023) to localize them. The available control API is pick_place(obj1, obj2), which is a scripted primitive skill that picks up obj1 and places it on top of obj2. There is also a text-to-speech API say(sentence) that allows the robot to communicate with the user. 

**Experimental Setup.** We evaluate with a number of tabletop pick-and-place robotics tasks that involve semantic reasoning. For few-shot prompting, we include a single example: “Serve a meal that follows the user’s dietary restrictions”, so that the language model understands the expected structure as well as the available robot APIs. During test time, we query the model with each of the following instructions. 

1. “Pack a lunch box for someone who is on a vegan diet.” 

2. “Assemble a sandwich for someone who is vegetarian.” 

3. “Gather ingredients for a peanut butter sandwich in a plate.” 

4. “Prepare `西红柿炒蛋` in the pot.” (interleaving English and Chinese on purpose) 

5. “Place all paper-made objects in the grass-colored container.” 

6. “Sort the objects on the table into the compost bin and the recycle bin.” 

7. “My steak is too bland. Can you help?” 

**Results.** With a single example in our prompt, we see that our model is able to generalize to new objects, languages, and task domains (see Figure A3 and an example trajectory in Figure A4). Note that for these robotics tasks, unlike the previous language 

14 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

reasoning tasks, our main method CoC (Interweave) is the only capable approach, as the code requires line-by-line interplay between the Python interpreter execution (robot APIs) and the LMulator (commonsense QA like is_compostable). 

Figure A3 shows the one-shot prompt as well as the model outputs and how they are executed for a few test instructions. 

15 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

## (b) Hyperbaton 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0016-02.png)


**----- Start of picture text -----**<br>
(a) Movie Recommendation<br>Q: Find a movie similar to Batman, The Mask, The Fugitive, Pretty<br>Woman:<br>Options:<br>(A) The Front Page<br>(B) Maelstrom<br>(C) The Lion King<br>(D) Lamerica<br>options = {"The Front Page": "(A)", "Maelstrom": "(B)", "The<br>Lion King": "(C)", "Lamerica": "(D)"}<br>delta state: {options = {‘The Front Page’: ..., ...}}<br>query_movies = list(options.keys())<br>delta state: {query_movies = [‘The Front Page’, ...]}<br>ref_movies = ["Batman", "The Mask", "The Fugitive", "Pretty<br>Woman"]<br>delta state: {ref_movies = [‘Batman’, ..., ‘Pretty Woman’]}<br>ref_movie_infos = get_movie_genre_and_year(ref_movies,<br>ret_type="list[tuple[str, int]]")<br>delta state: {ref_movie_infos = [(‘action’, 1989), ...]}<br>most_similar_movie = get_most_similar_movie(query_movies,<br>ref_movies, ref_movie_infos, ret_type="str")<br>delta state: {most_similar_movie = ‘The Lion King’}<br>answer = options[most_similar_movie] if most_similar_movie in<br>options else None<br>delta state: {answer = ‘(C)’}<br>**----- End of picture text -----**<br>


||(b) Hyperbaton|(b) Hyperbaton||||
|---|---|---|---|---|---|
||Q: Which sentence has the correct adjective order:|||||
||Options:<br>(A) small grey square cloth Japanese shoe|||||
||(B) small square grey Japanese cloth shoe"|||||
||import numpy as np|||||
||delta state:<br>{}|||||
||options = {"(A)":<br>"small grey square cloth Japanese||shoe",|||
||"(B)":<br>"small square grey Japanese cloth shoe"}<br>delta state:<br>{options = {‘(A)’: ..., ‘(B)’: ...}}|||||
||priority = {"opinion":<br>1, "size":<br>2, "age":|3, "shape":|||4,|
||"color":<br>5, "origin":<br>6, "material":<br>7, "purpose":|||8}||
||delta state:<br>{priority = {‘opinion’:<br>.., ..,|‘purpose’:||||
||..}}|||||
||valid_types = list(priority.keys())|||||
||delta state:<br>{valid_types = [‘opinion’, ..., ‘purpose’]}|||||
||scores = []|||||
||delta state:<br>{scores = []}|||||
||for option, sentence in options.items():|||||
||delta state:<br>{option, sentence = ‘(A)’, ‘small|...|shoe’}|||
||# updated for each loop|||||
||adjs = sentence.split(" ")[:-1]|||||
||delta state:<br>{adjs = [‘small’, ‘grey’, ‘square’, ‘cloth’]}|||||
||order = [priority[get_adjective_type(adj, valid_types,<br>ret_type=str)] for adj in adjs]|||||
||delta state:<br>{order = [2, 5, 4, 6]}|||||
||scores.append([order[i+1] > order[i] for i in<br>range(len(order) - 1)].count(True))|||||
||delta state:<br>{scores = [2]}|||||
||answer = list(options.keys())[np.argmax(scores)]|||||
||delta state:<br>{answer = ‘(B)’}|||||
|||||||



## (d) Disambiguation QA 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0016-05.png)


**----- Start of picture text -----**<br>
(c) Logical Deduction<br>Q: The following paragraphs each describe a set of three objects<br>arranged in a fixed order. The statements are logically consis-<br>tent within each paragraph. On a shelf, there are three books:<br>a green book, a red book, and a blue book. The red book is the<br>rightmost. The blue book is to the right of the green book.<br>Options:<br>(A) The green book is the leftmost<br>(B) The red book is the leftmost<br>(C) The blue book is the leftmost<br>options = {"green": "(A)", "red": "(B)", "blue": "(C)"}<br>delta state: {options = {‘green’: ..., ..., ‘blue’: ...}}<br>order_info = "left to right"<br>delta state: {order_info = ‘left to right’}<br>full_order = [None, None, None]<br>delta state: {full_order = [None, None, None]}<br>partial_order = []<br>delta state: {partial_order = []}<br>full_order[-1] = "red"<br>delta state: {full_order = [None, None, ‘red’]}<br>partial_order.append(("green", "blue"))<br>delta state: {partial_order = [(‘green’, ‘blue’)]}<br>full_order = generate_full_order(full_order, partial_order,<br>ret_type=list)<br>delta state: {full_order = [‘green’, ‘blue’, ‘red’]}<br>query = "leftmost"<br>delta state: {query = ‘leftmost’}<br>result = query_result(order_info, full_order, query,<br>ret_type=str)<br>delta state: {result = ‘green’}<br>answer = options[result] if result in options else None<br>delta state: {answer = ‘(A)’}<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0016-06.png)


**----- Start of picture text -----**<br>
Q: In the following sentences, explain the antecedent of the<br>pronoun (which thing the pronoun refers to), or state that it is<br>ambiguous.<br>Sentence: The homeowner asked the inspector if the house they<br>had purchased was structurally sound.<br>Options:<br>(A) The homeowner had purchased<br>(B) The inspector had purchased<br>(C) Ambiguous<br>context = "The homeowner asked the inspector if the house they<br>had purchased was structurally sound."<br>delta state: {context = ‘The homeowner asked ... sound.’}<br>pronoun = "they"<br>delta state: {pronoun = ‘they’}<br>a = "homeowner"<br>delta state: {a = ‘homeowner’}<br>b = "inspector"<br>delta state: {b = ‘inspector’}<br>version_a = "The homeowner asked the inspector if the house<br>the homeowner had purchased was structurally sound."<br>delta state: {version_a = ‘The homeowner asked ... sound.’}<br>version_b = "The homeowner asked the inspector if the house<br>the inspector had purchased was structurally sound."<br>delta state: {version_b = ‘The homeowner asked ... sound.’}<br>valid_a = can_pronoun_refer_to_noun(pronoun=pronoun, noun=a,<br>full_sentence=version_a, ret_type=bool)<br>delta state: {valid_a = True}<br>valid_b = can_pronoun_refer_to_noun(pronoun=pronoun, noun=b,<br>full_sentence=version_b, ret_type=bool)<br>delta state: {valid_b = False}<br>if valid_a and not valid_b:<br>delta state: {}<br>answer = "(A)"<br>delta state: {answer = ‘(A)’}<br>elif valid_b and not valid_a:<br>answer = "(B)"<br>else:<br>answer = "(C)"<br>**----- End of picture text -----**<br>


_Figure A1._ Model outputs for a few reasoning tasks from BIG-Bench Hard (BBH). We observe that CoC can apply to a wide variety of complex reasoning tasks that involve both semantic and numeric reasoning. Red highlight indicates LM generated code being executed by the Python interpreter, and purple highlight indicates LM simulating the code execution. 

16 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0017-01.png)


**----- Start of picture text -----**<br>
Direct answer only Chain of Code<br>Q: What holiday is 314 days after Valentine’s Day Q: What holiday is 314 days after Valentine’s Day in 2024?<br>in 2024?<br>A: Christmas (35%, ✗ ), New Year’s (15%, ✗ ), 1 from datetime import date, timedelta<br>Other (45%, ✗ ), Christmas Eve (0%, ✓ ) delta_state: {}<br>2 day1 = get_valentines_day_date(2024)<br>Chain of Thought delta_state: {day1 = date(year=2024, month=2, day=14)}<br>Q:in What2024?holidayLet’s isthink314stepdaysbyafterstep.Valentine’s Day 3delta_state:day2 = day1{day2+ timedelta(days=314)= date(year=2024, month=12, day=24)}<br>Step 1: Count the days in February after Valen- 4 answer = get_holiday(day2)<br>tine’s Day: ... delta_state: {answer = ‘Christmas Eve’}<br>A: Christmas Eve (35%, ✓ ), Christmas (5%, ✗ ),<br>Nov X (20%, ✗ ), Jan X (20%, ✗ ), Other (20%, ✗ ) A: Christmas Eve (100%, ✓ )<br>**----- End of picture text -----**<br>


_Figure A2._ A demonstrative example of how Chain of Code generates code and reasons through an LM-augmented code emulator. Lines evaluated with Python are in red and with an LM are in purple. The chain of thought and direct answers were evaluated with gpt-4 in October 2023, and we note the current model (as of December 2023) writes code to solve this problem and gets the same solution as Chain of Code. 

17 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 

## (b) Novel Objects 

(a) Given Prompt Q: Serve a meal that follows the user’s dietary restrictions. restrictions = input("Do you have any dietary restrictions?") _, _, objs, _ = robot.detect_objects() container_obj = None for obj in objs: is_match_description = match_description(obj, "food container", ret_type="bool") if is_match_description: container_obj = obj break if container_obj is None: robot.say("No food container found.") else: for obj in objs: if obj == container_obj: continue is_follow_restrictions = follow_restrictions(obj, restrictions, ret_type="bool") if is_follow_restrictions: robot.pick_place(obj, container_obj) 

Q: Sort the objects on the table into the compost bin and the recycle bin. _, _, objs, _ = robot.detect_objects() compost_bin_obj = None recycle_bin_obj = None for obj in objs: is_match_description = match_description(obj, "compost bin", ret_type="bool") if is_match_description: compost_bin_obj = obj break for obj in objs: is_match_description = match_description(obj, "recycle bin", ret_type="bool") if is_match_description: recycle_bin_obj = obj break if compost_bin_obj is None or recycle_bin_obj is None: robot.say("No compost bin or recycle bin found.") else: for obj in objs: if obj == compost_bin_obj or obj == recycle_bin_obj: continue is_compostable = is_compostable(obj, ret_type="bool") if is_compostable: robot.pick_place(obj, compost_bin_obj) else: robot.pick_place(obj, recycle_bin_obj) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0018-04.png)


**----- Start of picture text -----**<br>
(c) Novel Languages<br>Q: Prepare 西红柿炒蛋 (stir-fried tomato and eggs) in the pot.<br>ingredients = [" 西红柿 ", " 鸡蛋 "] ["tomato", "egg"]<br>_, _, objs, _ = robot.detect_objects()<br>pot_obj = None<br>for obj in objs:<br>is_match_description = match_description(obj, "cooking pot",<br>ret_type="bool")<br>if is_match_description:<br>pot_obj = obj<br>break<br>if pot_obj is None:<br>robot.say("No cooking pot found.")<br>else:<br>for ingredient in ingredients:<br>for obj in objs:<br>is_match_description = match_description(obj, ingredient,<br>ret_type="bool")<br>if is_match_description:<br>robot.pick_place(obj, pot_obj)<br>**----- End of picture text -----**<br>


## (d) Novel Tasks 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0018-06.png)


**----- Start of picture text -----**<br>
Q: My steak is too bland. Can you help?<br>robot.say("Sure, let me see what I can do.")<br>_, _, objs, _ = robot.detect_objects()<br>for obj in objs:<br>is_match_description = match_description(obj, "steak",<br>ret_type="bool")<br>if is_match_description:<br>steak_obj = obj<br>break<br>if steak_obj is None:<br>robot.say("No steak found.")<br>else:<br>robot.say("I’m going to season your steak with some salt and<br>pepper.")<br>robot.pick_place("salt", steak_obj)<br>robot.pick_place("pepper", steak_obj)<br>**----- End of picture text -----**<br>


_Figure A3._ The one-shot prompt as well as the model outputs for a few test instructions for the robotics tasks. When given a single example in the prompt (a), our method can generalize (b-d) to new objects, languages, and task domains. Red highlight indicates LM generated code being executed by the Python interpreter, and purple highlight indicates LM simulating the code execution. Gray text is for illustration purpose only, and not provided to our model. Note that code in the form of robot.<func_name> invokes robot APIs. 

18 

**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0019-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0019-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0019-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator_images/icml_2024_Chain_of_Code_Reasoning_with_a_Language_Model-Augmented_Code_Emulator.pdf-0019-04.png)


_Figure A4._ Robot trajectory visualization for task “sort the objects on the table into the compost bin and the recycle bin”. CoC first generates code to solve the problem, and then executes the code with Python if possible (e.g., robot APIs like detect_objects and pick_place), and with LMulator if not (e.g., commonsense QA like is_compostable). The robot successfully picks and places the Post-it note to the recycle bin and the orange peel to the compost bin. See the full code in Fig. A3. 

How many countries have I been to? I’ve been to Mumbai, London, Washington, Grand Canyon, Baltimore, Longsheng, Guilin, Beijing, Galapagos, Quito, Barcelona, Paris, Prague, Nice, Dehli, Agra, Rome, Florence, Amalfi, Athens, Míkonos, Málaga, Monaco, Berlin, Munich, Innsbruck, Bern, Milan, Lucerne, Gimmelwald (Schilthornbahn), St Moritz, St Petersburg, Helsinki, Amsterdam, Gda´nsk, Vancouver, Anchorage, Montreal, Belize, The Bahamas, Jamaica, Hawaii, Acadia National Park, Stockholm, Copenhagen, Dover, Lyon, Madrid, Toulouse, Santorini, Oslo, Kusadasi, Souda, Rhodes, Tallinn, Venice, Naples, Cape Town, Johannesburg, Addis Abeba, Nairobi, Seattle, San Francisco, Chicago, St Louis, Memphis, Chinle, Stanford, New York, Philadelphia, Boston, Miami, New Orleans, Walt Disney World Resort, Jacksonville, Las Vegas, Los Angeles, Portland, Salt Lake City, Tahoe City, Phoenix, Albuquerque, Cleveland, Charlottesville, Nags Head, Newfoundland and Labrador, Burlington, Wilmington, Myrtle Beach, St Lucia, Barbados, Banff, Haiti, Montego Bay, Sao Palo, Rio, Lima, Cusco, Cozumel, Amarillo, Yosemite National Park, Joshua Tree, Zion National Park, Bryce Canyon National Park, Grand Teton National Park, Yellowstone National Park, Glacier National Park, Mount Hood, Paso Robles, San Diego, Bend, North Cascades National Park, Olympic National Park Visitor Center, Jasper National Park, Sequoia National Park, Kings Canyon National Park, Shasta National Forest, Mount Saint Helens, Mount Rainier, Austin, Buenos Aires, El Calafate, El Chaltén, Fitz Roy, Torres del Paine National Park, Puerto Natales, Puerto Varas, Santiago, Marble Caves, Cerro Castillo, Coyhaique, Singapore, Casablanca, Marrakesh, Cairo, Jerusalem, Tokyo, Kyoto Prefecture, Taipei City, Taichung City, Krk, Naturpark Puez-Geisler, Ljubljana, Plitvice Lakes National Park, Fairbanks, Juneau, Dallas, Sydney, Cairns, Brisbane, Hook Island, Charleston, Panama City, Bangkok, Chiang Mai, Bengaluru, Denver, Indianapolis, Nashville, Blacksburg, Lisbon, Porto, Estes Park, Coeur d’Alene, Hood River, Denali, Sitka, Mexico City, Warsaw, Geneva, Auckland, Queenstown, Whitefish, Minneapolis, Sioux Falls, Bozeman, Missoula, Springfield, Skye, Edinburgh, Honolulu, Kauai, Haleakal¯a National Park, Wrangell-St. Elias National Park & Preserve, Atlanta, Tirana, Corfu, Siena. 

_Figure A5._ Full question used in Fig. 1 

19 

