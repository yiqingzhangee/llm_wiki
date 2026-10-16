Published as a conference paper at ICLR 2023 

## _Decomposed Prompting_ : A MODULAR APPROACH FOR SOLVING COMPLEX TASKS 

**Tushar Khot** _[♣]_ **, Harsh Trivedi** _[♥]_ **, Matthew Finlayson** _[♣]_ **, Yao Fu** _[♠∗]_ **, Kyle Richardson** _[♣]_ **, Peter Clark** _[♣]_ **, Ashish Sabharwal** _[♣]_ 

> _♣_ Allen Institute for AI _♥_ Stony Brook University _♠_ University of Edinburgh tushark@allenai.org, hjtrivedi@cs.stonybrook.edu, matthewf@allenai.org, yao.fu@ed.ac.uk, kyler@allenai.org, peterc@allenai.org, ashishs@allenai.org 

## ABSTRACT 

Few-shot prompting is a surprisingly powerful way to use Large Language Models (LLMs) to solve various tasks. However, this approach struggles as the task complexity increases or when the individual reasoning steps of the task themselves are hard to learn, especially when embedded in more complex tasks. To address this, we propose Decomposed Prompting, a new approach to solve complex tasks by decomposing them (via prompting) into simpler sub-tasks that can be delegated to a shared library of prompting-based LLMs dedicated to these sub-tasks. This modular structure allows each prompt to be optimized for its specific sub-task, further decomposed if necessary, and even easily replaced with more effective prompts, trained models, or symbolic functions if desired. 

We show that the flexibility and modularity of Decomposed Prompting allows it to outperform prior work on few-shot prompting using GPT-3. On symbolic reasoning tasks, we can further decompose sub-tasks that are hard for LLMs into even simpler solvable sub-tasks. When the complexity comes from the input length, we can recursively decompose the task into the same task but with smaller inputs. We also evaluate our approach on textual multi-step reasoning tasks: on long-context multi-hop QA, we can more effectively teach the sub-tasks via our separate sub-tasks prompts; and on open-domain multi-hop QA, we can easily incorporate a symbolic information retrieval module within our decomposition framework, leading to improved performance on both tasks.[1] 

## 1 INTRODUCTION 

Large Language Models (LLMs) such as GPT-3 (Brown et al., 2020) have been shown to solve various tasks given only a few examples as prompts, also referred to as in-context learning. These models can even perform more complex reasoning tasks when shown the sequence of simple reasoning steps needed to perform the complex task as a prompt (Wei et al., 2022; Nye et al., 2021). In essence, the sequence of reasoning steps, such as in Chains-of-Thought (CoT) prompting (Wei et al., 2022), demonstrates how to decompose the complex task as well as how each reasoning step should be performed. However, as tasks become more complex, few demonstrations of the complex task aren’t sufficient for current models to learn to perform all necessary reasoning steps. E.g., fewshot demonstrations of concatenating the _k_[th] letter of words in a string is insufficient for GPT-3 to learn to extract the _k_[th] letter, or learn to answer hard single-hop questions when only provided a few demonstrations of multi-hop questions. Additionally, it is unclear whether tasks such as document retrieval and integration, for knowledge-intensive tasks, can even be done by few-shot prompts. 

To address these limitations, we propose **Decomposed Prompting** (DECOMP), a new approach to solve complex tasks by instead decomposing them into simpler sub-tasks and delegating these to sub-task specific LLMs, with both the decomposer and the sub-task LLMs (henceforth, _sub-task handlers_ ) having their own few-shot prompts. Fig 1 illustrates our approach. The decomposer 

> _∗_ Work done during internship at Allen Institute for AI 

> 1 Datasets, Code and Prompts available at https://github.com/allenai/DecomP. 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0002-01.png)


**----- Start of picture text -----**<br>
A B C A C<br>A B C<br>Standard Chain-of-Thought Decomposer Prompt Sub-Task Handlers<br>Prompting Prompting<br>Decomposed Prompting<br>**----- End of picture text -----**<br>


Figure 1: While standard approaches only provide labeled examples (shown as a grey input box with green label box), Chain-of-Thought prompting also describes the reasoning steps to arrive at the answer for every example in the prompt. Decomposed Prompting, on the other hand, uses the decomposer prompt to only describe the procedure to solve the complex tasks using certain subtasks. Each sub-task, indicated here with A, B and C is handled by sub-task specific handlers which can vary from a standard prompt (sub-task A), a further decomposed prompt (sub-task B) or a symbolic function such as retrieval (sub-task C) 

prompt only describes a sequence of sub-tasks (A, B, and C) needed to solve the complex tasks, indicated with the dashed lines. Each sub-task is then delegated to the corresponding sub-task handler shown on the right. 

Using a software engineering analogy, the decomposer defines the top-level _program_ for the complex task using interfaces to simpler, sub-task functions. The sub-task handlers serve as modular, debuggable, and upgradable _implementations_ of these simpler functions, akin to a software library. If a particular sub-task handler, say the one for identifying the _k_[th] letter or retrieving a document, is not performing well enough, we can debug this handler in isolation, explore alternative prompts or implementations, and seamlessly plug the improved module back into the overall system, as a systematic way to try to improve performance on the complex end-task. 

This approach has several advantages over prior work (as also shown in the figure). The sub-task handlers can be shown a broader and richer set of examples (of the simpler task) than the specific ones needed for the complex task prompt (task A). If a sub-task is too complex, it can be further decomposed into simpler sub-tasks (task B). Similar to software libraries, these sub-task handlers can be shared across multiple tasks; e.g., here tasks A and C are reused in the model for task B. As noted above, a sub-task handler can be easily swapped with an improved implementation without any change to the rest of the system. Few-shot prompt based LLMs can be even replaced with a symbolic system for tasks more suited for non-neural methods; e.g., task C uses a symbolic retrieval system such as Elasticsearch that can handle very large-scale corpora. Lastly, we can even improve upon prior work by simply adding an _error-correcting_ sub-task handler as a post-processing step. 

To illustrate these advantages of DECOMP, we empirically evaluate it against prior work on eight challenging datasets using GPT3 models: (1) On a task of concatenating the _k_[th] letter, we show that our approach of factoring out each sub-task allows us to more effectively teach the sub-problem of extracting the _k_[th] letter(specifically, by decomposing it into even easier sub-tasks). (2) On a task of reversing a list, we show that DECOMP allows us to extend the capabilities of a weaker model and build a scale-invariant system by recursively decomposing the task into reversal of smaller and smaller lists. (3) On a task of long-context QA (Khot et al., 2022), our approach allows each subtask handler to accommodate more examples than feasible with CoT prompting leading to better QA performance. (4) On three multi-hop open-domain QA datasets (Yang et al., 2018; Ho et al., 2020; Trivedi et al., 2022), we can incorporate a symbolic retrieval (ElasticSearch) API as the handler for the retrieval sub-task leading to better results than CoT. (5) On two Math QA datasets (Cobbe et al., 2021; Roy & Roth, 2015), we can post-process CoT to easily fix frequent formatting errors, resulting in a surprisingly high improvement of 14-17 pts. 

## 2 RELATED WORK 

**Few-shot Prompts for Multi-Step Reasoning** Large-scale Language models (LLMs) have been shown to learn various NLP tasks given just few examples as prompts (Brown et al., 2020). Recently, they have also been successfully applied to various multi-step reasoning tasks by providing the intermediate reasoning steps, i.e. Chain-of-Thought (Wei et al., 2022; Chowdhery et al., 2022), needed to arrive at the answer. An alternate approach has been to compose multiple LLMs or LLMs with symbolic functions to perform multi-step reasoning (Jung et al., 2022; Creswell et al., 2023; 

2 

Published as a conference paper at ICLR 2023 

Press et al., 2022; Parisi et al., 2022; Gao et al., 2022; Schick et al., 2023, inter alia). We view these prior works as specialized systems with a pre-defined decomposition structure. 

The closest works to our approach are the ideas of least-to-most prompting (Zhou et al., 2023) and successive prompting (Dua et al., 2022) where one prompt/model is used to generate the subquestions needed to answer a complex question and a second prompt/model sequentially answers these sub-questions. In contrast, our approach allows for diverse decomposition structures including recursion and other non-linear decomposition structures. E.g., by definition, least-to-most asks questions from easiest to the hardest and requires an LLM to eventually answer the complete question (“most” in least-to-most) whereas we have no such restriction. Additionally, we iteratively generate new questions based on previous answers (similar to successive prompting) and can explicitly assign different prompts or symbolic systems to answer each sub-question. 

**Modular Approaches for Multi-Step Reasoning** Our work follows a long literature in NLP on neural modular modeling architectures (Andreas et al., 2016; Talmor & Berant, 2018; Min et al., 2019; Jiang & Bansal, 2019; Gupta et al., 2020; Perez et al., 2020; Khot et al., 2021; Levine et al., 2022) for question-answering and other tasks. We take particular inspiration from the _Text Modular Networks_ approach of Khot et al. (2021), whereby problem decomposition consists of a learned _next question_ generator trained to generate questions in the language of a collection of textual and symbolic agents. Best-first search strategy was used to explore the space of possible decompositions during inference. In contrast to this work, which largely centered around supervised training of the next-question generator _given existing agents_ , we leverage the power and recent successes of fewshot LLMs to build both the decomposer and the sub-task agents that best fit the ideal decomposition. This has the advantage of obviating the need for specialized supervised training data that may not always be available for all sub-tasks – a key bottleneck of this prior work. 

## 3 DECOMPOSED PROMPTING 

As with conventional _few-shot_ prompting, the goal is to teach an LLM to find an answer _A_ to a query _Q_ using a small set of _in-context_ examples _D_ = _{E_ 1 _, ..., E|D|}_ . The answer _A_ is obtained from the underlying distribution _p_ ( _A | Q, D, θ_ ) (Dohan et al., 2022). In the most basic few-shot setup, examples take the form _Ej_ = ( _Qj, Aj_ ). In the case of CoT-style prompting, the goal is to obtain answers by first generating a sequence or chain of intermediate reasoning steps or “thoughts” _T_ , and then deriving the final answer based on _T_ . To teach this ability, one uses more sophisticated in-context examples that take the form _Ej_ = ( _Qj,_ ( _Tj,_ 1 _, . . . , Tj,k_ ) _, Aj_ ). 

In DECOMP, the core is a _decomposer_ LLM that tries to solve a complex task by generating a **prompting program** _P_ for it. Each step of _P_ directs a simpler sub-query to a function in an auxiliary set of **sub-task functions** _F_ available to the system. Given a query _Q_ whose answer is _A_ , the program _P_ is a sequence of the form �( _f_ 1 _, Q_ 1 _, A_ 1) _, ...,_ ( _fk, Qk, Ak_ )� where _Ak_ is the final answer predicted by _P_ and _Qi_ is a sub-query directed to the sub-task function _fi ∈F_ . _P_ is executed by a high-level imperative **controller** , which passes the inputs and outputs between the decomposer and sub-task handler until a stopping condition in _P_ is met and the final output obtained. 

To teach the decomposer LLM in a few-shot prompting manner, we use in-context examples that take the form _Ej_ = �( _Qj,_ � _fj,_ 1 _, Qj,_ 1 _, Aj,_ 1) _, ...,_ ( _fj,kj , Qj,kj , Aj,kj_ )�� where _Aj,kj_ = _Aj_ is the final answer for _Qj_ and ( _Qj,_ 1 _, . . . , Qj,kj_ ) is a decomposition of _Qj_ . Each sub-task function _f_ , in turn, is operationalized via a sub-task handler as an in-context prompting LLM (e.g., a separate CoT-style prompt or a additional prompting program dedicated to that sub-task), or any other symbolic or learned function (e.g., a calculator or specialized supervised trained model). 

## 3.1 DECOMPOSED PROMPTS 

To illustrate this with an example, consider a multi-step task such as “Concatenate the first letter of every word in _str_ using a space”. We can solve this task by decomposing it into a sequence of three simple sub-tasks: 1) Collect the list of words in the _str_ ; 2) For each word, extract the third letter; 3) Concatenate the extracted letters using space as the separator. Fig. 2 shows an example decomposition prompt for this task. Much like a conventional structured program, the top-level decomp prompt provides an example program _Ej_ using three sub-task functions: _f_ 1 :split that 

3 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0004-01.png)


**----- Start of picture text -----**<br>
QC: Concatenate the first letter of every word in "Jack Q: What are the words in "Elon Musk Tesla"?<br>Ryan" using spaces  A: ["Elon", "Musk", "Tesla"]<br>Q1: [split] What are the words in "Jack Ryan"?  Q: What are the letters in "C++"?<br>#1: ["Jack". "Ryan"]  A: ["C", "+", "+"]<br>Q2: (foreach) [str_pos] What is the first letter of #1?  ... split<br>#2: ["J", "R"]<br>Q3: [merge] Concatenate #2 with spaces  Q: Concatenate ["n", "i", "e"]<br>#3: "J R"  A: "nie"<br>Q4: [EOQ]<br>Q: Concatenate ["n", "i", "c", "e"] using spaces<br>A: "n i c e"<br>... decomp ... merge<br>**----- End of picture text -----**<br>


Figure 2: Prompts for the decomposer and the split and merge sub-tasks used by the decomposer. The decomposer specifies the sequence of questions and corresponding sub-tasks (within square braces). The sub-task prompts can be written independent of the complex task examples and can even capture generalizations, e.g., letters in word (split) and no delimiter (merge). 

_splits words in an input string_ , _f_ 2 :str ~~p~~ os that _finds character positions in strings_ and _f_ 3 :merge that _concatenates characters_ . In this case, we operationalize each sub-task function as a separate in-context prompt (e.g., using a standard prompting approach for split and merge on the right side), each containing a set of in-context examples that are independent of the original complex task. 

In addition to the three functions described above, additional control structure is included, such as the symbolic function foreach, which iterates over arrays and references to previous answers such as #1. We note that such a helper function is not strictly necessary (e.g., we could directly generate “Q2’: What is the first letter of Jack?” and “Q3’: What is the first letter of Ryan?” instead of Q2 in the figure) and is added to reduce the manual effort needed to specify the decomposition and also reduce potential errors during decomposition. In our experiments we use two of the compositional operators defined by Khot et al. (2022) (see appendix for details), although it is capable of using all their operators (which also capture the QDMR operators from Wolfson et al. (2020)). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0004-05.png)


**----- Start of picture text -----**<br>
QC: Concatenate the second letter of every word in<br>"John Smith" using spaces Q: What are the words in "John Smith"? A:<br>["John", "Smith"]<br>Q1:  [split]  What are the words in "John Smith"?<br>#1: ["John", "Smith"] Q: What is the second letter in "John"? A:<br>"o"<br>Q2: (foreach) [str_pos] What is the second letter in #1? Q: What is the second letter in "Smith"? A:<br>#2: ["o", "m"] "m"<br>Q3:  [merge]  Concatenate #2 with spaces Q: Concatenate ["o, "m"] with spaces A:<br>#3: "o m" "o m"<br>Q4: [EOQ]<br>A: "o m"<br>decomp<br>decomp<br>decomp<br>decomp<br>split<br>str_pos<br>merge<br>**----- End of picture text -----**<br>


Figure 3: The inference procedure in DECOMP iteratively calls the decomposer prompt to generate the next question and sub-task at each step, given the current history of question and answers. The generated question is then routed to the assigned sub-task handler (with some handling of special operators, when needed). When the special end-of-questions [EOQ] marker is generated, the previous answer is returned as the final prediction. 

## 3.2 PROMPT EXECUTION AND INFERENCE 

Given a new question and a set of background in-context examples _D_ , the inference (i.e., the program construction and execution) process is illustrated in Fig. 3. The new complex question is fed to the decomposer prompt to get the first sub-question to be asked to the split prompt. With the help of our symbolic controller, the answer generated from this prompt is then appended to the decomposer prompt to get the second sub-question, _Q_ 2. Due to the foreach operator in the generated question, _Q_ 2 results in two questions (one for each word in #1) to be fed to the str ~~p~~ os prompt. The answers are combined into an array to get the answer #2. The entire decomposition history is used to generate _Q_ 3 and passed to the merge prompt to get the final answer. Since the task has been solved, the decomposition prompt produces the special end-of-sequence marker([EOQ]) and the last 

4 

Published as a conference paper at ICLR 2023 

answer is returned as the final answer. Formally, performing inference involves finding the best answer _A_ to a new query _Q_ , which in the simplest form involves computing the MAP answer using the LLMs predictive distribution for _A_ , i.e., _A_[ˆ] = arg max _A p_ ( _A | D, Q, θ_ ) (Dohan et al., 2022). For practicality, such computations are approximated using greedy search in our experiments. 

## 3.3 DECOMP CAPABILITIES 

**Hierarchical Decomposition** Certain sub-tasks, even when given many examples, are not solvable with few-shot prompting. E.g., we found identifying the _k_[th] letter of a string to be challenging for the GPT3 text-davinci-002 model. In such a scenario, we can decompose the sub-task prompt further, to first identify the letters and their position and then select the _k_[th] element of this array (see Fig. 4). We can also re-use existing sub-task prompts in our framework. E.g., the split prompt can be reused since it was developed for the general task of splitting strings.[2] 

Q: What is at position 1 in "[("Colorless", 1), ("green", 2), ("ideas", QC: What is the letter at position 1 of the word "Google"? 3), ("sleep", 4), ("furiously", 5)]"? QS: [split] What are the letters and their positions in "Google"? A: "Colorless" A: "[(G, 1), (o, 2), (o, 3), (g, 4), (l, 5), (e, 6)]" QS: [arr_pos] What is at position 1 in #1? A: "G" Q: What is at position 2 in "[(G, 1), (o, 2), (o, 3), (g, 4), (l, 5), (e, 6)]"? QS: [EOQ] A: "o" .... **str_pos** .... **arr_pos** 

Figure 4: Since identifying the _k[th]_ character is challenging for GPT3 davinci-002 model, we further decompose it into two simpler sub-tasks: split the word into its letters (using the shared sub-task split) and then return the _k[th]_ item of this list using the arr ~~p~~ os prompt. 

**Recursive Decomposition** Some problems can be naturally broken down into one or more smaller problems of the same form. Recursive algorithms such as merge sort use this idea to solve large problems efficiently, using a succinctly described method. We apply this same principle in DECOMP by allowing the decomposer prompt to recursively call itself, as shown in Fig. 5 for the task of list reversal. By using recursion, we are able to generalize any base prompting approach (CoT in this figure) to much longer lists by breaking the input into smaller and smaller lists till we reach a list length where the model is highly accurate. Such recursive approaches can not be described by current methods such as CoT and standard prompting. Least-to-most prompting (Zhou et al., 2023) also proposes a similar solution but differs in two key aspects (a) it has to identify all the subproblems in one-shot instead of our iterative top-down decomposition (b) it has to learn to identify the relevant answers from the previous solutions which we get for free from our decomposition. 

|**Base Case (using CoT)**|QC: Reverse the sequence "newspaper, glasses, laptop, bottle".<br>QS: [list_split] Split the sequence "...".||
|---|---|---|
|QC: Reverse the sequence "laptop, photo, clip".|A: "newspaper, glasses" and "laptop, bottle"||
|QS: [extract] First is laptop. Second is photo. Third is|QS: [<br>reverse] Reverse the sequence "newspaper, glasses"||
|clip. Now to reverse, change the order to: Third is|A: "glasses, newspaper"|**Recursive calls**|
|clip. Second is photo. First is laptop. So the answer|QS: [<br>reverse] Reverse the sequence "laptop, bottle"||
|is "clip, photo, laptop".|A: "bottle, laptop"||
|A: "laptop, photo, clip"|QS: [merge] Concatenate "bottle, laptop" and "glasses, newspaper" using a comma||
|QS: [EOQ]|A: "bottle, laptop, glasses, newspaper"||
||QS: [EOQ]|**reverse**|



Figure 5: Sample prompt for recursive decomposition for reversing lists. Each list is split into two halves and each half is reversed and concatenated in the reverse order. We can recursively split a list till we hit the base case (lists of length 3 here) where existing approaches such as CoT are accurate. 

**External API Calls** In certain cases, the sub-tasks may not be feasible to solve using only a LLM. E.g., retrieving knowledge from a KB or large corpus. Such sub-tasks, however, can be easily solved using existing systems such as retrieving documents using an Elasticsearch index or webpages using Google search (Lazaridou et al., 2022). Fig. 6 shows how DECOMP can easily use such a system to retrieve the relevant documents and answer a single-hop open-domain question. 

> 2Appendix G contains the complete split prompt which has examples for questions such as “Q: What are the letters and their positions in “Mathison”?”. 

5 

Published as a conference paper at ICLR 2023 

|**QC**: Which company manufactured Lost Gravity?<br>**QS**: [<br>retrieve<br>] Which company manufactured Lost Gravity?<br>**A**:    ["Lost Gravity (roller coaster)", "The Roller Coaster", ...]<br>**QS**: [<br>singlehop_rcqa<br>] Titles: ["Lost Gravity (roller coaster)", ... ].<br>Question: Which company manufactured Lost Gravity?<br>**A**:    {"titles": ["Lost Gravity (roller coaster)", ...] "answer": "Mack Rides"}<br>**QS**: [EOQ]<br>retrieve_odqa|**QC**: Which company manufactured Lost Gravity?<br>**QS**: [<br>retrieve<br>] Which company manufactured Lost Gravity?<br>**A**:    ["Lost Gravity (roller coaster)", "The Roller Coaster", ...]<br>**QS**: [<br>singlehop_rcqa<br>] Titles: ["Lost Gravity (roller coaster)", ... ].<br>Question: Which company manufactured Lost Gravity?<br>**A**:    {"titles": ["Lost Gravity (roller coaster)", ...] "answer": "Mack Rides"}<br>**QS**: [EOQ]<br>retrieve_odqa|retrieve|retrieve|
|---|---|---|---|
|||Wikipedia Title:Mack Rides<br>Mack Rides GmbH & Co ...<br>**Q:**Which company manufactured The<br>Lost Gravit?||
||retrieve_odqa|y<br>**A:**"Mack Rides"|singlehop_rcqa|



Figure 6: A Decomposed Prompt to answer open-domain questions using Elasticsearch-based retrieval. Full usage of this prompt for open-domain multihop questions is given in Fig. 11. 

## 4 CASE STUDIES 

We showcase DECOMP’s strengths through four tasks; two symbolic manipulation tasks similar to those investigated by Wei et al. (2022) and two existing textual multi-hop reasoning tasks. Unless specified, we use text-davinci-002 InstructGPT3 model (Ouyang et al., 2022) as the LLM and report the Exact Match (EM) numbers, following prior work. For order-independent list answers, we evaluate set equality as EM. We compare our approach to CoT rather than each specific decomposition structure used in prior work. See App. G for the complete prompts for all our tasks. 

## 4.1 _k_[th] LETTER CONCATENATION (HIERARCHICAL DECOMPOSITION) 

We compare DECOMP to CoT prompting for concatenating letters at the _k_[th] position. All prompts contain examples of concatenating letters in position 1, 4, and last position of strings with 3 words. We create three different prompts for all our baselines and present the average to account for variance due to the choice of examples following Perez et al. (2021). We use the decomp, split, str ~~p~~ os (further decomposed as shown in Fig. 4), and merge prompts for decomposition prompting. We adapt the CoT for last letter concatenation from prior work (Wei et al., 2022) for this task as shown below. In addition, we consider a _rolled out_ version of our decomposition prompts in terms of a CoT, i.e., we describe the entire decomposition process (identify words, split each word into letters, take _k_[th] letter and concatenate) as a single CoT. e.g, for the question “Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space.”, we use the CoT: 

|**Chain-Of-Thought**<br>The letter at position 4 of ”Herbert” is ”b”. The let-<br>ter at position 4 of ”Alexander” is ”x”. The letter<br>at position 4 of ”Simon” is ”o”. Concatenating ”b”,<br>”x”, ”o” using a space leads to ”b x o”. So, ”Herbert<br>Alexander Simon” outputs ”b x o”. ...|**Chain-Of-Thought (rolled out)**<br>The words in ”Herbert Alexander Simon” are ”Her-<br>bert”, ”Alexander”, and ”Simon”. The letters and<br>their positions in ”Herbert” are ”[(H, 1), (e, 2), (r,<br>3), (b, 4), (e, 5), (r, 6), (t, 7)]”. The letter at position<br>4 in this sequence is ”b”. _· · ·_ outputs ”b x o”. ...|
|---|---|



We similarly adapt the least-to-most prompt (Zhou et al., 2023) to include rollout. (see App. G). We compare these four prompting techniques on 4 datasets to evaluate generalization along 3 axes: (1) new letter position _k_ = 3;[3] (2) longer inputs, #words=4 and 5; (3) new delimiter ”;”. The words in the test examples come from a list of most popular first and last names.[4] All evaluation datasets have 100 examples. We present results on space as a delimiter averaged across three prompts in Fig. 7.[5] 

**DECOMP outperforms chain-of-thought and least-to-most prompting** , even when the prompt uses the same reasoning procedure as the rolled out decomposition. This shows that the separate prompts are more effective at teaching hard sub-tasks than a single CoT prompt. 

**DECOMP generalizes perfectly to longer sequences.** As the length of the input sequence increases, our approach continues to achieve close to 100% accuracy on this task.[6] The CoT-based approaches drop noticeably in their scores with longer input lengths, widening the performance gap. 

> 3Note that none of the sub-task prompts contain examples for this position. 

> 4forebears.io/earth/forenames and forebears.io/earth/surnames 

> 5We obtain similar results with semi-colon shown in Fig. 22 in the appendix. 

> 6Note that we report aggregate metrics for DECOMP too but the std. dev is zero here 

6 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0007-01.png)


**----- Start of picture text -----**<br>
CoT CoT w/ rollout L2M w/ rollout DecomP<br>100 98.0 96.0 97.0 CoT CoT w/ rollout DecomP<br>100<br>86<br>80 74.7<br>71.3 70.5 80 76<br>66.0<br>60 53.7 53.3 60 53 59 64<br>40 42<br>40<br>22.7<br>20<br>12.0 20 16<br>6.0<br>0 5.5 3.0 6.5 4.5 1<br>N=3(IID) N=4(OOD) N=5(OOD) 0<br>N=4(IID) N=6(OOD) N=8(OOD) N=10(OOD)<br>**----- End of picture text -----**<br>


Figure 7: EM Results on the _k[th]_ letter concatenation task (k=3) using space as delimiter with different number of words in the input. DECOMP outperforms and generalizes better than CoT as well as Least-to-most prompting. 

Figure 8: EM results on reversing sequences. Incorporating CoT in DECOMP greatly increases the ability of the model to generalize to new sequence lengths. 

## 4.2 LIST REVERSAL (RECURSIVE DECOMPOSITION) 

We use the task of reversing lists of words[7] to show how recursive DECOMP enables length generalization. We adapt the relevant CoT prompt from Wei et al. (2022), and integrate it in a decomposed prompt. As a control, we also compare to a CoT version w/ rollout of our decomposed prompt. All prompts contain the same 3 examples of reversing word sequences with 3-5 items. We evaluate all prompts for generalization to 4, 6, 8, and 10-item sequences. Here we use davinci-001 to show that DECOMP enables a weaker model approach davinci-002’s performance (which does solve this task). We use the strategy from Fig. 5 and provide our prompts in App. G. Fig. 8 shows the results of the prompting strategies on different input lengths. 

**DECOMP improves the length generalization of few-shot prompting.** While our base CoT prompt does not generalize at all to longer sequences, our approach can recursively decompose the problem and achieve better length generalization. Moreover, the CoT version of our decomposition strategy fails because the unrolled prompt becomes too long and convoluted without the ability to abstract away sub-modules. 

## 4.3 LONG-CONTEXT QUESTION ANSWERING 

We next evaluate on the CommaQA-E dataset (Khot et al., 2022) under the reading comprehension setting. The dataset consists of synthetically generated entities (e.g. Erowid award), facts (“Wetherality was an actor in the movie Dewbar.”) and multi-hop questions (e.g., “What awards have the actors of the Erowid winning movies received?”). Due to the presence of many distractors and, as a result, longer context, this dataset has been shown to be hard for standard LMs even when fine-tuned. 

|What awards have movies produced by people born in 1910 won?<br>QS: [<br>qa] Who were born in the year 1910?<br>A: ["Teeplemole", "Muntaril"]<br>QS: (foreach_merge) [<br>qa] For which movies was #1 the producer?<br>A: ["Featsaw", "Zalate", "Premercy"]<br>QS: (foreach_merge) [<br>qa] Which awards were given to #2?<br>A: ["Zorgion", "Chowwurst", "Hallowcock"]<br>QS: [EOQ]<br>...<br>**coarse**|What awards have movies produced by people born in 1910 won?<br>QS: [<br>simp_qa] Who were born in the year 1910?<br>A: ["Teeplemole", "Muntaril"]<br>QS: (foreach_merge) [<br>pos_qa] For which movies was #1 the producer?<br>A: ["Featsaw", "Zalate", "Premercy"]<br>QS: (foreach_merge) [<br>aw_qa] Which awards were given to #2?<br>A: ["Zorgion", "Chowwurst", "Hallowcock"]<br>QS: [EOQ]<br>...<br>**fine**|
|---|---|



Figure 9: Sample prompts used for the CommaQA dataset. On the left, the coarse-grained decomposition defines a single QA sub-task with all single-hop questions being delegated to a single sub-task handler. On the right, the fine-grained decomposition assigns questions to three different sub-tasks (see App. G for their prompts) depending on the question type. This allows us to provide more examples for each question type allowing the model to learn the sub-task more effectively. 

To fit these questions within GPT3’s context limit (2049 tokens), we generate a smaller version of the CommaQA-E dataset and of the compositional generalization split such that we can fit at least 

> 7We use the vocabulary from Wei et al. (2022): https://www.vocabulary.com/lists/189583 

7 

Published as a conference paper at ICLR 2023 

four examples in the context for CoT prompts. The CoT prompts describe the sequence of facts needed to arrive at the answer (see App. G for all the prompts). 

For DECOMP, we can separate the task of decomposition (independent of the context) from the sub-tasks of single-hop question answering. As shown in Fig. 9, we provide examples of the context-independent decomposition in the decomposer prompt and use the separate sub-task prompts to teach the QA skill over the given context. Additionally, we can choose the granularity of decomposition to trade off human effort for increased accuracy. For example, we could have single QA prompt to handle all the questions or create QA prompts for different classes of questions. In our experiments, each sub-task prompt contains 8 QA examples (2 questions/para). We evaluate three different prompts and report the average results in Fig. 10. 

We make three observations on CommaQA. **DECOMP is more accurate than CoT** irrespective of the granularity of decomposition or the evaluation split. **Finer grained decomposition can help im-** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0008-04.png)


**----- Start of picture text -----**<br>
CoT DecomP(coarse) DecomP(fine)<br>100<br>GPT3 (text-davinci-002)<br>80<br>64.2<br>59.7 59.7<br>60 55.0<br>42.0<br>40 33.8<br>20<br>0<br>IID Comp. Gen<br>**----- End of picture text -----**<br>


Figure 10: EM results on the CommaQA-E datasets. DECOMP always outperforms CoT, with fine-grained marginally out-performing coarse-grained decomposition. 

**prove task performance** by providing more examples for each class of questions, which in turn increases single-hop QA accuracy. **DECOMP generalizes to new compositions** such as the compositional generalization split of CommaQA, which tests models on unseen compositions of relations observed in the training set. While CoT has a drop in score, both decomposition-based approaches actually get a small bump (the subset of relations used in this split are easier for our QA models). 

## 4.4 OPEN-DOMAIN QUESTION ANSWERING 

Next, we demonstrate the ability of our approach to integrate external API calls on the task of open-domain multihop question answering. We evaluate our approach on three datasets: (1) 2WikiMultihopQA (Ho et al., 2020) (2) MuSiQue (Trivedi et al., 2022) (3) HotpotQA (Yang et al., 2018). We describe the open-domain versions of these datasets in more detail in App. A We use the Codex (code-davinci-002) model here since it can fit the much longer contexts needed. We also evaluate the impact of model scale on DECOMP by using models from the Flan-T5 family: Flan-T5-Large (0.7B), Flan-T5-XL (3B), and Flan-T5-XXL (11B).[8] 

|**QC**: In what country was Lost Gravity manufactured?<br>**QS**: [<br>retrieve_odqa] Which company manufactured Lost Gravity?<br>**A:**{"titles": ["Lost Gravity (roller coaster)" ... ], "answer": ["Mack Rides"]}<br>**QS**: [<br>retrieve_odqa] The company Mack Rides is from which country?<br>**A:**{"titles": ["Mack Rides", ...], "answer": ["Germany"]}<br>**QS**: [<br>multihop_rcqa] Titles: ["Lost Gravity (roller coaster)" ..., "Mack Rides" ...].<br>Question: In what country was Lost Gravity manufactured?<br>decomposer|**QC**: In what country was Lost Gravity manufactured?<br>**QS**: [<br>retrieve_odqa] Which company manufactured Lost Gravity?<br>**A:**{"titles": ["Lost Gravity (roller coaster)" ... ], "answer": ["Mack Rides"]}<br>**QS**: [<br>retrieve_odqa] The company Mack Rides is from which country?<br>**A:**{"titles": ["Mack Rides", ...], "answer": ["Germany"]}<br>**QS**: [<br>multihop_rcqa] Titles: ["Lost Gravity (roller coaster)" ..., "Mack Rides" ...].<br>Question: In what country was Lost Gravity manufactured?<br>decomposer|Wikipedia Title:Lost Gravity (roller ... )<br>Lost Gravity is a steel roller coaster ...<br>Wikipedia Title:Mack Rides<br>Mack Rides GmbH & Co ...<br>...<br>**Q:**In what country was Lost Gravity<br>manufactured?|Wikipedia Title:Lost Gravity (roller ... )<br>Lost Gravity is a steel roller coaster ...<br>Wikipedia Title:Mack Rides<br>Mack Rides GmbH & Co ...<br>...<br>**Q:**In what country was Lost Gravity<br>manufactured?|
|---|---|---|---|
||decomposer|**A**: "Germany"|multihop_rcqa|



Figure 11: The prompt used to answer open-domain multihop questions using Elasticsearch-based retrieval. The retrieve ~~o~~ dqa prompt is given in Fig. 6. 

Fig. 11 shows the decomposition prompt we use. The decomposer generates (singlehop) subquestions and delegates them to retrieve ~~o~~ dqa (described in Fig. 6). As we showed earlier, this module retrieves relevant documents then uses an RC model to answer. retrieve ~~o~~ dqa returns both the answer and the documents, allowing subsequent sub-questions to use the answers (e.g. “Mack Rides”) and the multihop ~~r~~ cqa model to use the documents. The final multihop ~~r~~ cqa model is prompted to produce the answer directly or using CoT given K paragraphs. 

> 8We still use GPT3-sized models for decomposition since only these models are reliably able to produce the required structured outputs. 

8 

Published as a conference paper at ICLR 2023 

We compare our approach against two baselines: **A. No Context (No-Ctxt),** A closed-book setting baseline where the model must rely only on its parametric knowledge. **B. NoDecomp Context (NoDecomp-Ctxt),** A simple retrieval baseline where we retrieve K paragraphs using the multi-hop question as the input and use that as context. For both NoDecomp-Ctxt and Decomp-Ctxt, K is selected by hyperparameter tuning (App. A). We manually annotate CoTs and decompositions for 20 training set questions, and sample 3 prompts of 15 questions each for all approaches. The detailed prompts are given in the Appendix G. We evaluate on 300 held-out dev questions in each dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0009-02.png)


**----- Start of picture text -----**<br>
No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA<br>80 80<br>code-davinci-002 Flan-T5-XXL<br>70 64.1 70 63.3<br>60 60 56.4<br>50 50.2 49.9 47.1 50 49.0 49.9<br>41.2<br>40 38.1 40<br>32.3<br>29.3<br>30 21.4 25.4 30 23.3 24.8<br>20 18.3 20 15.1<br>10 10<br>0 0<br>MuSiQue HotpotQA 2WikiMultihopQA MuSiQue HotpotQA 2WikiMultihopQA<br>**----- End of picture text -----**<br>


Figure 12: Answer F1[9] on three open-domain QA datasets using two base LMs: Codex (left) and Flan-T5-XXL (right) with direct prompting. Decomp-Ctxt models (ours) significantly outperforms the No-Ctxt models (no retrieval) in all settings and also outperforms our strong retrieval baseline (NoDecomp-Ctxt QA), with the exception of Codex on HotpotQA where it is comparable. See App. A.3 for results on smaller Flan-T5 models and CoT prompting. 

We present results on all three datasets with direct QA prompts in Fig. 12 with other results in App. A. The Decomp-Ctxt models performs significantly better than No-Ctxt models in all the settings showing that external knowledge can be leveraged to improve few-shot models on open-domain mulithop QA. Furthermore, we show that our Decomp-Ctxt models outperform the strong retrieval baseline (NoDecomp-Ctxt) in all settings except one (Codex with HotpotQA). Finally, we show that even with the much smaller Flan-T5-XXL model, Decomp-Ctxt outperforms all the baselines and can even achieve scores comparable to the Codex-only systems. 

## 4.5 ADDITIONAL RESULTS 

**Post-processing CoT for error correction** DECOMP also allows us to create a targeted sub-task handler to focus on the source of error in any system. For example, CoT for arithmetic reasoning often rely on patterns (answer is .*) to extract answers but the CoT does not always fit this pattern. Instead, we can assign the answer extraction to a better sub-task handler (GPT3) and reduce these types of errors. This results in a 17 pt improvement on MultiArith (78 _→_ 95) and 14 pt improvement on GSM8K (36 _→_ 50.6) compared to CoT prompting (details in App. B). 

While DECOMP outperforms the baselines in aggregate, we also see the **gains of DECOMP are consistent across prompt choices** (see App. D) **and decomposition schemes** (see App. E). 

## 5 CONCLUSION 

We proposed a new approach, Decomposed Prompting, to solve complex tasks using few-shot prompts, by decomposing them into a prompting program built out of simpler sub-tasks. Drawing inspiration from software libraries, our decomposer and shared sub-tasks are designed in a modular fashion: they use their own few-shot prompts, allowing one to independently optimize each prompt, decompose a sub-task further if necessary, or even seamlessly replace it with a symbolic system. We show that Decomposed Prompting outperforms prior work on four different tasks and generalization settings, establishing it as an effective few-shot paradigm for solving complex tasks. 

> 9Answer F1 is computed by treating prediction and ground truth answer as bags of tokens and computing their precision and recall (Rajpurkar et al., 2016). See HotpotQA (Yang et al., 2018) for details. 

9 

Published as a conference paper at ICLR 2023 

## ACKNOWLEDGEMENTS 

We thank members of the Aristo team at the Allen Institute for AI (AI2) for their constructive feedback and the reviewers for their invaluable suggestions. This work was supported in part by the National Science Foundation under grants IIS2007290. 

## REFERENCES 

Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In _CVPR_ , 2016. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. In _ICLR_ , 2023. URL https:// openreview.net/forum?id=3Pf3Wg6o-A4. 

- David Dohan, Winnie Xu, Aitor Lewkowycz, Jacob Austin, David Bieber, Raphael Gontijo Lopes, Yuhuai Wu, Henryk Michalewski, Rif A Saurous, Jascha Sohl-dickstein, et al. Language model cascades. _arXiv preprint arXiv:2207.10342_ , 2022. 

- Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. Successive prompting for decomposing complex questions. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 1251–1265, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL https://aclanthology.org/ 2022.emnlp-main.81. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. _ArXiv_ , abs/2211.10435, 2022. 

- Nitish Gupta, Kevin Lin, Dan Roth, Sameer Singh, and Matt Gardner. Neural module networks for reasoning over text. In _ICLR_ , 2020. 

- Xanh Ho, A. Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In _COLING_ , 2020. 

- Yichen Jiang and Mohit Bansal. Self-assembling modular networks for interpretable multi-hop reasoning. In _EMNLP_ , 2019. 

- Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, and Yejin Choi. Maieutic prompting: Logically consistent reasoning with recursive explanations. In _EMNLP_ , 2022. 

- Tushar Khot, Daniel Khashabi, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Text modular networks: Learning to decompose tasks in the language of existing models. In _NAACL_ , 2021. 

10 

Published as a conference paper at ICLR 2023 

Tushar Khot, Kyle Richardson, Daniel Khashabi, and Ashish Sabharwal. Hey AI, can you solve complex tasks by talking to agents? In _Findings of ACL_ , 2022. 

- Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. Internetaugmented language models through few-shot prompting for open-domain question answering. _ArXiv_ , abs/2203.05115, 2022. 

- Yoav Levine, Itay Dalmedigos, Ori Ram, Yoel Zeldes, Daniel Jannai, Dor Muhlgay, Yoni Osin, Opher Lieber, Barak Lenz, Shai Shalev-Shwartz, et al. Standing on the shoulders of giant frozen language models. _arXiv preprint arXiv:2204.10019_ , 2022. 

- Sewon Min, Victor Zhong, Luke Zettlemoyer, and Hannaneh Hajishirzi. Multi-hop reading comprehension through question decomposition and rescoring. In _ACL_ , 2019. 

- Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models. _ArXiv_ , abs/2112.00114, 2021. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. In _NeurIPS_ , 2022. 

- Aaron Parisi, Yao Zhao, and Noah Fiedel. TALM: Tool augmented language models. _arXiv preprint arXiv:2205.12255_ , 2022. 

- Ethan Perez, Patrick Lewis, Wen-tau Yih, Kyunghyun Cho, and Douwe Kiela. Unsupervised question decomposition for question answering. In _EMNLP_ , 2020. 

- Ethan Perez, Douwe Kiela, and Kyunghyun Cho. True few-shot learning with language models. In _NeurIPS_ , 2021. 

- Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. _arXiv preprint arXiv:2210.03350_ , 2022. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100, 000+ questions for machine comprehension of text. In _EMNLP_ , 2016. 

- Subhro Roy and Dan Roth. Solving general arithmetic word problems. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pp. 1743–1752, 2015. doi: 10.18653/v1/D15-1202. URL https://aclanthology.org/D15-1202. 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. _ArXiv_ , abs/2302.04761, 2023. 

- Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions. In _NAACL_ , 2018. 

- Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. _TACL_ , 2022. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _ICLR_ , 2023. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. 

- Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. Break it down: A question understanding benchmark. _TACL_ , 2020. 

11 

Published as a conference paper at ICLR 2023 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In _EMNLP_ , 2018. 

- Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models. In _ICLR_ , 2023. 

12 

Published as a conference paper at ICLR 2023 

## A OPEN DOMAIN QA DETAILS 

## A.1 RETRIEVAL CORPUSES FOR OPEN DOMAIN QA 

We use HotpotQA in the fullwiki setting where it comes with the associated Wikipedia corpus for open-domain QA. 2WikiMultihopQA and MuSiQue, however, are originally reading comprehension datasets. Questions in 2WikiMultihopQA and MuSiQue are associated with 10 and 20 paragraphs respectively. To turn these datasets into open-domain QA datasets, we create a corpora for each dataset by combining all the paragraphs in the train, dev and test questions. As a result we get a corpus size of 430,225 paragraphs for 2WikiMultihopQA and 139,416 for MuSiQue. 

## A.2 HYPERPARAMETER TUNING FOR OPEN DOMAIN QA 

We treat the number of paragraphs to retrieve ( _K_ ) in NoDecomp-Ctxt and Decomp-Ctxt models as a hyperparameter. We select it based on a grid search on a set of values to maximize performance on a held out set of 100 questions for each dataset. For NoDecomp-Ctxt, we search _K ∈{_ 6 _,_ 8 _,_ 10 _}_ for GPT3 models and _K ∈_ 2 _,_ 4 _,_ 6 _,_ 8 for Flan-T5-* models. For Decomp-Ctxt, we search _K ∈{_ 2 _,_ 4 _,_ 6 _}_ for GPT3 and Flan-T5-* models. Note that the ranges are different between GPT3 and Flan-T5-* as GPT3 can fit in more number of tokens. The ranges are different for NoDecomp-Ctxt and DecompCtxt as _K_ refers to number of paragraphs retrieved in each round of retrieval, and NoDecomp-Ctxt has only one step of retrieval whereas Decomp-Ctxt usually has multiple retrieval steps. 

## A.3 ADDITIONAL RESULTS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0013-07.png)


**----- Start of picture text -----**<br>
No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA<br>40<br>MuSiQue<br>35 29.7 30.9<br>30 26.8<br>25.3 24.4<br>25 21.7 22.0<br>19.0 19.6<br>20<br>15<br>10.6 10.3<br>10 8.4<br>5<br>0<br>Flan-T5-Large Flan-T5-XL Flan-T5-XXL code-davinci-002<br>(a) CoT QA Prompt<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0013-08.png)


**----- Start of picture text -----**<br>
No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA<br>40<br>MuSiQue<br>35<br>29.3<br>30<br>25.6 25.4<br>25 22.3 23.3 21.4<br>20 18.8 18.3<br>15.4 15.1<br>15 11.1<br>10 7.9<br>5<br>0<br>Flan-T5-Large Flan-T5-XL Flan-T5-XXL code-davinci-002<br>(b) Direct QA Prompt<br>**----- End of picture text -----**<br>


Figure 13: Results on MuSiQue dataset 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0013-10.png)


**----- Start of picture text -----**<br>
No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA<br>6050 HotpotQA 47.8 49.4 51.6 49.1 51.8 49.0 54.9 53.5 6050 HotpotQA 47.9 52.7 49.0 56.4 50.2 49.9<br>41.6 42.7 41.2<br>40 40 36.6<br>30 30 24.8<br>21.3 22.6<br>18.9 19.3<br>20 16.8 20<br>10 10<br>0 0<br>Flan-T5-Large Flan-T5-XL Flan-T5-XXL code-davinci-002 Flan-T5-Large Flan-T5-XL Flan-T5-XXL code-davinci-002<br>(a) CoT QA Prompt (b) Direct QA Prompt<br>**----- End of picture text -----**<br>


Figure 14: Results on HotpotQA dataset 

13 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0014-01.png)


**----- Start of picture text -----**<br>
No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA No-Ctxt QA NoDecomp-Ctxt QA Decomp-Ctxt QA<br>80 80<br>2WikiMultihopQA 70.8 2WikiMultihopQA<br>70 63.0 70 63.3 64.1<br>60 58.2 58.4 60 56.7<br>50 47.8 44.7 48.8 50 48.9 44.4 49.9 47.1<br>40 38.9 40.9 40 35.0 38.1<br>30.4 31.4 32.1 32.4 32.3<br>30 30 27.3<br>20 20<br>10 10<br>0 0<br>Flan-T5-Large Flan-T5-XL Flan-T5-XXL code-davinci-002 Flan-T5-Large Flan-T5-XL Flan-T5-XXL code-davinci-002<br>(a) CoT QA Prompt (b) Direct QA Prompt<br>**----- End of picture text -----**<br>


Figure 15: Results on 2WikiMultihopQA dataset 

## A.3.1 MUSIQUE 

We present all the results on the MuSiQue dataset in Fig. 13. Across all settings, we can see that retrieval helps substantially (large gains over No-Ctxt QA) with further improvements achieved by our DecomP-based Decomp-Ctxt QA model. 

## A.3.2 HOTPOTQA 

We present all the results on the HotpotQA dataset in Fig. 14. On this dataset too, we can see large gains by incorporating retrieval but the gains from using DecomP are mostly seen in the smaller models. 

## A.3.3 2WIKIMULTIHOPQA 

We present all the results on the 2WikiMultihopQA dataset in Fig. 15. On this dataset, we can see large gains by incorporating retrieval and also observe substantial gains by incorporating DecomP (as compared to NoDecomp-Ctxt). 

## B MATH QA 

We apply Decomposed Prompting to two math QA datasets: GSM8K Cobbe et al. (2021) and MultiArith Roy & Roth (2015). For Chain-of-thought, we used the original prompts for math reasoning Wei et al. (2022). For example: 

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. 

Most CoT systems Wei et al. (2022); Wang et al. (2023) rely on extracting the answer by finding the number following “answer is”. However, this may not always be accurate. For example, the following CoT would be unanswerable by relying on simple patterns. 

Parker chews 4 pieces of gum a day. There are 15 pieces of gum in a pack. So he will need 4 * 30 / 15 = 8 packs of gum to last him 30 days. 

Rather than relying on patterns with limited generalization, we can use a language model to extract the answer more reliably. Specifically, we use Decomposed Prompting to decompose the task into first identifying the chain-of-thought reasoning and then using a second GPT3-based sub-module to extract the answer from the CoT. We show examples of our prompts here (full prompt in App. G): 

## **Example from the Decomposition Prompt** 

14 

Published as a conference paper at ICLR 2023 

QC: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? QS: [cot] There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. 

QS: [gpt ~~a~~ ns] There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. A: 6 QS: [EOQ] 

## **Example from the gpt** ~~**a**~~ **ns prompt** 

Q: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. A: 6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0015-05.png)


**----- Start of picture text -----**<br>
CoT DecomP(coarse) DecomP(fine)<br>CoT DecomP<br>100<br>100 95.0<br>80<br>80 78.0<br>60 56<br>49<br>60<br>50.7 40 33<br>40 36.0<br>20 14 19<br>7 9 8 10<br>20<br>0<br>text-curie-001 text-davinci-001 text-davinci-002<br>0 GSM8K MultiArith Figure 17: As the models become weaker<br>**----- End of picture text -----**<br>


Figure 17: As the models become weaker (davinci-001) and smaller (curie-001), the performance of all the models drop. DECOMP still outperforms CoT till the performance reaches close to zero with curie. 

Figure 16: Our simple decomposition results in 14-17 pts on two MathQA datasets: GSM8k and MultiArith. 

We present our results in Fig. 16. On the GSM8K data set[10] , we outperform CoT by 14 points. On the MultiArith dataset[11] , we achieve a 17 pt improvement compare to CoT. While this is a simple change, it illustrates the possibility of using DECOMP for other complex answer types, e.g. nonextractive answer generation from chain-of-thoughts. 

## C EFFECT OF SCALE ON COMMAQA 

We evaluate text-curie-001, text-davinci-001 and text-davinci-002 on the CommAQA dataset. Since the curie-001 and davinci-001 have a smaller context window size, we further reduced our prompts to fit within their context windows (2048 tokens). As shown in Fig. 17, both CoT and DECOMP are effected by the model size. 

## D RESULTS ON ALL PROMPTS 

## D.1 PER-PROMPT RESULT ON LETTER CONCATENATION 

We present the results of the letter concatenation task (with space delimiter) for different values of N in Fig. 18. Our results are stable across the different prompts (P1, P2 and P3) and always outperform CoT and Least-to-Most prompting. 

> 10We randomly sample 300 examples from the test set due to costs with API usage 

> 11We randomly sample 200 examples from the test set due to costs with API usage 

15 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0016-01.png)


**----- Start of picture text -----**<br>
N=3<br>10 COT COT w/ rollout L2M w/ rollout Decomp N=4<br>0 COT COT w/ rollout L2M w/ rollout Decomp<br>100<br>75<br>75<br>50<br>50<br>25 25<br>0<br>0 P1 P2 P3<br>P1 P2 P3<br>N=5<br>COT COT w/ rollout L2M w/ rollout Decomp<br>100<br>75<br>50<br>25<br>0<br>P1 P2 P3<br>EM EM<br>EM<br>**----- End of picture text -----**<br>


Figure 18: Across all values of N and different prompts (P1, P2 and P3), DECOMP outperform chain-of-thought reasoning and even least-to-most prompting. 

## D.2 PER-PROMPT RESULTS ON COMMAQA 

We also present the results of all the prompts on the CommAQA dataset in Fig. 19. Here too, we can observe that DECOMP outperforms CoT on each prompt set. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0016-05.png)


**----- Start of picture text -----**<br>
COT Decomp (coarse) Decomp (fine) COT Decomp (coarse) Decomp (fine)<br>80 80<br>60 60<br>40 40<br>20 20<br>0 0<br>P1 P2 P3 P1 P2 P3<br>(a) Test Set (b) Comp. Gen. Set<br>EM EM<br>**----- End of picture text -----**<br>


Figure 19: Results of different prompts on the CommAQA dataset. 

## E EFFECT OF DECOMPOSITION SCHEME 

To evaluate the effect of the decomposition scheme, we experiment with two other simple decomposition structures for the letter concatenation and reversal tasks. 

**Letter Concatenation** For letter concatenation, we consider an alternate scheme where we use GPT3 to generate each question rather than loop over the answers, e.g., 

16 

Published as a conference paper at ICLR 2023 

QC: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. QS: [split] What are the words in ”Augusta Ada King”? 

A: [”Augusta”, ”Ada”, ”King”] QS: [str ~~p~~ osition] What is the last letter in ”Augusta”? A: ”a” QS: [str ~~p~~ osition] What is the last letter in ”Ada”? A: ”a” QS: [str ~~p~~ osition] What is the last letter in ”King”? A: ”g” QS: [merge] Concatenate [”a”, ”a”, ”g”] using a space. A: ”a a g” QS: [EOQ] 

By using the decomposer prompt model to generate the sub-questions, we can be more robust to formatting issues in the output answers, e.g., we can expect GPT3 to still generate the appropriate sub-questions even if the first answer is not a valid array. However, the generated sub-questions may not correctly use all the elements of the list (change in order, missed element, repeated elements, etc). 

**List Reversal** For list reversal, instead of splitting into halves, we take the tail of the list, reverse it and then concatenate it to the head. i.e. reverse(list) = reverse(list[1:]) + list[0]. This requires more GPT3 calls (O(n)) compared to the original approach of splitting the list into halves (O(log(n))). In both these cases, we noticed that the performance did not drop as shown in Fig. 20 and Fig. 21. On the letter concatenation task, the results were exactly the same. The new reversal decomposition schema was actually stronger on longer inputs at the cost of more calls to GPT3 (O(ln(n)) using binary splits vs O(n) one element at a time). Both these decomposition schemes are still better than CoT. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0017-05.png)


**----- Start of picture text -----**<br>
Decomp (Loop) Decomp (Generate)<br>100<br>75<br>50<br>25<br>0<br>N=3 N=4 N=5<br>**----- End of picture text -----**<br>


Figure 20: Both decomposition schemes for the letter concatenation task have the same scores. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0017-07.png)


**----- Start of picture text -----**<br>
Decomp (Mid Split) Decomp (Tail Split)<br>100<br>75<br>50<br>25<br>0<br>4 6 8 10<br>N<br>**----- End of picture text -----**<br>


Figure 21: Recursively reversing the tail of a list is more stable at longer lengths but comes at the cost of more calls to GPT3. 

## F ERROR ANALYSIS 

## F.1 LETTER CONCATENATION 

## F.1.1 DECOMP 

We analyzed the errors in DECOMP on the letter concatenation task and only found errors in the sub-task execution. 

17 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks_images/iclr_2023_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf-0018-01.png)


**----- Start of picture text -----**<br>
CoT CoT w/ rollout DecomP<br>100 97 94 98<br>82<br>80<br>70<br>60<br>51<br>40<br>31<br>20<br>12<br>7<br>0<br>N=3(IID) N=4(OOD) N=5(OOD)<br>**----- End of picture text -----**<br>


Figure 22: EM Results on the _k[th]_ letter concatenation task (k=3) using semi-colon as delimiter with different values for N, the number of words in the input. DECOMP always outperforms and generalizes better than CoT. 

Q: Take the letters at position 3 of the words in ”Nancy Samina Abbas Caudhari Bano” and concatenate them using a space. 

A: n m b u n Prediction: c m b u n Error: Incorrect letter extraction (Sub-task) _What is at position 3 in ”[(N, 1), (a, 2), (n, 3), (c, 4), (y, 5)]”? ⇒ ”c”_ 

Q: Take the letters at position 3 of the words in ”Orlando Stephen Cho Teixeira Pierre” and concatenate them using a space. A: l e o i e Prediction: leoie Error: Incorrect concatenation (Sub-task) _Concatenate [”l”, ”e”, ”o”, ”i”, ”e”] using a space. ⇒ ”leoie”_ 

## F.1.2 COT W/ ROLLOUT 

We analyzed the errors in CoT on the letter concatenation task and found similar errors during the generation of CoT. But the frequency of these errors was higher than DECOMP, as it is not possible to effectively teach each sub-task with CoT. 

Q: Take the letters at position 3 of the words in ”Sheila Nicolas Verma Sha Sousa” and concatenate them using a space. A: e c r a u Pred: i c r a u Error: Incorrect letter extraction _...The letters and their positions in ”Sheila” are ”[(S, 1), (h, 2), (e, 3), (i, 4), (l, 5), (a, 6)]”. The letter at position 3 in this sequence is ”i”..._ 

Q:Take the letters at position 3 of the words in ”Shobha Kailash Nakamura Peter Benitez” and concatenate them using a space. A: o i k t n Pred: o l k t i Error: Incorrect letter extraction _...”Benitez” are ”[(B, 1), (e, 2), (n, 3), (i, 4), (t, 5), (e, 6), (z, 7)]”. The letter at position 3 in this sequence is ”i”..._ 

18 

Published as a conference paper at ICLR 2023 

## F.2 COMMAQA 

Similarly in CommaQA, the errors are mostly due to sub-task errors, which in this dataset correspond to answering single-hop questions. CoT also makes the same types of errors but they are more frequent since this QA sub-task can not be delegated to a specialized prompt in CoT. Since all errors are of this type, we show only one example here. 

Q: What awards have movies written by people born in 1933 won? A: [”Hydrallium”, ”Pompasole”] Pred: [”Pompasole”] Error: Incorrect sub-question answer _Sub-Q:What movies has Haldron written? Sub-A: [”Polytetrafluoromethane”, ”Skia”, ”Warpstone”] Pred: [”Skia”, ”Warpstone”]_ 

## G TASK PROMPTS 

We have provided the task prompts for all the datasets for COT and our Decomposed Prompting approach. 

**CoT** Since CoT methods also perform 2-step reasoning: first generate the chain-of-thought and second extract the answer from the CoT, we use the same decomposition-based framework for COT baselines too. For example, consider the following example in our COT prompt: 

QC: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

QS: [extract] The letter at position 1 of ”Alan” is ”A”. The letter at position 1 of ”Mathison” is ”M”. The letter at position 1 of ”Turing” is ”T”. Concatenating ”A”, ”M”, ”T” using a space leads to ”A M T”. So, ”Alan Mathison Turing” outputs ”A M T”. A: ”A M T” QS: [EOQ] 

GPT3 generates the chain-of-thought during the ”decomposition” step and a regex-based answer extractor extract (’.* outputs "(.*)" _\_ .’) then takes this CoT and generates the answer. In some cases, the module name is skipped in the prompt (the CoT is sent to the extractor by default). 

**Operators** In this work, we use the same operators as defined by Khot et al.. Their select operator is just the basic operator that replaces references to an answer index with its answer. When not specified, select is assumed to be the default operator. In addition, we consider two operators in our experiments: project ~~v~~ alues and project ~~v~~ alues ~~f~~ lat ~~u~~ nique. 

   - project ~~v~~ alues: This operator takes a list answer # _i_ = _X_ and iterates over it to generate new questions by replacing mentions of # _i_ i.e. Q = [q.replace(#i, x) for x _∈_ X]. The answer to each question is simply concatenated to get the final answer i.e. A = [model(q) for q _∈_ Q]. We refer to this as foreach for simplicity in the main text. 

   - project ~~v~~ alues ~~f~~ lat ~~u~~ nique: This operator performs the same steps as project ~~v~~ alues but then additionally flattens the list and only returns the unique entities in the flattened list. We refer to this as foreach ~~m~~ erge in the main text for simplicity. 

- G.1 LETTER CONCATENTATION 

We show one of the prompts used for experiments here. The entire set of prompts is provided as supplemetary material. 

19 

Published as a conference paper at ICLR 2023 

## G.1.1 DECOMPOSED PROMPTING 

## decomp 

QC: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. QS: [split] What are the words in ”Augusta Ada King”? 

A: [”Augusta”, ”Ada”, ”King”] QS: (project ~~v~~ alues) [str ~~p~~ osition] What is the last letter in ”#1”? A: [”a”, ”a”, ”g”] QS: [merge] Concatenate #2 using a space. A: ”a a g” QS: [EOQ] 

QC: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

QS: [split] What are the words in ”Alan Mathison Turing”? 

A: [”Alan”, ”Mathison”, ”Turing”] QS: (project ~~v~~ alues) [str ~~p~~ osition] What is the letter at position 1 in ”#1”? A: [”A”, ”M”, ”T”] QS: [merge] Concatenate #2 using a space. A: ”A M T” QS: [EOQ] 

QC: Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. QS: [split] What are the words in ”Herbert Alexander Simon”? A: [”Herbert”, ”Alexander”, ”Simon”] QS: (project ~~v~~ alues) [str ~~p~~ osition] What is the letter at position 4 in ”#1”? A: [”b”, ”x”, ”o”] QS: [merge] Concatenate #2 using a space. A: ”b x o” QS: [EOQ] 

## split 

Q: What are the words in ”Alan Mathison Turing”? 

A: [”Alan”, ”Mathison”, ”Turing”] 

Q: What are the letters in ”Alan”? A: [”A”, ”l”, ”a”, ”n”] 

Q: What are the letters and their positions in ”Mathison”? A: ”[(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]” 

Q: What are the words and their positions in ”Herbert Alexander Simon”? A: ”[(Herbert, 1), (Alexander, 2), (Simon, 3)]” 

## str ~~p~~ osition 

QC: What is the letter at position 1 of the word ”Augusta”? QS: (select) [split] What are the letters and their positions in ”Augusta”? A: ”[(A, 1), (u, 2), (g, 3), (u, 4), (s, 5), (t, 6), (a, 7)]” QS: (select) [arr ~~p~~ osition] What is at position 1 in #1? A: ”A” QS: [EOQ] 

QC: What is the last letter of the word ”Mathison”? QS: (select) [split] What are the letters and their positions in ”Mathison”? A: ”[(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]” QS: (select) [arr ~~p~~ osition] What is the last letter in #1? A: ”n” 

20 

Published as a conference paper at ICLR 2023 

## QS: [EOQ] 

QC: What is the word at the position 4 in ”Colorless green ideas sleep furiously”? 

QS: (select) [split] What are the words and their positions in ”Colorless green ideas sleep furiously ”? 

A: ”[(Colorless, 1), (green, 2), (ideas, 3), (sleep, 4), (furiously, 5)]” 

QS: (select) [arr ~~p~~ osition] What is at the position 4 in #1? 

A: ”sleep” QS: [EOQ] 

## merge 

Q: Concatenate [”A”, ”l”, ”a”, ”n”]. 

A: ”Alan” 

Q: Concatenate [”b”, ”x”, ”o”] using a space. 

A: ”b x o” 

Q: Concatenate [”a”, ”a”, ”g”] using a comma. 

A: ”a,a,g” 

Q: Concatenate [”Alan”, ”Mathison”, ”Turing”] using a space. 

A: ”Alan Mathison Turing” 

Q: Concatenate [”Allen”, ”Institute”]. 

A: ”AllenInstitute” 

## arr ~~p~~ osition 

Q: What is at position 4 in ”[(”Colorless”, 1), (”green”, 2), (”ideas”, 3), (”sleep”, 4), (”furiously”, 5)]”? 

A: ”sleep” 

Q: What is at position 1 in ”[(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]”? A: ”M” 

Q: What is at the last position in ”[(A, 1), (u, 2), (g, 3), (u, 4), (s, 5), (t, 6), (a, 7)]”? A: ”a” 

Q: What is at position 1 in ”[(Herbert, 1), (Alexander, 2), (Simon, 3)]”? 

A: ”Herbert” 

Q: What is at last position in ”[(Allen, 1), (Institute, 2), (for, 3), (Artificial, 4), (Intelligence, 5)]”? A: ”Intelligence” 

Q: What is at position 4 in ”[(A, 1), (l, 2), (e, 3), (x, 4), (a, 5), (n, 6), (d, 7), (e, 8), (r, 9)]”? A: ”x” 

## G.1.2 COT WITH ROLLOUT 

COT w/ rollout 

- QC: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. QS: The words in ”Augusta Ada King” are ”Augusta”, ”Ada” and ”King”. The letters and their positions in ”Augusta” are ”[(A, 1), (u, 2), (g, 3), (u, 4), (s, 5), (t, 6), (a, 7)]”. The last letter in this sequence is ”a”. The letters and their positions in ”Ada” are ”[(A, 1), (d, 2), (a, 3)]”. The last letter in this sequence is ”a”. The letters and their positions in ”King” are ”[(K, 1), (i, 2), ( n, 3), (g, 4)]”. The last letter in this sequence is ”g”. Concatenating ”a”, ”a”, ”g” using a space leads to ”a a g”. So, ”Augusta Ada King” outputs ”a a g”. 

A: ”a a g” 

QS: [EOQ] 

21 

Published as a conference paper at ICLR 2023 

- QC: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

- QS: The words in ”Alan Mathison Turing” are ”Alan”, ”Mathison”, and ”Turing”. The letters and their positions in ”Alan” are ”[(A, 1), (l, 2), (a, 3), (n, 4)]”. The letter at position 1 in this sequence is ”A”. The letters and their positions in ”Mathison” are [(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]”. The letter at position 1 in this sequence is ”M”. The letters and their positions in ”Turing” are ”[(T, 1), (u, 2), (r, 3), (i, 4), (n, 5), (g, 6)]”. The letter at position 1 in this sequence is ”T”. Concatenating ”A”, ”M”, ”T” using a space leads to ”A M T”. So, ”Alan Mathison Turing” outputs ”A M T”. 

- A: ”A M T” 

QS: [EOQ] 

- QC: Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. 

- QS: The words in ”Herbert Alexander Simon” are ”Herbert”, ”Alexander”, and ”Simon”. The letters and their positions in ”Herbert” are ”[(H, 1), (e, 2), (r, 3), (b, 4), (e, 5), (r, 6), (t, 7)]”. The letter at position 4 in this sequence is ”b”. The letters and their positions in ”Alexander” are ”[(A, 1), (l, 2), (e, 3), (x, 4), (a, 5), (n, 6), (d, 7), (e, 8), (r, 9)]”. The letter at position 4 in this sequence is ”x”. The letters and their positions in ”Simon” are ”[(S, 1), (i, 2), (m, 3), (o, 4), (n, 5)]”. The letter at position 4 in this sequence is ”o”. Concatenating ”b”, ”x”, ”o” using a space leads to ”b x o”. So, ”Herbert Alexander Simon” outputs ”b x o”. 

- A: ”b x o” QS: [EOQ] 

## G.1.3 COT 

COT 

- QC: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. 

- QS: The last letter of ”Augusta” is ”a”. The last letter of ”Ada” is ”a”. The last letter of ”King” is ” g”. Concatenating ”a”, ”a”, ”g” using a space leads to ”a a g”. So, ”Augusta Ada King” outputs ”a a g”. 

- A: ”a a g” 

QS: [EOQ] 

- QC: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

- QS: The letter at position 1 of ”Alan” is ”A”. The letter at position 1 of ”Mathison” is ”M”. The letter at position 1 of ”Turing” is ”T”. Concatenating ”A”, ”M”, ”T” using a space leads to ”A M T”. So, ”Alan Mathison Turing” outputs ”A M T”. 

- A: ”A M T” 

- QS: [EOQ] 

- QC: Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. 

- QS: The letter at position 4 of ”Herbert” is ”b”. The letter at position 4 of ”Alexander” is ”x”. The letter at position 4 of ”Simon” is ”o”. Concatenating ”b”, ”x”, ”o” using a space leads to ”b x o”. So, ”Herbert Alexander Simon” outputs ”b x o”. 

- A: ”b x o” 

- QS: [EOQ] 

## G.1.4 LEAST-TO-MOST W/ ROLLOUT 

## Least-to-most Decomp 

- QC: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. QS: [l2m] Take the last letters of the words in ”Augusta Ada” and concatenate them using a space. 

- A: The words in ”Augusta Ada” are ”Augusta” and ”Ada”. The letters and their positions in ” Augusta” are ”[(A, 1), (u, 2), (g, 3), (u, 4), (s, 5), (t, 6), (a, 7)]”. The last letter in this sequence 

22 

Published as a conference paper at ICLR 2023 

   - is ”a”. The letters and their positions in ”Ada” are ”[(A, 1), (d, 2), (a, 3)]”. The last letter in 

   - this sequence is ”a”. Concatenating ”a”, ”a” using a space leads to ”a a”. So, ”Augusta Ada” outputs ”a a”. 

- QS: [l2m] Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. 

- A: ”Augusta Ada” outputs ”a a”. The letters and their positions in ”King” are ”[(K, 1), (i, 2), (n, 3), (g, 4)]”. The last letter in this sequence is ”g”. Concatenating ”a a”, ”g” using a space leads to ”a a g”. So, ”Augusta Ada King” outputs ”a a g”. 

- QS: [ans ~~e~~ xt] So, ”Augusta Ada King” outputs ”a a g”. 

- A: ”a a g” 

QS: [EOQ] 

- QC: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

- QS: [l2m] Take the letters at position 1 of the words in ”Alan Mathison” and concatenate them using a space. 

- A: The words in ”Alan Mathison” are ”Alan” and ”Mathison”. The letters and their positions in ” Alan” are ”[(A, 1), (l, 2), (a, 3), (n, 4)]”. The letter at position 1 in this sequence is ”A”. The letters and their positions in ”Mathison” are [(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]”. The letter at position 1 in this sequence is ”M”. Concatenating ”A”, ”M” using a space 

- leads to ”A M”. So, ”Alan Mathison” outputs ”A M”. 

- QS: [l2m] Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

- A: ”Alan Mathison” outputs ”A M”. The letters and their positions in ”Turing” are ”[(T, 1), (u, 2), (r, 3), (i, 4), (n, 5), (g, 6)]”. The letter at position 1 in this sequence is ”T”. Concatenating ”A M”, ”T” using a space leads to ”A M T”. So, ”Alan Mathison Turing” outputs ”A M T”. 

- QS: [ans ~~e~~ xt] So, ”Alan Mathison Turing” outputs ”A M T”. 

- A: ”A M T” 

QS: [EOQ] 

- QC: Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. 

- QS: [l2m] Take the letters at position 4 of the words in ”Herbert Alexander” and concatenate them using a space. 

- A: The words in ”Herbert Alexander” are ”Herbert” and ”Alexander”. The letters and their positions in ”Herbert” are ”[(H, 1), (e, 2), (r, 3), (b, 4), (e, 5), (r, 6), (t, 7)]”. The letter at position 4 in this sequence is ”b”. The letters and their positions in ”Alexander” are ”[(A, 1), ( l, 2), (e, 3), (x, 4), (a, 5), (n, 6), (d, 7), (e, 8), (r, 9)]”. The letter at position 4 in this sequence is ”x”. Concatenating ”b”, ”x” using a space leads to ”b x”. So, ”Herbert Alexander” outputs ” b x”. 

- QS: [l2m] Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. 

- A: ”Herbert Alexander” outputs ”b x”. The letters and their positions in ”Simon” are ”[(S, 1), (i, 2) , (m, 3), (o, 4), (n, 5)]”. The letter at position 4 in this sequence is ”o”. Concatenating ”b x”, ” o” using a space leads to ”b x o”. So, ”Herbert Alexander Simon” outputs ”b x o”. 

- QS: [ans ~~e~~ xt] So, ”Herbert Alexander Simon” outputs ”b x o”. 

A: ”b x o” 

QS: [EOQ] 

Least-to-most COT(l2m) 

- Q: Take the last letters of the words in ”Augusta Ada” and concatenate them using a space. 

- A: The words in ”Augusta Ada” are ”Augusta” and ”Ada”. The letters and their positions in ” Augusta” are ”[(A, 1), (u, 2), (g, 3), (u, 4), (s, 5), (t, 6), (a, 7)]”. The last letter in this sequence is ”a”. The letters and their positions in ”Ada” are ”[(A, 1), (d, 2), (a, 3)]”. The last letter in 

- this sequence is ”a”. Concatenating ”a”, ”a” using a space leads to ”a a”. So, ”Augusta Ada” outputs ”a a”. 

- Q: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. 

23 

Published as a conference paper at ICLR 2023 

- A: ”Augusta Ada” outputs ”a a”. The letters and their positions in ”King” are ”[(K, 1), (i, 2), (n, 3), (g, 4)]”. The last letter in this sequence is ”g”. Concatenating ”a a”, ”g” using a space leads to ”a a g”. So, ”Augusta Ada King” outputs ”a a g”. 

- Q: Take the letters at position 1 of the words in ”Alan Mathison” and concatenate them using a space. 

- A: The words in ”Alan Mathison” are ”Alan” and ”Mathison”. The letters and their positions in ” Alan” are ”[(A, 1), (l, 2), (a, 3), (n, 4)]”. The letter at position 1 in this sequence is ”A”. The letters and their positions in ”Mathison” are [(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]”. The letter at position 1 in this sequence is ”M”. Concatenating ”A”, ”M” using a space 

- leads to ”A M”. So, ”Alan Mathison” outputs ”A M”. 

- Q: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

- A: ”Alan Mathison” outputs ”A M”. The letters and their positions in ”Turing” are ”[(T, 1), (u, 2), (r, 3), (i, 4), (n, 5), (g, 6)]”. The letter at position 1 in this sequence is ”T”. Concatenating ”A M”, ”T” using a space leads to ”A M T”. So, ”Alan Mathison Turing” outputs ”A M T”. 

- Q: Take the letters at position 4 of the words in ”Herbert Alexander” and concatenate them using a space. 

- A: The words in ”Herbert Alexander” are ”Herbert” and ”Alexander”. The letters and their positions in ”Herbert” are ”[(H, 1), (e, 2), (r, 3), (b, 4), (e, 5), (r, 6), (t, 7)]”. The letter at position 4 in this sequence is ”b”. The letters and their positions in ”Alexander” are ”[(A, 1), ( l, 2), (e, 3), (x, 4), (a, 5), (n, 6), (d, 7), (e, 8), (r, 9)]”. The letter at position 4 in this sequence is ”x”. Concatenating ”b”, ”x” using a space leads to ”b x”. So, ”Herbert Alexander” outputs ” b x”. 

- Q: Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. 

- A: ”Herbert Alexander” outputs ”b x”. The letters and their positions in ”Simon” are ”[(S, 1), (i, 2) , (m, 3), (o, 4), (n, 5)]”. The letter at position 4 in this sequence is ”o”. Concatenating ”b x”, ” o” using a space leads to ”b x o”. So, ”Herbert Alexander Simon” outputs ”b x o”. 

- G.1.5 ALT DECOMP SCHEMA (GENERATE EACH SUB-QUESTION) 

decomp 

QC: Take the last letters of the words in ”Augusta Ada King” and concatenate them using a space. QS: [split] What are the words in ”Augusta Ada King”? 

A: [”Augusta”, ”Ada”, ”King”] 

QS: [str ~~p~~ osition] What is the last letter in ”Augusta”? A: ”a” 

QS: [str ~~p~~ osition] What is the last letter in ”Ada”? A: ”a” 

QS: [str ~~p~~ osition] What is the last letter in ”King”? 

A: ”g” 

QS: [merge] Concatenate [”a”, ”a”, ”g”] using a space. 

A: ”a a g” QS: [EOQ] 

- QC: Take the letters at position 1 of the words in ”Alan Mathison Turing” and concatenate them using a space. 

QS: [split] What are the words in ”Alan Mathison Turing”? 

A: [”Alan”, ”Mathison”, ”Turing”] 

QS: [str ~~p~~ osition] What is the letter at position 1 in ”Alan”? 

A: ”A” 

QS: [str ~~p~~ osition] What is the letter at position 1 in ”Mathison”? A: ”M” 

QS: [str ~~p~~ osition] What is the letter at position 1 in ”Turing”? A: ”T” 

24 

Published as a conference paper at ICLR 2023 

QS: [merge] Concatenate [”A”, ”M”, ”T”] using a space. A: ”A M T” QS: [EOQ] 

QC: Take the letters at position 4 of the words in ”Herbert Alexander Simon” and concatenate them using a space. 

QS: [split] What are the words in ”Herbert Alexander Simon”? 

A: [”Herbert”, ”Alexander”, ”Simon”] 

QS: [str ~~p~~ osition] What is the letter at position 4 in ”Herbert”? A: ”b” QS: [str ~~p~~ osition] What is the letter at position 4 in ”Alexander”? A: ”x” QS: [str ~~p~~ osition] What is the letter at position 4 in ”Simon”? A: ”o” 

QS: [merge] Concatenate [”b”, ”x”, ”o”] using a space. A: ”b x o” QS: [EOQ] 

G.2 SEQUENCE REVERSAL 

G.2.1 SPLIT REVERSAL 

The prompts in this section implement Algorithm 1. 

**Algorithm 1** A recursive reversal strategy that splits the sequence in half, reverses each half, and concatenates them. Runs in _O_ (log _n_ ) calls to the LM where _n_ is the number of items in the sequence. 

**procedure** SPLITREVERSE( _x_ ) _▷_ The reversal of _x_ **if** _|x| <_ 4 **then** _▷_ The base case **return** _x|x|, . . . , x_ 1 _▷_ The reversed sequence **else** _▷_ The inductive case _n ←|x|/_ 2 _▷_ Half the length of _x ℓ ← x_ 1 _, . . . , xn ▷_ The first half of _x ℓ[R] ←_ SPLITREVERSE( _ℓ_ ) _▷_ The reversed first half _r ← xn_ +1 _, . . . , x|x| ▷_ The second half of _x r[R] ←_ SPLITREVERSE( _r_ ) _▷_ The reversed second half **return** _r_ 1 _[R][, . . . , r] n[R][, ℓ][R] n_ +1 _[, . . . , ℓ][R] |x| ▷_ The concatenated reversed halves **end if end procedure** 

## reverse 

- QC: Reverse the sequence ”driving license, button, packet, identity card, shoe”. 

- QS: [extract] The sequence is ”1. driving license, 2. button, 3. packet, 4. identity card, 5. shoe”. The sequence is 5 items long, which is more than the minimum length of 4, so we split it. Half of 5 is 5 / 2 = 2.5. Dropping the decimal, we get that the first half will be 2 items long, ending in ”2. button”. The first half (2 items) is ”1. driving license, 2. button”. 

- A: ”1. driving license, 2. button” 

- QS: [extract] The first half of the sequence ends with ”2. button”, so the second half starts after ”2. button” with ”3. packet”. The full sequence is 5 items long, and the first half is 2 items long, so the second half will be 5 −2 = 3 items long. The second half of the sequence (3 items) is ”3. packet, 4. identity card, 5. shoe”. 

- A: ”3. packet, 4. identity card, 5. shoe” 

- QS: [remove ~~n~~ umbers] Remove the numbers from #1. 

- A: ”driving license, button” 

- QS: [remove ~~n~~ umbers] Remove the numbers from #2. 

A: ”packet, identity card, shoe” 

- QS: [reverse] Reverse the sequence #3. 

- A: ”button, driving license” 

25 

Published as a conference paper at ICLR 2023 

- QS: [reverse] Reverse the sequence #4. 

- A: ”shoe, identity card, packet” 

- QS: [join] #6 #5 

- A: ”shoe, identity card, packet, button, driving license” QS: [EOQ] 

- QC: Reverse the sequence ”laptop, photo, clip”. 

- QS: [extract] The sequence is ”1. laptop, 2. photo, 3. clip”. The sequence is 3 items long, which is less than the minimum length of 4, so we don’t need to split it. All we need to do is reverse ” laptop, photo, clip”. 

- A: ”laptop, photo, clip” 

- QS: [cot] Reverse the sequence #1. 

- A: ”clip, photo, laptop” QS: [EOQ] 

- QC: Reverse the sequence ”newspaper, glasses, laptop, bottle”. 

- QS: [extract] The sequence is ”1. newspaper, 2. glasses, 3. laptop, 4. bottle”. The sequence is 4 items long, which is equal to the minimum length of 4, so we split it. Half of 4 is 4 / 2 = 2.0. Dropping the decimal, we get that the first half will be 2 items long. The first half (2 items) of the sequence is ”1. newspaper, 2. glasses”. 

- A: ”1. newspaper, 2. glasses” 

- QS: [extract] The first half of the sequence ends with ”2. glasses”, so the second half starts after ”2. glasses” with ”3. laptop”. The full sequence is 4 items long and the first half is 2 items long, 

- so the second half will be 4 −2 = 2 items long, ending in ”2. glasses”. The second half of the sequence (2 items) is ”3. laptop, 4. bottle”. 

- A: ”3. laptop, 4. bottle” 

- QS: [remove ~~n~~ umbers] Remove the numbers from #1. 

A: ”newspaper, glasses” 

- QS: [remove ~~n~~ umbers] Remove the numbers from #2. 

A: ”laptop, bottle” 

QS: [reverse] Reverse the sequence #3. 

A: ”glasses, newspaper” QS: [reverse] Reverse the sequence #4. 

A: ”bottle, laptop” QS: [join] #6 #5 A: ”bottle, laptop, glasses, newspaper” QS: [EOQ] 

## remove ~~n~~ umbers 

Q: Remove the numbers from ”4. bottle, 3. laptop, 2. glasses, 1. newspaper”. A: ”bottle, laptop, glasses, newspaper” 

Q: Remove the numbers from ”1. identity card, 2. packet, 3. button”. 

A: ”identity card, packet, button” 

Q: Remove the numbers from ”1. player, 2. passport, 3. umbrella, 4. radio”. A: ”player, passport, umbrella, radio” 

## join 

Q: ”bottle, laptop” ”glasses, newspaper”. 

A: ”bottle, laptop, glasses, newspaper” 

Q: ”identity card, packet, button” ”magazine, notebook, glasses”. 

A: ”identity card, packet, button, magazine, notebook, glasses” 

Q: ”passport, umbrella, radio, mobile phone, photo” ”player”. 

A: ”passport, umbrella, radio, mobile phone, photo, player” 

26 

Published as a conference paper at ICLR 2023 

Q: ”mirror, case” ”toothbrush, alarm clock”. 

A: ”mirror, case, toothbrush, alarm clock” 

- Q: ”light bulb, clip, umbrella” ”driving licence, watch”. 

A: ”light bulb, clip, umbrella, driving licence, watch” 

## cot 

- QC: Reverse the sequence ”newspaper, glasses, laptop, bottle”. 

- QS: First is newspaper. Second is glasses. Third is laptop. Fourth is bottle. Now to reverse, change the order to: Fourth is bottle. Third is laptop. Second is glasses. First is newspaper. So the answer is ”bottle, laptop, glasses, newspaper”. 

A: ”bottle, laptop, glasses, newspaper” 

QC: Reverse the sequence ”laptop, photo, clip”. 

QS: First is laptop. Second is photo. Third is clip. Now to reverse, change the order to: Third is clip . Second is photo. First is laptop. So the answer is ”clip, photo, laptop”. A: ”clip, photo, laptop” 

QC: Reverse the sequence ”driving license, button, packet, identity card, pineapple”. 

- QS: First is driving license. Second is button. Third is packet. Fourth is identity card. Fifth is pineapple. Now to reverse, change the order to: Fifth is pineapple. Fourth is identity card. Third is packet. Second is button. First is driving license. So the answer is ”pineapple, identity card, packet, button, driving license”. 

- A: ”pineapple, identity card, packet, button, driving license” 

## unrolled ~~d~~ ecomp 

- QC: Reverse the sequence ”driving license, button, packet, identity card, shoe”. 

- QS: The sequence is ”1. driving license, 2. button, 3. packet, 4. identity card, 5. shoe”. The sequence is 5 items long, which is more than the minimum length of 4, so we split it. Half of 5 is 5 / 2 = 2.5. Dropping the decimal, we get that the first half will be 2 items long, ending in ”2. button”. The first half (2 items) is ”1. driving license, 2. button”. The first half of the sequence ends with ”2. button”, so the second half starts after ”2. button” with ”3. packet”. The full sequence is 5 items long, and the first half is 2 items long, so the second half will be 5 −2 = 3 items long. The second half of the sequence (3 items) is ”3. packet, 4. identity card, 5. shoe”. Removing the numbers from ”1. driving license, 2. button”, we get ”driving license, button”. Removing the numbers from ”3. packet, 4. identity card, 5. shoe”, we get ”packet, identity card, shoe”. Reversing the sequence ”driving license, button”, First is driving license. Second is button. Now to reverse, change the order to: Second is button. First is driving license. So the answer is ”button, driving license”. Reversing the sequence ”packet, identity card, shoe”, First is packet. Second is identity card. Third is shoe. Now to reverse, change the order to: Third is shoe. Second is identity card. First is packet. So the answer is ”shoe, identity card, packet”. Joining ”shoe, identity card, packet” and ”button, driving license”, the answer 

- is ”shoe, identity card, packet, button, driving license”. 

A: ”shoe, identity card, packet, button, driving license” 

QS: [EOQ] 

- QC: Reverse the sequence ”laptop, photo, clip”. 

- QS: The sequence is ”1. laptop, 2. photo, 3. clip”. The sequence is 3 items long, which is less than the minimum length of 4, so we don’t need to split it. All we need to do is reverse ”laptop, photo, clip”. First is laptop. Second is photo. Third is clip. Now to reverse, change the order to : Third is clip. Second is photo. First is laptop. So the answer is ”clip, photo, laptop”. 

A: ”clip, photo, laptop” QS: [EOQ] 

- QC: Reverse the sequence ”newspaper, glasses, laptop, bottle”. 

- QS: The sequence is ”1. newspaper, 2. glasses, 3. laptop, 4. bottle”. The sequence is 4 items long, which is equal to the minimum length of 4, so we split it. Half of 4 is 4 / 2 = 2.0. Dropping the decimal, we get that the first half will be 2 items long, ending in ”2. glasses”. The first half 

27 

Published as a conference paper at ICLR 2023 

(2 items) is ”1. newspaper, 2. glasses”. The first half of the sequence ends with ”2. glasses”, so the second half starts after ”2. glasses” with ”3. laptop”. The full sequence is 4 items long, and the first half is 2 items long, so the second half will be 4 −2 = 2 items long. The second half of the sequence (2 items) is ”3. laptop, 4. bottle”. Removing the numbers from ”1. newspaper, 2. glasses”, we get ”newspaper, glasses”. Removing the numbers from ”3. laptop, 4. bottle”, we get ”laptop, bottle”. Reversing the sequence ”newspaper, glasses”, First is newspaper. Second is glasses. Now to reverse, change the order to: Second is glasses. First is newspaper. So the answer is ”glasses, newspaper”. Reversing the sequence ”laptop, bottle”, First is laptop. Second is bottle. Now to reverse, change the order to: Second is bottle. First is laptop. So the answer is ”bottle, laptop”. Joining ”bottle, laptop” and ”glasses, newspaper”, the answer is ”bottle, laptop, glasses, newspaper”. 

A: ”bottle, laptop, glasses, newspaper” QS: [EOQ] 

reverse (tail) 

- QC: Reverse the sequence ”driving license, button, packet, identity card, shoe”. 

- Q1: [extract] The sequence is ”1. driving license, 2. button, 3. packet, 4. identity card, 5. shoe”. The sequence is 5 items long, which is more than the minimum length of 4, so we split it. We take the first element in the sequence which is ”1. driving license”. 

- #1: ”1. driving license” 

- Q2: [extract] The full sequence is 5 items long, so the remaining sequence will be 5 −1 = 4 items long. The tail of the sequence with 4 items is ”2. button, 3. packet, 4. identity card, 5. shoe”. 

#2: ”2. button, 3. packet, 4. identity card, 5. shoe” 

- Q3: [remove ~~n~~ umbers] Remove the numbers from #1. 

#3: ”driving license” 

- Q4: [remove ~~n~~ umbers] Remove the numbers from #2. 

#4: ”button, packet, identity card, shoe” 

Q5: [reverse] Reverse the sequence #4. 

#5: ”shoe, identity card, packet, button” 

Q6: [join] #5 #3 

#6: ”shoe, identity card, packet, button, driving license” Q7: [EOQ] 

QC: Reverse the sequence ”laptop, photo, clip”. 

- Q1: [extract] The sequence is ”1. laptop, 2. photo, 3. clip”. The sequence is 3 items long, which is less than the minimum length of 4, so we don’t need to split it. All we need to do is reverse ” laptop, photo, clip”. 

- #1: ”laptop, photo, clip” 

- Q2: [cot] Reverse the sequence #1. 

#2: ”clip, photo, laptop” 

- Q3: [EOQ] 

- QC: Reverse the sequence ”newspaper, glasses, laptop, bottle”. 

- Q1: [extract] The sequence is ”1. newspaper, 2. glasses, 3. laptop, 4. bottle”. The sequence is 4 items long, which is more than the minimum length of 4, so we split it. We take the first element in the sequence which is ”1. newspaper”. 

- #1: ”1. newspaper” 

- Q2: [extract] The full sequence is 4 items long, so the remaining sequence will be 4 −1 = 3 items long. The tail of the sequence with 3 items is ”2. glasses, 3. laptop, 4. bottle”. 

#2: ”2. glasses, 3. laptop, 4. bottle” 

Q3: [remove ~~n~~ umbers] Remove the numbers from #1. 

#3: ”newspaper” 

- Q4: [remove ~~n~~ umbers] Remove the numbers from #2. 

#4: ”glasses, laptop, bottle” 

Q5: [reverse] Reverse the sequence #4. 

#5: ”bottle, laptop, glasses” 

Q6: [join] #5 #3 

#6: ”bottle, laptop, glasses, newspaper” Q7: [EOQ] 

28 

Published as a conference paper at ICLR 2023 

## G.3 LONG-DOCUMENT QA 

We show one of the prompts used for CommaQA experiments here. The entire set of prompts is provided as supplemetary material. 

## G.3.1 DECOMPOSED PROMPTING: (COARSE) 

decomp 

What awards have movies produced by people born in 1910 won? QS: (select) [qa] Who were born in the year 1910? A: [”Teeplemole”, ”Muntaril”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] For which movies was #1 the producer? A: [”Featsaw”, ”Zalate”, ”Premercy”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Which awards were given to #2? A: [”Zorgion”, ”Chowwurst”, ”Hallowcock”] QS: [EOQ] 

QC: What movies have people from the country Stridery acted in? QS: (select) [qa] Who is from the country Stridery? A: [”Gastrat”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Which movies has #1 been an actor in? A: [”Partnershipmaker”, ”Nilitude”, ”Warpstone”] QS: [EOQ] 

QC: What awards have the actors of the Erowid winning movies received? QS: (select) [qa] Which movies were given the Erowid award? A: [”Dewbar”, ”Caudacite”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Who are the actors in the movie #1? A: [”Wetherality”, ”Lougeri`ere”, ”Gigabut”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Which awards were given to #2? A: [”Aniconder”, ”Trifogation”] QS: [EOQ] 

QC: What awards did the movies directed by the Modiparity winners receive? QS: (select) [qa] Who has been awarded the Modiparity award? A: [”Bioperatology”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Which movies has #1 directed? A: [”Pestok”, ”Vitrilateral”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Which awards were given to #2? A: [”Gutney”, ”Antipositive”] QS: [EOQ] 

QC: What awards have movies written by people born in 1935 won? QS: (select) [qa] Who were born in the year 1935? A: [”Sclerocybin”, ”Zayage”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] What movies has #1 written? A: [”Noenometer”, ”Tayenne”, ”Pneumodendron”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] Which awards were given to #2? A: [”Brownbeard”, ”Goosehead”, ”Handt”] QS: [EOQ] 

QC: What movies have the directors from Legault directed? QS: (select) [qa] Who is from the country Legault? A: [”Metatoun”, ”Sapien”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [qa] What movies has #1 been the director of? A: [”Coacheship”, ”Misapportionment”] QS: [EOQ] 

29 

Published as a conference paper at ICLR 2023 

## qa 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw 

; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: For which movies was Teeplemole the producer? 

A: [”Featsaw”, ”Zalate”] 

Q: Which awards were given to Featsaw? A: [”Zorgion”] 

movie: Misgendery ; directed by: Wetherality. movie: Dewbar ; director: Gigabut. movie: Caudacite ; director: Lougeri`ere. movie: Tayenne ; directed by: Lougeri`ere. movie: 

Misgendery ; awarded: Microsouenesis. movie: Dewbar ; awarded: Erowid. movie: Tayenne ; awarded: Cockspit. movie: Caudacite ; award: Erowid. award: Aniconder ; winner: Wetherality. award: Aniconder ; winner: Lougeri`ere. person: Gigabut ; award: Trifogation. movie: Dewbar ; release year: 1991. movie: Tayenne ; year: 2013. movie: Caudacite ; release year: 2008. movie: Misgendery ; year: 1991. Wetherality was an actor in the movie Dewbar. Gigabut was an actor in the movie Tayenne. Lougeri`ere was an actor in the movie Tayenne. Lougeri`ere acted in the movie Caudacite. Lougeri`ere acted in the movie Misgendery. Gigabut was an actor in the movie Caudacite. Wetherality was an actor in the movie Misgendery. Wetherality was born in the year 1917. Lougeri`ere was born in 1926. Gigabut was born in the year 1917. Gigabut grew up in the nation of Triclops. Lougeri`ere is from the country of Tatkin . Wetherality grew up in the nation of Tatkin. Lougeri`ere produced the movie Dewbar with others. Gigabut produced the movie Tayenne with others. Gigabut produced the movie Dewbar with others. Lougeri`ere was one of the producers of the movie Misgendery. Wetherality was one of the producers of the movie Caudacite. Gigabut was one of the producers of the movie Caudacite. Wetherality produced the movie Misgendery with others. Wetherality produced the movie Tayenne with others. Wetherality wrote for the movie Tayenne. Gigabut wrote for the movie Misgendery. Lougeri`ere was one of the writers for the movie Caudacite. Wetherality wrote for the movie Misgendery. Gigabut wrote for the movie Tayenne. Gigabut wrote for the movie Dewbar. Lougeri`ere wrote for the movie Dewbar. Wetherality wrote for the movie Caudacite. 

Q: Who are the actors in the movie Caudacite? 

A: [”Lougeri`ere, ”Gigabut”] 

Q: Which movies were given the Erowid award? 

A: [”Dewbar”, ”Caudacite”] 

movie: Pastillobox ; directed by: Firmline. movie: Clenestration ; directed by: Carblock. movie: Pestok ; directed by: Bioperatology. movie: Vitrilateral ; director: Bioperatology. movie: Vitrilateral ; award: Antipositive. movie: Clenestration ; awarded: Handt. movie: Pastillobox ; awarded: Handt. movie: Pestok ; awarded: Gutney. movie: Pestok ; writer: Firmline. movie: 

30 

Published as a conference paper at ICLR 2023 

Clenestration ; written by: Carblock. movie: Pastillobox ; written by: Bioperatology. movie: Pestok ; writer: Bioperatology. movie: Clenestration ; written by: Firmline. movie: Vitrilateral ; writer: Bioperatology. movie: Pastillobox ; writer: Carblock. movie: Vitrilateral ; written by : Carblock. movie: Pestok ; release year: 1986. movie: Clenestration ; year: 1986. movie: Vitrilateral ; year: 1999. movie: Pastillobox ; release year: 1984. Carblock was an actor in the movie Pastillobox. Firmline was an actor in the movie Vitrilateral. Bioperatology was an actor in the movie Clenestration. Firmline acted in the movie Pastillobox. Carblock was an actor in the movie Clenestration. Bioperatology was an actor in the movie Pestok. Firmline was born in the year 1904. Bioperatology was born in the year 1935. Carblock was born in 1935. Carblock grew up in the nation of Knoppock. Firmline grew up in the nation of Tatkin. Bioperatology grew up in the nation of Tatkin. Bioperatology won the Modiparity award. Halfbill was awarded to Firmline. Halfbill was awarded to Carblock. Bioperatology was one of the producers of the movie Pestok. Bioperatology produced the movie Vitrilateral with others. Firmline produced the movie Pastillobox with others. Firmline produced the movie Clenestration with others. Carblock was one of the producers of the movie Pastillobox. Carblock produced the movie Vitrilateral with others. Carblock produced the movie Clenestration with others. Firmline was one of the producers of the movie Pestok. Q: Who has been awarded the Modiparity award? A: [”Bioperatology”] 

Q: Which movies has Bioperatology directed? A: [”Pestok”, ”Vitrilateral”] 

movie: Nohit ; director: Mimicocycle. movie: Noenometer ; director: Mimicocycle. movie: Tayenne ; directed by: Zayage. movie: Pneumodendron ; director: Sclerocybin. movie: Tayenne ; awarded: Goosehead. movie: Nohit ; awarded: Handt. movie: Pneumodendron ; award: Handt. movie: Noenometer ; awarded: Brownbeard. movie: Nohit ; writer: Mimicocycle. movie: Noenometer ; written by: Sclerocybin. movie: Tayenne ; writer: Sclerocybin. movie: Pneumodendron ; written by: Zayage. movie: Tayenne ; writer: Zayage. movie: Pneumodendron ; written by: Mimicocycle. movie: Noenometer ; release year: 1991. movie: Tayenne ; year: 2013. movie: Nohit ; year: 2005. movie: Pneumodendron ; year: 2005. Mimicocycle was an actor in the movie Tayenne. Zayage acted in the movie Pneumodendron. Zayage was an actor in the movie Nohit. Sclerocybin was an actor in the movie Nohit. Sclerocybin was an actor in the movie Tayenne. Mimicocycle was an actor in the movie Noenometer. Zayage was born in 1935. Sclerocybin was born in 1935. Mimicocycle was born in 1930. Mimicocycle is from the country of Calderita. Sclerocybin grew up in the nation of Calderita. Zayage is from the country of Obility. Quinion was awarded to Zayage. Fannyxist was awarded to Sclerocybin. Fannyxist was awarded to Mimicocycle. Mimicocycle produced the movie Nohit with others. Zayage was one of the producers of the movie Nohit. Sclerocybin was one of the producers of the movie Tayenne. Sclerocybin produced the movie Pneumodendron with others. Zayage produced the movie Pneumodendron with others. Mimicocycle was one of the producers of the movie Tayenne. Sclerocybin was one of the producers of the movie Noenometer. Zayage produced the movie Noenometer with others Q: What movies has Sclerocybin written? 

A: [”Noenometer”, ”Tayenne”] 

Q: Who were born in the year 1935? 

A: [”Sclerocybin”, ”Zayage”] 

G.3.2 DECOMPOSED PROMPTING: (FINE) 

decomp 

What awards have movies produced by people born in 1910 won? QS: (select) [simp ~~q~~ a] Who were born in the year 1910? A: [”Teeplemole”, ”Muntaril”] 

QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [pos ~~q~~ a] For which movies was #1 the producer? A: [”Featsaw”, ”Zalate”, ”Premercy”] 

QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [aw ~~q~~ a] Which awards were given to #2? A: [”Zorgion”, ”Chowwurst”, ”Hallowcock”] 

31 

Published as a conference paper at ICLR 2023 

## QS: [EOQ] 

QC: What movies have people from the country Stridery acted in? QS: (select) [simp ~~q~~ a] Who is from the country Stridery? A: [”Gastrat”] 

QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [pos ~~q~~ a] Which movies has #1 been an actor in? A: [”Partnershipmaker”, ”Nilitude”, ”Warpstone”] QS: [EOQ] 

QC: What awards have the actors of the Erowid winning movies received? QS: (select) [aw ~~q~~ a] Which movies were given the Erowid award? A: [”Dewbar”, ”Caudacite”] 

QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [pos ~~q~~ a] Who are the actors in the movie #1? A: [”Wetherality”, ”Lougeri`ere”, ”Gigabut”] 

QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [aw ~~q~~ a] Which awards were given to #2? A: [”Aniconder”, ”Trifogation”] QS: [EOQ] 

QC: What awards did the movies directed by the Modiparity winners receive? QS: (select) [aw ~~q~~ a] Who has been awarded the Modiparity award? A: [”Bioperatology”] 

QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [pos ~~q~~ a] Which movies has #1 directed? A: [”Pestok”, ”Vitrilateral”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [aw ~~q~~ a] Which awards were given to #2? A: [”Gutney”, ”Antipositive”] QS: [EOQ] 

QC: What awards have movies written by people born in 1935 won? QS: (select) [simp ~~q~~ a] Who were born in the year 1935? A: [”Sclerocybin”, ”Zayage”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [pos ~~q~~ a] What movies has #1 written? A: [”Noenometer”, ”Tayenne”, ”Pneumodendron”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [aw ~~q~~ a] Which awards were given to #2? A: [”Brownbeard”, ”Goosehead”, ”Handt”] QS: [EOQ] 

QC: What movies have the directors from Legault directed? QS: (select) [simp ~~q~~ a] Who is from the country Legault? A: [”Metatoun”, ”Sapien”] QS: (project ~~v~~ alues ~~f~~ lat ~~u~~ nique) [pos ~~q~~ a] What movies has #1 been the director of? A: [”Coacheship”, ”Misapportionment”] QS: [EOQ] 

## aw ~~q~~ a 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw ; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with 

32 

Published as a conference paper at ICLR 2023 

others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: Which awards were given to Zalate? 

A: [”Hallowcock”] 

Q: Which awards were given to Featsaw? A: [”Zorgion”] 

movie: Misgendery ; directed by: Wetherality. movie: Dewbar ; director: Gigabut. movie: Caudacite ; director: Lougeri`ere. movie: Tayenne ; directed by: Lougeri`ere. movie: 

Misgendery ; awarded: Microsouenesis. movie: Dewbar ; awarded: Erowid. movie: Tayenne ; awarded: Cockspit. movie: Caudacite ; award: Erowid. award: Aniconder ; winner: Wetherality. award: Aniconder ; winner: Lougeri`ere. person: Gigabut ; award: Trifogation. movie: Dewbar ; release year: 1991. movie: Tayenne ; year: 2013. movie: Caudacite ; release year: 2008. movie: Misgendery ; year: 1991. Wetherality was an actor in the movie Dewbar. Gigabut was an actor in the movie Tayenne. Lougeri`ere was an actor in the movie Tayenne. Lougeri`ere acted in the movie Caudacite. Lougeri`ere acted in the movie Misgendery. Gigabut was an actor in the movie Caudacite. Wetherality was an actor in the movie Misgendery. Wetherality was born in the year 1917. Lougeri`ere was born in 1926. Gigabut was born in the year 1917. Gigabut grew up in the nation of Triclops. Lougeri`ere is from the country of Tatkin . Wetherality grew up in the nation of Tatkin. Lougeri`ere produced the movie Dewbar with others. Gigabut produced the movie Tayenne with others. Gigabut produced the movie Dewbar with others. Lougeri`ere was one of the producers of the movie Misgendery. Wetherality was one of the producers of the movie Caudacite. Gigabut was one of the producers of the movie Caudacite. Wetherality produced the movie Misgendery with others. Wetherality produced the movie Tayenne with others. Wetherality wrote for the movie Tayenne. Gigabut wrote for the movie Misgendery. Lougeri`ere was one of the writers for the movie Caudacite. Wetherality wrote for the movie Misgendery. Gigabut wrote for the movie Tayenne. Gigabut wrote for the movie Dewbar. Lougeri`ere wrote for the movie Dewbar. Wetherality wrote for the movie Caudacite. 

Q: Which movies were given the Erowid award? 

A: [”Dewbar”, ”Caudacite”] 

Q: Which awards were given to Wetherality? A: [”Aniconder”] 

movie: Pastillobox ; directed by: Firmline. movie: Clenestration ; directed by: Carblock. movie: Pestok ; directed by: Bioperatology. movie: Vitrilateral ; director: Bioperatology. movie: Vitrilateral ; award: Antipositive. movie: Clenestration ; awarded: Handt. movie: Pastillobox ; awarded: Handt. movie: Pestok ; awarded: Gutney. movie: Pestok ; writer: Firmline. movie: Clenestration ; written by: Carblock. movie: Pastillobox ; written by: Bioperatology. movie: Pestok ; writer: Bioperatology. movie: Clenestration ; written by: Firmline. movie: Vitrilateral ; writer: Bioperatology. movie: Pastillobox ; writer: Carblock. movie: Vitrilateral ; written by : Carblock. movie: Pestok ; release year: 1986. movie: Clenestration ; year: 1986. movie: Vitrilateral ; year: 1999. movie: Pastillobox ; release year: 1984. Carblock was an actor in the movie Pastillobox. Firmline was an actor in the movie Vitrilateral. Bioperatology was an actor in the movie Clenestration. Firmline acted in the movie Pastillobox. Carblock was an actor in the movie Clenestration. Bioperatology was an actor in the movie Pestok. Firmline was born in the year 1904. Bioperatology was born in the year 1935. Carblock was born in 1935. Carblock grew up in the nation of Knoppock. Firmline grew up in the nation of Tatkin. Bioperatology grew up in the nation of Tatkin. Bioperatology won the Modiparity award. Halfbill was awarded to Firmline. Halfbill was awarded to Carblock. Bioperatology was one of the producers of the movie Pestok. Bioperatology produced the movie Vitrilateral with others. Firmline produced the movie Pastillobox with others. Firmline produced the movie Clenestration with others. Carblock was one of the producers of the movie Pastillobox. Carblock produced the movie Vitrilateral with others. Carblock produced the movie Clenestration with others. Firmline was one of the producers of the movie Pestok. 

33 

Published as a conference paper at ICLR 2023 

Q: Who has been awarded the Modiparity award? A: [”Bioperatology”] 

Q: Which awards were given to Pestok? A: [”Gutney”] 

movie: Nohit ; director: Mimicocycle. movie: Noenometer ; director: Mimicocycle. movie: Tayenne ; directed by: Zayage. movie: Pneumodendron ; director: Sclerocybin. movie: Tayenne ; awarded: Goosehead. movie: Nohit ; awarded: Handt. movie: Pneumodendron ; award: Handt. movie: Noenometer ; awarded: Brownbeard. movie: Nohit ; writer: Mimicocycle. movie: Noenometer ; written by: Sclerocybin. movie: Tayenne ; writer: Sclerocybin. movie: Pneumodendron ; written by: Zayage. movie: Tayenne ; writer: Zayage. movie: Pneumodendron ; written by: Mimicocycle. movie: Noenometer ; release year: 1991. movie: Tayenne ; year: 2013. movie: Nohit ; year: 2005. movie: Pneumodendron ; year: 2005. Mimicocycle was an actor in the movie Tayenne. Zayage acted in the movie Pneumodendron. Zayage was an actor in the movie Nohit. Sclerocybin was an actor in the movie Nohit. Sclerocybin was an actor in the movie Tayenne. Mimicocycle was an actor in the movie Noenometer. Zayage was born in 1935. Sclerocybin was born in 1935. Mimicocycle was born in 1930. Mimicocycle is from the country of Calderita. Sclerocybin grew up in the nation of Calderita. Zayage is from the country of Obility. Quinion was awarded to Zayage. Fannyxist was awarded to Sclerocybin. Fannyxist was awarded to Mimicocycle. Mimicocycle produced the movie Nohit with others. Zayage was one of the producers of the movie Nohit. Sclerocybin was one of the producers of the movie Tayenne. Sclerocybin produced the movie Pneumodendron with others. Zayage produced the movie Pneumodendron with others. Mimicocycle was one of the producers of the movie Tayenne. Sclerocybin was one of the producers of the movie Noenometer. Zayage produced the movie Noenometer with others Q: Which awards were given to Noenometer? 

A: [”Brownbeard”] 

Q: Which awards were given to Pneumodendron? A: [”Handt”] 

## pos ~~q~~ a 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw ; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: For which movies was Teeplemole the producer? 

A: [”Featsaw”, ”Zalate”] 

Q: For which movies was Muntaril the producer? A: [”Premercy] 

34 

Published as a conference paper at ICLR 2023 

movie: Misgendery ; directed by: Wetherality. movie: Dewbar ; director: Gigabut. movie: Caudacite ; director: Lougeri`ere. movie: Tayenne ; directed by: Lougeri`ere. movie: Misgendery ; awarded: Microsouenesis. movie: Dewbar ; awarded: Erowid. movie: Tayenne ; awarded: Cockspit. movie: Caudacite ; award: Erowid. award: Aniconder ; winner: Wetherality. award: Aniconder ; winner: Lougeri`ere. person: Gigabut ; award: Trifogation. movie: Dewbar ; release year: 1991. movie: Tayenne ; year: 2013. movie: Caudacite ; release year: 2008. movie: Misgendery ; year: 1991. Wetherality was an actor in the movie Dewbar. Gigabut was an actor in the movie Tayenne. Lougeri`ere was an actor in the movie Tayenne. Lougeri`ere acted in the movie Caudacite. Lougeri`ere acted in the movie Misgendery. Gigabut was an actor in the movie Caudacite. Wetherality was an actor in the movie Misgendery. Wetherality was born in the year 1917. Lougeri`ere was born in 1926. Gigabut was born in the year 1917. Gigabut grew up in the nation of Triclops. Lougeri`ere is from the country of Tatkin . Wetherality grew up in the nation of Tatkin. Lougeri`ere produced the movie Dewbar with others. Gigabut produced the movie Tayenne with others. Gigabut produced the movie Dewbar with others. Lougeri`ere was one of the producers of the movie Misgendery. Wetherality was one of the producers of the movie Caudacite. Gigabut was one of the producers of the movie Caudacite. Wetherality produced the movie Misgendery with others. Wetherality produced the movie Tayenne with others. Wetherality wrote for the movie Tayenne. Gigabut wrote for the movie Misgendery. Lougeri`ere was one of the writers for the movie Caudacite. Wetherality wrote for the movie Misgendery. Gigabut wrote for the movie Tayenne. Gigabut wrote for the movie Dewbar. Lougeri`ere wrote for the movie Dewbar. Wetherality wrote for the movie Caudacite. 

Q: Who are the actors in the movie Dewbar? 

A: [”Wetherality”] 

Q: Who are the actors in the movie Caudacite? A: [”Lougeri`ere, ”Gigabut”] 

movie: Pastillobox ; directed by: Firmline. movie: Clenestration ; directed by: Carblock. movie: Pestok ; directed by: Bioperatology. movie: Vitrilateral ; director: Bioperatology. movie: Vitrilateral ; award: Antipositive. movie: Clenestration ; awarded: Handt. movie: Pastillobox ; awarded: Handt. movie: Pestok ; awarded: Gutney. movie: Pestok ; writer: Firmline. movie: Clenestration ; written by: Carblock. movie: Pastillobox ; written by: Bioperatology. movie: Pestok ; writer: Bioperatology. movie: Clenestration ; written by: Firmline. movie: Vitrilateral ; writer: Bioperatology. movie: Pastillobox ; writer: Carblock. movie: Vitrilateral ; written by : Carblock. movie: Pestok ; release year: 1986. movie: Clenestration ; year: 1986. movie: Vitrilateral ; year: 1999. movie: Pastillobox ; release year: 1984. Carblock was an actor in the movie Pastillobox. Firmline was an actor in the movie Vitrilateral. Bioperatology was an actor in the movie Clenestration. Firmline acted in the movie Pastillobox. Carblock was an actor in the movie Clenestration. Bioperatology was an actor in the movie Pestok. Firmline was born in the year 1904. Bioperatology was born in the year 1935. Carblock was born in 1935. Carblock grew up in the nation of Knoppock. Firmline grew up in the nation of Tatkin. Bioperatology grew up in the nation of Tatkin. Bioperatology won the Modiparity award. Halfbill was awarded to Firmline. Halfbill was awarded to Carblock. Bioperatology was one of the producers of the movie Pestok. Bioperatology produced the movie Vitrilateral with others. Firmline produced the movie Pastillobox with others. Firmline produced the movie Clenestration with others. Carblock was one of the producers of the movie Pastillobox. Carblock produced the movie Vitrilateral with others. Carblock produced the movie Clenestration with others. Firmline was one of the producers of the movie Pestok. Q: Which movies has Bioperatology directed? 

A: [”Pestok”, ”Vitrilateral”] 

Q: Which movies has Carblock directed? A: [”Clenestration”] 

movie: Nohit ; director: Mimicocycle. movie: Noenometer ; director: Mimicocycle. movie: Tayenne ; directed by: Zayage. movie: Pneumodendron ; director: Sclerocybin. movie: Tayenne ; awarded: Goosehead. movie: Nohit ; awarded: Handt. movie: Pneumodendron ; award: Handt. movie: Noenometer ; awarded: Brownbeard. movie: Nohit ; writer: 

35 

Published as a conference paper at ICLR 2023 

Mimicocycle. movie: Noenometer ; written by: Sclerocybin. movie: Tayenne ; writer: Sclerocybin. movie: Pneumodendron ; written by: Zayage. movie: Tayenne ; writer: Zayage. movie: Pneumodendron ; written by: Mimicocycle. movie: Noenometer ; release year: 1991. movie: Tayenne ; year: 2013. movie: Nohit ; year: 2005. movie: Pneumodendron ; year: 2005. Mimicocycle was an actor in the movie Tayenne. Zayage acted in the movie Pneumodendron. Zayage was an actor in the movie Nohit. Sclerocybin was an actor in the movie Nohit. Sclerocybin was an actor in the movie Tayenne. Mimicocycle was an actor in the movie Noenometer. Zayage was born in 1935. Sclerocybin was born in 1935. Mimicocycle was born in 1930. Mimicocycle is from the country of Calderita. Sclerocybin grew up in the nation of Calderita. Zayage is from the country of Obility. Quinion was awarded to Zayage. Fannyxist was awarded to Sclerocybin. Fannyxist was awarded to Mimicocycle. Mimicocycle produced the movie Nohit with others. Zayage was one of the producers of the movie Nohit. Sclerocybin was one of the producers of the movie Tayenne. Sclerocybin produced the movie Pneumodendron with others. Zayage produced the movie Pneumodendron with others. Mimicocycle was one of the producers of the movie Tayenne. Sclerocybin was one of the producers of the movie Noenometer. Zayage produced the movie Noenometer with others Q: What movies has Sclerocybin written? 

A: [”Noenometer”, ”Tayenne”] 

Q: What movies has Zayage written? A: [”Pneumodendron”, ”Tayenne”] 

## simp ~~q~~ a 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw ; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: Who were born in the year 1910? 

A: [”Teeplemole”, ”Muntaril”] 

Q: From which country is Monsterscar? A: [”Piperfish”] 

movie: Nilitude ; director: Monsterscar. movie: Dewbar ; directed by: Metatoun. movie: Warpstone ; directed by: Gastrat. movie: Partnershipmaker ; director: Metatoun. movie: Dewbar ; award: Tachychronograph. movie: Partnershipmaker ; awarded: Tachychronograph. movie: Nilitude ; award: Paleodactyl. movie: Warpstone ; award: Sabonade. person: Gastrat ; award: Trifogation. award: Polyquadrase ; winner: Monsterscar. award: Trifogation ; winner: Metatoun. movie: Warpstone ; release year: 1956. movie: Dewbar ; release year: 1984. movie: Nilitude ; year: 1984. movie: Partnershipmaker ; year: 1962. Gastrat was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Nilitude. Gastrat acted in the movie Nilitude. Monsterscar was an actor in the movie Dewbar. Gastrat acted in the movie Warpstone. Metatoun acted in the movie Warpstone. Metatoun was born in 1939. Gastrat was born in the year 1933. 

36 

Published as a conference paper at ICLR 2023 

Monsterscar was born in 1933. Metatoun grew up in the nation of Moulole. Gastrat is from the country of Stridery. Monsterscar grew up in the nation of Moulole. Monsterscar produced the movie Nilitude with others. Monsterscar was one of the producers of the movie Warpstone . Metatoun was one of the producers of the movie Warpstone. Gastrat was one of the producers of the movie Nilitude. Metatoun produced the movie Partnershipmaker with others. Metatoun produced the movie Dewbar with others. Monsterscar was one of the producers of the movie Partnershipmaker. Gastrat produced the movie Dewbar with others. Metatoun wrote for the movie Partnershipmaker. Gastrat wrote for the movie Warpstone. Gastrat was one of the writers for the movie Dewbar. Monsterscar was one of the writers for the movie Nilitude. Metatoun wrote for the movie Warpstone. 

Q: Who is from the country Stridery? 

A: [”Gastrat”] 

Q: Which movies were released in 1984? A: [”Dewbar”, ”Nilitude”] 

movie: Nohit ; director: Mimicocycle. movie: Noenometer ; director: Mimicocycle. movie: Tayenne ; directed by: Zayage. movie: Pneumodendron ; director: Sclerocybin. movie: Tayenne ; awarded: Goosehead. movie: Nohit ; awarded: Handt. movie: Pneumodendron ; award: Handt. movie: Noenometer ; awarded: Brownbeard. movie: Nohit ; writer: Mimicocycle. movie: Noenometer ; written by: Sclerocybin. movie: Tayenne ; writer: Sclerocybin. movie: Pneumodendron ; written by: Zayage. movie: Tayenne ; writer: Zayage. movie: Pneumodendron ; written by: Mimicocycle. movie: Noenometer ; release year: 1991. movie: Tayenne ; year: 2013. movie: Nohit ; year: 2005. movie: Pneumodendron ; year: 2005. Mimicocycle was an actor in the movie Tayenne. Zayage acted in the movie Pneumodendron. Zayage was an actor in the movie Nohit. Sclerocybin was an actor in the movie Nohit. Sclerocybin was an actor in the movie Tayenne. Mimicocycle was an actor in the movie Noenometer. Zayage was born in 1935. Sclerocybin was born in 1935. Mimicocycle was born in 1930. Mimicocycle is from the country of Calderita. Sclerocybin grew up in the nation of Calderita. Zayage is from the country of Obility. Quinion was awarded to Zayage. Fannyxist was awarded to Sclerocybin. Fannyxist was awarded to Mimicocycle. Mimicocycle produced the movie Nohit with others. Zayage was one of the producers of the movie Nohit. Sclerocybin was one of the producers of the movie Tayenne. Sclerocybin produced the movie Pneumodendron with others. Zayage produced the movie Pneumodendron with others. Mimicocycle was one of the producers of the movie Tayenne. Sclerocybin was one of the producers of the movie Noenometer. Zayage produced the movie Noenometer with others Q: Who were born in the year 1935? 

A: [”Sclerocybin”, ”Zayage”] 

Q: When was the movie Nohit released? A: [”2005”] 

movie: Coacheship ; director: Metatoun. movie: Assamplifier ; director: Kapod. movie: Misapportionment ; director: Sapien. movie: Quinsid ; director: Kapod. movie: Assamplifier ; award: Zorgion. movie: Quinsid ; awarded: Airpipe. movie: Coacheship ; award: Electrodesal. movie: Misapportionment ; award: Airpipe. movie: Coacheship ; written by: Metatoun. movie: Misapportionment ; written by: Kapod. movie: Coacheship ; written by: Kapod. movie : Quinsid ; writer: Sapien. movie: Misapportionment ; written by: Metatoun. movie: Assamplifier ; written by: Kapod. movie: Assamplifier ; written by: Sapien. movie: Assamplifier ; release year: 2000. movie: Coacheship ; year: 2001. movie: Quinsid ; year: 2005. movie: Misapportionment ; year: 2005. Sapien was an actor in the movie Misapportionment. Sapien acted in the movie Assamplifier. Kapod acted in the movie Quinsid . Sapien acted in the movie Coacheship. Metatoun was an actor in the movie Quinsid. Kapod acted in the movie Misapportionment. Metatoun acted in the movie Coacheship. Kapod acted in the movie Assamplifier. Sapien was born in the year 1910. Metatoun was born in 1928. Kapod was born in the year 1910. Metatoun is from the country of Legault. Kapod grew up in the nation of Tatkin. Sapien is from the country of Legault. Malwarp was awarded to Sapien. Metatoun won the Monkeynote award. Kapod won the Monkeynote award. Kapod was one of the producers of the movie Quinsid. Metatoun was one of the producers of the movie 

37 

Published as a conference paper at ICLR 2023 

Misapportionment. Metatoun was one of the producers of the movie Quinsid. Sapien was one of the producers of the movie Assamplifier. Sapien produced the movie Coacheship with others. Metatoun was one of the producers of the movie Assamplifier. Kapod was one of the producers of the movie Misapportionment. Kapod was one of the producers of the movie Coacheship. 

- Q: Who is from the country Legault? 

- A: [”Metatoun”, ”Sapien”] 

Q: When was Kapod born? 

- A: [”1910”] 

## G.3.3 COT 

## COT 

- movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw ; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: 

- Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was 

- born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was 

- one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

- QC: What awards have movies produced by people born in 1910 won? 

- QS: The people born in 1910 were Muntaril and Teeplemole. Teeplemole produced the movies Featsaw and Zalate. Muntaril produced the movie Premercy. Featsaw was awarded the Zorgion award. Premercy was awarded the Chowwurst award. Zalate was awarded the Hallowcock award. So the answer is [”Zorgion”, ”Chowwurst”, ”Hallowcock”]. 

A: [”Zorgion”, ”Chowwurst”, ”Hallowcock”] QS: [EOQ] 

- movie: Misgendery ; directed by: Wetherality. movie: Dewbar ; director: Gigabut. movie: Caudacite ; director: Lougeri`ere. movie: Tayenne ; directed by: Lougeri`ere. movie: Misgendery ; awarded: Microsouenesis. movie: Dewbar ; awarded: Erowid. movie: Tayenne ; awarded: Cockspit. movie: Caudacite ; award: Erowid. award: Aniconder ; winner: Wetherality. award: Aniconder ; winner: Lougeri`ere. person: Gigabut ; award: Trifogation. movie: Dewbar ; release year: 1991. movie: Tayenne ; year: 2013. movie: Caudacite ; release year: 2008. movie: Misgendery ; year: 1991. Wetherality was an actor in the movie Dewbar. Gigabut was an actor in the movie Tayenne. Lougeri`ere was an actor in the movie Tayenne. Lougeri`ere acted in the movie Caudacite. Lougeri`ere acted in the movie Misgendery. Gigabut was an actor in the movie Caudacite. Wetherality was an actor in the movie Misgendery. Wetherality was born in the year 1917. Lougeri`ere was born in 1926. Gigabut was born in the year 1917. Gigabut grew up in the nation of Triclops. Lougeri`ere is from the country of Tatkin . Wetherality grew up in the nation of Tatkin. Lougeri`ere produced the movie Dewbar with others. Gigabut produced the movie Tayenne with others. Gigabut produced the movie Dewbar with others. Lougeri`ere was one of the producers of the movie Misgendery. Wetherality was one of the producers of the movie Caudacite. Gigabut was one of the producers of the movie Caudacite. Wetherality produced the movie Misgendery with others. Wetherality produced the movie Tayenne with others. Wetherality wrote for the movie 

38 

Published as a conference paper at ICLR 2023 

   - Tayenne. Gigabut wrote for the movie Misgendery. Lougeri`ere was one of the writers for the movie Caudacite. Wetherality wrote for the movie Misgendery. Gigabut wrote for the movie Tayenne. Gigabut wrote for the movie Dewbar. Lougeri`ere wrote for the movie Dewbar. Wetherality wrote for the movie Caudacite. 

- QC: What awards have the actors of the Erowid winning movies received? 

- QS: The movies that won the Erowid award are Dewbar and Caudacite. Wetherality was an actor in Dewbar. Lougeri`ere and Gigabut acted in Caudacite. Wetherality won the Aniconder award. 

- Lougeri`ere also won the Aniconder award. Gigabut won the Trifogation award. So the answer is [”Aniconder”, ”Trifogation”]. 

- A: [”Aniconder”, ”Trifogation”] 

QS: [EOQ] 

- movie: Pastillobox ; directed by: Firmline. movie: Clenestration ; directed by: Carblock. movie: Pestok ; directed by: Bioperatology. movie: Vitrilateral ; director: Bioperatology. movie: Vitrilateral ; award: Antipositive. movie: Clenestration ; awarded: Handt. movie: Pastillobox ; awarded: Handt. movie: Pestok ; awarded: Gutney. movie: Pestok ; writer: Firmline. movie: Clenestration ; written by: Carblock. movie: Pastillobox ; written by: Bioperatology. movie: Pestok ; writer: Bioperatology. movie: Clenestration ; written by: Firmline. movie: Vitrilateral ; writer: Bioperatology. movie: Pastillobox ; writer: Carblock. movie: Vitrilateral ; written by 

- : Carblock. movie: Pestok ; release year: 1986. movie: Clenestration ; year: 1986. movie: Vitrilateral ; year: 1999. movie: Pastillobox ; release year: 1984. Carblock was an actor in the movie Pastillobox. Firmline was an actor in the movie Vitrilateral. Bioperatology was an actor in the movie Clenestration. Firmline acted in the movie Pastillobox. Carblock was an actor in the movie Clenestration. Bioperatology was an actor in the movie Pestok. Firmline was born 

- in the year 1904. Bioperatology was born in the year 1935. Carblock was born in 1935. Carblock grew up in the nation of Knoppock. Firmline grew up in the nation of Tatkin. Bioperatology grew up in the nation of Tatkin. Bioperatology won the Modiparity award. Halfbill was awarded to Firmline. Halfbill was awarded to Carblock. Bioperatology was one of the producers of the movie Pestok. Bioperatology produced the movie Vitrilateral with others. Firmline produced the movie Pastillobox with others. Firmline produced the movie Clenestration with others. Carblock was one of the producers of the movie Pastillobox. Carblock produced the movie Vitrilateral with others. Carblock produced the movie Clenestration with others. Firmline was one of the producers of the movie Pestok. 

- QC: What awards did the movies directed by the Modiparity winners receive? 

- QS: The Modiparity winners are Bioperatology. Bioperatology has directed the movies Pestok and Vitrilateral. Pestok has won the Gutney award. Vitrilateral has won the Antipositive award. So the answer is [”Gutney”, ”Antipositive”]. 

- A: [”Gutney”, ”Antipositive”] QS: [EOQ] 

- movie: Nohit ; director: Mimicocycle. movie: Noenometer ; director: Mimicocycle. movie: Tayenne ; directed by: Zayage. movie: Pneumodendron ; director: Sclerocybin. movie: Tayenne ; awarded: Goosehead. movie: Nohit ; awarded: Handt. movie: Pneumodendron ; award: Handt. movie: Noenometer ; awarded: Brownbeard. movie: Nohit ; writer: Mimicocycle. movie: Noenometer ; written by: Sclerocybin. movie: Tayenne ; writer: Sclerocybin. movie: Pneumodendron ; written by: Zayage. movie: Tayenne ; writer: Zayage. movie: Pneumodendron ; written by: Mimicocycle. movie: Noenometer ; release year: 1991. movie: Tayenne ; year: 2013. movie: Nohit ; year: 2005. movie: Pneumodendron ; year: 2005. Mimicocycle was an actor in the movie Tayenne. Zayage acted in the movie Pneumodendron. Zayage was an actor in the movie Nohit. Sclerocybin was an actor in the movie Nohit. 

- Sclerocybin was an actor in the movie Tayenne. Mimicocycle was an actor in the movie Noenometer. Zayage was born in 1935. Sclerocybin was born in 1935. Mimicocycle was born in 1930. Mimicocycle is from the country of Calderita. Sclerocybin grew up in the nation of 

- Calderita. Zayage is from the country of Obility. Quinion was awarded to Zayage. Fannyxist was awarded to Sclerocybin. Fannyxist was awarded to Mimicocycle. Mimicocycle produced the movie Nohit with others. Zayage was one of the producers of the movie Nohit. Sclerocybin was one of the producers of the movie Tayenne. Sclerocybin produced the movie Pneumodendron with others. Zayage produced the movie Pneumodendron with others. 

39 

Published as a conference paper at ICLR 2023 

Mimicocycle was one of the producers of the movie Tayenne. Sclerocybin was one of the producers of the movie Noenometer. Zayage produced the movie Noenometer with others 

- QC: What awards have movies written by people born in 1935 won? 

- QS: The people born in 1935 were Sclerocybin and Zayage. Sclerocybin has written the movies Noenometer and Tayenne. Zayage has written the movies Pneumodendron and Tayenne. Noenometer has won the Brownbeard award. Tayenne has won the Goosehead award. Pneumodendron has won the Handt award. So the answer is [”Brownbeard”, ”Goosehead”, ” Handt”]. 

- A: [”Brownbeard”, ”Goosehead”, ”Handt”] QS: [EOQ] 

- G.4 SHORTER PROMPTS FOR SMALLER CONTEXT WINDOWS 

## qa(small) 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw 

; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: Which awards were given to Zalate? 

A: [”Hallowcock”] 

Q: For which movies was Muntaril the producer? A: [”Premercy] 

Q: Who are the actors in the movie Premercy? A: [”Monsterscar”] 

movie: Nilitude ; director: Monsterscar. movie: Dewbar ; directed by: Metatoun. movie: Warpstone ; directed by: Gastrat. movie: Partnershipmaker ; director: Metatoun. movie: Dewbar ; award: Tachychronograph. movie: Partnershipmaker ; awarded: Tachychronograph. movie: Nilitude ; award: Paleodactyl. movie: Warpstone ; award: Sabonade. person: Gastrat ; award: Trifogation. award: Polyquadrase ; winner: Monsterscar. award: Trifogation ; winner: Metatoun. movie: Warpstone ; release year: 1956. movie: Dewbar ; release year: 1984. movie: Nilitude ; year: 1984. movie: Partnershipmaker ; year: 1962. Gastrat was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Nilitude. Gastrat acted in the movie Nilitude. Monsterscar was an actor in the movie Dewbar. Gastrat acted in the movie Warpstone. Metatoun acted in the movie Warpstone. Metatoun was born in 1939. Gastrat was born in the year 1933. Monsterscar was born in 1933. Metatoun grew up in the nation of Moulole. Gastrat is from the country of Stridery. Monsterscar grew up in the nation of Moulole. Monsterscar produced the movie Nilitude with others. Monsterscar was one of the producers of the movie Warpstone . Metatoun was one of the producers of the movie Warpstone. Gastrat was one of the producers of the movie Nilitude. Metatoun produced the movie Partnershipmaker with others. Metatoun produced the movie Dewbar with others. Monsterscar was one of the producers of 

40 

Published as a conference paper at ICLR 2023 

the movie Partnershipmaker. Gastrat produced the movie Dewbar with others. Metatoun wrote for the movie Partnershipmaker. Gastrat wrote for the movie Warpstone. Gastrat was one of the writers for the movie Dewbar. Monsterscar was one of the writers for the movie Nilitude. Metatoun wrote for the movie Warpstone. 

Q: Which movies has Gastrat been an actor in? 

A: [”Partnershipmaker”, ”Nilitude”, ”Warpstone”] 

Q: Who is from the country Stridery? 

A: [”Gastrat”] 

Q: Which movies were given the Tachychronograph award? 

A: [”Dewbar”, ”Partnershipmaker”] 

## aw ~~q~~ a(small) 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw 

; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: Which awards were given to Zalate? 

A: [”Hallowcock”] 

Q: Which awards were given to Premercy? A: [”Chowwurst”] 

movie: Misgendery ; directed by: Wetherality. movie: Dewbar ; director: Gigabut. movie: Caudacite ; director: Lougeri`ere. movie: Tayenne ; directed by: Lougeri`ere. movie: Misgendery ; awarded: Microsouenesis. movie: Dewbar ; awarded: Erowid. movie: Tayenne ; awarded: Cockspit. movie: Caudacite ; award: Erowid. award: Aniconder ; winner: Wetherality. award: Aniconder ; winner: Lougeri`ere. person: Gigabut ; award: Trifogation. movie: Dewbar ; release year: 1991. movie: Tayenne ; year: 2013. movie: Caudacite ; release year: 2008. movie: Misgendery ; year: 1991. Wetherality was an actor in the movie Dewbar. Gigabut was an actor in the movie Tayenne. Lougeri`ere was an actor in the movie Tayenne. Lougeri`ere acted in the movie Caudacite. Lougeri`ere acted in the movie Misgendery. Gigabut was an actor in the movie Caudacite. Wetherality was an actor in the movie Misgendery. Wetherality was born in the year 1917. Lougeri`ere was born in 1926. Gigabut was born in the year 1917. Gigabut grew up in the nation of Triclops. Lougeri`ere is from the country of Tatkin . Wetherality grew up in the nation of Tatkin. Lougeri`ere produced the movie Dewbar with others. Gigabut produced the movie Tayenne with others. Gigabut produced the movie Dewbar with others. Lougeri`ere was one of the producers of the movie Misgendery. Wetherality was one of the producers of the movie Caudacite. Gigabut was one of the producers of the movie Caudacite. Wetherality produced the movie Misgendery with others. Wetherality produced the movie Tayenne with others. Wetherality wrote for the movie Tayenne. Gigabut wrote for the movie Misgendery. Lougeri`ere was one of the writers for the movie Caudacite. Wetherality wrote for the movie Misgendery. Gigabut wrote for the movie 

41 

Published as a conference paper at ICLR 2023 

Tayenne. Gigabut wrote for the movie Dewbar. Lougeri`ere wrote for the movie Dewbar. Wetherality wrote for the movie Caudacite. 

Q: Which movies were given the Erowid award? 

A: [”Dewbar”, ”Caudacite”] 

Q: Which awards were given to Wetherality? A: [”Aniconder”] 

## pos ~~q~~ a(small) 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw 

; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: For which movies was Teeplemole the producer? A: [”Featsaw”, ”Zalate”] 

Q: For which movies was Muntaril the producer? A: [”Premercy] 

- movie: Nilitude ; director: Monsterscar. movie: Dewbar ; directed by: Metatoun. movie: Warpstone ; directed by: Gastrat. movie: Partnershipmaker ; director: Metatoun. movie: Dewbar ; award: Tachychronograph. movie: Partnershipmaker ; awarded: Tachychronograph. movie: Nilitude ; award: Paleodactyl. movie: Warpstone ; award: Sabonade. person: Gastrat ; award: 

- Trifogation. award: Polyquadrase ; winner: Monsterscar. award: Trifogation ; winner: Metatoun. movie: Warpstone ; release year: 1956. movie: Dewbar ; release year: 1984. movie: Nilitude ; year: 1984. movie: Partnershipmaker ; year: 1962. Gastrat was an actor in the 

- movie Partnershipmaker. Metatoun was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Nilitude. Gastrat acted in the movie Nilitude. Monsterscar was an actor in the movie Dewbar. Gastrat acted in the movie Warpstone. Metatoun acted in the movie Warpstone. Metatoun was born in 1939. Gastrat was born in the year 1933. Monsterscar was born in 1933. Metatoun grew up in the nation of Moulole. Gastrat is from the country of Stridery. Monsterscar grew up in the nation of Moulole. Monsterscar produced the movie Nilitude with others. Monsterscar was one of the producers of the movie Warpstone . Metatoun was one of the producers of the movie Warpstone. Gastrat was one of the producers of the movie Nilitude. Metatoun produced the movie Partnershipmaker with others. Metatoun produced the movie Dewbar with others. Monsterscar was one of the producers of 

- the movie Partnershipmaker. Gastrat produced the movie Dewbar with others. Metatoun wrote for the movie Partnershipmaker. Gastrat wrote for the movie Warpstone. Gastrat was one of the writers for the movie Dewbar. Monsterscar was one of the writers for the movie Nilitude. Metatoun wrote for the movie Warpstone. 

Q: Which movies has Gastrat been an actor in? 

A: [”Partnershipmaker”, ”Nilitude”, ”Warpstone”] 

Q: Which movies has Monsterscar been an actor in? 

42 

Published as a conference paper at ICLR 2023 

## A: [”Dewbar”] 

## simp ~~q~~ a(small) 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw ; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

Q: Who were born in the year 1910? 

A: [”Teeplemole”, ”Muntaril”] 

Q: Who is from the country Piperfish? A: [”Teeplemole”, ”Monsterscar”] 

movie: Nilitude ; director: Monsterscar. movie: Dewbar ; directed by: Metatoun. movie: Warpstone ; directed by: Gastrat. movie: Partnershipmaker ; director: Metatoun. movie: Dewbar ; award: Tachychronograph. movie: Partnershipmaker ; awarded: Tachychronograph. movie: Nilitude ; award: Paleodactyl. movie: Warpstone ; award: Sabonade. person: Gastrat ; award: Trifogation. award: Polyquadrase ; winner: Monsterscar. award: Trifogation ; winner: Metatoun. movie: Warpstone ; release year: 1956. movie: Dewbar ; release year: 1984. movie: Nilitude ; year: 1984. movie: Partnershipmaker ; year: 1962. Gastrat was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Nilitude. Gastrat acted in the movie Nilitude. Monsterscar was an actor in the movie Dewbar. Gastrat acted in the movie Warpstone. Metatoun acted in the movie Warpstone. Metatoun was born in 1939. Gastrat was born in the year 1933. Monsterscar was born in 1933. Metatoun grew up in the nation of Moulole. Gastrat is from the country of Stridery. Monsterscar grew up in the nation of Moulole. Monsterscar produced the movie Nilitude with others. Monsterscar was one of the producers of the movie Warpstone . Metatoun was one of the producers of the movie Warpstone. Gastrat was one of the producers of the movie Nilitude. Metatoun produced the movie Partnershipmaker with others. Metatoun produced the movie Dewbar with others. Monsterscar was one of the producers of the movie Partnershipmaker. Gastrat produced the movie Dewbar with others. Metatoun wrote for the movie Partnershipmaker. Gastrat wrote for the movie Warpstone. Gastrat was one of the writers for the movie Dewbar. Monsterscar was one of the writers for the movie Nilitude. Metatoun wrote for the movie Warpstone. 

Q: Who is from the country Stridery? 

A: [”Gastrat”] 

Q: Who were born in the year 1939? 

A: [”Metatoun”] 

## COT(small) 

movie: Premercy ; directed by: Muntaril. movie: Skirtsicine ; director: Teeplemole. movie: Featsaw ; directed by: Monsterscar. movie: Zalate ; director: Monsterscar. movie: Zalate ; awarded: 

43 

Published as a conference paper at ICLR 2023 

Hallowcock. movie: Featsaw ; awarded: Zorgion. movie: Premercy ; award: Chowwurst. movie: Skirtsicine ; award: Hallowcock. award: Goatfly ; winner: Teeplemole. person: Monsterscar ; award: Glodome. person: Muntaril ; award: Goatfly. movie: Featsaw ; release year: 1973. movie: Zalate ; release year: 1964. movie: Skirtsicine ; release year: 1973. movie: Premercy ; year: 1961. Teeplemole was an actor in the movie Skirtsicine. Muntaril was an actor in the movie Skirtsicine. Monsterscar was an actor in the movie Premercy. Muntaril was an actor in the movie Featsaw. Teeplemole was an actor in the movie Zalate. Muntaril was born in the year 1910. Teeplemole was born in 1910. Monsterscar was born in 1942. Teeplemole is from the country of Piperfish. Monsterscar is from the country of Piperfish. Muntaril is from the country of Clony. Muntaril produced the movie Skirtsicine with others. Monsterscar was one of the producers of the movie Featsaw. Monsterscar produced the movie Premercy with others. Monsterscar produced the movie Zalate with others. Teeplemole was one of the producers of the movie Featsaw. Teeplemole produced the movie Zalate with others. Muntaril produced the movie Premercy with others. Monsterscar wrote for the movie Premercy. Muntaril was one of the writers for the movie Zalate. Muntaril wrote for the movie Featsaw. Teeplemole wrote for the movie Featsaw. Monsterscar was one of the writers for the movie Zalate. Teeplemole was one of the writers for the movie Skirtsicine. 

- QC: What awards have movies produced by people born in 1910 won? 

- QS: The people born in 1910 were Muntaril and Teeplemole. Teeplemole produced the movies Featsaw and Zalate. Muntaril produced the movie Premercy. Featsaw was awarded the Zorgion award. Premercy was awarded the Chowwurst award. Zalate was awarded the Hallowcock award. So the answer is [”Zorgion”, ”Chowwurst”, ”Hallowcock”]. 

- A: [”Zorgion”, ”Chowwurst”, ”Hallowcock”] QS: [EOQ] 

- movie: Nilitude ; director: Monsterscar. movie: Dewbar ; directed by: Metatoun. movie: Warpstone ; directed by: Gastrat. movie: Partnershipmaker ; director: Metatoun. movie: Dewbar ; award: Tachychronograph. movie: Partnershipmaker ; awarded: Tachychronograph. movie: Nilitude ; award: Paleodactyl. movie: Warpstone ; award: Sabonade. person: Gastrat ; award: 

- Trifogation. award: Polyquadrase ; winner: Monsterscar. award: Trifogation ; winner: Metatoun. movie: Warpstone ; release year: 1956. movie: Dewbar ; release year: 1984. movie: Nilitude ; year: 1984. movie: Partnershipmaker ; year: 1962. Gastrat was an actor in the 

- movie Partnershipmaker. Metatoun was an actor in the movie Partnershipmaker. Metatoun was an actor in the movie Nilitude. Gastrat acted in the movie Nilitude. Monsterscar was an actor in the movie Dewbar. Gastrat acted in the movie Warpstone. Metatoun acted in the movie Warpstone. Metatoun was born in 1939. Gastrat was born in the year 1933. Monsterscar was born in 1933. Metatoun grew up in the nation of Moulole. Gastrat is from the country of Stridery. Monsterscar grew up in the nation of Moulole. Monsterscar produced the movie Nilitude with others. Monsterscar was one of the producers of the movie Warpstone . Metatoun was one of the producers of the movie Warpstone. Gastrat was one of the producers of the movie Nilitude. Metatoun produced the movie Partnershipmaker with others. Metatoun produced the movie Dewbar with others. Monsterscar was one of the producers of 

- the movie Partnershipmaker. Gastrat produced the movie Dewbar with others. Metatoun wrote for the movie Partnershipmaker. Gastrat wrote for the movie Warpstone. Gastrat was one of the writers for the movie Dewbar. Monsterscar was one of the writers for the movie Nilitude. Metatoun wrote for the movie Warpstone. 

- QC: What movies have people from the country Stridery acted in? 

- QS: The people born in Stridery are Gastrat. Gastrat acted in Partenrshipmaker, Nilitude and Warpstone. So the answer is [”Partnershipmaker”, ”Nilitude”, ”Warpstone”]. 

- A: [”Partnershipmaker”, ”Nilitude”, ”Warpstone”] 

- QS: [EOQ] 

## G.5 MATH QA 

The decomposer here deterministically calls cot to generate the CoT and then calls gpt ~~a~~ ns to extract the answer. 

cot 

44 

Published as a conference paper at ICLR 2023 

- Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 

- A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 −15 = 6. The answer is 6. 

- Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot ? 

- A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. 

- Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 

- A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 −35 = 39. The answer is 39. 

- Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? 

- A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 −12 = 8. The answer is 8. 

- Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? 

- A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. 

- Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? 

- A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. 

- Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? 

- A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 −23 = 35. After losing 2 more, he had 35 −2 = 33 golf balls. The answer is 33. 

- Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 

- A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 −15 dollars left. 23 −15 is 8. The answer is 8. 

## gpt ~~a~~ ns 

- Q: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 −15 = 6 trees planted. 

- A: 6 

- Q: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. A: 5 

- Q: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 −35 = 39. The answer is 39. 

- A: 39 

- Q: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 −12 = 8 apples. 

- A: 8 

- Q: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. 

- A: 9 

45 

Published as a conference paper at ICLR 2023 

- Q: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. 

- A: 29 

- Q: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 −23 = 35. After losing 2 more, he had 35 −2 = 33 golf balls. 

- A: 33 

- Q: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 −15 = 8 dollars left. 

- A: 8 

## G.6 OPEN DOMAIN QA 

The prompts in this section implement Decomposed Prompting approach to open-domain multihop QA. For brevity we’ve included prompts for 5 of 20 randomly sampled questions. The full prompts are attached with the submission and will also be released with the code. Note that we selected a set of 100 questions from the development set to tune the hyperparameter (number of paragraphs to retrieve for all of the retrieval-based approaches). 

hotpotqa: decomp 

- QC: In which country did this Australian who was detained in Guantanamo Bay detention camp and published ”Guantanamo: My Journey” receive para−military training? 

- QS: (select) [retrieve ~~o~~ dqa] Who is the Australian who was detained in Guantanamo Bay detention camp and published ”Guantanamo: My Journey”? 

- A: _{_ ”titles”: [”Guantanamo: My Journey”, ”Bismullah v. Gates”, ”Guantanamo Bay detention camp”], ”answer”: [”David Hicks”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] In which country did David Hicks receive his para−military training? 

- A: _{_ ”titles”: [”John Adams Project”, ”Camp Echo (Guantanamo Bay)”, ”Guantanamo Bay Museum of Art and History”, ”David Hicks”], ”answer”: [”Afghanistan”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Guantanamo: My Journey”, ”Bismullah v. Gates”, ” Guantanamo Bay detention camp”, ”John Adams Project”, ”Camp Echo (Guantanamo Bay)”, ”Guantanamo Bay Museum of Art and History”, ”David Hicks”]. Question: In which country did this Australian who was detained in Guantanamo Bay detention camp and published ” Guantanamo: My Journey” receive para−military training? 

- A: [”Afghanistan”] 

QS: [EOQ] 

- QC: How many awards did the ”A Girl Like Me” singer win at the American Music Awards of 2012? 

- QS: (select) [retrieve ~~o~~ dqa] Who is the singer of ”A Girl Like Me”? 

- A: _{_ ”titles”: [”American Music Award”, ”Rihanna”], ”answer”: [”Rihanna”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] How many awards did Rihana won in the American Music Awards of 2012? 

- A: _{_ ”titles”: [”American Music Awards of 2012”, ”APRA Music Awards of 2012”, ”List of awards and nominations received by TLC”, ”Native American Music Awards”], ”answer”: [”one”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”American Music Award”, ”Rihanna”, ”American Music Awards of 2012”, ”APRA Music Awards of 2012”, ”List of awards and nominations received by TLC”, ”Native American Music Awards”]. Question: How many awards did the ”A Girl Like Me” singer win at the American Music Awards of 2012? 

- A: [”one”] QS: [EOQ] 

QC: who is older Jeremy Horn or Renato Sobral ? QS: (select) [retrieve ~~o~~ dqa] When was Jeremy Horn born? 

46 

Published as a conference paper at ICLR 2023 

- A: _{_ ”titles”: [”Zaza Tkeshelashvili”, ”Jeremy Horn”, ”Jeremy Horn (singer)”, ”Ricardo Arona”], ” answer”: [”August 25, 1975”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] When was Renato Sobral born? 

- A: _{_ ”titles”: [”Brian Warren”, ”Renato Sobral”], ”answer”: [”September 7, 1975”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Zaza Tkeshelashvili”, ”Jeremy Horn”, ”Jeremy Horn ( singer)”, ”Ricardo Arona”, ”Brian Warren”, ”Renato Sobral”]. Question: who is older Jeremy Horn or Renato Sobral ? 

- A: [”Jeremy Horn”] 

QS: [EOQ] 

- QC: What was the 2014 population of the city where Lake Wales Medical Center is located? QS: (select) [retrieve ~~o~~ dqa] Lake Wales Medical Center is located in what city? 

- A: _{_ ”titles”: [”Baylor College of Medicine”, ”Lake Wales Medical Center”, ”Tufts University School of Medicine”, ”Hanford Community Medical Center”], ”answer”: [”Polk County, Florida”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] What was the population of Polk County in 2014? 

- A: _{_ ”titles”: [”Banner University Medical Center Tucson”, ”Lake Wales, Florida”], ”answer”: [”15,140”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Baylor College of Medicine”, ”Lake Wales Medical Center ”, ”Tufts University School of Medicine”, ”Hanford Community Medical Center”, ”Banner University Medical Center Tucson”, ”Lake Wales, Florida”]. Question: What was the 2014 population of the city where Lake Wales Medical Center is located? 

- A: [”15,140”] 

- QS: [EOQ] 

- QC: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono? 

- QS: (select) [retrieve ~~o~~ dqa] What album was issued by Apple Records, and written, recorded, and released during John Lennon’s 18 month separation from Yoko Ono? 

- A: _{_ ”titles”: [”John Lennon/Plastic Ono Band”, ”Milk and Honey (album)”, ”Walls and Bridges”], ”answer”: [”Walls and Bridges”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] Nobody Loves You was written by John Lennon on what album? 

- A: _{_ ”titles”: [”John Lennon Museum”, ”Nobody Loves You (When You’re Down and Out)”], ” answer”: [”Walls and Bridges”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”John Lennon/Plastic Ono Band”, ”Milk and Honey (album) ”, ”Walls and Bridges”, ”John Lennon Museum”, ”Nobody Loves You (When You’re Down and Out)”]. Question: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono? 

- A: [”Walls and Bridges”] 

- QS: [EOQ] 

hotpotqa: retrieve ~~o~~ dqa 

- QC: Who is the Australian who was detained in Guantanamo Bay detention camp and published ” Guantanamo: My Journey”? 

- QS: (select) [retrieve] Who is the Australian who was detained in Guantanamo Bay detention camp and published ”Guantanamo: My Journey”? 

- A: [”Guantanamo: My Journey”, ”Bismullah v. Gates”, ”Guantanamo Bay detention camp”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Guantanamo: My Journey”, ”Bismullah v. Gates”, ” Guantanamo Bay detention camp”]. Question: Who is the Australian who was detained in Guantanamo Bay detention camp and published ”Guantanamo: My Journey”? 

- A: _{_ ”titles”: [”Guantanamo: My Journey”, ”Bismullah v. Gates”, ”Guantanamo Bay detention camp”], ”answer”: [”David Hicks”] _}_ 

- QS: [EOQ] 

47 

Published as a conference paper at ICLR 2023 

- QC: In which country did David Hicks receive his para−military training? 

- QS: (select) [retrieve] In which country did David Hicks receive his para−military training? 

- A: [”John Adams Project”, ”Camp Echo (Guantanamo Bay)”, ”Guantanamo Bay Museum of Art and History”, ”David Hicks”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”John Adams Project”, ”Camp Echo (Guantanamo Bay)”, ” Guantanamo Bay Museum of Art and History”, ”David Hicks”]. Question: In which country did David Hicks receive his para−military training? 

- A: _{_ ”titles”: [”John Adams Project”, ”Camp Echo (Guantanamo Bay)”, ”Guantanamo Bay Museum of Art and History”, ”David Hicks”], ”answer”: [”Afghanistan”] _}_ 

- QS: [EOQ] 

QC: Who is the singer of ”A Girl Like Me”? 

- QS: (select) [retrieve] Who is the singer of ”A Girl Like Me”? 

- A: [”American Music Award”, ”Rihanna”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”American Music Award”, ”Rihanna”]. Question: Who is the singer of ”A Girl Like Me”? 

- A: _{_ ”titles”: [”American Music Award”, ”Rihanna”], ”answer”: [”Rihanna”] _}_ 

QS: [EOQ] 

- QC: How many awards did Rihana won in the American Music Awards of 2012? 

QS: (select) [retrieve] How many awards did Rihana won in the American Music Awards of 2012? 

- A: [”American Music Awards of 2012”, ”APRA Music Awards of 2012”, ”List of awards and nominations received by TLC”, ”Native American Music Awards”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”American Music Awards of 2012”, ”APRA Music Awards of 2012”, ”List of awards and nominations received by TLC”, ”Native American Music 

- Awards”]. Question: How many awards did Rihana won in the American Music Awards of 2012? 

- A: _{_ ”titles”: [”American Music Awards of 2012”, ”APRA Music Awards of 2012”, ”List of awards and nominations received by TLC”, ”Native American Music Awards”], ”answer”: [”one”] _}_ 

- QS: [EOQ] 

QC: When was Jeremy Horn born? 

- QS: (select) [retrieve] When was Jeremy Horn born? 

- A: [”Zaza Tkeshelashvili”, ”Jeremy Horn”, ”Jeremy Horn (singer)”, ”Ricardo Arona”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Zaza Tkeshelashvili”, ”Jeremy Horn”, ”Jeremy Horn ( singer)”, ”Ricardo Arona”]. Question: When was Jeremy Horn born? 

- A: _{_ ”titles”: [”Zaza Tkeshelashvili”, ”Jeremy Horn”, ”Jeremy Horn (singer)”, ”Ricardo Arona”], ” answer”: [”August 25, 1975”] _}_ 

- QS: [EOQ] 

QC: When was Renato Sobral born? 

- QS: (select) [retrieve] When was Renato Sobral born? 

- A: [”Brian Warren”, ”Renato Sobral”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Brian Warren”, ”Renato Sobral”]. Question: When was Renato Sobral born? 

- A: _{_ ”titles”: [”Brian Warren”, ”Renato Sobral”], ”answer”: [”September 7, 1975”] _}_ 

QS: [EOQ] 

QC: Lake Wales Medical Center is located in what city? 

QS: (select) [retrieve] Lake Wales Medical Center is located in what city? 

- A: [”Baylor College of Medicine”, ”Lake Wales Medical Center”, ”Tufts University School of Medicine”, ”Hanford Community Medical Center”] 

48 

Published as a conference paper at ICLR 2023 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Baylor College of Medicine”, ”Lake Wales Medical Center”, ”Tufts University School of Medicine”, ”Hanford Community Medical Center”]. Question: Lake Wales Medical Center is located in what city? 

- A: _{_ ”titles”: [”Baylor College of Medicine”, ”Lake Wales Medical Center”, ”Tufts University School of Medicine”, ”Hanford Community Medical Center”], ”answer”: [”Polk County, Florida”] _}_ 

- QS: [EOQ] 

QC: What was the population of Polk County in 2014? 

- QS: (select) [retrieve] What was the population of Polk County in 2014? 

- A: [”Banner University Medical Center Tucson”, ”Lake Wales, Florida”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Banner University Medical Center Tucson”, ”Lake Wales, Florida”]. Question: What was the population of Polk County in 2014? 

- A: _{_ ”titles”: [”Banner University Medical Center Tucson”, ”Lake Wales, Florida”], ”answer”: [”15,140”] _}_ 

- QS: [EOQ] 

- QC: What album was issued by Apple Records, and written, recorded, and released during John Lennon’s 18 month separation from Yoko Ono? 

- QS: (select) [retrieve] What album was issued by Apple Records, and written, recorded, and released during John Lennon’s 18 month separation from Yoko Ono? 

- A: [”John Lennon/Plastic Ono Band”, ”Milk and Honey (album)”, ”Walls and Bridges”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”John Lennon/Plastic Ono Band”, ”Milk and Honey (album )”, ”Walls and Bridges”]. Question: What album was issued by Apple Records, and written, recorded, and released during John Lennon’s 18 month separation from Yoko Ono? 

- A: _{_ ”titles”: [”John Lennon/Plastic Ono Band”, ”Milk and Honey (album)”, ”Walls and Bridges”], ”answer”: [”Walls and Bridges”] _}_ 

- QS: [EOQ] 

QC: Nobody Loves You was written by John Lennon on what album? 

- QS: (select) [retrieve] Nobody Loves You was written by John Lennon on what album? 

- A: [”John Lennon Museum”, ”Nobody Loves You (When You’re Down and Out)”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”John Lennon Museum”, ”Nobody Loves You (When You’ re Down and Out)”]. Question: Nobody Loves You was written by John Lennon on what album? 

- A: _{_ ”titles”: [”John Lennon Museum”, ”Nobody Loves You (When You’re Down and Out)”], ” answer”: [”Walls and Bridges”] _}_ 

- QS: [EOQ] 

hotpotqa: singlehop ~~t~~ itleqa 

Wikipedia Title: David Hicks _<_ hidden for brevity _>_ 

Wikipedia Title: Guantanamo: My Journey _<_ hidden for brevity _>_ 

Wikipedia Title: Murat Kurnaz _<_ hidden for brevity _>_ 

- Q: Who is the Australian who was detained in Guantanamo Bay detention camp and published ” Guantanamo: My Journey”? 

- A: [”David Hicks”] 

- Q: In which country did David Hicks receive his para−military training? 

A: [”Afghanistan”] 

49 

Published as a conference paper at ICLR 2023 

Wikipedia Title: American Music Award 

_<_ hidden for brevity _>_ 

Wikipedia Title: Rihanna _<_ hidden for brevity _>_ 

Wikipedia Title: American Music Awards of 2012 

_<_ hidden for brevity _>_ 

Q: Who is the singer of ”A Girl Like Me”? A: [”Rihanna”] 

Q: How many awards did Rihana won in the American Music Awards of 2012? A: [”one”] 

Wikipedia Title: Renato Sobral _<_ hidden for brevity _>_ 

Wikipedia Title: Zaza Tkeshelashvili _<_ hidden for brevity _>_ 

Wikipedia Title: Jeremy Horn _<_ hidden for brevity _>_ 

Q: When was Jeremy Horn born? A: [”August 25, 1975”] 

Q: When was Renato Sobral born? A: [”September 7, 1975”] 

Wikipedia Title: Lake Wales Medical Center _<_ hidden for brevity _>_ 

Wikipedia Title: Lake Wales, Florida _<_ hidden for brevity _>_ 

Wikipedia Title: Tufts University School of Medicine _<_ hidden for brevity _>_ 

Q: Lake Wales Medical Center is located in what city? A: [”Polk County, Florida”] 

Q: What was the population of Polk County in 2014? A: [”15,140”] 

Wikipedia Title: Nobody Loves You (When You’re Down and Out) _<_ hidden for brevity _>_ 

Wikipedia Title: Walls and Bridges _<_ hidden for brevity _>_ 

Wikipedia Title: Mother (John Lennon song) _<_ hidden for brevity _>_ 

50 

Published as a conference paper at ICLR 2023 

- Q: What album was issued by Apple Records, and written, recorded, and released during John Lennon’s 18 month separation from Yoko Ono? 

- A: [”Walls and Bridges”] 

- Q: Nobody Loves You was written by John Lennon on what album? 

A: [”Walls and Bridges”] 

hotpotqa: multihop ~~t~~ itleqa (direct) 

Wikipedia Title: David Hicks 

_<_ hidden for brevity _>_ 

Wikipedia Title: Guantanamo: My Journey 

_<_ hidden for brevity _>_ 

Wikipedia Title: Murat Kurnaz 

_<_ hidden for brevity _>_ 

Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published ”Guantanamo: My Journey” receive para−military training? A: [”Afghanistan”] 

Wikipedia Title: American Music Award 

_<_ hidden for brevity _>_ 

Wikipedia Title: Rihanna 

_<_ hidden for brevity _>_ 

Wikipedia Title: American Music Awards of 2012 

_<_ hidden for brevity _>_ 

- Q: How many awards did the ”A Girl Like Me” singer win at the American Music Awards of 2012? 

A: [”one”] 

Wikipedia Title: Renato Sobral 

_<_ hidden for brevity _>_ 

Wikipedia Title: Zaza Tkeshelashvili _<_ hidden for brevity _>_ 

Wikipedia Title: Jeremy Horn _<_ hidden for brevity _>_ 

Q: who is older Jeremy Horn or Renato Sobral ? A: [”Jeremy Horn”] 

Wikipedia Title: Lake Wales Medical Center _<_ hidden for brevity _>_ 

Wikipedia Title: Lake Wales, Florida _<_ hidden for brevity _>_ 

Wikipedia Title: Tufts University School of Medicine _<_ hidden for brevity _>_ 

Q: What was the 2014 population of the city where Lake Wales Medical Center is located? 

51 

Published as a conference paper at ICLR 2023 

A: [”15,140”] 

Wikipedia Title: Nobody Loves You (When You’re Down and Out) 

_<_ hidden for brevity _>_ 

Wikipedia Title: Walls and Bridges _<_ hidden for brevity _>_ 

Wikipedia Title: Mother (John Lennon song) 

_<_ hidden for brevity _>_ 

- Q: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from 

- Yoko Ono? 

- A: [”Walls and Bridges”] 

hotpotqa: multihop ~~t~~ itleqa (cot) 

Wikipedia Title: David Hicks 

_<_ hidden for brevity _>_ 

Wikipedia Title: Guantanamo: My Journey 

_<_ hidden for brevity _>_ 

Wikipedia Title: Murat Kurnaz 

_<_ hidden for brevity _>_ 

- Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published ”Guantanamo: My Journey” receive para−military training? 

- A: The Australian who was detained in Guantanamo Bay detention camp and published ” Guantanamo: My Journey” is David Hicks. David Hicks received his para−military training in Afghanistan. So the answer is: Afghanistan. 

Wikipedia Title: American Music Award 

_<_ hidden for brevity _>_ 

Wikipedia Title: Rihanna 

_<_ hidden for brevity _>_ 

Wikipedia Title: American Music Awards of 2012 

_<_ hidden for brevity _>_ 

- Q: How many awards did the ”A Girl Like Me” singer win at the American Music Awards of 2012? 

- A: The singer of ”A Girl Like Me” singer is Rihanna. In the American Music Awards of 2012, Rihana won one award. So the answer is: one. 

Wikipedia Title: Renato Sobral 

_<_ hidden for brevity _>_ 

Wikipedia Title: Zaza Tkeshelashvili _<_ hidden for brevity _>_ 

Wikipedia Title: Jeremy Horn _<_ hidden for brevity _>_ 

Q: who is older Jeremy Horn or Renato Sobral ? 

52 

Published as a conference paper at ICLR 2023 

- A: Jeremy Horn was born on August 25, 1975. Renato Sobral was born on September 7, 1975. Thus, Jeremy Horn is older. So the answer is: Jeremy Horn. 

Wikipedia Title: Lake Wales Medical Center 

_<_ hidden for brevity _>_ 

Wikipedia Title: Lake Wales, Florida 

_<_ hidden for brevity _>_ 

Wikipedia Title: Tufts University School of Medicine 

_<_ hidden for brevity _>_ 

- Q: What was the 2014 population of the city where Lake Wales Medical Center is located? 

- A: Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is: 15,140. 

Wikipedia Title: Nobody Loves You (When You’re Down and Out) 

_<_ hidden for brevity _>_ 

Wikipedia Title: Walls and Bridges 

_<_ hidden for brevity _>_ 

Wikipedia Title: Mother (John Lennon song) 

_<_ hidden for brevity _>_ 

- Q: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from 

- Yoko Ono? 

- A: The album issued by Apple Records, and written, recorded, and released during John Lennon’s 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. So the answer is: Walls and Bridges. 

2wikimultihopqa: decomp 

- QC: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality? 

QS: (select) [retrieve ~~o~~ dqa] Who is the director of the film Coolie No. 1 (1995 film)? 

A: _{_ ”titles”: [”Hanro Smitsman”, ”Coolie No. 1 (1995 film)”], ”answer”: [”David Dhawan”] _}_ QS: (select) [retrieve ~~o~~ dqa] Who is the director of the film The Sensational Trial? 

A: _{_ ”titles”: [”Rachel Feldman”, ”The Sensational Trial”], ”answer”: [”Karl Freund”] _}_ QS: (select) [retrieve ~~o~~ dqa] What is David Dhawan’s nationality? 

A: _{_ ”titles”: [”David Dhawan”, ”Brian Johnson (special effects artist)”], ”answer”: [”India”] _}_ QS: (select) [retrieve ~~o~~ dqa] What is Karl Freund’s nationality? 

A: _{_ ”titles”: [”Karl Freund”, ”Ian Barry (director)”], ”answer”: [”Germany”] _}_ 

QS: (select) [multihop ~~t~~ itleqa] Titles: [”Hanro Smitsman”, ”Coolie No. 1 (1995 film)”, ”Rachel Feldman”, ”The Sensational Trial”, ”David Dhawan”, ”Brian Johnson (special effects artist)”, ”Karl Freund”, ”Ian Barry (director)”]. Question: Do director of film Coolie No. 1 (1995 Film ) and director of film The Sensational Trial have the same nationality? 

A: [”no”] QS: [EOQ] 

QC: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two? QS: (select) [retrieve ~~o~~ dqa] Who is the director of the film Twenty Plus? 

A: _{_ ”titles”: [”Riki Gal”, ”Twenty Plus Two”], ”answer”: [”Joseph M. Newman”] _}_ QS: (select) [retrieve ~~o~~ dqa] Who is the director of the film The Gal Who Took the West? A: _{_ ”titles”: [”Querelle”, ”The Gal Who Took the West”], ”answer”: [”Frederick de Cordova”] _}_ QS: (select) [retrieve ~~o~~ dqa] When did Joseph M. Newman die? 

53 

Published as a conference paper at ICLR 2023 

- A: _{_ ”titles”: [”Joseph M. Newman”, ”Thulasi (1987 film)”], ”answer”: [”January 23, 2006”] _}_ 

QS: (select) [retrieve ~~o~~ dqa] When did Fred de Cordova die? 

- A: _{_ ”titles”: [”Fred de Cordova”, ”Le Masque de la Meduse”], ”answer”: [”September 15, 2001”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Riki Gal”, ”Twenty Plus Two”, ”Querelle”, ”The Gal Who Took the West”, ”Joseph M. Newman”, ”Thulasi (1987 film)”, ”Fred de Cordova”, ”Le Masque de la Meduse”]. Question: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two? 

- A: [”Twenty Plus Two”] 

- QS: [EOQ] 

- QC: Who is the grandchild of Krishna Shah (Nepalese Royal)? 

- QS: (select) [retrieve ~~o~~ dqa] Who is the child of Krishna Shah? 

- A: _{_ ”titles”: [”Ana Gruzinsky−Golitsyn”, ”Krishna Shah (Nepalese royal)”, ”Diana Weston”, ” Albina du Boisrouvray”], ”answer”: [”Rudra Shah”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] Who is the child of Rudra Shah? 

- A: _{_ ”titles”: [”Jim Ramel Kjellgren”, ”Constance Anne Herschel”, ”Rudra Shah”], ”answer”: [” Prithvipati Shah”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Ana Gruzinsky−Golitsyn”, ”Krishna Shah (Nepalese royal) ”, ”Diana Weston”, ”Albina du Boisrouvray”, ”Jim Ramel Kjellgren”, ”Constance Anne Herschel”, ”Rudra Shah”]. Question: Who is the grandchild of Krishna Shah (Nepalese Royal )? 

- A: [”Prithvipati Shah”] 

QS: [EOQ] 

- QC: Where did the director of film Maddalena (1954 Film) die? 

- QS: (select) [retrieve ~~o~~ dqa] Who is the director of the film Maddalena? 

- A: _{_ ”titles”: [”Ian Barry (director)”, ”Maddalena (1954 film)”, ”Brian Kennedy (gallery director) ”], ”answer”: [”Augusto Genina”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] Where did Augusto Genina die? 

- A: _{_ ”titles”: [”Dana Blankstein”, ”Augusto Genina”, ”Peter Levin”, ”Olav Aaraas”], ”answer”: [” Rome”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Ian Barry (director)”, ”Maddalena (1954 film)”, ”Brian Kennedy (gallery director)”, ”Dana Blankstein”, ”Augusto Genina”, ”Peter Levin”, ”Olav Aaraas”]. Question: Where did the director of film Maddalena (1954 Film) die? 

- A: [”Rome”] 

- QS: [EOQ] 

- QC: Who was born first out of Martin Hodge and Ivania Martinich? 

QS: (select) [retrieve ~~o~~ dqa] When was Martin Hodge born? 

- A: _{_ ”titles”: [”Greg A. Hill (artist)”, ”Martin Hodge”, ”John Allen (Oxford University cricketer)”], ”answer”: [”4 February 1959”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] When was Ivania Martinich born? 

- A: _{_ ”titles”: [”Tom Dickinson”, ”Ivania Martinich”, ”Wesley Barresi”, ”Henry Moore (cricketer) ”], ”answer”: [”25 July 1995”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Greg A. Hill (artist)”, ”Martin Hodge”, ”John Allen ( Oxford University cricketer)”, ”Tom Dickinson”, ”Ivania Martinich”, ”Wesley Barresi”, ” Henry Moore (cricketer)”]. Question: Who was born first out of Martin Hodge and Ivania Martinich? 

- A: [”Martin Hodge”] 

- QS: [EOQ] 

2wikimultihopqa: retrieve ~~o~~ dqa 

QC: Who is the director of the film Coolie No. 1 (1995 film)? QS: (select) [retrieve] Who is the director of the film Coolie No. 1 (1995 film)? 

A: [”Hanro Smitsman”, ”Coolie No. 1 (1995 film)”] 

54 

Published as a conference paper at ICLR 2023 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Hanro Smitsman”, ”Coolie No. 1 (1995 film)”]. Question: Who is the director of the film Coolie No. 1 (1995 film)? 

A: _{_ ”titles”: [”Hanro Smitsman”, ”Coolie No. 1 (1995 film)”], ”answer”: [”David Dhawan”] _}_ QS: [EOQ] 

QC: Who is the director of the film The Sensational Trial? 

QS: (select) [retrieve] Who is the director of the film The Sensational Trial? 

A: [”Rachel Feldman”, ”The Sensational Trial”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Rachel Feldman”, ”The Sensational Trial”]. Question: 

A: _{_ ”titles”: [”Rachel Feldman”, ”The Sensational Trial”], ”answer”: [”Karl Freund”] _}_ QS: [EOQ] 

QC: What is David Dhawan’s nationality? 

QS: (select) [retrieve] What is David Dhawan’s nationality? 

A: [”David Dhawan”, ”Brian Johnson (special effects artist)”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”David Dhawan”, ”Brian Johnson (special effects artist)”]. Question: What is David Dhawan’s nationality? 

A: _{_ ”titles”: [”David Dhawan”, ”Brian Johnson (special effects artist)”], ”answer”: [”India”] _}_ QS: [EOQ] 

QC: What is Karl Freund’s nationality? 

QS: (select) [retrieve] What is Karl Freund’s nationality? 

A: [”Karl Freund”, ”Ian Barry (director)”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Karl Freund”, ”Ian Barry (director)”]. Question: What is Karl Freund’s nationality? 

A: _{_ ”titles”: [”Karl Freund”, ”Ian Barry (director)”], ”answer”: [”Germany”] _}_ 

QS: [EOQ] 

QC: Who is the director of the film Twenty Plus? 

QS: (select) [retrieve] Who is the director of the film Twenty Plus? 

A: [”Riki Gal”, ”Twenty Plus Two”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Riki Gal”, ”Twenty Plus Two”]. Question: Who is the director of the film Twenty Plus? 

A: _{_ ”titles”: [”Riki Gal”, ”Twenty Plus Two”], ”answer”: [”Joseph M. Newman”] _}_ QS: [EOQ] 

QC: Who is the director of the film The Gal Who Took the West? QS: (select) [retrieve] Who is the director of the film The Gal Who Took the West? 

A: [”Querelle”, ”The Gal Who Took the West”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Querelle”, ”The Gal Who Took the West”]. Question: 

A: _{_ ”titles”: [”Querelle”, ”The Gal Who Took the West”], ”answer”: [”Frederick de Cordova”] _}_ QS: [EOQ] 

QC: When did Joseph M. Newman die? 

QS: (select) [retrieve] When did Joseph M. Newman die? 

A: [”Joseph M. Newman”, ”Thulasi (1987 film)”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Joseph M. Newman”, ”Thulasi (1987 film)”]. Question: When did Joseph M. Newman die? 

A: _{_ ”titles”: [”Joseph M. Newman”, ”Thulasi (1987 film)”], ”answer”: [”January 23, 2006”] _}_ QS: [EOQ] 

55 

Published as a conference paper at ICLR 2023 

## QC: When did Fred de Cordova die? 

- QS: (select) [retrieve] When did Fred de Cordova die? 

- A: [”Fred de Cordova”, ”Le Masque de la Meduse”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Fred de Cordova”, ”Le Masque de la Meduse”]. Question: When did Fred de Cordova die? 

- A: _{_ ”titles”: [”Fred de Cordova”, ”Le Masque de la Meduse”], ”answer”: [”September 15, 2001”] _}_ QS: [EOQ] 

## QC: Who is the child of Krishna Shah? 

- QS: (select) [retrieve] Who is the child of Krishna Shah? 

- A: [”Ana Gruzinsky−Golitsyn”, ”Krishna Shah (Nepalese royal)”, ”Diana Weston”, ”Albina du Boisrouvray”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Ana Gruzinsky−Golitsyn”, ”Krishna Shah (Nepalese royal )”, ”Diana Weston”, ”Albina du Boisrouvray”]. Question: Who is the child of Krishna Shah? 

- A: _{_ ”titles”: [”Ana Gruzinsky−Golitsyn”, ”Krishna Shah (Nepalese royal)”, ”Diana Weston”, ” Albina du Boisrouvray”], ”answer”: [”Rudra Shah”] _}_ 

- QS: [EOQ] 

## QC: Who is the child of Rudra Shah? 

- QS: (select) [retrieve] Who is the child of Rudra Shah? 

- A: [”Jim Ramel Kjellgren”, ”Constance Anne Herschel”, ”Rudra Shah”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Jim Ramel Kjellgren”, ”Constance Anne Herschel”, ” Rudra Shah”]. Question: Who is the child of Rudra Shah? 

- A: _{_ ”titles”: [”Jim Ramel Kjellgren”, ”Constance Anne Herschel”, ”Rudra Shah”], ”answer”: [” Prithvipati Shah”] _}_ 

- QS: [EOQ] 

## QC: Who is the director of the film Maddalena? 

QS: (select) [retrieve] Who is the director of the film Maddalena? 

- A: [”Ian Barry (director)”, ”Maddalena (1954 film)”, ”Brian Kennedy (gallery director)”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Ian Barry (director)”, ”Maddalena (1954 film)”, ”Brian Kennedy (gallery director)”]. Question: Who is the director of the film Maddalena? 

- A: _{_ ”titles”: [”Ian Barry (director)”, ”Maddalena (1954 film)”, ”Brian Kennedy (gallery director) ”], ”answer”: [”Augusto Genina”] _}_ 

QS: [EOQ] 

## QC: Where did Augusto Genina die? 

- QS: (select) [retrieve] Where did Augusto Genina die? 

- A: [”Dana Blankstein”, ”Augusto Genina”, ”Peter Levin”, ”Olav Aaraas”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Dana Blankstein”, ”Augusto Genina”, ”Peter Levin”, ” Olav Aaraas”]. Question: Where did Augusto Genina die? 

- A: _{_ ”titles”: [”Dana Blankstein”, ”Augusto Genina”, ”Peter Levin”, ”Olav Aaraas”], ”answer”: [” Rome”] _}_ 

- QS: [EOQ] 

QC: When was Martin Hodge born? 

- QS: (select) [retrieve] When was Martin Hodge born? 

- A: [”Greg A. Hill (artist)”, ”Martin Hodge”, ”John Allen (Oxford University cricketer)”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Greg A. Hill (artist)”, ”Martin Hodge”, ”John Allen ( Oxford University cricketer)”]. Question: When was Martin Hodge born? 

- A: _{_ ”titles”: [”Greg A. Hill (artist)”, ”Martin Hodge”, ”John Allen (Oxford University cricketer)”], ”answer”: [”4 February 1959”] _}_ 

- QS: [EOQ] 

56 

Published as a conference paper at ICLR 2023 

QC: When was Ivania Martinich born? 

QS: (select) [retrieve] When was Ivania Martinich born? 

A: [”Tom Dickinson”, ”Ivania Martinich”, ”Wesley Barresi”, ”Henry Moore (cricketer)”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Tom Dickinson”, ”Ivania Martinich”, ”Wesley Barresi”, ” Henry Moore (cricketer)”]. Question: When was Ivania Martinich born? 

A: _{_ ”titles”: [”Tom Dickinson”, ”Ivania Martinich”, ”Wesley Barresi”, ”Henry Moore (cricketer) ”], ”answer”: [”25 July 1995”] _}_ 

QS: [EOQ] 

2wikimultihopqa: 

singlehop ~~t~~ itleqa 

Wikipedia Title: David Dhawan 

_<_ hidden for brevity _>_ 

Wikipedia Title: Howard W. Koch _<_ hidden for brevity _>_ 

Wikipedia Title: The Sensational Trial _<_ hidden for brevity _>_ 

Wikipedia Title: Coolie No. 1 (1995 film) 

_<_ hidden for brevity _>_ 

Wikipedia Title: Karl Freund _<_ hidden for brevity _>_ 

Q: Who is the director of the film Coolie No. 1 (1995 film)? A: [”David Dhawan”] 

Q: Who is the director of the film The Sensational Trial? A: [”Karl Freund”] 

Q: What is David Dhawan’s nationality? A: [”India”] 

Q: What is Karl Freund’s nationality? A: [”Germany”] 

Wikipedia Title: Fred de Cordova _<_ hidden for brevity _>_ 

Wikipedia Title: Thulasi (1987 film) _<_ hidden for brevity _>_ 

Wikipedia Title: Joseph M. Newman _<_ hidden for brevity _>_ 

Wikipedia Title: The Gal Who Took the West _<_ hidden for brevity _>_ 

Wikipedia Title: Twenty Plus Two _<_ hidden for brevity _>_ 

Q: Who is the director of the film Twenty Plus? A: [”Joseph M. Newman”] 

Q: Who is the director of the film The Gal Who Took the West? 

57 

Published as a conference paper at ICLR 2023 

A: [”Frederick de Cordova”] 

Q: When did Joseph M. Newman die? A: [”January 23, 2006”] 

Q: When did Fred de Cordova die? A: [”September 15, 2001”] 

Wikipedia Title: Ana Gruzinsky−Golitsyn _<_ hidden for brevity _>_ 

Wikipedia Title: Rudra Shah _<_ hidden for brevity _>_ 

Wikipedia Title: Krishna Shah (Nepalese royal) _<_ hidden for brevity _>_ 

Q: Who is the child of Krishna Shah? A: [”Rudra Shah”] 

Q: Who is the child of Rudra Shah? A: [”Prithvipati Shah”] 

Wikipedia Title: Augusto Genina _<_ hidden for brevity _>_ 

Wikipedia Title: Ian Barry (director) _<_ hidden for brevity _>_ 

Wikipedia Title: Maddalena (1954 film) _<_ hidden for brevity _>_ 

Q: Who is the director of the film Maddalena? A: [”Augusto Genina”] 

Q: Where did Augusto Genina die? A: [”Rome”] 

Wikipedia Title: Martin Hodge _<_ hidden for brevity _>_ 

Wikipedia Title: Ivania Martinich _<_ hidden for brevity _>_ 

Wikipedia Title: Tom Dickinson _<_ hidden for brevity _>_ 

Q: When was Martin Hodge born? A: [”4 February 1959”] 

Q: When was Ivania Martinich born? A: [”25 July 1995”] 

2wikimultihopqa: multihop ~~t~~ itleqa (direct) 

Wikipedia Title: David Dhawan _<_ hidden for brevity _>_ 

58 

Published as a conference paper at ICLR 2023 

Wikipedia Title: Howard W. Koch _<_ hidden for brevity _>_ 

Wikipedia Title: The Sensational Trial _<_ hidden for brevity _>_ 

Wikipedia Title: Coolie No. 1 (1995 film) _<_ hidden for brevity _>_ 

Wikipedia Title: Karl Freund _<_ hidden for brevity _>_ 

Q: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality? 

A: [”no”] 

Wikipedia Title: Fred de Cordova _<_ hidden for brevity _>_ 

Wikipedia Title: Thulasi (1987 film) _<_ hidden for brevity _>_ 

Wikipedia Title: Joseph M. Newman _<_ hidden for brevity _>_ 

Wikipedia Title: The Gal Who Took the West _<_ hidden for brevity _>_ 

Wikipedia Title: Twenty Plus Two _<_ hidden for brevity _>_ 

Q: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two? A: [”Twenty Plus Two”] 

Wikipedia Title: Ana Gruzinsky−Golitsyn _<_ hidden for brevity _>_ 

Wikipedia Title: Rudra Shah _<_ hidden for brevity _>_ 

Wikipedia Title: Krishna Shah (Nepalese royal) _<_ hidden for brevity _>_ 

Q: Who is the grandchild of Krishna Shah (Nepalese Royal)? 

A: [”Prithvipati Shah”] 

Wikipedia Title: Augusto Genina _<_ hidden for brevity _>_ 

Wikipedia Title: Ian Barry (director) _<_ hidden for brevity _>_ 

Wikipedia Title: Maddalena (1954 film) _<_ hidden for brevity _>_ 

59 

Published as a conference paper at ICLR 2023 

Q: Where did the director of film Maddalena (1954 Film) die? A: [”Rome”] 

Wikipedia Title: Martin Hodge _<_ hidden for brevity _>_ 

Wikipedia Title: Ivania Martinich _<_ hidden for brevity _>_ 

Wikipedia Title: Tom Dickinson _<_ hidden for brevity _>_ 

Q: Who was born first out of Martin Hodge and Ivania Martinich? A: [”Martin Hodge”] 

2wikimultihopqa: multihop ~~t~~ itleqa (cot) 

Wikipedia Title: David Dhawan _<_ hidden for brevity _>_ 

Wikipedia Title: Howard W. Koch _<_ hidden for brevity _>_ 

Wikipedia Title: The Sensational Trial _<_ hidden for brevity _>_ 

Wikipedia Title: Coolie No. 1 (1995 film) _<_ hidden for brevity _>_ 

Wikipedia Title: Karl Freund _<_ hidden for brevity _>_ 

- Q: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality? 

- A: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan’s nationality is India. Karl Freund’s nationality is Germany. 

- Thus, they do not have the same nationality. So the answer is: no. 

Wikipedia Title: Fred de Cordova _<_ hidden for brevity _>_ 

Wikipedia Title: Thulasi (1987 film) _<_ hidden for brevity _>_ 

Wikipedia Title: Joseph M. Newman _<_ hidden for brevity _>_ 

Wikipedia Title: The Gal Who Took the West _<_ hidden for brevity _>_ 

Wikipedia Title: Twenty Plus Two 

_<_ hidden for brevity _>_ 

- Q: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two? 

- A: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty 

- Plus Two. So the answer is: Twenty Plus Two. 

60 

Published as a conference paper at ICLR 2023 

Wikipedia Title: Ana Gruzinsky−Golitsyn 

_<_ hidden for brevity _>_ 

Wikipedia Title: Rudra Shah _<_ hidden for brevity _>_ 

Wikipedia Title: Krishna Shah (Nepalese royal) 

_<_ hidden for brevity _>_ 

- Q: Who is the grandchild of Krishna Shah (Nepalese Royal)? 

A: Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is: Prithvipati Shah. 

Wikipedia Title: Augusto Genina 

_<_ hidden for brevity _>_ 

Wikipedia Title: Ian Barry (director) 

_<_ hidden for brevity _>_ 

Wikipedia Title: Maddalena (1954 film) 

_<_ hidden for brevity _>_ 

- Q: Where did the director of film Maddalena (1954 Film) die? 

- A: The film Maddalena is directed by Augusto Genina. Augusto Genina died in Rome. So the answer is: Rome. 

Wikipedia Title: Martin Hodge 

_<_ hidden for brevity _>_ 

Wikipedia Title: Ivania Martinich 

_<_ hidden for brevity _>_ 

Wikipedia Title: Tom Dickinson 

_<_ hidden for brevity _>_ 

- Q: Who was born first out of Martin Hodge and Ivania Martinich? 

- A: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is: Martin Hodge. 

musique ~~a~~ ns: decomp 

- QC: When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast? 

- QS: (select) [retrieve ~~o~~ dqa] What city is CIMI−FM is licensed to broadcast in? 

- A: _{_ ”titles”: [”KWMZ−FM”, ”CIMI−FM”, ”KSAO (FM)”, ”WTTL−FM”, ”KOLU”, ”WORW”, ” WLRX (FM)”], ”answer”: [”Quebec City”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] When did the first large winter carnival in Quebec City take place? 

- A: _{_ ”titles”: [”WRQY”, ”Quebec Winter Carnival”, ”KBCR−FM”], ”answer”: [”1894”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”KWMZ−FM”, ”CIMI−FM”, ”KSAO (FM)”, ”WTTL−FM ”, ”KOLU”, ”WORW”, ”WLRX (FM)”, ”WRQY”, ”Quebec Winter Carnival”, ”KBCR−FM 

   - ”]. Question: When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast? 

- A: [”1894”] 

QS: [EOQ] 

61 

Published as a conference paper at ICLR 2023 

- QC: What is the headquarters for the organization who sets the standards for ISO 21500? QS: (select) [retrieve ~~o~~ dqa] Who sets the standards for ISO 21500? 

- A: _{_ ”titles”: [”ISO 3166−2:BM”, ”ISO 3166−2:IS”, ”ISO 3166−2:IT”, ”ISO 3166−2:BB”, ”ISO 21500”, ”ISO 3166−2:ET”, ”ISO 3166−2:AT”, ”ISO 3166−2:CG”, ”ISO 22000”], ”answer”: [”International Organization for Standardization”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] Where is the headquarters for International Organization for Standardization? 

- A: _{_ ”titles”: [”International Organization for Standardization”, ”ISO 4031”, ”ISO 3166−2:CN”, ” Unicode”, ”ISO 3166−2:GH”, ”ISO 3166−2:AO”, ”ISO/TC 68”, ”ISO 7001”, ”ISO 3307”], ” answer”: [”Geneva”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”ISO 3166−2:BM”, ”ISO 3166−2:IS”, ”ISO 3166−2:IT”, ” ISO 3166−2:BB”, ”ISO 21500”, ”ISO 3166−2:ET”, ”ISO 3166−2:AT”, ”ISO 3166−2:CG”, ” ISO 22000”, ”International Organization for Standardization”, ”ISO 4031”, ”ISO 3166−2:CN ”, ”Unicode”, ”ISO 3166−2:GH”, ”ISO 3166−2:AO”, ”ISO/TC 68”, ”ISO 7001”, ”ISO 3307”]. Question: What is the headquarters for the organization who sets the standards for ISO 21500? 

- A: [”Geneva”] 

QS: [EOQ] 

- QC: How long is the US border with the country that borders the state where Finding Dory takes place? 

- QS: (select) [retrieve ~~o~~ dqa] In which state does Finding Dory take place? 

- A: _{_ ”titles”: [”Finding Dory”, ”Latvia”, ”List of countries that border only one other country”, ” Removal of Hungary’s border fence with Austria”], ”answer”: [”California”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] Which country shares a border with California? 

- A: _{_ ”titles”: [”Mexico−United States border”, ”Kingdom of Gera”, ”Pesticide”, ”Currie Cup”], ” answer”: [”Mexico”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] What is the length of the US border with Mexico? 

- A: _{_ ”titles”: [”Piscataqua River border dispute”, ”Share a Coke”, ”Mexico−United States border”, ”Hotel Arbez”], ”answer”: [”1,989 mi”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Finding Dory”, ”Latvia”, ”List of countries that border only one other country”, ”Removal of Hungary’s border fence with Austria”, ”Mexico− United States border”, ”Kingdom of Gera”, ”Pesticide”, ”Currie Cup”, ”Piscataqua River border dispute”, ”Share a Coke”, ”Mexico−United States border”, ”Hotel Arbez”]. Question: How long is the US border with the country that borders the state where Finding Dory takes place? 

- A: [”1,989 mi”] 

QS: [EOQ] 

- QC: When was Neville A. Stanton’s employer founded? 

- QS: (select) [retrieve ~~o~~ dqa] Who is the employer of Neville A. Stanton? 

- A: _{_ ”titles”: [”Harriot Stanton Blatch”, ”Robichaud v Canada (Treasury Board)”, ”The Peggy Neville Show”, ”Stanton Township, Champaign County, Illinois”, ”Women’s suffrage in the United States”, ”Neville A. Stanton”], ”answer”: [”University of Southampton”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] When was the University of Southampton founded? 

- A: _{_ ”titles”: [”Southampton”, ”Presley Neville”], ”answer”: [”1862”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Harriot Stanton Blatch”, ”Robichaud v Canada (Treasury Board)”, ”The Peggy Neville Show”, ”Stanton Township, Champaign County, Illinois”, ” Women’s suffrage in the United States”, ”Neville A. Stanton”, ”Southampton”, ”Presley Neville”]. Question: When was Neville A. Stanton’s employer founded? 

A: [”1862”] 

QS: [EOQ] 

QC: In which county was the birthplace of the Smoke in tha City performer? QS: (select) [retrieve ~~o~~ dqa] Who is the performer of Smoke in the City? 

62 

Published as a conference paper at ICLR 2023 

- A: _{_ ”titles”: [”Nicotine marketing”, ”Khabarovsky District”, ”Beyonce”, ”Asthma”, ”Smoke in tha City”], ”answer”: [”MC Eiht”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] What is the birthplace of MC Eiht? 

- A: _{_ ”titles”: [”Cherokee City, Arkansas”, ”MC Eiht”, ”Biysky District”], ”answer”: [”Compton”] _}_ 

- QS: (select) [retrieve ~~o~~ dqa] Compton is located in which county? 

- A: _{_ ”titles”: [”Vilnius County”, ”Biblioteca Ayacucho”, ”Compton, California”], ”answer”: [”Los Angeles County”] _}_ 

- QS: (select) [multihop ~~t~~ itleqa] Titles: [”Nicotine marketing”, ”Khabarovsky District”, ”Beyonce”, ”Asthma”, ”Smoke in tha City”, ”Cherokee City, Arkansas”, ”MC Eiht”, ”Biysky District”, ” Vilnius County”, ”Biblioteca Ayacucho”, ”Compton, California”]. Question: In which county was the birthplace of the Smoke in tha City performer? 

- A: [”Los Angeles County”] 

- QS: [EOQ] 

musique ~~a~~ ns: retrieve ~~o~~ dqa 

- QC: What city is CIMI−FM is licensed to broadcast in? 

- QS: (select) [retrieve] What city is CIMI−FM is licensed to broadcast in? 

- A: [”KWMZ−FM”, ”CIMI−FM”, ”KSAO (FM)”, ”WTTL−FM”, ”KOLU”, ”WORW”, ”WLRX ( FM)”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”KWMZ−FM”, ”CIMI−FM”, ”KSAO (FM)”, ”WTTL−FM ”, ”KOLU”, ”WORW”, ”WLRX (FM)”]. Question: What city is CIMI−FM is licensed to broadcast in? 

- A: _{_ ”titles”: [”KWMZ−FM”, ”CIMI−FM”, ”KSAO (FM)”, ”WTTL−FM”, ”KOLU”, ”WORW”, ” WLRX (FM)”], ”answer”: [”Quebec City”] _}_ 

- QS: [EOQ] 

QC: When did the first large winter carnival in Quebec City take place? 

- QS: (select) [retrieve] When did the first large winter carnival in Quebec City take place? 

- A: [”WRQY”, ”Quebec Winter Carnival”, ”KBCR−FM”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”WRQY”, ”Quebec Winter Carnival”, ”KBCR−FM”]. Question: When did the first large winter carnival in Quebec City take place? 

- A: _{_ ”titles”: [”WRQY”, ”Quebec Winter Carnival”, ”KBCR−FM”], ”answer”: [”1894”] _}_ QS: [EOQ] 

QC: Who sets the standards for ISO 21500? 

- QS: (select) [retrieve] Who sets the standards for ISO 21500? 

- A: [”ISO 3166−2:BM”, ”ISO 3166−2:IS”, ”ISO 3166−2:IT”, ”ISO 3166−2:BB”, ”ISO 21500”, ” ISO 3166−2:ET”, ”ISO 3166−2:AT”, ”ISO 3166−2:CG”, ”ISO 22000”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”ISO 3166−2:BM”, ”ISO 3166−2:IS”, ”ISO 3166−2:IT”, ” ISO 3166−2:BB”, ”ISO 21500”, ”ISO 3166−2:ET”, ”ISO 3166−2:AT”, ”ISO 3166−2:CG”, ” ISO 22000”]. Question: Who sets the standards for ISO 21500? 

- A: _{_ ”titles”: [”ISO 3166−2:BM”, ”ISO 3166−2:IS”, ”ISO 3166−2:IT”, ”ISO 3166−2:BB”, ”ISO 21500”, ”ISO 3166−2:ET”, ”ISO 3166−2:AT”, ”ISO 3166−2:CG”, ”ISO 22000”], ”answer”: [”International Organization for Standardization”] _}_ 

- QS: [EOQ] 

QC: Where is the headquarters for International Organization for Standardization? 

- QS: (select) [retrieve] Where is the headquarters for International Organization for Standardization ? 

- A: [”International Organization for Standardization”, ”ISO 4031”, ”ISO 3166−2:CN”, ”Unicode”, ”ISO 3166−2:GH”, ”ISO 3166−2:AO”, ”ISO/TC 68”, ”ISO 7001”, ”ISO 3307”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”International Organization for Standardization”, ”ISO 4031”, ”ISO 3166−2:CN”, ”Unicode”, ”ISO 3166−2:GH”, ”ISO 3166−2:AO”, ”ISO/TC 68”, 

   - ”ISO 7001”, ”ISO 3307”]. Question: Where is the headquarters for International Organization for Standardization? 

63 

Published as a conference paper at ICLR 2023 

- A: _{_ ”titles”: [”International Organization for Standardization”, ”ISO 4031”, ”ISO 3166−2:CN”, ” Unicode”, ”ISO 3166−2:GH”, ”ISO 3166−2:AO”, ”ISO/TC 68”, ”ISO 7001”, ”ISO 3307”], ” answer”: [”Geneva”] _}_ 

- QS: [EOQ] 

- QC: In which state does Finding Dory take place? 

- QS: (select) [retrieve] In which state does Finding Dory take place? 

- A: [”Finding Dory”, ”Latvia”, ”List of countries that border only one other country”, ”Removal of Hungary’s border fence with Austria”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Finding Dory”, ”Latvia”, ”List of countries that border only one other country”, ”Removal of Hungary’s border fence with Austria”]. Question: In which state does Finding Dory take place? 

- A: _{_ ”titles”: [”Finding Dory”, ”Latvia”, ”List of countries that border only one other country”, ” Removal of Hungary’s border fence with Austria”], ”answer”: [”California”] _}_ 

- QS: [EOQ] 

- QC: Which country shares a border with California? 

- QS: (select) [retrieve] Which country shares a border with California? 

- A: [”Mexico−United States border”, ”Kingdom of Gera”, ”Pesticide”, ”Currie Cup”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Mexico−United States border”, ”Kingdom of Gera”, ” Pesticide”, ”Currie Cup”]. Question: Which country shares a border with California? 

- A: _{_ ”titles”: [”Mexico−United States border”, ”Kingdom of Gera”, ”Pesticide”, ”Currie Cup”], ” answer”: [”Mexico”] _}_ 

- QS: [EOQ] 

- QC: What is the length of the US border with Mexico? 

- QS: (select) [retrieve] What is the length of the US border with Mexico? 

- A: [”Piscataqua River border dispute”, ”Share a Coke”, ”Mexico−United States border”, ”Hotel Arbez”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Piscataqua River border dispute”, ”Share a Coke”, ” Mexico−United States border”, ”Hotel Arbez”]. Question: What is the length of the US border with Mexico? 

- A: _{_ ”titles”: [”Piscataqua River border dispute”, ”Share a Coke”, ”Mexico−United States border”, ”Hotel Arbez”], ”answer”: [”1,989 mi”] _}_ 

- QS: [EOQ] 

- QC: Who is the employer of Neville A. Stanton? 

- QS: (select) [retrieve] Who is the employer of Neville A. Stanton? 

- A: [”Harriot Stanton Blatch”, ”Robichaud v Canada (Treasury Board)”, ”The Peggy Neville Show ”, ”Stanton Township, Champaign County, Illinois”, ”Women’s suffrage in the United States”, ”Neville A. Stanton”] 

- QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Harriot Stanton Blatch”, ”Robichaud v Canada (Treasury Board)”, ”The Peggy Neville Show”, ”Stanton Township, Champaign County, Illinois”, ” Women’s suffrage in the United States”, ”Neville A. Stanton”]. Question: Who is the employer of Neville A. Stanton? 

- A: _{_ ”titles”: [”Harriot Stanton Blatch”, ”Robichaud v Canada (Treasury Board)”, ”The Peggy Neville Show”, ”Stanton Township, Champaign County, Illinois”, ”Women’s suffrage in the United States”, ”Neville A. Stanton”], ”answer”: [”University of Southampton”] _}_ 

- QS: [EOQ] 

QC: When was the University of Southampton founded? 

QS: (select) [retrieve] When was the University of Southampton founded? A: [”Southampton”, ”Presley Neville”] 

64 

Published as a conference paper at ICLR 2023 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Southampton”, ”Presley Neville”]. Question: When was the University of Southampton founded? 

A: _{_ ”titles”: [”Southampton”, ”Presley Neville”], ”answer”: [”1862”] _}_ 

QS: [EOQ] 

QC: Who is the performer of Smoke in the City? 

QS: (select) [retrieve] Who is the performer of Smoke in the City? 

A: [”Nicotine marketing”, ”Khabarovsky District”, ”Beyonce”, ”Asthma”, ”Smoke in tha City”] QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Nicotine marketing”, ”Khabarovsky District”, ”Beyonce”, 

”Asthma”, ”Smoke in tha City”]. Question: Who is the performer of Smoke in the City? 

A: _{_ ”titles”: [”Nicotine marketing”, ”Khabarovsky District”, ”Beyonce”, ”Asthma”, ”Smoke in tha City”], ”answer”: [”MC Eiht”] _}_ 

QS: [EOQ] 

QC: What is the birthplace of MC Eiht? 

QS: (select) [retrieve] What is the birthplace of MC Eiht? 

A: [”Cherokee City, Arkansas”, ”MC Eiht”, ”Biysky District”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Cherokee City, Arkansas”, ”MC Eiht”, ”Biysky District”]. Question: What is the birthplace of MC Eiht? 

A: _{_ ”titles”: [”Cherokee City, Arkansas”, ”MC Eiht”, ”Biysky District”], ”answer”: [”Compton”] _}_ QS: [EOQ] 

QC: Compton is located in which county? 

QS: (select) [retrieve] Compton is located in which county? 

A: [”Vilnius County”, ”Biblioteca Ayacucho”, ”Compton, California”] 

QS: (select) [singlehop ~~t~~ itleqa] Titles: [”Vilnius County”, ”Biblioteca Ayacucho”, ”Compton, California”]. Question: Compton is located in which county? 

A: _{_ ”titles”: [”Vilnius County”, ”Biblioteca Ayacucho”, ”Compton, California”], ”answer”: [”Los Angeles County”] _}_ 

QS: [EOQ] 

musique ~~a~~ ns: singlehop ~~t~~ itleqa 

Wikipedia Title: CIMI−FM _<_ hidden for brevity _>_ 

Wikipedia Title: Quebec Winter Carnival _<_ hidden for brevity _>_ 

Wikipedia Title: KELD−FM _<_ hidden for brevity _>_ 

Q: What city is CIMI−FM is licensed to broadcast in? 

A: [”Quebec City”] 

Q: When did the first large winter carnival in Quebec City take place? A: [”1894”] 

Wikipedia Title: ISO/TC 68 _<_ hidden for brevity _>_ 

Wikipedia Title: ISO 21500 _<_ hidden for brevity _>_ Wikipedia Title: ISO 3166−2:GH _<_ hidden for brevity _>_ 

65 

Published as a conference paper at ICLR 2023 

Q: Who sets the standards for ISO 21500? 

A: [”International Organization for Standardization”] 

Q: Where is the headquarters for International Organization for Standardization? A: [”Geneva”] 

Wikipedia Title: Mexico−United States border _<_ hidden for brevity _>_ 

Wikipedia Title: Mexico−United States border _<_ hidden for brevity _>_ 

Wikipedia Title: Finding Dory _<_ hidden for brevity _>_ 

Wikipedia Title: India _<_ hidden for brevity _>_ 

Q: In which state does Finding Dory take place? A: [”California”] 

Q: Which country shares a border with California? A: [”Mexico”] 

Q: What is the length of the US border with Mexico? A: [”1,989 mi”] 

Wikipedia Title: Southampton 

_<_ hidden for brevity _>_ 

Wikipedia Title: Jonathan Stanton _<_ hidden for brevity _>_ 

Wikipedia Title: Neville A. Stanton _<_ hidden for brevity _>_ 

Q: Who is the employer of Neville A. Stanton? A: [”University of Southampton”] 

Q: When was the University of Southampton founded? A: [”1862”] 

Wikipedia Title: Compton, California _<_ hidden for brevity _>_ 

Wikipedia Title: Smoke in tha City _<_ hidden for brevity _>_ 

Wikipedia Title: MC Eiht _<_ hidden for brevity _>_ 

Wikipedia Title: Eastern Bengal and Assam _<_ hidden for brevity _>_ 

Q: Who is the performer of Smoke in the City? 

66 

Published as a conference paper at ICLR 2023 

A: [”MC Eiht”] 

Q: What is the birthplace of MC Eiht? A: [”Compton”] 

Q: Compton is located in which county? 

A: [”Los Angeles County”] 

musique ~~a~~ ns: multihop ~~t~~ itleqa (direct) 

Wikipedia Title: CIMI−FM 

_<_ hidden for brevity _>_ 

Wikipedia Title: Quebec Winter Carnival _<_ hidden for brevity _>_ 

Wikipedia Title: KELD−FM 

_<_ hidden for brevity _>_ 

Q: When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast? 

A: [”1894”] 

Wikipedia Title: ISO/TC 68 

_<_ hidden for brevity _>_ 

Wikipedia Title: ISO 21500 

_<_ hidden for brevity _>_ 

Wikipedia Title: ISO 3166−2:GH 

_<_ hidden for brevity _>_ 

Q: What is the headquarters for the organization who sets the standards for ISO 21500? A: [”Geneva”] 

Wikipedia Title: Mexico−United States border _<_ hidden for brevity _>_ 

Wikipedia Title: Mexico−United States border 

_<_ hidden for brevity _>_ 

Wikipedia Title: Finding Dory _<_ hidden for brevity _>_ 

Wikipedia Title: India _<_ hidden for brevity _>_ 

Q: How long is the US border with the country that borders the state where Finding Dory takes place? 

A: [”1,989 mi”] 

Wikipedia Title: Southampton _<_ hidden for brevity _>_ 

Wikipedia Title: Jonathan Stanton _<_ hidden for brevity _>_ 

67 

Published as a conference paper at ICLR 2023 

Wikipedia Title: Neville A. Stanton 

_<_ hidden for brevity _>_ 

Q: When was Neville A. Stanton’s employer founded? A: [”1862”] 

Wikipedia Title: Compton, California _<_ hidden for brevity _>_ 

Wikipedia Title: Smoke in tha City _<_ hidden for brevity _>_ 

Wikipedia Title: MC Eiht _<_ hidden for brevity _>_ 

Wikipedia Title: Eastern Bengal and Assam _<_ hidden for brevity _>_ 

Q: In which county was the birthplace of the Smoke in tha City performer? A: [”Los Angeles County”] 

musique ~~a~~ ns: multihop ~~t~~ itleqa (cot) 

Wikipedia Title: CIMI−FM _<_ hidden for brevity _>_ 

Wikipedia Title: Quebec Winter Carnival 

_<_ hidden for brevity _>_ 

Wikipedia Title: KELD−FM 

_<_ hidden for brevity _>_ 

- Q: When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast? 

- A: CIMI−FM is licensed to broadcast in Quebec City. The first large winter carnival in Quebec City took place in 1894. So the answer is: 1894. 

Wikipedia Title: ISO/TC 68 

_<_ hidden for brevity _>_ 

Wikipedia Title: ISO 21500 

_<_ hidden for brevity _>_ 

Wikipedia Title: ISO 3166−2:GH 

_<_ hidden for brevity _>_ 

- Q: What is the headquarters for the organization who sets the standards for ISO 21500? 

- A: The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva. So the answer is: Geneva. 

Wikipedia Title: Mexico−United States border 

_<_ hidden for brevity _>_ 

Wikipedia Title: Mexico−United States border _<_ hidden for brevity _>_ 

68 

Published as a conference paper at ICLR 2023 

Wikipedia Title: Finding Dory 

_<_ hidden for brevity _>_ 

Wikipedia Title: India 

_<_ hidden for brevity _>_ 

- Q: How long is the US border with the country that borders the state where Finding Dory takes place? 

- A: Finding Dory is supposed to take place in California. The country that shares a border with California is Mexico. The length of the us border with Mexico is 1,989 mi. So the answer is: 1,989 mi. 

Wikipedia Title: Southampton 

_<_ hidden for brevity _>_ 

Wikipedia Title: Jonathan Stanton 

_<_ hidden for brevity _>_ 

Wikipedia Title: Neville A. Stanton 

_<_ hidden for brevity _>_ 

- Q: When was Neville A. Stanton’s employer founded? 

- A: The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862. 

Wikipedia Title: Compton, California 

_<_ hidden for brevity _>_ 

Wikipedia Title: Smoke in tha City 

_<_ hidden for brevity _>_ 

Wikipedia Title: MC Eiht 

_<_ hidden for brevity _>_ 

Wikipedia Title: Eastern Bengal and Assam 

_<_ hidden for brevity _>_ 

- Q: In which county was the birthplace of the Smoke in tha City performer? 

- A: The performer of Smoke in tha City is MC Eiht. MC Eiht’s birthplace is Compton. Compton is located in the county of Los Angeles County. So the answer is: Los Angeles County. 

69 

