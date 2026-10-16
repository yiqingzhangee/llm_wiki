Under review as a conference paper at ICLR 2024 

## - EVALUATING THE ZERO SHOT ROBUSTNESS OF - INSTRUCTION TUNED LANGUAGE MODELS 

## **Anonymous authors** 

Paper under double-blind review 

## ABSTRACT 

_Instruction fine-tuning_ has recently emerged as a promising approach for improving the zero-shot capabilities of Large Language Models (LLMs) on new tasks. This technique has shown particular strength in improving the performance of modestly sized LLMs, sometimes inducing performance competitive with much larger model variants. In this paper we ask two questions: (1) How sensitive are instruction-tuned models to the particular phrasings of instructions, and, (2) How can we make them more robust to such natural language variation? To answer the former, we collect a set of 319 English instructions manually written by NLP practitioners for over 80 unique tasks included in widely used benchmarks, and we evaluate the variance and average performance of these instructions as compared to instruction phrasings observed during instruction fine-tuning. We find that using novel (unobserved) but appropriate instruction phrasings consistently degrades model performance, sometimes substantially so. Further, such natural instructions yield a wide variance in downstream performance, despite their semantic equivalence. Put another way, instruction-tuned models are not especially robust to instruction re-phrasings. We propose a simple method to mitigate this issue by introducing “soft prompt” embedding parameters and optimizing these to maximize the similarity between representations of semantically equivalent instructions. We show that this method consistently improves the robustness of instruction-tuned models. 

## 1 INTRODUCTION 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0001-07.png)


**----- Start of picture text -----**<br>
Multi-task instruction-tuning<br>In the following, please note …<br>Extract all names from the below text …<br>Summarize the following text<br>Given the task definition and input, reply with output. In this task, you are given a  Instruction-tuned LLM<br>multiple-choice question and you have to pick the correct option.<br>Answer with option indexes (i.e., "A", "B", "C", and "D")<br>A<br>The numen of Augustus referred to which of the following characteristics?<br>(A) Divine power (B) Sexual virility (C) Military acumen (D) Philosophical intellect ↑ Training<br>Given the task definition and input, reply with output. In this task, you are given a multiple-choice  Testing<br>question and you have to pick the correct option.<br>{ Answer with option indexes (i.e., "A", "B", "C", and "D") C<br>Deserts are in extremely dry environments, so liquids will naturally be<br>(A) rainbow (B) plentiful (C) meager (D) talented<br>A<br>The following is a multiple-choice question that requires expert-level domain knowledge.<br>Please select the correct answer to the question below from the options “A”, “B”, “C”, or “D”<br>{ after carefully examining each answer.<br>Deserts are in extremely dry environments, so liquids will naturally be<br>(A) rainbow (B) plentiful (C) meager (D) talented<br>…<br>Many tasks<br>Observed  instruction<br>Unobserved  instruction<br>↑<br>**----- End of picture text -----**<br>


Figure 1: How well do models trained on instruction-tuning datasets generalize to novel instructions (unobserved in training)? Our analysis suggests that they do not do so very well. Shown is a case where pairing an example with an observed instruction yields the correct output, while providing a distinct but equivalent instruction produces an incorrect response. We analyze this fragility in-depth, and based on this analysis we propose and evaluate a simple method that mitigates it. 

Large Language Models (LLMs) now dominate NLP, in part because they enable zero- and few-shot adaptation to new tasks via _prompting_ (Brown et al., 2020; Chowdhery et al., 2022; Hoffmann et al., 2022; Zeng et al., 2022). Recent work has shown the promise of _instruction-tuning_ , or fine-tuning 

1 

Under review as a conference paper at ICLR 2024 

LLMs with natural language instructions. This improves LLM performance in zero- and fewshot settings, sometimes dramatically, especially “mid-sized” models (Chung et al., 2022; Ouyang et al., 2022). For example, the instruction-tuned Flan-T5-XL (3B parameters; Chung et al. 2022) outperforms GPT-3 (175B) on several benchmarks, despite being dramatically smaller. Similarly, Alpaca (Taori et al., 2023)—an instruction-tuned LLaMa-7B Touvron et al. (2023b)—bests GPT-3 on multiple NLP tasks. This has motivated curation of instruction-augmented task collections (Wang et al., 2022b; Wei et al., 2021), and research into improving instruction-tuning (Longpre et al., 2023; Xu et al., 2022; Sanh et al., 2021). In this work, we ask: How sensitive are instruction-tuned LMs to natural variations in instruction phrasings? This is important given that the primary motivation of instruction-tuning is to permit zero-shot adaptation to new tasks via natural language: If models are overly sensitive to particular phrasings of instructions, this may greatly limit their utility in practice. 

Prior work has established that LLMs do not seem to intuitively “understand” prompts (Webson & Pavlick, 2022; Jang et al., 2023; Zhang et al., 2023a), but these efforts did not focus on instructiontuned models and their sensitivity to plausible (appropriate) instruction phrasings. Recent, contemporaneous work (Gu et al., 2023) investigated the robustness of instruction-tuned models, and found that instruction-tuned T5 (Raffel et al., 2020) is robust to instruction perturbations in few-shot settings, but less so in zero-shot application. We contribute a more in-depth analysis of this phenomenon across a much wider set of instruction-tuned models and benchmarks. Based on this analysis, we also introduce and evaluate a method for improving the robustness of such models. 

We collect a large set of task instructions manually composed by NLP researchers; these are valid instructions but distinct from those used in instruction tuning corpora. We then assess the performance of LLMs fine-tuned on different instruction corpora when provided novel (but appropriate) instructions on two large benchmarks: MMLU (Hendrycks et al., 2020) and BBL (Srivastava et al., 2022). We find that using novel instructions in zero-shot application degrades accuracy considerably (Figure 1). For example, for Flan-T5 XXL, using instructions not observed in training but which are appropriate for tasks leads to a 6.9 point drop in performance on average across benchmarks. 

Our **main contributions** are summarized as follows. (1) We perform a comprehensive and in-depth analysis of the robustness of instruction-tuned LLMs across three “families” of such models (Flan-T5, Alpaca, and T0) using large benchmarks. For this we collect a large set of new task instructions composed by researchers in NLP; we release this dataset to facilitate additional work on instruction robustness. We observe substantial performance degradation when using “novel” (unseen in training) instructions. (2) We propose a simple method to improve robustness by encouraging LLMs to induce similar representations for semantically equivalent instructions; this yields promising results. 

## 2 RELATED WORK 

**Multitask learning and instruction-tuning** . Training a single text-to-text model capable of handling arbitrary queries has been an aspiration in NLP for at least half a decade. Prior to prompting and instructing LLMs, there were efforts to unify disparate tasks by reframing them as instances of general _question answering_ (McCann et al., 2018; Khashabi et al., 2020; Keskar et al., 2019). Recent efforts have focussed on compiling and fine-tuning LLMs on corpora comprising diverse tasks with associated natural language instructions (Wei et al., 2021; Mishra et al., 2021; Sanh et al., 2021), i.e., _instruction-tuning_ . For example, Wang et al. (2022b) compiles over 1600 tasks and enriches these with both instructions and negative examples. Similarly, the recently released OPT-IML Bench Iyer et al. (2022) comprises 2000 NLP tasks. The Flan 2022 task collection Longpre et al. (2023) additionally features _Chain-of-Thought_ (CoT) style “reasoning” chains in instruction templates. 

These meta-resources—collections of instructions, tasks, and samples—have facilitated the training of instruction-tuned model families such as Flan-T5, Flan-PaLM, and OPT-IML (Iyer et al., 2022). 

**Evaluating prompting and instruction capabilities** . Instructions can be seen as a special sort of model prompting, which a few recent efforts have critically evaluated. Webson & Pavlick (2022) asked whether models meaningfully “understand” prompts, finding that they largely do not: Performance is often unaffected when irrelevant and misleading prompts are provided. In follow up work, Jang et al. (2023) evaluated performance on negated prompts and observed an “inverse-scaling” phenomenon where performance seems to degrade with model size. PROMPTBENCH (Zhu et al., 2023) provides a set of automatically generated datasets to assess the adversarial robustness of instruction-tuned LLMs. Similarly, Li et al. (2023) evaluated LLMs under adversarial instruction “attacks”. 

2 

Under review as a conference paper at ICLR 2024 

Other work has attempted to characterize when and how _in-context learning_ (ICL)—i.e., including a few examples in prompts—works (Min et al., 2022; Wang et al., 2023; Dai et al., 2022; Akyürek et al., 2022; Yu et al., 2022). ICL is a form of prompting orthogonal to the present effort, as we are primarily interested in the zero-shot adaptability of instruction-tuned LLMs. 

In work contemporaneous to ours, Gu et al. (2023) investigated how robust instruction-tuned models are to instruction perturbations (e.g., dropping words). This is qualitatively in line with our findings. Our work differs in important ways from this coincident research: (1) We provide a much more comprehensive analysis of robustness: Gu et al. (2023) considered _only_ T5 instruction-tuned on a single dataset, whereas we evaluate three LLM families (and varying sizes of each) using five instruction tuning datasets, and we evaluate over 80 test tasks in all (Gu et al. 2023 considered only 12); from this we identify representational similarity as a source of fragility. (2) On the basis of this analysis, we propose and evaluate a new approach to _improving_ the robustness of instruction-tuned models; Gu et al. (2023) offered no mechanism to improve robustness. 

**Improving instruction-tuning** . Past work has also sought to improve instruction-tuning in various ways. One means to do so is to instruction tune based on human feedback (Ouyang et al., 2022; Glaese et al., 2022; Bai et al., 2022; Nakano et al., 2021; Zhang et al., 2023b). This tends to improve open-ended model responses but degrade performance on downstream tasks. Another strategy is to leverage existing resources to automatically generate instruction-tuning datasets at scale. For example, Wang et al. (2022a) use LLMs to generate instructions, inputs, and outputs and use these to improve their own instruction-following capabilities. In a similarly meta vein, Zhou and colleagues 2022 propose using LLMs to engineer prompts. Finally, Ye et al. (2022) propose “flipping” the standard task by tasking LLMs with generating _instructions_ , given an input and label. 

## 3 INSTRUCTION DATASETS 

**Evaluation benchmarks** . We evaluate a set of instruction-tuned models on two large benchmarks— MMLU (Hendrycks et al., 2020) and BIG-BENCH (Srivastava et al., 2022)—and summarization datasets. MMLU is a multiple-choice question-answering benchmark comprising 57 tasks requiring expert knowledge. BIG-BENCH is a collaboratively built benchmark of 204 diverse tasks from various domains; we use the BIG-BENCH LITE subset, and include only QA, multi-class, binary classification tasks, and translation datasets, yielding 15 tasks in all. 

**Collecting new instructions from NLP researchers** . We aim to evaluate instruction-tuned models when they are provided instructions that are appropriate for a given task but superficially different from instructions observed in training. To this end we enlisted NLP researchers (graduate students) to compose novel instructions for the tasks considered; these particular instruction phrasings were therefore _unobserved_ during instruction fine-tuning. 

Briefly, we recruited 36 NLP graduate students working in NLP. All had at least some experience with instruction-tuned models and the downstream tasks included in the evaluation benchmarks. For all tasks in BBL and MMLU, 12 graduate students wrote one instruction they would use for zero-shot inference with an instruction-tuned model. We provide additional details on this instruction collection process in Appendix J. We will release all 319 instructions collected for reproducibility and to facilitate further research on instruction-tuned model robustness. 

## 4 EVALUATING THE ROBUSTNESS OF INSTRUCTION-TUNED LLMS 

## 4.1 MODELS AND DATA 

We conduct experiments with publicly accessible model variants trained over three instruction collections (which provide _observed_ task instructions): P3 (Sanh et al., 2021), Flan-2022 (Chung et al., 2022), and Alpaca (Taori et al., 2023). We do not include Llama2-Chat (Touvron et al., 2023b) because its instruction-tuning data is not available. But we do offer a comparison between LLama2-Alpaca and LLama-Alpaca in Appendix D (which suggests that better pre-training does not markedly improve robustness). We manually identified all instructions that correspond to the following task types: (a) Multiple-choice question answering (QA); (b) Binary classification (BC); (c) Multi-class classification (MC); (d) Summarization and; (e) Translation. 

3 

Under review as a conference paper at ICLR 2024 

||In this task, you are given a multiple-choice question and you have to pick the|
|---|---|
|QA|correct option. Answer with option indexes (i.e., "A", "B", "C", and "D").|
||Q:{question}A.{choiceA}B.{choiceB}C.{choiceC}D.{choiceD}|
|MC|Pick one category for the following text. The options are -{options}{text}|
|BC|{paragraph}Choose your answer: According to the above paragraph, the<br>question "{question}" is "{response}"?|



Table 1: Examples of observed instructions we collected for three general types of tasks. 

|OBSERVEDINSTRUCTIONS<br>_Instruction Type_<br>QA<br>MC<br>BC<br>Sum<br>Trans<br>Flan<br>50<br>35<br>18<br>37<br>14<br>Alpaca<br>20<br>20<br>11<br>20<br>10<br>P3<br>13<br>8<br>7<br>24<br>-|UNOBSERVEDINSTRUCTIONS|
|---|---|
||Number of tasks<br>1<br>15<br>Instructionsper task<br>20<br>10|
||Total instructions<br>20<br>140|



Table 2: Counts of instruction phrasings (unobserved and observed) we use for evaluations. 

To evaluate model robustness to instruction phrasings, we use two large benchmarks: MMLU and BIG-BENCH LITE (BBL) along with the acquired set of novel instructions described above. 

We include all 57 tasks from MMLU and 14 of 24 tasks from BBL.[1] We use the same instructions for all tasks in the same category, taken from the instruction tuning datasets associated with each model. These instructions are general, e.g., in the case of classification they request that the model consider an example with respect to categorization criteria and label space provided by the instance, and select an appropriate category (examples in Table 1). One can “mix-and-match” such instructions so long as they are appropriate for the task type. 

## 4.2 MAIN RESULTS 

We present the main aggregated analysis results in Figure 2 and Table 3. The take-away here is that using instructions unobserved in training—but manually composed for the task at hand and so semantically appropriate—leads to considerable degradation in performance: On average, unobserved instructions reduce accuracy by over five points across models considered. Table 3 reports results disaggregated by task type; we observe that classification tasks are most harmed by use of novel instructions. We provide additional, more granular (dataset-level) results in Appendix K. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0004-09.png)


**----- Start of picture text -----**<br>
60 -2.2 Observed<br>Unobserved Observed<br>55 -3.1 7575 BBL BC BBL MC Unobserved<br>6565<br>50 -2.8 55455545<br>3535<br>45 -5.2 25152515<br>757575 MMLU BBL QA<br>40 -5.6 655565556555<br>454545<br>35 353535<br>252525<br>151515<br>30 10 [7] 10 [8] 10 [9] 10 [7] 10 [8] 10 [9]<br>Flan-T5-3B Flan-T5-11B Alpaca-7B Alpaca-13B T0++ 11B Number of Parameters<br>Performance<br>Performance<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0004-10.png)


**----- Start of picture text -----**<br>
(a) Average zero-shot performance over all tasks<br>when using observed and unobserved instructions.<br>**----- End of picture text -----**<br>


- (b) Performances of Flan-T5 using observed and unobserved instructions as a function of model size. 

Figure 2: Using novel but valid instructions at test time (phrasings unobserved in training) consistently degrades the performance of instruction-tuned LLMs (a). Scale does not necessarily fix this (b). 

We present results for NLG tasks (translation and summarization) in Table 14. For these we use ROUGE (Lin, 2004), BLEU (Papineni et al., 2002), and BERTScore (Zhang et al., 2019) as automatic 

> 1We exclude 4 BBL exact-matching tasks, and 5 including tokens not recognized by T5 (e.g., emojis). 

4 

Under review as a conference paper at ICLR 2024 

|**Model**<br>MMLU<br>BBL-QA<br>BBL-BC<br>BBL-MC<br>Avg.<br>Std.<br>Avg.<br>Std.<br>Avg.<br>Std.<br>Avg.<br>Std.|**Overall**<br>Avg.<br>Std.|
|---|---|
|Flan-T5-3B<br>OBSERVED<br>**48.1** (_±_0_._3)<br>**59.0** (_±_2_._1)<br>**66.5** (_±_3_._8)<br>**55.6** (_±_0_._7)<br>UNOBSERVED<br>47_._5 (_±_0_._9)<br>56_._0 (_±_7_._3)<br>61_._1 (_±_6_._9)<br>52_._1 (_±_5_._4)<br>**Performance**∆<br>_↓_0_._6<br>_↓_3_._0<br>_↓_5_._5<br>_↓_3_._5<br>Alpaca-7B<br>OBSERVED<br>**41.9** (_±_0_._6)<br>**48.6** (_±_2_._8)<br>**53.8** (_±_3_._4)<br>**32.1** (_±_2_._2)<br>UNOBSERVED<br>39_._7 (_±_2_._2)<br>45_._3 (_±_6_._5)<br>52_._4 (_±_6_._5)<br>16_._4 (_±_3_._5)<br>**Performance**∆<br>_↓_2_._2<br>_↓_3_._3<br>_↓_1_._4<br>_↓_15_._7<br>T0++ 11B<br>OBSERVED<br>48_._3 (_±_0_._9)<br>54_._1 (_±_4_._1)<br>**66.1** (_±_2_._1)<br>**42.0** (_±_2_._1)<br>UNOBSERVED<br>**48.5** (_±_0_._9)<br>**54.7** (_±_3_._7)<br>54_._7 (_±_4_._3)<br>41_._4 (_±_2_._4)<br>**Performance**∆<br>_↑_0_._2<br>_↑_0_._7<br>_↓_11_._4<br>_↓_0_._6<br>Flan-T5-11B<br>OBSERVED<br>**53.2** (_±_0_._2)<br>**67.9** (_±_1_._8)<br>**65.6** (_±_6_._0)<br>**58.7** (_±_0_._5)<br>UNOBSERVED<br>52_._7 (_±_0_._8)<br>64_._6 (_±_8_._5)<br>63_._6 (_±_6_._1)<br>55_._9 (_±_5_._5)<br>**Performance**∆<br>_↓_0_._5<br>_↓_3_._4<br>_↓_2_._0<br>_↓_2_._8<br>Alpaca-13B<br>OBSERVED<br>**47.8** (_±_0_._5)<br>**53.9** (_±_2_._2)<br>**57.9** (_±_4_._8)<br>**36.7** (_±_1_._8)<br>UNOBSERVED<br>47_._0 (_±_0_._8)<br>51_._7 (_±_5_._7)<br>54_._1 (_±_5_._6)<br>22_._7 (_±_7_._5)<br>**Performance**∆<br>_↓_0_._9<br>_↓_2_._2<br>_↓_3_._8<br>_↓_14_._0|**57.3** (_±_1_._7)<br>54_._2 (_±_5_._1)<br>_↓_3_._1_†_<br>**44.1** (_±_2_._3)<br>38_._5 (_±_4_._7)<br>_↓_5_._6_†_<br>**52.6** (_±_2_._3)<br>49_._8 (_±_2_._8)<br>_↓_2_._8<br>**61.4** (_±_2_._1)<br>59_._2 (_±_5_._2)<br>_↓_2_._2_†_<br>**49.1** (_±_2_._3)<br>43_._9 (_±_14_._0)<br>_↓_5_._2_†_|



Table 3: Results using observed and unobserved instructions across benchmark tasks (grouped by type). Performance degrades—sometimes by 10+ points—when one uses (UNOBSERVED) instructions, suggesting that instruction-tuned models are not particularly robust. BC, MC, and QA stand for binary classification, multi-class classification, and question answering, respectively. A _[†]_ denotes a statistically significant difference ( _p <_ 0 _._ 05) over datasets for a given model under a paired _t_ -test. 

metrics. To ensure a fair comparison, we evaluate only examples comprising _<_ 512 tokens (including instructions). We use 61 _._ 5% test set data for XSUM (Narayan et al., 2018), 17 _._ 6% for CNN/DM (Nallapati et al., 2016), and 4 _._ 0% for MEDIASUM (Zhu et al., 2021). 

||**Held-in**|**Datasets**|**Held-out**|**Datasets**|
|---|---|---|---|---|
|**Model**|CNN/DM<br>_R_2|XSUM<br>_R_2|MEDIASUM<br>_R_2|WMT19 FR-DE<br>_B_2|
||Avg. Std.|Avg. Std.|Avg. Std.|Avg. Std.|
|Flan-T5-3B|||||
|OBSERVED|**41.5** (_±_0_._1)|**36.1** (_±_1_._0)|**23.4** (_±_6_._2)|**19.2** (_±_1_._8)|
|UNOBSERVED|32_._3 (_±_2_._4)|35_._7 (_±_1_._1)|18_._0 (_±_1_._4)|18_._4 (_±_3_._0)|
|**Performance**∆|_↓_9_._1|_↓_0_._4|_↓_5_._4|_↓_0_._7|
|Alpaca-7B|||||
|OBSERVED|**19.2** (_±_3_._4)|**12.4** (_±_1_._3)|**9.9** (_±_1_._6)|**16.5** (_±_6_._8)|
|UNOBSERVED|16_._8 (_±_0_._9)|8_._3 (_±_0_._4)|9_._1 (_±_0_._7)|16_._2 (_±_6_._9)|
|**Performance**∆|_↓_2_._4|_↓_4_._1|_↓_0_._7|_↓_0_._3|



Table 4: Results using observed and unobserved instructions on NLG tasks. We measure the performance gap on both held-in datasets and held-out datasets for instruction-tuning. Performance is again worse when unobserved instructions are used. _R_ 2 refers to ROUGE-2 F1 and _B_ 2 refers BLEU with _n_ -gram. These are representative results; detailed results with more metrics are in Appendix B 

## 4.3 DOES INSTRUCTION ROBUSTNESS EMERGE AS A FUNCTION OF SCALE? 

We repeated all experiments from Table 3 with Flan-T5 model sizes ranging from small (80M parameters) to XXL (11B). Figure 2b shows that the disparity between results achieved with observed versus unobserved instructions **does not** decrease with model scale, at least up to this point. That said, massive models (175B+) may offer improved robustness. However, we reiterate that much of 

5 

Under review as a conference paper at ICLR 2024 

the excitement about instruction-tuning that this technique appears to allow much smaller models to achieve results competitive with massive alternatives. We report more details in Appendix H 

## 4.4 ROBUSTNESS UNDER IN-CONTEXT LEARNING (ICL) 

Recent work (Gu et al., 2023) showed that LLMs are less sensitive to prompt variation when provided “few-shot” examples in context. We are focused on zero-shot, instruction-following capabilities; but for completeness we re-ran all experiments in a few-shot setting. We followed the sub-sampling strategy from 4.2, and we evaluate the same examples in the few-shot settings we did for the zero shot case. Given space constraints, we report the full results in Appendix C, but briefly summarize our findings here briefly. (1) On average, ICL examples improve both **performance** (avg. performance) and **robustness** (i.e., performance with novel compared to observed instructions) slightly for most models and tasks. (2) For each task, the _variance_ of **performance** and **robustness** significantly increases when more shots are provided, suggesting a sensitivity to the specific ICL examples selected. In particular, model **robustness** varies significantly with choice of ICL examples. 

In sum, ICL does mitigate the robustness issue of medium-sized LLMs we have highlighted, but this is **unstable** and depends on the particular selection of examples. 

## 4.5 A CLOSER LOOK AT INSTRUCTION ROBUSTNESS 

Above we used general instructions to induce LLMs to perform tasks (Table 1). Here we delve further into the performance degradation observed when using novel instructions. We report a curious result: _Incorrect but observed instructions outperform appropriate but unobserved instructions_ (Figure 3). 

We come to this observation by evaluating the performance of Flan-T5-XXL (11B) using six instruction types over seven datasets from BIG-BENCH. This includes (variants of) two instructions _observed_ in training: **Closest** is the instruction from the most similar task in the instruction-tuning set; **Incorrect** is an observed instruction for a _completely different_ and inappropriate task (but which has the same desired output format)—intuitively, these should not yield the desired behavior; **Negated** is the same as **closest** , but we negate the instruction to indicate that it should _not_ perform the task. For _unobserved_ instructions, we consider: **Task designer** , the instruction (task prefix) provided by the author of the task in BIG-BENCH, and; **Newly collected** , novel and appropriate instructions we collected as described above. For reference, we also consider **Nonsensical** , which is a random “instruction” completely irrelevant to any task. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0006-08.png)


**----- Start of picture text -----**<br>
Examples of instructions (for language classification)<br>Nonsensical  The council of street raccoons demands you respond to their inquisition ...<br>Closest  Classify the given news commentary into the language in which it is written in.<br>There are 11 languages to classify the sentences into …<br>Incorrect  Given a set of four words, generate the category that the words belong to.<br>Words are separated by commas. The possible categories are …<br>Negated  Classify the given news commentary into the language in which it is  not<br>written in. There are 11 languages to classify the sentences into …<br>Task designer  Given the sentence: … select the correct language among the choices …<br>Newly collected Given a sentence, select the correct language among the choices.<br>The possible answers are: …<br>Observed<br>Unobserved<br>**----- End of picture text -----**<br>


Figure 3: _**Incorrect**_ **but observed instructions perform better on average than** _**correct**_ **but unobserved instructions** . We report averages over benchmarks, but show example instructions on the right for a specific, illustrative task. We provide all instructions in the Appendix. 

Figure 3 reports average results for these variants. Consistent with our findings, using instructions unobserved in training degrades performance. Strikingly, here we also find that using _inappropriate but observed_ instructions outperforms using _appropriate but unobserved_ instructions. This indicates that instruction-tuned models—at least the modestly sized ones we have evaluated here—may overly rely on having observed instructions in training, and not generalize to new instructions and phrasings as we might hope. We provide a full set of disaggregated results in Appendix B.2. 

6 

Under review as a conference paper at ICLR 2024 

## 4.6 ROBUSTNESS AND “SEMANTIC” DISTANCE 

One observation in 4.2 is that performance on MMLU is less affected by using unobserved instructions. MMLU is a benchmark with 57 QA tasks covering different domains; these tasks all share a similar form of input-output (question, four choices _→_ answer). During instruction collection, we treated all tasks in MMLU as a general QA task and asked NLP researchers to write general QA instructions. We therefore hypothesize that these instructions are comparatively similar to observed instructions, and this in turn explains the relative robustness in this case. 

Figure 4 and Table 5 support this hypothesis. For each instance (instruction plus example), we extract the representation at the penultimate layer for the first decoded token. We use tSNE (Van der Maaten & Hinton, 2008) to visualize these representations of observed and unobserved instructions over instances in MMLU and BBL. Figure 4 shows that for MMLU, the unobserved instructions we collected are quite similar to the observed, while there is a greater separation between unobserved and observed instructions in BBL. Table 5 further quantifies this phenomenon: We report the average _ℓ_ 2 distance between representations of unobserved instructions and those of their nearest observed counterparts. MMLU unobserved instructions are, on average, closer to the nearest observed instruction; this correlates with the smaller observed performance drop. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0007-04.png)


**----- Start of picture text -----**<br>
MMLU BBL<br>Observed<br>Unobserved<br>**----- End of picture text -----**<br>


Figure 4: tSNE plots of representations for the first decoded tokens of 300 randomly sampled examples from MMLU and BBL with Flan-T5 (XXL). Embeddings of observed and unobserved instructions for MMLU are similar, while for BBL they are quite different. This result holds across **most but not all** models considered: See Appendix I for visualizations over all models. 

We plot mean performance degradation (as %) as a function of average similarity between the similarity of the first decoded tokens (following _unobserved_ instructions) and the same for the _most similar observed_ instruction. The negative slope implies the intuitive relationship: Instructions that are dissimilar (in terms of model representations) correlate with poorer performance. However, the relationship is relatively weak, yielding a slope estimate of -0.2 ( _p_ =0.08). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0007-07.png)


**----- Start of picture text -----**<br>
0<br>− 5<br>− 10<br>− 15<br>− 20<br>− 25<br>− 30<br>20 30 40 50 60 70 80<br>Distance<br>(%)<br>performance<br>in<br>Change<br>**----- End of picture text -----**<br>


||**Dataset**<br>MMLU<br>BBL-QA<br>BBL-BC|**Avg.** <br>**19.8**<br>37.9<br>25.3|∆_ℓ_2|**Avg.** <br>-**0.5**<br>-3.4<br>-2.0|∆**Acc.**|
|---|---|---|---|---|---|
||BBL-MC|26.1||-2.8||



Figure 5: Average degradations in performance versus “semantic” distance while using unobserved instructions. In general, larger semantic distances correspond to increased degradation. 

Table 5: Average degradations in performance for four categories (Flan-T5-XXL). MMLU has minimal average distance, indicating a smaller distribution shift; we hypothesize that this is what leads to the relatively small degradation in performance. 

7 

Under review as a conference paper at ICLR 2024 

## 5 ALIGNING REPRESENTATIONS OF EQUIVALENT INSTRUCTIONS 

Based on the preceding analysis, we hypothesize that that _to be robust instruction-following LLMs should induce close internal representations for instructions corresponding to similar tasks_ . Here we propose a simple approach to explicitly realize this aim: We introduce a term in the objective that encourages the model to yield similar predictions (and, correspondingly, similar representations) for the same input when provided distinct but semantically equivalent instructions. 

Specifically, we add soft embedding parameters with dimensions R _[d][×][n]_ ; this is equivalent to adding _n_ novel tokens (with embedding dimension _d_ ) as prefixes to inputs (preceding instructions). The intuition is to push representations for semantically equivalent tasks close together: To this end, fine-tune the soft embeddings (Li & Liang, 2021) under an additional loss term: The KL-divergence _L_ KL of the output probabilities between a reference instruction for a given task and a paraphrased (effectively equivalent) version of the same. We combine this with the standard cross-entropy loss and fit _only_ the introduced soft prompt parameters under this objective (Figure 6). Here _λ_ is a loss-weighting hyper-parameter, _y_ ˆ _i_[(] _[j]_[)] and _y_ ˆ _r_[(] _[j]_[)] are the distributions over the vocabulary _V_ induced by the model with paraphrased instruction _i_ and the reference instruction _r_ at token position _j_ .[2] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0008-04.png)


**----- Start of picture text -----**<br>
CE(ˆ yr [(] [j] [)] [, y] [(] [j] [)][)]<br>KL(ˆ yr [(] [j] [)] [,] [ ˆ] [y] i [(] [j] [)][)]<br>y ˆ r [(] [j] [)] … y ˆ i [(] [j] [)] …<br>distributions over vocabulary<br>L  = (1  − λ ) L CE +  λL KL<br>LLM<br>N<br>1<br>L KL = N − 1 � � KL(ˆ yi [(] [j] [)] [||][y] [ˆ] r [(] [j] [)][)]<br>i = r j<br>… …<br>soft prompt  Select the best answer … original (reference)  soft prompt  Pick the correct … paraphrased  y ˆ i [(] [j] [)] = Softmax( p [(] i [j] [)][)][,] [ p] [(] i [j] [)] ∈ R [|V|]<br>parameters instruction r parameters instruction       i<br>} }<br>**----- End of picture text -----**<br>


Figure 6: Schematic depiction of the proposed instruction alignment method (left) and associated loss terms (right). Dotted (red) lines indicate backpropagation; we update only the soft prompt parameters, which we show yields performance superior to fine-tuning all model parameters. 

Optimizing for the above objective requires similar instructions _i_ for each task in the training data; we generate these automatically as follows. For the instruction-tuning dataset, we sample a small amount of training data to use for representation alignment. We paraphrase these reference instructions using GPT-4. For the Alpaca collection, we randomly sampled 1k tasks, paraphrased them with three prompts, and collected the top three candidates under temperature 0.5. For the Flan collection, we randomly sampled 1k instances from the mixture with 3 prompts with greedy decoding (see B.3). 

For fine-tuning, we create instances for each example by pairing them with every distinct instruction available for the corresponding task. We then form batches by including one instance featuring the original instruction, and the rest comprising paraphrased instructions. 

## 6 RESULTS 

We implement the proposed method to push together the representations of instructions for two models: Flan-T5-XL (3B) and Alpaca (7B). After training with the added prefix, we reproduce the main results from Section 4.2. Then, following the approach in Section 4.6, we take the average distance between observed and unobserved instructions before and after. Table 6 shows that this brings observed and unobserved instruction representations closer together. The similarity is most increased in the case of the biggest accuracy gain, supporting the hypothesized mechanism of improvement. We compare the canonical versions of these models trained in the usual way (the same evaluated in Table 3) to variants fine-tuned with the proposed approach. We ablate components of our method to tease out the contributions of the extra parameters and closer representation. Specifically, we 

> 2We pad instances such that the lengths in a given batch are effectively equal; the sum is therefore from 1 to the length associated with the current batch, we omit this for simplicity. 

8 

Under review as a conference paper at ICLR 2024 

|**Dataset**|**Closest Distance Before**|**Closest Distance After**|∆**Acc. Improvement**(%)|
|---|---|---|---|
|MMLU|22.2|21.3|+ 0.3%|
|BBL QA|22.4|23.0|+ 0.4%|
|BBL BC|30.1|**27.9**|**+ 4.2%**|
|BBL MC|26.0|24.6|+ 0.3%|



Table 6: Average distances before and after soft prompt alignment with Flan-T5-XL. 

consider variants where we: Fine-tune all model parameters on the additional, automatically generated instruction paraphrases (FT); impose the new KL loss term (again fine-tuning all model parameters; FT+KL); introduce the additional soft prompt parameters and fine-tune on the paraphrase instances, but without KL (PT); and then the full proposed strategy, which introduces the soft prompt parameters and optimizes them for the loss augmented with the KL term ( **PT+KL** ). 

|||MMLU|||BBL||
|---|---|---|---|---|---|---|
|**Model**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|FLAN-T5-3B|48.1|47.5|47.8|**56.1**|51.9|54.0|
|FT|39.4**(-8.7)**|40.1**(-7.4)**|39.8**(-8.0)**|48.2**(-7.9)**|42.3**(-9.2)**|45.3**(-8.7)**|
|FT+KL|41.8**(-6.3)**|43.6**(-3.9)**|45.9**(-1.9)**|47.7**(-8.4)**|43.1**(-8.8)**|45.4**(-8.6)**|
|PT|48.1**(+0.0)**|47.6**(+0.1)**|47.9**(+0.1)**|55.9**(-0.2)**|52.1**(+0.2)**|54.0**(+0.0)**|
|**PT+KL**_†_|**48.1 (+0.1)**|**47.9 (+0.4)**|**48.0 (+0.2)**|55.9**(-0.2)**|**53.7 (+1.8)**|**54.8 (+0.8)**|
|ALPACA-7B|41.9|39.7|40.8|47.6|42.9|45.3|
|FT|40.3**(-1.6)**|39.1**(-0.6)**|39.7**(-1.1)**|44.4**(-3.2)**|42.1**(-0.8)**|43.4**(-2.0)**|
|FT+KL|39.7**(-2.2)**|40.2**(+0.5)**|40.0**(-0.8)**|45.6**(-2.0)**|42.8**(-0.1)**|44.2**(-1.1)**|
|PT|42.1**(+0.2)**|40.0**(+0.3)**|41.1**(+0.3)**|47.5**(-0.1)**|43.0**(+0.1)**|45.3**(+0.0)**|
|**PT+KL**_†_|**42.4 (+0.5)**|**41.8 (+2.1)**|**42.1 (+1.3)**|**47.9 (+0.3)**|**46.6 (+3.7)**|**47.3 (+2.0)**|



Table 7: Results and ablations of the proposed soft prompt alignment method. Ablated versions use the augmented set with paraphrased instructions. FT refers to fine-tuning (with teacher-forcing) on this additional data; PT denotes prefix tuning (i.e., introducing soft prompt parameters); KL refers to the alignment objective that we proposed above. Using all components together yields the best performance, especially on unobserved instructions; _[†]_ this improvement is statistically significant ( _p <_ 0 _._ 05) for both models under separate paired _t_ -tests. We report variances in Appendix B. 

We report results in Table 7. Two observations: (1) The proposed soft prompt alignment strategy ( **PT+KL** ) yields consistent improvements across the tasks and models considered and especially improves performance on unobserved instructions, as anticipated. (2) Simply fine-tuning with extra parameters or data (e.g., the newly collected paraphrases) on the same training tasks does not improve the overall performance and robustness. 

## 7 CONCLUSIONS 

Instruction-tuned LLMs have emerged as a promising means of achieving zero-shot performance with smaller models that are competitive to that observed using much larger LLMs. We characterized the _robustness_ of such models with respect to instruction rephrasings. We collected manually composed instructions from 36 graduate students in NLP across 75 tasks, and we evaluated different families of instruction-tuned LLMs (Flan, Alpaca, and T0) when provided observed and unobserved instructions (seen in training and not, respectively). We found that using the latter consistently degrades model performance, indicating that models are unduly sensitive to instruction phrasings. 

Our analysis suggested that an issue may be the internal representations of instructions: Equivalent instructions should be near each other in latent space. On this hypothesis, we introduced a simple mechanism intended to improve the robustness of instruction-tuned LLMs. This approach entails introducing an additional loss term that penalizes the model for inducing dissimilar distributions when using (a) paraphrased instructions as opposed to (b) a reference instruction for the same task. We found that training under this objective consistently improves results and mitigates the degradation observed when previously unobserved instructions are used. 

9 

Under review as a conference paper at ICLR 2024 

## 8 ETHICS STATEMENT 

This work does not have an explicit ethical dimension, but we acknowledge that all LLMs are likely to encode problematic biases; it is currently unclear how instruction-tuning might interact with these, and we think this is a topic worth pursuing in future work. 

## 9 REPRODUCIBILITY STATEMENT 

All the results reported in the paper are reproducible. We submit the code and include all the implementation details in Appendix B. We present the disaggregated results in Appendix K. All the instructions used in the paper are formatted and listed in the supplementary materials. 

## REFERENCES 

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. _arXiv preprint arXiv:2211.15661_ , 2022. 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_ , 2022. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 

- Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. Why can gpt learn incontext? language models secretly perform gradient descent as meta optimizers. _arXiv preprint arXiv:2212.10559_ , 2022. 

- Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. _arXiv preprint arXiv:2209.14375_ , 2022. 

- Jiasheng Gu, Hanzi Xu, Liangyu Nie, and Wenpeng Yin. Robustness of learning from task instructions. 2023. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. _arXiv preprint arXiv:2203.15556_ , 2022. 

- Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Dániel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. OPT-IML: Scaling language model instruction meta learning through the lens of generalization. _arXiv preprint arXiv:2212.12017_ , 2022. 

- Joel Jang, Seonghyeon Ye, and Minjoon Seo. Can large language models truly understand prompts? a case study with negated prompts. In _Transfer Learning for Natural Language Processing Workshop_ , pp. 52–62. PMLR, 2023. 

10 

Under review as a conference paper at ICLR 2024 

- Nitish Shirish Keskar, Bryan McCann, Caiming Xiong, and Richard Socher. Unifying question answering, text classification, and regression via span extraction. _arXiv preprint arXiv:1904.09286_ , 2019. 

- Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. Unifiedqa: Crossing format boundaries with a single qa system. _arXiv preprint arXiv:2005.00700_ , 2020. 

- Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. _arXiv preprint arXiv:1910.09700_ , 2019. 

- Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. _arXiv preprint arXiv:2101.00190_ , 2021. 

- Zekun Li, Baolin Peng, Pengcheng He, and Xifeng Yan. Do you really follow me? adversarial instructions for evaluating the robustness of large language models. _arXiv preprint arXiv:2308.10819_ , 2023. 

- Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pp. 74–81, 2004. 

- Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. _arXiv preprint arXiv:2301.13688_ , 2023. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. _arXiv preprint arXiv:1806.08730_ , 2018. 

- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? _arXiv preprint arXiv:2202.12837_ , 2022. 

- Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. _arXiv preprint arXiv:2104.08773_ , 2021. 

- Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. _arXiv preprint arXiv:2112.09332_ , 2021. 

- Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al. Abstractive text summarization using sequence-to-sequence rnns and beyond. _arXiv preprint arXiv:1602.06023_ , 2016. 

- Shashi Narayan, Shay B Cohen, and Mirella Lapata. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. _arXiv preprint arXiv:1808.08745_ , 2018. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _arXiv preprint arXiv:2203.02155_ , 2022. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pp. 311–318, 2002. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _The Journal of Machine Learning Research_ , 21(1):5485–5551, 2020. 

- Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. _arXiv preprint arXiv:2110.08207_ , 2021. 

11 

Under review as a conference paper at ICLR 2024 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ , 2022. 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpaca: A strong, replicable instruction-following model, 2023. URL https://crfm.stanford.edu/2023/03/13/alpaca.html. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023a. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023b. 

- Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. _Journal of machine learning research_ , 9(11), 2008. 

- Xinyi Wang, Wanrong Zhu, and William Yang Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. _arXiv preprint arXiv:2301.11916_ , 2023. 

- Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. _arXiv preprint arXiv:2212.10560_ , 2022a. 

- Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Benchmarking generalization via in-context instructions on 1,600+ language tasks. _arXiv preprint arXiv:2204.07705_ , 2022b. 

- Albert Webson and Ellie Pavlick. Do prompt-based models really understand the meaning of their prompts? In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2300–2344, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main. 167. URL https://aclanthology.org/2022.naacl-main.167. 

- Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. _arXiv preprint arXiv:2109.01652_ , 2021. 

- Zhiyang Xu, Ying Shen, and Lifu Huang. Multiinstruct: Improving multi-modal zero-shot learning via instruction tuning. _arXiv preprint arXiv:2212.10773_ , 2022. 

- Seonghyeon Ye, Doyoung Kim, Joel Jang, Joongbo Shin, and Minjoon Seo. Guess the instruction! making language models stronger zero-shot learners. _arXiv preprint arXiv:2210.02969_ , 2022. 

- Ping Yu, Tianlu Wang, Olga Golovneva, Badr Alkhamissy, Gargi Ghosh, Mona Diab, and Asli Celikyilmaz. Alert: Adapting language models to reasoning tasks. _arXiv preprint arXiv:2212.08286_ , 2022. 

- Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. _arXiv preprint arXiv:2210.02414_ , 2022. 

- Kai Zhang, Bernal Jiménez Gutiérrez, and Yu Su. Aligning instruction tasks unlocks large language models as zero-shot relation extractors. _arXiv preprint arXiv:2305.11159_ , 2023a. 

- Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, and Joseph E Gonzalez. The wisdom of hindsight makes language models better instruction followers. _arXiv preprint arXiv:2302.05206_ , 2023b. 

12 

Under review as a conference paper at ICLR 2024 

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. _arXiv preprint arXiv:1904.09675_ , 2019. 

- Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. _arXiv preprint arXiv:2211.01910_ , 2022. 

- Chenguang Zhu, Yang Liu, Jie Mei, and Michael Zeng. Mediasum: A large-scale media interview dataset for dialogue summarization. _arXiv preprint arXiv:2103.06410_ , 2021. 

- Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, et al. Promptbench: Towards evaluating the robustness of large language models on adversarial prompts. _arXiv preprint arXiv:2306.04528_ , 2023. 

13 

Under review as a conference paper at ICLR 2024 

## A LIMITATIONS 

This work has important limitations: For example, we only evaluated “mid-sized” models (<20B parameters), it is unclear if our findings would generalize to much larger instruction-tuned models. (However, we note that instruction tuning has been most promising for smaller models.) We also considered only a somewhat limited set of task types: QA, multi-class and binary classification, and then summarization and translation for generation tasks. 

## B EXPERIMENTS 

To ensure reproducibility, we provide all details regarding our evaluation of the robustness of instruction-tuned LLMs. 

## B.1 MAIN RESULTS 

LLMs sometimes generate outputs that are _correct_ but different from a (natural language) target. Therefore, we predict answers according to “multiple-choice” grading suggested by BIG-BENCH, by which we take the logits score and argmax over all the possible choices to obtain the prediction. In most cases, this approach yields the same accuracy as using exact match for evaluation. Here are the configurations for all the models we evaluated. 

|**Models**|Node Type|Precision|Batch Size|Hours|_CO_2emission (KG)|
|---|---|---|---|---|---|
|**Inference**||||||
|Flan-T5-Small|V100-SXM2-32G|FP16|128|64|4.0|
|Flan-T5-Base|V100-SXM2-32G|FP16|128|128|8.1|
|Flan-T5-Large|V100-SXM2-32G|FP16|32|256|16.2|
|Flan-T5-XL|V100-SXM2-32G|FP16|32|512|32.3|
|Flan-T5-XXL|RTX-A6000-46G|BF16|8|600|37.8|
|T0++|RTX-A6000-46G|BF16|2|128|8.1|
|Alpaca-7B|A100-SXM4-80G|BF16|16|160|13.4|
|Alpaca-13B|A100-SXM4-80G|BF16|8|192|16.1|
|**Training**||||||
|Flan-T5-XL|A100-SXM4-80G|BF16|256|256|21.5|
|Alpaca-7B|A100-SXM4-80G|BF16|128|80|6.7|
|**Estimated Total**|_CO_2**Emission (KG)**||||164.2|



Table 8: The configurations for evaluating different instruction-tuned LMs. The _CO_ 2 emission is estimated by Lacoste et al. (2019). Total estimated emissions are about equivalent to a 679 KM drive by an average ICE car. 

## B.2 “CLOSER LOOK” EXPERIMENT RESULTS 

We provide detailed results that correspond to the aggregated values we reported in Figure 3. 

Table 9 shows that among the 7 benchmark datasets that we experiment with, the "Incorrect" instructions outperform the "Collected" instructions. And then for the other 2 datasets the performance of "Incorrect" instructions is comparable ( _−_ 1%) to the "collected" ones. This indicates that the instruction-following ability of the LLMs is not robust to the fine-grained semantics within the instructions, as it only cares about the high-level requirement of the prompt. (classification, QA, etc.) 

## B.2.1 MOTIVATION AND INSTRUCTION COLLECTION 

Here, we report the detailed procedure and motivation for collecting the instruction templates for the six settings we experiment with. 

14 

Under review as a conference paper at ICLR 2024 

|**Dataset**|**Observed**<br>Closest<br>Incorrect|**Observed**<br>Closest<br>Incorrect|**Unobserved**<br>Collected<br>Task Designer|**Unobserved**<br>Collected<br>Task Designer|**Control**<br>Negated<br>Nonsensical|**Control**<br>Negated<br>Nonsensical|
|---|---|---|---|---|---|---|
|Intent<br>Recognition|93.6<br>_±_0_._3|93.1<br>_±_1_._0|94.1<br>_±_0_._6|**94.7**|28.0<br>_±_6_._5|40.7<br>_±_7_._2|
|Empirical<br>Judgments|39.2<br>_±_0_._8|**41.6**<br>_±_6_._3|37.6<br>_±_1_._7|37.4|28.1<br>_±_3_._5|30.9<br>_±_2_._5|
|Conceptual<br>Combinations|78.0<br>_±_1_._6|**78.9**<br>_±_0_._5|75.3<br>_±_3_._3|58.3|11.2<br>_±_2_._1|63.7<br>_±_2_._6|
|Language<br>Identifcation|**38.9**<br>_±_0_._3|29.3<br>_±_5_._0|28.8<br>_±_5_._8|27.6|36.9<br>_±_0_._5|12.4<br>_±_0_._5|
|Logical<br>Sequence|**56.9**<br>_±_6_._6|49.4<br>_±_6_._1|52.8<br>_±_5_._3|53.8|11.8<br>_±_6_._9|34.4<br>_±_5_._9|
|Crash<br>Blossom|53.6<br>_±_2_._8|50.0<br>_±_5_._3|50.5<br>_±_2_._2|**63.2**|28.6<br>_±_6_._2|43.7<br>_±_1_._4|
|Epistemic<br>Reasoning|62.8<br>_±_2_._9|59.3<br>_±_3_._4|58.1<br>_±_1_._7|60.2|**65.5**<br>_±_4_._6|49.5<br>_±_1_._3|
|Overall|**60.5**<br>_±_2_._1|57.4<br>_±_2_._2|56.8<br>_±_1_._8|56.4|30.0<br>_±_2_._2|39.3<br>_±_2_._3|



Table 9: The detailed results of “A Closer Look" experiment. Using “Incorrect” but observed instructions in most cases outperforms correct but unobserved instructions (“Closest” and “Collected”, respectively). 

**Closest Instructions** In the main results reported in 4.2, we explicitly use the observed instruction templates that are general enough to be used for **all** the task belongs to the same categories. We show that even with such generality, the observed instructions template still outperforms the unobserved ones by a large margin. Next we evaluate whether the observed instructions for similar trained tasks perform better than the general ones. For each task, we collected the closest instructions by **manually** going through the instruction tuning set, and selecting suitable instructions from the train tasks that are similar to the evaluation tasks, e.g., instructions from WMT16 (trained) for WMT19 (evaluate). 

**Incorrect Instructions** As an extreme opposite setting to **Closest Instruction** , we investigate how the models perform when the given instructions are from completely different tasks, but have a similar input-output format (e.g., language classification and movie genre classification). This setting **should not** work if the model “understands” instruction semantics. For each task, we collect the closest instructions with the following procedure. We again **manually** go through the instruction tuning set and filter a set of tasks that belongs to the **same category** (QA, summarization, Multi-class classification, and so on). Among tasks in the same category, we identify instrctions from the most different task. 

**Negated Instructions** Following the design of Jang et al. (2023), we use negated instructions as an additional probe to assess whether the model understands instructions or mostly recalls back to the training data. For each task, we collect negated instructions by: 

- (1) Taking all of the **Closest Instructions** for this task. 

- (2) Insert a negation word (e.g., not) in the appropriate place of the instruction. 

**Task Designer Instruction** For the benchmark BBL that we evaluate, for each task, it provides a prompt template written by the task designer. This serves as a gold unobserved instruction that is guaranteed valid for this task. 

**Newly Collected Instruction** This is the same as the "Unobserved Instruction" that we used throughout the paper. Following the exact same procedure in Section 4.2, we collect 10 unobserved instructions for each task we evaluate in this section. 

15 

Under review as a conference paper at ICLR 2024 

**Nonsensical Instruction** This setting is used as a baseline. We perform inference of the task with an instruction prepended to the input, that is irrelevant and doesn’t provide any extra information. (e.g., _"The council of street squirrel demands you response: Question: ..."_ ) 

## B.3 DETAILS ON ALIGNING REPRESENTATIONS OF EQUIVALENT INSTRUCTIONS 

We conduct all training and ablation studies on 8 A100s with 80GB memory. We kept the KL-Loss weight fixed at 0 _._ 8. We trained both Flan-T5-XL and Alpaca-7B with a batch size of 4 and a batch gradient accumulation of 4. We set the weight decay to 1 _e −_ 5 and the learning rate to 5 _e −_ 4 for all experiments. We use a prefix length of 10. We explore the choice of the prefix length and found no significant correlation with the performance and "semantic" distance between instructions. 

|||MMLU|||BBL||
|---|---|---|---|---|---|---|
|**Model**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|FLAN-T5-3B|48.1(_±_0_._3)|47.5(_±_0_._9)|47.8(_±_0_._6)|**56.1**(_±_5_._8)|51.9(_±_9_._6)|54.0(_±_7_._7)|
|FT|39.4(_±_0_._5)|40.1(_±_0_._4)|39.8(_±_0_._4)|48.2(_±_5_._6)|42.3(_±_8_._8)|45.3(_±_7_._2)|
|FT+KL|41.8(_±_0_._3)|43.6(_±_0_._2)|45.9(_±_0_._3)|47.7(_±_4_._2)|43.1(_±_7_._3)|45.4(_±_5_._8)|
|PT|48.1(_±_0_._3)|47.6(_±_0_._8)|47.9(_±_0_._6)|55.9(_±_6_._0)|52.1(_±_9_._9)|54.0(_±_8_._0)|
|**PT+KL**|**48.1**(_±_0_._2)|**47.9**(_±_0_._4)|**48.0**(_±_0_._3)|55.9(_±_4_._4)|**53.7**(_±_5_._9)|**54.8**(_±_5_._1)|
|ALPACA-7B|41.9(_±_0_._6)|39.7(_±_2_._2)|40.8(_±_1_._4)|47.6(_±_2_._8)|42.9(_±_5_._5)|45.3(_±_4_._2)|
|FT|40.3(_±_0_._7)|39.1(_±_1_._9)|39.7(_±_1_._3)|44.4(_±_3_._3)|42.1(_±_4_._9)|43.4(_±_4_._1)|
|FT+KL|39.7(_±_0_._6)|40.2(_±_1_._4)|40.0(_±_1_._0)|45.6(_±_2_._2)|42.8(_±_6_._4)|44.2(_±_4_._3)|
|PT|42.1(_±_0_._8)|40.0(_±_2_._6)|41.1(_±_1_._7)|47.5(_±_2_._8)|43.0(_±_6_._1)|45.3(_±_4_._5)|
|**PT+KL**|**42.4**(_±_0_._4)|**41.8**(_±_1_._3)|**42.1**(_±_0_._9)|**47.9**(_±_2_._1)|**46.6**(_±_3_._6)|**47.3**(_±_2_._8)|



Table 10: For the simplicity of the main paper, we report the StD of the results in Table 7 

To generate “equivalent” instructions for tasks from seed instructions, we prompted GPT-4 to produce paraphrases of given instructions. We detail our procedure for doing this for each model evaluated below. We used different prompts for these two instruction collections because, in informal development, we found them to produce better paraphrases (respectively). We emphasize that these are instructions from the _training_ instruction-tuning set, so this cannot be construed as “cheating”, although we acknowledge that there is a small amount of effort necessary to ensure LLMs (GPT-4) produces “good” paraphrases at scale. 

**Alpaca** To generate paraphrases of observed instructions in the Alpaca collection, we sampled 1000 out of 52002 tasks at i.i.d. random and generated instruction paraphrases with GPT-4 via the following prompts. 

- “Paraphrase this sentence:\n\n _{_ instruction _}_ Paraphrased sentence:\n\n” 

- “Paraphrase this instruction into a longer sentence\n\n\n _{_ instruction _}_ New sentence:\n” 

- “You are given an instruction:\n\n _{_ instruction _}_ Now, paraphrase it into a new instruction with equivalent meaning:\n\n” 

**Flan** We first reproduced the held-in instruction-tuning set of Flan-T5 with the pipeline. We randomly sampled 986 data samples from the generated data following the proportion reported in Chung et al. (2022). We generate paraphrases of the selected data with GPT-4 using the following prompts. 

- “Here’s an input utterance:\n\n{instruction}\n \n \n Now, your task is to paraphrase the input by only changing the instruction but leaving everything else the same.\n Here’s the new utterance:\n\n” 

- “You are given an utterance which is a combination of task instruction and the actual input. Your job is to paraphrase the task instruction and leave the input unchanged. Here’s the utterance to be paraphrased:\n\n\n{instruction}\n\n\n Now, generate the new utterance:\n\n\n” 

- “You are provided with the utterance of a specific task and I need you to paraphrase it. The actual input, question, and examples in the task should not be changed. You 

16 

Under review as a conference paper at ICLR 2024 

should only paraphrase the instructions. Task:\n\n\n {instruction}\n\n\nThe paraphrased utterance:\n\n\n” 

On our informal and subjective assessment, these prompts yielded seemingly good paraphrases; we therefore did not invest time in further “prompt engineering” for this problem, given that our focus in this work is not automated paraphrasing _per se_ —rather, we aim only to collect “reasonable” paraphrases that allow us to realize the proposed objective which looks to push representations of semantically similar instructions relatively close together; GPT-4 readily provides outputs servicable in this regard. 

## B.3.1 QUALITY STUDY OF INSTRUCTION TUNING DATASETS AND PARAPHRASES 

Here we investigate the quality and effectiveness of aligning instruction representations when paraphrases are generated under models aside from GPT-4. We do this to explore the sensitivity of our approach to specific sources of instruction paraphrases, although we emphasize that these might also be manually composed (at an associated cost in effort, of course). 

**Training with More Instructions v.s. Inducing Closer Representations** We show that the robustness of instruction-following LLMs could be improved by aligning the representations of semantically equivalent instructions. However, it remains unclear whether this approach is better than training new LLMs from scratch with the extra instructions. Here, we compare the performance and robustness of LLMs re-trained versus LLMs aligned with the new instructions. 

|**Method**|Avg Acc. (MMLU)|Avg Degr. (MMLU)|# Tokens|# Parameters|
|---|---|---|---|---|
|**Flan-T5-XL (3B)**|47.8|-0.6|||
|- Training w/ new paraphrases|47.7|-0.5|4.4B|3B|
|- Aligning w/ new paraphrases|**48.0**|**-0.2**|**390K**|**61K**|
|**Alpaca (7B)**|40.8|-2.2|||
|- Training w/ new paraphrases|41.3|-2.1|3.4M|7B|
|- Aligning w/ new paraphrases|**42.1**|**-0.6**|**233K**|**122K**|



Table 11: Performance on MMLU and BBL when the model is retrained from scratch (with the new instructions) and when the model representation is aligned with the new instructions. # Tokens refers to the total number of tokens used in the process; #Parameters refers to the total number of trainable parameters in the process. 

Table 11 shows that even with significantly larger training text and trainable parameters, re-training the model from scratch still underperforms, aligning the representations of equivalent instructions. 

**Aligning with Paraphrases from Different Sources** In Section 5, we adopt GPT-4 as the source to generate semantically equivalent instructions. We agree that whether results hold with alternative sources of paraphrases is a worthwhile scientific question. We have replicated the experiment from Section 5 with two other reasonable sources of paraphrases: Llama2-13B Touvron et al. (2023b) and text-DaVinci-003 Ouyang et al. (2022). We reproduce the experiment in Table 7 with Flan-T5-XL. 

|||MMLU|||BBL||
|---|---|---|---|---|---|---|
|**Source Model**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|Text-DaVinci-003|48.1|47.7|47.9|55.4|53.0|54.2|
|Llama2-13B|48.0|47.6|47.8|55.4|50.8|53.1|
|**GPT-4**|48.1|**47.9**|**48.0**|**55.9**|**53.7**|**54.8**|



Table 12: The performance of the models after aligning representation with soft prefix ( **PT+KL** in 7). GPT-4 generated paraphrases improve the performance and robustness of the instruction-following LLMs better than the other two sources. 

We take a more in-depth look at the quality of paraphrased instructions generated by GPT-4, Davinci003, and Llama2-13B. We observed that the performance improvement using the paraphrases as 

17 

Under review as a conference paper at ICLR 2024 

equivalent instructions is **correlated with the diversity and quality of the paraphrases** . For each source of the paraphrases, we measure the average token length generated, as well as the similarity (ROUGE-2) between the seed instruction and its paraphrases. 

Table 13 shows that GPT-4 and DaVinci produce relatively similar paraphrases. The ROUGE-2 score compared with the original instruction is significantly higher than GPT-4 and DaVinci-003, implying a lower diversity and significance in the equivalent instruction set. 

|**Source Model**|**Avg. Token Length**|**Avg. ROUGE-2 w/ seed instruction**|
|---|---|---|
|Text-DaVinci-003|154 (_±_76)|47_._2 (_±_39_._5)|
|Llama2-13B|286 (_±_201)|55_._3 (_±_40_._0)|
|GPT-4|165 (_±_76)|45_._8 (_±_39_._1)|



Table 13: The statistics of the paraphrased instructions generated from different LLM sources 

## B.3.2 EFFICACY OF ALIGNMENT TERM 

In Table 7, we conduct a detailed ablation study to show the efficacy of the KL term on migitating the robustness issue. Comparing with prefix tuning, the proposed method introduces extra term to cluster the hidden representation produced by the equivalent instructions together, which results in a **1.0** and **3.6** improvement for the unobserved instructions on MMLU and BBL respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0018-07.png)


**----- Start of picture text -----**<br>
20.0 Method<br>PT<br>17.5 PT + KL<br>15.0<br>12.5<br>10.0<br>7.5<br>5.0<br>2.5<br>0.0<br>BBQ Lite Code Desc. Conc. Com. Hindu Know. Known Unk. Lang ID Logical Ded. MMLU Novel Conc. Play Dialog Strange St. Strat. QA Vitamin C Winowhy<br>Dataset<br>Distance<br>**----- End of picture text -----**<br>


Figure 7: The average _l_ 2 distance between the unobserved instructions to their closest observed instructions. The result shows that the semantic distance between equivalent instructions is smaller for PT+KL in 13 out of 14 datasets. 

## B.4 NLG EXPERIMENT RESULTS 

We report the full results on four NLG datasets. We use ROUGE-2 (F1) Lin (2004) and BERTScore Zhang et al. (2019) as the automatic metrics for summarization; we use BLEU Papineni et al. (2002) and BERTScore as the automatic metrics for translation. T0++ Sanh et al. (2021) is not evaluated for translation tasks because it is not trained on any non-English tasks. 

## C INSTRUCTION ROBUSTNESS WITH IN-CONTEXT LEARNING 

We have been focused on zero-shot settings, but here we also report results achieved under In-context Learning (ICL). We repeat experiments from Section 4.2 on MMLU with the one-shot and three-shot ICL below. Additionally, we compare the effect of few-shot examples to our proposed representation alignment method: 

We observe that while few-shot prompting does indeed mitigate robustness issues to some degree, it does so with high variance, compared to the proposed alignment technique. Furthermore, the selection of in-context examples also affects the performance and robustness of k-shot prompting. We repeat the Alpaca-7B (Three-shots) experiment 10 more times with random examples sampled from the test set and the standard deviation raises from **3.9** to **5.8** , indicating the instability of the 

18 

Under review as a conference paper at ICLR 2024 

|||**Held-in**|**Datasets**|||**Held-out**|**Datasets**||
|---|---|---|---|---|---|---|---|---|
|**Model**|CNN/DM<br>_R_2<br>BERT||XSUM<br>_R_2<br>BERT||MEDIASUM<br>_R_2<br>BERT||WMT19 FR-DE<br>_B_2<br>BERT||
||Avg. Std.|Avg. Std.|Avg. Std.|Avg. Std.|Avg. Std.|Avg. Std.|Avg. Std.|Avg. Std.|
|Flan-T5-Small|||||||||
|OBSERVED|**32.0** (_±_6_._1)|**95.2** (_±_0_._2)|**24.6** (_±_0_._3)|**96.1** (_±_0_._0)|16_._5 (_±_1_._4)|**95.9** (_±_0_._1)|**4.6** (_±_0_._5)|**94.2** (_±_0_._2)|
|UNOBSERVED|31_._5 (_±_5_._6)|95_._1 (_±_0_._2)|23_._6 (_±_1_._3)|96_._0 (_±_0_._0)|**18.6** (_±_2_._3)|95_._9 (_±_0_._1)|3_._8 (_±_1_._5)|93_._6 (_±_1_._0)|
|**Performance**∆|_↓_0_._5|_↓_0_._1|_↓_0_._9|_↓_0_._0|_↑_2_._1|_↓_0_._0|_↓_0_._8|_↓_0_._6|
|Flan-T5-Base|||||||||
|OBSERVED|27_._7 (_±_0_._3)|**95.1** (_±_0_._1)|**29.4** (_±_0_._2)|**96.0** (_±_0_._0)|15_._3 (_±_0_._8)|**95.8** (_±_0_._1)|**13.5** (_±_1_._0)|**94.7** (_±_0_._0)|
|UNOBSERVED|**28.7** (_±_0_._6)|95_._0 (_±_0_._0)|29_._1 (_±_0_._5)|96_._0 (_±_0_._0)|**17.1** (_±_4_._5)|95_._8 (_±_0_._1)|11_._8 (_±_3_._9)|94_._5 (_±_0_._4)|
|**Performance**∆|_↑_1_._0|_↓_0_._1|_↓_0_._4|_↓_0_._0|_↑_1_._7|_↓_0_._0|_↓_1_._7|_↓_0_._2|
|Flan-T5-Large|||||||||
|OBSERVED|**39.6** (_±_1_._0)|**95.4** (_±_0_._2)|**32.9** (_±_1_._1)|95_._9 (_±_0_._1)|**20.5** (_±_2_._5)|**95.9** (_±_0_._2)|**15.9** (_±_2_._3)|**94.5** (_±_0_._1)|
|UNOBSERVED|34_._3 (_±_3_._9)|95_._2 (_±_0_._3)|32_._1 (_±_1_._4)|**95.9** (_±_0_._1)|18_._7 (_±_1_._5)|95_._9 (_±_0_._1)|15_._2 (_±_5_._4)|94_._5 (_±_0_._3)|
|**Performance**∆|_↓_5_._3|_↓_0_._2|_↓_0_._9|_↓_0_._0|_↓_1_._8|_↓_0_._1|_↓_0_._7|_↓_0_._0|
|Flan-T5-3B|||||||||
|OBSERVED|**41.5** (_±_0_._1)|**95.3** (_±_0_._0)|**36.1** (_±_1_._0)|**95.9** (_±_0_._0)|**23.4** (_±_6_._2)|**96.0** (_±_0_._1)|**19.2** (_±_1_._8)|**94.5** (_±_0_._1)|
|UNOBSERVED|32_._3 (_±_2_._4)|95_._3 (_±_0_._1)|35_._7 (_±_1_._1)|95_._9 (_±_0_._0)|18_._0 (_±_1_._4)|95_._7 (_±_0_._1)|18_._4 (_±_3_._0)|94_._5 (_±_0_._1)|
|**Performance**∆|_↓_9_._1|_↓_0_._0|_↓_0_._4|_↓_0_._0|_↓_5_._4|_↓_0_._3|_↓_0_._7|_↓_0_._0|
|Alpaca-7B|||||||||
|OBSERVED|**19.2** (_±_3_._4)|93_._0 (_±_0_._6)|**12.4** (_±_1_._3)|92_._3 (_±_0_._2)|**9.9** (_±_1_._6)|**92.9** (_±_0_._5)|**16.5** (_±_6_._8)|**94.1** (_±_0_._6)|
|UNOBSERVED|16_._8 (_±_0_._9)|**93.3** (_±_0_._5)|8_._3 (_±_0_._4)|**93.3** (_±_0_._9)|9_._1 (_±_0_._7)|92_._9 (_±_0_._6)|16_._2 (_±_6_._9)|93_._7 (_±_1_._0)|
|**Performance**∆|_↓_2_._4|_↑_0_._3|_↓_4_._1|_↑_1_._0|_↓_0_._7|_↓_0_._0|_↓_0_._3|_↓_0_._3|
|Flan-T5-11B|||||||||
|OBSERVED|**33.6** (_±_0_._2)|**95.1** (_±_0_._0)|**39.9** (_±_0_._1)|**95.9** (_±_0_._0)|**22.2** (_±_2_._3)|95_._7 (_±_0_._1)|**22.2** (_±_0_._4)|**94.6** (_±_0_._0)|
|UNOBSERVED|32_._0 (_±_2_._3)|95_._1 (_±_0_._1)|39_._7 (_±_0_._3)|95_._9 (_±_0_._0)|18_._0 (_±_2_._2)|**95.8** (_±_0_._1)|22_._0 (_±_1_._3)|94_._6 (_±_0_._0)|
|**Performance**∆|_↓_1_._6|_↓_0_._0|_↓_0_._3|_↓_0_._0|_↓_4_._2|_↑_0_._2|_↓_0_._2|_↓_0_._0|
|T0-11B|||||||||
|OBSERVED|**35.1** (_±_0_._1)|**95.3** (_±_0_._0)|**40.2** (_±_0_._3)|**95.9** (_±_0_._0)|**18.9** (_±_4_._8)|**95.6** (_±_0_._2)|-|-|
|UNOBSERVED|35_._0 (_±_0_._2)|95_._3 (_±_0_._0)|40_._1 (_±_0_._5)|95_._9 (_±_0_._0)|16_._5 (_±_0_._8)|95_._4 (_±_0_._1)|-|-|
|**Performance**∆|_↓_0_._1|_↓_0_._0|_↓_0_._1|_↓_0_._0|_↓_2_._4|_↓_0_._1|-|-|
|Alpaca-13B|||||||||
|OBSERVED|**19.3** (_±_3_._0)|**94.3** (_±_0_._3)|**10.3** (_±_3_._0)|**94.9** (_±_0_._2)|**7.1** (_±_0_._3)|94_._2 (_±_0_._3)|14_._0 (_±_5_._8)|**94.8** (_±_0_._0)|
|UNOBSERVED|16_._1 (_±_0_._9)|94_._1 (_±_0_._1)|8_._0 (_±_0_._5)|94_._8 (_±_0_._1)|6_._9 (_±_0_._3)|**94.5** (_±_0_._3)|**15.4** (_±_6_._1)|94_._8 (_±_0_._0)|
|**Performance**∆|_↓_3_._3|_↓_0_._2|_↓_2_._3|_↓_0_._1|_↓_0_._2|_↑_0_._2|_↑_1_._4|_↓_0_._0|



Table 14: Full results using observed and unobserved instructions across NLG tasks. 

|**Models**|Avg. Acc. (Performance)|(Obs - Unobs) Acc. (Robustness)|
|---|---|---|
|**Flan-T5-XL**|||
|- 0-shot|47_._8 (_±_0_._6)|0_._6|
|- 1-shot|47_._5 (_±_0_._9)|0_._2|
|- 3-shot|**48.1** (_±_1.9)|**0.2**|
|- 0-shot + Aligned|48_._0 (_±_0_._3)|**0.2**|
|**Alpaca-7B**|||
|- 0-shot|40_._8 (_±_1_._4)|2_._2|
|- 1-shot|41_._3 (_±_2_._7)|1_._1|
|- 3-shot|41_._7 (_±_3.9)|1_._4|
|- 0-shot + Aligned|**42.1** (_±_0_._3)|**0.6**|



Table 15: The ICL experiments on MMLU with one-shot and three-shot examples provided. 

method. With the above results, we argue that the effect of in-context learning on robustness is a hybrid research question, in which the effect of examples needs to be studied independently. 

We provide the disaggregated results for few-shot experiments on MMLU and BBL in Appendix K.2. 

19 

Under review as a conference paper at ICLR 2024 

## D LLAMA2 V.S. LLAMA: DOES BETTER PRE-TRAINING PROVIDE STRONGER INSTRUCTION-FOLLOWING ROBUSTNESS? 

To provide a better understanding of what factors (aside from instruction-tuning) correlate with the ability of LLMs to robustly follow instructions, we compare the performance and brittleness of Llama2 Touvron et al. (2023b) and Llama Touvron et al. (2023a) when instruction-tuned with Alpaca Taori et al. (2023). Llama2 Touvron et al. (2023b) is an LLM trained on a relatively new mix of publicly available data. This pre-training corpus is about 40% larger than that used to train the original Llama. 

## D.1 TRAINING DETAILS 

We trained both Llama-7B and Llama2-7B on 4 A100 80GB GPUs with the Fully Sharded Data Parallel (FSDP) strategy. We adopted the exact same hyperparameters for training: We use the AdamW optimizer Loshchilov & Hutter (2017) with a learning rate of 2 _×_ 10 _[−]_[5] . We use a cosine learning rate scheduler with a warmup ratio of 0.03. We trained for a total of 3 epochs over 52k instruction and answer pairs collected in Alpaca, with a batch size of 128. Llama2 and Llama have different max context window sizes; to simplify things we set both to use a fixed max length of 512. 

## D.2 EVALUATION DETAILS 

Following the same procedure in Section 4.2, we reproduce experiments to evaluate the performance of Llama2-Alpaca on observed and unobserved instructions. We omit two held-in datasets that we evaluated to avoid data leakage. For each category of tasks (MMLU, QA, MC, BC, Sum, Trans), the average performance is calculated, as well as the percentage delta of the performance between observed and unobserved instructions on each task. 

## D.3 RESULTS 

Figure 8 shows that although Llama2-Alpaca is consistently superior across all categories of tasks compared to Llama-Alpaca, the robustness (performance difference between observed and unobserved instructions) of the model **does not** always improve. On summarization datasets, binary classification, and MMLU, the robustness of Llama2-Alpaca is better than Llama-Alpaca, whereas, on translation datasets, QA, and multi-class classification, the instruction-following robustness is worse. Hence, it cannot be concluded that better pre-training provides stronger instruction-following robustness after instruction-tuning. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0020-09.png)


**----- Start of picture text -----**<br>
0.6<br>1.75 Model<br>Llama-Alpaca-7B<br>0.5 Llama2-Alpaca-7B<br>1.50<br>0.4 1.25<br>1.00<br>0.3<br>0.75<br>0.2<br>0.50<br>0.1<br>0.25<br>0.0 0.00<br>MMLU Translation Summarization BBL-QA BBL-MC BBL-BC MMLU Translation Summarization BBL-QA BBL-MC BBL-BC<br>Categories Categories<br>Performance<br>%  Performance (Obs - Unobs)<br>**----- End of picture text -----**<br>


Figure 8: The overall results of Llama-Alpaca v.s. Llama2-Alpaca on the six aforementioned categories of held-out datasets. It could be seen that although the overall average performance of Llama2-Alpaca is slightly better than Llama-Alpaca (Left), there is no clear pattern that the performance difference between observed and unobserved instructions are narrowed down (Right). 

20 

Under review as a conference paper at ICLR 2024 

## E DIFFERENT (SIMILAR) OBJECTIVES INDUCE SIMILAR REPRESENTATIONS 

Besides the KL objective reported in the main paper, we explored two other objectives that similarly aim to align representations for practically equivalent instructions. Our goal with these experiments is to establish that the primary mechanism to improve robustness has to do with encouraging the model to internally represent equivalent instructions similarly; the particular choice of objective for realizing this is not critical. 

## E.1 CONTRASTIVE LEARNING ON PENULTIMATE LAYER LOGITS 

We use an extra contrastive term to push the inequivalent instructions away while aligning the equivalent instructions. Following the same batching strategy in Section 5, one training batch contains 2 seed task instructions and _n_ paraphrases of each. While all the other terms and loss remain the same as 5, we introduce the contrastive KL loss _LKL_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0021-05.png)


ˆ ˆ Where for vocabulary distribution of each paraphrase _yi_ , _yr_ is the seed instruction it is equivalent to, ˆ and _yr′_ is the seed instruction it isn’t equivalent to. 

## E.2 MSE LOSS ON THE PREFIX ENCODER HIDDEN STATE 

Instead of having to optimize the logits to induce closer representations for equivalent instructions, we tried to directly align the hidden representation of the soft prefix from the last encoder layer we they are concatenated to equivalent instructions. With the same batching strategy in Section 5, we optimize: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0021-09.png)


Where _h_[ˆ] _i_ and _h_[ˆ] _r_ are given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0021-11.png)


This method is effective and fast but cannot be used with decoder-only architecture LLMs. 

## E.3 RESULTS COMPARISON 

In some sense all of these objectives are equivalent in that we are trying to induce closer representations for similar instructions. The experimental results are reproduced here (we omit variances for simplicity): 

|**Methods**|**Avg. Obs. Acc. (MMLU)**|**Avg. Unobs. Acc. (MMLU)**|
|---|---|---|
|FT + KL|41.8|43.6|
|PT + KL|**48.1**|47.9|
|FT + Contrastive KL|41.9|43.6|
|PT + Contrastive KL|47.9|47.7|
|MSE on EncoderHS|48.1|**48.0**|



Table 16: The results of the representation alignment experiment on Flan-T5-3B with two alternative objectives that incent closer distance between similar instructions. 

Using these objectives to attempt to align instructions performs comparably (in terms of robustness) to the proposed soft prompt alignment strategy, but they are slower to converge than this method. Furthermore, the second approach is only suitable for encoder-decoder LLMs like Flan-T5. 

21 

Under review as a conference paper at ICLR 2024 

## F “CLOSER LOOK” AT SEMANTIC DISTANCES 

In Section 4.5 we reported a rather peculiar result: Over all the tasks we evaluated, incorrect but observed instructions perform better on average than correct but unobserved instructions. Does our proposed method for improving robustness (by aligning representations in latent space) change this particular result? We re-run the experiment reported in Figure 3 again with a model fit using the objective proposed in Section 5. 

In Table 17, we report a non-trivial improvement in all settings. Interestingly, with the proposed method ( **PT+KL** ) the average performance when using instructions that are **observed** and incorrect (but which belong to the same genre of task) still performs better than the task designer result; that said, the gap in performance is slightly narrowed compared to the baseline (7.1 difference for PT + KL vs 8). 

## F.1 NEGATION 

Previous studies Webson & Pavlick (2022); Jang et al. (2023) suggest the insensitivity to negation as an indicator of unrobust instruction-following ability of LLMs. The results reported in Table 17 and Section 4.5 also support such finding. Additionally, Table 17 shows that despite generally improving model’s performance and robustness in other settings, the proposed method has failed to bring significant improvement (drop in accuracy) in the **Negated** setting. Hence, we slight tilt the objective to see if training on "pushing opposite instructions away" would result in a more robust instruction-following ability for recognizing negation. 

Similar to Section 5, we take the same 1k seed instances from the flan collection and prompt (with 3 prompts) the GPT-4 to rewrite them with completely opposite instruction. All the other parameters for generation stay the same. 

Naturally, instead of clustering equivalent instructions together, we want to push the negated instructions away from the original instruction: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0022-08.png)


We train the model with the exact same hyperparameters in Section 5 for PT + KL and run it on the same benchmarks with the six different settings. We report the results along with the original method, denoted as Negation _[†]_ . Additionlly, We also report the results of jointly optimizing with negated and paraphrased instructions, denoted as Joint. 

|**Model**|Closest|Incorrect|Negated|Task Designer|Collected|Nonsensical|
|---|---|---|---|---|---|---|
|FLAN-T5-XL|||||||
|- Baseline|58.1_±_1_._8|52.2_±_8_._6|48.3_±_3_._8|44.2|55.3_±_4_._5|30.6_±_4_._4|
|- PT + KL|**60.5**_±_2_._4|**54.2**_±_7_._2|47.1_±_4_._9|**47.1**|**56.7**_±_3_._9|38.0_±_4_._1|
|- Negation_†_|58.6_±_2_._0|52.7_±_9_._2|**43.0**_±_4_._9|46.5|55.6_±_4_._3|35.7_±_5_._6|
|- Joint|59.3_±_1_._7|52.8_±_8_._7|44.2_±_4_._5|46.6|56.1_±_3_._5|36.1_±_5_._1|



Table 17: We report average result of six settings over benchmarks. For Negated setting, a negation is applied to the instruction so a lower accuracy is desired. Despite the fact that in Negation _[†]_ setting the model performs comparably with the baseline, its ability to recognize negation in the instruction is significantly improved. Training with the joint loss does realize some of the advantages of both negative and paraphrase alignment methods. Compared with the standard fine-tuned baseline (no KL), we see consistent improvements across all settings. But we see stronger improvements in the respective settings when we optimize for only paraphrases or negations alone. 

22 

Under review as a conference paper at ICLR 2024 

## G CATASTROPHIC FORGETTING 

In Table 7 one can observe that simply fine-tuning (FT) on the augmented paraphrase data significantly degredes performance when using both observed and unobserved instructions. We hypothesize that this may be an instance of _catastrophic forgetting_ , where when we fine-tune the instruction-tuned model in this way (on a relatively small set of paraphrased instruction instances) it compromises its ability to follow instructions more general. 

We design a simple study to test this hypothesis. Following the same setting of **FT** in Section 5, we finetune FLAN-T5-XL (which is instruction tuned) and T5-XL (which is not) on the same augmented data (4k paraphrases of task instructions) for 25 epochs, and evaluate the performance on (A) the validation set sampled from the flan collection Chung et al. (2022), (B) MMLU with observed instructions, and (C) MMLU with unobserved instructions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0023-04.png)


**----- Start of picture text -----**<br>
①<br>③<br>④<br>②<br>**----- End of picture text -----**<br>


Figure 9: Loss as a function of training epoch for FLAN-T5-3B and T5-3B, fine-tuned on the paraphrased instruction set. The first (left) plot shows the validation loss on the flan collection for both models; the second (middle) plot shows the validation loss on MMLU for T5-3B; the third (right) plot shows the validation loss on MMU for FLAN-T5-3B. The loss for FLAN-T5-3B increases almost monotonically (Line 1 and Lines 4); this strongly suggests catastrophic forgetting. Coult it be that the collected paraphrased instructions are simply not informative? The decreasing loss for T5-3B — indicated by Line 2 and Lines 3 — suggests otherwise, i.e., that the data does contain signal for instruction tuning (on a base model), but the set is somewhat small and degrades performance when the Flan checkpoint is used for initialization. 

What is happening at the representational level in these cases? In Figure 10 we compare the average similarity (analogous to Table 5) between observed and unobserved instructions after 25 epochs of fine-tuning. Rather than closing the “semantic gap”, fine-tuning FLAN-T5-XL (without the proposed KL term) _increases_ it. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0023-07.png)


**----- Start of picture text -----**<br>
Finetuning Epoch<br>1<br>35 25<br>30<br>25<br>20<br>15<br>10<br>5<br>0<br>Winowhy Language ID BBQ Lite Play Dialog Novel Conc.Hindu Know.Strange St. Strat. QA Logical Ded. Vitamin C Code Desc.Known Unk.Conc. Com. MMLU<br>Dataset<br>L2 Distance<br>**----- End of picture text -----**<br>


Figure 10: The average _ℓ_ 2 Distance between the unobserved instructions to their closest observed instructions on each datasets at epoch 1 and epoch 25. This result indicates that simply fine-tuning the model with paraphrases **does not** help to align equivalent instructions. 

23 

Under review as a conference paper at ICLR 2024 

## H CLOSER LOOK TO ROBUSTNESS AS FUNCTION OF SCALE? 

While we maintain our stance that the success of instruction-tuning for boosting the performance of modestly-sized LLMs is the most exciting aspect of this method, and therefore worthy on its own of study, we nonetheless agree that studying the relationship between robustness and scale of such models is interesting. 

However, we can draw valid comparisons only if we keep all other variables (model architecture, pretraining data, and instruction tuning set) fixed. Therefore, we are not able to study black-boxed models (like GPT-3.5 or GPT-4) using this approach. To include additional scale in our work, we therefore conduct the full experiments on different sizes of Flan-T5 ranging from 70M to 11B 4.3, as well as Flan-UL2 (20B), and we report the change of **delta performance** (observed instructions performance - unobserved instructions performance) in the table below: 

|**Model**|Number of Parameters|MMLU|QA|MC|BC|
|---|---|---|---|---|---|
|Flan-T5-Small|70M|-2.6_±_8.3|4.1_±_3.1|-0.4_±_1.4|-7.9_±_12.0|
|Flan-T5-Base|250M|0.0_±_0.9|1.6_±_6.2|1.2_±_4.3|1.5_±_8.3|
|Flan-T5-Large|770M|0.1_±_0.5|3.1_±_6.4|11.4_±_2.1|1.6_±_6.1|
|Flan-T5-XL|3B|0.6_±_0.7|3.0_±_4.6|3.5_±_3.2|5.5_±_7.8|
|Flan-T5-XXL|11B|0.5_±_0.6|3.4_±_5.7|2.8_±_3.1|2.0_±_9.1|
|Flan-T5-UL2|20B|0.2_±_0.5|3.4_±_7.3|0.6_±_0.3|1.9_±_7.1|



Table 18: The aggregated result of the delta performance (bserved instructions performance - unobserved instructions performance) as the Flan-T5 model scales. The results above (aggregated by task category) show that there is no apparent relationship between model size and the delta performance (our measure of robustness). 

More scientifically, we perform linear regression on the size of the model v.s. (observed - unobserved performance) over data points from all datasets, this yields a slope of 4 _._ 45 _×_ 10 _[−]_[13] with a p-value of 0.087 ( _R_[2] is 0.002). This suggests that model size may weakly correlate with increased robustness, but this relationship is not significant the _p_ = 0 _._ 05 level, and the relationship is tenuous. In other words, scale does seem to improve robustness, but only very moderately. 

24 

Under review as a conference paper at ICLR 2024 

## I REPRESENTATIONAL SIMILARITY AND MODEL PERFORMANCE 

We re-generate the visualization from Figure 4 for all T5 model sizes in Figure . The qualitative result—representations of tokens following observed instructions are generally dissimilar from those following unobserved instructions—remains largely consistent, although it is less pronounced, e.g., for XL (in particular here, the MMLU samples are not as entangled as we might expect). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0025-03.png)


Figure 11: We reproduce Figure 4 from the main paper over all T5 model sizes. 

25 

Under review as a conference paper at ICLR 2024 

## J COLLECTION PROCEDURES AND SURVEYS 

We recruited in total 36 annotators to contribute diverse human-written instructions for various NLP tasks. These were graduate students in NLP who volunteered to write instructions following an open request. We explicitly asked volunteers not to spend more than 30 minutes on this task. We estimate that the maximum time spent by any one individual on this task was one hour, and most were (much) quicker. In light of the minimal workload per individual being done on a strictly volunteer basis, we did not offer compensationl; we emphasize that it was made clear upfront to volunteers that this quick amount of work would be uncompensated. 

The entire annotation process was done online (via a shared Google Drive directory) with detailed instructions provided. We reproduce the template of the invitation letter, annotation instructions, and annotation form below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0026-04.png)


**----- Start of picture text -----**<br>
Hi XXX!<br>Thank you for helping me with my research! It may take up to 30 minutes of your time, and your<br>participation is deeply thanked and will be acknowledged in the final research paper.<br>The tasks/settings/ID of your instructions are:<br>MMLU General - Zero-shot - 1<br>MMLU General - Few-shot - 1<br>Hindu Knowledge - Zero-shot - 1<br>Hindu Knowledge - Few-shot - 1<br>Known Unknowns - Zero-shot - 1<br>Known Unknowns - Few-shot - 1<br>Novel Concepts - Zero-shot - 1<br>Novel Concepts - Few-shot - 1<br>winowhy - Zero-shot - 1<br>winowhy - Few-shot - 1<br>BBQ-Lite - Zero-shot - 1<br>BBQ-Lite - Few-shot - 1<br>Strange Stories - Zero-shot - 1<br>Strange Stories - Few-shot - 1<br>Emoji Movie - Zero-shot - 1<br>Emoji Movie - Few-shot - 1<br>To enter the google docs, click on this link: http://xxx.com/xxx<br>Be sure to read the instructions.docx on the front page for detailed instructions!<br>It is optimal that you can get it done before May.1st. If you have any questions regarding any of<br>the procedures, please feel free to text me anytime for clarification!<br>Thank you!<br>**----- End of picture text -----**<br>


Figure 12: Invitation note sent to participant 

26 

Under review as a conference paper at ICLR 2024 

First, I would like to express my appreciation for helping with my research project again! 

## **Background** 

This research aims to evaluate the robustness of the instruction-tuned Language Models (LMs) with respect to the variation of instructions in zero-shot or few-shot settings. It is commonly acknowledged that multitask instruction tuning on a language model improves its zero-shot and few-shot ability. The model can understand and generalize to unseen instructions that users provide at inference time. 

For instance, I use this instruction (prompt) as the Prefix: 

“Complete this code written in Java SE11 …” 

to the actual code, I want to complete, the LM can understand the task and perform inference accordingly. 

## **Goal** 

As an NLP practitioner and expert, you can provide **instructions** that will prompt the instruction-tuned LMs well for **the given tasks** . The models in which the instructions might be evaluated are: 

- GPT-4 / ChatGPT 

- Text Davinci 

- Flan-PaLM 

- Flan-T5 

- T0++ 

- mT0 

- MetalICL 

- OPT-IML 

- ChatGLM 

- Alpaca 

You are very well come to use your experience on these models to come up with the instruction you think will **perform the best** . 

## **Tasks** 

The participation will take approximately **30 minutes** . You will be given **10-15** tasks/settings. For each task/setting, you are going to put your instruction in the row indicated by the order number. For instance: 

- “Auto Debugging - Zero-Shot - 5” 

means that you are assigned to write an instruction on the task **“Auto Debugging”** with the setting **“Zero-shot,** ” and you are putting your answer in the row with ID **5** . 

Figure 13: The first page of the instruction given to the annotator 

27 

Under review as a conference paper at ICLR 2024 

For each task, there is a folder with the exact same name. In the folder, there is a .docx file. Open the file, you will see detailed information about the task, including **input-output format, task description,** and **examples.** There will be two main tables - one to record instructions in **Zeroshot** and one to record instructions in **Few-shot** s. In each table, there will be an example provided. **Be sure to put your instruction under the correct table and follow the format of the example!** 

If you want to see more examples, the “task.json” all the examples in the test set so you may have a better idea 

**Data source** 

The tasks given to you are sampled from the benchmarks BBH-Lite and MMLU. Thank you! 

Figure 14: The second page of the instruction given to the annotator 

28 

Under review as a conference paper at ICLR 2024 

- Thank you for helping me on this research project! The goal is to gather instructions from experienced **NLP researchers** on various downstream tasks incorporated in the benchmark _BBH_ . Your task is to: - Write down the instruction (prompt) for this task that you think will **work the best** for this task on **instruction-tuned** Seq2Seq LMs (Flan-T5-XXL, Davinci-text-003, OPTIML, etc.) at zero-shot and few-shots (in-context learning). 

- - Please put your instruction in the **corresponding row** in the tables. The few-shots table is one page below the zero-shot table. Please use {…} to denote corresponding information. Note: you **do not** need to use all the information if you think some are distractions. 

- - For multiple choice tasks, you may either formulate the instruction to let the model output the **exact text** or **number/letter** of the text. Same goes for classification task. 

- - **Task Information** provides an overview of the task, including its input, output, and task description; **Example** provides an example to the test set so you may have a better grasp of the nature of the task; the tables of **Zero-shot Instruction** and **Few-shots Instruction** are in the following pages. 

- - Instead of using “\n” or “\t”, you may directly use enter or tab. - The given example also represents the **average** length of the input/output for this task. You may assume the maximum token length of the LM is **4096** 

- **Task Information Dataset** BIG-Bench **Task** Code Line Description **Metric** Accuracy **Task description** Give an English language description of Python code **Input** program, choiceA, choiceB, choiceC, choiceD **Output** answer 

Figure 15: The first page of the dataset information 

29 

Under review as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models_images/iclr_2024_Evaluating_the_Zero-shot_Robustness_of_Instruction-tuned_Language_Models.pdf-0030-01.png)


**----- Start of picture text -----**<br>
Example:<br>Input<br>- program:  for i in range(23):\n\t print(i)<br>- choiceA:  prints values from 0 to 22<br>- choiceB:  computes first 10 prime numbers<br>- choiceC:  prints values from 1 to 10<br>- choiceD:  prints 'hello world' to the terminal<br>Output<br>- answer:  prints values from 0 to 22 /  A<br>Zero-shot Instruction:<br>You are given:<br>- {program}: the text sequence of the input code<br>- {choiceA}, {choiceB}, {choiceC}, {choiceD}: choices of the interpretation<br>ID  Instruction<br>Example  [Give an English language description of Python code{program} A.  {choiceA} B.  {choiceB} C.  ]<br>{choiceC} D.  {choiceD}<br>English language description:<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>**----- End of picture text -----**<br>


Figure 16: The second page of the dataset information 

30 

Under review as a conference paper at ICLR 2024 

## K DISAGGREGATED RESULTS 

## K.1 MAIN RESULTS AND SCALING RESULTS 

In the main paper we reported aggregated results over benchmark corpora. Here we report results on individual datasets for BBL. For MMLU, we evaluate all 57 datasets together, because these are all QA tasks (and we would want a QA model to be capable of answering questions across a diverse set of domains). We report means and stadard deviations of the accuracies achieved over all instructions in Table 20. The numbers on the left of the setting suggest the number of instructions used. We also share even more granular results—reporting the performance for each instruction—in CSV files provided in the supplemental material. 

|||MMLU|MMLU|||
|---|---|---|---|---|---|
|**Model**|Flan-T5-XL|Flan-T5-XXL|T0pp-11B|Alpaca-7B|Alpaca-13B|
|**MMLU**||||||
|OBSERVED|**48.1 (**_±_**0.3)**|**53.2 (**_±_**0.2)**|48.3 (_±_0.9)|**41.9 (**_±_**0.6)**|**47.8 (**_±_**0.5)**|
|UNOBSERVED|47.5 (_±_0.9)|52.7 (_±_0.8)|**48.5 (**_±_**0.9)**|39.7 (_±_2.2)|47.0 (_±_0.8)|



Table 19: Granular results corresponding to the subset reported in Table 3, on each dataset of MMLU. We treat all tasks in MMLU equally as general QA and computed the overall accuracy. 

|||MMLU|MMLU|||
|---|---|---|---|---|---|
|**Size Variance**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**MMLU**||||||
|OBSERVED|29.4 (_±_1.0)|**34.1 (**_±_**0.4)**|**41.1 (**_±_**0.2)**|**48.1 (**_±_**0.3)**|**53.2 (**_±_**0.2)**|
|UNOBSERVED|**29.6 (**_±_**0.9)**|33.8 (_±_1.2)|40.7 (_±_0.7)|47.5 (_±_0.9)|52.7 (_±_0.8)|



Table 20: Granular results corresponding to the subset reported in Figure 2b, on each dataset of MMLU. We treat all tasks in MMLU equally as general QA and computed the overall accuracy. 

31 

Under review as a conference paper at ICLR 2024 

|||BBL-QA|BBL-QA|||
|---|---|---|---|---|---|
|**Model**|Flan-T5-XL|Flan-T5-XXL|T0pp-11B|Alpaca-7B|Alpaca-13B|
|**BBQ Lite**||||||
|OBSERVED|66.5 (_±_1.5)|**77.4 (**_±_**2.4)**|**51.8 (**_±_**5.3)**|32.6 (_±_1.0)|43.5 (_±_1.4)|
|UNOBSERVED|**67.0 (**_±_**7.0)**|73.7 (_±_11.4)|51.6 (_±_3.0)|**33.1 (**_±_**1.3)**|**45.5 (**_±_**2.9)**|
|**Code Desc.**||||||
|OBSERVED|**73.6 (**_±_**3.4)**|**83.6 (**_±_**1.7)**|70.3 (_±_3.0)|**70.2 (**_±_**2.5)**|**85.2 (**_±_**2.4)**|
|UNOBSERVED|69.7 (_±_12.4)|72.9 (_±_22.2)|**70.5 (**_±_**3.7)**|67.5 (_±_11.3)|82.2 (_±_8.5)|
|**Hindu Know.**||||||
|OBSERVED|**52.4 (**_±_**1.6)**|53.9 (_±_1.8)|**57.1 (**_±_**2.5)**|50.9**(**_±_**2.1)**|63.8 (_±_0.7)|
|UNOBSERVED|47.1 (_±_5.4)|**56.5 (**_±_**3.5)**|53.2 (_±_3.0)|49.8 (_±_5.1)|**63.9 (**_±_**1.1)**|
|**Known Unk.**||||||
|OBSERVED|**79.3 (**_±_**2.5)**|**84.7 (**_±_**2.1)**|70.9 (_±_10.2)|**75.2 (**_±_**4.7)**|**81.9 (**_±_**4.3)**|
|UNOBSERVED|69.0 (_±_6.7)|80.6 (_±_8.1)|**76.1 (**_±_**5.9)**|60.9 (_±_11.2)|71.1 (_±_16.3)|
|**Logical Ded.**||||||
|OBSERVED|**52.5 (**_±_**1.0)**|**58.0 (**_±_**0.7)**|**45.5 (**_±_**0.8)**|**25.5 (**_±_**1.1)**|**29.2 (**_±_**1.3)**|
|UNOBSERVED|52.1 (_±_1.1)|57.8 (_±_0.6)|45.3 (_±_1.2)|24.5 (_±_2.3)|28.0 (_±_1.6)|
|**Novel Conc.**||||||
|OBSERVED|29.8 (_±_2.4)|**50.1 (**_±_**1.9)**|28.8 (_±_2.9)|**37.2 (**_±_**5.2)**|**20.0 (**_±_**3.1)**|
|UNOBSERVED|**31.2 (**_±_**5.0)**|46.0 (_±_5.4)|**31.5 (**_±_**5.1)**|36.1 (_±_7.6)|19.6 (_±_4.0)|
|**Logic Grid**||||||
|OBSERVED|**41.8 (**_±_**1.1)**|**43.2 (**_±_**1.8)**|**37.6 (**_±_**1.7)**|24.4 (_±_1.9)|**29.3 (**_±_**0.8)**|
|UNOBSERVED|38.6 (_±_5.4)|39.6 (_±_4.4)|36.4 (_±_3.6)|**25.4 (**_±_**1.1)**|28.7 (_±_1.2)|
|**Conc. Com.**||||||
|OBSERVED|**75.9 (**_±_**1.8)**|**75.0 (**_±_**2.6)**|73.3 (_±_2.3)|**58.7 (**_±_**4.0)**|**63.0 (**_±_**2.2)**|
|UNOBSERVED|75.0 (_±_1.9)|73.6 (_±_4.8)|**74.2 (**_±_**2.6)**|55.9 (_±_6.2)|61.1 (_±_3.3)|



Table 21: Granular results corresponding to the subset reported in Table 3, on each dataset of category BBL-QA. 

32 

Under review as a conference paper at ICLR 2024 

|||BBL-QA|BBL-QA|||
|---|---|---|---|---|---|
|**Size Variance**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**BBQ Lite**||||||
|OBSERVED|28.3 (_±_1.3)|**51.5 (**_±_**1.4)**|56.6 (_±_2.0)|66.5 (_±_1.5)|**77.4 (**_±_**2.4)**|
|UNOBSERVED|**28.6 (**_±_**4.3)**|50.5 (_±_4.1)|**56.7 (**_±_**4.7)**|**67.0 (**_±_**7.0)**|73.7 (_±_11.4)|
|**Code Desc.**||||||
|OBSERVED|22.0 (_±_4.0)|**55.7 (**_±_**3.3)**|**72.4 (**_±_**3.2)**|**73.6 (**_±_**3.4)**|**83.6 (**_±_**1.7)**|
|UNOBSERVED|**32.1 (**_±_**7.2)**|48.6 (_±_7.2)|63.3 (_±_14.2)|69.7 (_±_12.4)|72.9 (_±_22.2)|
|**Hindu Know.**||||||
|OBSERVED|25.1 (_±_15.2)|**30.7 (**_±_**2.6)**|34.9 (_±_0.9)|**52.4 (**_±_**1.6)**|53.9 (_±_1.8)|
|UNOBSERVED|**31.6 (**_±_**10.7)**|26.9 (_±_4.4)|**37.5 (**_±_**7.0)**|47.1 (_±_5.4)|**56.5 (**_±_**3.5)**|
|**Known Unk.**||||||
|OBSERVED|49.9 (_±_1.9)|**66.9 (**_±_**4.7)**|**76.2 (**_±_**3.5)**|**79.3 (**_±_**2.5)**|**84.7 (**_±_**2.1)**|
|UNOBSERVED|**52.8 (**_±_**5.2)**|63.8 (_±_7.3)|68.4 (_±_11.1)|69.0 (_±_6.7)|80.6 (_±_8.1)|
|**Logical Ded.**||||||
|OBSERVED|19.8 (_±_0.7)|27.1 (_±_1.3)|45.9 (_±_1.1)|**52.5 (**_±_**1.0)**|**58.0 (**_±_**0.7)**|
|UNOBSERVED|**19.9 (**_±_**0.4)**|**28.9 (**_±_**2.8)**|**46.4 (**_±_**2.6)**|52.1 (_±_1.1)|57.8 (_±_0.6)|
|**Novel Conc.**||||||
|OBSERVED|**22.9 (**_±_**9.2)**|15.9 (_±_5.4)|**31.0 (**_±_**2.3)**|29.8 (_±_2.4)|**50.1 (**_±_**1.9)**|
|UNOBSERVED|19.3 (_±_3.6)|**16.8 (**_±_**4.0)**|28.4 (_±_6.9)|**31.2 (**_±_**5.0)**|46.0 (_±_5.4)|
|**Logic Grid**||||||
|OBSERVED|22.3 (_±_4.0)|**31.7 (**_±_**0.8)**|32.6 (_±_2.1)|**41.8 (**_±_**1.1)**|**43.2 (**_±_**1.8)**|
|UNOBSERVED|**28.8 (**_±_**3.1)**|29.4 (_±_5.1)|**34.1 (**_±_**2.8)**|38.6 (_±_5.4)|39.6 (_±_4.4)|
|**Conc. Com.**||||||
|OBSERVED|30.4 (_±_10.6)|**55.6 (**_±_**5.1)**|**64.2 (**_±_**1.9)**|**75.9 (**_±_**1.8)**|**75.0 (**_±_**2.6)**|
|UNOBSERVED|**32.2 (**_±_**16.7)**|54.5 (_±_9.4)|58.1 (_±_11.6)|75.0 (_±_1.9)|73.6 (_±_4.8)|



Table 22: Granular results corresponding to the subset reported in Figure 2b, on each dataset of category BBL-QA. 

33 

Under review as a conference paper at ICLR 2024 

|||BBL-BC|BBL-BC|||
|---|---|---|---|---|---|
|**Model**|Flan-T5-XL|Flan-T5-XXL|T0pp-11B|Alpaca-7B|Alpaca-13B|
|**Play Dialog**||||||
|OBSERVED|**61.6 (**_±_**5.8)**|51.8 (_±_9.5)|**62.7 (**_±_**0.4)**|**45.0 (**_±_**2.0)**|**53.4 (**_±_**5.8)**|
|UNOBSERVED|53.0 (_±_6.9)|**58.1 (**_±_**4.4)**|55.2 (_±_8.1)|42.9 (_±_7.9)|42.9 (_±_8.8)|
|**Strat. QA**||||||
|OBSERVED|58.7 (_±_3.3)|**64.2 (**_±_**3.0)**|51.0 (_±_1.8)|53.0 (_±_2.1)|56.7 (_±_3.8)|
|UNOBSERVED|**60.7 (**_±_**7.5)**|59.3 (_±_6.1)|**54.5 (**_±_**0.9)**|**53.3 (**_±_**4.1)**|**61.0 (**_±_**1.9)**|
|**Strange St.**||||||
|OBSERVED|69.3 (_±_4.4)|71.0 (_±_7.3)|**51.2 (**_±_**5.1)**|**67.0 (**_±_**4.7)**|**69.8 (**_±_**5.0)**|
|UNOBSERVED|**70.5 (**_±_**7.0)**|**77.4 (**_±_**6.1)**|48.4 (_±_3.1)|59.9 (_±_9.4)|57.5 (_±_5.6)|
|**Winowhy**||||||
|OBSERVED|**76.5 (**_±_**1.9)**|**75.6 (**_±_**4.0)**|**99.6 (**_±_**1.0)**|50.1 (_±_4.8)|51.9 (_±_4.6)|
|UNOBSERVED|60.2 (_±_6.2)|59.7 (_±_7.7)|60.9 (_±_5.1)|**53.4 (**_±_**4.6)**|**55.2 (**_±_**6.0)**|



Table 23: Granular results corresponding to the subset reported in Table 3, on each dataset of category BBL-BC. 

|||BBL-BC|BBL-BC|||
|---|---|---|---|---|---|
|**Size Variance**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**Play Dialog**||||||
|OBSERVED|51.6 (_±_13.3)|54.6 (_±_10.7)|**59.0 (**_±_**6.7)**|**61.6 (**_±_**5.8)**|51.8 (_±_9.5)|
|UNOBSERVED|**61.6 (**_±_**4.6)**|**56.3 (**_±_**10.9)**|57.3 (_±_7.8)|53.0 (_±_6.9)|**58.1 (**_±_**4.4)**|
|**Strat. QA**||||||
|OBSERVED|**52.3 (**_±_**1.0)**|48.9 (_±_2.1)|**60.9 (**_±_**1.3)**|58.7 (_±_3.3)|**64.2 (**_±_**3.0)**|
|UNOBSERVED|51.5 (_±_2.7)|**52.9 (**_±_**1.3)**|53.9 (_±_3.8)|**60.7 (**_±_**7.5)**|59.3 (_±_6.1)|
|**Strange St.**||||||
|OBSERVED|41.3 (_±_10.3)|**43.1 (**_±_**4.2)**|54.4 (_±_1.2)|69.3 (_±_4.4)|71.0 (_±_7.3)|
|UNOBSERVED|**55.9 (**_±_**18.5)**|42.0 (_±_5.5)|**67.9 (**_±_**8.0)**|**70.5 (**_±_**7.0)**|**77.4 (**_±_**6.1)**|
|**Winowhy**||||||
|OBSERVED|**54.8 (**_±_**1.6)**|55.9 (_±_7.6)|**60.4 (**_±_**9.8)**|**76.5 (**_±_**1.9)**|**75.6 (**_±_**4.0)**|
|UNOBSERVED|53.7 (_±_5.1)|**57.1 (**_±_**6.7)**|53.5 (_±_4.9)|60.2 (_±_6.2)|59.7 (_±_7.7)|



Table 24: Granular results for Figure 2b on each dataset of category BBL-BC 

|||BBL-MC|BBL-MC|||
|---|---|---|---|---|---|
|**Model**|Flan-T5-XL|Flan-T5-XXL|T0pp-11B|Alpaca-7B|Alpaca-13B|
|**Language ID**||||||
|OBSERVED|**32.6 (**_±_**0.2)**|**38.9 (**_±_**0.3)**|**15.7 (**_±_**3.0)**|12.9 (_±_0.7)|18.5 (_±_0.7)|
|UNOBSERVED|25.5 (_±_7.3)|31.6 (_±_9.4)|14.3 (_±_2.4)|**14.7 (**_±_**1.7)**|**21.7 (**_±_**0.7)**|
|**Vitamin C**||||||
|OBSERVED|78.6 (_±_1.1)|78.5 (_±_0.7)|68.3 (_±_1.1)|**51.4 (**_±_**3.6)**|**54.9 (**_±_**2.9)**|
|UNOBSERVED|78.6 (_±_3.6)|**80.2 (**_±_**1.6)**|**68.5 (**_±_**2.4)**|18.1 (_±_5.3)|23.6 (_±_14.2)|



Table 25: Granular results for Table 3 on each dataset of category BBL-MC 

34 

Under review as a conference paper at ICLR 2024 

|||BBL-MC|BBL-MC|||
|---|---|---|---|---|---|
|**Size Variance**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**Language ID**||||||
|OBSERVED|**11.9 (**_±_**0.2)**|**17.0 (**_±_**0.3)**|**25.8 (**_±_**0.3)**|**32.6 (**_±_**0.2)**|**38.9 (**_±_**0.3)**|
|UNOBSERVED|9.5 (_±_0.2)|12.4 (_±_1.5)|19.2 (_±_4.4)|25.5 (_±_7.3)|31.6 (_±_9.4)|
|**Vitamin C**||||||
|OBSERVED|**46.6 (**_±_**4.0)**|60.7 (_±_5.6)|**72.6 (**_±_**1.5)**|78.6 (_±_1.1)|78.5 (_±_0.7)|
|UNOBSERVED|40.8 (_±_4.2)|**63.0 (**_±_**4.6)**|36.4 (_±_0.8)|78.6 (_±_3.6)|**80.2 (**_±_**1.6)**|



Table 26: Granular results for Figure 2b on each dataset of category BBL-MC 

35 

Under review as a conference paper at ICLR 2024 

## K.2 ICL FEW-SHOT EXPERIMENT RESULTS 

|||MMLU **(One-shot)**|MMLU **(One-shot)**|||
|---|---|---|---|---|---|
|**Flan-T5**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**MMLU**||||||
|OBSERVED|29.3 (_±_0.9)|**33.9 (**_±_**0.5)**|**40.7 (**_±_**0.2)**|**47.5 (**_±_**0.2)**|52.7 (_±_0.2)|
|UNOBSERVED|**29.6 (**_±_**0.6)**|33.8 (_±_1.0)|40.4 (_±_0.9)|47.5 (_±_0.7)|**52.8 (**_±_**1.0)**|



Table 27: Granular results on each dataset of MMLU for Flan-T5 models and one-shot ICL. We group all QA tasks in MMLU and report overall accuracy on these. 

|||BBL-QA|**(One-shot)**|||
|---|---|---|---|---|---|
|**Flan-T5**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**BBQ Lite**||||||
|OBSERVED|**29.6 (**_±_**2.2)**|50.5 (_±_1.7)|57.0 (_±_2.0)|66.7 (_±_1.6)|77.2 (_±_2.7)|
|UNOBSERVED|28.9 (_±_1.4)|**51.0 (**_±_**3.6)**|**58.0 (**_±_**2.7)**|**69.0 (**_±_**5.9)**|**77.6 (**_±_**5.5)**|
|**Code Desc.**||||||
|OBSERVED|20.5 (_±_3.1)|**56.9 (**_±_**5.1)**|**76.0 (**_±_**2.2)**|**73.6 (**_±_**1.8)**|**85.3 (**_±_**1.6)**|
|UNOBSERVED|**25.6 (**_±_**8.7)**|43.6 (_±_8.4)|53.5 (_±_16.0)|65.8 (_±_15.3)|75.0 (_±_12.3)|
|**Hindu Know.**||||||
|OBSERVED|**24.5 (**_±_**12.7)**|**30.5 (**_±_**3.6)**|36.2 (_±_1.7)|**50.9 (**_±_**1.3)**|52.9 (_±_1.9)|
|UNOBSERVED|23.0 (_±_5.8)|22.4 (_±_5.8)|**37.7 (**_±_**4.7)**|50.2 (_±_5.8)|**54.6 (**_±_**3.2)**|
|**Known Unk.**||||||
|OBSERVED|**49.2 (**_±_**3.8)**|**66.7 (**_±_**8.3)**|**73.6 (**_±_**2.7)**|**76.3 (**_±_**2.0)**|**84.7 (**_±_**3.8)**|
|UNOBSERVED|49.1 (_±_4.1)|60.2 (_±_7.3)|67.7 (_±_12.9)|63.7 (_±_9.8)|76.0 (_±_12.7)|
|**Logical Ded.**||||||
|OBSERVED|20.2 (_±_0.4)|26.8 (_±_1.0)|**45.9 (**_±_**1.1)**|**53.0 (**_±_**0.7)**|**58.2 (**_±_**0.5)**|
|UNOBSERVED|**20.2 (**_±_**0.8)**|**27.8 (**_±_**3.5)**|44.3 (_±_8.6)|48.6 (_±_10.1)|55.0 (_±_10.5)|
|**Novel Conc.**||||||
|OBSERVED|**21.3 (**_±_**4.3)**|**14.8 (**_±_**4.9)**|**29.1 (**_±_**4.0)**|31.2 (_±_2.0)|**47.7 (**_±_**3.1)**|
|UNOBSERVED|17.2 (_±_6.8)|14.7 (_±_9.0)|24.7 (_±_6.2)|**35.1 (**_±_**5.2)**|42.5 (_±_9.0)|
|**Conc. Com.**||||||
|OBSERVED|28.0 (_±_3.6)|**38.5 (**_±_**2.3)**|**65.0 (**_±_**1.8)**|**77.7 (**_±_**1.6)**|**77.1 (**_±_**2.2)**|
|UNOBSERVED|**28.6 (**_±_**4.5)**|36.3 (_±_6.4)|58.2 (_±_10.7)|74.8 (_±_11.9)|75.2 (_±_2.8)|



Table 28: Granular results on each dataset of category BBL-QA with one-shot in-context learning. 

36 

Under review as a conference paper at ICLR 2024 

|||BBL-BC **(One-shot)**|BBL-BC **(One-shot)**|||
|---|---|---|---|---|---|
|**Flan-T5**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**Play Dialog**||||||
|OBSERVED|49.8 (_±_11.9)|**54.4 (**_±_**10.3)**|**58.1 (**_±_**5.7)**|**57.8 (**_±_**3.3)**|43.9 (_±_4.2)|
|UNOBSERVED|**55.8 (**_±_**10.2)**|52.5 (_±_10.7)|48.6 (_±_8.7)|46.3 (_±_3.9)|**51.3 (**_±_**3.3)**|
|**Strat. QA**||||||
|OBSERVED|52.5 (_±_0.8)|49.3 (_±_3.1)|**60.6 (**_±_**1.5)**|60.6 (_±_6.2)|**66.2 (**_±_**2.6)**|
|UNOBSERVED|**53.2 (**_±_**0.0)**|**53.3 (**_±_**0.8)**|55.9 (_±_4.3)|**61.2 (**_±_**6.0)**|62.2 (_±_5.5)|
|**Strange St.**||||||
|OBSERVED|40.7 (_±_12.3)|**41.8 (**_±_**2.3)**|51.7 (_±_1.6)|74.4 (_±_3.6)|78.5 (_±_2.1)|
|UNOBSERVED|**46.7 (**_±_**5.1)**|37.9 (_±_8.6)|**56.3 (**_±_**3.0)**|**78.7 (**_±_**3.2)**|**83.2 (**_±_**7.6)**|
|**Winowhy**||||||
|OBSERVED|**52.7 (**_±_**2.8)**|57.3 (_±_6.2)|**62.1 (**_±_**5.9)**|**77.2 (**_±_**0.6)**|**76.7 (**_±_**1.0)**|
|UNOBSERVED|48.1 (_±_4.2)|**58.3 (**_±_**8.1)**|57.1 (_±_8.8)|66.3 (_±_8.8)|65.5 (_±_9.9)|



Table 29: Granular results on each dataset of category BBL-BC with one-shot in-context learning. 

|||BBL-MC|**(One-shot)**|||
|---|---|---|---|---|---|
|**Flan-T5**|Small (80M)|Base (250M)|Large (780M)|XL (3B)|XXL (11B)|
|**Language ID**||||||
|OBSERVED|**11.7 (**_±_**0.2)**|**13.5 (**_±_**2.8)**|**25.6 (**_±_**0.4)**|**31.8 (**_±_**0.3)**|**38.7 (**_±_**0.5)**|
|UNOBSERVED|9.7 (_±_0.9)|11.6 (_±_1.5)|16.9 (_±_5.4)|20.3 (_±_7.5)|28.4 (_±_10.3)|
|**Vitamin C**||||||
|OBSERVED|**50.9 (**_±_**1.6)**|60.9 (_±_5.4)|**73.3 (**_±_**0.8)**|**78.6 (**_±_**1.4)**|80.4 (_±_0.5)|
|UNOBSERVED|50.1 (_±_0.9)|**64.5 (**_±_**1.8)**|73.2 (_±_1.5)|77.5 (_±_9.3)|**80.7 (**_±_**3.6)**|



Table 30: Granular results on each dataset of category BBL-MC with one-shot in-context learning. 

37 

Under review as a conference paper at ICLR 2024 

## K.3 REPRESENTATION ALIGNMENT RESULTS 

In the main paper we reported the aggregated results over benchmark corpora. We report means and stadard deviations of the accuracies achieved over all instructions in Table 31 and 32. We also share even more granular results—reporting the performance for each instruction—in CSV files provided in the supplemental material. 

|||Flan-T5-XL|||Alpaca-7B||
|---|---|---|---|---|---|---|
||||MMLU||||
|**Dataset**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|**MMLU**|||||||
|- Baseline|48.0_±_0.3|47.3_±_0.9|47.7_±_0.6|42.1_±_0.4|40.6_±_1.6|41.3_±_1.0|
|- Aligned|**48.4**_±_0.2|**47.9**_±_0.5|**48.2**_±_0.6|**42.9**_±_1.4|**41.7**_±_2.0|**42.3**_±_1.0|
|**Overall**∆|0.4_±_0_._0|0.6_±_0_._0|0.5_±_0_._0|0.8_±_0_._0|1.1_±_0_._0|1.0_±_0_._0|
||||BBL-MC||||
|**Dataset**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|**Language ID**|||||||
|- Baseline|**32.7**_±_0.1|29.2_±_1.0|30.9_±_0.6|12.6_±_0.9|14.8_±_1.7|13.7_±_1.3|
|- Aligned|32.6_±_0.2|**29.4**_±_0.8|**31.0**_±_0.6|**12.9**_±_0.5|**16.7**_±_1.0|**14.8**_±_1.3|
|**Vitamin C**|||||||
|- Baseline|77.6_±_0.0|80.9_±_1.3|79.3_±_0.7|45.3_±_0.5|13.3_±_0.1|29.3_±_0.3|
|- Aligned|**78.3**_±_0.1|**81.4**_±_1.5|**79.8**_±_0.7|**45.6**_±_0.9|**18.5**_±_0.4|**32.1**_±_0.3|
|**Overall**∆|0.3_±_0_._4|0.3_±_0_._1|0.3_±_0_._2|0.3_±_0_._0|3.5_±_1_._6|1.9_±_0_._8|
||||BBL-BC||||
|**Dataset**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|**Strat. QA**|||||||
|- Baseline|58.4_±_1.6|59.9_±_8.0|59.2_±_4.8|53.4_±_2.0|51.6_±_4.4|52.5_±_3.2|
|- Aligned|**58.6**_±_2.9|**60.6**_±_6.6|**59.6**_±_4.8|**54.6**_±_3.1|**55.2**_±_5.5|**54.9**_±_3.2|
|**Strange St.**|||||||
|- Baseline|**69.1**_±_2.9|69.4_±_4.1|**69.2**_±_3.5|69.6_±_1.3|56.1_±_4.3|62.9_±_2.8|
|- Aligned|59.0_±_4.1|**69.8**_±_5.0|64.4_±_3.5|**70.5**_±_1.1|**59.5**_±_6.8|**65.0**_±_2.8|
|**Play Dialog**|||||||
|- Baseline|58.0_±_10.9|52.2_±_6.8|55.1_±_8.9|45.0_±_2.0|45.6_±_10.8|45.3_±_6.4|
|- Aligned|**61.4**_±_3.0|**58.6**_±_7.0|**60.0**_±_8.9|**45.7**_±_3.3|**48.8**_±_11.8|**47.2**_±_6.4|
|**Winowhy**|||||||
|- Baseline|76.2_±_1.9|54.3_±_11.8|65.3_±_6.9|49.5_±_8.9|55.7_±_0.1|52.6_±_4.5|
|- Aligned|**77.6**_±_3.3|**54.8**_±_11.6|**66.2**_±_6.9|**50.2**_±_8.4|**58.6**_±_1.9|**54.4**_±_4.5|
|**Overall**∆|-1.3_±_5_._2|2.0_±_2_._5|0.4_±_3_._5|0.9_±_0_._2|3.2_±_0_._3|2.1_±_0_._2|



Table 31: Granular results corresponding to the MMLU, BBL-MC, and BBL-BC subset reported in Table 7. Baseline refers to the performance of the original instruction-tuned model. We observe the significant improvement on following unobserved instructions consistently. 

38 

Under review as a conference paper at ICLR 2024 

|||Flan-T5-XL|Flan-T5-XL||Alpaca-7B||
|---|---|---|---|---|---|---|
||||BBL-QA||||
|**Dataset**|OBS.|UNOBS.|Avg.|OBS.|UNOBS.|Avg.|
|**Code Desc.**|||||||
|- Baseline|75.0_±_3.1|69.3_±_11.8|72.2_±_7.5|71.9_±_3.1|65.6_±_14.7|68.8_±_8.9|
|- Aligned|**76.4**_±_1.8|**74.9**_±_1.7|**75.7**_±_7.5|**72.8**_±_3.5|**71.7**_±_17.3|**72.2**_±_8.9|
|**Hindu Know.**|||||||
|- Baseline|52.7_±_2.4|45.5_±_6.2|49.1_±_4.3|52.5_±_2.3|49.0_±_5.3|50.7_±_3.8|
|- Aligned|**54.0**_±_2.6|**49.6**_±_1.2|**51.8**_±_4.3|**53.8**_±_0.2|**55.9**_±_1.5|**54.8**_±_3.8|
|**BBQ Lite**|||||||
|- Baseline|**66.7**_±_1.9|62.9_±_8.0|64.8_±_5.0|31.8_±_1.3|33.2_±_0.6|32.5_±_0.9|
|- Aligned|66.0_±_2.3|**69.0**_±_2.2|**67.5**_±_5.0|**32.7**_±_2.0|**36.3**_±_2.2|**34.5**_±_0.9|
|**Conc. Com.**|||||||
|- Baseline|75.7_±_1.5|75.0_±_2.1|75.3_±_1.8|56.3_±_5.5|53.1_±_6.2|54.7_±_5.9|
|- Aligned|**81.4**_±_1.1|**79.0**_±_1.1|**80.2**_±_1.8|**56.4**_±_5.7|**56.1**_±_6.9|**56.3**_±_5.9|
|**Logical Ded.**|||||||
|- Baseline|52.6_±_0.3|52.4_±_0.7|52.5_±_0.5|25.7_±_0.8|25.4_±_1.4|25.5_±_1.1|
|- Aligned|**52.6**_±_0.7|**52.8**_±_0.6|**52.7**_±_0.5|**25.9**_±_0.5|**28.4**_±_1.2|**27.2**_±_1.1|
|**Novel Conc.**|||||||
|- Baseline|**31.9**_±_2.6|**32.5**_±_4.2|**32.2**_±_3.4|36.9_±_4.1|35.0_±_2.6|35.9_±_3.3|
|- Aligned|28.7_±_2.0|31.6_±_4.3|30.2_±_3.4|**37.5**_±_3.9|**39.2**_±_2.4|**38.3**_±_3.3|
|**Logic Grid**|||||||
|- Baseline|42.0_±_0.9|40.0_±_4.4|41.0_±_2.7|25.7_±_0.9|25.6_±_1.5|25.7_±_1.2|
|- Aligned|**45.1**_±_3.6|**41.1**_±_4.1|**43.1**_±_2.7|**26.1**_±_1.0|**30.4**_±_4.2|**28.2**_±_1.2|
|**Known Unk.**|||||||
|- Baseline|78.3_±_1.5|**67.4**_±_6.3|**72.8**_±_3.9|72.9_±_4.7|61.3_±_9.3|67.1_±_7.0|
|- Aligned|**79.3**_±_1.6|65.5_±_10.0|72.4_±_3.9|**74.0**_±_4.3|**67.4**_±_7.1|**70.7**_±_7.0|
|**Overall**∆|1.1_±_2_._5|2.3_±_2_._8|1.7_±_2_._1|0.7_±_0_._4|4.6_±_1_._5|2.7_±_0_._9|



Table 32: Granular results corresponding to the BBL-QA subset reported in Table 7. Baseline refers to the performance of the original instruction-tuned model. We observe the significant improvement on following unobserved instructions consistently. 

39 

