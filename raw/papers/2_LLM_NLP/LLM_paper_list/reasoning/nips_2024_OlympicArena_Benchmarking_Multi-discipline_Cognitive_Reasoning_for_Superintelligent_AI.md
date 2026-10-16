# **OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI** 

**Zhen Huang**[3] **, Zengzhi Wang**[1] _[,]_[3] **, Shijie Xia**[1] _[,]_[3] **, Xuefeng Li**[1] _[,]_[3] **, Haoyang Zou**[3] **, Ruijie Xu**[1] _[,]_[3] **, Run-Ze Fan**[1] _[,]_[3] **, Lyumanshan Ye**[1] _[,]_[3] **, Ethan Chern**[1] _[,]_[3] **, Yixin Ye**[1] _[,]_[3] **, Yikai Zhang**[1] _[,]_[3] **Yuqing Yang**[3] **, Ting Wu**[3] **, Binjie Wang**[3] **, Shichao Sun**[3] **, Yang Xiao**[3] **, Yiyuan Li**[3] **, Fan Zhou**[1] _[,]_[3] **Steffi Chern**[3] **, Yiwei Qin**[3] **, Yan Ma**[3] **, Jiadi Su**[3] **, Yixiu Liu**[1] _[,]_[3] **, Yuxiang Zheng**[1] _[,]_[3] **Shaoting Zhang**[2] **, Dahua Lin** _[∗]_[2] _[∗]_ **, Yu Qiao**[2] _[∗]_ **, Pengfei Liu**[1] _[,]_[2] _[,]_[3] _[∗]_ 

1Shanghai Jiao Tong University, 2Shanghai Artificial Intelligence Laboratory, 

3Generative AI Research Lab (GAIR) `gair.olympicarena@gmail.com https://gair-nlp.github.io/OlympicArena/` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0001-05.png)


Figure 1: AI participates in the Olympics from the Gaokao [57] venue. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0001-07.png)


Figure 2: The overview of our _OlympicArena_ benchmark. 

> _∗_ Corresponding authors 

Preprint. Under review. 

## **Abstract** 

The evolution of Artificial Intelligence (AI) has been significantly accelerated by advancements in Large Language Models (LLMs) and Large Multimodal Models (LMMs), gradually showcasing potential _cognitive reasoning_ abilities in problemsolving and scientific discovery (i.e., AI4Science) once exclusive to human intellect. To comprehensively evaluate current models’ performance in cognitive reasoning abilities, we introduce _OlympicArena_ , which includes 11,163 bilingual problems across both text-only and interleaved text-image modalities. These challenges encompass a wide range of disciplines spanning seven fields and 62 international Olympic competitions, rigorously examined for data leakage. We argue that the challenges in Olympic competition problems are ideal for evaluating AI’s cognitive reasoning due to their complexity and interdisciplinary nature, which are essential for tackling complex scientific challenges and facilitating discoveries. Beyond evaluating performance across various disciplines using answer-only criteria, we conduct detailed experiments and analyses from multiple perspectives. We delve into the models’ cognitive reasoning abilities, their performance across different modalities, and their outcomes in _process-level_ evaluations, which are vital for tasks requiring complex reasoning with lengthy solutions. Our extensive evaluations reveal that even advanced models like GPT-4o only achieve a 39.97% overall accuracy (28.67% for mathematics and 29.71% for physics), illustrating current AI limitations in complex reasoning and multimodal integration. Through the _OlympicArena_ , we aim to advance AI towards superintelligence, equipping it to address more complex challenges in science and beyond. We also provide a comprehensive set of resources to support AI research, including a benchmark dataset, an open-source annotation platform, a detailed evaluation tool, and a leaderboard with automatic submission features.[2] 

## **1 Introduction** 

The landscape of Artificial Intelligence (AI) has undergone a transformative evolution with advances in technologies like Large Language Models [2, 3] and Large Multimodal Models (LMMs) [31]. These models represent significant milestones on the path to Artificial General Intelligence (AGI) [47, 15], demonstrating remarkable _cognitive reasoning_ abilities, which represent drawing meaningful conclusions from incomplete and inconsistent knowledge to solve problems in complex scenarios [16, 34]. They adeptly handle tasks ranging from simple grade school math problems [13, 56, 59, 64] to complex challenges like those presented at the International Mathematical Olympiad (IMO) [46, 42]. Furthermore, they are progressively being applied to intricate real-world scenarios, such as using AI agents for software development [37], collaborating on complex decision-making processes [11] and even boosting the field of scientific research (i.e., AI4Science) [50]. 

These applications highlight AI’s growing proficiency in cognitive reasoning, a crucial element in the pursuit of AGI and, potentially, superintelligence [35]. Therefore, how to benchmark these abilities has sparked extensive research. Existing benchmarks [18, 22, 26, 63, 44, 62] utilize multidisciplinary exam problems to assess the problem-solving skills of LLMs, but these problems are predominantly knowledge-intensive which has become relatively easy for current LLMs. Also, these benchmarks primarily focus on text-only modalities. Although some benchmarks begin to target college-level problems [52, 40] and incorporate multimodal assessments [58, 60, 61], they still predominantly focus on knowledge-intensive tasks or simple concept applications (shown in Table 1). Concurrent to our work, He et al. [17] introduces an Olympic-level benchmark yet it is limited to only mathematics and physics. Furthermore, all the above benchmarks lack a systematic and fine-grained evaluation of various cognitive reasoning abilities. For example, they mostly do the evaluation only based on answers, neglecting potential errors in the reasoning process. This underscores the need for more comprehensive evaluations that not only cover a broader range of disciplines but also focus on higher levels of cognitive reasoning as well as fine-grained evaluation. 

In this paper, we introduce _OlympicArena_ , a comprehensive, highly-challenging, and rigorously curated benchmark featuring a detailed, fine-grained evaluation mechanism designed to assess 

> 2 `https://github.com/GAIR-NLP/OlympicArena` 

2 

advanced AI capabilities across a broad spectrum of Olympic-level challenges (as illustrated in Figure 2). We extensively select, collect, and process problems from seven disciplines—mathematics, physics, chemistry, biology, geography, astronomy, and computer science—encompassing 62 different Olympic-level competitions. This extensive collection has culminated in a benchmark comprising 11,163 problems, categorized into 13 types of answers (e.g., expression, interval). Importantly, _OlympicArena_ enhances its evaluation framework by incorporating _process-level evaluations_ that scrutinize the step-by-step reasoning processes of AI models. This approach is critical for understanding the depth of cognitive reasoning beyond correct answers [29, 53], allowing us to identify and rectify gaps in AI reasoning pathways and ensuring more robust AI capabilities. The benchmark is bilingual, featuring both English and Chinese, to enhance its accessibility and global applicability. Additionally, it supports two modalities: text-only and interleaved text and images, catering to the evolving complexity of tasks that modern AI systems must handle. We also perform data leakage detection experiments [54] on some mainstream models to validate our benchmark’s effectiveness. 

We conduct a series of experiments across existing top-performing LMMs, encompassing both proprietary models (e.g., GPT-4o [36]) and open-source models (e.g., LLaVa-NeXT [31]). Additionally, we evaluate various types of LLMs (e.g., GPT-3.5) in two settings: text-only and image-caption and conduct comprehensive evaluations from both the answer-level and process-level perspectives. For answer-level evaluations, we combine rule-based and model-based (GPT-4V[3] in this paper) methods to cover a more diverse range of answer types. For process-level evaluations, we score each reasoning step of the model output, which we consider quite critical in reasoning scenarios. Additionally, we perform fine-grained evaluations and analyses on different types of cognitive reasoning, from both logical and visual perspectives to better interpret the current capabilities of AI. 

Our observations from the _OlympicArena_ benchmark are summarized as follows: (1) Even the most advanced model, GPT-4o, achieves only a 39.97% overall accuracy, while other open-source models struggle to reach a 20% overall accuracy, underscoring current models’ limitations in handling complex, multidisciplinary problems that require advanced cognitive reasoning—key aspects of scientific discovery. (2) Through more fine-grained analysis § 4.4, we find that LMMs are particularly weak in handling complex, decompositional reasoning problems and exhibit poor spatial and geometric perception visual abilities, as well as difficulties in understanding abstract symbols. (3) Additionally, we discover that current LMMs seem to struggle significantly in leveraging interleaved visual information for complex cognitive reasoning problems. Various LMMs fail to show notable enhancements compared to their text-only counterparts. (4) The process-level evaluation also indicates that most models can correctly execute some reasoning steps in spite of providing incorrect answers, demonstrating the models’ significant potential. (5) Through data leakage detection, we find that instances of data leakage in our benchmark are exceedingly rare. Even on the infrequent occasions when leakage does occur, the corresponding models do not consistently solve these problems correctly. This suggests the need for more advanced training strategies to enhance cognitive reasoning capabilities. These observations highlight the immense value of the _OlympicArena_ benchmark in advancing our understanding of AI’s capabilities and limitations. 

## **2 Related Work** 

**Benchmark AI Intelligence** How to benchmark AI intelligence has always been a challenging problem. Initially, the Turing Test [47] provided a conceptual framework for evaluating AI Intelligence. However, limitations in past AI technology lead researchers to focus on specialized domains. In computer vision, benchmarks like MNIST [25] and ImageNet [14] catalyze progress, while in natural language processing, GLUE [49] and XTREME [21] set the standard for evaluating linguistic capabilities across tasks and languages. The success of pretrained language models [38, 23] particularly recent LLMs emphasizes the evaluation of foundational knowledge and innate abilities as shown in Figure 2. This leads to the creation of benchmarks such as MMLU [18], AGIEval[63], C-Eval [22], and CMMLU [26], which pushed the limits of language models with multidisciplinary, multilingual, and knowledge-intensive tasks. However, the rapid progress of LLMs has rendered these benchmarks insufficient to fully assess the models’ growing capabilities. 

> 3At the time of doing most part of this work, GPT-4o has not been released yet, so GPT-4V is mainly used for annotating, evaluation, and case study. 

3 

Table 1: Comparison of various benchmarks. “Subjects”: Math, Physics, Chemistry, Biology, Geography, Astronomy, Computer Science. “Multimodal” indicates whether the benchmark contains visual information. “Language”: “EN” for English and “ZH” for Chinese. “Size” represents the number of test problems. “#Answer” shows the number of answer types (from Appendix A.2). “Eval.” details evaluation methods: rule-based, model-based, answer-level, process-level. “Leak Det.” indicates if data leakage detection is conducted. “Difficulty” shows problem proportions at difficulty levels: Knowledge Recall, Concept Application, Cognitive Reasoning. "#Logic." indicates the average logical reasoning abilities per question, and "#Visual." indicates the average visual reasoning abilities per multimodal question. Cognitive reasoning abilities are detailed in § 3.3. 

|Benchmark|Subjects|Multimodal|Language|Size|#Answer|Eval.|Leak Det.|||Diffculty|Diffculty|#Logic.|#Visual.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|SciBench<br>CMMLU<br>MMLU<br>C-Eval<br>MMMU<br>SciEval<br>AGIEval<br>GPQA<br>JEEBench<br>OlympiadBench||✓<br>_×_<br>_×_<br>_×_<br>✓<br>_×_<br>_×_<br>_×_<br>_×_<br>✓|EN<br>ZH<br>EN<br>ZH<br>EN<br>EN<br>EN & ZH<br>EN<br>EN<br>EN & ZH|789<br>1594<br>2554<br>3362<br>3007<br>15901<br>3300<br>448<br>515<br>8952|1<br>1<br>1<br>1<br>2<br>4<br>2<br>1<br>3<br>7|/<br>/<br>/<br>/<br>/<br>/<br>/<br>/<br>/<br>/|_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_|||||0.39<br>0.36<br>0.44<br>0.6<br>0.25<br>1.12<br>1.07<br>2.24<br>2.41<br>2.26|2.35<br>-<br>-<br>-<br>2.75<br>-<br>-<br>-<br>-<br>2.96|
|**OlympicArena**||✓|EN & ZH|11163|**13**|/|✓|||||**2.73**|**3.15**|



**Cognitive Reasoning** is crucial as it allows AI systems to apply prior knowledge and logical principles to complex tasks in a more human-like manner, ensuring better robustness and generalization in real-world applications [43]. Thus, more attention is paid to more intricate reasoning tasks, benchmarks like GSM8K [13] focused on grade-school mathematical reasoning problems, while MATH [20] introduced high-school level mathematical competition tasks. Furthermore, benchmarks such as JEEBench [4], SciBench [52], GPQA [40] and MMMU [58] have expanded the scope by incorporating multidisciplinary university-level subjects and even multimodal tasks. To further challenge AI systems, researchers have turned to problems from some of the most difficult competitions, specifically International Olympiads [17, 46, 30] and algorithmic challenges [28, 19, 41]. Nevertheless, there is currently no Olympic-level, multidisciplinary benchmark that comprehensively evaluates comprehensive problem-solving abilities to fully test all-rounded AI’s cognitive ability. Table 1 presents a comparison of several related scientific benchmarks. 

**Rigorous Evaluation for Reasoning** While curating comprehensive and appropriate data is crucial in benchmarks, adopting rigorous evaluation methodologies is equally important. Most existing benchmarks, as mentioned above, primarily focus on answer-level evaluation (i.e., only comparing the model’s output with the standard answer). Recently, some works have started to focus on the models’ intermediate reasoning steps. Some of them [48, 29, 51] explore using process supervision to train better reward models. Lanham et al. [24] delves into the faithfulness of the chain-of-thought reasoning process, while Xia et al. [53] trains models specifically designed to evaluate the validity and redundancy of reasoning steps for mathematical problems. However, in the evaluation methodologies of existing benchmarks as listed in Table 1, few of them incorporate process-level evaluation. This insufficient evaluation often neglects the reliability and faithfulness of AI models, especially in complex cognitive reasoning scenarios requiring lengthy solutions. In this work, the introduced _OlympicArena_ is equipped with a more fine-grained evaluation methodology (i.e., process-level evaluation), allowing developers to better understand the true reasoning behaviors of models. 

## **3 The OlympicArena Benchmark** 

## **3.1 Overview** 

We introduce the _OlympicArena_ , an Olympic-level, multidisciplinary benchmark designed to rigorously assess the cognitive reasoning abilities of LLMs and LMMs. Our benchmark features a combination of text-only and interleaved text-image modalities, presented bilingually to promote accessibility and inclusivity. It spans seven core disciplines: **mathematics** , **physics** , **chemistry** , **biology** , **geography** , **astronomy** , and **computer science** , encompassing a total of 34 specialized branches (details are in Appendix A.1) which represent fundamental scientific fields. The benchmark 

4 

includes a comprehensive set of 11,163 problems from 62 distinct Olympic competitions, structured with 13 answer types (shown in Appendix A.2) from objective types (e.g., multiple choice and fill-inthe-blanks) to subjective types (e.g., short answers and programming tasks), which distinguishes it from many other benchmarks that primarily focus on objective problems. Detailed statistics of _OlympicArena_ are described in Table 2. Table 2: Benchmark Statistics Also, to identify potential data leakage, we conduct specialized data leakage detection experiments on several models. **Statistic Number** 

||**Statistic**|**Number**|
|---|---|---|
||Total Problems<br>Total Competitions<br>Total Subjects/Subfelds<br>Total Answer Types<br>Problems with Solutions|11163<br>62<br>7/34<br>13<br>7904|
||Language (EN: ZH)|7054: 4109|
||Total Images<br>Problems with Images<br>Image Types|7571<br>4960<br>5|
||Cognitive Complexity Levels<br>Logical Reasoning Abilities<br>Visual Reasoning Abilities<br>Average Problem Tokens|3<br>8<br>5<br>244.8|
||Average Solution Tokens|417.1|



Furthermore, in pursuit of a granular analysis of model performance, we categorize cognitive reasoning into 8 types of logical reasoning abilities and 5 types of visual reasoning abilities. This comprehensive categorization aids in the detailed evaluation of the diverse and complex reasoning skills that both LLMs and LMMs can exhibit. Additionally, we specifically investigate all multimodal problems to compare the performance of LMMs against their text-based counterparts, aiming to better assess LMMs’ capabilities in handling visual information. Finally, we evaluate the correctness and efficiency of the reasoning process, not just limited to an answer-based assessment. 

## **3.2 Data Collection** 

To ensure comprehensive coverage of Olympic-level problems across various disciplines, we begin by collecting URLs of various competitions where problems are publicly available for download in PDF format. Then, we utilize the Mathpix[4] tool to convert these PDF documents into markdown format, making them compatible with input requirements for models. Specifically, for the programming problems of Computer Science, we additionally collect corresponding test cases. We strictly adhere to copyright and licensing considerations, ensuring compliance with all relevant regulations. 

## **3.3 Data Annotation** 

**Problem Extraction and Annotation.** To extract individual problems from the markdown format of the test papers, we employ about 30 students with background in science and engineering. We have developed a user interface for annotating multimodality data, which has been released.[5] To facilitate further research and the process-level evaluation of models, we annotate meta-information like solutions if provided. To ensure data quality, we implement a multi-step validation process after the initial annotation is completed. More details can be seen in Appendix B.1. After collecting all the problems, we perform deduplication within each competition based on model embeddings to remove repeated problems that may appear in multiple test papers from the same year. To further demonstrate that our benchmark emphasizes cognitive reasoning more than most other benchmarks, we categorize the difficulty of the problems into three levels and make comparison with other related benchmarks. Specifically, we classify all problems into: _knowledge recall_ , _concept application_ and _cognitive reasoning_ . We utilize GPT-4V as the annotator for categorizing different difficulty levels[6] (detailed definitions and specific prompts can be found in Appendix B.2).[7] 

**Annotation of Cognitive Reasoning Abilities.** To facilitate better fine-grained analysis, we categorize cognitive reasoning abilities from both logical and visual perspectives [16, 43]. The logical reasoning abilities encompass _Deductive Reasoning (DED)_ , _Inductive Reasoning (IND)_ , _Abductive Reasoning (ABD)_ , _Analogical Reasoning (ANA)_ , _Cause-and-Effect Reasoning (CAE)_ , _Critical Thinking (CT)_ , _Decompositional Reasoning (DEC)_ , and _Quantitative Reasoning (QUA)_ . Meanwhile, the visual reasoning abilities include _Pattern Recognition (PR)_ , _Spatial Reasoning (SPA)_ , _Diagrammatic Reasoning (DIA)_ , _Symbol Interpretation (SYB)_ , and _Comparative Visualization (COM)_ . We also utilize GPT-4V as the annotator for categorizing different cognitive abilities (detailed definitions and specific prompts can be found in Appendix B.3).[7] With these annotations, we can conduct a more fine-grained analysis of the current cognitive reasoning abilities of AI. 

> 4 `https://mathpix.com/` 

> 5 `https://github.com/GAIR-NLP/OlympicArena/tree/main/annotation` 

> 6We annotate the validation sets to highlight their characteristics and save costs. 

> 7All annotations using GPT-4V are manually verified for reliability. 

5 

## **3.4 Data Splitting** 

Our benchmark includes 11,163 problems, with 548 designated for model-based evaluation as _OlympicArena-ot_ . We sample 638 problems across subjects to create _OlympicArena-val_ for hyperparameter tuning or small-scale testing. _OlympicArena-val_ problems have step-by-step solutions, supporting research like process-level evaluation. The remaining problems form _OlympicArena-test_ , the official test set with unreleased answers for formal testing. The results in this paper are based on the entire benchmark dataset, including _OlympicArena-ot_ , _OlympicArena-val_ , and _OlympicArena-test_ . 

## **4 Experiments** 

## **4.1 Experimental Setup** 

To comprehensively evaluate the capabilities of LLMs and LMMs (selected models are listed in Appendix C.2) across different modalities, we design our experiments to include three distinct settings: multimodal, image-caption, and text-only. In the multimodal setting, we assess the ability of LMMs to leverage visual information by interleaving text and images, simulating real-world scenarios. For models unable to handle interleaved inputs, we concatenate multiple images into a single input. For LMMs requiring necessary image inputs, their text-based counterparts handle text-only problems. In the image-caption setting, we explore whether textual descriptions of images enhance the problem-solving capabilities of LLMs. Using InternVL-Chat-V1.5[8] [12], we generate captions for all images based on prompts detailed in Appendix C.1. These captions replace the original image inputs. In the text-only setting, we evaluate the performance of LLMs without any visual information, serving as a baseline to compare against the multimodal and image-caption settings. All experiments use zero-shot prompts, tailored to each answer type and specifying output formats to facilitate answer extraction and rule-based matching. It also minimizes biases typically associated with few-shot learning [32, 33]. Detailed prompt designs are provided in Appendix C.3. 

## **4.2 Evaluation** 

**Answer-level Evaluation** We combine rule-based and model-based methods to cover a diverse range of problems. For problems with fixed answers, we extract the final answer and perform rulebased matching according to the answer type. For code generation tasks, we use the unbiased pass@k metric [10] to test all test cases. For problems with answer types categorized as "others" which are difficult to be evaluated using rule-based matching (e.g., chemical equation writing problems), we employ GPT-4V as an evaluator to assess the responses. To ensure the reliability of GPT-4V as an evaluator, we manually sample and check the correctness. See Appendix C.5 for more details. 

**Process-level Evaluation** To further investigate the correctness of the reasoning steps, ensuring a rigorous assessment of the cognitive abilities of models, we conduct the process-level evaluation. We first sample 96 problems with reference solutions from _OlympicArena_ . We employ GPT-4 to convert both the references (i.e., gold solutions) and the model-generated solutions into a structured step-by-step format. We then provide these solutions to GPT-4V and score each step for its correctness on a scale ranging from 0 to 1.[9] The experimental details can be seen in Appendix C.6. To validate the consistency with human judgment, we obtain some samples for human annotations. The results indicate that our model-based evaluation method is highly accurate, with an 83% inter-annotator agreement. 

## **4.3 Main Results** 

Table 3 presents the evaluation results of various LMMs and LLMs on _OlympicArena_ . We obtain the following observations: (1) Even the most advanced large model, GPT-4o, achieves only a 39.97% overall accuracy, while other open-source models struggle to reach a 20% overall accuracy. This stark contrast highlights the significant difficulty and rigor of our benchmark, demonstrating its effectiveness in pushing the boundaries of current AI capabilities. (2) Furthermore, compared to subjects like biology and geography, we observe that mathematics and physics remain the two most 

> 8We use InternVL-Chat-V1.5 for its high performance and cost-effective captioning. 

> 9We leave more research on open-source model-based evaluation for future work. 

6 

Table 3: Experimental results on _OlympicArena_ , expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. We use the pass@k metric (Equation 1) for CS problems. When calculating the overall accuracy, for code generation problems, if any generated code for a problem passes all test cases, the problem is considered correct. 

|Model|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**CS**<br>**Overall**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Pass@1<br>Accuracy|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**CS**<br>**Overall**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Pass@1<br>Accuracy|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**CS**<br>**Overall**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Pass@1<br>Accuracy|
|---|---|---|---|
|||LLMs||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat|1.58<br>3.06<br>5.88<br>9.65|3.74<br>7.01<br>7.31<br>4.53<br>5.48<br>9.77<br>23.53<br>32.67<br>35.03<br>18.15<br>9.48<br>18.36<br>31.90<br>32.14<br>16.03<br>14.54<br>29.84<br>38.58<br>40.69<br>28.05|0<br>4.31<br>0.17<br>17.31<br>0.60<br>16.62<br>0.51<br>23.69|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o|7.27<br>7.76<br>19.46<br>28.33|10.92<br>23.03<br>31.19<br>31.13<br>16.93<br>17.24<br>29.46<br>38.25<br>40.94<br>24.04<br>24.77<br>42.52<br>46.47<br>44.97<br>33.44<br>29.54<br>46.24<br>49.42<br>48.36<br>43.25|3.85<br>18.27<br>1.62<br>23.02<br>7.78<br>32.37<br>8.46<br>38.17|
|||Image caption + LLMs||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat|1.76<br>3.01<br>5.94<br>9.56|3.56<br>6.75<br>7.83<br>7.17<br>6.87<br>9.94<br>21.45<br>31.26<br>34.78<br>17.33<br>10.40<br>20.25<br>31.00<br>32.52<br>16.93<br>14.31<br>29.84<br>38.51<br>40.75<br>27.2|0<br>4.89<br>0.17<br>16.72<br>0.73<br>17.07<br>0.60<br>23.43|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o|7.16<br>7.52<br>19.46<br>28.27|14.48<br>23.97<br>30.94<br>33.52<br>18.56<br>18.10<br>29.84<br>38.77<br>41.14<br>22.65<br>26.21<br>41.58<br>45.89<br>48.18<br>35<br>29.71<br>45.87<br>51.16<br>49.12<br>43.17|4.70<br>18.83<br>2.39<br>23.10<br>7.63<br>33.00<br>**9.57**<br>38.50|
|||LMMs||
|Qwen-VL-Chat<br>Yi-VL-34B<br>InternVL-Chat-V1.5<br>LLaVA-NeXT-34B|1.73<br>2.94<br>6.03<br>3.03|4.25<br>8.64<br>12.13<br>13.77<br>7.85<br>9.94<br>19.81<br>27.73<br>25.16<br>16.60<br>9.25<br>19.12<br>30.39<br>32.96<br>15.94<br>10.06<br>21.45<br>33.18<br>36.92<br>18.15|0<br>6.90<br>0<br>14.49<br>0.38<br>16.63<br>0.18<br>17.38|
|Qwen-VL-Max<br>Gemini Pro Vision<br>Claude3 Sonnet<br>GPT-4V<br>GPT-4o|6.93<br>6.28<br>7.52<br>19.27<br>**28.67**|12.36<br>23.79<br>36<br>40.19<br>23.39<br>12.47<br>28.14<br>37.48<br>37.42<br>20.20<br>18.16<br>29.27<br>38.96<br>40.13<br>25.02<br>24.83<br>41.45<br>46.79<br>49.62<br>32.46<br>**29.71**<br>**46.69**<br>**52.18**<br>**56.23**<br>**43.91**|0.77<br>20.65<br>1.45<br>20.97<br>1.45<br>23.13<br>7.00<br>32.76<br>9.00<br>**39.97**|



challenging disciplines, likely due to their reliance on complex reasoning abilities. (3) Computer programming competitions also prove to be highly difficult, with some open-source models failing to solve any of them, indicating current models’ poor abilities to design efficient algorithms to solve complex problems. 

## **4.4 Fine-grained Analysis** 

To achieve a more fine-grained analysis of the experimental results, we conduct further evaluations based on different modalities and reasoning abilities. Additionally, we also conduct an analysis of the process-level evaluation. Key findings are as follows: 

**Models exhibit varied performance across different logical and visual reasoning abilities.** As shown in Figure 3, almost all models demonstrate similar performance trends across different logical reasoning abilities. They tend to excel in Abductive Reasoning and Cause-and-Effect Reasoning, doing well in identifying causal relationships from the provided information. Conversely, models perform poorly in Inductive Reasoning and Decompositional Reasoning. This is due to the diverse and unconventional nature of Olympic-level problems, which require the ability to break down complex problems into smaller sub-problems. In terms of visual reasoning abilities, models tend to be better at Pattern Recognition and Comparative Visualization. However, they struggle with tasks involving spatial and geometric reasoning as well as those need to understand abstract symbols. The completed results are presented in Appendix D.1. 

**Most LMMs are still not proficient at utilizing visual information.** As displayed in Figure 4a, only a few LMMs (such as GPT-4o and Qwen-VL-Chat) show significant improvement with image inputs compared to their text-based counterpart. Many LMMs do not exhibit enhanced performance 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Logical Reasoning Visual Reasoning<br>DED PR<br>50 40<br>QUA 40 IND<br>30<br>30<br>20<br>20<br>COM SPA<br>10 10<br>DEC A BD<br>CT ANA<br>SYB DIA<br>CAE<br>GPT-4o GPT-4V Claude3-sonnet LLaVa-NeXT-34B InternVL-Chat-V1.5<br>**----- End of picture text -----**<br>


Figure 3: Performance of various models on logical and visual reasoning abilities. Logical reasoning abilities: Deductive Reasoning (DED), Inductive Reasoning (IND), Abductive Reasoning (ABD), Analogical Reasoning (ANA), Cause-and-Effect Reasoning (CAE), Critical Thinking (CT), Decompositional Reasoning (DEC), and Quantitative Reasoning (QUA). Visual reasoning abilities: Pattern Recognition (PR), Spatial Reasoning (SPA), Diagrammatic Reasoning (DIA), Symbol Interpretation (SYB), and Comparative Visualization (COM). 

with image inputs and some even show decreased effectiveness when handling images. Possible reasons include: (1) When text and images are input together, LMMs may focus more on the text, neglecting the information in the images. This conclusion has also been found in some other works [61, 9]. (2) Some LMMs, while training their visual capabilities based on their text-based models, may lose some of their inherent language abilities (e.g., reasoning abilities), which is particularly evident in our scenarios. (3) Our problems use a complex interleaved text and image format, which some models do not support well, leading to difficulties in processing and understanding the positional information of images embedded within the text.[10] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0008-03.png)


**----- Start of picture text -----**<br>
LMMs<br>Qwen-VL-Chat Image caption + LLMsLLMs 65 17.5<br>60<br>InternVL-Chat-V1.5 15.0<br>55<br>LLaVA-NeXT-34B 50 12.5<br>45 10.0<br>Claude3-Sonnet<br>40 7.5<br>GPT-4v 35 5.0<br>30<br>GPT-4o 2.5<br>25<br>0 5 10 15 20 25 30 35 40 5 10 15 20 25 30 35 40 0.0 0.2 0.4 0.6 0.8 1.0<br>Accuracy (%) Answer-level Accuracy Location of Incorrect Steps<br>(a) (b) (c)<br>Percentage<br>Process-level Score<br>**----- End of picture text -----**<br>


Figure 4: (a) Comparison of different LMMs and their corresponding LLMs across three different experimental settings. For details on the corresponding LLMs for each LMM, refer to the Appendix C.2. (b) The correlation between answer-level and process-level scores of all the models over all the sampled problems. (c) Distribution of the locations of incorrect steps, represented as the proportion of steps from left to right in the entire process, over all the sampled problems. 

**Analysis of process-level evaluation results** Through process-level evaluation (complete results are in Table 14), we discover following insights: (1) There is generally a high consistency between process-level evaluation and answer-level evaluation. When a model produces a correct answer, the quality of the reasoning process tends to be higher most of the time (see Figure 4b). (2) The accuracy at the process-level is often higher than at the answer-level. This indicates that even for very 

> 10We exclude Yi-VL-34B as it doesn’t support multiple image inputs, which may cause an unfair comparison. 

8 

complex problems, the model can correctly perform some of the intermediate steps. Therefore, the model likely has significant untapped potential for cognitive reasoning, which opens new avenues for researchers to explore. We also find that in a few disciplines, some models that perform well at the answer level fall behind at the process level. We speculate that this is because models sometimes tend to overlook the reasonableness of intermediate steps when generating answers, even though these steps may not be crucial to the final result. (3) Additionally, we conduct a statistical analysis of the location distribution of error steps (see Figure 4c). We identify that a higher proportion of errors occur in the later stages. This suggests that models are more prone to making mistakes as reasoning accumulates, indicating a need for improvement in handling long chains of logical deductions. 

**Error analysis** To further concretize models’ performance, we sample incorrect responses from GPT-4V (16 problems per subject, with 8 text-only and 8 multimodal) and have human evaluators analyze and annotate the reasons for these errors. As depicted in Figure 5, reasoning errors (both logical and visual) constitute the largest category, indicating that our benchmark effectively highlights the current models’ deficiencies in cognitive reasoning abilities. Additionally, a significant portion of errors stem from knowledge deficits, suggesting that current models still lack expert-level domain knowledge and the ability to leverage this knowledge to assist in reasoning. Another category of errors arise from understanding biases, which can be attributed to the models’ misinterpretation of context and difficulties in integrating complex language structures and multimodal information. More relevant cases are shown in Appendix F.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0009-02.png)


Figure 5: Error types distribution for sampled error problems from GPT-4V. 

## **4.5 Efforts on Data Leakage Detection** 

Given the increasing scale of pre-training corpora, it is crucial to detect potential benchmark leakage. The opacity of pre-training often makes this task challenging. To this end, we employ a recently proposed instance-level leakage detection metric, _N-gram Prediction Accuracy_ [54]. This metric uniformly samples several starting points for each instance, predicts the next n-gram for each starting point, and checks whether all predicted n-grams are correct, indicating that the model has potentially encountered this instance. We apply this metric to all available base or text-only chat models of the evaluated models. As shown in Figure 6, it is surprising yet reasonable that some base models or text-only chat models behind these evaluated models have potentially encountered a few benchmark instances, although the number is negligible compared to the complete benchmark. For instance, the base model of Qwen1.5-32B-Chat has 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0009-06.png)


**----- Start of picture text -----**<br>
43<br>40<br>30<br>20 19 19<br>12 11<br>2 0 0<br>0<br># Leakage # Correct in Text-only Chat Model # Correct in MM Chat Model<br>InternLM2-20BInternLM2-20B-Chat Yi-34BNous-Hermes-2-Yi-34BYi-34B-Chat Qwen-7BQwen1.5-32BQwen1.5-32B-Chat GPT-4o<br>Number of Instances<br>**----- End of picture text -----**<br>


Figure 6: Detected number of leaked samples and the number of correct responses by corresponding text-only and multimodal chat models on these samples. 

potentially encountered 43 benchmark instances. Furthermore, this raises a natural question: _can the model correctly answer these instances?_ Interestingly, the corresponding text-only chat models and multimodal chat models can correctly answer even fewer of these instances. These results demonstrate that our benchmark has minimal leakage[11] and is sufficiently challenging, as the models cannot correctly answer most of the leaked instances. See Appendix E for more results and analysis. 

> 11We also look forward to the development of more advanced detection tools and approaches. 

9 

## **5 Conclusion** 

In this work, we introduce _OlympicArena_ , a comprehensive benchmark for evaluating the cognitive reasoning abilities of LMMs and LLMs on Olympic-level problems. Through our detailed experiments, we find that even the most powerful model at present, GPT-4o, does not perform well in applying cognitive reasoning abilities to solve complex problems. We hope that our OlympicArena benchmark serves as a valuable stepping stone for future advancements in AI for science and engineering. 

## **Acknowledgements** 

We sincerely appreciate all the laboratory members for their contributions in data annotation, project discussions, and providing valuable suggestions. Additionally, we extend our gratitude to Teacher Xiaoxia Yu from Hefei No. 168 Middle School for providing us with extensive information on various subjects. We also thank everyone who helps annotate the data for our benchmark dataset. 

## **References** 

- [1] Gpt-4v(ision) system card. 2023. URL `https://api.semanticscholar.org/CorpusID: 263218031` . 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [3] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. _Claude-3 Model Card_ , 2024. 

- [4] Daman Arora, Himanshu Gaurav Singh, et al. Have llms advanced enough? a challenging problem solving benchmark for large language models. _arXiv preprint arXiv:2305.15074_ , 2023. 

- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv preprint arXiv:2309.16609_ , 2023. 

- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_ , 2023. 

- [7] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. 

- [8] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. _arXiv preprint arXiv:2403.17297_ , 2024. 

- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? _arXiv preprint arXiv:2403.20330_ , 2024. 

- [10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- [11] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum?id=EHg5GDnyq1` . 

- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_ , 2023. 

10 

- [13] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- [14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pp. 248–255. Ieee, 2009. 

- [15] Edward A Feigenbaum, Julian Feldman, et al. _Computers and thought_ . New York McGraw-Hill, 1963. 

- [16] Ulrich Furbach, Steffen Hölldobler, Marco Ragni, Claudia Schon, and Frieder Stolzenburg. Cognitive reasoning: A personal view. _KI-Künstliche Intelligenz_ , 33:209–217, 2019. 

- [17] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. _arXiv preprint arXiv:2402.14008_ , 2024. 

- [18] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

- [19] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. _arXiv preprint arXiv:2105.09938_ , 2021. 

- [20] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- [21] Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. In _International Conference on Machine Learning_ , pp. 4411–4421. PMLR, 2020. 

- [22] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [23] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of NAACL-HLT_ , pp. 4171–4186, 2019. 

- [24] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. Measuring faithfulness in chain-of-thought reasoning. _arXiv preprint arXiv:2307.13702_ , 2023. 

- [25] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. _Proceedings of the IEEE_ , 86(11):2278–2324, 1998. 

- [26] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. _arXiv preprint arXiv:2306.09212_ , 2023. 

- [27] Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, and Jing Ma. Mmcode: Evaluating multimodal code large language models with visually rich programming problems. _arXiv preprint arXiv:2404.09486_ , 2024. 

- [28] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. _Science_ , 378(6624):1092–1097, 2022. 

11 

- [29] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- [30] Chengwu Liu, Jianhao Shen, Huajian Xin, Zhengying Liu, Ye Yuan, Haiming Wang, Wei Ju, Chuanyang Zheng, Yichun Yin, Lin Li, et al. Fimo: A challenge formal dataset for automated theorem proving. _arXiv preprint arXiv:2309.04295_ , 2023. 

- [31] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 

- [32] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. _arXiv preprint arXiv:2104.08786_ , 2021. 

- [33] Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, and Bingzhe Wu. Fairness-guided few-shot prompting for large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [34] Meredith Ringel Morris, Jascha Sohl-dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, and Shane Legg. Levels of agi: Operationalizing progress on the path to agi. _arXiv preprint arXiv:2311.02462_ , 2023. 

- [35] OpenAI. Introducing superalignment. _OpenAI Blog_ , 2023. URL `https://openai.com/ superalignment` . 

- [36] OpenAI. Hello gpt-4o. _OpenAI Blog_ , 2024. URL `https://openai.com/index/ hello-gpt-4o/` . 

- [37] Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. _arXiv preprint arXiv:2307.07924_ , 2023. 

- [38] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 

- [39] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- [40] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. _arXiv preprint arXiv:2311.12022_ , 2023. 

- [41] Quan Shi, Michael Tang, Karthik Narasimhan, and Shunyu Yao. Can language models solve olympiad programming? _arXiv preprint arXiv:2404.10952_ , 2024. 

- [42] Shiven Sinha, Ameya Prabhu, Ponnurangam Kumaraguru, Siddharth Bhat, and Matthias Bethge. Wu’s method can boost symbolic ai to rival silver medalists and alphageometry to outperform gold medalists at imo geometry. _arXiv preprint arXiv:2404.06405_ , 2024. 

- [43] Jiankai Sun, Chuanyang Zheng, Enze Xie, Zhengying Liu, Ruihang Chu, Jianing Qiu, Jiaqi Xu, Mingyu Ding, Hongyang Li, Mengzhe Geng, et al. A survey of reasoning with foundation models. _arXiv preprint arXiv:2312.11562_ , 2023. 

- [44] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 19053–19061, 2024. 

- [45] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

12 

- [46] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. _Nature_ , 625(7995):476–482, 2024. 

- [47] Alan M Turing and J Haugeland. Computing machinery and intelligence. _The Turing Test: Verbal Behavior as the Hallmark of Intelligence_ , pp. 29–56, 1950. 

- [48] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. _arXiv preprint arXiv:2211.14275_ , 2022. 

- [49] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In _Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_ , pp. 353–355, 2018. 

- [50] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. _Nature_ , 620(7972):47–60, 2023. 

- [51] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. _arXiv preprint arXiv:2312.08935_ , 2023. 

- [52] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. _arXiv preprint arXiv:2307.10635_ , 2023. 

- [53] Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. Evaluating mathematical reasoning beyond accuracy. _arXiv preprint arXiv:2404.05692_ , 2024. 

- [54] Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. _arXiv preprint arXiv:2404.18824_ , 2024. 

- [55] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. _arXiv preprint arXiv:2403.04652_ , 2024. 

- [56] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

- [57] Weizhe Yuan and Pengfei Liu. restructured pre-training. _arXiv preprint arXiv:2206.11147_ , 2022. 

- [58] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv preprint arXiv:2311.16502_ , 2023. 

- [59] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_ , 2023. 

- [60] Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, et al. Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark. _arXiv preprint arXiv:2401.11944_ , 2024. 

- [61] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? _arXiv preprint arXiv:2403.14624_ , 2024. 

- [62] Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. _arXiv preprint arXiv:2305.12474_ , 2023. 

13 

- [63] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _arXiv preprint arXiv:2304.06364_ , 2023. 

- [64] Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. _arXiv preprint arXiv:2308.07921_ , 2023. 

14 

## **A Detailed Statistics of the Benchmark** 

## **A.1 Distribution of Problems** 

Our benchmark collects data from various competitions. The detailed list can be found in Table 4. Note that a small portion of the problems are sampled from other related benchmarks which are marked in the table. The subfields covered by each competition subject are shown in Table 5. Additionally, the distribution information of our benchmark across different languages and modalities is presented in Table 6. 

## **A.2 Answer Types** 

Through extensive observation of a large number of problems and a thorough examination of multiple previous benchmarks, we have finally distilled 13 comprehensive answer types. These types are designed to cover as many problems as possible. The specific definitions for each answer type are provided in Table 7. 

## **A.3 Image Types** 

We categorize and summarize the five most common types of images in our multimodal scientific problems. The definitions of these types can be found in Table 8, and examples are provided in Figure 7. The distribution of different image types in our benchmark is shown in Figure 8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0015-07.png)


**----- Start of picture text -----**<br>
Geometric and  Statistical and Data  Natural and Environmental<br>Mathematical Diagrams Representation Images<br>Scientific and Technical  Abstract and Conceptual<br>Diagrams Visuals<br>**----- End of picture text -----**<br>


Figure 7: Examples of Image Types 

15 

Table 4: List of competitions included in OlympicArena. Competitions marked with * are partially sourced from OlympiadBench [17], and those marked with _†_ are partially sourced from MMcode [27]. 

|**Competition Name**|**Abbreviation**|**Subject**|**# Problems**|
|---|---|---|---|
|UK Senior Kangaroo|UKMT_SK|Math|20|
|Math Majors of America Tournament for High Schools|MMATHS|Math|47|
|Math Kangaroo|MK|Math|35|
|Euclid Mathematics Contest|EMC|Math|215|
|Canadian Open Mathematics Challenge|COMC|Math|26|
|Johns Hopkins Mathematics Tournament|JHMT|Math|100|
|Berkeley Math Tournament|BMT|Math|93|
|Stanford Mathematics Tournament|SMT|Math|473|
|Chinese High School Mathematics League (Pre Round)|ZH_Math_PRE|Math|546|
|Chinese High School Mathematics League (1st&2nd Round)|ZH_Math_12|Math|279|
|Duke University Math Meet|DMM|Math|107|
|The Princeton University Mathematics Competition|PUMaC|Math|296|
|Harvard-MIT Mathematics Tournament|HMMT|Math|392|
|William Lowell Putnam Mathematics Competition|Putnam|Math|136|
|International Mathematical Olympiad*|IMO|Math|79|
|Romanian Master of Mathematics*|RMM|Math|8|
|American Regions Mathematics League*|ARML|Math|374|
|Euclid Mathematics Competition*|EMC|Math|215|
|European Girls’ Mathematical Olympiad*|EGMO|Math|7|
|F=MA|FMA|Physics|122|
|Intermediate Physics Challenge (Y11)|BPhO_IPC|Physics|50|
|Senior Physics Challenge|BPhO_SPC|Physics|38|
|Australian Science Olympaids Physics|ASOP|Physics|48|
|European Physics Olympiad|EPhO|Physics|15|
|Nordic-Baltic Physics Olympiad|NBPhO|Physics|102|
|World Physics Olympics|WoPhO|Physics|38|
|Asian Physics Olympiad|APhO|Physics|126|
|International Physics Olympiad|IPhO|Physics|307|
|Canadian Association of Physicists|CAP|Physics|100|
|Physics Bowl|PB|Physics|100|
|USA Physics Olympiad|USAPhO|Physics|188|
|Chinese Physics Olympiad|CPhO|Physics|462|
|Physics Challenge (Y13)|PCY13|Physics|44|
|Chinese High School Biology Challenge|GAOKAO_Bio|Biology|652|
|International Biology Olympiad|IBO|Biology|300|
|The USA Biology Olympiad|USABO|Biology|96|
|Indian Biology Olympiad|INBO|Biology|86|
|Australian Science Olympiad Biology|ASOB|Biology|119|
|British Biology Olympiad|BBO|Biology|82|
|New Zealand Biology Olympiad|NZIBO|Biology|223|
|Chem 13 News|Chem13News|Chemistry|56|
|Avogadro|Avogadro|Chemistry|55|
|U.S. National Chemistry Olympiad (local)|USNCO (local)|Chemistry|54|
|U.S. National Chemistry Olympiad|USNCO|Chemistry|98|
|Chinese High School Chemistry Challenge|GAOKAO_Chem|Chemistry|568|
|Canadian Chemistry Olympic|CCO|Chemistry|100|
|Australian Science Olympiad Chemistry|ASOC|Chemistry|91|
|Cambridge Chemistry Challenge|C3H6|Chemistry|61|
|UK Chemistry Olympiad|UKChO|Chemistry|100|
|International Chemistry Olympiad|IChO|Chemistry|402|
|Chinese High School Geography Challenge|GAOKAO_Geo|Geography|862|
|US Earth Science Organization|USESO|Geography|301|
|Australian Science Olympiad Earth Science|ASOE|Geography|100|
|The International Geography Olympaid|IGeO|Geography|327|
|Chinese High School Astronomy Challenge|GAOKAO_Astro|Astronomy|740|
|The International Astronomy and Astrophysics Competition|IAAC|Astronomy|50|
|USA Astronomy and Astrophysics Organization|USAAAO|Astronomy|100|
|British Astronomy and Astrophysics Olympaid Challenge|BAAO_challenge|Astronomy|148|
|British Astronomy and Astrophysics Olympaid-round2|BAAO|Astronomy|185|
|USA Computing Olympiad|USACO|CS|48|
|Atcoder|Atcoder|CS|48|
|Codeforces_†_|CF|CS|138|



16 

Table 5: Subfields of each subject included in OlympicArena. 

||Table 5: Subfelds of each subject included in OlympicArena.|
|---|---|
|**Subject**|**Subfelds**|
|Math|Algebra, Geometry, Number Theory, Combinatorics|
|Physics|Mechanics, Electricity and Magnetism, Waves and Optics, Thermodynamics,|
||Modern Physics, Fluid Mechanics|
|Chemistry|General Chemistry, Organic Chemistry, Inorganic Chemistry, Analytical Chem-|
||istry, Physical Chemistry, Environmental Chemistry|
|Biology|Cell biology, Plant Anatomy and Physiology, Animal Anatomy and Physiology,|
||Ethology, Genetics and Evolution, Ecology , Biosystematics|
|Geography|Physical Geography, Human Geography, Regional Geography, Environmental|
||Geography, Geospatial Techniques|
|Astronomy|Fundamentals of Astronomy, Stellar Astronomy, Galactic and Extragalactic|
||Astronomy, Astrophysics|
|CS|Data Structures, Algorithm|



Table 6: Statistics of OlympicArena benchmark across different disciplines and modalities. 

||**Mathematics**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**CS**|
|---|---|
||EN & text<br>2215<br>632<br>782<br>352<br>211<br>219<br>90<br>EN & multi-modal<br>193<br>646<br>235<br>554<br>517<br>264<br>144<br>ZH & text<br>780<br>164<br>124<br>312<br>58<br>264<br>0<br>ZH & multi-modal<br>45<br>298<br>444<br>340<br>804<br>476<br>0|
||Total EN<br>2408<br>1278<br>1017<br>906<br>728<br>483<br>234<br>Total ZH<br>825<br>462<br>568<br>652<br>862<br>740<br>0<br>Total text<br>2995<br>796<br>906<br>664<br>269<br>483<br>90<br>Total multi-modal<br>238<br>944<br>679<br>894<br>1321<br>740<br>144|
||Grand Total<br>3233<br>1740<br>1585<br>1558<br>1590<br>1223<br>234|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0017-04.png)


**----- Start of picture text -----**<br>
39.16% Image Types<br>(2965) Geometric and Mathematical Diagrams<br>Statistical and Data Representation<br>Natural and Environmental Images<br>9.19% Scientific and Technical Diagrams<br>(696) Abstract and Conceptual Visuals<br>3.17%<br>(240)<br>17.12%<br>(1296)<br>31.36%<br>(2374)<br>**----- End of picture text -----**<br>


Figure 8: Distribution of Image Types 

17 

Table 7: Answer Types and Definitions 

|**Answer Type**|**Defnition**|
|---|---|
|Single Choice (SC)|Problems with only one correct option (e.g., one out of four, one<br>out of fve, etc.).|
|Multiple-choice (MC)|Problems with multiple correct options (e.g., two out of four, two<br>out of fve, two out of six, etc.).|
|True/False (TF)|Problems where the answer is either True or False.|
|Numerical Value (NV)|Problems where the answer is a numerical value, including special|
||values like_π_,_e_,<br>_√_<br>7,log29, etc., represented in LaTeX.|
|Set (SET)|Problems where the answer is a set, such as {1, 2, 3}.|
|Interval (IN)|Problems where the answer is a range of values, represented as|
||an interval in LaTeX.|
|Expression (EX)|Problems requiring an expression containing variables, repre-|
||sented in LaTeX.|
|Equation (EQ)|Problems requiring an equation containing variables, represented|
||in LaTeX.|
|Tuple (TUP)|Problems requiring a tuple, usually representing a pair of numbers,|
||such as (x, y).|
|Multi-part Value (MPV)|Problems requiring multiple quantities to be determined within a|
||single sub-problem, such as solving both velocity and time in a|
||physics problem.|
|Multiple Answers (MA)|Problems with multiple solutions for a single sub-problem, such<br>as a math fll-in-the-blank problem with answers 1 or -2.|
|Code Generation (CODE)|Problems where the answer is a piece of code, requiring the|
||generation of functional code snippets or complete programs to|
||solve the given task.|
|Others (OT)|Problems that do not ft into the above categories, such as writing|
||chemical equations or explaining reasons, which require human|
||expert evaluation.|



Table 8: Definitions and examples of five image types in our multi-modal scientific problems. 

|**Image Type**|**Defnition**|
|---|---|
|Geometric and Mathematical Diagrams|Includes diagrams representing mathematical concepts, such as<br>2D and 3D shapes, mathematical notations, functionplots.|
||Visualizations for statistical or data information, including|
|Statistical and Data Representation|multivariate plots, tables, charts (histograms, bar charts, line|
||plots), and infographics.|
||Images of natural scenes or phenomena, including environmental|
|Natural and Environmental Images|studies visualizations, geological and geographical maps, and|
||satellite images.|
||Diagrams used in science, such as cell structures and genetic|
|Scientifc and Technical Diagrams|diagrams in Biology, molecular structures and reaction pathways<br>in Chemistry, force diagrams, circuit diagrams, and astrophysical|
||maps in Physics and Astronomy.|
|Abstract and Conceptual Visuals|Visuals explaining theories and concepts, including fowcharts,<br>algorithms, logic models, and symbolic diagrams.|



18 

## **B Data Annotation** 

## **B.1 Problem Extraction and Annotation** 

We develop a simple and practical annotation interface using Streamlit[12] (as shown in Figure 9). Approximately 30 university students are employed to use this interface for annotation. We provide each annotator with a wage higher than the local average hourly rate. The specific fields annotated are shown in Figure 10. We use image URLs to represent pictures, which allows for efficient storage and easy access without embedding large image files directly in the dataset. Each annotated problem is ultimately stored as a JSON file, facilitating subsequent processing. It is worth mentioning that we embed several rule-based checks and filtering mechanisms in the annotation interface to minimize noise from the annotations. When the following situations arise, we promptly identify and correct the annotations: 

1) When the answer type is **Numerical Value** , and the annotated answer contains a variable. 

2) When the answer type is not **Numerical Value** , but the annotated answer can be parsed as a numerical value. 

3) When the answer type is **Expression** , and the annotated answer contains an equals sign. 

4) When the answer type is **Equation** , and the annotated answer does not contain an equals sign. 

5) When the annotated answer contains images that should not be present. 

6) When the annotated answer contains units (since units are a separate field according to Figure 10, we compile a list of common units and manually check and correct answers when suspected units are detected). 

7) When the annotated image links cannot be previewed properly. 

Additionally, we implement a multi-step validation process after the initial annotation is completed. First, we conduct a preliminary check using predefined rules to identify any error data, which is then corrected. Following this, a secondary review is performed by different annotators to further check and correct any errors in the annotations. This cross-checking mechanism helps ensure the accuracy and consistency of the annotations. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0019-11.png)


Figure 9: Annotation Page 

## **B.2 Annotation for Difficulty Levels** 

The definitions of three levels of difficulty are as follows: 

> 12 `https://streamlit.io/` 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0020-00.png)


Figure 10: Example of a json-formatted representation of an annotated problem. 

1) **Knowledge Recall:** This involves the direct recall of factual information and well-defined procedures. It examines the memory of simple knowledge points, i.e., whether certain information is known. 

2) **Concept Application:** This category covers the very basic use of simple concepts to solve easy problems or perform straightforward calculations. It involves applying known information to situations without any complex or multi-step reasoning. The focus is on straightforward application rather than reasoning. 

3) **Cognitive Reasoning:** This involves the use of logical reasoning or visual reasoning to solve problems. It includes problems that require clear thinking and problem-solving techniques. It focuses on the ability to reason and analyze to understand and address the issues. 

The prompt we use for categorizing each problem is shown in Figure 11 

## **B.3 Cognitive Reasoning Abilities Annotation** 

We provide detailed definitions for each of these cognitive reasoning abilities. 

The logical reasoning abilities: 

1) **Deductive Reasoning** involves starting with a general principle or hypothesis and logically deriving specific conclusions. This process ensures that the conclusion necessarily follows from the premises. 

2) **Inductive Reasoning** involves making broad generalizations from specific observations. This type of reasoning infers general principles from specific instances, enhancing our confidence in the generality of certain phenomena. 

3) **Abductive Reasoning** starts with incomplete observations and seeks the most likely explanation. It is used to form hypotheses that best explain the available data. 

4) **Analogical Reasoning** involves using knowledge from one situation to solve problems in a similar situation by drawing parallels. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0021-00.png)


**----- Start of picture text -----**<br>
Problem description:<br>{problem}<br>Answer:<br>{answer}<br>Solution:<br>{solution}*<br>Classification Categories:<br>1. Knowledge Recall: Direct recall of factual information and well-defined procedures. This<br>category examines the memory of simple knowledge points, i.e., whether certain information<br>is known.<br>2. Concept Application: Very basic use of simple concepts to solve easy problems or do<br>straightforward calculations. This involves applying known information to situations without<br>any complex or multi-step reasoning. The focus is on straightforward application rather than<br>reasoning.<br>3. Cognitive Reasoning: Use of logical reasoning or visual reasoning to solve prob-<br>lems. This category includes problems that require clear thinking and problem-solving<br>techniques. It focuses on the ability to reason and analyze to understand and address the issues.<br>Instructions for Classification: Please classify the above problem by selecting the most<br>appropriate category that best represents the type of thinking and approach required to<br>address the problem. Consider the complexity, the need for creativity, and the depth of<br>knowledge required. You should conclude your response with "So, the problem can be<br>categorized as ANSWER .", where ANSWER should be one of the indexes in 1, 2, 3.<br>**----- End of picture text -----**<br>


Figure 11: The prompt template used for annotating the difficulty level of problems. The "solution" part marked with * is optional. 

5) **Cause-and-Effect Reasoning** identifies the reasons behind occurrences and their consequences. This reasoning establishes causal relationships between events. 

6) **Critical Thinking** involves objectively analyzing and evaluating information to form a reasoned judgment. It encompasses questioning assumptions and considering alternative explanations. 

7) **Decompositional Reasoning** breaks down complex problems or information into smaller, more manageable parts for detailed analysis. 

8) **Quantitative Reasoning** involves using mathematical skills to handle quantities and numerical concepts, essential for interpreting data and performing calculations. 

The visual reasoning abilities: 

1) **Pattern Recognition** is the ability to identify and understand repeating forms, structures, or recurring themes, especially when presented visually. This skill is critical in subjects like Chemistry for recognizing molecular structures, Biology for identifying cellular components, and Geography for interpreting topographic maps. 

2) **Spatial Reasoning** is the ability to understand objects in both two and three-dimensional terms and draw conclusions about them with limited information. This skill is often applied in subjects like Math. 

Two-Dimensional Examples: Plane geometry, segments, lengths. 

Three-Dimensional Examples: Solid geometry, spatial visualization 

3) **Diagrammatic Reasoning** represents the capability to solve problems expressed in diagrammatic form, understanding the logical connections between shapes, symbols, and texts. 

21 

Examples: Reading various forms of charts and graphs, obtaining and analyzing statistical information from diagrams. 

4) **Symbol Interpretation** is the ability to decode and understand abstract and symbolic visual information. Examples: Understanding abstract diagrams, interpreting symbols, including representations of data structures such as graphs and linked lists 

5) **Comparative Visualization** represents comparing and contrasting visual elements to discern differences or similarities, often required in problem-solving to determine the relationship between variable components. 

The prompt we use for annotating different logical reasoning abilities and visual reasoning abilities are shown separately in Figure 12 and Figure 13. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0022-04.png)


**----- Start of picture text -----**<br>
Problem description:<br>{problem}<br>Answer:<br>{answer}<br>Solution:<br>{solution}*<br>You need to identify and select the specific types of logical reasoning abilities required to<br>solve the question from the list provided below.<br>Logical Reasoning Abilities:<br>1. Deductive Reasoning: Deductive reasoning involves starting with a general principle<br>or hypothesis and logically deriving specific conclusions. This process ensures that the<br>conclusion necessarily follows from the premises.<br>2. Inductive Reasoning: Inductive reasoning involves making broad generalizations from<br>specific observations. This type of reasoning infers general principles from specific instances,<br>enhancing our confidence in the generality of certain phenomena.<br>3. Abductive Reasoning: Abductive reasoning starts with incomplete observations and seeks<br>the most likely explanation. It is used to form hypotheses that best explain the available data.<br>4. Analogical Reasoning: Analogical reasoning involves using knowledge from one situation<br>to solve problems in a similar situation by drawing parallels.<br>5. Cause-and-Effect Reasoning: Cause-and-effect reasoning identifies the reasons behind<br>occurrences and their consequences. This reasoning establishes causal relationships between<br>events.<br>6. Critical Thinking: Critical thinking involves objectively analyzing and evaluating informa-<br>tion to form a reasoned judgment. It encompasses questioning assumptions and considering<br>alternative explanations.<br>7. Decompositional Reasoning: Decompositional reasoning breaks down complex problems<br>or information into smaller, more manageable parts for detailed analysis.<br>8. Quantitative Reasoning: Quantitative reasoning involves using mathematical skills to<br>handle quantities and numerical concepts, essential for interpreting data and performing<br>calculations.<br>Analyze the question, its answer and explanation (if provided) to determine which of the<br>above reasoning abilities are necessary. Conclude by clearly stating which reasoning abil-<br>ities are involved in solving the question using "So, the involved reasoning abilities are<br>ABILITIES ", where "ABILITIES" represents the numbers corresponding to the list<br>above, separated by commas if multiple abilities are relevant.<br>**----- End of picture text -----**<br>


Figure 12: The prompt template used for annotating different logical reasoning abilities of problems. The "solution" part marked with * is optional. 

22 

Problem description: {problem} Answer: {answer} Solution: {solution}* 

You need to identify and select the specific types of visual reasoning abilities required to solve the question from the list provided below. 

Visual Reasoning Abilities: 

1. Pattern Recognition: The ability to identify and understand repeating forms, structures, or recurring themes, especially when presented visually. This skill is critical in subjects like Chemistry for recognizing molecular structures, Biology for identifying cellular components, and Geography for interpreting topographic maps. 

2. Spatial Reasoning: Spatial reasoning is the ability to understand objects in both two and three-dimensional terms and draw conclusions about them with limited information. This skill is often applied in subjects like Math. Two-Dimensional Examples: Plane geometry, segments, lengths Three-Dimensional Examples: Solid geometry, spatial visualization 3. Diagrammatic Reasoning: The capability to solve problems expressed in diagrammatic form, understanding the logical connections between shapes, symbols, and texts. Examples: Reading various forms of charts and graphs, obtaining and analyzing statistical information from diagrams 

4. Symbol Interpretation: The ability to decode and understand abstract and symbolic visual information. Examples: Understanding abstract diagrams, interpreting symbols, including representations of data structures such as graphs and linked lists 5. Comparative Visualization: Comparing and contrasting visual elements to discern differences or similarities, often required in problem-solving to determine the relationship between variable components. 

Analyze the question, its answer, and any explanation provided to determine which of the above reasoning abilities are necessary. Conclude by clearly stating which reasoning abilities are involved in solving the question using "So, the involved reasoning abilities are _ABILITIES_ ", where "ABILITIES" represents the numbers corresponding to the list above, separated by commas if multiple abilities are relevant. 

Figure 13: The prompt template used for annotating different visual reasoning abilities of problems which have multi-modal inputs. The "solution" part marked with * is optional. 

## **C Experiment Details** 

## **C.1 Prompt for Image Caption** 

The prompt we use for captioning each image in the benchmark for LMMs is shown in Figure 14. 

## **C.2 Models** 

In our experiments, we evaluate a range of both open-source and proprietary LMMs and LLMs. For LMMs, we select the newly released GPT-4o [36] and the powerful GPT-4V [1] from OpenAI. Additionally, we include Claude3 Sonnet [3] from Anthropic, and Gemini Pro Vision[13] [45] from Google, and Qwen-VL-Max [6] from Alibaba. We also evaluate several open-source models, includ- 

13We do not test Gemini-1.5-pro [39] as there are significant rate limits on accessing the model’s API during the time we do experiments. 

23 

[Image] 

Describe the fine-grained content of the image or figure, including scenes, objects, relationships, and any text present. 

Figure 14: The prompt template used for image caption. 

Table 9: LMMs and their corresponding LLMs. 

|**LMM**|**LLM**|
|---|---|
|GPT-4o|GPT-4o|
|GPT-4v|GPT-4|
|Claude3 Sonnet|Claude3 Sonnet|
|Gemini Pro Vision|Gemini Pro|
|LLaVA-NeXT-34B|Nous-Hermes-2-Yi-34B|
|InternVL-Chat-V1.5|InternLM2-20B-Chat|
|Yi-VL-34B|Yi-34B-Chat|
|Qwen-VL-Chat|Qwen-7B-Chat|



ing LLaVA-NeXT-34B [31], InternVL-Chat-V1.5 [12], Yi-VL-34B [55], and Qwen-VL-Chat [7]. For LLMs, we primarily select the corresponding text models of the aforementioned LMMs, such as GPT-4 [2]. Additionally, we include open-source models like Qwen-7B-Chat, Qwen1.5-32B-Chat [5], Yi-34B-Chat [55], and InternLM2-Chat-20B [8]. Table 9 shows the relationship between LMMs and their corresponding LLMs. For the proprietary models, we call the APIs, while for the open-source models, we run them on an 8-card A800 cluster. 

## **C.3 Evaluation Prompts** 

We meticulously design the prompts used for model input during experiments. These prompts are tailored to different answer types, with specific output formats specified for each type. The detailed prompt templates are shown in Figure 15, and the different instructions for each answer type are provided in Table 10. 

You are participating in an international {subject} competition and need to solve the following question. 

{answer type description} 

Here is some context information for this question, which might assist you in solving it: {context}* Problem: {problem} 

All mathematical formulas and symbols you output should be represented with LaTeX. You can solve it step by step and please end your response with: {answer format instruction}. 

Figure 15: The prompt template used for problem input. The "context" part marked with * is optional and refers to supplementary information provided during manual annotation when the problem relies on conclusions from previous questions. The {answer type description} and {answer format instruction} are specified in Table 10. 

24 

## **C.4 Model Hyperparameters** 

For all models, we set the maximum number of output tokens to 2048 and the temperature to 0.0. When performing code generation **(CODE)** tasks, the temperature is set to 0.2. 

## **C.5 Answer-level Evaluation Protocols** 

**Rule-based Evaluation** For problems with fixed answers, we extract the final answer enclosed in "\boxed{}" (using prompts to instruct models to conclude their final answers with boxes) and perform rule-based matching according to the answer type. 

1) For numerical value **(NV)** answers, we handle units by explicitly stating them in the prompts provided to the model, if applicable. During evaluation, we assess only the numerical value output by the model, disregarding the unit. In cases where numerical answers are subject to estimation, such as in physics or chemistry problems, we convert both the model’s output and the correct answer to scientific notation. If the exponent of 10 is the same for both, we allow a deviation of 0.1 in the coefficient before the exponent, accounting for minor estimation errors in the model’s calculations. 

2) For problems where the answer type is an expression **(EX)** or an equation **(EQ)** , we use the SymPy[14] library for comparison. This allows us to accurately assess the equivalence of algebraic expressions and equations by symbolic computation. 

3) For problems requiring the solution of multiple quantities **(MPV)** , our evaluation strictly follows the order of output specified in the prompt, ensuring consistency and correctness in the sequence of results. 

4) In the case of problems with multiple answers **(MA)** , we require the model to output all possible answers, adequately considering various scenarios. 

5) For problems where the answer type is an interval **(IN)** , we strictly compare the open and closed intervals as well as the boundary values of the endpoints. 

6) For problems where the answer type is a set **(SET)** , we compare the set output by the model with the standard answer set to ensure they are completely identical. For problems where the answer type is a tuple **(TUP)** , we compare the tuple output by the model with the standard answer tuple to ensure that each corresponding position is exactly equal. 

7) For code generation **(CODE)** problems, we extract the code output by the model and test it through all provided test cases. Specifically, we use the unbiased pass@k metric, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0025-11.png)


where we set _k_ = 1 and _n_ = 5, and _c_ indicates the number of correct samples that pass all test cases. 

**Model-based Evaluation** To deal with those problems with answer types that cannot be appropriately evaluated using rule-based matching, we employ model-based evaluation. In this approach, we utilize GPT-4V as the evaluator. We design prompts that include the problem, the correct answer, the solution (if provided), and the response from the model being tested (see Figure 16 for details). The evaluator model then judges the correctness of the tested model’s response. 

To further ensure the reliability of using a model as an evaluator, we uniformly sampled 100 problems across various subjects that involved model evaluation. We have several students with backgrounds in science and engineering independently conduct manual evaluations. It turns out that out of the 100 sampled problems, there is nearly 80% agreement between the human evaluations and the model evaluations. Considering that problems requiring model-based evaluation account for approximately 5% of the total, the error rate can be controlled at around 20% _×_ 5%, which is approximately 1%. Therefore, we consider this method to be reliable. 

> 14https://www.sympy.org/ 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0026-00.png)


**----- Start of picture text -----**<br>
You are an experienced teacher tasked with grading an Olympic-level {subject} exam paper.<br>The problem’s context:<br>{context}*<br>Problem:<br>{problem}<br>The student’s answer:<br>{the tested model’s response}<br>The reference answer:<br>{the reference answer}<br>The reference solution:<br>{the reference solution}*<br>Note:<br>(1) You can tolerate some markdown formatting issues.<br>(2) You need to make judgments based on the provided reference answer and reference<br>solution (if provided).<br>You can analyze the answer step by step, and then output correct or incorrect at the end<br>to express your final judgment.<br>**----- End of picture text -----**<br>


Figure 16: The prompt used for model-based evaluation. The "context" and "the reference solution" parts marked with * are optional. 

## **C.6 Process-level Evaluation Protocols** 

To conduct the process-level evaluation, we utilize a method based on GPT-4V. First, we reformat both the gold solution and the model-generated solution for the sampled problems into a neat step-by-step format using GPT-4. Then, we employ a carefully designed prompt(see Figure 17) to guide GPT-4V using the reformatted gold solution to evaluate the correctness of each step in the model’s output, assigning a score of 0 for incorrect and 1 for correct steps. The final process-level score for each problem is determined by averaging the scores of all the steps. 

## **D Fine-grained Results** 

## **D.1 Results across Logical and Visual Reasoning Abilities** 

Table 11 and Table 12 show the performance of different models across various logical and visual reasoning abilities separately. 

## **D.2 Results on Multimodal Problems** 

Table 13 shows the performance of different models on multimodal problems across different subjects. 

## **D.3 Process-level Evaluation Results** 

Table 14 shows process-level results of different models across different subjects. 

## **D.4 Results across Different Languages** 

Table 15 shows results of different models in different languages. 

26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0027-00.png)


**----- Start of picture text -----**<br>
You are a teacher skilled in evaluating the intermediate steps of a student’s solution to a given<br>problem.<br>You are given two types of step-by-step solutions: one from the reference answer and the<br>other from the student. Your task is to evaluate the correctness of each step in the student’s<br>solutions using binary scoring: assign a score of 1 for correct steps and 0 for incorrect steps.<br>Use the reference solutions to guide your evaluation.<br>Follow the format:<br>Step 1: ...<br>Step 2: ...<br>Step 3: ...<br>Please provide the results directly, omitting any introductory or concluding remarks.<br># The given question<br>{the question}<br># The reference solution<br>{the reference solution}<br># The student’s solution<br>{the model’s solution}<br># Your scores for each step of the student’s solutions<br>**----- End of picture text -----**<br>


Figure 17: The prompt used for process-level evaluation. 

## **E Data Leakage Detection Details** 

We combine the questions and detailed solutions (or answers if there are no steps) of the problems, then use the n-gram prediction accuracy metric. Specifically, for each sample, we sample k starting points and predict the next 5-gram each time. To evaluate whether the n-gram prediction is correct, we use exact match and more lenient metrics such as edit distance and ROUGE-L. Here, we consider a prediction correct if either the edit distance or ROUGE-L similarity exceeds 75%, to mitigate some reformatting issues during pre-training. We take the union of instances detected by different metrics to obtain the final set of detected instances. 

As shown in Tables 16, 17, and 18, the experimental results reveal that indeed, different models exhibit minor leakage across different subjects. An interesting observation is that some leakages detected by the base model are no longer detectable when using the chat model based on the same base model. We hypothesize that optimization for dialogue capabilities potentially impacts the model’s ability and performance on the next token prediction. Another similar observation is that leakages detected by text-only chat models tend to decrease when evaluated on multimodal chat models based on the same chat models. Figure 18 presents a data leakage case from Qwen1.5-32B-Chat. 

27 

## An exact match case of Qwen1.5-32B-Chat 

**Text:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times 10^{-31} … 1.6 \times 10^{-19} … \mathrm{~J}$$ **Prompt:** The fractional **Prediction:** The fractional quantum Hall effect (F **Prompt:** The fractional quantum Hall effect (FQHE) was … fraction **Prediction:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasip **Prompt:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times **Prediction:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times 10^{-3 **Prompt:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times 10^{-31} … 1. **Prediction:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times 10^{-31} … 1.6 \times 1 **Prompt:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times 10^{-31} … 1.6 \times 10^{-19} … \mathrm **Prediction:** The fractional quantum Hall effect (FQHE) was … fractionally charged quasiparticles … \times 10^{-31} … 1.6 \times 10^{-19} … \mathrm{~J}$$ 

Figure 18: A potential data leakage case of Qwen1.5-32B-Chat which is presented with the original problem and solution concatenated, separated by a space. 

28 

Table 10: Descriptions of answer types and corresponding format instructions included in the problem input prompts. Specifically, {unit description} indicates: "Remember, your answer should be calculated in the unit of {unit}, but do not include the unit in your final answer." 

|**Answer Type**<br>**Answer Type Description**|**Answer Format Instruction**|
|---|---|
|SC<br>This is a multiple choice question (only one<br>correct answer).|Please end your response with: "The fnal answer is<br>_ANSWER_<br>", where ANSWER should be one of the<br>options: {the options of the problem}.|
|MC<br>This is a multiple choice question (more<br>than one correct answer).|Please end your response with: "The fnal answer is<br>_ANSWER_<br>", where ANSWER should be two or more<br>of the options: {the options of the problem}.|
|TF<br>This is a True or False question.|Please end your response with: "The fnal answer is<br>_ANSWER_<br>", where ANSWER should be either "True"<br>or "False".|
|NV<br>The answer to this question is a numerical<br>value.|{unit instruction}Please end your response with: "The<br>fnal answer is _ANSWER_<br>", where ANSWER is the<br>numerical value without any units.|
|SET<br>The answer to this question is a set.|{unit instruction}Please end your response with: "The<br>fnal answer is _ANSWER_<br>", where ANSWER is the<br>set of all distinct answers, each expressed as a numerical<br>value without any units, e.g. ANSWER = {3, 4, 5}.|
|IN<br>The answer to this question is a range inter-<br>val.|{unit instruction}Please end your response with: "The<br>fnal answer is _ANSWER_<br>", where ANSWER is an in-<br>terval without any units, e.g. ANSWER =(1_,_2]_∪_[7_,_+_∞_).|
|EX<br>The answer to this question is an expres-<br>sion.|{unit instruction}Please end your response with: "The<br>fnal answer is _ANSWER_<br>", where ANSWER is an<br>expression without any units and equals signs, e.g. AN-<br>SWER = 1<br>2_gt_2.|
|EQ<br>The answer to this question is an equation.|{unit instruction}Please end your response with: "The<br>fnal answer is _ANSWER_<br>", where ANSWER is an<br>equation without any units, e.g. ANSWER = _x_2<br>4 + _y_2<br>2 = 1.|
|TUP<br>The answer to this question is a tuple.|{unit instruction}Please end your response with: "The<br>fnal answer is _ANSWER_<br>", where ANSWER is a tuple<br>without any units, e.g. ANSWER=(3, 5).|
|MPV<br>This question involves multiple quantities<br>to be determined.<br>|Your fnal quantities should be output in the following<br>order: {the ordered sequence of the name of multiple<br>quantities}. Their units are, in order, {the ordered se-<br>quence of the units}, but units shouldn’t be included in<br>your concluded answer. Their answer types are, in order,<br>{the ordered sequence of answer types}. Please endyour<br>response with: "The fnal answers are _ANSWER_<br>",<br>where ANSWER should be the sequence of your fnal<br>answers, separated by commas, for example: 5, 7, 2.5.|
|MA<br>This question has more than one correct<br>answer, you need to include them all.|Their units are, in order, {the ordered sequence of the<br>units}, but units shouldn’t be included in your concluded<br>answer. Their answer types are, in order, {the ordered<br>sequence of answer types}. Please end your response with:<br>"The fnal answers are _ANSWER_<br>", where ANSWER<br>should be the sequence of your fnal answers, separated<br>by commas, for example: 5, 7, 2.5.|
|CODE<br>Write a Python program to solve the given<br>competitive programming problem using<br>standard input and output methods. Pay<br>attention to time and space complexities to<br>ensure effciency.|Notes: (1) Your solution must handle standard input and<br>output. Use`input()`for reading input and`print()`for<br>output. (2) Be mindful of the problem’s time and space<br>complexity. The solution should be effcient and designed<br>to handle the upper limits of input sizes within the given<br>constraints. (3) It’s encouraged to analyze and reason<br>about the problem before coding.<br>You can think step by step, and fnally output your fnal<br>code in the following format:<br>`Your Python code here`|
|OT<br>-|-|



29 

Table 11: Experimental results across different logical reasoning abilities on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. **DED** : Deductive Reasoning, **IND** : Inductive Reasoning, **ABD** : Abductive Reasoning, **ANA** : Analogical Reasoning, **CAE** : Cause-and-Effect Reasoning, **CT** : Critical Thinking, **DEC** : Decompositional Reasoning, **QUA** : Quantitative Reasoning. 

|Model|**DED**<br>**IND**<br>**ABD**<br>**ANA**<br>**CAE**<br>**CT**<br>**DEC**<br>**QUA**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|**DED**<br>**IND**<br>**ABD**<br>**ANA**<br>**CAE**<br>**CT**<br>**DEC**<br>**QUA**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|
|---|---|---|
|||LLMs|
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||4.85<br>4.18<br>4.84<br>5.29<br>5.54<br>5.16<br>4.09<br>4.64<br>19.65<br>13.84<br>26.82<br>18.73<br>26.51<br>25.71<br>15.00<br>15.55<br>17.43<br>13.12<br>24.74<br>16.30<br>22.81<br>22.51<br>13.03<br>13.42<br>25.94<br>21.20<br>33.39<br>24.87<br>32.33<br>31.82<br>20.19<br>22.19|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||19.38<br>13.19<br>26.64<br>16.30<br>23.32<br>24.31<br>14.43<br>17.35<br>25.40<br>17.88<br>34.78<br>23.28<br>30.64<br>31.15<br>18.59<br>22.67<br>33.93<br>24.80<br>40.66<br>33.33<br>38.48<br>39.32<br>26.84<br>31.72<br>39.1<br>30.14<br>43.43<br>37.78<br>42.89<br>44.06<br>31.79<br>36.56|
|||Image caption + LLMs|
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||5.66<br>4.69<br>7.27<br>6.88<br>6.66<br>6.02<br>4.60<br>4.81<br>19.08<br>13.34<br>29.24<br>20.11<br>25.53<br>24.91<br>13.79<br>14.64<br>18.25<br>12.69<br>28.37<br>17.67<br>23.84<br>23.35<br>13.40<br>14.52<br>25.50<br>20.55<br>35.81<br>26.35<br>31.35<br>31.39<br>19.55<br>21.51|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||20.71<br>13.91<br>29.76<br>17.78<br>25.72<br>26.01<br>15.74<br>17.73<br>25.69<br>19.11<br>35.12<br>24.02<br>30.88<br>31.55<br>18.71<br>22.89<br>35.06<br>24.44<br>41.35<br>34.39<br>40.17<br>40.72<br>27.26<br>32.47<br>39.26<br>30.93<br>45.50<br>39.37<br>43.17<br>44.19<br>31.35<br>36.56|
|||LMMs|
|Qwen-VL-Chat<br>Yi-VL-34B<br>InternVL-Chat-V1.5<br>LLaVA-NeXT-34B||7.87<br>6.06<br>12.80<br>8.68<br>9.90<br>9.92<br>5.29<br>6.29<br>16.30<br>10.60<br>21.11<br>16.40<br>21.35<br>20.76<br>11.89<br>13.42<br>17.65<br>12.55<br>30.28<br>17.25<br>22.06<br>22.70<br>12.56<br>14.37<br>19.72<br>14.70<br>30.62<br>19.37<br>27.40<br>25.39<br>13.62<br>14.79|
|Qwen-VL-Max<br>Gemini Pro Vision<br>Claude3 Sonnet<br>GPT-4V<br>GPT-4o||22.97<br>16.87<br>33.91<br>21.38<br>29.28<br>28.51<br>17.26<br>18.13<br>22.45<br>17.16<br>35.47<br>21.59<br>25.67<br>27.54<br>17.24<br>18.91<br>25.59<br>18.89<br>36.51<br>23.70<br>29.89<br>31.71<br>18.99<br>22.87<br>34.59<br>25.59<br>46.54<br>33.33<br>39.61<br>41.15<br>26.47<br>30.79<br>**41.18**<br>**32.73**<br>**50.35**<br>**40.53**<br>**45.94**<br>**47.12**<br>**33.17**<br>**37.58**|



30 

Table 12: Experimental results across different visual reasoning abilities on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. **PR** : Pattern Recognition, **SPA** : Spatial Reasoning, **DIA** : Diagrammatic Reasoning, **SYB** : Symbol Interpretation, **COM** : Comparative Visualization. 

|Model|**PR**<br>**SPA**<br>**DIA**<br>**SYB**<br>**COM**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|**PR**<br>**SPA**<br>**DIA**<br>**SYB**<br>**COM**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|**PR**<br>**SPA**<br>**DIA**<br>**SYB**<br>**COM**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|**PR**<br>**SPA**<br>**DIA**<br>**SYB**<br>**COM**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|**PR**<br>**SPA**<br>**DIA**<br>**SYB**<br>**COM**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|**PR**<br>**SPA**<br>**DIA**<br>**SYB**<br>**COM**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|
|---|---|---|---|---|---|---|
||||LLMs||||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||4.59<br>23.70<br>22.89<br>28.93|2.64<br>4.26<br>13.58<br>19.56<br>13.06<br>18.63<br>17.94<br>24.67||4.01<br>17.61<br>15.73<br>22.18|4.66<br>22.37<br>21.16<br>27.83|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||22.33<br>26.88<br>33.65<br>35.96|13.27<br>18.40<br>17.60<br>22.86<br>23.99<br>30.09<br>28.71<br>33.29||16.05<br>20.49<br>27.94<br>31.54|21.05<br>25.98<br>32.54<br>35.00|
|||Image caption + LLMs|||||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||5.96<br>21.69<br>22.97<br>28.59|4.11<br>21.19<br>12.75<br>17.81|5.21<br>18.01<br>18.27<br>23.90|5.11<br>14.92<br>15.49<br>20.95|6.30<br>20.48<br>21.05<br>26.73|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||23.96<br>27.60<br>34.29<br>37.08|15.26<br>17.03<br>26.07<br>29.10|19.72<br>22.84<br>31.07<br>33.60|17.34<br>20.17<br>28.61<br>31.22|22.30<br>26.28<br>33.11<br>35.91|
||||LMMs||||
|Qwen-VL-Chat<br>Yi-VL-34B<br>InternVL-Chat-V1.5<br>LLaVA-NeXT-34B||9.90<br>16.72<br>22.85<br>24.69|4.93<br>9.60<br>12.11<br>12.75|7.46<br>13.78<br>17.68<br>19.72|6.48<br>12.10<br>15.11<br>16.38|8.91<br>15.09<br>21.27<br>22.90|
|Qwen-VL-Max<br>Gemini Pro Vision<br>Claude3 Sonnet<br>GPT-4V<br>GPT-4o||27.43<br>28.98<br>27.18<br>35.28<br>**41.49**|16.26<br>14.83<br>17.55<br>23.91<br>**30.65**|22.35<br>21.65<br>22.43<br>30.25<br>**36.98**|19.47<br>19.79<br>20.84<br>27.70<br>**33.91**|26.01<br>26.13<br>25.56<br>34.40<br>**40.58**|



31 

Table 13: Experimental results on multimodal problems on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. 

|<br>ettings bolded.||||
|---|---|---|---|
|Model|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**CS**<br>**Overall**<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Accuracy<br>Pass@1<br>Accuracy|||
|||LLMs||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||1.26<br>2.54<br>6.92<br>5.59<br>4.16<br>2.70<br>0<br>4.01<br>5.04<br>6.14<br>19.15<br>27.40<br>33.91<br>10.00<br>0.28<br>19.54<br>6.30<br>6.46<br>15.76<br>26.96<br>31.87<br>9.19<br>0.97<br>18.51<br>7.98<br>8.90<br>23.86<br>32.21<br>39.74<br>18.65<br>0.83<br>24.58||
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||6.30<br>7.20<br>15.46<br>26.85<br>30.66<br>11.62<br>6.25<br>18.79<br>8.82<br>11.76<br>19.59<br>31.99<br>38.00<br>15.68<br>2.64<br>23.79<br>16.81<br>18.43<br>32.11<br>39.71<br>41.26<br>23.92<br>12.50<br>31.05<br>21.85<br>21.82<br>32.11<br>42.17<br>44.28<br>30.68<br>12.78<br>34.11||
|||Image caption + LLMs||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||3.78<br>2.22<br>6.19<br>5.04<br>6.57<br>14.29<br>6.72<br>6.89<br>16.35<br>6.72<br>8.69<br>21.80|6.49<br>7.34<br>5.00<br>0<br>5.32<br>24.94<br>33.61<br>8.78<br>0.28<br>18.25<br>25.39<br>31.26<br>10.27<br>1.18<br>18.41<br>32.10<br>39.82<br>17.30<br>0.97<br>23.99|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||4.20<br>12.39<br>18.11<br>5.46<br>13.35<br>20.47<br>16.81<br>20.44<br>29.90<br>21.01<br>22.14<br>31.22|25.73<br>32.32<br>9.73<br>7.64<br>20.04<br>32.89<br>38.23<br>13.38<br>3.89<br>23.97<br>38.7<br>45.12<br>26.62<br>12.26<br>32.36<br>45.19<br>45.19<br>30.54<br>**14.58**<br>34.86|
|||LMMs||
|Qwen-VL-Chat<br>Yi-VL-34B<br>InternVL-Chat-V1.5<br>LLaVA-NeXT-34B||3.36<br>2.65<br>6.63<br>9.84<br>13.85<br>5.27<br>0<br>7.82<br>3.36<br>6.46<br>9.13<br>18.79<br>22.03<br>7.43<br>0<br>13.00<br>7.56<br>6.25<br>16.05<br>24.94<br>32.55<br>9.73<br>0.62<br>18.43<br>4.62<br>6.46<br>14.43<br>28.30<br>36.11<br>10.00<br>0.28<br>19.66||
|Qwen-VL-Max<br>Gemini Pro Vision<br>Claude3 Sonnet<br>GPT-4V<br>GPT-4o||6.30<br>7.63<br>17.82<br>28.86<br>40.05<br>15.14<br>1.25<br>22.38<br>7.56<br>9.11<br>24.30<br>32.55<br>35.81<br>11.22<br>2.36<br>22.58<br>5.46<br>13.45<br>19.15<br>33.22<br>37.02<br>17.30<br>2.36<br>24.05<br>13.87<br>18.22<br>29.31<br>40.27<br>46.86<br>22.43<br>11.25<br>31.81<br>**26.47**<br>**22.14**<br>**33.14**<br>**46.98**<br>**53.75**<br>**31.76**<br>13.61<br>**38.17**||



32 

Table 14: Results of the process-level evaluation on our comprehensive OlympicArena benchmark. Each step of every problem is assigned a score of 0 (indicating incorrect) or 1 (indicating correct), with the highest score in each setting underlined and the highest scores across all settings highlighted in bold. The subject of computer science is neglected in this part due to the lack of solutions. 

|Model|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**Overall**<br>Score<br>Score<br>Score<br>Score<br>Score<br>Score<br>Score|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**Overall**<br>Score<br>Score<br>Score<br>Score<br>Score<br>Score<br>Score|**Math**<br>**Physics**<br>**Chemistry**<br>**Biology**<br>**Geography**<br>**Astronomy**<br>**Overall**<br>Score<br>Score<br>Score<br>Score<br>Score<br>Score<br>Score|
|---|---|---|---|
|||LLMs||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||18.7<br>43.7<br>35.1<br>18.9<br>34.5<br>31.5<br>30.4<br>30.2<br>51.0<br>54.0<br>31.9<br>36.5<br>40.3<br>40.7<br>21.2<br>35.0<br>51.2<br>22.7<br>32.9<br>33.3<br>32.7<br>32.0<br>44.0<br>61.1<br>32.0<br>45.2<br>48.6<br>43.8||
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||37.6<br>46.9<br>32.7<br>30.2<br>38.7<br>26.7<br>35.4<br>40.8<br>42.7<br>65.3<br>30.8<br>52.6<br>50.5<br>47.1<br>57.0<br>53.8<br>73.6<br>50.0<br>50.1<br>65.0<br>58.2<br>59.9<br>**65.9**<br>67.4<br>49.6<br>**61.4**<br>69.5<br>62.3||
|||Image caption + LLMs||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||23.0<br>42.6<br>34.6<br>26.3<br>45.6<br>49.5<br>27.7<br>42.6<br>46.3<br>35.9<br>49.7<br>56.8|17.4<br>34.4<br>32.3<br>30.7<br>20.0<br>45.7<br>42.0<br>38.2<br>19.4<br>25.5<br>43.1<br>34.1<br>33.5<br>43.6<br>51.4<br>45.1|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||32.1<br>46.7<br>51.2<br>50.7<br>51.7<br>66.1<br>61.4<br>53.8<br>62.7<br>54.3<br>63.3<br>71.8|29.1<br>38.4<br>38.2<br>39.3<br>33.4<br>55.8<br>52.2<br>51.7<br>51.1<br>52.0<br>62.2<br>57.2<br>**58.6**<br>56.6<br>72.6<br>**62.9**|
|||LMMs||
|Qwen-VL-Chat<br>Yi-VL-34B<br>InternVL-Chat-V1.5<br>LLaVA-NeXT-34B||14.3<br>41.7<br>35.7<br>28.9<br>41.0<br>44.2<br>26.6<br>40.5<br>42.7<br>30.2<br>47.1<br>50.1|21.0<br>31.0<br>23.6<br>27.9<br>18.7<br>30.2<br>40.3<br>33.9<br>29.4<br>43.1<br>44.8<br>37.8<br>19.0<br>40.6<br>47.1<br>39.0|
|Qwen-VL-Max<br>Gemini Pro Vision<br>Claude3 Sonnet<br>GPT-4V<br>GPT-4o||27.5<br>52.4<br>65.5<br>28.5<br>46.4<br>45.2<br>47.3<br>46.8<br>63.2<br>49.9<br>54.0<br>71.1<br>**60.2**<br>54.8<br>**72.2**|24.3<br>36.0<br>48.4<br>42.3<br>19.9<br>33.5<br>40.5<br>35.7<br>24.2<br>43.2<br>48.1<br>45.5<br>51.4<br>56.3<br>64.3<br>57.8<br>51.6<br>59.6<br>**74.4**<br>62.1|



33 

Table 15: Experimental results across different languages (English and Chinese) on OlympicArena benchmark, expressed as percentages, with the highest score in each setting underlined and the highest scores across all settings bolded. 

|Model|**English**<br>**Chinese**<br>Accuracy<br>Accuracy|**English**<br>**Chinese**<br>Accuracy<br>Accuracy|**English**<br>**Chinese**<br>Accuracy<br>Accuracy|
|---|---|---|---|
|LLMs||||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||4.17<br>16.37<br>16.56<br>22.73|4.55<br>18.89<br>16.62<br>25.29|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||19.83<br>25.73<br>35.13<br>40.65|15.50<br>18.20<br>27.31<br>33.66|
|Image caption + LLMs||||
|Qwen-7B-Chat<br>Yi-34B-Chat<br>Internlm2-20B-Chat<br>Qwen1.5-32B-Chat||4.71<br>16.96<br>17.40<br>22.93|5.21<br>16.26<br>16.43<br>24.24|
|GPT-3.5<br>Claude3 Sonnet<br>GPT-4<br>GPT-4o||20.56<br>26.31<br>36.08<br>41.50|15.77<br>17.43<br>27.40<br>33.07|
|LMMs||||
|Qwen-VL-Chat<br>Yi-VL-34B<br>InternVL-Chat-V1.5<br>LLaVA-NeXT-34B||7.70<br>17.34<br>17.07<br>17.74|5.55<br>14.68<br>15.82<br>16.74|
|Qwen-VL-Max<br>Gemini Pro Vision<br>Claude3 Sonnet<br>GPT-4V<br>GPT-4o||20.14<br>21.61<br>26.52<br>36.18<br>**43.04**|21.49<br>18.76<br>17.21<br>26.55<br>**34.39**|



Table 16: Full results of Data Leakage Detection on the base models or text-only chat models behind the evaluated models (continued). The “Correspondence” column indicates the text-only chat model and multimodal (MM) chat model corresponding to the model being detected. “# Leak.” denotes the number of leakage instances. “# T” represents the number of instances correctly answered among these leaks by the text-only chat model, while “# MM” represents the number of instances correctly answered among these leaks by the multimodal chat model. 

|**Model to-be-detected**|**Correspondence**|**Math**<br>**Physics**<br>**Chemistry**|**Math**<br>**Physics**<br>**Chemistry**|**Math**<br>**Physics**<br>**Chemistry**|
|---|---|---|---|---|
||**Text-only Chat Model**<br>**MM Chat Model**|**# Leak.**<br>**# T**<br>**# MM**|**# Leak.**<br>**# T**<br>**# MM**|**# Leak.**<br>**# T**<br>**# MM**|
|InternLM2-20B<br>internLM2-20B-Chat<br>Yi-34B<br>Yi-34B-Chat<br>Nous-Hermes-2-Yi-34B<br>Qwen-7B<br>Qwen1.5-32B<br>Qwen1.5-32B-Chat<br>GPT-4o|InternLM2-20B-Chat<br>InternVL-Chat-1.5<br>InternLM2-20B-Chat<br>InternVL-Chat1.5<br>Yi-34B-Chat<br>Yi-VL-34B<br>Yi-34B-Chat<br>Yi-VL-34B<br>-<br>LLaVA-NeXT-34B<br>Qwen-7B-Chat<br>Qwen-VL-Chat<br>Qwen1.5-32B-Chat<br>-<br>Qwen1.5-32B-Chat<br>-<br>GPT-4o<br>GPT-4o|14<br>1<br>2<br>17<br>1<br>0<br>10<br>2<br>2<br>2<br>0<br>0<br>0<br>-<br>0<br>8<br>0<br>0<br>24<br>3<br>-<br>19<br>2<br>-<br>0<br>0<br>0|3<br>0<br>0<br>0<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>1<br>1<br>0<br>3<br>1<br>-<br>4<br>2<br>-<br>0<br>0<br>0|0<br>0<br>0<br>1<br>1<br>1<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>0<br>0<br>0<br>1<br>0<br>-<br>3<br>1<br>-<br>0<br>0<br>0|



34 

Table 17: Full results of Data Leakage Detection on the base models or text-only chat models behind the evaluated models (continued). The “Correspondence” column indicates the text-only chat model and multimodal (MM) chat model corresponding to the model being detected. “# Leak.” denotes the number of leakage instances. “# T” represents the number of instances correctly answered among these leaks by the text-only chat model, while “# MM” represents the number of instances correctly answered among these leaks by the multimodal chat model. 

|**Model to-be-detected**|**Correspondence**|**Biology**<br>**Geography**<br>**Astronomy**|**Biology**<br>**Geography**<br>**Astronomy**|**Biology**<br>**Geography**<br>**Astronomy**|
|---|---|---|---|---|
||**Text-only Chat Model**<br>**MM Chat Model**|**# Leak.**<br>**# T**<br>**# MM**|**# Leak.**<br>**# T**<br>**# MM**|**# Leak.**<br>**# T**<br>**# MM**|
|InternLM2-20B<br>InternLM2-20B-Chat<br>Yi-34B<br>Yi-34B-Chat<br>Nous-Hermes-2-Yi-34B<br>Qwen-7B<br>Qwen1.5-32B<br>Qwen1.5-32B-Chat<br>GPT-4o|InternLM2-20B-Chat<br>InternVL-Chat-1.5<br>InternLM2-20B-Chat<br>InternVL-Chat-1.5<br>Yi-34B-Chat<br>Yi-VL-34B<br>Yi-34B-Chat<br>Yi-VL-34B<br>-<br>LLaVA-NeXT-34B<br>Qwen-7B-Chat<br>Qwen-VL-Chat<br>Qwen1.5-32B-Chat<br>-<br>Qwen1.5-32B-Chat<br>-<br>GPT-4o<br>GPT-4o|0<br>0<br>0<br>0<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>0<br>0<br>0<br>1<br>0<br>-<br>1<br>0<br>-<br>0<br>0<br>0|0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>0<br>-<br>0<br>0<br>0|1<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>1<br>0<br>0<br>5<br>1<br>-<br>0<br>0<br>-<br>0<br>0<br>0|



Table 18: Full results of Data Leakage Detection on the base models or text-only chat models behind the evaluated models (continued). The “Correspondence” column indicates the text-only chat model and multimodal (MM) chat model corresponding to the model being detected. “# Leak.” denotes the number of leakage instances. “# T” represents the number of instances correctly answered among these leaks by the text-only chat model, while “# MM” represents the number of instances correctly answered among these leaks by the multimodal chat model. 

|**Model to-be-detected**|**Correspondence**|**CS**|**Overall**|
|---|---|---|---|
||**Text-only Chat Model**<br>**MM Chat Model**|**# Leak.**<br>**# T**<br>**# MM**|**# Leak.**<br>**# T**<br>**# MM**|
|InternLM2-20B<br>InternLM2-20B-Chat<br>Yi-34B<br>Yi-34B-Chat<br>Nous-Hermes-2-Yi-34B<br>Qwen-7B<br>Qwen1.5-32B<br>Qwen1.5-32B-Chat<br>GPT-4o|InternLM2-20B-Chat<br>InternVL-Chat-1.5<br>InternLM2-20B-Chat<br>InternVL-Chat-1.5<br>Yi-34B-Chat<br>Yi-VL-34B<br>Yi-34B-Chat<br>Yi-VL-34B<br>-<br>LLaVA-NeXT-34B<br>Qwen-7B-Chat<br>Qwen-VL-Chat<br>Qwen1.5-32B-Chat<br>-<br>Qwen1.5-32B-Chat<br>-<br>GPT-4o<br>GPT-4o|1<br>1<br>1<br>1<br>1<br>1<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>-<br>0<br>1<br>1<br>1<br>9<br>-<br>-<br>3<br>3<br>-<br>0<br>0<br>0|19<br>2<br>3<br>19<br>3<br>2<br>12<br>2<br>2<br>2<br>0<br>0<br>0<br>0<br>0<br>11<br>2<br>1<br>43<br>14<br>0<br>30<br>8<br>0<br>0<br>0<br>0|



35 

## **F Case Study** 

## **F.1 Cases for Error Analysis** 

From Figure 19 to Figure 25, we showcase examples of various error types across different disciplines. 

36 

## Math - Logical Reasoning Error 

## **Problem:** 

In the diagram, rectangle _P QRS_ is placed inside rectangle _ABCD_ in two different ways: first, with _Q_ at _B_ and _R_ at _C_ ; second, with _P_ on _AB, Q_ on _BC, R_ on _CD_ , and _S_ on _DA_ . [figure1]. If _AB_ = 718 and _P Q_ = 250, determine the length of _BC_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0037-03.png)


[figure1] **Solution:** Let _BC_ = _x, P B_ = _b_ , and _BQ_ = _a_ . Since _BC_ = _x_ , then _AD_ = _P S_ = _QR_ = _x_ . Since _BC_ = _x_ and _BQ_ = _a_ , then _QC_ = _x − a_ . Since _AB_ = 718 and _P B_ = _b_ , then _AP_ = 718 _− b_ . Note that _P Q_ = _SR_ = 250. Let ∠ _BQP_ = _θ_ . Since _△P BQ_ is right-angled at _B_ , then ∠ _BP Q_ = 90 _[◦] − θ_ . Since _BQC_ is a straight angle and ∠ _P QR_ = 90 _[◦]_ , then ∠ _RQC_ = 180 _[◦] −_ 90 _[◦] − θ_ = 90 _[◦] − θ_ . Since _AP B_ is a straight angle and ∠ _SP Q_ = 90 _[◦]_ , then ∠ _AP S_ = 180 _[◦] −_ 90 _[◦] −_ �90 _[◦] − θ_ � = _θ_ . Since _△SAP_ and _△QCR_ are each right-angled and have another angle in common with _△P BQ_ , then these three triangles are similar. Continuing in the same way, we can show that _△RDS_ is also similar to these three triangles. Since _RS_ = _P Q_ , then _△RDS_ is actually congruent to _△P BQ_ (angle-side-angle). Similarly, _△SAP_ is congruent to _△QCR_ . In particular, this means that _AS_ = _x − a, SD_ = _a, DR_ = _b_ , and _RC_ = 718 _− b_ . Since _△SAP_ and _△P BQ_ are similar, then _P B[SA]_[=] _BQ[AP]_[=] _P Q[SP]_[.][Thus,] _x−b a_ =[7][1][8] _a[−][b]_ = 250 _x_[.][Also, by the Pythagorean Theorem in] _[ △][P BQ]_[, we obtain] _[ a]_[2][+] _[ b]_[2][=][250][2][.][By the Pythagorean Theorem in] _[ △][SAP]_[ ,] _x_[2] = ( _x − a_ )[2] + (718 _− b_ )[2] _x_[2] = _x_[2] _−_ 2 _ax_ + _a_[2] + (718 _− b_ )[2] 0 = _−_ 2 _ax_ + _a_[2] + (718 _− b_ )[2] Since _a_[2] + _b_[2] = 250[2] , then _a_[2] = 250[2] _− b_[2] . Since[7][1][8] _a[−][b]_ = 250 _x_[, then] _[ ax]_[=][250(718] _[ −][b]_[)][.][Therefore, substituting into][ (] _[∗]_[)][, we obtain] 0 = _−_ 2(250)(718 _− b_ ) + 250[2] _− b_[2] + (718 _− b_ )[2] _b_[2] = 250[2] _−_ 2(250)(718 _− b_ ) + (718 _− b_ )[2] _b_[2] = ((718 _− b_ ) _−_ 250)[2] � since _y_[2] _−_ 2 _yz_ + _z_[2] = ( _y − z_ )[2][�] _b_[2] = (468 _− b_ )[2] _b_ = 468 _− b_ ( since _b_ = _b −_ 468) 2 _b_ = 468 _b_ = 234 Therefore, _a_[2] = 250[2] _− b_[2] = 250[2] _−_ 234[2] = (250 + 234)(250 _−_ 234) = 484 _·_ 16 = 22[2] _·_ 4[2] = 88[2] and so _a_ = 88. Finally, _x_ =[250][(][718] _a[−][b]_[)] =[250] 88 _[·]_[484] = 1375. Therefore, _BC_ = 1375 . 

## **GPT 4V’s output:** 

To solve this problem, we can use the Pythagorean theorem. In the first placement of rectangle _P QRS_ inside rectangle _ABCD_ , we see that _P Q_ is parallel to _AB_ and _QR_ is parallel to _BC_ . Since _P Q_ = 250 and _AB_ = 718, we can denote the length of _BC_ as _x_ . In the second placement, the rectangle _P QRS_ is rotated such that its corners touch the sides of rectangle _ABCD_ . The diagonal _P R_ of rectangle _P QRS_ becomes the diameter of an inscribed circle within rectangle _ABCD_ . This means that the diagonal _P R_ is the hypotenuse of a right triangle with sides _P Q_ and _QR_ . Since _P QRS_ is a rectangle, _P Q_ = _RS_ and _P S_ = _QR_ . Therefore, the length of _QR_ is also _x_ . Now we have a right triangle with sides 250 and _x_ , and the hypotenuse is the diagonal _P R_ . Using the Pythagorean theorem, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0037-07.png)


Figure 19: An example of a math problem with a logical reasoning error. 

37 

## Physics - Logical Reasoning Error 

## **Problem:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0038-02.png)


**----- Start of picture text -----**<br>
A bead is placed on a horizontal rail, along which it can slide frictionlessly. It is attached to the end of a rigid, massless rod of length  R . A ball is attached at<br>the other end. Both the bead and the ball have mass  M . The system is initially stationary, with the ball directly above the bead. The ball is then given an<br>infinitesimal push, parallel to the rail.[figure1] Assume that the rod and ball are designed in such a way (not shown explicitly in the diagram) so that they<br>can pass through the rail without hitting it. In other words, the rail only constrains the motion of the bead. Two subsequent states of the system are shown<br>below.[figure2] Derive an expression for the force in the rod when the ball is directly below the bead, as shown at right above.<br>[figure1] [figure2]<br>Solution:<br>At this point the released gravitational potential energy is 2 MgR , and both masses are moving horizontally with speed  v , where<br>1<br>2 Mv [2] = 2 MgR ⇒ v = �2 gR<br>2<br>Work in the frame moving to the right with speed  v . In this frame the bead is stationary and the ball has velocity 2 v  and is instantaneously rotating about the<br>bead, so it must be experiencing a centripetal force<br>M (2 v ) [2] = 8 Mg<br>R<br>Unlike in part (a), there are no additional contribution from the acceleration of the rotation center, because the bead can only ever accelerate horizontally, and<br>the force in the rod at this moment is vertical. Since the ball also experiences a downward force of  Mg  due to gravity, the force in the rod is a tension<br>T = 9 Mg<br>GPT 4V’s output:<br>To derive an expression for the force in the rod when the ball is directly below the bead, we can employ the conservation of energy and principles from circular<br>motion.<br>Since the bead and ball are on a frictionless rail and assuming no other non-conservative forces are acting on the system, the total mechanical energy is conserved.<br>The initial potential energy of the ball when it is given an infinitesimal push will convert into kinetic energy as the ball swings down.<br>At the top position (initial state), the potential energy (PE) is maximum and kinetic energy (KE) is zero. We define the gravitational potential energy to be zero<br>at the lowest point. At the highest point, the potential energy is  MgR  because the height of the ball above the lowest point is  R .<br>Hence,  P E initial = MgR .<br>At the bottom position (final state), when the ball is directly below the bead, all the potential energy will have been converted to kinetic energy (assuming<br>conservation of energy). If  v  is the velocity of the ball at the bottom, then  KE final = [1] 2 [Mv] [2][.]<br>Since energy is conserved,  P E initial = KE final. Thus,<br>1<br>MgR = Mv [2]<br>2<br>2 gR = v [2]<br>v = �2 gR<br>Now, let us consider the forces acting on the ball at the bottom position. We have the gravitational force  Mg  acting downwards and the force in the rod  F<br>acting upwards. When the ball is at the bottom, it is in circular motion and the centripetal force required to keep it in circular motion is provided entirely by the<br>force in the rod, which is the centripetal force  F centripetal, since gravity acts in the same direction as the centripetal acceleration at this point.<br>The centripetal force is given by  F centripetal = [Mv] R [2] .<br>Substituting  v = [√] 2 gR  into the above equation, we get<br>F centripetal = M (2 RgR )<br>F centripetal = 2 Mg<br>Since the centripetal force is provided by the force in the rod and directed towards the center of the circular motion (upwards towards the ball), at this moment,<br>the rod is under tension, not compression.<br>Thus, the force in the rod when the ball is directly below the bead and in tension is  F = 2 Mg .<br>The final answer is 2 Mg .<br>**----- End of picture text -----**<br>


Figure 20: An example of a physics problem with a logical reasoning error. 

38 

## Chemistry - Visual Reasoning Error 

## **Problem:** 

At temperature t _[◦]_ C, a solution is prepared such that c (H2CO3) + c �HCO _[−]_ 3 � + c �CO[2] 3 _[−]_ � = 1 _._ 000 _×_ 10 _[−]_[3] mol _·_ L _[−]_[1] . This solution is a mixture of H2CO3 and HCl or H2CO3 and NaOH. The negative logarithms of the concentrations of some particles in the solution ( _−_ lgc) versus pH are shown in the figure. Which of the following statements are incorrect: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0039-03.png)


[figure1] A: In the solution with pH = a: c �HCO _[−]_ 3 � _>_ c (H2CO3) _>_ c �CO[2] 3 _[−]_ � _>_ c �H[+][�] B: In the solution with pH = 7: c �Na[+][�] _>_ c (H2CO3) C: In the solution with pH = 10 _._ 3: c �Na[+][�] _<_ 1 _._ 000 _×_ 10 _[−]_[3] mol _·_ L _[−]_[1] D: At 25 _[◦]_ C, the equilibrium constant for the reaction H2CO3 + CO[2] 3 _[−]_ ⇌ 2HCO _[−]_ 3[is][ 1] _[.]_[0] _[ ×]_[ 10][4] **Solution:** 

A. According to the information from the figure, in the solution with pH = a, c �CO[2] 3 _[−]_ � _>_ c (H2CO3), so statement A is incorrect. B. In the solution with pH = 7, according to the information from the figure, c �HCO _[−]_ 3 � _>_ c (H2CO3) _>_ c �CO[2] 3 _[−]_ �. According to charge conservation and pH = 7, c �Na[+][�] = c �HCO _[−]_ 3 � + 2c �CO[2] 3 _[−]_ �, so c �Na[+][�] _>_ c (H2CO3). Therefore, statement B is correct. C. At 25 _[◦]_ C, when pH = 6 _._ 3, c (H2CO3) = c �HCO _[−]_ 3 �, then _Ka_ (H2CO3) = c �H[+][�] = 1 _._ 000 _×_ 10 _[−]_[6] _[.]_[3] mol _·_ L _[−]_[1] . In the solution with pH = 10 _._ 3, c �CO[2] 3 _[−]_ � = c �HCO _[−]_ 3 �, _Ka_ �HCO _[−]_ 3 � = c �H[+][�] = 1 _._ 000 _×_ 10 _[−]_[10] _[.]_[3] mol _·_ L _[−]_[1] . Then _Ka_ (H2CO3) = c�HCOc(H2CO3) _[−]_ 3 � _×_ c�H[+][�] = c�HCOc(H2CO3) _[−]_ 3 � _×_ 10 _[−]_[10] _[.]_[3] = 10 _[−]_[6] _[.]_[3] . When pH = 10 _._ 3, c �CO[2] 3 _[−]_ � = c �HCO _[−]_ 3 �, so c�HCOc(H2CO3) _[−]_ 3 �+c�CO[2] 3 _[−]_ � = 2 _×_ 10[4] , c (H2CO3) + c �HCO _[−]_ 3 � + c �CO[2] 3 _[−]_ � = 1 _._ 000 _×_ 10 _[−]_[3] mol _·_ L _[−]_[1] . Therefore, c (H2CO3) = 5 _._ 000 _×_ 10 _[−]_[8] mol _·_ L _[−]_[1] , and 2c �CO[2] 3 _[−]_ � = 1 _._ 000 _×_ 10 _[−]_[3] mol _·_ L _[−]_[1] _−_ 5 _._ 000 _×_ 10 _[−]_[8] mol _·_ L _[−]_[1] . According to charge conservation, c �Na[+][�] _>_ c �HCO _[−]_ 3 � + 2c �CO[2] 3 _[−]_ �, i.e., c �Na[+][�] _>_ 3c �CO[2] 3 _[−]_ �, and c �Na[+][�] _>_ 1 _._ 000 _×_ 10 _[−]_[3] mol _·_ L _[−]_[1] . Therefore, statement C is incorrect. D. At 25 _[◦]_ C, when pH = 6 _._ 3, c (H2CO3) = c �HCO _[−]_ 3 �, then _Ka_ (H2CO3) = c �H[+][�] = 1 _._ 000 _×_ 10 _[−]_[6] _[.]_[3] mol _·_ L _[−]_[1] . Similarly, at pH = 10 _._ 3, c �CO[2] 3 _[−]_ � = c �HCO _[−]_ 3 �, then _Ka_ �HCO _[−]_ 3 � = c �H[+][�] = 1 _._ 000 _×_ 10 _[−]_[10] _[.]_[3] mol _·_ L _[−]_[1] . The equilibrium constant for the reaction H2CO3 + CO[2] 3 _[−]_ ⇌ 2HCO _[−]_ 3[is:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0039-06.png)


## **GPT 4V’s output:** 

First, we need to analyze the chart, which shows the variation of the negative logarithm of concentrations (-lgc) of H2CO3, HCO3 _[−]_ , CO3[2] _[−]_ , and H[+] at different pH values. This chart is a typical acid-base titration curve where H2CO3 is a diprotic weak acid that can dissociate into HCO3 _[−]_ and CO3[2] _[−]_ . Option A: At pH = a, according to the chart, the concentration of HCO3 _[−]_ is the highest, followed by H2CO3, then CO3[2] _[−]_ , and the concentration of H[+] is the lowest. This is consistent with the chart. 

Option B: At pH = 7, the concentration of H2CO3 should be relatively low because it is a weak acid and does not dissociate completely at neutral pH. The concentration of Na[+] depends on whether NaOH has been added. If NaOH has been added, the concentration of Na[+] will increase. However, this option does not provide enough information to determine the exact concentration relationship between Na[+] and H2CO3, so we cannot confirm if this option is correct. Option C: At pH = 10.3, according to the chart, the concentration of CO3[2] _[−]_ is the highest, and the concentration of HCO3 _[−]_ is very low at this pH. If NaOH is added to the solution, the concentration of Na[+] will increase. However, since the total concentration _c_ (H2CO3) + _c_ (HCO _[−]_ 3[) +] _[ c]_[(][CO][2] 3 _[−]_ ) = 1 _._ 000 _×_ 10 _[−]_[3] mol·L _[−]_[1] , the concentration of Na[+] cannot be lower than this value. Therefore, this option is incorrect. Option D: The equilibrium constant K for the reaction H2CO3 + CO[2] 3 _[−]_ ⇌ 2HCO _[−]_ 3[can be expressed by the following equation:] 

_K_ = [HCO _[−]_ 3[]][2] [H2CO3][CO[2] 3 _[−]_ ] According to the chart, at pH = 10.3, the concentrations of HCO3 _[−]_ and CO3[2] _[−]_ are equal, which means _K_ = 1 _._ 0 _×_ 10[4] is impossible because this would require [HCO _[−]_ 3[]][ to be much greater than][ [][CO][2] 3 _[−]_ ]. Therefore, this option is also incorrect. In summary, options C and D are incorrect. So the final answer is _C, D_ 

Figure 21: An example of a chemistry problem with a visual reasoning error. 

39 

## Biology - Logical Reasoning Error 

## **Problem:** 

The bacterium Bradyrhizobium japonicum can infect soybean (Glycine max) roots and form nodules. The nitrogen fixation catalyzed by nitrogenase occurs in the nodules and the nitrogenase activity can be measured easily by acetylene reduction instead of nitrogen reduction. Scientists generated a defective mutation of NAD[+] -dependent malic enzyme, the enzyme that generates pyruvate and NADH, and infected soybean seedling roots with wildtype and mutant bacteria. The seedlings were grown in nitrogen-free media. After 14 and 28 days of inoculation, the number and weight of nodules in the seedlings and acetylene reduction activity were recorded [figure1]. Which of the following statements are correct: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0040-03.png)


- [figure1]: Nodule number and dry weight and acetylene reduction acitivity of soybean. Soybean nodules infected with wild-type B. japonicum (open bars) and the dme mutant (solid bars) are presented. A: Nitrogen fixation activity in nodules of the same treament at 28 days after inoculation is higher than that at 14 days after inoculation. B: Both number and size of nodules increase with time from 14 to 28 days after inoculation with _B_ . japonicum. C: The reduction in nitrogen-fixing activity of nodules infected by the mutant at 28 days after inoculation compared to those at 14 days after inoculation is due to the reduction of nitrogenase activity and nodule formation. D: Nitrogen fixation in B. japonicum -induced nodule is down-regulated by NAD[+] dependent malic enzyme. 

- **Solution:** A. True. As shown in the figure, in both wildtype and mutant treaments, acetylene reduction activity in nodules at 28 days after inoculation is higher than that at 14 days after inoculation, indicating higher nitrogen fixation actitity. B. True. In both treaments, number and size of nodules are higher at 28 days compared to those at 14 days after inoculation C. False. Number of nodules at 28 days after inoculation is higher than at 14 days after inoculation 

- D. False. The mutation of NAD[+] -dependent malic enzyme in the bacteria results in a reduction of acetylene reduction, indicating that the enzyme (in wildtype) up-regulates the nitrogen fixation. So we choose _A, B_ . 

## **GPT 4V’s output:** 

- Let’s analyze the data presented in Fig.Q13 step by step: A: We need to determine if nitrogen fixation activity in nodules of the same treatment at 28 days after inoculation is higher than that at 14 days after inoculation. Looking at the graphs C and D, which represent acetylene reduction activity (a proxy for nitrogen fixation activity), we can see that for both wild-type (open bars) and mutant (solid bars) B. japonicum, the activity is higher at 28 days than at 14 days. This is true for both per gram of nodule dry weight (graph C) and per nodule (graph D). Therefore, statement A is correct. B: To assess this statement, we need to look at graphs A and B. Graph A shows the nodule number per plant, and graph B shows the nodule dry weight. For both wild-type and mutant B. japonicum, the number and size (weight) of nodules increase from 14 to 28 days after inoculation. Therefore, statement B is correct. C: This statement suggests that the reduction in nitrogen-fixing activity of nodules infected by the mutant at 28 days after inoculation compared to those at 14 days is due to the reduction of nitrogenase activity and nodule formation. From graph C, we can see that the acetylene reduction activity per gram of nodule dry weight for the mutant does decrease from 14 to 28 days, indicating a reduction in nitrogenase activity. However, graph A shows that the number of nodules actually increases for the mutant from 14 to 28 days, which contradicts the second part of the statement. Therefore, statement C is not entirely correct. D: The statement suggests that nitrogen fixation in B. japonicum-induced nodules is down-regulated by NAD+-dependent malic enzyme. The mutant, which has a defective mutation in the NAD+-dependent malic enzyme, shows lower acetylene reduction activity compared to the wild-type in both graphs C and D at both time points. This suggests that the enzyme plays a role in nitrogen fixation, and its absence (or defectiveness) leads to down-regulation of nitrogen fixation. Therefore, statement D is correct. The final answer is _A, B, D_ . 

Figure 22: An example of a biology problem with a logical reasoning error. 

40 

## Geography - Knowledge Deficit Error 

## **Problem:** 

From December 17 to 20, 2020, the Beijing team went to Tengchong, Yunnan to participate in the study tour course design competition, using the Beihai Wetland in Tengchong, known for its plateau volcanic dammed lakes, as the theme of the study tour course design. Based on this, complete the following small questions. During the study tour, the team members found that compared to Beijing, Tengchong has ( ). 

A. The tree shadows at sunrise are more towards the north. B. The temperature difference between day and night is larger. C. The sunset time is earlier. D. The tree shadows at noon are shorter. 

**Solution:** According to the material, Tengchong �25 _[◦]_ N _,_ 98 _[◦]_ E� belongs to Yunnan and is located to the southwest of Beijing �39 _[◦]_ N _,_ 116 _[◦]_ E�. From December 17 to 20, 2020, it was winter, around the winter solstice, with the sun rising in the southeast and the shadow at sunrise being in the northwest. Beijing is at a higher latitude than Tengchong, causing a greater shadow deviation, so A is incorrect. Tengchong is located near a plateau volcanic dammed lake, rich in geothermal energy, resulting in high nighttime temperatures and smaller temperature differences between day and night, so B is incorrect. Tengchong, being at a lower latitude, has longer daylight hours and a later sunset time, making C incorrect. Tengchong’s lower latitude also means a larger solar altitude angle at noon, leading to shorter tree shadows, so _D_ is correct. **GPT 4V’s output:** First, we need to understand the geographical locations of Beijing and Tengchong. Beijing is in the northern part of China, while Tengchong is in the southwestern part of China, near the Myanmar border. This means that Tengchong is to the southwest of Beijing. A. The tree shadows at sunrise are more towards the north - Due to the Earth’s rotation, the direction of tree shadows at sunrise is mainly determined by latitude, not by the east-west position. Therefore, this option is incorrect. B. The temperature difference between day and night is larger - Tengchong is located in a plateau area, while Beijing is in a plain area. Generally, the temperature difference between day and night in plateau areas is larger than in plain areas because high-altitude areas cool down quickly at night and receive strong solar radiation during the day. Therefore, this option might be correct. C. The sunset time is earlier - The sunset time is mainly determined by longitude, as the Earth rotates from west to east. Tengchong is to the southwest of Beijing, but this does not necessarily mean that the sunset time is earlier. This option requires more information to determine. D. The tree shadows at noon are shorter - The solar altitude angle at noon depends on latitude; the lower the latitude, the higher the solar altitude angle, and the shorter the tree shadows. Tengchong has a lower latitude than Beijing, so the tree shadows at noon should be shorter. In summary, options B and D are both possible. However, we need to determine which one is most related to the "plateau volcanic dammed lake" mentioned in the problem. Since this description emphasizes the plateau characteristics of Tengchong, we can reasonably infer that the larger temperature difference between day and night is most relevant to this characteristic. Therefore, the final answer is _B_ . 

Figure 23: An example of a geography problem with a knowledge deficit error. 

41 

## Astronomy - Incomplete Response 

**Problem:** 

In order to move a spacecraft between orbits we must apply a thrust using rockets, which changes the velocity of the spacecraft by ∆ _v_ . In this question we will ignore changes in the mass of the spacecraft due to the burning of fuel. For an object of mass _m_ in a circular orbit of radius _r_ around an object with mass _M_ (where _m ≪ M_ ) the orbital velocity, _v_ orb , is given by the formula _v_ orb = � _GMr_ . Approximating Mars’ orbit as circular with a radius of 1 _._ 52AU, calculate the _△v_ to go from Earth LEO to Mars i.e. ∆ _v_ = _|_ ∆ _vA|_ + _|_ ∆ _vB |_ . Compare your answer to the ∆ _v_ to reach Earth LEO. 

## **Solution:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0042-04.png)


**----- Start of picture text -----**<br>
6 . 67  ×  10 [−] [11] ×  1 . 99  ×  10 [30] 2  ×  1 . 52<br>∆ v A = − 1<br>� 1 . 49  ×  10 [11] �� 1 + 1 . 52 �<br>= 2935 m s [−] [1] [�] = 2 . 94 km s [−] [1][�]<br>6 . 67  ×  10 [−] [11] ×  1 . 99  ×  10 [30] 2  ×  1<br>∆ v B = 1  −<br>� 1 . 52  ×  1 . 49  ×  10 [11] � � 1 + 1 . 52 �<br>= 2642 m s [−] [1][ �] = 2 . 64 km s [−] [1][�]<br>∆ v = 2 . 94 + 2 . 64<br>= 5 . 58 km s [−] [1]<br>**----- End of picture text -----**<br>


[The ∆ _v_ calculated here would be for the spacecraft to enter a circular orbit around the Sun at the same distance as Mars, but this would not constitute landing - the extra ∆ _v_ to get to the Martian surface increases the total for the whole transfer to roughly 8 km s _[−]_[1] (same as to get into LEO). Other, more complicated routes can be taken, some of which offer substantial efficiencies so the ∆ _v_ for the trip can be much lower (although the time taken to complete the manoeuvre will be longer, and the ∆ _v_ to reach LEO will still be the biggest single step)] 

## **GPT 4V’s output:** 

To calculate the ∆ _v_ to go from Earth Low Earth Orbit (LEO) to Mars, we need to consider two main maneuvers: the transfer from Earth’s orbit to the transfer orbit (Hohmann transfer orbit), and then from the transfer orbit to Mars’ orbit. 1. The first part of the ∆ _v_ , denoted as ∆ _vA_ , is the change in velocity needed to move from Earth’s orbit to the transfer orbit. This is the velocity at perihelion of the transfer orbit minus the Earth’s orbital velocity. 2. The second part, ∆ _vB_ , is the change in velocity needed to move from the transfer orbit to Mars’ orbit. This is Mars’ orbital velocity minus the velocity at aphelion of the transfer orbit. 

The Hohmann transfer orbit is an elliptical orbit that touches Earth’s orbit at its closest point to the Sun (perihelion) and Mars’ orbit at its farthest point from the Sun (aphelion). The semi-major axis of the transfer orbit ( _at_ ) is the average of the Earth’s orbital radius ( _rE_ ) and Mars’ orbital radius ( _rM_ ). Given that 1 AU (Astronomical Unit) is approximately 1 _._ 496 _×_ 10[8] km, we have: 

|Unit) is approximately1_._496_×_108 km, we have:|Unit) is approximately1_._496_×_108 km, we have:|Unit) is approximately1_._496_×_108 km, we have:|Unit) is approximately1_._496_×_108 km, we have:||||||
|---|---|---|---|---|---|---|---|---|
|_rE_ = 1_×_1_._496_×_108 km (Earth’s orbit radius)||||_rM_||= 1_._52_×_1_._496_×_108 km (Mars’ orbit radius)|||
|_at_ = _rE_ +_rM_<br>2|||||||||
|The orbital velocity of an object is given by_v_orb =|||�||_GM_<br>_r_||, where_G_is the gravitational constant and_M_ is the mass of the central object (in this case, the||
|Sun).|||||||||
|The velocity at perihelion of the transfer orbit (_vp,t_) is<br>_vp,t_ =<br>�<br>_GM_<br>_rE_<br>_×_<br>�<br>2_rM_<br>_rE_+_rM_|||||given||by:||
|The velocity at aphelion of the transfer orbit (_va,t_)<br>_va,t_ =<br>�<br>_GM_<br>_rM ×_<br>�<br>2_rE_<br>_rE_+_rM_|||is|given by:|||||
|The Earth’s orbital velocity (_vE_) is:|||||||||
|_vE_ =<br>�<br>_GM_<br>_rE_|||||||||
|Mars’ orbital velocity (_vM_) is:|||||||||
|_vM_ =<br>�<br>_GM_<br>_rM_|||||||||
|Now we can calculate∆_vA_ and∆_vB_:|||||||||
|∆_vA_ = _vp,t −vE_ ∆_vB_ = _vM −va,t_|||||||||
|And the total∆_v_is:|||||||||
|∆_v_ = _|_∆_vA|_+_|_∆_vB|_|||||||||
|Let’s calculate these values step by step. We|will use the||||||gravitational constant _G_ = 6_._674 _×_ 10_−_20 km3kg_−_1s_−_2 and the mass of the Sun||
|_M_ = 1_._989_×_1030 kg.|||||||||
|First, we calculate the semi-major axis of the transfer orbit:|||||||||
|_at_ = 1_×_1_._496_×_108+1_._52_×_1_._496_×_108<br>2|||||||||
|Now, we calculate the velocities:|||||||||
|_vp,t_ =<br>�<br>6_._674_×_10_−_20_×_1_._989_×_1030<br>1_._496_×_108|_×_|�||2_×_1_._52_×_1_._496_×_108<br>1_._496_×_108+1_._52_×_1_._496_×_108|||||
|_va,t_ =<br>�<br>6_._674_×_10_−_20_×_1_._989_×_1030<br>1_._52_×_1_._496_×_108|_×_|�||2_×_1_._496_×_108<br>1_._496_×_108+1_._52_×_1_._496_×_108|||||
|_vE_ =<br>�<br>6_._674_×_10_−_20_×_1_._989_×_1030<br>1_._496_×_108|||||||||
|_vM_ =<br>�<br>6_._674_×_10_−_20_×_1_._989_×_1030<br>1_._52_×_1_._496_×_108|||||||||
|Now we can calculate∆_vA_ and∆_vB_:|||||||||
|∆_vA_ = _vp,t −vE_|||||||||
|∆_vB_ = _vM −va,t_<br>And fnally, the total∆_v_:|||||||||
|∆_v_ = _|_∆_vA|_+_|_∆_vB|_|||||||||
|Let’s calculate these values.|||||||||



Figure 24: An example of an astronomy problem with an incomplete response. 

42 

## Computer Science - Understanding Error 

## **Problem:** 

A girl named Xenia has a cupboard that looks like an arc from ahead. The arc is made of a semicircle with radius r (the cupboard’s top) and two walls of height h (the cupboard’s sides). The cupboard’s depth is r, that is, it looks like a rectangle with base r and height h+r from the sides. The figure below shows what the cupboard looks like (the front view is on the left, the side view is on the right). Xenia got lots of balloons for her birthday. The girl hates the mess, so she wants to store the balloons in the cupboard. Luckily, each balloon is a sphere with radius. Help Xenia calculate the maximum number of balloons she can put in her cupboard. [figure1] You can say that a balloon is in the cupboard if you can’t see any part of the balloon on the left or right view. The balloons in the cupboard can touch each other. It is not allowed to squeeze the balloons or deform them in any way. You can assume that the cupboard’s walls are negligibly thin. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI_images/nips_2024_OlympicArena_Benchmarking_Multi-discipline_Cognitive_Reasoning_for_Superintelligent_AI.pdf-0043-03.png)


## [figure1] 

Input The single line contains two integers r, h (1 <= r, h <= 10[7] ) Output Print a single integer — the maximum number of balloons Xenia can put in the cupboard. Examples Input 1 1 Output 3 Input 1 2 Output 5 Input 2 1 Output 2 time limit: 2 seconds memory limit: 256 megabytes 

## **GPT 4V’s output:** 

|**GPT 4V’s output:**|**GPT 4V’s output:**|**GPT 4V’s output:**||||||||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**import** math||||||||||||||||||||||||||
|_# _|_Function_<br>_to_<br>_c a l c u l a t e_<br>_the _||_maximum number_<br>_of_||||_ba l_||_lo on_|||_s_||||||||||||||
|**def**||max_balloons ( r ,<br>h ) :||||||||||||||||||||||||
|||_#_<br>_C a l c u l a t e_<br>_the_<br>_number_|_of_<br>_ba ll o on_|_s_|_t h_|_a t_<br>_f_|_i t_|_in_|||_the_||_h e i g h t_|_of_|_the_||_walls_|||||||||
|||b a l l o o n s _ i n _ h e i g h t = (2|* h )<br>/ /<br>r|||||||||||||||||||||||
|||_#_<br>_C a l c u l a t e_<br>_the_<br>_number_|_of_<br>_ba ll o on_|_s_|_t h_|_a t_<br>_f_|_i t_|_in_|||_the_||_s e m i c i r_|_c l e_||_on _|_top_|||||||||
|||_# The_<br>_diameter_<br>_of_<br>_the_<br>_s_|_e m i c i r c l e_|_i s_|_r _|_,_<br>_so_|_i t_||_can_|||_f_|_i t_<br>_one_<br>_balloon_||||_of_||_radius_||_r /2_|||||
|||_# The_<br>_remaining_<br>_h e i g h t_|_in_<br>_the_<br>_s e m i c i r_|||_c l e_|_i s_|_r_|_/_|_2 _|_,_|_which can_||_f i_|_t_|_another_||||_balloon_||||||
|||b a l l o o n s _ i n _ s e m i c i r c l e|= 1 + ( h *|2|+|r )|/ /|r||||||||||||||||||
|||_#_<br>_C a l c u l a t e_<br>_the_<br>_number_|_of_<br>_a d d i t i o_|_n a_|_l_|_ba l lo on_||_s_||_t h a_||_t_|_can_<br>_f i t_|_in_||_the_||_corners_|||_of_|_the_||_s e m i c i r c l e_||
|||_# This_<br>_i s_<br>_the_<br>_number_<br>_of_|_f u l l_<br>_r /2_|_r a_|_d i_|_i_<br>_t h_|_a t_|_f_|_i_|_t_|_i_|_n_|_t o_<br>_the_<br>_remaining_|||||_arc_||_l e n_|_g t h_||_a f t e r_||_placing_|
|||_# the_<br>_f i r s t_<br>_balloon_<br>_in_|_the_<br>_s e m i c i_|_r c_|_l e_|||||||||||||||||||||
|||r e m a i n i n g _ a r c _ l e n g t h =|( math . pi *|r|/|2)|−(|r||/|2)|||||||||||||||
|||a d d i t i o n a l _ b a l l o o n s = **i n t**( r e m a i n i||n g|_ a r c _ l||e n g|t h|||/ /||( r<br>/<br>2 ) )|||||||||||||
|||_# The_<br>_t o t a l_<br>_number_<br>_of_<br>_ba ll o on s_<br>_i s_||_the _||_sum _|_of_||_the_|||_above_<br>_c a l_||_c u l_|_a_|_t e d_||_values_||||||||
|||t o t a l _ b a l l o o n s = b a l l o o|n s _ i n _ h e i g|h t|+|b a l|l o o|n|s|_|i n|_|s e m i c i r c|l e|+|a d|d i t||i o n a l _||b a l|l o o n s||||
|||**return**<br>t o t a l _ b a l l o o n s||||||||||||||||||||||||
|_# _|_Read_<br>_i n p u t_|||||||||||||||||||||||||
|r ,||h = **map**(**int** ,<br>**input** ( ) . s p l i t ( ) )||||||||||||||||||||||||
|_#_|_C a l c u l a t e_<br>_and_<br>_p r i n t_<br>_the_<br>_r_||_e s u l t_|||||||||||||||||||||||
|**print**( max_balloons ( r ,<br>h ) )||||||||||||||||||||||||||
|This program defnes a function "max balloons" that calculates the maximum number of balloons that can ft in the cupboard based on the given dimensions. It<br>then reads the input values for "r" and "h", calls the function with these values, and prints the result. The program takes into account the balloons that ft in the<br>height of the walls, the semicircle on top, and additional balloons that can ft in the corners of the semicircle.||||||||||||||||||||||||||



Figure 25: An example of a programming problem with an understanding error. 

43 

## **G Consideration for Social Impact** 

Certainly, it is essential to point out that as AI performs increasingly well on our benchmark, potentially even surpassing human capabilities, there are some potential ethical and moral risks that require collective oversight. 

## **H Limitations and Future Work** 

Despite the value of this benchmark, there remains work to be done in the future. Firstly, our benchmark inevitably introduces some noisy problems, we will actively utilize community feedback to continuously refine it. Additionally, we aim to release new versions of the benchmark annually to mitigate issues related to data leakage. Moreover, this benchmark is currently limited to evaluating models’ abilities to solve complex problems. In the future, we aspire for AI to assist with complex tasks and demonstrate value in real-world applications such as AI4Science and AI4Engineering rather than just problem-solving. This will be the goal of our future benchmark designs for evaluating AI capabilities. Nonetheless, at present, OlympicArena plays an essential role as a catalyst for further advancements. 

44 

