# **SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models** 

**Dan Zhang**[1] _[,]_[2] _[,][∗]_ **, Ziniu Hu**[3] **, Sining Zhoubian**[1] _[,]_[2] _[,][∗]_ **, Zhengxiao Du**[1] _[,]_[2] _[,][∗]_ **Kaiyu Yang**[3] **, Zihan Wang**[1] _[,]_[2] _[,][∗]_ **, Yisong Yue**[3] **, Yuxiao Dong**[1] **, Jie Tang**[1] _[†]_ 1The Knowledge Engineering Group (KEG), Tsinghua University; 2Zhipu AI; 

3California Institute of Technology 

https://SciGLM.github.io/ 

## **Abstract** 

Large Language Models (LLMs) have shown promise in assisting scientific discovery. However, such applications are currently limited by LLMs’ deficiencies in understanding intricate scientific concepts, deriving symbolic equations, and solving advanced numerical calculations. To bridge these gaps, we introduce SciInstruct, a suite of scientific instructions for training scientific language models capable of college-level scientific reasoning. Central to our approach is a novel self-reflective instruction annotation framework to address the data scarcity challenge in the science domain. This framework leverages existing LLMs to generate step-by-step reasoning for unlabelled scientific questions, followed by a process of self-reflective critic-and-revise. Applying this framework, we curated a diverse and high-quality dataset encompassing physics, chemistry, math, and formal proofs. We analyze the curated SciInstruct from multiple interesting perspectives (e.g., domain, scale, source, question type, answer length, etc.). To verify the effectiveness of SciInstruct, we fine-tuned different language models with SciInstruct, i.e., ChatGLM3 (6B and 32B), Llama3-8B-Instruct, and Mistral-7B: MetaMath, enhancing their scientific and mathematical reasoning capabilities, without sacrificing the language understanding capabilities of the base model. We release all codes and SciInstruct at https://github.com/THUDM/SciGLM. 

## **1 Introduction** 

Large language models (LLMs) have shown potential to assist and accelerate scientific discovery [1; 2], helping tasks like protein prediction [3], weather forecasting [4] and geoscience understanding [5]. Despite these promising proof-of-concept trials, recent studies [6; 7; 8; 9] show that even advanced LLMs like GPT-3.5 and GPT-4 struggle with basic scientific problems, achieving only 28.52% accuracy on some college-level textbook questions. These scientific questions, such as calculating energy with the Planck distribution, require a diverse set of skills, including finding the correct combination of physical concepts and axioms, choice and deduction of formal equations, and rigorous numerical computing. Before letting LLMs equip these skills to solve basic scientific questions, all ambitious visions of building LLM agents to assist scientific discovery could be unreliable. This brings substantial incentives for building scientific instructions and using them to develop foundational scientific language models. 

> *Work done while these authors interned at Zhipu AI. 

> †Corresponding author. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Table 1: Comparison between SciInstruct and SciInstruct and and 60 GPT-4<br>other related instruction datasets. SciGLM<br>(32B)<br>Dataset Domain College 50 SciGLM<br>Math Physics Chemistry Lean Level (6B) ChatGLM3GPT-3.5<br>Galactica [1] � � � � Unknown (32B)<br>MathInstruct [10] � � � � � 40 ChatGLM3<br>WebInstruct [MetaMath [1112]] �� �� �� �� �� (6B) Sci: Mistral (7B)<br>SciInstruct � � � � � 30 ChatGLM2(6B) ChatGLM2(12B)<br>Domain Proportions Question Type Proportions<br>Simple  Mistral (7B) LLaMA-2<br>Lean  Solution    Problem 20 LLaMA-2 (7B) (13B)<br>15.8% Filling 14.6% Galactica<br>ChemistryPhysics  &       48.8% 33.2% 20.1% Complex  Solution  Problem 10 6(6.7B) 12 32 ··· API<br>35.4% Math 32.0% Figure 3: Number of Parameters (Billions, Log Scale) Average accuracy on CEval-Sci, Sci-<br>Choice Question Eval, SciBench, MATH, and SAT-Math bench-<br>Figure 1: Domain Figure 2: Question Type marks of different LLMs.<br>Avg. Accuracy of Scientic Benchmarks<br>**----- End of picture text -----**<br>


Table 1: Comparison between SciInstruct and SciInstruct and and other related instruction datasets. 

However, training LLMs to understand science (e.g., physics, chemistry, math) is much more challenging than many general reasoning tasks, requiring more complicated skills. Therefore, the core of improving LLMs’ scientific problem-solving capabilities is to build large-scale and highquality instruction datasets, which shall cover all the required skills. We thus aggregate data to improve each skill: 1) for scientific concept understanding, we gather a substantial amount of physics and chemistry questions that require basic science knowledge; 2) for numerical calculation, we crawl and utilize additional and more advanced mathematical calculation data; 3) for rigorous deduction of symbolic equations, we incorporate formal theorem proofs written in Lean. Such a mixture of data sources enables the trained model not to overfit to a single subject but to acquire some general and fundamental skills to solve different scientific tasks. 

On the other hand, the scale of available instruction data from the internet for scientific problems is way smaller than other tasks. As scientific content often requires certain expertise to create, and high-quality information is often protected by intellectual property, most data we can legally access only contain question-answer (QA) pairs without detailed chain-of-thought reasoning steps ( _R_ ). However, merely training LLMs on QA pairs will lead to very bad results and even harm their general language capability. To get high-quality reasoning steps ( _R_ ) as instruction, we propose a selfreflective instruction annotation framework that asks LLM to autonomously annotate, critique, and revise reasoning steps, with minimal human intervention. Specifically, LLM first tries to generate both reasoning steps and answer the given question ( _Q_ ) only; then, for those outputs with incorrect answer prediction, we ask LLM itself to identify the error type, based on which to address the error and revise the output, until getting the correct answer. Such a self-reflective annotation framework solely utilizes AI rather than humans to collect reasoning traces ( _R_ ) as instructions, while guaranteeing the quality and addressing the potential mistakes of existing LLM with careful answer checking and LLM self-reflection. 

After consolidating the questions and answers produced by self-reflective annotation, we construct SciInstruct, a comprehensive dataset instruction tuning scientific language models. Figure 1 and Figure 2 present the domain and question type proportions of SciInstruct. Table 1 concludes the key differences between existing datasets and ours. In this work, we choose three LLMs, i.e., the ChatGLM3 [13; 14] (6B and 32B), Llama3-8B-Instruct [15], and Mistral-7B: MetaMATH [16; 11], as our backbone base models. For example, by fine-tuning the ChatGLM series model on SciInstruct, we obtain the SciGLM model. We then evaluate the fine-tuned models through three types of evaluation tasks, including **scientific test sets** , **mathematical benchmarks** , and **general language and coding tasks** , and show the average accuracy on scientific benchmarks of different LLMs in Figure 3. Through instruction tuning, we achieve a 4.87% improvement over the 6B model, and 2.67% improvement over the 32B model, outperforming many previous state-of-the-art models with the same parameter size, including Galactica [1] for science problems, and MAmmoTH [10] for math problems. We also show tuning our instruction datasets does not sacrifice general language understanding capabilities, making SciGLM a good suite of scientific language models for both human-AI communication as well as scientific domain-knowledge expertise. 

We highlight our contributions as follows: 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0003-00.png)


**----- Start of picture text -----**<br>
SciInstruct<br>How to determine the harmonic Physics QA dataset Bootstrapped Human Conclusion<br>vibration frequency of a single degree of freedom system? Problem 1 Problem 1CoT 1 Comprehension<br>Try to calculate the boiling point of water at 670kPa external pressure.Chemistry Problem nAnswer 1Answer n······ LLMs Problem nAnswer 1······CoT n AnnotateLabeledPositiveSampleDataData LMs Calculation mistakesmistakes<br>Math Answer n<br>The number of asymptotes of the  Instruction False reasoning<br>curve y=\\frac{x^2+x}{x^2-1} is___. Self-Reflective Annotation Negative Data -quality<br>Formal Proofs (Lean) CoT Critic & Revise Sample Classifier Filter Stage<br>theorem const_add_term[add_comm_monoid α] (k n x a {α}  w/o RQA Filter negatives<br>a') (h : k + a = a') :\n k + @term  Reasoning steps (R) Retain positives<br>α _ n x a = term n x a' :=<br>Collection Annotation Filtering<br>**----- End of picture text -----**<br>


Figure 4: **The pipeline of constructing SciInstruct** . On the far left is a mix of training datasets. The purpose of the annotation is to supplement chain-of-thought processes with reflective generation. The goal of the filter is to train an instruction-quality classifier and only keep high-quality reasoning traces as instructions. 

- From the **data** perspective, we construct SciInstruct, a comprehensive scientific instruction tuning dataset that includes physics, chemistry problems, math, and formal proofs. 

- From the **method** perspective, we propose a self-reflective annotation pipeline for LLMs to autonomously curate instruction datasets. 

- From the **model** perspective, to verify the effectiveness of our SciInstruct, we finetune different LLMs (the ChatGLM3 series model, Llama3-8B-Instruct, and Mistral-7B: MetaMATH) on SciInstruct and show performance improvements on various scientific and mathematical benchmarks, without sacrificing general language understanding tasks. 

## **2 SciInstruct** 

Many research [17; 10; 12] have shown that fine-tuning pre-trained LLMs on high-quality CoT reasoning data can gain performance improvement by enabling the model to better utilize the knowledge memorized through pre-training, follow more accurate and human-readable reasoning styles and language formats. However, the main challenges of constructing scientific instructions include the knowledge and complexity required and the smaller scale of available data. Therefore, we seek to tackle these obstacles by creating SciInstruct to enhance the LLMs’ scientific problemsolving capabilities. Figure 4 illustrates our meticulous design of essential sub-modules aimed at gathering large-scale, high-quality instructions. These critical sub-modules encompass self-reflective instruction annotation and noisy instruction filtering. SciInstruct comprises a total of **254,051** verified instructions. 

## **2.1 Diverse Instruction Collection** 

Our objective is to build a comprehensive and diverse dataset that encompasses scientific knowledge in terms of depth, wide coverage, and diversity. To achieve this, we will focus on scientific fields and curate several top-tier datasets that are extensively utilized and cover a wide range of scientific disciplines, such as physics, chemistry problems, mathematics, and formal proofs. To initiate the process, we collect questions from a variety of sources, including textbooks, pedagogical materials, and problem sets. 

**Instruction Subject.** As show on the left side of Figure 4, we create data from the following subjects: 

_•_ **Physics.** This subject aims to address the challenge of processing complex physical problems with step-by-step solutions and assessing the ability of LLMs to comprehend and analyze physics problems. Public training datasets such as Fundamentals of Physics and Physical Chemistry are observed to lack college-level physics knowledge. To address this gap, we collected a large set of physical questions from a wide array of subjects (e.g., dynamics, quantum physics, electrodynamics, etc.) from typical physical textbooks, along with a comprehensive dictionary of physics problems 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0004-00.png)


**----- Start of picture text -----**<br>
[Prompt]  On a ray of simple harmonics, the vibration phase difference between two points<br>0.2\\mathrm{~m} is \\pi/6. It is also known that the vibration period is 0.4\\mathrm{~s}, then the<br>wavelength is, and the wave speed is? All Questions<br>LLM 42,497<br>Chain-of-Thought<br>[Sampled Output]  The phase difference Δφ = π/6, the distance between<br>[Prompt 1]  The following input consists of a  two points Δx = 0.2m, and the vibration period T = 0.4s are known. According to the phase difference and the relationship between the wavelength: Δφ = 2 PI<br>science problem, please generate an elaborate  (Δx/lambda), we have wavelength lambda = 1.2m. Then according to the  Generate Label=1 Correct Solutions<br>step-by-step  solution to the problem. relationship between wavelength λ and period T and wave speed v: v = λ/T, the known value is substituted into the above formula: v = 1.2/0.4 = 3m/s. The  Stage 1 Left Label=0 19,824 Stage 1<br>wave speed is 3m/s. 22,673<br>Reflective Generation LLM<br>[Prompt 2]  The following input consists of a  [Reflection] phase difference and wavelength used in the calculation of wavelength is  The error in the solution is that the relationship between  Generate Label=1 Correct Solutions<br>science problem and a corresponding solution. However, this solution is incorrect, please  incorrect. The correct relation should be Δφ = 2PI (Δx/lambda).  [Answer] First, according to the formula ∆phi = 2 PI (∆x/lambda), we  Stage 2 Left Label=0 5,458 Stage 2<br>reflect on its errors  and then generate a  know that lambda = 2 PI (0.2)/(PI / 6) = 1.2 m. Next, known vibration period T = 0.4s, according to the definition of wave speed v formula v = lambda/T, we  17,215<br>correct step-by-step solution to the problem. have v = 1.2m / 0.4s = 3m/s. In summary, the wavelength is 1.2m and the<br>wave speed is 3m/s.<br>Generate Correct Solutions<br>Label=1<br>Prompt Answer LLM 7,687 Stage 3<br>Stage 3 Left Label=0<br>[Prompt 3]  The following input consists of a  [Reflection]  The error in the solution is that the ratio of the phase  9,528<br>science problem, a corresponding difference π/6 to the full wave phase 2π, rather than half wave phase π, i.e. π/6 ÷ 2π, should be used to determine the ratio of the distance between two<br>solution and  the real answer . The given  points to the wavelength. Total Correct Solutions<br>solution is incorrect, please reflect [Answer] two points ΔxFirst,= 0.2knownm, we canphaseusedifferencethe formulaΔphi=PIΔphi =/ 6,2 PI (Δx/lambda)distance betweento 32,969<br>on its errors and then generate a correct step-by-step solution to the problem based on the  solve the wavelength lambda.=to the definition of wave speed v formula v = λ/T, we obtain v = λ/ T = 2.4m(2 x 0.2)/(1/6) = 2.4 m. Next,We can know that lambda = (2 PI (0.2)/(PI / 6)known vibration period T = 0.4s, according<br>real answer. / 0.4s = 6m/s. To sum up, the wavelength is 2.4m and the wave speed is 6m/s.<br>**----- End of picture text -----**<br>


Figure 5: The workflow and quantity of data for self-reflective instruction annotation framework. 

and solutions. Note that most of these physics questions only contain a single answer without a step-by-step solution. 

_•_ **Chemistry.** We gathered questions from various chemistry subjects, including college-level General Chemistry, Analytical Chemistry, Biochemistry, and Inorganic Chemistry. Unlike Physics, these Chemistry problems emphasize more on knowledge concepts and require less calculation. 

_•_ **Math.** College-level mathematical tasks are characterized by their intricate logic and high complexity. We have gathered math problems encompassing various mathematical subjects such as Calculus, Algebra, and Advanced Mathematics. These problems come in different formats, including multi-choice questions, calculations, and complex problem-solving scenarios. Our collection process involves sourcing problems from public Q&A websites, problem sets, and textbooks. 

_•_ **Formal Proofs (Lean).** SciInstruct also includes data from formal mathematics. In formal math, theorems and proofs are expressed in formal logic instead of natural language. They look like computer programs. For example, “theorem gcd_self (n : Nat) : gcd n n = n” is a theorem that says: For any natural number n, the greatest common divisor of n and itself is n. And “cases n <;> simp [gcd, mod_self]” is a proof of this theorem. There are many languages for writing formal theorems and proofs; examples include Coq [18], Isabelle [19], and Lean [20]. We chose to include data from Lean, as it offers a vast math library and is currently the most popular in the math community. Therefore, it has abundant theorems and proofs written by mathematicians, covering diverse topics in graduate-level mathematics, such as analysis, geometry, and algebra. Specifically, we process the data from LeanDojo [21] and format it to align with the successive deduction process, ensuring its relevance and applicability to the model’s training for mathematical reasoning tasks in natural language. This preprocessing step helps bridge the gap between the Lean dataset’s nature and the model’s expected learning signals. Finally, we obtained 40,248 instructions for theorem proof. Like Appendix A.3, we form an instruction for each theorem and its proof. 

**Multi-lingual Instruction.** To enhance the overall quality and effectiveness of the curated dataset, we also translate the default Chinese questions into English. We found that LLMs tend to generate correct solutions after translating these problems into English for some Chinese problems that do not obtain correct solutions. This improved performance is likely due to the higher-quality English corpus used during the pre-training of LLMs. Therefore, we have embraced this strategy to construct SciInstruct for Chinese questions. 

**Summary.** In total, we gathered 257,143 raw questions, the majority of which lacked step-by-step reasoning steps. We aim to supplement these through a self-reflective annotation process. 

4 

Table 2: **Ablation study of filter step** . We arranged the samples by score and excluded the lowest 10%. The resulting numbers represent the average weighted accuracy on evaluation tasks. 

|Type|Ave. Sci|Ave. Math|Ave. {Sci+Math}|
|---|---|---|---|
|Unfltered|43.57|48.50|46.03|
|Filtered|43.85|49.24|46.54|



Table 3: Statistics of SciInstruct across subjects. 

|ubjects.|||
|---|---|---|
|Subject|# Number|Proportion|
|Physics & Chemistry|123,869|48.76%|
|Math|89,934|35.40%|
|Formal Proofs (Lean)|40,248|15.84%|
|Total|254,051||



## **2.2 Self-Reflective Instruction Annotation** 

Given a language model _π_ to answer question _Q_ , recent studies [22] have shown that by first forcing it to generate step-by-step reasoning steps ( _R_ ) first, the overall performance for correctly generating answer _A_ can be significantly improved, via: _Pπ_ ( _A | Q_ ) = E _R∼Pπ_ ( _R|Q_ )� _P_ ( _A | R, Q_ )�. This is why many instruction datasets aim to collect high-quality intermediate solutions to train LLMs generating correct step-by-step solutions. The key challenge for the science domain, as we state above, is that most of the QA pairs we collect do not contain ground-truth reasoning paths ( _R_ ). Getting the correct intermediate reasoning _R_ given QA can be regarded as a discrete latent variable inference problem via posterior sampling. However, in practice, we cannot afford to sample all possible _R_ from LLM. Here we adopt a simplified strategy for solving it: 1) Utilizing a powerful LLM (we use GPT-4-0613), we sample each multiple times for each question, recording its reasoning traces as well as the predicted answer; 2) We filter out only the traces with a correct predicted answer, by assuming only with correct traces, LLM can get a correct answer. 

**LLM Self-Reflection.** However, even GPT-4 cannot consistently produce accurate answers after multiple trials, and the above procedure can only collect _R_ for a portion of questions. Drawing on prior research demonstrating the capacity of language models for self-correction [23; 24], we refine the CoT process using a self-reflective framework, as depicted in the middle of Figure 4. The final reflective generation process entails three stages. Initially, we employ a simple CoT prompt (Prompt 1) to obtain step-by-step solutions for each question. To obtain an accurate assessment of reasoning results, we employ a GPT-4 labeling method based on an outcome reward model (ORM) [25] as a basic implementation in our work, rather than the expensive process reward model (PRM) [25] that typically requires manual annotation, especially for complex scientific reasoning. We filter out incorrect solutions by applying the GPT-4 labeling method, resulting in 19,824 correct solutions. Subsequently, the incorrect solutions from stage one and their respective questions enter stage two. Here, a reflective prompt (Prompt 2) assists GPT-4 in generating correct answers by analyzing its errors from stage one. The newly generated solutions are then filtered again by GPT-4, and the undesirable ones proceed to stage three. In stage three, based on Prompt 2, we incorporate the real answer as a direct hint in the prompt (Prompt 3) to further aid in question resolution. The final segment of correct solutions is obtained after the generation and filtering process. We illustrate the reflective generation in Figure 5 and quantify the amount of data generated by each process. 

## **2.3 Filtering Out Low-Quality Instruction** 

Though the above procedures give us the annotated reasoning steps ( _R_ ), not all of them are correct. The error can come from two sources: 1) though LLM generates a correct answer, the intermediate reasoning can still be wrong [26]; 2) the question and ground-truth solutions transformed via Optical character recognition (OCR) may be incomplete and unable to be successfully compiled. Therefore, we propose another step to train an instruction-quality classifier and filter out low-quality instructions. 

**Quality Data Synthesis.** We randomly selected a subset of questions from our labeled dataset of 11,553 questions as positive samples. To generate negative samples, we prompted ChatGLM2-6B, GPT-3.5-turbo-0613, and GPT-4-0613 to provide step-by-step answers to selected questions. We filtered inaccurate answers from ChatGLM2-6B and labeled the solutions from GPT-3.5-turbo and GPT-4 using a formatted prompt method demonstrated in Figure 11 in Appendix A.4. These solutions were merged with annotated solutions from the original dataset to train our classifier. The composition of the merged dataset is detailed in Table 7 in Appendix A.5. 

5 

**Instruction-quality Classifier.** We improved dataset quality by training an instruction-quality classifier based on Table 7 using ChatGLM3-6B-Base features. Using these data, we train an instruction-quality classifier via the feature pre-extracted by a ChatGLM3-6B-Base model. The classifier outputs a logit ranging from -1 to 1, with higher scores indicating more reliable answers. This logit is used to rank and select high-quality data from the noisy dataset. Table 2 demonstrates the effectiveness of supervised fine-tuning on both filtered and unfiltered datasets at a 6B scale, showing that models trained on a filtered dataset perform better. 

**Error Analysis.** As our classifier filter is trained on labeled datasets and generated solutions, errors in negatively labeled solutions from ChatGLM2-6B, GPT-3.5-turbo, and GPT-4 can significantly impact the classifier’s performance. Therefore, we conduct an error analysis and categorize them into Comprehensive mistakes, Calculation mistakes, and False reasoning. This analysis is detailed in Figure 12 in A.6, demonstrating the classifier’s capacity to recognize these errors in the dataset. 

**Summary.** Based on the aforementioned key sub-modules, we have constructed the SciInstruct dataset, which comprises 254,051 instructions, as illustrated in Table 3. 

## **2.4 Instruction-Tuning with SciInstruct** 

As our foundational model, we choose three LLMs, i.e., ChatGLM3 (6B and 32B), Llama3-8BInstruct, and Mistral-7B: MetaMATH. After establishing the base model, we have standardized all data into a chatbot-style format. Subsequently, we have fine-tuned the foundational model using the SciInstruct, enabling us to validate our constructed SciInstruct. Throughout the fine-tuning process, we have conducted experiments using the Huggingface transformers library. For both the 6B and 32B models, we have utilized a learning rate of 3e-6, employed a linear scheduler, and trained for two epochs. To efficiently train the model, we have leveraged DeepSpeed [27] training. 

Table 4: **Results on scientific reasoning tasks** . Experiments indicate that fine-tuning on SciInstruct consistently outperforms the base model across various parameter scales. Avg. Sci represents the weighted average score of all scientific tasks within the same evaluation category, while Avg. {Sci+Math} signifies the weighted average score on both scientific and mathematical tasks. Within each parameter setup, **Bold** spotlights the one with best performance, and Underline denotes the second-best result. Results marked as _[†]_ are benchmarked by ourselves. 

|Model<br>CEval-Hard<br>C|Eval-Sci<br>MMLU-Sci<br>SciEval<br>SciBench<br>GPQA_Diamond<br>Avg. Sci<br>Avg. {Sci+Math}|
|---|---|
||(API, parameter details unknown)|
|GPT-4<br>54.96<br>GPT-3.5-turbo<br>41.37<br>Claude-v1.3<br>39.14|60.55<br>-<br>**73.93**<br>**28.52**<br>39.70<br>-<br>-<br>46.83<br>-<br>66.97<br>12.17<br>-<br>-<br>-<br>44.64<br>-<br>63.45<br>-<br>-<br>-<br>-|
||(# parameter = 6B_∼_7B)|
|LLaMA-2-7B<br>28.29_†_<br>Galactica-6.7B<br>11.84_†_<br>ChatGLM2-6B<br>29.61_†_<br>ChatGLM2-6B-Base<br>32.90_†_<br>ChatGLM3-6B<br>36.84_†_<br>ChatGLM3-6B-Base<br>45.40<br>_†_<br>**SciGLM**(ChatGLM3-6B-Base)<br>**51.97**<br>Llama3-8B-Instruct (zero-shot)<br>26.32_†_<br>Llama3-8B-Instruct (few-shot)<br>25.66_†_<br>+**SciInstruct**<br>32.24<br>Mistral-7B: MetaMATH (zero-shot)<br>9.87_†_<br>Mistral-7B: MetaMATH (few-shot)<br>9.21_†_<br>+**SciInstruct**<br>30.92|30.00_†_<br>30.41<br>28.37<br>0.40<br>-<br>-<br>-<br>11.44_†_<br>30.68<br>50.87<br>-<br>-<br>-<br>-<br>45.71_†_<br>37.09_†_<br>53.02_†_<br>1.54_†_<br>-<br>-<br>-<br>40.95_†_<br>38.06_†_<br>50.38_†_<br>1.20_†_<br>-<br>-<br>-<br>38.57_†_<br>41.78_†_<br>56.56_†_<br>2.40_†_<br>28.70<br>34.14<br>29.73<br>54.29<br>_†_<br>40.16_†_<br>61.69_†_<br>2.40_†_<br>24.75<br>38.12<br>40.34<br>**60.00**<br>45.34<br>62.09<br>**3.77**<br>25.25<br>**41.40**<br>**45.32**<br>27.62_†_<br>26.90_†_<br>**71.38**_†_<br>1.03_†_<br>27.27_†_<br>30.09<br>28.58<br>23.33_†_<br>**52.67**_†_<br>71.38_†_<br>3.60_†_<br>31.31_†_<br>34.66<br>37.92<br>34.76<br>40.86<br>66.47<br>3.60<br>29.29<br>34.54<br>36.04<br>8.57_†_<br>28.25_†_<br>63.61_†_<br>4.63_†_<br>27.78_†_<br>23.79<br>25.57<br>19.52_†_<br>44.74_†_<br>63.61_†_<br>6.17_†_<br>29.29_†_<br>28.76<br>33.92<br>38.10<br>42.16<br>64.16<br>6.23<br>27.27<br>34.81<br>37.91|
||(# parameter = 12B_∼_13B)|
|LLaMA-2-13B<br>19.74_†_<br>Vicuna-13B<br>-|19.05_†_<br>35.85<br>36.96<br>1.37<br>26.20<br>22.59<br>22.13<br>-<br>32.13<br>53.93<br>-<br>-<br>-<br>-|
||(# parameter = 30B_∼_32B)|
|Galactica-30B<br>-<br>ChatGLM3-32B-Base<br>53.95_†_<br>**SciGLM**(ChatGLM3-32B-Base)<br>**56.58**|-<br>35.53<br>54.96<br>-<br>-<br>-<br>-<br>64.29_†_<br>**50.30**_†_<br>67.38_†_<br>4.29_†_<br>22.22<br>43.74<br>48.62<br>**66.19**<br>49.38<br>**69.84**<br>**5.15**<br>**25.76**<br>**45.48**<br>**51.47**|



## **3 Benchmark on SciInstruct** 

## **3.1 Experimental Setup** 

**Scientific and Mathematical Tasks.** The evaluation tasks are summarized in Table 8 in Appendix A.7. These tasks have been chosen as out-of-domain benchmark datasets, encompassing CEval-Hard [28], 

6 

Table 5: **Results on mathematical reasoning** . The Avg. Math represents the weighted average score of all mathematical tasks within the evaluation category. Results marked as _[†]_ are benchmarked by ourselves. 

|Model|GSM8K|MATH|Mathematics|SAT-Math|MMLU-Math|CEval-Math|Avg. Math|
|---|---|---|---|---|---|---|---|
|||(API, parameter details unknown)||||||
|GPT-4|92.00|42.50|-|95.00|-|53.91|-|
|GPT-3.5-turbo|80.80|34.10|-|70.90|-|40.81|-|
|Claude-v1.3|-|-|-|-|-|37.66|-|
|||(# parameter = 6B_∼_7B)||||||
|LLaMA-2-7B|14.60|2.50|6.00|26.80|29.80|30.00_†_|18.28|
|Galactica-6.7B|10.20|2.20|4.60|17.50|28.00|14.48_†_|-|
|WizardMath-7B|54.90|10.70|9.30|25.40|31.10|-|-|
|MAmmoTH (CoT)-7B|50.50|10.40|9.20|32.70|39.90|-|-|
|MetaMath-7B|66.50|19.80|-|-|-|-|-|
|MAmmoTH & MetaMath-7B|66.30|24.10|18.30|41.40|44.40|-|-|
|ChatGLM2-6B|25.85|6.90_†_|14.30_†_|39.55_†_|38.91_†_|36.67_†_|27.03|
|ChatGLM2-6B-Base|31.54|7.84_†_|17.10_†_|34.55_†_|40.45_†_|32.22_†_|27.28|
|ChatGLM3-6B|29.05|9.92_†_|11.60_†_|39.09_†_|41.07_†_|21.11_†_|25.31|
|ChatGLM3-6B-Base|72.93|25.38_†_|29.30_†_|55.91_†_|31.83_†_|40.00_†_|42.56|
|**SciGLM**(ChatGLM3-6B-Base)|73.62|25.18|31.80|65.46|49.38|50.00|**49.24**|
|Llama3-8B-Instruct (zero-shot)|7.35_†_|20.76_†_|4.50_†_|58.64_†_|54.52_†_|16.67_†_|27.07|
|Llama3-8B-Instruct (few-shot)|63.38_†_|30.00_†_|22.30_†_|57.27_†_|55.24_†_|18.89_†_|41.18|
|+**SciInstruct**|63.76|31.40|22.60|43.64|42.81|21.11|37.55|
|Mistral-7B: MetaMATH (zero-shot)|76.12_†_|29.34_†_|23.90_†_|15.45_†_|10.37_†_|8.89_†_|27.35|
|Mistral-7B: MetaMATH (few-shot)|72.33_†_|28.28_†_|24.40_†_|50.45_†_|42.40_†_|16.67_†_|39.09|
|+**SciInstruct**|76.65|30.30|25.00|43.82|41.48|28.89|41.02|
|||(# parameter = 12B_∼_13B)||||||
|LLaMA-2-13B|28.70|3.90|11.50|32.70|34.40|18.89_†_|21.68|
|Vicuna-13B|28.40|5.80|10.00|34.00|34.10|-|-|
|WizardMath-13B|63.90|14.00|14.10|24.50|32.10|-|-|
|MAmmoTH (CoT)-13B|56.30|12.90|11.70|43.60|42.30|-|-|
|MAmmoTH & MetaMath-13B|71.04|26.18|20.60|48.18|48.25|-|-|
|||(# parameter = 30B_∼_32B)||||||
|Galactica-30B|41.70|12.70|11.80|37.70|37.90|-|-|
|ChatGLM3-32B-Base|81.80|31.60_†_|38.60_†_|67.73_†_|50.10_†_|51.11_†_|53.49|
|**SciGLM**(ChatGLM3-32B-Base)|83.70|32.86|35.00|76.36|61.29|55.56|**57.46**|



CEval-Sci [28], MMLU-Sci [29], SciEval [6], SciBench [7], GPGQ [8], GSM8K [30], MATH [31], Mathematics [32], SAT-Math [33], MMLU-Math [29] from MathInstruction, and CEval-MATH [28]. These evaluation datasets cover a broad range of subjects, including physics, chemistry, and math. 

**General Tasks.** Furthermore, we assess the generalization abilities of tasks across various scales when fine-tuning models. These tasks include assessing knowledge abilities (MMLU [29] and CEval [28]) and code generation (MBPP) [34]. 

**Evaluation Metrics.** The default setting for the fine-tuned base model inherently provides the CoT solution. Hence, we conduct all experiments using CoT settings. To thoroughly and accurately evaluate the capabilities of different models, we employ the accuracy metric for all tasks except for code generation, for which we use the pass@1 metric. 

**Baselines.** We consider the following baselines(e.g., GPT-4 [35], GLM [13; 14; 36] and LLaMA Base [37], Continue Pre-training, and Dataset-specific Tuning, etc.) and describe details in Appendix A.8. We employ a standardized evaluation framework to compare GLM and LLaMA Base baselines fairly. To gauge performance in the MATH task, we utilize zero-shot and 8-shot configurations to determine the highest accuracy. Additionally, for Mathematics, SAT-Math, MMLU, and CEval, we employ a chat module for assessment. When dealing with multiple-choice questions, we formulate the prompt as “Therefore, among A through D, the answer is”. 

**Data Contamination.** Both SciInstruct and the evaluation benchmarks fall within the science domain. To minimize any potential data contamination and strengthen the integrity of our results, we ensure that the training set used to construct SciInstruct was not derived from the test sets utilized in our evaluation. In other words, there was no overlap between the SciInstruct and the evaluation benchmarks used in our experiments. 

7 

## **3.2 Main Results and Analysis** 

Table 4, Table 5, and Table 6 present experimental findings for scientific and mathematical benchmarks, as well as general tasks. The results show that training on the provided SciInstruct benefits reasoning tasks, improving performance in scientific reasoning (i.e., CEval-Sci, SciEval, SciBench, MMLU-Sci) and transferring to mathematical tasks. Such performance improvement is consistent with different scales of based model parameters, across 6B and 32B. Table 6: **Results on general language** In addition, SciGLM’s performance improvement in sci- **understanding tasks** . Fine-tuning does entific reasoning does not sacrifice its general language not sacrifice most language tasks and only understanding capabilities. As shown in Figure 13 in A.9, drops a bit on the code generation task. on standard tasks like MMLU and CEval, SciGLM even achieves slight performance improvement. On code genModel MMLU CEval MBPP Avg. eration like MBPP, the performance is slightly lower but GPT-4 86.40 68.70 83.00 79.37 is overall consistent. As shown in Figure 15 in A.10, we ChatGLM3-6B-Base **SciGLM** (6B-Base) 61.3861.32 67.1667.09 45.0055.80 57.8561.40 present a statistics problem in SciBench that is accurately ChatGLM3-32B-Base 69.05 79.94 58.20 69.06 solved with the SciGLM (32B). **SciGLM** (32B-Base) 70.08 79.64 56.60 68.78 

## **4 SciInstruct Analysis** 

**Influence of Data Mixture.** We further explore how the diverse subjects within the SciInstruct mixture affect downstream tasks when training the SciGLM-6B model. By employing a Leave-One-Out strategy, i.e., omitting one subject at a time from the dataset and retraining, we assess the significance of each subject based on the performance impact across various tasks. As shown in Figure 6, we have an interesting findCEval-Math ing: each subject contributes tasks that are not reMATH stricted to its immediate domains. For instance, Physics and Chemistry data significantly aid in CEval-Math tasks, while Math and Formal Proof improve SciBench performance. This shows that SciBench Physics & Chemistry our mixture dataset enables LLMs to acquire some general reasoning skills for solving scientific questions instead of merely overfitting to a certain task distribution, and achieving universal improvement on different downstream tasks. Mathematics 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0008-04.png)


**----- Start of picture text -----**<br>
CEval-Math<br>MATH<br>SciBench<br>Physics & Chemistry<br>Mathematics<br>CEval-Sci<br>Formal Proofs<br>CEval-Hard<br>GSM8K<br>SAT-MATH<br>Math<br>SciEval<br>MMLU-Math<br>MMLU-Sci<br>Evaluation Dataset<br>Domain in SciInstruct<br>**----- End of picture text -----**<br>


**Influence of Data Scaling.** One central question for instruction dataset creation is how many samples are needed for a model to learn specific skills [38]. Prior works [39] have shown that for dialogue tasks, as few as 1000 high-quality instruction samples can lead to significant improvements. We’re interested in analyzing data scaling law for scientific problemsolving. Through retraining the model with varying data proportions and analyzing the outcomes in science and math, as shown in Figure 7, one interesting pattern we find is initial data augmentation of 10% yields improvements, but further additions show no significant gains until surpassing a 50% threshold. We hypothesize that the early gains are due to that finetuned LLM learning basic reasoning and task 

Figure 6: **Influence of different domains in fine-tuning data towards target tasks** . Weight is calculated by (Acc. (SciGLM) - Acc. (Exclude subjects) / Acc. (SciGLM)) under the leave-one-out subject setting. 

formatting, which requires fewer instruction data (less than 30k). Advancing to more complex skills, such as equation deduction, necessitates a larger dataset for effective learning and generalize. Future research on improving data quality could potentially lower the data requirement for LLM skill learning. 

**Pass@** _K_ **Analysis on Sample Diversity.** One interesting observation of LLM reasoning is that with non-zero temperature and sampling multiple times, even for those very hard questions, LLM still has a high chance of providing a correct answer. Pass@ _K_ is widely used for code generation [40; 41] and 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0009-00.png)


**----- Start of picture text -----**<br>
1.00<br>0.98<br>0.96<br>0.94<br>0.92<br>0.90<br>0.88 Ave. Sci<br>0.86 Ave. Math<br>0  10%  20%  50%  90%  (254051)<br>Acc (x% Data)/ Acc (100% Data)  (base) (25405) (50810)  (127025)  (228645) 100%<br>**----- End of picture text -----**<br>


Figure 7: **Performance improvement over different scale of instruction data** . The x-axis represents the proportion of instruction per domain, and the y-axis represents the relative performance compared with SciGLM trained with the whole set. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0009-02.png)


Figure 8: **Evaluating Pass@** _K_ **on SciBench (Quantum Chemistry) and SciEval.** All samples are generated at temperature 1.0. Results show that our instruction tuning does not influence the sample diversity, and increases the performance even with large _K_ . 

math reasoning [25]. To analyze whether our SciInstruct can really improve the general reasoning, we simulate different Pass@ _K_ values as shown in Figure 8. We use fine-tuned SciGLM (32B) and ChatGLM (32B) to generate _N ≥ K_ (in this paper, _N_ = 30 and _K ≤_ 25) solutions per question, allowing for a more accurate examination of the LLM’s true pass rate on that question. We find fine-tuning does not influence the sample diversity. SciGLM (32B) with _K_ =25 on SciBench and _K_ =3 on SciEval can achieve comparable performance to GPT-4, showing the potential of our fine-tuned model to achieve better results. We hypothesize that high-quality and diverse reasoning data indeed lead the model to good behavior/skills for analyzing and solving hard scientific problems instead of overfitting the training set, showing the general usability of our self-annotated SciInstruct dataset. 

## **5 Conclusion** 

In this work, we present a self-instructive annotation framework to create a high-level and high-quality dataset, SciInstruct, to enrich the scientific knowledge of LLMs. Using SciInstruct, we train three LLMs, which significantly improve many scientific and mathematical benchmarks over the base models and outperform many state-of-the-art LLMs that have an order of magnitude more parameters. Our research underscores the significance of diverse training data as well as LLM self-annotation and correctness for enhancing general reasoning capability, even for hard domains like science. 

## **Limitation** 

In this section, we discuss more limitations during the research of SciInstruct. 

**Scale of Dataset and Model.** Even though our training dataset has expanded to approximately 254k, improving model performance still necessitates access to an even larger dataset. Our model’s _∼_ experimental outcomes are carried out at 6 8B and 32B parameters, leading to a relatively better performance. However, it’s important to note that these performances are constrained by the model’s scale. Moving forward, it’s worth exploring the potential benefits of leveraging larger-scale datasets and models to further improve performance. 

**Using Data Classifier to Enhance the Generation of Models.** In line with what was discussed in Section 2.3, we employ an instruction-quality classifier to boost the instruction quality, yielding improved performance as shown in Table 2. However, we anticipate that the instruction-quality classifier, also referred to as the reward model, could provide even greater benefits. One particular avenue of improvement could be bootstrapping data to improve the ability of the base model. 

## **Broader Impact** 

**Positive impact.** This paper aims to construct high-level and high-quality instruction to improve the scientific reasoning capability of LLMs, which helps LLMs to better give the answers to questions at the college level. Collecting diverse instructions, annotating self-reflective instructions, and filtering out low-quality instructions provide researchers insights to prepare training datasets. 

**Negative impact.** A drawback of this work is that the scale of the training dataset and model is relatively small, and we can address this by bootstrapping a more large training dataset. We believe that the benefits of data generation manner outweigh the downside. 

9 

## **Acknowledgments** 

This work is supported by the NSFC 62276148, NSFC for Distinguished Young Scholar 62425601, a research fund from Zhipu, New Cornerstone Science Foundation through the XPLORER PRIZE and Tsinghua University (Department of Computer Science and Technology) - Siemens Ltd., China Joint Research Center for Industrial Intelligence and Internet of Things (JCIIOT). 

## **References** 

- [1] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. _arXiv preprint arXiv:2211.09085_ , 2022. 

- [2] Microsoft Research AI4Science and Microsoft Azure Quantum. The impact of large language models on scientific discovery: a preliminary study using gpt-4. _arXiv preprint arXiv:2311.07361_ , 2023. 

- [3] Bo Chen, Xingyi Cheng, Yangli ao Geng, Shen Li, Xin Zeng, Boyan Wang, Jing Gong, Chiming Liu, Aohan Zeng, Yuxiao Dong, Jie Tang, and Le Song. xtrimopglm: Unified 100b-scale pre-trained transformer for deciphering the language of protein. _bioRxiv_ , 2023. 

- [4] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3d neural networks. _Nature_ , 619:1–6, 07 2023. 

- [5] Cheng Deng, Tianhang Zhang, Zhongmou He, Qiyuan Chen, Yuanyuan Shi, Le Zhou, Luoyi Fu, Weinan Zhang, Xinbing Wang, Chenghu Zhou, et al. Learning a foundation language model for geoscience knowledge understanding and utilization. _arXiv preprint arXiv:2306.05064_ , 2023. 

- [6] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. _arXiv preprint arXiv:2308.13149_ , 2023. 

- [7] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. _arXiv preprint arXiv:2307.10635_ , 2023. 

- [8] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. _arXiv preprint arXiv:2311.12022_ , 2023. 

- [9] Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: Llm self-training via process reward guided tree search. _arXiv preprint arXiv:2406.03816_ , 2024. 

- [10] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_ , 2023. 

- [11] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

- [12] Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. _arXiv preprint arXiv:2405.03548_ , 2024. 

- [13] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 320–335, 2022. 

- [14] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. _arXiv preprint arXiv:2210.02414_ , 2022. 

10 

- [15] Meta AI. Meta llama 3. https://ai.meta.com/blog/meta-llama-3/, 2024. Accessed: 2024-05-22. 

- [16] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- [17] Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. Agenttuning: Enabling generalized agent abilities for llms. _arXiv preprint arXiv:2310.12823_ , 2023. 

- [18] Bruno Barras, Samuel Boutin, Cristina Cornes, Judicaël Courant, Jean-Christophe Filliatre, Eduardo Gimenez, Hugo Herbelin, Gerard Huet, Cesar Munoz, Chetan Murthy, et al. _The Coq proof assistant reference manual: Version 6.1_ . PhD thesis, Inria, 1997. 

- [19] Tobias Nipkow, Markus Wenzel, and Lawrence C Paulson. _Isabelle/HOL: a proof assistant for higher-order logic_ . Springer, 2002. 

- [20] Leonardo De Moura, Soonho Kong, Jeremy Avigad, Floris Van Doorn, and Jakob von Raumer. The lean theorem prover (system description). In _Automated Deduction-CADE-25: 25th International Conference on Automated Deduction, Berlin, Germany, August 1-7, 2015, Proceedings 25_ , pages 378–388. Springer, 2015. 

- [21] Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. LeanDojo: Theorem proving with retrieval-augmented language models. In _Neural Information Processing Systems (NeurIPS)_ , 2023. 

- [22] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. 

- [23] Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. _arXiv preprint arXiv:2303.11366_ , 2023. 

- [24] Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Heng Huang, Jiuxiang Gu, and Tianyi Zhou. Reflection-tuning: Data recycling improves llm instruction-tuning. _arXiv preprint arXiv:2310.11716_ , 2023. 

- [25] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _CoRR_ , abs/2305.20050, 2023. 

- [26] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukosiute, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning. _CoRR_ , abs/2307.13702, 2023. 

- [27] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In _SC20: International Conference for High Performance Computing, Networking, Storage and Analysis_ , pages 1–16. IEEE, 2020. 

- [28] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. _arXiv preprint arXiv:2305.08322_ , 2023. 

- [29] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_ , 2020. 

11 

- [30] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- [31] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- [32] Alex Davies, Petar Veliˇckovi´c, Lars Buesing, Sam Blackwell, Daniel Zheng, Nenad Tomašev, Richard Tanburn, Peter Battaglia, Charles Blundell, András Juhász, et al. Advancing mathematics by guiding human intuition with ai. _Nature_ , 600(7887):70–74, 2021. 

- [33] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _arXiv preprint arXiv:2304.06364_ , 2023. 

- [34] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- [35] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [36] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. _arXiv preprint arXiv:2406.12793_ , 2024. 

- [37] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- [38] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. _arXiv preprint arXiv:2308.01825_ , 2023. 

- [39] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. _arXiv preprint arXiv:2305.11206_ , 2023. 

- [40] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- [41] Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. _CoRR_ , abs/2203.07814, 2022. 

- [42] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _Advances in Neural Information Processing Systems_ , 35:3843–3857, 2022. 

- [43] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ , 2023. 

- [44] Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena Dumas Ang. Parsing algebraic word problems into equations. _Transactions of the Association for Computational Linguistics_ , 3:585–597, 2015. 

12 

- [45] Subhro Roy and Dan Roth. Solving general arithmetic word problems. _arXiv preprint arXiv:1608.01413_ , 2016. 

- [46] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? _arXiv preprint arXiv:2103.07191_ , 2021. 

- [47] Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 523–533, 2014. 

- [48] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. _arXiv preprint arXiv:1705.04146_ , 2017. 

- [49] Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. _arXiv preprint arXiv:1905.13319_ , 2019. 

- [50] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. _arXiv preprint arXiv:2310.10631_ , 2023. 

- [51] Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. _arXiv preprint arXiv:2204.05660_ , 2022. 

- [52] Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, et al. Lila: A unified benchmark for mathematical reasoning. _arXiv preprint arXiv:2210.17517_ , 2022. 

- [53] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset. _arXiv preprint arXiv:2305.12524_ , 2023. 

- [54] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022. 

- [55] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- [56] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- [57] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [58] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging bigbench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ , 2022. 

- [59] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

- [60] Boshi Wang, Xiang Deng, and Huan Sun. Iteratively prompt pre-trained language models for chain of thought. _arXiv preprint arXiv:2203.08383_ , 2022. 

13 

- [61] Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters. _arXiv preprint arXiv:2212.10001_ , 2022. 

- [62] Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 5315–5333, 2023. 

- [63] Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, and Zhifang Sui. Making large language models better reasoners with alignment. _arXiv preprint arXiv:2309.02144_ , 2023. 

- [64] Andrew Drozdov, Nathanael Schärli, Ekin Akyürek, Nathan Scales, Xinying Song, Xinyun Chen, Olivier Bousquet, and Denny Zhou. Compositional semantic parsing with large language models. _arXiv preprint arXiv:2209.15003_ , 2022. 

- [65] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. _arXiv preprint arXiv:2205.10625_ , 2022. 

- [66] Xue Lilong, Zhang Dan, Dong Yuxiao, and Tang Jie. Autore: Document-level relation extraction with large language models. _arXiv preprint arXiv:2403.14888_ , 2024. 

- [67] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

- [68] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _arXiv preprint arXiv:2211.12588_ , 2022. 

- [69] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. _arXiv preprint arXiv:2305.11738_ , 2023. 

- [70] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_ , 2021. 

- [71] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. _arXiv preprint arXiv:2305.04091_ , 2023. 

- [72] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _arXiv preprint arXiv:2304.12244_ , 2023. 

- [73] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. _arXiv preprint arXiv:2109.01652_ , 2021. 

- [74] Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. _arXiv preprint arXiv:2110.08207_ , 2021. 

- [75] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _arXiv preprint arXiv:2210.11416_ , 2022. 

- [76] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. _arXiv preprint arXiv:2301.13688_ , 2023. 

14 

- [77] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. _arXiv preprint arXiv:2212.10560_ , 2022. 

- [78] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_ , 2023. 

- [79] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. _arXiv preprint arXiv:2306.04751_ , 2023. 

- [80] Ariel N Lee, Cole J Hunter, and Nataniel Ruiz. Platypus: Quick, cheap, and powerful refinement of llms. _arXiv preprint arXiv:2308.07317_ , 2023. 

## **Part I** 

## **Appendix** 

## **Table of Contents** 

|**le**|**of Contents**|**of Contents**|||
|---|---|---|---|---|
|**A**|**Appendix**|||**16**|
||A.1|Related Works<br>. . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|16|
||A.2|SciInstructAnalysis . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|16|
||A.3|Instruction Examples . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|17|
||A.4|GPT Labeling. . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|17|
||A.5|Dataset of Instruction-quality|Classifer<br>. . . . . . . . . . . . . . . . . . . . .|17|
||A.6|Detailed Error Analysis. . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|17|
||A.7|Evaluation Tasks . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||A.8|Evaluation Baselines . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|20|
||A.9|Detailed Experimental Results . . . . . . . . . . . . . . . . . . . . . . . . . .||20|
||A.10|Output Examples ofSciGLM|. . . . . . . . . . . . . . . . . . . . . . . . . . .|21|
||A.11|Examples ofSciInstruct .|. . . . . . . . . . . . . . . . . . . . . . . . . . .|21|
||A.12|Scientifc Reasoning Datasets|. . . . . . . . . . . . . . . . . . . . . . . . . .|27|
||A.13|General Reasoning with LLMs . . . . . . . . . . . . . . . . . . . . . . . . . .||27|
||A.14|Instruction Tuning in LLMs|. . . . . . . . . . . . . . . . . . . . . . . . . . .|27|



15 

## **A Appendix** 

## **A.1 Related Works** 

Recently, there have been advances to bridge the gaps in reasoning difficulty and evaluation subjects from three perspectives for scientific reasoning with LLMs. 

**High-level evaluation** like SciBench [7] and GPQA [8], which evaluate the scientific reasoning capabilities of LLMs at the college level and even graduate level. In addition, SciEval [6] provides a multi-level LLMs evaluation benchmark to address data leakage problems and subjective question/answer evaluation ability issues. 

**Continued pre-training** like Galactica [1] and MINERVA [42], which continue to train their respective base LLMs on multiple web texts including science-related or math-related corpus. This continued pre-training approach explores the potential of LLMs for science and contributes to the open-source models for the scientific community, but it is computationally expensive. 

**Dataset-specific fine-tuning** like RFT [38], WizardMath [43], MAmmoTH [10], and MetaMath [11], which constructs certain datasets including GSM8K and MATH, conducts supervised fine-tuning of LLMs and evaluates the popular benchmarks GSM8K and MATH. MAmmoTH not only improves the in-domain (IND) performance like GSM8K but also generalizes to broader mathematical tasks by building a spectrum of math instruction tuning datasets including in-domain and out-of-domain datasets. This series of methods mainly focuses on mathematical reasoning tasks. 

## **A.2 SciInstruct Analysis** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0016-07.png)


**----- Start of picture text -----**<br>
Question Type Proportions in Mathematics<br>Complex Solution Problem<br>54.8%<br>0.4% Choice Question<br>9.3%<br>Filling<br>35.5%<br>Simple Solution Problem<br>(b)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0016-08.png)


**----- Start of picture text -----**<br>
Question Type Proportions<br>Complex Solution Problem<br>Simple Solution Problem<br>54.8%<br>Filling 14.6%<br>33.2%<br>20.1% Complex Solution Problem 0.4% Choice Question<br>9.3%<br>Filling<br>32.0% 35.5%<br>Choice Question Simple Solution Problem<br>(a) (b)<br>Question Type Proportions in Physics Question Type Proportions in Chemistry<br>Complex Solution Problem Simple Solution Problem<br>Complex Solution Problem<br>Filling 12.2% 0.4% 18.9%<br>35.0% Simple Solution Problem Filling<br>17.7% 40.3%<br>35.0% 40.3%<br>Choice Question<br>Choice Question<br>(c) (d)<br>**----- End of picture text -----**<br>


Figure 9: We present the proportions of question types in all subjects and each subject. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0017-00.png)


**----- Start of picture text -----**<br>
1000 7000<br>6000<br>800<br>5000<br>600<br>4000<br>400 3000<br>2000<br>200<br>1000<br>0 0<br>0 200 400 600 800 1000 0 200 400 600 800 1000<br>The length of questions The length of questions<br>(a) (b)<br>Number of questions<br>The average length of answers<br>**----- End of picture text -----**<br>


Figure 10: We show the number of questions of that length and the average length of all answers under each question. 

## **A.3 Instruction Examples** 

**Instruction Format for Physics, Chemistry, and Math.** To make our dataset consistent, we define a unified template for the problem-solving process that matches the logical flow of human thinking for Physics, Chemistry, and Math. For each question, we first list the basic analysis of which scientific concepts (or knowledge points) this question asks, and then present step-by-step reasoning steps of the detailed solution, and eventually summarize the answer. 

**This is an example of an instruction format. Problem:** _∗∗∗_ . **Answer:** Analysis: This question examines _∗∗∗_ knowledge points. Step 1: _∗∗∗_ . Step i: _∗∗∗_ . Step n: _∗∗∗_ . To sum up, the answer to this question is _∗∗∗_ . 

**Instruction Format for Formal Math.** 

**This is an example of an instruction format for Lean. Problem:** theorem gcd_self (n : Nat) : gcd n n = n **Answer:** cases n <;> simp [gcd, mod_self] 

## **A.4 GPT Labeling** 

As shown in Figure 11, we present the process of GPT-4 labeling. It is important to note that addressing the solution comparison by obtaining the accurate assessment of reasoning steps, such as through the use of a process reward model (PRM), typically requires costly human annotations, especially for complex scientific reasoning. Therefore, our work employs a labeling method based on an outcome reward model (ORM) as a basic implementation. While this approach may have limitations, it serves as a foundation for constructing the training set for our instruction-quality classifier. Moreover, Table 2 demonstrates the effectiveness of this classifier. 

## **A.5 Dataset of Instruction-quality Classifier** 

As shown in Table 7, we list the data statistics used to train the instruction-quality classifier. 

## **A.6 Detailed Error Analysis** 

As shown in Figure 12, we give detailed error examples, which are summarized in Section 2.3. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0018-00.png)


**----- Start of picture text -----**<br>
GPT-4 Labeling<br>Prompt: The following input includes three segments. It starts with a question, followed<br>by a step-by-step solution to the question, and ends with the true answer. If the answer<br>obtained in the solution is equivalent to the real one, output ‘1’, otherwise output ‘0’.<br>Input<br>Q:  A photon moving at speed c, a person chasing light at 0.9c, that person observes the<br>speed of light is?<br>Solution: According to Einstein’s theory of special relativity, we cannot simply add up the<br>velocities. In relativity, the superposition of velocities follows the Lorentz transformation,<br>rather than the classical addition. For the speed of light, there is a special case in which<br>the observer measures the speed of light as c, regardless of the observer’s speed. This<br>means that even if an observer is moving at 0.9c, the speed of light he measures is still c.<br>Therefore, the speed of light observed by this person is c.<br>Real Answer: c.<br>GPT-4 Output: 1<br>**----- End of picture text -----**<br>


Figure 11: We present the process of GPT-4 labeling. Given a prompt, question, solution, and real answer, GPT-4 can give the final label output. 

Table 7: Composing data based on a limited set of labeled examples to train a data classifier and filter out noisy instructions. 

|<br>ns.||||
|---|---|---|---|
|Source|# Positive Samples|# Negative Samples|Total|
|Original Clean Data|11,553|0|11,553|
|ChatGLM2-6B|0|23,106|23,106|
|GPT-3.5-Turbo-0613|745|2075|2,820|
|GPT-4-0613|332|3,898|4,230|
|Total|12,630|29,079|41,709|



## **A.7 Evaluation Tasks** 

Table 8: Overview of Scientific Evaluation Tasks. ( _†_ : Pre-Algebra, _‡_ : Inter-Algebra, _§_ : Algebra, _△_ : Probability, _♢_ : NumTheory, △: Statistics, ▲: Calculus, ▼: Geometry, ♦: Physics, ⋆: Chemistry, _♣_ : Biology) 

|Eval Dataset|# Samples|Answer Form|Subject|
|---|---|---|---|
|CEval-Hard|152|Multi-choice|_△,_▲_, ♢,_△,♦,⋆|
|CEval-Sci|210|Multi-choice|_△, ♢,_△_,_▲,♦,⋆,_♣_|
|MMLU-Sci|1,855|Multi-choice|_§, △, ♢,_△_,_▲,♦,⋆,_♣_|
|SciEval|15,901|Multi-choice|♦,⋆,_♣_|
|MATH|5,000|Open-formed|_†, ‡, §, △, ♢,_▲_,_▼|
|Mathematics|1,000|Open-formed|_†, ‡, ♢,_▲|
|SAT-Math|220|Multi-choice|_‡, △,_▼|
|MMLU-Math|974|Multi-choice|_§, △, ♢,_▲|
|CEval-Math|90|Multi-choice|△_, △, ♢,_▲|



. 

- CEval-Hard [28] comprises the eight subjects at the hard level from CEval. 

- CEval-Sci [28] is selected from CEval including physical, chemical, and mathematical, subjects. 

- MMLU-Sci [29] is similar to CEval-Sci and includes physical, chemical, and mathematical subjects selected from MMLU. 

18 

**Comprehension mistakes Explanation** Answers that do not resolve the task exactly as instructed, often including an output that is insufficient. **Example Q:** A particle moving in a straight line in the _x_ direction has the coordinates _x_ = 5 _t_[2] _− t_[3] at time _t_ , where _x_ is measured in meters and _t_ in seconds. Find: displacement and average velocity in the 4th second. 

**A:** The displacement in 4 seconds can be obtained by calculating the coordinates at _t_ = 4 minus the coordinates at _t_ = 0, Displacement = _x_ (4) _− x_ (0) = 5 _×_ 4[2] _−_ 4[3] _−_ 5 _×_ 0[2] + 0[3] = 5 _×_ 16 _−_ 64 _−_ 0 + 0 = 80 _−_ 64 = 16. Average velocity in 4 seconds can be obtained by calculating displacement divided by time, that is _average_  velocity_ = _displacement/time_ = 16 _/_ 4 = 4. In summary, the displacement in 4 seconds is 16 meters, the average speed is 4 meters/second, and the distance traveled by the particle is 16 meters. 

**Error:** Wrong comprehension. The answer should be displacement and average velocity ’in the 4th second’ rather than ’in 4 seconds’. 

## **Calculation mistakes Explanation** 

Answers that fail to accurately calculate or simplify mathematical expressions, especially when encountering integration and differentiation. **Example** 

**Q:** In a shell with an inner radius of _R_ 1 and an outer radius of _R_ 2, the charge density in the shell is _ρ_ = _a_ + _br_ , Where r is the distance from the center to the observation point, and there is no charge distribution elsewhere in the space. (a) Find the expression that the electric field everywhere in the space depends on r. **A:** First of all, according to Gauss’s law, the total electric flux of any closed surface inside the closed surface is equal to the total charge inside the closed surface. According to the conditions in the topic, the charge density in the shell is _ρ_ = _a_ + _br_ , the total charge in the shell can be obtained as: _Q_ = � _RR_ 12 �0 _π_ �02 _π_[(] _[a]_[+] _[br]_[)] _[·][r]_[2][ sin] _[ θdϕdθdr]_[=][4] _[π]_ � _RR_ 12[(] _[a]_[+] _[br]_[)] _[r]_[2] _[dr]_[=] 4 _π_ � _a_ 4[(] _[R]_ 2[4] _[−][R]_ 1[4][) +] 5 _[b]_[(] _[R]_ 2[5] _[−][R]_ 1[5][)] �. According to the Gauss theorem, inside the spherical shell, the flux of the electric field is zero. Therefore, we only need to produce the electric field outside the shell. Outside the shell, according to Coulomb’s Law, the electric field generated by the charge can be expressed as: _E_ = 4 _πϵ_ 1 0 _rQ_[2][Substituting] _[Q]_[into][this][formula][yields:] _[E]_[=] 1 4 _π_ ( _[a]_ 4[(] _[R]_ 2[4] _[−][R]_ 1[4][)+] 5 _[b]_[(] _[R]_ 2[5] _[−][R]_ 1[5][)][)] 4 _πϵ_ 0 _r_[2] simplified, we can get electric field depends on the expression of _r_ are as follows: _E_ = _ϵ_ 10 � _ra_[2][(] _[R]_ 2[4] _[−][R]_ 1[4][) +] _rb_[2][(] _[R]_ 2[5] _[−][R]_ 1[5][)] � **Error:** Wrong integration. 

## **False reasoning Explanation** 

Solutions consist of false reasoning steps or made-up information in reasoning. 

## **Example** 

**Q:** The escape work of polonium and barium is 4.52 eV, and 2.50 eV, respectively. Calculate the cut-off frequencies of barium and polonium. **A:** According to the theory of photoelectric effect, the relationship between escape work and cutting potential can be expressed by the following formula escape work = cutting potential _−_ photon energy. Apply the relationship between escape work and cutting potential to the case of barium and polonium. For polonium, cutting potential = escape work + photon energy = 4 _._ 52 eV + _hν_ . For barium, cutting potential = escape work + photon energy = 2 _._ 50 eV + _hν_ , where _h_ is Planck’s constant and _ν_ is the frequency of light. 

In the visible range, the frequency range of light is approximately 4 _._ 3 _×_ 10[14] Hz to 7 _._ 5 _×_ 10[14] Hz. Now we can calculate the cut-off frequencies of barium and polonium respectively. For polonium, 4 _._ 52 eV + _hν_ cut = 4 _._ 30 eV. _hν_ cut = 4 _._ 30 eV _−_ 4 _._ 52 eV. _hν_ cut = _−_ 0 _._ 22 eV. For barium, 2 _._ 50 eV + _hν_ cut = 4 _._ 30 eV. _hν_ cut = 4 _._ 30 eV _−_ 2 _._ 50 eV. _hν_ cut = 1 _._ 80 eV. 

In summary, the cut-off frequency of polonium is _−_ 0 _._ 22 eV, and that of barium is 1 _._ 80 eV. Because the cutoff frequency is the frequency of light and must be positive, barium can be used as a phototube cathode material in the visible range. Final answer: Barium can be used as a phototube cathode material in the visible range. **Error:** False reasoning, cutting potential should be W/h. 

Figure 12: Error analysis of three types of self-reflection for noisy instruction filtering. 

19 

- SciEval [6] serves as a comprehensive, multi-disciplinary benchmark for assessing subjective question-answering capabilities. 

- MATH [31] is a standard benchmark for evaluating the mathematical ability of LLMs. 

- Mathematics [32] mainly includes Algebra, NumTheory, and Calculus. 

- SAT-Math [33] is a math task including Intermediate Algebra, Probability, and Calculus. 

- MMLU-Math [29] evaluation is the same as MAmmoTH [10], including Algebra, Probability, NumTheory, and Calculus. 

- CEval-MATH [28] only includes mathematical subjects extracted from CEval. 

## **A.8 Evaluation Baselines** 

- GPT-4 [35]: We consider the closed-source model GPT-4, which uses CoT prompting. 

- GLM [13; 14] and LLaMA Base [37]: We select ChatGLM2-6B (Base), ChatGLM3-6B (Base), and ChatGLM3-32B (Base) as the base models. In addition, we consider LLaMA-2-7B and LLaMA-2-13B as baselines. 

- Continue Pre-training: We include Galactica [1] although we note that its continued pre-training does not necessarily include college-level scientific reasoning. 

- Dataset-specific Tuning: For the dataset-specific tuning, we consider WizardMath [43], MAmmoTH-CoT [10], and MetaMath [11], which adapt to the MATH dataset. 

## **A.9 Detailed Experimental Results** 

Regarding domain-wise results in Figure 13, we present the detailed subject results in Table 9 and Table 10. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models_images/nips_2024_SciInstruct_a_Self-Reflective_Instruction_Annotated_Dataset_for_Training_Scientific_Language_Models.pdf-0020-13.png)


**----- Start of picture text -----**<br>
50.0 Mathematical Tasks 70 Physical Tasks<br>47.5 68<br>45.0 66<br>42.5 64<br>40.0 62<br>37.5 60<br>35.0 6B 32B 58 6B 32B<br>Chemical Tasks General Tasks<br>70.0 74<br>ChatGLM3<br>67.5 72 SciGLM (Ours)<br>65.0 70<br>62.5 68<br>60.0 66<br>57.5 64<br>55.0 62<br>6B 32B 60 6B 32B<br>Average Score<br>Average Score Average Score<br>**----- End of picture text -----**<br>


Figure 13: SciGLM consistently improving the ChatGLM3-Base for different scientific tasks, without sacrificing general language understanding capabilities. 

Table 9: Detailed results of physical subjects for Figure 3. **Bold** denotes the best result in the closed-source models and open-source models. 

|**model**|**Ave.**||CEval|CEval||||MMLU||SciEval|
|---|---|---|---|---|---|---|---|---|---|---|
|||**_Avg._**|College<br>Physics|High<br>School<br>Physics|Middle<br>School<br>Physics|**_Avg._**|College<br>Physics|Conceptual<br>Physics|Middle<br>School<br>Physics|**_Physics._**|
|ChatGLM3-6B-Base|54.32|_63.16_|42.11|63.16|84.21|_45.59_|**39.22**|55.32|34.44|46.95|
|SciGLM (ChatGLM3-6B-Base)|**56.59**|**_66.67_**|**47.37**|**63.16**|**89.47**|**_46.52_**|34.31|55.32|**41.06**|**47.56**|
|ChatGLM3-32B-Base|67.85|_78.95_|57.89|78.95|100.00|_56.76_|**42.16**|62.55|57.62|51.22|
|SciGLM (ChatGLM3-32B-Base)|**69.65**|**_80.70_**|**63.16**|78.95|100.00|**_58.61_**|**41.18**|**66.81**|57.62|**56.71**|



20 

Table 10: Detailed results of chemical subjects for Figure 3. **Bold** denotes the best result in the closed-source models and open-source models. 

|**model**<br>**Ave.**|**model**<br>**Ave.**|CEval<br>**_Avg._**<br>College<br>Chemistry<br>High<br>School<br>Chemistry<br>Middle<br>School<br>Chemistry|CEval<br>**_Avg._**<br>College<br>Chemistry<br>High<br>School<br>Chemistry<br>Middle<br>School<br>Chemistry|MMLU<br>SciEval<br>**_Avg._**<br>College<br>Chemistry<br>High<br>School<br>Chemistry<br>**_Chemistry._**|MMLU<br>SciEval<br>**_Avg._**<br>College<br>Chemistry<br>High<br>School<br>Chemistry<br>**_Chemistry._**|SciEval|
|---|---|---|---|---|---|---|
||||College<br>Chemistry<br>High<br>School<br>Chemistry<br>Middle<br>School<br>Chemistry||College<br>Chemistry<br>High<br>School<br>Chemistry|**_Chemistry._**|
|ChatGLM3-6B-Base<br>SciGLM (ChatGLM3-6B-Base)|58.25<br>**58.55**|_66.67_<br>**_68.25_**|45.83<br>63.16<br>95.00<br>**41.67**<br>**73.68**<br>**95.00**|**_49.83_**<br>_48.84_|45.00<br>**52.22**<br>**47.00**<br>49.75|55.76<br>**57.27**|
|ChatGLM3-32B-Base<br>SciGLM (ChatGLM3-32B-Base)|66.44<br>**69.05**|_69.84_<br>**_71.42_**|41.67<br>78.95<br>95.00<br>**45.83**<br>78.95<br>95.00|_63.04_<br>**_66.67_**|**51.00**<br>68.97<br>**57.00**<br>**71.43**|60.15<br>**61.40**|



## **A.10 Output Examples of SciGLM** 

We show several concrete examples in SciBench, as well as the output generated by SciGLM (32B) and ChatGLM3-32B-Base in Figure 14, Figure 15, Figure 16, Figure 17, and Figure 18. 

**Problem** When an electron in a certain excited energy level in a one-dimensional box of length 2.00 _\u_ 00 _c_ 5 makes a transition to the ground state, a photon of wavelength 8 _._ 79 nm is emitted. Find the quantum number of the initial state. **Correct Answer** : 4 **Predicted Solution by ChatGLM3-32B-Base** (Served as baseline) The ground state has _n_ = 1 _._ The energy in the _n_ = 1 state is _L_ 1[2] _[.]_[ For the] _[ n]_[=][2][ state, this becomes] � 21 _L_ �2 _._ Thus, this energy difference is 2[2] 1[2] _[−]_ 11[2][=][9] 4 _[.]_ By de Broglie’s equation, _hc/λ_ = 4 _hcν_ 9 _[,]_[ so] _[ ν]_[=][4] 9 _[hc] λ[.]_[ Thus,] _[ E]_[=] _[hν]_[=] _[hc]_ 9[4] _λ[.]_ Therefore, _n_ = 2, thus the answer is 2 . **Predicted Solution by SciGLM** (Finetuned from ChatGLM3-32B-Base) Concept: The concept of this problem is based on the photoelectric effect. The photoelectric effect is the emission of electrons when light shines on a material. The photoelectric effect occurs when photons with energies greater than the binding energy of electrons cause the ejection of electrons from a material. When a photon is emitted by an electron, the energy of the photon is equal to the energy of the electron in the excited state minus the energy in the ground state. The energy of a photon is given by the equation E = hc/ _λ_ , where h is Planck’s constant (6 _._ 626 _×_ 10 _[−]_[34] Js), c is the speed of light (3 _._ 000 _×_ 10[8] m/s), and _λ_ is the wavelength of the photon in meters. The energy of an electron in a one-dimensional box is given by the equation _En_ = ( _n_[2] _× h_[2] _/_ 8 _l_[2] ), where n is the quantum number of the state and l is the length of the box in meters. The problem asks for the quantum number of the initial state. This can be found by setting the energies of the initial and final states equal to the energy of the photon. Step 1: Convert the length of the box and the wavelength of the photon to meters. - The length of the box is given as 2.00 Å. We need to convert this to meters by multiplying by 10 _[−]_[10] (since 1 Å = 10 _[−]_[10] _m_ ). - The wavelength of the photon is given as 8.79 nm. We need to convert this to meters by multiplying by 10 _[−]_[9] (since 1 nm = 10 _[−]_[9] m). Step 2: Calculate the energy of the photon. - Use the equation E = hc/ _λ_ , where h is Planck’s constant (6 _._ 626 _×_ 10 _[−]_[34] Js) and c is the speed of light (3 _._ 00 _×_ 10[8] m/s). Step 3: Set the energy of the final state (n = 1) equal to the energy of the photon and solve for n. - Use the equation _En_ = ( _n_[2] _∗ h_[2] _/_ 8 _l_[2] ), where l is the length of the box in meters. Step 4: Convert the answer to scientific notation and round to 3 decimal places. To sum up, the answer to the question is therefore _n_ = 4 _._ 000 . ✓ 

Figure 14: An example **Quantum Chemistry** problem in SciBench that has been solved accurately with SciGLM (32B). It shows that after instruction-tuning, the SciGLM learns skills/behaviors to first analyze the knowledge required for each problem, and then step-by-step solve the problem with correct formula and calculations. 

## **A.11 Examples of SciInstruct** 

We present SciInstruct examples for Math (Figure 19), Physics (Figure 20), Chemistry (Figure 21), and Formal Proofs (Figure 22). 

21 

|**Problem**Extend Example 1.4-6 to an_n_-sided die. That is, suppose|**Problem**Extend Example 1.4-6 to an_n_-sided die. That is, suppose|**Problem**Extend Example 1.4-6 to an_n_-sided die. That is, suppose|that a fair_n_-sided die is rolled_n_independent times. A match occurs if|that a fair_n_-sided die is rolled_n_independent times. A match occurs if|that a fair_n_-sided die is rolled_n_independent times. A match occurs if|that a fair_n_-sided die is rolled_n_independent times. A match occurs if|
|---|---|---|---|---|---|---|
|side_i_is observed on the_i_th trial,_i_ = 1_,_2_, . . . , n_. Find the limit||of||this probability as_n_increases without bound.|||
|**Correct Answer**: 1_−_1_/e_ = 0_._6321_. . ._|||||||
|**Predicted Solution by ChatGLM3-32B-Base**|||||||
|We need the limit of_P_(_A_)as_n_goes to infnity. _P_(_A_)equals<br>1<br>_n_~~_n_ ~~|||as there are_n_possibilities for each of the_n_rolls. Thus we wish to fnd||||
|lim_n→∞_<br>1<br>_n_~~_n_ ~~= lim_x→∞_<br>1<br>_x_~~_x_ ~~= 0.|||||||
|Conclusion: The answer is therefore 0<br>.|||||||
||||||||
|**Predicted Solution by SciGLM**(Finetuned from ChatGLM3-32B-Base)|||||||
|The probability that all_i_sides of the die are rolled_i_times in a row||is||�1<br>_n_|�_n_ =|1<br>_n_~~_n_ ~~.|
|Step 1: If we write the sum of the_n_terms of this geometric<br>there is a total of_n_terms being summed. Thus, if we want to|sequence from_i_ = 1to_n_, it equals1because each term in the sum is 1<br>_n_ and<br>fnd the probability that no match occurs, we sum all probabilities that no match||||||
|occurs over_k_ = 1to_n_. This becomes|||||||
|_n_|||||||
|�<br>1<br>_k_~~_n_ ~~=<br>1<br>1~~_n_ ~~+<br>1<br>2~~_n_ ~~+<br>1<br>3~~_n_ ~~+_· · ·_+<br>1<br>_n_~~_n_~~|||||||
|_k_=1|||||||
|As_n →∞_, the sum approaches a limiting terme_−_1. Since|the total probability|||||must equal1for all_n_, and we just established that the limit|
|of the sum of all probabilities that no match occurs ise_−_1, this must||||be|the probability that no match occurs at all. The probability that at||
|least one match occurs is then1_−_e_−_1.|||||||
|Step 2:|||||||
|Thus, as_n →∞_, the limit of the probability that no match occurs||ise_−_1 = 0_._367, and the limit of the probability that at least one match|||||
|occurs is1_−_e_−_1 _≈_0_._63.|||||||
|To sum up, the answer to the question is 0.63.|||||||
|The limit of the probability that at least one match occurs is|0_._63||.|✓|||
||||||||



Figure 15: An example statistics problem in SciBench that is accurately solved with the SciGLM (32B). 

|**Problem**The World Series in baseball continues until either the American League team or the National League team wins four games. How|**Problem**The World Series in baseball continues until either the American League team or the National League team wins four games. How|**Problem**The World Series in baseball continues until either the American League team or the National League team wins four games. How|**Problem**The World Series in baseball continues until either the American League team or the National League team wins four games. How|**Problem**The World Series in baseball continues until either the American League team or the National League team wins four games. How|
|---|---|---|---|---|
|many different orders are possible (e.g.,||ANNAAA means the American League team wins in six games) if the series goes four games?|||
|**Correct Answer Latex**: 2|||||
|**Correct Answer Number**: 2|||||
|**Answer generated by ChatGLM3-32B-Base**|||||
|The answer is therefore 20<br>.|||||
||||||
|**Answer generated by SciGLM (32B)**|||||
|Solution:|||||
|Step 1: American league team wins in 4||games(4<br>4).|||
|National team wins in 4 games(4<br>4).|||||
|Step 2: Total number of ways|||||
|=(4<br>4)+(4<br>4)=2_×_1=2.|||||
|Conclusion: The answer is therefore|2||.|✓|
||||||



Figure 16: A statistics example problem in SciBench that has been solved accurately with SciGLM (32B). 

22 

|**Problem**7.3-9. Consider the following two groups of women: Group 1 consists of women who spend less than 500 annually on clothes;|
|---|
|Group 2 comprises women who spend over1000annually on clothes. Let_p_1 and_p_2 equal the proportions of women in these two groups,|
|respectively, who believe that clothes are too expensive. If 1009 out of a random sample of 1230 women from group 1 and 207 out of a|
|random sample 340 from group 2 believe that clothes are too expensive,|
|(a) Give a point estimate of_p_1 _−p_2.|
|**Correct Answer Latex**: $0.2115$|
|**Correct Answer Number**: 0.2115|
|**Answer generated by ChatGLM3-32B-Base**|
|To fnd the point estimate of_p_1_p_2, which is the joint probability that a woman from group 1 and a woman from group2, believe that clothes|
|are too expensive, we will use the number of women in each group that believe clothes are too expensive.|
|From group1, we have that 10.0907 of women believe that clothes are too expensive, And from group 2, we have that 0.6076|
|of women believe that clothes are too expensive.|
|The point estimate is approximately(123)=0.000000.|
|The answer is therefore 0_._000<br>.|
||
|**Answer generated by SciGLM (32B)**|
|Solution:|
|_n_1 = 1230_, p_1 = 1009<br>_n_1<br>_≈_0_._819_,_<br>_n_2 = 340_, p_2 _≈_207<br>340 = 0_._609|
|_p_1 _−p_2 _≈_0_._819_−_0_._609 = 0_._210|
|The answer is therefore 0_._210<br>.<br>✓|
||



Figure 17: A statistics example problem in SciBench that has been solved accurately with SciGLM (32B). 

|**Problem**Calculate the uncertainty∆_Lz_ for the hydrogen-atom stationary state:|**Problem**Calculate the uncertainty∆_Lz_ for the hydrogen-atom stationary state:|2_pz_.|
|---|---|---|
|**Correct Answer Latex**: 0<br>**Correct Answer Number**: 0|||
|**Answer generated by ChatGLM3-32B-Base**<br>Since the angular momentum of a revolving electron around the nucleus is: _Lz_ = <br>the proton is about . The uncertainty is: ∆_Lz_ = ℏ<br>2<br>~~_√_~~<br>2. The answer is therefore||_µB_ℏ. And the radius of revolution for the electron around<br> 7_._605<br>.|
|**Answer generated by Sci**<br>Solution:<br>∆_Lz_ =<br>_h_<br>2_π_<br>�<br>ℏ<br>_mc_ℏ_c_<br>�<br>=<br>~~�~~<br>1_·_(1_−_1)<br>2<br>=<br>~~�~~<br>0<br>2 = 0<br>The answer is therefore 0|**GLM (32B)**<br>_n_(_n−_1)<br>2<br>.<br>✓||



Figure 18: A Quantum Chemistry example problem in SciBench that has been solved accurately with SciGLM (32B). 

23 

**Problem** Using the method of undetermined coefficients to answer the below question. _y[′′] −_ 2 _y[′]_ + 5 _y_ = sin _x, y_ (0) = 1 _, y[′]_ (0) = 1 **Answer Analysis:** The key points of this problem are the undetermined coefficients and the differential equation of parameter variations. This test uses the Method of undetermined coefficients to solve differential equation problems. Step1: the auxiliary equation is _r_[2] _−_ 2 _r_ + 5 = 0 with roots _r_ = 1 _±_ 2 _i_ , so the complementary solution is: _ιe_ ( _x_ ) = _e[x]_ ( _c_ 1 cos 2 _x_ + _c_ 2 sin 2 _x_ ) _._ Try the particular solution: _yp_ ( _x_ ) = _A_ cos _x_ + _B_ sin _x_ , _yp[′]_[=] _[−][A]_[ sin] _[ x]_[ +] _[ B]_[ cos] _[ x]_ Step2: substituting, we have: ( _−A_ cos _x − B_ sin _x_ ) _−_ 2( _−A_ sin _x_ + _B_ cos _x_ ) + 5( _A_ cos _x_ + _B_ (sin _x_ ) = sin _x ⇒_ (4 _A −_ 2 _B_ ) cos _x_ + (2 _A_ + 4 _B_ ) sin _x_ = sin _x._ Then: 4 _A −_ 2 _B_ = 0 _,_ 2 _A_ + 4 _B_ = 1 _⇒ A_ = 101[,] _[ B]_[=] 3[1] and the general solution is: _y_ ( _x_ ) = _y_ 1( _x_ ) + _yp_ ( _x_ ) = _e[′]_ ( _c_ 1 cos 2 _x_ + _c_ 22 _x_ ) + 101[cos] _[ x]_[ +][1] 3[sin] _[ x]_ But, 1 = _y_ (0) = _c_ 1 + 101 _⇒ c_ 1 = 103 and 1 = _y[′]_ (0) = 2 _c_ 2 + _c_ 1 +[1] 5 _⇒ c_ 2 = _−_ 26[1] Step3: Thus the solution to the initial-value problem is: _y_ ( _x_ ) = _e[x]_[ �] 10[9][cos 2] _[x][ −]_ 201[sin 2] _[x]_ � + 101[cos] _[ x]_[ +][1] 5[sin] _[ x.]_ To sum up, the answer is: _y_ ( _x_ ) = _e[x]_[�] 10[9][cos 2] _[x][ −]_ 201[sin 2] _[x]_ � + 101[cos] _[ x]_[ +][1] 5[sin] _[ x.]_ 

Figure 19: An instruction example on the math subject. 

24 

|**Problem**|
|---|
|Two lightbulbs, one rated 30 W at 120 V and another rated 40 W at 120 V, are arranged in two different circuits.|
|a. The two bulbs are frst connected in parallel to a 120 V source.|
|i. Determine the resistance of the bulb rated 30 W and the current in it when it is connected in this circuit.|
|ii. Determine the resistance of the bulb rated 40 W and the current in it when it is connected to this circuit.|
|b. The bulbs are now connected in series with each other and have a 120 V source.|
|i. Determine the resistance of the bulb rated 30 W and the current in it when it is connected in this circuit.|
|ii. Determine the resistance of the bulb rated 40 W and the current in it when it is connected to this circuit.|
|c. In the spaces below, number the bulbs in each situation described, in order of their brightness.|
|(1= brightest, 4 = dimmest)|
|____30 W bulb in the parallel circuit|
|____40 W bulb in the parallel circuit|
|____30 W bulb in the series circuit|
|____40 W bulb in the series circuit|
|d. Calculate the total power dissipated by the two bulbs in each of the following cases.|
|i. The parallel circuit|
|ii. The series circuit|
|**Answer**|
|**Aanalysis:** Use formulas to calculate resistance, current, and total power.|
|(a) i: Calculate the resistance and current of the light bulb.|
|P = V2_/_Rgiven R=480Ωand|
|V=IR given I=0.25 A|
|ii: Calculate the resistance and current of the light bulb.|
|P = V2_/R_given R=360Ωand|
|V=IR given I=0.33 A|
|(b) i./ii. Calculate the resistance and current of the light bulb.|
|The resistances are unchanged 480 Ωand 360 Ω. The total resistance in series is 480 Ω+ 360 Ω= 840 Ωmaking the total current|
|I = V_/_R = 0_._14 Awhich is the same value for both resistors in series.|
|(c) Compare the brightness of the light bulb.|
|The bulbs are brightest in parallel, where they provide their labeled values of 40 W and 30 W. In series, it is the larger resistor (the 30 W bulb)|
|that glows brighter with a larger potential difference across it in series. This gives the order from top to bottom as 2 1 3 4.|
|(d) i: Calculate the total power consumed by two light bulbs.|
|In parallel, they each operate at their rated voltage so they each provide their rated power andPT = 30W + 40W = 70W|
|ii: Calculate the total power consumed by two light bulbs|
|In series**PT** = **VT**<br>2_/_**RT** = 17**W**|
|In summary,|
|(a) i: P = 480 ΩandV = 0_._25A|
|ii: P = 360 ΩandV = 0_._33A|
|(b) i/ii: P = 840 ΩandV = 0_._14A|
|(c) This gives the order from top to bottom as 2 1 3 4.|
|(d) i: PT = 70W|
|ii: **PT** = 17**W**|



Figure 20: An instruction example on the physics subject. 

25 

**Problem** Consider a mixture of the two solids, BaCl2 + 2H2O (FM 244.26) and KCl (FM 74.551), in an unknown ratio. (The notation BaCl2 _·_ 2H2O means that a crystal is formed with two water molecules for each BaCl2.) When the unknown is heated to 160 _[◦]_ C for 1 h, the water of crystallization is driven off: 

BaCl2 _·_ 2H2O( _s_ ) 160 _−→[◦]_ C BaCl2( _s_ ) + 2H2O(g) A sample originally weighing 1 _._ 7839 g weighed 1 _._ 5623 g after heating. Calculate the weight percent of Ba _,_ K, and Cl in the original sample. **Answer Analysis:** The content of this question is to calculate the weight percentage. Step1: Formula and atomic masses: Ba(137 _._ 327) _,_ Cl(35 _._ 453) _,_ K(39 _._ 098), H2O(18 _._ 015) _,_ KCl(74 _._ 551) _,_ BaCl2 _·_ 2H2O(244 _._ 26) _,_ H2O lost = 1 _._ 7839 _−_ 1 _._ 5623 = 0 _._ 2216 g = 1 _._ 2301 _×_ 10 _[−]_[2] mol of H2O. For 2 molH2O lost, 1 mol BaCl2 _·_ 2H2O must have been present.[1] 2 �1 _._ 2301 _×_ 10 _[−]_[2] molH2O lost ) = 6 _._ 1504 _×_ 10 _[−]_[3] molBaCl2 _·_ 2H2O = 1 _._ 5024 g. The Ba and Cl contents of the BaCl2 _·_ 2H2O are Ba = � 244137 _.._ 2633 � (1 _._ 5024 g) = 0 _._ 84469 g Cl = � 2(24435 _.._ 45326 ) � (1 _._ 5024 g) = 0 _._ 43613 g Step2: Because the total sample weighs 1 _._ 783 g and contains 1 _._ 5024 g of BaCl2 _·_ 2H2O, the sample must contain 1 _._ 7839 _−_ 1 _._ 5024 = 0 _._ 2815 g of KCl, which contains K = � 3974 _.._ 098551 � (0 _._ 2815) = 0 _._ 14763 g Cl = � 7435 _.._ 551453 � (0 _._ 2815) = 0 _._ 13387 g Weight percent of each element: Ba =[0] 1 _[.] ._[84469] 7839[=][47] _[.]_[35%] K =[0] 1 _[.] ._[14763] 7839[=][8] _[.]_[28%] Cl =[0] _[.]_[4][36] 1[1] _._[3+0] 7839 _[.]_[1][3387] = 31 _._ 95% In summary, the weight percent of Ba is 47 _._ 35%,the weight percent of K is 8 _._ 28%, the weight percent of Cl is 31 _._ 95%. 

Figure 21: An instruction example on the chemistry subject. 

## **Problem** 

lemma unfold_sub _α_ [subtraction_monoid _α_ ] (a b c : _α_ ) (h : a + -b = c) : a - b = c :=. 

**Answer** by rw [sub_eq_add_neg, h] 

**Problem** lemma subtype.exists_pi_extension `ι` : Sort* _α_ : `ι` → Sort* [ne : `Π` i, nonempty ( `α` i)] p : `ι` → Prop (f : `Π` i : subtype p, _α_ i) : g : `Π` i : `ι` , _α_ i, ( `λ` i : subtype p, g i) = f := **Answer** begin tactic.classical, refine 〈 `λ` i, if hi : p i then f 〈i, hi〉 else classical.choice (ne i), funext _〉, rintro 〈i, hi〉, exact dif_pos hi end 

Figure 22: Examples on formal proofs. 

26 

## **A.12 Scientific Reasoning Datasets** 

**Mathematical Datasets.** In the past, the emphasis in mathematical reasoning tasks was primarily on fundamental arithmetic reasoning datasets [44; 45; 46], addressing basic mathematical problems such as AddSub [47]. Later on, to address realistic math word problems, some more difficult datasets are proposed [48; 49; 30; 50]. To construct a more difficult and diversified dataset, NumGLUE [51] and LiLA [52] have been introduced to enhance current research. However, their focus remains primarily on elementary, middle school, and high school math problems. Given the rapid advancement of LLMs, to assess their capacity and constraints in addressing more intricate math problems, MMLU [29] with college-level math problems has been proposed. To address the more challenging college-level math problems, PoT [53; 7] is proposed. Our proposed SciInstruct covers more complex and more diversified math problems at the college level. 

**Scientific Datasets.** To address the scientific ability of LLMs, SciBench [7] proposes various solutions like calling existing tools (python and Wolfram). SciEval [6] presents a multi-level evaluation benchmark for scientific research. GPQA [8] builds a graduate-level Q&A benchmark to evaluate more difficult physics and chemistry questions. 

## **A.13 General Reasoning with LLMs** 

Assisted by Chain-of-Thought prompting [54; 55; 56] and Tree-or-Thought [57], LLMs have brought decent reasoning performance improvements. Especially, on the challenging BIG-Bench tasks, the CoT technique has already outperformed human ability [58]. Later on, to address reasoning tasks, a few studies [59; 60; 61; 62; 63; 11; 64; 65; 66] propose various methods to leverage LLMs and retain step-by-step processes. With the rapid development of LLM Agents, several works [67; 17] propose to utilize external tools to enhance the reasoning capabilities of LLMs. For example, ReAct [67] attempts to call existing tools like web search engines to improve the reasoning skills of LLMs. Indeed, leveraging the programs as thought processes is a natural way. Several recent studies have utilized programs as thought processes, such as PoT [68], to improve the reasoning abilities of LLMs. To enhance LLMs’ reasoning performance in solving mathematical or scientific problems with PoT, several methods have been suggested, including Self-critic [69], Self-eval [70], and Planand-solve [71]. For example, self-critic [69] and self-eval [70] propose to adopt self-evaluation to improve the generated program’s robustness. Different from these two methods, plan-and-solve [71] adopts more detailed planning instructions to generate a high-level reasoning plan for LLMs. Indeed, these methods have demonstrated that they can obtain great capabilities over PoT. 

## **A.14 Instruction Tuning in LLMs** 

To align language models with human preferences and effective objectives, some works [17; 72; 66] design instruction tuning. The instruction tuning aims to mine the potential capabilities by aligning with and responding to human preference. Early on, instruction tuning focuses on improving the instruction-following abilities of LLMs for general purposes. Represented by FLAN [73] and T0 [74], they aim to understand the generalization abilities of LLMs for instruction tuning. Later, to comprehend the efficacy and performance of enlarging the instructional tuning datasets on models, FLAN-v2 [75; 76] has been proposed to validate this goal. However, these methods build training instruction tuning datasets in a human-annotated manner. Recently, various studies [77; 72; 78; 17; 39; 79] have started to construct synthetic instruction following datasets distilled from some LLMs like GPT-3.5-turbo and GPT-4. Like these works, Platypus [80] constructs a small-scale instruction for the following dataset by utilizing a domain-specialized dataset, aiming to enhance the reasoning capabilities of LLMs. 

27 

## **Part II** 

## **Supplementary Materials** 

## **B Datasheet** 

## **B.1 Motivation** 

**1. For what purpose was the dataset created?** _Our benchmark dataset was created to address the data scarcity challenge in the science domain and provide a suite of scientific instructions for training scientific language models capable of college-level scientific reasoning._ 

**2. Who created the dataset and on behalf of which entity?** _The dataset was developed by LLM researchers (undergraduate students, doctoral students, and postdocs) listed in the author list._ 

**3. Who funded the creation of the dataset?** _This work is supported by the NSFC 62276148, NSFC for Distinguished Young Scholar 62425601, a research fund from Zhipu, New Cornerstone Science Foundation through the XPLORER PRIZE and Tsinghua University (Department of Computer Science and Technology) - Siemens Ltd., China Joint Research Center for Industrial Intelligence and Internet of Things (JCIIOT)._ 

## **B.2 Distribution** 

**1. Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** _Yes, the dataset is open to the public._ 

**2. How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?** _The dataset has been distributed through Hugging Face, Google Drive, Tsinghua Cloud, and the code used for developing baseline models through GitHub._ 

**3. Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** _No._ 

**4. Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** _No._ 

**5. When will the dataset be distributed?** _It has been released now._ 

**6. Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** _The dataset will be distributed under the CC BY 4.0 license._ 

## **B.3 Maintenance** 

**1. Who will be supporting/hosting/maintaining the dataset?** _Zhipu AI and THUDM will support, host, and maintain the dataset._ 

**2. How can the owner/curator/manager of the dataset be contacted (e.g., email address)?** _The owner/curator/manager(s) of the dataset can be contacted through the following emails: Dan Zhang (zd18@tsinghua.org.cn) and Sining Zhoubian (zbsn21@mails.tsinghua.edu.cn)._ 

**3. Is there an erratum?** _No. If errors are found in the future, we will release errata on the main web page for the dataset (_ https://github.com/THUDM/SciGLM _and_ https://huggingface.co/ datasets/zd21/SciInstruct _)._ 

**4. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?** _Yes, the datasets will be updated whenever necessary to ensure accuracy, and announcements will be made accordingly. These updates will be posted on the main web page for the dataset (_ https://github.com/THUDM/SciGLM _and_ https://huggingface.co/datasets/zd21/ SciInstruct _)._ 

28 

**5. If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted?)** _N/A_ 

**6. Will older version of the dataset continue to be supported/hosted/maintained?** _Yes, older versions of the dataset will continue to be maintained and hosted._ 

**7. If others want to extend/augment/build on/contribute to the dataset, is there a mechanisms for them to do so?** _If others want to extend/augment/build on/contribute to the dataset, the most efficient way to reach us is via GitHub pull requests. For more questions, don’t hesitate to get in touch with Dan Zhang (zd18@tsinghua.org.cn), who will be responsible for maintenance._ 

## **B.4 Composition** 

**1. What do the instance that comprise the dataset represent (e.g., documents, photos, people, countries?)** _Each instance includes a question, corresponding answer, and subject. These attributes are used to fine-tune LLMs capable of college-level scientific reasoning._ 

**2. How many instances are there in total (of each type, if appropriate)?** _The constructed SciInstruct dataset comprises 254,051 instructions, including 123,869 physics and chemistry, 89,934 math, and 40,248 formal proof (Lean)._ 

**3. Does the dataset contain all possible instances or is it a sample of instances from a larger set?** _The datasets contain all possible instances._ 

**4. Is there a label or target associated with each instance?** _Yes, each instance includes the corresponding subject._ 

**5. Is any information missing from individual instances?** _No._ 

**6. Are there recommended data splits (e.g., training, development/validation, testing)?** _We use all instances as a training set because there are various public benchmark datasets as test sets that cover a broad range of subjects._ 

**7. Are there any errors, sources of noise, or redundancies in the dataset?** _No._ 

**8. Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** _The dataset is self-contained._ 

**9.** _No._ **Does the dataset contain data that might be considered confidential?** 

**10. Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** _No._ 

## **B.5 Collection Process** 

**1. How was the data associated with each instance acquired?** _The data associated with each instance is acquired from a variety of sources, including textbooks, pedagogical materials, and problem sets. References for these sources are provided in Table 7 in Appendix 7.2._ 

**2. What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** _We first use Optical character recognition (OCR) to transform the question and final answer in the collected materials, then utilize a powerful LLM API to generate the intermediate reasoning, and finally filter out the traces with a correct predicted answer._ 

**3. Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?** _Regular LLM researchers (e.g., undergraduate students, PhD students, and postdocs listed in the author list) at Tsinghua and Caltech were involved in the data collection process._ 

**4. Does the dataset relate to people?** _No._ 

**5. Did you collect the data from the individuals in questions directly, or obtain it via third parties or other sources (e.g., websites)?** _We obtained the dataset from textbooks, pedagogical materials, and problem sets. References for these sources are provided in Table 7 in Appendix 7.2._ 

29 

## **B.6 Uses** 

**1. Has the dataset been used for any tasks already?** _Yes, this dataset has been used to generate new datasets, for example, to convert the text-formatted calculations into code formats and to fine-tune the code-based language models._ 

**2. What (other) tasks could be the dataset be used for?** _No._ 

**3. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** _The current composition of the datasets is self-sufficient to train a scientific language model. Any changes in the next release and updates will be documented and shared through the dataset webpage (_ https://github.com/THUDM/ SciGLM _and_ https://huggingface.co/datasets/zd21/SciInstruct _)._ 

**4. Are there tasks for which the dataset should not be used?** _No._ 

30 

## **Checklist** 

1. For all authors... 

   - (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] See Section 1. 

   - (b) Did you describe the limitations of your work? [Yes] See Section 5. 

   - (c) Did you discuss any potential negative societal impacts of your work? [Yes] See Section 5. 

   - (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes] We have read the ethics review guidelines and ensured that your paper conforms to them. 

2. If you are including theoretical results... 

   - (a) Did you state the full set of assumptions of all theoretical results? [N/A] 

   - (b) Did you include complete proofs of all theoretical results? [N/A] 

3. If you ran experiments (e.g. for benchmarks)... 

   - (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] We have provided a URL. 

   - (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] See Section 2.4. 

   - (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [N/A] 

   - (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] See Section 2.4. 

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets... 

   - (a) If your work uses existing assets, did you cite the creators? [Yes] See Section 2.4. 

   - (b) Did you mention the license of the assets? [Yes] See Section 2.4. 

   - (c) Did you include any new assets either in the supplemental material or as a URL? [Yes] We have provided a URL. 

   - (d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [N/A] 

   - (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A] 

5. If you used crowdsourcing or conducted research with human subjects... 

   - (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A] 

   - (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A] 

   - (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A] 

31 

