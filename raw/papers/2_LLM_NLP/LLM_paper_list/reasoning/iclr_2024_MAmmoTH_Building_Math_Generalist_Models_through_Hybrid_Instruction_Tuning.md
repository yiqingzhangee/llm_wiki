Preprint. Work in Progress 

# MAMMOTH: BUILDING MATH GENERALIST MODELS THROUGH HYBRID INSTRUCTION TUNING 

♣ ∗ ‡ † ○ § **Xiang Yue , Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang,** 

> ♣ **Huan Sun,** ♣ **Yu Su,** † **Wenhu Chen** ∗ 

†University of Waterloo, ♣The Ohio State University, ‡HKUST, ○University of Edinburgh, §01.AI yue.149@osu.edu, wenhuchen@uwaterloo.ca 

https://tiger-ai-lab.github.io/MAmmoTH/ 

## ABSTRACT 

We introduce MAmmoTH, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MAmmoTH models are trained on MathInstruct, our meticulously curated instruction tuning dataset. MathInstruct is compiled from 13 math datasets with intermediate rationales, six of which have rationales newly curated by us. It presents a unique hybrid of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT not only unleashes the potential of tool use but also allows different thought processes for different math problems. As a result, the MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%. Remarkably, our MAmmoTH-7B model reaches 33% on MATH (a competition-level dataset), which exceeds the best open-source 7B model (WizardMath) by 23%, and the MAmmoTH-34B model achieves 44% accuracy on MATH, even surpassing GPT4’s CoT result. Our work underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning_images/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning.pdf-0001-09.png)


**----- Start of picture text -----**<br>
MathInstruct Hybrid Instruction Tuning<br>Chain-of-Thought (CoT)<br>Weng earns $12 an hour for babysitting.<br>Weng earns 12/60 = 0.2 per minute.<br>Yesterday, she just did 50 minutes of<br>Doing 50 mins, she earned 0.2 x 50 = 10<br>babysitting. How much did she earn?<br>Program-of-Thought (PoT)<br>hourly_rate = 12; time_worked = 50/60;<br>earnings = hourly_rate * time_worked;<br>MAmmoTH print(round(earnings, 2))<br>Diverse Math Problems<br>In-domain datasets Out-of-domain datasets<br>80 Base SoTA MAmmoTH (Ours) 80 Base SoTA MAmmoTH (Ours)<br>+28 +16 +32 +21<br>60 +20 +19 60 +24 +26<br>40 40<br>20 20<br>0 0<br>7B 13B 30B 70B 7B 13B 30B 70B<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: The superior performance of MAmmoTH, a series of models instruction-tuned to solve a diverse set of mathematical problems using hybrid CoT and PoT rationales. MAmmoTH significantly outperforms base and SoTA models on both in-domain and out-of-domain test sets, across all scales. 

> ∗Xiang Yue and Wenhu Chen are the leading authors of the paper. They contributed equally to this project. 

1 

Preprint. Work in Progress 

## 1 INTRODUCTION 

This work focuses on mathematical reasoning, a critical capability of modern large language models (LLMs) (OpenAI, 2023; Anil et al., 2023). Despite the recent advances in this field, a noticeable gap exists between closed-source and open-source LLMs—closed-source models like GPT-4 (OpenAI, 2023), PaLM-2 (Anil et al., 2023), and Claude 2 (Bai et al., 2022) dominate popular mathematical reasoning benchmarks such as GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b), while open-source models like Llama (Touvron et al., 2023a;b), Falcon (Penedo et al., 2023), OPT (Zhang et al., 2022) lag behind on all benchmarks by a wide margin. 

Current efforts to bridge this gap are twofold: (1) _Continued pre-training_ like Galactica (Taylor et al., 2022) and MINERVA (Lewkowycz et al., 2022), which continues to train an LLM on math-related web data of more than 100B tokens. This approach improves a model’s general scientific reasoning capability but incurs a high computation cost. (2) _Dataset-specific fine-tuning_ like rejection sampling fine-tuning (RFT) (Yuan et al., 2023) and WizardMath (Luo et al., 2023), which fine-tunes LLMs using supervised data specific to certain datasets. Although such approaches improve in-domain performance, they cannot generalize to a wider range of math reasoning tasks beyond their finetuning data. For instance, both RFT and WizardMath can increase the accuracy on GSM8K (Cobbe et al., 2021) by 30%+, one of their fine-tuning datasets, but hurt the accuracy on out-of-domain datasets like MMLU-Math (Hendrycks et al., 2021a) or AQuA (Ling et al., 2017) by up to 10%. 

In this paper, we aim to propose a lightweight yet generalizable math instruction-tuning approach to enhance the general (i.e., not limited to the fine-tuning tasks) mathematical reasoning capabilities of LLMs. Existing methods (Luo et al., 2023; Yuan et al., 2023; Taylor et al., 2022) primarily focus on Chain-of-Thought (CoT) approaches (Wei et al., 2022b; Nye et al., 2022) to solve math problems through step-by-step natural language descriptions. This approach excels in its generality to cover most math subjects but struggles with computation precision, and complex mathematical or algorithmic reasoning procedures (e.g., solving quadratic equation roots and calculating matrix eigenvalues). In contrast, prompts in the format of code like Program-of-Thought (PoT) approaches (Chen et al., 2022) and PAL (Madaan et al., 2022; Gao et al., 2023) utilize external tools (i.e., Python interpreter) to greatly simplify the math solving process. This approach advocates offloading the computation process to the external Python interpreter to solve complex mathematical and algorithmic reasoning procedures (e.g., solving quadratic equations with sympy or calculating matrix eigenvalues with numpy). However, PoT falls short in dealing with more abstract reasoning scenarios, like commonsense reasoning, formal logic, and abstract algebra, especially when there exist no built-in APIs. 

To leverage the strengths of both CoT and PoT approaches, we introduce a new math hybrid instruction-tuning dataset MathInstruct, which has two main characteristics: (1) **broad coverage of different math fields and complexity levels** , and (2) **hybrid CoT & PoT rationales** . MathInstruct is based on seven existing math rationale datasets and six newly-curated datasets (see details in Table 1). We use MathInstruct to fine-tune Llama (Touvron et al., 2023a;b; Rozi`ere et al., 2023) models of different scales ranging from 7B to 70B. The resulting MAmmoTH models ( Figure 1) demonstrate unprecedented potential in serving as math generalists. 

We evaluate MAmmoTH on a spectrum of datasets, including in-domain (IND) test sets— GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), AQuA-RAT (Ling et al., 2017), NumGLUE (Mishra et al., 2022b)—and out-of-domain (OOD) test sets—SVAMP (Patel et al., 2021), SAT (Zhong et al., 2023), MMLU-Math (Hendrycks et al., 2021a), Mathematics (Davies et al., 2021), and SimulEq (Koncel-Kedziorski et al., 2016). Compared with existing methods, our models generalize better to OOD datasets and substantially improve the performance of open-source LLMs in mathematical reasoning. Notably, on the popular competition-level MATH dataset (Hendrycks et al., 2021b), our 7B model can beat WizardMath (open-source MATH SoTA) (Luo et al., 2023) by 3.5x (35.2% vs 10.7%), and our 34B MAmmoTH-Coder (fine-tuned on Code Llama (Rozi`ere et al., 2023)) can even beat the result of GPT-4 (using CoT). 

We highlight our contributions from two perspectives: (1) From the **data engineering perspective** , we present MathInstruct, a high-quality math instruction tuning dataset, combining a variety of math problems and hybrid rationales. (2) From the **modeling perspective** , we investigate the impact of various data sources and input-output formats through training and evaluating over 50 different models and baselines ranging from 7B to 70B. Our models, including MAmmoTH and MAmmoTH-Coder, achieve substantial accuracy gains over existing open-source models. 

2 

Preprint. Work in Progress 

|Training Dataset|Type|Annotation|# Samples|Characteristics|Fields|
|---|---|---|---|---|---|
|GSM8K (Cobbe et al., 2021)|CoT|Human|7K|Grade Schol Exam|∎|
|GSM8K-RFT (Yuan et al., 2023)|CoT|Llama|28K|Llama + Validated|∎|
|AQuA-RAT (Ling et al., 2017)|CoT|Human|90K|GRE/GMAT Exam|∎|
|MATH (Hendrycks et al., 2021b)|CoT|Human|7K|Math Competition|∎∎∎∎∎∎∎|
|TheoremQA (Chen et al., 2023)�|CoT|GPT-4|600|GPT4 + Validated|∎∎∎∎∎|
|Camel-Math (Li et al., 2023a)|CoT|GPT-4|50K|GPT4 (Unvalidated)|∎∎∎∎∎|
|College-Math�|CoT|GPT-4|1.8K|GPT4 (Unvalidated)|∎|
|GSM8K�|PoT|GPT4|14K|GPT4 + Validated|∎|
|AQuA-RAT�|PoT|GPT4|9.7K|GPT4 + Validated|∎|
|MATH�|PoT|GPT4|7K|GPT4 + Validated|∎∎∎∎|
|TheoremQA�|PoT|GPT4|700|GPT4 + Validated|∎∎∎∎∎|
|MathQA (Amini et al., 2019)|PoT|Human|25K|AQuA-RAT Subset|∎|
|NumGLUE (Mishra et al., 2022a)|PoT|Human|13K|Lila Annotated|∎|
|MathInstruct|||260K||∎∎∎∎∎∎∎|



Table 1: Overview of our MathInstruct. �means with NEW rationales curated by us by prompting GPT-4. We have filtered out augmented samples that have answers inconsistent with the original dataset’s annotations. Different colored squares represent different fields in mathematics: ∎ Pre-Algebra; ∎ Inter-Algebra; ∎ Algebra; ∎ Probability; ∎ NumTheory; ∎ Calculus; ∎ Geometry. 

## 2 OUR APPROACH 

## 2.1 BACKGROUND 

Mathematical reasoning serves as a vital gauge for assessing the ability of LLMs to execute complex multi-hop and quantitative reasoning. Previously, this has been a challenging task for neural networks, which struggle to solve even basic addition and subtraction problems (Yang et al., 2023). However, recent LLMs have considerable advancements in mathematical reasoning. Key breakthroughs have been made through CoT prompting (Wei et al., 2022b; Nye et al., 2022) and PoT prompting (Chen et al., 2022; Gao et al., 2023). CoT prompting encourages LLMs to solve problems incrementally on a scratchpad, enhancing both accuracy and explainability in mathematical reasoning. This approach contrasts with traditional methods that generate answers directly. PoT prompting, on the other hand, formulates the intermediate reasoning process as a program, executed with an external tool like Python, to compute the answer. This method improves robustness in solving complex mathematical problems by offloading the calculations to external tools. However, most existing work (Zhou et al., 2023a) in PoT is limited to proprietary models like GPT-4 (OpenAI, 2023) and Codex (Chen et al., 2021). The PoT potential of open-source models is yet to be seen. Our work aims at optimizing LLMs’ CoT and PoT reasoning capabilities through instruction tuning. 

## 2.2 CURATING A DIVERSE AND HYBRID INSTRUCTION TUNING DATASET 

Our study aims to compile a list of high-quality and diverse math instruction-tuning datasets, standing out with three main characteristics: (1) broad coverage of different mathematical fields and complexity levels, and (2) hybrid CoT & PoT rationales. 

**Broad Coverage of Different Math Fields and Complexity Levels:** We aim for a broad representation of math fields and complexity levels in our dataset. This ensures exposure to a diverse set of mathematical knowledge, fostering versatility in our models. Based on these criteria, we narrow down our choices to a few high-quality datasets that are widely adopted and encompass different math fields and complexity levels, such as GSM8K, MATH, AQuA, Camel, and TheoremQA. Furthermore, we notice a lack of coverage for college-level math knowledge, such as abstract algebra and formal logic, in existing datasets. To rectify this, we use GPT-4 to synthesize CoT rationales for questions in TheoremQA and create question-CoT pairs through Self-Instruct (Wang et al., 2023h), utilizing a few seed exemplars found online. 

3 

Preprint. Work in Progress 

**Hybrid CoT and PoT Rationales:** Contrary to previous work (Yuan et al., 2023; Luo et al., 2023; Lee et al., 2023; Wang et al., 2023g) that focus on CoT, our dataset strategically combines both. This integration enhances the dataset’s versatility, catering to varying mathematical problem-solving approaches. However, most existing datasets provide limited program rationales, leading to an imbalance between CoT and PoT rationales. To fill the gap, we utilize GPT-4 to supplement the PoT rationales for selected datasets, including MATH, AQuA, GSM8K, and TheoremQA. We then filter these GPT-4 synthesized programs by comparing their executed results with human-annotated ground truth, which ensures the high quality of the added rationales. 

Following these guidelines, our instruction dataset, detailed in Table 1, encompasses 260K (instruction, response) pairs, covering a wide range of core mathematical fields (arithmetic, algebra, probability, calculus, and geometry, etc.), including hybrid CoT and PoT rationales, and offering diversity in both language and difficulty levels. This attests to its high quality and unique characteristics. 

## 2.3 TRAINING SETUP 

We unify all the subsets in our MathInstruct to conform to the structure of an Alpaca-like instruction dataset (Taori et al., 2023). This standardization ensures that the fine-tuned models can process data consistently, regardless of the original dataset formats. We choose the open-source models Llama-2 (Touvron et al., 2023b) and Code Llama (Rozi`ere et al., 2023) as our base models. We fine-tune these models including 7B, 13B, 34B, and 70B on MathInstruct, which allows us to validate our MathInstruct at multiple scales. We fine-tune all the models with Huggingface transformers library (Wolf et al., 2019). We use a learning rate of 2e-5 for the 7B and 13B models, and 1e-5 for the 34B and 70B models. We set the batch size at 128 and used a cosine scheduler with a 3% warm-up period for three epochs. To efficiently train the computationally intensive 34B and 70B models, we employ DeepSpeed training with ZeRO-3 stage (Rajbhandari et al., 2020). 

## 2.4 EVALUATION SETUP 

Our hybrid training enables models to solve problems using either the CoT or PoT approach. By default, the model provides the CoT solution. To switch to the PoT approach, one can add the trigger phrase “Let’s write a program to solve the problem” following the question. 

Our preliminary evaluation reveals that PoT generally outperforms CoT, notably in open-form questions like GSM8K and MATH, as programmable solutions are better at solving complex mathematical and algorithmic reasoning procedures. However, PoT struggles with abstract reasoning scenarios such as commonsense reasoning, formal logic, and abstract algebra, particularly in the absence of built-in APIs. To further combine the power of both approaches, we introduce a simple hybrid decoding strategy: The model first attempts PoT prompting. If the program is not executable, we falls back to CoT prompting. This heuristic significantly enhances our model’s overall performance (see more discussions in section 3.4). 

## 3 EXPERIMENTS 

## 3.1 EVALUATION DATASETS 

We have selected diverse evaluation datasets (Table 2), encompassing a variety of **in-domain** and **out-of-domain** samples across diverse fields of mathematics, to assess the models’ capabilities in general mathematical reasoning. 

For the in-domain datasets, we consider GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), AQuA-RAT (Ling et al., 2017), and NumGLUE (Mishra et al., 2022b). For the out-of-domain datasets, we choose SVAMP (Patel et al., 2021), Mathematics (Davies et al., 2021), SimulEq (Koncel-Kedziorski et al., 2016), SAT-Math (Zhong et al., 2023), and MMLUMath (Hendrycks et al., 2021a). The wide selection of evaluation datasets includes math problems from elementary, high school, and college levels. Some of the datasets even include formal logic and commonsense reasoning. The choice of these datasets is to ensure a comprehensive evaluation of the models’ capabilities to generalize to unfamiliar situations and different math fields. The chosen evaluation datasets consist of both open-formed questions and multi-choice questions. 

4 

Preprint. Work in Progress 

|Eval Dataset|# Samples|In-Domain?|Answer Form|Fields|
|---|---|---|---|---|
|GSM8K (Cobbe et al., 2021)|1319|YES|Open-formed|∎|
|MATH (Hendrycks et al., 2021b)|5000|YES|Open-formed|∎∎∎∎∎∎∎|
|AQuA-RAT (Ling et al., 2017)|254|YES|Multi-choice|∎|
|NumGLUE (Mishra et al., 2022b)|1042|YES|Open-formed|∎|
|SVAMP (Patel et al., 2021)|1000|NO|Open-formed|∎|
|Mathematics (Davies et al., 2021)|1000|NO|Open-formed|∎∎∎∎|
|SimulEq (Koncel-Kedziorski et al., 2016)|514|NO|Open-formed|∎|
|SAT-Math (Zhong et al., 2023)|220|NO|Multi-choice|∎∎∎|
|MMLU-Math (Hendrycks et al., 2021a)|974|NO|Multi-choice|∎∎∎∎|



Table 2: Comprehensive overview of our evaluation datasets, featuring a variety of in-domain and out-of-domain problems across diverse fields of mathematics. Different colored squares represent different fields in mathematics: ∎ Pre-Algebra; ∎ Inter-Algebra; ∎ Algebra; ∎ Probability; ∎ NumTheory; ∎ Calculus; ∎ Geometry. 

## 3.2 BASELINES 

We partition our baselines into the following four categories: 

- **Closed-source LLMs:** We consider 4 closed-source LLMs including GPT-4 (OpenAI, 2023), GPT-4 (Code Interpreter), PaLM-2 Unicorn (Anil et al., 2023), Claude-2 (Bai et al., 2022) and Codex (Chen et al., 2021). GPT-4, PaLM-2, and Claude-2 use CoT prompting while GPT-4 (Code Interpreter) and Codex use PoT prompting. 

- **Llama Base:** For the base models, we consider Llama-1/2 (Touvron et al., 2023a;b), Llama-2Chat (Touvron et al., 2023b). 

- **Coder Model:** To compare with different coder models, we choose Code-Llama (Rozi`ere et al., 2023), CodeT5+ (Wang et al., 2023i) and CodeGen (Nijkamp et al., 2023). 

- **STEM Pre-training:** We cover Galactica (Taylor et al., 2022) mainly to understand the performance of models specialized in STEM knowledge. 

- **Instruction Tuning:** We include Orca-Platypus (Mukherjee et al., 2023), Vicuna-1.5 (Zheng et al., 2023b), Tulu (Wang et al., 2023g), Platypus-2 (Lee et al., 2023) and Guanaco (Dettmers et al., 2023). We cover a wide spectrum of models trained with different types of datasets. 

- **Dataset-Specific Tuning:** We include both RFT (Yuan et al., 2023) and WizardMath (Luo et al., 2023), which specifically tune the models to adapt to GSM8K and MATH datasets. We include them to understand their generalization. 

For most baselines, we choose CoT prompting to maximize their performance due to their incompetence in program generation. All the ‘Code Model’ use PoT prompting. For GSM8K, MATH, AQuA, and NumGLUE, we will evaluate both 8-shot in-context-learning and zero-shot setups to report the highest score. For SVAMP, Mathematics, SimulEq, SAT, and MMLU, we use 5-shot in-context-learning to maintain consistency with prior work (Wei et al., 2022b; Chen et al., 2023). Our few-shot exemplars are mostly taken from PHP[1] (Zheng et al., 2023a). For MAmmoTH and MAmmoTH-Coder, we always evaluate under 0-shot setting. For all models, we allow a maximum sequence length of 2048 tokens for decoding. For multiple-choice questions, if the generated answer lacks an option, we map it by re-prompting the model: “Please find the closest option to [generated answer]. The options are [options]”. 

## 3.3 MAIN RESULTS 

We report our in-domain and out-of-domain results in Table 3 and Table 4 respectively. Overall, we can see that MAmmoTH and MAmmoTH-Coder are able to outperform the SoTA model at different scales. In general, the performance gain for OOD datasets is more significant than IND datasets. These results show us the potential of our models as a mathematical generalist. On several datasets, MAmmoTH-Coder-34B and MAmmoTH-70B are even surpassing closed-source LLMs. 

> 1https://github.com/chuanyang-Zheng/Progressive-Hint 

5 

Preprint. Work in Progress 

|Model<br>Base<br>Math-SFT?<br>GSM8K<br>MATH<br>AQuA<br>NumGLUE|Avg|
|---|---|
|Closed-source Model||
|GPT-4<br>-<br>Unknown<br>92.0†<br>42.5†<br>72.6†<br>-<br>GPT-4 (Code-Interpreter)<br>-<br>Unknown<br>97.0†<br>69.7†<br>-<br>-<br>PaLM-2<br>-<br>Unknown<br>80.7†<br>34.3†<br>64.1<br>-<br>Claude-2<br>-<br>Unknown<br>85.2†<br>32.5†<br>60.9<br>-<br>Codex (PoT)<br>-<br>No<br>71.6†<br>36.8†<br>54.1†<br>-<br>ART (InstructGPT)<br>-<br>Unknown<br>71.0<br>-<br>54.2<br>-|-<br>-<br>-<br>-<br>-<br>-|
|7B Parameter Model||
|Llama-1<br>-<br>No<br>10.7†<br>2.9†<br>22.6<br>24.7<br>Llama-2<br>-<br>No<br>14.6†<br>2.5†<br>30.3<br>29.9<br>Galactica-6.7B<br>GAL<br>GAL-Instruct<br>10.2<br>2.2<br>25.6<br>25.8<br>Code-Llama (PoT)<br>-<br>No<br>25.2<br>13.0<br>24.0<br>26.8<br>AQuA-SFT<br>Llama-2<br>AQuA<br>11.2<br>3.6<br>35.6<br>12.2<br>Llama-1 RFT<br>Llama-1<br>GSM8K<br>46.5†<br>5.2<br>18.8<br>21.1<br>WizardMath<br>Llama-2<br>GSM8K+MATH<br>54.9†<br>10.7†<br>26.3<br>36.1|15.5<br>19.3<br>15.9<br>22.2<br>15.6<br>22.9<br>32.0|
|MAmmoTH<br>Llama-2<br>MathInstruct<br>53.6<br>31.5<br>44.5<br>61.2<br>MAmmoTH-Coder<br>Code-Llama<br>MathInstruct<br>59.4<br>33.4<br>47.2<br>66.4|47.7<br>51.6|
|∆<br>+5<br>+21<br>+12<br>+30|**+20**|
|13-15B Parameter Model||
|Llama-1<br>-<br>No<br>17.8†<br>3.9†<br>26.0<br>24.8<br>Llama-2<br>-<br>No<br>28.7†<br>3.9†<br>25.1<br>8.8<br>Code-Llama (PoT)<br>-<br>No<br>36.1<br>16.4<br>28.7<br>29.2<br>CodeT5+ (PoT)<br>-<br>No<br>12.5<br>2.4<br>20.5<br>19.4<br>CodeGen+ (PoT)<br>-<br>No<br>12.7<br>3.4<br>24.5<br>22.5<br>Vicuna-1.5<br>Llama-2<br>No<br>28.4†<br>5.8<br>24.8<br>36.9<br>Llama-1 RFT<br>Llama-1<br>GSM8K<br>52.1†<br>5.1<br>16.1<br>24.5<br>Orca-Platypus<br>Llama-2<br>Platypus<br>38.4<br>3.0<br>18.9<br>35.3<br>Platypus<br>Llama-2<br>Platypus<br>25.7<br>2.5<br>33.4<br>42.3<br>WizardMath<br>Llama-2<br>GSM8K+MATH<br>63.9†<br>14.0†<br>21.2<br>40.8|18.1<br>16.6<br>27.6<br>13.7<br>15.7<br>23.9<br>24.4<br>23.9<br>25.9<br>34.9|
|MAmmoTH<br>Llama-2<br>MathInstruct<br>62.0<br>34.2<br>51.6<br>68.7<br>MAmmoTH-Coder<br>Code-Llama<br>MathInstruct<br>64.7<br>36.3<br>46.9<br>66.8|54.1<br>53.7|
|∆<br>+1<br>+20<br>+18<br>+26|**+19**|
|30-34B Parameter Model||
|Llama-1<br>-<br>No<br>35.6†<br>7.1†<br>33.4<br>28.4<br>Code-Llama (PoT)<br>-<br>No<br>44.0<br>23.1<br>25.2<br>29.3<br>Llama-1 RFT<br>Llama-1<br>GSM8K<br>56.5†<br>7.4†<br>18.5<br>24.3<br>Galactica-30B<br>GAL<br>GAL-Instruct<br>41.7<br>12.7<br>28.7<br>34.7<br>Platypus<br>Llama-1<br>Platypus<br>37.8<br>9.3<br>27.9<br>40.5<br>Tulu<br>Llama-2<br>Tulu<br>51.0<br>10.8<br>25.5<br>43.4|26.1<br>30.4<br>26.6<br>29.4<br>28.8<br>32.6|
|MAmmoTH-Coder<br>Code-Llama<br>MathInstruct<br>72.7<br>43.6<br>54.7<br>71.6|60.7|
|∆<br>+16<br>+21<br>+21<br>+28|**+28**|
|65-70B Parameter Model||
|Llama-1<br>-<br>No<br>50.9†<br>10.6†<br>35.0<br>50.2<br>Llama-2<br>-<br>No<br>56.8†<br>13.5†<br>40.9<br>50.4<br>Llama-2-Chat<br>Llama-2<br>No<br>54.9<br>18.6<br>37.0<br>51.6<br>Guanaco<br>Llama-2<br>No<br>59.2<br>4.1<br>45.2<br>53.5<br>WizardMath<br>Llama-2<br>GSM8K+MATH<br>81.6†<br>22.7†<br>20.0<br>48.9<br>Platypus<br>Llama-2<br>Platypus<br>70.6<br>15.6<br>51.2<br>55.4|36.6<br>40.4<br>40.5<br>40.5<br>43.3<br>48.1|
|MAmmoTH<br>Llama-2<br>MathInstruct<br>76.9<br>41.8<br>65.0<br>74.4|64.5|
|∆<br>-5<br>+19<br>+14<br>+19|**+16**|



Table 3: The table compiles all the **in-domain** evaluation results. Results marked as † are copied from other papers, which can be found on paperswithcode leaderboards. Math-SFT? means whether the model has been instruction-tuned on any math reasoning datasets. Pink numbers highlight the highest number within the corresponding scale and dataset. Note that there does not exist a 30B+ version for Llama-2 or a 70B version for Code-Llama. 

From Table 3, we can observe that our main competitors for IND datasets are WizardMath (Luo et al., 2023) and Platypus (Lee et al., 2023). WizardMath’s training is heavily rooted in GSM8K 

6 

Preprint. Work in Progress 

|Model<br>SVAMP<br>Mathematics<br>SimulEq<br>SAT-Math<br>MMLU-Math|Avg|
|---|---|
|Closed-source Model||
|||
|GPT-4<br>97.0†<br>-<br>-<br>95†<br>-<br>Codex (PoT)<br>85.2†<br>-<br>-<br>68†<br>-|-<br>-|
|7B Parameter Model||
|||
|Llama-1<br>24.5<br>6.2<br>4.6<br>22.7<br>30.6<br>Llama-2<br>34.5<br>6.0<br>5.0<br>26.8<br>29.8<br>Code-Llama (PoT)<br>49.4<br>21.7<br>3.5<br>28.6<br>26.9<br>Llama-1 RFT<br>21.1<br>5.1<br>11.0<br>12.5<br>21.7<br>Galactica-6.7B<br>25.6<br>4.6<br>4.2<br>17.5<br>28.0<br>WizardMath<br>36.1<br>9.3<br>12.8<br>25.4<br>31.1<br>Toolformer<br>29.4†<br>-<br>-<br>-<br>-|17.7<br>20.4<br>26.0<br>14.3<br>16.0<br>28.6<br>-|
|||
|MAmmoTH<br>67.7<br>46.3<br>41.2<br>42.7<br>42.6<br>MAmmoTH-Coder<br>71.4<br>55.4<br>45.9<br>40.5<br>48.3|48.1<br>52.3|
|∆<br>+22<br>+34<br>+33<br>+14<br>+17|**+24**|
|13B Parameter Model||
|||
|Llama-1<br>34.7<br>6.9<br>5.4<br>27.7<br>30.7<br>Llama-2<br>35.1<br>11.5<br>5.8<br>32.7<br>34.4<br>Code-Llama (PoT)<br>60.0<br>21.3<br>3.8<br>25.9<br>27.7<br>Vicuna-1.5<br>55.7<br>10<br>6.6<br>34.0<br>34.1<br>Llama-1 RFT<br>46.5<br>6.7<br>10.1<br>13.2<br>21.6<br>WizardMath<br>51.9<br>14.1<br>14.9<br>24.5<br>32.1<br>Platypus<br>55.4<br>11.4<br>7.4<br>36.8<br>35.5<br>Orca-Platypus<br>56.8<br>12.6<br>7.9<br>29.5<br>41.6|21.0<br>23.9<br>27.7<br>28.1<br>19.6<br>27.5<br>29.3<br>29.7|
|||
|MAmmoTH<br>72.4<br>49.2<br>43.2<br>46.8<br>47.6<br>MAmmoTH-Coder<br>73.7<br>61.5<br>47.1<br>48.6<br>48.3|51.8<br>55.8|
|∆<br>+14<br>+40<br>+33<br>+12<br>+7|**+26**|
|30-34B Parameter Model||
|||
|Llama-1<br>48.8<br>12.8<br>11.2<br>33.4<br>39.0<br>Code-Llama (PoT)<br>69.1<br>34.5<br>6.8<br>26.8<br>21.6<br>Llama-1 RFT<br>55.4<br>7.6<br>12.8<br>20.4<br>37.9<br>Galactica-30B<br>41.6<br>11.8<br>13.2<br>37.7<br>37.9<br>Tulu<br>59.0<br>10.7<br>10.3<br>31.3<br>39.8<br>Platypus<br>51.7<br>13.8<br>13.6<br>38.6<br>41.0|29.0<br>31.7<br>26.8<br>28.4<br>30.2<br>31.7|
|||
|MAmmoTH-Coder<br>84.3<br>65.4<br>51.8<br>60.9<br>53.8|63.2|
|∆<br>+15<br>+31<br>+38<br>+22<br>+13|**+32**|
|65-70B Parameter Model||
|||
|Llama-1<br>55.3<br>14.2<br>15.2<br>37.4<br>44.1<br>Llama-2<br>63.8<br>20.5<br>14.0<br>51.3<br>47.1<br>Llama-2-Chat<br>71.5<br>19.2<br>21.7<br>44.1<br>46.9<br>WizardMath<br>71.8<br>17.1<br>37.9<br>13.2<br>27.4<br>Guanaco<br>66.8<br>17.8<br>20.2<br>50.0<br>47.3<br>Platypus<br>51.8<br>26.3<br>21.7<br>55.9<br>52.5|33.2<br>39.3<br>40.6<br>33.4<br>40.4<br>41.6|
|||
|MAmmoTH<br>82.4<br>55.6<br>51.4<br>66.4<br>56.7|62.5|
|∆<br>+11<br>+29<br>+14<br>+11<br>+4|**+21**|



Table 4: The table compiles all the **out-of-domain** evaluation results. Results marked as † are copied from other papers, which can be found on paperswithcode leaderboards. 

and MATH datasets. Therefore, WizardMath’s results are highly competitive on these two datasets. However, the dataset-specific training can be detrimental to OOD datasets like AQuA. In contrast, Platypus fine-tunes LLMs on a wide range of text and math reasoning datasets. it improves the opensource SoTA on several datasets. Similarly, MAmmoTH can achieve universal improvement across the board. A major observation is that MAmmoTH is particularly strong at solving more complex math problems in MATH, where the gain of our model over WizardMath (open-source SoTA on MATH) can exceed 25% at different scales. 

7 

Preprint. Work in Progress 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning_images/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
60.0<br>48.1 47.9<br>39.9 42.6 41.6 41.0<br>40.0 32.8 29.6 29.8 27.0 32.0<br>20.4 22.9 19.9<br>20.0<br>8.6<br>0.0<br>GSM + MATH Out-of-domain Overall<br>Llama-2 Base WizardMath (GSM + MATH CoT) MAmmoTH (MathInstruct - CoT)<br>MAmmoTH (MathInstruct - PoT) MAmmoTH (MathInstruct - Hybrid)<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 2: Investigation of the influence of CoT & PoT hybrid training on the 7B Llama-2 model. “Out-of-domain” refers to the five datasets detailed in Table 2. Key insights include: 1) The SoTA model, utilizing dataset-specific CoT fine-tuning on GSM and MATH, displays strong performance within its domains but struggles in OOD scenarios; 2) Diverse data sources in MathInstruct enable better math generalist model; 3) Fine-tuning on the PoT subsets generally outperforms finetuning on the CoT subsets; 4) Hybrid training yields the best-performing model. The breakdown results on each dataset can be found in Appendix Table 6. 

From Table 4, we can observe that our main competitor for OOD datasets is Platypus (Lee et al., 2023). Similar to in-domain results, Platypus is able to yield gains over the baseline models universally across the board, especially on the MMLU-Math dataset, which is tied with MAmmoTH-70B. It is worth noting that the performance gains of our model on OOD datasets are even more significant than on in-domain datasets. This demonstrates our models’ remarkable generalizability to unseen math problems. Notably, MAmmoTH-7B also boosts the CoT performance of WizardMath-7B greatly on MMLU-Math by 9%, which contains a substantial number of questions beyond the subjects we covered in our training dataset. 

**Comparison between Different Base Models.** In our experiments, we experimented with both Llama-2 and Code-Llama as the base models. From the two tables, we can observe that CodeLlama is consistently better than Llama-2, especially on OOD datasets. The gap between MAmmoTH and MAmmoTH-Coder can even reach up to 5%. Surprisingly, the average performance on OOD datasets of MAmmoTH-Coder (34B) is actually higher than MAmmoTH (70B). We believe MAmmoTH-Coder benefits greatly from the continuous code training of Code-Llama, which not only enhances the PoT capabilities but also improves Llama’s general reasoning skills. 

## 3.4 ABLATION STUDY ON DATA SOURCE 

**Ablation of the Data Source.** In order to better understand what factors contribute to the great gain of MAmmoTH over existing baselines, we set up a group of control experiments in Figure 2. We study the following setups: 

(1) MAmmoTH (MathInstruct- CoT): This experiment aims to understand how much our curated CoT data could improve the generalization over the SoTA model WizardMath (Luo et al., 2023) trained specifically on GSM + MATH. As can be seen, while sacrificing accuracy on GSM + MATH by 3%, our CoT subset fine-tuning improves the overall nine-dataset accuracy from 27% to 32%. 

(2) MAmmoTH (MathInstruct- PoT): This experiment aims to understand the advantage of our PoT subset. As can be observed, our PoT subset fine-tuning can significantly improve the overall accuracy from 27% to 41%. This ablation reflects the importance of unlocking the program generation capabilities of our model. 

(3) MAmmoTH (MathInstruct- Hybrid): We further combine CoT and PoT as the hybrid training data to achieve the best overall performance of 47.9%. This combined gain comes from two aspects: 

- The CoT subset helps maintain generic language-based reasoning skills to handle scenarios where PoT cannot handle well, e.g., abstract reasoning multi-choice questions in AQuA and MMLU. 

- The PoT subset can teach the model how to utilize Python APIs to solve complex math problems with high precision, e.g., the MATH problems requiring complex computation. 

8 

Preprint. Work in Progress 

|Training Data<br>GSM<br>MATH<br>AQuA<br>NumG<br>SVA<br>Mat<br>Sim<br>SAT<br>MMLU|**AVG**|
|---|---|
|-<br>14.6<br>2.5<br>30.3<br>29.9<br>34.5<br>6.0<br>5.0<br>26.8<br>29.8|**-25.3**|
|G<br>56.6<br>9.2<br>24.4<br>32.1<br>65.4<br>20.5<br>12.3<br>27.2<br>25.2<br>G + M<br>58.1<br>28.2<br>26.0<br>34.7<br>64.8<br>50.1<br>17.1<br>28.6<br>28.4<br>G + M + C<br>57.4<br>28.5<br>26.2<br>37.5<br>65.3<br>50.4<br>17.7<br>29.3<br>28.7<br>G + M + C + A<br>57.5<br>29.1<br>46.9<br>42.2<br>65.8<br>49.6<br>32.7<br>42.3<br>43.1<br>G + M + C + A + N<br>56.5<br>28.9<br>38.2<br>63.7<br>64.1<br>47.9<br>40.8<br>38.6<br>44.5|**-22.7**<br>**-19.5**<br>**-19.2**<br>**-4.8**<br>**-3.4**|
|Existing Data<br>31.4<br>18.4<br>40.3<br>53.3<br>61.8<br>27.9<br>45.6<br>32.7<br>38.4|**-9.0**|
|MathInstruct<br>53.6<br>31.5<br>44.5<br>61.2<br>67.7<br>46.3<br>41.2<br>42.7<br>42.6|**47.9**|



Table 5: Influence of different major subsets in MathInstruct based on Llama-2 7B. G: GSM8K, M: MATH, C: Camel, A: AQuA, N: NumGLUE. “Existing data”: the subset of MathInstruct in Table 1 by excluding all the NEW rationales curated by us. We shorten Mathematics as Mat, SimulEq as Sim, NumGLUE as NumG, and SVAMP as SVA to save space. 

We put some case studies in Appendix B to demonstrate the respective advantages of PoT and CoT in solving different types of math problems. To summarize, we attribute our substantial gain to: 1) diverse data sources covering different math fields and complexity levels and 2) a hybrid of CoT & PoT instruction tuning and decoding strategy. 

**Influence of Major Subsets.** Given the diverse sources of MathInstruct used in training MAmmoTH, it is important to understand how each dataset contributes to the overall performance of the model. We focus on five significant subsets: GSM8K, MATH, Camel, AQuA and NumGLUE. We conduct an experiment gradually adding each dataset into training and compare the performance with the one fine-tuned on the whole MathInstruct. As we can see from Table 5, when the data is not very diverse in training at the beginning (e.g., GSM8K only), the overall generalization performance is very bad: the model only fits in-distribution data and struggles to answer questions beyond GSM questions. And when gradually adding other major subsets, besides seeing the improvements on its own test sets overall, we could observe MAmmoTH becomes a better math generalist. 

These results underscore the significant impact of diverse data sources on MAmmoTH performance, a core aspect of making MAmmoTH a math generalist. The results also provide valuable insights for future data curation and collection efforts (e.g., we should always collect diverse data and avoid collecting only specific types of data). 

To help understand the contribution of the 6 newly curated datasets as shown in Table 1, we remove them from MathInstruct, and train a model on the existing data. As shown in the last two rows of Table 5, our new curated data substantially improves the performance on many datasets and leads to a 9% overall increase, which reflects the importance of the NEWLY curated dataset. 

**Influence of Hybrid Decoding.** To demonstrate the effectiveness of the hybrid decoding method, we conduct an experiment as outlined in subsection 2.4. By default, we initially attempt the PoT decoding method for a given question. If it fails to generate an executable query, we then transition to the CoT decoding method. The performance of different decoding methods (CoT, PoT, and Hybrid) is shown in Table 7. This hybrid decoding improves performance on every test set, showcasing that our model can effectively leverage the strengths of both CoT and PoT decoding strategies. 

## 4 CONCLUSION 

In this paper, we propose a novel math instruction tuning approach to activate open-source LLMs’ mathematical reasoning capabilities. Through a comprehensive study, we show that our models can outperform the SoTA performance at different scales by a huge margin. Our models benefit massively from: 1) the broad coverage of different math fields and complexity levels, and 2) a hybrid of CoT and PoT training. Our instruction tuning dataset contains 260K samples, which makes fine-tuning highly affordable even for academic labs. Our work paves the road for future studies to activate LLMs’ core capabilities in specialized domains. 

9 

Preprint. Work in Progress 

## REFERENCES 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 2357–2367, 2019. doi: 10.18653/v1/N19-1245. URL https:// aclanthology.org/N19-1245. 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. _ArXiv preprint_ , abs/2305.10403, 2023. URL https://arxiv.org/abs/2305.10403. 

- Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. _ArXiv preprint_ , abs/2212.08073, 2022. URL https://arxiv. org/abs/2212.08073. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _ArXiv preprint_ , abs/2107.03374, 2021. URL https:// arxiv.org/abs/2107.03374. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _ArXiv preprint_ , abs/2211.12588, 2022. URL https://arxiv.org/abs/2211.12588. 

- Wenhu Chen, Ming Yin, Max Ku, Elaine Wan, Xueguang Ma, Jianyu Xu, Tony Xia, Xinyi Wang, and Pan Lu. Theoremqa: A theorem-driven question answering dataset. _ArXiv preprint_ , abs/2305.12524, 2023. URL https://arxiv.org/abs/2305.12524. 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. _ArXiv preprint_ , abs/2210.11416, 2022. URL https://arxiv.org/abs/2210.11416. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _ArXiv preprint_ , abs/2110.14168, 2021. URL https://arxiv. org/abs/2110.14168. 

- Alex Davies, Petar Veliˇckovi´c, Lars Buesing, Sam Blackwell, Daniel Zheng, Nenad Tomaˇsev, Richard Tanburn, Peter Battaglia, Charles Blundell, Andr´as Juh´asz, et al. Advancing mathematics by guiding human intuition with ai. _Nature_ , 600(7887):70–74, 2021. URL https: //www.nature.com/articles/s41586-021-04086-x. 

- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. _ArXiv preprint_ , abs/2305.14314, 2023. URL https://arxiv.org/abs/ 2305.14314. 

- Andrew Drozdov, Nathanael Sch¨arli, Ekin Aky¨urek, Nathan Scales, Xinying Song, Xinyun Chen, Olivier Bousquet, and Denny Zhou. Compositional semantic parsing with large language models. _International Conference on Learning Representations (ICLR)_ , 2023. URL https: //openreview.net/forum?id=gJW8hSGBys8. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. URL https://proceedings.mlr.press/ v202/gao23f/gao23f.pdf. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. _ArXiv preprint_ , abs/2305.11738, 2023. URL https://arxiv.org/abs/2305.11738. 

10 

Preprint. Work in Progress 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ , 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)_ , 2021b. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/ be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf. 

- Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 523–533, 2014. doi: 10. 3115/v1/D14-1058. URL https://aclanthology.org/D14-1058. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _NeurIPS_ , 2022. 

- Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena Dumas Ang. Parsing algebraic word problems into equations. _Transactions of the Association for Computational Linguistics_ , 3:585–597, 2015. doi: 10.1162/tacl ~~a 0~~ 0160. URL https: //aclanthology.org/Q15-1042. 

- Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 1152–1157, 2016. doi: 10.18653/v1/N16-1136. URL https://aclanthology. org/N16-1136. 

- Ariel N Lee, Cole J Hunter, and Nataniel Ruiz. Platypus: Quick, cheap, and powerful refinement of llms. _ArXiv preprint_ , abs/2308.07317, 2023. URL https://arxiv.org/abs/2308. 07317. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _Advances in Neural Information Processing Systems_ , 35:3843–3857, 2022. URL https://openreview.net/pdf?id=IFXTZERXdM7. 

- Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for” mind” exploration of large scale language model society. _ArXiv preprint_ , abs/2303.17760, 2023a. URL https://arxiv.org/abs/2303.17760. 

- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5315– 5333, 2023b. URL https://aclanthology.org/2023.acl-long.291.pdf. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, 2017. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015. 

- Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. _ICML_ , 2023. URL https://openreview.net/pdf?id=ZX4uS605XV. 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _ArXiv preprint_ , abs/2308.09583, 2023. URL https://arxiv.org/abs/2308.09583. 

11 

Preprint. Work in Progress 

- Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. Language models of code are few-shot commonsense learners. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 1384–1403, 2022. URL https: //aclanthology.org/2022.emnlp-main.90.pdf. 

- Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. LILA: A unified benchmark for mathematical reasoning. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 5807–5832, 2022a. URL https: //aclanthology.org/2022.emnlp-main.392. 

- Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. NumGLUE: A suite of fundamental yet challenging mathematical reasoning tasks. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 3505–3523, 2022b. doi: 10.18653/v1/2022.acl-long.246. URL https://aclanthology.org/2022.acl-long.246. 

- Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. _ArXiv preprint_ , abs/2306.02707, 2023. URL https://arxiv.org/abs/2306.02707. 

- Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In _International Conference on Learning Representations (ICLR)_ , 2023. URL https: //openreview.net/pdf?id=iaYcJKpY2B_. 

- Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. In _Deep Learning for Code Workshop_ , 2022. URL https://arxiv.org/abs/2112.00114. 

- OpenAI. Gpt-4 technical report. _ArXiv preprint_ , abs/2303.08774, 2023. URL https://arxiv. org/abs/2303.08774. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2080– 2094, 2021. doi: 10.18653/v1/2021.naacl-main.168. URL https://aclanthology.org/ 2021.naacl-main.168. 

- Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only. _ArXiv preprint_ , abs/2306.01116, 2023. URL https://arxiv.org/abs/2306.01116. 

- Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. _ArXiv preprint_ , abs/2304.03277, 2023. URL https://arxiv.org/abs/2304. 03277. 

- Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In _SC20: International Conference for High Performance Computing, Networking, Storage and Analysis_ , pp. 1–16. IEEE, 2020. URL https://dl.acm.org/doi/10.5555/3433701.3433727. 

- Subhro Roy and Dan Roth. Solving general arithmetic word problems. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pp. 1743–1752, 2015. doi: 10.18653/v1/D15-1202. URL https://aclanthology.org/D15-1202. 

- Baptiste Rozi`ere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, J´er´emy Rapin, et al. Code llama: Open foundation models for code. _ArXiv preprint_ , abs/2308.12950, 2023. URL https://arxiv.org/abs/2308.12950. 

12 

Preprint. Work in Progress 

- Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault F´evry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ , 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4. 

- Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. _ArXiv preprint_ , abs/2210.09261, 2022. URL https://arxiv.org/abs/2210.09261. 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023. 

- Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. _ArXiv preprint_ , abs/2211.09085, 2022. URL https://arxiv.org/abs/2211. 09085. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _ArXiv preprint_ , abs/2302.13971, 2023a. URL https: //arxiv.org/abs/2302.13971. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _ArXiv preprint_ , abs/2307.09288, 2023b. URL https: //arxiv.org/abs/2307.09288. 

- Boshi Wang, Xiang Deng, and Huan Sun. Iteratively prompt pre-trained language models for chain of thought. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 2714–2730. Association for Computational Linguistics, 2022a. URL https: //aclanthology.org/2022.emnlp-main.174. 

- Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2717–2739. Association for Computational Linguistics, 2023a. doi: 10.18653/ v1/2023.acl-long.153. URL https://aclanthology.org/2023.acl-long.153. 

- Boshi Wang, Xiang Yue, and Huan Sun. Can chatgpt defend the truth? automatic dialectical evaluation elicits llms’ deficiencies in reasoning. _ArXiv preprint_ , abs/2305.13160, 2023b. URL https://arxiv.org/abs/2305.13160. 

- Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. _ArXiv preprint_ , abs/2305.04091, 2023c. URL https://arxiv.org/abs/2305. 04091. 

- Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, and Zhifang Sui. Making large language models better reasoners with alignment. _ArXiv preprint_ , abs/2309.02144, 2023d. URL https://arxiv.org/abs/2309.02144. 

- Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. _ArXiv preprint_ , abs/2307.10635, 2023e. URL https://arxiv.org/abs/2307.10635. 

13 

Preprint. Work in Progress 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. Selfconsistency improves chain of thought reasoning in language models. _International Conference on Learning Representations (ICLR)_ , 2023f. URL https://openreview.net/pdf?id= 1PL1NIMMrw. 

- Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 5085–5109, 2022b. URL https://aclanthology.org/2022.emnlp-main.340. 

- Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. _ArXiv preprint_ , abs/2306.04751, 2023g. URL https://arxiv.org/abs/2306.04751. 

- Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. _The 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)_ , 2023h. URL https://aclanthology.org/2023.acl-long.754.pdf. 

- Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. Codet5+: Open code large language models for code understanding and generation. _ArXiv preprint_ , abs/2305.07922, 2023i. URL https://arxiv.org/abs/2305.07922. 

- Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ , 2022a. URL https://openreview.net/forum?id=gEZrGCozdqR. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022b. URL https: //openreview.net/pdf?id=_VjQlMeSB_J. 

- Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V Le. Simple synthetic data reduces sycophancy in large language models. _ArXiv preprint_ , abs/2308.03958, 2023. URL https: //arxiv.org/abs/2308.03958. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. _ArXiv preprint_ , abs/1910.03771, 2019. URL https://arxiv.org/abs/1910.03771. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of incontext learning as implicit bayesian inference. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ , 2022. URL https: //openreview.net/forum?id=RdJVFCHjUMI. 

- Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, Min-Yen Kan, Junxian He, and Qizhe Xie. Decomposition enhances reasoning via self-evaluation guided decoding. _ArXiv preprint_ , abs/2305.00633, 2023. URL https://arxiv.org/abs/2305.00633. 

- Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _ArXiv preprint_ , abs/2304.12244, 2023. URL https://arxiv.org/abs/2304.12244. 

14 

Preprint. Work in Progress 

Zhen Yang, Ming Ding, Qingsong Lv, Zhihuan Jiang, Zehai He, Yuyi Guo, Jinfeng Bai, and Jie Tang. Gpt can solve mathematical problems without a calculator. _ArXiv preprint_ , abs/2309.03241, 2023. URL https://arxiv.org/abs/2309.03241. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In _International Conference on Learning Representations (ICLR)_ , 2023. URL https://openreview.net/pdf?id=WE_ vluYUL-X. 

- Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren. CrossFit: A few-shot learning challenge for crosstask generalization in NLP. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 7163–7189, 2021. doi: 10.18653/v1/2021.emnlp-main.572. URL https://aclanthology.org/2021.emnlp-main.572. 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _ArXiv preprint_ , abs/2309.12284, 2023. URL https: //arxiv.org/abs/2309.12284. 

- Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. _ArXiv preprint_ , abs/2308.01825, 2023. URL https://arxiv.org/abs/2308.01825. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. _ArXiv preprint_ , abs/2205.01068, 2022. URL https://arxiv.org/abs/ 2205.01068. 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. _ArXiv preprint_ , abs/2304.09797, 2023a. URL https://arxiv.org/abs/2304.09797. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _ArXiv preprint_ , abs/2306.05685, 2023b. URL https://arxiv.org/abs/ 2306.05685. 

- Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _ArXiv preprint_ , abs/2304.06364, 2023. URL https://arxiv.org/abs/2304. 06364. 

- Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. _ArXiv preprint_ , abs/2308.07921, 2023a. URL https://arxiv.org/abs/2308.07921. 

- Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. _ArXiv preprint_ , abs/2305.11206, 2023b. URL https://arxiv.org/abs/2305.11206. 

- Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models. _International Conference on Learning Representations (ICLR)_ , 2023c. URL https://openreview.net/pdf?id=WZH7099tgfM. 

15 

Preprint. Work in Progress 

## A RELATED WORK 

## A.1 MATHEMATICAL REASONING DATASETS 

Our work builds upon the existing mathematical reasoning literature. Early on, mathematical reasoning is mostly focused on solving synthetic basic math problems like AddSub (Hosseini et al., 2014) and other arithmetic reasoning datasets (Koncel-Kedziorski et al., 2015; Roy & Roth, 2015; Patel et al., 2021). Later on, more difficult math word problem datasets (Cobbe et al., 2021; Amini et al., 2019; Ling et al., 2017; Hendrycks et al., 2021b) have been proposed to focus on addressing realistic math word problems. NumGLUE (Mishra et al., 2022b) and LiLA (Mishra et al., 2022a) compile the existing literature to build a more diversified dataset collection. However, these datasets are mostly focused on grade school math problems. To further test LLMs’ limits in addressing more complex math problems, MMLU (Hendrycks et al., 2021a) includes college math problems in its evaluation suite. More recently, (Chen et al., 2023; Wang et al., 2023e) have proposed to tackle more challenging college-level science and math problems. Our instruction tuning dataset is built upon existing work to include a diversified collection of math problems from different subfields. 

## A.2 REASONING WITH LARGE LANGUAGE MODELS 

LLMs have demonstrated great capabilities to reason with the help of Chain-of-Thought prompting (Wei et al., 2022b; Kojima et al., 2022; Wang et al., 2023f). Suzgun et al. (2022) have shown that CoT can already surpass human performance on challenging BIG-Bench tasks. Later on, several other works (Drozdov et al., 2023; Zhou et al., 2023c; Nye et al., 2022; Wang et al., 2022a; 2023a; Li et al., 2023b; Wang et al., 2023d; Yu et al., 2023) also propose different approaches to utilize LLMs to solve reasoning tasks by allowing intermediate steps. ReAct Yao et al. (2023) proposes to leverage external tools like search engines to enhance LLM reasoning skills. Another trend is to enable LLMs’ capabilities to use programs as thought processes like PoT (Chen et al., 2022). Some follow-up works include self-critic (Gou et al., 2023), self-eval (Xie et al., 2023), plan-andsolve (Wang et al., 2023c). These methods propose to enhance LLMs’ capabilities to solve math problems with PoT. Self-critic (Gou et al., 2023) and self-eval (Xie et al., 2022) both adopt selfevaluation to enhance the robustness of the generated program. Plan-and-solve (Wang et al., 2023c) instead adopts more detailed planning instructions to help LLMs create a high-level reasoning plan. These methods all prove to bring decent improvements over PoT. 

## A.3 INSTRUCTION TUNING IN LANGUAGE MODELS 

Instruction tuning is part of a line of work designed to “align” language models with more useful objectives and human preferences. The instruction tuning step is seen as a major step to activate LLMs’ certain capabilities to respond to human instructions. Previously, instruction tuning is mainly focused on enhancing LLMs’ general-purpose instruction following abilities. Since 2021, CrossFit (Ye et al., 2021) and NaturalInstruction (Wang et al., 2022b), FLAN (Wei et al., 2022a) and T0 (Sanh et al., 2022) are amongst the first wave of instruction tuning effort to understand LLMs’ generalization capabilities. Later on, FLAN-v2 (Chung et al., 2022; Longpre et al., 2023) have been proposed to understand the effect of scaling up the instruction datasets to understand its impact on model performance. These approaches mainly adopt human-annotated datasets to build the instruction following dataset. More recently, multiple works (Wang et al., 2023h; Xu et al., 2023; Peng et al., 2023; Zhou et al., 2023b; Wang et al., 2023g) propose to utilize synthetic instruction following data distilled from GPT-3/4 to align open-source LLMs. The most similar effort to ours is Platypus (Lee et al., 2023) which aims to utilize a domain-specialized dataset to construct a small-scale instruction following dataset to enhance LLMs’ reasoning capabilities. 

16 

Preprint. Work in Progress 

## B CASE STUDY 

We conduct a comparison between our PoT results vs. CoT results in Figure 3, Figure 4 and Figure 5. In the first example, even though PoT and CoT can both solve the problem, CoT gives a very tedious solution to derive the answer. Such solution is not only slow but also unstable. In the second and third case, we can further see the advantages of PoT over CoT by utilizing external tools and Python packages to greatly simplify the solution. Figure 6 shows some types of questions (especially the formal logic question) that are not easily handled by programs. In order to address these types of questions, CoT is a better choice. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning_images/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning.pdf-0017-03.png)


**----- Start of picture text -----**<br>
!! !<br>What is the value of ?<br>PoT !! CoT<br>from sympy import factorial First, we need to understand the given expression:  !!!! !<br>Step 1: Simplify the expression inside the parentheses.<br>value = factorial(factorial(3)) / factorial(3)print(value) 3! = 3 ×2 ×1 = 6So, the expression becomes:  ## !<br>Step 2: Recall the property of factorials that states  𝑛! [$] = 𝑛! for any<br>>>> 120 positive integer 𝑘 and any positive integer 𝑛 ≥𝑘. In our case, we have<br>𝑛 =  6 and 𝑘 =  6.<br>Using this property, we can rewrite the expression as: [#!] #<br>Step 3: Simplify the expression further by applying the property<br>mentioned in Step 2. 6! =  6 ×5 ×4 ×3 ×2 ×1 =  720<br>So, the expression becomes:  [%&'] #<br>Step 4: Divide 720 by 6. 720 ÷ 6 =  120<br>So, the final simplified expression is: 120<br>Therefore, the value of  !!!! ! is  120 .<br>The answer is:  𝟏𝟐𝟎<br>**----- End of picture text -----**<br>


Figure 3: Example 1: PoT and CoT can both solve the problem, however, CoT gives a very tedious solution to derive the answer. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning_images/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning.pdf-0017-05.png)


**----- Start of picture text -----**<br>
Determine the number of ways to arrange the letters of the word TEPEE.<br>PoT CoT<br>from sympy import factorial<br>First we count the arrangements if all the letters are unique, which is 4!.<br>word = "TEEPE" Then since the E's are not unique, we divide by 2! for the arrangements<br>n = len(word) of the two E's, for an answer of  [4!] 2! = 60 .<br>d = word.count('E')<br>ways = factorial(n)/factorial(d) The answer is:  𝟔𝟎<br>print(ways)<br>>>> 20<br>**----- End of picture text -----**<br>


Figure 4: Example 2: PoT generates the correct solution while CoT fails. 

17 

Preprint. Work in Progress 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning_images/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning.pdf-0018-01.png)


**----- Start of picture text -----**<br>
Question: 𝑎 is chosen from the set {1,2,3,4}, and 𝑏 is chosen from the set<br>{5,6,7,8,9}. What is the probability that the product 𝑎𝑏 is a prime number?<br>PoT CoT<br>from sympy import isprime First, we need to find the total number of possible combinations of<br># define the sets choosing a number from each set. There are 4 numbers in the first set<br>a_set = {1,2,3,4} and 5 numbers in the second set. So, there are combinations. 4 ⋅5 =  20 possible<br>b_set = {5,6,7,8,9}<br>n_a = len(a_set) Now, let's consider the prime numbers. A prime number is a number<br>n_b = len(b_set) greater than 1 that has no divisors other than 1 and itself. We need to<br>prime_count = 0 find the pairs of (a, b) where either a or b is a prime number, or both.<br>prime_ab_count = 0<br>prime_a_count = prime_ab_count 1. If a is a prime number, there are 4 prime numbers in the first set: 2,<br>3, 5, and 7. So, there are 4 possible combinations where a is a prime<br>prime_b_count = prime_ab_count number.<br># iterate over all possible pairs<br>2. If b is a prime number, there are 5 prime numbers in the second set:<br>for a in a_set: 3, 5, 7, 11, and 13. So, there are 5 possible combinations where b is a<br>for b in b_set: prime number.<br>if isprime(a) or isprime(b): 3. If both a and b are prime numbers, we need to find the pairs where<br>if isprime(a*b): the prime numbers are 2, 3, 5, or 7. There are 4 pairs in this case: (2, 3),<br># increment the counters (2, 5), (2, 7), and (3, 5).<br>prime_count += 1<br>Now, let's add up the number of combinations where either a or b is a<br>prime_ab_count += 1<br>prime number: 4 (a is prime) + 5 (b is prime) + 4 (both are prime) = 13.<br># check if a is prime<br>if isprime(a): So, there are 13 combinations where either a or b is a prime number.<br>prime_a_count += 1 Since there are 20 possible combinations in total, the probability that<br>if isprime(b): the product ab is a prime number is  [!"] #$<br>prime_b_count += 1<br>probability = prime_ab_count / (n_a * n_b) Therefore, the probability is  [!"] #$<br>print(probability)  The answer is:  [𝟏𝟑]<br>𝟐𝟎<br>>>> 0.1<br>**----- End of picture text -----**<br>


Figure 5: Example 3: PoT generates the correct solution while CoT fails. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning_images/iclr_2024_MAmmoTH_Building_Math_Generalist_Models_through_Hybrid_Instruction_Tuning.pdf-0018-03.png)


**----- Start of picture text -----**<br>
If 𝑛 is such that 36 ≤ 𝑛 ≤ 72, then 𝑥= 𝑛2 + 2 𝑛 𝑛+ 4 + 16 / 𝑛+ 4 𝑛+ 4 satisfies.<br>Answer Choices: (A) 20 <  𝑥 <  54 (B) 23 <  𝑥 <  58 (C) 25 <  𝑥 <  64 (D) 28 <  𝑥 <  60 (E) None of these<br>The expression in the numerator can be rewritten as  𝑛+ 2 𝑛+ 4 [!] , and so the expression is equal to<br>𝑛+ 2 𝑛+ 4. We substitute the smallest value of 𝑛, which is 36, and get 42. We substitute the largest<br>value of 𝑛, which is 72, and get 88.  So the answer is (E) None of these CoT<br>Which of the following could NOT be the sides of an isosceles triangle?<br>Answer Choices: (A) 2,2,3 (B) 2,2,4 (C) 4,12,12 (D) 5,5,5<br>This is because in an isosceles triangle, the lengths of two sides are equal, and the length of the third side must be<br>less than the sum of the lengths of the other two sides. In this case, 4 is equal to the sum of the other two<br>sides (2+2), not less.  Therefore, the answer is (B) 2,2,4. CoT<br>Select the best translation into predicate logic. Some CEO is wealthier than everyone. (𝐶𝑥: 𝑥 is a CEO; 𝑃𝑥: 𝑥 is a<br>person; 𝑊𝑥𝑦: 𝑥 is wealthier than 𝑦)<br>Answer Choices: (A)  ∃𝑥 𝐶𝑥• ∀𝑦 𝑃𝑦• 𝑊𝑥𝑦 (B)  ∃𝑥 𝐶𝑥⊃ ∀𝑦 𝑃𝑦⊃𝑊𝑥𝑦<br>(C)  ∃𝑥 𝐶𝑥• ∀𝑦 𝑃𝑦⊃𝑊𝑥𝑦 (D)  ∀𝑥 𝑃𝑥⊃ ∃𝑦 𝐶𝑦⊃𝑊𝑥𝑦<br>∃𝑥: There exists some 𝑥 𝐶𝑥: 𝑥 is a CEO, and<br>∀𝑦: For all 𝑦 𝑃𝑦: 𝑦 is a person, and 𝑊𝑥𝑦: 𝑥 is wealthier than 𝑦<br>So, in plain English, this expression says: "There exists some 𝑥 such that 𝑥 is a CEO and for all 𝑦, if 𝑦 is a person<br>then 𝑥 is wealthier than 𝑦." Which matches the original statement.<br>Therefore, the answer is (A)  ∃𝒙 𝑪𝒙• ∀𝒚 𝑷𝒚• 𝑾𝒙𝒚 CoT<br>**----- End of picture text -----**<br>


Figure 6: Example 4: Some types of questions (e.g., formal logic) are hard to be solved by PoT but could be handled by CoT. 

18 

Preprint. Work in Progress 

|Model|GSM|MATH|AQuA|NumG|SVA|Mat|Sim|SAT|MMLU|Overall|
|---|---|---|---|---|---|---|---|---|---|---|
|Base|14.6|2.5|30.3|29.9|34.5|6.0|5.0|26.8|29.8|19.9|
|WizzardMath|54.9|10.7|26.3|36.1|36.1|9.3|12.8|25.4|31.1|27.0|
|MAmmoTH(MathInstruct- CoT)|49.2|9.9|42.2|37.1|48.5|9.5|17.3|34.1|39.8|32.0|
|MAmmoTH(MathInstruct- PoT)|50.8|28.9|28.6|52.7|65.0|46.7|42.0|25.9|28.3|41.0|
|MAmmoTH(MathInstruct)|53.6|31.5|44.5|61.2|67.7|46.3|41.2|42.7|42.6|47.9|



Table 6: Breakdown results of Figure 2. Investigation of the influence of CoT & PoT hybrid training on the 7B Llama-2 model. 

|Model|Decoding|GSM|MATH|AQuA|NumG|SVA|Mat|Sim|SAT|MMLU|AVG|
|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT|50.5|10.4|43.7|44.0|47.3|9.2|18.9|32.7|39.9|33.0|
|MAmmoTH-7B|PoT|51.6|28.7|43.3|52.3|65.1|41.9|48.2|39.1|44.6|46.1|
||**Hybrid**|**53.6**|**31.5**|**44.5**|**61.2**|**67.7**|**46.3**|**41.2**|**42.7**|**42.6**|**47.9**|
||CoT|22.4|7.9|36.2|36.0|37.0|8.2|7.2|32.7|34.6|24.7|
|MAmmoTH-Coder-7B|PoT|58.8|32.1|47.2|57.1|71.1|53.9|44.6|40.0|47.8|50.3|
||**Hybrid**|**59.4**|**33.4**|**47.2**|**66.4**|**71.4**|**55.4**|**45.9**|**40.5**|**48.3**|**52.0**|
||CoT|56.3|12.9|45.3|45.6|53.8|11.7|22.4|43.6|42.3|37.1|
|MAmmoTH-13B|PoT|61.3|32.6|48.8|59.6|72.2|48.5|40.3|46.8|45.4|50.6|
||**Hybrid**|**62.0**|**34.2**|**51.6**|**68.7**|**72.4**|**49.2**|**43.2**|**46.8**|**47.6**|**52.9**|
||CoT|32.1|10.2|40.6|36.2|43.0|9.6|10.1|40.9|36.6|28.8|
|MAmmoTH-Coder-13B|PoT|64.3|35.2|46.8|54.2|73.2|60.0|44.2|48.2|48.2|52.7|
||**Hybrid**|**64.7**|**36.3**|**46.9**|**66.8**|**73.7**|**61.5**|**47.1**|**48.6**|**48.3**|**54.9**|
||CoT|34.3|11.6|39.0|36.2|44.6|10.8|10.9|46.4|42.9|30.7|
|MAmmoTH-Coder-33B|PoT|72.3|42.8|53.8|59.6|84.0|64.7|50.6|58.6|52.7|59.9|
||**Hybrid**|**72.7**|**43.6**|**54.7**|**71.6**|**84.3**|**65.4**|**51.8**|**60.9**|**53.8**|**62.1**|
||CoT|72.4|21.1|57.9|58.9|71.6|20.0|31.9|57.3|52.1|49.2|
|MAmmoTH-70B|PoT|76.7|40.1|60.2|64.3|81.7|55.3|45.3|64.1|53.5|60.1|
||**Hybrid**|**76.9**|**41.8**|**65.0**|**74.4**|**82.4**|**55.6**|**51.4**|**66.4**|**56.7**|**63.4**|



Table 7: Influence of different decoding methods on each dataset. 

## C LIMITATIONS 

Despite their training on a diverse set of mathematical rationale datasets, the MAmmoTH models might exhibit limitations when faced with problems outside their primary domain of expertise like mathematical analysis, complex analysis, graph theory, numerical analysis, etc. Thus, our models are not suitable for solving more complex problems in these fields. Also, they have not been trained with proof-type problems, thus their theorem-proving capability is also limited. In the future, we would like to expand the models’ skill set to cover more fields and theorem-proving problems. 

There is also a risk of the MAmmoTH models generating potentially harmful, offensive, or biased content, especially if they are asked to answer questions beyond math. The MAmmoTH series could be misused for malicious purposes, such as spreading misinformation or probing sensitive topics. Developers should conduct safety testing and tuning tailored to their specific applications before deploying any MAmmoTH model. While we have made every effort to ensure the cleanliness and purity of our training data, we cannot guarantee absolute perfection. It is unlikely but not impossible that some inappropriate questions slipped through the curation process. 

Future work may continue to explore how to further improve the robustness and generalizability of MAmmoTH in mathematical reasoning. For example, recent work identifies “sycophancy” and “Clever Hans effect” in reasoning: LLMs cannot maintain truthful solutions to reasoning tasks when challenged by the user’s absurdly invalid arguments and critiques (Wang et al., 2023b). Potential methods to improve the models’ reasoning robustness could involve the exploration of synthetic data intervention methods as explored in (Wei et al., 2023). 

19 

