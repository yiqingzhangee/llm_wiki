Published as a conference paper at ICLR 2024 

# MUSTARD: MASTERING UNIFORM SYNTHESIS OF THEOREM AND PROOF DATA 

**Yinya Huang**[1,6] **Xiaohan Lin**[2] **Zhengying Liu**[3] _[†]_ **Qingxing Cao**[2] _[†]_ **Huajian Xin**[2] **Haiming Wang**[2] **Zhenguo Li**[3] **Linqi Song**[1,6] _[†]_ **Xiaodan Liang**[2,4,5] _[†]_ 1City University of Hong Kong 2Shenzhen Campus of Sun Yat-sen University 3Huawei Noah’s Ark Lab 4DarkMatter AI Research 5MBZUAI 6CityUSRI yinya.huang@hotmail.com, linxh55@mail2.sysu.edu.cn, _{_ liuzhengying2, Li.Zhenguo _}_ @huawei.com, caoqx8@sysu.edu.cn, linqi.song@cityu.edu.hk, xdliang328@gmail.com 

## ABSTRACT 

Recent large language models (LLMs) have witnessed significant advancement in various tasks, including mathematical reasoning and theorem proving. As these two tasks require strict and formal multi-step inference, they are appealing domains for exploring the reasoning ability of LLMs but still face important challenges. Previous studies such as Chain-of-Thought (CoT) have revealed the effectiveness of intermediate steps guidance. However, such step-wise annotation requires heavy labor, leading to insufficient training steps for current benchmarks. To fill this gap, this work introduces MUSTARD, a data generation framework that **m** asters **u** niform **s** ynthesis of **t** heorem **a** nd p **r** oof **d** ata of high quality and diversity. MUSTARD synthesizes data in three stages: (1) It samples a few mathematical concept seeds as the problem category. (2) Then, it prompts a generative language model with the sampled concepts to obtain both the problems and their step-wise formal solutions. (3) Lastly, the framework utilizes a proof assistant (e.g., Lean Prover) to filter the valid proofs. With the proposed MUSTARD, we present a theorem-and-proof benchmark MUSTARDSAUCE with 5,866 valid data points. Each data point contains an informal statement, an informal proof, and a translated formal proof that passes the prover validation. We perform extensive analysis and demonstrate that MUSTARD generates validated high-quality step-by-step data. We further apply the MUSTARDSAUCE for fine-tuning smaller language models. The fine-tuned Llama 2-7B achieves a 15.41% average relative performance gain in automated theorem proving, and 8.18% in math word problems.[1] 

## 1 INTRODUCTION 

Large language models (LLMs) (OpenAI, 2023; 2022) have shown promising reasoning capabilities in various domains, including math word problem and theorem proving (Cobbe et al., 2021; Hendrycks et al., 2021; Zheng et al., 2022; Wu et al., 2021). These two tasks, which require strictly and successively multi-step inference, have become appeal domains to evaluate and develop LLMs’ ability in complex reasoning. Recent works progress LLMs in solving math problems mainly through two techniques. The first is the chain-of-thoughts (CoT) prompting (Wei et al., 2022; Kojima et al., 2022; Wang et al., 2023b), which provides step-by-step solutions to the LLMs. The second is to leverage the LLMs’ ability in code generation to generate formalized languages and utilize external solvers to obtain strict inference results (Wu et al., 2022; Jiang et al., 2023; Polu & Sutskever, 2020; Han et al., 2022; Polu et al., 2023). Both techniques rely on step-wise annotation to improve LLMs’ performance and interpretability on the math problem. 

Correct intermediate steps are crucial for LLMs to perform complex reasoning. However, highquality step-wise annotations are hard to obtain, and Figure 1 demonstrates a few representative 

> _†_ Corresponding author. 

> 1Codes and data are available at: https://github.com/Eleanor-H/MUSTARD. 

1 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0002-01.png)


Figure 1: A comparison of methods of synthesizing and validating intermediate reasoning steps. 

works. Previous works such as miniF2F (Zheng et al., 2022) resorts to manual annotation and validation to obtain high-quality step-wise labels. However, manual annotation requires heavy labor of knowledgeable experts, resulting in an extremely small-scale dataset. Manual checking also does not guarantee the correctness of data as the labelers would make mistakes in labeling. On the other hand, generating data with rule-based checking such as ROSCOE (Golovneva et al., 2023) can produce large-scale reasoning data. Given that the generated data are more friendly and readable for human beings, the correctness of the reasoning is not guaranteed by those rules. Moreover, another line of work such as INT (Wu et al., 2021) performs rule-based synthesis to generate validated proofs, which are both correct and large-scale. However, the data are brutally synthesized so that many generated proofs lack actual meaning. Therefore, we need a more efficient way to generate mathematical data that are large-scale, with accurate intermediate steps, and also meaningful mathematical knowledge to human beings. 

To fill this gap, we propose MUSTARD, a data generation framework that uniformly synthesizes large-scale and high-quality mathematical data by combining the advantages of LLMs in verbalization and formal theorem provers in rigorous data validation. Specifically, MUSTARD first samples a few mathematical concepts from a predefined list and prompts an LLM to generate a related question described in natural language. Then, it applies the LLM to generate the corresponding solution in both natural and formal language. Given the generated solution, MUSTARD further validates them using a theorem prover. The passed one is considered to be correct and is a high-quality data point. The invalid one on the other hand is considered to be a challenging sample, which will be further combined with the error messages to prompt the LLM for a solution revision, and added as a challenging data point. 

By applying the proposed MUSTARD one can obtain large amounts of problems and theorems with desired mathematical concepts and domains. Eventually, we build a mathematical dataset with validated informal and formal solutions, named MUSTARDSAUCE (MUSTARD resource). 

We conduct extensive data analysis and experiments on the generated MUSTARDSAUCE. Through deep inspection of the data, we find that MUSTARD generates interesting and reasonable math problems by creatively combining two mathematical concepts, and MUSTARDSAUCE is diverse and has a high proportion of difficult data. We also observe that the prover is consistent with human evaluation, where humans usually consider a validated solution to have a higher quality than those without a formal validation process. Lastly, we fine-tune smaller-scale language models on MUSTARDSAUCE. The fine-tuned Llama 2-7B achieves improvements by 20.9% on zero-shot inference on GSM8K and achieves 8.7 of pass@1 on mathlib. These results demonstrate the effectiveness of MUSTARDSAUCE in improving the mathematical reasoning capabilities of language models. The contributions of this paper are summarized as follows: 

1. We propose a novel framework MUSTARD that can generate high-quality mathematical data (both informal and formal) with an interplay between generative language model and theorem prover assistants. 

2. We release the MUSTARDSAUCE, which contains both math word problems and theoremproving problems spanning over four educational levels. Each sample has corresponding informal and formal solutions. 

3. We conduct extensive analysis and experiments on the generated data, demonstrating their quality, diversity, and effectiveness in improving language models’ mathematical reasoning performance. 

2 

Published as a conference paper at ICLR 2024 

## 2 RELATED WORKS 

**Large Language Models for Mathematical Reasoning** The growing generative language models (Brown et al., 2020; OpenAI, 2022; 2023) show compelling potential for solving mathematical problems both in natural language proofs (OpenAI, 2023; Zheng et al., 2023a; Xiong et al., 2024) and in formal languages with theorem provers (Polu & Sutskever, 2020; Han et al., 2022; Zheng et al., 2023b; Polu et al., 2023). On the other hand, some works explore using language models to automatically translate natural language proofs into formal ones given few-shot demonstrations (Wu et al., 2022; Jiang et al., 2023; Liu et al., 2023; Wang et al., 2024). Chain-of-though reasoning (Wei et al., 2022; Kojima et al., 2022; Wang et al., 2023b) is demonstrated beneficial for the LLMs to derive correct answers. However, some recent works (Saparov & He, 2023; Golovneva et al., 2023) observe that the intermediate reasoning steps can be inconsistent. This paper proposes a data generation framework that taps the comprehensive mathematical reasoning capabilities of large language models. It generates mathematical reasoning problems with informal and formal solutions that are step-wise validated by a formal theorem prover. With the framework, we obtain high-quality mathematical data. 

**Synthesizing Mathematical Data** Obtaining large-scale high-quality mathematical data is a longstanding challenge. Previous data relies on well-trained annotators to hand-craft and review the formal proofs (Zheng et al., 2022), which is time and labour-consuming and results in a small data scale. Wang & Deng (2020) and Xiong et al. (2023) construct a neural generator for data synthesis, but it still requires the intervention of human-written data. Besides, Wu et al. (2021) explore using a theorem generator to automatically generate formal proofs with rules. However, the rule-based generation depends on given axioms in specified orders. As a result, the generated data is restricted to a few domains. On the other hand, recent works demonstrate the effectiveness of distilling knowledge from large language models (West et al., 2022; Yuan et al., 2023; Li et al., 2023), and some of them (Wang et al., 2023c; Xu et al., 2023) explore data evolution by properly prompting the language models. The proposed framework explores eliciting mathematical knowledge from large language models to achieve diverse and large-scale mathematical data. In this framework, an interplay between the language model and a formal proof assistant controls the quality and difficulties of data. Using the proposed framework, we collect a large-scale mathematical dataset that contains diverse and multiple-difficulty math questions with high-quality solutions. 

## 3 MUSTARD 

In this work, we aim to obtain large-scale mathematical data with multi-step annotations and propose MUSTARD to generate diverse and high-quality math and theorem-proving problems with multi-step informal and formal solutions. As shown in Figure 2, MUSTARD consists of three stages. In the first concept seeding stage, MUSTARD samples a set of math concepts as the problem domain. Then in the second solution generation stage, it generates the concept-related problem and solution by prompting an LLM. In the third stage, a theorem prover is used to validate the generated solution. If the solution can not pass the prover, the error message is returned to the second stage for another turn of solution generation. Through interaction between the LLM and a formal proof assistant, MUSTARD can generate diverse and high-quality data that contains both informal and formal solutions. We describe the details of each stage in this section. 

3.1 CONCEPT SEEDING 

We first define and build a mathematical concept pool that covers as complete sub-subjects in mathematics and educational levels as possible. Specifically, we collect all math courses on the Khan Academy website[2] , the large-scale online educational platform. The resulting pool includes concepts in four educational levels: elementary school, middle school, high school, and higher education. Each educational level has 5 to 9 math domains, covering different types of math problems such as algebra and geometry. Each domain contains subdivided mathematical concepts to inspect different mathematical abilities like polynomial arithmetic or factorization. Concept statistics and detailed concepts in each domain are demonstrated in Appendix B. 

Given the concept pool, for each educational level, MUSTARD uniformly samples 1 or 2 concepts from all domains as seeds, and then generates mathematical problems that cover the concepts. In particular, given an educational level, taking 2 concepts from different subjects challenges the model to generate problems that join diverse domains while keeping the problems reasonable. 

> 2https://www.khanacademy.org/math 

3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0004-01.png)


Figure 2: Overview of MUSTARD. The mathematical theorem (indicated by “# Problem:”) is created according to the randomly selected seed concepts. Following the theorem, a corresponding informal proof is generated, which is then translated into a formal proof in Lean 3. The formal proof is passed into Lean Prover for quality feedback, and according to the error messages the formal proof is revised and improved. The proofs that pass the Lean Prover are regarded as high-quality data points and collected. 

## 3.2 PROOF GENERATION 

Given the sampled mathematical concepts, MUSTARD generates math problems and their corresponding solutions. Specifically, MUSTARD leverages the capability of LLMs in generating natural language and code and prompts an LLM to generate the problem statement, its natural language solution, and a formal solution written in Lean. As a result, the LLM needs to complete the following three tasks: (T1) Generating a math problem that relates to the given concepts; (T2) Solving the math problem with a natural language proof; (T3) Performing auto-formalization to translate the written natural language proof into a formalized proof. In this work, we use GPT-4 OpenAI (2023) as the LLM for proof generation. 

We intend to generate a problem based on educational level, math domains, and concepts. Considering that mathematical problems include proof and calculation, we also introduce the question types into the prompt for generating theorem proving and word problems respectively. Moreover, we do not include any exemplars or other manual interventions except for the sampled concepts. We intend to avoid potential biases brought by the concepts inside the exemplars and achieve a more diverse generation. The prompt template is shown as the following: 

You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ‘‘Problem: ’’), a corresponding solution in natural language (indicated by ‘‘Informal proof:’’), and a translated formal solution in Lean 3 (indicated by ‘‘Formal proof in Lean 3:’’). Please note that the informal proof and the formal proof need to be identical. 

Please create a [QUESTION TYPE] in the level of [EDUCATIONAL LEVEL] based on the following knowledge point(s): [CONCEPT] in [DOMAIN]; [CONCEPT] in [DOMAIN]. You must respond in the following format: 

# Problem: ... 

# Informal proof: ... 

# Formal proof in Lean 3: ... 

The “[]” indicates the placeholders for the corresponding question type, educational level, concepts, and domains. Multiple concepts are separated by “;”. We retrieved the text after the “Problem:”, “Informal proof:” and “Formal proof in Lean 3:” as the generated sample. 

4 

Published as a conference paper at ICLR 2024 

## 3.3 PROOF FILTERING 

In the proof-filtering stage, MUSTARD interacts with the Lean Prover (de Moura et al., 2015) to obtain validation messages of the proof steps, which guides data revision and filtering. Specifically, after a formal solution is passed to the Lean Prover and if the prover returns no error message, the corresponding data point is collected into the valid dataset. Otherwise, MUSTARD collects the error messages from the prover and prompts the language model to revise the invalid solution. To help the language model locate the incorrect lines described in the error messages, we also add a line number at the beginning of each line in the formal solutions. The verification and self-refinement are performed in multiple rounds until LLM generates a valid solution. We use the number of rounds to measure the difficulty of the generated sample, assuming a difficult problem is hard to solve by an LLM and requires more rounds of correction. The prompt template of a single round of correction is demonstrated as follows, and the complete prompt template is shown in Table 13 in Appendix C.1: 

# Formal proof (c) in Lean 3: ‘‘‘lean line 1 <code> line 2 <code> line 3 <code> ... ‘‘‘ 

# Error messages for Formal proof (c) from Lean Prover: <error messages> 

## 4 EXPERIMENTS 

## 4.1 CASE STUDY 

We first inspect the data points generated by MUSTARD. Table 1 shows a generated math problem in which MUSTARD creatively combines two mathematical concepts and constructs a reasonable question. The generated question includes knowledge from both concepts. It is suggested that MUSTARD can join the concepts and construct a reasonable question. Furthermore, Table 2 demonstrates a case that MUSTARD provides solid and comprehensive solutions in both natural language and Lean. Although the formal proof is long, it is consistent and passes the prover’s validation. It is demonstrated that MUSTARD can generate long valid solutions. 

## 4.2 HUMAN EVALUATION 

To further explore the quality of the data generated by MUSTARD, we recruit professionals who have expertise in mathematics and the Lean language to perform a sanity check on the data points. We randomly select 200 data points from the generated data, 100 of which pass the Lean Prover (Group Valid) and 100 of which do not (Group Invalid). The sanity check covers the four sections in each data point (i.e., informal statement, informal proof, formal statement, and formal proof), and includes factuality check and consistency check. Specifically, a high-quality data point should have a factually correct informal statement (D1) and a correct solution (D4). The formal statement and proof should be aligned with the informal descriptions (D5, D6). Moreover, the desired data point should meet the specified seed concepts (D2) and question type (D3). The six inspection dimensions and their requirements are demonstrated in Table 3. A data point is scored 1 in a dimension if it meets the requirement, otherwise, it gets 0. 

The accuracies of Group Valid and Group Invalid in each dimension are demonstrated in Table 3. We also report the corresponding _p_ -value in each dimension. (D4) and (D6) show significant differences in accuracy between the two groups. The results indicate that high-quality data points have significantly better auto-formalization results. As a result, given the validation of formal proofs by the Lean Prover, the data points have guaranteed high-quality informal proofs. Moreover, (D1) also shows significance with the inspected data scaled up. The differences in statement alignment (D5) and informal statement relevance (D2) of the two groups are less significant. Furthermore, no significant differences are observed in question type classification (D3), which indicates that Lean Prover validation in MUSTARD does not significantly influence the classification. Overall, the human evaluation results suggest that formally validated data have significantly higher quality. 

## 4.3 DATA QUALITY BY DOWNSTREAM APPLICATION 

To evaluate the impact of MUSTARDSAUCE on enhancing mathematical reasoning abilities, we use the data to fine-tune smaller-scale language models and evaluate them on math word problems (MWP) and automated theorem proving (ATP). Specifically, given all the generated data, we first 

5 

Published as a conference paper at ICLR 2024 

Table 1: An informal statement generated by MUSTARD. The complete data point is demonstrated in Table 18 in Appendix E. 

**Question Type: Theorem Proving. Educational Level: Middle School. k=2.** 

**Concept(s):** _Geometry_ in 8th grade; _Algebraic expressions_ in Algebra basics. 

**Informal Statement** Given a rectangle ABCD where AB is x + 5 and AD is 2x - 3. Prove that the area of the rectangle ABCD is (2 _x_[2] + 7 _x −_ 15) square units. 

Table 2: A data point generated by MUSTARD. 

**Question Type: Word Problem. Educational Level: Higher Education. k=1. Concept(s):** _Series_ in Integral Calculus. **Informal Statement** Prove that the integral of the series[�] (from _n_ = 1 to infinity) of _[−]_[1][(] _n[n]_[+1)] equals ln(2). **Informal Proof** We start by recognizing the series[�] (from _n_ = 1 to infinity) of _[−]_[1][(] _n[n]_[+1)] as the alternating harmonic series, which is known to converge to ln(2). 

**Formal Statement and Proof import** data.real.basic **import** analysis.special ~~f~~ unctions.pow **open** real **lemma** integral ~~o~~ f ~~s~~ eries : � x **in** 0..1, (ln (1 + x)) =[�] _[′]_ (n : N), ((-1)ˆ(n+1) / (n + 1)) := **begin have** h ~~d~~ er : _∀_ x _∈_ set.Icc (0 : R) 1, deriv ( _λ_ x, (ln (1 + x))) x = 1 / (1 + x), _{_ intros x hx, **have** h ~~p~~ os : 0 < 1 + x := **by** nlinarith, exact deriv ~~l~~ n ~~o~~ ne ~~a~~ dd h ~~p~~ os _}_ , **have** h ~~i~~ nt : interval ~~i~~ ntegral ( _λ_ x, 1 / (1 + x)) 0 1 volume =[�] _[′]_ (n : N), ((-1)ˆ(n+1) / (n + 1)), _{_ **have** h ~~f~~ rac : _∀_ (n : N), � x **in** 0..1, xˆn = 1 / (n + 1), _{_ intro n, **calc** � x **in** 0..1, xˆn = [xˆ(n+1) / (n+1)] | 0..1 : integral ~~p~~ ow n ... = 1 / (n + 1) : **by** _{_ rw integral ~~i~~ nterval, simp _} }_ , rw [interval ~~i~~ ntegral.integral ~~o~~ f ~~l~~ e, h ~~f~~ rac], simp, linarith _}_ , **have** h ~~e~~ q : � x **in** 0..1, (ln (1 + x)) = � x **in** 0..1, 1 / (1 + x), _{_ congr, ext, exact h ~~d~~ er x (set.mem ~~I~~ cc.mpr `〈` **by** nlinarith, **by** nlinarith `〉` ) _}_ , rw [h ~~e~~ q, h ~~i~~ nt], **end** 

have 5,866 valid data points that pass the Lean Prover. We denote this subset MUSTARDSAUCEvalid. We then extract the same number of invalid data points as the subset of MUSTARDSAUCEinvalid, and extract an equal size of random subset MUSTARDSAUCE-random. Each MUSTARDSAUCE subset contains 5,866 data points. Moreover, we randomly split MUSTARDSAUCEvalid into 4,866 training data, 500 validation data, and 500 test data for benchmarking model performances on the dataset. We denote the test set as MUSTARDSAUCE-test. Furthermore, we also test on the entire generated data set with 28,316 data points, which we denote MUSTARDSAUCE-tt. 

We employ LoRA (Hu et al., 2021) for fine-tuning the open-source GPT2-large (Radford et al., 2019), Llama 2-7B and Llama 2-70B (Touvron et al., 2023) on each MUSTARDSAUCE subset. De- 

6 

Published as a conference paper at ICLR 2024 

Table 3: Inspection dimensions and requirements in human evaluation. IS: Informal Statement. IP: Informal Proof. FS: Formal Statement. FP: Formal Proof. RT: Reasoning Type. Significant _p <_ 0 _._ 005 are marked with **bold** . 

|0_._005are marked with**bold**.||
|---|---|
|Inspection Dimension<br>Requirement|Valid<br>Invalid<br>_p_-value|
|||
|(D1) IS Correctness<br>_Whether the informal statement is factually correct._<br>(D2) IS Relevance<br>_Whether the informal statement is relevant to each seed_<br>_concept._<br>(D3) RT Classifcation<br>_Whether the informal statement is of the required question type._<br>(D4) IP Correctness<br>_Whether the informal proof correctly solves the informal_<br>_statement._<br>(D5) IS-FS Alignment<br>_Whether the informal statement and the formal statement_<br>_describe the same problem and are aligned with each other._<br>(D6) IP-FP Alignment<br>_Whether the informal proof and the formal proof describe the_<br>_same solution and have aligned proof steps._|93.50<br>83.50<br>**0.00167**<br>87.50<br>92.50<br>0.09604<br>67.00<br>68.50<br>0.74903<br>88.50<br>73.50<br>**0.00012**<br>74.00<br>66.50<br>0.10138<br>72.00<br>54.00<br>**0.00018**|



Table 4: Maj1@1 results on GSM8K (G) and MATH (M). Zero: Zero-shot. Few: Few-shot. _>_ denotes a fine-tuning step. in: MUSTARDSAUCE-invalid. ra: MUSTARDSAUCE-random. va: MUSTARDSAUCE-valid. tt: MUSTARDSAUCE-tt. gt: GSM8K training split. 

|MODEL|Zero|Few|Zero|Few|MODEL|Zero|Few|Zero|Few|
|---|---|---|---|---|---|---|---|---|---|
||(G)|(G)|(M)|(M)||(G)|(G)|(M)|(M)|
|_Baselines_||||||||||
|GPT2-large|3.4|5.1|0.6|1.0|GPT2-large _>_|14.6|17.4|4.6|6.8|
||||||gt|||||
|Llama 2-7B|7.2|12.8|2.0|2.6|Llama 2-7B _>_|24.5|28.2|10.4|12.6|
||||||gt|||||
|_Fine-tuning_||||||||||
|GPT2-large _>_|4.2|6.8|1.4|2.4|GPT2-large _>_|16.1|18.9|4.8|8.0|
|tt|||||tt_>_gt|||||
|GPT2-large _>_|3.9|6.4|1.2|2.0|GPT2-large _>_|15.4|17.7|4.4|7.6|
|in|||||in_>_gt|||||
|GPT2-large _>_|4.1|6.7|1.4|2.2|GPT2-large _>_|15.7|18.5|4.8|7.8|
|ra|||||ra_>_gt|||||
|GPT2-large _>_|4.6|7.0|1.8|2.8|GPT2-large _>_|16.5|20.1|5.6|8.4|
|va|(+12.20%)|(+4.48%)|(+28.57%)|(+27.27%)|va_>_gt|(+5.10%)|(+8.65%)|(+16.67%)|(+7.69%)|
|Llama 2-7B _>_|9.6|16.0|3.2|3.8|Llama 2-7B _>_|27.4|31.5|13.0|14.4|
|tt|||||tt_>_gt|||||
|Llama 2-7B _>_|9.1|14.9|2.4|3.2|Llama 2-7B _>_|26.9|30.3|12.4|13.8|
|in|||||in_>_gt|||||
|Llama 2-7B _>_|9.5|15.4|3.0|3.6|Llama 2-7B _>_|27.1|30.7|12.6|14.2|
|ra|||||ra_>_gt|||||
|Llama 2-7B _>_|10.3|16.9|3.2|4.2|Llama 2-7B _>_|27.9|32.5|13.8|15.0|
|va|(+8.42%)|(+9.74%)|(+6.67%)|(+16.67%)|va_>_gt|(+2.95%)|(+5.86%)|(+9.52%)|(+5.63%)|



tailed model configuration and training procedure are described in Appendix F. For the task of math word problems, we use GSM8K (Cobbe et al., 2021) and MATH dataset Hendrycks et al. (2021)[3] for evaluation. For evaluating automated theorem proving, we use Mathlib[4] and the miniF2F (Zheng et al., 2022) benchmark. We also evaluate models on MUSTARDSAUCE-test after being finetuned on the MUSTARDSAUCE-valid training split. Tables 4 and 5 demonstrate the model performances. We also follow Han et al. (2022) to ablate the fine-tuning steps and demonstrate the results in Table 6. 

In general, fine-tuning the models on MUSTARDSAUCE improves the mathematical reasoning of the models. On average, we have an 18.15% relative performance gain after fine-tuning with MUSTARDSAUCE-valid compared with fine-tuning with MUSTARDSAUCE-random in ATP (Table 5) and 11.01% in MWP (Table 4). The fine-tuned Llama 2-7B achieves average gains of 15.41% and 8.18% on ATP and MWP, and the fine-tuned GPT 2-large 20.89% and 15.41%, respectively. Specifically, in ATP, the Llama 2-7B achieves significant performance gains of 16.00% on both mathlib and miniF2F, while increasing 17.31% on the MUSTARDSAUCE-test. In MWP, the performance improvements are also consistent in two datasets and both zero-shot and few-shot inference. We further compare the results fine-tuned with MUSTARDSAUCE-tt and MUSTARDSAUCE-valid. We find that models fine-tuned with the entire generated data are inferior to models fine-tuned with MUSTARDSAUCE-valid. Although the increase in the amount of fine-tuned data makes the model perform better compared to fine-tuning on MUSTARDSAUCE-invalid and MUSTARDSAUCE- 

> 3Specifically, we use the released test split with 500 problems in PRM800K (Lightman et al., 2023) which is demonstrated to be representative of the MATH test set as a whole. 

> 4https://github.com/leanprover-community/mathlib 

7 

Published as a conference paper at ICLR 2024 

Table 5: Pass@1 results on automated theorem proving tasks. _>_ denotes a fine-tuning step. test: MUSTARDSAUCE-test. in: MUSTARDSAUCE-invalid. ra: MUSTARDSAUCE-random. va: MUSTARDSAUCE-valid. tt: MUSTARDSAUCE-tt. mt: mathlib training split. Note that the reported results on MUSTARDSAUCE-test are obtained by only fine-tuning on the MUSTARDSAUCE-valid training split. 

|MODEL|mathlib|miniF2F|test|MODEL|mathlib|miniF2F|test|
|---|---|---|---|---|---|---|---|
|_Baselines_||||||||
|GPT2-large|0.0|0.0|0.0|GPT2-large_>_mt|5.6|2.9|8.6|
|Llama 2-7B|0.0|0.0|0.0|Llama 2-7B_>_mt|14.3|7.0|10.8|
|_Fine-tuning_||||||||
|GPT2-large_>_in|2.0|0.0|6.0|GPT2-large_>_in_>_mt|5.9|2.0|8.2|
|GPT2-large_>_ra|3.0|1.2|7.0|GPT2-large_>_ra_>_mt|6.6|2.9|9.6|
|GPT2-large_>_va|3.7|1.6|8.3|GPT2-large_>_va_>_mt|7.4|3.7|10.6|
||(+23.33%)|(+33.33%)|(+18.57%)||(+12.12%)|(+27.59%)|(+10.42%)|
|Llama 2-7B_>_tt|8.3|2.6|11.7|Llama 2-7B_>_tt_>_mt|15.1|7.0|13.6|
|Llama 2-7B_>_in|5.8|1.2|8.6|Llama 2-7B_>_in_>_mt|11.6|5.7|12.6|
|Llama 2-7B_>_ra|7.5|2.5|10.4|Llama 2-7B_>_ra_>_mt|14.7|6.6|13.2|
|Llama 2-7B_>_va|8.7|2.9|12.2|Llama 2-7B_>_va_>_mt|15.7|7.8|14.4|
||(+16.00%)|(+16.00%)|(+17.31%)||(+6.80%)|(+18.18%)|(+18.18%)|



Table 6: Ablation study of different finetuning settings. test: MUSTARDSAUCEtest. va: MUSTARDSAUCE-valid. mt: mathlib training split. 

|MODEL|test|
|---|---|
|GPT2-large _>_ va|8.3|
|GPT2-large _>_ va _>_ mt|10.6|
|GPT2-large _>_ mt _>_ va|9.8|
|Llama 2-7B _>_ va|12.2|
|Llama 2-7B _>_ va _>_ mt|14.4|
|Llama 2-7B _>_ mt _>_ va|13.8|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0008-05.png)


Figure 3: Model performances fine-tuned with different data scales. 

random, the model’s performance still lags behind that of fine-tuning on smaller amounts but higher quality data. Therefore, our proposed framework that introduces the theorem prover is effective and beneficial. Furthermore, complementary experimental results of larger Llama 2-70B are demonstrated in Table 28 in Appendix G. The results suggest that our method remains effective when fine-tuning a larger language model. 

4.4 IMPACT OF DATA SCALABILITY 

To further study the impact of data scale on the fine-tuning results, we randomly sample 75%, 50%, 25%, and 0% data from MUSTARDSAUCE-valid and fine-tune Llama 2-7B. The results are shown in Figure 3. In general, the results on all datasets increase as the fine-tuning data scales up. Specifically, performances on the MUSTARD-test and mathlib have the most significant growth without a decrease in the growth rate. Therefore we expect further performance improvements when more high-quality data are included. 

4.5 PASS RATE 

We study the mathematical generation ability of MUSTARD by investigating its pass rates on generating valid data points. The pass@1 results of the generated formal proofs of GPT-4 OpenAI (2023) and GPT-3.5 OpenAI (2022) are shown in Table 7. We have the following observations. First of all, the overall pass@1 results are high, showing the LLMs especially GPT-4 capable of performing zero-shot mathematical reasoning. Second, the pass rates of word problems are generally higher than those of theorem proving. It indicates that the word problems are relatively easier and more familiar to the LLMs, while theorem proving is more challenging. Third, the all-at-once generation and step-by-step generation have similar pass rates at lower educational levels. For more challenging questions such as those at the high school level and higher educational level, step-bystep generation shows slight advantages over the all-at-once generation. This indicates that dividing and conquering (T1), (T2), and (T3) helps the model to generate higher-quality formal proofs, but the improvement is limited. Last but not least, the improvements in pass rates after 1-step and 2- 

8 

Published as a conference paper at ICLR 2024 

Table 7: Pass@1 of formal proofs generated by MUSTARD. k: Number of seed concepts. 

||Thoerem Proving<br>All (GPT-4)<br>Step<br>(GPT-<br>4)<br>All<br>(GPT-<br>3.5)<br>#correct=0<br>1 (∆)<br>2 (∆)<br>0<br>0|Word Problem<br>All (GPT-4)<br>Step<br>(GPT-<br>4)<br>All<br>(GPT-<br>3.5)|
|---|---|---|
|||0<br>1 (∆)<br>2 (∆)<br>0<br>0|
|k=1<br>elem<br>26.0<br>48.0 (+22.0)<br>55.9 (+33.9)<br>25.5<br>15.1<br>midd<br>16.4<br>31.8 (+15.4)<br>39.6 (+24.2)<br>17.0<br>4.3<br>high<br>6.8<br>14.4 (+7.6)<br>17.2 (+9.6)<br>6.6<br>1.9<br>higher<br>2.1<br>5.6 (+3.5)<br>9.8 (+6.3)<br>3.0<br>0.7||38.0<br>59.7 (+21.7)<br>67.0 (+45.3)<br>40.1<br>22.2<br>22.9<br>39.7 (+16.8)<br>47.4 (+30.6)<br>28.4<br>7.0<br>6.7<br>16.8 (+10.1)<br>21.9 (+11.8)<br>8.1<br>3.4<br>3.6<br>10.9 (+7.3)<br>16.0 (+8.7)<br>4.3<br>2.7|
|k=2<br>elem<br>24.1<br>42.3 (+18.2)<br>52.3 (+34.1)<br>23.2<br>12.8<br>midd<br>14.0<br>25.2 (+11.2)<br>34.1 (+22.9)<br>15.0<br>5.3<br>high<br>3.8<br>8.3 (+4.5)<br>12.1 (+7.6)<br>5.4<br>2.2<br>higher<br>1.1<br>3.3 (+2.2)<br>5.0 (+2.8)<br>2.6<br>1.3||32.2<br>49.5 (+17.3)<br>58.2 (+40.9)<br>31.9<br>22.1<br>16.9<br>27.3 (+10.4)<br>34.5 (+24.1)<br>17.3<br>5.7<br>5.7<br>11.0 (+5.3)<br>16.2 (+10.9)<br>4.5<br>2.5<br>2.6<br>7.0 (+4.4)<br>10.5 (+6.1)<br>2.9<br>2.1|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0009-03.png)


Figure 4: Distributions of formal proof lengths. 

step corrections are significant. For example, theorem proving at elementary-school level with 1 seed concept improves by 22.0% after 1-step correction, and 33.9% after 2-step correction. Word problem at the elementary-school level with 1 seed concept improves 45.3% after 2-step correction. The most difficult setting of generating theorem proving data at the higher-educational level with 2 seed concepts achieves 2.8% improvement after 2-step correction, and the word problem counterpart achieves 6.1% improvement. This indicates that the LLMs have a great potential for self-correction given the error message feedback and limited instructions. 

## 4.6 DIVERSITY AND DIFFICULTY 

We compute ROUGE-L (Lin, 2004) to check the diversity of generated informal statements and proofs. The resulting ROUGE-L scores are below 0.25 and indicate high data diversity. We demonstrate detailed computation and results in Appendix D.2. We then investigate the proof lengths in MUSTARDSAUCE and the distributions are demonstrated in Figure 4. We count both reasoning steps of formal statement-proof pairs and steps of formal proof only, which are shown on the leftand right-hand side of Figure 4, respectively. It is demonstrated that proof length increases over educational levels. Solving elementary problems needs about 5 to 10 steps while solving highereducational problems requires a median number between 10 to 15 steps. The most challenging problems require around 30 reasoning steps or about 20 formal proof steps. Therefore, MUSTARDSAUCE produces diverse mathematical problems with multiple topics and difficulties. 

## 5 CONCLUSION 

In this paper, we introduce MUSTARD to automatically generate mathematical datasets with highquality solutions that cover a variety of mathematical skills. Leveraging the LLM and Lean Prover, MUSTARD can generate the problem statement, informal solution, and formal solution. and use a Lean Prover to automatically verify the formal solution and provide feedback for revision. At last, we apply the proposed MUSTARD and obtain 5,866 problems with step-by-step solutions that cover different educational levels and mathematical abilities. The obtained dataset has shown its high quality, diversity, and effectiveness in improving language models’ mathematical reasoning performance, showing the great potential of our proposed MUSTARD and dataset in further research on language models. 

9 

Published as a conference paper at ICLR 2024 

## ACKNOWLEDGEMENTS 

This work was supported in part by the National Key R&D Program of China under Grant No. 2020AAA0109700, Guangdong Outstanding Youth Fund (Grant No. 2021B1515020061), National Natural Science Foundation of China (NSFC) under Grant No.61976233, Mobility Grant Award under Grant No. M-0461, Shenzhen Science and Technology Program (Grant No. GJHZ20220913142600001), Nansha Key RD Program under Grant No.2022ZD014, National Natural Science Foundation of China under Grant No.62006255, and the National Natural Science Foundation of China under Grant 62371411, the Research Grants Council of the Hong Kong SAR under Grant GRF 11217823, InnoHK initiative, the Government of the HKSAR, Laboratory for AIPowered Financial Technologies. We thank MindSpore for the partial support of this work, which is a new deep learning computing framework[5] . 

The authors would also like to thank Hui Jin, Jianhao Shen, Chengwu Liu, Cen Li, and Junhao Cheng for their hard work on the manual check related to Section 4.2. 

## REFERENCES 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168, 2021. 

- Leonardo Mendonc¸a de Moura, Soonho Kong, Jeremy Avigad, Floris van Doorn, and Jakob von Raumer. The lean theorem prover (system description). In Amy P. Felty and Aart Middeldorp (eds.), _Automated Deduction - CADE-25 - 25th International Conference on Automated Deduction, Berlin, Germany, August 1-7, 2015, Proceedings_ , volume 9195 of _Lecture Notes in Computer Science_ , pp. 378–388. Springer, 2015. doi: 10.1007/978-3-319-21401-6 _\_ ~~2~~ 6. URL https://doi.org/10.1007/978-3-319-21401-6_26. 

- Olga Golovneva, Moya Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam FazelZarandi, and Asli Celikyilmaz. ROSCOE: A suite of metrics for scoring step-by-step reasoning. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL https://openreview.net/pdf? id=xYlJRpzZtsY. 

- Jesse Michael Han, Jason Rute, Yuhuai Wu, Edward W. Ayers, and Stanislas Polu. Proof artifact co-training for theorem proving with language models. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net, 2022. URL https://openreview.net/forum?id=rpxJc9j04U. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021. 

> 5https://www.mindspore.cn/ 

10 

Published as a conference paper at ICLR 2024 

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- Albert Qiaochu Jiang, Sean Welleck, Jin Peng Zhou, Timoth´ee Lacroix, Jiacheng Liu, Wenda Li, Mateja Jamnik, Guillaume Lample, and Yuhuai Wu. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL https://openreview.net/pdf?id=SMa9EAovKMC. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html. 

- Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation. _CoRR_ , abs/2308.06259, 2023. doi: 10.48550/arXiv.2308.06259. URL https://doi.org/10.48550/arXiv.2308.06259. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013. 

- Chengwu Liu, Jianhao Shen, Huajian Xin, Zhengying Liu, Ye Yuan, Haiming Wang, Wei Ju, Chuanyang Zheng, Yichun Yin, Lin Li, Ming Zhang, and Qun Liu. FIMO: A challenge formal dataset for automated theorem proving. _CoRR_ , abs/2309.04295, 2023. 

OpenAI. Introducing chatgpt, 2022. 

OpenAI. GPT-4 technical report. _CoRR_ , abs/2303.08774, 2023. 

- Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. _CoRR_ , abs/2009.03393, 2020. URL https://arxiv.org/abs/2009.03393. 

- Stanislas Polu, Jesse Michael Han, Kunhao Zheng, Mantas Baksys, Igor Babuschkin, and Ilya Sutskever. Formal mathematics statement curriculum learning. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL https://openreview.net/pdf?id=-P7G-8dmSh4. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. _CoRR_ , abs/2307.09288, 2023. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288. 

11 

Published as a conference paper at ICLR 2024 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Haiming Wang, Ye Yuan, Zhengying Liu, Jianhao Shen, Yichun Yin, Jing Xiong, Enze Xie, Han Shi, Yujun Li, Lin Li, Jian Yin, Zhenguo Li, and Xiaodan Liang. Dt-solver: Automated theorem proving with dynamic-tree sampling guided by proof-level value function. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pp. 12632–12646. Association for Computational Linguistics, 2023a. doi: 10.18653/ v1/2023.acl-long.706. URL https://doi.org/10.18653/v1/2023.acl-long.706. 

- Haiming Wang, Huajian Xin, Chuanyang Zheng, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, Jian Yin, Zhenguo Li, and Xiaodan Liang. LEGO-prover: Neural theorem proving with growing libraries. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=3f5PALef5B. 

- Mingzhe Wang and Jia Deng. Learning to prove theorems by learning to generate theorems. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ d2a27e83d429f0dcae6b937cf440aeb1-Abstract.html. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023b. URL https://openreview.net/ pdf?id=1PL1NIMMrw. 

- Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pp. 13484–13508. Association for Computational Linguistics, 2023c. doi: 10.18653/v1/2023.acl-long.754. URL https://doi.org/10.18653/ v1/2023.acl-long.754. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/ 2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference. html. 

- Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. Symbolic knowledge distillation: from general language models to commonsense models. In Marine Carpuat, Marie-Catherine de Marneffe, and Iv´an Vladimir Meza Ru´ız (eds.), _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pp. 4602–4625. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.naacl-main.341. URL https://doi.org/10. 18653/v1/2022.naacl-main.341. 

- Yuhuai Wu, Albert Q. Jiang, Jimmy Ba, and Roger Baker Grosse. INT: an inequality benchmark for evaluating generalization in theorem proving. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021. URL https://openreview.net/forum?id=O6LPudowNQm. 

- Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus N. Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. Autoformalization with large language models. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ d0c6bc641a56bebee9d985b937307367-Abstract-Conference.html. 

12 

Published as a conference paper at ICLR 2024 

- Jing Xiong, Jianhao Shen, Ye Yuan, Haiming Wang, Yichun Yin, Zhengying Liu, Lin Li, Zhijiang Guo, Qingxing Cao, Yinya Huang, Chuanyang Zheng, Xiaodan Liang, Ming Zhang, and Qun Liu. TRIGO: Benchmarking formal mathematical proof reduction for generative language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 11594–11632, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.711. URL https://aclanthology.org/2023.emnlp-main.711. 

- Jing Xiong, Zixuan Li, Chuanyang Zheng, Zhijiang Guo, Yichun Yin, Enze Xie, Zhicheng YANG, Qingxing Cao, Haiming Wang, Xiongwei Han, Jing Tang, Chengming Li, and Xiaodan Liang. DQ-lore: Dual queries with low rank approximation re-ranking for in-context learning. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=qAoxvePSlq. 

- Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _CoRR_ , abs/2304.12244, 2023. doi: 10.48550/arXiv.2304.12244. URL https://doi.org/ 10.48550/arXiv.2304.12244. 

- Siyu Yuan, Jiangjie Chen, Ziquan Fu, Xuyang Ge, Soham Shah, Charles Robert Jankowski, Yanghua Xiao, and Deqing Yang. Distilling script knowledge from large language models for constrained language planning. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pp. 4303–4325. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.236. URL https://doi.org/10.18653/v1/2023.acl-long.236. 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. _arXiv preprint arXiv:2304.09797_ , 2023a. 

- Chuanyang Zheng, Haiming Wang, Enze Xie, Zhengying Liu, Jiankai Sun, Huajian Xin, Jianhao Shen, Zhenguo Li, and Yu Li. Lyra: Orchestrating dual correction in automated theorem proving. _arXiv preprint arXiv:2309.15806_ , 2023b. 

- Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: a cross-system benchmark for formal olympiad-level mathematics. In _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net, 2022. 

## A FUTURE WORKS 

Current formal validation still suffers mild inconsistency between the #reduce statements and various kinds of theorem proofs. In the future, we will explore more rigorous and careful data filtering. We will also explore data generation and mathematical reasoning via the same language model, which is an interesting setup to study large language models’ proficiency of mathematical reasoning. Moreover, the ablation study on data scalability shows consistent performance increases when more data from MUSTARD are introduced, suggesting a great potential for scalability. Fortunately, MUSTARD reduces the cost of acquiring such high-quality step-by-step complex reasoning data and obtains correct, scalable, and reusable data. Therefore, in future work, we would love to build a community in which all members can join the data synthesis process, and acquire and share more high-quality data with the whole community. 

## B MATHEMATICAL CONCEPTS 

Table 8 shows the concept statistics. Concepts are grouped into multiple domains mainly according to subjects, except that at the elementary level are grouped by grades due to the lack of subject division. Tables 9, 10, 11 and 12 demonstrate the detailed concepts in each domain. 

13 

Published as a conference paper at ICLR 2024 

Table 8: Statistics of the mathematical concept pool. 

|Elementary School<br>Domains<br>#|Middle School<br>Domains<br>#|High School<br>Domains<br>#|Higher Education|
|---|---|---|---|
||||Domains<br>#|
|1st grade<br>2nd grade<br>3rd grade<br>4th grade<br>5th grade<br>6th grade<br>3<br>8<br>14<br>14<br>16<br>8|7th grade<br>8th grade<br>Algebra basics<br>Pre-algebra<br>Basic geometry<br>and measurement<br>7<br>7<br>8<br>15<br>14|Algebra 1<br>Algebra 2<br>High school<br>geometry<br>Trigonometry<br>Statistics and<br>probability<br>High school statistics<br>Precalculus<br>Calculus 1<br>Calculus 2<br>16<br>12<br>9<br>4<br>16<br>7<br>10<br>8<br>6|AP College Statistics<br>College Algebra<br>Differential Calculus<br>Integral Calculus<br>AP College<br>Calculus AB<br>AP College<br>Calculus BC<br>Multivariable<br>calculus<br>Differential<br>equations<br>Linear algebra<br>14<br>14<br>6<br>5<br>10<br>12<br>5<br>3<br>3|
|# Domain<br>6<br># Concept<br>63|5<br>51|9<br>88|9<br>72|



Table 9: Mathematical concepts at the level of elementary school. 

|**Domain**|**Concept**|
|---|---|
|1st grade<br>2nd grade<br>3rd grade<br>4th grade<br>5th grade<br>6th grade|“Place value”, “Addition and subtraction”, “Measurement, data, and<br>geometry”,|
||“Add and subtract within 20”, “Place value”, “Add and subtract<br>within 100”, “Add and subtract within 1,000”, “Money and time”,<br>“Measurement”, “Data”, “Geometry”,|
||“Intro to multiplication”, “1-digit multiplication”, “Addition,<br>subtraction, and estimation”, “Intro to division”, “Understand<br>fractions”, “Equivalent fractions and comparing fractions”, “More<br>with multiplication and division”, “Arithmetic patterns and problem<br>solving”, “Quadrilaterals”, “Area”, “Perimeter”, “Time”,<br>“Measurement”, “Represent and interpret data”,|
||“Place value”, “Addition, subtraction, and estimation”, “Multiply by<br>1-digit numbers”, “Multiply by 2-digit numbers”, “Division”,<br>“Factors, multiples and patterns”, “Equivalent fractions and<br>comparing fractions”, “Add and subtract fractions”, “Multiply<br>fractions”, “Understand decimals”, “Plane fgures”, “Measuring<br>angles”, “Area and perimeter”, “Units of measurement”,|
||“Decimal place value”, “Add decimals”, “Subtract decimals”, “Add<br>and subtract fractions”, “Multi-digit multiplication and division”,<br>“Multiply fractions”, “Divide fractions”, “Multiply decimals”,<br>“Divide decimals”, “Powers of ten”, “Volume”, “Coordinate plane”,<br>“Algebraic thinking”, “Converting units of measure”, “Line plots”,<br>“Properties of shapes”,|
||“Ratios”, “Arithmetic with rational numbers”, “Rates and<br>percentages”, “Exponents and order of operations”, “Negative<br>numbers”, “Variables & expressions”, “Equations & inequalities”,<br>“Plane fgures”,|



14 

Published as a conference paper at ICLR 2024 

Table 10: Mathematical concepts at the level of middle school. 

|**Domain**|**Concept**|
|---|---|
|7th grade<br>8th grade<br>Algebra basics<br>Pre-algebra<br>Basic geometry<br>and measurement|“Negative numbers: addition and subtraction”, “Negative numbers:<br>multiplication and division”, “Fractions, decimals, & percentages”,<br>“Rates & proportional relationships”, “Expressions, equations, &<br>inequalities”, “Geometry”, “Statistics and probability”,|
||“Numbers and operations”, “Solving equations with one unknown”,<br>“Linear equations and functions”, “Systems of equations”,<br>“Geometry”, “Geometric transformations”, “Data and modeling”,|
||“Foundations”, “Algebraic expressions”, “Linear equations and<br>inequalities”, “Graphing lines and slope”, “Systems of equations”,<br>“Expressions with exponents”, “Quadratics and polynomials”,<br>“Equations and geometry”,|
||“Factors and multiples”, “Patterns”, “Ratios and rates”,<br>“Percentages”, “Exponents intro and order of operations”, “Variables<br>& expressions”, “Equations & inequalities introduction”, “Percent &<br>rational number word problems”, “Proportional relationships”,<br>“One-step and two-step equations & inequalities”, “Roots, exponents,<br>& scientifc notation”, “Multi-step equations”, “Two-variable<br>equations”, “Functions and linear models”, “Systems of equations”,|
||“Intro to area and perimeter”, “Intro to mass and volume”,<br>“Measuring angles”, “Plane fgures”, “Units of measurement”,<br>“Volume”, “Coordinate plane”, “Decomposing to fnd area”, “3D<br>fgures”, “Circles, cylinders, cones, and spheres”, “Angle<br>relationships”, “Scale”, “Triangle side lengths”, “Geometric<br>transformations”,|



## C PROMPT TEMPLATES IN MUSTARD 

## C.1 PROMPT TEMPLATE FOR PROOF FILTERING 

Table 13 demonstrates the prompt template used in the proof-filtering stage in MUSTARD. 

## C.2 PROMPT TEMPLATES FOR STEP-BY-STEP GENERATION 

Table 14 demonstrates the variation of prompt templates used in the proof-generation stage in MUSTARD. In this variation, an LLM is conducted to perform (T1), (T2), and (T3) separately to generate the informal statement, informal solution, and formal solution. It is noted that to prompt the LLM to fulfill (T3), we assign the character of “a master in Lean” rather than the previous “a math expert” to obtain higher quality Lean proofs. 

15 

Published as a conference paper at ICLR 2024 

Table 11: Mathematical concepts at the level of high school. 

|**Domain**|**Concept**|
|---|---|
|Algebra 1<br>Algebra 2<br>High school geometry<br>Trigonometry<br>Statistics and probability<br>High school statistics<br>Precalculus<br>Calculus 1<br>Calculus 2|“Algebra foundations”, “Solving equations & inequalities”,<br>“Working with units”, “Linear equations & graphs”, “Forms of linear<br>equations”, “Systems of equations”, “Inequalities (systems &<br>graphs)”, “Functions”, “Sequences”, “Absolute value & piecewise<br>functions”, “Exponents & radicals”, “Exponential growth & decay”,<br>“Quadratics: Multiplying & factoring”, “Quadratic functions &<br>equations”, “Irrational numbers”, “Creativity in algebra”,|
||“Polynomial arithmetic”, “Complex numbers”, “Polynomial<br>factorization”, “Polynomial division”, “Polynomial graphs”,<br>“Rational exponents and radicals”, “Exponential models”,<br>“Logarithms”, “Transformations of functions”, “Equations”,<br>“Trigonometry”, “Modeling”,|
||“Performing transformations”, “Transformation properties and<br>proofs”, “Congruence”, “Similarity”, “Right triangles &<br>trigonometry”, “Analytic geometry”, “Conic sections”, “Circles”,<br>“Solid geometry”,|
||“Right triangles & trigonometry”, “Trigonometric functions”,<br>“Non-right triangles & trigonometry”, “Trigonometric equations and<br>identities”,|
||“Analyzing categorical data”, “Displaying and comparing<br>quantitative data”, “Summarizing quantitative data”, “Modeling data<br>distributions”, “Exploring bivariate numerical data”, “Study design”,<br>“Probability”, “Counting, permutations, and combinations”,<br>“Random variables”, “Sampling distributions”, “Confdence<br>intervals”, “Signifcance tests (hypothesis testing)”, “Two-sample<br>inference for the difference between groups”, “Inference for<br>categorical data (chi-square tests)”, “Advanced regression (inference<br>and transforming)”, “Analysis of variance (ANOVA)”,|
||“Displaying a single quantitative variable”, “Analyzing a single<br>quantitative variable”, “Two-way tables”, “Scatterplots”, “Study<br>design”, “Probability”, “Probability distributions & expected value”,|
||“Composite and inverse functions”, “Trigonometry”, “Complex<br>numbers”, “Rational functions”, “Conic sections”, “Vectors”,<br>“Matrices”, “Probability and combinatorics”, “Series”, “Limits and<br>continuity”,|
||“Limits and continuity”, “Derivatives: defnition and basic rules”,<br>“Derivatives: chain rule and other advanced topics”, “Applications of<br>derivatives”, “Analyzing functions”, “Integrals”, “Differential<br>equations”, “Applications of integrals”,|
||“Integrals review”, “Integration techniques”, “Differential<br>equations”, “Applications of integrals”, “Parametric equations, polar<br>coordinates, and vector-valued functions”, “Series”,|



## D MORE STATISTIC RESULTS OF MUSTARDSAUCE 

## D.1 DIFFICULTY OF MUSTARDSAUCE BY NUMBER OF CORRECTION 

Figure 5 demonstrates the proportions of data points that obtain valid proof after different numbers of corrections. Generally speaking, data points without correction are relatively less difficult for the LLMs, while those that require multiple corrections are challenging. Overall, theorem-proving problems are more challenging for LLMs to solve than the generated word problems. For data 

16 

Published as a conference paper at ICLR 2024 

Table 12: Mathematical concepts at the level of higher education. 

|**Domain**|**Concept**|
|---|---|
|AP College Statistics<br>College Algebra<br>Differential Calculus<br>Integral Calculus<br>AP College Calculus AB<br>AP College Calculus BC<br>Multivariable calculus<br>Differential equations<br>Linear algebra|“Exploring categorical data”, “Exploring one-variable quantitative<br>data: Displaying and describing”, “Exploring one-variable<br>quantitative data: Summary statistics”, “Exploring one-variable<br>quantitative data: Percentiles, z-scores, and the normal distribution”,<br>“Exploring two-variable quantitative data”, “Collecting data”,<br>“Probability”, “Random variables and probability distributions”,<br>“Sampling distributions”, “Inference for categorical data:<br>Proportions”, “Inference for quantitative data: Means”, “Inference<br>for categorical data: Chi-square”, “Inference for quantitative data:<br>slopes”, “Prepare for the 2022 AP Statistics Exam”,|
||“Linear equations and inequalities”, “Graphs and forms of linear<br>equations”, “Functions”, “Quadratics: Multiplying and factoring”,<br>“Quadratic functions and equations”, “Complex numbers”,<br>“Exponents and radicals”, “Rational expressions and equations”,<br>“Relating algebra and geometry”, “Polynomial arithmetic”,<br>“Advanced function types”, “Transformations of functions”,<br>“Rational exponents and radicals”, “Logarithms”,|
||“Limits and continuity”, “Derivatives: defnition and basic rules”,<br>“Derivatives: chain rule and other advanced topics”, “Applications of<br>derivatives”, “Analyzing functions”, “Parametric equations, polar<br>coordinates, and vector-va”,|
||“Integrals”, “Differential equations”, “Applications of integrals”,<br>“Parametric equations, polar coordinates, and vector-valued<br>functions”, “Series”,|
||“Limits and continuity”, “Differentiation: defnition and basic<br>derivative rules”, “Differentiation: composite, implicit, and inverse<br>functions”, “Contextual applications of differentiation”, “Applying<br>derivatives to analyze functions”, “Integration and accumulation of<br>change”, “Differential equations”, “Applications of integration”, “AP<br>Calculus AB solved free response questions from past exams”, “AP<br>Calculus AB Standards mappings”,|
||“Limits and continuity”, “Differentiation: defnition and basic<br>derivative rules”, “Differentiation: composite, implicit, and inverse<br>functions”, “Contextual applications of differentiation”, “Applying<br>derivatives to analyze functions”, “Integration and accumulation of<br>change”, “Differential equations”, “Applications of integration”,<br>“Parametric equations, polar coordinates, and vector-valued<br>functions”, “Infnite sequences and series”, “AP Calculus BC solved<br>exams”, “AP Calculus BC Standards mappings”,|
||“Thinking about multivariable functions”, “Derivatives of<br>multivariable functions”, “Applications of multivariable derivatives”,<br>“Integrating multivariable functions”, “Green’s, Stokes’, and the<br>divergence theorems”,|
||“First order differential equations”, “Second order linear equations”,<br>“Laplace transform”,|
||“Vectors and spaces”, “Matrix transformations”, “Alternate<br>coordinate systems (bases)”,|



points with 2 seed concepts, more than 90% of the data can not pass the prover validation at the first generation. And almost 30% of them cost 2 correction steps to obtain valid proof. Similar 

17 

Published as a conference paper at ICLR 2024 

Table 13: Prompt template for proof filtering. 

**Prompt Template for Proof Filtering** 

In the following, you are given a ‘‘Problem’’, a pair of corresponding ‘‘Informal proof’’ and ‘‘Formal proof in Lean 3’’, along with error messages from a Lean Prover corresponding to the ‘‘Formal proof in Lean 3’’. Now please carefully modify the ‘‘Formal proof in Lean 3’’ section so that it passes the Lean Prover without error. You should write the modified complete proof in your response. 

# Problem: <generated problem> # Informal proof: <generated informal proof> # Formal proof (1) in Lean 3: ‘‘‘lean line 1 <code> line 2 <code> line 3 <code> ... ‘‘‘ # Error messages for Formal proof (1) from Lean Prover: <error messages> ... # Formal proof (k) in Lean 3: ‘‘‘lean line 1 <code> line 2 <code> line 3 <code> ... ‘‘‘ # Error messages for Formal proof (k) from Lean Prover: <error messages> 

observations are found in word problems with 2 seed concepts. It is suggested that the data subset with 2 seed concepts is challenging to the LLMs in general. In contrast, data with 1 seed concept are easier for LLMs. But there are still more than half data points that need proof improvements based on error messages from the theorem prover. Therefore, overall, MUSTARD generates valid data points in different difficulty levels, and the majority of the problems are challenging for the LLMs. 

## D.2 DATA DIVERSITY 

We compute ROUGE-L (Lin, 2004) to check the diversity of generated informal statements and proofs. Specifically, given a data set, we perform 10 rounds of bootstrapping. In each round, we randomly sample 10 data points from the data set, each of which is paired with the remaining data points, and compute pair-wise ROUGE-L scores. The ROUGE-L score per round is obtained by averaging the pair-wise scores. The final ROUGE-L score is an average score over the bootstrapping. We compare the scores among the generation settings, and the results are shown in Figure 6. 

The results show that all settings have a ROUGE-L score beneath 0.25, which indicates a high diversity of the generated informal statements and informal proofs. All-at-once and step-by-step generation share similar data diversity. The ROUGE-L scores slightly increase over the educational levels. Generating higher-educational theorem proving data with 2 concepts step-by-step exceeds 0.2 in ROUGE-L. Therefore, this setting is relatively challenging. 

18 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0019-01.png)


Figure 5: Data proportions according to correction steps. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0019-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0019-04.png)


Figure 6: Diversity of informal statements and proofs by ROUGE. All: All-at-once generation. Step: Step-by-step generation. k: Number of seed concepts. 

We further investigate the diversity of formal statements and proofs. We collect all the occurrences of used tactics and lemmas, and Figure 7 shows the distributions. It is demonstrated that LLMs tend to use diverse lemmas to solve the problems. And LLMs memorize and understand multiple lemmas. The most commonly used lemmas in both theorem proving and word problems include of ~~a~~ s ~~t~~ rue, mul ~~c~~ omm, and nat.mul ~~c~~ omm. of ~~a~~ t ~~t~~ rue is often used in automated proofs for complex propositions to first prove their decidability. mul ~~c~~ omm and nat.mul ~~c~~ omm are about the multiplicative commutative law and are often used for expression simplification. Similarly, the frequently used neg ~~p~~ os, pow ~~t~~ wo, nat.div ~~e~~ q ~~o~~ f ~~e~~ q ~~m~~ ul ~~r~~ ight and so forth are basic lemmas for proving advanced properties. Similar observations are found in tactics. Therefore, LLMs are good at breaking problems into basic steps and using basic lemmas to complete complex proving. 

19 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0020-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0020-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data_images/iclr_2024_MUSTARD_Mastering_Uniform_Synthesis_of_Theorem_and_Proof_Data.pdf-0020-04.png)


Figure 7: Distribution of tactics (left-hand side) and lemmas (right-hand side) used by the formal statements and proofs in theorem proving (upper row) and word problem (lower row). 

## E CASE STUDY 

Tables 15 to 26 demonstrates successful and failed cases generated by MUSTARD. 

20 

Published as a conference paper at ICLR 2024 

## F IMPLEMENTATION DETAILS OF DOWNSTREAM TASK 

## F.1 DATASETS 

**GSM8K (Cobbe et al., 2021)** GSM8K consists of 8.5K elementary mathematics word problems that require 2 to 8 arithmetic operations to arrive at the final answer. The dataset comprises 7.5K training questions and 1K test questions. Inspired by Kojima et al. (2022), during inference, we use appropriate prompts and examples to prompt the model for zero-shot and few-shot reasoning. The used prompts are demonstrated in Table 27. 

**Mathlib** Mathlib[6] is a community-maintained library designed for the Lean theorem prover. It encompasses both programming tools and mathematical content, along with tactics that leverage these tools to facilitate mathematical development. The version of Mathlib we use is consistent with Wang et al. (2023a). The lengths of the training, test, and validation sets were 36,960, 1,621, and 1,580, respectively. 

**miniF2F (Zheng et al., 2022)** MiniF2F serves as a formal mathematics benchmark, which has been translated to work with multiple formal systems. It encompasses exercise statements from olympiads like AMC, AIME, and IMO, in addition to content from high-school and undergraduate mathematics courses. The MiniF2F test split contains 244 formal Olympiad-level mathematics problem statements. We use the auto-regression method to construct the training corpus like Polu & Sutskever (2020); Han et al. (2022); Wang et al. (2023a). We perform an evaluation using best-first search with a number of expansions per proof search of _d_ = 8 during inference. 

## F.2 MODELS 

**GPT2-large (Radford et al., 2019)** The GPT2-large model is a transformer language model, following the decoder-only architecture introduced by Vaswani et al. (2017). The model boasts an impressive 774 million parameters, 36 layers, 20 attention heads, and a hidden dimension of 1,280. Additionally, it employs a tokenizer featuring a vocabulary size of 50,400. The model is pre-trained on Github python codes and the arXiv library. 

**Llama 2-7B (Touvron et al., 2023)** Llama 2 is a language model that employs an auto-regressive transformer architecture, pre-trained on open-source corpus. The model utilizes both supervised fine-tuning and reinforcement learning with human feedback techniques to align with human preferences. The 7B model is configured with 32 layers, 32 attention heads, and a hidden dimension of 4,096. 

## F.3 IMPLEMENTATION DETAILS 

We employ LoRA (Hu et al., 2021) for fine-tuning the pre-trained models on MUSTARDSAUCE, where the trainable parameters of GPT2-large and Llama 2-7B constitute 19% and 6%, respectively. The training is conducted with a maximum of 10 epochs using a batch size of 16 and a warm-up step of 1,000, with a maximum learning rate of 1e-4 and a minimum learning rate of 5e-6. The best checkpoint is selected based on the minimum perplexity of the validation split. 

## G MORE EXPERIMENTAL RESULTS 

Table 28 demonstrates the compared results on GSM8K and MATH between Llama 2-7B and Llama 2-70B. Llama 2-70B fine-tuned with MUSTARDSAUCE-valid consistently outperforms the model fine-tuned with MUSTARDSAUCE-random by 8.33% in the zero-shot manner and 5.30% in the few-shot manner. It also surpasses the model fine-tuned with the invalid subset and the entire generated dataset. The results also suggest the effectiveness of the framework with a larger fine-tuned LM. 

> 6https://github.com/leanprover-community/mathlib 

21 

Published as a conference paper at ICLR 2024 

## H DATA CONTAMINATION CHECK 

We check cross-contamination between MUSTARDSAUCE and the evaluation datasets inspired by OpenAI (2023). However, instead of using a substring match that may result in false negatives and false positives, we compute cosine similarities based on text-embedding-ada-002[7] , and then inspect the nearest data points in the paired datasets. The automated theorem proving (ATP) dataset miniF2F only contains formal statements and proofs, while the math word problem (MWP) dataset GSM8K only contains informal statements and proofs. For a more detailed inspection, we check the corresponding fractions in MUSTARDSAUCE. Tables 29, 32, 31, and 30 demonstrate the inspected cases. The nearest data points are dissimilar. Therefore, we exclude data contamination of the generated MUSTARDSAUCE according to these observations. 

> 7https://openai.com/blog/new-and-improved-embedding-model 

22 

Published as a conference paper at ICLR 2024 

Table 14: Prompt templates for step-by-step generation in the proof-generation stage. 

## **Prompt Templates for Step-by-Step Generation** 

You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ‘‘Problem: ’’), a corresponding solution in natural language (indicated by ‘‘Informal proof:’’), and a translated formal solution in Lean 3 (indicated by ‘‘Formal proof in Lean 3:’’). Please note that the informal proof and the formal proof need to be identical. Please create a [QUESTION TYPE] in the level of [EDUCATIONAL LEVEL] based on the following knowledge point(s): [CONCEPT] in [DOMAIN]; [CONCEPT] in [DOMAIN]. (T1) Please first write the question part regardless of the other parts. You must write the following format, filling in the ‘‘# Problem: ’’ section, and leaving the other two sections empty. # Problem: ... # Informal proof: ... # Formal proof in Lean 3: ... You are a math expert. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ‘‘Problem: ’’), a corresponding solution in natural language (indicated by ‘‘Informal proof:’’), and a translated formal solution in Lean 3 (indicated by ‘‘Formal proof in Lean 3:’’). Please note that the informal proof and the formal proof need to be identical. Please create a [QUESTION TYPE] in the level of [EDUCATIONAL LEVEL] based on the following knowledge point(s): [CONCEPT] in [DOMAIN]; [CONCEPT] in [DOMAIN]. (T2) Please then write the corresponding solution in natural language (indicated by ‘‘Informal proof:’’) given the ‘‘# Problem: ’’, filling in the ‘‘# Informal proof: ’’ section, and leaving the other section empty. # Problem: <generated problem> # Informal proof: ... # Formal proof in Lean 3: ... You are a master in Lean. Now please come up with a math problem according to the following requirements. The math problem should contain a question part (indicated by ‘‘Problem: ’’), a corresponding solution in natural language (indicated by ‘‘Informal proof:’’), and a translated formal solution in Lean 3 (indicated by ‘‘Formal proof in Lean 3:’’). Please note that the informal proof and the formal proof need to be identical. Please create a [QUESTION TYPE] in the level of [EDUCATIONAL LEVEL] based on the following knowledge point(s): [CONCEPT] in [DOMAIN]; [CONCEPT] in (T3) [DOMAIN]. Please translate the ‘‘# Informal proof:’’ section into Lean 3 and fill in the ‘‘# Formal proof in Lean 3: ’’ section. # Problem: <generated problem> # Informal proof: <generated informal proof> # Formal proof in Lean 3: ... 

23 

Published as a conference paper at ICLR 2024 

Table 15: A successful case generated by MUSTARD. 

**Question Type: Word Problem. Educational Level: Elementary School. k=1. Successful case. Concept(s):** _Division_ in 4th grade. **Comments:** MUSTARD accurately counts the number of people and correctly performs division. 

## **Informal Statement** 

John and his 3 friends are on a trip. They bought a huge box of chocolates containing 240 pieces. They decided to split the chocolates equally among themselves. How many chocolates will each one of them get? 

## **Informal Proof** 

Since there are 4 people in total and they have 240 chocolates, we can find out how many chocolates each person will get by dividing the total number of chocolates by the total number of people. So, 240 divided by 4 equals 60. Therefore, each person will get 60 chocolates. 

## **Formal Statement and Proof** 

**def** divide (x y : N) : N := x / y 

#reduce divide 240 4 _-- The output should be 60, which means each person will get 60 chocolates._ 

Table 16: A failed cased generated by MUSTARD. 

|**Question Type: Theorem Proving.**<br>**Educational Level: Elementary School.**|**k=1.**|**Failed case.**|
|---|---|---|
|**Concept(s):** _Quadrilaterals_in 3rd grade.|||
|**Comments:**<br>MUSTARD believes that a quadrilateral with four equal sides is a|square. It treats a necessary||
|condition as a suffcient condition.|||



## **Informal Statement** 

Given a quadrilateral ABCD where AB = BC = CD = DA = 5 units, prove that this quadrilateral is a square. **Informal Proof** The given quadrilateral ABCD has all its sides equal. This is a property of a square. Hence, ABCD is a square. 

## **Formal Statement and Proof** 

**import** data.real.basic **def** is ~~s~~ quare ~~q~~ uad (AB BC CD DA : R) : Prop := AB = BC _∧_ BC = CD _∧_ CD = DA **def** ABCD ~~i~~ s ~~s~~ quare : _∀_ (AB BC CD DA : R), AB = 5 _∧_ BC = 5 _∧_ CD = 5 _∧_ DA = 5 _→_ is ~~s~~ quare ~~q~~ uad AB BC CD DA := **begin** intros AB BC CD DA h, cases h **with** h1 h2, cases h2 **with** h3 h4, cases h4 **with** h5 h6, split, _{_ rw h1, exact h3 _}_ , split, _{_ rw h3, exact h5 _}_ , _{_ rw h5, exact h6 _}_ , **end** 

24 

Published as a conference paper at ICLR 2024 

Table 17: A successful case generated by MUSTARD. 

**Question Type: Word Problem. Educational Level: Elementary School. k=2. Successful case. Concept(s):** _Addition and subtraction_ in 1st grade; Measuring angles in 4th grade. **Comments:** MUSTARD creatively combines the two concepts and constructs a reasonable question. **Informal Statement** John has a favorite toy car that he likes to move around a circular track. He starts moving the car from a certain point and stops after the car has moved 90 degrees. Then he moves the car again but this time in the opposite direction, stopping when the car has moved 45 degrees. What is the net degree movement of John’s toy car? **Informal Proof** Initially, John moves his car 90 degrees. Then he moves the car in the opposite direction 45 degrees. Since the car is moving in the opposite direction, we consider this as -45 degrees. To find the net movement, we add the two movements together. So, 90 + (-45) equals 45. Therefore, the net degree movement of John’s toy car is 45 degrees. 

## **Formal Statement and Proof** 

**import** data.int.basic **open** int _-- define the initial movement_ **def** initial ~~m~~ ovement : Z := 90 _-- define the backward movement_ **def** backward ~~m~~ ovement : Z := -45 _-- define the function to calculate the net movement_ **def** net ~~m~~ ovement (initial backward : Z) : Z := initial + backward _-- proof that the net movement is 45 degrees_ **example** : net ~~m~~ ovement initial ~~m~~ ovement backward ~~m~~ ovement = 45 := **begin** unfold net ~~m~~ ovement, unfold initial ~~m~~ ovement, unfold backward ~~m~~ ovement, exact dec ~~t~~ rivial, **end** 

25 

Published as a conference paper at ICLR 2024 

Table 18: A successful case generated by MUSTARD. 

**Question Type: Theorem Proving. Educational Level: Middle School. k=2. Successful case. Concept(s):** _Geometry_ in 8th grade; _Algebraic expressions_ in Algebra basics. 

**Comments:** MUSTARD understands polynomials and writes a valid proof in Lean. 

## **Informal Statement** 

Given a rectangle ABCD where AB is x + 5 and AD is 2x - 3. Prove that the area of the rectangle ABCD is (2 _x_[2] + 7 _x −_ 15) square units. 

## **Informal Proof** 

In a rectangle, the area is calculated by multiplying the length and the width. Here, the length is represented by the expression _x_ + 5 and the width is represented by the expression 2 _x −_ 3. By multiplying these two expressions together, we get: 

( _x_ + 5) _∗_ (2 _x −_ 3) = 2 _x_[2] _−_ 3 _x_ + 10 _x −_ 15 = 2 _x_[2] + 7 _x −_ 15 Therefore, the area of the rectangle ABCD is (2 _x_[2] + 7 _x −_ 15) square units. 

## **Formal Statement and Proof** 

**import** algebra.ring.basic **variables** (x : Z) **def** rectangle ~~a~~ rea : (x + 5) * (2 * x - 3) = 2 * (x * x) + 7 * x - 15 := **begin have** h1 : (x + 5) * (2 * x - 3) = 2 * (x * x) - 3 * x + 10 * x - 15, **by** ring, **have** h2 : 2 * (x * x) - 3 * x + 10 * x - 15 = 2 * (x * x) + 7 * x - 15, **by** simp, rw h1, exact h2, **end** 

26 

Published as a conference paper at ICLR 2024 

Table 19: A failed case generated by MUSTARD. 

**Question Type: Theorem Proving. Educational Level: High School. k=1. Failed case. Concept(s):** _Displaying and comparing quantitative data_ in Statistics and probability. **Comments:** MUSTARD gives the definition of median in the formal proof. 

## **Informal Statement** 

Given two sets of data A = _{_ 5, 7, 10, 12, 15 _}_ and B = _{_ 6, 8, 11, 14, 16 _}_ , we are to find the median of the two sets and prove that the median of set A is less than the median of set B. 

## **Informal Proof** 

To find the median of a given data set, we need to sort the data in ascending order and then find the middle value. In the case of set A and set B, both are already sorted in ascending order. For set A, the median is 10 and for set B, the median is 11. Therefore, it is clear that the median of set A is less than the median of set B. 

## **Formal Statement and Proof** 

**import** data.set.finite **import** data.set.lattice **def** A : finset N := 5, 7, 10, 12, 15.to ~~f~~ inset **def** B : finset N := 6, 8, 11, 14, 16.to ~~f~~ inset **def** median (A : finset N) : N := **if** A.card % 2 = 1 **then** (A.sort ( _≤_ )).get ((A.card - 1) / 2) **else** (A.sort ( _≤_ )).get (A.card / 2 - 1) / 2 + (A.sort ( _≤_ )).get (A.card / 2) / 2 **lemma** median ~~A l~~ ess ~~t~~ han ~~B~~ : (median A) < (median B) := **begin** simp [median, A, B, finset.sort, finset.get], norm ~~n~~ um, **end** 

27 

Published as a conference paper at ICLR 2024 

Table 20: A failed case generated by MUSTARD. 

**Question Type: Word Problem. Educational Level: High School. k=2. Failed case. Concept(s):** _Integrals_ in Calculus 1; _Summarizing quantitative data_ in Statistics and probability. 

**Comments:** The informal statement does not cover the situation of leap year. The solution does not give the final answer. 

## **Informal Statement** 

A company produced a new model of a car and started selling it on January 1. The number of cars sold _t_ days after January 1 is given by the function _f_ ( _t_ ) = 2 _t_[2] + 3 _t_ + 1. Use integrals to find the total number of cars sold from January 1 to March 31. Then calculate the average number of cars sold per day during this period. 

## **Informal Proof** 

The total number of cars sold from January 1 to March 31 can be found by integrating the function _f_ ( _t_ ) from _t_ = 0 (January 1) to _t_ = 90 (March 31), as each day corresponds to one unit of _t_ . 

Performing the integration, we get � (2 _t_[2] + 3 _t_ + 1) d _t_ from 0 to 90, which equals [ 3[2] _[×][ t]_[3][ +][3] 2 _[×][ t]_[2][ +] _[ t]_[]][ from] 0 to 90. Substituting _t_ = 90 into the integrated function and subtracting the value at _t_ = 0, we get the total number of cars sold. 

The average number of cars sold per day is then the total number of cars sold divided by 90 (the number of days from January 1 to March 31). 

## **Formal Statement and Proof** 

**import** data.real.basic 

**open** real **def** f := _λ_ (t : R), 2*t[2] + 3*t + 1 **lemma** f ~~v~~ alue ~~a~~ t ~~9~~ 0 : f 90 = 2*(90)[2] + 3*(90) + 1 := **begin** simp [f], **end** 

28 

Published as a conference paper at ICLR 2024 

Table 21: A successful case generated by MUSTARD. 

**Question Type: Theorem Proving. Educational Level: Higher Education. k=1. Successful case. Concept(s):** _Alternate coordinate systems (bases)_ in Linear algebra. **Comments:** MUSTARD correctly applies the lemma. All the concepts of linear algebra are found. 

**Informal Statement** Let _V_ be a vector space, and let _B_ 1 = _v_ 1 _, v_ 2 _, ..., vn_ and _B_ 2 = _w_ 1 _, w_ 2 _, ..., wn_ be two bases for _V_ . Prove that there exists an invertible linear map _T_ : _V → V_ such that _T_ ( _vi_ ) = _wi_ for each _i_ in 1 _,_ 2 _, ..., n_ . 

## **Informal Proof** 

The map _T_ : _V → V_ defined by _T_ ( _vi_ ) = _wi_ for each _i_ is linear: indeed, for any vectors _v_ , _v[′]_ in _B_ 1 and any scalars _c_ , _c[′]_ , we have _T_ ( _cv_ + _c[′] v[′]_ ) = _cw_ + _c[′] w[′]_ where _w_ = _T_ ( _v_ ) and _w[′]_ = _T_ ( _v[′]_ ). This follows from the linearity properties of vector addition and scalar multiplication in the vector space _V_ . 

## **Formal Statement and Proof** 

**import** linear ~~a~~ lgebra.basis **variables** V : **Type** * [add ~~c~~ omm ~~g~~ roup V] [module R V] **variables** v w : V **variables** B1 B2 : set V **variables** (hB1 : is ~~b~~ asis R B1) (hB2 : is ~~b~~ asis R B2) **lemma** exists ~~i~~ nvertible ~~l~~ inear ~~m~~ ap : _∃_ (T : V _→_ [R] V) (hT : is ~~l~~ inear ~~m~~ ap R T) (hinv : function.bijective T), _∀_ (v _∈_ B1), T v _∈_ B2 := **begin** use [hB2.constr hB1, hB2.is ~~l~~ inear hB1, hB2.bijective hB1], intros v hv, apply hB2.mem ~~s~~ pan, rw hB1.repr ~~s~~ elf hv, exact _λ_ ~~,~~ hB2.smul ~~,~~ **end** 

29 

Published as a conference paper at ICLR 2024 

Table 22: A successful case generated by MUSTARD. 

**Question Type: Word Problem. Educational Level: Higher Education. k=1. Successful case. Concept(s):** _Series_ in Integral Calculus. **Comments:** MUSTARD is proficient in number series and is able to perform detailed proof by integration. **Informal Statement** Prove that the integral of the series[�] (from _n_ = 1 to infinity) of _[−]_[1][(] _n[n]_[+1)] equals ln(2). **Informal Proof** We start by recognizing the series[�] (from _n_ = 1 to infinity) of _[−]_[1][(] _n[n]_[+1)] as the alternating harmonic series, which is known to converge to ln(2). 

## **Formal Statement and Proof** 

**import** data.real.basic **import** analysis.special ~~f~~ unctions.pow **open** real **lemma** integral ~~o~~ f ~~s~~ eries : � x **in** 0..1, (ln (1 + x)) =[�] _[′]_ (n : N), ((-1)ˆ(n+1) / (n + 1)) := **begin have** h ~~d~~ er : _∀_ x _∈_ set.Icc (0 : R) 1, deriv ( _λ_ x, (ln (1 + x))) x = 1 / (1 + x), _{_ intros x hx, **have** h ~~p~~ os : 0 < 1 + x := **by** nlinarith, exact deriv ~~l~~ n ~~o~~ ne ~~a~~ dd h ~~p~~ os _}_ , **have** h ~~i~~ nt : interval ~~i~~ ntegral ( _λ_ x, 1 / (1 + x)) 0 1 volume =[�] _[′]_ (n : N), ((-1)ˆ(n+1) / (n + 1)), _{_ **have** h ~~f~~ rac : _∀_ (n : N), � x **in** 0..1, xˆn = 1 / (n + 1), _{_ intro n, **calc** � x **in** 0..1, xˆn = [xˆ(n+1) / (n+1)] | 0..1 : integral ~~p~~ ow n ... = 1 / (n + 1) : **by** _{_ rw integral ~~i~~ nterval, simp _} }_ , rw [interval ~~i~~ ntegral.integral ~~o~~ f ~~l~~ e, h ~~f~~ rac], simp, linarith _}_ , **have** h ~~e~~ q : � x **in** 0..1, (ln (1 + x)) = � x **in** 0..1, 1 / (1 + x), _{_ congr, ext, exact h ~~d~~ er x (set.mem ~~I~~ cc.mpr `〈` **by** nlinarith, **by** nlinarith `〉` ) _}_ , rw [h ~~e~~ q, h ~~i~~ nt], **end** 

30 

Published as a conference paper at ICLR 2024 

Table 23: A failed case generated by MUSTARD. 

|**Question Type: Theorem Proving.**<br>**Educational Level: Higher Education.**|**Question Type: Theorem Proving.**<br>**Educational Level: Higher Education.**|**k=2.**|**Failed case.**|
|---|---|---|---|
|**Concept(s):**|_Second order linear equations_ in Differential equations; _Limits _|_and continuity_ in AP College||
|Calculus BC.||||
|**Comments:**|MUSTARDdoes not understand second-order linear equations and|continuity.||



## **Informal Statement** 

Given a second order linear differential equation of the form _y[′′]_ + _p_ ( _x_ ) _y[′]_ + _q_ ( _x_ ) _y_ = 0, where _p_ ( _x_ ) and _q_ ( _x_ ) are continuous on the interval [ _a, b_ ], prove that if there exist two solutions _y_ 1 and _y_ 2 that are linearly independent, then any solution y of the differential equation can be expressed as _y_ = _c_ 1 _∗ y_ 1 + _c_ 2 _∗ y_ 2, where _c_ 1 and _c_ 2 are constants. 

## **Informal Proof** 

The general solution to a homogeneous second order linear differential equation is given by the superposition of two linearly independent solutions. This is a consequence of the linearity of the differential operator. 

Suppose we have a second order linear differential equation, and _y_ 1 and _y_ 2 are two solutions that are linearly independent. Then, any solution _y_ of the differential equation can be expressed as a linear combination of _y_ 1 and _y_ 2. This means there exist constants _c_ 1 and _c_ 2 such that _y_ = _c_ 1 _∗ y_ 1 + _c_ 2 _∗ y_ 2. 

The reason is that any linear combination of solutions to a linear differential equation is also a solution to the differential equation. This follows from the properties of the derivative, which allow us to distribute the derivative over addition and to factor out constants. 

## **Formal Statement and Proof** 

**import** data.real.basic **variables** _α_ : **Type** * [field _α_ ] _/-- Placeholder for the differential equation -/_ **def** diff ~~e~~ q (p q y : _α → α_ ) (x : _α_ ) : Prop := y x = 0 _-- This is just a placeholder. A real differential equation would have a more complex form. /-- Theorem: Given a second order linear differential equation of the form y’’ + p(x)y’ + q(x)y = 0, where p(x) and q(x) are continuous on the interval [a, b], if there exist two solutions y1 and y2 that are linearly independent, then any solution y of the differential equation can be expressed as y = c1*y1 + c2*y2, where c1 and c2 are constants. -/_ **lemma** second ~~o~~ rder ~~l~~ inear ~~d~~ ifferential ~~e~~ q (p q y1 y2 y : _α → α_ ) (c1 c2 : _α_ ) : ( _∃_ x, diff ~~e~~ q p q y1 x _∧_ diff ~~e~~ q p q y2 x) _→_ ( _∀_ x, diff ~~e~~ q p q y x) _→_ ( _∀_ x, y x = c1 * y1 x + c2 * y2 x) := **begin** intros h ~~s~~ ol h ~~y~~ x, _-- This is a placeholder. A real proof would show that y x can be written as a linear combination -- of y1 x and y2 x using the properties of the differential equation, the assumption that y1 and y2 -- are solutions, and the linearity of the differential operator._ **sorry end** 

31 

Published as a conference paper at ICLR 2024 

Table 24: A successful case generated by MUSTARD. 

|**Question Type: Word Problem.**|**Educational Level: Higher Education.**|**k=2.**|**Successful case.**|
|---|---|---|---|
|**Concept(s):** _Matrix transformations_in Linearalgebra;_Collecting data_in AP||College|Statistics.|
|**Comments:** MUSTARDperforms|correct matrix transformation.|||



## **Informal Statement** 

Consider two matrices _A_ = [1 _,_ 2; 3 _,_ 4] and _B_ = [5 _,_ 6; 7 _,_ 8] which represent two different data collection methods applied to a survey of a population. We want to find the collective effect of these two data collection methods when applied simultaneously. In other words, find the matrix _C_ which is the result of the matrix transformation of _A_ by _B_ . 

## **Informal Proof** 

Matrix multiplication is a form of transformation of one matrix by another. In this case, we want to find the matrix _C_ that is a result of the transformation of _A_ by _B_ . 

Matrix multiplication is done by taking the dot product of the rows of the first matrix with the columns of the second matrix. 

Therefore, _C_ [0 _,_ 0] = ( _A_ [0 _,_ 0] _∗ B_ [0 _,_ 0]) + ( _A_ [0 _,_ 1] _∗ B_ [1 _,_ 0]) = (1 _∗_ 5) + (2 _∗_ 7) = 19, _C_ [0 _,_ 1] = ( _A_ [0 _,_ 0] _∗ B_ [0 _,_ 1]) + ( _A_ [0 _,_ 1] _∗ B_ [1 _,_ 1]) = (1 _∗_ 6) + (2 _∗_ 8) = 22, _C_ [1 _,_ 0] = ( _A_ [1 _,_ 0] _∗ B_ [0 _,_ 0]) + ( _A_ [1 _,_ 1] _∗ B_ [1 _,_ 0]) = (3 _∗_ 5) + (4 _∗_ 7) = 43, and _C_ [1 _,_ 1] = ( _A_ [1 _,_ 0] _∗ B_ [0 _,_ 1]) + ( _A_ [1 _,_ 1] _∗ B_ [1 _,_ 1]) = (3 _∗_ 6) + (4 _∗_ 8) = 50. So, the matrix _C_ = [19 _,_ 22; 43 _,_ 50]. 

## **Formal Statement and Proof** 

**import** data.matrix.basic **def** A : matrix (fin 2) (fin 2) N := ![![1, 2], ![3, 4]] **def** B : matrix (fin 2) (fin 2) N := ![![5, 6], ![7, 8]] **def** C : matrix (fin 2) (fin 2) N := A.mul B **#eval** C _-- Output: ![![19, 22], ![43, 50]]_ 

32 

Published as a conference paper at ICLR 2024 

Table 25: A failed case generated by MUSTARD. 

**Question Type: Theorem Proving. Educational Level: Higher Education. k=2. Failed case.** 

**Concept(s):** _Green’s, Stokes’, and the divergence theorems_ in Multivariable calculus; _Differential equations_ in Integral Calculus. **Comments:** MUSTARD does not finish the formal proof with sorry. 

## **Informal Statement** 

Let _F_ be a continuously differentiable vector field on R[3] . Let _S_ be the boundary of the unit ball _B_ in R _[e]_ , oriented by the outward normal. Let _n_ denote the outward unit normal on _S_ . Prove that � _S_ div _F_ d _S_ = � _SF_ • _n_ d _S,_ where “div” denote the divergence of _F_ , “•” denotes the dot product, and “� ”, “� ” denote the surface integral and line integral respectively. **Informal Proof** This is an application of the divergence theorem, also known as Gauss’s theorem. The divergence theorem states that the surface integral of a vector field over a surface _S_ is equal to the volume integral of the divergence of the field over the volume _V_ enclosed by _S_ . In mathematical terms, � _SF_ • _n_ d _S_ = � _V_ div _F_ d _V_ . 

In our case, the volume _V_ is the unit ball _B_ in R[3] , and the surface _S_ is the boundary of _B_ . The divergence theorem thus gives us � _SF_ • _n_ d _S_ = � _B_ div _F_ d _V_ . 

However, since _B_ is the unit ball, its volume is 1, so the volume integral over _B_ is just the average value of div _F_ over _B_ . Therefore, � _B_ div _F_ d _V_ = � _S_ div _F_ d _S_ . 

Putting everything together, we obtain � _SF_ • _n_ d _S_ = � _S_ div _F_ d _S_ , as required. 

## **Formal Statement and Proof** 

**import** analysis.calculus.fderiv **import** measure ~~t~~ heory.integration **import** analysis.normed ~~s~~ pace.inner ~~p~~ roduct **import** data.real.basic **import** topology.instances.real **import** analysis.special ~~f~~ unctions.trigonometric 

open ~~l~~ ocale classical 

**noncomputable theory** 

_/- Let’s define_ R[3] _as Euclidean space -/_ **def** R[3] := euclidean ~~s~~ pace R (fin 3) _/- Define the ball and boundary -/_ **def** B := metric.ball (0:R[3] ) 1 **def** S := metric.sphere (0:R[3] ) 1 _/- Assume F is a continuously differentiable vector field on_ R[3] _-/_ **variables** (F : R[3] _→_ R[3] ) (hF : _∀_ x, differentiable ~~a~~ t R F x) _/- Define the divergence of F -/_ **def** divF := _λ_ x, (fderiv R F x).trace _/- Define the outward unit normal on S -/_ **def** n := _λ_ x:R[3] , x _/- Now we can state the theorem -/_ **def** div ~~e~~ q ~~s~~ urface ~~i~~ ntegral : � x in B, divF F x = � x in S, inner (F x) (n x) := **begin** _-- This is where the main proof would go, using the divergence theorem and other facts about integration._ **sorry end** 

33 

Published as a conference paper at ICLR 2024 

Table 26: A failed case generated by MUSTARD. 

|**Question Type: Theorem Proving.**<br>**Educational Level: Higher Education.**<br>**k=2.**|**Failed case.**|
|---|---|
|**Concept(s):** _Matrix transformations_in Linear algebra.||
|**Comments:** The generated informal statement has factual errors.||
|**Informal Statement**||
|Let’s consider a function_f_ : R _→_Rthat is differentiable at a point_x_0. Let’s also consider a2_×_2matrix_A_.||
|We defne a new function_g_(_x_) = _A ∗f_(_x_). Prove that_g_is also differentiable at_x_0, and|fnd the derivative of|
|_g_at_x_0. **Informal Proof**||
|Since _f_ is differentiable at _x_0, the limit of the difference quotient exists at this point.|Denote this limit as|
|_f ′_(_x_0).||
|We know that matrix multiplication is a linear operation. So we can move the limit inside the matrix multipli-||
|cation to fnd the derivative of_g_at_x_0.||



So, _g[′]_ ( _x_ 0) = _A ∗ f[′]_ ( _x_ 0). 

## **Formal Statement and Proof** 

**import** data.matrix.basic **import** analysis.calculus.deriv **variables** f : R _→_ R A : matrix (fin 2) (fin 2) R _x_ 0 : R **variable** (h : differentiable ~~a~~ t R f _x_ 0) **include** h **lemma** matrix ~~t~~ imes ~~d~~ ifferentiable ~~f~~ unction : differentiable ~~a~~ t R ( _λ_ x, A _·_ ( _λ_ i, f x) i) _x_ 0 := **begin** apply matrix.has ~~f~~ deriv ~~a~~ t diff ~~a~~ t, exact h.has ~~d~~ eriv ~~a~~ t, **end lemma** derivative ~~o~~ f ~~m~~ atrix ~~t~~ imes ~~f~~ unction : deriv ( _λ_ x, A _·_ ( _λ_ i, f x) i) _x_ 0 = A _·_ ( _λ_ i, deriv f _x_ 0) i := **begin** apply has ~~d~~ eriv ~~a~~ t.deriv, exact matrix ~~t~~ imes ~~d~~ ifferentiable ~~f~~ unction h, **end** 

34 

Published as a conference paper at ICLR 2024 

Table 27: Prompt template for math word problem inference. 

**Zero shot prompt template for math word problem inference** 

You are an expert in math. Answer the following math word problem. Question: <question> Answer: Let’s think step by step. 

**Few shot prompt template for math word problem inference** 

You are an expert in math. Answer the following math word problem. 

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left? Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. Question: <question> Answer: 

35 

Published as a conference paper at ICLR 2024 

Table 28: Compared performances on GSM8K (G) and MATH (M) between Llama 2-7B and Llama 2-70B. 

|MODEL||Zero (G)||Few (G)|Zero (M)||Few (M)|
|---|---|---|---|---|---|---|---|
|_Baselines_||||||||
|Llama 2-7B||7.2||12.8|2.0||2.6|
|Llama 2-70B||31.7||54.1|8.8||13.4|
|_Fine-tuning_||||||||
|Llama 2-7B_>_tt||9.6||16.0|3.2||3.8|
|Llama 2-7B_>_in||9.1||14.9|2.4||3.2|
|Llama 2-7B_>_ra||9.5||15.4|3.0||3.6|
|Llama 2-7B_>_va|10.3 (+8.42%)||16.9 (+9.74%)||3.2 (+6.67%)|4.2 (+16.67%)||
|Llama 2-70B_>_tt||36.6||55.8|10.0||14.4|
|Llama 2-70B_>_in||33.4||53.7|9.2||13.6|
|Llama 2-70B_>_ra||36.1||55.4|9.6||14.2|
|Llama 2-70B_>_va|39.5 (+9.42%)||59.1 (+6.68%)||10.4 (+8.33%)|15.0 (+5.30%)||



Table 29: Nearest data points between MUSTARDSAUCE and miniF2F. 

## **MUSTARDSAUCE v.s. miniF2F (cosine similarity = 0.6439)** 

## **MUSTARDSAUCE Case** 

## **Informal Statement:** 

Alex has 5 ten-dollar bills and 3 one-dollar bills. How much money does Alex have in total? 

## **Informal Proof:** 

To find out how much money Alex has in total, we need to multiply the number of each type of bill by its value. So, Alex has 5 ten-dollar bills, which equals 5 * 10 = 50 dollars. He also has 3 one-dollar bills, which equals 3 * 1 = 3 dollars. Adding these two amounts together gives 50 + 3 = 53 dollars. Therefore, Alex has 53 dollars in total. 

## **Formal Statement and Proof:** 

**def** calculate ~~m~~ oney (tens : N) (ones : N) : N := tens * 10 + ones * 1 **example** : calculate ~~m~~ oney 5 3 = 53 := 

**begin** 

rw calculate ~~m~~ oney, 

refl, 

## **end** 

## **miniF2F Case** 

**theorem** algebra ~~s~~ qineq ~~u~~ nitcircatbpamblt1 

(a b: R) (h0 : a[2] + b[2] = 1) : a * b + (a - b) _le_ 1 := **begin** 

nlinarith [sq ~~n~~ onneg (a - b)], 

**end** 

36 

Published as a conference paper at ICLR 2024 

Table 30: Nearest data points between MUSTARDSAUCE and mathlib. 

**MUSTARDSAUCE v.s. mathlib (cosine similarity = -0.0361)** 

**MUSTARDSAUCE Case** 

## **Informal Statement:** 

A cube has a side length of 5 cm. What is the volume of the cube? 

## **Informal Proof:** 

The volume of a cube is calculated by raising the side length to the power of 3. So in this case, the volume is 5 cm * 5 cm * 5 cm = 125 cubic centimeters. 

## **Formal Statement and Proof:** 

**def** cube ~~v~~ olume (side ~~l~~ ength : N) : N := side ~~l~~ ength * side ~~l~~ ength * side ~~l~~ ength 

**#eval** cube ~~v~~ olume 5 -- returns 125 

## **mathlib Case** 

**GOAL** _α_ : Type u, _β_ : Type v , _γ_ : Type w, f : _α → β → γ_ , l1 : list _α ⊢∀_ (l2 : list _β_ ), (list.map2 f l1 l2).length = linear ~~o~~ rder.min l1.length l2.length **PROOFSTEP** induction l1; intro l2; cases l2; simp [*, add ~~o~~ ne, min ~~s~~ ucc ~~s~~ ucc, nat.zero ~~m~~ in, nat.min ~~z~~ ero] 

37 

Published as a conference paper at ICLR 2024 

## Table 31: Nearest data points between MUSTARDSAUCE and GSM8K. 

## **MUSTARDSAUCE v.s. GSM8K (cosine similarity = 0.5975)** 

## **MUSTARDSAUCE Case** 

## **Informal Statement:** 

Given two vectors u = (u1,u2) and v = (v1,v2) in R[2] , prove that the operation of vector addition, defined as u + v = (u1+v1, u2+v2), is commutative. That is, prove that for all u and v in R[2] , u + v = v + u. 

## **Informal Proof:** 

We can prove this by direct computation. Given the vectors u = (u1, u2) and v = (v1, v2), we have: 

u + v = (u1 + v1, u2 + v2) 

and 

v + u = (v1 + u1, v2 + u2). 

Since addition is commutative in R (the set of all real numbers), we have u1 + v1 = v1 + u1 and u2 + v2 = v2 + u2. Thus, we conclude that (u1 + v1, u2 + v2) = (v1 + u1, v2 + u2), which means that u + v = v + u. 

## **Formal Statement and Proof:** 

**import** data.real.basic **variables** (u1 u2 v1 v2 : R) 

-- defining vectors as pair of real numbers 

**def** vector := R × R -- defining vector addition **def** vadd (u v : vector) : vector := (u.1 + v.1, u.2 + v.2) -- defining vectors u and v **def** u : vector := (u1, u2) **def** v : vector := (v1, v2) -- commutativity of vector addition **theorem** vadd ~~c~~ omm : vadd (u u1 u2) (v v1 v2) = vadd (v v1 v2) (u u1 u2) := 

**begin** 

unfold vadd, unfold u, unfold v, rw add ~~c~~ omm u1 v1, rw add ~~c~~ omm u2 v2, 

## **end** 

## **GSM8K Case** 

**Question:** The local firefighters are doing a ”fill the boot” fundraiser. Their goal is to raise $6300. After the first 3 hours, they have raised $2100. For how many hours do they have to fundraise in total to reach their goal, assuming an equal amount raised in every hour? 

## **Answer:** 

The fireman raise 2100 / 3 = $ _<<_ 2100/3=700 _>>_ 700 per hour. 

They have to fundraise for 6300 / 700 = _<<_ 6300/700=9 _>>_ 9 hours. 

#### 9 

38 

Published as a conference paper at ICLR 2024 

## Table 32: Nearest data points between MUSTARDSAUCE and MATH. 

## **MUSTARDSAUCE v.s. MATH (cosine similarity = 0.6064)** 

## **MUSTARDSAUCE Case** 

## **Informal Statement:** 

Given two vectors u = (u1,u2) and v = (v1,v2) in R[2] , prove that the operation of vector addition, defined as u + v = (u1+v1, u2+v2), is commutative. That is, prove that for all u and v in R[2] , u + v = v + u. 

## **Informal Proof:** 

We can prove this by direct computation. Given the vectors u = (u1, u2) and v = (v1, v2), we have: 

u + v = (u1 + v1, u2 + v2) 

and 

v + u = (v1 + u1, v2 + u2). 

Since addition is commutative in R (the set of all real numbers), we have u1 + v1 = v1 + u1 and u2 + v2 = v2 + u2. Thus, we conclude that (u1 + v1, u2 + v2) = (v1 + u1, v2 + u2), which means that u + v = v + u. 

## **Formal Statement and Proof:** 

**import** data.real.basic **variables** (u1 u2 v1 v2 : R) -- defining vectors as pair of real numbers **def** vector := R × R -- defining vector addition **def** vadd (u v : vector) : vector := (u.1 + v.1, u.2 + v.2) -- defining vectors u and v **def** u : vector := (u1, u2) **def** v : vector := (v1, v2) -- commutativity of vector addition **theorem** vadd ~~c~~ omm : vadd (u u1 u2) (v v1 v2) = vadd (v v1 v2) (u u1 u2) := **begin** unfold vadd, unfold u, unfold v, rw add ~~c~~ omm u1 v1, rw add ~~c~~ omm u2 v2, **end** 

## **MATH Case** 

**Problem:** If a snack-size tin of peaches has 40 calories and is 2% of a person’s daily caloric requirement, how many calories fulfill a person’s daily caloric requirement? 

**Solution:** If 40 calories is equal to 2% = 1002[=] 501[of][a][person’s][daily][requirement,][then][a][person’s][daily] caloric requirement is: $4 _·_ 50 = 2000 

**Answer** : 2000 

39 

