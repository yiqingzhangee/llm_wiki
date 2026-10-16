# **Fact-Checking Complex Claims with Program-Guided Reasoning** 

**Liangming Pan**[1] _[,]_[2] **Xiaobao Wu**[3] **Xinyuan Lu**[4] **Anh Tuan Luu**[3] **William Yang Wang**[1] **Min-Yen Kan**[4] **Preslav Nakov**[2] 

1 University of California, Santa Barbara 

2 Mohamed bin Zayed University of Artificial Intelligence 

3 Nanyang Technological University 4 National University of Singapore liangmingpan@ucsb.edu xiaobao002@e.ntu.edu.sg luxinyuan@u.nus.edu anhtuan.luu@ntu.edu.sg william@cs.ucsb.edu kanmy@comp.nus.edu.sg preslav.nakov@mbzuai.ac.ae 

## **Abstract** 

Fact-checking real-world claims often requires collecting multiple pieces of evidence and applying complex multi-step reasoning. In this paper, we present _Program-Guided Fact-Checking_ (PROGRAMFC), a novel factchecking model that decomposes complex claims into simpler sub-tasks that can be solved using a shared library of specialized functions. We first leverage the in-context learning ability of large language models to generate _reasoning programs_ to guide the verification process. Afterward, we _execute_ the program by delegating each sub-task to the corresponding sub-task handler. This process makes our model both explanatory and dataefficient, providing clear explanations of its reasoning process and requiring minimal training data. We evaluate PROGRAMFC on two challenging fact-checking datasets and show that it outperforms seven fact-checking baselines across different settings of evidence availability, with explicit output programs that benefit human debugging.[1] 

## **1 Introduction** 

The proliferation of disinformation, _e.g._ , in social media, has made _automated fact-checking_ a crucial application of natural language processing (NLP). Given a _claim_ , the goal is to find _evidence_ and then to make a _verdict_ about the claim’s veracity based on that evidence (Thorne and Vlachos, 2018; Glockner et al., 2022; Guo et al., 2022). 

Evaluating the veracity of real-world claims often involves collecting multiple pieces of evidence and applying complex reasoning (Jiang et al., 2020; Nguyen et al., 2020; Aly and Vlachos, 2022; Chen et al., 2022a). For instance, consider the claim “ _Both James Cameron and the director of the film Interstellar were born in Canada_ ”. It may be challenging to find direct evidence on the web that 

1The program code and the data are publicly available at https://github.com/mbzuai-nlp/ProgramFC 

refutes or supports this claim. Instead, a human fact-checker needs to decompose the claim, gather multiple pieces of evidence, and perform step-bystep reasoning (Nakov et al., 2021a), as illustrated in Figure 1. This makes verifying complex claims much more challenging than the typical setting explored in previous work, where information from a single article is sufficient to support/refute the claim (Thorne et al., 2018; Saakyan et al., 2021; Schuster et al., 2021; Pan et al., 2021; Wadden et al., 2022a; Krishna et al., 2022). 

Besides multi-step reasoning, we still need to consider two key aspects for developing a reliable fact-checking system: ( _i_ ) _Explanability_ : The model should not only predict the veracity of the claim, but it should also provide a clear explanation of its reasoning process to help users understand and trust the results. ( _ii_ ) _Data efficiency_ : Human annotation is often time-consuming, costly, and potentially biased, making it difficult to collect sufficient highquality labeled data for model training, particularly for complex claims. Therefore, it is desirable to build a model that can perform well with minimal or no training data. Despite a few models (Zhou et al., 2019; Zhong et al., 2020; Aly and Vlachos, 2022) being proposed to facilitate multi-step reasoning in fact-checking, they either lack explainability in their reasoning process or require a large number of task-specific training examples. 

In this paper, we present _Program-Guided FactChecking_ (PROGRAMFC), a novel fact-checking framework that is both explanatory and dataefficient. Figure 1 illustrates our approach. To verify complex claims, PROGRAMFC decomposes them into simpler sub-tasks that can be solved using a shared library of specialized sub-task functions. To be specific, PROGRAMFC begins by generating a _reasoning program_ for the input claim, which is a sequence of sub-tasks ( _e.g._ , S1-S4 in Figure 1) in the form of ACTION[ARGUMENT], where ACTION and ARGUMENT define the type and the 

**Claim:** Both James Cameron and the director of the film Interstellar were born in Canada. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning_images/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Knowledge<br>Reasoning Program Functions Source<br>Language<br>Models S1 Verify [James Cameron was born in Canada.] Fact<br>(Codex, GPT3, …) FACT_1 = TRUE Checker<br>Gold Evidence<br>S2 Question  [Who is the director of the film Interstellar?] QA<br>ANSWER_1 =  Christopher Nolan Model<br>S3 Verify [  {ANSWER_1}  was born in Canada.] Fact  Open-book<br>Claim: ⋯<br>Program: Claim: ⋯⋯ FACT_2 = FALSE Checker<br>Program: Claim: ⋯⋯<br>Program: ⋯ S4 Predict [  {FACT_1} AND {FACT_2}] Logical  Closed-book<br>Reasoner<br>PREDICTED_LABEL = REFUTES<br>Exemplars<br>**----- End of picture text -----**<br>


Figure 1: Overview of our PROGRAMFC model, which consists of two modules: ( _i_ ) _Program Generation_ generates a reasoning program for the input claim using Codex with in-context learning, and then ( _ii_ ) _Program Execution_ sequentially interprets the program by delegating each step to the corresponding sub-task function. 

content of the sub-task, respectively. The generated reasoning program serves as a step-by-step guide for verifying the claim. We then _execute_ the program by sequentially delegating each sub-task to the corresponding sub-task handler, as shown in the _functions_ columns in Figure 1. These sub-tasks may include answering questions, verifying simple claims, or conducting logical reasoning. 

PROGRAMFC combines explainability with data efficiency. It uses reasoning programs to provide clear explanations of its reasoning process. For data efficiency, Large Language Models (LLMs) can solve various tasks given only a few examples as prompts, _e.g._ , _in-context learning_ (Brown et al., 2020). We leverage this ability of LLMs to generate reasoning programs for a given claim by showing the model just a few dozen of (claim, program) pairs as demonstrations. PROGRAMFC is also flexible as it allows for easy swapping of subtask function implementations to work under different settings of fact-checking, without affecting the rest of the system. We can allow the functions to retrieve information from external sources (in an open-book setting) or we can ask them to generate answers based solely on the LLM’s internal parametric knowledge (in a closed-book setting). 

We evaluate PROGRAMFC on two challenging datasets designed for fact-checking complex claims: HOVER (Jiang et al., 2020) and FEVEROUS (Aly et al., 2021), and we show that it outperforms seven few-shot fact-checking baselines on both datasets (§ 4.1). The strategy of program- 

guided reasoning becomes increasingly effective as the required reasoning depth increases (§ 4.1). In the open-domain setting, we find that reasoning programs can enhance the retrieval of relevant evidence from knowledge sources (§ 4.2). PROGRAMFC is robust even when we use weak models as sub-task solvers (§ 4.2). We also evaluate the interpretability of the reasoning programs through human evaluation and error analysis (§ 4.3). 

## **2 Related Work** 

**Fact-Checking.** Automated fact-checking has gained significant attention in the NLP research community in recent years as a means of combating misinformation and disinformation. Various datasets have been proposed that enable the development and the evaluation of systems for automatic fact-checking, the most popular ones being based on human-crafted claims from Wikipedia content (Thorne et al., 2018; Sathe et al., 2020; Schuster et al., 2021) and naturally occurring claims in the political or in the scientific domain (Wang, 2017; Nakov et al., 2021b; Augenstein et al., 2019; Saakyan et al., 2021; Gupta and Srikumar, 2021; Wadden et al., 2020, 2022a). Notably, most of these datasets are constructed in a way that the evidence to support or to refute a claim can be found in a _single_ document. For example, in FEVER (Thorne et al., 2018), more than 87% of the claims only require information from a single Wikipedia article (Jiang et al., 2020). 

To bridge this gap, datasets have been proposed 

to study fact-checking complex claims that require multi-step reasoning (Jiang et al., 2020; Aly et al., 2021). Graph-based models (Zhou et al., 2019; Liu et al., 2020; Zhong et al., 2020; Nguyen et al., 2020; Barnabò et al., 2022, 2023) are used to facilitate the reasoning over multiple pieces of evidence. Although such models achieve sizable performance gains, they lack explanability and thet rely on large amounts of training data. To address the above problems, we propose an explainable, flexible, and data-efficient model that generates reasoning graphs as explanations and utilizes incontext learning to enable few-shot learning. 

**Explanation Generation.** Facing the complexities of real-world claims, simply giving a final veracity to a claim often fails to be persuasive (Guo et al., 2022). Previous research has proposed various approaches to provide post-hoc explanations for model predictions, such as using attention weights to highlight relevant parts of the evidence (Popat et al., 2017; Cui et al., 2019; Yang et al., 2019; Lu and Li, 2020), generating justifications with logic-based systems based on knowledge graphs (Gad-Elrab et al., 2019; Ahmadi et al., 2019), and generating a summary of the retrieved relevant evidence (Atanasova et al., 2020; Kotonya and Toni, 2020; Jolly et al., 2022). In contrast, we propose to use reasoning programs to provide explanations that consist of sub-tasks described in a program-like natural language. This offers several advantages: it allows for explanations that are not confined to the evidence, like attention weights, it is more flexible than logic-based explanations, and it is more concise than free-form summarization. 

**Chain-of-Thought Reasoning.** Moreover, unlike previous work that generates post-hoc explanations, we also use reasoning programs as guidance for predicting the veracity of the claim. This is motivated by the recent success of chain-of-thought prompting (CoT) (Wei et al., 2022; Kojima et al., 2022; Wang et al., 2022), which generates step-bystep natural language reasoning steps to guide the model in answering complex questions. We adopt this idea to fact-checking complex claims. Unlike the original CoT, which uses a single LLM for both decomposition and question answering, we use the language model only to generate reasoning programs as the blueprint for problem-solving, and we delegate each sub-task to specialized functions. 

This approach reduces the burden on the lan- 

guage model and allows for more flexibility in incorporating necessary components for factchecking such as an evidence retriever. The strategy of program-guided reasoning is also in line with the recent trend of tool-augmented language models (Mialon et al., 2023; Schick et al., 2023), _i.e._ , augmenting language models with access to external tools and resources. 

## **3 PROGRAMFC** 

We first formulate the problem of fact-checking and then we introduce our proposed model for _ProgramGuided Fact-Checking_ (PROGRAMFC). 

## **3.1 Problem Formulation** 

Given a _claim C_ , a _fact-checking model F_ aims to predict a _label Y_ to evaluate the claim as TRUE or FALSE, based on a _knowledge source K_ . The model is also required to output an _explanation E_ to justify the predicted veracity label. We summarize three different settings of fact-checking depending on the type of knowledge source _K_ . 

_•_ **Gold evidence** : For each claim, _K_ is the set of gold evidence documents that can support or refute the claim. This setting is also called _claim verification_ (Pan et al., 2021; Wright et al., 2022). _•_ **Open-book setting** : _K_ is a large textual corpus such as Wikipedia. The model first retrieves relevant _evidence_ from the corpus and then predicts the veracity label based on the evidence (Jiang et al., 2021; Wadden et al., 2022b). 

_•_ **Closed-book setting** : The model does not have access to any external knowledge source ( _K_ = _∅_ ). It needs to leverage the knowledge stored in its parameters (acquired during pre-training and finetuning) to verify the claim. This setting was explored in work that applies large language models for fact-checking (Lee et al., 2020, 2021). 

## **3.2 Program-Guided Reasoning** 

Our goal is to fact-check a complex claim _C_ that requires multi-step reasoning. We focus on the _fewshot_ setting, where only a small set of in-domain examples are available to teach the model. To solve this, PROGRAMFC follows a _program generationand-execution_ paradigm, as shown in Figure 1. 

**Program Generation.** At this stage, given the input claim _C_ , a _planner P_ generates a _reasoning program P_ = [ _S_ 1 _, · · · , Sn_ ] for it, which consists of _n_ sequentially ordered _reasoning steps Si_ . 

Each _reasoning step Si ∈ P_ is an instruction in controlled natural language that directs _Si_ to a function in an auxiliary set of sub-task functions _F_ available to the system. To be specific, we define _Si_ = ( _fi, Ai, Vi_ ), where _fi_ specifies the sub-task function _fi ∈F_ , _Ai_ is the _argument_ passed to the function _fi_ , and _Vi_ is the _variable_ that stores the returned result from the function call _fi_ ( _Ai_ ). For a valid reasoning program, the return value of the last reasoning step must be a Boolean value indicating the veracity label of the claim _C_ , _i.e._ , _Vn ∈{_ TRUE _,_ FALSE _}_ . 

**Program Execution.** In the execution stage, the reasoning program _P_ is run by an _interpreter_ to derive the veracity label of the claim _C_ . The interpreter sequentially parses the reasoning steps in _P_ . For each step _Si_ = ( _fi, Ai, Vi_ ), it calls the corresponding off-the-shelf _sub-task function fi_ and passes the argument _Ai_ to it. The argument _Ai_ is either a logical expression or a natural language sentence, _e.g._ , a question or a simple claim. The result of the function call is then stored in the variable _Vi_ . As it is common for a subsequent step to depend on the results from previous steps, we allow the argument _Ai_ to _refer to_ variables _V_ 1 _, · · · , Vi−_ 1 in previous steps. For example, in Figure 1, the argument in _S_ 3 is “ _{ANSWER_1} was born in Canada._ ”, which refers to the return variable _{ANSWER_1}_ from _S_ 2. When executing _S_ 3, the variable is replaced by its actual value, and the argument becomes “ _Christopher Nolan was born in Canada_ ”. After executing the last step, the return value is the predicted veracity of the claim _C_ . 

**Aggregating Reasoning Paths.** Note that there might be multiple reasoning paths that can reach the final veracity label. Therefore, we generate a diverse set of _N_ candidate reasoning programs _P_ = _{P_ 1 _, · · · , PN }_ for the input claim. After executing all programs in _P_ , we take the majority vote over all _N_ predicted labels as the final label. This approach is similar to how humans rely on multiple methods of validation to increase their confidence in fact-checking. It also makes the model less susceptible to errors in individual reasoning programs. 

## **3.3 Reasoning Program Generation** 

We base our program generator on _Codex_ (Chen et al., 2021), a code-pretrained LLM, which can parse natural language into symbolic representations such as SQL (Cheng et al., 2022) or Python programs (Gao et al., 2022; Chen et al., 2022b). 

However, the grammar of a reasoning program is different from the grammar of a programming language. We take advantage of Codex’s few-shot generalization ability and we find that it can learn effectively from only a small number of in-context examples _D_ = _{d_ 1 _, · · · , d|D|}_ . Each example _di_ consists of a claim and a program. The program has a Python-like grammar, where each reasoning step is written in the format _Vi_ = _fi_ ( _Ai_ ). At inference time, we prompt Codex with an instruction of the task, _K_ in-context examples, and the input claim _C_ . Codex then attempts to complete the following texts, and thereby generates a program for _C_ . The prompt template is shown in Figure 2. We use _K_ = 20 to maintain a tradeoff between the diversity of reasoning types and the model’s maximum input capacity. We use sampling-based decoding (temperature of 0.7) to generate different reasoning programs for multiple runs. 

## **3.4 Sub-Task Functions** 

We implement three sub-task functions for the model to call during the program execution. 

_•_ **QUESTION:** This sub-task function is a questionanswering module that takes a question _Q_ as the input argument and returns the answer _A_ to the question. We use FLAN-T5 (Chung et al., 2022), an improved T5 model (Raffel et al., 2020) pretrained on more than 1.8K tasks with instruction tuning, which has achieved state-of-the-art zero/few-shot performance on many QA benchmarks. As shown in Figure 3, we prompt the model differently depending on the settings defined in Section 3.1. For the closed-book setting, the input prompt is 

## Q: QUESTION ? The answer is: 

For the other two settings, the input prompt is 

EVIDENCE Q: QUESTION ? The answer is: 

_•_ **VERIFY:** This is a fact verification module that takes a claim _C_ as the input argument and returns a label of either TRUE or FALSE. We also use FLAN-T5 for this module, by prompting the model with the following question-answering format. 

## EVIDENCE 

Q: Is it true that CLAIM ? True or False? The answer is: 

_•_ **PREDICT:** This module takes as input a logical expression that performs AND, OR, NOT operations over the variables in the previous steps. Its output is returned as the predicted veracity label. 

'''Generate a python -like program that describes the reasoning steps required to verify the claim step -by -step. You can call three functions in the program: 1. Question () to answer a question; 2. Verify () to verify a simple claim; 3. Predict () to predict the veracity label.''' 

# The claim is that Both James Cameron and the director of the film Interstellar were born in Canada. def program (): fact_1 = Verify("James Cameron was born in Canada.") Answer_1 = Question("Who is the director of the film Interstellar?") fact_2 = Verify("{Answer_1} was born in Canada.") label = Predict(fact_1 and fact_2) 

**(** _· · ·_ **more in-context examples here** _· · ·_ **)** 

# The claim is that <input_claim> def program (): 

Figure 2: The Codex prompt template used to generate reasoning programs, consisting of a task instruction, incontext examples, and a prompt for the <input_claim>. The full templates are given in Appendix D. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning_images/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning.pdf-0005-05.png)


**----- Start of picture text -----**<br>
<Gold Evidence><br>Q:  <Question><br>Gold Evidence<br>The answer is:<br>Open-book<br><Retrieved Evidence><br><Question> Q:  <Question> FLAN-T5 Ans<br>Retriever The answer is:<br>Closed-book Q:  <Question><br>The answer is:<br>**----- End of picture text -----**<br>


Figure 3: Implementation of the question-answering sub-task function for three different settings. 

## **4 Experiments** 

**Datasets.** Most fact-checking datasets consist primarily of simple claims that can be substantiated through a single piece of evidence. However, here we focus on complex claims that need multi-step reasoning. Given this context, we opt to evaluate our model on the only two datasets that, to the best of our knowledge, fulfill these criteria: HOVER (Jiang et al., 2020) and FEVEROUS (Aly et al., 2021). We use the validation sets for evaluation since the test sets are not publicly released. HOVER contains claims that require integration and reasoning over multiple Wikipedia articles. We divide its validation set into three subsets based on the number of “hops” required to verify the claim: 1,126 two-hop claims, 1,835 three-hop claims, and 1,039 four-hop claims. FEVEROUS focuses on fact-checking complex claims over unstructured and structured data, where each claim is annotated with evidence in the form of sentences and/or cells from tables in Wikipedia. Since we focus on textual fact-checking, we only selected claims that require exclusively sentence evidence, constituting 2,962 claims. We call this subset FEVEROUS-S. 

For evaluation in the open-book setting, we use the corresponding Wikipedia corpus constructed for these two datasets as the knowledge sources. HOVER uses the October 2017 Wikipedia dump processed by Yang et al. (2018), consisting of the introductory sections of 5.2 million Wikipedia pages. FEVEROUS uses the December 2020 dump, including 5.4 million full Wikipedia articles. 

> **Baselines.** We compare PROGRAMFC to seven baselines, categorized into three groups. ( _i_ ) _Pretrained models_ : BERT-FC (Soleimani et al., 2020) and LisT5 (Jiang et al., 2021) are two models that leverage BERT and T5 for fact verification, respectively. ( _ii_ ) _FC/NLI fine-tuned models_ : we choose three pretrained models that are fine-tuned on other fact-checking datasets or natural language inference (NLI) datasets. RoBERTa-NLI (Nie et al., 2020) uses fine-tuned RoBERTa-large on four NLI datasets; DeBERTaV3-NLI (He et al., 2021) finetunes the DeBERTaV3 model on 885,242 (claim, evidence, label) annotations from FEVER and four NLI datasets. MULTIVERS (Wadden et al., 2022b) is a LongFormer (Beltagy et al., 2020) model finetuned on FEVER. ( _iii_ ) _In-context learning models_ : one baseline is that we directly use the FLAN-T5 model in our VERIFY module for fact-checking. The other baseline uses the in-context learning of Codex for few-shot fact-checking. The implementation details are given in Appendix A. 

**Few-Shot Learning.** We study few-shot learning where only a few in-domain examples are available. Therefore, for a fair comparison, we restrict all models to have access to only 20 examples from HOVER or FEVEROUS-S. 

|**Few-shot learning models**|**HOVER**|**(2-hop)**|**HOVER**|**(3-hop)**|**HOVER**|**(4-hop)**|**FEVEROUS-S**|**FEVEROUS-S**|
|---|---|---|---|---|---|---|---|---|
||**Gold**|**Open**|**Gold**|**Open**|**Gold**|**Open**|**Gold**|**Open**|
|I<br>BERT-FC(Soleimani et al.,2020)<br>LisT5(Jiang et al.,2021)|53.40<br>56.15|50.68<br>52.56|50.90<br>53.76|49.86<br>51.89|50.86<br>51.67|48.57<br>50.46|74.71<br>77.88|51.67<br>54.15|
||||||||||
|II<br>RoBERTa-NLI(Nie et al.,2020)<br>DeBERTaV3-NLI(He et al.,2021)<br>MULTIVERS(Wadden et al.,2022b)|74.62<br>**77.22**<br>68.86|63.62<br>68.72<br>60.17|62.23<br>65.98<br>59.87|53.99<br>60.76<br>52.55|57.98<br>60.49<br>55.67|52.40<br>56.00<br>51.86|88.28<br>91.98<br>86.03|57.80<br>58.81<br>56.61|
||||||||||
|III<br>Codex(Chen et al.,2021)<br>FLAN-T5(Chung et al.,2022)|70.63<br>73.69|65.07<br>69.02|66.46<br>65.66|56.63<br>60.23|63.49<br>58.08|57.27<br>55.42|89.77<br>90.81|62.58<br>63.73|
||||||||||
|IV<br>ProgramFC (N=1)<br>ProgramFC (N=5)|74.10<br>75.65|69.36<br>**70.30**|66.13<br>**68.48**|60.63<br>**63.43**|65.69<br>**66.75**|**59.16**<br>57.74|91.77<br>**92.69**|67.80<br>**68.06**|



Table 1: Macro-F1 scores of PROGRAMFC (IV) and baselines (I-III) on the evaluation set of HOVER and FEVEROUS-S for few-shot fact-checking. _Gold_ and _Open_ represent the gold evidence setting and the open book setting, respectively. I: pretrained Transformers; II: FC/NLI fine-tuned models; III: in-context learning models. 

We use these examples either for fine-tuning pre-trained models (BERT-FC and LisT5), for continuous fine-tuning the FC/NLI fine-tuned models, or as in-context examples for FLAN-T5 and Codex. For PROGRAMFC, we use them as in-context examples for reasoning program generation. 

We evaluate both the _gold evidence setting_ and the _open-book setting_ . The baseline models are the same for both settings. However, during testing in the open-book setting, the models are given the retrieved evidence rather than the ground-truth evidence. We use BM25 (Robertson and Zaragoza, 2009) implemented with the Pyserini toolkit (Lin et al., 2021) as the retriever for both PROGRAMFC and the baselines. We use as evidence the top-10 paragraphs retrieved from the knowledge corpus. 

## **4.1 Main Results** 

We report the overall results for PROGRAMFC and for the baselines for few-shot fact-checking in Table 1. PROGRAMFC achieves the best performance on 7 out of 8 evaluations, demonstrating its effectiveness. We have three more specific observations. 

**ProgramFC is more effective on deeper claims.** On the HOVER dataset, ProgramFC (N=5) outperforms the baselines on average by 10.38%, 11.37%, and 14.77% on two-hop, three-hop, and four-hop claims, respectively. This suggests that ProgramFC becomes increasingly effective as the required reasoning depth increases. Among the baselines, DeBERTaV3-NLI performs comparably to ProgramFC on two-hop claims, indicating that large-scale pre-training on simpler claims can help the model generalize to more complex claims. 

However, this generalization becomes more challenging as the complexity of the claims increases. On HOVER, the F1 score of DeBERTaV3-NLI drops from 77.22 for 2-hop claims to 60.49 for 4-hop claims, which is a decrease of 21.7%. In contrast, the performance drop for ProgramFC, which uses the strategy of program-guided reasoning, is much smaller: just 11.7%. 

## **Decomposition is more effective than one-step** 

**prediction.** The ProgramFC model, which uses the same FLAN-T5 model as the sub-task functions, outperforms the baseline of directly verifying claims with FLAN-T5 on all four datasets. On average, there is a 6.0% improvement in the gold evidence setting and a 4.5% improvement in the open-book setting. This suggests that decomposing a complex claim into simpler steps with a program can facilitate more accurate reasoning. This is especially evident when the required reasoning is complex: there is a 14.9% improvement in the gold evidence setting and a 6.7% improvement in the open-book setting for 4-hop claims. 

## **Aggregating reasoning programs is helpful.** 

We find that aggregating the predictions of _N_ = 5 reasoning programs improves the performance over using a single program by an average of 1.5%. This aligns with the findings of Wang et al. (2022), where the idea was applied for question answering: if multiple different ways of thinking lead to the same answer, we can have greater confidence that the final answer is correct. This intuition also applies to fact-checking, as each program represents a unique reasoning chain to verify the claim. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning_images/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning.pdf-0007-00.png)


**----- Start of picture text -----**<br>
FLAN-T5 ProgramFC HOVER (2-hop) FLAN-T5 ProgramFC HOVER (3-hop) FLAN-T5 ProgramFC HOVER (4-hop)<br>80 76.11 75.65 77.62 80 80<br>72.56 77.07<br>7064.35 71.69 73.69 70 67.88 68.55 68.48 69.56 70 68.37 68.56 66.75 68.18<br>68.24 62.23 65.07 65.66 66.89 62.46<br>60 60 63.05 60 61.36 63.39<br>56.58 58.08<br>50 50 50<br>47.75 49.29 48.59<br>40 40 40<br>80M 250M 780M 3B 11B 80M 250M 780M 3B 11B 80M 250M 780M 3B 11B<br>**----- End of picture text -----**<br>


Figure 4: F1 score for fact-checking with gold evidence using FLAN-T5 (blue line) and PROGRAMFC (green line) for language models of increasing sizes: FLAN-T5-small (80M), FLAN-T5-base (250M), FLAN-large (780M), FLAN-T5-XL (3B), and FLAN-T5-XXL (11B) on HOVER 2-hop (left), 3-hop (middle), and 4-hop (right). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning_images/acl_2023_Fact-Checking_Complex_Claims_with_Program-Guided_Reasoning.pdf-0007-02.png)


**----- Start of picture text -----**<br>
90<br>80 One-step Retrieval ProgramFC 85.65<br>70 73.18 77.13 76.25<br>60<br>59.17<br>50 51.33 49.93<br>40<br>30 36.43<br>20<br>HOVER (2-hop) HOVER (3-hop) HOVER (4-hop) FEVEROUS-S<br>**----- End of picture text -----**<br>


Figure 5: Retrieval recall@10 for the one-step retrieval and the iterative retrieval in PROGRAMFC. 

## **4.2 How Does the Reasoning Program Help?** 

To further understand how reasoning programs facilitate fact-checking, we compare the performance of PROGRAMFC with FLAN-T5 using different language model sizes: small, base, large, XL, and XXL. The results are shown in Figure 4 and indicate that program-guided reasoning is particularly effective when the model size is small. As smaller models have less capacity for complex reasoning, the performance of the end-to-end FLAN-T5 model decreases significantly with decreasing model size. However, this trend is less notable for PROGRAMFC. The high-level reasoning plan offered by reasoning programs substantially alleviates the demands on the subsequent subtask solvers. Our results show that the programguided model using FLAN-T5-small (80M parameters) as sub-task solvers can achieve comparable performance to the 137x larger FLAN-T5-XXL (11B) model with end-to-end reasoning for 4-hop claims. 

In the open-domain setting, we find that reasoning programs can enhance the retrieval of relevant evidence from the knowledge source. Figure 5 compares the retrieval performance of the one-step BM25 retriever used in the baselines to the iterative step-by-step BM25 retriever in PROGRAMFC. 

We measure the recall of the gold paragraphs for the top-10 retrieved paragraphs (recall@10). For PROGRAMFC, we combine the retrieved paragraphs of all steps and we consider the top-10 results. We can see in Figure 5 that PROGRAMFC outperforms one-step retrieval on all datasets, with the largest improvement of 37.1% on HOVER 4- hop. This is because some information may not be present in the original claim, but is only revealed during the reasoning process ( _e.g._ , “Christopher Nolan” in Figure 1). Thus, iterative retrieval guided by the reasoning program yields better results. 

## **4.3 Interpretability of Reasoning Programs** 

An advantage of PROGRAMFC is that it improves the interpretability of fact-checking compared to end-to-end models, as the explicit program can aid human understanding and debugging. Examples of generated reasoning programs can be found in Figure 7 of Appendix B. To assess the quality of the generated reasoning programs, we sampled 300 claims where PROGRAMFC _incorrectly_ predicted the final veracity labels from the HOVER 2-hop, 3-hop, and 4-hop datasets, with 100 examples per dataset. We asked human annotators to analyze the error types and we classified the results into three categories: ( _i_ ) _Syntactic errors_ , where the program does not conform to the defined grammar and cannot be parsed, ( _ii_ ) _Semantic errors_ , which include incorrect or missing arguments/variables ( _Token_ ), incorrect program structure ( _Structure_ ), and incorrect sub-task calls ( _Subtask_ ), and ( _iii_ ) _Incorrect execution_ , where the program is correct, but where the incorrect prediction is a result of its execution. 

We show the error analysis in Table 2. First, no syntax errors were found in our samples, indicating that Codex effectively generates executable programs through few-shot in-context learning. 

## **`Claim:`** 

```
Emery, located in the same state as Edison Local School District, is a ghost town. It is near the
city that lies close to the Ohio Turnpike, a 241.26 mi highway.
```

```
Predicted Program:
answer_1 = Question("Which state is Emery located in?")
answer_2 = Question("Which state is Edison Local School District located in?")
fact_1 = Verify("{answer_1} and {answer_2} are the same state.")
fact_2 = Verify("Emery is a ghost town.")
answer_3 = Question("Which city is near Emery?")
answer_4 = Question("Which city lies close to the Ohio Turnpike, a 241.26 mi highway?")
fact_3 = Verify("{answer_3} is near {answer_4}.") fact_3 = Verify(“Emery is near {answer_4}.”)
label = Predict(fact_1 and fact_2 and fact_3)
```

Figure 6: An error case from the HOVER 4-hop dataset where the generated reasoning program has an incorrect program structure. The incorrect segment(s) are marked in **red** , and the correct revisions are marked in **green** . 

|**Error Type**<br>**Proportion(%)**<br>2-hop<br>3-hop<br>4-hop|Model|2-hop|HOVER<br>3-hop|4-hop|FEVEROUS|
|---|---|---|---|---|---|
|Syntax error<br>**0%**<br>**0%**<br>**0%**|InstructGPT|||||
|Semantic error<br>**29%**<br>**38%**<br>**77%**|- Direct|56.51|51.75|49.68|60.13|
|Token<br>8%<br>20%<br>18%<br>Structure<br>19%<br>13%<br>57%<br>Subtask<br>2%<br>5%<br>2%|- ZS-CoT<br>- CoT|50.30<br>**57.20**|52.30<br>53.66|51.58<br>51.83|54.78<br>**61.05**|
|Incorrect execution<br>**71%**<br>**62%**<br>**23%**|- Self-Ask|51.54|51.47|52.45|56.82|
||Codex|55.57|53.42|45.59|57.85|
|Table 2: Reasoning program evaluation for incorrectly-|FLAN-T5|48.27|52.11|51.13|55.16|
|predicted examples from each hop length in HOVER.|ProgramFC|54.27|**54.18**|**52.88**|59.66|



Table 2: Reasoning program evaluation for incorrectlypredicted examples from each hop length in HOVER. 

Second, for 2-hop claims, 71% of the programs are found to be correct and the majority of the errors are the result of incorrect program execution, where the question answering or the fact-checking modules failed to return the correct answer. 

Third, as the complexity of the claims increased, the proportion of semantic errors in the programs increased, with structural errors becoming particularly prevalent. This highlights the difficulty of generating the appropriate step-by-step reasoning strategies for claims that require long-chain reasoning. An example structural error is shown in Figure 6, where the model fails to parse the second sentence of the claim into correct program instructions. Additional error examples can be found in Appendix C. 

## **4.4 Closed-Book Fact-Checking** 

Finally, we evaluate the closed-book setting, where the model does not have access to any knowledge source and needs to rely on its parametric knowledge only. The baseline models from groups I and II in Table 1 are trained with (evidence, claim) pairs and thus are not applicable in this setting. We compare our method to the baselines that use large language models for in-context learning, including Codex (code-davinci-002) and FLAN-T5 from Table 1. 

Table 3: Closed-book setting: macro-F1 scores for PROGRAMFC and for the baselines. 

We also include the 175B-parameter InstructGPT (text-davinci-002) (Ouyang et al., 2022) with four different prompts: ( _i_ ) _direct_ prompting with the claim, ( _ii_ ) CoT (Wei et al., 2022) or chain-of-thought prompting with demonstrations, ( _iii_ ) ZS-CoT (Kojima et al., 2022) or zero-shot chain-of-thought with the prompt “let’s think step by step”, and ( _iv_ ) Self-Ask (Press et al., 2022), which is a variant of CoT that guides the model reasoning by asking a series of questions. The detailed prompting templates are given in Appendix E. 

Our results, presented in Table 3, show that most models achieve a Macro-F1 score only slightly above random guessing on the HOVER dataset, indicating the difficulty of solely relying on parametric knowledge of large language models for factchecking complex claims. Similarly to the observations in Section 4.1, we see a trend of improved performance as the number of the required reasoning hops increases. Chain-of-thought prompting scores an average 2.7 points higher than direct prompting, highlighting the importance of step-by-step reasoning for complex fact-checking. It outperforms our PROGRAMFC on HOVER 2-hop and FEVEROUS but performs worse on HOVER 3-hop and 4-hop. 

This can be due to CoT generating free-form explanations, which can lead to unpredictable errors in long reasoning chains. In contrast, our program generation-and-execution strategy is more stable for longer reasoning chains. 

## **5 Conclusion and Future Work** 

We proposed PROGRAMFC, a few-shot neurosymbolic model for fact-checking that learns to map input claims to a reasoning program consisting of a sequence of sub-task function calls for answering a question, for fact-checking a simple claim, and for computing a logical expression. Then factchecking is performed by executing that program. PROGRAMFC combines the advantages of symbolic programs, such as explainability, with the flexibility of end-to-end neural models. Using Codex as the program generator, PROGRAMFC demonstrates promising performance on HOVER and FEVEROUS with only a small number of incontext demonstrations and no additional training. We also investigated the impact of model size and the benefits of programs for retrieval, and we analyzed the errors. The results indicated that PROGRAMFC effectively balances model capability, learning efficiency, and interpretability. 

In future work, we want to adapt PROGRAMFC to more real-world fact-checking scenarios, such as fake news detection and multi-modal fact-checking, with advanced reasoning program design and subtask functionalities. 

## **Limitations** 

We identify two main limitations of PROGRAMFC. First, the claims in the HOVER and the FEVEROUS datasets, despite being complex in their surface form, mostly only require _explicit_ multi-step reasoning, _i.e._ , the decomposition can be derived from the claim’s syntactic structure or how the claim is framed. This lowers the difficulty of generating reasoning programs. However, for many real-world complex claims, the reasoning is often _implicit_ . For example, for the claim _“Aristotle couldn’t have used a laptop”_ , the reasoning program looks as follows: 

answer_1 = Question(“When did Aristotle live?”); 

answer_2 = Question(“When was the laptop invented?”); 

fact_1 = Verify(“answer_1 is before answer_2.”); label = Predict(fact_1) 

Generating reasoning programs for such implicit complex claims requires a deeper understanding of the claim and also access to world and commonsense knowledge. We conducted preliminary experiments on these types of claims, but we found that our Codex-based generator struggled to produce a correct reasoning program. This highlights the gap in applying our PROGRAMFC to fact-check real-world claims. Addressing these challenges is an important direction for future work. 

Second, we have found that PROGRAMFC has a higher computational cost than the baseline endto-end fact-checking models. It requires calling large language models for program generation and further calling multiple sub-task models. This results in the actual computational time that is _∼_ 4- 5 _×_ higher than for an end-to-end FLAN-T5 model. Thus, developing more efficient methods for program generation and execution is one of the important directions for future work. 

## **Ethics Statement** 

**Biases.** We note that there might be some biases in the data used to train the LLMs, as well as in some factuality judgments. Both of these are beyond our control. 

**Intended Use and Misuse Potential.** Our models can be of interest to the general public and could also save a lot of time to human fact-checkers. However, they could also be misused by malicious actors. We, therefore, ask researchers to exercise caution. 

**Environmental Impact.** We would like to warn that the use of large language models requires a significant amount of computation and the use of GPUs/TPUs for training, which contributes to global warming. This is a bit less of an issue in our case, as we do not train such models from scratch; rather, we perform few-shot in-context learning. Still, the large language model (Codex) we are calling is likely running on a GPU. 

## **Acknowledgements** 

This work was supported by the National Science Foundation award #2048122. The views expressed are those of the author and do not reflect the official policy or position of the US government. We thank Alex Mei, Xinyi Wang, Danqing Wang, Sharon Levy, Gyuwan Kim, and other members of the UCSB NLP group for their valuable feedback. 

## **References** 

- Naser Ahmadi, Joohyung Lee, Paolo Papotti, and Mohammed Saeed. 2019. Explainable fact checking with probabilistic answer set programming. In _Proceedings of the Truth and Trust Online Conference (TTO)_ , London, UK. 

- Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021. FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information. In _Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks_ , Online. 

- Rami Aly and Andreas Vlachos. 2022. Natural logicguided autoregressive multi-hop document retrieval for fact verification. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 6123–6135, Abu Dhabi, United Arab Emirates. 

- Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. 2020. Generating fact checking explanations. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 7352–7364, Online. 

- Isabelle Augenstein, Christina Lioma, Dongsheng Wang, Lucas Chaves Lima, Casper Hansen, Christian Hansen, and Jakob Grue Simonsen. 2019. MultiFC: A real-world multi-domain dataset for evidence-based fact checking of claims. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 4685–4697, Hong Kong, China. 

- Giorgio Barnabò, Federico Siciliano, Carlos Castillo, Stefano Leonardi, Preslav Nakov, Giovanni Da San Martino, and Fabrizio Silvestri. 2022. FbMultiLingMisinfo: Challenging large-scale multilingual benchmark for misinformation detection. In _Proceedings of the 2022 International Joint Conference on Neural Networks (IJCNN)_ , pages 1–8, Padova, Italy. 

- Giorgio Barnabò, Federico Siciliano, Carlos Castillo, Stefano Leonardi, Preslav Nakov, Giovanni Da San Martino, and Fabrizio Silvestri. 2023. Deep active learning for misinformation detection using geometric deep learning. _Online Social Networks and Media_ , 33:100244. 

- Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. _ArXiv preprint_ , abs/2004.05150. 

- Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In _Proceedings of the 2015 Conference on Empirical_ 

_Methods in Natural Language Processing (EMNLP)_ , pages 632–642, Lisbon, Portugal. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)_ , Online. 

- Jifan Chen, Aniruddh Sriram, Eunsol Choi, and Greg Durrett. 2022a. Generating literal and implied subquestions to fact-check complex claims. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 3495–3516, Abu Dhabi, United Arab Emirates. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. _ArXiv preprint_ , abs/2107.03374. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2022b. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _CoRR_ , abs/2211.12588. 

- Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. 2022. Binding language models in symbolic languages. _CoRR_ , abs/2210.02875. 

- Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. 

Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. _CoRR_ , abs/2210.11416. 

- Limeng Cui, Kai Shu, Suhang Wang, Dongwon Lee, and Huan Liu. 2019. dEFEND: A system for explainable fake news detection. In _Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM)_ , pages 2961– 2964, Beijing, China. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 4171–4186, Minneapolis, Minnesota, USA. 

- Mohamed H. Gad-Elrab, Daria Stepanova, Jacopo Urbani, and Gerhard Weikum. 2019. Exfakt: A framework for explaining facts over knowledge graphs and text. In _Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining (WSDM)_ , pages 87–95, Melbourne, Australia. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2022. PAL: program-aided language models. _CoRR_ , abs/2211.10435. 

- Max Glockner, Yufang Hou, and Iryna Gurevych. 2022. Missing counter-evidence renders NLP factchecking unrealistic for misinformation. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 5916–5936, Abu Dhabi, United Arab Emirates. 

- Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. 2022. A survey on automated fact-checking. _Transactions of the Association for Computational Linguistics_ , 10:178–206. 

- Ashim Gupta and Vivek Srikumar. 2021. X-Fact: A new benchmark dataset for multilingual fact checking. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)_ , pages 675– 682, Online. 

- Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2021. DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradientdisentangled embedding sharing. _ArXiv preprint_ , abs/2111.09543. 

- Kelvin Jiang, Ronak Pradeep, and Jimmy Lin. 2021. Exploring listwise evidence reasoning with T5 for fact verification. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference_ 

   - _on Natural Language Processing (ACL-IJCNLP)_ , pages 402–410, Online. 

- Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. HoVer: A dataset for many-hop fact extraction and claim verification. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 3441–3460, Online. 

- Shailza Jolly, Pepa Atanasova, and Isabelle Augenstein. 2022. Generating fluent fact checking explanations with unsupervised post-editing. _Information_ , 13(10):500. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. _CoRR_ , abs/2205.11916. 

- Neema Kotonya and Francesca Toni. 2020. Explainable automated fact-checking for public health claims. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7740–7754, Online. 

- Amrith Krishna, Sebastian Riedel, and Andreas Vlachos. 2022. ProoFVer: Natural logic theorem proving for fact verification. _Transactions of the Association for Computational Linguistics (TACL)_ , 10:1013–1030. 

- Nayeon Lee, Yejin Bang, Andrea Madotto, and Pascale Fung. 2021. Towards few-shot fact-checking via perplexity. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 1971–1981, Online. 

- Nayeon Lee, Belinda Z. Li, Sinong Wang, Wen-tau Yih, Hao Ma, and Madian Khabsa. 2020. Language models as fact checkers? In _Proceedings of the Third Workshop on Fact Extraction and VERification (FEVER)_ , pages 36–41, Online. 

- Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, JhengHong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)_ , pages 2356–2362, Online. 

- Alisa Liu, Swabha Swayamdipta, Noah A. Smith, and Yejin Choi. 2022. WANLI: Worker and AI collaboration for natural language inference dataset creation. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 6826–6847, Abu Dhabi, United Arab Emirates. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. _ArXiv preprint_ , abs/1907.11692. 

- Zhenghao Liu, Chenyan Xiong, Maosong Sun, and Zhiyuan Liu. 2020. Fine-grained fact verification with kernel graph attention network. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 7342–7351, Online. 

- Yi-Ju Lu and Cheng-Te Li. 2020. GCAN: Graph-aware co-attention networks for explainable fake news detection on social media. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 505–514, Online. 

- Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ramakanth Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun, and Thomas Scialom. 2023. Augmented language models: a survey. _CoRR_ , abs/2302.07842. 

- Preslav Nakov, David Corney, Maram Hasanain, Firoj Alam, Tamer Elsayed, Alberto BarrónCedeño, Paolo Papotti, Shaden Shaar, and Giovanni Da San Martino. 2021a. Automated fact-checking for assisting human fact-checkers. In _Proceedings of the Joint Conference on Artificial Intelligence (IJCAI)_ , pages 4551–4558, Online. 

- Preslav Nakov, Giovanni Da San Martino, Tamer Elsayed, Alberto Barrón-Cedeño, Rubén Míguez, Shaden Shaar, Firoj Alam, Fatima Haouari, Maram Hasanain, Nikolay Babulkov, Alex Nikolov, Gautam Kishore Shahi, Julia Maria Struß, and Thomas Mandl. 2021b. The CLEF-2021 CheckThat! lab on detecting check-worthy claims, previously factchecked claims, and fake news. In _Proceedings of the 43rd European Conference on Information Retrieval (ECIR)_ , pages 639–649, Lucca, Italy. 

- Van-Hoang Nguyen, Kazunari Sugiyama, Preslav Nakov, and Min-Yen Kan. 2020. FANG: leveraging social context for fake news detection using graph representation. In _Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)_ , pages 1165–1174. 

- Yixin Nie, Haonan Chen, and Mohit Bansal. 2019. Combining fact extraction and verification with neural semantic matching networks. In _Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI)_ , pages 6859–6866, Honolulu, Hawaii, USA. 

- Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial NLI: A new benchmark for natural language understanding. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 4885–4901, Online. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, 

Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. _CoRR_ , abs/2203.02155. 

- Liangming Pan, Wenhu Chen, Wenhan Xiong, MinYen Kan, and William Yang Wang. 2021. Zero-shot fact verification by claim generation. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)_ , pages 476–483, Online. 

- Alicia Parrish, William Huang, Omar Agha, Soo-Hwan Lee, Nikita Nangia, Alexia Warstadt, Karmanya Aggarwal, Emily Allaway, Tal Linzen, and Samuel R. Bowman. 2021. Does putting a linguist in the loop improve NLU data collection? In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 4886–4901, Punta Cana, Dominican Republic. 

- Kashyap Popat, Subhabrata Mukherjee, Jannik Strötgen, and Gerhard Weikum. 2017. Where the truth lies: Explaining the credibility of emerging claims on the web and social media. In _Proceedngs of the International World Wide Web Conference (WWW)_ , pages 1003–1012. 

- Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. 2022. Measuring and narrowing the compositionality gap in language models. _CoRR_ , abs/2210.03350. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 21:140:1–140:67. 

- Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. _Foundations and Trends in Information Retrieval_ , 3(4):333–389. 

- Arkadiy Saakyan, Tuhin Chakrabarty, and Smaranda Muresan. 2021. COVID-fact: Fact extraction and verification of real-world claims on COVID-19 pandemic. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)_ , pages 2116–2129, Online. 

- Aalok Sathe, Salar Ather, Tuan Manh Le, Nathan Perry, and Joonsuk Park. 2020. Automated fact-checking of claims from Wikipedia. In _Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC)_ , pages 6874–6882, Marseille, France. 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. _CoRR_ , abs/2302.04761. 

- Tal Schuster, Adam Fisch, and Regina Barzilay. 2021. Get your vitamin C! robust fact verification with contrastive evidence. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 624–643, Online. 

- Amir Soleimani, Christof Monz, and Marcel Worring. 2020. BERT for evidence retrieval and claim verification. In _Advances in Information Retrieval (ECIR)_ , volume 12036, pages 359–366. 

- James Thorne and Andreas Vlachos. 2018. Automated fact checking: Task formulations, methods and future directions. In _Proceedings of the 27th International Conference on Computational Linguistics (COLING)_ , pages 3346–3359, Santa Fe, New Mexico, USA. 

- James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 809–819, New Orleans, Louisiana. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems (NeurIPS)_ , pages 5998–6008, Long Beach, California, USA. 

- David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7534–7550, Online. 

- David Wadden, Kyle Lo, Bailey Kuehl, Arman Cohan, Iz Beltagy, Lucy Lu Wang, and Hannaneh Hajishirzi. 2022a. SciFact-open: Towards open-domain scientific claim verification. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 4719–4734, Abu Dhabi, United Arab Emirates. 

- David Wadden, Kyle Lo, Lucy Wang, Arman Cohan, Iz Beltagy, and Hannaneh Hajishirzi. 2022b. MultiVerS: Improving scientific claim verification with weak supervision and full-document context. In _Findings of the Association for Computational Linguistics: NAACL 2022_ , pages 61–76, Seattle, Washington, USA. 

   - Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, and Denny Zhou. 2022. Selfconsistency improves chain of thought reasoning in language models. _CoRR_ , abs/2203.11171. 

   - Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. _ArXiv preprint_ , abs/2201.11903. 

   - Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACLHLT)_ , pages 1112–1122, New Orleans, Louisiana, USA. 

   - Dustin Wright, David Wadden, Kyle Lo, Bailey Kuehl, Arman Cohan, Isabelle Augenstein, and Lucy Wang. 2022. Generating scientific claims for zero-shot scientific fact checking. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 2448–2460, Dublin, Ireland. 

   - Fan Yang, Shiva K. Pentyala, Sina Mohseni, Mengnan Du, Hao Yuan, Rhema Linder, Eric D. Ragan, Shuiwang Ji, and Xia (Ben) Hu. 2019. XFake: Explainable fake news detector with visualizations. In _Proceedings of the The World Wide Web Conference (WWW)_ , pages 3600–3604, San Francisco, California, USA. 

   - Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 2369–2380, Brussels, Belgium. 

   - Wanjun Zhong, Jingjing Xu, Duyu Tang, Zenan Xu, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. 2020. Reasoning over semantic-level graph for fact checking. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 6170–6180, Online. 

   - Jie Zhou, Xu Han, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. 2019. GEAR: Graph-based evidence aggregating and reasoning for fact verification. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 892–901, Florence, Italy. 

- William Yang Wang. 2017. “Liar, liar pants on fire”: A new benchmark dataset for fake news detection. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 422–426, Vancouver, Canada. 

## **A Implementation Details about the Baselines** 

In this section, we give the implementation details for the seven baselines we used in our work. Typical ways to perform few-shot fact-checking using large language models are fine-tuning and incontext learning. Thus, we categorize the baselines into three categories. 

## **A.1 Pre-trained Models** 

Pre-trained models use pretrained Transformers (Vaswani et al., 2017) such as BERT (Devlin et al., 2019) and T5 (Raffel et al., 2020) for factchecking. For few-shot learning, we fine-tune them using 20 randomly sampled training examples from HOVER or FEVEROUS. We ran the training 10 times with different random seeds and report the average performance on the validation set. We chose two models: 

- BERT-FC (Soleimani et al., 2020): It uses BERT for claim verification. The claim and the evidence are concatenated ([CLS] claim [SEP] evidence) and used as input for a binary classification task to predict the veracity label of the claim. We use the bert-large-uncased (345M parameters) model provided in HuggingFace.[2] 

- LisT5 (Jiang et al., 2021): This is a factchecking framework built with a pretrained sequence-to-sequence transformer, namely T5 (Raffel et al., 2020), as its backbone. We adopt the “listwise concatenation” proposed in the paper for label prediction, which concatenates all candidate evidence sentences into a single input and we train the t5-large model to directly classify the claim as Supported or Refuted. We use the original implementation of this model.[3] 

## **A.2 FC/NLI Fine-Tuned Models** 

These models are pretrained Transformer models that have been specifically fine-tuned on singlehop fact-checking datasets ( _e.g._ , FEVER) or natural language inference (NLI) datasets. This additional training allows these models to excel at fact-checking simple claims, and thus they can generalize better to complex claims that require multihop reasoning during further few-shot fine-tuning. 

2https://huggingface.co/ 

> 3https://github.com/castorini/pygaggle/tree/ master/experiments/list5 

In this category, we selected the following three 

- RoBERTa-NLI (Nie et al., 2020) fine-tunes RoBERTa-large (Liu et al., 2019) on a combination of four well-known NLI datasets: SNLI (Bowman et al., 2015), MNLI (Williams et al., 2018), FEVER-NLI (Nie et al., 2019), ANLI (R1, R2, R3) (Nie et al., 2020). We used the public model checkpoint available at HuggingFace[4] and we further fine-tuned it with 20 random examples from HOVER/FEVEROUS. 

- DeBERTaV3-NLI (He et al., 2021) finetunes the DeBERTaV3-large model on 885,242 NLI hypothesis–premise pairs from FEVER and on four NLI datasets: MNLI, ANLI, LingNLI (Parrish et al., 2021), and WANLI (Liu et al., 2022). This is the bestperforming NLI model on HuggingFace as of 06/06/2022.[5] 

- MULTIVERS (Wadden et al., 2022b), formerly known as LongChecker, uses the LongFormer (Beltagy et al., 2020) for claim verification to address the long input evidence problem. We use a model checkpoint finetuned on FEVER.[6] 

## **A.3 In-Context Learning Models** 

These models have recently shown strong few-shot learning ability in various NLP tasks. By prompting a large language model with a few in-context examples, the model can quickly learn a task from demonstrations. To make a fair comparison to our model, we choose two in-context learning baselines as follows. 

- Codex (Chen et al., 2021) is used in our model to generate reasoning programs. One straightforward baseline directly uses it for fact-checking. To this end, we prompt Codex (code-davinci-002) as follows: “<Evidence> Based on the above information, is it true that <Claim>? True or False? The answer is:”. We prefix the same 20 in-context examples for our model before the prompt as demonstrations. 

4https://huggingface.co/ynie/ roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli 5https://huggingface.co/MoritzLaurer/ 

DeBERTa-v3-large-mnli-fever-anli-ling-wanli 

6https://github.com/dwadden/multivers 

- FLAN-T5 (Chung et al., 2022) is an improved version of T5, which is fine-tuned on 1.8K tasks phrased as instructions, with and without exemplars, _i.e._ , zero-shot and few-shot. The model has shown strong performance in various in-context few-shot learning NLP tasks, such as reasoning, and question-answering. We prompt the model with the same format as we used in Section 3.4: “<Evidence> Q: <Claim> Is it true that <Claim>? True or False? The answer is:”, prefixing with the same 20 in-context examples. We also use the same model size (FLAN-T5-XXL 3B) with our model for fair comparison. 

## **B Examples of Generated Reasoning Programs** 

Figure 7 shows six examples of generated reasoning programs by PROGRAMFC that cover diverse reasoning chains. 

## **C Error Analysis for Reasoning Programs** 

Figure 8 shows five examples of erroneous cases where the generated reasoning programs are incorrect. We provide explanations for each of the error cases below: 

**Example 1** It generates a wrong logical reasoning operator for the final step. The correct logic should be “not (fact_1 and fact_2)” instead of “fact_1 and fact_2”. 

**Example 2** It fails to perform co-reference resolution for the arguments in the third and the fourth reasoning steps. “This album” should be replaced with “The bluegrass” to make the sub-task contextindependent. “This musical” should be replaced with the variable “answer_1” from the first step. 

**Example 3** It fails to create a meaningful problem decomposition for the claim. It generates a trivial program that simply repeats the original claim. 

## **D Program Generation Prompts** 

Our manually written prompts for the HOVER and the FEVEROUS-S datasets are given in Listings 1 and 2, respectively. 

## **E Prompts for Closed-Book Fact-Checking** 

Below we show the templates for the four prompting methods used for InstructGPT for the closedbook fact-checking setting in Section 4.4. 

## **Direct Prompting** 

# Answer the following true/false questions: 

Is it true that The woman the story behind Girl Crazy is credited to is older than Ted Kotcheff? The answer is: False 

**(** _· · ·_ **more in-context examples here** _· · ·_ **)** 

Is it true that <input_claim>? 

The answer is: 

## **ZS-CoT Prompting** 

# Answer the following true/false question: Is it true that <input_claim>? True or False? Let us think step -by -step. The answer is: 

## **CoT Prompting** 

# Answer the following true/false questions: 

Is it true that The woman the story behind Girl Crazy is credited to is older than Ted Kotcheff? Let's think step by step. Girl Crazy's story is credited to Hampton Del Ruth. Hampton Del Ruth was born on September 7, 1879. Ted Kotcheff was born on April 7, 1931. Therefore , the answer is: False. 

**(** _· · ·_ **more in-context examples here** _· · ·_ **)** 

Is it true that <input_claim>? Let's think step by step. 

## **Self-Ask Prompting** 

# Answer the following true/false questions: 

Is it true that The woman the story behind Girl Crazy is credited to is older than Ted Kotcheff? Q: The story behind Girl Crazy is credited to whom? A: Hampton Del Ruth Q: Is Hampton Del Ruth older than Ted Kotcheff? A: No So the final answer is: False. 

**(** _· · ·_ **more in-context examples here** _· · ·_ **)** Is it true that <input_claim>? 

**Example 4** It fails to generate a fine-grained reasoning structure for the input claim. It also generates a trivial program that simply separates the claim into sentences. 

**Example 5** It generates a redundant reasoning step “Question("When was the musician born?")”, which does not add any new information to the reasoning chain. 

## **`Claim:`** 

```
Tritoniaand Phyteumaare both names for a plant genus.
```

## **`Predicted Program:`** 

```
fact_1 = Verify("Tritoniais a name for a plant genus.")
fact_2 = Verify("Phyteumais a name for a plant genus.")
label = Predict(fact_1and fact_2)
```

## **`Claim:`** 

```
The country that Fujairah College is located inhad a 2013 population of 9.2 million until it was
hit by the plague in 1483 when the population was halved.
```

## **`Predicted Program:`** 

```
answer_1 = Question("Which country is Fujairah College located in?")
fact_1 = Verify("{answer_1} had a 2013 population of 9.2 million.")
fact_2 = Verify("{answer_1} was hit by the plague in 1483.")
fact_3 = Verify("The population of {answer_1} was halved in 1483.")
label = Predict(fact_1and fact_2and fact_3)
```

## **`Claim:`** 

```
The first female board member for the Liberal Party, she was born in Vestfold county in Norway.
```

## **`Predicted Program:`** 

```
answer_1 = Question("Who was the first female board member for the Liberal Party?")
fact_1 = Verify("{answer_1} was born in Vestfold county in Norway.")
label = Predict(fact_1)
```

## **`Claim:`** 

```
The solicitor who won the show  Back to Reality ahead of Maureen Rees and Craig Phillips is
English. The solicitor that was a chair of Global Witness is also English.
```

## **`Predicted Program:`** 

```
answer_1 = Question("Which solicitor won the show Back to Reality ahead of Maureen Rees and Craig
Phillips?")
```

```
answer_2 = Question("Which solicitor was a chair of Global Witness?")
fact_1 = Verify("{answer_1} is English.")
fact_2 = Verify("{answer_2} is English.")
label = Predict(fact_1and fact_2)
```

## **`Claim:`** 

```
Anthony Burgess addressed the novelist and essayist, the author of Grimus, in a lengthy love
letter. The author is of the same nationality as Raj Koothrappali.
```

## **`Predicted Program:`** 

```
answer_1 = Question("Which novelist and essayist did Anthony Burgess address in a lengthy love
letter?")
fact_1 = Verify("{answer_1} is the author of Grimus.")
answer_2 = Question("What is the nationality of Raj Koothrappali?")
fact_2 = Verify("{answer_1} is of the same nationality as {answer_2}.")
label = Predict(fact_1 and fact_2)
```

## **`Claim:`** 

```
The critically acclaimed film, that Buddy Baker scored in 1975, is a Walt Disney film. It was
produced first before the film that featured Bruce M. Fischer as Mr. Coogar.
```

## **`Predicted Program:`** 

```
answer_1 = Question("Which critically acclaimed film did Buddy Baker score in 1975?")
fact_1 = Verify("{answer_1} is a Walt Disney film.")
answer_2 = Question("Which film featured Bruce M. Fischer as Mr. Coogar?")
fact_2 = Verify("{answer_1} was produced first before {answer_2}.")
label = Predict(fact_1 and fact_2)
```

Figure 7: Examples of generated reasoning programs by PROGRAMFC. 

**Semantic Error — Token:** incorrect or missing arguments/variables **`Example 1:`** `Bitter Jester and The Future of Food are not both documentaries about food.` **`Predicted Program:`** `fact_1 = Verify("Bitter Jester is a documentary about food.") fact_2 = Verify("The Future of Food is a documentary about food.") label = Predict(fact_1 and fact_2)    label = Predict(not (fact_1 and fact_2))` **`Example 2:`** `The record producer that produced the bluegrass album was born on 22 June, 1944. This album inspired a Tony award winning musical. This musical had a character that was originated by Carmen Cusack.` **`Predicted Program:`** `answer_1 = Question("Which record producer produced the bluegrass album?") fact_1 = Verify("{answer_1} was born on 22 June, 1944.") fact_2 = Verify("This album inspired a Tony award winning musical.") fact_3 = Verify("This musical had a character that was originated by Carmen Cusack.") label = Predict(fact_1 and fact_2 and fact_3) answer_1 = Question("Which record producer produced the bluegrass album?") fact_1 = Verify("{answer_1} was born on 22 June, 1944.") fact_2 = Verify("The bluegrass album inspired a Tony award winning musical.") fact_3 = Verify("{answer_1} had a character that was originated by Carmen Cusack.") label = Predict(fact_1 and fact_2 and fact_3)` **Semantic Error — Structure:** incorrect program structure **`Example 3:`** `Richard Strauss, German composer of Die Nacht, composed another well-established opera called Der Rosenkavalier.` **`Predicted Program:`** `fact_1 = Verify("Richard Strauss, German composer of Die Nacht, composed another well-established opera called Der Rosenkavalier.") label = Predict(fact_1) fact_1 = Verify("Richard Strauss is a German composer of Die Nacht.”) fact_2 = Verify("Richard Strauss composed a well-established opera called Der Rosenkavalier.") label = Predict(fact_1 and fact_2)` **`Example 4:`** `The film Deanna Oliver produced in 1999 grossed $36.8 million domestically. The musical film based on coach Herman Boone, did not.` **`Predicted Program:`** `fact_1 = Verify("The film Deanna Oliver produced in 1999 grossed $36.8 million domestically.") fact_2 = Verify("The musical film based on coach Herman Boone, did not.") label = Predict(fact_1 and fact_2) answer_1 = Question("What film did Deanna Oliver produce in 1999?") fact_1 = Verify("{answer_1} grossed $36.8 million domestically.") answer_2 = Question("What musical film was based on coach Herman Boone?") fact_2 = Verify("{answer_2} grossed less than $36.8 million domestically.") label = Predict(fact_1 and fact_2)` **Semantic Error — Subtask:** missing / redundant / incorrect sub-task calls **`Example 5:`** `The musician, who founded Morningwood with Max Green, is older than Max Green.` **`Predicted Program:`** `answer_1 = Question("Who founded Morningwood with Max Green?") answer_2 = Question("When was Max Green born?") answer_3 = Question("When was the musician born?") fact_1 = Verify("{answer_3} is older than {answer_2}.")     {answer_1} is older than {answer_2}. label = Verify(fact_1)` 

Figure 8: Example error cases where the generated reasoning programs are incorrect. The incorrect segment(s) are marked in **red** , and the correct revisions are marked in **green** . 

- '''Generate a python -like program that describes the reasoning steps required to verify the claim step -by-step. You can call three functions in the program: 1. Question () to answer a question; 2. Verify () to verify a simple claim; 3. Predict () to predict the veracity label.''' 

- # The claim is that Howard University Hospital and Providence Hospital are both located in Washington , D.C. 

- def program (): fact_1 = Verify("Howard University Hospital is located in Washington , D.C.") fact_2 = Verify("Providence Hospital is located in Washington , D.C.") label = Predict(fact_1 and fact_2) 

- # The claim is that WWE Super Tuesday took place at an arena that currently goes by the name TD Garden. 

- def program (): 

   - answer_1 = Question("Which arena the WWE Super Tuesday took place?") fact_1 = Verify(f"{answer_1} currently goes by the name TD Garden.") label = Predict(fact_1) 

- # The claim is that Talking Heads , an American rock band that was "one of the most critically acclaimed bands of the 80's" is featured in KSPN's AAA format. 

- def program (): 

   - fact_1 = Verify("Talking Heads is an American rock band that was 'one of the most critically acclaimed bands of the 80's'.") 

   - fact_2 = Verify("Talking Heads is featured in KSPN's AAA format.") label = Predict(fact_1 and fact_2) 

- # The claim is that An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season. 

- def program (): 

   - answer_1 = Question("Which Formula 1 car was designed by Peter McCool during the 2007 Formula One season?") 

   - fact_1 = Verify(f"An IndyCar race driver drove the car {answer_1 }.") label = Predict(fact_1) 

- # The claim is that Gina Bramhill was born in a village. The 2011 population of the area that includes this village was 167 ,446. 

- def program (): 

   - answer_1 = Question("Which village was Gina Bramhill born in?") fact_1 = Verify(f"The 2011 population of the area that includes {answer_1} was 167 ,446.") 

   - label = Predict(fact_1) 

- # The claim is that Don Ashley Turlington graduated from Saint Joseph 's College , a private Catholic liberal arts college in Standish. 

- def program (): 

   - fact_1 = Verify("Saint Joseph 's College is a private Catholic liberal arts college is located in Standish.") 

   - fact_2 = Verify(f"Don Ashley Turlington graduated from Saint Joseph 's College." ) label = Predict(fact_1 and fact_2) 

- # The claim is that Gael and Fitness are not published in the same country. def program (): answer_1 = Question("Which country was Gael published in?") answer_2 = Question("Which country was Fitness published in?") fact_1 = Verify(f"{answer_1} and {answer_2} are not the same country.") label = Predict(fact_1) 

- # The claim is that Blackstar is the name of the album released by David Bowie that was recorded in secret. 

- def program (): 

   - fact_1 = Verify("David Bowie released an album called Blackstar.") fact_2 = Verify("David Bowie recorded an album in secret.") 

   - label = Predict(fact_1 and fact_2) 

- # The claim is that In the 2004 Hockey film produced by a former major league baseball pitcher Kurt Russell played the USA coach. 

- def program (): 

   - answer_1 = Question("Which 2004 Hockey film was produced a former major league 

baseball pitcher?") fact_1 = Verify("Kurt Russell played the USA coach in the film {answer_1 }.") label = Predict(fact_1) 

# The claim is that Along with the New York Islanders and the New York Rangers , the New Jersey Devils NFL franchise is popular in the New York metropolitan area. def program (): 

fact_1 = Verify("The New York Islanders and the New York Rangers are popular in the New York metropolitan area.") fact_2 = Verify("The New Jersey Devils NFL franchise is popular in the New York metropolitan area.") label = Predict(fact_1 and fact_2) 

# The claim is that Jack McFarland is the best known role of the host of the 64th Annual Tony Awards. def program (): answer_1 = Question("Who is the host of the 64th Annual Tony Awards?") fact_1 = Verify(f\"Jack McFarland is the best known role of {answer_1 }.") label = Predict(fact_1) 

# The claim is that The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$. def program (): fact_1 = Verify("M.I.L.F.$ was recorded by Fergie that was produced by Polow da Don.") fact_2 = Verify("M.I.L.F.$ was was followed by Life Goes On.") label = Predict(fact_1 and fact_2) 

# The claim is that Eatza Pizza and Your Pie were not founded in the same state. def program (): answer_1 = Question("Which state was Eatza Pizza founded in?") answer_2 = Question("Which state was Your Pie founded in?") fact_1 = Verify(f"{answer_1} and {answer_2} are not the same state.") label = Predict(fact_1) 

- # The claim is that Gregg Rolie and Rob Tyner , are not a keyboardist. def program (): fact_1 = Verify("Gregg Rolie is not a keyboardist.") fact_2 = Verify("Rob Tyner is not a keyboardist.") label = Predict(fact_1 and fact_2) 

- # The claim is that Maria Esther Andion Bueno , not Jimmy Connors , is the player that is from Brazil. 

- def program (): 

fact_1 = Verify("Maria Esther Andion Bueno is from Brazil.") fact_2 = Verify("Jimmy Connors is not from Brazil.") label = Predict(fact_1 and fact_2) 

# The claim is that Vladimir Igorevich Arnold died after Georg Cantor. def program (): answer_1 = Question("When did Vladimir Igorevich Arnold die?") answer_2 = Question("When did Georg Cantor die?") fact_1 = Verify(f"{answer_1} is after {answer_2 }.") label = Predict(fact_1) 

# The claim is that Barton Mine was halted by a natural disaster not Camlaren Mine. def program (): fact_1 = Verify("Barton Mine was halted by a natural disaster.") fact_2 = Verify("Camlaren Mine was not halted by a natural disaster.") label = Predict(fact_1 and fact_2) 

# The claim is that John O'Hara and Rabindranath Tagore are not the same nationality . 

def program (): answer_1 = Question("What is the nationality of John O'Hara?") answer_2 = Question("What is the nationality of Rabindranath Tagore?") fact_1 = Verify(f"{answer_1} and {answer_2} are not the same nationality.") label = Predict(fact_1) 

# The claim is that Thomas Loren Friedman has won more Pulitzer Prizes than Colson Whitehead. def program (): answer_1 = Question("How many Pulitzer Prizes has Thomas Loren Friedman won?") answer_2 = Question("How many Pulitzer Prizes has Colson Whitehead won?") fact_1 = Verify(f"{answer_1} is more than {answer_2 }.") label = Predict(fact_1) # The claim is that The model of car Trevor Bayne drives was introduced for model year 2006. The Rookie of The Year in the 1997 CART season drives it in the NASCAR Sprint Cup. def program (): answer_1 = Question("Which model of car is drived by Trevor Bayne?") fact_1 = Verify(f"{answer_1} was introduced for model year 2006.") answer_2 = Question("Who is the Rookie of The Year in the 1997 CART season?") fact_2 = Verify(f"{answer_2} drives the model of car Trevor Bayne drives in the NASCAR Sprint Cup.") label = predict(fact_1 and fact_2) # The claim is that <input_claim> def program (): 

Listing 1: The prompt used for Program Generation for HOVER. 

'''Generate a python -like program that describes the reasoning steps required to verify the claim step -by-step. You can call three functions in the program: 1. Question () to answer a question; 2. Verify () to verify a simple claim; 3. Predict () to predict the veracity label.''' 

- # The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago , United States , from August 27 to September 7) in Chicago , United States , and the world amateur welterweight title in Mexico City. 

- def program (): 

   - fact_1 = Verify("Alfredo Cornejo Cuevas was born in June 6, 1933.") fact_2 = Verify("Alfredo Cornejo Cuevas won the gold medal in the welterweight division at the Pan American Games in 1959.") fact_3 = Verify("The Pan American Games in 1959 was held in Chicago , United States , from August 27 to September 7.") 

   - fact_4 = Verify("Alfredo Cornejo Cuevas won the world amateur welterweight title in Mexico City.") 

   - label = Predict(fact_1 and fact_2 and fact_3 and fact_4) 

- # The claim is that The Footwork FA12 , which was intended to start the season , finally debuted at the San Marino Grand Prix , a Formula One motor race held at Imola on 28 April 1991. 

- def program (): 

   - fact_1 = Verify("The Footwork FA12 , which was intended to start the season.") 

   - fact_2 = Verify("The Footwork FA12 finally debuted at the San Marino Grand Prix. ") 

   - fact_3 = Verify("The San Marino Grand Prix was a Formula One motor race held at Imola on 28 April 1991.") 

   - label = Predict(fact_1 and fact_2 and fact_3) 

- # The claim is that SkyHigh Mount Dandenong (formerly Mount Dandenong Observatory) is a restaurant located on top of Mount Dandenong , Victoria , Australia. 

- def program (): 

   - fact_1 = Verify("SkyHigh Mount Dandenong is a restaurant located on top of Mount Dandenong , Victoria , Australia.") 

   - fact_2 = Verify("SkyHigh Mount Dandenong is formerly known as Mount Dandenong Observatory.") 

   - label = Predict(fact_1 and fact_2) 

- # The claim is that Before the first Europeans arrived or copra companies leased it , Maupihaa was home to Inca's in ancient times. 

- def program (): 

   - fact_1 = Verify("Maupihaa was home to Inca's in ancient times.") 

   - fact_2 = Verify("Maupihaa was home to Inca's before the first Europeans arrived or copra companies leased it.") 

   - label = Predict(fact_1 and fact_2) 

- # The claim is that Shulin , a 33.1288 km (12.7911 sq mi) land located in New Taipei City , China , a country in East Asia , has a total population of 183 ,946 in December 2018. 

- def program (): 

   - fact_1 = Verify("Shulin is a 33.1288 km (12.7911 sq mi) land located in New Taipei City , China.") 

   - fact_2 = Verify("Shulin has a total population of 183 ,946 in December 2018.") label = Predict(fact_1 and fact_2) 

- # The claim is that Sumo wrestler Toyozakura Toshiaki committed match -fixing , ending his career in 2011 that started in 1989. 

- def program (): 

   - fact_1 = Verify("Toyozakura Toshiaki ended his career in 2011 that started in 1989.") 

   - fact_2 = Verify("Toyozakura Toshiaki is a Sumo wrestler.") 

   - fact_3 = Verify("Toyozakura Toshiaki committed match -fixing.") label = Predict(fact_1 and fact_2 and fact_3) 

- # The claim is that In 1959, former Chilean boxer Alfredo Cornejo Cuevas (born June 6, 1933) won the gold medal in the welterweight division at the Pan American Games (held in Chicago , United States , from August 27 to September 7) in Chicago 

- , United States , and the world amateur welterweight title in Mexico City. 

- def program (): fact_1 = Verify("Alfredo Cornejo Cuevas is a former Chilean boxer.") fact_2 = Verify("Alfredo Cornejo won the gold medal in the welterweight division at the Pan American Games.") 

- fact_3 = Verify("The Pan American Games was held in Chicago , United States , from August 27 to September 7.") 

- fact_4 = Verify("Alfredo Cornejo won the world amateur welterweight title in Mexico City.") label = Predict(fact_1 and fact_2 and fact_3 and fact_4) 

- # The claim is that Adductor hiatus is associated with nine structures , seven of which enter and leave through hiatus. 

- def program (): 

   - fact_1 = Verify("Adductor hiatus is associated with nine structures.") 

   - fact_2 = Verify("Seven of the nine structures associated with Adductor hiatus enter and leave through hiatus.") label = Predict(fact_1 and fact_2) 

- # The claim is that Ifor Bowen Lloyd was educated at Winchester (an independent boarding school for boys in the British public school tradition) and Exeter College , Oxford where he was a member of the Library Committee of the Oxford Union Society , as well as, received a BA in Modern History in 1924. 

- def program (): 

   - fact_1 = Verify("Ifor Bowen Lloyd was educated at Winchester and Exeter College , Oxford.") 

   - fact_2 = Verify("Winchester is an independent boarding school for boys in the British public school tradition.") fact_3 = Verify("While at Oxford , Ifor Bowen Lloyd was a member of the Library Committee of the Oxford Union Society.") fact_4 = Verify("Ifor Bowen Lloyd received a BA in Modern History in 1924 at Oxford.") 

   - label = Predict(fact_1 and fact_2 and fact_3 and fact_4) 

- # The claim is that In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils ' Elias scored and Maple Leafs ' left Devils player Scott Neidermayer hurt. 

- def program (): fact_1 = Verify("In the 2001 Stanley Cup playoffs Eastern Conference Semifinals Devils ' Elias scored.") fact_2 = Verify("Maple Leafs ' left Devils player Scott Neidermayer hurt.") label = Predict(fact_1 and fact_2) 

- # The claim is that Teldenia helena is a moth first described in 1967 by Wilkinson. def program (): 

   - fact_1 = Verify("Teldenia helena is a moth.") 

   - fact_2 = Verify("Teldenia helena was first described by Wilkinson in 1967.") label = Predict(fact_1 and fact_2) 

- # The claim is that Born December 30, 1974, William Frick was a dark horse candidate in the Maryland House of Delegates appointment process. 

- def program (): 

   - fact_1 = Verify("William Frick was born in December 30, 1974.") 

   - fact_2 = Verify("William Frick was a dark horse candidate in the Maryland House of Delegates appointment process.") label = Predict(fact_1 and fact_2) 

# The claim is that <input_claim> def program (): 

Listing 2: The prompt used for Program Generation for FEVEROUS-S. 

