# **JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models** 

**Kun Zhou**[1] _[,]_[3] _[∗]_ **, Beichen Zhang**[2] _[,]_[3] _[∗]_ **, Jiapeng Wang**[2] **, Zhipeng Chen**[2] _[,]_[3] **, Wayne Xin Zhao**[2] _[,]_[3] _[†]_ **, Jing Sha**[4] **, Zhichao Sheng**[4] **, Shijin Wang**[4] _[,]_[5] **, Ji-Rong Wen**[1] _[,]_[2] _[,]_[3] 

1School of Information, Renmin University of China. 

2Gaoling School of Artificial Intelligence, Renmin University of China. 

3Beijing Key Laboratory of Big Data Management and Analysis Methods. 

4iFLYTEK Research. 5iFLYTEK AI Research (Central China). `francis_kun_zhou@163.com` , `{zhangbeichen724,batmanfly}@gmail.com` , `{jingsha,sjwang3}@iflytek.com` , `jrwen@ruc.edu.cn` 

## **Abstract** 

Mathematical reasoning is an important capability of large language models (LLMs) for real-world applications. To enhance this capability, existing work either collects large-scale math-related texts for pre-training, or relies on stronger LLMs ( _e.g.,_ GPT-4) to synthesize massive math problems. Both types of work generally lead to large costs in training or synthesis. To reduce the cost, based on open-source available texts, we propose an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data. To achieve it, we create a dataset using GPT-4 to distill its data synthesis capability into the small LLM. Concretely, we craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels. Besides, we adopt the gradient-based influence estimation method to select the most valuable math-related texts. The both are fed into GPT-4 for creating the knowledge distillation dataset to train the small LLM. We leverage it to synthesize 6 million math problems for pre-training our JiuZhang3.0 model, which only needs to invoke GPT-4 API 9.3k times and pre-train on 4.6B data. Experimental results have shown that JiuZhang3.0 achieves state-of-theart performance on several mathematical reasoning datasets, under both natural language reasoning and tool manipulation settings. Our code and data will be publicly released in `https://github.com/RUCAIBox/JiuZhang3.0` . 

## **1 Introduction** 

Large language models (LLMs) have shown remarkable capabilities on a variety of tasks [1–3]. However, they still struggle in solving complex mathematical problems [4]. Recent work has shown that it is an effective approach to training LLMs on math-related data for improving the mathematical reasoning ability [5, 6]. Typically, they either collect the math-related data from the available corpora ( _e.g.,_ webpages and books) for pre-training [7–9], or rely on stronger LLMs to synthesize high-quality math problems for fine-tuning [10–12]. Despite the success, existing approaches would generally cause large training or inference costs. Due to the complexity and diversity of mathematical problems, the former type of work mostly needs to collect a large-scale corpus ( _e.g.,_ 120B data for DeepseekMath) for training, which greatly increases the training cost [5, 7, 8]. Similarly, to guarantee the 

> _∗_ Equal contributions. 

> _†_ Corresponding author. 

Preprint. Under review. 

knowledge coverage and effectiveness of the synthetic problems, the latter type of work relies on stronger LLMs with larger scales ( _e.g.,_ GPT-4) to create massive math problems, leading to larger inference cost [6, 10, 11]. In Figure 1, we show our estimated total costs of re-implementing two math-related LLMs (>$40000), details are in Appendix C 

In this work, we aim to develop a relatively lowcost data synthesis approach for improving the mathematical reasoning abilities of LLMs. Our key idea is that **the data synthesis capability can be well learned by small LLMs** . Here, _small_ is a relative wording, which is in contrast with the extremely large or costly data synthesis models used in prior studies [6, 13], such as GPT-4 or Qwen-72B. Actually, existing work [10, 14, 15] has extensive evidence of strong learning and adaptation abilities of small LLMs for new tasks and domains with suitable strategies ( _e.g.,_ training with high-quality supervised data), including math, science, and complex multimodal tasks. However, this exploration has been neglected in prior efforts on data 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0002-02.png)


**----- Start of picture text -----**<br>
91.8 [92.6]<br>88.2  [88.6] 88.9<br>83.9<br>48.8 [50.2 52.8]<br>Expense ($)<br>**----- End of picture text -----**<br>


Figure 1: The comparison of existing work and our method in task performance and the total cost. 

synthesis. This attempt can be essentially generalized to a broader problem: _whether a small (or weak) model can produce high-quality data that is useful for training a large (or strong) model?_ Inspired by this motivation, we seek to train a relatively small yet powerful LLM for synthesizing high-quality math-related data. 

However, due to the diverse and complex nature of math problems, it is challenging to train wellperforming small LLMs for synthesizing high-quality ones. Although we can leverage GPT-4, it is not efficient to use it for synthesizing a large-scale knowledge distillation (KD) dataset. Specifically, we aim to build the KD dataset through a low-cost strategy, but it can sufficiently capture diverse and useful knowledge about math problem synthesis. Thus, we should guarantee the knowledge coverage and usefulness of the instances within the KD dataset. To achieve it, we first craft a set of prompts, and each prompt corresponds to an education stage of humans, _e.g.,_ middle school and college. Using the above prompts, the instances within the dataset can well cover broad mathematical knowledge and different difficulty levels. Besides, for usefulness, we estimate the influence of the available math-related texts and downstream tasks, by computing the gradient similarity between their corresponding synthetic data and the task instances. Then, we select the top-ranking math-related texts into the KD dataset, which are high-value ones with more positive influence on downstream tasks. By feeding the texts with prompts into GPT-4, we collect the outputs to build the KD dataset. 

Based on the KD dataset, we train DeepSeekMath-7B as our data synthesis model, which is much smaller than other commonly-used LLMs in existing work [6, 12, 13], _e.g.,_ GPT-4 and Qwen-72B. Owing to our data selection strategy, we only require GPT-4 to generate 9,335 instances based on the selected most valuable texts for training it. Then, we utilize the crafted prompts to guide it for synthesizing high-quality problems. Benefiting from the strong data synthesis capability of the small model, we only need to synthesize 5,984,584 high-quality math problems (4.6B tokens) for pre-training our JiuZhang3.0. Thus, the total inference and training cost is much less than existing work, as shown in Figure 1. After pre-training, we also collect open-source math instructions to fine-tune JiuZhang3.0. The experimental results have shown that JiuZhang3.0 can mostly outperform state-of-the-art methods across 18 evaluation datasets, in both the natural language reasoning and tool manipulation settings. Our contributions are summarized as follows: 

(1) our research provides compelling evidence that it is feasible to efficiently train a small LLM (7B) for synthesizing training data to improve the mathematical reasoning of LLMs. As results shown in Section 3.3, its synthetic data is more useful than larger LLMs in improving the performance. 

(2) we propose an efficient solution for training LLMs to improve mathematical reasoning, which only needs to invoke GPT-4 API 9.3k times and pre-train on 4.6B high-quality synthetic data, with nearly 20% total cost of existing state-of-the-art methods. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Multi-source Corpus<br>Sampled<br>5.3k Texts Training Reference Model<br>Value Estimation<br>using Gradients  Synthetic<br>4.6B Data<br>Test Sets Filter<br>Natural Language  Tool Manipulation  Initial KD<br>Reasoning Prompts Prompts Dataset<br>4k High-Value Data<br>Grade/Middle/High  Grade School<br>School<br>AIME College Secondary School JiuZhang3.0-<br>7B/8B/8x7B<br>AMC 8/10/12<br>1.Initialize Synthesis LLM 2.Boost Synthesis LLM 3.Pre-train JiuZhang3.0<br>**----- End of picture text -----**<br>


Figure 2: The pipeline of our approach. We first initialize the data synthesis LLM by distilling the knowledge from GPT-4 on randomly sampled data, then boost it using the high-value data selected by gradient-based value estimation strategy, finally utilize it for synthesizing data to train JiuZhang3.0. 

(3) JiuZhang3.0 achieves state-of-the-art performance among open-source LLMs on several tasks and settings, _e.g.,_ 52.8 (JiuZhang3.0-7B) vs. 50.2 (DeepSeekMath-7B-RL) on MATH, 89.8 (JiuZhang3.08 _×_ 7B) vs. 86.4 (MAmmoTH2-8 _×_ 7B-Plus) on GSM8k in the natural language reasoning setting. 

## **2 Approach** 

In this section, we present our approach that aims to train a small LLM for synthesizing math problems. First, we initialize the data synthesis model by training it on the KD dataset, composed of crafted prompts, randomly sampled math-related texts, and the corresponding synthetic problems and solutions from GPT-4. Then, we improve its data synthesis capability by retraining it on the updated knowledge distillation dataset, where we add the high-value math-related texts selected by gradient-based influence estimation strategy. Finally, we utilize the model to synthesize massive high-quality math problems for training JiuZhang3.0, based on the multi-source math-related corpus. 

## **2.1 Preliminary** 

We focus on training a small data synthesis LLM, for synthesizing high-quality math problemsolution pairs to pre-train LLMs and improve its mathematical reasoning capability. To guarantee the quality of the synthetic data, we utilize GPT-4 to create the knowledge distillation (KD) dataset _DpiKD_ are the input of GPT-4,= _{pi, ti,_ ˆ _qi,_ ˆ _si}[N] i_ =1[for training the small LLM, where the math-related text] _q_ ˆ _i_ and ˆ _si_ are its synthetic math problem and solution respectively. _[ t][i]_[and the prompt] Then, we train the small LLM on _DKD_ to imitate the data synthesis ability of GPT-4. Finally, we leverage it to synthesize the pre-training dataset _D_ = _{qi, si}[M] i_ =1[based on all the collected math-related texts] with randomly selected prompts _{pi, ti}[M] i_ =1[, which are used for training our JiuZhang3.0.] 

## **2.2 Initializing Data Synthesis Model** 

In this work, we consider the natural language reasoning and tool manipulation settings, where LLMs require solving the problem by generating a natural language solution [16] and an executable program with external interpreters [17], respectively. Thus, we train the data synthesis LLMs on the initial KD dataset, containing special prompts, math-related texts, and GPT-4 outputs for the two settings. 

## **2.2.1 Prompts for Math Problem Synthesis** 

We aim to craft a prompt set that can well cover the knowledge points and difficulty levels in human math education. Thus, for the natural language reasoning and tool manipulation settings, we manually craft prompt templates respectively, and each corresponds to a certain education stage. 

3 

**Prompts for Natural Language Reasoning.** We consider the following 4 human education stages and 4 worldwide competitions, _i.e.,_ Grade School, Middle School, High School, and College; AMC (American Mathematics Competition) 8, AMC 10, AMC 12, and AIME (American Invitational Mathematics Examination). Based on these, we design 8 prompts with corresponding instructions and guidelines. We show an example of grade school math problem synthesis: 

_**## Instruction:** Create an age-appropriate math word problem for grade school students based on the provided math content._ 

_**## Guidelines:** [Problem]: Craft a concise math word problem suitable for grade school, focusing on basic arithmetic operations, number sense, simple shapes, ... [Solution]: Provide a clear, step-by-step solution to the problem using simple language that a grade school student could understand, ..._ 

**Prompts for Tool Manipulation.** We consider 2 types of math problems, _i.e.,_ Grade School and Secondary School Competitions, as the competition math problems may need tools for advanced math operations ( _e.g.,_ integral computation), while grade school math problems are much easier and can be solved by basic operations ( _e.g.,_ +-*/). As tool manipulation solves the problem via executable programs, we use more words in prompts to emphasize the data format and show an example as follows: 

_**## Instruction:** Please gain inspiration from the following random math content to create a highquality ... Present your output in two distinct sections: [Problem Description] and [Solution]._ 

_**## Guidelines:** [Problem]: This should be completely self-contained, providing all the contextual information one needs to understand, ... [Solution]: Offer a comprehensive, correct solution that accurately addresses the [Problem] you provided using Python code, ..._ 

_**## Example:** [Problem Description] Janet buys 3 pounds of broccoli, ... [/Problem Description] [Solution] def spending(): cost = 4 ...[/Solution]_ 

## **2.2.2 Knowledge Distillation from GPT-4** 

Based on the prompts, we build the KD dataset to train our data synthesis LLM. We randomly sample 5,336 math-related texts, and concatenate each one with a randomly selected prompt, to compose the input _i.e.,_ [ _pi_ ; _wi_ ]. Then, we feed it into GPT-4, and extract the synthetic math problem ˆ _qi_ and solution ˆ _si_ from its output using regular expressions, to compose the KD dataset _DKD_ = _{pi, wi,_ ˆ _qi,_ ˆ _si}[N] i_ =1 _[ini]_[.] Next, we utilize it to train the synthesis model, and the learning objective is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0004-09.png)


where _θsyn_ denotes the parameters of the data synthesis LLM. In this way, we can teach it to generate new math problem-solution pairs based on prompts and math-related texts by imitating GPT-4. 

## **2.3 Boosting Synthesis Model using High-Value Data** 

After initialization, we further improve the data synthesis LLM using high-value KD data. However, it would be costly if we first utilize GPT-4 to generate candidates and then select high-value ones. Therefore, we propose an efficient way that leverages the data synthesis LLM for generating candidates, and then selects valuable ones to feed into GPT-4. Specifically, we incorporate the gradient-based method [18] to estimate the influence of each synthetic instance for downstream math-related tasks, and select the top-ranking ones to update the KD dataset for retraining the data synthesis LLM. 

## **2.3.1 Gradient-based Data Value Estimation** 

According to the influence formulation [19], at a certain training step of a model parameterized by _θ_ , the influence of a training instance _z_ on another instance _z[′]_ can be estimated by computing the similarity between their produced gradients, denoted as: 

`Inf` ( _z, z[′]_ ) _∝ Sim_ ( _∇l_ ( _z, θ_ ) _, ∇l_ ( _z[′] , θ_ )) _._ (2) 

By using it, we can measure the value of each synthetic data by computing its gradient similarity with downstream math-related task data. Concretely, we first train a reference model using LoRA, then compute its gradients on LoRA parameters as the features, to help estimate the data value. 

4 

**Training Reference Model using LoRA.** Inspired by existing work [18], we train a LLM for mathematical problem solving using LoRA [20] as the reference model. As LoRA only requires to optimize the low-rank adapters in the LLM, we can efficiently train the reference model on limited computation resources, and reduce the number of trainable parameters for efficient computation of gradient similarity. Besides, to further reduce the training cost, we randomly select a subset of synthetic math problems generated by the data synthesis model, denoted as _Dlora_ = _{qi, si}[M] i_ =1 _[l]_[.] Then, we train the reference model to predict the solution based on the given problem, denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0005-01.png)


where _θlora_ denotes the parameters of LoRA, and _ml_ is the number of training data. 

**Computing Gradient Features.** After training the reference model, we compute the gradients of LoRA parameters as the feature of each synthetic instance. As their dimension is large, we follow existing work [21] that performs random projection to obtain the low-dimensional features as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0005-04.png)


where _z_ = _⟨pi, wi, qi, si⟩_ denotes a synthetic instance, Π _∈_ R _[d][′][×][d]_ is a projection matrix initialized by the Rademacher distribution, its entries are -1 or 1, _d[′]_ and _d_ are the dimensions before and after projection, respectively. According to the Johnson-Lindenstrauss Lemmas [22], this operation can nearly preserve the gradient distances, ensuring the usefulness of the low-dimensional features. 

**Estimating Data Value.** By using Eq. 4, we can compute the gradient features for synthetic instances. Then, we randomly sample _MD_ instances from the training sets of downstream mathrelated datasets _{zi[′][}][M] i_ =1 _[D]_[, where] _[ z] i[′]_[=] _[⟨][q]_[˜] _[i][,]_[ ˜] _[s][i][⟩]_[, and also compute their gradient features.][Next, we] estimate the value of each synthetic instance by computing the similarity between its gradient feature and the average feature of all the sampled downstream instances as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0005-07.png)


where `Cosine` ( _x, y_ ) computes the cosine similarity between the two vectors. In this way, the instance with higher data value would lead to a more positive influence on the downstream math-related tasks. 

## **2.3.2 Retraining Data Synthesis Model** 

Based on the estimated values, we can rank all the synthetic instances, and the top-ranking _Nadd_ ones can be regarded as the most valuable data _{⟨pi, wi, qi, si⟩}[N] i_ =1 _[add]_ for improving downstream math-related tasks. Thus, we utilize GPT-4 to regenerate the synthetic math problems based on their prompts and original math-related texts, to acquire corresponding more high-quality math problems and solutions. Then, we add the new GPT-4 synthetic data _{pi, wi,_ ˆ _qi,_ ˆ _si}[N] i_ =1 _[add]_ into the KD dataset, and the new data is capable of guiding the small LLM to generate more useful math problems for downstream tasks. Next, we retrain the data synthesis LLM with the updated KD dataset using Eq. 1. 

## **2.4 Pre-training JiuZhang3.0 using Synthetic Data** 

After training the data synthesis LLM, we construct the multi-source corpus containing rich mathrelated texts to cover more knowledge and scenarios. Then, we synthesize massive math problems based on it, which are used for pre-training JiuZhang3.0. 

**Constructing Multi-source Corpus.** We consider the following data types and select the corresponding open-source datasets to compose the math-related multi-source corpus. _• Webpages:_ we use the OpenWebText corpus [23], which consists of 6.3M math-related web documents extracted from Common Crawl. 

_• Books:_ we use the Mathpile-textbook dataset [24], including 4K educational textbooks, lecture notes and synthetic books. 

5 

_• Papers:_ we use the Mathpile-Arxiv dataset [24], and select the high-quality ones according to the estimated scores (0.6-0.9), which are released by AutoMathText [25]. 

_• QA Data:_ we select the StackExchange subset of the MMIQC dataset [26], which contains 1.2M processed real-world math question-answering pairs. 

_• Wikipedia:_ we use the Mathpile-Wikipedia dataset [24], consisting of 106K documents from math-related entries in Wikipedia. 

**Data Synthesis for Training JiuZhang3.0.** For each instance within the multi-source corpus, we randomly select a prompt from the prompt set and embed the text into the prompt to compose the input. Then, we feed inputs into the data synthesis model, to generate the math problems and solutions for composing the synthesis dataset _D_ = _{qi, si}[M] i_ =1[.][Here, we follow existing work [5, 13] to filter] out the instances with 10-grams overlap to both inputs and outputs from test sets of downstream evaluation tasks. We synthesize about 6M math problems (4.6B tokens) in total, which are used for pre-training JiuZhang3.0 to predict the solution based on the given problems. 

## **3 Experiments** 

## **3.1 Experimental Settings** 

For our JiuZhang3.0, we follow existing work [13] that train the 7B, 8B and 8 _×_ 7B versions based on Mistral-7B [27], LLaMA-3-8B [28], and Mixtral-8 _×_ 7B [29]. During training, we first pre-train it on our synthetic 4.6B math problem-solution pairs and then fine-tune it on the collected multiple open-source instruction datasets. We evaluate JiuZhang3.0 in two settings, _i.e.,_ natural language reasoning and tool manipulation. More details about the fine-tuning data, evaluation datasets, baseline methods, and implementation details are in Appendix D, E, F and G, respectively. 

## **3.2 Results and Analysis** 

**Natural Language Reasoning.** The results of this setting are shown in Table 1. First, the baseline methods trained on math-related data perform better than others. Among them, DeepSeekMath-7B is the best-performed base LLM, and DeepSeekMath-7B-RL also performs better than other baselines, since they have been pre-trained on 120B corpus containing rich math-related data. Besides, KPMathDSMath-7B and MAmmoTH2 also perform well. Concretely, KPMath-DSMath-7B is trained on nearly 1M synthetic math problems produced by GPT-4, and MAmmoTH2 also utilizes the GPT-4, Mixtral-8 _×_ 7B, and Qwen-72B to extract and refine the problems existing in the webpages. The acquired problems can greatly improve their performance in math problem solving. In our approach, we also utilize synthetic math problems to train our JiuZhang3.0-7B and 8B models. Differently, our used synthesis model is a much smaller 7B LLM, which has been trained by distilling the data synthesis capability from GPT-4. Thus, it can guarantee the quality of the synthetic data, and helps JiuZhang3.0 models perform the best across most of the dataset. The higher quality also reduces the data amount requirement for pre-training. Owing to our designed high-value data selection strategy, we can also reduce the times of invoking GPT-4 API for knowledge distillation. As noted in Figure 1, the total cost of our approach is nearly only 20% of the compared baselines, indicating its efficiency. 

The results of other datasets with different data formats or related to other fields are shown in Table 2. As the listed datasets focus on evaluating the different aspects, the performance of LLMs also differ a lot. For TabMWP, AQuA, and OCW-Math, our JiuZhang3.0-8B and JiuZhang3.0-8 _×_ 7B achieve the best performance. The three datasets require the understanding of table data, algebra, and undergraduate-level science knowledge respectively, which may have been covered in our synthetic math problems guided by the multi-source corpus. However, our JiuZhang3.0 models perform not well on MMLU-STEM. It indicates the shortcoming of our approach that our prompts and math-related texts might not well cover the knowledge from other subjects. 

**Tool Manipulation.** The results are shown in Table 3. We can see that our JiuZhang3.0-7B and 8B models outperform all the baseline methods by a large margin, indicating the effectiveness of our approach in this setting. The reason is that we synthesize massive math problems in this format, which can teach JiuZhang3.0 models to accurately utilize tools by generating programs. Besides, the mixed synthetic math problem from the natural language reasoning setting can also benefit the required 

6 

Table 1: Results on 6 datasets in the natural language reasoning setting. The best and second-best ones among LLMs with similar scales are marked in bold and underlined respectively. 

|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**Models**<br>**GSM8k**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|**MATH**<br>**SVAMP**<br>**ASDiv**<br>**MAWPS**<br>**CARP**<br>**Avg.**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ChatGPT<br>76.6<br>GPT-4<br>92.2||||||||38.2<br>65.4||||83.7<br>87.7<br>92.9<br>94.3||||96.9<br>41.3<br>70.7<br>96.6<br>53.6<br>82.5|||||||
|Qwen-1.5-110B<br>85.4<br>Qwen-1.5-72B<br>77.6<br>Mixtral-8×7B<br>74.4<br>Llemma-34B<br>60.2<br>Intern-Math-20B<br>64.9<br>ChatGLM-Math-32B<br>82.6<br>MAmmoTH2-8x7B-Plus<br>86.4||||||||49.4<br>39.4<br>29.0<br>24.6<br>27.4<br>40.6<br>47.0||||86.2<br>85.1<br>83.1<br>85.1<br>76.5<br>78.5<br>68.0<br>75.6<br>74.9<br>79.6<br>-<br>-<br>90.0<br>92.2||||94.3<br>**53.6**<br>75.7<br>95.8<br>53.0<br>72.3<br>93.9<br>38.8<br>65.2<br>89.8<br>36.5<br>59.1<br>94.4<br>42.3<br>63.9<br>-<br>-<br>-<br>**97.0**<br>45.8<br>76.4|||||||
|JiuZhang3.0-8x7B (Ours)<br>**89.8**||||||||**53.8**||||**90.2**<br>**93.1**||||96.7<br>52.3<br>**79.3**|||||||
|DeepSeek-7B<br>13.6<br>Mistral-7B<br>41.2<br>LLaMA-3-8B<br>54.5<br>Gemma-7B<br>54.1<br>Qwen-1.5-7B<br>60.5||||||||4.8<br>13.6<br>19.6<br>19.6<br>28.2||||40.8<br>52.1<br>64.7<br>68.5<br>68.5<br>72.8<br>69.7<br>74.2<br>64.9<br>74.9||||65.4<br>10.3<br>31.2<br>87.5<br>14.9<br>48.4<br>90.5<br>29.2<br>55.9<br>89.0<br>30.5<br>56.2<br>90.1<br>38.6<br>59.5|||||||
|Llemma-7B<br>39.2<br>InternLM-Math-7B<br>45.9<br>Rho-1-Math-7B<br>66.3<br>DeepSeekMath-7B<br>64.1||||||||18.4<br>15.8<br>31.0<br>34.2||||56.9<br>69.0<br>67.3<br>71.2<br>78.5<br>79.2<br>73.7<br>82.7||||82.7<br>31.8<br>49.7<br>88.3<br>28.0<br>52.8<br>94.0<br>36.7<br>64.3<br>92.7<br>44.4<br>65.3|||||||
|Mistral-7B-MMIQC<br>75.0<br>MetaMath-Mistral-7B<br>77.8<br>Abel-7B-002<br>80.4<br>WizardMath-7B-1.1<br>82.2<br>Math-Shepherd-Mistral-7B<br>84.3<br>KPMath-DSMath-7B<br>83.9<br>MAmmoTH2-7B-Plus<br>84.2<br>MAmmoTH2-8B-Plus<br>84.4<br>DeepSeekMath-7B-Instruct<br>82.3<br>DeepSeekMath-7B-RL<br>88.2||||||||34.2<br>29.6<br>29.6<br>32.8<br>34.4<br>48.8<br>46.2<br>41.2<br>45.8<br>50.2||||73.5<br>82.1<br>79.6<br>81.2<br>78.8<br>82.7<br>80.7<br>84.2<br>82.9<br>82.8<br>81.5<br>88.9<br>90.3<br>90.3<br>89.9<br>89.9<br>83.7<br>90.1<br>87.3<br>91.8||||90.1<br>36.5<br>65.2<br>93.7<br>30.5<br>65.4<br>93.5<br>33.2<br>66.4<br>93.8<br>31.9<br>67.6<br>92.5<br>32.9<br>68.3<br>94.8<br>-<br>-<br>95.8<br>44.3<br>75.2<br>97.1<br>44.8<br>74.6<br>95.7<br>45.8<br>73.9<br>95.5<br>**51.6**<br>77.4|||||||
|JiuZhang3.0-7B (Ours)<br>**88.6**<br>JiuZhang3.0-8B (Ours)<br>**88.6**||||||||**52.8**<br>51.0||||**90.4**<br>**92.6**<br>89.4<br>**92.6**||||**97.3**<br>51.0<br>**78.8**<br>97.1<br>50.9<br>78.3|||||||
||||||||||||||||||||||||
|60<br>65<br>70<br>75<br>GSM8k||||||||||||||||20%<br>40%<br>60%<br>80%<br>100%<br>Pre-training Data Proportion<br>DeepSeekMath-7B<br>Mistral-7B<br>LLaMA-3-8B|||||||
|||Mistr|al-7B|LL|aMA-3-|8B|30<br>35<br>40<br>45<br>MATH|||Mistr|al-7B|LL|aMA-3-|8B|70<br>75<br>80<br>85<br>90<br>ASDiv|||Mistr|al-7B|LL|aMA-3-|8B|
||||||||||||||||||||||||
||||||||||||||||||||||||
||~~D~~|~~eeSeek~~|~~Math-7~~|~~B~~||||||||||||||DeepSe|ekMath|-7B|||
||||||||||||~~D~~|~~eepSee~~|~~Math-7~~|~~B~~|||||||||
|||~~p~~|||||||||||||||||||||
||||||||||||||||||||||||
||||||||||||||||||||||||
||||||||||||||||||||||||
||20%<br>40%<br>60%<br>80%<br>100%<br>Pre-training Data Proportion|||||||20%<br>40%<br>60%<br>80%<br>100%<br>Pre-training Data Proportion|||||||||||||||



Figure 3: Performance changes with the increasing of the pre-training data proportion for our approach. We also show the best-performed base LLM DeepSeekMath-7B using dashed line. 

capabilities for this setting. Different from the natural language reasoning setting, JiuZhang3.0-8B (based on LLaMA-3-8B) performs better than the 7B version (based on Mistral-7B). The reason may be that LLaMA-3-8B owns stronger code synthesis and tool manipulation capability than Mistral-7B. 

## **3.3 Further Analysis** 

**Performance w.r.t. Pre-training Data Amount.** In this part, we study how the scaling of synthetic data amount affects the model performance. We train Mistral-7B and LLaMA-3-8B using varying ratios of our synthetic entire dataset, _i.e.,_ 20%, 40%, 60%, 80%, 100%, and report the performance 

7 

Table 2: Results on 5 other datasets with different data formats or related to interdisciplinary fields, and we abbreviate MMLU-STEM into M-STEM. The best and second-best methods among LLMs with similar scales are marked in bold and underlined respectively. 

|**Models**|**TabMWP**|**AQuA**|**SAT-Math**|**M-STEM**|**OCW-Math**|**Avg.**|
|---|---|---|---|---|---|---|
|ChatGPT|82.0|53.9|78.1|63.5|11.0|57.7|
|GPT-4|90.8|76.9|96.9|77.1|26.5|73.6|
|Qwen-1.5-110B|80.5|64.6|**87.5**|71.5|14.0|63.6|
|Qwen-1.5-72B|56.1|55.1|**87.5**|68.8|7.7|55.0|
|Mixtral-8×7B|67.3|48.0|65.6|62.3|8.8|50.4|
|Llemma-34B|57.1|46.1|71.9|54.3|11.8|48.2|
|Intern-Math-20B|63.4|44.1|65.6|62.3|7.0|48.5|
|MAmmoTH2-8x7B-Plus|62.7|55.9|81.2|**71.8**|18.8|58.1|
||||||||
|JiuZhang3.0-8x7B (Ours)|**84.7**|**65.4**|81.2|66.9|**23.5**|**64.3**|
|Mistral-7B|37.3|34.3|56.2|49.5|3.3|36.1|
|LLaMA-3-8B|67.5|46.5|56.2|54.4|7.7|46.5|
|Gemma-7B|60.9|42.9|71.9|57.7|4.8|47.6|
|Llemma-7B|49.2|37.8|62.5|45.8|7.7|40.6|
|Rho-1-Math-7B|55.5|49.2|75.0|54.9|6.2|48.2|
|DeepSeekMath-7B|69.8|51.6|84.4|56.1|17.6|55.9|
|DeepSeekMath-7B-Instruct|70.5|60.6|84.4|57.9|19.5|58.6|
|MAmmoTH2-7B-Plus|54.7|**62.2**|84.4|64.0|15.1|56.1|
|MAmmoTH2-8B-Plus|75.1|57.5|**87.5**|**65.7**|14.7|60.1|
||||||||
|JiuZhang3.0-7B (Ours)|74.8|59.4|81.2|53.6|20.2|57.8|
|JiuZhang3.0-8B (Ours)|**79.2**|**62.2**|84.4|60.4|**21.3**|**61.5**|



Table 3: Results on 6 mathematical reasoning datasets under the tool manipulation setting. The best and second-best methods are marked in bold and underlined respectively. 

|**Models**<br>**GSM8k**<br>**MATH**<br>**G-Hard**<br>**SVAMP**<br>**TabMWP**<br>**ASDiv**<br>**MAWPS**<br>**Avg.**|**Models**<br>**GSM8k**<br>**MATH**<br>**G-Hard**<br>**SVAMP**<br>**TabMWP**<br>**ASDiv**<br>**MAWPS**<br>**Avg.**|
|---|---|
|ChatGPT (PAL)<br>78.6<br>38.7<br>67.6<br>77.8<br>79.9<br>81.0<br>GPT-4 (PAL)<br>97.0<br>69.7<br>77.6<br>94.8<br>95.9<br>92.6|89.4<br>73.3<br>97.7<br>89.3|
|CodeLLama<br>34.0<br>16.6<br>33.6<br>59.0<br>61.4<br>79.6|-<br>-|
|MAmmoTH-7B-Mistral<br>75.0<br>40.0<br>-<br>-<br>-<br>-<br>MathCoder-7B-CL<br>67.8<br>30.2<br>-<br>70.7<br>-<br>-<br>ToRA-7B-Code<br>72.6<br>44.6<br>56.0<br>70.4<br>51.6<br>78.7<br>MARIO-OVM-7B<br>74.5<br>47.7<br>-<br>-<br>-<br>-<br>MMOS-CODE-7B<br>73.9<br>44.3<br>-<br>76.4<br>-<br>78.6<br>OpenMath-Mistral-7B<br>80.2<br>44.5<br>63.7<br>82.4<br>70.0<br>82.7<br>Rho-1-Math-7B-Code<br>81.3<br>51.8<br>63.1<br>80.8<br>70.1<br>85.5|-<br>-<br>-<br>91.3<br>66.5<br>-<br>-<br>-<br>-<br>95.4<br>74.1<br>94.5<br>75.3|
|JiuZhang3.0-7B (Ours)<br>82.4<br>53.0<br>**64.9**<br>**89.2**<br>75.6<br>**88.3**<br>JiuZhang3.0-8B (Ours)<br>**82.9**<br>**53.4**<br>64.4<br>**89.2**<br>**79.9**<br>87.5|96.6<br>78.6<br>**97.3**<br>**79.2**|



on GSM8k, MATH, and ASDiv under the natural language reasoning setting. For comparison, we also show the results of the best-performed base LLM, _i.e.,_ DeepSeekMath-Base-7B. 

As shown in Figure 3, with the increasing of the training data ratio, the performance of our model improves consistently. Based on Mistral-7B, it can outperform the best-performed baseline using only 80% or 60% of the pre-training data, indicating the high quality of our synthetic pre-training data. Based on LLaMA-3-8B, it can perform better than the baseline using 40% or even 20% data, and the performance is consistently better than using Mistral-7B. It demonstrates that LLaMA-3-8B can better adapt into our synthetic data. Besides, the performance of our model can surpass the baseline more on MATH, which is a very complex dataset consisting of competitive problems, exhibiting the superiority of our method for improving the advanced mathematical reasoning capability. 

**Ablation Study.** We conduct the ablation study to verify the effectiveness of key components in our proposed method. We test the following variations based on our approach, _i.e.,_ (1) _w/o Prompt Set_ : uses a simple prompt for guiding data synthesis instead of our crafted prompt set; (2) _w/o Mathrelated Texts_ : directly synthesizes the math problems without math related texts; (3) _w/o Boosting_ 

8 

Table 4: Ablation and variation studies in the natural language reasoning setting. We randomly sample 100k synthetic data and 50k instruction data for efficient test. 

|**Variation**|**Models**|**GSM8k**|**MATH**|**ASDiv**|**CARP**|
|---|---|---|---|---|---|
|-|Ours|78.6|**32.8**|**84.5**|36.2|
||w/o Prompt Set|76.9|27.8|81.4|34.5|
||w/o Math-related Texts|76.4|28.6|83.8|31.9|
|Ablation|w/o Boosting Retraining|77.9|31.6|83.8|34.7|
||w/o Value Estimation|78.8|31.0|83.4|34.3|
||w/o using GPT-4 for Boosting|79.0|28.4|82.9|**36.3**|
||- ChatGPT|77.0|26.6|83.0|34.3|
|Synthesis LLM|- Mixtral-8_×_7B<br>- DeepSeekMath-RL-7B|77.6<br>77.1|26.8<br>27.2|82.9<br>82.5|33.1<br>32.9|
||- LLaMA-3-8B-Instruct|75.7|26.2|81.5|31.4|
||- Random Sampling|78.8|31.0|83.4|34.3|
|Data Selection|- Perplexity<br>- Reward Model|77.5<br>78.0|30.8<br>31.6|83.1<br>84.3|**36.3**<br>34.8|
||- One-shot ICL|**79.2**|30.2|83.3|36.0|



_Retraining_ : uses the data synthesis model without retraining; (4) _w/o Value Estimation_ : ignores the estimated value but randomly samples the instances for boosting training; (5) _w/o using GPT-4 for Boosting_ : directly uses the high-value instance for boosting data synthesis model instead of using GPT-4. Limited by the computing resource, we conduct the ablation study under the natural language reasoning setting, and use 100k synthetic instances and randomly select 50k instructions from the instruction set. We report the results on GSM8K, MATH, ASDiv and CARP-en. 

As shown in Table 4, all the variations mostly underperform the original model, indicating the effectiveness of all the components. Among them, the variation w/o using GPT-4 for Boosting performs slightly better in GSM8k and CARP, but degrades a lot in MATH (32.8 _−→_ 27.8). A possible reason is that it can benefit from the selected high-value data. But without the help of GPT-4, it can not synthesize helpful complex math problems for the competitive problems within MATH dataset. 

**Variation Study for Data Synthesis LLMs.** To verify the effectiveness of our trained data synthesis LLM, we conduct the variation study using other existing LLMs for synthesizing the pre-training data. We select the following four LLMs, _i.e.,_ ChatGPT, Mixtral-8 _×_ 7B, DeepSeekMath-RL-7B, and LLaMA-3-8B-Instruct to replace our data synthesis LLM. We follow the efficient test setting in the ablation study, and report the results on GSM8K, MATH, ASDiv and CARP-en. 

As shown in Table 4, all the variations mostly perform worse than the original model. It demonstrates that existing LLMs without adapted training might not be suitable to directly synthesize the data for pre-training. Besides, the performance of all the variation degrades a lot in MATH, which consists of complex competitive problems. It indicates that these existing LLMs are hard to synthesize the data that is useful for improving the performance in solving complex math problems. 

**Variation Study for Data Selection Strategies.** To study the effectiveness of our gradient-based data selection strategy, we implement the following variations that replace it by other methods, _i.e.,_ (1) _Random Sampling_ : randomly samples the same number of instances; (2) _Perplexity_ : selects the instances with lowest perplexity evaluated by Mistral-7B; (3) _Reward Model_ : uses a well-trained reward model [30] for scoring; (4) _One-shot ICL_ : concatenates the synthetic math problem and solution with the downstream task data to construct the one-shot in-context learning (ICL) example, and computes the decrease of loss as the estimated value [31]. We follow the efficient test setting. 

As shown in Table 4, our original model mostly performs the best among all the variations, indicating the superiority of our gradient-based strategy. Whereas, the variation using one-hot ICL performs relatively better than others, and achieves the best performance on GSM8k. As the problems in GSM8k typically require more natural language reasoning steps, the ICL loss can well detect the instances with helpful context for solving these problems. However, it performs not well on MATH, where the math problems are complex and require using more math symbols and formulas. 

9 

## **4 Conclusion** 

In this paper, we proposed an efficient way to improve the mathematical reasoning of LLMs, where we trained a small LLM to synthesize sufficient high-quality math problems for pre-training. Concretely, we crafted a set of prompts that cover the knowledge and difficulty levels of human education stages, and selected the high-value math-related texts for downstream math-related tasks via the gradientbased strategy. Then, we fed them into GPT-4 to create the knowledge distillation dataset, which can better teach the data synthesis model to generate diverse and useful math problems. We utilized the synthetic data to pre-train JiuZhang3.0, and the whole process only required to invoke GPT-4 API 9.3k times and pre-train on 4.6B data. JiuZhang3.0 achieved state-of-the-art performance on several datasets under the natural language reasoning and tool manipulation settings, surpassing competitive LLMs that requires much larger cost on data synthesis or pre-training. 

## **References** 

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _NeurIPS_ , 2020. 

- [2] OpenAI. Gpt-4 technical report. _ArXiv_ , abs/2303.08774, 2023. 

- [3] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Z. Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jianyun Nie, and Ji rong Wen. A survey of large language models. _ArXiv_ , abs/2303.18223, 2023. 

- [4] Pan Lu, Liang Qiu, Wenhao Yu, Sean Welleck, and Kai-Wei Chang. A survey of deep learning for mathematical reasoning. _ArXiv_ , abs/2212.10535, 2022. 

- [5] Zhihong Shao, Peiyi Wang, Qihao Zhu, R. X. Xu, Jun-Mei Song, Mingchuan Zhang, Y. K. Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _ArXiv_ , abs/2402.03300, 2024. 

- [6] Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, and Weizhu Chen. Key-point-driven data synthesis with its enhancement on mathematical reasoning. _ArXiv_ , abs/2403.02333, 2024. 

- [7] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. _ArXiv_ , abs/2310.10631, 2023. 

- [8] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. _ArXiv_ , abs/2206.14858, 2022. 

- [9] Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu, Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin. Internlm-math: Open math large language models toward verifiable reasoning. _ArXiv_ , abs/2402.06332, 2024. 

- [10] Long Long Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zheng Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _ArXiv_ , abs/2309.12284, 2023. 

- [11] Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math. _ArXiv_ , abs/2402.14830, 2024. 

- [12] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. _ArXiv_ , abs/2309.17452, 2023. 

10 

- [13] Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. 2024. 

- [14] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, L. Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. Lima: Less is more for alignment. _ArXiv_ , abs/2305.11206, 2023. 

- [15] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _ArXiv_ , abs/2304.08485, 2023. 

- [16] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _ArXiv_ , abs/2201.11903, 2022. 

- [17] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. _ArXiv_ , abs/2305.11738, 2023. 

- [18] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. _ArXiv_ , abs/2402.04333, 2024. 

- [19] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. _Advances in Neural Information Processing Systems_ , 33:19920–19930, 2020. 

- [20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- [21] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander M ˛adry. Trak: attributing model behavior at scale. In _Proceedings of the 40th International Conference on Machine Learning_ , pages 27074–27113, 2023. 

- [22] William B. Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into hilbert space. _Contemporary mathematics_ , 26:189–206, 1984. 

- [23] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. _ArXiv_ , abs/2310.06786, 2023. 

- [24] Zengzhi Wang, Rui Xia, and Pengfei Liu. Generative ai for math: Part i - mathpile: A billion-token-scale pretraining corpus for math. _ArXiv_ , abs/2312.17120, 2023. 

- [25] Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. Autonomous data selection with language models for mathematical texts. 2024. 

- [26] Haoxiong Liu, Yifan Zhang, Yifan Luo, and Andrew Chi-Chih Yao. Augmenting math word problems via iterative question composing. _ArXiv_ , abs/2401.09003, 2024. 

- [27] Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L’elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. _ArXiv_ , abs/2310.06825, 2023. 

- [28] Meta. Introducing meta llama 3: The most capable openly available llm to date, April 2024. `https://ai.meta.com/blog/meta-llama-3/` . 

- [29] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, L’elio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. _ArXiv_ , abs/2401.04088, 2024. 

11 

- [30] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun. Advancing llm reasoning generalists with preference trees. _ArXiv_ , abs/2404.02078, 2024. 

- [31] Yunshui Li, Binyuan Hui, Xiaobo Xia, Jiaxi Yang, Min Yang, Lei Zhang, Shuzheng Si, Junhao Liu, Tongliang Liu, Fei Huang, and Yongbin Li. One shot learning as instruction data prospector for large language models. _ArXiv_ , abs/2312.10302, 2023. 

- [32] Anthropic. The claude 3 model family: Opus, sonnet, haiku. Technical report, Anthropic, March 2024. 

- [33] Google Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _ArXiv_ , abs/2403.05530, 2024. 

- [34] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _ArXiv_ , abs/2305.10601, 2023. 

- [35] K. Singhal, Shekoofeh Azizi, Tao Tu, Said Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Kumar Tanwani, Heather J. Cole-Lewis, Stephen J. Pfohl, P A Payne, Martin G. Seneviratne, Paul Gamble, Chris Kelly, Nathaneal Scharli, Aakanksha Chowdhery, P. A. Mansfield, Blaise Aguera y Arcas, Dale R. Webster, Greg S. Corrado, Yossi Matias, Katherine Hui-Ling Chou, Juraj Gottweis, Nenad Tomavsev, Yun Liu, Alvin Rajkomar, Joelle K. Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge. _Nature_ , 620:172 – 180, 2022. 

- [36] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony S. Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. _ArXiv_ , abs/2211.09085, 2022. 

- [37] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. _ArXiv_ , abs/2109.01652, 2021. 

- [38] Hyung Won Chung, Le Hou, S. Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Wei Yu, Vincent Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed Huai hsin Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models. _ArXiv_ , abs/2210.11416, 2022. 

- [39] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. _ArXiv_ , abs/2203.02155, 2022. 

- [40] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. _arXiv preprint arXiv:2211.10435_ , 2022. 

- [41] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In _Annual Meeting of the Association for Computational Linguistics_ , 2022. 

- [42] Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. _arXiv preprint arXiv:2205.12255_ , 2022. 

- [43] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _ArXiv_ , abs/2211.12588, 2022. 

12 

- [44] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. _arXiv preprint arXiv:2302.04761_ , 2023. 

- [45] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _ArXiv_ , abs/2210.03629, 2022. 

- [46] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- [47] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. 2023. 

- [48] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _ArXiv_ , abs/2309.05653, 2023. 

- [49] Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng. Common 7b language models already possess strong math capabilities. _ArXiv_ , abs/2403.04706, 2024. 

- [50] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. Cosmopedia, 2024. 

- [51] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Source code is all you need. _ArXiv_ , abs/2312.02120, 2023. 

- [52] Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, and Andrew M. Dai. Best practices and lessons learned on synthetic data for language models. _ArXiv_ , abs/2404.07503, 2024. 

- [53] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. _ArXiv_ , abs/2304.12244, 2023. 

- [54] Run-Ze Fan, Xuefeng Li, Haoyang Zou, Junlong Li, Shwai He, Ethan Chern, Jiewen Hu, and Pengfei Liu. Reformatted alignment. _ArXiv_ , abs/2402.12219, 2024. 

- [55] Saumya Gandhi, Ritu Gala, Vijay Viswanathan, Tongshuang Wu, and Graham Neubig. Better synthetic data by retrieving and transforming existing datasets. 2024. 

- [56] Ruida Wang, Wangchunshu Zhou, and Mrinmaya Sachan. Let’s synthesize step by step: Iterative dataset synthesis with large language models by extrapolating errors from small models. _ArXiv_ , abs/2310.13671, 2023. 

- [57] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-Guang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _ArXiv_ , abs/2308.09583, 2023. 

- [58] Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, and Hongsheng Li. Mathgenie: Generating synthetic data with question back-translation for enhancing mathematical reasoning of llms. _ArXiv_ , abs/2402.16352, 2024. 

- [59] Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, and Furu Wei. Synthetic data (almost) from scratch: Generalized instruction tuning for language models. _ArXiv_ , abs/2402.13064, 2024. 

- [60] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _ArXiv_ , abs/2103.03874, 2021. 

13 

- [61] Beichen Zhang, Kun Zhou, Xilin Wei, Wayne Xin Zhao, Jing Sha, Shijin Wang, and Ji rong Wen. Evaluating and improving tool-augmented computation-intensive math reasoning. _ArXiv_ , abs/2306.02408, 2023. 

- [62] Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y.Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. _ArXiv_ , abs/2312.08935, 2023. 

- [63] Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. _ArXiv_ , abs/2402.10176, 2024. 

- [64] Zui Chen, Yezeng Chen, Jiaqi Han, Zhijie Huang, Ji Qi, and Yi Zhou. An empirical study of data ability boundary in llms’ math reasoning. _ArXiv_ , abs/2403.00799, 2024. 

- [65] Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. _ArXiv_ , abs/2106.15772, 2020. 

- [66] Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In _North American Chapter of the Association for Computational Linguistics_ , 2016. 

- [67] Arkil Patel, S. Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In _North American Chapter of the Association for Computational Linguistics_ , 2021. 

- [68] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 158–167, 2017. 

- [69] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied Sanosi Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _ArXiv_ , abs/2304.06364, 2023. 

- [70] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _ArXiv_ , abs/2009.03300, 2020. 

- [71] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. _ArXiv_ , abs/2206.14858, 2022. 

- [72] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. _ArXiv_ , abs/2304.09842, 2023. 

- [73] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenhang Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, K. Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Yu Bowen, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xing Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _ArXiv_ , abs/2309.16609, 2023. 

- [74] DeepSeek-AI Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wen-Hui Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, 

14 

Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Jun-Mei Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Min Tang, Bing-Li Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Yu Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yi Xiong, Hanwei Xu, Ronald X Xu, Yanhong Xu, Dejian Yang, Yu mei You, Shuiping Yu, Xin yuan Yu, Bo Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghu Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. Deepseek llm: Scaling open-source language models with longtermism. _ArXiv_ , abs/2401.02954, 2024. 

- [75] Yifan Xu, Xiao Liu, Xinghan Liu, Zhenyu Hou, Yueyan Li, Xiaohan Zhang, Zihan Wang, Aohan Zeng, Zhengxiao Du, Wenyi Zhao, Jie Tang, and Yuxiao Dong. Chatglm-math: Improving math problem-solving in large language models with a self-critique pipeline. _ArXiv_ , abs/2404.02893, 2024. 

- [76] Gemma Team Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, L. Sifre, Morgane Riviere, Mihir Kale, J Christopher Love, Pouya Dehghani Tafti, L’eonard Hussenot, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Am’elie H’eliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Cl’ement Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Christian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, Justin Mao-Jones, Katherine Lee, Kathy Yu, Katie Millican, Lars Lowe Sjoesund, Lisa Lee, Lucas Dixon, Machel Reid, Maciej Mikula, Mateo Wirth, Michael Sharman, Nikolai Chinaev, Nithum Thain, Olivier Bachem, Oscar Chang, Oscar Wahltinez, Paige Bailey, Paul Michel, Petko Yotov, Pier Giuseppe Sessa, Rahma Chaabouni, Ramona Comanescu, Reena Jana, Rohan Anil, Ross McIlroy, Ruibo Liu, Ryan Mullins, Samuel L Smith, Sebastian Borgeaud, Sertan Girgin, Sholto Douglas, Shree Pandya, Siamak Shakeri, Soham De, Ted Klimenko, Tom Hennigan, Vladimir Feinberg, Wojciech Stokowiec, Yu hui Chen, Zafarali Ahmed, Zhitao Gong, Tris Brian Warkentin, Ludovic Peran, Minh Giang, Cl’ement Farabet, Oriol Vinyals, Jeffrey Dean, Koray Kavukcuoglu, Demis Hassabis, Zoubin Ghahramani, Douglas Eck, Joelle Barral, Fernando Pereira, Eli Collins, Armand Joulin, Noah Fiedel, Evan Senter, Alek Andreev, and Kathleen Kenealy. Gemma: Open models based on gemini research and technology. _ArXiv_ , abs/2403.08295, 2024. 

- [77] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ , 2023. 

- [78] Zheng-Wen Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. Rho-1: Not all tokens are what you need. _ArXiv_ , abs/2404.07965, 2024. 

- [79] Ethan Chern, Haoyang Zou, Xuefeng Li, Jiewen Hu, Kehua Feng, Junlong Li, and Pengfei Liu. Generative ai for math: Abel. `https://github.com/GAIR-NLP/abel` , 2023. 

- [80] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. _ArXiv_ , abs/2310.03731, 2023. 

- [81] Minpeng Liao, Wei Luo, Chengxi Li, Jing Wu, and Kai Fan. Mario: Math reasoning with code interpreter output - a reproducible pipeline. _ArXiv_ , abs/2401.08190, 2024. 

- [82] Xinzhe Ni, Yeyun Gong, Zhibin Gou, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Exploring the mystery of influential data for mathematical reasoning. _ArXiv_ , abs/2404.01067, 2024. 

- [83] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zhen Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chaochao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies. _ArXiv_ , abs/2404.06395, 2024. 

15 

## **A Related Work** 

**Large Language Models.** LLMs have demonstrated remarkable capabilities in a variety of NLP tasks, and commercial LLMs like ChatGPT, Claude, and Gemini [2, 32, 33], represent cutting-edge capabilities. Meanwhile, the performance of open-source models ( _e.g.,_ LLaMA-3, Mixtral) has also developed rapidly [28, 29]. To further improve the capability of LLMs on special tasks or domains, existing work mainly focuses on optimizing the following aspects: (1) prompt engineerings such as chain-of-thought and tree-of-thought [16, 34]; (2) continual pre-training on a domain-specific or task-specific corpus, improving the model to deal with downstream tasks [5, 7, 8, 35, 36]; (3) supervised fine-tuning, which involves fine-tuning the model on related instruction datasets, enhancing LLMs to follow special task instructions [37, 38]; (4) other strategies including RLHF [39], tool augmentation [40], decoding optimization [41] and _et al._ We aim to efficiently improve the capability of LLMs for mathematical reasoning by pre-training on synthetic data. 

**Mathematical Reasoning.** Despite the impressive progress, mathematical reasoning remains a weak aspect of LLMs. To enhance LLMs’ ability in mathematical reasoning, researchers have also proposed a surge of methods from the aspects of prompting, pre-training and fine-tuning. For prompting, the chain-of-thought (CoT) prompts have been widely used to guide LLMs for performing multi-step reasoning on complex math problems [16]. Based on it, following work utilizes tools [40, 42–45] and verifiers [9, 30, 46, 47], to further improve the accuracy of the mathematical reasoning process. For pre-training, existing work [5, 7, 9] collects a large-scale math-related corpus and continually pre-training open-source LLMs on it. In contrast, supervised fine-tuning methods focus on using relatively less high-quality data for training the LLM, which are typically mathrelated instructions [10, 12, 48]. Due to the complexity of mathematical reasoning, recent work also demands a large number of instructions, hence they rely on strong LLMs like GPT-4 to synthesize the data [6, 26, 49]. The pre-training and fine-tuning methods generally lead to large training and data annotation costs, respectively. Our work aims to train a small LLM specially for math problem synthesis, which can efficiently produce sufficient data for training. 

**Data Synthesis.** For complex tasks and scenarios ( _e.g.,_ mathematical reasoning), it is necessary to collect a substantial amount of data for training the LLM to enhance it. However, the available data may not be sufficient, hence researchers have explored using automatically synthetic data with consistent distribution to real data, to enrich the training corpus [50–56]. For data synthesis on mathematical reasoning tasks, existing work can be roughly categorized into the following two types, according to their based guided information. The first type of work starts with existing problems or math-related texts to synthesize similar problems or solutions [10, 26, 49, 57, 58]. The other type of work relies on available knowledge points, and devises special prompts to guide LLMs for synthesizing related problems with the solutions [6, 59]. As correctness is important, the two types of work generally design rules to check and remove wrong ones. In this work, based on the data synthesis model, we also construct the multi-source math corpora, and craft several prompts to guide it in producing diverse and useful math problems. 

## **B Limitation** 

First, although we train a strong yet small LLM that can synthesize high-quality math problems for training, its capabilities on synthesizing the data for other domains or tasks might be relatively weaker. The reason is that we only use the math-related data to train it. In future work, we will try to train a general-purpose model, to enable the data synthesis for other requirements. Second, in this work, we only focus on mathematical reasoning capability, and our trained JiuZhang3.0 is also mainly for solving math problems. Limited by the computation resource, we do not test its performance on other complex reasoning tasks, _e.g.,_ planning, commonsense reasoning. We will also conduct the corresponding experiments in the future. Third, the 4.6B pre-training data is still a large scale for training, and we do not perform data filter to control its quality. Future work should focus on reducing its scale by proposing better data filter strategy. Forth, we only utilize GPT-4 for knowledge distillation, but do not use other well-performed LLMs, _e.g.,_ Claude 3, GLM-4, and the latest GPT-4o. More experiments should be conducted on these LLMs to study the effect of the teacher LLMs. Fifth, for cost estimation, our reported results are all estimated according to our experience, without the 

16 

Table 5: The price of the potential service during the training procedure. 

|**Items**|**Details**|
|---|---|
|||
|**OpenAI API**|30.00 USD per 1M tokens for input, and 60.00 USD per 1M tokens for output.|
|||
|**AWS GPU Server**|40.96 USD per 8_×_A100 per hour.|



Table 6: The estimated cost of different LLMs using the official GPT-4 API and 8 nodes of 8 _×_ A100 GPU servers for training. 

|**Models**|**#API Calling (Token)**|**#Server Time (Hour)**<br>**Total Expenses**|**#Server Time (Hour)**<br>**Total Expenses**|
|---|---|---|---|
||Input<br>Output<br>Data Num.|Synthesizing<br>Training||
|||||
|**KPMath-DSMath-7B**|1090<br>218<br>865K|-<br>-|39,599 USD|
|||||
|**DeepSeekMath-7B-RL**|-<br>-<br>-|-<br>160|52,428 USD|
|||||
|**JiuZhang3.0-7B**|300<br>877<br>10K|14<br>10|8,480 USD|



re-implementation of these methods. We also appeal to report the true cost of training LLMs in existing work. 

## **C Cost Estimation** 

To estimate the expenses of previous work and our proposed JiuZhang3.0, we survey the price of potential service during the entire procedure, including calling the OpenAI API for GPT-4 and renting the AWS GPU server for LLMs training. The details of the price are presented in Table 5. For a fair comparison, we assume that GPT-4 is utilized to synthesize training data, and 8 nodes of 8 _×_ A100 GPU servers (64 GPUs in total) are leveraged for LLMs training. 

For our data synthesis process, the average length of the prompting (including selected math-related texts) is about 300 tokens, and the average length of problems and solutions is 877 tokens. Under the setting of 64 GPUs, we spend 4 hours selecting valuable data for natural language reasoning and tool manipulation in total, and 10 hours synthesizing the 4.6B pre-training corpus. We train the JiuZhang3.0-7B model for 10 hours, including both pre-training and fine-tuning. 

As the dataset and the construction process of KPMath are not publicly available, we adopt the average length of synthesis prompts, problems, and solutions in MetaMathQA for estimating the input and the output tokens, respectively. We estimate the training time on 120B tokens for a 7B model as 160 hours. Since the training details in the RLHF stage of DeepSeekMath-7B-RL are not publicly available, we do not count the fine-tuning cost for the KPMath model and the DeepSeekMath model. In this case, we estimate the expenses of the LLMs training process as follows, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0017-09.png)


The details and estimation of the expenses for different LLMs are present in Table 6. 

## **D Fine-tuning Data** 

After pre-training, we collect a set of open-source math-related instructions to fine-tune JiuZhang3.0. For natural language reasoning, we collect the training sets of MATH [60], GSM8k [46], CARP [61], and open-source synthetic datasets based on them, _i.e.,_ MetaMATH [10], MMIQC [26], MathShepherd (without problems from original MATH test sets) [62], Orca-MATH [11]. Besides, we also collect the positive examples from the PRM800k dataset (without problems from original MATH test sets) [47], and the TAL-SCQ5K dataset consisting of multi-choice questions[3] . Whereas, we observe 

> 3 `https://github.com/math-eval/TAL-SCQ5K` 

17 

that the varying data styles in the above datasets might cause the LLM outputs to be irregular. Thus, we utilize a unified prompt DeepSeekMath-7B-RL, to synthesize 700k solutions for the problems from the above datasets. 

For tool manipulation, we use the synthetic datasets, _i.e.,_ OpenMathInstruct-1 [63] and MMOS [64], consisting of a mixture of text reasoning and code blocks executed by a Python interpreter. 

## **E Evaluation Datasets** 

We test our JiuZhang3.0 and baseline methods in the following two settings for evaluating the mathematical reasoning capability. 

_• Natural Language Reasoning_ : we prompt LLMs to perform multi-step reasoning via natural language, and select the following publicly avaibable datasets: GSM8k [46] contains grade school math problems to test the basic arithmetic and reasoning ability. MATH [60] and CARP-en [61] consist of complex competition-level problems, and CARP-en is the English version of the original CARP dataset. ASDiv [65], MAWPS [66] and SVAMP [67] are grade-school math word problem (MWP) datasets, and SVAMP focuses on the robust reasoning ability. Besides, we also consider the following datasets with different data formats or related to other interdisciplinary field, _i.e.,_ TabMWP, AQuA, SAT-Math, MathQA, MMLU-STEM. AQuA [68], SAT-Math [69], MMLU-STEM [70] are composed of multiple-choice questions for human exams across math and other STEM disciplines. OCW-Math [71] is a challenging dataset containing undergraduate-level math and science problems. 

_• Tool Manipulation_ : we prompt LLMs to manipulate external tools via Python to solve the problems, and select GSM8k, MATH, GSM-Hard, SVAMP, TabMWP, ASDiv, and MAWPS for testing. GSMHard [40] replaces the numbers in the questions of GSM8K with larger numbers to increase the difficulty of calculation. TabMWP [72] is an open-domain MWP dataset containing tabular data. 

## **F Baseline Methods** 

We consider diverse types of baseline methods for comparison. 

- _Closed-source LLMs_ : ChatGPT and GPT-4 [2]; 

_• Larger LLMs (>20B)_ : Qwen-1.5-110B [73], Qwen-1.5-72B [73], Deepseek-LM-67B [74], Mixtral8×7B [29], Llemma-34B [7], Intern-Math-20B [9], MAmmoTH2-8 _×_ 7B-Plus [13], ChatGLM-Math32B [75]; 

_• Smaller LLMs (<10B)_ : DeepSeek-7B [74], Qwen-1.5-7B [73], Mistral-7B [27], LLaMA-3-8B [28], Gemma-7B [76], and CodeLLama [77]; 

- _LLMs pre-trained on Math Corpus (<10B)_ : Llemma-7B [7], InternLM-Math-7B [9], Rho-1-Math7B [78], DeepSeekMath-7B [5]; 

_• LLMs fine-tuned on Math Instructions (<10B)_ : MetaMath-Mistral-7B [10], WizardMath7B-1.1 [57], Abel-7B-002 [79], Mistral-7B-MMIQC [26], Math-Shepherd-Mistral-7B-RL [62], DeepSeekMath-7B-Instruct [5], DeepSeekMath-7B-RL [5], Llama-3-8B-Instruct [28], MAmmoTH2 [13], KPMath-DSMath-7B [6] 

_• LLMs fine-tuned on tool-augmented math instructions (<10B)_ : MAmmoTH-7B-Mistral [48], MathCoder-7B-CL [80], ToRA-7B-Code [12], MARIO-OVM-7B [81], MMOS-CODE-7B [64], OpenMath-Mistral-7B [63], Rho-1-Math-7B-Code [78]. 

Our evaluation framework and in-context examples follow the existing work [6, 12, 82]. For general and math domain base models, we adopt the few-shot prompting method. For fine-tuned models, we adopt the zero-shot prompting method for open-ended natural language reasoning and tool manipulation tasks, and the few-shot prompting method for multiple choice problems. We cite the performance results reported in existing work [6, 12, 48, 63, 64, 75, 78, 80, 81]. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0019-00.png)


**----- Start of picture text -----**<br>
74 79<br>72 77<br>70 MATH 75 MATH<br>GSM8K GSM8K<br>44<br>32<br>42<br>30<br>40<br>28<br>1:1 2:1 4:1 8:1 0k 2k 4k 6k 8k<br>NLC:TM Data Ratio High-value Data Number<br>MATH/GSM8k MATH/GSM8k<br>on on<br>Performance Performance<br>**----- End of picture text -----**<br>


Figure 4: Hyper-parameter tuning results of the pre-training data proportion and high-value data amount, under the tool manipulation and natural language reasoning settings, respectively. 

## **G Implementation Details** 

**Data Synthesis Models.** We train two data synthesis models for the natural language reasoning and tool manipulation settings, respectively. We first initialize them by training DeepSeekMath-7B-RL with 4k and 1.3k KD datasets, respectively. Then, we utilize the synthesis models to generate 100k problem-solution pairs for each one. During value estimation, we adopt the training set of GSM8k and MATH for natural language reasoning, and the 5k subset from a mixture of OpenMathInstruct and MMOS for tool manipulation, as the instances from downstream math-related tasks. We select 2k the most valuable math texts in each setting for GPT-4 to boost the quality, and then add them into the KD dataset. During training, following existing work [14], we adopt a cosine learning rate schedule with a 0% warm-up ratio and select a learning rate of 1e-5 for 5 epochs and 10 epochs for natural language reasoning and tool manipulation, respectively. 

**JiuZhang3.0.** Before training, we first filter out the instance from the synthetic dataset that have 10-grams overlap to the test set data, and also remove deduplicate data and the synthetic tool manipulation data containing unexecutable code. Then, we follow existing work that trains 7B, 8B, and 8 _×_ 7B versions based on Mistral-7B, LLaMA-3-8B, and Mixtral-8 _×_ 7B [13]. During training, we first pre-train it on our synthetic 4.6B math problem-solution pairs and then fine-tune it on the multi-source instruction set. We reuse the optimizer to initialize the fine-tuning stage and adopt the Warmup-Stable-Decay learning rate scheduler [83] with 3% warm-up ratio and 85% stable training ratio for 1 epoch in the whole training process. We set the maximum learning rate to 1e-5 and the minimum learning rate to 1e-6 with a total batch size of 512. To boost the training efficiency, we pack multiple instances in the same context window of the model and modify the attention to avoid mutual interference among difference instances. The maximum length of model is set to 2048. We train all models with BFloat16 numerical format, Flash Attention 2.0, DeepSpeed Stage 2 for 7B and 8B models, and Stage 3 for 8 _×_ 7B models. 

## **H Hyper-parameter Tuning** 

In this part, we conduct the experiments about tuning two important hyper-parameters, _i.e.,_ pretraining data proportion, and the number of high-value data. 

**Pre-train Data Proportion.** In the synthetic pre-training data proportion, we should determine the ratio between the data from the natural language reasoning and tool manipulation settings (NLC : TM). We set it to 1:1, 2:1, 4:1, and 8:1, and control the total data amount unchanged (200k) for comparison. We follow the efficient test setting in Section 3.3, and report the MATH and GSM8k results under the tool manipulation setting, as there is only slight performance changes on natural language reasoning setting. As shown in Figure 4, 2:1 is more suited and leads to better performance than other proportions. The reason may be that smaller or larger proportion for tool manipulation data would cause underfitting or overfitting on the tool manipulation data, affecting the corresponding capability. 

19 

**Number of Selected High-Value Data.** We also study the effect of changing the number of selected high-value data according to the ranking, as more data requires larger cost for invoking the GPT-4 API. We set it to 2k, 4k, 6k, and 8k, and do not change other settings for fair comparison. We follow the efficient test setting in Section 3.3. As shown in Figure 4, using the top 2k high-value data achieves the best performance, and more data even causes the performance degradation. It indicates that the data quality is more important than data amount for boosting the data synthesis capability, as using more data would inevitably involve low-quality data. 

## **I Case Study** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0020-02.png)


**----- Start of picture text -----**<br>
Good Case 1<br>Type: Natural Language Reasoning – Grade School<br>Math-Related Texts:<br>Question<br>The Huka Falls on the Waikato River is one of New Zealand’s most visited natural tourist attractions (see Figure 12.30). On average<br>the river has a flow rate of about 300,000 Ls. At the gorge, the river narrows to 20 m wide and averages 20 m deep. (a) What is the<br>average speed of the river in the gorge? (b) What is the average speed of the water in the river downstream of the falls when it widens<br>to 60 m and its depth increases to an average of 40 m?<br>Question Image<br>1. 0 . 75 ms<br>2. 0 . 13 ms<br>Solution Video# OpenStax College Physics Solution, Chapter 12, Problem 5 (Problems & Exercises) (1:14)<br>View sample solution## Calculator Screenshots<br>Video Transcript<br>This is College Physics Answers with Shaun Dychko. At the gorge where the Waikato River of New Zealand narrows to a width of 20<br>meters and a height of 20 meters, we2019re going to figure out what its speed is. [˘] And so the volume flow rate is equal to the cross<br>sectional area of the river multiply by its<br>Synthesis QA:<br>Question: The Waikato River flows through a gorge where it narrows down to a width of 20 meters and a depth of 20 meters. The<br>average flow rate of the river is 300,000 liters per second.<br>**Question:** What is the average speed of the river in this gorge?<br>Answer:<br>To find the average speed of the river in the gorge, we need to use the formula for the volume flow rate:<br>Volume Flow Rate = Cross-Sectional Area  ×  Speed<br>First, let2019s calculate the cross-sectional area of the river: [˘]<br>Cross-Sectional Area = Width  ×  Depth<br>Given:<br>- Width = 20 meters<br>- Depth = 20 meters<br>So,<br>Cross-Sectional Area = 20 meters  ×  20 meters = 400 square meters<br>Now, we know the volume flow rate and the cross-sectional area. The volume flow rate ( Q ) is 300,000 liters per second, which we<br>need to convert into cubic meters because the area is in square meters:<br>1 liter = 1 cubic decimeter = 0 . 001 cubic meters<br>So,<br>Q = 300 ,  000 liters per second = 300 ,  000  ×  0 . 001 cubic meters per second = 300 cubic meters per second<br>We can now use the formula for the volume flow rate to find the speed ( v ):<br>Q = Cross-Sectional Area  × v<br>Rearranging to solve for  v :<br>v = Q<br>Cross-Sectional Area<br>Substituting the values we have:<br>v = [300][ cubic meters per second] = 0 . 75 meters per second<br>400 square meters<br>So, the average speed of the river in the gorge is **0.75 meters per second**<br>**----- End of picture text -----**<br>


20 

## **Good Case 2** 

**Type:** Natural Language Reasoning – Middle **Math-Related Texts:** 

Successive parabolic interpolation 

Successive parabolic interpolation is a technique for finding the extremum (minimum or maximum) of a continuous unimodal function by successively fitting parabolas (polynomials of degree two) to a function of one variable at three unique points or, in general, a function of n variables at 1+n(n+3)2 points, and at each iteration replacing the öldest¨point with the extremum of the fitted parabola. Advantages 

Only function values are used, and when this method converges to an extremum, it does so with an order of convergence of approximately 1.325. The superlinear rate of convergence is superior to that of other methods with only linear convergence (such as line search). Moreover, not requiring the computation or approximation of function derivatives makes successive parabolic interpolation a popular alternative to other methods that do require them (such as gradient descent and Newton’s method). Disadvantages On the other hand, convergence (even 

## **Synthesis QA:** 

> `Question:` Alex and Jamie are working on a science project and need to find the maximum height of a rocket launched upward. They record the height of the rocket at three different times: - At _t_ = 0 seconds, the height is 0 meters. - At _t_ = 5 seconds, the height is 25 meters. - At _t_ = 10 seconds, the height is 40 meters. To find the maximum height, Alex and Jamie decide to use successive parabolic interpolation. They start by fitting a parabola to these three points and then finding the maximum point of this parabola. 1. Write the general form of a quadratic (parabolic) equation: _y_ = _ax_[2] + _bx_ + _c_ . 2. Use the three points to create a system of equations to find the coefficients _a_ , _b_ , and _c_ . 3. Find the vertex of the parabola to determine the maximum height. 

## `Answer:` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0021-08.png)


**----- Start of picture text -----**<br>
1. **General Form of a Quadratic Equation:**<br>The general form of a quadratic equation is:<br>y = ax [2] +  bx  +  c<br>2. **Creating a System of Equations:**<br>We have three points:<br>(0 ,  0) , (5 ,  25) , (10 ,  40)<br>Plugging these points into the equation  y = ax [2] +  bx  +  c :<br>- For (0 ,  0):<br>0 = a (0) [2] +  b (0) +  c = ⇒ c = 0<br>- For (5 ,  25):<br>25 = a (5) [2] +  b (5) + 0 = ⇒ 25 = 25 a  + 5 b (Equation 1)<br>- For (10 ,  40):<br>40 = a (10) [2] +  b (10) + 0 = ⇒ 40 = 100 a  + 10 b (Equation 2)<br>3. **Solving the System of Equations:**<br>Use Equation 1:<br>25 = 25 a  + 5 b (Equation 1)<br>Simplify:<br>5 = 5 a  +  b (Equation 3)<br>Use Equation 2:<br>40 = 100 a  + 10 b (Equation 2)<br>Simplify:<br>4 = 10 a  +  b (Equation 4)<br>Subtract Equation 4 from Equation 3:<br>(5 a  +  b )  − (10 a  +  b ) = 5  − 4 = ⇒− 5 a = 1 = ⇒ a = − [1]<br>5<br>Plug  a = − 5 [1] [back into Equation 3:]<br>5 = 5 − [1] +  b = ⇒ 5 = − 1 +  b = ⇒ b = 6<br>� 5 �<br>**----- End of picture text -----**<br>


21 

## **Good Case 2** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-01.png)


**----- Start of picture text -----**<br>
4. **Vertex of the Parabola:**<br>The vertex of a parabola  y = ax [2] +  bx  +  c  is given by:<br>x = − [b]<br>2 a<br>Here,  a = − 5 [1] [,] [ b] [=] [6][, and] [ c] [=] [0][:]<br>6<br>x = − = − [6] = 15<br>2 � − [1] 5 � − [2] 5<br>So, the maximum height occurs at t = 15 seconds. However, this t value exceeds the range of our data points. In practical<br>interpolation methods, this would indicate a correction point is needed beyond the given data. But for demonstration, let’s assume<br>t = 15 is within bounds (extrapolation for students):<br>Plug  t = 15 back into the original equation with adjusted coefficients:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-02.png)


**----- Start of picture text -----**<br>
y = − [1] [+ 6(15)] [=] [−] [1] [=] [−] [45 + 90] [=] [45]<br>5  [(15)][2] 5  [(225) + 90]<br>Thus, the maximum height is **45 meters**<br>**----- End of picture text -----**<br>


## **Good Case 3** 

**Type:** Natural Language Reasoning – High **Math-Related Texts:** 

Let _k_ be a positive integer. Find all polynomials with real coefficients which satisfy the equation _P_ ( _P_ ( _x_ )) = ( _P_ ( _x_ )) _[k]_ . Let _k_ be a positive integer. Find all polynomials with real coefficients which satisfy the equation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-06.png)


I simply don’t even know how to think about this problem. I’ve tried simple stuff just to get my head on the problem. For example for _P_ ( _x_ ) = _x[n]_ I have _P_ ( _P_ ( _x_ )) = ( _P_ ( _x_ )) _[n]_ , and I think that any polynomial _P_ ( _x_ ) = _x[n]_ + _x[n][−]_[1] + _· · ·_ + _c_ can’t be a solution as I would have _P_ ( _x_ ) = _P_ ( _x_ ) _q_ 1( _x_ ) + _R_ . After that I simply stare at the problem. Can you guys give some help ? Note: I would like to understand how to tackle these kind of problems, so I would be really grateful if you would explain the thinking process behind the solution. (This is optional, so feel free to give an answer as you prefer.) Thanks in advance. 

First we note that only the constant polynomials _P ≡_ 0 and _P ≡_ 1, as well als _P ≡−_ 1 for odd _k_ , satisfy the equation. (This holds true for _k >_ 1; for _k_ = 1 any constant polynomial will do.) Now if _P_ is not constant, then the range _Y_ = _{P_ ( _x_ ) : _x ∈_ R _}_ of _P_ is an infinite set. For every _y ∈ Y_ we have _P_ ( _y_ ) = _y[k]_ , which means that the polynomial _Q_ ( _x_ ) = _P_ ( _x_ ) _− x[k]_ is zero on _Y_ . Since _Y_ is infinite, this implies that _Q_ is the zero polynomial. In conclusion, _P_ ( _x_ ) = _x[k]_ . 

## **Synthesis QA:** 

`Question:` Consider the polynomial _P_ ( _x_ ) = _x_[2] . We want to find out how _P_ behaves when we apply it twice, _P_ ( _P_ ( _x_ )), and compare it with _P_ ( _x_ ) raised to some power, ( _P_ ( _x_ )) _[k]_ . Let’s explore the following scenario: 

Given _P_ ( _x_ ) = _x_[2] , find all values of _k_ such that the equation _P_ ( _P_ ( _x_ )) = ( _P_ ( _x_ )) _[k]_ holds for all real numbers _x_ . `Answer:` First, we evaluate _P_ ( _P_ ( _x_ )): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-12.png)


Next, we need to express _P_ ( _x_ ) raised to some power _k_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-14.png)


We want _P_ ( _P_ ( _x_ )) = ( _P_ ( _x_ )) _[k]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-16.png)


This equation holds for all real numbers _x_ if and only if the exponents are equal. Therefore, we set up the following equation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-18.png)


Solving for _k_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0022-20.png)


Thus, the equation _P_ ( _P_ ( _x_ )) = ( _P_ ( _x_ )) _[k]_ holds for _k_ = 2. Now let’s verify this: For _k_ = 2: 

Since both sides of the equation are equal: 

This equation holds true for all real numbers _x_ . Therefore, the only value of _k_ that makes _P_ ( _P_ ( _x_ )) = ( _P_ ( _x_ )) _[k]_ true for all real numbers _x_ is _k_ = 2. 

22 

## **Good Case 4** 

**Type:** Tool Manipulation – Grade School 

## **Math-Related Texts:** 

Refer to a unit in the Units package by indexing the name or symbol with the context, for example, bit[information] or bps[information]; or, if the context is indicated as the default, by using only the unit name or symbol, for example, bit or bps. The units of information are defined as follows. A crumb is defined as 2 bits. A nibble is defined as 4 bits. 

## **Synthesis QA:** 

> `Question:` In a digital communication system, information is transmitted in units called bits. A crumb is defined as 2 bits, and a nibble is defined as 4 bits. If a user sends a sequence of bits, and the sequence can be broken down into whole crumbs and nibbles, how many different ways can the user send exactly 18 bits? `Answer:` 

```
defcount_ways_to_send_bits(total_bits):
ways=0
forcrumbinrange(total_bits//2+1):
fornibbleinrange(total_bits//4+1):
```

```
ifcrumb*2+nibble*4==total_bits:
ways+=1
```

```
returnways
total_bits=18
```

```
ways_to_send=count_ways_to_send_bits(total_bits)
print(ways_to_send)
```

“‘output 5 “‘ There are 5 different ways the user can send exactly 18 bits. 

23 

## **Good Case 5** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0024-01.png)


**----- Start of picture text -----**<br>
Type: Tool Manipulation – Secondary School Competition<br>Math-Related Texts:<br>The generalization of this is 0: I = −log ( p ( x )) Mean and variance of continuous random variables<br>The mean is the sum of x * density probability function at that point. µ = � −∞∞ [xp] [(] [x] [)]<br>The variance is the sum of distance to the mean times the density probability function at that point. σ [2] = � −∞∞ [(] [x][ −] [µ] [)][2] [p] [(] [x] [)]<br>Esperance of a probability distribution is the value we could expect to have if we repeat some experiment a lot of times. It’s the sum<br>of the value of the events ponderated by the probability of the event to occur.<br>Synthesis QA:<br>Question: Consider a continuous random variable X with probability density function (pdf) given by: p ( x ) =<br>12 [e][−] [x] 2 &if  x ≥ 0 1. Calculate the mean ( µ ) of the distribution.<br>�0&if  x < 0<br>2. Calculate the variance ( σ [2] ) of the distribution.<br>Answer:<br>import sympy as sp<br># Define the variable and the pdf<br>x = sp.Symbol(’x’, real=True, positive=True)<br>pdf = (1/2) * sp.exp(-x/2)<br># Calculate the mean<br>mean = sp.integrate(x * pdf, (x, 0, sp.oo))<br># Calculate the second moment about the origin<br>second_moment = sp.integrate(x**2 * pdf, (x, 0, sp.oo))<br># Calculate the variance<br>variance = second_moment - mean**2<br>mean_value = mean.evalf()<br>variance_value = variance.evalf()<br>mean_value, variance_value<br>“‘output<br>(2.0, 4.0)<br>“‘<br>The mean ( µ ) of the distribution is 2 . 0 , and the variance ( σ [2] ) of the distribution is 4 . 0 .<br>**----- End of picture text -----**<br>


24 

## **Bad Case 1** 

**Type:** Natural Language Reasoning – AMC12 **Math-Related Texts:** Julia Robinson Mathematics Festival The Julia Robinson Mathematics Festival (JRMF) is an educational organization that sponsors locally organized mathematics festivals and online webinars targeting K–12 students. The events are designed to introduce students to mathematics in a collaborative and non-competitive forum.[1][2][3] AbbreviationJRMF FounderNancy Blachman Location • c/o SEE, 23564 Calabasas Road, Suite 201, Calabasas, CA 91302 Coordinates37 _[◦]_ 22’48.57"N 121 _[◦]_ 54’25.98"W Key people Nancy Blachman, Founder Daniel Kline, Executive Director Websitejrmf.org History 

In the 1970s, Saint Mary’s College of California produced a mathematics contest that was popular with secondary schools throughout the San Francisco Bay Area.[4] In 2005, Nancy Blachman attended an education forum sponsored by the Mathematical Sciences Research Institute (MSRI) and remembered how the Saint Mary’s contest had inspired her as a student.[5] Unfortunately, the contest no longer existed. Seeking to possibl 

## **Synthesis QA:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0025-04.png)


**----- Start of picture text -----**<br>
Question: At the Julia Robinson Mathematics Festival, students are challenged with a geometry puzzle. The problem is as follows:<br>A square  ABCD  with side length 10 is inscribed in a circle. Points  P and  Q  lie on the circle such that  P is on the same side of line<br>AD  as  C , and  Q  is on the same side of line  AB  as  C . If  AP = 16 and  AQ = 12, find the length of  P Q .<br>Answer: To solve this problem, we’ll start by finding the radius of the circle, using the fact that the square is inscribed in the circle.<br>Then, we’ll use properties of circles and the given lengths to determine the length of  P Q .<br>1. **Finding the radius of the circle:**<br>Since the square  ABCD  is inscribed in the circle, the diameter of the circle is equal to the diagonal of the square. The diagonal of a<br>square with side length  s  can be calculated using the Pythagorean theorem:<br>Diagonal = s √ 2<br>For the given square with side length 10, the diagonal is:<br>Diagonal = 10 √ 2<br>Thus, the radius of the circle, which is half the diagonal, is:<br>√ 2<br>Radius = [10] = 5 √ 2<br>2<br>**----- End of picture text -----**<br>


2. **Using the Power of a Point Theorem:** The Power of a Point Theorem states that for a point _P_ and a point _Q_ in a circle with center _O_ and radius _r_ , if _P_ and _Q_ are on the same side of a chord _CD_ passing through _O_ , then the product of the distances from _P_ and _Q_ to _CD_ is equal to the square of the distance from _O_ to _CD_ . 

In our case, _CD_ is the diameter of the circle, and _O_ is the center of the circle. Point _P_ is 16 units from _A_ on side _AD_ , and _Q_ is 12 units from _A_ on side _AB_ . _AP_ = 16 and _AQ_ = 12. Since _P_ and _Q_ are on the circle and the diameter passes through the center _O_ , the Power of a Point Theorem gives: 

_AP · AQ_ = _OP_[2] 16 _×_ 12 = _OP_[2] _OP_ = ~~_√_~~ 192 = 8 ~~_√_~~ 3 3. **Determining the length of _P Q_ :** Finally, we use the radius of the circle and the calculated power of point distance to find _OP_ to determine _P Q_ . Since _O_ is the midpoint of _P Q_ and _OP_ = _OQ_ = 8 ~~_√_~~ 3: _P Q_ = 2 _× OP_ = 2 _×_ 8 ~~_√_~~ 3 = 16 ~~_√_~~ 3 Thus, the length of _P Q_ is 16 ~~_√_~~ 3 units. This problem challenges students to use geometric principles (properties of circles and squares) and algebraic reasoning to solve for lengths in a geometric configuration, enhancing analytic and problem-solving skills suitable for the AMC 12 level. 

25 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0026-00.png)


**----- Start of picture text -----**<br>
Bad Case 2<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0026-01.png)


**----- Start of picture text -----**<br>
Type: Tool Manipulation – Secondary School Competition<br>Math-Related Texts:<br>0Nick D. Jan 5 ’17 at 14:48<br>• I guess it comes from a wifi code.Are you a chinese student? –0Renascence5. Jan 6 ’17 at 8:50<br>HINT:<br>Observe that the function is odd and the interval [ − 5 ,  5] is symmetric about 0. See more in this question:<br>Definite integral of an odd function is 0 (symmetric interval)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0026-02.png)


**----- Start of picture text -----**<br>
2<br>x [4] + 2 x [2] + 1 = x [2] + 1<br>� �<br>so we have an integral of the form<br>x [3]<br>� ( x [2] + 1) [2] [sin][2] [ xdx]<br>we can clearly see that<br>x [3]<br>f ( x ) = ( x [2] + 1) [2] [sin][2] [ x] [=] [⇒] [f] [(] [−][x] [)] [=] [−][f] [(] [x] [)]<br>i.e.<br>“‘<br>Guidelines for each section:<br>Synthesis QA:<br>Question: Consider the function  f ( x ) = ( x [2] x +1) [3] [2] [sin][2] [ x] [ over the interval][ [] [−] [5] [,] [ 5]][.] [We are interested in evaluating the definite]<br>integral of this function with respect to  x . Given the properties of the function  f ( x ), which is odd, and the symmetric interval about<br>the origin, determine the value of the integral.<br>Answer:<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0026-03.png)


**----- Start of picture text -----**<br>
so we have an integral of the form<br>we can clearly see that<br>i.e.<br>“‘<br>Guidelines for each section:<br>Synthesis QA:<br>**----- End of picture text -----**<br>


```
importsympyassp
```

```
#Definethevariable
```

```
x=sp.symbols(’x’)
```

```
#Definethefunction
```

```
f=(x**3)/(x**2+1)**2*sp.sin(x)**2
```

```
#Sincethefunctionisoddoverasymmetricinterval,theintegraliszero
integral_result=sp.integrate(f,(x,-5,5))
```

## `print(integral_result)` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models_images/nips_2024_JiuZhang3.0_Efficiently_Improving_Mathematical_Reasoning_by_Training_Small_Data_Synthesis_Models.pdf-0026-11.png)


**----- Start of picture text -----**<br>
“‘output<br>0<br>“‘<br>The answer is 0 .<br>**----- End of picture text -----**<br>


## **J Prompts** 

## **J.1 Prompts for Natural Language Reasoning Data Synthesis** 

## **Prompts for Grade School-Level Problem** 

## **Instruction** 

Create an age-appropriate math word problem for grade school students based on the provided math content. 

## **Math Content** 

[Math Text Placeholder] 

## **Guidelines** 

> [Problem]: Craft a concise math word problem suitable for grade school, focusing on basic arithmetic operations (addition, subtraction, multiplication, division), number sense, simple shapes, or introductory measurements. Use relatable, real-world scenarios appropriate for the age group. Ensure the problem is purely text-based and solvable without images. [Solution]: Provide a clear, step-by-step solution to the problem using simple language that a grade school student could understand. Explain the reasoning behind each step. 

26 

## **Prompts for Middle School-Level Problem** 

## **Instruction** 

Create an middle school level math problem and solution based on the provided math content excerpt. 

## **Math Content** 

[Math Text Placeholder] 

## **Guidelines** 

[Problem]: Create a self-contained problem for middle school student that directly incorporates a concept from the provided math content. Target a difficulty level appropriate for grades 6-8 (ages 11-14), assuming knowledge of arithmetic, pre-algebra, basic probability/statistics, and geometry. Ensure the problem is fully text-based and solvable without images. Use concepts typically covered by the end of 8th grade. 

[Solution]: Provide a detailed, step-by-step solution that demonstrates the mathematical reasoning from problem statement to conclusion, around 250-350 words long. Utilize LaTeX formatting for all mathematical expressions. Explain each step to reinforce the underlying math principles being applied. 

## **Prompts for High-Level Problem** 

## **Instruction** 

Inspired by the provided math content extract, create high school-level math problem that combines concepts from at least two math subjects. 

## **Math Content** 

[Math Text Placeholder] 

## **Guidelines** 

[Problem]: Draft a self-contained math problem for high school students based on the given math content. The problem should require knowledge from one of these subjects: Algebra I and II, Pre-Calculus, Calculus, Geometry, Trigonometry, Statistics and Probability. Ensure the problem is fully text-based and solvable without images. Use concepts typically covered by the end of 11th grade. [Solution]: Provide a detailed, step-by-step solution that demonstrates the mathematical reasoning from problem statement to conclusion, around 250-350 words long. Utilize LaTeX formatting for all mathematical expressions. Explain each step to reinforce the underlying math principles being applied. 

## **Prompts for High-Level Problem** 

## **Instruction** 

Inspired by the math content, create a college-level math problem. 

## **Math Content** 

[Math Text Placeholder] 

## **Guidelines** 

[Problem]: Draft a self-contained, college-level math problem inspired by the math content. It should be intellectually stimulating and designed for an audience familiar with advanced mathematics, such as Calculus, Linear Algebra, Abstract Algebra, etc. Ensure the problem includes all necessary information for solving it. Aim for a problem statement around 100-150 words. [Solution]: Provide a step-by-step solution to your problem, around 250-350 words long. The solution should clearly explain the reasoning, mathematical principles, and steps used. Call out any key theorems or properties being applied at each step. 

## **Prompts for AMC 8-Level Problem** 

## **Instruction** 

Inspired by the provided math content, craft a math problem suitable for the AMC 8 competition, engaging top grade students with a challenge in areas of arithmetic, algebra, counting, geometry, logic, number theory, or probability. Ensure the problem stimulates problem-solving techniques and heuristics within a text-based and solvable format, avoiding advanced subjects like calculus or physics. 

## **Math Content** 

[Math Text Placeholder] 

## **Guidelines** 

[Problem]: Design a compelling, self-contained math problem for AMC 8 contestants inspired by the math content, incorporating elements of basic math disciplines. The problem should be approachable through logical reasoning and fundamental problem-solving strategies. Ensure the problem is fully text-based, solvable without the aid of images, and has a difficulty level appropriate for the AMC 8 competition. 

[Solution]: Provide a detailed, step-by-step solution that is educational and tailored to the AMC 8 audience, around 250-350 words long. Explain the logic and reasoning behind each step thoroughly, using clear and age-appropriate language and terminology to facilitate understanding. Highlight the use of problem-solving techniques and heuristics employed in the solution. 

27 

## **Prompts for AMC 12-Level Problem** 

## **Instruction** 

Inspired by the provided math content, develop a math problem suitable for the AMC 12 competition, targeting high-school students with capabilities in advanced areas like algebra, geometry, trigonometry, counting, probability, and number theory. The problem should challenge students to utilize sophisticated problem-solving skills and mathematical reasoning. Focus on concepts from the given content, but adapt them into an original problem. Avoid delving into higher-level college mathematics topics beyond the AMC 12 scope. 

## **Math Content** 

- [Math Text Placeholder] 

## **Guidelines** 

[Problem]: Formulate a compelling and complex math problem for AMC 12 participants inspired by the provided math content. The problem should encourage students to employ advanced logical reasoning and problem-solving strategies related to the given concepts. Craft it to be solvable in the AMC 12 format and difficulty level. 

[Solution]: Present a comprehensive, step-by-step solution that both solves the problem and educates the student, around 250-350 words long. Clearly articulate the reasoning and methods used at each step, providing insight into the problem-solving process. Use language that challenges yet instructs high school students looking to improve their skills. Take care to format any equations properly using LaTeX or appropriate notation. 

## **Prompts for AIME-Level Problem** 

## **Instruction** 

Inspired by the math content, create a math problem appropriate for advanced high school mathematics competitions like the AIME. The problem should challenge students in core areas tested on the AIME such as algebra, geometry, combinatorics, number theory and probability. Encourage creative problem-solving and deep mathematical thinking using pre-calculus level techniques. Avoid delving into college-level topics like abstract/linear algebra, topology, multivariable calculus, or advanced physics. Any physics concepts used should be kept at a basic mechanics level. 

## **Math Content** 

- [Math Text Placeholder] 

## **Guidelines** 

- [Problem]: Design an insightful and challenging problem suitable for AIME participants inspired by the math content. Focus on core AIME topics like algebra, geometry, combinatorics, number theory and probability. The problem should be solvable using clever applications of high school math and mathematical reasoning, without requiring knowledge of advanced college-level mathematics or physics. Aim for a difficulty level on par with actual AIME questions. [Solution]: Provide a detailed, step-by-step solution that would be enlightening to an advanced high school student, around 250-350 words long. Explain the reasoning process and techniques applied in each step. Use language that is mathematically sophisticated yet understandable to a well-prepared AIME participant. Justify any key insights or creative leaps in the solution process. 

## **J.2 Prompts for Tool Manipulation Data Synthesis** 

## **Prompts for Tool Manipulation where Grade School or Secondary School Competition Level is Determined by Example** 

## **Instruction** 

Please gain inspiration from the following random math content to create a high-quality math problem and solve it with Python code. Present your output in two distinct sections: [Problem Description] and [Solution]. 

## **Math Content** 

- [Math Text Placeholder] 

## **Guidelines** 

1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. 

2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided using Python code. Summarize in natural language in the end and put the answer in /boxed. **Example** 

- [Problem Description] 

- [Example Question Placeholder] 

- [/Problem Description] 

- [Solution] 

- [Example Python Solution Placeholder] [/Solution] 

28 

