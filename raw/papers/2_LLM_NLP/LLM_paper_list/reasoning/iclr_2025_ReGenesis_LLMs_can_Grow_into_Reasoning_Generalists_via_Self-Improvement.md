# - REGENESIS: LLMS CAN GROW INTO REASONING GEN ERALISTS VIA SELF-IMPROVEMENT 

**Xiangyu Peng Congying Xia** _[∗]_ **Xinyi Yang** _[∗]_ **Caiming Xiong Chien-Sheng Wu Chen Xing** _[†]_ Salesforce AI Research _{_ `becky.peng, c.xia, x.yang, cxiong, wu.json, cxing` _}_ `@salesforce.com` 

## ABSTRACT 

Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement ( `ReGenesis` ), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, `ReGenesis` self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that `ReGenesis` achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4 _._ 6% after post training, `ReGenesis` delivers around 6 _._ 1% performance improvement. We also conduct in-depth analysis of our framework and show `ReGenesis` is effective across various LLMs and design choices. 

## 1 INTRODUCTION 

Recent research has demonstrated that post-training with explicit intermediate reasoning trajectories can improve the performance of large language models (LLMs) (Yuan et al., 2024; Mukherjee et al., 2023; Lightman et al., 2023) across a wide range of complicated reasoning tasks, such as mathematical reasoning, commonsense reasoning, etc. However, their acquisition of high-quality reasoning trajectory data in the post-training phase demands meticulous supervision for each reasoning step, either from humans or superior models. Therefore, although high-quality explicit reasoning paths can help post training, they are either expensive and time-consuming due to the additional substantial human supervision, or constrained by the license of superior models. It motivates us to explore: how far an LLM can improve its reasoning by self-generating reasoning paths as training data, without any additional supervision beyond final answers? 

Existing approaches towards this direction, STaR(Zelikman et al., 2022) and LMSI(Huang et al., 2023), prompt the given LLM to generate a chain-of-thought (CoT) (Wei et al., 2022) reasoning steps and filter them with final ground truth labels or self consistency (Wang et al., 2022). They also use a small fixed set of human-designed reasoning examples and incorporate them directly into models’ prompt for in-context learning. After self-generating reasoning paths for given data sets, these approaches use these data sets with reasoning paths to further fine-tune the given model. These self-improvement approaches can improve the model’s in-domain reasoning capabilities without the need of additional external supervision. However, through our extensive experiments, we have found 

> _∗_ Equal contribution. 

> _†_ This author completed this work during her time at Salesforce AI Research. 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement_images/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement.pdf-0002-00.png)


**----- Start of picture text -----**<br>
      Question : Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by<br>your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?<br>Ground Truth : $990 Ground Truth : $990<br>Candidate Reasoning Solution:<br>1. Tina's regular hourly wage is $18.00 per hour.<br>2. Tina works 10 hours every day, which exceeds the<br>8-hour limit, so she works 2 shifts (10 hours / 8 hours<br>Reasoning Structure :  per shift).<br>1. Understand the hourly wage  3. For each day, Tina works 8 regular hours. That's 8 *<br>and overtime rules.  $18.00 = $144.00 per day.<br>General Reasoning Guideline  Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.ply them one by one to the problem to see if any progress can be made.ply them one by one to the problem to see if any progress can be made. Task Specific Reasoning Guideline  Make a list of ideas for understanding wageand apply them one by one to  calculate wages.  and  accurately  overtime rule hourly  2. Determine the number of shifts. 3. Calculate regular hours per day. 4. Calculate overtime hours per day. 5. Calculate total daily earnings. 6. Calculate total earnings for 5 days. 4. For each day, she works an additional 2 overtime hours. That's 2 × ($18.00 + $9.00) = $54.00 per day. 5. Adding regular earnings and overtime earnings, we get $144.00 (regular) + $54.00 (overtime) = $198.00 per day. 6. To calculate total earnings for 5 days, we multiply the daily earnings by the number of days: $198.00 (daily earnings) × $5 (days) = $990.00. So, Tina makes  $990.00  for working 10 hours every day for 5 days.<br>Filter by ground truth label or majority vote<br>Reasoning Solution:  1. Identify the base hourly wage: Tina makes $18.00 an hour.. 2. Determine the number<br>Final ReasoningSolution :  of hours Tina works per day: She works 10 hours every day.\n\n3. Identify the normal working hours per Normal working hours are 8 hours per day (as stated in the problem).\n\n4. Identify thand ni Reasoning Solution:  exceeds the 8-hour limit, so she works 2 shifts (10 hours / 8 hours per shift).3. For ach day, Tina works 8 regular hours. That's 8 * $18.00 = $144.00 per day. 4. For each day, she works an additional 2 overtime hours. That's 2 × ($18 Reasoning Solution:  Calculate the overtime hours: 10 * 5 hours - 40 hours = 10 hours. 3. Calculate the overtime earnings for the 10 overtime hours: 10 hours * $27.00 = $270. 4. Add the base wage earnings and overtime earnings to find Tina's total earnings: $720 (base wage earnings) + $270 (overtime earnings) = $990. Therefore, Tina makes $990 for  .00 + $9.00) = $54.00 per day. 5. Adding regular earnings and overtime earnings, we get $144.001. Tina's regular hourly wage is $18.00 per hour. 2. Tina works 10 hours every 1.Calculate the base wage earnings for the first 8 * 5 hours: 40 hours * $18.00 = $720. 2.  e  overtime hours per day  day , which :<br>working 10 hours every day for 5 days.<br>Finetune(Onetime)<br>**----- End of picture text -----**<br>


Figure 1: Overview of `ReGenesis` . We first create a set of reasoning guidelines , and use language model **M** to tailor them more specifically to the task . The resulting adapted task-specific reasoning ~~gu~~ idelines are fed into again to develop a set of detailed reasoning struct ~~ure~~ s . Then each reasoning structure is used ~~to~~ generate a potential reasoning path and predicted answer ( _ri,j[path] ,_ ˆ _ai,j_ ). We filter out incorrect reasoning paths using either ground truth or majority vote answers. For instructions that lack correct reasoning paths after filtering, we provide the ground truth as a hint and repeat the reasoning path generation. Finally, the correct reasoning paths, along with the instructions and answers, are used to fine-tune , enhancing its reasoning capability. 

that they struggle to improve the given LLMs on out-of-domain (OOD) tasks that are different from the fine-tuning data that they generate reasoning paths and further train the model on. Specifically, we evaluate them on six OOD tasks including math, logic, common sense, and natural language inference. Our results indicate that further post-training using data produced by these methods leads to a 4 _._ 6% decline on the given LLM’s performance on such OOD tasks. In other words, existing methods don’t generalize well on OOD tasks and cannot make given LLMs reasoning generalists. 

To fill this gap, in this paper we explore how to self-synthesize reasoning paths as post training data to make LLMs generalize well across various OOD reasoning tasks. We hypothesize that existing methods fail to generalize to OOD tasks because their synthesized reasoning paths are overly tailored to specific training tasks and lack diversity. These methods predominantly utilize a chain-of-thought approach, which leads to uniform and narrow reasoning patterns, rather than employing a range of cognitive strategies that foster diverse thinking steps. To resolve these shortcomings, we introduce `ReGenesis` , a reasoning generalist framework designed to generate general and diverse reasoning paths guided by various reasoning principles. `ReGenesis` adopts a methodological approach that transitions from abstract and general reasoning guidelines to concrete and specific reasoning solutions, as shown in Figure 1. More specifically, we first create multiple general task-agnostic _reasoning guidelines_ in the beginning of the reasoning path generation. Then, given each general reasoning guideline, we prompt the LLM to augment such general guideline into a _task-specific reasoning guideline_ . Note that in this step we fill in task-specific details into general guidelines while still maintaining their task-agnostic reasoning structure, as shown in Figure 1. Then we feed each taskspecific reasoning guideline into the given LLM to generate a _reasoning structure_ without directly solving the tasks. Each reasoning structure is subsequently used to create one _candidate reasoning path_ . Finally, all the candidate the reasoning paths are filtered using ground truth answers and, along with the instructions and answers, are used to fine-tune the same LLM, enhancing its reasoning capabilities. 

To demonstrate the effectiveness of our proposed framework, we conducted evaluations of `ReGenesis` across various in-domain and out-of-domain (OOD) settings. In all scenarios tested, `ReGenesis` consistently outperformed existing methods, showcasing its superior performance in both in-domain and OOD contexts. For in-domain settings, across five in-domain tasks involving math, logic, and 

2 

common sense, `ReGenesis` outperforms all 5 baselines by 7 _._ 1% to 18 _._ 9%. Moreover, we evaluate `ReGenesis` against baseline methods on 6 OOD tasks, which include math, logic, common sense, and natural language inference. Our results indicate that `ReGenesis` generalize well on OOD reasoning tasks, achieving a 6 _._ 1% improvement compared to the model without fine-tuning. In contrast, existing methods such as STaR, exhibited an average performance decrease of approximately 4 _._ 6% on OOD tasks when compared to the given model’s respective performances prior to fine-tuning. These findings underscore our method’s effectiveness in not only enhancing the reasoning abilities of LLMs but also in transforming these LLMs into reasoning generalists through self-improvement. 

We summarize our contributions as follows: 

- We propose `ReGenesis` that enables LLMs to self-improve their general reasoning capabilities in Section 3. `ReGenesis` uses a given LLM to self-synthesize reasoning paths for post-training by progressing from abstract to concrete and enables the given LLM to generalize well to OOD reasoning tasks. 

- We evaluate our framework in Section 4 on five in-domain tasks and six OOD tasks to show that `ReGenesis` significantly improves the reasoning capabilities of the LLM, in both in-domain and OOD settings. 

- We perform a comprehensive analysis of our framework, evaluating the effects of various models, design choices, filtering methods, and guideline preferences across different language models. We summarize our findings in Section 5. 

## 2 RELATED WORK 

**Learning from reasoning paths with supervision.** Recent research shows that employing explicit reasoning paths with extensive supervision from either humans or superior models, can lead to better reasoning capabilities after post-training a LLM (Lightman et al., 2023; Ye et al., 2023; Yuan et al., 2024; Mukherjee et al., 2023; Achiam et al., 2023; Gou et al., 2023; Mitra et al., 2023; Qu et al., 2024; Sun et al., 2024). For example, Mukherjee et al. (2023) and Mitra et al. (2023) illustrate how smaller models can benefit from detailed explanations and step-by-step reasoning processes provided by GPT-4 (Achiam et al., 2023). TRIPOST (Yu et al., 2023) further illustrates this by effectively training smaller models to learn from mistakes and enhance their performance on mathematical and reasoning tasks with guidance from a more powerful language model like Codex (Chen et al., 2021). Yuan et al. (2024) also demonstrates that incorporating high-quality multi-turn reasoning generated by a superior model during training can improve the overall capabilities of LLMs. Moreover, process supervision methods (Lightman et al., 2023; Luo et al., 2023; Wang et al., 2024) uses extensive finer-grained human annotations on step-by-step reasoning paths as post-training data. These approaches all rely on more advanced teacher models or human experts for extensive supervision to obtain high-quality, explicit intermediate reasoning paths. When more advanced models or human supervision are not available due to license and cost issues, we are curious to see how well LLMs can leveraging itself on generating reasoning paths to enhance its own reasoning capabilities. 

**Learning from self-synthesized reasoning paths.** Contrary to the large number of methods above which rely on external supervision to create reasoning paths for post-training, there are limited existing work that employs self-synthesized reasoning paths from a LLM to improve its own reasoning capacity. Self-Taught Reasoner (STaR) (Zelikman et al., 2022) improves the reasoning capabilities of language models using self-generated, ground-truth-filtered reasoning paths which are prompted with a few human-written CoT examples. Huang et al. (2023) trains language models with self-generated reasoning paths filtered by self-consistent answers, also aided by human-written examples. However, we observed that they fail to improve the given LLMs on OOD tasks through our experiments while our framework improves model performance across both in domain and OOD reasoning tasks. 

**Diverse prompting methods.** Recent research has greatly improved the reasoning capability of LLMs through more effective prompting techniques (Zhou et al., 2022; Kojima et al., 2022; Zheng et al., 2023). For example, Wei et al. (2022) introduced CoT prompting, where the model generates a series of intermediate reasoning steps, making it easier to solve complex, multi-step reasoning tasks. Wang et al. (2022) enhanced this approach by sampling multiple different reasoning paths and then using majority voting to identify the most consistent answers, referred as Self-Consistency. Zhou et al. (2024) further developed these techniques by enabling LLMs to select and combine simple reasoning modules—like critical thinking and step-by-step analysis—into a clear reasoning structure during 

3 

decoding. These prompting methods, particularly those described by Zhou et al. (2024), have inspired us to develop a reasoning generalist that can generate a more diverse and general self-synthesized reasoning paths for a given task. 

## 3 `ReGenesis` : REASONING GENERALISTS VIA SELF-IMPROVEMENT 

Existing methods have demonstrated that we can self-improve LLM’s reasoning performance in in-domain settings by self-synthesizing reasoning paths, as also verified by our experiments in Section 4.2. However, when applying existing methods in out-of-domain (OOD) settings, we have found that they lead to non-neglectable performance drop, shown in Section 4.3. We hypothesize that the main reason for previous methods being unable to generalize to OOD tasks is that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance and diversity. 

Therefore, to verify this hypothesis and to provide a self-improving method that can help LLMs to generalize well on diverse reasoning tasks after training LLMs on limited reasoning tasks, we propose `ReGenesis` . `ReGenesis` adopts a strategy to self-synthesize reasoning paths by progressing from abstract to concrete, from general reasoning structures to task-specific reasoning paths. Building on this idea, `ReGenesis` can automatically generate and integrate reasoning paths into post-training datasets that originally contain only instructions and ground-truth answers. Figure 1 shows how we create self-synthesized reasoning paths for a task. We start with a trained LLM **M** and a training dataset **D**[train] = _{qi, ai}i[|]_ **[D]** =1 _[|]_[, where] _[ q][i]_[ is a text instruction and] _[ a][i]_[ is the corresponding ground-truth] solution. All system prompts we used during this process are shown in Appendix A.3. The goal of `ReGenesis` is to use **M** to generate reasoning paths as post-training data to fine-tune **M** , to further improve **M** ’s performance on both in-domain and OOD reasoning tasks. 

## 3.1 SELF-SYNTHESIZED REASONING PATHS 

**Guidance Adaption.** We first create a set of the seed general reasoning guidelines **P**[raw] = _{p_[raw] _j }[|] j_ **[P]** =1 _[|]_ , which provide general descriptions on problem-solving strategies which aim at activating the language model’s reasoning capabilities. These prompts are designed to be broad and applicable to a wide range of different tasks. For example, “How can I break down this problem into smaller, more manageable parts?”. The full list of seed guidelines can be found in Appendix A.3.1. For each instruction _qi_ , we use **M** to edit and expand each seed guideline, tailoring it more specifically to the task, thereby generating an adapted task-specific reasoning guidelines **P**[adp] _i,j_[=] _[ {][p]_[adp] _i,j[}][|] j_ **[P]** =1 _[|]_[.] 

As demonstrated in Figure 1, “ _Make a list of ideas for solving this problem_ ” in one of the seed guidelines can be tailored for a salary calculation task to: “ _Make a list of ideas for understanding hourly wage and overtime rules, and systematically apply each idea to accurately calculate wages_ .” In this step, for each instruction _qi_ , we generate an adapted guidance for every corresponding seed guidance, resulting in a total of _|_ **P** _|_ adapted prompts per instruction. 

**Reasoning Structure Generation.** We prompt the same language model **M** to develop a set of detailed reasoning structures **R**[s] _i_[=] _[{][r] i,j_[s] _[}][|] j_ **[P]** =1 _[|]_[for][each][instruction,] _[q][i]_[,][utilizing][the][adapted] guidance **P**[adp] _i_ accordingly. Specifically, for each adapted prompt _p_[adp] _i,j_[, the model] **[ M]**[ generates one] corresponding reasoning structure _ri,j_[s][.][Instead of developing a detailed reasoning path directly, the] reasoning structures convert the adapted prompts into a more detailed framework without solving the problem. Figure 1 illustrates how the adapted prompt for the salary calculation task can be transformed into a practical reasoning structure, without directly solving the entire task. For example, “ _1) Understand the hourly wage and overtime rules, 2) Determine the number of shifts, ..._ ” This reasoning structure serves as a thinking principle, bridging the gap between the general seed prompts and the detailed reasoning paths necessary to complete the task. 

**Reasoning Paths Generation.** Each reasoning structure _ri,j_[s][is used to generate a potential reasoning] path and predicted answer ( _ri,j[path] ,_ ˆ _ai,j_ ) for a given instruction _qi_ . The reasoning path addresses instruction _qi_ step by step with the reasoning structure, thereby providing robust data for subsequent fine-tuning. For example, as shown in Figure 1, the reasoning structure for _calculating salary_ is developed into a detailed reasoning path, “ _1) Tina’s regular hourly wage is $18.00 per hour, 2) Tina_ 

4 

_works 10 hours every day, ..._ ”, with the predicted answer _$990.00_ . After this process, we obtain a total of _|_ **P** _|_ reasoning solutions per instruction. The ablation study is presented in Section 5.1. 

## 3.2 FILTERING REASONING PATHS 

**Filtering with the Ground-truth.** Same as previous methods such as Zelikman et al. (2022) that generate reasoning paths given prompts and ground-truth answers of existing data sets, we use the given ground-truth answers to filter our self-synthesized reasoning paths. Due to the limitations in the reasoning capabilities of language models, there is no guarantee that the predicted solution _a_ ˆ _i,j_ using the reasoning path is the same as the given answer _ai_ . Therefore, we use _ai_ to filter out those reasoning solutions that are incorrect, assuming that only reasoning paths leading to the correct answer are valid. Additionally, we explore self-consistency as an alternative method for selecting valid reasoning paths in Section 5.2. 

**Adding Ground-truth as a Hint.** For instruction _qi_ that does not have any correct reasoning solutions after filtering, we include the ground-truth _ai_ as a hint to generate adapted guidance and reasoning structures (system prompts can be found in Appendix A.3.3). Unlike Zelikman et al. (2022) which provides hints directly in the final reasoning path generation process, we only add hints during “Guidance Adaption” step and “Reasoning Structure Generation” step. By not having ground-truth answers in the final reasoning generation stage, we reduce the risk of the final reasoning solutions producing a correct answer but having an incorrect reasoning path. We show the effectiveness of this design choice in Section 5.3. Additionally, incorporating ground truth as a hint significantly improved the utilization of the training set, increasing the average usage from 96 _._ 2% to 99 _._ 4% across 5 datasets, as shown in Table 1. 

Table 1: The proportion of correct reasoning paths corresponding to each instruction before and after adding hints in reasoning paths generation in the training set across the five tasks. 

|Tasks<br>Steps|`ReGenesis`<br>w/o hints|`ReGenesis`|
|---|---|---|
|GSM8K<br>NumGLUE<br>ARC-c<br>Reclor<br>StrategyQA|98_._4%<br>89_._6%<br>98_._6%<br>97_._5%<br>96_._9%|99_._5%<br>98_._2%<br>100_._0%<br>100_._0%<br>99_._4%|



Finally, for each dataset **D**[train] , we create a new dataset incorporating reasoning paths, denoted as _|_ **D** _|,|<_ **P** _|_ **D**[train] _[′]_ = �� _qi, ri,j_[path] _[, a][i]_ �� _i_ =1 _,j_ =1[, where] _[ r] i,j_[path][represents the correct reasoning path after filtering.] We randomly select up to _p_ correct reasoning paths for each instruction _qi_ . The language model **M** is then fine-tuned using the input _qi_ and the target outputs _{ri,j_[path] _[, a][i][}]_[.] 

## 4 EXPERIMENTS 

In this section, we assess the effectiveness of `ReGenesis` and existing methods in improving LLM’s reasoning capabilities for both in-domain tasks and OOD tasks. 

## 4.1 EXPERIMENT DETAILS 

## 4.1.1 TASKS AND DATASETS 

We selected the following five datasets for training, including mathematic, logical and common sense reasoning tasks. The training sets of these datasets are used to generate self-synthetic reasoning paths and to fine-tune the language models. The test sets are employed to evaluate the in-domain performance of the fine-tuned models. 

- Mathematic reasoning: GSM8K math problem dataset (Cobbe et al., 2021) and NumGLUE dataset (Mishra et al., 2022). In this experiment, we only use the numerical answers as the ground truth in the GSM8K dataset, excluding the human-provided reasoning paths. This approach enables us to independently evaluate the contribution of reasoning to the overall performance. 

- Logical reasoning: We use logical reasoning dataset ReClor (Yu et al., 2020). 

- Commonsense reasoning: AI2 Reasoning Challenge (ARC) (Clark et al., 2018) and StrategyQA (Geva et al., 2021) dataset. For ARC, we specifically use only the Challenge subset (ARC-c). 

5 

## 4.1.2 TRAINING DETAILS 

We conduct our main experiments using the instruction fine-tuned version of the Mistral-7B-v0.3 (Jiang et al., 2023) model, known as `Mistral-7B-Instruct-v0.3` . We also evaluate our method on other LLMs such as `Meta-Llama-3-8B-Instruct` (AI@Meta, 2024) and present results in Section 5. We first perform self-synthesized reasoning path generation as illustrated in Section 3.1. Using the Mistral-7B-Instruct-v0.3 with a temperature setting of 0 _._ 85, we first generate 25 diverse reasoning paths for every instruction within the training set. Subsequently, we employ an exact match method to filter out incorrect paths by comparing the answer in the reasoning path with the ground truth. For instructions that fail to generate any correct reasoning paths in the preceding steps, we incorporate the ground truth as a hint for reasoning path generation using the same model and temperature, as detailed in Section 3.2. Finally, we randomly select up to _p_ = 5 reasoning paths per question as target outputs. Therefore final training samples are up to the size of 5 _× |D_[train] _|_ , where _|D_[train] _|_ denotes the size of the training set. The final step involves fine-tuning the Mistral-7B-Instruct-v0.3 model using the generated reasoning paths, with a learning rate of 1 _e[−]_[6] and training for 3 epochs and a batch size of 16, utilizing an 8-GPU node of A100 GPUs, each with 40GB of memory. More details can be found in Appendix A.4.4. 

## 4.1.3 BASELINES 

**Base model without further fine-tuning** ( `w/o FT` ). For this baseline we just report the performance of the base LLM, Mistral-7B-Instruct-v0.3 (Jiang et al., 2023). No further fine-tuning is performed. 

**Fine-Tuning with Ground Truth Only** ( `FT w/ GT` ). In this approach, we fine-tune the base LLM using the original instructions and ground-truth answers in the training sets, without incorporating any self-synthesized reasoning paths. More details in Appendix A.4.1. 

**Fine-Tuning with Self-Improvement approach** ( `LMSI` ). LMSI (Huang et al., 2023) first samples self-synthesized reasoning paths using few-shot CoT prompting. It then filters these reasoning paths by selecting the one with the majority vote answer. Finally for this baseline, we fine-tune the base LLM using these self-generated solutions as the target outputs. More details in Appendix A.4.2. 

**Fine-Tuning with Self-Improvement approach and ground truth** ( `LMSI w/ GT` ). For this baseline, we follow the same reasoning generation process as LMSI. However, instead of using a majority vote to filter out incorrect reasoning paths as above, we utilize ground truth answers. The base LLM is then fine-tuned using these self-generated solutions as the target outputs. 

**Fine-Tuning with Self-Taught Reasoner approach** ( `STaR` ). STaR (Zelikman et al., 2022) begins by sampling self-synthesized reasoning paths directly using the few-shot CoT prompting outlined in their paper. In contrast, our framework progresses from abstract to concrete reasoning without relying on CoT examples. Then, it filters these paths based on ground truth and incorporate the ground truth as hints directly in the instructions to generate reasoning paths for instructions lacking correct solutions. Subsequently, fine-tune the base LLM using these self-synthesized solutions as target outputs. More details in Appendix A.4.3. 

## 4.1.4 EVALUATION METHODS 

At inference, for a fair and accurate comparison, we use exact match accuracy on all experiments. We have also tested multiple prompting methods for `ReGenesis` and each baseline at inference. In the “CoT Prompting” method (Wei et al., 2022), we prompt the LM models with:“ _Solve the following problem step-by-step. Question: ” + {question} + “Answer:_ ”, using a temperature setting of 0 _._ 8. In “Self-Consistency” (Wang et al., 2022), we use the same prompt but generate 15 responses at a temperature of 0.8, then select the final answer based on majority voting. 

## 4.2 IN-DOMAIN EXPERIMENTS 

In this experimental setup, we evaluate the enhancement of LLMs’ in-domain reasoning capabilities using `ReGenesis` , comparing it with existing self-improvement methods (§4.1.3). For each reasoning dataset described in Section 4.1.1, we construct reasoning paths (§3) from the training sets, finetune the model on the synthesized reasoning data (§4.1.2), and subsequently evaluate the model’s performance (§4.1.4) on the respective test sets. 

6 

Table 2: Comparison of zero-shot accuracy between fine-tuned and non-fine-tuned Mistral-7BInstruct-v0.3 models using different prompting methods. All fine-tuned models are trained on a single training set from one dataset and evaluated on the corresponding test set across 5 math, logical and commonsense reasoning datasets. 

|**Training**<br>**Methods**|**Prompting Method**<br>**at Inference**|**GSM8K**<br>**NumGlue**<br>**(Math)**<br>**(Math)**|**ARC-c**<br>**ReClor**<br>**(Logical)**<br>**(Logical)**|**StrategyQA**<br>**(Commonsense)**|
|---|---|---|---|---|
|`w/o FT`|CoT Prompting<br>Self-Consistency|44_._0%<br>40_._6%<br>60_._0%<br>38_._2%|77_._2%<br>57_._6%<br>80_._6%<br>56_._2%|77_._4%<br>80_._8%|
|`FT w/ GT`|CoT Prompting<br>Self-Consistency|13_._8%<br>55_._0%<br>15_._2%<br>55_._9%|77_._4%<br>70_._4%<br>77_._2%<br>**71**_._**6**%|85_._6%<br>85_._6%|
|`LMSI`|CoT Prompting<br>Self-Consistency|51_._8%<br>46_._5%<br>62_._3%<br>57_._1%|67_._9%<br>51_._8%<br>71_._7%<br>50_._8%|78_._3%<br>79_._0%|
|`LMSI w/ GT`|CoT Prompting<br>Self-Consistency|57_._4%<br>51_._8%<br>66_._3%<br>62_._2%|75_._9%<br>58_._0%<br>77_._5%<br>59_._4%|80_._2%<br>81_._7%|
|`STaR`|CoT Prompting<br>Self-Consistency|46_._3%<br>48_._3%<br>66_._0%<br>64_._5%|76_._5%<br>57_._8%<br>84_._1%<br>63_._8%|84_._4%<br>85_._9%|
|`ReGenesis`|CoT Prompting<br>Self-Consistency|63_._6%<br>52_._2%<br>**76**_._**6**%<br>**74**_._**7**%|78_._0%<br>68_._4%<br>**85**_._**4**%<br>70_._6%|81_._5%<br>**91**_._**3**%|



As presented in Table 2, `ReGenesis` significantly surpasses all baselines, achieving an average performance enhancement of 16 _._ 56% over the original model without fine-tuning. Baseline approaches, such as `LMSI` and `STaR` , which incorporate self-synthesized reasoning paths into their fine-tuning processes, also demonstrated improvements, albeit less significant than those achieved by `ReGenesis` . This suggests that the quality or diversity of the reasoning paths generated by these models is inferior to those created by `ReGenesis` . `LMSI` (Huang et al., 2023) adds self-synthesized reasoning paths to the fine-tuning training set, filtering these paths through self-consistency checks. However, the lack of guaranteed accuracy in these self-consistency reasoning paths compromises data quality, leading to a modest improvement of only 1 _._ 02% in average test performance. To address this issue, we implemented a modification where ground truth is utilized for filtering ( `LMSI w/ GT` ), thereby enhancing the quality of the reasoning paths. This adjustment raised the improvement from 1 _._ 02% to 6 _._ 26%. `STaR` (Zelikman et al., 2022) also employs ground-truth-verified reasoning paths but adopts a slightly different prompting and reasoning path format during post-training compared to `LMSI` , leading to a 9 _._ 7% test set improvement, which remains below our performance significantly. 

During our experiments, we also noted that training solely with ground-truth answers ( `FT w/ GT` ) and excluding reasoning paths does not consistently improve the performance of the base large language model (LLM), even on in-domain test sets. Specifically, in the GSM8K and ARC-C tasks, relying exclusively on ground-truth answers led to performance declines of 44.8% and 3.4%, respectively. This observation underscores the importance of incorporating reasoning paths during the fine-tuning stage to enhance the LLM’s reasoning capabilities. 

## 4.3 OUT-OF-DOMAIN EXPERIMENTS 

## 4.3.1 TASKS AND DATASETS 

In this experiment, we assess the fine-tuned language models from Section 4.2 on six OOD tasks. Our objective is to determine whether fine-tuning with or without self-synthesized reasoning paths influences the models’ general reasoning capabilities. Each model, fine-tuned on a specific task from Section 4.2, is tested across all the following six OOD tasks. 

- Mathematic reasoning: We use ASDIV (Miao et al., 2020), SVAMP (Patel et al., 2021) and the AQUA-RAT (Algebra Question Answering with Rationales) (Ling et al., 2017) datasets. 

- Logical reasoning: BIG-Bench Hard (BBH) (Suzgun et al., 2023) dataset, a subset of BIG-Bench. 

- Natural Language Inference (NLI): We utilize the Adversarial NLI (ANLI) (Mihaylov et al., 2018b) subsets ANLI-A2 and ANLI-A3. These subsets are more challenging than ANLI-A1 and include sentence pairs with entailment, neutral, or contradiction relations. 

- Commonsense Reasoning: We use OpenBookQA (Mihaylov et al., 2018a), a question-answering dataset modeled after open book exams for assessing human understanding of a subject. 

7 

Table 3: Zero-shot accuracy comparison between non-fine-tuned Mistral-7B-Instruct-v0.3 model and the models finetuned on one of five in-domain datasets separately and evaluated using the “Self-Consistency” prompting method across six out-of-domain tasks. 

||**Training**<br>**Methods**|**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|
|---|---|---|---|---|---|
|**Training**<br>||**ASDIV**<br>**SVAMP**<br>**AQUA**<br>**(Math)**<br>**(Math)**<br>**(Math)**|**BBH**<br>**(Logical)**|**ANLI**<br>**(NLI)**|**OpenbookQA**<br>**(Commonsense)**|
|**Datasets**||||||
||`w/o FT`<br>`FT w/ GT`<br>`LMSI w/ GT`<br>`STaR`<br>`Ours`|77_._2%–<br>75_._4%–<br>41_._3%–<br>54_._0%_↓_<br>40_._0%_↓_<br>29_._1%_↓_<br>77_._3%_↑_<br>72_._2%_↓_<br>31_._1%_↓_<br>79_._6%_↑_<br>71_._5%_↓_<br>46_._9%_↑_<br>**81**_._**2**% _↑_**83**_._**9**% _↑_**48**_._**8**% _↑_|60_._8%–<br>53_._6%_↓_<br>59_._5%_↓_<br>47_._4%_↓_<br>**69**_._**3**% _↑_|38_._4%–<br>44_._6%_↑_<br>43_._4%_↑_<br>45_._0%_↑_<br>**49**_._**5**% _↑_|75_._6%–<br>72_._6%_↓_<br>73_._2%_↓_<br>72_._8%_↓_<br>**81**_._**4**% _↑_|
|||||||
|**GSM8K**||||||
|**(Math)**||||||
|||||||
||`w/o FT`<br>`FT w/ GT`<br>`LMSI w/ GT`<br>`STaR`<br>`Ours`|77_._2%–<br>75_._4%–<br>41_._3%–<br>53_._8%_↓_<br>54_._9%_↓_<br>32_._3%_↓_<br>75_._7%_↓_<br>78_._2%_↑_<br>40_._6%_↓_<br>**79**_._**6**%_↑_<br>76_._1%_↑_<br>37_._0%_↓_<br>76_._9% _↓_<br>**79**_._**4**% _↑_**48**_._**4**% _↑_|60_._8%–<br>42_._4%_↓_<br>59_._5%_↓_<br>58_._5%_↓_<br>**61**_._**7**% _↑_|38_._4%–<br>37_._0%_↓_<br>35_._1%_↓_<br>41_._9%_↑_<br>**50**_._**0**% _↑_|75_._6%–<br>64_._8%_↓_<br>72_._6%_↓_<br>71_._6%_↓_<br>**79**_._**8**% _↑_|
|||||||
|**NumGLUE**||||||
|**(Math)**||||||
|||||||
||`w/o FT`<br>`FT w/ GT`<br>`LMSI w/ GT`<br>`STaR`<br>`Ours`|77_._2%–<br>75_._4%–<br>41_._3%–<br>62_._0%_↓_<br>56_._0%_↓_<br>22_._0%_↓_<br>77_._9%_↑_<br>76_._8%_↑_<br>45_._6%_↑_<br>76_._1%_↓_<br>74_._6%_↓_<br>46_._1%_↑_<br>**76**_._**4**% _↓_**76**_._**5**% _↑_**49**_._**6**% _↑_|60_._8%–<br>44_._6%_↓_<br>53_._9%_↓_<br>60_._6%_↓_<br>**66**_._**8**% _↑_|38_._4%–<br>47_._1%_↑_<br>35_._1%_↓_<br>42_._5%_↑_<br>**44**_._**8**% _↑_|75_._6%–<br>71_._4%_↓_<br>72_._4%_↓_<br>77_._2%_↑_<br>**81**_._**4**% _↑_|
|||||||
|**ReClor**||||||
|**(Logical)**||||||
|||||||
||`w/o FT`<br>`FT w/ GT`<br>`LMSI w/ GT`<br>`STaR`<br>`Ours`|77_._2%–<br>75_._4%–<br>41_._3%–<br>57_._7%_↓_<br>57_._2%_↓_<br>18_._9%_↓_<br>70_._9%_↓_<br>72_._0%_↓_<br>32_._7%_↓_<br>77_._0%_↓_<br>76_._2%_↑_<br>40_._6%_↓_<br>**81**_._**6**% _↑_**79**_._**5**% _↑_**46**_._**5**% _↑_|60_._8%–<br>37_._6%_↓_<br>60_._8%–<br>60_._7%_↓_<br>**66**_._**0**% _↑_|38_._4%–<br>36_._3%_↓_<br>32_._5%_↓_<br>**47**_._**4**%_↑_<br>46_._4% _↑_|75_._6%–<br>78_._4%_↑_<br>79_._4%_↑_<br>**84**_._**2**%_↑_<br>82_._8% _↑_|
|||||||
|**ARC-c**||||||
|**(Logical)**||||||
|||||||
||`w/o FT`<br>`FT w/ GT`<br>`LMSI w/ GT`<br>`STaR`<br>`Ours`|77_._2%–<br>75_._4%–<br>41_._3%–<br>69_._4%_↓_<br>72_._3%_↓_<br>43_._7%_↑_<br>56_._3%_↓_<br>56_._8%_↓_<br>40_._6%_↓_<br>79_._8%_↑_<br>76_._2%_↑_<br>**43**_._**3**%_↑_<br>**81**_._**3**% _↑_**81**_._**1**% _↑_<br>42_._9% _↑_|60_._8%–<br>52_._3%_↓_<br>60_._6%_↓_<br>62_._9%_↑_<br>**65**_._**9**% _↑_|38_._4%–<br>44_._9%_↑_<br>39_._8%_↑_<br>37_._3%_↓_<br>**55**_._**3**% _↑_|75_._6%–<br>62_._2%_↓_<br>68_._4%_↓_<br>77_._4%_↑_<br>**80**_._**4**% _↑_|
|||||||
|**StrategyQA**||||||
|**(Commonsense)**||||||
|||||||



## 4.3.2 RESULTS 

As illustrated in Table 3, `ReGenesis` surpasses all baseline models in OOD tasks. Specifically, `ReGenesis` shows a remarkable 6.1% improvement compared to the original model without finetuning. Conversely, existing methods display an average performance drop of roughly 4.6% on OOD tasks. These findings underscore the effectiveness of `ReGenesis` in enhancing the general reasoning capabilities of LLMs, enabling them to evolve into reasoning generalists through self-improvement. 

Firstly, as shown in Table 3, fine-tuning the language model solely on ground-truth solutions ( `FT w/ GT` ) significantly decrease its performance on all OOD tasks, indicating that employing reasoning paths for fine-tuning is even more important in OOD settings. Secondly, other baselines that fine-tune the language model using CoT prompted task-specific reasoning paths ( `LMSI w/ GT` , `STaR` ) do not improve their performance across all OOD domains. Specifically, the performance fluctuations range from a decrease of 20 _._ 9% to an increase of 8 _._ 6% on these OOD tasks. We further observe that the increase in OOD performance for such baseline methods almost only happen on OOD tasks closely relevant to its training data. For example, training on NumGLUE consistently boosts performance on SVAMP across all models that incorporate self-synthesized reasoning paths, with an average improvement of 2 _._ 5%. However, when the tasks are more OOD, performance tends to drop significantly. For example, training on NumGLUE or GSM8K leads to a consistent performance drop of approximately 3 _._ 1% on OpenBookQA and 4 _._ 6% on BBH when using the two baselines that incorporate reasoning paths, namely `LMSI w/ GT` and `STaR` . In contrast, `ReGenesis` does not restrict the self-synthesized reasoning paths with task-specific patterns or human-designed CoT examples, leading to performance improvement in almost all OOD scenarios tested. 

8 

Table 4: Zero-shot accuracy comparison of various component combinations of `ReGenesis` in Section 3.1 on GSM8K (Cobbe et al., 2021) datasets, using the “Self-Consistency” prompting method, evaluated on one in-domain and six out-of-domain tasks. 

|||**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|
|---|---|---|---|---|---|---|
|**Training**<br>|**Ablation**<br>|**GSM8K**|**ASDIV**<br>**SVAMP**<br>**AQUA**<br>**(Math)**<br>**(Math)**<br>**(Math)**|**BBH**<br>**(Logical)**|**ANLI**<br>**(NLI)**|**OpenbookQA**<br>**(Commonsense)**|
|**Datasets**|**Set**|**(Test)**|||||
||A+P|66_._2%|76_._1%<br>80_._3%<br>47_._6%<br>77_._5%<br>80_._1%<br>47_._2%<br>81_._2%<br>83_._9%<br>48_._8%|53_._4%<br>57_._9%<br>69_._3%|42_._1%<br>43_._0%<br>49_._5%|75_._6%<br>76_._4%<br>81_._4%|
|**GSM8K**<br>|S+P|70_._0%|||||
|**(Train)**|A+S+P|76_._6%|||||



## 5 ANALYSIS 

## 5.1 CHOICES OF DESIGN 

In this ablation study, we validate that each component of `ReGenesis` —“Guidance Adaptation,” “Reasoning Structure Generation,” and “Reasoning Path Generation” — plays a critical role in the `ReGenesis` ’s overall effectiveness. To evaluate the impact of different components on performance, we compared three combinations: “A+S”, which utilizes only “Guidance Adaptation” and “Reasoning Structure Generation”; “S+P”, which includes only “Reasoning Structure Generation” and “Reasoning Path Generation”; and “A+S+P”, which represents `ReGenesis` , employing all components. As shown in Table 4, performance significantly declines in both in-domain and 6 out-of-domain evaluations when either “Reasoning Structure Generation” (A+P) or “Guidance Adaptation” (S+P) is removed, demonstrating the necessity of all three processes. 

## 5.2 FILTERING METHODS 

In Section 3.2, we utilize the ground truth answers to filter out incorrect reasoning paths. For datasets lacking ground truth answers, Huang et al. (2023) proposed a method that relies on self-consistency in answers. Specifically, this approach filters self-synthesized reasoning paths by choosing the answer that matches the majority vote among the answers from these paths. We also applied this filtering technique to our framework on the GSM8K (Cobbe et al., 

Table 5: Zero-shot accuracy of models fine-tuned using majority-vote filtering on GSM8K (Cobbe et al., 2021) and StrategyQA (Geva et al., 2021), tested on their respective datasets with various prompting methods. 

|**Training**<br>**Method**|**GSM8K**<br>**(Math)**|**StrategyQA**<br>**(Commonsense)**|
|---|---|---|
|`w/o FT`<br>`LMSI`<br>`ReGenesis`(Majority)|56_._3%<br>62_._3%<br>**62**_._**8**%|80_._8%<br>79_._0%<br>**83**_._**1**%|



2021) and StrategyQA (Geva et al., 2021) tasks. The results, shown in Table 5, demonstrate that our framework remains effective even in the absence of ground truth answers, leading to a significant improvement in accuracy. Compared to `LMSI` (Huang et al., 2023), our framework achieves better performance due to the greater diversity in reasoning paths, which enhances the fine-tuned model’s reasoning capabilities. In conclusion, our framework is effective even when ground truth answers are not available. 

## 5.3 DIFFERENT POSITIONS TO ADD HINT 

In this experiment, we validated the hypothesis presented in Section 3.2, demonstrating that removing ground-truth answers in the final reasoning generation stage reduces the likelihood of producing correct answers with incorrect reasoning paths. Unlike `ReGenesis` , which first generates reasoning paths without hints and then adds hints only to those instructions lacking correct solutions, here we applied hints to all instructions in the GSM8K (Cobbe et al., 2021) training set for generating reasoning paths. To evaluate the impact of adding hints on performance, we compared two approaches: adding hints during “Guidance **A** daptation” and “Reasoning **S** tructure Generation” (A+S) versus adding ground-truth answers at every step including the “Reasoning **P** ath Generation (A+S+P). As shown in Table 6 , the A+S approach outperformed A+S+P, showing better results on in-domain tests and 5 out of 6 OOD tests. This experiment also confirmed that the reasoning paths generated with hints were of lower quality than those generated without them, supporting our approach of generating reasoning paths first and adding hints only for instructions without correct solutions in `ReGenesis` . 

9 

Table 6: Zero-shot accuracy comparison of different positions to add hints in Section 3.2 on GSM8K (Cobbe et al., 2021)datasets using the “Self-Consistency” prompting method, evaluated on one in-domain and six out-of-domain tasks. 

|||**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|
|---|---|---|---|---|---|---|
|**Training**<br>|**Hint**<br>|**GSM8K**|**ASDIV**<br>**SVAMP**<br>**AQUA**<br>**(Math)**<br>**(Math)**<br>**(Math)**|**BBH**<br>**(Logical)**|**ANLI**<br>**(NLI)**|**OpenbookQA**<br>**(Commonsense)**|
|**Datasets**|**Position**|**(Test)**|||||
|**GSM8K**|A+S|**75**_._**7**%|**79**_._**3**%<br>**79**_._**7**%<br>**50**_._**0**%<br>78_._8%<br>78_._4%<br>44_._1%|67_._3%<br>**69**_._**0**%|**41**_._**9**%<br>41_._3%|**77**_._**4**%<br>77_._0%|
|**(Train)**|A+S+P|73_._7%|||||



Table 7: Zero-shot accuracy comparison of fine-tuned and non-fine-tuned Meta-Llama-3-8B-Instruct model on NumGLUE datasets (Mishra et al., 2022) using the “Self-Consistency” prompting method, evaluated on one in-domain and six out-of-domain tasks. 

|||**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|**Test Datsets**|
|---|---|---|---|---|---|---|
|**Training**<br>|**Training**<br>|**NumGLUE**|**ASDIV SVAMP**<br>**AQUA**<br>**(Math)**<br>**(Math)**<br>**(Math)**|**BBH**<br>**(Logical)**|**ANLI**<br>**(NLI)**|**OpenbookQA**<br>**(Commonsense)**|
|**Datasets**|**Methods**|**(Test)**|||||
||`w/o FT`|58_._9%|88_._2%<br>86_._8%<br>**64**_._**6**%<br>88_._3%<br>87_._9%<br>61_._0%<br>**89**_._**2**%<br>**91**_._**1**%<br>63_._4%|36_._5%<br>69_._4%<br>**70**_._**1**%|53_._4%<br>52_._3%<br>**53**_._**9**%|82_._2%<br>81_._2%<br>**83**_._**8**%|
|**NumGLUE**<br>|`STaR`|75_._6%|||||
|**(Train)**|`ReGenesis`|**79**_._**2**%|||||



## 5.4 EFFECT OF THE LANGUAGE MODEL 

In Section 4.2, we evaluated our framework using the `Mistral-7B-Instruct-v0.3` as the base language model for generating and refining reasoning paths. In this section, we investigated whether a more advanced language model could also benefit from our framework. Hence we applied the same self-synthesized reasoning path generation process to `Meta-Llama-3-8B-Instruct` (AI@Meta, 2024), a model already trained on a diverse set of instruction datasets with robust reasoning paths and process supervision. To determine whether our framework could further enhance the performance of such a powerful model on unfamiliar tasks, we selected the NumGLUE (Mishra et al., 2022) benchmark, where the self-consistency zero-shot performance was initially only 58 _._ 9%. As shown in Table 7, fine-tuning Meta-Llama-3-8B-Instruct with 10 _,_ 000 training samples of NumGLUE using our framework boosted zero-shot test performance to 79 _._ 2%. The fine-tuned model also demonstrated a performance improvement of 6 _._ 46% on six OOD tasks. In contrast, the OOD performance of the strongest baseline, `STaR` , declined on 3 out of 6 datasets. This demonstrates that more advanced language models can also benefit significantly from our framework. 

## 5.5 DIVERSE PREFERENCES FOR REASONING GUIDELINES 

As detailed in Section 3.1, `ReGenesis` converts 25 general seed reasoning guidelines into 25 different reasoning paths for each instruction within the dataset. We are curious to see if the effectiveness of these guidelines vary across different base models. Therefore, for two different models, the Mistral-7B-Instructv0.3 and Meta-Llama-3-8B-Instruct models, we apply `ReGenesis` to the 25 guidelines for each instruction in the NumGLUE (Mishra et al., 2022) dataset and compute the average success rate ( _xi_ ) for each guideline across the entire dataset. To facilitate comparison of guideline preferences between models, we perform z-score normalization ( _z_ = _[x][−] σ[µ]_[, where] _[ µ]_ and _σ_ are mean and standard deviation of the success rate _x_ = _{x_ 1 _, ...., x_ 25 _}_ ) on the 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement_images/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement.pdf-0010-08.png)


Figure 2: Normalized percentage of successful guideline utilization for selective general reasoning guidelines on the NumGLUE dataset, comparing Mistral-7BInstruct-v0.3 and Meta-Llama-3-8B-Instruct models. 

success rates and select the 10 guidelines with the largest discrepancies between the two models. 

10 

As shown in Figure 2, different language models vary in their preference for guidelines when self-synthesizing reasoning paths. For instance, Mistral-7B-Instruct-v0.3 demonstrates a preference for “ _7. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements._ ”. In contrast, Meta-Llama-3-8B-Instruct favors “ _21. Let’s think step by step._ ”. It also suggests that existing methods, such as `STaR` , which generate reasoning paths as post-training data, might overlook these preferences in their designed CoT examples. In contrast, `ReGenesis` automates the process, leading to superior OOD performance. 

## 6 CONCLUSION 

We have introduced `ReGenesis` , a novel framework designed to self-synthesize reasoning paths as post- training data to self-improve their general reasoning capacities without requiring additional supervision beyond final answers and human-designed reasoning examples. This framework effectively self-synthesizes reasoning paths of any datasets, regardless of whether they include ground-truth answers. Fine-tuning the language model with such self-synthesized dataset leads to significant improvements in performance on both in-domain and out-of-domain tasks. We also investigated how various design choices, language model selection, and filtering methods, affect the performance of `ReGenesis` . In future work, we plan to explore the potential of leveraging filtered-out reasoning paths to further enhance language model performance and to develop strategies for intelligently selecting the most effective reasoning paths for model improvement. 

11 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/blob/main/ MODEL ~~C~~ ARD.md. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _arXiv preprint arXiv:1803.05457_ , 2018. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , 9:346–361, 2021. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. _arXiv preprint arXiv:2305.11738_ , 2023. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ , 2021. 

- Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 1051–1068, 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35: 22199–22213, 2022. 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_ , 2023. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, 2017. 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ , 2023. 

12 

- Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 975–984, 2020. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _EMNLP_ , 2018a. 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pp. 2381–2391, 2018b. 

- Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 3505–3523, 2022. 

- Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al. Orca 2: Teaching small language models how to reason. _arXiv preprint arXiv:2311.11045_ , 2023. 

- Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4. _arXiv preprint arXiv:2306.02707_ , 2023. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 2080–2094, 2021. 

- Yuxiao Qu, Tianjun Zhang, Naman Garg, and Aviral Kumar. Recursive introspection: Teaching language model agents how to self-improve. _arXiv preprint arXiv:2407.18219_ , 2024. 

- Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. Principle-driven self-alignment of language models from scratch with minimal human supervision. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 13003–13051, 2023. 

- Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 9426–9439, 2024. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Seonghyeon Ye, Yongrae Jo, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, and Minjoon Seo. Selfee: Iterative self-revising llm empowered by self-feedback generation. Blog post, May 2023. URL https://kaistai.github.io/SelFee/. 

- Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. Reclor: A reading comprehension dataset requiring logical reasoning. _arXiv preprint arXiv:2002.04326_ , 2020. 

Xiao Yu, Baolin Peng, Michel Galley, Jianfeng Gao, and Zhou Yu. Teaching language models to self-improve through interactive demonstrations. _arXiv preprint arXiv:2310.13522_ , 2023. 

13 

- Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al. Advancing llm reasoning generalists with preference trees. _arXiv preprint arXiv:2404.02078_ , 2024. 

- Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. _Advances in Neural Information Processing Systems_ , 35:15476–15488, 2022. 

- Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H Chi, Quoc V Le, and Denny Zhou. Take a step back: Evoking reasoning via abstraction in large language models. _arXiv preprint arXiv:2310.06117_ , 2023. 

- Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)_ , Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372. 

- Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans,¨ Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. _arXiv preprint arXiv:2205.10625_ , 2022. 

- Pei Zhou, Jay Pujara, Xiang Ren, Xinyun Chen, Heng-Tze Cheng, Quoc V Le, Ed H Chi, Denny Zhou, Swaroop Mishra, and Huaixiu Steven Zheng. Self-discover: Large language models self-compose reasoning structures. _arXiv preprint arXiv:2402.03620_ , 2024. 

14 

## A APPENDIX 

## A.1 DIVERSITY OF THE TRAINING DATASET 

To evaluate the effect of training set diversity on OOD task performance, we randomly combined 2, 3, and 4 datasets to create a training set of size 7 _,_ 000. We then fine-tuned the Mistral-7B-Instructv0.3 model on these datasets, using 3 different random seeds for each combination. More training details can be found in Appendix A.5.1. The average accuracy results are presented in Figure 3. As the number of combined datasets increases, the performance on OOD tasks generally tends to improve or stabilize. This indicates that an increase in training set diversity can posiFigure 3: tively impact the model’s performance on OOD tasks. Different tasks show varying combinations, using 3 3 random seeds for each combinasensitivity to training set diversity. Tasks tion. The total training sample size is 7 like BBH appear relatively unaffected by dataset diversity, whereas ASDIV, ANLI and AQUA benefit substantially from it. 

Figure 3: Accuracy is assessed across various dataset combinations, using 3 3 random seeds for each combination. The total training sample size is 7 _,_ 000. 

## A.2 HYPERPARAMETERS OF THE SELF-SYNTHESIZED DATASET CREATION 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement_images/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement.pdf-0015-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement_images/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement.pdf-0015-06.png)


**----- Start of picture text -----**<br>
(a) Number of Randomly Selected Reasoning Paths per<br>instruction<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement_images/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement.pdf-0015-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement_images/iclr_2025_ReGenesis_LLMs_can_Grow_into_Reasoning_Generalists_via_Self-Improvement.pdf-0015-08.png)


**----- Start of picture text -----**<br>
(b) Proportion of Training Data<br>**----- End of picture text -----**<br>


Figure 4: Zero-shot accuracy results on test set of GSM8K and StrategyQA. 

We first investigated the impact of varying the number of randomly sampled reasoning paths ( _p_ ) for each instruction-answer pair on the performance of our methods. The results, presented in Figure 4a, demonstrate a clear improvement in accuracy across both the GSM8K and StrategyQA tasks as the number of reasoning paths for each instruction-answer pair increases from 1 to 5. These findings suggest that training models with multiple reasoning paths for a given instruction enhances their capability to capture and utilize relevant reasoning pattern, thereby improving overall performance. 

We further examined the effect of varying the proportion of training data on the performance of our methods. As shown in Figure 4b, increasing the proportion of training data from 25% to 100% resulted in a significant improvement in accuracy for both the GSM8K and StrategyQA tasks. This finding highlights the critical role of sufficient and diverse training data in enhancing model performance, as it allows the models to capture and utilize more relevant reasoning, leading to better overall performance. 

15 

## A.3 DETAILS OF REASONING GENERALIST 

## A.3.1 SEED PROMPTS 

Table 8 shows 25 general seed prompts designed for activating the language model’s reasoning capabilities. 

Table 8: 25 Seed Prompts designed for `Ours` . 

How could I devise an experiment to help solve that problem? 

Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made. 

How could I measure progress on this problem? 

How can I simplify the problem so that it is easier to solve? 

How can I break down this problem into smaller, more manageable parts? 

Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidencebased decision-making, and identifying potential biases or flaws in thinking. 

Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality. 

Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole. 

Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches. 

What is the core issue or problem that needs to be addressed? 

What are the potential obstacles or challenges that might arise in solving this problem? 

Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed? 

How can progress or success in solving the problem be measured or evaluated? 

What indicators or metrics can be used? 

Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem? 

Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives? 

Is the problem an analytical one that requires data analysis, modeling, or optimization techniques? 

Is the problem a design challenge that requires creative solutions and innovation? 

Does the problem require addressing systemic or structural issues rather than just individual instances? 

What kinds of solution typically are produced for this kind of problem specification? 

Let’s think step by step. 

Let’s make a step by step plan and implement it with good notation and explanation. 

Ignoring the current best solution, create an entirely new solution to the problem. 

Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification? 

16 

What is the best way to modify this current best solution, given what you know about these kinds of problem specification? 

A.3.2 PROMPTS USED FOR GENERATING REASONING PATHS WITHOUT HINTS 

Table 9 shows the prompts we used in different steps of our framework. 

Table 9: Prompts used for generating reasoning paths. 

**Prompt for prompt adaption** : 

Without working out the solution, adapt the following reasoning modules to be specific to our task Reasoning Module: _{_ Seed Prompt _p_[raw] _j }_ Task: _{_ Instruction _qi}_ 

**Prompt for generating reasoning structure** : 

Without working out the solution, create an actionable and concise reasoning structure step by step for the task using this adapted reasoning module: _{_ Adapted Prompt _p_[adp] _i,j[}]_ Task: _{_ Instruction _qi}_ 

**Prompt for generating reasoning solutions** : Using the following reasoning structure: _{_ Reasoning Structure _ri,j_[s] _[}]_ Task: _{_ Instruction _qi}_ Solve this task step by step based on the above reasoning structure. 

## A.3.3 PROMPTS USED FOR GENERATING REASONING PATHS WITH HINTS 

In order to add ground truth answers as hints to our reasoning paths generation process, we use the prompts in Table 10. 

Table 10: Prompts used for generating reasoning paths with hints. 

**Prompt for prompt adaption** : Without working out the solution: _::::::::{_ Solution _:::ai}_ , adapt the following reasoning modules to be specific to our task Reasoning Module: _{_ Seed Prompt _p_[raw] _j }_ Task: _{_ Instruction _qi}_ **Prompt for generating reasoning structure** : Without working out the solution: _{::::::::_ Solution _:::ai}_ , create an actionable and concise reasoning structure step by step for the task using this adapted reasoning module: _{_ Adapted Prompt _p_[adp] _i,j[}]_ Task: _{_ Instruction _qi}_ 

**Prompt for generating reasoning solutions** : Using the following reasoning structure: _{_ Reasoning Structure _ri,j_[s] _[}]_ Task: _{_ Instruction _qi}_ Solve this task step by step based on the above reasoning structure. 

17 

## A.4 DETAILS OF TRAINING 

## A.4.1 BASELINE: FINE-TUNING WITH GROUND TRUTH (FT W/ GT) 

For each instruction _qi_ of the training set of each task in Section in 4.1.1, we fine-tune the Mistral7B-Instruct-v0.3 model specifically to generate the corresponding answer _ai_ . Training details are shown in Table 11. 

Table 11: Fine-Tuning with Ground Truth: Training Parameters. 

learning rate 1 _e[−]_[6] epochs 3 batch size 16 weight decay 0 _._ 1 lr scheduler type cosine warmup ratio 0.03 

## A.4.2 BASELINE: FINE-TUNING WITH SELF-IMPROVEMENT APPROACH ( **LMSI** ) 

Following the methods of Huang et al. (2023), for each instruction _qi_ , we use few-shot CoT prompting to sample 32 reasoning paths. We list the few-shot prompts for each dataset in 4.1.1. We filter the correct reasoning paths by majority vote or the ground truth. To keep the dataset size at the same scale, we further randomly sub-sample 5 reasoning paths from the correct paths. Finally, we fine-tune Mistral-7B-Instruct-v0.3 using these self-generated solutions as the target outputs. Training details are shown in Table 12. 

Table 12: Fine-Tuning with LMSI: Training Parameters. 

learning rate 1 _e[−]_[6] epochs 3 batch size 16 weight decay 0 _._ 1 lr scheduler type cosine warmup ratio 0.03 

**Chain-of-Thought Prompts for Each Dataset for** `LMSI` **.** We list the Chain-of-Thought Prompts for for each dataset `LMSI` experiments. 

Table 13: Few-shot CoT prompts for GSM8K. 

**Q** : There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 

**A** : We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6. 

**Q** : If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? 

**A** : There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5. 

**Q** : Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 

**A** : Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39. 

**Q** : Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? 

18 

**A** : Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8. 

**Q** : Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? 

**A** : He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9. 

**Q** : There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? 

**A** : There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29. 

**Q** : Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? 

**A** : Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33. 

**Q** : Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 

**A** : She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8. 

Table 14: Few-shot CoT prompts for ARC-c. 

**Q** : George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? (a) dry palms. (b) wet palms. (c) palms covered with oil. (d) palms covered with lotion. 

**A** : Dry surfaces will more likely cause more friction via rubbing than other smoother surfaces, hence dry palms will produce the most heat. The answer is (a). 

**Q** : Which factor will most likely cause a person to develop a fever? (a) a leg muscle relaxing after exercise. (b) a bacterial population in the bloodstream. (c) several viral particles on the skin. (d) carbohydrates being digested in the stomach. 

**A** : Option (b), bacterial population is the most likely cause for a person developing fever. The answer is (b). 

**Q** : Which change in the state of water particles causes the particles to become arranged in a fixed position? (a) boiling. (b) melting. (c) freezing. (d) evaporating. 

**A** : When water is freezed, the particles are arranged in a fixed position; the particles are still moving for all other options. The answer is (c). 

**Q** : When a switch is used in an electrical circuit, the switch can (a) cause the charge to build. (b) increase and decrease the voltage. (c) cause the current to change direction. (d) stop and start the flow of current. 

**A** : The function of a switch is to start and stop the flow of a current. The answer is (d). 

Table 15: Few-shot CoT prompts for StrategyQA. 

**Q** : Do hamsters provide food for any animals? 

**A** : Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is yes. 

**Q** : Could Brooke Shields succeed at University of Pennsylvania? 

19 

**A** : Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is yes. 

**Q** : Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? 

**A** : Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. The answer is no. 

**Q** : Yes or no: Is it common to see frost during some college commencements? 

**A** : College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. The answer is yes. 

**Q** : Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? 

**A** : The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. The answer is no.”, **Q** : Yes or no: Would a pear sink in water? 

**A** : The density of a pear is about 0.6 g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. The answer is no. 

Table 16: Few-shot CoT prompts for StrategyQA. 

**Q** : Given the context and corresponding question, choose the correct answer from the options. 

## Context: 

Paula will visit the dentist tomorrow morning only if Bill goes golfing in the morning. Bill will not go golfing unless Damien agrees to go golfing too. However, Damien has decided not to go golfing. Therefore, Paula will not be visiting the dentist tomorrow morning. 

## Question: 

The pattern of reasoning displayed above most closely parallels which of the following? 

Options: 

(A) If Marge goes to the bank today, Lauren will not cash her check tomorrow. Marge will not wash her car unless it is sunny. However, it is sunny, so Marge will wash her car and go shopping with Lauren. 

(B) Kevin will wash his car tomorrow only if Brittany has to go visit her grandmother. Unless Aunt Susan has to run errands, Brittany will not have to go visit her grandmother. Since Aunt Susan does not have to run errands, Kevin will not wash his car tomorrow. 

(C) Renee will do her homework tonight if there is nothing good on television and if her neighbors do not have a party. Although, there is something good on television; her neighbors are also having a party. Ttherefore, Renee will attend the party. 

(D) Maddie will plan a picnic only if one of her friends, Lisa or Kenny, will come. Kenny will not come to the picnic, but Lisa will. Ttherefore, Maddie will plan a picnic. 

**A** : We need to find an option that mirrors this logical structure: A condition leading to another condition (P→B). A dependency for the second condition (B→D). The negation of the dependency (¬D). The negation leading to the negation of the first condition (¬B→¬P). Let’s analyze the options: (A) involves different conditions and does not follow the same structure of dependencies and negations. (B) mirrors the structure closely: K→B (Kevin will wash his car only if Brittany visits her grandmother). B→A (Brittany will visit her grandmother only if Aunt Susan runs errands). ¬A (Aunt Susan does not have to run errands). ¬B (Brittany will not visit her grandmother). ¬K (Kevin will not wash his car). (C) involves multiple conditions and does not follow the same dependency and negation structure. (D) does not follow the same structure as it involves an either/or condition and a confirmation rather than negation. The answer is (B). 

20 

**Q** : Given the context and corresponding question, choose the correct answer from the options. 

## Context: 

When politicians describe their opponents’ positions, they typically make those positions seem implausible and unattractive. In contrast, scholars try to make opposing positions seem as plausible and attractive as possible. Doing so makes their arguments against those positions more persuasive to their professional colleagues. Politicians should take note: they could persuade more voters with their arguments if they simply followed the scholars in charitably formulating their opponents’ positions. 

## Question: 

The reasoning in the argument is most vulnerable to criticism on the grounds that it 

## Options: 

(A) fails to address the possibility that an approach that works with one kind of audience will not work with another 

(B) takes for granted that both scholars and politicians have persuasion as their aim 

(C) fails to account for the difficulty of coming up with charitable formulations of positions to which one is opposed 

(D) focuses on the differences between two styles of argumentation even though those styles might be suited to similar audiences. 

**A** : The argument is most vulnerable to criticism because it assumes that the approach used by scholars (charitably formulating opponents’ positions) will be equally effective for politicians without considering the differences between their audiences (professional colleagues vs. voters). The effectiveness of an argumentation style can vary significantly based on the audience’s expectations, context, and preferences. Scholars’ audiences (professional colleagues) may appreciate and be persuaded by a more charitable approach, while voters might respond differently. Therefore, the main flaw in the argument is that it overlooks the possibility that what works for one type of audience (scholars’ colleagues) may not work for another (politicians’ voters). The answer is (A). 

**Q** : Given the context and corresponding question, choose the correct answer from the options. 

## Context: 

A recent study monitored the blood pressure of people petting domestic animals in the laboratory. The blood pressure of some of these people lowered while petting the animals. Therefore, for any one of the people so affected, owning a pet would result in that person having a lower average blood pressure. 

## Question: 

The flawed pattern of reasoning in the argument above is most similar to that in which one of the following? 

## Options: 

(A) Since riding in a boat for a few minutes is relaxing for some people, those people would be more relaxed generally if those people owned boats. 

(B) Since pruning houseplants is enjoyable for some people, those people should get rid of houseplants that do not require frequent pruning. 

(C) Because buying an automobile is very expensive, people should hold on to an automobile, once bought, for as long as it can be maintained in running condition. 

21 

(D) Since giving a fence one coat of white paint makes the fence white, giving it two coats of white paint would make it even whiter. 

**A** : The flaw in this reasoning is the assumption that a short-term effect observed in a specific setting (the laboratory) will translate directly to a long-term effect in a different setting (daily life). This pattern of reasoning is flawed because it does not account for differences between the laboratory setting and real-life conditions. Let’s analyze the options to find the one that mirrors this flawed reasoning: (A) is similar to the original argument. It assumes that a short-term effect (relaxation from a few minutes of boat riding) will translate to a long-term effect (being more relaxed generally) without considering other factors. (B) does not follow the same pattern of reasoning. It suggests a different course of action (getting rid of certain houseplants) rather than assuming a short-term effect will lead to a long-term effect. (C) does not mirror the flawed reasoning. It is about a financial decision related to the cost of buying an automobile, not about short-term effects leading to long-term outcomes. (D) involves a misunderstanding of a physical process (painting) and assumes that adding more of the same will enhance the effect. It does not parallel the argument about short-term effects and long-term outcomes. The answer is (A). 

## Table 17: Few-shot CoT prompts for NumGLUE 

**Q** : A football team practices for 6 hours daily. This week they could not practice due to rain on 1 days. Find the total number of hours they practiced this week. 

**A** : A week has 7 days. Since it rained on 1 day, the team practiced on the remaining days of the week: 7 - 1 = 6 days. The total number of hours they practiced this week are 6 * 6 = 36 hours. The answer is 36. 

**Q** : A shopkeeper has 7 decks of playing cards. How many red color cards does he have in total? 

**A** : A standard deck of playing cards contains 52 cards. In a standard deck, half of the cards are red. Therefore, the total number of red color cards the shopkeeper has is 7 * 52 * 0.5 = 182. The answer is 182. 

**Q** : The hiking team needs to arrange gloves for every participant. If total number of participants is 43, how many minimum number of gloves the hiking team needs to arrange? 

**A** : Each participant needs 2 gloves. The minimum number of gloves for 42 participants are 43 * 2 = 86. The answer is 86. 

## A.4.3 BASELINE: FINE-TUNING WITH SELF-TAUGHT REASONER APPROACH ( **STAR** ) 

Following Zelikman et al. (2022), we first sample 32 reasoning paths using few-shot CoT prompting. An example is shown in Table 18. Next, we filter these paths based on ground truth and incorporate this ground truth as hints in the questions to generate reasoning paths for questions lacking correct solutions. An example is shown in Table 19. Subsequently, we fine-tune Mistral-7B-Instruct-v0.3 using these self-generated solutions as target outputs. Training details are shown in Table 20. 

Table 18: CoT prompts used in GSM8K dataset for fine-tuning with **STaR** . 

**Q** : Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? 

**A** : Natalia sold 48 _/_ 2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May. ####72 

**Q** : Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? 

**A** : In the beginning, Betty has only 100 _/_ 2 = 50. Betty’s grandparents gave her 15 _×_ 2 = 30. This means, Betty needs 100 _−_ 50 _−_ 30 _−_ 15 = 5 more. ####5 

22 

**Q** :Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read? 

**A** : Maila read 12 _×_ 2 = 24 pages today. So she was able to read a total of 12 + 24 = 36 pages since yesterday. There are 120 _−_ 36 = 84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84 _/_ 2 = 42 pages. ####42 

Table 19: CoT prompts used in GSM8K dataset for fine-tuning with **STaR** with hints. 

**Q** : Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? (Hints: 72) 

**A** : Natalia sold 48 _/_ 2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May. ####72 

**Q** : Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? (Hints: 5) 

**A** : In the beginning, Betty has only 100 _/_ 2 = 50. Betty’s grandparents gave her 15 _×_ 2 = 30. This means, Betty needs 100 _−_ 50 _−_ 30 _−_ 15 = 5 more. ####5 

**Q** :Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read? (Hints: 42) 

**A** : Maila read 12 _×_ 2 = 24 pages today. So she was able to read a total of 12 + 24 = 36 pages since yesterday. There are 120 _−_ 36 = 84 pages left to be read. Since she wants to read half of the remaining pages tomorrow, then she should read 84 _/_ 2 = 42 pages. ####42 

Table 20: Fine-Tuning with **STaR** : Training Parameters. 

learning rate 1 _e[−]_[6] epochs 3 batch size 16 weight decay 0 _._ 1 lr scheduler type cosine warmup ratio 0.03 

## A.4.4 `ReGenesis` 

We fine-tune the Mistral-7B-Instruct-v0.3 model specifically to generate the reasoning path _ri,j_ corresponding answer _ai_ given the instruction _qi_ . Training details are shown in Table 21. 

## A.5 TRAINING DETAILS OF SECTION 5 

## A.5.1 TRAINING DETAILS: DIVERSITY OF THE TRAINING DATASET 

We randomly select a subset of the training data comprising five InD tasks. To integrate the datasets of two, three, and four tasks, we randomly choose 3 _,_ 500, 2 _,_ 333, and 1 _,_ 750 instruction-reasoningpaths-answer pairs from two, three, and four of these five datasets, respectively. Each combination 

23 

Table 21: Fine-Tuning with `Ours` : Training Parameters. 

|learning rate<br>epochs<br>batch size<br>weight decay<br>lr scheduler type<br>warmup ratio|1_e−_6<br>3<br>16<br>0_._1<br>cosine<br>0.03|
|---|---|



is evaluated using three different random seeds. Then we use the same hyper-parameters to train Mistral-7B-Instruct-v0.3 model as in Table 21. 

## A.5.2 TRAINING DETAILS: EFFECT OF THE LANGUAGE MODEL 

We apply our data creation pipeline with Meta-Llama-3-8B-Instruct (AI@Meta, 2024) model with the help of vLLM (Kwon et al., 2023). We then fine-tune (Zheng et al., 2024) Meta-Llama-3-8B-Instruct with Low-Rank Adaptation (Hu et al., 2021). The training details are shown in Table 22. 

Table 22: Fine-Tuning Meta-Llama-3-8B-Instruct: Training Parameters. 

|learning rate<br>epochs<br>batch size<br>weight decay<br>lr scheduler type<br>warmup ratio|1_e−_4<br>3<br>8<br>0_._1<br>cosine<br>0.1|
|---|---|



A.6 EXAMPLES OF SELF-SYNTHESIZED REASONING PATHS 

A.6.1 SELF-IMPROVEMENT APPROACH (LMSI) 

Table 23: Self-synthesized reasoning paths generated by LMSI. 

**Q** : Jerry needs to shingle 3 roofs. Each roof is made of two slanted rectangular sides measuring 20 feet by 40 feet. If he needs 8 shingles to cover one square foot of roof, how many shingles does he need total? 

**Reasoning Path** (Correct): Each roof has two slanted rectangular sides. Let’s find the area of one roof. The area of a rectangle is the product of its length and width, so the area of one roof is 2 _×_ (20feet _×_ 40feet) = 2 _×_ 800 square feet = 1600 square feet. Since he needs 8 shingles to cover one square foot, he needs 1600 _×_ 8 = 12800 shingles for one roof. Since he needs to shingle 3 roofs, he needs 12800 _×_ 3 = 38400 shingles in total. The answer is 38400. 

**Reasoning Path** (Wrong): 20 _×_ 40 = 800 square feet. 8 shingles cover 1 square foot. So 8 _×_ 800 = 6400 shingles are needed. The answer is 6400. 

**Q** : Kobe and Pau went to a restaurant. Kobe ordered five pieces of fried chicken, and Pau ordered twice as much fried chicken as Kobe did. If they order another set of fried chicken, how many pieces of fried chicken will Pau have eaten in all? 

**Reasoning Path** (Correct): Kobe ordered 5 pieces of fried chicken. Since Pau ordered twice as much, Pau ordered 5 _×_ 2 = 10 pieces of fried chicken. If they order another set, Pau will eat 10 + 10 = 20 pieces of fried chicken in all. The answer is 20. 

24 

**Reasoning Path** (Wrong): 5 pieces of fried chicken were ordered the first time. Pau ordered twice as much as Kobe did, so he must have ordered 10 pieces of fried chicken. Now they order another set of fried chicken. So, Pau will have eaten 5 + 10 = 15 pieces of fried chicken. The answer is 15. 

**Q** : Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? 

**Reasoning Path** (Correct): Betty needs $100 for the wallet. She has only half of the money she needs, which means she has $50. Her parents give her $15, so she already has 50 + 15 = 65. Her grandparents give her twice as much as her parents, which is 15 _×_ 2 = 30. So now Betty has 65 + 30 = 95. She still needs 100 _−_ 95 = 5 more to buy the wallet. The answer is 5. 

**Reasoning Path** (Wrong): 1. Betty needs 100 dollars for a new wallet. 2. Her parents gave her 15 dollars. 3. Her grandparents gave her twice as much as her parents, so they gave her 15 _×_ 2 = 30 dollars. 4. Betty has 15 + 30 = $45. This is 45/100 = 45% of the money she needs. The answer is 55. 

**Q** : Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds? 

**Reasoning Path** (Correct): First, the weight of the box with jelly beans that make it weigh 2 pounds is 2 pounds. Then, when he added enough brownies to triple the weight, the weight became 2 _×_ 3 = 6 pounds. Then he added another 2 pounds of jelly beans, so the weight became 6 + 2 = 8 pounds. And finally, he added enough gummy worms to double the weight again, so the weight became 8 _×_ 2 = 16 pounds. The answer is 16. 

**Reasoning Path** (Wrong): 2 pounds of jelly beans + 6 pounds + 2 pounds of jelly beans + 2 pounds of gummy worms = 12 pounds. The answer is 12. 

**Q** : Tim rides his bike back and forth to work for each of his 5 workdays. His work is 20 miles away. He also goes for a weekend bike ride of 200 miles. If he can bike at 25 mph how much time does he spend biking a week? 

**Reasoning Path** (Correct): Tim rides his bike to and from work for each of the 5 workdays. That is 5 _×_ 2 _×_ 20 = 200 miles for his work rides. He also goes for a weekend bike ride of 200 miles. So in total he rides 200 + 200 = 400 miles. He bikes at 25 mph, so the total time he spends biking in a week is 400 miles / 25 mph = 16 hours. The answer is 16. 

**Reasoning Path** (Wrong): 25 mph is 25 miles per hour. He bikes back and forth to work for each of his 5 workdays. His work is 20 miles away. That means he bikes 5 _×_ 2 = 10 miles in each direction. He also goes for a weekend bike ride of 200 miles. That means he bikes 200 miles altogether during the week. So for 7 days he bikes a total of 30 miles. That means he bikes 30 _/_ 25 = 1 _._ 2 hours in a week. The answer is 12. 

## A.6.2 FINE-TUNING WITH SELF-TAUGHT REASONER APPROACH ( `STaR` ) 

Table 24: Self-synthesized reasoning paths generated by `STaR` . 

**Q** : Jerry needs to shingle 3 roofs. Each roof is made of two slanted rectangular sides measuring 20 feet by 40 feet. If he needs 8 shingles to cover one square foot of roof, how many shingles does he need total? 

**Reasoning Path** (Correct): Jerry needs to shingle two slanted rectangular sides for each roof. The combined area for one roof is 2 _×_ (20 _∗_ 40) = 2 _∗_ 800 = 1600 square feet per roof. So the total area Jerry needs to shingle is 3 _×_ 1600 = 4800 square feet. Since he needs 8 shingles to cover one square foot, the number of shingles he needs in total is 4800 _×_ 8 = 38 _,_ 400 shingles. #### 38400 

25 

**Reasoning Path** (Wrong): First, let’s calculate the area of each roof. Each roof is made of two slanted rectangular sides of 20 feet by 40 feet. So, the area of one roof is (20 _×_ 40) + (20 _∗_ 40) = 800 square feet. Since he needs 8 shingles to cover one square foot of roof, he needs (800 _×_ 8) shingles for a single roof. Since Jerry needs to shingle 3 roofs, he needs (800 _×_ 8) _×_ 3 = 24000 shingles. #### 24,000 

**Q** : Kobe and Pau went to a restaurant. Kobe ordered five pieces of fried chicken, and Pau ordered twice as much fried chicken as Kobe did. If they order another set of fried chicken, how many pieces of fried chicken will Pau have eaten in all? 

**Reasoning Path** (Correct): Pau ordered 2 _×_ 5 = 10 pieces of fried chicken. If they order another set of fried chicken, Pau will eat an additional 2 _×_ 5 = 10 pieces. Pau ate 10 + 10 = 20 pieces of fried chicken in all. #### 20 

**Reasoning Path** (Wrong): Kobe ordered 5 pieces of fried chicken, so Pau ordered twice as much: 5 _×_ 2 = 10 pieces. After they order another set, Pau will have eaten a total of 10 + 5 = 15 pieces of fried chicken. #### 15 

**Q** : Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? 

**Reasoning Path** (Correct): In the beginning, Betty has only 100 _/_ 2 = 50. Betty’s parents gave her 15, and her grandparents gave her 15 _×_ 2 = 30. This means, Betty now has 50 + 15 + 30 = 95. She needs 100 - 95 = 5 more to buy the wallet. #### 5 

**Reasoning Path** (Wrong): Betty needs to have 100 + 15 + (2 _×_ 15) = 130 to buy the wallet. However, she has 100 _/_ 2 = 50, and she received 15 from her parents and 2 _×_ 15 = 30 from her grandparents, which is a total of 50 + 15 + 30 = 95. So, she doesn’t need any more money. She actually has already saved enough to buy the wallet. #### 0 (doesn’t need any more money). 

**Q** : Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds? 

**Reasoning Path** (Correct): Initially, the weight was 2 pounds, and then it tripled, so its weight became 2 _×_ 3 = 6 pounds. Ken then added another 2 pounds of jelly beans, so the total weight was 6 + 2 = 8 pounds. Finally, Ken added gummy worms to double the weight again, so the final weight was 8 _×_ 2 = 16 pounds. #### 16 

**Reasoning Path** (Wrong): The weight was 2 pounds, and then he added enough jelly beans to triple the weight, we need to find x, the number of pounds of jelly beans, and then double it to find the final weight: 2 + 3 _x_ = 3(2 + _x_ ), so _x_ = 1. This means the weight of the jelly beans after the first addition is 1 pound, and then the total weight (including the original 2 pounds of weight) after the second addition is 3(2 + 1) = 7 pounds. After adding another 2 pounds of jelly beans, the weight is 7 + 2 = 9 pounds. Finally, he doubled the weight again, so the final weight is 9 _×_ 2 = 18 pounds. The final weight of the box of goodies was 18 pounds. #### 18 

**Q** : Tim rides his bike back and forth to work for each of his 5 workdays. His work is 20 miles away. He also goes for a weekend bike ride of 200 miles. If he can bike at 25 mph how much time does he spend biking a week? 

**Reasoning Path** (Correct): Tim bikes to work and back, so his distance for work is 2 x 20 = 40 miles for each of 5 workdays. This makes it 40 * 5 = 200 miles for work in a week. He goes for a weekend bike ride of 200 miles. The total distance for a week is 200 + 200 = 400 miles. Finally, he can bike at 25 mph, so he spends 400 / 25 = 16 hours biking a week. #### 16 hours 

26 

**Reasoning Path** (Wrong): Since the trip to work and back is 20 miles each way and he goes to work every day, we can calculate the distance he bikes per week as 20 _×_ 5 = 100 miles. The round trip to work takes him twice the distance divided by his speed, so it takes him 100/2/25 = 8 hours per week just for going to work and back. The weekend bike ride of 200 miles takes 200/25 = 8 hours as well, so in total Tim bikes for 8 _×_ 5 + 8 = 44 hours a week. #### 44 

## A.6.3 FINE-TUNING WITH `ReGenesis` 

Table 25: Self-synthesized reasoning paths generated by `Ours` . 

**Q** : Jerry needs to shingle 3 roofs. Each roof is made of two slanted rectangular sides measuring 20 feet by 40 feet. If he needs 8 shingles to cover one square foot of roof, how many shingles does he need total? 

**Reasoning Path** (Correct): To find the total number of shingles Jerry needs to shingle 3 roofs, let’s follow these steps: 1. Calculate the area of one roof: The area of one slanted rectangular side is 20 feet _×_ 40 feet = 800 square feet. Since each roof has two sides, the total area of one roof is 2 _×_ 800 square feet = 1600 square feet. 2. Determine the number of shingles required to cover one square foot of roof: Jerry needs 8 shingles to cover one square foot of roof, so for the entire roof, he will need 1600 square feet _×_ 8 shingles/square foot = 12,800 shingles. 3. Calculate the total number of shingles Jerry needs for all 3 roofs: Since Jerry needs to shingle 3 roofs, the total number of shingles he needs is 3 _×_ 12,800 shingles = 38,400 shingles. So, Jerry needs a total of 38,400 shingles to shingle all 3 roofs. 

**Reasoning Path** (Wrong): To solve this problem, let’s follow these steps: 1. Calculate the area of one roof: Area of one roof = Length _×_ Width = 20 feet _×_ 40 feet = 800 square feet. 2. Calculate the total area of all 3 roofs: Total area = Area of one roof _×_ Number of roofs = 800 square feet _×_ 3 = 2400 square feet. 3. Calculate the total number of shingles needed for all 3 roofs: Total shingles = Total area _×_ Shingles per square foot = 2400 square feet _×_ 8 = 19,200 shingles. Therefore, Jerry needs 19,200 shingles to cover all 3 roofs. 

**Q** : Kobe and Pau went to a restaurant. Kobe ordered five pieces of fried chicken, and Pau ordered twice as much fried chicken as Kobe did. If they order another set of fried chicken, how many pieces of fried chicken will Pau have eaten in all? 

**Reasoning Path** (Correct): To solve this problem, follow these steps: 1. Understand the given information: Kobe ordered 5 pieces of fried chicken, and Pau ordered twice as much as Kobe. 2. Calculate the number of pieces Pau ordered: Pau ordered 2 _×_ Kobe’s order = 2 _×_ 5 = 10 pieces of fried chicken. 3. Consider the initial serving: Since they haven’t eaten any fried chicken yet, the number of pieces Pau has eaten so far is 0. 4. Calculate the number of pieces Pau will have eaten after another set: If they eat another set of fried chicken together, the total number of pieces Pau will have eaten in all will be the initial amount he ordered (10 pieces) plus the amount he will eat in the new order. As both Kobe and Pau are eating the same amount, the number of pieces Pau will eat in the new order will be equal to the number of pieces he initially ordered, which is 10 pieces. So, the total number of pieces Pau will have eaten after another set (if they eat the new order together) is 10 (initial order) + 10 (new order) = 20 pieces. 

**Reasoning Path** (Wrong): Unfortunately, I cannot provide the exact number of pieces of fried chicken Pau will have eaten in all, as I don’t have the information about the number of pieces in the second order. However, based on the reasoning structure you provided, we can calculate the number of pieces Pau will eat in total as follows: 1. Pau ordered twice as much as Kobe, so he ordered 2 _×_ Kobe’s order = 2 _×_ 5 = 10 pieces. 2. The number of pieces Pau has eaten from his initial order is not known at this point. 3. In the second order, Pau will receive the same number of pieces as Kobe, which is 5 pieces. 4. The total number of pieces Pau will have eaten is the sum of the pieces he has eaten from his initial order and the pieces he will eat in the second order. However, since the number of pieces Pau has eaten from his initial order is not known yet, I can’t provide the total number of pieces Pau will have eaten in all. 5. If they don’t eat all the pieces in the second order, Pau’s total number of eaten pieces will be less than the sum calculated in step 4. 6. Once the number of pieces in the second order is known, you can find the exact number of pieces Pau has eaten in all by following steps 2, 3, and 4 again. 

27 

**Q** : Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? 

**Reasoning Path** (Correct): To find out how much more money Betty needs, let’s follow these steps: 1. Betty currently has half the amount needed for the wallet, which is $50. 2. Her parents will give her $15, and her grandparents will give twice as much as her parents, which is 15 _×_ 2 = 30. 3. The total amount Betty will receive is the sum of her initial savings, her parents’ contribution, and her grandparents’ contribution: $50 + $15 + $30 = $95. 4. To find out how much more money Betty needs to buy the wallet, subtract the total amount she received from the wallet’s cost: $100 - $95 = $5. Therefore, Betty needs an additional $5 to buy the wallet. 

**Reasoning Path** (Wrong): To find out how much more money Betty needs to save up to afford the new wallet, we can follow the reasoning structure as follows: 1. Determine the amount of money Betty currently has: Betty has half of the money she needs for the wallet. The cost of the wallet is $100. Since Betty has half of the money needed, she currently has $50. 2. Calculate the amount her parents are giving her: Her parents are giving her $15. 3. Calculate the amount her grandparents are giving her (twice as much as her parents): Her grandparents are giving her twice as much as her parents. Her parents are giving $15. So, her grandparents are giving $15 _×_ 2 = 30. 4. Add the amounts her parents and grandparents are giving her to find the total amount she will receive: Betty will receive $15 (from parents) + $30 (from grandparents) = $45. 5. Subtract the total amount she will receive from the cost of the wallet to find out how much more she needs to save: Betty still needs to save up $100 (cost of the wallet) - $45 (amount she will receive) = $55. So, Betty still needs to save up an additional $55 to afford the new wallet. 

**Q** : Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds? 

**Reasoning Path** (Correct): To find the final weight of the care package, let’s follow the steps: First, let’s understand the weights involved: 1. Ken started with a box of jelly beans that weighed 2 pounds. 2. He added brownies to triple the weight, so the new weight was 2 _×_ 3 = 6 pounds. 3. He then added another 2 pounds of jelly beans, bringing the weight to 6 + 2 = 8 pounds. 4. Finally, he added gummy worms to double the weight again. So, the final weight is 8 _×_ 2 = 16 pounds. 

**Reasoning Path** (Wrong): To find the final weight of the care package, let’s follow the steps: 1. Identify the weights of each item added to the care package: Jelly beans (first addition): 2 pounds. Brownies: 6 pounds (tripled the initial weight of 2 pounds). Gummy worms: 20 pounds (doubled the total weight after adding jelly beans and brownies). 2. Calculate the weight of jelly beans after the first addition and the final weight before adding gummy worms: After adding 2 pounds of jelly beans, the weight was 4 pounds. So, the weight of the first addition of jelly beans is 4 pounds - 2 pounds (initial weight) = 2 pounds. Since the total weight before adding gummy worms was 10 pounds (4 pounds from jelly beans + 6 pounds from brownies), the weight of the second addition of jelly beans (which caused the weight to double) is 10 pounds _×_ 0.5 = 5 pounds. Adding this to the weight of the first addition of jelly beans, we get the total weight of jelly beans: 2 pounds + 5 pounds = 7 pounds. 3. Calculate the final weight of the care package: The final weight of the care package is the sum of the weights of the jelly beans, brownies, and gummy worms: 4 pounds (first jelly beans) + 6 pounds (brownies) + 20 pounds (gummy worms) = 30 pounds. 

**Q** : Tim rides his bike back and forth to work for each of his 5 workdays. His work is 20 miles away. He also goes for a weekend bike ride of 200 miles. If he can bike at 25 mph how much time does he spend biking a week? 

**Reasoning Path** (Correct): To determine the amount of time Tim spends biking in a week, we can follow these steps: 1. Calculate the total distance Tim bikes for commuting to work each day: Distance per one-way commute: 20 miles. Distance for daily commuting: 20 miles _×_ 2 (round trip) = 40 miles per day. 2. Calculate the total distance Tim bikes for commuting in a week: Distance per week for commuting: 40 miles/day _×_ 5 days = 200 miles. 3. Since Tim doesn’t have a separate weekend ride, the total distance for the weekend remains the same as the total distance for commuting: 200 miles. 4. Calculate the total distance Tim bikes in a week: Total distance per week: Commuting distance per week + Weekend distance = 200 miles + 200 miles = 400 miles. 5. Calculate the total time Tim spends biking in a week: Time spent biking per week: Total distance per week / Biking speed = 400 miles / 25 mph = 16 hours. 

28 

**Reasoning Path** (Wrong): To find out how much time Tim spends biking in a week: 1. Tim bikes 20 miles to work each day for 5 workdays, so he bikes a total of 100 miles for workdays (20 miles _×_ 5 days). 2. Tim bikes 200 miles for the weekend ride. 3. The total distance Tim bikes in a week is 100 miles (workdays) + 200 miles (weekend ride) = 300 miles. 4. It takes Tim 300 miles / 25 mph to bike this distance, which is 12 hours (300 miles / 25 mph). 5. Since Tim bikes for 12 hours during the week but the question asks for the total time spent biking in a week, we also need to account for the 8 hours he spends biking during the weekend ride (200 miles / 25 mph). 6. Therefore, the total time Tim spends biking in a week is 12 hours (workdays) + 8 hours (weekend ride) = 20 hours. 

29 

