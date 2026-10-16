Published as a conference paper at ICLR 2025 

SUPERCORRECT: ADVANCING SMALL LLM REASONING - WITH THOUGHT TEMPLATE DISTILLATION AND SELF CORRECTION 

**Ling Yang**[1] _[∗]_[�] **, Zhaochen Yu**[1] _[∗]_ **, Tianjun Zhang**[4] **, Minkai Xu**[5] **, Joseph E. Gonzalez**[4] **Bin Cui**[1] _[†]_ **, Shuicheng Yan**[2] _[,]_[3] _[†]_ 1Peking University, 2Skywork AI, 3National University of Singapore, 4UC Berkeley, 5Stanford University Project: https://github.com/YangLing0818/SuperCorrect-llm 

## ABSTRACT 

Large language models (LLMs) like GPT-4, DeepSeek-R1, and ReasonFlux have shown significant improvements in various reasoning tasks. However, smaller LLMs still struggle with complex mathematical reasoning because they fail to effectively identify and correct reasoning errors. Recent reflection-based methods aim to address these issues by enabling self-reflection and self-correction, but they still face challenges in independently detecting errors in their reasoning steps. To overcome these limitations, we propose SUPERCORRECT, a novel two-stage framework that uses a large teacher model to _supervise_ and _correct_ both the reasoning and reflection processes of a smaller student model. In the first stage, we extract hierarchical high-level and detailed thought templates from the teacher model to guide the student model in eliciting more fine-grained reasoning thoughts. In the second stage, we introduce cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher’s correction traces during training. This cross-model DPO approach teaches the student model to effectively locate and resolve erroneous thoughts with errordriven insights from the teacher model, breaking the bottleneck of its thoughts and acquiring new skills and knowledge to tackle challenging problems. Extensive experiments consistently demonstrate our superiority over previous methods. Notably, our SUPERCORRECT-7B model significantly **surpasses powerful DeepSeekMath-7B by 7.8%/5.3% and Qwen2.5-Math-7B by 15.1%/6.3%** on MATH/GSM8K benchmarks, achieving new SOTA performance among all 7B models. 

## 1 INTRODUCTION 

Large language models (LLMs) (Brown et al., 2020; Anil et al., 2023; Achiam et al., 2023; Du et al., 2022; Jiang et al., 2024; Touvron et al., 2023a;b), such as GPT-4 (Achiam et al., 2023), DeepSeek-R1 (Guo et al., 2025), and ReasonFlux (Yang et al., 2025), have demonstrated significant improvements in various reasoning tasks. However, despite being pre-trained on large-scale mathematical datasets using diverse techniques, smaller models like Llama-3-8B (Dubey et al., 2024) and Qwen2.5-Math-7B (Yang et al., 2024a) continue to struggle with complex mathematical reasoning tasks. 

Existing works aim to enhance the mathematical performance of LLMs through various approaches. We categorize these methods into two types: **traditional fine-tuning optimization** and **reflection-based optimization** . Traditional fine-tuning methods mainly focus on the exploration in training techniques like Supervised Fine-Tuning (SFT) (Roziere et al., 2023; Shao et al., 2024; Dubey et al., 2024), and LLM-alignment strategies like Reinforcement Learning from Human Feedback (RLHF) (Achiam et al., 2023; Ouyang et al., 2022; Bai et al., 2022a;b) and alternative methods like Direct Preference Optimization (DPO) (Rafailov et al., 2024). Although these methods have shown remarkable progress across a wide range of language tasks, their optimization objectives only focus on direct answers or simple reasoning rationales. Consequently, they struggle to locate the errors in the reasoning process and fail to revise the flawed reasoning logic of language models. 

Recent reflection-based methods attempt to address the shortcomings of fine-tuning methods and leverage the pre-designed prompts or general rules to instruct language models for self-reflection and self-correction during reasoning process (Shinn et al., 2024; Kim et al., 2024). Some methods (Li et al., 2023; 2024c) further employ LLMs to synthesize rule-based datasets for enhancing their self-correction abilities in training stage. However, as mentioned in Tyen et al. (2024), LLMs still struggle to independently identify errors in their reasoning steps. With- 

> _∗_ Equal Contribution. � yangling0818@163.com 

> _†_ Corresponding authors. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Reasoning Thought<br>Data Student LLM Student LLM Student LLM<br>Self-Correction Trace<br>DPO/RLHF<br>Supervise<br>Paired<br>Teacher  Extract Teacher  Correction<br>LLM LLM Traces<br>Hierarchical  Thought  Template Cross-model Correction Trace<br>Stage-1 SFT with Hierarchical Thought Template  Stage-2 Cross-model Collaborative DPO<br>SuperCorrect<br>Reasoning Thought<br>**----- End of picture text -----**<br>


Figure 1: Overview of our proposed two-stage framework SUPERCORRECT. In the first stage, we extract hierarchical thought template from teacher LLM to supervise student LLM for producing more specific thoughts. In the second stage, we collect a dataset of paired self- and cross-correction traces for cross-model collaborative DPO. 

out accurate error identifications, self-correction becomes more challenging. In complex mathematical reasoning, even when mistake locations are provided, LLMs often remain biased or misled by their previous reasoning context. Thus it remains difficult for language models to clarify the causes of reasoning errors within a single LLM. 

To address these limitations, we propose a novel two-stage framework, namely **SUPERCORRECT** , utilizing a large teacher model’s thoughts to _supervise_ and _correct_ both the reasoning and reflection processes of a smaller student model. As depicted in Figure 1, in the first stage, we extract _hierarchical thought template_ from the teacher LLM to guide the student model in generating more fine-grained reasoning thoughts. The template contains a high-level thought providing a summarized and generalized solution for similar problems, and a detailed solution offering a detailed explanation of the critical reasoning steps. Compare to previous thought format such as CoT (Wei et al., 2022) and BoT (Yang et al., 2024b; 2025), our hierarchical thought templates offer deeper and more informative reasoning insights for later error corrections. In second stage, we propose _cross-model collaborative DPO_ to optimize the student model and enhance its self-correction abilities by following the teacher’s crossmodel correction traces during training. Specifically, instead of merely simulating correct answers or preferred reasoning process, we instruct teacher LLM to identify and correct the error parts in student’s thoughts. This cross-model correction trace is then used to guide the student model in performing better self-correction, enabling it to avoid and rectify specific errors. The critical insight of our cross-model DPO approach is enabling student language models to break the bottleneck of its thoughts and acquiring new error-driven insights and knowledge from teacher’s correction traces. 

Furthermore, we construct a high-quality fine-tuning dataset equipped with designed hierarchical thought templates containing 100k samples, and a pair-wise preference dataset for thought-level correction optimization containing 10k samples, which consists of: 1) a math problem, 2) prior reasoning steps in our pre-designed format, 3) the step with chosen analysis and corrective guidance, generated by teacher LLMs based on the ground truth solution 4) the step with rejected analysis and correction guidance, generated by student LLMs without access to the ground truth solution. 

We summarize our contribution as follows: **(i)** We propose a novel two-stage fine-tuning method SUPERCORRECT for improving both reasoning accuracy and self-correction ability for LLMs. **(ii)** We propose hierarchical thought based fine-tuning to enable small-sized LLMs to produce more accurate and fine-grained reasoning thoughts. **(iii)** We propose cross-model collaborative DPO, which innovatively leverage SOTA LLMs to locate and correct the specific error thoughts in the reasoning process of smaller student LLMs, thus advancing their self-correction ability and breaking their thought bottleneck. **(iv)** We construct two high-quality datasets and develop three powerful reasoning LLMs SUPERCORRECT-Qwen/DeepSeek/Llama-7B, **achieving 70.2% accuracy on the MATH dataset and 89.5% on the GSM8K dataset, setting new SOTA performance among all 7B models** . 

## 2 RELATED WORK 

**Reinforcement Learning from Human Feedback for Large Language Models** To improve the performance and reliability of LLMs, RLHF methods like Christiano et al. (2017) and Ouyang et al. (2022) are introduced for LLM alignment. This method is more demanding in dataset because it requires pair-wise annotated data to train a reward model thus reflecting human preferences. And then train the policy model using reinforcement learning to maximize the estimated reward. Although this method proves to be effective, due to its reliance on the quality of reward model, this process is complex and computationally intensive. To simplify this process, Direct Preference Optimization (DPO) (Rafailov et al., 2024) was proposed which directly uses pair-wise data for optimization. By defining the preference loss as a function of the policy, DPO can optimize the policy using straightforward 

2 

Published as a conference paper at ICLR 2025 

training techniques, avoiding the complexities of reinforcement learning. However, current methods only show limited improvements in mathematical reasoning due to the design of optimization unit. Works like Step-DPO(Lai et al., 2024) establish a more fine-grained reward unit by considering each intermediate reasoning step as a basic unit. However, they fail to clarify error causes and provide explicit guidance for correcting errors. In this paper, we specifically design a cross-model teacher-student collaborative thought-based reward, which takes each correction step as a basic optimization unit. 

**Reasoning with Self-Correction/Reflection** Self-correction for reasoning has shown promise in improving LLM outputs in terms of style and quality. Previous works (Li et al., 2023; Shinn et al., 2024; Madaan et al., 2024; Saunders et al., 2022; Miao et al., 2023; Chen et al., 2023a) focus on the concept of self-correction, i.e. having an LLM correct its own outputs. However, as mentioned in Huang et al. (2023), while self-correction may prove effective for improving model outputs in terms of style and quality, when it comes to reasoning tasks, LLMs struggle to identify and fix errors without external feedback. For example, Reflexion (Shinn et al., 2024) and RCI (Kim et al., 2024) both use ground truth correctness as a signal to halt the self-correction loop. Moreover, some attempts to self-correct logical or reasoning errors can sometimes turn correct answers into incorrect ones, resulting in worse overall performances (Huang et al., 2023). While previous works typically present self-correction as a process conducted within a specific LLM, our method leverage large-sized LLMs to explicitly identify the errors and gain correction insights from the errors. With this corss-model reward, we can revise the weaknesses exposed by small-sized LLMs during reasoning tasks through fine-tuning and correction-based preference optimization. 

**Thought Expansion for Mathematical Reasoning** Thought expansion for reasoning mainly focus on predesigned reasoning structure or template, which leverage prompting techniques to enhance mathematical reasoning capabilities of LLMs. Chain-of-Thought (CoT) prompting (Wei et al., 2022) and its variants (Kojima et al., 2022; Press et al., 2023; Arora et al., 2022), such as Least-to-Most (Zhou et al., 2022), Decomposed Prompting (Khot et al., 2022), and Auto-CoT (Zhang et al., 2022)—prompt LLMs to break down complex questions into simpler subtasks and systematically solve them before summarizing a final answer. Innovations like Tree-ofThought (Yao et al., 2024) and Graph-of-Thought (Besta et al., 2024), have further complex this field by exploring dynamic, non-linear reasoning pathways to expand heuristic capabilities of LLMs (Chen et al., 2023b; Ning et al., 2023). Other methods like PoT (Chen et al., 2022), PAL (Gao et al., 2023b) and (Gou et al., 2023) attempt to utilize external tools such as code to avoid hallucination of LLMs in the mathematical reasoning process. However, they suffer from increased resource demands and greater time complexity, depend on manual prompt crafting, and are often tailored to specific task types. Recent BoT (Yang et al., 2024b) propose a task-agnostic paradigm with meta buffer to efficiently solve the problems based on accumulated thought templates. However, it is a trainingfree framework which may not essentially boost the reasoning ability of LLMs. To further improve the internal reasoning ability of LLMs, Quiet-STaR (Zelikman et al., 2024) uses RLHF-based self-teaching with LLMs’ selfgenerated thoughts to improve reasoning in normal tasks and simple math problems. For more complex problems that are beyond the students’ capabilities, this think-before-reasoning pattern may not work well. In this paper, we utilize a new cross-model paradigm to enable LLMs to boost both reasoning and self-correction abilities from external model feedbacks, thereby breaking the bottleneck of original thoughts of LLMs and broadening the model’s capability to address a wider range of issues. 

## 3 PRELIMINARY 

**Reinforcement Learning from Human Feedback** Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017) is an effective approach for enhancing the robustness, factuality, and safety of LLMs (Ouyang et al., 2022). RLHF consists of three training phases: 1) supervised fine-tuning (SFT); 2) reward model training, and 3) policy model fine-tuning. **SFT Phase** : RLHF typically begins by fine-tuning a pre-trained LM with supervised learning on high-quality data for the downstream task(s) of interest (dialogue, summarization, etc.), to obtain a model _πsft_ . **Reward Modelling Phase** :given any text, the reward model will assign a scalar reward value to the last token, and the larger the reward value, the better the sample. Following Stiennon et al. (2020), training reward models often involves utilizing a dataset comprised of paired comparisons between two responses generated for the same input. The modeling loss for each pair of preferred and dis-preferred samples is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0003-06.png)


where _σ_ is the sigmoid function. _r_ represents the reward model with parameters _ψ_ , and _r_ ( _x, y_ ) is the a single scalar predicted reward for input prompt _x_ and response _y_ . However, this method is often considered complex due to the complex training pipeline. **RL Fine-Tuning Phase** : During the RL phase, the learned reward function is used to provide feedback to the language model. Following prior works (Tutor; Jaques et al., 2020), the optimization is formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0003-08.png)


where _β_ is a parameter controlling the deviation from the base reference policy _πref_ , namely the initial SFT model _πsft_ . In practice, the language model policy _πθ_ is also initialized to _πsft_ . Due to the discrete nature of language 

3 

Published as a conference paper at ICLR 2025 

generation, this objective is not differentiable and is typically optimized with reinforcement learning. The standard approach (Ziegler et al., 2019; Bai et al., 2022a; Ouyang et al., 2022) has been to construct the reward function as metioned in Equation (1), and maximize using PPO Schulman et al. (2017). 

**Direct Preference Optimization (DPO)** As an competitive alternative for traditional RLHF method, DPO (Rafailov et al., 2024) was introduced to directly leverage pair-wise preference to optimize the policy model with an equivalent optimization objective. Specifically, given an input prompt _x_ , and a preference data pair ( _y_[+] _, y[−]_ ), DPO aims to maximize the probability of the preferred output _y_[+] and minimize that of the undesirable output _y[−]_ . The optimization objective is formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0004-03.png)


where _D_ is the pair-wise preference dataset, _σ_ is the sigmoid function, _πθ_ ( _·|x_ ) is the policy model to be optimized, _πref_ ( _·|x_ ) is the reference model kept unchanged during training, and the hyperparameter _β_ controls the distance from the reference model. 

## 4 METHOD 

## 4.1 SUPERVISED FINE-TUNING WITH HIERARCHICAL THOUGHT TEMPLATE 

**Constructing Hierarchical Thought Templates from Teacher LLMs** The traditional instruction-response datasets for training LLMs (Ouyang et al., 2022) mainly focus on the correctness of the response, leading LLMs to merely simulate the provided solution and the answer, while ignoring the importance of the intermediate reasoning thought. Recent work such as BoT (Yang et al., 2024b) utilizes a high-level reasoning guideline (thought template) to enable LLMs to efficiently solve similar problems in a training-free manner. However, for complex and diverse mathematical reasoning tasks, we find that using only a high-level thought template is insufficient, especially for small-sized LLMs. To empower small LLMs to tackle complex reasoning tasks, we specifically design a **hierarchical thought template** extracted from large teacher LLMs for transfer to small student LLMs. This new hierarchical thought template comprises both _a high-level thought_ and _a detailed solution_ . The former provides a summarized and generalized solution for similar problems, while the latter offers a detailed explanation of the critical reasoning steps. 

Based on this hierarchical thought template, we can propose a new fine-tuning objective that aims to incorporate human-like hierarchical problem-solving thought structures into the model reasoning and explicitly produce hierarchical thought during reasoning process. We first collect a set _D_ = _{_ ( _x,_ ˆ _y,_ ˆ _s_ ) _}_ of mathematical problems _x_ with ground-truth answers _y_ ˆ and solution _s_ ˆ. For each problem _x ∈ D_ , we first utilize our pre-defined prompt denoted as _Ptea_ , as shown in the below text box, to extract hierarchical thought templates from teacher LLMs (e.g., SOTA LLMs like o1-preview/o1-mini). For more details about our prompt, we present all of our prompts in Appendix A. 

## _**Prompt for Extracting Hierarchical Thought Template**_ 

_Transform the solution of the following math problem into a step-by-step XML format, each step should be enclosed within tags like ⟨Step1⟩⟨/Step1⟩. For each step enclosed within the tags, determine if this step is challenging and tricky, if so, add detailed explanation and analysis enclosed within ⟨Key⟩⟨/Key⟩ in this step, as helpful annotations to make the student better understand this step correctly thus mastering the solution. After all the reasoning steps, summarize the common solution and reasoning steps to help him generalize to similar problems within ⟨Generalized⟩⟨/Generalized⟩. Finally present the final answer enclosed within ⟨Answer⟩⟨/Answer⟩._ 

Then we can obtain the high-quality fine-tuning dataset _Dsft_ as: 

_Dsft_ = _πtea_ ( _Ptea, x,_ ˆ _s_ ) = _{x, stea, Ttea, ytea|x ∈ D},_ (4) 

where _stea_ is the formalized solution steps, _Ttea_ is the hierarchical thought for the solution, and _ytea_ is the final answer extracted from _stea_ . Here we provide an example of our hierarchical thought template as shown in the below text box. For normal and easy steps, we provide brief explanation and direct solution, as for tricky and difficult reasoning steps, we provide a **detailed solution** and in-depth explanation within _⟨_ Key _⟩_ which will help student LLMs to better grasp the insight within the detailed thought. Furthermore, we provide a **high-level thought** within _⟨_ Generalized _⟩_ as a generalized guidance which helps to efficiently solve similar problems. 

**Thought-based Supervised Fine-tuning** After curating our thought-based dataset _Dsft_ , our optimization objective is to make student LLMs _π_ reasoning with hierarchical thought and have a more comprehensive understanding for each problem-solving process, which can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0004-15.png)


4 

Published as a conference paper at ICLR 2025 

Starting from the base student LLM _π_ , _L_ sft maximizes the likelihood of response ( _Ttea, stea_ ) given prompt _Pstu_ and input problem _x_ , where _Pstu_ denotes the pre-defined prompt as _Ptea_ . After the fine-tuning process, we greatly enhance the reasoning ability of base student LLMs by learning the hierarchical thought from SOTA reasoning LLMs and enable the student LLMs to produce similar hierarchical thought along with final answer. Then, we obtain fine-tuned student LLMs _πref_ that could be used for cross-model collaborative dpo in Section 4.2. 

## **Hierarchical Thought Template** 

- _⟨_ Step 1 _⟩_ 

- _⟨_ /Step 1 _⟩_ 

...... ...... 

_⟨_ Step 4 _⟩_ **Calculate the Number of Ways to Roll Exactly 2 Sixes** 

_⟨_ Key _⟩_ 

- _♠_ **Starting point of detailed solution** _♠_ **:** 

## **Understanding Combinations and Independent Events** 

The most challenging step is determining the number of ways to roll exactly two sixes. This involves two key concepts: 

1. **Combinations (** �52� **)** : This represents the number of ways to choose which two out of the five rolls will be sixes. 

2. **Independent Choices for Remaining Rolls (** 5[3] **)** : For the other three rolls that are not sixes, each has 5 possible outcomes (1 through 5). 

By combining these, the total number of ways to get exactly two sixes is: 

_♠_ **End point of detailed solution** _♠_ **:** 

_⟨_ /Key _⟩ ⟨_ /Step 4 _⟩ ⟨_ Step 5 _⟩_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0005-16.png)


_⟨_ /Step 5 _⟩_ 

_⟨_ Step 6 _⟩_ **Calculate the Probability** 

The probability of getting at most two sixes in five rolls is the ratio of the number of favorable outcomes to the total number of possible outcomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0005-20.png)


_⟨_ /Step 6 _⟩_ 

_⟨_ Generalized _⟩_ 

- _♢_ **Starting point of high-level thought** 

## **Generalizing to Similar Problems** 

When dealing with probabilities of obtaining a certain number of specific outcomes in multiple trials (like rolling a die several times), follow these steps: 

1. **Identify Total Outcomes** : Calculate the total number of possible outcomes using _n[k]_ . 

2. **Determine Favorable Outcomes** : For each desired number of specific outcomes (e.g., exactly _r_ sixes), use combinations and multiply by the number of ways the remaining trials can occur. 

3. **Sum Relevant Cases** : If the problem asks for ”at most” or ”at least,” sum the favorable outcomes accordingly. 

4. **Compute Probability** : Divide the total favorable outcomes by the total possible outcomes. 

This approach is based on the **binomial probability formula** , which is widely applicable in scenarios with independent trials. 

- _♢_ **End point of high-level thought** 

- _⟨_ /Generalized _⟩_ 

- _⟨_ Answer _⟩_ The probability of rolling a six in at most 2 of the 5 rolls is 648[625][.] _[⟨]_[/Answer] _[⟩]_ 

4.2 CROSS-MODEL COLLABORATIVE DPO 

**Boosting DPO with Thought Correction** While DPO proves to be effective in some areas (e.g., chat, style, etc.), its optimization objective is less effective for complex mathematical reasoning tasks. As noted in Lai et al. (2024), the issue arises because errors in solving complex mathematical problems often occur at the most challenging steps (e.g., complicated calculations, tricky transformations). This may lead to wrong optimization during training, as correct previous steps are also rejected. Furthermore, it is challenging for a single LLM to detect and 

5 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0006-01.png)


**----- Start of picture text -----**<br>
Input problem: Let 𝑓 𝑥= "#$%! Find the largest x which is not in the domain of g(x)=f(f(x)). Problematic Solution Example <Step 1></Step 1><Step 4>  Self-Correction…………<br>Solution: ## Step 1:  𝑔 𝑥= 𝑥−2.51 1 −5 = 𝑥−2.51 −5𝑥1 = [𝑥−2.5] 1 −5𝑥<br>Identify the function f(x) and its domain <Evaluate><br>The function 𝑓 𝑥= "#$%! is defined for all real numbers except when the denominator is zero, i.e.,2𝑥− This step is correct.</Evaluate> Error overlooked<br>5 ≠0## Step 2: . </Step 4><Step 7> ……<br>Solve for the values of x that make the denominator of f(x) zeroSolving 2𝑥−5 ≠0 gives 𝑥≠ [%] " The largest $x$ that is not in the domain of $g(x)$ is the larger of the two values found in steps 5 and 6, which is <Cause>  𝑥= !" Hallucination<br>## Step 3:  This step is incorrect because it considers the value that makes the denominator of f(x) zero,<br>Substitute f(x) into g(x) to get the expression for g(x) ## Step 4: (Error Reasoning Step) 𝑔 𝑥= 𝑓 𝑓 𝑥 = 𝑓 2𝑥−51 = 2 2𝑥−511 −5 which is not relevant to the domain of g(x).<Correction>This step should be rephrased to consider the values that make the denominator of g(x) zero, which is </Step 7>𝑥= #! The largest x that is not in the domain of g(x) is  Misled and biased by itself #!<br>Simplify the expression for g(x)<br>𝑔 𝑥= 𝑥−2.51 1 −5 = 𝑥−2.51 −5𝑥1 = [𝑥−2.5] 1 −5𝑥 <Step 1> Cross-model Correction……<br>## Step 5: Identify the values of x that make the denominator of g(x) zero </Step 1><Step 4>  ……<br>The denominator of g(x) is zero when 1-5x = 0, which gives ## Step 6: Identify the values of x that make the denominator of f(x) zero, which are also in the domain of g(x)𝑥= [!] % Simplify the expression for g(x) . <Cause> 𝑔 𝑥= 𝑥−2.51 1 −5 = 1 −5𝑥−2.5𝑥−2.51 = 1 −5𝑥−2.5𝑥−2.5 Error located<br>The values of $x$ that make the denominator of f(x) zero are  ## Step 7:Determine the largest x that is not in the domain of g(x) 𝑥= [%] " The simplification in the original reasoning incorrectly reduced simplification should account for distributing the negative sign properly.</Cause> 𝒙&𝟐.𝟓𝟏 −𝟓  to  𝒙&𝟐.𝟓 [𝟏&𝟓𝒙] The correct<br>The largest $x$ that is not in the domain of $g(x)$ is the larger of the two values found in steps 5 and 6, which is  𝑥= [%] " <Correction> 𝒈 𝒙= 𝒙−𝟐. 𝟓𝟏 𝟏 −𝟓 = 𝟏−𝟓𝒙−𝟐. 𝟓𝒙−𝟐. 𝟓𝟏 = 𝟏−𝟓𝒙−𝟐. 𝟓𝒙−𝟐. 𝟓 = 𝟏−𝟓𝒙+ 𝟏𝟐. 𝟓𝒙−𝟐. 𝟓 = 𝟏𝟑. 𝟓−𝟓𝒙𝒙−𝟐. 𝟓<br>Therefore, the final answer is: [%] " </Correction> </Step 4><Answer> …… Accurate Correction<br>"*<br></Answer> #+ Correct Answer<br>**----- End of picture text -----**<br>


Figure 2: An illustrative comparison between self-correction and our cross-model correction. Cross-model correction can enable more precise error localization and thought correction. 

correct its own errors (Tyen et al., 2024). This is akin to students struggling to gain insights from their own incorrect solutions. The root of the error lies in flawed reasoning, making it inefficient to merely imitate the correct solution without addressing the underlying thought-level mistakes. To address this, we have carefully designed novel and fine-grained optimization objectives that prioritize thought-level correction over traditional instancelevel preference. Specifically, we first accurately locate the error step and then use the correction trace of this error step as the optimization unit. This approach prioritizes cross-model correction traces from teacher LLMs _πtea_ over self-correction traces from student LLMs _πref_ , thereby enhancing the error detection and self-correction abilities of student LLMs. 

**Collecting Error Thoughts and Corrections** To achieve thought-level correction, we need to collect a dataset containing fine-grained paired data of self- and cross-correction traces. Specifically, we utilize the finetuned student LLM _πref_ to conduct thought-based reasoning on our sampled test dataset denoted as _Dtest_ = _{xtest,_ ˆ _ytest,_ ˆ _stest}_ , and we obtain the test results denoted as _πsft_ ( _xtest_ ) = _{x_ ˆ _test, stest, Ttest, ytest|xtest ∈ Dtest}_ . After filtering out erroneous problem-solution pairs that satisfy _ytest_ = _ytest_ and finally obtain the erroneous dataset: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0006-05.png)


here _serr_ is the error solution and _Terr_ is the corresponding error thought, _yerr_ represents the error answer extracted from _serr_ . Given that each erroneous solution is explicitly presented as a sequence of reasoning steps _serr_ = _s_ 1 _, s_ 2 _, . . . , sn_ , we proceed to verify the correctness of each reasoning step until we find the first error and record its step number _k_ . Here we utilize current powerful models (e.g., gpt-4o, o1-mini) in mathematical reasoning to function as an experienced teacher model _πtea_ . To obtain the corresponding error steps and cause analysis, we design a prompt _Pc_ to instruct _πtea_ to search for the logic flaws and errors in the provided reasoning steps. After searching _serr_ and evaluating each reasoning steps, we could locate each error steps and annotate each error step with error cause analysis _ai_ and correction guidance _ci_ . Thus we could obtain an annotated dataset of pair-wise self- and cross-corrections: 

_Dcorr_ = _{_ ( _x, {si}[k] i_ =0 _[−]_[1] _[,]_[ (] _[a]_[+] _k[, c]_[+] _k_[)] _[,]_[ (] _[a][−] k[, c][−] k_[)] _[,]_[ )] _[|][x][ ∈][D][err][}][,]_ (7) where _k_ denotes the first error step. Here ( _a_[+] _k[, c]_[+] _k_[)][is][chosen][as][the][corrected][step][with][analysis][from][teacher] model, ( _a[−] k[, c][−] k_[)][ is chosen as the rejected correction step and cause analysis from the student model, utilizing the] same correction prompt as the teacher. To further ensure the quality of our dataset, we additionally propose an inspector LLM to conduct iterative evaluation which verifies the accuracy of the correction trace by comparing it against the input problem and the ground-truth solution. If issues are detected, the problematic parts are sent back to the teacher LLMs for revision. This iterative checking process continues until no errors remain, with a maximum of three iterations allowed. In our implementation, we apply inspector LLM both in the curation process of HSFT dataset and pair-wise self-and corrections dataset. For more detail, please refer to Section 5.5, we also make detailed analysis of the dataset quality in Section 5.5.2. 

**Improving Self-correction Ability with Cross-model Correction** In the second stage of our method, our proposed **cross-model collaborative DPO** leverages cross-model correction from teacher LLMs to enhance the error 

6 

Published as a conference paper at ICLR 2025 

detection and self-correction ability of student LLMs. As noted in Equation (7), the previous _k −_ 1 correct reasoning steps _{si}[k] i_ =0 _[−]_[1][are combined with input problem] _[ x]_[, our cross-model collaborative DPO aims to maximize the] probability of the teacher LLM’s correction and analysis of the error step ( _a_[+] _k[, c]_[+] _k_[)][, while minimizing the proba-] bility of the student LLM’s self-correction and analysis ( _a[−] k[, c][−] k_[)][.][The optimization objective of our cross-model] collaborative DPO can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0007-02.png)


By prioritizing cross-model correction over self-correction, as illustrated in Figure 2, our method helps student model to accurately locate the erroneous steps of the mathematical reasoning process and effectively conduct selfcorrection. Furthermore, this process also helps the student LLMs to rectify its original flawed thoughts and avoid specific errors thus improving the reasoning ability and mitigate hallucination problems. 

Table 1: Quantitative comparison. Models are evaluated with chain-of-thought reasoning using open-source evaluation framework (Gao et al., 2023a) _[†]_ . ”general” denotes whether the model is for general tasks or designed for specific tasks. ”open” denotes open-source or not. 

|tasks. ”open” denotes open-source or not.|||||
|---|---|---|---|---|
|Model|size|general|open|MATH (%) GSM8K (%)|
|GPT-3.5-Turbo<br>Gemini-1.5-Pro (Reid et al., 2024)<br>Claude-3-Sonnet<br>GPT-4-1106 (Achiam et al., 2023)<br>GPT-4-Turbo-0409 (Achiam et al., 2023)<br>GPT-4o-0806|-<br>-<br>-<br>-<br>-<br>-|�<br>�<br>�<br>�<br>�<br>�|�<br>�<br>�<br>�<br>�<br>�|42.5<br>92.0<br>67.7<br>90.8<br>71.1<br>96.4<br>64.3<br>91.4<br>73.4<br>93.7<br>76.6<br>95.8|
|Llama-3-8B-Instruct (Touvron et al., 2023a)<br>Qwen2-7B-Instruct (Yang et al., 2024a)<br>Llama-3-70B-Instruct (Touvron et al., 2023a)<br>DeepSeek-Coder-V2-Instruct (Zhu et al., 2024)|8B<br>7B<br>70B<br>236B|�<br>�<br>�<br>�|�<br>�<br>�<br>�|30.0<br>79.6<br>49.6<br>82.3<br>50.4<br>93.0<br>75.7<br>94.9|
|Code-Llama-7B (Roziere et al., 2023)<br>MAmooTH-CoT (Yue et al., 2023)<br>WizardMath (Luo et al., 2023)<br>MetaMath (Yu et al., 2023)<br>MetaMath-Mistral-7B (Yu et al., 2023)<br>MathScale-Mistral Tang et al. (2024)<br>InternLM-Math-7B (Ying et al., 2024)<br>Xwin-Math-Mistral-7B (Li et al., 2024a)<br>MAmmoTH2-7B-Plus (Yue et al., 2024)<br>MathGenieLM-Mistral (Lu et al., 2024)<br>InternLM-Math-20B (Ying et al., 2024)<br>MathGenieLM-InternLM2 (Lu et al., 2024)|7B<br>7B<br>7B<br>7B<br>7B<br>7B<br>7B<br>7B<br>7B<br>7B<br>20B<br>20B|�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�|�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�|13.0<br>25.2<br>10.4<br>50.5<br>10.7<br>54.9<br>19.8<br>66.5<br>28.2<br>77.7<br>35.2<br>74.8<br>34.6<br>78.1<br>43.7<br>89.2<br>45.0<br>84.7<br>45.1<br>80.5<br>37.7<br>82.6<br>55.7<br>87.7|
|Meta-Llama3.1-8B-Instruct (Dubey et al., 2024)|8B|�|�|51.9<br>84.5|
|**SUPERCORRECT-Llama-8B (Ours)**|8B|�|�|**58.2**<br>**89.7**|
|DeepSeekMath-7B-Instruct(Shao et al., 2024)|7B|�|�|46.8<br>82.9|
|**SUPERCORRECT-DeepSeek-7B (Ours)**|7B|�|�|**54.6**<br>**88.2**|
|Qwen2.5-Math-7B-Instruct (Yang et al., 2024a)|7B|�|�|55.1<br>83.2|
|**SUPERCORRECT-Qwen-7B (Ours)**|7B|�|�|**70.2**<br>**89.5**|



> _†_ lm-evaluation: https://github.com/EleutherAI/lm-evaluation-harness. 

7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0008-01.png)


Figure 3: Comparison between different models and our SUPERCORRECT. Here we chose SUPERCORRECTQwen-7B as our model. The differences of the accuracy has been marked by arrows with different colors, red means accuracy decreased, and green means accuracy improved. 

## 5 EXPERIMENTS 

## 5.1 EXPERIMENTAL SETUP 

**Base Models, Datasets and Evaluations** We apply SUPERCORRECT to different base models to demonstrate its generalization ability and achieve new SOTA results, including recent powerful **Qwen2.5-Math-7B** (Yang et al., 2024a), **Meta-Llama3.1-8B** (Dubey et al., 2024), **DeepSeek-Math-7B** (Liu et al., 2024), these models have been recognized to be reasoning-efficient with smaller size and strong reasoning ability especially in mathematical problems. In the SFT stage, we use mathematical problems from the training set of Math (Hendrycks et al., 2021) which consists of 7500 challenging competition mathematics problems, and training set of GSM8K (Cobbe et al., 2021) consists of 7473 high quality linguistically diverse grade school math word problems. Furthermore, we additionally translated 670 challenging math problems from GaoKao Bench (Zhang et al., 2023a) which is based on Chinese 2010-2022 GAOKAO examinations. To further enrich the diversity of our dataset, we sampled some challenging problems from NuminaMath (Li et al., 2024b) and MetaMath(Yu et al., 2023). To align with our hierarchical thought reasoning process, we leverage SOTA LLMs o1-mini/gpt-4o-mini to create hierarchical thought based on the ground truth solution as mentioned in Section 4.1, and establish a hierarchical thought based dataset. In the Cross-model DPO stage, we collect 20k incorrect reasoning results from three different SFT models and processed as described in Section 4.2. For evaluation, we use the test set from **MATH** (Hendrycks et al., 2021) and **GSM8K** (Cobbe et al., 2021) datasets, and test chain-of-thought reasoning accuracy utilizing open-source evaluation framework (Gao et al., 2023a). 

**Implementation Details** We conduct our experiments on 8 NVIDIA A100-PCIE-40GB GPUs. Here we denote our hierarchical thought based supervised fine-tuning as HSFT for simplicity. Initially, we utilize the 100K HSFT data for hierarchical thought supervised fine-tuning on the base models to obtain our HSFT models. We train all of our models for 4 epochs, with training batch size set to 8 and gradient accumulation steps set to 16. The learning rate is set to 2 _e_[5] and we use AdamW optimizer along with the cosine learning rate scheduler. The warmup ratio is set to 0.02 and we use flash-attention (Dao et al., 2022) to save GPU memory. Subsequently, we perform Crossmodel DPO based on the HSFT models. For Cross-model DPO, we train for 8 epochs, with a global batch size of 128 and a learning rate of 1 _×_ 10 _[−]_[6] . And we use the AdamW optimizer along with cosine learning rate scheduler, and the warmup ratio is set to 0.05. 

## 5.2 MAIN RESULTS 

**Enhanced Reasoning Accuracy** As shown in Table 1, our method **achieves new SOTA performance among all 7B models, significantly surpassing powerful DeepSeekMath-7B by 7.8% and Qwen2.5-Math-7B by 15.1% on MATH benchmark** . This promising results demonstrates our superiority and effectiveness in handling complicated reasoning tasks. Notably, we can achieve better results than larger-sized models such as Llama370B-Instruct (Touvron et al., 2023a) in GSM8K and MATH, and achieve accuracy comparable to GPT-4o and 

8 

Published as a conference paper at ICLR 2025 

Table 2: Accuracy comparison between different methods, here we choose Qwen2.5-Math-Instruct as Base model denoted as Base and our Cross-model DPO is denoted as Cross-DPO. Here we separately compare our first HSFT stage with traditional SFT method and Cross-DPO stage with Reflexion(Shinn et al., 2024). We show the improved accuracy in green compare to previous methods. We provide quantitative results with more base LLMs (i.e., Llama3.1 and DeepSeek-Math) in Table 7 of Section 5.6. 

|**Model**|Base|Base + SFT|Base + HSFT|Base-HSFT + Refexion(Shinn et al., 2024)|Base-HSFT + Cross-DPO|
|---|---|---|---|---|---|
|**MATH**(%)|55.1|57.4|**62.4**(+5.0)|63.1|**70.2**(+7.1)|
|**Model**|Base|Base + SFT|Base + HSFT|Base-HSFT + Refection|Base-HSFT + Cross-DPO|
|**GSM8K**(%)|83.2|84.3|**87.2**(+2.9)|86.8|**89.5**(+2.7)|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0009-03.png)


Figure 4: Improvement comparison between different topics. Here we chose Qwen2.5-Math-7B-Instruct and our SUPERCORRECT-Qwen-7B to show the improvement in performance of different mathematical problem Types. The part in green is the improved part of our SUPERCORRECT, and the part in black is the original reasoning accuracy of Qwen2.5-Math-7B-Instruct. 

GPT-4o-mini with our best model SUPERCORRECT-Qwen-7B. We attribute this improvement in reasoning accuracy in two folds: 1) The first HSFT stage that equips student LLMs with a deeper and fine-grained reasoning process. Compare to conventional CoT reasoning process, it helps the student LLMs to think more carefully thus improving the reasoning consistency and reduce hallucinations issues on the problems that the student LLMs already mastered. 2) The second cross-model DPO stage that leverages the error-driven insights from teacher LLM to help student LLMs break the bottleneck of their thoughts thus making it possible to deal with the problems that the student LLMs in acquiring the skills and knowledge to tackle problems they were previously unable to solve. We also present some detailed examples of hierarchical reasoning in Appendix B from different datasets, please check them to have a comprehensive understanding of our SUPERCORRECT. 

**Improved Self-Correction Ability** Here we also show the improved self-correction ability of our SUPERCORRECT as shown in Figure 3. After initial reasoning stage, we let all the LLMs to verify the reasoning process and detect the logic flaws and errors within each reasoning step, and try to correct them. As a result of self-correction, our SUPERCORRECT further increase the accuracy by 5 _∼_ 6%, while other LLMs are ineffective to increase the accuracy, and some LLMs even decrease the original accuracy. Because our Cross-model DPO helps the LLMs to accurately locate the errors and logic flaws within each steps by learning teacher’s correction traces, and use a fine-grained analysis and correction to help LLMs better correct them. After the Cross-model DPO process, the LLMs are not only able to consistently solve problems within its capabilities, but they are also able to solve wider range of problems with error-driven insights gained from teacher LLMs. We provide more quantitative analysis in Table 6 on how far cross-model DPO brings the student model and the teacher model closer to each other. We also provide some self-correction examples from different datasets, for more detail, please check Appendix C. 

**Ablation Study** We conduct ablation study of our SUPERCORRECT and put results in Table 2. As we can see, the improvement of traditional SFT is limited compare to our HSFT, which falls behind by 5% in accuracy. Based on our HSFT models, we further apply some self-correction methods such as Reflexion (Shinn et al., 2024) to compare with our Cross-DPO. From the results, we can find that our method wins again with lead of 7% in 

9 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Higher<br>Mean<br>Lower<br>Variance<br>**----- End of picture text -----**<br>


Figure 5: Quantitative analysis on reasoning stability. The higher mean value denotes higher average accuracy rate, and lower variance denotes higher reasoning stability. 

accuracy compare to Reflexion. These promising results demonstrate the effectiveness of our HSFT and crossmodel DPO. Here we take an illustrative example in Table 3 of Section 5.3 for better understanding of our effective hierarchical thought reasoning. The CoT prompting method shows misunderstanding of ”empty set” as it fails to account for the fact that the 512 sets already include the empty set. Equipped with our hierarchical thought-based reasoning (denoted as HT in Appendix A), we can see that the model realizes that the 512 sets include empty set. However, it fails to correctly recall the fact that the problem requires to include the empty set in the final answer, which is caused by hallucination issue. Finally, our HSFT LLMs could correctly resolve the problem with accurate understanding of empty set and avoid the hallucination issue. 

**SupperCorrect Breaks Thought Bottleneck** The problems within MATH dataset encompass a wide range of seven topics including algebra, counting & probability, intermediate algebra, number theory, geometry, prealgebra and precalculus. During our experiments, we observe that the accuracy for each topics are quiet different. For most LLMs, they tend to show better performance on algebra and prealgebra, but for other topics, it always show degradation in accuracy because they may have some thought bottleneck on those topics. As shown in Figure 4, our SUPERCORRECT improves the reasoning performance on all topics. It is noted that for the topics which are originally difficult for LLMs, it shows a more significant improvement compare to topics that the models are already mastered. This is because we utilize the error-driven insights during the Cross-model DPO stage to break the original thought bottleneck of LLMs, thus enlightening them with new techniques and tricks to solve the problems that they used have no idea to solve. The results further proves that our SUPERCORRECT could help to break the original thought bottleneck thus significantly improve the reasoning ability of LLMs, and narrowing the performance gap for different topics. More detail reasoning and self-correction results can be found in Appendix B. and Appendix C. 

**SuperCorrect Achieves Better Reasoning Stability** The test set of MATH dataset consists of 5000 problems in 5 different difficulty levels. To further evaluate the reasoning stability of our method, we additionally sample 300 problems of level-5 (hardest) from MATH test dataset. We conduct a quantitative analysis by repeating the experiment 256 times and compute the mean and variance of accuracy as shown in Figure 5. We can observe that, compare to the base model, our SUPERCORRECT helps to achieve higher mean value of accuracy rate. Moreover, our SUPERCORRECT significantly reduce the variance of accuracy distribution of multiple reasoning times. These phenomenons demonstrate our SUPERCORRECT can effectively improve both accuracy and stability for difficult reasoning problems. 

## 5.3 DETAILED QUALITATIVE ANALYSIS 

In this section, we provide a detailed comparison for error-prone reasoning steps and reasoning results between three different methods, including CoT prompting, our first-stage HSFT models and our SUPERCORRECT. 

10 

Published as a conference paper at ICLR 2025 

Table 3: Qualitative comparison between error-prone steps for different methods. Here we use different colors to represents different parts of reasoning. We denote the erroneous reasoning steps in purple, the error cause in red, correct reasoning steps that show improvement in black and the summary for improvement in green. 

|Problem|How many subsets of the set of divisors of 72 contain only composite num-|
|---|---|
||bers? For example, _{_8,9_}_ and _{_4,8,12_}_ are two such sets. **Include the**|
||**empty set in your count.** (The error-prone parts within the problem)|
|CoT|## Step 4: Account for the empty set. We need to include the empty set in|
||our count, so we add 1 to the total number of subsets.|
||(Error Cause: Ignore the fact that empty set has already been counted)|
||## Step 5: Calculate the fnal answer. The total number of subsets of the|
||composite numbersis29 + 1 = 512 + 1 = 513. �|
||(Error Cause: Repeatedly counted empty set)|
|HSFT|_⟨_Step4_⟩_|
||...... which is29 = 512.|
||However, this includes the empty set, which is not a subset of the set of|
||composite numbers.|
||(Improvement:Notice the empty set is included within the 512 subsets)|
||_⟨_/Step4_⟩_|
||_⟨_Step5_⟩_|
||We need to subtract 1 from 512 to exclude the empty set. Therefore, the|
||total number of subsets of the set of composite numbersis512_−_1 = 511.|
||�|
||_⟨_/Step5_⟩_|
||(Error Cause: Hallucination issue makes LLMs forget the fact that it is|
||required to count empty set in the problem.)|
|**SUPERCORRECT**||
||_⟨_Step6_⟩_|
||Calculate the number of subsets containing only composite numbers.|
||_⟨_Key_⟩_|
||The number of subsets of a set with n elements is 2_n_. Here, with 9 com-|
||posite divisors, the number of subsets is29 = 512. This includes the empty|
||set, which is required by the problem statement. �|
||_⟨_Key_⟩_|
||_⟨_/Step6_⟩_|
||Improvement: 1. Successfully notice the empty set is included within 512|
||subset 2. Mitigate the infuence of hallucination issue|



## 5.4 COMPARISON BETWEEN STEP-DPO AND CROSS-MODEL DPO 

We conduct qualitative analysis between Step-DPO and our Cross-model DPO. We choose Qwen2.5-Math-Instruct as base model, and we apply Step-DPO on the base model to compare the results. It should be noted that Step-DPO utilize CoT style prompt, for fair comparison, we choose the most suitable prompting method for each model. As shown in Table 4, based on the previous unsolved problems, Step-DPO could locate the error reasoning steps and make corrections (e.g. further identify another multiples of 7), but it but struggles to fully correct them. Compare to Step-DPO, our method could not only locate the erroneous steps, but also conduct accurate self-correction thus solving previous unsolvable problems. 

11 

Published as a conference paper at ICLR 2025 

Table 4: Qualitative comparison between Step-DPO and Cross-model DPO. 

|Problem<br>How many of the following numbers are factors of 34 or multiples of 7?<br>Numbers: 1, 2, 3, 4, 8, 14, 17, 29, 56, 91|Problem<br>How many of the following numbers are factors of 34 or multiples of 7?<br>Numbers: 1, 2, 3, 4, 8, 14, 17, 29, 56, 91|Problem<br>How many of the following numbers are factors of 34 or multiples of 7?<br>Numbers: 1, 2, 3, 4, 8, 14, 17, 29, 56, 91|
|---|---|---|
|Base Model<br>### Step 1: Factors of 34 The prime factorization of 34 is:<br>34 = 2_×_17<br>The factors of 34 are:<br>1_,_2_,_17_,_34<br>......<br>### Step 4: Identify multiples of 7 from the list From the provided list, the<br>multiples of 7 are: - 14 (since7_×_2 = 14)<br>(Error Cause: Ignore the fact that 56,91 is also multiples of 7)<br>......<br>### Step 7: Count the unique numbers The unique numbers that are either<br>factors of 34 or multiples of 7 are: -1_,_2_,_14_,_17<br>Thus, the total count is:<br>|||
||4|�|
|Step-DPO<br>...... ### Step 2: Identify Multiples of 7 The multiples of 7 are the numbers<br>that can be expressed as 7_× k_, where _k_ is an integer. Among the given<br>numbers, the multiples of 7 are: -14 = 7_×_2-91 = 7_×_13<br>Thus, the multiples of 7 from the list are:<br>14_,_91<br>(Error Cause: Still unable to fully recognize all the multiples of 7, missing<br>56 here)<br>### Step 3: Combine the Results Now, we combine the factors of 34 and<br>the multiples of 7, ensuring not to double-count any numbers. The unique<br>numbers that are either factors of 34 or multiples of 7 are: - Factors of 34:<br>1_,_2_,_17- Multiples of 7: 14_,_91<br>Combining these, we have:<br>1_,_2_,_14_,_17_,_91<br>### Conclusion Counting the unique numbers, we fnd that there are:<br>5<br>�|||
|**Cross-model DPO**<br>......<br>_⟨_Step2_⟩_<br>Identify the multiples of 7 from the given list.<br>_⟨_Key_⟩_<br>Multiples of 7 are numbers that can be expressed as 7 multiplied by an<br>integer. It’s important to check each number’s divisibility by 7 to determine<br>if it qualifes.<br>Numbers in the list: 1, 2, 3, 4, 8, 14, 17, 29, 56, 91<br>Multiples of 7: 14, 56, 91<br>(Self-Correction: Correctly identifes all the multiples of 7)<br>_⟨_/Step2_⟩_<br>_⟨_Step4_⟩_<br>Count the total number of unique numbers identifed.<br>_⟨_Key_⟩_<br>Counting accurately ensures the fnal answer refects the correct quantity of<br>numbers that meet the criteria.<br>Total numbers: 6<br>_⟨_/Step4_⟩_<br>......<br>_⟨_Answer_⟩_6_⟨_/Answer_⟩_✓|||



12 

Published as a conference paper at ICLR 2025 

## 5.5 QUALITY EVALUATION FOR TEACHER LLM GENERATED CONTENT 

## 5.5.1 EVALUATION OF INSPECTOR LLM 

We discuss the effectiveness of inspector LLM which further ensures the quality of the generated content of Teacher LLMs. As shown in Table 5, we compare the correctness of correction traces generated by three different teacher LLMs across three datasets. The application of the Inspector LLM significantly improves the quality of the final correction traces compared to direct generation. Notably, for LLMs with advanced capabilities that already produce high-quality outputs, it still shows clear improvements. These results demonstrate that the Inspector LLM markedly enhances the accuracy of correction traces, especially for datasets where initial performance was lower. 

Table 5: Quantitative analysis of inspector LLM regarding the correctness of correction traces on various datasets. 

|**Model/Dataset**|**GSM8K**|**MATH**|**GaoKao**|
|---|---|---|---|
|Teacher LLM(GPT-4o-mini)|100%|92.4%|89.6%|
|Teacher LLM(GPT-4o-mini)+ Inspector LLM(o1-preview)|100%|**98.8%**|**96.2%**|
|Teacher LLM(GPT-4o)|100%|94.4%|91.3%|
|Teacher LLM(GPT-4o)+ Inspector LLM(o1-preview)|100%|**99.2%**|**97.5%**|
|Teacher LLM(o1-mini)|100%|98.2%|94.8%|
|Teacher LLM(o1-mini)+ Inspector LLM(o1-preview)|100%|**99.6%**|**98.7%**|



## 5.5.2 ANALYSIS ON THE QUALITY OF DIRECT GENERATION 

Based on the results in Table 5, the experimental results without the Inspector LLM demonstrate that our directly generated correction traces are already of high quality. We attribute this to our design approach, as outlined below: 

- **1. Leveraging Frontier Teacher LLMs:** To ensure the quality of content generated by the teacher LLM, we utilize state-of-the-art LLMs, specifically o1-mini, as the teacher LLM. These models are capable of identifying logical flaws and errors, and they generate high-quality analysis and corrections, as evidenced by the quantitative results. 

- **2. Grounding Correction Traces with Ground-Truth Context:** To ensure the accuracy of the correction traces generated by the teacher LLM, as demonstrated in Appendix A, the prompts for generating analysis ( _ai_ ) and correction ( _ci_ ) are based on the input question along with the ground-truth solution. This approach grounds the correction trace with the ground-truth solution as context, thereby ensuring the accuracy of the generated content. 

## 5.6 MORE ABLATION STUDIES 

**Further Analysis on Cross-model DPO** We first sample 500 erroneous solutions from our dataset, and we use o1-mini to conduct correction trace on the dataset as the ground truth to measure the model alignment (Xu et al., 2022; Zhang et al., 2023b; Khope & Elias, 2022; Guo et al., 2022). We conduct our experiments on three different models after HSFT stage, as shown in Table 6. We additionally introduce two metrics to evaluate the effectiveness of our Cross-model DPO: (1) **Locate correctness** : representing whether the model correctly finds the error steps. (2) **Correction accuracy** : representing whether the model accurately corrects the error steps. We utilize o1-preview as a judger to compare each correction trace generated by the models after Cross-model DPO with the ground truth. From the results, our cross-model DPO shows significant improvements across all models, demonstrating its effectiveness. 

Table 6: Quantitative analysis on the effectiveness of our Our Cross-model DPO. 

|**Model/Metric**|**Locate correctness**|**Correction accuracy**|
|---|---|---|
|Meta-Llama-3.1 + HSFT|0.31|0.08|
|Meta-Llama-3.1 + HSFT + Cross-model DPO|**0.49**|**0.27**|
|DeepSeek + HSFT|0.23|0.07|
|DeepSeek + HSFT + Cross-model DPO|**0.42**|**0.23**|
|Qwen2.5-Math + HSFT|0.43|0.12|
|Qwen2.5-Math + HSFT+ Cross-model DPO|**0.67**|**0.46**|



13 

Published as a conference paper at ICLR 2025 

**Ablation Study with More Base LLMs** As shown in Table 7. The result shows that our SuperCorrect can generalize to different LLM architectures, and consistently achieves better performance in both HSFT stage and Cross-model DPO stage, further validating our effectiveness. 

Table 7: Ablation study with more base LLMs on MATH and GSM8K. Base1: Llama3.1, Base2: DeepSeek-Math. 

|Model|Base1|Base1 + SFT|Base1 + HSFT|Base1-HSFT + Refexion|Base1-HSFT + Cross-DPO|
|---|---|---|---|---|---|
|MATH (%)<br>GSM8K(%)|51.9<br>84.5|53.7<br>86.2|**55.4**<br>**87.2**|56.7<br>86.8|**58.2**<br>**89.7**|
|Model|Base2|Base2 + SFT|Base2 + HSFT|Base2-HSFT + Refexion|Base2-HSFT + Cross-DPO|
|MATH (%)<br>GSM8K(%)|46.8<br>82.9|49.2<br>84.5|**50.9**<br>**85.7**|51.2<br>85.8|**54.6**<br>**88.2**|



**Ablation Study on Prompt Style** To further evaluate the effectiveness of our meticulously designed hierarchical thought template, we additionally conduct quantitative experiments to show the impact of prompt styles and our hierarchical prompt design. Here we use five prompt styles: 1) CoT 2) CoT + Hierarchical Prompt (without generalization step) 3) CoT + Hierarchical Prompt (with generalization step) 4) Our hierarchical prompt (Not in XML) 5) Our hierarchical prompt (XML). We additionally curated four datasets based on the same 100k math problems with the first four prompt styles. We then trained Qwen2.5-Math-Instruct, Llama3.1-8B-Instruct and DeepSeek-Math-7B on these dataset with the same training settings and evaluate the accuracy on Math dataset. As shown in Table 8, the experimental results indicate that hierarchical reasoning significantly improves model accuracy compared to using CoT as a baseline. Additionally, changing the prompt style (e.g., to XML format) has a small impact on the final accuracy, further demonstrating the effectiveness of our hierarchical reasoning design. Although adding generalization steps helps the model better summarize tasks and thereby enhances its performance, our experimental results indicate that the primary contribution to performance improvements in the HSFT stage comes from the hierarchical reasoning style we designed. 

Table 8: Ablation study with different prompt styles. **H** denotes with hierarchical reasoning style and **Gen** denotes with generalization step. 

|**Models/Prompt Style**|**CoT**|**CoT + H(No Gen)**|**CoT + H(With Gen)**|**Ours(Not XML)**|**Ours(XML)**|
|---|---|---|---|---|---|
|**Qwen2.5-Math-7B**|57.4|59.7|60.8|61.8|62.4|
|**Llama3.1-8B**|52.6|53.3|53.6|53.7|54.1|
|**DeepSeek-Math-7B**|46.8|49.6|50.2|50.6|51.6|



## 6 CONCLUSION 

In this paper, we propose SUPERCORRECT, a novel two-stage framework that significantly improve both reasoning and reflection processes of language models. In SUPERCORRECT, We propose hierarchical thought-based fine-tuning to enable LLMs to produce more fine-grained reasoning thoughts and introduce cross-model collaborative DPO to enhance the self-correction abilities of the student LLMS by following the teacher’s correction traces. Extensive experiments consistently demonstrate our superiority over previous methods, surpasses powerful DeepSeekMath-7B by 5.3% _∼_ 7.8% and Qwen2.5-Math-7B by 6.3% _∼_ 15.1% on MATH and GSM8K benchmarks. For future work, we will generalize this new framework to larger models and more complex datasets. 

## ACKNOWLEDGEMENT 

This work is supported by National Natural Science Foundation of China (U23B2048, U22B2037), Beijing Municipal Science and Technology Project (Z231100010323002), research grant No. SH2024JK29 and Highperformance Computing Platform of Peking University and in part by NUS Start-up Grant A-0010106-00-00. 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

14 

Published as a conference paper at ICLR 2025 

- Simran Arora, Avanika Narayan, Mayee F Chen, Laurel Orr, Neel Guha, Kush Bhatia, Ines Chami, and Christopher Re. Ask me anything: A simple strategy for prompting language models. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_ , 2022a. 

- Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. _arXiv preprint arXiv:2212.08073_ , 2022b. 

- Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pp. 17682–17690, 2024. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Pinzhen Chen, Zhicheng Guo, Barry Haddow, and Kenneth Heafield. Iterative translation refinement with large language models. _arXiv preprint arXiv:2306.03856_ , 2023a. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _arXiv preprint arXiv:2211.12588_ , 2022. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _Transactions on Machine Learning Research_ , 2023b. 

- Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. _Advances in neural information processing systems_ , 30, 2017. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. _Advances in Neural Information Processing Systems_ , 35:16344–16359, 2022. 

- Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 320–335, 2022. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- L Gao, J Tow, B Abbasi, S Biderman, S Black, A DiPofi, C Foster, L Golding, J Hsu, A Le Noac’h, et al. A framework for few-shot language model evaluation, 12 2023. _URL https://zenodo. org/records/10256836_ , 7, 2023a. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023b. 

- Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. _arXiv preprint arXiv:2309.17452_ , 2023. 

- Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_ , 2025. 

- Lihua Guo, Dawu Chen, and Kui Jia. Knowledge transferred adaptive filter pruning for cnn compression and acceleration. _Science China. Information Sciences_ , 65(12):229101, 2022. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

15 

Published as a conference paper at ICLR 2025 

- Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. _arXiv preprint arXiv:2310.01798_ , 2023. 

- Natasha Jaques, Judy Hanwen Shen, Asma Ghandeharioun, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Shane Gu, and Rosalind Picard. Human-centric dialog training via offline reinforcement learning. _arXiv preprint arXiv:2010.05848_ , 2020. 

- Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- Sarika R Khope and Susan Elias. Critical correlation of predictors for an efficient risk prediction framework of icu patient using correlation and transformation of mimic-iii dataset. _Data Science and Engineering_ , 7(1):71–86, 2022. 

- Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. _arXiv preprint arXiv:2406.18629_ , 2024. 

- Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng. Common 7b language models already possess strong math capabilities. _arXiv preprint arXiv:2403.04706_ , 2024a. 

- Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. 2024b. 

- Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, and Tianyi Zhou. Reflection-tuning: Recycling data for better instruction-tuning. In _NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following_ , 2023. 

- Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Jiuxiang Gu, and Tianyi Zhou. Selective reflection-tuning: Student-selected data recycling for llm instruction-tuning. _arXiv preprint arXiv:2402.10110_ , 2024c. 

- Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. _arXiv preprint arXiv:2405.04434_ , 2024. 

- Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang, Weikang Shi, Junting Pan, Mingjie Zhan, and Hongsheng Li. Mathgenie: Generating synthetic data with question back-translation for enhancing mathematical reasoning of llms. _arXiv preprint arXiv:2402.16352_ , 2024. 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _arXiv preprint arXiv:2308.09583_ , 2023. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Ning Miao, Yee Whye Teh, and Tom Rainforth. Selfcheck: Using llms to zero-shot check their own step-by-step reasoning. _arXiv preprint arXiv:2308.00436_ , 2023. 

- Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. Skeleton-of-thought: Large language models can do parallel decoding. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730–27744, 2022. 

16 

Published as a conference paper at ICLR 2025 

- Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 5687–5711, 2023. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ , 2023. 

- William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Selfcritiquing models for assisting human evaluators. _arXiv preprint arXiv:2206.05802_ , 2022. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_ , 2017. 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_ , 2024. 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. _Advances in Neural Information Processing Systems_ , 33:3008–3021, 2020. 

- Zhengyang Tang, Xingxing Zhang, Benyou Wan, and Furu Wei. Mathscale: Scaling instruction tuning for mathematical reasoning. _arXiv preprint arXiv:2403.02884_ , 2024. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023a. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023b. 

- Sequence Tutor. Conservative fine-tuning of sequence generation models with kl-control natasha jaques, shixiang gu, dzmitry bahdanau, jos´e miguel hern´andez-lobato, richard e. _Turner, Douglas Eck arXiv (2016-11-09) https://arxiv. org/abs/1611.02796 v9_ . 

- Gladys Tyen, Hassan Mansoor, Victor C˘arbune, Yuanzhu Peter Chen, and Tony Mak. Llms cannot find reasoning errors, but can correct them given the error location. In _Findings of the Association for Computational Linguistics ACL 2024_ , pp. 13894–13908, 2024. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Yuemei Xu, Han Cao, Wanze Du, and Wenqing Wang. A survey of cross-lingual sentiment analysis: Methodologies, models and evaluations. _Data Science and Engineering_ , 7(3):279–299, 2022. 

- An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. _arXiv preprint arXiv:2409.12122_ , 2024a. 

- Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. Buffer of thoughts: Thought-augmented reasoning with large language models. _Advances in Neural Information Processing Systems_ , 2024b. 

- Ling Yang, Zhaochen Yu, Bin Cui, and Mengdi Wang. Reasonflux: Hierarchical llm reasoning via scaling thought templates. _arXiv preprint arXiv:2502.06772_ , 2025. 

17 

Published as a conference paper at ICLR 2025 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, et al. Internlm-math: Open math large language models toward verifiable reasoning. _arXiv preprint arXiv:2402.06332_ , 2024. 

- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. _arXiv preprint arXiv:2309.12284_ , 2023. 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_ , 2023. 

- Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. _arXiv preprint arXiv:2405.03548_ , 2024. 

- Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D Goodman. Quiet-star: Language models can teach themselves to think before speaking. _arXiv preprint arXiv:2403.09629_ , 2024. 

- Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. _arXiv preprint arXiv:2305.12474_ , 2023a. 

- Zhao Zhang, Yong Zhang, Da Guo, Shuang Zhao, and Xiaolin Zhu. Communication-efficient federated continual learning for distributed learning system with non-iid data. _Science China Information Sciences_ , 66(2):122102, 2023b. 

- Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. _arXiv preprint arXiv:2406.11931_ , 2024. 

- Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. _arXiv preprint arXiv:1909.08593_ , 2019. 

18 

Published as a conference paper at ICLR 2025 

- A ADDITIONAL PROMPTING DETAILS 

## **Prompt for Extracting Hierarchical Thought Template** 

Transform the solution of the following math problem into a step-by-step XML format, each step should be enclosed within tags like _⟨_ Step1 _⟩⟨/_ Step1 _⟩_ . For each step enclosed within the tags, determine if this step is challenging and tricky, if so, add detailed explanation and analysis enclosed within _⟨_ Key _⟩⟨/_ Key _⟩_ in this step, as helpful annotations to make the student better understand this step correctly thus mastering the solution. After all the reasoning steps, summarize the common solution and reasoning steps to help him generalize to similar problems within _⟨_ Generalized _⟩⟨/_ Generalized _⟩_ . Finally present the final answer enclosed within _⟨_ Answer _⟩⟨/_ Answer _⟩_ . 

## **Hierarchical Thought-based Reasoning Prompt (HT):** 

Solve the following math problem in a step-by-step XML format, each step should be enclosed within tags like _⟨_ Step1 _⟩⟨/_ Step1 _⟩_ . For each step enclosed within the tags, determine if this step is challenging and tricky, if so, add detailed explanation and analysis enclosed within _⟨_ Key _⟩⟨/_ Key _⟩_ in this step, as helpful annotations to help you thinking and remind yourself how to conduct reasoning correctly. After all the reasoning steps, summarize the common solution and reasoning steps to help you and your classmates who are not good at math generalize to similar problems within _⟨_ Generalized _⟩⟨/_ Generalized _⟩_ . Finally present the final answer within _⟨_ Answer _⟩⟨/_ Answer _⟩_ . 

## **Grounded Correction Trace Prompt:** 

First transform the Reasoning steps to be Checked into our required XML format as follow: for each step, the steps should be within corresponding tags like _⟨_ Step1 _⟩⟨/_ Step1 _⟩_ , and next based on the problem and reference solution, evaluate each steps and find the fundamental logic flaws and errors in the given reasoning steps, if error detected, using _⟨_ Cause _⟩⟨/_ Cause _⟩_ to give a Refined and Concise explanation for the error cause within the corresponding Step tags along with _⟨_ Correction _⟩⟨/_ Correction _⟩_ to correct the error step and output correct step. And finally, present the correct final answer within _⟨_ Answer _⟩⟨/_ Answer _⟩_ . Output All the transformed reasoning steps from _⟨_ Step1 _⟩⟨/_ Step1 _⟩_ . 

## **Correction Trace Prompt:** 

First transform the Reasoning steps to be Checked into our required XML format as follow: for each step, the steps should be within corresponding tags like _⟨_ Step1 _⟩⟨/_ Step1 _⟩_ , and next based on the problem, evaluate each steps and find the fundamental logic flaws and errors in the given reasoning steps, if error detected, using _⟨_ Cause _⟩⟨/_ Cause _⟩_ to give a Refined and Concise explanation for the error cause within the corresponding Step tags along with _⟨_ Correction _⟩⟨/_ Correction _⟩_ to correct the error step and output correct step. And finally, present the correct final answer within _⟨_ Answer _⟩⟨/_ Answer _⟩_ . Output All the transformed reasoning steps from _⟨_ Step1 _⟩⟨/_ Step1 _⟩_ . 

As shown above, we present our meticulously designed prompt template used in our experiments. The prompt for extracting hierarchical thought template is designed for teacher LLMs to transform the original solution into hierarchical thought template. And for hierarchical thought-based reasoning prompt denoted as HT, we utilize this prompt during the HSFT process and the evaluation process. Grounded correction trace prompt is also designed for teacher LLMs to locate and find the error-driven insight from the erroneous reasoning process. And finally, the correction trace prompt is used during our Cross-DPO stage, and further evaluation for self-correction. 

## B RESULTS OF HIERARCHICAL THOUGHT-BASED REASONING 

In this section, we show more detailed hierarchical reasoning process produced by SUPERCORRECT-Qwen-7b on three datasets, including GaoKao, MATH, GSM8K. For each dataset, we present two samples for demonstration. To better present the hierarchical thought during the reasoning process, we denote detailed thought within each step in black, the high-level generalized thought in purple. 

19 

Published as a conference paper at ICLR 2025 

**I GaoKao problem:** Given the function _x_ + 1 _, x ≤_ 0 _f_ ( _x_ ) = �2 _[x] , x >_ 0 what is the range of values of _x_ that satisfies _f_ ( _x_ ) + _f_ � _x −_[1] 2 � _>_ 1? **Solution Description:** _⟨_ Step 1: _⟩_ Determine the value of _f_ ( _x_ ) based on the value of _x_ . _⟨_ key _⟩_ If _x ≤_ 0, then _f_ ( _x_ ) = _x_ + 1. If _x >_ 0, then _f_ ( _x_ ) = 2 _[x]_ . _⟨_ /key _⟩ ⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Determine the value of _f_ � _x −_ 2[1] � based on the value of _x_ . _⟨_ key _⟩_ If _x −_[1] 2 _[≤]_[0][ (i.e.,] _[x][≤]_[1] 2[),][then] _[ f]_ � _x −_[1] 2 � = � _x −_[1] 2 � + 1 = _x_ +[1] 2[.][If] _[ x][ −]_[1] 2 _[>]_[0][ (i.e.,] _[x][>]_[1] 2[),][then] _f_ � _x −_[1] 2 � = 2 _[x][−]_ 2[1] . _⟨_ /key _⟩ ⟨_ /Step 2: _⟩ ⟨_ Step 3: _⟩_ Analyze the inequality _f_ ( _x_ ) + _f_ � _x −_[1] 2 � _>_ 1 in different intervals of _x_ . _⟨_ key _⟩_ This crucial step involves breaking down different cases based on the value of _x_ to apply the correct piecewise definitions of the function. _⟨_ /key _⟩ ⟨_ /Step 3: _⟩ ⟨_ Step 4: _⟩_ Combine the solutions from all cases. _⟨_ key _⟩_ 

We find ranges from each case to generate a comprehensive final solution, ensuring we encompass all possible intervals of _x_ . _⟨_ /key _⟩_ 

_⟨_ /Step 4: _⟩_ 

_⟨_ Generalized _⟩_ 

To solve inequalities involving piecewise functions, follow these common steps: 

1. **Identify the different cases** based on the definitions of the piecewise function. 

2. **Evaluate the function expressions** for each case, including any shifted or transformed inputs. 

3. **Set up and solve the inequality** within each interval determined by the cases. 

4. **Combine all valid solutions** from each case to form the complete solution set. This systematic approach ensures accurate and comprehensive solutions to similar problems involving piecewise-defined functions and inequalities. 

_⟨_ /Generalized _⟩_ 

_⟨_ Answer: _⟩ x > −_[1] 4 _⟨_ /Answer: _⟩_ 

20 

Published as a conference paper at ICLR 2025 

## **II GaoKao Problem:** 

The secondary product rate of a certain product is 0.02. If you randomly select one from this product, it has been placed back into the pool 100 times. Let _X_ represent the number of secondary products collected, then _D_ ( _X_ ) = ? 

## **Solution Description:** 

_⟨_ Step 1: _⟩_ Identify the probability of selecting a secondary product. _⟨_ key _⟩_ Understanding the probability _p_ is crucial as it forms the basis for calculating the variance. In this case, the secondary product rate is given as 0 _._ 02, which means there’s a 2% chance of selecting a secondary product each time. _⟨_ /key _⟩ ⟨_ /Step 1: _⟩_ 

_⟨_ Step 2: _⟩_ Determine the number of trials. _⟨_ key _⟩_ 

Recognizing the number of trials _n_ is essential. Here, the product has been placed back into the pool 100 times, indicating that each selection is independent and the total number of trials is 100. _⟨_ /key _⟩ ⟨_ /Step 2: _⟩_ 

_⟨_ Step 3: _⟩_ Calculate the expected value _E_ ( _X_ ) of the number of secondary products collected. _⟨_ key _⟩_ The expected value _E_ ( _X_ ) for a binomial distribution is calculated using the formula _E_ ( _X_ ) = _n × p_ . Substituting the given values: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0021-08.png)


_⟨_ /key _⟩ ⟨_ /Step 3: _⟩_ 

_⟨_ Step 4: _⟩_ Calculate the variance _D_ ( _X_ ) of the number of secondary products collected. _⟨_ key _⟩_ 

The variance _D_ ( _X_ ) for a binomial distribution is given by _D_ ( _X_ ) = _n×p×_ (1 _−p_ ). Plugging in the values: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights_images/iclr_2025_SuperCorrect_Supervising_and_Correcting_Language_Models_with_Error-Driven_Insights.pdf-0021-12.png)


_⟨_ /key _⟩ ⟨_ /Step 4: _⟩_ 

_⟨_ Generalized: _⟩_ To solve similar problems involving the number of successes in a fixed number of independent trials with a constant probability of success: 

1. **Identify** the probability of success _p_ and the number of trials _n_ . 

2. **Calculate** the expected value _E_ ( _X_ ) = _n × p_ . 3. **Determine** the variance _D_ ( _X_ ) = _n × p ×_ (1 _− p_ ). This approach uses the properties of the binomial distribution, which is applicable when each trial has two possible outcomes (success or failure) and the probability of success remains constant. _⟨_ /Generalized: _⟩_ 

_⟨_ Answer: _⟩_ The variance _D_ ( _X_ ) of the number of secondary products collected is 1 _._ 96. _⟨_ /Answer: _⟩_ 

21 

Published as a conference paper at ICLR 2025 

## **III MATH Problem:** 

What is the tens digit in the sum 

7! + 8! + 9! + _· · ·_ + 2006!? 

## **Solution Description:** 

_⟨_ Step 1: _⟩_ Identify the problem: We need to find the tens digit in the sum 7! + 8! + 9! + _· · ·_ + 2006!. _⟨_ key _⟩_ Understanding the problem is crucial. We are looking for a specific digit in a large sum, which means we need to focus on the contributions of each factorial to the tens place. _⟨_ /key _⟩ ⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Recognize that for _n ≥_ 10, _n_ ! is divisible by 100. _⟨_ key _⟩_ This is a key insight because it means that 10! _,_ 11! _,_ 12! _, . . . ,_ 2006! do not affect the tens digit of the sum. Only the factorials from 7! to 9! will influence the tens digit. _⟨_ /key _⟩ ⟨_ /Step 2: _⟩ ⟨_ Step 3: _⟩_ Calculate 7!, 8!, and 9!: 7! = 5040 _,_ 8! = 40320 _,_ 9! = 362880 

_⟨_ key _⟩_ 

Calculating factorials can be tedious, but it’s essential to ensure accuracy. Each factorial contributes to the overall sum, and we need their last two digits to determine the tens digit of the total sum. _⟨_ /key _⟩ ⟨_ /Step 3: _⟩ ⟨_ Step 4: _⟩_ 

Extract the last two digits of each factorial: 

7! _≡_ 40 (mod 100) _,_ 8! _≡_ 20 (mod 100) _,_ 9! _≡_ 80 (mod 100) 

_⟨_ key _⟩_ 

Finding the last two digits involves taking the factorial modulo 100. This simplifies the problem by focusing only on the relevant digits that affect the tens place. 

_⟨_ /key _⟩ ⟨_ /Step 4: _⟩ ⟨_ Step 5: _⟩_ Sum the last two digits: 

40 + 20 + 80 = 140 

_⟨_ key _⟩_ Adding the last two digits together gives us the intermediate sum. It’s important to keep track of the total to determine the final tens digit. 

_⟨_ /key _⟩ ⟨_ /Step 5: _⟩ ⟨_ Step 6: _⟩_ Determine the tens digit of the sum 140: Tens digit = 4 

_⟨_ key _⟩_ The tens digit is the second digit from the right in a number. Here, it’s clear that the tens digit of 140 is 4, which is the answer we seek. 

_⟨_ /key _⟩ ⟨_ /Step 6: _⟩ ⟨_ Generalized: _⟩_ To solve similar problems, follow these steps: 1. Identify the range of factorials and determine which ones affect the desired digit. 2. Calculate the relevant factorials and extract the necessary digits. 3. Sum these digits and focus on the specific place value required (e.g., tens, hundreds). 4. Extract the desired digit from the final sum. 

_⟨_ Generalized: _⟩ ⟨_ Answer: _⟩_ The tens digit is 4 . _⟨_ /Answer: _⟩_ 

22 

Published as a conference paper at ICLR 2025 

**IV MATH Problem:** In how many ways can the letters of the word ”COPYRIGHT” be arranged? 

## **Solution Description:** 

_⟨_ Step 1: _⟩_ Identify the total number of letters in the word ”COPYRIGHT”. _⟨_ key _⟩_ The word ”COPYRIGHT” consists of 9 letters. It’s important to count all letters accurately to determine the total arrangements. _⟨_ /key _⟩ ⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Determine if any letters are repeated in the word. _⟨_ key _⟩_ In ”COPYRIGHT”, all letters are unique. Understanding whether letters are repeated is crucial because repeated letters reduce the number of unique arrangements. _⟨_ /key _⟩_ 

_⟨_ /Step 2: _⟩ ⟨_ Step 3: _⟩_ Apply the formula for permutations of unique objects. _⟨_ key _⟩_ Since all letters are unique, the number of arrangements is calculated using the factorial of the total number of letters: 9!. 

_⟨_ /key _⟩_ 

_⟨_ /Step 3: _⟩ ⟨_ Step 4: _⟩_ Calculate 9! to find the total number of arrangements. _⟨_ key _⟩_ 9! = 9 _×_ 8 _×_ 7 _×_ 6 _×_ 5 _×_ 4 _×_ 3 _×_ 2 _×_ 1 = 362 _,_ 880. This step involves multiplying a series of numbers, which can be calculated step-by-step to avoid errors. _⟨_ /key _⟩ ⟨_ /Step 4: _⟩ ⟨_ Generalized: _⟩_ To determine the number of arrangements of letters in a word, first count the total number of letters. If all letters are unique, use the factorial of the total count. If there are repeated letters, adjust the formula by dividing by the factorial of the count of each repeated letter. 

_⟨_ /Generalized: _⟩ ⟨_ Answer: _⟩_ 

The letters of the word ”COPYRIGHT” can be arranged in 362,880 different ways. _⟨_ /Answer: _⟩_ 

23 

Published as a conference paper at ICLR 2025 

## **V GSM8K Problem:** 

Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree? 

## **Solution Description:** 

_⟨_ Step 1: _⟩_ Calculate the revenue earned from selling lemons each year. 

_⟨_ key _⟩_ 

7 lemons * $1.5 per lemon = $10.5. 

This step involves multiplying the number of lemons produced by the price per lemon. Understand the importance of identifying fixed production numbers (in this case, 7 lemons) and their respective market price ($1.5). The multiplication here gives us the total income from the lemons before accounting for expenses. 

_⟨_ /key _⟩_ 

_⟨_ /Step 1: _⟩_ 

_⟨_ Step 2: _⟩_ 

Calculate the net earnings after deducting annual costs for watering and feeding the tree. _⟨_ key _⟩_ 

$10.5 - $3 = $7.5. 

This step is about understanding how to subtract fixed expenses from total revenue to find net income. It’s crucial to separate income from costs to ascertain true profit. The $3 cost for watering and feeding is constant each year, impacting the net returns from the lemon sales. 

_⟨_ /key _⟩_ 

_⟨_ /Step 2: _⟩_ 

_⟨_ Step 3: _⟩_ 

Determine how many years it takes to cover the initial cost of planting the tree. _⟨_ key _⟩_ 

$90 / $7.5 = 12. 

In this step, you’re figuring out how long it takes to break even on the initial investment of $90. This involves dividing the total investment by the annual net earnings. Remember that this result indicates the breakeven year, but does not count the year in which the profits actually start. _⟨_ /key _⟩_ 

_⟨_ /Step 3: _⟩_ 

_⟨_ Step 4: _⟩_ 

Identify the year when he starts earning profit from the lemon tree. _⟨_ key _⟩_ 

12 (years to break even) + 1 = 13. 

This final calculation shifts the perspective from a breakeven analysis to profitability. Since he reaches the breakeven point at the end of year 12, he only begins to profit in year 13. This step emphasizes the importance of understanding financial timelines in cash flow analysis. 

_⟨_ /key _⟩_ 

_⟨_ /Step 4: _⟩_ 

_⟨_ Generalized: _⟩_ 

The solution involves calculating total income from sales, subtracting operating costs to find net earnings, and determining the break-even point by dividing the initial investment by annual net earnings. Finally, knowing when profit occurs adds critical insight into business investment analysis. For similar problems, follow these steps: identify revenues, calculate net profits, find break-even time, and ascertain the timeline for profitability. 

_⟨_ /Generalized _⟩_ 

_⟨_ Answer: _⟩_ 13 _⟨_ /Answer _⟩_ 

24 

Published as a conference paper at ICLR 2025 

## **VI GSM8K Problem:** 

Tommy is fundraising for his charity by selling brownies for $3 a slice and cheesecakes for $4 a slice. If Tommy sells 43 brownies and 23 slices of cheesecake, how much money does Tommy raise? 

## **Solution Description:** 

_⟨_ Step 1: _⟩_ Calculate the total money raised from selling brownies. _⟨_ key _⟩_ 

To find the total money raised from brownies, we multiply the number of brownies sold by the price per brownie. This is a straightforward multiplication problem. Here, Tommy sold 43 brownies at $3 each. The calculation is: 43 _×_ 3 = 129. 

Understanding multiplication is crucial as it forms the basis for calculating total revenue from sales. _⟨_ /key _⟩ ⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Calculate the total money raised from selling cheesecakes. _⟨_ key _⟩_ Similar to the previous step, we need to multiply the number of cheesecakes sold by the price per cheesecake. 

Tommy sold 23 slices of cheesecake at $4 each. The calculation is: 23 _×_ 4 = 92. 

This step reinforces the concept of multiplication and helps in understanding how to calculate total sales from different products. _⟨_ /key _⟩_ 

_⟨_ /Step 2: _⟩_ 

_⟨_ Step 3: _⟩_ 

Add the total money raised from both brownies and cheesecakes. 

_⟨_ key _⟩_ 

Now, we need to combine the total amounts raised from both products to find the overall total. This involves simple addition: 129 (from brownies) + 92 (from cheesecakes) = 221. 

This step is important as it teaches how to aggregate totals from different sources, a common task in finance and fundraising. 

_⟨_ /key _⟩_ 

_⟨_ /Step 3: _⟩_ 

_⟨_ Generalized _⟩_ 

To calculate the total funds raised from selling different items, follow these common steps: 

1. **Identify the number of items sold** for each product. 

2. **Determine the price per item** for each product. 

3. **Calculate the total revenue** for each product by multiplying the number of items sold by the price per item. 

4. **Sum all individual revenues** to find the overall total funds raised. 

This systematic approach ensures accurate calculation of total revenue from multiple sources, which is essential in various financial and fundraising scenarios. 

_⟨_ /Generalized _⟩_ 

_⟨_ Answer: _⟩_ Tommy raises $221. _⟨_ /Answer _⟩_ 

## C IMPROVED SELF-CORRECTION RESULTS 

In this section, we select three different self-correction results each from different datasets including MATH, GaoKao, and GSM8K. It should be noted that we split the incorrect reasoning steps with error cause analysis and teacher correction into two parts for better presentation. We denote the error cause in brown and we denoted the origianl error answer in red, and the correction along with correct answer are denoted in green. 

25 

Published as a conference paper at ICLR 2025 

**I GSM8K Problem:** Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree? 

**Correct reasoning process before error steps:** _⟨_ Step 1: _⟩_ Calculate the total revenue from selling lemons each year. _⟨_ key _⟩_ To find the total revenue from selling lemons, multiply the number of lemons by the price per lemon. Here, Carlos sells 7 lemons at $1.5 each. This step is straightforward but crucial as it sets the foundation for understanding the income generated from the lemon tree. _⟨_ /key _⟩_ Total Revenue = 7 lemons * $1.5 per lemon = $10.5 _⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Calculate the total cost of feeding and watering the tree each year. _⟨_ key _⟩_ This step involves identifying the annual costs of maintaining the lemon tree. The cost of feeding and watering is $3 per year. Understanding this cost helps in determining the net profit after accounting for all expenses. _⟨_ /key _⟩_ Total Cost = $3 _⟨_ /Step 2: _⟩ ⟨_ Step 3: _⟩_ Calculate the net profit per year. _⟨_ key _⟩_ Net profit is calculated by subtracting the total costs from the total revenue. This step shows how much money Carlos earns each year after all expenses are accounted for. _⟨_ /key _⟩_ Net Profit = Total Revenue - Total Cost = $10.5 - $3 = $7.5 _⟨_ /Step 3: _⟩ ⟨_ Step 4: _⟩_ Identify the initial cost of planting the tree. _⟨_ key _⟩_ The initial cost of planting the tree is a one-time expense of $90. This represents the initial investment that needs to be recovered through profits over multiple years. _⟨_ /key _⟩_ Initial Cost = $90 _⟨_ /Step 4: _⟩_ 

26 

Published as a conference paper at ICLR 2025 

**I Self-Correction For GSM8K Problem: Theses steps below are incorrect in the original reasoning steps** 

_⟨_ Step 5: _⟩_ 

Calculate the number of years it will take to break even. 

_⟨_ key _⟩_ 

This step involves dividing the initial investment by the annual net profit to find out how many years it will take for the net profit to cover the initial cost. _⟨_ /key _⟩_ Years to Break Even = Initial Cost / Net Profit = $90 / $7.5 = 12 years _⟨_ cause _⟩_ 

While this calculation correctly determines that Carlos will break even after 12 years, it does not answer the question of when he starts earning money. The break-even point occurs at the end of the 12th year, meaning he hasn’t earned a profit yet within those 12 years. _⟨_ /cause _⟩_ 

_⟨_ /Step 5: _⟩⟨_ Step 6: _⟩_ Determine when Carlos starts earning money from the lemon tree. _⟨_ key _⟩_ 

Since Carlos breaks even at the end of year 12, he will start earning profit from year 13 onwards. _⟨_ /key _⟩_ Year when profit starts = Years to Break Even + 1 = 12 + 1 = 13 years _⟨_ correction _⟩_ Therefore, Carlos will start earning money on the lemon tree in 13 years, not 12 years. _⟨_ /correction _⟩ ⟨_ /Step 6: _⟩ ⟨_ Answer: _⟩_ 13 _⟨_ /Answer: _⟩_ 

27 

Published as a conference paper at ICLR 2025 

## **II MATH Problem:** 

How many different bracelets have 3 identical orange beads, 3 identical black beads, and 1 teal bead, if rotating or flipping a bracelet does not change it? 

## **Correct reasoning process before error steps:** 

_⟨_ Step 1: _⟩_ Identify the total number of beads and their types. 

_⟨_ key _⟩_ Understanding the composition of the bracelet is crucial. Here, we have 3 identical orange beads, 3 identical black beads, and 1 teal bead, making a total of 7 beads. 

_⟨_ /key _⟩_ 

_⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Calculate the total number of arrangements without considering rotations or reflections. _⟨_ key _⟩_ Since the orange and black beads are identical within their groups, we use the formula for permutations of a multiset: 

7! 3! _×_ 3! _×_ 1! _[.]_ 

This accounts for the indistinguishability of the identical beads. 

_⟨_ /key _⟩_ Total Arrangements = 

7! 3! _×_ 3![= 140] _[.]_ 

_⟨_ /Step 2: _⟩ ⟨_ Step 3: _⟩_ Apply Burnside’s Lemma to account for rotational and reflectional symmetries. _⟨_ key _⟩_ 

Burnside’s Lemma is essential in counting distinct arrangements when symmetries (like rotations and reflections) are considered. It requires evaluating the number of arrangements fixed by each symmetry operation and averaging over the total number of symmetries. _⟨_ /key _⟩_ 

_⟨_ /Step 3: _⟩ ⟨_ Step 4: _⟩_ Determine the number of arrangements fixed by each rotation. _⟨_ key _⟩_ 

- **Identity rotation (0°)** : All 140 arrangements are fixed. - **Nontrivial rotations (rotations by**[360] 7 _[◦]_ **and its multiples)** : No arrangement is fixed because rotating the bracelet changes the positions of the beads, and with the given bead counts, it’s impossible to have a configuration that looks the same after such rotations. 

_⟨_ /key _⟩_ Number of fixed arrangements under rotations: - Identity rotation: 140 - Other rotations: 0 _⟨_ /Step 4: _⟩_ 

28 

Published as a conference paper at ICLR 2025 

## **II Self-Correction For MATH Problem:** 

**Theses steps below are incorrect in original reasoning steps.** 

_⟨_ Step 5: _⟩_ Determine the number of arrangements fixed by each reflection. 

_⟨_ key _⟩_ For reflections, each reflection fixes one bead (the center of reflection) and swaps the remaining beads in pairs. To have an arrangement fixed under reflection, the colors of the beads in each swapped pair must be the same. _⟨_ /key _⟩ ⟨_ cause _⟩_ With 3 orange beads and 3 black beads (both odd counts), it’s impossible to pair all beads of the same color. Each pair requires 2 beads of the same color, so we need even counts of each color. Therefore, no arrangements are fixed under reflections because the color counts do not allow for all swapped pairs to be identical in color. 

_⟨_ /cause _⟩_ Number of fixed arrangements under reflections: 0 _⟨_ /Step 5: _⟩ ⟨_ Step 6: _⟩_ Apply Burnside’s Lemma to find the number of distinct bracelets. The original error answer is 20 _⟨_ key _⟩_ 

According to Burnside’s Lemma, the number of distinct arrangements is the average number of arrangements fixed by each symmetry operation. The total number of symmetries in a bracelet with 7 beads is 14 (7 rotations and 7 reflections). _⟨_ /key _⟩_ Calculation: Number of distinct bracelets =[Total fixed arran][g][ements][140] Number of symmetries[=] 14[= 10] _[.]_ 

_⟨_ correction _⟩_ Therefore, the correct number of different bracelets is **10**, not 20. The error in the original calculation was assuming that there were arrangements fixed under reflections, but due to the odd counts of identical beads, no such fixed arrangements exist. _⟨_ /correction _⟩ ⟨_ /Step 6: _⟩ ⟨_ Answer: _⟩_ There are 10 different bracelets. _⟨_ /Answer: _⟩_ 

## **III GaoKao Problem:** 

Set point _O_ as the origin, the vertical line _x_ = _a_ , and the two branches of the hyperbola _C_ : _[x] a_[2][2] _[−][y] b_[2][2][= 1] (where _a >_ 0 and _b >_ 0) intersect at points _D_ and _E_ . If the area of quadrilateral _ODE_ is 8, then the minimum value of the distance between the two branches of _C_ is ( ) A. 4 B. 8 C. 16 D. 32 

**Correct reasoning process before error steps:** _⟨_ Step 1: _⟩_ Set point _O_ as the origin and the vertical line _x_ = _a_ . _⟨_ /Step 1: _⟩ ⟨_ Step 2: _⟩_ Identify the two branches of the hyperbola _C_ : _[x] a_[2][2] _[−][y] b_[2][2][= 1][, where] _[ a >]_[ 0][ and] _[ b >]_[ 0][.] _⟨_ /Step 2: _⟩_ 

29 

Published as a conference paper at ICLR 2025 

**III Self-Correction For GaoKao Problem: Theses steps below are incorrect in original reasoning steps.** 

_⟨_ Step 3: _⟩_ Determine the points _D_ and _E_ where the vertical line _x_ = _a_ intersects the hyperbola _C_ . _⟨_ cause _⟩_ Substituting _x_ = _a_ into the hyperbola equation yields _y_ = 0, resulting in only one intersection point ( _a,_ 0). The assumption that there are two distinct points _D_ ( _a, b_ ) and _E_ ( _a, −b_ ) is incorrect. _⟨_ /cause _⟩ ⟨_ correction _⟩_ Substitute _x_ = _a_ into the hyperbola equation: _a_[2][=] _[⇒]_[1] _[ −][y]_[2][=] _[⇒][y]_[2][=] _[⇒][y]_[= 0] _a_[2] _[−][y] b_[2][2][= 1] _b_[2][= 1] _b_[2][= 0] Therefore, the vertical line _x_ = _a_ intersects the hyperbola _C_ only at the point _D_ ( _a,_ 0). Point _E_ does not exist as there is only one intersection point. _⟨_ /correction _⟩ ⟨_ /Step 3: _⟩ ⟨_ Step 4: _⟩_ Calculate the area of quadrilateral _ODE_ . _⟨_ cause _⟩_ Since there is only one intersection point _D_ ( _a,_ 0), the quadrilateral _ODE_ does not exist. The area calculation based on non-existent points _E_ ( _a, b_ ) and _A_ ( _a, −b_ ) is invalid. _⟨_ /cause _⟩ ⟨_ correction _⟩_ Reevaluate the problem statement. If the vertical line _x_ = _a_ intersects the hyperbola at only one point _D_ ( _a,_ 0), the area calculation of quadrilateral _ODE_ is not applicable. It is likely that there is a misunderstanding in the problem setup. 

_⟨_ /correction _⟩ ⟨_ /Step 4: _⟩ ⟨_ Step 5: _⟩_ Find the minimum value of the distance between the two branches of the hyperbola. The original error answer is A = 4 _⟨_ cause _⟩_ The reasoning that the distance between the two branches is 2 _a_ is incorrect. Additionally, the approach to minimize 2 _a_ under the condition _ab_ = 4 does not align with the correct geometric interpretation of the hyperbola. _⟨_ /cause _⟩ ⟨_ correction _⟩_ The distance between the two branches of the hyperbola _C_ is determined by the minimum vertical distance between corresponding points on the branches. To find this, consider the closest points on each branch: Distance = 2 _b._ Given that the area condition was misapplied, we refer to the correct relationship from the Reference Solution where the minimum distance is found using optimization techniques. The correct minimum distance of the hyperbola _C_ is 8. _⟨_ /correction _⟩ ⟨_ /Step 5: _⟩ ⟨_ Answer: _⟩_ The minimum value of the distance of the hyperbola _C_ is 8. _⟨_ /Answer: _⟩_ � 

30 

