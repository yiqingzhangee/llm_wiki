# **Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

**Zhiheng Xi**[1 *] _[ †]_ **Wenxiang Chen**[1 *] **Boyang Hong**[1 *] **Senjie Jin**[1 *] **Rui Zheng**[1] _[ †]_ **Wei He**[1] **Yiwen Ding**[1] **Shichun Liu**[1] **Xin Guo**[1] **Junzhe Wang**[1] **Honglin Guo**[1] **Wei Shen**[1] **Xiaoran Fan**[1] **Yuhao Zhou**[1] **Shihan Dou**[1] **Xiao Wang**[1] **Xinbo Zhang**[2] **Peng Sun**[2] **Tao Gui**[3] _[ †]_ **Qi Zhang**[1] _[ †]_ **Xuanjing Huang**[1] 

zhxi22@m.fudan.edu.cn, _{_ rzheng20,tgui,qz _}_ @fudan.edu.cn 

## **Abstract** 

In this paper, we propose **R**[3] : Learning **R** easoning through **R** everse Curriculum **R** einforcement Learning (RL), a novel method that employs only outcome supervision to achieve the benefits of process supervision for large language models. The core challenge in applying RL to complex reasoning is to identify a sequence of actions that result in positive rewards and provide appropriate supervision for optimization. Outcome supervision provides sparse rewards for final results without identifying error locations, whereas process supervision offers step-wise rewards but requires extensive manual annotation. **R**[3] overcomes these limitations by learning from correct demonstrations. Specifically, **R**[3] progressively slides the start state of reasoning from a demonstration’s end to its beginning, facilitating easier model exploration at all stages. Thus, **R**[3] establishes a stepwise curriculum, allowing outcome supervision to offer step-level signals and precisely pinpoint errors. Using Llama2-7B, our method surpasses RL baseline on eight reasoning tasks by 4 _._ 1 points on average. Notebaly, in program-based reasoning on GSM8K, it exceeds the baseline by 4 _._ 2 points across three backbone models, and without any extra data, Codellama-7B + **R**[3] performs comparable to larger models or closed-source models.[1] 

## **1. Introduction** 

Large language models (LLMs) have made impressive advancements in complex, multi-step reasoning, by prompting 

*Equal contribution _†_ Corresponding author 1School of Computer Science, Fudan University[2] ByteDance Research[3] Institute of Modern Languages and Linguistics, Fudan University. 

> 1Our codes and data are available at Github : https://github.com/WooooDyy/LLM-Reverse-Curriculum-RL. 

_Table 1._ Comparison of the proposed R[3] with other supervision methods in terms of three key features. **Golden** means whether the supervisory signals are based on golden labels (e.g., correctness) or human preference; **Human-Annotation-free** indicates that the method does not require detailed annotations for each intermediate step; **Step-level Sup.** means whether the method can provide stepby-step supervisory signals. 

|FEATURE|SUPERVISIONMETHOD|
|---|---|
||Outcome Sup.<br>Process Sup.<br>Ours|
|Golden<br>Human-Annotation-free<br>Step-level Sup.|�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�|



or learning to generate solutions in a step-by-step Chainof-Thought manner (Wei et al., 2022; Kojima et al., 2022; Kim et al., 2023). Training a language model specialized in reasoning is proved to be superior to prompting-based approaches (Uesato et al., 2022; Yu et al., 2023b). However, Supervised Fine-tuning (SFT) focuses on imitating human demonstrations, requiring large-scale, diverse annotations to achieve generalization (Lightman et al., 2023; Yuan et al., 2023; Shen et al., 2021). Reinforcement learning (RL) offers a viable alternative to improve reasoning via exploration and learning (Bai et al., 2022; Ouyang et al., 2022; Zheng et al., 2023; Luo et al., 2023). 

When applying RL to complex reasoning tasks, the core challenge lies in identifying a sequence of actions that yield positive rewards and providing appropriate supervisory signals for optimization (Sutton et al., 1998). On one hand, as task difficulty increases, so does the complexity and length of the reasoning chain. LLMs struggle with the accumulation of errors and uncertainties across multiple intermediate steps (Lightman et al., 2023; Yu et al., 2023a; Zhang et al., 2023). The increase of reasoning steps leads to an exponential growth in the search space for reasoning, making it challenging to obtain correct final results (Xie et al., 2023). On the other hand, existing methods for supervised signals require a trade-off between feedback quality and annotation 

1 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
S upervised  F ine- T uning O utcome- S upervised  RL P rocess- S upervised  RL<br>Outcome-based Rewards Process-based Rewards<br>:  Learning  R easoning through  R everse Curriculum  R L : Policy LM<br>Stage1 : Question<br>Q:  Julie is reading a 120-page book …<br>: Demonstration step<br>How many pages will be read tomorrow?<br>: LM-generated step<br>Demonstration Rationale Stage2<br>: OS reward function<br>Step1:  She read 12 × 2=24 pages today.<br>Step2:  So she was able to read a total  : PS reward function<br>of 12+24=36 pages since yesterday. Stage3<br>Step3:  There are 120-36=84 pages left<br>to be read.<br>Step4:  Since she wants to read half of  R everse  Stage4<br>the remaining pages tomorrow, then<br>she should read 84 ÷ 2=42 pages. Curriculum<br>**----- End of picture text -----**<br>


_Figure 1._ Schematic comparison between R[3] and other methods for training LLMs for reasoning. _L_ ( _·_ ) represents the optimization objective for each method. Supervised Fine-Tuning optimizes models using annotated rationales, without additional exploration. In RL, the model first generates a reasoning path and receives supervisory signals for optimization. Outcome-Supervised (OS) RL rewards the final result, while Process-Supervised (PS) RL rewards each reasoning step. The proposed R[3] provides approximately step-by-step supervisory signals similar to PS with only an OS reward function. 

cost (Uesato et al., 2022). Outcome supervision (OS, Cobbe et al., 2021; Yu et al., 2023a) rewards only the final outcome (top center in Figure 1), but sparse rewards make it difficult to determine which actions led to success or failure (Wang et al., 2023b). Process supervision (PS, Uesato et al., 2022; Lightman et al., 2023) provides detailed feedback at every step of reasoning (top right in Figure 1), but this approach requires highly skilled annotators to select better reasoning paths, significantly increasing costs (Lightman et al., 2023). 

In this work, we propose **R**[3] : Learning **R** easoning through **R** everse Curriculum **R** einforcement Learning (bottom in Figure 1) to address the limitations. It employs only outcome supervision to achieve an effect similar to process supervision. Specifically, R[3] let the model begin reasoning from a state sampled from a correct demonstration, and provide feedback to supervise the generated actions with outcome supervision. By slowly moving the start state from the end of the demonstration to the beginning, the model faces an easy exploration problem at each point where it is likely to succeed, since it has already learned to solve most of the remaining parts. In this way, a curriculum of gradually increasing exploration difficulty is created, and we can provide approximately step-by-step supervisory signals for the model. 

This method facilitates the model’s exploration as it shortens the reasoning chain and narrows the sampling space, aiding the model in gaining positive rewards more efficiently. 

We can interpret **R**[3] as a form of dynamic programming (Bertsekas, 2012). If _N_ reasoning steps are required to obtain a reward, this reasoning can now be learned in a time that is linear in _N_ , rather than exponential (Florensa et al., 2017; Salimans & Chen, 2018). To improve the training stability and model generalization, we mix the start states of various exploration difficulties for training. Thorough experiments on Llama2-7B demonstrate that R[3] outperforms both the SFT and RL baselines across eight reasoning tasks, achieving an average improvement of 5 _._ 4 points and 4 _._ 1 points, respectively. Notably, in program-based reasoning on GSM8K, it surpasses SFT and RL by an average of 11 _._ 4 points and 4 _._ 2 points, respectively. Moreover, Codellama7B + R[3] outshines models that use extra annotated data like MAmmoTH-Coder (Yue et al., 2023) and Tora (Gou et al., 2023), and is comparable to larger or closed-source models such as GPT-3.5-Turbo. 

In summary, we make the following contributions: 

1. We propose R[3] , a novel method which employs outcome supervision to achieve an effect similar to process supervision, to enhance the reasoning ability of LLMs. 

2. We conduct extensive experiments across eight reasoning tasks to highlight the effectiveness of our method. Furthermore, we showcase the superiority of R[3] in program-based reasoning through its application on three models for solving math problems. 

2 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

3. We perform in-depth ablation and analysis to provide insights into the training dynamics of R[3] and how it works. 

## **2. RL with Outcome and Process Supervision** 

We use RL notations to describe the language generation process. At each timestep _t_ , the policy language model (LM) _πθ[RL]_ parameterized by _θ_ receives a state _st_ , which consists of the input prompt and the generated text up to this point. Then, the policy’s action _at_ +1 is to generate next token conditioned on the state, and the probability is as _πθ_ ( _at_ +1 _|st_ ). After that, the environment returns a reward _r_ ( _st, at_ +1), and the state is transitioned to _st_ +1 with the transition probability _p_ ( _st_ +1 _|st, at_ +1). The goal of RL is to find an optimal policy to maximize the cumulative reward (i.e., return) over a trajectory _τ_ = _{s_ 0 _, a_ 1 _, ..., sT , aT }_ where _s_ 0 is the initial state (i.e., the prompt) and _T_ is the length of actions. The general form of the policy gradient is gaven as (Mnih et al., 2016): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0003-04.png)


where E _τ ∼πθ_ RL refers to the expectation under the distribution of trajectories sampled from the policy _πθ_[RL][.][The] return _R_ ( _st−_ 1 _, at_ ) =[�] _[T] t[′]_ = _t[γ][t][′][−][t]_[+1] _[r]_[(] _[s][t][′][−]_[1] _[, a][t][′]_[)][is][the] discounted sum of rewards from timestep _t_ with factor _γ ∈_ [0 _,_ 1). With this gradient, we can perform gradient ascent to optimize the model. If the return is favorable, the actions are “reinforced” by increasing their probability of being selected. Given a dataset _D_ = _{_ ( _s[i]_ 0 _[,]_ **[ a]** _[i]_[)] _[}][N] i_ =1[of] _N_ pairs of input _s_ 0 and human-generated output sequence **a** , where **a** = ( _a_ 1 _, a_ 2 _, ..., aT_ ) and the whole trajectory is _τ_ = _{s_ 0 _, a_ 1 _, ..., sT −_ 1 _, aT }_ . The policy gradient becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0003-06.png)


## **2.1. Outcome Supervision and Process Supervision** 

Here we present the operating mechanisms of outcome supervision and process supervision, along with their advantages and limitations, as briefly summarized in Table 1. 

**Outcome supervision.** In outcome supervision, only the final result of the sampled sequence is assigned a reward score, and the score for other tokens are 0 (Cobbe et al., 2021; Yu et al., 2023a): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0003-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0003-11.png)


**----- Start of picture text -----**<br>
100 100<br>90 90<br>80<br>80<br>70<br>70<br>GSM8K 60<br>60 MNLIrace@high 50<br>BGQA<br>50 0 10 20 30 40 50 60 70 80 90 40 0 10 20 30 40 50 60 70 80 90<br>Starting Point (%) Starting Point (%)<br>(a) Train Set (b) Test Set<br>Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


_Figure 2._ Accuracy v.s. different start state for exploration. The horizontal axis represents the start state for exploration, with the values indicating the percentage of given actions out of the total actions in the demonstration. The results demonstrate a trend that starting the reasoning from a position closer to the target state makes it easier for the model to obtain a positive reward. 

where _rfo_ ( _·_ ) is a reward function that returns 1 is the answer is correct else 0. In this paradigm, we don’t require detailed annotations for each reasoning step or the training a reward model to allocate rewards. Instead, the golden answer to the question is enough. This supervision is solely based on the correctness, not on the preference of humans. Despite this simplicity, the supervisory signals are sparse, making it challenging for the policy LM to pinpoint reasoning errors accurately. The policy may fall into aimless exploration and struggle in obtaining positive rewards due to the large action space of the LM and the long decision-making chain. 

**Process supervision.** In process supervision, a reward model _rmp_ ( _·_ ) is trained to assign a reward score for each intermediate reasoning step (Uesato et al., 2022; Lightman et al., 2023): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0003-15.png)


where _T[Delimiter]_ represents the set of timesteps that delimite each step (e.g., newline or some special symbols). In this paradigm, the rewards are dense, then provide more precise supervision. However, the training for reward model needs fine-grained annotations, which demands skilled annotators and can be very expensive (Lightman et al., 2023; Luo et al., 2023). Additionally, the reward model reflects human preferences, which might introduce bias, and may not always align perfectly with objective correctness or usefulness (Wang et al., 2024b; Pitis, 2023). 

## **3. Methodology** 

**Motivation.** From our previous analysis in Section 2, we seek to merge the benefits of outcome and process supervision while avoiding their drawbacks. We aim to develop a method that doesn’t need fine-grained annotations for every step or training a reward model, avoids personal biases by 

3 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
86 Difficulty Level-1 74 Difficulty Level-2 66 Difficulty Level-3 57 Difficulty Level-4 52 Difficulty Level-5<br>R [3]<br>85 Vanilla Staged RL 63 54 48<br>72<br>84 60 51 44<br>70<br>83 57 48 40<br>68<br>82 54 45 36<br>81 Training process 66 Training process 51 Training process 42 Training process 32 Training process<br>Performance Performance Performance Performance Performance<br>**----- End of picture text -----**<br>


_Figure 3._ Learning curves on test sets with 5 different difficulty level for Staged RL and R[3] . The farther the starting point for exploration is from the target, the higher the difficulty level. The horizontal axis represents the training process. The vertical dashed lines indicate the transitions between training stages for staged RL. The experiments are conducted on GSM8K reasoning. Staged RL suffers significant performance drops when transitioning stages, while the performance of R[3] improves stably. 

using only golden outcome supervision, and still provides an effect akin to step-level supervision. Hence, we assume access only to the outcome-based reward function _rfo_ ( _·_ ) and propose **R**[3] : Learning **R** easoning through **R** everse Curriculum **R** einforcement Learning. 

## **3.1. Start Exploration from Intermediate States of Demonstrations** 

For a multi-hop reasoning problem, there is a golden answer that can be derived through different reasoning paths. We assume to have access to at least one demonstration, i.e., correct reasoning path that leads to a the golden answer as in supervised fine-tuning. When the model begins exploration from the initial start state _s_ 0, it might face difficulty in obtaining positive rewards as discussed in Section 2.1. 

Inspired by previous work in the area of RL with demonstrations (Kakade & Langford, 2002; Subramanian et al., 2016b; Florensa et al., 2017; Salimans & Chen, 2018), we define the set of intermediate states of a given demonstration as _S[Inter] ⊂S_ , and let the policy LM _πθ_ start exploration from an intermediate state _sk ∈S[Inter]_ close to the target state: _πθ_ ( **a** _k_ +1: _T |sk_ ) where **a** _k_ +1: _T_ = ( _ak_ +1 _, ..., aT_ ). An outcome-based reward function then provides feedback for the final result, serving as a supervisory signal for actions taken after _sk_ . In this strategy, the trajectory preceding _sk_ in the demonstration (i.e., _{s_ 0 _, a_ 1 _, s_ 1 _, ..., ak}_ ) can serve as a form of guidance, enabling the model to get positive rewards more easily and avoid getting stuck in directionless, inefficient exploration processes, as shown in Figure 2. 

## **3.2. Reverse Curriculum Learning for Step-level Supervision** 

Once the policy learns to achieve the goal starting from the selected state close to the target, it can extend its training to more distant states (e.g., _sk−_ 1), bootstrapping the knowledge it has already acquired. At each point, the model faces an easy exploration problem where it is likely to succeed, as it has already learned to solve most of the remaining parts. In this way, a curriculum of gradually increasing exploration 

difficulty is created, allowing us to provide approximately step-by-step supervisory signals for the model. Now the policy gradient can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0004-10.png)


where _S[Inter]_ refers to the set of intermediate states of a demonstration sampled from dataset _D_ ; _k_ starts from _T −_ 1 and progressively slides back to 0. In the final step, the model begins rolling out from the initial state _s_ 0, which is equivalent to the original outcome-supervised RL. 

In multi-step reasoning, language models may generate a large number of actions (i.e., tokens), making it difficult to enumerate all possible intermediate states and explore from these states. Therefore, the number of start states in the reverse curriculum will affect training costs and final reasoning performance. In our method, we sample _M_ intermediate states from demonstrations either at line breaks (if present) or uniformly, as start states for exploration. Thus, a reverse curriculum with _M_ stages is created using these selected starting points[2] . We refer to this method in this paper as **vanilla staged RL** . In our experiments, _M_ is typically 5 or 6 and in Section 5.1, we analyze the impact of the number of stages on reasoning performance. 

## **3.3. Mixing Start States for Generalization** 

As shown in preliminary experiments in Figure 3, staged RL may have potential limitations. Models might overfit to simple patterns presented in the early stages of the curriculum and fail to generalize effectively when the difficulty increases, leading to a degradation of previously acquired knowledge. Furthermore, our findings indicate that staged 

> 2Please note that ‘stage’ here refer to training stages, where the intermediate states sampled in the first stage are those closest to the goal, while the states sampled in the last stage are those farthest from the goal. 

4 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

RL may struggle to adequately capture and model complex interactions and dependencies inherent within the data. To address this issue, we draw inspiration from the field of multi-task learning (Ruder, 2017; Zhang & Yang, 2022) and treat each stage as an independent task. In the final **R**[3] , we adopt a mixed strategy to ensure smooth transitions and cooperative optimization between stages of different difficulty levels, stabilizing the training process and enhancing reasoning performance. 

## **3.4. Reward Design and Policy Optimization** 

We employ proximal policy optimization (PPO, Schulman et al., 2017) as our basic policy gradient algorithm as it has proved effective in RLHF of LLMs. We apply partial reward _ϵ_ (e.g., _ϵ_ = 0 _._ 1) on mathematical reasoning tasks when answer can be extracted and of numeric type to make the reward denser following (Zhong et al., 2017; Le et al., 2022): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0005-04.png)


We also design reward functions based on the exploration difficulty, which will be discussed in Section 5.3. Following (Lu et al., 2023), our total reward is the sum of reward function score and the Kullback-Leibler (KL) divergence between the learned RL policy and initial policy _πθ[Init]_ scaled by a coefficient factor _β_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0005-06.png)


We calculate advantages with generalized advantage estimate (GAE) and perform optimization similar to Schulman et al. (2017). Our algorithm is outlined in Algorithm 1. We first construct the curriculum datasets of different stages and describes procedures for vanilla staged RL and the final R[3] . 

widely used datasets. For the logical reasoning, we utilize the BoardgameQA (BGQA, Kazemi et al., 2023), which is a challenging reasoning task containing contradictory information from various sources. We select its “main” subset and “conflict” subset. For NLI, we select the commonly used datasets SNLI (Bowman et al., 2015) and MNLI (Williams et al., 2018), and acquire their rationales from CoT-Collection (Kim et al., 2023). For reading comprehension, we choose race@Middle and race@High (Lai et al., 2017), two challenging reading comprehension tasks, and obtain their rationales from CoT-Collection (Kim et al., 2023). 

**Models and baselines.** For CoT reasoning, we choose Llama2-Base-7B (Touvron et al., 2023) as our backbone model because it is widely used. We include few-shot CoT, SFT and RL as our baselines. For P-CoT reasoning, we choose Llama2-Base-7B (Touvron et al., 2023), Glactica (Taylor et al., 2022), and Codellama-7B (Roziere` et al., 2023) as our backbone. We include few-shot P-CoT, SFT and RL as baselines. We also consider recently proposed methods/models that require data augmentation, including MAmmoTH-Coder (7B & 34B, Yue et al., 2023), Tora and Tora-coder (7B & 13B, Gou et al., 2023). 

**Implementation details.** Our training is done with eight A100-80GB GPUs and using DeepSpeed framework (Rasley et al., 2020). For few-shot CoT, we run five times with different demonstrations and report the average performance. For SFT, we set the learning rate to 2 _e −_ 5. For each RL-related method, we first perform SFT to warm-up and then perform RL. We set the partial reward _ϵ_ to 0 _._ 1 for SVAMP and 0 _._ 2 for GSM8K. For CoT experiments, we set _β_ to 0 _._ 05 in math reasoning and set _β_ to 0 _._ 3 in other tasks; for P-CoT experiments, we set _β_ to 0 _._ 01. For mathematical tasks, we perform 50 epochs for RL and report the best performance, including CoT and P-CoT. For other tasks, we perform 5 epochs for RL and report the best performance. 

## **4.2. Experimental Results** 

## **4. Experiments** 

## **4.1. Experimental Setup** 

**Datasets.** Given that our work focuses on enhancing the reasoning capabilities of LLMs, we select various task types that require reasoning abilities, including logical reasoning, mathematical reasoning, reading comprehension, and natural language inference (NLI). We also consider programbased reasoning (i.e., P-CoT) for math problem solving following Gao et al. (2023), where we execute the generated Python program to obtain the answer. 

Regarding mathematical reasoning, we choose GSM8K (Cobbe et al., 2021) and SVAMP (Patel et al., 2021), two 

**Results on CoT reasoning.** The main results are demonstrated in Table 2. Generally, we can find that: (1) RL methods consistently perform better than prompt-based methods and SFT, showing that by continuously performing exploration and learning, models can refine their reasoning capabilities over time, similar to (Luo et al., 2023). (2) R[3] outperforms other baselines in all tasks, with an average improvement of 5 _._ 4 over SFT and 4 _._ 1 over RL, indicating that our method can provide stable and significant optimization. However, staged RL is only a bit better than the RL baseline, possibly due to overfitting and ineffective stage-to-stage adaptation mentioned before. 

Specifically, our method can enhance different reasoning 

5 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

_Table 2._ Evaluating results on CoT Reasoning. The best results of each dataset is in **bold** and marked with underline, while the second is marked with underline. Generally, “Staged RL” represents RL with a reverse, staged manner, while R[3] represents the final method with mixed stages. While the vanilla staged RL is only slightly better than RL baseline, R[3] outperforms all other baselines significantly. 

|**MATH REASONING**<br>**LOGICAL REASONING**|**MATH REASONING**<br>**LOGICAL REASONING**|**MATH REASONING**<br>**LOGICAL REASONING**|**MATH REASONING**<br>**LOGICAL REASONING**|**MATH REASONING**<br>**LOGICAL REASONING**|**MATH REASONING**<br>**LOGICAL REASONING**|**NL INFERENCE**<br>**READING COMPRE.**<br>**AVERAGE**<br>MNLI<br>SNLI<br>race@High<br>race@Middle|**NL INFERENCE**<br>**READING COMPRE.**<br>**AVERAGE**<br>MNLI<br>SNLI<br>race@High<br>race@Middle|**NL INFERENCE**<br>**READING COMPRE.**<br>**AVERAGE**<br>MNLI<br>SNLI<br>race@High<br>race@Middle|**NL INFERENCE**<br>**READING COMPRE.**<br>**AVERAGE**<br>MNLI<br>SNLI<br>race@High<br>race@Middle|**NL INFERENCE**<br>**READING COMPRE.**<br>**AVERAGE**<br>MNLI<br>SNLI<br>race@High<br>race@Middle|
|---|---|---|---|---|---|---|---|---|---|---|
|GSM8K||SVAMP<br>BGQAmain<br>BGQAconfict||||MNLI|||||
||15_._13<br>41_._55<br>44_._67<br>47_._69<br>**50**_._**49**||39_._62<br>58_._40<br>57_._30<br>61_._00<br>**64**_._**40**|39_._73<br>62_._50<br>65_._50<br>67_._00<br>**67**_._**75**|34_._97<br>57_._25<br>58_._15<br>58_._60<br>**59**_._**35**|47_._69<br>65_._38<br>66_._15<br>67_._69<br>**72**_._**31**|28_._96<br>68_._00<br>69_._60<br>68_._00<br>**72**_._**80**|38_._00<br>60_._50<br>61_._50<br>63_._00<br>**68**_._**50**|39_._20<br>68_._00<br>69_._00<br>69_._50<br>**71**_._**50**|35_._41<br>60_._19<br>61_._48<br>62_._81<br>**65**_._**62**|



_Table 3._ Evaluating results of P-CoT reasoning on GSM8K. Our method is marked in blue and outperforms Few-shot, SFT, and RL. Even against methods needing data augmentation, Codellama + R[3] achieves better performance on a 7B model scale. **Note** that _†_ indicates Tora and Tora-code are trained on additional data in SFT, but this data is not used for R[3] as it’s not released. 

|**P-COT METHOD**|**MODEL SIZE**|**AUG DATA**|**PERFOR.**|
|---|---|---|---|
|Glactica + Few-shot<br>Glactica + SFT|6.7B<br>6.7B|-<br>-|18_._6<br>57_._1|
|Glactica + RL<br>Glactica + R3<br>Llama2 + Few-shot|6.7B<br>6.7B<br>7B|-<br>-<br>-|66_._1<br>**69**_._**3**<br>18_._3|
|Llama2 + SFT|7B|-|57_._7|
|Llama2 + RL|7B|-|63_._1|
|Llama2 + R3|7B|-|**68**_._**9**|
|Codellama + Few-shot|7B|-|32_._7|
|Codellama + SFT|7B|-|63_._3|
|Codellama + RL<br>7B<br>-<br>Codellama + R3<br>7B<br>-<br>**Models Using Extra Training Data**<br>MAmmoTH-Coder (Yue et al.,2023)<br>7B<br>260k|||70_._7<br>**74**_._**2**<br>59_._4|
|Tora (Gou et al.,2023)<br>Tora (Gou et al.,2023) + R3<br>Tora-code (Gou et al.,2023)|7B<br>7B<br>7B|16k<br>16k_†_<br>16k|68_._8<br>**73**_._**2**<br>72_._6|
|Tora-code (Gou et al.,2023) + R3|7B|16k_†_|**76**_._**3**|
|**Larger Models / Close-sourced Models**||||
|MAmmoTH-Coder (Yue et al.,2023)<br>MAmmoTH-Coder (Yue et al.,2023)|13B<br>34B|260k<br>260k|64_._7<br>72_._7|
|Codex (Chen et al.,2021)|N.A.|-|71_._6|
|GPT-3.5-Turbo (Jie et al.,2023)<br>GPT-4 (OpenAI,2023;Zhou et al.,2023)|N.A.<br>N.A.|-<br>-|78_._0<br>97_._0|



ability of models. For example, on mathematical tasks, R[3] shows significant improvements compared to SFT and RL Baselines, suggesting that our method effectively helps models to acquire and refine structured and formal reasoning abilities through exploration. Our method also allows models to handle reasoning tasks with contradictory information (BGQA), demonstrating a notable enhancement in their defeasible reasoning ability (i.e., reasoning with conflicting information guided by preference, Pollock, 1987; Hecham et al., 2018; Maher et al., 2020). 

**Results on P-CoT reasoning.** The evaluating results on program-based reasoning is shown in table 3. We can find that: (1) R[3] outperforms other baselines on P-CoT reasoning across all three models. On average, it exceeds SFT by 11 _._ 4 

_Table 4._ Ablation study on GSM8K CoT, by default _β_ = 0 _._ 05, partial reward _ϵ_ = 0 _._ 2. 

||**Method Setting**<br>Llama2-Base 7B + R3|**Performance**<br>**50**_._**5**|
|---|---|---|
||- KL coeffcient_β_ = 0|46_._6|
||- KL coeffcient _β_ = 0_._1|44_._1|
||- remove partial reward_ϵ_|44_._6|
||- partial reward_ϵ_= 0_._1|45_._9|
||- partial reward_ϵ_= 0_._3|45_._4|



points and surpasses the RL Baseline by 4 _._ 2 points. This demonstrates that our method is not only highly effective but also versatile and adaptable, capable of extending to various reasoning styles like programs. (2) Compared to other methods that require data augmentation, e.g., MAmmoTH (Yue et al., 2023), Tora and Tora-code (Gou et al., 2023), Codellama-7B + R[3] achieves the better results in 7Bsized models and matches up well with larger models and closed-source model GPT-3.5-Turbo. (3) When our method is applied to models like Tora and Tora-code, which were trained with additional data during SFT, it still yields significant performance gain using only the original data in the reinforcement learning phase, demonstrating its adaptability and wide applicability. 

## **5. Analysis and Discussion** 

## **5.1. Ablation Study** 

**KL coefficient** _β_ **and partial reward** _ϵ_ **.** We first conduct ablation study on GSM8K CoT to study the impact of _β_ and _ϵ_ , and the results are shown in Table 4.[3] If we set _β_ = 0, the exploration space of the model becomes unconstrained, and we observe that R[3] can still perform well, which is different from the conclusions of previous RL methods where the model may collapse without KL penalty (Luong et al., 2024). This may be because R[3] does not require the model to constantly perform exploration from scratch, reducing the 

3See Appendix B.1 for more ablation results on other tasks. 

6 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0007-01.png)


**----- Start of picture text -----**<br>
1.0 1.5 50 R [3] 75<br>0.8 1.0 45 RL 7065 M = 3M = 4M = 5M = 6<br>M = 7<br>0.6 0.5 60<br>40<br>0.4 0.0 55<br>35 50<br>R [3] R [3]<br>0.2 RL 0.5 RL 45<br>0 4000 8000 12000 0 4000 8000 12000 300 4000 8000 12000 40<br>Step Step Step<br>35<br>(a) Mean Training Reward (b) Mean Training Return (c) Evaluation Accuracy GSM8k MNLI race@High<br>Performance<br>**----- End of picture text -----**<br>


_Figure 4._ Training dynamics of RL and R[3] on GSM8K CoT, including training reward, training return and evaluation accuracy. 

_Figure 5._ Ablation study of different stage numbers _M_ . 

sampling space and making it easier to obtain rewards, thus facilitating training. If we set _β_ = 0 _._ 1 to impose higher constraints, we observe a more significant drop in performance, indicating that overly strong KL constraints may hinder the model’s optimization. 

If we set a small partial reward _ϵ_ or remove it, R[3] obtains a lower performance yet it still outperforms RL and SFT. On the other hand, if we set _ϵ_ to a bigger value 0 _._ 3, the performance also drops as too large partial reward might lead the model to settle for obtaining simple rewards (outputting numbers) rather than striving for the correct answer. 

_Table 5._ Performance when adopting different reward functions. The “Original” one is the basic reward function that returns 1 if the answer is correct else 0. Other functions assign various rewards according to the difficulty of exploration. 

|DATASET|REWARDFUNCTION|
|---|---|
||Original<br>_Rlinear_<br>_Rsquare_<br>_Rsqrt_<br>_Rdiscount_|
|MNLI<br>race@High<br>GSM8K|72_._3<br>68_._5<br>68_._5<br>70_._0<br>68_._5<br>68_._5<br>65_._0<br>65_._5<br>66_._0<br>66_._0<br>50_._5<br>43_._7<br>41_._9<br>44_._1<br>45_._1|
|Average|63_._8<br>59_._1<br>58_._7<br>60_._0<br>59_._9|



**Number of intermediate states selected** _M_ **.** As mentioned before, if we include all possible intermediate states as starting points, the cost can be extremely high. However, too small value of _M_ might lead to large gaps between stages. Therefore, we need to find a balance and identify an appropriate _M_ . We perform ablation experiments and the results in Figure 5 show that the performance converges when _M_ reaches an appropriate value, such as 5 or 6, and larger _M_ does not yield significant benefits. 

## **5.2. R**[3] **Delivers Stable Reinforcement Learning** 

Figures 4(a) and 4(b) illustrate the training dynamics of vanilla RL and R[3] throughout the training process. We observe that RL encounters instability and fluctuations in training rewards, whereas our method is significantly more stable and yields higher returns. This can be attributed to R[3] providing denser, more detailed, and accurate supervisory signals, facilitating model’s exploration and learning. The distinction is also evident in test performance, as shown in Figure 4(c), where our method achieves more stable improvements. We also provide case studies in Appendix D to intuitively show the superiority of our method. 

## **5.3. Difficulty-based Reward Function Design** 

As mentioned before, when perform exploration from different states of the demonstration, the difficulty for the model to obtain a positive reward varies. This leads to an intuitive 

question: should we set different amounts of rewards for rollouts of varying difficulty, instead of setting them all to 1 when the final results are correct? Consequently, we use different variants of the reward function to observe their performance changes. Specifically, assuming the length of a demonstration _τ_ is _T_ : _τ_ = ( _s_ 0 _, a_ 1 _, s_ 1 _, a_ 2 _...sT_ ), with the starting point as _sk_ , we approximately define the difficulty of the rolling out process as: _µ_ = ( _T − k_ ) _/T_ . 

We then consider different reward functions related to the difficulty. These functions have different trends of change on the slope, including linear reward function _Rlinear_ = _µ_ , square reward function _Rsquare_ = _µ_[2] , and square root reward function _Rsqrt_ = _[√] µ_ . Inspired by the conception of discount factor in RL, we also consider another discount reward function: _Rdiscount_ = _γ_[(] _[T][ −][k]_[)] , where _γ_ = 0 _._ 9. Experiments in Table 5 show counter-intuitive results that these modified reward functions do not bring performance improvements, but rather, performance decreases. This implies that we should treat each start state fairly. 

## **5.4. Analysis of Training Data Construction** 

**Scaling of training data.** We first study the data efficiency of R[3] , and the results are shown in Figure 6(a). Overall, as the amount of data decreases, the performance of R[3] shows a decreasing trend. However, the sensitivity of R[3] to data scale varies by task. For instance, on GSM8K, using a limited amount of data leads to a significant decline in performance. This may be because such tasks require a large 

7 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
24 12<br>GSM8K GSM8K<br>20 MNLI 10 MNLI<br>race@high race@high<br>16 BGQA 8 BGQA<br>12 6<br>8 4<br>4 2<br>0 10 20 40 60 80 w/o part 10 w/o part 2 w/o part 3 w/o part 4 w/o part 5<br>Data Scale (%) Data Composition<br>(a) Impact of Data Scale (b) Impact of Data Composition<br>Relative Accuracy Decrease(%) Relative Accuracy Decrease(%)<br>**----- End of picture text -----**<br>


_Figure 6._ Impact of data scale and composition. The vertical axis represents the percentage of performance decrease relative to training with full dataset. The horizontal axis of the left subfigure represents the amount of data used, while the horizontal axis of the right subfigure, labeled “w/o part _j_ ”, indicates removing a part of training data corresponding to a specific difficulty level _j_ . 

amount of data to learn enough specialized mathematical knowledge to enable the model to generalize. In contrast, for BGQA, even with limited data scale, the model might still achieve better generalization performance by learning patterns and relationships in the language. Moreover, we demonstrate the absolute values of performance in Appendix B.2, and the results show that R[3] can outperform the RL baseline with only a portion of the data. 

**Which part of data matters?** Next we investigate which part of training data is crucial. We remove training data of varying difficulties (i.e., the farther the starting point is from the target, the greater the difficulty) and conduct experiments. Results in Figure 6(b) demonstrate a trend that removing the more difficult data results in poorer performance, highlighting the importance of challenging data. Conversely, removing the simplest data does not significantly degrade performance. We also provide the absolute performance values in the Appendix B.3. 

## **6. Related Work** 

**Reasoning with large language models.** Multi-hop complex reasoning is considered one of the most challenging task for LLMs (Rae et al., 2021; Bommasani et al., 2021; Qiao et al., 2022), and researchers have developed several categories of methods, including prompting, supervised fine-tuning methods and reinforcement learning methods. Prompting, with chain-of-thought as a representative one, involves constructing demonstrations and instructions in the prompt to improve model’s reasoning performance (Wei et al., 2022; Kojima et al., 2022; Xi et al., 2023; Chu et al., 2023). However, they proved to be sensitive to many factors and model-dependent (Shi et al., 2023; Zellers et al., 2018; Ye & Durrett, 2022). In SFT, models are trained with collected rationales, and their effectiveness largely relies on the scale and quality of the training data (Yuan et al., 

2023; Yu et al., 2023b; Yue et al., 2023), necessitating considerable effort in gathering annotations. RL is also used in LLM reasoning, which will be discussed in detail in the next paragraph. 

**Reinforcement learning for large language models.** RL has garnered much attention in LLM alignment (Askell et al., 2021; Bai et al., 2022; Ouyang et al., 2022; Zheng et al., 2023; Wang et al., 2024a), and has been applied in many other tasks like summarization (Ouyang et al., 2022; Stiennon et al., 2020), web navigation (Nakano et al., 2021; Qin et al., 2023) and machine translation (Gul¨ c¸ehre et al., 2023). There are also some work explores enhancing model’s reasoning capabilities with RL, based on outcome supervision or process supervision (Lightman et al., 2023; Luo et al., 2023; Wang et al., 2023a; Luong et al., 2024). Furthermore, these two types of supervision are also utilized to perform answer reranking at inference time (Uesato et al., 2022; Cobbe et al., 2021; Yu et al., 2023a), which involves training a reward model based on either outcome or process supervision to rank multiple generated solutions and select the top one. These approaches are orthogonal to our method and can be seamlessly integrated for further improvement. 

**Reinforcement learning with reverse curriculum.** In goal-oriented RL, reverse curriculum learning (Florensa et al., 2018a;b) effectively addresses the problem of sparse rewards (Ladosz et al., 2022). This method involves initially training the agent achieve the target from a starting point near the target, and subsequently relocating the starting point to more distant positions (Wu et al., 2021). Notably, methods that sample starting points from intermediate states of quality demonstrations (Subramanian et al., 2016a; Popov et al., 2017) and trajectories are commonly applied to tasks like the games (Hosu & Rebedea, 2016; Salimans & Chen, 2018) and robotics (Peng et al., 2018; Nair et al., 2018; Plappert et al., 2018). We employ such strategy to address the issue of sparse rewards in outcome supervision of LLM reasoning and provide an effect akin to process supervision. 

## **7. Conclusion and Future Work** 

In this work, we rethink the existing supervision paradigms of reinforcement learning for large language model reasoning, and propose R[3] that employs only outcome supervision to achieve the benefits of process supervision via reverse curriculum reinforcement learning. We perform thorough experiments on natural language-based and program-based CoT to demonstrate the effectiveness of our method. Moreover, we conduct detailed ablation and analysis to showcase the stability and operating mechanism of our method. In the future, we will attempt to scale up the model size for better performance. Additionally, we will explore the impact of training data with larger scale and diversity on R[3] . 

8 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

## **Impact Statements** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

Linguistics, 2015. doi: 10.18653/V1/D15-1075. URL https://doi.org/10.18653/v1/d15-1075. 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

## **References** 

- Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Kernion, J., Ndousse, K., Olsson, C., Amodei, D., Brown, T. B., Clark, J., McCandlish, S., Olah, C., and Kaplan, J. A general language assistant as a laboratory for alignment. _CoRR_ , abs/2112.00861, 2021. URL https://arxiv.org/abs/2112.00861. 

- Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., Showk, S. E., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T. B., Clark, J., McCandlish, S., Olah, C., Mann, B., and Kaplan, J. Training a helpful and harmless assistant with reinforcement learning from human feedback. _CoRR_ , abs/2204.05862, 2022. doi: 10.48550/ARXIV.2204.05862. URL https: //doi.org/10.48550/arXiv.2204.05862. 

- Bertsekas, D. _Dynamic programming and optimal control: Volume I_ , volume 4. Athena scientific, 2012. 

- Bommasani, R., Hudson, D. A., Adeli, E., Altman, R. B., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, R., Chatterji, N. S., Chen, A. S., Creel, K., Davis, J. Q., Demszky, D., Donahue, C., Doumbouya, M., Durmus, E., Ermon, S., Etchemendy, J., Ethayarajh, K., Fei-Fei, L., Finn, C., Gale, T., Gillespie, L., Goel, K., Goodman, N. D., Grossman, S., Guha, N., Hashimoto, T., Henderson, P., Hewitt, J., Ho, D. E., Hong, J., Hsu, K., Huang, J., Icard, T., Jain, S., Jurafsky, D., Kalluri, P., Karamcheti, S., Keeling, G., Khani, F., Khattab, O., Koh, P. W., Krass, M. S., Krishna, R., Kuditipudi, R., and et al. On the opportunities and risks of foundation models. _CoRR_ , abs/2108.07258, 2021. URL https://arxiv.org/abs/2108.07258. 

- Bowman, S. R., Angeli, G., Potts, C., and Manning, C. D. A large annotated corpus for learning natural language inference. In Marquez, L., Callison-Burch, C., Su, J., Pighin,` D., and Marton, Y. (eds.), _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015_ , pp. 632–642. The Association for Computational 

- Chu, Z., Chen, J., Chen, Q., Yu, W., He, T., Wang, H., Peng, W., Liu, M., Qin, B., and Liu, T. A survey of chain of thought reasoning: Advances, frontiers and future. _CoRR_ , abs/2309.15402, 2023. doi: 10.48550/ARXIV. 2309.15402. URL https://doi.org/10.48550/ arXiv.2309.15402. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168. 

- Florensa, C., Held, D., Wulfmeier, M., Zhang, M., and Abbeel, P. Reverse curriculum generation for reinforcement learning. In _1st Annual Conference on Robot Learning, CoRL 2017, Mountain View, California, USA, November 13-15, 2017, Proceedings_ , volume 78 of _Proceedings of Machine Learning Research_ , pp. 482–495. PMLR, 2017. URL http://proceedings.mlr.press/ v78/florensa17a.html. 

- Florensa, C., Held, D., Geng, X., and Abbeel, P. Automatic Goal Generation for Reinforcement Learning Agents, July 2018a. URL http://arxiv.org/abs/1705. 06366. arXiv:1705.06366 [cs]. 

- Florensa, C., Held, D., Wulfmeier, M., Zhang, M., and Abbeel, P. Reverse Curriculum Generation for Reinforcement Learning, July 2018b. URL http://arxiv. org/abs/1707.05300. arXiv:1707.05300 [cs]. 

- Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. PAL: program-aided language models. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 10764–10799. PMLR, 2023. URL https://proceedings.mlr. press/v202/gao23f.html. 

- Gou, Z., Shao, Z., Gong, Y., Shen, Y., Yang, Y., Huang, M., Duan, N., and Chen, W. Tora: A tool-integrated reasoning agent for mathematical problem solving. _CoRR_ , abs/2309.17452, 2023. doi: 10.48550/ARXIV. 2309.17452. URL https://doi.org/10.48550/ arXiv.2309.17452. 

9 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

- Gul¨ c¸ehre, C¸ ., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., Macherey, W., Doucet, A., Firat, O., and de Freitas, N. Reinforced self-training (rest) for language modeling. _CoRR_ , abs/2308.08998, 2023. doi: 10.48550/ARXIV.2308.08998. URL https://doi. org/10.48550/arXiv.2308.08998. 

- Hecham, A., Bisquert, P., and Croitoru, M. On a flexible representation for defeasible reasoning variants. In Andre,´ E., Koenig, S., Dastani, M., and Sukthankar, G. (eds.), _Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, AAMAS 2018, Stockholm, Sweden, July 10-15, 2018_ , pp. 1123–1131. International Foundation for Autonomous Agents and Multiagent Systems Richland, SC, USA / ACM, 2018. URL http://dl.acm.org/ citation.cfm?id=3237863. 

- Hosu, I.-A. and Rebedea, T. Playing atari games with deep reinforcement learning and human checkpoint replay, 2016. 

- Jie, Z., Luong, T. Q., Zhang, X., Jin, X., and Li, H. Design of chain-of-thought in math problem solving. _CoRR_ , abs/2309.11054, 2023. doi: 10.48550/ARXIV. 2309.11054. URL https://doi.org/10.48550/ arXiv.2309.11054. 

- Kakade, S. M. and Langford, J. Approximately optimal approximate reinforcement learning. In Sammut, C. and Hoffmann, A. G. (eds.), _Machine Learning, Proceedings of the Nineteenth International Conference (ICML 2002), University of New South Wales, Sydney, Australia, July 8-12, 2002_ , pp. 267–274. Morgan Kaufmann, 2002. 

- Kazemi, M., Yuan, Q., Bhatia, D., Kim, N., Xu, X., Imbrasaite, V., and Ramachandran, D. Boardgameqa: A dataset for natural language reasoning with contradictory information. _CoRR_ , abs/2306.07934, 2023. doi: 10.48550/ARXIV.2306.07934. URL https://doi. org/10.48550/arXiv.2306.07934. 

- Kim, S., Joo, S. J., Kim, D., Jang, J., Ye, S., Shin, J., and Seo, M. The cot collection: Improving zero-shot and few-shot learning of language models via chain-of-thought finetuning. In Bouamor, H., Pino, J., and Bali, K. (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pp. 12685–12708. Association for Computational Linguistics, 2023. URL https:// aclanthology.org/2023.emnlp-main.782. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. In _NeurIPS_ , 2022. URL http://papers. nips.cc/paper_files/paper/2022/hash/ 

- html. 

- Ladosz, P., Weng, L., Kim, M., and Oh, H. Exploration in deep reinforcement learning: A survey. _Information Fusion_ , 85:1–22, September 2022. ISSN 1566-2535. doi: 10.1016/j.inffus.2022.03. 003. URL https://www.sciencedirect.com/ science/article/pii/S1566253522000288. 

- Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. H. RACE: large-scale reading comprehension dataset from examinations. In Palmer, M., Hwa, R., and Riedel, S. (eds.), _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017_ , pp. 785–794. Association for Computational Linguistics, 2017. doi: 10.18653/V1/D17-1082. URL https: //doi.org/10.18653/v1/d17-1082. 

- Le, H., Wang, Y., Gotmare, A. D., Savarese, S., and Hoi, S. C. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL http://papers. nips.cc/paper_files/paper/2022/hash/ html. 

- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. _CoRR_ , abs/2305.20050, 2023. doi: 10.48550/ARXIV. 2305.20050. URL https://doi.org/10.48550/ arXiv.2305.20050. 

- Lu, X., Roy, B. V., Dwaracherla, V., Ibrahimi, M., Osband, I., and Wen, Z. Reinforcement learning, bit by bit. _Found. Trends Mach. Learn._ , 16(6):733–865, 2023. doi: 10.1561/ 2200000097. URL https://doi.org/10.1561/ 2200000097. 

- Luo, H., Sun, Q., Xu, C., Zhao, P., Lou, J., Tao, C., Geng, X., Lin, Q., Chen, S., and Zhang, D. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. _CoRR_ , abs/2308.09583, 2023. doi: 10.48550/ARXIV.2308.09583. URL https: //doi.org/10.48550/arXiv.2308.09583. 

- Luong, T. Q., Zhang, X., Jie, Z., Sun, P., Jin, X., and Li, H. Reft: Reasoning with reinforced fine-tuning, 2024. 

10 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

   - 2021.naacl-main.168. URL https://doi.org/10. 18653/v1/2021.naacl-main.168. 

- Maher, M. J., Tachmazidis, I., Antoniou, G., Wade, S., and Cheng, L. Rethinking defeasible reasoning: A scalable approach. _Theory Pract. Log. Program._ , 20(4):552–586, 2020. doi: 10.1017/S1471068420000010. URL https: //doi.org/10.1017/S1471068420000010. 

   - Peng, X. B., Abbeel, P., Levine, S., and van de Panne, M. DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills. _ACM Transactions on Graphics_ , 37(4):1–14, August 2018. ISSN 0730-0301, 1557-7368. doi: 10.1145/3197517. 3201311. URL http://arxiv.org/abs/1804. 02717. arXiv:1804.02717 [cs]. 

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In Balcan, M. and Weinberger, K. Q. (eds.), _Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016_ , volume 48 of _JMLR Workshop and Conference Proceedings_ , pp. 1928–1937. JMLR.org, 2016. URL http://proceedings.mlr.press/ v48/mniha16.html. 

   - Pitis, S. Failure modes of learning reward models for llms and other sequence models. In _ICML 2023 Workshop The Many Facets of Preference-Based Learning_ , 2023. 

   - Plappert, M., Andrychowicz, M., Ray, A., McGrew, B., Baker, B., Powell, G., Schneider, J., Tobin, J., Chociej, M., Welinder, P., Kumar, V., and Zaremba, W. Multi-goal reinforcement learning: Challenging robotics environments and request for research, 2018. 

- Nair, A., McGrew, B., Andrychowicz, M., Zaremba, W., and Abbeel, P. Overcoming Exploration in Reinforcement Learning with Demonstrations, February 2018. URL http://arxiv.org/abs/1709. 10089. arXiv:1709.10089 [cs]. 

   - Pollock, J. L. Defeasible reasoning. _Cognitive science_ , 11 (4):481–518, 1987. 

- Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., Jiang, X., Cobbe, K., Eloundou, T., Krueger, G., Button, K., Knight, M., Chess, B., and Schulman, J. Webgpt: Browser-assisted question-answering with human feedback. _CoRR_ , abs/2112.09332, 2021. URL https://arxiv.org/abs/2112.09332. 

   - Popov, I., Heess, N., Lillicrap, T., Hafner, R., Barth-Maron, G., Vecerik, M., Lampe, T., Tassa, Y., Erez, T., and Riedmiller, M. Data-efficient Deep Reinforcement Learning for Dexterous Manipulation, April 2017. URL http:// arxiv.org/abs/1704.03073. arXiv:1704.03073 [cs]. 

   - Qiao, S., Ou, Y., Zhang, N., Chen, X., Yao, Y., Deng, S., Tan, C., Huang, F., and Chen, H. Reasoning with language model prompting: A survey. _CoRR_ , abs/2212.09597, 2022. doi: 10.48550/arXiv.2212.09597. URL https: //doi.org/10.48550/arXiv.2212.09597. 

- OpenAI. GPT-4 technical report. _CoRR_ , abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https: //doi.org/10.48550/arXiv.2303.08774. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, Qin, Y., Cai, Z., K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P. F., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback. In _NeurIPS_ , 2022. URL http://papers. nips.cc/paper_files/paper/2022/hash/ _putational_ b1efde53be364a73914f58805a001731-Abstract-Conference. html. 

   - Qin, Y., Cai, Z., Jin, D., Yan, L., Liang, S., Zhu, K., Lin, Y., Han, X., Ding, N., Wang, H., Xie, R., Qi, F., Liu, Z., Sun, M., and Zhou, J. Webcpm: Interactive web search for chinese long-form question answering. In Rogers, A., Boyd-Graber, J. L., and Okazaki, N. (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pp. 8968–8988. Association for Computational Linguistics, 2023. doi: 10. 18653/V1/2023.ACL-LONG.499. URL https://doi. org/10.18653/v1/2023.acl-long.499. 

- Patel, A., Bhattamishra, S., and Goyal, N. Are NLP models really able to solve simple math word problems? In Toutanova, K., Rumshisky, A., Zettlemoyer, L., Hakkani-Tur,¨ D., Beltagy, I., Bethard, S., Cotterell, R., Chakraborty, T., and Zhou, Y. (eds.), _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021_ , pp. 2080–2094. Association for Computational Linguistics, 2021. doi: 10.18653/v1/ 

- Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, H. F., Aslanides, J., Henderson, S., Ring, R., Young, S., Rutherford, E., Hennigan, T., Menick, J., Cassirer, A., Powell, R., van den Driessche, G., Hendricks, L. A., Rauh, M., Huang, P., Glaese, A., Welbl, J., Dathathri, S., Huang, S., Uesato, J., Mellor, J., Higgins, I., Creswell, A., McAleese, N., Wu, A., Elsen, E., Jayakumar, S. M., Buchatskaya, E., Budden, D., Sutherland, E., Simonyan, 

11 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

- K., Paganini, M., Sifre, L., Martens, L., Li, X. L., Kuncoro, A., Nematzadeh, A., Gribovskaya, E., Donato, D., Lazaridou, A., Mensch, A., Lespiau, J., Tsimpoukelli, M., Grigorev, N., Fritz, D., Sottiaux, T., Pajarskas, M., Pohlen, T., Gong, Z., Toyama, D., de Masson d’Autume, C., Li, Y., Terzi, T., Mikulik, V., Babuschkin, I., Clark, A., de Las Casas, D., Guy, A., Jones, C., Bradbury, J., Johnson, M. J., Hechtman, B. A., Weidinger, L., Gabriel, I., Isaac, W., Lockhart, E., Osindero, S., Rimell, L., Dyer, C., Vinyals, O., Ayoub, K., Stanway, J., Bennett, L., Hassabis, D., Kavukcuoglu, K., and Irving, G. Scaling language models: Methods, analysis & insights from training gopher. _CoRR_ , abs/2112.11446, 2021. URL https://arxiv.org/abs/2112.11446. 

- Rasley, J., Rajbhandari, S., Ruwase, O., and He, Y. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Gupta, R., Liu, Y., Tang, J., and Prakash, B. A. (eds.), _KDD_ 

- _’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020_ , pp. 3505–3506. ACM, 2020. doi: 10.1145/3394486.3406703. URL https://doi. org/10.1145/3394486.3406703. 

- Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan,` X. E., Adi, Y., Liu, J., Remez, T., Rapin, J., Kozhevnikov, A., Evtimov, I., Bitton, J., Bhatt, M., Canton-Ferrer, C., Grattafiori, A., Xiong, W., Defossez, A., Copet, J., Azhar,´ F., Touvron, H., Martin, L., Usunier, N., Scialom, T., and Synnaeve, G. Code llama: Open foundation models for code. _CoRR_ , abs/2308.12950, 2023. doi: 10.48550/ ARXIV.2308.12950. URL https://doi.org/10. 48550/arXiv.2308.12950. 

- Ruder, S. An overview of multi-task learning in deep neural networks. _CoRR_ , abs/1706.05098, 2017. URL http: //arxiv.org/abs/1706.05098. 

- Salimans, T. and Chen, R. Learning Montezuma’s Revenge from a single demonstration, 2018. 

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. _CoRR_ , abs/1707.06347, 2017. URL http://arxiv. org/abs/1707.06347. 

- Shen, J., Yin, Y., Li, L., Shang, L., Jiang, X., Zhang, M., and Liu, Q. Generate & rank: A multi-task framework for math word problems. In Moens, M., Huang, X., Specia, L., and Yih, S. W. (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021_ , pp. 2269–2279. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021. FINDINGS-EMNLP.195. URL https://doi.org/ 

- 10.18653/v1/2021.findings-emnlp.195. 

- Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E. H., Scharli,¨ N., and Zhou, D. Large language models can be easily distracted by irrelevant context. _CoRR_ , abs/2302.00093, 2023. doi: 10.48550/arXiv. 2302.00093. URL https://doi.org/10.48550/ arXiv.2302.00093. 

- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize from human feedback. _CoRR_ , abs/2009.01325, 2020. URL https://arxiv.org/ abs/2009.01325. 

- Subramanian, K., Isbell, C. L., and Thomaz, A. L. Exploration from Demonstration for Interactive Reinforcement Learning. In _Proceedings of the 2016 International Conference on Autonomous Agents & Multiagent Systems_ , AAMAS ’16, pp. 447–456, Richland, SC, May 2016a. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 978-1-4503-4239-1. 

- Subramanian, K., Jr., C. L. I., and Thomaz, A. L. Exploration from demonstration for interactive reinforcement learning. In Jonker, C. M., Marsella, S., Thangarajah, J., and Tuyls, K. (eds.), _Proceedings of the 2016 International Conference on Autonomous Agents & Multiagent Systems, Singapore, May 9-13, 2016_ , pp. 447– 456. ACM, 2016b. URL http://dl.acm.org/ citation.cfm?id=2936990. 

- Sutton, R. S., Barto, A. G., et al. _Introduction to reinforcement learning_ , volume 135. MIT press Cambridge, 1998. 

- Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpaca: A strong, replicable instruction-following model. https://crfm.stanford.edu/2023/ 03/13/alpaca.html, 2023. 

- Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E., Poulton, A., Kerkez, V., and Stojnic, R. Galactica: A large language model for science. _CoRR_ , abs/2211.09085, 2022. doi: 10.48550/ARXIV. 2211.09085. URL https://doi.org/10.48550/ arXiv.2211.09085. 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Canton-Ferrer, C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, 

12 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

for Computational Linguistics, 2018. doi: 10.18653/V1/ N18-1101. URL https://doi.org/10.18653/ v1/n18-1101. 

- R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models. _CoRR_ , abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https: //doi.org/10.48550/arXiv.2307.09288. 

   - Wu, J., Zhang, D., Zhong, S., and Qiao, H. Trajectorybased split hindsight reverse curriculum learning. In _2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pp. 3971–3978. IEEE Press, 2021. doi: 10.1109/IROS51168.2021.9636842. URL https://doi.org/10.1109/IROS51168. 2021.9636842. 

- Uesato, J., Kushman, N., Kumar, R., Song, H. F., Siegel, N. Y., Wang, L., Creswell, A., Irving, G., and Higgins, I. Solving math word problems with processand outcome-based feedback. _CoRR_ , abs/2211.14275, 2022. doi: 10.48550/ARXIV.2211.14275. URL https: //doi.org/10.48550/arXiv.2211.14275. 

   - Xi, Z., Jin, S., Zhou, Y., Zheng, R., Gao, S., Liu, J., Gui, T., Zhang, Q., and Huang, X. Self-polish: Enhance reasoning in large language models via problem refinement. In Bouamor, H., Pino, J., and Bali, K. (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pp. 11383–11406. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023. findings-emnlp.762. 

- Wang, B., Zheng, R., Chen, L., Liu, Y., Dou, S., Huang, C., Shen, W., Jin, S., Zhou, E., Shi, C., Gao, S., Xu, N., Zhou, Y., Fan, X., Xi, Z., Zhao, J., Wang, X., Ji, T., Yan, H., Shen, L., Chen, Z., Gui, T., Zhang, Q., Qiu, X., Huang, X., Wu, Z., and Jiang, Y. Secrets of RLHF in large language models part II: reward modeling. _CoRR_ , abs/2401.06080, 2024a. doi: 10.48550/ARXIV.2401.06080. URL https: //doi.org/10.48550/arXiv.2401.06080. 

   - Xie, Y., Kawaguchi, K., Zhao, Y., Zhao, X., Kan, M.-Y., He, J., and Xie, Q. Self-evaluation guided beam search for reasoning. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- Wang, B., Zheng, R., Chen, L., Liu, Y., Dou, S., Huang, C., Shen, W., Jin, S., Zhou, E., Shi, C., et al. Secrets of rlhf in large language models part ii: Reward modeling. _arXiv preprint arXiv:2401.06080_ , 2024b. 

   - Ye, X. and Durrett, G. The unreliability of explanations in few-shot in-context learning. _CoRR_ , abs/2205.03401, 2022. doi: 10.48550/arXiv.2205.03401. URL https: //doi.org/10.48550/arXiv.2205.03401. 

- Wang, P., Li, L., Chen, L., Song, F., Lin, B., Cao, Y., Liu, T., and Sui, Z. Making large language models better reasoners with alignment. _CoRR_ , abs/2309.02144, 2023a. doi: 10.48550/ARXIV.2309.02144. URL https:// doi.org/10.48550/arXiv.2309.02144. 

   - Yu, F., Gao, A., and Wang, B. Outcome-supervised verifiers for planning in mathematical reasoning. _CoRR_ , abs/2311.09724, 2023a. doi: 10.48550/ARXIV. 2311.09724. URL https://doi.org/10.48550/ arXiv.2311.09724. 

- Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. _arXiv preprint arXiv:2312.08935_ , 2023b. 

   - Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A., and Liu, W. Metamath: Bootstrap your own mathematical questions for large language models. _CoRR_ , abs/2309.12284, 2023b. doi: 10.48550/ARXIV.2309.12284. URL https://doi. org/10.48550/arXiv.2309.12284. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., language models. 

- Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. URL http://papers. nips.cc/paper_files/paper/2022/hash/ Yuan, Z., Yuan, 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.C. html. 

   - Yuan, Z., Yuan, H., Li, C., Dong, G., Tan, C., and Zhou, C. Scaling relationship on learning mathematical reasoning with large language models. _CoRR_ , abs/2308.01825, 2023. doi: 10.48550/ARXIV.2308.01825. URL https: //doi.org/10.48550/arXiv.2308.01825. 

- Williams, A., Nangia, N., and Bowman, S. R. A broadcoverage challenge corpus for sentence understanding through inference. In Walker, M. A., Ji, H., and Stent, A. (eds.), _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACLHLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers)_ , pp. 1112–1122. Association 

- Yue, X., Qu, X., Zhang, G., Fu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. Mammoth: Building math generalist models through hybrid instruction tuning. _CoRR_ , abs/2309.05653, 2023. doi: 10.48550/ARXIV. 2309.05653. URL https://doi.org/10.48550/ arXiv.2309.05653. 

13 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

- Zellers, R., Bisk, Y., Schwartz, R., and Choi, Y. SWAG: A large-scale adversarial dataset for grounded commonsense inference. In Riloff, E., Chiang, D., Hockenmaier, J., and Tsujii, J. (eds.), _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018_ , pp. 93–104. Association for Computational Linguistics, 2018. doi: 10.18653/v1/d18-1009. URL https://doi.org/10.18653/v1/d18-1009. 

- Zhang, Y. and Yang, Q. A survey on multi-task learning. _IEEE Trans. Knowl. Data Eng._ , 34(12):5586–5609, 2022. doi: 10.1109/TKDE.2021.3070203. URL https:// doi.org/10.1109/TKDE.2021.3070203. 

- Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., et al. Siren’s song in the ai ocean: A survey on hallucination in large language models. _arXiv preprint arXiv:2309.01219_ , 2023. 

- Zheng, R., Dou, S., Gao, S., Hua, Y., Shen, W., Wang, B., Liu, Y., Jin, S., Liu, Q., Zhou, Y., Xiong, L., Chen, L., Xi, Z., Xu, N., Lai, W., Zhu, M., Chang, C., Yin, Z., Weng, R., Cheng, W., Huang, H., Sun, T., Yan, H., Gui, T., Zhang, Q., Qiu, X., and Huang, X. Secrets of RLHF in large language models part I: PPO. _CoRR_ , abs/2307.04964, 2023. doi: 10.48550/ARXIV.2307.04964. URL https: //doi.org/10.48550/arXiv.2307.04964. 

- Zhong, V., Xiong, C., and Socher, R. Seq2sql: Generating structured queries from natural language using reinforcement learning. _CoRR_ , abs/1709.00103, 2017. URL http://arxiv.org/abs/1709.00103. 

- Zhou, A., Wang, K., Lu, Z., Shi, W., Luo, S., Qin, Z., Lu, S., Jia, A., Song, L., Zhan, M., and Li, H. Solving challenging math word problems using GPT-4 code interpreter with code-based self-verification. _CoRR_ , abs/2308.07921, 2023. doi: 10.48550/ARXIV.2308.07921. URL https: //doi.org/10.48550/arXiv.2308.07921. 

14 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

## **A. Algorithm** 

**Algorithm 1** R[3] 

**Input:** Policy language model _πθ_ , training data _D_ with _N_ data points, maximum rollout length _T_ , number of stages _M_ , outcome-based reward function _rfo_ ( _·_ ). ] Initialize policy model _πθ_ ; **Procedure** Construct reverse curriculum datasets: _Dlist ←_ [ ]; **for** stage _m ←_ 1 _...M_ **do** _D[m] ←∅_ ; **for** data point ( _s_ 0 _,_ **a** ) = _{s_ 0 _, a_ 1 _, ..., sT −_ 1 _, aT }_ in _D_ **do** Select an intermediate state _sk_ as the start state for stage _m_ ; _D[m] ←D[m] ∪_ ( _sk,_ **a** _k_ +1: _T_ ); **end end** _Dlist._ append( _D[m]_ ) **Procedure** Rinforcement learning in a reverse, staged manner: **for** _D[m]_ in _Dlist_ **do** Perform Reinforcement Learning with _πθ_ and _rfo_ ( _·_ ) on _D[m]_ ; **end Procedure** Rinforcement learning with mixed stages: _M D[mixed] ←_ � _Dlist_ [ _m_ ]; _m_ =1 Perform Reinforcement Learning with _πθ_ and _rfo_ ( _·_ ) on _D[mixed]_ ; 

## **B. Additional Experiments** 

## **B.1. Ablation Study** 

In Table 6, we conduct supplementary ablation studies on Section 5.1, providing results on BGQAmain, MNLI and race@High datasets. We can observe that if we set _β_ = 0 _._ 4, imposing a stronger KL constraint, there will be a noticeable decrease in performance. If we set _β_ to 0 or 0 _._ 1, the performance loss is not as pronounced but still falls below the optimal result. 

_Table 6._ Ablation study on BGQAmain, MNLI and race@High, by default _β_ = 0 _._ 3. 

|**Dataset**|**Performance**|
|---|---|
|BGQAmain|**67**_._**8**|
|- KL coeffcient_β_ = 0|66_._3|
|- KL coeffcient_β_ = 0_._2|66_._5|
|- KL coeffcient _β_ = 0_._4|65_._5|
|MNLI|**72**_._**3**|
|- KL coeffcient_β_ = 0|70_._0|
|- KL coeffcient_β_ = 0_._2|70_._0|
|- KL coeffcient _β_ = 0_._4|68_._4|
|race@High|**68**_._**5**|
|- KL coeffcient_β_ = 0|64_._5|
|- KL coeffcient_β_ = 0_._2|66_._5|
|- KL coeffcient_β_ = 0_._4|64_._5|



## **B.2. Experimental Results of Data Scale** 

As a supplement to Section 5.4, Table 7 presents detailed values of performance. The table illustrates that R[3] achieves performance comparable to full-data training of SFT and RL baselines, using only a fraction of the available data. 

15 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

_Table 7._ Impact of data scale 

||_Table 7._ Impact of data scale||
|---|---|---|
|Dataset|R3 with Data Scaling (%)<br>10<br>20<br>40<br>60<br>80<br>100|Baseline (Full Train Set)<br>SFT<br>RL|
|GSM8K<br>MNLI<br>race@High<br>BGQA|40.2<br>43.4<br>44.7<br>47.4<br>48.8<br>50.5<br>65.4<br>66.2<br>66.9<br>68.5<br>69.2<br>72.3<br>60.5<br>61.0<br>62.0<br>62.5<br>64.5<br>68.5<br>62.5<br>64.8<br>65.3<br>67.3<br>67.3<br>67.8|41.6<br>44.7<br>65.4<br>66.2<br>60.5<br>61.5<br>62.5<br>65.5|



## **B.3. Impact of different parts of data.** 

Table 8 and Table 9 present the accuracy achieved when training the model without specific data parts. Notably, columns 1 through 5 ( For race@High, columns1 through 6 ) signify the ascending difficulty levels of excluded training data, with higher part numbers indicating greater difficulty. The “All Parts” column reflects accuracy when utilizing the entire dataset. Furthermore, based on the results in Section 5.1, we can conclude that for the race@High dataset, optimal performance can be achieved when the number of intermediate states _M_ is set to 6. Therefore, we supplement experiments with race@High containing 6 data parts in Table 9. 

_Table 8._ Comparison of accuracy in training on different data parts 

|Dataset|w/o Part<br>All Parts<br>1<br>2<br>3<br>4<br>5|Baseline (Full Train Set)<br>SFT<br>RL|
|---|---|---|
|GSM8K<br>MNLI<br>race@High<br>BGQA|48.9<br>48.0<br>47.2<br>46.9<br>44.7<br>50.5<br>71.5<br>69.2<br>66.9<br>66.9<br>65.4<br>72.3<br>64.5<br>64.0<br>64.0<br>63.0<br>61.0<br>65.0<br>67.4<br>65.8<br>64.5<br>63.8<br>62.9<br>67.8|41.6<br>44.7<br>65.4<br>66.2<br>60.5<br>61.5<br>62.5<br>65.5|



_Table 9._ Performance for race@High with 6 intermediate states 

|Dataset|w/o Part<br>All Parts<br>1<br>2<br>3<br>4<br>5<br>6|Baseline (Full Train Set)<br>SFT<br>RL|
|---|---|---|
|race@High|65.5<br>63.5<br>63.5<br>63.0<br>61.0<br>62.0<br>68.5|60.5<br>61.5|



## **C. Prompts** 

We follow the Alpaca (Taori et al., 2023) prompts format in our experiments. The specific prompts are as follows. 

_Listing 1._ Prompts used in R[3] experiments 

Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction: {instruction} 

### Response: 

## **D. Case Study** 

We provide case studies of R[3] and vanilla RL on GSM8K-CoT, GSM8K-P-CoT and MNLI Datasets. Wrong reasoning steps are highlighted in **red** , and reasoning steps corrected by the R[3] method are indicated in **green** . It is evident that the model 

16 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

trained by R[3] has clearer logic and more accurate reasoning when facing complex reasoning tasks, often achieving better task completion. 

**Question** : You can buy a movie super ticket for $20 that includes rights to watch any movie and you get one popcorn with a soda for only $1 extra. Or you can buy a movie ticket for $12 and buy one popcorn and a soda at regular price. What's the regular price of popcorn if the soda costs $3 and you didn't buy the super ticket and ended up saving $2? 

**RL  Baseline R[3]** `If I buy a movie super ticket for $20 that If you buy the super ticket, you pay $20 + includes rights to watch any movie and I $1 = $21 for popcorn and soda. If you buy get one popcorn with a soda for only $1 the regular ticket, you pay $12 + $3 = $15 extra, then the regular price of popcorn for popcorn and soda. If you save $2, that is $20 + $1 = $21. If I buy a movie ticket means you paid $21 - $2 = $19 for popcorn for $12 and buy one popcorn and a soda at and soda on the super ticket.` **`So the`** `regular price, then the regular price of` **`regular price of popcorn is $15 - $19 = $4.`** `popcorn is $12 + $3 = $15.` **`If I didn't buy the super ticket and ended up saving $2, ### 4 then the regular price of popcorn is $21 - $2 = $19. ### 19`** 

**Ground truth** : The super ticket costs $20 and for $1 extra I can get a popcorn and a soda, there I'd pay in total $20+$1 = $<<20+1=21>>21. If I didn't buy the ($21) super ticket but bought everything included on it and still saved $2, then I spent in total $21-$2 = $<<21-2=19>>19. If the regular movie ticket costs $12 and the soda costs $3 then the popcorn would cost $19-$12-$3=$4 

_Figure 7._ Comparison of RL Baseline and **R**[3] on GSM8K-CoT. 

17 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

**Question** : Stephen placed an online order for groceries. His final bill came to $40.00. Because this was through a delivery vendor, they tacked on a 25% fee to his final total and charged him $3.00 in delivery fees. Stephen also added a $4.00 tip. After the extra fees, what was the final price of Stephen's groceries? 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0018-02.png)


**----- Start of picture text -----**<br>
RL  Baseline R [3]<br>def solution( ): def solution( ):<br>groceries_cost = 40.00 bill_initial = 40<br>delivery_fee = 3.00 fee_percent = 0.25<br>tip = 4.00 fee_amount = bill_initial * fee_percent<br>delivery_fee_percent = 0.25  bill_after_fee = bill_initial + fee_amount<br>total_cost = groceries_cost + delivery_fee + tip     delivery_fee = 3<br>delivery_fee_amount = total_cost *                      bill_after_delivery_fee = bill_after_fee +<br>                            delivery_fee_percent     delivery_fee<br>total_cost_with_fee = total_cost +  tip = 4<br>                            delivery_fee_amount     bill_after_tip = bill_after_delivery_fee + tip<br>result = total_cost_with_fee     result = bill_after_tip<br>return result return result<br>58.75 57.00<br>Ground truth : 57.00<br>**----- End of picture text -----**<br>


_Figure 8._ Comparison of RL Baseline and **R**[3] on GSM8K-P-CoT. 

18 

**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning** 

**Question** : <Premise> 

Limits on IDA balances range from $4,000 in Virginia to $10,000 in South Carolina and $50,000 in Missouri. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning_images/icml_2024_Training_Large_Language_Models_for_Reasoning_through_Reverse_Curriculum_Reinforcement_Learning.pdf-0019-03.png)


**----- Start of picture text -----**<br>
<Hypothesis><br>The limit on a IDA balance in Missouri is $125<br>Does the premise entail the hypothesis?<br>OPTIONS:<br>- yes<br>- it is not possible to tell<br>- no<br>RL  Baseline R [3]<br>The premise states that the limit on a  The premise states that the limit on a<br>IDA balance in Missouri is $50,000. The  IDA balance in Missouri is $50,000.  The<br>hypothesis also mentions this same number,  hypothesis suggests a limit of $125,<br>but with a different unit of measurement  which is much lower than what was<br>(dollars instead of thousands).  So the  mentioned in the premise.  Hence, it can<br>answer is "yes". ### yes be concluded that the answer is "no".<br>### no<br>**----- End of picture text -----**<br>


**Ground truth** : The premise mentions that the limit on IDA balance in Missouri is $50,000. Hence, it does not entail the hypothesis of a limit being $125 for this location. ### no 

_Figure 9._ Comparison of RL Baseline and **R**[3] on MNLI. 

19 

