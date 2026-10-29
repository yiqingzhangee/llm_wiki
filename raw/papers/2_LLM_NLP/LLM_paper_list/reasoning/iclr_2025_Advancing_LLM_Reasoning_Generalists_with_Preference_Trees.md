Preprint 

## **Advancing LLM Reasoning Generalists with Preference Trees** 

**Lifan Yuan**[1,2] **, Ganqu Cui** _[∗]_[1] _[∗]_ **, Hanbin Wang**[†][3,4] _[∗]_ **, Ning Ding**[1†] **, Xingyao Wang**[2] **, Jia Deng**[5] **, Boji Shan**[6] **, Huimin Chen**[1] **, Ruobing Xie**[7] **, Yankai Lin**[5] **, Zhenghao Liu**[3] **, Bowen Zhou**[1] **, Hao Peng**[2] **, Zhiyuan Liu**[1†] **, Maosong Sun**[1] 

1Tsinghua University 2University of Illinois Urbana-Champaign 3Northeastern University 4 ModelBest.Inc 5 Renmin University of China 6 BUPT 7 Tencent lifan4@illinois.edu cgq22@mails.tsinghua.edu.cn wanghanbinpanda@gmail.com 

## **Abstract** 

We introduce **EURUS** , a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, EURUS models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, EURUS-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. The strong performance of EURUS can be primarily attributed to **ULTRAINTERACT** , our newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. ULTRAINTERACT can be used in both supervised fine-tuning and preference learning. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. ULTRAINTERACT allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. Inspired by this, we derive a novel reward modeling objective which, together with ULTRAINTERACT, leads to a strong reward model.[1] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees_images/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees.pdf-0001-07.png)


**----- Start of picture text -----**<br>
45<br>~7B GPT-4<br>40 ~40B<br>~70B<br>GPT-Series<br>35 Ours<br>Eurus-70B-NCA<br>30 DeepSeek-Coder-33B-Instruct<br>25 MagiCoder-S-DS-6.7B GPT-3.5-Turbo<br>20 DeepSeek-LLM-67B-Chat OpenCI-CL-70B Eurus-7B-KTO<br>OpenCI-DS-6.7B<br>15 CL-70B-Instruct QWen1.5-72B-Chat<br>OpenMath-CL-70B<br>10 OpenMath-Mistral-7B<br>WizardMath-7B -v1.1<br>5 Mistral-7B-Instruct-v0.2 Mixtral-8x7B-Instruct<br>Zephyr-7B- 𝛽<br>0<br>5 10 15 20 25 30 35 40 5545<br>TheoremQA<br>LeetCode<br>**----- End of picture text -----**<br>


Figure 1: Evaluation results on LeetCode and TheoremQA, two challenging OOD coding and math benchmarks with only test sets. Our EURUS-7B is comparable with baselines that are 10x larger and EURUS-70B is the only one on par with GPT-3.5 Turbo. 

> _∗_ Equal Contribution. 

> †Corresponding Authors. 

> 1 Models and data are available at: https://github.com/OpenBMB/Eurus. 

1 

Preprint 

## **1 Introduction** 

Current alignment techniques have significantly advanced the development of open-source large language models (LLMs) that effectively meet user expectations and align with human values (Touvron et al., 2023; Tunstall et al., 2023). On complex reasoning, success has been achieved by specializing models for specific capabilities, such as coding (Wei et al., 2023; Guo et al., 2024a; Zheng et al., 2024) and solving math problems (Fu et al., 2023; Yue et al., 2023; Luo et al., 2023a; Toshniwal et al., 2024). However, these models still fall short, by large margins, of the most advanced proprietary models in their all-around capabilities to tackle a diverse range of challenging problems. We conjecture that this performance gap can be primarily attributed to (1) the lack of high-quality alignment data and (2) the underexploration of preference learning techniques for improving models’ complex reasoning capabilities. In this paper, we take strides towards bridging this gap by addressing both factors and developing EURUS. 

EURUS consists of a suite of LLMs finetuned from Mistral-7B (Jiang et al., 2023a) and CodeLLaMA-70B (Roziere et al., 2023). Across a diverse set of complex reasoning benchmarks that are mostly out-of-distribution (OOD), EURUS achieves state-of-the-art overall performance among all open-source models. In particular, EURUS excels in solving challenging problems that often require sophisticated planning, reasoning, tool integration, and the ability to interact with and learn from the environment and users. As shown in Figure 1, on university-level STEM questions TheoremQA (Chen et al., 2023) and competition-level coding problems LeetCode Contest (Guo et al., 2024a), EURUS-70B significantly outperforms all open-source models, achieving comparable performance to GPT-3.5 Turbo. 

EURUS models are trained on ULTRAINTERACT, our newly-curated, large-scale, and high-quality alignment data specifically designed to improve LLMs’ reasoning capabilities. ULTRAINTERACT consists of a diverse set of instructions spanning math, coding, and logical reasoning problems from 12 established datasets. For each instruction, ULTRAINTERACT collects a preference tree that includes: (1) **Diverse planning strategies in a unified pattern,** such as sequential processing (Wei et al., 2022) and tool creation (Qian et al., 2023), followed by executing step-by-step actions formatted in either text or code, to provide divserse reasoning trajectories. (2) **Multi-turn interaction trajectories with the environment and the critique,** to improve models’ capabilities to learn from feedback and correct previous errors (Wang et al., 2023b). (3) **Paired correct and incorrect actions organized in tree structures,** to facilitate preference learning. In total, ULTRAINTERACT contains 86K instructions and 220K action pairs, where each pair consists of an instruction, a correct response, and an incorrect one. Conceptually, ULTRAINTERACT’s data resemble imbalanced binary trees as shown in Figure 2. 

ULTRAINTERACT can be used in both supervised fine-tuning and preference learning. Our experiments show that, using ULTRAINTERACT along with established datasets in instruction fine-tuning already achieves strong performance. ULTRAINTERACT further facilitates preference learning for reasoning tasks, improving the performance even further with KTO (Ethayarajh et al., 2024) and NCA (Chen et al., 2024a). Surprisingly, applied to an instruction finetuned EURUS model, DPO (Rafailov et al., 2023) hurts the performance. 

Through careful analysis, we provide evidence that the performance in reasoning correlates with the value of rewards of chosen data—a higher final reward often indicates a better reasoning capability. Besides, our investigation suggests that DPO may be less suitable for reasoning tasks than KTO and NCA. Inspired by this fresh finding, we devise a new objective for reward modeling to augment the Bradley-Terry objective (Bradley & Terry, 1952), explicitly encouraging training to increase the absolute rewards of chosen solution and decrease those of rejected data. Furthermore, ULTRAINTERACT leads to our reward model EURUS-RM-7B, which achieves a better correlation with human annotators than all existing models on AutoJ (Li et al., 2023a) and MT-Bench (Zheng et al., 2023), including GPT-4 (OpenAI, 2023). EURUS-RM-7B demonstrates especially strong preference modeling performance on reasoning tasks. 

Checkpoints of our EURUS models, accompanying ULTRAINTERACT alignment data to reproduce this research, will be publicly available. 

2 

Preprint 

## **2 ULTRAINTERACT: Tree-structured Alignment Data for Reasoning** 

Solving complex problems often requires the model’s **U U U U UserInstruction** capability in planning and reasoning, integrating with **A A R C A UnpairedAction** tools, and interacting with **O U O&C C Chosen** and learning from both the **Action** environment and the users. **A A R C R Rejected** This is reflected in ULTRAIN- **Action** TERACT’s design choices: (1) **O U O&C O Observation** Itschallenging,instructionsandareof diverse,a large **A R C R C O&C Observation& Critique** scale (§2.1); (2) It provides multi-turn trajectories that solve the input instruction Figure 2: Left: CodeActInstruct (WangWang et al.,, 2024)) and through multiple turns of Code-Feedback (Zheng et al., 2024); Middle:Zheng et al., 2024); Middle:, 2024); Middle: 2024); Middle:); Middle: HH-RLHF (BaiBai interaction with and learning et al., 2022);, 2022); 2022);); Right: ULTRAINTERACT.LTRAINTERACT.INTERACT.NTERACT.. Each instruction in from the environment and ULTRAINTERACT is constructed as a preference tree.LTRAINTERACT is constructed as a preference tree.INTERACT is constructed as a preference tree.NTERACT is constructed as a preference tree. is constructed as a preference tree. critique. At each turn, it breaks down the problem into smaller ones (§2.2). (3) ULTRAINTERACT includes pairwise data to facilitate preference learning (§2.3). 

Figure 2: Left: CodeActInstruct (WangWang et al.,, 2024)) and Code-Feedback (Zheng et al., 2024); Middle:Zheng et al., 2024); Middle:, 2024); Middle: 2024); Middle:); Middle: HH-RLHF (BaiBai et al., 2022);, 2022); 2022);); Right: ULTRAINTERACT.LTRAINTERACT.INTERACT.NTERACT.. Each instruction in ULTRAINTERACT is constructed as a preference tree.LTRAINTERACT is constructed as a preference tree.INTERACT is constructed as a preference tree.NTERACT is constructed as a preference tree. is constructed as a preference tree. 

Conceptually, ULTRAINTERACT collects a preference tree for each instruction, with the instruction being the root and each action a node (Figure 2). A trajectory is a root-to-leaf path consisting of a sequence of actions. In each preference tree, all nodes of correct actions and all trajectories ending with correct actions can be used for SFT. Paired correct and incorrect nodes or trajectories can be used for preference learning. 

## **2.1 Instruction Selection Emphasizing Complexity, Quality, and Diversity** 

We target three representative reasoning tasks: math problem-solving, code generation, and logical reasoning. The complexity, quality, and diversity of the alignment data are crucial to the model’s performance (Liu et al., 2023). Following Wang et al. (2023b), we select challenging problems that GPT-3.5-Turbo fails to solve. We intentionally restrict the selection of the datasets to those with ground-truth solutions, aiming to ensure highquality oversight signals rather than relying on LLM-as-a-judge annotation (Weyssow et al., 2024). Besides, the gold solutions also serve as references for the critique model to generate feedback. To promote ULTRAINTERACT’s diversity, we pick datasets of different categories. For each dataset, we include distinct reasoning patterns based on question categories or formulations necessary to solve the problems. Table 6 summarizes the datasets selected by ULTRAINTERACT. Except for MATH, none of the training datasets is used in our evaluation. 

## **2.2 Decomposition and Interaction at Each Turn** 

Figure 3 provides an illustrative example. In what follows, we connect the actor model with a Python interpreter as the “environment”. Unless otherwise specified, we use GPT-3.5 Turbo as the actor model. 

Following Wang et al. (2024), the actor model first decomposes the input problem into several sub-problems and then solves each by generating Python code pieces as actions and using the environment to execute them. To promote solution diversity, the actor model randomly samples one reasoning schema in the form of either CoT (Wei et al., 2022) or modularization programming (Qian et al., 2023; Yuan et al., 2023). The actor then generates actions in text or code to solve each sub-problem, with each step being marked by explicit notations. 

Multi-turn interactions with the environment are often necessary to solve challenging problems (Wang et al., 2023b). To improve such capabilities of the models, ULTRAINTERACT collects trajectories in which the actor model interacts with the environment and a critique model (a proxy for user) and refines its action based on their feedback. 

3 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees_images/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees.pdf-0004-01.png)


In a classroom, _**x**_ children have _**7**_ books each. […] what is the value of unknown variable _**x**_ ? 

|**Generation**<br>**1**|**Observation**<br>**2**|**Generation**<br>**4**|**Observation**<br>**5**|
|---|---|---|---|
|**`<thought>`**|`[…]`|**`<thought>`**|`[…]`|
|Step 1: First, let's calculate the total […].|`NameError: name 'x' is`|Step 1: First, create an equation […].|**`Out: 10.0`**|
|Step 2: Then, we need to add the […].<br>Step 3: The sum of the books should […].<br>**`</thought>`**<br>**`<execute>`**<br>`# Step 4: First, let's calculate […].`<br>`total_books_child =`**_`x`_**`*`**_`7`_**<br>`# Step 5: Next, we need to add […].`<br>`total_books = total_books_child +8`<br>`# Step 6: We can now solve for […].`<br>**_`x`_**`= (`**_`78`_**`-`**_`8`_**`) /`**_`7`_**<br>**`</execute>`**|`not defined.`<br>**`Your answer is wrong.`**<br>**`Expert feedback:`**<br>Your thought process is<br>accurate and your error lies<br>in your code. You need to<br>**Critique**<br>**3**|Step 2: Each child has 7 books, so […].<br>Step 3: The teacher brings 8 […].<br>Step 4: Now we can solve for x by […].<br>**`</thought>`**<br>**`<execute>`**<br>`# Step 5: Setting up equation:`<br>`equation =`**_`"7x + 8 = 78"`_**<br>`# Step 6: Solving for x:`<br>**_`x`_**`= (`**_`78`_**`-`**_`8`_**`) /`**_`7`_**<br>**`</execute>`**<br>**`<solution>`**|**`Your answer is`**<br>**`correct.`**<br>**`Expert feedback:`**<br>Good job! You have fixed<br>the error in last turn.<br>Now your answer is<br>**Critique**<br>**6**|
|**`<solution>`**|define `x` first […]|**_`x`_**|correct. […]|
|**_`x`_**<br>**`</solution>`**||**`</solution>`**||



Figure 3: An illustrative example of an ULTRAINTERACT trajectory over two turns. In each turn, the actor model generates step-by-step reasoning chains, and the environment and the critique model provide observations and textual critique respectively. 

The environment receives an action from the actor model along with the interaction history, and then the code interpreter returns two kinds of “ _Observation_ ”: (1) Python execution results, either program outputs or error traceback messages; (2) binary feedback, indicating whether the solution is correct or not. Then, the observations along with the history will be passed to a _critique_ model, which locates the errors and provides suggestions for improvements. To avoid potential bias introduced by self-correction (Wang et al., 2023b; Xu et al., 2024), we adopt a stronger model, GPT-4, as the critique and ensure critique quality by providing GPT-4 with ground truth answers as references. 

This procedure resembles Wang et al. (2024). However, we adopt more diverse reasoning patterns to teach LLMs to learn rationales rather than simply memorizing answers (Mitra et al., 2023), and learn to create and use tools (Qian et al., 2023; Yuan et al., 2023; Qin et al., 2023). Besides, we believe that it is important for LLMs to learn from the feedback provided by the critique rather than solely from observations of the environment. 

## **2.3 Preference Trees Facilitates Preference Learning Across Multiple Turns** 

Unlike open-ended conversations, where human preference is ambiguous and challenging to specify, many reasoning tasks have clear and objective preferences for _correct_ actions. The preference annotation is threfore an evaluation of the correctness of the solutions _conditioning ground truth ones_ , which come with the datasets in ULTRAINTERACT. This eliminates the need for human or LLM-based preference annotation and ensures high data quality. To facilitate preference learning, ULTRAINTERACT pairs correct and incorrect actions. 

**Sampling Paired Correct and Incorrect Actions at Each Turn.** For each instruction in ULTRAINTERACT, we sample, from the actor model, a pair of correct and incorrect actions following §2.2. We follow Cui et al. (2023) to sample the pair from different actor models to ensure response diversity. To prevent models from exploiting shortcuts based on surface features, we exclude instances that fail to pass the Python syntax check. 

Certain challenging problems in ULTRAINTERACT pose difficulties in obtaining correct actions, even using strong actors such as GPT-4, with nearly zero _pass@100_ accuracies. To improve the pass rates of the actor models while keeping the expense under control, we sequentially take the following steps. (1) Directly sampling 20 actions and randomly keeping a correct one, if any. (2) If no correct action is obtained, we repeat the above process up to three times, progressively switching from more cost-effective models to the strong yet expensive GPT-4 Turbo. (3) For the remaining difficult problems where no correct action is acquired after the previous two steps, we provide the actor with ground-truth rationales and answers, and then apply various techniques to elicit correct actions. The specific information provided and the techniques applied vary depending on the tasks (Appendix A.2). 

4 

Preprint 

**Tree-structured Action Pairs Across Multiple Turns.** After each turn, the correct action concludes its trajectory. We expand the _incorrect_ action into the next turn, and have the actor interact with the environment and the critique to refine its solution (§2.2). We then repeat the procedures introduced earlier in this section to collect an additional action pair. By expanding the _incorrect_ action, ULTRAINTERACT can provide data to help models learn from feedback, and collect multiple action pairs for preference learning across multiple turns. 

Conceptually, for every instruction, ULTRAINTERACT constructs a binary preference tree with each action being a node (Figure 2). We cap the tree at a maximum of five turns. 

**Additional Instruction-action Pairs for Challenging Problems.** We believe the challenging instructions that make it to step (3) above can provide valuable training signals. Therefore, for a subset of these problems with multiple ground truth solutions, we further sample additional correct actions to cover all ground truths. Accordingly, we further sample incorrect actions to pair with these additional correct actions, so that they can be used in both supervised fine-tuning and preference learning. 

With the tree-structured data, ULTRAINTERACT enables comparisons at every turn, in contrast to comparing only at the last turn (Bai et al., 2022), and thus can improve the models’ interaction ability. Closing this section, Table 1 summarizes some statistics of ULTRAINTERACT, and more details are in Appendix A.4. 

Table 1: Some statistics of ULTRAINTERACT. 

|**Task**|**Type**<br>**# Instructions**<br>**w/ Interaction?**<br>**w/ Tool?**|**# Turns per Traj.**<br>**# Tokens**<br>**per Traj.**<br>**Avg. # Traj**<br>**per Ins.**<br>**Total**<br>**# Pairs**<br>**# Correct**<br>**Answers**<br>**T1**<br>**T2**<br>**T3**<br>**T4**<br>**T5**|
|---|---|---|
|**Math**|�<br>�<br>22,928<br>�<br>�<br>2,757<br>�<br>�<br>22,639<br>�<br>�<br>2,083|10,440<br>4,122<br>1,898<br>904<br>5,564<br>1,750.0<br>1.0<br>42,780<br>68,033<br>16,154<br>-<br>-<br>-<br>-<br>439.1<br>5.9<br>13,217<br>16,154<br>10,708<br>3,521<br>1,459<br>723<br>6,228<br>1,521.9<br>1.0<br>44,750<br>62,182<br>16,348<br>-<br>-<br>-<br>-<br>538.1<br>7.8<br>12,624<br>16,348|
|**Coding**|�<br>-<br>20,463<br>�<br>-<br>8,495|13,265<br>2,584<br>987<br>379<br>3,248<br>1,728.5<br>1.0<br>18,106<br>22,215<br>92,618<br>-<br>-<br>-<br>-<br>1,070.4<br>5.5<br>78,634<br>92,618|
|**Logic**|�<br>�<br>2,086<br>�<br>�<br>4,467|1,685<br>298<br>72<br>8<br>23<br>1,299.8<br>1.0<br>1,750<br>2,198<br>2,453<br>1,674<br>340<br>0<br>0<br>1,266.7<br>1.0<br>7,958<br>7,231|
|**Total**|-<br>-<br>85,918|163,671<br>12,199<br>4,756<br>2,014<br>15,063<br>1,201.8<br>2.3<br>219,819<br>286,979|



## **3 EURUS: State-of-the-art Open LLMs in Reasoning** 

ULTRAINTERACT helps us develop EURUS, a suite of LLMs and a reward model (RM). 

**Supervised Fine-Tuning.** EURUS-7B-SFT is fine-tuned from Mistral-7B (Jiang et al., 2023a) and EURUS-70B-SFT from CodeLLaMA-70B (Roziere et al., 2023). First, we perform SFT using all correct actions (287K) in ULTRAINTERACT. We find it yields better performance to discard interaction history and train only on correct leaf nodes in each tree. To improve general instruction-following ability, we include into our SFT data mixture UltraChat (Ding et al., 2023), ShareGPT[2] , and OpenOrca (Lian et al., 2023). Please find mixture ratios in Appendix B. 

**Perference Learning.** Based on EURUS-SFT models, we explore three preference learning algorithms, DPO (Rafailov et al., 2023), KTO (Ethayarajh et al., 2024), and NCA (Chen et al., 2024a). Differently from SFT, here we include all multi-turn trajectory pairs in our ULTRAINTERACT (220K) and include all UltraFeedback (Cui et al., 2023) pairs (340K). 

**Reward Modeling.** Similarly to the preference learning, we use all 220K multi-turn trajectory pairs from ULTRAINTERACT; it is further augmented with the 240K single-turn action pairs from ULTRAINTERACT. More details are in the Appendix B. We include all 340K pairs from UltraFeedback and one pair for each instruction from UltraSafety (Guo et al., 2024b), totaling 3K. EURUS-RM-7B is initialized from EURUS-7B-SFT with a new linear layer. 

Our findings in §6 indicate that the absolute values of rewards make a big difference in the models’ reasoning performance. We therefore augment the established Bradley-Terry (BT) objective _L_ BT with an additional term _L_ DR to **d** i **r** ectly increase the reward of the chosen actions for instances from ULTRAINTERACT, and decrease those of the rejected ones: 

> 2 https://huggingface.co/datasets/openchat/openchat ~~s~~ haregpt4 ~~d~~ ataset 

5 

Preprint 

Table 2: Open-source LLM baselines that we compare to. 

|**Type**|**Models**|
|---|---|
||Mistral-7B-Instruct-v0.2 (Jiang et al.,2023a), Zephyr-7B-_β_(Tunstall et al.,2023), OpenChat-3.5-1210 (Wang|
|**General Purpose**|et al.,2023a), Starling-LM-7B-_α_(Zhu et al.,2023), Mixtral-8x7B-Instruct (Jiang et al.,2023a), DeepSeek-|
||LLM-67B-Chat (DeepSeek-AI,2024), QWen1.5-72B-Chat (Bai et al.,2023)|
|**Coding**|Magicoder-S-DS-6.7B (Wei et al.,2023), OpenCodeInterpreter (OpenCI for short, DS-6.7B/CL-70B) (Zheng<br>et al.,2024), DeepSeek-Coder-33B-Instruct (Guo et al.,2024a), and CodeLLaMA-70B-Instruct(Roziere et al.,|
||2023).|
|**Math**|MAmmoTH-7B-Mistral (Yue et al.,2023), WizardMath-7B-v1.1 (Luo et al.,2023a), OpenMath (Mistral-<br>7B/CodeLLaMA-70B) (Toshniwal et al.,2024).|
|_L_ULTRAINTERACT|=_−_log<br>�<br>_σ_<br>�<br>_rθ_(_x_,_yc_)_−rθ_(_x_,_yr_)<br>��<br>_−_log<br>�<br>_σ_<br>�<br>_rθ_(_x_,_yc_)<br>��<br>_−_log<br>�<br>_σ_<br>�_−rθ_(_x_,_yr_)<br>��|
||�<br>��<br>�<br>�<br>��<br>�|
||_L_BT: optimize relative rewards<br>_L_DR: increase_rθ_(_x_,_yc_)and decrease_rθ_(_x_,_yr_)|



For instances from other datasets, we train with _L_ BT. _θ_ denotes the reward model’s parameters, _rθ_ ( _·_ ) and _rθ_ ( _x_ , _yr_ ) the rewards on the chosen and rejected actions respectively. Our ablation study demonstrates the importance of both _L_ BT and _L_ DR. 

## **4 Evaluation of EURUS-7B and EURUS-70B** 

**Evaluation Setup.** We consider both single-turn and multi-turn reasoning. For single-turn evaluation, we consider HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and LeetCode (Guo et al., 2024a) for coding, GSM-Plus (Li et al., 2024), MATH, TheoremQA (Chen et al., 2023), SVAMP (Patel et al., 2021), and ASDiv (Miao et al., 2020) for math, and BBH-Hard (Suzgun et al., 2022) for reasoning. We evaluate with pass@1 accuracy. We also use IFEval (Zhou et al., 2023) to assess the instruction-following ability and report the prompt-level loose score. For multi-turn evaluation, we adopt MINT (Wang et al., 2023b) and only consider the coding and math problems. We report the success rate at Turn 5. Please find further details on evaluation setups and evaluations beyond reasoning in Appendix C. 

As shown in Table 2, we compare our EURUS with general-purpose models, and those specialized in coding and math of various sizes. We also summarize the results of GPT-3.5 Turbo and GPT-4 reported in previous works. 

Table 3: Overall performance. All test sets except MATH are out-of-distribution to our models and most baselines. MAmmoTH, OpenChat, and Starling-LM have been trained on TheoremQA test sets. We ~~strikethrough~~ the contaminated numbers. 

|**Model**|**Coding**|**Math**|**Reasoning**|**Ins-Following**|**Multi-Turn**|**Avg.**|
|---|---|---|---|---|---|---|
||HumanE.<br>MBPP<br>LeetC.|GSM-Plus<br>MATH<br>Theo.QA<br>SVAMP<br>ASDiv|BBH|IFEval|Code<br>Math||
||_∼_7B||||||
|**Mistral-7B-Instruct-v0.2**<br>**Zephyr-7B-**_β_<br>**OpenChat-3.5-1210**<br>**Starling-LM-7B-**_α_<br>**Magicoder-S-DS-6.7B**<br>**OpenCI-DS-6.7B**<br>**MAmmoTH-7B-Mistral**<br>**WizardMath-7B-v1.1**<br>**OpenMath-Mistral-7B**|39.0<br>30.8<br>6.1<br>29.3<br>35.8<br>2.2<br>64.0<br>61.7<br>11.7<br>46.3<br>51.1<br>8.9<br>75.6<br>**70.4**<br>**23.9**<br>**76.8**<br>66.2<br>16.1<br>24.4<br>42.4<br>7.2<br>50.0<br>53.9<br>6.7<br>33.5<br>46.6<br>11.7|15.7<br>9.5<br>8.5<br>42.9<br>49.5<br>23.3<br>5.0<br>7.8<br>19.1<br>28.0<br>46.7<br>28.1<br>~~19.1~~<br>75.4<br>77.0<br>23.7<br>21.5<br>~~12.0~~<br>26.3<br>39.8<br>16.4<br>19.9<br>13.1<br>61.6<br>62.8<br>41.5<br>31.6<br>16.1<br>74.5<br>79.8<br>40.1<br>36.0<br>~~26.3~~<br>60.7<br>72.3<br>54.6<br>30.0<br>16.5<br>57.8<br>73.5<br>**59.4**<br>**39.1**<br>13.1<br>83.4<br>79.8|62.4<br>61.8<br>67.0<br>67.1<br>57.0<br>53.9<br>57.7<br>64.4<br>58.6|44.4<br>39.7<br>**50.3**<br>26.1<br>21.1<br>22.6<br>34.9<br>22.6<br>15.0|7.4<br>26.2<br>5.2<br>16.9<br>21.3<br>32.4<br>18.4<br>28.9<br>**27.9**<br>8.0<br>5.9<br>1.3<br>3.7<br>6.7<br>16.2<br>8.9<br>2.9<br>5.3|28.5<br>22.8<br>~~46.2~~<br>~~30.8~~<br>38.1<br>40.5<br>~~34.4~~<br>37.9<br>37.4|
|**EURUS-7B-SFT**|55.5<br>59.1<br>20.0|52.1<br>32.6<br>20.0<br>82.2<br>84.1|64.6|44.0|15.4<br>28.4|46.5|
|**+ DPO**|50.6<br>52.1<br>8.3|51.0<br>28.3<br>**20.9**<br>78.7<br>83.8|65.0|42.5|20.6<br>32.4|44.5|
|**+ KTO**|56.1<br>58.6<br>18.9|55.0<br>33.2<br>20.6<br>84.4<br>85.0|**67.6**|43.1|19.1<br>**43.6**|**48.8**|
|**+ NCA**|55.5<br>60.2<br>14.4|54.9<br>34.2<br>**20.9**<br>**84.6**<br>**85.4**|64.3|42.7|21.3<br>38.7|48.1|
||_∼_40B||||||
|**Mixtral-8x7B-Instruct**<br>**DeepSeek-Coder-33B-Ins**|50.6<br>50.1<br>5.6<br>**82.3**<br>**73.9**<br>**27.8**|**49.6**<br>**25.9**<br>20.4<br>66.4<br>68.8<br>29.5<br>20.2<br>**21.9**<br>**75.2**<br>**85.0**|**73.5**<br>61.5|**48.8**<br>26.1|12.5<br>**37.3**<br>**35.3**<br>21.8|42.5<br>**46.7**|
||_∼_70B||||||
|**CodeLLaMA-70B-Instruct**<br>**DeepSeek-LM-67B-Chat**<br>**QWen1.5-72B-Chat**<br>**OpenCI-CL-70B**<br>**OpenMath-CL-70B**|56.7<br>58.6<br>14.4<br>70.7<br>65.7<br>20.0<br>71.3<br>56.9<br>15.6<br>77.4<br>71.7<br>20.0<br>39.0<br>52.6<br>15.0|34.9<br>12.0<br>8.4<br>63.5<br>70.1<br>65.0<br>41.0<br>17.9<br>74.0<br>84.0<br>**65.4**<br>43.4<br>18.5<br>79.5<br>79.1<br>46.1<br>29.2<br>18.8<br>76.1<br>79.4<br>62.2<br>**45.9**<br>15.9<br>86.6<br>82.8|74.5<br>78.9<br>78.0<br>66.7<br>59.9|24.0<br>52.7<br>**53.4**<br>26.8<br>15.7|3.7<br>14.2<br>30.9<br>41.8<br>27.2<br>38.2<br>30.9<br>12.0<br>14.0<br>0.4|36.3<br>53.5<br>52.2<br>46.3<br>40.8|
|**EURUS-70B-SFT**|75.6<br>**74.2**<br>**33.3**|58.1<br>40.6<br>28.0<br>86.3<br>88.5|79.9|49.2|31.6<br>40.4|57.1|
|**+ KTO**|76.8<br>68.2<br>26.1|62.2<br>41.3<br>30.6<br>**90.4**<br>89.0|**80.8**|46.4|**39.0**<br>**49.8**|58.4|
|**+ NCA**|**79.3**<br>71.9<br>**33.3**|62.8<br>41.7<br>**32.6**<br>89.5<br>**90.3**|80.0|49.2|38.2<br>39.6|**59.0**|
||ProprietaryModels||||||
|**GPT-3.5 Turbo**<br>**GPT-4**|76.8<br>82.5<br>23.3<br>**85.4**<br>**83.5**<br>**41.8**|61.2<br>37.8<br>35.6<br>83.0<br>90.6<br>**85.6**<br>**69.7**<br>**52.4**<br>**94.8**<br>**92.6**|70.1<br>**86.7**|56.6<br>**79.7**|29.4<br>36.9<br>**59.6**<br>**65.8**|57.0<br>**74.8**|



6 

Preprint 

## **4.1 Results** 

Results are shown in Table 3. We summarize the takeaways as follows: 

**EURUS, both the 7B and 70B variants, achieve the best overall performance among open-source models of similar sizes. EURUS even outperform specialized models in corresponding domains in many cases.** Notably, EURUS-7B outperforms baselines that are 5 _×_ larger and **EURUS-70B achieves better performance than GPT-3.5 Turbo** . EURUS’s instruction-following performance is among the best general-purpose models, substantially better than specialized ones. 

**Preference learning with ULTRAINTERACT can further improve the performance, especially in math and the multi-turn ability.** KTO and NCA consistently improve the models’ performance in all five math benchmarks and mult-turn evaluations, while their effects vary in others. Since SFT models only use the single-turn data from ULTRAINTERACT while preference learning uses the multi-turn ones, the improvements in interaction ability should also be attributed to ULTRAINTERACT rather than the algorithms alone. Surprisingly, we observe that **DPO hurts model performance on most benchmarks** . DPO training of our 70B model fails since the rewards go down to _−_ ∞. We analyze this phenomenon in §6.1. 

## **5 Evaluation of EURUS-RM-7B** 

**Evaluation Setup.** We evaluate EURUS-RM-7B on three RM benchmarks, RewardBench (Lambert et al., 2024), AutoJ (Li et al., 2023a), and MT-Bench (Zheng et al., 2023). Aiming for a more realistic OOD evalation, we exclude the “prior sets” split from RewardBench, since many baselines train on the datasets that this split contains. We compare with PairRM (Jiang et al., 2023b), Starling-RM-7B/34B (Zhu et al., 2023), UltraRM-13B (Cui et al., 2023), GPT-3.5 Turbo, and GPT-4. To further explore EURUS-RM-7B’s potential in improving models’ performance through reranking, we use it to rerank Mistral-7B-Instruct-v0.2’s responses on HumanEval, MBPP, GSM8K, and MATH. We report the results of random sampling, self-consistency, and Starling-RM-34B as baselines. 

## **5.1 Results** 

Table 4 summarizes reward modeling performance, and Figure 4 plots some reranking results with others in Appendix D.1. 

**EURUS-RM-7B stands out as the best 7B RM overall, and achieves similar or better performance than much larger baselines. Particularly, it outperforms GPT-4 in certain tasks.** EURUS-RM-7B achieves a better correlation with human experts than all existing models on AutoJ and MT-Bench, and it achieves comparable performance to the 5 _×_ larger Starling-RM-34B on RewardBench. On RewardBench, EURUS-RM-7B outperforms all baselines on the “Chat-Hard” split while achieving very competitive performance on the “Reasoning” split. Across the AutoJ splits, EURUS-RM-7B outperforms nearly all existing models, with the only exception being GPT-4’s results on Coding. 

**Our training objective is beneficial in improving RM performance on hard problems and reasoning.** Table 4 shows that optimizing _L_ DR improves RM’s reasoning ability, but BT modeling is still beneficial in equipping RM with abilities in general chatting as suggested in the “Chat-Hard” column, though its effect on reasoning may vary. 

**ULTRAINTERACT is compatible with other datasets like UltraFeedback and UltraSafety, and mixing these datasets can balance different RM abilities.** Improving RM’s capabilities in reasoning with ULTRAINTERACT does not sacrifice others, which indicates that ULTRAINTERACT can be a great ingredient for the training data mixture of reward models. 

**EURUS-RM-7B improves LLMs’ reasoning performance by a large margin through reranking.** EURUS-RM-7B consistently improves pass@1 accuracy across all tasks and performs better than 5 _×_ larger baseline Starling-RM-34B. Also, EURUS-RM-7B’s reranking performance scales well with #responses per instruction, except a slight decrease in HumanEval 

7 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees_images/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees.pdf-0008-01.png)


**----- Start of picture text -----**<br>
HumanEval MBPP GSM8K MATH<br>44 17.5<br>48 42 55 15.0<br>46 40 12.5<br>44 38 50 10.0<br>42 36 7.5<br>1 2 4 8 16 1 2 4 8 16 1 2 4 8 16 1 2 4 8 16<br>#Responses Per Instruction #Responses Per Instruction #Responses Per Instruction #Responses Per Instruction<br>Self-Consistency Starling-RM-34B Eurus-RM-7B<br>Pass@1 Accuracy Pass@1 Accuracy Pass@1 Accuracy Pass@1 Accuracy<br>**----- End of picture text -----**<br>


Figure 4: Results on reranking Mistral-7B-Instruct-v0.2’s responses. Full results in Table 9. 

Table 4: Results on reward modeling benchmarks. UF: UltraFeedback; US: UltraSafety. The best performance in each benchmark is in bold and the second best one is underlined. Most baseline results are from Jiang et al. (2023b) and Lambert et al. (2024). 

|Model|Reward Bench<br>Chat<br>Chat-Hard<br>Safety<br>Reasoning<br>A|AutoJ<br>MT-Bench<br>vg.<br>Code<br>Math<br>Others<br>Overall|
|---|---|---|
|PairRM<br>Starling-RM-7<br>Starling-RM-3<br>UltraRM-13B<br>GPT-3.5 Turb<br>GPT -4|90.2<br>53.0<br>31.5<br>60.0<br>58<br>B<br>98.0<br>43.4<br>88.6<br>74.6<br>76<br>4B<br>96.9<br>59.0<br>89.9<br>90.3<br>**84**<br>96.1<br>55.3<br>45.8<br>82.0<br>69<br>o<br>-<br>-<br>-<br>-<br><br>-<br>-<br>-<br>-<br>|.7<br>58.3<br>52.8<br>58.9<br>59.1<br>59.0<br>.2<br>59.2<br>47.2<br>61.4<br>60.8<br>56.8<br>**.0**<br>65.8<br>54.2<br>62.3<br>62.6<br>60.4<br>.8<br>55.0<br>43.1<br>59.6<br>59.9<br>56.0<br>-<br>36.6<br>40.3<br>41.2<br>42.7<br>57.1<br>-<br>69.2<br>51.4<br>61.4<br>61.9<br>63.9|
|EURUS-RM-7<br>w/o_L_DR<br>w/o_L_BT<br>w/o US<br>w/o UF + US|B<br>96.5<br>65.3<br>80.7<br>87.0<br>82<br>96.4<br>59.9<br>79.5<br>77.5<br>78<br>96.8<br>58.5<br>83.8<br>84.2<br>80<br>96.5<br>66.2<br>67.7<br>81.7<br>73<br>95.1<br>61.1<br>63.7<br>73.4<br>78|.4<br>67.5<br>62.5<br>63.6<br>64.5<br>**72.9**<br>.3<br>64.2<br>59.7<br>64.7<br>65.0<br>72.8<br>.8<br>67.5<br>66.7<br>64.8<br>65.6<br>72.6<br>.3<br>66.7<br>61.1<br>65.0<br>**65.7**<br>72.6<br>.0<br>55.8<br>58.3<br>59.0<br>58.7<br>67.2|



when increasing response number form 8 to 16. In contrast, Starling-RM-34B suffers from severe performance drop on HumanEval and it consistently hurts model accuracy on MATH. 

## **6 Analysis** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees_images/iclr_2025_Advancing_LLM_Reasoning_Generalists_with_Preference_Trees.pdf-0008-07.png)


**----- Start of picture text -----**<br>
DPO KTO NCA<br>3<br>4<br>2 2<br>2 1<br>0 0.4 0.16<br>-1.26 0 0<br>2 Margins 2 Margins 1 Margins<br>Rewards/Chosen Rewards/Chosen 2 Rewards/Chosen<br>4 Rewards/Rejected 4 Rewards/Rejected Rewards/Rejected<br>3<br>0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100<br>Steps (%) Steps (%) Steps (%)<br>Value Value Value<br>**----- End of picture text -----**<br>


Figure 5: Reward patterns of EURUS-7B preference learning with DPO, KTO, and NCA. For all algorithms, the rewards of rejected data keep decreasing and the margins between chosen and rejected data keep increasing. However, the rewards of chosen data decrease below zero in DPO while keeping increasing and staying positive in KTO and NCA. The absolute values of the reward in the last step (in red) of the three algorithms positively correlate with their performance in Table 3. 

## **6.1 Explicit Reward as A Proxy? Hypothesis for Preference Learning in Reasoning** 

We investigate the reason why DPO behaves differently than KTO and NCA. We start by empirically inspecting the rewards throughout the preference learning process, as shown in Figure 5. Rewards for chosen rejected data both keep decreasing through DPO, though the rewards for chosen data is still higher hence the loss decreases. In KTO and NCA, the rewards of chosen data keep increasing with those of rejected data decreasing. 

Therefore, we hypothesize it is the distinction in the trend of rewards that leads to the performance gap between DPO and the other two algorithms. This distinction can be attributed to that DPO, derived from the Bradley-Terry model, only optimizes the relative differences between chosen and rejected data overlooking the absolute values of the rewards. This is a non-issue in alignment with general human values where preference is “relative” and there 

8 

Preprint 

can be many valid answers to the same input. However, in reasoning tasks, the space of correct answers is much smaller than that of incorrect ones. Further, we notice that the rewards of chosen data in the last training step follow the ranking order of KTO _>_ NCA _>_ DPO, positively correlate with their performance trends. Therefore, we believe that increasing the rewards of the chosen data is especially beneficial in preference learning for reasoning tasks. 

## **6.2 Ablation Study** 

We study the impact of ULTRAINTERACT and other open-source alignment data on EURUS-7B-SFT’s performance. We consider three settings: (1) **With original ground-truth answers,** which replaces the generated actions with ground-truth 

Table 5: Ablation study of SFT data. 

||**Model**|**Coding**|**Math**|**BBH**|**IFEval**|**Avg.**|
|---|---|---|---|---|---|---|
||**EURUS-7B-SFT**<br>**Ground-truth**<br>**Open-source Only**<br>**ULTRAINTERACT Only**|44.9<br>33.9<br>31.2<br>37.3|58.5<br>46.1<br>33.5<br>56.2|64.6<br>64.4<br>65.3<br>67.0|44.0<br>42.9<br>43.6<br>17.4|53.6<br>44.0<br>37.0<br>47.7|



rationales and answers from the original datasets. If no rationales are available, we use those from ULTRAINTERACT. (2) **Open-source data only.** (3) **ULTRAINTERACT only.** We evaluate with the same setting as §4 and report the averaged scores. See full results in Appendix E. 

In Table 5, EURUS outperforms the “Grouth-truth” model on all tasks, confirming the advantage of ULTRAINTERACT’s designs of divide-and-conquer and code-as-action patterns, in line with conclusions of concurrent work (Chen et al., 2024b; Wang et al., 2024). Training only on open-source data without ULTRAINTERACT greatly hurts the reasoning performance, confirming the effectiveness of ULTRAINTERACT. Meanwhile, training only on ULTRAINTERACT suffers a performance drop except for BBH, especially in instruction following. We attribute the performance drop to a worse instruction-following ability. This suggests the necessity of mixing ULTRAINTERACT with other alignment data for better all-around supervised fine-tuning. 

## **7 Related Work** 

**Open LLMs in Reasoning.** Open-source LLMs have shown remarkable progress in building _specialists_ that excel in mathematics reasoning (Luo et al., 2023a; Yue et al., 2023; Toshniwal et al., 2024) or coding abilities (Roziere et al., 2023; Wei et al., 2023; Guo et al., 2024a; Zheng et al., 2024). On the contrary, mastering general reasoning capabilities still challenges open models, while the most advanced ones (DeepSeek-AI, 2024; Bai et al., 2023; Touvron et al., 2023; Jiang et al., 2024) are well behind proprietary models. More, these cutting-edge open general-purpose models maintain their alignment recipes confidential, which further hinders the replication and development of open-source reasoning models. 

**Preference Learning for Reasoning.** Aligning language models from human or AI preferences has emerged as a prevalent approach in the open-source community (Tunstall et al., 2023; Bai et al., 2023) with the proposal of DPO (Rafailov et al., 2023) and high-quality preference datasets (Cui et al., 2023; Zhu et al., 2023). Different from open-domain chatbots, preference learning is largely underexplored in complex reasoning. Recent research showed performance degradation when applying DPO on reasoning tasks, but some newly proposed algorithms demonstrated a positive effect (Ethayarajh et al., 2024; Chen et al., 2024a; Mitra et al., 2024; Shao et al., 2024). However, a deep understanding of preference learning, specifically its efficacy on complex reasoning, is not yet established. 

## **8 Conclusion** 

We strive to narrow the huge gap between open-source models and proprietary models from the perspective of alignment. Our work pushes the boundaries of open-source reasoning generalists by (1) releasing a high-quality multi-turn reasoning dataset ULTRAINTERACT with preference trees, (2) introducing EURUS-series LLMs which achieve new SOTA on challenging reasoning benchmarks and (3) providing insights on preference learning for reasoning through analysis, leading to new reward modeling objectives as well as a powerful reward model for reasoning. 

9 

Preprint 

## **References** 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proc. of NAACL-HLT, 2019. 

- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. ArXiv preprint, abs/2108.07732, 2021. 

- Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. ArXiv preprint, abs/2309.16609, 2023. 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, T. J. Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac HatfieldDodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, Benjamin Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. ArXiv preprint, abs/2204.05862, 2022. 

- Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39, 1952. 

- Huayu Chen, Guande He, Hang Su, and Jun Zhu. Noise contrastive alignment of language models with explicit rewards. ArXiv preprint, abs/2402.05369, 2024a. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. 

- Wenhu Chen, Ming Yin, Max W.F. Ku, Yixin Wan, Xueguang Ma, Jianyu Xu, Tony Xia, Xinyi Wang, and Pan Lu. Theoremqa: A theorem-driven question answering dataset. ArXiv preprint, abs/2305.12524, 2023. 

- Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang, Jiangning Liu, Dahua Lin, Kai Chen, and Feng Zhao. Agent-flan: Designing data and methods of effective agent tuning for large language models. volume abs/2403.12881, 2024b. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. volume abs/2110.14168, 2021. 

- Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with highquality feedback. ArXiv preprint, abs/2310.01377, 2023. 

10 

Preprint 

DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism. ArXiv preprint, abs/2401.02954, 2024. 

- Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Conference on Empirical Methods in Natural Language Processing, 2023. 

- Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. ArXiv preprint, abs/2402.01306, 2024. 

- Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. In Proceedings of the International Conference on Machine Learning, 2023. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9, 2021. 

- Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming - the rise of code intelligence. ArXiv preprint, abs/2401.14196, 2024a. 

- Yiju Guo, Ganqu Cui, Lifan Yuan, Ning Ding, Jiexin Wang, Huimin Chen, Bowen Sun, Ruobing Xie, Jie Zhou, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Controllable preference optimization: Toward controllable multi-objective alignment. ArXiv preprint, abs/2402.19085, 2024b. 

- Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021a. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. ArXiv preprint, abs/2310.06825, 2023a. 

- Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, L’elio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile´ Gervet, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mixtral of experts. 2024. 

- Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. In Annual Meeting of the Association for Computational Linguistics, 2023b. 

- Nathan Lambert, Valentina Pyatkin, Jacob Daniel Morrison, Lester James Validad Miranda, Bill Yuchen Lin, Khyathi Raghavi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hanna Hajishirzi. Rewardbench: Evaluating reward models for language modeling. 2024. 

- Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. Generative judge for evaluating alignment. ArXiv preprint, abs/2310.05470, 2023a. 

11 

Preprint 

Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, and Wei Bi. Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers. ArXiv preprint, abs/2402.19255, 2024. 

- Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. volume abs/2312.14852, 2023b. 

- Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Remi Leblond,´ Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. volume abs/2203.07814, 2022. 

- Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and ”Teknium”. Openorca: An open dataset of gpt augmented flan reasoning traces. https://https://huggingface.co/Open-Orca/OpenOrca, 2023. 

- Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. 2023. 

- Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In Proceedings of ICLR, 2023. 

- Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. ArXiv preprint, abs/2308.09583, 2023a. 

- Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct, 2023b. 

Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing English math word problem solvers. In Proc. of ACL, 2020. 

- Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. NumGLUE: A suite of fundamental yet challenging mathematical reasoning tasks. In Proc. of ACL, 2022. 

- Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agrawal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching small language models how to reason. ArXiv preprint, abs/2311.11045, 2023. 

- Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math. ArXiv preprint, abs/2402.14830, 2024. 

OpenAI. Gpt-4 technical report, 2023. 

Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. In Proc. of ACL, 2015. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021. 

12 

Preprint 

Cheng Qian, Chi Han, Yi Ren Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. Creator: Tool creation for disentangling abstract and concrete reasoning of large language models. In Conference on Empirical Methods in Natural Language Processing, 2023. 

- Yujia Qin, Shi Liang, Yining Ye, Kunlun Zhu, Lan Yan, Ya-Ting Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Marc H. Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis. ArXiv preprint, abs/2307.16789, 2023. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. ArXiv preprint, abs/2305.18290, 2023. 

- Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jer´ emy´ Rapin, et al. Code llama: Open foundation models for code. ArXiv preprint, abs/2308.12950, 2023. 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Y Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. ArXiv preprint, abs/2402.03300, 2024. 

- Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won¨ Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. ArXiv preprint, abs/2210.09261, 2022. 

- Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. arXiv preprint arXiv: Arxiv-2402.10176, 2024. 

- Hugo Touvron, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,´ Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, A. V. Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288, 2023. 

- Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clementine Fourrier, Nathan Habib, et al.´ Zephyr: Direct distillation of lm alignment. ArXiv preprint, abs/2310.16944, 2023. 

- Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. Openchat: Advancing open-source language models with mixed-quality data. ArXiv preprint, abs/2309.11235, 2023a. 

- Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. Mint: Evaluating llms in multi-turn interaction with tools and language feedback. ArXiv preprint, abs/2309.10691, 2023b. 

- Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. ArXiv preprint, abs/2402.01030, 2024. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. ArXiv preprint, abs/2201.11903, 2022. 

13 

Preprint 

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Source code is all you need, 2023. 

- Martin Weyssow, Aton Kamanda, and Houari Sahraoui. Codeultrafeedback: An llm-as-ajudge dataset for aligning large language models to coding preferences. ArXiv preprint, abs/2403.09032, 2024. 

- Wenda Xu, Guanglei Zhu, Xuandong Zhao, Liangming Pan, Lei Li, and William Yang Wang. Perils of self-feedback: Self-bias amplifies in large language models. ArXiv preprint, abs/2402.11436, 2024. 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proc. of EMNLP, 2018. 

Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. Reclor: A reading comprehension dataset requiring logical reasoning. In Proc. of ICLR, 2020. 

- Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi Ren Fung, Hao Peng, and Heng Ji. Craft: Customizing llms by creating and retrieving from specialized toolsets. ArXiv preprint, abs/2309.17428, 2023. 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. ArXiv preprint, abs/2309.05653, 2023. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Haotong Zhang, Joseph Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. ArXiv preprint, abs/2306.05685, 2023. 

- Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. Opencodeinterpreter: Integrating code generation with execution and refinement. ArXiv preprint, abs/2402.14658, 2024. 

- Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. ArXiv preprint, abs/2311.07911, 2023. 

- Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. Starling-7b: Improving llm helpfulness & harmlessness with rlaif, 2023. 

14 

Preprint 

Table 6: ULTRAINTERACT covers a diverse set of datasets spanning three tasks. 

|**Task**|**Datasets**|
|---|---|
|**Math**|GSM8K (Cobbe et al.,2021), MATH (Hendrycks et al.,2021b), MathQA (Amini et al.,2019),<br>NumGlue (Mishra et al.,2022), TabMWP (Lu et al.,2023)|
|**Coding**|CodeContest (Li et al.,2022), TACO (Li et al.,2023b), WikiTableQuestions (Pasupat & Liang,<br>2015), Magicoder-Evol-Instruct (Luo et al.,2023b;Wei et al.,2023)|
|**Logic**|ReClor (Yu et al.,2020), HotpotQA (Yang et al.,2018), StrategyQA (Geva et al.,2021)|



## **A Additional Details in ULTRAINTERACT Construction** 

## **A.1 Dataset Details** 

**Math.** We adopt **GSM8K** (Cobbe et al., 2021), **MATH** (Hendrycks et al., 2021b), **MathQA** (Amini et al., 2019), and **NumGLUE** (Mishra et al., 2022)for mathematic reasoning, and include **TabMWP** (Lu et al., 2023) for tabular processing. We retain all the instructions for all datasets except MathQA, NumGLUE, and TabMWP. MathQA divides problems into different categories according to the topics and annotates the formula that indicates the pattern needed to solve each problem. We apply stratified sampling to sample at most five problems for each pattern and prioritize the problems that come from the longtail category. Numglue contains eight different reasoning tasks and we discard Task 5 (Reading Comprehension + Explicit Numerical Reasoning), Task 6 (Reading Comprehension + Implicit Numerical Reasoning), and Task 7 (Quantitative NLI) due to the simplicity Mishra et al. (2022). For TabMWP, we only keep the questions with difficulty levels 4 and 5 since the rest are too easy for current state-of-the-art models. 

**Code.** We focus on programming with Python for the simplicity of integration of the interpreter. We use **CodeContest** (Li et al., 2022) and **TACO** (Li et al., 2023b), two competitionlevel coding datasets collected from various online platforms. We filter out the overlapped questions. Note that part of the questions in TACO only contain ground-truth solutions and do not contain test cases for evaluation, hence we apply GPT-4 to generate 12 test case inputs (4 basic inputs, 4 edge cases, and 4 large numbers) for each question and then execute the ground-truth solution snippets to produce outputs. Given that the two datasets mainly focus on competition problems that may deviate from real-world daily uses, we exclusively adopt **Magicoder-Evol-Instruct** (Luo et al., 2023b; Wei et al., 2023), the only dataset in our selection that does not contain test cases or ground-truth solutions. We employ GPT-4 Turbo to judge the correctness of generated code during interaction, and therefore we do not use this dataset for preference learning since we cannot rigorously construct pairs of correct and incorrect actions limited by the evaluation reliability. We also include **WikiTableQuestions** (Pasupat & Liang, 2015) for table processing with code. 

**Logical Reasoning.** we use the multi-hop reasoning datasets **HotpotQA** (Yang et al., 2018) and **StrategyQA** (Geva et al., 2021), and the logical reasoning dataset **ReClor** (Yu et al., 2020). We follow the setting of Wang et al. (2023b) and convert HotpotQA to a generation task, removing the contexts and requiring LLMs to search relevant information using Wikipedia API. 

## **A.2 Details on Preference Tree Construction** 

**Models Adopted for Incorrect Action Sampling.** We randomly sample one model from Mistral-7B-Instruct-v0.2, DeepSeek-Coder-33B-Instruct, Mixtral-8x7B-Instruct, and DeepSeek-LLM-67B-Chat to generate one incorrect action to pair with each correct one. 

## **Correct Action Generation Based on Ground Truth Annotations.** 

We adopt GPT-3.5 Turbo as the generator to generate correct actions based on ground truth considering the instruction-following ability. We provide different access to the ground truth information for different tasks, specifically: (1) For coding, where test cases are black boxes to reference solutions, we provide full access to the solution codes. The actor model will add step marks and corresponding explanations to the ground-truth code to make it 

15 

Preprint 

Table 7: Stats breakdown 

|**Task**<br>**Dataset**<br>**w/ Tool?**<br>**# Prompts**<br>**# Pairs**<br>**# Correct Answers.**<br>**Avg. Length**|**Task**<br>**Dataset**<br>**w/ Tool?**<br>**# Prompts**<br>**# Pairs**<br>**# Correct Answers.**<br>**Avg. Length**|**Human Annotation**|
|---|---|---|
|||**Has Answer?**<br>**Has Rationale?**|
|Math|GSM8K<br>�<br>4,522<br>10,277<br>17,392<br>1,746.7<br>�<br>7,257<br>10,879<br>15,752<br>823.3<br>MATH<br>�<br>7,474<br>22,905<br>34,667<br>1,189.0<br>�<br>7,471<br>25,765<br>36,005<br>1,735.0<br>MathQA<br>�<br>7,552<br>15,079<br>20,328<br>2,338.5<br>�<br>7,159<br>17,743<br>22,500<br>1,916.3<br>NumGLUE<br>�<br>3,020<br>3,601<br>5,717<br>1,474.6<br>�<br>2,835<br>2,987<br>4,273<br>1,056.1<br>TabMWP<br>�<br>3,117<br>4,135<br>6,083<br>842.6|�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�<br>�|
|Coding|CodeContest<br>-<br>8,167<br>44,319<br>44,666<br>2,061.7<br>TACO<br>-<br>9,016<br>50,877<br>58,191<br>2,143.5<br>WikiTableQuestions<br>-<br>1,401<br>1,544<br>1,738<br>1,794.8<br>Magicoder-Evol-Instruct<br>-<br>10,374<br>0<br>10,238<br>687.1|�<br>�<br>�<br>�<br>�<br>�<br>�<br>�|
|Logic|Reclor<br>�<br>4,467<br>7,958<br>7,231<br>1,266.7<br>HotpotQA<br>�<br>1,182<br>1,009<br>1,230<br>1,333.2<br>StrategyQA<br>�<br>904<br>741<br>968<br>1,256.2|�<br>�<br>�<br>�<br>�<br>�|



easier to understand, or further refine the code for optimization. (2) For tool-free math problems, to avoid the actor model directly copying the answers to pass the correctness checking, we mask the answer numbers in the rationale before providing it to LLMs. This approach can better ensure response quality since it encourages LLMs to generate responses with complete reasoning chains with each step clearly marked. (3) For program-enhanced math reasoning, we first translate the textual rationale into code. Then, we either directly provide it to the actor model to generate plans, or ask the actor model to convert the code into modularization programming and then make plans to create tools to solve problems. 

## **A.3 Data Decomtamination** 

We conduct careful decontamination. Firstly, for LeetCode, we apply the Exact Substring Matching Algorithm[3] to compare with each instruction in the ULTRAINTERACT and find no overlaps. For others, we perform 8-gram exact matching to compare ULTRAINTERACT instructions with test sets of the same task. We remove those instructions that overlap 8 grams with any test sample. 

## **A.4 Detailed Statistics** 

In total, ULTRAINTERACT has 86K instructions and 220K action pairs. The Total # Pairs does not equal Total # Turns in ULTRAINTERACT, since we fail to generate sufficient correct actions for every incorrect action in multi-turn trajectories mainly due to a lack of sufficient ground truth annotations. The total # pairs may not equal # correct answers, either, because it is also difficult and unnecessary to sample incorrect actions for the correct ones for some simple instructions. We present the specific information for each dataset. In particular, we list information on human annotation in each dataset, which plays an important role in correct action generation (§2.3 and Appendix A.2). All three steps of correct action sampling methods mentioned in §2.3 can be applied to datasets that have rationales, while for datasets only containing answers, only the first two steps are applicable. We do not apply any of the three-step methods to generate correct answers for Magicoder, the only dataset without any human annotation, to construct preference pairs. 

## **B Additional Details on Training EURUS Models** 

**Supervised Fine-Tuning.** We finetune base models for 1 epoch with a 2e-5 learning rate and 0.1 warmup ratio using a cosine scheduler. For EURUS-7B, we mix 32K UltraChat, 30K ShareGPT, and 50K OpenOrca. For For EURUS-70B, we mix 63K UltraChat, 30K ShareGPT, and 70K OpenOrca. 

> 3https://github.com/bigcode-project/bigcode-dataset/tree/main/decontamination 

16 

Preprint 

**Preference Learning.** For hyperparameters, all _β_ is set to 0.1, and _λ_ +/ _λ−_ in KTO is set to 1.33 as recommended. We finetune models for 1 epoch with a 5e-7 learning rate and 0.1 warmup ratio using a cosine scheduler. 

**Reward Modeling.** We train RM for 1 epoch with lr=1e-5 learning rate. We also use a cosine scheduler with a warmup ratio of 0.1. 

Regarding pair augmentation, we scale up the pairs by matching every correct action for each instruction with one incorrect action of other turns. This leads to NxN pairs of singleturn actions for a trajectory of depth N. We remove the action pairs consisting of nodes at the same turn, as they are already part of the multi-turn trajectory pairs we included. Next, to avoid overfitting on the training set, we only select instructions with NxN _≤_ 10, and for these instructions, we randomly sample at most 9 pairs with each action occurring no more than 3 times. This leads to an augmentation of 240k single-turn action pairs. 

## **C Additional Evaluation Results of EURUS** 

**Detailed Setup in §4.** For math, we test both textual reasoning and program-enhanced settings and report the best performance of the two. All evaluations are conducted in 0-shot CoT with two exceptions: BBH uses 3 shots and IFEval does not use CoT. For MINT, we select MATH, TheoremQA, and MMLU-math from “reasoning” as a new “math” split. We also evaluate 5-shot MMLU (Hendrycks et al., 2021a) for STEM knowledge and MT-Bench (Zheng et al., 2023) for conversation abilities to study whether EURUS needs to trade off other capabilities for reasoning. 

**Results.** Results are shown in Table 8. 

On MMLU, EURUS outperforms baselines dedicated to coding and math, and achieves higher results than Mistral-Instruct-v0.2 and CodeLLaMA-70B-Instruct, the official aligned versions of our base model built by their authors. Compared to general-purpose baseline models, EURUS-7B achieves comparable performance with the top-performance OpenChat and Starling-LM, though EURUS-70B does not achieve the same level of performance as other generalpurpose models, which is expected due to the gap in the base models since CodeLLaMA-70B has not been intentionally optimized for knowledge. 

Table 8: MMLU and MT-Bench. 

|**Mistral-7B-Instruct-v0.2**<br>**Zephyr-7B-**_β_<br>**OpenChat-3.5-1210**<br>**Starling-LM-7B-**_α_<br>**Magicoder-S-DS-6.7B**<br>**OpenCI-DS-6.7B**<br>**MAmmoTH-7B-Mistral**<br>**WizardMath-7B-v1.1**<br>**OpenMath-Mistral-7B**|58.9<br>7.60<br>59.7<br>7.34<br>63.4<br>7.81<br>**64.0**<br>**8.09**<br>37.1<br>-<br>37.2<br>-<br>56.2<br>-<br>60.3<br>-<br>58.3<br>-|58.9<br>7.60<br>59.7<br>7.34<br>63.4<br>7.81<br>**64.0**<br>**8.09**<br>37.1<br>-<br>37.2<br>-<br>56.2<br>-<br>60.3<br>-<br>58.3<br>-|
|---|---|---|
|**EURUS-7B-SFT**|61.8<br>7.15||
|**+ DPO**|62.4<br>7.38||
|**+ KTO**|62.2<br>7.38||
|**+ NCA**|62.2<br>7.38||
|_∼_40B<br>**Mixtral-8x7B-Instruct**<br>**70.3**<br>**8.30**<br>**DeepSeek-Coder-33B-Ins**<br>40.2<br>-<br>_∼_70B<br>**CodeLLaMA-70B-Instruct**<br>55.1<br>-<br>**DeepSeek-LM-67B-Chat**<br>72.3<br>-<br>**QWen1.5-72B-Chat**<br>**72.9**<br>**8.61**<br>**OpenCI-CL-70B**<br>52.4<br>-<br>**OpenMath-CL-70B**<br>60.2<br>-|||
|**CodeLLaMA-70B-Instruct**<br>**DeepSeek-LM-67B-Chat**<br>**QWen1.5-72B-Chat**<br>**OpenCI-CL-70B**<br>**OpenMath-CL-70B**|55.1<br>-<br>72.3<br>-<br>**72.9**<br>**8.61**<br>52.4<br>-<br>60.2<br>-||
|**EURUS-70B-SFT**|59.1<br>7.69||
|**+ KTO**|59.5<br>7.93||
|**+ NCA**|59.4<br>7.54||
|ProprietaryModels<br>**GPT-3.5 Turbo**<br>70.0<br>7.94<br>**GPT-4**<br>**86.4**<br>**8.96**|||
|**GPT-3.5 Turbo**<br>**GPT-4**|70.0<br>7.94<br>**86.4**<br>**8.96**||



On MT-Bench, we report baseline numbers from the official leaderboard[4] . EURUS matches the performance of mainstream open-source general-purpose models, and EURUS-70B-KTO further achieves the score of GPT-3.5 Turbo. 

## **D Detailed Results on Reward Modeling** 

## **D.1 Additional Results on Reranking** 

We present the full results on reranking in Table 9, where the conclusions are consistent with those drawn from §D: (1) Our reward models always achieve the highest accuracy on all test sets across different N, except when N=2 on HumanEval. (2) Both _L_ BT and _L_ DR consistently help improve reranking performance on three test sets except for HumanEval, where removing either of the objectives can prevent the accuracy from dropping when increasing N from 8 to 16. (3) Modeling safety hurts reranking performance in reasoning. 

> 4https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard 

17 

Preprint 

Table 9: Detailed results of reranking Mistral-Instruct-v0.2’s responses on coding and math. 

|Datasets|HumanEval|MBPP|GSM8K|MATH|
|---|---|---|---|---|
|N|2<br>4<br>8<br>16|2<br>4<br>8<br>16|2<br>4<br>8<br>16|2<br>4<br>8<br>16|
|Random<br>Top Logits<br>Self-Consistency<br>Starling-RM-34B|41.5<br>39.0<br>40.2<br>39.6<br>43.3<br>43.3<br>43.3<br>43.3<br>43.3<br>42.7<br>42.1<br>40.9<br>**47.6**<br>**47.0**<br>**49.4**<br>45.7|33.1<br>33.6<br>34.3<br>30.1<br>35.3<br>35.3<br>35.3<br>35.3<br>35.3<br>36.3<br>36.6<br>37.1<br>37.8<br>38.8<br>39.6<br>40.4|45.0<br>43.1<br>44.5<br>40.2<br>45.7<br>45.7<br>45.7<br>45.7<br>45.7<br>49.5<br>52.2<br>52.8<br>49.1<br>52.8<br>56.0<br>56.5|11.5<br>11.3<br>10.0<br>8.5<br>12.1<br>12.1<br>12.1<br>12.1<br>12.1<br>13.8<br>15.8<br>16.8<br>6.5<br>7.2<br>7.7<br>7.7|
|EURUS-RM-7B<br>w/o_LDR_<br>w/o_LBT_<br>w/o US<br>w/o UF + US|44.5<br>45.7<br>47.6<br>47.0<br>45.7<br>44.5<br>46.3<br>50.0<br>45.1<br>44.5<br>47.0<br>48.2<br>45.7<br>**47.0**<br>**49.4**<br>**50.6**<br>43.9<br>43.3<br>47.0<br>46.3|39.3<br>**42.6**<br>**43.4**<br>**43.9**<br>39.3<br>42.4<br>42.4<br>42.1<br>38.6<br>40.6<br>39.6<br>40.1<br>**39.3**<br>41.1<br>41.4<br>42.9<br>36.3<br>38.1<br>36.6<br>35.3|**49.8**<br>53.7<br>56.3<br>57.3<br>49.4<br>53.2<br>55.4<br>56.3<br>49.1<br>52.5<br>55.2<br>57.8<br>49.4<br>**53.8**<br>**57.4**<br>**58.7**<br>49.4<br>52.3<br>54.6<br>57.2|14.3<br>16.2<br>17.1<br>17.3<br>14.2<br>16.1<br>17.0<br>16.9<br>14.3<br>16.3<br>17.2<br>17.1<br>**14.5**<br>**16.6**<br>17.2<br>**17.5**<br>14.3<br>16.5<br>**17.4**<br>17.4|
|Pass@N|62.8<br>73.8<br>88.4<br>92.7|42.4<br>48.1<br>52.6<br>58.6|54.9<br>64.1<br>73.2<br>80.4|16.9<br>22.7<br>28.9<br>35.5|



Table 10: Ablation Study. 

|Model|Coding|Math|Reasoning|Ins-Following|Avg.|
|---|---|---|---|---|---|
||HumanEval<br>MBPP<br>LeetCode|GSM8K<br>MATH<br>TheoremQA<br>SVAMP<br>ASDiv|BBH|IFEval||
|**EURUS-7B-SFT**<br>**Ground-Truth**<br>**Open-Source Only**<br>**ULTRAINTERACT Only**|**55.5**<br>**59.1**<br>**20.0**<br>46.3<br>46.4<br>8.9<br>38.4<br>44.1<br>11.1<br>46.3<br>50.1<br>15.6|**73.7**<br>**32.6**<br>20.0<br>**82.2**<br>**84.1**<br>62.2<br>15.0<br>9.6<br>75.1<br>68.8<br>45.3<br>10.8<br>9.3<br>52.7<br>49.4<br>67.6<br>30.9<br>**20.1**<br>80.4<br>82.0|64.6<br>64.4<br>65.3<br>**67.0**|**44.0**<br>42.9<br>43.6<br>17.4|**53.6**<br>44.0<br>37.0<br>47.7|



When removing UltraSafety from the training data, the RM achieves higher accuracies than EURUS-RM-7B except on MBPP. 

## **E Detailed Ablation Results** 

We present the full results of §5 in Table 10, with detailed metrics on all coding and math datasets. 

18 

