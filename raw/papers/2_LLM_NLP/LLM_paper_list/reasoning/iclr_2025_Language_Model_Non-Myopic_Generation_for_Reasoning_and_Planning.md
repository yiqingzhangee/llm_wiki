Preprint 

# - NON MYOPIC GENERATION OF LANGUAGE MODELS FOR REASONING AND PLANNING 

**Chang Ma**[♠] **Haiteng Zhao**[♡] **Junlei Zhang**[♢][∆] **Junxian He[∗]**[♣] **Lingpeng Kong**[∗♠] 

♠The University of Hong Kong 

♡Peking University ♢Zhejiang University 

∆Westlake University ♣The Hong Kong University of Science and Technology 

_{_ cma, lpk _}_ @cs.hku.hk, junxianh@cse.ust.hk 

## ABSTRACT 

Large Language Models (LLMs) have demonstrated remarkable abilities in reasoning and planning by breaking down complex problems into sequential steps. Despite their success in various domains, such as mathematical problem-solving and coding, LLMs face challenges in ensuring reliable and optimal planning due to the inherent myopic nature of autoregressive decoding. This paper revisits LLM reasoning from an optimal control perspective, proposing a novel method, Predictive-Decoding, that leverages Model Predictive Control to enhance planning accuracy. By reweighting LLM distributions based on foresight trajectories, Predictive-Decoding aims to mitigate early errors and promote non-myopic planning. Our experiments show significant improvements across a wide range of tasks in math, coding, and agent-based scenarios. Furthermore, PredictiveDecoding demonstrates computational efficiency, outperforming search baselines while utilizing inference compute more effectively. This study provides insights into optimizing LLM planning capabilities. Code is available at this repo. 

## 1 INTRODUCTION 

Large Language Models (LLMs) are extensively pretrained on large corpus to predict the next tokens. Models like GPT-4 (Achiam et al., 2023) have demonstrated a capacity for step-by-step reasoning and planning, breaking down complex problems into sequential steps that progressively lead to solutions (Wei et al., 2022). This _sequential planning_ capability has led to significant advancements in mathematical problem-solving (Cobbe et al., 2021) and programming tasks (Chen et al., 2021). It has also enabled essential applications such as tool use (Qin et al., 2023) and the development of LLM agents (Yao et al., 2022). Consider a scenario where an agent is tasked with cooking a meal: given an instruction like “ _Put salt on steak_ ” and an initial observation such as “ _There is a shelf in front of you_ ,” the LLM assigns the highest probability to the most appropriate action. After executing this action, the LLM generates subsequent actions, each building upon the previous steps. This process creates a sequential chain of actions that advances toward task completion, guided by continuous interaction with environmental observations: 

_p_ ( “ _Go to the shelf._ ” ∣ “ _There is a shelf in front of you._ ”) 

_p_ ( “ _Pick up the saltbottle._ ” ∣ “ _Go to the shelf. A saltbottle is on the shelf._ ”) _,_ 

where “ _A saltbottle is on the shelf_ ” is the new observation after going to the shelf. This iterative process continues until the specified goal state is achieved. However, this autoregressive decoding process could easily lead to irreversible errors, as LLMs tend to follow the most natural local followups. For instance, given a different observation “ _A spice bottle is on the shelf_ ”, the LLM is likely to generate actions “ _Pick up the spice bottle_ ” followed by “ _Shake the bottle over the steak_ ”. Such a sequence of actions totally conflicts with the intended instruction “ _Put salt on the steak_ ”. This contrasts with reinforcement learning paradigms, which use reward-based training to teach models to anticipate long-term action consequences and learn optimal decision policies (Silver et al., 2014). The short-sightedness of LLMs raises critical questions about their planning capabilities: (i) Can an LLM proactively avoid erroneous steps without necessitating their occurrence? (ii) What is the degree of optimality achievable in LLM-based planning strategies? 

> ∗Equal advising. 

1 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Reasoning Challenge: Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download,<br>Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from  the beginning . How long does it<br>take to download the file?<br>Step 1: Before restart, Carla  Step 2: Then after the restart,  Step 3: After restart, she needs to  Error has already<br>downloaded for 200 *40%/2 =  Carla has 200 * 60% = 120GB  download  the remaining  for  been made. Too  LLM<br>40 minutes. to download.  120/2 = 60 minutes. late! Probability<br>LLM Prob: 0.99, Human: Correct LLM Prob: 0.89, Human: Wrong LLM Prob: 0.45, Human: Wrong<br>Myopic Reasoner Global Correctness<br>Step 1 Step 2:  Then after the restart, Carla has 200 * 60% = 120GB to download.  Step 3: She needs to download  remaining  for 60 minutes. the  LLM Prob: 0.45<br>Step 2:  Then after the restart, Carla has 200 * 60% = 120GB to download.  Step 3: She needs to download  remaining  for 60 minutes. the  LLM Prob: 0.45<br>Predictive Decoding Step 2:  Then after the restart, Carla has 200 GB to download.  Step 3: She needs to download  from start  for 100 minutes. LLM Prob: 0.91 LLM Rescaled Probability<br>**----- End of picture text -----**<br>


Figure 1: The illustrative overview of Predictive-Decoding on one GSM8K example. LLM autoregressive planning often suffers from near sight. Predictive-Decoding rescales LLM generation distribution based on evaluation of foresight, enabling non-myopic planning. 

To investigate the causes and extent of myopia (short-sightedness) in LLM planning, we refer to optimal control theory (Qin & Badgwell, 1997). Optimal planning is framed as a combinatorial optimization problem aimed at maximizing goal achievement through _T_ steps of actions ( _a_ 1 _,a_ 2 _,...,aT_ ). Model Predictive Control (MPC; Camacho et al., 2007) outlines the optimality condition for LLMlike planners – sequential planning must be non-myopic, which requires the anticipation of future outcomes in addition to considering past actions, represented as _p_ ( _at_ ∣ _History, Future_ ). Conversely, focusing solely on _p_ ( _at_ ∣ _History_ ) can lead to irreversible mistakes and potential planning failures. Recent studies (Shih et al., 2023; Bachmann & Nagarajan, 2024) have observed myopia in language models pretrained with next-token prediction. Our research further reveals that even in advanced LLMs like Llama-3 (Dubey et al., 2024), more than half of the reasoning processes on popular math tasks demonstrate a lack of global awareness, with myopia being particularly pronounced in incorrect instances (§3.1). Additionally, LLMs initially do not seem sensitive to mistakes; however, errors accumulate during multi-step planning and become increasingly noticeable over several steps (§3.2). These findings emphasize the importance of thinking with foresight. 

To address this, we propose Predictive-Decoding, a training-free approach to improve LLM planning with non-myopic generation. Predictive-Decoding re-weights LLM distribution based on the closedform solution in Model Predictive Control (Eq. 2), such that sampling from the re-weighted solution is approximately solving planning with global-optimality. For each step _at_ , Predictive-Decoding samples multiple foresight trajectories ( _at,at_ +1 _,...,at_ + _T_ 0) and rescales the original generation distribution based on the evaluations of foresight trajectories. Best action _at_ is sampled from this distribution. The process repeats until a solution is generated at timestep _T_ . An overview is illustrated in Figure 1. Extensive experiments on math, coding, and agent tasks show notable improvements in planning and reasoning accuracy. For example, without using any additional supervision, we achieve a 7.2% improvement on GSM8K (Cobbe et al., 2021) and a 25.3% improvement on AlfWorld (Shridhar et al., 2021) accuracy over ReAct (Yao et al., 2022). Predictive-Decoding’s conditioning on future foresight promotes non-myopic planning and reduces early errors. It requires no special prompt design and preserves generation diversity. 

Furthermore, Predictive-Decoding utilizes inference compute more effectively than other planning approaches. Previous methods have implemented various techniques to enhance planning globalawareness, such as weighted voting of diverse generations (Wang et al., 2022) and searching algorithms (Yao et al., 2024; Xie et al., 2024; Hao et al., 2023; Zhou et al., 2023). While these methods can improve performance, they often require significantly higher computational resources, resulting in sub-optimal computation-performance trade-offs, as noted in recent studies (Snell et al., 2024; Wu et al., 2024b). In contrast, Predictive-Decoding adopts a “no-regrets” approach that optimizes sequentially and simplifies the solution space, avoiding the exponential growth associated with iterative planning methods. This efficiency is reflected in our inference scaling law results with a fixed LLM and reward model, where Predictive-Decoding achieves Pareto superiority over all state-of-the-art (SOTA) baselines (§5.4). Specifically, on the GSM8K benchmark, Predictive-Decoding outperforms MCTS (Hao et al., 2023) by 2.4% and Guided Decoding (Xie et al., 2024) by 1.4% , while utilizing only about 50% of the FLOPS, demonstrating its effectiveness and efficiency. 

2 

Preprint 

Table 1: Non-iterative planning is efficient and suitable for closed-loop settings. Iterative planning, though better for global optimal decisions, is slow and impractical for realistic agent tasks. PredictiveDecoding combines the best of both. Advantages include: G lobal-Optimality of solution, S caling to long-range planning, E fficiency, and C losed-Loop Interaction. 

|**Decision Making**|**Paradigm Illustration**|**Paradigm Illustration**|**Representative Methods**|**Type**|**Advantage**|**Advantage**|**Advantage**|
|---|---|---|---|---|---|---|---|
|||Solution 1|Self-Consistency (Wang et al., 2022)|Sample||||
||…||Tree-of-Thought (Yao et al., 2024)|Search||||
|**Iterative Planning**||Solution N|Guided-Decoding (Xie et al., 2024)|Search|G|||
|||Best Solution|Monte Carlo Tree Search (Hao et al., 2023)<br>Refexion (Shinn et al., 2024)|Search<br>Prompt||||
|**Sequential Planning**||Best Solution|Chain-of-Thought (Wei et al., 2022)<br>ReAct (Yao et al., 2022)|Sample<br>Sample|S<br>E|C||
||||Predictive-Decoding (Ours)|Sample|G<br>S|E|C|



## 2 PROBLEM FORMULATION: AN OPTIMAL CONTROL VIEW 

One of the fundamental problems in decision-making is whether we can solve a long-term goal step by step (Qin & Badgwell, 1997). Traditionally, reinforcement learning approaches have relied on multiple iterations of training (Silver et al., 2014) and searching (Silver et al., 2017) to achieve optimal results. However, recent advancements demonstrate that LLMs possess the capability for sequential planning (Wei et al., 2022). Sequential planning can achieve optimal outcomes if non-myopic, while also offering other benefits, as shown in Table 1. In this section, we first outline the context of the problem and reconsider LLM planning from the perspective of trajectory optimization. 

**Planning as Trajectory Optimization** We consider Partially Observable Markov Decision Processes (POMDPs) defined by tuple ⟨ _g,_ S _,_ A _,_ T _,_ R⟩, with goal _g_ , state space S, valid actions space A, transition function T ∶S × A →S, and R ∶ _r_ ( **s** _,_ **a** ∣ _g_ ) ↦ R[+] is the reward function that measures how well each step aligns with the goal (with the goal dependency omitted for simplicity). _Global-Optimal Planning_ aims to find an action sequence that maximizes cumulative reward over a set number of steps. We model this as a Trajectory Optimization process: find a sequence of actions **a** 0∶ _T_ that maximizes an objective J (Return) factorized over per-timestep rewards _r_ ( _st,at_ ) over a planning horizon with _T_ steps: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0003-06.png)


Conversely, a _Myopic Planner_ maximizes the return on a shorter horizon. For instance, a planner that maximizes the immediate reward at current step, i.e _at_ = max _atr_ ( **s** _t,at_ ), is myopic, while _non-myopic planning_ works towards global-optimality. 

Previous approaches to trajectory optimization follow a straightforward intuition: using large language models (LLMs) to generate diverse trajectories through sampling or search, and then selecting the best one (Wang et al., 2022; Yao et al., 2024; Hao et al., 2023; Xie et al., 2024; Shinn et al., 2024). However, a significant drawback is the exponential growth of the solution space. Each action step offers ∣A∣ possibilities, leading to ∣A∣ _[T]_ potential solutions after _T_ steps. This exponential scaling complicates the optimization process, and despite attempts to prune trajectories and accelerate search (Zhuang et al., 2023; Wang et al., 2024), most iterative planning methods are either time-consuming or perform poorly with limited iterations (Chen et al., 2024b). Additionally, iterative methods are often unsuitable for closed-loop scenarios, where agents interact with their environment in real-time, as each decision alters the environment and is irreversible. 

**Sequential Planning with Model Predictive Control** Model Predictive Control (Qin & Badgwell, 1997) introduces a different paradigm for planning: instead of optimizing an entire action sequence **a**[′] 0∶ _T_[at once, this method selects the best action at each timestep] _[ a]_ 0[′] _[,...,a] t_[′][, fixing each as it progresses] and then optimizing the subsequent steps. This transforms Eq. 1 into a series of optimization problems, each with linear size solution space ∣A∣. The colors denote: the current action to decide _at_ , _history_ , 

3 

Preprint 

and _future_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0004-02.png)


This method requires two steps: first, for each action _at_ , compute the maximum future outcomes (tail subproblem), and then select the action that yields the best future outcomes. Model Predictive Control can also use sampling for inference, but it improves sampling efficiency in two folds: the solution space for each equation is linear and the size of the tail subproblem is reduced after each step. We further provide sample efficiency theoretical analysis and simulations in Appendix B.1. 

**LLM Autoregressive Planning** Solving Eq. 2 is challenging – We need to identify an action that achieves the best outcome after _T_ steps. LLM provides a straightforward solution to this problem: it leverages its extensive knowledge to anticipate the next best action, _a_[′] _t_[, for achieving the goal.][This] method can be formalized as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0004-05.png)


This approach is widely adopted in LLM reasoning and planning, encompassing Chain-of-Thought (CoT; Wei et al. 2022), ReAct (Yao et al., 2022), and Voyager (Wang et al., 2023a). Research has demonstrated that with carefully designed prompts and in-context examples, LLM autoregressive planning can achieve competitive performance in various tasks (Cobbe et al., 2021; Chen et al., 2021; Shridhar et al., 2021). We further investigate whether LLM autoregressive planning is global-optimal. 

## 3 DIAGNOSING THE DEFICIENCY OF LLM PLANNERS 

Comparing Eq.3 with Eq.2, we observe that LLM-based action selection at time step _t_ differs from the optimal control approach. Instead of maximizing future outcomes in the tail subproblem, LLMs select actions based on the immediate conditional probability arg max _at P_[LLM] ( _at_ ∣ **s**[0] _,a_[′] 0∶ _t_ −1[)][.][1] Consequently, the optimality of LLM planning hinges on two hypotheses: (1) _Non-myopic_ : LLMs inherently plan ahead, with current step probabilities accounting for the success of future steps. (2) _Evaluation capability_ : The probability distribution in LLMs can effectively substitute for J , distinguishing successful trajectories from failed ones. In the following sections, we will examine and investigate the validity of these two hypotheses. 

## 3.1 FINDING 1: LLM AUTOREGRESSIVE PLANNING IS MYOPIC 

We first examine whether LLM inherently plan ahead. Next token prediction is greedy; however previous evidence suggests that extensive pretraining could allow LLMs to implicitly plan ahead for future tokens (Wu et al., 2024a). Here we analyze the myopic gap for LLM planning. 

**Definition 3.1.** _(Myopic Gap for LLM Planning)_ Given a language model with distribution _P_ ( _a_ 0 _,a_ 1 _,...,aT_ ), let P be the support set of the distribution. **a**[′] 0∶ _T_[are generated autoregressively] following Eq. 3. Then the myopic gap for planning is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0004-12.png)


> 1 Each token within _at_ is generated based on the highest current probability; we express this at the action level for brevity. 

4 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0005-01.png)


**----- Start of picture text -----**<br>
12<br>20.0 Wrong Wrong Wrong Correct Overall<br>17.5 Correct 10 Correct GSM8K<br>15.0 8 Myopic 290 539 829<br>12.5 Non-myopic 89 401 490<br>10.0 6 Myopic Rate 76.5 % 57.3% 62.8%<br>7.5 4 MATH<br>5.0 Myopic 2172 785 2947<br>2.50.00.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 200.00 0.05 0.10 0.15 0.20 0.25 0.30 Non-myopicMyopic Rate 61.8 1341% 52.7%702 59.1%2043<br>Myopic Gap p* (GSM8K) Myopic Gap p* (MATH)<br>Density Density<br>**----- End of picture text -----**<br>


Figure 2: Myopic Gap distribution for correct and wrong samples (drawn with kde-plot). Myopic examples are defined as _p_[∗] > 0 _._ 01. Wrong samples show a higher myopic rate on both tasks. 

_p_[∗] represents the difference between _P_[LLM] ( _at_ ∣ _history, future_ ) and _P_[LLM] ( _at_ ∣ _history_ ). A positive gap ( _p_[∗] > 0) indicates that the LLM’s action selection is myopic for at least one intermediate step, while _p_[∗] = 0 proves consistent global-awareness during planning. Here we include a small margin ( _p_[∗] > 0 _._ 01 ⇒ myopic) to discriminate myopic and non-myopic cases. 

We empirically calculate the myopic gap of Llama3-8B on two challenging reasoning tasks, GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). We ensemble the generation of token-level Beam Search (Freitag & Al-Onaizan, 2017) and Predictive Decoding (detailed in §4) as the non-myopic target. Results in Figure 2 show that around 60% of all examples show are myopic. Additionally, correct samples are often less myopic, while 76.5% of GSM8K errors and 61.8% of MATH errors occur when LLM planning is myopic. Correct samples tend to have smaller gaps, this motivates us that optimizing generation towards non-myopic may lead to improved results. 

## 3.2 FINDING 2: LLM STRUGGLES TO IDENTIFY MISTAKES IN PLANNING EARLY 

In this section, we explore another bottleneck in LLM sequential planning: Can LLMs evaluate intermediate steps and identify mistakes early on? 

We perform an LLM score calibration analysis by com0.975 Incorrect Answer Score w.r.t steps paring LLM evaluations of intermediate steps with ground 0.950 First Incorrect Step truth human annotations. We collect samples of trajec0.925 Correct Steps Score Density tories _a_ 0∶ _t,t_ ∈(0 _,T_ ] and use the LLM to evaluate these trajectories _P_[LLM] ( _a_ 0 _,a_ 1 _,...,at_ ). GPT-3.5-Turbo is 0.900 used to evaluate trajectories on the agent task AlfWorld, 0.875 while Llama3-8B evaluates steps of the GSM8K trajec0 2 4 6 8 tory. The LLM score for Llama3 is calculated using LLM Step probability, and the score for proprietary GPT-3.5 is obFigure 3: In GSM8K, the first incorrect tained through prompt-based self-evaluation (Xie et al., step’s average score is among correct 2024). Human annotators label each step in the AlfWorld steps, but not after a few steps. trajectory with scores from 0 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _,_ 1, where 0 indicates that the task is unlikely to be completed and 1 indicates a high likelihood of successful completion. Each step in the GSM8K trajectory is labeled as either Correct or Incorrect following Lightman et al. (2023). 

As shown in Table 2, GPT-3.5-Turbo could barely estimate intermediate progress simply with the current trajectory **a** 0∶ _t_ , with only _ρ_ = 0 _._ 133 correlation to human ground truth and Calibration Error 0.165. However, after given future trajectory (foresight of 3 steps), LLM evaluation improves. This is in line with previous observations (Uesato et al., 2022; Lightman et al., 2023) that LLMs are more natural at evaluating complete trajectories, rather than intermediate actions. 

The evaluation of intermediate steps for Llama3-8B on GSM8K shows a calibration error of 0.332 when compared to human ground truth. We observe that incorporating foresight into the evaluation could help avoid early mistakes. As shown in Figure 3, the score for the first incorrect step falls within the high-density region of correct step scores but becomes easily distinguishable from correct steps after a few iterations. Consequently, the GSM8K calibration score decreases to 0.323 when foresight of three steps is applied (detailed in Table 2). However, the correlation does not improve, as the scores of positive steps slightly drops, leading to some overlap with ambiguous incorrect steps. 

5 

Preprint 

Table 2: LLM Score Calibration vs. Human Ground Truth.[‡] ECE for AlfWorld uses a 0.5 threshold for binary classification. 

|Task|Model|Calibration Error ECE<br>(wo. Foresight)|Calibration Error ECE<br>(w. Foresight)↓|Correlation_ρ_<br>(wo. Foresight)|Correlation_ρ_<br>(w. Foresight)↑|
|---|---|---|---|---|---|
|AlfWorld|GPT-3.5-TURBO|0.165‡|0.108‡ ↓|0.133|0.291↑|
|GSM8K|LLAMA-3|0.332|0.323↓|0.417|0.376↓|



## 4 BEYOND MYOPIC GENERATION FOR BETTER PLANNING 

Our findings presented above highlight critical issues with current LLM decoding methods, particularly their tendency toward myopic planning. Additionally, the findings suggest that incorporating global planning strategies can significantly enhance performance. By incorporating future generations, we can calibrate erroneous actions, which directly motivates our proposed method. In this section, we introduce Predictive-Decoding, which follows MPC principles to reduce myopic planning in LLMs. The pipeline is introduced in following sections and the pseudocode is presented in Algorithm 1 in Appendix A. 

## 4.1 PREDICTIVE-DECODING 

Our decision-making strategy draws inspiration from model predictive control (MPC) (Qin & Badgwell, 1997). It solves the _T_ sub-optimization problems (Eq. 2) and addresses myopia with foresight. Here in order to ensure non-myopic planning, our main objective is to generate _a_[′] _t_[according] to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0006-07.png)


where P is the support set of distribution _P_[LLM] ( _at_ ∣ **a**[′] < _t[,]_ **[s]** <[′] _t_[)][, and] **[ a]**[>] _[t]_[∼] _[P]_[ LLM][(⋅∣] **[a]**[′] < _t[,]_ **[s]** <[′] _t[,a][t]_[)][.][Note] that based on results from §3.2, LLMs could accurately evaluate future steps after incorporating a few steps of foresight. Therefore here we evaluate the future constrained to _T_ 0 steps of foresight, i.e. **a** > _t_ ∶= **a** _t_ +1∶ _t_ + _T_ 0. 

We use constrained decoding and sampling to apply soft constraints on outputs, promoting diverse generation and preventing overfitting to our partial ( _T_ 0 steps) foresight. Our method follows the sampling-importance-resampling (SIR) technique from energy-based models (EBM) (Smith & Gelfand, 1992; Ji et al., 2023) to achieve the optimization goal. 

**Proposition 4.1.** _The distribution that solves the optimization problem in Eq.5 is in the form of:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0006-11.png)


The detailed proof is in Appendix B.2. Specifically we first use the LLM to sample a set with size K of foresight rollouts **a** _t_ ∶ _t_ + _T_ 0 given prefix **a**[′] < _t_[in parallel, as well as obtaining the exponential] value of generation probability _wk_ of each rollout. Then we obtain the categorical distribution Categorical ( ∑ _[K] kw_ =11 _[w][k][,]_[⋯] _[,]_ ∑ _[K] kw_ =1 _K[w][k]_[)][. The next action step is determined as the first step in the sampled] rollout from this distribution. In the limit _K_ →∞ and temperature _τ_ → 0, the method recovers the exact maximum value of the distribution. In practice, we can choose _K_ according to computational budget and set _τ_ according to whether we want the model to fit more towards the distribution or generate more diversely. The computation overhead of this method mainly stems from the tokens decoded additionally for foresight. The parallel sampling of multiple rollouts is generally fast for LLM inference infrastructures (Kwon et al., 2023). 

In previous studies, generating text with a lookahead approach has been applied to controllable language model generation (Deng et al., 2020; Lu et al., 2021; Qin et al., 2022). These methods typically involve generating extra tokens to verify if constraints are met, accepting previous generations more likely to satisfy future constraints. Our approach builds on these work, focusing on future generations at the action level rather than the token level, better aligning with large language models’ needs for reasoning and planning. While (Deng et al., 2020; Qin et al., 2022) also use EBM sampling, our method is tailored for pretrained models without additional training. Unlike controllable generation, 

6 

Preprint 

which often requires task-specific heuristics, we demonstrate that lookahead decoding alone can enhance performance in complex planning. 

Predictive-Decoding mainly uses LLM self-evaluation of foresight to improve planning, but this method could also be integrated with an external heuristics _J_ . Eq. 6 can be rewritten as _p_ ( _at_ ) ∝ _P_[LLM] ( _at_ ∣ **a** < _t,_ **s** < _t_ ) exp [E **a** > _t_ J ( **s** 0; **a** 0∶ _T_ )], which uses the heuristic function _J_ rather than LLM probability to evaluate the foresight. This setting enables LLM to utilize discrete heuristics more effectively and makes it easier to compare with other search baselines which often use external reward models. More details are elaborated in §5.3. 

## 4.2 RECYCLING TRAJECTORY ROLLOUTS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0007-04.png)


In previous work, foresight in generation has been applied for controllable tasks (Deng et al., 2020; Lu et al., 2021), typically using token-level constraints with a short lookahead of only a few tokens. In contrast, generating future actions involves creating _K_ samples with longer foresights, which could lead to increased computation with complexity _O_ ( _KT_ 0). Moreover, insufficient sampling number _K_ can inaccurately model the distribution _pτ_ ( _at_ ), while larger _K_ is inefficient. Inspired by work on accelerating LLM inference (Fu et al., 2024), we design a memory pool to recycle sampled trajectories that could reduce the number of sampling _K_ . 

We observe that different sampled trajectories at different time-steps often overlap. For example, when an agent needs to heat a tomato, the trajectory starting from “ _go to the microwave_ → _put the tomato in a microwave_ → _heat the tomato with microwave_ ” overlaps with “open the fridge → take the tomato from fridge → _go to the microwave_ → _put the tomato in a microwave_ → _heat the tomato with microwave_ ” since the tomato needs heating after being taken from the fridge. Thus, previously sampled trajectories and their probabilities ( **a** _t_ ∶ _t_ + _T_ 0 _,pt_ ) can be reused for evaluating timestep _t_[′] ∈( _t,t_ + _T_ 0), provided their trajectory matches **a** _t_ ∶ _t_ ′ = **a**[′] _t_ ∶ _t_[′][.][Similarly, this could be applied] for mathematical reasoning as well as coding, as separate reasoning paths often share steps. 

In implementation, we aggregate all sampled foresights in a trajectory pool. For deciding each new action, we retrieve trajectories with the correct history from the pool and perform the second “resampling” stage based on these valid trajectories. This method recycles trajectories used in estimating foresight, enhancing sampling accuracy. 

## 5 EXPERIMENTS 

In our experiments, we evaluate Predictive-Decoding in two settings: (i) **without additional supervision** , using only LLM self-evaluation (§5.2); (ii) **with an additional reward** (heuristics or reward model) to analyze Predictive-Decoding’s sample efficiency in trajectory optimization (§5.3). 

## 5.1 EXPERIMENT SETTINGS 

**Benchmarks** Our evaluation covers three domains: _math_ , _coding_ , and _agents_ . Math tasks GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) are essential reasoning benchmarks, while coding tasks HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) are also closely related to reasoning ability. We also evaluate on two agent tasks AlfWorld (Shridhar et al., 2021) and PDDL (from Agentboard, Ma et al., 2024) to understand planning ability in closed-loop interactions. 

**Evaluation Settings** We use Llama3 (Dubey et al., 2024) and Mistral-v0.3 (Jiang et al., 2023) for math tasks, and Deepseek-Coder (Guo et al., 2024) for coding tasks. For agent planning which requires stronger models, we include the proprietary GPT-3.5 (Achiam et al., 2023) alongside Llama3.1-70B. We use vLLM inference infrastructure for efficiency (Kwon et al., 2023). To ensure fair comparisons, we use standardized prompts from Guo et al. (2024), Gao et al. (2023), Cobbe et al. (2021), and Ma et al. (2024). Hyperparameter and prompt details are stated in Appendix F 

**FLOPS calculation** Many of our experiments report computational efficiency to better illustrate performance-efficiency tradeoff. We using FLOPS as the metric, Kaplan et al. (2020), i.e. FLOPS ≈ 6 _nP_ , where _P_ is the number of parameters in the LLM, and _n_ the number of generated tokens. We record the average number of generated tokens per example for this calculation. 

7 

Preprint 

Table 3: The accuracy(%) on GSM8K(eight-shot) and MATH (four-shot).[†] CODEX results are taken from Xie et al. (2024). Inference FLOPS scale with the number of generated tokens, denoted as times of _k_ for proprietary models and calculated following (Kaplan et al., 2020) for opensource models. 

|Model<br>Size|Method|Inference FLOPS<br>#Sample|GSM8K|MATH|
|---|---|---|---|---|
|CODEX<br>175B|Autoregressive (PAL, Gao et al. (2023))†<br>_k_<br>_N_ =1<br>72.0<br>-<br>Guided-Decoding(Xie et al., 2024)†<br>138_._5_k_<br>_N_ =1<br>80.2<br>-||||
|LLAMA-3<br>8B|Autoregressive (PAL)<br>Beam Search<br>Autoregressive + Self-Consistency<br>Beam Search + Self-Consistency<br>Guided-Decoding|7_._5×1012<br>_N_ =1<br>478_._6×1012<br>_N_ =1|71.3<br>78.4|29.7<br>**34.4**|
|||59_._8×1012<br>_N_ =8<br>|80.9|31.8|
|||478_._6×1012<br>_N_ =8<br>|79.5|39.4|
|||119_._3×1012<br>_N_ =8|63.9|-|
||Predictive-Decoding<br>Predictive-Decoding + Self-Consistency|161_._3×1012<br>_N_ =1|**78.5**|34.0|
|||129_._0×1013<br>_N_ =8|**81.3**|**40.3**|
|MISTRAL-0.3<br>7B|Autoregressive (PAL)<br>Beam Search|6_._6×1012<br>_N_ =1<br>418_._8×1012<br>_N_ =1|53.4<br>65.5|12.7<br>18.2|
||Predictive-Decoding|141_._1×1012<br>_N_ =1|**66.7**|**19.5**|



Table 4: The pass@1(%) and pass@10(%) rates on HumanEval (zero-shot) and MBPP (zero-shot). MBPP uses the 500 samples test set from huggingface.[†] Results are taken from Roziere et al. (2023). 

|Model<br>Size|Method|FLOPS|HUMANEVAL|MBPP|
|---|---|---|---|---|
|||pass@1|pass@1<br>pass@10|pass@1<br>pass@10|
|GPT-3.5 TURBO†<br>-<br>GPT-4†<br>-|Autoregressive<br>Autoregressive|-|48.1<br>-|52.2<br>-|
|||-|67.0<br>-|-<br>-|
|LLAMA-3<br>8B|Autoregressive<br>Beam Search<br>Self-inflling (Zheng et al., 2023)|2_._9×1012<br>|44.3<br>72.1|33.5<br>48.4|
|||290_._0×1012<br>|53.5<br>70.1|33.6<br>50.2|
|||5_._8×1012|51.8<br>75.6|33.4<br>48.6|
||Predictive-Decoding|76_._6×1012|**57.3**<br>**76.8**|**38.4**<br>**52.6**|
|DEEPSEEKCODER<br>6.7B|Autoregressive<br>Beam Search<br>Self-inflling|2_._4×1012<br>|45.2<br>66.5|38.6<br>53.8|
|||240_._0×1012<br>|46.7<br>50.0|**39.7**<br>42.6|
|||4_._8×1012|46.3<br>76.2|38.4<br>51.6|
||Predictive-Decoding|62_._4×1012|**47.6**<br>**79.9**|34.0<br>**55.2**|



**Baselines** We incoporate iterative planning baselines – Self-Consistency (Wang et al., 2022), GuidedDecoding (Xie et al., 2024), MCTS (Hao et al., 2023) and Self-Infilling (Zheng et al., 2023) – as well as sequential planning baselines – ReAct (Yao et al., 2022), Beam Search, PAL (Gao et al., 2023), COT (Wei et al., 2022)) – in our evaluation. We excluded search agents such as LATS (Zhou et al., 2023) from our comparison, as they primarily address non-environment altering agents (e.g., QA and Retrieval agents). 

## 5.2 PREDICTIVE DECODING MAIN RESULTS 

**Results on Math Tasks** Following (Xie et al., 2024), we use the Program Aided Language Model (PAL) format for each step, and execute the generated code in a Python environment to obtain answers. As shown in Table 3, Predictive-Decoding significantly improves problem-solving accuracy: improve by 7.2% on GSM8K and 4.3% on MATH with Llama3, and 13.3% and 6.8% with Mistral-0.3. Beam Search achieves similar gains, which reduces myopia but requires over 3× the computation. Our method matches or exceeds Beam Search performance and can be further enhanced with SelfConsistency, outperforming both autoregressive and beam search methods in that setting. GuidedDecoding performance well using Codex, but its performance drops using Llama3 due to its overconfident confidence scores. Its performance improves when guided by a reward model (in § 5.3). 

**Results on Coding Tasks** Table 4 presents results for two code generation tasks evaluated using Pass@1 and Pass@10 settings. Pass@10 selects the best from 10 generations, highlighting both accuracy and diversity. Predictive-Decoding outperforms all baselines with Llama3, achieving a strong quality-diversity balance, as detailed in the subsequent temperature analysis. Interestingly, beam search shows less stability in the Pass@10 setting for coding tasks, suggesting that search-based methods may result in lower diversity in generation. 

8 

Preprint 

Table 5: The success rates(%) and the progress rates(%) on AlfWorld (one-shot) and PDDL (one-shot) within 20 steps. Predictive-Decoding uses the same prompt as Act. 

|Model<br>Size|Method|FLOPS<br># Sample|ALFWORLD<br>Success<br>Progress|PDDL<br>Success<br>Progress|
|---|---|---|---|---|
|GPT-3.5 TURBO<br>-|Act (Yao et al., 2022)<br>_k_<br>_N_ =1<br>6.7<br>25.6<br>5.0<br>20.0<br>ReAct (Yao et al., 2022)<br>8_._4_k_<br>_N_ =1<br>7.5<br>**34.8**<br>3.3<br>15.6||||
||Predictive-Decoding|40_._7_k_<br>_N_ =1|**14.9**<br>33.2|**11.7**<br>**28.9**|
|LLAMA-3.1<br>70B|Act<br>ReAct|43_._2×1012<br>_N_ =1<br>488_._0×1012<br>_N_ =1|19.4<br>43.0<br>18.7<br>43.4|35.0<br>54.6<br>18.3<br>36.9|
||Predictive-Decoding|224_._1×1013<br>_N_ =1|**44.0**<br>**53.9**|**38.3**<br>**59.7**|



Table 6: Impact of Foresight _T_ 0, sampling number _K_ and trajectory recycle . gressive generation is denoted as _k_ and is scaled based on number of generated tokens. 

. FLOPS of autore- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0009-05.png)


**----- Start of picture text -----**<br>
# Foresight  T 0 FLOPS GSM8K HUMANEVAL # Sampling  K FLOPS HUMANEVAL w. , wo.<br>T 0 = 1 3 . 6 k 77.0 45.1 K = 2 6 . 6 k 51.2 47.6<br>T 0 = 2 7 . 2 k 76.4 48.8 K = 4 13 . 2 k 54.3 46.3<br>T 0 = 4 14 . 4 k 77.9 52.4 K = 6 19 . 9 k 50.6 49.4<br>T 0 = 6 21 . 5 k 78.5 57.3 K = 8 26 . 4 k 57.3 49.4<br>Performance v.s. Efficiency Performance v.s. Diversity<br>T =6 =0.01 Predictive Decoding =0.6<br>78 T =4 56 =0.3 Predictive Decoding =0.05<br>77 T =1 54 =0.1 Beam Search =0.05<br>T =2<br>76 52<br>75<br>50<br>74 =0.1 =0.8 =5.0<br>=0.5<br>48<br>73<br>Predictive Decoding 46<br>72 Beam Search<br>71 Autoregressive 44 Autoregressive Decoding, =0.6 =1 .0<br>10 20 40 80 160 320 0.01 0.02 0.03 0.04 0.05 0.06<br>Inference FLOPS (10^12) Diversity<br>Pass@1<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 4: Performance v.s. Efficiency on GSM8K. Predictive Decoding is Pareto superior to Beam Search with longer foresight. 

Figure 5: Illustrating Performance and Diversity tradeoff on HumanEval by controlling parameters _τ_ and _α_ . Diversity uses 1- ROUGE score. 

**Results on Agent Tasks** Table 5 presents the performance of two agent tasks: Predictive-Decoding shows improvements of up to 24.6% on two agent tasks and outperforms consistently over Act and ReAct. Notably, Predictive-Decoding achieved more significant enhancements in success rates than in progress rates. While progress rates often reflect an agent’s exploration skills, PredictiveDecoding excels in strategic planning and error reduction, leading to higher success rates. Case studies compared to baselines are available in Appendix E. 

_**Impact**_ **of Foresight Length** _T_ 0 We analyze the impact of foresight length in Table 6. Performances on both GSM8K and HumanEval both improve with longer foresight, supporting our premise that non-myopic generation enhances planning. Notably, Llama3’s myopia on GSM8K is mitigated with just 4 foresight steps, indicating that LLMs already possess partial capability to plan ahead. 

On HumanEval, Predictive-Decoding scale significantly with increased foresight length. This aligns with findings from studies on inference time scaling (Snell et al., 2024), which we discuss further in later sections. Our method is more efficient in mitigating myopia compared to Beam Search, as shown in Figure 4. We achieve this with lower FLOPS, since beam search does not directly optimize for myopia. We further discuss inference compute scaling in §5.4. 

_**Impact**_ **of Sampling number** _K_ **and Trajectory Recycle** We must sample from the original distribution as described in §4.1, but more sampling may reduce efficiency. As shown in Table 6, increasing _K_ leads to better performances as it could give a more accurate account of the original generation distribution. Also, trajectory recycle significantly improve the sampling efficienc. 

_**Impact**_ **of Sampling temperature** _τ_ **and LLM generation temperature** _α_ As shown in Figure 5 when _α_ remains constant, all values of _τ_ enhance the diversity compared to vanilla generation, as 

9 

Preprint 

Table 7: The success rates(%) and the progress rates(%) on AlfWorld (one-shot) and PDDL (one-shot) with different large language models and matching function as heuristics. 

|Model<br>Method<br># Foresight_T_0|ALFWORLD<br>Success<br>Progress|PDDL<br>Success<br>Progress|
|---|---|---|
|GPT-3.5 TURBO<br>Act (Yao et al., 2022)<br>_T_0=0<br>6.7<br>25.6<br>5.0<br>20.0<br>ReAct (Yao et al., 2022)<br>_T_0=0<br>7.5<br>34.8<br>3.3<br>15.6<br>Predictive-Decoding + Reward<br>_T_0=0<br>7.5<br>29.5<br>6.7<br>23.9<br>Predictive-Decoding + Reward<br>_T_0=6<br>**13.4**<br>**37.6**<br>**8.3**<br>**27.9**|||
||7.5<br>29.5<br>**13.4**<br>**37.6**|6.7<br>23.9<br>**8.3**<br>**27.9**|



Table 8: The accuracy(%) on GSM8K (8-shot) with reward model(RM) Math-Shepherd. 

|Model<br>Size|Method<br>Inference FLOPS<br>Reward FLOPS|GSM8K|
|---|---|---|
|LLAMA-3<br>8B|Autoregressive<br>5_._6×1012<br>0_._0×1012<br>70.4<br>RM-weighted Self-Consistency<br>44_._5×1012<br>0_._4×1012<br>82.8<br>RM-based Ranking<br>44_._5×1012<br>0_._4×1012<br>85.9<br>Guided-Decoding + RM<br>276_._2×1012<br>11_._4×1012<br>86.5<br>Monte Carlo Tree Search + RM<br>360_._0×1012<br>16_._2×1012<br>85.5||
|||85.5|
||Predictive-Decoding + RM<br>182_._3×1012<br>2_._9×1012<br>87.9<br>Predictive-Decoding + RM<br>360_._5×1012<br>5_._6×1012<br>**89.9**||
|||**89.9**|



opposed to search-based methods that often sacrifice diversity. Higher _τ_ values increase diversity but can degrade performance by causing myopia. _α_ significantly influences performance outcomes, with the best balance between diversity and performance achieved at moderate _α_ values (0.3-0.6). 

**More analysis** More analysis including agent world model and prompt sensitivity are in Appendix C. 

## 5.3 PLANNING AS REWARD OPTIMIZATION 

Recall that Predictive-Decoding samples from a distribution to solve the trajectory optimization problem using LLM evaluation as the proxy for calculating future return. Similarly we can define such distribution for any target. The same sampling technique could maximize the objective J left to right, achieving global-optimal planning. We examine how Predictive-Decoding performs with two different types of objective: 

**Guiding Agent Planning with Heuristics** We employ human-designed reward heuristics _J_ to guide LLM planning, using a semantic matching score to compare the current observation to the goal state, following the approach recommended by Ma et al. (2024). The heuristic rewards are sparse, requiring multiple steps to increase rewards. As shown in Table 7, even without foresight, the reward enhances LLM agent performance. Predictive-Decoding further overcomes the sparsity with foresight and improves the utility of reward functions. 

## **Guiding LLM Reasoning with Reward Model** We 

use Math-Shepherd (Wang et al., 2023b) as the reward model for GSM8K. We compare PredictiveDecoding against other reward model based planning methods, including: (1) Reward Model weighted Self-Consistency; (2) Reward Model based Ranking, selecting the top result from multiple samples; (3) Guided-Decoding; and (4) Monte Carlo Tree Search. Results in Table 8 demonstrate that all algorithms improve by over 10% with Math-Shepherd. Although strong baselines like Guided-Decoding and MCTS are effective, they require extensive computation. Predictive-Decoding achieves a 1.5% improvement over Guided-Decoding with only 65% of the computation and a 2.4% improvement over MCTS with just 50% of the computation. We further compare the sample efficiency of these methods in §5.4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0010-12.png)


**----- Start of picture text -----**<br>
Inference Scaling Law on GSM8K<br>90<br>88<br>86<br>84<br>82<br>80 Autorgressive + Rank<br>Autorgressive + Weighted Voting<br>78 Guided Decoding<br>MCTS<br>76 Predictive DecodinPredictive Decoding (K=2,Tg (T =6, K=2,4,8,16)=1,2,4,6)<br>4 8 16 32 64 128 256 512<br>Inference FLOPS (10^12)<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 6: Inference compute scaling law on 189 examples sampled from GSM8K. All algorithms use Llama3-8B and Math-Shepherd. 

10 

Preprint 

## 5.4 DISCUSSION: INFERENCE SCALING LAW 

Snell et al. (2024) recently stressed the importance of inference scaling law, which states that LLM inference could benefit from using more computation to support more expressive algorithms. Wu et al. (2024b) further discuss the efficiency-performance trade-off of different search algorithms. However, it is yet to be discussed if searching is the most efficient and effective choice for LLM reasoning. 

We follow setting in §5.3 using reward model Math-Shepherd. In Figure 6, we plot the inference scaling law of various methods and observe that: (i) Sampling-based methods have better computation efficiency compared to search-based methods. However autoregressive generation using simple reward ranking tends to saturate when more computation is available. (ii) Predictive-Decoding which directly samples from optimal solution space achieves better scaling law than all sampling and search baselines. Predictive-Decoding performance also consistently improves with more computation. 

## 6 RELATED WORK 

**LLM-based Planning and Reasoning** One major development is the emergence of LLM step-by-step reasoning abilities (Wei et al., 2022). This capability is further enhanced by more expressive inference algorithms like searching (Yao et al., 2022; 2024; Hao et al., 2023; Wang et al., 2023a; Zheng et al., 2023; Xie et al., 2024; Zhao et al., 2024; Sun et al., 2024). However, most of these algorithms are computation-intensive. Recent work discusses the inference time scaling law for LLM reasoning (Wu et al., 2024b; Snell et al., 2024). This highlights the need for an efficient yet global-optimal method. Our work attempts to close this research gap by introducing Predictive-Decoding. Another line of work studies the expressiveness of LLM reasoning. Feng et al. (2024) confirms LLM reasoning’s expressiveness equals dynamic programming. Li et al. (2024) discusses Turing completeness in chain-of-thought reasoning. Our theory does not contradict their claims but examines whether any pretrained LLM could find such an optimal solution. 

**Model Predictive Control** Model Predictive Control (MPC) is a widely used control strategy that involves solving an optimization problem at each time step by forecasting future results (Witkin & Kass, 1988; Qin & Badgwell, 1997). Similarly, in model-based reinforcement learning (RL), an agent aims to maximize future success by using a dynamics model to simulate state transitions (Silver et al., 2017; Ha & Schmidhuber, 2018; Anthony et al., 2017; Racaniere et al., 2017; Nagabandi et al., 2018).` These algorithms enable agents to think ahead, envisioning the outcomes of various potential actions, and making deliberate selections among alternatives. Our method follows this line of work. 

**Combinatorial Optimization via Sequential Sampling** Sampling-based methods have been extensively used to solve combinatorial optimization efficiently (Sun & Yang, 2023; Janner et al., 2022; Qin et al., 2022; Du et al., 2024), by constructing an energy-based model where the generative probability reflects the objective. However, these methods often use non-autoregressive sampling, which undermines the Markov property of sequences critical to tasks like planning and language modeling. Autoregressive diffusion models (Wu et al., 2023; Chen et al., 2024a) have been proposed to sample based on optimization constraints while enabling causal generation. However, these methods rely on diffusion training loss and cannot be directly applied to LLMs. In this work, we follow Deng et al. (2020); Ji et al. (2023) and use the sampling-importance-resampling (SIR) technique, which enables us to maintain autoregressive next-token prediction while achieving global optimality. This method can be easily combined with model predictive control and used on any LLM. 

## 7 CONCLUSIONS AND LIMITATION 

In this paper, we analyzed the limitations and potential of Large Language Models (LLMs) in planning and reasoning tasks, particularly focusing on their myopic tendencies. We introduced Predictive-Decoding, which employs Model Predictive Control to enhance non-myopic planning, significantly improving performance across various domains. Our experiments confirmed PredictiveDecoding’s effectiveness in boosting planning accuracy and computational efficiency. These results open promising avenues for future research into optimizing LLM reasoning and incorporating foresight into decision-making, paving the way for more robust and efficient LLM applications. 

11 

Preprint 

The main limitation of our work is that we are targeted towards improving reasoning and planning at inference times, without exploring how our method could aid in generating training data—an advantage of search-based methods. Additionally, Predictive-Decoding is limited to tasks with clear reasoning steps, like math or coding. While it potentially could aid global awareness in free-form generation, such as long-context tasks, we leave this for future research. 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. _Advances in neural information processing systems_ , 30, 2017. 

- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. _arXiv preprint arXiv:2403.06963_ , 2024. 

- Eduardo F Camacho, Carlos Bordons, Eduardo F Camacho, and Carlos Bordons. _Constrained model predictive control_ . Springer, 2007. 

- Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. _arXiv preprint arXiv:2407.01392_ , 2024a. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. 

Ziru Chen, Michael White, Raymond Mooney, Ali Payani, Yu Su, and Huan Sun. When is tree search useful for llm planning? it depends on the discriminator. _arXiv preprint arXiv:2402.10890_ , 2024b. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Yuntian Deng, Anton Bakhtin, Myle Ott, Arthur Szlam, and Marc’Aurelio Ranzato. Residual energy-based models for text generation. _arXiv preprint arXiv:2004.11714_ , 2020. 

- Yilun Du, Jiayuan Mao, and Joshua B Tenenbaum. Learning iterative reasoning through energy diffusion. _arXiv preprint arXiv:2406.11179_ , 2024. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. _Advances in Neural Information Processing Systems_ , 36, 2024. 

12 

Preprint 

Markus Freitag and Yaser Al-Onaizan. Beam search strategies for neural machine translation. _arXiv preprint arXiv:1702.01806_ , 2017. 

- Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding. _arXiv preprint arXiv:2402.02057_ , 2024. 

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. 

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. _arXiv preprint arXiv:2401.14196_ , 2024. 

David Ha and J¨urgen Schmidhuber. World models. _arXiv preprint arXiv:1803.10122_ , 2018. 

- Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. _arXiv preprint arXiv:2305.14992_ , 2023. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. _arXiv preprint arXiv:2205.09991_ , 2022. 

- Haozhe Ji, Pei Ke, Hongning Wang, and Minlie Huang. Language model decoding as direct metrics optimization. _arXiv preprint arXiv:2310.01041_ , 2023. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , 2023. 

- Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. _arXiv preprint arXiv:2001.08361_ , 2020. 

- Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble trust-region policy optimization. _arXiv preprint arXiv:1802.10592_ , 2018. 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Yu, Joey Gonzalez, Hao Zhang, and Ion Stoica. vllm: Easy, fast, and cheap llm serving with pagedattention. _See https://vllm. ai/(accessed 9 August 2023)_ , 2023. 

Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, Fujie Huang, et al. A tutorial on energy-based learning. _Predicting structured data_ , 1(0), 2006. 

- Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. _arXiv preprint arXiv:2402.12875_ , 2024. 

- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- Ximing Lu, Sean Welleck, Peter West, Liwei Jiang, Jungo Kasai, Daniel Khashabi, Ronan Le Bras, Lianhui Qin, Youngjae Yu, Rowan Zellers, et al. Neurologic a* esque decoding: Constrained text generation with lookahead heuristics. _arXiv preprint arXiv:2112.08726_ , 2021. 

- Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents. _arXiv preprint arXiv:2401.13178_ , 2024. 

13 

Preprint 

- Anusha Nagabandi, Gregory Kahn, Ronald S Fearing, and Sergey Levine. Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. In _2018 IEEE international conference on robotics and automation (ICRA)_ , pp. 7559–7566. IEEE, 2018. 

- Lianhui Qin, Sean Welleck, Daniel Khashabi, and Yejin Choi. Cold decoding: Energy-based constrained text generation with langevin dynamics. _Advances in Neural Information Processing Systems_ , 35:9538–9551, 2022. 

- S Joe Qin and Thomas A Badgwell. An overview of industrial model predictive control technology. In _AIche symposium series_ , volume 93, pp. 232–256. New York, NY: American Institute of Chemical Engineers, 1971-c2002., 1997. 

- Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. _arXiv preprint arXiv:2307.16789_ , 2023. 

- Sebastien´ Racaniere,` Theophane´ Weber, David Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adria` Puigdomenech` Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, et al. Imagination-augmented agents for deep reinforcement learning. _Advances in neural information processing systems_ , 30, 2017. 

- Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jer´ emy Rapin, et al.´ Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ , 2023. 

- Andy Shih, Dorsa Sadigh, and Stefano Ermon. Long horizon temperature scaling. In _International Conference on Machine Learning_ , pp. 31422–31434. PMLR, 2023. 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and Matthew´ Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2021. URL https://arxiv.org/abs/2010.03768. 

- David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In _International conference on machine learning_ , pp. 387–395. Pmlr, 2014. 

- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. _nature_ , 550(7676):354–359, 2017. 

- Adrian FM Smith and Alan E Gelfand. Bayesian statistics without tears: a sampling–resampling perspective. _The American Statistician_ , 46(2):84–88, 1992. 

- Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. _arXiv preprint arXiv:2408.03314_ , 2024. 

- Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. Adaplanner: Adaptive planning from feedback with language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Zhiqing Sun and Yiming Yang. Difusco: Graph-based diffusion solvers for combinatorial optimization. _Advances in Neural Information Processing Systems_ , 36:3706–3731, 2023. 

- Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. _arXiv preprint arXiv:2211.14275_ , 2022. 

- Ante Wang, Linfeng Song, Ye Tian, Baolin Peng, Dian Yu, Haitao Mi, Jinsong Su, and Dong Yu. Litesearch: Efficacious tree search for llm. _arXiv preprint arXiv:2407.00320_ , 2024. 

14 

Preprint 

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. _arXiv preprint arXiv:2305.16291_ , 2023a. 

Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: A label-free step-by-step verifier for llms in mathematical reasoning. _arXiv preprint arXiv:2312.08935_ , 2023b. 

- Xuezhi Wang and Denny Zhou. Chain-of-thought reasoning without prompting. _arXiv preprint arXiv:2402.10200_ , 2024. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- Andrew Witkin and Michael Kass. Spacetime constraints. _ACM Siggraph Computer Graphics_ , 22(4): 159–168, 1988. 

- Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, Jian Jiao, Juntao Li, Jian Guo, Nan Duan, Weizhu Chen, et al. Ar-diffusion: Auto-regressive diffusion model for text generation. _Advances in Neural Information Processing Systems_ , 36:39957–39974, 2023. 

Wilson Wu, John X Morris, and Lionel Levine. Do language models plan ahead for future tokens? _arXiv preprint arXiv:2404.00859_ , 2024a. 

- Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. An empirical analysis of compute-optimal inference for problem-solving with language models. _arXiv preprint arXiv:2408.00724_ , 2024b. 

- Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. Self-evaluation guided beam search for reasoning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Haiteng Zhao, Chang Ma, Guoyin Wang, Jing Su, Lingpeng Kong, Jingjing Xu, Zhi-Hong Deng, and Hongxia Yang. Empowering large language model agents through action learning. _arXiv preprint arXiv:2402.15809_ , 2024. 

- Lin Zheng, Jianbo Yuan, Zhi Zhang, Hongxia Yang, and Lingpeng Kong. Self-infilling code generation. _arXiv preprint arXiv:2311.17972_ , 2023. 

- Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models. _arXiv preprint arXiv:2310.04406_ , 2023. 

- Yuchen Zhuang, Xiang Chen, Tong Yu, Saayan Mitra, Victor Bursztyn, Ryan A Rossi, Somdeb Sarkhel, and Chao Zhang. Toolchain*: Efficient action space navigation in large language models with a* search. _arXiv preprint arXiv:2310.13227_ , 2023. 

15 

Preprint 

## A PSEUDO CODE 

## **Algorithm 1** Predictive-Decoding for Planning 

**Input:** prompt, the language model, maximum number of iterations _T_ , sampling number _K_ , sampling temperature _τ_ , environment env, rollout length _T_ 0. **Output:** Action sequence _a_[′] 0 _[,...a]_[′] _T_[.] 

Set **s** 0 ← Initialize env, finish ← False **for** _t_ = 1 _,_ 2 _,...,T_ **do** ▷ Sample Foresight. Input prompt and **s**[′] 0 _[,a]_[′] 0 _[,...,]_ **[s]** _t_[′][to language model][ context][;] **for** _k_ = 1 _,_ 2 _,...,K_ **do** // In parallel Sample _a[k] t[,]_ **[s]** _[k] t_ +1 _[,a][k] t_ +1 _[,...,a][k] t_ + _T_ 0 _[,]_ **[s]** _[t]_[+] _[T]_ 0[+][1][ ∼] _[P]_[ LLM][(⋅∣][context][)][;] _Pk_ ← _P_[LLM] ( **a** _[k]_ ≥ _t[,]_ **[s]** > _[k] t_[∣][context][)][;] _wk_ ← exp ( _Pk_ / _τ_ ); **end for** ▷ Re-sample based on foresight. ∼ _w_ 1 _wK_ Sample _j_ Categorical ( ∑ _[K] k_ =1 _[w][k][,]_[⋯] _[,]_ ∑ _[K] k_ =1 _[w][k]_[)][;] Set _a_[′] _t_[←] _[a][j] t_[;] ▷ Takes the action _a_[′] _t_ Update **s** _t_ +1 _,_ finish ← Execute _a_[′] _t_[in environment][ env][;] **break if** finish is True; **end for Return** the action sequence _a_[′] 0 _[,...a]_[′] _T_[.] 

## B THEORETICAL RESULTS 

- B.1 SAMPLE EFFICIENCY OF MODEL PREDICTIVE CONTROL 

In this section, we explain in detail why model predictive control has the potential to improve sample efficiency in solving trajectory optimization problems. 

We begin by introducing a stochastic setting, where trajectory _τ_ = ( _a_ 0 _,a_ 1 _,...,aT_ ) is a chain of actions that could take values in set A. A policy model models the distribution of trajectories as _πθ_ ( _τ_ ), denoting the probability that the policy model samples the trajectory. 

For the best scenario, the policy model is perfect and the sample _τ_[′] ∼ _πθ_ ( _τ_ ) maximizes the objective J ( _τ_ ) and _τ_[′] is the best solution. However, in most cases our LLM is not the perfect policy model and prone to make errors during sampling. Instead, we sample multiple solutions _τ_ 1 _,τ_ 2 _,...,τk_ ∼ _πθ_ ( _τ_ ) and selects the best solution _τ_[′] = max _i_ =1 _,...,k_ J ( _τi_ ). A different method is using model predictive control during sampling. We thereby discuss the number of _k_ necessary to obtain the correct _τ_[′] . 

**Proposition B.1.** _Given the optimal trajectory τ_[′] = ( _a_[′] 0 _[,...,a]_[′] _T_[)] _[that][maximizes]_[J] _[ .] τ_[1] _,τ_[2] _,...,τ[k]_[∣A∣] ∼ _πθ_ ( _τ_ ) _. The best a[r]_ 0 _[is the first step of the trajectory][ τ][ j][that maximizes]_[ J] _[ .]_ 

_If a[m]_ 0 _[satisfies:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0016-12.png)


_Informal proof_ : If _π_ has no knowledge of J , then the best case in sampling would be _τ_[1] _,τ_[2] _,...,τ[k]_[∣A∣] first sample _a_ 0 from _U_ ( _a_ 0), then samples _τ_ → _a_ 0 ∼ _πθ_ (⋅∣ _a_ 0), in order to gain thorough information about which _a_ 0 would be the best for the trajectory. In this case, the sampling process is same as model predictive control and thus the best case results in _P_ ( _a[m]_ 0[=] _[ a]_[′] 0[) =] _[ P]_[(] _[a][j]_ 0[=] _[ a]_ 0[′][)][.][However, when] _πθ_ places more importance on _a_ 0 /= _a_[′] 0[, i.e.][∃] _[a]_[0] _[,π]_[(] _[a]_[∗] 0[) >] _[ π]_[(] _[a]_[′] 0[)][, there exists a larger possibility that] _a_[∗] 0[is selected by] _[ J]_[instead of] _[ a]_[′] 0[than model predictive control, as let] _[ n]_[ be the times] _[ a]_[∗] 0[is sampled,] 

16 

Preprint 

_k_ > _n_ , then _E_ [max _i_ =1 _,...,n_ J ( _a_[∗] 0 _[,τ][ i]_ → _a_[∗] 0[)] ≥] _[E]_[ [][max] _[i]_[=][1] _[,...,k]_[ J (] _[a]_ 0[∗] _[,τ][ i]_ → _a_[∗] 0[)]][.][Thus the probability that] _a_[∗] 0[is selected would be larger and] _[ P]_[(] _[a][m]_ 0[=] _[ a]_[′] 0[) ≥] _[P]_[(] _[a][j]_ 0[=] _[ a]_ 0[′][)][.] 

This proposition show that if directly sample solution from an exponential solution space, it would be unlikely to obtain the optimal solution. We also showcase this phenomenon that model predictive control improves sample efficiency than direct sampling through simulated experiments. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0017-03.png)


**----- Start of picture text -----**<br>
1.0 Naive Iterative Sampling<br>Model Predictive Control 1.0<br>0.8<br>0.8<br>0.6<br>0.6<br>0.4<br>0.4<br>0.2 Naive Iterative Sampling =1<br>0.2 Model Predictive Control =1<br>Naive Iterative Sampling =5<br>Model Predictive Control =5<br>0.0 Naive Iterative Sampling =10<br>4 6 8 10 12 14 0.0 Model Predictive Control =10<br>Problem Scale (T) 250 500 750 1000 1250 1500 1750<br>Sampling Number (N)<br>Accuracy<br>Success Rate With 1000 Samples<br>**----- End of picture text -----**<br>


Figure 7: Pass@1000 accuracy v.s. problem scale on the simulation task. _πθ_ follows a uniform distribution. The performance of iterative sampling decreases significantly when the problem scales up. 

Figure 8: Pass@k accuracy on simulation task with different sampling numbers. Larger _τ_ denotes that the sampling distribution _πθ_ is close to ground truth model. 

We focus on a very simple combinatorial optimization problem, where a list ( _a_ 0 _,a_ 1 _,...,aT_ ) satisfies _a_ 0 ∈A takes the value from ∣A∣ integers. Our objective is to find the list of numbers with maximum sum. We control the sample complexity of “random sampling + ranking” and “model predictive control”. Results are shown in Figure 7 and Figure 8. On all settings, MPC is more sample efficient than naive iterative sampling. Notably, we alter the prior distribution _πθ_ such that it samples the number based on value _e_[number][/] _[τ]_ . Larger _τ_ is closer to uniform distribution yet smaller _τ_ is closer to the optimal distribution. We can see that a good prior always results in improved performances, and the gap between iterative sampling and MPC is smaller. However, when the _πθ_ is not good (e.g. using a LLM that is not well-pretrained), the sample efficiency of MPC is much better than iterative sampling. 

## B.2 PROOF FOR PROPOSITION 4.1 

The distribution that solves the optimization problem in Eq.5 is in the form of: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0017-09.png)


_Proof._ The goal is to solve this optimization problem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0017-11.png)


Energy-based Model can transfer the optimization problem into probability distribution, such as, 

**Lemma B.2.** _(LeCun et al., 2006) Let_ _**f**_ = { _fi_ } _i_ =1 _...n is a set of evaluation metrics, the distribution p_ ( _x_ ) _solves the optimization problem x_ = arg min ∑ _i µifi_ ( _x_ ) _, only if p_ ( _x_ ) _satisfies: x_ ∈P 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0017-14.png)


_where_ P _is the support set of probability distribution of pθ._ 

Here _at_ is generated by LLM through autoregressive generation, i.e. _at_ ∼ _P_[LLM] ( _at_ ∣ **a** < _t,_ **s** < _t_ ), and the constraint can be rewritten as _f_ ( _at_ ) = E **a** > _t,_ **s** > _tP_[LLM] ( _at,_ **a** > _t,_ **s** > _t_ ∣ **a**[′] < _t[,]_ **[s]** <[′] _t_[)][.][Plugging into Eq.][9] and we would yield _pat_ . _τ_ can be added as a temperature coefficient and does not lead to difference in optimization. 

17 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0018-01.png)


**----- Start of picture text -----**<br>
1.00 Pearson :0.1331 1.00 Pearson :0.291 1.00 Pearson :0.5212<br>0.75 0.75 0.75<br>0.50 0.50 0.50<br>0.25 0.25 0.25<br>0.00 0.00 0.00<br>0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00<br>Self-Evaluation Score Self-Evaluation with Foresight (World Model)  Self-Evaluation with Foresight<br>Human Evaluation Score Human Evaluation Score Human Evaluation Score<br>**----- End of picture text -----**<br>


Figure 9: Comparison of GPT-3.5-Turbo Self-Evaluation _r_ ˆ( _s,a_ ) w/o. modeling on 152 trajectory data from AlfWorld. We compare from left-to-right (1) LLM direct self-evaluation based on past trajectory( _ρ_ = 0 _._ 133),(2) LLM first perform model rollout and then perform self-evaluation based on both past and future trajectory( _ρ_ = 0 _._ 291), (3)substituting LLM-as-world-model for an accurate world model for rollout( _ρ_ = 0 _._ 521) against human evaluation. All scores are chosen from _{_ 0,0.25,0.5,0.75,1 _}_ and we use KDE plots to visualize overlapping discrete points. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0018-03.png)


**----- Start of picture text -----**<br>
80.0 0.0<br>77.5 -3.4 Vanilla PAL Predictive Decoding<br>75.0 -4.4 -5.1<br>72.5 0.0<br>70.0<br>67.5 -4.1<br>65.0 -8.2 -8.5<br>62.5<br>60.0 Original Brief No example 1-shot example<br>Prompt Type<br>Performance<br>**----- End of picture text -----**<br>


Figure 10: Prompt sensitivity to instruction brevity, as well as the number of in context samples on GSM8K. Both methods use Llama3-8B. 

## C MORE ANALYSIS 

_**Impact**_ **of Agent World Model Hallucination** Predictive-Decoding requires the LLM to act as a world model to infer states after each step. However, we often observe hallucinations. For instance, in AlfWorld, when the agent tries to find an apple and place it on desk 1, the imagined state after ”open fridge 1” might incorrectly include an apple in the fridge, even if it’s not there. This reflects an inherent model bias issue (Kurutach et al., 2018), where the world model fails to accurately represent the environment. Figure 9 shows that this results in lower LLM score accuracy compared to using real environment transitions. Most evaluation errors are false positives, indicating LLM overconfidence in predicting step outcomes. Increasing _τ_ to 0.05-0.1 helps mitigate this by preventing overfitting. 

**Prompt Sensitivity Analysis** Wang & Zhou (2024) proposes an intereting problem: improving LLM reasoning without prompting. We find Predictive-Decoding can also improves performances and maintain high performance even without in-context samples and detailed instructions. As shown in Figure 10, vanilla generation with LLM show significant drop in performance when the detailed instructionsa are not given, or when there are no in-context examples. Predictive-Decoding shows smaller decline in performance, demonstrating better prompt sensitivity. Moreover, PredictiveDecoding without in-context examples could outperforms vanilla generation with in-context examples. This demonstrate our method also has the potential to improve reasoning without prompting. 

## D FURTHER DISCUSSION ON RELATED WORK 

**Combinatorial Optimization via Sequential Sampling** Sampling-based methods have been extensively used to solve combinatorial optimization efficiently (Sun & Yang, 2023; Janner et al., 2022; Qin et al., 2022; Du et al., 2024), by constructing an energy-based model where the generative probability reflects the objective. However, these methods often use non-autoregressive sampling, 

18 

Preprint 

which undermines the Markov property of sequences critical to tasks like planning and language modeling. Autoregressive diffusion models (Wu et al., 2023; Chen et al., 2024a) have been proposed to sample based on optimization constraints while enabling causal generation. However, these methods rely on diffusion training loss and cannot be directly applied to LLMs. In this work, we follow Deng et al. (2020); Ji et al. (2023) and use the sampling-importance-resampling (SIR) technique, which enables us to maintain autoregressive next-token prediction while achieving global optimality. This method can be easily combined with model predictive control and used on any LLM. 

## E CASE STUDY 

**Case Study for Reasoning** Here we provide the generations to show how Predictive-Decoding solves a math problem. 

Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep? 

def solution(): 

_{_ ‘ #Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?’: 1.0 _}_ 

Line 1: #Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep? 

_{_ ‘ seattle ~~s~~ heep = 20’: 1.0 _}_ 

Line 2: seattle ~~s~~ heep = 20 

_{_ ‘ charleston ~~s~~ heep = seattle ~~s~~ heep * 4’: 0.9767, ‘ charleston ~~s~~ heep = 4 * seattle ~~s~~ heep’: 0.0233 _}_ 

Line 3: charleston ~~s~~ heep = seattle ~~s~~ heep * 4 

_{_ ‘ toulouse ~~s~~ heep = charleston ~~s~~ heep * 2’: 0.9377, ‘ toulouse ~~s~~ heep = seattle ~~s~~ heep * 2’: 0.0623 _}_ 

Line 4: toulouse ~~s~~ heep = charleston ~~s~~ heep * 2 

_{_ ‘ total ~~s~~ heep = seattle ~~s~~ heep + charleston ~~s~~ heep + toulouse ~~s~~ heep’: 1.0 _}_ 

Line 5: total ~~s~~ heep = seattle ~~s~~ heep + charleston ~~s~~ heep + toulouse ~~s~~ heep’ 

_{_ ‘ result = total ~~s~~ heep’: 0.9999 _}_ 

Line 6: result = total ~~s~~ heep 

_{_ ‘ return result’:1.0 _}_ 

Line 7: return result 

Examples in trajectory pool: 

Foresight 1 for the first line (composed of 6 steps): 

_{_ 'Action': None, 'Reward': 0.9984305701520592 _}_ 

_{_ 'Action': ' """Toulouse has twice as many sheep as Charleston. ↪ Charleston has 4 times as many sheep as Seattle. How many sheep do ↪ Toulouse, Charleston, and Seattle have together if Seattle has 20 ↪ sheep?"""\n', 'Reward': 0.9984305701520592 _}_ 

_{_ 'Action': ' seattle_sheep = 20\n', 'Reward': 0.9984305701520592 _}_ 

_{_ 'Action': ' charleston_sheep = seattle_sheep * 4\n', 'Reward': 0 ↪ .9984305701520592 _}_ , 

19 

Preprint 

_{_ 'Action': ' toulouse_sheep = charleston_sheep * 2\n', 'Reward': 0 ↪ .9984305701520592 _}_ 

_{_ 'Action': ' total_sheep = seattle_sheep + charleston_sheep + ↪ toulouse_sheep\n', 'Reward': 0.9984305701520592 _}_ 

Foresight 1 for the third line (composed of 4 steps): 

_{_ 'Action': ' charleston_sheep = seattle_sheep * 4\n', 'Reward': 0 ↪ .9842750955836792 _}_ , 

_{_ 'Action': ' toulouse_sheep = 2 * charleston_sheep\n', 'Reward': 0 ↪ .9842750955836792 _}_ 

_{_ 'Action': ' total_sheep = seattle_sheep + charleston_sheep + ↪ toulouse_sheep\n', 'Reward': 0.9842750955836792 _}_ 

_{_ 'Action': ' result = total_sheep\n', 'Reward': 0.9842750955836792 _} {_ 'Action': ' return result\n', 'Reward': 0.9842750955836792 _}_ 

From the case study we could observe that for easier steps, the distribution is concentrated on a single action. This is due to LLM generation is inherently confident on these lines. Our method mainly helps improving the lines where LLM generation is less confident (e.g. Line 3 & 4). 

**Case Study for Agent** In this section we also provide a case study for agent task. Agent tasks are different from reasoning tasks in two ways: (i) Agent tasks make predictions not only based on prior actions, but also on environment observations. Therefore, during foresight, the agent needs to use LLM as a world model (Hao et al., 2023) to infer the next state. (ii) Exploration is crucial for agents. Therefore maintaining stochastic during sampling is important. 

Goal: look at bowl under the desklamp. 

Observation: You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1. 

_{_ ‘go to desk 1’: 0.9975, ‘go to desk 2’: 0.0025 _}_ 

Step 1: go to desk 1 

Observation: use desklamp 1 

_{_ ’use desklamp 1’: 0.4530, ’examine desklamp 1’: 0.5469, ’examine desk 2’: 8.551e-06 _}_ Step 2: use desklamp 1 

Observation: You turn on the desklamp 1. 

_{_ ’examine desk 1’: 0.9822, ’examine desklamp 1’: 0.0177 _}_ 

Step 3: examine desk 1 

Observation: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2. 

_{_ ’examine desklamp 1’: 0.5711, ’go to desk 2’: 0.4289 _}_ 

Step 4: examine desklamp 1 

Observation: Nothing happens. 

_{_ ’go to shelf 1’: 0.9974, ’go to desk 2’: 0.00258 _}_ 

20 

Preprint 

Step 5: examine shelf 1 

Observation: On the shelf 1, you see a cellphone 3, and a creditcard 2. 

_{_ ’go to desk 2’: 0.8146, ’go to shelf 2’: 0.1853 _}_ Step 6: go to desk 2 

Observation: On the desk 2, you see a alarmclock 1, a bowl 1, and a cd 3. 

_{_ ’take bowl 1 from desk 2’: 0.9626, ’examine bowl 1’: 0.03733 _}_ Step 7: take bowl 1 from desk 2 

Observation: You pick up the bowl 1 from the desk 2. 

_{_ ’go to desk 1’: 1.0 _}_ 

Step 8: go to desk 1 

Observation: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2. The task is finished. 

Examples in trajectory pool: 

Foresight 1 for the first step (composed of 5 steps): 

_{_ 'Action': None, 'Observation': 'You are in the middle of a room. ↪ Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a ↪ drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer ↪ 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf ↪ 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.', 'Reward': 0 ↪ .9281983462766966 _}_ 

_{_ 'Action': 'go to desk 1', 'Observation': 'On the desk 1, you see a book ↪ 1, a computer 1, a desklamp 1, a notebook 1, and a pen 1.', ↪ 'Reward': 0.9281983462766966 _}_ 

_{_ 'Action': 'examine desklamp 1', 'Observation': 'The desklamp 1 is ↪ currently turned off.', 'Reward': 0.9281983462766966 _}_ 

_{_ 'Action': 'use desklamp 1', 'Observation': 'You turn on the desklamp 1 ↪ .', 'Reward': 0.9281983462766966 _}_ 

_{_ 'Action': 'examine desk 1', 'Observation': 'On the desk 1, you see a ↪ book 1, a computer 1, a desklamp 1, a notebook 1, a pen 1, and 

↪ nothing is under the desklamp 1.', 'Reward': 0.9281983462766966 _}_ 

_{_ 'Action': 'go to desk 2', 'Observation': 'On the desk 2, you see a bowl ↪ 1, a cup 1, and a desklamp 2.', 'Reward': 0.9281983462766966 _}_ 

Foresight 1 for the second step (composed of 5 steps): 

_{_ 'Action': 'go to desk 1','Observation': 'On the desk 1, you see a ↪ desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2 ↪ .', 'Reward': 0.9279943292829683 _}_ 

_{_ 'Action': 'use desklamp 1', 'Observation': 'You turn on the desklamp 1 ↪ .', 'Reward': 0.9279943292829683 _}_ 

_{_ 'Action': 'examine desk 1', 'Observation': 'On the desk 1, you see a ↪ desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, a pencil 2, ↪ and a drawer 1.', 'Reward': 0.9279943292829683 _}_ 

- _{_ 'Action': 'open drawer 1', 'Observation': 'You open the drawer 1. ↪ Inside, you see a bowl 1.', 'Reward': 0.9279943292829683 _}_ 

21 

Preprint 

- _{_ 'Action': 'take bowl 1 from drawer 1', 'Verified': None, 'Observation': ↪ 'You pick up the bowl 1 from the drawer 1.', 'Reward': 0 ↪ .9279943292829683 _}_ 

- _{_ 'Action': 'put bowl 1 on desk 1', 'Observation': 'You put the bowl 1 on ↪ the desk 1, under the desklamp 1.', 'Reward': 0.9279943292829683 _}_ 

Foresight 2 for the second step (composed of 5 steps, lower quality): 

- _{_ 'Action': 'go to desk 1', 'Observation': 'On the desk 1, you see a ↪ desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2 ↪ .', 'Reward': 0.7375680280039042 _}_ 

- _{_ 'Action': 'examine desklamp 1', 'Observation': 'The desklamp 1 is ↪ currently turned off.', 'Reward': 0.7375680280039042 _}_ 

- _{_ 'Action': 'use desklamp 1', 'Observation': 'You turn on the desklamp 1 ↪ .', 'Reward': 0.7375680280039042 _}_ 

- _{_ 'Action': 'go to desk 2', 'Observation': 'On the desk 2, you see a book ↪ 1, a bowl 1, a clock 1, and a stapler 1.', 'Reward': 0 ↪ .7375680280039042 _}_ 

- _{_ 'Action': 'examine bowl 1', 'Observation': 'You are now looking at the ↪ bowl 1 under the light from the desklamp 1.', 'Reward': 0 ↪ .7375680280039042 _}_ 

- _{_ 'Action': 'inventory', 'Observation': 'You have nothing in your ↪ inventory.', 'Reward': 0.7375680280039042 _}_ 

## F EXPERIMENT PROMPTS AND PARAMETERS 

- F.1 IMPLEMENTATIONS AND HYPERPARAMETERS 

**Large Language Models** Table 9 show the source of LLMs we used in our experiment. We use 1-4 A100s to launch the LLMs with vLLM. The vLLM configuration uses standard configs and use fp16 for all models. 

|Model Name|Model Code/API|
|---|---|
|GPT-3.5-Turbo (Achiam et al., 2023)|Azure api: gpt-35-turbo|
|Mistral-7b (Jiang et al., 2023)|mistralai/Mistral-7B-v0.3|
|Llama3-8b (Dubey et al., 2024)|meta-llama/Meta-Llama-3-8B-Instruct|
|Llama3.1-70b (Dubey et al., 2024)|meta-llama/Llama-3.1-70B-Instruct|
|Deepseek-Coder-6.7b (Guo et al., 2024)|deepseek-ai/deepseek-coder-6.7b-instruct|



Table 9: Model code/API of our evaluated models. 

## **Baseline Implementations** 

- **PAL** : We use the setting of first generating the entire code and then obtain the answers through a Python environment. We use the implementation from Guo et al. (2024). 

- **Beam Search** : We use the standard implementation provided by vLLM. 

- **Guided Decoding** : We use a widely used implementation from LLM-Reasoners (Hao et al., 2023). 

- **Self-Consistency** : We reimplement Self-Consistency following (Wang et al., 2022). For PAL generation, we only weight the answers from successful executed code. 

22 

Preprint 

Table 10: Hyperparameters for Predictive-Decoding main experiments. 

|Method|Model<br>Task<br>Hyperparameters|
|---|---|
|Predictive-Decoding|Llama3-8B<br>MATH<br>_α_=1_._0_,τ_ =0_._01_,K_ =8_,T_0=6<br>GSM8K<br>_α_=1_._0_,τ_ =0_._01_,K_ =8_,T_0=6<br>HumanEval<br>_α_=0_._3_,τ_ =0_._05_,K_ =8_,T_0=6<br>MBPP<br>_α_=1_._0_,τ_ =0_._1_,K_ =8_,T_0=6|
||Mistral-v0.3<br>MATH<br>_α_=1_._0_,τ_ =0_._01_,K_ =8_,T_0=6<br>GSM8K<br>_α_=1_._0_,τ_ =0_._01_,K_ =8_,T_0=6|
||Deepseek-Coder<br>HumanEval<br>_α_=0_._4_,τ_ =1_._0_,K_ =8_,T_0=6<br>MBPP<br>_α_=0_._4_,τ_ =1_._0_,K_ =8_,T_0=6|
||Llama3.1-70B<br>Alfworld<br>_α_=1_._0_,τ_ =0_._01_,K_ =8_,T_0=5<br>PDDL<br>_α_=1_._0_,τ_ =0_._01_,K_ =8_,T_0=5|
||GPT-35-Turbo (azure API)<br>Alfworld<br>_α_=0_._6_,τ_ =0_._01_,K_ =8_,T_0=5<br>PDDL<br>_α_=0_._8_,τ_ =0_._05_,K_ =8_,T_0=5|



- **Self Infilling** : We use Self Infilling implementation following Zheng et al. (2023). However this method requires suffix tokens, which is not available for Llama3. Instead, we first generate an entire code and then use logp to heuristically find the start of suffix. 

- **Act** : We use the implementation from AgentBoard (Ma et al., 2024). 

- **ReAct** : We reimplement ReAct following Yao et al. (2022). Note that we use the setting that LLM could freely choose to alternate between thinking and acting. We find this performs more stably compared to compulsory thinking at each step. 

- **MCTS** : We use the MCTS implementation from LLM-Reasoners (Hao et al., 2023). However, the original implementation for GSM8K uses subquestions, which is different from other baselines. We unify each step as COT format. 

**Hyperparameters for Main Experiments** Table 10 shows the hyperparameters used for main experiments for Predictive-Decoding. Most tasks prefer a moderate LLM generation temperature and very low selection temperature. However, we find that Deepseek-Coder prefer otherwise. This is due to Deepseek-Coder generation is more random, making accurate sampling of foresight difficult, therefore we need smaller _τ_ to prevent overfitting. The FLOPS for Predictive-Decoding is roughly _O_ ( _KT_ 0), however we use parallel accelerated implementation and the actual speed is roughtly _O_ ( _T_ 0). 

For math and coding tasks _N_ = 1 Sampling baselines (PAL, COT, Self Infilling) use _α_ = 0 _._ 6, _N_ > 1 use _α_ = 1 _._ 0 if otherwise stated. For agent tasks, Act and ReAct use temperature _α_ = 0 _._ 0. 

Beam Search uses the same beam size and best-of-n as _N_ , _N_ = 8 for math tasks and _N_ = 10 for coding tasks. The FLOPS for beamsearch is roughly _O_ ( _N_[2] ) under this implementation. 

Guided Decoding uses the standard hyperparameters from LLM-Reasoners implementation: beam size = 5, depth limit = 16, number of generations for each step = 8, beam search temperature 0.5, and reject minimum reward = 0.6. 

- F.2 PROMPTS 

## Prompt Details for Coding Tasks 

**System Prompt** Finish writing the python function. You will only write code blocks. Write # finish after the last line of the function. 

**Instruction** _{_ Instruction of the problem _}_ 

23 

Preprint 

## Prompt Details for GSM8K-COT style 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0024-02.png)


**----- Start of picture text -----**<br>
System Prompt<br>You will solve math problems following examples.<br>Examples<br>Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After<br>they are done, there will be 21 trees. How many trees did the grove workers plant today?<br>A: There are 15 trees originally. Then there were 21 trees after some more were planted. So<br>there must have been 21 - 15 = 6. The answer is 6.<br>Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the<br>parking lot?<br>A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.<br>...<br>Q:...<br>A:...<br>Instruction<br>Solve this problem following previous examples: Q:  { question }<br>**----- End of picture text -----**<br>


## Prompt Details for GSM8K-PAL style 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning_images/iclr_2025_Language_Model_Non-Myopic_Generation_for_Reasoning_and_Planning.pdf-0024-04.png)


**----- Start of picture text -----**<br>
System Prompt<br>You will write python program to solve math problems. You will only write code blocks.<br>Examples<br>Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?<br># solution in Python:<br>def s o l u t i o n ( ) :<br>””” O l i v i a has $23 . She bought f i v e bagels f o r $3 each .<br>How much money does she have l e f t ?”””<br>m o n e y i n i t i a l = 23<br>bagels = 5<br>b a g e l c o s t = 3<br>money spent = bagels * b a g e l c o s t<br>money left = m o n e y i n i t i a l −money spent<br>r e s u l t = money left<br>return r e s u l t<br>Q: ...<br># solution in Python:<br>...<br>Instruction<br>Solve this problem following previous examples: Q:  { question }<br># solution in Python:<br>def s o l u t i o n ( ) :<br>**----- End of picture text -----**<br>


## Prompt Details for MATH-PAL style 

## **System Prompt** 

You will write python program to solve math problems. You will only write code blocks. 

24 

Preprint 

## **Examples** 

Let’s write python function to solve math problems. You must return the executed result at the end of the function in float. If the final result is an expression, return it in LaTeX in simplest form. You can only write a single function. Here are some examples: Question: Find the coefficient of _x_[3] when 3( _x_[2] − _x_[3] + _x_ ) + 3( _x_ + 2 _x_[3] − 3 _x_[2] + 3 _x_[5] + _x_[3] ) − 5(1 + _x_ − 4 _x_[3] − _x_[2] ) is simplifie. solution in Python: 

**from** sympy **import** symbols , s i m p l i f y 

**def** s o l u t i o n ( ) : x = symbols ( ’x ’ ) expr = 3*( x**2 −x**3 + x ) + 3*( x + 2*x**3 −3*x**2 _\_ + 3*x**5 + x **3) −5*(1 + x −4*x**3 −x **2) s i m p l i f i e d e x p r = s i m p l i f y ( expr ) 

x3 coef = s i m p l i f i e d e x p r . a s c o e f f i c i e n t s d i c t ( ) [ x **3] r e s u l t = x3 coef **return** r e s u l t 

Q: ... solution in Python: 

... 

## **Instruction** 

Solve this problem following previous examples: Q: _{_ question _}_ solution in Python: 

**def** s o l u t i o n ( ) : 

## Prompt Details for AlfWorld 

## **System Prompt** 

You are a helpful assistant. Generate your next step of action after Action. Action must not be empty. e.g. Action: put down cup. 

## **Instruction** 

Your task is to interact with a virtual household simulator to accomplish a specific task. With each interaction, you will receive an observation. Your role is to decide on an action based on the observation. Please ensure that any objects (’obj’) and receptacles (’recep’) you mention in your response are present in the observation provided. Here are the available actions you can take: 

- take obj from recep 

- put obj in/on recep 

- open recep 

- close recep 

- toggle obj/recep 

- clean obj using recep 

- cool obj using recep 

- heat obj using recep - inventory - examine recep/obj - go to recep 

**Examples** 

25 

Preprint 

Your task is to: look at statue under the desklamp. 

You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1. Action: go to dresser 1 

Observation: On the dresser 1, you see a cellphone 3, a newspaper 2, a statue 1, and a television 1. 

Action: take statue 1 from dresser 1 Observation: You pick up the statue 1 from the dresser 1. Action: go to sidetable 1 Observation: On the sidetable 1, you see nothing. Action: go to sidetable 2 

Observation: On the sidetable 2, you see a desklamp 3, a newspaper 1, and a statue 2. Action: use desklamp 3 Observation: You turn on the desklamp 3. 

## Prompt Details for PDDL 

## **System Prompt** 

You are a master in planning. Generate your next step of action after Action. Action must not be empty. e.g. Action: put down cup. 

## **Instruction** 

The robot has four actions: pickup, putdown, stack, and unstack. The domain assumes a world where there are a set of blocks that can be stacked on top of each other, an arm that can hold one block at a time, and a table where blocks can be placed. The actions defined in this domain include: 

pickup <block>: allows the arm to pick up a block from the table if it is clear and the arm is empty. After the pickup action, the arm will be holding the block, and the block will no longer be on the table or clear. 

putdown <block>: allows the arm to put down a block on the table if it is holding a block. After the putdown action, the arm will be empty, and the block will be on the table and clear. stack <block> <block>: allows the arm to stack a block on top of another block if the arm is holding the top block and the bottom block is clear. After the stack action, the arm will be empty, the top block will be on top of the bottom block, and the bottom block will no longer be clear. 

unstack <block> <block>: allows the arm to unstack a block from on top of another block if the arm is empty and the top block is clear. After the unstack action, the arm will be holding the top block, the top block will no longer be on top of the bottom block, and the bottom block will be clear. 

## **Examples** 

Goal: The goal is to satisfy the following conditions: b1 is on b2., b2 is on b3. Observation: b1 is on the table. b2 is on the table. B3 is on the table. Robot arm is empty. The b1 is clear. The b2 is clear. The b3 is clear. 

Action: pickup b2 

Observation: b1 is on the table. B2 is on the table. The b1 is clear. The b3 is clear. You are holding b2. 

Action: stack b2 b3 

Observation: b1 is on the table. b1 is on b2. B3 is on the table. Robot arm is empty. The b1 is clear. The b2 is clear. 

Action: pickup b2. 

Observation: The action is not valid and therefore takes no effect. Please remember to satisfy the restriction of actions. You can also check valid actions. Action: check valid actions. 

26 

Preprint 

Observation: valid actions are: pickup b2, unstack b1 b2. Action: pickup b1 

Observation: b2 is on b3. B3 is on the table. Robot arm is empty. The b2 is clear. You are holding b1. Action: stack b1 b2 

Observation: b1 is on b2. b2 is on b3. B3 is on the table. Robot arm is empty. The b1 is clear. The goal is satisfied. 

27 

