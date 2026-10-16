## **Skill Set Optimization: Reinforcing Language Model Behavior via Transferable Skills** 

**Kolby Nottingham**[1] **Bodhisattwa Prasad Majumder**[* 2] **Bhavana Dalvi Mishra**[* 2] **Sameer Singh**[1] **Peter Clark**[2] **Roy Fox**[1] 

## **Abstract** 

Large language models (LLMs) have recently been used for sequential decision making in interactive environments. However, leveraging environment reward signals for continual LLM actor improvement is not straightforward. We propose Skill Set Optimization (SSO) for improving LLM actor performance through constructing and refining sets of transferable skills. SSO constructs skills by extracting common subtrajectories with high rewards and generating subgoals and instructions to represent each skill. These skills are provided to the LLM actor in-context to reinforce behaviors with high rewards. Then, SSO further refines the skill set by pruning skills that do not continue to result in high rewards. We evaluate our method in the classic videogame NetHack and the text environment ScienceWorld to demonstrate SSO’s ability to optimize a set of skills and perform in-context policy improvement. SSO outperforms baselines by 40% in our custom NetHack task and outperforms the previous state-of-the-art in ScienceWorld by 35%. 

## **1. Introduction** 

Large Language Model (LLM) actors have been deployed in interactive domains such as robotics (Ichter et al., 2022; Huang et al., 2022a;b), games (Nottingham et al., 2023a;b), and programming (Chen et al., 2022). Similar to the reinforcement learning (RL) setting, these domains often provide a reinforcement signal in the form of reward, task success, or user feedback. For example, the task in Figure 1 may 

> 1Department of Computer Science, University of California Irvine, Irvine CA, United States[2] Allen Institute for AI, Seattle Washington, United States. * Equal contribution. Correspondence to: Kolby Nottingham _<_ knotting@uci.edu _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0001-09.png)


**----- Start of picture text -----**<br>
Task: Measure the melting temp of chocolate.<br>State: You see chocolate and a stove.<br>LLM Actor move chocolate to stove<br>State: You see a stove with chocolate.<br>LLM Actor activate stove<br>State: You see a stove with melted chocolate.<br>LLM Actor use thermometer on chocolate<br>State: The thermometer reads 40 C.<br>SSO<br>Task: Measure the melting Melt substance:<br>temp of water. 1. Move substance<br>to stove<br>State: You see a stove<br>and  a freezer with ice. 2. Activate stove<br>LLM Actor LLM Actor<br>use thermometer move ice to stove<br>**----- End of picture text -----**<br>


Figure 1: Example of a interactive text task and skill. 

provide a success signal upon measuring the substance’s temperature and intermediate rewards for completing subgoals such as activating the stove. However, finetuning an LLM actor directly using a traditional RL policy gradient is often impractical with contemporary LLMs and impossible with black-box closed-source LLMs. Instead, we explore a new paradigm of in-context policy improvement. 

In natural language processing (NLP) tasks, in-context learning improves task performance by editing LLM inputs with instructions (Brown et al., 2020), task examples (Wei et al., 2021), or auxiliary tasks (Wei et al., 2022). However, interactive domains require sequential decision making with long trajectories and complex credit assignment (see Section 3), so naively applying in-context learning techniques generalizes poorly and does not scale well. Instead, recent approaches for improving LLM actors construct a “memory” of knowledge about the world (Majumder et al., 2023), skills (Wang et al., 2023a), or task insights (Shinn et al., 2023) to use in-context for policy improvement. 

1 

**Skill Set Optimization** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0002-01.png)


**----- Start of picture text -----**<br>
SSO Iteration Extract<br>Environment Interaction History Trajectory Executed<br>Skill<br>State<br>Skill Set Subtrajs<br>Discounted Observed<br>Reward Value<br>Retrieval<br>Score Reward 0.87 + w1 +<br>Skill LLM Actor Action Subtrajs Similarity 0.98 + w2 0.88<br>Length 0.12 + w3 ObservedValue<br>Skill Set Sample<br>Beam<br>Trajectory Skill Set Subtrajs Search Subtrajs <= Ɛ Filter<br>Construction<br>> Ɛ<br>Generate<br>Executed Skill Set Skill<br>Skills Refinement Subtrajs LLM Subgoal:1. Instruction Skill<br>2. Instruction<br>Skill Set Refinement<br>Skill Set Construction<br>**----- End of picture text -----**<br>


Figure 2: Each iteration of SSO collects a trajectory of interactions with the current LLM actor, uses this trajectory to construct new skills and filter poorly performing skills, and updates the skill set for use in the next iteration. New skills are constructed by extracting, scoring, and sampling sets of similar subtrajectories that are then used to generate subgoals and instructions for skills. Skills are filtered based on the discounted future rewards observed after executing a skill. 

Current approaches to constructing “memory” for LLM actors have shortcomings such as a lack of continual memory evaluation, intermediate subgoals, and memory retrieval. To address these shortcomings, we propose **Skill Set Optimization** ( **SSO** )[1] , a method for automatically constructing skills for in-context policy improvement, where a skill is composed of an initial state, a list of language instructions, and a language subgoal. SSO takes inspiration from both in-context learning and policy optimization to optimize a set of skills for in-context policy improvement. Figure 2 shows how skills are constructed from past trajectories by extracting similar subtrajectories, scoring and sampling sets of subtrajectories that do not overlap, and finally generating an abstract subgoal and instructions for each set of subtrajectories. The constructed skill set is later refined by filtering skills that do not lead to high future rewards when executed. By prioritizing skills according to environment reward at the construction and refinement steps, we identify subgoals that have high impact on policy improvement. Additionally, the subtrajectories’ initial states provide a reference for skill retrieval allowing SSO to retrieve skills that are immediately relevant. Finally, because each skill subgoal and instructions are generated from multiple subtrajectories, the resulting subgoals and instructions are often abstract (as in Figure 1) and further facilitate task transfer. 

We empirically demonstrate the advantage of using skills for in-context policy improvement and evaluate SSO’s ability 

> 1https://allenai.github.io/sso/ 

to rapidly adapt and transfer skill sets in two sequential decision making domains. First, we evaluate SSO on the text-based environment ScienceWorld (Wang et al., 2022) and on the grid-based game NetHack (Kuttler et al.¨ , 2020). SSO achieves state-of-the-art performance on ScienceWorld, outperforming the previously top performing CLIN agent (Majumder et al., 2023) by an average of 35%. SSO also outperforms baselines on our custom NetHack task by 40% despite its low-level action space. Our analysis and ablations show that SSO continually optimizes an LLM actor’s policy by extracting increasingly helpful skills that maximize the task’s reinforcement signal. 

## **2. Related Work** 

## **2.1. In-Context Learning** 

Finetuning state-of-the-art LLMs can be prohibitively expensive. In-context learning utilizes an LLM’s ability to recognize patterns by augmenting prompts to modify LLM behavior (Brown et al., 2020; Wei et al., 2021; 2022). Instruction tuning facilitates this by finetuning LLMs to follow custom instructions (Brown et al., 2020). Previous work has attempted to learn prompts that maximize performance on a dataset (Deng et al., 2022; Zhang et al., 2023; Fernando et al., 2023). However, to maintain interpretability, these methods typically merely edit existing prompts. Also, they require constant evaluation on a downstream task which is often impractical for sequential tasks with delayed rewards. 

2 

**Skill Set Optimization** 

## **2.2. LLMs for Sequential Decision Making** 

LLMs are popular tools for planning and high-level decision making in robotic applications (Ichter et al., 2022; Huang et al., 2022a;b) games (Nottingham et al., 2023a;b), programming (Chen et al., 2022), and computer tasks (Kim et al., 2023; Liang et al., 2023). Previous work focuses on improved prompting and evaluation methods for LLM actors without considering approaches for continual learning. This is partly because continual learning for LLMs via finetuning is very expensive and often impractical, especially in sequential decision making domains with long contexts and noisy credit assignment. However, in this paper, we leverage in-context learning for sequential decision making to develop a powerful method for quickly adapting to and generalizing between tasks. 

## **2.3. LLMs with Environment Feedback** 

Some recent work has leveraged task success signals from an environment for in-context policy improvement. The Reflexion (Shinn et al., 2023), Voyager (Wang et al., 2023a), DEPS (Wang et al., 2023b), and ADAPT (Prasad et al., 2023) agents attempt to retry tasks or subgoals after a failure. These methods work by prompting an LLM to reflect on the failed attempt and suggest improvements to make on the next attempt. However, none of these methods, with the exception of Voyager (see below), leverage successful task completions for learning and are not compatible with task transfer or generalization. 

This paper focuses on the problem of building a memorylike collection of helpful information for adaptation to new tasks and transfer between tasks. Previous work that has pursued this research direction include the Voyager (Wang et al., 2023a), ExpeL (Zhao et al., 2023), and CLIN (Majumder et al., 2023) agents. Voyager generates javascript skills, self-corrects mistakes with a reflexion-like process, and then stores all successful skills in-context. Similar to SSO, Voyager’s skills store instructions for reaching a subgoal. However, Voyager’s skills are executed as code and explore future subgoals that are yet to be reached rather than extracting subgoals from past experience to maximize reward. ExpeL prompts an LLM to generate or edit freeform insights based on successful trajectories. Unlike SSO, ExpeL does not leverage partially successful trajectories when generating insights. CLIN generates a memory of insights in the form “action X [may/should] be necessary to do action Y” by prompting an LLM with past trajectories. Like SSO, CLIN does not require a past trajectory to be successful in learning useful information, but CLIN does not take task performance or policy improvement into account when constructing its memory. Additionally, no previous method continually evaluates memories based on environment feedback like in SSO’s skill refinement step. 

Most of the methods for in-context policy improvement in this section are complementary. For example, the ExpeL agent utilizes reflexion and fewshot trajectories in addition to its memory of insights. However, for the sake of making a direct comparison, our agent only uses SSO for policy improvement, and we choose to compare to one self-correcting method (Reflexion), one insight method (CLIN), and fewshot example trajectories. 

## **3. In-Context Policy Improvement** 

With the increased dominance of LLMs in NLP, in-context learning has become an essential tool for improving performance on NLP tasks. Rather than doing expensive finetuning on downstream tasks, in-context learning enables significant increases in task performance just by adding supplementary inputs to the LLM. In-context changes to an LLM actor may be an efficient way to make continual changes to the actor’s policy. However, previous work has done little to explore effective methods of leveraging environment feedback for LLM actor in-context learning. 

Unlike other NLP tasks, sequential decision making in an interactive environment requires multiple outputs from an LLM actor in the correct sequence to generate a successful trajectory. Sequential decision making is often represented as a Markov Decision Process (MDP) with states _s ∈S_ , actions _a ∈A_ , a transition propability function _T_ : _S × A × S[′] →_ [0 _,_ 1], and a reward function _R_ : _S × A →_ R. In the case of an LLM actor, environment states are the text inputs to the LLM, and the output of the LLM is executed as an action in the environment. The objective of an LLM actor is to model an optimal policy _π_ ( _a|s_ ) that maximizes rewards in a trajectory _τ_ = _s_ 0 _, a_ 0 _, ..., sT , aT_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0003-10.png)


To avoid traditional gradient-based policy optimization techniques, which can be computationally prohibitive with LLMs or impossible with black-box LLMs, we assume that the policy _π_ is parameterized by text inputs _θ_ that are provided to the agent alongside the state _s_ . A successful in-context policy improvement method will identify a _θ_ that increases the objective _J_ ( _πθ_ ). 

A straightforward adaptation of in-context learning for incontext policy improvement would be to provide an LLM actor with examples of successful trajectories, thus providing information to the LLM actor about the optimal policy. However, this approach does not scale well as it results in long context lengths and may include redundant information or information that does not transfer well between tasks. Instead, we develop a scalable and transferable mem- 

3 

**Skill Set Optimization** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Melting Temp NetHack<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>None Fewshot Skills None Fewshot Skills<br>Score<br>**----- End of picture text -----**<br>


Figure 3: Comparison between in-context skills, fewshot trajectory examples, and no in-context information on the Melting Temperature ScienceWorld and NetHack tasks. 

ory structure for in-context policy improvement that we call skills. Skills leverage the sequential information from trajectory examples while keeping information brief and modular. A skill is composed of an initial state for used for retrieval, a final state used as a natural language subgoal, and a sequence of actions used as natural language instructions for reaching that subgoal. 

To demonstrate the difficultly of using fewshot trajectory examples for in-context policy improvement, we compare the use of fewshot trajectories vs. skills on ScienceWorld (Wang et al., 2022) and NetHack (Kuttler et al.¨ , 2020) domains (see Section 5). For the fewshot LLM actor, we gather 30 trajectories using an LLM actor and select the three best trajectories to provide in-context at each step. For the skill-based LLM actor, we use the same 30 trajectories to create skills using SSO, and we retrieve three skills to provide in-context at each step. Figure 3 shows how well skills perform vs. fewshot examples for in-context policy improvement. This is especially the case for the NetHack domain where trajectories are longer and actions are expressed characters instead of language actions. Also, the context for the fewshot actor was on average 5x longer than the context for the skill-based actor. The following section ellaborates on how we use SSO to learn and optimize a set of skills for continual in-context policy improvement. 

## **4. Skill Set Optimization** 

We propose optimizing LLM actors by providing transferable skills in-context. First, we construct new skills for our skill set from subtrajectories with high rewards to reinforce successful behaviors. Second, we further refine the constructed skill set by filtering skills that do not result in high rewards when used in subsequent trajectories. When using skills in-context, we retrieve relevant skills to include in context based on cosine similarity of skill initial states and the current environment state. Each iteration of SSO includes 

rolling out a single trajectory with the current LLM actor and skills, constructing new skills, and filtering out skills that did not result in positive rewards in the last trajectory. The updated skill set is then used by the LLM actor in the following iteration. This process is illustrated in Figure 2 and described in the following sections. 

## **4.1. Skill Set Construction** 

A skill is expressed to the actor as an abstract LLMgenerated subgoal and list of instructions for reaching that subgoal. We define a unique skill using one or more subtrajectories where the final states in each of the subtrajectories are used to generate the subgoal and the actions in each subtrajectory are used to generate the instructions for reaching that subgoal. Given the LLM actor’s previous trajectories, we **extract** potential subtrajectories, **score** them using several heuristics, **sample** a skill set using beam search, and **generate** subgoals and instructions for each skill. 

## 4.1.1. EXTRACT 

To identify transferable skills, we extract multiple similar subtrajectories to be used to generate each skill. The subtrajectories of a skill must be similar enough that a common abstract subgoal and instructions can be generated. We estimate the similarity of two subtrajectories using the average cosine similarity of each of their state and action embeddings. Using multiple subtrajectories for each skill has two important benefits: (1) the resulting skills are abstract and more transferable, and (2) repeated similar subtrajectories is a strong signal for identifying useful subgoals. 

Enumerating every possible set of subtrajectories from an LLM actor’s experience would be infeasible, so we only consider pairs of subtrajectories of certain lengths ([2,5] in our experiments) that come from different trajectories. After each completed trajectory, we enumerate its subtrajectories and extract the most similar subtrajectory from each of the _N_ previous completed trajectories, as shown in Algorithm 1. Each of these subtrajectory pairs are considered for potential skills, although most would not result in a useful skill. 

## 4.1.2. SCORE 

The majority of extracted subtrajectory pairs must be removed from canidacy for the skill set. Pairs should be similar enough to have common subgoal and instructions, lead to high rewards, and have high coverage of past experience. To accomplish this, we calculate the average state and action similarity, discounted future reward, and length of each pair. We compute a score using a weighted sum of these values to identify which subtrajectory pairs would make useful skills. In our experiments, we set score weights to _w_ 1 = 1, _w_ 2 = 0 _._ 1, and _w_ 3 = 0 _._ 01 to prioritize first similarity, then reward, and finally length. 

4 

**Skill Set Optimization** 

## **Algorithm 1** Extract 

**Require:** _D // N past trajectories_ **Require:** _τ // latest trajectory_ **Require:** _min, max // min and max skill length SS ←∅ // skill set // iterate subtrajectories of length [min,max]_ **for** _τsub ∈ GetAllSubtraj_ ( _τ, min, max_ ) **do for** _τ_ ˆ _∈D_ **do** _// get the most similar subtrajectory of same length τ_ ˆ _sub ← SimilarSubtraj_ ( _τsub,_ ˆ _τ_ ) _SS ← SS ∪{_ ( _τsub,_ ˆ _τsub_ ) _}_ **end for end for** 

## 4.1.3. SAMPLE 

We utilize subtrajectory pair scores to conduct a beamsearch over pairs to maximize the sum of scores. Many of the extracted potential skills have significant overlap with each other, so we impose a constraint on the beamsearch that there can be no overlap of skill subtrajectories in a skill set. We also include all subtrajectory pairs in the search that were sampled during previous iterations of SSO. The final sampled skill set includes unique skills that prioritize similar, high reward, multi-step subtrajectory pairs. 

## 4.1.4. GENERATE 

Finally, the subtrajectories in each pair are summarized into a single subgoal and list of instructions. We prompt the actor LLM with the pair of subtrajectories and ask it to generate “a numbered list of instructions for completing the skill” and “a single target observation that would indicate the success of the skill”. We also ask the LLM to remove any semantically identical skills that may remain. Note that this is the only step in SSO that utilizes a generative-LLM. We found that depending on generative-LLMs for extraction and scoring was less reliable in addition to increasing costs. 

## **4.2. Skill Set Refinement** 

The previously described skill set construction prioritizes including transferable high-reward skills the skill set. Experimentally, including constructed skills in context improves action accuracy across each subgoal in Table 1. We further propose the following method for refining the skill set by evaluating environment rewards after executing a skill. 

To evaluate skill performance, we record the discounted future return after executing a particular skill. Since it is not straightforward to identify which skill is being executed at each step, we ask the LLM actor to self-report when it is executing a skill with the prompt “ _output which of the given subgoals you are targeting next and then output the next_ 

_action to execute_ ”. Table 1 reports correctly self-reports skill execution about 70% of the time. Each time the LLM actor self-reports using a skill, we compute the discounted sum of future rewards and add this to the skill’s “observed value”. If a skill’s observed value is ever below or equal to a threshold _ϵ_ (zero in our experiments), we filter that skill out of the skill set. This process refines our skill set to include skills that demonstrably result in high rewards. Example code for self-refinement can be found in Appendix C. 

|**Subgoal**|**w/o Skill w/ Skill**|**Self-Reporting**|
|---|---|---|
||||
|Fill container<br>Heat substance<br>Mix ingredients|0.07<br>0.37<br>0.04<br>0.22<br>0.30<br>0.36|0.69<br>0.68<br>0.71|



Table 1: Action accuracy from example trajectories with and without learned skills in-context, and LLM actor success rate self-reporting that it is executing the provided skill. 

## **5. Experimental Setup** 

## **5.1. Science World** 

We evaluate SSO’s ability to quickly adapt to and transfer knowledge between tasks in the ScienceWorld domain (Wang et al., 2022). ScienceWorld is a text-based simulator that tests common sense and reasoning. It is organized into various tasks with train and test variants of each. For each of the 18 task classes listed in Figure 4, we test SSO and baselines on 7 to 10 test variants. For example, the _Melting Temp_ task requires the LLM actor to measure the melting temperature of a substance, but what that substance is and the environment setup will be different in each variation. 

We evaluate on two training modes: adaptation and transfer. When evaluating adaptation, we allow 5 attempts on each test variant with the ability to learn between each trial. When evaluating transfer, we train the LLM actor on 10 training variants for 30 episodes (3 trajectories each). After training, we evaluate on the same test variants that we used to evaluate adaptation. The environment provides intermediate rewards for completing subtasks. A successfully completed task will have a final cumulative reward, or score, of 100. 

We compare with the following GPT-4 based methods: 

**ReAct** (Yao et al., 2022) prompts an LLM actor to reason about the task before outputting an action. All other methods also incorporate this in their prompts. 

**Reflexion** (Shinn et al., 2023) prompts an LLM actor to reflect on failed task attempts and then retry the task. 

**CLIN** (Majumder et al., 2023) reflects on past experience to learn transition information of the form “ _A_ [may/should] be necessary to _A_ ”. 

5 

**Skill Set Optimization** 

|ScienceWorld||Adaptation|Adaptation||Transfer|Transfer||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Task|ReAct|Refexion|CLIN|SSO|CLIN|SSO||||||||||||||||||
|Temperature|7.2|5.9|14.3|**100**|15.7|**71.6**||||||||||||||||||
|Melting Temp|6.1|28.6|51.8|**97.3**|49.7|**69.2**||||||||||||||||||
|Find Plant|26.7|64.9|**100**|**100**|59.2|**100**||||||||||||||||||
|Find Living|53.3|16.4|**100**|96.7|**100**|90||||||||||||||||||
|Chemistry|51|70.4|44.4|**82.6**|42.2|**48**||||||||||||||||||
|Color Mixing|58.9|70.7|56.7|**81.1**|**85.6**|71.1||||||||||||||||||
|Lifespan, Longest|61|**100**|**100**|**100**|65|**90**|||||||||NetHack|||||||||
|Lifespan, Shortest<br>Life Stages, Plant<br>Life Stages, Animal<br>Boil<br>Freeze<br>Grow Plant<br>Grow Fruit|67.5<br>8<br>27.7<br>3.5<br>7.8<br>9.1<br>18.6|84.4<br>**8**<br>2.6<br>4.2<br>7.8<br>7.3<br>13|90<br>**8**<br>81<br>15.2<br>10<br>11<br>71.6|**100**<br>6.2<br>**100**<br>**81.7**<br>**74.3**<br>**86.6**<br>**78**|75<br>**32**<br>42.8<br>4.4<br>8.9<br>10.9<br>**70.8**|**80**<br>3.4<br>**77**<br>**48.7**<br>**38.9**<br>**61.2**<br>28.3|Task Score|40<br>45<br>50<br>55|||SSO<br>Reflexion<br>ReAct|||||||||||||
|Gravity<br>Friction|40.5<br>44|50.6<br>**100**|**100**<br>72.5|**100**<br>94|70<br>**70**|**74**<br>67.5||35||||||||||||||||
|Genetics, Known|25.7|50.9|**100**|78.5|**84.5**|42.5||30||||||||||||||||
|Genetics, Unknown|16.8|23.7|**92.6**|48.7|**61.4**|20.3|||0||5||10||15||20||25||30|||
|Average|29.6|39.4|62.2|**83.7**|52.7|**60.1**|||||||||Iteration|||||||||



Figure 4: We compare SSO with ReAct and Reflexion baselines in ScienceWorld and NetHack domains. We also compare with the previous state-of-the-art for ScienceWorld, CLIN. In ScienceWorld we evaluate adaptation—attempting a single variant five times—and transfer—learning on 10 train variants for 30 iterations before testing on the heldout test variants. In NetHack we test task adaptation across 30 iterations. 

## **5.2. NetHack** 

NetHack is a grid-based videogame that requires challenging exploration and problem solving (Kuttler et al.¨ , 2020). Unlike ScienceWorld, NetHack requires low-level navigation actions. We choose to include NetHack in our evaluation because it poses a potential challenge for our method. SSO requires aligning common sequences of states and actions when extracting skills. However, in environments with lower-level actions such as NetHack, dissimilar action sequences can be used to achieve the same subgoal. Additionally, NetHack uses character-based actions instead of natural language (“k” moves the player north and “,” picks up an item). Despite this, SSO is able to successfully learn helpful skills in NetHack. 

We utilize the MiniHack library (Samvelyan et al., 2021) to design a custom level that tests an LLM actor’s ability to explore and learn several skills to complete a task. Figure 4 shows the layout of our custom task. The LLM actor must pick up and use a key to unlock a door, pick up and use an item to begin levitating, and safely cross the lava to the goal. The task provides intermediate rewards for achieving each subgoal. As in ScienceWorld, task completion results in a score of 100. Starting locations of the actor, items, and staircase are randomized in each episode so that SSO cannot learn memorized paths between subgoals. Also, the item that is used to levitate may be a potion or a magic ring, each of which requires different actions to activate. 

In our custom NetHack task, we compare SSO with ReAct and Reflexion baselines. We omit CLIN for this task because CLIN was specifically designed for ScienceWorld and reimplementing CLIN for a new domain would be non-trivial. Similar to adaptation on ScienceWorld, we allow Reflexion and SSO to adapt to our NetHack task. However, unlike the adaptation setup in ScienceWorld, we run these methods for 30 iterations and evaluate every 10 iterations by attempting the task 10 times with a frozen set of skills/reflections. All LLM actors for NetHack utilize GPT-4-Turbo instead of GPT-4 to save on costs, but we found performance between the two LLMs to be similar. 

## **6. Experimental Results** 

In both ScienceWorld and NetHack domains, SSO provides large performance gains by informing the LLM actor about potential subgoals and how to accomplish them. Figure 4 reports SSO’s state-of-the-art performance for both task adaptation and transfer in ScienceWorld as well as SSO’s superior performance and continual learning ability when compared with ReAct and Reflexion baselines in NetHack. Error bars for NetHack show standard error across five training runs. We do not provide standard error for ScienceWorld because the large number of tasks and expensive GPT-4 actor limited our computational budget. In the following sections, we analyze what makes SSO successful. 

6 

**Skill Set Optimization** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Skill Statistics<br>30<br>70<br>4<br>60<br>20<br>3 50<br>10<br>40<br>2<br>30<br>0 5 10 15 20 0 5 10 15 20 0 5 10 15 20<br>Iteration Iteration Iteration<br>ScienceWorld NetHack<br>Task Score<br>Skill Set Size<br>Executed Skills<br>**----- End of picture text -----**<br>


Figure 5: Skill Set statistics for ScienceWorld and NetHack during training. _Skill set size_ measures the number of skills created minus those that were pruned during refinement. _Executed skills_ is the average number of unique skills executed in a trajectory as reported by the LLM actor. Finally, the average _task score_ is reported throughout training. 

## **6.1. Skill Analysis** 

Figure 5 reports several skill statistics throughout SSO training. SSO continuously increases the size of its skill set during training, and the rate of constructing new skills decreases as the skill set increases in experience coverage. In our experiments, we retrieve up to three skills to include in-context at each step, but much fewer than this are selfreported as “executed” during the trajectory. Despite this, our experience suggests that even unreported skills improve LLM actor performance. Also, the number of skills being executed per trajectory steadily increases, suggesting that learned skills become more useful throughout training. Finally, the subtrajectories used to create skills have an average length of 2.64 and 2.95 for ScienceWorld and NetHack respectively. Remember that we limit subtrajectory length to [2, 5] in our experiments. The length of generated instructions is slightly longer at 3.18 and 3.27 respectively. 

## **ScienceWorld Melting Temp Task** 

Subgoal: The stove is turned on. on the stove is: a substance called liquid [substance]. 1. Focus on the thermometer 2. Focus on the substance you want to heat 3. Move the focused substance to the stove 4. Activate the stove 

## **NetHack Task** 

Subgoal: You succeed in unlocking the door. 1. Stand adjacent to the closed door that needs to be unlocked 2. Use the action ’a’ to apply the relevant key or tool that can unlock the door 3. Confirm the unlock action by responding affirmatively when prompted, typically by using the action ’y’ 

Table 2: Example generated skills. 

Qualitatively, we found that SSO was especially helpful with less intuitive aspects of the action space. For example, the ScienceWorld Melting Temperature task requires using the _focus_ action on the thermometer and substance before attempting to melt it, and NetHack requires using the _apply_ action on a key to unlock a nearby door. Humans can quickly adapt to unintuitive domain specific requirements after a few tries, but we found that the unaltered LLM actors often got stuck in situations where actions were unintuitive outputting text such as “I apologize for the confusion. I am trying to...” before attempting the invalid action again. However, skills generated by SSO, such as those in Table 2, allow the actor to reliably pass bottlenecks and continue to explore the next steps of the task. 

We hypothesize that SSO continually optimizes the LLM actor’s policy by iteratively creating better skills and gathering better data. We look for evidence of this by visualizing the skill lifecycle in Figure 6. Each row of the figure represents the skills that were created in the corresponding iteration of SSO and when those skills were executed during training. The densities in the figure are normalized by the total number of skills, so the the density width indicates both how many skills were created in that iteration and in which iterations they were executed. 

Figure 6 shows that skill set refinement prunes most skills shortly after they are created. Also, while skills from the initial iterations are used for longer, the LLM actor tends to use skills that were created more recently, suggesting that more useful skills are discovered later in training. This is further supported by visualizing when certain actions first appear in skills, as shown by the icons in Figure 6. Note that the icons indicate when a skill first appears, but improved versions of that same skill may be created in later iterations. In general, actions that are required later in a trajectory are included in skills later in training. 

7 

**Skill Set Optimization** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0008-01.png)


**----- Start of picture text -----**<br>
ScienceWorld NetHack<br>1<br>3<br>5<br>7<br>9<br>11<br>13<br>15 "go to kitchen"<br>17 first appears in a "pick up key" first<br>19 Melting Temp skill appears in a skill<br>21 "activate stove"<br>23 first appears in a "open door" first<br>25 Melting Temp skill appears in a skill<br>27<br>29<br>5 10 15 20 25 30 5 10 15 20 25 30<br>Executed Iteration Executed Iteration<br>Created Iteration<br>**----- End of picture text -----**<br>


Figure 6: Visualization of the lifecycle of SSO skills. For every skill created in an iteration of SSO, the corresponding row shows when that iteration’s skills were executed throughout training. Most skills are pruned soon after creation and replaced with improved skills. The icons in the figure indicate when skills with the corresponding actions were first created. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/behavior/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills_images/icml_2024_Skill_Set_Optimization_Reinforcing_Language_Model_Behavior_via_Transferable_Skills.pdf-0008-03.png)


**----- Start of picture text -----**<br>
Melting Temp NetHack<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>ReAct SSOw/o Refinew/o Rewardw/o Similarity ReAct SSOw/o Refinew/o Rewardw/o Similarity<br>Score<br>**----- End of picture text -----**<br>


Figure 7: SSO ablations with and without refinement, reward weight during scoring, and similarity-based extraction. 

## **6.2. Ablations** 

We compare SSO to several ablations on ScienceWorld’s Melting Temperature task and the NetHack task. First, we ablate the skill refinement method described in Section 4.2 and do not ever prune the skill set. Next, we ablate the use of task reward in scoring skills for sampling by reducing the corresponding score weight to zero. Finally, we ablate similarity-based extraction and instead use the three steps before every environment reward to generate skills. 

Figure 7 shows that each of these ablations decreases the performance of SSO but still regularly outperform the baseline. Skill set refinement appears to be slightly more impactful than environment rewards and extracting pairs of similar subtrajectories. Using similarity as an extraction method appears less impactful than simply using environment rewards. However, including similarity-based extraction still contributes to SSO by automatically determining the start and end states for high-reward subtrajectories. 

## **7. Discussion & Conclusion** 

Skill Set Optimization (SSO) is a new in-context policy optimization method that allows LLM actors to quickly adapt to and transfer between tasks. SSO achieves state-of-the-art results in the ScienceWorld domain and outperforms all experimented baselines on our custom NetHack task. Every iteration, SSO constructs commonly executed high-reward skills to add to the skill set and filters out poorly performing skills. Unlike previous work, SSO utilizes ongoing environment feedback to evaluate and prune skills, leverages intermediate rewards to identify subgoals, facilitates skill retrieval, and learns abstract transferable skills. 

One limitation of SSO is the current similarity metric used for extracting potential skills. State and action embedding similarity may be less effective in environments with distracting state information or low-level actions. Also, although SSO is capable of operating in environments without intermediate rewards, for best performance task designers must include intermediate subgoal rewards in the environment feedback. Fortunately, this is a common paradigm for sequential decision making. Finally, SSO on its own does not include a method for leveraging negative environment feedback outside skill set refinement, but a method such as Reflexion (Shinn et al., 2023) can easily be used in addition to SSO to provide this feature. We provide these limitations as potential inspiration for future work. 

We believe that SSO takes a significant step towards reliable in-context policy optimization methods for LLM actors. SSO manages this while being a general purpose solution, not limited by specific output formats (Wang et al., 2023a), and capable of operating in scenarios with (Zhao et al., 2023) or without (Majumder et al., 2023) environment reinforcement signals. We hope SSO inspires continued research in improving the effectiveness and learning efficiency of continual learning for LLM actors. 

8 

**Skill Set Optimization** 

## **Impact Statement** 

This work aims to improve the ability of deployed LLMs to continually adapt and transfer knowledge between tasks without retraining. This research direction has the potential to improve many state-of-the-art LLM applications, but does not alter any of their fundamental limitations. 

## **Acknowledgements** 

We would like to thank the Aristo team at the Allen Institute for Artificial Intelligence for their discussions and feedback. This work was funded in part by the DARPA ANSR program under award FA8750-23-2-0004. 

## **References** 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/ 

- 1457c0d6bfcb4967418bfb8ac142f64a-Paper. pdf. 

- Chen, B., Zhang, F., Nguyen, A., Zan, D., Lin, Z., Lou, J.-G., and Chen, W. Codet: Code generation with generated tests. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Deng, M., Wang, J., Hsieh, C.-P., Wang, Y., Guo, H., Shu, T., Song, M., Xing, E., and Hu, Z. RLPrompt: Optimizing discrete text prompts with reinforcement learning. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 3369–3391, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL https:// aclanthology.org/2022.emnlp-main.222. 

- Fernando, C., Banarse, D., Michalewski, H., Osindero, S., and Rocktaschel, T.¨ Promptbreeder: Self-referential self-improvement via prompt evolution. _arXiv preprint arXiv:2309.16797_ , 2023. 

- Huang, W., Abbeel, P., Pathak, D., and Mordatch, I. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In _Interna-_ 

_tional Conference on Machine Learning_ , pp. 9118–9147. PMLR, 2022a. 

- Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al. Inner monologue: Embodied reasoning through planning with language models. In _6th Annual Conference on Robot Learning_ , 2022b. 

- Ichter, B., Brohan, A., Chebotar, Y., Finn, C., Hausman, K., Herzog, A., Ho, D., Ibarz, J., Irpan, A., Jang, E., Julian, R., Kalashnikov, D., Levine, S., Lu, Y., Parada, C., Rao, K., Sermanet, P., Toshev, A. T., Vanhoucke, V., Xia, F., Xiao, T., Xu, P., Yan, M., Brown, N., Ahn, M., Cortes, O., Sievers, N., Tan, C., Xu, S., Reyes, D., Rettinghouse, J., Quiambao, J., Pastor, P., Luu, L., Lee, K.-H., Kuang, Y., Jesmonth, S., Jeffrey, K., Ruano, R. J., Hsu, J., Gopalakrishnan, K., David, B., Zeng, A., and Fu, C. K. Do as i can, not as i say: Grounding language in robotic affordances. In _6th Annual Conference on Robot Learning_ , 2022. URL https://openreview.net/ forum?id=bdHkMjBJG_w. 

- Kim, G., Baldi, P., and McAleer, S. Language models can solve computer tasks. _arXiv preprint arXiv:2303.17491_ , 2023. 

- Kuttler, H., Nardelli, N., Miller, A., Raileanu, R., Selvatici,¨ M., Grefenstette, E., and Rocktaschel,¨ T. The nethack learning environment. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 7671–7684. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/ 569ff987c643b4bedf504efda8f786c2-Paper. pdf. 

- Liang, Y., Wu, C., Song, T., Wu, W., Xia, Y., Liu, Y., Ou, Y., Lu, S., Ji, L., Mao, S., et al. Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis. _arXiv preprint arXiv:2303.16434_ , 2023. 

- Majumder, B. P., Mishra, B. D., Jansen, P., Tafjord, O., Tandon, N., Zhang, L., Callison-Burch, C., and Clark, P. Clin: A continually learning language agent for rapid task adaptation and generalization. _arXiv preprint arXiv:2310.10134_ , 2023. 

- Nottingham, K., Ammanabrolu, P., Suhr, A., Choi, Y., Hajishirzi, H., Singh, S., and Fox, R. Do embodied agents dream of pixelated sheep? embodied decision making using language guided world modelling. In _Proceedings of the 40th International Conference on Machine Learning_ , 2023a. URL https://proceedings.mlr.press/v202/ nottingham23a/nottingham23a.pdf. 

9 

**Skill Set Optimization** 

- Nottingham, K., Razeghi, Y., Kim, K., Lanier, J., Baldi, P., Fox, R., and Singh, S. Selective perception: Learning concise state descriptions for language model actors. In _NeurIPS 2023 Foundation Models for Decision Making Workshop_ , 2023b. URL https://openreview. net/pdf?id=siFopuPuCS. 

- Prasad, A., Koller, A., Hartmann, M., Clark, P., Sabharwal, A., Bansal, M., and Khot, T. Adapt: As-needed decomposition and planning with language models. _arXiv preprint arXiv:2311.05772_ , 2023. 

   - Zhang, T., Wang, X., Zhou, D., Schuurmans, D., and Gonzalez, J. E. Tempera: Test-time prompt editing via reinforcement learning. In _The Eleventh International Conference on Learning Representations_ , 2023. 

   - Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., and Huang, G. Expel: Llm agents are experiential learners. _arXiv preprint arXiv:2308.10144_ , 2023. 

- Samvelyan, M., Kirk, R., Kurin, V., Parker-Holder, J., Jiang, M., Hambro, E., Petroni, F., Kuttler, H., Grefenstette, E., and Rocktaschel,¨ T. Minihack the planet: A sandbox for open-ended reinforcement learning research. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)_ , 2021. 

- Shinn, N., Labash, B., and Gopinath, A. Reflexion: an autonomous agent with dynamic memory and self-reflection. _arXiv preprint arXiv:2303.11366_ , 2023. 

- Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models. _arXiv preprint arXiv:2305.16291_ , 2023a. 

- Wang, R., Jansen, P., Cotˆ e,´ M.-A., and Ammanabrolu, P. Scienceworld: Is your agent smarter than a 5th grader? In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 11279– 11298, 2022. 

- Wang, Z., Cai, S., Chen, G., Liu, A., Ma, X., and Liang, Y. Describe, explain, plan and select: interactive planning with llms enables open-world multi-task agents. In _Thirtyseventh Conference on Neural Information Processing Systems_ , 2023b. 

- Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In _International Conference on Learning Representations_ , 2021. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022. 

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In _The Eleventh International Conference on Learning Representations_ , 2022. 

10 

**Skill Set Optimization** 

## **A. Models and Hyperparameters** 

All of our ScienceWorld experiments were completed with OpenAI’s gpt-4-0613, and our NetHack experiments were completed with gpt-4-1106-preview. We used text-embedding-ada-002 as the embedding model for all similarity metrics. Table 3 shows the hyperparameters we used in all of our experiments. We found that these hyperparameters we robust and worked well in all of our experiments on both domains. However, we expect that the parameters regulating skill length and number of retrieved skills may need to be adjusted in domains with skills that operate at a different granularity. 

|Parameter|Value|
|---|---|
|Max skill length|5|
|Min skill length|2|
|Adaptation training episodes|5|
|Transfer training episodes|30|
|Sampling temp (train)|0.7|
|Sampling temp (test)|0.0|
|Max retrieved skills|3|
|Skill refnement threshold|0|
|Skill length score weight|0.01|
|Reward score weight|0.1|
|State similarity score weight|1.0|
|Action similarity score weight|1.0|



Table 3: Skill Set Optimization hyperparameters. 

## **B. Prompts and Examples** 

Below we detail the prompts used for our LLM actor and generating skills. All prompts are the same across domains besides task, state, and admissible action descriptions. Text marked with <> indicate where variable text is inserted. We also include example skills in Tables 4 and 5 and corresponding subtrajectories in Tables 6 and 7. 

## **B.1. LLM Actor Prompt** 

You are playing a text-based game in which you must interact with your surroundings to complete a task. You will occasionally be given posisible subgoals. You may choose to target one of these subgoals or ignore them. 

<task description> 

Given the state, reflect on what has happened so far, explain your plan to accomplish the task, output which of the given subgoals you are targeting next (match one of the subgoals in the prompt word for word or output "none"), and then output the next action to execute (use one of the action templates below). 

For example: 

The last action had the effect of... To accomplish the task, I will need to... Current subgoal: [subgoal] Next action: [action] 

<admissible actions> 

The following instructions contain potentially useful information about reaching subgoals: 

11 

**Skill Set Optimization** 

Instructions for reaching the subgoal <skill subgoal>: 1. <skill instruction> 

2. <skill instruction> 

Instructions for reaching the subgoal <skill subgoal>: 

1. <skill instruction> 

2. <skill instruction> 

<state description> 

## **B.2. Skill Generation Prompt** 

You are an expert planning system. You are creating reusable skills to execute when completing various tasks. You create skills by looking at successful examples of task completions. A skill is composed of a list of instructions and a target state. After creating a skill, it will be used to execute actions in an environment. The environment will return a set of observations that summarize the new environment state. These observations will be used in conjunction with the skill’s target state to determine whether the last skill was successful. 

Consider the example trajectories of states and actions below. You’ll be asked to analyze the similarities between each. Pay attention to the wording of the state observations and actions. Then you’ll be asked to generate the common instructions, and target state for them. 

Example 1: <subtrajectory> 

Example 2: <subtrajectory> 

Generate a summary of what is happening in the examples above and the similarities between them. Provide a name for the skill that is being executed in the examples above. Do not generate skill instructions or target yet. 

<LLM output, unused> 

Generate a numbered list of instructions for completing the skill. The instructions should be similar to the actions in the examples. Instructions should use the action templates provided below. Create generic instructions that would be valid for every example but specific enough to be useful in the examples. Do not mention the examples in the instructions. Use the output format: 

Skill [skill name] instructions: 1. instruction 1 2. instruction 2 

... 

Action templates: <admissible actions> 

## <LLM output, skill instructions> 

Generate a single target observation that would indicate the success of the 

12 

**Skill Set Optimization** 

skill. The target should be similar to one of the observations in the final states. Create a generic target that would be valid for every example. Do not mention the examples in the target. Use the output format: Skill [skill name] target: [target observation] 

<LLM output, skill subgoal> 

13 

**Skill Set Optimization** 

## **C. SSO Code** 

We include the following python code as a high-level overview of SSO. However, the complete codebase is available at https://github.com/allenai/sso. 

def train(env, agent, skillset, iterations=30): for _ in range(iterations): state = env.reset() trajectory = [] done = False while not done: skills = skillset.retrieve(state) executed_skill, action = agent.act(state, skills) next_state, reward, done = env.step(action) trajectory.append((state, action, executed_skill, reward)) state = next_state 

skillset = construct(skillset, trajectory) skillset = refine(skillset, trajectory) 

def construct(skillset, trajectory, N=10, min=2, max=5, weights=[1, .1, .01]): # Extract sets of similar subtrajectories skillset.trajectories.append(trajectory) subtraj_sets = extract(skillset.trajectories[-N:], min, max) 

# Score sets of subtrajectories according to similarity, reward, and length scored_subtraj_sets = score(scored_subtraj_sets, weights) 

# Sample sets of subtrajectories according to score without overlapping sampled_subtraj_sets = sample(scored_subtraj_sets) # Generate skills from sets of subtrajectories new_skills = generate(sampled_subtraj_sets) skillset.skills.update(new_skills) return skillset def refine(skillset, trajectory, gamma=.9, epsilon=0): for t in range(len(trajectory)): state, action, executed_skill, reward = trajectory[t] if executed_skill: discounted_reward = sum( trajectory[t+i][3] * gamma ** i for i in range(len(trajectory)-t) ) skillset.observed_values[executed_skill] += discounted_reward if skillset.observed_values[executed_skill] <= epsilon: skillset.skills.remove(executed_skill) 

return skillset 

14 

**Skill Set Optimization** 

|**Created**|**Executed**|**Observed**|**Subgoal**|**Instructions**|
|---|---|---|---|---|
|**Iter**|**Count**|**Value**|||
|3|19|2.25|you focus on the thermometer|1. wait<br>2. focus on thermometer|
|5|22|5.06|you move to the kitchen|1. go hallway<br>2. go kitchen|
|||||1. focus on the thermometer|
|11|6|1.28|the stove is turned on. on the stove is: a<br>substance called liquid [substance]|2. focus on the substance you want to heat<br>3. move the focused substance to the stove<br>4. activate the stove|
|||||1. if the thermometer is not already on the|
|||||stove, move thermometer to stove|
|11|9|0.37|“it’s not clear how to read that.”|2. focus on the thermometer|
|||||3. attempt to read the thermometer|
|||||4. wait|
|||||1. if the stove is not activated, activate stove|
|13|1|0.65|you focus on the thermometer|2. move the object (if any) to the stove|
|||||3. focus on thermometer|
|24|1|0.66|successfully read the thermometer|1. focus on thermometer<br>2. read thermometer|



Table 4: Top six learned skills for the Melting Temperature ScienceWorld task according to observed value at the end of training. 

|**Created**|**Executed**|**Observed**|**Subgoal**|**Instructions**|
|---|---|---|---|---|
|**Iter**|**Count**|**Value**|||
|||||1. a: apply/use item (choose key from the inventory)|
|9|2|0.20|you succeed in unlocking the<br>door|2. k: move north (to indicate the direction of the door<br>to unlock)<br>3. y: confrm action (to unlock the door when prompted)|
|10|2|0.15|you have a key named the mas-<br>ter key of thievery|1. u: move northeast towards the key<br>2. ,: pick up the key at the current location|
|||||1. press ’o’ to initiate the open door action|
|16|1|0.31|you see an open door very near|2. press ’y’ to specify the northwest direction for the|
||||northwest|open door action|
|||||1. o - open door|
|||||2. y - move or target northwest direction when prompted|
|||||for direction after attempting to open a door|
|||||3. a - apply/use item from inventory|
|20|2|0.26|the door opens, or you unlock|4. g - choose the key for application when prompted|
||||the door|for the item to use (assuming ’g’ correlates to the|
|||||key in the inventory list)|
|||||5. y - move or target northwest direction when prompted|
|||||for direction after selecting the key to apply|
|||||1. correctly orient towards the door if not already facing it|
|23|2|0.18|the door is open|2. use ’o’ to attempt to open the door<br>3. if the door does not open and is locked, use ’a’ to apply|
|||||the key or lock pick to the door|
|||||1. move to the location of the key if not already adjacent|
|||||(using k, l, j, h, y, u, b, or n as appropriate)|
|25|3|0.29|you have a key named the mas-|2. identify the key at the current location (’y’ to move|
||||ter key of thievery|northwest if the key is northwest of the agent)|
|||||3. pick up the key at the current location (’,’)|



Table 5: Top six learned skills for the NetHack task according to observed value at the end of training. 

15 

**Skill Set Optimization** 

## **Subtraj 1** 

## Initial State: 

This room is called the art studio. In it, you see:; the agent; a substance called air; a large cupboard. The large cupboard door is closed.; a table. On the table is: a jug (containing nothing).; a wood cup (containing yellow paint); a wood cup (containing red paint); a wood cup (containing blue paint); You also see: A door to the hallway (that is open); In your inventory, you see: an orange 

## Trajectory: 

Action: go hallway Observation: You move to the hallway Action: focus go kitchen Observation: You move to the kitchen 

## Final State: 

You move to the kitchen.; This room is called the kitchen. In it, you see:; the agent; a substance called air; a ceramic cup (containing a substance called lead); a chair. On the chair is: nothing.; a counter. On the counter is: a bowl (containing a red apple, a banana, an orange, a potato), a drawer.; a cupboard. The cupboard door is closed.; a freezer. The freezer door is closed.; a fridge. The fridge door is closed.; a glass jar (containing a substance called sodium chloride); a lighter; a orange box (containing nothing); a oven, which is turned off. The oven door is closed.; a painting; a sink, which is turned off. In the sink is: nothing.; a substance called soap; a stopwatch, which is deactivated.; a stove, which is turned off. On the stove is: nothing.; a table. On the table is: a glass cup (containing nothing).; a thermometer, currently reading a temperature of 10 degrees celsius; a yellow box (containing nothing); You also see: A door to the bathroom (that is open), A door to the hallway (that is open), A door to the outside (that is open); In your inventory, you see: an orange 

## **Subtraj 2** 

## Initial State: 

This room is called the greenhouse. In it, you see:; the agent; a substance called air; a bee hive. The bee hive door is closed.; a jug (containing nothing); a shovel; a sink, which is turned off. In the sink is: nothing.; You also see: A door to the hallway (that is open), A door to the outside (that is open); In your inventory, you see: an orange 

Trajectory: 

Action: go hallway 

Observation: You move to the hallway Action: focus go kitchen Observation: You move to the kitchen 

## Final State: 

You move to the kitchen.; This room is called the kitchen. In it, you see:; the agent; a substance called air; a chair. On the chair is: nothing.; a counter. On the counter is: a bowl (containing a red apple, a banana, an orange, a potato), a drawer.; a cupboard. The cupboard door is closed.; a freezer. The freezer door is closed.; a fridge. The fridge door is closed.; a glass jar (containing a substance called sodium chloride); a lighter; a orange box (containing nothing); a oven, which is turned off. The oven door is closed.; a painting; a plastic cup (containing a substance called gallium); a sink, which is turned off. In the sink is: nothing.; a substance called soap; a stopwatch, which is deactivated.; a stove, which is turned off. On the stove is: nothing.; a table. On the table is: a glass cup (containing nothing).; a thermometer, currently reading a temperature of 10 degrees celsius; a yellow box (containing nothing); You also see: A door to the bathroom (that is open), A door to the hallway (that is open), A door to the outside (that is open); In your inventory, you see: an orange 

Table 6: Subtrajectories used to generate the “you move to the kitchen” skill described in Table 4. 

16 

**Skill Set Optimization** 

## **Subtraj 1** 

## Initial State: 

You have a +0 short sword (weapon in hand). You have 15 +0 daggers (alternate weapon; not wielded). You have an uncursed +1 leather armor (being worn). You have an uncursed potion of sickness. You have an uncursed lock pick. You have an empty uncursed sack. You see a vertical wall far east. You see a horizontal wall near north and northeast. You see a area of lava near northeast. You see a stairs down near northeast. You see a vertical wall near west. You see a horizontal closed door near northwest. You see a dark area near northwest. You see a lava very near northeast, northeast, and east. You see a horizontal wall adjacent southeast, south, and southwest. You see a key adjacent northwest. Hello Agent, welcome to NetHack! You are a chaotic male human Rogue. 

Trajectory: Action: y 

Observation: You see here a key named The Master Key of Thievery 

Action: , 

Observation: g - a key named The Master Key of Thievery 

Final State: 

You have a +0 short sword (weapon in hand). You have 15 +0 daggers (alternate weapon; not wielded). You have an uncursed +1 leather armor (being worn). You have an uncursed potion of sickness. You have an uncursed lock pick. You have an empty uncursed sack. You have a key named The Master Key of Thievery. You see a vertical wall far east. You see a horizontal wall near north and northeast. You see a lava near northeast, east, and southeast. You see a area of lava near northeast. You see a stairs down near east. You see a vertical wall near west. You see a dark area near northwest. You see a horizontal wall very near southeast, south, and southwest. You see a horizontal closed door very near northwest. You see a stairs up adjacent southeast. g - a key named The Master Key of Thievery. 

## **Subtraj 2** 

## Initial State: 

You have a +0 short sword (weapon in hand). You have 8 +0 daggers (alternate weapon; not wielded). You have an uncursed +1 leather armor (being worn). You have an uncursed potion of sickness. You have an uncursed lock pick. You have an empty uncursed sack. You have an uncursed blindfold. You see a vertical wall far east. You see a horizontal wall near north and northeast. You see a area of lava near northeast. You see a stairs down near northeast. You see a vertical wall near west. You see a horizontal closed door near northwest. You see a dark area near northwest. You see a lava very near northeast, northeast, and east. You see a horizontal wall adjacent southeast, south, and southwest. You see a key adjacent northwest. Hello Agent, welcome to NetHack! You are a chaotic male human Rogue. 

Trajectory: Action: y 

Observation: You see here a key named The Master Key of Thievery 

Action: , 

Observation: h - a key named The Master Key of Thievery 

Final State: 

You have a +0 short sword (weapon in hand). You have 8 +0 daggers (alternate weapon; not wielded). You have an uncursed +1 leather armor (being worn). You have an uncursed potion of sickness. You have an uncursed lock pick. You have an empty uncursed sack. You have an uncursed blindfold. You have a key named The Master Key of Thievery. You see a stairs down far northeast. You see a vertical wall far east. You see a horizontal wall near north and northeast. You see a lava near northeast, east, and southeast. You see a area of lava near northeast. You see a vertical wall near west. You see a dark area near northwest. You see a horizontal wall very near southeast, south, and southwest. You see a horizontal closed door very near northwest. You see a stairs up adjacent southeast. h - a key named The Master Key of Thievery. 

Table 7: Subtrajectories used to generate the “you have a key named the master key of thievery” skill described in Table 5. 

17 

