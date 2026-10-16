_2024-10-11_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0001-02.png)


# **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** 

**Amrith Setlur**[1,3,*] **, Chirag Nagpal**[1,*] **, Adam Fisch**[2] **, Xinyang Geng**[2] **, Jacob Eisenstein**[2] **, Rishabh Agarwal**[2] **, Alekh Agarwal**[1] **, Jonathan Berant**[†][,2] **and Aviral Kumar**[†][,2,3] 

1Google Research, 2Google DeepMind, 3Carnegie Mellon University, *Equal contribution, †Equal advising 

**A promising approach for improving reasoning in large language models is to use process reward models (PRMs). PRMs provide feedback at each step of a multi-step reasoning trace, potentially improving credit assignment over outcome reward models (ORMs) that only provide feedback at the final step. However, collecting dense, per-step human labels is not scalable, and training PRMs from automatically-labeled data has thus far led to limited gains. To improve a** _**base**_ **policy by running search against a PRM or using it as dense rewards for reinforcement learning (RL), we ask: “How should we design process rewards?”. Our key insight is that, to be effective, the process reward for a step should measure** _**progress**_ **: a change in the likelihood of producing a correct response in the future, before and after taking the step, corresponding to the notion of step-level advantages in RL. Crucially, this progress should be measured under a** _**prover**_ **policy distinct from the base policy. We theoretically characterize the set of good provers and our results show that optimizing process rewards from such provers improves exploration during test-time search and online RL. In fact, our characterization shows that weak prover policies can substantially improve a stronger base policy, which we also observe empirically. We validate our claims by training** _**process advantage verifiers (PAVs)**_ **to predict progress under such provers, and show that compared to ORMs, test-time search against PAVs is** _>_ 8% **more accurate, and** 1 _._ 5 − 5× **more compute-efficient. Online RL with dense rewards from PAVs enables** _**one of the first results**_ **with** 5 − 6× **gain in sample efficiency, and** _>_ 6% **gain in accuracy, over ORMs.** 

## **1. Introduction** 

Trained reward models or _verifiers_ are often used to improve math reasoning in large language models, either by re-ranking solutions at test-time (Collins, 2000) or via reinforcement learning (RL) (Uesato et al., 2022). Typically, verifiers are trained to predict the outcome of an entire reasoning trace, often referred to as _outcome_ reward models (ORM) (Cobbe et al., 2021b; Hosseini et al., 2024). However, ORMs only provide a sparse signal of correctness, which can be hard to learn from and inefficient to search against. This challenge is alleviated by fine-grained supervision, in theory. For reasoning, prior works train _process_ reward models (PRMs) that assign intermediate rewards after each step of search (Snell et al., 2024) or during RL. While Lightman et al. (2023) obtains PRM annotations from human raters, this approach is not scalable. More recent works (Luo et al., 2024; Wang et al., 2024) train PRMs to predict automatically-generated annotations that estimate future success of solving the problem, akin to value functions in RL. So far, automated PRMs, especially as dense rewards in RL, only improve by 1-2% over ORMs (Shao et al., 2024), raising serious doubts over their utility. 

To resolve these uncertainties, in this paper, we train PRMs with automated annotations, such that optimizing the dense rewards from trained PRMs can improve a _base_ policy compute- and sampleefficiently, during test-time search and online RL. For this, we first ask: **(i)** what should the _per-step_ process rewards measure, and **(ii)** what kind of automated data collection strategy should we use to train PRMs that predict this measure. For **(i)** , conventional belief (Lightman et al., 2023; Uesato et al., 2022) 

Corresponding author(s): asetlur@cs.cmu.edu.[1] Work done during an internship at Google Research. 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
1 Follow Gaussian Elimination.  … Final answer: x=1, y=7<br>0.0<br>We eliminate  y  from<br>system of equations.<br>1.0<br>2 2 x Eqn. 1  – Eqn. 2 gives us …. … Final answer: x=1, y=7<br>The equations imply:<br>10x + 9y = 25 0.0<br>Subtract Eqn. 2 from the  … Final answer: x=3, y=-1<br>previous step: 3x + 3y = -24<br>Start Question : Let 4 x +3 y =25,<br>7x+6y=49. Solve for  x ,  y . …<br>0.0 Previous implication seems incorrect.  Final answer: x=1, y=7<br>(a)<br>Very capable  Good prover policy:<br>Base Policy<br>prover policy  complementary to base<br>(b) (c)<br>**----- End of picture text -----**<br>


**Figure 1** | _**Process advantage verifiers (PAV):**_ Process reward for a step is defined as progress (advantage) under the prover policy, _i.e._ , change in prover policy’s success rate before and after the step. **(a):** The base policy samples both correct 1 and incorrect 2 steps but struggles to succeed from either. A strong prover policy completes the solution from both steps, and is unable to adequately reflect progress made by 1 and 2 (both scored 0.0). Conversely, a complementary prover policy distinguishes 1 , 2 more prominently (only succeeds from 1 ). **(b,c):** Compared to ORMs, PAVs are 5x more compute efficient, 10% more accurate in test-time search, and 6x more sample efficient, 7% more accurate for online reinforcement learning (RL). 

has been to measure mathematical correctness or relevance of steps. But, it is unclear if this supervision yields the most improvement in the base policy ( _e.g._ , a policy may need to generate simpler, repetitive, and even incorrect steps to explore and discover the final answer during test-time search and RL). **Our key insight** is that per-step, process rewards that measure a notion of _progress_ : change in the likelihood of arriving at a correct final answer before and after taking the step, are effective, for both test-time beam search and online RL. Reinforcing steps that make progress regardless of whether they appear in a correct or incorrect trace diversifies the _**exploration**_ of possible answers at initial steps, which is crucial when the approach to solve a problem is not clear. Formally, such rewards correspond to per-step _advantages_ of steps from the RL literature (Sutton and Barto, 2018). We empirically show that using advantages in addition to ORM rewards outperforms the typical use of future probabilities of success or _𝑄_ -values (Wang et al., 2024) for both search and RL. This is because, when given a combinatorial space of responses, under bounded computational and sampling constraints, _𝑄_ -values mainly “exploit” states whereas advantages also “explore” steps that make the most progress towards the final answer (Fig. 2). 

To answer **(ii)** , we first note that advantages under a poor base policy are ≈ 0 on most steps, and thus will not be informative for search or RL. In addition, regardless of the strength of the base policy, using its own per-step advantages as process rewards in RL will result in base policy updates equivalent to _only_ 

2 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

using outcome rewards for RL (since a standard policy gradient algorithm already computes advantages). Hence, we propose to use advantages estimated via rollouts under a different _**prover policy**_ as process rewards (Fig. 1(a)). How should we choose this prover policy? A natural guess would be to use a very capable prover. However, we show advantages under an overly capable prover policy, that can succeed from any step, fail to distinguish good and bad steps. A similar argument holds for very weak provers. 

In theory, we formalize this intuition to define good provers as policies that are _complementary_ to the base policy ( _i.e._ , policies with advantages that can contrast steps produced by the base policy sufficiently), while still producing step-level advantages correlated with those of the base policy. For _e.g._ , for Bestof- _𝐾_ policies (Nakano et al., 2021) corresponding to a base policy, we empirically find that provers corresponding to _𝐾>_ 1 (but not too large) are more capable at improving the base policy. Contrary to intuition, the set of complementary provers also contains policies that are worse than the base policy. To predict the advantages of such provers we train dense verifiers, called _**process advantage verifiers (PAVs)**_ , that accelerate sample and compute efficiency of RL and search. 

With the conceptual design of PAVs in place, we prescribe practical workflows for training PAVs and demonstrate their efficacy on a series of 2B, 9B, and 27B Gemma2 models (Gemma Team et al., 2024). PAV training data is gathered by sampling “seed” solution traces from the prover and partial rollouts from the same to estimate the _𝑄_ -value at each prefix of the seed trace. Our workflow prescribes favorable ratios for seed and partial rollouts. Our first set of empirical results show that for an equal budget on test-time compute, beam search against trained PAVs is >8% better in accuracy, and **1** _._ **5 − 5×** more compute efficient compared to re-ranking complete traces against an ORM (Fig. 1(b)). Dense rewards from PAVs improve the efficiency of step-level exploration during search by pruning the combinatorial space of solutions aggressively and honing in on a diverse set of possible sequences. Finally, we demonstrate _**for the first time**_ , that using PAVs as dense rewards in RL scales up data efficiency by **6×** compared to only using outcome rewards (Fig. 1(c)). Moreover, base policies trained with PAVs also achieve **8×** better Pass @ _𝑁_ performance (probability of sampling the correct solution in _𝑁_ attempts), and consequently afford a higher ceiling on the performance of any test-time re-ranker. Finally, running RL with PAVs discovers solutions to hard problems that sampling from the SFT policy with a very large budget can’t solve. 

## **2. Preliminaries, Definitions, and Notation** 

Following protocols from Lightman et al. (2023); Uesato et al. (2022), a reasoning trace from an LLM consists of multiple logical steps separated by a demarcation token. An outcome reward model (ORM) is a trained verifier that assigns a numerical score after the last step of the trace, and a process reward model (PRM) is a trained verifier that scores each step of the trace individually. 

**Problem setup and notation.** Given a math problem _𝒙_ ∈X, our goal is to improve a _base policy 𝜋_ that samples a response _𝒚_ ∼ _𝜋_ (· | _𝒙_ ) in the set Y. A response _𝒚_ consists of multiple reasoning steps (maximum _𝐻_ ), separated by a delimiter (‘next line’ in our case), _i.e._ , _𝒚_ = ( _𝑎_ 1 _, 𝑎_ 2 _, . . . , 𝑎𝐻_ ). Since sampling is auto-regressive, we can view each step as an action taken by the agent _𝜋_ in a Markov decision process (MDP) with deterministic dynamics. Specifically, we treat the prefix ( _𝒙, 𝑎_ 1 _, . . . , 𝑎ℎ_ −1) as the current _state 𝒔ℎ_ and next step _𝑎ℎ_ ∼ _𝜋_ (· | _𝒙_ ) as the _action_ taken by _𝜋_ at _𝒔ℎ_ , resulting in the next state _𝒔ℎ_ +1. For problem _𝒙_ , with ground-truth response _𝒚𝒙[★]_[, we can evaluate the accuracy of] _[𝜋]_[by running a regular expression] match on the final answer (Hendrycks et al., 2021): Rex( _𝒚, 𝒚𝒙[★]_[)][↦→{][0] _[,]_[ 1][}][,] _[i.e.]_[,][accuracy][is][given][by] 𝔼 _𝒚_ ∼ _𝜋_ (·| _𝒙_ ) �Rex( _𝒚, 𝒚𝒙[★]_[)] �. Now, given a dataset D = {( _𝒙𝑖, 𝒚𝒙[★] 𝑖_[)}] _[𝑖]_[of problem-solution pairs, the main goal is to] learn a good base policy by optimizing this outcome reward on D. Next, we see how we can leverage the final answer verifier Rex available on D to train ORMs and PRMs. 

3 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

**Outcome reward model (ORM).** Given a response _𝒚_ , an ORM estimates the ground-truth correctness Rex( _𝒚, 𝒚𝒙[★]_[)][.][To][train][such][a][model][we][first][take][problems][in][D][,][and][collect][training][data][of][the][form] {( _𝒙, 𝒚_ ∼ _𝜋_ (· | _𝒙_ ) _,_ Rex( _𝒚, 𝒚𝒙[★]_[))}][.][Then][we][train][an][ORM][that][takes][as][input][a][problem-response][pair] ( _𝒙, 𝒚_ ) and predicts Rex( _𝒚, 𝒚𝒙[★]_[)][.][At test time, when] _[𝒚] 𝒙[★]_[is unknown, the ORM is used to score candidate] solutions revealed by test-time search. Given a base policy _𝜋_ , a Best-of- _𝐾_ policy: BoK( _𝜋_ ), is a policy that samples _𝐾_ responses from _𝜋_ , scores them against an ORM, and returns the one with the highest score. Whenever the ORM matches Rex, the performance of BoK( _𝜋_ ) is referred to as Pass @ _𝐾_ . Furthermore, when the likelihood of _𝜋_ solving problem _𝒙_ is _𝑝𝒙_ , then for BoK( _𝜋_ ) this likelihood is given by the expression: 1 −(1 − _𝑝𝒙_ ) _[𝐾]_ . In general, this is larger than _𝑝𝒙_ , making BoK( _𝜋_ ) stronger than _𝜋_ for _𝐾>_ 1. 

**Standard process reward models (PRMs).** A PRM scores every step _𝑎ℎ_ in a multi-step response _𝒚_ ∼ _𝜋_ ( _e.g._ , in Lightman et al. (2023) PRMs are trained to score correct steps over incorrect and irrelevant ones). But, unlike ORMs, which only require Rex for data collection, PRM training data requires expensive step-level human annotations. Prior works (Luo et al., 2024; Wang et al., 2024) attempted to scale process rewards automatically by sampling from the model to provide a heuristic understanding of when a step is actually correct. In particular, they evaluate a prefix by computing the expected future accuracy of multiple completions sampled from _𝜋_ , after conditioning on the prefix, _i.e._ , value function _𝑄[𝜋]_ (Eq. 1) from RL. Similarly, we define _𝑉[𝜋]_ ( _𝒔ℎ_ ) � 𝔼 _𝑎ℎ_ ∼ _𝜋_ (·| _𝒔ℎ_ ) _𝑄[𝜋]_ ( _𝒔ℎ, 𝑎ℎ_ ) as value of state _𝒔ℎ_ . These works use _𝑄[𝜋]_ as the PRM that assigns a score of _𝑄[𝜋]_ ( _𝒔ℎ, 𝑎ℎ_ ) to the action _𝑎ℎ_ , at state _𝒔ℎ_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0004-03.png)


**Using PRMs for beam search at test-time.** Given a PRM, a natural way to spend test-time compute is to use it as a step-level re-ranker within a beam search procedure (Snell et al., 2024). For each problem, at step 0, a beam of maximum width _𝐵_ , is initialized with a single state consisting of just the problem. At step _ℎ_ , a beam contains partial responses unrolled till a set of states or prefixes { _𝒔𝑖_ } _𝑖[𝐵]_ =1[.][From each state] _[𝒔][𝑖]_ in this set, _𝐶_ independent actions or steps { _𝑎𝑖,𝑗_ } _[𝐶] 𝑗_ =1[are sampled from] _[𝜋]_[(·][|] _[𝒔][𝑖]_[)][, each of which leads to a] new state. Process rewards from PRMs assign a score to every new state ( _𝒔𝑖, 𝑎𝑖,𝑗_ ), and only the states corresponding to the top _𝐵_ values are retained in the beam for the next step. 

## **3. How Should we Define Process Rewards and Why?** 

Ultimately, we are interested in test-time search and RL methods that can most efficiently and reliably discover solution traces with the correct final answer, thus maximizing Rex. To this end, _process rewards should serve as step-level supervision to indirectly maximize outcome-level_ Rex. Our position contrasts with conventional belief that process rewards should mainly evaluate mathematical correctness or relevance of individual steps (Lightman et al., 2023; Uesato et al., 2022), since LLMs might need to generate trivial or repetitive intermediate steps in order to discover a trace with the correct final answer. With this insight, in this section we approach the design of dense automated step rewards as a form of supervision to be used in conjunction with sparse outcome rewards to improve the base policy. 

In an MDP, a starting point to design step-level dense feedback that is eventually meant to optimize a sparse outcome reward Rex is to consider the notion of a _potential function_ (Ng et al., 1999): in our case, this is a function that summarizes the difference between some statistic of the policy at the future state and the same statistic computed at the current state. By appealing to this framework, in Sec. 3.1, we 

4 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0005-01.png)


**----- Start of picture text -----**<br>
Q [π] (s, a) 0.55 0.50 0.70 0.50 0.35 0.25<br>Question<br>Q [μ] =1  on each state,<br>𝑎1,1 𝑎1,2 𝑎1,3 𝑎2,1 𝑎2,2 𝑎2,3 ∼ π trivial actions<br>A [π] (s, a) -0.05 -0.1 +0.10 +0.20 +0.05 -0.05 Rephrased ∼ μ  are rewarded<br>Question<br>V [π] s 0.60 𝑠1 0.30 𝑠2 Correct solution<br>Beam ∼ π from prover<br>States chosen by  Steps chosen by<br>Repeated ∼ μ<br>(a) Q [π] (s, a) A [π] (s, a) (b) Question<br>**----- End of picture text -----**<br>


**Figure 2** | _**Issues with using** 𝑄_ _**-values as process rewards**_ **: (a):** Unlike _𝐴[𝜋]_ , _𝑄[𝜋]_ mixes action evaluation with the _𝑄_ -value of the previous state. Beam search with _𝑄[𝜋]_ exploits high-likelihood states, while adding _𝐴[𝜋]_ ( _e.g._ , _𝑄[𝜋]_ + _𝛼𝐴[𝜋]_ in Eq. 5) aids in exploring states reached by making actions that induce progress, _i.e._ , increase likelihood of success. **(b):** _𝑄[𝜇]_ from a strong prover _𝜇_ can assign unmerited bonuses to trivial actions. 

show that advantages – not value functions (Luo et al., 2024; Wang et al., 2024) – that measure a notion of “progress” at each new step are more appropriate for use as dense rewards in search and RL (primarily for exploration). Then in Secs. 3.3 and 3.4, we show that this progress or advantage vakue is measured best under a policy _𝜇_ , different from the base policy _𝜋_ . We call this policy _𝜇_ , the _**prover policy**_ . 

## **3.1. Process Rewards Should be Advantages, Not Value Functions** 

To understand the relationship to potential functions, we first study test-time beam search, and present some challenges with the reward design of Snell et al. (2024), that uses value function _𝑄[𝜋]_ ( _𝒔, 𝑎_ ) of the base policy _𝜋_ to reward action _𝑎_ at state _𝒔_ . Consider the example in Fig. 2(a), where from the 2 states in the beam, we sample 3 actions. If we pick next states purely based on highest values of _𝑄[𝜋]_ , we would be comparing steps sampled from different states ( _e.g._ , _𝑎_ 1 _,_ 1 vs. _𝑎_ 2 _,_ 1) against each other. Clearly, a reduction in expected final outcome, _i.e._ , _𝑄[𝜋]_ ( _𝒔_ 1 _, 𝑎_ 1 _,_ 1) − _𝑉[𝜋]_ ( _𝒔_ 1), means that _𝑎_ 1 _,_ 1 _by itself_ has a negative effect of −0 _._ 05 on the probability of success from _𝒔_ 1, whereas _𝑎_ 2 _,_ 1 has a positive effect of +0 _._ 20 from _𝒔_ 2. However, expanding the beam based on _absolute_ values of _𝑄[𝜋]_ retains the action that makes negative progress, and removes state _𝒔_ 2 from the beam (as beam size is 2). In other words, _𝑄[𝜋]_ fails to decouple the “evaluation” of an action (step), from the “promise” shown by the previous state. This will not be an issue for every problem, and particularly not when the beam capacity is unbounded, but under finite computational and sampling constraints, using _𝑄[𝜋]_ might retain states with potentially unfavorable steps that hurt the overall likelihood of success. **If we could also also utilize the** _**progress**_ **made by the previous step** along with the likelihood of success _𝑄[𝜋]_ when deciding what to retain in the beam, then we can address this tradeoff. 

_**How can we measure the “progress” made by a step?**_ One approach is to consider the relative increase/decrease in the likelihood of success, before and after the step. This notion is formalized by the advantage (Eq. 2) of a step under policy _𝜋_ . Furthermore, since advantages can attach either positive or negative values to a step, training the base policy against advantages supervises the base policy when it generates a step that makes progress (where _𝐴[𝜋] >_ 0), and also when it fails to produce one, employing a “negative gradient” that speeds up RL training (Tajwar et al., 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0005-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0005-08.png)


Recall that since we view process rewards as potential functions in the MDP, they can be computed under 

5 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

any policy _𝜇_ , which can be the base policy. However, in the above example, reasons for which _𝑄[𝜋]_ is a seemingly unfit choice for process rewards also apply to _𝑄[𝜇]_ . Nevertheless, we can possibly use advantage under _𝜇_ : _𝐴[𝜇]_ , which measures the progress made by a step to improve the likelihood of success under _𝜇_ . In that case, how should we choose this policy _𝜇_ , that we call the prover policy, and should it be necessarily different from base policy _𝜋_ ? Before diving into the choice of _𝜇_ , we discuss a more pertinent question: how should we use _𝐴[𝜇]_ in conjunction with outcome rewards for improving the base policy _𝜋_ ? We will then formally reason about the choice of _𝜇_ in Secs. 3.3 and 3.4. 

## **3.2. Our Approach: Process Advantage Verifiers (PAV)** 

For building an approach that uses process rewards _𝐴[𝜇]_ together with the outcome reward Rex to improve the base policy _𝜋_ , we situate ourselves in the context of improving _𝜋_ with online RL. If all we had was access to Rex on D, the standard RL objective is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0006-04.png)


Inspired by how reward bonuses (and potential functions) are additive (Bellemare et al., 2016; Ng et al., 1999), one way to use process rewards _𝐴[𝜇]_ is to combine it with the standard RL objective as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0006-06.png)


The term in red is the difference in likelihoods of success of the prover _𝜇_ , summed over consecutive steps (a notion of _**progress**_ ). Here, _𝑑ℎ[𝜋]_[′][denotes the distribution over states at step] _[ℎ]_[, visited by the old policy] _[ 𝜋]_[′] (policy at previous iterate). Following policy gradient derivations (Williams, 1992): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0006-08.png)


At a glance, we can view _𝑄[𝜋]_ ( _𝒔ℎ, 𝑎ℎ_ ) + _𝛼𝐴[𝜇]_ ( _𝒔ℎ, 𝑎ℎ_ ) as the effective reward for step _𝑎ℎ_ when scored against a combination of the outcome evaluation Rex, i.e., _𝑄[𝜋]_ , and process rewards _𝐴[𝜇]_ . Thus, we can optimize Eq. 4 indirectly via **(a)** running beam-search against the effective reward; or **(b)** online RL where the policy gradients are given by Eq. 5. For either of these, we need access to verifiers that are trained to predict the advantage _𝐴[𝜇]_ ( _𝒔ℎ, 𝑎ℎ_ ) under the prover. We refer to these verifiers as _**process advantage verifiers (PAVs)**_ . In Sec. 4.2 we describe how to train PAVs, but now we use the above formulation to reason about how to choose prover _𝜇_ that is most effective at improving base _𝜋_ . 

We also remark that the term in red resembles prior work on imitation learning via policy optimization (Ross and Bagnell, 2014; Sun et al., 2017), where the main aim is to learn a policy _𝜋_ that imitates the prover _𝜇_ , or to improve upon it to some extent. Of course, this is limiting since our goal is to not just take actions that perform at a similar level as _𝜇_ , but to improve the base policy even further, and using a combination of _𝑄[𝜋]_ and _𝐴[𝜇]_ is critical towards this goal. 

**How should we choose the prover** _𝜇_ **?** Perhaps a natural starting point is to set the prover to be identical to the base policy, _i.e._ , _𝜇_ = _𝜋_ , which produces process rewards that prior works have considered Shao et al. (2024). However, setting _𝐴[𝜋]_ = _𝐴[𝜇]_ in Eq. 5 results in exactly the same policy gradient update as only optimizing outcome evaluation Rex. Moreover, for a poor base policy _𝜋_ , where _𝑄[𝜋]_ ≈ 0 on most states, 

6 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

the term _𝐴[𝜋]_ would also be ≈ 0, and hence running beam search with the effective rewards would not be informative at all. Hence, _**a better approach is to use a different prover policy**_ , but a very weak prover _𝜇_ will likely run into similar issues as a poor base policy. We could instead use a very capable prover _𝜇_ , but unfortunately even this may not be any better than optimizing only the outcome reward either. To see why, consider a scenario where _𝜋_ ’s response contains an intermediate step that does not help make progress towards the solution ( _e.g._ , _𝜋_ simply restates the question, see Fig. 2(b)). Here, _𝑄[𝜇]_ for a capable prover before and after this irrelevant step will be identical since _𝜇_ can succeed from either step. This means that _𝜇_ fails to distinguish steps, resulting in _𝐴[𝜇]_ ≈ 0 in most cases. Training with this process reward during RL will then lead to gradients that are equivalent to those observed when purely optimizing _ℓ_ ORM−RL. In fact, empirically, we observe that online RL with _𝑄[𝜇]_ from strong provers leads to polices that only produce re-phrasings of the question (App. G) and do not succeed at solving the question. Clearly, _any_ policy different from the base policy cannot serve as a prover. So, how do we identify a set of good provers? Can they indeed be weaker than the base policy? We answer next. 

## Takeaway: What should process rewards measure during test-time search and online RL? 

- Process rewards should correspond to progress, or **advantage** , as opposed to absolute _𝑄_ -values, for a better explore-exploit tradeoff during beam search and online RL. 

- Advantages should be computed using a **prover** policy, different from the base policy. 

## **3.3. Analysis in a Didactic Setting: Learning a Planted Sub-sequence** 

In this section, we aim to characterize prover policies that are effective in improving the base policy. To do so, we first introduce a didactic example, representative of real reasoning scenarios to illustrate the main intuition. Then, we will formalize these intuitions in the form of theoretical results. 

**Didactic example setup.** Given an unknown sub-sequence _𝒚[★]_ consisting of tokens from vocabulary V � {1 _,_ 2 _, . . . ,_ 15}, we train a policy _𝜋_ to produce a response which contains this sub-sequence. The task completion reward is terminal and sparse, _i.e._ , _𝑟_ ( _𝒚, 𝒚[★]_ ) = 1 for a _𝒚_ if and only if _𝒚[★]_ appears in _𝒚_ . By design, the reward _𝑟_ ( _𝒚, 𝒚[★]_ ) resembles outcome reward Rex( _𝒚, 𝒚𝒙[★]_[)][in Sec.][ 2][.][The prover policy] _[𝜇]_[is a] procedural policy, parameterized by a scalar _𝛾>_ 0 (details in App. B). As _𝛾_ increases, the performance of _𝜇_ improves and → 1 as _𝛾_ →∞. For simplicity, we assume oracle access to ground-truth _𝐴[𝜇]_ and _𝑄[𝜋]_ , and alleviate errors from learned verifiers approximating these values. 

**(1) RL with effective reward** _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ **is** 10× **more sample-efficient than only outcome reward.** In Fig. 3(a), we first note that training _𝜋_ with this effective reward under a prover _𝜇_ with strength _𝛾_ = 10, produces optimal performance (100% accuracy) in 350 iterations, despite starting from a mediocre initialization for _𝜋_ ( _𝛾_ = 5 _._ 0). Training with only outcome reward is ineffective. More importantly, in Fig. 3(b), we note that effective rewards only help for a set of provers, in _𝛾_ ∈[8 _._ 0 _,_ 15 _._ 0]. Outside this range, we observed advantages _𝐴[𝜇]_ were close to 0 on most states, either because _𝜇_ was poor (small _𝛾_ ) and was unable to generate _𝒚[★]_ even when _𝜋_ got the sequence partially correct, or because _𝜇_ was strong (large _𝛾_ ) that it generated _𝒚_[∗] with almost equal likelihood from all prefixes. 

**(2) Effective reward improves Pass @N by** 5× **over only outcome reward.** We report the “Pass @N” performance in Fig. 3(c), which measures the maximum reward _𝑟_ across _𝑁_ traces sampled _i.i.d._ from _𝜋_ and hence, represents the ceiling on the performance of any test-time search method that picks a single response from multiple draws ( _e.g._ , as in Best-of-N). For a policy trained with the effective reward for 100 iterations, the Pass @N performance grows 5× faster with _𝑁_ , compared to the policy trained with only 

7 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
RL with Different Rewards Choice of Prover Policy  Ceiling on Test-time Re-ranking<br>1.0 0.25<br>Effective Reward<br>0.8 0.8 0.20 Only Q<br>0.6 Only Q 0.6 Effective Reward 0.15 Initial<br>Only Q<br>0.4 Effective Reward 0.4  for initial  0.10<br>0.2 0.2 0.05<br>0.0 0.0 0.00<br>(a) 0.5k 1k 1.5k 2k 2.5k 3k 3.5k (b) 1.0 5.0 9.0 13.0 17.0 (c) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] 2 [8]<br>Training iterations N<br>Accuracy Pass @N<br>Accuracy (avg. over iters)<br>**----- End of picture text -----**<br>


**Figure 3** | _**Results for our didactic analysis:**_ **(a):** We train base policy via RL with either effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ , or the typical _𝑄[𝜋]_ (computed via Monte-Carlo sampling). **(b):** We vary the strength _𝛾_ of the prover _𝜇_ used to compute advantages _𝐴[𝜇]_ in the effective reward, and plot the base policy accuracy averaged over the RL run. **(c):** We plot the max score out of _𝑁_ responses (Pass @N) sampled _i.i.d._ from an undertrained base policy (iter 100) . 

the outcome reward. Due to only sparse feedback, the latter policy does not learn to sample partially correct _𝒚[★]_ , whereas a policy trained with the effective reward produces partially correct _𝒚[★]_ , and is able to sample the complete _𝒚[★]_ with higher likelihood during Pass @N. 

## Takeaway: Online RL with process rewards from different prover policies. 

Effective rewards _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ from prover _𝜇_ : (i) improve sample efficiency of online RL, and (ii) yield policies with better Pass @N performance, over using only outcome rewards. But, advantages of very capable or poor _𝜇_ do not improve base policy beyond outcome rewards. 

## **3.4. Theory: Provers Complementary to the Base Policy Boost Improvement** 

From our didactic analysis, it is clear that process rewards _𝐴[𝜇]_ under different provers _𝜇_ disparately affect the base policy that optimizes _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ via online RL. We now present a formal analysis of why this happens and characterize a class of provers that can guarantee non-trivial improvements to the base policy. For simplicity, we assume oracle access to _𝑄[𝜋] , 𝐴[𝜇]_ at every state-action pair ( _𝒔ℎ, 𝑎ℎ_ ) and prove our result in the tabular RL setting, where the policy class is parameterized using the softmax parameterization in Agarwal et al. (2021). Proofs for this section are in App. F. 

**Main intuitions.** We expect a prover _𝜇_ to improve a base policy _𝜋_ only when _𝝁_ **is able to** _**distinguish**_ **different actions taken by** _𝜋_ , by attaining sufficiently varying advantage values _𝐴[𝜇]_ ( _𝒔ℎ, 𝑎_ ) for actions _𝑎_ at state _𝒔ℎ_ . This can be formalized under the notion of sufficiently large variance across actions, 𝕍 _𝑎_ ∼ _𝜋_ [ _𝐴[𝜇]_ ( _𝒔ℎ, 𝑎_ )]. In that case, can we simply use a policy with large advantage variance under any measure? No, because when the prover _𝜇_ ranks actions at a given state very differently compared to the base policy _𝜋_ (e.g., if _𝐴[𝜇]_ and _𝐴[𝜋]_ are opposite), then effective rewards _𝑄[𝜇]_ + _𝛼𝐴[𝜋]_ will be less reliable due to conflicting learning signals. Thus, we want 𝔼 _𝜋_ [⟨ _𝐴[𝜇] , 𝐴[𝜋]_ ⟩] to not be too negative, so that _𝝁_ **and** _𝝅_ **are reasonably** _**aligned**_ on their assessment of steps from _𝜋_ . 

In Theorem 3.1, we present our result on policy improvement where the base policy is updated with natural policy gradient (Kakade, 2001a): _𝜋𝑡_ +1( _𝑎_ | _𝒔ℎ_ ) ∝ exp( _𝛾_ · ( _𝑄[𝜋]_ ( _𝒔ℎ, 𝑎_ ) + _𝐴[𝜇]_ ( _𝒔ℎ, 𝑎_ ))). We note that in this idealized update rule, swapping _𝑄_ values (of _𝜇_ or _𝜋_ ) with advantages does not affect the update since we assume access to all possible actions when running the update. Nonetheless, despite this simplifying assumption, the analysis is able to uncover good choices for the prover policy _𝜇_ for computing process 

8 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

reward _𝐴[𝜇]_ , and is orthogonal to the design consideration of advantages or _𝑄_ -values as process rewards that we have discussed so far in this paper. Theorem 3.1 formalizes our intuition by showing that policy improvement at iteration _𝑡_ , grows as the variance in _𝐴[𝜇]_ values increases (higher distinguishability) and reduces when _𝐴[𝜇]_ and _𝐴[𝜋]_ become extremely misaligned. This will then allow us to discuss a special case for the case of Best-of-K policies as provers as an immediate corollary. 

**Theorem 3.1** (Lower bound on policy improvement; informal) **.** _For base policy iterate 𝜋𝑡, after one step of policy update, with learning rate 𝛾_ ≪ 1 _, the improvement over a distribution of states 𝜌:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0009-03.png)


It may seem that the base policy _𝜋_ can only learn from an improved prover _𝜇_ , but our result shows that a **weak prover can also amplify a stronger base policy** , since a weak prover _𝜇_ may have a lower average of _𝑄[𝜇]_ under its own measure, but still have higher variance across _𝑄[𝜇]_ (compared to _𝑄[𝜋]_ ) when evaluated under _𝜋_ (see Proposition F.1 in App. F.5 for formal discussion). This tells us that _**rewarding progress under a prover is different from typical knowledge distillation or imitation learning algorithms**_ (Hinton, 2015; Rusu et al., 2015) that in most cases remain upper bounded by the performance of the stronger teacher. So provers cannot be characterized purely by strength, what is a class of provers that is a reasonable starting point if we were to improve any base policy _𝜋_ ? 

**The policy class of “Best-of-K” (computed over base policies) contain complementary provers.** A good starting point to identify good provers for a base policy _𝜋_ , is the class of Best-of-K policies or BoK( _𝜋_ ). Recall from Sec. 2 that the performance of BoK( _𝜋_ ) increases monotonically with _𝐾_ . Applying Theorem 3.1 to this class, we arrive at Remark 3.1 that recommends using BoK( _𝜋_ ) with _𝐾>_ 1 as a prover policy for a poor base policy _𝜋_ . However, _𝐾_ cannot be too large always since when _𝑄[𝜋]_ ( _𝒔, 𝑎_ ) ≈ 1 , increasing _𝐾_ too much can hurt distinguishability of different steps at that state. In the next section, we empirically note that the policies in the class of BoK( _𝜋_ ) indeed induce different performance gains when used as prover policies, and we find Bo4 to be a good choice for test-time search over most base policies. 

**Remark 3.1.** _When 𝑄[𝜋]_ ( _𝒔, 𝑎_ ) = _𝑂_ (1/ _𝐾_ ) _,_ ∀ _𝒔, 𝑎, using_ BoK( _𝜋_ ) _as a prover for base 𝜋 improves distinguishability (and improvement) by_ Ω( _𝐾_[2] ) _, and make alignment worse at most by 𝑂_ ( _𝐾_ ) _._ 

## Takeaway: Formal characterization of good prover policies that improve the base policy. 

Provers with advantages that can **distinguish** actions taken by the base policy (more strongly than the base policy itself) but are **not too misaligned** from the base, boost improvements on each update of the base policy. We call such policies _**complementary provers**_ . BoK( _𝜋_ ) for any base policy _𝜋_ for _𝐾>_ 1 can provide a good starting choice of prover policies. 

## **4. Results: Scaling Test-Time Compute with PAVs** 

Now, we study how process verifiers can scale up test-time compute. While our derivations from Sec. 3.2 were with RL, we can also use the _effective reward 𝑄[𝜋]_ ( _𝒔ℎ, 𝑎ℎ_ ) + _𝛼_ · _𝐴[𝜇]_ ( _𝒔ℎ, 𝑎ℎ_ ) for running beam search over intermediate steps sampled from base policy _𝜋_ . To do so, we train a process advantage verifier to predict _𝐴[𝜇]_ , along with a process reward model _𝑄[𝜋]_ . PAV training is done using procedures discussed in Sec. 4.2. While the candidates of the beam are selected using a combination of both the PAV and the PRM _𝑄[𝜋]_ , 

9 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Gemma-2B Gemma-9B Gemma-27B<br>ORM ORM ORM<br>0.4 PAV (ours) 0.6 PAV (ours) PAV (ours)<br>0.6<br>Pass @N Pass @N Pass @N<br>0.3 8%<br>0.5 10% 1.5 ×<br>5 × 10% 2 × 0.5<br>0.2<br>0.4<br>0.1 0.4<br>(a) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] (b) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] (c) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7]<br>N N N<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Figure 4** | _**For test-time search, PAVs are**_ 8 − 10% _**more accurate and**_ 1 _._ 5 − 5× _**more compute efficient over ORMs:**_ On samples from **(a)** Gemma-2B , **(b)** 9B , and **(c)** 27B SFT policies, we run test-time beam search with the estimate of effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ (PAV), where _𝜇_ is the Bo4( _𝜋_ ) policy. We compare beam search performance with best-of-N, re-ranking with a trained outcome verifier (ORM), or the oracle Rex (Pass @N). 

the final candidate is selected using the outcome reward prediction from _𝑄[𝜋]_ itself (i.e., we repurpose the PRM representing _𝑄[𝜋]_ as an ORM). For clarity, we abuse notation and refer to the estimated effective reward (ORM + _𝛼_ PAV) as PAV directly. 

**Setup** . We finetune Gemma 2B, 9B, and 27B (Gemma Team et al., 2024) on MATH (Hendrycks et al., 2021) via supervised fine-tuning (SFT) to get three base policies. The set of provers consists of the three base SFT policies themselves as well as their best-of-K policies for different values of _𝐾_ ∈{2[0] _, . . . ,_ 2[5] }. Additional details for the experiments in this section are in App. C. 

## **4.1. PAVs Scale Test-Time Compute by** 5 − 10× **Over ORMs** 

**Result 1: PAVs are more compute efficient than ORMs.** In Fig. 4, we plot the performance of beam search with PAVs for different sizes of the beam _𝑁_ , and compare it with best-of- _𝑁_ using ORMs, _i.e._ , sampling _𝑁_ complete solutions from the base policy and returning the one with the highest ORM score. To compare PAVs and ORMs, we evaluate the _compute efficiency_ of PAVs over ORMs, given by the ratio of total compute needed by PAVs to obtain the same performance as running best-of-128 with ORM. Even when accounting for the fact that running beam search with PAVs does require additional compute _per solution trace_ (since each element in the beam samples _𝐶_ = 3 next steps, before scoring and pruning the beam), PAVs are able to scale the compute efficiency by **10×** over ORMs for Gemma-2B, 9B base models, and by 5× for Gemma-27B model. We use BoK( _𝜋_ ) with _𝐾_ = 4 as the prover policy for all base policies _𝜋_ . 

We also compare performance with beam search using process verifiers that only predict _𝑄[𝜋]_ , and best-of-N where the ORM is replaced with PAV (PAV-as-ORM). At _𝑁_ = 128, similar to Luo et al. (2024), we note a similar gain of 4% for “PAV-as-ORM” Fig. 5(a) over only ORMs, for base Gemma-9B _𝜋_ . When comparing beam search with _𝑄[𝜋]_ (Snell et al., 2024), we find that PAVs scale compute efficiency by 8×. Evidently, advantages from the prover in the effective reward positively impact the beam search. Why does _𝐴[𝜇]_ help, and for what choice of the prover _𝜇_ ? 

**Result 2: Beam search with too weak/strong provers is sub-optimal.** In Fig. 5(b), for the setting when the base policy _𝜋_ is a Gemma-2B SFT model, we compare beam search with PAVs where the provers are given by BoK( _𝜋_ ), for different values of _𝐾_ . Recall that as _𝐾_ increases, BoK( _𝜋_ ) becomes stronger. Corroborating our analysis in Sec. 3.4, our results show that neither too weak (Bo2) or too strong (Bo32) provers perform best. Instead, across all values of _𝑁_ , we find Bo4 to be dominant. The advantage values 

10 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0011-01.png)


**----- Start of picture text -----**<br>
Comparing PRMs Choice of Prover Base × Prover<br>0.55<br>Provers<br>40<br>0.50 0.25 2B 27B<br>30<br>9B<br>0.45 0.20 20<br>ORM<br>0.40 10<br>PAV-as-ORM Bo2 Bo16<br>0.15 0<br>PRM (Q ) Bo4 Bo32<br>0.35<br>PAV (ours) Bo8 10<br>0.10<br>0.30<br>(a) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] 2 [8] (b) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] (c) 2B 9B 27B<br>N N Base Policy<br>Accuracy Accuracy<br>% Improved Over ORM<br>**----- End of picture text -----**<br>


**Figure 5** | _**Comparing PAVs with search baselines and ablating over the prover policy:**_ **(a):** We compare beam search over Gemma 9B SFT, using either effective reward (PAV), or _𝑄[𝜋]_ (Snell et al., 2024), and report best-of-N performance where the re-ranker is either the ORM or PAV-as-ORM. **(b):** For the base Gemma 2B SFT policy, we run beam search with the effective reward where the prover is BoK( _𝜋_ ) for different values of _𝐾_ . In both ( _𝑎_ ) _,_ ( _𝑏_ ) the x-axis scales the size of the beam or _𝑁_ for best-of-N. **(c):** For each base policy in the set: Gemma 2B, 9B, 27B policies, we run beam search with PAVs (beam size of 16) where the prover is another policy from the same set. 

_𝐴[𝜇]_ ≈ 0 on all steps for very large _𝐾_ , since _𝑄[𝜇]_ ( _𝒔ℎ, 𝑎ℎ_ ) = 1 −(1 − _𝑄[𝜋]_ ( _𝒔ℎ, 𝑎ℎ_ )) _[𝐾]_ → 1 on all steps, as we increase _𝐾_ . Hence, _**in order to succeed we need an intermediate-level prover policy**_ . 

We make similar observations in Figure 5(c) where we use the three base policies (Gemma 2B/9B/27B) as provers for training PAVs. In this scenario, we evaluate beam search with PAVs at _𝑁_ = 16 on top of different base policies. We find that for the 2B and 9B base models, the 9B and 27B provers are most effective respectively, whereas for the 27B model, _**surprisingly a weaker 9B policy is more effective than the stronger 27B model.**_ The weaker model presumably offers a complementary signal that distinguishes between different actions taken by 27B, aligning with our theoretical observations in Sec. 3.4. 

**Result 3: Advantages from the prover policy enable exploration.** As discussed in Sec. 3.1, advantage _𝐴[𝜇]_ measures the progress made by an action agnostic of the value of the previous state, where as _𝑄[𝜋]_ measures the promise of a particular state. Given a finite capacity beam, our effective reward (Eq. 5), which linearly combines _𝑄[𝜋] and 𝐴[𝜇] induces a better tradeoff between exploring new prefixes (states) from where progress can be made and exploiting currently known prefixes with high Q-values._ Exploration at 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0011-06.png)


**----- Start of picture text -----**<br>
Pass @N over Gemma 2B Base Scaling Laws: nmc/ncov<br>0.5 n mc /n cov<br>0.4 2<br>4<br>0.4<br>8<br>0.3<br>16<br>IID Sampling<br>0.2 Search: PAV (ours) 0.3 32<br>Search: PRM (Q )<br>(a) 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] (b) 64k 128k 256k 512k 1024k<br>N Training Data Size<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Figure 6** | **(a):** Beam search with PAVs improves exploration efficiency (higher Pass@N), over typical PRMs. **(b):** Performance of beam search over Gemma 9B SFT for PAVs trained on datasets with different _𝑛_ mc/ _𝑛_ cov. 

11 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

initial steps is critical to ensure that the beam at later steps covers diverse partial rollouts each with a high likelihood of producing the correct answer. Thus over-committing to the beam with actions from the same state, regardless of the progress made by each can prove to be sub-optimal over a selection strategy that balances rewarding previous actions _𝐴[𝜇]_ and current states _𝑄[𝜋]_ . Indeed, we observe in Fig. 6(a), beam search with PAV enhances pass@N performance vs. beam search with _𝑄[𝜋]_ and _i.i.d._ sampling. 

## Takeaways: Scaling test-time compute with process advantage verifiers. 

- Beam search with PAVs boosts accuracy by >8% & compute efficiency by 1.5-5x over ORMs. 

- Utilizing Best-of-K policies (corresponding to the base policy) as provers induce better exploration to maximize outcome reward. Optimal provers for a base policy appear at _𝐾>_ 1. 

## **4.2. How to Collect Data to Train PAVs?: PAV Training Data Scaling Laws** 

We now describe the procedure for training outcome verifiers and PAVs. We can learn to predict _𝑄[𝜋]_ for a policy _𝜋_ (similar for _𝑄[𝜇]_ ) by finetuning LLMs with a cross-entropy loss on the following data with triplets ( _𝒔, 𝑎, 𝑄[𝜋]_ mc[(] _[𝒔][, 𝑎]_[))][.][To collect this data, we first sample] _[𝑛]_[cov][“seed” rollouts from the base or prover] policy respectively for ORM and PAVs, to promote coverage over prefixes and steps. Then we sample _𝑛_ mc additional rollouts, conditioned on each prefix in the seed rollout to compute the Monte-Carlo estimate of _𝑄[𝜋]_ at each prefix. In Fig. 6(b) we plot the beam search performance of PAVs trained with different ratios of _[𝑛]_[mc] / _𝑛_ cov, as we scale the total dataset size. Here, the beam size is fixed to 128 and the base policy is the Gemma 9B SFT policy and prover is _𝐵𝑜_ 4 policy. **We find that** under low sampling budgets, optimizing for coverage ( _𝑛_ cov _> 𝑛_ mc) is better for performance, and when budget is higher, reducing label noise in _𝑄[𝜋]_ mc[by setting] _[𝑛]_[mc] _[>][𝑛]_[cov][gets us more improvements.][In addition, we also spend some initial sampling] budget is spent to identify “high value” states where _𝑄[𝜋]_ is larger than a threshold, and identify the first step with low _𝑄[𝜋]_ on an incorrect partial rollout from this state. We found this strategy to scale better with dataset size, as we discuss in App. D. 

## **5. Results: Scaling Dense-Reward RL with PAVs** 

We can also use PAVs to train policies via online reinforcement learning (RL), by using the effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ as dense, per-step rewards. We compare the sample efficiency of PAV-RL (i.e., _ℓ_ PAV−RL in Eq. 4) with standard ORM-RL (i.e., _ℓ_ ORM−RL in Eq. 3) on Gemma 2B and Gemma 9B SFT models, which are further optimized via rejection finetuning (RFT) (Yuan et al., 2023), before using them to initialize RL. To our knowledge, no prior work has successfully demonstrated the use of dense per-step feedback with a process reward model for RL, and we present the first significant set of results establishing the efficacy of this approach. We show that PAV-RL is much more sample-efficient, and enjoys a higher ceiling on the performance of any test-time re-ranker. Additional details for the experiments are in App. E. 

**Result 1: PAV-RL is** _>_ 7% **better than ORM-RL in test accuracy, and** 6× **sample efficient.** In Fig. 7(a), we report the test accuracies of Gemma 2B and 9B models trained with SFT, RFT, ORM-RL and PAV-RL. PAV-RL improves the RFT policy by 11% for 2B, and 15% for 9B, with _>_ 7% gain over ORM-RL in both cases. Not only do the effective rewards from PAV improve the raw accuracy after RL, this higher accuracy is attained 6× faster (see Fig. 7(b)) for the 2B run and similarly for the 9B RL run (Fig. 7(c)). For both 2B and 9B, RL runs, we experiment with two options for the prover policy: **(i)** 2B SFT policy; and **(ii)** 9B SFT policy. While both of these provers rapidly become weaker than the base policy within a few gradient steps of RL, a fixed PAV trained with each of these provers is able to still sustain performance gains in RL. More interestingly, we find that the 2B SFT policy serves as the best choice of the prover for both 2B and 

12 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0013-01.png)


**----- Start of picture text -----**<br>
PAVs vs. ORMs Sample Efficiency (2B) Sample Efficiency (9B)<br>SFT 0.55<br>0.5 RFT 0.25 6%<br>0.4 ORM-RL 7% 0.50 5 ×<br>PAV-RL 6 ×<br>0.3 0.20<br>0.45<br>0.2<br>0.15 ORM-RL ORM-RL<br>0.1 0.40<br>PAV-RL PAV-RL<br>0.0<br>(a) 2B 9B (b) 0 1 2 3 4 5 6 7 8 9 10 (c) 2 10 18 26 34 42 50<br>Base Policy Training Iterations (×10 [3] ) Training Iterations (×10 [2] )<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


**Figure 7** | _**PAVs as dense rewards in RL improve sample efficiency compared to ORMs, along with gains on raw accuracy:**_ **(a)** We report the performance of a base policy trained using RL with effective rewards (PAV-RL), or only outcome rewards (ORM-RL), and baselines SFT, RFT. **(b,c):** Across training iterations, we report the test performance of policies trained with PAV-RL and ORM-RL, on Gemma 2B and 9B SFT base policies. 

9B policies. This observation that a weak prover can still improve the base policy corroborates our results in the didactic setup and our analysis in Sec. 3.4. While we were not able to run experiments where the prover policy is dynamically updated on the fly, we believe that updating the prover through the process of RL training should only amplify these benefits. 

**Result 2: PAV-RL achieves higher performance ceiling on test-time re-ranking.** In Fig. 8(a), for Gemma 2B, we plot the Pass @N performance for each method, and find **(i)** Pass @N is higher ( _>_ 7%) for PAV-RL, compared to ORM-RL, for any _𝑁_ ≤ 128; and **(ii)** the rate at which Pass @N improves for PAV-RL is higher than ORM-RL. Both trends are consistent with our observations on the didactic example in Sec. 3.3. Notably, for _𝑁_ ≥ 64, ORM-RL is worse than the SFT policy, perhaps due to lower entropy over the distribution at the next step resulting in non-diverse candidates. Why does PAV-RL produce diverse candidates, and does not suffer from the low diversity problem in ORM-RL? We answer this with a key insight on how _**the primary benefit of PAVs is to promote efficient exploration.**_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0013-05.png)


**----- Start of picture text -----**<br>
Pass @N (2B) Solving Hard Questions<br>0.5 0.15<br>ORM<br>PAV<br>0.4<br>0.10<br>0.3<br>SFT 0.05<br>RFT<br>0.2<br>ORM-RL<br>0.00<br>PAV-RL<br>2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] 2 [8]<br>(a) (b)<br>N N<br>Accuracy<br> Unsolved by SFT @256<br>Success Rate on Problems<br>**----- End of picture text -----**<br>


**Figure 8** | **(a):** For the policies trained in (a) we report the best-of-N performance where the oracle reward Rex is used to rank _𝑁_ candidates sampled from the base policy (Pass @N). **(b):** Amongst hard problems that remain unsolved by Best-of-256 over the base SFT policy, we check how many are solved by Best-of-N over PAV-RL or ORM-RL. PAV-RL is able to solve a substantially more problems than what ORM-RL was able to solve. 

13 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

**Result 3: PAVs improve exploration and discover correct solutions to novel problems.** An outcome reward model rewards downweight all steps in an incorrect rollout equally during RL, whereas the effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ in PAVs, up-weights steps that make progress under the prover, even when the complete rollout is incorrect. This increases the coverage over individual steps that can improve the likelihood of the base policy to succeed (since the prover policy is not too misaligned with the base policy). These can now be proposed by the base policy at a given prefix. This mechanism for exploration is analogous to test-time search we discussed in Sec. 4.1. Hence, the directed supervision from PAVs improves sample-efficiency throughout the course of training (Fig. 7(c)). In fact, we also find that combining the PAV-RL policy with test-time beam search is able to solve a substantially larger number of _new_ problems within smaller compute budgets ( _𝑁_ = 16 _,_ 32) that the SFT policy cannot solve with a much larger budget _𝑁_ = 256 (Fig. 8(b)). 

## Takeaway: RL with process advantage verifiers (PAVs) as dense rewards 

- Using trained PAVs as dense rewards in RL boosts scales sample efficiency by 5 − 6×, compared to only using sparse ORM rewards, and results in policies with a higher Pass @N performance. 

- Advantages from a complementary prover policy improves the sample efficiency of exploration in RL, and produces policies that can discover solutions to hard novel questions. 

## **6. Related Work** 

We briefly discuss some key related works here, and leave the detailed discussion for App. A. To address issues of sparse feedback in ORMs (Cobbe et al., 2021b), recent works (Lightman et al., 2023; Uesato et al., 2022) trained process reward models (PRMs) to densely predict incorrect steps in a multi-step reasoning trace. Since human data collection for process labels is not scalable enough, recent work (Luo et al., 2024; Wang et al., 2024) used automated supervision to annotate steps with _𝑄_ values under the base policy, _i.e._ , the PRMs score a step with the likelihood of future success, when continuing to sample from the step. While _𝑄_ -value PRMs in Lightman et al. (2023); Luo et al. (2024) were mainly used as verifiers for re-ranking, Snell et al. (2024) used them for test-time beam search. Shao et al. (2024) uses PRMs for RL but found a gain of only 1 − 2% with PRMs. In our work, we question solely relying on _𝑄_ -values or advantages of the base policy, and find that measuring progress (i.e., advantages) under a different prover policy can amplify exploration, thus boosting test-time search and RL. To our knowledge, we are the first to show substantial gains in compute and sample efficiency with PRMs. Our methodology for data collection is similar to (Hwang et al., 2024; Setlur et al., 2024) (i.e., identify “first pits” in reasoning traces), these works only use it to collect preference pairs. Beyond all of these, we also characterize which policy to use for computing advantages. 

Concurrently to us, akin to the methodology in Hwang et al. (2024); Setlur et al. (2024), Kazemnejad et al. (2024) optimize the base policy _𝜋_ with online RL, where the dense step-level rewards correspond to advantages _𝐴[𝜋]_ under the base policy _𝜋_ itself. This is a special case of our setting, where the prover policy _𝜇_ = _𝜋_ , but as we note in Sec. 3.2, setting _𝜇_ = _𝜋_ in our effective reward (Eq. 5) results in exactly the same policy gradient updates as only optimizing the outcome reward. Since Kazemnejad et al. (2024) use “on-the-fly” Monte-Carlo rollout estimation to estimate advantages, they are able to avoid estimation errors in the process reward model. Nonetheless, our theoretical result and the didactic example (both of which assume access to perfect advantage estimates) show that gains from this approach are significantly smaller than using an appropriate prover policy, which is distinct from the base policy. 

14 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

## **7. Discussion and Conclusion** 

We began our exposition with the following question: how to define process rewards such that optimizing the base policy against process rewards ultimately improves the outcome level correctness of the final answer? Our key finding is that process rewards defined as advantages of a prover policy, distinct from the base policy improve the efficiency of exploration for steps sampled from the base policy during test-time search and online RL. This improved exploration in turn leads to discovery of better solutions, resulting in a higher accuracy on the math reasoning task. We also formally characterized the set of good prover policies as policies with step-level advantages that meaningfully contrast steps generated by the base policy, while still producing step-level advantages that are aligned with the base policy. Having trained process advantage verifiers (PAVs) to predict advantages under the prover policy, we empirically observed that test-time search against the trained PAVs improve the compute-efficiency of search by 1 _._ 5 − 5×, and accuracy of search by over 8% compared to running best-of- _𝑁_ against an ORM. Next, we present one of the significant results that validate the use of dense supervision when optimizing the base policy with online RL. Specifically, we show that dense online RL with rewards from our trained PAVs, improves sample efficiency of online RL by 5 − 6×, and results in an accuracy gain of over 6%. 

**Limitations.** Despite the promise of our results, there are several limitations to our work that present important avenues for future research. First, while we can easily compute the right hand side of our result in Theorem 3.1 to understand whether a given prover policy will improve a fixed base policy, it is unclear how to automatically design a flexible class of optimal (or very good) prover policies for a sequence of base policy iterates. Perhaps simultaneously optimizing the prover and the base policy (in a two-player game) might provide for an approach to obtain the best prover during RL, but this is largely an open question. Second, since inevitably learning a process advantage verifier (PAV) will incur fitting errors and this upper bounds peformance of our method. Fitting errors can be circumvented if our approach if we can simply run rollouts from prover policies during online RL or search to estimate advantages without training verifiers, and extending our approach to this setup is a good avenue for future work. 

## **Acknowledgements** 

The authors would like to thank Charlie Snell, Yi Su, Katherine Heller, and Virginia Smith for feedback on an earlier version of this paper. We also thank Ahmad Beirami, Sergey Levine, Victor Veitch, Idan Shenfeld, Arian Hosseini, Stephen Pfohl, Xiangyu Qi, Tianhe Yu, and Christina Baek for technical discussions. AS and CN also thank Preston Robinette, Sho Kannan, Tianze Shi, Diana Mincu, Hritik Bansal, and Liangchen Luo for code, infrastructure and data analytics support. 

## **References** 

- A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. _Journal of Machine Learning Research_ , 22(98):1–76, 2021. 

- T. Anthony, Z. Tian, and D. Barber. Thinking fast and slow with deep learning and tree search. _Advances in neural information processing systems_ , 30, 2017. 

- M. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos. Unifying count-based exploration and intrinsic motivation. In _Advances in Neural Information Processing Systems_ , pages 1471–1479, 2016. 

15 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

- X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. _arXiv preprint arXiv:2401.02954_ , 2024. 

- J. D. Chang, K. Brantley, R. Ramamurthy, D. Misra, and W. Sun. Learning to generate better than your llm. _arXiv preprint arXiv:2306.11816_ , 2023. 

- K.-W. Chang, A. Krishnamurthy, A. Agarwal, H. Daumé III, and J. Langford. Learning to search better than your teacher. In _International Conference on Machine Learning_ , pages 2058–2066. PMLR, 2015. 

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021a. 

- K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021b. 

- M. Collins. Discriminative reranking for natural language parsing. In _Proceedings of the International Conference on Machine Learning_ , 2000. 

- Gemma Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, et al. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ , 2024. 

- M. Germain, K. Gregor, I. Murray, and H. Larochelle. Made: Masked autoencoder for distribution estimation. In _International conference on machine learning_ , pages 881–889. PMLR, 2015. 

- A. Havrilla, Y. Du, S. C. Raparthy, C. Nalmpantis, J. Dwivedi-Yu, M. Zhuravinskyi, E. Hambro, S. Sukhbaatar, and R. Raileanu. Teaching large language models to reason with reinforcement learning. _arXiv preprint arXiv:2403.04642_ , 2024. 

- D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. _NeurIPS_ , 2021. 

- G. Hinton. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_ , 2015. 

- A. Hosseini, X. Yuan, N. Malkin, A. Courville, A. Sordoni, and R. Agarwal. V-star: Training verifiers for self-taught reasoners. _arXiv preprint arXiv:2402.06457_ , 2024. 

- H. Hwang, D. Kim, S. Kim, S. Ye, and M. Seo. Self-explore to avoid the pit: Improving the reasoning capabilities of language models with fine-grained rewards. _arXiv preprint arXiv:2404.10346_ , 2024. 

- S. Kakade and J. Langford. Approximately optimal approximate reinforcement learning. In _Proceedings of the Nineteenth International Conference on Machine Learning_ , pages 267–274, 2002. 

- S. M. Kakade. A natural policy gradient. In _Advances in neural information processing systems_ , volume 14. Advances in neural information processing systems, 2001a. 

- S. M. Kakade. A natural policy gradient. _Advances in neural information processing systems_ , 14, 2001b. 

- A. Kazemnejad, M. Aghajohari, E. Portelance, A. Sordoni, S. Reddy, A. Courville, and N. L. Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. _arXiv preprint arXiv:2410.01679_ , 2024. 

16 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

- H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let’s verify step by step. _arXiv preprint arXiv:2305.20050_ , 2023. 

- L. Luo, Y. Liu, R. Liu, S. Phatale, H. Lara, Y. Li, L. Shu, Y. Zhu, L. Meng, J. Sun, et al. Improve mathematical reasoning in language models by automated process supervision. _arXiv preprint arXiv:2406.06592_ , 2024. 

- Q. Ma, H. Zhou, T. Liu, J. Yuan, P. Liu, Y. You, and H. Yang. Let’s reward step by step: Step-level reward model as the navigators for reasoning. _arXiv preprint arXiv:2310.10080_ , 2023. 

- R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. _arXiv preprint arXiv:2112.09332_ , 2021. 

- A. Y. Ng, D. Harada, and S. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In _ICML_ , volume 99, pages 278–287, 1999. 

- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_ , 35:27730–27744, 2022. 

- R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. _arXiv preprint arXiv:2305.18290_ , 2023. 

- S. Ross and J. A. Bagnell. Reinforcement and imitation learning via interactive no-regret learning. _arXiv preprint arXiv:1406.5979_ , 2014. 

- A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell. Policy distillation. _arXiv preprint arXiv:1511.06295_ , 2015. 

- A. Setlur, S. Garg, X. Geng, N. Garg, V. Smith, and A. Kumar. Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold. _arXiv preprint arXiv:2406.14532_ , 2024. 

- Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_ , 2024. 

- A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, P. J. Liu, J. Harrison, J. Lee, K. Xu, A. Parisi, et al. Beyond human data: Scaling self-training for problem-solving with language models. _arXiv preprint arXiv:2312.06585_ , 2023a. 

- I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg. Progprompt: Generating situated robot task plans using large language models. In _2023 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 11523–11530. IEEE, 2023b. 

- C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. _arXiv preprint arXiv:2408.03314_ , 2024. 

- W. Sun, A. Venkatraman, G. J. Gordon, B. Boots, and J. A. Bagnell. Deeply aggrevated: Differentiable imitation learning for sequential prediction. In _International conference on machine learning_ , pages 3309–3318. PMLR, 2017. 

17 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

- R. S. Sutton and A. G. Barto. _Reinforcement learning: An introduction_ . The MIT Press, second edition, 2018. 

- R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. _Advances in neural information processing systems_ , 12, 1999. 

- F. Tajwar, A. Singh, A. Sharma, R. Rafailov, J. Schneider, T. Xie, S. Ermon, C. Finn, and A. Kumar. Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data, 2024. 

- J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins. Solving math word problems with process-and outcome-based feedback. _arXiv preprint arXiv:2211.14275_ , 2022. 

- P. Wang, L. Li, Z. Shao, R. X. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024. 

- R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine learning_ , 8(3-4):229–256, 1992. 

- Y. Wu, Z. Sun, S. Li, S. Welleck, and Y. Yang. An empirical analysis of compute-optimal inference for problem-solving with language models. _arXiv preprint arXiv:2408.00724_ , 2024. 

- F. Yu, A. Gao, and B. Wang. Outcome-supervised verifiers for planning in mathematical reasoning. _arXiv preprint arXiv:2311.09724_ , 2023. 

- Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. Scaling relationship on learning mathematical reasoning with large language models. _arXiv preprint arXiv:2308.01825_ , 2023. 

- E. Zelikman, Y. Wu, J. Mu, and N. Goodman. Star: Bootstrapping reasoning with reasoning. _Advances in Neural Information Processing Systems_ , 35:15476–15488, 2022. 

- L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, and R. Agarwal. Generative verifiers: Reward modeling as next-token prediction. _arXiv preprint arXiv:2408.15240_ , 2024. 

18 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

## **Appendices** 

## **A. Additional Related Work** 

In this section, we highlight works from four relevant streams, expanding on discussion in Section 6. First, we look at works that train verifiers to provide outcome level feedback (Cobbe et al., 2021b; Hosseini et al., 2024; Singh et al., 2023b; Zelikman et al., 2022) on the correctness of the full response (ORM). Here, the trained ORMs are mainly used for test-time search (best-of- _𝑁_ ). Next, we look at works that alleviate issues with sparse feedback in ORMs, and instead train process reward models (PRMs), that can perform credit assignment. PRMs are trained either through human annotations (Lightman et al., 2023; Uesato et al., 2022), or automated forms of supervision (Luo et al., 2024; Snell et al., 2024; Wang et al., 2024). While some works use PRMs and ORMs to collect data for supervised fine-tuning Hosseini et al. (2024) or offline RL Setlur et al. (2024), other works directly use them as rewards in online RL (Shao et al., 2024; Uesato et al., 2022; Wang et al., 2024). Finally, we contrast our work against papers on imitating stronger teacher policies via RL objectives that optimize potential functions of teacher policies. 

**Outcome reward models.** ORMs are verifiers (Cobbe et al., 2021b; Uesato et al., 2022) commonly used to improve the test-time performance using best-of- _𝑁_ , where we generate multiple candidate solutions from the base policy (LLM), rank them using the ORM, and pick the best one. ORMs are trained to assess correctness of a solution either using binary classification (Cobbe et al., 2021a; Yu et al., 2023), preference optimization using DPO (Hosseini et al., 2024), or next-token prediction (Zhang et al., 2024). Furthermore, prior works train LLMs on self-generated data using ground-truth outcome rewards (Rex), either via supervised fine-tuning (Singh et al., 2023a; Yuan et al., 2023; Zelikman et al., 2022), or online RL (Bi et al., 2024). In contrast to these approaches, our work focuses on process reward models (PRMs) for improving performance with beam-search at test time as well as online RL where we maximize the effective reward in Eq. 5 which linearly combines both Rex (outcome supervision) and process supervision in the form of advantages _𝐴[𝜇]_ under a prover policy _𝜇_ . 

**PRMs and credit assignment.** Several works focus on training step-level PRMs on math reasoning tasks, either using human labels (Lightman et al., 2023) or automated LLM-generated data to estimate value functions _𝑄[𝜋]_ (Luo et al., 2024; Wang et al., 2024). Our work also focus on automated data collection for PRMs but empirically argues for using the advantage function _𝐴[𝜇]_ as step-level rewards along with _𝑄[𝜋]_ , with a conceptual explanation in Section 3.1. Several prior works have explored step-level search algorithms with PRMs, such as beam search (Snell et al., 2024), heuristic greedy search (Ma et al., 2023), and reward-balanced tree search (Wu et al., 2024). Hwang et al. (2024); Setlur et al. (2024) use advantages to identify the “first pit” in an incorrect reasoning trace. Specifically, they collect data by computing advantages at each step using Monte Carlo rollouts. Then in an incorrect trace, they identify the step with the least advantage, and use the prefix of that step to construct preference pairs for offline direct preference optimization (Rafailov et al., 2023). In contrast, our work computes advantages under a prover policy, that we formally characterize, and use the computed advantages for improving test-time search and efficiency of online reinforcement learning. 

**Online RL for math reasoning.** Once we have a trained outcome or process verifiers, it is natural update a policy by optimizing it against the learned signal, similar to how learned reward models are optimized in RLHF (Ouyang et al., 2022). In the context of math reasoning, Havrilla et al. (2024); Shao et al. (2024); Uesato et al. (2022) trained policies with RL, experimenting with both dense and sparse 

19 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

|Planted sub-sequence𝒚⋆:|||||||
|---|---|---|---|---|---|---|
|Samples from policy corresponding to𝛾=||15|||Samples from policy corresponding to𝛾= 100||
||||||||
||||||||
||||||||



**Figure 9** | _**Pictorial description of our planted sub-sequence didactic setup**_ **:** An example showing five samples drawn _i.i.d._ from a very strong policy ( _𝛾_ = 100), and a relatively weaker ( _𝛾_ = 15) policy in our didactic setup. 

rewards. In all three works, the gains observed by using PRMs that predict step-level correctness (similar to Lightman et al. (2023)) is quite small, compared to simply using trained ORMs, or the ground-truth outcome supervision Rex. In fact, Havrilla et al. (2024) states that the only algorithm that does well is a form of expert iteration (Anthony et al., 2017), which does not inhibit exploration as severely as some other approaches they compare with. Our work presents one of the first results, where trained PRMs, used in conjunction with the outcome rewards during online RL, result in policies with substantially higher (+6%) performance, than the one trained only with outcome supervision. Our results also indicate a 5 − 6× sample efficiency boost for online RL, with our trained PAVs. 

**Connections to imitation learning through RL.** The idea of mixing potential functions from different policies _𝜇_ and _𝜋_ , in order to improve upon a sub-optimal expert _𝜇_ appears in Chang et al. (2015), but this work considers the structured prediction problem which is vastly different from our setting. Related to this, is the work by Chang et al. (2023), which uses a “guide” policy to rollout from prefixes generated by a base policy. The base policy can now imitate the guide by cloning those rollouts, and eventually surpass. Our work also uses a prover policy which can complete rollouts from states where the base policy fails. But, we also show that weak provers in many cases are able to improve the base policy, or search over its responses, better than a stronger prover policy. We tie this observation to the insight that the main goal of the prover policy is to distinguish steps taken by the base policy, as measured by advantages under the prover. Thus, we do not require the prover policy to be something better than the base policy, which is a key distinction with Chang et al. (2023). 

## **B. Didactic Analysis** 

We consider sequences of length 10 from a 15-token vocabulary V � {1 _,_ 2 _, . . . ,_ 14}, where the end-ofsequence token is given by 14, and all tokens following the end-of-sequence token (including it) are masked. Given an unknown planted sequence _𝒚[★]_ (in Fig. 9), we train a policy _𝜋_ with policy gradient, where the outcome reward we wish to optimize is terminal and sparse, _i.e._ , for _𝒚_ ∼ _𝜋_ we have _𝑟_ ( _𝒚, 𝒚[★]_ ) = 1 if and only if _𝒚[★]_ appears in _𝒚_ , and 0 otherwise (Fig. 9). The policy _𝜋_ in our experiments is represented by a multi-layer neural network, similar to the MADE architecture (Germain et al., 2015). The prover policy _𝜇_ is parameterized by a scalar _𝛾>_ 0. In particular, at any state _𝒔_ , where the last _𝑘_ tokens leading up to _𝒔_ match first _𝑘_ tokens of _𝒚[★]_ , then: 

_𝜇_ ( _𝒚𝑘[★]_ +1[|] _[𝒔]_[)][∝] _[𝛾,]_ 

and uniform on all other tokens. Thus, as _𝛾_ increases, the performance of _𝜇_ improves and → 1 as _𝛾_ →∞. For our experiments, we assume (almost) oracle access to ground-truth advantage and _𝑄_ -values, thus 

20 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

mitigating any confounding issues due to usage of a learned verifier. We are able to approximate exact _𝑄_ -values very accurately by using Monte Carlo estimates with large _>_ 100 rollouts. With the goal of optimizing the terminal reward _𝑟_ ( _𝒚, 𝒚[★]_ ), we optimize _𝜋_ with two types of rewards: (i) only the outcome reward _𝑟_ ( _𝒚, 𝒚[★]_ ), which is equivalent to using only _𝑄[𝜋]_ as step-level rewards; and (ii) using the effective reward: _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ as the step-level reward. 

**Training details.** We use effective rewards with _𝛼_ = 1, and use the gradient in Eq. 5 to update the policy via policy gradient iterations. For the ORM runs, where we only use the outcome reward _𝑟_ ( _𝒚, 𝒚[★]_ ), the policy gradient is equivalent to the case where _𝛼_ = 0 in Eq. 5. We train for 10,000 iterations in both cases, with a batch size of 64, and a constant learning rate of 1 _𝑒_ − 3 for the Adam optimizer. The RL runs are initialized with a supervised finetuned policy. For this we take a randomly initialized network, based on the MADE architecture (Germain et al., 2015), with 3 layers, and 128 hidden units in each. Then we train it with supervised next-token prediction loss for 50 iterations on a dataset of 3200 samples from a weak policy ( _𝛾_ = 5 _._ 0). The batch size for the SFT training is also set to 64. For evaluating Pass @ _𝑁_ performance, we either sample _𝑁_ independent trajectories (temperature 1.0) from the base policy trained using effective rewards, or only _𝑄[𝜋]_ . We also evaluate Pass @ _𝑁_ for the SFT policy for comparison. 

## **C. Additional: Experiments on Test-time Search with PAVs** 

**Implementation details.** For our experiments in Sec. 4, we use three pretrained models: Gemma 2B, 9B and 27B. We finetune each of these on the MATH (Hendrycks et al., 2021) dataset. The finetuning is done for 5000 iterations, with a batchsize of 32, and a maximum learning rate of 5 _𝑒_ − 6 for 2B, 9B and 5 _𝑒_ − 7 for the 27B models. We trained the policies using the Adam optimizer, with a linear warm up and cosine decay learning rate schedule. The linear warm up is done for the first 500 iterations. For the base policies, we choose the SFT checkpoints with the best accuracy on a holdout validation set of the MATH dataset. Given the SFT checkpoints, we next train PAVs using the procedure in Sec. 4.2. We do this for a class of provers, which include the base policies themselves. As we discuss in Sec. 4, the prover class also includes the best-of- _𝐾_ policy for _𝐾_ in {2 _,_ 4 _,_ 8 _,_ 16 _,_ 32}. 

We use the hold out validation set to ascertain the value of _𝛼_ in the effective reward. For each base policy we run beam search with a beam size of 16 on this hold out validation set, and using the base policy itself as a prover, we evaluate the value of _𝛼_ that works best in the effective reward. We find that _𝛼_ = 0 _._ 5 worked best for Gemma 2B and 9B base policies, while a lower value of _𝛼_ = 0 _._ 2 was optimal for Gemma 27B. To tune _𝛼_ we ran a grid search over the range [0 _._ 0 _,_ 1 _._ 0], evaluating at an interval of 0 _._ 1. We observe that the choice of _𝛼_ is a relatively robust one, since for all three base policies, we saw improvements (over only _𝑄[𝜋]_ as the reward) for values in the range of [0 _._ 2 _,_ 0 _._ 6]. Having a separate value of _𝛼_ for each base policy, we use the same value in the effective reward given by any choice of the prover policy that is used for that base policy. Next, we present an experiment that compares the predictive power of effective reward vs. just _𝑄[𝜋]_ at initial states of a rollout under the base policy _𝜋_ , when either is used to predict the final outcome given by Rex. 

**Experiment: Is the effective reward able to predict the final outcome better than** _𝑄[𝜋]_ **?** In Fig. 10, we describe an experiment where for both the effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ (PAV) and just _𝑄[𝜋]_ (PQV), we compute the error of the classifier that makes a prediction on the final outcome by thresholding on either reward value at each step of the rollout. This threshold is computed using a validation set, and is separate for each step and reward combination. The figure tells us that the outcome prediction error drops for both rewards as the base policy is rolled out more, but clearly the effective reward dominates _𝑄[𝜋]_ (PQV) 

21 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

across all steps. Thus, the effective reward is a more informative signal (lower classification error) for the problem of predicting the success of a partial rollout, especially in the earlier steps of the rollout. This helps to explain the better performance of beam search with a finite capacity beam that re-ranks partial rollouts with the effective reward. For this experiment, we use the Gemma 9B SFT policy as the base policy and the best-of-4 policy corresponding to the same SFT policy as the prover. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0022-02.png)


**----- Start of picture text -----**<br>
Predicting Final Outcome<br>PAV<br>0.8<br>PQV<br>0.6<br>0.4<br>0.2<br>2 3 4 5 6 7 8 9 10<br>Number of steps taken<br>Error<br>**----- End of picture text -----**<br>


**Figure 10** | _**Effective rewards at any step are able to predict the outcome rewards at the final step, better than** 𝑸[𝝅]_ **:** For both the effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ and just _𝑄[𝜋]_ , we compute the error of the classifier that makes a prediction on the final outcome by thresholding on either reward value at each step of the rollout. This threshold is computed using a validation set, and is separate for each step and reward combination. The figure tells us that the outcome prediction error drops for both rewards, as the base policy is rolled out more, but compared to _𝑄[𝜋]_ at an intermediate step, the effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ at an intermediate step is able to reliably predict the final outcome level correctness of the future rollout under the base policy _𝜋_ . 

## **D. Details on Collecting Data and Training PAVs** 

In Fig. 6(b) in Sec. 4.2, our seed rollouts are _i.i.d._ sampled from _𝜋_ , but prior works (Hwang et al., 2024; Luo et al., 2024; Setlur et al., 2024) employed a “first pit” strategy for coverage. Here, some initial sampling budget is spent to identify “high value” states where _𝑄[𝜋]_ is larger than a threshold. Then, for any incorrect sample from these high value states greater budget is spent to estimate the first step (first pit) with _𝑄[𝜋]_ = 0. All prefixes (and their estimated _𝑄_ -values) until the first pit are then added to the training data. In Fig. 11, we compare beam search using PAVs trained using data from the first pit strategy, and the random sampling strategy. Both of them use the best value of _[𝑛]_[mc] / _𝑛_ cov from Fig. 6(b) for every dataset size. We find the first pit strategy to be better than random, especially when the number of seed rollouts are limited. Once we get coverage over such pits, we sample a large number of partial rollouts conditioned on each prefix until the first pit. This is used to compute the Monte Carlo estimate of _𝑄_ values more accurately on the path to the first pit. Each prefix and estimated _𝑄_ value pair is then added to the dataset used to train PAVs. 

**Training details.** All PAVs used in this work are trained by taking the Gemma 9B pretrained checkpoint and finetuning it on the data collected from the above strategy. The data collection uses first pit strategy for better coverage over pits in the seed rollouts. Based on findings in Fig. 6(b), we use a high value of _𝑛_ mc = 20 to estimate the _𝑄_ -values accurately for each step in the seed rollout. For each base policy, 

22 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

in total, we collect a dataset of over 300 _,_ 000 (prefix, _𝑄_[ˆ] _[𝜋]_ -value) pairs. Here, _𝑄_[ˆ] _[𝜋]_ is the Monte Carlo estimate for the _𝑄_ -value at the prefix, under the policy _𝜋_ . on which we finetune the Gemma 9B model with cross-entropy loss. Since the distribution of values for _𝑄_[ˆ] _[𝜋]_ can be skewed, we split the range of _𝑄_ ˆ _[𝜋]_ -values into two buckets, based on which we also partition the training data. The first bucket is the set of all prefixes with _𝑄_[ˆ] _[𝜋] <_ 0 _._ 5 and the second is the set of all prefixes with _𝑄_[ˆ] _[𝜋]_ ≥ 0 _._ 5. Then, we use class-balanced sampling over these buckets to finetune the pretrained model for 20000 training iterations, using a batch size of 32. We use an Adam optimizer with a maximum learning rate of 5 _𝑒_ − 7. We use a linear warm up (till 2000 steps), followed by a cosine decay learning rate schedule to train the models. Since a pretrained LLM would output a matrix of logits (vocabulary size × sequence length) we fix a token as the “scoring token” to be the end of the sequence / prefix that needs to be scored. The logits of this scoring token are then used to determine the prediction for the LLM being trained. 

Once we have models that predict the _𝑄[𝜋]_ for a base policy _𝜋_ , we compute the _𝑄_ -value under the BoK policy corresponding to _𝜋_ , by setting: _𝑄_[BoK][(] _[𝜋]_[)] ( _𝒔, 𝑎_ ) = 1 −(1 − _𝑄[𝜋]_ ( _𝒔, 𝑎_ )) _[𝐾]_ . Next, given the _𝑄_ -values we for _𝜋_ , and their corresponding best-of- _𝐾_ policies, we can compute any effective reward in Eq. 5, using the definition of advantage value of a step in Eq. 2. 

Scaling Laws: First Pit 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0023-04.png)


**----- Start of picture text -----**<br>
0.5 First pit<br>Random<br>0.4<br>0.3<br>64k 128k 256k 512k 1024k<br>Training Data Size<br>Accuracy<br>**----- End of picture text -----**<br>


**Figure 11** | _**First pit strategy from Luo et al. (2024); Setlur et al. (2024)**_ : We compare the beam search performance (with beam size 128) for a PAV trained on data collected using two types of seed rollouts. For the seed rollouts, we either randomly sample from the distribution of state action pairs induced by the base policy. Or we improve coverage particularly by using the ”first pit” strategy of identifying the first state with low _𝑄_ values on a partial rollout that starts from a high _𝑄_ -value state and ends with an outcome reward of 0, i.e., _𝑄_ value is 0. 

## **E. Additional: Experiments on RL Training with PAVs** 

**Training details.** As discussed in Section 5, the initialization for RL training is the RFT (rejection finetuned) checkpoint for the corresponding base policies. More specifically, we consider two base policies Gemma 2B SFT, and Gemma 9B SFT, where the RL training is initialized with the policy obtained by further optimizing the base policy via rejection finetuning (RFT) Yuan et al. (2023). This is done to improve coverage over states and actions during the initial iterations of RL training. For rejection finetuning we train the SFT model (base policy) on all correct trajectories sampled when collecting seed rollouts for training PAVs (that predict the advantage of the same base policy). The training details for RFT remain same as SFT and are detailed in Appendix C. We use the REINFORCE (Sutton et al., 1999) 

23 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

algorithm to improve the base policy. The RL training is run for 10000 iterations for the 2B model, and for 5000 iterations for the 9B model. For both we use the Adam optimizer with a learning rate of 1e-7, and a batchsize of 32. The maximum response length is set to 512. For both we used a learning rate schedule with a linear warm up for 10% of the total training iterations, followed by a cosine decay. The implementation of our policy gradient algorithm also uses a token-level value function that is initialized to be the base policy itself, and is trained with a square loss. The value function is only used as a baseline during RL training, i.e., at any state _𝒔_ it is only predicting 𝔼 _𝑎_ ∼ _𝜋_ (·| _𝒔_ ) _𝑄[𝜋]_ ( _𝒔, 𝑎_ ) + _𝛼𝐴[𝜇]_ ( _𝒔, 𝑎_ ). 

Most importantly, we use a validation set to identify a good choice of _𝛼_ in the effective reward (in Eq. 5). For Gemma 2B this value is 5 _._ 0 and for the 9B policy _𝛼_ = 3 _._ 0 works best. Similar to the choice of _𝛼_ for test-time search, we find that most values of _𝛼_ in the range of 0 _._ 5 to 6 _._ 0 improved performance over ORM-RL, to different degrees. Both policies are also optimized with KL regularization, against the initial iterate of the RL policy, where the strength of the KL penalty is set to 0.001 for both. 

## **F. Theoretical Analysis: Complementary Provers Amplify Base Policy Improvement** 

In this section, we present the proofs for our theoretical results in the main paper (Sec. 3.4). We begin by describing some notation we use in our proofs, the natural policy gradient algorithm we use for the policy update, followed by the proof for Theorem 3.1. We also present a simple application of this result in Proposition F.1. Our results in this section are in the tabular setting, with softmax parameterization of the policies. Note that for the deterministic Markov decision process induced by the LLM, we are indeed in a tabular setting, where the set of states and actions is discrete, but large and grows exponentially in sequence length. 

**Notation and preliminaries.** We use _𝑑ℎ[𝜋][, 𝑑] ℎ[𝜇]_[to][denote][the][distribution][over][states] _[𝒔][ℎ]_[at][time][step] _[ℎ]_[,] starting from the initial state distribution given by the empirical distribution over the questions in the dataset D, and following the base policy _𝜋_ , or prover policy _𝜇_ respectively. The term _𝑑𝒔[𝜋]_[denotes][the] distribution over future states, starting from state _𝒔_ , and following policy _𝜋_ . Here, _𝒔_ can be a state at any time _ℎ_ ∈[0 _, . . . , 𝐻_ ]. For convenience, we overload the notation _𝑑𝒔[𝜋]_[(the distribution over future states] induced by a policy starting from state _𝒔_ ), and use _𝑑𝜌[𝜋]_[to denote the mixture distribution over] _[𝑑] 𝒔[𝜋]_[starting] from a random state _𝒔_ drawn from _𝒔_ ∼ _𝜌_ , and following policy _𝜋_ . 

The term _𝑄[𝜋]_ ( _𝒔ℎ, 𝑎ℎ_ ) refers to the value of state-action pair _𝒔ℎ, 𝑎ℎ_ , _i.e._ , the expected return in the future, starting the policy from state _𝒔ℎ_ and taking action _𝑎ℎ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0024-07.png)


Note that _𝒚𝒙[★]_[is known on the dataset][ D][, and state] _[𝒔][ℎ]_[contains the question] _[𝒙]_[as part of it.][Similarly, we] can define the value function _𝑉[𝜋]_ ( _𝒔ℎ_ ) of a state _𝒔ℎ_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0024-09.png)


The advantage function is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0024-11.png)


The policy gradient algorithm we use to update the base policy iteratively is natural policy gradient (Kakade, 2001b), and we use _𝜋𝑡_ to refer to the base policy iterate at time _𝑡_ of this iterative algorithm. 

24 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

Finally, we use S to denote the set of all states (prefixes) and A for the set of all actions (steps) that the LLM can take at any state. 

**Parameterization of the base policy.** We adopt the softmax parameterization for the base policy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0025-03.png)


Here _𝜽𝒔ℎ,𝑎_ ∈ **𝚯** ⊆ ℝ _[𝑑]_ controls the probability of taking action _𝑎_ at state _𝒔ℎ_ . The full set of parameters across all states and actions is denoted by _𝜽_ ∈ ℝ _[𝑑]_[×|S|×|A|] . Whenever clear from context, we overload the notation _𝜋𝑡_ to denote both the policy at iterate _𝑡_ , _i.e._ , _𝜋𝜽𝑡_ and the parameter _𝜽𝑡_ itself. _E.g._ , the gradient operator ∇ _𝜋𝑡_ [·] is referring to ∇ _𝜽𝑡_ [·]. 

**Defining policy improvement.** Let _𝜌_ be a distribution over all states { _𝒔ℎ_ : _ℎ_ ∈[0 _,_ 1 _, . . . , 𝐻_ ]}, then 𝔼 _𝒔_ ∼ _𝜌𝑉[𝜋]_ ( _𝒔_ ), and 𝔼 _𝒔_ ∼ _𝜌𝑉[𝜇]_ ( _𝒔_ ) give us the expected value functions over states across time steps, measured under _𝜌_ , for policies _𝜋_ and _𝜇_ respectively. We assume that _𝑑ℎ[𝜋]_[and] _[𝑑] ℎ[𝜇]_[are][both][absolutely][continuous] with respect to _𝜌_ , and use the expected value function over _𝜌_ as the quantity we track before and after a policy update. A positive change in 𝔼 _𝒔_ ∼ _𝜌𝑉[𝜋]_ ( _𝒔_ ) implies a net positive improvement in the base policy. Thus, progress is made at each update of the policy when: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0025-06.png)


## **F.1. Natural Policy Gradient** 

The natural policy gradient (NPG) algorithm (Kakade, 2001a) defines a Fisher information matrix (induced by the policy), and performs gradient updates in the geometry induced by the following matrix: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0025-09.png)


Typically, the NPG update does gradient updates on the objective _ℓ_ ORM−RL in Eq. 3, but in our case, the objective of interest is _ℓ_ PAV−RL in Eq. 4, and thus the natural gradient is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0025-11.png)


where _𝑀_[†] denotes the Moore-Penrose pseudoinverse of the matrix _𝑀_ . We restrict to using the initial state distribution _𝜌_ in our update rule, _i.e._ , we restrict attention to states _𝒔_ reachable from _𝜌_ , since _𝜌_ governs the performance measure of interest when evaluating the expected value of a policy. Thus, without loss of generality, we can exclude states that are not reachable under _𝜌_ . Specifically, we restrict the MDP to the set: { _𝒔ℎ_ : ∃ _𝜋_ such that _𝑑𝜌[𝜋]_[(] _[𝒔][ℎ]_[)] _[>]_[ 0] _[, ℎ]_[∈[][0] _[, . . . , 𝐻]_[]}][.][The scalar] _[𝛾>]_[ 0 determines the learning rate.] 

## **F.2. Useful Lemmas** 

**Lemma F.1.** _[The performance difference lemma; (Kakade and Langford, 2002)] For all policies 𝜋, 𝜋_[′] _and states 𝑠_ 0 _,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0025-15.png)


25 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

_Proof._ See proof of Lemma 6.1 in Kakade and Langford (2002). □ 

**Lemma F.2** (Natural policy gradient update) **.** _For the natural policy gradient in Eq. 12, the corresponding policy update is given by:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-04.png)


_Proof._ We use arguments similar to the proof of Lemma 15 in Agarwal et al. (2021), with the key difference of separately accounting for the term _𝐴[𝜇]_ ( _𝒔, 𝑎_ ) in the effective reward. For the sake of completeness we reproduce some of the derivation, accounting for the _𝐴[𝜇]_ term in the process. We follow compatible function approximation in Sutton et al. (1999) and Kakade (2001a). For a vector _𝒘_ ∈ ℝ _[𝑑]_[×|S||A|] , we define the error function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-06.png)


Let _𝒘[★]_ be the minimizer of _𝐿[𝜋]_ ( _𝒘_ ) with the smallest _ℓ_ 2 norm. Then by definition of Moore-Penrose pseudoinverse: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-08.png)


In other words, _𝒘[★]_ is precisely proportional to the NPG update direction. Note further that for the Softmax policy parameterization, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-10.png)


Since[�] _𝑎_ ∈A _[𝜋]_[(] _[𝑎]_[|] _[𝑠]_[)] _[𝐴][𝜋]_[(] _[𝑠, 𝑎]_[)][=][ 0, this immediately yields that:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-12.png)


However, this might not be the unique minimizer of _𝐿[𝜋]_ , which is problematic since _𝒘[★]_ ( _𝜋_ ) as defined in terms of the Moore-Penrose pseudoinverse is formally the smallest norm solution to the least-squares problem, which _𝐴[𝜋]_ + _𝛼𝐴[𝜇]_ may not be. However, given any vector _𝒗_ ∈ ℝ[|S|×|A|] , let us consider solutions of the form _𝐴[𝜋]_ + _𝛼𝐴[𝜇]_ + _𝒗_ . Due to the form of the derivatives of the policy for the softmax parameterization, we have for any state _𝒔, 𝑎_ such that _𝒔_ is reachable under _𝜌_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-14.png)


This is because _𝜋_ is a stochastic policy with _𝜋_ ( _𝑎_ | _𝒔_ ) _>_ 0 for all actions _𝑎_ in each state _𝒔_ , so that if a state is reachable under _𝜌_ , it will also be reachable using _𝜋_ , and hence the zero derivative conditions apply at each reachable state. For _𝐴[𝜋]_ + _𝛼𝐴[𝜇]_ + _𝒗_ to minimize _𝐿[𝜋]_ , we would like _𝑣_[⊤] ∇ _𝜋_ log _𝜋_ ( _𝑎_ | _𝒔_ ) = 0 for all _𝒔, 𝑎_ so that _𝒗𝒔,𝑎_ is independent of the action and can be written as a constant _𝑐𝒔_ for each _𝒔_ by the above equality. Hence, the minimizer of _𝐿[𝜋]_ ( _𝒘_ ) is determined up to a state-dependent offset, and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0026-16.png)


26 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

where _𝒗𝒔,𝑎_ = _𝑐𝒔_ for some _𝑐𝒔_ ∈ ℝ for each state _𝒔_ and action _𝑎_ . Finally, we observe that this yields the updates 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-02.png)


Owing to the normalization factor _𝑍[𝑡]_ ( _𝒔_ ), the state dependent offsets cancel in the updates for _𝜋_ , which yields the statement of the lemma. □ 

## **F.3. Proof of Theorem 3.1** 

_Proof._ For some notational convenience, we use _𝑉[𝑡] , 𝑉[𝑡]_[+][1] , to denote the value functions _𝑉[𝜋][𝑡] , 𝑉[𝜋][𝑡]_[+][1] for policies at _𝜋𝑡_ , _𝜋𝑡_ +1 at time _𝑡_ and _𝑡_ + 1 respectively. Similarly, we use _𝐴[𝑡] , 𝐴[𝑡]_[+][1] for _𝐴[𝜋][𝑡] , 𝐴[𝜋][𝑡]_[+][1] respectively. For the distribution over states _𝑑𝜌[𝜋][𝑡]_[+][1] induced by the policy _𝜋𝑡_ +1, starting from an initial distribution of states given by _𝜌_ , we simplify the notation and use _𝑑𝜌[𝑡]_[+][1][.][Similarly we use] _[𝑑] 𝜌[𝑡]_[for] _[𝑑] 𝜌[𝜋][𝑡]_[.] 

Next, for simplicity we set _𝛼_ = 1 in the natural policy gradient update in Lemma F.2. It is easy to see that the lower bound result we show holds for any value of _𝛼>_ 0, and the term in the lower bound scales linearly with _𝛼_ . Note, that this is not a free variable, and _𝛼_ has to be _𝑂_ (1), since as we increase the value of _𝛼_ we would have to correspondingly reduce _𝛾_ for our result to hold. For now, we fix _𝛼_ = 1. 

From the policy difference Lemma F.1, we can write: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-08.png)


Next, from the natural policy gradient update in Lemma F.2, we can write _𝐴[𝑡]_[+][1] ( _𝒔, 𝑎_ ) as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-10.png)


Substituting Eq. 18 in Eq. 17 we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-12.png)


Recall that for _𝛼_ = 1, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-14.png)


Applying Jensen’s inequality we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-16.png)


since 𝔼 _𝜋[𝑡]_ [ _𝐴[𝑡]_ ( _𝒔, 𝑎_ )] = 0. Note that in Eq. 20 the KL term is always non-negative. Thus, we can lower bound our policy improvement: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0027-18.png)


27 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

where the inner product is the standard euclidean product as our actions space A is discrete. 

In the following we will treat the distribution _𝜋[𝑡]_[+][1] (· | _𝒔_ ) as a vector denoted by _𝜋_ Next, from the NPG update we know that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-03.png)


We note that for _𝛾_ ≪ 1, exp _𝛾_ ( _𝐴[𝑡]_ ( _𝒔, 𝑎_ )) + _𝛾_ ( _𝐴[𝜇]_ ( _𝒔, 𝑎_ )) = Θ(1 + _𝛾_ ( _𝐴[𝑡]_ ( _𝒔, 𝑎_ ) + _𝐴[𝜇]_ ( _𝒔, 𝑎_ ))), where the terms that grow as _𝜔_ ( _𝛾_ ) are being ignored. Based on this, for _𝛾_ ≪ 1, ∃ constants 0 _< 𝐶_ 1 _< 𝐶_ 2 such that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-05.png)


Applying the above claim in Eq. 24 gives us: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-08.png)


where we reused: 𝔼 _𝜋[𝑡]_ [ _𝐴[𝑡]_ ( _𝒔, 𝑎_ )] = 0. Here, _𝐶_ 3 _>_ 0 is a constant. 

We now plug in the above lower bound into Eq. 21 to get the final lower bound on the policy improvement in Theorem 3.1. For this, we will once again use the assumption that the learning rate _𝛾_ ≪ 1, which allows us to use 1 + _𝛾_ 𝔼 _𝑎_ ∼ _𝜋𝑡_ ( _𝑎_ | _𝑠_ ) [ _𝐴[𝑡]_ ( _𝒔, 𝑎_ ) + _𝐴[𝜇]_ ( _𝒔, 𝑎_ )] ≥ _𝐶_ 4 for some constant _𝐶_ 4 _>_ 0. This is because, in our setting the range of the advantages is [−1 _,_ 1]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-11.png)


Now Eq. 27 gives us, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-13.png)


Now, for the last step we note that _𝑑𝜌[𝑡]_[+][1] is component wise larger than _𝜌_ , and this gives us the final result: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-15.png)


## **F.4. Discussion on Remark 3.1** 

First, we note that if the _𝑄_ -value of a base policy _𝜋_ at state, action pair ( _𝒔, 𝑎_ ) is _𝑄[𝜋]_ ( _𝒔, 𝑎_ ), then for the prover _𝜇_ set to the best-of-K policy BoK( _𝜋_ ), the _𝑄_ -value at the same state, action pair is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0028-18.png)


28 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

This is because, in our setup the final outcome of correctness given by Rex is a random variable taking values in {0 _,_ 1}, with expectation _𝑄[𝜋]_ ( _𝒔, 𝑎_ ), when completing a rollout starting from prefix ( _𝒔, 𝑎_ ). Thus, we can treat the outcome of function Rex as a Bernoulli random variable. Next, we can simply compute the probability of sampling a single correct answer out of _𝐾_ attempts, which is also a Bernoulli random variable with mean given by _𝑄[𝜇]_ ( _𝒔, 𝑎_ ) in Eq. 30. 

Now for Remark 3.1, we observe that whenever _𝑄[𝜋]_ ( _𝒔, 𝑎_ ) ≪ 1, _e.g._ , when _𝑄[𝜋]_ ( _𝒔, 𝑎_ ) = _𝑂_ (1/ _𝐾_ ), for all values of ( _𝒔, 𝑎_ ), we can do the Taylor approximation of _𝑄[𝜇]_ ( _𝒔, 𝑎_ ) around 0, and note that _𝑄[𝜇]_ ( _𝒔, 𝑎_ ) = Θ( _𝐾_ · _𝑄[𝜋]_ ( _𝒔, 𝑎_ )). Next, note the following calculation for the first term (distinguishability) in Theorem 3.1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0029-03.png)


This means that the first term in Theorem 3.1 which measures “distinguishability” now increases by a factor of _𝐾_[2] . Similarly, we can see that the the term which measures “misalignment” can change in magnitude by atmost a factor of _𝑂_ ( _𝐾_ ), since the misalignment term is linear in _𝑄[𝜇]_ . These two observations combined lead us to the conclusion in Remark 3.1. 

## **F.5. Improving a Stronger Base policy with a Weaker Prover Policy** 

In Proposition F.1, we consider the case where the _𝜋_ and _𝜇_ differ in performance, as measured under the distribution of states _𝜌_ in the following way: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0029-07.png)


Next, whenever the prover’s preference over actions is complementary to the base policy, by a factor of _𝜂_ , i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0029-09.png)


then the variance of _𝐴[𝜇]_ or _𝐴[𝜋][𝑡]_ under _𝜋𝑡_ should scale as _𝜂_[2] . 

Thus, we see that when _𝜋𝑡_ fails to distinguish actions ( _i.e._ , 𝔼 _𝒔_ ∼ _𝜌_ 𝕍 _𝜋𝑡_ [ _𝐴[𝜋][𝑡]_ ( _𝒔, 𝑎_ )] is small) regardless of the strength of prover policy _𝜇_ , as long as it is sufficiently complementary to _𝜋𝑡_ , the prover policy induces an improvement in base policy, that scales as _𝜂_[2] . 

**Proposition F.1** (Complementary _𝜇_ boosts improvements in _𝜋_ ) **.** _Under the distribution over states given by 𝜌, let prover 𝜇 and base policy at iterate 𝑡, 𝜋𝑡, differ in absolute performance, i.e.,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0029-13.png)


_When_ 𝔼 _𝒔_ ∼ _𝜌_ 𝕍 _𝑎_ ∼ _𝜋𝑡_ [ _𝐴[𝜋][𝑡]_ ( _𝒔, 𝑎_ )] _<_ 𝔼 _𝒔_ ∼ _𝜌_ 𝕍 _𝑎_ ∼ _𝜋𝑡_ [ _𝐴[𝜇]_ ( _𝒔, 𝑎_ )] _, and 𝜇 is complementary to 𝜋𝑡, i.e.,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0029-15.png)


_then_ 𝔼 _𝒔_ ∼ _𝜌_ [ _𝑉[𝜋][𝑡]_[+][1] ( _𝒔_ ) − _𝑉[𝜋][𝑡]_ ( _𝒔_ )] ∼ _[>] 𝜂_[2] _._ 

29 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

_Proof._ We begin by proving an upper bound on the disagreement between prover and base policy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0030-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0030-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0030-04.png)


where the last inequality uses Cauchy-Schwartz. Next we apply Jensen’s inequality on the terms in the square root, and the conditions that 𝔼 _𝒔_ ∼ _𝜌_ 𝔼 _𝑎_ ∼ _𝜋𝑡_ | _𝑄[𝜋][𝑡]_ ( _𝒔, 𝑎_ ) − _𝑄[𝜇]_ ( _𝒔ℎ, 𝑎_ )| = Ω( _𝜂_ ) to conclude: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0030-06.png)


whenever 𝔼 _𝒔_ ∼ _𝜌_ 𝕍 _𝑎_ ∼ _𝜋𝑡_ [ _𝐴[𝜋][𝑡]_ ( _𝒔, 𝑎_ )] _<_ 𝔼 _𝒔_ ∼ _𝜌_ 𝕍 _𝑎_ ∼ _𝜋𝑡_ [ _𝐴[𝜇]_ ( _𝒔, 𝑎_ )]. Now that we lower bound “distinguishability”, it is easy to see that a similar derivation would upper bound the magnitude of the misalignment term by _𝑂_ ( _𝜂_[2] ). Invoking the result in Theorem 3.1 yields: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0030-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning_images/iclr_2025_Rewarding_Progress_Scaling_Automated_Process_Verifiers_for_LLM_Reasoning.pdf-0030-09.png)


## **G. Examples Generated by Base Policy Trained on Rewards** _𝑄[𝜋]_ + _𝛼𝑄[𝜇]_ 

When we train the base policy with reinforcement learning where the reward is _𝑄[𝜋]_ + _𝛼𝑄[𝜇]_ , instead of the effective reward _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ , we find that the policy training converges to contrived solutions with degenerate failures. 

For example, we see in multiple examples (like the four below), that for every question, the model learns to output "REPHRASE THE PROBLEM" as the first step. This is because under _𝑄[𝜇]_ this trivial step achieves a positive reward. While there exists a better solution where the policy outputs the correct answer and achieves higher rewards, it fails to recover from this local optima and find the better solution. Across multiple training runs, we observed similar trends when training with _𝑄[𝜋]_ + _𝛼𝑄[𝜇]_ . Note that in the idealized update, where we observe advantages _𝐴[𝜇]_ or value function _𝑄[𝜇]_ on all actions at a state ( _e.g._ , in the NPG update in Sec. 3.4), using _𝑄[𝜋]_ + _𝛼𝑄[𝜇]_ is equivalent to _𝑄[𝜋]_ + _𝛼𝐴[𝜇]_ . But, in practice, _𝐴[𝜇]_ offers significant benefits in the form of variance reduction in policy gradient iterates. In this instance, it does that by not assigning reward to steps like “REPHRASE THE PROBLEM”, since the likelihood of a prover solving before or after this step is expected to be roughly the same under any choice of the prover policy ( _i.e._ , the prover policy makes no progress). 

## Example 1 

```
Q:Myclubhas25members.InhowmanywayscanIchoosememberstoforma4-person
executivecommittee?
```

```
A:
```

```
>>>
REPHRASETHEPROBLEM
WENEEDTOFINDTHENUMBEROFWAYSTOCHOOSE4MEMBERSOUTOF25FORTHEEXECUTIVE
COMMITTEESOWECANUSETHECOMBINATIONFORMULAWHICHIS
```

30 

Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning 

```
NCR=N!/(R!*(N-R)!)WHERENISTHETOTALNUMBEROFMEMBERSANDRIS
THENUMBEROFMEMBERSWECHOOSESOINTHISCASEN=25ANDR=4SOWEPLUGTHEMINAND
SIMPLIFY25C4=25!/(4!*(25-4)!)=(25*24*23*22)/(4*3*2*1)=104005
#Answer
```

```
104005
```

## Example 2 

```
Q:Wehaveatriangle$\triangleABC$andapoint$K$on$BC$suchthat$AK$is
analtitudeof$\triangleABC$.If$AC=10,$$BK=7$,and$BC=13,$then
whatistheareaof$\triangleABC$?
```

```
A:
>>>
REPHRASETHEPROBLEM
WENEEDTOFINDTHEAREAOF$\triangleABC$SOWECANUSETHEFORMULA$A=\frac{1}{2}bh$
WHERE$b$ISTHEBASEAND$h$ISTHEHEIGHTSOSINCE$AK$ISANALTITUDE,THATMEANS
$AK\perpBC$SOWECANUSE$BC$ASTHEBASEAND$AK$ASTHEHEIGHTSONOWWENEED
TOFIND$AK$SOWECANUSETHEPYTHAGORETHEOREMON$\triangleAKC$SINCE$AC=10$
#Answer
```

```
48
```

## Example 3 

```
Q:Supposethat$n,n+1,n+2,n+3,n+4$arefiveconsecutiveintegers.
```

```
Determineasimplifiedexpressionforthesumofthesefiveconsecutiveintegers.
```

```
A:
```

```
>>>
REPHRASETHEPROBLEM
WENEEDTOFINDTHESUMOFTHESEFIVECONSECUTIVEINTEGERSSOWECANUSETHEFORMULA
$S_n=\frac{n}{2}(a_1+a_n)$WHERE$S_n$ISTHESUM,$n$ISTHENUMBEROFTERMS,
$a_1$ISTHEFIRSTTERM,AND$a_n$ISTHELASTTERMSOINTHISCASE$n=5,a_1=n,
a_5=n+4$
#Answer
5n+10
```

## Example 4 

```
Q:Whatisthevalueof$x^2+y^2-z^2+2xy$when$x=-3$,$y=5$,and$z=-4$?
```

```
A:
```

```
>>>
REPHRASETHEPROBLEM
WENEEDTOFINDTHEVALUEOF$x^2+y^2-z^2+2xy$WHEN$x=-3$,$y=5$,AND$z=-4$
SOWEJUSTPLUGINTHEVALUESANDSIMPLIFY
#Answer
```

```
-2
```

31 

