# **Lean-STaR: Learning to Interleave Thinking and Proving** 

**Haohan Lin**[2] _[∗]_ **Zhiqing Sun**[1] 

**Yiming Yang**[1] **Sean Welleck**[1] 

1Language Technologies Institute, Carnegie Mellon University 2Institute for Interdisciplinary Information Sciences, Tsinghua University 

```
{haohanl,zhiqings,yiming,swelleck}@cs.cmu.edu
```

```
https://leanstar.github.io/
```

## **Abstract** 

Traditional language model-based theorem proving assumes that by training on a sufficient amount of formal proof data, a model will learn to prove theorems. Our key observation is that a wealth of _informal_ information that is not present in formal proofs can be useful for learning to prove theorems. For instance, humans think through steps of a proof, but this thought process is not visible in the resulting code. We present Lean-STaR, a framework for training language models to produce informal thoughts prior to each step of a proof, thereby boosting the model’s theorem-proving capabilities. Lean-STaR uses retrospective ground-truth tactics to generate synthetic thoughts for training the language model. At inference time, the trained model directly generates the thoughts prior to the prediction of the tactics in each proof step. Building on the self-taught reasoner framework, we then apply expert iteration to further fine-tune the model on the correct proofs it samples and verifies using the Lean solver. Lean-STaR achieves state-of-the-art results on the miniF2F-test benchmark within the Lean theorem proving environment, significantly outperforming base models ( **43** _**.**_ **4** % _**→**_ **46** _**.**_ **3** % _**,**_ Pass@64). We also analyze the impact of the augmented thoughts on various aspects of the theorem proving process, providing insights into their effectiveness. 

## **1 Introduction** 

Theorem proving is a fundamental aspect of mathematics, and mathematical reasoning is an important part of artificial intelligence [28, 55]. _Formalized mathematics_ in particular provides a challenging testbed for assessing mathematical reasoning capabilities. Since theorems and proofs in this setting can be represented in the form of checkable source code, it is easy to evaluate proofs of arbitrary complexity [15]. Automated theorem proving, if successful, can also help discover unknown errors in previous proofs[1] , and make it easier to guarantee that new proofs are correct. More broadly, formal mathematics coupled with powerful automation may unlock new forms of education and collaboration, mathematical insights, and applications to verifying critical software [4, 16, 10, 31]. 

> _∗_ Work done during the visit at CMU. 

> 1For example, Terence Tao found a non-trivial error while using Lean to formalize a project [38]. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0002-00.png)


Figure 1: The illustration of tactic prediction in one proof step with and without thought. 

Recently, language models have shown promising progress in formal theorem proving [33, 35, 47, 19, 25, 50, 26]. Existing approaches typically train a model solely based on the proofs in a formal language (code) such as Lean [15], Isabelle [29], or Coq [13]. Our key observation is that such approaches ignore a wealth of _informal_ information that may be useful for learning to prove theorems [45, 46]. For instance, the underlying _thought process_ prior to each step of a proof is not present in formal source code. Based on this insight, we propose to train a language model that can produce a natural language chain-of-thought (“thought”) prior to each step (“tactic”) of a formal proof. 

We introduce Lean-STaR, a framework for learning to interleave informal thoughts with steps of formal proving. Building on the Self-Taught Reasoner (STaR) framework [52], we enable language models to interleave step-by-step rationales (i.e., thoughts) [30, 43] with formal proving in a two-stage process. In an initial phase, we prompt a sufficiently capable language model, such as GPT-4 [1], and generate retrospective thoughts based on a dataset of human-written proofs, such as Mathlib, the largest collection of human-written proofs in Lean [27]. Subsequently, we fine-tune a thoughtaugmented tactic predictor [8, 7, 18, 14] that, given a Lean state, can generate a thought and predict the subsequent tactic. In a second phase, we optimize this thought-augmented tactic predictor with the expert iteration algorithm [3, 36], using multi-step success rate in theorem proving as the reward. 

Our work presents a new link between informal and formal mathematics, complementary to prior explorations that translate standalone mathematical statements [37, 42, 48] or translate informal proofs into formal proofs [2, 23, 5, 56, 22]. Lean-STaR generates natural language thoughts specifically for each proof step, improving formal proving capabilities by interleaving natural and formal languages. 

We instantiate Lean-STaR by generating roughly 50,000 thought-augmented examples from Lean’s Mathlib [27], then synthesize an additional 50k examples through two iterations of expert iteration. To the best of our knowledge, this yields the first thought-augmented dataset for theorem proving. After fine-tuning an InternLM2-7b base model [51] on our thought-augmented data, our final Lean-STaR model can solve **34** _._ **8** % (pass@32) or **36** _._ **1** % (pass@64) of the problems on miniF2F-test [54]. Using stronger base model InternLM2-7b-plus, Lean-STaR can achieve **45** _._ **4** % (pass@32), significantly surpassing the previous state-of-the-art results of 43 _._ 4% (pass@32). In summary, Lean-STaR offers a framework for teaching language models to interleave informal thoughts with formal verification, advancing the capabilities of language models in automated theorem proving. 

## **2 Related Work** 

**Automatic Theorem Proving & Autoformalization.** Previous work on learning-based theorem proving typically follows the GPT-f framework [33], which trains a language model on (proof state, next-tactic) pairs, then proves theorems by using the model within a best-first tree search. Subsequent work has explored several directions, including data augmentation [20], novel proof search methods [25, 41], further training through curriculum learning [34], retrieval augmentation [50], or practical tools [44]. Others use prompted models to generate tactics [6, 39], or fine-tune models to generate a full proof [17]. A second _auto-formalization_ [48] thread incorporates informal mathematics into 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0003-00.png)


Figure 2: **An example of Lean proof and thoughts generated by Lean-STaR** . Note that there is a calculation error in the thought (in red), but this does not affect the correctness of the proof because the calculation task is actually completed by the interactive theorem prover (i.e., Lean’s `nlinarith` ) instead of the language model. This shows a benefit of combining neural and symbolic systems. 

formal theorem proving. Draft-Sketch-Prove [23] shows that language models have some ability to use informal proofs to improve a model’s formal proving abilities, by drafting an informal proof, translating into a formal proof sketch, then completing the proof with tools like Sledgehammer [8]. Draft-Sketch-Prove and related methods [40, 53, 57] are limited to the Isabelle prover, since they use powerful automatic proving tools like Sledgehammer. Lean lacks these tools, so generating the entire proof at once would be more unlikely in Lean. We focus on Lean, and train language models to generate a thought and predict the subsequent tactic in each proof step. To the best of our knowledge, we are the first to introduce thought-augmented reasoning in automatic theorem proving. 

**Rationale-augmented Reasoning.** Recently, many works demonstrated that letting language models reason before an answer can improve their performance on tasks including math, science, and code [30, 43, 12]. Although the corresponding techniques (e.g., Scratchpad and Chain-of-Thought) have proven to be effective, they require either extensive annotated training examples or exposure to numerous similar examples during pre-training [9]. The scarcity of natural language reasoning in formal theorem proving, coupled with the impracticality of manually annotating rationales for formal mathematics, thus presents a challenge. We propose a new Lean-STaR framework for _synthesizing_ training examples by taking advantage of the correctness signal from the formal system. 

**Bootstrapping Language Model Reasoning.** Recently, several works suggest that language models may be taught to reason via synthetic data that they generate themselves, akin to a reinforcement learning method that improves a policy through self-play. Polu et al. [34] showed that a simple RL algorithm, expert iteration, paired with curriculum learning can improve a formal theorem proving model. Self-Taught Reasoner (STaR) [52] showed that we can iteratively fine-tune the language model on the correct (reasoning, answer) pairs generated by itself to gradually improve performance. Singh et al. [36] proposed ReST-EM, which filters data generated by language model with a binary feedback signal rather than using fully manually annotated data (similar to expert iteration in [34]). Our work builds on these ideas, providing the first study of bootstrapped thought-augmented proving. 

## **3 Our Method: Lean-STaR** 

We introduce Lean-STaR, a new method for combining informal thoughts with formal theorem proving. First, we recap interactive theorem proving (§3.1). Then we present Lean-STaR’s datageneration (§3.2.1, §3.2.2) and reinforcement learning (§3.2.3) phases. Finally, we present our evaluation protocols (§3.3). 

3 

### State a b n : N _⊢_ ¬(n _̸_ = 0 _∧_ a _̸_ = 0) _↔_ n = 0 _∨_ a = 0 ### Tactic ```lean4 simp only [not_and_or, ne_eq, not_not] ``` 

### Reasoning To prove the equivalence between the negation of a conjunction and a disjunction, we utilize logical equivalences, specifically the negation of a conjunction ¬(`(P _∧_ Q)`) being equivalent to the disjunction of the negations (`P = 0 _∨_ Q = 0`) and simplifications related to negation and inequality. 

Figure 3: Examples of (input, output) pairs of retrospective rationale generation with GPT-4. The full prompt is given in Appendix F. See §3.2.2 for the detailed explanation. 

## **3.1 Preliminaries** 

_Interactive Theorem Provers_ (ITPs) are typically used for step-by-step automatic theorem proving in formal mathematics. At each step, we can provide the ITP with a high-level “tactic” to simplify the current goal state (e.g., the initial goal theorems to be proven) into subgoals. These subgoals will form new states, and proving all the subgoals results in a complete proof of the given theorem. We use Lean [15], a popular interactive theorem prover. An example formal proof in Lean and its explanation are shown in Appendix C. 

## **3.2 Data Generation & Training** 

We describe the data generation and training of the direct tactic prediction model (SFT), the thoughtaugmented tactic prediction model trained with synthetic data (Lean-CoT), and the final model trained with expert iteration (Lean-STaR). 

## **3.2.1 Direct Tactic Prediction** 

We define the theorem-proving problem as a _Markov Decision Process_ (MDP) ( _S, A, Pa, Ra_ ) where proof states serve as states in MDP and tactics serve as actions. From this perspective, a proof is a trajectory ( _s_ 1 _, a_ 1 _, r_ 1) _,_ ( _s_ 2 _, a_ 2 _, r_ 2) _, · · ·_ of states _si_ , tactics _ai_ , and rewards _ri ∈_ R, and the ITP (e.g., Lean) provides each new state _si_ +1. 

In the typical setting [33], proving a theorem consists of providing a proof state _s_ to the language model and then generating a tactic from the language model _M_ , i.e., _πM_ ( _a|s_ ). The language model can be fine-tuned for this task using a dataset of (proof state, next-tactic) pairs from successful proof trajectories, i.e. _D_ = _{_ ( _s[i] , a[i]_ ) : _i_ = 1 _, · · · , M }_ , where final states have a reward of 1. We refer to a language model fine-tuned on such a dataset as a _supervised fine-tuning (SFT)_ model. 

## **3.2.2 Thought-augmented Tactic Prediction** 

Existing approaches typically train only on formal states and tactics [33]. We hypothesize that incorporating a latent _thought_ can improve a model’s ability to predict the next tactic. Formally, we introduce a hidden “thought” variable _ti_ prior to each tactic, and then extend the model to the form _πM_ ( _ai, ti|si_ ) = _πM_ ( _ai|ti, si_ ) _πM_ ( _ti|si_ ). In thought-augmented tactic prediction, the distribution over the next tactic can then be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0004-12.png)


The key challenge is obtaining (state, thought, tactic) pairs for training a model. To this end, we introduce **retrospective rationale generation** . Our motivating observation is that the distribution of natural language thoughts in theorem-proving _πM_ ( _ti|si_ ) is scarce in the pre-training corpus of large language models. In turn, we find that even the most powerful GPT-4 model does not perform well in generating the correct rationale through few-shot prompting [9]. To develop a language model capable 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0005-00.png)


Figure 4: **The diagram of our pipeline.** (1) Produce CoT dataset through GPT-4. (2) Fine-tune the SFT model with the CoT dataset to obtain Lean-CoT. (3) Use expert iteration to generate the STaR dataset through the model in the last iteration (Lean-CoT in the first iteration) and then fine-tune Lean-CoT on the updated STaR dataset to obtain the model in the next iteration. We continue performing this step until a stopping condition is met (e.g., a fixed number of iterations). 

of generating thoughts and tactics _ai, ti|si_ , we need an entirely new dataset _DT_ = _{_ ( _s[i] , t[i] , a[i]_ ) : _i_ = 1 _, · · · , N }_ . However, in Lean, we only have a dataset of _DS_ = _{_ ( _s[i] , a[i]_ ) : _i_ = 1 _, · · · , N }_ where ( _s[i] , a[i]_ ) is one step in some successful proof trajectories. Given a powerful large language model _G_ , which we refer to as the oracle model[2] , we give the oracle model the ground-truth tactic _ai_ and let the oracle model produce the thought _ti_ given the current state _si_ and ground-truth tactic _ai_ . This helps improve the pass rate and produce thought-augmented data more efficiently. Our few-shot prompt is provided in Appendix F. The design principle of the prompt is to prevent the oracle model from generating hindsight-like thoughts. 

We randomly select _M_ pairs ( _s[i] , a[i]_ ) _∈ DS_ . Then the oracle model is used to produce a thought _t[i]_ for each pair ( _s[i] , a[i]_ ) to create a new dataset _DT {_ ( _s[i] , t[i] , a[i]_ ) : _i_ = 1 _, · · · , M }_ . With this retrospectively annotated dataset by the oracle model _DT_ , we obtained our first thought-augmented tactic prediction model, Lean-CoT, by fine-tuning from the SFT model. 

## **3.2.3 Bootstrapping Thought-augmented Theorem Proving** 

We propose to apply expert iteration to further improve the performance of Lean-CoT. Specifically, we start from the initial Lean-CoT model _M_ 0 and the initial dataset _D_ = _{s[i]_ : _i_ = 1 _, · · · , M }_ , which consists of all initial states _s[i]_ of the theorems to be proved. In iteration 1, we use model _M_ to sample _K_ times per problem. Each time the model will produce a proof trajectory [( _s_ 0 _, t_ 0 _, a_ 0) _,_ ( _s_ 1 _, t_ 1 _, a_ 1) _, · · · ,_ ( _sn, tn, an_ )]. Then we create a new dataset _D_ 1 by filtering the generated trajectories to include only the successful ones. De-duplication is then applied to the collected trajectories. Now, we can further fine-tune the SFT model _M_ on dataset _DT ∪ D_ 1 to produce Lean-STaR model _M_ 1. Then we can use _M_ 1 as initial model to produce dataset _D_ 2 and further fine-tune to obtain model _M_ 2 in the next iteration. 

This method can be seen as an offline RL method [36] in the theorem proving MDP. In this MDP, the cumulative reward _R_ (( _s_ 0 _, t_ 0 _, a_ 0) _,_ ( _s_ 1 _, t_ 1 _, a_ 1) _, · · · ,_ ( _sn, tn, an_ )) = 1 if and only if the proof trajectory is successful. The total expected reward is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0005-07.png)


and Lean-STaR’s expert iteration can be seen as optimizing this reward [36]. 

## **3.3 Evaluation** 

**Setup.** We evaluate the model on formal theorem proving – given a theorem statement, produce a theorem that is correct according to the formal system. This requires an algorithm for producing a 

> 2For instance, in our experiments we use the best available large language model, GPT-4. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0006-00.png)


Figure 5: **The visualization of Best-first Search (** _K_ = 1 **) and Sampling (** _S_ = 1 **).** Search method maintains a search tree and explores _S_ tactics on each expanded node. Sampling method explores _K_ tactic trajectories from the root and ignores illegal tactics in the trajectories. 

full proof by interacting with Lean. As a new form of theorem-proving system, it is unclear what the best strategy is when we have informal thoughts. Our preliminary experiments indicate that best-first search with beam search does not work well for the thoughts in the natural language format. Thus we describe the traditional strategy (best-first search), and our new approach based on sampling. 

**Best-First Search.** The most popular method to evaluate the theorem proving ability of a language model _M_ is to use best-first search like GPT-f [33, 50, 6, 44]. In best-first search, we keep all unexpanded states _si_ . Each time, we expand the “best” state _si_ and use the language model to sample _S_ next tactics _ai,_ 1 _···S_ for the current state _si_ . For each legal tactic _ai,j_ , a new state can be obtained by applying tactic _ai,j_ on state _si_ . Following standard practice [33, 50, 44], we assume the state with maximum log-probabilities is the “best”s. Specifically, we select state _si_ with max- _i−_ 1 imum � log _p_ ( _aj, sj_ ), where ( _s_ 0 _, a_ 0) _, · · · ,_ ( _si−_ 1 _, ai−_ 1) is the proof trajectory before state _si_ and _j_ =0 

log _p_ ( _aj, sj_ ) is the average log probability of each generated token. We expand up to _N_ states and we get a successful proof search when we reach any proof state with no goals. Then, we can attempt the search _K_ times to obtain a pass rate _pass_ @ _K_ . However, we found that the best-first search method performs poorly with the Lean-CoT and Lean-STaR models. We attribute this to using average log probabilities, which may not be a reliable quality indicator when the thought sequence _tj_ is generated. 

**Sampling.** Motivated by these issues with applying best-first search to thought-augmented proving, we develop a new method based on sampling trajectories in parallel. Specifically, our method samples _K_ times in parallel for each problem, each time generating at most _N_ tactics. Also, illegal sampled tactics will be ignored during sampling. Specifically, in a sample, suppose our current state is _si_ , the proof trajectory before _si_ is ( _s_ 0 _, a_ 0) _, · · · ,_ ( _si−_ 1 _, ai−_ 1) and the sampled tactic is _ai_ . If _ai_ is a legal tactic, ( _si, ai_ ) will be added to the proof trajectory and we will reach a new state obtained by applying tactic _ai,j_ on state _si_ . Otherwise, we ignore this _ai_ and use language model _M_ to sample a new tactic given state _si_ . We limit the number of times a tactic can be generated by language model _M_ to a total of _N_ per time in _K_ sampling times. The sampling method is roughly equivalent to the search with _S_ = 1, except that the sampling ignores illegal tactics. We assume that in the sampling method we have _S_ = 1. In this setting, evaluating our sampling method and best-first search with equal _S × K_ took approximately the same amount of GPU time. This sampling method can easily accommodate hidden variable “thoughts” _tj_ . Figure 5 compares best-first search and our sampling method. 

6 

Table 1: **Pass rates on the minif2f-test dataset with Lean.** This table shows the pass rates of previous works and our work. _S_ is the number of tactics attempted at each expanded node (assumed to be 1 in sampling) and _K_ is the total number of search or sampling attempts per problem. In sampling we use temperature 0.7, and in search we use beam search when generating the next tactic. Note that we sample 32 examples twice when _K_ = 64 in sampling. 

|APPROACH|DECODING|_N_|_K_|_S_|PASS RATE|
|---|---|---|---|---|---|
|GPT-3.5 [1] (FEW-SHOT)|SAMPLING|50|1|1|2_._8%|
|GPT-4 [1] (FEW-SHOT)|SAMPLING|50|1|1|11_._9%|
|TRANSFORMER[34] (W/ORL)|SEARCH|512|1|8|24_._6%|
|LLEMMA-34B[6] (FEW-SHOT)|SEARCH|50|1|32|25_._8%|
|LLEMMA-7B[6] (FEW-SHOT)|SEARCH|50|1|32|26_._2%|
|REPROVER[50]|SEARCH|50|1|64|26_._5%|
|TRANSFORMER[34] (W/ RL)|SEARCH|512|1|8|29_._6%|
|INTERNLM2-34B[51] (FEW-SHOT)|SEARCH|50|1|32|29_._5%|
|COPRA (WITHGPT-4) [39]|CUSTOMIZED|-|60|1|29_._9%|
|COPRA (WITHGPT-4) [39]|CUSTOMIZED|-|100|1|30_._7%|
|INTERNLM2-7B[51] (FEW-SHOT)|SAMPLING|50|32|1|28_._7%|
|INTERNLM2-7B[51] (FEW-SHOT)|SEARCH|50|1|32|30_._3%|
|SFT (INTERNLM2-7B)|SAMPLING|50|32|1|29_._5%|
|SFT (INTERNLM2-7B)|SEARCH|50|1|32|30_._7%|
|**LEAN-COT** (INTERNLM2-7B)|SAMPLING|50|32|1|32_._8%|
|**LEAN-STAR (ITER-1)**(INTERNLM2-7B)|SAMPLING|50|32|1|34_._0%|
|**LEAN-STAR (ITER-2)**(INTERNLM2-7B)|SAMPLING|50|32|1|**34**_._**8**%|
|**LEAN-STAR (ITER-2)**(INTERNLM2-7B)|SAMPLING|50|64|1|**36**_._**1**%|



## **4 Experiments** 

We instantiate Lean-STaR using the best available open language model pre-trained on the Lean corpus (InternLM2-Math-base-7b [51]), and follow standard practice in using Lean’s Mathlib as the underlying training set (via the Lean Dojo dataset [50]). We generate an initial set of thoughts for Mathlib using GPT-4, perform two rounds of expert iteration, then evaluate the model on miniF2F [54], the de-facto standard benchmark for evaluating language-model based theorem provers. Our experimental results show that both retrospective rationale generation and expert iteration significantly improve the theorem-proving capabilities of language models in this setting. We describe our setup and findings in detail below. 

## **4.1 Experimental Setup** 

We use _LeanDojo Benchmark 4 v9_ as the supervised fine-tuning (SFT) dataset containing 231 _,_ 240 data examples. We fine-tune for 1 epoch to obtain the SFT model. For the learning rate, we use a warmup in the first 20% steps from 0 to 2 _×_ 10 _[−]_[5] , followed by a cosine schedule decaying to zero. 

We randomly select 17 _,_ 256 different successful proof trajectories from _LeanDojo Benchmark 4 dataset_ [50], and use GPT-4-0125 [32] to annotate 52 _,_ 438 thoughts from those proof trajectories. We filtered out all proof steps ( _s[i] , a[i]_ ) for which _a[i]_ contains the newline symbol “\n” before annotating. We perform two iterations of expert iteration, and provide the details in Appendix A.1 due to space. 

We evaluate our method on the _MiniF2F_ benchmark [54]. We use a similar evaluation setting as previous works [50, 44, 51], but use our sampling method instead of best-first search for the evaluation 

7 

Table 2: Counts of problems successfully proved in _minif2f-test_ benchmark, split by type and difficulty. The methods use sampling with _N_ = 50 _, K_ = 32. Thought-augmented methods improve performance on all categories, while Lean-STaR significantly improves Number Theory performance. 

||TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-7B SFT LEAN-COT LEAN-STAR<br>(ITER-2)|TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-7B SFT LEAN-COT LEAN-STAR<br>(ITER-2)|TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-7B SFT LEAN-COT LEAN-STAR<br>(ITER-2)|TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-7B SFT LEAN-COT LEAN-STAR<br>(ITER-2)|
|---|---|---|---|---|
||IMO<br>AIME<br>AMC||20<br>15<br>45<br>14<br>14<br>14<br>14<br>14<br>16<br>11<br>11<br>11<br>11<br>18<br>8<br>8|0<br>0<br>0<br>0<br>2<br>1<br>2<br>**3**<br>3<br>3<br>**7**<br>5<br>1<br>2<br>**3**<br>**3**<br>**7**<br>**7**<br>**7**<br>**7**<br>9<br>9<br>**11**<br>**11**<br>10<br>10<br>9<br>**11**<br>9<br>**10**<br>**10**<br>**10**<br>6<br>6<br>6<br>**7**<br>**5**<br>**5**<br>4<br>**5**<br>4<br>5<br>5<br>**6**<br>**6**<br>5<br>5<br>**6**<br>8<br>8<br>**9**<br>**9**<br>0<br>**1**<br>**1**<br>**1**<br>0<br>0<br>0<br>0<br>0<br>0<br>**1**<br>**1**|
|MATH|ALGEBRA|LEVEL5<br>LEVEL4<br>LEVEL3<br>LEVEL2<br>LEVEL1|||
||NUMBERTHEORY|LEVEL5<br>LEVEL4<br>LEVEL3<br>LEVEL2<br>LEVEL1|||
|CUSTOM|ALGEBRA<br>NUMBERTHEORY<br>INDUCTION||||



of our thought-augmented theorem proving model as discussed in (§3.3). For each problem, LeanSTaR samples _K_ = 32 times in parallel with temperature _T_ = 0 _._ 7, and limits the number of times a tactic can be generated to a total of _N_ = 50. We also limit sampling to 10 minutes per problem. We choose these settings to resemble the inference budget used in our baselines, which follow previous work [44, 6, 51]. Namely, for best-first search baselines we use beam search to generate the next tactic with _S_ = 32 _, K_ = 1 [44, 6, 51]. We do not compare with methods designed for other formal languages such as Jiang et al. [23], Xin et al. [49] since language differences greatly influence the pass rate due to the different tactics and automation. We also do not compare with Lample et al. [25] since they only report _S_ = 32 _, K_ = 64 on best-first search, which is approximately equivalent to _S_ = 1 _, K_ = 512 for the sampling method, which is too computationally expensive for us. 

## **4.2 Main Results** 

Our main results are reported in Table 1. Lean-STaR gives a significant improvement over the previous state-of-the-art in Lean. For instance, with a similar inference budget, Lean-STaR achieves 34.8% versus 30 _._ 3% in InternLM2 [51] using best-first search and 30 _._ 7% in COPRA [39] using GPT-4. With a larger compute budget, Lean-STaR’s performance improves further to 36.1%. 

**Thought augmentation improves theorem proving.** The first phase of Lean-STaR trains a model to interleave thoughts and tactics, by fine-tuning on a synthesized dataset of thought-augmented examples. The fine-tuned model from this phase, denoted LEAN-COT in Table 1, achieves a pass rate of 32 _._ 8%, which is higher than the model prior to this phase, denoted SFT (29.5%). We conclude that the first phase of Lean-STaR can improve the theorem proving ability of a language model, even one that is already specialized for generating tactics in Lean such as the SFT model. 

**Bootstrapping improves thought-augmented theorem proving.** The second phase of Lean-STaR consists of generating new thoughts and tactics with the current language model, saving those that result in correct proofs, and training on the union of the initial thought-augmented dataset and the saved examples (i.e., expert iteration [34, 52, 36]). Refer to Appendix A.1 for details. 

We perform two iterations of expert iteration, and present the results in Table 1, denoted LEAN-STAR. Each iteration improves the model’s theorem proving performance, from 32.8% (the initial model) to 34% (LEAN-STAR after iteration 1) to 34.8% (LEAN-STAR after iteration 2). Furthermore, we find that the model is amenable to further improvement via additional sampling, achieving 36.1% 

8 

Table 3: **Pass rates about InternLM2-Plus-7B on the minif2f-test dataset with Lean.** This table shows the pass rates of previous works and our work. _S_ is the number of tactics attempted at each expanded node (assumed to be 1 in sampling) and _K_ is the total number of search or sampling attempts per problem. In sampling we use temperature 0.7, and in search we use beam search when generating the next tactic. Note that we sample 32 examples twice when _K_ = 64 in sampling. In search we follow the setup in [51] and limit each search to 20 minutes. 

|APPROACH|DECODING|_N_|_K_|_S_|PASS RATE|
|---|---|---|---|---|---|
|INTERNLM2-PLUS-7B[51] (FEW-SHOT) (FROM|SEARCH|1000|1|32|43_._4%|
|PAPER)||||||
|INTERNLM2-PLUS-7B [51] (FEW-SHOT) (REPRO-|SEARCH|1000|1|32|42_._6%|
|DUCED)||||||
|INTERNLM2-PLUS-7B[51] (FEW-SHOT)|SAMPLING|50|32|1|40_._9%|
|SFT (INTERNLM2-PLUS-7B) [51] (FEW-SHOT)|SAMPLING|50|32|1|41_._3%|
|**LEAN-COT** (INTERNLM2-PLUS-7B)|SAMPLING|50|32|1|43_._4%|
|**LEAN-STAR (ITER-1)**(INTERNLM2-PLUS-7B)|SAMPLING|50|32|1|45_._4%|
|INTERNLM2-PLUS-7B[51] (FEW-SHOT)|SAMPLING|50|64|1|42_._2%|
|SFT (INTERNLM2-PLUS-7B) [51] (FEW-SHOT)|SAMPLING|50|64|1|43_._4%|
|**LEAN-COT** (INTERNLM2-PLUS-7B)|SAMPLING|50|64|1|45_._5%|
|**LEAN-STAR (ITER-1)**(INTERNLM2-PLUS-7B)|SAMPLING|50|64|1|**46**_._**3**%|



Table 4: Performence of SFT-Direct and our Lean-STaR at different search size or sampling times _S × K_ . We fix _N_ = 50. We use beam search in search and temperature _T_ = 0 _._ 7 in sampling when generating the next tactic. We have _K_ = 1 in search and _S_ = 1 in sampling. Note that we sample 32 examples twice when _K_ = 64 in sampling. 

||||SFT-DIRECT(SEARCH)|SFT-DIRECT(SAMPLING)|LEAN-STAR (ITER-2) (SAMPLING)|
|---|---|---|---|---|---|
|_S × K_|=|1|13_._5%|20_._9%|21_._7%|
|_S × K_|=|2|18_._0%(+4_._5%)|22_._5%(+1_._6%)|24_._6%(+2_._9%)|
|_S × K_|=|4|23_._3%(+5_._3%)|25_._0%(+2_._5%)|27_._5%(+2_._9%)|
|_S × K_|=|8|27_._5%(+4_._2%)|27_._0%(+2_._0%)|30_._7%(+3_._2%)|
|_S × K_|=|16|29_._9%(+2_._4%)|28_._3%(+1_._3%)|33_._6%(+2_._9%)|
|_S × K_|=|32|30_._7%(+0_._8%)|29_._5%(+1_._2%)|34_._8%(+1_._2%)|
|_S × K_|=|64|30_._7%(+0_._0%)|30_._3%(+0_._8%)|36_._1%(+1_._3%)|



by doubling the sampling budget. We conclude that Lean-STaR’s second phase can further improve a model’s ability to generate thoughts and tactics that lead to correct proofs. We include three qualitative examples in the Appendix, which show the model interleaving thoughts and proof steps. 

## **4.3 Performance Analysis by Types and Difficulties** 

Tasks in _minif2f-test_ are manually formalized from Olympiad type problems, drawn from multiple sources including AIME, AMC, IMO problems, and problems from the MATH dataset [21]. These problems can have different levels of difficulty and types. Table 2 reports the number of problems successfully proved, partitioned by type and difficulty. We see that Lean-CoT improves performance in solving difficult problems on all categories, especially those from mathematics competitions. On top of these improvements, Lean-STaR’s improvements come mainly in Number Theory. 

9 

Table 5: Results for the InternLM2-plus-7b and our Lean-CoT, Lean-STaR, and expert iteration without CoT. We use sampling with _N_ = 50 _, K_ = 32 _,_ & _T_ = 0 _._ 7. 

|ithout CoT. We use samplingwith_N_|= 50_, K_ = 32_,_&_T_ = 0_._7.|||
|---|---|---|---|
|APPROACH|# (CONTINUAL) TRAININGDATA|_Pass@32_||
|INTERNLM2-PLUS-7B(FEW-SHOT)|-|40_._9%|-|
|SFT|231_,_240|41_._3%|+0_._4%|
|LEAN-COT|140_,_000|43_._4%|+**2**_._**1**%|
|LEAN-STAR|60_,_000|45_._5%|+**2**_._**1**%|
|EXPERT ITERATION(SFT)|60_,_000|43_._0%|+1_._7%|



## **4.4 Search and sampling budget** 

Table 4 reports the trends of the pass rate against the search size or sampling budget _S × K_ . We find that Lean-STaR benefits more as _K_ increases, especially when _K_ is relatively large. The result suggests that additional sampling with thoughts improves performance, while additional sampling without thoughts may saturate. We believe this is because thoughts increase the diversity of outputs and contribute to exploration in the theorem proving space. Therefore, Lean-STaR is more scalable (in terms of inference-time compute), and may be amenable to further improvements with additional iterations of expert iteration. 

## **4.5 Experiments with stronger base model and more data** 

We instantiate Lean-STaR using a stronger language model (InternLM2-Math-plus-7b [51]), which was released after the experiment above. We follow a similar setup to the previous experiment. 

In this experiment, we used 140 _,_ 000 thoughts annotated by GPT-4o [32] to fine-tune a model (“LeanCoT”). Then we performed only one iteration of expert iteration and collected about 60 _,_ 000 (proof state, thoughts, next-tactic) pairs in data, named “STaR dataset” _D_ 1. We further fine-tuned the Lean-CoT model on dataset _D_ 1 to get the Lean-STaR model. 

Our new results are reported in Table 3. We can see that Lean-STaR still gives a significant improvement over the baseline. For instance, Lean-STaR achieves 45 _._ 4% versus 39 _._ 8% in InternLM-plus using sampling with a similar inference budget and 43 _._ 4% using best-first search with more inference budget reported in [51]. This results show that both retrospective rationale generation and expert iteration can improve the theorem-proving capabilities on a stronger base model. 

## **4.6 Experiments on expert iteration without CoT** 

Table 5 shows the result of expert iteration without CoT (i.e., using (state, tactic) pairs only) as well as the result of Lean-CoT and Lean-STaR. Expert iteration alone achieves 43.0%, which is less than Lean-STaR (45.4%). This shows that Lean-STaR’s performance gains do not only come from the use of expert iteration. 

## **5 Conclusion & Limitations** 

In this paper, we presented Lean-STaR, a novel approach that significantly enhances the theoremproving capabilities of language models in formal mathematics by integrating Chain-of-Thought (CoT) rationales into each proof step. Our method begins with generating synthetic rationales using ground-truth tactics retrospectively, followed by fine-tuning the language model to generate these rationales and predict subsequent tactics, resulting in the Lean-CoT model. We further improved this model using expert iteration, fine-tuning it on correct proofs it samples and verifies using the Lean solver. Our contributions include the introduction of the first thought-augmented theorem proving dataset, demonstrating that expert iteration can further improve performance, and achieving new state-of-the-art results on the miniF2F-test benchmark, increasing the pass rate from 30 _._ 3% to 36 _._ 1%. 

10 

These advancements are not only about improving the accuracy of automated theorem proving, but also offer a scalable and efficient framework for advancing human understanding of mathematics, which may lead to significant impacts in education, scientific discovery, and program verification [11, 24, 37, 4, 16, 31]. 

The primary limitation of our method is that its performance may be constrained by issues of computational scalability. Both Lean-CoT and Lean-STaR have been fine-tuned on a dataset that is not very large. Additionally, the use of GPT-4 to generate synthetic data may incur a significant cost and possibly introduce biases. Also, expert iteration could face a bottleneck due to CPU and IO limitations, which might slow down the process due to a sluggish speed of Lean ITP. 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [2] Ayush Agrawal, Siddhartha Gadgil, Navin Goyal, Ashvni Narayanan, and Anand Tadipatri. Towards a mathematics formalisation assistant using large language models. _arXiv preprint arXiv:2211.07524_ , 2022. 

- [3] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. _Advances in neural information processing systems_ , 30, 2017. 

- [4] Jeremy Avigad. Mathematics and the formal turn, 2023. 

- [5] Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W Ayers, Dragomir Radev, and Jeremy Avigad. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. _arXiv preprint arXiv:2302.12433_ , 2023. 

- [6] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. _arXiv preprint arXiv:2310.10631_ , 2023. 

- [7] Jasmin Christian Blanchette, Cezary Kaliszyk, Lawrence C Paulson, and Josef Urban. Hammering towards qed. _Journal of Formalized Reasoning_ , 9(1):101–148, 2016. 

- [8] Sascha Bohme and Tobias Nipkow. Sledgehammer: judgement day. In _Automated Reasoning: 5th International Joint Conference, IJCAR 2010, Edinburgh, UK, July 16-19, 2010. Proceedings 5_ , pages 107–121. Springer, 2010. 

- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [10] Kevin Buzzard. Lean in 2024. `https://xenaproject.wordpress.com/2024/01/20/ lean-in-2024/` , 2024. 

- [11] Nathan C Carter and Kenneth G Monks. Lurch: a word processor that can grade students’ proofs. In _CICM Workshops_ , 2013. 

- [12] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _arXiv preprint arXiv:2211.12588_ , 2022. 

- [13] Projet Coq. The coq proof assistant-reference manual. _INRIA Rocquencourt and ENS Lyon, version_ , 5, 1996. 

- [14] Lukasz Czajka and Cezary Kaliszyk. Hammer for coq: Automation for dependent type theory. _Journal of automated reasoning_ , 61:423–453, 2018. 

- [15] Leonardo De Moura, Soonho Kong, Jeremy Avigad, Floris Van Doorn, and Jakob von Raumer. The lean theorem prover (system description). In _Automated Deduction-CADE-25: 25th International Conference on Automated Deduction, Berlin, Germany, August 1-7, 2015, Proceedings 25_ , pages 378–388. Springer, 2015. 

11 

- [16] Emily First. _Automating the Formal Verification of Software_ . PhD thesis, 2023. URL `https: //scholarworks.umass.edu/dissertations_2/2812` . 

- [17] Emily First, Markus Rabe, Talia Ringer, and Yuriy Brun. Baldur: Whole-proof generation and repair with large language models. In _Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering_ , ESEC/FSE 2023, page 1229–1241, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400703270. doi: 10.1145/3611643.3616243. URL `https://doi.org/ 10.1145/3611643.3616243` . 

- [18] Fabian Gloeckle, Baptiste Roziere, Amaury Hayat, and Gabriel Synnaeve. Temperature-scaled large language models for lean proofstep prediction. In _The 3rd Workshop on Mathematical Reasoning and AI at NeurIPS’23_ , 2023. 

- [19] Jesse Michael Han, Jason Rute, Yuhuai Wu, Edward W Ayers, and Stanislas Polu. Proof artifact co-training for theorem proving with language models. _arXiv preprint arXiv:2102.06203_ , 2021. 

- [20] Jesse Michael Han, Jason Rute, Yuhuai Wu, Edward Ayers, and Stanislas Polu. Proof artifact co-training for theorem proving with language models. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum?id=rpxJc9j04U` . 

- [21] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- [22] Yinya Huang, Xiaohan Lin, Zhengying Liu, Qingxing Cao, Huajian Xin, Haiming Wang, Zhenguo Li, Linqi Song, and Xiaodan Liang. Mustard: Mastering uniform synthesis of theorem and proof data. _arXiv preprint arXiv:2402.08957_ , 2024. 

- [23] Albert Q Jiang, Sean Welleck, Jin Peng Zhou, Wenda Li, Jiacheng Liu, Mateja Jamnik, Timothee Lacroix, Yuhuai Wu, and Guillaume Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. _arXiv preprint arXiv:2210.12283_ , 2022. 

- [24] Dongyeop Kang, Andrew Head, Risham Sidhu, Kyle Lo, Daniel S Weld, and Marti A Hearst. Document-level definition detection in scholarly documents: Existing models, error analyses, and future directions. _arXiv preprint arXiv:2010.05129_ , 2020. 

- [25] Guillaume Lample, Timothee Lacroix, Marie-Anne Lachaux, Aurelien Rodriguez, Amaury Hayat, Thibaut Lavril, Gabriel Ebner, and Xavier Martinet. Hypertree proof search for neural theorem proving. _Advances in neural information processing systems_ , 35:26337–26349, 2022. 

- [26] Zhaoyu Li, Jialiang Sun, Logan Murphy, Qidong Su, Zenan Li, Xian Zhang, Kaiyu Yang, and Xujie Si. A survey on deep learning for theorem proving, 2024. 

- [27] The mathlib Community. The lean mathematical library. In _Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs_ , CPP 2020, pages 367– 381, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450370974. doi: 10.1145/3372885.3373824. URL `https://doi.org/10.1145/3372885.3373824` . 

- [28] Allen Newell and Herbert Simon. The logic theory machine–a complex information processing system. _IRE Transactions on information theory_ , 2(3):61–79, 1956. 

- [29] Tobias Nipkow, Markus Wenzel, and Lawrence C Paulson. _Isabelle/HOL: a proof assistant for higher-order logic_ . Springer, 2002. 

- [30] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

- [31] National Academies of Sciences. Artificial intelligence to assist mathematical reasoning: Proceedings of a workshop, 2023. 

- [32] OpenAI. OpenAI: GPT-4, 2023. URL `https://openai.com/research/gpt-4` . 

12 

- [33] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. _arXiv preprint arXiv:2009.03393_ , 2020. 

- [34] Stanislas Polu, Jesse Michael Han, Kunhao Zheng, Mantas Baksys, Igor Babuschkin, and Ilya Sutskever. Formal mathematics statement curriculum learning. _arXiv preprint arXiv:2202.01344_ , 2022. 

- [35] Markus N Rabe, Dennis Lee, Kshitij Bansal, and Christian Szegedy. Mathematical reasoning via self-supervised skip-tree training. _arXiv preprint arXiv:2006.04757_ , 2020. 

- [36] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. Beyond human data: Scaling self-training for problem-solving with language models. _arXiv preprint arXiv:2312.06585_ , 2023. 

- [37] Christian Szegedy. A promising path towards autoformalization and general artificial intelligence. In _Intelligent Computer Mathematics: 13th International Conference, CICM 2020, Bertinoro, Italy, July 26–31, 2020, Proceedings 13_ , pages 3–20. Springer, 2020. 

- [38] Terence Tao. `https://mathstodon.xyz/@tao/111287749336059662` , 2023. 

- [39] Amitayush Thakur, Yeming Wen, and Swarat Chaudhuri. A language-agent approach to formal theorem-proving. _arXiv preprint arXiv:2310.04353_ , 2023. 

- [40] Haiming Wang, Huajian Xin, Chuanyang Zheng, Lin Li, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, Jian Yin, Zhenguo Li, Heng Liao, and Xiaodan Liang. Lego-prover: Neural theorem proving with growing libraries, 2023. 

- [41] Haiming Wang, Ye Yuan, Zhengying Liu, Jianhao Shen, Yichun Yin, Jing Xiong, Enze Xie, Han Shi, Yujun Li, Lin Li, et al. Dt-solver: Automated theorem proving with dynamic-tree sampling guided by proof-level value function. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 12632–12646, 2023. 

- [42] Qingxiang Wang, Chad Brown, Cezary Kaliszyk, and Josef Urban. Exploration of neural machine translation in autoformalization of mathematics in mizar. In _Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs_ , pages 85–98, 2020. 

- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [44] Sean Welleck and Rahul Saha. Llmstep: Llm proofstep suggestions in lean. _arXiv preprint arXiv:2310.18457_ , 2023. 

- [45] Sean Welleck, Jiacheng Liu, Ronan Le Bras, Hannaneh Hajishirzi, Yejin Choi, and Kyunghyun Cho. Naturalproofs: Mathematical theorem proving in natural language. _arXiv preprint arXiv:2104.01112_ , 2021. 

- [46] Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, and Yejin Choi. Naturalprover: Grounded mathematical proof generation with language models. _Advances in Neural Information Processing Systems_ , 35:4913–4927, 2022. 

- [47] Yuhuai Wu, Markus N Rabe, Wenda Li, Jimmy Ba, Roger B Grosse, and Christian Szegedy. Lime: Learning inductive bias for primitives of mathematical reasoning. In _International Conference on Machine Learning_ , pages 11251–11262. PMLR, 2021. 

- [48] Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. Autoformalization with large language models. _Advances in Neural Information Processing Systems_ , 35:32353–32368, 2022. 

- [49] Huajian Xin, Haiming Wang, Chuanyang Zheng, Lin Li, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, et al. Lego-prover: Neural theorem proving with growing libraries. _arXiv preprint arXiv:2310.00656_ , 2023. 

13 

- [50] Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. LeanDojo: Theorem proving with retrieval-augmented language models. In _Neural Information Processing Systems (NeurIPS)_ , 2023. 

- [51] Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu, Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin. Internlm-math: Open math large language models toward verifiable reasoning, 2024. 

- [52] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. _Advances in Neural Information Processing Systems_ , 35:15476–15488, 2022. 

- [53] Xueliang Zhao, Wenda Li, and Lingpeng Kong. Decomposing the enigma: Subgoal-based demonstration learning for formal theorem proving, 2024. URL `https://openreview.net/ forum?id=xLoxMvO695` . 

- [54] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. _arXiv preprint arXiv:2109.00110_ , 2021. 

- [55] Denny Zhou. Teach language models to reason. `https://dennyzhou.github.io/ LLMs-Reason-Taiwan-2023.pdf` , 2023. Accessed: 2024-05-21. 

- [56] Jin Peng Zhou, Charles Staats, Wenda Li, Christian Szegedy, Kilian Q Weinberger, and Yuhuai Wu. Don’t trust: Verify–grounding llm quantitative reasoning with autoformalization. _arXiv preprint arXiv:2403.18120_ , 2024. 

- [57] Jin Peng Zhou, Charles E Staats, Wenda Li, Christian Szegedy, Kilian Q Weinberger, and Yuhuai Wu. Don’t trust: Verify – grounding LLM quantitative reasoning with autoformalization. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https: //openreview.net/forum?id=V5tdi14ple` . 

14 

## **A Additional Experiment Setup** 

## **A.1 Lean-STaR Expert Iteration** 

The second phase of Lean-STaR consists of generating new thoughts and tactics with the current language model, saving those that result in correct proofs, and training on the union of the initial thought-augmented dataset and the saved examples (i.e., expert iteration [34, 52, 36]). We perform two iterations of expert iteration, and provide details on our specific experimental setup below. 

In each iteration we use sampling on the _LeanDojo Benchmark 4_ dataset, and save the (state, thought, tactic) examples that are part of successful proofs. For each problem, we sample _K_ = 32 times in parallel with temperature _T_ = 1 _._ 0, and limit the number of times a tactic can be generated to a total of _N_ = 5 per problem. Also, sampling is limited to 1 minute per problem. In this setup, each problem needs on average about 0 _._ 5 A100 minutes. We collect successfully sampled trajectories to produce a “STaR dataset” _D_ 1, and up to 3 proof trajectories were collected for each problem. We collected 32 _,_ 231 different (proof state, thoughts, next-tactic) pairs in successful proof trajectories during expert iteration, which takes about 4 days with 8 _× A_ 100 GPUs. Then, we further fine-tune SFT model for 1 epoch on the combination of GPT-4 annotated reasoning data and expert iteration data _DT ∪ D_ 1 to get the Lean-STaR model. We use the same learning rate setup that was used for the SFT model. In the second iteration, we generate a dataset _D_ 2 in a similar fashion. Then, we chose to further fine-tune model from iteration 1, _M_ 1, on the generated dataset _D_ 2 (roughly 19k pairs). 

The setup of experiment about InternLM2-plus is slightly different. The details are shown in Section 4.5 and Appendix E. 

## **B Statistics for our methods as well as the baselines** 

Table 6: Statistics for the baselines and our Lean-CoT, Lean-STaR on _MiniF2F_ dataset. We use sampling method with hyperparameters _N_ = 50 & _K_ = 32 & _T_ = 0 _._ 7. 

|mplingmethod with hyperparameters|_N_ = 50 &_K_ = 32 &_T_ = 0_._7.|||
|---|---|---|---|
|APPROACH|# (CONTINUAL) TRAININGDATA|_Pass@32_||
|INTERNLM2-MATH-7B(FEW-SHOT)|-|28_._7%|-|
|SFT|231_,_240|29_._5%|+0_._8%|
|**LEAN-COT**|52_,_438|32_._8%|**+3****_._3**%|
|**LEAN-STAR (ITER-1)**|32_,_231|34_._0%|+1_._2%|
|**LEAN-STAR (ITER-2)**|19_,_324|**34**_._**8**%|+0_._8%|



15 

## **C An Example and Explanation of A Formal Proof in Lean** 

An example of a formal proof in Lean with its visualization is shown in Figure 6, taken from [25]. In the proof, the tactic `induction k` is is applied to the initial state ( _n ≤ m ⇒ n_ + _k ≤ m_ + _k_ ) and the ITP converts the current state to subgoals `case 0` _∧_ `case ih` : _n ≤ m ∧ n_ + _k ≤ m_ + _k ⇒ n_ + ( _k_ + 1) _≤ m_ + ( _k_ + 1). The `case 0:` _n ≤ m_ is our hypothesis _h_ 0 so it can be proven by `case 0:exact` _h_ 0 tactic. Then, we rewrite the `case ih` through the `nat.succ_le_succ_iff` which is a theorem in Lean library means _n ≤ m ⇔ n_ + 1 _≤ m_ + 1. After tactics `case 0:exact` _h_ 0 and `case ih:rw nat.succ_le_succ_iff` , the goal state is converted to _n_ + _k ≤ m_ + _k_ which is the hypothesis introduced by induction. Therefore, we can complete this proof using tactic `exact k_ih` . 

`theorem add_le_add_right (m n k :` N `) (h` 0 `: n` _≤_ `m) : n + k` _≤_ `m + k := induction k with | zero => exact h` 0 `| succ k ih => rw Nat.succ_le_succ_iff exact ih` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving_images/iclr_2025_Lean-STaR_Learning_to_Interleave_Thinking_and_Proving.pdf-0016-03.png)


Figure 6: **A example proof and its visualization of** _n ≤ m ⇒ n_ + _k ≤ m_ + _k_ **in Lean, taken from [25].** The `induction` tactic reduces the initial statement to two subgoals. Then tactics `case 0:exact` _h_ 0 and `case ih:rw nat.succ_le_succ_iff` , `case ih:exact k_ih` can be applied in turn to complete the proof. 

16 

## **D Performance Analysis by Types and Difficulties using InternLM2-plus-7b** 

Table 7 reports the number of problems successfully proved, partitioned by type and difficulty using InternLM2-plus. We see that Lean-CoT improves performance mainly in Number Theory and LeanSTaR improves performance in solving difficult problems on all categories, which is the opposite of the performance of the InternLM2-base. 

Table 7: Counts of problems successfully proved in _minif2f-test_ benchmark using InternLM2-plus-7b, split by type and difficulty. The methods use sampling with _N_ = 50 _, K_ = 32. 

||TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-PLUS-7B LEAN-COT LEAN-STAR<br>(ITER-1)|TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-PLUS-7B LEAN-COT LEAN-STAR<br>(ITER-1)|TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-PLUS-7B LEAN-COT LEAN-STAR<br>(ITER-1)|TOTAL<br>TESTSET<br>SIZE<br>INTERNLM2-PLUS-7B LEAN-COT LEAN-STAR<br>(ITER-1)|
|---|---|---|---|---|
||IMO<br>AIME<br>AMC||20<br>15<br>45<br>14<br>14<br>14<br>14<br>14<br>16<br>11<br>11<br>11<br>11<br>18<br>8<br>8|0<br>0<br>0<br>3<br>3<br>**4**<br>9<br>9<br>**10**<br>**6**<br>**6**<br>**6**<br>**9**<br>**9**<br>**9**<br>11<br>**13**<br>**13**<br>**11**<br>**11**<br>**11**<br>**10**<br>**10**<br>**10**<br>**7**<br>**7**<br>**7**<br>6<br>**8**<br>**8**<br>6<br>7<br>**9**<br>7<br>**9**<br>**9**<br>**10**<br>**10**<br>**10**<br>**4**<br>3<br>**4**<br>0<br>0<br>0<br>**1**<br>**1**<br>**1**|
|MATH|ALGEBRA|LEVEL5<br>LEVEL4<br>LEVEL3<br>LEVEL2<br>LEVEL1|||
||NUMBERTHEORY|LEVEL5<br>LEVEL4<br>LEVEL3<br>LEVEL2<br>LEVEL1|||
|CUSTOM|ALGEBRA<br>NUMBERTHEORY<br>INDUCTION||||



17 

## **E Performance difference of joint training and continue training** 

As shown in Table 8, the joint training method performs better using InternLM2-base but training method performs much better using InternLM2-plus. It seems that there are no difference between these two methods. Therefore, this performance can be depend on the quantity of data or the model. (We use much more data when using InternLM2-plus and the quantity of "STaR data" is relatively small.) 

Table 8: Performance difference of joint training and continue training on Lean-STaR. We use sampling method with hyperparameters _N_ = 50 & _K_ = 32 & _T_ = 0 _._ 7. In continue training, we further fine-tune the Lean-CoT model on "STaR data" to get Lean-STaR model and in joint training we fine-tune the SFT model on combination of GPT-4 annotated reasoning data and "STaR data". 

|APPROACH|INTERNLM2-BASE-7B|INTERNLM2-PLUS-7B|
|---|---|---|
|**LEAN-COT**|32_._8%|43_._4%|
|**LEAN-STAR (ITER-1)**<br>(JOINT TRAINING)|**34.0%**|43_._9%|
|**LEAN-STAR (ITER-1)**<br>(CONTINUE TRAINING)|33_._2%|**45.5%**|



## **F Retrospective Rationale Generation Prompt of GPT-4** 

```
Pleaseactasaprofessionalmathematician.
YourgoalistoaccuratelyproveamaththeoreminLean4.
YouaregiventhefirsttacticthatshouldbetakentoprovetheGivenTheorem.
```

```
#Example:
```

`Given Theorem: ```lean4 X : Type u_1 Y : Type u_2 inst` ✝[1] `: TopologicalSpace X inst` ✝ `: TopologicalSpace Y f : X` _→_ `. Y h : PContinuous f` _⊢_ `IsOpen (PFun.preimage f Set.univ) ``` Given Tactic: ```lean4 exact h _ isOpen_univ ``` Assistant Output: ### PLANNING OF REASONING Given the above goal and the given the target tactic `exact h _ isOpen_univ`, I should produce a reasoning that only based on the goal, and conclude to the ` exact h _ isOpen_univ`, but not in the hindsight format.` 

```
###REASONING
```

```
Bydirectlyusingthehypothesis`h:PContinuousf`,westatethatforanyopenset
,itspreimagevia`f`isalsoopen,andthisappliesto`Set.univ`aswell.
```

```
###TACTIC
```

18 

```
```lean4
```

```
exacth_isOpen_univ
```
```

```
Toachievethegoal,youhavethreejobs.
```

```
#Writedowntheplanningofreasoning.
```

```
#WritedownthereasoningthatcanresultintheselectionoftheGivenTactic.
#WritedowntheGivenTactic.
Youhavefourprinciplestodothis.
```

```
#Craftreasoningthatiscomprehensiveyetconcise,containedideallywithintwo
sentences.
```

```
#BeginreasoningwiththeobjectiveoftheGivenTheorem,ensuringlogical
progression,ratherthanconstructingitretrospectivelyfromtheGivenTactic.
#AvoidmentioningtheGivenTacticandspecificstatementnamesfromtheGiven
Tacticthroughoutyourreasoning.
```

```
#Positiontheconclusionofthetacticafterpresentingallreasons,ensuringthat
detailsregardingtheGivenTacticarenotintroducedatthestartofyour
reasoning.
Youroutputshouldbestrictlyinthefollowingformatandshouldnotcontainextra
content:
```

## `### PLANNING OF REASONING` 

```
Giventheabovegoalandthegiventhetargettactic<GivenTactic>,Ishould
produceareasoningthatonlybasedonthegoal,andconcludetotheexact<
GivenTactic>,butnotinthehindsightformat.
```

## `### REASONING` 

```
<yourreasoningtotheGivenTactic>
```

```
###TACTIC
```

```
<GivenTactic>
```

```
#GivenTheorem:
```lean4
{theorem["state_before"]}
```
```

```
#GivenTactic:
```lean4
{theorem["tactic"]}
```
```

19 

## **G Examples of generated Lean proofs with thoughts** 

`theorem aime_1990_p4 (x :` R `) (h` 0 `: 0 < x) (h` 1 `: x ^ 2 - 10 * x - 29` = `0) (h` 2 `: x ^ 2 - 10 * x - 45` = `0) (h` 3 `: x ^ 2 - 10 * x - 69` = `0)` 

`(h` 4 `: 1 / (x ^ 2 - 10 * x - 29) + 1 / (x ^ 2 - 10 * x - 45) - 2 / (x ^ 2 - 10 * x - 69) = 0)` 

```
:x=13:=
```

`norm_num at h` 1 `h` 2 `h` 3 _`-- Reasoning: To prove `x = 13`, we examine the conditions under which the given equation holds, specifically focusing on the non-equality of the expressions involving `x ^ 2 - 10 * x - 29`, `x ^ 2 - 10 * x - 45`, and `x ^ 2 - 10 * x - 69`. Simplifying these expressions using properties of real numbers and algebra allows us to directly compare them with their respective values, ultimately identifying `x = 13` as the unique solution that satisfies all constraints.`_ 

```
field_simpat*--Reasoning:Toprove\(x=13\),weleveragethegiven
equalitiesinvolvingrationalfractionsof\(x^2-10x-c\)forvarious
constants\(c\),where\(0<x\)andeach\(x^2-10x-c\)isnon-zero.By
algebraicallymanipulatingtheseexpressions,weaimtoderivearelationship
among\(x\)andtheseconstants,specificallyfocusingon\(13\),underthe
givennon-zeroconditionsforthequadraticexpressions.
```

`ring_nf at h` 4 _⊢_ _`-- Reasoning: To establish the equality `x = 13` under the given conditions, we need to manipulate the algebraic expression given by `h`_ 4 _`` to reveal its dependence on `x`, recognizing that the presence of `x` in the expressions may affect the equation's truth value. The application of algebraic transformations on both sides of the equation will facilitate the identification of `x`'s value, taking into consideration the inequality conditions provided (`h`_ 1 _``, `h`_ 2 _``, `h`_ 3 _``).`_ 

`nlinarith` _`-- Reasoning: To prove \(x = 13\), we must manipulate the equation given by \(h`_ 4 _`\) to isolate \(x\), using the algebraic properties of equality and the assumption \(h`_ 0 _`\) which states \(0 < x\). This requires careful rearrangement and application of algebraic identities to simplify and rearrange terms, facilitating the direct application of the properties of real numbers to reach the desired conclusion.`_ 

`theorem amc12a_2002_p6 (n :` N `) (h` 0 `: 0 < n) :` _∃_ `m, m > n` _∧∃_ `p, m * p` _≤_ `m + p :=` 

`use n + 1, by linarith only [h` 0 `]` _`-- Reasoning: To demonstrate the existence of such an `m` and `p` that satisfy the given inequalities, we leverage the property of natural numbers that ensures an increase in `n` from `0` leads to the possibility of finding such `m` and `p`. This approach is supported by the fundamental property of natural numbers and their arithmetic operations, suggesting that for a given `n` greater than `0`, there exists a successor `m` and a number `p` that satisfy the conditions outlined.`_ 

`exact 〈0, by simp 〉` _`-- Reasoning: To demonstrate that there exists an \(n \in`_ N _`\) such that \((n + 1) * p \leq n + 1 + p\) holds for some \(p \in`_ N _`\), we recognize that setting \(p = 0\) satisfies the inequality by the properties of zero in arithmetic. This is because \(n + 1 > 0\) by assumption, ensuring the left-hand side is non-negative and the right-hand side simplifies to \(n + 1\), making the inequality trivially true.`_ 

20 

`theorem aime_1983_p2 (x p :` R `) (f :` R _→_ R `) (h` 0 `: 0 < p` _∧_ `p < 15) (h` 1 `: p` _≤_ `x` _∧_ `x` _≤_ `15) (h` 2 `: f x = abs (x - p) + abs (x - 15) + abs (x - p - 15)) : 15` _≤_ `f x :=` 

`cases' h` 1 `with h` 1 `h` 2 _`-- Reasoning: Given a function \(f\) defined over the real numbers, where \(f(x)\) equals the sum of the absolute values of the differences \(x - p\), \(x - 15\), and \(x - p - 15\), along with conditions on \(p\) and \(x\) that ensure \(x\) lies between \(p\) and \(15\) and \(p\) is positive, we aim to show that \(f(x) \geq 15\). To demonstrate this, we consider all possible cases for the position of \(x\) relative to \(p\) and \(15\), specifically at \(p\) itself, between \(p\) and \(15\), and at \(15\). By evaluating \(f(x)\) for each case, we can establish a lower bound of \(f(x)\) that is greater than or equal to \(15\).`_ 

`simp [h` 0 `, abs_of_nonneg (sub_nonneg.2 h` 1 `), abs_of_nonneg (sub_nonneg.2 h` 2 `), *]` _`-- Reasoning: The objective is to show that \(f(x)\) is at least 15 for \(x\) within the bounds \(p \le x \le 15\), where \(p\) is positive but less than 15. The provided expression for \(f(x)\) suggests using properties of absolute values and inequalities to simplify and analyze \(f(x)\). By applying simplification rules for absolute values and inequalities, along with the given bounds, we aim to demonstrate the desired inequality \(f(x) \ge 15\).`_ `simp only [abs_of_nonneg] at *` _`-- Reasoning: The objective is to establish an inequality involving an expression with absolute values and a sum, where all terms are functions of \(x\). By recognizing that the absolute value functions are always non-negative, we can simplify the given expression by removing the absolute value symbols, thus focusing on the arithmetic manipulation of the terms to demonstrate the inequality.`_ 

`simp only [abs_of_nonneg, sub_add_eq_add_sub] at *` _`-- Reasoning: The objective is to demonstrate that the expression `15` ≤_ _`x - p + |x - 15| + |x - p - 15|` holds given the conditions on `p`, `x`, and the function `f`. Since the expression in question involves absolute values which inherently involve non-negativity, manipulating these expressions to highlight their non-negativity or equality is a logical initial step. This involves recognizing that the absolute value of a difference can be rewritten as the difference of absolute values, which simplifies the manipulation of the expression to prove the inequality.`_ 

`congr 1` _`-- Reasoning: To establish the inequality `15` ≤_ _`x + |x - 15| + |x - p - 15| - p`, we recognize that the left-hand side of the inequality involves a mix of additions and subtractions of real numbers. By introducing a congruence transformation that simplifies operations involving absolute values, we aim to align both sides of the inequality in a form that directly reflects the properties of real numbers under these operations.`_ 

`simp only [abs_of_nonpos (sub_nonpos_of_le h` 2 `), abs_of_nonneg (sub_nonneg_of_le h` 1 `)]` _`-- Reasoning: The objective is to prove an inequality involving absolute values and subtractions that aligns the given conditions and constraints. Simplification using the properties of absolute values and the provided bounds can directly lead to the desired conclusion by expressing the absolute values in terms of their underlying expressions and applying the given constraints.`_ `simp [abs_of_pos, *]` _`-- Reasoning: To prove the given inequality involving absolute values, simplification using algebraic properties and given hypotheses will directly lead us to the desired inequality by recognizing the algebraic manipulation and application of the known bounds `h`_ 1 _`` and `h`_ 2 _``, as well as the non-negativity of `p`.`_ 

`rw [abs_of_nonpos]` _`-- Reasoning: The objective is to demonstrate that `15` ≤_ _`15 + |x - p - 15| - p` under the given constraints on `p` and `x`. Given `p < 15` and `p` ≤_ _`x` ≤_ _`15`, it's apparent that `p - 15` is non-positive. This observation allows us to apply the property that the absolute value of a non-positive number is equal to its negative value, thereby simplifying the expression on the right-hand side of the inequality.`_ `all_goals linarith` _`-- Reasoning: The goal involves demonstrating a relationship between expressions involving inequalities and subtraction. The simplification of these expressions into a form that directly compares their numerical values can lead to a straightforward application of known inequalities and algebraic properties, demonstrating the inequality's validity under the given conditions.`_ 

21 

