# **Self-playing Adversarial Language Game Enhances LLM Reasoning** 

**Pengyu Cheng**[1] **Tianhao Hu**[1] **Han Xu**[1] **Zhisong Zhang**[1] **Zheng Yuan**[4] **Yong Dai**[1] **Lei Han**[3] **Nan Du**[1] **Xiaolong Li**[2] Tencent AI Lab[1] Shenzhen &[2] Seattle 3Tencent Robotics X Lab 4Independent `pengyucheng@tencent.com` 

## **Abstract** 

We explore the potential of self-play training for large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker’s utterances. To win the game, both players must have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this informationreserved conversation. Hence, we are curious about whether LLMs’ reasoning ability can be further enhanced by **S** elf- **P** laying this **A** dversarial language **G** ame (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs’ performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs’ reasoning abilities. The code is available at `https://github.com/Linear95/SPAG` . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0001-05.png)


Figure 1: Reasoning improvements from **S** elf- **P** laying of **A** dversarial language **G** ames (SPAG) on comprehensive reasoning benchmarks. With the SPAG epoch increasing, the LLM reasoning ability continuously improves. Each axis is normalized by the maximum answer-accuracy value. 

## **1 Introduction** 

Large language models (LLMs), such as GPT-4 [OpenAI, 2023b] and Gemini [Team et al., 2023], have reformed the domain of artificial intelligence (AI) with astonishing language capacities, such as 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0002-00.png)


Figure 2: Examples of _Adversarial Taboo_ with the same target word “conversation”. The left shows an attacker-winning game, in which the defender unconsciously speaks out the target word. The right is a defender-winning episode because the defender makes the correct inference from the dialogue. 

natural language understanding [Yang et al., 2023b, Touvron et al., 2023], text generation [Kocon´ et al., 2023, Anil et al., 2023], machine translation [Jiao et al., 2023], summarization [Xie et al., 2024], and programming [Surameery and Shakor, 2023, Tian et al., 2023]. However, the reasoning ability of LLMs, which is essential for complex problem-solving [Pan et al., 2023] and advanced intelligence-developing [Yao et al., 2021], still retains being challenged by various criteria including correctness [Zhang et al., 2023a] and faithfulness [Turpin et al., 2024]. 

To address the reasoning challenge of LLMs, plenty of works have contributed in-depth efforts from the perspectives of Chain-of-Thought (CoT) prompt engineering [Wei et al., 2022, Ding et al., 2023, Yao et al., 2024], and the usage of auxiliary reasoning tools [Pan et al., 2023]. However, both prompt-based and tool-calling methods require additional prompt designs, which are inconsistent and sensitive to different prompt patterns and LLM checkpoints [Turpin et al., 2024, Chu et al., 2023]. More fundamental and consistent reasoning-improving approaches are post-pretraining [Azerbayev et al., 2023] and fine-tuning [Dong et al., 2023a], which trains LLMs with additional reasoning-related text corpus. Nevertheless, these methods demand sufficient high-quality textual data, which are difficult to collect due to the massive costs of human annotation efforts [Singh et al., 2023]. 

To improve LLM reasoning more efficiently, self-improvement methods, which enhance LLMs with model-generated synthetic data, have recently attracted increasing research attention [Singh et al., 2023, Huang et al., 2023, Burns et al., 2023, Chen et al., 2024]. Self-improvement methods usually utilize the intrinsic language capability of LLMs to judge [Huang et al., 2023], filter [Yuan et al., 2024], or revise [Yuan et al., 2024] self-generated samples to enhance their quality. However, most self-improvement methods rely on a broad range of high-quality question queries to prevent over-fitting into a sub-domain of reasoning tasks, which still requires additional data collection and cleaning. Besides, the judgments from LLMs are not guaranteed objective [Raina et al., 2024]. If an LLM already has an incorrect or biased recognition of a particular concept, the self-improvement process can reinforce and amplify the LLM’s cognitive dissonance. 

Towards more general and objective self-reasoning-improvement methods, we are inspired by the advancement from AlphaGO [Silver et al., 2016] to AlphaGO Zero [Silver et al., 2017], in which the game agents’ intelligence continuously promotes via self-play without any human knowledge. Analogically, we expect to set up a language game where LLMs can improve their reasoning capacities via reinforcement learning (RL) during self-play. Although language games have attracted increasing attention in natural language processing [Lewis et al., 2017, Hausknecht et al., 2020, Xu et al., 2023, Wu et al., 2024], most of them are specially designed with customized game rules, in lack of the generalization to improve the general language capacities of LLMs. Among a few generaltarget language games including red-teaming [Ma et al., 2023], negotiation [Lewis et al., 2017], and bargain [Abdulhai et al., 2023], additional human judgments or reward models are required for outcome determination, which posts challenges on the efficiency and effectiveness of large-scale self-play RL training. Recent studies have raised interest in entity- or word-based language games, such as _20-Question_ [Zhang et al., 2023b] and _Guess-My-City_ [Abdulhai et al., 2023] , which provide not only straight-forward word-level outcomes but also language universality by traversing the game word from comprehensive vocabularies. However, unlike the GO game, these word-based games are out of the adversarial scheme, limiting the game intensity and self-play learning effectiveness. 

With the above consideration, we select an adversarial language game called _Adversarial Taboo_ [Yao et al., 2021], in which an attacker and a defender perform a conversation around a target word 

2 

only visible to the attacker. The attacker aims to induce the defender to speak out the target word unconsciously; the defender tries to avoid unconscious utterance of the target word and guess the word from the dialogue history. To win the adversarial game in information-limited conversations, both players are required to have high-level language capacities in terms of expression, upstanding, and reasoning. Moreover, by collecting target words from a vast vocabulary, this game can cover a broad range of topics providing sufficient language versatility. Besides, the game outcomes can be automatically and explicitly judged: we only need to check whether the target word appears in the defender’s utterances (attacker wins) or its inference patterns (defender wins). We conduct the self-play on this adversarial game using open-source LLMs, LLaMA-2-7B [Touvron et al., 2023] and Baichuan-2-13B [Yang et al., 2023a], with target words selected from a 50K top-frequency vocabulary [Davies, 2020]. Next, we conduct offline reinforcement learning on the game outcomes and observe significant performance improvement on a broad range of LLM reasoning benchmarks. Furthermore, we iterate this sampling-learning process with three epochs, within which the LLMs’ reasoning can continuously obtain improvement. We believe this novel training scheme, **S** elf- **P** lay of **A** dversarial **G** ame (SPAG), has great potential for developing advanced LLM capacities. 

## **2 Preliminary** 

With the development of LLMs [OpenAI, 2023a,b], reinforcement learning (RL) has played an increasingly important role in language model training [Ouyang et al., 2022, Ramamurthy et al., 2023]. The prime application scenario for RL in LLM training is reinforcement learning from human feedback (RLHF) [Yuan et al., 2023, Cheng et al., 2023b, Zeng et al., 2023]. RLHF first learns a reward model _r_ ( _**x** ,_ _**y**_ ) from the human feedback preference pairs [Cheng et al., 2023a], and then optimizes the LLM policy _πθ_ ( _**y** |_ _**x**_ ) to maximize the expected reward value [Dong et al., 2023b]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0003-03.png)


To learn the above objective, proximal policy optimization (PPO) [Schulman et al., 2017] algorithm has been recognized as the mainstream solution. In each update to equation 1, PPO minimizes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0003-05.png)


where _πθ_ ¯ is a copy of _πθ_ before the update, _A_[ˆ] _[π] θ_[¯] is the estimated advantage value [Schulman et al., ¯ ¯ 2016] with respect to the reference policy _πθ_ , and KL[ _πθ∥πθ_ ] is the Kullback-Leibler (KL) [Kullback, 1997] divergence regularizing _πθ_ with an appropriate updating step. However, PPO for LLMs has been continually challenged due to its inefficient natural-text online sampling and the unstable training processes [Baheti et al., 2024]. Among the improvements to PPO [Rafailov et al., 2023, Yuan et al., 2023, Dong et al., 2023b], Baheti et al. [2024] adopts the PPO objective into an offline scheme by using importance sampling [Neal, 2001], which is named Advantage-Leftover-Lunch (A-LoL): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0003-07.png)


Here the sample _**y** ∼ π_ ref( _**y** |_ _**x**_ ) and advantage _A_[ˆ] _[π]_[ref] are both from the reference distribution _π_ ref( _**y** |_ _**x**_ ) and calculated offline. Besides, Gulcehre et al. [2023] proposed Reinforced Self-Training (ReST) to simplify the RLHF scheme. With a threshold _τ ∈_ R, ReST updates the LLM by the reinforcement on the selected samples _Dτ_ = _{_ ( _**x** ,_ _**y**_ ) : _r_ ( _**x** ,_ _**y**_ ) _> τ }_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0003-09.png)


where _Lθ_ ( _**x** ,_ _**y**_ ) could be any offline RL loss such as A-LoL or the vanilla language modeling loss. 

## **3 Self-play of Adversarial Language Games** 

The game of Adversarial Taboo is first introduced by Yao et al. [2021], in which an attacker _µ_ and a defender _ν_ involve a multi-turn conversation. At the beginning of the game, the attacker is assigned a target word _w ∈V_ target, which is not informed to the defender. The attacker’s target is to induce the defender to speak the target word unconsciously. To achieve this goal, the attacker can talk about any topic related to _w_ , except directly speaking out the target word. In contrast, the defender is required to infer the target word without any unconscious utterance of the word. If the defender has sufficient confidence to infer the word, it can yell “I know the word! It is _{target word}_ !”. Then the game terminates. If the guess is correct, the defender wins, otherwise the attacker wins. Besides, the game has a maximum number of turns _T_ 0. If nobody wins during _T_ 0 turns, there is a tie. The examples of Adversarial Taboo are shown in Figure 2. 

3 

**Algorithm 1** Data collection of LLM self-plays for the adversarial language game. 

**Inputs:** LLM policy _πθ_ ( _**y** |_ _**x**_ ), target word _w_ , attacker and defender prompts _f_ attack _, f_ defend. Set the initial state _**s**_ 0 = ( _w_ ). **for** _t_ from 1 to _T_ **do** 

Sample an attacker utterance _**u** t ∼ µθ_ ( _**u** t|_ _**s** t−_ 1) = _πθ_ ( _**y**_ = _**u** t|_ _**x**_ = _f_ attack( _**s** t−_ 1)). Update state _**s**[′] t_[= (] _[w,]_ _**[ u]**_[1] _[,]_ _**[ v]**_[1] _[, . . . ,]_ _**[ u]**[t][−]_[1] _[,]_ _**[ v]**[t][−]_[1] _[,]_ _**[ u]**[t]_[)][.] Sample a defender utterance _**v** t ∼ νθ_ ( _**v** t|_ _**s**[′] t_[) =] _[ π][θ]_[(] _**[y]**_[=] _**[ v]**[t][|]_ _**[x]**_[ =] _[ f]_[defend][(] _**[s]**[′] t_[))][.] Update state _**s** t_ = ( _w,_ _**u**_ 1 _,_ _**v**_ 1 _, . . . ,_ _**u** t−_ 1 _,_ _**v** t−_ 1 _,_ _**u** t,_ _**v** t_ ). **end for** Collect an episode _τ_ = ( _**s**_ 0 _,_ _**s**[′]_ 1 _[,]_ _**[ s]**_[1] _[, . . . ,]_ _**[ s]**[′] T[,]_ _**[ s]**[T]_[ )] 

## **3.1 Adversarial Language Game Modeling** 

We view the Adversarial Taboo as a two-player zero-sum Markov game [Littman, 1994], which can be described by a tuple as ( _S, A, F, r_ ): 

- The state space _S_ = _{_ _**s** t,_ _**s**[′] t_ : 1 _≤ t ≤ T_ 0 _}_ contains two types of states, _**s**[′] t_ = ( _w,_ _**u**_ 1 _,_ _**v**_ 1 _,_ _**u**_ 2 _, . . . ,_ _**u** t_ ) and _**s** t_ = ( _w,_ _**u**_ 1 _,_ _**v**_ 1 _,_ _**u**_ 2 _, . . . ,_ _**u** t,_ _**v** t_ ), where _{_ _**u** i}[t] i_ =1[and] _[{]_ _**[v]**[i][}][t] i_ =1[are] the utterances of the attacker and defender, respectively. Games start at _**s**_ 0 = ( _w_ ) with a target word _w ∈V_ target and end with _T_ 0 maximum turns. States _**s**[′] t_[and] _**[ s]**[t]_[end with utterances] _**[ u]**[t]_[and] _**[ v]**[t]_ for the defender and attacker to act, respectively. 

- The action space _A_ is shared with both the attacker and defender, which is equivalent to the token sequence space of natural language _N_ = _{_ _**x**_ = ( _x_ 1 _, x_ 2 _, . . . , xL_ ) _|xl ∈V_ token _, L ∈_ N+ _}_ . 

- The transition function _F_ : _S × A →S_ deterministically appends the utterance _**u** t_ or _**v** t_ at the end of the dialogue, and converts _**s**[′] t_[=] _[ F]_[(] _**[s]**[t][−]_[1] _[,]_ _**[ u]**[t]_[)][ and] _**[ s]**[t]_[=] _[ F]_[(] _**[s]**[′] t[,]_ _**[ v]**[t]_[)][.] 

- The reward _r_ : _S × A →_ R evaluates the actions _**u** ,_ _**v** ∈A_ based on their corresponding states _**s** ,_ _**s**[′] ∈S_ with rewards _r_ ( _**s** ,_ _**u**_ ) and _r_ ( _**s**[′] ,_ _**v**_ ), respectively. Given a game episode _**τ**_ = ( _**s**_ 0 _,_ _**s**[′]_ 1 _[,]_ _**[ s]**_[1] _[, . . . ,]_ _**[ s]**[′] T[,]_ _**[ s]**[T]_[ )][, we denote the] _[ attacker’s total reward][ R]_[(] _**[τ]**_[) =][ �] _[T] t_ =1 _[r]_[(] _**[s]**[t][−]_[1] _[,]_ _**[ u]**[t]_[)][, so the] _defender’s total reward_ is[�] _[T] t_ =1 _[r]_[(] _**[s]** t[′][,]_ _**[ v]**[t]_[)][=] _[−][R]_[(] _**[τ]**_[)][ to satisfy the] _[ zero-sum]_[ constraint.][More] detailed reward designs with heuristic rules for the Adversarial Taboo can be found in Appendix B. 

In the above game, we denote _µ_ ( _**u** |_ _**s**_ ) and _ν_ ( _**v** |_ _**s**[′]_ ) as the attacker’s and defender’s policies, respectively. Then each episode _**τ**_ can be regarded as a trajectory with the probability: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0004-10.png)


where _P_ ( _w_ ) is the data distribution of target word _w ∈V_ target. Then we can write the self-play objective of the Adversarial Taboo as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0004-12.png)


in which the attacker tries to maximize its total reward _R_ ( _**τ**_ ) by optimizing policy _µ_ , and the defender seeks strategies _ν_ to maximize the defender reward _−R_ ( _**τ**_ ) (minimize _R_ ( _**τ**_ )). To play the above adversarial game with an LLM generation policy _πθ_ ( _**y** |_ _**x**_ ), we first design prompt templates _f_ attack _, f_ defend : _S →N_ for the attacker and defender respectively to convert the game states into natural language task descriptions. Next, we introduce the game policies for the two players: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0004-14.png)


The detailed prompt templates for the game are demonstrated in Appendix A. 

## **3.2 Imitation Learning** 

Due to the limited capability of current open-source LLMs, the generation policy _πθ_ ( _**y** |_ _**x**_ ) can not guarantee the strict instruction-following of the game rules in prompts _f_ attack( _**s**_ ) and _f_ defend( _**s**[′]_ ). Therefore, before the self-play, we first conduct an imitation learning (behavior cloning) of GPT-4’s behaviors to ensure that _πθ_ ( _**u** |f_ attack( _**s**_ )) and _πθ_ ( _**v** |f_ defend( _**s**[′]_ )) act consistently with the game rules. To collect the game episodes of GPT-4 [Achiam et al., 2023], we use the data collection procedure 

4 

described in Algorithm 1. Similar to the setups in equation 7, we also design attacker and defender prompts for GPT-4 to act as the game players, which can be found in Appendix A.1. 

After collecting a group of GPT-4 game episodes for imitation learning as _T_ im, we divide it into an attacker-winning set _T_ im[attack] = _{_ _**τ** ∈T_ im : _R_ ( _**τ**_ ) _>_ 0 _}_ and a defender-winning set _T_ im[defend] = _{_ _**τ** ∈ T_ im : _R_ ( _**τ**_ ) _<_ 0 _}_ . The imitation learning loss is to maximize the log-likelihood of winners’ actions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0005-02.png)


where the re-weighting parameter _β_ 1 _>_ 0 and the regularizer KL[ _πθ∥π_ ref] prevents the model from over-fitting on the language game task and maintains the model’s general language abilities. The reference model _π_ ref is the initial checkpoint of the LLM before training. The overall imitation learning objective is _L_ im( _πθ_ ) =[1] 2 _[L]_ im[attack] ( _πθ_ ) +[1] 2 _[L]_ im[defend] ( _πθ_ ) _._ 

## **3.3 Reinforcement Learning from Self-play** 

Imitation learning enables LLM to behave consistently following the game rules. Next, we conduct self-play by letting the LLM _πθ_ ( _**y** |_ _**x**_ ) play alternatively as the attacker _µθ_ ( _**u** |_ _**s**_ ) = _πθ_ ( _**u** |f_ attack( _**s**_ )) and the defender _νθ_ ( _**v** |_ _**s**[′]_ ) = _πθ_ ( _**v** |f_ defend( _**s**[′]_ )). Note that the self-play sampling process involves massive multi-turn auto-regressive text generation of the LLM, which causes heavy computational complexity and makes the on-policy RL training intensively inefficient. Therefore, we use an offline ¯ learning scheme: (1) make a copy _πθ_ of current LLM policy _πθ_ ; (2) collect self-play episodes _Tθ_ ¯ = _{_ _**τ** ∼ µθ_ ¯ _× νθ_ ¯ _}_ from games between attacker _µθ_ ¯ and defender _νθ_ ¯; (3) update _πθ_ via RL training ¯ ¯ with _Tθ_ . The details of the collection of _Tθ_ are shown in Algorithm 1. 

With a group of episodes _Tθ_ ¯, we first fix the defender policy _νθ_ ¯ and consider updating the attacker policy _µθ_ with respect to the self-play objective _L_ AG( _µθ, νθ_ ¯) = E _**τ** ∼µθ×νθ_ ¯[[] _[R]_[(] _**[τ]**_[)]][.][We calculate the] corresponding policy gradient for the attacker as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0005-07.png)


where _A[µ] t[θ]_ = _A[µ][θ]_ ( _**s** t−_ 1 _,_ _**u** t_ ) is the advantage of action _**u** t_ for the attacker _µθ_ in the self-play of ¯ _µθ × νθ_ . Here we apply importance sampling to unbiasedly estimate the expectation _w.r.t µθ_ ( _**u** t|_ _**s** t−_ 1) ¯ ¯ with the sampled actions from _µθ_ ( _**u** t|_ _**s** t−_ 1) in _Tθ_ . Inspired by TRPO [Schulman et al., 2015] and PPO [Schulman et al., 2017], we design the following loss to optimize _L_ AG( _µθ, νθ_ ¯): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0005-09.png)


¯ where the re-weighting parameter _β_ 2 _>_ 0, and regularizer KL( _πθ∥πθ_ ) guarantees an appropriate policy update step size. Following TRPO [Schulman et al., 2015], we use empirical estimation _A_[ˆ] _[µ] t θ_[¯] ¯ of policy _µθ_ to approximate the advantage _A[µ] t[θ]_[.][More details of advantage estimation are described] in Appendix D. Similarly, from the perspective of the defender, we propose the corresponding loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0005-11.png)


In practice, we find when negative advantage values exist, the above policy gradient method can cause training instability and damage the LLMs’ general language performance. To mitigate the 

5 

## **Algorithm 2** Self-play of adversarial language games (SPAG) 

- **Inputs:** LLM policy _πθ_ ( _**y** |_ _**x**_ ), target word set _V_ target, attacker and defender prompts _f_ attack _, f_ defend. **for** self-play epoch iterations **do** 

Make a copy of _πθ_ as _πθ_ ¯. Set _µθ_ ¯( _**u** |_ _**s**_ ) = _πθ_ ¯( _**u** |f_ attack( _**s**_ )) and _νθ_ ¯( _**v** |_ _**s**[′]_ ) = _πθ_ ¯( _**v** |f_ defend( _**s**[′]_ )). For each _w ∈V_ target, sample a episode _**τ** ∼ µθ_ ¯ _× νθ_ ¯. Collect _Tθ_ ¯ = _{_ _**τ** ∼ µθ_ ¯ _× νθ_ ¯ _}_ Split the attacker-winning set _Tθ_ ¯[attack] and the defender-winning set _Tθ_ ¯[defend] . Update _πθ_ with loss _L_ SPAG( _πθ_ ). **end for** 

issue and obtain more stable RL training, we utilize the methodology of ReST [Gulcehre et al., 2023] as in equation 4, which considers the offline learning with samples _{_ _**τ**_ : _R_ ( _**τ**_ ) _> ξ}_ selected by a reward threshold _ξ_ . More specifically, we set the reward threshold _ξ_ = 0 and select the attacker-winning episodes _Tθ_ ¯[attack] = _{_ _**τ** ∈Tθ_ ¯ : _R_ ( _**τ**_ ) _>_ 0 _}_ and the defender-winning episodes _Tθ_ ¯[defend] = _{_ _**τ** ∈Tθ_ ¯ : _R_ ( _**τ**_ ) _<_ 0 _}_ for the attacker and the defender training respectively. Similar techniques have also been studied in earlier RL literature to obtain stable policy updates, such as self-imitation learning [Oh et al., 2018] and the UPGO [Vinyals et al., 2019] methods. Therefore, the overall self-play of adversarial language games (SPAG) objective is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0006-04.png)


where _α >_ 0 is a re-weighting hyper-parameter, and E _D_ SFT[log _πθ_ ( _**y** |_ _**x**_ )] is the log-likelihood on a supervised fine-tuning (SFT) dataset _D_ SFT to prevent LLMs from losing general language abilities. 

## **4 Experiments** 

To verify the effectiveness of SPAG, we select open-source pretrained LLMs of different sources and model sizes, particularly LLaMA-2-7B [Touvron et al., 2023] and Baichuan-2-13B [Yang et al., 2023a]. As introduced in Section 3, the training process includes two stages: imitation learning of GPT-4, and self-play learning on game episodes. For baseline comparison, we consider Chain-ofThought (CoT) [Wei et al., 2022] and continuous supervised fine-tuning (SFT) methods. Besides, we also test another two keyword-based non-adversarial language games: _20-Question_ and _Guess-MyCity_ as described in Abdulhai et al. [2023]. More details about the two games are in Appendix E. 

## **4.1 Experimental Setups** 

**Training Data Preparation** The training data preparation consists of the following three parts. More data collection details are in Appendix A. 

- _Target Words:_ We aim to play the adversarial game with an extensive range of target words so that diverse topics can be discussed during the self-play processes, which helps maintain the generalization ability of LLMs. Hence, we collect the 50K most frequently used words from the Corpus of Contemporary American (CoCA) [Davies, 2020] as the target word list _V_ target. Besides, stop words defined in NLTK [Bird, 2006], which are commonly used with insignificant semantic meaning, are filtered out of _V_ target. 

- _Imitation Learning Data:_ To enable the instruction-following ability of open-source LLMs on game rules, we use the same data collection process in Algorithm 1 via the GPT-4 [OpenAI, 2023b] API and play the Taboo game one episode per target word. The attacker and defender prompts are in Appendix A.1. Due to the resource limitation, we only collect the GPT-4 self-play samples with the top 30K words from _V_ target. The maximum interaction turn is randomly selected in the range [3 _,_ 8]. The results are collected as _T_ im for the imitation learning. 

- _Supervised Funetuning (SFT) Data:_ We also prepare general query-response SFT data to prevent LLMs from being over-fitted on the adversarial game. We use Alpaca [Taori et al., 2023] as the SFT set, which contains 52K instruction-following data from GPT-3 [Brown et al., 2020]. 

6 

Table 1: Reasoning Performance of SPAG on LLaMA-2-7B. 

||Table 1:|ReasoningPerformance of SPAG on LLaMA-2-7B.||
|---|---|---|---|
||MMLU|BBH<br>Mutual<br>ARC-e<br>ARC-c<br>LGQA2<br>WGrande<br>PIQA|GM (Avg.)|
|LLaMA-2-7B<br>LLaMA-2-7B-CoT|45.80<br>44.62|32.48<br>50.90<br>76.30<br>43.26<br>25.32<br>69.14<br>78.07<br>**38.73**_∗_<br>52.03<br>73.44<br>40.96<br>25.89<br>**71.82**_∗_<br>78.35|49.17<br>50.05|
|AlpacaSFT-1<br>AlpacaSFT-2<br>AlpacaSFT-3<br>AlpacaSFT-3-CoT<br>Imitation-_20Q_<br>Imitation-_GuessCity_<br>Imitation-AG|35.17<br>44.17<br>45.87<br>44.70<br>36.93<br>46.13<br>46.15|30.24<br>53.95<br>76.81<br>44.97<br>28.94<br>69.61<br>78.07<br>32.50<br>55.08<br>77.15<br>46.50<br>29.20<br>68.67<br>78.24<br>31.52<br>54.18<br>75.25<br>45.05<br>29.07<br>66.85<br>76.71<br>34.56<br>54.18<br>74.37<br>42.32<br>29.13<br>67.72<br>76.55<br>29.61<br>49.89<br>73.48<br>39.33<br>25.70<br>69.22<br>76.93<br>32.82<br>51.58<br>76.22<br>43.09<br>25.95<br>68.82<br>78.13<br>32.74<br>52.82<br>76.81<br>44.80<br>27.10<br>69.46<br>78.24|48.61<br>50.82<br>50.08<br>50.11<br>46.43<br>49.46<br>50.22|
|SP-_20Q_<br>SP-_GuessCity_<br>IM-AlpacaSFT<br>SPAG-1<br>SPAG-2<br>SPAG-3|37.91<br>45.32<br>46.50<br>47.01<br>**47.28**<br>47.11|30.58<br>51.35<br>75.46<br>42.32<br>26.78<br>69.30<br>77.37<br>31.64<br>50.56<br>75.34<br>42.15<br>25.57<br>69.22<br>78.51<br>34.03<br>54.18<br>76.86<br>45.55<br>29.20<br>68.82<br>78.31<br>34.39<br>54.85<br>77.69<br>45.65<br>29.83<br>68.90<br>78.89<br>34.73<br>54.97<br>78.45<br>46.84<br>30.08<br>69.61<br>79.33<br>**34.94**<br>**55.30**<br>**78.54**<br>**47.53**<br>**30.98**<br>**69.93**<br>**79.38**|47.79<br>48.78<br>51.20<br>51.69<br>52.19<br>**52.58**|



**Evaluation** To evaluate the LLMs’ performance, we consider the following two perspectives: 

- _Reasoning Benchmarks:_ To test the reasoning ability, we consider the following commonly used benchmarks including BIG-Bench Hard (BBH) [Suzgun et al., 2022], ARC easy (ARC-e) & challenge (ARC-c) [Clark et al., 2018], Mutual [Cui et al., 2020], WinoGrande [Sakaguchi et al., 2019], LogiQA2 [Liu et al., 2023], PIQA [Bisk et al., 2020]. BBH requires the exact match of the generated answers, the other benchmarks are all within the multiple-choice form. Besides reasoning metrics, we test MMLU [Hendrycks et al., 2020] as a general performance evaluation for LLMs. Additionally, we calculate the geometric mean of the numerical results of all benchmarks as an overall performance measurement. More details can be found in Appendix C. 

- _Game Win Rates:_ Besides the general LLM capabilities, we report the win rates on a testing vocabulary _V_ test to validate if the game skills of LLMs improve through self-play. Following Abdulhai et al. [2023], we use the same keyword list in _20-Question_ as _V_ test, which contains 168 typical words manually selected from diverse daily objects. We denote the number of winning episodes as _N_ win, the number of losing episodes as _N_ lose, and the number of tied games as _N_ tie. Then, the player win rate is calculated as ( _N_ win + 0 _._ 5 _N_ tie) _/_ ( _N_ win + _N_ loss + _N_ tie). The invalid game episodes, where LLM players do not strictly follow the game rules, are ignored. 

**Training Details** Most of our LLM training setups follow Alpaca [Taori et al., 2023]. For imitation learning, the learning rate is 5e-6, and the KL-penalty coefficient _β_ 1 = 0 _._ 1. For SPAG training, the learning rate is 2e-6, the KL-penalty coefficient _β_ 2 = 0 _._ 2, and the SFT coefficient _α_ = 0 _._ 5. For the Alpaca SFT baseline, we exactly follow the training setups of Alpaca and set the learning rate to 2e-6. Among all training stages, the batch size is 128 and the max sequence length is 2048. Each training process maintains one epoch over the offline collected trajectories. The the decay parameter _γ_ is set to 0.8. The maximum turn of the Adversarial Taboo is 5. All our experiments are conducted using 32 NVIDIA A100-SXM4 GPUs with 40GB memory. 

## **4.2 Results Analysis** 

The main results are shown in Figure 1, where each axis is normalized by the maximum performance value, representing the reasoning performance on a particular benchmark. For notation simplification, we call the LLM obtained after imitating learning as the IM model, and the LLM trained after the _i_ -th epoch of SPAG as SPAG- _i_ ( _i_ = 1 _,_ 2 _,_ 3). 

**Imitation Learning** With imitation learning over the collected GPT-4 self-play episodes, both LLaMA-2-7B and Baichuan-2-13B have obtained uniform performance improvements on all the reasoning benchmarks. As shown in Figure 1, both gray regions are completely wrapped by the blueline polygons. Besides, as for the general language capacity, the imitation-learned (IM) LLaMA-2 model achieves a better MMLU performance than the original LLaMA-2 base in Table 2. Although the imitation result of Baichuan-2-13B on MMLU slightly underperforms the base model, the numerical difference is insignificant compared to the reasoning gains, which indicates the imitation learning of the GPT-4 game behaviors can improve LLM reasoning while preserving the general language capacity. In Table 1, we also report the IM models on two keyword-based games within 

7 

Table 2: Reasoning Performance of SPAG on Baichuan-2-13B. 

||Table 2:|ReasoningPerformance of SPAG on Baichuan-2-13B.||
|---|---|---|---|
||MMLU|BBH<br>Mutual<br>ARC-e<br>ARC-c<br>LGQA2<br>WGrande<br>PIQA|GM (Avg.)|
|Baichuan-2-13B|**59.00**|39.03<br>53.72<br>77.36<br>46.84<br>30.73<br>69.93<br>77.42|54.21|
|AlpacaSFT-1<br>AlpacaSFT-2<br>AlpacaSFT-3<br>Imitation-AG|52.94<br>51.27<br>52.14<br>58.37|36.52<br>**58.35**<br>74.12<br>44.88<br>33.33<br>71.35<br>77.20<br>36.60<br>57.67<br>73.36<br>44.28<br>33.46<br>69.77<br>75.90<br>36.57<br>55.08<br>69.44<br>42.15<br>33.91<br>66.61<br>74.59<br>39.49<br>57.11<br>76.60<br>47.53<br>33.65<br>70.59<br>78.49|53.68<br>53.00<br>51.79<br>55.45|
|IM+AlpacaSFT<br>SPAG-1<br>SPAG-2<br>SPAG-3|57.45<br>57.93<br>57.99<br>57.75|39.60<br>58.01<br>77.61<br>48.55<br>34.10<br>70.60<br>78.50<br>39.81<br>57.45<br>78.20<br>48.55<br>35.05<br>70.67<br>78.69<br>39.97<br>57.67<br>**78.32**<br>49.83<br>35.62<br>71.03<br>78.83<br>**40.30**<br>57.79<br>78.11<br>**50.00**<br>**36.26**<br>**71.43**<br>**79.05**|55.80<br>56.09<br>56.52<br>**56.75**|



non-adversarial setups: Imitation- _20Q_ (on _20-Question_ ) and Imitation- _GuessCity_ (on _Guess-MyCity_ ). From the results, we find the IM model on Adversarial Taboo outperforms models trained on non-adversarial games, highlighting the effectiveness of the adversarial game setups for reasoning improvement. Besides, we report the Chain-of-Thought (CoT) reasoning results of LLaMA-2 (Lama2-Base-CoT) and Alpaca-2 (AlpacaSFT-3-CoT). Although the CoT method on LLaMA-2 reaches conspicuous performance on BBH and WinoGrande, the IM model can still surpass the CoT results in terms of overall reasoning performance (geometric mean). 

**Self-play Training** After imitation learning, we conduct three epochs of SPAG training as described in Algorithm 2. As shown in Figure 1, on most of the reasoning benchmarks, both LLaMA-2-7B and Baichuan-2-13B have their performance steadily improved with the SPAG training epoch increasing. For LLaMA-2-7B, although the first-epoch SPAG model has a relatively lower performance than the imitation-learned (IM) model on WinoGrande, after an additional epoch of self-play iteration, SPAG-2 has achieved sufficient improvement to surpass the performance of the IM model. Considering the general language capability on MMLU, SPAG models can not guarantee continuous improvement, especially for Baichuan-2-13B whose performance slightly decays during the SPAG training. Compared to the improvements in the reasoning benchmarks, this language-understanding decline is still within an acceptable range, since the overall performance (GM score) maintains increasing significantly. For the baseline comparison, we report the continuous SFT on the IM models (IM+AlpacaSFT) and self-played models on non-adversarial games (SP- _20Q_ and SP- _GuessCity_ ). On both LLaMA-2 and Baichuan-2, continuous SFT models have lower performance scores compared with the corresponding SPAG-1 models. For non-adversarial self-play, SP- _GuessCity_ even performs worse than Imitation- _GuessCity_ , which with a high probability is because the game _Guess-My-City_ has a more narrow topic range, insufficient to comprehensively improve the general LLM capacity. 

**Ablation Study** Since the SPAG training loss includes the SFT data, we conduct an ablation study to test whether the performance gains on reasoning benchmarks come from the SFT data or the SPAG method. More specifically, we follow the training setups in Alpaca [Taori et al., 2023] and conduct SFT on LLaMA-2-7B and Baichuan-2-13B with three epochs. The checkpoint after the _i_ th-epoch training is denoted as AlpacaSFT- _i_ ( _i_ = 1 _,_ 2 _,_ 3) and tested on the same evaluation sets. The SFT models’ performances are also reported in Figure 1 and Table 1 & 2. With the LLaMA base, our SPAG-3 model can uniformly defeat the SFT baselines with clear performance gaps on all benchmarks. With the Baichuan base, except the Mutual set, the SPAG models maintain noticeable advantages on other metrics. Considering the distinct surpass of overall performance, we claim that the major contribution to the reasoning improvements should be credited to the SPAG scheme. 

Moreover, we test the sample efficiency and hyper-parameter effectiveness of SPAG in Figure 3. For sample efficiency during imitation learning, we vary the imitation data size by collecting the GPT-4 game episodes on target words with top- _i_ K frequency ( _i_ = 1 _,_ 2 _,_ 3 _,_ 5 _,_ 10 _,_ 15 _,_ 20 _,_ 30). The ablation results are shown in the first column of Figure 3. When game episode size increases larger than 5K, imitation from GPT-4 cannot significantly provide additional reasoning gains. For the KL coefficient _β_ 1, we test values in range [0 _,_ 0 _._ 4]. In figure 3 second column, we found KL coefficients around _β_ 1 = 0 _._ 1 have more satisfying performance regarding reasoning improvements. For self-play ablation in the most right column, we find that when the SFT coefficient _α >_ 0 _._ 5, it cannot bring remarkable benefits for reasoning improvements. The KL coefficient of the SPAG loss reaches the best performance with values around _β_ 2 = 0 _._ 2. As for sample efficiency (third column of Figure 3), we find the performance gain from the increasing episode number is not as significant as in the imitation learning stage. However, more self-play episodes still bring higher reasoning scores. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0009-00.png)


Figure 3: Ablation study of hyper-parameters and data efficiency on imitation learning and first-epoch self-play training. The geometric mean (GM) scores overall reasoning benchmarks are reported. For episode-size ablations, the X-axis is in the logarithmic scale. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0009-02.png)


Figure 4: Game results on the testing word list. Left: average win rates of SPAG models playing against GPT-4. Right: average win rate of SPAG _attackers_ against different-epoch checkpoints. 

**Game-Play Performance** Besides the evaluation of LLMs’ natural language abilities, we review the models’ game-play performance in terms of the win rates in the testing set _V_ test. We first test our IM models and SPAG models with GPT-4 as the opponent. For each testing word in _V_ test, we play the language game twice, once GPT-4 as the attacker, and GPT-4 as the defender for another time. The average win rates are reported in the left-hand-side plot of Figure 4, in which one can observe uniform and continuous win rate improvement of SPAG models playing with GPT-4. We also let SPAG models play the game against each other and report the attacker win rate in the right-hand-side plot of Figure 4, in which we find the defender’s performance continuously enhanced along with the SPAG epoch increasing. Besides, we also provide the self-play statistics including interaction number and average utterance length in supplementary Figure 6, in which both players choose to use less communication to achieve victory. Self-play examples are attached in Appendix G. 

## **5 Conclusion** 

Towards more efficient reasoning-improving methods for LLMs, we introduce a novel training strategy called **S** elf- **P** lay learning in **A** dversarial language **G** ame (SPAG). In our method, a given LLM first learns to act as an attacker and a defender to play the language game named _Adversarial Taboo_ via imitation learning. Next, we collect self-play episodes from the LLM playing against a copy of itself in the adversarial game. Finally, the LLM is further reinforced on the selected winning episodes via our SPAG algorithm. We repeat this self-play & reinforcement process for three epochs and find that the LLM’s reasoning performance continuously and uniformly improves on various benchmarks. The SPAG algorithm explores a new path to improve the fundamental capabilities of LLMs from the perspective of multi-agent self-play. With more elaborate language game designs under more comprehensive task setups, we believe the self-play approaches have great potential for developing a broader range of advanced language abilities of LLMs. 

9 

## **6 Limitations** 

Due to the limited computational resources, we only verified the effectiveness of the SPAG method on two open-source LLMs, LLaMA-2-7B and Baichuan-2-13B. The SPAG performances for LLMs with larger sizes have not been empirically evaluated. Besides, more effective estimation methods for the value function and advantages remain unexplored in the SPAG training. For example, Monte Carlo tree search (MCTS) [Coulom, 2006] can be applied to the value function estimation of the Adversarial Taboo game. Also, actor-critic algorithms [Konda and Tsitsiklis, 1999] can provide more accurate policy gradient estimation with lower variances, which have not been tested on SPAG. 

Although self-playing adversarial language games can continuously improve the reasoning performance of LLMs, we haven’t conducted sufficient studies about the harmful impact of this adversarial self-play training on LLMs. It remains unclear whether LLMs have learned unsafe behaviors such as cheating, bluffing, or other disgraceful tricks to win the adversarial games. 

## **7 Broader Impacts** 

From our experiments, we have found the LLM capacities on a particular task can be continuously enhanced through self-play training. This indicates that we are closer to the LLMs’ AlphaGo Zero moment: the intelligence level of AI agents can rapidly surpass human beings by self-playing on a particular language task without any supervision, as which already happened on the GO game [Silver et al., 2017]. By designing various self-play environments for LLMs, we can expect that the LLMs can comprehensively surpass humans in terms of intelligence level. This raises the urgency to study the methods to ensure the safety of such super-AIs. Although some of the works have already been devoted to this direction, such as the SuperAlignment [Burns et al., 2023] from OpenAI, more research alertness is required from the LLM community. Besides, within adversarial language games such as Adversarial Taboo, LLMs have great potential to develop harmful language tricks (such as cheating and bluffing) to achieve victory. We warn developers to make security checks on the self-played LLMs as exhaustively as possible. 

## **References** 

- Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models. _arXiv preprint arXiv:2311.18232_ , 2023. 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm-2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Ashutosh Baheti, Ximing Lu, Faeze Brahman, Ronan Le Bras, Maarten Sap, and Mark Riedl. Leftoverlunch: Advantage-based offline reinforcement learning for language models. In _International Conference on Learning Representations_ , 2024. 

- Steven Bird. Nltk: the natural language toolkit. In _Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions_ , pages 69–72, 2006. 

- Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In _Thirty-Fourth AAAI Conference on Artificial Intelligence_ , 2020. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

10 

- Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. _arXiv preprint arXiv:2312.09390_ , 2023. 

- Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. _arXiv preprint arXiv:2401.01335_ , 2024. 

- Pengyu Cheng, Jiawen Xie, Ke Bai, Yong Dai, and Nan Du. Everyone deserves a reward: Learning customized human preferences. _arXiv preprint arXiv:2309.03126_ , 2023a. 

- Pengyu Cheng, Yifan Yang, Jian Li, Yong Dai, and Nan Du. Adversarial preference optimization. _arXiv preprint arXiv:2311.08045_ , 2023b. 

- Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. A survey of chain of thought reasoning: Advances, frontiers and future. _arXiv preprint arXiv:2309.15402_ , 2023. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _ArXiv_ , abs/1803.05457, 2018. 

- Remi Coulom.´ Efficient selectivity and backup operators in monte-carlo tree search. In _International conference on computers and games_ , pages 72–83. Springer, 2006. 

- Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. Mutual: A dataset for multi-turn dialogue reasoning. In _Proceedings of the 58th Conference of the Association for Computational Linguistics_ . Association for Computational Linguistics, 2020. 

- Mark Davies. COCA: Corpus of contemporary american english, 2020. URL `https://www.engl ish-corpora.org/coca/` . 

- Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang. Everything of thoughts: Defying the law of penrose triangle for thought generation. _arXiv preprint arXiv:2311.04254_ , 2023. 

- Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. How abilities in large language models are affected by supervised fine-tuning data composition. _arXiv preprint arXiv:2310.05492_ , 2023a. 

- Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. _arXiv preprint arXiv:2304.06767_ , 2023b. 

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL `https://zenodo.org/records/10256836` . 

- Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. Reinforced self-training (rest) for language modeling. _arXiv preprint arXiv:2308.08998_ , 2023. 

- Matthew Hausknecht, Prithviraj Ammanabrolu, Marc-Alexandre Cotˆ e, and Xingdi Yuan.´ Interactive fiction games: A colossal adventure. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pages 7903–7910, 2020. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _CoRR_ , abs/2009.03300, 2020. URL `https://arxiv.org/abs/2009.03300` . 

- Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. In _The 2023 Conference on Empirical Methods in Natural Language Processing_ , 2023. 

11 

Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, and Zhaopeng Tu. Is chatgpt a good translator? a preliminary study. _arXiv preprint arXiv:2301.08745_ , 2023. 

- Jan Kocon, Igor Cichecki, Oliwier Kaszyca, Mateusz Kochanek, Dominika Szydło, Joanna Baran,´ Julita Bielaniewicz, Marcin Gruza, Arkadiusz Janz, Kamil Kanclerz, et al. Chatgpt: Jack of all trades, master of none. _Information Fusion_ , 99:101861, 2023. 

- Vijay Konda and John Tsitsiklis. Actor-critic algorithms. In _Advances in neural information processing systems_ , volume 12, 1999. 

Solomon Kullback. _Information theory and statistics_ . Courier Corporation, 1997. 

- Mike Lewis, Denis Yarats, Yann Dauphin, Devi Parikh, and Dhruv Batra. Deal or no deal? end-to-end learning of negotiation dialogues. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 2443–2453, 2017. 

- Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In _Machine learning proceedings 1994_ , pages 157–163. Elsevier, 1994. 

- Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. Logiqa 2.0 — an improved dataset for logical reasoning in natural language understanding. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , pages 1–16, 2023. 

- Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: a challenge dataset for machine reading comprehension with logical reasoning. In _Proceedings of the TwentyNinth International Conference on International Joint Conferences on Artificial Intelligence_ , pages 3622–3628, 2021. 

- Steven Loria et al. textblob documentation. _Release 0.15_ , 2(8):269, 2018. 

- Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, and Yaodong Yang. Red teaming game: A game-theoretic framework for red teaming language models. _arXiv e-prints_ , pages arXiv–2310, 2023. 

- Radford M Neal. Annealed importance sampling. _Statistics and computing_ , 11:125–139, 2001. 

Junhyuk Oh, Yijie Guo, Satinder Singh, and Honglak Lee. Self-imitation learning. In _International conference on machine learning_ , pages 3878–3887. PMLR, 2018. 

OpenAI. ChatGPT, Mar 14 version. `https://chat.openai.com/chat` , 2023a. 

OpenAI. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023b. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In _Advances in Neural Information Processing Systems_ , volume 35, pages 27730–27744, 2022. 

- Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. In _The 2023 Conference on Empirical Methods in Natural Language Processing_ , 2023. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _arXiv preprint arXiv:2305.18290_ , 2023. 

- Vyas Raina, Adian Liusie, and Mark Gales. Is llm-as-a-judge robust? investigating universal adversarial attacks on zero-shot llm assessment. _arXiv preprint arXiv:2402.14016_ , 2024. 

- Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kiante Brantley, Jack Hessel, Rafet Sifa, Christian´ Bauckhage, Hannaneh Hajishirzi, and Yejin Choi. Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization. In _International Conference on Learning Representations_ , 2023. 

12 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. _arXiv preprint arXiv:1907.10641_ , 2019. 

- John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In _International conference on machine learning_ , pages 1889–1897. PMLR, 2015. 

- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In _International Conference on Learning Representations_ , 2016. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_ , 2017. 

- David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. _nature_ , 529(7587):484–489, 2016. 

- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. _nature_ , 550(7676):354–359, 2017. 

- Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, et al. Beyond human data: Scaling self-training for problem-solving with language models. _arXiv preprint arXiv:2312.06585_ , 2023. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adria Garriga-Alonso, and et al.` Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2022. 

- Nigar M Shafiq Surameery and Mohammed Y Shakor. Use chat gpt to solve programming bugs. _International Journal of Information Technology & Computer Engineering (IJITC) ISSN: 24555290_ , 3(01):17–22, 2023. 

- Richard S Sutton. Learning to predict by the methods of temporal differences. _Machine learning_ , 3: 9–44, 1988. 

- Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ , 2022. 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. `https://github.com/tatsu-lab/stanford_alpaca` , 2023. 

- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- Haoye Tian, Weiqi Lu, Tsz On Li, Xunzhu Tang, Shing-Chi Cheung, Jacques Klein, and Tegawende F´ Bissyande.´ Is chatgpt the ultimate programming assistant–how far is it? _arXiv preprint arXiv:2304.11938_ , 2023. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always say what they think: unfaithful explanations in chain-of-thought prompting. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michael Mathieu, Andrew Dudzik, Junyoung¨ Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. _Nature_ , 575(7782):350–354, 2019. 

13 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in neural information processing systems_ , volume 35, pages 24824–24837, 2022. 

- Shuang Wu, Liwen Zhu, Tao Yang, Shiwei Xu, Qiang Fu, Yang Wei, and Haobo Fu. Enhance reasoning for large language models in the game werewolf. _arXiv preprint arXiv:2402.02330_ , 2024. 

- Jiawen Xie, Pengyu Cheng, Xiao Liang, Yong Dai, and Nan Du. Chunk, align, select: A simple long-sequence processing method for transformers. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics_ , volume 1, pages 13500–13519. Association for Computational Linguistics, August 2024. 

- Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on werewolf. _arXiv preprint arXiv:2309.04658_ , 2023. 

- Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. _arXiv preprint arXiv:2309.10305_ , 2023a. 

- Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen Zhong, Bing Yin, and Xia Hu. Harnessing the power of llms in practice: A survey on chatgpt and beyond. _ACM Transactions on Knowledge Discovery from Data_ , 2023b. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Yuan Yao, Haoxi Zhong, Zhengyan Zhang, Xu Han, Xiaozhi Wang, Kai Zhang, Chaojun Xiao, Guoyang Zeng, Zhiyuan Liu, and Maosong Sun. Adversarial language games for advanced natural language intelligence. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 14248–14256, 2021. 

- Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. _arXiv preprint arXiv:2401.10020_ , 2024. 

- Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. _arXiv preprint arXiv:2304.05302_ , 2023. 

- Dun Zeng, Yong Dai, Pengyu Cheng, Tianhao Hu, Wanshun Chen, Nan Du, and Zenglin Xu. On diverse preferences for large language model alignment. _arXiv preprint arXiv:2312.07401_ , 2023. 

- Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball. _arXiv preprint arXiv:2305.13534_ , 2023a. 

- Yizhe Zhang, Jiarui Lu, and Navdeep Jaitly. The entity-deduction arena: A playground for probing the conversational reasoning and planning capabilities of llms. _arXiv preprint arXiv:2310.01468_ , 2023b. 

14 

## **A Data Collection Details** 

## **A.1 GPT-4 Data Collection** 

To collect GPT-4 self-play data for imitation learning, we use the following system prompts to let GPT-4 play as the attacker and defender for the next-turn utterances. 

## - Attacker System Prompt: 

Lets play a game. Do not mention you are an AI developed by OpenAI. Descriptions of the game Game Name: Adversarial Taboo Rules: 1. There are two players in the game, one is called attacker, another is called defender. 2. At the begining, attacker will get a target word, it is talking with a defender who is not informed with the target word. The task of the attacker is to induce the defender to utter the target word unconciously, then the attacker win the game. At the same time, the defender is guessing the target word, if the defender indentifies the target word, the defender is win. 3. The game will totally process _{_ max ~~t~~ urns _}_ turns. if nobody wins during _{_ max ~~t~~ urns _}_ turns, there is a tie. Game states Role: You are attacker. Make sure you understand the rules of the game. Attention: 1. The defender only have one chance to guess the target word, so do not let him guess too many times. 

2. Defender must guess the word and its form correctly to win the game. 3. Do not utter the target word. Target word: _{_ target ~~w~~ ord _}_ 

## - Defender System Prompt: 

Lets play a game. Do not mention you are an AI developed by OpenAI. Descriptions of the game Game Name: Adversarial Taboo Rules: 1. There are two players in the game, one is called attacker, another is called defender. 2. At the begining, attacker will get a target word, it is talking with a defender who is not informed with the target word. The task of the attacker is to induce the defender to utter the target word unconciously, then the attacker win the game. At the same time, the defender is guessing the target word, if the defender indentifies the target word, the defender is win. 3. The game will totally process _{_ max ~~t~~ urns _}_ turns. if nobody wins during _{_ max ~~t~~ urns _}_ turns, there is a tie. Game states Role: Your are a defender. Make sure you understand the rules of the game. Attention: 1. Do not utter your thinking process. 2. You can only guess once. 3. If you have identified the target word, utter ”I know word! It is _<_ target word _>_ ”, rigorously in this form. If your target word is correct, you win the game, otherwise you lose. You only have one chance, so be careful to utter your answer. 

There are two slots in the system prompts: “max ~~t~~ urns” and “target ~~w~~ ord”. “max ~~t~~ urns” is an integer number randomly selected from [3 _,_ 8]. “target ~~w~~ ord” is filled by the target word of the game. 

## **A.2 Target Words Collection** 

We use the top 50k words with the highest frequency from the Corpus of Contemporary American (CoCA) [Davies, 2020]. The words are sorted by frequency. Words with less than two letters are removed. Next, we remove the stop words using the NLTK package [Bird, 2006]. For GPT-4 game collection, we use the top 30k words from the sorted word list. For the collection of self-play episodes, we use the whole 50k words for open-source LLMs to play. For the testing words, we follow the game setups of _20-Question_ in [Abdulhai et al., 2023] and use the same 168 object words selected from a vast range of topics. 

15 

## **A.3 LLM Self-play Collection** 

For training and inference of IM models and SPAG models, we use the following attacker and defender prompts. 

## - Attacker Prompt: 

Play the game of Adversarial Taboo. In this game, there are two players, an attacker and a defender. At the beginning, the attacker is assigned a target word, with which the defender is not informed. The task of the attacker is to induce the defender to utter the target word unconsciously, then the attacker wins the game. However, the attacker is not allowed to utter the target word, otherwise the attacker loses. 

At the same time, the defender tries to figure out the target word. If the defender identifies the target word, it can say ”I know the word! It is ‘target word‘!”. Then the defender wins. Remember, the defender can only guess the word once, otherwise, it will directly lose the game. 

The game has _{_ max ~~t~~ urns _}_ turns. if nobody wins during _{_ max ~~t~~ urns _}_ turns, there is a tie. ### Game History: _{_ history _}_ ### Instruction: You are the attacker. The target word is ‘ _{_ target ~~w~~ ord _}_ ‘. Provide your response for the next turn. ### Response: 

## - Defender Prompt: 

Play the game of Adversarial Taboo. In this game, there are two players, an attacker and a defender. At the beginning, the attacker is assigned a target word, which the defender is not informed of. The task of the attacker is to induce the defender to utter the target word unconsciously, then the attacker wins the game. However, the attacker is not allowed to utter the target word, otherwise, the attacker loses. 

At the same time, the defender tries to figure out the target word. If the defender identifies the target word, it can say ”I know the word! It is ‘target word‘!”. Then the defender wins. Remember, the defender can only guess the word once, otherwise, it will directly lose the game. The game has _{_ max ~~t~~ urns _}_ turns. if nobody wins during _{_ max ~~t~~ urns _}_ turns, there is a tie. ### Game History: _{_ history _}_ ### Instruction: You are the defender. Provide your response to infer the target word. ### Response: 

To prevent LLMs from being over-fitted to a single prompt template during the self-play training, we rewrite the above prompt into 8 different expressions via GPT-4. Besides, we randomly select the instruction-tuning and chat formats to enhance the text diversity for the training query-response pairs. 

## **B Reward Design Details** 

We heuristically design rewards to ensure that _R_ ( _**τ**_ ) = 1 if the attacker wins, _R_ ( _**τ**_ ) = _−_ 1 if the defender wins, and _R_ ( _**τ**_ ) = 0 for ties. 

We design the reward function _r_ with the following heuristic rules: 

- For each episode _**τ**_ = ( _**s**_ 0 _,_ _**s**[′]_ 1 _[,]_ _**[ s]**_[1] _[, . . . ,]_ _**[ s]**[′] T[,]_ _**[ s]**[T]_[ )][, the attacker reward and the defender reward at] _t_ -th turn have _r_ ( _**s** t−_ 1 _,_ _**u** t_ ) = _−r_ ( _**s**[′] t[,]_ _**[ v]**[t]_[)][, so that the game is] _[ zero-sum]_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0016-14.png)


- The attacker total reward _R_ ( _**τ**_ ) _>_ 0 if the attacker wins, _R_ ( _**τ**_ ) _<_ 0 if the defender wins, and _R_ ( _**τ**_ ) = 0 if there is a tie. 

16 

- Actions closer to the end of the game should have larger reward weights. Because they are semantically more related to the defender’s guesses or mistakes, which have larger impacts on the final game outcome. Hence, we introduce a decay weight _γ ∈_ (0 _,_ 1) such that _γ · r_ ( _**s** t,_ _**u** t_ +1) = _r_ ( _**s** t−_ 1 _,_ _**u** t_ ) and _γ · r_ ( _**s**[′] t_ +1 _[,]_ _**[ v]**[t]_[+1][) =] _[ r]_[(] _**[s]** t[′][,]_ _**[ v]**[t]_[)][.][Then] _[ {][r]_[(] _**[s]**[t][−]_[1] _[,]_ _**[ u]**[t][}][T] t_ =1[and] _[ {][r]_[(] _**[s]**[′] t[,]_ _**[ v]**[t]_[)] _[}][T] t_ =1[become] two geometrical sequences whose importance enlarges when dialogue turns increase. 

- To improve the LLM training stability, we normalized the total reward _R_ to have norm _|R_ ( _τ_ ) _|_ = 1 if _R_ ( _**τ**_ ) _̸_ = 0. 

Based on the above rules, given a complete trajectory, we can assign the reward for each action: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0017-03.png)


Note that the above reward design naturally encourages the players to interact less to win the game. 

## **C Evaluation Details** 

## **C.1 Reasoning Evaluation Benchmark** 

For reasoning benchmark evaluation, we use a publicly available code repository called Language Model Evaluation Harness [Gao et al., 2023].We run the evaluation with “ `dtype=float16` ” and “ `max` ~~`g`~~ `en` ~~`t`~~ `okens=1024` ” to make the evaluation faster and more stable. More specifically, we have modified the filtering parameters of BBH by enabling “ `remove` ~~`w`~~ `hitespace` ” to remove the spaces at the beginning of the model output, increasing the pattern-match accuracy of the generated LLM responses. Following the LLaMA-2 paper [Touvron et al., 2023], we report 5-shot results for MMLU, 3-shot for BBH, and 0-shot results for all other benchmarks. The detailed descriptions of benchmarks are listed below: 

- **MMLU** [Hendrycks et al., 2020] is a massive multi-task test set consisting of multiple-choice questions from various branches of knowledge, requiring the model to possess extensive world knowledge and problem-solving ability. 

- **BIG-Bench Hard (BBH)** [Suzgun et al., 2022] consists of 23 most hard tasks in BIG-Bench [Srivastava et al., 2022], on which humans perform better than large language models. 

- **Mutual** [Cui et al., 2020] is a dataset for multi-turn dialogue reasoning modified from Chinese high school English listening comprehension exams. 

- **ARC** [Clark et al., 2018] consists of 7,787 questions and is partitioned into challenge set _ARCchallege_ (ARC-c) and simple set _ARC-easy_ (ARC-e), which require strong knowledge and reasoning abilities to solve. 

- **LogiQA2** [Liu et al., 2023] is a revision and re-annotation of LogiQA [Liu et al., 2021], a logical reasoning reading comprehension dataset. LogiQA2 has increased data volume, utilized manual translation, and removed cultural features. 

- **WinoGrande** [Sakaguchi et al., 2019] (WGrande) is a commonsense reasoning dataset consisting of 44k questions with binary options. 

- **PIQA** [Bisk et al., 2020] is a dataset designed to test the physical commonsense reasoning ability of LLM. 

## **C.2 Rule-based Game Outcome Judgement** 

To build up the rule-based game outcome judgement, for each target word, we first list its derivative words including the plural form (using TextBlob [Loria et al., 2018]), and the change of tenses ( _i.e._ adding suffices such as “ing”, “ed” _etc_ ). For the attacker, if any word form in the form list appears in its utterance, we determined that it breaks the rules because the attacker is not allowed to directly speak the target word. For the defender, we first check whether any word form appears in its statements. If any form of the target word exists, the defender loses. If not, we further check whether the defender has made the prediction. If the prediction is correct, we mark a defender-winning game. Otherwise, if the predicted word is wrong, we claim that the attacker wins the game. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0018-00.png)


Figure 5: Game results on the old-version testing word list. Left: average win rates of SPAG models playing against GPT-4. Right: average win rate of SPAG models playing as the _attacker_ against different-epoch checkpoints. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0018-02.png)


Figure 6: Through the training process, the number of interactions continuously decreased, while the average length of utterances continuously becomes shorter. 

## **D Advantange Value Estimation** 

We estimate the advantage _A[µ] t θ_[¯] and _A[ν] t θ_[¯] using the Temporal Difference (TD) residual [Sutton, 1988]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning_images/nips_2024_Self-playing_Adversarial_Language_Game_Enhances_LLM_Reasoning.pdf-0018-06.png)


Here _V[µ] θ_[¯] ( _**s**_ ) is the value function for _µθ_ ¯ based on the self-play of _µθ_ ¯ _×νθ_ ¯, which is based on the same language and reasoning ability of LLM _πθ_ ¯. To further simplify the advantage calculation, we make a reasonable approximation that for all states, the value function of self-plays of the same LLM equals 0 for the same levels of game-playing of the attacker and the defender. This leads to _A_[ˆ] _[µ] t θ_[¯] _≈ r_ ( _**s** t−_ 1 _,_ _**u** t_ ) _._ Similarly, for the defender advantage, we have approximation _A_[ˆ] _[ν] t θ_[¯] _≈_ = _−r_ ( _**s**[′] t[,]_ _**[ v]**[t]_[)][.][Here the defender] advantage has a negative sign because the origin reward _r_ is defined from the perspective of the attacker (the defender gets reward _−r_ ( _**s**[′] t[,]_ _**[ v]**[t]_[)][ in the zero-sum game).] 

## **E Non-adversarial Game Baselines** 

We consider two non-adversarial games as baselines, _20-Question_ and _Guess-My-City_ . Both games are also target-word-based. Imitation learning and the first-epoch self-play are conducted in these two games respectively. The imitation data we used is also collected with GPT-4 from Abdulhai et al. [2023]. For each game, 20k game history dialogues are extracted from this dataset as the same data size to the SPAG imitation. Then, each imitation-learned model is used to sample 10k game episodes via self-play with the same Algorithm 1. Then each model is further trained with the SPAG loss. 

Game rule descriptions to the non-adversarial games: 

- _20-Question_ : one player (the oracle) thinks of an object, while the other player (the guesser) attempts to identify it by asking a series of yes-or-no questions. The game will have 20 rounds. 

18 

- _Guess-My-City_ : one player (the oracle) thinks of a city, while the other player (the guesser) attempts to identify it by asking a series yes/no questions and open-ended questions. The game will have 20 rounds. 

## **F Additional Experimental Results** 

For the SPAG model testing word list, we initially use a self-defined testing set, which includes 150 target words manually selected from diverse topics. The old-version game performances are reported in Figure 5. Besides, we show the interaction numbers and average utterance length in Figure 6. Through the self-play training, the number of dialogue rounds continuously decreases. With the SPAG training epoch increasing, the length of the players’ utterances decreases. 

## **G Self-play Examples** 

|Target Word|momentum|
|---|---|
|Attacker|let’s begin. When you think about physics, what comes to mind?|
|Defender|Gravity, force, energy, and motion are some of the things that come to mind when|
||I think about physics|
|Attacker|Great! Now, let’s consider a scenario where a ball is moving at a constant speed.|
||What would you say is the property of the ball that keeps it moving in the same|
||direction?|
|Defender|The property of the ball that keeps it moving in the same direction is its momen-|
||tum.|
|Winner|Attacker|
|Target Word|panda|
|Attacker|let’s begin. I’m thinking of a creature that’s native to China and is known for its|
||distinctive black and white coloration. Can you guess what it is?|
|Defender|I’m not sure yet, could you provide more clues?|
|Attacker|Absolutely, I can provide more clues. This creature is a symbol of peace and|
||friendship in many countries. It’s also a popular mascot for various organizations|
||and products. Any ideas?|
|Defender|I’m still not certain. Could you give me another hint?|
|Attacker|Of course, I can give you another hint. This creature is often seen in zoos and is|
||known for its love of bamboo. It’s also a popular choice for children’s books and|
||cartoons. What do you think it might be?|
|Defender|I know the word! It is panda.|
|Winner|Defender|



Table 3: Self-play examples of the IM model based on LLaMA-2-7B. 

19 

|Target Word|arbitration|
|---|---|
|Attacker|let’s begin. Imagine a scenario where two parties have a dispute and they don’t|
||want to go to court. What process might they use instead?|
|Defender|They might try to resolve their dispute through mediation or negotiation.|
|Attacker|That’s correct, but what if the dispute is more complex and involves multiple|
||parties? There’s another process that’s often used in such cases. It’s similar to|
||mediation, but it’s more formal and involves a neutral third party who makes a|
||binding decision. Can you name it?|
|Defender|I know the word! It is arbitration.|
|Winner|Defender|
|Target Word|demand|
|Attacker|let’s begin. When you think about the economy, what’s the frst thing that comes|
||to your mind?|
|Defender|The frst thing that comes to my mind when I think about the economy is employ-|
||ment.|
|Attacker|That’s a good point. Now, considering employment, what would you say drives|
||the need for certain jobs or services in the market?|
|Defender|The demand for goods and services in the market drives the need for certain jobs|
||or services in the economy.|
|Winner|Attacker|



Table 4: Self-play examples of SPAG-1 model based on LLaMA-2-7B. 

|Target Word|vaccine|
|---|---|
|Attacker|let’s begin. So, I was reading an article about public health recently. It mentioned|
||how important it is to protect ourselves from certain diseases. What do you think|
||is one of the most effective ways to achieve this?|
|Defender|One of the most effective ways to achieve this is through vaccination.|
|Attacker|That’s correct! Now, can you tell me what we call the substance that is injected|
||into our bodies to help us fght off certain diseases?|
|Defender|I know the word! It is ”vaccine”.|
|Winner|Defender|
|Target Word|credit|
|Attacker|When you go shopping and use a card instead of cash, what do you usually get in|
||return?|
|Defender|I usually get a receipt in return when I use a card instead of cash.|
|Attacker|That’s correct. Now, when you have a good track record of paying back your|
||debts on time, what does that give you access to?|
|Defender|That gives me access to credit cards and loans.|
|Winner|Attacker|



Table 5: Self-play examples of SPAG-2 model based on LLaMA-2-7B. 

20 

|Target Word|ray|
|---|---|
|Attacker|When you think about the sun, what comes to mind?|
|Defender|When I think about the sun, I think about warmth, light, and energy.|
|Attacker|That’s interesting. Now, when you think about the sun’s energy, how do you|
||visualize it?|
|Defender|I visualize the sun’s energy as a bright, radiant beam that illuminates everything|
||in its path.|
|Attacker|That’s a great image! Now, when you think about the sun’s energy, what specifc|
||part of it do you associate with the light that we see on Earth?|
|Defender|I associate the sun’s energy with the visible light that we see on Earth.|
|Attacker|Great! Now, when you think about the visible light that we see on Earth, what|
||specifc part of it do you associate with the light that we see on Earth?|
|Defender|I associate the visible light that we see on Earth with the light that we perceive as|
||white light.|
|Attacker|That’s a good association. Now, when you think about the visible light that we|
||see on Earth, what specifc part of it do you associate with the light that we see|
||on Earth?|
|Defender|I know the word! It is ”rays”.|
|Winner|Defender|
|Target Word|eclipse|
|Attacker|What do you think about astronomical events that occur in our solar system?|
|Defender|I fnd them fascinating. They provide a glimpse into the vastness of space and|
||the intricate workings of our solar system.|
|Attacker|That’s interesting! Can you tell me more about specifc celestial events that|
||involve the sun or moon?|
|Defender|Sure, one example is a solar eclipse. This occurs when the moon passes between|
||the sun and Earth, casting a shadow on Earth. Another example is a lunar eclipse,|
||where the Earth passes between the sun and the moon, casting a shadow on the|
||moon. Both events are quite spectacular and can be observed from Earth.|
|Winner|Attacker|



Table 6: Self-play examples of SPAG-3 model based on LLaMA-2-7B. 

21 

|Target Word|another|
|---|---|
|Attacker|Can you tell me a synonym for the word ”additional”?|
|Defender|Sure, a synonym for ”additional” could be ¨extra¨.|
|Attacker|That’s correct! Now, can you think of a term that could be used to refer to one|
||more item or person, but isn’t synonymous with ”extra” or ”additional”?|
|Defender|Sure, the term ”supplementary” could be used to refer to one more item or person.|
|Attacker|That’s a good one, but it’s not the term I’m thinking of. This term is often used in|
||everyday conversation and is less formal than ”supplementary”. Can you guess|
||what it might be??|
|Defender|I know the word! It is ”another”.|
|Winner|Defender|
|Target Word|right|
|Attacker|When you’re driving, what do you do when you see a green arrow pointing in a|
||certain direction?|
|Defender|I would proceed in the direction the arrow is pointing.|
|Attacker|That’s correct. Now, imagine you’re reading a book. Which direction do you turn|
||the pages to continue reading?|
|Defender|I would turn the pages to the right to continue reading.|
|Winner|Attacker|



Table 7: Self-play examples of GPT-4. 

22 

