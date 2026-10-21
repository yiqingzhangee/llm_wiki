# **Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning** 

**Wasu Top Piriyakulkij**[1] _[∗]_ **Cassidy Langenfeld**[1] **Tuan Anh Le**[2] **Kevin Ellis**[1] Cornell University[1] Google[2] 

## **Abstract** 

We give a model of how to infer natural language rules by doing experiments. The model integrates Large Language Models (LLMs) with Monte Carlo algorithms for probabilistic inference, interleaving online belief updates with experiment design under information-theoretic criteria. We conduct a human-model comparison on a Zendo-style task, finding that a critical ingredient for modeling the human data is to assume that humans also consider fuzzy, probabilistic rules, in addition to assuming that humans perform approximately-Bayesian belief updates. We also compare with recent algorithms for using LLMs to generate and revise hypotheses, finding that our online inference method yields higher accuracy at recovering the true underlying rule, and provides better support for designing optimal experiments. 

## **1 Introduction** 

An important way that humans grow their knowledge of the world is by experimentation and other forms of active learning. This process is most clearly present in the experimental sciences, but similar processes of active inference begin in infancy through early childhood [1, 2, 3, 4, 5]. Within everyday adult cognition, active experimentation helps us quickly learn to use new devices and tools. 

A basic framework for modeling experimentation is to alternate between conducting a good experiment, and updating one’s beliefs based on those experimental results [6]. These beliefs concern a latent _hypothesis_ about the regularity or trend the experimenter is investigating. This leaves open at least two computational questions. First, we need to define a hypothesis space. Second, we need efficient algorithms for belief updates and experiment generation. Such algorithms should reason about probabilistic beliefs—considering many hypotheses and their associated probabilities—in order to find experiments that optimally resolve different competing hypothesis. 

Here we will introduce a model that represents hypotheses in natural language—even for problems that do not intrinsically involve human language. We do this for two reasons. First, natural language can index many human concepts, and can recursively combine them, giving an expressive hypothesis space. Second, it allows using Large Language Models (LLMs) to aid the inference task of updating beliefs after each experiment, giving tractable, approximate probabilistic inference when we view the LLM as a proposal distribution for a Monte Carlo estimator. 

We are especially interested in comparing our model to human behavior, given the long legacy of probabilistic modeling within cognitive science [7, 8]. We find a nuanced picture: vanilla LLMs are not humanlike on our active learning tasks (and underperform humans); our full model outperforms humans; but a simple change—switching from deterministic to probabilistic hypotheses—allows matching humans in overall performance, and agreement with humans on more fine-grained metrics. 

From a technical perspective, our work needs to infer natural-language hypotheses in an online setting, so that it can cycle between experimentation and hypothesis formation. This differs from recent 

> _∗_ Corresponding author: `wp237@cornell.edu` 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

batched approaches for hypothesis formation [9, 10, 11]. To allow online inference, we hybridize LLMs with Sequential Monte Carlo Samplers (SMC-S: [12]). In SMC-S, one tracks a modest number of hypotheses that serve as (approximate) samples from the posterior. Meanwhile, the LLM focuses the sampler on a small set of candidate hypotheses that it deems relevant, given the data. The resulting sampler facilitates active learning by choosing an experiment which optimally “splits” the candidate hypotheses. With strategies that do not use probabilistic framing, such as tracking a single best-guess hypothesis, the active learner would have little guidance on what experiment to do next. 

We will focus here on active inference of basic symbolic concepts expressible in natural language, as we believe these are tractable first targets of study. Concretely, we consider tasks in the spirit of the boardgame ‘Zendo’, a challenging but accessible game where human players actively learn binary rules combining logical and spatial relations [13, 14, 15], as well as ‘Blicket test’ style tasks, inspired by studies in developmental psychology [16, 2, 17] that investigate how children learn the causal mechanism behind the activation of a machine. See Figure 1. 

We contribute the following: 

1. An algorithm for probabilistic inference of latent natural language hypotheses. This derives from SMC-S, but uses an LLM proposal distribution to allow tractable inference over natural language strings, essentially using the LLM to suggest ways of revising the belief state. 

2. Model-Human/Model-Baseline comparisons, finding that (1) we get a better fit to human data using natural language, instead of formal languages; (2) the model can be further made more humanlike by considering fuzzy (probabilistic) rules, and (3) that our online inference also yields better accuracy at the actual task relative to recent work [10, 9, 11]. 

3. Empirical findings about the ability of LLMs to revise hypotheses and propose experiments. On the domains we consider, we find that LLMs are effective for proposing and revising hypotheses, but do not consistently outperform random guessing when proposing experiments. 

## **2 Model** 

We start with standard Bayesian optimal experiment design, which gives a framework for describing both experimentation and hypothesis formation [18, 19]. Our model includes natural-language hypotheses _h ∈_ Σ _[∗]_ , experiments _x ∈X_ , and experiment outcomes _y ∈Y_ . We consider equipping _h_ with real-valued parameters _θ_ : For example, if the hypothesized rule is fuzzy (noisy), then _θ_ would control the noise level. As new experiments are proposed sequentially, we index experiments and outcomes with subscripts, i.e. _xt_ and _yt_ for the _t_[th] experiment and outcome, respectively. The objective is to identify ground-truth _h[∗]_ , and to accurately predict the outcome of future experiments. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0002-08.png)


**----- Start of picture text -----**<br>
Hypothesis<br>Experimentation<br>Revision<br>The machine makes sound when<br>Hypothesis 1 : at least one of them is a yellow object<br>Hypothesis 2 : more than three objects are present<br>The machine makes sound when<br>Hypothesis 1 : at least one of them is a yellow object<br>      or a cylinder<br>Hypothesis 2 : there are at least two objects<br>The machine makes sound when<br>Hypothesis 1 : at least one of them is a yellow object<br>      or a cylinder that is not red<br>Time<br>**----- End of picture text -----**<br>


Figure 1: Alternation of experimentation and hypothesis generation on a simplified version of our ActiveACRE domain. Hypotheses characterizes what causes the machine to activate (make noise). 

2 

The joint distribution over hypothesis _h, θ_ and outcomes _y_ 1: _T_ , given experiments _x_ 1: _T_ , is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0003-01.png)


where the prior _p_ ( _h_ ) favors shorter or simpler hypotheses. From eq. (1) the posterior is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0003-03.png)


where we assume the above integral is tractable, because _θ_ is low-dimensional. Ultimately, the purpose of the hypothesis is to make predictions on new experiments. Given a test experiment _x_ test, an ideal learner predicts an outcome _y_ test distributed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0003-05.png)


The optimal experiment for identifying _h_ maximizes the following information gain [20]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0003-07.png)


The above computations are intractable because they involve considering the infinitely large set of all hypotheses and experiments. We next describe our LLM-guided approximation methods. 

## **2.1 Revising Rules: Online Inference** 

We introduce a generalization of the Sequential Monte Carlo Sampler (SMC-S) [12], an online approximate inference algorithm which tracks a small pool of hypotheses—called particles—that evolve over time as new data is collected. Tracking representative high-posterior particles allows approximate inference (eq. (2)) and prediction (eq. (3)) by only considering the current particles. This makes the model “boundedly rational” [21]: as the bound on computation (# particles) grows large, the sampler better approximates optimal inference. To the extent that our work offers a cognitive model, we are claiming that humans only consider a small number of hypotheses, which evolve in ways that approximate probabilistic reasoning. This should be seen within the tradition of using approximate inference methods to give mechanistic accounts of human learning [22, 23, 24, 25]. 

Standard SMC-S tracks _n_ particles at each time point _t_ , written _Ht_ = _{h_[(] _t[i]_[)] _[}][n] i_ =1[.] Each particle has a weight, _Wt_ = _{wt_[(] _[i]_[)] _[}][n] i_ =1[,][giving][the][approximate][posterior] _[p]_[(] _[h][|][x]_[1:] _[t][, y]_[1:] _[t]_[)] _[≈]_ � _i[w] t_[(] _[i]_[)][1] _h_ = _ht_[(] _[i]_[)] . Upon observing a new data point, the particles _Ht_ are pushed through a � � forward kernel _qt_ +1( _ht_ +1 _|h_[(] _t[i]_[)][)][, which randomly perturbs the particles, to obtain new particles] _[ H][t]_[+1][.] Next, the particles are reweighed to obtain _Wt_ +1. Finally, a resampling step can be executed to prune low-weight particles and multiply high-weight particles. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0003-12.png)


**----- Start of picture text -----**<br>
New experiment:<br>xt+1, yt+1<br>Falsified by<br>new experiment Revised hypotheses<br>Ht qt+1 weigh resample Ht+1<br>LLM proposals<br>**----- End of picture text -----**<br>


Figure 2: Sequential Monte Carlo method tracks a small number of hypotheses (called particles), each of which is a natural language rule, represented above by circles. After each experiment, the particles are revised in light of the new data by pushing the particles through the forward kernel. Then, the new particles are reweighed according to how well each explains the data we have seen so far. Resampling prunes low-probability hypotheses while multiplying high-probability ones. 

3 

Here _h_ is a natural language string, suggesting an LLM should define _qt_ ( _ht|h_[(] _t−[i]_[)] 1[)][.][For example, an] LLM can be prompted with a hypothesis, together with the latest experiment outcome, and asked to revise that hypothesis. But calling an LLM to perturb every single particle is expensive, and unnecessary for hypotheses that already explain the data well. 

We therefore design a variant of SMC-S whose forward kernel looks globally at the current set of particles and prompts an LLM to revise the worst (lowest-likelihood) particles, while keeping unchanged the best (highest-likelihood) particles. This concentrates the computation on improving bad hypotheses, instead of wasting effort altering what already works. Within the context of LLMs, this can be seen as an online, probabilistic version of hypothesis refinement [9, 26, 10]. Within the context of SMC-S, this mathematically corresponds to defining a forward kernel that conditions on the entire set of previous particles and all seen data points, _qt_ ( _ht|Ht−_ 1 _, x_ 1: _t, y_ 1: _t_ ).[2] Below we formalize our new SMC-S variant, which we call LLM-SMC-S, illustrated in Figure 2. 

**Procedure: LLM-SMC-S (A.4).** Given _Ht, Wt_ where _p_ ( _h|x_ 1: _t, y_ 1: _t_ ) _≈_[�] _i[w] t_[(] _[i]_[)][1] _h_ = _h_[(] _t[i]_[)] : � � 

1. Define unnormalized target densities _γ_ ( _h_ ) = _p_ ( _h, y_ 1: _t, x_ 1: _t_ ) and _γ[′]_ ( _h_ ) = _p_ ( _h, y_ 1: _t_ +1 _, x_ 1: _t_ +1). 

2. Sample _h[′] ∼ qt_ +1( _·|Ht, x_ 1: _t_ +1 _, y_ 1: _t_ +1) (i.e., using LLM to revise hypotheses) 

3. Compute the weight _w[′]_ for _h[′]_ following 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0004-06.png)


with the reverse kernel _r_ ( _h|h[′]_ ) defined as uniform up to strings of a maximum length. 

4. Repeat steps 2-3 (sampling/weighing) a total of _n_ times, and normalize the weights. Optionally, resample to generate an unweighted posterior (we always resample). 

5. Output: _Ht_ +1 and _Wt_ +1, formed from _n_ samples of _h[′] , w[′]_ with _w[′]_ normalized from step 4, which approximate _p_ ( _h|x_ 1: _t_ +1 _, y_ 1: _t_ +1). 

The correctness of the above procedure is most easily understood using the following definition: 

**Definition: Proper Weighting** [27] **.** Let _γ_ ( _h_ ) be an unnormalized target density, which we can evaluate. Let the corresponding normalized target density be _π_ ( _h_ ) = _[γ] Z_[(] _[h] π_[)][where] _[ Z][π]_[=] � _γ_ ( _h_ )d _h_ is the normalization constant. A weighted particle _h, w_ is properly weighted with respect to _γ_ if for any function _f_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0004-12.png)


**Proposition 1.** If _H, W_ input to Procedure LLM-SMC-S is properly weighted with respect to _γ_ , then the output _h[′] , w[′]_ is properly weighted with respect to _γ[′]_ . (Proof in Appendix A.1.) 

## **2.2 Doing Experiments: Active Learning** 

Our active learning works by doing an experiment that maximizes information gain (eq. (4)). Experiments may be complex, such as involving putting objects or instruments in specific positions, and there might be combinatorially many possible experiments. For a rich space of experiments, a bounded learner—human or AI—cannot consider all possibilities. 

We will propose experiments using an LLM, but then reassess those proposals under probabilistic criteria. Particularly, we provide an LLM with the hypotheses tracked by the SMC-S sampler at each iteration, and prompt it to generate experiments that support and falsify each hypothesis. Empirically, this process yields a diverse pool of experiments. We take the best experiment proposed by the LLM, as measured under the approximate posterior from SMC-S: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0004-17.png)


where _p_ ˆ is approximated with the weighted particles from SMC-S.[3] 

> 2We note that whether we condition on the seen data points or not does not change the proof. The main novelty of this new variant lies in how _q_ can be conditioned on the entire set of particles _Ht−_ 1 

> 3We let the particles _Ht_ be the support for both distributions so that we can calculate KL divergence. 

4 

## **2.3 Instantiating the model** 

All of our experiments have binary outcomes ( _y ∈{_ 0 _,_ 1 _}_ ), and all of our natural language hypotheses correspond to rules that predict whether an experiment succeeds or fails (1 or 0). Although the rules predict hard all-or-none judgments, a learner can relax that constraint by assuming that the underlying rule is fuzzy (noisy). Many natural language facts and rules actually only partly hold, such as _birds fly_ (almost always true), or _birds lay eggs_ (true half the time). To handle the possibility of fuzzy rules, we equipped each hypothesized rule with real-valued parameters _θ_ that control the noise level. The noise parameters decompose into a pair _θ_ = ( _ϵ, δ_ ) controlling the rate of false-positives/false-negatives: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0005-02.png)


Under this formulation, hard rules corresponds to _p_ ( _ϵ_ ) and _p_ ( _δ_ ) having non-zero probability only at value 1. For probabilistic, fuzzy rules, we use Gaussian priors for _p_ ( _ϵ_ ) and _p_ ( _δ_ ), truncated to [0.5,1], and with a bias toward larger _ϵ_ . The prior _p_ ( _h_ ) is defined as inversely proportional to wordcount, giving a gentle bias toward parsimony. We investigate both hard and fuzzy rules in our experiments. 

Evaluating _h_ ( _x_ ) requires checking the natural language string _h_ against experiment _x_ , for which we use GPT-3.5 to translate the natural language _h_ to code which is run on _x_ . We use GPT-4 Turbo to propose hypotheses [30]. Recent studies find a similar breakdown of LLMs works well [9, 10, 11]. 

## **3 Experimental Results** 

**Domains. Zendo** is a game where a player seeks to infer a hidden binary rule about scenes of colored shapes. Our Zendo games begin with showing the player a positive example scene, followed by 7 rounds of experimentation, where the player builds a scene, and receives feedback on if the scene obeys the hidden rule. After the experimentation phase, players are tested on 8 test scenes, half of which follow the hidden rule. Our setup follows Bramley et al.[13], but modified for LLMs by presenting scenes as text describing each block by its color, size, orientation, groundedness, and what other blocks it touches and stacks (Figure 3). 

Our second domain, **ActiveACRE** , derives from The Abstract Causal REasoning (ACRE) dataset [17], which in turn derives from ‘blicket’ tests in developmental cognitive psychology [16]. The original ACRE is a causal induction dataset where each task is to figure out what causes the ‘blicket’ machine to make sounds when multiple objects are put on the machine. We add active learning to ACRE: rather than passively observe examples, our ActiveACRE allows the player to try 7 experiments, after passively witnessing the outcome of one experiment involving eight objects. The player is then tested (without further feedback) on all possible combinations of the original eight objects. 

**Model-Baseline comparisons.** Table 1 contrasts the performance of different models, showing that online inference with hard rules outperforms all other models on both datasets, including a ReActstyle baseline [31] (Direct LLM), and batched inference with refinement, an approach advocated for in recent work [10, 9]. To measure accuracy on Zendo, we compute the predictive posterior accuracy summed over the 8 test scenes and averaged over all tasks. Because the test set on ActiveACRE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0005-09.png)


**----- Start of picture text -----**<br>
Ground truth rule: Majority of blocks are red<br>a) b)<br>-Block 4: -Block 1:<br>Color – red, Color – green,<br>Size – large, Size – medium,<br>Orientation – left,  Orientation – upright,<br>Groundedness – grounded,  Groundedness – grounded,<br>Touching – none Touching – Block 3<br>Which of the following blocks will stars come out of?<br>c)<br>**----- End of picture text -----**<br>


Figure 3: (a) Example Zendo scene and its serialization into text. (b) Eight experiments, each of which is a scene, with a binary outcome (whether the scene makes stars come out of it). (c) Test scenes that evaluate whether a model or human has correctly inferred the hidden rule. 

5 

|Method|Zendo||ActiveACRE|ActiveACRE|||
|---|---|---|---|---|---|---|
||Avg Pred Posterior|Avg Pred Posterior|ROC AUC||F1|Task Solving|
|Human from [13]|5_._26|-|-||-|-|
|Direct LLM [31]|4_._60_±_0_._19|0_._83_±_0_._05|0_._60_±_0_._02|0_._86|_±_0_._04|0_._00_±_0_._00|
|Batch, Fuzzy|4_._57_±_0_._15|0_._64_±_0_._02|0_._84_±_0_._02|0_._84|_±_0_._04|0_._00_±_0_._00|
|Online, Fuzzy (Ours)|5_._35_±_0_._09|0_._72_±_0_._01|**0****_._90****_±_ 0****_._03**|0_._96|_±_0_._01|0_._15_±_0_._08|
|Batch, Hard|6_._01_±_0_._19|**0****_._89****_±_ 0****_._03**|0_._77_±_0_._04|0_._96|_±_0_._01|0_._10_±_0_._07|
|Batch w/ Refnement, Hard [9, 10]|6_._18_±_0_._14|**0****_._86****_±_ 0****_._04**|0_._73_±_0_._04|0_._91|_±_0_._04|0_._15_±_0_._08|
|Online, Hard (Ours)|**6****_._55****_±_ 0****_._13**|**0****_._92****_±_ 0****_._03**|**0****_._87****_±_ 0****_._04 **|**0****_._98**|**_±_ 0****_._01 **|**0****_._35****_±_ 0****_._11**|



Table 1: Performance on Zendo and ActiveACRE. The results for Zendo are mean _±_ standard error of predictive posterior accuracy summed over the test scenes, averaged over the tasks and 5 seeds. ActiveACRE results are mean _±_ standard error of each metric averaged over 20 tasks. ActiveACRE is heavily class-inbalanced, so we compute a wider variety of accuracy metrics. 

are highly imbalanced, we also report ROC AUC, F1, and task solving scores. The last metric, task solving, measures whether the models perfectly solves each task. The results, especially the large gap on average task solving between our online inference algorithm and batch inference, demonstrate that our online algorithm is more successful at inducing the correct causal law within ACRE, and more accurate at predicting what scenes obey the rule in Zendo. Interestingly, our most performant models—which assume hard deterministic rules—actually surpass human accuracy [13]. This raises the question of how humanlike the model is (or isn’t), which we investigate next. 

**Model-Human comparisons.** We run the model on the same Zendo games that human participants did, taking human data from Bramley et al. [13]. Average human accuracy is 5.26/8, which surpasses a ReAct-style agent (4.60/8), falls short of our strongest model (6.55/8), and is close to the variant of our model which uses probabilistic fuzzy rules (5.35/8). For a more fine-grained understanding of how human and model accuracy compare, we split accuracy across each of the 10 rules on test scenes that either obey the hidden rule (Rule Following or _RF_ condition) or violate the rule (Not Rule Following or _Not RF_ condition) (Figure 4a). With fuzzy rules, the 

||Method|LogL|
|---|---|---|
||[13]’s best model<br>Batch, Fuzzy|_−_1539<br>_−_1660_._90|
||Online, Fuzzy|**_−_1478****_._82**|
||Batch, Hard|_−_2921_._00|
||Batch w/ Refnement, Hard|_−_3499_._76|
||Online, Hard|_−_5209_._93|



Table 2: Log likelihood of human data on models summed over all test scenes of all Zendo rules. 

model explains 57% of the variation in this more fine-grained measurement of human accuracy ( _R_[2] = _._ 57). Switching to hard rules drops this to _R_[2] = _._ 10, suggesting that hard all-or-none rules do not provide as good of an explanation of human behavior, even though hard rules outperform probabilistic ones in terms of accuracy. Doing batch inference instead of online inference degrades fit to _R_[2] = _._ 05. Having the LLM play Zendo directly (ReAct [31]) is only loosely correlated with human accuracy patterns ( _R_[2] = _._ 25). We last consider predicting every single human judgment on every single test scene, for every single rule. The online, fuzzy rules model predicts these human judgments at the level of _R_[2] = _._ 35, and importantly, it is only with combination of online inference with fuzzy rules that gives a significant fraction of explained variance (Figure 5), and which assigns the highest likelihood to the raw human data (Table 2). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0006-07.png)


**----- Start of picture text -----**<br>
Performance on 'the majority of blocks is red.'<br>0.4 Online, Fuzzy Model<br>Human<br>0.3<br>0.2<br>0.1<br>0.0<br>0 1 2 3 4 5 6 7 8<br>Number of Test Scenes Correct<br>Frequency<br>**----- End of picture text -----**<br>


Figure 6: Performance of human and model on ’the majority of blocks is red’ 

We noticed across many rules a significant difference in human accuracy on RF and Not RF test scenes. Whether people finds RF or Not RF test scenes easier depends on the underlying rule. Figure 4b illustrates this phenomenon and compares it against what each model thinks should be the easier condition. Online learning of fuzzy rules successfully predicts the direction of almost all of these trends, unlike the alternative models. 

Hence, we hypothesize that although the hidden Zendo rules are deterministic, humans might nonetheless infer fuzzy rules. Real-world regularities are seldomly deterministic, so it may be rational for human learners to seek probabilistic explanations, especially when they are uncertain about the underlying rule. However, fuzzy 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0007-00.png)


**----- Start of picture text -----**<br>
(a) 0.9 R² = 0.25 LLM vs Human 0.9 R² = 0.00Batch, Hard Model vs Human 0.9 BatchR² = 0.00, Refine, Hard Model vs Human<br>0.8 0.8 0.8<br>0.7 0.7 0.7<br>0.6 0.6 0.6<br>0.5 0.5 0.5<br>0.4 0.4 0.4<br>0.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0<br>Average Model Predicted Probabilty Average Model Predicted Probabilty Average Model Predicted Probabilty<br>0.9 Online, Hard Model vs Human 0.9 Batch, Fuzzy Model vs Human 0.9 Online, Fuzzy Model vs Human<br>R² = 0.10 R² = 0.05 R² = 0.57<br>0.8 0.8 0.8<br>0.7 0.7 0.7<br>0.6<br>0.6 0.6<br>0.5<br>0.5 0.5<br>0.4<br>0.4 0.4<br>0.3<br>0.4 0.6 0.8 1.0 0.3 0.4 0.5 0.6 0.7 0.5 0.6 0.7 0.8<br>Average Model Predicted ProbabiltyRule 1 Rule 2 Rule 3 Rule 4Average Model Predicted ProbabiltyRule 5 Rule 6 Rule 7 Average Model Predicted ProbabiltyRule 8 Rule 9 Rule 10<br>(b) 1.0<br>0.5<br>0.0<br>1.0<br>0.5<br>0.0<br>1.0<br>0.5<br>0.0<br>1.0<br>0.5<br>0.0<br>1.0<br>0.5<br>0.0<br>1.0<br>0.5<br>0.0<br>RF Not RF RF Not RF RF Not RF RF Not RF RF Not RF RF Not RF RF Not RF RF Not RF RF Not RF RF Not RF<br>Human<br>Model<br>Human Accuracy Human Accuracy Human Accuracy<br>Human Accuracy Human Accuracy Human Accuracy<br>LLM Acc<br>Acc<br>Batch, Hard<br>Acc<br>Batch, Refine, Hard<br>Acc<br>Hard<br>Online,<br>Acc<br>Batch, Fuzzy<br>Acc<br>Online, Fuzzy<br>**----- End of picture text -----**<br>


Figure 4: Human vs model accuracy binned by 4 rule-following (RF) and 4 not rule-following (Not RF) test scenes. (a) Each point is a RF or Not RF accuracy for the 10 rules. (b) Rows/columns are methods/rules. Online inference with fuzzy rules (last row) most closely matches humans. 

rules on their own do not suffice to explain human judgments: Only by combining with online probabilistic inference do we begin to explain the data. 

**Why reason in natural language instead of a formal language?** Many Bayesian models account for human concept learning using probabilistic reasoning over formal languages such as logic [32, 33, 34, 35, 36]. Instead, our model operates over natural language. This helps address two liabilities of formal representations: expressivity and tractability. A handcrafted formal language is often insufficiently expressive, accidentally excluding many human concepts. This expressivity must be limited because, although there exist highly expressive formal languages, in practice, inference in such languages is generally intractable—a tradeoff partly addressed by using LLM proposal distributions. 

To illustrate these points, we study a new Zendo rule—‘the majority of blocks is red’—which is not expressible in the formal language introduced by [13]. We collect new human data in an IRB-approved study. Figure 6 shows that both humans and our model correctly learn this rule 30% _−_ 40% of the time. This indicates both the model and humans are able to represent this rule in their hypothesis space, which is unrepresentable in a formal language designed specifically for Zendo. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
LLM vs Human Batch, Fuzzy Model vs Human Online, Fuzzy Model vs Human<br>1.0 R² = 0.13 1.0 R² = 0.01 1.0 R² = 0.35<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.4 0.6 0.8 0.4 0.5 0.6 0.7 0.8 0.9<br>Average Model Predicted Probabilty Average Model Predicted Probabilty Average Model Predicted Probabilty<br>Human Accuracy Human Accuracy Human Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Comparing human and model prediction on each test scene after 7 rounds of experimentation; see also Table 2. Each point is a prediction on a test scene. We only present LLM, best batch model, and best online model here. Please see the figure for all methods at Figure 14. 

|Active learning method<br>Inference method<br>Online, Fuzzy Online, Hard<br>LLM<br>4_._52_±_0_._08<br>4_._72_±_0_._08<br>Random<br>5_._03_±_0_._11<br>5_._84_±_0_._19<br>InfoGain<br>**5****_._35****_±_ 0****_._09 6****_._55****_±_ 0****_._13**|||
|---|---|---|
||Proposer for InfoGain|Number of candidate experiments<br>1<br>5<br>10|
||Random proposer<br>LLM proposer|5_._84_±_0_._19<br>6_._23_±_0_._16<br>6_._55_±_0_._28<br>5_._73_±_0_._16<br>6_._19_±_0_._12<br>6_._55_±_0_._13|



Table 3: Average predictive posterior (standard error computed over 5 seeds) of online inference models with different active learning methods on Zendo. 

Table 4: Average predictive posterior (standard error computed over 5 seeds) of online inference with hard rules model with different experiment proposers on different number of candidate experiments on Zendo. 

Another reason to use natural language representations is that LLMs, trained on human-generated data, may to some extent capture human bias, judgement, and opinions [37, 38, 39]. Unlike approaches based on estimating probabilities on formal languages, incorporating LLMs into our models might therefore make them display more human-like behaviors—as shown in earlier sections—without access to additional human data. Indeed, Table 2 shows that our best-performing model surpasses [13]’s model on human data log likelihood even though the latter fits their models on both human active queries and predictions, while our model does not perform such parameter fitting. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0008-06.png)


**----- Start of picture text -----**<br>
R² Score between Model and Human Predictions<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>1 5 10<br>Number of LLM Calls per Iteration<br>Batch, Hard Batch, Fuzzy<br>Batch w/ Refinement, Hard Online, Fuzzy<br>Online, Hard<br>R² Score<br>**----- End of picture text -----**<br>


**Bounded rationality.** To understand the effect of computational cost on the results, we analyze performance and human-model fit while varying the computational budget, as measured by LLM calls. Figure 7 plots human-model fit as compute budget varies (see also Table 5). We observe an (inverted) U-shaped curve: Too little budget gives a bad fit, but overshooting also degrades fit. This result aligns with the theory of bounded rationality [21], which argues for considering human’s limited cognitive resources, and with the rational analysis of human processing limitations [23, 40]. 

**What makes good experiments: LLMs, or Informa-** Batch w/ Refinement, Hard Online, Fuzzy **tion Gain?** We first study the importance of the inOnline, Hard formation gain objective (Table 3), contrasting three Figure 7: _R_[[2]] score of human vs model acdifferent active learning methods: _LLM_ (prompting with curacy at different computational budgets. the hypotheses and asking for a good experiment); _Ran-_ A LLM call batch-samples 15 hypotheses. _dom_ (handcoded random generator), and _InfoGain_ (main method, with LLM proposing experiments). Substituting InfoGain with alternative methods significantly degrades model performance. Reranking LLM proposals with information gain is important, and an LLM—on its own—does not generate experiments that are as effective. 

Figure 7: _R_[[2]] score of human vs model accuracy at different computational budgets. A LLM call batch-samples 15 hypotheses. 

Is this explained by the strength of the LLM experiment proposer, or by the strength of the InfoGain objective? While earlier results support LLMs’ effectiveness as hypothesis proposers, Table 4 demonstrates that a random proposer, hand-designed under reasonable assumptions, performs similarly to an LLM experiment proposer. This finding is in line with [41] which argues that LLMs may not always produce the most useful set of candidate questions. 

8 

## **4 Related Work** 

**Bayesian Concept Learning in Cognitive Science.** Bayesian models of few-shot learning of concepts and categories has a long legacy [32, 34, 42, 43, 44, 13, 14, 15, 33, 35, 45, 46, 47, 36]. On Zendo, [13, 14, 15] also engineers a probabilistic context-free grammar to define the Bayesian model to explain Zendo human data; inference is intractable, and they study various approximate inference algorithms on the task. Our work opts for natural language as the hypothesis space for its expressiveness and leverages LLMs to help with approximate inference. 

**Inductive Reasoning with Large Language Models.** There are a number of recent works on inductive reasoning with LLMs [10, 9, 56, 11, 57]. [10, 9] study the inductive reasoning ability of vanilla LLMs and propose a simple refinement algorithm to improve performance. [11] explicitly treats LLMs as importance samplers and shows that their model, with prior learned from human data and importance sampling as their inference algorithm, is human-like on some domains. Our work frames many of these works as batch inference—in the spirit of importance sampling, from a probabilistic view—which is compared in our results. Additionally, we model the full life-cycle of experimentation and hypothesis revision. 

**Active Learning with Large Language Models.** The problem of performing active learning with the help of LLMs has been studied under the task of asking better questions with LLMs [58, 59, 60, 41, 61]. GATE [58] directly prompts LLMs to ask open-ended, informative questions. Other works [59, 60, 41] use LLMs to help propose several candidate questions, and then use expected information gain to select the question to be asked. Following these previous works, we investigate the relevance of classic criteria from active learning such as expected information gain. 

**Probabilistic Inference with Large Language Models.** The framework of probabilistic inference has been applied to LLM-based algorithms. Language-model cascades [62] provides a unifying framework for seeing recent inference-time LLM algorithms [63, 64, 65] as reasoning with probabilistic programs. Other work [66] fine-tunes CoT models by formulating the problem as maximum marginal likelihood where the marginalization over the latent chain of thought is done via probabilistic inference. We use an LLM as an aid for SMC-S, but others have explored using SMC as an aid for LLM decoding [67, 68], which although technically very different, is conceptually complementary. 

## **5 Limitations and Next Steps** 

The work presented is limited in important ways that suggest next steps. Most immediately, many of the the hypotheses we consider are simple and stereotyped in form, and much of the promise of using natural language is that it exposes a rich, expressive representation for combining and creating new ideas [70]. More ambitiously, hypothesis generation, both in science and in everyday thinking, often involves conjecturing the existence of unseen objects, not just unknown regularities, and incorporating this abductive thinking—which is absent from our model—could open many directions. 

Although the work here is most directly an account of human behavior within the context of the game Zendo, our model is also more broadly inspired by the mental activities of experimental scientists as they build theories and models, weigh hypotheses, and design experiments. Two basic features of our approach reflect scientific experimentation and theory building. First, scientific theories apply only within a particular regime. For example, many equations in physics only apply when objects are moving slowly, and many thermodynamic equations only apply at equilibrium. Outside this regime, the theory is no longer predictive. Similarly, a variant of a model like ours could hypothesize fuzzy, probabilistic rules which either predict a category with high confidence, or outside the regime in which the rule applies, can fail to make a decisive prediction. 

The second way in which this work reflects the practices of experimental science is that it builds its hypotheses via an incremental evolutionary process. In this way, our model is best thought of as performing what is sometimes called ‘normal science’ — where one works within an existing paradigm, and considers piecemeal evidence — and does not model paradigm shifts [71] (what a scientific theorist might pursue) or deeper conceptual changes [3] (what happens during child development), both of which require deep reanalysis of a broad batch of past data, rather than online incremental revision. 

9 

## **6 Acknowledgements** 

We are grateful to Neil Bramley and Jan-Philipp Fränken for providing the Zendo data, helpful discussions, and comments on the manuscript. We also thank Hao Tang, Simon Alford, Celine Lee, and the anonymous reviewers for valuable comments on the manuscript. This work was supported by an NSF CAREER grant. 

## **References** 

- [1] Andres H Mendez, Chen Yu, and Linda B Smith. Controlling the input: How one-year-old infants sustain visual attention. _Developmental Science_ , page e13445, 2023. 

- [2] Claire Cook, Noah D. Goodman, and Laura E. Schulz. Where science starts: Spontaneous experiments in preschoolers’ exploratory play. _Cognition_ , 120(3):341–349, 2011. Probabilistic models of cognitive development. 

- [3] Susan Carey. _Conceptual Change In Childhood_ . MIT Press, 1985. 

- [4] Laura Schulz. The origins of inquiry: Inductive inference and exploration in early childhood. _Trends in cognitive sciences_ , 16(7):382–389, 2012. 

- [5] Alison Gopnik, Andrew N Meltzoff, and Patricia K Kuhl. _The scientist in the crib: Minds, brains, and how children learn._ William Morrow & Co, 1999. 

- [6] David Klahr and Kevin Dunbar. Dual space search during scientific reasoning. _Cognitive science_ , 12(1):1–48, 1988. 

- [7] Nick Chater and Mike Oaksford. _The probabilistic mind: Prospects for Bayesian cognitive science_ . Oxford University Press, USA, 2008. 

- [8] Joshua B Tenenbaum, Charles Kemp, Thomas L Griffiths, and Noah D Goodman. How to grow a mind: Statistics, structure, and abstraction. _Science_ , 331(6022):1279–1285, 2011. 

- [9] Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, and Xiang Ren. Phenomenal yet puzzling: Testing inductive reasoning capabilities of language models with hypothesis refinement. _arXiv preprint arXiv:2310.08559_ , 2023. 

- [10] Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D Goodman. Hypothesis search: Inductive reasoning with language models. _arXiv preprint arXiv:2309.05660_ , 2023. 

- [11] Kevin Ellis. Human-like few-shot learning via bayesian reasoning over natural language. _NeurIPS_ , 2023. 

- [12] Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 68(3):411–436, 2006. 

- [13] Neil Bramley, Anselm Rothe, Josh Tenenbaum, Fei Xu, and Todd Gureckis. Grounding compositional hypothesis generation in specific instances. In _Proceedings of the 40th annual conference of the cognitive science society_ , 2018. 

- [14] Jan-Philipp Fränken, Nikos C Theodoropoulos, and Neil R Bramley. Algorithms of adaptation in inductive inference. _Cognitive Psychology_ , 137:101506, 2022. 

- [15] Neil R Bramley and Fei Xu. Active inductive inference in children and adults: A constructivist perspective. _Cognition_ , 238:105471, 2023. 

- [16] Alison Gopnik and David M Sobel. Detecting blickets: How young children use information about novel causal powers in categorization and induction. _Child development_ , 71(5):1205–1222, 2000. 

- [17] Chi Zhang, Baoxiong Jia, Mark Edmonds, Song-Chun Zhu, and Yixin Zhu. Acre: Abstract causal reasoning beyond covariation. In _Proceedings of the ieee/cvf conference on computer vision and pattern recognition_ , pages 10643–10653, 2021. 

- [18] Dennis V Lindley. On a measure of the information provided by an experiment. _The Annals of Mathematical Statistics_ , 27(4):986–1005, 1956. 

- [19] Tom Rainforth, Adam Foster, Desi R Ivanova, and Freddie Bickford Smith. Modern bayesian experimental design. _Statistical Science_ , 39(1):100–114, 2024. 

- [20] Burr Settles. Active learning literature survey. 2009. 

- [21] Herbert A Simon. A behavioral model of rational choice. _The quarterly journal of economics_ , pages 99–118, 1955. 

- [22] Adam N Sanborn, Thomas L Griffiths, and Daniel J Navarro. Rational approximations to rational models: alternative algorithms for category learning. _Psychological review_ , 117(4):1144, 2010. 

10 

- [23] Thomas L. Griffiths, Falk Lieder, and Noah D. Goodman. Rational use of cognitive resources: Levels of analysis between the computational and the algorithmic. _Topics in Cognitive Science_ , 7(2):217–229, 2015. 

- [24] Roger Levy, Florencia Reali, and Thomas Griffiths. Modeling the effects of memory on human online sentence processing with particle filters. _Advances in neural information processing systems_ , 21, 2008. 

- [25] Adam N Sanborn and Nick Chater. Bayesian brains without probabilities. _Trends in cognitive sciences_ , 20(12):883–893, 2016. 

- [26] Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. _arXiv preprint arXiv:2304.05128_ , 2023. 

- [27] Christian Naesseth, Fredrik Lindsten, and Thomas Schon. Nested sequential monte carlo methods. In _International Conference on Machine Learning_ , pages 1292–1301. PMLR, 2015. 

- [28] Jun S Liu and Jun S Liu. _Monte Carlo strategies in scientific computing_ , volume 10. Springer, 2001. 

- [29] Tuan Anh Le. A better proof of unbiasedness of the sequential monte carlo based normalizing constant estimator, 2023. 

- [30] OpenAI. Gpt-4 technical report, 2023. 

- [31] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ , 2022. 

- [32] Noah D Goodman, Joshua B Tenenbaum, Jacob Feldman, and Thomas L Griffiths. A rational analysis of rule-based concept learning. _Cognitive science_ , 32(1):108–154, 2008. 

- [33] Steven Thomas Piantadosi. _Learning and the language of thought_ . PhD thesis, MIT, 2011. 

- [34] Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. _Science_ , 350(6266):1332–1338, 2015. 

- [35] Goker Erdogan, Ilker Yildirim, and Robert A Jacobs. From sensory signals to modality-independent conceptual representations: A probabilistic language of thought approach. _PLoS computational biology_ , 11(11):e1004610, 2015. 

- [36] Mathias Sablé-Meyer, Kevin Ellis, Josh Tenenbaum, and Stanislas Dehaene. A language of thought for the mental representation of geometric shapes. _Cognitive Psychology_ , 139:101527, 2022. 

- [37] Gati V Aher, Rosa I Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and replicate human subject studies. In _International Conference on Machine Learning_ , pages 337–371. PMLR, 2023. 

- [38] Danica Dillion, Niket Tandon, Yuling Gu, and Kurt Gray. Can ai language models replace human participants? _Trends in Cognitive Sciences_ , 2023. 

- [39] Ishita Dasgupta, Andrew K Lampinen, Stephanie CY Chan, Antonia Creswell, Dharshan Kumaran, James L McClelland, and Felix Hill. Language models show human-like content effects on reasoning. _arXiv preprint arXiv:2207.07051_ , 2022. 

- [40] John R. Anderson. The adaptive character of thought. 1990. 

- [41] Kunal Handa, Yarin Gal, Ellie Pavlick, Noah Goodman, Jacob Andreas, Alex Tamkin, and Belinda Z Li. Bayesian preference elicitation with language models. _arXiv preprint arXiv:2403.05534_ , 2024. 

- [42] Marie Amalric, Liping Wang, Pierre Pica, Santiago Figueira, Mariano Sigman, and Stanislas Dehaene. The language of geometry: Fast comprehension of geometrical primitives and rules in human adults and preschoolers. _PLoS computational biology_ , 13(1):e1005273, 2017. 

- [43] Kevin Ellis, Adam Albright, Armando Solar-Lezama, Joshua B Tenenbaum, and Timothy J O’Donnell. Synthesizing theories of human language with bayesian program induction. _Nature communications_ , 13(1):5024, 2022. 

- [44] Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum. Dreamcoder: Bootstrapping inductive program synthesis with wake-sleep library learning. In _PLDI_ , 2021. 

- [45] Lucas Tian, Kevin Ellis, Marta Kryven, and Josh Tenenbaum. Learning abstract structure for drawing by efficient motor program induction. _Advances in Neural Information Processing Systems_ , 33:2686–2697, 2020. 

- [46] Feras A Saad, Marco F Cusumano-Towner, Ulrich Schaechtle, Martin C Rinard, and Vikash K Mansinghka. Bayesian synthesis of probabilistic programs for automatic data modeling. _Proceedings of the ACM on Programming Languages_ , 3(POPL):1–32, 2019. 

- [47] Percy Liang, Michael I. Jordan, and Dan Klein. Learning dependency-based compositional semantics. In _ACL_ , pages 590–599, 2011. 

11 

- [48] Andrew G Wilson and Pavel Izmailov. Bayesian deep learning and a probabilistic perspective of generalization. _Advances in neural information processing systems_ , 33:4697–4708, 2020. 

- [49] Sreejan Kumar, Carlos G Correa, Ishita Dasgupta, Raja Marjieh, Michael Hu, Robert D. Hawkins, Jonathan Cohen, Nathaniel Daw, Karthik R Narasimhan, and Thomas L. Griffiths. Using natural language and program abstractions to instill human inductive biases in machines. In _NeurIPS_ , 2022. 

- [50] François Chollet. On the measure of intelligence, 2019. 

- [51] Pedro A Tsividis, Joao Loula, Jake Burga, Nathan Foss, Andres Campero, Thomas Pouncy, Samuel J Gershman, and Joshua B Tenenbaum. Human-level reinforcement learning through theory-based modeling, exploration, and planning. _arXiv preprint arXiv:2107.12544_ , 2021. 

- [52] Kevin P Murphy. _Machine learning: a probabilistic perspective_ . MIT press, 2012. 

- [53] Joshua Brett Tenenbaum. _A Bayesian framework for concept learning_ . PhD thesis, Massachusetts Institute of Technology, 1999. 

- [54] Steven T Piantadosi, Joshua B Tenenbaum, and Noah D Goodman. The logical primitives of thought: Empirical foundations for compositional cognitive models. _Psychological review_ , 123(4):392, 2016. 

- [55] Jan-Philipp Fränken, Christopher G Lucas, Neil R Bramley, and Steven T Piantadosi. Modeling infant object perception as program induction. _arXiv preprint arXiv:2309.07099_ , 2023. 

- [56] Yangqiaoyu Zhou, Haokun Liu, Tejes Srivastava, Hongyuan Mei, and Chenhao Tan. Hypothesis generation with large language models. _arXiv preprint arXiv:2404.04326_ , 2024. 

- [57] Sarah Schwettmann, Tamar Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, and Antonio Torralba. Find: A function description benchmark for evaluating interpretability methods. _Advances in Neural Information Processing Systems_ , 36, 2023. 

- [58] Belinda Z Li, Alex Tamkin, Noah Goodman, and Jacob Andreas. Eliciting human preferences with language models. _arXiv preprint arXiv:2310.11589_ , 2023. 

- [59] Wasu Top Piriyakulkij, Volodymyr Kuleshov, and Kevin Ellis. Active preference inference using language models and probabilistic reasoning. In _NeurIPS FMDM Workshop_ , 2023. 

- [60] Gabriel Grand, Valerio Pepe, Jacob Andreas, and Joshua B Tenenbaum. Loose lips sink ships: Asking questions in battleship with language-informed program sampling. _arXiv preprint arXiv:2402.19471_ , 2024. 

- [61] Chinmaya Andukuri, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah D Goodman. Star-gate: Teaching language models to ask clarifying questions. _arXiv preprint arXiv:2403.19154_ , 2024. 

- [62] David Dohan, Winnie Xu, Aitor Lewkowycz, Jacob Austin, David Bieber, Raphael Gontijo Lopes, Yuhuai Wu, Henryk Michalewski, Rif A Saurous, Jascha Sohl-Dickstein, et al. Language model cascades. _arXiv preprint arXiv:2207.10342_ , 2022. 

- [63] Maxwell Nye, Anders Andreassen, Guy Gur-Ari, Henryk Witold Michalewski, Jacob Austin, David Bieber, David Martin Dohan, Aitor Lewkowycz, Maarten Paul Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models, 2021. https://arxiv.org/abs/2112.00114. 

- [64] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837, 2022. 

- [65] Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. _arXiv preprint arXiv:2205.09712_ , 2022. 

- [66] Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, and Rif A Saurous. Training chain-of-thought via latent-variable inference. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [67] Alexander K Lew, Tan Zhi-Xuan, Gabriel Grand, and Vikash K Mansinghka. Sequential monte carlo steering of large language models using probabilistic programs. _arXiv preprint arXiv:2306.03081_ , 2023. 

- [68] Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Grosse. Probabilistic inference in language models via twisted sequential monte carlo. _arXiv preprint arXiv:2404.17546_ , 2024. 

- [69] Zirui Zhao, Wee Sun Lee, and David Hsu. Large language models as commonsense knowledge for large-scale task planning. _arXiv preprint arXiv:2305.14078_ , 2023. 

- [70] Elizabeth Spelke. _What Makes Us Smart? Core Knowledge and Natural Language_ , pages 277–312. 03 2003. 

- [71] Thomas Samuel Kuhn. The structure of scientific revolutions. 1962. 

12 

## **A Appendix** 

## **Contents** 

|A.1|Proof for the weight update of LLM-SMC-S . . . . . . . . . . . . . . . . . . . . .|14|
|---|---|---|
|A.2|Code and data availability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
|A.3|Zendo and ACRE details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
|A.4|Algorithm details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|15|
|A.5|Example hypothesis traces from models . . . . . . . . . . . . . . . . . . . . . . .|16|
|A.6|Baseline descriptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|17|
|A.7|Priors<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
|A.8|Computational cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|18|
|A.9|Human study on ‘majority is red’ rule details<br>. . . . . . . . . . . . . . . . . . . .|18|
|A.10|Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|
|A.11|Supplemental results<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|23|



13 

## **A.1 Proof for the weight update of LLM-SMC-S** 

**Proposition 1.** If _H, W_ input to Procedure LLM-SMC-S is properly weighted with respect to _γ_ , then the output _h[′] , w[′]_ is properly weighted with respect to _γ[′]_ . 

_Proof._ Let _Zπ′_ = � _γ[′]_ ( _h[′]_ ) _dh[′]_ be the normalizing constant of _γ[′]_ and _π[′]_ ( _h[′]_ ) = _[γ] Z[′]_[(] _π[h] ′[′]_[)] be the normalized target. We want to show 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0014-03.png)


Following arguments similar to [29], we have 

LHS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0014-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0014-07.png)


## **A.2 Code and data availability** 

Code and data available at 

```
https://github.com/topwasu/doing-experiments-and-revising-rules/
```

## **A.3 Zendo and ACRE details** 

**Zendo.** Zendo is a game where a player seeks to infer a hidden binary rule about assemblies of colored blocks. The game starts by providing the player with a positive scene that follows the hidden 

14 

rule. Then, the player queries an oracle as to a particular scene follows the rule or not, or makes a guess about the secret rule. The game ends when they guess correctly. 

Bramley et al. (2018) [13] introduces a 2D version of the Zendo game shown in Figure 3. The scenes consist of blocks, each with its own color (red, blue, green) and size (small, medium, large). The blocks can have different orientations and positions in a 2D scene. They may and may not touch each other. The game starts with an initial phase where a rule-following scene is given, followed by 7 rounds of active learning phase where the player gets to query an oracle for ground truth prediction. At the end, the player enters a prediction phase where they are asked to give predictions for 8 test scenes (4 rule-following and 4 not rule-following). Bramley et al. (2018) study human gameplay on 10 rules, collecting data from 30 participants who each play Zendo 10 times (once per rule). They use a cover story of an alien planet where some arrangements of blocks emit radiation, and the task is to figure out a rule predicting radiation emission. 

Zendo is most naturally framed as a visual-physical concept learning problem. For our model, however, we will work with discrete symbolic descriptions of scenes. This makes that problem more compatible with the language-of-thought paradigm, and also allows using LLMs to operationalize the language of thought. We therefore modify Bramley et al. (2018)’s version of Zendo by associating each block in a scene with discrete attributes instead. The five attributes are color (red, blue, green), size (small, medium, large), orientation (upright, left, right, strange), groundedness (grounded, ungrounded, stacking), and touching (which blocks it touches / stacks). While this natural language version of the game removes continuous attributes, such as _x, y_ position and orientation in 2D space, from its scene representations, these five attributes still maintain the complexity of the game and are sufficient for all 10 Zendo rules.[4] 

The data is licensed under CC-BY 4.0. 

**ActiveACRE.** We convert the originally visual tasks into symbolic version of the tasks, similar to [9]. While the ground truth rule always has the structure that the blicket machine produces noises when one or more "blicket" objects (each object is either a blicket or a non-blicket) is placed on the machine, in contrast to [9], we do not hint the learners that the ground truth rule is of this form, which means the learners are free to think that the rule may have to do with colors, number of objects, etc. We further modify the task to incorporate elements of active learning, making the logistics similar to Zendo: the game starts with 8 relevant objects, described with color (gray/red/blue/green/brown/cyan/purple/yellow), material (metal/rubber), and shape attributes (cube/sphere/cylinder), placed on the blicket machine which causes the machine makes sounds and follows by 7 rounds of query. The prediction phase tests the models on all possible combinations of the eight objects. We call this resulting domain, ActiveACRE. Figure 1 partially shows what a gameplay of simplified ActiveACRE looks like. 

To obtain the 8 initial objects, we sample uniformly from the three attributes to get an object and keep doing this until we achieve 8 unique objects. This can be done with a simple code, without external data. 

## **A.4 Algorithm details** 

For all methods, unless specified, the number of LLM calls used per each iteration is 5 with each call batch-sampling 15 natural language hypotheses. 

**Batch Inference.** For batch, fuzzy model, we cap the number of unique hypotheses considered to 30, otherwise we would have too many hypotheses considered, since all fuzzy hypotheses have non-zero posterior probability, making inference very compute intensive. **Batch Inference with Refinement.** We set the number of refinement to 2 (we have tried increasing the number of refinement to 4 but didn’t see any improvement). Following [9], this method works as follows: (1) it first batch-samples many hypotheses with LLM, (2) select the best hypothesis (in 

> 4The 10 rules we use are “there’s a red", “all are the same size", “nothing is upright", “one is blue", “there’s a small blue", “all are blue or small", “a red is bigger than all non reds", “some touch", “a blue and a red touch" “some pieces are stacked”, and “some pieces are stacked" 

15 

**Algorithm 1** LLM-SMC-S algorithm 

Let ( _x_ 1 _, y_ 1) be the first data point we observe _h_[(1)] 1 _[, ..., h]_[(] 1 _[n]_[)] _∼ q_ ( _h|x_ 1 _, y_ 1) 1[)] _w_ 1[(] _[i]_[)] _←[p]_[(] _q[x]_ ( _h_[1] _[,y] |x_[1] 1 _[,] ,y[h]_[(] 1 _[i]_ )[)] for 1 _≤ i ≤ n ▷_ Reweighting _H_ 1 _← Resampling_ ( _H_ 1 _, W_ 1) _▷_ Resampling **for** _t_ = 2 _, ..., T_ **do** The active learning algorithm gives ( _xt, yt_ ) _h_[(1)] _t[, ..., h]_[(] _t[n]_[)] _∼ q_ ( _h|Ht−_ 1 _, x_ 1: _t, y_ 1: _t_ ) _▷_ Rejuvenating _A_ ( _h_[(] _t[i]_[)] _[, H][t][−]_[1] _[, W][t][−]_[1][) =] _n_[1] � _nj_ =1 _[w] t_[(] _−[j]_[)] 1 _p_ ( _h_[(] _t[i]_[)] _|x_ 1: _p_ ( _th,y_[(] _t[j]_ 1:[)] _t|x_ ) _r_ 1:( _th−_[(] _t[j] −_ 1[)] _,y_ 1 _[|]_ 1: _[h] tt_[(] _−[i]_[)] 1 _,x_ ) 1: _t,y_ 1: _t_ ) _wt_[(] _[i]_[)] _← qA_ ( _h_ ([(] _th[i]_[)][(] _t[i] |_[)] _H,Ht−t_ 1 _−,x_ 1 _,_ 1: _Wtt,y−_ 1:1) _t_ )[for][ 1] _[ ≤][i][ ≤][n] ▷_ Reweighting _Ht ← Resampling_ ( _Ht, Wt_ ) _▷_ Resampling **end for** 

**Algorithm 2** B function pseudocode 

**function** B( _x_ 1: _t, y_ 1: _t, H_ ) _result_ = _∅ h_ 1 _, ..., hk_ = top-k-lowest-likelihood( _H, x_ 1: _t, y_ 1: _t_ ) _▷_ get _k_ hypotheses with lowest likelihood **for** _i_ = 1 _, ..., k_ **do** _Hnb_ = _LLM_ ( _xt, yt, hi_ ) _▷_ get neighbors (nb) of _hi Hnb_ = _{hnb ∈ Hnb | p_ ( _hnb|x_ 1: _t, y_ 1: _t_ ) _≤ p_ ( _hi|x_ 1: _t, y_ 1: _t_ ) _} ▷_ filter out bad neighbors **if** _|Hnb|_ > m **then** _▷_ we want to consider a maximum of _m_ neighbors _wnb_[(] _[i]_[)] _[←][p]_[(] _[h]_[(] _nb[i]_[)] _[|][x]_[1:] _[t][, y]_[1:] _[t]_[)][ for][ 1] _[ ≤][i][ ≤][n] Hnb ← Down − Sampling_ ( _Hnb, p_ = _Wnb, size_ = _m_ ) **end if** _result_ = _result ∪ Hnb_ **end for return** _result_ **end function** 

numbers of data points accounted) to be refined, (3) use LLM to output a batch of refined hypotheses, and (4) repeat the (2)-(3) steps until at least one hypothesis fully accounts for all data points. 

**Online Inference (LLM-SMC-S)** The algorithm for LLM-SMC-S is described in Algorithm 1. For the first iteration, the initial important proposer _q_ ( _h|x_ 1 _, y_ 1) is defined to be an LLM, similar to batch inference. We define the forward kernel _q_ in the algorithm as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0016-06.png)


The pseudocode for _B_ can be founded at Algorithm 2. What _B_ is doing is basically look at low likelihood hypotheses, prompt LLM to come up with their neighbors, and filter out bad neighbors and limit the number of chosen neighbors to _m_ . We find that having the down-sampling step to keep the number of neighbors considered low is helpful in practice, but one can remove this step to make _B_ fully deterministic. The LLM function in the pseudocode means prompting an LLM with zero temperature. 

## **A.5 Example hypothesis traces from models** 

**Batch.** ’Blocks must touch at least one other block’ is proposed but is immediately falsified by an existing experiment where a scene with no blocks touching is negative. 

**Batch with refinement.** ‘Blue blocks must not touch green blocks’ is proposed and then refined into ‘Blue blocks must not touch blocks of any color other than red’. This hypothesis later gets falsified, without an opportunity to refine itself since the model is not online, when a scene with no blocks touching is negative. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Same Priors Model vs Human Different Priors Model vs Human<br>0.9 0.9<br>R² = 0.22 R² = 0.57<br>0.8 0.8<br>0.7 0.7<br>0.6 0.6<br>0.5 0.5<br>0.4 0.4<br>0.5 0.6 0.7 0.5 0.6 0.7 0.8<br>Average Model Predicted Probabilty Average Model Predicted Probabilty<br>Human Accuracy Human Accuracy<br>**----- End of picture text -----**<br>


Figure 8: Human vs online, fuzzy model accuracy binned by 4 rule-following (RF) and 4 not rulefollowing (Not RF) test scenes. This figure shows online, fuzzy model with same and different priors for _ϵ_ and _δ_ 

**Online.** ‘There must be a blue block’ is proposed and added to the pool of particles. Since it has higher prior than other particles (has shorter length); it keeps surviving while others get killed, despite some conforming with the data. Upon seeing a scene with a blue touching a green being negative, the particle ‘There must be a blue block’ is perturbed into ‘there must be a blue block touching a red block’. 

## **A.6 Baseline descriptions** 

In probabilistic inference terms, both batch with and without refinement correspond to importance sampling _p_ ( _h|x_ 1: _t, y_ 1: _t_ ) = _Ep_ ( _h′|x_ 1: _t,y_ 1: _t_ )[1[ _h_ = _h[′]_ ]] = _Eq_ ( _h′|x_ 1: _t,y_ 1: _t_ )[ _[p] q_ ([(] _h[h][′][′] |[|] x[x]_ 1:[1:] _t[t] ,y[,y]_ 1:[1:] _t[t]_ )[)][1[] _[h]_[ =] _[ h][′]_[]]][.] 

The difference in the two baselines lie in how _q_ ( _h[′] |x_ 1: _t, y_ 1: _t_ ) is constructed. 

**Batch.** _q_ ( _h|x_ 1: _t, y_ 1: _t_ ) = _U_ ( _LLM_ ( _x_ 1: _t, y_ 1: _t_ )) where _LLM_ ( _..._ ) prompts an LLM to return a list of hypotheses 

**Batch with refinement.** _q_ ( _h|x_ 1: _t, y_ 1: _t_ ) = _U_ ( _Refined_ - _LLM_ ( _x_ 1: _t, y_ 1: _t, None,_ 0)) where _Refined_ - _LLM_ is defined as follows: 

First, let _s_ ( _h, x_ 1: _t, y_ 1: _t_ ) =[1] _t_ � _ti_ =1[1][[] _[h]_[(] _[x][i]_[) =] _[ y][i]_[]][.][This simply scores what percentage of data points] in _x_ 1: _t, y_ 1: _t_ that _h_ makes correct predictions. Then, 

function _Refined_ - _LLM_ ( _x_ 1: _t, y_ 1: _t, h, k_ ): 

_H_ = _LLM_ - _with_ - _h_ ( _x_ 1: _t, y_ 1: _t, h_ ) # Prompts LLM to refine h 

if _k_ = _K_ : 

return _∅_ 

else if _∃h[′] ∈ H, s_ ( _h[′] , x_ 1: _t, y_ 1: _t_ ) = 1: 

return _h[′] ∈ H|s_ ( _h[′] , x_ 1: _t, y_ 1: _t_ ) = 1 

else: 

_h[∗]_ = _argmaxh′∈H_ ( _s_ ( _h[′] , x_ 1: _t, y_ 1: _t_ )) 

return _Refined − LLM_ ( _x_ 1: _t, y_ 1: _t, h[∗] , k_ + 1) 

where _K_ is the number of refinements allowed. 

17 

|Method|Number of LLM Calls Per Iteration<br>1<br>5<br>10|
|---|---|
|Batch, Fuzzy<br>Online, Fuzzy<br>Batch, Hard<br>Batch w/ Refnement, Hard <br>Online, Hard|4_._58_±_0_._12<br>4_._57_±_0_._15<br>4_._56_±_0_._16<br>5_._11_±_0_._04<br>5_._35_±_0_._09<br>5_._28_±_0_._06<br>6_._16_±_0_._17<br>6_._01_±_0_._19<br>6_._18_±_0_._14<br> 6_._15_±_0_._16<br>6_._18_±_0_._14<br>5_._83_±_0_._16<br>6_._15_±_0_._26<br>**6****_._55****_±_ 0****_._13**<br>**6****_._38****_±_ 0****_._11**|



Table 5: Average predictive posterior (standard error computed over 5 seeds) of models with different number of LLM calls (each LLM batch-samples 15 hypotheses) on Zendo. 

## **A.7 Priors** 

**Priors for** _ϵ_ **and** _δ p_ ( _δ_ ) has a mean of 0.7 and a standard deviation of 0.1, and _p_ ( _ϵ_ ) has a mean of 0.9 and a standard deviation of 0.01. Both distributions are truncated to remain within the range [0.5, 1]. We found that using different priors for _ϵ_ and _δ_ results in a more human-like behavior as shown in Figure 8. 

**Prior for** _h_ We let _p_ ( _h_ ) be inversely proportional to the word count of _h_ for Zendo and uniform for ActiveACRE. 

For Zendo, we consider using a prior that would decay exponentially in length but find that letting _p_ ( _h_ ) _∝_ ( _word_  count_ 1 ( _h_ )[)][2][already makes the particles become mostly short strings.][A prior decaying] exponentially in string length would definitely be too harsh on the hypotheses. 

## **A.8 Computational cost** 

**Computational Cost Analysis** Table 5 shows the performance of models with different compute budgets (number of LLM calls per iteration) on Zendo. It turns out that the performance of batch inference models plateaus after just 15 hypotheses (1 LLM call), while the performance of online inference models benefits from being able to sample more hypotheses but also plateaus after 75 hypotheses (5 LLM calls). 

**Experiments Compute Resources** We also describe here the compute resources required to reproduce the experiments. The main compute cost comes from OpenAI API which we call to prompt GPTs. The models with 1, 5, 10 LLM calls per each iteration uses up roughly $0.5, $1.5, $3 OpenAI API credit to run a Zendo task. For Zendo, one needs to run 50 tasks—10 Zendo tasks on 5 different seeds—to get the performance numbers of a method like we reported. The actual cost, however, could be lower than calculated since one can cache LLM responses. 

## **A.9 Human study on ‘majority is red’ rule details** 

20 participants from our academic department were recruited via Slack to attempt the rule "the majority of blocks are red". The participants are compensated $10-$20 depending on their performance ($10 base rate + $1.25 bonus for each correct test scence prediction – there are 8 test scenes). Figure 9 shows the web interface displayed to participants. The full instructions given to human participants are displayed at Figures 12 and 13. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0019-00.png)


Figure 9: Example of the web interface shown to participants. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0019-02.png)


Figure 10: First figure for human participants instructions shown at Figure 12 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0020-00.png)


Figure 11: Second figure for human participants instructions shown at Figure 12 

20 

## `Thank you for playing my game!` 

```
Inthisgameyouwillbelearningaboutanalienplanet.ThisplanetiscalledZorb.
```

```
OnZorb,therearethesespecialblocksthatlooklikethis:
```

## `{Figure 9}` 

```
Thesespecialblocksmaylookthesame,buttherearemanydifferentkindsof
specialblocksontheplanetZorb.Theyallhavedifferentnames,andtheyall
workdifferently.
```

```
Sometimes,whentheblocksaresetupincertainways,starswillshootoutofthem!
Everykindofspecialblockhasadifferentruleformakingstarsshootoutofthem.
```

```
Yourjobistofigureouteachruleforhowtomakestarscomeoutofallofthe
differentkindsofspecialblocks!
```

## `{Figure 10}` 

```
Sotherearealotofthingsthatmightmakestarscomeoutofcertainspecial
blocks!
```

```
Youmightgetstarsfromblocksofdifferentnumbers,fromblocksofdifferent
colors,fromblocksofdifferentsizes,fromblocksfacingdifferentdirections,and
more!
```

```
Wefoundoutthatthereare2morekindsofspecialblocksonZorb!Let’scallthem
’Bemmies’and’Yoks’.Again,youdonothavetomemorizethenames--wejustwant
toemphasizethatdifferentkindsofblocksworkunderdifferentrules
```

```
Butwedon’tknowtheruleforsettingupeachkindofspecialblockssostarswill
comeoutofthem.Yourjobistofigureoutthetwodifferentrulesforhowtoset
upeachdifferentkindofspecialblockssostarscomeout!
```

```
Nowwe’regoingtowatchavideo.Thisvideoisgoingtoshowyouhowyoucanmove
thespecialblocksaroundyourself!
Youmustwatchthevideotocontinue.
```

```
So,intheinterfaceyoucan:
```

```
Pressbuttonsatthebottomtoaddblocks
```

```
Movetheblocksaroundbypickingthemupwiththemouse(leftclickingand
holding)
```

```
Turnthemusingthe"Z"(counterclockwise)and"X"(clockwise)keys
Rightclickonthemtoremovethem(command+clickifyouareusingmac
trackpad)
```

```
Whenyou’redonemovingthespecialblocks,you’regoingtotestthemtoseeif
starswillcomeoutofthem.Ifyousetthemupintherightwayaccordingtothe
rule,you’llseeabunchofstarsappear!Otherwise,nothingwillhappen.
```

Figure 12: (Part 1) Instructions for participants. Please find instruction figure 1 and 2 at Figures 10 and 11 

21 

```
Afteryoumovethespecialblocksaroundandtestthem,you’regoingtoseeifyou
canpickoutwhichpicturesoftheblocksyouthinkwillshootoutstars.Thisvideo
willshowyouhowtodothat:
```

## `{demo video}` 

```
Inthevideoyoucanseethatthisparticipantthinksthatfourofthepictures
showbemmiesthatstarswillcomeoutof(theonesmarkedingrey).Therightanswer
couldbeanywherebetweenoneandsevenofthepictures.
```

```
Adults:Youwillearnabonusof$1.25foreachofthepicturesinthemaintask
whereyouguesscorrectlywhetheritwillshootoutstars(demotaskperformance
doesnotcount).Thatmeans,ifyougetalleightpicturescorrectinthemaintask,
youwillearnabonusof$10!
```

```
Youmustwatchthevideotocontinue.
```

## `{demo video}` 

```
Finally,youmayguesstheruleforhowthiskindofspecialblocksworks.
```

```
Forexample,ifitlookslikestarsonlyshootoutoftheblocksifallofthemare
green,youwouldwritesomethinglike:"alltheblockshavetobegreen"!
```

```
Warning:YourresponseswillbecheckedbyahumanbeforeHITapproval.Nonsensical
orcopy-pastedanswerswillleadtoyourHITbeingrejected.Ifyoutrulyhaveno
ideasaboutarule,pleasejustwrite"Idonotknow".
```

## `Instructions Summary:` 

```
Youwilllookat2differentkindsofspecialblocks(includingonedemotask
forlearningthegame)thatwillshootoutstarsiftheyaresetupincertainways.
```

```
Youmustfigureouttheruleforhoweachkindofspecialblocksworks.
```

```
Youwillsetupthespecialblocksandtestthemtoseeifstarswillshootout
ofthemseventimesforeachtype.
```

```
Yourgoalistofigureoutwhichoutof8newpicturesofeachkindofspecial
blockswillshootoutstars($1.25bonusforeachcorrectinthemaintask)...
```

```
...andtowritedownyourbestguessoftheruleforthatkindofspecialblock!
```

Figure 13: (Part 2) Instructions for participants. 

22 

## **A.10 Prompts** 

The prompts used in all of our experiments can be found in Tables 7 to 12. 

For Zendo, we engineer the prompts for initial importance sampler _q_ ( _h|x, y_ ) for online inference so that they only output simple rules (see Table 7); this approach helps the proposer output hypotheses with higher priors, since our prior is defined by the number of words in the rule. We cannot apply this trick to batch inference because, unlike online inference, it does not evolve simpler rules into more complex ones. Additionally, we also design the importance sampler prompts to avoid proposing negative rules (‘there is no ...’) (see Table 7). We found that this leads to a more human-like behavior and also better performance. 

## **A.11 Supplemental results** 

See Table 6 and figs. 14 and 15 

23 

|Method|LogL|
|---|---|
|[13]’s best model|3085|
|Batch, Fuzzy|3352_._93|
|Online, Fuzzy|**2988****_._77**|
|Batch, Hard|5842_._00|
|Batch w/ Refnement, Hard|6999_._52|
|Online, Hard|10419_._86|



Table 6: BIC scores of models on human data. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0024-02.png)


**----- Start of picture text -----**<br>
LLM vs Human Batch, Hard Model vs Human Batch, Refine, Hard Model vs Human<br>1.0 R² = 0.13 1.0 R² = 0.00 1.0 R² = 0.00<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Average Model Predicted Probabilty Average Model Predicted Probabilty Average Model Predicted Probabilty<br>Online, Hard Model vs Human Batch, Fuzzy Model vs Human Online, Fuzzy Model vs Human<br>1.0 R² = 0.08 1.0 R² = 0.01 1.0 R² = 0.35<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0.2 0.4 0.6 0.8 1.0 0.4 0.6 0.8 0.4 0.5 0.6 0.7 0.8 0.9<br>Average Model Predicted Probabilty Average Model Predicted Probabilty Average Model Predicted Probabilty<br>Human Accuracy Human Accuracy Human Accuracy<br>Human Accuracy Human Accuracy Human Accuracy<br>**----- End of picture text -----**<br>


Figure 14: Comparing human and model prediction on each test scene after 7 rounds of experimentation. Each point is a prediction on a test scene. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning_images/nips_2024_Doing_Experiments_and_Revising_Rules_with_Natural_Language_and_Probabilistic_Reasoning.pdf-0024-04.png)


**----- Start of picture text -----**<br>
0.9 LLM Active Learning vs Human 0.9 Random Active Learning vs Human 0.9 Info Gain Active Learning vs Human<br>R² = 0.08 R² = 0.30 R² = 0.57<br>0.8 0.8 0.8<br>0.7 0.7 0.7<br>0.6<br>0.6 0.6<br>0.5<br>0.5 0.5<br>0.4<br>0.4 0.4<br>0.3 0.4 0.5 0.6 0.7 0.8 0.4 0.5 0.6 0.7 0.8 0.5 0.6 0.7 0.8<br>Average Model Predicted Probabilty Average Model Predicted Probabilty Average Model Predicted Probabilty<br>Human Accuracy Human Accuracy Human Accuracy<br>**----- End of picture text -----**<br>


Figure 15: Human vs online, fuzzy model accuracy binned by 4 rule-following (RF) and 4 not rulefollowing (Not RF) test scenes. This figure shows online, fuzzy model with three different active learning methods: LLM, Random, and InfoGain 

24 

|Method|Prompts for Zendo|Prompts for ActiveACRE|
|---|---|---|
||`Given the following`||
||`structures described with`||
||`{att_summary} of blocks in`||
||`the structures:`||
||`{text_c}`|`A group of objects may`|
||`Please list {num} possible`|`make the "blicket machine"`|
||`rules about the attributes`|`have lights turned on`|
||`in a structure that`|`or off depending on the`|
||`differentiate the good`|`objects in it. We seek`|
||`structures from the bad`|`to figure out the rule`|
||`structures.`|`underlying this. Consider`|
|Batch Inference|`Keep in mind that`|`the following:`|
||`1. All bad structures must`||
||`violate the rules.`|`{text_c}`|
||`2. Orders of blocks in a`||
||`structure do NOT matter.`|`Please state {num}`|
||`3. Do NOT propose`|`possible rules what makes`|
||`"negative rules" such as`|`the light turned on. State`|
||`"there is no green block".`|`them in a listed number.`|
||`4. The rules are short,`|`Do not explain.`|
||`concise, single sentences.`||
||`Please number them from`||
||`1-{num} and do not say`||
||`anything else`||
||`Please list {num} possible`||
||`rules about the {att}.`||
|||`A group of objects may`|
||`Example 1:`|`make the "blicket machine"`|
||`Structure: blue, blue`|`have lights turned on`|
||`Simple rules (Orders do`|`or off depending on the`|
||`NOT matter):`|`objects in it. We seek`|
||`1. There is a blue block`|`to figure out the rule`|
||`2. All blocks are blue`|`underlying this. Consider`|
|Online Inference|`Do NOT propose "negative`|`the following:`|
||`rules" such as "there`||
||`is no green block". Do`|`{text_c}`|
||`NOT propose rules with`||
||`quantifier such as "there`|`Please state {num}`|
||`are two blue blocks"`|`possible rules what makes`|
|||`the light turned on. State`|
||`Task 1:`|`them in a listed number.`|
||`{x}`|`Do not explain.`|
||`Simple rules (Orders do`||
||`NOT matter):`||



Table 7: Prompts used for the importance sampler _q_ ( _h|x_ 1 _, y_ 1) of all methods 

25 

Prompts for Zendo Prompts for ActiveACRE `An object contains the following attributes: color (gray/red/blue/green/\ brown/cyan/purple/yellow) A structure has one or more blocks. material (metal/rubber) Each block should contain the shape(cube/sphere/cylinder) following attributes: {att_par} Example of rule modifications: Additional conjunction: ’The light Example of rule modifications: turns on when there is a cylinder Quantifier change: ’There must be a present’ -> ’The light turns on green block’ -> ’There are two green when there is a cylinder and a cube blocks’ present’ Additional attribute: ’There must be Additional disjunction: ’The light a green block’ -> ’There must be a turns on when there is a cylinder green block that is upright’ present’ -> ’The light turns on Attribute change: ’There must be a when there is a cylinder or a cube green block’ -> ’There must be a blue present’ block’ Additional attribute: ’The light These modifications are "local": only turns on when there is a cylinder one attribute/quantifier is changed present’ -> ’The light turns on when or added for each modification. there is a blue cylinder present’ These modifications are "local": only Please modify the rule ’{h}’. one disjunction/conjunction/attribute Generate {num} rules for each is changed or added for each of modification (Quantifier modification.` 

```
Pleasemodifytherule’{h}’.onedisjunction/conjunction/attribute
Generate{num}rulesforeachischangedoraddedforeach
typeofmodification(Quantifiermodification.
change,Additionalattribute,
Attributechange)sothatthePleasemodifytherule’{h}’.
followingstructureis{text_y}aGenerate{num}rulesforeachtypeof
goodstructure:modification(Additionalconjunction,
{x}Additionaldisjunction,Additional
Notethatthenumberoftheblocksdoattribute)sothatthelightdoes
notmatter.{text_y}turnonwhenthefollowing
objectsarepresent:
Maketheformatanumberedlist{x}
(1.,2.,...,15.)RememberthatNotethatthenumberoftheblocksdo
thenewrulesshouldbea"local"notmatter.
modificationfromtherule’{h}’.Do
notuseattributevaluesthatareMaketheformatanumberedlist
notmentionedearlier.Donotsay(1.,2.,...,15.)Rememberthat
anythingotherthanthemodifiedthenewrulesshouldbea"local"
rules.modificationfromtherule’{h}’.Do
notuseattributevaluesthatare
notmentionedearlier.Donotsay
anythingotherthanthemodified
rules.
```

Table 8: Prompts used for the forward kernel _q_ ( _h[′] |Ht, x_ 1: _t, y_ 1: _t_ ) of online inference methods 

26 

Prompts for Zendo Prompts for ActiveACRE `Given the rule ’{h}’, please give one Given the rule ’{h}’, please give one group of objects that makes the light structure that conforms with the rule turned on and another that makes the and another structure that violates light turned off with the rule. The list of available of objects are A structure has one of more blocks. {all_objects}. Each block should contain the following attributes: The format of your answer should as {spec}{stacking_note} follows: The format of each structure should light on group of objects: obj_1, be as follows: obj_2, ... (conforms with the rule) Structure 1: {example_block} light off group of objects: obj_1, obj_2, ... (violates the rule) Structure 2: {example_block} All objects in a group must be unique. Do not say anything else.` 

Table 9: Prompts used for experiment proposers. 

27 

Prompts for Zendo 

```
Pleasesynthesizeapythonprogram
thatimplementstherule’{h}’
```

```
Theprogramshouldtakesina
ZendoStructurewhichrepresentsa
structureandreturnsTrueifit’sa
goodstructureandFalseotherwise.
```

```
Thedocstringsfortheclassesareas
follow:
```

```
classZendoStructure:
```

```
:paramblocks:listofZendoBlock
classZendoBlock:
:paramcolor:str
(blue/red/green)
:paramsize:str
(small/medium/large)
:paramorientation:str
(upright/left/right/strange)
{groundedness_param_msg}
:paramtouching:listofint
(indexstartsat1)
```

```
Thesignatureforthesynthesized
programshouldbe
defrule(structure:ZendoStructure)
->bool
```

```
Onlyoutputthe’rule’function.Do
notincludeanythingelse.
```

Prompts for ActiveACRE 

```
Pleasesynthesizeapythonprogram
thatimplementstherule’{h}’
```

```
Theprogramshouldtakesina
ACREGroupwhichrepresentsagroup
ofobjectsandreturnsTrueifit’sa
goodgroupandFalseotherwise.
```

```
Thedocstringsfortheclassesareas
follow:
```

```
classACREGroup:
```

```
:paramobjs:listofACREObject
classACREObject:
:paramcolor:str(gray,red,
blue,green,brown,cyan,purple,
yellow)
:parammaterial:str(metal,
rubber)
:paramshape:str(cube,sphere,
cylinder)
```

```
Thesignatureforthesynthesized
programshouldbe
defrule(group:ACREGroup)->bool
Onlyoutputthe’rule’function.Do
notincludeanythingelse.
```

Table 10: Prompts used to translate natural language _h_ to code. 

28 

Prompts for Zendo Prompts for ActiveACRE `A structure has one or more blocks. Each block should contain the following attributes: {att_par} An object contains the following attributes: Consider the following rule: ’{h}’ color (gray/red/blue/green/\ brown/cyan/purple/yellow) Given a structure, the output is yes material (metal/rubber) if it follows the rule (or is a good shape(cube/sphere/cylinder)` 

```
Givenastructure,theoutputisyes
ifitfollowstherule(orisagood
structure)andnoifitdoesnot(or
isabadstructure)
```

```
Considerthefollowingrule:’{h}’
Thegivenrulegivesincorrectoutput
forthefollowinggroupsofobjects:
```

```
Thegivenrulegivesincorrectoutput
forthefollowingstructures:
```

```
{feedback}
```

```
{feedback}
```

```
Basedonthegivenrule,generate
{num}newrefinedrulesthatfixthe
outputsforallmentionedstructures.
Thenewrulesmayinvolveanyofthe
mentionedattributes(color,size,
orientation,grounded,touching).
Pleasenumberthemfrom1-{num}and
donotsayanythingelse
```

```
Basedonthegivenrule,generate
{num}newrefinedrulesthatfixthe
outputsforallmentionedstructures.
Pleasenumberthemfrom1-{num}and
donotsayanythingelse
```

Table 11: Prompts used to perform refinement in batch inference with refinement 

29 

||Prompts for Zendo|Prompts for ActiveACRE|
|---|---|---|
|||`You are playing an`|
|||`inductive game with me.`|
||`You are playing an`<br>`inductive game with me.`<br>`I’ll be the moderator,`<br>`and your task is to figure`<br>`out the secret rule that`<br>`I know by coming up with`<br>`a structure of blocks to`<br>`ask me whether it conforms`|`I’ll be the moderator,`<br>`and your task is to figure`<br>`out the secret rule that I`<br>`know by coming up with a`<br>`group of blocks to ask me`<br>`whether the group conforms`<br>`with the secret rule or`<br>`not.`|
||`with the secret rule or`<br>`not.`|`An object contains the`<br>`following attributes:`|
|Initial prompt|`The structure has one`<br>`of more blocks. Each`<br>`block should contain the`<br>`following attributes:`<br>`{att_par}`|`color`<br>`(gray/red/blue/green/\`<br>`brown/cyan/purple/yellow)`<br>`material (metal/rubber)`<br>`shape(cube/sphere/cylinder)`<br>`The list of available of`|
||`To give you a start, I’ll`|`objects are {all_objects}.`|
||`describe one structure`<br>`that follows the rule:`|`To give you a start, I’ll`<br>`describe one group of`|
||`{text_c}`|`objects that follows the`<br>`rule:`|
||`Give a very short summary`<br>`on what you currently`|`{text_c}`|
||`think the secret rule is.`|`Give a very short summary`|
|||`on what you currently`|
|||`think the secret rule is.`|
||`The verdict on whether the`|`The verdict on whether the`|
||`queried structure follows`|`queried structure follows`|
|Follow-up prompt|`the rule is {verdict}.`<br>`Give a very short summary`|`the rule is {verdict}.`<br>`Give a very short summary`|
||`on what you currently`|`on what you currently`|
||`think the secret rule is.`|`think the secret rule is.`|
||`Give one structure you`|`Give one structure you`|
||`want to test whether it`|`want to test whether it`|
|Active learning prompt|`follows the secret rule`<br>`or not.`<br>`Do not include`|`follows the secret rule`<br>`or not.`<br>`Do not include`|
||`anything other than the`|`anything other than the`|
||`structure.`|`structure.`|
||`Now, do you think this`|`Now, do you think this`|
||`structure follow the`|`group of objects follow`|
||`rule?\n: {x}\nAnswer only`|`the rule?\n: {x}\nAnswer`|
|Prediction prompt|`yes or no. Give your best`<br>`guess even if you are`|`only yes or no. Give your`<br>`best guess even if you are`|
||`uncertain. Do not explain.`|`uncertain. Do not explain.`|
||`Just say yes or no`|`Just say yes or no’`|



Table 12: Prompts used for vanilla, direct LLM method 

30 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: The claims we made in abstract and introduction accurately reflect the paper’s contributions and scope. 

Guidelines: 

   - The answer NA means that the abstract and introduction do not include the claims made in the paper. 

   - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

   - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

   - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: The last section discusses limitations. 

Guidelines: 

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

- The authors are encouraged to create a separate "Limitations" section in their paper. 

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

## 3. **Theory Assumptions and Proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [Yes] 

31 

Justification: We provide the formal proof of our proposition in the appendix. Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] 

Justification: We have fully described the algorithms used in the paper, with details included in the appendix. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

## 5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

32 

## Answer: [Yes] 

Justification: Code and data available at `https://github.com/topwasu/ doing-experiments-and-revising-rules/` Guidelines: 

- The answer NA means that paper does not include experiments requiring code. 

- Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/ public/guides/CodeSubmissionPolicy` ) for more details. 

- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( `https: //nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. **Experimental Setting/Details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: Algorithm details are described in the appendix. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: The paper reports error bars – most of them computed over 5 seeds. Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

- The assumptions made should be given (e.g., Normally distributed errors). 

33 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments Compute Resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: We have provided estimated OpenAI API cost for our experiments. Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: The research conducted in the paper conforms, in every aspect, with the NeurIPS Code of Ethics. 

Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [NA] 

Justification: Our work is foundational research and has no direct societal impact. 

Guidelines: 

- The answer NA means that there is no societal impact of the work performed. 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

34 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: The paper poses no such risks. 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: The creators of original data are all properly credited, and the license and terms of use of the data are explicitly mentioned and properly respected. 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

35 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: The code and data introduced in the paper is well documented. 

Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [Yes] 

Justification: We have described our human study in details in the appendix. 

Guidelines: 

   - The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

   - Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

   - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [Yes] 

Justification: We did not anticipate any risks from participating in our study. IRB approval was obtained. 

Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

36 

