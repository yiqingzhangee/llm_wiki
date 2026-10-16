# **Self-Evaluation Guided Beam Search for Reasoning** 

**Yuxi Xie**[1] _**[∗]**_ **Kenji Kawaguchi**[1] **Yiran Zhao**[1] **James Xu Zhao**[1] **Min-Yen Kan**[1] _[†]_ **Junxian He**[2] _[†]_ **Michael Qizhe Xie**[1] _[†]_ 1 National University of Singapore 2 The Hong Kong University of Science and Technology 

## **Abstract** 

Breaking down a problem into intermediate steps has demonstrated impressive performance in Large Language Model (LLM) reasoning. However, the growth of the reasoning chain introduces uncertainty and error accumulation, making it challenging to elicit accurate final results. To tackle this challenge of uncertainty in multi-step reasoning, we introduce a stepwise self-evaluation mechanism to guide and calibrate the reasoning process of LLMs. We propose a decoding algorithm integrating the self-evaluation guidance via stochastic beam search. The selfevaluation guidance serves as a better-calibrated automatic criterion, facilitating an efficient search in the reasoning space and resulting in superior prediction quality. Stochastic beam search balances exploitation and exploration of the search space with temperature-controlled randomness. Our approach surpasses the corresponding Codex-backboned baselines in few-shot accuracy by 6 _._ 34%, 9 _._ 56%, and 5 _._ 46% on the GSM8K, AQuA, and StrategyQA benchmarks, respectively. Experiment results with Llama-2 on arithmetic reasoning demonstrate the efficiency of our method in outperforming the baseline methods with comparable computational budgets. Further analysis in multi-step reasoning finds our self-evaluation guidance pinpoints logic failures and leads to higher consistency and robustness. Our code is publicly available at https://guideddecoding.github.io/. 

## **1 Introduction** 

The remarkable empirical achievements of Large Language Models (LLMs) have recently ushered in a new era in machine reasoning through few-shot prompting techniques (Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2023a; OpenAI, 2023). In particular, breaking down a problem into intermediate stages, or a reasoning chain, can significantly improve model performance on reasoning tasks (Cobbe et al., 2021). Various prompting approaches have been proposed to define these chains, such as _scratchpads_ (Nye et al., 2021), _chain-of-thought_ (CoT) (Wei et al., 2022b), _least-to-most_ (Zhou et al., 2023), and _program-aided language models_ (PAL) (Gao et al., 2023; Chen et al., 2022). However, as the complexity and length of reasoning chains increase with the difficulty of tasks, LLMs struggle with errors and imperfections that accumulate across multiple intermediate steps (Wu et al., 2016; Guo et al., 2018; Chen et al., 2022). Furthermore, the growing number of steps leads to an exponential growth in the search space for reasoning, making it exceedingly difficult to obtain accurate final outcomes. 

Confronted with the challenges of uncertainty in multi-step chaining, several previous studies have worked on different aspects to alleviate the impact of reasoning errors. For instance, Wang et al. (2023) introduce _self-consistency_ as a method to determine the final answer through majority voting using multiple sampled reasoning paths, while Li et al. (2022) investigate various prompts to diversify the sampling outcomes. Gao et al. (2023) and Chen et al. (2022) utilize Python programs to 

> _∗_ Correspondence to: Yuxi Xie (xieyuxi@u.nus.edu). 

> _†_ Equal advising. Ordering is determined by dice rolling. 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Multi-step Reasoning Chain<br>Question Basic Greedy Decoding marilyn_copies = 88000 ratio = 10<br>Marilyn's 1 [st]  record sold  answer = marilyn_copies // ratio<br>10 times as many<br>copies as Harald's. If  _marilyn_copies = 88000_<br>they sold 88,000 copies  _copies_mar = total_sold / 10_ Answer : 8,800<br>combined, how many  _total_sold = 88000_<br>copies did Harald sell? _ratio = 10_ _answer = total_sold // marilyn_harald_ratio_<br>_marilyn_copies = 10_<br>_marilyn_harald_ratio = 10_ _answer = total_sold / (marilyn_harald_ratio + 1)_<br>_…_ _…_ _…_  total_sold = 88000 Multi-step Reasoning Chain<br>Self-Evaluation Guided Stochastic Beam Search  marilyn_harald_ratio = 10<br> answer = total_sold / (marilyn_harald_ratio + 1)<br>Generation<br>LLM Evaluation Answer : 8,000<br>**----- End of picture text -----**<br>


Figure 1: Self-Evaluation can calibrate the decoding direction in multi-step reasoning. We illustrate our method in the form of stepwise stochastic beam search with the beam size equal to 1. The scale of the self-evaluation score is visualized in the colormap. We adopt Program-Aided Language models (PAL) reasoning (Gao et al., 2023; Chen et al., 2022) for this math word problem. 

achieve higher accuracy in mathematical computations. While these approaches have contributed to significant performance improvements in reasoning, the process of generating reasoning chains has been parameterized as a standard autoregressive process and intrinsically faces the challenge of sampling within an exponentially large search space. 

Motivated by this challenge, we employ LLM _self-evaluation_ (Kadavath et al., 2022) as a bettercalibrated criterion to automatically guide the search in the reasoning space, drawing inspiration from prior works on utilizing LLMs for self-evaluation (Rae et al., 2021; Paul et al., 2023; Madaan et al., 2023; Shinn et al., 2023). We integrate the self-evaluation guidance for reasoning in a stepwise and generalizable manner. Specifically, we formulate the reasoning chain generation as a decoding process consisting of multiple intermediate steps. Unlike traditional text decoding where each step produces a single token, we consider each decoding step as a reasoning logic composed of a sequence of tokens. This framework enables us to employ beam search (Jurafsky and Martin, 2009; Graves, 2012) decoding tailored for intermediate steps and guide the beam searching process by controlling the error of each reasoning step to prevent potential error accumulation throughout the chaining. Figure 1 illustrates an example of decoding a chain of program-aided reasoning steps. Furthermore, we incorporate temperature-controlled randomness (Ackley et al., 1985; Kool et al., 2019; Meister et al., 2021) into the traditional (deterministic) beam search to balance the quality–diversity trade-off in searching for better reasoning chains. Our approach has resulted in respectable improvements across various arithmetic, symbolic, and commonsense reasoning tasks. For instance, by guiding the reasoning decoding process of the Codex model (Chen et al., 2021), we achieve accuracies of 85 _._ 5%, 64 _._ 2%, and 77 _._ 2% on the GSM8K, AQuA, and StrategyQA benchmarks, compared to the vanilla reasoning-enhanced Codex performance of 80 _._ 4%, 58 _._ 6%, and 73 _._ 2%, respectively. Our further analysis on Llama-2 (Touvron et al., 2023b) demonstrates the efficiency of our method in surpassing the self-consistency baseline under equivalent computational budgets. 

## **2 Self-Evaluation Guided Stochastic Beam Search** 

Considering the input prompt and question _Q_ represented as _x_ , we formulate the answer distribution _P_ ( _a | x_ ) by decomposing it as a reasoning chain generation process _P_ ( _R | x_ ) and an answer generation process _P_ ( _a | R, x_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0002-06.png)


where _R_ is the intermediate reasoning chain variable that is typically a text sequence. _P_ ( _a | R, x_ ) = 1 _A_ ( _a_ ) max ( _|A|,_ 1)[, where] _[ A]_[=] `[execute]`[(] _[R]_[)][ represents the set of predicted answer(s) interpreted from] _[ R]_[,] and 1 _A_ is the indicator function of the subset _A_ . In practice, _|A| ≥_ 0 can be 0 or larger than 1 when the reasoning _R_ returns no valid answer or produces more than one possible answers, respectively. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Timestep 1 Timestep 2 Timestep  T Prompting Framework<br>s [2] 1 s [T] 1 Reasoning Chain<br>s [1] 1 s [T] 2 s [1] s [2] … s [T] R Q<br>s [1] 2 s [2] n … R  = [ s [1] ,  s [2] , …,  s [T] ] =  s [1:] [T] LLM<br>Q ss [1][1] nn -1 ss [2][2] nn +1+2 … ss [T][T] nn +1 Predicted Final Answer a Gen s [t]<br>Multi-step Reasoning s [2] 2 n s [T] 2 n marilyn = total_sold // maril…  yn_harald_ratio<br>s [t] 1 Self-Eval # Is the above step of reasoning:<br>Kept Candidate  Q s [1] …  1 s [t] [-1] s [t] n -1 P LM …( s [1:] [t] 1) C (… s [1:] [t] 1) Beam SearchStochastic( k = 2 ) # # ((AB)) Correct Incorrect<br>Kept Candidate 2 ss [t] n [t] n +1 PPP LMLMLM (( s ( ss [1:][1:][1:] [t][t] nn [t] n +1-1))) CCC (( s ( ss [1:][1:][1:] [t][t] nn [t] n +1-1))) Select  k  Paths # The above step of reasonin(B), because Marilyn sold 10 times as many …g is:<br>… …<br>Q s [1] …  s [t] [-1]<br>s [t] 2 n -1 P LM ( s [1:] [t] 2 n -1) C ( s [1:] [t] 2 n -1) Self-Eval<br>Decoding at Timestep  t s [t] 2 n P LM ( s [1:] [t] 2 n ) C ( s [1:] [t] 2 n )<br>…<br>…<br>…<br>… …<br>…<br>Gen<br>Gen …<br>**----- End of picture text -----**<br>


Figure 2: Our framework of self-evaluation guided stochastic beam search for multi-step reasoning. The schema of the decoding process is on the left, where we keep _k_ = 2 candidates at each timestep, with the detailed illustration of timestep _t_ at the bottom. Here “Gen” and “Self-Eval” represent the generation and evaluation LLMs, respectively. The corresponding prompt formulations are provided on the right, where the questions _Q_ , reasoning steps _R_ , and evaluation scripts are highlighted in orange , green , and yellow , respectively. Steps in light green ( _e.g._ , _s[t]_ ) are for models to generate or evaluate at the current timestep. Specifically, we follow Kadavath et al. (2022) to prompt the LLM evaluation by answering the multiple-choice question, _i.e._ , the lines starting with `#` . 

Prior research has modeled the reasoning chain generation _P_ ( _R | x_ ) by prompting LLMs to explicitly elaborate on the required intermediate steps _R_ . Through setting different prompting schemes, the reasoning process _P_ ( _R | x_ ) can be modeled as chain-of-thought free-text reasoning (Kojima et al., 2022; Wei et al., 2022b), a two-stage question decomposition and answering pipeline (Zhou et al., 2023), or program-aided reasoning to generate a python program (Gao et al., 2023; Chen et al., 2022). While effective, previous work mostly uses a single sample of _R_ from the LLMs to approximate the expectation in Eq. 1 – the generated reasoning chain is often unreliable and causes incorrect answers. To mitigate this issue, Wang et al. (2023) conduct majority voting to approximate the expectation via sampling and aggregating multiple reasoning chains. Li et al. (2022) take a further step to diversify the sampling and calibrate _P_ ( _R | x_ ) with a task-specific fine-tuned verifier. Another line of work focuses on improving _P_ ( _a | R, x_ ) instead. For example, Gao et al. (2023) and Chen et al. (2022) employ Python programs for more accurate calculations in math word problems. 

In this work, we focus on improving _P_ ( _R | x_ ) to enhance the consistency of the sampled reasoning chains. To this end, we propose to explicitly break down the reasoning process into multiple steps, as shown in Figure 2, where each step yields a semantically integrated sequence of tokens, representing a single step within the overall reasoning chain. From this perspective, we can approach the task of enhancing _P_ ( _R | x_ ) as a decoding problem over the reasoning chains. Considering the exponentially large search space and the potential unreliability of LLM-produced chains in reasoning, we propose a constrained stochastic beam search decoding approach to improve the reasoning step by step and obtain high-quality reasoning with a limited number of samples. We detail our approach next. 

## **2.1 Multi-step Reasoning via Stochastic Beam Search** 

In multi-step reasoning, a reasoning chain of _T_ steps is sequentially generated through several timesteps as _R_ = [ _s_[1] _, s_[2] _, · · · , s[T]_ ] = _s_[1:] _[T]_ , where _s[t]_ represents a sequence of tokens as the _t_ -th step. Formally, the reasoning generation process _P_ ( _R | x_ ) can be factorized in an autoregressive manner: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0003-06.png)


which resembles the typical token-level autoregressive distribution of language models. Stepwise reasoning allows us to formulate the process as a step-by-step decoding problem, where we can utilize widely-used strategies such as beam search for the generation. Different from the typical text 

3 

decoding process where each step consists of a single token, here we view a sequence of reasoning tokens as a single step. One of the most severe issues in LLM-based reasoning is the potential unreliability and inaccuracy of each reasoning step generated by the model. Furthermore, errors from individual steps may accumulate throughout the reasoning chain, exacerbating the problem. To address the issue, we define a constraint function _C_ ( _s[t] , s_[1:] _[t][−]_[1] ) _∈_ [0 _,_ 1] within each reasoning step[3] that outputs the LLM confidence in the correctness of the reasoning sequence _s[t]_ based on the previous context _s_[1:] _[t][−]_[1] . Then, we present a constrained decoding approach that combines the language model probability and the correctness confidence as a new decoding objective function _E_ ( _s_[1:] _[T]_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0004-01.png)


where _P_ LM _G_ is the language model distribution[4] . _λ ∈_ [0 _,_ 1] is a weight hyperparameter to balance the LM score and the confidence score. We will detail the design of _C_ ( _s[t]_ ) in Section 2.2. Eq 3 follows an autoregressive factorization form, and thus traditional token-level decoding methods such as beam search can be applied here on the chain level. As it is desirable to obtain high-quality reasoning chains with limited samples that are scored high by _E_ ( _s_[1:] _[T]_ ), it is natural to utilize greedy or beam search decoding to approximate the reasoning sequences that maximize _E_ ( _s_[1:] _[T]_ ). 

Additionally, multiple diverse reasoning chains could be aggregated to further improve the final accuracy, as suggested by Eq 1 and empirically confirmed by self-consistency reasoning (Wang et al., 2023). To this end, we propose a variant of stochastic beam search (Kool et al., 2019; Meister et al., 2021) to strike a tradeoff between exploration and exploitation. Concretely, for beam size _k_ , at each reasoning step we draw _n_ samples of _s[t]_ following _P_ LM _G_ ( _s[t] | x, s_[1:] _[t][−]_[1] ) for each beam, and we end up with _nk_ chain hypotheses of _s_[1:] _[t]_ to form the candidate set _S_ , then we perform beam pruning through sampling – we sample _k_ reasoning beams without replacement, rather than finding the arg max _k_ , following a distribution defined by the accumulated score:[5] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0004-04.png)


where the temperature _τ_ is a hyperparameter to control the randomness in stochastic beam search; when _τ →_ 0, stochastic beam search becomes the vanilla beam search algorithm. The reasoning beams _s_[1:] _[t]_ can be sampled efficiently since _|S|_ = _nk_ is a finite set. To enable fine-grained control of sampling randomness in decoding, we also introduce a hyperparameter _α ∈_ [0 _,_ 1] so that _τ_ can decay step by step as _τ → ατ_ . By annealing _τ_ with _α_ , we can mitigate the error accumulation due to aggregated randomness throughout chaining, as discussed in Section 3.4. 

By incorporating controllable randomness, we not only achieve a more reliable single reasoning chain generation by setting randomness to be small, but also leverage multiple diverse reasoning chains with larger variance. Next, we introduce our constraint function _C_ ( _s[t] , s_[1:] _[t][−]_[1] ) that utilizes a self-evaluation scheme to improve the consistency of each reasoning step. 

## **2.2 Self-Evaluation as Correctness Control** 

Inspired by the recent success of self-evaluation (Kadavath et al., 2022; Shinn et al., 2023; Madaan et al., 2023; Paul et al., 2023), a scheme to prompt LLMs to evaluate their own generation, we use LLMs to judge the correctness of _s[t]_ based on _s_[1:] _[t][−]_[1] . Specifically, the evaluation and generation models use the same backend LLM with different prompts, which consist of few-shot exemplars. We follow previous works of CoT (Wei et al., 2022b) or PAL (Gao et al., 2023) to formulate the generation prompts. To construct the in-context exemplars prompt _C_ for the self-evaluation LLM LM _C_ , we provide stepwise evaluation examples (as question answering with rationales) in each instance. Inspired by Kadavath et al. (2022), we design prompt _C_ in the form of multiple-choice questioning (as shown in Figure 2) to better calibrate the model predictions, where we adopt the token-level probability of option A to represent the correctness score as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0004-09.png)


> 3 _t_ For ease of notation, we will use _C_ ( _s_ ) throughout the paper when there is no confusion. 

> 4We will denote the LM generation probability by _P_ throughout the paper for simplification. 

> 5In Appendix A.1, we justify the approximation error rate of Eq 4, which computes normalized probability on the subset _S_ instead of on the entire set. 

4 

## **3 Experiments** 

## **3.1 Setup** 

We present and analyze the results of our self-evaluation guided beam search with different LLM backbones on various reasoning benchmarks. Implementation details including prompt examples and hyperparameter setup can be found in Appendix A.3. 

**Benchmarks.** We evaluate the effectiveness of our approach across three types of reasoning tasks: (1) Arithmetic Reasoning on five math word problem benchmarks, including GSM8K (Cobbe et al., 2021) on math word problems, AQuA (Ling et al., 2017) on algebraic word problems, SVAMP (Patel et al., 2021) on structure variations of math word problems, ASDiv (Miao et al., 2020) on diverse math word problems, and TabMWP (Lu et al., 2023) on tabular math word problems; (2) Symbolic Reasoning on BIG-Bench (Srivastava et al., 2022), which involves `Date Understanding` of contextbased date inferring and `Object Counting` of enumerating and counting objects of different types; (3) Commonsense Reasoning on three benchmarks, including CommonsenseQA (Talmor et al., 2019) of commonsense questions that require prior world knowledge to answer, StrategyQA (Geva et al., 2021) of questions that require a multi-hop strategy to answer, and `Sports Understanding` from BIG-Bench (Srivastava et al., 2022) to determine whether a sports-related sentence is plausible. 

**Baselines.** We consider two types of baselines: (1) Chain-of-Thought (CoT) (Wei et al., 2022b) prompting in free-text reasoning and (2) Program-Aided Language models (PAL) (Ling et al., 2017) and Program-of-Thought (PoT) (Chen et al., 2022) prompting in program-aided reasoning. We also include their self-consistency (Wang et al., 2023) variants for multiple-chain reasoning. For generation, we follow the few-shot exemplars of baselines. For self-evaluation, we manually create a set of few-shot exemplars based on the baseline outputs on corresponding training data. We formulate self-evaluation as a task of multiple-choice question answering, following Kadavath et al. (2022). For baselines, we represent the cost as the number of generated tokens. For the cost of our method, we also include the number of additional input tokens in self-evaluation for a fair comparison. 

**Backboned LLMs.** We assess our approach on closed- and open-source LLMs using both PAL and CoT prompting. For closed-source LLMs, we choose Codex ( `code-davinci-002` ) (Chen et al., 2021) to report and compare the results on all datasets. We use Llama-2 ( `13B` ) (Touvron et al., 2023b) as our open-source LLM to conduct cost–performance analysis on different datasets. 

## **3.2 Main Results** 

**Arithmetic and Symbolic Reasoning.** Table 1 shows the results for arithmetic and symbolic reasoning. Our method achieves significant performance improvements on most benchmarks in both single- ( _τ_ = 0) and multiple-chain scenarios, with PAL as the baseline. For arithmetic reasoning, we observe absolute increases in accuracy of 5 _._ 3%, 8 _._ 3%, and 0 _._ 7% on GSM8K, AQuA, and SVAMP, respectively. One possible explanation for this discrepancy in improvements is the reduced diversity in LLM generations due to higher confidence in predictions, as evidenced by the relatively high performance on the tasks. This highlights the importance of incorporating controllable randomness into the candidate generations to expand the search space for self-evaluation guided decoding. We further explore the impact of generation diversity by varying the temperature _γ_ in Section 3.4. 

For symbolic reasoning, our approach also leads to consistent performance gains. However, when the baseline itself performs well on the task ( _e.g._ , 96 _._ 7% on `Object Counting` ), our approach may not yield substantial improvement. This can also be attributed to the constrained accessible search space for self-evaluation guidance to refine the generation distribution. This limit suggests the inherent deficiency in our LLM-based prompting method that it becomes increasingly challenging to calibrate the generation direction when the model LM _G_ is more confident in its predictions. In other words, the high baseline performance usually indicates lower diversity in the LLM generations even with a large temperature _γ_ , resulting in a limited accessible search space for the model to find a better solution. 

**Commonsense Reasoning.** Table 2 compares methods using CoT prompting in commonsense reasoning. Our approach shows consistent performance improvements across several tasks. For example, on StrategyQA, we achieve an accuracy of 77 _._ 2% compared with 73 _._ 2% of the baseline. 

5 

Table 1: Result comparison (accuracy %) on arithmetic and symbolic reasoning tasks. The best result is in **bold** and the lowest cost is in green. We report methods all with Codex backbone for a fair comparison. Similar to Huang et al. (2022), Diverse (Li et al., 2022) fine-tune task-specific verifiers to apply weights on samples in self-consistency (SC). Other fine-tuning methods include reward-based supervision (Uesato et al., 2022) and content-specific training (Lewkowycz et al., 2022). We also report the number of tokens (# Tokens) on GSM8K to compare the costs of different methods. 

|Approach|GSM8K|# Tokens|Arithmetic<br>AQuA<br>SVAMP|Arithmetic<br>AQuA<br>SVAMP|ASDiv|TabMWP|Symbolic<br>`DATE`<br>`OBJECT`|Symbolic<br>`DATE`<br>`OBJECT`|
|---|---|---|---|---|---|---|---|---|
||||single reasoning chain||||||
|CoT|65_._6|0_._2_k_|45_._3|74_._8|76_._9|65_._2|64_._8|73_._0|
|PoT|71_._6|_−_|54_._1|85_._2|_−_|73_._2|_−_|_−_|
|PAL|72_._0|0_._3_k_|_−_|79_._4|79_._6|_−_|76_._2|96_._7|
|Ours-PAL|80_._2|27_._7_k_|55_._9|89_._6|84_._9|79_._1|**78**_._**6**|**96**_._**8**|
||||multiple reasoning chains||||||
|CoT, SC|78_._0|5_._3_k_|52_._0|86_._8|_−_|75_._4|_−_|_−_|
|CoT, Diverse|82_._3|_−_|_−_|87_._0|**88**_._**7**|_−_|_−_|_−_|
|PoT, SC|80_._0|_−_|58_._6|89_._1|_−_|**81**_._**8**|_−_|_−_|
|PAL, SC|80_._4|7_._4_k_|_−_|_−_|_−_|_−_|_−_|_−_|
|Ours-PAL|**85**_._**5**|550_._0_k_|**64**_._**2**|**90**_._**3**|85_._8|80_._9|_−_|_−_|



Table 2: Result comparison (accuracy %) on commonsense reasoning tasks, with Codex backbone. Here we only report results in the single reasoning chain scenario following Wei et al. (2022b). We report # Tokens on StrategyQA for cost comparison. 

|Approach|StrategyQA|# Tokens|CommonsenseQA|`Sports`|
|---|---|---|---|---|
|CoT|73_._2|0_._06_k_|77_._9|**98**_._**5**|
|Ours-CoT|**77**_._**2**|11_._6_k_|**78**_._**6**|98_._4|
|Human|87_._0|_−_|88_._9|_−_|



Likewise, the performance of our approach is constrained by the low diversity of LLM generations on `Sporting Understanding` , as we observe on `Object Counting` in symbolic reasoning. 

**Computational Cost Overhead.** Despite the fact that our approach achieves significant improvement on various benchmarks, we observe an overhead of computational cost compared with the corresponding baselines. For example, the single-chain version of our approach using PAL costs about 3 times more than the self-consistency baseline on GSM8K. As detailed in Appendix A.3, this is due to a relatively large hyperparameter – the number of rollouts per beam _n_ – which we set as 16 for better performance. To strike a balance between performance and cost and present a complete picture, we adopt _n_ = 2 and conduct cost–performance analysis on our approach in Section 3.3. 

## **3.3 Cost Analysis** 

Table 3 compares the baseline and our approach under comparable computational budgets (measured in # Tokens). Our method consistently outperforms self-consistency on the arithmetic reasoning tasks even when benchmarked for relatively less computational cost. For example, we achieve 46 _._ 1% on GSM8K with a cost of 12 _._ 6 _k_ tokens, compared with the accuracy of 41 _._ 8% of self-consistency which costs 13 _._ 9 _k_ tokens. Figure 4 further illustrates the cost-efficiency of our approach on GSM8K using different prompting methods under various levels of costs. Our approach significantly outperforms the corresponding equal-cost baseline especially when the computational budget increases, indicating the improvement in the performance upper bound brought by our method. 

However, our approach lags behind the CoT baseline on commonsense reasoning. This implies the limitation of our method when applied to shorter reasoning chains, _i.e._ , decreasing the number 

6 

Table 3: Cost (# Tokens) and result (accuracy %) comparison on arithmetic and commonsense reasoning tasks. We base our experiments on Llama-2 (13B) since Codex is not available. We show the results of the baseline and our method both in the multiple-chain scenario for a fair comparison. Here we use PAL and CoT prompting for arithmetic and commonsense reasoning, respectively. 

|Approach|GSM8K|Arithmetic (PAL)<br>AQuA<br>SVAMP<br>ASDiv|Arithmetic (PAL)<br>AQuA<br>SVAMP<br>ASDiv|Arithmetic (PAL)<br>AQuA<br>SVAMP<br>ASDiv|TabMWP|Commonsense (CoT)<br>StrategyQA<br>CommonsenseQA|Commonsense (CoT)<br>StrategyQA<br>CommonsenseQA|
|---|---|---|---|---|---|---|---|
|Baseline|41_._8|30_._7|71_._2|66_._2|43_._7|**71**_._**0**|**74**_._**4**|
|# Tokens|13_._9_k_|6_._6_k_|5_._9_k_|2_._7_k_|1_._9_k_|2_._7_k_|1_._2_k_|
|Ours|**46**_._**1**|**31**_._**5**|**74**_._**6**|**67**_._**7**|**49**_._**6**|70_._6|74_._0|
|# Tokens|12_._6_k_|6_._0_k_|5_._0_k_|2_._5_k_|1_._2_k_|2_._6_k_|1_._2_k_|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0007-02.png)


**----- Start of picture text -----**<br>
46.1 45.4<br>46 45.2 44.9<br>44 44 43.3 43.0 43.2<br>42.9<br>42 41.7 41.8 42 40.8 42.9<br>40 40.538.9 40.4 40 38.8 40.6<br>38 PAL-SC 38 37.1 CoT-SC<br>36.1 Ours-PAL (single) 36.5 Ours-CoT (single)<br>36 35.0 35.7 Ours-PAL (multiple) 36 Ours-CoT (multiple)<br>34.6 34.5<br>34 34<br>2000 4000 6000 8000 10000 12000 14000 2000 4000 6000 8000 10000 12000<br>Cost (# Tokens) Cost (# Tokens)<br>(a) PAL Prompting Methods on GSM8K (b) CoT Prompting Methods on GSM8K<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 4: Accuracy curves on GSM8K of different methods with the change of the cost. We conduct the performance comparison using both PAL and CoT prompting with Llama-2 ( `13B` ) backbone. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0007-04.png)


**----- Start of picture text -----**<br>
1500 overall 0.83 0.88 overall 0.71 0.84 overall 0.76 0.86<br>correct correct correct<br>1000 wrong wrong wrong<br>500<br>0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>generation probability  correctness confidence  score  = (1 ), = 0.5<br>(a) Score distribution of PAL baseline predictions on GSM8K.<br>2000 overall correct 0.63 0.65 overall correct 0.70 0.75 overall correct 0.66 0.69<br>wrong wrong wrong<br>1000<br>0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>generation probability  correctness confidence  score  = (1 ), = 0.5<br>(b) Score distribution of CoT baseline predictions on StrategyQA.<br>**----- End of picture text -----**<br>


Figure 5: Distributions of the self-evaluation score and its components ( _i.e._ , generation confidence _P_ and correctness confidence _C_ ) on correct/incorrect baseline predictions. We highlight the median scores of the positive and negative cases using lines of the same colors respectively. 

of intermediate steps weakens the effect of stepwise self-evaluation in beam search in reducing error accumulation. On the other hand, self-consistency can directly improve performance through instance-level aggregation without additional cost for self-evaluation. We analyze how our method benefits longer reasoning chains on different tasks in Section 3.4. 

## **3.4 Further Analysis** 

We now provide a detailed analysis of why our method achieves significant gains. 

**Generation and Self-evaluation Calibration.** We investigate the distributions of generation confidence ( _i.e._ , the LM probability _P_ ) and correctness confidence _C_ in our self-evaluation score _E_ . By 

7 

Table 4: Absolute accuracy (in %) increases on instances of different complexity determined by the length of reasoning chains (represented as # Steps). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
GSM8K StrategyQA<br># Steps # Ins. PAL Ours ∆Accu. # Steps # Ins. CoT Ours ∆Accu.<br><  7 437 85 . 8 91 . 3 +5 . 49 <  4 637 84 . 6 84 . 9 +0 . 31<br>∈ (7 ,  9] 524 74 . 8 82 . 6 +7 . 82 ∈ [4 ,  5) 1 ,  301 78 . 6 79 . 1 +0 . 46<br>≥ 9 358 72 . 9 82 . 6 +9 . 78 ≥ 5 351 68 . 4 71 . 8 +3 . 42<br>42 Ours-PAL (single) 42.8 82 81.3<br>Ours-PAL (multiple) 84 84.5 84.8 80<br>40 PAL 83.3 78 77. 8<br>38 36.738.3 82 80.8 Ours-PAL (single) Ours-PAL (multiple) 76 Ours-CoT (single)Ours-CoT (multiple)<br>36 35.7 35.7 80 79.1 7 9.2 74<br>34 33.4 34.134.2 33.2 78 78.4 77.9 72 71. 8 71.4<br>32.6<br>70<br>2 4 6 8 10 0.2 0.5 0.8 1.0 0.0 0.5<br>Beam Size (k) LM temperature  ( = 0.5, = 0.5) Sampling temperature  ( = 0.5, = 0.5)<br>(a) effect of beam size (b) effect of generation and sampling diversity<br>Accuracy Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 6: Accuracy curves and distributions of our approach on GSM8K with different hyperparameter settings: (a) Changes in performance (Llama-2 backboned) when the beam size _k_ varies. Methods of the same _k_ have equal computational costs; (b) Accuracy distributions (Codex backboned) with different generation temperature _γ_ and sampling temperature _τ_ (with decay ratio _α_ ). 

comparing the score distributions for correct and wrong predictions, we aim to gain an intuitive understanding of whether these confidence scores are reliable. Figure 5 shows different score distributions on correct and wrong baseline predictions. The difference in distribution between the two prediction sets is substantial for arithmetic reasoning, but negligible for commonsense reasoning. Notably, in both instances, correctness confidence is more discriminatory than generation confidence. 

To achieve a balance between these two confidence scores, we utilize a tunable hyperparameter _λ_ , setting _λ_ = 0 _._ 5 for all datasets. Nevertheless, varying its value can lead to distinct outcomes. For instance, when setting _λ_ to 1 ( _E_ = _C_ ) or 0 ( _E_ = _P_ ), the performance on GSM8K decreases from 80 _._ 2% to 74 _._ 5% and 77 _._ 1%, respectively. This indicates that both scores play a crucial role in our final performance. A more comprehensive analysis of _λ_ can be found in Appendix A.2. 

**Reasoning Complexity.** We investigate if our approach is more beneficial for instances needing more reasoning steps. Table 4 shows that performance gains (in absolute accuracy % increase) increase as reasoning chains become longer on both GSM8K and StrategyQA. Notably, the improvement on StrategyQA primarily comes from improvements in longer reasoning chains, showcasing the effectiveness of our method in navigating lengthy and intricate reasoning chains. 

**Hyperparameters in Stochastic Beam Search.** We examine the significance of hyperparameters associated with stochastic beam search, including the beam size _k_ and the temperatures _γ_ and _τ_ controlling the generation and sampling diversity, respectively. 

Figure 6a shows the trend of performance improvement with the increase of beam size _k_ . Notably, our beam search approach inherently enables majority voting on the final beam without additional cost, resulting in a more significant performance improvement in the multiple-chain reasoning when the beam size is larger ( _e.g._ , 42 _._ 8% compared with 35 _._ 7% when _k_ = 10). 

For generation and sampling diversity, it is clear that more diversity resulting from higher temperatures generally leads to a decline in performance when only considering a single reasoning chain. However, diversity significantly benefits majority voting on multiple reasoning chains[6] . This benefit comes 

> 6In this study, we did not explore higher generation temperatures ( _i.e._ , _γ >_ 1 _._ 0) since this hyperparameter is limited to 1 _._ 0 in the OpenAI API. 

8 

|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_1]** **Grace**weighs**125**pounds.**Alex**weighs**2** pounds less than**4** times what<br>**Grace**weighs. What are their**combined**weights in pounds?<br>**[Ground-Truth****_a_1*]** 623.0|**[****_Q_2] Mariah**and**grandma**used**1/4**and**1/2**, respectively, from**364**<br>yards in a skein of yarn. How many yards of yarn did they use**in total**?<br>**[Ground-Truth****_a_2*]** 273.0|**[****_Q_2] Mariah**and**grandma**used**1/4**and**1/2**, respectively, from**364**<br>yards in a skein of yarn. How many yards of yarn did they use**in total**?<br>**[Ground-Truth****_a_2*]** 273.0|**[****_Q_2] Mariah**and**grandma**used**1/4**and**1/2**, respectively, from**364**<br>yards in a skein of yarn. How many yards of yarn did they use**in total**?<br>**[Ground-Truth****_a_2*]** 273.0|**[****_Q_2] Mariah**and**grandma**used**1/4**and**1/2**, respectively, from**364**<br>yards in a skein of yarn. How many yards of yarn did they use**in total**?<br>**[Ground-Truth****_a_2*]** 273.0|
|---|---|---|---|---|---|---|---|---|---|---|---|
|**[Predicted****_a_11]** 623.0<br>**[****_R_11]**inPython||||||||**[Predicted****_a_21]** 273.0<br>**[****_R_21]**inPython<br>C P Ɛ  **yards_per_skein =364**<br>C P Ɛ  **mariah_yards =1 / 4 *yards_per_skein**<br>C P Ɛ  **grandma_yards =1 / 2 *yards_per_skein**<br>C P Ɛ  **answer = mariah_yards +grandma_yards**||||
|C|P||Ɛ||**grace_weight =125**|||C|P|Ɛ|**yards_per_skein =364**|
|C|P||Ɛ||**alex_weight =4 *grace_weight -2**|||C|P|Ɛ|**mariah_yards =1 / 4 *yards_per_skein**|
|C|P||Ɛ||**answer =grace_weight + alex_weight**|||C|P|Ɛ|**grandma_yards =1 / 2 *yards_per_skein**|
|||||||||C|P|Ɛ|**answer = mariah_yards +grandma_yards**|
|**[Predicted****_a_12]** 627.0<br>**[****_R_12]**inPython||||||||**[Predicted****_a_22]** 273.0<br>**[****_R_22]**inPython<br>C P Ɛ  **yarn_mariah =1 / 4**<br>C P Ɛ  **yarn_grandma =1 / 2**<br>C P Ɛ  **yards_per_skein =364**<br>C P Ɛ  **total_yards =yarn_mariah +yarn_grandma**<br>C P Ɛ  **yards_used = total_yards *yards_per_skein**||||
|C|P||Ɛ||**grace_weight =125**|||C|P|Ɛ|**yarn_mariah =1 / 4**|
|C|P||Ɛ||**alex_weight =2**|||C|P|Ɛ|**yarn_grandma =1 / 2**|
|C|P||Ɛ||**weight_multiplier =4**|||C|P|Ɛ|**yards_per_skein =364**|
|C|P||Ɛ||**alex_total = alex_weight + weight_multiplier *grace_weight**|||C|P|Ɛ|**total_yards =yarn_mariah +yarn_grandma**|
|C|P||Ɛ||**answer =grace_weight + alex_total**|||C|P|Ɛ|**yards_used = total_yards *yards_per_skein**|
|(a) Examples of self-evaluation score distribution of differentpredictions on the GSM8K dataset.||||||||||||
|<br><br><br> <br><br>**[****_Q_3]**Did**Columbus**obtain his**funding**from the rulers of the**Portugese**Empire?<br>**[Ground-Truth****_a_3*]**no<br>**[Explanation]**King Ferdinand and Queen Isabella**funded Columbus**' voyage. They<br>were the joint rulers of the**Iberian Peninsula**, which included**Spain**but**excludes**<br>**Portugal**.  King John II of**Portugalrejected** **Columbus**' request for**funding**.|||||||**[****_Q_4]**Is**Freya**a combination of**Athena**and**Aphrodite**?<br>**[Ground-Truth****_a_4*]**yes<br>**[Explanation]** **Athena**was the**Greek**goddess of**war**.**Aphrodite**was the**Greek**<br>goddess of**love**.**Freya **was the**Norse**goddess of**war**,**love**, and**fertility**.|||||
||||||||**[Predicted****_a_41]**no<br>**[****_R_41]**inChain-of-Thought<br>C<br>P<br>Ɛ  **Freya is aNorsegoddess.**<br>C<br>P<br>Ɛ  **Athena and Aphrodite areGreekgoddesses.**<br>C<br>P<br>Ɛ  **Thus, Freya is not a combination of Athena and Aphrodite.**<br>C<br>P<br>Ɛ  **So the answer is no.**|||||
||||||||C|P|Ɛ|**Freya is aNorsegoddess.**||
|<br><br>**[Predicted****_a_31]**no<br>**[****_R_31]**inChain-of-Thought<br>C<br>P<br>Ɛ  **Columbus obtained his funding from the rulers of theSpanishEmpire.**<br>C<br>P<br>Ɛ  **Thus, Columbus didn’t obtain his funding ...PortugeseEmpire.**<br>C<br>P<br>Ɛ  **So the answer is no.**|||||||C|P|Ɛ|<br> **Athena and Aphrodite areGreekgoddesses.**||
||||||||C|P|Ɛ|**Thus, Freya is not a combination of Athena and Aphrodite.**||
|C<br>C|P<br>P|Ɛ <br>Ɛ||**Columbus obtained his funding from the rulers of theSpanishEmpire.**<br>**Th Clb did’t bti hi fdiPtEi**|||C|P|Ɛ|<br> **So the answer is no.**||
|C|P|<br>Ɛ||**us, oumus n oan s unng ...orugesempre.**<br>  **So the answer is no.**|||**[Predicted****_a_42]**yes<br>**[****_R_42]**inChain-of-Thought<br>C<br>P<br>Ɛ  **Freya is the goddess of love, beauty and fertility.**<br>C<br>P<br>Ɛ  **You can compare these to theGreekgods Athena and Aphrodite.**<br>C<br>P<br>Ɛ  **It is not clear how their powers exactly compare.**<br>C<br>P<br>Ɛ  **However, it seems that Freya must have attributes that match both.**<br>C<br>P<br>Ɛ  **So the answer is yes.**|||||
|**[Predicted****_a_32]**no<br>**[****_R_32]**inChain-of-Thought<br>C<br>P<br>Ɛ  **Columbus wasItalianand thePortugeseEmpire wasPortugese.**<br>C<br>P<br>Ɛ  **Thus, Columbus didn’t obtain funding ...PortugeseEmpire.**<br>C<br>P<br>Ɛ  **So the answer is no.**|||||||C|P|Ɛ|**Freya is the goddess of love, beauty and fertility.**||
||||||||C|P|Ɛ|**You can compare these to theGreekgods Athena and Aphrodite.**||
||||||||C|P|Ɛ|**It is not clear how their powers exactly compare.**||
||||||||C|P|Ɛ|**However, it seems that Freya must have attributes that match both.**||
||||||||C|P|Ɛ|**So the answer is yes.**||



(b) Examples of self-evaluation score distribution of different predictions on the StrategyQA dataset. We also provide explanations corresponding to the ground-truth answers for reference. 

Figure 7: Comparisons among predictions of high and low self-evaluation scores on arithmetic (7a for GSM8K) and commonsense (7b for StrategyQA) reasoning tasks. Scores from low to high are visualized from orange (0 _._ 0), yellow (0 _._ 4), to green (1 _._ 0). Here _C, P_ , and _E_ represent the evaluation confidence, the generation confidence, and their combination as the final score, respectively. 

from the improved coverage of the plausible generations and the ensembling effect. Nevertheless, one can adjust the sampling-related parameters ( _i.e._ , _τ_ and _α_ ) to incorporate more randomness into the generations. In practice, we find that a moderate temperature decay ( _e.g._ , _α_ = 0 _._ 5) results in improved performance. We conduct further analysis of the effect of sampling diversity in Appendix A.2. 

**Qualitative Analysis.** We examine particular instances to investigate the behavior of correctness confidence scores _C_ and generation probabilities _P_ in different scenarios. From the comparison shown in Figure 7, we have the following main observations: 

_•_ In general, the correctness confidence is more effective at identifying logical errors, taking into account the accumulated mistakes from prior steps, while the generation probability focuses more on text perplexity as the confidence of the generation LLM. 

_•_ When comparing arithmetic and commonsense tasks, LLMs exhibit greater confidence in dealing with structured and objective reasoning chains such as problems in GSM8K, for both generation and self-evaluation, as opposed to reasoning chains in StrategyQA. 

_•_ Reasoning chains that appear logically plausible can achieve high correctness confidence scores but still result in incorrect answers, as demonstrated in _R_ 41 in Figure 7b. Moreover, the correctness confidence can be influenced by minor details ( _e.g._ , imperfect variable naming in PAL reasoning) and assign low scores regardless of the correctness of the final answers as shown in _R_ 22 in Figure 7a. 

_•_ Incoherence due to a sudden jump in reasoning ( _e.g._ , _R_ 32 in Figure 7b) can lead to low correctness confidence. Additionally, the correctness confidence tends to be lower when the generation LLM makes a probability statement with less certainty, such as “it seems” as illustrated by _R_ 42 in Figure 7b. 

9 

## **4 Related Work** 

**Reasoning Formulation.** Several studies have attempted to better formulate the reasoning problem. One approach is to generate rationales to enhance model interpretability (Zhou et al., 2020; Wiegreffe and Marasovic, 2021; Wiegreffe et al., 2021). Recently, the focus has shifted towards decomposing the reasoning process into intermediate steps before reaching the final answer (Wei et al., 2022b; Zhou et al., 2023; Gao et al., 2023; Chen et al., 2022). Various decomposition techniques have been explored, such as question reduction (Zhou et al., 2023; Yang et al., 2022), iterative prompting (Wang et al., 2022), and chaining the steps (Wu et al., 2022). While incorporating intermediate reasoning steps has resulted in substantial performance improvements, errors or imperfections can accumulate, especially when the chains become longer (Wu et al., 2016; Guo et al., 2018). As such, we utilize LLM self-evaluation as a stepwise criterion to improve the chaining process. 

**LLM Self-Evaluation.** Recent research on LLM calibration shows that current LLMs’ probabilistic predictions correspond well with actual token occurrence frequencies, leading to well-calibrated predictions for specific tasks (Rae et al., 2021; Kadavath et al., 2022; Guo et al., 2017; Kadavath et al., 2022; Jiang et al., 2021; Kuhn et al., 2023). Notably, scaling model size plays a crucial role in enhancing calibration (Rae et al., 2021; Wei et al., 2022a). As LLMs exhibit good calibration, an increasing number of studies focus on prompting LLMs to perform self-evaluation as a means of verification (Zhang et al., 2023; Shinn et al., 2023; Madaan et al., 2023; Paul et al., 2023). Selfevaluation provides an effective and efficient assessment method without requiring task-specific verifier fine-tuning, which typically involves additional annotations (Li et al., 2022). In contrast to existing works that refine generation results through instance-level self-evaluation, our approach applies self-evaluation results as a stepwise criterion to calibrate generation at a finer granularity. By focusing on step-by-step self-evaluation, our method enables fine-grained guided decoding, addressing the challenges associated with complex or lengthy reasoning. 

**Decoding Strategies.** A tradeoff typically exists between diversity and quality. Deterministic decoding methods such as greedy decoding and beam search (Jurafsky and Martin, 2009; Graves, 2012) often produce high-quality results but lack diversity (Stahlberg and Byrne, 2019; Meister et al., 2020). Temperature sampling (Ackley et al., 1985), top- _k_ sampling (Fan et al., 2018), and top- _p_ sampling (Holtzman et al., 2020) are various techniques used to enhance diversity. The recent work of _tree-of-thought_ (Yao et al., 2023) explores different search algorithms such as breadth-first and depthfirst searches tailored for different tasks. Differently, we propose a unified framework of stochastic beam search (Caccia et al., 2020; Kool et al., 2019; Meister et al., 2021), which combines beam search and temperature sampling to balance the quality–diversity trade-off in multi-step reasoning. 

## **5 Discussion** 

We have introduced a multi-step decoding method that calibrates reasoning with stepwise selfevaluation guidance via stochastic beam search for current large language models. The empirical success of our method across a broad range of tasks, from arithmetic and symbolic to commonsense reasoning, demonstrates its robustness and generalizability in various application areas. The significant performance gains of our method on long reasoning chains also highlight its applicability to other multi-step tasks, such as multi-hop question answering and more complex scenarios involving multi-modal understanding, reasoning, and planning. In future work, we will investigate how to utilize external tools to further enhance the calibration and explore its generalizability on other multi-step scenarios to deal with more complex information such as external knowledge and multimodalities. 

## **Potential Impacts and Limitations** 

We propose self-evaluation guided stochastic beam search to facilitate multi-step reasoning. However, our approach, based on stepwise self-evaluation guidance, has certain limitations. It requires access to LLM logits to calculate the self-evaluation score, restricting its applicability to more powerful LLMs, such as GPT-4, which do not provide token likelihoods. Plus, multi-step decoding inherently causes additional costs from candidate sampling and self-evaluation. For optimal balance between efficiency and cost, our approach is best applied to longer reasoning chains, where the cumulative effect of calibration across multiple steps can improve the overall performance more significantly. 

10 

## **Acknowledgments and Disclosure of Funding** 

The computational work for this article was partially performed on resources of the National Supercomputing Centre (NSCC), Singapore[7] . We would like to thank Prof. Hwee Tou Ng for his insightful discussions that enhanced the depth and quality of our study. 

## **References** 

David H. Ackley, Geoffrey E. Hinton, and Terrence J. Sejnowski. 1985. A learning algorithm for boltzmann machines. _Cognitive Science_ , 9(1):147–169. 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ . 

- Massimo Caccia, Lucas Caccia, William Fedus, Hugo Larochelle, Joelle Pineau, and Laurent Charlin. 2020. Language gans falling short. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _CoRR_ , abs/2211.12588. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. _CoRR_ , abs/2204.02311. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168. 

- Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 889–898, Melbourne, Australia. Association for Computational Linguistics. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: program-aided language models. In _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 10764–10799. PMLR. 

> 7https://www.nscc.sg/ 

11 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. _Trans. Assoc. Comput. Linguistics_ , 9:346–361. 

- Alex Graves. 2012. Sequence transduction with recurrent neural networks. _CoRR_ , abs/1211.3711. 

- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In _Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017_ , volume 70 of _Proceedings of Machine Learning Research_ , pages 1321–1330. PMLR. 

- Jiaxian Guo, Sidi Lu, Han Cai, Weinan Zhang, Yong Yu, and Jun Wang. 2018. Long text generation via adversarial training with leaked information. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 32(1). 

- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve. 

- Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. 2021. How can we know _When_ language models know? on the calibration of language models for question answering. _Trans. Assoc. Comput. Linguistics_ , 9:962–977. 

- Dan Jurafsky and James H. Martin. 2009. _Speech and language processing : an introduction to natural language processing, computational linguistics, and speech recognition_ . Pearson Prentice Hall, Upper Saddle River, N.J. 

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. 2022. Language models (mostly) know what they know. _CoRR_ , abs/2207.05221. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In _NeurIPS_ . 

- Wouter Kool, Herke van Hoof, and Max Welling. 2019. Stochastic beams and where to find them: The gumbeltop-k trick for sampling sequences without replacement. In _Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA_ , volume 97 of _Proceedings of Machine Learning Research_ , pages 3499–3508. PMLR. 

- Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. In _NeurIPS_ . 

- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2022. On the advance of making language models better reasoners. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers_ , pages 158–167. Association for Computational Linguistics. 

- Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2023. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. _CoRR_ , abs/2303.17651. 

12 

- Clara Meister, Afra Amini, Tim Vieira, and Ryan Cotterell. 2021. Conditional poisson stochastic beam search. _CoRR_ , abs/2109.11034. 

- Clara Meister, Ryan Cotterell, and Tim Vieira. 2020. Best-first beam search. _Trans. Assoc. Comput. Linguistics_ , 8:795–809. 

- Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing english math word problem solvers. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020_ , pages 975–984. Association for Computational Linguistics. 

- Maxwell I. Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2021. Show your work: Scratchpads for intermediate computation with language models. _CoRR_ , abs/2112.00114. 

OpenAI. 2023. GPT-4 technical report. _CoRR_ , abs/2303.08774. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021_ , pages 2080–2094. Association for Computational Linguistics. 

- Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boi Faltings. 2023. REFINER: reasoning feedback on intermediate representations. _CoRR_ , abs/2304.01904. 

- Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, H. Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, PoSen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew J. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. 2021. Scaling language models: Methods, analysis & insights from training gopher. _CoRR_ , abs/2112.11446. 

- Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. _CoRR_ , abs/2303.11366. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Santilli, Andreas Stuhlmüller, Andrew M. Dai, Andrew La, Andrew K. Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakas, and et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _CoRR_ , abs/2206.04615. 

- Felix Stahlberg and Bill Byrne. 2019. On NMT search errors and model errors: Cat got your tongue? In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 3356–3362, Hong Kong, China. Association for Computational Linguistics. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , pages 4149–4158. Association for Computational Linguistics. 

13 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. _CoRR_ , abs/2307.09288. 

- Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. 2022. Solving math word problems with process- and outcome-based feedback. 

- Boshi Wang, Xiang Deng, and Huan Sun. 2022. Shepherd pre-trained language models to develop a train of thought: An iterative prompting approach. _CoRR_ , abs/2203.08383. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022a. Emergent abilities of large language models. _Trans. Mach. Learn. Res._ , 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In _NeurIPS_ . 

- Sarah Wiegreffe and Ana Marasovic. 2021. Teach me to explain: A review of datasets for explainable natural language processing. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ . 

- Sarah Wiegreffe, Ana Marasovic, and Noah A. Smith. 2021. Measuring association between labels and free-text rationales. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , pages 10266–10284. Association for Computational Linguistics. 

- Tongshuang Wu, Michael Terry, and Carrie Jun Cai. 2022. Ai chains: Transparent and controllable human-ai interaction by chaining large language model prompts. In _Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems_ , CHI ’22, New York, NY, USA. Association for Computing Machinery. 

- Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. _CoRR_ , abs/1609.08144. 

- Jingfeng Yang, Haoming Jiang, Qingyu Yin, Danqing Zhang, Bing Yin, and Diyi Yang. 2022. SEQZERO: few-shot compositional semantic parsing with sequential prompts and zero-shot models. In _Findings of the Association for Computational Linguistics: NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pages 49–60. Association for Computational Linguistics. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. _CoRR_ , abs/2305.10601. 

- Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-Tau Yih, Daniel Fried, and Sida Wang. 2023. Coder reviewer reranking for code generation. In _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 41832–41846. PMLR. 

14 

- Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Wangchunshu Zhou, Jinyi Hu, Hanlin Zhang, Xiaodan Liang, Maosong Sun, Chenyan Xiong, and Jian Tang. 2020. Towards interpretable natural language understanding with explanations as latent variables. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ . 

15 

## **A Appendix** 

## **A.1 Theoretical Analysis of Eq. 4** 

In Eq. 4, we use _S_ sampled from the language model LM _G_ generations. This is an approximation for sampling from the infinite set of all possible chaining paths. And the finite set _S_ is constructed based on the generation LM _P_ LM _G_ , which is different from our target distribution as shown in Eq. 4. 

Specifically, denote the infinite set of all possible generated completions till the _t_ -th step as _S[∗]_ , we approximate sampling from _Pbeam[∗]_[(] _[s]_[1:] _[t]_[) =] � _s_[1:] _[t]_ ex _∈S_ p _[∗]_ ( _E_[exp (] ( _s_[1:] _[t][E]_ )[(] _/[s] τ_[1:] ) _[t]_[)] _[/τ]_[)][via] _[ P][beam]_[(] _[s]_[1:] _[t]_[) =] � _s_[1:] ex _[t] ∈S_ p ([exp (] _E_ ( _s_[1:] _[E][t]_ )[(] _/[s]_[1:] _τ_ ) _[t]_[)] _[/τ]_[)][, where] _[ S]_[is] the approximation of _S[∗]_ with _|S|_ = _nk_ = _M ≤|S[∗] |_ . 

Define the upper bound _s_[1:] _[t] ∈S[∗]_ . Define the ratio as ¯ _c_ and the lower bound _r_ = _c_ ¯ _/c_ . Note that _c c_ on each _≥_ 1 since exp ( _E_ ( _Es_ ([1:] _s[t]_[1:] ) _[t] /τ_ ) _/τ≥_ ) as0. ¯Thus, we can take _c ≥_ exp ( _E_ ( _s_[1:] _[t]_ ) _/τ r_ ) _≤≥c_ ¯. _c_ for all We now give the following proposition which shows that _|Pbeam[∗]_[(] _[s]_[1:] _[t]_[)] _[ −][P][beam]_[(] _[s]_[1:] _[t]_[)] _[|]_[ decreases at the rate of] _O_ ([1] _[−][M] M[/][|][S][∗][|]_ ) toward 0 as _M_ increases. Note that as _M_ increases toward _|S[∗] |_ , the numerator 1 _− M/|S[∗] |_ decreases toward 0 while the factor for the denominator _M_ 1[also decreases.] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0016-05.png)


_Proof._ We now prove the second statement by analyzing the absolute difference: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0016-07.png)


Since exp ( _E_ ( _s_[1:] _[t]_ ) _/τ_ ) is nonnegative, using the upper bound on each exp ( _E_ ( _s_[1:] _[t]_ ) _/τ_ ), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0016-09.png)


Similarly, using the lower bound on each exp ( _E_ ( _s_[1:] _[t]_ ) _/τ_ ), 

16 

## **A.2 Extended Experiments** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0017-01.png)


**----- Start of picture text -----**<br>
80 80.2<br>79.4 79.4<br>79.0 78.8 78.8<br>78 78.4<br>77.9<br>77.6<br>76.6<br>76<br>LM temperature = 0.8 LM temperature = 0.8<br>74 LM temperature = 1.0 LM temperature = 1.0 7 4.1<br>73. 4<br>0.00 0.25 0.50 0.75 1.00 0.2 0.4 0.6 0.8 1.0<br>Sampling temperature  ( = 0.5) Decay ratio  ( = 0.5)<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 8: Accuracy curves with different sampling diversity. The two plots show the changes in performance on GSM8K when the sampling temperature _τ_ and its decay ratio _α_ vary, respectively. 

**Sampling Diversity.** In accordance with Figure 6b, we observe similar results when ablating the sampling hyperparameters _τ_ and _α_ for the single reasoning chain case, as shown in Figure 8. Increasing _τ_ and _α_ generally adds more diversity to the decoding process, but excessive randomness negatively impacts the performance of the single-chain decoding. Generally, a moderate temperature decay results in improved performance. Therefore, we set _α_ = 0 _._ 5 throughout our experiments for simplicity and only tune _τ_ for randomness control. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0017-04.png)


**----- Start of picture text -----**<br>
0.62 = 0.2<br>0.82<br>= 0.5<br>0.61 = 0.8<br>0.80<br>= 0.2 0.60<br>0.78 = 0.4<br>= 0.8<br>0.59<br>0.76<br>0.58<br>0.74<br>0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8<br>(a)  λ -AUC curves of  Eλ on GSM8K (PAL). (b)  λ -AUC curves of  Eλ on StrategyQA (CoT).<br>AUC AUC<br>**----- End of picture text -----**<br>


Figure 9: The change of AUC scores with different values of _λ_ in _Eλ_ . We calculate the AUC score as how _Eλ_ can successfully determine whether the corresponding predicted reasoning chain can produce the ground-truth answer. The predictions here are from the baseline methods ( _i.e._ , CoT & PAL) with different LM temperatures _γ_ , as represented by curves of different colors. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0017-06.png)


**----- Start of picture text -----**<br>
1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>High-scored ( > 0.90) Predictions Low-scored ( < 0.70) Predictions High-scored ( > 0.70) Predictions Low-scored ( < 0.50) Predictions<br>(a) GSM8K (PAL prompting) (b) StrategyQA (CoT prompting)<br>Instance-level Accuracy Instance-level Accuracy<br>**----- End of picture text -----**<br>


Figure 10: Comparison between predictions of high v.s. low self-evaluation scores on instance-level accuracy. 

17 

**More Analysis on Self-Evaluation.** Recall that we use a combination of generation confidence and faithfulness score as _Eλ_ = _C[λ] · P_[(1] _[−][λ]_[)] , with _λ ∈_ [0 _,_ 1]. In our experiments, we set _λ_ = 0 _._ 5 for all tasks for simplicity. However, we investigate its effects here since, intuitively, it is an important hyperparameter for distinguishing correct / incorrect predictions and might require different values for various reasoning tasks and datasets. Its effect is also coupled with the language model temperature _γ_ . 

Figure 9 demonstrates how _λ_ functions on arithmetic (GSM8K) and commonsense (StrategyQA). In general, we observe that the performance remains relatively stable with different choices of _λ_ on different datasets, although fine-tuning this hyperparameter might lead to further improvements. This stability suggests that the choice of _λ_ is not overly sensitive across various reasoning tasks and datasets, but exploring its optimal value for specific tasks could potentially lead to even better performances. 

To examine the influence of incorporating faithfulness on LLM final predictions, we plot the distributions of changes in different scores, specifically the faithfulness score _C_ , the generation confidence _P_ , and the overall decoding score _Eλ_ on the baseline reasoning chains and the reasoning chains generated by our method. We categorize the data points into 4 sets based on whether our approach changes the final prediction. Since the majority of the data points belong to the “both correct” set (in blue), where both baselines and our method generate accurate predictions, we particularly highlight the last two sets (in green and red), where our method results in improvement and degradation, respectively. 

As shown in Figure 11, faithfulness typically works by significantly increasing the evaluation confidence _C_ of model predictions, while the generation confidence _P_ remains similar to that of the baseline methods. Specifically, for the evaluation confidence _C_ , our approach corrects the original predictions by increasing the confidence scores. This indicates that evaluation confidence plays a crucial role in guiding the decoding toward a better reasoning choice in decomposed reasoning. The increase is more significant for PAL when compared with CoT. This demonstrates that LLMs are generally better at judging the logic in reasoning that is more structured, while the free-text intermediate steps ( _e.g._ , CoT reasoning) may be challenging to conduct information extraction and soundness checking. 

A similar conclusion can be drawn from Figure 10, where the difference in instance-level accuracy distributions between high-scored and low-scored predictions is more significant on the GSM8K dataset. For StrategyQA, while the incorporation of faithfulness helps, the level of the score value does not align well with whether the prediction is correct. For example, most of the low-scored predictions can still obtain the correct answers, as shown by the plot on the right of Figure 10b. 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0019-00.png)


**----- Start of picture text -----**<br>
200 both correct both correct both correct<br>both wrong both wrong both wrong<br>wrong   correct wrong   correct wrong   correct<br>100 correct   wrong correct   wrong correct   wrong<br>0 change in generation probability ( ) change in correctness confidence ( ) change in score ( ),  = (1 ), = 0.5<br>20 wrong   correct wrong   correct wrong   correct<br>correct   wrong correct   wrong correct   wrong<br>10<br>0<br>0.8 0.6 0.4 0.2 0.0 0.2 0.4 0.6 0.2 0.0 0.2 0.4 0.6 0.2 0.0 0.2 0.4 0.6<br>change in generation probability ( ) change in correctness confidence ( ) change in score ( ),  = (1 ), = 0.5<br>(a) Distributions of score shifts on GSM8K using PAL prompting.<br>75 both correct both correct both correct<br>both wrong both wrong both wrong<br>50 wrong   correct wrong   correct wrong   correct<br>correct   wrong correct   wrong correct   wrong<br>25<br>0 change in generation probability ( ) change in correctness confidence ( ) change in score ( ),  = (1 ), = 0.5<br>20 wrong   correct wrong   correct wrong   correct<br>correct   wrong correct   wrong correct   wrong<br>10<br>0<br>0.4 0.2 0.0 0.2 0.4 0.1 0.0 0.1 0.2 0.3 0.1 0.0 0.1 0.2 0.3<br>change in generation probability ( ) change in correctness confidence ( ) change in score ( ),  = (1 ), = 0.5<br>(b) Distributions of score shifts on GSM8K using CoT prompting.<br>both correct both correct both correct<br>150<br>both wrong both wrong both wrong<br>100 wrong   correct wrong   correct wrong   correct<br>correct   wrong correct   wrong correct   wrong<br>50<br>0 change in generation probability ( ) change in correctness confidence ( ) change in score ( ),  = (1 ), = 0.5<br>wrong   correct wrong   correct wrong   correct<br>15 correct   wrong correct   wrong correct   wrong<br>10<br>5<br>0<br>0.6 0.4 0.2 0.0 0.2 0.4 0.2 0.0 0.2 0.4 0.2 0.1 0.0 0.1 0.2 0.3<br>change in generation probability ( ) change in correctness confidence ( ) change in score ( ),  = (1 ), = 0.5<br>(c) Distributions of score shifts on StrategyQA using CoT prompting.<br>**----- End of picture text -----**<br>


Figure 11: Distributions of changes in scores from baselines to our method. Since the prediction correctness keeps unchanged most of the time ( _i.e._ , “both correct/incorrect” in blue/orange), we specifically plot how the scores shift on data points where the predictions get corrected or incorrect, as shown in green and red, respectively. 

19 

Table 5: Impact of LLM backends (Codex vs. ChatGPT vs. GPT-4) and prompting methods (PAL vs. CoT). The results of ChatGPT ( `gpt-3.5-turbo` ) were obtained on 20 March 2023. 

|Model|GSM8K|StrategyQA|
|---|---|---|
|CoT Codex|65_._6|73_._2|
|PAL Codex|72_._0|_−_|
|CoT ChatGPT|80_._8|65_._9|
|PAL ChatGPT|78_._7|_−_|
|CoT GPT_−_4|**92**_._**0**|_−_|
|Ours (CoT) Codex|71_._9|**77**_._**2**|
|Ours (PAL) Codex|80_._2|_−_|



**LLM Backbone Study.** We are interested in how stronger LLMs ( _i.e._ , ChatGPT, GPT-4 (OpenAI, 2023)) work, but they are not directly compatible with our approach since the API does not return token logits. 

Table 5 compares the results of various backend LLMs ( _i.e._ , Codex, ChatGPT, and GPT-4) on GSM8K. In arithmetic reasoning with PAL prompting, our Codex-based method achieves competitive results (80 _._ 2% vs. 78 _._ 7%) even when compared with ChatGPT. The results are consistent across other datasets, including AQuA (55 _._ 9% vs. 54 _._ 7%), SVAMP (89 _._ 6% vs. 84 _._ 1%), ASDiv (84 _._ 9% vs. 84 _._ 1%), and TabMWP (79 _._ 1% vs. 80 _._ 6%). In commonsense reasoning, our method using Codex significantly outperforms ChatGPT-based methods across different datasets, including StrategyQA (77 _._ 2% vs. 65 _._ 9%), CommonsenseQA (78 _._ 6% vs. 75 _._ 2%) and `Sports Understanding` (98 _._ 4% vs. 95 _._ 9%). One possible explanation is that ChatGPT lacks sufficient world knowledge for effective fact checking and commonsense reasoning. Given the significant performance improvement of GPT-4, we conduct further analysis about how to synergistically combine it with our method. 

**GPT-4 Experiments** The recently launched GPT-4 has demonstrated notable improvements in reasoning capabilities across a variety of tasks. In this section, we examine and compare the reasoning skills of different large language models (LLMs), specifically Codex and GPT-4, in assessing and determining the accuracy of each step in a reasoning chain. We contrast the confidence scores and corresponding explanations for Codex ( _C_ ) and GPT-4 ( _S_ ) in the context of both arithmetic and commonsense reasoning, as shown in Figure 13 and Figure 14, respectively. For ease of visualization, we employ the same colormap (shown in Figure 12) as in Figure 7 to represent the scale of scores. Since OpenAI has not provided access to the token-wise likelihood of generated text, we directly request GPT-4 to score the reasoning steps using binary values 8. Moreover, we report the average of three evaluation results to reduce the variance of sampling discrete values, _i.e._ , _S_ = ( _S_ 1 + _S_ 2 + _S_ 3) _/_ 3 _, Si ∈_ [0 _,_ 1]. 

As illustrated in Figure 13, GPT-4 demonstrates greater effectiveness in pinpointing the central logical error in arithmetic reasoning. For instance, we can observe that _S < C_ for `alex_total = alex_weight + weight_multiplier * grace_weight` and _S > C_ for `answer = grace_weight + alex_total` , where the former leads to an incorrect final answer. Additionally, GPT-4 typically offers detailed explanations and alternative solutions. As seen in the step `answer = grace_weight + alex_total` , GPT-4 can correct minor errors even when it arrives at the correct final answer. However, GPT-4 may still encounter difficulties in detecting small errors within the text, which can have a significant impact on logical consistency. This challenge is illustrated by the substantial variance in _S_ for the step `alex_total = alex_weight + weight_multiplier * grace_weight` . 

The benefits of well-crafted explanations in GPT-4 become more significant when handling complex reasoning tasks, as demonstrated in Figure 14. For instance, in the _R_ 42 of _Q_ 4 shown in Figure 7b, Codex generally assigns high evaluation scores for all steps. Although this reasoning chain leads to the correct final answer, it makes some overly definitive assumptions without reasonable justification (e.g., “must have attributes that match both”). In such cases, GPT-4 can accurately identify these vague statements through meticulous analysis. Moreover, the comprehensive analysis helps address the growing uncertainty arising from the ambiguity in understanding commonsense questions. For example, in the final step, GPT-4 offers extensive explanations for assigning low _S_ scores, considering the limited information available to comprehend the question. 

In summary, GPT-4 demonstrates promising improvements in LLM evaluation by providing more rigorous and in-depth explanations. While there is significant variation in its generated explanations for some challenging cases, these explanations are typically beneficial as they elaborate on understanding and interpretation based on the given information. This suggests the potential for using LLM explanations to enhance self-evaluation in future research. 

> 8The results using GPT-4 were obtained on 22 April 2023. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0021-00.png)


**----- Start of picture text -----**<br>
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0<br>self-evaluation scores:  |  |<br>**----- End of picture text -----**<br>


Figure 12: Score from low to high values visualized using colormap from orange to green . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning_images/nips_2023_Self-Evaluation_Guided_Beam_Search_for_Reasoning.pdf-0021-02.png)


**----- Start of picture text -----**<br>
[ Q 1] Grace  weighs  125  pounds.  Alex  weighs  2 pounds less than  4 times what  Grace  weighs.<br>What are their  combined  weights in pounds?<br>[Ground-Truth  a 1*] 623.0<br>[Predicted  a 12] 627.0<br>[ R 12]  in Python<br>grace_weight = 125<br>Ɛ C correct<br>S correct<br>1<br>The line of code correctly assigns the given weight of Grace ( 125  pounds) to the variable<br>S S2 grace_weight<br>The given line of code correctly assigns the value of  125  to the variable  grace_weight , representing<br>S3 Grace's weight in pounds.<br>alex_weight = 2<br>Ɛ C alex_weight  should be calculated based on  grace_weight<br>S1 alex_weight  should be calculated as  4  times  grace_weight  minus  2<br>The line of code defining  alex_weight  should be calculated as  4  times Grace's weight minus  2 , which<br>S S<br>2 is ( 4  *  grace_weight ) -  2 .<br>S3 The calculation for Alex's weight is not correct. It should be calculated as  4  *  grace_weight  -  2 .<br>weight_multiplier = 4<br>Ɛ C correct<br>S1 The line of code is correct as it defines the multiplier for Alex's weight based on Grace's weight.<br>Correct because it assigns the correct value ( 4 ) to the variable  weight_multiplier , which represents<br>S S2 the factor by which Grace's weight is multiplied to calculate Alex's weight.<br>The code correctly assigns the value  4  to the variable  weight_multiplier , which will be used in the<br>S<br>3 calculation of Alex's weight.<br>alex_total = alex_weight + weight_multiplier * grace_weight<br>Ɛ C correct formula to calculate  alex_total<br>S1 The formula used to calculate Alex's weight is incorrect.<br>The above line of code calculates Alex's weight correctly as  2  pounds less than  4  times Grace's weight<br>S S<br>2 ( 4  *  grace_weight  -  2 ).<br>S3 Alex weighs  2  pounds less than  4  times what Grace weighs, not  2  pounds more.<br>answer = grace_weight + alex_total<br>Ɛ C should be the sum of  grace_weight  and  alex_weight<br>The code calculates Alex's weight correctly using the given relationship and adds it to Grace's weight to<br>calculate the combined weight. The only minor issue is the unnecessary use of the separate variable<br>S<br>1 alex_weight  which could directly be replaced with the calculation ( 4  *  grace_weight  -  2 ).<br>However, this does not affect the correctness of the answer.<br>S S The result variable holds the correct combined weight of Grace and Alex, and this line of code correctly<br>2 calculates the value of  answer .<br>The code above correctly calculates the total weight of Grace and Alex based on the given problem<br>statement. The only minor issue is that the  alex_weight  variable is initialized as  2 , which might be a<br>S<br>3 bit confusing. It would be better to directly calculate  alex_total  as ( 4  *  grace_weight  -  2 ), but the<br>current code does not produce an incorrect result.<br>**----- End of picture text -----**<br>


Figure 13: Comparison between Codex ( _C_ ) and GPT-4 ( _S_ ) on evaluating the reasoning chains in arithmetic reasoning. The example ( _R_ 12 for _Q_ 1) is the same one from Figure 7a. We underline the key statements in explanations that justify the judgment of GPT-4. 

21 

**[** _**Q**_ **4]** Is **Freya** a combination of **Athena** and **Aphrodite** ? **[Ground-Truth** _**a**_ **4*]** yes **[Explanation] Athena** was the **Greek** goddess of **war** . **Aphrodite** was the **Greek** goddess of **love** . **Freya** was the **Norse** goddess of **war** , **love** , and **fertility** . 

## **[Predicted** _**a**_ **42]** yes **[** _**R**_ **42]** in Chain-of-Thought 

- **Freya is the goddess of love, beauty and fertility.** Ɛ C correct Freya indeed is associated with love, beauty, and fertility in **Norse** mythology. However, it is important to note that while the 

- S1 comparison to Athena and Aphrodite is understandable, it is not a direct or complete equivalence. The statement correctly identifies Freya. However, the statement does not mention that Freya is a **Norse** goddess, which could 

- S S2 slightly impact the understanding of the comparison to **Greek** goddesses Athena and Aphrodite. The statement is mostly correct as Freya is indeed a goddess associated with love, beauty, and fertility. However, it might be 

- S3 worth mentioning that she is a **Norse** goddess, while Athena and Aphrodite are **Greek** goddesses, to provide more context. 

- **You can compare these to the Greek gods Athena and Aphrodite.** Ɛ C correct It is true that Freya can be compared to the **Greek** gods Athena and Aphrodite in terms of their respective domains. However, 

- S1 the comparison is not entirely accurate as Freya comes from **Norse** mythology, and the attributes of the gods may not be an exact match. The comparison is a simplification, which is why the probability is not higher. The statement establishes a comparison between Freya and the **Greek** goddesses Athena and Aphrodite. While this step of 

- S S2 reasoning is generally correct, it could be improved by specifically mentioning that Freya is a **Norse** goddess and by providing more information about the roles and attributes of Athena and Aphrodite for a clearer comparison. While it is true that Freya shares some similarities with Athena and Aphrodite, the comparison is not entirely accurate. Freya is 

- S3 a some domains, but it might be **Norse** goddess, while Athena and Aphrodite are misleading to suggest that Freya is a **Greek** goddesses. The comparison is correct in the sense that they share direct "combination" of these two goddesses. 

- **It is not clear how their powers exactly compare.** Ɛ C correct This step of reasoning acknowledges that it is not entirely clear how the powers of Freya, Athena, and Aphrodite compare. Although there are some similarities, the differences between **Norse** and **Greek** mythologies make it difficult to determine 

- S1 whether Freya is a direct combination of Athena and Aphrodite. The high probability reflects the acknowledgment of the uncertainty in comparing the powers of the gods across different mythologies. The statement acknowledges that it is not clear how the powers of Freya, Athena, and Aphrodite exactly compare. However, 

- S the question asks if Freya is a combination of Athena and Aphrodite, which implies a synthesis or blending of their roles or S2 attributes. The reasoning could be improved by specifically mentioning the roles and attributes of Athena and Aphrodite, and whether or not they overlap with Freya's roles and attributes. The reasoning here is more accurate, acknowledging that while there are similarities between Freya and Athena and 

- S3 Aphrodite, it is not clear how their powers exactly compare. It is important to consider that Freya is a **Norse** goddess and Athena and Aphrodite are **Greek** goddesses, so their roles, domains, and characteristics may not be directly comparable. 

- **However, it seems that Freya must have attributes that match both.** Ɛ C correct incorrect because it assumes that Freya must have attributes that match both Athena and Aphrodite. While there may be some 

- S1 similarities in their attributes, it is not appropriate to conclude that Freya is a combination of the two **Greek** goddesses. Norse and **Greek** mythologies are distinct, and the deities in these pantheons may have different origins and roles. The statement acknowledges that Freya has attributes that match both Athena and Aphrodite. However, it would be more 

- S S2 accurate to discuss the specific attributes of Athena and Aphrodite to determine the extent of the similarities. It assumes that Freya must have attributes that match both Athena and Aphrodite, but this may not be the case. While there are similarities between these goddesses, they are from different mythologies (Freya – **Norse** , Athena & Aphrodite – **Greek** ), 

- S3 and their roles, domains, and characteristics may not be directly comparable or combined. It is not accurate to assume that Freya is a combination of Athena and Aphrodite without a more in-depth analysis in their respective mythologies. 

- **So the answer is yes.** Ɛ C correct This step of reasoning is incorrect because it is just based on some similarities in their attributes. While there may be some 

- S1 similarities, it is not accurate to say that Freya is a combination of the two **Greek** goddesses. **Norse** and **Greek** mythologies are distinct, and the deities in these pantheons have different origins and roles. Although Freya shares some attributes with both Athena and Aphrodite, it is an oversimplification to claim that she is a 

- S S2 combination of the two. The question seems to imply that Freya might be a direct mix of the two, but this is not accurate. The conclusion that Freya is a combination of Athena and Aphrodite is not accurate based on the information provided. While 

- S3 there are similarities between these goddesses, they belong to different mythologies (Freya is **Norse** , Athena and Aphrodite are **Greek** ), and their roles, domains, and characteristics may not be directly comparable or combined. 

Figure 14: Comparison between Codex ( _C_ ) and GPT-4 ( _S_ ) on evaluating the reasoning chains in commonsense reasoning. The example ( _R_ 42 for _Q_ 4) is the same one from Figure 7b. We underline the key points in GPT-4 rationales that explain the detailed understanding and analysis on the steps. 

22 

## **A.3 Implementation Details** 

Similar to beam search, we maintain _k_ distinct candidates in the beam and sample _n_ completions for each one. Thus, for each reasoning step _s[t]_ , the search space has a size of _k · n_ . After acquiring _k · n_ samples, we retain _k_ candidates by sampling from _Pbeam_ as Eq. 4. We set _k_ = 5 _, n_ = 16 with Codex backbone to balance the quality and efficiency. The maximum number of steps to decode is capped at 16. To control the computational cost and time complexity, one can also reduce the number of rollouts per beam and the beam size to _n_ = 2 and _k ∈_ [3 _,_ 4] respectively, as we illustrate with Llama-2 backbone. 

We set generation temperatures differently for various tasks and baselines. Regarding the generation temperature _γ_ on Codex, for arithmetic and symbolic reasoning with PAL using deterministic beam search ( _τ_ = 0 _._ 0), we find that _γ ∈_ [0 _._ 4 _,_ 0 _._ 8] generally works well. In contrast, for commonsense reasoning with CoT, a lower temperature ( _γ ∈_ [0 _._ 1 _,_ 0 _._ 5]) is more effective, likely due to the increased randomness from the free-text format. Differently, when using Llama-2 backbone, PAL generally works better with lower generation temperature _γ ≤_ 0 _._ 5, while CoT can tolerate larger _γ >_ 0 _._ 5 with better or comparable performance. This difference between Codex and Llama-2 may come from the different training tasks and data adopted for the two models, where the PAL reasoning is especially enhanced in Codex. 

In majority voting, higher _γ_ is preferred to better explore the search space in reasoning, _e.g._ , _γ ≥_ 0 _._ 5 for arithmetic reasoning. To further introduce sampling randomness in stochastic beam search for majority voting on multiple reasoning chains, we use _α_ = 0 _._ 5 for all datasets but different values of _τ_ for each task. Specifically, we choose _τ_ = 0 _._ 5 for PAL and _τ_ = 0 _._ 2 for CoT, as PAL typically decomposes the reasoning problem into more steps than CoT. Here we tune _τ_ instead of _α_ to be smaller in CoT as CoT naturally contains more randomness due to its free-text formulation as we observe in practice, where a smaller _τ_ is more efficient to balance this randomness. 

In previous works, majority voting on multiple reasoning chains involves sampling _N_ (usually _≥_ 20) reasoning chains and conducting a vote to determine the final answer, which can be time-consuming. In our approach, we simply perform majority voting among the _N_ candidates in the last step of beam search from only a few times ( _≤_ 10) of searching. As a result, our method does not introduce additional time complexity compared to the conventional majority voting method, although we sacrifice some diversity in the final outcomes due to the similarity of candidates within a beam. 

**Prompts.** We show examples of both the generation and evaluation prompts we use on different tasks in the following tables, where we only show one instance for each case. Full prompts and detailed formulations can be found in our code. 

23 

Table 6: Examples of few-shot exemplars of generation and evaluation CoT prompts on GSM8K. We adopt 8-shot and 5-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Q** : There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? **A** : 

There are 15 trees originally. And there were 21 trees after some more were planted. So 21 - 15 = 6 trees were planted. So the answer is 6. 

## **EVALUATION** 

**Q** : Allison brought some CDs online. Each CD cost $7. There was an additional charge of $4 per order for shipping costs. The total bill came to $60. How many CDs did Allison buy? **A** : 

Each CD cost 7 dollars. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is (A) And there was an additional charge of 4 dollars. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is (A) So the total cost of each CD is 7 + 4 = 11 dollars. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is (B), because each CD costs 7 dollars. So 60 / 11 = 5.45. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is (B), because it cost 4 dollars for shipping costs. So the cost of CDs is 60 - 4 = 56 dollars. So Allison bought 56 / 7 = 8 CDs. So the answer is 5. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is (A), but the value of the number of CDs is incorrect. 

24 

Table 7: Examples of few-shot exemplars of generation and evaluation PAL prompts on **GSM8K** . We adopt 9-shot and 5-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Q** : Olivia has $23. She bought five bagels for $3 each. How much money does she have left? 

```
defsolution():
money_initial=23
bagels=5
bagel_cost=3
money_spent=bagels*bagel_cost
=-
money_leftmoney_initialmoney_spent
result=money_left
returnresult
```

## **EVALUATION** 

**Q** : A piece of square paper has a perimeter of 32 centimeters. Nicky’s dog, Rocky, tore off 1/4 of the paper. What is the area of the remaining paper? `def solution(): perimeter = 32 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) fraction_torn = 1 / 4 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) area_total = (perimeter / 4) ** 2 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), because the total area of the square` _�→_ `can be calculated by (perimeter / 4) ** 2 area_remaining = (1 - fraction_torn) * area_total # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) result = area_total # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (B), because the result should be` _�→_ `area_remaining return result # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), but the value of result is incorrect` 

25 

Table 8: Examples of few-shot exemplars of generation and evaluation PAL prompts on **AQuA** . Here `...` represents the same evaluation script as those in the previous steps. We adopt 8-shot and 5-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Question** : In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is: **Answer Choices** : A)1 hour; B)2 hours; C)3 hours; D)4 hours; E)5 hours `def solution(): duration = Symbol('duration', positive=True) delay = 30 / 60 total_disntace = 600 original_speed = total_disntace / duration reduced_speed = total_disntace / (duration + delay) solution = solve_it(original_speed - reduced_speed - 200, duration) duration = solution[duration] result = duration return result` 

## **EVALUATION** 

**Question** : Two trains of length 150 m and 200 m are 100 m apart. They start moving towards each other on parallel tracks, at speeds 54 kmph and 72 kmph. In how much time will the trains cross each other? **Answer Choices** : A)100/7 sec; B)80/7 sec; C)57/7 sec; D)110/7 sec; E)50/7 sec `def solution(): train_1_speed = 54 / 60 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) train_2_speed = 72 / 60 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) distance_between_trains = 100 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) train_1_length = 150 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) train_2_length = 200 # ... # The above line of code is: (A) time_to_cross = distance_between_trains / (train_1_speed +` _�→_ `train_2_speed) # ... # The above line of code is: (B), because to cross each other, the` _�→_ `total distance should also contain the train length result = time_to_cross # ... # The above line of code is: (B), because the final result should be in` _�→_ `seconds, and the value of time_to_cross is incorrect return result # ... # The above line of code is: (A), but the value of result is incorrect` 

26 

Table 9: Examples of few-shot exemplars of generation and evaluation PAL prompts on **SVAMP** and **ASDiv** . Here we utilize the same prompts as they have the same task formulation. We adopt 7-shot and 5-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Passage** : James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. 

**Question** : How many red stickers does James have? 

```
defsolution():
original_red_stickers=93
used_red_stickers=31
red_stickers=original_red_stickers-used_red_stickers
result=red_stickers
returnresult
```

## **EVALUATION** 

**Passage** : A piece of square paper has a perimeter of 32 centimeters. Nicky’s dog, Rocky, tore off 1/4 of the paper. 

**Question** : What is the area of the remaining paper? `def solution(): perimeter = 32 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) side_length = perimeter / 4 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) area = side_length ** 2 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) result = area # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (B), because should calculate the` _�→_ `remaining area after torn off as result return result # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), but the value of result is incorrect` 

27 

Table 10: Examples of few-shot exemplars of generation and evaluation PAL prompts on **TabMWP** . We adopt 4-shot and 5-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Table of "Coin collections"** : Name | Number of coins Braden | 76 _\\_ Camilla | 94 _\\_ Rick | 86 Mary | 84 _\\_ Hector | 80 _\\_ Devin | 83 Emily | 82 _\\_ Avery | 87 

**Question** : Some friends discussed the sizes of their coin collections. What is the mean of the numbers? 

`def solution(): number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87] mean_of_the_numbers = sum(number_of_coins_for_different_person) /` _�→_ `len(number_of_coins_for_different_person) result = mean_of_the_numbers return result` 

## **EVALUATION** 

**Table of "Roller coasters per amusement park"** : 

Stem | Leaf 1 | 0, 0, 1, 6, 8, 9 _\\_ 2 | 4, 4, 5, 7, 8, 8 

3 | 1, 2, 4, 4, 9, 9 _\\_ 4 | 2, 3, 5, 6, 8, 9, 9 **Question** : Rodrigo found a list of the number of roller coasters at each amusement park in the state. How many amusement parks have fewer than 40 roller coasters? `def solution():` 

`number_of_roller_coasters_per_amusement_park = [10, 14, 14, 15, 16, 18, 19, 20, 24, 25, 26, 28, 29, 29, 29,` _�→_ `30, 34, 35, 36, 39, 40, 40, 40, 41, 42, 43, 44, 44, 45, 45, 46, 46, 47, 48, 48, 49, 49, 49, 50, 50, 51,` _�→_ `51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64,` _�→_ `64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77,` _�→_ `77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90,` _�→_ `90, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99] # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (B), beacuse values in the rows of Stem and Leaf represent the decimal and` _�→_ `individual digits, respectively number_of_amusement_parks_with_fewer_than_40_roller_coasters = 0 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), because this is to initialize the` _�→_ `number_of_amusement_parks_with_fewer_than_40_roller_coasters for number_of_roller_coasters in` _�→_ `number_of_roller_coasters_per_amusement_park: if number_of_roller_coasters < 40: number_of_amusement_parks_with_fewer_than_40_roller_coasters +=` _�→_ `1 # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), but the value of` _�→_ `number_of_roller_coasters_per_amusement_park is incorrect result = number_of_amusement_parks_with_fewer_than_40_roller_coasters # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), but the value of` _�→_ `number_of_amusement_parks_with_fewer_than_40_roller_coasters is` _�→_ `incorrect return result # ... # The above line of code is: (A), but the value of result is incorrect` 

28 

Table 11: Examples of few-shot exemplars of generation and evaluation PAL prompts on `Date Understanding` from **Big-Bench** . We adopt 6-shot and 3-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Q** : 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY? `def solution(): # If 2015 is coming in 36 hours, then today is 36 hours before. today = datetime(2015, 1, 1) - relativedelta(hours=36) # One week from today, one_week_from_today = today + relativedelta(weeks=1) # The answer formatted with %m/%d/%Y is result = one_week_from_today.strftime('%m/%d/%Y') return result` 

## **EVALUATION** 

**Q** : Yesterday was April 30, 2021. What is the date yesterday in MM/DD/YYYY? `def solution(): # Yesterday was April 30, 2021. yesterday = datetime(2021, 4, 30) # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A) # Yesterday, = - yesterday yesterday relativedelta(days=1) # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (B), because the date of yesterday was` _�→_ `already got and this is step is redundant # The answer formatted with %m/%d/%Y is result = yesterday.strftime('%m/%d/%Y') # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), but the value of yesterday is` _�→_ `incorrect return result # Is the above line of code: # (A) Correct # (B) Incorrect # The above line of code is: (A), but the value of result if incorrect` 

29 

Table 12: Examples of few-shot exemplars of generation and evaluation PAL prompts on `Object Counting` from **Big-Bench** . We adopt 4-shot and 1-shot for generation and evaluation, respectively. 

## **GENERATION** 

**Q** : I have a drum, a flute, a clarinet, a violin, four accordions, a piano, a trombone, and a trumpet. How many musical instruments do I have? `def solution(): musical_instruments_to_count = { 'drum': 1, 'flute': 1, 'clarinet': 1, 'violin': 1, 'accordion': 4, 'piano': 1, 'trombone': 1, 'trumpet': 1 } num_musical_instruments = sum(musical_instruments_to_count.values()) result = num_instruments return result` 

## **EVALUATION** 

**Q** : I have a peach, an onion, two garlics, a nectarine, a yam, a carrot, a potato, a head of broccoli, a cabbage, a banana, and a stalk of celery. How many vegetables do I have? `def solution(): # note: I'm not counting the peach, nectarine, banana, or celery as` _�→_ `vegetables vegetables_to_count = { 'onion': 1, # ... # The above line of code is: (A) 'garlic': 2, # ... # The above line of code is: (A) 'yam': 1, # ... # The above line of code is: (A) 'carrot': 1, # ... # The above line of code is: (A) 'potato': 1, # ... # The above line of code is: (A) 'broccoli': 1, # ... # The above line of code is: (A) 'cabbage': 1 # ... # The above line of code is: (A) } # ... # The above line of code is: (B), because celery is vegetable num_vegetables = sum(vegetables_to_count.values()) # ... # ... (A), but the value of vegetables_to_count is incorrect result = num_vegetables # ... # ... (A), but the value of num_vegetables is incorrect return result # ... # ... (A), but the value of result is incorrect` 

30 

Table 13: Examples of few-shot exemplars of generation and evaluation CoT prompts on **CSQA** . We adopt 7-shot and 3-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Q** : What do people use to absorb extra ink from a fountain pen? **Answer Choices** : (a) shirt pocket; (b) calligrapher’s hand; (c) inkwell; (d) desk drawer; (e) blotter 

**A** : 

The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (e). 

## **EVALUATION** 

**Q** : John’s took his appointment book with him to a checkup but couldn’t find it anywhere. He looked and he looked but he didn’t know where it was. It wasn’t until he bought a new one, went home, took a shower that he found it. Where was it? **Answer Choices** : (a) office supply store; (b) wallet; (c) doctor’s office; (d) handbag; (e) pocket 

**A** : 

The answer must be a place where John could have put his appointment book. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (B), because it fails to reason out that John will check the place when taking a shower (taking off clothes) Of the above choices, only wallet makes sense. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (B), because wallet is too small to put his appointment book, and it cannot be the only one which makes sense (pocket is also plausible) So the answer is (b). # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (A), but the previous step is incorrect 

31 

Table 14: Examples of few-shot exemplars of generation and evaluation CoT prompts on **StrategyQA** . We adopt 6-shot and 4-shot learning for generation and evaluation, respectively. 

**GENERATION Q** : Do hamsters provide food for any animals? **A** : Hamsters are prey animals. Prey animals are food for predators. Thus, hamsters provide food for some animals. So the answer is yes. **EVALUATION Q** : While viewing "Scary Movie" is the viewer likely to experience an increase in adrenaline? **A** : Adrenaline is a hormone that is released when the body is under stress. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (A) Scary movies are designed to make the viewer feel stressed. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (B), because Scary Movie is a film that is a comedy take on horror, intended to make viewers laugh Thus, the viewer is likely to experience an increase in adrenaline. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (A), but the previous step is incorrect So the answer is yes. # Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (A), but the previous step is incorrect 

32 

Table 15: Examples of few-shot exemplars of generation and evaluation CoT prompts on `Sports Understanding` on **Big-Bench** . We adopt 8-shot and 2-shot learning for generation and evaluation, respectively. 

## **GENERATION** 

**Q** : Is the following sentence plausible? 

“KYLE PALMIERI WAS CALLED FOR SLASHING.” 

**A** : 

Kyle Palmieri is a hockey player. Being called for slashing is part of hockey. So the answer is yes. 

## **EVALUATION** 

**Q** : Is the following sentence plausible? 

“ROBIN LEHNER LAUNCHED THE DESPERATION HEAVE” 

**A** : 

Robin Lehner is a hockey player. 

# Is the above step of reasoning: 

# (A) Correct 

# (B) Incorrect 

# The above step of reasoning is: (B), because Robin Lehner is a goaltender which is different from a player Launching the desperation heave is part of hockey. 

# Is the above step of reasoning: 

# (A) Correct 

# (B) Incorrect 

# The above step of reasoning is: (B), because launching the desperation heave is for player, not goaltender So the answer is yes. 

# Is the above step of reasoning: # (A) Correct # (B) Incorrect # The above step of reasoning is: (A), but the previous step is incorrect 

33 

