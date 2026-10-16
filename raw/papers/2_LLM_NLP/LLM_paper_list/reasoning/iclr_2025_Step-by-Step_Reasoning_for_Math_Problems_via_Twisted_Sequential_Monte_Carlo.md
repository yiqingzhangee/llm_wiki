Under Review 

# - STEP BY-STEP REASONING FOR MATH PROBLEMS VIA TWISTED SEQUENTIAL MONTE CARLO 

**Shengyu Feng**[1] _[∗]_ **Xiang Kong**[2] **Shuang Ma**[2] **Aonan Zhang**[2] **Dong Yin**[2] **Chong Wang**[2] **Ruoming Pang**[2] **Yiming Yang**[1] 1Language Technologies Institute, Carnegie Mellon University 2Apple 

## ABSTRACT 

Augmenting the multi-step reasoning abilities of Large Language Models (LLMs) has been a persistent challenge. Recently, verification has shown promise in improving solution consistency by evaluating generated outputs. However, current verification approaches suffer from sampling inefficiencies, requiring a large number of samples to achieve satisfactory performance. Additionally, training an effective verifier often depends on extensive process supervision, which is costly to acquire. In this paper, we address these limitations by introducing a novel verification method based on Twisted Sequential Monte Carlo (TSMC). TSMC sequentially refines its sampling effort to focus exploration on promising candidates, resulting in more efficient generation of high-quality solutions. We apply TSMC to LLMs by estimating the expected future rewards at partial solutions. This approach results in a more straightforward training target that eliminates the need for step-wise human annotations. We empirically demonstrate the advantages of our method across multiple math benchmarks, and also validate our theoretical analysis of both our approach and existing verification methods. 

## 1 INTRODUCTION 

In recent years, Large Language Models (LLMs) have achieved significant breakthroughs across various domains (Park et al., 2023; Kaddour et al., 2023; Song et al.; Li et al., 2023a; Wang et al., 2023a; Chen et al., 2023; Zheng et al., 2023; Wang et al., 2023c). However, their performance in multi-step reasoning tasks, such as solving complex mathematical or coding problems, remains notably constrained (Lightman et al., 2024; Huang et al., 2023). A key challenge arises from the high sensitivity of these tasks to individual errors at each step of reasoning. Autoregressive LLMs, in particular, struggle with maintaining consistency throughout the reasoning process, leading to solutions that are prone to mistakes or logical inconsistencies (Shen et al., 2021; Cobbe et al., 2021). 

Verification (Cobbe et al., 2021; Uesato et al., 2022; Lightman et al., 2024) has emerged as an effective strategy to mitigate these issues. In a typical verification process, multiple solutions are sampled from the generator (the LLM), and an external verifier evaluates each of these solutions. The verification outcomes are then used to adjust the weight of each solution in determining the final answer. Since verification is generally simpler than generation, it tends to achieve higher accuracy and consistency compared to the methods that rely solely on the generator, such as majority voting (Wang et al., 2023e). There are two primary types of verifiers: the Outcome Reward Model (ORM) (Cobbe et al., 2021) and the Process Reward Model (PRM) (Uesato et al., 2022). The ORM evaluates the fully generated solution with a single scalar output representing the confidence score, and its training is straightforward, using outcome supervision based on comparing generated answers with ground truth. In contrast, the PRM focuses on providing rewards at each step of the reasoning process, giving more detailed feedback on the intermediate steps. Although empirical evidence suggests that PRM outperforms ORM (Lightman et al., 2024), there exists no simple metric in evaluating the correctness of each step and efficiently collecting process supervision for such intermediate steps remains a great challenge. 

Despite being promising, existing verification methods are still limited in the following two areas: 

> _∗_ Work done during an internship at Apple. Correspondence to: Shengyu Feng shengyuf@cs.cmu.edu and Xiang Kong xiang ~~k~~ ong@apple.com. 

1 

Under Review 

- **(Problem I) Low sampling efficiency:** Current verification methods only evaluate fully generated solutions, without refining their quality during the generation process. Sampling efforts would be wasted on partial solutions that are clearly incorrect. As a result, a large number of samples are needed to obtain even one correct solution, making the process inefficient and resource-intensive. 

- **(Problem II) Difficulty in obtaining process supervision:** Training powerful verifiers like the PRM requires detailed step-wise supervision. Existing approaches either rely on human effort (Uesato et al., 2022; Lightman et al., 2024) or tree search (Wang et al., 2023d; Luo et al., 2024) for intermediate step annotations. However, both approaches are inefficient and lack scalability, limiting their practical application for large-scale tasks. 

To address these two significant problems in existing verification methods, we propose a novel approach based on _Twisted Sequential Monte Carlo (TSMC)_ (Doucet et al., 2001; Del Moral et al., 2006; Briers et al., 2009; Chopin & Papaspiliopoulos, 2020). TSMC is a significant advancement in the _Importance Sampling (IS)_ technology. Building on the foundation of Sequential Monte Carlo (SMC), TSMC is intended to enhance sampling efficiency of IS in the high-dimensional space. It employs a series of intermediate target distributions at each resampling step, which are defined through twist functions. This function strategically guides the samples towards the high-density region in the target distribution. By retaining the most promising samples, TSMC effectively reduces the variance in estimated quantities and boosts the efficiency of the sampling process. 

Notably, the application of TSMC to improve the verification method in LLMs has not been explored previously, making our study the first of its kind in this area. Our approach is inspired by the realization that existing verification methods employing reward-weighted majority voting (Li et al., 2023b) essentially performs IS, where the sampling efficiency deteriorates as the disparity between the proposal distribution (which generates potential solutions) and the target distribution (concentrated around correct solutions) widens. We identify Problem I—low sampling efficiency—as a consequence of high variance in IS when there is a substantial deviation between the proposal and target distributions. Multi-step reasoning, even minor discrepancies at each step can cumulate into a substantial mismatch between the two distributions. We therefore apply TSMC to improve the sampling efficiency of verification by focusing the sampling effort on promising partial solutions during the intermediate decoding process. We have shown the optimal twist functions in our case, which is used to guide the sampling of TSMC, is proportional to expected future rewards, also known as the value function. The value function could be simply learnt through a neural regressor on the data independently sampled from the generator. This simplifies the training target by eliminating the need for human annotations or tree search. We also highlight the relationship between TSMC and the PRM in existing verification methods, allowing for a comprehensive analysis of bias and variance. 

We compare our proposed method with baseline approaches on two math benchmarks: GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), utilizing fine-tuned models from Llemma7B (Azerbayev et al., 2023) and DeepSeek-7B (Shao et al., 2024) as the generators. Our results indicate that TSMC consistently improves both the quality of the generated solutions and the overall verification performance. Additionally, we empirically validate the theoretical advantage of TSMC as an unbiased estimator with reduced variance, further highlighting its effectiveness. 

Our main contributions can be summarized as follows: 

1. We propose a novel method based on TSMC that enhances the sampling efficiency of verification and reduces the reliance on process supervision obtained through human annotations or tree search in training verifiers. 

2. We introduce a new theoretical framework for analyzing verification methods, providing deeper insights into their effectiveness and limitations. 

3. Our empirical results demonstrate that TSMC consistently outperforms existing verification methods across multiple math benchmarks, utilizing various generators. 

## 2 PRELIMINARIES 

- 2.1 LLMS FOR MATH 

Following Lightman et al. (2024), we fix the generator without further fine-tuning via reinforcement learning. For a problem statement **x** 0, a (tokenized) candidate solution can be sampled from the 

2 

Under Review 

generator, denoted as **x** 1: _T ∼ p_ ( _·|_ **x** 0). For simplicity, we always assume the dependence on **x** 0 and no longer explicitly write it out in the following text. The solution is assumed to be decomposable as **x** 1: _T_ = [ **x** 1 _, · · · ,_ **x** _T_ ], where **x** _i_ is a variable-length reasoning step. By default, the LLM generates all steps in an autoregressive manner, i.e., **x** _t ∼ p_ ( _·|_ **x** 1: _t−_ 1). Each solution **x** 1: _T_ contains the reasoning process and an answer to the problem, with examples shown in Appendix F. We represent the extracted answer from the solution as _a_ = Ans( **x** 1: _T_ ), and its correctness as _ϕ_ ( _a_ ), which is 1 if it is correct (matched with the ground-truth answer) and 0 otherwise. 

The primary methods for solving math problems with LLMs include majority voting (Wang et al., 2023e) and verification (Cobbe et al., 2021; Uesato et al., 2022; Lightman et al., 2024). 

**Majority Voting.** Majority voting independently samples _N_ (tokenized) candidate solutions _{_ **x** _[i]_ 1: _T[}] i[N]_ =1[from][the][generator.][It][selects][the][final][answer][as][the][one][with][the][most][votes,][i.e.,] _N a[∗]_ = arg max _a_ � _i_ =1[I][(] _[a][i]_[=] _[ a]_[)][, where][ I][(] _[·]_[)][ is the indicator function.] 

**Verification.** Verification introduces an external verifier _r_ ( _·_ ) to evaluate the _N_ solutions produced by the LLM generator. Existing methods can be roughly divided into two kinds: the outcome reward model (ORM) family and the process reward model (PRM) family. ORM directly evaluates the confidence score for each full solution as _s_ = _rORM_ ( **x** 1: _T_ ), while PRM instead aggregates the confidence scores of sub-sequences as _s_ = _rP RM_ ( **x** 1: _T_ ) = Aggr( _{rP RM_ ( **x** _t|_ **x** 1: _t−_ 1) _}[T] t_ =1[)][.][Here,] _rP RM_ ( **x** _t|_ **x** 1: _t−_ 1) corresponds to the process reward, and Aggr( _·_ ) is the aggregation function such as the minimum or product: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0003-05.png)


The final answer could either be selected from the solution with the highest score _a[∗]_ = arg max _ai s[i] N_ (best-of- _N_ ), or the answer with the highest total weight _a[∗]_ = arg max _a_ � _i_ =1 _[s][i]_[I][(] _[a][i]_[=] _[a]_[)] (weighted majority voting) (Li et al., 2023b). In this work, we mainly develop our method on top of the weighted majority voting due to its empirical better performance (Sun et al., 2024). 

## 2.2 IMPORTANCE SAMPLING AND TWISTED SEQUENTIAL MONTE CARLO 

**Importance Sampling.** Consider a target distribution _σ_ ( **x** 1: _T_ ) = _σ_ ˜( **x** _Z_ 1: _[σ] T_ ) , where _σ_ ˜( **x** 1: _T_ ) _≥_ 0 is the unnormalized probability density and _Z[σ]_ = � **x** 1: _T[σ]_[˜][(] **[x]**[1:] _[T]_[ )] _[d]_ **[x]**[1:] _[T]_[is][the][normalizing][factor,] typically intractable. For a given function _h_ ( **x** 1: _T_ ), it could be difficult to estimate its expectation under _σ_ ( **x** 1: _T_ ) via directly sampling. Importance sampling (IS) (Robert & Casella, 2000) instead introduces a proposal distribution _q_ ( **x** 1: _T_ ) and provides an estimator of the expectation as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0003-09.png)


_σ_ ˜( **x** 1: _T_ ) Here, _q_ ( **x** 1: _T_ )[is][known][as][the] _[importance][weight][w]_[(] **[x]**[1:] _[T]_[ )][.][Using][some] _[q]_[(] **[x]**[1:] _[T]_[ )][that][is][easy][to] sample from, we can leverage Equation 2 to estimate the expectation via the Monte Carlo method, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0003-11.png)


Although ideally a zero variance of the importance weight could be achieved when _q_ ( **x** 1: _T_ ) = _σ_ ( **x** 1: _T_ ), such a case rarely holds in practice. Remarkably, the distribution mismatches at each step are accumulated as the generation proceeds, leading to an exponentially increasing variance with respect to _T_ (Doucet & Johansen, 2009). Such a limitation makes IS inefficient in the highdimensional space since extensive sampling is needed to reduce the variance. 

**Twisted Sequential Monte Carlo.** Twisted Sequential Monte Carlo (TSMC) enhances the sampling efficiency of IS by modifying the marginal distribution of the proposal, _q_ ( **x** 1: _t_ ), to a more informative intermediate distribution, _πt_ ( **x** 1: _t_ ). The aim is to ensure that partial sequences from _πt_ ( **x** 1: _t_ ) are more likely to result in higher-quality samples in the final target distribution _σ_ ( **x** 1: _T_ ). 

3 

Under Review 

_π_ ˜ _t_ ( **x** 1: _T_ ) finalHere,target _{πt}[T] t_ is=1aligned[is][known] with[as] _π_ ˜[the] _T ≡_[(twisted)] _σ_ ˜. In standard[intermediate] Sequential[targets] Monte[where] Carlo, _[π][t]_[(] **[x]**[1:] _π[T] t_ ([)] **x**[=] 1: _t_ ) is _Z_ typically _t[π]_ and thethe marginal of the target distribution _σ_ ( **x** 1: _T_ ), to ensure that at each time step, the marginal distribution matches the target. However, if our primary interest is only the final target _σ_ ( **x** 1: _T_ ), we are free to design _{πt}[T] t_ =1 _[ −]_[1][on the specific problem at hand, leading to the flexibility of the TSMC method.] 

TSMC operates recursively, alternating between generation and resampling. At each step, TSMC takes the input of _N_ partial sequences, _{_ **x** _[i]_ 1: _t−_ 1 _[}][N] i_ =1[,][following][the][distribution] _[π][t][−]_[1][(] **[x]**[1:] _[t][−]_[1][)][,][and] extends these sequences by sampling the next step from the proposal, i.e., **x** _t ∼ q_ ( _·|_ **x** 1: _t−_ 1). It computes the _incremental importance weight_ for each sequence as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0004-03.png)


These weights are used to approximate the distribution _πt_ by resampling the partial sequences from a categorical distribution with the self-normalized weights: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0004-05.png)


This new set of _N_ sequences would serve as the input to the next step of TSMC. With informative intermediate targets, the resampling step could promptly discard the sequences with a low potential in the target distribution and avoid a large variance in the importance weights. More importantly, since _πT_ ( **x** 1: _T_ ) is matched with the target _σ_ ( **x** 1: _T_ ), TSMC always yields an unbiased estimator of � **x** 1: _T[σ]_[˜][(] **[x]**[1:] _[T]_[ )] _[h]_[(] **[x]**[1:] _[T]_[ )][ regardless of the twist functions (Del Moral, 2004).] 

## 3 METHODOLOGY 

## 3.1 EXISTING VERIFICATION METHODS ARE PERFORMING IMPORTANCE SAMPLING 

The motivation of our method is based on the observation that existing verification methods are essentially performing IS. To see this, compare the normalized voting weight of each answer in majority voting and weighted majority voting when _N_ is large 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0004-10.png)


the importance weight in Equation 2.It can be seen that the weighting process actually introduces a factorIn particular, we can let _σ_ ˜( **x** 1: _T r_ )(= **x** 1: _pT_ ( ) **x** with a similar role of1: _T_ ) _r_ ( **x** 1: _T_ ) and treat weighted majority voting as IS to estimate the answer voting weight 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0004-12.png)


However, as described in Section 2.2, the importance weight in IS suffers from a large variance in the high-dimensional space, and so do the estimation objectives according to Proposition 3.1. 

**Proposition 3.1.** _For IS with the target σ_ ( **x** 1: _T_ ) _and proposal q_ ( **x** 1: _T_ ) _, up to a constant C independent of q_ ( **x** 1: _T_ ) _, the following identity in the variance holds for the set of all answers A:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0004-15.png)


We include the proof in Appendix A.1. This issue also accounts for Problem I, i.e., plenty of samples are needed to reduce the variance of the estimator. Therefore, we aim to address this problem via TSMC, which provides the unbiased estimator of _w_ ( _a_ ), but with less variance. We visualize the comparison between existing IS-based verification and our TSMC-based verification in Figure 1. 

4 

Under Review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0005-01.png)


Figure 1: IS-based verification vs. TSMC-based verification: (a) **Typical IS-based verification** only weights (verifies) the solutions until they are fully generated, which often leads to generating incorrect solutions with high probability, aka low sampling efficiency. **(b) Our TSMC-based verification** weights and resamples partial solutions at each step of the generation process. This sequential resampling process reduces the discrepancy between the proposal and target distributions, improving the overall correctness of the generated solutions and thus the sampling efficiency. 

## 3.2 VERIFICATION VIA TSMC 

The optimal reward model in our problem is simply the correctness function of each solution, i.e., _r[∗]_ ( **x** 1: _T_ ) = _ϕ_ (Ans( **x** 1: _T_ )). In the following section, we fix our target distribution as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0005-05.png)


since it corresponds to the actual target distribution we try to sample from. 

To utilize TSMC for verification, we still need to decide the proposal distribution _q_ ( **x** _t|_ **x** 1: _t−_ 1) and intermediate targets _{πt}[T] t_ =1 _[ −]_[1][.][Following][Zhao][et][al.][(2024),][we][define][the][intermediate][targets] through the twist functions _{ψt}[T] t_ =1 _[ −]_[1][where] _[ ψ][t]_[(] **[x]**[1:] _[t]_[)] _[ ≥]_[0][ are functions to be optimized] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0005-08.png)


Let _ψ_ 0( **x** 0) _≡_ 1 and _ψT_ ( **x** 1: _T_ ) _≡ ϕ_ (Ans( **x** 1: _T_ )), the incremental importance weight is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0005-10.png)


Zhao et al. (2024) have shown that the optimal proposal and intermediate targets correspond to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0005-12.png)


where _σ_ ( **x** 1: _t_ ) =[�] **x** _t_ +1: _T[σ]_[(] **[x]**[1:] _[T]_[ )][ stands for the target marginal distribution.][However, it is hard to] directly apply these optimal choices to our case. We outline the reason and our approach as follows. 

**Proposal.** There are two challenges in preventing us from using the optimal proposal. First, the combinatorial nature of the step **x** _t_ , which consists of multiple tokens, makes _qt[∗]_[(] **[x]** _[t][|]_ **[x]**[1:] _[t][−]_[1][)][gen-] erally intractable. Moreover, even if we can approximate it via finetuning, we still encounter the second challenge as the large variance in _qt[∗]_[(] **[x]** _[t][|]_ **[x]**[1:] _[t][−]_[1][)][caused][by][the][high][dimensionality][of] **[x]** _[t]_[.] 

5 

Under Review 

This would result in the _weight degeneracy_ issue (Naesseth et al., 2019) of TSMC, where the incremental importance weights would be dominated by a single sample, resulting in a poor diversity of solutions after resampling. We therefore simply let _q_ ( **x** _t|_ **x** 1: _t−_ 1) = _p_ ( **x** _t|_ **x** 1: _t−_ 1) for the ease of sampling and weight degeneracy would be alleviated when _q_ ( **x** _t|_ **x** 1: _t−_ 1) cancels in Equation 12. 

**Intermediate targets.** The intermediate targets in Equation 13 is no longer optimal when we use _p_ ( **x** _t|_ **x** 1: _t−_ 1) as our proposal. However, it is also hard to solve the globally optimal intermediate targets for an arbitrary proposal. We instead seek to sequentially derive the locally optimal twists in a greedy manner. Since our ultimate goal is to estimate the answer weights _w_ ( _a_ ), we start by looking for the optimal intermediate target _πT[∗] −_ 1[(] **[x]**[1:] _[T][ −]_[1][)][in][minimizing][the][variance][of][the][incremental] importance weight in the last TSMC step. We prove the following proposition in Appendix A.2. 

**Proposition 3.2.** _Given an (intermediate) target πt_ ( **x** 1: _t_ ) _and the proposal q_ ( **x** _t|_ **x** 1: _t−_ 1) _, the optimal πt−_ 1( **x** 1: _t−_ 1) _in minimizing the variance of the incremental importance weight corresponds to_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0006-04.png)


Taking _t_ = _T_ and _q_ = _p_ implies _πT[p] −_ 1[(] **[x]**[1:] _[T][ −]_[1][)] _[ ∝][p]_[(] **[x]**[1:] _[T][ −]_[1][)] �� **x** _T[p]_[(] **[x]** _[T][ |]_ **[x]**[1:] _[T][ −]_[1][)] _[ϕ]_[(][Ans][(] **[x]**[1:] _[T]_[ ))][.] Here we apply the fact that _ϕ_ (Ans( **x** 1: _T_ ))[2] = _ϕ_ (Ans( **x** 1: _T_ )) as it is binary. If we fix the intermediate target as the choice above, we could further propagate the derivation to previous steps by recursively applying Proposition 3.2, getting the locally optimal intermediate targets for _t < T_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0006-06.png)


In particular,[�] **x** _t_ +1: _T[p]_[(] **[x]** _[t]_[+1:] _[T][ |]_ **[x]**[1:] _[t]_[)] _[ϕ]_[(][Ans][(] **[x]**[1:] _[T]_[ ))][ actually represents the value function] _[ V][p]_[(] **[x]**[1:] _[t]_[)] in reinforcement learning (Ouyang et al., 2022). Hence, the locally optimal twists are given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0006-08.png)


## 3.3 CONNECTION WITH THE PRM 

Based on our above choices, the incremental importance weights in Equation 12 becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0006-11.png)


The incremental importance weight could also be treated as a measurement of the step quality, similar to the process reward in the PRM. To further augment this connection, note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0006-13.png)


which is in the same format as the PRM with prod aggregation. The key observation here is that TSMC always yields an unbiased estimator of the importance weight _ϕ_ (Ans( **x** 1: _T_ )) when there is no estimation error of _V[p]_ . We continue to compare this estimator to some existing PRMs. 

**The PRM learnt through automatic supervision.** This class of PRMs (Wang et al., 2023b; Luo et al., 2024) computes the process reward by evaluating the value function at each partial solution with respect to a roll-out policy _µ_ . The solution confidence score will be computed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0006-16.png)


But such an estimator is always biased no matter min or prod is used for aggregation. 

6 

Under Review 

**The PRM learnt through human supervision.** The human supervision is generated through the logical sense of the step correctness. We formally establish the definition of the step correctness in Definition A.1 and prove the following proposition in Appendix A.3. 

**Proposition 3.3.** _The ground-truth PRM over the step correctness corresponds to_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0007-03.png)


Therefore, the solution confidence score of this PRM is always an unbiased estimator of _ϕ_ (Ans( **x** 1: _T_ )) for both min and prod aggregation. However, using I( _σ_ ( **x** 1: _t_ ) _>_ 0) for the intermediate target, as tried by Uesato et al. (2022), does not effectively reduce the sampling variance or improve the verification performance since it ignores the likelihood of the proposal _p_ ( **x** _t_ +1: _T |_ **x** 1: _t_ ). 

## 3.4 VALUE FUNCTION ESTIMATION 

The approximation of _{ψt[p][}][T] t_ =1 _[ −]_[1][and] _[ r][∗]_[can be consolidated into a single learning task:][estimating] the value function _V[p]_ . We therefore use a single neural model parameterized by _θ_ for approximation. Estimating the value function through independently sampled data from the policy (generator) is a well-studied topic (Bertsekas, 2012). It therefore eliminates the need for explicit process supervision during training, as outlined in Problem II. 

In this paper, we adopt the Contrastive Twist Learning (CTL) method developed by Zhao et al. (2024). Directly approximating _πt[p]_[would][be][hard,][so][we][still][approximate][the][target][marginal] _σ_ ( **x** 1: _t_ ) to learn the value function and take the square root of the value function during the inference time. Let _V[θ]_ be our approximation of the value function _V[p]_ , and define the intermediate _[θ]_[(] **[x]**[1:] _[t]_[)] target _πt[θ]_[(] **[x]**[1:] _[t]_[)][=] _[p]_[(] **[x]** _Z_[1:] _t[t][π]_[)][(] _[V]_ **[x]**[1:] _[t]_[)] . CTL minimizes the KL divergence between the target marginal distributions and the intermediate targets, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0007-08.png)


whose gradient at _t_ -th step can be derived as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0007-10.png)


We approximate the gradient in the first term via rejection sampling while the gradient in the second term via importance sampling, as done in Equation 3. We include more training details in Appendix C.2 and summarize our entire TSMC-based verification algorithm in Appendix B. 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL SETUP 

We outline the basic experimental setup in this section and include more details in Appendix C. 

**Datasets.** Building on prior work (Uesato et al., 2022; Lightman et al., 2024; Wang et al., 2023b), we assess our TSMC method using two widely used math datasets: GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). For GSM8K, we evaluate model performance across all testing instances. Whereas for math, we follow Lightman et al. (2024) to select a representative subset of 500 testing instances, referred to as “MATH500” in the following text. 

**Generators.** We fine-tune our solution generators using two different pretrained LLMs, Llemma7B (Azerbayev et al., 2023) and DeepSeek-7B (Shao et al., 2024). Following Sun et al. (2024), we use the filtered PRM800K (Lightman et al., 2024) as the supervised fine-tuning dataset. 

**Baselines.** We compare our method to both the non-verification method using majority voting (MV) (Wang et al., 2023e) and the verification methods using weighted majority voting (WMV). For verification methods, we utilize various types of verifiers, including the ORM, the PRM trained with human supervision on PRM800K (Lightman et al., 2024)) and the PRM trained with automatic 

7 

Under Review 

supervision on MATH-SHEPHERD (Wang et al., 2023b). We employ min for aggregation on both PRMs. We keep the architecture and pretrained weights of all the verifiers, including the reward model and the parameterized value function, the same as the generator across all settings. The problem solving rate (in %) is used as the metric for comparison. 

**TSMC details.** Our TSMC is applied on the step level. We implement a warm-up stage that skips resampling in the initial stage, setting this threshold at 50 tokens across all experiments. A maximum of five resampling steps is allowed to reduce the latency. For sequences that terminate early, we assign an incremental importance weight of 1 during the remaining resampling steps. We employ stratified sampling (Kitagawa, 1996) for resampling to reduce the variance. Instead of resampling across the full batch of _N_ solutions, we perform resampling over a mini-batch with _M_ samples. The batch size _M_ is fixed as 80 (the maximum number of sequences fit into our GPUs) by default. 

## 4.2 MAIN RESULTS 

To verify if TSMC actually improves the sampling efficiency with a better solution quality, we also directly perform the majority voting on the solutions generated by TSMC. We denote this method as TSMC + MV and the full TSMC as TSMC + WMV. We present our main results in Table 1. 

Table 1: Comparative results in the problem solving rate (%) on GSM8K and MATH500 datasets. Llemma-7B and DeepSeek-8B are used as generators. We bold the best results in each category. The voting is performed on 240 samples. 

|is performed on|240 samples.|
|---|---|
|**Generators**|**Methods**<br>**GSM8K**<br>**MATH500**|
|Llemma-7B|MV<br>72.5<br>41.2<br>WMV w. ORM<br>78.7<br>43.0<br>WMV w. PRM (PRM800K)<br>73.6<br>43.2<br>WMV w. PRM(SHEPHERD)<br>79.2<br>43.6|
||TSMC + MV (Ours)<br>77.6<br>44.0<br>TSMC + WMV (Ours)<br>**80.6**<br>**45.6**|
|DeepSeek-7B|MV<br>86.4<br>52.8<br>WMV w. ORM<br>86.6<br>55.0<br>WMV w. PRM (PRM800K)<br>87.0<br>55.2<br>WMV w. PRM(SHEPHERD)<br>89.5<br>52.6|
||TSMC + MV (Ours)<br>88.7<br>54.8<br>TSMC + WMV (Ours)<br>**90.8**<br>**56.2**|



It is evident that TSMC + MV demonstrates a significant improvement over vanilla MV, highlighting its effectiveness in enhancing the overall solution quality. This advantage is particularly pronounced when using the relatively weaker generator Llemma-7B, which corresponds to cases where the proposal distribution deviates significantly from the target. Moreover, TSMC consistently outperforms other methods in terms of final verification performance. It is worth noting that the final verification step in TSMC operates independently of the generator, meaning that a better reward model could further improve TSMC’s performance in WMV. Overall, our TSMC-based verification method shows a clear advantage over existing verification methods, with a simpler training target. 

## 4.3 IMPACT OF THE BIAS IN THE ESTIMATOR 

TSMC is characterized by its unbiased estimation of the importance weight, which is _ϕ_ (Ans( **x** 1: _T_ )) in our task. However, since the training error is unavoidable in practice, it remains unclear whether such kind of unbiased estimation at the theoretical optimal case is useful. We thus look into this problem by comparing different biased and unbiased estimators analyzed in Section 3.3. 

We consider both the PRM predicting the step correctness, PRM (PRM800K), and the PRM predicting the value function, including PRM (SHEPHERD) and the value function estimated in TSMC. Beyond the min and prod aggregations, we also consider an unbiased ORM-like strategy that only 

8 

Under Review 

uses the process reward at the last step as the solution score: 

last = _rP RM_ ( **x** _T |_ **x** 1: _T −_ 1) _._ (23) 

Exceptionally, we evaluate the value function of TSMC on the data generated by TSMC, so we use the last incremental importance weight for last in this scenario as we do in our original TSMC method. This is equivalent to the product of incremental importance weights, which is also an unbiased estimator. We compare all estimators in Figure 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0009-04.png)


Figure 2: Comparison among all biased and unbiased estimators of the importance weight. 

The trend is highly consistent when the value function is used as the process reward, as in PRM (SHEPHERD) and Value (TSMC). The prod strategy shows an overall inferior performance across all settings. Since the step value could be small, the product of step values is highly biased towards the solutions with fewer steps. In contrast, min could overcome such a bias as its value is insensitive to the number of steps. It also shows a clear advantage in comparison to prod, in line with the choice of (Wang et al., 2023b). However, its performance is still consistently worse than the unbiased estimator using last. A different pattern shows up in PRM (PRM800K), where the step correctness is used as the process reward. The min strategy still achieves the overall best result, but prod is also comparably good. We find no advantage of last in this case as all three estimators are unbiased. Instead, prod and min would benefit from less modeling error with its ensemble classifier. 

We find our results consistent with the observation from Sun et al. (2024). Basically, the advantage of PRM against ORM holds only when both of them are unbiased estimators of _ϕ_ (Ans( **x** 1: _T_ )). When the PRM is biased, there is no clear guarantee of a better performance against the ORM, which is always an unbiased estimator. While TSMC assimilates the strengths of the unbiased estimation from the ORM and the intermedidate step modeling from the PRM, leading to the best performance. 

## 4.4 IMPACT OF THE VARIANCE IN THE ESTIMATOR 

Besides of being unbiased, TSMC reduces the variance of the importance weight via informative twist functions. To investigate the impact of the variance, we consider the following TSMC variants: using the step correctness predicted by PRM (PRM800K) as the incremental importance weight; using the process reward in PRM (SHEPHERD) for the estimation of the value function; and using _V[p]_ ( **x** 1: _t_ ) rather than � _V[p]_ ( **x** 1: _t_ ) for the twist, which approximates the target marginal _σ_ ( **x** 1: _t_ ) as the intermediate target. Using Llemma-7B on MATH500 as the example, we examine the performance of these variants in Figure 3. 

As observed by Uesato et al. (2022), using the step correctness in the intermediate decoding does not bring any improvement since the likelihood from the generator is ignored. The value function provided by PRM (SHEPHERD) is more informative and leads to a better performance. But since the value function is evaluated on a different generator _µ_ , it still achieves a worse performance than our TSMC using the approximated _V[p]_ as the twist. Finally, using _V[p]_ ( **x** 1: _t_ ) as the twist also leads to an inferior performance than using ~~�~~ _V[p]_ ( **x** 1: _t_ ), indicating the necessity to optimize the variance. 

9 

Under Review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0010-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0010-02.png)


Figure 3: TSMC with different intermediate targets. Variance are visualized across many subsamples of the 240 solutions per problem. 

Figure 4: Ablation study on the TSMC batch size. Variance are visualized across many subsamples of the 240 solutions per problem. 

## 4.5 SENSITIVITY ANALYSIS OF THE TSMC BATCH SIZE 

In above experiments, we use the maximum number of samples that can fit into the memory of our GPUs as the batch size _M_ of TSMC. In practice, a smaller memory and or a larger model would lead to a smaller TSMC batch size. We thus try to investigate the effect of _M_ on the TSMC performance by varying _M_ over the values in _{_ 10 _,_ 20 _,_ 40 _,_ 80 _}_ . The comparative results are shown in Figure 4. It can be seen that smaller _M_ achieves comparable and even better results. The advantage of TSMC still clearly holds when _M_ = 10, which is a batch size that can be fit into most GPUs. 

## 5 RELATED WORK 

**Verification for reasoning.** Verification has proven to be an effective approach for enhancing the multi-step reasoning ability of LLMs. Two widely adopted verification methods are the Outcome Reward Model (ORM) (Cobbe et al., 2021) and the Process Reward Model (PRM) (Uesato et al., 2022). While empirical evidence suggests that PRM outperforms ORM (Lightman et al., 2024), training the PRM presents a significant challenge due to the need for process supervision, which is often difficult to obtain. Recent research has therefore increasingly focused on automatic supervision to train PRMs more efficiently (Wang et al., 2023d; Luo et al., 2024; Wang et al., 2024). 

**(Twisted) Sequential Monte Carlo.** Sequential Monte Carlo (SMC) is a generic statistical inference approach that has been widely applied across various domains, including signal processing (Doucet & Johansen, 2009; Simon J Godsill & West, 2004), financial econometrics (Johannes & Polson, 2010; Creal, 2012), and robotics (Montemerlo et al., 2002; Bailey & Durrant-Whyte, 2006; Thrun et al., 2005). Recently, SMC has been integrated with neural models to enhance sequential generative models, such as diffusion models (Trippe et al., 2023; Wu et al., 2023) and LLMs (Lew et al., 2023; Zhao et al., 2024). The most relevant work to ours is Zhao et al. (2024), which presents a general framework for controlled text generation using Twisted Sequential Monte Carlo (TSMC). Our work primarily focuses on multi-step reasoning, one of the most challenging areas for LLMs. Additionally, we are the first to bridge TSMC with the predominant verification methods, offering a novel theoretical perspective for explainability. 

## 6 CONCLUSION & LIMITATION 

In this paper, we introduce a novel verification method for multi-step reasoning using Twisted Sequential Monte Carlo (TSMC). Our approach sequentially approximates intermediate targets, enhancing the reasoning process of large language models and improving both solution quality and sampling efficiency. By incorporating step-wise guidance without human supervision for training, our method provides a scalable framework for various multi-step reasoning tasks. 

Although promising, our method also introduces the additional latency in the inference time due to variable step lengths. A potential optimization involves blockwise resampling over a fixed number of tokens. Future work could also explore the impact of TSMC batch size and refine algorithmic design for further efficiency gains. 

10 

Under Review 

## REFERENCES 

- Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. _arXiv preprint arXiv:2310.10631_ , 2023. 

- T. Bailey and H. Durrant-Whyte. Simultaneous localization and mapping (slam): part ii. _IEEE Robotics & Automation Magazine_ , 13(3):108–117, 2006. doi: 10.1109/MRA.2006.1678144. 

- D. Bertsekas. _Dynamic Programming and Optimal Control: Volume II; Approximate Dynamic Programming_ . Athena Scientific optimization and computation series. Athena Scientific, 2012. ISBN 9781886529441. URL https://books.google.com/books?id=C1JEEAAAQBAJ. 

- Mark Briers, Arnaud Doucet, and Simon Maskell. Smoothing algorithms for state–space models. _Annals of the Institute of Statistical Mathematics_ , 62:61–89, 02 2009. doi: 10.1007/ s10463-009-0236-2. 

- Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. Alphamath almost zero: process supervision without process, 2024. URL https://arxiv.org/abs/2405.03553. 

- Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang, Tianyu Liu, and Baobao Chang. Towards end-to-end embodied decision making via multi-modal large language model: Explorations with gpt4-vision and beyond. _arXiv preprint arXiv:2310.02071_ , 2023. 

- N. Chopin and O. Papaspiliopoulos. _An Introduction to Sequential Monte Carlo_ . Springer Series in Statistics. Springer International Publishing, 2020. ISBN 9783030478452. URL https: //books.google.com/books?id=ZZEAEAAAQBAJ. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- R´emi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In _International conference on computers and games_ , pp. 72–83. Springer, 2006. 

- Drew Creal. A survey of sequential monte carlo methods for economics and finance. _Econometric Reviews_ , 31(3):245–296, 2012. doi: 10.1080/07474938.2011.607333. URL https://doi. org/10.1080/07474938.2011.607333. 

- Pierre Del Moral. _Feynman-Kac Formulae: Genealogical and Interacting Particle Systems With Applications_ , volume 100. 05 2004. ISBN 0387202684. doi: 10.1007/978-1-4684-9393-1. 

- Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential Monte Carlo Samplers. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 68(3):411–436, 05 2006. ISSN 1369-7412. doi: 10.1111/j.1467-9868.2006.00553.x. URL https://doi.org/10.1111/ j.1467-9868.2006.00553.x. 

- A. Doucet, A. Smith, N. de Freitas, and N. Gordon. _Sequential Monte Carlo Methods in Practice_ . Information Science and Statistics. Springer New York, 2001. ISBN 9780387951461. URL https://books.google.com/books?id=uxX-koqKtMMC. 

- Arnaud Doucet and Adam Johansen. A tutorial on particle filtering and smoothing: Fifteen years later. _Handbook of Nonlinear Filtering_ , 12, 01 2009. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021. 

- Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. _arXiv preprint arXiv:2310.01798_ , 2023. 

11 

Under Review 

- Michael Johannes and Nicholas Polson. Chapter 13 - mcmc methods for continuous-time financial econometrics. In YACINE A[¨] IT-SAHALIA and LARS PETER HANSEN (eds.), _Handbook of Financial Econometrics: Applications_ , volume 2 of _Handbooks in Finance_ , pp. 1–72. Elsevier, San Diego, 2010. doi: https://doi.org/10.1016/B978-0-444-53548-1.50003-9. URL https: //www.sciencedirect.com/science/article/pii/B9780444535481500039. 

- Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. Challenges and applications of large language models. _arXiv preprint arXiv:2307.10169_ , 2023. 

- Genshiro Kitagawa. Monte carlo filter and smoother for non-gaussian nonlinear state space models. _Journal of Computational and Graphical Statistics_ , 5(1):1–25, 1996. doi: 10. 1080/10618600.1996.10474692. URL https://www.tandfonline.com/doi/abs/ 10.1080/10618600.1996.10474692. 

- Levente Kocsis and Csaba Szepesv´ari. Bandit based monte-carlo planning. In _European conference on machine learning_ , pp. 282–293. Springer, 2006. 

- Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, and Vikash K. Mansinghka. Sequential monte carlo steering of large language models using probabilistic programs, 2023. URL https:// arxiv.org/abs/2306.03081. 

- Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. M3it: A large-scale dataset towards multi-modal multilingual instruction tuning. _arXiv preprint arXiv:2306.04387_ , 2023a. 

- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Anna Rogers, Jordan BoydGraber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5315–5333, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.291. URL https://aclanthology.org/2023.acl-long.291. 

- Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview. net/forum?id=v8L0pN6EOi. 

- Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, and Abhinav Rastogi. Improve mathematical reasoning in language models by automated process supervision, 2024. URL https://arxiv.org/abs/2406. 06592. 

- Michael Montemerlo, Sebastian Thrun, Daphne Koller, and Ben Wegbreit. Fastslam: a factored solution to the simultaneous localization and mapping problem. In _Eighteenth National Conference on Artificial Intelligence_ , pp. 593–598, USA, 2002. American Association for Artificial Intelligence. ISBN 0262511290. 

- Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, Jilin Chen, Alex Beutel, and Ahmad Beirami. Controlled decoding from language models. In _Forty-first International Conference on Machine Learning_ , 2024. URL https://openreview.net/forum?id=bVIcZb7Qa0. 

- Christian Naesseth, Fredrik Lindsten, and Thomas Sch¨on. _Elements of Sequential Monte Carlo_ . 01 2019. ISBN 978-1-68083-632-5. doi: 10.1561/9781680836332. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155. 

12 

Under Review 

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology_ , pp. 1–22, 2023. 

- Christian Robert and George Casella. Monte carlo statistical method. _Technometrics_ , 42, 11 2000. doi: 10.2307/1270959. 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, and Daya Guo Y. Wu. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300. 

- Jianhao Shen, Yichun Yin, Lin Li, Lifeng Shang, Xin Jiang, Ming Zhang, and Qun Liu. Generate & rank: A multi-task framework for math word problems. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pp. 2269–2279, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.195. URL https://aclanthology.org/2021.findings-emnlp.195. 

- David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. _nature_ , 529(7587):484–489, 2016. 

- Arnaud Doucet Simon J Godsill and Mike West. Monte carlo smoothing for nonlinear time series. _Journal of the American Statistical Association_ , 99(465):156–168, 2004. doi: 10.1198/ 016214504000000151. URL https://doi.org/10.1198/016214504000000151. 

- Yifan Song, Weimin Xiong, Dawei Zhu, Cheng Li, Ke Wang, Ye Tian, and Sujian Li. Restgpt: Connecting large language models with real-world applications via restful apis. corr, abs/2306.06624, 2023. doi: 10.48550. _arXiv preprint arXiv.2306.06624_ . 

- Zhiqing Sun, Longhui Yu, Yikang Shen, Weiyang Liu, Yiming Yang, Sean Welleck, and Chuang Gan. Easy-to-hard generalization: Scalable alignment beyond human supervision. _arXiv preprint arXiv:2403.09472_ , 2024. 

- Maciej Swiechowski, Konrad Godlewski, Bartosz Sawicki, and Jacek Ma´ndziuk.[´] Monte carlo tree search: A review of recent modifications and applications. _Artificial Intelligence Review_ , 56(3): 2497–2562, 2023. 

- S. Thrun, W. Burgard, and D. Fox. _Probabilistic Robotics_ . Intelligent Robotics and Autonomous Agents series. MIT Press, 2005. ISBN 9780262201629. URL https://books.google. com/books?id=2Zn6AQAAQBAJ. 

- Brian L. Trippe, Jason Yim, Doug Tischer, David Baker, Tamara Broderick, Regina Barzilay, and Tommi S. Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motifscaffolding problem. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=6TxBxqNME1Y. 

- Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. _arXiv preprint arXiv:2211.14275_ , 2022. 

- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. _arXiv preprint arXiv:2305.16291_ , 2023a. 

- Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, and Zhifang Sui. Making large language models better reasoners with alignment. _arXiv preprint arXiv:2309.02144_ , 2023b. 

- Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. _arXiv preprint arXiv:2305.17926_ , 2023c. 

13 

Under Review 

- Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. _CoRR, abs/2312.08935_ , 2023d. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023e. URL https://openreview.net/forum?id=1PL1NIMMrw. 

- Zihan Wang, Yunxuan Li, Yuexin Wu, Liangchen Luo, Le Hou, Hongkun Yu, and Jingbo Shang. Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision, 2024. URL https://arxiv.org/abs/2402.02658. 

- Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. _arXiv preprint arXiv:2306.01693_ , 2023. 

- Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Baker Grosse. Probabilistic inference in language models via twisted sequential Monte Carlo. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 60704–60748. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/zhao24c.html. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _arXiv preprint arXiv:2306.05685_ , 2023. 

## A PROOFS 

## A.1 PROOF FOR ANSWER WEIGHT VARIANCE 

**Proposition 3.1.** _For IS with the target σ_ ( **x** 1: _T_ ) _and proposal q_ ( **x** 1: _T_ ) _, up to a constant C independent of q_ ( **x** 1: _T_ ) _, the following identity in the variance holds for the set of all answers A:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0014-10.png)


_Proof._ For simplicity, denote _fa_ ( **x** 1: _T_ ) = I(Ans( **x** 1: _T_ ) = _a_ ). Using the fact that _fa_ ( **x** 1: _T_ )[2] = _fa_ ( **x** 1: _T_ ) and[�] _a∈A[f][a]_[(] **[x]**[1:] _[T]_[ ) = 1][, we have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0014-12.png)


Here, _C_ = ([�] **x** 1: _T[σ]_[˜][(] **[x]**[1:] _[T]_[ ))][2] _[ −]_[�] _a∈A_[(][�] **x** 1: _T[σ]_[˜][(] **[x]**[1:] _[T]_[ )] _[f][a]_[(] **[x]**[1:] _[T]_[ ))][2][ is independent of] _[ q]_[(] **[x]**[1:] _[T]_[ )][.] 

14 

Under Review 

## A.2 PROOF FOR THE LOCALLY OPTIMAL INTERMEDIATE TARGET 

**Proposition 3.2.** _Given an (intermediate) target πt_ ( **x** 1: _t_ ) _and the proposal q_ ( **x** _t|_ **x** 1: _t−_ 1) _, the optimal πt−_ 1( **x** 1: _t−_ 1) _in minimizing the variance of the incremental importance weight corresponds to_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-04.png)


_πt_ ( **x** 1: _t_ ) Minimizing the variance is thus equivalent to minimizing E _qπt−_ 1[( _p_ ( **x** _t|_ **x** 1: _t−_ 1) _πt−_ 1( **x** 1: _t−_ 1)[)][2][]][.][Sub-] ject to the constraint of the probability, we introduce the Lagrange multiplier _λ_ in our objective 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-06.png)


_·_ ( ) Taking _πt−_ 1( **x** 1: _t−_ 1)[= 0][, we get] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-08.png)


This yields the optimal _πt[q] −_ 1[given] _[ q]_[ as] 

where _λ_ is chosen to normalize the densities to have a sum of 1. Especially, when _πt_ and q correspond to the optimal choices in Equation 13, we recover 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-11.png)


A.3 PROOF FOR THE GROUND-TRUTH PRM 

**Definition A.1.** _Each step_ **x** _t|_ **x** 1: _t−_ 1 _is either correct or incorrect, following the two axioms below:_ 

• _For any solution_ **x** 1: _T , if_ **x** _t|_ **x** 1: _t−_ 1 _is correct for t_ = 1 _, · · · , T , then ϕ_ ( _Ans_ ( **x** 1: _T_ )) = 1 _._ 

• _For any solution_ **x** 1: _T , if any step_ **x** _t|_ **x** 1: _t−_ 1 _is incorrect, then ϕ_ ( _Ans_ ( **x** 1: _T_ )) = 0 _._ **Proposition 3.3.** _The ground-truth PRM over the step correctness corresponds to_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-16.png)


_Proof._ According to Definition A.1, a step would be correct if there is at least one correct solution contains it, and incorrect otherwise. So its process reward corresponds to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0015-18.png)


Here we treat all steps as incorrect if they follow an incorrect step. While in Lightman et al. (2024), a step could still be labeled as correct even its prior steps are incorrect. This is because the logical thinking is not always in a linear dependency, i.e., a future step is not necessarily dependent on all steps prior to it. How to label these steps is an inductive bias, which does not affect the solution score at the theoretical optimal case. 

15 

Under Review 

## B PSEUDOCODE FOR TSMC 

Here we summarize the pseudocode for our TSMC-based verification method. 

**Algorithm 1** TSMC for Verification 1: **Input** : Generator _p_ , estimated value function _V[θ]_ 2: **for** _t_ = 1 _, . . . , T_ **do** 3: **for** _i_ = 1 _, · · · , N_ **do** 4: # Sample the next step 5: **x** _[i] t[∼][p]_[(] _[·|]_ **[x]** _[i]_ 1: _t−_ 1[)] 6: **x** _[i]_ 1: _t[←]_[CONCAT][(] **[x]** _[i]_ 1: _t−_ 1 _[,]_ **[ x]** _[i] t_[)] 7: # Evaluate the incremental importance weight 8: **if** _t < T_ **then** _V[θ]_ ( **x** _[i]_ 1: _t_[)] 9: _w_ ( **x** _[i]_ 1: _t_[)] _[ ←] V[θ]_ ( **x** _[i]_ 1: _t−_ 1[)] � 10: **else** _V[θ]_ ( **x** _[i]_ 1: _T_[)] 11: _w_ ( **x** _[i]_ 1: _T_[)] _[ ←]_ ~~_√_~~ _V[θ]_ ( **x** _[i]_ 1: _T −_ 1[)] 12: **end if** 13: **end for** 14: **if** _t < T_ **then** 15: **for** _i_ = 1 _, · · · , N_ **do** 16: # Resample the sequences 17: _ω[i] ∼_ Cat( _{_ � _Njw_ =1 _t_ ( _[w]_ **x** _[t][i]_ 1:[(] _t_ **[x]**[)] 1: _[j] t_[)][)] _[}] i[N]_ =1[)] 18: **x** _[i]_ 1: _t[←]_ **[x]** _[ω]_ 1: _[i] t_ 19: **end for** 20: **end if** 21: **end for** 22: # Create a dictionary to store the voting weight 23: _W ←{}_ 24: **for** _i_ = 1 _, · · · , N_ **do** 25: # Extract the answer 26: _a[i] ←_ Ans( **x** _[i]_ 1: _T_[)] 27: **if** _a[i] ∈ W_ **then** 28: # Update the answer voting weight 29: _W_ [ _a[i]_ ] _← W_ [ _a[i]_ ] + _w_ ( **x** _[i]_ 1: _T_[)] 30: **else** 31: _W_ [ _a[i]_ ] _← w_ ( **x** _[i]_ 1: _T_[)] 32: **end if** 33: **end for** 34: # Majority voting 35: **return** : arg max _a W_ [ _a_ ] 

## C ADDITIONAL EXPERIMENTAL DETAILS 

All our experiments, including training and inference, are conducted on a single machine with 8 H100 GPUs. The summary of the model hyperparameters is presented in Table 2, and we include individual details as below. 

## C.1 THE GENERATOR 

We follow Sun et al. (2024) to fine-tune the generators on a filtered subset from PRM800K (Lightman et al., 2024). The hyperparameters are kept the same across the fine-tuning over Llemma-7B (Azerbayev et al., 2023) and DeepSeek-7B (Shao et al., 2024). The generators are fixed once the supervised fine-tuning is over and no additional reinforcement learning is applied. 

16 

Under Review 

Table 2: The summary of training hyperparameters for all models. 

||Generator|Value|ORM|PRM (PRM800k)|PRM (SHEPHERD)|
|---|---|---|---|---|---|
|||||||
|Learning rate|2_×_10_−_5|10_−_5|2_×_10_−_5|2_×_10_−_5|2_×_10_−_5|
|||||||
|Batch size|128|80|128|128|128|
|||||||
|# Epochs|3|2|2|2|2|
|||||||
|Warmup ratio|0.2|0.05|0.2|0.2|0.2|
|||||||
|Max. length|768|1024|1024|1024|1024|
|Dtype<br>BF16||||||



During the inference time, we generate the solution using top- _K_ sampling with _K_ = 20 and set the temperature as 0 _._ 7. The maximum length of the solution is fixed as 768. 

## C.2 THE VALUE FUNCTION 

For each math problem in the training dataset, we generate _B_ = 80 solutions independently with the generator and we only keep them for training if at least one solution is correct. The inference hyperparameters for the generator are kept the same as above. We use the same way to create the validation set using 500 validation instances. 

We apply the CTL loss (Zhao et al., 2024) on the step-level. The steps are separated by double newline indicators, i.e., _\_ n _\_ n, and the value function is trained on the token corresponding to the second newline indicator, along with the end of sentence token <eos>. Since the CTL loss is computed over all solutions to a single problem, we fill each training batch with all 80 samples collected from that problem. 

Recall that the gradient of the CTL loss at _t_ -th step is given as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0017-08.png)


We approximate the gradient in the first term via rejection sampling while the gradient in the second term via IS. The first term is approximated as[�] _[B] i_ =1 � _Bjϕ_ =1(Ans _[ϕ]_[(][Ans] ( **x** _[i]_ 1:[(] _T_ **[x]**[))] _[j]_ 1: _T_[))] _[∇][θ]_[ log] _[ V][θ]_[(] **[x]** _[i]_ 1: _t_[)][.] In the second term, we first compute the importance weight via the current approximated value function as _wt[θ]_[(] **[x]**[1:] _[t]_[)] = _VV[θ]_ ( _[θ]_ **x** ( **x** _[j]_ 1: _[i]_ 1: _t−t_[)] 1[)][,][then][we][approximate][the][expected][gradient][via][IS][as] � _Bi_ =1 � _Bjw_ =1 _t[θ]_[(] _[w]_ **[x]** _t[θ][i]_ 1:[(] _t_ **[x]**[)] _[j]_ 1: _t_[)] _[∇][θ]_[ log] _[ V][θ]_[(] **[x]** _[i]_ 1: _t_[)][.][Therefore, we can approximate the gradient of] _[ θ]_[ on the train-] ing problems as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0017-10.png)


## C.3 THE ORM 

To ensure a fair comparison, the ORM is trained on the same data used to train and validate our value function, but with a different data processing strategy and training method. 

We basically follow the same procedure in Cobbe et al. (2021) to train the ORM. We balance the positive and negative samples in the dataset by selecting the same number of correct and incorrect solutions per problem. The ORM is trained with the binary cross-entropy loss on each token while only the last token is used for prediction during the inference time. 

## C.4 THE PRM 

We use PRM800K (Lightman et al., 2024) and MATH-SHEPHERD (Wang et al., 2023b) datasets to train two PRMs separately. Especially, we use the PRM800K data to train the PRM once and 

17 

Under Review 

apply it on both GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) datasets. While for MATH-SHEPHERD, it consists of the samples from both the GSM8K and MATH datasets, and we use the corresponding portion to train two PRMs separately. A validation set of 4096 samples is held from the training set of each model. We apply the binary cross entropy loss on the second newline token of a step and the last token of each sample. 

## D COMPARISON WITH GREEDY DECODING ALGORITHMS 

Although TSMC yields an unbiased estimation of the importance weight with less variance, the ultimate goal in reasoning tasks is simply to generate the solution with the highest correctness probability. In case we have a perfect estimation of the value function, we could greedily select the partial sequence with the highest value and continue the search since then, which should in theory give the highest chance in reaching the correct answer (Mudgal et al., 2024). While in practice, the training error is unavoidable and estimation of the value function is always imperfect, which makes the sampling necessary. In this section, we compare our TSMC method with some greedy approaches to verify the necessity of sampling. 

The main baseline we consider is the Step-Level Beam Search (SBS) (Chen et al., 2024). At each step, SBS selects the top- _K_ samples with the highest values from _M_ parallel decoded sequences. It then clones each sample for _[M] K_[times][to][continue][the][search][in][the][next][step.][It][is][also][shown] to outperform the Monte Carlo Tree Search method (Kocsis & Szepesv´ari, 2006; Coulom, 2006; Silver et al., 2016; Swiechowski et al., 2023) in both the efficiency and solving rate.[´] Following the parameters in Chen et al. (2024), we set _K_ = 8, and fix _M_ = 80 as in our TSMC method. Besides, we also consider another variant of TSMC in selecting the top- _K_ incremental importance weights and clone them for _[M]_[This method is named as TSMC (greedy).] _K_[times during the resampling stage.] We fix the value function across all methods and compare TSMC to these variants in Figure 5. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo_images/iclr_2025_Step-by-Step_Reasoning_for_Math_Problems_via_Twisted_Sequential_Monte_Carlo.pdf-0018-05.png)


Figure 5: Comparison to greedy methods. Variance are visualized across many sub-samples of the 240 solutions per problem. 

It can be seen that the greedy approaches do not achieve a comparable performance to our TSMC method. Although SBS outperforms all other baselines in Table 1, there is still a substantial gap between its performance and our TSMC method. This result verifies the robustness of our TSMC method to the estimation error in the value function, which also verifies the necessity of sampling in practice. 

## E QUALITATIVE EXAMPLES 

Here we include some qualitative examples generated by our TSMC method in Table 3. We compare the samples discarded (do not get sampled) and cloned by TSMC (get sampled multiple times) during the same resampling step. We also include the value and the incremental importance weight estimated at each sample. Notably, the incremental importance weight is proportional to the resampling probability up to a constant, so a lower weight means a lower chance to be sampled. 

18 

Under Review 

|QUESTION|What is1(2235423523)?|||||||||
|---|---|---|---|---|---|---|---|---|---|
||I notice that this expression is a power of1, so it is just1by the defnition of|||exponents.||||||
|DISCARDED|However, I also notice that the exponent is very large, so I wonder if there<br>or a shortcut to simplify it.|||is|a pattern|||||
|VALUE: 0_._09|I<br>re<br>call<br>that<br>any<br>num<br>ber<br>raised<br>to<br>a<br>mul<br>ti<br>ple<br>of<br>2<br>has<br>a<br>sim<br>ple<br>pat<br>tern:|it<br>is||ei<br>ther|||1|or||
|WEIGHT: 0_._77|_−_1<br>de<br>pend<br>ing<br>on<br>whether<br>the<br>ex<br>po<br>nent<br>is<br>even<br>or<br>odd.|||||||||
||So the question is asking us to evaluate1(2235423523).|||||||||
|CLONED|Right, and we know that anything to the power of1is just1.|||||||||
|VALUE: 0_._18|So now we just need to evaluate12235423523.|||||||||
|WEIGHT: 1_._25|And since1is a power of1, then12235423523 = 1.|||||||||
|QUESTION|A regular hexagon can be divided into six equilateral triangles. If the perimeter of one<br>of the triangles is21inches, what is the perimeter, in inches, of the regular hexagon?|||||||||
||Since the hexagon is regular, all the sides are equal.|||||||||
|DISCARDED|And<br>since<br>the<br>hexagon<br>can<br>be<br>di<br>vided<br>into<br>six<br>equi<br>lat<br>eral<br>tri<br>an<br>gles,<br>all|the|sides|||of<br>the||||
|VALUE: 0_._02<br>WEIGHT: 0_._27|hexagon<br>are<br>also<br>equal<br>to<br>the<br>perime<br>ter<br>of<br>one<br>of<br>the<br>tri<br>an<br>gles<br>di<br>vided<br>by<br>six.<br>So<br>the<br>perime<br>ter<br>of<br>the<br>hexagon<br>is<br>21_/_6 = 3_._5<br>inches.|||||||||
||I know that the perimeter of a regular hexagon is six times the length of one of||||its sides,|||||
||so I need to fnd the length of one side of the hexagon.|||||||||
|CLONED|If I divide a regular hexagon into six equilateral triangles, each triangle must<br>same side length as the hexagon, because they are congruent.||||have the|||||
|VALUE: 0_._36|So, if the perimeter of one of the triangles is21inches, then each side of||the triangle|||||||
|WEIGHT: 1_._37|must be21_/_3 = 7inches.|||||||||
|QUESTION|How many positive whole-number divisors does196have?|||||||||
||To fnd the number of divisors of a positive whole-number, I can use|the||fact||that||it|is|
||equal to the product of one more than the exponents of its prime factorization.|||||||||
||For example,12has six divisors because it can be written as22 _∗_31, and||the||product||||of|
||one more than the exponents is(2 + 1)_∗_(1 + 1) = 6.|||||||||
|DISCARDED|So I need to fnd the prime factorization of196|||||||||
|VALUE: 0_._17|I can start by dividing by2, the smallest prime, and see if I get an integer.|||||||||
|WEIGHT: 0_._85|196_/_2 = 98,<br>which<br>is<br>not<br>an<br>in<br>te<br>ger,<br>so<br>2<br>is<br>not<br>a<br>fac<br>tor<br>of<br>196.|||||||||
||I can try the next prime,3, and see if I get an integer.|||||||||
||I need to fnd all the positive integers that divide196evenly, without leaving a||||remainder.|||||
||One way to do this is to factor196into its prime factors, and then use the formula||||||for the|||
||number of divisors of a product of prime powers.|||||||||
|CLONED|196is divisible by2, so I can write it as2times98.|||||||||
|VALUE: 0_._33|98is also divisible by2, so I can write it as2times49.|||||||||
|WEIGHT: 1_._10|49is a perfect square, and it is also divisible by7, so I can write it as7|times|||7.|||||
||Therefore,196can be written as2times2times7times7, or22 times|72.||||||||



Table 3: Qualitative examples for partial sequences discarded and cloned by TSMC. The estimated value function and incremental importance weight from the TSMC are included. We also highlight the incorrect/unpromising reasoning steps in red through human annotations. 

19 

Under Review 

## F EXAMPLE OUTPUTS 

Here we include some raw outputs from the generator, which follows the basic format in PRM800K (Lightman et al., 2024). 

# Question Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0 $ and $0 \le \theta < 2 \pi.$ # Solution To convert from rectangular to polar coordinates, I need to use the formulas $r = \sqrt{xˆ2 + yˆ2}$ and $\theta = \tanˆ{-1} \frac{y}{x}.$ For this point, $x = 0$ and $y = 3,$ so $r = \sqrt{0ˆ2 + 3ˆ2} = 3$ and $ \theta = \tanˆ{-1} \frac{3}{0}.$ However, I can’t divide by zero, so I need to use a limit to avoid that. As the angle $\theta$ approaches zero from the right, the ratio $\frac{y }{x}$ approaches $\frac{3}{0ˆ+}.$ This means that $\theta$ approaches $\frac{\pi}{2},$ which is the angle in the first quadrant. Since $r$ is always positive, I don’t need to worry about the sign or the quadrant. Therefore, the polar coordinates are $(3, \frac{\pi}{2}).$ # Answer (3, \frac{\pi}{2}) 

# Question The expression $2\cdot 3 \cdot 4\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \[ (2\cdot (3\cdot 4)) \cdot (5+1) = 144. \]In total, how many values can be obtained from the expression $2\cdot 3\cdot 4 \cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses). # Solution I need to count how many ways I can group the four factors of the product into two pairs, and then combine those pairs using either multiplication or addition. I can use the binomial coefficient to count the number of ways to choose two factors out of four, and then multiply that by the number of ways to group the remaining two factors. The binomial coefficient $\binom{4}{2} = \frac{4!}{2!2!} 

= 6$ is the number of ways to choose two factors out of four, and it represents the number of ways to arrange the two pairs of factors. 

The number of ways to combine those pairs into a single product or sum depends on which pair I choose to be the first one. 

20 

Under Review 

- If I choose the pair $2\cdot 3$, then I have two options: either to multiply them $(2\cdot 3)$, or to add them to the rest of the expression $(2\cdot 3) \cdot 4\cdot 5 + 1$. 

- If I choose the pair $4\cdot 5$, then I have two options: either to multiply them $(4\cdot 5)$, or to add them to the rest of the expression $(2\cdot 3) \cdot 4 + 1$. 

- In either case, the rest of the expression $(4\cdot 5) \cdot 2\cdot 3 + 1$ or $(2\cdot 3) \cdot 4 + 1$ is always a constant, and I can ignore it for now. 

- So, for each pair of factors, I have two options to combine them into a product or a sum. 

- Therefore, for each way to arrange the two pairs of factors, I have $2\ cdot 2 = 4$ options to combine them into a product or a sum. 

- This means that there are $6\cdot 4 = 24$ ways to insert parentheses in the expression, and obtain values other than 121. 

# Answer 

24 

21 

