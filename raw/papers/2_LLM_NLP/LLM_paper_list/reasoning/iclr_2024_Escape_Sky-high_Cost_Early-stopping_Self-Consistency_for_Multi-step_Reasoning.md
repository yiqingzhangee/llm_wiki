Published as a conference paper at ICLR 2024 

# - - - **E** SCAPE **S** KY HIGH **C** OST: **E** ARLY STOPPING **S** ELF - **C** ONSISTENCY FOR MULTI STEP REASONING 

**Yiwei Li**[1] _[†]_ **, Peiwen Yuan**[1] _[†]_ **, Shaoxiong Feng**[2] **, Boyuan Pan**[2] **, Xinglin Wang**[1] **, Bin Sun**[1] , **Heda Wang**[2] , **Kan Li**[1] _[‡]_ 

> 1 School of Computer Science, Beijing Institute of Technology 

> 2 Xiaohongshu Inc 

_{_ liyiwei,peiwenyuan,binsun,wangxinglin,likan _}_ @bit.edu.cn 

_{_ shaoxiongfeng2023,whd.thu _}_ @gmail.com 

_{_ panboyuan _}_ @xiaohongshu.com 

## ABSTRACT 

Self-consistency (SC) has been a widely used decoding strategy for chain-ofthought reasoning. Despite bringing significant performance improvements across a variety of multi-step reasoning tasks, it is a high-cost method that requires multiple sampling with the preset size. In this paper, we propose a simple and scalable sampling process, **E** arly-Stopping **S** elf- **C** onsistency (ESC), to greatly reduce the cost of SC without sacrificing performance. On this basis, one control scheme for ESC is further derivated to dynamically choose the performance-cost balance for different tasks and models. To demonstrate ESC’s effectiveness, we conducted extensive experiments on three popular categories of reasoning tasks: arithmetic, commonsense and symbolic reasoning over language models with varying scales. The empirical results show that ESC reduces the average number of sampling of chain-of-thought reasoning by a significant margin on six benchmarks, including MATH (-33.8%), GSM8K (-80.1%), StrategyQA (-76.8%), CommonsenseQA (- 78.5%), Coin Flip (-84.2%) and Last Letters (-67.4%), while attaining comparable performances[*] . 

## 1 INTRODUCTION 

Large language models (LLMs) have exhibited strong reasoning capabilities (Bubeck et al., 2023), especially with chain-of-thought (CoT) prompting (Wei et al., 2022). Based on this, Wang et al. (2023) introduce a simple decoding strategy called self-consistency (SC) to further improve reasoning performances, which takes advantage of the fact that complex reasoning tasks typically allow for more than one reasoning paths leading to the correct answer. In contrast to the standard chainof-thought prompting which only generates the greedy one, this method samples multiple reasoning paths according to the predetermined sample size, and then derives the final answer through votingbased scheme. 

However, despite generally leading to improvements, the SC strategy incurs a significant overhead proportional to the number of sampled outputs, under the assumption that the sampled outputs are of equal length. 

Taking the most popular arithmetic reasoning benchmark for LLMs as an example, testing the entire MATH dataset with SC (sampling size is 64 as Lewkowycz et al. (2022)) costs about 2000$ through GPT-4 API, which is a significant burden for many researchers and organizations. Therefore, it is essential to minimize the cost of SC while maintaining performance. 

The process of generating multiple samples in SC can be viewed as approximating the true answer distribution predicted by the language model under a specific sampling temperature. Then the most 

> †Equal contributions. 

> ‡Corresponding author. 

> * Our code and data have been released on https://github.com/Yiwei98/ESC. 

1 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
SC<br>Demonstrations  (optional) cost:20 samples Voting:  3<br>Question : A robe takes 2  Sampling<br>bolts of blue fiber and  3  2  3  3  4  3  3  3  3  3  3  1  3  3  3  4  3  2  3  4<br>half that much white  1 [st]  window 2 [nd]  window<br>fiber.  How many bolts in<br>total does it take? LLM ESC Voting:  3 Saving<br>cost:10 samples Entropy is zero        Stopping<br>**----- End of picture text -----**<br>


Figure 1: Full process of ESC compared with original SC. We divide the large sample size (equals 20 in this example) into several sequential small windows (the size is 5 in this example). Stop sampling when answers within a window are all the same, i.e., the entropy score of predicted answer distribution is zero. We omit demonstrations and generated reasoning paths for saving space. 

frequent one is taken as the final answer to mitigate the stochasticity of the single-sampling strategy. However, given that only the most confident answer is needed for SC, it is not necessary whether the whole answer distribution fits perfectly. In some cases, the final result can be obtained by voting only within small part of the generated samples. Therefore, we argue that it is not necessary to directly generate all reasoning paths aligning with the preset sampling size for every input. Instead, the generation process can be serialized as smaller parts, each of which is named as sampling window. 

Considering both the small window and the large number of sampling outputs are generated from the same predicted answer distribution, it can be deemed as a probe to reveal some information of the true distribution with only a small sampling number. 

For the answer distribution, one conjecture is that the correct ones are often concentrated, and the incorrect answers are scattered. We employ entropy as a representation of the answer distribution shape. Figure 2 shows the mean entropy value of distributions of correct and incorrect voting answer within a window respectively, showing that distributions with correct one as highest probability answer typically have much lower entropy values. Thus, it can be a indicator to determine whether sampling should continue. Base on this, we propose early-stopping self-consistency (ESC), truncating the sampling process with low entropy window. Figure 1 illustrates its process with an example. In order to maintain the performance as much as possible, we set the strictest threshold: the entropy equals zero, i.e., all generated samples within a window have the same answer. Stop sampling when this situation occurs to reduce sampling consumption while minimize performance impact. 

Early stopping (Yao et al., 2007) is a widely used technique to prevent poor generalization when training models. But in this paper, we introduce this strategy to early stop the generation process for saving the cost. As with original SC, ESC is completely unsupervised and modelagnostic, without any human annotation and additional training. We derive the theoretical upper bound of inconsistent probability of the results with or without early stopping scheme in SC, indicating that ESC is highly likely to maintain performance. In addition, one control scheme for ESC is further derivated to dynamically choose the performance-cost balance for different tasks and models by selecting the size of window and maximum sampling times, thereby meeting the practice requirements. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0002-07.png)


**----- Start of picture text -----**<br>
Correct<br>2.5 Incorrect<br>2.0<br>1.5<br>1.0<br>0.5<br>2 3 4 5 8 10<br>Window Size<br>Figure 2: The mean entropy score<br>within the window on MATH dataset<br>from GPT-4.<br>Entropy<br>**----- End of picture text -----**<br>


We evaluate ESC on a wide range of arithmetic, commonsense and symbolic reasoning tasks over three language models with varying scales: GPT-4, GPT-3.5-Turbo and Llama-2 7b. The empirical results show that ESC reduces the sampling times of chain-of-thought reasoning by a significant margin on six popular benchmarks, including MATH (Hendrycks et al., 2021) (-33.8%), GSM8K (Cobbe et al., 2021) (-80.1%), StrategyQA (Geva et al., 2021) (-76.8%), CommonsenseQA (Talmor et al., 2019) (-78.5%), Coin Flip (-84.2%) and Last Letters (Wei et al., 2022) (-67.4%) , while attaining comparable performances. In additional experiments, we show our control scheme for ESC can 

2 

Published as a conference paper at ICLR 2024 

predict the performance-cost balance accurately across various tasks and models, showcasing reliable application prospects. We also show ESC can robustly save cost considering different decoding settings and prompts. 

## 2 METHOD 

## 2.1 REVISITING SELF-CONSISTENCY 

Self-consistency (Wang et al., 2023) capitalizes on the notion that a intricate problem requiring logical thinking usually admits several distinct approaches that all lead to the same correct answer. Based on this, multiple candidate predictions _{y_ ˆ _[l] }[L]_ to problem _x_ are suggested to generate through sampling, and the most consistent _y_ ˆ is selected as the final prediction through a voting process: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0003-05.png)


where **1** ˆyl=i is the indicator function (equal to 1 when _y_ ˆ _[l]_ is equal to prediction _i_ , and 0 otherwise). We conduct a more in-depth derivation as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0003-07.png)


where _f[L]_ ( _i_ ) denotes the frequency of the model’s predicted outcome being _i_ in _L_ sampling instances. According to the Law of Large Numbers (Papoulis, 1990), as the sample size _L_ approaches infinity, _f[L]_ ( _i_ ) = _P_ ( _i_ ), where _P_ ( _i_ ) represents the true probability of the model predicting the outcome as _i_ . On this basis, we can further deduce: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0003-09.png)


According to equation 3, we can re-conceptualize SC as the process of mitigating noise introduced by individual sampling through multiple samplings. Its objective is to ensure that the prediction with the highest probability, denoted as arg max _i P_ ( _i_ ) is chosen as the final answer. 

From this perspective, we contend that when the entropy of _P_ is low (an extreme case being the one-hot-like distribution), a smaller value of _L_ ( _L_ = 1 is enough for one-hot-like distribution) can mitigate the impact of sampling noise according to the Law of Large Numbers. 

## 2.2 EARLY-STOPPING SELF-CONSISTENCY 

Building upon the aforementioned analysis, we propose **E** arly-Stopping **S** elf- **C** onsistency (ESC) to achieve comparable performance to SC at a much lower sampling cost by adaptively adjusting the sampling times. The execution flow of ESC is illustrated in Algorithm 1. 

**Algorithm 1** Early-Stopping Self-Consistency. 

**Require:** model _M_ , dataset D = _{_ ( _x, y_ ) _}[N]_ , window size _w_ , max sampling size _L_ , past sampling set S _past_ **Ensure:** predictions set S _predictions_ S _predictions ←_ ∅ **for** _i ∈_ range _N_ **do** S _candidates ←_ ∅, S _window ←_ S _[i] past_ **[if]**[ S] _past_[=][ ∅] **[else]**[ ∅] **for** _j ∈_ range ( _L//w_ ) **do** S _window ←_ Sampling predictions _w_ times from _M_ given _x[i]_ **if** S _window_ = ∅ **else** S _window_ **if** predictions in S _window_ are the same **then** S _candidates ←_ S _window_ **break end if** S _candidates ←_ S _candidates_ + S _window_ , S _window ←_ ∅ **end for** S _predictions ←_ S _predictions_ .Append(arg max _i_ � _p∈_ S _candidates_ **[1]** _[p]_[=] _[i]_[)] **end for return** S _predictions_ 

3 

Published as a conference paper at ICLR 2024 

We view a consecutive set of _w_ sampling predictions as an observation window for the probability distribution _P_ predicted by model _M_ for input _x_ . When these _w_ predictions are all the same, the entropy of _P_ is likely to be sufficiently low. At this point, the voting result of these _w_ samples is very likely to be equal to arg max _i P_ ( _i_ ), which is exactly the voting result that SC aims for. Therefore, we stop sampling when this situation occurs to save additional sampling consumption while barely affecting performance. We will iterate to obtain multiple observation windows until the preset sampling size _L_ is reached if no observation window meets the need during this procedure. The predictions of all the samples will constitute the final outcome according to equation 1. 

## 2.3 THEORETICAL ANALYSIS 

Now, we analyze to what extent ESC will effect the performance compared with SC. For simplicity, we study the case where _L_ is infinite (we will examine the case of bounded _L_ in section 2.4.2). We conduct a _one proportion z-test_ with the following null hypothesis, 

_H_ 0 : Prediction _p_ appears in an observation window _T_ times, where _p ̸_ = arg max _P_ ( _i_ ). _i_ 

According to the definition of _one proportion z-test_ , the calculation formula for the z-statistic is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0004-06.png)


where _T_[ˆ] is the observed mean, _Tµ_ represents the expected mean, _Tσ_ represents the expected standard deviation, and _n_ represents the observed times. Considering the null hypothesis _H_ 0, where _p_ = arg max _i P_ ( _i_ ), we can derive as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0004-08.png)


As _w_ samples make up an observation window, according to Bernoulli distribution and equation 5: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0004-10.png)


Taking equation 6 into equation 4, z-statistic has the following lower bound when _T_[ˆ] _≥ Tµ_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0004-12.png)


The voting results of ESC are inconsistent with the voting results of SC if and only if _T_ equals to _w_ (when early-stop happens while _p_ = arg max _i P_ ( _i_ )). When this situation occurs, we have the observed times _n_ = 1 and _T_[ˆ] = _w_ , which we take into equation 7: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0004-14.png)


Suppose _w_ = 8, and we choose to reject _H_ 0 if _z ≥ √_ 8 (the lower bound in equation 8). In this case, the probability of a false positive is _≤_ 2 _×_ 10 _[−]_[3] (one-sided p-value corresponding to _z_ = _√_ 8). This means that when we conduct ESC with _w_ = 8, the probability of the voting outcome being inconsistent with the voting outcome of SC (false positive) is _≤_ 2 _×_ 10 _[−]_[3] , which is extremely low. According to equation 8, we can see that as _w_ increases, the lower bound of _z_ increases accordingly, resulting in a smaller upper bound of the probability that ESC performance being affected. 

## 2.4 CONTROL SCHEME FOR EARLY-STOP SELF-CONSISTENCY 

In practical applications, the desired scenario is that we can adjust the ESC strategy (window size _w_ , max sampling size _L_ ) based on our sampling budgets and performance requirements. Therefore, we propose a control scheme for ESC to achieve this goal. Specifically, we will deduce the expectation of voting performance and sampling cost under different ( _w_ , _L_ ) settings based on the first observation window (denote its window size as _w_ 0). 

4 

Published as a conference paper at ICLR 2024 

## 2.4.1 THE EXPECTATION OF SAMPLING COST 

We first use the sampling frequency in the first observation window to approximate the true probability distribution _P_ , and denote it as _P_[ˆ] . Based on this, the probability of stopping sampling in each observation window (where the sampled values within the window are all the same) is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0005-03.png)


According to Algorithm 1, the expected average sampling times _L_[ˆ] of dataset D is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0005-05.png)


see Appendix A.1 for detailed derivation. According to equation 10, we can determine the mapping relationship from the choice of ( _w_ , _L_ ) to the expected sampling cost _L_[ˆ] based on the sampling results of the first observation window. 

## 2.4.2 THE EXPECTATION OF VOTING PERFORMANCE 

When _L_ is bounded (practical scenarios), we discuss the probability of ESC voting outcome being inconsistent with arg max _i P_ ( _i_ ) with (denoted as _Qw_ ( _P_[ˆ] )) and without (denoted as _Qo_ ( _P_[ˆ] )) the occurrence of early-stop. 

When early-stop happens, similar to the derivations in section 2.3, we substitute _P_[ˆ] to _P_ to calculate the z-statistic under the _P_[ˆ] distribution. We then calculate the sum of the probabilities that the voting outcome of ESC being inconsistent with the voting outcome of SC for all candidate _p_ when earlystop happens, where _p ̸_ = arg max _i P_[ˆ] ( _i_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0005-10.png)


where querying( _·_ ) represents the process of querying p-value corresponding to the z-statistic. 

When early-stop not happens, ESC degenerates into SC. We view the whole _L_ samples as an observation window and has the following derivation of z-statistic: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0005-13.png)


_T_ Ifˆ _≥_ prediction _L/_ 2. Combining with _p_ is selected asequation 12, we derive the upper bound ofthe voting outcome, then the the following _Qo_ ( ˆ _P_ inequality): should hold: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0005-15.png)


According to equation 9, the probability of that early-stop not happens is pow(1 _− P_[ˆ] _stop, L//w_ ). Thus, the upper bound of the expected probability of ESC voting outcome being inconsistent with arg max _i P_ ( _i_ ) is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0005-17.png)


## 2.4.3 CONTROLLABLE EARLY-STOP SELF-CONSISTENCY 

Based on the above derivation, we propose our control scheme for ESC as shown in Algorithm 2. First, we sample _w_ 0 times on the whole dataset. Based on the results of the first observation window, we calculate the expected sampling cost and performance under different settings of ( _w_ , _L_ ). Finally, considering the sampling budget and performance requirements, we choose appropriate values of ( _w_ , _L_ ) based on the respective expected values to execute ESC. 

5 

Published as a conference paper at ICLR 2024 

**Algorithm 2** Control Scheme for Early-Stop Self-Consistency. 

**Require:** model _M_ , dataset D = _{_ ( _x, y_ ) _}[N]_ , initial window size _w_ 0 (recommended as 5), sampling budget _B_ , performance expectation _P_ . **Ensure:** Predictions set S _predictions_ S _firstwindow ←_ ∅ **for** ( _x, y_ ) _∈_ D **do** S _window ←_ Sampling predictions _w_ 0 times from _M_ given _x_ S _firstwindow ←_ S _firstwindow_ + S _window_ **end for for** _∀_ ( _w, L_ ) **do** E( _L_[ˆ] ) _←_ section 2.4.1 (S _firstwindow, w, L_ ), E( _Q_[ˆ] ) _←_ section 2.4.2 (S _firstwindow, w, L_ ) **if** (E( _L_[ˆ] ) _,_ E( _Q_[ˆ] )) meets the need of ( _B, P_ ) **then** _w_ ˆ _← w, L_[ˆ] _← L_ **break end if end for return** Algorithm 1( _M_ , D = _{_ ( _x, y_ ) _}[N]_ , _w_ ˆ, _L_[ˆ] , S _firstwindow_ ) 

## 3 EXPERIMENTS 

## 3.1 EXPERIMENT SETUP 

We evaluate the proposed ESC on six benchmark datasets from three categories of reasoning tasks: For arithmetic reasoning, we consider MATH (Hendrycks et al., 2021) and GSM8K (Cobbe et al., 2021). MultiArith (Roy & Roth, 2015), SVAMP (Patel et al., 2021), AddSub (Hosseini et al., 2014) and ASDiv (Miao et al., 2020) are not chosen in this paper because they are relatively simple. For commonsense reasoning, CommonsenseQA (Talmor et al., 2019) and StrategyQA (Geva et al., 2021) are used. 

For symbolic reasoning, we use Last Letter Concatenation and Coin Flip from Wei et al. (2022). The data version is from Kojima et al. (2022). 

ESC is evaluated across three language models with varying scales: GPT-4 (OpenAI, 2023), GPT3.5-Turbo and LLaMA-2 7b (Touvron et al., 2023). All experiments are conducted in the few-shot setting without training or fine-tuning the language models. To ensure a fair comparison, we use the same prompts as Wei et al. (2022). Details on the prompts used are given in Appendix. 

The sampling temperature _T_ for MATH is 0.5 while for other datasets is 0.7. GPT-4 and GPT3.5-Turbo samples predictions without truncating. For Llama 2, the threshold for top p truncation (Holtzman et al., 2020) is 0.9. Similarly to Wang et al. (2023), we provide an ablation study in Section 3.6 to show that ESC is generally robust to sampling strategies and parameters. 

## 3.2 MAIN RESULTS 

The baseline we compare to is chain-of-thought prompting with greedy decoding (CoT) and selfconsistency (SC) with sampling. Following Lewkowycz et al. (2022), the sample size _L_ for MATH is 64 and for others is 40, and ESC uses the same value as maximum sample size. Accordingly, the window size _w_ for MATH is 8 and for others is 5. We report the results averaged over 10 runs and omit variance for limited space. 

**ESC significantly reduces costs while barely affecting performance.** Table 1 summarizes accuracy of CoT, SC, proposed ESC and _L_[ˆ] , the average actual number of generated samples in ESC, for each dataset among three language models. The first observation is that SC outperforms CoT substantially, which confirms the effectiveness of the voting process for reasoning. For ESC, the _L_[ˆ] is largely smaller than the corresponding maximum sampling size _L_ , while the accuracy has remained almost unchanged. Given that the accuracy of Coin Flip on GPT-4 is 100%, there is no need to conduct SC and ESC on it. We also test SC with _L_[ˆ] as the sampling size ( _L_[ˆ] -SC), whose accuracy drops in accordance with the performance curve relative to the number of samples from Wang et al. (2023). Overall, ESC can reduce costs significantly while barely affecting performance. In other words, ESC can get higher accuracy under the same sampling costs. 

6 

Published as a conference paper at ICLR 2024 

Table 1: Accuracy (%) and _L_[ˆ] (average actual number of generated samples in ESC, in gray) across six reasoning benchmarks. _L_[ˆ] -SC denotes the accuracy of SC with sample size as _L_[ˆ] . 

|||MATH|GSM8K|CSQA|SQA|Letter|Coinfip|
|---|---|---|---|---|---|---|---|
||CoT|50.44|87.70|83.71|78.63|93.12|100.00|
||SC|60.32|89.29|87.18|81.67|95.00|/|
|GPT-4|ESC|60.32 (0.00)|89.29 (0.00)|87.18 (0.00)|81.70 (+0.03)|94.98 (-0.02)|/|
||ˆ_L_|42.40 (-21.60)|7.98 (-32.02)|9.29 (-30.71)|7.19 (-31.39)|6.32 (-33.68)|/|
||ˆ_L_-SC|59.98 (-0.34)|89.07 (-0.22)|86.49 (-0.69)|81.40 (-0.27)|94.59 (-0.39)|/|
||CoT|35.53|75.83|74.17|67.66|80.50|83.74|
|GPT-3.5<br>Turbo|SC<br>ESC<br>ˆ_L_|49.97<br>49.96 (-0.01)<br>52.37 (-11.63)|85.69<br>85.67 (-0.02)<br>14.65 (-25.35)|78.10<br>78.10 (0.00)<br>11.70 (-28.30)|75.90<br>75.71 (-0.19)<br>8.51 (-27.93)|83.21<br>83.15 (-0.06)<br>8.82 (-31.18)|99.54<br>99.49 (-0.05)<br>13.03 (-26.97)|
||ˆ_L_-SC|49.79 (-0.13)|84.82 (-0.85)|77.67 (-0.43)|75.07 (-0.83)|82.74 (-0.41)|98.67 (-0.82)|
||CoT|5.09|18.07|65.28|46.23|14.87|54.74|
|Llama-2<br>7b|SC<br>ESC<br>ˆ_L_|7.68<br>7.68 (0.00)<br>62.48 (-1.52)|21.75<br>21.74 (-0.01)<br>31.21 (-8.79)|67.70<br>67.68 (-0.02)<br>11.82 (-28.18)|63.15<br>63.01 (-0.14)<br>11.00 (-23.96)|23.32<br>23.32 (0.00)<br>34.73 (-5.27)|59.13<br>58.99 (-0.14)<br>14.87 (-25.13)|
||ˆ_L_-SC|7.68 (0.00)|21.52 (-0.22)|66.97 (-0.71)|61.19 (-1.96)|23.11 (-0.21)|58.11 (-0.88)|



Table 2: Reasoning accuracy (%) and _L_[ˆ] with various max sampling size _L_ . The window size is 8. 

|Model|Method|16||24||32|40|48|64|
|---|---|---|---|---|---|---|---|---|---|
||SC|58.92||59.40||59.77|59.95|60.07|60.31|
|GPT-4|ESC|58.92|(0.00)|59.40|(0.00)|59.77 (0.00)|59.95 (0.00)|60.07 (0.00)|60.31 (0.00)|
||ˆ_L_|13.56|(-2.44)|18.72|(-5.28)|23.67(-8.33)|28.49 (-11.51)|33.21 (-14.79)|42.41 (-21.59)|
|GPT-3.5<br>Turbo|SC<br>ESC<br>ˆ_L_|47.34<br>47.33 <br>14.84|(-0.01)<br> (-1.16)|48.48<br>48.49 <br>21.38|(+0.01)<br> (-2.62)|49.02<br>49.02 (0.00)<br>27.76 (-4.24)|49.40<br>49.41 (+0.01)<br>34.02 (-5.98)|49.65<br>49.64 (-0.01)<br>40.20 (-7.80)|49.96<br>49.96 (0.00)<br>52.37 (-11.63)|
|Llama-2<br>7b|SC<br>ESC<br>ˆ_L_|7.10<br>7.10 <br>15.88|(0.00)<br> (-0.12)|7.28<br>7.28 <br>23.72|(0.00)<br> (-0.28)|7.40<br>7.40 (0.00)<br>31.52 (-0.48)|7.45<br>7.45 (0.00)<br>39.29 (-0.71)|7.54<br>7.54 (0.00)<br>47.04 (-0.96)|7.70<br>7.70 (0.00)<br>62.48 (-1.52)|



**ESC is a scalable decoding process across sampling and window size.** We conduct experiments with various window size and sampling size to valid the scalability of ESC. Table 2 shows the performance across different maximum sampling sizes. First we can see the performance of SC continuously improves as sampling size _L_ increases, which is consistent with the results in (Wang et al., 2023). On this basis, ESC can significantly save costs while maintaining performance for different _L_ . Figure 3 shows performance-cost balancing lines among three language models on GSM8K. ESC is robust to different window size and maximum sampling number. Please refer to Appendix-Figure 7 for results on other datasets. 

**Cost savings are positively correlated with performance.** As shown in both Table 1 and Table 2, an obvious phenomenon is that cost savings are positively correlated with performance. It is intuitive since better performance often eliminates the need for larger sample size as recommended in Wang et al. (2023). However, ESC does not require any prior knowledge of model capabilities and task difficulty. In addition, a control scheme for ESC is also proposed and will be evaluated in Section 3.3. 

## 3.3 EFFECTIVENESS OF CONTROL SCHEME FOR ESC 

To assess the effectiveness of control scheme for ESC, we compared the consistency between our predicted and actual values of sampling cost _L_[ˆ] and performance change percentage between SC and ESC _Pδ_ on the GSM8K dataset. We chose L1 norm and Pearson correlation to measure the 

7 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
GPT-4: GSM8K GPT-3.5-Turbo: GSM8K Llama-2-7b: GSM8K<br>40 w:3 40 w:3 40 w:3<br>w:4 w:4 w:4<br>w:5 w:5 w:5<br>30 w:8 30 w:8 30 w:8<br>w:10 w:10 w:10<br>SC SC SC<br>20 20 20<br>10 10 10<br>88.6 88.8 89.0 89.2 80 81 82 83 84 85 19.0 19.5 20.0 20.5 21.0 21.5<br>ACC ACC ACC<br>L L L<br>**----- End of picture text -----**<br>


Figure 3: Robustness analysis of ESC regarding the observation window size _w_ and max sampling size _L_ on GSM8K with different models. 

consistency.[*] As shown in Table 3, the Pearson correlations for both _L_[ˆ] and _Pδ_ exceed 0.8, indicating a strong linear correlation between the predicted and actual values. Also, the L1 norm for both _L_[ˆ] and _Pδ_ are very low. These results indicate that the predictions we obtain based on equation 10 and equation 14 are highly reliable for balancing sampling cost and voting performance. 

Table 3: _L_ ˆ: sampling times, andConsistency between the predicted (through control scheme for ESC) and actual values of _Pδ_ (%): performance change percentage between SC and ESC. We choose L1 norm and Pearson correlation to measure the consistency. All the p-values _<_ 0 _._ 05. 

|Model|_∥_ˆ_Lact −_ˆ_Lpre∥_1|Pearson(ˆ_Lact,_ ˆ_Lpre_)|_∥P act_<br>_δ_<br>_−P pre_<br>_δ_|_∥_1|Pearson(_P act_<br>_δ_|_, P pre_<br>_δ_|)|
|---|---|---|---|---|---|---|---|
|GPT-4|0.27|1.00|0.43||0.81|||
|GPT-3.5 Turbo|0.62|1.00|0.06||0.86|||
|Llama-2 7b|1.43|1.00|0.52||0.86|||



## 3.4 ESC FOR OPEN-ENDED GENERATIONS 

Original SC is only suitable for problems that have fixed answers, while Jain et al. (2023) extended it for open-ended generation tasks by replacing voting through text similarity matching. We conduct ESC on MBPP dataset (Austin et al., 2021) with various sampling size (window size is 5). The results in Table 4 shows that ESC is also suitable for open-ended task. 

Table 4: Reasoning accuracy (%) and _L_[ˆ] for MBPP with various max length L on GPT-3.5-Turbo. 

|Method|10|15|20|25|30|
|---|---|---|---|---|---|
|SC|61.96|61.96|62.04|62.15|62.18|
|ESC|61.96 (0.00)|61.96 (0.00)|62.02 (-0.02)|62.11 (-0.04)|62.15 (-0.03)|
|ˆ_L_|5.62 (-4.38)|6.02 (-8.98)|6.32 (-13.68)|6.57 (-18.43)|6.79 (-23.21)|



## 3.5 INTERSECTION BETWEEN ESC AND SC 

Table 5: Intersection ratio between ESC and SC. 

|According to Section 2.3, the voting results of<br>ESC should have a high probability of being con-<br>sistent with SC. From Table 5 we can see that<br>the intersection ratios of the voting results of SC<br>and ESC are quite high, which indicates the upper<br>bound of the performance being affected by ESC<br>derived in Section 2.4.2 is reliable.|Model<br>GSM8K<br>SQA<br>Letter<br>GPT-4<br>99.69<br>99.69<br>99.96<br>GPT-3.5-Turbo<br>99.92<br>99.25<br>99.76<br>Llama-2-7b<br>99.77<br>99.13<br>99.75|
|---|---|



*We vary the max sampling size to obtain multiple sets of results for calculating the Pearson correlation and mean L1 norm. 

8 

Published as a conference paper at ICLR 2024 

## 3.6 ROBUSTNESS OF ESC 

Additional experiments were conducted to further test the robustness of the proposed ESC, including its robustness to sampling parameters and prompts: (1) In Figure 4 (up) we show how ESC behaves for GSM8K as the decoding sampling temperature increases. Savings are consistent across different generation temperatures. 

(2) Figure 4 (bottom left corner) shows that ESC is robust to p values for top-p sampling. (3) Figure 4 (bottom right corner) indicates ESC can generalize to zero-shot manner. (4) Table 6 shows the accuracy of ESC and SC with different groups of demonstrations. We can see that ESC is robust to various demonstrations (see Appendix-Figure 6 for results on StrategyQA). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0009-04.png)


**----- Start of picture text -----**<br>
GSM8K T=0.3 GSM8K T=0.5 GSM8K T=0.7 GSM8K T=0.9<br>40 SC 40 SC 40 SC 40 SC<br>35 ESC 35 ESC 35 ESC 35 ESC<br>30 30 30 30<br>25 25 25 25<br>20 20 20 20<br>15 15 15 15<br>10 10 10 10<br>5 5 5 5<br>80.5 81.0 81.5 81.5 82.0 82.5 83.0 83.5 84.0 82.0 82.5 83.0 83.5 84.0 84.5 85.0 85.5 83 84 85 86<br>ACC ACC ACC ACC<br>GSM8K p=0.9 GSM8K p=0.95 GSM8K p=1 GSM8K Zero-shot<br>40 SC 40 SC 40 SC 40 SC<br>35 ESC 35 ESC 35 ESC 35 ESC<br>30 30 30 30<br>25 25 25 25<br>20 20 20 20<br>15 15 15 15<br>10 10 10 10<br>5 5 5 5<br>82.0 82.5 83.0 83.5 84.0 84.5 85.0 83 84 85 82.5 83.0 83.5 84.0 84.5 85.0 85.5 79 80 81 82<br>ACC ACC ACC ACC<br>L L L L<br>L L L L<br>**----- End of picture text -----**<br>


Figure 4: Robustness analysis of ESC regarding the sampling temperature _T_ , _p_ in nucleus sampling, and zero-shot demonstration on GSM8K with GPT-3.5-Turbo. 

Table 6: Reasoning accuracy (%) and _L_[ˆ] for GSM8K on GPT-3.5-Turbo with different demonstrations. The max sampling size is 40 and window size is 5. 

|Demonstration Groups|1st||2nd||3rd||4th||5th||
|---|---|---|---|---|---|---|---|---|---|---|
|SC|85.69||85.56||84.80||85.63||85.24||
|ESC|85.67|(-0.02)|85.58|(0.02)|84.80|(0.00)|85.64|(0.01)|85.23|(-0.01)|
|ˆ_L_|14.65|(-25.35)|14.53|(-25.47)|15.64|(-24.36)|14.76|(-25.24)|14.59|(-25.41)|



## 4 RELATED WORK 

**Chain-of-thought Reasoning** Chain-of-thought prompting has been proven to be an effective method of solving complex reasoning problems (Wei et al., 2022). By following the pattern of gradually solving sub-problems, both few-shot CoT (Fu et al., 2023) and zero-shot CoT (Kojima et al., 2022) are capable of stimulating LLM reasoning abilities. On this basis, Least-to-most prompting (Zhou et al., 2023) suggests explicitly splitting the problem and solving them step by step. Zheng et al. (2023) iteratively generating answers and adding the historically generated answers as hints to the context to achieve the final convergence on the answer. 

**Self Consistency** Self-consistency (Wang et al., 2023) refers to a simple decoding strategy for further improving reasoning performance, leveraging the fact that complex reasoning tasks typically 

9 

Published as a conference paper at ICLR 2024 

allow for more than one correct reasoning path. Jain et al. (2023) extend it for open-ended generation tasks like code generation and text summarization by replacing voting through text similarity matching. Li et al. (2023) assign appropriate weights for answer aggregation to achieve adaptive self-consistency. However, all of them require multiple sampling with the pre-set size, which will incur much more computation cost. Aggarwal et al. (2023) introduce an adaptive stopping criterion based on the amount of agreement between the samples so far, but it needs additional data to tune the hyperparameter and is sensitive to threshold. By contrast, ESC has no hyperparamter for stopping criterion and has a additional control scheme to meet the realistic requirements. 

## 5 CONCLUSION 

We have introduced a simple yet effective sampling process called early-stopping self-consistency (ESC). By stopping the decoding process with high confident window, ESC greatly reduce the cost of SC without sacrificing performance. A control scheme for ESC is further derivated to dynamically select the performance-cost balance for different tasks and models, which requires no extra prior knowledge of model capabilities and task difficulty. The empirical results show that ESC reduces the actual number of samples of chain-of-thought reasoning by a significant margin on six popular benchmarks, while attaining comparable performances. We also show control scheme for ESC can predict the performance-cost trade-off accurately across various tasks and models. The additional evaluations indicate that ESC can robustly save cost considering different decoding settings and prompts, and even on open-ended generation tasks. 

## ACKNOWLEDGMENTS 

This work is supported by Beijing Natural Science Foundation (No.4222037, L181010). 

## REFERENCES 

- Pranjal Aggarwal, Aman Madaan, Yiming Yang, and Mausam. Let’s sample step by step: Adaptiveconsistency for efficient reasoning with llms. _CoRR_ , abs/2305.11860, 2023. doi: 10.48550/ ARXIV.2305.11860. URL https://doi.org/10.48550/arXiv.2305.11860. 

- Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. _CoRR_ , abs/2108.07732, 2021. URL https: //arxiv.org/abs/2108.07732. 

- S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco T´ulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with GPT-4. _CoRR_ , abs/2303.12712, 2023. doi: 10.48550/arXiv.2303.12712. URL https://doi.org/10.48550/arXiv.2303.12712. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168. 

- Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. _CoRR_ , abs/2301.12726, 2023. doi: 10.48550/arXiv.2301. 12726. URL https://doi.org/10.48550/arXiv.2301.12726. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. _Trans. Assoc. Comput. Linguistics_ , 9:346–361, 2021. doi: 10.1162/tacl _\_ ~~a~~ _\_ ~~0~~ 0370. URL https://doi. org/10.1162/tacl_a_00370. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In _Proceedings of the Neural Information Processing Systems Track on Datasets and_ 

10 

Published as a conference paper at ICLR 2024 

_Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html. 

- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. URL https://openreview. net/forum?id=rygGQyrFvH. 

- Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL_ , pp. 523–533. ACL, 2014. doi: 10.3115/v1/d14-1058. URL https://doi.org/10.3115/v1/d14-1058. 

- Siddhartha Jain, Xiaofei Ma, Anoop Deoras, and Bing Xiang. Self-consistency for open-ended generations. _CoRR_ , abs/2307.06857, 2023. doi: 10.48550/arXiv.2307.06857. URL https: //doi.org/10.48550/arXiv.2307.06857. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo GutmanSolo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html. 

- Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Bin Sun, Xinglin Wang, Heda Wang, and Kan Li. Turning dust into gold: Distilling complex reasoning capabilities from llms by leveraging negative data. _CoRR_ , abs/2312.12832, 2023. doi: 10.48550/ARXIV.2312.12832. URL https: //doi.org/10.48550/arXiv.2312.12832. 

- Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020_ , pp. 975–984. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.92. URL https://doi. org/10.18653/v1/2020.acl-main.92. 

- OpenAI. GPT-4 technical report. _CoRR_ , abs/2303.08774, 2023. doi: 10.48550/arXiv.2303.08774. URL https://doi.org/10.48550/arXiv.2303.08774. 

- Athanasios Papoulis. _Probability and statistics_ . Prentice-Hall, Inc., 1990. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021_ , pp. 2080–2094. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.naacl-main.168. URL https://doi.org/10.18653/v1/2021. naacl-main.168. 

- Subhro Roy and Dan Roth. Solving general arithmetic word problems. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015_ , pp. 1743–1752. The Association for Computational Linguistics, 2015. doi: 10.18653/v1/d15-1202. URL https://doi.org/10.18653/v1/d15-1202. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human_ 

11 

Published as a conference paper at ICLR 2024 

_Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , pp. 4149–4158. Association for Computational Linguistics, 2019. doi: 10.18653/v1/n19-1421. URL https://doi.org/10.18653/v1/n19-1421. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. _CoRR_ , abs/2307.09288, 2023. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL https://openreview.net/ pdf?id=1PL1NIMMrw. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. URL http://papers.nips.cc/paper_files/paper/ 2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference. html. 

- Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning. _Constructive Approximation_ , 26:289–315, 2007. 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. _CoRR_ , abs/2304.09797, 2023. doi: 10.48550/ arXiv.2304.09797. URL https://doi.org/10.48550/arXiv.2304.09797. 

- Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL https://openreview.net/pdf?id=WZH7099tgfM. 

12 

Published as a conference paper at ICLR 2024 

## A APPENDIX 

## A.1 DERIVATION OF THE EXPECTATION OF SAMPLING COST 

Given the probability of stopping sampling in each observation window as _P_[ˆ] _stop_ , the probability of _P_ continue sampling isˆ _stop ×_ pow(1 _− P_ ˆ _stop_ 1 _−, jP_ )[ˆ] . _stop_ On this basis, the expected sampling cost in the first. Therefore, the probability of early-stop occurring after _L//w − j_ windows is1 windows can be denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0013-04.png)


If no early-stop occurs after observing _L//w −_ 1 windows, then the maximum sampling size _L_ is reached upon the next observation window. The expected sampling cost of this part can be denoted as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0013-06.png)


Including _w_ 0, the overall expected sampling cost is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0013-08.png)


## A.2 COMPARISON WITH ADAPTIVE-CONSISTENCY (AC) 

Aggarwal et al. (2023) introduce an adaptive stopping criterion based on the amount of agreement between the samples so far, but it needs additional data to tune the hyperparameter and is sensitive to threshold. But for proposed ESC, the stopping criterion needs no hyperparameter due to the most conservative strategy to maintain the performance, i.e., all the answers within a window are same. Thus ESC can be conducted directly for different tasks and models, without any validation set. 

Another drawback of AC is generating samples step by step, which means each sample requires one input. Considering the demonstrations for in-context learning (usually 8 examples) have a lot of tokens, it will cost quite a portion of the budget. By contrast, ESC generates samples in multiple sampling windows, thus samples within one window can share the same input. Table 7 shows that ESC can get higher accuracy with less sampling cost comparing with AC. 

Table 7: Prompt token count (tokens/item, denoted as #prompt), completion token count (tokens/item, denoted as # completion), average sampling cost ($/item) and accuracy (%) resulting from different methods. The best and second-best results are emphasized with bold and underline. 

|Datasets Method|GPT-3.5 Turbo|GPT-3.5 Turbo||GPT-4||
|---|---|---|---|---|---|
||# prompt # completion|Cost|Acc|# prompt # completion|Cost<br>Acc|
|GSM8K<br>SC<br>AC<br>ESC|496.9<br>2909.0<br>4930.3<br>813.2<br>1469.2<br>1220.8|0.0084|**85.69**<br> 85.66<br> 85.67|495.9<br>2646.5<br>2927.6<br>412.5<br>793.0<br>564.8|0.2316 **89.29**<br>0.1501<br>89.28<br>**0.0769 89.29**|
|||0.0087 <br>**0.0052**||||
|CSQA<br>SC<br>AC<br>ESC|428.7<br>1382.9<br>3522.3<br>294.2<br>1003.2<br>420.9|0.0043|**78.10**<br> 78.05<br> **78.10**|427.8<br>1474.6<br>2844.5<br>271.5<br>794.6<br>384.0|0.1351<br>**87.18**<br>0.1355 87.15<br>**0.0625 87.18**|
|||0.0055 <br>**0.0024 **||||
|Letter<br>SC<br>AC<br>ESC|166.0<br>1463.1<br>1057.5<br>233.6<br>293.0<br>323.7|0.0041 <br>0.0020|**83.21**<br>83.15<br> 83.15|165.0<br>1396.9<br>791.7<br>167.5<br>208.7<br>220.7|0.1184 **95.00**<br>0.0451<br>94.97<br>**0.0260** 94.98|
|||**0.0012**||||



13 

Published as a conference paper at ICLR 2024 

## A.3 COMPARISONS BETWEEN SC AND ESC WHEN L IS RELATIVELY SMALL (10 AND 20) 

we conduct experiments with smaller sampling sizes _L_ . From the results in Table 8, we found that ESC can consistently save sampling costs in such scenarios, while achieving improvement compared to SC under the same costs. 

Table 8: Reasoning accuracy (%) and _L_[ˆ] with various max sampling size _L_ . The window size is 5. 

|Model<br>Method|CSQA<br>_L_= 10<br>_L_= 20|GSM8K<br>_L_= 10<br>_L_= 20|Letter|
|---|---|---|---|
||||_L_= 10<br>_L_= 20|
|GPT-4<br>SC<br>ESC<br>ˆ_L_<br>ˆ_L_-SC|86.49<br>86.83<br>86.49 (0.00) 86.84 (+0.01)<br>5.92 (-4.08)<br>7.27 (-12.73)<br>86.07 (-0.42) 86.31 (-0.52)|89.16<br>89.24<br>89.16 (0.00) 89.24 (0.00)<br>5.79 (-4.21)<br>6.76 (-13.24)<br>88.95 (-0.21) 89.02 (-0.22)|94.66<br>94.79<br>94.66 (0.00)<br>94.80 (+0.01)<br>5.57(-4.43)<br>5.92(-14.08)<br>94.58(-0.08)<br>94.58(-0.21)|
|GPT-3.5<br>Turbo<br>SC<br>ESC<br>ˆ_L_<br>ˆ_L_-SC|77.63<br>77.93<br>77.63 (0.00) 77.91 (-0.02)<br>6.47 (-3.53)<br>8.60 (-11.40)<br>77.39 (-0.24) 77.55 (-0.38)|84.10<br>85.15<br>84.10 (0.00) 85.15 (0.00)<br>7.09 (-2.91) 10.14 (-9.86)<br>83.31 (-0.79) 84.10 (-1.05)|82.80<br>83.00<br>82.81 (+0.01) 82.98 (-0.02)<br>6.14(-3.86)<br>7.36(-12.64)<br>82.45(-0.35)<br>82.65(-0.35)|
|Llama2-7b<br>SC<br>ESC<br>ˆ_L_<br>ˆ_L_-SC|66.89<br>67.25<br>66.89 (0.00) 67.26 (+0.01)<br>6.59 (-3.41)<br>8.78 (-11.22)<br>66.80 (-0.09) 66.84 (-0.41)|20.74<br>21.32<br>20.74 (0.00) 21.32 (0.00)<br>8.99 (-1.01) 17.06 (-2.94)<br>20.59 (-0.15) 21.21 (-0.11)|21.21<br>22.49<br>21.21 (0.00)<br>22.48 (-0.01)<br>9.67(-0.33)<br>18.47(-1.53)<br>21.21(-0.00)<br>22.32(-0.17)|



## A.4 DISCUSSION ON THE CHOICE OF INTRODUCING OBSERVATION WINDOW FOR THE DESIGN OF EARLY-STOPPING STRATEGY 

We design the stopping strategy with the introduction of window for the following two reasons. Firstly, we break the sampling process only if all of the samples in the latest window are consistent, thus avoiding any hyper-parameter. If sample one by one and stop based on the observation of the sampled samples, obviously we cannot adopt such a strict truncation condition. In this case, we need to introduce a certain statistic and its corresponding threshold (hyper-parameter), which is hard to be determined in prior. Secondly, we have actually considered using the normalized entropy of the sampled samples as the statistical value for cut-off. As shown in Figure 5 , we found that this method not only has the hyper-parameter problem mentioned above, but also has no advantage over ESC in terms of performance-cost trade-off. We believe this is because examining the cut-off point after each single sampling is too frequent, introducing greater randomness. This makes the model more likely to early-stop without sufficient sampling. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0014-07.png)


**----- Start of picture text -----**<br>
Llama2-7b: CSQA GPT-3.5 Turbo: CSQA GPT-4: CSQA<br>SC SC 87.2 SC<br>67.6 ESC ENT 78.00 ESCENT 87.0 ESCENT<br>67.4 77.75 86.8<br>67.2 77.50 86.6<br>77.25 86.4<br>67.0<br>86.2<br>77.00<br>66.8 86.0<br>76.75<br>66.6 85.8<br>76.50 85.6<br>66.4 4 6 8 10 12 14 16 18 4 6 8 10 12 14 16 4 6 8 10 12 14<br>L L L<br>ACC ACC ACC<br>**----- End of picture text -----**<br>


Figure 5: Comparisons between SC, ESC and ENT(using entropy to determine the cut-off point) on CSQA with different models. 

## A.5 THE COMPARISONS BETWEEN ESC AND OTHER STRONG REASONING BASELINES. 

ESC achieves better results compared to SC with the same overhead by saving sampling costs while almost not sacrificing performance. From the perspective of performance improvement, we are curious about the comparison results between ESC and advanced reasoning methods that focus on per- 

14 

Published as a conference paper at ICLR 2024 

formance enhancement. For this purpose, we compared ESC with Progressive-Hint Prompting(PHP) (Zheng et al., 2023) in terms of both performance and overhead. PHP improves performance by approaching answers through the stepwise generation of clues. As shown in Table 9, we found that ESC outperformed CoT by 9.84%, higher than PHP’s 9.25%. Additionally, ESC incurs less sampling overhead. PHP, due to the need for multiple inputs to the model with previously enhanced prompts, has a larger overhead in the # prompt part. Overall, ESC achieves better performance with lower sampling costs. 

Table 9: Prompt token count (tokens/item, denoted as #prompt), completion token count (tokens/item, denoted as # completion), average sampling cost ($/item) and accuracy (%) comparison between CoT, PHP (Zheng et al., 2023) and ESC on GSM8K with GPT-3.5-Turbo. Max sampling size of ESC is 40. CoT and PHP apply greedy search as sampling strategy. 

|Method|#prompt|# completion|SamplingCost|Accuracy|
|---|---|---|---|---|
|CoT|496.9|72.7|0.0006|75.83|
|PHP|6552.0|360.8|0.0072|85.08|
|ESC|1469.2|1220.8|0.0052|85.67|



## A.6 THE ORTHOGONALITY OF ESC AND OTHER STRONG REASONING BASELINE. 

To assess the orthogonality of ESC with other advanced reasoning methods, we applied ESC on PHP. As shown in Table 10, PHP w. ESC achieved similar performance with PHP w. SC while significantly reduced sampling overhead. This indicates that existing advanced methods can significantly reduce costs by applying ESC with almost no sacrifice in performance. 

Table 10: Reasoning accuracy (%) and _L_[ˆ] for GSM8K on GPT-3.5-Turbo with Progressive-Hint Prompting(PHP) (Zheng et al., 2023). The window size is 5. 

|Max samplingsize|10|20|30|40|
|---|---|---|---|---|
|PHP w. SC|86.32|86.64|86.76|87.00|
|PHP w. ESC|86.32 (0.00)|86.62 (-0.02)|86.77 (+0.01)|86.98 (-0.02)|
|ˆ_L_|6.15 (-3.85)|7.83 (-12.17)|9.15 (-20.85)|10.26 (-29.74)|
|ˆ_L_-PHP w. SC|86.02 (-0.30)|86.29 (-0.35)|86.32 (-0.44)|86.35 (-0.65)|



## A.7 ADDITIONAL RESULTS 

Figure 6 shows the robustness of ESC regarding the sampling temperature _T_ , _p_ in nucleus sampling, and zero-shot demonstration on StrategyQA with GPT-3.5-Turbo. Figure 7 shows the robustness of ESC regarding the observation window size _w_ and max sampling size _L_ on multiple datasets with different models. 

## A.8 PROMPTING DETAILS 

We list the details of the prompts used for MATH dataset. Following Wang et al. (2023), we use the same prompts for other datasets as in Wei et al. (2022) for fair comparison. 

15 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0016-01.png)


**----- Start of picture text -----**<br>
StrategyQA T=0.3 StrategyQA T=0.5 StrategyQA T=0.7 StrategyQA T=0.9<br>40 SC 40 SC 40 SC 40 SC<br>35 ESC 35 ESC 35 ESC 35 ESC<br>30 30 30 30<br>25 25 25 25<br>20 20 20 20<br>15 15 15 15<br>10 10 10 10<br>5 5 5 5<br>73.25 73.50 73.75 74.00 74.25 74.50 74.75 73.75 74.00 74.25 74.50 74.75 75.00 74.0 74.5 75.0 75.5 76.0 74.5 75.0 75.5 76.0<br>ACC ACC ACC ACC<br>StrategyQA p=0.9 StrategyQA p=0.95 StrategyQA p=1 StrategyQA Zero-shot<br>40 SC 40 SC 40 SC 40 SC<br>35 ESC 35 ESC 35 ESC 35 ESC<br>30 30 30 30<br>25 25 25 25<br>20 20 20 20<br>15 15 15 15<br>10 10 10 10<br>5 5 5 5<br>74.0 74.5 75.0 75.5 74.0 74.5 75.0 75.5 74.5 75.0 75.5 69.00 69.25 69.50 69.75 70.00 70.25<br>ACC ACC ACC ACC<br>L L L L<br>L L L L<br>**----- End of picture text -----**<br>


Figure 6: Robustness analysis of ESC regarding the sampling temperature _T_ , _p_ in nucleus sampling, and zero-shot demonstration on StrategyQA with GPT-3.5-Turbo. 

16 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0017-01.png)


**----- Start of picture text -----**<br>
GPT-4: Last Letters, GPT-3.5-Turbo: Last Letters, Llama-2-7b: Last Letters,<br>40 w:3 40 w:3 40 w:3<br>w:4 w:4 w:4<br>w:5 w:5 w:5<br>30 w:8 30 w:8 30 w:8<br>w:10 w:10 w:10<br>SC SC SC<br>20 20 20<br>10 10 10<br>94.4 94.5 94.6 94.7 94.8 94.9 95.0 82.0 82.2 82.4 82.6 82.8 83.0 83.2 17 18 19 20 21 22 23<br>ACC ACC ACC<br>GPT-4: CommonsenseQA GPT-3.5-Turbo: CommonsenseQA Llama-2-7b: CommonsenseQA<br>40 w:3 40 w:3 40 w:3<br>w:4 w:4 w:4<br>w:5 w:5 w:5<br>30 w:8 30 w:8 30 w:8<br>w:10 w:10 w:10<br>SC SC SC<br>20 20 20<br>10 10 10<br>85.0 85.5 86.0 86.5 87.0 76.0 76.5 77.0 77.5 78.0 66.4 66.6 66.8 67.0 67.2 67.4 67.6<br>ACC ACC ACC<br>GPT-4: MATH GPT-3.5-Turbo: MATH Llama-2-7b: MATH<br>40 w:3 40 w:3 40 w:3<br>w:4 w:4 w:4<br>w:5 w:5 w:5<br>30 w:8 30 w:8 30 w:8<br>w:10 w:10 w:10<br>SC SC SC<br>20 20 20<br>10 10 10<br>53 54 55 56 57 58 59 60 38 40 42 44 46 48 6.0 6.5 7.0 7.5<br>ACC ACC ACC<br>GPT-4: StrategyQA GPT-3.5-Turbo: StrategyQA Llama-2-7b: StrategyQA<br>40 w:3 40 w:3 40 w:3<br>w:4 w:4 w:4<br>w:5 w:5 w:5<br>30 w:8 30 w:8 30 w:8<br>w:10 w:10 w:10<br>SC SC SC<br>20 20 20<br>10 10 10<br>81.0 81.2 81.4 81.6 73.5 74.0 74.5 75.0 75.5 76.0 56 57 58 59 60 61 62 63<br>ACC ACC ACC<br>GPT-3.5-Turbo: Coin Flip Llama-2-7b: Coin Flip<br>40 w:3 40 w:3<br>w:4 w:4<br>w:5 w:5<br>30 w:8 30 w:8<br>w:10 w:10<br>SC SC<br>20 20<br>10 10<br>92 94 96 98 56.0 56.5 57.0 57.5 58.0 58.5 59.0<br>ACC ACC<br>L L L<br>L L L<br>L L L<br>L L L<br>L L<br>**----- End of picture text -----**<br>


Figure 7: Robustness analysis of ESC regarding the observation window size _w_ and max sampling size _L_ on multiple datasets with different models. 

17 

Published as a conference paper at ICLR 2024 

Table 11: Few-shot prompts for MATH dataset. 

**system:** Think the question step by step and give the answer. 

~~_√_~~ _x−_ 2 **user:** Problem: Find the domain of the expression ~~_√_~~ 5 _−x_[.] 

**assistant:** Solution: The expressions inside each square root must be non-negative. Therefore, _x −_ 2 _≥_ 0, so _x ≥_ 2, and 5 _− x ≥_ 0, so _x ≤_ 5. Also, the denominator cannot be equal to zero, so 5 _− x >_ 0, which gives _x <_ 5. Therefore, the domain of the expression is [2 _,_ 5) . **user:** Problem: Compute: 55 _×_ 1212 _−_ 15 _×_ 1212 . 

**assistant:** Solution: We have 55 _×_ 1212 _−_ 15 _×_ 1212 = 1212(55 _−_ 15) = 1212(40) = 4848(10) = 48480 . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0018-06.png)


**user:** Problem: Find _x_ , where _x_ is a square number which is divisible by four, and between 39 and 80. 

**assistant:** Solution: We know that _x_ is between 39 and 80, and since 6[2] = 36 _<_ 39 and 9[2] = 81 _>_ 80, this means that 6[2] _< x <_ 9[2] . This leaves us with two possibilities for _x_ , which are 7[2] = 49, and 8[2] = 64. We then see that only 64 is divisible by four, so _x_ = 64 . **user:** Problem: Solve the inequality 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0018-09.png)


**assistant:** Solution: We can build a sign chart, but since all of the factors are linear, we can track what happens to the expression as _x_ increases. At _x_ = 0 _,_ the expression is positive. As _x_ increases past 1, the expression becomes negative. As _x_ increases past 2, the expression becomes positive, and so on. Thus, the solution is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning_images/iclr_2024_Escape_Sky-high_Cost_Early-stopping_Self-Consistency_for_Multi-step_Reasoning.pdf-0018-11.png)


**user:** Problem: A right circular cone has a volume of 12 _π_ cubic centimeters. The height of the cone is 4 cm. How many centimeters is the circumference of the base of the cone, in terms of _π_ ? 

**assistant:** Solution: The volume of a cone is[1][We][are][given][that][the][volume][is][12] _[π]_ 3 _[πr]_[2] _[h]_[.] and the height is 4. Thus,[1] 3 _[πr]_[2] _[·]_[ 4][=][12] _[π]_[.][Solving][for] _[r]_[,][we][find] _[r]_[=][3][.][Therefore,][the] circumference of the base is 2 _πr_ = 6 _π_ . 

**user:** Problem: How many perfect squares less than 1000 have a ones digit of 2, 3 or 4? **assistant:** Solution: Checking the squares from 1[2] to 10[2] , we see that no squares end in 2 or 3, while a square ends in 4 if its square root ends in 2 or 8. Since 31[2] _<_ 1000 _<_ 32[2] , we see that the squares less than 1000 ending in 4 are 2 _,_ 8 _,_ 12 _,_ 18 _,_ 22 _,_ 28. Thus the desired answer is 6 . 

**user:** Problem: The diagonals of a rhombus are 10 inches and 24 inches. What is the perimeter of the rhombus, in inches? 

**assistant:** Solution: The diagonals are perpendicular bisectors of each other, so therefore the side length of the rhombus can be calculated as _√_ 5[2] + 12[2] = 13. Therefore, the perimeter of the rhombus is 4 _×_ 13 = 52 inches. 

18 

