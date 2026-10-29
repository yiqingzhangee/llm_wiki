# **Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

**Weijia Xu**[1] **Andrzej Banburski-Fahey**[1] **Nebojsa Jojic**[1] 

## **Abstract** 

We introduce Reprompting, an iterative sampling algorithm that automatically learns the Chain-ofThought (CoT) recipes for a given task without human intervention. Through Gibbs sampling, Reprompting infers the CoT recipes that work consistently well for a set of training samples by iteratively sampling new recipes using previously sampled recipes as parent prompts to solve other training problems. We conduct extensive experiments on 20 challenging reasoning tasks. Results show that Reprompting outperforms humanwritten CoT prompts substantially by +9.4 points on average. It also achieves consistently better performance than the state-of-the-art prompt optimization and decoding algorithms. 

## **1. Introduction** 

Few-shot prompting with large language models (LLMs) has revolutionized the landscape of natural language processing. Given natural language instructions and a few demonstrations as in-context examples, LLMs can quickly adapt to new tasks, approaching or even surpassing the performance of models fine-tuned on larger datasets on a wide range of tasks (Brown et al., 2020). However, such prompting techniques fall short on tasks that require multi-step reasoning and constraint propagation (Wei et al., 2022), such as _logical deduction_ in the Big-Bench Hard benchmark (Suzgun et al., 2022). To address these limitations, prior works proposed to teach LLMs to reason step by step like humans by prompting them with chain-of-thought (CoT) reasoning steps for a few example problems (Wei et al., 2022). Despite the improved performance, such a method requires human experts with not only the task knowledge but also an understanding of how prompting works to craft the CoT prompt for each task (Zamfirescu-Pereira et al., 2023), which limits 

> 1Microsoft Research, Redmond, USA. Correspondence to: Weijia Xu <weijiaxu@microsoft.com>. 

> _Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

the scalability and generalizability of the method. Furthermore, a problem can be reasoned in many different ways, and some of them may work well on some LLMs but not on others. To fairly compare the performance of various LLMs on each task, we need to find the CoT prompt that works best for each model in a feasible way, which remains a challenge. 

In this paper, we propose _Reprompting_ , an iterative sampling algorithm that **automatically** finds effective CoT prompt for each model given a few question-answer pairs without human intervention. Specifically, the algorithm aims to infer a set of CoT recipes that perform consistently well as in-context examples for a set of training problems. We frame it as a problem of sampling from a joint distribution of CoT recipes given the training question-answer pairs, which is infeasible to characterize directly but can be approached using Gibbs sampling – we initially sample a set of recipes through zero-shot prompting, expand the set with new recipes sampled iteratively by using previously sampled recipes as parent prompts to solve a different training problem, and weed out the least-fit recipes that lead to wrong answers. Thus, the algorithm will eventually converge to a set of recipes that share similar chains of thought for effectively solving the training problems. These CoT recipes optimized on the training set then serve as effective CoT prompts for solving unseen test problems. 

We evaluate _Reprompting_ on 20 tasks from three reasoning benchmarks including Big-Bench Hard (BBH) (Suzgun et al., 2022), GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) using ChatGPT (OpenAI, 2023) and InstructGPT (Ouyang et al., 2022) as LLMs. Compared with human-written CoT prompts, _Reprompting_ achieves +9.4 higher accuracy on average. It also consistently outperforms self-consistency decoding (Wang et al., 2022b), Auto-CoT (Zhang et al., 2022) and Automatic Prompt Optimization (Pryzant et al., 2023) by 11–33 points on average. Furthermore, _Reprompting_ facilitates model combination by using different LLMs for initializing and sampling new recipes. Empirically, leveraging ChatGPT to sample initial recipes for InstructGPT brings up to +71 point improvements over using InstructGPT alone and even outperforms ChatGPT alone on certain tasks. Lastly, our results confirm that the CoT recipes that work well on one model 

1 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0002-01.png)


Figure 1: An example that ChatGPT can propose various different solutions to the same problem in zero-shot. 

may work poorly on another, even when the latter may approach the best performance using prompts optimized for itself. These findings emphasize the need to optimize the prompt for each model for fair comparisons. 

## **2.** _**Reprompting**_ **: Prompt Inference Through Gibbs Sampling** 

## **2.1. In-Context Learning** 

**In-context learning** has become the cornerstone of evaluating large language models (LLMs) (Brown et al., 2020; Srivastava et al., 2022). To facilitate this evaluation approach, data is provided for a large number of different tasks, with each task consisting of dozens or, more often, hundreds of instances with varying problem setup and question texts _xi_ and their corresponding text answers _yi_ , where _i ∈_ [1 _..N_ ] and _N_ is the number of problem instances for the task. Formally, in-context learning infers the answer for a given test question _x_ by prompting an LLM with a set of demonstration examples _{xi, yi}[K] i_ =1[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0002-07.png)


The performance of in-context learning can be significantly enhanced by incorporating auxiliary knowledge or humanwritten instructions in a prompt (Shwartz et al., 2020; Zelikman et al., 2022; Nye et al., 2021), particularly in the form of Chain-of-Thought (CoT) reasoning (Wei et al., 2022; Wang et al., 2022b; Zhou et al., 2022; Creswell et al., 2022; Wang et al., 2022a; Liu et al., 2022; Kojima et al., 2022; Li et al., 2022). 

In-context learning with CoT (Wei et al., 2022) can be seen in a similar light, statistically. In addition to the questionanswer pairs _{xi, yi}_ , the CoT prompt also contains worked out step-by-step reasoning “recipes” _zi_ in text, which are inserted between the question and answer: _{xi, zi, yi}_ . These recipes can play two roles. First, they further explain the intent of the question _xi_ , as a small collection of questionanswer pairs alone may be insufficient to disambiguate 

among different patterns an LLM might detect. The second role is more important: it provides step-by-step guidance on one problem and thus teaches an LLM to solve similar problems following the same routine as it continues the text conditioned on the previous tokens. In the extreme, with prompts that strictly regiment self-attention, GPT models can be turned into Turing Machines to execute standard computer algorithms (Jojic et al., 2023). In practice, the CoT prompts commonly used in prior work fall somewhere between colloquial explanations and regimented recipes. Formally, **in-context learning with CoT** infers the answer for a given test question _x_ by prompting an LLM with an optional instruction message _m_ and a set of demonstration examples with step-by-step solutions _{xi, zi, yi}[K] i_ =1[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0002-11.png)


Here, _m_ is a textual message that instructs the model to generate the step-by-step solution _z j_ before the answer text _y j_ and the specific format to present the answer.[1] It can be task-specific or generic, as in the case of our experiments. Such an instruction message can trigger instruction-tuned LLMs to generate step-by-step solutions given [ _x j, m_ ] alone without any demonstration examples (i.e. _K_ = 0), as illustrated in Figure 1. These solutions follow varying styles and often lead to incorrect answers. However, we argue that good recipes for solving the set of problems on a given task can evolve from these zero-shot solutions. In the next section, we introduce _Reprompting_ , an iterative sampling algorithm that automatically produces the CoT recipes for a given set of problems without human intervention. 

## **2.2. Prompt Inference Through Gibbs Sampling** 

We introduce the _Reprompting_ algorithm, which aims to find a set of CoT recipes _zi_ that work **consistently** well as few-shot in-context examples for a dataset _{xi, yi}[N] i_ =1[.] Specifically, we formulate it as the problem of sampling from a joint distribution 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0002-15.png)


such that _z_ 1 _...N_ are generalized enough so that given any test question _x_ , the distribution over _z_ and _y_ is approximately invariant to the choice of the _K_ -shot CoT recipes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0002-17.png)


Without characterizing the joint distribution, we can use Gibbs sampling (Geman & Geman, 1984) to generate such 

> 1This enables us to separate the generated answer _y j_ from the step-by-step solution _z j_ and forces the model to stop after generating the answer. 

2 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

samples _{z_ 1 _, z_ 2 _,...zN}_ by first sampling _{z_ 1 _, z_ 2 _,...zN}_ independently from the distributions _p_ ( _z j|x j,, y j_ ), and then iteratively drawing samples from the conditional distributions _p_ ( _z j|z_ 1 (4) of the joint distribution, we have the follow- _,..., z j−−_ 1 _, z j_ +11 _,...zN,zN,N,, {xi,xi,i,, yi}i}}[[N]] i_ =11 _[[,]][[m]]_[[)][.]][[.]][[Based]][[on]] the property (4) of the joint distribution, we have the follow-1 (4) of the joint distribution, we have the following approximation: 

**Algorithm 1:** _Reprompting_ algorithm 

_j,, j_ , samples from the conditional distribu- **Input :** Training set _{xi, yi}[N] i_ =1[, number of examples in] the prompt _K_ , number of iterations _M_ , rejection the property (4) of the joint distribution, we have the follow-1 (4) of the joint distribution, we have the follow- _,...,_ 4) of the joint distribution, we have the follow-) of the joint distribution, we have the follow- of the joint distribution, we have the follow- _z j−−_ 1 _, z j_ +11 _,...zN,zN,N,, {xi,xi,i,, yi}i}}[[N]] i_ =11 _[[,]][[m]]_[[)][.]][[.]][[Based]][[on]] probability _prej_ , the initialization model _LLM_ 1 and the sampling model _LLM_ 2 **1 Initialization:** _z j|z_ 1 _,..., z j−_ 1 _, z j_ +1 _,...zN, {xi, yi}[N] i_ =1 _[,][m]_[)] **2 for** _each j_ **do 3** _z j ←_ /0 _LLM_ ( _z j|{xi, zi, yi}i_ = _j, x j, y j, m_ ) **4** Sample _z_ ˆ _j,_ ˆ _y j ∼ pLLM_ 1( _z, y|x j, m_ ) _LLM_ ( _z j, y j|{xi, zi, yi}i_ = _j, x j, m_ ) (5) **5** Sample _u ∼ Uniform_ ([0 _,_ 1]) _LLMj ⊂_ ( _z_ [1 _j,,yNj|{_ ] _\{xi j,}zi,,|ySij}|i_ = _∈S K j , x j, m_ ) _,_ **687 ifend** _y_ ˆ _jz_ = _j ← y jz_ ˆ _or uj > prej_ **then 9 end** can sample _z j j_ by randomly picking _K_ data _j_ ) and then sampling _z j_ with weights **10 Sampling: 11 repeat 12** Randomly select _j ∈_ [1 _, N_ ] = _ppLLMLLM_ (( _zz jj,|{yxj|{i, zxii,,yzii}, iy∈iS} ji∈, xS jj,, xmj_ ) _, m_ ) (6) **1314** Randomly selectSample _z_ ˆ _j,_ ˆ _y j ∼ p SLLM j ⊂_ 2([ _z_ 1 _,,yN|{_ ] _\{xi, jz}i,_ of size _yi}i∈S j , Kx j, m_ ) **15** Sample _u ∼ Uniform_ ([0 _,_ 1]) _· pLLM_ ( _y j|{xi, zi, yi}i∈S j , x j, m, z j_ ) **16 if** _y_ ˆ _j_ = _y j or u > prej_ **then** ˆ **17** _z j ← z_ ˆ _j_ approximate it is to sample several _z j_ from **18 end** ˆ _{xi,xi,i,, zi,i,, yi}i∈S ji}i∈S j}i∈S ji∈S j_ , _x j_ andprobability _m_ , computeof **19 until** _convergence or M iterations are reached_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0003-04.png)


Thus, we can sample _z j j_ by randomly picking _K_ data points (excluding _j_ ) and then sampling _z j_ with weights proportional to the conditional probability 

One way to approximate it is to sample several _z_ ˆ _j_ from the LLM conditioned onthe weight for each _z_ ˆ _j_ using _{xi,xi,i,, zi,i,,_ the _yi}i∈S ji}i∈S j}i∈S ji∈S j_ model’s _∈S jS j j_ , _x j_ andprobability _m_ , computeof theandcorrect _z_ ˆ _j_ , andanswersample _y_ a _j_ conditioned _z j_ from _{z_ ˆ _j}_ onbased _{xi, z_ on _i, yi_ the _}i∈S_ weights. _j_ , _x j_ , _m_ In practice, however, the model likelihood of a given text may be inaccessible. Thus, we approximate it using reˆ ˆ jection sampling – we sample _z j_ by sampling _z j_ and _y j_ from _pLLM_ ( _z, y|{xi, zi, yi}_ ˆ _i∈S j , x j, m_ ) and then reject _z_ ˆ _j_ withˆ a probability of _pre j_ if _y j_ = _y j_ . Otherwise, we accept _z j_ and update the sample. Algorithm 1 shows the complete _Reprompting_ algorithm consisting of the initialization and iterative sampling steps. Note that we set the rejection probability _prej_ in a way that allows solutions that lead to incorrect answers to be kept occasionally, as these solutions may still contain useful segments that evolve into good recipes through _Reprompting_ . 

by randomly selecting K recipes from the population as parent recipes, which are then used to prompt the LLM to generate recipes for a new problem. By repeating the 2a and 2b steps, initial recipes can be recombined (Figure 4) and evolve into better recipes (Figure 3) through iterations. And eventually, the fittest recipes (i.e. ones that can be followed to solve similar problems) will survive. 

During testing, we select _K_ tuples _{xi, zi, yi}_ from the inferred _{z j}_ based on the training accuracy when using each tuple individually in a prompt. 

## **3. Experimental Setup** 

Based on the properties of Gibbs sampling (Casella & George, 1992; Roberts & Smith, 1994), the algorithm should converge to the point where the probability _pLLM_ ( _z j, y j|{xi, zi, yi}i∈S j , x j, m_ ) is high and agnostic to the choice of _S j_ , which leads to a set of _{z j}_ that work well as a prompt for solving similar problems in a separate test set. 

We evaluate the _Reprompting_ algorithm against various baselines including zero-shot, few-shot, Chainof-Thought (CoT), Chain-of-Thought combined with self-consistency decoding (Wang et al., 2022b), AutoCoT (Zhang et al., 2022) and Automatic Prompt Optimization (Pryzant et al., 2023) on 20 challenging reasoning tasks, including 12 challenging tasks in the BigBench Hard (BBH) benchmark (Suzgun et al., 2022),[2] 

The algorithm can also be viewed as a variant of evolutionary algorithms: 1) First, we generate the initial population of individuals (where each individual is a CoT recipe given a problem). 2) Next, we repeat the following regeneration steps iteratively: 2a) we first evaluate the fitness of each CoT recipe by comparing the answer that follows the recipe with the correct answer and weed out the least-fit recipes; 2b) we then breed new individuals through crossover and mutation 

> 2The BBH tasks include _Logical Deduction_ , _Geometric Shapes_ , _Object Counting_ , _Penguins in a Table_ , _Temporal Sequences_ , _Date Understanding_ , _Formal Fallacies_ , _Movie Recommendation_ , _Reasoning About Colored Objects_ , _Ruin Names_ , _Salient Translation Error Detection_ , and _Word Sorting_ . 

3 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). We choose both tasks that have been shown to benefit substantially from human-written CoT recipes, such as Logical Deduction, Geometric Shapes, Temporal Sequences, GSM8K and MATH, and tasks on which CoT does not improve much or does not improve consistently over zero-shot prompting, such as Formal Fallacies, Movie Recommendation and Word Sorting. 

## **3.1.** _**Reprompting**_ **Setup** 

For each task, we randomly select 20 training examples from the Big-Bench dataset excluding the test examples in the BBH benchmark.[3] We experiment with having _k ∈{_ 1 _,_ 3 _}_ clones of the same training example in the set _{xi, yi}[N] i_ =1[to allow for more diverse recipe samples (so] the number of recipes we need to sample from the joint distribution (3) is _N_ = 20 _∗ k_ ) and choose _k_ that obtains the highest training accuracy. We set the number of examples in the prompt by _K_ = 5. We run _Reprompting_ for a maximum of _M_ = 20 _,_ 000 iterations. We allow for early stopping if the average training accuracy stops increasing for 1 _,_ 000 iterations. For the rejection probability, we experiment with _prej ∈{_ 0 _._ 95 _,_ 0 _._ 99 _}_ and choose _pre j_ = 0 _._ 99 as it leads to higher training accuracy on various tasks. 

## **3.2. Baselines** 

**Prompting Baselines** For **zero-shot prompting** , we only include the test question _xi_ and the special message _m_ in the prompt, which triggers the model to generate a step-by-step solution prior to the answer text. For **few-shot prompting** , we randomly select 20 training examples in the same way as in _Reprompting_ and concatenate these examples in the form of question-answer pairs in the prompt, followed by the test question. For **CoT prompting** , we use the human-written CoT prompts from Suzgun et al. (2022). For **CoT with selfconsistency decoding** , we use the same CoT prompts and follow Wang et al. (2022b) by sampling 10 reasoning paths per question and taking the majority vote on the answer. For both approaches, we randomly select 20 training examples in the same way as in _Reprompting_ .[4] 

> 3Except for _Penguins in a Table_ where there are only three samples in the Big-Bench dataset that are excluded from BBH, so we randomly select 17 more examples from BBH into the training set. 

4Recent prompting methods that are more annotation-intensive, such as Complex-CoT (Fu et al., 2022) and Progressive-Hint Prompting (Zheng et al., 2023), are shown to outperform _Reprompting_ by 3.3–5.6 points on GSM8K. However, these methods leverage substantially more human-annotated examples (e.g. 7.5K annotated examples on GSM8K) than _Reprompting_ , thus they are not directly comparable. 

**Prompt Optimization Baselines** We also compare _Reprompting_ with two previous state-of-the-art prompt optimization algorithms, including **Auto-CoT** (Zhang et al., 2022) and **APO** (Pryzant et al., 2023). For **Auto-CoT** , since the original Auto-CoT algorithm differs from our setting as it focuses on the unsupervised setting without exploiting any labeled examples, we adapt the algorithm to our few-shot setting where it follows the original algorithm to generate diverse CoT recipes through zero-shot prompting but selects the demonstration examples based on the training accuracy when used individually in a prompt.[5] We also evaluate **APO** , a recently proposed nonparametric prompt optimization algorithm that uses LLMs to generate “textual gradient” – criticism of the current prompt – based on training samples and edit the prompt accordingly. The algorithm has been shown to outperform other prompt optimization methods, such as TEMPERA (Zhang et al., 2023), Automatic Prompt Engineering (Zhou et al., 2023), and AutoGPT.[6] 

## **3.3. Large Language Models (LLMs)** 

We experiment with two powerful LLMs including ChatGPT (gpt-3.5-turbo; OpenAI (2023)) and InstructGPT (textdavinci-003; Ouyang et al. (2022)). We also experiment with a combo model for _Reprompting_ where we use ChatGPT as _LLM_ 1 for initialization and InstructGPT as _LLM_ 2 for sampling. For both LLMs, we set the maximum number of output tokens to 500, _top_  p_ = 0 _._ 5, zero frequency and presence penalty. Additionally, we include “END” as the stop word. We set the temperature to 1.0 for _Reprompting_ and 0 _._ 0 for testing. 

## **3.4. Evaluation Protocol** 

We extract the final answer from the model output by extracting the text between “<answer>” and “</answer>”, except for the CoT baseline where we extract the final answer in the same way as in Suzgun et al. (2022). We measure accuracy based on exact match by comparing the extracted answer with the ground truth. 

## **4. Results** 

## **4.1. Main Results** 

We first compare the performance of _Reprompting_ with all the baselines on five BBH tasks. As shown in Table 1, results confirm the previous finding that few-shot in-context prompting improves the performance over zero-shot (Brown et al., 2020) and that CoT prompting outperforms both zeroshot and few-shot prompting by a large margin. However, human-written CoT prompting requires costly prompt en- 

5The original Auto-CoT algorithm selects the demonstration examples based on the diversity of the demonstration questions. 

> 6https://news.agpt.co/ 

4 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

|BBH Task|SOTA|ZS<br>FS<br>CoT<br>CoT+SC<br>APO<br>AutoCoT<br>ChatGPT|_Reprompting_<br>ChatGPT<br>InsGPT<br>Chat+Ins|
|---|---|---|---|
|Logical<br>Geometric<br>ObjectCount<br>Penguins<br>Temporal|60.4<br>56.0<br>93.2<br>81.5<br>96.8|35.1<br>46.4<br>63.1<br>62.7<br>28.0<br>53.2<br>13.6<br>20.0<br>58.0<br>60.0<br>52.0<br>52.4<br>52.4<br>46.8<br>95.6<br>95.2<br>74.8<br>88.8<br>50.7<br>60.3<br>67.1<br>71.2<br>45.2<br>**85.6**<br>38.4<br>41.2<br>66.8<br>66.8<br>50.4<br>80.8|**66.3**<br>53.7<br>60.0<br>**72.8**<br>40.8<br>64.4<br>97.2<br>42.8<br>**99.6**<br>**85.6**<br>78.1<br>82.9<br>93.2<br>28.4<br>**99.2**|
|**Average**|77.6|38.0<br>42.9<br>70.1<br>71.2<br>50.1<br>72.2|**83.0**<br>48.8<br>81.2|



Table 1: Performance of several large language models (LLMs) using _Reprompting_ versus the baseline prompting and prompt optimization methods on Big-Bench Hard (BBH) tasks. _SOTA_ refers to the state-of-the-art performance among InstructGPT (text-davinci-002; Ouyang et al. (2022)), Codex (Chen et al., 2021), and PaLM 540B (Chowdhery et al., 2022) using CoT prompting from Suzgun et al. (2022). We also compare _Reprompting_ with ChatGPT using _ZS_ (zero-shot), _FS_ (few-shot), _CoT_ , _CoT+SC_ (CoT prompting combined with self-consistency decoding (Wang et al., 2022b)), _APO_ (automatic prompt optimization using textual gradient (Pryzant et al., 2023)), and _AutoCoT_ (the few-shot version of Auto-CoT (Zhang et al., 2022)). For _Reprompting_ , we show the performance of various LLMs – including _ChatGPT_ (gpt-3.5-turbo; OpenAI (2023)), _InstructGPT_ (text-davinci-003), and _Chat+Instruct_ (a combo version that uses ChatGPT for initialization and InstructGPT at sampling steps). 

gineering, as not all CoT recipes work equally well on LLMs (Madaan & Yazdanbakhsh, 2022; Jojic et al., 2023). Crucially, we show that using _Reprompting_ , LLMs can achieve better performance compared to the existing CoT prompts, but without requiring any human guidance on how to solve problems step by step. Specifically, comparing the performance of ChatGPT using _Reprompting_ versus the best human-written CoT prompts from Suzgun et al. (2022), _Reprompting_ achieves consistently higher scores on all tasks. 

Next, we compare _Reprompting_ with self-consistency (SC) decoding (Wang et al., 2022b). CoT+SC improves over CoT on two of the five tasks, but the improvements are not consistent. By contrast, _Reprompting_ consistently outperforms CoT+SC by 2–26 points on all five tasks. 

Additionally, we compare _Reprompting_ with existing prompt optimization algorithms. APO improves over zeroshot prompting on three out of five tasks but underperforms it on the two tasks where the model needs to search through a wide range of strategies to find effective solutions. By contrast, _Reprompting_ consistently outperforms zero-shot and CoT prompting, and improves over APO by 20–43 points on all five tasks. When compared against Auto-CoT (Zhang et al., 2022), _Reprompting_ also archives higher accuracy by +11 points on average. In summary, _Reprompting_ outperforms strong decoding and prompt optimization baselines by 11–33 points on average. 

Comparing the performance of _Reprompting_ on different LLMs, we observe that InstructGPT underperforms ChatGPT on most tasks. However, we show that by using ChatGPT just as the initialization model _LLM_ 1 to bootstrap InstructGPT as _LLM_ 2 in _Reprompting_ , we can improve performance over InstructGPT alone by 5–71 points and achieve competitive or even better performance than ChatGPT alone on two of the five tasks. We show in the Appendix why that is: while InstructGPT can follow a given recipe and even be 

used for recombining and evolving them, it is less capable of generating diverse initial solutions in a zero-shot manner. However, through _Reprompting_ , we can use ChatGPT to “teach” InstructGPT diverse strategies for solving the training problems, which are then recombined and evolved by InstructGPT into better CoT prompts for itself. 

Furthermore, Table 2 shows the performance of _Reprompting_ against zero-shot, few-shot and CoT prompting (all using ChatGPT) on the remaining 15 tasks.[7] _Reprompting_ still outperforms zero-shot and few-shot prompting consistently and substantially by 14-15 points on average. Compared with CoT, _Reprompting_ achieves better performance on 11 out of 15 tasks. On average, _Reprompting_ outperforms CoT by +8.2 points. Interestingly, on tasks where CoT even underperforms zero-shot prompting, such as Movie Recommendation, Salient Translation Error Detection, and Word Sorting, _Reprompting_ still improves over zero-shot prompting by large margins. This suggests that not all CoT recipes improve model performance, and some may even lead to degradation. This further emphasizes the need for algorithms like _Reprompting_ for discovering and optimizing the CoT prompt to best exploit and compare LLMs. 

Overall, these findings highlight the potential of _Reprompting_ as a powerful method for automating CoT prompting on a wide range of tasks. 

## **4.2. Quantitative Analysis** 

**Ablation Study** We conduct an ablation study on the rejection sampling and recombination process. Results in Table 3 show that, without rejection sampling, the test performance degrades substantially by 25 point on average. Always 

7Based on the main results in Table 1, CoT+SC and Auto-CoT are more complicated than CoT but only slightly improves over CoT. Thus, we select CoT as a baseline here. 

5 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0006-01.png)


**----- Start of picture text -----**<br>
0.90 1.0<br>0.9<br>0.85 0.9 0.8<br>0.80<br>0.8 0.7<br>0.75<br>0.6<br>0.7<br>0.70<br>0.5<br>0.65 0.6<br>0.4<br>0.60<br>0.5 0.3<br>0.55<br>0 2000 4000 6000 8000 10000 0 2000 4000 6000 8000 10000 0 4000 8000 12000 16000 20000<br>Iterations Iterations Iterations<br>(a) InstructGPT (b) ChatGPT (c) ChatGPT+InstructGPT<br>Average Training Accuracy Average Training Accuracy Average Training Accuracy<br>**----- End of picture text -----**<br>


Figure 2: Learning curves of the _Reprompting_ algorithm using InstructGPT, ChatGPT, and the combo ChatGPT+InstructGPT models on the _Logical Deduction_ task. The y-axis shows the accuracy on training samples averaged over the current and all previous iterations. 

||ZS|FS|CoT|_Reprompting_|_prej_=0<br>_prej_=1|NoRec<br>Orig.|
|---|---|---|---|---|---|---|
|**BBH**|||||Logical<br>56.3<br>61.9|54.7<br>**66.3**|
|Date|63.6|46.4|**76.8**|76.4|ObjectCount<br>52.0<br>**97.2**|95.6<br>**97.2**|
|Formal|49.2|53.6|48.4|**56.8**|Temporal<br>74.8<br>74.4|90.4<br>**93.2**|
|Movie<br>ColoredObj|59.2<br>66.8|72.4<br>48.8|25.6<br>**76.0**|**78.4**<br>74.0|**Average**<br>61.0<br>77.8|80.2<br>**85.6**|
|Ruin|53.2|66.8|60.8|**74.8**|||
|Salient<br>WordSort|43.2<br>58.0|53.2<br>72.0|32.8<br>46.0|**54.8**<br>**73.2**|Table 3: Ablation study on rejection sampling (including no re<br>jection (_pre j_=0) and always rejecting (_pre j_=1)) and recombin<br>tion (_NoRec_represents_Reprompting_without recombination of pr||
|**GSM8K**|45.6|26.5|75.6|**79.5**|viously sampled recipes) on Logical Deduction, Object Countin<br>and Temporal Sequences from Big-Bench Hard (BBH) (Suzgu||
|**MATH**|||||et al.,2022). The_Orig._ column represents|the standard_Repromp_|
|Algebra|37.6|23.7|52.0|**53.1**|_ing_algorithm without ablation.||
|Counting<br>Geometry|17.1<br>12.4|19.8<br>16.2|26.6<br>28.5|**32.3**<br>**29.2**|Tasks<br>InsGPT|ChatGPT|
|IntAlgebra|9.4|12.1|**18.0**|16.8|Logical<br>65.9|**66.3**_∗_|
|Number|20.8|17.1|32.9|**33.3**|Geometric<br>53.6|**72.8**_∗_|
|Prealgebra|31.4|33.2|**54.0**|43.8|ObjectCount<br>**99.6**_∗_|96.8|
|Precalculus|7.4|18.4|19.0|**19.3**|Penguins<br>82.2|**85.6**_∗_|
|**Average**|38.3|38.7|44.9|**53.0**|Temporal<br>**99.2**_∗_|81.6|



Table 3: Ablation study on rejection sampling (including no rejection ( _pre j_ = 0) and always rejecting ( _pre j_ = 1)) and recombination ( _NoRec_ represents _Reprompting_ without recombination of previously sampled recipes) on Logical Deduction, Object Counting, and Temporal Sequences from Big-Bench Hard (BBH) (Suzgun et al., 2022). The _Orig._ column represents the standard _Reprompting_ algorithm without ablation. 

Table 4: Testing the best performing CoT prompt learned on ChatGPT, InstructGPT or InstructGPT+ChatGPT through _Reprompting_ on both ChatGPT and InstructGPT. The superscript _[∗]_ denotes the model used as _LLM_ 2 in _Reprompting_ . 

Table 2: Performance of ChatGPT using _Reprompting_ versus _ZS_ (zero-shot), _FS_ (few-shot), and _CoT_ prompting methods on seven additional tasks from Big-Bench Hard (BBH) (Suzgun et al., 2022), GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). 

work as well for other models. Specifically, we observe that on tasks such as _Logical Deduction_ and _Object Counting_ , the best CoT recipes achieve similar performance on both InstructGPT and ChatGPT. However, on _Geometric Shapes_ and _Temporal Sequences_ , the best CoT prompts optimized for _LLM_ 2 work well on _LLM_ 2, but poorly with the other LLM – using them on the other LLM leads to 18–19 points lower accuracy than testing with _LLM_ 2 (see examples in Figure A.2). On such tasks, using the prompt optimized for the testing LLM improves accuracy by 11–12 points over the same testing LLM with prompt optimized for other LLMs. These results suggest that, to make a fair comparison 

rejecting solutions that lead to incorrect answers also causes a degradation of 8 point. Additionally, not allowing multiple solutions to be recombined when sampling new solutions at the iterative sampling stage also hurts performance. 

**Do the generated CoT recipes generalize across models?** We test the best-performing CoT recipes optimized with InstructGPT, ChatGPT, or InstructGPT+ChatGPT through _Reprompting_ on both InstructGPT and ChatGPT. As shown in Table 4, the CoT recipes optimized for one model may not 

6 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0007-01.png)


Figure 3: An example of how the CoT recipes evolve through _Reprompting_ . In the left-most recipe, the model (ChatGPT) first reorders the constraints so that the ones with absolute ranking positions are considered prior to the ones with relative positions (highlighted in dark blue). Next, the model attempts to deduce the objects at specific positions but makes a mistake (see the red underlined part). Despite the error, this recipe still provides a useful strategy for solving similar problems – when it is used in a prompt to solve another problem, the model first adopts the same strategy to reorder the constraints and then proposes another way to deal with the constraints (highlighted in orange). Although the resulting solution still contains errors, it makes a good recipe for solving this type of problem. Thus, when using it in a new prompt to solve yet another problem, the model can follow the same recipe and deduce the correct answer. 

between different LLMs, one needs to optimize the CoT prompt for each model. 

_**Reprompting**_ **improves CoT recipes over iterations.** In Figure 2, we plot the average training accuracy (averaged over iterations up to the current iteration) over training iterations on _Logical Deduction_ . For all three model variants, the initial training accuracy is relatively low, but it gradually increases (with occasional fluctuations) over iterations until convergence. This is the result of evolution and recombination of the recipes associated with training examples. 

**Compute and Resources** We use the OpenAI APIs for all our experiments.[8] Running _Reprompting_ costs around $80 (in US dollars) on gpt-3.5-turbo and $800 on text-davinci003 based on the standard pricing,[9] while being exempted from any human cost. By contrast, CoT prompting requires manual prompt construction and engineering, which costs not only human labor (including the cost for humans to get familiar with the task itself and how LLM prompting works, write down various CoT solutions for each problem, test and optimize the solutions on the LLM) but also LLM queries, but these costs are typically neglected in previous works. In addition, previous works typically compare different LLMs using the same CoT prompt. While this strategy avoids additional costs for custimizing CoT prompt for each 

> 8https://platform.openai.com/docs/ api-reference?lang=python 

> 9https://openai.com/pricing 

LLM (even with _Reprompting_ , one can also save the cost by running it with ChatGPT and using the inferred CoT prompt on other LLMs), it risks making unfair comparisons as we have shown in Table 4 that the CoT prompt that works well on one model may be sub-optimal for another. 

## **4.3. Qualitative Analysis** 

We observe that **even model outputs containing errors and unreasonable deductions can evolve into a high-quality recipe through** _**Reprompting**_ **.** This is illustrated by the _Logical Deduction_ example in Figure 3, when _K_ = 1, where the model initially generates a recipe that is erroneous and contains illogical deductions. However, when this recipe is used as the new prompt for solving a similar problem, the model is able to exploit parts of the recipe and propose an alternative way to continue reasoning. Although the subsequent recipe still contains errors, it aids the model in correctly solving other problems when incorporated into a prompt. As a result, such recipes will be populated on other training samples, while the recipes that lead to low accuracy will eventually die out. 

_**Reprompting**_ **combines fragments from different recipes into a better one.** _Reprompting_ benefits from having multiple examples in the prompt, which allows the model to integrate various segments from different prompt recipes into a new recipe. As illustrated by the _Object Counting_ examples in Figure 4, the model can combine large segments of reasoning steps, as well as small segments that address 

7 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0008-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0008-02.png)


Figure 4: Examples of how fragments from different recipes in a prompt can be (re)combined into a better recipe to solve a new problem through _Reprompting_ . 

distinct cases to solve a more complex problem. The resulting prompts sometimes, but not always, share similarities with the human-written prompts (See the Appendix). 

## **5. Related Work** 

**In-Context Learning** is an emergent ability of LLMs as they scale up in model sizes and training data, where an LLMs can learn to perform a task from a few examples in the context (which is also referred to as few-shot prompting) (Brown et al., 2020). It has been shown to achieve promising few-shot and even zero-shot performance on various natural language processing (Brown et al., 2020; Schick & Schütze, 2020; Perez et al., 2021) and program synthesis (Austin et al., 2021) tasks. 

**Reasoning via Chain-of-Thought Prompting** Chain-ofThought (CoT) prompting is a technique that enables LLMs to perform complex reasoning tasks by prompting them with a few examples with step-by-step solutions (Wei et al., 2022; Suzgun et al., 2022). CoT prompting has been shown to improve performance on various reasoning tasks, such as arithmetic reasoning (Wei et al., 2022; Zhou et al., 2022), symbolic reasoning (Wei et al., 2022; Zhou et al., 2022), multi-hop question answering (Press et al., 2022; Arora et al., 2022), and natural language inference (Wang et al., 2022b). However, designing effective CoT prompts requires human experts with an understanding of both the task and the prompting technique (Zamfirescu-Pereira et al., 2023), which limits the scalability and generalizability of CoT prompting. 

Several works have attempted to **automate the process of CoT prompt discovery** . Zhang et al. (2022) proposed AutoCoT, which uses LLMs to generate CoT solutions for diverse training questions in zero-shot and integrates the generated 

CoT solutions in the prompt for solving test questions. This method differs from _Reprompting_ in that: 1) it focuses on the unsupervised setting and exploits a large set of example questions without annotated answers, and 2) it relies more heavily on the correctness of the zero-shot recipes as it does not have any iterative algorithm (as in _Reprompting_ ) to further improve the recipes. In our experiments, we adapted Auto-CoT to the few-shot setting and showed that _Reprompting_ outperforms the few-shot version of Auto-CoT. 

Deng et al. (2022); Zhang et al. (2023) proposed to train an additional policy model to find the best prompt through reinforcement learning, but their approaches are limited to prompt optimization within a relatively small search space (i.e. it is restricted to the prompts that are either extremely short or within a small edit distance from an initial prompt). Zhou et al. (2023) proposed a method for automatically generating, scoring and selecting effective instruction messages _m_ for zero-shot chain-of-thought reasoning, which is orthogonal and can be potentially combined with our algorithm. Paranjape et al. (2023) introduced a framework that automatically retrieves demonstrations of related tasks from a task library and generates CoT solutions for the new task. However, this framework still requires collective human efforts to write demonstrations for a diverse set of tasks in the task library. In contrast, our _Reprompting_ algorithm enables LLMs to solve complex reasoning tasks without any human guidance. Additionally, Yoran et al. (2023) proposed a multi-chain reasoning (MCR) method that prompts LLMs to combine pieces of information from multiple chains of thought to predict the final answer, which differs from our method in two ways: first, MCR combines multiple CoT solutions to the same question at test time, while _Reprompting_ combines CoT solutions generated for different training questions before testing; second, MCR combines solutions only once, whereas _Reprompting_ iteratively samples new 

8 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

solutions and recombines them. As a result, _Reprompting_ generates effective CoT recipes from only a few training examples, resulting in improved test performance without slowing down test inference. 

## **6. Conclusion** 

We introduce _Reprompting_ , an automated prompt inference algorithm which, without human effort, discovers effective chain-of-thought (CoT) prompts for each task given a few question-answer pairs. Experiments on 20 challenging reasoning tasks show that _Reprompting_ achieves +9.4 higher accuracy than human-written CoT on average. It also outperforms self-consistency decoding and the state-of-the-art prompt optimization algorithms by 11–33 points on average. Our results also suggest that LLM comparisons can be highly sensitive to the choice of CoT prompts, further emphasizing the need for automatic prompt discovery and optimization using algorithms such as _Reprompting_ . 

## **Acknowledgements** 

We thank Bill Dolan, Sudha Rao and the reviewers for their valuable feedback. 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## **References** 

- Arora, S., Narayan, A., Chen, M. F., Orr, L. J., Guha, N., Bhatia, K., Chami, I., Sala, F., and Ré, C. Ask me anything: A simple strategy for prompting language models. _arXiv preprint arXiv:2210.02441_ , 2022. 

- Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_ , 2021. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. _Neural Information Processing Systems (NeurIPS)_ , 2020. 

- Casella, G. and George, E. I. Explaining the gibbs sampler. _The American Statistician_ , 46(3):167–174, 1992. 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021. URL https://arxiv. org/abs/2107.03374. 

- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. PaLM: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Creswell, A., Shanahan, M., and Higgins, I. Selectioninference: Exploiting large language models for interpretable logical reasoning. _arXiv preprint arXiv:2205.09712_ , 2022. 

- Deng, M., Wang, J., Hsieh, C.-P., Wang, Y., Guo, H., Shu, T., Song, M., Xing, E. P., and Hu, Z. Rlprompt: Optimizing discrete text prompts with reinforcement learning. _arXiv preprint arXiv:2205.12548_ , 2022. 

- Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning. _arXiv preprint arXiv:2210.00720_ , 2022. 

- Geman, S. and Geman, D. Stochastic relaxation, gibbs distributions, and the bayesian restoration of images. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , PAMI-6:721–741, 1984. 

- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. _CoRR_ , abs/2103.03874, 2021. URL https://arxiv. org/abs/2103.03874. 

- Jojic, A., Wang, Z., and Jojic, N. Gpt is becoming a turing machine: Here are some ways to program it, 2023. 

9 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. _arXiv preprint arXiv:2205.11916_ , 2022. 

- Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J.-G., and Chen, W. On the advance of making language models better reasoners. _arXiv preprint arXiv:2206.02336_ , 2022. 

- Liu, Z., Patwary, M., Prenger, R., Prabhumoye, S., Ping, W., Shoeybi, M., and Catanzaro, B. Multi-stage prompting for knowledgeable dialogue generation. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pp. 1317–1337, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. findings-acl.104. URL https://aclanthology. org/2022.findings-acl.104. 

- Madaan, A. and Yazdanbakhsh, A. Text and patterns: For effective chain of thought, it takes two to tango. _arXiv preprint arXiv:2209.07686_ , 2022. 

- Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

OpenAI. Gpt-4 technical report, 2023. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. _arXiv preprint arXiv:2203.02155_ , 2022. 

- Paranjape, B., Lundberg, S., Singh, S., Hajishirzi, H., Zettlemoyer, L., and Ribeiro, M. T. Art: Automatic multi-step reasoning and tool-use for large language models. _arXiv preprint arXiv:2303.09014_ , 2023. 

- Perez, E., Kiela, D., and Cho, K. True few-shot learning with language models. _Advances in neural information processing systems_ , 34:11054–11070, 2021. 

- Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., and Lewis, M. Measuring and narrowing the compositionality gap in language models. _arXiv preprint arXiv:2210.03350_ , 2022. 

- Pryzant, R., Iter, D., Li, J., Lee, Y., Zhu, C., and Zeng, M. Automatic prompt optimization with “gradient descent” and beam search. In Bouamor, H., Pino, J., and Bali, K. (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 7957–7968, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.494. URL https://aclanthology. org/2023.emnlp-main.494. 

- Roberts, G. O. and Smith, A. F. Simple conditions for the convergence of the gibbs sampler and metropolis-hastings algorithms. _Stochastic processes and their applications_ , 49(2):207–216, 1994. 

- Schick, T. and Schütze, H. Exploiting cloze questions for few shot text classification and natural language inference. _arXiv preprint arXiv:2001.07676_ , 2020. 

- Shwartz, V., West, P., Le Bras, R., Bhagavatula, C., and Choi, Y. Unsupervised commonsense question answering with self-talk. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 4615–4629, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.373. URL https:// aclanthology.org/2020.emnlp-main.373. 

- Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _arXiv preprint arXiv:2206.04615_ , 2022. 

- Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ , 2022. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., and Zhou, D. Rationale-augmented ensembles in language models. _arXiv preprint arXiv:2207.00747_ , 2022a. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022b. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022. 

- Yoran, O., Wolfson, T., Bogin, B., Katz, U., Deutch, D., and Berant, J. Answering questions by metareasoning over multiple chains of thought. _arXiv preprint arXiv:2304.13007_ , 2023. 

- Zamfirescu-Pereira, J., Wong, R. Y., Hartmann, B., and Yang, Q. Why johnny can’t prompt: How non-ai experts try (and fail) to design llm prompts. In _Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems_ , CHI ’23, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394215. doi: 10.1145/3544548.3581388. URL https://doi. org/10.1145/3544548.3581388. 

10 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 

- Zelikman, E., Wu, Y., and Goodman, N. D. STaR: Bootstrapping reasoning with reasoning. _arXiv preprint arXiv:2203.14465_ , 2022. 

- Zhang, T., Wang, X., Zhou, D., Schuurmans, D., and Gonzalez, J. E. TEMPERA: Test-time prompt editing via reinforcement learning. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https: //openreview.net/forum?id=gSHyqBijPFO. 

- Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. _arXiv preprint arXiv:2210.03493_ , 2022. 

- Zheng, C., Liu, Z., Xie, E., Li, Z., and Li, Y. Progressivehint prompting improves reasoning in large language models. _arXiv preprint arXiv:2304.09797_ , 2023. 

- Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Bousquet, O., Le, Q., and Chi, E. Least-to-most prompting enables complex reasoning in large language models. _arXiv preprint arXiv:2205.10625_ , 2022. 

- Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., and Ba, J. Large language models are human-level prompt engineers. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum? id=92gvk82DE-. 

11 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0012-01.png)


Figure A.1: Comparing the CoT recipes inferred through _Reprompting_ using IntructGPT alone versus ChatGPT (for initialization) + InstructGPT (for sampling). 

## **A. Additional Illustrations** 

**On sensitivity to initialization** We have shown that _Reprompting_ can be sensitive to initial recipe generation. Armed with the optimal prompts discovered with ChatGPT+InstructGPT through _Reprompting_ , InstructGPT can reach test accuracy equalling or besting ChatGPT on most challenging reasoning tasks. However, on some tasks, such prompts could not be discovered using InstructGPT itself as the initialization model _LLM_ 1. Figure A.1 points to a likely explanation: ChatGPT can generate a wider range of useful recipes, and whether these initial recipes lead to the correct solution or not, InstructGPT can follow them and, through _Reprompting_ , refine and correct them iteratively. Thus, as we have shown in our experiments, with a diverse pool of initial recipes, LLMs that may appear inferior based on their zero-shot performance may end up performing just as well or better than LLMs whose zero-shot performance is more encouraging. It would be interesting to see if _Reprompting_ can use a mixture of LLMs in initialization to perform even better, or if humans can be put back into the loop to provide some initial recipes or some generic instructions on how to generate such recipes. 

**On transferability of discovered recipes** The fact that _LLM_ 1 (ChatGPT) can point _LLM_ 2 (InstructGPT) in the right directions for prompt discovery does not mean that the discovered prompts, having been optimized for training performance on _LLM_ 2, will perform well when used to prompt _LLM_ 1. In fact, Table 4 indicates that the discovered CoT recipes that work for one model may not necessarily work for other models. For example, in the case of _Temporal Sequences_ , the best performance is achieved with a prompt trained with InstructGPT (after initialization with ChatGPT as _LLM_ 1). But when using that prompt on ChatGPT, the test performance is by 18% lower. Figure A.2 illustrates how ChatGPT and InstructGPT follow the same CoT prompt differently. Following the prompt recipes, the time intervals that need to be reasoned over are sorted, and among the sorted list, the missing interval was inserted as the possible interval when the person in question could have performed an activity. InstructGPT follows this procedure with accuracy over 99%, but ChatGPT sometimes skips the crucial line (for this recipe) with the missing interval within the timeline and therefore obtains suboptimal test accuracy. However, the best performance of ChatGPT (using the CoT prompt optimized for itself through _Reprompting_ ) is only slightly lower than that of the ChatGPT+InstructGPT combination. 

These results suggest that, for a fair comparison between different LLMs, one needs to optimize the CoT prompt for each LLM using prompt optimization algorithms such as _Reprompting_ . 

12 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0013-01.png)


Figure A.2: An example on _Temporal Sequences (BBH)_ where ChatGPT underperforms InstructGPT using the same CoT prompt optimized for InstructGPT via _Reprompting_ (using ChatGPT+InstructGPT). ChatGPT fails to correctly execute the recipe as it skips a key step (the blue underlined text from InstructGPT) to reach the final answer. (The illustration does not show the full prompt that precedes the puzzle _x_ for brevity; it consists of 5 training examples with worked-out solutions that all follow the same strategy of solving these types of problems.) 

**How do the model-generated CoT recipes differ from human-written ones?** In the paper, We evaluated the performance of the CoT prompt discovered through _Reprompting_ and contrasted it with human-written ones. As illustrated by the example recipes in Figure A.3, the automatically discovered CoT recipes share some similarities to human-written ones on some tasks (such as _Logical Deduction_ ), but differs on other tasks. For instance, on _Object Counting_ , the CoT generated using _Reprompting_ computes the total number of objects by incrementing the count one by one (e.g. adding 4 to the count 5 by “[6 _,_ 7 _,_ 8 _,_ 9]”), while in the human written recipe, it computes the addition through an arithmetic formula at the end. 

13 

**Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0014-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling_images/icml_2024_Reprompting_Automated_Chain-of-Thought_Prompt_Inference_Through_Gibbs_Sampling.pdf-0014-02.png)


**----- Start of picture text -----**<br>
(a)<br>(b)<br>**----- End of picture text -----**<br>


Figure A.3: Examples of the best-performing CoT recipes inferred via _Reprompting_ on _Logical Deduction_ , _Geometric Shapes_ , _Object Counting_ , _Penguins in a Table_ , and _Temporal Sequences_ . 

14 

