# **Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation** 

**Jingchang Chen** _[∗]_ **Hongxuan Tang** _[∗]_ **Zheng Chu** Harbin Institute of Technology Harbin Institute of Technology Harbin Institute of Technology `jcchen@ir.hit.edu.cn jeffswt@outlook.com zchu@ir.hit.edu.cn` 

**Qianglong Chen** _[†]_ Zhejiang University `chenqianglong.ai@gmail.com` 

**Zekun Wang** Harbin Institute of Technology `zkwang@ir.hit.edu.cn` 

**Ming Liu** Harbin Institute of Technology `mliu@ir.hit.edu.cn` 

**Bing Qin** _[†]_ Harbin Institute of Technology `qbin@ir.hit.edu.cn` 

## **Abstract** 

Despite recent progress made by large language models in code generation, they still struggle with programs that meet complex requirements. Recent work utilizes plan-and-solve decomposition to decrease the complexity and leverage selftests to refine the generated program. Yet, planning deep-inside requirements in advance can be challenging, and the tests need to be accurate to accomplish self-improvement. To this end, we propose FUNCODER, a code generation framework incorporating the divide-and-conquer strategy with functional consensus. Specifically, FUNCODER recursively branches off sub-functions as smaller goals during code generation, represented by a tree hierarchy. These sub-functions are then composited to attain more complex objectives. Additionally, we designate functions via a consensus formed by identifying similarities in program behavior, mitigating error propagation. FUNCODER outperforms state-of-the-art methods by +9.8% on average in HumanEval, MBPP, xCodeEval and MATH with GPT-3.5 and GPT-4. Moreover, our method demonstrates superiority on smaller models: With FUNCODER, StableCode3 _b_ surpasses GPT-3.5 by +18.6% and achieves 97.7% of GPT-4’s performance on HumanEval. Further analysis reveals that our proposed dynamic function decomposition is capable of handling complex requirements, and the functional consensus prevails over self-testing in correctness evaluation. 

## **1 Introduction** 

Over the past few years, large language models have been observed to attain significant advancements in coding capabilities (OpenAI, 2023; Touvron et al., 2023). Meanwhile, models designed specifically for coding tasks have also been introduced (Rozière et al., 2023; Lozhkov et al., 2024; Pinnaparaju et al., 2024). Although LLMs can proficiently generate simple code snippets, they suffer from a decline in performance as code requirements become complicated. 

Numerous efforts have been made to tackle this complexity. The two-stage methods (Jiang et al., 2023; Zelikman et al., 2023) employ the plan-and-solve strategy, which first generates a draft outline 

> _∗_ Equal contribution. 

> _†_ Corresponding Authors: Bing Qin, Qianglong Chen. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation_images/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
FUNCODER [(            )] 𝑓(𝑥)<br>Sampled Functional Consensus<br>Implementations 𝒇 [∗] 𝒙=<br>𝒇′ 𝒙= divide     conquer 𝑓(")$ ⋮<br>𝑓(%)$ 𝑓$&'"<br>⋮ 𝑓$&<br>FUNCODER 𝑥"<br>𝑔(𝑦) 𝑔 [∗] (𝑦) 𝑋= 𝑥# 𝑔 [∗] (𝑦)<br>FUNCODER …<br>ℎ(𝑧) ℎ [∗] (𝑧) 𝑥$ ℎ [∗] (𝑧)<br>Possible Inputs<br>Legends Example Divide(𝑑, 𝑒) Divide( c )<br>Divide(𝑏) Conquer(𝐷, 𝐸) Conquer(𝐵) Conquer(𝐶) Conquer(𝐴)<br>𝑓(𝑥) Problem Divide(𝑎)<br> 𝐴  𝐴  𝐴  𝐴  𝐴<br> 𝐴<br>𝑓′(𝑥) Partially Implemented  𝑎  𝐵  𝑐  𝐵  𝑐  𝐵  𝑐  𝐵  𝐶  𝐵  𝐶<br> 𝑏  𝑐<br>𝑓 [∗] (𝑥) Reached Consensus  𝑑  𝑒  𝐷  𝐸  𝐷  𝐸  𝐷  𝐸  𝐷  𝐸<br>**----- End of picture text -----**<br>


Figure 1: A flowgraph illustrates FUNCODER. FUNCODER branches off new functions to have sub-goals tackled iteratively (left), re-composites sub-functions, and selects the best using functional consensus (right). Bottom-right figure shows how FUNCODER writes functions at hierarchy-level. 

for the complex task and uses it as guidance for implementing the code in the second stage. Multiagent development frameworks (Hong et al., 2024; Qian et al., 2023) mimic real-world software development workflows, assign different roles to LLMs and collaborate to solve a complex goal. Self-improvement (Shinn et al., 2023; Chen et al., 2024), on the other hand, refines the program in accordance with execution feedback from self-generated unit tests. 

Despite fruitful efforts made by the previous methods in dealing with complex problems, certain challenges still remain unsolved: (1) Two-stage approaches need to design a complete plan at the beginning and lack the ability to adjust the top-level design during implementation, leading to suboptimal decomposition. (2) Multi-agent collaboration frameworks are cumbersome and rely heavily on LLM capabilities, making them difficult to generalize to smaller open-source models. (3) Code refinement through self-tests depends on the correctness of generated unit-tests. Our preliminary study (§3.1.3) finds that models generate unreliable self-tests in abundance. These incorrect tests may mislead self-improvement and, at worse, exacerbate program errors. 

To address these issues, we propose FUNCODER, a code generation framework utilizing a divide-andconquer strategy and a novel functional consensus mechanism on functions to decompose complex problems. Starting from the main problem, FUNCODER introduces new functions to cope with certain sub-problems. The new functions will be decomposed recursively, eventually forming a tree of functions. FUNCODER then combines functions bottom-up to achieve increasingly complicated objectives. By dividing-and-conquering tasks into simpler sub-functions, complexity can be gradually reduced. However, errors in sub-functions may propagate to the whole program, thereby damaging overall reliability. We propose functional consensus that samples multiple functions and selects the one demonstrating consensus, measured by the aggregated similarity among candidates. By reaching a consensus, we reduce the discrepancies in code behavior and thus alleviate cascading errors. 

We conduct extensive experiments on code generation benchmarks (Chen et al., 2021; Austin et al., 2021; Khan et al., 2023) with GPT-3.5 (Ouyang et al., 2022) and GPT-4 (OpenAI, 2023), outperforming state-of-the-art methods by + **9** _._ **8** % on average. Experiments are further carried out on the mathematical competition benchmark, MATH (Hendrycks et al., 2021b), achieving a + **6** _._ **0** improvement with GPT-4, indicating that FUNCODER can also generalize to complex reasoning. Our method is observed to be equally effective on open-source models (Meta AI, 2024; Mistral AI, 2024; Pinnaparaju et al., 2024; Rozière et al., 2023; Lozhkov et al., 2024), with an average gain over baseline of + **31** _._ **5** % on HumanEval and + **47** _._ **7** % on MATH. Additional analysis also shows the advantage of both divide-and-conquer and functional consensus. Our code is made openly available at `https://github.com/cometeme/funcoder` . 

2 

|**Algorithm 1**FUNCODER procedure<br>**Require:** Entry func,_f_root =_{h_root_, d_root_, ϕ}_<br>**Require:** Large language model, LLM<br>1: **function**FUNCODER(_f_cur)<br>2:<br>_— Divide —_<br>3:<br>_f ′_<br>cur_, {fi} ←_EXTRACT(LLM(_f_cur))<br>4:<br>**for**_fi ∈{fi}_**do**<br>5:<br>**if**_bi_ **is**NOTIMPLEMENTED**then**<br>6:<br>_f ∗_<br>_i ←_FUNCODER(_fi_) _▷_recursion<br>7:<br>**end if**<br>8:<br>ADDCHILD(_f_cur_, f ∗_<br>_i_ )<br>9:<br>**end for**<br>10:<br>_— Conquer —_<br>11:<br>_F_cur _←_SAMPLE(LLM(_f ′_<br>cur_,_CHILD(_f_cur))<br>12:<br>_f ∗_<br>cur _←_FUNCONSENSUS(_F_cur)<br>13:<br>**return**_f ∗_<br>cur<br>14: **end function**<br>15: **return**FUNCODER(_f_root)<br>_▷_starts from root|_plan_<br>**1.**Create a function …<br>**2.**Generate a sequence<br>**3.**Check if ... then ...<br>**4.**Return ...<br>**(a)Planning-based Decomposition**<br>_code_<br>**Q**|
|---|---|
||**(b) Decompose Through Coding (ours)**<br>𝑓<br>𝑔<br>ℎ<br>3. Extract<br>code to a tree<br>1. Writing current function<br>deffind_factors(x:int) ->list:<br>""“return factors of x"""<br>raiseNotImplementedError()<br>defsum_common_factors(a,b):<br>"""sum the common factors"""<br>fa=find_factors(a)<br>fb=find_factors(b)<br>returnsum_common(fa, fb)<br>defsum_common(a:list,b:list) ->int:<br>""“sum of common elements"""<br>raiseNotImplementedError()<br>2. Introduce new functions|



Figure 2: Left: Algorithm for FUNCODER, explained in detail in Appendix A.6. Right: Comparison between decomposition by planning and our approach. FUNCODER introduces new functions to describe sub-goals solely with code, achieving a more natural way of requirement decomposition. 

## **2 FUNCODER: Divide-and-Conquer Meets Consensus** 

## **2.1 Divide-and-Conquer for Iterative Programming** 

A function is defined as a relation between a set of inputs and outputs where each input is assigned exactly one output (Halmos, 1998), denoted as _y_ = _f_ ( _x_ ). In computer programming, a function is identified by its header _hf_ with its body _bf_ , and is commonly accompanied by a documentation _df_ to improve readability. Functions can be invoked from other procedures, allowing for the decomposition of large and complicated requirements into smaller structures that exhibit high comprehensibility and quality (Dahl et al., 1972). Generally, human programmers tend to decompose tasks into clearly defined sub-functions and then implement them recursively, making functions eligible for re-usage, taking advantage of the _divide-and-conquer_ principle. Inspired by this, FUNCODER recursively _divides_ the requirement and _conquers_ functions to formulate a sophisticated solution, unleashing the potential of LLMs in code generation. 

**Divide** is a top-down process that iteratively breaks down problems. Given a code generation problem, the process begins from the entry function _f_ root. We instruct the model to introduce new functions _fi ∈_ CHILD( _f_ cur) that solve certain sub-goals while writing the current _f_ cur. To reduce the complexity involved in each generation, we only require the headers _hfi_ and documentation _dfi_ of new functions to be generated, while their implementations _bfi_ can be postponed. After completing the current function, the model starts to address those unimplemented sub-functions and complete _bfi_ into _fi[′]_[.][This process stops when the model deems functions too simple to be further divided, finally] forming a dependency tree _T_ = TREE( _f_ root _,_ CHILD( _f_ root)). The _divide_ process is similar to a search starting from the entry function, gradually involving new sub-functions while writing the current, and implementing them recursively. We guide the entire process through a depth-first search. 

**Conquer** is a process of achieving complex objectives through aggregating smaller functions. We notice that child functions are not yet implemented during the top-down process of writing parent functions. As a result, these parent functions may not be able to effectively utilize the child functions, or misuse them at worst. FUNCODER deals with this issue by re-generating functions in inverse topological order on the dependency tree _T_ - starting from leaves, complex goals are handled by compositing solved children as _f_ cur _[∗][←F]_[(] _[f]_ cur _[ ′][,][ {][f]_ 1 _[ ∗][, f]_ 2 _[ ∗][, . . .][ }]_[)] _[ |][ f] i[ ∗][∈]_[C][HILD][(] _[f]_[cur][)][.] 

_Divide_ and _conquer_ naturally achieve both decomposition and composition during code generation. Unlike two-stage and agent-based methods, our approach dynamically introduces new functions 

3 

along the process, making it less burdensome than producing a complete plan at the very beginning. Moreover, while planning or agents require chat capabilities, FUNCODER represents sub-tasks through functions (Figure 2), making it more applicable to specialized code generation models. 

## **2.2 Functionality Similarity as a Consensus** 

The decomposition of complex tasks benefits from solving easier sub-goals, but might introduce the risks of cascading errors, which refers to errors in sub-functions that lead to errors in ancestral functions. To mitigate this, we introduce Functional Consensus which aims at reducing inconsistencies in program behavior. This is achieved by sampling multiple functions and selecting the one that exhibits consensus, as measured by the aggregated similarity of functionality between candidates, thus abating outlier functionalities. 

**Functionality Similarity** A program specifies its functionality (or behavior) through the control flow and logic defined by its code semantics. However, comparing the functionalities between two programs based on their semantics is somewhat challenging. By decomposing the requirement into functions, FUNCODER is able to view the function behavior as a black box that maps arguments into return values. Considering two functions _f_ and _g_ with the same input domain _D_ ( _f_ ) = _D_ ( _g_ ), we define the similarity between them _sim_ ( _f, g_ ) as the identicalness of outputs when given the same input values. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation_images/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation.pdf-0004-04.png)


The similarity becomes 1 if and only if two functions output consistent values for all inputs: _∀x ∈ D_ ( _f_ ) : _f_ ( _x_ ) = _g_ ( _x_ ) _⇔ sim_ ( _f, g_ ) = 1. We notice that the input domain _D_ ( _f_ ) is unbounded in most cases, making its measurement barely feasible in practice. Thus, we approximate it by sampling a subset of possible inputs _X ∼ D_ ( _f_ ) with an LLM. 

**Consensus** is reached by selecting the candidate _f[∗]_ holding maximal similarity with others after sampling multiple function implementations _F_ = _{f_ ( _i_ ) _}_ for the same requirements. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation_images/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation.pdf-0004-07.png)


By introducing functional consensus, FUNCODER produces functions that are more consistent and common in functionality, while omitting abnormal samples. The process is applied to not just the final program, but also to every sub-tree during the bottom-up _conquering_ stage, resulting in step-by-step, thorough verification from the most fundamental functions all the way up to the whole program. 

## **2.3 FUNCODER is a Function Coder** 

We design FUNCODER as a procedure that takes a problem in the form of a function signature _f_ ( _x_ ), and produces a final solution _f[∗]_ ( _x_ ), as exemplified in Figure 1. Given a problem _f_ ( _x_ ), FUNCODER partially implements the function as _f[′]_ ( _x_ ) referring to unimplemented sub-functions _g_ ( _y_ ) and _h_ ( _z_ ). These sub-functions are then fed into FUNCODER to be recursively coped with. We then sample _k_ implementations _f_ ( _[′] i_ )[(] _[x]_[)][based][on][solved][children] _[g][∗]_[(] _[y]_[)][and] _[h][∗]_[(] _[z]_[)][.][Functional][consensus][is] calculated by evaluating candidates on possible inputs. The function sharing maximal behavioral similarity is combined with solved children to formulate the final solution. 

## **3 Experiments** 

We conduct experiments on competition-level code generation and mathematical reasoning benchmarks with state-of-the-art LLMs, which are covered in section §3.1 and §3.2, respectively. In addition to GPT models (Ouyang et al., 2022; OpenAI, 2023), we also conduct experiments with community models like Llama38 _b_ (Meta AI, 2024), StableCode3 _b_ (Pinnaparaju et al., 2024), and CodeLlama34 _b_ (Rozière et al., 2023). We use the _instruct_ variant of these models and inference on a single A100-80G under BF16 precision with vLLM (Kwon et al., 2023). 

4 

Table 1: Experiment results on code generation benchmarks. We report Pass@1 as evaluate metric. Results from the original paper are underlined, and the best results are bold. 

|**Model**<br>**Method**|**HumanEval**<br>Pass@1<br>∆_↑_|**HumanEval**<br>Pass@1<br>∆_↑_|**MBPP**<br>Pass@1<br>∆_↑_|**xCodeEval**<br>_Easy_<br>_Mid_<br>_Hard_<br>_Expert_<br>All|
|---|---|---|---|---|
|GPT-3.5<br>Standard<br>CodeT<br>Refexion<br>LDB<br>FUNCODER||68.3<br>-<br>81.1<br>+12.8<br>69.5<br>+1.2<br>82.9<br>+14.6<br>**85.4**<br>+17.1|72.0<br>-<br>76.0<br>+4.0<br>72.5<br>+0.5<br>76.0<br>+4.0<br>**78.5**<br>+6.5|44.4<br>15.2<br>4.6<br>0.0<br>20.2<br>50.6<br>16.1<br>8.0<br>0.0<br>23.2<br>44.4<br>17.0<br>5.7<br>0.0<br>20.6<br>-<br>-<br>-<br>-<br>-<br>**62.4**<br>**29.5**<br>**11.6**<br>0.0<br>**31.4**|
|GPT-4<br>Standard<br>Parsel<br>CodeT<br>Refexion<br>MetaGPT<br>FUNCODER||82.9<br>-<br>85.0<br>+2.1<br>90.9<br>+8.0<br>91.0<br>+8.1<br>85.9<br>+3.0<br>**94.5**<br>+11.6|73.5<br>-<br>-<br>-<br>77.0<br>+3.5<br>77.1<br>+3.6<br>-<br>-<br>**79.5**<br>+6.0|68.5<br>39.3<br>19.5<br>1.7<br>37.4<br>-<br>-<br>-<br>-<br>-<br>76.4<br>51.8<br>21.8<br>**3.4**<br>44.0<br>71.3<br>41.1<br>19.5<br>2.5<br>38.6<br>-<br>-<br>-<br>-<br>-<br>**83.1**<br>**58.0**<br>**26.4**<br>**3.4**<br>**48.6**|
|Llama38_b_<br>Standard<br>CodeT<br>FUNCODER||61.6<br>-<br>68.9<br>+7.3<br>**79.7**<br>+18.1|60.5<br>-<br>61.5<br>+1.0<br>**62.5**<br>+2.0|9.0<br>**1.8**<br>0.0<br>0.0<br>3.6<br>12.4<br>0.0<br>0.0<br>0.0<br>4.4<br>**22.0**<br>0.9<br>0.0<br>0.0<br>**8.0**|
|StableCode3_b_<br>Standard<br>CodeT<br>FUNCODER||61.0<br>-<br>75.0<br>+14.0<br>**81.0**<br>+20.0|51.5<br>-<br>57.5<br>+6.0<br>**63.5**<br>+12.0|7.3<br>0.9<br>0.0<br>0.0<br>2.8<br>11.2<br>1.8<br>0.0<br>0.0<br>4.6<br>**13.5**<br>**4.5**<br>**1.1**<br>0.0<br>**6.2**|
|CodeLlama34_b_<br>Standard<br>CodeT<br>FUNCODER||43.9<br>-<br>55.5<br>+11.6<br>**66.5**<br>+22.6|53.5<br>-<br>56.5<br>+3.0<br>**58.5**<br>+5.0|2.3<br>0.0<br>0.0<br>0.0<br>0.8<br>10.1<br>0.0<br>0.0<br>0.0<br>3.6<br>**10.2**<br>0.0<br>0.0<br>0.0<br>**3.6**|



## **3.1 Code Generation** 

We choose three benchmarks for code generation evaluation: (a) HumanEval (Chen et al., 2021) includes entry-level coding questions; (b) MBPP (Austin et al., 2021) contains questions of standard library invocation and programming basics; and (c) xCodeEval (Khan et al., 2023) consists of algorithmic challenges sourced from the competitive programming platform CodeForces. 

## **3.1.1 Experiment Setup** 

**Benchmarks** We adopt the full test set (164 problems) for HumanEval, and sample 200 for MBPP and 500 for xCodeEval, respectively. Following EbTech (2024), we split the xCodeEval into 4 subsets based on problem difficulty: Easy ( _≤_ 1200), Mid (1200-1599), Hard (1600-1999) and Expert ( _≥_ 2000). The evaluation metric for code generation is Pass@1 unless specified. 

**Baselines** We compare FUNCODER with standard prompting (Brown et al., 2020), two-stage decomposition method Parsel (Zelikman et al., 2023), self-testing method CodeT (Chen et al., 2023a), self-improvement methods Reflexion and LDB (Shinn et al., 2023; Zhong et al., 2024), and multiagent developing framework MetaGPT (Hong et al., 2024). We implement Standard prompting with a 1-shot demonstration. CodeT samples 11 solutions with standard prompting and evaluates them on model-generated tests. The results for Reflexion are reproduced from the original code. 

**Implementation Details** FUNCODER uses a 2-shot prompt in the divide stage and 1-shot for conquering sub-functions. The number of sampled implementations in the functional consensus is set to 11 for code generation tasks. For further implementation details, please refer to Appendix A.1. 

## **3.1.2 Results** 

Table 1 shows the code generation performance on advanced proprietary models, GPT-3.5 (Ouyang et al., 2022) and GPT-4 (OpenAI, 2023). For basic programming questions, HumanEval and MBPP, FUNCODER surpass previous SOTA methods by +3.3% in Pass@1 and reduce the error rate by 18.6%. Furthermore, FUNCODER demonstrates a substantial improvement on competition-level problems, outperforming others by 10.4% in GPT-4 and 35.3% with GPT-3.5. We observe that FUNCODER can 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation_images/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation.pdf-0006-00.png)


**----- Start of picture text -----**<br>
(a) Preliminary Study on Self-testing (b) Effectiveness of Ranking Strategy<br>90.2<br>final passed 43.9% 44.5% 90<br>85.4<br>final failed 4.3% 19.5% 85<br>program wrong 12.8% 5.5% 80.5 Strategy 80<br>unit-test wrong 25.0% 16.5% Self-Test Result passed consensusself-test 75<br>both incorrect 14.0% 14.0% failed 69.5 random 70<br>50 40 30 20 10 0 0 10 20 30 40 50 1 2 3 4 5 6 7 8 9 10 11<br>GPT-3.5 StableCode 3 b Num Selected Programs<br>Pass@k<br>**----- End of picture text -----**<br>


Figure 3: (a) Preliminary study on self-testing, the programs are evaluated using unit-tests generated by LLMs. (b) The effectiveness of different ranking strategies. We compute the Pass@k over top-k programs ranked by functional consensus, self-test, and random on 11 candidates. (higher is better) 

enhance LLM’s capability of solving more complex programming tasks, with an average accuracy improvement of 82.3% over the baseline on the _Mid_ and _Hard_ subsets of xCodeEval. _Expert_ level programs, however, still remain a colossal challenge for even the most cutting-edge LLMs. 

Evaluation is also performed over community LLMs, Llama3 (Meta AI, 2024), StableCode (Pinnaparaju et al., 2024) and CodeLlama (Rozière et al., 2023) with results in Table 1. FUNCODER consistently boosts the performance of smaller models in code generation, demonstrating notable improvements compared to standard prompting on HumanEval, which gained +29.4% on Llama3, +32.8% on StableCode, and even +51.5% on CodeLlama, outperforming that from the previous best method CodeT. We also supplement results on GPT-4o mini, Codestral and StarCoder2 in Table 11. Experiment results demonstrate that our method archives state-of-the-art performance on various models, ranging from basic programming to competition contests. 

## **3.1.3 Analysis** 

**FUNCODER Democratize to Smaller LLMs** Limited by the LLM capabilities, the application of selfimprovement or multi-agent methods on smaller models is without ease. By keeping decomposition and composition within the code generation process, our approach exhibits better generalization. As shown in Table 1, with FUNCODER, StableCode3 _b_ achieves around 118 _._ 6% relative performance to standard GPT-3.5, and also aligns closely with GPT-4 by about 97 _._ 7% on HumanEval. 

**Preliminary Study on Self-Testing Method** We conduct a preliminary study targeting the self-testing method on HumanEval, results are shown in Figure 3.a with further details in Appendix A.5. We first verify whether model-generated programs can also pass model-generated self-tests: (a) If a program passes self-tests, most from GPT-3.5 would also work on system tests, as much as[19] _[.]_[5%] _/_ 64% _≈_ 30 _._ 5% programs from StableCode are rejected, indicating that smaller models like StableCode may not effectively self-test and detect program errors on its own. (b) In the event of failed self-tests, a large portion of failures are attributed to issues in self-tests instead of the programs, on both GPT-3.5 and StableCode. These phenomena indicate that self-testing methods have limitations in generating correct and reliable unit tests. As a result, we design functional consensus to not require any assertion, but perform _mutual verification_ between solutions instead, as opposed to self-testing. 

**Effectiveness of Functional Consensus** Functional consensus or self-testing may be viewed as ranking algorithms for selecting functions. To measure ranking effectiveness, we conduct an analysis on HumanEval with GPT-3.5. For each problem, 11 candidates are ranked with 3 strategies: consensus, self-test, and random shuffle (as a baseline). Effectiveness is measured via Pass@k, i.e. if any of the top-k ranked programs pass the system test. Figure 3.b shows that functional consensus achieves 94 _._ 7% upper bound (Pass@11) performance by selecting _a single_ function (Pass@1), and is close to that of self-test on Pass@4. This clearly demonstrates that functional consensus can effectively evaluate correctness and pick the most promising implementation on the first attempt. 

**Ablation and Token Usage** To analyze the impact of dividing, conquering, and functional consensus in FUNCODER, we carry out an ablation study with different settings. Studies that replace consensus with self-testing, or with AlphaCode-like (Li et al., 2022) clustering, are also included. The ablation is constructed on HumanEval with GPT-3.5, as shown in Table 2. Note that to generate every program FUNCODER costs only _O_ ( _kN_ ) tokens, where _k_ is the number of sampled candidates, 

6 

Table 2: Ablation study of FUNCODER on HumanEval with GPT-3.5. The setting in our main experiment is highlighted in bold. Tokens are calculated as the sum of prompts and completions. 

|**Setting**|Divide|Conquer|Ranking|**Pass@1**|**Avg. Tokens**|
|---|---|---|---|---|---|
|Standard|✗|✗|✗|68.3|**886.7**|
|One-pass|✓|✗|✗|72.6(+4.3)|1233.7|
|Two-pass|✓|✓|✗|78.7(+10.4)|3343.2|
|Two-pass + ST@11|✓|✓|Self-Test@11|80.5(+12.2)|5408.3|
|Two-pass + CL@11|✓|✓|Clustering@11|75.0(+6.7)|5070.7|
|FUNCODER@5|✓|✓|Consensus@5|83.5(+15.2)|4040.9|
|**FUNCODER@11**|✓|✓|Consensus@11|**85.4 (+17.1)**|5402.0|



and _N_ is the token length of the final program. This is further exemplified and explained in §A.7. We observe that function decomposition and re-composition deliver cumulative performance improvements. Functional consistency is also shown to prevail over self-testing. Putting them all together, FUNCODER received a +17 _._ 1 improvement with just 5 _._ 09 _×_ more tokens over baseline. Compared to previous SOTA LDB ( _≈_ 23K tokens), we are able to gain +2 _._ 5 in performance with 76 _._ 5% token usage reduction. 

## **3.2 Mathematical Reasoning** 

Code can be viewed as a tool for augmenting the reasoning capabilities of LLMs (Chen et al., 2023b). Alternative to text-based reasoning like Chain-of-Thought (Wei et al., 2022), programs can offer unique advantages in terms of iteration and calculations. To test the generalizability of FUNCODER beyond algorithm challenges, we conduct an experiment on MATH (Hendrycks et al., 2021b), a competition-level mathematical reasoning benchmark. 

## **3.2.1 Experiment Setup** 

**Benchmark** The experiment is conducted on a subset of the MATH test set, including 500 randomly sampled problems that can be classified into 7 disjoint subjects or 5 difficulty levels. It can be noticed that labels in MATH are formatted in L[A] TEX, rendering exact-match verdicts impractical. We, therefore, follow previous work (Zhang et al., 2024) and adopt GPT-4 to determine the correspondence between predictions and labels, with further details provided in Appendix A.4. 

**Baselines** We compare FUNCODER with the text-based baselines: Standard Prompting and Chainof-Thought (Wei et al., 2022), and program-aided baselines: Program-of-Thought (Chen et al., 2023b), Self-Refine (Madaan et al., 2023), Cumulative Reasoning (Zhang et al., 2024). The results of Cumulative reasoning are reported in the original paper. Standard prompting and chain-of-thought reasoning use 7-shot demonstrations constructed from the train set. Program-of-Thought and SelfRefine prompt the model with 1-shot demonstration to generate a `solution()` function that solves the problem. Additionally, self-refine iteratively refines programs based on runtime feedback. All baseline methods are run with self-consistency (Wang et al., 2023) at 5. 

**Implementation Details** FUNCODER adopts a program-aided reasoning setting that writes a `solution()` function and obtains the final prediction by running this program. The number of sampled implementations _|F |_ in functional consensus is set to 5 to match baseline methods. 

## **3.2.2 Results** 

The experimental results on MATH are shown in Table 3. It shows that program-aided reasoning generally outperforms text-based reasoning. With GPT-4 as the backbone, FUNCODER outperforms the strongest baseline Cumulative Reasoning (Zhang et al., 2024) by (6.0 / 8.3%) and surpasses the vanilla program-aided baseline PoT (Chen et al., 2023b) by (10.0 / 14.7%). When using GPT-3.5turbo as the backbone, FUNCODER exceeds the strongest baseline by (6.2 / 11.1%) and outperforms PoT by as much as (13.0 / 31.7%), which indicates that our approach has a strong advantage over both text-based reasoning and other program-aided reasoning methods. 

7 

Table 3: Experimental results on MATH, a competition-level mathematical reasoning benchmark. Best results are in bold. Text-based reasoning methods are denoted with _[†]_ , while others use programaided reasoning. We report both overall results and results in seven subjects: _Prealgebra_ , _Algebra_ , _Number Theory_ , _Counting & Probability_ , _Geometry_ , _Intermediate Algebra_ , and _Precalculus_ . 

|**Model**|**Method**|_Prealg._|_Alg._|_NT_|_Prob._|_Geo._|_InterAlg._|_Precalc._|**Overall**|
|---|---|---|---|---|---|---|---|---|---|
||Standard_†_|62.2|37.4|20.0|29.8|31.0|24.4|21.8|34.6|
||CoT_†_|59.8|51.1|28.9|29.8|28.6|26.7|30.9|40.0|
|GPT-3.5|PoT|68.3|50.4|33.3|48.9|21.4|18.2|29.1|41.0|
||Self-Refne|74.4|49.6|48.9|57.4|28.6|35.6|36.4|48.6|
||FUNCODER|**76.8**|**61.2**|**55.6**|**59.6**|**34.1**|**36.0**|**41.8**|**54.0**|
||Standard_†_|81.7|82.7|71.1|72.3|**59.5**|46.7|47.3|68.2|
||CoT_†_|84.1|87.1|62.2|68.1|45.2|48.9|54.5|68.6|
|GPT-4|PoT<br>Self-Refne|79.3<br>82.9|80.6<br>82.0|75.6<br>77.8|72.3<br>76.6|50.0<br>54.8|47.8<br>55.6|58.2<br>**63.6**|68.2<br>72.2|
||CR|86.6|86.3|**88.7**|71.1|53.7|51.5|51.8|72.2|
||FUNCODER|**89.0**|**92.8**|82.2|**83.0**|**59.5**|**63.3**|56.4|**78.2**|
||CoT_†_|56.1|**47.5**|31.1|34.0|**40.5**|14.4|**38.2**|38.6|
|Llama38_b_|PoT|67.1|32.4|24.4|34.0|16.7|21.1|18.2|32.6|
||FUNCODER|**67.9**|45.7|**51.1**|**53.2**|19.0|**37.8**|30.9|**45.0**|
|StableCode3_b_|PoT<br>FUNCODER|20.7<br>**46.3**|14.4<br>**30.2**|17.8<br>**20.0**|25.5<br>**29.8**|**4.8**<br>**4.8**|8.9<br>**20.0**|9.1<br>**18.2**|14.4<br>**26.6**|
|CodeLlama34_b_|PoT<br>FUNCODER|35.5<br>**44.8**|26.1<br>**46.1**|15.0<br>**37.8**|16.7<br>**34.1**|0.0<br>**13.6**|5.5<br>**24.6**|33.3<br>**37.5**|15.2<br>**24.4**|



On open-source models, FUNCODER with Llama3 outperforms PoT by (12.4 / 38.0%). It has even reached competitive performance against the state-of-the-art method based on GPT-3.5 (45.0 v.s. 48.6). When employing StableCode and CodeLLaMA as the backbone, our approach achieves significant improvements by (12.2 / 84.7%) and (9.2 / 60.5%), respectively. This improvement demonstrates that our approach can significantly boost smaller LLMs, democratizing the complex reasoning capabilities of open-source LLMs through programming. 

## **3.2.3 Analysis** 

**FUNCODER Can Handle Harder Questions** Figure 4 compares between CoT, PoT, and FUNCODER across varying difficulty levels. It illustrates that CoT performs comparatively well on the easiest questions, but suffers from a steep decline in performance as difficulty increases. This suggests that text-based reasoning is inadequate for tackling challenging mathematical reasoning problems. The same situation is also observed in PoT. In contrast, our method consistently demonstrates high performance even on challenging problems, particularly excelling on level 5 difficulty with nearly double the perfor- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation_images/nips_2024_Divide-and-Conquer_Meets_Consensus_Unleashing_the_Power_of_Functions_in_Code_Generation.pdf-0008-05.png)


**----- Start of picture text -----**<br>
GPT-3.5 StableCode 3 b<br>80 Ours 50<br>PoT<br>60 CoT 40<br>30<br>40<br>20<br>20 10<br>0 0<br>Level 1 Level 2 Level 3 Level 4 Level 5 Level 1Level 2Level 3Level 4Level 5<br>**----- End of picture text -----**<br>


Figure 4: Average accuracy in each level with the chat model (GPT-3.5) and the code model (StableCode3 _b_ ) on the MATH benchmark. 

mance compared to PoT and CoT. This reflects that our method, with divide-and-conquer applied, can effectively cope with complex problems. 

**Decomposed Functions are Domain-Specific** We hypothesize that questions from the same subject require similar knowledge reserves, which should be reflected in the functionality of the sub-functions. To verify this hypothesis, we statisticize the common sub-functions of FUNCODER in each MATH subject, as shown in Table 4. It is apparent that different subjects require different abilities, each with its own set of sub-functions closely associated with the domain knowledge. In addition, these common sub-functions are fundamentally basic and straightforward. As exemplified in Appendix B.2, our method is able to leverage and combine these basic sub-functions to achieve more complex goals, thereby reducing the complexity of reasoning and enhancing performance. 

8 

Table 4: Top-3 most commonly used functions in each subject of MATH, listed in descending order. 

|**Subject**|**Functions**|
|---|---|
|Prealgebra|is_prime / factorial / gcd|
|Algebra|fnd_roots / is_perfect_square / fnd_domain|
|Number Theory|get_divisors / mod_inverse / gcd|
|Counting & Probability|factorial / combinations / binomial_coeffcient|
|Geometry|distance / simplify_fraction / calculate_triangle_area|
|Intermediate Algebra|fnd_roots / evaluate_polynomial / lagrange_interpolation|
|Precalculus|cross_product / fraction_from_angle / dot|



## **4 Related Work** 

**Large Language Model for Code** Code pre-training has received widespread attention, with early models based on small language models (SLM) (Feng et al., 2020; Lu et al., 2021; Wang et al., 2021). In recent years, with the development of large-scale pre-training techniques, code LLM has emerged, showing remarkable performance in downstream code tasks (Chen et al., 2021; Nijkamp et al., 2023; Li et al., 2022; Rozière et al., 2023; Li et al., 2023b; Guo et al., 2024). Tasks between code and natural language (NL) can be generally divided into three major categories: NL2Code tasks such as code generation (Austin et al., 2021; Chen et al., 2021; Hendrycks et al., 2021a; Khan et al., 2023) and code search (Husain et al., 2019); Code2Code tasks including code completion (Lu et al., 2021; Zhang et al., 2023a; Liu et al., 2024), code translation (Ahmad et al., 2023; Zhu et al., 2022; Yan et al., 2023), and test generation (Siddiq et al., 2023; Schäfer et al., 2024); Code2NL tasks like code summarization (Husain et al., 2019; Jin et al., 2023). This paper focuses on code generation tasks, ranging from basic to competition level. 

**Code Refinement and Self-Testing** Code doesn’t always run as expected; it could contain syntax errors, dead loops, or bugs. It’s essential to debug and refine the code to ensure better quality. CodeT (Chen et al., 2023a) generates unit-tests to score the implementation. AlphaCode (Li et al., 2022) clusters programs based on whether generated program outputs were identical or not. Selfimprovement methods (Madaan et al., 2023; Shinn et al., 2023; Chen et al., 2024; Zhong et al., 2024) design closed-loop procedures that repeatedly refine the code based on the feedback. Like real-life software development processes, multi-agent frameworks (Hong et al., 2024; Qian et al., 2023) construct specific LLM roles, _Tester_ or _QA_ to generate tests. These studies adopt a shared paradigm wherein self-tests are generated through LLMs. However, Olausson et al. (2024) points out the challenge that LLMs have certain shortcomings in self-repairing their code. This paper avoids these shortcomings by proposing _functional consensus_ as a reliable method of evaluation. 

**Program-Aided Reasoning and Agents** Aside from code generation tasks, the program can be a tool that augments LLM to solve complex reasoning questions or interact with external environments. Program-of-Thought (Chen et al., 2023b) and PAL (Gao et al., 2023) prompt the model to generate a program that solves mathematical or symbolic problems. MathPrompter (Imani et al., 2023) and Chain-of-Code (Li et al., 2023a) fuse the text-based chain-of-thought with code-based program-of-thought prompting to complement each other in mathematical reasoning. Cumulative Reasoning (Zhang et al., 2024) conducts bottom-up reasoning to derive the final answer progressively. Numerous work (Sun et al., 2023; Wang et al., 2024; Yang et al., 2024) also use code as an intermediate component to bridge LLM agents with external environments. 

**Decompose for Complex Problems** Several recent works employ decomposition to reduce the complexity of hard problems. Least-to-Most (Zhou et al., 2023) adopts a two-stage approach, which first decomposes complex problems, and then solves each sub-problem individually to tackle complex reasoning tasks. Successive Prompting (Dua et al., 2022) adopts a dynamic decomposition, iteratively breaking down problems and addressing sub-problems. Tree-of-Thought (Yao et al., 2023) breaks down complex problems into state spaces and uses tree search to solve them. Parsel (Zelikman et al., 2023) introduces decomposition to code generation tasks, taking a three-stage to break down requirements into draft and intermediate parsel programs. RepoCoder (Zhang et al., 2023b) performs a retrieval in repositories to complete unfinished code one by one. Unlike these methods, FUNCODER recursively decomposes problems into a tree structure, hence gradually reduces its complexity. 

9 

## **5 Discussion** 

**Limitations** Our approach unleashes the potential power of functions in programming, which is advantageous on well-defined problems such as competitive programming, or program-augmented reasoning tasks. These scenarios do not however represent all use cases, such as open-ended problems or casual software development. Nevertheless, we believe that the idea of divide-and-conquer and sub-modular consensus utilized by FUNCODER can be extended to a wider range of problems, and we consider this as a future exploration. 

**Broader Impact** While code generation is increasingly utilized in software development, Large Language Models (LLMs) are still prone to generating toxic, vulnerable, or malicious code. Such programs pose risks and should be used or executed with extra caution. 

## **6 Conclusion** 

In this paper, we presented FUNCODER, a novel code generation framework that integrates the divideand-conquer strategy with functional consensus to address complex requirements. FUNCODER had demonstrated superior performance compared to state-of-the-art methods on various benchmarks and models. Our findings highlighted the effectiveness of dynamic decomposition and functional consensus in writing complex code, which suggests that FUNCODER may have the potential to empower further improvements in code generation and other fields. 

## **Acknowledgments** 

We would like to acknowledge the reviewers and chairs for their inspiring and constructive feedback. The research in this article is supported by the National Key Research and Development Project (2021YFF0901602), the National Science Foundation of China (U22B2059, 62276083). Bing Qin and Qianglong Chen are the corresponding authors. 

## **References** 

- Wasi Uddin Ahmad, Md Golam Rahman Tushar, Saikat Chakraborty, and Kai-Wei Chang. AVATAR: A parallel corpus for Java-python program translation. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 2268–2281, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.143. URL `https://aclanthology.org/2023.findings-acl.143` . 

- Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. _ArXiv preprint_ , abs/2108.07732, 2021. URL `https://arxiv.org/abs/2108.07732` . 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. URL `https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html` . 

- Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q. Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. Multipl-e: A scalable and polyglot approach to benchmarking neural code generation. _IEEE Trans. Software Eng._ , 49(7):3675–3691, 2023. doi: 10.1109/TSE.2023.3267446. URL `https: //doi.org/10.1109/TSE.2023.3267446` . 

- Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023a. URL `https://openreview.net/ forum?id=ktrw68Cmu9c` . 

10 

- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel HerbertVoss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. _ArXiv preprint_ , abs/2107.03374, 2021. URL `https: //arxiv.org/abs/2107.03374` . 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _Transactions on Machine Learning Research_ , 2023b. ISSN 2835-8856. URL `https://openreview.net/forum?id=YfZ4ZPt8zd` . 

- Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In _The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024_ . OpenReview.net, 2024. URL `https://openreview.net/forum?id=KuPixIqPiq` . 

- Ole-Johan Dahl, Edsger W. Dijkstra, and Charles Antony Richard Hoare. _Structured programming_ , volume 8 of _A.P.I.C. Studies in data processing_ . Academic Press, 1972. ISBN 978-0-12-200550-3. 

- Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. Successive prompting for decomposing complex questions. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 1251–1265, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.81. URL `https://aclanthology.org/2022.emnlp-main.81` . 

- EbTech. How to Interpret Contest Ratings - Codeforces, 2024. URL `https://codeforces.com/blog/ entry/68288` . 

- Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. CodeBERT: A pre-trained model for programming and natural languages. In Trevor Cohn, Yulan He, and Yang Liu (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pp. 1536–1547, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.findings-emnlp.139. URL `https://aclanthology.org/2020.findings-emnlp.139` . 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: program-aided language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 10764–10799. PMLR, 2023. URL `https://proceedings.mlr.press/v202/gao23f.html` . 

- Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming - the rise of code intelligence. _ArXiv preprint_ , abs/2401.14196, 2024. URL `https: //arxiv.org/abs/2401.14196` . 

- P.R. Halmos. _Naive Set Theory_ . Undergraduate Texts in Mathematics. Springer New York, 1998. ISBN 9780387900926. URL `https://books.google.com.hk/books?id=x6cZBQ9qtgoC` . 

- Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with APPS. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021a. URL `https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html` . 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021b. URL `https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html` . 

11 

- Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In _The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024_ . OpenReview.net, 2024. URL `https://openreview.net/forum?id=VtmBAGCN7o` . 

- Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. _ArXiv preprint_ , abs/1909.09436, 2019. URL `https://arxiv.org/abs/1909.09436` . 

- Shima Imani, Liang Du, and Harsh Shrivastava. MathPrompter: Mathematical reasoning using large language models. In Sunayana Sitaram, Beata Beigman Klebanov, and Jason D Williams (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track)_ , pp. 37–42, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-industry.4. URL `https://aclanthology.org/2023.acl-industry.4` . 

- Xue Jiang, Yihong Dong, Lecheng Wang, Qiwei Shang, and Ge Li. Self-planning code generation with large language model. _ArXiv preprint_ , abs/2303.06689, 2023. URL `https://arxiv.org/abs/2303.06689` . 

- Xin Jin, Jonathan Larson, Weiwei Yang, and Zhiqiang Lin. Binary code summarization: Benchmarking chatgpt/gpt-4 and other large language models. _ArXiv preprint_ , abs/2312.09601, 2023. URL `https: //arxiv.org/abs/2312.09601` . 

- Mohammad Abdullah Matin Khan, M. Saiful Bari, Xuan Long Do, Weishi Wang, Md. Rizwan Parvez, and Shafiq R. Joty. xcodeeval: A large scale multilingual multitask benchmark for code understanding, generation, translation and retrieval. _ArXiv preprint_ , abs/2303.03004, 2023. URL `https://arxiv.org/abs/2303. 03004` . 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_ , 2023. 

- Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, and Brian Ichter. Chain of code: Reasoning with a language model-augmented code emulator. _ArXiv preprint_ , abs/2312.04474, 2023a. URL `https://arxiv.org/abs/2312.04474` . 

- Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Moustafa-Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. Starcoder: may the source be with you! _ArXiv preprint_ , abs/2305.06161, 2023b. URL `https://arxiv.org/abs/2305.06161` . 

- Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. _ArXiv preprint_ , abs/2203.07814, 2022. URL `https://arxiv.org/abs/2203.07814` . 

- Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code autocompletion systems. In _The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024_ . OpenReview.net, 2024. URL `https://openreview.net/forum?id= pPjZIOuQuF` . 

- Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex 

12 

Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian J. McAuley, Han Hu, Torsten Scholak, Sébastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, and et al. Starcoder 2 and the stack v2: The next generation. _ArXiv preprint_ , abs/2402.19173, 2024. URL `https://arxiv.org/abs/2402.19173` . 

- Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin B. Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. Codexglue: A machine learning benchmark dataset for code understanding and generation. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual_ , 2021. URL `https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ c16a5320fa475530d9583c34fd356ef5-Abstract-round1.html` . 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with selffeedback. In _Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023_ , 2023. URL `https://openreview.net/forum?id=S37hOerQLB` . 

Meta AI. Introducing Meta Llama 3: The most capable openly available llm to date, 2024. URL `https: //ai.meta.com/blog/meta-llama-3/` . 

Mistral AI. Codestral: Hello, world!, 2024. URL `https://mistral.ai/news/codestral/` . 

- Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL `https://openreview.net/forum?id=iaYcJKpY2B_` . 

- Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and Armando Solar-Lezama. Is self-repair a silver bullet for code generation? In _The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna Austria, May 7-11, 2024_ . OpenReview.net, 2024. URL `https:// openreview.net/forum?id=y0GJXRungR` . 

- OpenAI. GPT-4 technical report. _ArXiv preprint_ , abs/2303.08774, 2023. URL `https://arxiv.org/abs/ 2303.08774` . 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In _NeurIPS_ , 2022. URL `http://papers.nips.cc/paper_files/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html` . 

- Nikhil Pinnaparaju, Reshinth Adithyan, Duy Phung, Jonathan Tow, James Baicoianu, Ashish Datta, Maksym Zhuravinskyi, Dakota Mahan, Marco Bellagente, Carlos Riquelme, and Nathan Cooper. Stable code technical report. _ArXiv preprint_ , abs/2404.01226, 2024. URL `https://arxiv.org/abs/2404.01226` . 

- Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. _ArXiv preprint_ , abs/2307.07924, 2023. URL `https://arxiv.org/abs/2307.07924` . 

- Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code. _ArXiv preprint_ , abs/2308.12950, 2023. URL `https://arxiv.org/abs/2308. 12950` . 

- Max Schäfer, Sarah Nadi, Aryaz Eghbali, and Frank Tip. An empirical evaluation of using large language models for automated unit test generation. _IEEE Trans. Software Eng._ , 50(1):85–105, 2024. doi: 10.1109/ TSE.2023.3334955. URL `https://doi.org/10.1109/TSE.2023.3334955` . 

- Kensen Shi, Jacob Steinhardt, and Percy Liang. Frangel: component-based synthesis with control structures. _Proc. ACM Program. Lang._ , 3(POPL), 2019. doi: 10.1145/3290386. URL `https://doi.org/10.1145/3290386` . 

13 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In _Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023_ , 2023. URL `https://openreview.net/forum?id=vAElhFcKW6` . 

- Mohammed Latif Siddiq, Joanna C. S. Santos, Ridwanul Hasan Tanvir, Noshin Ulfat, Fahmid Al Rifat, and Vinicius Carvalho Lopes. Exploring the effectiveness of large language models in generating unit tests. _ArXiv preprint_ , abs/2305.00418, 2023. URL `https://arxiv.org/abs/2305.00418` . 

- Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. Adaplanner: Adaptive planning from feedback with language models. In _Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023_ , 2023. URL `https://openreview.net/forum?id=rnKgbKmelt` . 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. _ArXiv preprint_ , abs/2307.09288, 2023. URL `https://arxiv.org/abs/2307.09288` . 

- Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better LLM agents. In _ICLR 2024 Workshop on Large Language Model (LLM) Agents_ , 2024. URL `https://openreview.net/forum?id=8oJyuXfrPv` . 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL `https://openreview.net/forum?id=1PL1NIMMrw` . 

- Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 8696–8708, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.685. URL `https: //aclanthology.org/2021.emnlp-main.685` . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. URL `http://papers.nips.cc/paper_ files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html` . 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. _ArXiv preprint_ , abs/1910.03771, 2019. URL `https://arxiv.org/abs/1910. 03771` . 

- Weixiang Yan, Yuchen Tian, Yunzhe Li, Qian Chen, and Wen Wang. CodeTransOcean: A comprehensive multilingual benchmark for code translation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 5067–5089, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.337. URL `https://aclanthology. org/2023.findings-emnlp.337` . 

- Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Heng Ji, and ChengXiang Zhai. If LLM is the wizard, then code is the wand: A survey on how code empowers large language models to serve as intelligent agents. In _ICLR 2024 Workshop on Large Language Model (LLM) Agents_ , 2024. URL `https://openreview.net/forum?id=8dmNOD9hbq` . 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In _Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023_ , 2023. URL `https://openreview.net/forum? id=5Xc1ecxO1h` . 

14 

- Eric Zelikman, Qian Huang, Gabriel Poesia, Noah Goodman, and Nick Haber. Parsel: Algorithmic reasoning with language models by composing decompositions. In _Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS 2023_ , 2023. URL `https://openreview.net/forum?id=qd9qcbVAwQ` . 

- Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 2471–2484, Singapore, 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.151. URL `https://aclanthology.org/2023.emnlp-main.151` . 

- Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 2471–2484, Singapore, 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.151. URL `https://aclanthology.org/2023.emnlp-main.151` . 

- Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. In _ICLR 2024 Workshop on Bridging the Gap Between Practice and Theory in Deep Learning_ , 2024. URL `https://openreview.net/forum?id=XAAYyRxTlQ` . 

- Lily Zhong, Zilong Wang, and Jingbo Shang. LDB: A large language model debugger via verifying runtime execution step-by-step. _ArXiv preprint_ , abs/2402.16906, 2024. URL `https://arxiv.org/abs/2402. 16906` . 

- Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL `https://openreview.net/forum? id=WZH7099tgfM` . 

- Ming Zhu, Karthik Suresh, and Chandan K. Reddy. Multilingual code snippets training for program translation. In _Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022_ , pp. 11783–11790. AAAI Press, 2022. URL `https://ojs.aaai.org/index.php/AAAI/article/view/21434` . 

15 

## **A Appendix** 

In the supplementary materials, we provide the details of implementation (A.1), baseline information and settings (A.2), benchmarks (A.3), metrics (A.4), settings in the analysis (A.5), and additional experiments (A.9). We also demonstrate the example solutions of our method and baseline in Appendix B, and include all the prompts in Appendix C. 

Table 5: Symbols and Glossary. 

||||**Alias**|**Description**|
|---|---|---|---|---|
|_(i) Symbols_|||||
|_f_(_x_)|||Function|In the programming language, a function consists of|
|||||header, documentation, and its body_{hf, df, bf}_. A func-|
|||||tion can also be viewed as a mapping_f_ :_D_(_f_)_→Y_.|
|_hf_|||Function Header|Declares the function name, arguments, and return type,|
|||||and is used as a signature to identify the function in a|
|||||program.|
||_df_||Function Docstring|Provides additional usage details for this function, but is|
||||(or Documentation)|optional. We encourage the model to generate docstrings|
|||||to describe sub-goals precisely.|
||_bf_||Function Body|The function body contains a subroutine that describes its|
||||(or Implementation)|control fow and behavior. Functions may be invoked from|
|||||within.|
|_f _|_′_(_x_)||Partially Implemented|A provisional function structure generated by the LLM|
|||||where sub-procedures are not yet implemented.|
|_f ∗_(_x_)|||Solved Function|A fnal implementation that is no longer changed and|
|||||represents FUNCODER’s fnal comprehension and solution|
|||||on the original problem.|
|_F_ =|�<br>_f_(_i_)|�|Sampled Implementation|Functions that re-implement_f ′_(_x_)based on solved sub-|
|||||functions, generated by models using the same input|
|||||prompt.|
|CHILD(_f_(_x_))|||Dependency|Functions that are used in_f_(_x_). (exclude_f_(_x_)itself)|
||_T_||Dependency Tree|Defned byTREE(_f,_CHILD(_f_)), where_f_ is the root node|
|||||of the current sub-task. Circular references are ignored.|
||_F_||Function Composition|To implement a certain function _f_ respecting sub-|
|||||procedures as potentially reusable components.|
|_(ii) Glossary_|||||
|System Test|||Hidden Test|System testing is a phase where a set of previously invis-|
|||||ible test cases are run against the submitted program to|
|||||validate if the code is correct and produces the expected|
|||||output for different categories of inputs.|
|Unit Test|||Assertion|A unit test is an assertion consisting of given input and|
|||||expected output, whereas in Python, it takes the form of|
|||||`assert func(x) == y`.|
|Self-testing|||-|Self-testing is an evaluation process that prompts the|
|||||model to generate unit tests (assertions) to assess the cor-|
|||||rectness of the generated program.|
|AlphaCode-like|||-|AlphaCode proposed a clustering process that elects can-|
|Clustering||||didate program from a number of samples, recognizing|
|||||programs that produce exactly identical outputs as equiva-|
|||||lent, and picks one program from the largest cluster.|



## **A.1 Implementation Details** 

**Models** We access the OpenAI models GPT-3.5 (gpt-3.5-turbo-0613), GPT-4 (gpt-4-1106-preview) and GPT-4o mini (gpt-4o-mini-2024-07-18) through Azure OpenAI. Weights of community models Llama3 (Meta-Llama-3-8B-Instruct), Codestral (Codestral-22B-v0.1), StableCode (stable-codeinstruct-3b), CodeLlama (CodeLlama-34b-Instruct-hf) and StarCoder2 (starcoder2-15b-instruct-v0.1) 

16 

are downloaded from HuggingFace (Wolf et al., 2019) and served over an OpenAI-like API on a single A100-80G GPU under BF16 precision with vLLM (Kwon et al., 2023). 

**Divide** We instruct the model to write the current function and introduce new functions with clearly defined sub-goals. The prompt C.2 for the _divide_ process includes two examples: one example needs to involve new functions that are left unimplemented; and another where the sub-goal is simple enough that no further decomposition is necessary. The model generates a Python code block with a temperature of 0.2, and the code block will be extracted to represent a tree of functions with new functions as the children of the current. We require that any new sub-function do not refer to existing functions, to avoid circular references. This generation process will be attempted at most 3 times until any valid code with a proper signature is found in the output. FUNCODER then traverses the function tree via depth-first search and restricts the max depth of the tree to 6. 

**Conquer** We apply the composition of sub-functions to rewrite the parent function after all subfunctions have been fully decomposed. Code for sub-functions is made visible to the LLM, which is requested to rewrite the current function with a 1-shot demonstration (C.3). With functional consensus applied, the model samples multiple implementations with a temperature of 0.8, and the one that reaches consensus will be kept for further bottom-up processing. 

**Functional Consensus** The functional consensus is applied in the conquer stage. Formally, Consensus@k samples k-1 implementations in the conquer stage, and reuses the one produced in the divide stage, resulting in a set _F_ of k candidate programs. Then we prompt the model with 1-shot (C.4) to generate potential inputs _X_ for the given function and use them to feed and execute the program. As described in Eq 2, when two functions output the same value in a given input, they will both add 1 point to the overall similarity. A thrown exception or timeout during execution assigns -100 points to the candidate as it indicates potentially problematic code. Similar to self-testing methods, we also leverage the example input/output at the root node to filter out candidates that have wrong functionality. Finally, the one candidate with maximum scores over all inputs is selected, as it reaches consensus with other implementations. 

**Hierarchical Code Interpreter** Divide-and-conquer represents the problem hierarchy through structured code. To gain insights of this information, we design an interpreter that syntactically parses the generated output and organizes them into a graph of functions. We are thus able to decompose complex tasks by representing sub-goals through the connections of multiple functions. LLMs may produce vulnerable code even if prompted by trusted inputs, making direct execution or `eval()` on generated code especially hazardous. Our framework addresses this with the use of a sandboxed environment to contain untrusted code execution, preventing the LLM from hanging up or even breaking the system. 

## **A.2 Baseline Details** 

**Standard Prompting** conducts one-time generation and directly output the entire code or final results. In code generation tasks, we use a 1-shot prompting setting with 0.3 temperature. For MATH, we sample 1 question-answer pair per subject in the train set, resulting in a 7-shot prompt, and run self-consistency (Wang et al., 2023) with consistency@5 and temperature 0.7. 

**CodeT** (Chen et al., 2023a) samples multiple code solutions _X_ and unit-tests _Y_ . A unit test is an assertion consisting of given input and expected output, whereas in Python it takes the form of " `assert func(x) == y` ", CodeT then checks the programs over self-tests and divides the functions into sets; the score of such a set is defined as the number of functions within multiplied by the number of succeeded tests. Finally, CodeT selects the function with the most agreement (in the biggest set). Similar to the setting of FUNCODER, we sample 11 candidate solutions with 0.8 temperature. 

**AlphaCode-like Clustering** is introduced with the model AlphaCode (Li et al., 2022), and samples multiple code solutions _X_ . A fine-tuned model is used to generate test inputs upon which programs are evaluated. Programs are then clustered by whether the outputs are (exactly) identical, in which the largest group was selected. The final result comes from this largest group, where any result within would suffice. In our ablation study, we use similar settings to FUNCODER, sampling 11 candidate solutions with 0.8 temperature and generate sample inputs likewise. 

17 

**Parsel** (Zelikman et al., 2023) consists of three generation stages: high-level sketch, Parsel program, and final program. The Parsel program is an intermediate representation of code that describes and organizes program structure. We report the result of HumanEval with GPT-4 from the original paper. 

**Reflexion** (Shinn et al., 2023) is a closed-loop agent system that generates unit tests and iteratively refines the program based on the self-test feedback. The results for GPT-4 on HumanEval and MBPP are reported in the original paper. Based on officially released code[3] , we test results with GPT-3.5 and community models under the reflexion strategy with `max_iters=2` and Pass@1. For the xCodeEval benchmark, as it is judged through standard input/output, we wrap the standard input into function arguments and obtain the return value as the output in the form of " `def main(input_str: str) -> str` ", and the sample input/output are also transformed to visible tests for reflexion process. 

**MetaGPT** (Hong et al., 2024) employs a multi-agent strategy that assigns roles and encodes humanlike software development procedures. The scripts for reproducing the results were not made public as of this paper was completed. Therefore, we include the original result for GPT-4 on the HumanEval dataset under the with feedback setting. 

**LDB** (Zhong et al., 2024) segments programs into basic blocks and tracks the values of intermediate variables after each block throughout runtime execution, allowing large language models to verify the correctness of smaller code units. We adopt the results as-is reported in the paper. 

**Chain-of-Thought Prompting** (Wei et al., 2022) generates step-by-step reasoning leading to the final output answer. The solution is formatted in L[A] TEX, and use \boxed to mark the final answer. We sample 1 shot per subject in the MATH train set, resulting in a 7-shot demonstration, and running with consistency@5 and a temperature of 0.7. 

**Program-of-Thought** (Chen et al., 2023b) utilizes the coding ability in LLMs to generate programs rather than text-based solutions for reasoning tasks. In MATH, we hint the model with 1-shot prompting to generate a `solution()` function that returns the final answer to the problem. The program is then executed in a Python environment and obtains the return value. If an exception is thrown during execution, the model will try to regenerate a new program until it succeeds or reaches 3 attempts. Similar to CoT, Program-of-Thought samples 5 programs at a temperature of 0.7 and votes the final result. 

**Self-Refine** (Madaan et al., 2023) iteratively prompts the model to give feedback and refine the generated code based on it. Self-refine does not incorporate self-tests, and the refinement is conducted solely on model feedback. In our preliminary study on HumanEval, this feedback is weak and cannot improve performance. However, in MATH, the solution program can be executed without the need for generated assertions. Thus, we extend the self-refine to capture the runtime error trace as feedback and refine the code until it can run or exceed 3 retries. 

**Cumulative Reasoning** (Zhang et al., 2024) starts from decomposing the input problem into propositions and conducts bottom-up reasoning until the final answer can be concluded. The results for Cumulative Reasoning are reported in the original paper under with code setting. 

## **A.3 Benchmark Details** 

Table 6: Overview and details of HumanEval, MBPP, xCodeEval, and MATH dataset. 

||**HumanEval**|**MBPP**|**xCodeEval**|**MATH**|
|---|---|---|---|---|
|Task|Code Generation|Code Generation|Programming Contest|Mathematical Reasoning|
|Attribute|-|-|tags, diffculty|subject, level|
|Metric|Pass@1|Pass@1|Pass@1|EM-gpt|
|# Sample (original)|164|427|7,635|5,000|
|# Sample (ours)|164|200|500|500|
|Entry func|variant|variant|main()|solution()|
|Input|arguments|arguments|standard input|n/a|
|Output|return|return|standard output|return|
|# Examples Tests|~2.8|0|~2.1|n/a|
|# System Tests|~8.1|~3.1|51.1|n/a|



> 3GitHub: noahshinn/reflexion 

18 

**HumanEval** (Chen et al., 2021) is a hand-crafted programming dataset designed to evaluate a model’s code generation capability. It consists of 164 instances involving programming skills in language comprehension, reasoning, algorithms, and simple mathematics. The problem contains 2.8 sample inputs and outputs on average in the function document, which can be leveraged to provide additional guidance for the LLM to select or self-improve the programs. We conduct experiments on all 164 instances using accuracy (Pass@1) as the evaluation metric. The details of the Pass@1 metric are described in Appendix A.4. 

**MBPP** (Austin et al., 2021) consists of fundamental Python programming problems, with a total of 974 examples covering Python programming basics, standard library usage, and related assessment. Following Shinn et al. (2023), we adopt the mbpp-typed split from MultiPL-E (Cassano et al., 2023) and sample 200 instances, using Pass@1 as the metric. The original prompt[4] from MBPP includes all hidden tests in the input problem, which may cause label leakage when using these tests to refine or select programs. To ensure a fair comparison, MultiPL-E removes the test information in the prompt. 

**xCodeEval** (Khan et al., 2023) is a competition-level multilingual and multitask benchmark consisting of 17 programming languages. xCodeEval collects 25 million openly available samples from codeforces.com, a platform for competitive programming. The data we use include problem descriptions in `problem_descriptions.jsonl` and system tests from `unittest_db.json` which consists of 7,635 competition problems and averaged 51.1 tests per problem. Note that the tests in xCodeEval are crawled, some of them are incomplete due to the website context limit (they end with an ellipsis and the further content is missing); we filter out problems having invalid test cases. Based on the CodeForces Rating (EbTech, 2024), we categorize the problems by their difficulty: Easy ( _≤_ 1200), Mid (1200-1599), Hard (1600-1999), and Expert ( _≥_ 2000). We sample 500 problems from the full split with the basic filter rule mentioned above, resulting in Table 7. The CodeForces problem has a different input/output style compared to HumanEval and MBPP; it scans input from Standard Input and prints the answer to Standard Output. Therefore, we judge the program on system tests using a CodeForces-style judger and use Pass@1 (the program must pass all system tests) as the evaluation metric. 

**MATH** (Hendrycks et al., 2021b) is a challenging competition-level mathematical reasoning dataset, with problems and solutions formatted in L[A] TEX. It covers seven categories: Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, and Precalculus. The original test set of MATH consists of 5000 samples, and we randomly sampled 500 problems as shown in Table 7. In addition to text-based reasoning, writing programs is another promising way to solve mathematical problems. These methods involve writing a `main()` or `solution()` function, and executing the program to obtain the final answer. Through experiments on MATH, we aim to demonstrate that FUNCODER can enhance LLM’s ability to address complex mathematical problems through programming. 

Table 7: Number of test samples in **(a)** xCodeEval difficulty, **(b)** MATH level, **(c)** MATH subject. 

|**Diffculty Ours Original**<br>Easy<br>178<br>1428<br>Mid<br>112<br>1319<br>Hard<br>87<br>1453<br>Expert<br>118<br>3289<br>n/a<br>5<br>146<br>Total<br>500<br>7635|**Level**<br>**Ours**<br>**Original**<br>Level 1<br>39<br>437<br>Level 2<br>90<br>894<br>Level 3<br>108<br>1131<br>Level 4<br>116<br>1214<br>Level 5<br>147<br>1324<br>Total<br>500<br>5000|**Subject**<br>**Ours Original**|
|---|---|---|
|||Prealgebra<br>82<br>871<br>Algebra<br>139<br>1187<br>Number Theory<br>45<br>540<br>Counting & Probability<br>47<br>474<br>Geometry<br>42<br>479<br>Intermediate Algebra<br>90<br>903<br>Precalculus<br>55<br>546|
|||Total<br>500<br>5000|



## **A.4 Metrics** 

**Pass@k** When a program is passed (or accepted), it means that the program must pass all system tests without errors and within the time limit. In our experiments, we set the time limit to 2.5 seconds. Pass@k judges k independent programs, and if any of them can pass, the result will be 1. 

> 4Original MBPP prompt: https://github.com/google-research/google-research/tree/master/mbpp 

19 

In most of our experiments, we use Pass@1 as the metric, as it reflects the accuracy of the method framework achieved without feedback from humans. Pass@k, on the other hand, is equivalent to filtering programs through hidden, human-annotated test labels. 

**EM-GPT** The ground truth label in MATH is written in L[A] TEX, and the accuracy between labels and model predictions cannot be directly calculated through exact-match (EM). MATH provides a judge program[5] that preprocesses LaTeX syntax and check whether two disambiguated strings are equal. However, this is insufficient for evaluating LaTeX-formatted labels with variant program outputs. We follow the evaluation criteria from previous work (Zhang et al., 2024), using GPT-4 to assess the consistency between predictions and ground truths, with prompt shown in C.6. 

## **A.5 Details of Analysis** 

**Details of Preliminary Analysis on Self-testing** (Figure 3.a) The preliminary study is conducted on the HumanEval dataset, which includes system tests _S_ to evaluate the accuracy of the program, as well as one human-annotated canonical solution _c_ . For each question, we: (1) Obtain one solution program _p_ from Standard Prompting. (2) Prompt the model to generate 7 self-tests _T_ based on the question and entry function. The self-test is in the form of the unit test `assert f(x) == y` . We then judge the generated program _p_ and canonical solution _c_ over the self-tests _T_ and system tests _S_ . Formally, a pair ( _x, Y_ ) is used to identify whether program _x_ passes test _Y_ . Where ( _p, S_ ) indicates that the program can pass the system tests, demonstrating its correctness. And _¬_ ( _c, T_ ) means the canonical solution can not pass self-tests, suggesting that the tests generated by model could be wrong. The self-test results on generated programs are first evaluated and divided into two classes: self-test passed or failed. If the self-test passes, the self-improvement methods will stop iteration and pick this program as a final result. The next step is to determine whether the program can pass system tests. If the self-test fails, it indicates that there could be an error in the program or test itself. In this case, the correctness of the program is checked using final tests ( _p, S_ ) and the correctness of the unit test by canonical program ( _c, T_ ). The results on GPT-3.5 and StableCode are shown in Figure 3 and detailed explanations about these conditions can be found in Table 8. 

Table 8: Explanation on how we classify cases in self-testing preliminary study. 

|**Class**|**Subclass**<br>**Condition**<br>**Explanation**|
|---|---|
|self-test<br>passed|fnal<br>passed<br>(_p, T_)_∧_(_p, S_)<br>The self-test result is consistent with the fnal judge. How-<br>ever, self-testing methods**cannot improve**performance<br>in this case, as the program from the baseline (Standard<br>Prompt) is already correct.<br>fnal<br>failed<br>(_p, T_)_∧¬_(_p, S_)<br>Self-test is**too weak**to detect errors in the program, there<br>could be edge cases that not been considered.|
|self-test<br>failed|program<br>wrong<br>_¬_(_p, T_)_∧¬_(_p, S_)_∧_(_c, T_)<br>This is a **good example** that self-testing detects errors<br>in the program. Feedback from the test will be used to<br>select or refne the solution.<br>unit-test<br>wrong<br>_¬_(_p, T_)_∧_(_p, S_)_∧¬_(_c, T_)<br>Bad case, the self-test produced an**error result**and fl-<br>tered out a correct solution. Continuously revising the<br>code for this test will lead to a performance downgrade.<br>both<br>wrong<br>_¬_(_p, T_)_∧¬_(_p, S_)_∧¬_(_c, T_)<br>The model is unable to generate a correct solution or test<br>cases. Refning the program over **faulty test samples**<br>will not lead to the correct answer.<br>-<br>_¬_(_p, T_)_∧_(_p, S_)_∧_(_c, T_)<br>In the event of self-test failure, there must have been at<br>least one error in either program or tests, so this condition<br>**should never occur**.|



**Details of Ranking Strategy Comparison** (Figure 3.b) We obtain 11 candidate programs from FUNCODER on HumanEval with GPT-3.5 and rank them through three strategies. This ensures that the same candidate set is used for a fair comparison. An effective ranking strategy should prioritize 

> 5math_equivalence: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py 

20 

placing correct programs at the forefront and filter out those with errors. Thus, we measure the effectiveness by computing Pass@k results on the top-k-ranked programs selected by each strategy. The Pass@11 result serves as an upper bound as it uses all programs to compute the pass rate. 

**How We Count Frequently Used Functions in MATH** (Table 4) In the mathematical reasoning experiments, we used a subset of 500 items from the MATH test set, with an average of 71.4 questions per subject. However, it is not very confident to represent common functions from only 71.4 programs. Therefore, we sample 3000 problems from the MATH test set for this experiment and run the _divideonly_ setting of FUNCODER on them. Then, the occurrence of sub-functions is counted based on their names after extracting the function nodes of code trees for each category. 

## **A.6 Detailed Explanation of Algorithm** 

We hereby provide a detailed explanation of FUNCODER algorithm works, with respect to Algorithm 1 from Figure 2 (a copy is included below for simple reading). As mentioned, FUNCODER is a recursive process following a DFS pattern. We use square brackets (e.g. `[L1]` ) below to denote line numbers in the pseudocode. 

|**Algorithm 1**FUNCODER procedure<br>**Require:** Entry func,_f_root =_{h_root_, d_root_, ϕ}_<br>**Require:** Large language model, LLM<br>1: **function**FUNCODER(_f_cur)<br>2:<br>_— Divide —_<br>3:<br>_f ′_<br>cur_, {fi} ←_EXTRACT(LLM(_f_cur))<br>4:<br>**for**_fi ∈{fi}_**do**<br>5:<br>**if**_bi_ **is**NOTIMPLEMENTED**then**<br>6:<br>_f ∗_<br>_i ←_FUNCODER(_fi_)<br>_▷_recursion<br>7:<br>**end if**<br>8:<br>ADDCHILD(_f_cur_, f ∗_<br>_i_ )<br>9:<br>**end for**<br>10:<br>_— Conquer —_<br>11:<br>_F_cur _←_SAMPLE(LLM(_f ′_<br>cur_,_CHILD(_f_cur))<br>12:<br>_f ∗_<br>cur _←_FUNCONSENSUS(_F_cur)<br>13:<br>**return**_f ∗_<br>cur<br>14: **end function**<br>15: **return**FUNCODER(_f_root)<br>_▷_starts from root|𝑎<br>𝐴<br>𝑏<br>𝑐<br>Divide(𝑎)<br>𝐴<br>𝐵<br>𝑐<br>𝑑<br>𝑒<br>Divide(𝑏)<br>𝐴<br>𝐵<br>𝑐<br>𝐷<br>𝐸<br>Divide(𝑑,𝑒)<br>Conquer(𝐷,𝐸)<br>𝐴<br>𝐵<br>𝑐<br>𝐷<br>𝐸<br>Conquer(𝐵)<br>𝐴<br>𝐵<br>𝐶<br>𝐷<br>𝐸<br>Divide(_c_)<br>Conquer(𝐶)<br>𝐴<br>𝐵<br>𝐶<br>𝐷<br>𝐸<br>Conquer(𝐴)<br>**Example**|
|---|---|



Figure 5: Left: Algorithm for FUNCODER. Right: Decomposition example of `A[B[DE]C]` . 

FUNCODER `[L1]` , when solving each function _f_ , first performs the _Divide_ stage `[L3-L9]` , where the LLM initially writes the function and identifies some potential sub-problems, represented as sub-function stubs (e.g., `def f(xs: list[int]) -> int` ) `[L3]` . In this process, we identify the sub-problems of the current problem, thereby understanding the dependency between functions. 

For each decomposed sub-problem _g_ 1 _, g_ 2 _, . . ._ , we recursively use FUNCODER to obtain the final implementation _Gi_ for that sub-problem `[L5-L8]` . This _Gi_ shall replace the previously incomplete subfunction stub signature in the final program. 

Now that all sub-problems of _f_ are implemented, we move on to the Conquer stage `[L11-13]` to complete the larger problem. By combining the signature _f_ and the final implementations _G_ 1 _, G_ 2 _, . . ._ of sub-problems, we generate the complete implementation _F_ `[L11]` and return it `[L13]` . 

We hierarchically describe how this algorithm works in detail by combining it with the example given in the right half of Figure 5. 

```
[a.1]FunCoder(a)
|[a.3]LLM(a)->A,{b,c}#divide
|--[b.1]FunCoder(b)
||[b.3]LLM(b)->B,{d,e}#divide
```

21 

```
||--[d.1]FunCoder(d)
|||[d.3]LLM(d)->D,{}#divide
||+--[d.13]returnD#nothingtoconquer
||--[e.1]FunCoder(e)
|||[e.3]LLM(e)->E,{}#divide
||+--[e.13]returnE#nothingtoconquer
||[b.11]LLM(B,{D,E})->B*#conquer
|+--[b.13]returnB*
|--[c.1]FunCoder(c)
||[c.3]LLM(c)->C,{}#divide
|+--[c.13]returnC#nothingtoconquer
|[a.11]LLM(A,{B,C})->A*#conquer
+--[a.13]returnA*#finalresult
```

## **A.7 Token-cost Complexity** 

**Example** We use the example from Figure 5, where the final program consists of 5 functions `A[B[D,E],C]` , and `A` serves as the entry to the program. Here we respect the aforementioned notations, and further use the lower-case letter _a_ to represent the number of stub tokens, upper-case _A_ to represent the number of result tokens, for the function `A` , and other functions likewise. Let _N_ = _A_ + _B_ + _C_ + _D_ + _E_ be the token-cost complexity of the final result. 

In _Standard_ , the code is only generated once to complete the given stub. We use parentheses to represent the order of LLM calls in a full process. 

```
(1)a->ABCDE
inputtokens=a
outputtokens=A+B+C+D+E
overall=O(N)
```

In each step of the FUNCODER _/Divide_ stage, the to-be-implemented function will serve as the context. The function (stub) will be implemented and sub-function stubs are to be declared. 

```
(1)a->Abc
(2)b->Bde
(3)d->D
(5)e->E
(8)c->C
inputtokens=a+b+c+d+e
outputtokens=A+b+B+c+C+d+D+e+E<2N
overall=O(N)
```

In every FUNCODER _/Conquer_ , the context shall include the current function’s definition and finalized implementations of sub-functions. The output is the re-implemented current function. 

```
(4)d->kD
(6)e->kE
(7)bDE->kB
(9)c->kC
(10)aBC->kA
inputtokens=a+b+B+c+C+d+D+e+E<2N
outputtokens=kA+kB+kC+kD+kE=kN
overall=O(kN)
```

These stages in all bring FUNCODER’s total token consumption to strictly _O_ ( _kN_ ) for every problem. 

**Token Complexity of FUNCODER is** _O_ ( _kN_ ) Define _N_ as the token length of the final program, which is correlated to the inherent complexity of the problem, and define _k_ as the number of sampled candidates. We first explain in detail that the worst-case token cost of FUNCODER is _O_ ( _kN_ ): 

22 

- The naive _Standard_ method should naturally generate _O_ ( _N_ ) tokens. Sampling-based baselines like _CodeT_ cost _O_ ( _kN_ ) tokens. 

- FunCoder goes through the _Divide_ stage and the _Conquer_ stage for each of the functions. 

- Based on the current function, _Divide_ generates an implementation of itself and stubs for subfunctions. Within this stage, each function would appear at most once in input and twice in output. All _Divide_ stages consume no more than 3 _N_ tokens. 

- _Conquer_ regenerates the parent function based on its stub and all finalized sub-functions. Herein each function will appear at most twice in input, and sampled _k_ times in output. If _k_ = 1, consensus is implicitly disabled. All _Conquer_ stages shall consume at most ( _k_ + 2) _N_ tokens. 

So FunCoder requires no more than ( _k_ + 5) _N_ tokens in input-output, making its token consumption _O_ ( _kN_ ) even at worst-case, aligning with other sampling-based methods such as CodeT and AlphaCode-like clustering. Furthermore, when sampling is disabled ( _k_ = 1), our method has a token consumption of _O_ ( _N_ ), which also aligns with the vanilla _Standard_ method. 

## **A.8 Discussion About Functional Consensus** 

This section focuses on why _functional consensus_ might enhance the correctness of programs and how it differs from other consistency-based methods. Self-consistency (Wang et al., 2023) is widely employed in the realm of LLM reasoning. It samples multiple sets of answers and uses voting to select the most consistent result, where the answers typically consist of named entities, choice options, or numbers. However, this approach faces challenges when voting on sampled programs, as programs describe executable logic instead of data, making it unobvious to determine whether two programs are equivalent just from the looks. When it comes to picking programs, _functional consensus_ in FUNCODER looked beyond the literal symbols and used a different approach. It uses inputs and execution results to compare behavioral differences among programs. There have been similar methods, such as the strict clustering approach in AlphaCode (Li et al., 2022), which samples a set of program inputs and then clustering programs with identical outputs into the same group. The final program is then selected from the largest cluster. 

However, the idea of grouping programs by the ‘identicalness’ of outputs is not without fallacies, since programs rarely specialize in solving one single irreducible problem – they deal with a variety of inputs, conditions and mysterious cases. The result of this, where different solutions could have many common behaviors and some distinct behaviors, is referred to as the term ‘special-case similarity’ in the FrAngel paper (Shi et al., 2019). We consider a correct program solution which has multiple ‘special-case similar’ programs that are partially correct in different ways, for example: 

- One program behaving correctly on the general case (almost all) but missed a few edge cases 

- Another program got one edge case correct but didn’t manage to deal with the general case 

- Yet another program got all edge cases correct but crashed on the general case 

- A buggy program that behaves correctly on all available test cases but none of these tests trigger the bug (literally test coverage problem) 

- And many programs that turn out to be frenzy mixtures of all the above 

If we had a pool of programs that contained the fully-correct program and an assortment of other programs that respected certain cases of the problem as aforementioned, it’d be obvious that the fullycorrect would be decently ‘special-case similar’ to the rest of the programs, for their similar behavior on inputs. These execution outputs are programmatically obtained and automatically compared against each other without any human intervention or LLM calling required, the process of which sits at the core of our _functional consensus_ algorithm. 

Therefore, with _functional consensus_ , where the solutions with common behaviors are promoted, we could intuitively expect the result to be a higher likelihood of a fully-correct program. Provided below is a hypothetical example demonstrating why _functional consensus_ prevails: 

**Example** Consider the problem of finding all square roots in the complex domain of a non-zero real number (stored in `float32` ). To get the answer right for all inputs, the function must consider 2 

23 

cases: A) non-zero numbers have 2 square roots; B) square roots of negative numbers are imaginary. 10 candidate functions are sampled as below: 

- 5 results ( _a_ 1 _, a_ 2 _, a_ 3 _, a_ 4 _, a_ 5) only considered case A and got just positive inputs right. For negative numbers, they literally gave up and crashed. 

- 3 results ( _b_ 6 _, b_ 7 _, b_ 8) remembered to consider case B, gave 2 imaginary results for negative numbers, but forgot to do the same for positive numbers, returning only 1 square root therein. 

- Only 2 results ( _c_ 9 _, c_ 10) considered all cases and returned correct results for all inputs. 

If we pick the program through ‘clustering’, the final result would be one of the 5 results ( _a_ 1 _, a_ 2 _, . . . , a_ 5) that only considered case A, which is evidently not the correct solution. But with _functional consensus_ , the final result is vastly different, since we consider the similarity between the functions based on their behavior on different inputs. Without loss of generality, suppose that there are 2 test inputs 4 _._ 0, _−_ 9 _._ 0, one for each of the 2 cases. We calculate the similarity as follows: 

- Programs _ai_ got only _√_ 4 _._ 0 = [2 _._ 0 _, −_ 2 _._ 0] right so each program here are similar with programs ( _a_ 1 _, a_ 2 _, a_ 3 _, a_ 4 _, a_ 5 _, c_ 9 _, c_ 10), scoring 7 points. 

- Since _bj_ only went well with _[√] −_ 9 _._ 0 = [3 _._ 0 _i, −_ 3 _._ 0 _i_ ], programs here only score 5 points for case B with the ones ( _b_ 6 _, b_ 7 _, b_ 8 _, c_ 9 _, c_ 10). 

- Each program in _ck_ gets 7 points for _√_ 4 _._ 0 with ( _a_ 1 _, a_ 2 _, a_ 3 _, a_ 4 _, a_ 5 _, c_ 9 _, c_ 10), and gets 5 points for _√−_ 9 _._ 0 with ( _b_ 6 _, b_ 7 _, b_ 8 _, c_ 9 _, c_ 10). Totals to 12 points. 

The final result apparently leaned towards _ck_ as the correct solution, even if their outputs as a whole weren’t even half as much as _ai_ is. Through this example, we illustrate that _functional consensus_ has the potential to identify the correct samples even at their minority, outperforming other methods such as self-consistency or clustering. 

## **A.9 Supplementary Results** 

**Token Usage** We provide token usage results in Table 9 for FUNCODER and baseline methods on the HumanEval dataset with the GPT-3.5 model, whereas usage results on other datasets are provided in Table 10. We report the average token usage per problem. The token usage is computed through the sum of prompt tokens and completion tokens returned by OpenAI API chat completion call[6] . For LDB, we report their token usage in the original paper (Zhong et al., 2024). 

Table 9: Token usage for different settings of FUNCODER and baseline methods on HumanEval, all evaluated on GPT-3.5-turbo. The LDB results are reported in the original paper. The main setting for LDB and FUNCODER is bolded. 

|**Method**<br>**Setting**<br>**Pass@1**|**Tokens**<br>Min.<br>Max.<br>Avg.<br>Med.|
|---|---|
|Standard<br>One-time<br>68.3<br>CodeT<br>One-time + Self-Test@11<br>81.1(+12.8)<br>Refexion<br>maxiter=2<br>69.5(+1.2)|648<br>**1477**<br>**886.7**<br>861<br>2298<br>9645<br>4479.1<br>4166<br>**416**<br>4906<br>1416.1<br>**754**|
|LDB_(reported)_<br>line-level<br>80.5(+12.2)<br>**block-level**<br>82.9(+14.6)<br>function-level<br>79.9(+11.6)|-<br>-<br>24K<br>-<br>-<br>-<br>23K<br>-<br>-<br>-<br>27K<br>-|
|FUNCODER<br>One-pass<br>72.6(+4.3)<br>Two-pass<br>78.7(+10.4)<br>Two-pass + Consensus@5<br>83.5(+15.2)<br>**Two-pass + Consensus@11**<br>**85.4 (+17.1)**|826<br>3489<br>1233.7<br>1132<br>2197<br>8406<br>3343.2<br>3078<br>2455<br>9432<br>4040.9<br>3800<br>3015<br>13850<br>5402.0<br>5166|
|FUNCODER<br>Two-pass + Self-Test@11<br>80.5(+12.2)<br>_(ablation)_<br>Two-pass + Clustering@11<br>75.0(+6.7)|2967<br>13758<br>5408.3<br>5184<br>3044<br>9958<br>5070.7<br>4888|



6https://platform.openai.com/docs/guides/text-generation/managing-tokens 

24 

Table 10: Token usage of FUNCODER and baseline methods on other datasets, i.e. MBPP, xCodeEval and MATH. Results are evaluated on GPT-3.5-turbo. 

|**Dataset**<br>**Method**<br>**Pass@1**|**Tokens**<br>Min.<br>Max.<br>Avg.<br>Med.|
|---|---|
|MBPP<br>Standard<br>72.0<br>CodeT<br>76.0(+4.0)<br>Refexion<br>72.5(+0.5)<br>FUNCODER<br>**78.5 (+6.5)**|577<br>**2102**<br>**744.9**<br>717.0<br>2232<br>8172<br>2945.3<br>2866.0<br>**391**<br>3379<br>1205<br>**569.5**<br>3462<br>13229<br>5049.9<br>4644.0|
|xCodeEval<br>Standard<br>20.2<br>CodeT<br>23.2(+3.0)<br>Refexion<br>20.6(+0.4)<br>FUNCODER<br>**31.4 (+11.2)**|**1051**<br>**3343**<br>**1599.5**<br>**1530.0**<br>2264<br>9245<br>3937.4<br>3733.0<br>2977<br>1003222<br>401767.3<br>328591.5<br>4883<br>53225<br>10559.7<br>8927.0|
|MATH<br>PoT<br>41.0<br>FUNCODER<br>**54.0 (+13.0)**|**551**<br>**5867**<br>**953.0**<br>**835.0**<br>2622<br>30139<br>7075.5<br>5666.5|



**Full Results for Code Generation** We provide results for all conducted experiments on code generation benchmarks in Table 11. Our method consistently improves the baseline on community models by averaging 11% on MBPP and 150% on xCodeEval. It is worth noting that small models have a tendency to have low pass rates on competition problems, leading to a relatively higher randomness, therefore we run 3 experiments and report the median result. 

**Full Results for MATH** The MATH dataset divides the problems into five levels of difficulty. The difficulty distribution of our test set can be found in Table 7. We report the average accuracy of FUNCODER and other methods for each math subject in Table 12 and results for each level in Table 13. The results of Cumulative Reasoning are obtained from the original paper (Zhang et al., 2024). Experiment results demonstrate that our method consistently enhances the model’s reasoning ability across all levels of MATH. 

25 

Table 11: Results for Code Generation. We report Pass@1 as evaluate metric. Results from the original paper are underlined, and the best results are bold. 

|**Model**<br>**Method**|**HumanEval**<br>Pass@1<br>∆_↑_|**HumanEval**<br>Pass@1<br>∆_↑_|**MBPP**<br>Pass@1<br>∆_↑_|**xCodeEval**<br>_Easy_<br>_Mid_<br>_Hard_<br>_Expert_<br>All|
|---|---|---|---|---|
|GPT-3.5<br>Standard<br>CodeT<br>Refexion<br>LDB<br>FUNCODER||68.3<br>-<br>81.1<br>+12.8<br>69.5<br>+1.2<br>82.9<br>+14.6<br>**85.4**<br>+17.1|72.0<br>-<br>76.0<br>+4.0<br>72.5<br>+0.5<br>76.0<br>+4.0<br>**78.5**<br>+6.5|44.4<br>15.2<br>4.6<br>0.0<br>20.2<br>50.6<br>16.1<br>8.0<br>0.0<br>23.2<br>44.4<br>17.0<br>5.7<br>0.0<br>20.6<br>-<br>-<br>-<br>-<br>-<br>**62.4**<br>**29.5**<br>**11.6**<br>0.0<br>**31.4**|
|GPT-4<br>Standard<br>Parsel<br>CodeT<br>Refexion<br>MetaGPT<br>FUNCODER||82.9<br>-<br>85.0<br>+2.1<br>90.9<br>+8.0<br>91.0<br>+8.1<br>85.9<br>+3.0<br>**94.5**<br>+11.6|73.5<br>-<br>-<br>-<br>77.0<br>+3.5<br>77.1<br>+3.6<br>-<br>-<br>**79.5**<br>+6.0|68.5<br>39.3<br>19.5<br>1.7<br>37.4<br>-<br>-<br>-<br>-<br>-<br>76.4<br>51.8<br>21.8<br>**3.4**<br>44.0<br>71.3<br>41.1<br>19.5<br>2.5<br>38.6<br>-<br>-<br>-<br>-<br>-<br>**83.1**<br>**58.0**<br>**26.4**<br>**3.4**<br>**48.6**|
|GPT-4o mini<br>Standard<br>CodeT<br>FUNCODER||87.3<br>-<br>90.9<br>+3.6<br>**91.5**<br>+4.2|76.0<br>-<br>75.5<br>-0.5<br>**77.5**<br>+1.5|65.7<br>44.6<br>9.2<br>0.0<br>35.4<br>71.9<br>49.1<br>**16.1**<br>0.0<br>39.6<br>**72.5**<br>**52.3**<br>11.5<br>0.0<br>**39.8**|
|Llama38_b_<br>Standard<br>CodeT<br>FUNCODER||61.6<br>-<br>68.9<br>+7.3<br>**79.7**<br>+18.1|60.5<br>-<br>61.5<br>+1.0<br>**62.5**<br>+2.0|9.0<br>**1.8**<br>0.0<br>0.0<br>3.6<br>12.4<br>0.0<br>0.0<br>0.0<br>4.4<br>**22.0**<br>0.9<br>0.0<br>0.0<br>**8.0**|
|Codestral22_b_<br>Standard<br>CodeT<br>FUNCODER||79.3<br>-<br>86.0<br>+7.3<br>**89.0**<br>+9.7|68.5<br>-<br>74.0<br>+5.5<br>**74.5**<br>+6.0|27.5<br>4.5<br>2.3<br>0.0<br>11.4<br>34.8<br>7.1<br>**3.4**<br>0.0<br>14.8<br>**49.4**<br>**15.2**<br>**3.4**<br>0.0<br>**22.0**|
|StableCode3_b_<br>Standard<br>CodeT<br>FUNCODER||61.0<br>-<br>75.0<br>+14.0<br>**81.0**<br>+20.0|51.5<br>-<br>57.5<br>+6.0<br>**63.5**<br>+12.0|7.3<br>0.9<br>0.0<br>0.0<br>2.8<br>11.2<br>1.8<br>0.0<br>0.0<br>4.6<br>**13.5**<br>**4.5**<br>**1.1**<br>0.0<br>**6.2**|
|CodeLlama34_b_<br>Standard<br>CodeT<br>FUNCODER||43.9<br>-<br>55.5<br>+11.6<br>**66.5**<br>+22.6|53.5<br>-<br>56.5<br>+3.0<br>**58.5**<br>+5.0|2.3<br>0.0<br>0.0<br>0.0<br>0.8<br>10.1<br>0.0<br>0.0<br>0.0<br>3.6<br>**10.2**<br>0.0<br>0.0<br>0.0<br>**3.6**|
|StarCoder215_b_<br>Standard<br>CodeT<br>FUNCODER||59.8<br>-<br>70.7<br>+10.9<br>**78.7**<br>+18.9|64.5<br>-<br>66.0<br>+1.5<br>**70.0**<br>+5.5|18.0<br>0.9<br>**2.3**<br>0.0<br>7.2<br>13.5<br>0.9<br>0.0<br>0.0<br>5.0<br>**29.2**<br>**4.5**<br>0.0<br>0.0<br>**11.6**|



26 

Table 12: Experimental results on MATH, a competition-level mathematical reasoning benchmark. Best results are in bold. Text-based reasoning methods are denoted with _[†]_ , while others use programaided reasoning. We report both overall results and results in seven subjects: _Prealgebra_ , _Algebra_ , _Number Theory_ , _Counting & Probability_ , _Geometry_ , _Intermediate Algebra_ , and _Precalculus_ . 

|**Model**|**Method**|_Prealg._|_Alg._|_NT_|_Prob._|_Geo._|_InterAlg._|_Precalc._|**Overall**|
|---|---|---|---|---|---|---|---|---|---|
||Standard_†_|62.2|37.4|20.0|29.8|31.0|24.4|21.8|34.6|
||CoT_†_|59.8|51.1|28.9|29.8|28.6|26.7|30.9|40.0|
|GPT-3.5|PoT|68.3|50.4|33.3|48.9|21.4|18.2|29.1|41.0|
||Self-Refne|74.4|49.6|48.9|57.4|28.6|35.6|36.4|48.6|
||FUNCODER|**76.8**|**61.2**|**55.6**|**59.6**|**34.1**|**36.0**|**41.8**|**54.0**|
||Standard_†_|81.7|82.7|71.1|72.3|**59.5**|46.7|47.3|68.2|
||CoT_†_|84.1|87.1|62.2|68.1|45.2|48.9|54.5|68.6|
|GPT-4|PoT<br>Self-Refne|79.3<br>82.9|80.6<br>82.0|75.6<br>77.8|72.3<br>76.6|50.0<br>54.8|47.8<br>55.6|58.2<br>**63.6**|68.2<br>72.2|
||CR|86.6|86.3|**88.7**|71.1|53.7|51.5|51.8|72.2|
||FUNCODER|**89.0**|**92.8**|82.2|**83.0**|**59.5**|**63.3**|56.4|**78.2**|
||Standard_†_|79.3|83.5|75.6|**87.2**|47.6|57.8|56.4|71.8|
||CoT_†_|**90.2**|**95.7**|**82.2**|68.1|50.0|**61.1**|**61.8**|**77.2**|
|GPT-4o mini|PoT|80.5|84.2|77.8|72.3|50.0|60.0|50.9|71.0|
||Self-Refne|79.3|83.5|75.6|**87.2**|47.6|57.8|56.4|71.8|
||FUNCODER|81.7|83.5|80.0|80.9|**59.5**|60.0|54.5|73.2|
||CoT_†_|56.1|**47.5**|31.1|34.0|**40.5**|14.4|**38.2**|38.6|
|Llama38_b_|PoT|67.1|32.4|24.4|34.0|16.7|21.1|18.2|32.6|
||FUNCODER|**67.9**|45.7|**51.1**|**53.2**|19.0|**37.8**|30.9|**45.0**|
|Codestral22_b_|PoT<br>FUNCODER|70.7<br>**81.7**|56.1<br>**61.9**|**46.7**<br>**46.7**|44.7<br>**55.3**|21.4<br>**28.6**|26.7<br>**45.6**|30.9<br>**38.2**|45.6<br>**54.8**|
|StableCode3_b_|PoT<br>FUNCODER|20.7<br>**46.3**|14.4<br>**30.2**|17.8<br>**20.0**|25.5<br>**29.8**|**4.8**<br>**4.8**|8.9<br>**20.0**|9.1<br>**18.2**|14.4<br>**26.6**|
|CodeLlama34_b_|PoT<br>FUNCODER|35.5<br>**44.8**|26.1<br>**46.1**|15.0<br>**37.8**|16.7<br>**34.1**|0.0<br>**13.6**|5.5<br>**24.6**|33.3<br>**37.5**|15.2<br>**24.4**|
|StarCoder215_b_|PoT<br>FUNCODER|46.3<br>**72.0**|29.5<br>**39.6**|28.9<br>**40.9**|25.5<br>**46.8**|21.4<br>**23.8**|27.8<br>**28.1**|23.6<br>**27.3**|30.2<br>**40.8**|



27 

Table 13: Full results of each method at different levels of MATH. The best results are in bold. Text-based reasoning methods are denoted with _[†]_ , while others use program-aided reasoning. 

|**Model**|**Method**|_Level 1_|_Level 2_|_Level 3_|_Level 4_|_Level 5_|**Overall**|
|---|---|---|---|---|---|---|---|
||Standard_†_|61.5|51.1|43.5|25.9|17.7|34.6|
||CoT_†_|76.9|48.9|50.9|33.6|21.8|40.0|
|GPT-3.5|PoT|61.5|51.1|56.5|33.6|24.1|41.0|
||Self-Refne|**84.6**|61.1|65.7|32.8|31.3|48.6|
||FUNCODER|**84.6**|**65.9**|**68.5**|**43.1**|**37.4**|**54.0**|
||Standard_†_|89.7|85.6|**83.3**|55.2|51.0|68.2|
||CoT_†_|94.9|81.1|77.8|64.7|50.3|68.6|
|GPT-4|PoT<br>Self-Refne|94.9<br>94.9|80.0<br>81.1|74.1<br>**83.3**|63.8<br>62.1|53.1<br>60.5|68.2<br>72.2|
||CR|90.7|**90.0**|81.9|66.4|52.2|72.2|
||FUNCODER|**94.9**|**90.0**|81.5|**75.9**|**66.0**|**78.2**|
||Standard_†_|87.2|82.2|80.6|62.9|61.9|71.8|
||CoT_†_|97.4|90.0|87.0|71.6|61.2|77.2|
|GPT-4o mini|PoT|89.7|81.1|76.9|63.8|61.2|71.0|
||Self-Refne|87.2|82.2|80.6|62.9|61.9|71.8|
||FUNCODER|94.9|82.2|81.5|62.9|63.9|73.2|
||CoT_†_|76.9|46.7|46.3|25.9|**27.9**|38.6|
|Llama38_b_|PoT|64.1|43.3|41.7|25.0|17.0|32.6|
||FUNCODER|**79.5**|**60.0**|**52.3**|**37.4**|**27.9**|**45.0**|
|Codestral22_b_|PoT<br>FUNCODER|79.5<br>84.6|56.7<br>66.7|57.4<br>67.6|34.5<br>43.1|29.9<br>39.5|45.6<br>54.8|
|StableCode3_b_|PoT<br>FUNCODER|35.9<br>**53.8**|22.2<br>**37.8**|19.4<br>**35.2**|7.8<br>**21.6**|5.4<br>**10.2**|14.4<br>**26.6**|
|CodeLlama34_b_|PoT<br>FUNCODER|36.1<br>**60.6**|30.7<br>**52.1**|28.0<br>**44.3**|13.0<br>**28.8**|8.8<br>**16.3**|15.2<br>**24.4**|
|StarCoder215_b_|PoT<br>FUNCODER|43.6<br>71.8|44.4<br>57.8|45.4<br>55.1|20.7<br>26.7|14.3<br>23.3|30.2<br>40.8|



28 

## **B Examples** 

We provide example solutions for the baseline and FUNCODER on code generation and mathematical reasoning. All results are generated with GPT-3.5. 

## **B.1 Code Generation** 

**Info:** `xCodeEval/ff3c39b759a049580a6e96c66c904fdc` (binary search, difficulty 1800) 

## **Problem** 

```
timelimitpertest:1second
memorylimitpertest:256megabytes
input:standardinput
output:standardoutput
```

```
##Description
```

```
Youhavetwofriends.Youwanttopresenteachofthemseveral
positiveintegers.Youwanttopresentcnt1numberstothefirst
friendandcnt2numberstothesecondfriend.Moreover ,youwant
allpresentednumberstobedistinct ,thatalsomeansthatno
numbershouldbepresentedtobothfriends.Inaddition ,thefirst
frienddoesnotlikethenumbersthataredivisiblewithout
remainderbyprimenumberx.Thesecondonedoesnotlikethe
numbersthataredivisiblewithoutremainderbyprimenumbery.Of
course ,you 'renotgoingtopresentyourfriendsnumberstheydon
'tlike.Yourtaskistofindsuchminimumnumberv,thatyoucan
formpresentsusingnumbersfromaset1,2,...,v.Ofcourseyou
maychoosenottopresentsomenumbersatall.Apositiveinteger
numbergreaterthan1iscalledprimeifithasnopositive
divisorsotherthan1anditself.
##Input
Theonlylinecontainsfourpositiveintegerscnt1 ,cnt2 ,x,y(1<=
cnt1 ,cnt2 <109;cnt1+cnt2 <=109;2<=x<y <=3*104) -thenumbersthat
aredescribedinthestatement.Itisguaranteedthatnumbersx,y
areprime.
##Output
Printasingleinteger-theanswertotheproblem.
##SampleInput1
```
3123
```
##SampleOutput1
```
5
```
##SampleInput2
```
1323
```
##SampleOutput2
```
```

29 

```
4
```
```

`## Notes In the first sample you give the set of numbers {1,3,5} to the first friend and the set of numbers {2} to the second friend. Note that if you give set {1, 3, 5} to the first friend , then we cannot give any of the numbers 1, 3, 5 to the second friend. In the second sample you give the set of numbers {3} to the first friend , and the set of numbers {1, 2, 4} to the second friend. Thus , the answer to the problem is 4.` **B.1.1 Standard** `def main () -> None: cnt1 , cnt2 , x, y = list ( map ( int , input ().split ())) # binary search l, r = 1, 2 * (cnt1 + cnt2) # upper bound while l < r: mid = (l + r) // 2 # number of numbers that can be given to the first friend cnt_x = mid // x # number of numbers that can be given to the second friend cnt_y = mid // y # number of numbers that can be given to both friend cnt_xy = mid // (x * y) # number of numbers that can't be given to any friend cnt_none = mid - cnt_x - cnt_y + cnt_xy if cnt_x >= cnt1 and cnt_y >= cnt2 and cnt_none >= cnt1 + cnt2 : r = mid else : l = mid + 1 print (l)` 

**Result:** Wrong answer on sample test 1: expected: ’5’, found: ’8’ [1th token] ✗ 

**B.1.2 FUNCODER** 

```
defmain ()->None:
cnt1 ,cnt2 ,x,y=map(int ,input ().split ())
left=1
right=int(2e9)
whileleft<right:
mid=(left+right)//2
ifis_valid(mid ,x,y,cnt1 ,cnt2):
right=mid
else:
left=mid+1
print(left)
defis_valid(v,x,y,cnt1 ,cnt2):
#checkiftherearecnt1numbersnotdivisiblebyx
non_x_numbers=v-(v//x)
ifnon_x_numbers<cnt1:
returnFalse
#checkiftherearecnt2numbersnotdivisiblebyy
non_y_numbers=v-(v//y)
ifnon_y_numbers<cnt2:
returnFalse
```

30 

```
#checkiftherearecnt1+cnt2numbersdivisiblebybothxandy
lcm=(x*y)//gcd(x,y)
common_numbers=v-(v//lcm)
ifcommon_numbers<cnt1+cnt2:
returnFalse
returnTrue
defgcd(a:int ,b:int)->int:
""" getthegreatestcommondivisoroftwonumbers$a$and$b$ """
whileb:
a,b=b,a%b
returna
```

**Result:** Accepted ✓ 

## **B.2 Mathematical Reasoning** 

**Info:** `MATH/number_theory/42` (level 5) 

**Problem** What is the sum of the smallest and second-smallest positive integers _a_ satisfying the congruence 27 _a ≡_ 17 (mod 40) ? 

**Ground Truth:** 62 

**B.2.1 CoT** 

```
Wecansolvefor$a$bymultiplyingbothsidesofthecongruenceby
themodularinverseof$27$modulo$40$.Since$27$and$40$are
relativelyprime ,weknowthat$27$hasamodularinversemodulo
$40$.WecanfindthisinverseusingtheEuclideanalgorithm:
\begin{align *}
\text{gcd }(40 ,27)&=\text{gcd }(27 ,13)
&=\text{gcd }(13 ,1)
&=1.
\end{align *}
Therefore ,themodularinverseof$27$modulo$40$exists.Wecanfind
itusingtheEuclideanalgorithmbackwards:
\begin{align *}
1&=13-3\cdot4
&=13-3(40-27)
&=4\cdot27-3\cdot40.
\end{align *}
Thus ,themodularinverseof$27$modulo$40$is$4$.Multiplyingboth
sidesofthecongruence$27a\equiv17\pmod {40}$by$4$gives
$108a\equiv68\pmod {40}$.Simplifyingtheleft -handsidegives
$8a\equiv28\pmod {40}$.Dividingbothsidesby$4$gives$2a\
equiv7\pmod {10}$.Thesmallestpositiveinteger$a$satisfying
thiscongruenceis$a=9$,andthesecond -smallestis$a=19$.
Therefore ,thesumofthesmallestandsecond -smallestpositive
integers$a$satisfyingthecongruence$27a\equiv17\pmod {40}$is
$9 +19=\ boxed {28}$.
```

**Result:** 28 ✗ 

31 

**B.2.2 PoT** `def solution (): r""" What is the sum of the smallest and second -smallest positive integers $a$ satisfying the congruence $$27a\equiv 17 \pmod {40}~? $$""" k = 1 candidates = [] while len (candidates) < 2: a = (40 * k + 17) // 81 if a > 0 and (40 * k + 17) % 81 == 0: candidates.append(a) k += 1 return sum (candidates)` 

**Result:** 74 ✗ 

**B.2.3 FUNCODER** `from typing import Tuple def solution (): r""" What is the sum of the smallest and second -smallest positive integers $a$ satisfying the congruence $$27a\equiv 17 \pmod {40}~? $$""" a = 27 m = 40 b = 17 inverse = find_inverse(a, m) x = (b * inverse) % m # Find the smallest and second -smallest positive integers satisfying the congruence equation # using the formula: x + k * (m // gcd(a, m)) gcd , _, _ = extended_euclid (a, m) s1 = x s2 = x + (m // gcd) return s1 + s2 def find_inverse(a: int , m: int ) -> int : """ find the inverse of $a$ modulo $m$ """ _, x, _ = extended_euclid (a, m) return x % m if x >= 0 else (x + m) % m def extended_euclid (a: int , b: int ) -> Tuple[ int , int , int ]: """ Extended Euclidean Algorithm """ if b == 0: return a, 1, 0 gcd , x1 , y1 = extended_euclid (b, a % b) x = y1 y = x1 - (a // b) * y1 return gcd , x, y` **Result:** 62 ✓ 

32 

## **C Prompts** 

**C.1 Prompt for Standard** 

```
Youareaprogrammingcopilot ,youcansolveaproblembywriting
Pythonfunctions.Yourtaskisto:
-YouneedtowriteaPythonfunctionthatreturnstheanswer.
-Youcanimportlibrariestobettersolvetheproblem.
-Donotwriteanycodeoutsidethefunction(importingisaccepted)
.
<User >:
Let 'sthinkstepbystepandcompletethefollowingPythonfunction`
sum_factor `thatsolves:
```python
defsum_factor(a:int ,b:int)->int:
""" Returnthesumofallcommonprimefactorsof$a$and$b$ """
raiseNotImplementedError ()
```
<Assistant >:
First ,Ineedtogettheprimefactorsof$a$and$b$.
Second ,Icanuse`for `looptofindcommonelementintwofactors
list.
Hereisthe`sum_factor `function:
```python
defsum_factor(a:int ,b:int)->int:
""" Returnthesumofallcommonprimefactorsof$a$and$b$ """
factors_a=prime_factor (a)
factors_b=prime_factor (b)
common_factors=get_common(factors_a ,factors_b)
returnsum(common_factors )
defprime_factor(x:int)->list:
""" getalistofprimefactorsofnumber$x$ """
ret=[]
i=1
whilei*i<=x:
i+=1
ifx%i==0andis_prime(i):
ret.append(i)
returnret
defis_prime(x:int)->bool:
""" determine$x$isaprimenumberornot """
ifx<2:
returnFalse
foriinrange(2,int(x**0.5)+1):
ifx%i==0:
returnFalse
returnTrue
defget_common(a:list ,b:list)->list:
""" getcommonelementintwolist$a$and$b$ """
ret=[]
foritemina:
ifiteminb:
ret.append(item)
returnret
```

33 

```
```
<User >:
Let 'sthinkstepbystepandcompletethefollowingPythonfunction`{
cur_func_name}`thatsolves:
```python
{cur_func}
```
```

## **C.2 Prompt for Divide** 

```
Youareaprogrammingcopilot ,youcansolveaproblembywriting
Pythonfunctions.Yourtaskisto:
-Foreveryturn ,youneedtowriteaPythonfunctionthatreturns
theanswerbasedonCurrentCode(notcodeinchathistory).
-Donotmodifyfunctionname ,argnames ,docstringingiven
functions.
-Youcanimportlibrariestobettersolvetheproblem.
-Ifasinglefunctionistoohardtosolve ,youcandecomposeit
intomultiplesmallerfunctions.
-Youcanleavenewfunctionunimplementedfornow ,butwritethe
functionattheendofthecodeandcommentwhatthefunctiondoes
."
<User >:
CurrentCode:
```python
defsum_common_factors (a:int ,b:int)->int:
""" Computethesumofallcommonprimefactorsof$a$and$b$ """
raiseNotImplementedError ()
```
Let 'sthinkstepbystepandcompletethefollowingPythonfunction`
sum_common_factors `thatsolves:
"Computethesumofallcommonprimefactorsof$a$and$b$"
<Assistant >:
First ,Ineedtogettheprimefactorsof$a$and$b$.
Second ,Icanuse`for `looptofindcommonelementintwofactors
list.
Finally ,sumthecommonfactorlistandreturntheanswer.
Hereisthe`sum_common_factors `function:
```python
defsum_common_factors (a:int ,b:int)->int:
""" Computethesumofallcommonprimefactorsof$a$and$b$ """
factors_a=prime_factor (a)
factors_b=prime_factor (b)
common_factors=get_common(factors_a ,factors_b)
returnsum(common_factors )
defprime_factor(x:int)->list:
""" getalistofprimefactorsofnumber$x$ """
raiseNotImplementedError ()
defget_common(a:list ,b:list)->list:
""" getcommonelementintwolist$a$and$b$ """
raiseNotImplementedError ()
```
```

34 

```
<User >:
CurrentCode:
```python
defsum_common_factors (a:int ,b:int)->int:
""" Computethesumofallcommonprimefactorsof$a$and$b$ """
factors_a=prime_factor (a)
factors_b=prime_factor (b)
common_factors=get_common(factors_a ,factors_b)
returnsum(common_factors )
defget_common(a:list ,b:list)->list:
""" getcommonelementintwolist$a$and$b$ """
raiseNotImplementedError ()
```
Let 'sthinkstepbystepandcompletethefollowingPythonfunction`
get_common `thatsolves:
"getcommonelementintwolist$a$and$b$"
<Assistant >:
Hereisthe`get_common `function:
```python
defget_common(a:list ,b:list)->list:
""" getcommonelementintwolist$a$and$b$ """
ret=[]
foritemina:
ifiteminb:
ret.append(item)
returnret
```
<User >:
CurrentCode:
```python
{prev_code}
```
```

```
Let 'sthinkstepbystepandcompletethefollowingPythonfunction`{
cur_func_name}`thatsolves:
"{ cur_func_doc }"
```

## **C.3 Prompt for Conquer** 

```
Youareaprogrammingcopilot ,youcansolveaproblembywriting
Pythonfunctions.Yourtaskisto:
-Foreveryturn ,youneedtowriteaPythonfunctionthatreturns
theanswer ,basedoncurrentcode(notcodeinchathistory)and
problemdescription.
-Donotmodifyfunctionname ,argnames ,docstringingiven
functions.
-Considerreusingexistingfunctionsthatarealreadyimplemented.
-Youcanimportlibrariestobettersolvetheproblem.
<User >:
CurrentCode:
```python
defprime_factor(x:int)->list:
""" getalistofprimefactorsofnumber$x$ """
```

35 

```
ret=[]
i=1
whilei*i<=x:
i+=1
ifx%i==0andis_prime(i):
ret.append(i)
returnret
defis_prime(x:int)->bool:
""" determine$x$isaprimenumberornot """
ifx<2:
returnFalse
foriinrange(2,int(x**0.5)+1):
ifx%i==0:
returnFalse
returnTrue
defget_common(a:list ,b:list)->list:
""" getcommonelementintwolist$a$and$b$ """
ret=[]
foritemina:
ifiteminb:
ret.append(item)
returnret
defsum_common_factors (a:int ,b:int)->int:
""" Returnthesumofallcommonprimefactorsof$a$and$b$ """
raiseNotImplementedError ()
```
```

```
Let 'sthinkstepbystepandimplementthefollowingmethod`
sum_common_factors `usingexistingfunctionstosolve:
"Returnthesumofallcommonprimefactorsof$a$and$b$"
```

```
<Assistant >:
```

```
First ,Ineedtogettheprimefactorsof$a$and$b$.
Second ,Icanuse`for `looptofindcommonelementintwofactors
list.
Finally ,sumthecommonfactorlistandreturntheanswer.
Hereisthe`sum_common_factors `function:
```python
defsum_common_factors (a:int ,b:int)->int:
""" Computethesumofallcommonprimefactorsof$a$and$b$ """
factors_a=prime_factor (a)
factors_b=prime_factor (b)
common_factors=get_common(factors_a ,factors_b)
returnsum(common_factors )
```
```

```
<User >:
```

```
CurrentCode:
```python
{prev_code}
```
```

```
Let 'sthinkstepbystepandimplementthefollowingmethod`{
cur_func_name}`usingexistingfunctionstosolve:
"{ cur_func_doc }"
```

36 

## **C.4 Prompt for Generate Possible Input** 

```
Youareaproficientsoftwareengineerandarchitect ,specializedin
testing ,capableofobservingandfindingsecurityholesand
logicalissuesincodethatspansacrossmultiplecomputerscience
fieldsandmathematics.YouaregivenapieceofPythonfunction ,
andareexpectedtoproducesomefunctioncallsforthatfunction
.Specifically:
-Youshouldinvokethefunctioninaone -linerfashion.
-Donotbringinimportsotherthanwhat 'salreadyimported.Use
thepre -declaredimportsintheoriginalfunctiononly.
-Thecalleemayhavemultiplearguments ,treatthemwithcare.
-You** must **respectthefunctionsignatureanddocstring ,andbe
awaresoyoudon 'tgenerateillegalinputs.
-Keeptheinputssimplebutgeneral ,andthateitheredgecasesor
commoncasesaremeaningful.
Herefollowsaseriesofmutuallyuncorrelatedfunctionstotest ,one
perconversation.
<User >:
Let 'sthinkstepbystepandcreatesometestsforthefollowing
function`check_valid_brackets (...) `inPython.
```python
defcheck_valid_brackets (seq:str)->bool:
""" Determineifabracketsequenceconsistingof'(',')','{',
'}','['
and']'isvalid."""
mapping={')':'(','}':'{',']':'['}
stack=[]
forcinseq:
ifcinmapping:
ifnotstackorstack [-1]!=mapping[c]:
returnFalse
stack.pop()
else:
stack.append(c)
returnnotstack
```
Storeyourfunctioncallsfor`check_valid_brackets (...) `asfunction
callss ,oneperline.Theywillbecalledlater.
<Assistant >:
Sure ,Icancreatesomefunctioncallsforthe`check_valid_brackets `
function.Wecaneitherchoosetotestitwithavalidbracket
sequenceoraninvalidone.Emptystringsarealsoconsiderable.
Herearesomefunctioncallsforthefunction:
```python
check_valid_brackets ("()")#True
check_valid_brackets ("(([[]]))")#True
check_valid_brackets ("((())")#False
check_valid_brackets ("() []{}")#True
check_valid_brackets ("([)]")#False
check_valid_brackets ("")#True
check_valid_brackets (")(")#False
```
```

```
<User >:
```

37 

```
Let 'sthinkstepbystepandcreatesometestsforthefollowing
function`{cur_func_name }(...) `inPython.
```python
{prev_code}
```
Storeyourfunctioncallsfor`{ cur_func_name }(...) `asfunction
callss ,oneperline.Theywillbecalledlater.
```

## **C.5 Prompt for Self-Test Generation** 

```
Youareaproficientsoftwareengineerandarchitect ,specializedin
testing ,capableofobservingandfindingsecurityholesand
logicalissuesincodethatspansacrossmultiplecomputerscience
fieldsandmathematics.YouaregivenapieceofPythonfunction ,
andareexpectedtoproducesometestcasesforthatfunction.
Specifically:
-Youshouldinvokethefunctionandassertitsresultsinaone -
linerfashion.
-Donotbringinimportsotherthanwhat 'salreadyimported.Use
thepre -declaredimportsintheoriginalfunctiononly.
-Thecalleemayhavemultiplearguments ,treatthemwithcare.
-You** must **respectthefunctionsignatureanddocstring ,andbe
awaresoyoudon 'tgenerateillegalinputs.
-Keeptheinputs&outputssimplebutgeneral ,andthateitheredge
casesorcommoncasesaremeaningful.
Herefollowsaseriesofmutuallyuncorrelatedfunctionstotest ,one
perconversation.
<User >:
Let 'sthinkstepbystepandcreatesometestsforthefollowing
function`lcm (...) `inPython.
```python
deflcm(a:int ,b:int)->int:
""" Findtheleastcommonmultipleof`a`and`b`.Samples:
>>>lcm(3,5)
15
>>>lcm(4,6)
12
"""
returnround(a*b/gcd(a,b))
```
Storeyourtestcasesfor`lcm (...) `asassertions ,oneperline.They
willbecalledlater.
<Assistant >:
Sure ,Icancreatesometestcasesforthe`check_valid_brackets `
function.Weconsiderthefollowingcases:1.thetwooperandsare
notco -primeandhascommonfactors;2.thetwooperandsare
equal;3.oneofthemis1;4.twoofthemis1;5.bothoperands
areprimes.HereisanexampleofthesetestcasesinPython:
```python
assertlcm(15,25)==75
assertlcm(32,32)==32
```

38 

```
assertlcm(1,5)==5
assertlcm(1,1)==1
assertlcm(17,19)==17*19
```
<User >:
Extracttestsforthefollowingfunction`{ cur_func_name }(...) `in
Python.
```python
{prev_code}
```
Storeyourtestcasesfor`{ cur_func_name }(...) `asassertions ,one
perline.Theywillbecalledlater.
```

## **C.6 Prompt for MATH judging** 

```
Youareamathematicalteacher ,yourtaskisto:
-Judgewhetherthepredictionismatchingtheanswer
-Output"Judge:Correct ."or"Judge:Wrong .",pleasedonot
outputredundantwords
-Numericalerrorsshouldbeignored($1$isequalto$0 .99999998$
)
-Someanswermightberepresentinlatexformat ,andsomemight
befloatnumber ,thisshouldbeconsiderascorrect($\frac {1}{2}$
isequalto$0.5$,$3$$\sqrt {66}$isequalto$24 .37211$)
-Unitinanswershouldbeignored ,andshouldbeconsideras
correct($13cm^2$isequalto$13.0$,$\$13$isequalto$13$)
Now ,theanswerandpredictionis:
Answer:{ground_truth}
Prediction:{model_output}
Pleaseoutput"Judge:Correct ."iftwoanswersareliterallythesame ,
or"Judge:Wrong ."fornotsame ,pleasedonotoutputredundant
words.
```

39 

