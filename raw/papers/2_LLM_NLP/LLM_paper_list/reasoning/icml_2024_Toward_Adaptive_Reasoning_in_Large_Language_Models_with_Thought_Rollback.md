**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

**Sijia Chen**[1] **Baochun Li**[1] 

## **Abstract** 

## **1. Introduction** 

Large language models (LLMs) have been routinely used to solve various tasks using step-bystep reasoning. However, the structure of intermediate reasoning steps, or _thoughts_ , is rigid and unidirectional, such as chains, trees, or acyclicdirected graphs. Consequently, the resulting inflexible and forward-only reasoning may not address challenging tasks and fail when the LLM frequently gives false responses, i.e., “hallucinations”. This paper proposes a new reasoning framework, called _Thought Rollback_ (TR), allowing LLMs to adaptively build thought structure while maintaining effective reasoning toward problem-solving under “hallucinations”. The core mechanism of TR is _rolling back thoughts_ , which allows LLMs to perform error analysis on thoughts, and thus roll back to any previously mistaken thought for revision. Subsequently, by including such trial-and-error in the prompt to guide the LLM, each rollback leads to one more reliable reasoning path. Therefore, starting with a simple prompt without human annotations, LLM with TR adaptively and gradually explores thoughts for a correct solution. Comprehensive experiments on mathematical problems and multi-task reasoning demonstrate the state-of-the-art performance of TR in terms of problem-solving rate and interaction cost. For instance, the solving rate of GPT-4 with TR outperforms the current best by 9% on the MATH dataset. The source code is available under the folder _examples/ThoughtRollback_ of https:// github.com/iQua/llmpebase. 

> 1Department of Electrical and Computer Engineering, University of Toronto, Toronto, Ontario, Canada. Correspondence to: Sijia Chen _<_ sjia.chen@mail.utoronto.ca _>_ . 

> _Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

Large Language Models, initially designed for text generation with autoregression, are widely recognized to excel in a diverse array of natural language processing (NLP) tasks. Yet, at a particular model scale, their reasoning abilities, particularly in scaled-up versions like GPT-4 (OpenAI, 2023) and Llama 2 (Touvron et al., 2023), heavily depend on prompt engineering. With well-crafted prompts — even just a simple _Let’s think step by step_ (Kojima et al., 2022) — LLMs are able to perform step-by-step reasoning and achieved noteworthy success in mathematical, symbolic, and common sense tasks. With reasoning, LLMs are capable of producing coherent language sequences, called _thoughts_ , which serve as intermediate reasoning steps toward solving the problem at hand. Extended from simple chain reasoning (Wei et al., 2022) with linear left-to-right thoughts, more complex reasoning became feasible in recent works by establishing thought structures that resembled trees (Yao et al., 2023) and graphs (Besta et al., 2023; Zhang et al., 2023; Luo et al., 2024). 

However, existing thought structures are unidirectional and thus allow a forward-only reasoning process, meaning that thoughts are generated sequentially from the start to the end. The efficacy of this reasoning process hinges on a redundant and, consequently, inefficient thought structure, requiring thorough explorations of each thought before progressing to the next. One major drawback of forward-only reasoning is that errors can propagate quickly (Yu et al., 2024). Consider the common case where one thought is incorrect or inaccurate: with forward-only reasoning, all thoughts derived from it can be misled. Even with revisions based on step-by-step evaluations (Weng et al., 2023), such propagation of errors can introduce further deviations from the correct path of reasoning, since LLMs have been found to confidently provide false information (i.e., “hallucinations” or “laziness”) (Jiang et al., 2023). 

Indeed, humans also provide false information as frequently and randomly as LLMs do during reasoning but can still solve challenging problems eventually. This is attributed to adaptive reasoning, in which one does not pre-defined a fixed structure for thoughts and does not simply deduce forward but adaptively adjusts the thought structure after evaluating errors during reasoning. Such reasoning enables 

1 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Q Question N-1, S-1  error  of Chain thought structure : N-2, S-2  error  of TR structure : N-2, S-2  error analysis  of TR structure :<br>ThoughtSolution The first step is to calculate the expected winnings for eachtype of card drawn. There are four Aces in a deck, so theexpected winnings for drawing an Ace is 4 * $1 = $4.Similarly, there are four of each card from 2 through 10, sothe expected winnings for these cards is 4 * (2 + 3 + 4 + 5 +6 + 7 + 8 + 9 + 10) = $180. There are 12 face cards in a The next step is to calculate the expected winnings for drawinga card from 2 through 10. For each of these nine cards, thereare four of each in the deck (one each of hearts, diamonds,clubs, and spades). .... Sum(2 through 10) = 9/2 * ($3.50 +$17.50) = $94.50. The error lies in the calculation for cards 2 through 10. Thecorrect approach would be to calculate the expected value foreach card from 2 to 10 individually, ..... After calculating theexpected value for each card, we would sum these valuestogether.<br>deck, so the expected winnings for drawing a face card is<br>12 * $20 = $240.<br>N- Node Identity<br>S- Step Index N-3, S-1  error  of Tree thought structure : N-7, S-2  revised thought  of TR<br>structure :<br>Bad Thought There are 4 Aces in a deck, so the probability of drawing an<br>Ace is 4/52 or 1/13.  .... E(Ace) = (1/13 * $1 * 2) + (1/13 * $1 The next step ...  However, remembering the<br>R Forward ReasoningRollback * 3) + (1/13 * $1) + (1/13 * $1) = $0.154 + $0.231 + $0.077+ $0.077 = $0.539 N-6, S-4 Adding up all these expected winnings, we get the totalexpected winnings for drawing any card from the deck: thought of TR structure : error from the previous experience, we need toremember for the cards 2 through 10, not 36 or 1. .... So, thecorrect calculation for the expected value of thecards 2 through 10 should be (4/52) * $2 + (4/52) *$3 + ... + (4/52) * $10.  that there are exactly 9 different values<br>R Thought Inducedby Rollback E(Total) = E(Ace) + Sum(2 through 10) + E(Face cards) =$1.75 + $94.50 + $105 = $201.25.<br>(a). Chain Thought Structure (b). Tree Thought Structure (c).  Thought Rollback (TR) Structure<br>**----- End of picture text -----**<br>


_Figure 1._ Schematic illustrating thought structures for problem solving with GPT-4. The chain, tree, and our thought rollback (TR) structures are plotted based on the NetworkX lib (Hagberg et al., 2008). The question from the MATH dataset (Hendrycks et al., 2021b) is: _I draw a card from a standard 52-card deck. If I draw an Ace, I win 1 dollar. If I draw a 2 through 10, I win a number of dollars equal to the value of the card. If I draw a face card (Jack, Queen, or King), I win 20 dollars. If I draw a ‘clubsuit’, my winnings are doubled, and if I draw a ‘spadesuit’, my winnings are tripled. ... What would be a fair price to pay to play the game?_ In (c), we present a partial thought structure built by GPT-4 with TR and place the full version of the TR structure in Figure 4 of the Appendix. 

humans to begin with one simple or wrong thought but frequently introspect during reasoning, that is, to reconsider previous steps and build new reasoning paths from these reflections. In this paper, we argue that this iterative error correction nature of adaptive reasoning is essentially supported by **rollback** — jumping to previous steps with a new experience to reconsider the reasoning. 

Therefore, we propose a new reasoning framework, Thought Rollback (TR), relying upon the rolling back of thoughts to enable the adaptive reasoning of LLMs for general problem solving. TR embraces a _rollback controller_ and a _prompt enhancer_ that works seamlessly to enable the LLMs to generate an effective thought structure from one thought derived from a simple input prompt, as shown by Figure 1. 

LLMs with TR start with generating one thought from a simple zero-shot prompt containing only the question description. Subsequently, for each generated thought, the _rollback controller_ allows the LLM to analyze the obtained chain of thoughts and thus determines whether to roll back and to which previous thought. Once rollback is triggered, _prompt enhancer_ accumulates this error analysis as experience in the prompt. As a result, by avoiding making similar mistakes mentioned by experience, LLM is able to generate a new and more effective reasoning path from the chosen thought. Therefore, “hallucinations” that occur in thought or analysis of LLM may not influence reasoning due to the continuous thought revision guaranteed by the iterative rollbacks during reasoning. For example, in Figure 1, different from chain (Wei et al., 2022) and tree (Yao et al., 2023) structures, which assume a fixed and unidirectional structure, reasoning with rollbacks enables LLM to build a thought structure adaptively and revise thoughts to achieve complex but reliable reasoning. Specifically, after reaching the _N_ -2 _S_ -2 and _N_ -6 _S_ -4, LLM finds an error in 2-th step 

_N_ -2 _S_ -2 and thus rolls back to 1-th step _N_ -1 _S_ -1 to create two new reasoning paths. The rollback _N_ -6 _→ N_ -1 leads to the revised thought _N_ -7 _S_ -2. The rollback _N_ -2 _→ N_ -1 utilizes the error analysis to enhance the prompt and obtains _N_ -3 _S_ -2, leading to a correct answer 15 _._ 48 compared to the previous mistaken 3 _._ 87. 

We observe four contributions of TR. First, it is a lightweight and generalized automatic prompting framework. TR allows LLMs to perform complex reasoning effectively on various tasks without introducing task-specific human annotations in the prompt or additional human-made designs. Second, the performance of TR is robust to the “hallucinations” as LLMs are able to reconsider and revise any existing thoughts adaptively and repeatedly during reasoning. Thus, third, TR is cost-friendly as the thought structure is built progressively to reach a solution instead of relying on bulky search structures (Yao et al., 2023) or question analogies (Yu et al., 2024). Finally, our evaluation of TR on mathematical problems and multi-task reasoning demonstrates that TR outperforms some state-of-the-art approaches while maintaining a lower interaction cost. 

## **2. Related Work** 

By only guiding the reasoning behavior of LLMs, such as GPT-4 (OpenAI, 2023) and Llama2 (Touvron et al., 2023) with the text prompt, prompt engineering (Brown et al., 2020; Kojima et al., 2022) is parameter efficient and often matches or surpasses fine-tuning performance. Therefore, plenty of work has been proposed to enable LLMs to perform **multi-step reasoning** containing a series of intermediate steps, each known as the thought presented as a text sequence. Starting from chain-of-thought (CoT) (Wei et al., 2022) prompting, which provides reasoning examples in the prompt to deduce a chain of thoughts, subsequent endeavors, 

2 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

especially SC (Wang et al., 2022) and Complex CoT (Fu et al., 2023) augment the chain reasoning. Recent advances extend the chain structure into structured thoughts. ToT (Yao et al., 2023) and BoT (Sijia et al., 2024) pre-defines the thought structure as a tree, thus supporting exploring multiple thoughts in each step before generating the next. The graph of thoughts (GoT) (Besta et al., 2023) and cumulative reasoning (CR) (Zhang et al., 2023) further instantiate thoughts toward a solution as the graph structure. Another line of work focuses on the thought structure that allows the thought ensemble in each step. Thought propagation (Yu et al., 2024) explores analogous problems and then aggregates their results to update the solution for the given question, leading to a radial thought structure. To the best of our knowledge, none of the existing work supports the cyclic structure to allow LLMs to revise previous thoughts or recreate a new reasoning path from the previous step after being blocked at the current reasoning step. We fill this gap by proposing the rollback of thought, leading to a thought structure of directed cyclic graphs. 

Despite these achievements, LLMs often struggle with complex tasks, primarily due to the frequent occurrence of “hallucinations”—producing false outputs (Jiang et al., 2023; Yu et al., 2024), and “laziness”—yielding invalid or no output. Therefore, after noticing that LLMs have **self-verification** (Weng et al., 2023; Madaan et al., 2023) abilities and thus can analyze the answer for further correcting the errors in reasoning (Zheng et al., 2023; Wu et al., 2024). However, the most recent work (Huang et al., 2024) argues that LLMs cannot self-correct their reasoning, emphasizing the invalidity of applying simple verification to the reasoning path. Thus, most recent work either builds iterative-based verification (BoT (Sijia et al., 2024)) or focuses on **step-bystep verification** (Ling et al., 2023; Lightman et al., 2024). Combining these insights, we aim to exploit LLMs to analyze intermediate thoughts during reasoning to correct these thoughts and adjust reasoning direction adaptively. Continuous verification and revision may eliminate the negative impact of “hallucinations” or mistakes on the solutions. 

Another related research stream is **automatic prompting** (Kojima et al., 2022; Zhang et al., 2022), which automatically constructs effective prompts to facilitate reasoning without human-made and task-specific demonstrations. As LLMs can learn from mistakes to become better reasoners (An et al., 2023; Sijia et al., 2024), this paper also releases human efforts from the prompt design by boosting the prompt with the error analysis of thoughts. We also show that by accumulating error analysis in the prompt during reasoning, LLMs are able to avoid making similar mistakes and explore correct solutions with interaction cost far less than ToT (Yao et al., 2023) and BoT (Sijia et al., 2024). 

## **3. Preliminary** 

## **3.1. Problem Statement** 

Given a pre-trained large language model (LLM) denoted as _f_ ( _·_ ), the prompt engineering is to design the prompt I ( _·_ ) to make the model perform desired reasoning behavior toward addressing the given problem _x_ . Specifically, multi-step reasoning contains _T_ intermediate steps _z_ 0 _...T_ = [ _z_ 0 _, z_ 1 _, ..., zT_ ] to bridge _z_ 0 := _x_ and the answer _zT_ := _y_ . To get _z_ 0 _...T_ , we focus on step-wise thought generation in which each thought is a coherent language sequence _zn_ , behaving as one intermediate reasoning step, and _zn_ is generated as _zn ∼ f_ ( _zn|_ I ( _z_ 0 _,_ 1 _...n−_ 1)). Therefore, as thought is the LLM’s output, we can define the bad thought caused by the “hallucinations”, “laziness”, or false reasoning of LLMs as � _zn_ . 

These generated thoughts _z_ 0 _...T_ naturally follow a specific structure, such as a chain or tree. These structures are unidirectional and thus only support **forward-only reasoning** , which proceeds in a linear, sequential manner, meaning that LLMs only generate the subsequent thought _zn_ +1 from _zn_ . For instance, any edge _en,m_ of the structure is limited to _m_ = _{n, n_ + 1 _}_ while _m >_ = _n_ . 

This paper focuses on alleviating the effect of bad thoughts on the solution by making LLM not simply perform forwardonly reasoning but achieve _adaptive reasoning_ , which allows LLMs to 1) start from a simple prompt I ( _z_ 0), 2). selforganize the thought structure adaptively during reasoning, and thus 3). when � _zn_ occurs, LLM can make revisions and create better new reasoning paths till getting the solution. Specifically, not only advancing the reasoning sequentially, any previously generated thought will be reconsidered by continuously rolling back from _n_ -th thought to one previous _m_ -th thought, where _m ∈_ [0 _, n −_ 1]. 

## **3.2. Motivation: Forward-only reasoning fails in bad thoughts** 

Forward-only reasoning may fail as the bad thought _z_ � _n_ is caused by the following three cases of **error propagation** . 

**Case** [ _z_ � _m,_ � _zm_ +1 _, ...,_ � _zn_ ]. A bad or illogical thought � _zm_ leads � � to all subsequent errors, where � _zn ∼ f_ ( _zn|_ I ( _z_ 0 _,...m...n−_ 1)) and _m < n_ . 

**Case** [ _z_ � _m, zm_ +1 _...,_ � _zn_ ]. _z_ � _m_ does not lead to direct mistakes but causes a bad thought � _zn_ after many steps. For instance, this appears when � _zm_ behaves as one part of a solution. 

**Case** [ _zm...n−_ 1 _,_ � _zn_ ]. A bad thought � _zn_ may arise from one previous correct though _zm_ because the wrong reasoning direction appears from this step. 

The chain reasoning, typically in Chain-of-thought (CoT) prompting, generates a chain of thoughts _z_ 0 _,_ 1 _...T_ (Wei et al., 

3 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

2022) sequentially; thus, when any cases appear, it gets a mistaken answer _y_ . Subsequent Complex CoT (Fu et al., 2023) provides well-designed examples in _z_ 0 to decrease the frequency of bad thoughts. And SC (Wang et al., 2022) performs majority voting for many generated chains. They are still trapped in the error propagation of chain structure, as shown in Figure 1 (a). 

The tree reasoning, such as Tree of Thoughts (ToT) (Yao et al., 2023), Graph of Thoughts (GoT) (Besta et al., 2023), and Thought Propagation (TP) (Yu et al., 2024), extends the chain thought structure by generating multiple thoughts in each step. For instance, ToT or GoT contains _P n_ -th thought, representing as _zn_[(1)] _[, ..., z] n_[(] _[P]_[ )] _∼_ � � _f_ ( _zn|_ I ( _z_ 0 _,_ 1 _...n−_ 1)). These approaches hope to explore more thoughts to increase the possibility of getting a correct thought. However, error propagation still exists, and any cases that appear in one reasoning path will inevitably cause the failure, as shown in Figure 1 (b). 

Therefore, we argue that to address the error propagation, the reasoning should continuously reconsider the source of error, that is, to find and fix the previous _m_ -th thought after reaching _n_ -th thought, where _m < n_ . This is equivalent to _human-like reasoning, in which one may not simply deduce forward but look back to check previous thoughts to decide to continue, revise old thoughts, or create a new reasoning path_ . Without losing generality, we refer to such reasoning behavior as the **rollback** . In the context of reasoning, to enable such a rollback _n → m_ , we allow LLM to adaptively build the edge _en,m_ with _m < n_ during reasoning, making the thought structure a directed graph with cycles. 

## **3.3. Motivation: Error analysis induces better thoughts** 

The rollback mechanism is insufficient to support adaptive reasoning in LLMs. Without introducing more information to prompt LLMs after each rollback, LLMs may repeat similar mistakes in the thought generation, which more rollbacks in reasoning cannot solve. Thus, it is essential to enable LLMs to know why rollback is triggered and how to avoid producing the � _zm_ +1. 

Motivated by the effectiveness of enhancing prompts by including an error report (Sijia et al., 2024), we propose that analysis of [ _z_ 1 _, ..., zn_ ] could be rolled back to _zm_ to guide the thought generation. Also, the work (Ling et al., 2023) pointed out that LLMs perform more reliable reasoning in CoT when using step-by-step verification. We aim to allow LLMs to perform rollback-by-rollback verification. First, this can produce more analysis to facilitate the subsequent reasoning. Second, invalid or mistake rollbacks can be removed, thus also eliminating the cycles in the thought structure. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0004-07.png)


**----- Start of picture text -----**<br>
R<br>LLM<br>LLM<br>LLM<br>R-1 Rollback-1<br>LLM<br>**----- End of picture text -----**<br>


_Figure 2._ Schematic illustrating the rollback of thought when the LLM with TR reaches the _n_ -th reasoning step. We add _A[χ] z_ 0[(] _...m[·]_[)] _−_ 1 in the reasoning from _zm−_ 1 to _zm_ to cover the case that _zm−_ 1 may also derive from a rollback. We present a clear example from SVAMP in Figure 5 of the Appendix. 

## **4. Thought Rollback Framework** 

## **4.1. Reasoning Overview** 

In contrast to existing approaches relying on pre-defined unidirectional thought structures, which are limited to forwardonly reasoning, Thought Rollback (TR) generates a bidirectional thought structure by adaptively deducing forward and rolling back of thoughts. After reaching a reasoning step _zn_ , as shown in Figure 2, TR allows LLM to roll back to the bad thought _zm_ after analyzing the existing thought chain _z_ 0 _...n_ . As the error analysis _A[n] m_[is to be accumulated in the prompt,] a new and more reliable reasoning path _zm[n]_[is][generated] from _zm_ . Therefore, iteratively performing this rollback develops the thought structure from a simple thought to a directed graph with cycles. Such adaptive reasoning is summarized into three stages. 

**Initialization** . Generate thought _z_ 1 _∼ f_ ( _z_ 1 _|_ I ( _z_ 0)). 

**Rollback of thoughts** . For each generated thought _zn_ , _rollback controller_ exploits LLM to determine a rollback to one thought _zm ∈ z_ 0 _...n_ . 

- Once the rollback to _m ∈_ [0 _,_ 1 _, ..., n_ ] is triggered, the reasoning of LLM rolls back to the thought _zm−_ 1 and _prompt enhancer_ is used to enhance the prompt. Subsequently, the reasoning continues by creating a new _m_ - th thought _zm[n]_[and generating] _[ z][n]_[+1][, where] _[ z] m[n]_[means] a _m_ -th thought deduced from a rollback from _n_ . 

- When no rollback is required, generate _zn_ +1. 

**Early stopping** . Stop reasoning when TR yields _K_ number of solutions obtained. Otherwise, continue the Rollback of thoughts. 

**Solution ensemble** . Perform weighted majority voting on _K_ solutions. 

4 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

## **4.2. Rolling Back Thought with Reasoning Analysis** 

During reasoning, _rollback controller_ enables an adaptive rollback by exploiting LLM to determine the rollback _n → m_ . However, making LLM know the concept of rollback may introduce unnecessary complexity. Thus, TR supports the rollback mechanism by performing error analysis on thoughts. Specifically, with a task-agnostic prompt I _R_ ( _R,_ [ _z_ 0 _...n_ ]), where _R_ is a common error analysis instruction, LLM is guided to analyze a thought chain [ _z_ 0 _...n_ ], leading to the error analysis _A[n] m[∼][f]_[ (] _[A][|]_[I] _[R]_[(] _[R,]_[ [] _[z]_[0] _[...n]_[]))][.] Eventually, LLM is able to identify the indexes _M_[�] of bad thoughts � _zm∈M_ �[.] 

To get which thought to roll back to from _zn_ , TR follows the rule to roll back to the one step before the first bad thought. There are two reasons for this. First, generating the next thoughts from a bad thought is unreasonable. Second, we aim not to remove the bad thought but to generate a new reasoning path. Thus, we choose _zm−_ 1 as the rollback destination, where _m_ = arg min _M_[�] . Besides, one thought will not be selected as the rollback destination more than _U_ times to avoid all subsequent thoughts rolling back to the same previous thought. Thus, when the number of rollbacks to a thought reaches _U_ , the next earliest one of _m_ = arg min _M_[�] _\_ arg min _M_[�] will be selected. � � 

Therefore, when _M_[�] is not empty, TR generates the next thought _zm[n]_[from] _[m][−]_[1][,][meaning][that][a][new][reasoning] path [ _z_ 0 _...zm−_ 1 _, zm[n]_[]][ derived from the rollback] _[ n][→][m]_[ is] created for the thought structure. As the existing _z_ 0 _...n_ remains unchanged, the reasoning continues by generating _z_ 0 _...n_ +1. For ease of description, we define _n → m_ as the _outgoing rollback_ for _z_ 0 _...n_ +1 and the _incoming rollback_ for the new reasoning path [ _z_ 0 _...zm−_ 1 _, zm[n]_[]][.] 

## **4.3. Enhancing the Prompt with Errors as Experience** 

Through iterative rollback from the _n_ -th to the _m −_ 1-th step, TR gains the opportunity to address the three scenarios outlined in subsection 3.2. However, as discussed in 3.3, regenerating a next thought _zm[n]_[based on the same prompt] may repeat existing mistakes in the new reasoning path. Especially considering that TR is built upon the prompt containing no human annotations, the thought regeneration after the rollback is equivalent to randomly exploring _zm[n]_[as] in unidirectional structures. 

Therefore, TR embraces _prompt enhancer_ to also roll back the error analysis _A[n] m_[to][the] _[m][−]_[1][-th][thought.][Unlike] BoT (Sijia et al., 2024) with outcome analysis, which utilizes error feedback on the final result, TR performs process analysis, i.e., rollback-by-rollback verification, to get error reports on intermediate thoughts, guiding the subsequent thought generation. With error analysis, each rollback is 

regarded as a trial on generating subsequent thoughts for _zm−_ 1 because the analysis contains a trial experience: _what mistakes may appear in the following steps of zm−_ 1. By including _A[n] m_[as an experience in the prompt,][LLMs can] avoid making existing bad thoughts after learning from mistakes. Eventually, each rollback _n → m_ creates the error experience _A[n] m_[.] 

**Experience accumulation** . The thoughts **Z** _[χ] z_ 0[(] _...q[·]_[)] _−_ 1 = � _zj[i][|][z] j[i][∈][z]_[0] _[...q][−]_[1] _[, z][i][∈][/][z]_[0] _[...q][−]_[1] _[, j][∈]_[[0] _[, q][ −]_[1]] _[ , i][ ∈][χ]_[ (] _[j]_[)] � of a reasoning path _z_ 0 _...q−_ 1 may derive from multiple _incoming rollbacks_ , where _χ_ ( _j_ ) is the set of rollbacks whose destination is _j_ -th thought of this path. As each rollback creates an error experience from one trial of the given question, _incoming rollbacks_ lead to a series of experiences _A[χ] z_ 0[(] _...q[·]_[)] _−_ 1[.][By][accumulating][an][ensemble][of] trial-and-error reasoning experiences as the in-context learning examples in the prompt, LLM will learn from more experiences to generate the correct next thought _zq ∼ f zq|_ I _A[χ] z_ 0[(] _...q[·]_[)] _−_ 1 _[, z]_ 0 _...q−_ 1 , as shown in Figure 2 � � �� and two examples of Figure 8 and Figure 10. 

## **4.4. Ensembling Solutions** 

TR may create massive final solutions as each adaptive triggered rollback leads to one more new reasoning path toward answering the question. Thus, we directly stop reasoning when there is _K_ number of solutions _{z_ 0 _...Tk }[K] k_ =1 obtained. Then, weighted majority voting (W-Voting) will be performed on them for a final solution. Specifically, for the solution _zTk_ , the weight _wt_ is higher when 1) it has a lower number of _outgoing rollbacks_ denoted as _αTk_ , meaning that fewer bad thoughts are identified; and 2) more experiences _βTk_ = _|A[χ] z_ 0[(] _...Tk[·]_[)] _[|]_[are][accumulated][along][this] reasoning path. Eventually, TR outputs the final solution as: arg max _v∈V_ � _Kk_ =1 _[I]_[ (] _[v][k]_[=] _[ v]_[) (] _[β][T] k[−][α][T] k_[)][,][where] _[V]_ is the collection of solutions and _vk_ is the value of _k_ -th solution. 

## **5. Experiments** 

**Datasets** . We conduct experiments on two streams of tasks. For the mathematical problems, we evaluate the performance of TR on test sets of GSM8K1319 (Cobbe et al., 2021), SVAMP300 (Patel et al., 2021), AQUA-RAT254 (Ling et al., 2017), MATH900 (Hendrycks et al., 2021b), TheoremQA400 (Chen et al., 2023b) datasets, where numerical subscripts indicate sample size. For TheoremQA400, we specifically use half of the test set without visual information, leading to 400 samples. Following ToT (Yao et al., 2023), we utilize 100 challenging games of Game of 24. For multi-task reasoning, such as symbolic reasoning, we extract 900 samples from 56 categories of MMLU (Hendrycks et al., 2021a), i.e. MMLU900. 

5 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

_Table 1._ Evaluating the reasoning ability of TR with GPT models under four well-known mathematical problems. TR is specifically evaluated against leading methods such as Faithful-CoT (Lyu et al., 2023), and CSV (Zhou et al., 2024), each achieving state-of-the-art (SOTA) performance on the SVAMP and MATH datasets, respectively. The best results, apart from the SOTA, are in bold. The 5 shots of CoT examples used by our TR experiments are extracted randomly from the trainset of the same category. 

|Methods<br>ZeroShot|GPT-4<br>GPT-3.5-turbo|
|---|---|
||GSM8K<br>SVAMP<br>AQuA-RAT<br>MATH<br>GSM8K<br>SVAMP<br>AQuA-RAT<br>MATH|
|SOTA<br>✗|98.78<br>95.4<br>85.048<br>84.32<br>89.2<br>84.3<br>60.68<br>40.56|
|ZeroShot<br>✓<br>ZeroShot-CoT<br>✓<br>CoT<br>✗<br>C-CoT<br>✗<br>PHP+C-CoT<br>✗<br>BoT<br>✓<br>BoT+CoT<br>✗|87.1<br>79.33_†_<br>50.4<br>42.2<br>76.3<br>74.8<br>53.5<br>24.5_†_<br>93.1_†_<br>84.67_†_<br>73.2<br>44.7<br>79.6<br>77.5<br>53.9<br>30_†_<br>94.25<br>_†_<br>91.95<br>75.28<br>48.98<br>87.45_,sc_15<br>835<br>59.45<br>-<br>94.98<br>90.58<br>77.58<br>50.48<br>82.88<br>81.08<br>57.48<br>34.18<br>95.58<br>91.98<br>79.98<br>53.98<br>85.18<br>83.18<br>60.68<br>36.58<br>97.1<br>92.67<br>81.5<br>62.44<br>-<br>-<br>-<br>**40.56**<br>**98.7**8<br>**95**8<br>85.048<br>66.338<br>-<br>-<br>-<br>-|
|Chain Reasoning_†_<br>✓<br>ToT Reasoning_†_<br>✓|89.76<br>80.33<br>74.41<br>45.44<br>76.72<br>71.67<br>47.64<br>26.89<br>90.9<br>84<br>76.38<br>48<br>79.83<br>78.33<br>54.72<br>30.44|
|TR_†_<br>✓<br>TR + CoT5<br>_†_<br>✗<br>TR + W-Voting_†_<br>✓<br>TR + CoT5 + W-Voting_†_<br>✗|94.24<br>89<br>79.92<br>55<br>82.49<br>77.67<br>56.69<br>32.78<br>96.06<br>91.33<br>84.25<br>62.56<br>86.5<br>79.67<br>57.87<br>31.44<br>96.36<br>93<br>**87.8**<br>71.89<br>85.9<br>82.33<br>**63.39**<br>39.78<br>96.97<br>93.33<br>87.4<br>**72.11**<br>87.79<br>82.67<br>62.6<br>35.89|



**Large language models** . We utilize GPT-3.5-turbo (gpt3.5-turbo-16k-0613), GPT-4 (gpt-4-1106-preview) (OpenAI, 2023) and Llama2 (Touvron et al., 2023), including Llama213b (Llama-2-13b-chat-hf) and Llama2-70b (Llama-2-70bchat-hf) where 1b means one billion parameters. For LLMs with TR, the default settings for temperature and top ~~p~~ are 0 _._ 7. 

**Baselines** . Apart from zero-shot prompting (Kojima et al., 2022), the comparison approaches include Chain-of-thought (CoT) (Wei et al., 2022), SC (Wang et al., 2022) and Complex CoT (Fu et al., 2023) (C-CoT), where the subscript 5 or 8 indicates the number of shots while the subscript _sc_ denotes the number of sample paths. Also, TR is compared with the related approaches, such as Boosting of thoughts (BoT) (Sijia et al., 2024), Tree of thoughts (ToT) (Yao et al., 2023), Cumulative Reasoning (CR) (Zhang et al., 2023), and Progressive-Hint Prompting (PHP) (Zheng et al., 2023). ToT follows the best first search (BFS). The breadth limit of ToT is 6 while BoT performs 10 boosting iterations on 15 binary trees. We also include the state-of-the-art (SOTA) methods, such as CSV (Zhou et al., 2024) that relies on GPT-4 Code Interpreter, on each dataset as an additional comparison. We set _K_ = 8 for possible early stopping of TR in all experiments. 

**Metrics** . All experiments report the Solve Rate (%) of the questions. We make LLM explicitly report the solution value after the strings, such as “The solution is” and “The choice is” in the _z_ 0 _...T_ . Thus, the value is directly extracted and compared with the ground truth. The Interaction Number refers to the frequency at which we must consult the LLM until we receive conclusive responses. 

**Reproducibility** . The results and methods marked with a superscript _[†]_ are the results we obtained based on the open source platform _llmpebase_ . Others without such a tag are collected from existing work, as shown in the Appendix’s subsection C.1. 

## **5.1. Main Evaluation Results** 

**Adaptive reasoning** . With zero-shot prompting and no pre-defined thought structures, such as chain of Chain Reasoning and the tree of ToT Reasoning, TR allows GPT3.5-turbo, GPT-4, and Llama2 to self-organize and explore thought structures toward answering the question. Under challenging tasks, LLMs with TR adaptively build complex structures, as shown by examples in the Appendix, by continuously rolling back from thoughts with “hallucinations”. For simpler tasks, lightweight structures are built by LLMs with TR. As such, with the ability to adjust thoughts and prompt the LLMs with accumulated experience of errors during reasoning, TR achieves a high solving rate and relatively lower resource cost. 

**Overall comparison** . We show, especially in Table 1, that compared to existing multi-step reasoning approaches, LLMs with TR achieve the best and the second best solving rate on AQuA-RAT &MATH and GSM8K & SVAMP , respectively. Meanwhile, contrary to the resource-cost SOTA ones, such as BoT, which undertakes reasoning through massive tree thought structures, and CSV, which relies on GPT-4 Code Interpreter, TR yields notable performance by interacting less with relatively simpler LLMs. First, TR surpasses BoT by 6 _._ 3 on AQuA-RAT and 9 _._ 45 on MATH using GPT-4. In particular, TR requires only around 40 in- 

6 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

teractions compared to the 500+ interactions of BoT. Using TR with zero-shot prompting, GPT-4 and GPT-3.5-turbo outperform the ones using few-shot CoT prompting and self-consistency. Under hard math problems, especially MATH, the solving rate of TR is 17 _._ 99 and 3 _._ 28 higher than PHP+C-CoT under GPT-4 and GPT-3.5-turbo, respectively. Second, LLMs with TR adaptively explore thought structures, which is significantly better than pre-defined forward-only Chain Reasoning and ToT Reasoning. After large-scale interactions with LLMs, the performance of the latter two zero-shot prompting methods only approaches the 8 shots Complex CoT. Figure 3 (a) shows that compared to ToT reasoning, our TR requires one-third or less of the interactions to achieve a new state-of-the-art. Finally, with an average 28 interactions with GPT-4, TR yields a competitive solving rate 87 _._ 56 on the multi-task dataset MMLU, which contains symbolic reasoning. 

We emphasize that TR is more effective in challenging problems, as shown in Table 2 and Table 3. In the level 5 difficulty of MATH, GPT-4 with TR is only 2 _._ 84 lower than the CSV that embraces the GPT-4 Code Interpreter as an auxiliary. The solving rate of GPT-4+TR is 4 _._ 35 higher than the current best in TheoremQA. Along with better performance, interaction cost is reduced to an acceptable range by TR. Another observation is that LLMs with TR introduce more interactions in hard problems than the simpler ones. For instance, as shown in Figure 3 (a) and the first two columns of Table 2 , the average interaction cost increases to around 60. Also, in Table 3, the results of the Game of 24 dataset show that GPT-4 with TR requires an average of 32 interactions to reach a solving rate of 87%, which is only 7% lower than the CR2 (Zhang et al., 2023). This is a remarkable achievement as CR-related approaches rely on human-made demonstrations while TR is zero-shot prompting. Moreover, introducing the CR’s demonstrations into the prompt of TR increases the solving rate to 93% while reducing the number of interactions to 24. In addition, the better performance of TR + CR-Prompt shows that including demonstrations reduces the reliance on majority voting. 

**Effect of the rollback of thoughts** . In Figure 3 (b) and (c), we specifically present the relation between rollbacks and the solving rate of reasoning paths and the decrease in the failure rates at the first step of the Game of 24. We define a reasoning path _z_ 0 _...T_ as _In Rollback_ if a majority of its thoughts, represented by **Z** _[χ] z_ 0[(] _...T[·]_[)][in the subsection][ 4.3][, are] derived from _incoming rollbacks_ . _z_ 0 _...T_ is defined as _Out Rollback_ if more than two of its thoughts trigger _outgoing rollbacks_ and as _No Rollback_ if it includes no rollbacks. Figure 3 (b) presents these three types of reasoning paths TR generates during reasoning in four datasets. As TR allows the error analysis of each rollback to be accumulated in the prompt, as discussed in subsection 4.3, a _In Rollback_ 

_Table 2._ Evaluating TR with the ZeroShot setting on challenging mathematical problems and multi-task reasoning. With GPT-4, the existing SOTA zero-shot methods on the level 5 difficulty of the MATH, TheoremQA, and MMLU are from CSV (Zhou et al., 2024), PoT (Chen et al., 2023a), and BoT (Sijia et al., 2024), respectively. We use 324 samples out of 1324 for the MATH dataset. The average interaction number with LLM to solve each problem is reported within () behind the number. 

|Methods|MATH-level5324|TheoremQA400|MMLU900|
|---|---|---|---|
|SOTA|55(3)|52.4(1)|93.2(900+)|
|Llama2-13b_†_|2.47(1)|9.3|51.78(1)|
|Llama2-70b_†_|8.64(1)|25.5|65.44(1)|
|GPT-4 + ZeroShot-CoT_†_|23.46(1)|43.75(1)|82.33(1)|
|GPT-4 + Chain Reasoning_†_|22.53(11)|36.5(9)|78.11(6)|
|GPT-4 + ToT Reasoning_†_|24.38(150)|38.25(110)|79.22(70)|
|Llama2-70b + TR + W-Voting _†_|12.65(36)|29(34)|58.33(22)|
|GPT-3.5-turbo + TR + W-Voting_†_|20.06(38)|39.5(30)|70.11(18)|
|GPT-4 + TR_†_|31.48|46.25|84.67|
|GPT-4 + TR + W-Voting_†_|52.16(62)|**56.75**(56)|87.56(28)|



_Table 3._ Utilizing TR with GPT-4 achieves competitive performance while maintaining low interaction cost on Game of 24 dataset. 

|dataset.|||||
|---|---|---|---|---|
|Method|Solving rate|#Interactions|Generate tokens|Prompt tokens|
|Standard|7.3|1|-|-|
|Standard (best of 100)|33|100|1.8k|1k|
|CoT|4|1|-|-|
|CoT (best of 100)|49|100|6.7k|2.2k|
|CoT-SC|9 _sc_100|100|-|-|
|ToT(b = 5)|74|30|5.5k|1.4k|
|CR(b = 2)|94|27.4|-|-|
|CR(b = 5)|**98**|29.72|-|-|
|BoT|83.7|724|15.8k|18.6k|
|BoT+CoT5|84.9|543|11.2k|15.5k|
|Chain Reasoning_†_|5|3|0.14k|0.22k|
|ToT(b = 5) Reasoning_†_|25.6|36|2.82k|1.63k|
|TR _†_<br>TR + W-Voting_†_|70<br>87|32|5.96k|9.98k|
|TR + CR-Prompt _†_<br>TR + W-Voting + CR-Prompt_†_|86<br>93|24|5.03k|8.1k|



path generally benefits from exploiting more experiences during reasoning. Therefore, as Figure 3 (b) verifies, _In Rollback_ paths consistently correspond to a higher solving rate because of prompting LLMs with these trial-and-error experiences. On the contrary, those _Out Rollback_ paths have significantly low solving rates because they include more bad thoughts (“hallucinations”), which consequently trigger more rollbacks after being identified by LLMs. Similarly, when the first step of Game of 24 derives from the number of 0 to 5 _incoming rollbacks_ , the failing rate decreases significantly from higher than 0 _._ 8 to lower than 0 _._ 3. The final observation is that longer spans of rolling back are more important for revising thoughts. Figure 3 (c) shows that the first step caused by a rollback 3 _→_ 0 has a higher success rate than the one caused by 2 _→_ 0 and 1 _→_ 0. Therefore, we argue that **a rollback, especially generated from more latter reasoning steps (thoughts), contribute more to the thought revision** . This may be because the error analysis brought by rollbacks of later reasoning steps contains more information and is more helpful in improving prompts. 

7 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0008-01.png)


**----- Start of picture text -----**<br>
15014013012011010090807060 TRChain ReasoningToT ReasoninPHP+C-CoT Interaction Cost Distribution g 550500450400350300250 Relation between Rollbacks and Reasoning In Rollback Out RollbackNo Rollback In Rollback Solving Rate Out Rollback Solving RateNo Rollback Solving Rate217 266 499547 1.00.90.80.70.60.50.4 0.90.80.70.60.50.4 ThouToT Tree Reasoning ghts failed at the first steNumber(#) of In Rollbacks In Rollbacks p<br>5040302010 20015010050 88 104 129 64 177 153161 83 0.30.20.10.0 0.30.20.1<br>0 0 0.0<br>GSM8K SVAMP AQuA-RAT MATH GSM8K SVAMP AQuA-RAT MATH Baseline 0 1 3 5 7 1->0 2->0 3->0<br>(a) (b) (c)<br>Error Rate<br>Solving Rate<br>Number of Interaction<br>Number of Reasoning Path<br>**----- End of picture text -----**<br>


_Figure 3._ Effectiveness of TR on interaction cost saving and thought revision through the rolling back of thoughts. a). distributions of the interactions required to address problems in four datasets; b). solving rates of three types of reasoning path: “No Rollback” — thoughts receive no rollbacks, “Out Rollback” — rollbacks triggered by mistaken thoughts, and “In Rollback” — thoughts derive from rollbacks; c). the reduction of failure rates due to the rollbacks in the first step of Game of 24, where [0 _,_ 1 _,_ 3 _,_ 5 _,_ 7] denotes the number of rollbacks that cause the first step and 2 _→_ 0 means the first step derives from the rollback from the 2-th step. 

**Concerns** . First, TR contributes less to the performance enhancement when the LLMs inherently do not have solid ability. Especially in AQuA-RAT and MATH of Table 1, GPT-3.5-turbo with TR is 2 _._ 91 and 3 _._ 25 higher than PHP+CCoT, which are significantly lower than these under GPT-4. Likewise, in Table 2, the solving rate of Llama2-70b with TR is around 4% higher than Llama2-70b but costs more interactions. Therefore, the performance of TR depends on LLMs’ abilities to perform correct error analysis and understand the experience in the prompt. Second, to address harder problems, LLMs with TR tend to build over-complex thought structures, which generally contain more than 100 thoughts. The main reason is that a rollback generated by bad thoughts identified by LLMs or a mistaken rollback caused by hallucinations of LLMs leads to one more reasoning path. This appears frequently in challenging tasks; thus, LLMs self-organize a large-scale thought structure toward solutions. We present more visible examples in Figure 4, Figure 9, and Figure 10 of the Appendix. 

## **5.2. Main Insights** 

The notable performance enhancement of TR in terms of both solving rate and interaction cost shows the insight that adaptive adjusting thoughts supported by the rollback of thoughts during reasoning is core to the success of LLMs in complex mathematical reasoning. In addition, we can gain three more insights. 

**Experience accumulation of error analysis from intermediate thoughts is better than that obtained by analyzing the whole reasoning path** . Existing work (Huang et al., 2024) pointed out that LLMs are unable to revise reasoning based on the outcome analysis, which gives feedback on the final reasoning. Thus, BoT (Sijia et al., 2024), which relies on outcome analysis, had to embrace more careful-selected outcome analysis to prompt LLMs. Our TR opens a new direction of relying on process analysis, which provides error 

analysis for each intermediate reasoning step (rollback-byrollback verification), to revise thought adaptively during reasoning. Besides, with the rollback of thought, outcome analysis is a special case of process analysis when analysis is used not to re-do reasoning but to adjust previous thought to create a correct reasoning path. 

**Experience-guided Solution Ensemble is critical to the effectiveness of trial-and-error reasoning** . After stopping reasoning, LLMs with TR yield _K_ reasoning paths due to the adaptive exploration. Each reasoning path caused by one rollback of TR can be regarded as a trial for addressing the problem. When LLMs frequently make mistakes and occur “hallucinations”, the solution obtained in any trial may not be correct. Since TR exploits the error analysis of each _incoming rollback_ as experience to prompt LLMs, a solution from the reasoning path with more _incoming rollback_ is more acceptable. Therefore, we should ensemble these solutions by filtering out ones with limited experiences or many bad thoughts. As shown by comparisons between TR and TR + W-Voting in Table 1, Table 2, Table 3 and Figure 3 (b), such an experience-guided solution ensemble is critical. 

**Weak LLMs may not identify multiple targets mentioned in the prompt** . Including CoT examples in the prompt improves the solving rate of LLMs with TR, as shown in the GPT-4 column of Table 1. However, for the more challenging AQUA-RAT and MATH datasets in the GPT-3.5-turbo column, adding CoT causes a significant performance decrease. We argue that it may be hard for the weak LLMs to understand and distinguish instructions with different targets in the prompt. For example, the instruction of CoT examples emphasizes how to follow demonstrations, while the prompt with experiences in TR focuses on how to avoid given errors. Weak LLMs may not benefit from the enhanced prompt containing these two different guidances, especially when the reasoning is complex. 

8 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

_Table 4._ Evaluating the token cost when using GPT-4 with TR to address questions from MATH and TheoremQA. The ratios in the table indicate how many times the token costs of various approaches exceed the token cost of ZeroShot CoT. _△_ represents the increase in the problem-solving rate of various approaches compared to ZeroShot CoT. We also provide the average and standard deviation values, expressed as mean _±_ std, for the tokens required to prompt LLMs and those generated by LLMs in addressing a single question from two challenging datasets. 

|Method|MATH|MATH|TheoremQA|TheoremQA|TheoremQA|
|---|---|---|---|---|---|
||Generate tokens<br>Ratios|Prompt tokens<br>Ratios<br>_△_|Generate tokens<br>Ratios||Prompt tokens<br>Ratios<br>_△_|
|ZeroShot CoT<br>CoT8<br>Chain Reasoning<br>ToT Reasoning|212.3_±_152.6<br>1<br>441.3_±_310.7<br>2.08<br>603.2_±_244.7<br>2.84<br>1569.8_±_474.7<br>7.39|110.3_±_25.5<br>1<br>0<br>4611.4_±_1952.2<br>41.8<br>4.2<br>2657.5_±_1187<br>24.1<br>0.74<br>9963.7_±_4487.5<br>90.33<br>3.3|240.8_±_100.5<br>-<br>976.3_±_1056.4<br>1886_±_982.5|1<br>-<br>4.05<br>7.83|121.2_±_43.1<br>1<br>0<br>-<br>-<br>-<br>4614.6_±_7632.1<br>38.07<br>-7.25<br>11982_±_8872.2<br>98.86<br>-5.5|
|TR + W-Voting|7484.9_±_5873.9<br>**35.26**|46904.9_±_37980.8<br>425.25<br>27.19|6762.9_±_6513|28.09|43444.8_±_49391.2<br>**358.46**<br>13|



## **5.3. Token Cost Analysis** 

As shown in Table 4, when addressing questions from challenging datasets, the token cost of GPT-4 with TR is significantly higher than baseline approaches. Specifically, on the MATH dataset, the TR approach, on average, generates 35.26 times more tokens and requires 425 _._ 25 times more prompt tokens than the ZeroShot CoT. The corresponding ratios in the TheoremQA dataset are 28 _._ 09 and 358 _._ 46. This resource-intensive nature of our proposed TR derives from continuously identifying the errors and appending the prompt with error analysis during reasoning. 

However, these additional operations and the high token cost are necessary as hallucinations of LLMs frequently appear. First, since numerous erroneous thoughts are generated during reasoning, consistently identifying and revising them is crucial to ensuring the correctness of the answer. Second, in many cases, the error analysis of LLMs is invalid and even incorrect due to the inevitable hallucinations. Thus, accumulating error analysis derived from different reasoning paths decreases the negative impact of the flawed analysis on thought revisions. Third, many reasoning paths of the thought structure derive from rollbacks triggered by erroneous feedback from LLMs. Since mistaken rollbacks cannot be identified, the TR approach retains all generated paths and ultimately employs majority voting to enhance reliability. 

Therefore, we conclude that there is a trade-off between the high token cost and the problem-solving rate. On the one hand, since the TR approach requires many tokens to address a single question, its application may be limited for users with insufficient resources. On the other hand, compared to zero-shot GPT-4, GPT-4 with TR gains 27.19% and 13% improvements in solving rates on MATH and TheoremQA datasets. When users prioritize problemsolving rates, integrating GPT-4 with the TR approach ensures its applicability in many challenging scenarios. 

## **6. Concluding Remarks** 

In this paper, we proposed Thought Rollback (TR), an effective reasoning framework supported by the rollback of thoughts that allows LLMs to perform adaptive reasoning to solve challenging problems. Without relying on human annotations and specific thought structure designs for reasoning, LLMs with TR can progressively self-organize and revise thoughts based on trial-and-error experiences until reaching a correct solution for various tasks. The _rollback controller_ and _prompt enhancer_ , together with the experience-guided weights majority voting, make TR achieve the state-of-the-art solving rate in many mathematical and multi-task reasoning datasets while maintaining a lower cost than the alternative leading approaches. We hope this work could shed light on the adaptive reasoning in LLMs toward addressing challenging tasks, especially when mathematical problems are involved. 

## **Impact Statement** 

Large language models (LLMs) can break a complex task into manageable subproblems and solve them through stepby-step reasoning. TR, a lightweight framework, guarantees the reliability of LLMs’ multi-step reasoning under hallucinations, thus extending their applications to a wider range of tasks. Furthermore, compared to the BOT that performs outcome analysis, TR built upon prompting the LLMs with feedback from process analysis is more effective and significantly reduces the interaction cost. This may open a research direction emphasizing the importance of exploiting feedback during the step-by-step reasoning of LLMs. In addition, the plug-and-play nature of TR allows other approaches, such as CR, to involve the thought rollback mechanism to improve performance further. Ultimately, thanks to the expansion in the context window of LLMs and the decreasing token prices, the token cost of TR may not be a major concern. 

9 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

## **References** 

- An, S., Ma, Z., Lin, Z., Zheng, N., Lou, J.-G., and Chen, W. Learning from mistakes makes llm better reasoner. _arXiv preprint arXiv:2310.20689_ , 2023. 

- Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Gianinazzi, L., Gajda, J., Lehmann, T., Podstawski, M., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. _arXiv preprint arXiv:2308.09687_ , 2023. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In _Advances in Neural Information Processing Systems_ , volume 33, pp. 1877–1901, 2020. 

- Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _Transactions on Machine Learning Research_ , 2023a. 

- Chen, W., Yin, M., Ku, M., Lu, P., Wan, Y., Ma, X., Xu, J., Wang, X., and Xia, T. Theoremqa: A theorem-driven question answering dataset. In _Proc. Conference on Empirical Methods in Natural Language Processing_ , 2023b. 

- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning. In _Proc. International Conference on Learning Representations_ , 2023. 

- Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. Pal: Program-aided language models. In _International Conference on Machine Learning_ , pp. 10764–10799. PMLR, 2023. 

- Hagberg, A. A., Schult, D. A., and Swart, P. J. Exploring network structure, dynamics, and function using networkx. In _Proc. 7th Python in Science Conference_ , pp. 11–15, 2008. 

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In _Proc. International Conference on Learning Representations_ , 2021a. 

- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. _arXiv preprint arXiv:2103.03874_ , 2021b. 

- Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. Large language models cannot self-correct reasoning yet. In _Proc. International Conference on Learning Representations_ , 2024. 

- Jiang, M., Ruan, Y., Huang, S., Liao, S., Pitis, S., Grosse, R. B., and Ba, J. Calibrating language models via augmented prompt ensembles. In _International Conference on Machine Learning_ , 2023. 

- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. In _Advances in Neural Information Processing Systems_ , volume 35, pp. 22199–22213, 2022. 

- Kong, A., Zhao, S., Chen, H., Li, Q., Qin, Y., Sun, R., and Zhou, X. Better zero-shot reasoning with role-play prompting. _arXiv preprint arXiv:2308.07702_ , 2023. 

- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In _Proc. International Conference on Learning Representations_ , 2024. 

- Ling, W., Yogatama, D., Dyer, C., and Blunsom, P. Program induction by rationale generation: Learning to solve and explain algebraic word problems. _arXiv preprint arXiv:1705.04146_ , 2017. 

- Ling, Z., Fang, Y., Li, X., Huang, Z., Lee, M., Memisevic, R., and Su, H. Deductive verification of chain-of-thought reasoning. In _Advances in Neural Information Processing Systems_ , 2023. 

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_ , 12:157–173, 2024. 

- Luo, L., Li, Y.-F., Haffari, G., and Pan, S. Reasoning on graphs: Faithful and interpretable large language model reasoning. In _Proc. International Conference on Learning Representations_ , 2024. 

- Lyu, Q., Havaldar, S., Stein, A., Zhang, L., Rao, D., Wong, E., Apidianaki, M., and Callison-Burch, C. Faithful chainof-thought reasoning. In _Proc. IJCNLP-AACL_ , 2023. 

- Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with self-feedback. _arXiv preprint arXiv:2303.17651_ , 2023. 

10 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

- OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Patel, A., Bhattamishra, S., and Goyal, N. Are nlp models really able to solve simple math word problems? _arXiv preprint arXiv:2103.07191_ , 2021. 

- Qin, C., Zhang, A., Zhang, Z., Chen, J., Yasunaga, M., and Yang, D. Is chatgpt a general-purpose natural language processing task solver? In _Proc. Conference on Empirical Methods in Natural Language Processing_ , 2023. 

- Sijia, C., Baochun, L., and Niu, D. Boosting of thoughts: Trial-and-error problem solving with large language models. In _Proc. International Conference on Learning Representations_ , 2024. 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

   - Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. In _Proc. International Conference on Learning Representations_ , 2022. 

   - Zhao, X., Xie, Y., Kawaguchi, K., He, J., and Xie, Q. Automatic model selection with large language models for reasoning. In _Proc. Conference on Empirical Methods in Natural Language Processing_ , 2023. 

   - Zheng, C., Liu, Z., Xie, E., Li, Z., and Li, Y. Progressivehint prompting improves reasoning in large language models. _arXiv preprint arXiv:2304.09797_ , 2023. 

   - Zhou, A., Wang, K., Lu, Z., Shi, W., Luo, S., Qin, Z., Lu, S., Jia, A., Song, L., Zhan, M., et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. In _Proc. International Conference on Learning Representations_ , 2024. 

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. In _Proc. International Conference on Learning Representations_ , 2022. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ , volume 35, pp. 24824–24837, 2022. 

- Weng, Y., Zhu, M., Xia, F., Li, B., He, S., Liu, S., Sun, B., Liu, K., and Zhao, J. Large language models are better reasoners with self-verification. In _Proc. Conference on Empirical Methods in Natural Language Processing_ , pp. 2550–2575, 2023. 

- Wu, Z., Jiang, M., and Shen, C. Get an a in math: Progressive rectification prompting. In _Proc. AAAI Conference on Artificial Intelligence_ , 2024. 

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. In _Advances in Neural Information Processing Systems_ , 2023. 

- Yu, J., He, R., and Ying, R. Thought propagation: An analogical approach to complex reasoning with large language models. In _Proc. International Conference on Learning Representations_ , 2024. 

- Zhang, Y., Yang, J., Yuan, Y., and Yao, A. C.-C. Cumulative reasoning with large language models. _arXiv preprint arXiv:2308.04371_ , 2023. 

11 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

## **A. Supplement to Figures of the Main Paper** 

Figure 4 is the full version of the partial one shown in Figure 1. This shows a complex thought structure built by GPT-4 with TR to address a challenging question from MATH dataset. It shows that GPT-4 with TR tends to build a large-scale thought structure by iteratively rolling back thoughts during reasoning. Figure 4 demonstrates the details about how TR exploits _rollback controller_ and _prompt enhancer_ to create a new and correct reasoning path after analyzing the bad thoughts. 

Specifically, from the details presented in Figure 5, we can observe what TR shown in Figure 2 does after reaching the thought. GPT-4 with TR starts from generating a first thought _N −_ 1 _S −_ 1 based on a simple and zero-shot prompt _Q_ . Then, **Rollback of thoughts** follow the process in Figure 2. First, _rollback controller_ analyzes the reasoning path _N −_ 1 _→ N −_ 2 _→ N −_ 3 _→ N −_ 5 _→ N −_ 6 to output the error analysis presented in “N-6, S-5 error analysis:”. This analysis shows that reasoning steps _N −_ 5 _S −_ 4 and _N −_ 6 _S −_ 5 are bad thoughts. According to our discussion in subsection 4.2, _rollback controller_ allows the LLM to roll back to the thought _N −_ 3 _S −_ 3, which is one step before the first bad thought _N −_ 5 _S −_ 4. Then, _prompt enhancer_ accumulates the error analysis as experience in the prompt, as shown in “ **N-3, S-3 to N-7 S-4 Prompt** ” which includes the “#### The 0-th Experience with Analysis ####”. As a result, by avoiding making similar mistakes mentioned by experience, LLM is able to generate a new thought _N −_ 7 _S −_ 4 from the chosen thought _N −_ 3 _S −_ 3. Therefore, “hallucinations” that occur in thought or analysis of LLM may not influence reasoning due to the continuous thought revision guaranteed by the iterative rollbacks during reasoning. As can be seen from _N −_ 9 _S −_ 6, the final solution is revised to be a correct answer 737. Additionally, we can also observe that each rollback will lead to a new reasoning path with the experience from the corresponding error analysis. Thus, two rollbacks _N −_ 6 _→ N −_ 3 and _N −_ 3 _→ N −_ 2 of GPT-4 with TR adaptively create two new reasoning paths. 

## **B. Reproducibility of Thought Rollback Framework** 

## **B.1. Source Code** 

One can also access the source code under the _examples/ThoughtRollback_ folder in the _code_ . The implementation is based on the _llmpebase_ library. The code is written in Python and imports the datasets from Hugging Face to build PyTorch’s data loader. 

Also, the source code for Chain Of Thought, Chain Reasoning, ToT reasoning, and GoT reasoning that we mention in experiments are available in the _examples/ChainOfThought_ , _examples/ChainReasoning_ , _examples/TreeReasoning_ , and _examples/GraphReasoning_ folders, respectively. 

All configuration files used to conduct the experiments are provided in the _configs/_ folder in the _code_ . About how to run, please read the README.md under the _examples/ThoughtRollback_ folder. 

## **B.2. Locations of Generated Thoughts and Reasoning Details** 

Our code will automatically save the generated thoughts and reasoning details under one folder of _LLMPE_ in the root directory. The direct results are placed under the _LLMPE/results_ while the corresponding visible thought structures are stored in _LLMPE/visualizations_ . Then, their sub-folder name will represent the setting of the configuration, such as “TRReasoning ~~g~~ pt-4 ~~z~~ eroshot ~~c~~ ot ~~M~~ ATH”, where “TRReasoning” is the name of our Thought Rollback. Eventually, as shown in Figure 6, you can access the sample by index in _thought_ ~~_s_~~ _tructure_ ~~_*_~~ while reading the results in _llm_ ~~_r_~~ _ecords_ . The thought structure for reasoning will be saved in the _.json_ format, and the visualizations will be in the _.pdf_ format. 

## **B.3. Prompts** 

This subsection presents the basic prompts used in our implementation of Thought Rollback framework. 

**System prompt for thought generation** : _You possess expertise in solving mathematical problems through a systematic, step-by-step reasoning process during which you are dedicated to preventing repeating any errors analyzed in experiences. Your objective is to address the question using a series of reasoning steps delivered in multiple responses, with each response containing one reasoning step. It is crucial to avoid repeating errors mentioned in the given experiences. Begin by reading the provided reasoning steps and then proceed to generate the most appropriate next step in the response, ensuring that the logical progression steadily leads towards a solution._ 

**System prompt for reasoning analysis** : _You are a mathematician specializing in checking and analyzing the reasoning_ 

12 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

_process containing multiple intermediate reasoning steps proposed to address a math question. Please check the correctness of the overall reasoning logic and each reasoning step regarding mathematical logic and rationality._ 

## **Prompt** I **for the next thought generation** : 

_Answer the question about the problem {Problem Name}. After getting the final solution, place it after the sentence ’The final solution is’ for readability.\n\nExperience containing previously made mistakes:\n\n#########{Experiences}#########\n\nConsider the analysis in the above experience to avoid making similar mistakes during reasoning for the question.\n\n\nQuestion: {Question} \n\nAnswer: Let’s think step by step. Let’s focus on carefully generating the next possible reasoning step for reasoning steps below.\n\n\n{Existing Reasoning Steps}\n\n\nFor reasoning steps within , please generate their best next step containing analysis and the corresponding mathematical expression._ 

where the _{Problem Name}_ contains what question to solve, such as “Multiplication”, _{Experiences}_ present the accumulated experiences _Az[χ]_ 0[(] _...q[·]_[)] _−_ 1[discussed in subsection][ 4.3][,] _[ {][Question][}]_[ is the given question description and finally] _[ {][Experiences][}]_[ is] a placeholder to be replaced by the preceding chain of thoughts _z_ 0 _..,n−_ 1. 

## **Prompt** I _R_ **for the error analysis** : 

_Analyze the reasoning steps proposed for the question about the problem {Problem Name}. \nQuestion: {Question} \n Toward addressing the given question, below is a reasoning process containing {Number of Steps} steps: \n\n\n {Existing Reasoning Steps} \n\n\nDouble-check the reasoning process within , please analyze its overall and each step’s correctness by checking whether they are mathematical logic and rationality. Please report an error when any step does not contain a clear mathematical expression. Output empty string when no steps are given.\n_ 

where the _{Number of Steps}_ presents the number of current reasoning steps, i.e., _n_ for a reasoning path _z_ 0 _..,n_ . 

## **B.4. Basic Engineering Settings for TR** 

During implementation, we set the upper bound _U_ of the _incoming rollback_ for one thought to be 3, making no more than 3 rollbacks use this thought as the destination. Besides, we utilize the depth-first search algorithm to find the current growing thought. As mentioned in the subsection 4.4, the reasoning path with more _incoming rollback_ is more important. Therefore, for this depth-first search algorithm, we assign higher priority to the reasoning path with a larger number of _incoming rollbacks_ . As LLMs with TR first generate subsequent thoughts for these reasoning paths, the corresponding solution will be explored in advance. Such a mechanism increases the possibility of getting better answers. 

Apart from these basic settings, our implementation also includes some engineering tricks. First, during reasoning, once a reasoning path causes _outgoing rollbacks_ more than 5 times, it will be ignored in the subsequent reasoning. Second, we do not allow one thought to cause more than 3 _outgoing rollbacks_ to avoid the case that LLMs repeatedly identify a bad thought and trigger rollbacks for it. Third, to increase the speed of reasoning, after noticing that different reasoning paths are independent of each other, we achieve parallel running, which makes LLM generate thoughts for all reasoning paths simultaneously. Therefore, once a new reasoning path is created, one process will be created to make LLM work on the path without blocking others. This can increase the reasoning speed from the scale of minutes to seconds. 

## **C. Discussion: Insights Gained from the TR Framework** 

We argue that the outstanding performance of TR is attributed to three insights. 

First, as mentioned by the work (Lightman et al., 2024), compared to outcome supervision, which provides feedback for a final result, process supervision, which provides feedback for each intermediate reasoning step, is more important to guarantee the reliable reasoning of LLMs. In summary, process supervision significantly outperforms outcome supervision. Actually, TR roughly belongs to process supervision because it performs rollback-by-rollback error analysis during reasoning and continues to revise thoughts based on the accumulation of experience. Thus, the reasoning of LLMs can be adjusted adaptively during reasoning rather than after the results are obtained. However, BoT (Sijia et al., 2024) belongs to outcome supervision as it only collects error analysis after reasoning. 

Second, the work (Huang et al., 2024) figured out that the outcome analysis is sometimes invalid as LLMs cannot use feedback to revise thoughts to improve the solving rate. As discussed in subsection 3.2 of our paper, this may be because when there are many errors in the intermediate reasoning steps, capturing the source mistake and performing useful analysis 

13 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

is challenging. Thus, BoT (Sijia et al., 2024) has to do many tree ensembles and design a complex boosting mechanism to mine effective error analysis to drive LLM’s reasoning revision. On the contrary, TR performs thought rollback triggered by continuous error analysis during reasoning. Thus, once the error is identified, LLMs can directly revise or adjust the reasoning based on timely accumulated experience in the prompt. 

Third, prompting LLMs with a long input context may cause the degradation of reasoning performance. As pointed out by the work (Liu et al., 2024), LLMs do not fairly utilize all the information in the prompt but focus more on the content at the beginning or end of the input context. For BoT, the prompt tends to become extremely long due to the continuous collection of generated reasoning steps and their error analysis over iterations, particularly since the error analysis focuses on the whole reasoning chain. For instance, in the 10-th iteration, the prompt contains ten long contents, each containing all reasoning steps and step-wise analysis. Some of them may not even be correct. However, our TR accumulates experience during reasoning; thus, each experience only contains analysis on very few intermediate steps, generally leading to a quick revision after the rollback. For instance, LLMs with TR can easily generate a correct second step based on the experience “Reasoning step 1: ... correct. Reasoning step 2: ... wrong because....”. However, the BoT’ experience “Reasoning step1: ....correct. Reasoning step 2: ... wrong because.... . Reasoning step 3: ... wrong because..... Reasoning step 4: ... wrong because.... . Reasoning step 5: ... wrong because....” may just mislead the LLM. 

Finally, regarding the time complexity of the TR framework, LLMs with TR incrementally construct the thought structure from #node 0 to #node _n_ . By focusing on the leading term and disregarding constants and lower order terms, the worst-case time complexity of TR is determined to be _O_ ( _n_[2] ). 

## **C.1. Source of Experimental Results** 

In Table 5 and Table 6, we collect experimental results of GPT-4 and GPT-3.5-turbo on various settings. We especially show the corresponding work that reports the results. 

_Table 5._ Source of experiment results of GSM8K. Methods are CSV (Zhou et al., 2024), PHP (Zheng et al., 2023) with GPT-3.5-turbo-, Model Selection (MS) (Zhao et al., 2023), PAL (Gao et al., 2023), PLAY (Kong et al., 2023), Faithful (Lyu et al., 2023), Exps (Bubeck et al., 2023), PoT (Chen et al., 2023a), IG (Qin et al., 2023), BoT (Sijia et al., 2024). 

|Source||GPT-4<br>GPT-3.5-turbo|
|---|---|---|
|||ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT<br>ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT|
|CSV<br>PHP<br>MS<br>PLAY<br>Faithful<br>Exps<br>PoT*<br>IG<br>BoT|Code<br>Code+CSV|-<br>-<br>-<br>92.05<br>-<br>-<br>57.15<br>-<br>-<br>-<br>92.9, 94.9_sc_5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>94.5,97_sc_5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
||+PHP|-<br>-<br>-<br>-<br>94.98<br>-<br>-<br>-<br>-<br>82.88<br>-<br>-<br>-<br>-<br>95.58<br>-<br>-<br>-<br>-<br>85.18|
||PAL<br>Ours|-<br>-<br>-<br>94.65, 95.65_,sc_5, 95.65_,sc_15<br>-<br>-<br>-<br>-<br>80.85, 85.45_,sc_5, 87.45_,sc_15<br>-<br>-<br>-<br>-<br>94.05, 94.75_,sc_5, 95.55_,sc_15<br>-<br>-<br>-<br>-<br>79.25, 80.95_,sc_5, 82.45_,sc_15<br>-<br>-<br>-<br>-<br>95.65,96.55_,sc_5,96.85_,sc_15<br>-<br>-<br>-<br>-<br>82.65,88.25_,sc_5,89.25_,sc_15<br>-|
||Role|-<br>-<br>-<br>-<br>-<br>76.0<br>-<br>79.6<br>76.9<br>-<br>-<br>-<br>-<br>-<br>-<br>78.2<br>-<br>-<br>-<br>-|
||Faithful|46.9<br>-<br>-<br>64.98<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>958<br>-<br>-<br>-<br>-<br>-<br>-|
|||87.1<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
|||-<br>-<br>-<br>-<br>-<br>76.3<br>-<br>-<br>-<br>-|
|||-<br>-<br>-<br>-<br>-<br>23.8<br>-<br>78.9<br>-<br>-|
||BoT|87.1<br>-<br>89.6<br>928<br>94.9<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>97.1<br>98.78<br>-<br>-<br>-<br>-<br>-<br>-|



## **D. Examples of GPT-4 with TR in GSM8K** 

In Figure 7, we present a simple reasoning performed by GPT-4 with TR. As no bad thoughts are identified during reasoning, GPT-4 with TR directly performs correct reasoning toward a correct solution. This simple example aims to give an overview of 1). how multi-step reasoning with multi-step prompts work; 2). how to prompt LLMs to generate the next thought, such as _N −_ 2 _S −_ 2 _→ N −_ 3 _S −_ 3; and 3) how LLMs with TR are able to perform normal reasoning when no rollback is triggered. Besides, as discussed in subsection 4.1, LLMs with TR start from a zero-shot prompt containing only the question and task information. 

Then, Figure 8 shows a more complex reasoning process conducted by GPT-4 with TR. In this case, _rollback controller_ triggers 5 rollbacks during reasoning, leading to 8 different solutions. After operating the experience-guided solution ensemble, we get 6 as the final answer, which is correct. LLMs with TR first generate one thought _N −_ 1 _S −_ 1. Then, _rollback controller_ exploits LLMs to analyze the current reasoning path _N −_ 0 _S −_ 1, _N −_ 1 _S −_ 1 and _N −_ 2 _S −_ 2 and thus identifies the error in _N −_ 2 _S −_ 2. This triggers a rollback _N −_ 2 _S −_ 2 _→ N −_ 1 _S −_ 1, leading to a new reasoning 

14 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

_Table 6._ Source of experiment results of SVAMP, AQuA-RAT, MATH, TheoremQA and MMLU. They are CSV (Zhou et al., 2024), PHP (Zheng et al., 2023), Model Selection (MS) (Zhao et al., 2023), PAL (Gao et al., 2023), PLAY (Kong et al., 2023), Faithful (Lyu et al., 2023), Exps (Bubeck et al., 2023), PoT (Chen et al., 2023a), IG (Qin et al., 2023), BoT (Sijia et al., 2024), CR (Zhang et al., 2023), TheoremQA (Chen et al., 2023b) and GPT4-report (OpenAI, 2023). 

|Source||SVAMP<br>GPT-4<br>GPT-3.5-turbo|
|---|---|---|
|||ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT<br>ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT|
|PHP<br>MS<br>PLAY<br>Faithful<br>PoT*<br>IG<br>BoT|+PHP|-<br>-<br>-<br>-<br>90.58<br>-<br>-<br>-<br>-<br>81.08<br>-<br>-<br>-<br>-<br>91.98<br>-<br>-<br>-<br>-<br>83.18|
||PAL<br>Ours|-<br>-<br>-<br>91.95<br>-<br>-<br>-<br>-<br>835<br>-<br>-<br>-<br>-<br>92.25<br>-<br>-<br>-<br>-<br>80.35<br>-<br>-<br>-<br>-<br>93.75<br>-<br>-<br>-<br>-<br>84.35<br>-|
||Role|-<br>-<br>-<br>-<br>-<br>75.3<br>-<br>76.3<br>82.2<br>-<br>-<br>-<br>-<br>-<br>-<br>83.8<br>-<br>-<br>-<br>-|
||Faithful|-<br>88.4<br>-<br>808<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>95.48<br>-<br>-<br>-<br>-<br>-<br>-|
|||-<br>-<br>-<br>-<br>-<br>88.2<br>-<br>-<br>-<br>-|
|||-<br>-<br>-<br>-<br>-<br>74.8<br>-<br>77.5<br>-<br>-|
||BoT|68.7<br>-<br>74.3<br>77.68<br>90.58<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>92.7<br>94.98<br>-<br>-<br>-<br>-<br>-<br>-|
|Source||AQuA-RAT<br>GPT-4<br>GPT-3.5-turbo|
|||ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT<br>ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT|
|PHP<br>PLAY<br>Faithful<br>PoT*<br>IG<br>BoT|+PHP|-<br>-<br>-<br>-<br>77.58<br>-<br>-<br>-<br>-<br>57.48<br>-<br>-<br>-<br>-<br>79.98<br>-<br>-<br>-<br>-<br>60.68|
||Role|-<br>-<br>-<br>-<br>-<br>53.5<br>-<br>53.9<br>59.4<br>-<br>-<br>-<br>-<br>-<br>-<br>63.8<br>-<br>-<br>-<br>-|
||Faithful|50.4<br>-<br>-<br>75.28<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>73.68<br>-<br>-<br>-<br>-<br>-<br>-|
|||-<br>-<br>-<br>72.4<br>-<br>-<br>-<br>-<br>-<br>-|
|||-<br>-<br>-<br>-<br>-<br>28.0<br>-<br>53.5<br>-<br>-|
||BoT|40.6<br>-<br>73.2<br>748<br>77.5<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>81.4<br>84.98<br>-<br>-<br>-<br>-<br>-<br>-|
|Source||MATH<br>GPT-4<br>GPT-3.5-turbo|
|||ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT<br>ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT|
|CSV<br>PHP<br>BoT|Code<br>Code+CSV|42.2<br>-<br>-<br>-<br>50.368<br>-<br>-<br>-<br>-<br>34.128<br>69.69, 79.88_sc_16<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>73.54, 83.54_sc_16, 84.32_scvw_<br>16<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|
||+PHP|-<br>-<br>-<br>42.58<br>50.368<br>-<br>-<br>-<br>-<br>34.128<br>-<br>-<br>-<br>-<br>53.98<br>-<br>-<br>-<br>-<br>36.58|
||BoT|42.5<br>-<br>47.7<br>48.938<br>50.48<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>62.5<br>66.38<br>-<br>-<br>-<br>-<br>-<br>-|
|CR500|CR<br>PHP+CR|-<br>-<br>-<br>54.24<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>584<br>-<br>-<br>-<br>-<br>-<br>-|
|Source||TheoremQA<br>GPT-4<br>GPT-3.5-turbo|
|||ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT<br>ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT|
|TheoremQA|PoT|-<br>-<br>-<br>43.8<br>-<br>-<br>-<br>-<br>30.2, 30.8_theorem_<br>-<br>-<br>52.4<br>-<br>-<br>-<br>-<br>35.6, 35.8_theorem_<br>-<br>-<br>-|
|Source||MMLU<br>GPT-4<br>GPT-3.5-turbo|
|||ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT<br>ZeroShot<br>FewShot<br>ZeroShot-CoT<br>CoT<br>Complex-CoT|
|GPT4-report|BoT|86.5<br>86.4<br>-<br>-<br>-<br>70<br>-<br>-<br>-<br>-<br>-<br>-<br>90.86<br>93.425<br>-<br>-<br>-<br>-<br>-<br>-|



path _N −_ 1 _S −_ 1 _→ N −_ 3 _S −_ 2 in which the prompt includes the error analysis, thereby revising the thought to gain a final correct solution _N −_ 9 _S −_ 4. 

As there are 3 _incoming rollbacks_ for the thought _N −_ 10 _S −_ 2, the corresponding three different error analyses are accumulated as shown by the **N-10 S-2 Experience Accumulation:** . _prompt enhancer_ include these error analyses as experiences in the prompt to guide LLMs to produce correct thoughts. For instance, **N-10 S-2** _→_ **N-17 S-3 Prompt** shows that GPT-4 generates the thought _N −_ 17 _S −_ 3 from the thought _N −_ 10 _S −_ 2 with the prompt that accumulates two experiences. 

Eventually, GPT-4 with TR adaptively builds a thought structure towards generating solutions that, most of which are correct due to the continuous thought revisions via rollback of thoughts. 

## **E. Examples of GPT-4 with TR in MATH** 

Limited by space, we store the detailed experimental result and visualization files of Figure 9 in the folder _MATH-example-1_ of the supplementary. Thus, we present the question, the thoughts of correct solutions, and the values of all solutions. Specifically, in response to our discussion in section 4, we present how GPT-4 with TR is able to identify the bad thought _N −_ 34 _S −_ 6 and thus the _rollback controller_ gets the _N −_ 34 _S −_ 6 error analysis. The triggered rollback _N −_ 34 _S −_ 6 _→ N −_ 32 _S −_ 4 leads to a new reasoning path _N −_ 32 _S −_ 4 _→ N −_ 35 _S −_ 5, which generates a correct thought and gets 

15 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

## the correct answer 0. 

We specifically utilize the example in Figure 10 to show how **Experience Accumulation** works in the TR framework. For the reasoning path from _N −_ 0 _S −_ 0 to _N −_ 17 _S −_ 3, there are 4 _incoming rollbacks_ , including the rollback _N −_ 3 _S −_ 2 _→ N −_ 0 _S −_ 0, the rollback _N −_ 10 _S −_ 3 _→ N −_ 4 _S −_ 1, and the rollback _N −_ 15 _S −_ 3 _→ N −_ 11 _S −_ 2. As each rollback creates an error experience from one trial of the given question, _incoming rollbacks_ lead to a series of experiences, as shown by **N-11 S-2 Experience Accumulation** :. Therefore, to generate _N −_ 17 _S −_ 3 from _N −_ 11 _S −_ 2, the _prompt enhancer_ includes these error analyses as experiences in the prompt, as shown in **N-11 S-2 -¿ N-17 S-3 Prompt:** . By learning from these experiences, LLMs are able to generate the correct thought _N −_ 17 _S −_ 3 and thus the correct answer 47. The detailed files of this example is presented in the folder _MATH-example-2_ of the supplementary. 

## **F. Examples of GPT-4 with TR in TheoremQA** 

GPT-4 with TR tends to build complex thought structures when reasoning with the challenging TheoremQA dataset (Chen et al., 2023b). As seen in Figure 11, the overall thought scale and the complexity of reasoning paths increase a lot compared to other examples. 

Also, we show in Figure 12 that GPT-4 with TR can build even more complex thought structures. 

In these two figures, we present the obtained reasoning path toward the correct answer. For example, the “Final solution of the reasoning path N-0 S-0 _→_ N-11 S-7:” in Figure 11 and the “Final solution of the reasoning path N-0 S-0 _→_ N-12 S-7:” in Figure 12 are the correct solutions obtained by GPT-4 with TR. 

16 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0017-01.png)


_Figure 4._ Complete thought structure of Figure 1 (c) built by GPT-4 with TR for the question from MATH dataset (Hendrycks et al., 2021b). This structure contains 23 nodes, i.e. 23 thoughts and leads to _K_ = 8 reasoning paths towards solutions. It is plotted by based on the NetworkX lib (Hagberg et al., 2008) under the “fdp” layout. 

17 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0018-01.png)


**----- Start of picture text -----**<br>
N-1 S-1 : N-3, S-3 to N-7 S-4 Prompt :<br>The first step is to find out how many students are in the school. We can do thisby multiplying the number of classrooms by the number of students in eachclassroom. \n\nMathematically, this is represented as: \n\n67 classrooms * 66 Q Answer the question about the problem Multiplication. After : Answer the question about the problem Multiplication. After getting the final solution, place itafter the sentence 'The final solution is' for readability.<br>students/classroom = Total number of students. getting the final solution, place it after the sentence 'The Experience containing previously made mistakes:<br>final solution is' for readability.\n\nQuestion: The school is<br>planning a field trip. The school has 67 classrooms. There ######### The 0-th Experience with Analysis #########<br>N-6, S-5  error analysis: are 66 students in each classroom in the school. If there are     {N-6, S-5 error analysis}.<br>6 seats on each school bus How many buses are needed to ####################<br>The overall reasoning logic is correct, and each reasoning step is logically sound andmathematically rational. Here is the analysis for each step:\n\nReasoning Step 1: The step iscorrect. The total number of students can be found by multiplying the number of classrooms take the trip? \n\nAnswer: Let's think step by step. Consider the analysis in the above experience to avoid making similar mistakes duringreasoning for the question.<br>by the number of students in each classroom.\n\nReasoning Step 2: The step is correct. The<br>calculation 67 classrooms * 66 students/classroom = 4422 students is correct.\n\nReasoningStep 3: The step is correct. The number of buses can be found by dividing the total number ofstudents by the number of seats on each bus.\n\nthere is a calculation error. The correct division is 4422 students / 6 seats/bus = 737 busesReasoning Step 4: The step is correct, but Question: The school is planning a field trip. The school has 67 classrooms. There are 66students in each classroom in the school. If there are 6 seats on each school bus How manybuses are needed to take the trip?<br>with a remainder. The step correctly identifies that we cannot have a fraction of a bus, so weneed to round up to the nearest whole number. buses with a remainder, not 738 buses.\n\nReasoning Step 5: The step is incorrect because itHowever, the calculation should result in 737 Answer: Let's think step by step.  Let's focus on carefully generating the next possiblereasoning step for reasoning steps below.<br>carries forward the calculation error from the previous step. The final solution should be thatthe school needs 737 buses for the trip, not 738.\n\nSo, the error is in Reasoning Step 4 and5, where the division result was incorrectly rounded up to 738 instead of 737. --------------------    The first step is to find out how many students are in the school. We can do this bymultiplying the number of classrooms by the number of students in each classroom.<br>N-5, S-4  error Thought : Mathematically, this is represented as:<br>The next step is to perform the division operation to find out thenumber of buses needed. \n\nMathematically, this is representedas:\n\n4422 students / 6 seats/bus = 737 buses. \n\nHowever, since 67 classrooms * 66 students/classroom = Total number of students.     The next step is to calculate the total number of students in the school. We can do this byperforming the multiplication operation as suggested in the first step.<br>we cannot have a fraction of a bus, we need to round up to thenearest whole number because even if a bus is not full, it will still be Mathematically, this is represented as:<br>needed. So, the school needs 738 buses for the trip. 67 classrooms * 66 students/classroom = 4422 students.<br>    The next step is to find out how many buses are needed to transport all the students. We<br>can do this by dividing the total number of students by the number of seats on each bus.<br>N-6, S-5  error Thought : Mathematically, this is represented as:<br>N-9, S-6 Thought : 4422 students / 6 seats/bus = Number of buses needed.<br>The final solution is to confirm the number of buses needed --------------------<br>for the trip.\n\nMathematically, this is representedas:\n\nThe school needs 738 buses for the trip. The next step is to finalize the solution. \n\nMathematically,this is represented as:\n\nThe final solution is that theschool needs 737 buses for the trip. For reasoning steps within --------------------, please generate their best next step containinganalysis and the corresponding mathematical expression.<br>**----- End of picture text -----**<br>


_Figure 5._ Complete thought structure of Figure 2 built by GPT-4 with TR for the question from SVAMP dataset (Patel et al., 2021). This structure contains 10 nodes, i.e. 10 thoughts and leads to _K_ = 3 reasoning paths towards solutions. It is plotted by based on the NetworkX lib (Hagberg et al., 2008) under the “dot” layout. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0018-03.png)


_Figure 6._ Illustrating of all results generated by LLMs with TR. The left sub-figure presents the details of the generated thought structure. The upper right sub-figure shows the files of obtained results, while the lower right sub-figure presents the visualizations. 

18 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 

## Q **:** 

## N-1 S-1 **:** 

_Given that one pair of heels costs $33 and the other costs twice as much, we can find the total cost of the two pairs of heels by adding the cost of the first pair to twice its cost. This can be represented by the mathematical expression: \n\nCost of two pairs of heels = $33 + 2*$33._ 

_Answer the question about the problem Algebra. After getting the final solution, place it after the sentence 'The final solution is' for readability.\n\nQuestion: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs twice as much, how many dollars are the boots? \n\nAnswer: Let's think step by step._ 

## N-2 S-2 **:** 

## N-3 S-3 **:** 

_Given the expression for the cost of two pairs of heels as $33 + 2*$33, we can calculate the total cost by performing the operations. This results in:\n\nCost of two pairs of heels = $33 + $66._ 

_The next reasoning step would be to add the costs together to find the total cost of the two pairs of heels. This can be represented by the mathematical expression:\n\nCost of two pairs of heels = $33 + $66 = $99._ 

## N-4 S-4 **:** 

## N-5 S-5 **:** 

_The next reasoning step would be to calculate the total cost of the boots by adding $99 and $5. This can be represented by the mathematical expression:\n\nCost of boots = $99 + $5 = $104. \n\nThe final solution is the boots cost $104._ 

_Given that the two pairs of high heels together cost five dollars less than the boots, we can represent the cost of the boots in terms of the cost of the two pairs of heels. This can be represented by the mathematical expression:\n\nCost of boots = Cost of two pairs of heels + $5. \n\nSubstituting the calculated cost of two pairs of heels ($99) into this expression gives:\n\nCost of boots = $99 + $5._ 

## **N-2 S-2 -> N-3 S-3 Prompt:** 

_Answer the question about the problem Algebra. After getting the final solution, place it after the sentence 'The final solution is' for readability.\n\nQuestion: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs twice as much, how many dollars are the boots? \n\nAnswer: Let's think step by step.  Let's focus on carefully generating the next possible reasoning step for reasoning steps below.\n\n--------------------\n\tGiven that one pair of heels costs $33 and the other costs twice as much, we can find the total cost of the two pairs of heels by adding the cost of the first pair to twice its cost. This can be represented by the mathematical expression: \n\nCost of two pairs of heels = $33 + 2*$33. \n\tGiven the expression for the cost of two pairs of heels as $33 + 2*$33, we can calculate the total cost by performing the operations. This results in:\n\nCost of two pairs of heels = $33 + $66. \n--------------------\n\nFor reasoning steps within --------------------, please generate their best next step containing analysis and the corresponding mathematical expression._ 

_Figure 7._ A simple thought structure built by GPT-4 with TR for the question from GSM8K dataset (Cobbe et al., 2021). This structure contains 5 nodes, i.e. 5 thoughts and leads to _K_ = 1 reasoning path towards one solution because no error is identified by the _rollback controller_ with GPT-4 during reasoning. 

19 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0020-01.png)


**----- Start of picture text -----**<br>
N-2 S-2  error analysis: Q :<br>The first step of the reasoning process is correct. Marissa has indeed walked a total of 6 milesin 2 hours, which gives her an average speed of 3 miles per hour. The mathematicalexpression is also correct: Average speed = Total distance/Total time = 6 miles/2 hours = 3miles/hour.\n\nHowever, mathematical expression is correct in that Total time = Total distance / Average speed = 12miles / 4 miles/hour = 3 hours, the second step of the reasoning process is incorrect. but the interpretation of this expression is wrong. ThisThe Answer the question about the problem Algebra. After getting the final solution, place it afterthe sentence 'The final solution is' for readability.\n\nQuestion: Marissa is hiking a 12-miletrail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. Ifshe wants her average speed to be 4 miles per hour, what speed (in miles per hour) doesshe need to walk the remaining distance? \n\nAnswer: Let's think step by step.<br>calculation gives the total time Marissa has to complete the entire trail to maintain an average<br>speed of 4 miles per hour, but she has already spent 2 hours hiking. Therefore, she only has 1hour left to complete the remaining 6 miles of the trail.\n\nTo find the speed she needs to walk N-3 S-2 :<br>the remaining distance, we should use the formula Speed = Distance / Time. So, the speed<br>she needs to maintain for the remaining 6 miles in 1 hour is 6 miles/hour. Next Reasoning Step: Marissa still has 6 miles left to hike. To achieve her target average<br>speed of 4 miles per hour for the entire 12-mile trail, she needs to complete the hike in 3hours. However, she has already spent 2 hours hiking. Therefore, she only has 1 hour left to N-10  S-2 Experience Accumulation:<br>complete the remaining 6 miles of the trail.\n\nMathematical Expression: Remaining time =Total time - Time already spent = 3 hours - 2 hours = 1 hour. Rollback N-11 -> N-10 :<br>The overall reasoning process is logical and the steps are mathematically rational. However,<br>N-2 S-2 : there is an error in Reasoning Step 3. The step incorrectly calculates the total time Marissahas to complete the hike. Instead, it should determine the time Marissa has left to complete<br>the hike after considering the time she has already spent.\n\nThe corrected Reasoning Step 3<br>Next Reasoning Step: Marissa still has 6 miles left to hike. To achieve her target averagespeed of 4 miles per hour for the entire 12-mile trail, she needs to complete the hike in 3hours.\n\nMathematical Expression: Total time = Total distance / Average speed = 12 miles /4 miles/hour = 3 hours. should be: Determine the time Marissa has left to complete the hike to achieve her targetaverage speed of 4 miles per hour.\n\nMathematical Expression: Time left = Total time - Timespent = 3 hours - 2 hours = 1 hour.\n\nThen, a new Reasoning Step 4 should be added:Determine the speed Marissa needs to walk the remaining distance in the time<br>left.\n\nMathematical Expression: Required speed = Remaining distance / Time left = (12<br>miles - 6 miles) / 1 hour = 6 miles/hour. \n\nSo, Marissa needs to walk at a speed of 6 miles<br>per hour to complete the remaining distance in time to achieve her target average speed of 4<br>N-4 S-2 : miles per hour.<br>Next Reasoning Step: Since Marissa has already spent 2 hours hiking, she has 1 hour left to Rollback N-12 -> N-10 :<br>complete the remaining 6 miles.\n\nMathematical Expression: Remaining time = Total time -<br>Time spent = 3 hours - 2 hours = 1 hour. The overall reasoning process is correct, and each step is logically sound and rational. Here is<br>the breakdown of each step:\n\nReasoning Step 1: This step is correct. The total distance<br>Marissa has covered so far is indeed 6 miles (4 miles + 2 miles), and the total time she has<br>spent is 2 hours (1 hour + 1 hour). The mathematical expressions are clear and<br>accurate.\n\nReasoning Step 2: This step is also correct. The average speed is calculated by<br>dividing the total distance by the total time, which in this case is 6 miles divided by 2 hours,<br>giving an average speed of 3 miles per hour. The mathematical expression is clear and<br>accurate.\n\nReasoning Step 3: This step is correct, but it is not complete. It correctly<br>calculates the total time Marissa has to complete the hike to achieve her target average speed<br>of 4 miles per hour. However, it does not answer the original question: what speed does she<br>N-9 S-4 : need to walk the remaining distance? \n\nTo answer the original question, we need an<br>additional step:\n\nReasoning Step 4: Calculate the remaining distance and the time she has<br>N-6 S-5 Next Reasoning Step: Therefore, Marissa needs to walk at a speed of 6 miles per hour forthe remaining 1 hour to achieve her target average speed of 4 miles per hour for the entire12-mile trail.\n\nMathematical Expression: The final solution is Marissa's required speed = 2miles/hour.. : Next Reasoning Step: Therefore, toachieve her target average speed of 4miles per hour for the entire 12-mile trail,Marissa needs to walk the remaining 6miles at a speed of 6 miles perhour.\n\nMathematical Expression: Thefinal solution is Marissa needs to walk ata speed of 6 miles/hour for the remaining6 miles. left, then find the speed she needs to maintain to cover the remaining distance in theremaining time.\n\nMathematical Expression: Remaining distance = Total distance - Distancecovered so far = 12 miles - 6 miles = 6 miles. Time left = Total time - Time spent so far = 3hours - 2 hours = 1 hour. Speed needed = Remaining distance / Time left = 6 miles / 1 hour =6 miles/hour.\n\nThis step would complete the reasoning process and answer the originalquestion. Rollback N-17 -> N-10 :<br>The overall reasoning process is mostly correct, but there is a logical error in the final step.<br>Let's analyze each step:\n\nReasoning Step 1: The calculations for total distance and total<br>time are correct. Marissa has indeed covered 6 miles in 2 hours. \n\nMathematical Expression:Total distance = 4 miles + 2 miles = 6 miles \nTotal time = 1 hour + 1 hour = 2 hours<br>\n\nReasoning Step 2: The calculation for Marissa's average speed so far is correct. She has<br>been hiking at an average speed of 3 miles per hour.\n\nMathematical Expression: Average<br>speed = Total distance / Total time = 6 miles / 2 hours = 3 miles/hour. \n\nReasoning Step 3:<br>This is where the logical error occurs. While it's correct that the total time for Marissa to<br>complete the 12 miles at an average speed of 4 miles per hour would be 3 hours, the question<br>asks for the speed she needs to maintain for the remaining distance. Since she has already<br>spent 2 hours, she has only 1 hour left to complete the remaining 6 miles. Therefore, she<br>needs to walk at a speed of 6 miles per hour for the remaining distance.\n\nThe corrected<br>Mathematical Expression should be: Remaining distance = Total distance - Distance covered<br>= 12 miles - 6 miles = 6 miles. Time left = Total time - Time spent = 3 hours - 2 hours = 1 hour.<br>Required speed = Remaining distance / Time left = 6 miles / 1 hour = 6 miles/hour.<br>N-10 S-2 -> N-22 S-3 Prompt:<br>**----- End of picture text -----**<br>


_Answer the question about the problem Algebra. After getting the final solution, place it after the sentence 'The final solution is' for readability.\n\nExperience containing previously made mistakes:\n\n######### The 0-th Experience with Analysis #########\n\t Experience of the rollback N-12 -> N-10 \n####################\n\n######### The 1-th Experience with Analysis #########\n\t  Experience of the rollback N-17-> N-10\n####################\n\nConsider the analysis in the above experience to avoid making similar mistakes during reasoning for the question.\n\n\nQuestion: Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance? \n\nAnswer: Let's think step by step.  Let's focus on carefully generating the next possible reasoning step for reasoning steps below.\n\n--------------------\n\tReasoning Step 1: Calculate Marissa's total distance covered and total time spent so far. \n\nMathematical Expression: Total distance = 4 miles + 2 miles = 6 miles \nTotal time = 1 hour + 1 hour = 2 hours \n\tReasoning Step 2: Calculate Marissa's average speed so far. \n\nMathematical Expression: Average speed = Total distance / Total time = 6 miles / 2 hours = 3 miles/hour. \n--------------------\n\nFor reasoning steps within -------------------, please generate their best next step containing analysis and the corresponding mathematical expression.._ 

_._ 

Final solution of the reasoning path N-0 S-0 --> N-27 S-6 **:** 

\tReasoning Step 1: Calculate Marissa's total distance covered and total time spent so far. \n\nMathematical Expression: Total distance = 4 miles + 2 miles = 6 miles \nTotal time = 1 hour + 1 hour = 2 hours \n\tReasoning Step 2: Calculate Marissa's average speed so far. \n\nMathematical Expression: Average speed = Total distance / Total time = 6 miles / 2 hours = 3 miles/hour. \n\tReasoning Step 3: Determine the total time Marissa has to complete the hike to achieve her target average speed of 4 miles per hour.\n\nMathematical Expression: Total time = Total distance / Target average speed = 12 miles / 4 miles/hour = 3 hours. \n\tReasoning Step 4: Determine the remaining distance Marissa needs to cover and the time she has left to complete the hike.\n\nMathematical Expression: Remaining distance = Total distance - Distance covered = 12 miles - 6 miles = 6 miles. \nTime left = Total time - Time spent = 3 hours - 2 hours = 1 hour. \n\tReasoning Step 5: Now that we know the remaining distance Marissa needs to cover and the time she has left to do so, we can calculate the speed she needs to maintain to complete the hike in the desired total time.\n\nMathematical Expression: Required speed = Remaining distance / Time left = 6 miles / 1 hour = 6 miles/hour. \n\tReasoning Step 6: Based on the calculations, Marissa needs to walk the remaining 6 miles in 1 hour to achieve her target average speed of 4 miles per hour for the entire 12-mile trail. This means she needs to walk at a speed of 6 miles per hour for the remaining distance.\n\nMathematical Expression: The final solution is 6 miles/hour. 

_Figure 8._ A slightly complex thought structure built by GPT-4 with TR for the question from GSM8K dataset (Cobbe et al., 2021). This structure contains 27 nodes, i.e. 27 thoughts and leads to _K_ = 8 reasoning paths towards 8 solutions as 5 number of rollbacks are triggered by the _rollback controller_ with GPT-4 during reasoning. 

20 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0021-01.png)


**----- Start of picture text -----**<br>
N-40 S-5 :<br>2*[a*(143^5 / 5 + 143^4 / 2 + a_2*143^3 +<br>a_1*143^2 + a_0*143) + b*(143^4 / 4 + 143^3 / 2<br>Reasoning Step:\n\nNow, let's observe that the polynomial p(x) has the same value at four different points: + b_2*143^2 + b_1*143 + b_0) + c*(143^3 / 3 +<br>55, 83, 204, and 232. This is a characteristic of periodic functions, not polynomials. However, we can take 143^2 / 2 + c_1*143 + c_0) + d*(143^2 / 2 + 143)<br>advantage of this observation by considering the polynomial modulo these differences. In other words, we + 143*e] - 64.<br>can consider the values of p(x) for x modulo 55, 83, 204, and 232. \n\nIf we look at the numbers 1 and 286<br>(the arguments of the terms that do not cancel out in the telescoping series), we can see that 286 is<br>congruent to 1 modulo 55, 83, 204, and 232. This means that p(286) should be equal to p(1) because the<br>polynomial has the same value at these congruent points. Therefore, the value of the expression p(1) -<br>p(286) should be zero. \n\nThe final solution is 0.\n\nMathematical Expression:\n\np(1) - p(286) = 0.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0021-02.png)


**----- Start of picture text -----**<br>
N-17 S-7 :<br>Reasoning Step:\n\nSince all pairs of consecutive terms cancel each other out and the roots of the<br>polynomial result in p(x) = 8, the final solution is 0.\n\nMathematical Expression:\n\nThe final solution is 0.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0021-04.png)


**----- Start of picture text -----**<br>
N-34 S-6 :<br>Reasoning Step:\n\nNow, we can calculate the sum by substituting the value of f into the expression for the<br>sum. \n\nMathematical Expression:\n\nSum = f*(1 - 55)*(1 - 83)*(1 - 204)*(1 - 232) + 8 - f*(2 - 55)*(2 - 83)*(2<br>- 204)*(2 - 232) + 8 + ... - f*(286 - 55)*(286 - 83)*(286 - 204)*(286 - 232) + 8\n\nBy calculating this sum, we<br>will get the final solution for the given problem. \n\nReasoning Step:\n\nAfter calculating the sum, we can<br>state the final solution to the problem.\n\nMathematical Expression:\n\nThe final solution is Sum.<br>N-34 S-6 error analysiserror analysis :<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0021-05.png)


**----- Start of picture text -----**<br>
N-34 S-6 error analysiserror analysis :<br>The overall reasoning process is correct and each step is logically sound and mathematically rational.<br>However, there is a critical step missing in this reasoning process. \n\nIn Reasoning Step 5, the alternating<br>sum pattern is correctly recognized, but the crucial observation that the polynomial p(x) is symmetric around<br>x = 143.5 is missed. Specifically, p(x) = p(287 - x) due to the symmetry of the polynomial's roots. This<br>means that for each pair of terms in the sum (e.g., p(1) and p(286), p(2) and p(285), etc.), they are equal<br>Q : and thus cancel out in the alternating sum. \n\nSo, the sum from p(1) to p(286) following the given pattern<br>would be 0, not the complicated expression provided in Reasoning Step 5. \n\nThe final solution should be<br>0, not the Sum as stated in Reasoning Step 6. \n\nSo, the error is in Reasoning Steps 5 and 6 where the<br>Answer the question about the problem Intermediate symmetry of the polynomial and the cancellation in the alternating sum is not taken into account, leading to<br>Algebra. After getting the final solution, place it after the an incorrect final solution.<br>sentence 'The final solution is' for readability.\n\nQuestion:<br>Let $p(x)$ be a polynomial of degree 4 such that $p(55) =<br>p(83) = p(204) = p(232) = 8$ and $p(103) = 13.$  Find\n\ N-35 S-5 :<br>[p(1) - p(2) + p(3) - p(4) + \dots + p(285) - p(286).\]<br>\n\nAnswer: Let's think step by step.  . Reasoning Step:\n\nAt this point, it's important to recognize the symmetry of the polynomial p(x) due to its<br>roots. Specifically, p(x) = p(287 - x) due to the symmetry of the polynomial's roots. This symmetry will<br>greatly simplify the calculation of the alternating sum given in the question.\n\nMathematical<br>Expression:\n\np(x) = p(287 - x)\n\nNow, let's consider the alternating sum from p(1) to p(286). Due to the<br>symmetry of the polynomial, for each pair of terms in the sum (e.g., p(1) and p(286), p(2) and p(285), etc.),<br>they are equal and thus cancel out in the alternating sum. This leads us to the conclusion that the sum from<br>p(1) to p(286) following the given pattern would be 0.\n\nMathematical Expression:\n\np(1) - p(2) + p(3) -<br>p(4) + ... + p(285) - p(286) = 0\n\nThe final solution to the problem is therefore 0.\n\nMathematical<br>Expression:\n\nThe final solution is 0.<br>p(1) + p(3) + p(5) + ... + p(283) - p(286) = -5 / (48*20*101*129) * [(1 - 55)(1 - 83)(1 - 204)(1 -<br>232) + (3 - 55)(3 - 83)(3 - 204)(3 - 232) + ... + (283 - 55)(283 - 83)(283 - 204)(283 - 232) -<br>(286 - 55)(286 - 83)(286 - 204)(286 - 232)] + 8 * 141.<br>**----- End of picture text -----**<br>


Final solution of the reasoning path N-0 S-0 --> N-35 S-5 **:** 

\tReasoning Step: \n\nThe polynomial p(x) has degree 4, which means it can be written in the form ax^4 + bx^3 + cx^2 + dx + e. Since p(55) = p(83) = p(204) = p(232) = 8, we know that p(x) - 8 = 0 for x = 55, 83, 204, 232. This means that x - 55, x - 83, x - 204, and x - 232 are factors of p(x) - 8. Since these are four factors and the polynomial is of degree 4, we can say that p(x) - 8 is of the form f(x)(x - 55)(x - 83)(x - 204)(x - 232), where f(x) is a constant function. \n\nMathematical Expression: \n\np(x) - 8 = f(x)(x - 55)(x - 83)(x - 204)(x - 232) \n\tReasoning Step:\n\nGiven that p(103) = 13, we can substitute x = 103 into the equation to solve for the constant function f(x). \n\nMathematical Expression:\n\n13 - 8 = f(103)(103 - 55)(103 - 83)(103 - 204)(103 - 232) \n\tReasoning Step:\n\nBy simplifying the right side of the equation, we can solve for the value of f(103).\n\nMathematical Expression:\n\n5 = f(103)(48)(20)(-101)(-129) \n\tReasoning Step:\n\nSolving for f(103) from the equation, we get f(103) as a constant value.\n\nMathematical Expression:\n\nf(103) = 5 / [(48)(20)(-101)(-129)] \n\tReasoning Step:\n\nThe constant function f(x) is the same for all x, so we can now write the polynomial p(x) in terms of x.\n\nMathematical Expression:\n\np(x) = f(103)(x - 55)(x - 83)(x - 204)(x - 232) + 8 \n\tReasoning Step:\n\nWe can now substitute the values of x from 1 to 286 into the equation. However, noticing the pattern of the expression, p(1) - p(2) + p(3) - p(4) + ... + p(285) - p(286), we can see that each pair of consecutive terms cancel each other out, except for the terms where x = 55, 83, 204, and 232. These are the roots of the polynomial, for which p(x) = 8.\n\nMathematical Expression:\n\np(1) - p(2) + p(3) - p(4) + ... + p(285) - p(286) = 8 - 8 + 8 - 8 + ... + 8 - 8 = 0 \n\tReasoning Step:\n\nSince all pairs of consecutive terms cancel each other out and the roots of the polynomial result in p(x) = 8, the final solution is 0.\n\nMathematical Expression:\n\nThe final solution is 0. 

_Figure 9._ A complex thought structure built by GPT-4 with TR for the question from MATH dataset (Hendrycks et al., 2021b). This structure contains 46 nodes, i.e. 46 thoughts and leads to _K_ = 8 reasoning path towards 8 solutions as 13 number of rollbacks are triggered by the _rollback controller_ with GPT-4 during reasoning. It is plotted by based on the NetworkX lib (Hagberg et al., 2008) under the “fdp” layout. 

21 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0022-01.png)


**----- Start of picture text -----**<br>
N-17 S-3 :<br>N-15 S-3 :<br>Reasoning Step 2: We can solve the equations 6 + (n1 - 1) * 4 = 94<br>and 7 + (n2 - 1) * 4 = 95 to find the number of terms in each arithmetic Reasoning Step 3: We have found the number of terms in each of<br>sequence. \n\nFor the first equation, we can subtract 6 from both the two sequences, which are 23 each. However, there is a<br>sides to get (n1 - 1) * 4 = 88. Then, we can divide both sides by 4 to mistake in the second equation solution. The correct simplification<br>get n1 - 1 = 22. Finally, we can add 1 to both sides to get n1 = 23. for the second equation should be 4n2 + 3 = 91, not 4n2 + 3 = 95.<br>\n\nFor the second equation, we can subtract 7 from both sides to get(n2 - 1) * 4 = 88. Then, we can divide both sides by 4 to get n2 - 1 = After subtracting 3 from both sides, we get 4n2 = 88. Finally,dividing both sides by 4 gives n2 = 22, not 23. \n\nSo, the correct<br>22. Finally, we can add 1 to both sides to get n2 = 23. \n\nTherefore, number of terms in the first sequence is 23 and in the second<br>there are 23 terms in each arithmetic sequence. \n\nIncluding the finalnumber 98, which is not part of either arithmetic sequence, the total sequence is 22. \n\nThe next step is to add these two totalstogether to find the total number of numbers in the list. The total<br>number of terms in the original sequence is 23 + 23 + 1 = 47. \n\nThe number of numbers in the list is n1 + n2 = 23 + 22 = 45. \n\nThe<br>final solution is 47. final solution is 45.<br>N-11  S-2 Experience Accumulation:<br>Rollback N-3 -> N-0 :<br>The overall reasoning process is incorrect due to a mistake in Reasoning Step 1. \n\nIn<br>Reasoning Step 1, the proposed sequence pattern is incorrect. The pattern is not an arithmetic<br>sequence with a common difference of 4. Instead, the sequence alternates between<br>increments of 1 and 3. This pattern is not an arithmetic sequence but rather a combination of<br>two arithmetic sequences: one starting at 6 and increasing by 4 (6, 10, 14, ...), and another<br>starting at 7 and increasing by 4 (7, 11, 15, ...). \n\nTherefore, the equation set up to find the<br>total number of terms is incorrect. Instead, we should find the number of terms in each of the<br>two sequences and add them together. \n\nThe incorrect reasoning in Step 1 leads to incorrect<br>conclusions in Reasoning Step 2. The equation 4n + 2 = 98 is not valid because the sequence<br>pattern was incorrectly identified. Therefore, the solution n = 24 is also incorrect. \n\nA correct<br>reasoning process would involve identifying the two separate arithmetic sequences, finding the<br>number of terms in each, and adding the two totals together. This would result in a correct total<br>count of numbers in the given list.<br>Rollback N-10 -> N-4 :<br>The overall reasoning process is correct and the steps follow a logical and rational<br>progression. The approach of identifying the two arithmetic sequences and calculating the<br>number of terms in each is a valid method for solving this problem. However, there is a<br>mistake in the calculation in Reasoning Step 2.\n\nReasoning Step 1: The step is correct. The<br>problem is accurately identified as a combination of two arithmetic sequences and the correct<br>formula for finding the nth term of an arithmetic sequence is used. The equations for finding<br>the number of terms in each sequence are correctly set up.\n\nReasoning Step 2: This step<br>contains a mistake. While the first equation is solved correctly, there is an error in the<br>simplification of the second equation. The correct simplification should be 7 + 4n2 - 4 = 95,<br>which simplifies to 4n2 + 3 = 95, not 4n2 + 3 = 91. After subtracting 3 from both sides, we<br>should get 4n2 = 92, not 4n2 = 88. Dividing both sides by 4 gives n2 = 23, not<br>22.\n\nReasoning Step 3: This step correctly identifies the mistake in Reasoning Step 2 and<br>correctly calculates the number of terms in the second sequence as 22. However, as identified<br>above, this is incorrect. The correct number of terms in the second sequence is 23, not 22.<br>The total number of numbers in the list should be 23 + 23 = 46, not 45.\n\nSo, the final<br>solution should be 46, not 45. The mistake in Reasoning Step 2 led to an incorrect final<br>solution.<br>Rollback N-15 -> N-11 :<br>Q :<br>The overall reasoning process is correct and the steps follow a logical and rational<br>progression. The approach of identifying the two arithmetic sequences and calculating the Answer the question about the problem Algebra. After getting the final solution, place it after<br>number of terms in each is a valid method for solving this problem. However, there is a the sentence 'The final solution is' for readability.\n\nQuestion: Marissa is hiking a 12-mile<br>mistake in the calculation in Reasoning Step 2.\n\nReasoning Step 1: The step is correct. The trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If<br>problem is accurately identified as a combination of two arithmetic sequences and the correct she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does<br>formula for finding the nth term of an arithmetic sequence is used. The equations for finding N-11 S-2 -> N-17 S-3 Prompt: she need to walk the remaining distance? \n\nAnswer: Let's think step by step.<br>the number of terms in each sequence are correctly set up.\n\nReasoning Step 2: This step<br>contains a mistake. While the first equation is solved correctly, there is an error in the<br>simplification of the second equation. The correct simplification should be 7 + 4n2 - 4 = 95, Answer the question about the problem Counting & Probability. After getting the final solution, place it after the sentence 'The final solution is'<br>which simplifies to 4n2 + 3 = 95, not 4n2 + 3 = 91. After subtracting 3 from both sides, we for readability.\n\nExperience containing previously made mistakes:\n\n######### The 0-th Experience with Analysis #########\n\t<br>should get 4n2 = 92, not 4n2 = 88. Dividing both sides by 4 gives n2 = 23, not Experience of the rollback N-3 -> N-0 \n####################\n\n######### The 1-th Experience with Analysis #########\n\t<br>22.\n\nReasoning Step 3: This step correctly identifies the mistake in Reasoning Step 2 and Experience of the rollback N-10 -> N-4 \n####################\n\n######### The 2-th Experience with Analysis #########\n\t<br>correctly calculates the number of terms in the second sequence as 22. However, as identifiedabove, this is incorrect. The correct number of terms in the second sequence is 23, not 22.The total number of numbers in the list should be 23 + 23 = 46, not 45.\n\nSo, the final Experience of the rollback N-15 -> N-11mistakes during reasoning for the question.\n\n\nQuestion: How many numbers are in the list 6,7,10,11,14,15,\ldots,94,95,98? $$\n####################\n\nConsider the analysis in the above experience to avoid making similar<br>solution should be 46, not 45. The mistake in Reasoning Step 2 led to an incorrect final \n\nAnswer: Let's think step by step. Let's focus on carefully generating the next possible reasoning step for reasoning steps below.\n" .<br>solution. .<br>**----- End of picture text -----**<br>


Final solution of the reasoning path N-0 S-0 --> N-17 S-3 **:** 

\tReasoning Step 1: The sequence provided in the problem can be divided into two separate arithmetic sequences: one starting with 6 and increasing by 4 (6, 10, 14, ..., 94) and another starting with 7 and also increasing by 4 (7, 11, 15, ..., 95). \n\nNext, we need to find the total number of terms in each of these arithmetic sequences. The formula to find the nth term of an arithmetic sequence is a + (n - 1) * d, where a is the first term, n is the term number, and d is the common difference. \n\nFor the first sequence, the first term (a1) is 6, the last term is 94, and the common difference (d1) is 4. We can set up the equation 6 + (n1 - 1) * 4 = 94 to solve for n1. \n\nFor the second sequence, the first term (a2) is 7, the last term is 95, and the common difference (d2) is 4. We can set up the equation 7 + (n2 - 1) * 4 = 95 to solve for n2. \n\nThese two equations will give us the total number of terms in each sequence. \n\nWe should also remember to include the final number 98 in our count, as it is not part of either arithmetic sequence. \n\nThe total number of terms in the original sequence will then be n1 + n2 + 1. \n\tReasoning Step 2: We can solve the equations 6 + (n1 - 1) * 4 = 94 and 7 + (n2 - 1) * 4 = 95 to find the number of terms in each arithmetic sequence. \n\nFor the first equation, we can subtract 6 from both sides to get (n1 - 1) * 4 = 88. Then, we can divide both sides by 4 to get n1 - 1 = 22. Finally, we can add 1 to both sides to get n1 = 23. \n\nFor the second equation, we can subtract 7 from both sides to get (n2 - 1) * 4 = 88. Then, we can divide both sides by 4 to get n2 - 1 = 22. Finally, we can add 1 to both sides to get n2 = 23. \n\nTherefore, there are 23 terms in each arithmetic sequence. \n\nIncluding the final number 98, which is not part of either arithmetic sequence, the total number of terms in the original sequence is 23 + 23 + 1 = 47. \n\nThe final solution is 47. 

_Figure 10._ A complex thought structure built by GPT-4 with TR for the question from MATH dataset (Hendrycks et al., 2021b). This structure contains 17 nodes, i.e. 17 thoughts and leads to _K_ = 8 reasoning path towards 8 solutions as 8 number of rollbacks are triggered by the _rollback controller_ with GPT-4 during reasoning. It is plotted by based on the NetworkX lib (Hagberg et al., 2008) under the “fdp” layout. 

22 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0023-01.png)


**----- Start of picture text -----**<br>
Q :<br>N-28 S-6 :<br>Answer the question about the problem Math. After getting the final solution, place it after the<br>sentence 'The final solution is' for readability.\n\nQuestion: suppose $u=\arctan \frac{y}{x}$, The final solution is $\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial<br>what is numeric of $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}$? y^2} = 0$.<br>\n\nAnswer: Let's think step by step.<br>N-27 S-5 :<br>The next step is to sum the second order partial derivatives. We add<br>$\frac{\partial^2 u}{\partial x^2} = \frac{2xy}{(x^2+y^2)^2}$ and<br>$\frac{\partial^2 u}{\partial y^2} = \frac{-2xy}{(x^2+y^2)^2}$ to get<br>$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = \frac{2xy -<br>2xy}{(x^2+y^2)^2}$. This step involves basic algebraic addition and<br>simplification. After simplifying, we find that $\frac{\partial^2 u}{\partial x^2} +<br>\frac{\partial^2 u}{\partial y^2} = 0$. The final solution is 0.<br>$\frac{\partial^2 u}{\partial x^2} +<br>\frac{\partial^2 u}{\partial y^2}<br>Q : = \frac{2}{x^2 + y^2}$.<br>Answer the question about the problem Math. After getting the final solution,<br>place it after the sentence 'The final solution is' for readability.\n\nQuestion:<br>N-19 S-4 : suppose $u=\arctan \frac{y}{x}$, what is numeric of $\frac{\partial^2 u}{\partial<br>x^2}+\frac{\partial^2 u}{\partial y^2}$? \n\nAnswer: Let's think step by step.<br>The next step is to conclude the solution based on the result we obtained.<br>The sum of the second order partial derivatives of $u$ with respect to $x$<br>and $y$ is zero. Therefore, the final solution is 0.<br>N-17 S-6 :<br>The next step is to find the numeric of<br>$\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}<br>{\partial y^2}$. By adding the expressions for<br>$\frac{\partial^2 u}{\partial x^2}$ and<br>$\frac{\partial^2 u}{\partial y^2}$, we get<br>$\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}<br>{\partial y^2} = \frac{2y^3}{(x^2+y^2)^2} -<br>\frac{2x^3}{(x^2+y^2)^2}$. The final solution is then<br>the simplification of this expression..<br>N-11 S-7 :<br>The best next step is to simplify the sum of the second order<br>partial derivatives. The terms $\frac{2xy}{(x^2+y^2)^2}$ and<br>$\frac{-2xy}{(x^2+y^2)^2}$ cancel each other out, giving us<br>$\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} =<br>0$. Therefore, the numeric of $\frac{\partial^2 u}{\partial<br>x^2}+\frac{\partial^2 u}{\partial y^2}$ is 0. \n\nThe final solution is<br>0.<br>N-16 S-6 :<br>The next step is to add the two second order partial derivatives<br>$\frac{\partial^2 u}{\partial x^2}$ and $\frac{\partial^2 u}{\partial y^2}$ that we<br>calculated in the previous step. This will give us the numeric of<br>$\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}$.\n\nWe have<br>$\frac{\partial^2 u}{\partial x^2} = \frac{2xy}{(x^2+y^2)^2}$ and<br>$\frac{\partial^2 u}{\partial y^2} = -\frac{2xy}{(x^2+y^2)^2}$.\n\nAdding these<br>two gives us $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} =<br>\frac{2xy}{(x^2+y^2)^2} - \frac{2xy}{(x^2+y^2)^2} = 0$.\n\nSo, the final<br>solution is $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} = 0$.<br>**----- End of picture text -----**<br>


Final solution of the reasoning path N-0 S-0 --> N-11 S-7 **:** 

_\tTo begin with, we need to find the first order partial derivatives of $u$ with respect to $x$ and $y$. Using the chain rule, we can write the derivative of $u$ with respect to $x$ as $\frac{\partial u}{\partial x} = \frac{1}{1+(y/x)^2} \cdot -\frac{y}{x^2}$, and the derivative of $u$ with respect to $y$ as $\frac{\partial u}{\partial y} = \frac{1}{1+(y/x)^2} \cdot \frac{1}{x}$. \n\tThe next step is to simplify these first order partial derivatives. We can simplify $\frac{\partial u}{\partial x}$ and $\frac{\partial u}{\partial y}$ by multiplying the numerator and the denominator by $x^2$ to get rid of the complex fraction. After simplifying, we have $\frac{\partial u}{\partial x} = -\frac{y}{x^2+y^2}$ and $\frac{\partial u}{\partial y} = \frac{x}{x^2+y^2}$. \n\tThe next step is to find the second order partial derivatives of $u$ with respect to $x$ and $y$. Using the chain rule again, we can write the second derivative of $u$ with respect to $x$ as $\frac{\partial^2 u}{\partial x^2}$ and the second derivative of $u$ with respect to $y$ as $\frac{\partial^2 u}{\partial y^2}$. We need to differentiate $\frac{\partial u}{\partial x} = -\frac{y}{x^2+y^2}$ with respect to $x$ and $\frac{\partial u}{\partial y} = \frac{x}{x^2+y^2}$ with respect to $y$. \n\tThe next step is to calculate the second order partial derivatives. For $\frac{\partial^2 u}{\partial x^2}$, we differentiate $-\frac{y}{x^2+y^2}$ with respect to $x$, and for $\frac{\partial^2 u}{\partial y^2}$, we differentiate $\frac{x}{x^2+y^2}$ with respect to $y$. The differentiation should be done carefully using the quotient rule and simplifying the result. \n\tThe next step is to perform the differentiation to find the second order partial derivatives. Using the quotient rule, we have:\n\nFor $\frac{\partial^2 u}{\partial x^2}$, differentiating $-\frac{y}{x^2+y^2}$ with respect to $x$ gives us $\frac{\partial^2 u}{\partial x^2} = \frac{2xy}{(x^2+y^2)^2}$.\n\nFor $\frac{\partial^2 u}{\partial y^2}$, differentiating $\frac{x}{x^2+y^2}$ with respect to $y$ gives us $\frac{\partial^2 u}{\partial y^2} = \frac{-2xy}{(x^2+y^2)^2}$.\n\nThe final step would be to add these two second order partial derivatives to find the numeric of $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}$. \n\tThe next step is to calculate the sum of the second order partial derivatives. We add $\frac{\partial^2 u}{\partial x^2} = \frac{2xy}{(x^2+y^2)^2}$ and $\frac{\partial^2 u}{\partial y^2} = \frac{-2xy}{(x^2+y^2)^2}$ to get $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} = \frac{2xy}{(x^2+y^2)^2} + \frac{-2xy}{(x^2+y^2)^2}$. Simplifying this expression will give us the numeric of $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}$. \n\tThe best next step is to simplify the sum of the second order partial derivatives. The terms $\frac{2xy}{(x^2+y^2)^2}$ and $\frac{-2xy}{(x^2+y^2)^2}$ cancel each other out, giving us $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} = 0$. Therefore, the numeric of $\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}$ is 0. \n\nThe final solution is 0. ._ 

_Figure 11._ A complex thought structure built by GPT-4 with TR for the question from TheoremQA dataset (Chen et al., 2023b). This structure contains 28 nodes, i.e. 28 thoughts and leads to _K_ = 8 reasoning paths towards 8 solutions as 11 number of rollbacks are triggered by the _rollback controller_ with GPT-4 during reasoning. 

23 

**Toward Adaptive Reasoning in Large Language Models with Thought Rollback** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0024-01.png)


**----- Start of picture text -----**<br>
Q :<br>Answer the question about the problem Finance. After getting the final<br>solution, place it after the sentence 'The final solution is' for<br>readability.\n\nQuestion: What is the effective rates for 3% compounded<br>monthly? \n\nAnswer: Let's think step by step.<br>N-19 S-8 :<br>The next step is to convert the EAR from decimal form to percentage form.<br>We do this by multiplying the decimal by 100. So, we have:\n\nEAR = 0.0304<br>* 100\n\nThis calculation will give us the effective annual rate as a<br>percentage. The final solution is then the result of this multiplication. It's<br>important to note that we are using the correct value of 0.0025 for the<br>monthly interest rate, avoiding the previous error of using 0.025. The final<br>solution is EAR = 3.04%.<br>N-25 S-7 :<br>The next step is to subtract 1 from 1.0304 to get the EAR in decimal form.<br>After performing this operation, we have:\n\nEAR = 1.0304 - 1 =<br>0.0304\n\nNow, to convert this decimal to a percentage, we multiply by 100.<br>This will give us the effective annual rate as a percentage, which is the final<br>answer we're looking for. Let's perform this calculation:\n\nEAR = 0.0304 *<br>100\n\nThis calculation will give us the effective annual rate as a percentage.<br>The final solution is then the result of this multiplication.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback_images/icml_2024_Toward_Adaptive_Reasoning_in_Large_Language_Models_with_Thought_Rollback.pdf-0024-02.png)


Final solution of the reasoning path N-0 S-0 --> N-19 S-8 **:** 

_\tTo calculate the effective rate for 3% compounded monthly, we need to use the formula for the effective annual rate (EAR) which is:\n\nEAR = (1 + r/n)^(nt) - 1\n\nWhere:\nr = nominal interest rate (annual interest rate)\nn = number of compounding periods per year\nt = number of years\n\nIn this case, r = 3% or 0.03 (converted to decimal), n = 12 (as it's compounded monthly), and t = 1 (as we're calculating the effective annual rate).\n\nLet's substitute these values into the formula. \n\tThe next step is to substitute the values of r, n, and t into the formula. So, we have:\n\nEAR = (1 + 0.03/12)^(12*1) - 1\n\nThis simplifies the calculation and prepares us for the next step, which is to perform the actual computation. \n\tThe next step is to simplify and calculate the expression we obtained. Performing the operation inside the brackets first, we have:\n\nEAR = (1 + 0.0025)^12 - 1\n\nNow, we need to raise the sum to the power of 12 (according to the order of operations, exponents come before addition or subtraction). After that, we subtract 1 from the result to find the effective annual rate. \n\tThe next step is to calculate the value of (1 + 0.0025)^12 using the rules of exponents. Once we have this value, we will subtract 1 from it to get the effective annual rate. Let's perform this calculation:\n\nEAR = (1.0025)^12 - 1\n\nBy calculating this expression, we will obtain the effective annual rate for a 3% interest rate compounded monthly. It's important to note that we must be careful to avoid the calculation error made in the previous experience, where the incorrect value was used for the expression inside the brackets. This step ensures we are using the correct value of 0.0025 for the monthly interest rate, avoiding the previous error of using 0.025. \n\tThe next step is to compute the value of (1.0025)^12 using a calculator or any computational tool. After obtaining this value, subtract 1 from the result to get the EAR. The computation is as follows:\n\nEAR = (1.0025)^12 - 1\n\nAfter this step, we will have the effective annual rate expressed as a decimal. To convert it to a percentage, we will multiply the result by 100 in the next step. It's important to note that we are using the correct value of 0.0025 for the monthly interest rate, avoiding the previous error of using 0.025. \n\tThe next step is to compute the value of (1.0025)^12 and then subtract 1 from the result. Using a calculator or computational tool, we find that (1.0025)^12 is approximately 1.0304. Subtracting 1 from this result, we get:\n\nEAR = 1.0304 - 1\n\nThis gives us the effective annual rate in decimal form. To convert this to a percentage, we will need to multiply the result by 100 in the next step. It's important to note that we are using the correct value of 0.0025 for the monthly interest rate, avoiding the previous error of using 0.025. \n\tThe next step is to subtract 1 from 1.0304 to get the EAR in decimal form. After performing this operation, we have:\n\nEAR = 1.0304 - 1 = 0.0304\n\nNow, to convert this decimal to a percentage, we multiply by 100. This will give us the effective annual rate as a percentage, which is the final answer we're looking for. It's important to note that we are using the correct value of 0.0025 for the monthly interest rate, avoiding the previous error of using 0.025. \n\tThe next step is to convert the EAR from decimal form to percentage form. We do this by multiplying the decimal by 100. So, we have:\n\nEAR = 0.0304 * 100\n\nThis calculation will give us the effective annual rate as a percentage. The final solution is then the result of this multiplication. It's important to note that we are using the correct value of 0.0025 for the monthly interest rate, avoiding the previous error of using 0.025. The final solution is EAR = 3.04%. ._ 

_Figure 12._ A complex thought structure built by GPT-4 with TR for the question from TheoremQA dataset (Chen et al., 2023b). This structure contains 43 nodes, i.e. 43 thoughts and leads to _K_ = 8 reasoning paths towards 8 solutions as 15 number of rollbacks are triggered by the _rollback controller_ with GPT-4 during reasoning. 

24 

