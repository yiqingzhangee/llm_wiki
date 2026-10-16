Published as a conference paper at ICLR 2024 

# - LARGE LANGUAGE MODEL CASCADES WITH MIX - TURE OF THOUGHT REPRESENTATIONS FOR COST EFFICIENT REASONING 

**Murong Yue**[1] **, Jie Zhao**[2] **, Min Zhang**[3] **, Liang Du**[2] **, Ziyu Yao**[1] 1George Mason University 2Microsoft 3Virginia Tech 1 _{_ myue, ziyuyao _}_ @gmu.edu 2 _{_ zhaojie, liang.du _}_ @microsoft.com 3 _{_ minzhang23 _}_ @vt.edu 

## ABSTRACT 

Large language models (LLMs) such as GPT-4 have exhibited remarkable performance in a variety of tasks, but this strong performance often comes with the high expense of using paid API services. In this paper, we are motivated to study building an LLM cascade to save the cost of using LLMs, particularly for performing reasoning (e.g., mathematical, causal) tasks. Our cascade pipeline follows the intuition that simpler questions can be addressed by a weaker but more affordable LLM, whereas only the challenging questions necessitate the stronger and more expensive LLM. To realize this decision-making, we consider “answer consistency” of the weaker LLM as a signal of the question difficulty and propose several methods for the answer sampling and consistency checking, including one leveraging a mixture of two thought representations (i.e., Chain-of-Thought (Wei et al., 2022) and Program-of-Thought (Chen et al., 2022; Gao et al., 2023)). Through experiments on six reasoning benchmark datasets, with GPT-3.5-turbo and GPT-4 being the weaker and stronger LLMs, respectively, we demonstrate that our proposed LLM cascades can achieve performance comparable to using solely the stronger LLM but require only 40% of its cost. Our codes are available at https://github.com/MurongYue/LLM_MoT_cascade. 

## 1 INTRODUCTION 

Large language models (LLMs) such as GPT-4 have exhibited remarkable performance in reasoning tasks (Rae et al., 2021; Lewkowycz et al., 2022; Zhong et al., 2023). Because of the intensive computing resources required for training and hosting the LLMs for inference, many such LLMs are only accessible via paid API services, thus leading to high monetary costs. In this work, we are motivated to study strategies for reducing the costs of using LLMs while not sacrificing task performance, particularly for LLMs’ applications to reasoning tasks. 

Different types and versions of LLMs often come with different capabilities and costs. Typically, LLMs with better performance (termed “stronger LLMs”) are more expensive than those with relatively worse overall performance (termed “weaker LLMs”). For example, GPT-4 (OpenAI, 2023) is 30 times more expensive than GPT-3.5-turbo for the output tokens.[1] It thus implies a promising solution to cost-saving. That is, simple questions could be answered by the weaker but more affordable LLM, whereas only the difficult questions need to be tackled by the more expensive, stronger LLM. Drawing inspirations from here, Chen et al. (2023a) explored the idea of “LLM cascades”, where a question is always first answered by a weaker LLM, and then optionally routed to a stronger LLM when the the weaker LLM’s answer is not accepted (Figure 1). To decide this routing, this work suggested fine-tuning a smaller LLM to score each question along with its answer produced by the weaker LLM. While this approach could work for some tasks, in practice, we observed that it did not yield satisfying performance for intricate reasoning tasks. Intuitively, it is very challenging to evaluate the difficulty and the answer correctness of a reasoning question solely based on its literal expression, even with a large enough LLM, since the errors could be nuanced despite the reasoning paths appearing promising (Madaan et al., 2023). 

> 1 https://openai.com/pricing. 

1 

Published as a conference paper at ICLR 2024 

In this work, we proposed to devise this routing decision-maker from a different angle, i.e., the “answer consistency” of the weaker LLM (Wang et al., 2023). This is inspired by the observation that answers from the weaker LLM tend to be consistent in multiple sampling paths when the question is easy, but inconsistent when the question is hard. To implement this idea, we proposed two types of methods, a vote-based method that examines if the agreement of multiple answer samples on the majority-voted answer surpasses a pre-defined confidence threshold, and a verification-based method that checks if the majority-voted answers sampled from different prompts are consistent. To realize the two methods in reasoning tasks, we further investigated multiple strategies for answer sampling, including sampling from a single set versus two sets of task demonstrations. In particular, we proposed to leverage a “mixture of thought (MoT) representations”, which samples answers from both Chain-of-Thought (Wei et al., 2022, CoT) and Program-of-Thought (Chen et al., 2022; Gao et al., 2023, PoT) prompts, emulating how experts can provide diverse perspectives to the same question. This follows the same spirit of ensembling (Rokach, 2010), but is applied to developing LLM cascades for the first time. By pairing different sampling strategies with the two answer consistency checking methods (i.e., vote and verification), we end up with ten approaches to implementing the LLM cascade. 

To evaluate the proposed approaches, we conducted experiments on six reasoning datasets, covering mathematical, symbolic, and causal reasoning tasks, using GPT-3.5-turbo as the weaker LLM and GPT-4 as the stronger one. The experimental results demonstrated the effectiveness of LLM cascades based on answering consistency. That is, different approaches that we proposed for LLM cascades can generally achieve performance comparable to or even better than fully using the stronger LLM, while they require only half or less relative cost to the latter. In particular, our approaches based on a mixture of thought representations achieved comparable task performance with only 40% of the cost of GPT-4. Our results also underscored the effectiveness of sampling answers from diverse prompt settings, such as sampling from different task demonstrations or different thought representations. Our further analysis revealed that different prompt settings can often provide different opinions for the more complex questions while tending to be consistent for easier ones, which allows us to distinguish questions at different difficulty levels more accurately for cascade decision-making. Finally, we also compared our consistency-based approaches with fine-tuned smaller LLMs (Chen et al., 2023a) as well as other variants that make the routing decisions based on the literal expressions of the question and its answer. Our approaches exhibited strong advantages over all of them. 

## 2 LLM CASCADES FOR COST-EFFICIENT REASONING 

## 2.1 OVERVIEW OF LLM CASCADES 

We leverage a cascade of LLMs to save the cost of in-context LLM reasoning, as illustrated in Figure 1. Specifically, we assume two LLMs. The weaker LLM (denoted as _LLM[w]_ ) yields relatively worse performance but is less costly, whereas the stronger LLM (denoted as _LLM[s]_ ) enjoys better task performance but is more expensive. Given a question _Q_ , the LLM cascade first employs the weaker LLM to obtain an initial answer _A[w]_ . This answer, along with other 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0002-06.png)


**----- Start of picture text -----**<br>
Final Answer<br>Answer Accept<br>Query LLM Cascade Answer<br>Weaker Decision Maker Reject Final Answer<br>LLM<br>Stronger<br>LLM<br>**----- End of picture text -----**<br>


Figure 1: Illustration of LLM cascade chaining a weaker but cheaper LLM with a stronger but more costly one. 

metadata produced by the weaker LLM, will then be fed to a cascade decision maker to decide whether the answer can be accepted as the final one. If the answer is rejected, the stronger LLM should be invoked to provide a more reliable answer _A[s]_ . As a consequence, the total cost of answering the question becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0002-09.png)


where _C[w]_ and _C[s]_ indicate the costs from calling the weaker and the stronger LLMs, respectively, _C[d]_ denotes any cost involved in the LLM cascade decision-making process, and 1 _reject_ = 1 holds if and only if the decision maker rejects the answer. 

Both LLMs solve the question via few-shot in-context learning, e.g., for the weaker LLM, an answer _A[w]_ is produced by sampling from _PLLM w_ ( _A[w] | E_ 1 _||E_ 2 _|| . . . ||EM ||Q_ ), where _E_ 1 _||E_ 2 _|| . . . ||EM ||Q_ denotes a concatenation of _M_ task demonstrations and the input question _Q_ , forming the “prompt input” to the LLM (Brown et al., 2020). As _M_ task examples are used to demonstrate the task, 

2 

Published as a conference paper at ICLR 2024 

it indicates a “ _M_ -shot in-context learning” of LLM. For reasoning tasks, in practice, a LLM will typically be prompted to elaborate on its reasoning process via “thought representations”, such as Chain-of-Thought (Wei et al., 2022, CoT) and Program-of-Thought (Chen et al., 2022; Gao et al., 2023, PoT), where the reasoning process is described step by step via natural language and programming language, respectively. The answer (e.g., a numerical result of the mathematical calculation) can then be extracted from the texts (for CoT) or obtained by executing the code (for PoT). 

## 2.2 ANSWER CONSISTENCY-BASED CASCADE DECISION-MAKING 

The core of our LLM cascade is the decision maker, which takes in the output from the weaker LLM, and then decides whether to route to the stronger LLM or not. An ideal cascade decision maker should call the stronger LLM only when the answer by the weaker LLM is wrong, such that the total cost _C_ can be minimized without degrading the overall task performance (compared with using the stronger LLM all the time). To this end, we propose two methodologies based on the “answer consistency” of the weaker LLM, which we elaborate on below. 

**Answer Consistency and Sources of Sampling** Answer consistency has been found helpful for improving the LLM performance in reasoning tasks (Wang et al., 2023). Instead of greedily decoding one answer for each question, Wang et al. (2023) sampled a diverse set of reasoning paths (or thought processes) and then selected the most consistent answer by marginalizing out the sampled paths. Drawing inspiration from the prior work, we make the following hypothesis: When the weaker LLM samples highly consistent answers for a given question, it reveals a high “confidence” in solving this question and its most consistent answer is likely to be correct; in this case, there is thus no need to invoke the stronger LLM. 

To realize this intuition, we generalize from Wang et al. (2023) and consider three sources of sampling consistency: 

- **In-distribution sampling** : As Wang et al. (2023), we consider sampling multiple answers given the same prompt input to the weaker LLM. In practice, this can be achieved by setting a non-zero temperature for the weaker LLM. 

- **Sampling from different in-context demonstrations** : We further consider sampling answers from two sets of task demonstrations under the same thought representation. For example, to demonstrate the CoT process in mathematical reasoning tasks, Wei et al. (2022) annotated eight math examples as the demonstrations and performed 8-shot in-context learning. We additionally annotated another eight examples as the second set of demonstrations, which allowed us to further diversify the sources of answer sampling. 

- **Sampling from different thought representations** : While existing literature typically investigated either CoT or PoT independently, in this work, we propose to leverage the synergy of both thought representations in a single task. We hypothesize that an LLM obtains truly high confidence in its problem-solving, only when it is able to produce a consistent answer agnostic to how the intermediate steps are represented. Therefore, we propose to sample the weaker LLM answers from a “mixture of thought (MoT) representations”, which includes both CoT and PoT prompts. 

Below, we introduce our methodologies for consistency checking based on the answer samples. 

**Method 1: Vote-based decision-making** The first method calculates the consistency of the weaker LLM’s answer samples by voting. Formally, for a single prompt, we denote the set of answers produced by the weaker LLM for each question _Q_ as ( _A[w]_ 1 _[, A][w]_ 2 _[, ..., A][w] K_[)][,][where] _[K]_ is the pre-defined number of samples. When sampling from two different prompts, we denote ( _A[w]_ 11 _[, A]_ 12 _[w][, ..., A][w]_ 1 _K_ 1[)][and][(] _[A][w]_ 21 _[, A][w]_ 22 _[, ..., A][w]_ 2 _K_ 2[)][as][the][answer][samples][produced][by][each][of][them,] where _K_ 1 and _K_ 2 represent the pre-defined sample size for each prompt setting, respectively. Note that for this method, we do not distinguish answers sampled with a single prompt or multiple prompts (e.g., samples from different prompts have exactly equal weights when voting). The most consistent answer can then be selected as the one that most samples agree with, and this answer will also be regarded as the final answer _A[w]_ by the weaker LLM. The decision maker measures the weaker LLM’s consistency via the agreement score 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0003-11.png)


3 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0004-01.png)


**----- Start of picture text -----**<br>
Query  "Joelle has 5 orchids and 4 Africanmany petals do the daisies havedaisies on her balcony...How CoT-1D-Vote Demo1 CoT-2D-Vote Demo1 Demo2 Majority<br>Question CoT: Kobe and Pau went to compared to the orchids?" ... ... ... Vote<br>a restaurant... PoT-1D-Vote Demo1 PoT-2D-Vote Demo1 Demo2<br>Answer : Pau ordered 5 x 2 = 10 Consistency<br>fried chickens in total.Therefore, ... ... ... Measure<br>Pau ate 10 x 2 = 20 pieces of fried Y Accept<br>chicken. Ans = 20 MoT-1D   /2D       -Vote Demo1 Demo1 Demo2<br>N Reject<br>Question : Joelle has 5 orchids and ... ... Vote-based<br>4 African daisies on her balcony...<br>Answer :<br>... Demo1 CoT-2D-Verify Demo2 ...<br>PoT Weaker<br>#Question:  Kobe and Pau went to LLM ... Demo1 PoT-2D-Verify Demo2 ...<br>a restaurant... ... Demo1 MoT-1D-Verify Demo1 ...<br>#Python code, return ans<br>kobe_order=5 ... Demo1 MoT-2D-Verify Demo2 ...<br>pau_order=kobe_order*2<br>pau_eaten =2*pau_order<br>#Question:  Joelle has 5 orchids and Majority Majority<br>4 African daisies on her balcony... Vote Y N Vote<br>#Python code, return ans Accept Reject Verify-based<br>**----- End of picture text -----**<br>


Figure 2: An overview of our approaches (6 vote-based and 4 verification-based). We use to represent the answers from PoT and to represent the answers from CoT. _Demoi_ is the _i_ -th set of demonstrations. 

The larger the _s_ , the more consistent the weaker LLM’s answer samples. In conjunction with a pre-defined threshold value _τ_ , the decision maker accepts the weaker LLM’s most consistent answer _A[w]_ when _s ≥ τ_ and rejects it otherwise. As a result, the total cost of answering a question (Eq 1) can vary depending on the threshold. 

**Method 2: Verification-based decision-making** In the case of producing samples from two different prompt settings (i.e., different demonstrations or thought representations), we propose the second method, which compares the most consistent answers produced by each prompt as the answer verification. As previously mentioned, we could obtain two sets of answers from distinct prompts: ( _A[w]_ 11 _[, A][w]_ 12 _[, ..., A][w]_ 1 _K_ 1[)][ and][ (] _[A]_ 21 _[w][, A][w]_ 22 _[, ..., A][w]_ 2 _K_ 2[)][.][Our method then verifies the most consis-] tent answers within each prompt, denoted as _A[w]_ 1 _[′]_[and] _[ A][w]_ 2 _[′]_[respectively, as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0004-05.png)


Only when _s_ equals 1, i.e., when the two answers are the same, the weaker LLM’s answer will be accepted by the decision maker. In this case, the final answer of the weaker LLM will be the same as the two most consistent answers, i.e., _A[w]_ = _A[w]_ 1 _[′]_[=] _[ A][w]_ 2 _[′]_[.] 

In comparison, these two methods have different applicable scenarios. The vote-based method is well-suited for scenarios with pre-defined cost constraints. In such cases, we have the flexibility to tune the threshold to ensure it aligns with the constraint. On the other hand, the verification-based method is capable of producing relatively optimal results without the need for threshold tuning, although it lacks flexibility. We will systematically compare the two methods in experiments. 

## 2.3 LLM CASCADES FOR REASONING TASKS 

We instantiate the proposed two methods in LLM reasoning tasks with different sampling sources, resulting in 10 approaches, as summarized in Figure 2. Specifically, 6 approaches adopt vote-based decision-making: CoT-1D-Vote collects _K_ answers sampled from prompting the weaker LLM with the CoT representation, and then calculates the answer consistency for decision-making, following Eq 2. Similarly, PoT-1D-Vote bases its decision-making on answers sampled from a PoT prompt. To diversify the sources of the answers, for each thought representation, we further consider sampling from two sets of CoT or PoT demonstrations, resulting in CoT- _2D_ -Vote and PoT- _2D_ -Vote, respectively. Finally, the vote-based approaches also include two variants leveraging a mixture of thought (MoT) representations. For MoT-1D-Vote, _K_ 1 answers are sampled from the CoT prompt and another _K_ 2 from the PoT prompt, and a union set of their answers are then used to compute the consistency score _s_ . For MoT-2D-Vote, the procedure is similar, except that the CoT and the PoT prompts are annotated from two sets of demonstration examples. 

The verification-based approaches assume answer samples from two different prompts. We instantiate 4 variants, including CoT-2D-Verify, where we prompt the weaker LLM with 

4 

Published as a conference paper at ICLR 2024 

two sets of CoT demonstrations, resulting in two answer sets for decision-making (Eq 3); PoT-2D-Verify, where we similarly prompt the weaker LLM with two sets of PoT demonstrations; MoT-1D-Verify, where we consider two sets of answers from two thought representation prompts (but on the same set of task demonstration examples); and MoT-2D-Verify, which additionally employs different sets of demonstrations when prompting the weaker LLM with different thought representations. 

**Cost-Comparable Sample Size Configuration** To fairly compare all approaches in terms of their effectiveness in identifying easy (or correct) vs. hard (or correct) questions, we aim to configure their costs to be comparable. Since we use the same prompt to the stronger LLM, _C[s]_ is agnostic to the specific approach. However, different approaches may require varying costs in calling the weaker LLM (i.e., _C[w]_ ). For example, even with the same total sampling size (i.e., _K_ = _K_ 1 + _K_ 2), CoT-1D-Vote and CoT-2D-Vote result in different input token usages because the former prompt the LLM once but the latter prompt it twice, doubling the token usages. Therefore, we unify the _C[w]_ by a coarse-grained cost alignment. Our analysis suggests different configurations of _K_ for different approaches. We refer readers to Appendix A for more details. Since our configuration can only yield comparable costs due to certain simplifications, rather than exact costs, we still report the actual token cost for each approach. 

## 3 EXPERIMENT 

## 3.1 EXPERIMENTAL SETTING 

We evaluate our LLM cascade approaches on six datasets, covering (1) mathematical reasoning, including GSM8k (Cobbe et al., 2021), ASDIV (Ling et al., 2017), and TabMWP (Lu et al., 2023); (2) symbolic reasoning from BIG-Bench Hard (bench authors, 2023), including DATE and Navigate; and (3) causal reasoning, including CREPE (Zhang et al., 2023). In our pipeline, we leverage the GPT-3.5-turbo (4k context) as the weaker LLM and the GPT-4 (8k context) with CoT selfconsistency (Wang et al., 2023, SC) as the stronger LLM. Throughout our experiments, we set the number of task demonstrations as _M_ = 8. We use the same demonstration examples as prior work (Chen et al., 2022; Wei et al., 2022). When additional task demonstrations are needed (e.g., for 2D approaches), we randomly sample examples from the training data and manually annotate them with thought representations. We set the number of sampling paths as _K_ = 20 for GPT-3.5turbo and _K_ = 3 for GPT-4. The sampling temperature by default is 0 _._ 4 for both LLMs. The metrics we use are the task accuracy and the relative cost compared with the cost of GPT-4 with CoT SC (denoted as GPT-4-CoT-SC). A lower relative cost and higher accuracy indicate better performance. We also compared our approaches with baselines using only the weaker LLM (i.e., GPT-3.5-CoT-SC, GPT-3.5-PoT-SC) or only the strong LLM in different ways (i.e., GPT-4-CoT-Greedy, GPT-4-PoT-Greedy, GPT-4-CoT-SC, GPT-4-PoT-SC). Our experimental details can be found in Appendix B and reproducible prompts in Appendix L. 

## 3.2 MAIN RESULTS 

Figure 3 illustrates the performance of our proposed approaches. For Vote-based approaches, we draw curves by changing the pre-defined threshold _τ_ varying from 0.4 to 1. A high value of threshold signifies a more rigorous criterion for trusting the answers from the weaker LLM, making more examples transferred to the stronger LLM. Our observations are as follows: 

**Our pipeline achieves comparable task performance with significantly reduced costs.** On average, all of our cascade variants (Vote or Verify) demonstrate significant cost efficiency. In particular, as shown in the average plot, the four MoT variants achieve comparable task performance ( _∼_ 0.929 accuracy) to GPT-4-CoT-SC (0.931) while demanding only 40% of its cost. On CREPE, MoT variants even outperform GPT-4-CoT-SC (0.885 vs. 0.871) at 47% of its cost. Our approaches based on CoT and PoT also exhibit the capability to save costs while maintaining the overall task performance. For example, CoT-2D-Vote achieved 0.924 task accuracy on average but demanded only 57% relative cost. These observations suggest the effectiveness of our cascade decision maker via checking the answer consistency of the weaker LLM. 

**Sampling from diverse prompt settings helps cascade decision-making.** Our results show that variants involving diverse sources of sampling, such as CoT/PoT-2D-Vote and MoT-1D/2D-Vote, can more precisely distinguish between easy and hard reasoning questions, 

5 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0006-01.png)


**----- Start of picture text -----**<br>
GSM8k ASDIV TabMWP<br>0.95 0.94 0.96<br>0.94<br>0.92<br>0.90 0.92<br>0.90 0.90<br>0.85 0.88 0.88<br>0.86<br>0.80 0.86<br>0.84<br>0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0<br>Relative Cost Relative Cost Relative Cost<br>DATE Navigate CREPE<br>0.90 0.98<br>0.96<br>0.85 0.85<br>0.94<br>0.80 0.92 0.80<br>0.75 0.90<br>0.88 0.75<br>0.70<br>0.86<br>0.2 0.4 0.6 0.8 1.0 1.2 1.4 0.2 0.4 0.6 0.8 1.0 1.2 0.2 0.4 0.6 0.8 1.0 1.2 1.4<br>Relative Cost Relative Cost Relative Cost<br>Average Legend<br>0.92 GPT-3.5-CoT-SC CoT-2D-Vote<br>GPT-3.5-PoT-SC PoT-2D-Vote<br>0.90 GPT-4-CoT-SC MoT-2D-Vote<br>0.88 GPT-4-PoT-SC CoT-2D-Verify<br>0.86 GPT-4-CoT-Greedy PoT-2D-Verify<br>0.84 GPT-4-PoT-GreedyCoT-1D-Vote MoT-1D-VerifyMoT-2D-Verify<br>0.82 PoT-1D-Vote<br>MoT-1D-Vote<br>0.2 0.4 0.6 0.8 1.0<br>Relative Cost<br>Task Accuracy Task Accuracy Task Accuracy<br>Task Accuracy Task Accuracy Task Accuracy<br>Task Accuracy<br>**----- End of picture text -----**<br>


Figure 3: Main experiment results over six reasoning datasets. The bottom figure represents the average performance. The exact numerical results are included in Appendix C. 

compared with their counterparts sampling from single sources, i.e., CoT/PoT-1D-Vote. For example, between CoT-2D-Vote and CoT-1D-Vote, the former outperforms the latter by 1.4% absolute accuracy under the same relative cost of 0.4 on average. 

**Mixing thought representations is particularly effective.** Furthermore, we find that mixing the two thought representations (i.e., MoT-1D/2D-Vote) outperforms decision-making using either of them (i.e., CoT-1D/2D-vote and PoT-1D/2D-vote). This is illustrated by the gap in the average plot and is consistent on most datasets except DATE, where many test questions are very similar to the demonstration examples. Intuitively, this is because different thought representations can bring in more diverse “opinions” of the weaker LLM on the same input question, resembling how a group of experts with diverse perspectives could contribute to more effective results in collaborative work. It can also be viewed as “ensembling” LLMs, which utilizes the intuition that variants of the same model typically share few mistakes (Rokach, 2010). We provide a further investigation of this effect in Section 3.3. We also note that when using MoT, no obvious difference is perceived between using one set (i.e., MoT-1D-Vote) or two sets (i.e., MoT-2D-Vote) of task demonstrations. This result reveals that tuning the thought representations is more helpful for measuring an LLM’s (un)certainty on its answer than tuning the task demonstrations. 

**Increasing the threshold yields marginal benefits for MoT-1D/2D-Vote.** As costs increase, the curves of MoT-1D/2D-Vote flatten out, showing that pushing the threshold to exceedingly high is unnecessary. This is because even for an easy question, the weaker LLM may still have hallucinations in a small set of answers. Setting the threshold too high can lead to the decision-making being influenced by them. It may result in easy questions being sent to the stronger LLM incorrectly, thus driving up the overall cost. In our practice, a threshold that can balance cost and accuracy typically falls between 0.5 and 0.6. When it comes to MoT-1D/2D-Verify, we can strike a balance between cost and accuracy as the verification method can tolerate a few hallucinations. 

## 3.3 ANALYSIS ON MIXTURE OF THOUGHT REPRESENTATIONS 

To understand the effect of MoT, we analyze three vote-based approaches. We first group questions into “easy” and “hard” based on whether the weaker LLM can answer them correctly (i.e., whether the majority-voted answer _A[w]_ is correct or not). For all easy/hard questions, we then calculate the average consistency score following Eq 2. We present the average consistency scores for each approach, categorizing easy and hard questions based on whether the weaker LLM can answer them correctly (i.e., _Aw_ being correct). A higher average consistency score indicates our decision-maker places greater trust in the answer from the weaker LLM. Hence, an effective cascade decision- 

6 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Question : Blake and Kelly are having a contest to see who can run the most in 15 minutes. They decide to do it on a football field that is 100<br>yards long. Blake runs back and forth 15 times. Kelly runs back and forth once, and then decides that she doesn't want to run next to Blake, so<br>she starts to run to the 40-yard line and back. She does this 34 times. How much farther does the winner run than the loser?<br>Gold CoT 1 PoT 1<br>Black runs 200 yards for each back and Answer: [...(Ignored)] Kelly runs to the 40- # Python code, return ans<br>forth because...He runs 3000 yards in total yard line and back 34 times, so she runs [...(Ignored)]<br>because...Kelly runs 200 yards at first 34*80=2720 yards. Thus, Blake runs 3000- distance_covered_by_kelly =<br>because 100 x 2 = 200. She runs 80 yards 2720=280 yards farther than Kelly. (length_of_football_field *<br>each time she does a back and forth to the40-yard line because... She runs 2720 from CoT 2 num_of_laps_for_kellynum_of_laps_for_kelly * 2) * 2) + (40 *<br>these back and forth. She runs 2920 in total Answer: [...(Ignored)] Kelly runs 34 x (40 x 2) ans = abs(distance_covered_by_blake<br>because...The winner ran 80 yards more = 2720 yards. Blake runs 3000 - 2720 = 280 - distance_covered_by_kelly)<br>because 3000 - 2920=80. yards farther than Kelly. (Answer via Python execution: 6520.0)<br>Logic Generation Error Value Grounding Error<br>**----- End of picture text -----**<br>


Figure 4: Average consistency scores (Top) and case studies (Bottom; simplified for presentation purposes) of various approaches showing the advantage of MoT. Shades highlight similar mistakes by CoT even based on different task demonstrations, and dissimilar mistakes by PoT. 

maker should reveal relatively higher consistency scores for easy questions and lower ones for hard questions, leading to a larger ”gap” between them. 

We show the results in Figure 4 (Top). It is observed that all approaches lead to higher consistency scores on the easy questions than those on the hard questions, which explains their overall effectiveness for cascade decision-making. Because of involving two different thought representations, MoT-1D-Vote tends to have a lower consistency score compared with the two CoT approaches. However, it still ends up with a larger “gap” in the consistency scores for the easy and the hard questions, which is particularly prominent on the Navigate dataset where it gives the best performance gain (Figure 3). In contrast, CoT-1D-Vote results in the smallest score gap, indicating its weakness in distinguishing between the easy and the hard questions, particularly for Navigate. This weakness is mitigated by diversifying the prompting with two sets of task demonstrations (i.e., CoT-2D-Vote), but it still underperforms mixing the thought representations. 

Finally, in Figure 4 (Bottom) we show that when CoT cannot answer a hard question, prompting the weaker LLM with another set of task demonstrations (but still under the CoT representation) often yields the same mistaken answer, which thus results in a high consistency score. On the contrary, PoT tends to make mistakes in a very different way and result in a different mistaken answer, which explains the low consistency score of MoT-1D-Vote. More cases are shown in Appendix D. 

## 3.4 ROBUSTNESS EVALUATION 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0007-07.png)


**----- Start of picture text -----**<br>
GSM8k variant T Date variant T CREPE variant T<br>0.96 0.90 0.89<br>CoT-2D-Vote T=0.4<br>0.94 0.88 MoT-1D-Vote T=0.4<br>0.85 CoT-2D-Vote T=0.8<br>0.92 0.87 MoT-1D-Vote T=0.8<br>0.80 0.86<br>0.90 0.85<br>0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8 1.0<br>Relative Cost Relative Cost Relative Cost<br>GSM8k variant K Date variant K CREPE variant K<br>0.90<br>0.88 CoT-2D-Vote K=20<br>0.94 MoT-1D-Vote K=20<br>0.85 0.87 CoT-2D-Vote K=40<br>0.92 0.86 MoT-1D-Vote K=40<br>0.80<br>0.85<br>0.90<br>0.2 0.4 0.6 0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8<br>Relative Cost Relative Cost Relative Cost<br>Task Accuracy Task Accuracy Task Accuracy<br>Task Accuracy Task Accuracy Task Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Robustness analysis with varying temperature _T_ and sample size _K_ . 

7 

Published as a conference paper at ICLR 2024 

We further analyze if our results are sensitive to the change of the sampling temperature _T_ or the sample size _K_ . We select datasets for each type of reasoning task and conduct experiments with CoT-2D-Vote and MoT-1D-Vote. Our results are shown in Figure 5. We first look into the effect when increasing the sampling temperature _T_ from 0.4 (our default setting) to 0.8. For both approaches, increasing their temperature yields comparable or slightly better performance. This is owing to the increased answer diversity when the temperature gets higher. However, in any case, MoT-1D-Vote consistently outperforms CoT-2D-Vote. Increasing the sample size _K_ from 20 to 40, on the other hand, leads to a rightward shift of the curves, implying that it requires higher cost to achieve the same task accuracy. This can be explained by the higher cost of the larger sample size, whereas increasing the sample size does not contribute to the detection of easy vs. hard questions. Like in the results of varying temperatures, MoT robustly outperforms CoT in any case. 

## 3.5 COMPARISON TO EXTERNAL TEXT-BASED VERIFIERS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0008-03.png)


**----- Start of picture text -----**<br>
GSM8k DATE CREPE<br>0.90<br>0.95 0.88 GPT-3.5-CoT-SC<br>0.85 GPT-4-CoT-SC<br>MoT-1D-Vote<br>0.90 0.80 0.86 CoT-2D-Vote<br>MoT-1D-Verify<br>0.75 0.84 LLM-Q<br>0.85 0.70 0.82 LLM-QAFinetuned-Q<br>0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0 Finetuned-QA<br>Relative Cost Relative Cost Relative Cost<br>Task Accuracy Task Accuracy Task Accuracy<br>**----- End of picture text -----**<br>


Figure 6: Comparison with external verifiers showing the advantages of our approaches based on answer consistency. We do not report finetuned verifiers for DATE as it does not have a training set. 

As mentioned, prior work implemented the LLM cascade decision-maker by training an external verifier, which scores a question and its answer (from the weaker LLM) based on their literal descriptions (Chen et al., 2023a; Sakota et al., 2023).[ˇ] Related to the above work, Chen et al. (2023b) and Madaan et al. (2023) also showed the promise of prompting LLMs to evaluate their own responses. To perform a general comparison with such external verifiers, we conducted experiments on GSM8k, DATE, and CREPE with the following baselines: Finetuned-Q, which is a RoBERTabase model (Liu et al., 2020) fine-tuned to decide whether a question should be routed to the stronger LLM based on its description; Finetuned-QA, which works similarly as Finetuned-Q but additionally takes the majority-voted answer from the weaker LLM (GPT-3.5-CoT-SC) as input; LLM-Q, where we instead prompt GPT-3.5-turbo as the verifier to judge based on the question description; and LLM-QA, which similarly employs GPT-3.5-turbo to decide upon the question and the weaker LLM’s majority-voted answer. We leave details of the baselines in Appendix E. 

The results in Figure 6 show that incorporating the external verifiers cannot achieve comparable accuracy with GPT-4-CoT-SC. For example, on the GSM8k dataset, the highest accuracy with the external verifiers is 0.892, which is way lower than the accuracy (0.958) of GPT-4-CoT-SC and the accuracy (0.951) of our approaches. They also show lower task accuracies than our approaches under the same cost. It indicates that the external verifiers cannot yield satisfying results in complex reasoning tasks, which can be due to the intrinsic challenge of deciding question difficulty and answer correctness solely based on their textual descriptions. For the calibration analysis, please refer to the appendix I. 

## 3.6 ADDITIONAL STUDIES 

**How weak can the weaker LLM be?** We evaluate our approaches when adopting LLAMA2-13B (Touvron et al., 2023) as the weaker LLM and GPT-4 as the stronger LLM. The results are shown in the Appendix F. On the DATE dataset, our approaches still works. However, LLAMA2-13B as the weaker LLM doesn’t yield ideal results on GSM8k and CREPE. That is because most of the questions in GSM8k and CREPE are excessively complex for LLAMA2-13B. Therefore, LLAMA213B often fails to answer the questions consistently across multiple samplings. Hence, the choice of a weaker LLM should be contingent on the task’s level of difficulty. When the current weaker LLM struggles with the task, it is advisable to consider switching to a more powerful LLM. 

**Can the stronger LLM benefit from the weaker LLM hints?** In our LLM cascade, all questions will obtain answers from the weaker LLM, no matter if they will be sent to the stronger LLM 

8 

Published as a conference paper at ICLR 2024 

or not. Therefore, an interesting question is whether the answer produced by the weaker LLM (correct or incorrect) can provide “hints” to enhance the stronger LLM. To answer this question, we experimented with MoT-1D-Verify, where we additionally passed the two inconsistent answers _A[w]_ 1 _[′]_[and] _[ A][w]_ 2 _[′]_[as hints to the stronger LLM following the format of prior work (Zheng et al., 2023).] We observed that the hints can only yield slight improvement on DATE, but greatly hurt the model performance on GSM8k and CREPE. Therefore, we conclude that hints from the weaker LLM, when it is uncertain about the question, do not help the stronger LLM. Details are in Appendix G. 

**Can our method generalize to factual-based tasks?** We also explored whether our method can be generalized to factual-based reasoning tasks in Appendix J. 

## 4 RELATED WORK 

Earlier research has delved into techniques to enhance the cost efficiency of LLMs, including quantization, pruning, and decoding methods(Bai et al., 2022; Kurtic et al., 2023; Leviathan et al., 2022). While many LLMs are closed-source, certain studies have concentrated on how to utilize the API efficiently. Chen et al. (2023a) proposed an LLM cascade, which sends a query to a list of LLMs sequentially if the answers provided by the prior LLMs are regarded unacceptable. A concurrent effort is made by Sakota et al. (2023), where the proposed approach determines which LLM to in-[ˇ] voke prior to sending the query to any LLM. Both of these approaches employed external verifiers fine-tuned from smaller LMs and therefore demanded substantial training data and additional model learning. In our experiments, we showed the ineffectiveness of these approaches when they were used to save costs for complex reasoning tasks, which echos the discovery of the shortcomings of LLM fine-tuning in recent work (Ni et al., 2023). In contrast, we proposed a novel solution for LLM cascades based on the answer consistency of the weaker LLM, which is training-free and is not limited to a specific weaker LLM. 

**Reasoning with LLMs** LLMs have demonstrated strong capability in solving reasoning tasks (Rae et al., 2021; Lewkowycz et al., 2022; Zhong et al., 2023). The standard few-shot incontext learning for LLMs is to feed the question to the LLM and let it generate the answer directly (Brown et al., 2020). Wei et al. (2022) then proposed the Chain-of-Thought approach, showing that prompting LLMs to think step by step can significantly improve their performance in solving reasoning problems, as it activates the built-in reasoning capabilities of LLMs (Madaan & Yazdanbakhsh, 2023). To further facilitate the mathematical steps in a reasoning task, Chen et al. (2022) and Gao et al. (2023) proposed the Program-of-Thought prompting, using code as the intermediate step and executing it with an external interpreter to entangle reasoning and numerical calculation. While most existing works studied CoT and PoT independently as alternative approaches, several other works have delved into the synergy between them, e.g., integrating code into CoT (Cheng et al., 2023b; He-Yueya et al., 2023), ensembling algebraic and code representations (Imani et al., 2023; Yue et al., 2023), or performing automatic selection between CoT and PoT (Zhao et al., 2023). Similar to these works, our approaches employ CoT and PoT jointly; however, unlike them, we leverage this synergy for the novel application of cost-saving for LLM reasoning. On the other hand, like our LLM cascades, there are also recent works building pipelines chaining multiple LLMs (Varshney & Baral, 2022; Xiong et al., 2023a; Li et al., 2023; Lin et al., 2023a), but these works have a different focus on improving the task accuracy, rather than reducing the LLM costs. 

**Uncertainty of LLMs** Our approach involves measuring the uncertainty of the results of LLMs. Multiple concurrent works have investigated using the voting score to evaluate the answer’s uncertainty (Xiong et al., 2023b; Si et al., 2023; Cai et al., 2023). However, they dismissed the potential application for different representations of intermediate steps in the evaluation. 

## 5 CONCLUSION 

We introduce a simple yet efficient and universally applicable economical pipeline to dynamically decide the LLMs in reasoning tasks, so as to save the token costs. Our approach based on checking the answer consistency of the weaker LLM is novel and effective. Our discoveries emphasize that leveraging prompts with a mixture of thought representations in weaker LLM achieves the best performance as it introduces diverse answers. Compared with fully employing the stronger LLM, our pipeline requires approximately 40% of expenses to achieve a comparable result. Future works are listed in Appendix K. 

9 

Published as a conference paper at ICLR 2024 

## 6 ETHICS STATEMENT 

Our research endeavors to build a cost-efficient pipeline in solving reasoning problems in the LLM era. The significance of this work holds the potential to benefit a diverse range of organizations, particularly those with limited financial resources such as local businesses, educational institutions, and non-profit organizations. By devising cost-effective solutions, we empower these resourceconstrained entities to harness the reasoning ability of LLMs cheaply, thereby fostering fairness and inclusivity within the NLP community. This aligns with the broader goal of ensuring that advancements in NLP do not remain exclusive to a select few, but are accessible to a wider audience, regardless of their financial constraints. 

Moreover, for these industry giants, the cost savings facilitated by our findings can also be substantial, particularly when facing extremely large throughputs. By optimizing cost-efficiency, our research contributes to not only economic savings but also environmental sustainability by mitigating the carbon footprint associated with running large-scale computations. 

## 7 REPRODUCIBILITY STATEMENT 

Our pipeline is simple to implement and reproducible. We have documented all the experimental details, both in the main text and the appendix sections. While we cannot include the complete text of every prompt due to their excessive length, we do provide examples of each prompt in Appendix L, facilitating the reader’s comprehension of the style employed. All of our implementations (including the complete prompt scripts, the code for training external verifiers, the code for approach evaluation, etc.) are publicly available at https://github.com/MurongYue/LLM_MoT_cascade. 

## 8 ACKNOWLEDGEMENT 

This project was supported by Microsoft. It was also partially supported by resources provided by the Office of Research Computing at George Mason University (URL: https://orc.gmu. edu) and funded in part by grants from the National Science Foundation (Awards Number 1625039 and 2018631). Murong and Ziyu appreciate the funding support from George Mason College of Engineering and Computing. 

## REFERENCES 

- Haoli Bai, Lu Hou, Lifeng Shang, Xin Jiang, Irwin King, and Michael Lyu. Towards efficient posttraining quantization of pre-trained language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum?id=tvDRmAxGIjw. 

- BIG bench authors. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions on Machine Learning Research_ , 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=uyTL5Bvosj. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- Zefan Cai, Baobao Chang, and Wenjuan Han. Human-in-the-loop through chain-of-thought. _arXiv preprint arXiv:2306.07932_ , 2023. 

- Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while reducing cost and improving performance. _arXiv preprint arXiv:2305.05176_ , 2023a. 

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. _arXiv preprint arXiv:2211.12588_ , 2022. 

- Xinyun Chen, Maxwell Lin, Nathanael Sch¨arli, and Denny Zhou. Teaching large language models to self-debug. _arXiv preprint arXiv:2304.05128_ , 2023b. 

10 

Published as a conference paper at ICLR 2024 

Zhoujun Cheng, Jungo Kasai, and Tao Yu. Batch prompting: Efficient inference with large language model apis. _arXiv preprint arXiv:2301.08721_ , 2023a. 

- Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. Binding language models in symbolic languages. In _The Eleventh International Conference on Learning Representations_ , 2023b. URL https://openreview.net/forum?id=lH1PV42cbF. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 10764–10799. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/gao23f.html. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. _Transactions of the Association for Computational Linguistics (TACL)_ , 2021. 

- Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 14953–14962, June 2023. 

- Joy He-Yueya, Gabriel Poesia, Rose E Wang, and Noah D Goodman. Solving math word problems by combining language models with symbolic solvers. _arXiv preprint arXiv:2304.09102_ , 2023. 

- Shima Imani, Liang Du, and Harsh Shrivastava. Mathprompter: Mathematical reasoning using large language models. _arXiv preprint arXiv:2303.05398_ , 2023. 

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know, 2022. 

- Eldar Kurtic, Elias Frantar, and Dan Alistarh. Ziplm: Hardware-aware structured pruning of language models. _arXiv preprint arXiv:2302.04089_ , 2023. 

- Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In _International Conference on Machine Learning_ , 2022. URL https://api. semanticscholar.org/CorpusID:254096365. 

- Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum?id=IFXTZERXdM7. 

- Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 12286– 12312, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/ v1/2023.acl-long.687. URL https://aclanthology.org/2023.acl-long.687. 

11 

Published as a conference paper at ICLR 2024 

- Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023a. URL https://openreview.net/forum?id= Rzk3GP1HN7. 

- Jianzhe Lin, Maurice Diesendruck, Liang Du, and Robin Abraham. Batchprompt: Accomplish more with less, 2023b. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 158–167, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Ro _{_ bert _}_ a: A robustly optimized _{_ bert _}_ pretraining approach, 2020. URL https://openreview.net/forum?id=SyxS0T4tvS. 

- Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In _International Conference on Learning Representations (ICLR)_ , 2023. 

- Aman Madaan and Amir Yazdanbakhsh. Text and patterns: For effective chain of thought it takes two to tango, 2023. URL https://openreview.net/forum?id=z9fXRC5XdT. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. _arXiv preprint arXiv:2303.17651_ , 2023. 

- Ansong Ni, Jeevana Priya Inala, Chenglong Wang, Alex Polozov, Christopher Meek, Dragomir Radev, and Jianfeng Gao. Learning math reasoning from self-sampled correct and partiallycorrect solutions. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=4D4TSJE6-K. 

OpenAI. Gpt-4 technical report, 2023. 

- Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. _arXiv preprint arXiv:2112.11446_ , 2021. 

- Lior Rokach. Ensemble-based classifiers. _Artificial intelligence review_ , 33:1–39, 2010. 

- Marija Sakota,[ˇ] Maxime Peyrard, and Robert West. Fly-swat or cannon? cost-effective language model choice via meta-modeling. _arXiv preprint arXiv:2308.06077_ , 2023. 

- Chenglei Si, Zhe Gan, Zhengyuan Yang, Shuohang Wang, Jianfeng Wang, Jordan Lee Boyd-Graber, and Lijuan Wang. Prompting GPT-3 to be reliable. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id= 98p5x51L5af. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Neeraj Varshney and Chitta Baral. Model cascading: Towards jointly improving efficiency and accuracy of NLP systems. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 11007–11021, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.756. URL https://aclanthology. org/2022.emnlp-main.756. 

12 

Published as a conference paper at ICLR 2024 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=1PL1NIMMrw. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J. 

- Kai Xiong, Xiao Ding, Yixin Cao, Ting Liu, and Bing Qin. Diving into the inter-consistency of large language models: An insightful analysis through debate. _arXiv preprint arXiv:2305.11595_ , 2023a. 

- Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. _arXiv preprint arXiv:2306.13063_ , 2023b. 

- Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. _arXiv preprint arXiv:2309.05653_ , 2023. 

- Li Zhang, Hainiu Xu, Yue Yang, Shuyan Zhou, Weiqiu You, Manni Arora, and Chris Callisonburch. Causal reasoning of entities and events in procedural texts. In _Findings of the Association for Computational Linguistics: EACL 2023_ , pp. 415–431, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023. findings-eacl.31. 

- Xu Zhao, Yuxi Xie, Kenji Kawaguchi, Junxian He, and Qizhe Xie. Automatic model selection with large language models for reasoning. _arXiv preprint arXiv:2305.14333_ , 2023. 

- Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. _arXiv preprint arXiv:2304.09797_ , 2023. 

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. _arXiv preprint arXiv:2304.06364_ , 2023. 

13 

Published as a conference paper at ICLR 2024 

## A COST ANALYSIS OF LLM CASCADE APPROACHES 

In Section 2.3, we introduced ten approaches to implement the LLM cascade decision maker. As we are interested in saving the cost of LLM usage, we first analyze the token usage of each approach. While different thought representations of the same task example could induce different token usages, the difference is very hard to quantify and also depends on the specific reasoning task. For example, for GSM8k (Cobbe et al., 2021), the PoT representation of the same example is typically shorter than its CoT counterpart, while a reversed comparison is observed for DATE and Navigate (bench authors, 2023). In addition, different demonstration examples may also consist of different numbers of tokens, which is hard to quantify as well. To provide a unified analysis, we assume that every demonstration example for any task, regardless of the thought representation in use, consumes _Ntok_ tokens. 

Based on our hypothesis, we conducted a cost analysis, with results summarized in Table 1. To give an example, consider comparing the token costs of CoT-1D-Vote and CoT-2D-Vote. For CoT-1D-Vote, the cost of calling weaker LLM is ( _MCi[w]_[+] _[KC] o[w]_[)] _[×][N][tok]_[; for][ CoT-2D-Vote][, it] is (2 _MCi[w]_[+] _[K]_[1] _[C] o[w]_[+] _[K]_[2] _[C] o[w]_[)] _[×][N][tok]_[.][Here,] _[ C] i[w]_[and] _[ C] o[w]_[are the input and output per-token cost for] the weaker LLM, respectively, and _M_ is the number of demonstrations in each prompt. In practice, we set _K_ 1 = _K_ 2 = _K_ 2 _D/_ 2, where _K_ 2 _D_ is the total sample size of two sets of demonstration examples, and the total cost for CoT-2D-Vote can then be rewritten to (2 _MCi[w]_[+] _[ K]_[2] _[D][C] o[w]_[)][.][To] keep the costs of CoT-1D-Vote and CoT-2D-Vote consistent, we set _K_ 2 _D_ = _K_ 1 _D − M ×[C] Cio[w][w]_[.] Similarly, we can get the total sample size of different representations _KMoT_ . The analysis guides us in configuring each approach to be “cost-comparable” (i.e., leading to similar _C[w]_ ’s). Finally, we note that we do not change the number of task demonstrations (i.e., _M_ ) because LLMs are very sensitive to this configuration. 

|Method|Certainty Value (_s_) Defnition|Cost Estimation|||
|---|---|---|---|---|
|CoT-1D-Vote|Agreement of _K_ CoT samples on the majority-voted answer _Aw_, _s_<br>=|(_MCw_<br>_i_ +_K_1_DCw_<br>_o_||)_× Ntok_|
||�_K_<br>_i_=1 1_Aw_<br>_i_ =_Aw/K_||||
|PoT-1D-Vote|Agreement of _K_ PoT samples on the majority-voted answer _Aw_, _s_<br>=|(_MCw_<br>_i_ +_K_1_DCw_<br>_o_||)_× Ntok_|
|MoT-1D-Vote|�_K_<br>_i_=1 1_Aw_<br>_i_ =_Aw/K_<br>Agreement of_KMoT /_2CoT and_KMoT /_2PoT samples on the majority-voted|2_×_(_MCw_<br>_i_ + _KMoT_<br>2||_Cw_<br>_o_ )_×Ntok_|
||answer_Aw_,_s_= �_KMoT_<br>_i_=1<br>1_Aw_<br>_i_ =_Aw/KMoT_||||
|CoT-2D-Vote|Agreement of _K_2_D/_2 CoT samples over 2 sets of task demonstrations on the|2_×_(_MCw_<br>_i_ + _K_2_D_<br>2|_Cw_<br>_o_ )_×Ntok_||
|PoT-2D-Vote|majority-voted answer_Aw_,_s_= �_K_2_D_<br>_i_=1 1_Aw_<br>_i_ =_Aw/K_2_D_<br>Agreement of _K_2_D/_2 PoT samples over 2 sets of task demonstrations on the|2_×_(_MCw_<br>_i_ + _K_2_D_<br>2|_Cw_<br>_o_ )_×Ntok_||
|MoT-2D-Vote|majority-voted answer_Aw_,_s_= �_K_2_D_<br>_i_=1 1_Aw_<br>_i_ =_Aw/K_2_D_<br>Agreement of _KMoT /_2 CoT samples on one set of task demos and _KMoT /_2|2_×_(_MCw_<br>_i_ + _KMoT_<br>2||_Cw_<br>_o_ )_×Ntok_|
||PoT samples on another set of task demos on their majority-voted answer_A∗_,||||
||_s_= �_KMoT_<br>_i_=1<br>1_Aw_<br>_i_ =_Aw/KMoT_||||
|CoT-2D-Verify|Agreement of the two majority-voted answers_Aw′_<br>1 and_Aw′_<br>2 over 2 sets of CoT|2_×_(_MCw_<br>_i_ + _K_2_D_<br>2|_Cw_<br>_o_ )_×Ntok_||
||task demonstrations with_K_2_D_ examples,_s_=1_Aw′_<br>1 =_Aw′_<br>2||||
|PoT-2D-Verify|Agreement of the two majority-voted answers_Aw′_<br>1 and_Aw′_<br>2 over 2 sets of PoT|2_×_(_MCw_<br>_i_ + _K_2_D_<br>2|_Cw_<br>_o_ )_×Ntok_||
||task demonstrations with_K_2_D_ examples,_s_=1_Aw′_<br>1 =_Aw′_<br>2||||
|MoT-2D-Verify|Agreement of the two majority-voted answers _Aw′_<br>1<br>and _Aw′_<br>2<br>over one set of|2_×_(_MCw_<br>_i_ + _KMoT_<br>2||_Cw_<br>_o_ )_×Ntok_|
||CoT and another set of PoT task demonstrations with_KMoT /_2examples,_s_ =||||
||1_Aw′_<br>1 =_Aw′_<br>2||||
|MoT-1D-Verify|Agreement of the two majority-voted answers _Aw′_<br>1<br>and _Aw′_<br>2<br>for CoT and|2_×_(_MCw_<br>_i_ +_KMoT_<br>2||_Cw_<br>_o_ )_×Ntok_|
||PoT over the same set of task demonstrations with _KMoT /_2 examples, _s_ =||||
||1_Aw′_<br>1 =_Aw′_<br>2||||



Table 1: Investigated approaches and cost calculations. For simplicity, we assume _M_ -shot demonstration prompting and that every demonstration example has _Ntok_ tokens. _Ci[w]_[is the input price per] token and _Co[w]_[is the output price per token.][To keep consistent costs of using the weaker LLM, we] set _KMoT_ = _K_ 2 _D_ = _K_ 1 _D − M × C[C] oi[w][w]_[.] 

## B MAIN EXPERIMENTAL DETAILS 

**Implementation Details** We run each approach two times to reduce variance and report the average results. Since DATE and Navigate do not have a training set, before experiments we sampled 8 shots of examples randomly and annotated them to be the second set of task demonstrations (for 2D 

14 

Published as a conference paper at ICLR 2024 

approaches). The remaining examples are used consistently across all experiments as the test set. We adopt _“Python code, return ans”_ in PoT prompt to let the LLM generate the Python code (Chen et al., 2022). The interpreter we used for executing PoT is Python 3.10, with some packages, such as DateTime[2] to facilitate the execution of the generated code. 

**Metrics Details** We evaluate the methods based on task accuracy and cost efficiency, with a lower cost and higher accuracy indicating better performance. For accuracy, when the answer is a string, we use exact matching. When the answer is a number, we relax the evaluation criteria due to potential variations in the exact computations carried out by the external interpreter. Following prior work (Chen et al., 2022), we adopt the tolerance to 0.001. For cost efficiency, we calculate the actual token number based on the tiktoken[3] and the total cost[4] for each method and get the relative cost by comparing it with the total cost of GPT-4-CoT-SC. 

## C MAIN RESULT TABLES 

In this section, we show the exact numerical results that are reported in Figure 3. The GSM8k result is in Table 2. The ASDIV result is in Table 3. The TabMWP result is in Table 4. The DATE result is in Table 5. The Navigation result is in Table 6. The CREPE result is in Table 7. The average results are in Table 8. 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.875|0.184|0.832|0.141|0.897|0.211|0.854|0.170|0.901|0.187|0.904|0.195|
|0.50|0.896|0.233|0.850|0.179|0.912|0.249|0.872|0.209|0.917|0.235|0.918|0.241|
|0.55|0.907|0.265|0.861|0.206|0.929|0.309|0.894|0.262|0.943|0.331|0.942|0.334|
|0.60|0.917|0.302|0.876|0.240|0.937|0.354|0.910|0.305|0.948|0.381|0.948|0.381|
|0.65|0.924|0.336|0.886|0.265|0.945|0.408|0.920|0.342|0.951|0.425|0.952|0.425|
|0.70|0.929|0.372|0.895|0.292|0.945|0.408|0.920|0.342|0.951|0.425|0.952|0.425|
|0.80|0.941|0.438|0.910|0.346|0.952|0.514|0.933|0.427|0.954|0.528|0.955|0.530|
|0.90|0.950|0.518|0.924|0.410|0.955|0.587|0.938|0.485|0.955|0.598|0.956|0.600|
|1.00|0.954|0.660|0.940|0.517|0.956|0.699|0.947|0.565|0.957|0.693|0.957|0.691|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.926|0.322|0.909|0.311|0.947|0.401|0.949|0.403|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.842|0.111|0.792|0.078|0.945|0.789|0.942|0.603|0.958|1.000|0.947|0.745|



Table 2: Exact numerical results on GSM8k. 

> 2https://pypi.org/project/DATETime/ 

> 3https://platform.openai.com/tokenizer 

> 4https://openai.com/pricing 

15 

Published as a conference paper at ICLR 2024 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.902|0.181|0.872|0.173|0.911|0.200|0.877|0.167|0.913|0.189|0.916|0.184|
|0.50|0.908|0.209|0.878|0.190|0.917|0.225|0.883|0.188|0.918|0.218|0.920|0.214|
|0.55|0.910|0.226|0.882|0.210|0.924|0.262|0.894|0.222|0.930|0.291|0.931|0.290|
|0.60|0.914|0.243|0.887|0.224|0.926|0.288|0.904|0.251|0.932|0.324|0.933|0.324|
|0.65|0.917|0.261|0.893|0.239|0.929|0.312|0.912|0.273|0.933|0.346|0.933|0.348|
|0.70|0.920|0.281|0.897|0.254|0.929|0.312|0.912|0.275|0.933|0.347|0.933|0.348|
|0.80|0.924|0.311|0.906|0.283|0.930|0.365|0.923|0.328|0.935|0.396|0.934|0.400|
|0.90|0.927|0.348|0.916|0.318|0.930|0.402|0.927|0.360|0.935|0.430|0.934|0.433|
|1.00|0.930|0.429|0.925|0.367|0.931|0.458|0.930|0.393|0.935|0.473|0.934|0.474|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.924|0.272|0.908|0.297|0.933|0.357|0.934|0.361|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.887|0.115|0.854|0.106|0.927|0.742|0.930|0.725|0.933|1.000|0.943|0.927|



Table 3: Exact numerical results on ASDIV. 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.868|0.141|0.883|0.150|0.900|0.152|0.892|0.151|0.902|0.155|0.912|0.149|
|0.50|0.880|0.169|0.894|0.177|0.909|0.183|0.897|0.179|0.919|0.193|0.924|0.185|
|0.55|0.890|0.191|0.899|0.195|0.934|0.273|0.916|0.238|0.951|0.324|0.952|0.329|
|0.60|0.899|0.216|0.905|0.214|0.941|0.318|0.925|0.288|0.952|0.370|0.952|0.381|
|0.65|0.910|0.245|0.910|0.236|0.945|0.355|0.935|0.327|0.954|0.409|0.952|0.421|
|0.70|0.917|0.273|0.915|0.254|0.945|0.355|0.935|0.327|0.954|0.409|0.952|0.421|
|0.80|0.927|0.327|0.924|0.292|0.950|0.448|0.946|0.411|0.957|0.484|0.956|0.496|
|0.90|0.935|0.393|0.934|0.345|0.950|0.513|0.952|0.460|0.955|0.531|0.956|0.548|
|1.00|0.946|0.535|0.947|0.425|0.950|0.601|0.955|0.524|0.955|0.610|0.955|0.630|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.940|0.333|0.920|0.262|0.950|0.342|0.951|0.357|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.844|0.092|0.849|0.072|0.958|0.819|0.933|0.728|0.961|1.000|0.941|0.849|



Table 4: Exact numerical results on TabMWP. 

## Voting Method 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.737|0.192|0.793|0.203|0.775|0.203|0.829|0.208|0.805|0.207|0.812|0.197|
|0.50|0.758|0.235|0.806|0.229|0.792|0.240|0.844|0.235|0.821|0.240|0.826|0.225|
|0.55|0.769|0.259|0.818|0.256|0.835|0.362|0.872|0.324|0.871|0.367|0.868|0.367|
|0.60|0.779|0.289|0.828|0.283|0.849|0.431|0.877|0.367|0.881|0.417|0.872|0.425|
|0.65|0.795|0.328|0.837|0.306|0.862|0.489|0.882|0.419|0.889|0.469|0.881|0.484|
|0.70|0.809|0.357|0.848|0.329|0.862|0.489|0.882|0.419|0.889|0.469|0.881|0.484|
|0.80|0.835|0.425|0.860|0.378|0.877|0.604|0.890|0.502|0.894|0.569|0.888|0.583|
|0.90|0.854|0.511|0.877|0.425|0.883|0.666|0.892|0.552|0.895|0.637|0.891|0.654|
|1.00|0.877|0.657|0.888|0.523|0.888|0.770|0.889|0.625|0.892|0.739|0.892|0.744|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.850|0.456|0.873|0.392|0.875|0.449|0.869|0.459|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.677|0.107|0.770|0.162|0.876|0.750|0.873|1.031|0.886|1.000|0.890|1.402|



Table 5: Exact numerical results on DATE. 

16 

Published as a conference paper at ICLR 2024 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.860|0.132|0.871|0.213|0.898|0.125|0.868|0.192|0.878|0.145|0.880|0.146|
|0.50|0.860|0.134|0.877|0.246|0.898|0.125|0.874|0.219|0.882|0.153|0.883|0.152|
|0.55|0.862|0.140|0.881|0.270|0.913|0.233|0.882|0.261|0.964|0.368|0.964|0.365|
|0.60|0.864|0.144|0.885|0.293|0.920|0.265|0.892|0.305|0.966|0.402|0.966|0.400|
|0.65|0.866|0.149|0.889|0.317|0.924|0.282|0.908|0.371|0.968|0.440|0.967|0.437|
|0.70|0.869|0.159|0.892|0.349|0.924|0.282|0.908|0.371|0.968|0.440|0.967|0.437|
|0.80|0.876|0.175|0.903|0.432|0.934|0.318|0.926|0.537|0.971|0.545|0.970|0.543|
|0.90|0.884|0.193|0.920|0.548|0.941|0.339|0.935|0.649|0.973|0.626|0.972|0.624|
|1.00|0.901|0.240|0.942|0.743|0.948|0.377|0.948|0.815|0.973|0.754|0.973|0.754|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.929|0.293|0.879|0.208|0.966|0.400|0.966|0.398|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.856|0.128|0.859|0.154|0.977|0.701|0.877|0.966|0.975|1.000|0.878|1.302|



Table 6: Exact numerical results on Navigate. 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.838|0.188|0.838|0.461|0.850|0.180|0.846|0.369|0.866|0.255|0.874|0.257|
|0.50|0.843|0.208|0.842|0.485|0.857|0.197|0.855|0.422|0.871|0.295|0.878|0.301|
|0.55|0.847|0.227|0.846|0.500|0.865|0.241|0.870|0.552|0.885|0.460|0.885|0.471|
|0.60|0.852|0.261|0.846|0.517|0.865|0.280|0.869|0.603|0.882|0.519|0.882|0.525|
|0.65|0.856|0.284|0.847|0.537|0.868|0.320|0.869|0.658|0.882|0.573|0.881|0.575|
|0.70|0.863|0.312|0.849|0.557|0.869|0.369|0.872|0.708|0.879|0.614|0.880|0.620|
|0.80|0.867|0.374|0.853|0.603|0.868|0.419|0.874|0.754|0.879|0.660|0.878|0.663|
|0.90|0.873|0.455|0.861|0.683|0.869|0.534|0.874|0.851|0.877|0.760|0.873|0.760|
|1.00|0.872|0.594|0.868|0.803|0.871|0.621|0.873|0.925|0.873|0.847|0.871|0.846|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.860|0.302|0.874|0.581|0.882|0.534|0.883|0.536|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.820|0.157|0.719|0.226|0.871|0.577|0.876|0.871|0.871|1.000|0.882|1.522|



Table 7: Exact numerical results on CREPE. 

## Voting Method 

|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|VotingMethod|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||CoT-1D-Vote||PoT-1D-Vote||CoT-2D-Vote||PoT-2D-Vote||MoT-1D-Vote||MoT-2d-Vote||
|Threshold|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
|0.40|0.847|0.167|0.848|0.202|0.872|0.178|0.861|0.195|0.878|0.183|0.883|0.183|
|0.50|0.857|0.195|0.858|0.231|0.881|0.203|0.871|0.227|0.888|0.216|0.891|0.216|
|0.55|0.864|0.216|0.864|0.253|0.900|0.273|0.888|0.286|0.924|0.345|0.924|0.347|
|0.60|0.871|0.240|0.871|0.276|0.906|0.312|0.896|0.329|0.927|0.389|0.925|0.391|
|0.65|0.878|0.263|0.877|0.297|0.912|0.348|0.904|0.371|0.929|0.428|0.928|0.431|
|0.70|0.884|0.289|0.882|0.320|0.912|0.354|0.905|0.377|0.929|0.433|0.927|0.436|
|0.80|0.895|0.336|0.893|0.370|0.918|0.428|0.915|0.465|0.932|0.513|0.930|0.517|
|0.90|0.904|0.394|0.905|0.436|0.921|0.488|0.920|0.530|0.932|0.576|0.930|0.581|
|1.00|0.913|0.506|0.918|0.543|0.924|0.567|0.924|0.611|0.931|0.663|0.930|0.667|
|VerifyMethod|||||||||||||
||CoT-2D-Verify||PoT-2D-Verify||MoT-1D-Verify||MoT-2D-Verify||||||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|||||
||0.905|0.311|0.894|0.316|0.925|0.398|0.925|0.402|||||
|Without Cascade|||||||||||||
||GPT-3-CoT-SC||GPT-3-PoT-SC||GPT-4 CoT-Greedy||GPT-4 PoT-Greedy||GPT-4 CoT-SC||GPT-4 PoT-SC||
||Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|Acc|Cost|
||0.821|0.116|0.807|0.117|0.926|0.743|0.905|0.763|0.931|1.000|0.913|1.017|



Table 8: Exact numerical results on average across all datasets. 

17 

Published as a conference paper at ICLR 2024 

## D CASE STUDY 

In this section, we perform a case study to learn why MoT can achieve better performance. When CoT makes a mistake, prompting the LLM with another set of CoT task demonstrations often leads to the same mistake. In contrast, PoT more frequently either directly gives a correct answer or makes a different mistake than CoT’s. Among those cases, 43% fall into the former category that PoT can solve the question, while the remaining 57% fall into the latter that PoT can make a different mistake. Here we use CoT1, CoT2, and PoT1 to represent the different prompts. CoT1 and CoT2 share the same representation but different demonstrations while CoT1 and PoT1 share the same demonstration but different representations. 

## D.1 CASES POT CAN SOLVE 

Here we show cases satisfying that PoT can solve the question. The first case shows that the PoT can provide different reasoning paths to avoid the same mistake with CoT and the second case shows that the PoT can avoid the computation errors. 

## D.1.1 CASE 1 

**Question:** Juan and his brother Carlos are selling lemonade. For each gallon they make it costs 3 for lemons and 2 for sugar. They sell each glass for 0 _._ 50 and get 20 glasses per gallon. If they made 25 in profit, how much did they spend on lemons? 

**Gold Answer:** Each gallon sells for a total of 10 because 20 _×_ 0 _._ 5 = 10 Each gallon costs 5 to make because 3 + 2 = 5 They make 5 off each gallon because 10 _−_ 5 = 5 They made and sold 5 gallons because 25 _/_ 5 = 5 They spent 15 on lemons because 5 _×_ 3 = 15 **CoT1:** Answer: They make 10 on each gallon of lemonade because 20 _×_ 0 _._ 5 = 10 They made 2 _._ 5 gallons of lemonade because 25 _/_ 10 = 2 _._ 5 They spent 7 _._ 5 on lemons because 2 _._ 5 _×_ 3 = 7 _._ 5 They spent 5 on sugar because 2 _._ 5 _×_ 2 = 5 They spent 12 _._ 5 in total because 7 _._ 5 + 5 = 12 _._ 5 ans = 7 _._ 5 **CoT2:** Answer: They make 10 per gallon because 20 _×_ 0 _._ 5 = 10 They sold 2 _._ 5 gallons because 25 _/_ 10 = 2 _._ 5 They spent 7 _._ 5 on sugar because 2 _._ 5 _×_ 2 _×_ 1 _._ 5 = 7 _._ 5 They spent 7 _._ 5 on lemons because 2 _._ 5 _×_ 3 = 7 _._ 5 ans = 7 _._ 5 

## **PoT1:** 

# Python code, return ans cost_per_gallon = 3 + 2 num_of_glasses_per_gallon = 20 profit_per_gallon = num_of_glasses_per_gallon * 0.5 -\ cost_per_gallon total_profit = 25 total_gallons = total_profit / profit_per_gallon cost_of_lemons = total_gallons * 3 ans = cost_of_lemons #ans = 15.0(This line is not generated by LLM.) D.1.2 CASE 2 

**Question:** Robert wants to buy a bookshelf and a sofa and has received two offers. Furniture United’s offer includes a 1 _,_ 350 advance payment and 6 monthly installments of 350 each. Cozy Homes’ offer includes a 1 _,_ 100 advance payment and 9 monthly installments of 250 each. What is the cost difference between the two offers? 

18 

Published as a conference paper at ICLR 2024 

**Gold Answer:** Furniture United’s offer is 1 _,_ 350 + (6 _×_ 350) = 3 _,_ 450 Cozy Homes’ offer is 1 _,_ 100 + (9 _×_ 250) = 3 _,_ 350. The difference is 3 _,_ 450 _−_ 3 _,_ 350 = 100 

**CoT1:** Answer: Furniture United’s total cost would be 1350 + 6 _×_ 350 = 3150. Cozy Homes’ total cost would be 1100+9 _×_ 250 = 3350. The cost difference between the two offers is 3350 _−_ 3150 = 200. ans = 200 

**CoT2:** Answer: The total cost of Furniture United’s offer is 1350 + (6 _×_ 350) = 3 _,_ 150. The total cost of Cozy Homes’ offer is 1100+(9 _×_ 250) = 3 _,_ 350. The cost difference between the two offers is 3350 _−_ 3150 = 200. ans = 200. 

## **PoT1:** 

furniture_united_advance = 1350 furniture_united_monthly_installment = 350 cozy_homes_advance = 1100 cozy_homes_monthly_installment = 250 furniture_united_total_cost = furniture_united_advance + \ (6 * furniture_united_monthly_installment) cozy_homes_total_cost = cozy_homes_advance + \ (9 * cozy_homes_monthly_installment) ans = furniture_united_total_cost - cozy_homes_total_cost # ans = 100.0(This line is not generated by LLM.) 

## D.2 CASES POT CANNOT SOLVE 

In this section, we show some cases that PoT cannot solve but will make a different mistake. In case 1, the CoT forgot to calculate the first 40-yard line but the PoT calculated the first 40-yard but with an incorrect grounding value. In case 2, the CoT forgot to remove the lowest score. The PoT removed the lowest score but grounded an incorrect number of total tests. We can see that although both the CoT and PoT make mistakes, the reasons why they make mistakes are different, making the final results different. 

## D.2.1 CASE 1 

**Question:** Blake and Kelly are having a contest to see who can run the most in 15 minutes. They decide to do it on a football field that is 100 yards long. Blake runs back and forth 15 times. Kelly runs back and forth once, and then decides that she doesn’t want to run next to Blake, so she starts to run to the 40-yard line and back. She does this 34 times. How much farther does the winner run than the loser? 

**Gold Answer:** Blake runs 200 yards for each back and forth because 100 x 2 = 200 He runs 3000 yards in total because 200 x 15 = 3000 Kelly runs 200 yards at first because 100 x 2 = 200 She runs 80 yards each time she does a back and forth to the 40-yard line because 40 x 2 = 80 She runs 2720 from these back and forth because 34 x 80 = 2720 She runs 2920 in total because 2720 + 200 = 2920 Blake won because 3000 _>_ 2920 The winner ran 80 yards more than the loser because 3000 - 2920 = 80 

**CoT1:** Answer: Blake runs back and forth 15 times, so she runs 15*100*2=3000 yards. Kelly runs to the 40-yard line and back 34 times, so she runs 34*80=2720 yards. Thus, Blake runs 30002720=280 yards farther than Kelly. ans = 280 

**CoT2:** Answer: Blake runs a total of 15 x 100 x 2 = 3000 yards. Kelly runs 34 x (40 x 2) = 2720 yards. Blake runs 3000 - 2720 = 280 yards farther than Kelly. ans = 280. 

## **PoT1:** 

# Python code, return ans length_of_football_field = 100 num_of_laps_for_blake = 15 num_of_laps_for_kelly = 34 distance_covered_by_blake = length_of_football_field * \ num_of_laps_for_blake * 2 distance_covered_by_kelly = (length_of_football_field *\ 

19 

Published as a conference paper at ICLR 2024 

num_of_laps_for_kelly * 2) + (40 * num_of_laps_for_kelly * 2) ans = abs(distance_covered_by_blake - distance_covered_by_kelly) # ans = 6520.0(This line is not generated by LLM.) 

## D.2.2 CASE 2 

**Question:** Brinley is in Mr. Bert’s math class. Mr. Bert gives six tests each semester. He allows the students to remove the lowest score from the average each semester. If Brinley has scores of 89, 71, 92, 100 and 86 on the first five tests, what score does she need on the sixth test to get an average of 93? 

**Gold Answer:** First, identify the lowest score, which is 71, and remove it. Since Brinley wants a score of 93, we know her total score needs to be 93 (average) * 5 (tests) = 465 points So far, Brinley has earned 89 + 92 +100 + 86 = 367 points The final test must be 465 - 367 = 98 points. 

**CoT1:** First, we need to find Brinley’s current average by adding up her five test scores and dividing by 5: (89 + 71 + 92 + 100 + 86) / 5 = 87.6 We know that Brinley wants her average to be 93, which means she needs to have a total of 6 test scores (since one will be dropped). The total of these 6 scores needs to be 93 x 6 = 558 To find out what score Brinley needs on the sixth test, we can subtract the total of her first five scores from the desired total: 558 - (89 + 71 + 92 + 100 + 86) = 120 So Brinley needs to get a score of 120 on her sixth test to have an average of 93 after dropping her lowest score. ans = 120 

**CoT2:** Answer: Brinley’s first five test scores add up to 438. To have an average of 93, Brinley needs a total of 93 x 6 = 558 points for all six tests. To get 558 points after five tests, Brinley needs to score 558 - 438 = 120 points on the sixth test. ans = 120 

## **PoT1:** 

# Python code, return ans num_of_tests = 6 total_of_first_five_tests_scores = 89 + 71 + 92 + 100 + 86 desired_average = 93 lowest_score = min(89, 71, 92, 100, 86) sum_of_best_five_scores = total_of_first_five_tests_scores -\ lowest_score ans = (desired_average * num_of_tests) - sum_of_best_five_scores # ans = 191.0(This line is not generated by LLM.) 

## E IMPLEMENTATION DETAILS OF EXTERNAL VERIFIERS 

Throughout these experiments, our choice of the weaker LLM remains GPT-3.5-turbo. We only perform our experiment with CoT representation because the finetuned model we utilize is not suitable for understanding Python code. 

The input to the verifier can be either the question alone or a concatenation of both the question and the answer produced by the weaker LLM. The question texts include the context, e.g., the goal and context in CREPE. The answer texts contain both intermediate thoughts and the final answer (e.g., an integer in mathematical reasoning). We have multiple generation results from the weaker LLM. To avoid excessive input length, we only employ the majority-voted answer and its intermediate thoughts in the test time. Specifically, in the test time, we extract the majority-voted answer from two CoT prompts with different demonstrations as this setting shows better performance in our method, and then randomly select one collection of intermediate thoughts that result in this majority-voted answer. 

We still use task accuracy and relative cost as the metric. Note that only the question as input doesn’t need the answers from the weaker LLM necessarily. Therefore, the total cost _C_ (Eq 1) for only the question as input becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0020-13.png)


To use an LLM as the verifier, we have designed two distinct prompts: one for “question difficulty prediction”, which solely relies on the questions (LLM-Q), and another for “answer evalua- 

20 

Published as a conference paper at ICLR 2024 

tion”, which incorporates both questions and answers (LLM-QA). The difficulty prediction (LLM-Q) prompt entails the LLM’s determination of whether a question necessitates redirection to the stronger LLM, based solely on the question’s complexity. If the LLM deems a question challenging for difficulty prediction, it triggers forwarding to the stronger LLM. The answer evaluation (LLM-QA) prompt is to give the LLM both the question and the answer. We then let the LLM rethink the solving steps and generate feedback to evaluate whether the solution is correct. If not, the question will be transferred to the stronger LLM. Both prompting methods are implemented with _M_ = 8 shots of demonstrations, and we keep the number of “easy” and “hard” questions or the number of “correct” and “incorrect” answers balanced. We set the sampling path as _K_ = 20 and temperature as _T_ = 0 _._ 4 to sample the major answer and then use the self-consistency (SC) to decide the final “difficulty” or “correctness of the answers”. The reproducible prompts are in the Appendix L. Notably, the cost for decision _C[d]_ in Eq. 1 is no longer 0 with an external verifier. For the LLM as a verifier, we use GPT-3.5-turbo as it’s much cheaper than GPT-4. We count the number of tokens used for the approaches LLM-Q and LLM-QA. 

For finetuning a smaller LLM (RoBERTa-base of Liu et al. (2020)) as the verifier, we similarly tested two variants, when the input contains only the question (Finetuned-Q) or when it includes both the question and the answers (Finetuned-QA), in which case we use a separation token to split them. The training data for fine-tuning the models come from the original training split of each benchmark dataset. Since DATE does not have a training split, we only performed experiments on GSM8k and CREPE. In each training instance, we have _K_ = 20 generated responses and we label all of them. We label the generated responses where GPT-3.5-CoT-SC correctly predicts the answer as positive. Notably, the number of total training cases is the number of sampling paths _K_ = 20 multiplied by the number of questions. To address the data imbalance issue (i.e., most of the training cases are positive), we perform simple under-sampling over the GSM8k. For CREPE, the resulting training set is relatively smaller, so we did not perform the under-sampling. 

In the training process, we fine-tune RoBERTa-base with the learning rate 1 _×_ 10 _[−]_[5] and batch size 32. We randomly split 10% of the training data to be the development set. The performance of the best accuracy over the development set is shown in Table 9. One notable approach, predicting “all positive”, is to assign a positive label to every example in the development set. We could learn that the external verifier could achieve better performance than “all positive” but not significant enough. 

|Dataset|Method|Dev|
|---|---|---|
||All positive|0.5|
|GSM8k|Based on Question|0.592|
||Based onQuestion and Answer|0.615|
||All positive|0.715|
|CREPE|Based on Question|0.749|
||Based onQuestion and Answer|0.812|



Table 9: Performance of finetuning RoBERTa as verifier over the dev set 

Table 10 shows the exact numerical results of those presented in Figure 6. We can learn that the external verifier could boost the performance but not that significant. Comparing LLM-Q with LLM-QA, an interesting observation is that the latter costs less than the former. This may seem counter-intuitive at first glance because LLM-QA includes a much longer input (a concatenation of question and answer) than LLM-Q (only question). However, we found out that for both approaches, their final costs are mainly determined by the number of cases transferred to the stronger LLM. That is, when including both the question and the answer as input, LLM-QA tends to trust the weaker LLM more frequently, leading to fewer cases being routed to the stronger LLM and hence the lower cost. However, both approaches do not yield accuracy as high as our MoT-based approaches. 

## F LLAMA2-13B AS THE WEAKER LLM 

Although GPT-3.5-turbo is considered to be weaker compared to GPT-4, it is still more powerful and expensive than most of the open-source LLMs. In this section, we tested the performance of 

21 

Published as a conference paper at ICLR 2024 

|Dataset|Method|Accuracy|Cost|
|---|---|---|---|
||GPT-3.5-turbo CoT-SC|0.842|0.111|
||LLM-Q|0.884|0.253|
||LLM-QA|0.862|0.135|
|GSM8k|Finetuned-Q|0.892|0.340|
||Finetuned-QA|0.882|0.262|
||**CoT-2D-Vote with threshold=0.7**|**0.945**|**0.408**|
||**MoT-1D-Vote with threshold=0.5**|**0.917**|**0.235**|
||**MoT-1D-Vote with threshold=0.7**|**0.951**|**0.425**|
||**MoT-1D-Verify**|**0.947**|**0.401**|
||GPT-4 CoT-SC|0.958|1.000|
||GPT-3.5-turbo CoT-SC|0.676|0.107|
||LLM-Q|0.750|0.367|
|DATE|LLM-QA|0.707|0.142|
||**CoT-2D-Vote with threshold=0.65**|**0.862**|**0.489**|
||**MoT-1D-Vote with threshold=0.65**|**0.889**|**0.469**|
||**MoT-1D-Verify**|**0.875**|**0.449**|
||GPT-4 CoT-SC|0.886|1.000|
||GPT-3.5-turbo CoT-SC|0.820|0.157|
||LLM-Q|0.864|0.406|
||LLM-QA|0.846|0.169|
|CREPE|Finetuned-Q|0.854|0.323|
||Finetuned-QA|0.847|0.322|
||**CoT-2D-Vote with threshold=0.55**|**0.865**|**0.241**|
||**MoT-1D-Vote with threshold=0.55**|**0.885**|**0.460**|
||**MoT-1D-Verify**|**0.882**|**0.534**|
||GPT-4 CoT-SC|0.871|1.000|



Table 10: Exact numerical results of various external verifiers. Specifically, because DATE does not have a training set, we do not report finetuned models’ performance on it. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0022-03.png)


Figure 7: The performance of using LLAMA2-13B as weaker LLM. 

LLAMA2-13B (Touvron et al., 2023), an open-source and easy-to-deploy LLM, as the weaker LLM and GPT-4 as the stronger LLM. 

The result is shown in the Figure 7. Through experimentation, we observe that choosing LLAMA213B as a weaker LLM and applying our strategy on GSM8k and CREPE did not yield ideal results, but it performed well on the DATE dataset. On GSM8k and CREPE datasets, accuracy and cost change approximately linearly and we cannot achieve a comparable result with a lower cost, but on the DATE dataset, we obtain a curve similar to the curve using GPT-3.5-turbo as a weaker LLM. To analyze the cause, we have selected cases in MoT-1D-Vote whose agreement score _s_ (Eq 2) is greater than the 0.6, 0.7, and 0.8. We report their task accuracy and proportion among all cases. A lower accuracy in those cases is indicative of reduced overall task performance, while a diminished proportion implies an increased cost. 

The results are displayed in Table 11. Comparing LLAMA2-13B to GPT-3.5-turbo, we observe that LLAMA2-13B exhibits a significantly lower proportion. This suggests that LLAMA2-13B is less likely to provide consistent answers across multiple samplings. For example, only 16.3% of the cases have an agreement score greater than the 0.6 threshold on GSM8k in LLAMA2-13B but 77.3% in GPT-3.5-turbo. However, the difference in accuracy is slight in GSM8k and DATE. This outcome aligns with our hypothesis that the LLM can consistently answer easy questions but is hesitant to the hard questions. The lower proportions show that the number of easy questions for LLAMA2-13B is 

22 

Published as a conference paper at ICLR 2024 

far fewer than that in GPT-3.5-turbo as LLAMA2-13B cannot solve these complex reasoning tasks well. Therefore, LLAMA2-13B is not a suitable choice as a weaker LLM for particular tasks. 

|Threshold|Dataset<br>Weak LLM<br>Accuracy<br>Proportion|
|---|---|
|0.6|GSM8k<br>GPT-3.5-turbo<br>0.964<br>0.773<br>LLAMA2-13B<br>0.923<br>0.163|
||DATE<br>GPT-3.5-turbo<br>0.887<br>0.721<br>LLAMA2-13B<br>0.824<br>0.461|
||CREPE<br>GPT-3.5-turbo<br>0.916<br>0.722<br>LLAMA2-13B<br>0.639<br>0.243|
|0.7|GSM8k<br>GPT-3.5-turbo<br>0.981<br>0.707<br>LLAMA2-13B<br>0.946<br>0.093|
||DATE<br>GPT-3.5-turbo<br>0.893<br>0.656<br>LLAMA2-13B<br>0.893<br>0.328|
||CREPE<br>GPT-3.5-turbo<br>0.934<br>0.655<br>LLAMA2-13B<br>0.658<br>0.095|
|0.8|GSM8k<br>GPT-3.5-turbo<br>0.985<br>0.639<br>LLAMA2-13B<br>1.000<br>0.070|
||DATE<br>GPT-3.5-turbo<br>0.900<br>0.593<br>LLAMA2-13B<br>0.933<br>0.203|
||CREPE<br>GPT-3.5-turbo<br>0.956<br>0.588<br>LLAMA2-13B<br>0.600<br>0.013|



Table 11: The performance of cases where the percentage of the major answer under MoT-1D-Vote be greater than the threshold with LLAMA2-13B as the weaker LLM. 

G CAN THE STRONGER LLM BENEFIT FROM THE WEAKER LLM’S HINTS? 

|Dataset|Method|Accuracy|
|---|---|---|
|GSM8k|W/o Hints<br>W/ Hints|0.891<br>0.867|
|DATE|W/o Hints<br>W/ Hints|0.892<br>0.910|
|CREPE|W/o Hints<br>W/ Hints|0.774<br>0.727|



Table 12: The GPT-4-CoT-SC accuracy over the instances that _A[w]_ 1 _[′]_ = _A[w]_ 2 _[′]_ in the MoT-1D-Verify approach 

We conducted an extended experiment to investigate whether progressive hints can further enhance our pipeline. Previous works have proven that progressive hints in multiple iterations can enhance the LLM’s reasoning performance (Zheng et al., 2023). For example, if the answer is 13 in the first iteration and 14 in the second one, the hint for the third iteration can be “ _Hint: The answers may be close to 13, 14_ ”. In MoT-1D/2D-Verify, before the decision maker transfers a question to a stronger LLM, we already have two different answers generated by the weaker LLM. However, the answers from the weaker LLM will be discarded when using the stronger LLM. To incorporate the answers in the leveraging process of the stronger LLM, we follow the prior work (Zheng et al., 2023) to provide these two inconsistent answers as hints. For example, if the main answer is 5 with CoT and 12 with PoT, the hint for the stronger LLM can be “ _Hint: The answers may be close to 5, 12_ ”. We perform the exploration of whether using these answers as hints can improve the stronger LLM’s performance. Prompts can be found in Appendix L. In the prompts, we acknowledge that the hints may not always be correct. In our exploration, we choose the data that will be sent to the stronger LLM by the MoT-1D-Verify approach. We then assess the GPT-4-CoT-SC accuracy of these cases, both with and without the use of hints, respectively. We conduct our experiments on GSM8k, DATE, and CREPE datasets. 

The results in Table 12 reveal that progressive hints may not necessarily further enhance our pipeline. Adding progressive hints only yields a slight improvement on the DATE dataset, with diminishing returns on GSM8k and GREPE. The reason is that GPT-4 tends to be misled by the incorrect answer, especially when both answers from GPT-3.5-turbo are incorrect. 

23 

Published as a conference paper at ICLR 2024 

||accuracy|cost|
|---|---|---|
|w/o Batch Prompting|0.947|0.401|
|w Batch Prompting|0.924|0.325|



Table 13: The GSM8k results with and without batch prompting 

## H CAN BATCHPROMPT FURTHER REDUCE THE COST? 

Batch Prompting is a method to reduce costs by inputting multiple questions into LLM at once (Cheng et al., 2023a; Lin et al., 2023b). In this section, we explore whether our method can be used in conjunction with batch prompting. Our experiments are based on the basic batch prompting setup (Cheng et al., 2023a). We use MoT-1D-Verify for decision making. Specifically, we grouped a batch of 4 test questions into each API call of the weaker LLM, in addition to the original 8-shot demonstrations. The prompt is shown in Figure 39. Like in our previous experiments, we obtain multiple samples from running the weaker LLM, and the verification-based method (Eq 3) can then be leveraged independently for each test question. For example, if two of four test instances are rejected by the decision maker, we only feed the rejected two to the stronger LLM. We perform the experiment on the GSM8k dataset. 

We show the result in Table 13. The results show that adding batch prompting can indeed reduce costs more. However, using batch prompting slightly affects the accuracy of both the weaker LLM and the stronger LLM. The result indicates that our method is orthogonal to the batch prompting approach. 

## I CALIBRATION ANALYSIS 

In this section, we performed a calibration analysis comparing our approaches with baselines. For our voting-based method, we take _n/K_ as the confidence score. For LLM as an external verifier, we followed the prior works’ prompting method (Kadavath et al., 2022) to let the LLM make the decision according to the textual contents.[5] Following their setting, we consider two temperature setups, i.e., _T_ = 1 and _T_ = 2, and donate the variants as LLM-QA T=1 and LLM-QA T=2, respectively. The prompts are shown in Table 31. We also note that, in the original method of Kadavath et al. (2022), the authors used the token probability of an LLM in predicting the answer to be “True” as its confidence score. Since we cannot obtain the token probability of GPT-3.5-turbo, the confidence score is obtained by sampling the True/False prediction _K_ times and reporting the frequency of “True” ( _n/K_ ). This will induce significantly more costs than our approach, but we adopted the strategy here for this calibration analysis. We perform the calibration analysis in the GSM8k dataset. 

The calibration results on GSM8k are illustrated in Figure 8(Left). Our major observations are: (1) All decision-making methods yield a monotone calibration curve, implying that when they have higher confidence in a certain answer, the answer is generally more likely to be true. For our three vote-based approaches, this upward trend also supports our initial hypothesis that a question is easy if the weaker LLM exhibits consistency in its responses across multiple sampling paths; (2) LLM-QA T=2 with a larger temperature can lead to the least calibrated decision maker; (3) However, there is no a significant difference among other approaches in terms of their calibration degree (though all our variants showed to be better than LLM-QA). 

We wanted to note that achieving perfect calibration with _n/K_ as the confidence score is not necessary for our task. To give an extreme example, let us consider the case that whenever an answer has _n/K >_ 0 _._ 5, the answer from the weaker LLM is correct (accuracy 1.0). In this case, we would have a very poor calibration curve, but our pipeline would actually perform very well, because any majority-voted answer it accepts in this case is perfectly true. Another factor to note is the “size” of each bin in the calibration plot (e.g., the number of instances that have confidence scores ranging between 0.6 and 0.7). While this factor will also dramatically impact the LLM cascade performance in practice, it cannot be shown in the calibration plot. 

> 5In our experiments in Section 3.5, the LLM-QA model is additionally prompted to explain its decision making, but we ignored this explanation following Kadavath et al. (2022) for the purpose of calibration analysis. 

24 

Published as a conference paper at ICLR 2024 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0025-01.png)


**----- Start of picture text -----**<br>
1.0 CoT-1D-Vote<br>0.98 CoT-2D-Vote<br>MoT-1D-Vote<br>0.8 0.96 LLM-QA T=1LLM-QA T=2<br>0.6 0.94<br>0.92<br>0.4<br>perfect 0.90<br>CoT-1D-Vote: ECE = 0.035<br>0.2 CoT-2D-Vote: ECE = 0.043 0.88<br>MoT-1D-Vote: ECE = 0.060<br>LLM-QA T=1: ECE = 0.078<br>0.0 LLM-QA T=2: ECE = 0.179 0.86<br>0.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8<br>Confidence Score Confidence Score<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 8: The left figure is for the calibration analysis of different approaches. The number of each method is the expected calibration error (ECE). The right figure plots the accuracy of the subset of answers satisfying _n/K_ greater than the confidence score. 

To give a more direct comparison among different decision-making approaches, we plot Figure 8(Right), where for each approach we report the accuracy of a subset of answers satisfying _n/K_ greater than the confidence score, following the exact way how the approach will be applied in the LLM cascade pipeline. As such, curves on this plot have taken into account the “bin sizes” and can reflect their actual effectiveness. A similar analysis was also performed by Kadavath et al. (2022). We first observed that as the confidence score grows, there is a consistent upward trend in the subset accuracy for all approaches, which echoes the monotone curves in the calibration analysis. The higher position of MoT compared with other variants and baselines also explains its superior performance in the main experimental results. 

## J CAN OUR METHOD GENERALIZE TO FACTUALITY-BASED TASKS? 

In prior studies, the efficacy of our approach in mathematical, symbolic, and causal reasoning has been evidenced. This section explores its potential to be generalized to factual reasoning tasks. Our investigation utilized StrategyQA (Geva et al., 2021), a factoid question answering dataset that requires implicit reasoning over facts. We employed an open-domain question-answering format, presenting only questions without context to the LLM. Our few-shot prompts are shown in Figure 27 and Figure 28. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning_images/iclr_2024_Large_Language_Model_Cascades_with_Mixture_of_Thought_Representations_for_Cost-Efficient_Reasoning.pdf-0025-06.png)


**----- Start of picture text -----**<br>
GPT-3.5-CoT-SC<br>0.80 GPT-3.5-PoT-SC<br>GPT-4-CoT-SC<br>MoT-1D-Vote<br>0.75<br>0.70<br>0.65<br>0.60<br>0.2 0.4 0.6 0.8 1.0<br>Relative Cost<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 9: Performance of MoT-1D-Vote and baselines on the dev set of strategyQA. 

The result is shown in Figure 9. It shows that our method is still effective on factual reasoning tasks. That is again owing to the fact that using different prompt representations could trigger different reasoning paths, which often results in more trustworthy answers when the two representations agree with each other. 

25 

Published as a conference paper at ICLR 2024 

|Is a curling iron<br>necessary in curling?<br>Question<br>A curling iron is a tool used to make the hair curly using<br>heat. The sport of curling requires curling brooms, stones<br>(rocks), and curling shoes. A curling iron is not necessary<br>in curling.<br>Golden<br>Yes, a curling iron is necessary for<br>curling as it is the tool used to create<br>curls in the hair.<br>CoT<br>necessary_equipment_for_curling = ["curling stone",<br>"curling broom"]<br>equipment_used_in_curling_with_curling_iron = ["curling<br>iron"]<br>ans = ("curling iron" in necessary_equipment_for_curling)<br>PoT|Golden||
|---|---|---|
||A curling iron is a tool used to make the hair curly using<br>heat. The sport of curling requires curling brooms, stones<br>(rocks), and curling shoes. A curling iron is not necessary<br>in curling.||
||PoT||
||necessary_equipment_for_curling = ["curling stone",<br>"curling broom"]<br>equipment_used_in_curling_with_curling_iron = ["curling<br>iron"]<br>ans = ("curling iron" in necessary_equipment_for_curling)||



Figure 10: An example showing that MoT helps in factual reasoning tasks such as strategyQA. 

An example is shown in Figure 10. For the question _“Is a curling iron necessary in curling?”_ , the golden answer is _“No, curling is an ice sport and doesn’t need a curling iron”_ . However, most of the CoT answers are “yes” with hallucinations about the concept “curling”. In contrast, most of the PoT answers are “No”. The PoT processes typically list the necessary equipment for curling, such as “curling stone” and “broom”, and then check if “curling iron” is on the list. By checking the consistency between CoT and PoT, MoT-1D-Vote is thus able to identify the incorrect or untrustworthy answer. It is worth noting that, judging from the results of Figure 9, PoT is not better than CoT, but the combination of CoT and PoT generates diverse thoughts and answers, instead of leaning towards one kind of thinking, thus reducing errors in factual reasoning. Therefore, we can still leverage consistency checking across MoT prompts in decision-making to check if the answer from the weaker LLM is trustworthy in factual-based reasoning tasks. 

## K LIMITATIONS AND FUTURE WORKS 

Our approach is still subject to some limitations. Firstly, its applicability is confined to tasks where questions have clear answers, such as mathematical reasoning. Secondly, the utilization of our pipeline might lead to increased latency in cases demanding a stronger LLM. Thirdly, our method is based on the assumption that the intermediate steps could be expressed via different representations. In addition, if weaker LLM has overconfidence in some incorrect beliefs, our method will fail in these cases. 

Considering these limitations, we identify some potential avenues for future research. In tackling the initial challenge, we could integrate other metrics, such as semantic similarity, to evaluate the consistency of the general textual generation tasks. To address the latency issue, it is essential to establish a framework for making decisions early for some obviously easy or hard questions. For the third one, we believe that how to represent the intermediate steps in a specific task relies on human designation. Recent research has shown some predefined programming interfaces effectively manage complex tasks, even those that appear unrelated to traditional code generation tasks, such as image understanding (Gupta & Kembhavi, 2023). If a task’s intermediate steps cannot be expressed directly via code, we can also introduce diverse answers by leveraging some programming interfaces/tools. 

Additionally, due to CoT and PoT being suitable for different tasks, the answer’s correctness with different representations varies across tasks, e.g., the CoT answers are more reliable in GSM8k but the PoT answers are more reliable in DATE. The incorporation of learning algorithms to perform weighted voting for specific tasks yields a promising improvement. 

## L FULL PROMPTS 

We show the prompts used for this paper according to Table 14. We show one example prompt in each case. 

26 

Published as a conference paper at ICLR 2024 

|Prompt Type|Dataset|Prompt Table|
|---|---|---|
||GSM8k|prompt 15|
||ASDIV|prompt 17|
|CoT|TabMWP<br>Date|prompt 19<br>prompt 21|
||Navigate|prompt 23|
||CREPE|prompt 25|
||GSM8k|prompt 16|
||ASDIV|prompt 18|
|PoT|TabMWP<br>Date|prompt**??**<br>prompt 22|
||Navigate|prompt 24|
||CREPE|prompt 26|
||GSM8k w/ Question|prompt 29|
||GSM8k w/ Question and Answer|prompt 30|
|LLM as External Verifer|DATE w/ Question<br>DATE w/ Question and Answer|prompt 32<br>prompt 33|
||CREPE w/ Question|prompt 34|
||CREPE w/Question and Answer|prompt 35|
||GSM8k|prompt 36|
|Learning from hints|Date|prompt 37|
||CREPE|prompt 38|



Table 14: The index of each prompt. 

## **GSM8k CoT** 

Complete the text , s t a r t with ’ Answer ’ and the l a s t l i n e s t a r t s with _�→_ ’ ans = ’ . Question : Manny had 3 b i r t h d a y cookie p i e s to share with h i s 24 _�→_ c l a s s m a t e s and h i s teacher , Mr . Keith . I f each of the cookie p i e s _�→_ were cut i n t o 10 s l i c e s and Manny , h i s classmates , and Mr . Keith _�→_ a l l had 1 piece , how many s l i c e s are l e f t ? Answer : There i s a t o t a l of 3 x 10 = _<<_ 3*10=30 _>>_ 30 cookie s l i c e s . There _�→_ are 24 + 1 + 1 = _<<_ 24+1+1=26 _>>_ 26 people who a t e the cookie p i e c e s . _�→_ There i s 30 −26 = _<<_ 30−26=4 _>>_ 4 cookie s l i c e s l e f t . ans = 4 . . . 

Table 15: GSM8k CoT task demonstrations 

27 

Published as a conference paper at ICLR 2024 

## **GSM8k PoT** 

# Question : Manny had 3 b i r t h d a y cookie p i e s to share with h i s 24 _�→_ c l a s s m a t e s and h i s teacher , Mr . Keith . I f each of the cookie p i e s _�→_ were cut i n t o 10 s l i c e s and Manny , h i s classmates , and Mr . Keith _�→_ a l l had 1 piece , how many s l i c e s are l e f t ? # Python code , r e t u r n ans num cookie pies = 3 n u m s l i c e s p e r c o o k i e p i e = 10 t o t a l p e o p l e = 24 + 1 + 1 t o t a l s l i c e s n e e d e d = t o t a l p e o p l e *[1] t o t a l s l i c e s = num cookie pies * n u m s l i c e s p e r c o o k i e p i e ans = t o t a l s l i c e s − t o t a l s l i c e s n e e d e d . . . 

Table 16: GSM8k PoT task demonstrations 

## **ASDIV CoT** 

Pl ease answer the math q u e s t i o n and r e t u r n a number as the answer . Question : I s a b e l l a ’ s h a i r i s 18 i nch es long . By the end of the year her _�→_ h a i r i s 24 inches long . How much h a i r did she grow? Answer : I s a b e l l a ’ s h a i r i s 18 i nch es long and 24 inch es long . So the l e n g t h she grow in one year i s 24 −18 = 6 ans = 6 . . . 

Table 17: ASDIV CoT task demonstrations 

## **ASDIV PoT** 

# Question : I s a b e l l a ’ s h a i r i s 18 i nch es long . By the end of the year _�→_ her h a i r i s 24 inches long . How much h a i r did she grow? # Python code , r e t u r n ans I s a b e l l a h a i r b e f o r e = 18 I s a b e l l a h a i r a f t e r y e a r = 24 h a i r g r o w t h = I s a b e l l a h a i r a f t e r y e a r − I s a b e l l a h a i r b e f o r e ans = h a i r g r o w t h . . . 

Table 18: ASDIV PoT task demonstrations 

28 

Published as a conference paper at ICLR 2024 

## **TabMWP CoT** 

Read the f o l l o w i n g t a b l e r e g a r d i n g ” Coins ” and then answer a q u e s t i o n : Name _|_ Number of coins Braden _|_ 76 Camilla _|_ 94 Rick _|_ 86 Mary _|_ 84 Hector _|_ 80 Devin _|_ 83 Emily _|_ 82 Avery _|_ 87 Question : Some f r i e n d s d i s c u s s e d the s i z e s of t h e i r coin c o l l e c t i o n s . _�→_ What i s the mean of the numbers ? Explain : Let ’ s t h i n k s t e p by s t e p . The numbers of coins of each one are in [76 , 94 , 86 , 84 , 80 , 83 , 82 , 8 7 ] . So the mean of the numbers i s (76+94+86+84+80+83+82+87) /8 = 648/8 = 81 Answer : 81 . . . 

Table 19: TabMWP CoT task demonstrations 

## **StrategyQA PoT** 

Question : Are more people today r e l a t e d to Genghis Khan than J u l i u s _�→_ Caesar ? # Python code , r e t u r n ans c h i l d r e n n u m o f J u l i u s C a e s a r = 3 children num of Genghis Khan = 6 ans = ( children num of Genghis Khan _>_ c h i l d r e n n u m o f J u l i u s C a e s a r ) 

Table 20: StrategyQA PoT task demonstrations 

## **DATE CoT** 

Q: Today i s Christmas Eve of 1937. What i s the date tomorrow in _�→_ MM/DD/YYYY? Explain : Today i s the Christmas Eve of 1937 , so today i s 12 /2 4/ 19 37 . Today i s 12/24/1937 , the date tomorrow i s 12 /2 5/ 19 37 . A: 12/25/1937 . . . 

Table 21: DATE CoT task demonstrations 

29 

Published as a conference paper at ICLR 2024 

## **DATE PoT** 

# Write Python Code to solve the f o l l o w i n g q u e s t i o n s . from da t et i me import date , t i m e d e l t a from d a t e u t i l . r e l a t i v e d e l t a import r e l a t i v e d e l t a # Q: Today i s Christmas Eve of 1937. What i s the date tomorrow in _�→_ MM/DD/YYYY? # today i s Christmas Eve of 1937 , then today i s 12/24/1937 today = date (1937 , 12 , 24) # tomorrow date tomorrow = today + r e l a t i v e d e l t a ( days =1) # The answer f o r m a t t e d with %m/%d/%Y i s ans = date tomorrow . s t r f t i m e (’%m/%d/%Y’ ) . . . 

Table 22: DATE PoT task demonstrations 

## **Navigate CoT** 

Following t h e s e i n s t r u c t i o n s , i f we r e t u r n to the s t a r t i n g point , r e t u r n _�→_ ’ yes ’ ; e l s e r e t u r n ’no ’ . I n s t r u c t i o n : Take 1 s t e p . Take 2 s t e p s . Take 3 s t e p s . Turn around . Take _�→_ 6 s t e p s . Turn l e f t . Explain : s t a r t p o s i t i o n = [0 , 0] , assume the s t a r t face i s to x−p o s i t i v e 1 . Take 1 s t e p . The c u r r e n t p o s i t i o n i s [1 , 0] 2 . Take 2 s t e p s . The c u r r e n t p o s i t i o n i s [3 , 0] 3 . Take 3 s t e p s . The c u r r e n t p o s i t i o n i s [6 , 0] 4 . Turn around . The face i s to x−n e g a t i v e in the f o l l o w i n g s t e p s . 5 . Take 6 s t e p s . The c u r r e n t p o s i t i o n i s [0 , 0] 6 . Turn l e f t . The face i s y−n e g a t i v e in the f o l l o w i n g s t e p s . After a l l the steps , the p o s i t i o n i s [0 , 0] , the same as the s t a r t i n g _�→_ p o i n t . Answer : yes . . . 

Table 23: Navigate CoT task demonstrations 

30 

Published as a conference paper at ICLR 2024 

## **Navigate PoT** 

Following t h e s e i n s t r u c t i o n s , i f we r e t u r n to the s t a r t i n g point , r e t u r n _�→_ ’ yes ’ ; e l s e r e t u r n ’no ’ . def l e f t r o t a t e ( f a c e d i r e c t ) : f a c e d i r e c t = t u p l e ( f a c e d i r e c t ) mapping dict = _{_ (1 , 0) : (0 , 1) , (0 , 1) : ( −1 , 0) , ( −1 , 0) : (0 , −1) , _�→_ (0 , −1) : (1 , 0) _}_ r e t u r n l i s t ( mapping dict [ f a c e d i r e c t ] ) def r i g h t r o t a t e ( f a c e d i r e c t ) : f a c e d i r e c t = t u p l e ( f a c e d i r e c t ) mapping dict = _{_ (1 , 0) : (0 , −1) , (0 , 1) : (1 , 0) , ( −1 , 0) : (0 , 1) , _�→_ (0 , −1) : ( −1 , 0) _}_ r e t u r n l i s t ( mapping dict [ f a c e d i r e c t ] ) def a r o u n d r o t a t e ( f a c e d i r e c t ) : f a c e d i r e c t = t u p l e ( f a c e d i r e c t ) mapping dict = _{_ (1 , 0) : ( −1 , 0) , (0 , 1) : (0 , −1) , ( −1 , 0) : (1 , 0) , _�→_ (0 , −1) : (0 , 1) _}_ r e t u r n l i s t ( mapping dict [ f a c e d i r e c t ] ) def move steps ( c u r r e n t p o s i t i o n , f a c e d i r e c t , step , s t e p d i r e c t = ’ ’) : n e w l i s t = [ ] i f s t e p d i r e c t == ’ l e f t ’ : f a c e d i r e c t = l e f t r o t a t e ( f a c e d i r e c t ) e l i f s t e p d i r e c t == ’ r i g h t ’ : f a c e d i r e c t = r i g h t r o t a t e ( f a c e d i r e c t ) e l i f s t e p d i r e c t == ’ backward ’ : f a c e d i r e c t = a r o u n d r o t a t e ( f a c e d i r e c t ) f o r i in range ( len ( c u r r e n t p o s i t i o n ) ) : n e w l i s t . append ( c u r r e n t p o s i t i o n [ i ] + f a c e d i r e c t [ i ] * s t e p ) r e t u r n n e w l i s t # I n s t r u c t i o n : Take 1 s t e p . Take 2 s t e p s . Take 3 s t e p s . Turn around . _�→_ Take 6 s t e p s . Turn l e f t . # Python code , r e t u r n ans s t a r t p o s i t i o n = [0 , 0] c u r r e n t p o s i t i o n = s t a r t p o s i t i o n # assume the s t a r t face i s to x−p o s i t i v e f a c e d i r e c t = [1 , 0] # Take 1 s t e p . c u r r e n t p o s i t i o n = move steps ( c u r r e n t p o s i t i o n , f a c e d i r e c t , 1) # Take 2 s t e p s . c u r r e n t p o s i t i o n = move steps ( c u r r e n t p o s i t i o n , f a c e d i r e c t , 2) # Take 3 s t e p s . c u r r e n t p o s i t i o n = move steps ( c u r r e n t p o s i t i o n , f a c e d i r e c t , 3) # Turn around . f a c e d i r e c t = a r o u n d r o t a t e ( f a c e d i r e c t ) # Take 6 s t e p s . c u r r e n t p o s i t i o n = move steps ( c u r r e n t p o s i t i o n , f a c e d i r e c t , 6) # Turn l e f t . Now the face i s to y−p o s i t i v e f a c e d i r e c t = l e f t r o t a t e ( f a c e d i r e c t ) ans = ’ yes ’ i f c u r r e n t p o s i t i o n == s t a r t p o s i t i o n e l s e ’no ’ . . . 

Table 24: Navigate PoT task demonstrations 

31 

Published as a conference paper at ICLR 2024 

## **CREPE CoT** 

Answer the q u e s t i o n . The f i n a l answers are ”more l i k e l y ” , ” e q u a l l y _�→_ l i k e l y ” and ” l e s s l i k e l y ” . Goal : Wash sn e a k er s C u r r e n t C o n t e x t : Brush o f f d i r t from the s u r f a c e of the s n e a k e r s . Remove _�→_ s h o e l a c e s . Rinse the s h o e l a c e s in soapy water and a i r dry . Apply _�→_ mild d e t e r g e n t to the s n e a k e r s and rub g e n t l y . Question : What ’ s the l i k e l i h o o d t h a t The s n e a k e r s are damp? Explain : For s t e p 1 , a f t e r brushing o f f d i r t , the s n e a k e r s are damp i s ” e q u a l l y _�→_ l i k e l y ” because t h e r e i s no change f o r the dampness of the shoes . For s t e p 2 , a f t e r removing shoelaces , the s n e a k e r s are damp i s ” e q u a l l y _�→_ l i k e l y ” because t h e r e i s no change f o r the dampness of the shoes . For s t e p 3 , a f t e r r i n s i n g the s h o e l a c e s in soapy water and a i r dry , the _�→_ s n ea k e r s are damp i s ” e q u a l l y l i k e l y ” because we removed the _�→_ s h oe l a ce from the sneaker so we don ’ t change the dampness of _�→_ sneaker . For s t e p 4 , a f t e r applying mild d e t e r g e n t to the s n e a k e r s and rubbing _�→_ gently , the s n e ak e r s are damp i s ”more l i k e l y ” because we put mild _�→_ d e t e r g e n t to the s n e a k e r s . Therefore , in the f i n a l step , the event i s ”more l i k e l y ” . Answer : more l i k e l y . . . 

Table 25: CREPE CoT task demonstrations 

32 

Published as a conference paper at ICLR 2024 

## **CREPE PoT** 

# Answer the q u e s t i o n . The answers are ”more l i k e l y ” , ” e q u a l l y l i k e l y ” _�→_ and ” l e s s l i k e l y ” . # Goal : Wash s ne a k e rs # C u r r e n t C o n t e x t : Brush o f f d i r t from the s u r f a c e of the s n e a k e r s . _�→_ Remove s h o e l a c e s . Rinse the s h o e l a c e s in soapy water and a i r dry . _�→_ Apply mild d e t e r g e n t to the s n e a k e r s and rub g e n t l y . # Question : What ’ s the l i k e l i h o o d t h a t The s n e a k e r s are damp? # Python code c l a s s Wash Sneakers ( ) : # I n i t from C u r r e n t C o n t e x t # Brush o f f d i r t from the s u r f a c e of the s n e a k e r s # Remove s h o e l a c e s # Rinse the s h o e l a c e s in soapy water and a i r dry # Apply mild d e t e r g e n t to the s n e a k e r s and rub g e n t l y . def i n i t ( s e l f ) : s e l f . event0 = None # event0 i s the l i k e l i h o o d t h a t The s n e a k e r s _�→_ are damp . def b r u s h o f f d i r t ( s e l f ) : # After brushing o f f d i r t , event0 becomes ” e q u a l l y l i k e l y ” s e l f . event0 = ” e q u a l l y l i k e l y ” def remove shoelaces ( s e l f ) : # After removing shoelaces , event0 becomes ” e q u a l l y l i k e l y ” s e l f . event0 = ” e q u a l l y l i k e l y ” def r i n s e s h o e l a c e i n w a t e r ( s e l f ) : # After r i n s i n g in water , event0 becomes ” e q u a l l y l i k e l y ” s e l f . event0 = ” e q u a l l y l i k e l y ” def a p p l y d e t e r g e n t s n e a k e r ( s e l f ) : # After r i n s i n g in water , event0 becomes ”more l i k e l y ” s e l f . event0 = ”more l i k e l y ” def c a l l a l l f u n c s i n o r d e r ( s e l f ) : s e l f . b r u s h o f f d i r t ( ) s e l f . remove shoelaces ( ) s e l f . r i n s e s h o e l a c e i n w a t e r ( ) s e l f . a p p l y d e t e r g e n t s n e a k e r ( ) c = Wash Sneakers ( ) c . c a l l a l l f u n c s i n o r d e r ( ) ans = c . event0 . . . 

Table 26: CREPE PoT task demonstrations 

## **StrategyQA CoT** 

Question : Are more people today r e l a t e d to Genghis Khan than J u l i u s _�→_ Caesar ? Explain : J u l i u s Caesar had t h r e e c h i l d r e n . Genghis Khan had s i x t e e n c h i l d r e n . The c h i l d r e n number of Genghis Khan i s l a r g e r than J u l i u s Caesar . Answer : Yes 

Table 27: StrategyQA CoT task demonstrations 

33 

Published as a conference paper at ICLR 2024 

## **StrategyQA PoT** 

Question : Are more people today r e l a t e d to Genghis Khan than J u l i u s _�→_ Caesar ? # Python code , r e t u r n ans c h i l d r e n n u m o f J u l i u s C a e s a r = 3 children num of Genghis Khan = 6 ans = ( children num of Genghis Khan _>_ c h i l d r e n n u m o f J u l i u s C a e s a r ) 

Table 28: StrategyQA PoT task demonstrations 

## **GSM8k Question Difficulty Prediction** 

P r e d i c t the hardness l e v e l of the q u e s t i o n s . Question : Shawna ’ s workout goal i s 30 s i t u p s . On Monday , Shawna was only _�→_ able to do 12 s i t u p s , so she decided t h a t she would make up f o r _�→_ the r e s t on Tuesday . However , she was only able to do 19 s i t u p s on _�→_ Tuesday . How many s i t u p s would Shawna have to do on Wednesday to _�→_ meet her minimum goal and make up f o r the ones she didn ’ t do? Level : Hard . . . 

Table 29: GSM8k question difficulty prediction task demonstrations 

## **GSM8k Question and Answer Difficulty Prediction** 

Generate feedback and p r e d i c t i f the g e n e r a t e d answer i s t r u s t f u l . Question : Shawna ’ s workout goal i s 30 s i t u p s . On Monday , Shawna was only _�→_ able to do 12 s i t u p s , so she decided t h a t she would make up f o r _�→_ the r e s t on Tuesday . However , she was only able to do 19 s i t u p s on _�→_ Tuesday . How many s i t u p s would Shawna have to do on Wednesday to _�→_ meet her minimum goal and make up f o r the ones she didn ’ t do? Answer : Shawna ’ s goal i s 30 s i t u p s , and she has a l r e a d y done 12 + 19 = _�→ <<_ 12+19=31 _>>_ 31 s i t u p s . _\_ nTo meet her goal , she needs to do 30 −31 _�→_ = _<<_ 30−31=−1 _>>_ −1 s i t u p s . _\_ nSince she can ’ t do n e g a t i v e s i t u p s , she _�→_ doesn ’ t need to do any more s i t u p s to meet her goal . _\_ nans = 0 Feedback : The answer i s i n c o r r e c t . The goal i s 30 s i t u p s everyday . So _�→_ the t o t a l i s 3*30 = 90 r a t h e r than 30. T r u s t f u l : No . . . 

Table 30: GSM8k question and answer difficulty prediction task demonstrations 

34 

Published as a conference paper at ICLR 2024 

## **Calibration** 

Question : Shawna ’ s workout goal i s 30 s i t u p s . On Monday , Shawna was only _�→_ able to do 12 s i t u p s , so she decided t h a t she would make up f o r _�→_ the r e s t on Tuesday . However , she was only able to do 19 s i t u p s on _�→_ Tuesday . How many s i t u p s would Shawna have to do on Wednesday to _�→_ meet her minimum goal and make up f o r the ones she didn ’ t do? Proposed Answer : Shawna ’ s goal i s 30 s i t u p s , and she has a l r e a d y done 12 _�→_ + 19 = _<<_ 12+19=31 _>>_ 31 s i t u p s . _\_ nTo meet her goal , she needs to do _�→_ 30 −31 = _<<_ 30−31=−1 _>>_ −1 s i t u p s . _\_ nSince she can ’ t do n e g a t i v e _�→_ s i t u p s , she doesn ’ t need to do any more s i t u p s to meet her _�→_ goal . _\_ nans = 0 I s the proposed answer : (A) True (B) False The proposed answer i s : B 

Table 31: Calibration task demonstrations 

## **DATE Question Difficulty Prediction** 

P r e d i c t the hardness l e v e l of the q u e s t i o n s . Question : Today i s Christmas Eve of 1937. What i s the date tomorrow in _�→_ MM/DD/YYYY? Level : Easy . . . 

Table 32: DATE question difficulty prediction task demonstrations 

## **DATE Question and Answer Difficulty Prediction** 

Generate feedback and p r e d i c t i f the g e n e r a t e d answer i s t r u s t f u l . Question : Today i s Christmas Eve of 1937. What i s the date tomorrow in _�→_ MM/DD/YYYY? Explain : Today i s the Christmas Eve of 1937 , so today i s 12 /2 4/ 19 37 . Today i s 12/24/1937 , the date tomorrow i s 12 /2 5/ 19 37 . A: 12/25/1937 Feedback : The answer i s c o r r e c t . T r u s t f u l : Yes . . . 

Table 33: DATE question and answer difficulty prediction task demonstrations 

35 

Published as a conference paper at ICLR 2024 

## **CREPE Question Difficulty Prediction** 

P r e d i c t the hardness l e v e l of the q u e s t i o n s . Goal : Making Matcha Green Tea the T r a d i t i o n a l Way C u r r e n t C o n t e x t : Boil 3/4 cup (180 ml ) of water and pour i t i n t o your _�→_ matcha bowl . Whisk the hot water with a chasen . Discard the hot _�→_ water from the bowl . Scoop 2 t s p ( 1 . 5 heaping teaspoons or 10 g ) _�→_ of matcha i n t o a f i n e mesh s t r a i n e r . S i f t the matcha i n t o your _�→_ empty , dry bowl . Pour b o i l i n g water i n t o a teacup . Add a small _�→_ amount of hot water i n t o the matcha bowl and whisk i t . Question : What ’ s the l i k e l i h o o d t h a t I drink the matcha ? Level : Easy . . . 

Table 34: CREPE question difficulty prediction task demonstrations 

## **CREPE Question and Answer Difficulty Prediction** 

Generate feedback and p r e d i c t i f the g e n e r a t e d answer i s t r u s t f u l . Goal : Boating in lake C u r r e n t C o n t e x t : Rent a boat . Find a lake which allows b o a t i n g . Drive to _�→_ the lake and s e t the boat in lake . Question : What ’ s the l i k e l i h o o d t h a t the boat i s in lake ? Explain : For s t e p 1 , a f t e r r e n t i n g a boat , t h a t the boat i s in the lake _�→_ i s ” l e s s l i k e l y ” because the boat i s not yet in the lake . For s t e p 2 , a f t e r f i n d i n g a lake which allows boating , t h a t the boat i s _�→_ in the lake i s ” e q u a l l y l i k e l y ” because we have found a lake but _�→_ the boat i s not yet in the lake . For s t e p 3 , a f t e r d r i v i n g to the lake and s e t t i n g the boat in the lake , _�→_ t h a t the boat i s in the lake i s ”more l i k e l y ” because we have s e t _�→_ the boat in the lake . Therefore , in the f i n a l step , the event i s ”more l i k e l y ” . Feedback : The answer i s c o r r e c t . T r u s t f u l : Yes . . . 

Table 35: CREPE question and answer difficulty prediction task demonstrations 

## **GSM8k Hints** 

Given the h i n t s ( may not be c o r r e c t ) , answer the q u e s t i o n . Question : Tara has a shoebox t h a t i s 4 i nch es t a l l and 6 inch es wide . _�→_ She puts a square block i n s i d e t h a t i s 4 i nch es per s i d e . How many _�→_ square inches of the box are l e f t uncovered ? Hints : The answer may be near to 24 or 8 . Answer : The shoebox i s 24 square inch es because 4 x 6 = _<<_ 4*6=24 _>>_ 24 The block i s 16 square inche s because 4 x 4 = _<<_ 4*4=16 _>>_ 16 There are 8 square inches uncovered because 24 −16 = _<<_ 24−16=8 _>>_ 8 ans = 8 . . . 

Table 36: GSM8k with hints task demonstrations 

36 

Published as a conference paper at ICLR 2024 

## **DATE Hints** 

Given the h i n t s ( may not be c o r r e c t ) , answer the q u e s t i o n . Question : Jane was born on the l a s t day of Feburary in 2000. Today i s _�→_ her 16−year −old b i r t h d a y . What i s the date 24 hours l a t e r in _�→_ MM/DD/YYYY? Hints : The answer may be near 02/29/2016 or 03 /0 1/ 20 16 . Answer : Jane was born on the l a s t day of Feburary in 2000 , so she born _�→_ on 02/29/2000. Today i s her 16−year −old b i r t h d a y . So today i s 02 /2 9/ 20 16 . The date 24 hours l a t e r than today i s 0 3/ 01 /2 01 6. A: 03/01/2016 . . . 

Table 37: DATE with hints task demonstrations 

## **CREPE Hints** 

Given the h i n t s ( which may not be c o r r e c t ) , answer the q u e s t i o n with _�→_ ”more l i k e l y ” , ” e q u a l l y l i k e l y ” or ” l e s s l i k e l y ” . Goal : Buy a f l i g h t t i c k e t from SF to Hawaii . Context : Open google f l i g h t s and check the f l i g h t s a v a i l a b i l i t y during _�→_ the d a t e s wanted . Choose non−stop f l i g h t s and check i f the time _�→_ s l o t s work . After d e c i d i n g r i g h t time s l o t s and price , proceed to _�→_ buy t i c k e t s . Got an email c o n f i r m a t i o n of r e s e r v a t i o n . Question : Jane was born on the l a s t day of Feburary in 2000. Today i s _�→_ her 16−year −old b i r t h d a y . What i s the date 24 hours l a t e r in _�→_ MM/DD/YYYY? Hints : The answer may be near ”more l i k e l y ” or ” l e s s l i k e l y ” . Explain : For s t e p 1 , a f t e r checking f l i g h t s a v a i l a b i l i t y , t h a t I can _�→_ refund the t i c k e t i f i t i s r e f u n d a b l e i s ” e q u a l l y l i k e l y ” because _�→_ I haven ’ t purchased the t i c k e t yet . For s t e p 2 , a f t e r choosing non−stop f l i g h t s and checking i f the time _�→_ s l o t s work , t h a t I can refund the t i c k e t i f i t i s r e f u n d a b l e i s _�→_ ” e q u a l l y l i k e l y ” because I haven ’ t purchased the t i c k e t yet . For s t e p 3 , a f t e r d ec i di n g on the r i g h t time s l o t s and p r i c e and _�→_ proceeding to buy t i c k e t s , t h a t I can refund the t i c k e t i f i t i s _�→_ r e f u n d a b l e i s ” l e s s l i k e l y ” because I have a l r e a d y purchased the _�→_ t i c k e t . For s t e p 4 , a f t e r g e t t i n g an email c o n f i r m a t i o n of r e s e r v a t i o n , t h a t I _�→_ can refund the t i c k e t i f i t i s r e f u n d a b l e i s ”more l i k e l y ” because _�→_ I can check the terms and c o n d i t i o n s of the t i c k e t and see i f i t _�→_ i s r e f u n d a b l e or not . Therefore , in the f i n a l step , the event i s ”more l i k e l y ” . Answer : more l i k e l y . . . 

Table 38: CREPE with hints task demonstrations 

37 

Published as a conference paper at ICLR 2024 

## **Batch Prompting** 

_{_ Question1 : A robe t a k e s 2 b o l t s of blue f i b e r and h a l f t h a t much white _�→_ f i b e r . How many b o l t s in t o t a l does i t take ? Question2 : Josh d ecides to t r y f l i p p i n g a house . He buys a house f o r _�→_ $80 ,000 and then puts in $50 ,000 in r e p a i r s . This i n c r e a s e d the _�→_ value of the house by 150%. How much p r o f i t did he make? Question3 : Every day , Wendi feeds each of her chickens t h r e e cups of _�→_ mixed chicken feed , c o n t a i n i n g seeds , mealworms and v e g e t a b l e s to _�→_ help keep them h e a l t h y . She gives the chickens t h e i r feed in _�→_ t h r e e s e p a r a t e meals . In the morning , she gives her f l o c k of _�→_ chickens 15 cups of feed . In the afternoon , she gives her _�→_ chickens another 25 cups of feed . How many cups of feed does she _�→_ need to give her chickens in the f i n a l meal of the day i f the s i z e _�→_ of Wendi ’ s f l o c k i s 20 chickens ? Question4 : Kylar went to the s t o r e to buy g l a s s e s f o r h i s new apartment . _�→_ One g l a s s c o s t s $5 , but every second g l a s s c o s t s only 60% of the _�→_ p r i c e . Kylar wants to buy 16 g l a s s e s . How much does he need to pay _�→_ f o r them ? Answer1 : I t t a k e s 2/2= _<<_ 2/2=1 _>>_ 1 b o l t of white f i b e r _\_ nSo the t o t a l _�→_ amount of f a b r i c i s 2+1= _<<_ 2+1=3 _>>_ 3 b o l t s of f a b r i c ans = 3 Answer2 : The c o s t of the house and r e p a i r s came out to _�→_ 80 ,000+50 ,000=$ _<<_ 80000+50000=130000 _>>_ 130,000 _\_ nHe i n c r e a s e d the _�→_ value of the house by 80 ,000*1.5= _< <_ 80000*1.5=120000 _> >_ 120 ,000 _\_ nSo _�→_ the new value of the house i s _�→_ 120 ,000+80 ,000=$ _<<_ 120000+80000=200000 _>>_ 200,000 _\_ nSo he made a _�→_ p r o f i t of 200 ,000 −130 ,000=$ _<<_ 200000−130000=70000 _>>_ 70,000 ans = 70000 Answer3 : I f each chicken e a t s 3 cups of feed per day , then f o r 20 _�→_ chickens they would need 3*20= _<<_ 3*20=60 _>>_ 60 cups of feed per _�→_ day . _\_ n I f she feeds the f l o c k 15 cups of feed in the morning , and _�→_ 25 cups in the afternoon , then the f i n a l meal would r e q u i r e _�→_ 60−15−25= _<<_ 60−15−25=20 _>>_ 20 cups of chicken feed . ans = 20 Answer4 : The d i s c o u n t p r i c e of one g l a s s i s 60/100 * 5 = _�→_ $ _<<_ 60/100*5=3 _>>_ 3. _\_ n I f every second g l a s s i s cheaper , t h a t means _�→_ Kylar i s going to buy 16 / 2 = _<<_ 16/2=8 _>>_ 8 cheaper g l a s s e s . _\_ nSo _�→_ f o r the cheaper g l a s s e s , Kylar i s going to pay 8 * 3 = _�→_ $ _<<_ 8*3=24 _>>_ 24. _\_ nAnd f o r the r e g u l a r −p r i c e d g l a s s e s , Kylar w i l l pay _�→_ 8 * 5 = $ _<<_ 8*5=40 _>>_ 40. _\_ nSo in t o t a l Kylar needs to pay 24 + 40 = _�→_ $ _<<_ 24+40=64 _>>_ 64 f o r the g l a s s e s he wants to buy . ans = 64 _}_ . . . 

Table 39: Batch Prompting task demonstrations 

38 

