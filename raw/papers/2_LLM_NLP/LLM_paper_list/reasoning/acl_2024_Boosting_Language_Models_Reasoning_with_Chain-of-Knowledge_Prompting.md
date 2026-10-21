# **Boosting Language Models Reasoning with Chain-of-Knowledge Prompting** 

**Jianing Wang**[1] _[∗]_ **Qiushi Sun**[2] _[∗]_ **Nuo Chen**[1] **Xiang Li**[1] _[†]_ **Ming Gao**[1] 

1 School of Data Science and Engineering, East China Normal University 2 National University of Singapore `lygwjn@gmail.com, qiushisun@u.nus.edu, xiangli@dase.ecnu.edu.cn` 

## **Abstract** 

Recently, Chain-of-Thought (CoT) prompting has delivered success on complex reasoning tasks, which aims at designing a simple prompt like “ _Let’s think step by step_ ” or multiple in-context exemplars with well-designed rationales to elicit Large Language Models (LLMs) to generate intermediate reasoning steps. However, the generated rationales often come with mistakes, making unfactual and unfaithful reasoning chains. To mitigate this brittleness, we propose a novel Chain-of-Knowledge (CoK) prompting, where we aim at eliciting LLMs to generate explicit pieces of knowledge evidence in the form of structure triple. This is inspired by our human behaviors, i.e., we can draw a mind map or knowledge map as the reasoning evidence in the brain before answering a complex question. Benefiting from CoK, we additionally introduce a F[2] -Verification method to estimate the reliability of the reasoning chains in terms of _factuality_ and _faithfulness_ . For the unreliable response, the wrong evidence can be indicated to prompt the LLM to rethink. Extensive experiments demonstrate that our method can further improve the performance of commonsense, factual, symbolic, and arithmetic reasoning tasks[3] . 

## **1 Introduction** 

Large Language Models (LLMs) have succeeded in advancing the state-of-the-arts for many Natural Language Processing (NLP) tasks [Devlin et al., 2019, Brown et al., 2020, Rae et al., 2021, Thoppilan et al., 2022, Chowdhery et al., 2022, Scao et al., 2022, Zhang et al., 2022b, Bai et al., 2022, Touvron et al., 2023], benefiting from the ultra-large-scale training corpora and computation resources. To unleash the LLMs’ power of adaptation on unseen tasks without any parameter updates, in-context learning (ICL) has become one of the flourishing research topics, aiming at generating the prediction by conditioning on a few labeled exemplars (Figure 1 (a)) [Dong et al., 2023, Liu et al., 2022, Zhao et al., 2021, Shin et al., 2022, Lu et al., 2022]. 

A series of recent works have explored that LLMs can spontaneously decompose the complex multi-step problem into intermediate reasoning chains , elicited by a simple prompt like “ _Let’s think step by step_ ” or well-designed demonstrations with human-annotated rationales, which are called chain-of-thought (CoT) prompting (Figure 1 (b)) [Wei et al., 2022, Kojima et al., 2022, Wang et al., 2023b, Zhou et al., 2023]. This finding is intriguing and has been sensational because CoT may mainly specify an output space/format that regularizes the model generation to look step-by-step while being in order and relevant to the query [Wang et al., 2023a, Min et al., 2022b, Yoo et al., 2022]. 

Despite impressive performances, current LLMs are susceptible to generating unreliable answers [Ji et al., 2023, Yin et al., 2023], along with providing unfactual or unfaithful reasoning chains that 

> _∗_ J. Wang and Q. Sun contributed equally to this work. 

> _†_ Corresponding author. 

> 3The code and data will be released upon acceptance. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting_images/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Input Input Input<br>Q: Is the following sentence plausible? Q: Is the following sentence plausible? Q: Is the following sentence plausible? 'Joao Moutinho was out<br>'Joao Moutinho was out at third.'  'Joao Moutinho was out at third.'  at third.'<br>A: No.  Joao Moutinho is a soccer player. Evidence triples:<br>Being out at third is part of baseball, 1. (Joao, isA, soccer player)<br>Q: Is the following sentence plausible? not soccer.  2. (being out at third, is part of, baseball)<br>'Derrick White backhanded a shot.'<br>A: No.  Explanation hints: Being out at third is part of baseball, yet,<br>Joao Moutinho is a soccer player.<br>Q: Is the following sentence plausible?<br>'Derrick White backhanded a shot.'  A: No.<br>Q: Is the following sentence plausible? 'Derrick White<br>backhanded a shot.'<br>Output Output Output<br>A: Yes.  ❌ A: Yes, it is plausible. This sentence Evidence triples:<br>suggests that Derrick White, who is 1. (Derrick White, isA, basketball player)<br>most likely a basketball or hockey 2. (backhanded shot, is commonly used in, hockey or tennis)<br>player, made a backhanded shot<br>attempt.  ❌ Explanation hints: Backhanded shot is commonly used in<br>hockey or tennis, but not in basketball.<br>A: No.  ✅<br>(a) Standard ICL Prompting (b) Chain-of-Thought Prompting (c) Ours: Chain-of-Knowledge Prompting<br>**----- End of picture text -----**<br>


Figure 1: Comparison of three prompting methods: (a) ICL, (b) Chain-of-Thought (CoT), and (c) Chain-of-Knowledge (CoK) solving a StrategyQA question. 

inevitably lead to a wrong conclusion. Take Figure 1 as an example. Given a query “ _Is the following sentence plausible ‘Derrick White backhanded a shot.’_ ” from StrategyQA [Geva et al., 2021], the standard ICL and CoT make a wrong answer. One of the reasoning steps “Derrick White is most likely a hockey” is fake (In fact, Derrick White is a basketball player), making the unfactual inference towards the question. In addition, the response may be unfaithful when the LLM generates logically sound reasoning chains while still providing an incorrect answer. 

To address these concerns, we propose a novel **Chain-of-Knowledge (CoK)** prompting method to boost the LLM’s reasoning capability by a series of exemplars that combine explicit structure knowledge evidence with textual explanations. To elaborate, CoK prompting consists of two compositions (Figure 1 (c)), i.e., evidence triples (CoK-ET) and explanation hints (CoK-EH), where CoK-ET is a list of structure triples can reflect the overall reasoning evidence from the query towards the answer and CoK-EH is the explanation of this evidence. To construct in-context exemplars with the CoK prompt, we first sample _K_ labeled examples and each of them can be concatenated with a simple hint “ _Let’s think step by step_ ” to prompt the LLM to generate reasoning chains. Then, we retrieve some structure triples from the external knowledge base (KB) and judiciously manually annotate evidence triples to obtain a well-designed CoK prompt. Like standard ICL and CoT, the CoK prompt can be perceived as a rule that regularizes the output space/format and urges LLMs to generate explicit evidence instead of only attempting to generate vague textual reasoning chains. 

Furthermore, we propose an F[2] -Verification strategy to estimate the reliability of the reasoning chains in terms of _factuality_ and _faithfulness_ , where _factuality_ is the quantification of the matching degree between reasoning evidence and ground-truth knowledge, and _faithfulness_ is the consistency degree between reasoning evidence and the textual explanation with the final answer. Particularly for the unreliable response, the wrong pieces of evidence can be indicated to prompt the LLM to rethink this problem. We design a _rethinking process algorithm_ to reach this goal. 

We have conducted empirical experiments across various reasoning tasks (e.g., commonsense, factual, arithmetic, and symbolic), showing that CoK prompting significantly outperforms standard ICL and CoT prompting. This indicates LLMs can be better enhanced by CoK prompting and post-verification. 

## **2 Related Work** 

**Prompting for LLMs with in-context learning.** In-Context Learning (ICL) is the task of causal language modeling, allowing LLMs to perform zero/few-shot learning with a well-designed text-based prompt [Brown et al., 2020, Chowdhery et al., 2022, Touvron et al., 2023, Thoppilan et al., 2022, Dong et al., 2023]. ICL can bypass the model parameter update and achieve the salient performance by conditioning on a few labeled examples. Previous works have explored some impact facets of ICL. For example, the input-output mapping and the template format [Pan et al., 2023, Min et al., 2022b, 

2 

Yoo et al., 2022], the different selection and permutations of the exemplars [Lu et al., 2022]. To improve ICL’s effectiveness, some novel methods have been proposed, involving meta-learning [Chen et al., 2022b, Min et al., 2022a], prompt and exemplars engineering [Liu et al., 2022, 2023a], obtaining LM preferences by the interaction between inputs and in-context examples [Ye et al., 2023], finding appropriate example organization through self-adaption mechanism [Wu et al., 2023], and prediction calibrating [Zhao et al., 2021], etc. Different from them, our work focuses on the reasoning ability of LLMs in two aspects, including reasoning prompt design and output verification. 

**Chain-of-thought prompting elicits reasoning.** Recently, CoT prompting has been presented to leverage reasoning and interpretable information to guide LLMs to generate reliable responses [Wei et al., 2022]. A series of CoT-enhanced methods are proposed to further improve the reasoning ability [Wang et al., 2023b, Zhou et al., 2023, Wang et al., 2022, Kojima et al., 2022, Zhang et al., 2023, Si et al., 2022, Huang et al., 2022]. For example, Wang et al. [2023b] introduce _Self-consistency_ to suppress the wrong rationales problem by replacing the greedy decoding strategy with marginalizing out the sampled reasoning paths to find the most consistent answer. _Program of Thoughts_ Chen et al. [2022a] employ a program interpreter to delegate the computation, aiming at decoupling computational processes from reasoning and understanding. In addition, a series of works improve the reasoning performance by the verification, which is similar to ours. For example, [Lyu et al., 2023] translate the complex problem into interleaved natural language or programming language to make the reasoning chains faithful. [Li et al., 2023] and [Asai and Hajishirzi, 2020] introduce coarse or fine-grained labels to verify the reasoning chains. In contrast, we focus on both the factuality and faithfulness of the reasoning chains and calibrate the prediction based on the KB. 

**Knowledge utilization for LLM reasoning.** While LLMs possess substantial capabilities, they suffer from the missing real-world grounded experience [Petroni et al., 2020] and the inability to access cutting-edge information [Komeili et al., 2022] as their learning is solely based on written text. Thus, researchers start utilizing external knowledge and tools to assist models in accomplishing reasoning tasks [Nakano et al., 2022, Gao et al., 2022, Schick et al., 2023]. Lately, Liu et al. [2023b] innovatively alleviates this limitation by integrating a computational physics engine into the language modeling process to simulate potential outcomes, which are further utilized as part of the input to drive models to perform reasoning. _Chameleon_ [Lu et al., 2023] augments LLMs by incorporating various tools and knowledge resources (e.g., web search engines, Python functions, and task-specific modules) by synthesizing programs. It strategically composes and executes a suitable sequence of tools, thereby delivering the final response for tackling long-tail and domain-specific tasks. 

Previous efforts to inject external knowledge into models have typically relied on incorporating knowledge into the prompt prior to the model inference stage. We hold the view that the knowledge acquired through this procedure may not necessarily align with the desiderata of LLMs. Therefore, our proposed CoK encourages the model to explicitly generate evidence triples, followed by identifying erroneous triples through post-verification. The corresponding correct knowledge is then derived during the _rethinking process_ , thus achieving meticulous knowledge injection. 

## **3 Methodology** 

The generated reasoning chains elicited by CoT prompting sometimes come with mistakes, ultimately leading to the wrong answer. We attribute this challenge to the textual reasoning chain: _LLMs may forcibly generate a textual rationale that conforms to the prompt format of CoT but is logically ambiguous and reaches the wrong answer_ . To address this challenge, we thus consider two core aspects. 1) Elicitation format of the prompt. Text-only reasoning chains are not enough to unleash LLMs to generate reliable and concrete reasoning processes. Inspired by the triple structure in the KB, we need to enhance the prompt with structured features. 2) Post-verification. LLMs are usually not capable of inspecting what answers they had responded to, indicating us to leverage external knowledge to make verification. Based on these considerations, we provide our specific solution on how to boost LLM’s reasoning ability. The overview of the framework is shown in Figure 2. 

## **3.1 Chain-of-Knowledge Prompting** 

It is widely recognized that reasoning can be modeled as induction and deduction on the existing knowledge system [Goswami, 2002]. This is inspired by human behaviors that draw a mind map 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting_images/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Exemplar #1<br>Q: Is the following sentenceplausible? 'Joao Moutinho Exemplar # 1 Exemplar # 2 ... Exemplar # K<br>was out at third.'<br>A: Let's think step by step. Test Example<br>Zero-shot Reasoning Knowledge Base Q: Is the following sentence plausible?'Derrick White backhanded a shot.' Knowledge Base<br>        LLM<br>Chain-of-Thought Prompt Factuality Verification<br>Retrieving Annotators         LLM If unreliable,<br>Chain-of-Knowledge let's rethink. 1. (Derrick White, isA, bask-etball player)   ✅<br>Exemplar #1 Output 2. (backhanded shot, is co-<br>Q: Is the following sentence plausible? 'Joao Moutinhowas out at third.'  Evidence triples: 1. (Derrick White, isA, basketball player)  mmonly used in, hockey ortennis)  ✅<br>2. (backhanded shot, is commonly used<br>Evidence triples:  in, hockey or tennis)  1.0<br>1. (Joao, isA, soccer player)<br>2. (being out at third, is part of, baseball)  Explanation hints: Backhanded shot is  Faithfulness Verification 0.9<br>commonly used in hockey or tennis, but<br>Explanation hints: Being out at third is part of baseball, not in basketball.<br>yet, Joao Moutinho is a soccer player.  If reliable, output the answer.<br>A: No.<br>A: No.  No.<br>Exemplars Construction Chain-of-Knowledge Reasoning F [2] -Verification<br>**----- End of picture text -----**<br>


Figure 2: The proposed framework. We first construct exemplars with chain-of-knowledge (CoK) prompts. Then, the CoK prompts can be indicated to let the LLM generate reasoning chains, including evidence triples, explanation hints, and the final answer. At last, we estimate the reliability of reasoning chains in terms of _factuality_ and _faithfulness_ , and the unreliable ones will be rethought. 

or knowledge map to analyze the question and find the correct path to the answer. Fortunately, we can adopt the concept of the triple in the KB, which can be viewed as a “(subject, relation, object)”, to formalize the explicit evidence of the reasoning chains. To elaborate, we propose **Chain-ofKnowledge** (CoK) prompting to facilitate a better elicitation prompt for LLMs, which consists of two main ingredients, i.e., evidence triples (CoK-ET) and explanation hints (CoK-EH). CoK-ET represents a list of multiple triples and each of them represents the knowledge evidence probed from LLMs to support the step-by-step thinking process. CoK-EH denotes the explanation of the reasoning chain, which is similar to the vanilla CoT. For example, in Figure 1, we can urge the LLM to generate two explicit shreds of evidence “(Derrick White, isA, basketball player), (basketball shot, is commonly used in, hockey or tennis)” and a textual explanation to support the final answer “No”. 

## **3.2 Exemplars Construction** 

Building upon the insights of previous studies [Zhang et al., 2023, Min et al., 2022b, Wang et al., 2023b], the performance of ICL hinges on the annotated rationale. This indicates that the key challenge of CoK prompting lies in constructing textual rationales with their structure evidence triples. As shown in Figure 2, we first perform exemplars construction to obtain a well-designed task-specific prompt. Specifically, we follow [Wei et al., 2022, Wang et al., 2023b] to randomly select _K_ questions as the basic demonstrations. To automatically obtain CoK-EH, we follow [Kojima et al., 2022] to generate a textual rationale for each question via zero-shot CoT with a simple prompt “ _Let’s think step by step_ ”. Another challenge is how to obtain annotated CoK-ET that better expresses the textual rationale. To figure it out, we first follow [Pan et al., 2022] to construct a KB _K_ from six domains, involving _dictionary_ , _commonsense_ , _entity_ , _event_ , _script_ , and _causality_ , which are in the form of triple. We then directly use the retrieving tool proposed by [Pan et al., 2022] to retrieve some candidate triples. Finally, we invite 5 teachers or Ph.D. students as professional annotators to manually design the corresponding CoK-ET based on the retrieved triples. More details can be found in Appendix C.1. 

In fact, during the exemplars construction, the generated textual reasoning chains and retrieved triples could have some mistakes. Fortunately, we found that there is no strong connection between the reasoning validity of both CoK-ET and CoK-EH and the performance of the model predictions, which is similar to findings in [Wang et al., 2023a]. We will bring detailed discussions at Section 4.3. 

## **3.3 F[2] -Verification** 

After the exemplars construction, we can obtain _K_ annotated data _E_ = _{_ ( _Qi, Ti, Hi, Ai_ ) _}[K] i_ =1[.] Symbolically, _Qi_ , _Hi_ and _Ai_ represent the input query, the explanation hint, and the final answer of 

4 

the _i_ -th exemplar, respectively; each of them is the token sequence. _Ti_ denotes the list of evidence triples, which contains multiple knowledge triples, i.e., _Ti_ = _{_ ( _sij, rij, oij_ ) _}j_ , where _sij_ , _rij_ and _oij_ are subject, relation and object, respectively. Given a test query input _Q_[ˆ] _i_ , we can directly choose one permutation of _E_ and concatenate them with this test query into a linear sequence _I_[ˆ] _i_ = [ _E_ ; _Q_[ˆ] _i_ ] to prompt the LLM to generate the prediction, i.e., _y_ ˆ _ik_ = arg max _σ_ (ˆ _yi_ ( _≤k_ )) _P_ ( _y|y_ ˆ _i_ ( _<k_ ) _, E, Q_[ˆ] _i_ ), where ˆ _P_ ( _y|·_ ) is the prediction distribution, _yik_ is the _k_ -th token, _σ_ ( _·_ ) denotes the decoding strategy (e.g., temperature sampling, beam search, and nucleus sampling), [ _·_ ; _·_ ] is the concatenation operation. 

ˆ Due to the well-designed format of the demonstrations, the final prediction derived from the LLM _yi_ consists of a list of evidence triples _T_[ˆ] _i_ , a sequence of explanation hints _H_[ˆ] _i_ and the final answer _A_[ˆ] _i_ . However, LLMs are not capable of inspecting the prediction, so the quality of the generated rationale and the final answer can not be guaranteed. We attribute this problem to two factors: 1) some steps in the rationale may not correspond to the fact, contributing to the wrongness, and 2) the relation between the final answer and the reasoning chains is still ambiguous, making the response unfaithful. To alleviate these drawbacks, we propose F[2] -Verification to estimate the answer reliability towards both **F** actuality and **F** aithfulness. 

**Factuality Verification.** We first verify the factuality, which can be viewed as the matching degree between each generated evidence triple and the ground-truth knowledge from KBs[4] . Concretely, we first define a function _fv_ (ˆ _rij|s_ ˆ _ij,_ ˆ _oij, K_ ) to represent the factuality of each evidence. We design two different strategies of _fv_ . 1) Exact verification. We can retrieve all relevant triples based on the subject ˆ _sij_ and object ˆ _oij_ , and then find whether the generated relation ˆ _rij_ . i.e., _fv_ (ˆ _rij|s_ ˆ _ij,_ ˆ _oij, K_ ) = I((ˆ _sij,_ ˆ _rij,_ ˆ _oij_ ) _∈K_ ) exists. 2) Implicit verification. For a triple that does not exist in KB, it could be corrected. Thus, we can transform the factuality verification into a graph completion task that predicts whether the triple is true. For simplicity, we use TransR [Lin et al., 2015] to pre-train the KB _K_ and use the off-the-shelf energy function to assign a score for each evidence triple, i.e., ˆ _fv_ (ˆ _rij|sij,_ ˆ _oij, K_ ) = _||_ **s**[(] _ij[r,c]_[)] + **r** _[c] −_ **o**[(] _ij[r,c]_[)] _||_ 2[2][+] _[ α][||]_ **[r]** _[c][ −]_ **[r]** _[ij][||]_[2] 2[, where] _[ α][>]_[0][ is the balancing factor,] _|| · ||_ 2[2][is Frobenius norm.] **[s]**[(] _ij[r,c]_[)] = **s** _ij_ **M** _r_ and **o**[(] _ij[r,c]_[)] = **o** _ij_ **M** _r_ denote the projection representations of the subject _sij_ and object _oij_ in the relation space _r_ , respectively. **s** _ij,_ **r** _ij_ , and **o** _ij ∈_ R _[d]_ are the _d_ -dimension embeddings of ˆ _sij_ , ˆ _rij_ and ˆ _oij_ , respectively. **r** _[c] ∈_ R _[d]_ is the prototype embeddings of the relation _r_ . **M** _r ∈_ R _[d][×][d]_ is the trainable projection matrix of relation _r_ . We join the two strategies in our framework. If the evidence triple exists in _K_ , we will use an exact verification strategy to assign a score, or we use an implicit verification strategy. 

**Faithfulness Verification.** As defined by [Jacovi and Goldberg, 2020, Lyu et al., 2023], _if the reasoning process derived from the model can accurately be expressed by an explanation, we call it faithful_ . Previous works based on chain-of-thought prompting are hard to verify faithfulness due to the lack of sufficient evidence to understand the relation between the explanation and the answer [Ye and Durrett, 2022]. So, we propose a faithfulness verification method to find out these cases. Specifically, given one test query _Q_[ˆ] _i_ , a list of evidence triples _T_[ˆ] _i_ and the final answer _A_[ˆ] _i_ , we directly concatenate them as a new sequence _H_[ˆ] _i[′]_[.][We leverage the pre-built sentence encoder] SimCSE [Gao et al., 2021] to calculate the similarity between _H_[ˆ] _i[′]_[and] _[H]_[ˆ] _[i]_[.][We denote this function as] _fu_ ( _H_[ˆ] _i|H_[ˆ] _i[′]_[= [ ˆ] _[Q][i]_[;] _[T]_[ˆ] _[i]_[;] _[A]_[ˆ] _[i]_[]) =] _[ SimCSE]_[( ˆ] _[H][i][,][H]_[ˆ] _i[′]_[)][.] 

Finally, for each query _Q_[ˆ] _i_ we can obtain a score _Ci_ (0 _< Ci <_ 1) that represents whether the rationale is reliable towards the answer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting_images/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting.pdf-0005-05.png)


where 0 _< γ <_ 1 is the balancing factor and set to be 0 _._ 5 as default, _|T_[ˆ] _i|_ is the number of triples. 

## **3.4 Rethinking Process** 

F[2] -Verification facilitates us to ensure the factuality and faithfulness of the triples and explanations generated by the model. Moreover, beyond the scope of verification, we can boost the performance 

> 4We assume that the knowledge from the KB is correct and up-to-date. 

5 

of LLMs even further by employing a _rethinking process_ . As shown in Algorithm 1, we initialize a reliability threshold _θ_ , iteration number _N_ , and an unreliability set _U_ . All the queries in the testing set _Dtest_ are unreliable at first. For each query _Q_[ˆ] _i ∈ U_ in the _n_ -th iteration, we obtain CoK prompt _I_ ˆ _i_[(] _[n]_[)] by combining the demonstrations _E_ and the query _Q_[ˆ] _i_ . The prompt can be used to elicit the LLM to generate a list of evidence triples _T_[ˆ] _i_[(] _[n]_[)] , explanation hints _H_[ˆ] _i_[(] _[n]_[)] , and the final answer _A_[ˆ][(] _i[n]_[)] . 

The proposed rethinking algorithm will allow the LLMs to assess the reliability of the rationales (i.e., _T_[ˆ] _i_[(] _[n]_[)] and _H_[ˆ] _i_[(] _[n]_[)] ) via calculating the score in Eq. 1. An entry is no longer considered unreliable if _Ci_[(] _[n]_[)] is not below the threshold _θ_ , which subsequently leads to the final answer _A_[ˆ] _i_ . Conversely, if _Ci_[(] _[n]_[)] fails to reach _θ_ , we can select the evidence triples with lower scores and inject the corresponding correct knowledge triples from the KB into the CoK prompt _I_[ˆ] _i_[(] _[n]_[+1)] in the next iteration (Line 12, Algorithm 1)[5] . 

This dynamic generate-evaluate procedure continues until all entries in _U_ are considered reliable or the maximum number of iterations _N_ is reached. For cases where the maximum number of iterations is reached without any triples’ reliability score surpassing _θ_ , triples with the highest reliability scores will be selected for inference (Line 15-17, Algorithm 1). 

**Algorithm 1** Rethinking Process 

|**Algorithm 1**RethinkingProcess|**Algorithm 1**RethinkingProcess||
|---|---|---|
|**Require:** Exemplars_E_, testing query set_Dtest ←{_ ˆ_Qi}M_<br>_i_=1,||KB|
|1: <br>2: <br>3:<br>4:<br>5:<br>6:<br>7:<br>8:<br>9:<br>10:|_K_, iterator number_N_(_≥_1), reliability threshold0_< θ <_1.<br> Initialize an unreliability set_U ←Dtest_.<br> **for**each iteration_n ←_1_, · · · , N_ **do**<br>**for**each query ˆ_Qi_ in_U_ **do**<br>Obtain a CoK prompt ˆ_I_(_n_)<br>_i_<br>. If_n_is1, ˆ_I_(_n_)<br>_i_<br>_←_[_E_; ˆ_Qi_].<br>Generate evidence triple ˆ_T_ (_n_)<br>_i_<br>, explanation hint ˆ_H_(_n_)<br>_i_<br>and<br>answer ˆ_A_(_n_)<br>_i_<br>from the LLM.<br>Calculate reliability score_C_(_n_)<br>_i_<br>in Eq. 1.<br>**if**_C_(_n_)<br>_i_<br>_≥θ_**then**<br>Obtain fnal answer ˆ_Ai ←_ˆ_A_(_n_)<br>_i_<br>.<br>Remove ˆ_Qi_ from_U_.<br>**continue**||
|11:<br>12:|**end if**<br>For the evidence triples that_fv_(ˆ_r_(_n_)<br>_ij |_ˆ_s_(_n_)<br>_ij ,_ˆ_o_(_n_)<br>_ij , K_)_< θ_,||
||inject the corresponding correct knowledge triples ˆ_T ′_<br>_i_|into|
||the prompt, i.e., ˆ_I_(_n_+1)<br>_i_<br>_←_[ˆ_I_(_n_)<br>_i_<br>; ˆ_T ′_<br>_i_].||
|13:|**end for**||
|14: <br>15: <br>16:|**end for**<br> **for**each query ˆ_Qi_ in_U_ **do**<br>Obtain the fnal answer ˆ_Ai ←_arg max ˆ<br>_A_(_n_)<br>_i_<br>_C_(_n_)<br>_i_<br>.||
|17: <br>18:|**end for**<br> **return**all the answers_{_ ˆ_Ai}M_<br>_i_=1.||



## **4 Experiments** 

## **4.1 Experimental Setups** 

**Tasks and Datasets.** During experiments, we choose five different kinds of tasks to evaluate the performance of our method. The datasets and corresponding implementation details are shown in the following. 1) **Commonsense & factual reasoning** . We select CommonsenseQA (CSQA) [Talmor et al., 2019], StrategyQA [Geva et al., 2021], OpenBookQA [Mihaylov et al., 2018] the AI2 Reasoning Challenge (ARC-c) [Clark et al., 2018], sports understanding from the BIG-Bench benchmark [bench collaboration., 2022], and BoolQ [Clark et al., 2019] for evaluating CoK on commonsense and factual reasoning. 3) **Arithmetic reasoning.** We use grade school math problems GSM8K [Cobbe et al., 2021], a challenging dataset over math word problems SVAMP [Patel et al., 2021], and two others AQuA [Ling et al., 2017], MultiArith [Roy and Roth, 2015] for math problem solving tasks. 4) **Symbolic reasoning.** Two symbolic reasoning tasks are evaluated in our experiments, specifically, Last Letter Concatenation and Coin Flip tasks [Wei et al., 2022]. 

**Implementation Details.** For the LLM, we employ the publicly accessible GPT-3 [Brown et al., 2020] model, namely, _text-davinci-002_ with 175B parameters unless otherwise stated. To make a fair comparison, we directly follow [Wei et al., 2022, Zhang et al., 2023, Wang et al., 2023b, Kojima et al., 2022] to choose the same labeled examples as the demonstrations. For the datasets from Commonsense reasoning and factual reasoning, the KBs we choose are a combination of Wiktionary[6] , ConceptNet [Speer et al., 2017], Wikidata5M [Wang et al., 2021], ATOMIC [Sap et al., 2019], GLUCOSE [Mostafazadeh et al., 2020], ASER [Zhang et al., 2020, 2022a], and 

> 5This is similar to correcting wrong reasoning paths, we ensure that the label is not leaked to the model. 

> 6 `https://en.wiktionary.org/wiki/Wiktionary:Main_Page` . 

6 

Table 1: Accuracy on commonsense, factual, and symbolic reasoning tasks. 

|Model|_Commonsense & Factual Reasoning_|_Symbolic Reasoning_<br> Letter<br>Coin|
|---|---|---|
||Common<br>Strategy<br>OpenBook ARC-c Sports<br>BoolQ <br>Sense QA<br>QA<br>QA||
|Fine-tuning|91.2<br>73.9<br>91.0<br>75.0<br>-<br>92.4|-<br>-|
|Zero-Shot SP<br>Zero-Shot CoT<br>Few-Shot SP<br>Manual CoT<br>Auto-CoT|68.8<br>12.7<br>44.7<br>46.8<br>38.1<br>50.2<br>64.6<br>54.8<br>68.4<br>64.7<br>77.5<br>52.7<br>**79.5**<br>65.9<br>**76.6**<br>68.2<br>69.6<br>53.6<br>73.5<br>65.4<br>73.0<br>69.9<br>82.4<br>55.0<br>74.4<br>65.4<br>-<br>-<br>-<br>-|0.2<br>12.8<br>57.6<br>91.4<br>0.0<br>49.1<br>59.0<br>74.5<br>59.7<br>**99.9**|
|CoK<br>CoK + F2-V|75.4<br>66.6<br>73.9<br>71.1<br>83.2<br>56.8<br>77.3<br>**67.9**<br>74.8<br>**73.0**<br>**84.1**<br>**59.9**|59.4<br>97.4<br>**61.1**<br>**99.9**|



CausalBank [Li et al., 2020]. For the Last Letter Connection task in Symbolic Reasoning, we manually construct a dictionary KB for each word in Wiktionary. For example, the triple of the word “system” is “(system, last letter, m)”. For the rest datasets (e.g., arithmetic reasoning and coin dataset), we do not perform F[2] -Verification and rethinking processing because we can not find any KBs for these tasks. More implementation details can be found in Appendix C. 

**Baselines.** In our experiments, we first consider **few-shot/zero-shot standard prompting (SP)** popularized by Brown et al. [2020] as the baselines. Then some SOTA CoT methods also serve as strong baselines. 1) **Chain-of-thought (CoT)** [Wei et al., 2022]: standard chain-of-thought prompting that provides four to eight hand-crafted examples, encompassing a sequence of intermediate reasoning steps. 2) **Zero-shot-CoT** [Kojima et al., 2022]: a zero-shot template-based approach for CoT, which diverges from the original method as it eliminates the need for step-by-step few-shot examples through adding task-agnostic prompts. 3) **Auto-CoT** [Zhang et al., 2023]: constructing exemplars for CoT automatically through selecting diverse questions to build illustrative demonstrations. 

## **4.2 Competitive Performance of CoK** 

## **Commonsense & Factual Reasoning.** 

Table 1 compares accuracy on six datasets from commonsense and factual reasoning respectively in the few-shot learning scenarios, which is challenging for LLMs. We make the following observations: 1) CoK prompting steadily exceeds the performance of the previous CoT strategies. Specifically, our method respectively achieves 1.9%, 2.5%, 1.2%, 0.9%, 0.8%, and 1.8% improvement on these tasks, demonstrating that the combination 

Table 2: Accuracy on arithmetic reasoning tasks. 

||Model|GSM8K|SVAMP|AQuA|MultiArith|
|---|---|---|---|---|---|
||Fine-tuning|55.0|57.4|37.9|-|
||Zero-Shot SP|10.4|58.8|22.4|17.7|
||Zero-Shot CoT|40.7|62.1|33.5|78.7|
||Few-Shot SP<br>Manual CoT<br>Auto-CoT<br>CoK|15.6<br>46.9<br>47.9<br>**48.9**|65.7<br>68.9<br>69.5<br>**69.9**|24.8<br>35.8<br>36.5<br>**37.8**|33.8<br>91.7<br>92.0<br>**94.6**|



of explicit evidence triples and explanation can boost the LLMs’ reasoning ability. This also suggests that a better elicitation format is critical for prompt-based learning. 2) Based on F[2] -Verification, the performances can be further enhanced. In particular, the performance of CoK + F[2] -Verification nearly approaches fine-tuning on the StrategyQA and ARC-c tasks. This indicates that conducting post-verification and correcting false evidence triples by injecting ground-truth knowledge is crucial to the reasoning. 3) For the CSQA and OpenBookQA tasks, few-shot SP occasionally obtains the best results, we think that the model may be stuck in a thinking dilemma on some samples. Despite this, our method can still outperform other strong baselines. 

**Symbolic & Arithmetic Reasoning.** We also explore how can CoK prompting adapt to nonknowledge-intensive tasks, such as symbolic and arithmetic reasoning, which are the most challenging tasks for LLMs. For symbolic reasoning, we choose the last letter connection and coin flip tasks. Results in Table 1 suggest that CoK prompting can also make high improvements on these tasks. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting_images/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting.pdf-0008-00.png)


**----- Start of picture text -----**<br>
100 CoK CoK+F [2] -V 90 = 0 = 40 = 80<br>90 CoK w/o. ET CoK+F [2] -V w/o. Factuality 80 = 20 = 60 = 100<br>CoK w/o. EH CoK+F [2] -V w/o. Faithfulness<br>80 70<br>70 60<br>60 50<br>50 40<br>CSQA ARC-c BoolQ Letter CSQA ARC-c Letter GSM8K SVAMP<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 3: Ablation study results: accuracy when Figure 4: Effect of wrong demonstrations with we remove different components. _β_ % wrong evidence triples. 

For arithmetic reasoning tasks, it is hard to find an off-the-shelf KB to estimate the evidence triples. Nonetheless, we still achieve the best accuracies in Table 2, indicating that decomposing the reasoning chains into explicit triples is helpful for LLMs to understand complex tasks. 

## **4.3 Discussions** 

**Ablation Study.** In this section, we aim to explore how much does each part of the component contributes to the performance. We perform an ablation study to see how the performance changes. Thus, we design six variants: 1) **CoK** is the basic prompt without the rethinking process; 2) **CoK w/o. ET** consists of only explanation hints (is the same as CoT); 3) **CoK w/o. EH** only has evidence triples; 4) **CoK + F[2] -V** denotes the whole framework; 5) **CoK + F[2] -V w/o. Factuality** aims to only use faithfulness score in the rethinking process; and 6) **CoK + F[2] -V w/o. Faithfulness** aims to only use a factuality score. We conduct the ablation study on four tasks, including CSQA, ARC-c, BoolQ, and the Last Letter Connection. 

Results in Figure 3 demonstrate that the performance drops when removing each component, which shows the significance of all components. For CoK, we can see that the performance of the variant CoK w/o. ET is lower than CoK w/o. EH on all the tasks, which suggests that urging the LLM to generate explicit evidence triples is the most important contribution to the performance. In addition, both the evidence triples and explanation hints can be fully utilized in the rethinking process because they can guide the LLM to verify and correct the reasoning chains via either factuality or faithfulness. 

**Effect of Wrong Demonstrations.** ReTable 3: Effect of wrong demonstrations with irrelevant call the discussion in Section 3.2 about demonstrations. the demonstrations that may have some mistakes. To see if chain-of-knowledge Model CSQA StrategyQA OpenBookQA ARC-c prompting has a similar phenomenon to CoK 75.4 66.6 59.4 48.9 previous works [Wang et al., 2023a] that CoK (GSM8K) 73.6 61.4 55.0 47.3 there is no strong connection between the validity of reasoning chains and the perfor- 

mance of model prediction. We design two settings: 1) _β_ % error rate of evidence triples: for each exemplar, we choose _β_ % evidence triples and replace them randomly from the KB. 2) Irrelevant demonstrations: for each exemplar, we replace the prompt with an alternative exemplar. Specifically, we choose CSQA, StrategyQA, OpenBookQA, and ARC-c tasks from commonsense reasoning, and for each task, we select the demonstrations from the GSM8K task, which is very different from them. 

For the first setting, we choose five tasks and draw some bar charts in Figure 4. Results illustrate that the accuracy rate drops slightly as the value of _β_ increases from 0 to 100. However, even if the evidence triplets are all wrong, the performance will not decrease significantly. This phenomenon is counter-intuitive, yet, in line with the expected situation we considered before. The results for the irrelevant demonstration settings are shown in Table 3. The performance is exciting because we find the LLM can easily know how to follow the chain-of-knowledge paradigm to solve a new problem, although the given prompt is completely irrelevant to the current task. 

**Effectiveness of Rethinking Process.** Recall the _rethinking process_ , when the reasoning chains generated by the LLMs fail to pass verifications and the reliability score is below the threshold _θ_ , we provide them with additional opportunities to regenerate in the rethinking stage. Figure 5 demonstrates the effectiveness over three tasks with different combinations of rethinking iteration number _N ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 5 _}_ and threshold _θ ∈{_ 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 6 _,_ 0 _._ 8 _,_ 1 _._ 0 _}_ . From the figure, we can 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting_images/acl_2024_Boosting_Language_Models_Reasoning_with_Chain-of-Knowledge_Prompting.pdf-0009-00.png)


**----- Start of picture text -----**<br>
CSQA BoolQ Letter<br>61<br>78<br>60 61<br>77<br>59<br>76 58 60<br>57<br>75<br>= 0 = 0.75 56 = 0 = 0.75 59 = 0 = 0.75<br>74 = 0.25 = 1.0 55 = 0.25 = 1.0 = 0.25 = 1.0<br>= 0.50 = 0.50 = 0.50<br>73 54 58<br>1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>Iteration Number (N) Iteration Number (N) Iteration Number (N)<br>Accuracy (%) Accuracy (%) Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 5: Effectiveness of different rethinking iteration _N_ and reliability threshold _θ_ . 

Table 4: Case study on CSQA and Last Letter Connection: the chain-of-knowledge prompting with the rethinking process. The tokens in red, blue, and green respectively denote the wrong rationales, the injected knowledge, and the corrected rationales. 

|**Query**|**Evidences Triples & Explanations**|**Injected Knowledge Prompt**|**Rethinking**|
|---|---|---|---|
|**[**_K_<br>**exemplars]**_\_n_\_n<br>What island country is<br>ferret popular?<br>Answer<br>Choices: (A) own home<br>(B) north carolina (C)<br>great britain (D) hutch (E)<br>outdoors|A: Evidence triples: 1. (ferret, isA, animal)_\_n<br>2.<br>(country, isA, place)_\_n 3.<br>(ferret, popu-<br>lar, Britain)_\_n 4.<br>(north carolina, located in,<br>Britain)_\_n Explanation hints: The answer must<br>be a place where ferrets are popular. Of the above<br>choices,only North Carolina is a place where fer-<br>rets are popular._\_nSo the answer is (D).|**[**_K_ **exemplars]**_\_n_\_n What is-<br>land country is ferret popu-<br>lar?<br>Answer Choices:<br>(A)<br>own home (B) north carolina<br>(C) great britain (D) hutch<br>(E) outdoors_\_n (North Car-<br>olina, located in, US.)_\_n(Great<br>Britain, located in, Britain)|A: Evidence triples: 1. (ferret, isA, animal)_\_n<br>2. (country, isA, place)_\_n 3. (ferret, popular,<br>Great Britain)_\_n 4. (Great Britain, located in,<br>Britain)_\_n Explanation hints: The answer must<br>be a place where ferrets are popular. Of the above<br>choices, only North Carolina is a place where<br>ferrets are popular._\_nSo the answer is (C).|
|**[**_K_<br>**exemplars]**_\_n_\_n<br>Take the last letters of<br>each words in “Prince<br>Rene Vishal Patrick” and<br>concatenate them.|A: Evidence triples:_\_n1. (Prince, last latter, e)<br>_\_n2. (Rene, last latter, e)_\_n3. (Vishal, last latter,<br>i)_\_n4. (Patrick, last latter, k)_\_n5. (fnal answer,<br>is, eelk)_\_n Explanation hints: The last letter of<br>“Prince” is “e”. The last letter of “Rene” is “e”.<br>The last letter of “Vishal” is “i”. The last letter of<br>“Patrick” is “k”. Concatenating them is “eeik”._\_n<br>So the answer is eeik.|**[**_K_ **exemplars]**_\_n_\_n Take the<br>last letters of each words in<br>“Prince Rene Vishal Patrick”<br>and<br>concatenate<br>them._\_n<br>(Vishal, last latter, l)|A: Evidence triples:_\_n1. (Prince, last latter, e)<br>_\_n2. (Rene, last latter, e)_\_n3. (Vishal, last latter,<br>l)_\_n4. (Patrick, last latter, k)_\_n5. (fnal answer,<br>is, eelk)_\_n Explanation hints: The last letter of<br>“Prince” is “e”. The last letter of “Rene” is “e”.<br>The last letter of “Vishal” is “l”. The last letter of<br>“Patrick” is “k”. Concatenating them is “eelk”._\_n<br>So the answer is eelk.|



draw some following suggestions. 1) In most cases, the accuracy increases when the LLM rethinks step-by-step with a threshold below one, 2) The rethinking process converges faster when using a smaller threshold. It is not difficult to understand that when the threshold is small or even zero, almost all testing queries will be injected with knowledge and regenerated, which is similar to the method of combining _self-consistency_ and F[2] -Verification. 

**Case Study.** We end this section with a case study to show the effectiveness of our proposed chain-of-knowledge prompting and the rethinking process with F[2] -Verification. We randomly choose two examples from CSQA and Last Letter Connection tasks, and the results are listed in Table 4. We can see that our proposed method can effectively generate explicit evidence triples with corresponding explanation hints, and the wrong triples can be detected through the proposed F[2] -Verification. During the rethinking process, the LLM can be guided with a new prompt with injected knowledge and then reach the correct answer. We also find the LLM can spontaneously check whether the injected knowledge is useful for the question solution (as shown in Appendix D.1), which indicates that our method can achieve meticulous injection and avoid injecting useless knowledge. 

## **5 Conclusion** 

We propose chain-of-knowledge prompting, which aims to decompose the reasoning chains derived from the LLMs into multiple evidence triples and explanation hints, to further improve the reasoning capabilities. Based on the chain-of-knowledge prompt, we introduce F[2] -Verification and fully exploit external knowledge bases to perform post-verification for the generated reasoning chains in terms of factuality and faithfulness. A rethinking process then be used to inject knowledge to correct the false evidence triples and elicit the LLM to regenerate the answer. Our extensive results show that it outperforms other prompt methods over multiple reasoning tasks. In the future, we will 1) further improve the performance of other scale LLMs, 2) extend the KB to search engines to realize real-time verification, and 3) perform interpretability analysis on LLMs’ reasoning. 

9 

## **References** 

- Akari Asai and Hannaneh Hajishirzi. Logic-guided data augmentation and regularization for consistent question answering. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020_ , pages 5642–5650. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.499. URL `https://doi.org/10.18653/v1/2020. acl-main.499` . 

- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022. 

- BIG bench collaboration. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2022. URL `https://github.com/google/BIG-bench/` . 

- Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , 2020. 

- Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks, 2022a. 

- Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. Meta-learning via language model in-context tuning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 719–730. Association for Computational Linguistics, 2022b. doi: 10.18653/v1/2022.acl-long.53. URL `https://doi.org/ 10.18653/v1/2022.acl-long.53` . 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URL `https://arxiv.org/abs/2204.02311` . 

- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 2924– 2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL `https://aclanthology.org/N19-1300` . 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _ArXiv_ , abs/1803.05457, 2018. 

10 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL_ , pages 4171–4186, 2019. 

- Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. A survey for in-context learning. _CoRR_ , abs/2301.00234, 2023. doi: 10.48550/arXiv.2301.00234. URL `https://doi.org/10.48550/arXiv.2301.00234` . 

- Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. _arXiv preprint arXiv:2211.10435_ , 2022. 

- Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , pages 6894–6910. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021. emnlp-main.552. URL `https://doi.org/10.18653/v1/2021.emnlp-main.552` . 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , 9:346–361, 2021. doi: 10.1162/tacl_a_00370. URL `https://aclanthology.org/2021.tacl-1.21` . 

- Usha Goswami. Inductive and deductive reasoning. _Blackwell handbook of childhood cognitive development_ , pages 282–302, 2002. 

- Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In _Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event_ , volume 119 of _Proceedings of Machine Learning Research_ , pages 3887–3896. PMLR, 2020. URL `http://proceedings.mlr.press/v119/guo20h.html` . 

- Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. _CoRR_ , abs/2210.11610, 2022. doi: 10.48550/arXiv.2210. 11610. URL `https://doi.org/10.48550/arXiv.2210.11610` . 

- Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020_ , pages 4198–4205. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.386. URL `https://doi.org/10.18653/ v1/2020.acl-main.386` . 

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Computing Surveys_ , 55(12):1–38, mar 2023. doi: 10.1145/3571730. URL `https://doi.org/ 10.1145%2F3571730` . 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id=e2TBb5y0yFf` . 

- Mojtaba Komeili, Kurt Shuster, and Jason Weston. Internet-augmented dialogue generation. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8460–8478, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.579. URL `https://aclanthology.org/2022. acl-long.579` . 

11 

- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 5315–5333, Toronto, Canada, July 2023. Association for Computational Linguistics. URL `https://aclanthology.org/2023.acl-long.291` . 

- Zhongyang Li, Xiao Ding, Ting Liu, J. Edward Hu, and Benjamin Van Durme. Guided generation of cause and effect. In Christian Bessiere, editor, _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20_ , pages 3629–3636. International Joint Conferences on Artificial Intelligence Organization, 7 2020. doi: 10.24963/ijcai.2020/502. URL `https: //doi.org/10.24963/ijcai.2020/502` . Main track. 

- Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. Learning entity and relation embeddings for knowledge graph completion. In Blai Bonet and Sven Koenig, editors, _Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA_ , pages 2181–2187. AAAI Press, 2015. URL `http://www.aaai.org/ocs/index.php/ AAAI/AAAI15/paper/view/9571` . 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Regina Barzilay and Min-Yen Kan, editors, _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers_ , pages 158–167. Association for Computational Linguistics, 2017. doi: 10.18653/v1/P17-1015. URL `https://doi.org/10.18653/v1/P17-1015` . 

- Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? In Eneko Agirre, Marianna Apidianaki, and Ivan Vulic, editors, _Proceedings of Deep Learning Inside Out: The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, DeeLIO@ACL 2022, Dublin, Ireland and Online, May 27, 2022_ , pages 100–114. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.deelio-1.10. URL `https://doi.org/10.18653/v1/2022.deelio-1.10` . 

- Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. _ACM Comput. Surv._ , 55(9):195:1–195:35, 2023a. doi: 10.1145/3560815. URL `https://doi.org/10.1145/3560815` . 

- Ruibo Liu, Jason Wei, Shixiang Shane Gu, Te-Yen Wu, Soroush Vosoughi, Claire Cui, Denny Zhou, and Andrew M. Dai. Mind’s eye: Grounded language model reasoning through simulation. In _The Eleventh International Conference on Learning Representations_ , 2023b. URL `https: //openreview.net/forum?id=4rXMRuoJlai` . 

- Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. _arXiv preprint arXiv:2304.09842_ , 2023. 

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 8086–8098. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.acl-long.556. URL `https://doi.org/10.18653/v1/2022.acl-long. 556` . 

- Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. _CoRR_ , abs/2301.13379, 2023. doi: 10.48550/arXiv.2301.13379. URL `https://doi.org/10.48550/arXiv.2301.13379` . 

- Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. URL `https://aclanthology.org/D18-1260` . 

12 

- Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz, editors, _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pages 2791–2809. Association for Computational Linguistics, 2022a. doi: 10.18653/ v1/2022.naacl-main.201. URL `https://doi.org/10.18653/v1/2022.naacl-main.201` . 

- Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 11048–11064. Association for Computational Linguistics, 2022b. URL `https://aclanthology.org/2022.emnlp-main.759` . 

- Nasrin Mostafazadeh, Aditya Kalyanpur, Lori Moon, David Buchanan, Lauren Berkowitz, Or Biran, and Jennifer Chu-Carroll. GLUCOSE: GeneraLized and COntextualized story explanations. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 4569–4586, Online, November 2020. Association for Computational Linguistics. doi: 10. 18653/v1/2020.emnlp-main.370. URL `https://aclanthology.org/2020.emnlp-main.370` . 

- Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. Webgpt: Browser-assisted question-answering with human feedback, 2022. 

- Jane Pan, Tianyu Gao, Howard Chen, and Danqi Chen. What in-context learning “learns” in-context: Disentangling task recognition and task learning. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 8298–8319, Toronto, Canada, July 2023. Association for Computational Linguistics. URL `https://aclanthology.org/2023.findings-acl.527` . 

- Xiaoman Pan, Wenlin Yao, Hongming Zhang, Dian Yu, Dong Yu, and Jianshu Chen. Knowledge-incontext: Towards knowledgeable semi-parametric language models. _CoRR_ , abs/2210.16433, 2022. doi: 10.48550/arXiv.2210.16433. URL `https://doi.org/10.48550/arXiv.2210.16433` . 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2080–2094, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main. 168. URL `https://aclanthology.org/2021.naacl-main.168` . 

- Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. How context affects language models’ factual predictions. In _Automated Knowledge Base Construction_ , 2020. URL `https://openreview.net/forum?id=025X0zPfn` . 

- Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher, 2021. URL `https://arxiv.org/abs/2112.11446` . 

13 

- Subhro Roy and Dan Roth. Solving general arithmetic word problems. In Lluís Màrquez, Chris Callison-Burch, Jian Su, Daniele Pighin, and Yuval Marton, editors, _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015_ , pages 1743–1752. The Association for Computational Linguistics, 2015. doi: 10.18653/v1/d15-1202. URL `https://doi.org/10.18653/v1/d15-1202` . 

- Maarten Sap, Ronan Le Bras, Emily Allaway, Chandra Bhagavatula, Nicholas Lourie, Hannah Rashkin, Brendan Roof, Noah A. Smith, and Yejin Choi. Atomic: An atlas of machine commonsense for if-then reasoning. _ArXiv_ , abs/1811.00146, 2019. 

- Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176bparameter open-access multilingual language model. _ArXiv preprint_ , abs/2211.05100, 2022. URL `https://arxiv.org/abs/2211.05100` . 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. 

- Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woo-Myoung Park, Jung-Woo Ha, and Nako Sung. On the effect of pretraining corpora on in-context learning by a large-scale language model. In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz, editors, _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022_ , pages 5168–5186. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.naacl-main. 380. URL `https://doi.org/10.18653/v1/2022.naacl-main.380` . 

- Chenglei Si, Zhe Gan, Zhengyuan Yang, Shuohang Wang, Jianfeng Wang, Jordan L. Boyd-Graber, and Lijuan Wang. Prompting GPT-3 to be reliable. _CoRR_ , abs/2210.09150, 2022. doi: 10.48550/ arXiv.2210.09150. URL `https://doi.org/10.48550/arXiv.2210.09150` . 

- Robyn Speer, Joshua Chin, and Catherine Havasi. Conceptnet 5.5: An open multilingual graph of general knowledge. In _Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence_ , AAAI’17, page 4444–4451. AAAI Press, 2017. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL `https: //aclanthology.org/N19-1421` . 

- Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. URL `https: //arxiv.org/abs/2201.08239` . 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971, 2023. doi: 10.48550/arXiv.2302.13971. URL `https://doi. org/10.48550/arXiv.2302.13971` . 

14 

- Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters. In _ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models_ , 2023a. URL `https://openreview.net/forum?id=L9UMeoeU2i` . 

- Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. Kepler: A unified model for knowledge embedding and pre-trained language representation. _Transactions of the Association for Computational Linguistics_ , 9:176–194, 2021. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, and Denny Zhou. Rationaleaugmented ensembles in language models. _CoRR_ , abs/2207.00747, 2022. doi: 10.48550/arXiv. 2207.00747. URL `https://doi.org/10.48550/arXiv.2207.00747` . 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023b. URL `https://openreview.net/forum?id=1PL1NIMMrw` . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id= _VjQlMeSB_J` . 

- Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. Self-adaptive in-context learning: An information compression perspective for in-context example selection and ordering. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1423–1436, Toronto, Canada, July 2023. Association for Computational Linguistics. URL `https://aclanthology.org/2023.acl-long.79` . 

- Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. Compositional exemplars for in-context learning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 39818–39833. PMLR, 23–29 Jul 2023. URL `https://proceedings.mlr.press/v202/ye23c.html` . 

- Xi Ye and Greg Durrett. The unreliability of explanations in few-shot prompting for textual reasoning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, _Advances in Neural Information Processing Systems_ , 2022. URL `https://openreview.net/forum?id= Bct2f8fRd8S` . 

- Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. Do large language models know what they don’t know? In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 8653–8665, Toronto, Canada, July 2023. Association for Computational Linguistics. URL `https://aclanthology.org/2023.findings-acl.551` . 

- Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sanggoo Lee, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label demonstrations. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 2422–2437. Association for Computational Linguistics, 2022. URL `https://aclanthology.org/2022.emnlp-main.155` . 

- Hongming Zhang, Xin Liu, Haojie Pan, Yangqiu Song, and Cane Wing-Ki Leung. ASER: A large-scale eventuality knowledge graph. In _WWW_ , pages 201–211, 2020. 

- Hongming Zhang, Xin Liu, Haojie Pan, Haowen Ke, Jiefu Ou, Tianqing Fang, and Yangqiu Song. ASER: towards large-scale commonsense knowledge acquisition via higher-order selectional preference over eventualities. _Artificial Intelligence_ , 309:103740, 2022a. 

- Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022b. 

15 

- Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview.net/forum?id=5NTt8GFjUHkr` . 

- Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In Marina Meila and Tong Zhang, editors, _Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event_ , volume 139 of _Proceedings of Machine Learning Research_ , pages 12697–12706. PMLR, 2021. URL `http://proceedings.mlr.press/v139/zhao21c.html` . 

- Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https://openreview.net/forum?id=WZH7099tgfM` . 

16 

## **A Limitations and Broader Impact** 

Our work is based on prompting methods for large language models and achieves outstanding performance across several benchmarks. However, it still carries the following limitations: (1) The evidence triples in knowledge bases are finite, which might not ensure comprehensive coverage of the model’s requirements for all questions. (2) In light of the integration of the re-thinking algorithm, CoK might require more API calls compared to vanilla CoT methods. 

In terms of social impact, the knowledge bases we utilize are all from publicly available data sources. Infusing factual knowledge into the model’s reasoning process will not introduce additional bias. Moreover, it can to some extent prevent the model from providing irresponsible and harmful answers. 

## **B Statistics of Datasets** 

We evaluate CoK on 12 publicly available benchmark datasets that cover arithmetic reasoning, commonsense reasoning, symbolic reasoning and natural language understanding tasks. The statistics of the datasets are shown in Table 5. 

Table 5: Dataset Descriptions. 

|Dataset|Number of samples|Average words|Answer Format|Licence|
|---|---|---|---|---|
|CSQA|1,221|27.8|Multi-choice|Unspecifed|
|StrategyQA|2,290|9.6|Yes or No|Apache-2.0|
|OpenBookQA|500|27.6|Multi-choice|Unspecifed|
|ARC-c|1,172|47.5|Multi-choice|CC BY SA-4.0|
|Sports|1,000|7.0|Yes or No|Apache-2.0|
|BoolQ|3,270|8.7|Yes or No|CC BY SA-3.0|
|Last Letters|500|15.0|String|Unspecifed|
|Coin Flip|500|37.0|Yes or No|Unspecifed|
|GSM8K|1,319|46.9|Number|MIT License|
|SVAMP|1,000|31.8|Number|MIT License|
|AQuA|254|51.9|Multi-choice|Apache-2.0|
|MultiArith|600|31.8|Number|CC BY SA-4.0|



The detailed information of each dataset is shown in the follow: 

## **Commonsense & Factual reasoning** 

- CSQA [Talmor et al., 2019]: it is a commonsense QA task and the answer format is multichoice. The homepage is `https://www.tau-nlp.org/commonsenseqa` , and `https: //github.com/jonathanherzig/commonsenseqa` . 

- StrategyQA [Geva et al., 2021]: it is a commonsense QA task with Yes or No answer format. We use the open-domain setting (question-only set) from bench collaboration. [2022]: `https://github.com/google/BIG-bench/tree/main/bigbench/ benchmark_tasks/strategyqa` . The original dataset is from `https://github. com/eladsegal/strategyqa` , MIT license: `https://github.com/eladsegal/ strategyqa/blob/main/LICENSE` . 

- OpenBookQA [Mihaylov et al., 2018]: it is a multi-choice QA task to evaluate commonsense knowledge. The original dataset is from `https://allenai.org/data/open-book-qa` . 

- ARC-c [Clark et al., 2018]: it is a multip-choice commonsense QA task. The original dataset is from `https://allenai.org/data/arc` . CC BY SA-4.0 license: `https:// creativecommons.org/licenses/by-sa/4.0/` . 

- Sports understanding from BIG-Bench [bench collaboration., 2022]: the answer format is Yes or No. Apache License v.2: `https://github.com/google/BIG-bench/blob/ main/LICENSE` . 

- BoolQ [Clark et al., 2019]: it is a knowledge-intensive task and the format is Yes or No. The original dataset is from `https://github.com/google-research-datasets/` 

17 

`boolean-questions` . CC BY SA-3.0 license: `https://creativecommons.org/ licenses/by-sa/3.0/` . 

## **Symbolic & Arithmetic reasoning** 

- Last Letters & Coin Flip [Wei et al., 2022] are novel benchmarks to evaluate whether the LLM can solve a simple symbolic reasoning problem. The last letters dataset is from `https: //huggingface.co/datasets/ChilleD/LastLetterConcat` . The coin flip dataset is from `https://huggingface.co/datasets/skrishna/coin_flip` . 

- GSM8K [Cobbe et al., 2021]: `https://github.com/openai/grade-school-math` , MIT license: `https://github.com/openai/grade-school-math/blob/master/ LICENSE` . 

- SVAMP [Patel et al., 2021]: `https://github.com/arkilpatel/SVAMP` , MIT license: `https://github.com/arkilpatel/SVAMP/blob/main/LICENSE` . 

- AQuA [Ling et al., 2017]: `https://github.com/deepmind/AQuA` , license: `https:// github.com/deepmind/AQuA/blob/master/LICENSE` . 

- Math Word Problem Repository MultiArith [Roy and Roth, 2015], license: CC BY 4.0, dataset: `https://huggingface.co/datasets/ChilleD/MultiArith` . 

## **C Implementation Details** 

## **C.1 CoK Construction** 

For each dataset, we aim to construct demonstrations with multiple well-designed exemplars. The prompt example of each dataset is shown in Appendix E. 

During the prompt construction, we first randomly select multiple labeled examples from the training set. For a fair comparison, we directly choose the selected labeled data from [Wei et al., 2022, Wang et al., 2023b, Kojima et al., 2022]. Specifically, we choose 8 labeled data for Coin Flip, ARC-c, AQuA, GSM8K, MUltiArith, CSQA, SVAMP, OpenBookQA; 4 labeled data for Last Letter Connection; 6 labeled data for Sports, BoolQ, StrategyQA. 

For each label data, we first use zero-shot CoT [Kojima et al., 2022] to perform textual reasoning chain generation. We directly connect a simple prompt “Let’s think step by step.” after the input query to elicit the LLM to generate rationale and the final answer[7] . We then remove this prompt and rebuild the input query by concatenating the input query and the generated textual reasoning chain. 

To construct evidence triples, we aim to retrieve some relevant knowledge triples from the pre-built knowledge base (as shown in Appendix C.2). During retrieval, given a textual reasoning chain (e.g., CoK-EH), we encode it with the basic sentence encoder model (e.g., BERT) and then retrieve the most relevant knowledge triple using the maximum inner product search tool SCaNN [Guo et al., 2020]. Due to the retrieved knowledge triples may consist of noises and redundant information. To improve the reliability of the evidence triples, we have invited five domain experts (including professors and Ph.D. students) to manually annotate the evidence triples based on the retrieved knowledge triples. 

To improve the annotation efficiency, we also use the ideal of self-training, which aims to generate annotated data based on very few data. Specifically, we can first manually annotate two labeled data with evidence triples and explanation hints to form a 2-shot CoK prompt. Then, each rest labeled data is concatenated with this 2-shot CoK prompt and the LLM can generate the corresponding rationale and answer. Thus, we can invite these experts to verify and correct them. 

Finally, we obtain five different annotated demonstrations. To select the best one for each dataset, we randomly choose a subset of the testing set to perform an evaluation. The prompt which has the best accuracy value is chosen. 

> 7To alleviate the generation bias problem, we also use the self-consistency decoder [Wang et al., 2023b] to sample one rationale. 

18 

Table 6: Performance of different knowledge injection. 

|Input|Response|
|---|---|
|Vanilla: _Knowledge Injection before Reasoning_||
|||
|Q: Who plays mary jane in<br>spiderman 3?<br>_\_n Knowl-<br>edge:<br>(Kirsten<br>Dunst,<br>(actors, jk simmons), (award<br>received, Milos Foreman)),<br>(Kirsten<br>Dunst,<br>actors,<br>Kirsten Dunst, occupation,<br>female model)|The answer to the question is Kirsten Dunst. To explain our reasoning<br>with the knowledge you provided:_\_n - Kirsten Dunst played Mary Jane<br>in Spiderman 3. To confrm this information, we can look at actors<br>who were in Spiderman 3, and we know that J.K. Simmons was also in<br>the flm. However, this information does not answer our question._\_n -<br>We can also look at awards received, and we know that Milos Forman<br>received awards for a flm he directed called "One Flew Over the<br>Cuckoo’s Nest" which Kirsten Dunst was not in._\_n -We can look at<br>Kirsten Dunst’s occupation and know she is a female model, which<br>does not help answer our question._\_n - Therefore, we can conclude<br>that Kirsten Dunst played Mary Jane in Spiderman 3.|



Ours: _Knowledge Injection after F[2] -Verification_ Q: Who plays mary jane in First, we need to recall which film we’re talking about, which is Spispiderman 3? _\_ n Knowl-derman 3. _\_ n Next, we need to think about the key players in the film, edge: (spiderman 3, stars, and who played Mary Jane in that particular movie. The actress who kirsten dunst) portrayed Mary Jane in Spiderman 3 is Kirsten Dunst. 

## **C.2 Knowledge Bases** 

We follow [Pan et al., 2022] to select six knowledge bases from different domains. 

**Dictionary** We utilize lexical knowledge, which comprises definitions and example sentences of English words (e.g., _< “apple”, definition, “A common, round fruit ...” >_ ) from Wiktionary[8] . Specifically, we use the Wiktionary dump dated April 30, 2022 that contains 1.3M word definitions and 470K example sentences for 1M words/phrases. 

**Commonsense** We incorporate commonsense knowledge from ConceptNet [Speer et al., 2017], which encompasses a wide range of general daily life knowledge, for instance, _< “rabbit”, CapableOf, “move fast” >_ . We follow previous works [Zhang et al., 2020] to include the core 600K high-quality triplets. 

**Causality** The external knowledge we utilize for causality is derived from the auto-extracted causal knowledge source, CausalBank [Li et al., 2020]. This database consists of large-scale English sentences that express cause-effect relationships, such as _< “babies cry”, because, “they are lonely” >_ ). 

**Entity** We integrate named entity knowledge in Wikidata5M [Wang et al., 2021]. For each entity, the properties are collected in the format of _< “Kepler space telescope”, Named after, “Johannes Kepler” >_ ). Here, related sentences refer to the sentences from an entity’s own article, or the sentences of other articles that link to this entity. 

**Event** We take into account knowledge of everyday events, utilizing both human-constructed event knowledge graphs such as ATOMIC [Sap et al., 2019] and GLUCOSE [Mostafazadeh et al., 2020], as well as automatically extracted event knowledge graphs like ASER [Zhang et al., 2020, 2022a]. 

**Scripts** Specifically, we use 325K triples that are in the form of < verbal information, context, nonverbal information >, where verbal information is an utterance, nonverbal information can be body movements, vocal tones, or facial expressions, etc., and context is the entire text of the scene from which the verbal-nonverbal pair is extracted. 

> 8 `https://en.wiktionary.org/wiki/Wiktionary:Main_Page` . 

19 

## **D Analysis** 

## **D.1 Why is verification useful?** 

We demonstrate some cases to show how the LLM verifies the usefulness of the injected knowledge. We use _gpt-3.5-turbo_ model because it can solve a problem through conversation. We define two following settings. 1) Vanilla: knowledge injection before reasoning. We directly concatenate the related knowledge triples with the input query, and prompt the LLM to think step by step. 2) Ours: knowledge injection after F[2] -Verification. We first use a CoK prompt to elicit the LLM to generate evidence triples and find error triples. Then, we correct them into corresponding ground truth triples and concatenate them with the input query. 

As shown in Table 6, we can see that the LLM can spontaneously detect and analyze each injected triple. For example, if the knowledge triple is useless or has no contribution to the reasoning, the LLM can talk to me about which and why are useless (the red text in Table 6). This indicates that the knowledge provided by the traditional knowledge injection method is not completely useful. In contrast, our approach can accurately locate the false reasoning evidence derived from the LLM after the verification stage, so as to perform targeted knowledge injection. 

## **E Exemplars with Chain-of-Knowledge Prompts** 

Table 7: Examples Chain-of-Knowledge prompting for the letter concatenation task. 

**Q:** Take the last letters of the words in "Elon Musk" and concatenate them. **A:** Evidence triples: 1. (Elon, last latter, n) 2. (Musk, last latter, k) 3. (final answer, is, nk) Explanation hints: The last letter of "Elon" is "n". The last letter of "Musk" is "k". Concatenating them is "nk". So the answer is nk. 

**Q:** Take the last letters of the words in "Larry Page" and concatenate them. **A:** Evidence triples: 1. (Larry, last latter, y) 2. (Page, last latter, e) 3. (final answer, is, ye) Explanation hints: The last letter of "Larry" is "y". The last letter of "Page" is "e". Concatenating them is "ye". So the answer is ye. 

**Q:** Take the last letters of the words in "Sergey Brin" and concatenate them. **A:** Evidence triples: 1. (Sergey, last latter, y) 2. (Brin, last latter, n) 3. (final answer, is, yn) Explanation hints: The last letter of "Sergey" is "y". The last letter of "Brin" is "n". Concatenating them is "yn". So the answer is yn. 

**Q:** Take the last letters of the words in "Bill Gates" and concatenate them. 

**A:** Evidence triples: 1. (Bill, last latter, l) 2. (Gates, last latter, s) 3. (final answer, is, ls) Explanation hints: The last letter of "Bill" is "l". The last letter of "Gates" is "s". Concatenating them is "ls". So the answer is ls. 

20 

Table 8: Examples Chain-of-Knowledge prompting for the coinflip task. 

**Q:** A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up? **A:** Evidence triples: 1. (coin, start with, head up) 2. (coin, flips, flipped) 3. (coin, not flips, flipped) Explanation hints:The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is yes. **Q:** A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up? **A:** Evidence triples: 1. (coin, start, head up) 2. (coin, flips, flipped) 3. (coin, flips, head up) Explanation hints: The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes. **Q:** A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up? **A:** Evidence triples: 1. (coin, start with, head up) 2. (coin, flips, flipped) 3. (coin, not flips, flipped) Explanation hints: The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no. **Q:** A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up? **A:** Evidence triples: 1. (coin, start with, head up) 2. (coin, flips, flipped) 3. (coin, not flips, flipped) Explanation hints: The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no. **Q:** A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up? **A:** Evidence triples: 1. (coin, start with, head up) 2. (coin, flips, flipped) 3. (coin, not flips, flipped) Explanation hints: The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no. **Q:** A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up? **A:** Evidence triples: 1. (coin, start with, head up) 2. (coin, flips, flipped) 3. (coin, not flips, flipped) Explanation hints: The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no. 

**Q:** A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up? 

**A:** Evidence triples: 

1. (coin, start with, head up) 2. (coin, not flips, head up) 3. (coin, not flips, head up) Explanation hints: The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up. So the answer is yes. 

21 

Table 9: Examples Chain-of-Knowledge prompting for the sports understanding task. 

**Q:** Is the following sentence plausible? ’Fernando Tatis Jr. walked on ball four.’ **A:** Evidence triples: 1. (Fernando Tatis Jr., isA, baseball player) 2. (walking on ball four, is part of, baseball) Explanation hints: A walk is when the batter reaches first base without hitting the ball. Reaching first base can happen a few ways, but one of them is by the pitcher throwing four balls. So the answer is yes. **Q:** Is the following sentence plausible? ’Jayson Tatum was caught offsides in the Champions Leage Semifinal.’ **A:** Evidence triples: 1. (Jayson Tatum, isA, basketball player) 2. (being caught offsides, is part of, soccer) Explanation hints: Being caught offsides is part of soccer, yet, Jayson Tatum is a basketball player. So the answer is no. **Q:** Is the following sentence plausible? ’Marouane Fellaini scored in extra time.’ **A:** Evidence triples: 1. (Marouane Fellaini, isA, soccer player) 2. (scoring in extra time, is part of, soccer) So, the answer is yes. **Q:** Is the following sentence plausible? ’Gerrit Cole committed a handball in the European Cup.’ **A:** Evidence triples: 1. (Gerrit Cole, isA, soccer player) 2. (committing a handball, is part of, soccer) Explanation hints: Committing a handball is part of soccer, yet, Gerrit Cole is a soccer player. So the answer is no. **Q:** Is the following sentence plausible? ’Klaas Jan Huntelaar scored the easy layup.’ **A:** Evidence triples: 1. (Klaas Jan Huntelaar, isA, soccer player) 2. (scoring a layup, is part of, basketball) Explanation hints: Scoring a layup is part of basketball, yet, Klaas Jan Huntelaar is a soccer player. So the answer is no. **Q:** Is the following sentence plausible? ’Mario Gomez earned a direct kick.’ **A:** Evidence triples: 1. (Mario, isA, soccer player) 2. (earning a direct kick, is part of, soccer) Explanation hints: Earning a direct kick is part of soccer, and Mario Gomez is a soccer player. So the answer is yes. 

22 

Table 10: Examples Chain-of-Knowledge prompting for the ARC-c task. 

**Q:** Putting a cardboard box in a bin instead of the trash can? Answer Choices: (A) conserve energy for later (B) save wild animal species (C) keep it from the trash (D) reduce the height of landfills **A:** Evidence triples: 1. (cardboard box, isA, item) 2. (bin, isA, container) 3. (cardboard box, usage, container) 4. (trash can, usage, container) 5. (bin, usage, container) Explanation hints: The answer must be an environmental reason. Of the above choices, only (D) are environmental reasons. So the answer is (D). **Q:** When the needle of a compass lines up with Earth ’s magnetic poles , the needle is? Answer Choices: (A) pointing east (B) pointing west (C) pointing at arctic (D) pointing to space **A:** Evidence triples: 1. (compass, is affected by, Earth’s magnetic field) 2. (magnetic field, go, from the south pole to the north pole) Explanation hints: The needle of the compass is affected by Earth’s magnetic field. The magnetic field lines go from the south pole to the north pole. So, when the needle of the compass is lined up with the Earth’s magnetic poles, the needle is pointing north. So the answer is (C). **Q:** Which uses gills to breathe? Answer Choices: (A) hermit crab (B) human (C) blue whale (D) bluebird **A:** Evidence triples: 1. (gills, isA, respiratory organ) 2. (respiratory organ, function, breathing) 3. (hermit crab, hasA, gills) 4. (blue whale, hasA, gills) 5. (bluebird, doesNotHave, gills) Explanation hints: The answer must be an item that has gills. Gills are respiratory organs that help with breathing. Of the above choices, only hermit crabs and blue whales have gills. So the answer is (A). **Q:** Which is the most accurate statement? Answer Choices: (A) finches with fat beaks can eat ants in a small anthill (B) finches with narrow beaks can eat only human meat (C) finches with narrow beaks can eat ants in a small anthill (D) finches with fat beaks can eat only human meat **A:** Evidence triples: 1. (finches, isA, bird) 2. (bird, hasA, beak) 3. (finches, hasA, fat beak) 4. (finches, hasA, narrow beak) 5. (finches, eats, ants) 6. (finches, eats, human meat) Explanation hints: The answer must be a statement about what finches can eat. Of the above choices, only finches with narrow beaks can eat ants in a small anthill. So the answer is (C). 

23 

Table 11: Examples Chain-of-Knowledge prompting for the ARC-c task. 

**Q:** What type of useful product can be made from the moving winds? Answer Choices: (A) metal (B) wood (C) bananas (D) electricity **A:** Evidence triples: 1. (wind, isA, moving air) 2. (electricity, isA, moving electrons) 3. (wind, can generate, electricity) Explanation hints: The answer must be a type of energy. Of the above choices, only electricity can be made from the moving winds. So the answer is (D). **Q:** Harriet wants to know the area of a rectangular sheet of paper. Its size is unknown to her. Which one of these will be the only tool that she needs? Answer Choices: (A) a ruler (B) a compass (C) a calculator (D) a protractor **A:** Evidence triples: 1. (area, isA, measurement) 2. (rectangular sheet of paper, isA, two-dimensional object) 3. (area, measurement of, two-dimensional object) 4. (ruler, usage, measuring) 5. (ruler, usage, measuring length) 6. (ruler, usage, measuring width) 7. (area, calculation of, length x width) Explanation hints: The answer must be an item that can help Harriet find the area of the rectangular sheet of paper. Of the above choices, only a ruler can help Harriet find the area of the rectangular sheet of paper. So the answer is (A). **Q:** What uses eyes to see by sensing light? Answer Choices: (A) trees (B) algae (C) worms (D) elephants **A:** Evidence triples: 1. (eyes, isA, organ) 2. (eyes, usage, seeing) 3. (eyes, sensing, light) 4. (trees, usage, photosynthesis) 5. (algae, usage, photosynthesis) 6. (worms, usage, burrowing) 7. (elephants, usage, trampling) Explanation hints: The answer must be an entity that uses eyes to see by sensing light. Of the above choices, only trees, algae, and worms use eyes to see by sensing light. So the answer is (D). **Q:** The cell of a plant when photosynthesizing may most require? Answer Choices: (A) access to roofless space (B) a complete vacuum in space (C) a room with blackened windows (D) dark basement living quarters **A:** Evidence triples: 1. (cell, isA, plant) 2. (photosynthesis, isA, process) 3. (cell, requires, photosynthesis) 4. (photosynthesis, requires, light) 5. (dark basement living quarters, blocks, light) Explanation hints: The cell of a plant when photosynthesizing may most require access to sunlight. Of the above choices, only (A) provides access to sunlight. So the answer is (A). 

24 

Table 12: Examples Chain-of-Knowledge prompting for the AQuA, GSM8K and MultiArith task. 

**Q:** There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? Answer Choices: (A) 12 (B) 6 (C) 8 (D) 14 **A:** Evidence triples: 1. (grove, originally have, 15 trees) 2. (grove, now have, 21 trees) 3. (21 - 15, equal, 6) Explanation hints: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is (B). **Q:** If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? Answer Choices: (A) 3 (B) 2 (C) 5 (D) 1 A: Evidence triples: 1. (parking lot, originally have, 3 cars) 2. (parking lot, add, 2 cars) 3. (parking lot, now have, 3 + 2 = 5 cars) Explanation hints: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is (C). **Q:** Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? Answer Choices: (A) 39 (B) 74 (C) 109 (D) 40 **A:** Evidence triples: 1. (Leah, have, 32 chocolates) 2. (her sister, have, 42 chocolates) 3. (they, have, 32 + 42 = 74 chocolates) 4. (they, eat, 35 chocolates) 5. (thay, now have, 74 - 35 = 39 chocolates) Explanation hints: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is (A). **Q:** Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? Answer Choices: (A) 32 (B) 16 (C) 9 (D) 8 **A:** Evidence triples: 1. (Jason, originally have, 20 lollipops) 2. (Jason, now have, 12 lollipops) 3. (Jason, give, 20 - 12 = 8 lollipops) Explanation hints: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is (D). 

25 

Table 13: Examples Chain-of-Knowledge prompting for the AQuA, GSM8K and MultiArith task. 

**Q:** Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? Answer Choices: (A) 5 (B) 20 (C) 9 (D) 1 **A:** Evidence triples: 1. (Shawn, have, 5 toys) 2. (his mon, give him, 2 toys) 3. (his dad, give him, 2 toys) 4. (Shawn, now have, 5 + 2 + 2 = 9 toys) Explanation hints: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is (C). **Q:** There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? Answer Choices: (A) 20 (B) 29 (C) 11 (D) 18 **A:** Evidence triples: 1. (server room, originally have, 9 computers) 2. (each day, installed, 5 computers) 3. (each from monday to thursday, have, 4 days) 4. (5 * 4, equal, 20) 5. (server room, now have, 9 + 20 = 29) Explanation hints: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is (B). **Q:** Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday? Answer Choices: (A) 33 (B) 35 (C) 81 (D) 83 **A:** Evidence triples: 1. (Michael, have, 58 golf balls) 2. (Michael, lost, 23 golf balls) 3. (Michael, lost, 2 golf balls) 4. (Michael, now have, 58 - 23 - 2 = 33 golf balls) Explanation hints: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is (A). **Q:** Olivia has $23. She bought five bagels for $3 each. How much money does she have left? Answer Choices: (A) 37 (B) 8 (C) 15 (D) 10 **A:** Evidence triples: 1. (Olivia, have, 23 dollars) 2. (Olivia, buy, 5 * 3 = 15 dollars) 3. (Olivia, now have, 23 - 15 = 8 dollars) Explanation hints: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is (B). 

26 

Table 14: Examples Chain-of-Knowledge prompting for the BoolQ task. 

**Q:** Do hamsters provide food for any animals? **A:** Evidence triples: 1. (Hamsters, isA, prey animals) 2. (Prey, is food for, predators) 3. (hamsters, provide food, animals) Explanation hints: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes. **Q:** Could Brooke Shields succeed at University of Pennsylvania? **A:** Evidence triples: 1. (Brooke Shields, isA, student) 2. (student, could succeed, at University of Pennsylvania) 3. (Brooke Shields, could succeed, at University of Pennsylvania) Explanation hints: Brooke Shields is a student. Students could succeed at University of Pennsylvania. Thus, Brooke Shields could succeed at University of Pennsylvania. So the answer is yes. **Q:** Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? **A:** Evidence triples: 1. (Hydrogen, has atomic number, 1) 2. (1, squared is, 1) 3. (1, exceeds, 5) 4. (5, is the number of, Spice Girls) Explanation hints: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no. **Q:** Yes or no: Is it common to see frost during some college commencements? **A:** Evidence triples: 1. (Frost, isA, weather condition) 2. (Weather condition, is, common during some college commencements) 3. (Frost, is common, during some college commencements) Explanation hints: Frost is a weather condition. Weather conditions are common during some college commencements. Thus, frost is common during some college commencements. So the answer is yes. 

**Q:** Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? 

**A:** Evidence triples: 1. (Llama, isA, mammal) 2. (Mammal, gives birth, live young) 3. (Llama, could give birth, during War in Vietnam) Explanation hints: Llamas are mammals. Mammals give birth to live young. Therefore, it is possible for a llama to give birth during the War in Vietnam. So the answer is no. **Q:** Yes or no: Would a pear sink in water? **A:** Evidence triples: 1. (pear, density, 0.6g/cm3) 2. (water, density, 1.0g/cm3) 3. (1.0g/cm3, is larger than, 0.6g/cm3) 4. (pear, can not sink in, water) Explanation hints: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no. 

27 

Table 15: Examples Chain-of-Knowledge prompting for the CSQA task. 

**Q:** What do people use to absorb extra ink from a fountain pen? Answer Choices: (A) shirt pocket (B) calligrapher’s hand (C) inkwell (D) desk drawer (E) blotter **A:** Evidence triples: 1. (fountain pen, isA, item) 2. (ink, isA, liquid) 3. (fountain pen, carrier of, liquid) 4. (fountain pen, usage, writing) 5. (blotters, usage, writing) 6. (blotters, absorb, liquid) Explanation hints: The answer must be an item that can absorb ink. A fountain pen which is full of liquid can writing on a blotter. Of the above choices, only blotters are used to absorb ink. So the answer is (E). **Q:** What home entertainment equipment requires cable? Answer Choices: (A) radio shack (B) substation (C) television (D) cabinet **A:** Evidence triples: 1. (home entertainment equipment, isA, item) 2. (cable, isA, item) 3. (home entertainment equipment, requires, cable) 4. (television, isA, home entertainment equipment) 5. (television, requires, cable) Explanation hints: The answer must be an item of home entertainment equipment that requires cable. Of the above choices, only television requires cable. So the answer is (C). **Q:** The fox walked from the city into the forest, what was it looking for? Answer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook **A:** Evidence triples: 1. (fox, isA, animal) 2. (forest, place of residence, animal) 3. (natural habitat, located in, forest) Explanation hints: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest which is the living place for fox animal. So the answer is (C). **Q:** Sammy wanted to go to where the people were. Where might he go? Answer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock **A:** Evidence triples: 1. (Sammy, isA, person) 2. (populated areas, place of residence, people) Explanation hints: Sammy is a person, so that the answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (A). **Q:** Where do you put your grapes just before checking out? Answer Choices: (A) mouth (B) grocery cart (C) super market (D) fruit basket (E) fruit market **A:** Evidence triples: 1. (checking out, isA, action) 2. (checking out, place of take place, mall) 3. (grapes, isA, merchandise) 4. (mall, sell, fruit) 5. (grocery cart, usage, hold merchandise before checking out) Explanation hints: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (B). **Q:** Google Maps and other highway and street GPS services have replaced what? Answer Choices: (A) united states (B) mexico (C) countryside (D) atlas **A:** Evidence triples: 1. (Google Maps, isA, webapp) 2. (GPS, isA, navigation systems) 3. (atlas, usage, navigation) Explanation hints: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (D). 

28 

Table 16: Examples Chain-of-Knowledge prompting for the SVAMP task. 

**Q:** There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? **A:** Explanation hints: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. Evidence triples: 1. (grove, originally have, 15 trees) 2. (grove, now have, 21 trees) 3. (21 - 15, equal, 6) So the answer is 6. **Q:** If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? **A:** Explanation hints: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. Evidence triples: 1. (parking lot, originally have, 3 cars) 2. (parking lot, add, 2 cars) 3. (parking lot, now have, 3 + 2 = 5 cars) So the answer is 5. **Q:** Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? **A:** Explanation hints: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. Evidence triples: 1. (Leah, have, 32 chocolates) 2. (her sister, have, 42 chocolates) 3. (they, have, 32 + 42 = 74 chocolates) 4. (they, eat, 35 chocolates) 5. (thay, now have, 74 - 35 = 39 chocolates) So the answer is 39. **Q:** Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? **A:** Explanation hints: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. v Evidence triples: 1. (Jason, originally have, 20 lollipops) 2. (Jason, now have, 12 lollipops) 3. (Jason, give, 20 - 12 = 8 lollipops) So the answer is 8. **Q:** Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? **A:** Explanation hints: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. Evidence triples: 1. (Shawn, have, 5 toys) 2. (his mon, give him, 2 toys) 3. (his dad, give him, 2 toys) 4. (Shawn, now have, 5 + 2 + 2 = 9 toys) So the answer is 9. **Q:** There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room? **A:** Explanation hints: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. Evidence triples: 1. (server room, originally have, 9 computers) 2. (each day, installed, 5 computers) 3. (each from monday to thursday, have, 4 days) 4. (5 * 4, equal, 20) 5. (server room, now have, 9 + 20 = 29) So the answer is 29. 

29 

Table 17: Examples Chain-of-Knowledge prompting for the OpenbookQA task. 

**Q:** Putting a cardboard box in a bin instead of the trash can? Answer Choices: (A) conserve energy for later (B) save wild animal species (C) keep it from the trash (D) reduce the height of landfills **A:** Evidence triples: 1. (cardboard box, isA, item) 2. (bin, isA, container) 3. (cardboard box, usage, container) 4. (trash can, usage, container) 5. (bin, usage, container) Explanation hints: The answer must be an environmental reason. Of the above choices, only (D) are environmental reasons. So the answer is (D). **Q:** When the needle of a compass lines up with Earth ’s magnetic poles , the needle is? Answer Choices: (A) pointing east (B) pointing west (C) pointing at arctic (D) pointing to space **A:** Evidence triples: 1. (compass, is affected by, Earth’s magnetic field) 2. (magnetic field, go, from the south pole to the north pole) Explanation hints: The needle of the compass is affected by Earth’s magnetic field. The magnetic field lines go from the south pole to the north pole. So, when the needle of the compass is lined up with the Earth’s magnetic poles, the needle is pointing north. So the answer is (C). **Q:** Which uses gills to breathe? Answer Choices: (A) hermit crab (B) human (C) blue whale (D) bluebird **A:** Evidence triples: 1. (gills, isA, respiratory organ) 2. (respiratory organ, function, breathing) 3. (hermit crab, hasA, gills) 4. (blue whale, hasA, gills) 5. (bluebird, doesNotHave, gills) Explanation hints: The answer must be an item that has gills. Gills are respiratory organs that help with breathing. Of the above choices, only hermit crabs and blue whales have gills. So the answer is (A). **Q:** Which is the most accurate statement? Answer Choices: (A) finches with fat beaks can eat ants in a small anthill (B) finches with narrow beaks can eat only human meat (C) finches with narrow beaks can eat ants in a small anthill (D) finches with fat beaks can eat only human meat A: Evidence triples: 1. (finches, isA, bird) 2. (bird, hasA, beak) 3. (finches, hasA, fat beak) 4. (finches, hasA, narrow beak) 5. (finches, eats, ants) 6. (finches, eats, human meat) Explanation hints: The answer must be a statement about what finches can eat. Of the above choices, only finches with narrow beaks can eat ants in a small anthill. So the answer is (C). **Q:** What type of useful product can be made from the moving winds? Answer Choices: (A) metal (B) wood (C) bananas (D) electricity **A:** Evidence triples: 1. (wind, isA, moving air) 2. (electricity, isA, moving electrons) 3. (wind, can generate, electricity) Explanation hints: The answer must be a type of energy. Of the above choices, only electricity can be made from the moving winds. So the answer is (D). 

30 

Table 18: Examples Chain-of-Knowledge prompting for the OpenbookQA task. 

**Q:** Harriet wants to know the area of a rectangular sheet of paper. Its size is unknown to her. Which one of these will be the only tool that she needs? Answer Choices: (A) a ruler (B) a compass (C) a calculator (D) a protractor **A:** Evidence triples: 1. (area, isA, measurement) 2. (rectangular sheet of paper, isA, two-dimensional object) 3. (area, measurement of, two-dimensional object) 4. (ruler, usage, measuring) 5. (ruler, usage, measuring length) 6. (ruler, usage, measuring width) 7. (area, calculation of, length x width) Explanation hints: The answer must be an item that can help Harriet find the area of the rectangular sheet of paper. Of the above choices, only a ruler can help Harriet find the area of the rectangular sheet of paper. So the answer is (A). **Q:** What uses eyes to see by sensing light? Answer Choices: (A) trees (B) algae (C) worms (D) elephants **A:** Evidence triples: 1. (eyes, isA, organ) 2. (eyes, usage, seeing) 3. (eyes, sensing, light) 4. (trees, usage, photosynthesis) 5. (algae, usage, photosynthesis) 6. (worms, usage, burrowing) 7. (elephants, usage, trampling) Explanation hints: The answer must be an entity that uses eyes to see by sensing light. Of the above choices, only trees, algae, and worms use eyes to see by sensing light. So the answer is (D). **Q:** The cell of a plant when photosynthesizing may most require? Answer Choices: (A) access to roofless space (B) a complete vacuum in space (C) a room with blackened windows (D) dark basement living quarters **A:** Evidence triples: 1. (cell, isA, plant) 2. (photosynthesis, isA, process) 3. (cell, requires, photosynthesis) 4. (photosynthesis, requires, light) 5. (dark basement living quarters, blocks, light) Explanation hints: The cell of a plant when photosynthesizing may most require access to sunlight. Of the above choices, only (A) provides access to sunlight. So the answer is (A). 

31 

Table 19: Examples Chain-of-Knowledge prompting for the StrategyQA task. 

**Q:** Do hamsters provide food for any animals? **A:** Evidence triples: 1. (Hamsters, isA, prey animals) 2. (Prey, is food for, predators) 3. (hamsters, provide food, animals) Explanation hints: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes. **Q:** Could Brooke Shields succeed at University of Pennsylvania? **A:** Evidence triples: 1. (Brooke Shields, isA, student) 2. (student, could succeed, at University of Pennsylvania) 3. (Brooke Shields, could succeed, at University of Pennsylvania) Explanation hints: Brooke Shields is a student. Students could succeed at University of Pennsylvania. Thus, Brooke Shields could succeed at University of Pennsylvania. So the answer is yes. **Q:** Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? **A:** Evidence triples: 1. (Hydrogen, has atomic number, 1) 2. (1, squared is, 1) 3. (1, exceeds, 5) 4. (5, is the number of, Spice Girls) Explanation hints: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no. **Q:** Yes or no: Is it common to see frost during some college commencements? **A:** Evidence triples: 1. (Frost, isA, weather condition) 2. (Weather condition, is, common during some college commencements) 3. (Frost, is common, during some college commencements) Explanation hints: Frost is a weather condition. Weather conditions are common during some college commencements. Thus, frost is common during some college commencements. So the answer is yes. **Q:** Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? **A:** Evidence triples: 1. (Llama, isA, mammal) 2. (Mammal, gives birth, live young) 3. (Llama, could give birth, during War in Vietnam) Explanation hints: Llamas are mammals. Mammals give birth to live young. Therefore, it is possible for a llama to give birth during the War in Vietnam. So the answer is no. **Q:** Yes or no: Would a pear sink in water? **A:** Evidence triples: 1. (pear, density, 0.6g/cm3) 2. (water, density, 1.0g/cm3) 3. (1.0g/cm3, is larger than, 0.6g/cm3) 4. (pear, can not sink in, water) Explanation hints: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no. 

32 

