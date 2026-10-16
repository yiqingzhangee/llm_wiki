# **Focus on Your Question! Interpreting and Mitigating Toxic CoT Problems in Commonsense Reasoning** 

**Jiachun Li[1,2] , Pengfei Cao[1,2] , Chenhao Wang[1,2] , Zhuoran Jin[1,2] , Yubo Chen[1,2,]**[*] **Daojian Zeng[3] , Kang Liu[1,2] , Jun Zhao[1,2,]**[*] 

1School of Artificial Intelligence, University of Chinese Academy of Sciences 

2The Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences 

3Hunan Normal University 

{jiachun.li, pengfei.cao, chenhao.wang, zhuoran.jin, yubo.chen, kliu, jzhao} @nlpr.ia.ac.cn zengdj916@163.com 

## **Abstract** 

Large language models exhibit high-level commonsense reasoning abilities, especially with enhancement methods like Chain-of-Thought (CoT). However, we find these CoT-like methods lead to a considerable number of originally correct answers turning wrong, which we define as the Toxic CoT problem. To interpret and mitigate this problem, we first utilize attribution tracing and causal tracing methods to probe the internal working mechanism of the LLM during CoT reasoning. Through comparisons, we prove that the model exhibits information loss from the question in the shallow attention layers when generating rationales or answers. Based on the probing results, we design a novel method called RIDERS ( **R** esidual decod **I** ng and s **ER** ial-position **S** wap), which compensates for the information deficit in the model from both decoding and serial-position perspectives. Through extensive experiments on multiple commonsense reasoning benchmarks, we validate that this method not only significantly eliminates Toxic CoT problems (decreased by **23.6%** ), but also effectively improves the model’s overall commonsense reasoning performance (increased by **5.5%** ). 

## **1 Introduction** 

With the increase in scale, large language models (LLMs) have demonstrated outstanding performance in different tasks (Li et al., 2023; Wen et al., 2023; Sun et al., 2024; Jin et al., 2024a), among them, commonsense reasoning has received significant attention due to its importance for general intelligence (Wang et al., 2022, 2024; Liu et al., 2024). In this task, researchers have proposed a series of chain-of-thought (CoT) like techniques to elicit models’ potential abilities (e.g. SelfConsistency (Wang et al., 2023c), Least-to-Most (Zhou et al., 2023), Reflexion (Shinn et al., 2023)). 

> *Corresponding authors. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0001-12.png)


**----- Start of picture text -----**<br>
Question Rationale Direct Answer<br>What kind of<br>The answer is: (3)<br>status is the bald  The bald eagle is<br>eagle given? (1)  a symbol of<br>outside (2) world  America. It is the<br>(3) protection  (4)  national bird of  CoT Answer<br>colorado (5)  the United States.<br>america The answer is: (5)<br>(a) Rationale Drift<br>Question Rationale Direct Answer<br>Metal is used to  Metal can be used  The answer is: (3)<br>to make various<br>make what? (1)<br>things, but the<br>junkyard (2) ore  most common use<br>(3) instruments  (4)<br>of metal is to  CoT Answer<br>metal fabrication<br>make tools and<br>shop (5) bowls machines.  The answer is: (4)<br>(b) Answer Drift<br>**----- End of picture text -----**<br>


Figure 1: Two examples for the Toxic CoT problem. 

Through them, LLMs can generate reasonable rationales and improve their reasoning performance. 

While these works have made great progress, we notice an overlooked problem in them, which we define as **Toxic CoT** — Sometimes LLMs can directly provide correct answers to questions, but after applying CoT-like methods, it brings extra reasoning paths to models, causing their answers to be wrong.[1] Figure 1 illustrates two main error types of this problem — Rationale Drift and Answer Drift. Specifically, for the Rationale Drift case, given the question “ _What kind of status is the bald eagle given?_ ”, the model can directly give the correct answer “ _protection_ ”. However, in the rationale, the model explains “ _what is the bald eagle_ ” as “ _a symbol of America_ ”, which has a semantic drift from the question. Thus, the model chooses the wrong option “ _america_ ” based on the drifting rationale. For the Answer Drift case, given the question 

1Notably, our definition here differs from the toxicity in generated content (Shaikh et al., 2023), emphasizing the harm that CoT can bring to the model. 

“ _Metal is used to make what?_ ”, the model can directly answer “ _instruments_ ”. It can also generate a correct rationale “ _metal is to make tools and machines_ ”, but when answering based on the rationale, the model drifts from it and selects the incorrect option “ _(4)_ ”. We further conduct a statistical analysis over extensive commonsense reasoning datasets and find that, among all CoT errors, this problem accounts for **37%** for the white-box model and **33%** for the black-box model on average, indicating this problem has become a crucial bottleneck in CoT reasoning.[2] 

So what is the mechanism behind this issue? In this paper, we attempt to answer this question by probing the inner workings of the LLM’s CoT reasoning. Specifically, we first make initial observations on examples of Rationale Drift and Answer Drift issues, which suggest that the model likely misses some important information from the question when generating corresponding rationales or answers. To further verify these findings, we use attribution tracing and causal tracing methods to probe the LLM in two stages (rationale generation stage and answer generation stage). By employing these methods under various experimental settings, **we find that there is a significant loss of information flow from the question in the shallow attention layers when generating drifting rationales and answers.** Therefore, we interpret the Toxic CoT problem as the model lacking information from the question in the two stages. 

To validate our interpretation and mitigate this problem, we design an approach called RIDERS ( **R** esidual decod **I** ng and s **ER** ial-position **S** wap) based on the interpretation. Concretely, for the Rationale Drift issue, we devise a decoding algorithm, promoting the model to generate tokens that pay more attention to question contexts. For the Answer Drift issue, we swap the positions of the output sequence, reducing the information loss from the question to the final prediction. We evaluate our method on five commonsense reasoning benchmarks and conduct extensive experiments. The results not only prove our interpretation, but also indicate that our method is effective in addressing the Toxic CoT problem and improving the model’s overall commonsense reasoning abilities. 

We summarize the contribution of this paper as follows: 

> 2Appendix A presents the details of settings and results in this statistical experiment. 

(1) We identify a crucial bottleneck affecting LLM’s reasoning performance called the Toxic CoT problem, probe this issue through attribution tracing and causal tracing methods, and interpret the mechanism behind it as the model missing information from questions in shallow attention layers. The results contribute to a more in-depth understanding of the LLM’s reasoning mechanisms. 

(2) To mitigate the Toxic CoT problem, we introduce RIDERS, which effectively compensates for the internal information loss during CoT reasoning from decoding and serial-position perspectives. 

(3) We conduct extensive experiments on various benchmarks. The results not only verify the rationality of our interpretation, but also demonstrate the effectiveness of our method in addressing the Toxic CoT problem (the proportion of the problem decreased by **23.6%** ) and enhancing commonsense reasoning performance (overall accuracy increased by **5.5%** ). Our code is available at: https://github.com/BugMakerzzz/toxic_cot. 

## **2 Problem Statement** 

## **2.1 Toxic Chain of Thought Reasoning** 

We start our work by formally defining the Toxic CoT problem as follows:[3] 

**Definition 2.1** ( **Toxic CoT** ) **.** _Given a question q and the correct answer o[∗] , if the model’s output M meets the following conditions, it is considered a case of Toxic CoT:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0002-12.png)


_where o[∗]_ = _M_ ( _q, Pd_ ) _indicates the model’s direct answering for q is correct, o[∗]_ = _M_ ( _q, Pc_ ) _indicates the model’s cot-like answering for q is wrong, Pd, Pc are the corresponding prompts._ 

## **2.2 Two-stage Drift Issues** 

To investigate the reasons for the problem, we classify these Toxic CoT cases and identify a main error causing this problem (On average, it accounts for 67% on two datasets, see more details in Appendix B). Furthermore, if we divide the CoT process into two stages: **rationale generation** and **answer generation** , there exist two types of issues in this error: 

**Definition 2.2** ( **Rationale Drift** ) **.** _If the reasoning chain is factually correct but logically inconsistent_ 

3In practice, CoT-type methods all have Toxic CoT problems, but to simplify the work, this paper mainly focuses on the basic CoT prompting. 

_with the question, this case is called “Rationale Drift” (see Figure 1a)._ 

**Definition 2.3** ( **Answer Drift** ) **.** _If the reasoning chain is both factually correct and logically consistent with the question, but the final answer is inconsistent with the rationale, this case is called “Answer Drift” (see Figure 1b)._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0003-02.png)


**----- Start of picture text -----**<br>
0.14<br>0.12 Drift Drift<br>0.10 Correct 0.12 Correct<br>0.10<br>0.08<br>0.08<br>0.06 0.06<br>0.04 0.04<br>0.02 0.02<br>0.00 0.00<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Llama2-13B (b) Baichuan2-13B<br>Attr Attr<br>**----- End of picture text -----**<br>


Figure 2: Attribution tracing results on Winogrande. 

## **2.3 Hypothesis Formulation** 

To provide a direction for subsequent probing experiments, here we attempt to propose hypotheses for the mechanism of issues by analyzing some examples. For the Rationale Drift issue, the model tends to focus on part of the essential reasoning conditions in the question context. As an example, in Figure 1a, the CoT only focuses on the “ _bald eagle_ ” in the question but misses another key information “ _status_ ”. As for the Answer Drift issue, the model seems to be disrupted by CoT, losing attention to the question and resulting in an off-topic prediction. For instance, in Figure 1b, though the CoT gives correct information “ _to make tools and machines_ ”, the model can only predict the wrong answer “ _(4) metal fabrication shop_ ”. This is likely because the model loses the question’s target “ _to make what_ ” and directly copies the entity “ _metal_ ”, which frequently appears in CoT, as the answer. Therefore, we summarize our hypotheses as follows: 

**Hypothesis 1.** _The Rationale Drift issue arises from the model lacking information from the question context in the rationale generation stage._ 

**Hypothesis 2.** _The Answer Drift issue arises from the model lacking information from the question in the answer generation stage._ 

To validate the above hypotheses, in the following two sections, we conduct probing experiments, exploring the LLM’s internal working mechanisms during the two stages of CoT reasoning. 

## **3 Tracing Information Flow in Rationale** 

In this section, we aim to verify the Hypothesis 1 by tracing the information flow in the rationale generation stage. To this end, we start by describing our attribution tracing method ( _§_ 3.1). Through this method, we conduct comparative experiments between the correct reasoning and the drifting one, figuring out the mechanism behind the issue ( _§_ 3.2). At last, we use the attention score to validate our findings from another perspective ( _§_ 3.3). 

## **3.1 Tracing Method** 

To investigate the roles of different model components in the rationale generation stage, we use attribution scores (Hao et al., 2021; Dai et al., 2022; Wang et al., 2023b) to compute the contribution of a neuron _ω_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0003-13.png)


where _F_ ( _·_ ) represents the model’s output. We compute the attribution score via Riemman approximation of the integration and _m_ is the number of approximation steps. For neurons _A_[(] _[l]_[)] in the i-th attention layer, we sum the absolute values of scores on all attention heads to get the final attribution score. Since the attention module involves interactions between different tokens, we can compute the information flow between the question context _q_ and the CoT _c_ on it: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0003-15.png)


Here, _Attr_ ( _A_[(] _i,j[l]_[)][)][ represents the intensity of infor-] mation flow from the _i_ -th token to the _j_ -th token in the _l_ -th attention layer and _|N |_ denotes the number of CoT steps. More implementation details of this method are reported in Appendix C. 

## **3.2 Attribution Tracing Experiment** 

**Experimental Settings** To validate the deficiency in Hypothesis 1, we need to figure out the context information flow difference between generating a drifting rationale and a correct one. Thus, we first use golden labels as hints to generate correct CoTs in the drifting cases. Then, we compute the average information flow under these two cases and compare their results. We choose Llama2-13B (Touvron et al., 2023) and Baichuan2-13B (Yang et al., 2023) as our probing models, since they are moderated-sized white-box models with decent 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
0.07<br>0.06 Drift 0.05 Drift<br>Normal Normal<br>0.05 0.04<br>0.04 0.03<br>0.03<br>0.02<br>0.02<br>0.01 0.01<br>0.00 0.00<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Llama2-13B (b) Baichuan2-13B<br>Attr Div Attr Div<br>**----- End of picture text -----**<br>


Figure 3: Information flow divergence comparison on Winogrande. 

CoT performance. For datasets, we select Winogrande (Sakaguchi et al., 2020) and CSQA (Talmor et al., 2019).[4] The detailed implementation of this experiment is shown in Appendix C. 

**Result and Analysis** Figure 2 illustrates our experimental results on Winogrande (The results on CSQA are shown in Appendix C). We can find that: **(Claim 1) When the Rationale Drift issue occurs, CoT receives less information from the question context compared to the correct case.** On both datasets and different models, there is a significantly lower information flow between the question context and CoT when the LLM generates a drifting rationale (the blue line) compared to the correct one (the orange line). This aligns with Hypothesis 1. **(Claim 2) The shallow attention layers are crucial for LLMs to extract contextual information.** In all cases, both the information flow and the gap peak at around the 15th attention layer, indicating these layers are significant sites for the rationale generation. 

**Supplementary Experiment** In the main experiment, we use golden labels to generate the correct CoT. To eliminate the influence of this additional factor on our results, we design a supplementary experiment. Concretely, we first use the label to generate CoTs from correct reasoning cases, which serves as a control group. Then, we compute the context information divergence between the newly generated CoT _cn_ and the original one _co_ , i.e.: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0004-05.png)


where _q_ is the question context. We compare this divergence between the control and drifting group, whose results are reported in Figure 3 and Figure 10. As we can see, the gap between the correct CoT and the drifting one (the blue line) is larger 

> 4Unless otherwise specified, we use the same models and datasets in the following probing experiments. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0004-08.png)


**----- Start of picture text -----**<br>
1 3.0 1 2.0<br>5 5<br>2.5<br>10 10<br>1.5<br>15 2.0 15<br>20 1.5 20 1.0<br>25 25<br>1.0<br>30 30 0.5<br>0.5<br>35 35<br>40 0.0 40 0.0<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Heads Heads<br>(a) Winogrande (b) CSQA<br>Layers Layers<br>**----- End of picture text -----**<br>


Figure 4: Attention tracing results across different attention heads on Llama2-13B. 

than the control group (the orange line) and the max divergence occurs in shallow layers (around the 15th layer). **This indicates that a correct CoT indeed gets more information flow from the context in shallow attention layers, validating the effectiveness of Claim 1 and 2.** 

## **3.3 Attention Tracing Experiment** 

**Experimental Settings** We also design an experiment based on attention scores to validate Hypothesis 1 from another perspective. For a pair of rationales < _c, c[∗]_ > targeting the same question context _q_ (c denotes the correct CoT and _c[∗]_ is the drifting one), we compute their attention divergence: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0004-13.png)


Here, we replace the _Attr_ ( _A_[(] _i,j[l]_[)][)][ in Equation][ 2][ with] the weights on the _h_ -th attention matrix head _Ai,j_[(] _[l,h]_[)] and repeat the calculation in Equation 3. 

**Result and Analysis** The results on two datasets are shown in Figure 4 and Figure 11. We can get the following observations: (1) The attention divergence is greater than 0 in most heads, which indicates **a lack of information from the question context in Rationale Drift cases (consistent with Claim 1).** (2) **The largest attention divergence appears around layer 15, which is consistent with the sites we find in Claim 2.** This once again illustrates that attention heads of these layers are crucial for the LLM to obtain contextual information when generating CoT. 

## **4 Tracing Information Flow in Answer** 

In this section, our goal is to verify the information loss based on Hypothesis 2. To achieve this, we first introduce the main tracing method in this section, which is called the causal tracing method 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0005-00.png)


**----- Start of picture text -----**<br>
0.5 0.4 0.5 0.4<br>0.4 0.4<br>0.3 0.3<br>0.3 0.3<br>0.2 0.2<br>0.2 0.2<br>0.1 0.1<br>0.1 0.1<br>0.0 0.0 0.0 0.0<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers Layers Layers<br>(a) Correct Case’s Attn (b) Correct Case’s MLP (c) Drifting Case’s Attn (d) Drifting Case’s MLP<br>context context context context<br>option option option option<br>cot cot cot cot<br>last last last last<br>**----- End of picture text -----**<br>


Figure 5: Intervention tracing results on Winogrande in correct and drifting answering cases. 

( _§_ 4.1). Next, by employing it, we trace the information flow in the answer generation stage and identify the mechanism behind the Answer Drift issue through comparative experiments ( _§_ 4.2). At last, we apply the attribution tracing method to verify our hypothesis from another perspective ( _§_ 4.3). 

## **4.1 Tracing Method** 

Since the task we study is in the form of multiplechoice questions, we set our focus on the feedforward pass that predicts the label. Inspired by the previous works (Meng et al., 2022; Stolfo et al., 2023; Geva et al., 2023), we take the causal tracing method to quantify the contribution of different intermediate variables during this pass. Specifically, for hidden states _h_[(] _i[l]_[)] in a clean run that predicts the answer, we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0005-05.png)


where _i, l_ is the _i_ -th token in the _l_ -th layer, _a_[(] _i[l]_[)] _[, m]_[(] _i[l]_[)] represents the activations of attention and MLP modules in Transformer (Vaswani et al., 2017). Supposing that a certain input part is represented as _z_ = [ _vi_[(] _[l]_[)] _[, ..., v] j_[(] _[l]_[)][]][ after passing through a model] component, we set _vk_[(] _[l]_[)] = _vk_[(] _[l]_[)] + _ϵ_ for _k ∈_ [ _i, j_ ] to intervene this hidden vector, where _ϵ_ is Gaussian noise.[5] Thus, we can compute the direct effect (DE) of this component: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0005-07.png)


where _P_ ( _o_ ) is the probability of the model’s final prediction, _Pz[∗]_[(] _[o]_[)][ is the probability after the inter-] vention. Therefore, through this metric, we can quantify the contributions of different components in changing the final prediction, thereby tracing the information flow in this stage. 

> 5We select _ϵ_ to be 3 times larger than the empirical standard deviation of hidden embeddings in each dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0005-10.png)


**----- Start of picture text -----**<br>
0.10<br>Drift 0.4 Drift<br>0.08 Correct Correct<br>0.3<br>0.06<br>0.04 0.2<br>0.02 0.1<br>0.00 0.0<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Winogrande (b) CSQA<br>Attr Attr<br>**----- End of picture text -----**<br>


Figure 6: Attribution tracing results on Llama2-13B during the answer generation stage. 

## **4.2 Intervention Tracing Experiment** 

**Experimental Settings** We sample correct and drifting answering cases from datasets, average over them and compute the average direct effect (ADE). Here we compute the impact of four components on the final prediction: context (question contexts), option (question options), CoT, and last (the last token before the label prediction). 

**Results and Analysis** We report the result of Llama2-13B on Winogrande in Figure 5 and others in Appendix D, from which we can get two conclusions: **(Claim 3) For attention modules, drifting cases loss information from the question.** When the answer is correct, we can observe a high effect on the context and option in the first layer (see Figure 5a). But for the drifting case, the LLM extracts limited information at these positions (see Figure 5c). This aligns with Hypothesis 2. **(Claim 4) For MLP modules, the information is not lost.** We observe the same high-effect sites in the last layer and shallow layers of the last token, they do not show regular differences (see Figure 5b and 5d). 

## **4.3 Attribution Tracing Experiment** 

**Experimental Settings** For further validation of our hypothesis, we also use the attribution score in _§_ 3 to trace the information flow in this stage. Referring to Equation 2, we compute the score between the question context and the last token 

|Needing a boost Emily asked Rachel to<br>Swap<br>A person who is ... Since Emily needed|Needing a boost Emily asked Rachel to<br>Swap<br>A person who is ... Since Emily needed|||𝜇𝐿𝑞+ 𝐿𝑐|𝜇𝐿𝑞+ 𝐿𝑐|𝜇𝐿𝑞+ 𝐿𝑐|𝜇𝐿𝑞+ 𝐿𝑐|𝜇𝐿𝑞+ 𝐿𝑐|||𝜆𝐿𝑐|𝜆𝐿𝑐|𝜆𝐿𝑐|𝜆𝐿𝑐|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||
||||||||||||||||
|||Needing a boost Emily asked|Rachel to|…  is tall in height||A person who is ... Since Emily needed|||a boost to|…, Emily is not tall in height.|||||
||<br>Swap<br>A person who is ... Since Emily needed|||<br>_<br>𝜆𝐿𝑐+ 𝐿𝑞||||||<br><br>Rachel<br>𝜇𝐿𝑞<br>Rachel to … _ is tall in height|||||
||||||||||||||||
||||||||||||||||
|||A person who is ... Since Emily needed||a boost to …, Emily is not tall in height.||||Needing a boost Emily asked||Rachel to … _ is tall in height|||||
|||||Question<br>CoT<br>Last Token|||||||||||



Figure 7: An example of our serial-position swap method. 

(since it’s used for generating the answer). We set the _F_ ( _·_ ) in Equation 1 as the loss for predicting the final answer, comparing the scores for correct and drifting cases after averaging across samples. 

**Results and Analysis** The results of this experiment are reported in Figure 6 and 15. We can observe that, **when the Answer Drift issue occurs, the information flow from the question significantly decreases** . This verifies the information loss we mention in Hypothesis 2 and Claim 3. 

## **5 Mitigating Toxic CoT Problem** 

In this section, we propose a novel method called RIDERS ( **R** esidual decod **I** ng and s **ER** ial-position **S** wap) to address the Toxic CoT problem. We first introduce the two components in it, which are designed based on Hypothesis 1 and 2, respectively ( _§_ 5.1). Then, we conduct experiments on commonsense reasoning benchmarks, demonstrating the effectiveness of our method ( _§_ 5.2 and _§_ 5.3). At last, we conduct extra experiments to further emphasize the contribution of our approach ( _§_ 5.4). 

## **5.1 Mitigation Method** 

**Residual Decoding** We design a new decoding methodology to address the Rationale Drift issue, in which we construct a virtual residual structure during the CoT generation, “connecting” the question context with each CoT token. Our decoding algorithm is demonstrated in Algorithm 1. In each iteration of generating a new token, we first select the top _n_ tokens with the highest probabilities and record their logits scores (line 3). Then we calculate the attention score between the context and current token like Equation 4, normalize it, and add it as an additional reward to promote more information flow (lines 6,7). Finally, we select the token with the highest score to update the input and repeat the process until the termination condition is met. We use the attention matrix in layer 15 to 

## **Algorithm 1** Residual Decoding Algorithm 

|**Algorithm 1**Residual DecodingAlgorithm|**Algorithm 1**Residual DecodingAlgorithm|
|---|---|
|**Require:** model _M_, input _x_, question context _q_, candi-||
||date_num_n_, weight_ω_.|
|1:|**for**iteration_i ∈_0_,_1_, ..._**do**|
|2:|_logits_=_M_(_x_)|
|3:|_tokens_,_scores_= top(_logits, n_)|
|4:|**for**_j ∈_1_, ...n_**do**|
|5:|_t_=_tokens_[_i_]|
|6:|_attn_score_=Attn(_q_,_t_) / Attn(_x_,_t_)|
|7:|_scores_[_j_] =_scores_[_j_]+_ω ∗attn_score_|
|8:|**end for**|
|9:|_idx_= argmax(_scores_)|
|10:|_t_=_tokens_[_idx_]|
|11:|_x_=_x_+_t_|
|12:|**if**stop(_t_)**then**|
|13:|**break**|
|14:|**end if**|
|15:|**end for**|
|16:|**return**_x_|



compute the attention score, since it is crucial for the exchange of contextual information according to Claim 2. More implementation details of this method are provided in Appendix E. 

**Serial-Position Swap** In this method, we attempt to compensate for the information lack in the Answer Drift issue. According to previous research on the serial-position effect in context, models tend to utilize information better at the beginning and end of the input (Qin et al., 2023; Liu et al., 2023b). In our work, the beginning of the input are prompts, while the current question and the generated CoT are both located at the end. Therefore, when they are closer to the last token, their information is more easily utilized in the final prediction. As in Figure 7, we denote the lengths of the question and CoT as _Lq_ and _Lc_ , and assume that the key information is located at positions _µLq_ and _λLc_ (similar to the center of mass in physics). We can infer that, after the swapping operation in Figure 7, the distance from the question to the end is reduced ( _µLq_ + _Lc → µLq_ ). Besides, if we consider the total distance from the question and CoT to the end, we can perform the following calculation: 

|**Method**|**Winogrande**<br>**CSQA**<br>**HellaSwag**<br>**SIQA**<br>**PIQA**<br>**Avg**|
|---|---|
||ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_|
|Few-shot Answer<br>Chain-of-Thought<br>Self-Consistency<br>Self-Refne<br>Least-to-Most<br>Contrasive CoT|57.1<br>-<br>66.8<br>-<br>45.3<br>-<br>68.2<br>-<br>61.6<br>-<br>59.8<br>-<br>56.7<br>43.1<br>69.9<br>28.8<br>45.4<br>34.8<br>67.3<br>37.3<br>62.7<br>41.3<br>60.4<br>37.1<br>56.4<br>40.9<br>72.4<br>24.9<br>46.2<br>33.7<br>64.7<br>41.0<br>54.6<br>45.8<br>58.9<br>37.3<br>48.7<br>50.3<br>55.5<br>47.0<br>42.0<br>37.2<br>65.1<br>38.8<br>50.8<br>48.3<br>52.4<br>44.3<br>58.9<br>38.8<br>69.0<br>27.5<br>31.3<br>39.4<br>66.3<br>36.6<br>65.7<br>32.2<br>58.2<br>34.9<br>59.4<br>38.0<br>71.1<br>23.2<br>45.9<br>35.6<br>67.0<br>36.0<br>68.1<br>44.9<br>62.3<br>35.5|
|**Ours**<br>RD Only<br>SPS Only<br>RIDERS|58.6<br>29.5<br>72.2<br>13.5<br>49.4<br>24.0<br>69.4<br>20.8<br>66.3<br>26.2<br>63.0<br>22.8<br>59.1<br>22.8<br>72.6<br>18.9<br>49.3<br>18.7<br>69.7<br>28.8<br>68.5<br>27.6<br>63.7<br>23.4<br>**60.7**<br>**16.6**<br>**73.2**<br>**6.7**<br>**50.6**<br>**12.8**<br>**71.9**<br>**15.3**<br>**69.9**<br>**15.9**<br>**65.3**<br>**13.5**|



Table 1: Performance comparison across five commonsense reasoning datasets on Llama2-13B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0007-02.png)


where _d_ 1 is the total distance in normal serial positions and _d_ 2 is the distance after swapping the two components. In most scenarios, we have _Lq < Lc_ , thus, we can infer that ∆ _d <_ 0. That means, if we replace the original order of “ _[Question] + [CoT]_ ” with the order of “ _[CoT] + [Question]_ ”, we can not only increase the intensify of information flow from the question to the final prediction, but also reduce the total information loss due to the reduction in total distance. Although this method is straightforward in implementation, it proves to be effective in both theory and experiments. 

## **5.2 Mitigation Experimental Settings** 

**Datasets** Following previous works, we use five representative commonsense reasoning benchmarks: **WinoGrande** (Sakaguchi et al., 2020), **CSQA** (Talmor et al., 2019), **HellaSwag** (Zellers et al., 2019), **SIQA** (Sap et al., 2019) and **PIQA** (Bisk et al., 2020). The specific information of each dataset is reported in Appendix F. 

**Metrics** In addition to the commonly used **Accuracy (ACC)** metric, we also introduce a new metric — **Toxic Rate (TR)** , to quantify the severity of Toxic CoT problems: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0007-07.png)


where _Cd_ denotes questions that models give correct answers directly and _Wf_ denotes questions that models give wrong answers after applying method _f_ . Thus, we can infer that the lower the TR, the fewer Toxic CoT problems the method introduces. 

**Baselines** As our research focuses on enhancing CoT methods in commonsense reasoning, we 

|**Method**|**Winogrande**<br>**CSQA**|
|---|---|
||Type1<br>Type2<br>Type1<br>Type2|
|CoT<br>RD Only<br>SPS Only<br>RIDERS|0.0<br>0.0<br>0.0<br>0.0<br>41.9<br>47.8<br>56.7<br>46.2<br>58.1<br>78.3<br>46.7<br>**92.3**<br>**74.2**<br>**82.6**<br>**73.3**<br>84.6|



Table 2: Accuracy on the two types of drifting issues. 

select some of the latest CoT-like methods applicable to this task for comparison: **Few-shot Answer** , **Chain-of-Thought** (Wei et al., 2022), **SelfConsistency** (Wang et al., 2023c), **Self-Refine** (Madaan et al., 2023), **Least-to-Most** (Zhou et al., 2023) and **Contrasive CoT** (Chia et al., 2023). For all methods, we employ a 5-shot prompt and use 4 NVIDIA GeForce RTX 3090 GPUs for inference. More implementation details can be found in the Appendix F. 

## **5.3 Mitigation Results** 

The main result of our experiments on Llama2-13B is shown in Table 1. (The results on more models are presented in Appendix F.) We can get the following conclusions: **(1) Our method effectively mitigates Toxic CoT problems.** Compared to CoT prompting, our method reduces the Toxic Rate by an average of **23.6%** across five datasets. Besides, compared to other advanced CoT-like methods, our method causes the fewest Toxic CoT problems (decreased by an average of **22.0%** over SOTA methods). **(2) Our method can also improve the model’s overall performance on commonsense reasoning.** Our work improves the accuracy on all benchmarks (improved by **5.5%** compared to CoT and **3.0%** compared to SOTA methods on average). This proves that the Toxic CoT problem poses a bottleneck in LLM’s commonsense reasoning, highlighting the value of our work. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
0.08 CoT 0.10 CoT<br>0.07 RD 0.08 SPS<br>0.06<br>0.05 0.06<br>0.04<br>0.03 0.04<br>0.02 0.02<br>0.01<br>0.00 0.00<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) RD method (b) SPS method<br>Attr Attr<br>**----- End of picture text -----**<br>


Figure 8: Information flow comparison on Winogrande after applying our two methods. 

|**Method**|**ProofWriter**<br>**FOLIO**<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_|
|---|---|
|oT<br>IDERS|56.8<br>19.9<br>45.1<br>11.6<br>**78.2**<br>**6.4**<br>**48.0**<br>**7.5**|



Table 3: Performance comparison across two logical reasoning tasks on Llama2-13B. 

## **5.4 Discussion and Analysis** 

**Performance on Two Drifting Issues** To demonstrate the effectiveness of our method in addressing the Rationale Drift issue (Type1) and Answer Drift issue (Type2), we conduct experiments on these samples and report the results in Table 2. Both of our methods can mitigate the corresponding issues (RD solves **49.3%** Rationale Drift issue on average, while SPS resolves **85.3** % Answer Drift issue on average). This verifies the validity of our hypothesis 1, 2, as all of these methods are built upon them. Besides, combining the two methods leads to even greater improvements, demonstrating the necessity of optimizing from both of these perspectives. 

**Performance in the Model** In _§_ 3 and _§_ 4, we probe information loss in two issues by tracing the information flow in models. Here, we repeat the attribution tracing experiments, comparing the differences before and after applying our method to further validate the effectiveness of our work. As we can see from Figure 8 and 16, our two methods (orange lines) increase the information flow from questions in two stages compared to CoT prompting (blue lines). This indicates our method indeed compensates for the information loss in the LLM. 

**Performance on Other Tasks** Our work mainly focuses on commonsense reasoning tasks. To further evaluate the effectiveness of our approach on other types and forms of tasks, we conduct the main experiments on two logical reasoning tasks: **ProofWriter** (Tafjord et al., 2021) and **FOLIO** (Han et al., 2022)). As presented in Table 3, we 

|**Method**|**Wino**|**CSQA**|**Hella**|**SIQA**|**PIQA**|**Avg**|
|---|---|---|---|---|---|---|
|CoT|3.2k|2.6k|4.2k|2.9k|2.7k|3.1k|
|SC|4.5k|3.2k|5.5k|3.8k|4.3k|4.3k|
|SR|9.2k|6.9k|10.2k|7.6k|6.5k|8.1k|
|L2M|8.7k|7.9k|11.4k|8.3k|6.7k|8.6k|
|CON|4.6k|3.5k|5.4k|3.8k|3.5k|4.2k|
|Ours|3.6k|2.9k|4.9k|3.4k|3.3k|3.6k|



Table 4: Token consumption per example comparison. 

can find that the Toxic CoT problem also exists in other tasks, and our method can still mitigate it. 

**Cost Analysis** For the applicability, we measure the computation and time cost of our approach. Here we compare the token cost between our method and the baseline. According to Table 4, our method requires fewer tokens compared to other SOTA methods (only 1 _._ 2 _×_ cost of the basic CoT method). We also compare the time cost of our decoding method in Appendix G and find that the speed of RD is comparable to existing decoding strategies. Therefore, we illustrate the costefficiency of our approach across different tasks. 

## **6 Related Work** 

## **6.1 CoT Problems Analysis and Mitigation** 

Recently, many works have focused on analyzing and mitigating problems in CoT reasoning. For analytical work, most studies focus on blackbox LLMs. Through intervening or paraphrasing prompts and comparing outputs, researchers can interpret the reasons leading to errors in the model’s reasoning (Lanham et al., 2023; Liu et al., 2023c; Wang et al., 2023a). For optimization works, they design additional supervision signals or training processes for the model (Ramnath et al., 2023; Liu et al., 2023a) or leverage external resources for the model (Shinn et al., 2023; He et al., 2023; Lyu et al., 2023). However, these works lack the probing of inner mechanisms behind these problems, leading to insufficient analysis or less universally applicable optimization methods. 

## **6.2 Mechanistic Interpretability** 

The work on mechanistic interpretability aims to understand the internal mechanisms of models when performing various tasks. Early work focused on how the model stores factual knowledge internally (Meng et al., 2022; Dai et al., 2022). In recent times, some research efforts have shifted towards examining how models retrieve and utilize knowledge. This includes internal knowledge 

retrieval (Geva et al., 2023), knowledge retrieval from prompts (Wang et al., 2023b; Jin et al., 2024b), and the utilization of knowledge for reasoning purposes, such as math reasoning (Stolfo et al., 2023) and multi-step reasoning (Hou et al., 2023). However, there is limited existing work that explains commonsense reasoning and CoT reasoning, which are significant contributions of this work. 

## **7 Conclusion** 

In this paper, we find a problem named Toxic CoT, which results in the model’s reasoning deviating from the original correct answer when utilizing CoT-like prompting. Through tracing the internal information flow of the LLM with attribution tracing and causal tracing methods, we prove that this problem is mainly caused by the model’s lack of information from the question in shallow attention layers when generating rationales or answers. Based on this result, we propose the RIDERS method to mitigate the Toxic CoT problem from both decoding and serial position perspectives. Through extensive experiments on multiple commonsense reasoning datasets, we verify the effectiveness of our approach in mitigating Toxic CoT problems and enhancing the model’s overall commonsense reasoning capabilities. 

## **Limitations** 

Although our work conducts an in-depth interpretation and mitigation of the Toxic CoT problem, it has several limitations. Firstly, like former commonsense reasoning works (Liu et al., 2022, 2023a; Xie et al., 2023), our research focuses on the form of multi-choice questions. This stems from the absence of effective evaluation methods for openended commonsense reasoning, leading to the predominance of benchmarks in this format. This calls for advancements in benchmark-related research. Secondly, we refrain from analyzing Toxic CoT problems in more reasoning tasks such as math, primarily due to the poor performance of current moderately-sized white-box models on these tasks. For instance, Llama2-13B achieves a mere 7.2% accuracy on GSM8K (Cobbe et al., 2021) without utilizing the CoT technique. This calls for developments in model-related research. We leave these limitations as our future work to explore. 

## **Acknowledgement** 

This work is supported by the Strategic Priority Research Program of Chinese Academy of Sciences (No. XDA27020203), the National Natural Science Foundation of China (No. 62176257,62276095). 

## **References** 

- Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: reasoning about physical commonsense in natural language. In _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020_ , pages 7432– 7439. AAAI Press. 

- Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, and Lidong Bing. 2023. Contrastive chain-ofthought prompting. _CoRR_ , abs/2311.09277. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _CoRR_ , abs/2110.14168. 

- Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 8493– 8502. Association for Computational Linguistics. 

- Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. 2023. Dissecting recall of factual associations in auto-regressive language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 12216–12235. Association for Computational Linguistics. 

- Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Luke Benson, Lucy Sun, Ekaterina Zubova, Yujie Qiao, Matthew Burtell, David Peng, Jonathan Fan, Yixin Liu, Brian Wong, Malcolm Sailor, Ansong Ni, Linyong Nan, Jungo Kasai, Tao Yu, Rui Zhang, Shafiq R. Joty, Alexander R. Fabbri, Wojciech Kryscinski, Xi Victoria Lin, Caiming Xiong, and Dragomir Radev. 2022. FOLIO: natural language reasoning with first-order logic. _CoRR_ , abs/2209.00840. 

- Yaru Hao, Li Dong, Furu Wei, and Ke Xu. 2021. Selfattention attribution: Interpreting information interactions inside transformer. In _Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, ThirtyThird Conference on Innovative Applications of Arti-_ 

_ficial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021_ , pages 12963–12971. AAAI Press. 

- Hangfeng He, Hongming Zhang, and Dan Roth. 2023. Rethinking with retrieval: Faithful large language model inference. _CoRR_ , abs/2301.00303. 

- Yifan Hou, Jiaoda Li, Yu Fei, Alessandro Stolfo, Wangchunshu Zhou, Guangtao Zeng, Antoine Bosselut, and Mrinmaya Sachan. 2023. Towards a mechanistic interpretation of multi-step reasoning capabilities of language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 4902–4919. Association for Computational Linguistics. 

- Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Qiuxia Li, and Jun Zhao. 2024a. Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy_ , pages 16867–16878. ELRA and ICCL. 

- Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. 2024b. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. _CoRR_ , abs/2402.18154. 

- Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukosiute, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. 2023. Measuring faithfulness in chainof-thought reasoning. _CoRR_ , abs/2307.13702. 

- Linhan Li, Huaping Zhang, Chunjin Li, Haowen You, and Wenyao Cui. 2023. Evaluation on chatgpt for chinese language understanding. _Data Intell._ , 5(4):885– 903. 

- Jiacheng Liu, Skyler Hallinan, Ximing Lu, Pengfei He, Sean Welleck, Hannaneh Hajishirzi, and Yejin Choi. 2022. Rainier: Reinforced knowledge introspector for commonsense question answering. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 8938–8958. Association for Computational Linguistics. 

- Jiacheng Liu, Ramakanth Pasunuru, Hannaneh Hajishirzi, Yejin Choi, and Asli Celikyilmaz. 2023a. 

   - Crystal: Introspective reasoners reinforced with selffeedback. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 11557–11572. Association for Computational Linguistics. 

- Kang Liu, Yangqiu Song, and Jeff Z. Pan. 2024. Editorial for special issue on commonsense knowledge and reasoning: Representation, acquisition and applications. _Mach. Intell. Res._ , 21(2):215–216. 

- Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023b. Lost in the middle: How language models use long contexts. _CoRR_ , abs/2307.03172. 

- Ziyi Liu, Isabelle Lee, Yongkang Du, Soumya Sanyal, and Jieyu Zhao. 2023c. SCORE: A framework for self-contradictory reasoning evaluation. _CoRR_ , abs/2311.09603. 

- Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-ofthought reasoning. _CoRR_ , abs/2301.13379. 

- Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. _CoRR_ , abs/2303.17651. 

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ . 

- Guanghui Qin, Yukun Feng, and Benjamin Van Durme. 2023. The NLP task effectiveness of long-range transformers. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2023, Dubrovnik, Croatia, May 2-6, 2023_ , pages 3756–3772. Association for Computational Linguistics. 

- Sahana Ramnath, Brihi Joshi, Skyler Hallinan, Ximing Lu, Liunian Harold Li, Aaron Chan, Jack Hessel, Yejin Choi, and Xiang Ren. 2023. Tailoring selfrationalizers with multi-reward distillation. _arXiv preprint arXiv:2311.02805_ . 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. Winogrande: An adversarial winograd schema challenge at scale. In _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020_ , pages 8732– 8740. AAAI Press. 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social iqa: Commonsense reasoning about social interactions. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019_ , pages 4462–4472. Association for Computational Linguistics. 

- Omar Shaikh, Hongxin Zhang, William Held, Michael S. Bernstein, and Diyi Yang. 2023. On second thought, let’s not think step by step! bias and toxicity in zeroshot reasoning. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 4454–4470. Association for Computational Linguistics. 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In _Thirty-seventh Conference on Neural Information Processing Systems_ . 

- Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan. 2023. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6- 10, 2023_ , pages 7035–7052. Association for Computational Linguistics. 

- Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Xiangyang Liu, Hang Yan, Yunfan Shao, Qiong Tang, Shiduo Zhang, et al. 2024. Moss: An open conversational large language model. _Machine Intelligence Research_ , pages 1–18. 

- Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. Proofwriter: Generating implications, proofs, and abductive statements over natural language. In _Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021_ , volume ACL/IJCNLP 2021 of _Findings of ACL_ , pages 3621–3634. Association for Computational Linguistics. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , pages 4149–4158. Association for Computational Linguistics. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, 

   - Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. _CoRR_ , abs/2307.09288. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pages 5998–6008. 

- Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. 2023a. Towards understanding chain-of-thought prompting: An empirical study of what matters. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 2717–2739. Association for Computational Linguistics. 

- Chenhao Wang, Pengfei Cao, Jiachun Li, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Qiuxia Li, and Jun Zhao. 2024. Leros: Learning explicit reasoning on synthesized data for commonsense question answering. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy_ , pages 10303–10315. ELRA and ICCL. 

- Chenhao Wang, Jiachun Li, Yubo Chen, Kang Liu, and Jun Zhao. 2022. Cn-automic: Distilling chinese commonsense knowledge from pretrained language models. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 9253–9265. Association for Computational Linguistics. 

- Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. 2023b. Label words are anchors: An information flow perspective for understanding in-context learning. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 9840–9855. Association for Computational Linguistics. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023c. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ . 

- Chaojie Wen, Xudong Jia, and Tao Chen. 2023. Improving extraction of chinese open relations using pre-trained language model and knowledge enhancement. _Data Intell._ , 5(4):962–989. 

- Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, Min-Yen Kan, Junxian He, and Qizhe Xie. 2023. Self-evaluation guided beam search for reasoning. In _Thirty-seventh Conference on Neural Information Processing Systems_ . 

- Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu. 2023. Baichuan 2: Open large-scale language models. _CoRR_ , abs/2309.10305. 

- Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers_ , pages 4791–4800. Association for Computational Linguistics. 

- Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

## **A Early Statistical Experiments** 

In this section, we conduct early experiments on existing representative commonsense reasoning datasets to analyze the prevalence of Toxic CoT problems through statistical methods. 

**Datasets** We utilize five representative commonsense reasoning datasets to analyze the distribution of Toxic CoT problems. The basic information of the dataset is outlined in Table 5. It is noteworthy that, owing to the extensive size of Hellaswag’s dev set (over 10,000), we extract 2,000 instances for the experiment. 

**Metric** We design a new metric called Toxic Rate, which measures the proportion of Toxic CoT problems among all errors. Its calculation method is shown in Equation 8. 

**Results** The result of our early statistical experiments is reported in Table 6. Here we use _**Llama213B-Chat-hf**_ to present the white-box LLM and use _**GPT-3.5-turbo-1106**_ to present the black-boxmodel. The average Toxic Rates are as high as **37.0%** and **32.8%** across the five datasets, indicating that this issue cannot be ignored and warrants further investigation. 

the preceding context when generating CoT or the final answers. In the factual error, the CoT contains incorrect factual knowledge, which leads to erroneous answers. The presence of question errors reflects the subpar quality of the dataset. In such cases, questions may exhibit multiple viable answers or all options are incorrect. As for the other error, the questions trigger certain refusal-toanswer mechanisms in the model (e.g., inquiries about how to commit murder), leading to the identification of incorrect answers. 

As the inconsistency error constitutes the predominant portion of all reasons, our work focuses on addressing this issue. We further categorize this error into Rationale Drift and Answer Drift based on the error occurrence (see _§_ 2.2 for their definitions). 

|**Reason**|**Winogrande**|**CSQA**|
|---|---|---|
|Inconsistent Error|**71(78.9%)**|**48(53.9%)**|
|Factual Error|16(17.8%)|17(19.1%)|
|Question Error|2(2.2%)|21(23.6%)|
|Other Error|1(1.1%)|3(3.4%)|
|Sum|90|89|



Table 7: The classification of CoT reasoning errors 

## **B Toxic Reason Statistical Experiments** 

In this section, we manually categorize the error types of Toxic CoT problems through statistical classification. Specifically, we sample 1,000 examples from CSQA and 1,000 examples from Winogrande, classifying the Toxic CoT problems (In all of the probing experiments in the main text, we use these samples as our probing data). The results are presented in Table 7. In the inconsistent error, the model exhibits logical inconsistency with 

## **C More Details for Reasoning Tracing** 

**Method Implementation** We introduce the attribution score method in _§_ 3.1. In Equation 1, we set _m_ = 20 following the previous works. For _F_ ( _·_ ), we set it as the language modeling loss (for next-token prediction) during the CoT generation. Here, we obtain this value directly from the output of the _LlamaForCausalLM_ module using the _Transformers_ library. In Equation 2, we partition the step num- 

||**Winogrande**<br>**CSQA**<br>**HellaSwag**<br>**SIQA**<br>**PIQA**<br>Split<br>dev<br>dev<br>dev<br>dev<br>dev<br>#Sample Num<br>1267<br>1221<br>2000<br>1954<br>1838<br>#Option Num<br>2<br>5<br>4<br>3<br>2<br>Table 5: Dataset information in this work.|
|---|---|
||**Winogrande**<br>**CSQA**<br>**HellaSwag**<br>**SIQA**<br>**PIQA**<br>**Avg**|
|Llama2-13B<br>43.1<br>28.8<br>34.8<br>37.3<br>41.2<br>37.0<br>GPT-3.5<br>34.8<br>37.8<br>27.5<br>37.5<br>26.4<br>32.8||



Table 6: Toxic rate on different datasets and models. 

bers _|N |_ in CoT based on the occurrence of periods in the text. For models, we use Llama2-13B-Chat and Baichuan2-13B-Chat. 

**Attribution Tracing Experiment** Figure 17 illustrates the prompt we use for generating correct CoTs from drifting cases. After the generation, we will manually filter out the wrong CoT and conduct the comparative experiment. We use a 5-shot to generate CoT and concatenate it to the question for our probing experiments. Figure 9 shows the remaining results of this experiment, from which we can get the same conclusions as Section 3.2. Additionally, we also conduct the supplementary experiments on CSQA and report the results in Figure 10. 

**Attention Tracing Experiment** Figure 11 reports more results in this experiment. 

## **D More Details for Answering Tracing** 

**Intervention Tracing Experiment** Figure 12, 13, 14 and 15 report the remaining intervention tracing experiment results, which are consistent with our conclusion in the main text. 

## **E Mitigation Method Implementation** 

**Residual Decoding** Here, we provide a detailed explanation of Algorithm 1. At the beginning, we set the input to the entire question (contexts + options). In line 2, we get the logits from the output of the _LlamaForCausalLM_ . In line 6, we calculate the attention score by summing the values on the attention matrix corresponding to the tokens. We use the output character “ _</s>_ ” as the termination condition for Llama2-13B generation in line 12. 

**Serial-Position Swap** In this method, we swap the positions of the question and the generated CoT, outputting the option with the highest logits score. This method can be implemented under both fewshot and zero-shot settings, demonstrating its costefficiency. 

of CoT reasoning and then follow it with one round of feedback to generate the final answer. For all the baselines, we release the prompts in our source code. We implement all of the methods on Llama2-13B-Chat-hf and Baichuan2-13B-Chat. 

**Our methods** In our RD method, we set two hyperparameters — candidate_num _n_ and weight _ω_ , and here are their specific values in the experiments: for Winogrande, we set _n_ to 4 and _ω_ to 80, for CSQA, we set _n_ to 10 and _ω_ to 135, for HellaSwag, we set _n_ to 3 and _ω_ to 80, for SIQA, we set _n_ to 10 and _ω_ to 160, for PIQA, we set _n_ to 4 and _ω_ to 120. Additionally, in Figure 18,19,20,21,22, we list our method’s few-shot prompts on five datasets. Note that both CoT prompting and our two methods utilize the same prompt. 

**Results** In our paper, we choose Llama2-13B for experiments because it is an open-source LLM of suitable size and has great influence. Here, we also repeat all experiments on Baichuan213B to verify the generality of our work (see Table 8). Additionally, in Table 9, we show our experimental results on two other representative models: Mistral-7B-Instruct-v0.2 and gpt-3.5-turbo-1106 (since it is a closed source model, here we only apply the SPS method). 

## **G More Details for Cost Analysis** 

Table 10 illustrates the time cost comparison of our residual decoding methods with other strategies. Here we set _num_beams_ in the beam search strategy as 5, and _candidate_num_ in the RD strategy as 3. We compute the average seconds cost per example over 50 samples for each dataset. On average, our decoding strategy takes 2.6 times longer than greedy search and 1.4 times longer than beam search. This reflects that our decoding method has significantly stronger performance while having a comparable overall time to these main decoding strategies. 

## **F More Details for the Mitigation Experiment** 

**Dataset** In this experiment, the specific information of all datasets can be found in Table 5. 

**Baselines** For the Self-Consistency method, we sample 5 CoTs and use a majority voting method to select the final predicted answer. For the Self-Refine method, we first conduct one round 

|**Method**|**Winogrande**<br>**CSQA**<br>**HellaSwag**<br>**SIQA**<br>**PIQA**<br>**Avg**|
|---|---|
||ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_|
|Few-shot Answer<br>Chain-of-Thought<br>Self-Consistency<br>Self-Refne<br>Least-to-Most<br>Contrasive CoT|59.7<br>-<br>69.9<br>-<br>**47.1**<br>-<br>68.8<br>-<br>60.7<br>-<br>61.2<br>-<br>58.7<br>34.8<br>66.8<br>37.8<br>41.1<br>27.5<br>66.5<br>37.5<br>66.1<br>26.4<br>59.8<br>32.8<br>56.1<br>38.8<br>64.3<br>41.5<br>39.4<br>30.6<br>67.8<br>33.5<br>**70.0**<br>21.0<br>59.5<br>33.1<br>59.6<br>35.7<br>66.7<br>38.8<br>39.2<br>29.9<br>62.2<br>43.1<br>62.6<br>31.0<br>58.1<br>35.7<br>59.7<br>35.6<br>62.8<br>45.6<br>37.8<br>32.1<br>66.5<br>36.8<br>62.8<br>20.6<br>57.9<br>34.1<br>55.2<br>40.6<br>67.0<br>37.4<br>38.7<br>29.7<br>65.0<br>37.3<br>65.9<br>26.2<br>58.4<br>34.2|
|**Ours**<br>RD Only<br>SPS Only<br>RIDERS|59.2<br>21.1<br>69.5<br>25.3<br>44.4<br>19.3<br>68.9<br>25.0<br>67.0<br>12.9<br>61.8<br>20.7<br>59.9<br>22.8<br>69.7<br>23.5<br>46.3<br>15.2<br>**69.2**<br>27.8<br>67.7<br>14.2<br>62.6<br>20.7<br>**60.1**<br>**18.6**<br>**71.3**<br>**14.6**<br>46.3<br>**13.0**<br>69.0<br>**20.5**<br>67.2<br>**8.3**<br>**62.8**<br>**15.0**|



Table 8: Performance comparison across five commonsense reasoning datasets on Baichuan2-13B. 

|**Model**<br>**Method**|**Winogrande**<br>**SIQA**|
|---|---|
||ACC_↑_<br>TR_↓_<br>ACC_↑_<br>TR_↓_|
|Mistral-7B<br>CoT<br>RIDERS|59.2<br>27.9<br>70.2<br>28.2<br>**62.0**<br>**15.3**<br>**70.6**<br>**24.5**|
|GPT-3.5<br>CoT<br>RIDERS|64.6<br>48.8<br>70.6<br>30.3<br>**70.1**<br>**41.7**<br>**72.1**<br>**29.0**|



Table 9: Performance comparison on Mistral-7B and GPT-3.5. 

|**Strategy**|**Wino**|**CSQA**|**Hella**|**SIQA**|**PIQA**|**Avg**|
|---|---|---|---|---|---|---|
|Greedy|5.2|2.8|4.9|3.6|5.8|4.5|
|Beam|8.1|4.9|10.1|5.8|9.4|7.7|
|Ours|13.4|6.6|16.2|7.4|15.9|11.9|



Table 10: Decoding time per example comparison. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0015-06.png)


**----- Start of picture text -----**<br>
0.10 0.14<br>Drift Drift<br>Correct 0.12 Correct<br>0.08<br>0.10<br>0.06 0.08<br>0.06<br>0.04<br>0.04<br>0.02<br>0.02<br>0.00 0.00<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Llama2-13B (b) Baichuan2-13B<br>Attr Attr<br>**----- End of picture text -----**<br>


Figure 9: Attribution tracing results on CSQA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0016-00.png)


**----- Start of picture text -----**<br>
0.035 0.05<br>Drift Drift<br>0.030 Normal Normal<br>0.04<br>0.025<br>0.020 0.03<br>0.015<br>0.02<br>0.010<br>0.005 0.01<br>0.000<br>0.00<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Llama2-13B (b) Baichuan2-13B<br>Attr Div Attr Div<br>**----- End of picture text -----**<br>


Figure 10: Information flow divergence comparison on CSQA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0016-02.png)


**----- Start of picture text -----**<br>
1 1<br>5 3.0 5 1.5<br>10 2.5 10<br>1.0<br>15 2.0 15<br>20 20<br>1.5 0.5<br>25 25<br>1.0<br>30 30 0.0<br>0.5<br>35 35<br>40 0.0 40 -0.5<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Heads Heads<br>(a) Winogrande (b) CSQA<br>Layers Layers<br>**----- End of picture text -----**<br>


Figure 11: Attention tracing results across different attention heads on Baichuan2-13B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
0.5 0.4<br>0.4<br>0.3<br>0.3<br>0.2<br>0.2<br>0.1<br>0.1<br>0.0 0.0<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Correct Case’s Attn (b) Correct Case’s MLP<br>0.5 0.4<br>0.4<br>0.3<br>0.3<br>0.2<br>0.2<br>0.1<br>0.1<br>0.0 0.0<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers<br>(c) Drifting Case’s Attn (d) Drifting Case’s MLP<br>context context<br>option option<br>cot cot<br>last last<br>context context<br>option option<br>cot cot<br>last last<br>**----- End of picture text -----**<br>


Figure 12: Intervention tracing results on CSQA in correct and drifting answering cases (Llama2-13B). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0018-00.png)


**----- Start of picture text -----**<br>
0.200 0.200<br>0.175 0.175<br>0.150 0.150<br>0.125 0.125<br>0.100 0.100<br>0.075 0.075<br>0.050 0.050<br>0.025 0.025<br>0.000 0.000<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Correct Case’s Attn (b) Correct Case’s MLP<br>0.200 0.200<br>0.175 0.175<br>0.150 0.150<br>0.125 0.125<br>0.100 0.100<br>0.075 0.075<br>0.050 0.050<br>0.025 0.025<br>0.000 0.000<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers<br>(c) Drifting Case’s Attn (d) Drifting Case’s MLP<br>context context<br>option option<br>cot cot<br>last last<br>context context<br>option option<br>cot cot<br>last last<br>**----- End of picture text -----**<br>


Figure 13: Intervention tracing results on Winogrande in correct and drifting answering cases (Baichuan2-13B). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0019-00.png)


**----- Start of picture text -----**<br>
0.200 0.200<br>0.175 0.175<br>0.150 0.150<br>0.125 0.125<br>0.100 0.100<br>0.075 0.075<br>0.050 0.050<br>0.025 0.025<br>0.000 0.000<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Correct Case’s Attn (b) Correct Case’s MLP<br>0.200 0.200<br>0.175 0.175<br>0.150 0.150<br>0.125 0.125<br>0.100 0.100<br>0.075 0.075<br>0.050 0.050<br>0.025 0.025<br>0.000 0.000<br>1 5 10 15 20 25 30 35 40 1 5 10 15 20 25 30 35 40<br>Layers Layers<br>(c) Drifting Case’s Attn (d) Drifting Case’s MLP<br>context context<br>option option<br>cot cot<br>last last<br>context context<br>option option<br>cot cot<br>last last<br>**----- End of picture text -----**<br>


Figure 14: Intervention tracing results on CSQA in correct and drifting answering cases (Baichuan2-13B). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Drift 0.35 Drift<br>0.25<br>Correct 0.30 Correct<br>0.20<br>0.25<br>0.15 0.20<br>0.15<br>0.10<br>0.10<br>0.05<br>0.05<br>0.00 0.00<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Winogrande (b) CSQA<br>Attr Attr<br>**----- End of picture text -----**<br>


Figure 15: Attribution tracing results on Baichuan2-13B during the answer generation stage. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning_images/acl_2024_Focus_on_Your_Question!_Interpreting_and_Mitigating_Toxic_CoT_Problems_in_Commonsense_Reasoning.pdf-0020-02.png)


**----- Start of picture text -----**<br>
0.10 0.5<br>CoT CoT<br>0.08 SPS 0.4 SPS<br>0.06 0.3<br>0.04 0.2<br>0.02 0.1<br>0.00 0.0<br>0 5 10 15 20 25 30 35 40 0 5 10 15 20 25 30 35 40<br>Layers Layers<br>(a) Winogrande (b) CSQA<br>Attr Attr<br>**----- End of picture text -----**<br>


Figure 16: Information flow comparison on CSQA after applying our two methods. 

**System Instruction:** You are a helpful, respectful and honest assistant. You should use your reasoning abilities to given the reason for the answer to the given questions. Your response should be in this form: 'Reason: {reason}’. 

**Example Input:** Question: {QUESTION}\nAnswer: {ANSWER} **Example Output:** Reason: {COT} 

**Input:** Question: {QUESTION}\nAnswer: {ANSWER} **Output:** 

Figure 17: Prompts for correct CoT generation. 

**System Instruction:** You are a helpful, respectful and honest assistant. You should use your reasoning abilities to answer the given questions in reasoning tasks. You should reply the correct rationales and the answer. Your response should be in this form: '{reason} So the answer is: ({option}) {answer}'. If you don't know the answer to a question, please reply 'Answer: None’. 

**Question:** The test was hard for Samuel but a breeze for Randy , since _ had failed to study for it.\n(1) Samuel (2) Randy 

**Answer:** To pass a test, a person need to study for it. If a person feel the test like a breeze, it means the test is easy for him. A person feels the test easy, because he studies hard for it. Since we know that Samuel feel the test very hard, she may fail to study for it. So the answer is: (1) Samuel. 

**Question:** Kyle slowly wormed their way into the life of Derrick, because _ was good and manipulating people.\n(1) Kyle (2) Derrick 

**Answer:** A person wormes his way into other's life, because he is friendly and approachable. A friendly person is considered good. A person is seen as manipulating people, that means he like to interact with others and others like him. Since Kyle slowly wormed their way into the life of Derric, Kyle will be seen as good and manipulating people. So the answer is: (1) Kyle. 

**Question:** Donald was very grounded but Michael often got lost in their daydreams. _ was very capricious all the time.\n(1) Donald (2) Michael **Answer:** A person is grounded means he works hard and does not like to fantasize. A person often gets lost in his daydreams, he is seen as unrealistic and egocentric. A person is capricious all the time means he does everything only according to his own ideas. Since Michael often gets lost in their daydreams but Donald does not, Michael is seen as very capricious all the time. So the answer is: (2) Michael. 

**Question:** After stopping when running, Betty was able to help Rachel treat their asthma because _ has it too.\n(1) Betty (2) Rachel **Answer:** A person who has a mild disease may knows how to treat it.  Since Since Betty was able to help Rachel treat their asthma, Betty may has it, too. So the answer is: (1) Betty. 

**Question:** Jeffrey was sick with the stomach flu and not hungry, but Christopher was starving. _ ordered food.\n(1) Jeffrey (2) Christopher **Answer:** Stomach flu is a viral infection in the digestive system. A person is starve so that he need to eat something next. A person is not hungry means he does not want to eat something. Ordering food means to buy food online. A person may eat food after ordering it. Since Christopher was starving and Jeffery was sick and not hungry, Christopher tended to order food. So the answer is: (2) Christopher. 

Figure 18: 5-shot prompts for Winogrande. 

**System Instruction:** You are a helpful, respectful and honest assistant. You should use your reasoning abilities to answer the given questions in reasoning tasks. You should reply the correct rationales and the answer. Your response should be in this form: '{reason} So the answer is: ({option}) {answer}'. If you don't know the answer to a question, please reply 'Answer: None’. 

**Question:** Google Maps and other highway and street GPS services have replaced what? \n (1) atlas (2) mexico (3) countryside (4) united states (5) oceans **Answer:** Electronic maps and GPS services are the modern version of paper atlas. In that case, the atlas have been replaced by Google Maps and other highway and street GPS services. So the answer is: (1) atlas. 

**Question:** The fox walked from the city into the forest, what was it looking for? \n (1) pretty flowers. (2) hen house (3) natural habitat (4) storybook (5) dense forest 

**Answer:** Since the fox walk from the city into the forest, he may looks for something in the forest but not in the city. From all of the options, the natural habitat are usually away from cities. So the answer is: (3) natural habitat. 

**Question** : You can share files with someone if you have a connection to a what? \n (1) freeway (2) radio (3) wires (4) computer network (5) electrical circuit 

**Answer:** Files usually can be stored in the computers. In that case, we can share them over the Internet. Thus, if we connect to a computer network, we can share the file with others. So the answer is: (4) computer network. 

**Question:** Too many people want exotic snakes. The demand is driving what to carry them? \n (1) ditch (2) shop (3) north america (4) outdoors (5) pet shops 

**Answer:** If people want exotic snakes, they may like to raise snakes as pets. If there is a demand for snakes as pets, pet shops will be pushed to carry them, in order to make more money. So the answer is: (5) pet shops. 

**Question:** The body guard was good at his duties, he made the person who hired him what? \n (1) better job (2) feel safe (3) irritated (4) save money (5) headache 

**Answer:** The job of body guards is to ensure the safety and security of the employer. People ususally hire the body guard to make themselves safe. So the answer is: (2) feel safe. 

Figure 19: 5-shot prompts for CSQA. 

**System Instruction:** You are a helpful, respectful and honest assistant. You should use your reasoning abilities to answer the given questions in reasoning tasks. You should reply the correct rationales and the answer. Your response should be in this form: '{reason} So the answer is: ({option}) {answer}'. If you don't know the answer to a question, please reply 'Answer: None’. 

**Question:** Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then \n (1) the man adds wax to the windshield and cuts it. (2) a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled. (3) the man puts on a christmas coat, knitted with netting. (4) the man continues removing the snow on his car. **Answer:** Snow must be removed from a car before one can drive it. Since there is some snow covering the car, so the man tends to remove it next. So the answer is: (4) the man continues removing the snow on his car. 

**Question:** The man in the center is demonstrating a hairstyle on the person wearing the blue shirt. the man in the blue shirt \n (1) is standing on the sponge cutting the hair of the person wearing the blue shirt. (2) is doing the hairstyle with his hand and the hairspray. (3) sits on the chair next to the sink. (4) is being shown eye to eye. 

**Answer:** One should sit still when getting a haircut. People often sits down when they are get a haircut. From the context, we can get that the man in the blue shirt is getting a haircut, so he should sit down and wait. So the answer is: (3) sits on the chair next to the sink. 

**Question:** The roof is done and a view of the entire house is shown to show off the finished roof. the woman \n (1) is standing in front of the home, smiling while talking. (2) interviews the man again and leaves the room. (3) shows the soil with two lay-ups of shingle and applies a layer onto the top shingle. (4) stacks the bags on the side and begins putting stencils on the top. 

**Answer:** One usually feels pleased after finishing a home improvement project. If a person feels pleased, he tends to smile. Since the woman has finished all the work, she may be proud of it and feel pleased. Then she may smile. So the answer is: (1) is standing in front of the home, smiling while talking. 

**Question:** People practice ballet in a studio alone and in couples. then \n (1) a man begins dancing and top dancing standing on the ground. (2) a boy and a girl dance ballet, then a man enter and dance with the girl. (3) the band performs ballet in the studio and in the open gathered. (4) people dances and dances together, dancing. 

**Answer:** Ballet is a kind of dance. If people pratice ballet, they often dance with their partener or dance alone, which matches the description of option (2). So the answer is: (2) a boy and a girl dance ballet, then a man enter and dance with the girl. 

**Question:** A person is seen standing on a tennis court bouncing a ball. another man \n (1) takes his turn speaking to the camera. (2) walks up hitting a birdie. (3) is seen standing ready on the other side in front of a large audience. (4) is holding the racket next to him. **Answer:** The player who bounces the ball is the one who serves. In the tennis race, two players are located on opposite sides of the court, one is serving and the other is preparing to receive the ball. Since there is a person who serves, another one should stands ready. So the answer is: (3) is seen standing ready on the other side in front of a large audience. 

Figure 20: 5-shot prompts for HellaSwag. 

**System Instruction:** You are a helpful, respectful and honest assistant. You should use your reasoning abilities to answer the given questions in reasoning tasks. You should reply the correct rationales and the answer. Your response should be in this form: '{reason} So the answer is: ({option}) {answer}'. If you don't know the answer to a question, please reply 'Answer: None’. 

**Question:** Quinn wanted to help me clean my room up because it was so messy. What will Quinn want to do next? \n (1) Eat messy snacks (2) help out a friend (3) Pick up the dirty clothes 

**Answer:** Quinn want to clean the room up. Picking up the dirty clothes is one way to clean the room. Thus, quinn will want to pick up the dirty clothes next. So the answer is: (3) Pick up the dirty clothes 

**Question:** Sasha's mom passed out in the middle of the party. Aubrey took Sasha's mom to the hospital. What will Aubrey want to do next? \n (1) help Aubrey go back home (2) keep on partying without the mom (3) going on with the mom 

**Answer:** Sasha's mom is sent to the hospital beacuse she passes out. After that, she becomes a patient. A patient needs to be cared for. Aubrey may need to take care of her. Thus, Aubrey will go on with the mom. So the answer is: (3) going on with the mom 

**Question:** Their cat kept trying to escape out of the window, so Jan placed an obstacle in the way. How would Jan feel afterwards? \n (1) scared of losing the cat (2) normal (3) relieved for fixing the problem 

**Answer:** The cat try to escape so Jan need to stop it to avoid losing the cat. Jan place an obstacle in the way so the cat can not escape. The problem is been solved. Thus, Jan will fell relieved for fixing the problem. So the answer is: (3) relieved for fixing the problem 

**Question:** Sydney had so much pent up emotion, they burst into tears at work. How would Sydney feel afterwards? \n (1) affected (2) like they released their tension (3) worse 

**Answer:** Crying is often a way to release tension. Sydney burst into tears at work. Thus, she would release the tension. So the answer is: (2) like they released their tension 

**Question:** Sydney got a raise and a new promotion. What does Sydney need to do before this? \n (1) be bad at her job (2) do a good job (3) be lazy **Answer:** To get a raise and a new promotion, Sydney needs to work hard and do a good job. Sydney got a raise and a new promotion. Thus, she need to do a good job before this. So the answer is: (2) do a good job 

Figure 21: 5-shot prompts for SIQA. 

**System Instruction:** You are a helpful, respectful and honest assistant. You should use your reasoning abilities to answer the given questions in reasoning tasks. You should reply the correct rationales and the answer. Your response should be in this form: '{reason} So the answer is: ({option}) {answer}'. If you don't know the answer to a question, please reply 'Answer: None’. 

**Question:** How do you flood a room?\n(1) fill it with objects. (2) fill it with water 

**Answer:** Too much water can cause flooding. Thus, if we want to flood a room, we should use water. So the answer is: (2) fill it with water 

**Question:** How can I get oil stains out of my driveway?\n(1) Douse each stain with a couple cans of beer. (2) Douse each stain with a couple cans of soda. 

**Answer:** Sodium carbonate solution can wash away oil stains. The soda is a kind of sodium carbonate solution. Thus, you can use cans of soda to get oil stains out of your driveway. So the answer is: (2) Douse each stain with a couple cans of soda. 

**Question:** Soothe a painful sunburn.\n(1) Wait until brewed tea bag is cool, then apply on burn. (2) Wait until brewed tea bag is hot, then apply on burn. 

**Answer:** Sunburn can be alleviated by applying cold material. Thus, you should apply cool tea rather than hot tea bag to soothe your sunburn. So the answer is: (1) Wait until brewed tea bag is cool, then apply on burn. 

**Question:** What can I use for fuel in an alcohol stove?\n(1) Use acetone. (2) Use vinegar. 

**Answer:** Acetone is flammable, while vinegar is not. If you want to use something for fuel, the thing you use should be flammable. Thus, you should use acetone for fuel in an alcohol stove. So the answer is: (1) Use acetone. 

**Question:** How can I cut the handles of metal cutlery?\n(1) Use a hand saw to cut the handles. (2) Use a hand drill to cut the handles. 

**Answer:** A hand saw is used for making cuts and a hand drill is used for making holes. If you want to cut something, you should use a hand saw rather than hand drill. So the answer is: (1) Use a hand saw to cut the handles. 

Figure 22: 5-shot prompts for PIQA. 

