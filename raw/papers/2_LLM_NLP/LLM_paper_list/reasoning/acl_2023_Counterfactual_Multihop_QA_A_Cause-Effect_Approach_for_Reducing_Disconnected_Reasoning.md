# **Counterfactual Multihop QA: A Cause-Effect Approach for Reducing Disconnected Reasoning** 

## **Wangzhen Guo Qinkang Gong Hanjiang Lai** _[∗]_ Sun Yat-Sen University 

{guowzh6,gongqk}@mail2.sysu.edu.cn laihanj3@mail.sysu.edu.cn 

## **Abstract** 

Multi-hop QA requires reasoning over multiple supporting facts to answer the question. However, the existing QA models always rely on shortcuts, e.g., providing the true answer by only one fact, rather than multi-hop reasoning, which is referred as _disconnected reasoning_ problem. To alleviate this issue, we propose a novel counterfactual multihop QA, a causal-effect approach that enables to reduce the disconnected reasoning. It builds upon explicitly modeling of causality: 1) the direct causal effects of disconnected reasoning and 2) the causal effect of true multi-hop reasoning from the total causal effect. With the causal graph, a counterfactual inference is proposed to disentangle the disconnected reasoning from the total causal effect, which provides us a new perspective and technology to learn a QA model that exploits the true multi-hop reasoning instead of shortcuts. Extensive experiments have conducted on the benchmark HotpotQA dataset, which demonstrate that the proposed method can achieve notable improvement on reducing disconnected reasoning. For example, our method achieves 5.8% higher points of its Supp _s_ score on HotpotQA through true multihop reasoning. The code is available at supplementary material. 

## **1 Introduction** 

Multi-hop question answering (QA) (Groeneveld et al., 2020; Ding et al., 2019; Asai et al., 2019; Shao et al., 2020) requires the model to reason over multiple supporting facts to correctly answer a complex question. It is a challenging task, and many datasets, e.g., HotpotQA (Yang et al., 2018) and approaches (Fang et al., 2019; Zhu et al., 2021b) have been proposed for this reasoning task. 

One of the main problems of multihop QA models is _disconnected reasoning_ (Trivedi et al., 2020), which allows the models to exploit the 

> _∗_ *Corresponding Author 

reasoning shortcuts (Jiang and Bansal, 2019; Lee et al., 2021) instead of multi-hop reasoning to cheat and obtain the right answer. Taking Fig. 1 as an example, to answer the question _“until when in the U.S. Senate"_ , we should consider two supporting facts to infer the answer “Devo- _Director of Research for_ rah Adler = _⇒_ Barack Obama _served from_ 2005 _to_ 2008 = _⇒_ 2008". However, one may also infer the correct answer by just utilizing the types of problems, e.g., we can find the corresponding fact _“from 2005 to 2008"_ in the contexts without reasoning to answer this type of question _“until when"_ . 

One possible solution for reducing the disconnected reasoning is to strengthen the training dataset via extra annotations or adversarial examples, which make it cannot find the correct answers by only one supporting fact. For example, Jiang and Bansal (2019) constructed the adversarial examples to generate better distractor facts. (Trivedi et al., 2020) firstly defined a evaluate measure, **DiRe** in short, to measure how much the QA model can cheat via disconnected reasoning. Then, a transformed dataset is constructed to reduce disconnected reasoning. Besides, counterfactual intervention (Lee et al., 2021; Ye et al., 2021) had also been explored to change the distribution of the training dataset. These methods improve the generalizability and interpretability of the multi-hop reasoning QA model via balancing the train data, which is noted as debiased training in QA model (Niu et al., 2021). However, when the existing approaches decrease the disconnected reasoning, the original performance also drops significantly. It is still challenging to reduce disconnected reasoning while maintaining the same accuracy on the original test set. 

Motivated by causal inference (Pearl and Mackenzie, 2018; Pearl, 2022; Niu et al., 2021), we utilize the counterfactual reasoning to reduce the disconnected reasoning in multi-hop QA and 

also obtain the robust performance on the original dataset. We formalize a causal graph to reflect the causal relationships between question ( _Q_ ), contexts and answer ( _Y_ ). To evaluate the disconnected reasoning, contexts are further divided into two subsets: _S_ is a supporting fact and _C_ are the remaining supporting facts. Hence, we can formulate the disconnected reasoning as two natural direct causal effects of ( _Q, S_ ) and ( _Q, C_ ) on _Y_ as shown in Fig. 1. With the proposed causal graph, we can relieve the disconnected reasoning by disentangling the two natural direct effects and the true multi-hop reasoning from the total causal effect. A novel counterfactual multihop QA is proposed to disentangle them from the total causal effect. We utilize the generated probing dataset proposed by (Trivedi et al., 2020) and DiRe to measures how much the proposed multi-hop QA model can reduce the disconnected reasoning. Experiment results show that our approach can substantially decrease the disconnected reasoning while guarantee the strong performance on the original test set. The results indicate that the proposed approach can reduce the disconnected reasoning and improve the true multi-hop reasoning capability. 

The main contribution of this paper is threefold. Firstly, our counterfactual multi-hop QA model formulates disconnected reasoning as two direct causal effects on answer, which is a new perspective and technology to learn the true multi-hop reasoning. Secondly, our approach achieves notable improvement on reducing disconnected reasoning compared to various state-of-the-arts. Thirdly, our causal-effect approach is model-agnostic and can be used for reducing disconnected reasoning in many multi-hop QA architectures. 

## **2 Related Work** 

Multi-hop question answering (QA) requires the model to retrieve the supporting facts to predict the answer. Many approaches and datasets have been proposed to train QA systems. For example, HotpotQA (Yang et al., 2018) dataset is a widely used dataset for multi-hop QA, which consists of fullwiki setting (Das et al., 2019; Nie et al., 2019; Qi et al., 2019; Chen et al., 2019; Li et al., 2021; Xiong et al., 2020) and distractor setting (Min et al., 2019b; Nishida et al., 2019; Qiu et al., 2019; Jiang and Bansal, 2019; Trivedi et al., 2020). 

In fullwiki setting, it firstly finds relevant facts from all Wikipedia articles, and then fin- 

ish the multi-hop QA task with the found facts. The retrieval model is important in this setting. For instance, SMRS (Nie et al., 2019) and DPR (Karpukhin et al., 2020) found the implicit importance of retrieving relevant information in the semantic space. Entity-centric (Das et al., 2019), CogQA (Ding et al., 2019) and Golden Retriever (Qi et al., 2019) explicitly used the entity that is mentioned or reformed in query key words to retrieve next hop document. Furthermore, PathRetriever (Asai et al., 2019) and HopRetriever (Li et al., 2021) can iteratively select the documents to form a paragraph-level reason path using RNN. MDPR (Xiong et al., 2020) retrieved passages only using dense query vector in many times. These methods hardly discuss the QA model’s disconnected reasoning problem. 

In distractor setting, 10 paragraphs, two gold paragraphs and eight distractors, are given. Many methods have been proposed to strengthen the model’s capability of multi-hop reasoning, using graph neural network (Qiu et al., 2019; Fang et al., 2019; Shao et al., 2020) or adversarial examples or counterfactual examples (Jiang and Bansal, 2019; Lee et al., 2021) or the sufficiency of the supporting evidences (Trivedi et al., 2020) or make use of the pretrained language models (Zhao et al., 2020; Zaheer et al., 2020). 

However, Min et al. (2019a) demonstrated that many compositional questions in HotpotQA can be answered with a single hop. It means that QA models can take shortcuts instead of multi-hop reasoning to produce the corrected answer. To relieve the issue, Jiang and Bansal (2019) added adversarial examples as hard distractors during training. Recently, (Trivedi et al., 2020) proposed an approach, DiRe, to measure the model’s disconnected reasoning behavior and use the supporting sufficiency label to reduce the disconnected reasoning. Lee et al. (2021) selected the supporting evidence according to the sentence causality to the predicted answer, which guarantees the explainability of the behavior of the model. While, the original performance also drops when reducing the disconnected reasoning. 

**Causal Inference.** Recently, causal inference (Pearl and Mackenzie, 2018; Pearl, 2022) has been applied to many tasks of natural language processing, and it shows promising results and provides strong interpretability and generalizability. The representative works include counterfactual interven- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0003-00.png)


**----- Start of picture text -----**<br>
𝑆 𝑄𝐴𝑚𝑜𝑑𝑒𝑙 𝑞 : Devorahcandidate, who served untilAdler worked as whenDirector in the U.S. Senate ?of Research for a 2008 presidential<br>𝐾1 𝑠 : Barack Obama previously served in the U.S. Senate representing Illinois<br>from 2005 to 2008…the first African American to have served as president…<br>𝑄 𝐾 𝑌 Adler…served as Director of Research for the Barack Obama campaign for<br>𝐶 𝐾2 𝑐 the 2008 Democratic presidential nomination.distractors facts…<br>(a) Answer: 2008 Supporting facts:<br>𝑠 𝑄𝐴𝑚𝑜𝑑𝑒𝑙 𝑠 [∗] 𝑄𝐴𝑚𝑜𝑑𝑒𝑙 𝑠 𝑄𝐴𝑚𝑜𝑑𝑒𝑙<br>𝐾1 𝐾1∗ 𝐾1<br>𝑞 𝐾 [∗] 𝑌 𝑞 𝐾 [∗] 𝑌 𝑞 𝐾 𝑌<br>𝑐 [∗] 𝐾2∗ 𝑌𝑘1,𝑘2 [∗] ,𝑘∗ 𝑐 𝐾2 𝑌𝑘1 [∗] ,𝑘2,𝑘∗ 𝑐 𝐾2 𝑌𝑘1 [∗] ,𝑘2 [∗] ,𝑘<br>(b) (c) (d)<br>Counterfactual Situations<br>𝑠 [∗] : Barack Obama previously served in the U.S. School representing Illinois from 2005 to 2008… the first<br>African American to have served as president…<br>(b) 𝑞, 𝑠, 𝑐 [∗] : 2008<br>(c) 𝑞, 𝑠 [∗] , 𝑐: 2008<br>**----- End of picture text -----**<br>


Figure 1: Illustration of disconnected reasoning in multi-hop QA, where red node denotes question _Q_ , blue node is a supporting fact _S_ , and orange and green nodes denote the remaining facts _C_ . Deep gray nodes mean their variables are reference values instead of the given values, (e.g., _S_ = _s[∗]_ instead of _S_ = _s_ ). **(a):** Causal graph of multi-hop QA model; **(b):** is a possible scenario of disconnected reasoning, which uses only one fact _s_ to answer the question. **(c):** is another possibility of disconnected reasoning, e.g., the exclusive method to find whether _s_ is a supporting fact by a process of elimination of other facts _c_ . **(d):** is the true multi-hop reasoning. All facts _s_ and _c_ are taken into considered to produce the answer. 

tion for visual attention (Rao et al., 2021), causal effect disentanglement for VQA (Niu et al., 2021), the back-door and front-door adjustments (Zhu et al., 2021a; Wang et al., 2021; Yang et al., 2021). Our method can be viewed as a complement of the recent approaches that utilize the counterfactual inference (Lee et al., 2021; Ye et al., 2021) to identify the supporting facts and predict answer. 

## **3 Preliminaries** 

In this section, we use the theory of causal inference (Pearl and Mackenzie, 2018; Pearl, 2022) to formalize our multi-hop reasoning method. Support that we have a multi-hop dataset _D_ and each instance has the form of ( _Q, P_ ; _Y_ ), where _Q_ is a question and _P_ = _{s_ 1 _, s_ 2 _, · · · , sn}_ is a context consisting of a set of _n_ paragraphs. And _Y_ is the ground-truth label. Given a question _Q_ with multiple paragraphs as a context _P_ , the multi-hop QA models are required to identify which paragraphs are the supporting facts and predict an answer using the supporting facts. 

**Causal graph** . In multi-hop QA, multiple supporting facts are required to predict the answer. 

While QA models may use only one fact to give the answer, which is referred as disconnected reasoning. For example, given a question _q_ and only a paragraph _s_ , the disconnected reasoning model can predict the correct answer or correctly determine whether the paragraph _s_ is the supporting fact. Hence, to define the causal graph of disconnected reasoning, the context _P_ is further divided into a paragraph _S_ and the remaining paragraphs _C_ . Now the ( _Q, P_ ; _Y_ ) becomes ( _Q, S, C_ ; _Y_ ). That is each instance ( _Q, P_ ; _Y_ ) is converted into _n_ examples, i.e., ( _q, s_ 1 _, C_ = _{s_ 2 _, · · · , sn}_ ; _Y_ ( _s_ 1)), _· · ·_ , ( _q, sn, C_ = _{s_ 1 _, · · · , sn−_ 1 _}_ ; _Y_ ( _sn_ )), where _Y_ ( _si_ ) = _{Fsi_ ; _A}_ includes the supporting fact and answer, where _A_ is the answer and _Fsi_ = 1 means the paragraph _s_ is the supporting fact otherwise it is not. For each example, we consider the disconnected reasoning with the fact _S_ (not _C_ ). 

The causal graph for multi-hop QA is shown in Figure 1 (a), where nodes denote the variables and directed edges represents the causal-and-effect relationships between variables. The paths in Figure 1 (a) are as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0003-08.png)


that the feature/knowledge _K_ 1 is extracted from the question ( _Q_ ) and the paragraph ( _S_ ) via the QA model backbone, e.g., BERT. _K_ 1 _→ Y_ represents the process that the label _Y_ is predicted by only using the _K_ 1. 

( _Q, C_ ) _→ K_ 2 _→ Y_ : Similarly, the feature/knowledge _K_ 2 extracted from the question ( _Q_ ) and the remaining paragraphs ( _C_ ) is used to predict the label _Y_ . 

( _Q, S, C_ ) _→_ ( _K_ 1 _, K_ 2) _→ K → Y_ : This path indicates that the QA model predicts the label _Y_ based on both the _K_ 1 and _K_ 2. 

Based on the above, the effect of _Q, S, C_ on Y can be divide into: 1) shortcut impacts, e.g., ( _Q, S_ ) _→ K_ 1 _→ Y_ and ( _Q, C_ ) _→ K_ 2 _→ Y_ , and 2) reasoning impact, e.g., ( _Q, S, C_ ) _→_ ( _K_ 1 _, K_ 2) _→ K → Y_ . The shortcut impacts capture the direct effect of ( _Q, S_ ) or ( _Q, C_ ) on _Y_ via _K_ 1 _→ Y_ or _K_ 2 _→ Y_ . The reasoning impact captures the indirect effect of ( _Q, S, C_ ) on _Y_ via _K → Y_ . 

Hence, to reduce the multi-hop QA model’s disconnected reasoning proposed in (Trivedi et al., 2020), we should exclude shortcut impacts ( _K_ 1 _→ Y_ and _K_ 2 _→ Y_ ) from the total effect. 

**Counterfactual definitions** . Figure 1 (a) shows the causal graph. From causal graph to formula, we denote the value of _Y_ , i.e., the answer _A_ (e.g., 2008) or the supporting paragraph _Fs_ (e.g., is the paragraph _s_ supporting fact?), would be obtained when question _Q_ is set to _q_ , the paragraph _S_ is set to _s_ and the remaining paragraphs _c_ = _P − s_ are used as the context _C_ , which is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0004-06.png)


For simplicity, we omit _A, s_ and unify both equations as _Yq,s,c_ = _Yq,s,c_ ( _A_ ) or _Yq,s,c_ = _Yq,s,c_ ( _s_ ). Since the causal effect of _q, s, c_ on _Y_ via _K_ 1 _, K_ 2 _, K_ on _Y_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0004-08.png)


in the following discussion. 

To disentangle the shortcut impacts from the total causal effect, we use the counterfactual causal inference to block other effects. To model the _K_ 1 _→ Y_ , the counterfactual formulation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0004-11.png)


which describes the situation where _K_ 1 is set to the original value _k_ 1 and _K_ and _K_ 2 are blocked. The 

_k[∗]_ and _k_ 2 _[∗]_[are the counterfactual notations.][The] _[ k]_[1] and _k[∗] , k_ 2 _[∗]_[represent the the two situations where] the _k_ 1 is under treatment in the factual scenario and _k[∗] , k_ 2 _[∗]_[are not under treatment (][Pearl and Macken-] zie, 2018) in the counterfactual scenario. The same 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0004-14.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0004-16.png)


**Causal effects** . According the counterfactual definitions, total effect of _q, s, c_ on _Y_ can be decomposed into the natural direct effects of _K_ 1 _, K_ 2 on _Y_ and the effect of _K_ on _Y_ as discussed before. The two natural direct effects cause the disconnected reasoning problem, and the effect of _K → Y_ is the desired multi-hop reasoning. 

As shown in Figure 1 (b), the effect of _K_ 1 on _Y_ with _K_ 2 _, K_ blocked and the effect of _K_ 2 on _Y_ with _K_ 1 _, K_ blocked can be easily obtained by setting the _S_ or _C_ to counterfactual values (please refer to Section 4 for more details). While the effect of _K_ on _Y_ can not be obtained by changing the values _S_ / _C_ to _S[∗]_ / _C[∗]_ . We follow (Niu et al., 2021) and total indirect effect (TIE) is used to express the effect of _K_ on _Y_ , which is formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0004-19.png)


## **4 Counterfactual Multihop QA** 

Following the former formulations, we propose to construct the counterfactual examples to estimate the natural direct effect of _K_ 1 and _K_ 2, as well as using parameters to estimate the total indirect effect of _K_ . And our calculation of _Y_ in Eq. (2), (3) and (5) is parametrized by a neural multi-hop QA model _F_ . Please note that _F_ can be any multi-hop QA models and our method is model-agnostic. 

## **4.1 Disentanglement of causal effect** 

_K_ 1 _→ Y_ . Specifically, in Eq. (2), the _Yk_ 1 _,k_ 2 _[∗][,k][∗]_[de-] scribes the situation where _K_ 1 is set to the factual value with _Q_ = _q, S_ = _s_ as inputs, and _K_ 2/ _K_ are set to the counterfactual values. Taking Figure 1 as an example, the QA model only consider the interaction between question _q_ and a given paragraph _s_ . The remaining paragraphs _c_ are not given. It is the disconnected reasoning (Trivedi et al., 2020). To obtain the counterfactual values of _K_ 2 and _K_ , we 

can set the context _C_ as its counterfactual sample, and we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0005-01.png)


In this paper, we randomly sample the remaining contexts from the training set to construct the counterfactual _c[∗]_ . It represents that the no-treatment or the prior knowledge of the remaining context is unknown. In the implementation, we randomly sample the remaining contexts in a mini-batch to replace the _c_ in original triple example ( _q, s, c_ ) and obtain the corresponding counterfactual triple example ( _q, s, c[∗]_ ). With that, we can feed it into the QA model to get _Yq,s,c[∗]_ . 

_K_ 2 _→ Y_ . Similarly, in Eq. (3), the _Yk_ 1 _[∗][,k]_[2] _[,k][∗]_ describes the situation where _K_ 2 is set to the factual value with the inputs _Q_ = _q_ and _C_ = _c_ . The _K_ 1 and _K_ are set to the counterfactual values with the counterfactual sample _S_ = _s[∗]_ as input, which is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0005-04.png)


One may argue that how to predict the label when _S_ is set to the counterfactual values? As the example shown in Figure 1, even without the paragraph _s_ as input, the QA model still can infer the paragraph _s_ is supporting fact via wrong reasoning: since all paragraphs in _C_ do not include the words about time range, the rest paragraph _s_ should be the supporting fact to answer the “until when" question. It is the exclusive method. The wrong reasoning is caused by an incorrect interaction between the paragraph _s_ and the remaining context _c_ . Hence, the _S_ is set to the counterfactual values can correct such incorrect interactions. 

Hence, in the implementation, we use the adversarial examples as suggested in (Jiang and Bansal, 2019) to construct the counterfactual _s[∗]_ , which aims to remove the incorrect interaction and perform multi-hop reasoning. Specifically, we randomly perturb the 15% tokens of the paragraph _s_ , 80% of which will be replaced by other random tokens, 10% of which will be replaced by the mask token (e.g. [ _MASK_ ]) of the tokenizer, and 10% of which will keep unchanged. After that, we obtain another counterfactual triple example ( _q, s[∗] , c_ ), we can feed it into QA model to get _Yq,s[∗] ,c_ . 

_K → Y_ . In Eq. (5), _Yq,s,c_ indicates that the question _s_ , paragraph _s_ and remaining context _c_ are visible to the QA model _F_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0005-08.png)


The main problem is that _Yk_ 1 _,k_ 2 _,k∗_ is unknown. It is also hard to use the counterfactual samples of _Q, S, C_ to obtain its value since the _q, s_ and _c_ should be the factual values for _k_ 1 _, k_ 2. In this paper, we follow the work (Niu et al., 2021) and also assume the model will guess the output probability under the no-treatment condition of _k[∗]_ , which is represented as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0005-10.png)


where _C_ is the output and a learnable parameter. Similar to Counterfactual VQA (Niu et al., 2021), we guarantee a safe estimation of _Yk_ 1 _,k_ 2 _,k∗_ in expectation. 

**Training objective** . From the above discussion, we can estimate the total causal effect: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0005-13.png)


Note that _Y_ can be any ground-truth labels of answer span prediction, supporting facts identification or answer type prediction. See Appendix A.1 for further implementation details. 

## **4.2 Training and Inference** 

**Training** . The model _F_ is expected to disentangle the two natural direct effect and the true multi-hop effect from the total causal effect. To achieve this goal, we apply the Eq. (10) to train the QA model _F_ . Our training strategy follows the HGN (Fang et al., 2019): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0005-17.png)


where _λ_ is a hyper-parameter and each term of _L_ is cross-entropy loss function. Specifically, for answer prediction, we utilize the Eq. (10) to obtain the predicted logits of the start and end position of the answer span, and respectively calculate the cross-entropy loss _Lstart_ and _Lend_ with corresponding ground truth labels. As for supporting facts prediction, similarly, we use the Eq. (10) to calculate the predicted logits in sentence level and paragraph level, and then calculate _Lsent_ and _Lpara_ . We also apply our counterfactual reasoning method to identify the answer type (Qiu et al., 2019; Fang et al., 2019), which consists of yes, no, span and entity. We use the [ _CLS_ ] token as the global representation to predict the answer type under the Eq. (10) and calculate _Ltype_ with the ground truth label. Entity prediction ( _Lentity_ ) (Fang et al., 2019) 

||Ans||Suppp|Suppp|Supps|Supps|Ans +|Suppp|Ans +|Supps|
|---|---|---|---|---|---|---|---|---|---|---|
||original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|
|BERT|74.2|47.5|94.1|73.9|83.8|64.1|71.0|36.5|64.2|32.5|
|+ours|74.0|**45.5**|**95.4**|**67.5**|82.8|**54.6**|**71.6**|**31.8**|63.9|**26.7**|
|XLNET|76.2|50.3|96.5|75.0|86.6|64.8|74.4|39.1|68.0|34.6|
|+ours|75.9|**49.8**|**96.6**|**74.1**|86.6|**63.6**|74.1|**38.0**|67.9|**33.6**|
|DFGN|71.7|44.5|94.4|73.8|83.8|64.0|68.7|34.2|62.1|30.4|
|+ours|**73.3**|48.3|**96.1**|**72.1**|**85.4**|**61.6**|71.5|36.0|65.1|31.7|
|HGN|73.3|47.0|91.1|67.4|81.4|59.0|68.3|33.6|62.0|30.2|
|+ours|70.9|**41.1**|**93.4**|67.6|**83.5**|**58.1**|67.6|**28.9**|61.8|**25.5**|



Table 1: F1 scores. The "original" denotes the model’s performance on the development set of HotpotQA in the distractor setting, and the "dire" indicates that the model scores on the corresponding probing set, which measures how much disconnected reasoning the model can achieve. The smaller score of "dire" is better. We can see that the proposed method can reduce disconnected reasoning while maintaining the same accuracy on the original dataset. 

||Ans||Suppp|Suppp|Supps|Supps|Ans +|Suppp|Ans +|Supps|
|---|---|---|---|---|---|---|---|---|---|---|
||original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|
|BERT|59.7|35.3|86.1|17.3|54.8|10.1|53.6|7.0|35.7|4.4|
|+ours|**60.2**|**33.7**|**87.9**|**3.7**|53.5|**2.0**|**55.2**|**1.4**|**36.6**|**0.8**|
|XLNET|62.0|38.0|91.8|14.5|58.9|8.5|58.6|6.5|40.1|4.2|
|+ours|61.7|**37.6**|**92.1**|**6.9**|**59.3**|**3.9**|58.5|**2.6**|**40.9**|**1.6**|
|DFGN|57.4|44.5|85.9|19.5|53.4|11.7|51.3|6.9|33.6|4.6|
|+ours|**59.6**|**35.9**|**90.9**|**6.2**|**57.7**|**3.3**|**55.9**|**2.3**|**38.4**|**1.4**|
|HGN|58.9|35.2|79.5|16.9|52.3|10.4|49.5|6.8|34.2|4.4|
|+ours|57.1|**30.1**|**85.1**|**6.4**|**56.1**|**3.3**|**51.3**|**2.2**|**36.0**|**1.2**|



Table 2: EM scores on the development set and probing set of HotpotQA in the distractor setting. 

is only a regularization term and the Eq. (10) is not applied to this term. 

**Inference** . As illustrated in Section 3, our goal is to exclude the natural direct effect ( _K_ 1 _→ Y, K_ 2 _→ Y_ ) and use the true multi-hop effect ( _K → Y_ ) to reduce the multi-hop QA model’s disconnected reasoning, so we use Eq. (5) for inference: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0006-06.png)


The time consumption of our approach is equal to the existing methods. 

## **5 Experiments** 

We extensively conduct the experiments on the HotpotQA (Yang et al., 2018) dataset. The compared results show that our proposed algorithm can reduce the amount of the disconnected reasoning while obtaining the strong performance on the original dev set. And it is general and suitable for other multi-hop QA architectures. 

**Datasets:** We utilize the distractor setting of HotpotQA, where each question is equipped with 

two ground truth supporting paragraphs and eight distractor paragraphs as the context. And the answer of the question is annotated as a span in one of the supporting paragraphs. To measure the disconnected reasoning (dire) in HotpotQA models, we generate the probing dataset following (Trivedi et al., 2020), which is only used in the test phase. Specifically, the probing dataset for HotpotQA in the distractor setting divides each example of the original dataset into two instances, both of which only contain one of two ground truth supporting paragraphs respectively. If the multi-hop QA model can arrive at the correct test output on two instances, it means that the model performs disconnected reasoning on the original example. Please refer to (Trivedi et al., 2020) for more details. 

**Baselines:** First, we simply use the BERT (Devlin et al., 2018) to predict the answer, supporting sentences and supporting paragraphs as the baseline. _BERT_ + _ours_ denotes that we apply our counterfactual multi-hop reasoning method based on BERT as the backbone. The proposed approach is model-agnostic and we also implement 

it on several multi-hop QA architectures, including DFGN (Qiu et al., 2019), HGN (Fang et al., 2019) and XLNet in Dire (Trivedi et al., 2020; Yang et al., 2019). Our proposed algorithm also can be implemented on other baselines. 

**Metrics:** Following the Dire (Trivedi et al., 2020), we report the metrics for HotpotQA: answer span (Ans), supporting paragraphs (Suppp), supporting sentences (Supps), joint metrics (Ans+ Suppp _,_ Ans + Supps). We show both EM scores and F1 scores to compare the performance between baselines and our counterfactual multi-hop reasoning method. 

## **5.1 Quantitative Results** 

For fairness, we conduct the experiments under the same preprocessing of the dataset following HGN (Fang et al., 2019), which select top K relevant paragraphs corresponding to each question example. And the experimental results are shown in Table 1 and Table 2. The main observation can be made as follows: 

_Our method can reduce disconnected reasoning._ Compared to BERT baseline, our proposed counterfactual multi-hop reasoning method can reduce the disconnected reasoning of answer prediction and supporting facts identification in both the paragraph level and sentence level. In particular, we can see big drops of 9.5 F1 points on Supps (from 64.1 to 54.6) and 13.6 EM points on Suppp (from 17.3 to 3.7) in disconnected reasoning (dire). Our method are better at reducing disconnected reasoning on the Exact Match (EM) evaluation metric. This is because EM is stricter evaluation metric. For example, EM requires both of the supporting facts should be predicted correctly while it has F1 scores even when only one supporting fact is predicted correctly. For dire evaluation where only one supporting fact is provided, our approach punishes this situation and achieves lower scores on EM metric of disconnected reasoning. It demonstrates that our method effectively reduce disconnected reasoning when there are without sufficient supporting facts. 

_Our method still guarantees the comparable performance on the original dev set_ . As seen from Table 1 and Table 2, the proposed method also maintain the same accuracy on the original set. It even shows the better performance on the supporting facts prediction in the paragraph level. 

_Our method is model-agnostic and it demonstrates effectiveness in several multi-hop QA mod-_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0007-07.png)


**----- Start of picture text -----**<br>
40 39.8<br>BERT<br>DFGN 37.2<br>HGN<br>BERT+ours<br>35 34.534.534.7<br>31.731.731.8<br>30<br>28.5 27.9 28.2<br>26.727.226.3<br>25<br>23.7<br>22.4<br>20 20.220.6 19.719.8<br>15<br>Ans Suppp Supps Ans + Suppp Ans + Supps<br>**----- End of picture text -----**<br>


Figure 2: F1 scores of real multi-hop reasoning, which is denoted as the original scores minus the dire scores. We compare BERT, DFGN, HGN and ours method except XLNET, as they utilize the same pretrained language model (i.e. bert-base-uncased). 

_els._ Based on our causal-effect insight, our proposed approach can easily be applied to other multihop QA architectures including XLNET, DFGN, HGN (Trivedi et al., 2020; Qiu et al., 2019; Fang et al., 2019). As shown in Table 1 and Table 2, our proposed counterfactual reasoning method achieves better performance. Our method can reduce the disconnected reasoning by introducing the proposed counterfactual approach in the training procedure. The dire scores of HGN (Fang et al., 2019) and XLNET (Trivedi et al., 2020) in Ans _,_ Suppp _,_ Supps all drop to some extent. Besides, the performances on the original dev set are comparable simultaneously. It indicates that our proposed method is more stable and general. 

In summary, reducing the disconnected reasoning and guarantee the strong performance on original development set indicate that the most progress of the model is attributed to the multi-top reasoning ( _K → Y_ ) capability. For intuitiveness, we also show the real multi-hop reasoning promoted by our proposed counterfactual reasoning approach, as shown in Fig. 2. 

## **5.2 Ablation Study** 

As illustrated in Section 4, our goal is to exclude the shortcut impacts ( _K_ 1 _→ Y, K_ 2 _→ Y_ ) to reduce the disconnected reasoning. Hence, we study the ablation experiments of excluding one of shortcuts impacts. We explore to remove _K_ 1 _→ Y_ or _K_ 2 _→ Y_ that reduce the disconnected reasoning, as shown in Table 3. We can see that exclude one of them can decrease the amount of disconnected reasoning to some extent on supporting facts identification except the answer span prediction. However, reliev- 

||Ans||Suppp|Suppp|Supps|Supps|Ans +|Suppp|Ans +|Supps|
|---|---|---|---|---|---|---|---|---|---|---|
||original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|
|BERT|74.2|47.5|94.1|73.9|83.8|64.1|71.0|36.5|64.2|32.5|
|+_K_1 _→Y_|74.2|50.2|96.6|73.5|85.7|61.9|72.5|37.8|65.7|32.8|
|+_K_2 _→Y_|74.6|48.8|96.4|68.2|85.6|57.7|72.8|34.5|66.0|30.1|
|+ours(full)|74.0|**45.5**|95.4|**67.5**|82.8|**54.6**|71.6|**31.8**|63.9|**26.7**|



Table 3: F1 scores of ablation study on the development set and probing set of HotpotQA in the Distractor setting. The "+ _K_ 1 _→ Y_ " denotes that we only utilize the counterfactual examples ( _q, s, c[∗]_ ) to estimate the shortcut impact of _K_ 1 _→ Y_ , and similarly the "+ _K_ 2 _→ Y_ " represents that only the counterfactual examples ( _q, s[∗] , c_ ) are used to estimate the shortcut impact of _K_ 2 _→ Y_ . 

||Ans||Suppp|Suppp|Supps|Supps|Ans +|Suppp|Ans +|Supps|
|---|---|---|---|---|---|---|---|---|---|---|
||original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|original|dire_↓_|
|random|**74.0**|**45.5**|**95.4**|67.5|**82.8**|54.6|**71.6**|31.8|**63.9**|26.7|
|uniform|73.8|45.9|92.1|**67.4**|76.6|**46.3**|69.2|31.8|59.6|**23.2**|



Table 4: F1 scores of ablation study of assumptions for counterfactual outputs _C_ on the development set and probing set of HotpotQA in the distractor setting. 

ing the both impacts of _K_ 1 _→ Y_ and _K_ 2 _→ Y_ can achieve better performance on decreasing disconnected reasoning. Because the model can always exploit another shortcut if only one of the shortcuts is blocked. 

We further conduct ablation studies to validate the distribution assumption for the counterfactual output of the parameter _C_ . Similar to CF-VQA (Niu et al., 2021), we empirically validate the two distribution assumptions, as shown in Table 4. The "random" denotes that _C_ are learned without constraint and it means that _CAns ∈_ R _[n] , Csupp ∈_ R[2] _, Ctype ∈_ R[4] respectively, and _n_ represents the length of the context. The "uniform" denotes that _C_ should satisfy uniform distribution and it means that _CAns, Csupp_ and _Ctype_ are scalar. As shown in Table 4, the random distribution assumption performs better than the uniform distribution assumption. 

## **6 Conclusion** 

In this work, we proposed a novel counterfactual reasoning approach to reduce the disconnected reasoning in multi-hop QA. We used the causal graph to explain the existing multi-hop QA approaches’ behaviors, which consists of the shortcut impacts and reasoning impacts. The shortcut impacts capture the disconnected reasoning. Thus the disconnected reasoning is formulated as natural direct causal effect and we construct the counterfactual examples during the training phase to estimate the both natural direct effects of question and context 

on answer prediction as well as supporting facts identification. The reasoning impact represents the multi-hop reasoning and is estimated via introducing learnable parameters. 

During the test phase, we excluded the natural direct effect and utilize the true multi-hop effect to decrease the disconnected reasoning. Experimental results demonstrate that our proposed counterfactual reasoning method can significantly drop the disconnected reasoning on probing dataset and guarantee the strong performance on original dataset, which indicates the most profress of the multi-hop QA model is attributed to the true multihop reasoning. Besides, our approach is modelagnostic, and can be applied to other multi-hop QA architectures to prevent the model from exploiting the shortcuts. 

Overall, our insights of reducing disconnected reasoning and learning the true multi-hop reasoning can motivate the development of new methods which contributes to promoting the multi-hop question answering task. 

## **Limitations** 

Our proposed method needs to construct counterfactual examples to estimate the natural direct effect of disconnected reasoning during the training phase, thus we need a little more GPU resources and computational time. However, the need of resource occupancy and time consumption of our approach does not increase during inference. Another limitation is that we use the learnable parameters 

to approximate the _Yk_ 1 _,k_ 2 _,k∗_ . In our future work, we will explore better approach to model it. 

## **References** 

- Akari Asai, Kazuma Hashimoto, Hannaneh Hajishirzi, Richard Socher, and Caiming Xiong. 2019. Learning to retrieve reasoning paths over wikipedia graph for question answering. _arXiv preprint arXiv:1911.10470_ . 

- Jifan Chen, Shih-ting Lin, and Greg Durrett. 2019. Multi-hop question answering via reasoning chains. _arXiv preprint arXiv:1910.02610_ . 

- Rajarshi Das, Ameya Godbole, Dilip Kavarthapu, Zhiyu Gong, Abhishek Singhal, Mo Yu, Xiaoxiao Guo, Tian Gao, Hamed Zamani, Manzil Zaheer, et al. 2019. Multi-step entity-centric information retrieval for multi-hop question answering. In _Proceedings of the 2nd Workshop on Machine Reading for Question Answering_ , pages 113–118. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ . 

- Ming Ding, Chang Zhou, Qibin Chen, Hongxia Yang, and Jie Tang. 2019. Cognitive graph for multihop reading comprehension at scale. _arXiv preprint arXiv:1905.05460_ . 

- Yuwei Fang, Siqi Sun, Zhe Gan, Rohit Pillai, Shuohang Wang, and Jingjing Liu. 2019. Hierarchical graph network for multi-hop question answering. _arXiv preprint arXiv:1911.03631_ . 

- Dirk Groeneveld, Tushar Khot, Ashish Sabharwal, et al. 2020. A simple yet strong pipeline for hotpotqa. _arXiv preprint arXiv:2004.06753_ . 

- Yichen Jiang and Mohit Bansal. 2019. Avoiding reasoning shortcuts: Adversarial evaluation, training, and model development for multi-hop qa. _arXiv preprint arXiv:1906.07132_ . 

- Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. _arXiv preprint arXiv:2004.04906_ . 

- Kyungjae Lee, Seung-won Hwang, Sang-eun Han, and Dohyeon Lee. 2021. Robustifying multi-hop qa through pseudo-evidentiality training. _arXiv preprint arXiv:2107.03242_ . 

- Shaobo Li, Xiaoguang Li, Lifeng Shang, Xin Jiang, Qun Liu, Chengjie Sun, Zhenzhou Ji, and Bingquan Liu. 2021. Hopretriever: Retrieve hops over wikipedia to answer complex questions. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 13279–13287. 

- Sewon Min, Eric Wallace, Sameer Singh, Matt Gardner, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2019a. Compositional questions do not necessitate multi-hop reasoning. _arXiv preprint arXiv:1906.02900_ . 

- Sewon Min, Victor Zhong, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2019b. Multi-hop reading comprehension through question decomposition and rescoring. _arXiv preprint arXiv:1906.02916_ . 

- Yixin Nie, Songhe Wang, and Mohit Bansal. 2019. Revealing the importance of semantic retrieval for machine reading at scale. _arXiv preprint arXiv:1909.08041_ . 

- Kosuke Nishida, Kyosuke Nishida, Masaaki Nagata, Atsushi Otsuka, Itsumi Saito, Hisako Asano, and Junji Tomita. 2019. Answering while summarizing: Multi-task learning for multi-hop qa with evidence extraction. _arXiv preprint arXiv:1905.08511_ . 

- Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. 2021. Counterfactual vqa: A cause-effect look at language bias. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12700– 12710. 

- Judea Pearl. 2022. Direct and indirect effects. In _Probabilistic and Causal Inference: The Works of Judea Pearl_ , pages 373–392. 

- Judea Pearl and Dana Mackenzie. 2018. _The book of why: the new science of cause and effect_ . Basic books. 

- Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, and Christopher D Manning. 2019. Answering complex open-domain questions through iterative query generation. _arXiv preprint arXiv:1910.07000_ . 

- Lin Qiu, Yunxuan Xiao, Yanru Qu, Hao Zhou, Lei Li, Weinan Zhang, and Yong Yu. 2019. Dynamically fused graph network for multi-hop reasoning. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 6140– 6150. 

- Yongming Rao, Guangyi Chen, Jiwen Lu, and Jie Zhou. 2021. Counterfactual attention learning for finegrained visual categorization and re-identification. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1025–1034. 

- Nan Shao, Yiming Cui, Ting Liu, Shijin Wang, and Guoping Hu. 2020. Is graph structure necessary for multi-hop question answering? _arXiv preprint arXiv:2004.03096_ . 

- Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2020. Is multihop qa in dire condition? measuring and reducing disconnected reasoning. _arXiv preprint arXiv:2005.00789_ . 

- Tan Wang, Chang Zhou, Qianru Sun, and Hanwang Zhang. 2021. Causal attention for unbiased visual recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 3091–3100. 

- Wenhan Xiong, Xiang Lorraine Li, Srini Iyer, Jingfei Du, Patrick Lewis, William Yang Wang, Yashar Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe Kiela, et al. 2020. Answering complex open-domain questions with multi-hop dense retrieval. _arXiv preprint arXiv:2009.12756_ . 

- Xu Yang, Hanwang Zhang, Guojun Qi, and Jianfei Cai. 2021. Causal attention for vision-language tasks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9847– 9857. 

- Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. _Advances in neural information processing systems_ , 32. 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. _arXiv preprint arXiv:1809.09600_ . 

- Xi Ye, Rohan Nair, and Greg Durrett. 2021. Connecting attributions and qa model behavior on realistic counterfactuals. _arXiv preprint arXiv:2104.04515_ . 

- Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_ , 33:17283–17297. 

- Chen Zhao, Chenyan Xiong, Corby Rosset, Xia Song, Paul Bennett, and Saurabh Tiwary. 2020. Transformer-xh: Multi-evidence reasoning with extra hop attention. 

- Beier Zhu, Yulei Niu, Xian-Sheng Hua, and Hanwang Zhang. 2021a. Cross-domain empirical risk minimization for unbiased long-tailed classification. _arXiv preprint arXiv:2112.14380_ . 

- Yunchang Zhu, Liang Pang, Yanyan Lan, Huawei Shen, and Xueqi Cheng. 2021b. Adaptive information seeking for open-domain question answering. _arXiv preprint arXiv:2109.06747_ . 

## **A Appendix** 

## **A.1 Implementation Details** 

Specifically, given the question _Q_ = _q_ , and context _P_ = _{s_ 1 _, s_ 2 _, ..., sm}_ where _m_ is the number of paragraphs, we denote the remaining context _ci_ = _P −{si}_ , 1 _≤ i ≤ m_ . To distinguish whether 

_si_ is supporting fact and get the answer distribution on _si_ , we construct the _s[∗] i_[and] _[c][∗] i_[as][illustrated] in the subsection 4.1. We respectively encode ( _q, si, ci_ ),( _q, s[∗] i[, c][i]_[)][and][(] _[q, s][i][, c][∗] i_[)][to][get][the][con-] textualized representation _O ∈ R[n][×][d] , Mi ∈[n][×][d] , Gi ∈ R[n][×][d]_ , where _n_ is the length of the question and context. 

For supporting facts identification, similar to Dire (Trivedi et al., 2020), we use the start token _s[start] i_ of the paragraph as its representation and obtain their predicted logits under factual and counterfactual scenario: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0010-15.png)


where _g_ is a classifier instantiated as _MLP_ layer in practice. And _s[start] i_ and _s[end] i_ are denoted as the start and end position of the paragraph _si_ respectively. In sentence level, we use the their start positions in paragraph _si_ and operate in the same way. 

As for answer span prediction, we concatenate the representation of _si_ in _Mi_ or _Gi_ (1 _≤ i ≤ m_ ) to construct the entire answer span prediction on the whole context: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0010-18.png)


(14) 

where [; ] denotes the operation of concatenation. And we can predict the end position of the answer in the same way. 

We also apply our counterfactual reasoning method to identify the answer type, consisting of yes, no, span and entity: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning_images/acl_2023_Counterfactual_Multihop_QA_A_Cause-Effect_Approach_for_Reducing_Disconnected_Reasoning.pdf-0010-22.png)


We use the [ _CLS_ ] token as the global representation to predict the answer type, following previous work (Qiu et al., 2019; Fang et al., 2019). 

