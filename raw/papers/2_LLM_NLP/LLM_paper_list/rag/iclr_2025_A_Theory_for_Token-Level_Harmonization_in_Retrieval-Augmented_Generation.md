## A THEORY FOR TOKEN-LEVEL HARMONIZATION IN RETRIEVAL-AUGMENTED GENERATION 

**Shicheng Xu Liang Pang** _[∗]_ **Huawei Shen Xueqi Cheng** CAS Key Laboratory of AI Safety, Institute of Computing Technology, CAS {xushicheng21s,pangliang,shenhuawei,cxq}@ict.ac.cn 

## ABSTRACT 

Retrieval-augmented generation (RAG) utilizes retrieved texts to enhance large language models (LLMs). Studies show that while RAG provides valuable external information (benefit), it may also mislead LLMs (detriment) with noisy or incorrect retrieved texts. Although many existing methods attempt to preserve benefit and avoid detriment, they lack a theoretical explanation for RAG. The benefit and detriment in the next token prediction of RAG remain a ’black box’ that cannot be quantified or compared in an explainable manner, so existing methods are datadriven, need additional utility evaluators or post-hoc. This paper takes the first step towards providing a theory to explain and trade off the benefit and detriment in RAG. First, we model RAG as the fusion between distribution of LLM’s knowledge and distribution of retrieved texts. Then, we formalize the trade-off between the value of external knowledge (benefit) and its potential risk of misleading LLMs (detriment) in next token prediction of RAG by distribution difference in this fusion. Finally, we prove that the actual effect of RAG on the token, which is the comparison between benefit and detriment, can be predicted without any training or accessing the utility of retrieval. Based on our theory, we propose a practical novel method, **Tok-RAG** , which achieves collaborative generation between the pure LLM and RAG at token level to preserve benefit and avoid detriment. Experiments in real-world tasks using LLMs such as OPT, LLaMA-2, and Mistral show the effectiveness of our method and support our theoretical findings. 

## 1 INTRODUCTION 

Retrieval-augmented generation (RAG) has shown promising performance in enhancing Large Language Models (LLMs) by integrating retrieved texts (Xu et al., 2023; Shi et al., 2023; Asai et al., 2023; Ram et al., 2023). Studies indicate that while RAG provides LLMs with valuable additional knowledge ( **benefit** ), it also poses a risk of misleading them ( **detriment** ) due to noisy or incorrect retrieved texts (Ram et al., 2023; Xu et al., 2024b;a; Jin et al., 2024a; Xie et al., 2023; Jin et al., 2024b). Existing methods attempt to preserve benefit and avoid detriment by adding utility evaluators for retrieval, prompt engineering, or fine-tuning LLMs (Asai et al., 2023; Ding et al., 2024; Xu et al., 2024b; Yoran et al., 2024; Ren et al., 2023; Feng et al., 2023; Mallen et al., 2022; Jiang et al., 2023). However, existing methods are data-driven, need evaluator for utility of retrieved texts or post-hoc. A theory-based method, focusing on core principles of RAG is urgently needed, which is crucial for consistent and reliable improvements without relying on additional training or utility evaluators and improving our understanding for RAG. 

This paper takes the first step in providing a theoretical framework to explain and trade off the benefit and detriment at token level in RAG and proposes a novel method to preserve benefit and avoid detriment based on our theoretical findings. Specifically, this paper pioneers in modeling next token prediction in RAG as the fusion between the distribution of LLM’s knowledge and the distribution of retrieved texts as shown in Figure 1. Our theoretical derivation based on this formalizes the core of this fusion as the **subtraction** between two terms measured by the distribution difference: one is **distribution completion** and the other is **distribution contradiction** . Further analysis indicates that the distribution completion measures how much out-of-distribution knowledge that retrieved texts 

> _∗_ Corresponding author 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Pure LLM: Query:  Who was the first Nigerian to  win the Nobel Prize, in which year?<br>Query LLM’s  Ernst LLM’s  Olanipekun … LLM’s  1986<br>LLM Distribution Distribution Distribution<br>RAG: RetrievedTexts Retrieved  DistributionDifference × Retrieved  DistributionDifference × Retrieved  DistributionDifference √<br>Distribution  Distribution  Distribution<br>√ √ ×<br>Query Wole Soyinka … 1980<br>Fusion Fusion Fusion<br>LLM’s  LLM’s  LLM’s<br>Distribution Distribution Distribution<br>LLM<br>**----- End of picture text -----**<br>


Figure 1: Framework of our Tok-RAG. It performs collaborative generation between pure LLM and RAG at the token-level by comparing benefit and detriment based on our theoretical findings about distribution difference. The selected tokens at each step are used as the prefix for both pure LLM and RAG. Tok-RAG preserves benefit and avoids detriment without any training or utility evaluators. 

provide to LLM in the prediction of the next token, representing the **benefit** of RAG for LLM at the token level. However, since the retrieved texts may contain noisy or incorrect information, posing the risk of misleading the LLM, distribution contradiction captures this risk by measuring the degree of conflict between the LLM’s knowledge and the external knowledge from the retrieved texts. This represents the **detriment** of RAG for LLM at token level. Thus, we prove that the fusion between distribution of LLM’s knowledge and the retrieved texts in next token prediction of RAG is governed by the combined effect of benefit and detriment, whose relationship is described as subtraction. 

In this way, we successfully decouple benefit and detriment from next token prediction of RAG and formalize them as the subtraction between two terms. This subtraction describes the trade-off between the value of external knowledge and its potential risk of misleading LLM without accessing the utility of the retrieved texts. We then prove that value of this subtraction is approximately positively correlated with the similarity between the representation of RAG’s output and the representation of retrieval texts. Finally, we establish a theory to predict the actual effect of RAG at token level. Specifically, in next token prediction given prefix and retrieved texts, this theory determines whether the benefit brought by retrieved texts on the prediction of the token outweighs the detriment. 

Based on our theoretical results, we propose a practical novel method called **Tok-RAG** that can achieve collaborative generation between pure LLM and RAG at token level to preserve benefit and avoid detriment **without any training or additional modules for utility evaluation of retrieved texts** . As shown in Figure 1, pure LLM and RAG generate the texts in parallel. At the generation step where LLM and RAG generate the different tokens, **Tok-RAG** uses our theoretical results to determine which token will be selected by comparing the values of benefit and detriment brought by RAG to the token. Experimental results in real-world tasks such as Q&A and Long-Form Q&A based on LLMs including OPT, LLaMA-2, and Mistral show the effectiveness of our method and support our theoretical results. Our method does not need any additional modules or training but outperforms baselines that need additional modules and fine-tuning LLMs, which indicates that our theoretical results are essential and fundamental for RAG. The main contributions of this paper are: 

_•_ This paper takes the first step in theoretically giving the essential explanation of benefit and detriment in RAG to make them explainable and comparable at token-level. 

_•_ We model RAG as distribution fusion and formalize the core of this fusion as the subtraction between distribution completion and distribution contradiction, our analysis shows the former is benefit and the latter is detriment for next token prediction of RAG. This subtraction describes the trade-off between the value of external knowledge and its potential risk of misleading LLM. 

_•_ We provide a theory that actual effect of RAG (i.e., the comparison between benefit and detriment) can be predicted at token level by representation similarity without any training or accessing the utility of the retrieved texts. 

_•_ Based on the theoretical results, we propose a practical novel method to enable pure LLM and RAG to collaboratively generate at token level. Experimental results on real-world tasks across different LLMs show the effectiveness of our method and support our theoretical results. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Equation 9 𝑝𝑝𝑅𝑅(𝑥𝑥𝑖𝑖|𝑥𝑥1:𝑖𝑖−1) 𝑝𝑝(𝑥𝑥𝑖𝑖|𝑥𝑥1:𝑖𝑖−1) 𝑝𝑝(𝑥𝑥𝑖𝑖|𝑅𝑅, 𝑥𝑥1:𝑖𝑖−1) Corollary 3: Actual effect of RAG can be predicted at token-level<br>Corollary1: Distribution difference  Corollary 2: Benefit and detriment is<br>formalizes benefit and detriment in RAG related to representation similarity Actual Effect Actual Effect  =  Benefit  −  Detriment<br>Distribution Distribution<br>Distribution Retrieved  Difference Completion: Benefit Theorem 1, 2 Theorem 1:  𝒢𝒢(Benefit, Detriment)  Upper Bound Theorem 2, 3 Sim(     ,     )  Benefit ==Sim(     ,     )   Detriment Sim(     ,     )  Benefit >>Sim(     ,     )   Detriment<br>Bound Similarity(       ,        )<br>Lower Bound O Sim(     ,     )<br>Distribution 𝒬𝒬(Benefit, Detriment)<br>Fusion Contradiction:<br>LLM’s  Detriment Theorem 2: Positively correlated  Benefit < Detriment Sim() is the similarity between representations<br>Distribution (Benefit − Detriment) ∝ Similarity(    ,  )   Sim(     ,     )  <Sim(     ,     )   Effect is positively correlated with Sim(     ,     )<br>**----- End of picture text -----**<br>


Figure 2: Derivation path of our theory. Reference: Equation 9, Theorem 1, 2, 3 and Corollary 1, 2, 3. 

## 2 THEORY TO UNDERSTAND BENEFIT AND DETRIMENT AT TOKEN LEVEL 

RAG exhibits the duality, although retrieved texts can provide LLM with external knowledge (benefit), they also pose a risk of misleading LLM due to the noise within the retrieved content (detriment). This section provides the theoretical framework to explain and trade off benefit and detriment in RAG at token level. First, we propose to use latent variable model to model the next token prediction in RAG as the fusion between the distribution of LLM’s knowledge and the distribution of retrieved texts. Second, our theoretical derivation formalizes the core of this fusion as the subtraction between two terms measured by the distribution difference: one is **distribution completion** , and the other is **distribution contradiction** . Our further discussion shows that distribution completion is actually the benefit and distribution contradiction is detriment in the next token prediction of RAG. Last but not least, we provide a theory to predict the actual effect of RAG (i.e., the comparison between benefit and detriment) at token level. Figure 2 is the derivation path of our theory. 

## 2.1 MODEL NEXT TOKEN PREDICTION IN RAG AS DISTRIBUTION FUSION 

To begin the analysis, inspired by previous studies that prove LLMs implicitly perform latent variable inference (Xie et al., 2021; Zhang et al., 2023; Wang et al., 2024), we first propose to analyse RAG by latent variable model to model the next token prediction in RAG as the fusion between the distribution of LLM’s knowledge and the distribution of retrieved texts. Specifically, given the prefix _x_ 1: _i−_ 1 = _{x_ 1 _, x_ 2 _, ...xi−_ 1 _}_ generated from time step 1 to _i −_ 1, from the perspective of the latent 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0003-06.png)


in which _Z_ is the space of high dimensional concept variable, _p_ ( _z|x_ 1: _i−_ 1) is the probability that the model samples latent concept _z_ from _Z_ given the prefix _x_ 1: _i−_ 1, and _p_ ( _xi|x_ 1: _i−_ 1 _, z_ ) means the probability for token _xi_ conditioned on the prefix _x_ 1: _i−_ 1 and the sampled latent concept _z_ . _p_ ( _xi|x_ 1: _i−_ 1) can be obtained by integrating over all latent concepts from the space _Z_ . Latent variable model has been applied in many methods such as LDA (Blei et al., 2003). Recent studies prove that in-context learning of LLMs can also be seen as the latent variable model, in which the LLMs sample the concept across the input examples (Xie et al., 2021; Zhang et al., 2023). Inspired by this, we analyse the next token prediction of RAG given prefix _x_ 1: _i−_ 1 as sampling the shared _Retrieved Concept z[∗]_ from the input retrieved texts list _R_ = _{r_ 1 _, r_ 2 _, ..., rn}_ ( _ri_ is a retrieved passage), and then predicting _p_ ( _xi|R, x_ 1: _i−_ 1), which can be formalized as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0003-08.png)


Equation 2 describes RAG as distribution fusion. The first term is the prediction that marginalizes out all latent concepts except _z[∗]_ , which is the knowledge distribution in LLMs. The second term is the prediction that is only conditioned on _z[∗]_ , which is the knowledge distribution from retrieved texts. 

## 2.2 FORMALIZE AND EXPLAIN BENEFIT AND DETRIMENT BY DISTRIBUTION DIFFERENCE 

In this section, we make further derivation on the distribution fusion in Equation 2 to formalize the core of this fusion as the subtraction between two terms measured by the distribution difference: one 

3 

is distribution completion, and the other is distribution contradiction. Our analysis reveals that for the next token prediction, distribution completion is actually the benefit and distribution contradiction is the detriment. Specifically, inspired by (Xie et al., 2021), the Equation 2 can be transformed as (detailed proof can be found in Appendix A): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0004-01.png)


Define _ri_ is a passage in the retrieved texts list _R_ , we can get (see detailed proof in Appendix B): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0004-03.png)


_pR_ ( _·_ ) is the distribution of the retrieved texts, _p_ ( _·_ ) is the distribution of the LLM’s knowledge. _v_ ( _z_ ) is an important term in distribution fusion because it reflects the proportion between the latent concept from the space of LLMs and from the retrieved texts. Details are in Appendix C. 

**Discuss the Benefit and Detriment Based on the Theoretical Results.** In Equation 9, the first term represents the distribution difference (KL divergence) between retrieved texts ( _pR_ ( _r_ )) and LLM’s knowledge ( _p_ ( _r|z_ )) given the concept _z_ , which is sampled from _Z_ (latent variables in LLM). This can be defined as **distribution completion** that measures how much out-of-distribution knowledge that retrieved texts provide to LLM in the prediction of the token _xi_ . This term is actually the **benefit** for the prediction of token _xi_ in RAG. This is because that if the retrieved text _r_ is perfect, _pR_ ( _r_ ) would be infinitely close to the ground-truth distribution. The larger the difference between _pR_ ( _r_ ) and _p_ ( _r|z_ ), the more the knowledge distribution about _xi_ in the LLM’s knowledge deviates from the ground truth. This means the retrieved texts can provide more valuable out-of-distribution knowledge to the LLM to predict _xi_ , resulting in greater benefit. 

Considering that the retrieved texts are not always perfect and may contain incorrect information and noise that contradict the correct knowledge of LLM, the second term corresponds to the **distribution contradiction** can be used to measure this risk. It represents the distribution difference (KL divergence) between the retrieved texts ( _pR_ ( _r_ )) and LLM’s knowledge given the concept _z[∗]_ , which is sampled from retrieved texts ( _p_ ( _r|z[∗]_ )). _p_ ( _r|z[∗]_ ) is the prediction made by LLM conditioned on the concept _z[∗]_ sampled from the retrieved texts. If the external knowledge in the retrieved texts contradicts LLM’s knowledge, _p_ ( _r|z[∗]_ ) will deviate from _pR_ ( _r_ ) (the actual distribution of the retrieved texts). Therefore, the difference between _p_ ( _r|z[∗]_ ) and _pR_ ( _r_ ) primarily stems from the LLM’s resistance to any external knowledge in the retrieved texts that conflicts with LLM’s knowledge. The larger difference indicates the stronger resistance from LLM, and the more confident the LLM is in its pre-trained knowledge, which means the greater the potential detriment caused by the retrieved texts. So this term is actually the **detriment** for the prediction of token _xi_ in RAG. 

**Corollary 1.** _Two terms about distribution difference in Equation 9 measure the benefit and detriment respectively. The subtraction between benefit and detriment describes the trade-off relationship between the value of external knowledge and its potential risk of misleading LLM in the next token prediction without accessing the utility of retrieved texts._ 

4 

## 2.3 ACTUAL EFFECT OF RAG CAN BE PREDICTED AT TOKEN LEVEL 

Based on above analysis, we successfully formalize the benefit and detriment in next token prediction of RAG by measuring distribution difference. Next, we further explore the method to compare the values of benefit and detriment. Specifically, we derive Theorem 1 and Theorem 2 from Equation 2: 

**Theorem 1.** _Define D_ = _∥p_ ( _xi|R, x_ 1: _i−_ 1) _− pR_ ( _xi|x_ 1: _i−_ 1) _∥_ 1 _to measure the difference between the distribution of RAG (p_ ( _xi|R, x_ 1: _i−_ 1) _) and the distribution of retrieved texts (pR_ ( _xi|x_ 1: _i−_ 1) _) in token prediction of xi conditioned on prefix x_ 1: _i−_ 1 _. Both benefit and detriment are important terms of the upper and lower bounds of D, which can be described as:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0005-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0005-04.png)


Our detailed proof of Theorem 2 is in Appendix E. 

**Corollary 2.** _The difference between values of benefit and detriment in Equation 11 indicates the extent to which the benefit (value of external knowledge) outweighs the detriment (potential risk of misleading LLM) in the prediction of token xi. This difference is approximately positively correlated with the representation similarity, which is the value that can be predicted._ 

Recapping our motivation that aims to build a theory to predict whether the positive impact of the retrieved texts _R_ on _xi_ ( **benefit** ) outweighs the potential risk of misleading the LLM ( **detriment** ). The key for this is comparing the values of benefit and detriment. As stated in Theorem 2, the result of benefit minus detriment is approximately positively correlated with _D_[1][.][Therefore, the value of] _D_[1] at which the benefit minus detriment equals zero is an important threshold. A _D_[1][value greater than] this threshold indicates that benefit exceeds detriment, while a value less than this threshold indicates that detriment outweighs benefit. We derive Theorem 3 to identify this dividing point and map the value order between benefit and detriment of token _xi_ to the relationship between representation similarity, which can be calculated in practical applications: 

**Theorem 3.** _Define M_ = _∥p_ ( _xi|R, x_ 1: _i−_ 1) _− p_ ( _xi|x_ 1: _i−_ 1) _∥_ 1 _to measure the difference between_ 1 _distribution of RAG (p_ ( _xi|R, x_ 1: _i−_ 1) _) and pure LLM (p_ ( _xi|x_ 1: _i−_ 1) _), so M[can][be][treated][as][the] similarity between them. D_[1][=] _M_ 1 _[is the dividing point in which benefit is equal to detriment, and the] value order between_[1] 1 _D[and] M[can indicate the value order between benefit and detriment as:]_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ )) _<_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )) _, detriment outweighs benefit. if D_[1] _[<] M_ 1 _J_ = KLKL(( _ppRR_ (( _rr_ )) _∥∥pp_ (( _rr||zz_ )) =)) _>_ KL KL(( _ppRR_ (( _rr_ )) _∥∥pp_ (( _rr||zz[∗][∗]_ )))) _, detriment is equal to benefit., benefit outweighs detriment. ifif DD_[1][1][=] _[>] MM_ 11 (12) 

Our detailed proof of Theorem 3 can be found in Appendix F. Equation 12 is a novel principle that can compare the values of benefit and detriment in RAG at token level. It does not rely on additional modules to access the utility of retrieved texts or training but simply compares _D_[1][and] _M_ 1[.] 

**Corollary 3.** _The actual effect of RAG, i.e., the comparison between benefit and detriment, can be predicted at token level by the similarity relationships between p_ ( _xi|R, x_ 1: _i−_ 1) _, p_ ( _xi|x_ 1: _i−_ 1) _, and pR_ ( _xi|x_ 1: _i−_ 1) _, without additional modules to access the utility of retrieved texts._ 

The next section introduces how to apply above theoretical findings to improve RAG in practice. 

## 3 TOK-RAG: IMPROVE RAG BASED ON TOKEN-LEVEL THEORY 

Tok-RAG is a novel method that enables the LLM and RAG to collaborate at token level for generation to preserve benefit and avoid detriment based on our token-level theory. Tok-RAG makes pure LLM 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0006-00.png)


**----- Start of picture text -----**<br>
le-5<br>stage1 stage2 stage3 1.6<br>1.4<br>0.8<br>1.2<br>1.0<br>0.6<br>0.8<br>0.4 0.6<br>0.4<br>0.2 0.2<br>0.0<br>0 5 10 15 20 25 30<br>layer<br>Attention score<br>difference of word distribution change<br>**----- End of picture text -----**<br>


Figure 3: Attention score for _xi_ (blue line) and difference of word distribution change (yellow line) vary with layers. stage 1: Lexical and Syntactic. stage 2: Text Matching. stage 3: Distribution Fusion. 

and RAG generate in parallel at token level as shown in Figure 1. It determines which token will be selected by comparing the values of benefit and detriment brought by RAG to the token according to the size relationship between _D_[1][and] _M_ 1[in Equation 12.][The terms related to the comparison between] 1 1[(1)] _[ p]_[(] _[x][i][|][R, x]_[1:] _[i][−]_[1][)][ can be directly obtained from the prediction of] _D_[and] _M_[consist of three parts:] RAG; (2) _p_ ( _xi|x_ 1: _i−_ 1) can be directly obtained from the prediction of pure LLM; (3) **however, the distribution of retrieved texts conditioned on the prefix** _x_ 1: _i−_ 1 **,** _pR_ ( _xi|x_ 1: _i−_ 1) **, is hard to directly obtained, which is the main challenge that the following Section 3.1 aims to solve.** 

## 3.1 DISTRIBUTION PREDICTION FOR RETRIEVED TEXTS 

Our theoretical analysis in Section 2.1 shows that RAG can be modeled as fusing the distribution from retrieved texts with LLMs’ distribution. Therefore, an intuitive idea is that the retrieved distribution _pR_ ( _xi|x_ 1: _i−_ 1) can be approximately predicted by capturing the signal from the retrieved texts in distribution fusion. The main challenges in achieving this are: (1) determining where distribution fusion occurs, and (2) capturing the signal fused from retrieved texts and transforming it to distribution _pR_ ( _xi|x_ 1: _i−_ 1). To address these challenges, in the following parts, we first explore the operating mechanism of RAG and then propose a novel method that dynamically determines the layers where distribution fusion occurs and use the signal from retrieved texts in these layers as _pR_ ( _xi|x_ 1: _i−_ 1). 

**Exploring the Mechanism of RAG.** This part finds that the mechanism of RAG can be decomposed into two parts. The first is text matching, which means extracting information relevant to the generation of _xi_ from the retrieved texts _R_ . The second is distribution fusion, which means fusing the distribution from the retrieved texts with the distribution in LLM’s knowledge. When performing RAG, LLMs first perform text matching in the middle layers, extracting relevant knowledge from the retrieved texts. As the depth increases, the matching becomes increasingly accurate, reaching a turning point. In the deep layers after this turning point, LLMs instead carry out distribution fusion, and the attention shifts from _R_ to _x_ 1: _i−_ 1. Distribution of _R_ used for fusion comes from the matching information around the turning point (because matching decreases after the turning point). Recapping the two challenges introduced at the beginning of Section 3.1, for the first challenge, we identify the layer where distribution fusion starts by detecting the turning point in Figure 3. For the second challenge, we use the matching information in the layer where distribution fusion starts to approximate the distribution _pR_ ( _xi|x_ 1: _i−_ 1). Our experiments about these findings are introduced in **Exp 1.** and **Exp 2.** . Experiments are conducted based on LLaMA-2-7B on Natural Question dataset. We also perform these experiments using more LLMs (OPT-6.7B, Mistral-7B) on more datasets (TriviaQA, WebQ and Squad). The conclusions of them are consistent with LLaMA-2-7B on Natural Question, and detailed results can be found in Appendix H. 

**Exp 1.** For text matching, we quantify the relevance of the information in the retrieved texts to the generation of token _xi_ by the attention score between token _xi_ and the tokens in the retrieved texts _R_ . We analyze how the sum of attention scores from token _xi_ to tokens in _R_ varies across layer. As shown by the blue line in Figure 3: (1) The value increases sharply to a peak in shallow layers (0-5), which is mainly because LLMs capture the low-level lexical and syntactic information on the entire input (Tenney et al., 2019). (2) The value first decreases, then increases to a maximum point in the middle layers (5-23), which is mainly because LLMs select the relevant semantics that can be used to generate _xi_ from _R_ and complete this selection at the maximum point. (3) The value decreases after the maximum point in deep layers (24-32). This is because that LLMs use the selected knowledge at the maximum point for distribution fusion to predict _xi_ , the attention shifts from _R_ to prefix _x_ 1: _i−_ 1. 

6 

**Exp 2.** For distribution fusion, since distribution fusion is often marked by a change in word distribution (Bengio et al., 2000), we identify the occurrence of distribution fusion in RAG by comparing the change in word distribution between pure LLM and RAG. (Chuang et al., 2023; Schuster et al., 2022) prove the language heads can be directly applied to the hidden states, so we propose to obtain the word distribution of hidden states in each layer by language heads _ϕ_ as _ϕ_ ( _h[l] i_[)][,][in][which] _[h][l] i_[is][the][hidden][states][for][token] _[x][i]_[in][the] _[l]_[-th][layer.][We][then][measure][the][word] distribution change in the _l_ -th layer by Jensen-Shannon Divergence (JSD) between _ϕ_ ( _h[l] i[−]_[1] ) and _ϕ_ ( _h[l] i_[)] as: _C_ = JSD( _ϕ_ ( _h[l] i[−]_[1] ) _∥ϕ_ ( _h[l] i_[))][.][The difference of word distribution change between pure LLM and] RAG in the _l_ -th layer is described as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0007-01.png)


in which _h_[˜] _[l] i[−]_[1] and _h_[˜] _[l] i_[are from RAG,] _[ h][l] i[−]_[1] and _h[l] i_[are from pure LLM. The yellow line in Figure 3] shows _D[l]_ is very small in the shallow and middle layers (0-23) and rises sharply in the deep layers (24-32). This suggests that distribution fusion occurs in deep layers. 

**Dynamically Identify the Layer Where Distribution Fusion Starts.** For _p_ ( _xi|R, x_ 1: _i−_ 1), the layer where distribution fusion starts can be located by detecting the turning point in Figure 3. Specifically, we use _f_ ( _l_ ) to denote the attention score for _xi_ and _g_ ( _l_ ) to denote the difference of word distribution change in Equation 13 varies with layer _l_ . The layer where distribution fusion starts is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0007-04.png)


The first term is the _l_ that maximizes _f_ ( _l_ ), which is the third turning point in the blue line of Figure 3. The second term means the minimum _l_ value for which _g_ ( _l_ ) is greater than _a_ (hyperparameter, can be set to 5e-7 according to our statistics), which is the turning point in the yellow line of Figure 3. We take the average of the two values and round down as the layer _l[∗]_ where distribution fusion starts. 

**Matching as Distribution.** The matching information between _R_ = [ _rt_ 1 _, rt_ 2 _, ..., rtm_ ] ( _rt_ is the token in _R_ ) and token _xi_ at turning point (the _l[∗]_ -th layer) can be used to approximate the distribution _pR_ ( _xi|x_ 1: _i−_ 1) of the retrieved texts _R_ conditioned on _x_ 1: _i−_ 1. The matching information consists of two parts, one is the attention score, which can measure the matching between retrieved tokens and current token _xi_ at the hidden state level. The other is the similarity of word embeddings, which can measure the matching between retrieved tokens and current token _xi_ at the word distribution level: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0007-07.png)


_Wh_ ˜ _[l]_ 1: _[∗] qm_ and[are hidden states of] _Wk_ are matrices in attention (Vaswani et al., 2017), _[ R]_[.] _[A]_[ is word embedding matrix,] _[ x] i[l][′] h_[˜] _[−][l] i[∗][l][∗]_[is the hidden state of token] is the token with the largest logits _[ x][i]_[ and] increase in word distribution from layer _l[∗]_ to the final layer _l[′]_ , _rt[l]_ 1: _[∗] m_[are tokens in] _[ R]_[.][Then:] 

_pR_ ( _xi|x_ 1: _i−_ 1) = softmax ( _Att ⊙ WordSim_ ) _, ⊙_ is element-wise multiplication. (16) 

3.2 TOKEN-LEVEL COMPARISON BETWEEN BENEFIT AND DETRIMENT IN PRACTICE 

Equation 12 shows that the relationship between Sim( _p_ ( _xi|R, x_ 1: _i−_ 1) _, pR_ ( _xi|x_ 1: _i−_ 1)) and Sim( _p_ ( _xi|R, x_ 1: _i−_ 1) _, p_ ( _xi|x_ 1: _i−_ 1)) indicates the value order between benefit and detriment (Sim( _·, ·_ ) is the similarity). We propose to use the token semantics as the representation for _p_ ( _xi|R, x_ 1: _i−_ 1), _pR_ ( _xi|x_ 1: _i−_ 1) and _p_ ( _xi|x_ 1: _i−_ 1) and use cosine to compute the similarity. It not only follows the principle of Equation 12 but also takes into account the semantic similarity, which is more robust in practical applications. Specifically, we use word embedding matrix of LLMs to calculate the weighted average word embedding for _p_ ( _xi|R, x_ 1: _i−_ 1) as **w** _RAG_ = �1 _p′_ �( _p[′] ,_ **w** ) _∈_ V _[p][′]_ **[w]**[,][for each token in] vocabulary V, _p[′]_ is its logits from _p_ ( _xi|R, x_ 1: _i−_ 1) and **w** is its word embedding. We can also use this to get the weighted average word embedding **w** _LLM_ for _p_ ( _xi|x_ 1: _i−_ 1) and **w** _IR_ for _pR_ ( _xi|x_ 1: _i−_ 1). The similarity between them can be calculated via cosine similarity as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0007-12.png)


Combining our theoretical analysis of Theorem 1, 2 and 3, we can derive this principle to compare the values of benefit and detriment brought by RAG to the token _xi_ in practical applications: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0007-14.png)


7 

|Methods<br>Train<br>LLM<br>Add<br>Module|TriviaQA<br>Ratio of Hard Negative Passages<br>100% 80% 60% 40% 20% 0%|WebQ|Squad|
|---|---|---|---|
|||Ratio of Hard Negative Passages<br>100% 80% 60% 40% 20% 0%|Ratio of Hard Negative Passages<br>100% 80% 60% 40% 20% 0%|
|Standard RAG<br>no✔<br>no✔<br>NLI+RAG<br>no✔<br>need✗<br>CRAG<br>no✔<br>need✗<br>RetRobust<br>need✗no✔<br>Self-RAG<br>need✗no✔<br>INFO-RAG<br>need✗no✔<br>Tok-RAG(Ours) no✔<br>no✔|43.8<br>67.0 71.3 76.2 78.2 81.9 <br>50.8<br>61.2 68.2 73.0 76.4 79.1 <br>48.2<br>68.3 72.5 76.7 81.5 82.2 <br>49.2<br>67.3 72.9 77.5 79.4 82.3 <br>43.0<br>68.7 73.5 76.4 80.8 82.2 <br>49.7<br>68.4 73.2 77.9 80.0 82.5 <br>**53.5**<br>**72.9 77.6 81.3 83.4 85.7 **|23.9<br>35.8 40.6 43.4 48.4 53.1 <br> 30.7<br>40.3 44.5 47.5 50.9 52.8 <br> 25.6<br>37.4 41.9 46.2 51.5 54.9 <br> 30.0<br>38.9 42.5 48.2 49.8 54.3 <br> 18.3<br>34.8 42.2 47.2 51.3 57.0 <br> 29.7<br>38.0 43.9 48.1 49.4 54.8 <br> **32.9**<br>**43.8 47.3 50.0 52.9 57.3 **|8.6<br>31.0 43.2 53.0 58.8 67.2<br> 9.9<br>21.1 33.7 43.4 51.7 60.5<br> 7.4<br>28.7 39.6 50.7 53.2 61.1<br> 10.5<br>30.8 43.3 52.5 58.4 66.0<br> 5.5<br>27.8 38.9 46.4 52.5 58.3<br> 10.7<br>30.1 43.5 53.7 59.2 67.5<br> **12.8**<br>**31.3 44.5 54.1 60.8 68.1**|



Table 1: Accuracy on open-domain Q&A given the retrieved texts containing different ratios (0% to 100%) of hard negative passages (irrelevant but are ranked in top-10 by retrieval model). Our Tok-RAG does not need any training or additional modules while baselines need. 

## 4 EXPERIMENTS 

## 4.1 EXPERIMENTAL DETAILS 

**Practical RAG for Open-domain Q&A.** One experiment is in a practical autoregressive generation setting for open-domain Q&A given retrieved texts with different qualities, it evaluates the robustness and performance of RAG. Baselines include the methods that use additional modules to filter irrelevant texts ( **NLI+RAG** (Yoran et al., 2024)) or as action triggers ( **CRAG** (Yan et al., 2024)), fine-tune LLMs for robust RAG ( **RetRobust** (Yoran et al., 2024) and **INFO-RAG** (Xu et al., 2024b)) and fine-tune LLMs to dynamically retrieve and critique retrieved texts ( **Self-RAG** (Asai et al., 2023)). We follow baselines to use Cover-EM (Rosset et al., 2020) that indicates the accuracy as the metric. 

**Setup for Benefit-Detriment Comparison Experiment.** The other is benefit-detriment comparison experiment. Given prefix _x_ 1: _i−_ 1 and retrieved texts _R_ , our motivation aims to build a theory to predict the actual effect of RAG on the prediction of _xi_ , i.e., whether the positive impact of the retrieved texts _R_ on _xi_ ( **benefit** ) outweighs the potential risk of misleading LLM ( **detriment** ). This is a binary classification task at token-level. To evaluate this task, we construct test data and ground-truth as: (1) For a sentence _x_ , we truncate it at the _i_ -th token to obtain the prefix _x_ 1: _i−_ 1 and the next token _xi_ . (2) We input the prefix _x_ 1: _i−_ 1 to LLM w/o RAG and LLM w/ RAG to get the next predicted token _a_ and _b_ respectively. 

(3) If _b_ is _xi_ but _a_ not, it means LLM w/ RAG performs better than LLM w/o RAG, so the benefit is greater than detriment, the ground-truth label is 1. 

(4) If _a_ is _xi_ but _b_ not, it means LLM w/o RAG performs better than LLM w/ RAG, so the benefit is lower than detriment, the ground-truth label is 0. 

We use the above method to traverse all sentences in the datasets to obtain prefix-ground-truth pairs as samples. In evaluation, we input prefix to LLM w/ RAG and use our theoretical findings to judge whether the benefit of the next predicted token is greater than detriment. We use AUC and F1-score as the evaluation metrics for this binary classification task. 

**Baselines for Benefit-Detriment Comparison Experiment.** Task in benefit-detriment comparison experiment can also be viewed as predicting the correctness of the generated tokens. Therefore, baselines for this are the methods that detect the LLMs’ hallucination. We use these baselines to compare the values of benefit and detriment by comparing the hallucination degree at token level between RAG and pure LLM (details in Appendix I.1). Baselines include: (1) **Logprobs-based** (Kuhn et al., 2023), we use the value order between top-1 log-probability of the tokens output by pure LLM and RAG to determine the value order between benefit and detriment. (2) **Uncertainty-based** , we use Length-normalized Entropy (Malinin & Gales, 2020) to measure the uncertainty of the tokens and compare it between pure LLM and RAG. (3) **Consistency-based** , we run LLMs multiple times and calculate consistency scores among multiple answers using Lexical and Semantic Similarity (Lin et al., 2022; Chen et al., 2024). We compare these scores between pure LLM and RAG to indicate the comparison between benefit and detriment. 

**Datasets.** For the open-domain Q&A experiment, since long-form tasks are not conducive to objectively and accurately evaluating the factual correctness of the answers, we use three short-form Q&A tasks including WebQuestions ( **WebQ** ) (Berant et al., 2013), **TriviaQA** (Joshi et al., 2017) and **SQuAD v1.1** (Rajpurkar et al., 2016). For the token level binary classification task in benefit- 

8 

|LLMs<br>Methods<br># Generation<br>Times|Wikitext<br>AUC<br>F1|ASQA|Bio|NQ|
|---|---|---|---|---|
|||AUC<br>F1|AUC<br>F1|AUC<br>F1|
|OPT-6.7B<br>Logprobs<br>2<br>Uncertainty<br>2<br>Consistency-Lexical<br>10<br>Consistency-Semantic<br>10<br>Tok-RAG(Ours)<br>2|65.25<br>64.33<br>64.12<br>63.50<br>64.01<br>62.17<br>65.93<br>64.22<br>**68.64**+ **66.88**+|68.96<br>67.55<br>66.14<br>63.96<br>69.42<br>67.04<br>70.11<br>69.50<br> **72.28**+ **72.05**+|65.24<br>64.59<br>65.78<br>64.60<br>65.41<br>65.28<br>65.76<br>64.37<br> **66.27**+ **66.04**+|55.31<br>51.41<br>56.03<br>52.15<br>55.06<br>51.13<br>56.24<br>52.88<br> **57.92**+ **52.90**+|
|Mistral-7B<br>Logprobs<br>2<br>Uncertainty<br>2<br>Consistency-Lexical<br>10<br>Consistency-Semantic<br>10<br>Tok-RAG(Ours)<br>2|73.52<br>72.90<br>73.72<br>72.71<br>72.15<br>70.44<br>73.98<br>72.26<br>**75.85**+ **74.11**+|68.05<br>66.86<br>67.47<br>65.63<br>69.16<br>67.33<br>70.05<br>69.54<br> **71.51**+ **71.47**+|65.22<br>64.39<br>65.59<br>65.83<br>64.79<br>64.33<br>65.68<br>65.12<br> **66.37**+ **66.04**+|57.04<br>57.23<br>57.19<br>57.10<br>56.95<br>54.37<br>57.43<br>56.12<br> **58.52**+ **57.56**+|
|LLaMA-2-7B<br>Logprobs<br>2<br>Uncertainty<br>2<br>Consistency-Lexical<br>10<br>Consistency-Semantic<br>10<br>Tok-RAG(Ours)<br>2|73.47<br>72.95<br>73.98<br>73.01<br>73.51<br>71.62<br>74.96<br>74.23<br>**81.89**+ **80.42**+|68.50<br>68.04<br>68.72<br>67.63<br>70.09<br>68.45<br>71.23<br>69.38<br> **76.96**+ **76.80**+|62.11<br>60.94<br>63.67<br>63.50<br>62.49<br>61.98<br>63.77<br>62.10<br> **64.08**+ **64.19**+|67.40<br>69.24<br>68.03<br>69.15<br>68.17<br>70.09<br>69.72<br>71.14<br> **70.50**+ **72.45**+|



Table 3: Performance on comparing the valurs of benefit and detriment at token level. Significant test with p-value _≤_ 0 _._ 05 compared with all baselines are denoted as ‘+’. Setup in this experiment is introduced in **Setup for Benefit-Detriment Comparison Experiment** in Section 4.1. 

detriment comparison experiment, we use three long-form generation tasks including long-form Q&A ( **ASQA** (Stelmakh et al., 2023)), people biographies generation ( **Bio** (Min et al., 2023)) and language modeling ( **Wikitext103** (Merity et al., 2016)) and one short-form task includes Q&A ( **Natural Questions** (Kwiatkowski et al., 2019)). 

**Implementation details.** As for retrieval in RAG, we follow (Xu et al., 2023) to use ColBERTv2 (Santhanam et al., 2021), an excellent generalizable model as the retriever, and use Wikipedia consisting of 21,015,324 passages (Karpukhin et al., 2020) as retrieval database. **All baselines and Tok-RAG share the same retrieval setup and input.** We use OPT-6.7B, LLaMA-2-7B, and Mistral-7B-v0.1 as LLMs in the benefit-detriment comparison experiment and use greedy-decoding strategy for generation. More details of **Tok-RAG** and baselines are in Appendix I. 

## 4.2 EXPERIMENTAL RESULTS 

**Experiment on Open-domain Q&A.** This experiment is under the practical autoregressive generation setting for open-domain Q&A. Table 1 shows that in RAG given the retrieved texts with various qualities, our Tok-RAG shows better robustness and performance than baselines. Tok-RAG does not need any additional modules or training and outperforms the strong baselines that need additional modules to evaluate the utility of retrieved texts or training LLMs. This means our Tok-RAG achieves a better trade-off between benefit and detriment in RAG, avoiding detriment while securing benefit. It is because our theoretical analysis helps us propose a more fundamental method in comparing benefit and detriment at token level, while baselines cannot achieve. In this experiment, we adjust the radio of irrelevant passages in the retrieved passage list from 0% to 100%, which can simulate the degree of noise in the retrieved texts. Since Open-domain Q&A is a short-form task, we calculate the accuracy by judging whether the ground truth appears exactly in the generated texts (Cover-EM (Rosset et al., **Method GPU (GB) ↓ Time (s) ↓ Per. (Acc)** 2020)). LLM in this is LLaMA-2-7B. Standard 17.50 1.95 51.96 

|**Method**<br>Standard|**GPU(GB) **<br>17.50|**↓Time(s) **<br>1.95|**↓Per. (Acc) ↑**<br>51.96|
|---|---|---|---|
|NLI+RAG|20.25|2.03|49.76|
|CRAG|20.60|2.15|51.53|
|RetRobust<br>Self-RAG<br>INFO-RAG<br>Tok-RAG|17.50<br>17.50<br>17.50<br>23.98|2.00<br>2.98<br>2.10<br>2.24|52.98<br>50.26<br>52.35<br>**56.12**|



**Experiment on Benefit-Detriment Comparison.** Table 3 shows that our Tok-RAG achieves better performance in comparing the values of benefit and detriment at token level in RAG than baselines across different tasks and LLMs. Baselines compare the benefit and detriment by detecting the degree of hallucination, while our Tok-RAG can directly compare the benefit and detriment based on our theoretical analysis, which is more fundamental so it performs better. The detailed setup for this experiment can be found in **Setup for Benefit-Detriment Comparison Experiment** in Section 4.1. 

Table 2: Comparison of methods based on GPU Memory, Time, and Performance. 

9 

**Analysis on Computational Costs.** Table 2 shows that out Tok-RAG achieves significant performance (accuracy) improvement with little increase of GPU memory and running time. In addition, the baselines all require training LLM or introducing additional modules, while our method is based on our solid theoretical findings, dose not require training LLM nor additional modules. The parallel generation at the token-level can be done in a batch with 2 batch size. In actual practice, this does not bring significant computational time or GPU memory overhead. Experiments are performed on three Q&A datasets (TrviaQA, WebQ, Squad) with V100 GPU, the LLM is LLaMA-2-7B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0010-01.png)


**----- Start of picture text -----**<br>
0.8<br>0.7<br>0.6<br>Wikitext103<br>0.5 ASQA<br>Bio<br>NQ<br>0 5 10 15 20 25 30<br>Layer<br>AUC<br>**----- End of picture text -----**<br>


Figure 4: AUC varies with layer. 

**Ablation study.** Figure 4 shows the effectiveness of our dynamic layer selection strategy in Equation 14 and supports our finding that RAG performs matching in middle layers. Figure 4 shows the AUC when _l[∗]_ in Equation 14 is set as a fixed value from 0 to 32. Our dynamic layer selection strategy (dashed line) is always better than any fixed layers (solid line). Besides, AUC is higher in middle layers, which supports that RAG performs matching in middle layers and the knowledge in retrieved texts is extracted in the turning point. After the turning point, LLMs instead perform distribution fusion, the matching cannot reflect the distribution of retrieved texts, so AUC decreases. LLM used in this experiment is LLaMA-2-7B. 

**Case study.** Case study of Figure 5 in Appendix J intuitively shows the collaborative generation between pure LLM and RAG in our **Tok-RAG** . **Tok-RAG** is effective to preserve benefit and avoid detriment at token level by dynamically selecting suitable tokens among pure LLM and RAG. 

## 5 RELATED WORK 

**Robust RAG.** To make LLMs robust in RAG to avoid the detriment caused from noisy in retrieved texts, some methods use additional modules to filter out irrelevant documents (Yoran et al., 2024; Yan et al., 2024). Some methods train LLMs to make them robust to noisy in retrieved texts (Xu et al., 2024b; Yoran et al., 2024). Some methods let LLMs dynamically determine whether the query needs RAG (Asai et al., 2023; Xu et al., 2023; Ren et al., 2023; Feng et al., 2023; Mallen et al., 2022; Jiang et al., 2023). All the previous works solve the contradiction between benefit and detriment in RAG from the perspective of application but lacking essential and theoretical analysis, which limits the understanding and cannot find the fundamental method to solve it. Therefore, they rely on additional modules or fine-tuning LLMs. Our paper explains the benefit and detriment in RAG by theoretical analysis and proposes a novel method to preserve benefit while avoiding detriment without any additional modules or training. 

**Theoretical analysis of ICL.** Our paper is inspired by theoretical analysis of ICL. Some works explain ICL as one-step gradient descent (Von Oswald et al., 2023; Akyürek et al., 2022; Dai et al., 2022). Besides, there are other explanations of ICL such as Bayes inference (Xie et al., 2021), Bayes model averaging (Zhang et al., 2023), leaning topic structure (Li et al., 2023) and kernel regression (Han et al., 2023). They focus on explaining why ICL occurs. Our contribution lies in analyzing the benefit and detriment in RAG and proposing a practical method based on our theory. 

## 6 CONCLUSIONS AND DISCUSSION 

This paper takes the first step in theoretically giving the essential explanation of benefit and detriment in RAG to make them explainable and comparable at token-level. We provide a theory about describing the trade-off between the value of external knowledge and its potential risk of misleading LLMs in next token prediction of RAG. We prove that the actual effect of RAG can be predicted at token level by representation similarity. Based on our theoretical results, we propose a practical novel method that enables pure LLM and RAG to collaborate at token level, gaining benefit while avoiding detriment. Experiments show the effectiveness of our method and support our theoretical results. **Limitations and Societal Impact:** The main limitation of this paper is that due to immense resource cost, we do not evaluate our method on LLMs with 33B and 65B scales. Our paper deepens society’s understanding of LLMs’ usage of external retrieved knowledge through theoretical analysis. After careful consideration, we believe that our paper does not have any potential negative societal impact. 

10 

## REFERENCES 

Mark A Aizerman, Emmanuil M Braverman, and Lev I Rozonoer. Theoretical foundation of potential functions method in pattern recognition. _Avtomatika i Telemekhanika_ , 25(6):917–936, 1964. 

- Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. _arXiv preprint arXiv:2211.15661_ , 2022. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. _arXiv preprint arXiv:2310.11511_ , 2023. 

- Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. A neural probabilistic language model. _Advances in neural information processing systems_ , 13, 2000. 

- Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In _Proceedings of the 2013 conference on empirical methods in natural language processing_ , pp. 1533–1544, 2013. 

- David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. _Journal of machine Learning research_ , 3(Jan):993–1022, 2003. 

- Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. Inside: Llms’ internal states retain the power of hallucination detection. _arXiv preprint arXiv:2402.03744_ , 2024. 

- Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. _arXiv preprint arXiv:2309.03883_ , 2023. 

- Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers. _arXiv preprint arXiv:2212.10559_ , 2022. 

- Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, and Xueqi Cheng. Retrieve only when it needs: Adaptive retrieval augmentation for hallucination mitigation in large language models. _arXiv preprint arXiv:2402.10612_ , 2024. 

- Shangbin Feng, Weijia Shi, Yuyang Bai, Vidhisha Balachandran, Tianxing He, and Yulia Tsvetkov. Knowledge card: Filling llms’ knowledge gaps with plug-in specialized language models. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. In-context learning of large language models explained as kernel regression. _arXiv preprint arXiv:2305.12766_ , 2023. 

- Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In _International Conference on Machine Learning_ , pp. 9639–9659. PMLR, 2022. 

- Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. _arXiv preprint arXiv:2305.06983_ , 2023. 

- Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Qiuxia Li, and Jun Zhao. Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models. _arXiv preprint arXiv:2402.14409_ , 2024a. 

- Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. _arXiv preprint arXiv:2402.18154_ , 2024b. 

- Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. _arXiv preprint arXiv:1705.03551_ , 2017. 

11 

- Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. _arXiv preprint arXiv:2004.04906_ , 2020. 

- Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. _arXiv preprint arXiv:2302.09664_ , 2023. 

- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:452–466, 2019. doi: 10.1162/tacl_a_00276. URL https://aclanthology.org/Q19-1026. 

- Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. _arXiv preprint arXiv:1910.13461_ , 2019. 

- Yuchen Li, Yuanzhi Li, and Andrej Risteski. How do transformers learn topic structure: Towards a mechanistic understanding. In _International Conference on Machine Learning_ , pp. 19689–19729. PMLR, 2023. 

- Zi Lin, Jeremiah Zhe Liu, and Jingbo Shang. Towards collaborative neural-symbolic graph semantic parsing via uncertainty. _Findings of the Association for Computational Linguistics: ACL 2022_ , 2022. 

- Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. _arXiv preprint arXiv:2002.07650_ , 2020. 

- Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. _arXiv preprint arXiv:2212.10511_ , 2022. 

- Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. _arXiv preprint arXiv:1609.07843_ , 2016. 

- Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. _arXiv preprint arXiv:2305.14251_ , 2023. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. _arXiv preprint arXiv:1606.05250_ , 2016. 

- Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. _arXiv preprint arXiv:2302.00083_ , 2023. 

- Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hao Tian, Hua Wu, Ji-Rong Wen, and Haifeng Wang. Investigating the factual knowledge boundary of large language models with retrieval augmentation. _arXiv preprint arXiv:2307.11019_ , 2023. 

- Corby Rosset, Chenyan Xiong, Minh Phan, Xia Song, Paul N. Bennett, and Saurabh Tiwary. Knowledge-aware language model pretraining. _CoRR_ , abs/2007.00655, 2020. URL https: //arxiv.org/abs/2007.00655. 

- Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. _arXiv preprint arXiv:2112.01488_ , 2021. 

12 

- Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. _Advances in Neural Information Processing Systems_ , 35:17456–17472, 2022. 

- Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. _arXiv preprint arXiv:2301.12652_ , 2023. 

- Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. Asqa: Factoid questions meet long-form answers, 2023. 

- Ian Tenney, Dipanjan Das, and Ellie Pavlick. Bert rediscovers the classical nlp pipeline. _arXiv preprint arXiv:1905.05950_ , 2019. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

- Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pp. 35151–35174. PMLR, 2023. 

- Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. Large language models are latent variable models: Explaining and finding good demonstrations for in-context learning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus for sentence understanding through inference. _arXiv preprint arXiv:1704.05426_ , 2017. 

- Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Unraveling the behavior of large language models in knowledge conflicts. _arXiv preprint arXiv:2305.13300_ , 2023. 

- Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_ , 2021. 

- Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-seng Chua. Search-in-the-chain: Towards the accurate, credible and traceable content generation for complex knowledge-intensive tasks. _arXiv preprint arXiv:2304.14732_ , 2023. 

- Shicheng Xu, Liang Pang, Jun Xu, Huawei Shen, and Xueqi Cheng. List-aware reranking-truncation joint model for search and retrieval-augmented generation. _arXiv preprint arXiv:2402.02764_ , 2024a. 

- Shicheng Xu, Liang Pang, Mo Yu, Fandong Meng, Huawei Shen, Xueqi Cheng, and Jie Zhou. Unsupervised information refinement training of large language models for retrieval-augmented generation, 2024b. 

- Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation. _arXiv preprint arXiv:2401.15884_ , 2024. 

- Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust to irrelevant context. In _International Conference on Learning Representations_ , 2024. 

- Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. What and how does in-context learning learn? bayesian model averaging, parameterization, and generalization. _arXiv preprint arXiv:2305.19420_ , 2023. 

13 

## A PROOF FOR EQUATION 5 

_Proof._ The transformation is motivated by (Xie et al., 2021) and we apply it to the analysis of RAG: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0014-02.png)


B PROOF FOR EQUATION 6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0014-04.png)


According to the definition of latent variable model in the analysis of in-context learning from (Xie et al., 2021) that views the latent variable inference as Hidden Markov Model (HMM) and the latent concept _z_ determines the transition probability matrix in HMM hidden states _h_ , we can get the following derivations : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0014-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0014-07.png)


Based on previous work (Xie et al., 2021; Zhang et al., 2023), we make the following assumptions that: 

**Assumption 1.** _All tokens can be predicted, which means that for every token x, there is some hidden state h lower bounds it that p_ ( _x|h, z[∗]_ ) _> c_ 1 _>_ 0 _._ 

**Assumption 2.** _Delimiter is an important distinguishing signal between each passage r in the retrieved texts R. For any delimiter hidden state h[d] and other hidden state h, there are upper and lower bounds on the transition probability from h to h[d] :_ 0 _≤ c_ 2 _≤ p_ ( _h[d] |h, z_ ) _≤ c_ 3 _._ 

Then we can get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0014-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0014-13.png)


14 

So we can get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0015-01.png)


_ri_ is a passage in the retrieved texts list _R_ , _h[d] i−_ 1[is][the][hidden][state][for][the][delimiter][between] _ri−_ 1 and _ri_ in _R_ . According to the Assumption 2, _p_ ( _h[d] i−_ 1 _[|][r]_[1:] _[i][−]_[1] _[, z]_[)][=] _[O]_[(1)][, then Equation 34 is] approximately equal to[�] _[n] i_ =1 _[O]_[(1)] _[p]_[(] _[r][i][|][z]_[)][, which means that] _[ p]_[(] _[R, x]_[1:] _[i][−]_[1] _[|][z]_[)] _[≈]_[�] _[n] i_ =1 _[O]_[(1)] _[p]_[(] _[r][i][|][z]_[)][,] so we can get that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0015-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0015-04.png)


_pR_ ( _·_ ) is the distribution of the retrieved texts, _p_ ( _·_ ) is the distribution of the LLMs’ pre-trianed knowledge. 

## C EFFECT OF _v_ ( _z_ ) IN DISTRIBUTION FUSION 

Recapping the Equation 38, we find _v_ ( _z_ ) actually regulates the proportion between LLMs’ pre-trained knowledge and retrieved knowledge in distribution fusion of RAG prediction: 

- The more benefit outweigh detriment, _v_ ( _z_ ) _→−∞_ and exp( _v_ ( _z_ )) _→_ 0 for all _z_ = _z[∗]_ , this indicates that concepts _z_ sampled from LLMs’ space contribute little to _p_ ( _xi|R, x_ 1: _i−_ 1). When _z_ = _z[∗]_ , exp( _r_ ( _z[∗]_ )) = 1, which means that latent variable model concentrates more on _z[∗]_ sampled from retrieved texts. As _v_ ( _z_ ) decreases, the proportion of retrieved knowledge in becomes larger and larger in fusion. 

- The more detriment outweigh benefit, _v_ ( _z_ ) _→_ + _∞_ and exp( _v_ ( _z_ )) _→_ + _∞_ for all _z_ = _z[∗]_ and when _z_ = _z[∗]_ , exp( _r_ ( _z[∗]_ )) = 1. This indicates that concepts _z_ sampled from LLMs’ space contribute more and more than _z[∗]_ sampled from retrieved texts as _v_ ( _z_ ) increases. 

## D PROOF FOR THEOREM 1 

_Proof._ Recapping the Equation 2 that describes the distribution fusion in RAG via latent variable model: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0015-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0015-13.png)


Since latent concept _z[∗]_ determines the hidden states _h_ , Λ can be transformed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0015-15.png)


15 

Let _p_ ( _z[∗] |R, x_ 1: _i−_ 1) = _β_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-02.png)


let probabilities _p_ ( _xi|x_ 1: _i−_ 1 _, h, z[∗]_ ) _p_ ( _x_ 1: _i−_ 1 _|h, z[∗]_ ) in Equation 40 is represented as matrix _W ∈_ R _[|X|×|H|]_ for all possible _xi ∈X_ and _h ∈H_ , _p_ ( _h|R, z[∗]_ ) in Equation 44 is matrix _B_ , _pR_ ( _h_ ) in Equation 43 is _u ∈_ R _[|H|]_ . We use 1-norm to calculate the difference between _p_ ( _xi|R, x_ 1: _i−_ 1) and _pR_ ( _xi|x_ 1: _i−_ 1), which can be formalized as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-04.png)


Then, according to the triangle inequality of 1-norm, the difference between _p_ ( _xi|R, x_ 1: _i−_ 1) and _pR_ ( _xi|x_ 1: _i−_ 1) is bouned by: 

_∥_ Φ _∥_ 1 _−∥βWB − Wu∥_ 1 _≤∥_ Φ + _βWB − Wu∥_ 1 _≤∥_ Φ _∥_ 1 + _∥βWB − Wu∥_ 1 _._ (46) We consider to further analyze _∥βWB − Wu∥_ 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-07.png)


Then: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-09.png)


in which _r_ is the passage in _R_ , KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )) is actually the **detriment** in Equation 9. Recapping Equation 46, we can get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-11.png)


Since 0 _≤∥βWB−Wu∥_ 1 _≤_ 2�2KL( _pR_ ( _·_ ) _∥p_ ( _·|z[∗]_ )) and _∥_ Φ+ _βWB−Wu∥_ 1 _≥∥_ Φ _∥_ 1 _−∥βWB− Wu∥_ 1, then the lower bound for _∥_ Φ + _βWB − Wu∥_ 1 is included in: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0016-13.png)


16 

we take the minimum value as the lower bound. Define _D_ = _∥p_ ( _xi|R, x_ 1: _i−_ 1) _− pR_ ( _xi|x_ 1: _i−_ 1) _∥_ 1 is the difference between _p_ ( _xi|R, x_ 1: _i−_ 1) and _pR_ ( _xi|x_ 1: _i−_ 1), according to Equation 45 and 46, the lower and upper bound for _D_ is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0017-01.png)


For ease of description, we denote benefit KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ ) as Ω and denote detriment KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ ) as Υ. Recapping Equation 5 and 9: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0017-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0017-04.png)


Φ in Equation 60 can be transformed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0017-06.png)


Now the Theorem 1 has been proven. 

## E PROOF FOR THEOREM 2 

In this section, we try to prove that the gap between values of benefit and detriment is approximately positively correlated with the similarity ( _D_[1][) between] _[ p]_[(] _[x][i][|][R, x]_[1:] _[i][−]_[1][)][ and] _[ p][R]_[(] _[x][i][|][x]_[1:] _[i][−]_[1][)][.][To achieve] this, we can start from Equation 60 to prove that the gap between values of benefit and detriment is negatively correlated with the difference ( _D_ ) between _p_ ( _xi|R, x_ 1: _i−_ 1) and _pR_ ( _xi|x_ 1: _i−_ 1), which is actually the reciprocal of similarity ( _D_[1][).][Specifically, we want to prove that the gap between values] of benefit and detriment (KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ ) _−_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )) is negatively correlated with both lower and upper bound in Equation 60. For ease of description, we denote benefit KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ ) as Ω and denote detriment KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ ) as Υ. 

17 

_Proof._ Recapping Equation 5 and 9: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0018-01.png)


Φ in Equation 60 can be transformed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0018-03.png)


Therefore, the lower bound of Equation 60 is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0018-05.png)


and the upper bound of Equation 60 is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0018-07.png)


Due to both Ω and Υ being variables, analyzing the result of subtraction between Ω and Υ under their simultaneous changes is complex. Therefore, we use the “Separation of variables“ to simplify our analysis. Specifically, we first assume that one is constant, and then analyze the changes caused by the variation of another: 

- Assume Ω is constant, as the value of Ω _−_ Υ increases, Υ decreases and the upper bound exp( _−_ (Ω _−_ Υ)) + _√_ 2Υ also deceases. In the lower bound exp( _−_ (Ω _−_ Υ)) _− √_ 2Υ, since the first term is an exponential function and the second term is a square root function, a decrease of Υ leads to the decrease in the entire lower bound. Therefore, both lower and upper bounds in Equation 60 decrease as Ω _−_ Υ increases. 

18 

- Assume Υ is constant, as the value of Ω _−_ Υ increases, Ω increases and the upper bound exp( _−_ (Ω _−_ Υ)) + _√_ 2Υ deceases. In the lower bound exp( _−_ (Ω _−_ Υ)) _− √_ 2Υ, since the first term is an exponential function and the second term is a square root function, an increase of Ω leads to the decrease in the entire lower bound. Therefore, both lower and upper bounds in Equation 60 decrease as Ω _−_ Υ increases. 

On behalf of the analysis above, we can derve that both lower and upper bounds in Equation 60 are approximately negatively correlated with the gap between values of benefit and detriment. Therefore, the difference _D_ between _p_ ( _xi|R, x_ 1: _i−_ 1) and _pR_ ( _xi|x_ 1: _i−_ 1) is approximately negatively correlated with the gap between values of benefit and detriment. In other words, _D_ 1[can][be][treated][as][the] similarity between _p_ ( _xi|R, x_ 1: _i−_ 1) and _pR_ ( _xi|x_ 1: _i−_ 1) and it is approximately positively correlated with the gap between values of benefit and detriment.: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0019-02.png)


So we have proved that the gap between values of benefit and detriment is approximately positively correlated with[1] _D_[.] 

## F PROOF FOR THEOREM 3 

This section aims to prove: 

KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ )) _<_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )), detriment outweighs benefit. if _D_[1] _[<] M_ 1 _J_ = KLKL(( _ppRR_ (( _rr_ )) _∥∥pp_ (( _rr||zz_ )) =)) _>_ KL KL(( _ppRR_ (( _rr_ )) _∥∥pp_ (( _rr||zz[∗][∗]_ )))), detriment is equal to benefit., benefit outweighs detriment. ifif _DD_[1][1][=] _[>] MM_ 11 (73) 

in which _M_ 1[is the similarity between] _[ p]_[(] _[x][i][|][R, x]_[1:] _[i][−]_[1][)][ and] _[ p]_[(] _[x][i][|][x]_[1:] _[i][−]_[1][)][ (LLMs’ pre-trained knowl-] edge), _D_[1][is the similarity between] _[ p]_[(] _[x][i][|][R, x]_[1:] _[i][−]_[1][)][ and] _[ p][R]_[(] _[x][i][|][x]_[1:] _[i][−]_[1][)][ (distribution of retrieved texts)] 

## _Proof._ When benefit is equal to detriment: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0019-09.png)


which means that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0019-11.png)


since _pR_ ( _r_ ) cannot be 0, then: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0019-13.png)


Recapping Equation 2 that _z[∗]_ is sampled from retrieved texts and _z_ is sampled from LLMs’ pretrained knowledge, Equation 78 indicates that the knowledge of retrieved texts has been involved in LLLs’ pre-trained knowledge, so: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0019-15.png)


then: 

_∥p_ ( _xi|R, x_ 1: _i−_ 1) _− p_ ( _xi∥x_ 1: _i−_ 1) _∥_ 1 = _∥p_ ( _xi|R, x_ 1: _i−_ 1) _− pR_ ( _xi∥x_ 1: _i−_ 1) _∥_ 1 _,_ (80) 

which means that[1] 1[When][1] 1 _D_[=] _M_[is an important dividing point.] _D_[=] _M_[, we can get that benefit] is equal to detriment and KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ )) _−_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )) = 0. Equation 72 indicates that the gap between values of benefit and detriment (KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ )) _−_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ ))) is approximately positively correlated with _D_[1][.][Therefore, when] _D_[1] _[>] M_ 1[we can get that benefit] outweighs detriment (KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ )) _−_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )) _>_ 0). When _D_[1] _[<] M_ 1[we can get] that detriment outweighs benefit (KL( _pR_ ( _r_ ) _∥p_ ( _r|z_ )) _−_ KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ )) _<_ 0). Now the proof of Theorem 3 has been finished. 

19 

## G PROOF FOR RAG IS ACTUALLY UNSUPERVISED IN-CONTEXT LEARNING 

This section aims to prove that RAG is actually unsupervised ICL from two perspectives. One is that previous studies find that ICL performs gradient descent as meta-optimizer (Von Oswald et al., 2023; Akyürek et al., 2022; Dai et al., 2022). We prove that in this perspective, the distribution of texts in context drives the learning even without explicit input-output supervision. Therefore, the distribution of unsupervised retrieved texts in RAG, which is actually the distribution of context for query, can also drives the learning. Then we can prove that RAG is actually unsupervised in-context learning. The specific proof is: 

_Proof._ From the perspective that ICL performs gradient descent as meta-optimizers, ICL can be formalized as the following: 

Gradient descent in optimization of linear layers have a dual form of linear attention (Irie et al., 2022; Aizerman et al., 1964), define a liner layer as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-04.png)


in which _W_ 0 is the initial weight matrix. Given a sequence of historical input vectors **xi** _∈_ R _[d][in]_ and corresponding error signals **ei** _∈_ R _[d][out]_ , _i ∈_ [1 _, N_ ] obtained by gradient descent, the update of the weight matrix can be represented as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-06.png)


Recap that the linear attention can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-08.png)


Then the dual form of updated linear layer with new input **x** _N_ +1 is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-11.png)


In in-context learning, the attention of a head is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-14.png)


in which **q** = _WQ_ **b** , **b** is the input _t_ -th token in query, _WQ_ , _WK_ , _Wv_ are projection matrices, _B[′]_ is demonstrations of the context and _X_ is the prefix for **b** of the query. To simplify qualitative analysis, we follow (Dai et al., 2022) to estimate the standard attention as relaxed linear attention, achieved by eliminating the softmax function and the scaling factor: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0020-16.png)


20 

According to Equation 88, the dual form of the Transformer attention is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-01.png)


Based on above derivation, we have this finding: comparing Equation 86 with Equation 96, we find that _WV B_ ( _WKB_ ) _[T]_ is equal to the initial weight matrix _W_ 0, which is zero-shot prediction give query prefix _B_ without demonstrations in the context. Besides, _WV_ **b** _[′] i_[is equal to] **[ e]** _[i]_[.][which is the] meta-gradient used to update the weighted matrix. _WK_ **b** _i_ is equal to the historical input vectors: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-04.png)


In the standard gradient descent with loss _L_ , _ei_ = _−η ∂y[∂][L] i_[and] _[ η]_[ is the learning rate and] _[ y][i]_[=] _[ W][i][x][i]_[is] the output of the linear layer using the weight matrix _Wi_ at step _t_ (Irie et al., 2022). So we can get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-07.png)


Therefore: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-09.png)


So the loss _L_ ca be represented as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-11.png)


Equation 97 and 102 show that the supervision signal, both loss and gradient, are directly related to the semantic representation of the tokens ( _bi_ ) in the demonstration of context. This suggests that the distribution of the text in context is a direct learning signal for in-context learning, without the need for explicit input-output pairs in the demonstration. From the perspective that ICL performs gradient descent as meta-optimizer, the proof has been finished. 

The other pespective is from our theoretical results in Equation 5 and 9 that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation_images/iclr_2025_A_Theory_for_Token-Level_Harmonization_in_Retrieval-Augmented_Generation.pdf-0021-15.png)


21 

We explain Equation 110 from the perspective of the loss function in gradient descent to explain the learning from retrieved texts of LLMs in RAG. _pR_ ( _r_ ) is the distribution of retrieved texts and it can serve as ground truth distribution in loss functions. _p_ ( _r|z_ ) and _p_ ( _r|z[∗]_ ) are distribution estimated by LLMs. Two KL-divergence between ground truth distribution _pR_ ( _r_ ) and estimated distribution _p_ ( _r|z_ ) and _p_ ( _r|z[∗]_ ) respectively are loss functions. _r_ is the retrieved passage that invariant in generation process, so what contributes to the change of loss function is sampling more and more accurate _retrieved concept z[∗]_ from the retrieved texts. So KL( _pR_ ( _r_ ) _∥p_ ( _r|z[∗]_ ) is the actual loss that can be meta-optimized in RAG, in which _pR_ ( _r_ ) is the ground truth distribution and _p_ ( _r|z[∗]_ ) is the estimated distribution. As this loss decreases, the value of _v_ ( _z_ ) when _z_ is not equal to _z[∗]_ also decreases, which means that in Equation 104, the ratio of the knowledge from LLMs’ pre-trained distribution decreases and meanwhile the ratio of knowledge from the retrieved texts increases. Lower loss means that the output of RAG is closer to the distribution of retrieved texts, which is actually that LLMs learning the distribution from retrieved texts in input context. Since _pR_ ( _r_ ) is the ground truth in this learning but dose not have any explicit input-output supervision like demonstrations in in-context learning, RAG is the **unsupervised** in-context learning and distribution of retrieved texts ( _pR_ ( _r_ )) is the unsupervised learning signal. 

Based on the above two perspectives, we successfully prove that: _The distribution of retrieved passage r in RAG (i.e., pR_ ( _r_ ) _) can serve as the unsupervised learning signal for LLMs learning from context, even without explicit input-output supervision. RAG is actually unsupervised in-context Learning that fuses the distribution from retrieved texts with LLMs’ pre-trained distribution._ 

## H EXPLORE THE MECHANISM OF RAG 

We also perform experiments in Section 3.1 using more LLMs (OPT-6.7B, Mistral-7B) on more datasets (TriviaQA, WebQ and Squad). The experimental results are shown in Table 4 and 5, and the conclusions of them are consistent with Section 3.1. 

|Datasets|Value|0|4|8|12|16|20|24|28|32|
|---|---|---|---|---|---|---|---|---|---|---|
|TriviaQA|Attention Score|0.12|0.79|0.65|0.51|0.59|0.72|0.86|0.69|0.43|
|TriviaQA|Word Distribution Change|0.10|0.11|0.10|0.10|0.13|0.22|0.38|0.56|0.82|
|WebQ|Attention Score|0.10|0.69|0.65|0.54|0.55|0.70|0.82|0.64|0.40|
|WebQ|Word Distribution Change|0.15|0.13|0.13|0.14|0.16|0.27|0.45|0.55|0.79|
|Squad|Attention Score|0.16|0.82|0.64|0.55|0.63|0.69|0.90|0.67|0.48|
|Squad|Word Distribution Change|0.11|0.11|0.13|0.12|0.12|0.24|0.36|0.69|0.89|



Table 4: Attention score for and difference of word distribution change vary with layers based on OPT-6.7B on TriviaQA, WebQ and Squad. 

|Datasets|Value|0|4|8|12|16|20|24|28|32|
|---|---|---|---|---|---|---|---|---|---|---|
|TriviaQA|Attention Score|0.05|0.64|0.50|0.45|0.60|0.69|0.81|0.65|0.37|
|TriviaQA|Word Distribution Change|0.08|0.10|0.11|0.15|0.19|0.30|0.42|0.65|0.82|
|WebQ|Attention Score|0.07|0.69|0.55|0.46|0.66|0.70|0.82|0.60|0.42|
|WebQ|Word Distribution Change|0.12|0.14|0.14|0.17|0.29|0.38|0.50|0.67|0.85|
|Squad|Attention Score|0.10|0.75|0.60|0.48|0.65|0.72|0.86|0.62|0.40|
|Squad|Word Distribution Change|0.13|0.13|0.15|0.19|0.35|0.40|0.57|0.70|0.84|



Table 5: Attention score for and difference of word distribution change vary with layers based on Mistral-7B on TriviaQA, WebQ and Squad. 

22 

## I EXPERIMENTAL DETAILS 

## I.1 BASELINES 

For benefit-detriment comparison experiment that needs methods to determine the value order between benefit and detriment for each token, it is actually a binary classification task (benefit outweigh detriment or not). The mainstream methods in this area are detecting and comparing the degree of hallucination between tokens generated by LLMs (w/o RAG) and RAG. Below we will describe in detail how we apply these baselines to this task. 

**Logprobs.** Logprobs can indicate the confidence for LLMs in generating the tokens(Kuhn et al., 2023). We use the value order between top-1 log-probability of the tokens output by pure LLM and RAG to determine the value order between benefit and detriment for these tokens. If the logprobs of tokens generated by RAG is greater than the logprobs of tokens generated by pure LLM, the benefit outweigh the detriment, otherwise the detriment outweigh the benefit. 

**Uncertainty.** We use Length-normalized Entropy (Malinin & Gales, 2020) to measure the uncertainty of the tokens generated by pure LLM and RAG respectively. If the uncertainty of tokens generated by RAG is lower than the uncertainty of tokens generated by pure LLM, the benefit outweigh the detriment, otherwise the detriment outweigh the benefit. 

**Consistency-Lexical (Lin et al., 2022).** Consistency-based methods make LLMs perform multiple generations for a question and calculate consistency score among multiple answers. If the consistency score of tokens generated by RAG is greater than the consistency score of tokens generated by pure LLM, the benefit outweigh the detriment, otherwise the detriment outweigh the benefit. Lexical-based consistency means calculating consistency score by lexical-similarity among multiple answers. Since the experiment is at token level, we use the number of tokens that are completely consistent in multiple generations as the consistency score. 

**Consistency-Semantic (Chen et al., 2024).** We follow (Chen et al., 2024) to use EigenScore to calculate the semantic similarity among hidden states of tokens in multiple generations and use it as the consistency score. 

For open-domain Q&A under practical autoregressive generation setting, baselines for this include the methods that introduce additional modules to filter irrelevant passages ( **NLI+RAG** (Yoran et al., 2024)) or as action triggers ( **CRAG** (Yan et al., 2024)), train more robust LLMs for RAG ( **RetRobust** (Yoran et al., 2024) and **INFO-RAG** (Xu et al., 2024b)) and train LLMs to dynamically retrieve and critique retrieved texts ( **Self-RAG** (Asai et al., 2023)). 

**NLI+RAG.** This method use a Natural Language Inference model to filter the possible irrelevant documents in retrieved results and provide the remaining documents to LLMs for generation. We follow (Yoran et al., 2024) to use a BART-Large model (Lewis et al., 2019) with 407 million parameters trained on the MNLI dataset (Williams et al., 2017). We consider a query-document pair as entailed if the probability for the entailment label is _≥_ 0 _._ 5 and filter the documents with probability for the entailment label _<_ 0 _._ 5. 

**CRAG.** This method uses a retrieval evaluator to assess the correctness of retrieved texts trigger different actions based on the evaluation results. One of the actions is using additional google search API for web search, which is unfair for baselines and our method. So we remove this action and use its knowledge refinement strategy for document filtering (Yan et al., 2024). 

**RetRobust.** This method fine-tunes LLMs to properly leverage retrieved passages with a mix of relevant and irrelevant contexts (Yoran et al., 2024). 

**INFO-RAG.** This method uses unsupervised method to make LLMs learn to use the retrieved texts robustly. It enables LLMs to judge the correctness of the retrieved texts, extract the correct content and revise the wrong content (Xu et al., 2024b). 

**Self-RAG.** This method trains LLMs to dynamically retrieve and critique retrieved texts. Self-RAG first decodes a retrieval token to evaluate the utility of retrieval and control a retrieval component. If retrieval is required, LLM calls an external retrieval module to find top relevant documents, using input query and previous generation. If retrieval is not required, LLM continues generation. If retrieval is needed, LLM first generates critique token evaluating whether retrieved documents 

23 

are relevant and support generation, and then generates continuation conditioned on the retrieved passages (Asai et al., 2023). 

## I.2 IMPLEMENTATION DETAILS 

All models are run on a V100 GPU with Pytorch (Paszke et al., 2019) and accelerated by DeepSpeed[1] . As for retrieval for RAG, we follow (Xu et al., 2023; 2024b) to use ColBERTv2 (Santhanam et al., 2021), an excellent generalizable model as the retriever, and use Wikipedia consisting of 21,015,324 passages (Karpukhin et al., 2020) as retrieval database. **All baselines and Tok-RAG share the same retrieval setup and prompt.** We use OPT-6.7B, LLaMA-2-7B and Mistral-7B-v0.1 as LLMs in benefit-detriment comparison experiment and use greedy-decoding strategy for generation. 

## J CASE STUDY 

Figure 5 shows the case study for collaborative generation between pure LLM and RAG at token level in our **Tok-RAG** . At the step that pure LLM and RAG generates the different tokens, **Tok-RAG** use our theoretical results in Theorem 3 to compare the benefit and detriment. If benefit is greater than detriment, the token from RAG is selected, otherwise, the token from pure LLM is selected. The selected tokens are marked by green color and bold. Then discarded tokens are marked by gray. The orange arrow represents the direction of token selection and usage. The selected tokens are used for the next step generation of both pure LLM and RAG. This case study visually demonstrates that our **Tok-RAG** effectively enables pure LLM and RAG for collaborative generation to preserve benefit and avoid detriment. 

> 1https://github.com/microsoft/DeepSpeed 

24 

Question: Who is the book of galatians written to? Pure LLM: It was written by the Apostle Peter to the churches in **Galatia** , a region of present-day Turkey. Benefit win Detriment win RAG: It was written by the Apostle **Paul** to the churches in Corinth, a region of present-day Turkey. Output: It was written by the Apostle Paul to the churches in Galatia, a region of present-day Turkey. Question: Who conducted the opening concert at carnegie hall? In which year? Pure LLM: It was conducted by Leopold Damrosch and composer Pyotr Ilyich Tchaikovsky in **1891** . Benefit win Detriment win RAG: It was conducted by **Walter** Damrosch and composer Pyotr Ilyich Tchaikovsky in 1901. Output: It was conducted by Walter Damrosch and composer Pyotr Ilyich Tchaikovsky in 1891. Question: Who says that which we call a rose? Pure LLM: It is said by the character **Juliet** in      Christopher Shakespeare's play " Romeo and Juliet. Benefit win Detriment win RAG: It is said by the character Lysander in **William** Shakespeare's play "Romeo and Juliet. Output: It is said by the character Juliet in William Shakespeare's play "Romeo and Juliet." Question: Who signed the largest on the declaration of independence? Pure LLM: It belongs to **John** Hancock, who was the **president** of the Continental Congress Detriment win Detriment win RAG: It belongs to Charles   Hancock, who was the member of the Continental Congress Output: It belongs to John Hancock, who was the president of the Continental Congress Question: Who is under the mask of darth vader? Pure LLM: It is the character   James  Skywalker, as revealed in the Star Wars series. Benefit win RAG: It is the character **Anakin** Skywalker, as revealed in the Star Wars series. Output: It is the character Anakin Skywalker, as revealed in the Star Wars series. Question: Where does US highway 1 start and end? Pure LLM: It starts in Chicago Kent, Maine, at the Canadian border and ends in **Key** West, Florida. Benefit win Detriment win RAG: It starts in **Fort** Kent, Maine, at the Canadian border and ends in     New    West, Florida. Output: It starts in Fort Kent, Maine, at the Canadian border and ends in Key West, Florida. 

Figure 5: Case study for collaborative generation between pure LLM and RAG at token level in our **Tok-RAG** . Pure LLM and RAG generate the texts in parallel at token level. At the step that pure LLM and RAG generate the different tokens, **Tok-RAG** use our theoretical results in Theorem 3 to compare the benefit and detriment. If benefit is greater than detriment, the token from RAG is selected, otherwise, the token from pure LLM is selected. The selected tokens are marked by green color and bold. The discarded tokens are marked by gray. The orange arrow represents the direction of token selection and usage. The selected tokens are used for the next step generation of both pure LLM and RAG. 

25 

