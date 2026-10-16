# **PathReasoner: Modeling Reasoning Path with Equivalent Extension for Logical Question Answering** 

**Fangzhi Xu**[1] _[,]_[2] **, Qika Lin**[1] _[,]_[2] _[∗]_ **, Tianzhe Zhao**[1] _[,]_[3] **, Jiawei Han**[1] _[,]_[3] **, Jun Liu**[1] _[,]_[2] _[∗]_ 

1School of Computer Science and Technology, Xi’an Jiaotong University 

2Ministry of Education Key Laboratory of Intelligent Networks and Network Security 3Shaanxi Province Key Laboratory of Big Data Knowledge Engineering fangzhixu98@gmail.com,qikalin@foxmail.com,liukeen@xjtu.edu.cn 

## **Abstract** 

Logical reasoning task has attracted great interest since it was proposed. Faced with such a task, current competitive models, even large language models (e.g., ChatGPT and PaLM 2), still perform badly. Previous promising LMs struggle in logical consistency modeling and logical structure perception. To this end, we model the logical reasoning task by transforming each logical sample into reasoning paths and propose an architecture **PathReasoner** . It addresses the task from the views of both data and model. To expand the diversity of the logical samples, we propose an atom extension strategy supported by equivalent logical formulas, to form new reasoning paths. From the model perspective, we design a stack of transformer-style blocks. In particular, we propose a path-attention module to joint model in-atom and cross-atom relations with the highorder diffusion strategy. Experiments show that PathReasoner achieves competitive performances on two logical reasoning benchmarks and great generalization abilities. 

## **1 Introduction** 

With the emergence of pre-trained language models (PLMs) (Kenton and Toutanova, 2019; Brown et al., 2020), recent years have witnessed remarkable progress in the task of machine reading comprehension (MRC) (Rajpurkar et al., 2016; Lai et al., 2017). To tackle more complex scenarios in reality, the challenging logical reasoning task (Yu et al., 2019; Liu et al., 2021a) has been proposed to exploit the model reasoning capability (Huang and Chang, 2023) over text[1] . Similar to the traditional MRC task, it also takes the context, question and options as inputs and requires the model to predict the final answer. Due to the diverse logical characteristics implied in the text, logical reasoning 

> _∗_ Correspondence to Qika Lin and Jun Liu. 

> 1Logical reasoning is a broad concept covering various tasks, but we mainly address the task in the form of MRC. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0001-11.png)


Figure 1: Probing tests on representative LMs (e.g., RoBERTa). (a) is about model prediction consistency. (b) is related to the perception of logical connectives. Detailed pilot experiments are shown in the Appendix. 

task brings huge challenges to current LMs. Especially, faced with such tasks, large language models (LLMs), e.g., ChatGPT[2] and PaLM 2[3] , also struggle a lot which is proved by previous evaluation works (Xu et al., 2023a; Liu et al., 2023). Under such circumstances, this paper will focus more on addressing logical reasoning tasks with small LMs, which are light-weighted and more flexible for future applications[4] . 

Previous competitive LMs expose two limitations in the performance. Firstly, it lacks consistent model predictions on samples with equal logical semantics. For example in Figure 1(a), we make changes to the expression of the original context while maintaining the semantic unchanged, where _not...unless_ is equally transformed into the expression of _only if_ . However, the LMs give inconsistent predictions between the original sample and the 

2https://chat.openai.com 

3https://ai.google/discover/palm2/ 

> 4The focus of this paper is mainly on small LMs, since they are more efficient and effective compared with LLMs on the logical reasoning tasks. But we still report LLM performances for comparison in the experiment section. 

modified one. We blame the problem on the lack of training samples in logical reasoning. Compared with some classic MRC datasets like SQuAD (Rajpurkar et al., 2016, 2018), CoQA (Reddy et al., 2019) with over 100,000 training samples, logical reasoning datasets like ReClor (Yu et al., 2019) and LogiQA (Liu et al., 2021a) are much more sparse with only several thousand samples. Thus, such sparsity limits the learning of logic semantics. Previous work (Jiao et al., 2022) leverages general corpus to conduct continual pretraining, but it does not address the sparsity of logical text in essential. 

Secondly, it remains a challenge to enhance the model perception for logical structures. For example in Figure 1(b), we randomly replace the explicit logical relation words or inverse the negations for the context, which destroys the original semantics. But the LMs fail to change the prediction accordingly. It demonstrates that current LMs are insensitive to the logical connectives, instead they focus more on facts within the text. Considering that current LMs are pre-trained with general objectives on the fact corpus (e.g., Wikipedia), they are naturally weak in capturing the logical structures usually existing in logical-specific scenarios. Some studies like DAGN (Huang et al., 2021), Logiformer (Xu et al., 2022), and AdaLoGN (Li et al., 2022) have attempted to model the explicit logical relations from various perspectives, such as causal and co-occurrence. All of them build text graphs to conduct the reasoning, which limits the scalability to larger text and more complex scenarios. 

In view of the above challenges, we propose an architecture **PathReasoner** , which considers a new paradigm for logical reasoning tasks via reasoning path modeling. Based on the predefined logical rule forms, we represent each natural sentence as an atom and transform each sample into reasoning paths with confidence scores. Under such a paradigm, PathReasoner addresses the task from two views. From the view of expanding the data diversity, we first obtain equivalent atom combinations through external logical formulas, generating new reasoning paths and textualizing them as new samples. From the model view, we propose a reasoning path modeling network. It encodes both function symbols and variables in atoms and forms an atom embedding sequence as the input. In a path-attention module, we model high-order relations from both in-atom and cross-atom perspectives. Through the fusion of token, atom, and path 

embedding, the prediction can be derived. 

Our technical contributions are as follows, and additional key values are in Appendix I: 

(1) We unify the text inputs into atoms and reasoning paths. Based on it, an architecture PathReasoner is proposed to improve both the diversity of samples and logic perception capability. 

(2) In light of the sparsity of training data, we propose an atom extension strategy to form new training samples. To better capture logical structures, we introduce a path-attention module with highorder relation modeling, enabling joint updates of information within atoms and across atoms. 

(3) Extensive experiments show superior performances on two logical reasoning benchmarks. Significant generalization capabilities are also verified. 

## **2 Related Work** 

Recent progress in MRC promotes the emergence of more complex tasks like logical reasoning. Previously, several datasets on logical reasoning have been proposed, including ReClor (Yu et al., 2019), LogiQA (Liu et al., 2021a) and AR-LSAT (Zhong et al., 2021). They have attracted much attention since some LMs fail to show superiority. Previous works on the logical reasoning task can be categorized into two folds. 

**Sequence-based.** These models are usually accompanied by data augmentation strategies. LReasoner (Wang et al., 2022) proposes to extend text with logical formulas to enrich the context information. MERIt (Jiao et al., 2022) proposes a contrastive strategy based on the meta-path and leverages the extra data to pre-train the model. However, both of them lack the relation modeling of logical units in the sequence. 

**Graph-based.** DAGN (Huang et al., 2021) is the first work to divide the text into discourse units and utilize the graph neural networks (Zhou et al., 2020) to update the representations. But its chaintype graph structure limits the expression of complex relations between logical units. FocalReasoner (Ouyang et al., 2021) focuses on the fact triplet extracted from the text and builds a supergraph for reasoning. But it ignores the effects of the logical connectives within the text. To better model the logic within text, AdaLoGN (Li et al., 2022) designs an adaptive network to update the text graph progressively. Logiformer (Xu et al., 2022) proposes a two-branch graph transformer 

network to address the text from syntax and logic. However, it is costly to form and update the text graph during the reasoning process. In general, the graph-based methods naturally lack expansibility, especially when the text becomes larger. 

Considering the above drawbacks, we propose a reasoning pattern based on the reasoning paths (instantiated logical rules) for the first time. It models the logical reasoning task from a special perspective and combines the advantages of both sequence and graph-based methods. 

## **3 Preliminary** 

This work considers unifying the inputs into the form of logical rules (Lin et al., 2021) since it is a more natural way to uncover logical structures of the text while maintaining the important facts. The distinctive values of such definitions over firstorder logic (Xu et al., 2023c) and propositional logic are in Appendix C. We introduce the following two definitions. 

**Definition 1: atom.** We transform each natural sentence into one atom (Hinman, 2018), which consists of one function symbol and several variables. For example, given the sentence _Paula will visit the dentist only if Bill goes golfing_ , we define the expression OnlyIf( _A, B_ ) as the atom. OnlyIf is the function symbol that denotes the explicit connective phrase in the sentence. And _A, B_ are called variables to represent abstract sentence constitutes, whose instantiation are _Paula will visit the dentist_ and _Bill goes golfing_ respectively. Similarly, we can also derive other atoms from the text, such as Unless( _A, B_ ), Since( _A, B_ ), InFact( _A_ ). 

According to the reasoning patterns, we define four categories of function symbols, shown in Table 1. The first is causal relations for deterministic facts. The second and third ones are conditional assumptions, where _NA_ focuses on the uniqueness of the condition. The last one is facts with no explicit logical relations. 

**Definition 2: reasoning path.** Based on Definition 1, we can unify the context, question and options of each input into the form of the logical rule (Lin et al., 2022; Pan et al., 2022), such as Eq. 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0003-07.png)


Rule body functions as the modeling of the context part, which is represented as the conjunction of 

|**Category**|**Representative Connectives**|
|---|---|
|Cause<br>SA<br>NA<br>Fact|Because, Since, DueTo, TheReasonsWhy...<br>If, When, Once, AsLongAs, ...<br>OnlyIf, Unless, ...<br>InFact, Actually, InAll, ToConclude ...|



Table 1: Categories of function symbols. ‘SA’ and ‘NA’ are short for _Sufficient Assumption_ and _Necessary Assumption_ respectively. 

atoms. Rule head consists of the concatenation of the question sentence and option _ai_ , which is also represented as the conjunction of atoms in the implementation. The symbol _ε_ indicates the confidence score of the logical rule. Since each option is bounded with one logical rule, _ε_ is also equal to the confidence of option _ai_ . In actual cases, the function symbols (e.g., _F_ 1 _, F_ 2) and variables (e.g., _A, B_ ) are instantiated as the natural language. Therefore, this paper defines the instantiated logical rule as the reasoning path. 

**Implementation** We use over 100 pre-defined function symbols, grouped into four categories, and apply hand-crafted rules to match them in the sentence. The function symbols along with the punctuation can be divided into one or two parts, as instantiated variables. This strategy is relatively complete, illustrated in Appendix B. 

## **4 Methods** 

To tackle the challenges in the logical reasoning task, we propose the architecture PathReasoner, shown in Figure 2. It includes two main parts: (a) Equivalent Path Extension (EPE) and (b) Reasoning Path Modeling (RPM). The former module is aimed at expanding the sample diversity to improve the consistency of model prediction. The latter one targets at improving the logic perception capability of the reasoning model. 

## **4.1 Equivalent Path Extension** 

After unifying the inputs into the logical rule form, it is natural to exploit the equivalent logic to facilitate the equivalent extension. 

## **4.1.1 External Logical Formulas** 

In the beginning, we introduce external logical formulas to achieve the atom extension. Corresponding to the function symbols, we employ the following logical formulas. 

_(A) Equivalence Logic._ It defines the bi-directional derivation between atoms as Eq. 2, where □ _∈_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0004-00.png)


Figure 2: The architecture of PathReasoner. Part (a) is Equivalent Path Extension, which aims to improve the diversity of samples. Part (b) is Reasoning Path Modeling, which is designed to model logical structures. 

_{_ Cause _,_ SA _}_ and _¬_ denotes the negation. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0004-03.png)


_(B) Single Atom Derivation_ . Such logical formula is targeted at transforming NA atoms to SA, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0004-05.png)


_(C) Multiple Atom Derivation_ . Depending on the conjunction of atoms, we can generate more diverse text. We only present the logical formulas with two atom conjunction in Eq. 4 and 5, since more complex situations can be derived by repeating the extension process. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0004-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0004-08.png)


In above equations, _⋆ ∈{_ Cause _,_ NA _,_ SA _}, △∈ {_ Cause _,_ NA _,_ SA _}_ and _▽∈{_ Cause _,_ NA _,_ SA _}_ . 

## **4.1.2 Reasoning Path Engine and Filter** 

Taking original reasoning paths and equivalent logical formulas as inputs, the reasoning path engine module aims to generate the candidate samples. 

Firstly, we conduct multi-round atom extension. For example in Fig. 2(a), there exist four atoms in the original reasoning path. At the first round, the atom Unless( _C, B_ ) can derive If( _¬C, ¬B_ ), and 

also a new atom OnlyIf( _C, A_ ) can be added into the atom base through the conjunction derivation of Unless( _C, B_ ) and OnlyIf( _B, A_ ). We repeat the extension process to include all potential atoms. Thus, an extended atom base is formed. 

Secondly, our purpose is to mine atom combinations to form new reasoning paths. By enumerating all possible combinations, we select the ones which can recover the original path in reverse. For example, the combination of OnlyIf( _B, A_ ), OnlyIf( _C, B_ ), Fact( _¬C_ ) and If( _¬C, ¬A_ ) is a valid candidate because it can derive the original path with external logical formulas. 

Thirdly, we replace the variables with the corresponding text and textualize the reasoning path form into regular sample form (with context, question and options). 

To reduce noise (e.g., incorrect syntax) in the newly generated candidates, we introduce the path filter module. Specifically, we leverage the PLM (e.g., RoBERTa (Liu et al., 2019)) to train the original samples from the downstream datasets. Therefore, a set of weight parameters is obtained, which is defined as the pre-trained filter in this paper. 

When feeding each sample into the pre-trained filter, we can obtain the confidence score _εi_ of the _i[th]_ reasoning path related to option _ai_ . The predicted option _ak_ is derived with the maximum confidence scores. We keep the samples with both 

correct predictions and high scores, which means _ak_ = _a[∗]_ and _εk > ε[∗]_ . _ak_ is the predicted option with confidence score _εk_ . _a[∗]_ is the ground-truth option and _ε[∗]_ is the threshold that controls the effectiveness of the reasoning path filter. 

## **4.2 Reasoning Path Modeling** 

From the model view, we propose the reasoning path modeling module. Given the input context, question, and options of one sample, we first unify them into the form of the reasoning path based on 3. The initial representation of instantiated variable set _V_ = _{V_ 1 _, V_ 2 _, ..., VK}_ and function symbols set _S_ = _{S_ 1 _, S_ 2 _, ..., SM }_ can be acquired respectively, where _K_ and _M_ are the number of variables and function symbols in the sample. 

For the variable _Vk_ with token sequence _{v_ 1[(] _[k]_[)] _[, v]_ 2[(] _[k]_[)] _[, ..., v] |_[(] _V[k] k_[)] _|[}]_[,][we][leverage][the][LM][as][the] encoder to obtain the token-level embedding _{_ **v**[(] 1 _[k]_[)] _[,]_ **[ v]**[(] 2 _[k]_[)] _[, ...,]_ **[ v]**[(] _|V[k] k_[)] _|[}]_[.][Thus,][its][initial][representa-] tion **V** _k ∈_ R _[d]_ is calculated by the average pooling. We randomly initialize the representations for the function symbol _Sm_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0005-04.png)


By aligning the variables and function symbol for each atom, we can form the atom embedding sequence **A** _∈_ R[(] _[M]_[+] _[K]_[)] _[×][d]_ . To take the order feature into consideration, we include the position embedding (Vaswani et al., 2017) to the input sequence: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0005-06.png)


where **A** _i_ is the embedding of the _i[th]_ unit in **A** , which can be either a variable or a function symbol. In this way, PathReasoner implements the sequential representation of logical rules. 

To perform message passing over the reasoning paths, we propose a stack of _L_ layer blocks in a similar style of Transformer. Specifically, we feed the input sequence into both the self-attention and the proposed path attention module. 

For the self-attention module of the _l[th]_ layer, we follow the regular method, which projects the input sequence into query **Q**[(] _[l]_[)] _∈_ R[(] _[M]_[+] _[K]_[)] _[×][d]_ , key **K**[(] _[l]_[)] _∈_ R[(] _[M]_[+] _[K]_[)] _[×][d]_ and value **V**[(] _[l]_[)] _∈_ R[(] _[M]_[+] _[K]_[)] _[×][d]_ by the projection matrices. Then, the output of the self-attention module can be derived as **H**[(] _[l]_[)] _SA_[.] 

For simplicity, we omit the description of multihead attention in the main paper, but the selection of head number will be discussed in Appendix E.3. 

For the path attention module, we first obtain the interaction matrix **M**[(] _seq[l]_[)] _[∈]_[R][(] _[M]_[+] _[K]_[)] _[×]_[(] _[M]_[+] _[K]_[)] by self multiplication of the input sequence **A** . It models the interaction between any two units. Besides, the importance of each unit can be further considered from the perspective of in-atom and cross-atom. 

In-atom interaction models the information aggregation within one atom. Take the atom _Si_ ( _Vj, Vk_ ) with two variables _Vj, Vk_ and one function symbol _Si_ as an example ( _i_ , _j_ and _k_ are index in the input sequence), the attention score can be computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0005-13.png)


where **S**[(] _i[l]_[)] _∈_ R _[d]_ denotes the embedding of function symbol _Si_ . **V**[(] _[l]_[)] _∈_ R _[d]_ is obtained by averaging the variable embedding **V** _j_[(] _[l]_[)] and **V** _k_[(] _[l]_[)][.][For] atom with a single variable, the average step can be omitted. _||_ represents the concatenation between feature vectors. **W** is the trainable projection parameters (the same below). 

To embed the in-atom attention, we leverage a score matrix **M**[(] _in[l]_[)] _[∈]_[R][(] _[M]_[+] _[K]_[)] _[×]_[(] _[M]_[+] _[K]_[)][:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0005-16.png)


We define **M**[(] _in[l]_[)][as][a][symmetric][attention][matrix,] thus there also exist **M**[(] _in[l]_[)][(] _[j, i]_[)][=] **[M]**[(] _in[l]_[)][(] _[i, j]_[)][and] **M**[(] _[l]_[)] _in_[(] _[k, i]_[) =] **[ M]**[(] _in[l]_[)][(] _[i, k]_[)][.] 

Cross-atom interaction models the message passing over different atoms. For the same variable _Vp_ and _Vq_ ( _p_ , _q_ are unit index of the input sequence), the attention score is obtained: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0005-19.png)


where **V** _p_[(] _[l]_[)] and **V** _q_[(] _[l]_[)] are the embeddings of two instantiated variables. 

Similar to in-atom attention, we obtain a crossatom score matrix **M**[(] _crs[l]_[)] _[∈]_[R][(] _[M]_[+] _[K]_[)] _[×]_[(] _[M]_[+] _[K]_[)][:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0005-22.png)


Since these two attention matrices only model one-order interaction between related units, the long-distance message passing is limited. Also, we extract the atom based on the explicit logical connectives, it ignores the implicit interactions within the logical text. Therefore, we introduce a diffusion aggregation strategy (Zhao et al., 2021; Liu et al., 2021b) to achieve high-order attention: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0006-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0006-02.png)


where _N_ is the maximum order number, _αi_ and _βi_ are the trade-off coefficients to control the diffusion procedure. In this way, the one-order attention flow can be efficiently diffused to high-order relations. We can update the feature of sequence **H**[(] _seq[l]_[)] _[∈]_ R[(] _[M]_[+] _[K]_[)] _[×][d]_ through joint utilization of these three attention matrices: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0006-04.png)


Within each atom, we aggregate the instantiated variable embedding in the function symbol to acquire a sequence of atom embedding, which can be represented as _{_ **H**[(] _S[l]_ 1[)] _[, ...,]_ **[ H]** _S_[(] _[l] M_[)] _[}]_[.][Next, we define] the reasoning path **H**[(] _p[l]_[)] _[∈]_[R] _[d]_[ embedding as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0006-06.png)


To align the output embedding of the selfattention module, we repeat and stack the reasoning path embedding for _M_ + _K_ times, obtaining the output of path-attention module **H**[(] _PA[l]_[)] _[∈]_[R][(] _[M]_[+] _[K]_[)] _[×][d]_[.] Note that the multi-head strategy is also applied in the path-attention module. 

We obtain the optimized sequence embedding by adding **H**[(] _SA[l]_[)][and] **[ H]**[(] _PA[l]_[)][.][Following the common] practice in the Transformer architecture, we feed the sequence into the feedforward block and obtain the final output **H**[(] _t[l]_[)] of _l[th]_ layer. 

After the respective mean pooling process on **H** _cls_ , **H**[(] _t[L]_[)] and **H**[(] _p[L]_[)][,][the][three][features][are][con-] catenated and projected for the final prediction. 

## **5 Experiments** 

This section provides comparison experiments with other strong baselines on two logical reasoning benchmarks. Extensive ablation studies and generalization evaluations are also followed. 

## **5.1 Datasets and Baselines** 

The main experiments are conducted on two logical reasoning datasets ReClor (Yu et al., 2019) and LogiQA (Liu et al., 2021a). To verify the superiority of PathReasoner, we compare it with strong baselines, including RoBERTa-large (Liu et al., 2019), DAGN (Huang et al., 2021), FocalReasoner (Ouyang et al., 2021), LReasoner (Wang et al., 2022), AdaLoGN (Li et al., 2022), MERIt (Jiao et al., 2022), Logiformer (Xu et al., 2022), as well as LLMs like text-davinci-003, GPT-3.5-turbo and PaLM 2. All the experiments are conducted with a single GPU of Tesla A100. All detailed experimental settings are listed in Appendix E.3. 

## **5.2 Comparison Results** 

The results of comparison experiments are presented in Table 2. Compared with previous SOTA baselines, PathReasoner presents superiority. In ReClor dataset, PathReasoner outperforms all the graph-based methods. Compared with the SOTA method Logiformer, PathReasoner achieves improvements of 2.00% and 0.60% on the validation and test splits respectively. PathReasoner also shows superiority over all sequence-based methods, especially outperforming MERIt by 2.50% on the test split. Importantly, it surpasses human performance, i.e., 64.10% vs 63.00%, which greatly pushes the boundary of machine reasoning. In LogiQA dataset, PathReasoner still shows competitive performances, improving the SOTA results by 2.46% on the test split. PathReasoner demonstrates excellent performance and generalization in logical reasoning, as evidenced by its consistent results across two benchmarks. 

Compared with representative LLMs, PathReasoner exhibits great superiority with a wide margin in the ReClor dataset. Also in the LogiQA dataset, it outperforms both text-davinci-003 and GPT-3.5 with obvious advantages and only falls behind PaLM 2 which is over x1000 in size. 

## **5.3 Ablation Studies** 

Ablation studies for two main parts EPE and RPM in Table 3. For _w/o whole_ of EPE, we remove 

|**Model**||Valid|Test<br>|**ReClor**<br>Test-E|Test-H<br>∆||**LogiQA**<br>Valid<br>Test<br>∆|**LogiQA**<br>Valid<br>Test<br>∆|
|---|---|---|---|---|---|---|---|---|
|Sequence<br>Random<br>Human Performance<br>BERT-Large<br>XLNet-Large<br>RoBERTa-Large<br>LReasoner<br>MERIt†||25.00|25.00|25.00|25.00<br>-||25.00<br>-|25.00<br>-<br>86.00<br>-|
|||-|63.00|57.10|67.20<br>-1.10||||
|||53.80|49.80|72.00|32.30<br>-14.30||34.10|31.03<br>-13.98|
|||62.00|56.00|75.70|40.50<br>-8.10||-|-<br>-|
|||62.60|55.60|75.50|40.00<br>-8.50||35.02|35.33<br>-9.68|
|||66.20|62.40|-|-<br>-1.70||38.10|40.60<br>-4.41|
|||69.40|61.60|79.30|47.80<br>-2.50||39.50|42.40<br>-2.61|
|Graph<br>DAGN<br>FocalReasoner<br>AdaLoGN<br>Logiformer||65.80|58.30|75.91|44.46<br>-5.80||36.87|39.32<br>-5.69|
|||66.80|58.90|77.05|44.64<br>-5.20||41.01|40.25<br>-4.76|
|||65.20|60.20|79.32|45.18<br>-3.90||39.94|40.71<br>-4.30|
|||68.40|63.50|79.09|**51.25**<br>-0.60||42.24|42.55<br>-2.46|
|LLM<br>text-davinci-003_♣_<br>GPT-3.5-turbo_♣_<br>GPT-4-0125-preview<br>PaLMv2_♣_||53.00<br>58.80<br>84.40<br>56.00|-<br>-<br>-<br>-|-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-||-<br>-<br>-<br>-|41.00<br>-<br>40.25<br>-<br>58.37<br>48.00<br>-|
|PathReasoner||**70.40**|**64.10**|**80.91**|50.89<br>-||**43.16**|**45.01**<br>-|



Table 2: Experimental results on ReClor and LogiQA. The percentage signs (%) of accuracy values are omitted. The optimal and sub-optimal results are marked in bold and underlined (comparisons do not include LLMs). The column ∆ presents the improvements of PathReasoner on the test split. _†_ means the utilization of extra data. _♣_ denotes results from (Xu et al., 2023a). 

|**Model**|**ReClor**<br>Valid<br>Test|**LogiQA**<br>Valid<br>Test|
|---|---|---|
|PathReasoner|70.40<br>64.10|43.16<br>45.01|
|_EPE Part_<br>_w/o whole_<br>∆<br>_w/o path flter_<br>∆|67.00<br>60.40|41.16<br>43.01|
||-3.40<br>-3.70|-2.00<br>-2.00|
||68.40<br>62.80|42.70<br>43.78|
||-2.00<br>-1.30|-0.46<br>-1.23|
|_RPM Part_<br>_w/o whole_<br>∆<br>_w/o path attention_<br>∆<br>_w/o in-atom att._<br>∆<br>_w/o cross-atom att._<br>∆<br>_w/o diffusion_<br>∆|63.00<br>56.20|38.40<br>39.17|
||-7.40<br>-7.90|-4.76<br>-5.84|
||67.60<br>60.80|41.94<br>43.16|
||-2.80<br>-3.30|-1.22<br>-1.85|
||70.00<br>62.80|42.09<br>44.85|
||-0.40<br>-1.30|-1.07<br>-0.16|
||67.80<br>62.40|43.63<br>42.70|
||-2.60<br>-1.70|+0.47<br>-2.31|
||69.00<br>61.80|42.70<br>43.63|
||-1.40<br>-2.30|-0.46<br>-1.38|



Table 3: Ablation studies on ReClor and LogiQA. 

the whole part of EPE and only utilize the origin samples for training. The performance witnesses obvious drops of 3.70% and 2.00% on the two datasets respectively. For _w/o path filter_ , we keep all the new paths to generate samples without filtering. The results prove the effectiveness of it with 1.30% and 1.23% gains on the test respectively. For _w/o whole_ of RPM, we ablate the whole RPM and simply leverage the input sequence to predict the answer through a text encoder and a classifier. In this case, the model degenerates to RoBERTa-large baseline with more samples from EPE part. The results prove that the modeling of path significantly enhances the reasoning process. 

To deeply verify modules in RPM, we carry out the following ablation studies. For _w/o path attention_ , we remove the path attention module. The performance gains prove that it is key to RPM part. For _in-atom att._ and _cross-atom att._ , we respectively ablate the attention modeling within atoms and across atoms. The former benefits the ReClor dataset a lot, while the latter is more helpful to the LogiQA dataset. It illustrates that in-atom and cross-atom attention are complementary to each other. For _w/o diffusion_ , we remove the high-order diffusion strategy. Experiments show that the diffusion strategy is also vital to RPM part. 

## **5.4 In-depth Analysis** 

We first analyze the model performances with different lengths of atoms in Fig. 3a. The bars represent the number of samples with different atom numbers, while the lines denote the performances with different atom numbers. For both ReClor and LogiQA datasets, PathReasoner maintains a high performance with moderate scale of atoms, which accounts for most samples in both datasets. Confronted with larger sample sizes, the performances decline. We argue that the gaps have been greatly narrowed with the proposed diffusion strategies, compared with previous models. 

Secondly, we provide an analysis of the impact of the number of new samples. By controlling the maximum scale of new atom combinations, we can generate different numbers of samples. Fig. 3b shows the model performances under various cases, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0008-00.png)


|**Model**|**Dream**<br>Valid<br>Test|**MuTual**<br>R@1|**MuTual**+<br>R@1|
|---|---|---|---|
|RoBERTa-L<br>Logiformer|83.18<br>84.74<br>84.47<br>83.76|87.46<br>88.04|80.47<br>79.68|
|PathReasoner|**85.05**<br>**86.84**|**88.93**|**81.49**|



Table 4: Experiments on model generalization. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0008-03.png)


**----- Start of picture text -----**<br>
(a) Performances with different numbers of atoms.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0008-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0008-05.png)


**----- Start of picture text -----**<br>
(b) Performances with different numbers of new samples.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0008-06.png)


**----- Start of picture text -----**<br>
��� ������������ ��� ������������<br>��� ���������� ��� ����������<br>��� ���<br>��� ���<br>��� ���<br>��� ���<br>��� ���<br>��� ���<br>��� ���� ���� ���� ���� ����� ����� ��� ���� ���� ����� ����� ����� �����<br>������������� �������������<br>(c) Training Efficiency Analysis.<br>���� ����<br>**----- End of picture text -----**<br>


Figure 3: In-depth analysis of the model. 

where the horizontal axis denotes the number of new samples (with & w/o path filter) and the vertical axis is the model performance on the test. On the two datasets, the path filter plays a positive role in reducing redundancy and noise. Additionally, the optimal results are obtained at a moderate scale of new samples, and larger amounts of samples do not always bring gains in performance. 

Thirdly, we discuss the model training efficiency in Fig. 3c. We make the comparison with the previous SOTA Logiformer on ReClor (left) and LogiQA (right). To make a clear illustration, we report the loss curve with steps (truncated at 0.1). From the results, PathReasoner shows faster convergence speed on both ReClor and LogiQA datasets. Detailedly, PathReasoner achieves 1.66x convergence speed than Logiformer on the ReClor dataset, and it has 1.34x speed on the LogiQA dataset. We provide more in-depth experiments in Appendix F. 

## **5.5 Model Generalization** 

PathReasoner is also evaluated on other reasoning tasks to verify the generalization capability in Table 4. The experiments are conducted on Dream (Sun et al., 2019) and MuTual (Cui et al., 2020), which are multi-turn dialogue datasets requiring complex reasoning. We utilize RoBERTa-Large 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0008-12.png)


Figure 4: Two case studies on LogiQA dataset. 

model and the previous SOTA model Logiformer as baselines. Among all comparison metrics, PathReasoner achieves consistent superiority over them. Compared with Logiformer, PathReasoner outperforms it with 3.08% in the test split of Dream, 0.89% of the R@1 metric of MuTual and 1.81% of the R@1 metric of MuTual[+] . It demonstrates that PathReasoner can well generalize to different reasoning tasks. Also, other generalization experiments on EPE module and zero-shot settings are included in Appendix G,H. 

## **5.6 Case Study** 

We provide the analysis for the interpretability of PathReasoner in Figure 4. In the successful case, PathReasoner correctly extracts the variables from the text and forms the reasoning path. In particular, We present the path attention map from RPM part to check the logical perception capability. Firstly, PathReasoner focuses more on the function symbols (e.g., If and Fact) and question sentences 

(i.e., variable _F_ ), with higher attention scores in the map. It verifies that PathReasoner is equipped with the perception of logic and question types. Secondly, the question is to match the logical structure between context and option. The corresponding atoms (e.g., If( _D, E_ ) and If( _J, K_ )) are considered together in the module. It illustrates that PathReasoner is good at understanding the question and reasoning over paths. 

In the failure case, PathReasoner wrongly categorizes the variable with different semantics together to _A_ which leads to the mistake. It demonstrates that the variable extraction in PathReasoner is not good at distinguishing the minor difference, which has space for improvement. 

## **6 Conclusion** 

To tackle the logical data scarcity and weak model perception of logical structures, we propose a new paradigm to model the logical reasoning task by representing each natural sentence as atom form and transforming logical samples into reasoning paths. Based on such unique modeling, an architecture PathReasoner is proposed to address the challenges. It achieves SOTA performances on two logical reasoning datasets. Also, extensive experiments demonstrate the effectiveness of each module and great generalization capability on other complex reasoning scenarios. In the future, we will propose a unified architecture based on PathReasoner to tackle the logical reasoning tasks over different modalities (e.g., images, text, graphs). 

## **7 Acknowledgement** 

This work was supported by National Key Research and Development Program of China (2022YFC3303600), National Natural Science Foundation of China (62137002, 62293550, 62293553, 62293554, 61937001, 62250066, 62176209, 62176207, and 62192781), "LENOVOXJTU" Intelligent Industry Joint Laboratory Project, Natural Science Basic Research Program of Shaanxi (2023-JC-YB-593), the Youth Innovation Team of Shaanxi Universities, XJTU Teaching Reform Research Project "Acquisition Learning Based on Knowledge Forest". 

## **Limitations** 

This paper proposes a novel direction for addressing logical reasoning tasks, which differs from the sequence-based methods and graph-based methods. 

The core of the proposed model is to transform the input text into the form of logical rules with the conjunction of atoms and realize the equivalent extension and path reasoning over it. However, the extraction process of atoms is still very challenging. Although the current algorithm predefines some basic logical relations in advance and achieves great progress, it also requires the help of more comprehensive external logic bases in the future to improve the accuracy of atom extraction. In addition, the logical text in reality often contains noise (e.g., wrong logic). Although this paper has conducted extensive experiments on other reasoning datasets and complex settings to verify the generalization capability, there still remain unsolved on how to promote the models to more complex settings, like multi-modality scenarios. 

## **References** 

- Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , pages 1877–1901. 

- Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. 2020. MuTual: A dataset for multi-turn dialogue reasoning. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , pages 1406–1416. 

- Peter G Hinman. 2018. _Fundamentals of mathematical logic_ . CRC Press. 

- Jie Huang and Kevin Chen-Chuan Chang. 2023. Towards reasoning in large language models: A survey. In _Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 1049–1065. Association for Computational Linguistics. 

- Yinya Huang, Meng Fang, Yu Cao, Liwei Wang, and Xiaodan Liang. 2021. Dagn: Discourse-aware graph network for logical reasoning. In _Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)_ . 

- Fangkai Jiao, Yangyang Guo, Xuemeng Song, and Liqiang Nie. 2022. Merit: Meta-path guided contrastive learning for logical reasoning. In _Findings of the Association for Computational Linguistic (ACL Findings)_ . 

- Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of NAACL-HLT_ , pages 4171–4186. 

- Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ . 

- Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. Race: Large-scale reading comprehension dataset from examinations. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 785– 794. 

- Xiao Li, Gong Cheng, Ziheng Chen, Yawei Sun, and Yuzhong Qu. 2022. Adalogn: Adaptive logic graph network for reasoning-based machine reading comprehension. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)_ . 

- Qika Lin, Jun Liu, Fangzhi Xu, Yudai Pan, Yifan Zhu, Lingling Zhang, and Tianzhe Zhao. 2022. Incorporating context graph with logical reasoning for inductive relation prediction. In _SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 893–903. ACM. 

- Qika Lin, Jun Liu, Lingling Zhang, Yudai Pan, Xin Hu, Fangzhi Xu, and Hongwei Zeng. 2021. Contrastive graph representations for logical formulas embedding. _IEEE Transactions on Knowledge and Data Engineering_ , 35(4):3563–3574. 

- Hanmeng Liu, Ruoxi Ning, Zhiyang Teng, Jian Liu, Qiji Zhou, and Yue Zhang. 2023. Evaluating the logical reasoning ability of chatgpt and gpt-4. _arXiv preprint arXiv:2304.03439_ . 

- Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2021a. Logiqa: a challenge dataset for machine reading comprehension with logical reasoning. In _Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence (IJCAI)_ . 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ . 

- Yonghao Liu, Renchu Guan, Fausto Giunchiglia, Yanchun Liang, and Xiaoyue Feng. 2021b. Deep attention diffusion graph neural networks for text classification. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 8142–8152. 

- Siru Ouyang, Zhuosheng Zhang, and Hai Zhao. 2021. Fact-driven logical reasoning. _arXiv preprint arXiv:2105.10334_ . 

- Yudai Pan, Jun Liu, Lingling Zhang, Tianzhe Zhao, Qika Lin, Xin Hu, and Qianying Wang. 2022. Inductive relation prediction with logical reasoning using contrastive representations. In _Proceedings of the_ 

_2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 4261–4274. 

- Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for squad. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 784–789. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pages 2383–2392. 

- Siva Reddy, Danqi Chen, and Christopher D Manning. 2019. Coqa: A conversational question answering challenge. _Transactions of the Association for Computational Linguistics_ , 7:249–266. 

- Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. 2019. DREAM: A challenge data set and models for dialogue-based reading comprehension. _Transactions of the Association for Computational Linguistics (TACL)_ , pages 217–231. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems_ , pages 5998–6008. 

- Siyuan Wang, Wanjun Zhong, Duyu Tang, Zhongyu Wei, Zhihao Fan, Daxin Jiang, Ming Zhou, and Nan Duan. 2022. Logic-driven context extension and data augmentation for logical reasoning of text. In _Findings of the Association for Computational Linguistics (ACL Findings)_ . 

- Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In _International Conference on Learning Representations_ . 

- Fangzhi Xu, Qika Lin, Jiawei Han, Tianzhe Zhao, Jun Liu, and Erik Cambria. 2023a. Are large language models really good logical reasoners? a comprehensive evaluation from deductive, inductive and abductive views. _arXiv preprint arXiv:2306.09841_ . 

- Fangzhi Xu, Jun Liu, Qika Lin, Yudai Pan, and Lingling Zhang. 2022. Logiformer: A two-branch graph transformer network for interpretable logical reasoning. In _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)_ . 

- Fangzhi Xu, Jun Liu, Qika Lin, Tianzhe Zhao, Jian Zhang, and Lingling Zhang. 2023b. Mind reasoning manners: Enhancing type perception for generalized zero-shot logical reasoning over text. _arXiv preprint arXiv:2301.02983_ . 

- Fangzhi Xu, Zhiyong Wu, Qiushi Sun, Siyu Ren, Fei Yuan, Shuai Yuan, Qika Lin, Yu Qiao, and Jun Liu. 2023c. Symbol-llm: Towards foundational symbolcentric interface for large language models. _arXiv preprint arXiv:2311.09278_ . 

- Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. 2019. Xlnet: generalized autoregressive pretraining for language understanding. In _Proceedings of the 33rd International Conference on Neural Information Processing Systems_ , pages 5753–5763. 

- Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2019. Reclor: A reading comprehension dataset requiring logical reasoning. In _International Conference on Learning Representations (ICLR)_ . 

- Jialin Zhao, Yuxiao Dong, Ming Ding, Evgeny Kharlamov, and Jie Tang. 2021. Adaptive diffusion in graph neural networks. In _Advances in Neural Information Processing Systems (NeurIPS)_ , pages 23321– 23333. 

- Wanjun Zhong, Siyuan Wang, Duyu Tang, Zenan Xu, Daya Guo, Jiahai Wang, Jian Yin, Ming Zhou, and Nan Duan. 2021. Ar-lsat: Investigating analytical reasoning of text. _arXiv preprint arXiv:2104.06598_ . 

- Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. 2020. Graph neural networks: A review of methods and applications. _AI open_ , 1:57–81. 

## **A Pilot Experiments** 

In this section, we provide the detailed pilot experiments mentioned in Fig. 1 of the main paper. For the model prediction consistency test, we equally replace the explicit logical connectives in a part of the samples on ReClor. The differences in performances are presented in Table 5. 

Table 5: Pilot experiments on prediction consistency. 

|**Model**|**Origin**<br>**Replace**<br>∆|
|---|---|
|BERT-L<br>RoBERTa-L|38.50<br>30.00<br>-8.50<br>55.00<br>48.50<br>-6.50|
|PathReasoner|62.50<br>61.00<br>-1.50|



It can be seen that current PLMs fail to maintain equal predictions on samples with the same logical semantics. It proves the motivation of the proposed method. Also, we provide the performances of PathReasoner in the same setting as the pilot experiments. Our model largely improves the prediction consistency, and only fails in 1.50% of the cases. It illustrates the robustness of PathReasoner in logic. 

In addition, we conducted experiments on the model perception of logical connectives. By adding, deleting, or modifying the explicit logical connectives on some samples, we randomly break the original semantics of the context. We report the ratio of samples that fail to follow the logical changes. It tests the sensitivity of the model for capturing the logical relations. Results are shown in Table 6. 

Table 6: Pilot experiments on model perception of explicit logical connectives. 

|**Model**|**Ratio**|
|---|---|
|BERT-L<br>RoBERTa-L|29.30%<br>21.63%|
|PathReasoner|71.95%|



From the results, current PLMs are not always sensitive to the changes of logical connectives. BERT and RoBERTa can merely distinguish 29.30% and 21.63% of changes respectively. Therefore, it is worth considering enhancing the logic modeling for the language models, which supports our motivations. Also, we report the performance of PathReasoner on the last row of the table. Our model shows great superiority on enhancing the model perception of explicit logical connectives, being sensitive to 71.95% of the cases. It well verifies and supports our motivations. 

## **B Key Questions for Extraction Process** 

The whole extraction process leads to several key questions: 

**(1) Scenario coverage.** Our predefined rules are relatively complete, and have covered extensive cases in syntax (guided by experts). We include over 100 instantiated function symbols (curated from NLTK) and it can cover most logical scenarios (details in Appendix D). Therefore, it can ensure the wide coverage of logical scenarios. Beyond that, we also include the _Fact_ category of function symbols. It can be adapted to facts that do not have obvious logic. To sum up, our heuristic rules can extend to any kind of text in theory. 

For an example out of the logical domain, the input is factual paragraph X, which consists of sentences A, B, C, and D. Our method adapts to such a scenario, and it can output _Fact_ ( _A_ ) _∧ Fact_ ( _B_ ) _∧ Fact_ ( _C_ ) _∧ Fact_ ( _D_ ). 

**(2) Extraction accuracy.** Based on the above descriptions, our method can cover any kind of text in theory. We randomly select 30 paragraphs, resulting in 148 pieces of sentences. We manually label the extraction accuracy of each sentence. To make a comparison, we also prompt GPT-4 (instruction+predefined function symbols + atom form+4shot examples) to finish this process. The results are listed in Table 7. 

Table 7: Experiments on the extraction accuracy. 

||**Ours**|**GPT-4**|**LLaMA-2-Chat**|
|---|---|---|---|
|Atom Acc|95.27|91.22|8.11|



## **C Distinction of Our Logical Forms** 

As some examples presented, our predefined logical forms are similar to first-order logic (FOL) and propositional logic. But our forms are more suitable for the scenarios in the following aspects. 

**(1) Customized function symbols.** We define four types of function symbols and they are effective in equivalent transformation. The general FOL and propositional logic can not satisfy our customized requirements. 

**(2) Perception of sentence-level logic.** In logical reasoning scenarios, rich logic exists more at the sentence level, thus we transform each sentence into an atom. However, FOL and propositional 

logic are conditioned at the entity level or span level, which is more fine-grained. They are not necessarily effective in capturing logic. 

## **D Statistics of Function Symbols** 

In this section, we present the statistics of the logical connectives in two logical reasoning datasets ReClor and LogiQA. It will provide intuitive proof of the necessity and rationality of the function symbol categories. 

Figure 5a presents the statistics of function symbols (i.e., _Cause_ , _SA_ , _NA_ , _Fact_ ) in the context of two benchmarks. The outer cycle represents the train split, the middle one is the validation split and the inner one is the test split. In the ReClor dataset, nearly 40% of atoms are non-fact, which contain explicit logical connectives (i.e., _Cause_ , _SA_ , _NA_ ). Among them, _Cause_ relations are the majority. In the LogiQA dataset, the ratio of the logical function symbols drops a lot, but it still accounts for about 20%. 

Figure 5b shows the statistics of logical samples. We categorize the samples with any one of the three logical function symbols into _has logic_ . Similar, we include samples with _Cause_ , _SA_ and _NA_ to _has Cause_ , _has SA_ and _has NA_ respectively. In ReClor, nearly 70% of the samples have explicit logical connectives. Also, over 60% of samples contain _Cause_ atoms. In LogiQA, samples with logical connectives account for 50%. The ratio of samples containing _Cause_ atoms drops to about 35% while the ratio of samples with _NA_ atoms increases. 

The above analysis illustrates that the two benchmark datasets are abundant in logical connectives. Thus, the modeling of logical atoms is of great necessity. 

## **E Experimental Settings** 

## **E.1 Benchmarks and other Datasets** 

ReClor and LogiQA are two representative datasets for the logical reasoning task. The details are presented as follows. 

**ReClor** (Yu et al., 2019) includes 6,138 samples total with 4,638 training samples, 500 validation samples, and 1,000 samples for test. All of them are collected from some standardized graduate admission examinations. To discriminate the difficulty of the questions, the test split is divided into _Test-E_ and _Test-H_ , where the former represents the easy version of the test samples and the latter denotes the harder parts. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0013-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0013-11.png)


**----- Start of picture text -----**<br>
(a) Statistics of function symbols in train (outer cycle), valida-<br>tion (middle cycle) and test (inner cycle) splits.<br>(b) Statistics of logical samples.<br>**----- End of picture text -----**<br>


Figure 5: Statistics of logical reasoning benchmarks. 

**LogiQA** (Liu et al., 2021a) includes 8,678 samples sourced from National Civil Servants Examinations of China. It is further split into the training set, development set, and test set, with 7,376, 651, and 651 samples respectively. 

Also, to verify the model generalization capability, we employ two dialogue datasets involving complex reasoning, which are Dream and MuTual. Also, we exploit the zero-shot logical reasoning capability of the proposed model on the recently proposed ZsLR benchmark. The details are presented below. 

**Dream** (Sun et al., 2019) contains 6,444 multiple choice questions, sourced from English-as-aforeign-language examinations. The samples are split into train, development and test sets with 3,869, 1,288 and 1,287 samples respectively. We report the exact match metric on both validation and test splits. 

**MuTual** (Cui et al., 2020) consists of 8,860 questions, divided into 7,088 training samples, 886 validation samples, 886 test samples. It is modified from Chinese high school English listening comprehension test data. Also, MuTual _[plus]_ dataset is proposed to test whether the model is capable of selecting a safe response when necessary. Since the test split of MuTual is not made public, we only report the R@1 metric (recall at position one) on the validation set. 

**ZsLR** (Xu et al., 2023b) includes 6 zero-shot splits modified from ReClor dataset. Since the dataset contains 17 reasoning types in total, some types of samples are classified as seen types during training. For the test, it defines two metrics, one is _Test-All_ 

which tests on all the types of samples, and another is _Test-Unseen_ which only tests on the unseen parts of types. 

Table 8: Categorization of recent works on logical reasoning task. ‘DA’ denotes the data augmentation strategy. ‘ _†_ ’ denotes the utilization of extra data. 

|**Model**|**Sequence**<br>**Graph**<br>**Path/Rule**|**DA**|
|---|---|---|
|ReClor<br>DAGN<br>FocalReasoner<br>LReasoner<br>AdaLoGN<br>MERIt †<br>Logiformer|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|✓<br>✓|
|PathReasoner|✓|✓|



## **E.2 Baselines** 

In this paper, we compare PathReasoner with all the previous methods of the logical reasoning task, including the SOTA model Logiformer. There methods can be categorized into sequence-based and graph-based, shown in Table 8. 

**(1) Random** . The results are obtained from the random predictions. 

**(2) RoBERTa-Large** (Liu et al., 2019). The trained language model RoBERTa is employed as the text encoder to obtain the predictions. It is also the same with the baselines of BERT-Large (Kenton and Toutanova, 2019) and XLNet-Large (Yang et al., 2019). 

**(3) Human Performance** (Yu et al., 2019; Liu et al., 2021a). The performances are averaged from the scores of some graduate students on the test split. 

**(4) DAGN** (Huang et al., 2021). It is the first graph-based work to tackle the logical reasoning task. It splits the text into nodes and leverages the graph neural networks to reason over the chain-type graph. 

**(5) FocalReasoner** (Ouyang et al., 2021). It focuses on the facts within the context and it extracts all the fact units to form a supergraph for reasoning. **(6) LReasoner** (Wang et al., 2022). It proposes to leverage the defined rules (e.g., De Morgan’s Laws) to extend the context. In addition, it employs data augmentation strategies (e.g., contrastive learning) to improve the diversity of the samples. 

**(7) MERIt** (Jiao et al., 2022). It proposes a metapath guided strategy to conduct the pretraining on the external corpus. The pre-trained module is further verified based on some off-the-shelf SOTA 

methods. For a fair comparison, we directly derive the results of MERIt with the RoBERa-Large backbones from the original paper. 

**(8) AdaLoGN** (Li et al., 2022). It first builds a text graph based on the off-the-shelf method and models it in an adaptive neuro-symbolic system. **(9) Logiformer** (Xu et al., 2022). It models the context from the perspective of both logic and syntax, building a causal graph and a co-occurrence graph. Specifically, it reasons on the graph transformer networks with biased attention. 

Additionally, we include the following representative large language models to make the comparisons. 

**(10) text-davinci-003** . It was created by OpenAI, of which the training data was collected up to Sep. 2021. The size of text-davinci-003 is 175B. 

**(11) GPT-3.5-turbo** . It is also from OpenAI and the training corpus is collected up to June. 2021. GPT-3.5-turbo is of the same size as text-davinci003. 

**(12) PaLM 2** . It was created by Google. It has a larger size than the above two LLMs, which is 540B. 

The results of the three LLMs on the logical reasoning benchmarks are collected from (Xu et al., 2023a). 

## **E.3 Implementation Details** 

In the implementation, to make a fair comparison, we employ the RoBERTa-large (Liu et al., 2019) model with the hidden size of 1024 as the encoder of text. We utilize the Adam (Kingma and Ba, 2014) for the optimization. Also, we set different hyper-parameters for the two logical reasoning datasets respectively. We tune some of the hyperparameters for the optimal within a scope. Table 9 presents the detailed information. 

The listed hyper-parameters belong to three parts: general settings, equivalent path extension module, and reasoning path modeling module. Considering the calculation cost, we do not utilize the grid search strategy, instead, we sequentially search the hyper-parameters for the optimal. For the reasoning path modeling module, we select the maximum diffusion order _N_ to be 2. Therefore, there only exist four diffusion trade-off co-efficient _α_ 1, _α_ 2, _β_ 1, _β_ 2, which satisfy _α_ 1 + _α_ 2 = 1 and _β_ 1 + _β_ 2 = 1. So we only list the tuning details of in-atom diffusion trade-off _α_ 1 and cross-atom diffusion trade-off _β_ 1. 

Table 9: The details of tuned hyper-parameters on the two logical reasoning benchmarks. 

|**Name of Parameter**|**ReClor**<br>Search Scope<br>Best|**LogiQA**<br>Search Scope<br>Best|
|---|---|---|
|_General Settings_<br>number of epoch<br>max sequence length<br>learningrate|{10,12,15,20}<br>20<br>{384,512}<br>384<br>{4e-6,5e-6,6e-6}<br>5e-6|{10,12,15,20}<br>20<br>{384,512}<br>512<br>{4e-6,5e-6,6e-6}<br>5e-6|
|_Equivalent Path Extension_<br>path flter threshold_ε∗_|{0.5,0.8,0.9}<br>0.9|{0.5,0.8,0.9}<br>0.9|
|_Reasoning Path Modeling_<br>number of layer<br>number of head<br>max diffusion order_N_<br>in-atom diffusion_α_1<br>cross-atom diffusion_β_1<br>leaky rate|{3,4,5,6}<br>3<br>{4,8}<br>4<br>{1,2,3}<br>2<br>{0,0.1,0.2,0.3,0.4}<br>0.2<br>{0,0.1,0.2,0.3,0.4}<br>0<br>{0.01,0,02,0.03,0.04}<br>0.02|{3,4,5,6}<br>3<br>{4,8}<br>4<br>{1,2,3}<br>2<br>{0,0.1,0.2,0.3,0.4}<br>0.1<br>{0,0.1,0.2,0.3,0.4}<br>0.1<br>{0.01,0,02,0.03,0.04}<br>0.02|



## **F In depth Analysis** 

In this section, we provide more experiments to analyze the model performances. 

## **F.1 Model Performance on Multiple Logics** 

In the category of function symbols, we take _Cause_ , _SA_ , _NA_ and _Fact_ into consideration. Among them, the first three represent the logical relations (nonfact) while the last one represents the factual expression. Therefore, we give an analysis of how our model performs on these factual or logical samples. We test the model performance on three types of samples: (1) Fact, where all atoms are factual; (2) Simple Logic, where there only exists one category of logical function symbols in each sample; (3) Complex Logic, where multiple categories of logical function symbols are included in one sample. Table 10 presents the results of PathReasoner on the above settings, compared with RoBERTaLarge model and Logiformer. Since ReClor does not make the test split public, we only report the results on the validation split. 

For simple logic samples and complex logic samples, PathReasoner also shows the superiority of 1.85% and 1.23% over Logiformer respectively. It demonstrates the competitiveness of PathReasoner in logical perception and reasoning. Meanwhile, we witness that PathReasoner does excellent in capturing simple logic and maintaining factual reasoning, but there still exists space for improvement on the complex logic. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering_images/acl_2024_PathReasoner_Modeling_Reasoning_Path_with_Equivalent_Extension_for_Logical_Question_Answering.pdf-0015-07.png)


**----- Start of picture text -----**<br>
�� �� �� �� �� ��<br>����� ����� ���������� ����� ����� ��������������� �����<br>���������� ��������������� ��������������� ����� �����<br>����� ���������� ���������� ���������� ����� ����� �����<br>���������� ����� ����� ����� ����� ����� ����� ����� �����<br>���������� ����� ����� ����� ����� ����� ���������������<br>� ��� ��� ��� ��� � ��� ��� ��� ���<br>β1 β1<br>(a) Validation split. (b) Test split.<br>� �<br>��� ���<br>α1 ��� α1 ���<br>��� ���<br>��� ���<br>**----- End of picture text -----**<br>


Figure 6: Analysis of high-order diffusion strategy. 

Table 10: Experiments on multiple logics on ReClor. 

|**Model**|**Factual**|**Simple**|**Complex**|
|---|---|---|---|
|RoBERTa-L<br>Logiformer|67.65<br>67.65|65.56<br>71.48|56.79<br>63.58|
|PathReasoner|**72.06**|**73.33**|**64.81**|



For factual types of samples, PathReasoner achieves 4.41% gains over the baselines. We argue that previous method like Logiformer focuses too much on the capture of logical relations but fails to better generalize to the fact-only samples. PathReasoner leverages the atom form to represent both the logical content and the factual content, thus it can also improve the performances on factual samples. 

## **F.2 Model Performance on Different Reasoning Types** 

In the ReClor dataset, the samples are divided into 17 reasoning types. Table 11 gives in-depth model performances on different reasoning types. Limited by space, we only present 11 types in the table. From the results, PathReasoner performs better in most cases. Specially, for _IF_ , _MF_ and _MS_ , PathReasoner achieves obvious superiority. Considering that these reasoning types require the perception of logical structures, the gains in performance prove the effectiveness of PathReasoner. 

Table 11: The details of ReClor Test Split on different reasoning types. **NA** : Necessary Assumption, **S** :Strengthen, **W** :Weaken, **E** :Evaluation, **I** :Implication, **ER** :Explain or Resolve, **T** :Technique, **IF** :Identify a Flaw, **MF** :Match Flaws, **MS** :Match the Structure, **O** :Others. 

|**Model**|**NA**<br>**S**<br>**W**<br>**E**<br>**I**<br>**ER**<br>**T**<br>**IF**<br>**MF**<br>**MS**<br>**O**|
|---|---|
|PathReasoner|74.56<br>62.77<br>59.29<br>76.92<br>52.17<br>67.86<br>83.33<br>67.52<br>58.06<br>83.33<br>67.12|
|Logiformer<br>∆|74.56<br>64.89<br>55.75<br>76.92<br>45.65<br>61.90<br>66.67<br>58.12<br>45.16<br>66.67<br>60.27|
||-<br>-2.12<br>+3.54<br>-<br>+6.52<br>+5.96<br>+6.66<br>+9.40<br>+12.90<br>+6.66<br>+6.85|
|RoBERTa-L<br>∆|71.05<br>61.70<br>47.79<br>69.23<br>39.13<br>58.33<br>52.78<br>61.54<br>45.16<br>56.67<br>52.05|
||+3.51<br>+1.07<br>+11.50<br>+7.69<br>+13.04<br>+9.53<br>+30.55<br>+5.98<br>+12.90<br>+16.66<br>+15.07|



Table 12: Experimental results on 6 zero-shot logical reasoning splits. _T-A_ and _T-U_ denote the abbreviations of the metrics _Test-All_ and _Test-Unseen_ respectively. 

|**Model**||**v1**<br>T-A<br>T-U|**v1**<br>T-A<br>T-U||**v2**<br>T-A<br>T-U|**v2**<br>T-A<br>T-U||**v3**<br>T-A<br>T-U|**v3**<br>T-A<br>T-U||**v4**<br>T-A<br>T-U|**v4**<br>T-A<br>T-U||**v5**<br>T-A<br>T-U|**v5**<br>T-A<br>T-U||**v6**<br>T-A<br>T-U|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|BERT-Large<br>RoBERTa-Large<br>DAGN<br>LReasoner<br>Logiformer<br>TaCo||38.00<br>47.70<br>49.20<br>46.90<br>43.50<br>52.20|34.36<br>39.47<br>41.37<br>40.60<br>39.31<br>**47.51**||42.00<br>50.60<br>52.70<br>50.20<br>54.80<br>**55.80**|33.39<br>39.90<br>43.56<br>43.49<br>46.30<br>**48.79**||37.50<br>46.10<br>49.60<br>48.40<br>48.80<br>52.20|31.61<br>40.58<br>39.73<br>42.76<br>42.24<br>44.26||38.00<br>50.40<br>52.50<br>49.20<br>52.10<br>54.70|33.26<br>42.45<br>44.51<br>44.12<br>44.85<br>**49.89**||29.60<br>53.00<br>52.40<br>51.90<br>52.10<br>56.00|28.02<br>43.66<br>42.63<br>42.02<br>40.88<br>46.67||28.80<br>32.24<br>49.90<br>50.92<br>48.50<br>49.15<br>46.30<br>44.93<br>51.50<br>51.44<br>54.70<br>**55.17**|
|PathReasoner||**52.70**|45.87||55.10|44.01||**52.20**|**45.43**||**56.60**|49.20||**57.20**|**47.43**||**54.90**<br>54.28|



## **F.3 High-order Diffusion Strategy Analysis** 

In the implementation, we set the maximum order of diffusion to 2, that is _N_ = 2. Therefore, we employ two trade-off coefficients _αi_ and _βi_ to control the diffusion procedure. We search _αi_ and _βi_ from the set of {0, 0.1, 0.2, 0.3, 0.4} and report the results on the test split of ReClor in Figure 6. From the results, PathReasoner achieves the optimal simultaneously on both the validation and test splits. 

## **G Generalization of Equivalent Path Extension Module** 

Beyond the main experiments on logical reasoning benchmarks, generalization experiments (Table 4) and zero-shot settings (Table 6), we add a simple experiment with our proposed equivalent path extension (EPE). The aim is to achieve a **plugin-and-play** function to augment the training of LLMs. In detail, we randomly sample from the Flan collection (Wei et al.), leading to 80K original instruction-following samples. Then, we apply EPE to generate equivalent instruction samples. These augmented samples are leveraged to tune LLaMA-2-Chat (7B). The test experiments on MMLU (57 tasks) and BigBenchHard (21 tasks) are presented in Table 13. 

With the EPE augmentation process, the tuned LLaMA-2-Chat can witness significant performance improvements, compared with two base- 

Table 13: Experiments on the generalization capability of equivalent path extension module. 

|**Model**|**MMLU**<br>**BBH**|
|---|---|
|LLaMA-2-Chat<br>LLaMA-2-Chat + Flan<br>LLaMA-2-Chat + EPE + Flan|45.78<br>35.01<br>46.94<br>36.99<br>**48.75**<br>**38.96**|



lines: one is LLaMA-2-Chat, and another is LLaMA-2-Chat tuned on sampled Flan collection. Such findings largely expand the application scope of PathReasoner, especially in empowering the training of off-the-shelf LLMs. 

## **H Model Generalization on Zero-shot Logical Reasoning Settings** 

Previous work (Xu et al., 2023b) argued that the ideal full-data setting is not sufficient to test the logical reasoning performances and has proposed a new benchmark for generalized zero-shot logical reasoning (named ZsLR). To verify the model generalization on the zero-shot settings, we conduct experiments on ZsLR and compare with several SOTA baselines. The results are shown in Table 12. 

From the results of 6 splits, PathReasoner is competitive on the majority of the cases compared with TaCo and Logiformer. For split v1, v3, v4, v5 and v6, PathReasoner outperforms all the strong baselines on the metric of _Test-All_ , which verifies the great generalization ability on both seen and unseen 

types of samples. Compared with the full-data setting SOTA model Logiformer, PathReasoner shows obvious superiority on all the splits and all the test metrics. The great advantages uncover the huge potential of modeling reasoning paths for the logical text, which improves the extensibility and generalization of the model. Also, it is worth noticing that there still exists space for improvement on the unseen types of samples, especially on the split of v2, v4, and v6. 

## **I Restatement of Our Key Novelty** 

We will clarify the obvious differences of our method compared with previous works (especially, the graph-based method). It can be divided into three points. 

**Extraction strategy.** Transforming the natural language into units is a common method in the reasoning field. However, we largely differ in the definition of relationships. Previous works only limit to a subset of relation words. For example, Logiformer only attends to causal relations as the connectives. It is sufficient for evaluations (see Appendix D), which overfit the logical reasoning benchmarks. However, our definition of function symbols is different from previous works, and our coverage is broad enough (over 100 relation words). Therefore, our method can be extended to other scenarios, which have been verified with generalization experiments and zero-shot settings. 

ods ignore logical structures but can handle longdistance dependency with Transformer structure. Graph-based methods usually rely on GNN-style modules to update the features, but lack the extensibility to larger context and fine-grained modeling within each unit (Logiformer attempts to solve it through attention bias, but still limits to a coarse level). In our path attention module, the advantages of sequence- and graph-based methods are inherited. It further achieves differentiable and interpretable reasoning (see Case Study). 

To sum up, the distinctions between PathReasoner and other methods are significant. Also, we would like to emphasize that the distinction of PathReasoner does not only benefit the logical reasoning benchmarks, which previous works are limited to. PathReasoner indeed shows strong generalization capability and plug-in-and-play property (see Appendix G for details). 

**Flexible extension strategy.** Benefiting from the distinctive definition of function symbols, we can formulate the context into the conjunction of atoms (i.e., reasoning paths). Therefore, we can easily conduct the equivalent path extension to derive new combinations of atoms. This advantage is distinctive from other works. It is also one of our main contributions. 

In some graph-based methods, the text graph is updated to capture new relations along with the message-passing process. The whole process is extremely time-consuming, which is the main shortcoming. Our method actually decouples the dynamic extension process with the formulation of atoms and paths. It augments data diversity and improves training efficiency. 

## **Incoporated advantages in the path attention** 

**module.** In fact, the path attention module combines the advantages of sequence-based and graphbased methods. Previous sequence-based meth- 

