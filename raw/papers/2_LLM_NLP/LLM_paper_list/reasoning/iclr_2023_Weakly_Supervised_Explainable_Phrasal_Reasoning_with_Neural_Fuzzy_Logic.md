Published as a conference paper at ICLR 2023 

## WEAKLY SUPERVISED EXPLAINABLE PHRASAL REASONING WITH NEURAL FUZZY LOGIC 

**Zijun Wu** _[∗]_[1] **, Zi Xuan Zhang** _[∗]_[1] **, Atharva Naik** _[†]_[2] **, Zhijian Mei**[1] **, Mauajama Firdaus**[1] **, Lili Mou**[1] 1Dept. Computing Science & Alberta Machine Intelligence Institute (Amii), University of Alberta 2Carnegie Mellon University 

{zijun4, zixuan7, zimei1}@ualberta.ca, arnaik@cs.cmu.edu, {mauzama.03, doublepower.mou}@gmail.com 

> _∗_ Equal contribution, _†_ Work done during the internship at UofA/Amii 

## ABSTRACT 

Natural language inference (NLI) aims to determine the logical relationship between two sentences, such as Entailment, Contradiction, and Neutral. In recent years, deep learning models have become a prevailing approach to NLI, but they lack interpretability and explainability. In this work, we address the explainability of NLI by weakly supervised logical reasoning, and propose an Explainable Phrasal Reasoning (EPR) approach. Our model first detects phrases as the semantic unit and aligns corresponding phrases in the two sentences. Then, the model predicts the NLI label for the aligned phrases, and induces the sentence label by fuzzy logic formulas. Our EPR is almost everywhere differentiable and thus the system can be trained end to end. In this way, we are able to provide explicit explanations of phrasal logical relationships in a weakly supervised manner. We further show that such reasoning results help textual explanation generation.[1] 

## 1 INTRODUCTION 

Natural language inference (NLI) aims to determine the logical relationship between two sentences (called a _premise_ and a _hypothesis_ ), and target labels include Entailment, Contradiction, and Neutral (Bowman et al., 2015; MacCartney & Manning, 2008). Figure 1 gives an example, where the hypothesis contradicts the premise. NLI is important to natural language processing, because it involves logical reasoning and is a key problem in artificial intelligence. Previous work shows that NLI can be used in various downstream tasks, such as information retrieval (Karpukhin et al., 2020) and text summarization (Liu & Lapata, 2019). 

In recent years, deep learning has become a prevailing approach to NLI (Bowman et al., 2015; Mou et al., 2016; Wang & Jiang, 2016; Yoon et al., 2018). Especially, pretrained language models with the Transformer architecture (Vaswani et al., 2017) achieve state-of-the-art performance for the NLI task (Radford et al., 2018; Zhang et al., 2020). However, such deep learning models are black-box machinery and lack interpretability. In real applications, it is important to understand how these models make decisions (Rudin, 2019). 

Several studies have addressed the explainability of NLI models. Camburu et al. (2018) generate a textual explanation by sequence-to-sequence supervised learning, in addition to NLI classification; such an approach is multi-task learning of text classification and generation, which does not perform reasoning itself. MacCartney et al. (2008) propose a scoring model to align related phrases; Parikh et al. (2016) and Jiang et al. (2021) propose to obtain alignment by attention mechanisms. However, they only provide correlation information, instead of logical reasoning. Other work incorporates upward and downward monotonicity entailment reasoning for NLI (Hu et al., 2020; Chen et al., 2021), but these approaches are based on hand-crafted rules (e.g., _every_ downward entailing _some_ ) and are restricted to Entailment only; they cannot handle Contradiction or Neutral. 

In this work, we address the explainability for NLI by weakly supervised phrasal logical reasoning. Our goal is to explain NLI predictions with phrasal logical relationships between the premise and 

> 1Code and resources available at https://github.com/MANGA-UOFA/EPR 

1 

Published as a conference paper at ICLR 2023 

hypothesis. Intuitively, an NLI system with an explainable reasoning mechanism should be equipped with the following functionalities: 

1. The system should be able to detect corresponding phrases and tell their logical relationship, e.g., _several men_ contradicting _one man_ , but _pull in a fishing net_ entailing _holding the net_ (Figure 1). 

2. The system should be able to induce sentence labels from phrasal reasoning. In the example, the two sentences are contradictory because there exists one contradictory phrase pair. 

3. More importantly, such reasoning should be trained in a weakly supervised manner, i.e., the phraselevel predictions are trained from sentence labels only. Otherwise, the reasoning mechanism degrades to multi-task learning, which requires massive fine-grained human annotations. 

To this end, we propose an Explainable Phrasal Reasoning (EPR) approach to the NLI task. Our model obtains phrases as semantic units, and aligns corresponding phrases by embedding similarity. Then, we predict the NLI labels (namely, Entailment, Contradiction, and Neutral) for the aligned phrases. Finally, we propose to induce the sentence-level label from phrasal labels in a fuzzy logic manner (Zadeh, 1988; 1996). Our model is differentiable, and the phrasal reasoning component can be trained 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0002-06.png)


**----- Start of picture text -----**<br>
Input:<br>Premise: Several men helping each other pull in a fishing net.<br>Hypothesis: There is one man holding the net.<br>Sentence-Level Prediction:<br>Entailment Contradiction Neutral<br>Phrase-Level Reasoning:<br>Entailment Contradiction Neutral Unaligned<br>**----- End of picture text -----**<br>


Figure 1: The natural language inference (NLI) task and desired phrasal reasoning. 

with the weak supervision of sentence NLI labels. In this way, our EPR approach satisfies all the desired properties mentioned above. 

In our experiments, we developed a comprehensive methodology (data annotation and evaluation metrics) to quantitatively evaluate phrasal reasoning performance, which has not been accomplished in previous work. We extend previous studies and obtain plausible baseline models. Results show that our EPR yields a much more meaningful explanation regarding _F_ scores against human annotation. 

To further demonstrate the quality of extracted phrasal relationships, we feed them to a textual explanation model. Results show that our EPR reasoning leads to an improvement of 2 points in BLEU scores, achieving a new state of the art on the e-SNLI dataset (Camburu et al., 2018). 

Our contributions are summarized as follows: 

1. We formulate a phrasal reasoning task for natural language inference (NLI), addressing the interpretability of neural models. 

2. We propose an EPR model that induces sentence-level NLI labels from explicit phrasal logical labels by neural fuzzy logic. EPR is able to perform reasoning in a weakly supervised way. 

3. We annotated phrasal logical labels and designed a set of metrics to evaluate phrasal reasoning. We further use our reasoning results to improve textual explanation generation. Our code and annotated data are released for future studies. 

To the best of our knowledge, we are the first to develop a weakly supervised phrasal reasoning model for the NLI task. 

## 2 RELATED WORK 

**Natural Language Inference.** MacCartney & Manning (2009) propose seven natural logic relations in addition to Entailment, Contradiction, and Neutral. MacCartney & Manning (2007) also distinguish upward entailment ( _every mammal_ upward entailing _some mammal_ ) and downward entailment ( _every mammal_ downward entailing _every dog_ ) as different categories. Manually designed lexicons and rules are used to interpret Entailment in a finer-grained manner, such as downward and upward entailment (Hu et al., 2020; Chen et al., 2021). Feng et al. (2020) apply such natural logic to NLI reasoning at the word level; however, our experiments will show that their word-level treatment is not an appropriate granularity, and they fail to achieve meaningful reasoning performance. 

The above reasoning schema focuses more on the quantifiers of first-order logic (Beltagy et al., 2016). However, the SNLI dataset (Bowman et al., 2015) we use only contains less than 5% samples with explicit quantifiers, and the seven-category schema complicates reasoning in the weakly supervised 

2 

Published as a conference paper at ICLR 2023 

setting. Instead, we adopt three-category NLI labels following the SNLI dataset. Our focus is entity-based reasoning, and the treatment of quantifiers is absorbed into phrases. 

We also notice that previous work lacks explicit evaluation on the reasoning performance for NLI. For example, the SNLI dataset only provides sentence-level labels. The HELP (Yanaka et al., 2019a) and MED (Yanaka et al., 2019b) datasets concern monotonicity inference problems, where the label is also at the sentence level; they only consider Entailment, ignoring Contradiction and Neutral. Thus, we propose a comprehensive framework for the evaluation of NLI reasoning. 

**e-SNLI.** Camburu et al. (2018) propose the e-SNLI task of textual explanation generation and use LSTM as a baseline. Kumar & Talukdar (2020) propose the NILE approach, using multiple decoders to generate explanations for all E, C, and N labels, and then predicting which to be selected. Zhao & Vydiswaran (2021) propose the LIREx approach, using additionally annotated rationales for explanation generation. Narang et al. (2020) finetune T5 with multiple explanation generation tasks. Although these systems can generate explanations, the nature of such finetuning approaches renders the explanation generator _per se_ unexplainable. By contrast, we design a textual explanation generation model that utilizes our EPR’s phrasal reasoning, obtained in a weakly supervised manner. 

**Neuro-Symbolic Approaches.** In recent years, neuro-symbolic approaches have attracted increasing interest in the AI and NLP communities for interpreting deep learning models. Typically, these approaches are trained by reinforcement learning or its relaxation, such as attention and Gumbelsoftmax (Jang et al., 2017), to reason about certain latent structures in a downstream task. 

For example, Lei et al. (2016) and Liu et al. (2018) extract key phrases or sentences for a text classification task. Lu et al. (2018) extract entities and relations for document understanding. Liang et al. (2017) and Mou et al. (2017) perform SQL-like execution based on input text for semantic parsing. Xiong et al. (2017) hop over a knowledge graph for reasoning the relationships between entities. Li et al. (2019) and Deshmukh et al. (2021) model symbolic actions for unsupervised syntactic structure induction. In the vision domain, Mao et al. (2019) propose a neuro-symbolic approach to learn visual concepts. Our work addresses logical reasoning for the NLI task, which is not tackled in previous neuro-symbolic studies. 

**Fuzzy Logic.** Fuzzy logic (Zadeh, 1988; 1996) models an assertion and performs logic calculation with probability. For example, a quantifier (e.g., “most”) and assertion (e.g., “ill”) are modeled by a score in (0 _,_ 1); the score of a conjunction _s_ ( _x_ 1 _∧ x_ 2) is the product of _s_ ( _x_ 1) and _s_ ( _x_ 2). In old-school fuzzy logic studies, the mapping from language to the score is usually given by human-defined heuristics (Zadeh, 1988; Nozaki et al., 1997), and may not be suited to the task of interest. By contrast, we train neural networks to predict the probability of phrasal logical relations, and induce the sentence NLI label by fuzzy logic formulas. Thus, our approach takes advantage of both worlds of symbolism and connectionism. Mahabadi et al. (2020) apply fuzzy logic formulas to replace multi-layer perceptrons for NLI. But they are unable to provide expressive reasoning because their fuzzy logic works on sentence features. Our work is inspired by Mahabadi et al. (2020). However, we propose to apply fuzzy logic to the detected and aligned phrases, enabling our approach to provide reasoning in a symbolic (i.e., expressive) way. We develop our own fuzzy logic formulas, which are also different from Mahabadi et al. (2020). 

## 3 OUR EPR APPROACH 

In this section, we describe our EPR approach in detail, also shown in Figure 2. It has three main components: phrase detection and alignment, phrasal NLI prediction, and sentence label induction. 

**Phrase Detection and Alignment.** In NLI, a data point consists of two sentences, a premise and a hypothesis. We first extract content phrases from both input sentences by rules and heuristics. For example, “[AUX] + [NOT] + VERB + [RP]” is treated as a verb phrase. Full details are presented in Appendix A.1. Compared with the word level (Parikh et al., 2016; Feng et al., 2020), a phrase is a more meaningful semantic unit for logical reasoning. 

We then align corresponding phrases in the two sentences based on cosine similarity. Let P = (p1 _, · · · ,_ p _M_ ) and H = (h1 _, · · · ,_ h _N_ ) be the premise and hypothesis, respectively, where p _m_ and h _n_ are extracted phrases. We apply Sentence-BERT (Reimers & Gurevych, 2019) to each individual phrase and obtain the local phrase embeddings by _**p** m_[(] _[L]_[)][= SBERT(p] _m_[)] _[,]_ _**[ h]** n_[(] _[L]_[)] = SBERT(h _n_ ). We 

3 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0004-01.png)


**----- Start of picture text -----**<br>
a)  Phrase Detection and Alignment c)  Sentence Label Induction     Scores Sentence<br>Premise Label<br>Several men helping each other pull in a fishing net .<br>Sentence-BERT<br>Local similarity matrix b)  Phrasal NLI Prediction<br>Feed-forward Network<br>Local Transformer Encoder Global Transformer Encoder<br>P Several men helping each other pull in a fishing net<br>Global similarity matrix H One man <EMPTY> holding the net<br>Phrase E    C    N Fuzzy Logic Normalize E    C    N<br>There  is  one man<br>Hypothesis<br>holding<br>Sentence-BERT<br>Alignment Selection<br>the net .<br>**----- End of picture text -----**<br>


Figure 2: An overview of our Explainable Phrasal Reasoning (EPR) model. 

Table 1: An example showing the importance of handling unaligned phrases (in highlight). 

|**Premise**|People are shopping for fruit.|People are shopping for fruit<br>in the market .<br>People are shoppingfor fruit.|
|---|---|---|
|**Hypothesis**|People are shoppingfor fruit<br>in the market .||
|**Sentence NLI**|[ ]Entailment [ ]Contradiction[✓]Neutral|[✓]Entailment [ ]Contradiction[ ] Neutral|



also apply Sentence-BERT to the entire premise and hypothesis sentences to obtain the global phrase embeddings _**p**_[(] _m[G]_[)] and _**h**_[(] _n[G]_[)] by mean-pooling the features of the words in the phrase. The phrase similarity is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0004-06.png)


where _γ_ is a hyperparameter balancing the lexical and contextual representations of a phrase (Hewitt & Manning, 2019). It is noted that Sentence-BERT is finetuned on paraphrase datasets, and thus is more suitable for phrasal similarity matching than pretrained language models (Devlin et al., 2019). 

We obtain phrase alignment between the premise and hypothesis in a heuristic way. For every phrase p _m_ in the premise, we look for the most similar phrase h _n_ from the hypothesis by 

_n_ = argmax _n′_ sim( _**p** m,_ _**h** n′_ ) (2) 

Likewise, for every phrase h _n_ in the hypothesis, we look for the most similar phrase p _m_ from the premise. A phrase pair (p _m,_ h _n_ ) is considered to be aligned if h _n_ is selected as the closest phrase to p _m_ , and p _m_ is the closest to h _n_ . Such hard alignment differs from commonly used soft attention-based approaches (Parikh et al., 2016). Our alignment method can ensure the quality of phrase alignment, and more importantly, leave other phrases unaligned (e.g., _helping each other_ in Figure 1), which are common in the NLI task. The process is illustrated in Figure 2a. 

**Phrasal NLI Prediction.** Our model then predicts the logical relationship of an aligned phrase pair ( _**p** ,_ _**h**_ ) among three target labels: Entailment, Contradiction, and Neutral. While previous work (Feng et al., 2020) identifies finer-grained labels for NLI, we do not follow their categorization, because it complicates the reasoning process and makes weakly supervised training more difficult. Instead, we adopt a three-way phrasal classification, which is consistent with sentence NLI labels. 

We represent a phrase, say, p in the premise, by a vector embedding, and we consider two types of features: a local feature _**p**_[(] _[L]_[)] and a global feature _**p**_[(] _[G]_[)] , re-used from the phrase alignment component. They are concatenated as the phrase representation _**p**_ = [ _**p**_[(] _[L]_[)] ; _**p**_[(] _[G]_[)] ]. Likewise, the phrase representation for a hypothesis phrase _**h**_ is obtained in a similar way. Intuitively, local features force the model to perform reasoning in a serious manner, but global features are important to sentence-level prediction. Such intuition is also verified in an ablation study (§ 4.2). 

Then, we use a neural network to predict the phrasal NLI label (Entailment, Contradiction, and Neutral). This is given by the standard heuristic matching (Mou et al., 2016) based on phrase embeddings, followed by a multi-layer perceptron (MLP) and a three-way softmax layer: 

[ _P_ phrase(E _|_ p _,_ h); _P_ phrase(C _|_ p _,_ h); _P_ phrase(N _|_ p _,_ h)] = softmax(MLP([ _**p**_ ; _**h**_ ; _|_ _**p** −_ _**h** |_ ; _**p** ◦_ _**h**_ ])) (3) where _◦_ is the element-wise product, and the semicolon refers to column vector concatenation. E, C, and N refer to the Entailment, Contradiction, and Neutral labels, respectively. 

It should be mentioned that a phrase may be unaligned, but plays an important role in sentence-level NLI prediction, as shown in Table 1. Thus, we would like to predict phrasal NLI labels for unaligned 

4 

Published as a conference paper at ICLR 2023 

phrases as well, but pair them with a special token (p _⟨_ EMPTY _⟩_ or h _⟨_ EMPTY _⟩_ ), whose embedding is randomly initialized and learned by back-propagation. 

**Sentence Label Induction.** We observe the sentence NLI label can be logically induced from phrasal NLI labels. Based on the definition of the NLI task, we develop the following induction rules. 

Entailment Rule: According to Bowman et al. (2015), a premise entailing a hypothesis means that, if the premise is true, then the hypothesis must be true. We find that this can be oftentimes transformed into phrasal relationships: a premise entails the hypothesis if all paired phrases have the label Entailment. 

Let _{_ (p _k,_ h _k_ ) _}[K] k_ =1 � _{_ (p _k,_ h _k_ ) _}Kk_ = _′ K_ +1[be][all][phrase][pairs.][For] _[k]_[=][1] _[,][ · · ·][, K]_[,][they][are][aligned] phrases; for _k_ = _K_ + 1 _, · · · , K[′]_ , they are unaligned phrases paired with the special token, i.e., p _k_ = p _⟨_ EMPTY _⟩_ or h _k_ = h _⟨_ EMPTY _⟩_ . Then, we induce a sentence-level Entailment score by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0005-05.png)


This works in a fuzzy logic fashion (Zadeh, 1988; 1996), deciding whether the sentence-level label should be Entailment considering the average of phrasal predictions.[2] Here, we use the geometric mean, because it is biased towards low scores, i.e., if there exists one phrase pair with a low Entailment score, then the chance of sentence label being Entailment is also low. Unaligned pairs should be considered in Eq. (4), because an unaligned phrase may indicate Entailment, shown in the second example of Table 1. Notice that the resulting value _S_ sentence( `E` _|_ P _,_ H) is not normalized with respect to Contradiction and Neutral; thus, we call it a score (instead of probability), which will be normalized afterwards. 

Contradiction Rule: Two sentences are contradictory if there exists (at least) one paired phrase labeled as Contradiction. The fuzzy logic version of this induction rule is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0005-08.png)


Here, the max operator is used in the induction, because the contradiction rule is an existential statement, i.e., _there exist(s) · · ·_ . Also, unaligned phrases are excluded in calculating the sentencelevel Contradiction score, because an unaligned phrase indicates the corresponding information is missing in the other sentence and it cannot be Contradiction (recall examples in Table 1). 

Rule for Neutral: Two sentences are neutral if there exists (at least) one neutral phrase pair, but there does not exist any contradictory phrase pair. The fuzzy logic formula is 

_S_ sentence(N _|_ P _,_ H) =� max _k_ =1 _,··· ,K′ P_ phrase(N _|_ p _k,_ h _k_ )� _·_ �1 _− S_ sentence(C _|_ P _,_ H)] (6) The first factor determines whether there exists a Neutral phrase pair (including unaligned phrases, illustrated in the first example in Table 1). The second factor evaluates the negation of “at least one contradictory phrase,” as suggested in the second clause of the Rule for Neutral. 

Finally, we normalize the scores into probabilities by dividing the sum, since all the scores are already positive. This is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0005-13.png)


where L _∈{_ E _,_ C _,_ N _}_ , and _Z_ = _S_ sentence(E _|·_ ) + _S_ sentence(C _|·_ ) + _S_ sentence(N _|·_ ) is the normalizing factor. 

**Training and Inference.** We use cross-entropy loss to train our EPR model by minimizing _−_ log _P_ sentence(t _|·_ ), where t _∈{_ E _,_ C _,_ N _}_ is the groundtruth sentence-level label. 

Our underlying logical reasoning component can be trained end-to-end by back-propagation in a weakly supervised manner, because the fuzzy logic rules are almost everywhere differentiable. Although the max operators in Eqs. (5) and (6) may not be differentiable at certain points, they are common in max-margin learning and the rectified linear unit (ReLU) activation functions, and do not cause trouble in back-propagation. 

Once our EPR model is trained, we can obtain both phrasal and sentence-level labels. This is accomplished by performing argmax on the predicted probabilities (3) and (7), respectively. 

> 2In traditional fuzzy logic, the conjunction is given by probability product (Zadeh, 1988). We find that this gives a too small Entailment score compared with Contradiction and Neutral scores, causing difficulties in end-to-end training. Thus, we take the geometric mean and maintain all the scores in the same magnitude. 

5 

Published as a conference paper at ICLR 2023 

**Improving Textual Explanation.** Camburu et al. (2018) annotated a dataset to address NLI interpretability by generating an explanation sentence. For the example in Figure 1, the reference explanation is “There cannot be one man and several men at same time.” 

In this part, we apply the predicted phrasal logical relationships to textual explanation generation and examine whether our EPR’s output can help a downstream task. Figure 3 shows the overview of our textual explanation generator. We concatenate the premise and hypothesis in the form of “ _Premise : · · · Hypothesis : · · ·_ ,” and feed it to a standard Transformer encoder (Vaswani et al., 2017). 

We utilize the phrase pairs and our predicted phrasal labels as factual knowledge to enhance the decoder. Specifically, our EPR model yields a set of tuples _{_ (p _k,_ h _k,_ l _k_ ) _}[K] k_ =1[for a sample, where] l _k ∈{_ E _,_ N _,_ C _}_ is the predicted phrasal label for the aligned phrases, p _k_ and h _k_ . We embed phrases by Sentence-BERT: _**p**_[(] _[L]_[)] and _**h**_[(] _[L]_[)] ; the phrasal label is represented by a one-hot vector _**l** k_ = onehot(l _k_ ). They are concatenated as a vector _**m** k_ = [ _**p** k_ ; _**h** k_ ; _**l** k_ ]. We compose the vectors as a factual memory matrix **M** = [ _**m**[⊤]_ 1[;] _[ · · ·]_[;] _**[ m]**[⊤] K_[]] _[ ∈]_[R] _[K][×][d]_[, where] _[ d]_[ is the dimension of] _**[ m]**[k]_[.] 

Our decoder follows a standard Transformer architecture (Vaswani et al., 2017), but is equipped with additional attention mechanisms to the factual memory. Consider the _i_ th decoding step. We feed the factual memory to an MLP as **M**[˜] = MLP( **M** ). We compute attention _**a**_ over **M**[˜] with the embedding of the input _**y** i−_ 1, and aggregate factual information _**c**_ for the rows _**m** t_ in **M** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0006-05.png)


**----- Start of picture text -----**<br>
Next word Factual Memory<br>Softmax Phrases in  𝐩 Phrases in  𝐡 Label<br>Feed-forward Network a woman the woman E<br>running a marathon running E<br>N× Transformer Decoder in a park in her backyard C<br>N× Transformer Encoder + MLP Factual Memory<br>Attention<br>Premise :  ⋯ Hypothesis :  ⋯ N×<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0006-06.png)


Figure 3: Overview of the model for textual explanation generation. 

where _ak_ is the _k_ th element of the vector _**a**_ and _**m**_ ˆ _t_ is the _k_ th row of the matrix **M**[˜] . The factual information _**c**_ is fed to another layer _**g** i_ = MLP([ _**c**_ ; _**y** i−_ 1]) + _**c**_ . 

˜ Our Transformer decoder layer starts with self-attention _**q** i_ = SelfAttn( _**g** i_ ). Then, residual connection and layer normalization are applied as _**q** i_ = LayerNorm(˜ _**q** i_ + _**g** i_ ). A cross-attention mechanism obtains input information by _**v** i_ = CrossAttn( _**q** i,_ **H** ), where **H** is the representation given by the encoder. _**v** i_ is fed to the Transformer’s residual connection and layer normalization sub-layer. Multiple Transformer layers as mentioned above are stacked to form a deep architecture. The model is trained by standard cross-entropy loss against the reference explanation as in previous work (Kumar & Talukdar, 2020; Zhao & Vydiswaran, 2021; Narang et al., 2020). 

In this way, the model is enhanced with factual information given by our EPR weakly supervised reasoning. Experiments will show that it largely improves the BLEU score by 2 points (§ 4.2), being a new state of the art. This further verifies that our EPR indeed yields meaningful phrasal explanations. 

## 4 EXPERIMENTS 

## 4.1 DATASETS AND EVALUATION METRICS 

The main dataset we used in our experiments is the Stanford Natural Language Inference (SNLI) dataset (Bowman et al., 2015), which consists of 550K training samples, 10K validation samples, and another 10K test samples. Each data sample consists of two sentences (premise and hypothesis) and a sentence-level groundtruth label.[3] For sentence-level NLI prediction, we still use accuracy to evaluate our approach, following previous work (Parikh et al., 2016; Chen et al., 2017; Radford et al., 2018). 

To evaluate the phrasal reasoning performance, we need additional human annotation and evaluation metrics, because most previous work only considers sentence-level performance (Feng et al., 2020) and has not performed quantitative phrasal reasoning evaluation. Although Camburu et al. (2018) annotated phrase highlights in their e-SNLI dataset, they are incomplete and do not provide logical relationships. Our annotators selected relevant phrases from two sentences and tagged them with phrasal NLI labels; they also selected and tagged unaligned phrases. 

> 3A groundtruth label is for a data point, which consists of two sentences. We call it a _sentence-level_ label instead of phrasal labels. 

6 

Published as a conference paper at ICLR 2023 

Table 2: Main results on the SNLI dataset. _[†]_ Quoted from respective papers. _[‡]_ Obtained from the checkpoint sent by the authors. Other results are obtained by our experiments. GM and AM are the geometric and arithmetic means of the _F_ scores. 

|Model|Sent Acc|Reasoning Performance<br>_F_E<br>_F_C<br>_F_N<br>_F_UP<br>_F_UH<br>GM<br>AM|
|---|---|---|
|**Human**|–|**84.71**<br>**71.01**<br>**55.12**<br>**82.46**<br>**61.80**<br>**70.07**<br>**71.02**|
|**Non-reasoning**<br>Mahabadi et al. (2020)_†_<br>85.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>LSTM (Wang & Jiang, 2016)_†_<br>86.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>Transformer (Radford et al., 2018)<br>89.9<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>SBERT(Reimers & Gurevych,2019)<br>**91.4**<br>–<br>–<br>–<br>–<br>–<br>–<br>–|||
|**Baselines**<br>NNL (Feng et al., 2020)_‡_<br>79.91<br>62.72<br>17.49<br>1.50<br>66.22<br>0.00<br>0.00<br>29.59<br>STP<br>85.76<br>62.40<br>34.76<br>37.04<br>**76.61**<br>**51.80**<br>50.20<br>52.52<br>GPT-3-Davinci(Brown et al.,2020)<br>–<br>53.75<br>**58.00**<br>16.12<br>52.24<br>31.08<br>38.23<br>42.24|||
|**Ours**<br>EPR (Local, LM unfnetuned)<br>76.33_±_0.48<br>**83.11**_±_0.29<br>38.73_±_0.85<br>44.63_±_0.88<br>**76.61**<br>**51.80**<br>56.39_±_0.43<br>58.98_±_0.34<br>EPR (Local, LM fnetuned)<br>79.36_±_0.13<br>82.44_±_0.26<br>44.10_±_1.32<br>**44.69**_±_3.22<br>**76.61**<br>**51.80**<br>**57.77**_±_0.85<br>**59.93**_±_0.67<br>EPR (Concat, LM unfnetuned)<br>84.53_±_0.19<br>73.29_±_0.68<br>37.95_±_1.16<br>40.56_±_1.10<br>**76.61**<br>**51.80**<br>53.73_±_0.39<br>56.04_±_0.33<br>EPR(Concat,LM fnetuned)<br>**87.56**_±_0.15<br>69.91_±_1.21<br>39.97_±_2.12<br>43.31_±_2.78<br>**76.61**<br>**51.80**<br>54.46_±_1.35<br>56.32_±_1.13|||



We further propose a set of _F_ -scores, which are a balanced measure of precision and recall between human annotation and model output for Entailment, Contradiction, Neutral, and Unaligned in terms of word indexes. Details of human annotation and evaluation metrics are shown in Appendix B. 

The inter-annotator agreement is presented in Table 2 in comparison with model performance (detailed in the next part). Here, we compute the agreement by treating one annotator as the ground truth and another as the system output; the score is averaged among all annotator pairs. As seen, humans generally achieve high agreement with each other, whereas model performance is relatively low. This shows that our task and metrics are well-defined, yet phrasal logical reasoning is a challenging task for machine learning models. 

Textual explanation generation was evaluated on the e-SNLI dataset (Camburu et al., 2018), which extends the SNLI dataset with one reference explanation for each training sample, and three reference explanations for each validation or test sample. Each reference explanation comes with highlighted rationales, a set of annotated words in the premise or hypothesis considered as the reason for the explanation annotation. We do not use these highlighted rationales, but enhance the neural model with EPR output for textual explanation generation. We follow previous work (Camburu et al., 2018; Narang et al., 2020), adopting BLEU (Papineni et al., 2002) and SacreBLEU (Post, 2018) scores as the evaluation metrics; they mainly differ in the tokenizer. Camburu et al. (2018) also report low consistency of the third annotated reference, and thus use only two references for evaluation. In our study, we consider both two-reference and three-reference BLEU/SacreBLEU. Appendix A.2 provides additional implementation details of textual explanation generation. 

## 4.2 RESULTS 

**Phrasal Reasoning Performance.** To the best of our knowledge, phrasal reasoning for NLI was not explicitly evaluated in previous literature. Therefore, we propose plausible extensions to previous studies as our baselines. We consider the study of Neural Natural Logic (NNL, Feng et al., 2020) as the first baseline. It applies an attention mechanism (Parikh et al., 2016), so that each word in the hypothesis is softly aligned with the words in the premise. Then, each word in the hypothesis is predicted with one of the seven natural logic relations proposed by MacCartney & Manning (2009). We consider the maximum attention score as the alignment, and map their seven natural logic relations to our three-category NLI labels: Equivalence, ForwardEntailment _�→_ Entailment; Negation, Alternation _�→_ Contradiction; and ReverseEntailment, Cover, Independence _�→_ Neutral. 

Table 2 shows that the word-level NNL approach cannot perform meaningful phrasal reasoning, although our metrics have already excluded explicit evaluation of phrases. The low performance is because their soft attention leads to many misalignments, whereas their seven-category logical relations are too fine-grained and cause complications in weakly supervised reasoning. In addition, NNL does not allow unaligned words in the hypothesis, showing that such a model is inadequate for NLI reasoning. By contrast, our EPR model extracts phrases of meaningful semantic units, being an appropriate granularity of logical reasoning. Moreover, we work with three-category NLI labels following the sentence-level NLI task formulation. This actually restricts the model’s capacity, forcing the model to perform serious phrasal reasoning. 

7 

Published as a conference paper at ICLR 2023 

Table 3: Results of ablation studies on SNLI. 

|Model|Features|Sent Acc|Reasoning Performance<br>_F_E<br>_F_C<br>_F_N<br>_F_UP<br>_F_UH<br>GM<br>AM|
|---|---|---|---|
|Full model|Local<br>Global<br>Concat|76.33_±_0.48<br>84.03_±_0.12<br>84.53_±_0.19|**83.11**_±_0.29<br>**38.73**_±_0.85<br>**44.63**_±_0.88<br>**76.61**<br>**51.80**<br>**56.39**_±_0.43<br>**58.98**_±_0.34<br>70.84_±_0.60<br>35.12_±_0.90<br>36.37_±_1.52<br>**76.61**<br>**51.80**<br>51.41_±_0.62<br>54.15_±_0.41<br>73.29_±_0.68<br>37.95_±_1.16<br>40.56_±_1.10<br>**76.61**<br>**51.80**<br>53.73_±_0.39<br>56.04_±_0.33|
|Random chunker|Local<br>Global<br>Concat|72.44<br>82.81<br>83.09|63.21<br>22.65<br>32.04<br>65.94<br>36.13<br>40.53<br>43.99<br>58.09<br>30.64<br>27.49<br>65.94<br>36.13<br>41.05<br>43.66<br>58.75<br>32.41<br>31.14<br>65.94<br>36.13<br>42.66<br>44.87|
|Semantic role labeling|Local<br>Global<br>Concat|71.10<br>82.81<br>83.11|73.79<br>29.39<br>28.99<br>70.19<br>43.11<br>45.27<br>49.09<br>60.14<br>32.07<br>30.48<br>70.19<br>43.11<br>44.67<br>47.20<br>61.64<br>31.76<br>28.33<br>70.19<br>43.11<br>44.15<br>47.01|
|Random alignment|Local<br>Global<br>Concat|68.52<br>81.99<br>82.49|59.32<br>21.79<br>26.20<br>51.43<br>16.50<br>31.02<br>35.05<br>53.85<br>35.10<br>31.39<br>51.43<br>16.50<br>34.71<br>37.66<br>57.22<br>34.83<br>30.91<br>51.43<br>16.50<br>34.97<br>38.18|
|Mean induction|Local<br>Global<br>Concat|79.61<br>83.82<br>**84.96**|77.38<br>37.14<br>36.13<br>**76.61**<br>**51.80**<br>52.84<br>55.81<br>55.08<br>29.92<br>24.70<br>**76.61**<br>**51.80**<br>43.82<br>47.62<br>57.12<br>31.93<br>31.41<br>**76.61**<br>**51.80**<br>46.92<br>49.77|



In addition, we include another intuitive SBERT-based competing model for comparison. We first apply our own heuristics of phrase detection and alignment (thus, the model will have the same _F_ UP and _F_ UH scores); then, we directly train the phrasal NLI predictor by sentence-level labels. We obtain the sentence NLI prediction by taking argmax over Eq. (7). We call this STP (Sentence label Training Phrases). As seen, STP provides some meaningful phrasal reasoning results, because the training can smooth out the noise of phrasal labels, which are directly set as the sentence-level labels. But still, its performance is significantly lower than our EPR model. 

We experimented with a baseline of few-shot prompting with GPT-3 (Brown et al., 2020), and the implementation detail is shown in Appendix A.2. We see that GPT-3 is able to provide more or less meaningful reasoning, and surprisingly the contradiction _F_ -score is higher than all competing methods. However, the overall mean _F_ scores are much lower. The results show that phrasal reasoning is challenging for pretrained language models, highlighting the importance of our task formulation and the proposed EPR approach even in the prompting era. 

Among our EPR variants, we see that EPR with local phrase embeddings achieves the highest reasoning performance, and that EPR with concatenated features achieves a good balance between sentence-level accuracy and reasoning. Our EPR variants were run 5 times with different initialization, and standard deviations are also reported in Table 3. As seen, our improvement compared with the best baseline is around 9 _._ 1–10 _._ 7 times the standard deviation in mean _F_ scores, which is a large margin. Suppose the _F_ scores are Gaussian distributed,[4] the improvement is also statistically significant ( _p_ -value _<_ 4.5e-20 comparing our worse variant with the best competing model by one-sided test). 

We further compare our EPR with non-reasoning models (Wang & Jiang, 2016; Radford et al., 2018), which are unable to provide phrasal explanations but may or may not achieve high sentence accuracy. The results show that our phrasal EPR model hurts the sentence-level accuracy by 2–4 points, when the model architecture is controlled. This resonates with traditional symbolic AI approaches (MacCartney & Manning, 2008), where interpretable models may not outperform blackbox neural networks. Nevertheless, our sentence-level accuracy is still decent, outperforming a few classic neural models, including fuzzy logic applied to sentence embeddings (Mahabadi et al., 2020). 

**Analysis.** We consider several ablated models to verify the effect of every component in our EPR model. (1) Random chunker, which splits the sentence randomly based on the number of chunks detected by our system. (2) Random aligner, which randomly aligns phrases but keeps the number of aligned phrases unchanged. (3) Semantic role labeling, which uses the semantic roles, detected by AllenNLP (Gardner et al., 2018), as the reasoning unit. (4) Mean induction, which induces the sentence NLI label by the geometric mean of phrasal NLI prediction. In addition, we consider local phrase embedding features, global features, and their concatenation for the above model variants. Due to a large number of settings, each variant was run only once; we do not view this as a concern because Table 2 shows a low variance of our approach. Also, the underlying language model is un-finetuned in our ablation study, as it yields slightly lower performance but is much more efficient. 

As seen in Table 3, the random chunker and aligner yield poor phrasal reasoning performance, showing that working with meaningful semantic units and their alignments is important to logical reasoning. This also verifies that our word index-based metrics are able to evaluate phrase detection 

4When the score has a low standard deviation, a Gaussian distribution is a reasonable assumption because the probability of exceeding the range of _F_ scores is extremely low. 

8 

Published as a conference paper at ICLR 2023 

and alignment in an implicit manner. We further applied semantic role labeling as our reasoning unit. We find its performance is higher than the random chunker but lower than our method. This is because semantic role labeling is verb-centric, and the extracted spans may be incomplete. 

Interestingly, local features yield higher reasoning performance, but global and concatenated features yield higher sentence accuracy. This is because global features provide aggregated information of the entire sentence and allow the model to bypass meaningful reasoning. In the variant of the mean induction, for example, the phrasal predictor can simply learn to predict the sentence-level label with global sentence information; then, the mean induction is an ensemble of multiple predictors. In this way, it achieves the highest sentence accuracy (0.43 points higher than our full model with concatenated features), but is 6 points lower in reasoning performance. 

This reminds us of the debate between old schools of AI (Chandrasekaran et al., 1988; Boucher & Dienes, 2003; Goel, 2022). Recent deep learning models take the connectionists’ view, and generally outperform symbolists’ approaches in terms of the ultimate prediction, but they lack expressible explanations. Combining neural and symbolic methods becomes a hot direction in recent AI research (Liang et al., 2017; Dong et al., 2018; Yi et al., 2018). In general, our EPR model with global features achieves high performance in both reasoning and ultimate prediction for the NLI task. 

**Results of Textual Explanation Generation.** In this part, we apply EPR’s predicted output—phrasal logical relationships—as factual knowledge to textual explanation generation. Most previous studies use the groundtruth sentence-level NLI label and/or highlighted rationales. This requires human annotations, which are resource-consuming to obtain. By contrast, we require no extra humanannotated resources; our factual knowledge is based on our weakly supervised reasoning approach. 

Table 4: Textual explanation results on e-SNLI. Previous work uses auxiliary information (L: the groundtruth NLI label; H: human-annotated highlights), but we use neither. _†_ Quoted from respective papers. _‡_ Evaluated by checkpoints. _∥_ Our replication with provided code. 

|Model|Info<br>L<br>H|BLEU<br>2 refs<br>3 refs|SacreBLEU<br>2 refs<br>3 refs|
|---|---|---|---|
|Camburu et al. (2018)_†_<br>NILE (Kumar & Talukdar, 2020)_∥_<br>NILE (Kumar & Talukdar, 2020)_‡_<br>FinetunedWT5220M(Narang et al., 2020)_†_<br>FinetunedWT511B(Narang et al., 2020)_†_<br>LIREx(Zhao & Vydiswaran,2021)_∥_|–<br>–<br><br>–<br><br>–<br><br>–<br><br>–<br><br>|27.58<br>–<br>28.57<br>37.73<br>28.67<br>37.84<br>–<br>–<br>–<br>–<br>17.22<br>22.40|–<br>–<br>32.51<br>41.78<br>32.74<br>42.06<br>32.40<br>–<br>33.70<br>–<br>21.24<br>26.68|
|Finetune T560M<br>+ Annotated Highlights64M<br>+ EPR Outputs64M (ours)|–<br>–<br><br><br>–<br>–|27.75<br>36.78<br>27.91<br>36.90<br>**29.91**<br>**38.30**|31.74<br>40.89<br>32.20<br>41.21<br>**33.96**<br>**42.63**|



Table 4 shows our explanation generation performance on e-SNLI. Since evaluation metrics are not consistently used for explanation generation in previous studies, we replicate the approaches when the code or checkpoint is available. For large pretrained models, we quote results from the previous paper (Narang et al., 2020). Their model is called WT5, having 220M or 11B parameters depending on the underlying T5 model. Profoundly, we achieve higher performance with 60M-parameter T5-small, which is 3.3x and 170x smaller in model size than the two WT5 variants. 

In addition, we conducted a controlled experiment using the rationale highlights annotated by Camburu et al. (2018) for e-SNLI. It achieves a relatively small increase of 0.2–0.5 BLEU points, whereas our EPR’s outputs yield a 2-point improvement. The difference in the performance gains shows that our EPR’s phrasal logical relationships provide more valuable information than humanannotated highlights. In general, we achieve a new state of the art on e-SNLI with a small language model, demonstrating the importance of phrasal reasoning in textual explanations. 

**Additional Results.** We show additional results as appendices. § C.1: Reasoning performance on the MNLI dataset; § C.2: Error analysis; § C.3: Case studies of our EPR model; and § C.4: Case studies of textual explanation generation. 

**Conclusion.** The paper proposes an explainable phrasal reasoning (EPR) model for NLI with neural fuzzy logic, trained in a weakly supervised manner. We further propose an experimental design, including data annotation, evaluation metrics, and plausible baselines. Results show that phrasal reasoning for NLI is a meaningfully defined task, as humans can achieve high agreement. Our EPR achieves decent sentence-level accuracy, but much higher reasoning performance than all competing models. We also achieve a new state-of-the-art performance on e-SNLI textual explanation generation by applying EPR’s phrasal logical relationships. 

9 

Published as a conference paper at ICLR 2023 

## REFERENCES 

- Islam Beltagy, Stephen Roller, Pengxiang Cheng, Katrin Erk, and Raymond J Mooney. Representing meaning with a combination of logical and distributional models. _Computational Linguistics_ , pp. 763–808, 2016. URL https://aclanthology.org/J16-4007/. 

- Luke Boucher and Zoltán Dienes. Two ways of learning associations. _Cognitive Science_ , 27(6):807–842, 2003. URL https://www.sciencedirect.com/science/article/pii/ S0364021303000715. 

- Samuel Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. A large annotated corpus for learning natural language inference. In _EMNLP_ , pp. 632–642, 2015. URL https: //aclanthology.org/D15-1075. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _NeurIPS_ , pp. 1877–1901, 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf. 

- Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. e- SNLI: Natural language inference with natural language explanations. In _NeurIPS_ , pp. 9539–9549, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 4c7a167bb329bd92580a99ce422d6fa6-Abstract.html. 

- Balakrishnan Chandrasekaran, Askhok Goel, and Dean Allemang. Connectionism and information processing abstractions. _AI Magazine_ , 9(4):24–24, 1988. URL https://ojs.aaai.org/index. php/aimagazine/article/view/951. 

- Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Si Wei, Hui Jiang, and Diana Inkpen. Enhanced LSTM for natural language inference. In _ACL_ , pp. 1657–1668, 2017. URL https://aclanthology.org/ P17-1152/. 

- Zeming Chen, Qiyue Gao, and Lawrence S Moss. NeuralLog: Natural language inference with joint neural and logical reasoning. _arXiv preprint arXiv:2105.14167_ , 2021. URL https://arxiv.org/ abs/2105.14167. 

- Anup Anand Deshmukh, Qianqiu Zhang, Ming Li, Jimmy Lin, and Lili Mou. Unsupervised chunking as syntactic structure induction with a knowledge-transfer approach. In _Findings of EMNLP_ , pp. 3626–3634, 2021. URL https://aclanthology.org/2021.findings-emnlp.307. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT_ , pp. 4171–4186, 2019. URL https://aclanthology.org/N19-1423. 

- Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. _arXiv preprint arXiv:2002.06305_ , 2020. URL https://arxiv.org/abs/2002.06305. 

- Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou. Neural logic machines. In _ICLR_ , 2018. URL https://openreview.net/forum?id=B1xY-hRctX. 

- Yufei Feng, Quan Liu, Michael Greenspan, Xiaodan Zhu, et al. Exploring end-to-end differentiable natural logic modeling. In _COLING_ , pp. 1172–1185, 2020. URL https://aclanthology.org/ 2020.coling-main.101. 

- Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew Peters, Michael Schmitz, and Luke Zettlemoyer. AllenNLP: A deep semantic natural language processing platform. In _Proc. Workshop for NLP Open Source Software (NLP-OSS)_ , pp. 1–6, 2018. URL https://aclanthology.org/W18-2501. 

10 

Published as a conference paper at ICLR 2023 

- Ashok Goel. Looking back, looking ahead: Symbolic versus connectionist AI. _AI Magazine_ , 42(4): 83–85, 2022. URL https://ojs.aaai.org/index.php/aimagazine/article/view/15111. 

- John Hewitt and Christopher D Manning. A structural probe for finding syntax in word representations. In _NAACL-HLT_ , pp. 4129–4138, 2019. URL https://aclanthology.org/N19-1419. 

- Hai Hu, Qi Chen, Kyle Richardson, Atreyee Mukherjee, Lawrence S Moss, and Sandra Kübler. MonaLog: A lightweight system for natural language inference based on monotonicity. In _Proc. Society for Computation in Linguistics_ , pp. 284–293, 2020. URL https://aclanthology.org/ 2020.scil-1.40/. 

- Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-softmax. In _ICLR_ , 2017. URL https://openreview.net/forum?id=rkE3y85ee. 

- Zhongtao Jiang, Yuanzhe Zhang, Zhao Yang, Jun Zhao, and Kang Liu. Alignment rationale for natural language inference. In _ACL-IJCNLP_ , pp. 5372–5387, 2021. URL https://aclanthology. org/2021.acl-long.417/. 

- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In _EMNLP_ , pp. 6769–6781, 2020. URL https://aclanthology.org/2020.emnlp-main.550/. 

- Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _ICLR_ , 2015. URL https://arxiv.org/abs/1412.6980. 

- Sawan Kumar and Partha Talukdar. NILE : Natural language inference with faithful natural language explanations. In _ACL_ , pp. 8730–8742, 2020. URL https://aclanthology.org/2020.acl-main. 771. 

- Tao Lei, Regina Barzilay, and Tommi Jaakkola. Rationalizing neural predictions. In _EMNLP_ , pp. 107–117, 2016. URL https://aclanthology.org/D16-1011/. 

- Bowen Li, Lili Mou, and Frank Keller. An imitation learning approach to unsupervised parsing. In _ACL_ , pp. 3485–3492, 2019. URL https://aclanthology.org/P19-1338. 

- Chen Liang, Jonathan Berant, Quoc Le, Kenneth Forbus, and Ni Lao. Neural symbolic machines: Learning semantic parsers on Freebase with weak supervision. In _ACL_ , pp. 23–33, 2017. URL https://aclanthology.org/P17-1003/. 

- Xianggen Liu, Lili Mou, Haotian Cui, Zhengdong Lu, and Sen Song. Jumper: Learning when to make classification decisions in reading. In _IJCAI_ , pp. 4237–4243, 2018. URL https://www. ijcai.org/proceedings/2018/0589.pdf. 

- Yang Liu and Mirella Lapata. Text summarization with pretrained encoders. In _EMNLP-IJCNLP_ , pp. 3730–3740, 2019. URL https://aclanthology.org/D19-1387/. 

- Zhengdong Lu, Xianggen Liu, Haotian Cui, Yukun Yan, and Daqi Zheng. Object-oriented neural programming (OONP) for document understanding. In _ACL_ , pp. 2717–2726, 2018. URL https: //aclanthology.org/P18-1253. 

- Bill MacCartney and Christopher D Manning. Natural logic for textual inference. In _Proc. ACLPASCAL Workshop on Textual Entailment and Paraphrasing_ , pp. 193–200, 2007. URL https: //aclanthology.org/W07-1431/. 

- Bill MacCartney and Christopher D. Manning. Modeling semantic containment and exclusion in natural language inference. In _COLING_ , pp. 521–528, 2008. URL https://aclanthology.org/ C08-1066. 

- Bill MacCartney and Christopher D Manning. An extended model of natural logic. In _Proc. International Conference on Computational Semantics_ , pp. 140–156, 2009. URL https:// aclanthology.org/W09-3714. 

- Bill MacCartney, Michel Galley, and Christopher D Manning. A phrase-based alignment model for natural language inference. In _EMNLP_ , pp. 802–811, 2008. URL https://aclanthology.org/ D08-1084. 

11 

Published as a conference paper at ICLR 2023 

Rabeeh Karimi Mahabadi, Florian Mai, and James Henderson. Learning entailment-based sentence embeddings from natural language inference. _Online Manuscript_ , 2020. URL https: //openreview.net/forum?id=BkxackSKvH. 

- Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B. Tenenbaum, and Jiajun Wu. The neurosymbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. In _ICLR_ , 2019. URL https://openreview.net/forum?id=rJgMlhRctm. 

- Lili Mou, Rui Men, Ge Li, Yan Xu, Lu Zhang, Rui Yan, and Zhi Jin. Natural language inference by tree-based convolution and heuristic matching. In _ACL_ , pp. 130–136, 2016. URL https: //aclanthology.org/P16-2022. 

- Lili Mou, Zhengdong Lu, Hang Li, and Zhi Jin. Coupling distributed and symbolic execution for natural language queries. In _ICML_ , pp. 2518–2526, 2017. URL https://proceedings.mlr. press/v70/mou17a.html. 

- Sharan Narang, Colin Raffel, Katherine Lee, Adam Roberts, Noah Fiedel, and Karishma Malkan. WT5?! Training text-to-text models to explain their predictions. _arXiv preprint arXiv:2004.14546_ , 2020. URL https://arxiv.org/abs/2004.14546. 

- Ken Nozaki, Hisao Ishibuchi, and Hideo Tanaka. A simple but powerful heuristic method for generating fuzzy rules from numerical data. _Fuzzy Sets and Systems_ , 86(3):251–270, 1997. URL https://www.sciencedirect.com/science/article/abs/pii/0165011495004130. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: A method for automatic evaluation of machine translation. In _ACL_ , pp. 311–318, 2002. URL https://aclanthology. org/P02-1040. 

- Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. In _EMNLP_ , pp. 2249–2255, 2016. URL https://aclanthology. org/D16-1244/. 

- Matt Post. A call for clarity in reporting BLEU scores. In _Proc. Conference on Machine Translation: Research Papers_ , pp. 186–191, 2018. URL https://aclanthology.org/W18-6319. 

- Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. _OpenAI Blog_ , 2018. URL https://cdn.openai.com/ research-covers/language-unsupervised/language_understanding_paper.pdf. 

- Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In _EMNLP_ , 2019. URL https://aclanthology.org/D19-1410. 

- Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. _Nature Machine Intelligence_ , 1(5):206–215, 2019. URL https://www.nature.com/articles/s42256-019-0048-x. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , pp. 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 

- Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In _ICLR_ , 2019. URL https://openreview.net/forum?id=rJ4km2R5t7. 

- Shuohang Wang and Jing Jiang. Learning natural language inference with LSTM. In _NAACL-HLT_ , pp. 1442–1451, 2016. URL https://aclanthology.org/N16-1170/. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In _NeurIPS_ , 2022. URL https://openreview.net/forum?id=_VjQlMeSB_J. 

12 

Published as a conference paper at ICLR 2023 

- Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In _NAACL-HLT_ , pp. 1112–1122, 2018. URL https: //aclanthology.org/N18-1101. 

- Wenhan Xiong, Thien Hoang, and William Yang Wang. DeepPath: A reinforcement learning method for knowledge graph reasoning. In _EMNLP_ , pp. 564–573, 2017. URL https://aclanthology. org/D17-1060/. 

- Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, Kentaro Inui, Satoshi Sekine, Lasha Abzianidze, and Johan Bos. HELP: A dataset for identifying shortcomings of neural models in monotonicity reasoning. In _Proc. Conference on Lexical and Computational Semantics_ , pp. 250–255, 2019a. URL https://aclanthology.org/S19-1027. 

- Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, Kentaro Inui, Satoshi Sekine, Lasha Abzianidze, and Johan Bos. Can neural networks understand monotonicity reasoning? In _ACL BlackboxNLP Workshop_ , pp. 31–40, 2019b. URL https://aclanthology.org/W19-4804. 

- Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neural-symbolic VQA: Disentangling reasoning from vision and language understanding. In _NeurIPS_ , 2018. URL https://proceedings.neurips.cc/paper/2018/file/ 5e388103a391daabe3de1d76a6739ccd-Paper.pdf. 

- Deunsol Yoon, Dongbok Lee, and SangKeun Lee. Dynamic self-attention: Computing attention over words dynamically for sentence embedding. _arXiv preprint arXiv:1808.07383_ , 2018. URL https://arxiv.org/abs/1808.07383. 

- Lotfi A Zadeh. Fuzzy logic. _Computer_ , 21(4):83–93, 1988. URL https://ieeexplore.ieee.org/ abstract/document/53. 

- Lotfi A Zadeh. Fuzzy sets. In _Fuzzy Sets, Fuzzy Logic, and Fuzzy Systems_ , pp. 394–432. World Scientific, 1996. URL https://www.worldscientific.com/doi/abs/10.1142/9789814261302_ 0021. 

- Zhuosheng Zhang, Yuwei Wu, Hai Zhao, Zuchao Li, Shuailiang Zhang, Xi Zhou, and Xiang Zhou. Semantics-aware BERT for language understanding. In _AAAI_ , pp. 9628–9635, 2020. URL https://ojs.aaai.org/index.php/AAAI/article/view/6510. 

- Xinyan Zhao and V.G.Vinod Vydiswaran. LIREx: Augmenting language inference with relevant explanations. In _AAAI_ , pp. 14532–14539, 2021. URL https://ojs.aaai.org/index.php/ AAAI/article/view/17708. 

## A IMPLEMENTATION DETAILS 

## A.1 PHRASE DETECTION 

We present more details about our phrase detection. We use SpaCy[5] to obtain the part-of-speech (POS) tag[6] of every word. SpaCy also tags noun phrases. However, if a noun phrase follows a preposition (with a fine-grained POS tag being IN), we remove it from noun phrases but tag it as a prepositional phrase. 

In addition, we extract verbs by the POS tag VERB. A verb may be followed by a particle with the fine-grained POS tag being RP (e.g., _show off_ ). It is treated as a verb phrase. In order to handle negation, we allow optional AUX NOT before a verb, (e.g., _could not help_ ). This, however, only counts less than 1% in the dataset, and does not affect our model much. 

To capture other potential semantic units, we treat remaining open class words[7] as individual phrases. Finally, the remaining non-content words (in the categories of closed words and others) are discarded (e.g., “there is”). This is appropriate, because they do not represent meaningful semantics or play a 

> 5https://spacy.io 

> 6See definitions in https://spacy.io/usage/linguistic-features 

> 7https://universaldependencies.org/u/pos/ 

13 

Published as a conference paper at ICLR 2023 

Table 5: Our rules for phrase detection. “[]” means the item is optional. 

|**Example:** The woman is showingoff her blue dogat theplayground.|**Example:** The woman is showingoff her blue dogat theplayground.|**Example:** The woman is showingoff her blue dogat theplayground.|**Example:** The woman is showingoff her blue dogat theplayground.|
|---|---|---|---|
|Number|Phrase type|Rule|Extractedphrase(s)|
|1<br>2<br>3<br>4|Prepositional phrase<br>Noun phrase<br>Verb phrase<br>Others|IN+NP<br>NP<br>[AUX] + [NOT] +VERB+ [RP]<br>Other open class words|_at the playground_<br>_The woman| her blue dog_<br>_is showing off_<br>-|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0014-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0014-04.png)


Figure 4: Results of tuning the coefficient of global features. 

role in reasoning. Table 5 summarizes all the rules used in our approach. They are executed in order and extracted phrases are exclusive. For example, _the playground_ in the phrase _at the playground_ will not be treated as a standalone noun phrase, as it is already part of a prepositional phrase. 

Empirically, our rule-based approach works well for the NLI dataset, and our logical reasoning is at the granularity of the extracted phrases. 

## A.2 SETTINGS 

**Details of the EPR Model.** We chose the pretrained model all-mpnet-base-v2[8] from the SentenceBERT study (Reimers & Gurevych, 2019) and obtained 768-dimensional local and global phrase embeddings. Our MLP had the same dimension as the embeddings, i.e., 768D for the local and global variants, or 1536D for the concatenation variant. We chose the coefficient for the global feature in Eq. (1) from a candidate set of _{_ 0 _._ 0 _,_ 0 _._ 2 _,_ 0 _._ 4 _,_ 0 _._ 6 _,_ 0 _._ 8 _,_ 1 _._ 0 _}_ . Figure 4 shows the hyperparameter tuning results on SNLI (mentioned in § 4.2) and MNLI (to be discussed in § C.1). We find that 0 _._ 4 yields the best sentence accuracy in SNLI, and that 1 _._ 0 is the best for MNLI. As our focus is on reasoning, we set the coefficient to be 0 _._ 6, because it yields the highest phrasal reasoning performance and decent sentence-level performance for both experiments and in terms of both geometric mean and arithmetic mean of _F_ scores. The pretrained language model (LM) was either finetuned or un-finetuned during training. Finetuning yields higher performance (Table 2), whereas un-finetuned LM is more efficient for in-depth analyses (Table 3). We trained the model with a batch size of 256. We used Adam (Kingma & Ba, 2015) with a learning rate of 5e-5, _β_ 1=0.9, _β_ 2=0.999, learning rate warm up over the first 10 percent of the total steps, and linear decay of the learning rate. The model was trained up to 3 epochs, following the common practice (Dodge et al., 2020). Our main model variants were trained 5 times with different parameter initializations, and we report the mean and standard deviation. 

**Details of Textual Explanation Generation.** We used the pretrained T5-small model for finetuning with a batch size of 32. The optimizer was Adam with an initial learning rate of 3e-4, _β_ 1=0.9, _β_ 2=0.999, learning rate warm-up for the first 2 epochs, and linear decay of the learning rate up to 10 

> 8https://www.sbert.net/docs/pretrained_models.html 

14 

Published as a conference paper at ICLR 2023 

The phrasal logical reasoning between two sentences: "Several men helping each other pull in a fishing net.” and "There is one man holding the net." is: 

1. Entailment: " **pull in a fishing net** " vs. " **holding the net** ". 

2. Contradiction: " **Several men** " vs. " **one man** ". 

3. Neutral: **None** . 

4. Unaligned premise: " **helping each other** ". 

5. Unaligned hypothesis: **None** . 

The phrasal logical reasoning between two sentences: "An elderly couple are looking at black and white photos displayed on a wall." and "Octogenarians in heavy coats admiring the old photographs that decorated the wall." is: 1. Entailment: " **An elderly couple** " vs. " **Octogenarians** "; " **displayed on a wall** " vs. " **decorated the wall** ". 

2. Contradiction: **None** . 

3. Neutral: " **looking at black and white photos** " vs. " **admiring the old photographs** ". 

4. Unaligned premise: **None** . 

5. Unaligned hypothesis: " **in heavy coats** ". 

The phrasal logical reasoning between two sentences: "[PREMISE]" and "[HYPOTHESIS]" is: 

Figure 5: The prompt for phrasal reasoning. 

epochs; then we decreased the learning rate to 3e-6 and trained the model until the validation BLEU score did not increase for 2 epochs. 

**Details of the Prompting Baseline.** We adopted the GPT-3 (the text-davinci-003 version with 175B parameters) (Brown et al., 2020) as a prompting baseline to demonstrate large language models (LLMs)’ phrasal reasoning ability. 

We consider exemplar-based prompting, because it is unlikely for an LLM to output structured reasoning results in a zero-shot manner. Moreover, our examples are chosen to cover all reasoning cases. We also set the temperature of decoding to 0 to obtain deterministic reasoning, following CoT prompting (Wei et al., 2022). Rule-based post-processing was applied to extract slot values. Figure 5 presents the prompt used for phrasal reasoning. 

## B DATA ANNOTATION AND REASONING EVALUATION METRICS 

Previous studies have not explicitly evaluated reasoning performance. Typically, they resort to sentence-level classification accuracy (Wang & Jiang, 2016; Mahabadi et al., 2020) or case studies (Parikh et al., 2016; Feng et al., 2020) to demonstrate the effectiveness of their alleged interpretable models, which we believe is inadequate. 

Therefore, we annotated a model-agnostic corpus about phrasal logical relationships and developed a set of metrics to evaluate the phrasal reasoning performance quantitatively. The resources are released on our website (Footnote 1) to facilitate future research. 

## B.1 DATA ANNOTATION 

We annotated the phrases and their logical relationships in a data sample. The annotators were asked to select corresponding phrases from both premise and hypothesis, and label them as either Entailment, Contradiction, or Neutral, with the sentence-level NLI label being given. Annotators could also select a phrase from either a premise or a hypothesis and label it as Unaligned. The process can be repeated until all phrases are labeled for a data sample. Figure 6 shows a screenshot of our annotation page. In the left panel, the annotator could select phrases in the two sentences and mark them with NLI labels. The annotator can view a sample’s annotated phrases in the right panel and navigate through different samples. 

The annotation was performed by three in-lab researchers who are familiar with the NLI task. Our preliminary study shows low agreement when the annotators are unfamiliar with the task; thus it is inappropriate to recruit Mechanical Turks for annotation. We randomly selected 100 samples for annotation, following previous work on the textual explanation for SNLI (Camburu et al., 2018), 

15 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0016-01.png)


Figure 6: A screenshot of the annotation page. 

Table 6: Examples illustrating the proposed metrics, where we consider the Entailment category. “ _|_ ” refers to a phrase segmentation. 

|**Example annotation of entailment (in highlight):**<br>Premise:<br>A kid in red is playing in a garden.<br>Hypothesis:<br>A child in red is watchingTV in the bedroom.|**Example annotation of entailment (in highlight):**<br>Premise:<br>A kid in red is playing in a garden.<br>Hypothesis:<br>A child in red is watchingTV in the bedroom.|**Example annotation of entailment (in highlight):**<br>Premise:<br>A kid in red is playing in a garden.<br>Hypothesis:<br>A child in red is watchingTV in the bedroom.|**Example annotation of entailment (in highlight):**<br>Premise:<br>A kid in red is playing in a garden.<br>Hypothesis:<br>A child in red is watchingTV in the bedroom.|**Example annotation of entailment (in highlight):**<br>Premise:<br>A kid in red is playing in a garden.<br>Hypothesis:<br>A child in red is watchingTV in the bedroom.|**Example annotation of entailment (in highlight):**<br>Premise:<br>A kid in red is playing in a garden.<br>Hypothesis:<br>A child in red is watchingTV in the bedroom.|
|---|---|---|---|---|---|
|#|Example Output|_P_ (_P_)<br>E<br>_P_ (_H_)<br>_E_<br>_P_E|_R_(_P_)<br>E<br>_R_(_H_)<br>_E_<br>_R_E|_F_E|Explanation|
|1|P<br>in a garden<br>H<br>in the bedroom|0<br>0<br>0|0<br>0<br>0|0|Although_in_occurs in the annotation, the word<br>indexes are different. The reasoningis wrong.|
|2|P<br>a kid in red<br>H<br>watchingTV|1<br>0<br>0|1<br>0<br>0|0|Mis-matched phrases in hypothesis.<br>The reasoningis wrong.|
|3|P<br>a kid_|_in red<br>H<br>a child _|_in red|1<br>1<br>1|1<br>1<br>1|1|All word indexes match the annotation.<br>The reasoningis correct.|



which is adequate to show statistical significance. Since our annotation only concerns data samples, it is agnostic to any machine learning model. 

## B.2 EVALUATION METRICS FOR PHRASAL REASONING 

We propose a set of _F_ -scores in Entailment, Contradiction, Neutral, and Unaligned to quantitatively evaluate the phrasal reasoning performance. We first introduce our metric for one data sample and then explain the extension to a corpus. 

Consider the Entailment category as an example. We first count the number of “hits” (true positives) between the word indexes of model output and annotation. Using word indexes (instead of words) rules out hitting the words in misaligned phrases (Example 1, Table 6). Then, we calculate precision scores for the premise and hypothesis, denoted by _P_ E[(] _[P]_[ )] and _P_ E[(] _[H]_[)] , respectively. Their geometric mean _P_ E = ( _P_ E[(] _[P]_[ )] _P_ E[(] _[H]_[)] )[1] _[/]_[2] is considered as the precision for Entailment. Here, the geometric mean rules out incorrect reasoning that hits either the premise or hypothesis, but not both (Example 2, Table 6). Further, we compute the recall score _R_ E in a similar way, and finally obtain the _F_ -score by _F_ E = _P_[2] _[P]_ E+[E] _[R] R_[E] E[.][Likewise,] _[ F]_[C][ and] _[ F]_[N][ are calculated for][ Contradiction][ and][ Neutral][.][In addition,] we compute the _F_ -score for unaligned phrases in premise and hypothesis, denoted by _F_ UP and _F_ UH, respectively. 

When calculating our _F_ -scores for a corpus, we use micro-average, i.e., the precision and recall ratios are calculated in the corpus level. This is more stable, especially considering the varying lengths of sentences. Moreover, we compare model output against three annotators and perform an arithmetic average, further reducing the variance caused by ambiguity. 

16 

Published as a conference paper at ICLR 2023 

Table 7: Results on MNLI. _[†]_ Quoted from respective papers. _[‡]_ Our replication. 

|Model|Sent Acc|Reasoning Performance<br>_F_E<br>_F_C<br>_F_UP<br>_F_UH<br>GM<br>AM|
|---|---|---|
|**Human**|–|**85.15**<br>**73.44**<br>**73.18**<br>**46.31**<br>**67.85**<br>**69.52**|
|**Non-reasoning methods**<br>Mahabadi et al. (2020)_†_<br>73.8<br>–<br>–<br>–<br>–<br>–<br>–<br>LSTM (Wang et al., 2019)_†_<br>72.2<br>–<br>–<br>–<br>–<br>–<br>–<br>Transformer(Radford et al.,2018)<br>**82.1**<br>–<br>–<br>–<br>–<br>–<br>–|||
|**Reasoning methods**<br>NNL (Feng et al., 2020)_‡_<br>61.28<br>50.33<br>32.00<br>49.78<br>0.00<br>0.00<br>33.03<br>STP<br>75.15<br>55.47<br>51.72<br>**64.32**<br>**37.57**<br>51.31<br>52.27<br>EPR(Concat,LM fnetuned)<br>**79.65**_±_0.19<br>**61.76**_±_0.32<br>**52.09**_±_0.41<br>**64.32**<br>**37.57**<br>**52.80**_±_0.07<br>**53.93**_±_0.07|||



It should be emphasized that our metrics evaluate phrase detection and alignment in an implicit manner. A poor phrase detector and aligner will result in a low reasoning score (shown in our ablation study), but we do not explicitly calculate phrase detection and alignment accuracy. This helps us cope with the ambiguity of the phrase granularity (Example 3, Table 6). 

To summarize, we propose an evaluation framework including data annotation (§ B.1) and evaluation metrics (§ B.2). These are our contributions in formulating the phrasal reasoning task for NLI. 

## C ADDITIONAL RESULTS 

## C.1 RESULTS ON MNLI 

In this appendix, we provide additional results on the matched section of the MNLI dataset (Williams et al., 2018), which consists of 393K training samples, 10K validation samples, and another 10K test samples. It has the same format as the SNLI dataset, but samples come from multiple domains and are more diverse. We follow § 4.1 and use the same protocol to create the phrasal reasoning annotation for the MNLI dataset based on 100 randomly selected samples. However, we found that MNLI is much noisier than SNLI; particularly, the sentences labeled as Neutral in MNLI share few related phrases. For example, the two sentences do not have much in common in the sample “ _Premise: If you still want to join, it might be worked._ ” and “ _Hypothesis: Your membership is the only way that this could work_ ”. Moreover, the inter-human agreement is low in the Neutral category. Therefore, we believe the corpus quality is less satisfactory for Neutral. To ensure meaningful evaluation, we ignored the evaluation of Neutral in this experiment, although our reasoning approach is not changed. The remaining 60 samples containing Entailment and Contradiction serve as the MNLI phrasal reasoning corpus. 

We consider the EPR variant with concatenated local and global features, since the SNLI experiment shows it achieves a good balance between sentence-level accuracy and reasoning. Our models were run 5 times with different initializations. 

As seen in Table 7, our EPR approach is again worse than humans, but largely improves the reasoning performance compared with NNL and STP baselines. Its sentence-level prediction is comparable to (although slightly lower than) finetuning Transformers. The results are highly consistent with SNLI experiments, showing the robustness of our approach. 

It is important to notice that the EPR model here is trained on MNLI sentence labels, and is not transferred from the SNLI dataset. In our preliminary experiments, we tried transfer learning from SNLI to MNLI and failed to obtain satisfactory performance. We found that our EPR is more prone to the out-of-vocabulary issue (i.e., it does not predict well for the phrases in the new domain), whereas a black-box neural network may learn biased sentence patterns and achieve higher performance in transfer learning. 

## C.2 ERROR ANALYSIS 

To show how phrasal reasoning affects sentence-level prediction, we perform an error analysis in Table 8. Specifically, we examine the reasoning performance (arithmetic mean of _F_ -scores) when the sentence label is correctly and incorrectly predicted on the SNLI dataset. As shown, EPR models 

17 

Published as a conference paper at ICLR 2023 

Table 8: Sentence-level prediction count and arithmetic average reasoning performance ( _F_ -score) when the sentence label is correctly and incorrectly predicted on the SNLI dataset. 

|Sentence-level prediction|Sentence-level prediction|Count (in percentage)<br>Local fnetuned<br>Concat fnetuned|Count (in percentage)<br>Local fnetuned<br>Concat fnetuned|Reasoning performance (AMF)<br>Local fnetuned<br>Concat fnetuned|Reasoning performance (AMF)<br>Local fnetuned<br>Concat fnetuned|
|---|---|---|---|---|---|
|Correct<br>Wrong||75_._4_±_1.36<br>87_._8_±_0.75<br>24_._6_±_1.36<br>12_._2_±_0.75||65_._71_±_0.83<br>58_._68_±_0.67<br>40_._74_±_2.01<br>37_._58_±_3.28||
|Overall||100_._0_±_0.00<br>100_._0_±_0.00||59.93_±_0.67<br>56.32_±_1.13||
|||||||
|Three young boys<br>enjoying<br>a day<br>at the beach.<br>(a)<br>The boys are<br>in the beach.<br>Walkers<br>on a concrete boardwalk<br>under a blue sky.<br>(c)<br>Walkers<br>under a blue sky<br>near the beach.<br>Groundtruth:EntailmentPrediction:Entailment<br>Groundtruth:NeutralPrediction:Neutral|||Groundtruth:ContradictionPrediction:Contradiction||Entailment<br>Contradiction<br>Neutral<br>Unaligned|
||||A man<br>playing<br>fetch<br>with two brown dogs.<br>(b)|||
|||||||
||||The dogs are<br>asleep.|||
||||Groundtruth:EntailmentPrediction:Neutral|||
||||An elderly couple<br>in heavy coats are<br>looking<br>at black and white photos<br>displayed on a wall.<br>(d)<br>Octogenarians<br>admiring<br>the old photographs that<br>decorated the wall.|||



Figure 7: Examples of explainable phrasal reasoning predicted by our EPR model. Words in one color block are detected phrases, a dotted line shows the alignment of two phrases, and the color represents the predicted phrasal NLI label. In Example (d), EPR’s prediction suggests the provided label in SNLI is incorrect. 

with both local and concatenated features have much higher reasoning performance when sentence labels are correctly predicted than incorrectly predicted. The positive correlation between phrasal reasoning performance and sentence-level accuracy shows our fuzzy logic induction rules indeed make sense. 

We also find that the model with local features has a higher reasoning performance than with concatenated features, even when the sentence-level prediction is wrong. This is because the local model is unaware of the context of the sentences. Thus, it must perform strict phrasal reasoning based on the induction rules, even if in this case the reasoning process is imperfect and leads to sentence-level errors. 

## C.3 CASE STUDY OF EPR 

We present case studies of EPR in Figure 7. Our EPR performs impressive reasoning for the NLI task, which is learned in a weakly supervised manner with only sentence-level labels. 

In Example (a), the two sentences are predicted Entailment because _three young boys_ entails _the boys_ and _at the beach_ entails _in the beach_ , whereas unaligned phrases _enjoying_ and _a day_ are allowed in the premise for Entailment. In Example (b), _playing_ contradicts _asleep_ , and the two sentences are also predicted Contradiction. Likewise, Example (c) is predicted Neutral because the aligned phrases _on a concrete boardwalk_ and _near the beach_ are neutral. 

In our study, we also find several interesting examples where EPR’s reasoning provides clues suggesting that the target labels may be incorrect in the SNLI dataset. In Example (d), our model predicts Neutral for _looking_ and _admiring_ , as well as for _at black and white photos_ and _the old photographs_ . Thus, the two sentences are predicted Neutral instead of the provided label Entailment. We believe our model’s reasoning and prediction are correct, because people looking at something may or may not admire it; a black-and-white photo may or may not be an old photo (as it could be a black-and-white artistic photo). 

## C.4 CASE STUDY OF THE TEXTUAL EXPLANATION GENERATION 

We conduct another case study to show how EPR’s reasoning is used in the textual explanation generation task. As seen in Figure 8, our EPR reasoning yields structured factual tuples: _on a deserted beach_ entailing _at the beach_ , _Some dogs_ contradicting _only one dog_ , and _running_ unaligned (matched with a special token [EMPTY]). Our explanation generation model attends to these factual tuples, and the heat map shows that our model gives the most attention weights (with an average of 

18 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic_images/iclr_2023_Weakly_Supervised_Explainable_Phrasal_Reasoning_with_Neural_Fuzzy_Logic.pdf-0019-01.png)


**----- Start of picture text -----**<br>
Input   Premise : Some dogs are running on a deserted beach.<br>Hypothesis : There is only one dog at the beach.<br>Label  Contradiction (not used during our explanation generation)<br>EPR’s Reasoning Output<br>Premise phrase Hypothesis phrase EPR label Attention score<br>on a deserted beach at the beach E 23.16<br>Some dogs only one dog C 61.22<br>running [EMPTY] E 15.62<br>Output explanation  Some dogs is more than one dog.<br>Reference explanations:<br>(1) Some is more than one, therefore there can’t be only one dog.<br>(2) Some indicates more than one dog. One dog is not  some dogs.<br>(3) Some dogs are not one dog.<br>**----- End of picture text -----**<br>


||||||||||||Average|Average||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||attention|||
|on a deserted beach<br>at the beach|some<br>0.10|dogs<br>0.23|is<br>0.28|more<br>0.25|than<br>0.19|one<br>0.24|dog<br>0.30|.<br>0.30|[EOS]<br>0.32||score<br>0.23||1.0|
|Some dogs<br>only one dog|0.57|0.62|0.50|0.59|0.70|0.54|0.66|0.68|0.58||0.61||0.5|
|running<br>[EMPTY]|0.33|0.15|0.22|0.16|0.11|0.22|0.04|0.02|0.10||0.16||0.0|



Figure 8: Case study of the textual explanation generation. The heat map shows the step-by-step and average attention weights to the factual tuples (vertical axis). 

0.61) to the tuple, _Some dogs_ contradicting _only one dog_ , to generate the explanation “Some dogs is more than one dog.” This example illustrates that the factual tuples given by our EPR model provide meaningful information and can improve textual explanation generation. 

## D LIMITATION AND FUTURE WORK 

This paper performs phrase detection and alignment by heuristics. They work well empirically in our experiments, although further improvement is possible (for example, by considering syntactic structures). However, our main focus is neural fuzzy logic for weakly supervised reasoning. This largely differs from previous work based on manually designed lexicons and rules (Hu et al., 2020; Chen et al., 2021). 

Our long-term goal is to develop a weakly supervised, end-to-end trained neuro-symbolic system that can extract semantic units and perform reasoning for a given downstream NLP task. This paper is an important milestone toward the long-term goal. 

## E ETHICAL STATEMENTS 

Our work involves human annotation of the phrasal logical relationships. Since the research subject here is logic (rather than humans), there are minimal ethical concerns. We nevertheless followed a standard protocol of human evaluation (involving identity protection, and proper compensation), approved by our institional ethics board. 

## ACKNOWLEDGMENTS 

We thank all reviewers and chairs for their valuable comments. The research is supported in part by the Natural Sciences and Engineering Research Council of Canada (NSERC) under Grant No. RGPIN2020-04465, the Amii Fellow Program, the Canada CIFAR AI Chair Program, a UAHJIC project, a donation from DeepMind, and the Digital Research Alliance of Canada (alliancecan.ca). Atharva Naik contributed to the research as an intern at the University of Alberta through the Mitacs Globalink program. 

19 

