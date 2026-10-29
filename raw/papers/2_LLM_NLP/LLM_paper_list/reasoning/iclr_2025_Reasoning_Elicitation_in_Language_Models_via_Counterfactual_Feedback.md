# **Reasoning Elicitation in Language Models via Counterfactual Feedback** 

**Alihan H¨uy¨uk** _[∗†]_ **Xinnuo Xu** _[‡]_ **Jacqueline Maasch**[§] **Aditya V. Nori** _[‡]_ **Javier Gonz´alez** _[‡] †_ Harvard University _‡_ Microsoft Research Cambridge §Cornell Tech 

## **Abstract** 

Despite the increasing effectiveness of language models, their reasoning capabilities remain underdeveloped. In particular, causal reasoning through counterfactual question answering is lacking. This work aims to bridge this gap. We first derive novel metrics that balance accuracy in factual and counterfactual questions, capturing a more complete view of the reasoning abilities of language models than traditional factual-only based metrics. Second, we propose several fine-tuning approaches that aim to elicit better reasoning mechanisms, in the sense of the proposed metrics. Finally, we evaluate the performance of the fine-tuned language models in a variety of realistic scenarios. In particular, we investigate to what extent our fine-tuning approaches systemically achieve better generalization with respect to the base models in several problems that require, among others, inductive and deductive reasoning capabilities. 

## **1 Introduction** 

Large language models (LLMs) are shown to be ca- **Factual Question** pable of delivering astounding performance in nuA number is divisible by six if it 0 _._ 3 merous tasks across various domains. Examples has both two and three as prime stretch from writing assistants (Gan et al., 2023), to factors. Is _{N }_ divisible by six? sentiment analysis in social media (Simmering and **Counterfactual Question** 0 _._ 2 Huoviala, 2023), and even applications in healthcare (Gonzalez´ et al., 2023; Wong et al., 2023). While Ahasnumberboth twois divisibleand threebyassixprimeif it the ever-increasing accuracy of these systems is now factors. Suppose that _{N }_ had 0 _._ 1 undeniable, it is still rather unclear to what extent three as one of its prime factors (retaining all its other prime facthis accuracy is due to effective _recall_ of their traintors). Then, would it have been ing data vs. a genuine ability to _reason_ by extractbeen divisible by six? ing, understanding, and adapting the fundamental 0 concepts underlying that training data (Huang and Figure 1: _Error rate of Phi3-Mini in_ Chang, 2023; Li et al., 2023). Previous work sug- _factual vs. counterfactual questions_ gests that LLMs might exhibit some emergent reaanswers for each _N ∈{_ 1 _, . . . ,_ 100 _}_ . soning capabilities (Bubeck et al., 2023; Kıcıman disproportionately better for the factual question et al., 2023). However, many have observed a sig(cf. recall) as opposed to the counterfactual quesnificant _reasoning-recall gap_ : LLMs still perform tion (cf. reasoning). substantially better on recall-based tasks that do not explicitly require reasoning (Zhang et al., 2023a; Ahn et al., 2024; Seals and Shalin, 2024). 

||**Factual Question**<br>A number is divisible by six if it<br>has both two and three as prime<br>factors. Is _{N}_ divisible by six?<br>**Counterfactual Question**<br>A number is divisible by six if it<br>has both two and three as prime<br>factors. Suppose that _{N}_ had<br>three as one of its prime factors<br>(retaining all its other prime fac-<br>tors). Then, would it have been<br>been divisible by six?<br>Phi3 Mini<br>0<br>0_._1<br>0_._2<br>0_._3<br>Error Rate of Answers|**Factual Question**<br>A number is divisible by six if it<br>has both two and three as prime<br>factors. Is _{N}_ divisible by six?<br>**Counterfactual Question**<br>A number is divisible by six if it<br>has both two and three as prime<br>factors. Suppose that _{N}_ had<br>three as one of its prime factors<br>(retaining all its other prime fac-<br>tors). Then, would it have been<br>been divisible by six?<br>Phi3 Mini<br>0<br>0_._1<br>0_._2<br>0_._3<br>Error Rate of Answers|
|---|---|---|
||||
||A number is divisible by six if it<br>has both two and three as prime<br>factors. Suppose that _{N}_ had<br>three as one of its prime factors<br>(retaining all its other prime fac-<br>tors). Then, would it have been<br>been divisible by six?||



Figure 1: _Error rate of Phi3-Mini in answering factual vs. counterfactual questions_ —sampling 10 answers for each _N ∈{_ 1 _, . . . ,_ 100 _}_ . It performs disproportionately better for the factual question (cf. recall) as opposed to the counterfactual question (cf. reasoning). 

Motivated by this discrepancy between how well LLMs can recall vs. reason, our goal in this paper is to see whether they can be fine-tuned explicitly to improve their reasoning. While reasoning can take different forms, we will focus on _causal reasoning_ as it provides us with a clear distinction between recall and reasoning[1] : the former is limited to inferring statistical correlations, whereas the latter involves working with interventions and counterfactuals (Pearl, 2000). It has been previously shown that LLMs struggle with counterfactual questions compared to purely factual questions (Jin et al., 2024; Gonzalez and Nori, 2024).´ This difficulty highlights the recall-reasoning discrepancy within the causal domain (Figure 1). 

> _∗_ Work done as an intern at Microsoft Research Cambridge 

> 1As an example, a different kind of reasoning would be _symbolic reasoning_ , which involves manipulating symbols that represent mathematical statements (MacColl, 1897; Kelley, 1992). 

1 

Adopting a causal framework also allows us to consider an LLM’s ability to identify higher concepts that are essential for connecting causes to their effects in causal reasoning, such as _necessity_ and _sufficiency_ (Mackie, 1965; Lewis, 1973; Halpern and Pearl, 2005). For instance, a cause _X_ is said to be necessary for an effect _Y_ if (i) without intervention, _X_ and _Y_ occur together and (ii) intervening to remove _X_ results in no _Y_ (Pearl, 1999). Therefore, for an LLM to be able to identify that _X_ is necessary for _Y_ , it needs to not only determine the factual in (i) is indeed the case but also simultaneously recognize the counterfactual would have been different as in (ii). This makes identification of necessity, or similar relationships like sufficiency, a particularly good test of reasoning because it requires the LLM to understand when to recall (cf. factual thinking) vs. when to reason (cf. counterfactual thinking). 

We improve the causal reasoning of LLMs by adapting established methods of fine-tuning. In particular, we consider _supervised fine-tuning_ (SFT, e.g. Dai and Le (2015); Peters et al. (2018); Radford et al. (2018); Khandelwal et al. (2019); Howard and Ruder (2018), used in Ziegler et al. (2019); Ouyang et al. (2022)) and _direct preference optimization_ (DPO, Rafailov et al. (2024), used in Tian et al. (2023); Lin et al. (2024a)). For both of these approaches, we propose procedures to generate supervised and preference-based datasets using factual questions as well as counterfactual questions. We argue that generating demonstrations on a question-by-question basis only improves the correctness of individual answers. As we discussed, identifying higher concepts such as necessity and sufficiency requires coordination between how factual and counterfactual questions are answered together. To target these higher concepts directly, we propose generating preference-based datasets over dialogues involving both factual and counterfactual questions. 

When the goal of fine-tuning is specifically to improve reasoning, a unique problem arises in evaluating the fine-tuned LLMs: we cannot just measure performance for a held-out set of test samples within the same reasoning task. If we do, it would be impossible to tell whether the LLM actually learned to reason or whether it is still recalling the demonstrations we have made during fine-tuning.[2] Hence, measuring the generalization performance with respect to new reasoning tasks becomes crucial. We cannot expect fine-tuning on one problem instance to arbitrarily generalize to all problem instances either. So, building a systematic understanding regarding to what extent fine-tuning for reasoning should be expected to generalize becomes important as well. 

To build that understanding, we identify different modes in which reasoning in one problem is transferred to other problems. Notably, we define _inductive generalization_ and _deductive generalization_ . Given a causal system where _X → Y → Z_ , inductive generalization is the ability to reason about the transitive relationship _X → Z_ when demonstrated how to reason about _X → Y_ and _Y → Z_ . Conversely, deductive generalization is the ability to reason about the relationships _X → Y_ and _Y → Z_ when demonstrated how to reason about _X → Z_ . We show that fine-tuning for reasoning generalizes much more effectively in an inductive mode rather than a deductive mode (among many other insights in Section 6). 

**Contributions.** We have four major contributions, corresponding to each of the following sections: 

- **§2** We describe a framework for fine-tuning based on causal reasoning and formally categorize the ways in which reasoning generalizes from one problem to another. These categories are _common-effect_ , _common-cause_ , _inductive_ , and _deductive_ . 

- **§3** We introduce novel metrics to measure the reasoning performance of an LLM, defining _necessity and sufficiency inconsistency rates_ (N-IR & S-IR) based on probabilities of necessity and sufficiency from the causality literature. We also introduce the concepts of _absent necessity_ and _absent sufficiency_ to supplement cause-effect relationships covered neither by necessity nor sufficiency. 

- **§4** We propose procedures to generate datasets to be used with SFT and DPO to fine-tune for reasoning by incorporating _counterfactual feedback_ . In particular, we argue for generating dialogues that involve paired factual and counterfactual questions to directly target the reasoning metrics we introduce in Section 3. We call this _causal consistency feedback_ . 

- **§6** Finally, we evaluate the performance of the procedures proposed in Section 4 using the metrics introduced in Section 3. Moreover, we investigate to what extent that performance generalizes in relation to our categorization in Section 2. 

> 2For instance, chain-of-thought prompting aims to improve reasoning by providing examples of how a problem can be solved in smaller steps. While such prompting is effective, its effectiveness can be attributed to successful imitation of the provided examples and is not necessarily the result of true reasoning (Wei et al., 2022). 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0003-00.png)


**----- Start of picture text -----**<br>
trained<br>Y X trained trained<br>X Y A B C A B C<br>Y ˜ X ˜ tested trained<br>tested<br>(a) Common-Cause (b) Common-Effect (c) Inductive (d) Deductive (Effect-Based)<br>trained trained<br>tested tested<br>**----- End of picture text -----**<br>


Figure 2: _Different modes of generalization,_ in terms of the cause-effect relationships demonstrated during fine-tuning (i.e. _D_ , **blue** ) vs. the relationship that the fine-tuned model is evaluated on (i.e. _PX→Y_ , **orange** ). 

## **2 Fine-tuning for Reasoning** 

**World Model.** We consider a causal world model, in which _X_ (cause) and _Y_ (effect) are two binary variables, indicating the absence or presence of some conditions. We will denote with _x, y_ the values taken by _X_ and _Y_ respectively when the conditions they represent are present, and with _x[′] , y[′]_ the complements of these values (i.e. the values taken by _X_ and _Y_ when the conditions they represent are absent). The _context_ , denoted by _U_ , consists of all exogenous variables. Without any loss of generality, we assume that all randomness in the model is captured through these exogenous variables, and all endogenous variables, including _X_ and _Y_ , are deterministic functions of the exogenous variables (i.e. the context _U_ ). We denote these deterministic functions as _X_ = _fX_ ( _U_ ) and _Y_ = _fY_ ( _X, U_ ). Additionally, we denote the the _potential effects_ under the potential interventions for each unit in the population as _Yx_ = _Y |do_ ( _X_ = _x_ ) = _fY_ ( _x, U_ ) and _Yx′_ = _Y |do_ ( _X_ = _x[′]_ ) = _fY_ ( _x[′] , U_ ). 

**Language Model.** We can estimate different effects using a language model. Formally, let _q_ ( _u_ ) be a _factual question template_ that describes the world model in natural language and asks what the factual effect would be for a specific context _u_ . Denoting the language model by _ℓ_ , let _a_ = _ℓ_ ( _q_ ( _u_ )) be the model’s answer to this question, which will be in natural language form. To transform the answer into binary form, we use a mapping _h_ such that _Y_[ˆ] = _h_ ( _a_ ) = _h_ ( _ℓ_ ( _q_ ( _u_ )))˜ _∈{y, y[′] }_ ˜.[3] Similar to the factual case, suppose we also have _interventional question templates qx_ ( _u_ ) and _qx′_ ( _u_ ) that describe the world model. However, these templates ask for the potential effects under interventions _do_ ( _X_ = _x_ ) or _do_ ( _X_ = _x[′]_ ). This leaves us with the following estimates for the two potential effects. For a given context _u_ , we rely on the factual question template when the effect is factual, and on the interventional question template when the effect is counterfactual: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0003-05.png)


**Problem.** Let _P_ describe the context distribution such that _U ∼P_ . Moreover, let _PX→Y_ denote the corresponding distribution of cause _X_ = _fX_ ( _U_ ) and potential effects _Yx_ ( _U_ ) = _fY_ ( _x, U_ ), _Yx′_ ( _U_ ) = _fY_ ( _x[′] , U_ ) such that _X, Yx, Yx′ ∼PX→Y_ . Suppose we are interesting in optimizing some metric V[ _ℓ_ ; _PX→Y_ ] _∈_ R that measures the reasoning performance of the language model _ℓ_ for the cause-effect relationship _PX→Y_ (we discuss the design of V in the subsequent section). Then, the problem of _fine-tuning for reasoning_ can be expressed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0003-07.png)


where _ℓ_ 0 is the _target language model_ , and _D_ is the set of different cause-effect relationships _PXi→Yi_ that are available as _demonstrations_ . These relationships may involve causes _{Xi}_ and effects _{Yi}_ other than the cause _X_ or the effect _Y_ of interest. We refer to the case where only the cause-effect relationship of interest is demonstrated such that _D_ = _{PX→Y }_ as the “ **in-domain** ” problem. 

**Modes of Generalization.** As we have discussed in the introduction, an in-domain evaluation is not sufficient alone to assess the success of fine-tuning for reasoning. Therefore, we categorize different ways in which reasoning can generalize—that is, how _D_ might relate to _PX→Y_ when _PX→Y ̸∈D_ . We identify four main structures, summarized in Figure 2: 

- (i) **Common-Cause:** When the relationship _X → Y_ is demonstrated, _common-cause generalization_ refers to the ability to reason about other relationships _X → Y_[˜] that involve the same cause _X_ . 

> 3In practice, this mapping would also be a language model prompted to reduce given answers to a binary “negative” or “positive” (see the appendix for the exact prompt used in this paper). 

3 

- (ii) **Common-Effect:** When the relationship _X → Y_ is demonstrated, _common-effect generalization_ refers to the ability to reason about other relationships _X_[˜] _→ Y_ that involve the same effect _Y_ . Unlike common-cause generalisation, the task of determining the factual effect without intervention remains the same, regardless of whether _X_ or _X_[˜] is the cause of interest. 

- (iii) **Inductive:** When the relationship _A → B_ and _B → C_ are demonstrated, _inductive generalization_ refers to the ability to reason about the transitive relationship _A → C_ . This ability may be hindered if _A_ has a direct effect on _C_ that is not mediated by _B_ . We will investigate this empirically in Section 6. 

- (iv) **Deductive:** Similar to inductive generalization, consider the causal relationship _A → B → C_ . When the relationships _A → C_ and _B → C_ are demonstrated, _effect-based deductive generalization_ is the ability to reason about the relationship _A → B_ . Similarly, when the relationships _A → C_ and _A → B_ are demonstrated, _cause-based deductive generalization_ refers to the ability to reason about the relationship _B → C_ . We will investigate the potential differences between the two scenarios empirically in Section 6. 

## **3 Metrics of Reasoning** 

Having defined the problem of fine-tuning for reasoning, we now discuss what would be a good measure of reasoning ability (i.e. a good choice for V). In Section 3.1, we define error rates based on the _correctness_ of answers given by the language model to individual questions. In Section 3.2, we go beyond these simple error rates and propose various inconsistency rates that capture the _causal consistency_ between the factual and counterfactual answers given within the same context. As emphasized in the introduction, such consistency is necessary to identify causal relationships such as necessity and sufficiency. Later, in Section 4, we will describe various methods for generating datasets that aim to optimize either of these metrics. 

## **3.1 Correctness** 

Ignoring relationship between factual and counterfactual effects, the correctness of an individual ˜ ˜ answer _a_ = _ℓ ◦ q_ ( _u_ ) _| ℓ ◦ qx_ ( _u_ ) _| ℓ ◦ qx′_ ( _u_ ) can be characterized by the _factual error rate_ (F-ER) and the _counterfactual error rate_ (CF-ER) respectively: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0004-07.png)


where _Y_[ˆ] , _Y_[ˆ] _x_ , and _Y_[ˆ] _x′_ represent the binary values implied by the answer _a_ . Using these two metrics, we define the _average error rate_ as Avg-ER = (F-ER + CF-ER) _/_ 2. 

**Why are factual and counterfactual correctness alone not enough?** Being able to correctly estimate factuals (cf. F-ER) or counterfactuals (cf. CF-ER) is, of course, an important step in causal reasoning. However, what we ultimately want is to characterize the relationship between a cause and its effect. For instance, is the cause necessary for the effect to occur? Is it sufficient? Or do the cause and the effect only occur together (necessary and sufficient)? Identifying such relationships rely on the estimated factuals and counterfactuals collectively—only getting one right but not the other might not always lead to a correct characterization of the cause-effect relationship. By measuring the factual and counterfactual accuracy separately, F-ER and CF-ER fail to capture any dependencies between the two answers and how they might be describing a larger relationship together. 

As a concrete example, consider _necessity_ . According to Pearl (1999), when a cause _X_ and an effect _Y_ occur together (i.e. _X_ = _x_ and _Y_ = _y_ ), the cause is said to have been necessary for the effect if the effect would not have occurred in the absence of the cause (i.e. _Yx′_ = _y[′]_ ). Making an accurate judgement regarding whether there is a necessity relationship between _X_ and _Y_ requires both _Y_[ˆ] and _Y_[ˆ] _x′_ to be correct when _X_ = _x_ and _Y_ = _y_ . However, no factual or counterfactual estimate needs to be correct when _X_ = _x[′]_ (as it is immediately apparent that cases where _X_ = _x[′]_ do not affect necessity), and similarly, only the factual estimates needs to be correct when _X_ = _x_ but _Y_ = _y[′]_ . F-ER and CF-ER do not account for this complex requirement at all. In particular, depending on how _X_ and _Y_ are distributed, a language model can achieve F-ER and CF-ER as high as 1 _/_ 2 by always estimating either _Yx_ or _Yx′_ correctly (but not both together) while never reaching an accurate conclusion regarding necessity. 

4 

## **3.2 Causal Consistency** 

Previous work (Gonzalez´ and Nori, 2024) has considered the use of “probabilities of causation” together with F-ER and CF-ER to provide a set metrics that fully characterize the relationship between a cause and its effect. Similar to necessity, Pearl (1999) also provides a causal definition of sufficiency: whether the cause would have produced the effect (i.e. _Yx_ = _y_ ) when both the cause and the effect are absent (i.e. _X_ = _x[′]_ and _Y_ = _y[′]_ ). The _probability of necessity_ (PN) and the _probability of sufficiency_ (PS) are defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0005-02.png)


The answers given by the language model to factual and counterfactual questions and the effectsˆ ˆ _Yx, Yx′_ estimated from those answers naturally induce an empirical pair of PN and PS values: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0005-04.png)


**Why are PN and PS correctness alone not enough?** To evaluate reasoning in language models, Gonzalez and Nori (2024) use (1) a probabilistic measure (´ _γ_ -overlap) to assess how well the distributions of PN[�] and PS[�] match the true PN and PS, and (2) the factual and counterfactual error rates. We refine this approach by defining unifying metrics that simultaneously take both aspects of the problem into account, thereby simplifying the evaluation process. 

Due to the averaging done by probabilities, achieving a perfect PN-PS with the language model only requires identifying correct vs. predicted marginal frequencies, without needing individual units to be accurate. Although this is captured by the factual and counterfactual error rates F-ER and CF-ER, it is convenient to have a single metric that encapsulates both dimensions of the problem. We address this by requiring the necessity or sufficiency relationships identified by the language model to be accurate on a unit-by-unit basis. A unit is a realization of the exogenous variable _U_ . It induces the values of _X_ and _Y_ as well as the counterfactual outcome _YX ′_ , where _X[′]_ represents the complement of the observed _X_ regardless of its value. Note that _YX_ = _Y_ is the factual outcome. 

We focus on _necessity_ where a unit/context might exhibit one of three situations: (i) Necessity occurs, denoted by “N”, meaning that both _X_ and _Y_ occur, _X_ = _x_ and _Y_ = _y_ , and the cause was necessary for the effect, _YX ′_ = _y[′]_ . (ii) Necessity does not occur, which we denote by “N _[′]_ ”, meaning that both _X_ and _Y_ occur but the cause was not necessary for the effect, _YX ′_ = _y[′]_ . (iii) Not relevant case as necessity is concerned, which we denote by _̸_ �, when neither _X_ nor _Y_ (or both) did occur. Since value of the context variable _U_ fully characterizes the unit, we can define unit-wise necessity as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0005-08.png)


The _necessity inconsistency rate_ (N-IR) is the frequency with which the language model estimates the unit-wise necessity _N_ inaccurately: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0005-10.png)


where E _P_ ( _U_ ) denotes the expectation over _U_ and _Y_[ˆ] , _Y_[ˆ] _X[′]_ are the analogous factual and counterfactuals to _Y_ , _YX ′_ estimated from the model. Remark that PN = E _P_ ( _U_ )[ _N_ = N _|N_ = �] by construction. Also note that N-IR = 0 implies that _PN_[�] = PN. However, errors made in different units can no longer ‘balance each other out’ to achieve N-IR = 0. We can also define context-wise sufficiency _S_ in an analogous way: (i) _S_ = S if _X_ = _x[′] , Y_ = _y[′] , YX ′_ = _y_ , (ii) _S_ = S _[′]_ if _X_ = _x[′] , Y_ = _y[′] , YX ′_ = _y_ , and (iii) _S_ = _̸_ � otherwise. This induces the _sufficiency inconsistency rate_ S-IR = P _{S_[ˆ] = _S}_ . 

Neither PN and PS nor the inconsistency rates N-IR and S-IR are sensitive to all answers given by the language model. This is because necessity and sufficiency only concern cases where _X_ = _x, Y_ = _y_ and _X_ = _y[′] , Y_ = _y[′]_ . For instance, when _X_ = _x[′]_ and _Y_ = _y_ and the factual effect has been estimated correctly such that _Y_[ˆ] = _Y_ , the counterfactual estimate _Y_[ˆ] _x_ has no impact on PN, PS, N-IR, or S-IR. Regardless of whether _Y_[ˆ] _x_ = _Yx_ , all four quantities stay the same. To cover all possible counterfactuals we can ask a language model for, it makes sense to also evaluate counterfactuals of the type _Yx′_ = _y|X_ = _x, Y_ = _y[′]_ and _Yx_ = _y[′] |X_ = _x[′] , Y_ = _y_ . Of course, the probabilities of these counterfactuals can be defined by means of PN and PS by changing the default observed state. However, here we name them as _absent necessity_ and _absent sufficiency_ to be explicit about the 

5 

two extra cases where language model can make mistakes.[4] In our context-based framework, the corresponding context-wise _AN_ and _AS_ are defined in a similar fashion as _N_ and _S_ , which induce the inconsistency rates AN-IR = P _{AN_[ˆ] = _AN}_ and AS-IR = P _{AS_[ˆ] = _AS}_ . As a reasoning metric, we define the _average inconsistency rate_ as Avg-IR = (N-IR + S-IR + AN-IR + AS-IR) _/_ 4. This metric has the following properties: (i) it accounts for all characterizations of the necessity and sufficiency of the target causal effect, and (ii) it is unit-dependent, so factual and counterfactual accuracy errors cannot be balanced out. 

> **An Illustrative Example.** the difference between correctnessWe illustrateand 0.55 Factually CorrectUniformly CorrectCausally ConsistentTrue PN-PS Avg-ER=0.200 0.70.6 causalthe following language models:consistency as follows. (i)Consider _Factu-_ 0.50 Avg-ER=0.225  Avg-ER=0.250    Avg-ER=0.250  Avg-ER=0.225  Avg-ER=0.200 0.5 0.4 _ally Correct_ answers all factual questions 0.45 correctly (F-ER = 0) but makes occasional 0.3 mistakes in answering counterfactual ques0.40 0.2 tions. This represents an extreme version 0.35 Avg-ER=0.200  Avg-ER=0.225 0.1 of the imbalance highlighted in the introAvg-ER=0.250 0.0 0.35 0.40 0.45 0.50 0.55 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 duction. (ii) _Uniformly Correct_ makes both Probability of Necessity (PN) Necessity Inconsistency Rate (N-IR) factual and counterfactual mistakes at equal **(a) PN & PS (b) N-IR & S-IR** rates (F-ER = CF-ER), but these mistakes happen independently of each other. Figure 3: _Causal consistency vs. correctness._ Despite having (iii) _Causally Consistent_ reasons on a unitthe same Avg-ER, different types of error distributions lead to by-unit basis (as opposed to question-bywidely different PN & PS characteristics. question) and either gets both the factual question and counterfactual question right or gets both of them wrong. Suppose the cause never prevents the effect such that ( _X, Yx, Yx′_ ) _∈{_ ( _x, y[′] , y[′]_ ) _,_ ( _x, y[′] , y_ ) _,_ ( _x, y, y_ ) _,_ ( _x[′] , y[′] , y[′]_ ) _,_ ( _x[′] , y[′] , y_ ) _,_ ( _x[′] , y, y_ ) _}_ (with equal probabilities). 

Figure 3: _Causal consistency vs. correctness._ Despite having the same Avg-ER, different types of error distributions lead to widely different PN & PS characteristics. 

Figure 3 shows the PN & PS as well as N-IR & S-IR of these models for fixed levels of Avg-ER as the error shifts between contexts where the cause may be necessary (i.e. _X_ = _x_ ) vs. contexts where it may be sufficient (i.e. _X_ = _x[′]_ ). Despite having the same Avg-ER, the three models induce widely different PN & PS values, representing different causal interpretations. While Figure 3a might suggest that the factually correct models are the best performing, this is purely coincidental. Due to the averaging done by PN & PS, the mistakes made in different units end up balancing each other out. Looking at N-IR & S-IR in Figure 3b reveals that the causally consistent models are actually the best, even outperforming models with significantly smaller Avg-ER. 

## **4 Fine-tuning with Counterfactual Feedback** 

Despite the significant differences between correctness and causal consistency, success in either metric relies on accurate estimates of counterfactual outcomes. Therefore, to solve the fine-tuning problem in (2), it is essential to leverage the counterfactual information available in demonstrations _D_ , irrespective of the metric we aim to target as V. We present a data-centric approach to achieve this and propose three methods for generating datasets using counterfactual feedback. These datasets can then be utlised by existing algorithms for fine-tuning such as SFT or DPO. These methods are summarized in Figure 4. 

**Supervised Counterfactual Feedback.** Recall that we assumed access to an extractor _h_ that can ˆ reduce answers given in natural language to binary outcomes _y_ = _h_ ( _a_ ) _∈{y, y[′] }_ . Now, further suppose that we can perform this extraction in reverse, denoted as _H_ : Given a question _q_ and the true outcome _y_ true corresponding to this question, we can form a natural language answer _a_ = _H_ ( _q, y_ true). In practice, we achieve this by prompting a language model to provide an answer to question _q_ that starts with _“Yes”_ or _“No”_ (see the appendix for the full prompt). Based on these answers, we generate a dataset D of both factual and counterfactual questions and their answers: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0006-07.png)


This dataset can directly be used with any SFT algorithm to fine-tune the target model _ℓ_ 0. 

> 4Note that the use of these quantities is an alternative but equivalent characterization of all possible counterfactual outcomes to the one in Pearl (1999), where the probabilities of disablement _Yx′_ = _y[′] |Y_ = _y_ and the probability of enablement _Yx_ = _y|Y_ = _y[′]_ are introduced. 

6 

**Preference-based Counterfactual Feedback.** SFT can be limited by the quality of answers generated as ground-truth and their similarity to the model’s original answers. Without access to a language model that is already better at reasoning than our target model, it might challenging to build an answer generator _H_ that provides high quality samples. In that case, it is desirable to provide direct feedback to the answers generated by the target language model. We do so by first generating multiple answers to different questions (using a high sampling temperature to get sufficient variation): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0007-01.png)


Then, we form a preference-based dataset where correct answers are preferred over incorrect answers: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0007-03.png)


The DPO algorithm can directly be used with this dataset to maximize the likelihood of preferred answers (i.e. _a_ [ _i_ ]) relative to the answers they are preferred over (i.e. _a_ [ _j_ ]). 

**Preference-based Causal Consistency Feedback.** Running DPO with preferences determined by a reward function, where alternatives with higher rewards are preferred over those with lower rewards, is equivalent to maximizing that reward function (Rafailov et al., 2024). In our case, this means that running DPO with the above preferences would, in effect, minimize the average error rate (i.e. Avg-ER), as these preferences are generated by treating correctness (i.e. 1 _{h_ ( _a_ ) = _Y_[ˆ] = _Y }_ ) as a reward function. To target the inconsistency rates introduced in Section 3.2, we propose to (i) pair factual and counterfactual questions, (ii) prompt the target language model to answer them simultaneously, and then (iii) elicit preferences based on the joint answer. Formally, 

( _af_ [ _i_ ] _, acf_ [ _i_ ]) _≻_ ( _af_ [ _j_ ] _, acf_ [ _j_ ]) _⇐⇒_ 

_R_ ( _h_ ( _af_ [ _i_ ]) _, h_ ( _acf_ [ _i_ ]); _U_ ) _> R_ ( _h_ ( _af_ [ _j_ ]) _, h_ ( _acf_ [ _j_ ]); _U_ ) (11) 

where _R_ ( _Y ,_[ˆ] _Y_[ˆ] _X_ ˆ _′_ ; _U_ )= 1 _{N_ = _N}_[ˆ] ˆ+ 1 _{S_ = _S}_[ˆ] + 1 _{AN_ = _AN}_ + 1 _{AS_ = _AS}_ . We call this _causal consistency feedback_ (CCF) (see Figure 4b vs. 4c). CFF explicitly targets Avg-IR rather than Avg-ER and can still be used directly with the DPO algorithm. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0007-09.png)


**----- Start of picture text -----**<br>
 Y Large LM Yes, ...For  U , will it be  Y  ? <br> Y 0 Large LM No, ...Forwould it have been  U , if it had not been  Y  ?  X , <br>U<br>(a) Supervised CF<br> Y Target LM ˆ = Yes For  U , will it be  Y  ? Y ˆ =?  Y Y Target LM ˆ = Yes For  U , will it be  Y  ? <br> Y ˆ Target LM 0 = NoForwould it have been  U , if it had not been  Y  ?  X , Y ˆ0 =?  Y 0 Y ˆ Target LM 0 = NoForwould it have been  U , if it had not been  Y  ?  X ,  U<br>(b) Preference-based CF<br> Y Target LM ˆ = Yes For  U , will it be  Y  ? Y Target LM ˆ = Yes For  U , will it be  Y  ? <br> Y ˆ0 = Nowould it have beenIf it had not been  X Y  ?, Y ˆ0 = Yeswould it have beenIf it had not been  X Y  ?, <br> R (  Y [ˆ] 0 , 1;  U )  U<br>(c) Preference-based CCF<br>**----- End of picture text -----**<br>


## **5 Related Work** 

**Reasoning Evaluation.** While our work focuses on reasoning _elicitation_ , there is a plethora of work on reasoning _evaluation_ (Chang et al., 2024). Parmar et al. (2024) evaluate logical reasoning, Cohn and Hernandez-Orallo (2023) evaluate spatial reasoning, Gandhi et al. (2024) evaluate social reasoning, and Li et al. (2022); Jin et al. (2023); Ashwani et al. (2024); Li et al. (2024); Wang (2024) evaluate causal reasoning. Of course, being able to determine which model is better at reasoning is an important aspect of reasoning elicitation. We have explored this aspect in Section 3 building on the work of Gonzalez and Nori (2024).´ This allowed us to consider relationships like necessity and sufficiency, which we have shown to require a higher level of reasoning than simply answering counterfactual prompts. 

Figure 4: _Summary of the proposed fine tuning methods._ Supervised and preference-based counterfactual feedback (CF) target correctness: the former by generating correct answers given each question and the latter by sampling answers and preferring the correct ones over the others. Causal consistency feedback (CCF) targets causal consistency instead: Asking both the factual and the counterfactual questions within the same dialogue allows us to elicit preferences according to relationships between the factual and counterfactual answers. 

**Counterfactual Frameworks.** Counterfactual frameworks have been employed to explore various aspects of large language models. For instance, Lin et al. (2024b) formulate preference alignment as a causal inference problem and de- 

7 

velop an alternative approach to algorithms like DPO. In Wu et al. (2021); Nguyen et al. (2024), counterfactual inputs are used to explain a model’s predictions. Additionally, Kandpal et al. (2023); Zhang et al. (2023b) model memorization through counterfactuals. 

**Fine-tuning with Factual Feedback.** Finally, while we fine-tune language models for reasoning with counterfactual feedback, previous work has considered fine-tuning for factuality: providing factually correct answers to questions that do not involve no interventions (Tian et al., 2023; Tong et al., 2024; Lin et al., 2024a). Khalifa et al. (2020); Korbak et al. (2022a;b) propose methods for _controlled generation_ , which aim to constrain a model’s answers using binary reward functions. While they suggest using correctness as the reward function to improve factuality, we have shown that targeting metrics like causal consistency require more fine-grained feedback (beyond a binary reward). 

## **6 Experiments** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0008-03.png)


We begin with a proof-of-concept case study. We analyze a hand-crafted puzzle to assess the effectiveness of all fine-tuning techniques introduced in Section 4 when trained on different types of datasets within the context of the in-domain causal reasoning scenarios ( **§6.1** ). We also address the research question posed in Section 1, _i.e._ , to what extent the performance improvements in causal reasoning achieved through the fine-tuning process generalize across all the generalization modes ( **§6.2** ). Subsequently, we use three additional real-world problems to examine our findings ( **§6.3** ). 

## **6.1 In-domain Reasoning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0008-06.png)


**----- Start of picture text -----**<br>
0.14 0.14<br>0.12 DPO-OnlyF BaseDPO-OnlyCF 0.12 DPO-OnlyCF DPO-OnlyF Base<br>0.10 SFT-OnlyF SFT-OnlyCF 0.10 SFT-OnlyCF<br>0.08 0.08 SFT-OnlyF<br>0.06 0.06<br>DPO-F&CF DPO-F&CF<br>0.04 0.04<br>0.020.00 0.020.00<br>0.00 SFT-F&CFDPO+CCFDPO+CCF 0.00 DPO+CCF SFT-F&CF<br>0.00 0.05 0.10 0.15Factual Error (F-ER)0.20 0.20 0.25 0.30 0.35 0.40 0.00 0.05 0.10Counterfactual Error (CF-ER)0.15 0.15 0.20 0.25 0.30 0.35<br>Factual Error (F-ER)0.20 Counterfactual Error (CF-ER)0.15<br>Sufficiency Inconsistency (S-IR) Sufficiency Inconsistency (S-IR)<br>**----- End of picture text -----**<br>


We evaluate all fine-tuning techniques when trained on various types of demonstrations in a synthetic in-domain reasoning problem. 

DPO-F&CF DPO-F&CF **Experimental Setup.** See Figure 5. The 0.04 0.04 puzzle describes a candy party. The context is 0.020.00 SFT-F&CFDPO+CCFDPO+CCF 0.020.00 DPO+CCF SFT-F&CF defined by the four-dimensional random vector 0.00 0.05 0.10 0.15Factual Error (F-ER)0.20 0.25 0.30 0.35 0.40 0.00 0.05 0.10Counterfactual Error (CF-ER)0.15 0.20 0.25 0.30 0.35 _U_ = ( _NA, NB, NC, ND_ ) where each element **(a) S-IR w.r.t. F-ER (b) S-IR w.r.t. CF-ER** follows the same uniform distribution _U_ (1 _,_ 12). Figure 5: **Top** : Hand-crafted puzzle with the original The causal structure, derived from the narratives, factual question, causal structure, and a counterfactual is presented in the middle section of the figure. question. **blue** and **orange** arrows show the cause-effect We selected _A: Anna is happy or not_ as the cause interventions demonstrated to the model during fine( _X_ ), and _D: Dave is happy or not_ as the effect tuning and evaluation phases. **Bottom** : In-domain re( _Y_ ). The factual questions _q_ ( _u_ ) are obtained sults. The y-axis in both figures represents S-IR, while by randomly drawing values for the four nuthe x-axes represent F-ER and CF-ER, respectively. We merical variables from the distribution _U_ , The focus on S-IR because, in this puzzle, the cause is more ˜ counterfactual questions _qX ′_ ( _u_ ) are generated sufficient than necessary for producing the effect. by introducing an assumption that negates the cause _i.e._ if in the context _A_ is “ _Anna is happy_ ” based on the value of _NA_ , the injected assumption would be “ _suppose that Anna is not happy_ ”, and vice versa. Since we are assessing the in-domain reasoning scenario, the cause-effect demonstration used during the fine-tuning phase are likewise employed in the evaluation phase. 

Figure 5: **Top** : Hand-crafted puzzle with the original factual question, causal structure, and a counterfactual question. **blue** and **orange** arrows show the cause-effect interventions demonstrated to the model during finetuning and evaluation phases. **Bottom** : In-domain results. The y-axis in both figures represents S-IR, while the x-axes represent F-ER and CF-ER, respectively. We focus on S-IR because, in this puzzle, the cause is more sufficient than necessary for producing the effect. 

We first generate dataset D = ( _{_ ( _q_ ( _u_ ) _, af_ ) _}, {_ (˜ _qX[′]_ ( _u_ ) _, acf_ ) _}_ ) for each fine-tuning techniques introduced in Section 4 following the algorithms shown in Appendix B. Then we fine-tune the mini version of Phi-3 (Abdin et al., 2024) on D. We include five baselines: the base language model (Phi-3 mini) without fine-tuning ( _**Base**_ ), the base model fine-tuned using the SFT and DPO methods on factual examples _{_ ( _q_ ( _u_ ) _, af_ ) _}_ exclusively ( _**SFT-OnlyF**_ and _**DPO-OnlyF**_ ), and the base model fine-tuned using the SFT and DPO methods on counterfactual examples _{_ (˜ _qX ′_ ( _u_ ) _, acf_ ) _}_ exclusively ( _**SFT-OnlyCF**_ and _**DPO-OnlyCF**_ ). As our proposed methods, we include the base model fine-tuned using SFT, DPO, and CCF methods on both factual and counterfactual examples ( _**SFT-F&CF**_ , _**DPO-F&CF**_ , and _**DPO+CCF**_ ). 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0009-00.png)


**----- Start of picture text -----**<br>
A C A C<br>A B C A B C A B C A B C A B C A B C<br>B D B D<br>Common-Cause Common-Effect No Direct Effect With Direct Effect No Direct Effect With Direct Effect No Direct Effect With Direct Effect<br>(“CC”) (“CE”) (“NDE”) (“WDE”) (“NDE”) (“WDE”) (“NDE”) (“WDE”)<br>0.300.250.200.150.100.050.00 SFT-F&CF (CE) DPO+CCF (CE)DPO-F&CF (CE) DPO-F&CFDPO+CCF(CC)(CC) SFT-F&CF(CC) Base(CE) Base(CC) 0.2000.1750.1500.1250.1000.0750.0500.0250.000 DPO+CCF (WDE) DPO-F&CF(WDE)SFT-F&CF(WDE)DPO-F&CF(NDE) DPO+CCF (NDE)SFT-F&CF (NDE) (WDE)Base(NDE) Base 0.100.080.060.040.020.00 SFT-F&CF(WDE)DPO+CCF (WDE) SFT-F&CFDPO-F&CF(NDE)DPO-F&CF(WDE) (NDE)DPO+CCF (NDE) (WDE)Base(BaseNDE) 0.2000.1750.1500.1250.1000.0750.0500.0250.000 SFT-F&CF(NDE)DPO-F&CF(NDE)DPO+CCF (NDE)DPO+CCF (WDE) SFT-F&CF (WDE)DPO-F&CF(WDE) (BaseNDE(WDE)Base)<br>0.0 0.1 0.2 0.3 0.4 0.5 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.0 0.1 0.2 0.3 0.4 0.5 0.6<br>Necessity Inconsistency (N-IR) Necessity Inconsistency (N-IR) Necessity Inconsistency (N-IR) Necessity Inconsistency (N-IR)<br>(a) Common-Cause/Effect (b) Inductive (c) Deductive, Cause-B. (d) Deductive, Effect-B.<br>Sufficiency Inconsistency (S-IR) Sufficiency Inconsistency (S-IR) Sufficiency Inconsistency (S-IR) Sufficiency Inconsistency (S-IR)<br>**----- End of picture text -----**<br>


Figure 6: _Generalization results in the candy party puzzle._ **Top:** Eight scenarios involving three different causal structures: the bipartite graph _{A, B} →{C, D}_ (Structure-1 in Appendix C.1) as well as the chain _A → B → C_ with and without a direct effect from _A_ to _C_ (Structure-2 and Structure-3 in Appendix C.1). **blue** and **orange** arrows show the cause-effect interventions demonstrated to the model during fine-tuning and evaluation phases. **Bottom:** The causal reasoning ability of the fine-tuned models generalizes most effectively in inductive demonstrations. However, with common-cause/effect and deductive demonstrations, they no longer show the same reasoning improvements as observed in the in-domain setting. 

**Results.** Figure 5 shows the sufficiency inconsistency rate (S-IR) in relation to the factual/counterfactual error rates (F/CF-ER) across all approaches.[5] SFT and DPO models, trained exclusively on either factual or counterfactual examples ( _SFT+OnlyF_ , _SFT+OnlyCF_ , _DPO+OnlyF_ , and _DPO+OnlyCF_ ) do not improve S-IR, even though they manage to reduce the corresponding F/CF-ER. However, when given access to both types of examples, _DPO+F&CF_ shows an improvement in S-IR, though this improvement is not as pronounced as the reduction observed in F/CF-ER, particularly in CF-ER. The _SFT+F&CF_ model shows a significant enhancement in both S-IR and F-ER, but it fails to make progress in CF-ER. Finally, by directly addressing causal consistency, with S-IR factored into the reward during fine-tuning, the _DPO+CCF_ model achieves substantial improvements across F-ER, CF-ER, and S-IR. These results highlight the crucial role of effectively coordinating factual and counterfactual feedback for advanced reasoning tasks. 

## **6.2 Modes of Generalization** 

In this section, we answer the question “ _to what extent the performance improvements in causal reasoning achieved through the fine-tuning process generalize across all the generalization modes defined in Section 2_ ”. As mentioned in Section 2, an in-domain evaluation alone is inadequate for fully assessing the success of fine-tuning for reasoning and differentiating it from basic recall. Therefore, we evaluate all fine-tuning methods in the generalization modes introduced in Section 2. 

**Experimental Setup.** To allow for the problem in Figure 5 to reflect all possible generalization modes we made slight modifications to the puzzle context, creating two variations: chain NDE and chain WDE (refer to Structure-2 and Structure-3 in Appendix C.1). The top section in Figure 6 displays all the causal structures used for each generalization mode, along with the cause-effect interventions demonstrated during the fine-tuning and evaluation phases. Based on the findings from the in-domain reasoning experiments (Section 6.1), where both SFT and DPO fine-tuning methods showed significantly better performance when provided with both factual and counterfactual examples, we include here only the methods _**SFT-F&CF**_ , _**DPO-F&CF**_ , and _**DPO+CCF**_ as well as the _**Base**_ model. 

**Results.** The bottom section in Figure 6 presents the causal reasoning performance of all systems across the different generalization modes. We observe that: **(i) Common-Cause/Effect.** Fine-tuning based on demonstrations that involve just the target cause or the target effect (but not both as in the in-domain case) no longer leads to improvements in S-IR (unlike the in-domain case). While we do see improvements in N-IR, this can be attributed to better recall and not necessarily to better reasoning. The common-effect case leads to the greater improvement in N-IR precisely because the task of identifying factuals remains the same in this mode of generalization. **(ii) Induction.** Fine-tuning generalizes best when performed inductively. This is because relationships involving both 

> 5When evaluating models, we sample 10 answers for each question, which gives us a distribution over ER/IR (rather than just a point estimation). 

9 

the target cause and the target effect have been demonstrated, albeit not together. **(iii) Deductions.** While harder than induction, deduction is also possible as long as there are no direct effects that circumvent the intermediate variable. If there are such effects, deduction based on a shared cause becomes virtually impossible: Without any intervention on the intermediate variable, it is challenging to tell how much of the shared cause’s effect is mediated through the intermediate variable vs. how much of it is not. Meanwhile, this seems to be identifiable to some extent when interventions on the intermediate variable are demonstrated as in deduction based on a shared effect. 

## **6.3 Real-world Problems** 

We validate our experimental findings in real-world problems from three domains. 

**Experimental Setup.** We present three real-world causal reasoning problems: in the **Healthcare** domain, we examine breast cancer treatment and develop a simplified problem that determines how different treatment options—namely, radiotherapy/chemotherapy and surgery—are assigned to patients based on cancer type, tumor size, and nodal involvement. This model is grounded in a real-world guideline (MD Anderson Cancer Center) and published statistics on the disease (OrrantiaBorunda et al., 2022; Sezgın et al., 2020; Carey et al., 2006). In the **Engineering** domain, we implement an automatic fault detection algorithm for transmission lines (Reddy et al., 2016). This algorithm aims 

Table 1: _Average generalization performance_ across three realworld causal reasoning problems. The scores are normalized relative to the _Base_ approach’s scores in each generalization mode. Higher scores indicate a greater number of errors made by the approach, with scores above 1.0 meaning that the approach makes more mistakes than the _Base_ model, which has not undergone any fine-tuning. 

|**Mode**<br>**Metric**<br>**Base**<br>Base|**Mode**<br>**Metric**<br>**Base**<br>Base|**OnlyF**<br>SFT<br>DPO|**F&CF (Ours)**|
|---|---|---|---|
||||SFT<br>DPO<br>DPO+CCF|
|In-Domain<br>Avg-ER<br>Avg-IR|1.00<br>1.00|0.82<br>0.86<br>0.73<br>0.72|0.42<br>0.53<br>0.48<br>0.44<br>0.51<br>0.47|
|Common-Cause<br>Avg-ER<br>Avg-IR|1.00<br>1.00|1.35<br>1.75<br>1.30<br>1.49|1.17<br>1.62<br>2.04<br>1.14<br>1.42<br>1.80|
|Common-Effect<br>Avg-ER<br>Avg-IR|1.00<br>1.00|0.60<br>0.71<br>0.49<br>0.60|0.53<br>0.66<br>0.64<br>0.42<br>0.53<br>0.50|
|Inductive<br>Avg-ER<br>Avg-IR|1.00<br>1.00|0.86<br>0.79<br>0.74<br>0.62|0.60<br>0.58<br>0.69<br>0.51<br>0.50<br>0.59|
|Deductive, Cause-B.<br>Avg-ER<br>Avg-IR|1.00<br>1.00|0.70<br>0.67<br>0.62<br>0.59|0.61<br>0.58<br>0.57<br>0.53<br>0.53<br>0.51|
|Deductive, Effect-B.<br>Avg-ER<br>Avg-IR|1.00<br>1.00|0.83<br>0.83<br>0.91<br>0.87|0.90<br>0.95<br>0.77<br>0.93<br>0.89<br>0.76|



to identify the type of fault occurring on a transmission line using three different measurements. In the **Math Benchmarking** domain, we select a math question from GSM8K (Cobbe et al., 2021), a widely used benchmark for evaluating language models on grade school math problems. A detailed explanation of these three problems, including the context, factual and counterfactual questions, causal structures, and the cause-effect interventions demonstrated during the fine-tuning and evaluation phases across different generalization modes, can be found in Appendix C.2, C.3, C.4 respectively. 

**Results** The results for all three problems across in-domain and different generalization modes are available in Table 2 in Appendix A. Given the extensive number of experiments in this table, we have summarized the _Average Error Rate_ (Avg-ER) and _Average Inconsistency Rate_ (Avg-IR) scores in Table 1. For this summary, we first normalized the scores of each approach relative to the scores of the corresponding _Base_ approach. Then, for each generalization mode (including the in-domain scenario), we calculated the average score of each tested method across all applicable problems[6] . In Table 1, higher scores indicate more errors, and scores above 1.0 signify that the approach makes more mistakes than the _Base_ model. We observe that: (i) In the in-domain scenario, when the fine-tuning is guided by both factual and counterfactual examples ( _-F&CF_ ), the language models show a significant improvement in causal reasoning ability. (ii) Similar to what we observed in previous experiments, this improvement generalizes to most generalization modes, with the exception of _common-cause_ and _effect-based deduction_ . (iii) In most of modes, language models trained with causal consistency feedback ( _DPO+CCF_ ) demonstrate a lower error and inconsistency rate. 

## **7 Conclusion** 

This work introduced the problem of _fine-tuning for reasoning_ , along with (1) a taxonomy for generalization modes, (2) multiple metrics that address the limitations of existing performance measures, and (3) methods for generating fine-tuning data with counterfactual feedback. A key limitation of our approach is the restriction of causes and effects to binary variables, which allows us to focus on high-level relationships like necessity and sufficiency present in human reasoning. 

> 6Note that not all generalization modes were tested for every problem due to differences in causal structures, so the average scores were calculated using only the problems that were tested for each generalization mode. 

10 

## **References** 

- Abdin, M., S. A. Jacobs, A. A. Awan, J. Aneja, A. Awadallah, H. Awadalla, N. Bach, A. Bahree, A. Bakhtiari, H. Behl _et al._ , “Phi-3 technical report: A highly capable language model locally on your phone,” _arXiv preprint arXiv:2404.14219_ , 2024. 

- Ahn, J., R. Verma, R. Lou, D. Liu, R. Zhang, and W. Yin, “Large language models for mathematical reasoning: Progresses and challenges,” _arXiv preprint arXiv:2402.00157_ , 2024. 

- Ashwani, S., K. Hegde, N. R. Mannuru, M. Jindal, D. S. Sengar, K. C. R. Kathala, D. Banga, V. Jain, and A. Chadha, “Cause and effect: Can large language models truly understand causality?” _arXiv preprint arXiv:2402.18139_ , 2024. 

- Bubeck, S., V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg, H. Nori, H. Palangi, M. T. Ribeiro, and Y. Zhang, “Sparks of artificial general intelligence: Early experiments with GPT-4,” _arXiv preprint arXiv:2303.12712_ , 2023. 

- Carey, L. A., C. M. Perou, C. A. Livasy, L. G. Dressler, D. Cowan, K. Conway, G. Karaca, M. A. Troester, C. K. Tse, S. Edmiston _et al._ , “Race, breast cancer subtypes, and survival in the carolina breast cancer study,” _JAMA_ , vol. 295, no. 21, pp. 2492–2502, 2006. 

- Chang, Y., X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang _et al._ , “A survey on evaluation of large language models,” _ACM Transactions on Intelligent Systems and Technology_ , vol. 15, no. 3, pp. 1–45, 2024. 

- Cobbe, K., V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano _et al._ , “Training verifiers to solve math word problems,” _arXiv preprint arXiv:2110.14168_ , 2021. 

- Cohn, A. G. and J. Hernandez-Orallo, “Dialectical language model evaluation: An initial appraisal of the commonsense spatial reasoning abilities of LLMs,” _arXiv preprint arXiv:2304.11164_ , 2023. 

- Dai, A. M. and Q. V. Le, “Semi-supervised sequence learning,” in _Conference on Neural Information Processing Systems_ , 2015. 

- Gan, W., Z. Qi, J. Wu, and J. Lin, “Large language models in education: Vision and opportunities,” in _IEEE International Conference on Big Data_ , 2023. 

- Gandhi, K., J.-P. Franken,¨ T. Gerstenberg, and N. Goodman, “Understanding social reasoning in language models with language models,” _Advances in Neural Information Processing Systems_ , vol. 36, 2024. 

- Gonzalez, J. and A. V. Nori, “Does reasoning emerge?´ examining the probabilities of causation in large language models,” _arXiv preprint arXiv:2408.08210_ , 2024. 

- Gonzalez,´ J., C. Wong, Z. Gero, J. Bagga, R. Ueno, I. Chien, E. Orakvin, E. Kiciman, A. Nori, R. Weerasinghe, R. S. Leidner, B. Piening, T. Naumann, C. Bifulco, and H. Poon, “TRIALSCOPE: A unifying causal framework for scaling real-world evidence generation with biomedical language models,” _arXiv preprint arXiv:2311.01301_ , 2023. 

- Halpern, J. Y. and J. Pearl, “Causes and explanations: A structural-model approach. Part I: Causes,” _The British journal for the Philosophy of Science_ , 2005. 

- Howard, J. and S. Ruder, “Universal language model fine-tuning for text classification,” in _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 2018. 

- Huang, J. and K. C.-C. Chang, “Towards reasoning in large language models: A survey,” in _The 61st Annual Meeting of the Association for Computational Linguistics_ , 2023. 

- Jin, Z., Y. Chen, F. Leeb, L. Gresele, O. Kamal, Z. Lyu, K. Blin, F. Gonzalez, M. Kleiman-Weiner, M. Sachan, and B. Scholkopf, “Cladder: Assessing causal reasoning in language models,” in _Conference on Neural Information Processing Systems_ , 2023. 

11 

- Jin, Z., J. Liu, L. Zhiheng, S. Poff, M. Sachan, R. Mihalcea, M. T. Diab, and B. Scholkopf, “Can¨ large language models infer causation from correlation?” in _The Twelfth International Conference on Learning Representations_ , 2024. 

- Kandpal, N., H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large language models struggle to learn long-tail knowledge,” in _International Conference on Machine Learning_ , 2023. 

Kelley, D., _The Art of Reasoning with Symbolic Logic_ . WW Norton & Co, 1992. 

- Khalifa, M., H. Elsahar, and M. Dymetman, “A distributional approach to controlled text generation,” _arXiv preprint arXiv:2012.11635_ , 2020. 

- Khandelwal, U., K. Clark, D. Jurafsky, and L. Kaiser, “Sample efficient text summarization using a single pre-trained transformer,” _arXiv preprint arXiv:1905.08836_ , 2019. 

- Kıcıman, E., R. Ness, A. Sharma, and C. Tan, “Causal reasoning and large language models: Opening a new frontier for causality,” _arXiv preprint arXiv:2305.00050_ , 2023. 

- Korbak, T., H. Elsahar, G. Kruszewski, and M. Dymetman, “Controlling conditional language models without catastrophic forgetting,” in _International Conference on Machine Learning_ . PMLR, 2022, pp. 11 499–11 528. 

- Korbak, T., H. Elsahar, G. Kruszewski, and M. Dymetman, “On reinforcement learning and distribution matching for fine-tuning language models with no catastrophic forgetting,” _Advances in Neural Information Processing Systems_ , vol. 35, pp. 16 203–16 220, 2022. 

Lewis, D., “Causation,” _Journal of Philosophy_ , vol. 70, pp. 556–567, 1973. 

- Li, J., L. Yu, and A. Ettinger, “Counterfactual reasoning: Do language models need world knowledge for causal inference?” in _NeurIPS 2022 Workshop on Neuro Causal and Symbolic AI (nCSI)_ , 2022. 

- Li, J., L. Yu, and A. Ettinger, “Counterfactual reasoning: Testing language models’ understanding of hypothetical scenarios,” in _The 61st Annual Meeting Of The Association For Computational Linguistics_ , 2023. 

- Li, Y., W. Tian, Y. Jiao, J. Chen, and Y.-G. Jiang, “Eyes can deceive: Benchmarking counterfactual reasoning abilities of multi-modal large language models,” _arXiv preprint arXiv:2404.12966_ , 2024. 

- Lin, S.-C., L. Gao, B. Oguz, W. Xiong, J. Lin, W.-t. Yih, and X. Chen, “Flame: Factuality-aware alignment for large language models,” _arXiv preprint arXiv:2405.01525_ , 2024. 

- Lin, V., E. Ben-Michael, and L.-P. Morency, “Optimizing language models for human preferences is a causal inference problem,” _arXiv preprint arXiv:2402.14979_ , 2024. 

MacColl, H., “Symbolic reasoning,” _Mind_ , vol. 6, no. 24, pp. 493–510, 1897. 

- Mackie, J. L., “Causes and conditions,” _American Philosophical Quarterly_ , vol. 2, no. 4, pp. 245–264, 1965. 

- MD Anderson Cancer Center,. Cancer treatment algorithms - breast cancer - invasive stage i-iii. [Online]. Available: https://www.mdanderson.org/content/dam/mdanderson/documents/ for-physicians/algorithms/cancer-treatment/ca-treatment-breast-invasive-web-algorithm.pdf 

- Nguyen, V. B., P. Youssef, J. Schlotterer,¨ and C. Seifert, “LLMs for generating and evaluating counterfactuals: A comprehensive study,” _arXiv preprint arXiv:2405.00722_ , 2024. 

- Orrantia-Borunda, E., P. Anchondo-Nunez,˜ L. E. Acuna-Aguilar,˜ F. O. Gomez-Valles,´ and C. A. Ram´ırez-Valdespino, “Subtypes of breast cancer,” _Breast Cancer [Internet]_ , 2022. 

- Ouyang, L., J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” _Conference on Neural Information Processing Systems_ , 2022. 

12 

- Parmar, M., N. Patel, N. Varshney, M. Nakamura, M. Luo, S. Mashetty, A. Mitra, and C. Baral, “LogicBench: Towards systematic evaluation of logical reasoning ability of large language models,” in _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , 2024. 

- Pearl, J., “Probabilities of causation: Three counterfactual interpretations and their identification,” _Synthese_ , vol. 121, no. 1–2, pp. 93–149, 1999. 

Pearl, J., _Causality: Models, Reasoning, and Inference_ . Cambridge University Press, 2000. 

- Peters, M. E., M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, “Deep contextualized word representations,” in _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , 2018. 

- Radford, A., K. Narasimhan, T. Salimans, and I. Sutskever. (2018) Improving language understanding by generative pre-training. [Online]. Available: https://s3-us-west-2.amazonaws. com/openai-assets/research-covers/language-unsupervised/language ~~u~~ nderstanding ~~p~~ aper.pdf 

- Rafailov, R., A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” _Conference on Neural Information Processing Systems_ , 2024. 

- Reddy, M. J. B., P. Gopakumar, and D. Mohanta, “A novel transmission line protection using dost and svm,” _Engineering Science and Technology, an International Journal_ , vol. 19, no. 2, pp. 1027–1039, 2016. 

- Seals, S. and V. Shalin, “Evaluating the deductive competence of large language models,” in _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , 2024. 

- Sezgın, G., M. Apaydın, D. Etıt, and M. K. Atahan, “Tumor size estimation of the breast cancer molecular subtypes using imaging techniques,” _Medicine and Pharmacy Reports_ , vol. 93, no. 3, p. 253, 2020. 

- Simmering, P. F. and P. Huoviala, “Large language models for aspect-based sentiment analysis,” _arXiv preprint arXiv:2310.18025_ , 2023. 

- Tian, K., E. Mitchell, H. Yao, C. D. Manning, and C. Finn, “Fine-tuning language models for factuality,” _arXiv preprint arXiv:2311.08401_ , 2023. 

- Tong, Y., D. Li, S. Wang, Y. Wang, F. Teng, and J. Shang, “Can llms learn from previous mistakes? investigating llms’ errors to boost for reasoning,” _arXiv preprint arXiv:2403.20046_ , 2024. 

- Wang, Z., “CausalBench: A comprehensive benchmark for evaluating causal reasoning capabilities of large language models,” in _Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10)_ , 2024. 

- Wei, J., X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou _et al._ , “Chain-ofthought prompting elicits reasoning in large language models,” _Conference on Neural Information Processing Systems_ , 2022. 

- Wong, C., S. Zhang, Y. Gu, C. Moung, J. Abel, N. Usuyama, R. Weerasinghe, B. Piening, T. Naumann, C. Bifulco, and H. Poon, “Scaling clinical trial matching using large language models: A case study in oncology,” in _Machine Learning for Healthcare Conference_ , 2023. 

- Wu, T., M. T. Ribeiro, J. Heer, and D. S. Weld, “Polyjuice: Generating counterfactuals for explaining, evaluating, and improving models,” in _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , 2021. 

- Zhang, C., S. Bauer, P. Bennett, J. Gao, W. Gong, A. Hilmkil, J. Jennings, C. Ma, T. Minka, N. Pawlowski _et al._ , “Understanding causality with large language models: Feasibility and opportunities,” _arXiv preprint arXiv:2304.05524_ , 2023. 

13 

Zhang, C., D. Ippolito, K. Lee, M. Jagielski, F. Tramer, and N. Carlini, “Counterfactual memorization` in neural language models,” in _Conference on Neural Information Processing Systems_ , 2023. 

- Ziegler, D. M., N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, “Fine-tuning language models from human preferences,” _arXiv preprint arXiv:1909.08593_ , 2019. 

14 

## **A Breakdown of the Results in Section 6** 

Table 2: _Results of Healthcare, Engineering, and Math Benchmarking problems._ For some scenarios in _Math Benchmarking_ , N-IR and AS-IR are equal to 0.00 for all algorithms because the target cause _X_ is never present without an intervention due to how these scenarios are structured. 

|**Scenar**|**io**<br>**Alg.**|**Correctness**<br>F-ER<br>C-ER<br>Avg-ER|**Causal Consistency**|
|---|---|---|---|
||||N-IR<br>S-IR<br>AN-IR<br>AS-IR<br>Avg-IR|
|**Healthcare**<br>Breast Cancer Treatment|Base|23.57(0.00)<br>28.93(0.00)<br>26.25(0.00)|20.62(0.00)<br>4.34(0.00)<br>11.47(0.00)<br>32.01(0.00)<br>17.11(0.00)|
||In-domain<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|3.12(0.04)<br>20.82(0.02)<br>11.97(0.02)<br>10.77(1.83)<br>28.57(2.07)<br>19.67(1.94)<br>1.64(0.01)<br>0.08(0.00)<br>0.86(0.00)<br>12.53(5.04)<br>8.74(1.69)<br>10.63(3.14)<br>9.55(1.16)<br>5.16(0.15)<br>7.36(0.52)|1.50(0.01)<br>2.09(0.02)<br>3.01(0.02)<br>19.93(0.01)<br>6.63(0.01)<br>9.10(2.24)<br>4.76(0.10)<br>6.00(0.16)<br>25.32(1.10)<br>11.29(0.66)<br>0.85(0.00)<br>0.80(0.00)<br>0.82(0.00)<br>0.87(0.00)<br>0.84(0.00)<br>8.65(3.04)<br>4.32(0.30)<br>5.00(0.28)<br>11.07(4.93)<br>7.26(1.58)<br>7.97(1.21)<br>1.70(0.01)<br>3.59(0.04)<br>9.31(1.51)<br>5.64(0.36)|
||Com-Cause<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|48.86(3.27)<br>38.82(0.91)<br>43.84(1.65)<br>64.56(1.93)<br>67.95(0.19)<br>66.25(0.70)<br>25.99(0.55)<br>38.14(0.81)<br>32.07(0.07)<br>53.92(2.45)<br>63.62(4.52)<br>58.77(2.10)<br>80.18(4.78)<br>72.85(0.21)<br>76.52(1.55)|42.38(2.82)<br>10.55(0.10)<br>13.96(0.03)<br>45.71(2.06)<br>28.15(0.65)<br>53.01(1.45)<br>13.20(0.06)<br>14.25(0.00)<br>64.77(0.58)<br>36.31(0.28)<br>25.83(0.35)<br>4.10(0.04)<br>13.07(0.01)<br>41.42(0.38)<br>21.10(0.06)<br>46.27(0.44)<br>14.09(0.01)<br>14.44(0.04)<br>53.23(2.34)<br>32.01(0.33)<br>70.55(1.67)<br>15.96(0.03)<br>14.93(0.00)<br>68.67(3.67)<br>42.53(0.69)|
||Com-Effect<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|3.16(0.04)<br>20.87(0.02)<br>12.01(0.02)<br>5.93(0.02)<br>30.30(2.44)<br>18.11(0.54)<br>1.45(0.01)<br>20.43(0.02)<br>10.94(0.00)<br>2.96(0.03)<br>22.44(0.02)<br>12.70(0.00)<br>2.17(0.04)<br>22.78(0.11)<br>12.48(0.07)|1.45(0.01)<br>2.21(0.02)<br>2.94(0.03)<br>20.06(0.01)<br>6.67(0.01)<br>11.75(2.24)<br>3.77(0.00)<br>4.61(0.00)<br>21.60(0.03)<br>10.43(0.12)<br>1.60(0.00)<br>0.53(0.00)<br>1.40(0.00)<br>19.68(0.01)<br>5.80(0.00)<br>1.40(0.01)<br>2.08(0.03)<br>4.14(0.01)<br>19.84(0.00)<br>6.86(0.00)<br>0.36(0.00)<br>2.58(0.07)<br>3.16(0.05)<br>19.60(0.00)<br>6.42(0.02)|
||Deductive<br>(Cause-Based)<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|1.56(0.01)<br>20.30(0.01)<br>10.93(0.01)<br>18.52(4.41)<br>32.93(4.36)<br>25.73(4.34)<br>1.32(0.00)<br>22.37(0.00)<br>11.84(0.00)<br>4.25(0.14)<br>24.37(0.14)<br>14.31(0.12)<br>5.05(0.04)<br>25.17(0.17)<br>15.11(0.07)|1.14(0.00)<br>0.65(0.01)<br>1.40(0.01)<br>20.12(0.00)<br>5.83(0.00)<br>13.03(2.76)<br>6.52(0.21)<br>6.80(0.20)<br>30.30(2.82)<br>14.16(1.11)<br>1.28(0.00)<br>0.15(0.00)<br>3.07(0.00)<br>20.48(0.00)<br>6.24(0.00)<br>1.45(0.04)<br>3.21(0.09)<br>5.99(0.09)<br>20.36(0.03)<br>7.75(0.04)<br>2.95(0.03)<br>3.97(0.07)<br>6.42(0.08)<br>20.12(0.00)<br>8.36(0.02)|
|**Engineering**<br>Transmission Line Protection|Base|16.39(0.00)<br>27.10(0.00)<br>21.75(0.00)|13.70(0.00)<br>27.52(0.00)<br>2.71(0.00)<br>13.96(0.00)<br>14.47(0.00)|
||In-domain<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|2.94(0.05)<br>29.64(0.06)<br>16.29(0.01)<br>3.76(0.05)<br>31.30(0.06)<br>17.53(0.00)<br>1.39(0.00)<br>6.43(0.05)<br>3.91(0.01)<br>10.23(2.60)<br>33.48(1.15)<br>21.86(1.76)<br>7.64(0.89)<br>23.00(0.60)<br>15.32(0.71)|4.84(0.02)<br>26.73(0.00)<br>0.94(0.02)<br>2.18(0.01)<br>8.67(0.00)<br>6.63(0.01)<br>26.42(0.00)<br>1.00(0.00)<br>3.23(0.05)<br>9.32(0.00)<br>3.27(0.04)<br>4.32(0.03)<br>0.46(0.00)<br>0.93(0.00)<br>2.25(0.00)<br>6.94(0.29)<br>30.13(0.77)<br>5.34(1.23)<br>6.87(0.82)<br>12.32(0.72)<br>6.61(0.12)<br>19.61(0.77)<br>4.44(0.32)<br>4.03(0.37)<br>8.67(0.32)|
||Com-Cause<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|14.61(0.08)<br>30.28(0.17)<br>22.44(0.11)<br>11.25(0.15)<br>30.78(0.20)<br>21.02(0.18)<br>14.05(0.10)<br>34.43(0.15)<br>24.24(0.12)<br>12.52(0.17)<br>31.46(0.10)<br>21.99(0.11)<br>14.20(0.09)<br>35.72(0.00)<br>24.96(0.02)|13.80(0.08)<br>27.23(0.00)<br>0.81(0.00)<br>14.02(0.07)<br>13.96(0.02)<br>11.04(0.15)<br>26.60(0.00)<br>0.21(0.00)<br>11.73(0.16)<br>12.39(0.04)<br>13.13(0.06)<br>32.64(0.18)<br>0.92(0.01)<br>13.60(0.06)<br>15.07(0.05)<br>10.84(0.12)<br>31.40(0.12)<br>1.72(0.08)<br>11.41(0.11)<br>13.84(0.06)<br>13.26(0.07)<br>35.68(0.01)<br>0.94(0.01)<br>13.56(0.06)<br>15.86(0.02)|
||Com-Effect<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|2.71(0.05)<br>29.26(0.04)<br>15.99(0.01)<br>2.71(0.05)<br>29.26(0.04)<br>15.99(0.01)<br>0.95(0.00)<br>26.41(0.00)<br>13.68(0.00)<br>2.28(0.09)<br>34.30(1.84)<br>18.29(0.48)<br>1.51(0.00)<br>32.98(2.12)<br>17.25(0.49)|4.39(0.03)<br>26.73(0.00)<br>0.78(0.01)<br>2.07(0.02)<br>8.49(0.01)<br>4.39(0.03)<br>26.73(0.00)<br>0.78(0.01)<br>2.07(0.02)<br>8.49(0.01)<br>0.82(0.00)<br>26.41(0.00)<br>0.15(0.00)<br>0.80(0.00)<br>7.04(0.00)<br>3.04(0.10)<br>25.71(0.02)<br>0.46(0.00)<br>8.74(1.32)<br>9.49(0.13)<br>2.41(0.04)<br>26.55(0.00)<br>0.22(0.00)<br>6.50(1.23)<br>8.92(0.11)|
||Inductive<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|2.58(0.02)<br>31.10(0.10)<br>16.84(0.03)<br>2.68(0.05)<br>31.36(0.07)<br>17.02(0.00)<br>1.43(0.00)<br>20.25(0.17)<br>10.84(0.04)<br>2.85(0.02)<br>22.23(0.69)<br>12.54(0.22)<br>2.34(0.06)<br>26.23(0.01)<br>14.29(0.02)|5.85(0.07)<br>26.41(0.00)<br>0.57(0.00)<br>2.38(0.03)<br>8.80(0.01)<br>6.22(0.00)<br>25.69(0.01)<br>0.46(0.00)<br>2.89(0.06)<br>8.81(0.00)<br>0.91(0.00)<br>20.53(0.16)<br>0.52(0.00)<br>0.91(0.00)<br>5.72(0.01)<br>2.53(0.03)<br>22.27(0.67)<br>0.33(0.00)<br>2.56(0.03)<br>6.92(0.07)<br>1.37(0.01)<br>26.03(0.01)<br>1.05(0.04)<br>1.47(0.01)<br>7.48(0.01)|
|**Math Benchmarking**<br>Testing on: _S →T_|Base|47.35(0.00)<br>26.32(0.00)<br>36.84(0.00)|0.00(0.00)<br>61.50(0.00)<br>48.24(0.00)<br>0.00(0.00)<br>27.44(0.00)|
||In-domain<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|37.21(0.58)<br>28.91(0.08)<br>33.06(0.25)<br>28.28(0.95)<br>57.39(3.22)<br>42.83(1.84)<br>22.52(0.18)<br>14.05(0.17)<br>18.29(0.07)<br>14.06(0.10)<br>26.87(1.71)<br>20.47(0.32)<br>19.38(0.04)<br>36.13(0.44)<br>27.75(0.19)|0.00(0.00)<br>52.17(0.41)<br>40.10(0.69)<br>0.00(0.00)<br>23.07(0.14)<br>0.00(0.00)<br>66.84(1.15)<br>29.78(0.79)<br>0.00(0.00)<br>24.15(0.24)<br>0.00(0.00)<br>32.07(0.22)<br>23.45(0.18)<br>0.00(0.00)<br>13.88(0.05)<br>0.00(0.00)<br>37.78(1.52)<br>15.90(0.07)<br>0.00(0.00)<br>13.42(0.07)<br>0.00(0.00)<br>50.89(1.13)<br>21.03(0.02)<br>0.00(0.00)<br>17.98(0.09)|
||Inductive<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|37.25(0.78)<br>33.46(0.02)<br>35.36(0.17)<br>15.31(0.27)<br>42.93(0.01)<br>29.12(0.10)<br>24.27(0.08)<br>27.13(0.20)<br>25.70(0.04)<br>16.13(1.16)<br>26.86(1.39)<br>21.49(0.31)<br>22.21(1.33)<br>29.94(1.19)<br>26.07(0.34)|0.00(0.00)<br>53.39(0.46)<br>42.11(0.62)<br>0.00(0.00)<br>23.88(0.13)<br>0.00(0.00)<br>52.54(0.31)<br>18.06(0.28)<br>0.00(0.00)<br>17.65(0.07)<br>0.00(0.00)<br>40.98(0.05)<br>27.00(0.11)<br>0.00(0.00)<br>16.99(0.02)<br>0.00(0.00)<br>38.28(0.95)<br>18.44(1.08)<br>0.00(0.00)<br>14.18(0.18)<br>0.00(0.00)<br>48.44(1.23)<br>23.51(1.16)<br>0.00(0.00)<br>17.99(0.23)|
|**Math Benchmarking**<br>Testing on: _S →R_|Base|50.25(0.00)<br>51.11(0.00)<br>50.68(0.00)|0.00(0.00)<br>67.30(0.00)<br>58.65(0.00)<br>0.00(0.00)<br>31.49(0.00)|
||In-domain<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|45.16(0.01)<br>56.90(0.13)<br>51.03(0.03)<br>39.11(0.06)<br>72.29(0.45)<br>55.70(0.16)<br>44.55(0.01)<br>32.88(0.09)<br>38.71(0.01)<br>15.43(3.99)<br>8.56(0.68)<br>11.99(1.99)<br>17.12(1.78)<br>8.06(0.87)<br>12.59(1.26)|0.00(0.00)<br>68.02(0.02)<br>52.95(0.05)<br>0.00(0.00)<br>30.24(0.00)<br>0.00(0.00)<br>76.30(0.06)<br>40.57(0.06)<br>0.00(0.00)<br>29.22(0.00)<br>0.00(0.00)<br>62.18(0.00)<br>47.15(0.01)<br>0.00(0.00)<br>27.33(0.00)<br>0.00(0.00)<br>20.06(5.01)<br>16.48(3.99)<br>0.00(0.00)<br>9.14(1.12)<br>0.00(0.00)<br>23.48(4.74)<br>17.88(1.72)<br>0.00(0.00)<br>10.34(0.75)|
||Deductive<br>(Effect-Based)<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|46.17(0.03)<br>42.46(0.18)<br>44.31(0.05)<br>39.36(0.78)<br>48.96(2.24)<br>44.16(0.38)<br>46.57(0.04)<br>48.58(1.08)<br>47.57(0.22)<br>39.50(0.00)<br>61.68(4.62)<br>50.59(1.15)<br>31.05(0.96)<br>50.27(4.58)<br>40.66(1.93)|0.00(0.00)<br>66.31(0.04)<br>49.46(0.10)<br>0.00(0.00)<br>28.94(0.00)<br>0.00(0.00)<br>65.83(0.41)<br>44.21(1.15)<br>0.00(0.00)<br>27.51(0.06)<br>0.00(0.00)<br>66.05(0.10)<br>52.63(0.05)<br>0.00(0.00)<br>29.67(0.01)<br>0.00(0.00)<br>73.43(0.47)<br>39.61(0.00)<br>0.00(0.00)<br>28.26(0.03)<br>0.00(0.00)<br>63.75(2.23)<br>32.92(0.80)<br>0.00(0.00)<br>24.17(0.34)|
|**Math Benchmarking**<br>Testing on: _R →T_|Base|47.35(0.00)<br>58.82(0.00)<br>53.08(0.00)|28.77(0.00)<br>50.11(0.00)<br>25.69(0.00)<br>22.44(0.00)<br>31.76(0.00)|
||In-domain<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|36.07(0.97)<br>65.96(0.02)<br>51.01(0.30)<br>28.28(0.95)<br>22.81(0.12)<br>25.54(0.42)<br>23.12(0.12)<br>45.03(0.13)<br>34.08(0.11)<br>24.33(2.42)<br>22.38(0.16)<br>23.35(0.47)<br>21.03(0.05)<br>20.91(0.55)<br>20.97(0.16)|21.32(0.09)<br>54.11(0.02)<br>26.60(0.27)<br>9.97(0.25)<br>28.00(0.13)<br>32.30(0.31)<br>2.88(0.01)<br>0.24(0.00)<br>28.46(0.92)<br>15.97(0.14)<br>16.87(0.02)<br>39.39(0.09)<br>18.67(0.03)<br>4.76(0.04)<br>19.92(0.04)<br>30.27(0.30)<br>6.11(0.45)<br>3.74(0.46)<br>21.18(1.63)<br>15.32(0.39)<br>27.75(0.11)<br>5.52(0.17)<br>0.83(0.01)<br>20.58(0.06)<br>13.67(0.03)|
||Deductive<br>(Cause-Based)<br>SFT-OnlyF<br>DPO-OnlyF<br>SFT-F&CF<br>DPO-F&CF<br>DPO+CCF|37.87(0.81)<br>62.75(0.10)<br>50.31(0.36)<br>15.31(0.27)<br>19.98(0.03)<br>17.65(0.08)<br>25.41(0.10)<br>53.63(0.15)<br>39.52(0.05)<br>24.33(1.01)<br>38.62(6.13)<br>31.48(2.94)<br>15.97(0.42)<br>40.45(3.76)<br>28.21(1.45)|23.38(0.11)<br>51.63(0.07)<br>26.27(0.28)<br>12.13(0.25)<br>28.35(0.14)<br>23.77(0.23)<br>4.04(0.02)<br>0.24(0.00)<br>15.61(0.28)<br>10.91(0.06)<br>19.12(0.04)<br>44.40(0.03)<br>18.22(0.07)<br>7.55(0.12)<br>22.32(0.02)<br>29.16(0.35)<br>22.64(5.93)<br>0.66(0.00)<br>24.69(1.07)<br>19.29(1.00)<br>23.14(0.23)<br>25.35(3.48)<br>1.85(0.02)<br>14.88(0.56)<br>16.31(0.46)|



15 

## **B Description of the Algorithms in Section 4** 

## **Algorithm 1** Supervised Counterfactual Feedback 

|1:|**Inputs:** Demonstrations_D_ =_{PXi→Yi}_, question templates_q,_˜_q_, answer generator_H_|**Inputs:** Demonstrations_D_ =_{PXi→Yi}_, question templates_q,_˜_q_, answer generator_H_|
|---|---|---|
|2:|**Output:** DatasetD=_{q, a}_of questions and answer pairs||
|3:|D_←{}_||
|4:|**for**_PXi→Yi ∈D_,_n ∈{_1_, . . . , N}_**do**||
|5:|_u, X, Y, YX′ ∼PXi→Yi_|_▷_Sample a context, cause, and potential effects|
|6:|_qf ←q_(_u_),_qcf ←_˜_qX′_(_u_)|_▷_Generate questions given the context|
|7:|_af ←H_(_qf, YX_),_acf ←H_(_qcf, YX′_)|_▷_Generate answers given the correct outcome|
|8:|D_←_D_∪{_(_qf, af_)_,_(_qcf, acf_)_}_||
|9:|**end for**||



## **Algorithm 2** Preference-based Counterfactual Feedback 

|**Alg**|**orithm 2**Preference-based Counterfactual Feedback|**orithm 2**Preference-based Counterfactual Feedback|
|---|---|---|
|1:|**Inputs:** Demonstrations_D_ =_{PXi→Yi}_, question templates_q,_˜_q_, answer extractor_h_, target model_ℓ_0||
|2:|**Output:** DatasetD=_{_(_q, a_)_≻_(_q′, a′_)_}_of preferences over question-answer pairs||
|3:|D_←{}_||
|4:|**for**_PXi→Yi ∈D_,_n ∈{_1_, . . . , N}_**do**||
|5:|_u, X, Y, YX′ ∼PXi→Yi_|_▷_Sample a context, cause, and potential effects|
|6:|_qf ←q_(_u_),_qcf ←_˜_qX′_(_u_)|_▷_Generate questions given the context|
|7:|**for**_m ∈{_1_, . . . , M}_**do**||
|8:|_af_[_m_]_∼ℓ_0(_qf_),_acf_[_m_]_∼ℓ_0(_qcf_)|_▷_Collect answers from the language model|
|9:|ˆ_YX_[_m_]_←h_(_af_[_m_]), ˆ_YX′_[_m_]_←h_(_acf_[_m_])|_▷_Extract outcome estimates from answers|
|10:|**end for**||
|11:|**for**_m ∈{_1_, . . . , M}_,_m′ ∈{_1_, . . . , M}_**do**||
|12:|**if**_YX_ = ˆ_YX_[_m_] = ˆ_YX_[_m′_]**then**||
|13:|D_←_D_∪{_(_qf, af_[_m_])_≻_(_qf, af_[_m′_])_}_|_▷_Elicit factual preferences|
|14:|**end if**||
|15:|**if**_YX′_ = ˆ_YX′_[_m_] = ˆ_YX′_[_m′_]**then**||
|16:|D_←_D_∪{_(_qcf, acf_[_m_])_≻_(_qcf, acf_[_m′_])_}_|_▷_Elicit counterfactual preferences|
|17:|**end if**||
|18:|**end for**||
|19:|**end for**||



## **Algorithm 3** Preference-based Causal Consistency Feedback 

1: **Inputs:** Demonstrations _D_ = _{PXi→Yi }_ , question templates _q,_ ˜ _q_ , answer extractor _h_ , target model _ℓ_ 0 2: **Output:** Dataset D = _{_ ( _qf, af, qcf, acf_ ) _≻_ ( _qf[′][, a][′] f[, q] cf[′][, a][′] cf_[)] _[}]_[ of preferences over dialogues] 3: D _←{}_ 4: **for** _PXi→Yi ∈D_ , _n ∈{_ 1 _, . . . , N }_ **do** 5:6: _u, X, Y, Yqf ← q_ ( _u_ ) _X_ , _q′ cf∼P← Xq_ ˜ _Xi→′_ ( _Yui_ ) _▷_ Sample a context, cause, and potential effects _▷_ Generate questions given the context 7: **for** _m ∈{_ 1 _, . . . , M }_ **do** 8: _a_ ˆ _f_ [ _m_ ] _∼ ℓ_ 0( _qf_ ), _acf_ [ _m_ ˆ] _∼ ℓ_ 0( _qf, af_ [ _m_ ] _, qcf_ ) _▷_ Collect answers from the language model 9: _YX_ [ _m_ ] _← h_ ( _af_ [ _m_ ]), _yX′_ [ _m_ ] _← h_ ( _acf_ [ _m_ ]) _▷_ Extract outcome estimates from answers 10: _r_ [ _m_ ] _←R_ ( _Y_[ˆ] _X_ [ _m_ ] _, Y_[ˆ] _X′_ [ _m_ ]; _u_ ), _r_ [ _m[′]_ ] _←R_ ( _Y_[ˆ] _X_ [ _m[′]_ ] _, Y_[ˆ] _X′_ [ _m[′]_ ]; _u_ ) _▷_ Compute consistency scores 11: **end for** 12: **for** _m ∈{_ 1 _, . . . , M }_ , _m[′] ∈{_ 1 _, . . . , M }_ **do** 13: **if** _r_ [ _m_ ] _≥ r_ [ _m[′]_ ] **then** 14: D _←_ D _∪{_ ( _qf, af_ [ _m_ ] _, qcf, acf_ [ _m_ ]) _≻_ ( _qf, af_ [ _m[′]_ ] _, qcf, acf_ [ _m[′]_ ]) _} ▷_ Elicit preferences 15: **end if** 16: **end for** 17: **end for** 

16 

## **C Details of the Experiments** 

When collecting datasets, we always sample 100 contexts and generate 10 answers for each question per context. In order to obtain error bars, we repeat each experiment five times. The extractor _h_ is implemented using Llama 3 8B with the following prompt: 

I will give you a question and its answer. Determine whether the meaning of the answer is ‘POSITIVE’ or ‘NEGATIVE’. An answer is ‘POSITIVE’ if it contains phrases like ‘yes’, ‘it holds’, ‘correct’, ‘true’, or similar affirmations. An answer is ‘NEGATIVE’ if it contains phrases like ‘no’, ‘it does not hold’, ‘incorrect’, ‘false’, or similar negations. Respond only with one word: ‘POSITIVE’ or ‘NEGATIVE’. Question: ‘ _{q}_ ’ Answer: ‘ _{a}_ ’ Is the meaning ‘POSITIVE’ or ‘NEGATIVE’? 

Similarly, the answer generator _H_ , in the case of supervised counterfactual feedback, is implemented using Llama 3 8B with the following prompt: 

I will give you a question and the initial word of its answer. Complete the answer starting form the provided word. Respond only with the complete answer. Question: _{q}_ Answer: _{_ No/Yes _}_ , ... 

## **C.1 Reasoning Problem: Puzzle** 

This hand-crafted puzzle, centered around a candy party, has been used in the experiments conducted in Section 6.1 and 6.2. Based on different generalization modes, we developed three variations of this puzzle, each featuring distinct causal structures: 

## **Structure-1: Bipartite Graph** 

- **Context:** Anna, Bill, Cory, and Dave are going to a party, where the host is going to distribute candies. Anna will be happy if she gets at least 4 candies. Bill will be happy if he gets at least 6 candies. Cory will be happy if Anna and Bill are both happy or if he gets at least 8 candies. Dave will be happy if Anna and Bill are both happy or if he gets at least 10 candies. After distributing the candies, Anna gets _{NA}_ , Bill gets _{NB}_ , Cory gets _{NC}_ , and Dave gets _{ND}_ . 

- **Factual Question:** Is _{_ Anna/Bill/Cory/Dave _}_ happy? Be as concise as possible. 

- **Interventional Question:** Now, suppose that _{_ Anna/Bill/Cory/Dave _} {_ is/is not _}_ happy regardless of the candy distribution. With this assumption, is _{_ Anna/Bill/Cory/Dave _}_ happy? Be as concise as possible. 

- **Causal Relationships:** 

_A_ = _NA ≥_ 4 (12) _B_ = _NB ≥_ 6 (13) _C_ = ( _A ∧ B_ ) _∨_ ( _NC ≥_ 8) (14) _D_ = ( _A ∧ B_ ) _∨_ ( _ND ≥_ 10) (15) 

- **Causal Structure:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0017-14.png)


## **Structure-2: Chain with No Direct Effect (NDE)** 

- **Context:** Anna, Bill, and Cory are going to a party, where the host is going to distribute candies. Anna will be happy if she gets at least 5 candies. Bill will be happy if Anna is happy or if he gets at least 7 candies. Cory will be happy if Bill is happy or if he gets at least 9 candies. After distributing the candies, Anna gets _{NA}_ , Bill gets _{NB}_ , and Cory gets _{NC}_ . 

17 

- **Factual Question:** Is _{_ Anna/Bill/Cory _}_ happy? Be as concise as possible. 

- **Interventional Question:** Now, suppose that _{_ Anna/Bill/Cory _} {_ is/is not _}_ happy regardless of the candy distribution. With this assumption, is _{_ Anna/Bill/Cory _}_ happy? Be as concise as possible. 

- **Causal Relationships:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0018-03.png)


- **Causal Structure:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0018-05.png)


## **Structure-3: Chain With Direct Effect (WDE)** 

- **Context:** Anna, Bill, and Cory are going to a party, where the host is going to distribute candies. Anna will be happy if she gets at least 5 candies. Bill will be happy if Anna is happy or if he gets at least 7 candies. Cory will be happy if Annd and Bill are both happy or if he gets at least 9 candies. After distributing the candies, Anna gets _{NA}_ , Bill gets _{NB}_ , and Cory gets _{NC}_ . 

- **Factual Question:** Is _{_ Anna/Bill/Cory _}_ happy? Be as concise as possible. 

- **Interventional Question:** Now, suppose that _{_ Anna/Bill/Cory _} {_ is/is not _}_ happy regardless of the candy distribution. With this assumption, is _{_ Anna/Bill/Cory _}_ happy? Be as concise as possible. 

- **Causal Relationships:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0018-11.png)


- **Causal Structure:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0018-13.png)


## **C.2 Real-world Reasoning Problem: Healthcare** 

In Section 6.3 we introduced three real-world problems to validate our experimental findings from the proof-of-concept puzzle reasoning problem. Here, we offer a more detailed explanation of one of these real-world reasoning problems—the **Healthcare** problem. 

- **Context:** There are four types of breast cancer patients (based on their ERPR and HER2 indicators): (1) If a patient is ERPR positive and HER2 negative, they are ‘Luminal A’. All luminal A patients should undergo surgery. (2) If a patient is ERPR positive and HER2 positive, they are ‘Luminal B’. Luminal B patients should undergo surgery if their tumor is smaller than 1 cm and there is no nodal involvement. Luminal B patients should undergo therapy if their tumor is larger than 1 cm or if there is nodal involvement. (3) If a patient is ERPR negative and HER2 positive, they are ‘Enriched’. Enriched patients should undergo surgery if their tumor is smaller than 1 cm and there is no nodal involvement. Enriched patients should undergo therapy only if their tumor is larger than 1 cm (even if there is nodal involvement). (4) If a patient is ERPR negative and HER2 negative, they are ‘Basal’. Basal patients should undergo surgery if their tumor is smaller than 1 cm and there is no nodal involvement. Basal patients should undergo therapy only if their tumor is larger than 1 cm (even if there is nodal involvement). 

18 

Jane is ERPR _{_ negative/positive _}_ and HER2 _{_ negative/positive _}_ . Her tumor is _{T_ cm _}_ cm and there is _{_ nodal involvement/no nodal involvement _}_ . 

- **Factual Question:** Will she undergo _{_ surgery/therapy _}_ ? Be as concise as possible. 

- **Possible Interventional Questions:** If _{_ Jane had been ERPR positive/Jane had been ERPR negative/Jane had been HER2 positive/Jane had been HER2 negative/the tumor had been larger than 1 cm/the tumor had been smaller than 1 cm/there had been nodal involvement/there had been no nodal involvement _}_ , would she have undergone _{_ surgery/therapy _}_ ? Be as concise as possible. 

- **Causal Relationships:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0019-04.png)


- **Causal Structure and In-domain Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0019-06.png)


19 

- **Causal Structure and Common-Cause Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0020-01.png)


- **Causal Structure and Common-Effect Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0020-03.png)


- **Causal Structure and Cause-Based Deduction Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0020-05.png)


## **C.3 Real-World Reasoning Problem: Engineering** 

In Section 6.3 we introduced three real-world problems to validate our experimental findings from the proof-of-concept puzzle reasoning problem. Here, we offer a more detailed explanation of one of these real-world reasoning problems—the **Engineering** problem. 

- **Context:** The type of fault on a transmission line is determined through three factors X, Y, and Z. These factors are ‘close to zero’ if they are less than 0.1. (1) If only one of the factors is close to zero, it is a line-to-line fault. When there is a line-to-line fault, it is BC fault if factor X is close to zero, AC fault if factor Y is close to zero, and AB fault if factor Z is close to zero. (2) If exactly two of the factors are close to zero, it is a line-to-ground fault. When there is a line-to-ground fault, it is AG fault if factors Y and Z are both close to zero, BG fault if factors X and Z are both close to zero, and CG fault if factors X and Y are both close to zero. For some faulty transmission line, X = X, Y = Y, and Z = Z. 

- **Possible Factual Question:** _{_ Is there a line-to-line/line-to-ground fault? / Is the fault type BC/AC/AB/AG/BG/CG? _}_ Be as concise as possible. 

- **Possible Interventional Questions:** If factor X/Y/Z had been/had not been close to zero, _{_ would there have been a line-to-line/line-to-ground fault? / would the fault have been type BC/AC/AB/AG/BG/CG _}_ ? Be as concise as possible. 

- **Causal Relationships:** 

_X ∼N_ ( _µ_ = _X, σ_[¯] = 0 _._ 1) 

(29) 

20 

|_Y _|_∼N_(_µ_= ¯_Y , σ_ = 0_._1)|(30)|
|---|---|---|
|_Z _|_∼N_(_µ_= ¯_Z, σ_ = 0_._1)|(31)|
|_X_0|= (_X <_0_._1)|(32)|
|_Y_0|= (_Y <_0_._1)|(33)|
|_Z_0|= (_Z <_0_._1)|(34)|
|_LL_|= (_X_0_∧¬Y_0_∧¬Z_0)_∨_(_¬X_0_∧Y_0_∧¬Z_0)_∨_(_¬X_0_∧¬Y_0_∧Z_0)|(35)|
|_LG_|= (_¬X_0_∧Y_0_∧Z_0)_∨_(_X_0_∧¬Y_0_∧Z_0)_∨_(_X_0_∧Y_0_∧¬Z_0)_∨_(_X_0_∧Y_0_∧Z_0)|(36)|
|_BC_|=_LL ∧X_0|(37)|
|_AC_|=_LL ∧Y_0|(38)|
|_AB_|=_LL ∧Z_0|(39)|
|_AG_|=_LG ∧Y_0_∧Z_0|(40)|
|_BG_|=_LG ∧X_0_∧Z_0|(41)|
|_CG_|=_LG ∧X_0_∧Y_0|(42)|



where _X_[¯] , _Y_[¯] , and _C_[¯] are drawn randomly from the values reported in Reddy et al. (2016). 

- **Causal Structure and In-domain Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0021-03.png)


- **Causal Structure and Common-Cause Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0021-05.png)


- **Causal Structure and Common-Effect Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0021-07.png)


- **Causal Structure and Induction Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0021-09.png)


21 

## **C.4 Real-World Reasoning Problem: Math Benchmarking** 

In Section 6.3 we introduced three real-world problems to validate our experimental findings from the proof-of-concept puzzle reasoning problem. Here, we offer a more detailed explanation of one of these real-world reasoning problems—the **Math Benchmarking** problem. 

- **Context:** Carla is downloading a _{N_ size _}_ GB file. Normally she can download 2 GB/minute, but in 100 minutes, Windows will force a restart to install updates, which takes _{N_ minutes _}_ minutes. After the restart, Carla can resume her download. 

- **Possible Factual Question:** _{_ Will Windows force a restart before the download is complete? / Will the download take longer than 120 minutes? _}_ Be as concise as possible. 

- **Possible Interventional Questions:** If _{_ she were downloading a file twice the size / Windows had forced a restart before the download was complete / Windows had not forced a restart before the download was complete _}_ , would _{_ Windows have forced a restart before the download was complete? / the download have taken longer than 120 minutes? _}_ Be as concise as possible. 

- **Causal Relationships:** 

|**ps:**||
|---|---|
|_Nsize ∼U_(50_,_300)|(43)|
|_Nminutes ∼U_(10_,_30)|(44)|
|_S ∼B_(_p_= 0_._5)|(45)|
|_Ndownload_<br>~~_t_~~_ime_ = [_Nsize ∗_2_∗S_+_Nsize_(1_−S_)]_/_2|(46)|
|_R_= (_Ndownload_<br>_time ≥_100)|(47)|
|_T_ = (_Ndownload_<br>~~_t_~~_ime_ +_R ∗Nminutes_)_≥_120)|(48)|



- **Causal Structure and Induction Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0022-08.png)


- **Causal Structure and Deduction Cause-Based Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0022-10.png)


- **Causal Structure and Deduction Effect-Based Fine-tune/Evaluate Relations:** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback_images/iclr_2025_Reasoning_Elicitation_in_Language_Models_via_Counterfactual_Feedback.pdf-0022-12.png)


22 

