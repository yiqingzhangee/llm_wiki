Preprint. 

# BIRD: A TRUSTWORTHY BAYESIAN INFERENCE FRAMEWORK FOR LARGE LANGUAGE MODELS 

**Yu Feng**[1] _[∗]_ **Ben Zhou**[2] **Weidong Lin**[1] **Dan Roth**[1] 1University of Pennsylvania 2Arizona State University 

## ABSTRACT 

Predictive models often need to work with incomplete information in real-world tasks. Consequently, they must provide reliable probability or confidence estimation, especially in large-scale decision making and planning tasks. Current large language models (LLM) are insufficient for such accurate estimations, but they can generate relevant factors that may affect the probabilities, produce coarse-grained probabilities when the information is more complete, and help determine which factors are relevant to specific downstream contexts. In this paper, we make use of these capabilities of LLMs to provide a significantly more accurate probabilistic estimation. We propose BIRD, a novel probabilistic inference framework that aligns a Bayesian network with LLM abductions and then estimates more accurate probabilities in a deduction step. We show BIRD provides reliable probability estimations that are 30% better than those provided directly by LLM baselines. These estimates can further contribute to better and more trustworthy decision-making. 

## 1 INTRODUCTION 

Accurate, interpretable, and unbiased probabilistic estimation is crucial to artificial intelligence (AI) systems in mission-critical tasks that require trustworthy decision-making. This is because, in many real-world applications, AI systems have to work with partial and incomplete observations that are insufficient for deductive and deterministic decisions (McCarthy & Hayes, 1981). Consider the charging station planning scenario in Fig. 1, where we want to use large language models (LLM) to decide where to build a certain number of charging stations over a region, subject to some constraints. Because we cannot always observe all the necessary information regarding a potential location, we can only provide confidence or probability estimates on whether we should use the candidate location. If the number of planned stations is significant, we need to perform an optimization process such as ILP (Roth & Yih, 2005) that maximizes the estimated probabilities of each individual station. A simple ranking of the candidates would not suffice in this scenario because it may yield sub-optimal solutions, as Fig. 1 shows. However, current LLMs are not ideal for such probability estimation tasks since 1) LLMs tend to be inaccurate and overconfident when estimating numerical confidences and probabilities (Xiong et al., 2024) and 2) they cannot provide an out-of-the-box interpretable and controllable process on how their estimates are derived (Li et al., 2024). This prevents us from using LLMs’s decision-making capabilities in large-scale automatic decision-making tasks. Consequently, there is a need to find ways to infer better probabilities from these models. 

In this work, we propose BIRD ( **B** ayesian **I** nference f **r** om Abduction and **D** eduction). BIRD builds on three observations. The first observation is that although LLMs cannot produce convincing numerical probabilities directly, they can generate neutral and comprehensive real-world factors relevant to these probabilistic estimates. Moreover, we observe that they can make relatively correct coarse probability estimations when using these factors as contexts that provide more complete information. The last observation is that LLMs can decide how specific contexts and conditions are related to these factors via entailment classfications. Together, these observations motivate BIRD, where we approximate a Bayesian network (Friedman et al., 1997) using LLM-generated factors, optimize its internal conditional probabilities by sampling LLM coarse predictions on combinations of values of all the factors, and infer more accurate probabilities for specific downstream conditions using LLMs to decide how the conditions are mapped to the factors. 

> _∗_ Corresponding author: fengyu1@seas.upenn.edu 

1 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0002-01.png)


Figure 1: An example of mission-critical tasks. We first ask OpenAI o1 to separately predict the probability of building a charging station at each specific location. It estimates the same probabilities for two different conditions twice, while BIRD (ours) successfully distinguishes the difference and can thus help the user make a more informed decision. This further demonstrates that while LLMs are capable of coarse estimations, they struggle to generate accurate probabilities. We further show through OpenAI o1 Ranking that although it can successfully rank all four locations based on how likely it should be used to build a new charging station, it is still insufficient to complete the task because of ties (e.g., 1/4 and 2/3 are the same from ranking perspectives.) 

We formulate the probability estimation problem as deciding _P_ ( _Oi|C_ ), where _Oi_ is a potential outcome and _C_ is the downstream condition or context (e.g., the location is on a busy highway) under a decision-making scenario _S_ (e.g., deciding whether to build a charging station at the location). Given _S_ , BIRD first generates a comprehensive set of real-world factors and values _f_ 1 _, f_ 2 _, ..., fn_ that may contribute to the probabilistic estimations. Then, it approximates a Bayesian network structure using the generated factors and the outcomes _{O}_ . We propose a novel method for estimating the conditional probability tables (CPT) for the outcome, i.e., _P_ ( _Oi|f_ ) _, f ∈F_ where _F_ is the product space of value sets for all factors.[1] Specifically, we assume a set of learnable parameters _P_ ( _Oi|fj_ ) representing the conditional probability of each possible outcome, given a single variable _fj_ . We then sample _f ∈F_ and acquire LLM’s coarse probability estimations _PLLM_ ( _Oi|f_ ). Next, we approximate the conditional probability _Pestimated_ ( _Oi|f_ ) in the CPT using the learnable parameters _P_ ( _Oi|fj_ ) based on a derived approximation formula and optimize these parameters by minimizing the distributional distance between _Pestimated_ ( _Oi|f_ ) and the LLM’s coarse predicted probability _PLLM_ ( _Oi|f_ ). From a high level, this abduction step aligns language models’ internal probability estimation with Bayesian constraints and has better error tolerance. The resulting Bayesian network is generic to all possible conditions in the scenario _S_ . In the following deduction step, we use the trained Bayesian parameters to estimate the probability of any possible conditions and observations _P_ ( _Oi|C_ ). To do this, we employ an entailment process to determine _P_ ( _fi|C_ ), as shown by “LLM Entailment” in Fig. 2, and the outcome we are interested in, _P_ ( _Oi|C_ ) =[�] _f ∈F[P]_[(] _[O][i][|][f]_[)] _[P]_[(] _[f][|][C]_[)][.] 

As we show in §4, the inferred probabilities from BIRD are more reliable, consistent with human judgments on 30% more instances than those directly predicted by LLMs. We further show that BIRD probability predictions are good enough to be directly used for decision-making (i.e., by computing an argmax on the predicted probabilities of each label for inference), outperforming standard chainof-thought (CoT) (Wei et al., 2022) on several reasoning benchmarks. More importantly, BIRD achieves better probability estimations by optimizing an interpretable Bayesian network with naturallanguage-based variables. There are two additional benefits of BIRD. First, we show that BIRD aligned probabilities can serve as more accurate and fine-grained supervision signals for training smaller models. In addition, we demonstrate that BIRD’s factor generation process is neutral and unbiased to the specific conditions, so it can be used to generate better follow-up questions to improve decision making confidence. Combining these findings, we show that BIRD improves trustworthiness: a crucial ability for mission-critical applications such as medical chatbots, where we need reliable decisions supported with enough evidence for high-confidence predictions. 

> 1 _F_ is the set of all possible combinations of different assignments of values to all factors (e.g., selecting a value from each of the factors a/b/c/d/e/f in Fig. 2). Refer to §3. 

2 

Preprint. 

## 2 RELATED WORK 

**Direct Inference in LLMs.** Our paper is first related to LLM direct inference methods (Wei et al., 2022; Wang et al., 2023), and decomposition-based reasoning (Wolfson et al., 2020; Tafjord et al., 2022; Zhou et al., 2022; Kassner et al., 2023). Compared to these works, BIRD considers using abduction and deduction to facilitate probabilistic induction, which aligns with the slow-thinking systems (McGlynn, 2014; Lin et al., 2023). Our work is also related to chain-of-thought via latentvariable inference. Given a question-answer pair ( _X, Y_ ), existing methods (Hoffman et al., 2023; Hu et al., 2024) aim to find latent chains of thought – token sequences _Z_ that contribute the most to the conditional likelihood. While BIRD has a more structured _Z_ and focuses more on the proper probability estimation instead of merely finding the optimal _Z_ . 

**Uncertainty Estimations and Decision making under Uncertainty in LLMs.** Most current works focus more on sampling to access the model’s uncertainty of factual knowledge (Kuhn et al., 2023; Yadkori et al., 2024; Lin et al., 2024). Some works verbalize the uncertainty of LLMs (Tian et al., 2023; Xiong et al., 2024). However, Xiong et al. (2024) shows that LLMs are often overconfident when directly verbalizing their confidence. Hou et al. (2024) is most related to us, as it also incorporates the intuition of decomposition, but it focuses more on addressing aleatoric uncertainty due to input ambiguity without rigorous probability calculation. This work is also related to decision making with uncertainty (Ren et al., 2023; Han et al., 2024; Liu et al., 2024) while all methods use LLM’s direct uncertainty estimation such as token logits or verbalization. BIRD introduces external reliable Bayesian modeling to more accurately calibrate uncertainty in LLMs for decision making. 

**Probabilistic Inference with LMs.** Chen et al. (2020) is one of the first papers to estimate a probability out of an NLI problem while with no explanation. Ozturkler et al. (2023) follows a similar intuition to ours, where the model first thinks by retrieving relevant associations, and then conducts probabilistic reasoning. Lew et al. (2020); Wong et al. (2023) discuss translating from natural language to the probabilistic language of thought. However, none of the papers tackle the real-world complex settings under incomplete information as comprehensively as we do with the level of interpretability to users of any level. 

## 3 MODELING 

## 3.1 BAYESIAN INFERENCE FROM ABDUCTION AND DEDUCTION 

On a high level, given a decision _Y_ and partial information _X_ , instead of direct inference using LLMs (induction : _X → Y_ ), BIRD conceptualizes the general scenario behind _X_ into intermediate factors (abduction : _X → Z_ ), and then fits a Bayesian model with _Z_ . The model then estimates probabilities for _Y_ (deduction : _X, Z → Y_ ). Our problem setting is a generic contextual inference problem of the type inspired by Feng et al. (2023). Given the context _C_ , which includes a general scenario _S_ and a corresponding downstream condition _U_ , our task is to estimate P( _Oi|C_ ) _, i_ = 1 _,_ 2 _, C_ = ( _S, U_ ), where _{O}_ is denoted as _O_ 1 and _O_ 2. We decompose the overall context _C_ into the general scenario _S_ and downstream conditions _U_ because we want to build a neutral intermediate space _Z_ based on only _S_ , so that the probability estimation is unbiased towards any specific conditions. This would mitigate models relying on incorrect signals, such as _charging stations are normally built in higher-income neighborhoods_ , but more generic factors like _accessibility_ for trustworthy decision making. 

Instead of directly making a prediction, we approximate a controllable and interpretable inference process based on explicit Bayesian modeling (Gelman et al., 1995) containing natural-language-based variables. For a scenario _S_ , we conceptualize _N_ factors _{Fj}[N] j_ =1[based on] _[ S]_[that are relevant to] the outcome _O_ , which can take on two possible values, _Oi, i_ = 1 _,_ 2, similar to the structure found in Bayesian networks (Koller & Friedman, 2009). Here the real-world factors and the outcome are equivalent to the variables in the network. Therefore, the choice between outcomes _O_ 1 and _O_ 2 depends on the factors, with each discrete factor _Fj_ having a set of possible values as _Fj_ , containing all possible details of the factor required for deciding between _O_ 1 and _O_ 2. We use _F_ =[�] _[N] j_ =1 _[F][j]_[to] denote the product space of value sets for all factors. For an instance in the space, each factor _Fj_ should be attributed a specific value _fj_ . We assume such derived factors are complete at this stage. That is, they include all the possible individual factors/angles that may affect the distribution of the 

3 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0004-01.png)


Figure 2: Overview of BIRD. Given a scenario _S_ , LLMs generate the factors _F_ (a/b/...) with potential values ( _f_ 1= _{_ a.1,a.2 _}_ ). _F_ is the set of all value combinations of _F_ (e.g., a.1/b.2/c.1/...). BIRD approximates a Bayesian network parameterized by P( _Oi|fj_ ), and optimizes by sampling LLM coarse predictions on PLLM( _Oi|f_ ) _, f ∈ F_ , and minimizes the distributional distance between an approximated Pestimated( _Oi|f_ ) and PLLM( _Oi|f_ ). At inference time, each context _C_ (S+U1/U2/...) is mapped to some _fj_ via entailment, and a probability is derived using Pestimated. BIRD can further be used to generate follow-up questions. 

outcomes.[2] As a result, _F_ is a complete information space for decisions. As in Fig. 2, (factor a value 1 _a_ 1, factor b value 1 _b_ 1, factor c value 1 _c_ 1, factor d value 1 _d_ 1, factor e value 1 _e_ 1, factor f value 1 _f_ 1) is an element _f_ in the product space _F_ where there are 2[6] elements in total. We then calculate the conditional probability table (CPT) as _P_ ( _Oi|f_ ) _, ∀f ∈F, f_ = ( _f_ 1 _, f_ 2 _, ..., fN_ ). 

We assume 1) _{Fj}[N] j_ =1[is complete, so] _[ O]_[does not depend on any other intermediate factors, and] 2) only one value can be assigned for each factor for a given _C_ . The predictive probability for the outcome is thus obtained by marginalizing over the product space _F_ for a context _C_ (details in §A.1): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0004-05.png)


where _i_ = 1 _,_ 2 _, C_ = ( _S, U_ ), _∀f ∈F, f_ = ( _f_ 1 _, f_ 2 _, ..., fN_ ) _, fj ∈Fj, j_ = 1 _, ..., N_ . _F_ is the product space of value sets for all factors. _f_ is a possible instance of complete information in the space. 

**Assumption on Conditional Independence of Variables.** To provide relaxation for real-world applications, and since the factors should cover as many different aspects as possible, we assume factors are conditionally independent given the context. Therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0004-08.png)


In the following sections, we describe the missing implementation details of the framework: (1) the generation of the factors _{Fj}[N] j_ =1[(§][3.2][);][(2) the assignment of the conditional probability tables] P( _Oi|f_ ) (§3.3), and (3) the computation of observations P( _fj|C_ ) _, j_ = 1 _, ..., N_ (§3.4). 

## 3.2 ABDUCTIVE FACTOR GENERATION 

Large language models, although they cannot solve some complicated cases, are shown to possess parametric knowledge that enables them to solve more common cases that can be directly found 

> 2Note that our framework can later accommodate any additional factors by selectively recalculating specific parts as needed. 

4 

Preprint. 

in their memory (Kambhampati et al., 2024; Li et al., 2024; Zhou et al., 2024). Our method for generating the factors follows this similar intuition. For a scenario _S_ , we first do abstractions with LLMs to derive _N_ factors _{Fj}[N] j_ =1[.][The most straightforward approach involves utilizing LLMs to] directly obtain a list of factors and their potential values. However, direct generation is effective but not sufficiently comprehensive, and we want to fully utilize past similar occurrences that the model memorizes in their parametric knowledge. 

Therefore, to encourage models to fully explore the range of alternatives, we divide the task into two stages: 1) generating sentences describing familiar situations that would increase the likelihood of an outcome, and 2) summarizing these sentences into factors with corresponding values. We show the prompts in Appendix Fig. 5 and Fig. 6. We further utilize LLMs to perform classification to assess how the value of each generated factor affect the decision, so that we can filter out factors that are unrelated to the outcomes. The prompt is listed in Appendix Fig. 7. In the example depicted in Fig. 2, a factor like “The color of the vehicle” will be eliminated because all potential values of this factor consistently support the same outcome, i.e., neutral. 

## 3.3 ASSIGNMENT OF P( _Oi|f_ ) AS A CONSTRAINED OPTIMIZATION PROBLEM 

We now calculate each P( _Oi|f_ ) _, i_ = 1 _,_ 2 _, ∀f ∈F_ in the conditional probability table of the Bayesian network. Since we do not have any observed data for approximation and LLMs alone cannot directly give an accurate assignment on P( _Oi|f_ ), we adopt the formula proposed in Bordley (1982), which derived a version of the logarithmic opinion pool using axioms from the theory of additive conjoint measurement. This formula is proved to be the only continuous (over the open interval (0, 1)) group probability formula under the assumption detailed in §A.2 that the decision maker has an intuitive weak ordering on the set of factor odds ratios and a “noninteraction” property of factors. Our setting fits the assumption as the weak ordering of decisions is the most fundamental condition of consistency in decision making and we assume the conditional independence of factors. Therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0005-05.png)


To determine the weights _wj_ ,[3] , we adopt the second case scenario mentioned in Bordley (1982), i.e.,[�] _[N] j_ =1 _[w][j][>]_[1][.][In][this][case,][the][decision][maker][with][complete][information][is][more][certain] than the decision maker with only one piece of information through acquiring all different pieces of information from different factors. Since there isn’t a priori information that suggests another choice exists, we use equal weight _wj_ = 1 _, j_ = 1 _, ..., N_ and assume P( _Oi_ ) = 50% _, i_ = 1 _,_ 2. The final estimation formula can be rewritten as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0005-07.png)


**Constrained optimization for estimating** P( _Oi|f_ ) **.** From Eq. 4, it is not hard to see that each 

P( _Oi|f_ ) _, ∀f ∈F_ can now be approximated using only P( _Oi|fj_ ) _, ∀fj ∈Fj, j_ = 1 _, ..., N_ , which we view as learnable parameters. As discussed in §1, LLMs can perform relatively correct coarse estimation under complete information, which refers to contexts with all factors’ values present (i.e. _f ∈F_ ). Consequently, we prompt the LLMs to output verbalized probability estimations for outcome _Oi_ given _f_ . This approximates LLM’s predictions of P _LLM_ ( _Oi|f_ ). With Pestimated and P _LLM_ , we can now form a constraint optimization problem that finds the values of parameters P( _Oi|fj_ ) that minimize the distributional distance between the Bayesian estimated probabilities Pestimated( _Oi|f_ ) and LLM estimated probabilities P _LLM_ ( _Oi|f_ ) _, ∀f ∈F_ , while conforming to the structure in Eq. 4 and preliminary assessments about each P( _Oi|fj_ ) _, ∀fj ∈Fj, j_ = 1 _, ..., N_ provided by the LLM classification as mentioned in §3.2. 

**Learning algorithm for constrained optimization to estimate** P( _Oi|f_ ) **.** We propose a gradientdescent algorithm for this optimization problem. The detailed algorithm can be found in §A.3, and we give a general description here. As the parameter initialization, we utilize the LLM classification 

3Weights here refer to a set of “hyperparameters” that decides how highly the final decision regards each factor and how correlated the factors are considered, not the learnable parameters in our Bayesian network. 

5 

Preprint. 

as mentioned in §3.2, where we direct the LLMs to ascertain which outcome is more likely supported by the value of each factor: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0006-02.png)


_′ ′ ′ ′_ = For the training data, given an instance of complete information _f_ ( _f_ 1 _[, f]_ 2 _[, ..., f] N_[)][, we prompt] LLM to output a verbalized probability score for each outcome _∈{_ very unlikely, unlikely, somewhat unlikely, neutral, somewhat likely, likely, very likely _}_ which we use as P _LLM_ ( _Oi|f ′_ ). The detailed prompt can be found in Appendix Fig. 10. We then assign a specific probability value for each coarse estimation, i.e., very unlikely: 0%, unlikely: 20%, somewhat unlikely: 40%, neutral: 50%, somewhat likely: 60%, likely: 80%, very likely: 100%, as supervision signals to learn the correct value of P( _Oi|fj′_[)] _[, j]_[=][1] _[, ..., N]_[for each] _[ f] j′_[in] _[ f] ′_ using Eq.4. We randomly sample 128 instances in the space of complete information as the training data for each scenario. Note that we manually assign probability values for LLM coarse decisions, but they can be changed as hyperparameters. 

We adopt two loss functions: 1) we use the regular mean squared error loss for optimization and denote the loss as _ℓ_ MSE(Pestimated( _Oi|f ′_ ) _,_ PLLM( _Oi|f ′_ )); 2) we use the margin ranking loss to ensure the final trained individual probability preserve the original direction of its supported outcome and denote the loss as _ℓMR_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0006-05.png)


_∀fj ∈Fj, j_ = 1 _, ..., N_ . We apply probability marginalization to get the trained individual probability and approximate it with the mean values since there are no explicit priors. We use _F[∗]_ to denote the product space of the remaining factors except _Fj_ , which is a subspace of _F_ and _f[∗]_ = ( _f_ 1 _, f_ 2 _, ..., fj−_ 1 _, fj_ +1 _, ..., fN_ ) _, f[∗] ∈F[∗]_ . Therefore, the final loss function we use for training is defined as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0006-07.png)


where _α_ balances the two losses. We use SGD as the optimization function. As a result, we can estimate any P( _Oi|f_ ) _, ∀f ∈F_ with the learned P( _Oi|fj_ ) _, ∀fj ∈Fj_ with Eq.4. 

**Customization to human preference.** A benefit of our design is that we can explicitly ask for human preferences on how a particular factor value will affect the outcome, i.e., manually setting P( _Oi|fj_ ). Such preferences are desirable in certain applications with known constraints, and they can be easily integrated in Eq.4 by replacing P( _Oi|fj_ ) with the human preferred value. 

## 3.4 COMPUTATION OF P( _fj|C_ ) WITH LLM ENTAILMENT 

This section discusses how we compute P( _fj|C_ ). As noted in §3.1, _C_ is the context containing the general scenario _S_ , and a specific additional condition _U_ . Given _C_ , we employ an entailment task formulation to find which factors and their corresponding values are implied or mapped by _C_ . Specifically, we adopt the prompt in Appendix Fig. 8 where we directly ask if the context entails a value from a factor. We conduct another round of self-correction using the prompt in Appendix Fig. 9 to check if the model believes all the previously identified values are actually entailed. 

We define the notion of observed factors, which refers to a sub-set of factors among all factors that are being implied by the context. For example, as shown in Fig. 2, context _S_ + _U_ 1 implies _a_ 1, _c_ 1, and _d_ 1, _S_ + _U_ 2 implies _c_ 1 and _d_ 1, while the additional condition _S_ + _U_ 3 implies _d_ 1 and _f_ 2. We assume only one value can be chosen per factor. We write the product value space of the implied factors as _F[′]_ =[�] _[M] m_ =1 _[F][j] m_[, which is a subspace of] _[ F]_[.][We define the conditional probabilities of the values of] these observed factors to be 1 if the value is implied by _C_ , a 0 if not: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0006-13.png)


6 

Preprint. 

Where the implied _fj[∗] m[∈F][j] m[,][ ∀][m]_[ = 1] _[, ..., M]_[.][For the unobserved factors, we have no information] regarding which value to be selected. In order to be neutral and unbiased, we assume that, in an unobserved factor, each value has an equal probability of being selected. We use _F[′⊥]_ =[�] _[N] m_ = _M_ +1 _[F][j] m_ to denote the space of unobserved factors, which is the orthogonal complement space of _F[′]_ in _F_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0007-02.png)


Where _∀m_ = _M_ + 1 _, ..., N_ . We can therefore calculate the final probability as presented in Eq.1, utilizing the formulations provided in Eqs.2 through 9. 

## 4 EXPERIMENTS 

## 4.1 DATASETS AND SETTINGS 

We convert three datasets focusing on reasoning and planning to our problem setting including _O_ , _S_ , and _U_ , as described in §3.1. We detail this process in §A.4. **COMMON2SENSE** (Singh et al., 2021) focuses on commonsense reasoning, and we gather 213 scenarios and 3822 instances, where 9 additional conditions on average support each outcome. **PLASMA** (Brahman et al., 2023) focuses on planning, and we gather 279 scenarios and 1395 instances, where 5 additional conditions support the less common outcome. **TODAY** (Feng et al., 2023) focuses on temporal reasoning, where there are 1000 instances. Among the three datasets, the most challenging task is TODAY and the easiest is COMMON2SENSE, judged by baseline model performances. The complexity of the scenario and conditions is highest in TODAY. The conditions support the less common outcome in PLASMA and COMMON2SENSE presents general decision making with common sense. 

We conduct our experiments with Llama-2-70b-instruct and Llama-3.1-70b-Instruct.[4] We set the temperature = 0.7. For the factor generation task in §3.2, we generate 10 sentences for each outcome before summarization. For the LLM classification task for factor-outcome mapping, we generate 3 times and conduct a majority vote. For the LLM entailment task in §3.4, we sample 3 times and conduct a majority vote with 1 round of self-reflection. We set learning rate = 1e-02, epoch = 20, batch size = 4, _ϵ_ = 0, _α_ = 10 for the learning algorithm mentioned in §3.3. 

## 4.2 INTRINSIC EVALUATIONS OF THE BIRD FRAMEWORK 

**Reliability of BIRD’s Estimated Probabilities.** We first propose an evaluation method that directly benchmarks the quality of probability estimations. We adopt a similar setting as mentioned in §3.1. Given a scenario _S_ and two potential outcomes _O_ 1 and _O_ 2, we find two specific additional conditions _U_ 1 and _U_ 2, both supporting _O_ 1 over _O_ 2.[5] _S_ + _U_ 1 and _S_ + _U_ 2 will form two contexts _C_ 1 and _C_ 2, respectively. The goal of this task is to evaluate if models can assign fine-grained probability estimations, that is, sufficient to distinguish which context between _C_ 1 and _C_ 2 _**better**_ supports _O_ 1 over _O_ 2 or if both _C_ 1 and _C_ 2 _**equally**_ support _O_ 1 over _O_ 2, by predicting probability values of _P_ ( _Oi|C_ 1) for _C_ 1 and _P_ ( _Oi|C_ 2) for _C_ 2 _**individually**_ .[6] That is, assuming _C_ 1 better support _O_ 1 over _O_ 2, we expect _P_ ( _O_ 1 _|C_ 1) _> P_ ( _O_ 1 _|C_ 2) _> P_ ( _O_ 2 _|C_ 2) _> P_ ( _O_ 2 _|C_ 1), and vice versa. If _C_ 1 and _C_ 2 equally support _O_ 1 over _O_ 2, we should see _P_ ( _O_ 1 _|C_ 1) = _P_ ( _O_ 1 _|C_ 2) _> P_ ( _O_ 2 _|C_ 1) = _P_ ( _O_ 2 _|C_ 2). This task is significantly more difficult than providing two contexts that support different outcomes, as systems will have to understand the nuances in probability estimation. 

We collect human gold labels for which context better supports the gold outcome with rigorous quality control using MTurk. We explain our choice of this relative evaluation over asking human annotators to provide absolute probabilities from a psychological perspective, along with the details of human annotation in §A.5. We evaluate on 350 randomly sampled instances from COMMON2SENSE when the conditions can be mapped to at least one factor and therefore BIRD does not produce “unknown”[7] and all methods to be tested below have accurate binary decisions on all instances, i.e., 

> 4https://huggingface.co/upstage/Llama-2-70b-instruct and https:// huggingface.co/meta-llama/Llama-3.1-70B-Instruct 

> 5Here _O_ 1 will be the gold outcome. During evaluation, the two conditions can also support _O_ 2 over _O_ 1. 

> 6For proper evaluation, systems predict probabilities one at a time, without seeing other contexts. 

> 7Although the setting may seem to favor our proposed method, it is a fair comparison, because “unknown” predictions are label-agnostic, and we can use the same direct inference approach, such as CoT, for such cases. 

7 

Preprint. 

|Model<br>Random Guessing|Different1<br>0.333|Different2<br>0.333|Same<br>0.333|Average|
|---|---|---|---|---|
|||||0.333|
|GPT3.5 CoT<br>GPT4 CoT<br>Llama2-70b Instruct Logit<br>Llama2-70b Instruct Vanilla<br>Llama2-70b Instruct CoT<br>Llama3.1-70b Instruct Logit<br>Llama3.1-70b Instruct Vanilla<br>Llama3.1-70b Instruct COT|0.306<br>0.312<br>0.263<br>0.375<br>0.315<br>0.300<br>0.365<br>0.426|0.306<br>0.357<br>0.228<br>0.333<br>0.323<br>0.282<br>0.301<br>0.370|0.242<br>0.216<br>0.205<br>0.243<br>0.254<br>0.242<br>0.251<br>0.227|0.283<br>0.289<br>0.228<br>0.311<br>0.294<br>0.269<br>0.303<br>0.346|
|Llama2-70b Instruct EC*<br>Llama3.1-70b Instruct EC*<br>GPT4 EC*|0.530<br>0.535<br>0.588|0.529<br>0.538<br>0.533|0.207<br>0.286<br>0.3|0.503<br>0.511<br>0.54|
|Llama3.1 BIRD(_ablation w 1/2 assumption_)<br>Llama3.1 BIRD(_ablation w 1/n assumption_)<br>Llama3.1 BIRD(_ablation w fxed initial prob_)|0.527<br>0.572<br>0.616|0.532<br>0.584<br>0.597|0.196<br>0.272<br>0.337|0.480<br>0.532<br>0.568|
|Llama2 BIRD(_ours w optimized prob_)<br>Llama3.1 BIRD(_ours w optimized prob_)|**0.614**<br>0.612|0.624<br>**0.625**|**0.450**<br>0.382|**0.592**<br>0.588|



Table 1: Preference-based pairwise evaluation using F1 score for predicted probabilities across various model settings. Different1 indicates the F1 score for a category where _C_ 1 supports the gold outcome better than _C_ 2, and vice versa for Different2. Same refers to the F1 score of the category where _C_ 1 and _C_ 2 equally support the outcome. Average refers to calculating F1 metrics globally by counting the total true positives, false negatives, and false positives. EC* settings see both _C_ 1 and _C_ 2 at the same time instead of predicting individual probabilities, hence not directly comparable. 

both conditions support the gold outcome over the opposite outcome. We report the F1 score as the evaluation metric. We further explain how the F1 score is computed in §A.5. 

We compare the following methods that directly acquire probability estimations from LLMs as baselines. 1) **Vanilla** (Wang et al., 2023): models directly verbalize the estimated probabilities. 2) **Logit** : we select the normalized token probability of the decision. 3) **CoT** (Wei et al., 2022): models output a reasoning process followed by verbalized estimated probabilities. 4) **EC** : models select which additional condition better supports the outcome without outputting any probabilities. Note that this formulation is unfair to BIRD and other baselines that individually estimate the probabilities. To demonstrate the effectiveness of our learning algorithm in §3.3, we propose two ablation baselines that use other methods to find the conditional probabilities in the Bayesian network: 1) **1/2 and 1/N Assumptions** : We adopt two neutral assumptions without LLM classifications for CPT calculation. 1/2: P( _Oi|f_ ) is 1 only if all factors’ values, except for any neutral ones, support _Oi_ ; if no, 0. 1/N: P( _Oi|f_ ) is calculated as the percentage of factors whose values support outcome _Oi_ . 2) **Fixed Initial Prob** : We utilize the initial probability of Eq.5 to directly estimate P( _Oi|f_ ) with Eq.4. The details of the baselines are presented also in §A.6. 

Table 1 lists the results. Our framework exhibits superior alignment with human preference, with both models achieving an F1 of 59%, more than 30% higher than GPT-4. Such results demonstrate that BIRD produces reliable probability estimation without additional human supervision. Notably, such probability estimation is done by judging each condition separately (i.e., BIRD assigns one probability to condition 1, and later another probability to condition 2). We find that vanilla language models do much better when both conditions are provided at the same time (i.e., the EC setting) where the model can perform argmax operations. However, even under this setting that is unfair to our framework, BIRD still outperforms GPT-4 by 5%. All results suggest that vanilla LLMs cannot properly estimate probabilities, and BIRD effectively improves them. In addition, by outperforming all the ablation settings, we demonstrate that our constrained optimization algorithm proposed in §3.3 effectively approximates better Bayesian network’s conditional probabilities. 

**Applying BIRD’s Probabilities in Decision-Making.** In this section, we evaluate the performance of applying the predicted probabilities from BIRD in decision-making tasks. In the general decisionmaking setting, we similarly assume a context _C_ = _S_ + _U_ , two possible outcomes _O_ 1 and _O_ 2, 

8 

Preprint. 

|Dataset|BIRD<br>~~1~~/2|BIRD<br>~~1~~/n|BIRD|~~f~~xed|~~p~~rob|BIRD|~~o~~ptimized|(ours)|CoT|
|---|---|---|---|---|---|---|---|---|---|
|||_Llama-2-70b-instruct_||||||||
|TODAY|73.7|72.8||73.4|||**73.9**||71.5|
|PLASMA|72.8|72.3||72.7|||74.0||**76.8**|
|COMMON2SENSE|86.9|86.8||87.5|||89.0||**93.8**|
|||_Llama-3.1-70b-Instruct_||||||||
|TODAY|65.5|68.9||65.5|||**74.3**||72.6|
|PLASMA|71.3|66.5||65.7|||**73.0**||71.5|
|COMMON2SENSE|78.1|85.4||86.7|||**92.3**||90.8|



Table 2: Performance comparisons of BIRD and baselines on decision-making benchmarks. BIRD ~~o~~ ptimized is our final model; chain-of-thought with self-consistency (CoT) is our main baseline; others are ablation baselines. 

and the task is to decide which outcome is more likely given _C_ , similar to Fig. 2. The inference process we use for BIRD is simply taking the argmax between P( _O_ 1 _|C_ ) and P( _O_ 2 _|C_ ). We remove all instances where they cannot be mapped to any factor values(i.e., BIRD’s decision is “unknown”). This does not undermine our experiment setting because such removal is label-agnostic, and we can always fall back to other methods such as CoT when BIRD predicts “unknown”. We compare our framework with chain-of-thought inference with self-consistency on the same instances where BIRD predictions are not “unknown”. We present the prompt for COT in Appendix Fig.14. 

The results are shown in Table 2. Our first observation is that BIRD’s probability estimations are accurate enough for direct decision-making. Our proposed method outperforms the chain-of-thought baseline with Llama-3.1 on all benchmarks and performs comparably using Llama-2 as the base model. This is partially because Llama-3.1 generates more diverse and accurate factors, and we analyze this in §A.7. At the same time, we see that BIRD performs better on the most challenging TODAY dataset on both models. This demonstrates that BIRD is a desirable solution for more complex scenarios where CoT tends to fail more. 

We also observe the inferior performance of our method compared to CoT may be attributed to the fundamental differences between probability estimation and hard label classification. Table 6 in the appendix shows an example that CoT sometimes does correctly due to the spurious signals and biases in the conditions, which highlights the importance of our controllable inference approach. We can conclude that BIRD outperforms direct inference methods like CoT in most cases and is at least comparable in decision-making while offering significantly better interpretability and controllability throughout the process. 

**Ablation Studies on BIRD’s Components.** We conduct an ablation study regarding factor generation. We compare the final decision-making accuracy based on our proposed abductive sampling method and the direct factor generation method mentioned in §3.2. Appendix Table 7 shows the results: we see that our proposed method, on average, surpasses the direct method by 4.4%/3.9% in accuracy for llama2/llama3.1, indicating the effectiveness of our proposed factor generation that collects more comprehensive signals from models’ parametric knowledge and memorization. We also conduct ablation studies on different ways of estimating the conditional probabilities of our Bayesian network (i.e., 1/2, 1/n, and fixed ~~p~~ rob). Results in Tables 1 and 2 show that our proposed constrained optimization learning algorithm effectively finds more accurate estimations by outperforming all ablation baselines. 

## 4.3 EXTRINSIC APPLICATIONS OF THE BIRD FRAMEWORK 

**The Usage of the Reliably Estimated Probability.** As shown in §4.2, BIRD can produce reliable probabilities that contribute to better decision-making. Such findings motivate us to investigate whether these probabilities can be used as supervision signals, as models may benefit from seeing more than the discrete classification labels (Zhou et al., 2020). Specifically, we use BIRD to estimate probabilities of some decision-making training instances and finetune a T5-large model using these generated probabilities. We then evaluate its performance on cross-domain datasets. We fully replicate 

9 

Preprint. 

the experimental settings described in Feng et al. (2023). We use instances from COMMON2SENSE, where the argmax of the estimated probabilities from our framework using Llama2 supports the correct outcome, as additional training data, besides the originally proposed training data in that paper. We test on two general temporal reasoning datasets, MATRES (Ning et al., 2018) and TRACIE (Zhou et al., 2021), as well as two commonsense reasoning and planning datasets, TODAY, and PLASMA. The results are shown in Table 3. We see that models achieve an average performance increase of 1.3% compared with BIRD-generated probabilities as additional signals. We use cross-entropy loss with class probabilities as the target for our soft probability setting and cross-entropy loss with class labels as the target for the hard-label setting. Such results suggest another effective application of the BIRD framework, where smaller models can benefit from soft probability alignment and make better overall predictions. 

|Model (Train Data)<br>T5<br>~~l~~arge<br>+ hard label<br>+ BIRDprob (_ours_)|TODAY(exp)<br>77.8<br>83.6<br>**84.3**|TODAY<br>57.9<br>62.9<br>**63.4**|TRACIE<br>**73.0**<br>68.2<br>71.4|MATRES<br>**83.5**<br>75.2<br>77.6|PLASMA<br>48.1<br>**65.2**<br>63.3|Average|
|---|---|---|---|---|---|---|
|||||||65.6<br>67.9<br>**68.9**|
|PatternTime<br>+ hard label<br>+ BIRDprob (_ours_)|82.2<br>83.9<br>**85.1**|60.9<br>61.8<br>**62.6**|**79.8**<br>76.2<br>75.7|**85.8**<br>83.7<br>85.1|50.5<br>56.9<br>**61.5**|69.3<br>69.7<br>**71.2**|



Table 3: System performances under different supervision data across three binary temporal benchmarks and one binary planning benchmark. For simplicity, we use “hard label” representing that we use COMMON2SENSE supervision data with explicit binary labels, and “BIRD prob” representing that we use COMMON2SENSE supervision data with estimated probabilities. TODAY (exp) uses gold explanations during evaluation. 

**Using BIRD to Generate Better Follow-up Questions.** Generating accurate and informative follow-up questions is an important goal for LLMs to be applied in mission-critical tasks that require unambiguous decision-making. A natural benefit of our proposed BIRD framework is that it analyzes what needs to be known through the comprehensive and unbiased abduction step (i.e., factor generation) and interprets how new information will change the probability or confidence of a desired decision during the deduction step (i.e., probability generation). We conduct a follow-up question generation experiment to demonstrate this benefit. For a given decision-making problem, we ask both BIRD and LLMs to generate binary (yes/no) follow-up questions. For BIRD, we identify implied factors, choose the most important unmentioned one, adopt the most important value aligning with the outcome, and rephrase it as a question. We directly ask LLM to generate questions for better decision making. The details can be referred to in §A.9. We then recruit crowd-workers on MTurk to provide pairwise preference between the BIRD generated question and the question generated directly by the LLM. We do this on 250 examples from COMMON2SENSE. An example can be seen in Appendix Fig. 4 and the details can also be seen in §A.9. Appendix Table 8 shows that the follow-up questions generated by BIRD are more often preferred: questions generated by BIRD are preferred 52.8% of the time, while those directly generated by LLM are preferred 32.8% of the time; in the remaining 14.4%, both are considered equally useful. This result motivates future work on applying BIRD in an interactive agent system for more efficient and controlled decision making. 

## 5 CONCLUSION 

In this paper, we propose BIRD, a Bayesian inference framework for large language models. BIRD utilizes LLMs’ existing reliable capabilities such as abductive thinking and coarse-grained classification to optimize an explicit Bayesian network for more reliable probability generation. We show that BIRD can output accurate probabilities, outperforming LLM vanilla probabilties by 30% on direct estimation tasks, and outperform chain-of-thought on decision-making tasks. At the same time, BIRD’s outputs are explainable, because of its natural-language-based variables in the fully-interpretable Bayesian network. We also show that BIRD can be used in several other settings, such as generating fine-grained supervision signals and asking informative follow-up questions, to further improve the trustworthiness and controllability of LLMs when applied to real-world applications. 

10 

Preprint. 

## REFERENCES 

- Robert F Bordley. A multiplicative formula for aggregating probability assessments. _Management science_ , 28(10):1137–1148, 1982. 

- Faeze Brahman, Chandra Bhagavatula, Valentina Pyatkin, Jena D. Hwang, Xiang Lorraine Li, Hirona J. Arai, Soumya Sanyal, Keisuke Sakaguchi, Xiang Ren, and Yejin Choi. Plasma: Making small language models better procedural knowledge models for (counterfactual) planning, 2023. 

- Tongfei Chen, Zhengping Jiang, Adam Poliak, Keisuke Sakaguchi, and Benjamin Van Durme. Uncertain natural language inference. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 8772–8779, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.774. URL https://aclanthology.org/2020.acl-main. 774. 

- Yu Feng, Ben Zhou, Haoyu Wang, Helen Jin, and Dan Roth. Generic temporal reasoning with differential analysis and explanation. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 12013–12029, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.671. URL https: //aclanthology.org/2023.acl-long.671. 

- Nir Friedman, Dan Geiger, and Moises Goldszmidt. Bayesian network classifiers. _Machine learning_ , 29:131–163, 1997. 

- Andrew B. Gelman, John B. Carlin, Hal S. Stern, and Donald B. Rubin. _Bayesian Data Analysis_ . Chapman and Hall/CRC, Boca Ratan, Florida, 1995. 

- Jiuzhou Han, Wray Buntine, and Ehsan Shareghi. Towards uncertainty-aware language agent, 2024. 

- Matthew Douglas Hoffman, Du Phan, david dohan, Sholto Douglas, Tuan Anh Le, Aaron T Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, and Rif A. Saurous. Training chain-of-thought via latent-variable inference. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum?id=a147pIS2Co. 

- Bairu Hou, Yujian Liu, Kaizhi Qian, Jacob Andreas, Shiyu Chang, and Yang Zhang. Decomposing uncertainty for large language models through input clarification ensembling. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 19023–19042. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/hou24b.html. 

- Edward J Hu, Moksh Jain, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, and Nikolay Malkin. Amortizing intractable inference in large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview. net/forum?id=Ouj6p4ca60. 

- Daniel Kahneman. A perspective on judgment and choice: mapping bounded rationality. In _The American psychologist vol. 58,9 (2003): 697-720_ , 2003. doi: 10.1037/0003-066X.58.9.697. 

- Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant Bhambri, Lucas Paul Saldyt, and Anil B Murthy. Position: LLMs can’t plan, but can help planning in LLM-modulo frameworks. In _Forty-first International Conference on Machine Learning_ , 2024. URL https://openreview.net/forum?id=Th8JPEmH4z. 

- Nora Kassner, Oyvind Tafjord, Ashish Sabharwal, Kyle Richardson, Hinrich Schuetze, and Peter Clark. Language models with rationality. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 14190–14201, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.877. URL https://aclanthology.org/2023. emnlp-main.877. 

11 

Preprint. 

- Daphne Koller and Nir Friedman. _Probabilistic Graphical Models: Principles and Techniques - Adaptive Computation and Machine Learning_ . The MIT Press, 2009. ISBN 0262013193. 

- Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id= VD-AYtP0dve. 

- Alexander K Lew, Michael Henry Tessler, Vikash K Mansinghka, and Joshua B Tenenbaum. Leveraging unstructured statistical knowledge in a probabilistic language of thought. In _Proceedings of the annual conference of the cognitive science society_ , 2020. 

- Bangzheng Li, Ben Zhou, Fei Wang, Xingyu Fu, Dan Roth, and Muhao Chen. Deceptive semantic shortcuts on reasoning chains: How far can models go without hallucination? In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pp. 7675–7688, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.424. URL https://aclanthology.org/ 2024.naacl-long.424. 

- Bill Yuchen Lin, Yicheng Fu, Karina Yang, Prithviraj Ammanabrolu, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. _ArXiv_ , abs/2305.17390, 2023. 

- Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty quantification for black-box large language models, 2024. URL https://arxiv.org/abs/2305. 19187. 

- Ollie Liu, Deqing Fu, Dani Yogatama, and Willie Neiswanger. Dellma: Decision making under uncertainty with large language models, 2024. URL https://arxiv.org/abs/2402.02392. 

- John McCarthy and Patrick J Hayes. Some philosophical problems from the standpoint of artificial intelligence. In _Readings in artificial intelligence_ , pp. 431–450. Elsevier, 1981. 

- N. F. McGlynn. Thinking fast and slow. _Australian veterinary journal_ , 92 12:N21, 2014. 

- Qiang Ning, Hao Wu, and Dan Roth. A multi-axis annotation scheme for event temporal relations. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 1318–1328, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1122. URL https://aclanthology.org/P18-1122. 

- Batu Ozturkler, Nikolay Malkin, Zhen Wang, and Nebojsa Jojic. ThinkSum: Probabilistic reasoning over sets using large language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.68. URL https://aclanthology.org/ 2023.acl-long.68. 

- Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, and Anirudha Majumdar. Robots that ask for help: Uncertainty alignment for large language model planners. In _7th Annual Conference on Robot Learning_ , 2023. URL https://openreview.net/ forum?id=4ZK8ODNyFXx. 

- Dan Roth and Wen-tau Yih. Integer linear programming inference for conditional random fields. In _Proceedings of the 22nd international conference on Machine learning_ , pp. 736–743, 2005. 

- Shikhar Singh, Nuan Wen, Yu Hou, Pegah Alipoormolabashi, Te-lin Wu, Xuezhe Ma, and Nanyun Peng. COM2SENSE: A commonsense reasoning benchmark with complementary sentences. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pp. 883– 898, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. findings-acl.78. URL https://aclanthology.org/2021.findings-acl.78. 

12 

Preprint. 

- Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter Clark. Entailer: Answering questions with faithful and truthful chains of reasoning. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 2078–2093, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.134. URL https://aclanthology.org/ 2022.emnlp-main.134. 

- Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 5433–5442, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.330. URL https://aclanthology.org/ 2023.emnlp-main.330. 

- AMOS TVERSKY and DANIEL KAHNEMAN. Judgment under uncertainty: Heuristics and biases. In _Science, 185(4157), 1124–1131_ , 1974. URL https://doi.org/10.1126/science. 185.4157.1124. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum?id=1PL1NIMMrw. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum? id=_VjQlMeSB_J. 

- Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. Break it down: A question understanding benchmark. _Transactions of the Association for Computational Linguistics_ , 8:183–198, 2020. doi: 10.1162/tacl ~~a 0~~ 0309. URL https: //aclanthology.org/2020.tacl-1.13. 

- Lionel Wong, Gabriel Grand, Alexander K. Lew, Noah D. Goodman, Vikash K. Mansinghka, Jacob Andreas, and Joshua B. Tenenbaum. From word models to world models: Translating from natural language to the probabilistic language of thought, 2023. URL https://arxiv.org/abs/ 2306.12672. 

- Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https: //openreview.net/forum?id=gjeQKFxFpZ. 

- Yasin Abbasi Yadkori, Ilja Kuzborskij, Andras Gy´ orgy, and Csaba Szepesv¨ ari.´ To believe or not to believe your llm, 2024. URL https://arxiv.org/abs/2406.02543. 

- Ben Zhou, Qiang Ning, Daniel Khashabi, and Dan Roth. Temporal common sense acquisition with minimal supervision. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 7579–7589, 2020. 

- Ben Zhou, Kyle Richardson, Qiang Ning, Tushar Khot, Ashish Sabharwal, and Dan Roth. Temporal reasoning on implicit events from distant supervision. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 1361–1371, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.107. URL https://aclanthology.org/2021. naacl-main.107. 

- Ben Zhou, Kyle Richardson, Xiaodong Yu, and Dan Roth. Learning to decompose: Hypothetical question decomposition based on comparable texts. In Yoav Goldberg, Zornitsa Kozareva, 

13 

Preprint. 

and Yue Zhang (eds.), _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pp. 2223–2235, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.142. URL https://aclanthology.org/2022.emnlp-main.142. 

Ben Zhou, Hongming Zhang, Sihao Chen, Dian Yu, Hongwei Wang, Baolin Peng, Dan Roth, and Dong Yu. Conceptual and unbiased reasoning in language models. _arXiv preprint arXiv:2404.00205_ , 2024. 

## A APPENDIX 

A.1 DERIVATION OF THE EQUATION FOR UNCERTAINTY ESTIMATION 

We derive the outcome probability for the given context _C_ of partial information, i.e., any additional condition _U_ under the scenario _S_ , based on the complete information space _F_ of scenario _S_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0014-06.png)


Where the fourth line follows from that P( _Oi|f, C_ ) = P( _Oi|f_ ) since _f ∩ C_ = _f_ . And the second last line is derived from that P( _f |C_ ) = 0 if _f ∩ C_ = _f_ as for a given _C_ , only one value can be chosen for each factor as in Fig. 2. 

A.2 ASSUMPTIONS FOR ESTIMATING P( _Oi|f_ ) 

**Week ordering of the set of factor odds ratios.** Suppose given factors _f_ 1 _, f_ 2 _, ..., fN_ , we assess odds of _o_ 1 _, o_ 2 _, ..., oN_ , respectively in favor of the outcome _Oi, i_ = 1 _,_ 2 occurring. _oj_ = 1 _−_ P(P _O_ ( _Oi|if|jf_ ) _j_ ) _[, j]_[=] 1 _, ..., N_ , the intuitive ordering _≥E_ is defined as: ( _o_ 1 _, o_ 2 _, ..., oN_ ) _≥E_ ( _o[∗]_ 1 _[, o][∗]_ 2 _[, ..., o][∗] N_[)] (11) 

if and only if the decision maker feels at least as sure about the outcome _Oi_ , occurring when given factors _f_ 1 _, f_ 2 _, ..., fN_ we assess odds _o_ 1 _, o_ 2 _, ..., oN_ -as when given factors _f_ 1 _, f_ 2 _, ..., fN_ we assess odds _o[∗]_ 1 _[, o][∗]_ 2 _[, ..., o][∗] N_[.] The ordering _≥E_ is a Weak Ordering if and only if for any ( _o_ 1 _, o_ 2 _, ..., oN_ ),( _o[∗]_ 1 _[, o][∗]_ 2 _[, ..., o][∗] N_[)][,][(] _[o]_ 1 _[∗∗][, o][∗∗]_ 2 _[, ..., o][∗∗] N_[)][, we have:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0014-11.png)


**“Noninteraction” property of factors.** The non-interaction condition says that the decision maker’s interpretation of how one factor affects the outcome does not depend on how other factors affect the outcome. 

14 

Preprint. 

## A.3 LEARNING ALGORITHM FOR ESTIMATING P( _Oi|f_ ) 

We present the complete algorithm in Appendix Algorithm 1. Since P( _O_ 1) and P( _O_ 2) are complementary, we only need to learn P( _O_ 1 _|fja_ ) _, fja ∈Fj, j_ = 1 _, ..., N, a_ = 1 _, ...,_ card( _Fj_ ), and P( _O_ 2 _|fja_ ) = 1 _−_ P( _O_ 1 _|fja_ ). We randomly sample 128 instances in the space of complete information as the training data for each scenario. For a scenario with fewer than 128 instances in the space of complete information, we employ a resampling technique, selecting instances repeatedly until we reach a total of 128 samples. 

## A.4 EXPERIMENT DATASET DESCRIPTION 

We convert three datasets focusing on reasoning and planning to our problem setting. 

- **COMMON2SENSE** (Singh et al., 2021): focuses on commonsense reasoning and planning. The original task constitutes a multi-domain complementary commonsense reasoning benchmark and comprises natural language true/false statements. We exclusively focus on instances of comparative reasoning in both train and test sets where a pre-trained smaller model exhibits low confidence in its prediction logits. Using GPT-4[8] , we rewrite each instance statement into its opposite, creating two distinct outcomes. We then outline a general scenario based on the two statements and generate 10 additional sentences for each statement, with detailed conditions that would support the statement. Additionally, we implement a reverse verification step to eliminate any incorrect additional sentences to ensure dataset quality. We gather 213 scenarios and 3822 instances. 9 additional conditions on average support each outcome. 

- **TODAY** (Feng et al., 2023): focuses on temporal reasoning. The dataset is designed to study how introducing an extra sentence to a specific scenario affects the temporal relation, aligning well with the context of our task. There are 1000 instances. 

- **PLASMA** (Brahman et al., 2023): focuses on planning. The original task is to generate or revise a plan based on a given goal while adhering to an additional condition. We’ve observed that the key variations between the initial and the revised plans given the additional condition typically focus on a specific step within the plan. We then utilize GPT-4 to identify the step and adapt the description of the step in the original and revised plans into two potential outcomes. We manually identify the less common outcome in the 279 scenarios and use GPT-4 to generate 5 additional sentences for the outcome, with detailed conditions that would support this outcome instead of the other. We gather 279 scenarios and 1395 instances with 5 additional conditions for each scenario. 

For the two GPT-4 generated datasets, we recruited the authors and their colleagues to conduct a human quality check on 100 randomly selected instances from each dataset, with three people reviewing each instance. All three reviewers agreed on the label for 91% of the generated data. For 94% of the data, at least two reviewers agreed on the label. 

## A.5 HUMAN EVALUATION ON BIRD ESTIMATED PROBABILITIES 

We first explain our choice of relative evaluation over asking human annotators to provide absolute probabilities from a psychological perspective. TVERSKY & KAHNEMAN (1974) shows that humans use heuristic principles to make probability judgments, which often suffer from systematic and predictable errors due to cognitive biases. Kahneman (2003) further discusses how human judgment in probabilistic situations is generally better in relative than absolute terms due to cognitive biases and limitations in processing complex information. We conducted a pilot study with human annotators on MTurk, asking them to provide absolute probabilities for each additional condition separately. However, we were unable to reach a consensus among the 7 annotators. 

We then introduce the human annotation process for pairwise comparison. We collect human gold labels with rigorous quality control for the preference-based pairwise evaluation of the estimated probabilities. We restrict each annotator to take 5 samples at most and there are 386 distinct annotators. We only allow participants who have 98% or more HITs acceptance rate and are located in the US. 

> 8We use GPT-4-0613 when utilizing GPT-4 for annotations. 

15 

Preprint. 

**Algorithm 1:** Training single probabilities to estimate P( _Oi|f_ ) . 

**input :** Training set of complete information _{f }, f ∈F_ , outcome _Oi, i_ = 1 _,_ 2, LLM _M_ , learning rate _η_ , number of epochs _E_ , batch size _B_ . **output :** Trained probabilities P( _Oi|fja_ ) _, fja ∈Fj, j_ = 1 _, ..., N, a_ = 1 _, ...,_ card( _Fj_ ) _, i_ = 1 _,_ 2. # Get verbalized probability scores from the LLM. _′_ **1 for** _f ∈{f }_ **do** _′ ′ ′_ **2** PLLM( _O_ 1 _|f_ ) _,_ PLLM( _O_ 2 _|f_ ) _← M_ ( _f , O_ 1 _, O_ 2) **3 end** # Initialize probabilities and calculate target for the margin ranking loss. **4 for** _j_ = 1 **to** _N_ **do 5 for** _a_ = 1 **to** card( _Fj_ ) **do** 75% _fja_ supports outcome 1 **6** Pinit( _O_ 1 _|fja_ ) = 50% _fja_ is neutral � 25% _fja_ supports opposite outcome 2 1 Pinit( _O_ 1 _|fja_ ) _>_ 50% **7** _y_ target( _O_ 1 _|fja_ ) = 0 Pinit( _O_ 1 _|fja_ ) = 50% � _−_ 1 Pinit( _O_ 1 _|fja_ ) _<_ 50% **8 end 9 end** # Training. **10 for** _epoch_ = 1 **to** _E_ **do 11 for** _each batch of B samples from the training set_ **do 12 for** _b ∈_ 1 _, ..., B(in parallel)_ **do 13** Pestimated( _O_ 1 _|f[b]_ ) = � _Nj_ =1[P][(] _[O]_[1] � _[|][f] Njj[ b]_ =1[)+][P][�][(] _[O][N] j_ =1[1] _[|][f]_[(1] _j[ b][−]_[)][P][(] _[O]_[1] _[|][f][ b] j_[))][#][Forward] propagation. **14** _ℓ[b]_ MSE[= (][P][LLM][(] _[O]_[1] _[|][f][ b]_[)] _[ −]_[P][estimated][(] _[O]_[1] _[|][f][ b]_[))][2][#][Compute][MSE][loss][for] each sample, for _fja ∈Fj, fja_ = _fj[b][,] ∂_ P _∂ℓ_ ( _O[b]_ MSE1 _|fja_ )[= 0] _[, j]_[= 1] _[, ..., N]_[.] **15 end** _B_ **16** _ℓ_ MSE = _B_[1] � _b_ =1 _[ℓ]_ MSE _[b]_[#][Compute][MSE][loss.] **17 for** _j_ = 1 **to** _N_ **do 18 for** _a_ = 1 **to** card( _Fj_ ) **do 19** Ptrained( _O_ 1 _|fja_ ) =[�] _f[∗] ∈F[∗]_[P][estimated][(] _[O]_[1] _[|][f][ ∗][∪][f][ja]_[)][P][(] _[f][ ∗][|][f][ja]_[) =] card(1 _F[∗]_ ) � _f[∗] ∈F[∗]_[P][estimated][(] _[O]_[1] _[|][f][ ∗][∪][f][ja]_[)] **20** _ℓ[ja]_ MR[= max(0] _[,][ −][y]_[target][(] _[O]_[1] _[|][f][ja]_[)(][P][trained][(] _[O]_[1] _[|][f][ja]_[)] _[ −]_[0] _[.]_[5) +] _[ ϵ]_[)] **21 end 22 end 23** _ℓ_ MR = _N_[1] � _Nj_ =1 card(1 _Fj_ ) �card( _a_ =1 _Fj_ ) _ℓ[ja]_ MR[#][Compute][MR][loss.] **24** _L_ = _ℓ_ MSE + _α ∗ ℓ_ MR # Compute final loss. **25 for** _j_ = 1 **to** _N_ **do 26 for** _a_ = 1 **to** card( _Fj_ ) **do** _∂L_ **27** P( _O_ 1 _|fja_ ) = P( _O_ 1 _|fja_ ) _− η ∂_ P( _O_ 1 _|fja_ )[#][Update][parameters.] **28** P( _O_ 2 _|fja_ ) = 1 _−_ P( _O_ 1 _|fja_ ) **29 end 30 end 31 end 32 end** 

16 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0017-01.png)


Figure 3: The interface for human evaluation on preference-based pairwise evaluation of the estimated probabilities. 

We also require annotators to spend at least 1 minute for each instance (the hourly salary is $15). We show the experiment interfaces in Appendix Fig. 3. We first carry out the human annotation process five times on 450 randomly sampled instances from COMMON2SENSE when the condition can be mapped to at least one factor and therefore BIRD does not produce “unknown”. If the 5 annotations have a less than 80% agreement, i.e., less than 4 out of the 5 annotators agree on the same outcome label, we invite 2 additional annotators to reassess this instance. If the agreement of the 7 annotators is more than 70%, i.e., at least 5 out of the 7 annotators agree on the same outcome label, we accept the instance. Other than the previously accepted instances, we also accept an instance as a neutral label if all three labels have been chosen during the annotations and the agreement for the neutral label is more than or equal 40% and the agreement for any of the remaining two labels is less than or equal 40%. This leaves us with 366 instances. We then test the instances across all our methods, retaining only those where all methods can produce an accurate binary decision, i.e., both conditions support the gold outcome over the opposite outcome. We get 350 instances in total, where 12.3% of the instances have a neutral label where condition 1 and condition 2 equally support the outcome, 44.0% of the instances have a label towards condition 1 where condition 1 supports the corresponding outcome better than condition 2, 43.7% of the instances have a label towards condition 2 where condition 2 supports the corresponding outcome better than condition 1. 

We now introduce how precision and recall are calculated. Given a testing instance, assuming the gold outcome is _O[∗]_ and the opposite outcome is _O[¬]_ , there will be 3 model output labels: 1) _P_ ( _O[∗] |C_ 1) _> P_ ( _O[∗] |C_ 2), i.e., the model thinks C1 better support the gold outcome; 2) _P_ ( _O[∗] |C_ 2) _> P_ ( _O[∗] |C_ 1), i.e., the model thinks C2 better support the gold outcome; 3) _P_ ( _O[∗] |C_ 2) = _P_ ( _O[∗] |C_ 1), i.e., the model thinks both _C_ 1 and _C_ 2 equally support the gold outcome. We want to check if the model’s label is the same as the human annotator’s label. Therefore, this can further be seen as a 3-label classification problem. 

A.6 BASELINE DESCRIPTION 

We adopt the following baselines to evaluate our BIRD framework. 

17 

Preprint. 

- **Vanilla Verbalization with Self-Consistency (Vanilla).** (Wang et al., 2023) Given the scenario, an additional condition, and two corresponding outcomes, the model is asked to directly verbalize an estimated probability for each outcome. We show the Prompt in Appendix Fig. 11. The above process is carried out independently for each of the two additional conditions. The additional conditions with higher output probabilities for the gold outcome are the predicted answers. We use temperature = 0.7 to generate 3 responses and take the majority vote answer as the final answer. 

- **Logit-based Probability.** We directly use the output token probability for the decision. We use greedy decoding here to set the temperature = 0. 

- **Chain-of-Thought Verbalization with Self-consistency (CoT).** (Wei et al., 2022) We ask for a reasoning process to think step by step before verbalizing an estimated probability. The remaining process is the same as vanilla verbalization with SC. We show the Prompt in Appendix Fig. 12. 

- **Explicit Comparison (EC).** Instead of evaluating each additional condition separately, both conditions are presented simultaneously to the model. Given the scenario and the gold outcome, the model is asked to select which additional condition better supports the outcome without outputting any probabilities. Note that the task is less challenging for this particular baseline than the others due to the explicit formatting of inference in this baseline. As a result, performances under the EC setting are not directly comparable with our model and other baselines’ performances since they only see one condition at a time and cannot perform argmax operations. We show the Prompt in Appendix Fig. 13. 

- **1/2 Assumption.** Instead of utilizing the learning algorithm mentioned in §3.3 to estimate P( _Oi|f_ ). We adopt a neutral assumption. An outcome is determined based on the factors only if, except the values of the factors that are neutral, the values of all remaining factors uniformly support that particular outcome. Conversely, if the values of the factors indicate different outcomes and there is no prior information, we cannot conclusively determine the outcome. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0018-06.png)


The intuition behind this assumption is that as we focus solely on the present context, if the current information supports both outcomes and no additional information is available to finalize the decision, it becomes impossible to predict which outcome is more likely to occur. 

- **1/N Assumption.** Instead of utilizing the learning algorithm mentioned in §3.3 to estimate P( _Oi|f_ ), we adopt another commonly used neutral assumption. The outcome _Oi_ ’s probability is determined by the aggregate count of factors whose values support outcome _Oi_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0018-09.png)


   - **Fixed Initial Prob.** Instead of training with the algorithm to get an estimated prob for each P( _Oi|fj_ ) in §3.3, we utilize the initial probability of Eq. 5 to directly estimate P( _Oi|f_ ) with Eq. 4. 

- A.7 ANALYSIS OF DIRECT DECISION MAKING 

We first conduct error analysis for llama-2-70b-instruct on direct decision making where BIRD gets the wrong outcomes while COT with SC makes the correct decisions. We detail the type of errors with its frequency percentage in Appendix Table 4. We observe that our proposed method outperforms the chain-of-thought baseline with Llama-3.1 rather than using llama-2. This is partially because Llama-3.1 generates more diverse and accurate factors, and we present an example in Appendix Table 5. 

18 

Preprint. 

|**Error Type**<br>The generated factors and their values are not comprehensive enough to cover all<br>elements of the conditions. The selected values only imply partial information in<br>the additional condition.|**Frequency Percentage**|
|---|---|
||56.7%|
|The additional condition is mapped to incorrect values of the factors that it does<br>not necessarily imply.|30%|
|The calculation of the fnal probability is inaccurate.|13.3%|



Table 4: Error analysis for Llama-2-70b-instruct for direct decision making. 

**Scenario** : the ease of carrying a cup is being compared between one person and six people. **Outcome 1** : It would be easier for one person to carry the cup, than it would be for six people to carry the cup. **Outcome 2** : It would be easier for six people to carry the cup, than it would be for one person to carry the cup. **Llama-2-70b-Instruct Generated Factors Weight and Size of the Cup** : The cup is light and small The cup is heavy and large The cup is of medium weight and size **Material of the Cup** : The cup is fragile The cup is sturdy The cup is made of moderately durable material **Strength of the People** : The person is able-bodied and strong enough to carry the cup The combined strength of six people is greater than that of one person **Experience of the Person or People Involved** : The person has more experience in carrying objects The six people have more experience in lifting objects **Formation of the People** : The six people carry the cup in a specific formation designed to distribute the weight more evenly The six people do not carry the cup in a specific formation **Llama-3.1-70b-Instruct Generated Factors Cup Size and Weight** : The cup is small and light The cup is large and heavy The cup is small and heavy The cup is large and light **Cup Material** : The cup is fragile The cup is sturdy The cup is constructed from a material of moderate durability **Cup Design and Features** : The cup has a comfortable grip and balanced ergonomics The cup has an uncomfortable grip and unbalanced ergonomics The cup has a neutral grip and average ergonomics **Carrier’s Physical Abilities** : The one carrier has more strength and endurance than the six carries The combined strength of the six carries is stronger than the one carrier **Group Dynamics and Coordination** : The group is well-coordinated and works together seamlessly The group is disorganized and has difficulty working together The group has average coordination and works together moderately well **Physical Demands of the Task** : The task requires exceptional strength and endurance The task requires minimal strength and endurance The task requires moderate strength and endurance **The Size and Shape of the Carrying Area** : The carrying area is narrow and confined The carrying area is wide and open 

Table 5: An example where LLama 3.1 generates better factors than Llama 2. 

19 

Preprint. 

We then give insights into LLM direct performance. We present an example from the COMMON2SENSE dataset in Appendix Table 6. where the LLM accurately infers the correct outcome based predominantly on inductive biases, ignoring the other primary information provided in the condition, i.e., the peppers are used in large quantities, which will be captured in our framework. 

## **Example** 

Scenario: The spiciness of a hot sauce is being discussed in relation to the choice between using habanero peppers or jalapeno peppers. Condition: The hot sauce recipe uses a significant amount of habanero peppers relative to the quantity of other ingredients. Outcome 1: A hot sauce is going to be hotter if it uses habanero peppers rather than jalapeno. Outcome 2: A hot sauce is going to be hotter if it uses jalapeno peppers rather than habanero. 

## **Llama-2 Generated Deicision Making Process** 

Habanero peppers are generally much hotter than jalapeno peppers. This is measured on the Scoville scale, which is a measure of the pungency (spiciness or heat) of chili peppers and other spicy foods. So the condition better supports Outcome 1: A hot sauce is going to be hotter if it uses habanero peppers rather than jalapeno. 

Table 6: An example where the LLM correctly makes the decision but with inadequate emphasis on the context. 

## A.8 ABLATION STUDY TOWARDS ABDUCTIVE FACTOR GENERATION 

We compare the final decision-making accuracy based on two different factor generation methods and present the results in Appendix Table 7. We see that our proposed method performs much better than the direct generation method, indicating the effectiveness of utilizing the model’s inductive bias inherited from training data. 

|Dataset<br>TODAY|Method<br>Direct<br>Abduction|BIRD_llama_2<br>70.9<br>73.9|%Unknown<br>74.1<br>54.0|BIRD_llama_3_._1<br>68.9<br>74.3|%Unknown|
|---|---|---|---|---|---|
||||||46.3<br>34.0|
|PLASMA|Direct<br>Abduction|68.6<br>74.0|40.7<br>31.4|69.0<br>73.0|24.1<br>19.0|
|C2S|Direct<br>Abduction|84.2<br>89.0|54.9<br>34.6|89.9<br>92.3|29.0<br>27.1|



Table 7: An ablation study towards different factor generation methods. Direct refers to the direct factor generation method. Abduction refers to our abductive factor generation. 

## A.9 YES-OR-NO FOLLOWUP QUESTION GENERATION 

We experiment to see if BIRD can further facilitate decision making in a controllable way, where we use BIRD to generate a yes-or-no follow-up question about an unobserved factor in the context. Specifically, given a scenario and an additional condition, we first identify which factors and their corresponding values are implied by the condition, we then select a factor that is not mentioned and is most important for the decision in the remaining unmentioned factors, as the target question area. We adopt the factor value that better aligns with the outcome direction suggested by the current implied factor values and has the largest condition probability. We then rewrite the statement of the factor value into its corresponding question. Simultaneously, we ask llama2-70b-instruct to directly generate a yes-or-no follow-up question. The prompt is in Appendix Fig. 15. 

We use MTurk and ask annotators to provide pairwise preference between the BIRD generated question and LLM directly generated question. We adopt the same Mturk setting as mentioned in §A.5. we carry out the human annotation process three times, using the majority vote to find the preferred generated question. We show the experiment interfaces and an example in Appendix Fig. 4. Appendix Table 8 shows that the follow-up questions generated by BIRD are more often preferred. 

20 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0021-01.png)


Figure 4: The interface for human evaluation on preference-based pairwise evaluation of the generated questions. Question 1 is generated by BIRD and question 2 is generated by LLM directly. The three annotators all prefer question 1. 

|Dataset<br>COMMON2SENSE|BIRDWin Direct<br>Tie<br>132<br>36|BIRDLose Direct|
|---|---|---|
|||82|



Table 8: Preference judgments for yes-or-no follow-up questions generated by BIRD versus questions directly generated by LLMs. We randomly sample 250 instances from COMMON2SENSE. 

## A.10 PROMPT 

21 

Preprint. 

## EXAMPLE **ADDITIONAL SENTENCE SAMPLING** PROMPT 

## **SYSTEM** 

You are given a scenario and an accompanying hypothesis. Generate 5 sentences covering different conditions that would add objective information relevant to the hypothesis such that the hypothesis is more likely to hold true. The information should not definitively imply the hypothesis. You must follow the below structure to just generate sentences with no explanations. 

## **USER** 

Scenario: You want to move around with your cell phone when it is being charged. Hypothesis: You can move around more freely with your cell phone if it is being charged with a one-foot cord rather than a six-foot cord. 

## **ASSISTANT** 

# The cell phone is being charged with a portable power bank located in your pocket, allowing you to move around without being tethered to a fixed outlet. # The user is working in a compact space where longer cords could easily snag on furniture or equipment, thus a one-foot cord could minimize this risk. # The phone is needed for tasks that require frequent handling and close proximity to the user, making a shorter cord more practical to avoid excessive dangling. 

# The charging setup includes a small desktop charger that keeps the phone elevated and stable, limiting the practicality of a longer cord. # The user is in a busy environment like a kitchen or workshop, where shorter cords can reduce the hazard of tripping or catching on moving objects. 

## Figure 5: Example **Additional Sentence Sampling** Prompt 

22 

Preprint. 

## EXAMPLE **FACTOR SUMMARIZATION FROM ADDITIONAL SENTENCES** PROMPT 

## **SYSTEM** 

From the given sentences for each outcome, identify and list distinct and concrete factors, ensuring each is broad yet specific and focuses on a unique aspect. 

Your response should strictly adhere to the JSON format provided, without additional explanations. For example: _{_ "distinct factor" <ENSURE each factor focuses on a unique aspect>: "factor values" <Each factor MUST cover at least one condition to support the Statement and one condition to support the Opposite statement.> _}_ 1. Ensure that each factor’s value MUST directly reference specific elements mentioned in the statements, avoiding vague terms like ’the object’. 2. Ensure the factor values are not too concrete. 3. Do not only mention the common situations. **USER** Scenario: You are charging your cell phone and wish to move around with your cell phone. Outcome 1: You can move around more freely with your cell phone if it is being charged with a six feet cord rather than a one feet cord. 

Sentences: 

#1 A longer cord provides more flexibility and allows for a greater range of movement while using the cell phone. This is because the additional length of the six-foot cord gives the user a larger radius of movement, enabling them to comfortably use their phone while it is charging without feeling restricted or confined to a specific location. Outcome 2: You can move around more freely with your cell phone if it is being charged with a one-foot cord rather than a six-foot cord. 

Sentences: 

#2 If the cell phone is plugged into a portable power bank or a USB port on a computer, a one-foot cord provides greater mobility because it is shorter and less likely to get tangled or caught on objects while moving. #3 If the cell phone is constantly being used while charging and the user prefers to keep the phone close to the charger at all times, a one-foot cord allows for easier mobility and reduces the risk of tripping over a longer cord." **ASSISTANT** 

_{_ "The cell phone’s charging method": 

["The charger is portable", "The charger is unmovable"], "The user’s movement range": ["The user stays very close to the charger", "The user has a large radius of movement"], "The location of the phone charger": ["The user leaves the charger somewhere", "The user carries the charger"] _}_ 

Figure 6: Example **Factor Summarization from Additional Sentences** Prompt 

23 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0024-01.png)


**----- Start of picture text -----**<br>
EXAMPLE PROMPT FOR DECIDING THE SCENARIO AND CONDITION BETTER SUP-<br>PORT WHICH OUTCOME<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0024-02.png)


**----- Start of picture text -----**<br>
SYSTEM<br>A scenario and two outcomes are provided. Determine which outcome<br>the condition better supports.<br>USER<br>Scenario: The government is planning the locations for building<br>charging stations.<br>Outcome 1: The government should build a charging station here.<br>Outcome 2: The government should not build a charging station<br>here.<br>Condition: High adoption rate of electric vehicles in the area<br>ASSISTANT<br>The rationale is that a high adoption rate of electric vehicles<br>indicates a strong demand for charging infrastructure. Therefore,<br>building a charging station would help meet the needs of the<br>electric vehicle owners in the area and support further adoption<br>of clean energy transportation.<br>Therefore, the condition provided better supports Outcome 1: The<br>government should build a charging station here.<br>Figure 7: Example Prompt for Deciding the Scenario and Condition Better Support<br>Which Outcome<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models_images/iclr_2025_BIRD_A_Trustworthy_Bayesian_Inference_Framework_for_Large_Language_Models.pdf-0024-03.png)


**----- Start of picture text -----**<br>
EXAMPLE PROMPT FOR  DECIDING IF THE SCENARIO WITH THE CONDITION IMPLIES<br>THE VALUE OF THE FACTOR<br>SYSTEM<br>Decide if the scenario with the condition implies the statement.<br>USER<br>Scenario: Dave was a scientist. Dave wanted to make a great<br>scientific discovery. Dave worked with algae to make electricity.<br>Dave discovered he could make electricity with algae! Dave was<br>awarded for his great discovery.<br>Condition: Dave is known to meticulously plan his investigations<br>and ensure all necessary resources and funds are obtained<br>beforehand.<br>Statement: Dave tends to plan ahead<br>ASSISTANT<br>The scenario and condition indicate that Dave meticulously plans<br>his investigations and ensures all necessary resources and funds<br>are obtained beforehand. This suggests that Dave is proactive and<br>plans ahead of time.<br>So we can conclude that the scenario with the condition implies the<br>statement.<br>Figure 8: Example Prompt for  Deciding If the Scenario with the Condition Implies the<br>Value of the Factor<br>**----- End of picture text -----**<br>


24 

Preprint. 

## EXAMPLE PROMPT FOR **REEVALUATING IMPLIED VALUE** 

## **SYSTEM** 

You are an AI assistant that verifies your own response. The user will give you your previous response. Your task is that given your answer of the implied factors as in the list for the key based on a scenario, check if the key necessarily implies all the values in the list. You should output a JSON with no explanation. Here are the rules that you must follow: 1. You should think about the scenario. 2. If you think the key implies all the values, keep the value list, otherwise only include the ones that are implied. 3. You are allowed to generate an empty list if you think none of the values are implied. 4. Make sure you check if all the conditions in the value are implied by the key, if not, remove the value. 5. Make sure the remaining values do not conflict with each other. **USER** Scenario: The government is planning the locations for building charging stations. _{_ "The location is on a busy highway with no existing charging stations.": ["No nearby charging stations", "Location is on a major travel route, serving long-distance EV travelers", "Nearby amenities like restaurants, shops, and rest areas for users while charging"] _}_ **ASSISTANT** _{_ "The location is on a busy highway with no existing charging stations.": ["No nearby charging stations", "Location is on a major travel route, serving long-distance EV travelers"] _}_ 

Figure 9: Example Prompt for **Reevaluating Implied Value** 

25 

Preprint. 

## EXAMPLE PROMPT FOR **OUTPUTTING A VERBALIZED PROBABILITY SCORE** 

## **SYSTEM** 

As an AI assistant, your role is to respond accurately to user queries. While answering think step-by-step and justify your answer. Analyze the given scenario and condition to determine the likelihood of the outcomes. Use only the information provided, without relying on prior knowledge. Assess the probability using the specified terms: ’very likely’, ’likely’, ’somewhat likely’, ’neutral’, ’somewhat unlikely’, ’unlikely’, ’very unlikely’. Ensure that your assessments are complementary: if one outcome is deemed ’likely’, the other must be ’unlikely’, and so on. You should first give your reasons and then format your final answer. **USER** Scenario: You want to move around with your cell phone when it is being charged. Condition: The charger is portable. The user stays very close to the charger. The user carries the charger. Outcome 1: You can move around more freely with your cell phone with a one-foot cord rather than a six-foot cord. Outcome 2: You can move around more freely with your cell phone with a six-foot cord rather than a one-foot cord. **ASSISTANT** Given that the user is carrying a portable charger, a shorter cord like one foot would indeed be far more manageable, making it easier for the user to move around freely. Given the same conditions, a longer cord like six feet might become an impediment, making it more challenging for the user who is carrying the charger to move around freely due to the possibility of tangling or managing the extra length. Therefore, Outcome 1: Likely Outcome 2: Unlikely 

Figure 10: Example Prompt for **Outputting a Verbalized Probability Score** 

## EXAMPLE PROMPT FOR **DIRECT PROBABILITY VERBALIZATION BASELINE** 

## **SYSTEM** 

You are an AI assistant that helps people make decisions. The user will give you a question. Choose which outcome the scenario and its corresponding additional condition better support. Provide your confidence level as a percentage to indicate how certain you are about your answer. **USER** Scenario: ... Additional Condition: ... Outcome 1: ... Outcome 2: ... **ASSISTANT** Answer and Confidence (0-100): Outcome i, X Figure 11: Example Prompt for **Direct Probability Verbalization Baseline** 

26 

Preprint. 

## EXAMPLE PROMPT FOR **DIRECT PROBABILITY VERBALIZATION WITH COT BASELINE** 

## **SYSTEM** 

You are an AI assistant that helps people make decisions. The user will give you a question. Your task is to answer as faithfully as you can. While answering, think step-by-step and justify your answer. You are asked to provide a probability for each outcome being correct (0% to 100%) based on the scenario and its corresponding additional condition. Ensure that P(outcome 1) + P(outcome 2) = 1 . **USER** Scenario: ... Additional Condition: ... Outcome 1: ... Outcome 2: ... **ASSISTANT** Explanation: [step-by-step analysis here] Confidence: P(outcome 1) = X%, P(outcome 2) = Y%. Figure 12: Example Prompt for **Direct Probability Verbalization with CoT Baseline** 

## EXAMPLE PROMPT FOR **EXPLICIT COMPARISON BASELINE** 

## **SYSTEM** 

You are an AI assistant that helps people make decisions. The user will give you a question. Your task is to answer as faithfully as you can. While answering, think step-by-step and justify your answer. You are asked to decide which condition makes the outcome more likely to be true given the scenario. **USER** Scenario: ... Outcome: ... Condition 1: ... Condition 2: ... **ASSISTANT** [step-by-step analysis here] Therefore, Condition i makes the outcome more likely to be true. Figure 13: Example Prompt for **Explicit Comparison Baseline** 

27 

Preprint. 

## EXAMPLE **COT FOR DECISION MAKING** PROMPT 

## **SYSTEM** 

You are a helpful assistant. While answering think step-by-step and justify your answer. Given a scenario and an additional condition, decide which of the two outcomes is supported. **USER** Scenario: The government is planning the locations for building charging stations. Condition: The location is on a busy highway with no existing charging stations. Outcome 1: The government should build a charging station here. Outcome 2: The government should not build a charging station here. **ASSISTANT** The scenario presents a busy highway with no existing charging stations, and the government is planning where to build new ones. Given the high traffic volume and the absence of charging infrastructure, building a station here would fill a critical gap, support electric vehicle drivers, and encourage the transition to sustainable transportation. 

Based on the available information, Outcome 1: The government should build a charging station here is more supported. 

Figure 14: Example **COT for Decision Making** Prompt, we further conduct self consistency for 3 times to choose the majority vote answer 

## EXAMPLE **DIRECT YES-OR-NO FOLLOW-UP QUESTION GENERATION** PROMPT 

**SYSTEM** You are a helpful assistant. While answering think step-by-step and justify your answer. Ask a follow-up yes or no question so that you can better decide which outcome is better supported by the scenario and the condition. You can only ask for a specific situation. **USER** Scenario: You want to move around with your cell phone when it is being charged. Condition: The charger is portable and the user carries the charger. Outcome 1: You can move around more freely with your cell phone with a one-foot cord rather than a six feet cord. Outcome 2: You can move around more freely with your cell phone with a six-foot cord rather than a one feet cord. 

## **ASSISTANT** 

In this scenario, the primary factor to consider is cord length, in most situations, a greater cord length allows more movement freedom, as you can be further from the outlet and still keep your device charged. However, considering that user carries the charger, it might bring about the confusion between the two outcomes. 

Thus, to further ascertain which outcome is supported better, I’d ask the following question: 

Does the portable charger need to be plugged into an outlet to charge the phone? 

Figure 15: Example **Direct Yes-or-no Follow-up Question Generation** Prompt 

28 

