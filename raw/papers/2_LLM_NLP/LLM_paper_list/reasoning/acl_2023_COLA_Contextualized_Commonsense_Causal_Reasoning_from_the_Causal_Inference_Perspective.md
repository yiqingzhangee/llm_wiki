# **COLA: Contextualized Commonsense Causal Reasoning from the Causal Inference Perspective** 

**Zhaowei Wang**[1] **, Quyet V. Do**[1] **, Hongming Zhang**[2] **, Jiayao Zhang**[3] **, Weiqi Wang**[1] **, Tianqing Fang**[1] **, Yangqiu Song**[1] **, Ginny Y. Wong**[4] **, & Simon See**[4] 

1Department of Computer Science and Engineering, HKUST 

2Tencent AI Lab, Bellevue, USA 3University of Pennsylvania 

4NVIDIA AI Technology Center (NVAITC), NVIDIA, Santa Clara, USA {zwanggy,yqsong}@cse.ust.hk, {gwong,ssee}@nvidia.com 

## **Abstract** 

Detecting commonsense causal relations (causation) between events has long been an essential yet challenging task. Given that events are complicated, an event may have different causes under various contexts. Thus, exploiting context plays an essential role in detecting causal relations. Meanwhile, previous works about commonsense causation only consider two events and ignore their context, simplifying the task formulation. This paper proposes a new task to detect commonsense causation between two events in an event sequence (i.e., context), called contextualized commonsense causal reasoning. We also design a zero-shot framework: **COLA** ( **C** ontextualized C **o** mmonsense Causa **l** ity Re **a** soner) to solve the task from the causal inference perspective. This framework obtains rich incidental supervision from temporality and balances covariates from multiple timestamps to remove confounding effects. Our extensive experiments show that **COLA**[1] can detect commonsense causality more accurately than baselines. 

## **1 Introduction** 

Commonsense Causal Reasoning (CCR) aims at identifying plausible causes and effects of events in natural language that are typically reasonable by an average person (Zhang et al., 2022c). To solve the task, existing efforts devoted by the community mainly rely on language models wholeheartedly with supervised learning approaches (Staliunait¯ e˙ et al., 2021; Sap et al., 2019; Tamborrino et al., 2020; He et al., 2020; Raffel et al., 2020). Those ingenious engineering works have brought significant progress in recent years. However, recent studies (Kavumba et al., 2019; Han and Wang, 2021) found that pure engineering designs are inadequate to seize commonsense causation, as language mod- 

> 1The code and data are available at https://github. com/HKUST-KnowComp/COLA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0001-11.png)


**----- Start of picture text -----**<br>
Which cause is more plausible?<br>Context-Free<br>Cause 1.  Emma felt hungry. Emma made a steak<br>in the kitchen.<br>Cause 2.  Emma was doing her job.<br>Context 1: Emma exercised for a while.<br>Cause 1.  Emma felt hungry. Emma made a steak<br>in the kitchen.<br>Cause 2.  Emma was doing her job.<br>Context 2: Emma is a chef at a steakhouse.<br>Cause 1.  Emma felt hungry. Emma made a steak<br>in the kitchen.<br>Cause 2.  Emma was doing her job.<br>**----- End of picture text -----**<br>


Figure 1: An illustration of leveraging context to conduct commonsense causal reasoning. Both causes could be plausible under different contexts, while only the frequent one (i.e., **Cause 1** ) is plausible without context. 

els tend to reach higher scores by exploiting super- 

Recently, Zhang et al. (2022c) first studied grasping commonsense causation from the causal inference perspective, by drawing analogies between observational studies (Cochran and Chambers, 1965; Rosenbaum, 2002) and natural languages (Zhang and Zhang, 2021). The proposed framework ROCK achieves good potential for the zero-shot CCR task (e.g., COPA by Gordon et al. (2012)). However, Zhang et al. (2022c) only focuses on the commonsense causation between a pair of events without specifying context. Given that events are complex (Chen et al., 2021), an event may have different causes under different contexts (Mostafazadeh et al., 2020). Thus, it is necessary to utilize context when detecting commonsense causation, such as other events related to given ones. Missing a clear and specific context simplifies commonsense causal knowledge and hinders models from detecting commonsense causal relations more accurately. For example, as shown in Figure 1, the frequent cause of “Emma made a steak in the kitchen.” is 

“Emma felt hungry.” However, the cause also could be “Emma was doing her job” if “Emma is a chef at a steakhouse.” Without the context of Emma’s job, models cannot distinguish those two causes and may return to the frequent one. 

To involve context when detecting commonsense causation, we propose a new task to detect causes between two events in an event sequence, called Contextualized Commonsense Causal Reasoning (Contextualized CCR). In this task, models are asked to detect commonsense causal relations between two given events enclosed in an event sequence. Other events in the event sequence can provide a clear and specific definition of the current context, helping models to capture commonsense causation more accurately. In fact, we find that contextualized CCR is a non-trivial task. Directly applying the framework ROCK (Zhang et al., 2022c) cannot achieve competitive performance on the contextualized CCR since it cannot integrate context information. 

We propose the framework **COLA** , which incorporates contextual information from an event sequence, to solve the Contextualized CCR. Our framework adopts the potential-outcomes framework (Rubin, 1974; Rosenbaum, 2002; Rubin, 2005) to estimate the causal estimand ∆ defined as a type of “ _average treatment effect_ ” (ATE), which measures the change in the likelihood of E _j_ ’s occurrence when intervening E _i_ (denoted by _¬_ E _i_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0002-03.png)


where P( _·_ ) can be estimated with a pre-trained language model, such as a masked language model (Devlin et al., 2018). The magnitude of _average treatment effect_ informs the strength of E _i_ ’s effect on E _j_ , and its sign indicates the direction of the effect. For instance, ∆ _≈_ 1 means E _j_ becomes more prone to occur due to the occurrences of E _i_ . In an ideal world (e.g., E _i_ and E _j_ on any study unit occur completely randomly), a plugging-in estimator in Equation (1) suffices for detecting commonsense causation. Nevertheless, spurious correlations introduced by pervasive confounding co-occurrences need to be eliminated for an unbiased estimation of the causal estimand. This can be done by _balancing_ events that precede E _i_ , or _covariates_ . To incorporate context, we design a mechanism to sample diversified covariates from multiple timestamps and use _temporal propensity_ (Zhang et al., 2022c) for balancing. 

We annotated commonsense causal relations between two events (~1.3k examples) within event sequences from ROCStories (Mostafazadeh et al., 2016) to benchmark our proposed contextualized CCR task. We conduct extensive experiments with multiple pre-trained language models, showing that **COLA** can detect cause-and-effect relations more accurately than competitive baselines by a large margin. Our experiments also show that temporality is essential in our framework but not sufficient to detect commonsense causation without covariates being appropriately balanced. 

## **2 Background and Related Works** 

Understanding events and relations between them have long been a challenging NLP task (Chen et al., 2021). The community has dedicated many works to studying various event-centric tasks, including event relation reasoning (Ning et al., 2018; Zhou et al., 2021; Wang et al., 2020), event extraction (Huang et al., 2018; Lai et al., 2020; Zhang et al., 2022b; Lin et al., 2023), event-centric KG construction (Zhang et al., 2020b, 2022a), and many others (Chambers and Jurafsky, 2008; Chen et al., 2020; Jin et al., 2022; Wang et al., 2022b). Among them, there are a few lines of work that are most related to our work: 

**Commonsense Causal Reasoning** Since our work is about Contextualized CCR, we first discuss related works about commonsense causal reasoning. Existing commonsense causal reasoning approaches are typically categorized under the general topic of commonsense reasoning (Rashkin et al., 2018; Sap et al., 2020). Most previous works depend on language models. Remarkable progress in CCR mainly comes from dataset augmentation, training procedure design, and external knowledge (Staliunait¯ e et al.˙ , 2021; Sap et al., 2019; Shwartz et al., 2020; Tamborrino et al., 2020; Iter et al., 2020). Studies (Kavumba et al., 2019; Han and Wang, 2021) show that language models exploit superficial artifacts to achieve suspicious high performance. 

Causal event detection (Mirza and Tonelli, 2014; Mirza et al., 2014) forms another line of work pertinent to CCR. The task aims to detect causal relations in documents, where various methods are proposed (Chang and Choi, 2005; Do et al., 2011; Ning et al., 2019). However, those works consider verbal (e.g., “attack”) or nominal predicates (e.g., “explosion”) as events, oversimplifying the rela- 

tion detection task. In this paper, we study events expressed in free text, facing a more challenging setup but being closer to real applications. 

**Narrative-related Tasks** Since Contextualized CCR is primarily about chains of events, our work is inspired by earlier research that deals with narratives and scripts (Chambers and Jurafsky, 2008; Granroth-Wilding and Clark, 2016; Mostafazadeh et al., 2016; Bhagavatula et al., 2019; Zhang et al., 2020a). In contrast, our work aims to identify causal relations in a chain of events. 

**Methodologies of Causal Inference** Zhang and Zhang (2021) provided the first study to solve the CCR task from the causal inference perspective. The human population studies have scrutinized extensively the causal inference, which identifies causal relations from ubiquitous associations, including biomedical research, agriculture, epidemiology, and economics (Fisher, 1958; Imbens and Rubin, 2015; Giannarakis et al., 2022; Rosenbaum, 2002; Cochran and Chambers, 1965), where researchers usually use the potential-outcomes framework (Splawa-Neyman et al., 1990; Rubin, 1974; Holland, 1986), graphical and structural equation models (Robins, 1986; Pearl, 1995; Heckman, 2005), and Granger causality (Granger, 1969). 

Recent studies have drawn causal inferences on textual data with the help of powerful pre-trained language models (Kang et al., 2017; Keith et al., 2020; Feder et al., 2022). Concurrently, causal inference can improve the robustness and fairness of NLP models (Feder et al., 2022) or boost performance on downstream tasks (Ghosal et al., 2021; Zheng et al., 2022; Alabdulkarim et al., 2021; Wang et al., 2022a). 

## **3 Problem Formulation** 

**Notation** We use sans _−_ serif fonts to represent an event, such as E _i_ in Figure 2, where the subscript _i_ means the _i_ -th event in a sequence. Simultaneously, uppercase serif letters denote _indicators_ of whether the corresponding event occurs: _Ei_ = 1 _{_ E _i_ occurs _}_ , and a lowercase serif letter means the realizations of this indicator: _ei,j_ = 1 _{_ E _i_ occurs to the _j_ -th study unit _}_ . We introduce the point process to more clearly describe the order of **a pair of** events: _E_ ( _t_ ) with _t ∈{_ 0 _,_ 1 _}_ (e.g., past versus present), so that _Ei_ (0) and _Ej_ (1) means that E _i_ happens before E _j_ . We also use E _i ≺_ E _j_ to indicate that E _i_ occurs before E _j_ for simplicity. We 

## write P(E _i ≺_ E _j_ ) = P( _Ei_ (0) _, Ej_ (1)). 

**Task Description** We articulate the Contextualized CCR as a tweaked form of the binary classification problem. Specifically, we provide models with an event sequence of _n_ events: E1 _,_ E2 _, . . . ,_ E _n_ . Models need to find the top- _k_ events in this sequence that more plausibly have commonsense causal relations with the last event E _n_ , where _k_ indicates the number of positive labels in the ground truth. Then, models need to predict the commonsense causal relation between each event pair (E _i_ , E _n_ ) as a positive/negative one. The strength of causal relations between E _i_ and E _n_ can be expressed with _average treatment effect_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0003-08.png)


## **4 Theoretical Mechanism of COLA** 

As discussed in Section 1, we articulate the Contextualized CCR problem as the estimation of the causal estimand ∆, which we model as the change of temporal likelihood _with contexts controlled_ . We adopt the potential-outcomes framework to design **COLA** to eliminate potential confounding effects due to co-occurrences of events when estimating the causal estimand from data. In this section, we first clarify all essential concepts in the theoretical mechanism of **COLA** one by one, including _study unit_ , _intervention_ , _covariate_ , and _propensity_ , by drawing analogies between the underlying causal mechanisms in natural languages with that in human population research. We then describe the implementation of each component in Section 5. 

## **4.1 The Analogy and Study Unit** 

Motivated by Zhang et al. (2022c), we draw the analogy between human subjects and semantic meanings through the following process: assuming that every human subject kept a textbook recording each event (s)he has experienced, we can then treat each textbook (in natural language) as a study unit and infer the temporal relationships between events from it. In this analogy, we clearly understand the study unit in semantic meanings. 

Then, we can formulate contextualized CCR with concepts from the potential-outcome framework. Given two events E _i_ and E _n_ from an event sequence E1 _,_ E2 _, . . . ,_ E _n_ , where we assume that E _ij_ represents the event that the _j_ -th study unit experienced at the timestamp _i_ when E _i_ is supposed to occur. Then for each unit _j_ , we can define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Covariates (1) Multistamp Covariate Sampling Event Timeline (2) Intervention Generation Interventions<br>Emma got up early. E#: … Bob felt sad.<br>Emma washed her face. … Emma washed her face<br>E!"#: Emma exercised for a while<br>Emma rode her bike. Emma felt sad.<br>E!: Emma felt hungry (3) Matched<br>Emma brushed her teeth. Interventions Bob felt hungry<br>…<br>E$: Emma made a steak in the kitchen<br>(4) Average Treatment Effect<br>**----- End of picture text -----**<br>


Figure 2: The overview of our proposed framework **COLA** . Given a pair of events (E _i,_ E _n_ ), the framework contains four steps to detect commonsense causal relations: (1) multistamp covariate sampling, (2) intervention generation (3) selecting a set of matched interventions, (4) computing _average treatment effect_ . 

the treatment assignment as _Eij_ = 1 _{_ E _ij_ = E _i}_ , realizations of covariates as _**x** j_ = ( _xjl_ ) _[N] l_ =1[for] _xjl_ = 1 _{_ X _l ≺_ E _ij}_ , and two potential-outcomes as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0004-03.png)


When the _j_ -th unit receives the treatment assignment _Eij_ , the hypothetical scenario is denoted by E _ij,Ei_ =1 _−Eij_ , which describes what if the assignment were flipped. Clearly, we can only observe either _r_ 0 _,ij_ or _r_ 1 _,ij_ , but not both of them. We can rewrite the causal estimand ∆ _i_ in Equation (2) exactly as an _average treatment effect_ by averaging over the unit index: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0004-05.png)


The above formulation naturally embodies the temporal nature of covariates (Rubin, 2005), which, by definition, are pretreatments that precede treatments. 

## **4.2 Intervention Beyond Negation** 

In human population studies, the generally accepted _stable unit treatment value assumption_ (Rubin, 1980) ensures only one type of non-treatment (usually negation) for each study unit. As events are complicated, we would interpret intervention (manipulation) of semantic meanings in a broader sense. Take E _i_ “Emma felt hungry” from Figure 2 as an example. While “Emma didn’t feel hungry” is the most direct intervention, it is nonetheless too restrictive: Emma may have felt happy; maybe Alice is the one who felt hungry, instead of Emma. Consequently, interventions in our framework are interpreted much broader as any event that could result in a plausible counterfactual of the outcome. 

We use _A_ to represent all possible interventions of an event E _i_ . 

## **4.3 Balancing Covariates and Comparable Study Units** 

We have discussed that the plugging-in estimator in Equation (1) suffers from biases due to potential confounders. One mitigation strategy is properly balancing the covariates (Rubin, 2005), namely events that occur before E _i_ , which ensures that covariates of untreated study units are comparable to those of treated ones. Consider the vaccine trial as an example; one needs to ensure that the health conditions (covariates) in the control group (not administered vaccines) are comparable to the treated group (administered vaccines). As such, we rewrite the causal estimand ∆ in Equation (2) as expectations conditional on the covariates _**x**_ among comparable study units: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0004-12.png)


provided that the treatment assignment is strongly ignorable with respect to potential outcomes (i.e., _r_ 1 _,ij_ and _r_ 0 _,ij_ ) according to the strong ignorability assumption. 

The strong ignorability assumption is essential in causal inference. Simply, it means that given a set of covariates, the treatment assignment among study units can be viewed as “random” (or “ignorable”) with respect to potential outcomes (see, e.g., Rosenbaum (2002); Rubin (2005) for textbook treatments and Zhang et al. (2022c) for more discussions on this assumption in CCR). 

## **4.4 Matching Temporal Propensities** 

Directly estimating Equation (5) may face the issue of data sparsity since we may sample multiple covariates, and combinations of their values grow exponentially. There are various methods to mitigate 

this issue in balancing covariates, including assignment modeling, outcome modeling, and doublyrobust estimations (Rubin, 2005), among which we base our method on propensity score matching. It is a simple and effective method that is widely used in observational studies (Rosenbaum and Rubin, 1983), which matches the propensity scores of study units to balance covariates. The propensity score is defined as _p_ ( _**x**_ ) = P( _Ei_ (1) = 1 _|_ _**x**_ (0)), which represents the probability of E _i_ taking place conditioning on covariates _**x**_ . Since it is unclear how to pack an unordered set of covariates (events) into a sequential input, Zhang et al. (2022c) proposed to use a relaxed notion of temporal propensity vector, defined as the vector of probabilities of E _i_ happening conditional on each covariate _x ∈_ _**x**_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0005-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0005-02.png)


Hence, we can rewrite the conditional expectation in Equation (5) in the form of matching temporal propensity vectors for some fixed threshold _ϵ_ , given below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0005-04.png)


where _f_ (E _i,_ E _n_ ) is an estimate for P(E _i ≺_ E _n_ ) produced by a language model. 

## **5 The COLA Framework** 

After establishing the theoretical mechanism for our framework **COLA** , we describe the implementation of each component of **COLA** in this section. Generally, since events are in free-text form in our task, pre-trained language models play a central role in our framework. Given that LMs are pre-trained on an enormous amount of textual data (Gao et al., 2020; Raffel et al., 2020), it is sensible to suppose that those LMs would emulate the responses of an average reasonable person. 

Specifically, our framework **COLA** takes two events E _i_ and E _n_ from a sequence E1 _,_ E2 _, . . . ,_ E _n_ as input. As shown in Figure 2, our framework **COLA** contains four steps: (1) a multistamp covariate sampler samples a set _X_ of covariates. (2) an intervention generator generates a set _A_ of interventions. (3) A score estimator builds temporal propensity vectors and selects a matched subset _A[′]_ out of _A_ , by estimating the temporality with a 

temporal predictor. (4) Eventually, the same score estimator computes ∆[ˆ] _i_ according to Equation (7). 

**Multistamp Covariate Sampler** Our multistamp covariate sampler is based on GPT-J (6b) (Wang and Komatsuzaki, 2021). For an input event E, we add “Before that,” at the end to build “E Before that,” as the prompt template. For the _i_ -th event in a sequence, we first sample a covariate set _Xi_ , which contains events before E _i_ . To diversify covariates, we also sample events before E1 _,_ E2 _, . . . ,_ E _i−_ 1 separately, forming _X_ 1 _, X_ 2 _, . . . , Xi−_ 1. Those events are also before E _i_ and can serve as covariates due to the transitivity of the temporal relation[2] . We evenly merge covariates before each timestamp to construct the final covariate set: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0005-11.png)


**Union vs. Intersection** While the aforementioned method takes the union of events sampled according to the left-side context of E _i_ , another intuitive approach is to take the intersection of events sampled according to multiple timestamps in the right-side context. In this way, we can collect covariates that happen before all of E _i,_ E _i_ +1 _, . . . ,_ E _n_ , that is, _X_ = _∩[n] l_ = _i[X][l]_[.][We discuss the experimental] results of these two methods in Section 7.3 and found taking _**union**_ works better, which will be the default in the remaining sections. 

**Intervention Generator** This component generates a set _A_ of interventions as discussed in Section 4.2. There are a variety of related works about generating interventions (counterfactuals) of an event (Gardner et al., 2020; Qin et al., 2019; Ribeiro et al., 2020) and we choose PolyJuice (Wu et al., 2021) in our framework owing to its non-task-specific training objective. PolyJuice generates interventions by masking some phrases individually and filling in masks with a fine-tuned GPT2. Then, we apply the semantic role labeling (SRL) tool provided by AllenNLP (Gardner et al., 2018) to extract the verb `V` and two arguments `ARG0` and `ARG1` as phrases to be manipulated (see Appendix A.2 for more details). 

**Temporal Predictor** We prompt a masked language model to estimate the temporal relation scores between two given events E _i_ and E _n_ . The 

> 2In a sequence of temporally ordered events if A _≺_ B and B _≺_ C, then A _≺_ C. 

prompt template “E _i_ <MASK> E _n_ ” predicts scores _fb_ (E _i,_ E _n_ ) and _fa_ (E _i,_ E _n_ ) for the output tokens before and after. Similarly, we can obtain a reversed estimation by inputting “E _n_ <MASK> E _i_ .” Final temporal score _f_ averages scores from both directions: _f_ (E _i,_ E _n_ ) = 12[(] _[f][b]_[(][E] _[i][,]_[ E] _[n]_[)][+] _fa_ (E _n,_ E _i_ )) 

Our temporal predictor needs to be fine-tuned on a temporal relation corpus. Directly applying a pretrained LM encounter the problem of low coverage, where the tokens before and after cannot be found in the top- _k_ prompted tokens (even _k_ = 30). Thus, we fine-tuned masked language models to predict the masked connectives in a prompt learning setting. Intuitively, temporal relations exist between each pair of adjacent events in a chronologically ordered event sequence. Assuming an event sequence contains two adjacent events E _i,_ E _i_ +1, we then can create an example E _i_ before E _i_ +1 and an symmetric example E _i_ +1 after E _i_ . We also construct negative samples by replacing E _i_ or E _i_ +1 with a randomly sampled event from other sequences. Those negative examples can teach models when no temporal relation exists. While we mask before or after in a positive example for models to predict, a special token [none] should be prompted for negative examples. Then, the cross-entropy loss is used to optimize the temporal predictor. We call this fine-tuning process _**temporal fine-tuning**_ . More details about the temporal relation dataset and fine-tuning process are shown in Appendix A.1. 

**Score Estimator** With the temporal predictor, we can estimate P(X _≺_ A) for all covariates X _∈X_ and interventions A _∈A_ : P( _X_ (0) _, A_ (1)) = P(X _≺_ A) = _f_ (X _,_ A). We also need to estimate P( _X_ (0)) to compute conditional probabilities P( _A_ (1) = 1 _|X_ (0)) in temporal propensity vectors _q_ ( _**x**_ ; A). As all covariates X are events preceding E _i_ sampled by GPT-J, there is an implicit conditioning on E _i_ . Thus, we can approximately get P( _X_ (0)) _≈ f_ (X _,_ E _i_ ) (see Appendix B for more details). Then, temporal propensity vectors are computed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0006-03.png)


Finally, the score estimator computes ∆[ˆ] _i_ in Equation (7). We also test normalization methods in Appendix E and observe that some of those nor- 

## **6 Experiment** 

We conduct extensive experiments and compare **COLA** with a wide selection of baselines. 

## **6.1 Dataset** 

Since our work is the first attempt to study Contextualized CCR, we carried out human annotation on Amazon Mechanical Turk. We randomly sampled event sequences from ROCStories (Mostafazadeh et al., 2016), where each sequence contains five chronologically ordered events. Workers are asked to annotate whether an event causes the last event in a sequence. There are two qualification tests to choose workers to maintain rigorous quality control. See more details in Appendix D. 

Eventually, we collected a dataset containing 1,360 event pairs, called _Choice of Plausible Event in Sequence_ ( **COPES** ). We equally divide them into a validation set and a testing set. 

## **6.2 Evaluation Metric** 

We calculate accuracy, F1-score, and Macro F1score between predicted labels and ground truth labels, to automatically evaluate all models on our dataset. Notice that our task definition provides the number of positive events in a sequence, so that recall, precision, and F1-score are the same. 

## **6.3 Baseline Methods** 

We compare our framework to three baselines: 

**CLM Perplexity** An intuitive solution to the contextualized CCR task would be computing perplexity scores for each pair of events with a causal language model (CLM). An event pair ( _Ei_ , _Ej_ ) within a sequence _E_ 1 _, E_ 2 _, . . . , En_ ( _n_ is sequence length) is converted into full-text input with the prompt template: “If _E_ 1 _, E_ 2 _, . . . , En_ , _Ej_ because _Ei_ ”. The causal language models we tested are GPT2, GPT2-medium/large/XL (Radford et al., 2019), and GPT-J (Wang and Komatsuzaki, 2021). 

**Cloze Prompt Score** This baseline proposed by Tamborrino et al. (2020) concatenates two events ( _Ei_ , _Ej_ ) into full-text input. Then, it masks and tries to recover each token with a masked language model. It averages log-likelihood over every token as the final score of two events. The prompt used is the same as **CLM Perplexity** . Multiple masked language models are tested: BERT-base/large (Devlin et al., 2018), 

|**Models**|**Validation**<br>**Acc**<br>**F1**<br>**Ma-F1**<br>∆_Acc_|**Testing**<br>**Acc**<br>**F1**<br>**Ma-F1**<br>∆_Acc_|
|---|---|---|
||||
|Random|59.47<br>42.35<br>55.55<br>-|58.94<br>41.10<br>54.79<br>-|
||||
|CLM Perplexity (GPT2)<br>CLM Perplexity (GPT2-medium)<br>CLM Perplexity (GPT2-large)<br>CLM Perplexity (GPT2-XL)<br>CLM Perplexity (GPT-J 6b)|61.76<br>45.61<br>58.06<br>-<br>60.29<br>43.51<br>56.45<br>-<br>62.94<br>47.28<br>59.35<br>-<br>62.65<br>46.86<br>59.03<br>-<br>63.82<br>48.54<br>60.32<br>-|61.47<br>44.73<br>57.58<br>-<br>61.76<br>45.15<br>57.90<br>-<br>62.65<br>46.41<br>58.87<br>-<br>62.35<br>45.99<br>58.55<br>-<br>62.06<br>45.57<br>58.22<br>-|
||||
|ClozePromptScore (BERT-base)<br>ClozePromptScore (BERT-large)<br>ClozePromptScore (RoBERTa-base)<br>ClozePromptScore (RoBERTa-large)<br>ClozePromptScore (DeBERTa-base)<br>ClozePromptScore (DeBERTa-large)|64.41<br>49.37<br>60.97<br>-<br>66.47<br>52.30<br>63.23<br>-<br>59.71<br>42.68<br>55.81<br>-<br>60.59<br>43.93<br>56.77<br>-<br>56.76<br>38.49<br>52.58<br>-<br>56.47<br>38.08<br>52.26<br>-|63.53<br>47.68<br>59.84<br>-<br>62.06<br>45.57<br>58.22<br>-<br>59.71<br>42.19<br>55.63<br>-<br>59.12<br>41.35<br>54.99<br>-<br>58.53<br>40.51<br>54.34<br>-<br>57.06<br>38.40<br>52.72<br>-|
||||
|ROCK (BERT-base)<br>ROCK (BERT-large)<br>ROCK (RoBERTa-base)<br>ROCK (RoBERTa-large)<br>ROCK (DeBERTa-base)<br>ROCK (DeBERTa-large)|66.18<br>51.88<br>62.90<br>-<br>65.59<br>51.05<br>62.26<br>-<br>61.76<br>45.61<br>58.06<br>-<br>62.94<br>47.28<br>59.35<br>-<br>62.65<br>46.86<br>59.03<br>-<br>64.41<br>49.37<br>60.97<br>-|65.29<br>50.21<br>61.79<br>-<br>66.47<br>51.90<br>63.08<br>-<br>61.18<br>44.30<br>57.25<br>-<br>65.59<br>50.63<br>62.11<br>-<br>60.59<br>43.46<br>56.61<br>-<br>64.12<br>48.52<br>60.49<br>-|
||||
|**COLA**(BERT-base)<br>**COLA**(BERT-large)<br>**COLA**(RoBERTa-base)<br>**COLA**(RoBERTa-large)<br>**COLA**(DeBERTa-base)<br>**COLA**(DeBERTa-large)|67.65<br>53.97<br>64.52<br>_↑_1.47<br>70.29<br>57.74<br>67.42<br>_↑_4.70<br>69.71<br>56.90<br>66.77<br>_↑_**7.95**<br>70.59<br>58.16<br>67.74<br>_↑_7.65<br>69.71<br>56.90<br>66.77<br>_↑_7.06<br>**71.18**<br>**59.00**<br>**68.39**<br>_↑_6.77|68.82<br>55.27<br>65.67<br>_↑_3.53<br>**70.29**<br>**57.38**<br>**67.29**<br>_↑_3.82<br>66.76<br>52.32<br>63.41<br>_↑_5.58<br>70.00<br>56.96<br>66.97<br>_↑_4.41<br>**70.29**<br>**57.38**<br>**67.29**<br>_↑_**9.70**<br>69.41<br>56.12<br>66.32<br>_↑_5.29|



Table 1: Performance of all frameworks on the validation and testing set of the **COPES** dataset. **COLA** is our model. We abbreviate Accuracy, F1-score, Macro F1-score to Acc, F1, Ma-F1, respectively. We test **COLA** with the temporal predictor based on different models: BERT-base/large, RoBERTa-base/large, and DeBERTabase/large. Compared with ROCK, improvements of our frameworks are shown under ∆ _Acc_ for each language model, respectively. 

RoBERTa-base/large (Liu et al., 2019), DeBERTabase/large (He et al., 2020). 

**ROCK** This baseline is a causal inference framework (Zhang et al., 2022c) that draws analogies between human subjects and natural language. We test different language models for the temporal predictor: BERT-base/large, RoBERTa-base/large, and DeBERTa-base/large. 

## **7 Main Evaluation** 

We provide results in Table 1 for baselines and **COLA** with the temporal predictor based on different language models. In general, **COLA** can detect commonsense causal relations more accurately, outperforming all baseline models by a large margin. Our framework **COLA** based on DeBERTa-large and DeBERTa-base (also BERT-large) achieves the best performance on the validation and testing set, respectively. Also, changing language models of the temporal predictor in **COLA** only involves a small fluctuation in performance, showing that our framework is robust to underlying language mod- 

els. 

Another observation is that CLM Perplexity and ClozePromptScore can achieve performance higher than the random result. This manifests that pre-trained language models with commonly used pre-training objectives can capture commonsense causal relations to some extent. 

## **7.1 Ablation Study** 

In this section, we conduct four ablation experiments to demonstrate that our causal inferencemotivated framework can mitigate spurious correlations between events and boost performance. 

**Temporal Propensity Matching** The first three ablation experiments prove the effectiveness of temporal propensity matching. Here, we separately remove three modules in our framework: (i) We drop the _multistamp covariate sampler_ and sample covariates only based on the last timestamp ( _⋄_ w/o Multi Step). This experiment verifies the benefit of utilizing context to detect commonsense causality. (ii) We remove all interventions ( _⋄_ w/o Inter) and use temporal precedence as causation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0008-00.png)


**----- Start of picture text -----**<br>
0 . 6 BERT-large 0 . 6 RoBERTa-large 0 . 6 DeBERTa-large<br>Ours CPS<br>ROCK Random<br>0 . 5 0 . 5 0 . 5<br>0 . 4 0 . 4 0 . 4<br>0 . 3 0 . 3 0 . 3<br>0 . 0 0 . 1 0 . 2 0 . 3 0 . 4 0 . 5 0 . 0 0 . 1 0 . 2 0 . 3 0 . 4 0 . 5 0 . 0 0 . 1 0 . 2 0 . 3 0 . 4 0 . 5<br>Temporal Propensity Threshold ϵ (on testing data)<br>F1-Score<br>Zero-Shot<br>**----- End of picture text -----**<br>


Figure 3: F1-score of **COLA** (“Ours”) with _ϵ_ ranging from 0 to 0.5. We also compare our framework with baselines ClozePromptScore (“CPS”) and ROCK. Balancing covariates in our framework can significantly outperform unbalanced variants ( _ϵ ≈_ 0 _._ 5). 

|**Models**|**Acc**<br>**F1**<br>**Ma-F1**<br>∆_A_|
|---|---|
|Ours(BERT-large)<br>**70.29**<br>**57.38**<br>**67.29**<br>-||
|||
|_⋄_w/o Multi Step<br>70.00<br>56.96<br>66.97<br>_↓_0.2<br>_⋄_w/o Inter<br>60.29<br>43.04<br>56.28<br>_↓_10<br>_⋄_w/o Cov<br>57.65<br>39.24<br>53.37<br>_↓_12<br>_⋄_w/o Temp<br>64.41<br>48.95<br>60.82<br>_↓_5.8||
|||
|Ours(RoBERTa-large)|**70.00**<br>**56.96**<br>**66.97**<br>-|
|||
|_⋄_w/o Multi Step<br>67.94<br>54.01<br>64.70<br>_↓_2.0<br>_⋄_w/o Inter<br>64.71<br>49.37<br>61.14<br>_↓_5.2<br>_⋄_w/o Cov<br>58.53<br>40.51<br>54.34<br>_↓_11<br>_⋄_w/o Temp<br>65.00<br>49.79<br>61.46<br>_↓_5.0||
|Ours(DeBERTa-large)<br>**69.41**<br>**56.12**<br>**66.32**<br>-||
|||
|_⋄_w/o Multi Step<br>_⋄_w/o Inter<br>_⋄_w/o Cov<br>_⋄_w/o Temp|68.24<br>54.43<br>65.03<br>_↓_1.1<br>67.65<br>53.59<br>64.38<br>_↓_1.7<br>55.88<br>36.71<br>51.42<br>_↓_13<br>64.41<br>48.95<br>60.82<br>_↓_5.0|



Table 2: Ablation study on **COLA** (“Ours”). The column ∆ _Acc_ indicates accuracy drops. The first three experiments shows the effectiveness of temporal propensity matching and the last one is about temporal predictor. 

ˆ∆ _i_ = P(E _i ≺_ E _n_ ), equivalent to _ϵ_ = 0 in Equation (7). (iii) Covariates are removed ( _⋄_ w/o Cov) so that interventions are not adjusted. This unadjusted score keeps all sampled interventions, equivalent to _ϵ_ = 1 in Equation (7). 

From the results in Table 2, we observe that balanced estimand ∆[ˆ] _i_ achieve better performances, showing that multiple timestamp sampling, treatment effect, and balancing covariates play essential roles in detecting commonsense causation accurately. Removing any of the three modules will result in sheer drops in all metrics. These experiments imply that temporal relation is vital in Contextualized CCR, but it is still insufficient due to spurious correlations. Thus, we need to measure 

|**Models**|**Acc**|**F1**|**Ma-F1**|∆_Acc_|
|---|---|---|---|---|
|**Uni**(BERT-large)|70.29|57.38|67.29|-|
|**Int**(BERT-large)|68.82|54.89|65.54|_↓_1.47|
|**Uni**(RoBERTa-large)|70.00|56.96|66.97|-|
|**Int**(RoBERTa-large)|68.82|54.89|65.54|_↓_1.18|
|**Uni**(DeBERTa-large)|69.41|56.12|66.32|-|
|**Int**(DeBERTa-large)|68.82|54.89|65.54|_↓_0.59|



Table 3: Comparison of the different covariate sampling methods. **Uni** and **Int** stand for “Union” and “Intersection,” respectively. 

_average treatment effect_ with balancing covariates. 

**Temporal Predictor** We also ablate the temporal predictor ( _⋄_ w/o Temp) to verify the effectiveness of _**temporal fine-tuning**_ (in Section 5). Here, we use pre-trained language models and increase _k_ to 30 to mitigate the problem of low coverage. 

As shown in Table 2, a directly pre-trained language model without fine-tuning cannot perform well. We conclude that pre-trained language models do not have sufficient “temporal awareness” and _**temporal fine-tuning**_ is necessary for our framework. 

## **7.2 Rules-of-thumb for Choosing** _ϵ_ **:** 

The hyperparameter _ϵ_ controls the number of interventions when balancing covariates in Equation (7). In Figure 3, we can observe that a recommended range for _ϵ_ is _ϵ ∈_ [0 _,_ 0 _._ 1]. We also list the optimal _ϵ_ in Appendix A.3. From Table 6, _ϵ_ should be fairly small within [0.001, 0.015]. Though the best solution relies on how to implement the components in **COLA** and data distribution, our analysis can provide a good starting point. We also study the effect of changing another important hyperparameter: covariate set size _N_ = _|X|_ , in Appendix C. 

|**Models**|**Acc**<br>**F1**<br>**Ma-F1**|
|---|---|
|||
|ChatGPT<br>ChatGPT w/o_k_<br>**COLA**(DeBERTa-large)|75.83<br>60.27<br>71.45<br>73.33<br>62.79<br>71.01<br>**80.00**<br>**65.71**<br>**75.80**|



Table 4: Comparison with ChatGPT. Our model outperforms ChatGPT with much fewer parameters. 

The experimental result in the selected 120 examples is shown in Table 4. From the table, we can find that providing ChatGPT with the number _k_ does not lead to too much change in all metrics. (More discussion about this in Appendix F) Also, our framework **COLA** achieves better zeroshot performance than ChatGPT with much fewer parameters. 

## **7.3 Union and Intersection** 

Our framework **COLA** samples covariates from multiple timestamps and take union on them to get the final covariate set. We also introduce another method to sample covariates preceding all timestamps after and including _Ei_ (“intersection”) in Section 5. Here, we conduct experiments to discuss the differences between these two methods. For the “intersection” method, we also use GPT-J to sample covariates with “There are temporally ordered events [ _Ei, Ei_ +1 _, . . . , En_ ]. Before all events,” being the prompt template. 

As shown in Table 3, the “union” method gets better performance since it can diversify covariate sets. It samples covariates conditioned on each event before _Ei_ separately. Meantime, each covariate of the “intersection” method is only conditioned on the same context _Ei, Ei_ +1 _, . . . , En_ . We compute the self-BLEU (Zhu et al., 2018) to evaluate the diversity of the generated covariates. The self-BLEU of the “intersection” method is 66.40% while that of the “union” method is 41.34%, quantitatively showing that our method can diversify the covariate set. 

## **7.4 Comparison with ChatGPT** 

Large language models have shown strong performance on extensive NLP tasks (Radford et al., 2019; Ouyang et al., 2022). Thus, we compare our framework with ChatGPT[3] , the latest large language model trained using Reinforcement Learning from Human Feedback (RLHF). To adopt ChatGPT to our task, we design the prompt template: “Given a story as a chain of events _E_ 1 _, E_ 2 _, . . . , En_ , which _k_ event(s) in the first _n −_ 1 events more plausibly cause the last event?” where _k_ is the number of positive events in the ground truth. We randomly sample 120 examples within 30 event sequences from our dataset **COPES** and manually read predicted labels from ChatGPT’s answers. We also test ChatGPT without providing the number _k_ , i.e., removing _k_ from the prompt. 

## **8 Conclusion** 

In this paper, we design a new task to consider the context when detecting commonsense causal relations. We also crowd-sourced a dataset with strict quality control to benchmark the task. Our **COLA** framework is motivated by the principles of causal inference and attempts to leverage context information. The extensive evaluation demonstrates the effectiveness of **COLA** for detecting commonsense causal relations. 

## **9 Acknowledgement** 

The authors of this paper were supported by the NSFC Fund (U20B2053) from the NSFC of China, the RIF (R6020-19 and R6021-20) and the GRF (16211520 and 16205322) from RGC of Hong Kong, the MHKJFS (MHP/001/19) from ITC of Hong Kong and the National Key R&D Program of China (2019YFE0198200) with special thanks to HKMAAC and CUSBLT. This paper was also supported by the Tencent Rhinobird Focused Research Program. We also thank the support from NVIDIA AI Technology Center (NVAITC) and the UGC Research Matching Grants (RMGS20EG01-D, RMGS20CR11, RMGS20CR12, RMGS20EG19, RMGS20EG21, RMGS23CR05, RMGS23EG08). 

3https://openai.com/blog/chatgpt/ 

## **Limitations** 

Our **COLA** framework is mainly based on pretrained language models, such as GPT-J and PolyJuice. We only utilize the simplest prompt to sample covariates and interventions. Further efforts such as prompt engineering are expected to boost the performance of **COLA** further. Also, pre-trained language models can be biased by implicit events and reporting biases in their training data. Such biases lead the framework to omit some critical covariates and interventions, hindering our framework from achieving more accurate detection of commonsense causal relations. How to account for this problem of language model remains to be studied. Last but not least, the temporal propensity used in our framework is only an approximation of the exact propensity (Zhang et al., 2022c) since it is unclear how to pack an unordered set of covariates into a sequential input for language models. Further studies are needed to design the exact propensity and improve performance. 

## **Ethics Statement** 

This work presents **COPES** , a free and open dataset for the research community to study the contextualized CCR problem. Examples in **COPES** are collected from ROCStories (Mostafazadeh et al., 2016), a free and open dataset about anonymized chains of events. Each chain logically follows everyday topics and does not involve privacy problems about any specific entities (e.g., a person or company). We carried out human annotation on Amazon Mechanical Turk. Annotators are fairly paid 1.2 USD for each HIT, which fulfills the minimum wage requirement. 

## **References** 

- Amal Alabdulkarim, Siyan Li, and Xiangyu Peng. 2021. Automatic story generation: Challenges and attempts. In _Proceedings of the Third Workshop on Narrative Understanding_ , pages 72–83. 

- Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Wen-tau Yih, and Yejin Choi. 2019. Abductive commonsense reasoning. In _International Conference on Learning Representations_ . 

- Nathanael Chambers and Dan Jurafsky. 2008. Unsupervised learning of narrative event chains. In _Proceedings of ACL-08: HLT_ , pages 789–797. 

- Du-Seong Chang and Key-Sun Choi. 2005. Causal relation extraction using cue phrase and lexical pair probabilities. In _International Conference on Natural Language Processing_ , pages 61–70. Springer. 

- Muhao Chen, Hongming Zhang, Qiang Ning, Manling Li, Heng Ji, Kathleen McKeown, and Dan Roth. 2021. Event-centric natural language processing. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Tutorial Abstracts_ , pages 6–14. 

- Muhao Chen, Hongming Zhang, Haoyu Wang, and Dan Roth. 2020. What are you trying to do? semantic typing of event processes. In _Proceedings of the 24th Conference on Computational Natural Language Learning_ , pages 531–542. 

- William G Cochran and S Paul Chambers. 1965. The planning of observational studies of human populations. _Journal of the Royal Statistical Society. Series A (General)_ , 128(2):234–266. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ . 

- Quang Do, Yee Seng Chan, and Dan Roth. 2011. Minimally supervised event causality identification. In _Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing_ , pages 294–303. 

- Amir Feder, Katherine A Keith, Emaad Manzoor, Reid Pryzant, Dhanya Sridhar, Zach Wood-Doughty, Jacob Eisenstein, Justin Grimmer, Roi Reichart, Margaret E Roberts, et al. 2022. Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. _Transactions of the Association for Computational Linguistics_ , 10:1138–1158. 

- Ronald A Fisher. 1958. Cancer and smoking. _Nature_ , 182(4635):596–596. 

- Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. _Psychological bulletin_ , 76(5):378. 

- Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020. The pile: An 800gb dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_ . 

- Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, et al. 2020. Evaluating models’ local decision boundaries via contrast sets. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 1307–1323. 

- Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F Liu, Matthew Peters, Michael Schmitz, and Luke Zettlemoyer. 2018. Allennlp: A deep semantic natural language processing platform. _ACL 2018_ , page 1. 

- Deepanway Ghosal, Pengfei Hong, Siqi Shen, Navonil Majumder, Rada Mihalcea, and Soujanya Poria. 2021. Cider: Commonsense inference for dialogue explanation and reasoning. In _Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue_ , pages 301–313. 

- Georgios Giannarakis, Vasileios Sitokonstantinou, Roxanne Suzette Lorilla, and Charalampos Kontoes. 2022. Towards assessing agricultural land suitability with causal machine learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 1442–1452. 

- Andrew Gordon, Zornitsa Kozareva, and Melissa Roemmele. 2012. Semeval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In _* SEM 2012: The First Joint Conference on Lexical and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012)_ , pages 394–398. 

- Clive WJ Granger. 1969. Investigating causal relations by econometric models and cross-spectral methods. _Econometrica: journal of the Econometric Society_ , pages 424–438. 

- Mark Granroth-Wilding and Stephen Clark. 2016. What happens next? event prediction using a compositional neural network model. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 30. 

- Mingyue Han and Yinglin Wang. 2021. Doing good or doing right? exploring the weakness of commonsense causal reasoning models. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)_ , pages 151–157. 

- Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. Deberta: Decoding-enhanced 

bert with disentangled attention. In _International Conference on Learning Representations_ . 

- James J Heckman. 2005. Rejoinder: response to sobel. _Sociological Methodology_ , 35(1):135–150. 

- Paul W Holland. 1986. Statistics and causal inference. _Journal of the American statistical Association_ , 81(396):945–960. 

- Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, and Clare Voss. 2018. Zero-shot transfer learning for event extraction. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2160–2170. 

- Guido W Imbens and Donald B Rubin. 2015. _Causal inference in statistics, social, and biomedical sciences_ . Cambridge University Press. 

- Dan Iter, Kelvin Guu, Larry Lansing, and Dan Jurafsky. 2020. Pretraining with contrastive sentence objectives improves discourse performance of language models. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4859–4870. 

- Zijian Jin, Xingyu Zhang, Mo Yu, and Lifu Huang. 2022. Probing script knowledge from pre-trained models. _arXiv preprint arXiv:2204.10176_ . 

- Dongyeop Kang, Varun Gangal, Ang Lu, Zheng Chen, and Eduard Hovy. 2017. Detecting and explaining causes from text for a time series event. _arXiv preprint arXiv:1707.08852_ . 

- Pride Kavumba, Naoya Inoue, Benjamin Heinzerling, Keshav Singh, Paul Reisert, and Kentaro Inui. 2019. When choosing plausible alternatives, clever hans can be clever. In _Proceedings of the First Workshop on Commonsense Inference in Natural Language Processing_ , pages 33–42. 

- Katherine A Keith, David Jensen, and Brendan O’Connor. 2020. Text and causal inference: A review of using text to remove confounding from causal estimates. _arXiv preprint arXiv:2005.00649_ . 

- Viet Dac Lai, Thien Huu Nguyen, and Franck Dernoncourt. 2020. Extensively matching for few-shot learning event detection. In _Proceedings of the First Joint Workshop on Narrative Understanding, Storylines, and Events_ , pages 38–45. 

- Zizheng Lin, Hongming Zhang, and Yangqiu Song. 2023. Global constraints with prompting for zeroshot event argument classification. In _Findings of the Association for Computational Linguistics: EACL 2023_ , pages 2482–2493. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ . 

- Paramita Mirza, Rachele Sprugnoli, Sara Tonelli, and Manuela Speranza. 2014. Annotating causality in the tempeval-3 corpus. In _Proceedings of the EACL 2014 Workshop on Computational Approaches to Causality in Language (CAtoCL)_ , pages 10–19. 

- Paramita Mirza and Sara Tonelli. 2014. An analysis of causality between events and its relation to temporal information. In _Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers_ , pages 2097–2106. 

- Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation for deeper understanding of commonsense stories. In _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 839–849. 

- Nasrin Mostafazadeh, Aditya Kalyanpur, Lori Moon, David Buchanan, Lauren Berkowitz, Or Biran, and Jennifer Chu-Carroll. 2020. Glucose: Generalized and contextualized story explanations. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 4569–4586. 

- Qiang Ning, Zhili Feng, Hao Wu, and Dan Roth. 2019. Joint reasoning for temporal and causal relations. _arXiv preprint arXiv:1906.04941_ . 

- Qiang Ning, Hao Wu, and Dan Roth. 2018. A multiaxis annotation scheme for event temporal relations. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1318–1328. 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. _arXiv preprint arXiv:2203.02155_ . 

- Judea Pearl. 1995. Causal diagrams for empirical research. _Biometrika_ , 82(4):669–688. 

- Lianhui Qin, Antoine Bosselut, Ari Holtzman, Chandra Bhagavatula, Elizabeth Clark, and Yejin Choi. 2019. Counterfactual story reasoning and generation. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 5043– 5053. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. 2020. Exploring the limits 

of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 21(140):1–67. 

- Hannah Rashkin, Maarten Sap, Emily Allaway, Noah A Smith, and Yejin Choi. 2018. Event2mind: Commonsense inference on events, intents, and reactions. _arXiv preprint arXiv:1805.06939_ . 

- Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of nlp models with checklist. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4902– 4912. 

- James Robins. 1986. A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. _Mathematical modelling_ , 7(9-12):1393–1512. 

- Paul R Rosenbaum. 2002. Observational studies. In _Observational Studies_ , pages 1–17. Springer. 

- Paul R Rosenbaum and Donald B Rubin. 1983. The central role of the propensity score in observational studies for causal effects. _Biometrika_ , 70(1):41–55. 

- Donald B Rubin. 1974. Estimating causal effects of treatments in randomized and nonrandomized studies. _Journal of educational Psychology_ , 66(5):688. 

- Donald B Rubin. 1980. Bias reduction using mahalanobis-metric matching. _Biometrics_ , pages 293–298. 

- Donald B Rubin. 2005. Causal inference using potential outcomes: Design, modeling, decisions. _Journal of the American Statistical Association_ , 100(469):322–331. 

- Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social iqa: Commonsense reasoning about social interactions. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 4463– 4473. 

- Maarten Sap, Vered Shwartz, Antoine Bosselut, Yejin Choi, and Dan Roth. 2020. Commonsense reasoning for natural language processing. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts_ , pages 27–33. 

- Vered Shwartz, Peter West, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. Unsupervised commonsense question answering with selftalk. _arXiv preprint arXiv:2004.05483_ . 

- Jerzy Splawa-Neyman, Dorota M Dabrowska, and TP Speed. 1990. On the application of probability theory to agricultural experiments. essay on principles. section 9. _Statistical Science_ , pages 465–472. 

- Ieva Stali¯unait˙e, Philip John Gorinski, and Ignacio Iacobacci. 2021. Improving commonsense causal reasoning by adversarial training and data augmentation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 13834–13842. 

- Alexandre Tamborrino, Nicola Pellicanò, Baptiste Pannier, Pascal Voitot, and Louise Naudin. 2020. Pretraining is (almost) all you need: An application to commonsense reasoning. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 3878–3887. 

- Ben Wang and Aran Komatsuzaki. 2021. GPTJ-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/ kingoflolz/mesh-transformer-jax. 

- Haoyu Wang, Muhao Chen, Hongming Zhang, and Dan Roth. 2020. Joint constrained learning for event-event relation extraction. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 696–706. 

- Yuxin Wang, Jieru Lin, Zhiwei Yu, Wei Hu, and Börje F Karlsson. 2022a. Open-world story generation with structured knowledge enhancement: A comprehensive survey. _arXiv preprint arXiv:2212.04634_ . 

- Zhaowei Wang, Hongming Zhang, Tianqing Fang, Yangqiu Song, Ginny Wong, and Simon See. 2022b. SubeventWriter: Iterative sub-event sequence generation with coherence controller. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 1590–1604, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer, and Daniel S Weld. 2021. Polyjuice: Generating counterfactuals for explaining, evaluating, and improving models. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 6707–6723. 

   - Hongming Zhang, Xin Liu, Haojie Pan, Yangqiu Song, and Cane Wing-Ki Leung. 2020b. Aser: A largescale eventuality knowledge graph. In _Proceedings of the web conference 2020_ , pages 201–211. 

   - Hongming Zhang, Wenlin Yao, and Dong Yu. 2022b. Efficient zero-shot event extraction with contextdefinition alignment. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 7169–7179, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

   - Jiayao Zhang, Hongming Zhang, Weijie Su, and Dan Roth. 2022c. Rock: Causal inference principles for reasoning about commonsense causality. In _International Conference on Machine Learning_ , pages 26750–26771. PMLR. 

   - Junhao Zheng, Zhanxian Liang, Haibin Chen, and Qianli Ma. 2022. Distilling causal effect from miscellaneous other-class for continual named entity recognition. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . 

   - Ben Zhou, Kyle Richardson, Qiang Ning, Tushar Khot, Ashish Sabharwal, and Dan Roth. 2021. Temporal reasoning on implicit events from distant supervision. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 1361–1371. 

   - Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation models. In _The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval_ , pages 1097–1100. 

- Bo Zhang and Jiayao Zhang. 2021. Some reflections on drawing causal inference using textual data: Parallels between human subjects and organized texts. In _First Conference on Causal Learning and Reasoning_ . 

- Hongming Zhang, Muhao Chen, Haoyu Wang, Yangqiu Song, and Dan Roth. 2020a. Analogous process structure induction for sub-event sequence prediction. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1541–1550. 

- Hongming Zhang, Xin Liu, Haojie Pan, Haowen Ke, Jiefu Ou, Tianqing Fang, and Yangqiu Song. 2022a. Aser: Towards large-scale commonsense knowledge acquisition via higher-order selectional preference over eventualities. _Artificial Intelligence_ , page 103740. 

|**Models**|**Valid Acc**<br>**Test Acc**|
|---|---|
|||
|Ours(BERT-base)<br>Ours(BERT-large)<br>Ours(RoBERTa-base)<br>Ours(RoBERTa-large)<br>Ours(DeBERTa-base)<br>Ours(DeBERTa-large)|87.10<br>85.81<br>88.93<br>87.38<br>89.76<br>88.97<br>91.91<br>91.10<br>90.07<br>89.06<br>93.10<br>92.40|



Table 5: The accuracy of each temporal predictor on the validation and testing splits of the temporal relation dataset. 

## **A Implementation Details** 

In this appendix, we introduce the implementation details of every component in our framework **COLA** . We conduct all experiments on 8 NVIDIA A100 GPUs. Since our **COLA** and baselines are zero-shot, we test each model once and report the results. 

**Parameter Number** We list parameter numbers of all pre-trained langauge models here. BERTbase/large, RoBERTa-base/large, and DeBERTabase/large have 110M, 340M, 125M, 355M, 100M, and 350M parameters, respectively. GPT, GPTmedium/large/XL, and GPT-6b contain 117M, 345M, 774M, 1.5B, and 6B parameters, respectively. 

## **A.1 Temporal Predictor** 

We fine-tune temporal predictors based on BERTbase/large, RoBERTa-base/large, and DeBERTabase/large for ten epochs. The learning rate is 1e-5, and the batch size is 256. We sampled 800k examples (event pairs) from ROCStories (Mostafazadeh et al., 2016) to build a temporal relation dataset. The ROCStories dataset contains about 100k temporally ordered event sequences. The proportion of training, validation, and testing splits of our sampled event pairs are 98:1:1. We show the accuracy of each fine-tuned temporal predictor on the validation and testing splits in Table 5. 

## **A.2 Intervention and Covariate** 

For each event pair, we sample 50 covariates using GPT-J with a maximum length of generated events of 15 and a temperature of 0.9. We also sample 50 interventions using PolyJuice with a maximum length of generated events of 40 and a temperature of 1.0. 

PolyJuice provided various control codes to manipulate events in various manners. The full list of control codes is negation, lexical, resemantic, quantifier, insert, 

|**Models**|**Best**_ϵ_|
|---|---|
|||
|Ours(BERT-large)<br>Ours(RoBERTa-large)<br>Ours(DeBERTa-large)|0.006<br>0.001<br>0.014|



Table 6: The best _ϵ_ for our framework **COLA** with the temporal predictor based on different language models. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0014-12.png)


**----- Start of picture text -----**<br>
0 . 68<br>0 . 66<br>0 . 64<br>ϵ<br>0 . 001 0 . 01 0 . 025<br>0 . 003 0 . 02<br>0 . 62<br>10 15 20 25 30 35 40<br>Covariate Set Size N<br>Accuracy<br>Zero-Shot<br>**----- End of picture text -----**<br>


Figure 4: Accuracy of **COLA** (“Ours”) with different covariate set size _N_ . Increasing covariate set size improves performance with a reasonable _ϵ_ , such as 0.001, 0.003, and 0.01. 

restructure, shuffle, and delete. We only use resemantic, negation, lexical, quantifier, insert, delete to generate interventions since restructure and shuffle do not generate counterfactual events. 

## **A.3 Temporal Propensity Threshold** _ϵ_ 

In this section, we list the best temporal propensity threshold _ϵ_ for our framework **COLA** with the temporal predictor based on different language models in Table 6. 

## **B Probability Approximation** 

Our approximation of the probability of a covariate P( _X_ (0)) in Section 5 is a consequence of the efficiency trade-off. Notice that even though a covariate X is an event preceding E _i_ sampled by GPTJ, it can occur before other events. Strictly, we must enumerate all possible events E, compute P(X _≺_ E), and marginalize out E. Here, we assume P( _E_ (1) _|X_ (0)) _≈_ 1 to compute P( _X_ (0)) efficiently. We approximately have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0014-19.png)


## **C Number of Covariates** 

We show that balancing covariates are essential in our framework **COLA** . Meanwhile, the number 

of covariates also has an impact on performance. Here we test different covariate set size _N_ = _|X|_ spanning from 10 to 40, shown in Figure 4 with various _ϵ_ values. 

We observe that gradually increasing the covariate set size _N_ can enhance the performance of our framework when the _ϵ_ is within [0.001, 0.01] as we recommend in Section 7.2. On the other hand, adding more covariates may introduce more noise if we test with a larger _ϵ_ . For example, when _ϵ_ equals 0.02 and 0.025, increasing covariates only brings about fluctuations in accuracy. 

## **D Dataset Annotation** 

Since our work is the first attempt to study Contextualized CCR, we carried out human annotation on Amazon Mechanical Turk to construct a new dataset. In this appendix, we discuss the annotation process of the dataset **COPES** in our paper. We randomly sampled event sequences from ROCStories (Mostafazadeh et al., 2016), where each sequence contains five chronologically ordered events. Workers are provided with event sequences and are asked to annotate whether an event causes the last event in a sequence[4] . Each human intelligence task (HIT) includes ten potential ( _cause_ , _effect_ ) event pairs (with corresponding event sequences), and each pair is labeled by seven workers. We take the majority vote among seven votes as the final result for each pair. 

We conduct two qualification tests to choose workers to maintain rigorous quality control. First, we invited annotators who meet the following conditions to take our qualification examinations: 1) an approval rate of at least 95% and 2) at least a thousand approved HITs. In the second round, a qualification question set including both effortless and tricky examples is collected by experts, namely the authors of this paper, who have a clear understanding of Contextualized CCR. The experts annotate 160 event pairs sampled from ROCStories. An annotator needs to answer a HIT involving ten questions from the qualification set, and the answers are compared with the experts’ answers. An annotator should correctly answer 8 out of 10 questions to pass the second round test. While 307 annotators participated in the second round qualification test, only 29 (9.45%) were selected as our main round annotators. 

Eventually, we collected the dataset containing 

4There may be multiple causes of an event. 

|**#Causal Relation**|**0**<br>**1**<br>**2**<br>**3**<br>**Total**|
|---|---|
|**#Seq**<br>12<br>192<br>124<br>12<br>340||
|||
|**Proportion (%)**|3.5<br>56.5<br>36.5<br>3.5<br>100|



Table 7: The breakdown of numbers of causal relations in all event sequences in the **COPES** dataset. Most event sequences (56.6%) own only one causal relation. 

human-annotated labels for 1,360 pairs from 340 event sequences (1360 = 340 _×_ 4), called _Choice of Plausible Event in Sequence_ ( **COPES** ). The IAA score is 61.13% calculated using pairwise agreement proportion, and Fleiss’s _κ_ (Fleiss, 1971) is 0.52. We equally divide them into a validation set and a testing set (e.g., each contains 680 examples). We also provide a screenshot of our annotation platform in Figure 5. 

## **D.1 Dataset Statistics** 

Here we provide the breakdown of numbers of causal relations _k_ in all 340 event sequences in Table 7. Also, 476 event pairs are positive (with causal relation), and 884 are negative (without causal relation). 

## **E Normalization** 

In this section, we enumerate some methods to normalize estimand ∆[ˆ] and temporal propensity vectors in Equation (7). 

**Direct Matching (D):** Instead of condition probability in Equation (9), we can directly use the vectors of temporal relation scores ( _f_ (A _,_ X))X _∈X_ as propensity vectors. 

**Score Simplification (S):** We fine-tune temporal predictors to prompt before, after, and [none], which might be difficult for smaller Masked LMs, like BERT-base. When constructing propensity vectors, we simplify the task to consider only before, after by normalizing scores: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0015-16.png)


**Propensity Covariate Normalization (Q):** We also try to normalize temporal relation scores on the covariate set _X_ before building propensity vectors: P( _X_ ) = _f_ (X _,_ E _i_ ) _/_[�] X _[′] ∈X[f]_[(][X] _[′][,]_[ E] _[i]_[)][ and] P( _X, A_ ) = _f_ (X _,_ A) _/_[�] X _[′] ∈X[f]_[(][X] _[′][,]_[ A][)][.] 

**Co-occurrence Normalization (C):** The finetuned temporal predictor may sometimes still faces 

|**Models**|**Acc**|**F1**|**Ma-F1**|
|---|---|---|---|
|Ours(BERT-base)|68.82|55.27|65.67|
|_⋄_w/o S|_↓_0.59|_↓_0.84|_↓_0.65|
|_⋄_w/o Q|_↓_3.24|_↓_4.64|_↓_3.56|
|_⋄_w/o E|_↓_5.29|_↓_7.59|_↓_5.83|
|Ours(BERT-large)|70.29|57.38|67.29|
|_⋄_w/o Q|_↓_0.59|_↓_0.84|_↓_0.65|
|_⋄_w/o E|_↓_7.06|_↓_10.13|_↓_7.77|



Table 8: Ablation study of normalization methods on BERT-base and BERT-large. 

|**Models**|**Acc**<br>**F1**<br>**Ma-F1**|
|---|---|
|||
|Ours(RoBERTa-base)|66.76<br>52.32<br>63.41|
|||
|_⋄_w/o C<br>_↓_1.47<br>_↓_2.11<br>_↓_1.62||
|||
|Ours(RoBERTa-large)|70.00<br>56.96<br>66.97|
|||
|_⋄_w/o S<br>_⋄_w/o E|_↓_0.59<br>_↓_0.84<br>_↓_0.65<br>_↓_2.06<br>_↓_2.95<br>_↓_2.27|



|**Models**|**Acc**<br>**F1**<br>**Ma-F1**|
|---|---|
|||
|Ours(DeBERTa-base)|70.29<br>57.38<br>67.29|
|||
|_⋄_w/o Q<br>_↓_0.59<br>_↓_0.84<br>_↓_0.65||
|||
|Ours(DeBERTa-large)<br>69.41<br>56.12<br>66.32||
|_⋄_w/o E|_↓_1.18<br>_↓_1.69<br>_↓_1.30|



Table 10: Ablation study of normalization methods on DeBERTa-base and DeBERTa-large. 

2. ChatGPT’s answer is different from the ground truth answer, but the correctness is debatable with the explanation from ChatGPT 

3. ChatGPT’s answer is incorrect. 

## **Example 1.** Which event mainly/directly causes the last event? 

E1: Charlie went to food truck event at the beach. 

E2: He said they had a ton of trucks there. 

Table 9: Ablation study of normalization methods on RoBERTa-base and RoBERTa-large. 

E3: He tried food from a few trucks. 

E4: He said the food was great. 

E5: I am going to try to go next year. 

the problem of low coverage, causing estimates _f_ (E _i,_ E _n_ ) and _f_ (A _,_ E _n_ ) inaccurate in Equation (7). We set them to ( _f_ (E _i,_ E _n_ ) + _f_ (E _n,_ E _i_ )) _/_ 2 and ( _f_ (A _,_ E _n_ ) + _f_ (E _n,_ A)) _/_ 2, respectively. 

**Estimand Normalization (E):** In this method, estimates _f_ (E _i,_ E _n_ ) and _f_ (A _,_ E _n_ ) in estimand ∆[ˆ] in Equation (7) are normalized by _f_ (E _i,_ E _n_ ) + _f_ (E _n,_ E _i_ ) and _f_ (A _,_ E _n_ ) + _f_ (E _n,_ A), respectively. 

We also conduct comprehensive analyses about removing these normalizations. The results are shown in Tables 8 to 10. We present drops in every metric when removing each normalization method. Some normalization methods cannot improve performance, and we delete those rows from tables. We observe that some of these normalization methods, such as **Estimand Normalization (E)** , can 

—————————— 

## **Gold answer** : E4 

## **ChatGPT answer (short)** : E4 

**ChatGPT answer (original)** : The event "He said the food was great." directly causes the last event "I am going to try to go next year." as it expresses the protagonist’s positive experience and desire to attend again in the future. 

## **Example 2.** Which event mainly/directly causes the last event? 

E1: Ericka enjoyed going to restaurants, but she liked to save money too. 

E2: She found a coupon for her favorite restaurant. 

E3: She went there for lunch and ordered a burger. 

E4: When the bill came, she was nervous about the coupon. E5: The restaurant accepted the coupon and Ericka saved a lot of money! 

—————————— 

## **Gold answer** : E2 _,_ E3 

## **F ChatGPT** 

In this section, we show and discuss the result of ChatGPT on **COPES** . Given the insignificant differences in performance between the prompts with and without the number _k_ in Section 7.4, we focus on the answers of the prompt with _k_ . We show three examples, respectively corresponding to three cases: 

1. ChatGPT’s answer is correct (with a good explanation) 

## **ChatGPT answer (short)** : E2 _,_ E4 

**ChatGPT answer (original)** : The second and fourth events mainly/directly cause the last event. The second event ("She found a coupon for her favorite restaurant.") sets up the possibility of her saving money, and the fourth event ("When the bill came, she was nervous about the coupon.") establishes that the coupon is being used in this specific instance. Together, these two events lead to the final event ("The restaurant accepted the coupon and ericka saved a lot of money!") where she is able to save money. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective_images/acl_2023_COLA_Contextualized_Commonsense_Causal_Reasoning_from_the_Causal_Inference_Perspective.pdf-0017-00.png)


Figure 5: A screenshot of our annotation platform. 

**Example 3.** Which event mainly/directly causes the last event? 

- E1: We decided to get married on the beach. 

cases where ChatGPT fails to give a persuasive answer. However, as a foundation model instead of a task model, ChatGPT’s performance is fair enough. 

- E2: Our family and friends met us there. 

- E3: It was a perfectly sunny summer day. 

- E4: Best of all, no one had to wear shoes!. 

- E5: We ended our day with a dip in the cool ocean. 

- —————————— 

## **Gold answer** : E1 

## **ChatGPT answer (short)** : E4 

**ChatGPT answer (original)** : The fourth event "Best of all, no one had to wear shoes!" mainly/directly causes the last event "We ended our day with a dip in the cool ocean." as it describes the condition that allowed the last event to happen: people were barefoot. 

In most cases, ChatGPT clearly explains its choices. For example, in the **Example 1** , it points out the positive **emotion** _“... food was great.”_ as the underlying fact that causes the last event _“... going to try to go next year.”_ (i.e., **decision** ). Similarly in the **Example 2** , even though ChatGPT’s answer is different from the ground truth answer, ChatGPT mentions two key points _“... The second event ... sets up the possibility of ...”_ (i.e., **potential** ) and _“.. the forth event ... establishes that the coupon is being used ...”_ (i.e., **action** ) that together causes the last event (i.e., **effect** ). The reasoning paradigm implicit in its explanation, e.g **emotion** _→_ **decision** and **potential + action** _→_ **effect** in the abovementioned two examples, implies the inherent reasoning capability of ChatGPT. Nonetheless, there are 

