# **A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models** 

**Alessandro Stolfo** _[∗]_ ETH Zürich 

stolfoa@ethz.ch 

## **Zhijing Jin** _[∗]_ MPI & ETH Zürich jinzhi@ethz.ch 

**Kumar Shridhar** ETH Zürich shkumar@ethz.ch 

**Bernhard Schölkopf Mrinmaya Sachan** MPI & ETH Zürich ETH Zürich bs@tue.mpg.de msachan@ethz.ch 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0001-07.png)


**----- Start of picture text -----**<br>
Distribution of the Predicted Numerical Answer<br>P(R) Pred = 143 (incorrect)<br>P(286)=0.001 After  do -intervention<br>Pred = 286 = g<br>P(286)=0.085 Original prediction<br>Prediction<br>LLMs<br>Kyle could fit n1=2 drawings on each page. If he has n2=143<br>pages, the number of drawings he can make is ___.<br>Example  do -intervention  Keep ground-truth g,<br>by our framework: but change n1, n2<br>Original text: Kyle could fit n1=26 drawings on each page. If he has n2=11<br>pages, the number of drawings he can make is ___.<br>**----- End of picture text -----**<br>


## **Abstract** 

We have recently witnessed a number of impressive results on hard mathematical reasoning problems with language models. At the same time, the robustness of these models has also been called into question; recent works have shown that models can rely on shallow patterns in the problem description when generating a solution. Building on the idea of behavioral testing, we propose a novel framework, which pins down the causal effect of various factors in the input, e.g., the surface form of the problem text, the operands, and math operators on the output solution. By grounding the behavioral analysis in a causal graph describing an intuitive reasoning process, we study the behavior of language models in terms of robustness and sensitivity to direct interventions in the input space. We apply our framework on a test bed of math word problems. Our analysis shows that robustness does not appear to continuously improve as a function of size, but the GPT-3 Davinci models (175B) achieve a dramatic improvement in both robustness and sensitivity compared to all other GPT variants.[1] 

Figure 1: Through our framework, we conduct dointerventions on the input and evaluate the change in the distribution P( _R_ ) of the prediction _R_ by LLMs, in this figure, GPT-J. This allows us to measure the causal effect of each factor in the input on the model’s response. 

Ouyang et al., 2022; Chowdhery et al., 2022), it remains unclear whether these models have learned mere artifacts in the data or have truly mastered the mathematical concepts needed to consistently solve all variations of the same problem (Patel et al., 2021; Razeghi et al., 2022; Welleck et al., 2022). In sharp contrast with a large number of papers on improving the performance of LLMs on various types of math-based problems, there has been little effort on behavioral analysis of LLMs for these tasks. Existing methods for understanding the robustness of these models (Patel et al., 2021) rely on manually constructing variations of math problems, and we do not yet have a principled, comprehensive framework for quantifying such robustness. 

## **1 Introduction** 

Many natural language understanding situations, such as understanding the financial news, require reasoning with text that includes numbers. However, such mathematical reasoning is challenging for NLP models (Cobbe et al., 2021; Mishra et al., 2022b). Mathematical reasoning for text has been an active area of research for a while (Seo et al., 2015; Sachan and Xing, 2017; Sachan et al., 2017, 2018, _inter alia_ ), and has also emerged as a key task to track the capabilities of large language models (LLMs) in recent years (Brown et al., 2020; Ouyang et al., 2022; Wei et al., 2022a, _inter alia_ ). 

Thus, in this work, we propose a formal framework based on causal inference, to quantify the robustness of NLP models’ math reasoning abilities. Specifically, we describe a causal graph formulation of math reasoning, where the graph allows us to measure the difference in the structural causal 

However, despite the impressive performance of LLMs on various math reasoning benchmarks (e.g., 

> _∗_ Equal contribution. 

> 1Our code and data are available at https://github. com/alestolfo/causal-math. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Intervention<br>Operands<br>N  = ( N 1,  N 2, …) TCE( N  on  R )<br>Math Word Problem  Q Gh Ground-truth Reasoning: Causal Graph of Human/ Correct Calculation  G  :=  f O ( N ) DCE( G  → R ) Model’s Prediction  R<br>Non-Operand Parts  Operations<br>T O<br>TCE( T  on  R ) Red Arrows:  Potential Spurious Correlations<br>Irrelevant Surface Form  Blue Arrow:  Desired Effect<br>S<br>Intervention<br>DCE( )<br>O  → R<br>DCE(<br>)<br>N  → R<br>R )<br>DCE( S  →<br>**----- End of picture text -----**<br>


Figure 2: Causal graph of model predictions on math questions. We highlight the difference between a cognitivelyinspired correct reasoning path ( _Gh_ ) and the undesired effects that some factors might have on the model’s prediction (red arrows). By performing controlled interventions of the numerical values ( _**N**_ ) and on the textual framing of the problem ( _**T**_ , _S_ ), we are able to quantify the causal effects of each factor. 

models of human reasoning and model judgment. We consider various causal factors such as the textual framing of the question, numerical operands, and operation types. Then, we identify a set of interventions in the context of math word problems (an example of which is illustrated in Figure 1), and provide a causal inference framework to obtain causal effects of each factor via direct dointerventions (Pearl, 1995) and causal mediation analysis (Pearl, 2001). While our approach is reminiscent of recent studies using causal analysis for LLMs (Finlayson et al., 2021; Vig et al., 2020; Meng et al., 2022), in this work, we provide a new theoretical analysis framework specifically suitable for math reasoning. Using our framework, we disentangle factors affecting the model’s predictions and measure their influences. This way, we are able to provide insights into the model’s reasoning in terms of _robustness_ and _sensitivity_ with respect to changes in these factors. 

We apply our framework to study a set of thirteen GPT models with various sizes and training procedures (i.e., instruction-tuned and non-instructiontuned). We observe that, among non-instructiontuned language models, the larger ones tend to be more sensitive to changes in the ground-truth result of a math word problem, but not necessarily more robust. However, we observe a different behavior in the instruction-tuned GPT-3 models (Ouyang et al., 2022), which show a remarkable improvement in both sensitivity and robustness, although the robustness reduces when problems get more complicated. We additionally investigate the role of size and instruction tuning on the model’s performance with three models of the LLaMA family 

(Touvron et al., 2023) and Stanford Alpaca (Taori et al., 2023). 

## **2 Problem Setup** 

We consider a dataset _D_ of math word problems (MWPs), where each MWP is denoted as a question _**Q**_ . _**Q**_ is a list ( _**T** ,_ _**N**_ ) consisting of a question template _**T**_ and an ordered list of operands _**N**_ = ( _N_ 1 _, N_ 2 _, . . . , Nm_ ). Each question template _**T**_ := ( _**O** , S_ ) further contains two types of information: a set of arithmetic operations _**O**_ implicitly expressed in the question, and the text surface form _S_ irrelevant to the arithmetic operations. _**O**_ incorporates the information relative to the operations as a collection of tuples _{_ ( _O_ 1 _, i_ 1 _, j_ 1) _,_ ( _O_ 2 _, i_ 2 _, j_ 2) _, . . . }_ , where _Ok ∈{_ + _, −, ×, ÷}_ ( _k ∈_ N) and _ik, jk ∈_ N represent the indices of the operands to which operator _Ok_ should be applied to.[2] The ground-truth result _G_ = _f_ _**O**_ ( _**N**_ ) is calculated by computing the function _f_ _**O**_ , which represents the application of all the operators in _**O**_ to the respective operands. We illustrate the factors in _**Q**_ and their inter-dependency in the causal graph in Figure 2. A two-operand instance _**q**_ of _**Q**_ in this form from Patel et al. (2021) is: 

**Template** _**t**_ : Mark has _n_ 1 trees in his backyard. If he plants _n_ 2 more, how many trees will he have? **Operands** _**n**_ : ( _n_ 1 = 12 _, n_ 2 = 13) **Operations** _**o**_ : {(“+”, 1, 2)} **Result** : _g_ = _f_ _**o**_ ( _**n**_ ) = _n_ 1 + _n_ 2 = 25 

> 2The intermediate result of operation _Ol_ is indicated by _ik_ = _m_ + _l_ . 

Our goal is to quantify the robustness of a model _M_ on the set of problems _**q** ∈D_ . Ideally, _D_ should be a dataset not seen by the model during training. We assume that a model takes _**q**_ as input and predicts a probability distribution of the result _R_ : P( _R |_ _**t** ,_ _**n**_ ). Our formulation below will be easier to understand using this finite discrete set and can be generalized to any kind of data pairing a natural language template with a function that maps a set of operands to a result (e.g., a Python program; Mishra et al. 2022a). 

## **3 A Causal Framework** 

In this section, we describe our framework in three steps. First, we define the idea of model robustness on MWPs. Then, we identify possible dointerventions (Pearl, 1995) that we can perform. Finally, we describe the causal effects that we measure to quantify the robustness of various models. 

## **3.1 Step 1. Question Reformulation** 

We address the research question “ _Is a model reasoning robustly on MWPs?_ ” by comparing the causal mechanisms of the model’s decisions to a hypothesized human reasoning mechanism. Note that we do not claim to know how humans reason about these problems. We simply propose a reasonable and intuitive way to judge model robustness given a reasonable and intuitive human reasoning mechanism inspired by findings regarding the independence of language and mathematical reasoning in humans (Brannon, 2005; Monti et al., 2012). 

**Human Reasoning Mechanisms.** The causal mechanisms of how humans might solve _**q**_ include 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0003-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0003-07.png)


where they first abstract the arithmetic operations _**o**_ from the problem _**q**_ by some cognitive process _f_ abstract, and then apply the operation to the operands to obtain the result _g_ . We show these mechanisms in the green subgraph _Gh_ of Figure 2. 

**Model Reasoning Mechanisms.** In contrast, the causal mechanisms of how a model might solve _**q**_ are as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0003-10.png)


where we are unsure about (1) _what_ part(s) of _**t**_ the model takes into account, and (2) _how_ it operates over the relevant variables. 

Thus, we draw all possible causal mechanisms that might take place in the black-box model _f_ blackBox in the complete causal graph in Figure 2. Some possible fine-grained causal mechanisms are 

1. The model might attend over the question template _**t**_ in two ways: paying attention to the text surface form _s_ via the causal path _**T** → S → R_ , or text relevant to the math operations _**o**_ via the causal path _**T** →_ _**O** → R_ . 

2. The model might also attend to the operands _**n**_ := ( _n_ 1 _, n_ 2 _, . . ._ ) via a causal path _**N** → R_ . 

3. If the model learns the correct causal mechanisms as in the human cognitive process, it should capture how the operator and the operands matter to the ground-truth result _g_ (via _**O** → G_ and _**N** → G_ ) and then the model prediction should be sensitive to any changes in the ground truth, namely _G → R_ . No spurious correlations can directly affect _R_ without going through the mediator _G_ . 

Hence, to answer the question “How robust is the mathematical reasoning of a model on MWPs?” we can answer the following subquestions: 

1. How does _R_ change in response to _G_ ? By quantifying this, we assess the _sensitivity_ (correct responsiveness) of the model to changes in the problem. In other words, does the model correctly adjust its prediction in response to a change in the correct solution of the problem? 

2. What is the (unwanted) direct causal effect size of _S → R_ , and _**N** → R_ ? We see the quantities as a measure of the _brittleness_ (i.e., wrong responsiveness) of the model to resultpreserving changes in the input. The lower the direct causal effect of _S_ and _**N**_ , the more _robust_ the model is. 

## **3.2 Step 2. Causal Intervention List** 

After formulating the cognitively-inspired subgraph _Gh_ and defining the undesired causal paths in Figure 2, we list all feasible limited actions that allow us to perform our causal analysis. In the context of MWPs, we use the following interventions: 

1. Direct intervention on all possible _n_ 1 _, n_ 2 _, . . ._ ; 

2. Partially controllable interventions on _**T**_ . We can replace the template _**T**_ in two ways: 

(a) both _S_ and _**O**_ are affected, or 

- (b) _S_ is affected but _**O**_ is not affected. 

## **3.3 Step 3. Turning Limited Actions into Causal Effect Sizes** 

Next, we explain how we can obtain the causal effect sizes we want (listed in Step 1) from the limited set of interventions we can do (listed in Step 2). Specifically, we first start from all the feasible interventions, and for variables that we cannot directly intervene on, we apply deductions from do-calculus (Pearl, 1995) to obtain or approximate the direct causal effect sizes. In the following, we describe a list of causal effect sizes that we need. 

**General Formulation.** Let us consider an intervention do( _X_ : _x → x[′]_ ), where _X ∈{_ _**T** , S,_ _**N** }_ and a problem _**Q**_ = _{_ _**T** ,_ _**N** }_ . The support of the numerical values _Ni_ ’s and _R_ is _I ⊆_ N, and we consider _**N**_ to be distributed uniformly over the set _{_ _**n** ∈I_[2] _| f_ _**O**_ ( _**n**_ ) _∈I}_ . We denote the distribution before the intervention P( _R |_ _**T** ,_ _**N**_ ) as _P_ and the distribution after the intervention as _P[′]_ . 

Following the distributional definition of causal effect by Pearl (1995), we quantify the effect of factor _X_ in our causal graph using a distance metric _δ_ between the distributions _P_ and _P[′]_ . That is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0004-06.png)


where CE can refer to the **total causal effect** (TCE, i.e., the joint effect through all the directed causal paths from a variable to another), or the **direct causal effect** (DCE, i.e., the effect from the directed causal path from a variable to another that does not go through any intermediate variables) (Pearl, 2001). We describe our choices for _δ_ in Section 3.4. 

**Causal Effects of the Operands.** When intervening on the operands _**N**_ := ( _N_ 1 _, N_ 2 _, . . ._ ), we can obtain the size of the total causal effect of _**N**_ on _R_ , namely 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0004-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0004-10.png)


Note that this TCE is not the exact desired quantity, because we want to separate two different paths of how _**N**_ affects _R_ : (1) the path _**N** → G → R_ , which is the correct decision path that we want the model to pick up (where the model reacts to 

the change in the ground-truth answer), and (2) the path _**N** → R_ , which is the spurious correlation that the model might have learned (where the model relies on some spurious correlations with certain numerical values, which could be traced to perhaps their frequencies in the training corpus). 

We can quantify the **direct causal effect** (DCE, i.e., the effect from the directed causal path from a variable to another that does not go through any intermediate variables) (Pearl, 2001) of _**N**_ on _R_ , namely the strength of the direct causal path _**N** → R_ , by controlling for _G_ to be fixed every time we intervene on _**N**_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0004-14.png)


For example, if we observe a model doing 100 + 100 = 200 correctly, we want to separate the math ability here into (1) the model’s sensitivity towards the ground-truth answer, and (2) the model’s decisions based on its familiarity with just the operand 100. Here, the overall effect is the calculable TCE( _**N**_ on _R_ ) by Eq. 5, and one of the subeffects is the calculable DCE( _**N** → R_ ) by Eq. 7. 

**Causal Effects of the Text Surface Form.** As for the operands, we can compute both the direct and indirect effects of the surface form representing the math problem. In particular, intervening on _**T**_ without controlling for _**O**_ (intervention 2a in Sec. 3.2), we can compute the total effect, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0004-17.png)


Controlling for the operations _**O**_ (intervention 2b in Sec. 3.2) will instead allow us to obtain the direct causal effect of the surface text: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0004-19.png)


Note that since there is no mediator between _S_ and _R_ , the DCE( _S → R_ ) is also TCE of _S_ on _R_ . The only adaptation that we need to make with regard to the MWPs is that it is not feasible to enumerate all possible perturbations of _S_ . Therefore, the practical results that researchers can achieve are over a certain subset of _S_ . In practice, we obtain this by intervening on _**T**_ without affecting _**O**_ . 

**Causal Effects of the Operators.** The ideal way to obtain the TCE of _**O**_ on _R_ is through some careful human annotation that minimally changes the templates as Kaushik et al. (2020) do for sentiment classification. The challenge for MWPs in our case is that with all our possible interventions, we cannot _only_ intervene on _**O**_ without introducing changes to the irrelevant surface form. However, we might get some information about TCE( _**O**_ on _R_ ) because, on the causal graph, the total causal influence of _**T**_ on _R_ actually flows into two directed paths, one through _S_ to _R_ (which is the DCE( _S → R_ )), and the other from _**O**_ to _R_ , which is our interested quantity TCE( _**O**_ on _R_ ). Therefore, we compare the two quantities we know, TCE( _**T** → R_ ) and DCE( _S → R_ ), to get a sense of the causal influence of _**O**_ on _R_ that we cannot obtain in any other way. 

## **3.4 Step 4. Quantifying the Causal Influence** 

Consider a realization of problem _**Q**_ with operands _**n**_ and ground-truth result _g_ = _f_ _**o**_ ( _**n**_ ), and denote by _g[′]_ the result after the intervention do( _X_ : _x → x[′]_ ). We quantify the causal effect of factor _X_ on the model’s prediction _R_ in two ways: by assessing the change in the predicted result, and by measuring the change in the probability assigned by the model to the correct result _g_ (or _g[′]_ ). 

**Change in the Prediction.** To account for the inability of LMs to capture the continuous property of numbers (Jin et al., 2021a), we measure the change in the model’s prediction using an indicator of the “change result” event: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0005-04.png)


where _r_ = arg max _x∈I P_ ( _x_ ), and _r[′]_ = arg max _x∈I P[′]_ ( _x_ ). 

**Relative Change in Confidence.** Inspired by Finlayson et al. (2021), we also highlight the change in terms of the relative difference in the probability assigned to _g_ and _g[′]_ . We formulate two types of relative change, one quantifying the relative change in the confidence of _g_ , and the other quantifying the relative change in the confidence of _g[′]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0005-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0005-08.png)


We quantify the overall relative change in confidence (RCC) as the average of the two relative 

changes above: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0005-11.png)


**A Unified Form.** We are interested in the average causal effect of the intervention across all problems in _D_ . Thus, we measure the average of the effects over all instances _**q** ∈D_ . We denote by the subscripts TCEcp/DCEcp and TCErcc/DCErcc the causal effects computed using the change in prediction metric and the relative change in confidence, respectively. We describe how we construct the dataset _D_ in Section 4.2. 

## **4 Experimental Setup** 

In this section, we describe the data used to perform the interventions and to measure the causal effects. 

## **4.1 Datasets** 

For our analyses, we use instances of math word problems from three popular datasets: ASDiv-A (Miao et al., 2020), MAWPS (Koncel-Kedziorski et al., 2016), and SVAMP (Patel et al., 2021). The examples contained in these collections are pairs ( _**t** ,_ _**o**_ ) consisting of a question template _**t**_ with its annotated operations _**o**_ . Each of these pairs can be instantiated multiple times into problems _**q**_ = ( _**t** ,_ _**n**_ ) by filling the template with numerical values ( _n_ 1 _, n_ 2 _, . . ._ ) and computing the groundtruth result _g_ = _f_ _**o**_ ( _**n**_ ) (most problems involve two to three operands, i.e., _|_ _**n** | ∈{_ 2 _,_ 3 _}_ ). We select a set of 437 two-operand and 307 three-operand template-expression pairs that we use to generate pairs of prompts representing an intervention. More details about the prompt generation procedure are in Appendix A. We use ( _**t** ,_ _**n**_ ) to refer to an instantiated template that we use as a prompt. 

## **4.2 Intervention Data** 

Given an MWP _**q**_ = ( _**t** ,_ _**n**_ ) and its solution _g_ , we generate a second problem-solution instance ( _**q**[′] , g[′]_ ) depending on the type of causal effect CE we want to measure and on the considered variable. When intervening on the operands of the problem, the text of the problem is kept unaltered and a set of new operands _**n**_ is sampled in such a way that the result _g_ is affected or not depending on the effect that is being measured. When changing the textual description of the problem, we change _**t**_ such that either _**o**[′]_ = _**o**_ , or _**o**[′]_ = _**o**_ . In the former case, we sample a different template _**t**[′]_ = ( _s[′] ,_ _**o**_ ) from the 

set of templates describing the same operations _**o**_ , in the latter case we sample a new _**t**[′]_ describing a different operation. In Appendix B.1 we report some examples of ( _**q** ,_ _**q**[′]_ ) pairs representing the different types of interventions. 

Given a model, we use the question pair ( _**q** ,_ _**q**[′]_ ) to obtain a pair of answer distributions P( _R|_ _**t** ,_ _**n**_ ) and P( _R|_ _**t**[′] ,_ _**n**[′]_ ), which we use to measure the causal effect of the intervention. We consider the space for the numerical values to be _I_ = _{_ 1 _,_ 2 _, . . . , C}_ consisting of integer values, following the setup of several existing MWP datasets (Miao et al., 2020; Koncel-Kedziorski et al., 2016; Patel et al., 2021). To control our experimental costs and make sure the models keep the number as one token, we set _C_ = 300. From all the tokens in a model’s vocabulary, we focus on the probability assigned to the numbers in our numerical space _I_ , and thus we use P( _R_ = _r_ ) to denote the normalized probability Praw( _R_ = _r_ ) _/Z_ , where _Z_ =[�] _[C]_[=] _[r]_[)][, and][ P][raw][(] _[x]_[)][ is the raw] _r_ =1[P][raw][(] _[R]_ probability score assigned to the vocabulary token _x_ . For each intervention type, we generate a dataset _D_ consisting of ( _**q** ,_ _**q**[′]_ ) pairs. Unless otherwise specified, for our experiments we generate 500 intervention pairs for each template, and results are averaged over three seeds. 

## **4.3 Models to Evaluate** 

We use our framework to assess the robustness of reasoning in thirteen pre-trained language models. We consider five sizes of the GPT-2 model (Radford et al., 2019): distilled (Sanh et al., 2019), small, medium, large, and XL. We evaluate four models from EleutherAI that were pre-trained on the Pile (Gao et al., 2020): GPT-Neo 1.3B and 2.7B (Black et al., 2021), GPT-J-6B (Wang and Komatsuzaki, 2021), and GPT-NeoX-20B (Black et al., 2022). We use HuggingFace Transformers (Wolf et al., 2019) to access the models. Additionally, we experiment with a set of instruction-tuned versions of GPT-3 (Brown et al., 2020): Instruct (Ouyang et al., 2022), Curie, Davinci-002, and Davinci-003.[3] Experiments with GPT-3 are carried out under the constraints set by the OpenAI APIs[4] , which prevent us from computing the causal effect using the same procedure as for the other models. We report the details about how the metrics were computed 

> 3The OpenAI ids for these models are, respectively, davinci-instruct-beta, text-curie-001, text-davinci-002, and text-davinci-003. 4https://openai.com/api/ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0006-05.png)


**----- Start of picture text -----**<br>
DCErcc of  N TCErcc of  N<br>10 [5]<br>10 [4]<br>10 [3]<br>10 [2]<br>10 [1]<br>10 [0]<br>1 DCEcp of  N TCEcp of  N<br>0 . 8<br>0 . 6<br>0 . 4<br>0 . 2<br>DistilledSmallMediumLargeNeo-1.3BNeo-2.7BXL J-6BNeoX3-Curie*3-Davinci-002*3-Instruct*3-Davinci-003*<br>DistilledSmallMediumLargeNeo-1.3BNeo-2.7BXL J-6BNeoX3-Curie3-Davinci-0023-Instruct3-Davinci-003<br>rcc<br>δ<br>cp<br>δ<br>**----- End of picture text -----**<br>


Figure 3: Comparison of DCE( _**N** → R_ ) and TCE( _**N**_ on _R_ ). _[∗]_ approx values, see Appendix C. 

for GPT-3 in Appendix C. In the reported results, we indicate with an asterisk ( _[∗]_ ) the metrics that were influenced by this limitation. 

## **5 Results** 

Our analyses focus primarily on two-operand problems (Sections 5.1 and 5.2) and later extend to more complex problems that involve three operands (Section 5.5) for the models that perform best on the two-operand test bed. We compare the direct causal effect DCE and the total causal effect TCE of _**N**_ and _**T**_ on _R_ . DCE represents the undesired effect for a model to being mistakenly responsive to a change in _**N**_ or _**T**_ not leading to a change in the result _g_ (low robustness), whereas higher values of TCE indicate a higher ability of the model to correctly adjust the probability weight assigned to the new solution _g[′]_ after the intervention (high sensitivity). 

## **5.1 Effect of** _**N**_ **on** _R_ 

From the results in Figure 3, we notice that larger models exhibit a larger TCErcc/DCErcc ratio. In particular, in GPT-J-6B and NeoX, the TCE is, respectively, 30x and 1000x larger than the DCE. However, this improvement in sensitivity is not manifested in terms of change of prediction ( _δ_ cp), for which the models show to be affected by resultpreserving changes almost as equally as by resultaltering interventions. This behavior changes sig- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0007-00.png)


**----- Start of picture text -----**<br>
0.200 0.200 0.200<br>0.175 0.175 0.175<br>0.150 0.150 0.150<br>0.125 0.125 0.125<br>0.100 0.100 0.100<br>0.075 0.075 0.075<br>0.050 0.050 0.050<br>0.025 0.025 0.025<br>0.000 0.000 0.000<br>0 5 10 15 20 25 30 35 40 45 50 0 5 10 15 20 25 30 35 40 45 50 0 5 10 15 20 25 30 35 40 45 50<br>50 50 50<br>45 45 45<br>40 40 40<br>35 35 35<br>30 30 30<br>25 25 25<br>20 20 20<br>15 15 15<br>10 10 10<br>5 5 5<br>0 0 0<br>**----- End of picture text -----**<br>


Figure 4: Heatmaps displaying _P_ ( _g_ ) for Distil-GPT-2 (left), GPT-J-6B (center), and GPT-3 Davinci-002 (right). _g_ is the ground-truth result _g_ = _n_ 1 + _n_ 2 ( _n_ 1 and _n_ 2 are represented by the x and y axes, respectively. The probability values for each combination of (( _n_ 1 _, n_ 2) _, g_ ) are averaged over 20 different templates. Probability values over 0.2 are displayed with the darkest color. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0007-02.png)


**----- Start of picture text -----**<br>
DCErcc of  S TCErcc of  T<br>10 [5]<br>10 [4]<br>10 [3]<br>10 [2]<br>10 [1]<br>10 [0]<br>1 DCEcp of  S TCEcp of  T<br>0 . 8<br>0 . 6<br>0 . 4<br>0 . 2<br>DistilledSmallMediumLargeNeo-1.3BNeo-2.7BXL J-6BNeoX3-Curie*3-Davinci-002*3-Instruct*3-Davinci-003*<br>DistilledSmallMediumLargeNeo-1.3BNeo-2.7BXL J-6BNeoX3-Curie3-Davinci-0023-Instruct3-Davinci-003<br>rcc<br>δ<br>cp<br>δ<br>**----- End of picture text -----**<br>


Figure 5: Comparison of DCE( _S → R_ ) and TCE( _**T**_ on _R_ ). We use _[∗]_ to denote approximated values, explained in Appendix C. 

nificantly in instruction-tuned models. In particular, for the 175B-parameter GPT-3, performance varies depending on the type of supervision, with the PPOtrained Davinci-003 exhibiting an 84% difference between direct and total effect. 

In Figure 4, we present a different visualization of the direct causal effect of _**N**_ on the model’s prediction. We report the heatmaps showing the probability assigned by the model to the result _g_ of a problem ( _**t** ,_ ( _n_ 1 _, n_ 2) _, g_ ) _| g_ = _n_ 1 + _n_ 2 _, ∀g ∈ {_ 0 _,_ 1 _, . . . ,_ 50 _}, ∀_ ( _n_ 1 _, n_ 2) _∈{_ 0 _,_ 1 _, . . . ,_ 50 _}_[2] . For Distil-GPT-2 we observe low overall probability assigned to _g_ and diagonal patterns indicating consistency in assigning higher probability to specific results (e.g., 10, 20, 30, 40, 50). For the two larger models we notice a higher probability mass as- 

signed to the problem’s result, but less consistency on the prediction of the same result with different sets of operands (this is true for GPT-J in particular). This result is consistent with the observed higher DCE and TCE in larger models: _P_ ( _g_ ) might vary more considerably when intervening on _**N**_ without affecting _g_ , but overall the model assigns higher probability weight to the correct result, which correlates with higher sensitivity. 

## **5.2 Effect of** _**T**_ **on** _R_ 

In Figure 5, we report the total causal effect of the textual framing _**T**_ and the direct causal effect of the irrelevant text elements _S_ on the model’s prediction. For the instruction-tuned models, the improvement in terms of prediction change ( _δ_ cp) follows a similar trend as for _**N**_ , with GPT-3 Davinci-003 showing a 76% difference between direct and total effect. An interesting observation is that the irrelevant textual information _S_ appears to have a lower direct effect than _**N**_ for all noninstruction-tuned models. However, in the GPT-3 Davinci-00x models, we observe the opposite (i.e., DCE( _**N** → R_ ) _≤_ DCE( _S → R_ )). This suggests that large instruction-based models tend to be more susceptible to variation in the textual framing of a problem, while smaller models are more responsive to changes in the numerical values (though not necessarily correctly). 

## **5.3 Overall Insights** 

In comparison to other models, GPT-3 Davinci shows the highest DCErcc, but low DCEcp. This discrepancy is related to the quantities that the two metrics consider. _δ_ rcc takes into account the probability assigned to _g_ , while _δ_ cp does not consider the ground truth solution. One interpretation of this result is that GPT-3 Davinci consistently predicts the same answer _r_ = _r[′]_ when _g_ = _g[′]_ , but the probabilities _P_ ( _g_ ) and _P[′]_ ( _g_ ) might vary significantly. 

The results observed for the two kinds of intervention do( _**T**_ : _**t** →_ _**t**[′]_ ) and do( _**N**_ : ( _n_ 1 _, n_ 2) _→_ ( _n[′]_ 1 _[, n][′]_ 2[))][ show similar trends.][Small models (Dis-] tilled and Small GPT-2) exhibit low sensitivity to interventions. Larger models (from GPT-2 Medium to GPT-Neo) appear to be more influenced by changes in both _**N**_ and _**T**_ . However, they display similar sensitivity to both result-altering and resultpreserving interventions. An improvement in sensitivity is noticeable in GPT-J and NeoX, though not accompanied by an improvement in robustness. Remarkably different behavior is instead shown by the GPT-3 Davinci models, which demonstrate substantially higher sensitivity to result-altering interventions (high TCE), and higher robustness (in terms of prediction change). In Appendix B.2, we report the accuracy of the models on the generated instances of MWPs, which exhibits a similar trend as the robustness/sensitivity changes we observed. 

Possible explanations for the improved robustness and sensitivity demonstrated by the large GPT3 models might be the dramatic size increase and extension/enhancement of the training procedure involving instructions. The former idea is aligned with the _emergent abilities_ hypothesis (Wei et al., 2022a), which postulates the existence of skills that are displayed by large-scale models but are not present in smaller-scale models. However, our observations show different performances in versions of GPT-3 Davinci that differ in the training procedure.[5] This raises the question of whether the capability of LLMs to reason about math problems benefits from instruction-based tuning. We address this question in the following section. 

## **5.4 Extending to LLaMA-Based Models** 

To further investigate the roles played by size and training method in the model’s performance, we carry out our experimental procedure on three versions with different sizes (7B, 13B, and 30B) of the LLaMA model (Touvron et al., 2023), and on Stanford Alpaca (which applies instruction tuning on LLaMA 7B) (Taori et al., 2023). We present these results separately, as the LLaMA tokenization makes the prediction setup different from the one used from the other models, and prevents us from computing the relative change in confidence 

> 5A high-level description of the training procedures for the models is provided at https://beta.openai.com/ docs/model-index-for-researchers. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0008-05.png)


**----- Start of picture text -----**<br>
0 . 7 DCE of  N TCE of  N<br>0 . 6<br>0 . 5<br>0 . 4<br>0 . 3<br>0 . 2<br>LLaMA 7BLLaMA 13BLLaMA 30B Alpaca<br>cp<br>δ<br>**----- End of picture text -----**<br>


Figure 6: Comparison of direct and total effects of _N_ on _R_ for LLaMA and Alpaca. 

## ( _δ_ rcc).[6] 

From the results (Figure 6), two notable observations emerge. Firstly, the increased difference between TCE and DCE observed with the increasing size of the LLaMA models suggests that a larger number of parameters can be a significant driver behind robustness/sensitivity improvement. However, this is not necessarily the case across different models: GPT-NeoX-20B shows a smaller TCEcp-DCEcp gap compared to LLaMA 7B (5.2% vs 9.0%). Secondly, the instruction tuning procedure of Alpaca does not seem to help significantly with mathematical computation: the decrease in both TCE and DCE shows that robustness improves at the expense of sensitivity. Nonetheless, overall, when comparing Alpaca compared to its base model, LLaMA 7B, we observe an increase in the gap between TCE and DCE, although this difference is minimal (9.5% vs 9.0%). 

The limited improvement of Alpaca might be attributed to its instruction tuning procedure consisting of “a list of user-oriented instructions including email writing, social media, and productivity tools” (Taori et al., 2023), which differs from reasoningintensive tasks. We suggest future work to examine different types of instruction tuning (e.g., focused on reasoning procedures or reinforcement learning from human feedback), which might help the model answer more complex types of questions in a step-by-step manner and more accurately. We hypothesize that the different performances in versions of GPT-3 Davinci might be produced by the specific type of instructions used for training, by the reinforcement learning component (Ouyang et al., 2022), or simply by an extension of the language modeling pre-training. It is challenging to 

> 6The LLaMA tokenizer considers each digit as an independent token in the vocabulary. This makes it problematic to compare the probability value assigned by the model to multi-digit numbers. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0009-00.png)


**----- Start of picture text -----**<br>
DCE of  N TCE of  N<br>10 [4] 1<br>10 [3]<br>0 . 9<br>10 [2]<br>10 [1] 0 . 8<br>10 [0]<br>0 . 7<br>3-Instruct*3-Davinci-002*3-Davinci-003* 3-Instruct3-Davinci-0023-Davinci-003<br>rcc cp<br>δ δ<br>**----- End of picture text -----**<br>


Figure 7: Comparison of direct and total effects of _**N**_ on _R_ for three-operand problems. 

pinpoint the exact factor in the training procedure that contributes to this improvement, as specific methodological details are not available. 

## **5.5 Moving to Three-Operand Problems** 

We extend our evaluation to consider the threeoperand problems in the dataset. In these experiments, we consider only the GPT-3 175Bparameter models, as they are the only models performing well on the simpler bivariate problems. The results regarding the effects of _**N**_ are reported in Figure 7. We notice that the large difference between the desired (TCE) and undesired (DCE) effects observed on simpler problems shrinks significantly for both metrics. In particular, for Davinci003, the direct effect of _**N**_ (measured as _δ_ cp) grows from 0.17 to 0.87. That is, GPT-3 Davinci-003 predicts a different result 87% of the time after an intervention that does not affect the ground-truth solution. The increase in direct effect indicates a performance degradation in terms of brittleness: even the models that show good performance on two-operand problems, now display an unstable behavior after result-preserving interventions. 

## **6 Related Work** 

**Causal NLP.** Causal inference aims to study the cause and effect from observational and interventional data (Pearl, 2009; Peters et al., 2017). Traditionally, researchers usually apply causal techniques to phenomena in nature and human society. With the rise of powerful models in NLP, recent research has started to explore the intersection of causal inference and NLP, forming the study of Causal NLP (Jin et al., 2022; Feder et al., 2021a). 

There are several formulations for Causal NLP: the _causality for NLP_ thread involves using the causal framework for data collection and task for- 

mulation (Jin et al., 2021c), inspecting the (pathspecific) causal effect of certain neurons on predictions (Vig et al., 2020; Meng et al., 2022), understanding the causal effect of data and learning paradigm for model performance (Ni et al., 2022), and as a way to frame prompts (Lyu et al., 2023); and _NLP for causality_ involves testing the pure causal inference skills of LLMs (Jin et al., 2023a,b), and use text as a variable for causal effect estimation (Roberts et al., 2020; Veitch et al., 2020; Jin et al., 2021b, 2023c). 

The most similar line of research to our work is the application of causal effect estimation on interpreting models’ behavior, such as how models understand syntactic agreement (Finlayson et al., 2021), and how interventions in the representations and weights affect the model prediction (Feder et al., 2021b). To the best of our knowledge, our work is the first to formulate a causal framework for robustness behavioral tests, and also we are the first to introduce the idea to quantify the differences in the causal mechanisms of human reasoning and model decisions. 

**Math Reasoning in NLP.** A growing body of work tries to improve the math reasoning capability in NLP models (Zhang et al., 2020; Geva et al., 2020; Spokoyny et al., 2021), and prompting techniques for LLMs (Cobbe et al., 2021; Shen et al., 2021; Kojima et al., 2022; Wei et al., 2022b; Chowdhery et al., 2022). For analysis, significant attention has been given to models’ ability to understand numerical quantities (Wallace et al., 2019; Thawani et al., 2021) and numerical operations (Pal and Baral, 2021; Berg-Kirkpatrick and Spokoyny, 2020; Pi˛ekos et al., 2021; Razeghi et al., 2022). 

## **7 Conclusion** 

We developed a framework to disentangle and separately measure the effect of different factors influencing the predictions of LLMs for math reasoning. Our results indicate that a drastic increase in both robustness and sensitivity emerges in the GPT-3 Davinci models. Additionally, we study the contribution of size and instruction tuning in the models of the LLaMA family, observing that the Alpaca instruction tuning, while increasing the model’s robustness, does not significantly improve the overall performance. Our framework provides a formalized theory of behavioral testing for math reasoning models and opens new future directions to design behavioral tests of models in a principled way. 

## **Ethical Considerations** 

As for the ethical practice in this work, the data involved are from existing MWP datasets with no private user information, and available under the MIT license. As for the ethical impact of the use of this work, the study is about providing a metric and analyzing existing models’ robustness, so there is less concern over harmful usage. Rather, it is more about putting checks on existing AI models and helping humans understand them better before use. Potential stakeholders that could benefit from this research include NLP researchers working on math models, practitioners working on various applications involving mathematical reasoning with text, and e-learning design. 

## **Limitations** 

A key limitation in our work is that LLMs might have seen these math problems. Our work theoretically assumes this is not the case. Another limitation is that for the sake of simplicity, our work makes some assumptions. For example, we assume all numbers in the range of integers 0 to _C_ = 300. This would not cover every MWP out there. And future work is needed to generalize our framework to other forms of MWPs. In this work, we are also constrained by the limitations of the OpenAI policy on the GPT-3 API. This limits the number of perturbations we consider in this work as well as the accuracy with which we can estimate our causal distributions. Finally, our work is restricted to English, and extending it to other languages will require us to create an MWP dataset in that language. 

## **Acknowledgments** 

This material is based in part upon works supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039B; by the Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645; by the John Templeton Foundation (grant #61156); by a Responsible AI grant by the Haslerstiftung; and an ETH Grant (ETH-19 21-1). Alessandro Stolfo is supported by armasuisse Science and Technology through a CYD Doctoral Fellowship. Zhijing Jin is supported by PhD fellowships from the Future of Life Institute and Open Philanthropy, as well as the travel support from ELISE (GA no 951847) for the ELLIS pro- 

gram. We also thank OpenAI Researcher Access Program for granting our team credits to their API. 

## **References** 

- Taylor Berg-Kirkpatrick and Daniel Spokoyny. 2020. An empirical investigation of contextualized number prediction. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 4754–4764, Online. Association for Computational Linguistics. 9 

- Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. 2022. GPT-NeoX-20B: An open-source autoregressive language model. _arXiv preprint arXiv:2204.06745_ . 6 

- Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. GPT-Neo: Large scale autoregressive language modeling with mesh-tensorflow. If you use this software, please cite it using these metadata. 6 

- Elizabeth M. Brannon. 2005. The independence of language and mathematical reasoning. _Proceedings of the National Academy of Sciences_ , 102(9):3177– 3178. 3 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Advances in Neural Information Processing Systems_ , volume 33, pages 1877–1901. Curran Associates, Inc. 1, 6 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ . 1, 9 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ . 1, 9 

- Amir Feder, Katherine A. Keith, Emaad Manzoor, Reid Pryzant, Dhanya Sridhar, Zach Wood-Doughty, Jacob Eisenstein, Justin Grimmer, Roi Reichart, Margaret E. Roberts, Brando n M. Stewart, Victor Veitch, and Diyi Yang. 2021a. Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. _CoRR_ , abs/2109.00725. 9 

- Amir Feder, Nadav Oved, Uri Shalit, and Roi Reichart. 2021b. CausaLM: Causal model explanation through counterfactual language models. _Computational Linguistics_ , 47(2):333–386. 9 

- Matthew Finlayson, Aaron Mueller, Sebastian Gehrmann, Stuart Shieber, Tal Linzen, and Yonatan Belinkov. 2021. Causal analysis of syntactic agreement mechanisms in neural language models. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 1828–1843, Online. Association for Computational Linguistics. 2, 5, 9 

- Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020. The pile: An 800gb dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_ . 6 

- Mor Geva, Ankit Gupta, and Jonathan Berant. 2020. Injecting numerical reasoning skills into language models. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 946–958, Online. Association for Computational Linguistics. 9 

- Zhihua Jin, Xin Jiang, Xingbo Wang, Qun Liu, Yong Wang, Xiaozhe Ren, and Huamin Qu. 2021a. Numgpt: Improving numeracy ability of generative pre-trained models. _arXiv preprint arXiv:2109.03137_ . 5 

- Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, and Bernhard Schoelkopf. 2023a. Cladder: Assessing causal reasoning in language models. 9 

- Zhijing Jin, Amir Feder, and Kun Zhang. 2022. CausalNLP tutorial: An introduction to causality for natural language processing. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Tutorial Abstracts_ , pages 17– 22, Abu Dubai, UAE. Association for Computational Linguistics. 9 

- Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrinmaya Sachan, Rada Mihalcea, Mona Diab, and Bernhard Schoelkopf. 2023b. Can large language models infer causation from correlation? 9 

- Zhijing Jin, Zhiheng Lyu, Yiwen Ding, Mrinmaya Sachan, Kun Zhang, Rada Mihalcea, and Bernhard Schoelkopf. 2023c. AI Scholars: A dataset for NLPinvolved causal inference. 9 

- Zhijing Jin, Zeyu Peng, Tejas Vaidhya, Bernhard Schoelkopf, and Rada Mihalcea. 2021b. Mining the cause of political decision-making from social media: A case study of COVID-19 policies across the US 

   - states. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 288–301, Punta Cana, Dominican Republic. Association for Computational Linguistics. 9 

- Zhijing Jin, Julius von Kügelgen, Jingwei Ni, Tejas Vaidhya, Ayush Kaushal, Mrinmaya Sachan, and Bernhard Schoelkopf. 2021c. Causal direction of data collection matters: Implications of causal and anticausal learning for NLP. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 9499–9513, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 9 

- Divyansh Kaushik, Eduard H. Hovy, and Zachary Chase Lipton. 2020. Learning the difference that makes A difference with counterfactually-augmented data. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 5 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. _arXiv preprint arXiv:2205.11916_ . 9 

- Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. MAWPS: A math word problem repository. In _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 1152–1157, San Diego, California. Association for Computational Linguistics. 5, 6 

- Zhiheng Lyu, Zhijing Jin, Justus Mattern, Rada Mihalcea, Mrinmaya Sachan, and Bernhard Schölkopf. 2023. Psychologically-inspired causal prompts. _CoRR_ , abs/2305.01764. 9 

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. _Advances in Neural Information Processing Systems_ , 35:17359–17372. 2, 9 

- Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing English math word problem solvers. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 975–984, Online. Association for Computational Linguistics. 5, 6 

- Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, et al. 2022a. Lila: A unified benchmark for mathematical reasoning. _arXiv preprint arXiv:2210.17517_ . 3 

- Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. 2022b. NumGLUE: A suite of fundamental yet challenging mathematical reasoning tasks. In _Proceedings of the 60th Annual Meeting of_ 

_the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3505–3523, Dublin, Ireland. Association for Computational Linguistics. 1 

- Martin M Monti, Lawrence M Parsons, and Daniel N Osherson. 2012. Thought beyond language: Neural dissociation of algebra and natural language. _Psychological science_ , 23(8):914–922. 3 

- Jingwei Ni, Zhijing Jin, Markus Freitag, Mrinmaya Sachan, and Bernhard Schölkopf. 2022. Original or translated? A causal analysis of the impact of translationese on machine translation performance. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 5303–5320, Seattle, United States. Association for Computational Linguistics. 9 

- Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. _CoRR_ , abs/2203.02155. 1, 2, 6, 8 

- Kuntal Kumar Pal and Chitta Baral. 2021. Investigating numeracy learning ability of a text-to-text transfer model. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 3095–3101, Punta Cana, Dominican Republic. Association for Computational Linguistics. 9 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2080–2094, Online. Association for Computational Linguistics. 1, 2, 5, 6, 14 

- Judea Pearl. 1995. Causal diagrams for empirical research. _Biometrika_ , 82(4):669–688. 2, 3, 4 

- Judea Pearl. 2001. Direct and indirect effects. In _UAI ’01: Proceedings of the 17th Conference in Uncertainty in Artificial Intelligence, University of Washington, Seattle, Washington, USA, August 2-5, 2001_ , pages 411–420. Morgan Kaufmann. 2, 4 

- Judea Pearl. 2009. _Causality_ . Cambridge University Press. 9 

- Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. 2017. _Elements of causal inference: Foundations and learning algorithms_ . The MIT Press. 9 

- Piotr Pi˛ekos, Mateusz Malinowski, and Henryk Michalewski. 2021. Measuring and improving BERT’s mathematical abilities by predicting the order of reasoning. In _Proceedings of the 59th Annual Meeting of the Association for Computational_ 

   - _Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)_ , pages 383–394, Online. Association for Computational Linguistics. 9 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. 6 

- Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. 2022. Impact of pretraining term frequencies on few-shot reasoning. _arXiv preprint arXiv:2202.07206_ . 1, 9 

- Margaret E Roberts, Brandon M Stewart, and Richard A Nielsen. 2020. Adjusting for confounding with text matching. _American Journal of Political Science_ , 64(4):887–903. 9 

- Mrinmaya Sachan, Kumar Dubey, and Eric Xing. 2017. From textbooks to knowledge: A case study in harvesting axiomatic knowledge from textbooks to solve geometry problems. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 773–784. 1 

- Mrinmaya Sachan, Kumar Avinava Dubey, Tom M Mitchell, Dan Roth, and Eric P Xing. 2018. Learning pipelines with limited data and domain knowledge: A study in parsing physics problems. _Advances in Neural Information Processing Systems_ , 31. 1 

- Mrinmaya Sachan and Eric Xing. 2017. Learning to solve geometry problems from natural language demonstrations in textbooks. In _Proceedings of the 6th Joint Conference on Lexical and Computational Semantics (* SEM 2017)_ , pages 251–261. 1 

- Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. In _NeurIPS EMC_[2] _Workshop_ . 6 

- Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. 2015. Solving geometry problems: Combining text and diagram interpretation. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pages 1466–1476, Lisbon, Portugal. Association for Computational Linguistics. 1 

- Jianhao Shen, Yichun Yin, Lin Li, Lifeng Shang, Xin Jiang, Ming Zhang, and Qun Liu. 2021. Generate & rank: A multi-task framework for math word problems. _arXiv preprint arXiv:2109.03034_ . 9 

- Daniel Spokoyny, Ivan Lee, Zhao Jin, and Taylor BergKirkpatrick. 2021. Masked measurement prediction: Learning to jointly predict quantities and units from textual context. _arXiv preprint arXiv:2112.08616_ . 9 

- Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca. 2, 8 

- Avijit Thawani, Jay Pujara, Filip Ilievski, and Pedro Szekely. 2021. Representing numbers in NLP: a survey and a vision. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 644–656, Online. Association for Computational Linguistics. 9 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ . 2, 8 

- Victor Veitch, Dhanya Sridhar, and David M. Blei. 2020. Adapting text embeddings for causal inference. In _Proceedings of the Thirty-Sixth Conference on Uncertainty in Artificial Intelligence, UAI 2020, virtual online, August 3-6, 2020_ , volume 124 of _Proceedings of Machine Learning Research_ , pages 919–928. AUAI Press. 9 

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. 2020. Investigating gender bias in language models using causal mediation analysis. _Advances in Neural Information Processing Systems_ , 33:12388– 12401. 2, 9 

- Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, and Matt Gardner. 2019. Do NLP models know numbers? probing numeracy in embeddings. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 5307–5315, Hong Kong, China. Association for Computational Linguistics. 9 

- Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 billion parameter autoregressive language model. 6 

- Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022a. Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_ . 1, 8 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022b. Chain of thought prompting elicits reasoning in large language models. _CoRR_ , abs/2201.11903. 9 

- Sean Welleck, Peter West, Jize Cao, and Yejin Choi. 2022. Symbolic brittleness in sequence models: on systematic generalization in symbolic mathematics. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pages 8629–8637. 1 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R’emi Louf, Morgan Funtowicz, and Jamie Brew. 2019. HuggingFace’s transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_ . 6 

- Xikun Zhang, Deepak Ramachandran, Ian Tenney, Yanai Elazar, and Dan Roth. 2020. Do language embeddings capture scales? In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 4889–4896, Online. Association for Computational Linguistics. 9 

## **A Creation of the Prompts** 

We consider MWP examples from the union of the three datasets SVAMP, ASDiv-A, and MAWPS. The textual template _**t**_ of a problem consists of a context (describing a real-world state and/or actions) and a question. In order to obtain suitable prompts for the models, we convert the problems’ questions into statements where the result of the problem is expected to be the first token after the prompt. E.g., in the example in section 2, _how many trees will he have?_ is converted into _the number of trees that he will have is __ . From the MWP templates of the SVAMP/ASDiv-A/MAWPS collection (we consider all splits), we filter out the templates whose questions do not start with _How many..._ , and we use spaCy[7] to identify the subject, the object and the verbs in the sentence. This allows us to convert the last sentence of the template from _The number of... is_ . This way, we obtain 437 statement-based MWP templates for two-operand problems and 307 for three-operand problems. We manually checked a subset of the templates to identify possible mistakes in the conversion procedure. 

## **B Frequently Asked Questions** 

## **B.1 How do the intervention data look like?** 

In Table 1 we report examples of MWP pairs representing different types of intervention. 

## **B.2 What is the accuracy of the evaluated models on the generated problems?** 

We report the accuracy of the models considered for evaluation in terms of accuracy at 1 and accuracy at 10. Results are displayed in Figure 8. 

## **B.3 What is the relation between accuracy and the RCC metric?** 

We examine the relationship between performance and robustness, computing the Pearson correlation coefficient between accuracy (accuracy@10) and the relative confidence change (RCC) metric. On a per-template basis (500 instances for each template), we found accuracy to be positively 

7https://spacy.io 

|TCE(**_N_** _→R_)|Ruby has 87 candies. If she shares the candies among 29<br>friends, the number of candies that each friend gets is<br>Ruby has 35 candies. If she shares the candies among 5<br>friends, the number of candies that each friend gets is|_g_ = 87_/_29 = 3<br>_g_ = 35_/_5 = 7|
|---|---|---|
|DCE(**_N_** _→R_)|The school is composed of 13 buildings each having 10<br>classrooms. The number of classrooms that the school has is<br>The school is composed of 65 buildings each having 2 class-<br>rooms. The number of classrooms that the school has is|_g_ = 10_×_13 = 130<br>_g_ = 65_×_2 = 130|
||The razorback t-shirt shop ordered 6 cases of t-shirts. If||
||each case contains 17 t-shirts the number of t-shirts that they|_g_ = 17_×_6 = 102|
|DCE(_S →R_)|ordered is<br>The roller coaster at the state fair costs 6 tickets per ride. If||
||17 friends were going to ride the roller coaster the number of|_g_ = 17_×_6 = 102|
||tickets that they would need is||
||Sean has 23 whistles. He has 6 more whistles than Charles.<br>The number of whistles that Charles has is|_g_ = 23_−_6 = 17|
|TCE(**_T_** _→R_)|Jovana flled her bucket with 23 pounds of shells. If she||
||adds 6 more pounds of shell to fll her bucket, the number of|_g_ = 23 + 6 = 29|
||pounds that she has is||



Table 1: For each of the causal effects measured (left column), we report a pair of MWPs illustrating the intervention performed (center), along with their respective ground-truth result (left column). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0014-02.png)


**----- Start of picture text -----**<br>
1<br>0 . 8 Accuracy@1 Accuracy@10<br>0 . 6<br>0 . 4<br>0 . 2<br>0<br>· 10 [−] [2]<br>4 Accuracy@1<br>2<br>0<br>DistilledRegularMediumLargeNeo-1.3BXLNeo-2.7BJ-6BNeoXDavinci-Instruct3-Curie3-Davinci-0023-Davinci-003<br>DistilledRegularMediumLargeNeo-1.3BXLNeo-2.7BJ-6BNeoX 3-Curie<br>Accuracy<br>Avg.<br>Accuracy<br>Avg.<br>**----- End of picture text -----**<br>


Figure 8: Average accuracy of the models on the generated instances of MWPs. Results are averaged over two sets consisting of 500 problem instances generated for each template. The lower figure shows a zoomed-in visualization of the accuracy at 1. 

correlated with TCE( _**N**_ on _R_ ) and TCE( _T_ on _R_ ) (0.24 and 0.49, respectively) and negatively correlated with DCE( _**N** → R_ )and DCE( _S → R_ ) (-0.26 and -0.36, respectively). We see these results as a quantitative validation of the intuition behind our framework: the better the model’s performance, the more the model tends to correctly adjust its prediction after a result-altering intervention (higher sensitivity) and to correctly not change its prediction after a result-preserving intervention (higher robustness). 

Moreover, we conduct an additional sanity check as in Patel et al. (2021): removing the question from the MWP templates, we observe a sensitivityrobustness degradation to random guessing (i.e., TCE _≃_ DCE). This indicates that the measurement of the causal effects within our framework is not affected by patterns in the templates that might have been picked up or memorized by large models. 

## **C Computation of Causal Effects for GPT-3** 

We access GPT-3 through the OpenAI APIs, which allow a user to prompt the model and obtain the probabilities assigned by the model to the _k_ -th most likely vocabulary entries, for each token generated. To overcome this limitation, we approximate the 

relative probability change _δ_ rcc as follows, depending on the kind of effect measured. 

The limit for _k_ is set by OpenAI to 5. However, for our main set of experiments (i.e., computing the causal effects of _**N**_ , _S_ , and _**T**_ ) we were granted an increased limit of _k_ to 100. This allowed us to obtain reasonable estimates for the causal effects, as the number of cases in which _P_ ( _g_ ) is not defined is less than 10% of the number of examples that we consider. 

## **Algorithm 1** Computation of _δ_ rcc for GPT-3 

_**Q**_ = ( _**t** ,_ _**n** , g_ ) _**Q**[′]_ = ( _**t**[′] ,_ _**n**[′] , g[′]_ ) 

**if** _P_ ( _g_ ) _is defined_ **then if** _P[′]_ ( _g_ ) _is defined_ **then** ∆= _[P]_[(] _[g]_[)] _[−][P][ ′]_[(] _[g]_[)] _P[′]_ ( _g_ ) **else** _P_ ˆ _[′] ← P[′]_ ( _k_ -th most likely token) ∆= _[P]_[(] _[g]_ ˆ[)] _[−][P]_[ˆ] _[ ′] P ′_ **end else** ∆= 0 

**end** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models_images/acl_2023_A_Causal_Framework_to_Quantify_the_Robustness_of_Mathematical_Reasoning_with_Language_Models.pdf-0015-06.png)


## **C.2** DCE( _**N** → R_ ) **and** DCE( _S → R_ ) 

In this case, we simply discard the examples for which _P_ ( _g_ ) is not defined or _P[′]_ ( _g_ ) are not defined. In that is not the case, then we compute _δ_ rcc as in Section 3.4. 

## **C.3 Heatmap Illustration** 

The heatmap for GPT-3 displayed in Figure 4 was computed by taking the raw probability score produced by the model over the whole vocabulary, as the limit on the available top predicted tokens makes it impossible to normalize it over the set _{_ 0 _, . . . ,_ 300 _}_ , as done for the other models. The probability was set to 0 when _g_ did not appear in the model’s top 5 predictions for the next token after the prompt. 

## **D Computing Infrastructure & Inference Details** 

To run our experiments, we used a single NVIDIA TITANRTX with 24GB of memory for all the versions of GPT-2 and GPT-Neo. We used a single NVIDIA A100 with 40GB of memory for GPT-J6B and a single NVIDIA A100 with 80GB of memory for GPT-NeoX and the LLaMA models (two for the 30B version). We accessed GPT-3 using the OpenAI APIs. The longest run (GPT-J) on the four kinds of experiments corresponding to the four kinds of effects measured took _∼_ 12 hours, using 500 MWP instances for each of the 437 templates. Due to budget and resource constraints, the experiments on GPT-3, GPT-NeoX, and LLaMA were carried out using 20 examples generated for each template and took _∼_ 7 hours. Experiment tracking was carried out using Weights & Biases[8] . 

**end** _δ_ rcc =[1] 2[(∆+ ∆] _[′]_[)] 

## **C.1** TCE( _**N**_ **on** _R_ ) **and** TCE( _**T**_ **on** _R_ ) 

In cases when _P_ ( _g_ ) is defined (i.e. when _g_ appears in the top _k_ token predictions) and _P[′]_ ( _g_ ) is not defined, we compute a lower bound on the relative change using the upper bound on _P[′]_ ( _g_ ) given by the probability of the _k_ -th most likely token. This gives us a conservative estimate of ∆. For cases in which _P_ ( _g_ ) is not defined, we cannot say anything about the relative change, and we set ∆= 0. The same applies when swapping _P_ and _P[′]_ . This procedure is illustrated by Algorithm 1. 

8http://wandb.ai/ 

