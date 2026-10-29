## **Neuro-Symbolic Continual Learning: Knowledge, Reasoning Shortcuts and Concept Rehearsal** 

**Emanuele Marconato**[* 1 2] **Gianpaolo Bontempo**[* 1 3] **Elisa Ficarra**[3] **Simone Calderara**[3] **Andrea Passerini**[2] **Stefano Teso**[4 2] 

## **Abstract** 

## **1. Introduction** 

We introduce Neuro-Symbolic Continual Learning, where a model has to solve _a sequence of neuro-symbolic tasks_ , that is, it has to map subsymbolic inputs to high-level concepts and compute predictions by _reasoning_ consistently with prior knowledge. Our key observation is that neuro-symbolic tasks, although different, often share concepts whose _semantics_ remains stable over time. Traditional approaches fall short: existing continual strategies ignore knowledge altogether, while stock neuro-symbolic architectures suffer from catastrophic forgetting. We show that leveraging prior knowledge by combining neurosymbolic architectures with continual strategies _does_ help avoid catastrophic forgetting, but also that doing so can yield models affected by _reasoning shortcuts_ . These undermine the semantics of the acquired concepts, even when detailed prior knowledge is provided upfront and inference is exact, and in turn continual performance. To overcome these issues, we introduce COOL, a **CO** ncept-level c **O** ntinual **L** earning strategy tailored for neuro-symbolic continual problems that acquires high-quality concepts and remembers them over time. Our experiments on three novel benchmarks highlights how COOL attains sustained high performance on neuro-symbolic continual learning tasks in which other strategies fail.[1] 

*Equal contribution 1University of Pisa, Italy 2DISI, University of Trento, Italy[3] University of Modena and Reggio Emilia, Italy[4] CIMeC, University of Trento, Italy. Correspondence to: Emanuele Marconato _<_ emanuele.marconato@unitn.it _>_ . _Proceedings of the 40[th] International Conference on Machine Learning_ , Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s). 

> 1The data and code are available at https://github.com/emamarconato/NeSy-CL 

We initiate the study of Neuro-Symbolic Continual Learning (NeSy-CL), in which the goal is to _solve a sequence of neuro-symbolic tasks_ . As is common in neuro-symbolic (NeSy) prediction (Manhaeve et al., 2018; Xu et al., 2018; Giunchiglia & Lukasiewicz, 2020; Hoernle et al., 2022; Ahmed et al., 2022a), the machine is provided _prior knowledge_ relating one or more target labels to symbolic, highlevel concepts _extracted_ from sub-symbolic data, and has to compute a prediction by _reasoning_ over said concepts. The central challenge of Nesy-CL is that the data distribution and the knowledge may vary across tasks. E.g., in medical diagnosis knowledge may encode known relationships between possible symptoms and conditions, while different tasks are characterized by different distributions of X-ray scans, symptoms and conditions. The goal, as in continual learning (CL) (Parisi et al., 2019), is to obtain a model that _attains high accuracy on new tasks without forgetting what it has already learned_ under a limited storage budget. 

Existing approaches are insufficient for NeSy-CL: neurosymbolic models are designed for offline learning and as such suffer from _catastrophic forgetting_ (Parisi et al., 2019), while continual learning strategies are designed for neural networks that neglect prior knowledge, preventing applications to tasks where compliance with regulations is key, e.g., safety critical tasks (Ahmed et al., 2022a; Hoernle et al., 2022). It is tempting to tackle NeSy-CL by pairing a SotA neuro-symbolic architecture, such as DeepProbLog (Manhaeve et al., 2018), with a proven rehearsal or distillation strategy, for instance dark experience replay (Buzzega et al., 2020). This yields immediate benefits, in the sense that prior knowledge makes the model more robust to catastrophic forgetting, as we will show. However, we show also that it is flawed, because it cannot prevent the model from acquiring _reasoning shortcuts_ (defined in Section 3), through which it attains high task accuracy by acquiring unintended concepts with _task-specific semantics_ , as illustrated in Figure 1. In turn, reasoning shortcuts entail poor cross-task transfer. 

Our key observation is that, even though neuro-symbolic tasks may differ in terms of knowledge and distribution, _the semantics of the concepts they rely on must remain stable_ 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Neuro-Symbolic Continual Learning<br>Task 1 Examples Task 2 Examples Future Task<br>K same ̱color  = 0<br>(   ,   ) Y  = 0 Y  = 1 Y  = 0 Y  = 1 Y  = 1<br>Pred Pred<br>? ?<br>nn<br>True True ☹<br>**----- End of picture text -----**<br>


_Figure 1._ **Left** : DeepProbLog extracts concepts **c** from a sub-symbolic input **x** and reasons over prior knowledge K provided upfront to obtain a prediction **y** . **Right** : Simplified illustration of our CLE4EVR benchmark, restricted to color concepts only. The full setting is reported in Section 5. The goal is to predict whether two objects have the _Y_ = `same` ~~`c`~~ `olor` . The first task includes examples of **gray** and **red** objects only, and the model classifies them by learning the intended mapping between input colors and concepts. The second one includes **green** and **blue** objects only, and the model can learn a _reasoning shortcut_ mapping the four colors to two concepts, achieving high accuracy on both tasks but compromising performance on _future_ tasks. This is exactly the issue addressed by our approach, COOL. 

_over time_ . For instance, in automated protein annotation inferred signal peptides entail a catalytic function (Rost et al., 2003) regardless of the specific protein under examination, and in vehicle routing presence of pedestrians on the road is sufficient to rule out certain routes (Xu et al., 2020) regardless of location and weather conditions. 

Prompted by this insight, we propose COOL, a simple but effective COncept-level cOntinual Learning strategy, that aims at _acquiring high-quality concepts and preserving them across tasks_ . COOL makes use of a small amount of concept supervision to acquire high-quality concepts and explicitly preserves them with a _concept_ rehearsal strategy, avoiding reasoning shortcuts in all tasks. COOL is applicable to a variety of NeSy architectures, and – as shown by our experiments with DeepProbLog (Manhaeve et al., 2018) and Concept-based Models (Koh et al., 2020) – easily outperforms state-of-the-art continual strategies on three novel, challenging NeSy-CL problems, achieving better concept quality and predictive accuracy on past and OOD tasks. 

## **Contributions.** Summarizing, we: 

1. Introduce neuro-symbolic continual learning as a novel and challenging machine learning problem. 

2. We show that knowledge readily improves forgetting in some scenarios, but also that it is insufficient to prevent reasoning shortcuts – which worsen forgetting and compromise transfer to new tasks – in others. 

3. Propose new NeSy-CL benchmarks for evaluating continual performance with and without reasoning shortcuts. 

4. Introduce COOL, a novel continual strategy that supports identifying concepts with the intended semantics and preserves them across tasks. 

5. Show empirically that COOL outperforms state-of-the-art continual strategies on these challenging benchmarks. 

## **2. Neuro-Symbolic Continual Learning** 

**Notation.** Throughout, we indicate scalar constants _x_ in lower-case, random variables _X_ in upper case, and ordered sets of constants **x** and random variables **X** in bold typeface. The symbol [ _n_ ] stands for the set _{_ 1 _, . . . , n}_ and **x** _|_ = K indicates that **x** satisfies a logical formula K. We say that a distribution _p_ ( **A** _|_ **B** ; K) is _consistent_ with K, written _p |_ = K, if it holds that _p_ ( **a** _|_ **b** ; K) _>_ 0 implies ( **a** _,_ **b** ) _|_ = K, i.e., if _p_ associates zero mass to all states that violate K. 

## **2.1. Problem Statement** 

We are concerned with solving a _sequence of neuro-symbolic prediction tasks_ , each requiring to learn a classifier mapping a (partially) sub-symbolic input **x** _∈_ R _[d]_ to _n ≥_ 1 labels **y** _∈_ N _[n]_ . What makes them neuro-symbolic is that: ( _i_ ) The labels **y** depend entirely on the state of _k symbolic concepts_ **c** = ( _c_ 1 _, . . . , ck_ ) _[⊤]_ capturing high-level aspects of the input **x** . ( _ii_ ) The concepts **c** depend on the _sub-symbolic input_ **x** in an intricate manner and are best extracted using deep learning techniques. ( _iii_ ) The way in which the labels **y** depend on the concepts **c** is specified by _prior knowledge_ K, necessitating reasoning during inference. 

We make the natural assumption that the _semantics of the concepts appearing in the various tasks remain constant over time_ . This assumption lies at the heart of knowledge representation and ontology design, where concepts serve as a _lingua franca_ for the exchange and reuse of knowledge across application boundaries (Gruber, 1995) and with human stakeholders (Kambhampati et al., 2022). 

Formally, each task _t ∈_ N is defined by a data generating distribution _p_[(] _[t]_[)] ( **X** _,_ **C** _,_ **Y** ; K[(] _[t]_[)] ) that factorizes as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0002-16.png)


Here, K[(] _[t]_[)] denotes the knowledge relevant to the _t_ -th task.[2] As customary in NeSy, we assume the knowledge to cor- 

> 2The knowledge might depend also on discrete variables in **X** ; we suppress this dependency in the notation for readability. 

2 

**Neuro-Symbolic Continual Learning** 

rectly describe the ground-truth generative process and that, therefore, the label distributions are _consistent_ with their respective prior knowledge, i.e., _p_[(] _[t]_[)] ( **Y** _|_ **X** ; K[(] _[t]_[)] ) _|_ = K[(] _[t]_[)] for all _t_ . The key feature of Equation (1) is that _p_ ( **C** _|_ **X** ) does not depend on _t_ , capturing our assumption that concept semantics are stable. For instance, if _C_ `dog` represents the notion of “dog”, then _p_ ( _C_ `dog` _|_ **X** ) only depends on whether an image **x** in fact depicts a dog, regardless of context, style, likelihood of observing a dog, and role of dogs in determining the label. See Appendix A.3 for an in-depth discussion. Critically, the distribution of observed inputs, concepts and labels, and the knowledge _are allowed to differ between tasks_ . This means that, e.g., known concepts may stop occurring, play different roles in K[(] _[t]_[+1)] than they did in K[(] _[t]_[)] , and entirely new concepts may appear. 

**Handling catastrophic forgetting.** At step _t_ , the machine obtains a task _T_[(] _[t]_[)] = ( _D_[(] _[t]_[)] _,_ K[(] _[t]_[)] ) consisting of a data set _D_[(] _[t]_[)] , sampled i.i.d. according to Equation (1), and knowledge K[(] _[t]_[)] . The goal is to find parameters _θ_ that achieve low _average risk_ over all tasks observed so far, defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0003-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0003-04.png)


where _T_[1:] _[t]_ = _{T_[(1)] _, . . . , T_[(] _[t]_[)] _}_ is the collection of all tasks 1 observed so far, _L_ ( _θ, T_ ) := _|D|_ �( **x** _,_ **y** ) _∈D[ℓ]_[(] _[θ,]_[ (] **[x]** _[,]_ **[ y]**[);][ K][)][,] and _ℓ_ is a loss function over **y** . 

As in regular continual learning, we assume storage is insufficient to hold data from all tasks, meaning that _L_ ( _θ, T_[1:(] _[t][−]_[1)] ) cannot be evaluated exactly. Ignoring this term, as done by offline approaches, leads to _catastrophic forgetting_ : by focusing on the loss of the current task, models tend to forget the information necessary to solve the previous tasks. CL algorithms mitigate forgetting using strategies like rehearsal (Buzzega et al., 2020; Parisi et al., 2019; De Lange et al., 2021; Boschini et al., 2022; Rebuffi et al., 2017) (i.e., replaying few examples of previous tasks), regularization (Huszar´ , 2018; Li & Hoiem, 2017) (i.e., slowing down parameter shift through additional terms in the loss function), or architectural modifications (Rusu et al., 2016) (i.e., freezing and adding new parameters at each task). See Section 6 for an overview. 

Importantly, all these strategies focus on _optimizing the accuracy on the labels only_ . This is sensible in CL but, as shown in Section 3, insufficient in NeSy-CL. 

## **2.2. DeepProbLog** 

DeepProbLog (Manhaeve et al., 2018) is a state-of-the-art model ideally suited to solve tasks of the form in Equation (1). It decomposes prediction into two steps, cf. Fig- 

ure 1 (left). At the lower level, it implements each concept as a Boolean or categorical random variable _Cj_ , whose distribution _pθ_ ( _Cj |_ **x** ) is flexibly parameterized by a neural network _nnj_ ( **x** ; _θ_ ). This implies that concepts are mutually independent given the input, i.e., _pθ_ ( **C** _|_ **x** ) =[�] _j∈_ [ _k_ ] _[p][θ]_[(] _[C][j][|]_ **x** ). At the upper level, it models the distribution of labels conditioned on concepts as a _uniform distribution_ over the support of K, and specifically as:[3] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0003-11.png)


where _Z_ ( **c** ; K) =[�] **y**[1] _[{]_[(] **[c]** _[,]_ **[ y]**[)] _[ |]_[=][ K] _[}]_[is][a][normalization] constant. The overall label distribution is obtained by marginalizing over **C** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0003-13.png)


Since the indicator function in Equation (4) evaluates to zero for all values of **c** that violate K, the label distribution in Equation (5) is _by construction_ consistent with K. 

**Example 1.** MNIST-Addition _(Manhaeve et al., 2018) is a prototypical neuro-symbolic task that requires learning a mapping from pairs of MNIST digits_ **x** = ( **x** 1 _,_ **x** 2) _to their sum Y , e.g., from_ **x** = ( _,_ ) _to y_ = 5 _. It can be readily modelled in DeepProbLog using two concepts C_ 1 _and C_ 2 _ranging in {_ 0 _, . . . ,_ 9 _}, each predicted by a convolutional network nn_ ( **x** _j_ ) _, and a constraint_ K = ( _C_ 1 + _C_ 2 = _Y_ ) _._ 

Given a task _T_ = ( _D,_ K), the parameters _θ_ are usually learned by maximizing the log-likelihood _L_ ( _θ, D_ ) := _|D|_ 1 �( **x** _,_ **y** ) _∈D_[log] _[ p][θ]_[(] **[y]** _[|]_ **[x]**[;][ K][)][via][stochastic][gradient][de-] scent. Computing the (gradient of the) likelihood _pθ_ ( **y** _|_ **x** ) � and the most likely prediction **y** _∈_ argmax **y** _pθ_ ( **y** _|_ **x** ; K) requires to evaluate _Z_ , which is intractable in general. To make inference practical, DeepProbLog exploits knowledge compilation (Darwiche & Marquis, 2002) to convert the distribution _u_ K into a probabilistic circuit. Once in this format, the above operations take time linear in the size of the circuit (Choi et al., 2020; Vergari et al., 2021). 

In the remainder of the paper, we focus on DeepProbLog as it offers a sound probabilistic architecture and exact inference. Our results, however, do transfer to many other neuro-symbolic architectures, as discussed in Section 6. 

## **3. Knowledge and Reasoning Shortcuts** 

All neuro-symbolic architectures, including DeepProbLog, are designed for _offline_ settings, and as such they easily fall prey of catastrophic forgetting when applied to NeSy-CL problems. This issue is illustrated by the following example and demonstrated empirically in Section 5. 

3Non-uniform distributions consistent with the knowledge can also be modelled, as done for instance by Ahmed et al. (2022a). 

3 

**Neuro-Symbolic Continual Learning** 

**Example 2.** _We introduce_ MNAdd-Seq _, a continual extension of_ MNIST-Addition _in which tasks differ in what digits are observed. Specifically, each task t_ = 0 _, . . . ,_ 8 _consists of all pairs of digits_ **x** = ( **x** 1 _,_ **x** 2) _whose sum is either_ 2 _t or_ 2 _t_ +1 _. By construction, sums above_ 9 _cannot be obtained by adding smaller digits, so these no longer occur in later tasks. Since DeepProbLog maximizes the likelihood of the current task, it quickly forgets small digits at t ≥_ 5 _and can no longer classify sums involving them correctly._ 

## **3.1. Knowledge Helps Remembering ...** 

A natural first step toward solving NeSy-CL is to bundle a NeSy predictor – say, DeepProbLog – with any state-ofthe-art CL strategy. Focusing on experience replay, doing so amounts to storing a handful of well chosen labeled (or predicted) examples ( **x** _,_ **y** ) from past tasks 1 _, . . . , t −_ 1, and replaying them when fitting DeepProbLog on the current task _t_ together with the corresponding prior knowledge K _[t]_ .[4] 

Doing so immediately brings a number of benefits. First and foremost, the knowledge encodes the valid, stable relationship between the concepts and the labels to be prediction. This implies that predicted concepts can be always correctly mapped to a corresponding label, and that this inference step is immune from forgetting. This is especially significant considering that the top layers of neural networks are those most affected by catastrophic forgetting (Wu et al., 2019). This effect is clearly visible in our experiments, cf. Section 5. 

Conversely, prior knowledge effectively reduces the space of candidate concepts, providing further guidance to the model. If it reduces the space to only those having the intended semantics, then this simple setup can be very effective at tackling NeSy-CL problems. 

## **3.2. . . . But Does Not Prevent Reasoning Shortcuts** 

In general, however, this setup is insufficient. The core issue is that knowledge might not be enough to identify the right concept distribution _p_ ( **C** _|_ **X** ) using label annotations alone, in the sense that – depending on how the knowledge and training data are structured – it may be possible (in both offline and continual settings) to _correctly classify all training examples even using concepts with unintended semantics_ . We refer to these situations as _reasoning shortcuts_ . 

This intuition is formalized in Theorem 1. Here, we write Θ to indicate the set of all possible parameters of (the neural networks implementing) _pθ_ ( **C** _|_ **X** ), and Θ _[∗]_ (K _, D_ ) _⊆_ Θ for the parameters that maximize the log-likelihood _L_ ( _θ, D_ ). Also, K[ **V** _/_ **v** ] is the knowledge obtained by substituting all 

> 4The only non-trivial aspect is that, in addition to the replay buffer, we also has to store the past knowledge, so as to ensure be able to match the updated concepts with the past labels. 

occurrences of variables **V** with constants **v** . For instance, in MNIST-Addition K[ _Y/_ 2] amounts to _C_ 1 + _C_ 2 = 2. **Theorem 1.** _A model with parameters θ attains maximal likelihood, i.e., θ ∈_ Θ _[∗]_ (K _, D_ ) _, if and only if, for all_ ( **x** _,_ **y** ) _∈ D, it holds that pθ_ ( **C** _|_ **x** ) _|_ = K[ **Y** _/_ **y** ] _._ 

All proofs can be found in Appendix A. Theorem 1 states that, as long as the concept distribution output by the learned neural network satisfies the knowledge for each training example, the log-likelihood is maximal.[5] The ground-truth concept distribution _p_ ( **C** _|_ **X** ) is a possible solution, but it is not necessarily _the only one_ . In this case, fitting DeepProbLog – and indeed any NeSy approach that optimizes for label accuracy only – _does not guarantee that the learned concepts have the correct semantics_ . To see this, consider the following example. 

**Example 3.** _Consider_ MNIST-Addition _and take a subset D including only pairs of examples of four possible sums:_ + = 6 _,_ + = 10 _,_ + = 10 _, and_ + = 12 _. Then, there exist many concept distributions that satisfy the knowledge on all examples, including:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0004-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0004-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0004-15.png)


_where the remaining concepts are allocated arbitrarily and_ **x** _�→ c is a shorthand for p_ ( _C_ = _c |_ **x** ) = 1 _{C_ = _c}. Only the first distribution has the intended semantics, whereas the second one is a reasoning shortcut. We remark that DeepProbLog does acquire this shortcut in practice, as illustrated by our experiments. Appendix D explains how shortcuts emerge in the data sets used in our experiments._ 

Notice that the Theorem applies to both offline learning (i.e., _D_ is fixed) and NeSy-CL (i.e., _D_ indicates the training set of any given task). Yet, reasoning shortcuts are especially impactful in the latter. This is exemplified in Figure 1 (right). Here, DeepProbLog has learned high-quality concepts to solve the first task, but quickly forgets them when solving the second task, precisely because it falls pray of a reasoning shortcut that achieves high training and rehearsal accuracy on both tasks by satisfying the knowledge using concepts with unintended semantics. We provide additional concrete examples in Appendix D. In turn, reasoning shortcuts can dramatically affect forgetting and performance on future and OOD NeSy tasks, as shown by our experiments. 

## **4. Addressing NeSy-CL with COOL** 

To this end, we introduce COOL, a COncept-level cOntinual Learning that acquires concepts with the intended semantics and preserves them over time, attaining sustained high 

5This theorem essentially shows that, from the neural network’s perspective, the reasoning layer of DeepProbLog has the same effect as the Semantic Loss (Xu et al., 2018). 

4 

**Neuro-Symbolic Continual Learning** 

performance. Formally, COOL is designed to satisfy two desiderata: (D1) _pθ_ ( **C** _|_ **X** ) should quickly approximate _p_ ( **C** _|_ **X** ), and (D2) _pθ_ ( **C** _|_ **X** ) should remain stable across tasks. D2 is straightforward, however we stress that it is only meaningful if D1 also holds: unless the learned concepts are high-quality, there is little benefit in remembering them. 

In order to comply with D1, COOL makes use of a small number of densely annotated examples to quickly identify high-quality concepts, which – as we have shown – cannot always be guaranteed using knowledge alone. In practice, an average cross-entropy is added to the loss of these examples. To cope with D2, COOL implements a novel _concept rehearsal_ strategy that stabilizes _pθ_ ( **C** _|_ **X** ) across tasks. This is motivated by the fact that concept stability helps to upper bound the average risk. Specifically, 

**Theorem 2.** _Consider tasks T_[1:] _[t] . If the current model θ and the past one θ_[(] _[t][−]_[1)] _assign non-zero likelihood to all examples in D_[1:] _[t] , there exists a finite constant γ, depending only on the model architecture, knowledge and data, such that the average risk in Equation_ (3) _is upper bounded by:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0005-04.png)


In words, this means that if the past model _θ_[(] _[t][−]_[1)] performs well on all past tasks (i.e., the middle term in Equation (6) is small), a new model that performs well on the current task (the first term is small) and whose concept distribution is close to that of the old model (the last term is small), also performs well on past tasks (the average risk in Equation (3) is small). Critically, this results holds regardless of how the prior knowledge K[1:] _[t]_ of the various task is chosen. 

COOL implement this requirement by combining the original training loss with an extra penalty _L_ COOL, defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0005-07.png)


Here, _M_ denotes the mini-batch of examples extracted from the replay buffer, _α_ denotes the scalar weight associated to the concept-reharsal strategy, and _β_ the weight of the replay strategy on **y** . The KL term is evaluated between ˜ the predicted concept distribution and the stored one **q** _c_ = _pθ_ ( _t−_ 1)( **C** _|_ **x** ). Notice that, by Pinsker’s inequality, the KL upper bounds the (square of the) _L_ 1 distance, meaning that COOL indirectly optimizes the bound in Equation (6). 

## **4.1. Benefits and Limitations** 

COOL is explicitly designed to acquire high-quality concepts and retain them across tasks by combining knowledge, 

concept rehearsal, and a modicum of concept supervision. This substantially improves performance on past, future, and OOD tasks sharing these concepts, as demonstrated in Section 5. COOL works even if the knowledge K[(] _[t]_[)] changes across tasks and new concepts appear over time: these can be encoded as additional neural predicates in DeepProbLog, and COOL will take care of remembering the known concepts while leaving room to learn the new ones. 

One limitation of COOL is that, in the general case, it requires a handful of densely annotated examples. The same requirement can be found in other settings where concept quality is critical. For instance, concept supervision is key in concept-based models – which strive to generate conceptlevel explanations for their predictions – to ensure the acquired concepts are _interpretable_ (Koh et al., 2020; Chen et al., 2020; Marconato et al., 2022b). It is also a prerequisite for guaranteeing that learned representations acquired by general (deep) latent variable models are _disentangled_ , as shown theoretically (Locatello et al., 2019) and empirically (Locatello et al., 2020). We stress that concept supervision is _not_ required if knowledge and data disallow shortcut solutions, as is the case in MNAdd-Seq (see Section 5), although it does help avoiding sub-optimal parameters even in this case. If reasoning shortcuts _are_ possible, however, concept supervision becomes essential, because – by construction – knowledge and labels alone are insufficient to pin down the correct semantics, hindering concept quality. Moreover, in many situations, annotating just _some_ concepts is sufficient to rule out reasoning shortcuts. 

## **5. Empirical Evaluation** 

We address empirically the following research questions: 

- **Q1** : Does knowledge help to stabilize the continual learning process and reduce the need for supervision? 

- **Q2** : Does COOL help avoid reasoning shortcuts when knowledge alone fails, thus facilitating past and future continual performance? 

- **Q3** : How much concept supervision does COOL need? 

To answer these questions, we compared COOL against several representative continual strategies on three _novel_ and challenging NeSy-CL benchmarks. Additional results and details on data sets, metrics, and hyperparameters can be found in the Appendices. 

**Data sets.** Existing NeSy and CL benchmarks are designed for offline settings or lack any sort of prior knowledge, respectively. Hence, in order to evaluate COOL, we introduce three novel NeSy-CL benchmarks specifically designed to evaluate impact of knowledge, concept quality and robustness to reasoning shortcuts, briefly described next. 

5 

**Neuro-Symbolic Continual Learning** 

MNAdd-Seq is the problem introduced in Example 2 and it is designed _not_ to contain reasoning shortcuts. In short, inputs **x** are pairs of MNIST (LeCun, 1998) digits labeled with their sum _y_ ; each digit is mapped to a concept, and K specifies that their sum must match the label. In each task _t_ = 0 _, . . . ,_ 8 includes only examples with labels _y ∈ {_ 2 _· t,_ 2 _· t_ +1 _}_ , making this problem both _label incremental_ (only two out of 18 possible labels are observed per task) and _concept incremental_ (higher digits only appear in later tasks, while lower digits disappear, see Appendix C). The data set holds 42k training examples, of which we used 8 _._ 4k for validation, and 6k test examples. 

MNAdd-Shortcut is a simple two-task version of MNIST-Addition used here to illustrate the impact of _reasoning shortcuts_ . The first task includes only even digits and the second one only odd ones. In the first task we include 4 types of examples: (i) + = 6, (ii) + = 10, (iii) + = 10, and (iv) + = 12. In the second task, we allow all possible sums of odd digits _{ , , , , }_ . As shown in Example 3 and further discussed in Appendix D, the four sums in the first task are not sufficient to identify the correct digits, i.e., different shortcuts are possible. This data consists of 13 _._ 8k examples, 2 _._ 8k of which are reserved for validation and 2k for testing. We also include an additional OOD test set with 4k unseen combinations of all concepts, like sums involving an odd and an even digit, allowing us to probe the efficacy on a plausible future task. 

CLE4EVR is a challenging new _concept-incremental_ NeSyCL benchmark based on CLEVR (Johnson et al., 2017). Inputs **x** are renderings of two randomly placed 3D objects with several possible shapes, colors, materials, and sizes. The goal is to predict whether the objects have the same color, same shape, both, or neither. The knowledge simply defines the three labels using four (one-hot encoded) concepts encoding shape and color of the two objects. There are 5 tasks. Objects in each task have only two colors out of ten and two shapes out of ten, with no overlap between tasks. Knowledge and labels allow for a large number of _reasoning shortcuts_ , as illustrated in Figure 1 and detailed in Appendix D. To evaluate quality of learned concepts, we also define an OOD test set containing _unseen combinations_ of training objects. Overall, the dataset contains almost 5 _._ 5k training data, 500 data for validation and 2 _._ 5k data for test. 

**Metrics.** We evaluate all models using common CL metrics (De Lange et al., 2021), namely class-incremental accuracy (Class-IL) of labels **Y** and concepts **C** , and forward transfer of labels (FWT). For MNAdd-Shortcut and CLE4EVR we also measure label and concept accuracy on the OOD test set. 

**Competitors.** We compare COOL against the following _label-based_ continual strategies: NA¨IVE fine-tunes the old 

|CBM @10%<br>DEEPPROBLOG|STRATEGY<br>CLASS-IL**Y**(_↑_)<br>CLASS-IL**C**(_↑_)<br>FWT (_↑_)|
|---|---|
||NA¨IVE<br>11_._71_±_0_._8<br>36_._2_±_2_._6<br>7_._5_±_0_._3<br>RESTART<br>10_._78_±_0_._1<br>29_._7_±_0_._1<br>7_._3_±_0_._2<br>LWF<br>18_._08_±_1_._8<br>63_._2_±_4_._4<br>_−_4_._7_±_1_._1<br>EWC<br>11_._57_±_0_._6<br>37_._4_±_0_._6<br>7_._6_±_0_._4<br>ER<br>13_._29_±_0_._4<br>43_._5_±_2_._0<br>13_._4_±_1_._6<br>DER<br>18_._63_±_2_._5<br>53_._1_±_1_._7<br>15_._7_±_0_._9<br>DER++<br>18_._17_±_1_._6<br>54_._1_±_3_._0<br>16_._6_±_1_._8<br>COOL<br>**38****_._0****_±_ 1****_._9**<br>**78****_._1****_±_ 2****_._5**<br>**29****_._0****_±_ 4****_._8**|
||NA¨IVE<br>6_._9_±_0_._2<br>6_._7_±_0_._4<br>6_._2_±_0_._2<br>RESTART<br>9_._6_±_0_._3<br>0_._2_±_0_._1<br>6_._9_±_0_._8<br>LWF<br>6_._8_±_0_._5<br>10_._8_±_4_._6<br>18_._3_±_0_._2<br>EWC<br>6_._8_±_0_._4<br>7_._8_±_0_._6<br>6_._1_±_0_._3<br>ER<br>44_._3_±_9_._7<br>62_._0_±_8_._6<br>8_._2_±_4_._1<br>DER<br>68_._3_±_9_._4<br>81_._3_±_6_._9<br>44_._5_±_23_._7<br>DER++<br>62_._2_±_5_._4<br>77_._1_±_4_._2<br>27_._1_±_5_._2<br>COOL<br>**71****_._9****_±_ 2****_._9**<br>**84****_._5****_±_ 1****_._9**<br>**83****_._2****_±_ 0****_._9**|



_Table 1._ **Knowledge helps, and COOL helps even more.** Top block: results on MNAdd-Seq for all competitors + CBM with 10% concept supervision, averaged over 10 seeds. Bottom block: same for DEEPPROBLOG with 0% supervision. COOL + DeepProbLog outperforms the neural baseline (despite the gap in supervision) and the other continual strategies. The additional results in Appendix E support these conclusions. 

model on each new task without any continual strategy. RESTART fits a model from scratch for each task, without any continual strategy. RESTART and NA¨IVE serve as baselines to quantify the impact of forgetting. LWF: Learning without Forgetting (Li & Hoiem, 2017), a regularization approach that performs knowledge distillation from the past model. EWC: Elastic Weight Consolidation (Kirkpatrick et al., 2017), a regularization approach that avoids drastic updates to important parameters based on the Fisher values of the previous model. ER: Experience Replay (Riemer et al., 2019), a popular rehearsal approach that stores a random selection of past examples and replays them when training on the new task. DER: Dark Experience Replay (Buzzega et al., 2020), a state-of-the-art rehearsal approach similar to ER that stores and distills the logits of the past model. DER++: an improvement of DER that also stores the true label. We do not compare against prototype-based strategies, like iCaRL (Rebuffi et al., 2017), because class prototypes cannot be easily defined in structured representation spaces. We also consider OFFLINE learning over the union of all tasks as an ideal upper bound. COOL is implemented as in Equation (7). Following DER, replay examples are selected with _reservoir sampling_ (Vitter, 1985), an efficient incremental method for random sampling with uniformity guarantees. All hyperparameters were chosen to optimize last-task Class-IL on the validation labels. 

**Q1: Knowledge helps, COOL helps even more.** We evaluate the impact of knowledge on MNAdd-Seq, where shortcuts are _absent_ . Specifically, we evaluate different continual strategies paired with DeepProbLog (Manhaeve et al., 2018) and Concept-bottleneck Models (CBMs) (Koh et al., 2020). Both architectures extract concepts using a convolutional network, but differ in how they infer the label. DeepProbLog uses a probabilistic-logic layer encoding the 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0007-00.png)


**----- Start of picture text -----**<br>
Neuro-Symbolic Continual Learning<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0007-01.png)


**----- Start of picture text -----**<br>
CLASS-IL ( Y ) CLASS-IL ( C ) OOD ACCURACY ( Y ) PREDICTED CONCEPTS<br>RUTH<br>-T<br>ROUND<br>G<br>MNAdd-Shortcut<br>RUTH<br>-T<br>ROUND<br>G<br>CLE4EVR<br>**----- End of picture text -----**<br>


_Figure 2._ **COOL avoids shortcuts with few concept-annotated examples.** (Left) Class-IL (on labels and concepts) and OOD accuracy for DER and COOL on MNAdd-Shortcut (top) and CLE4EVR (bottom). The _x_ -axis is the % of concept annotated examples per task. (Right) Confusion matrices of shape (top) and color (bottom), computed on the last task of CLE4EVR, obtained by DeepProbLog paired with COOL (in **blue** ) vs. DER (in **red** ) with 10% concept supervision. COOL is the only strategy that acquires and maintains the intended semantics. The complete numerical results are reported in Appendix E. 

available knowledge (see Section 2.2). CBMs aggregate the concepts using a learnable network _independent_ of the knowledge, and serve as a purely neural baseline. Additional architectural choices are reported in Appendix B. The two models were trained for 25 epochs per task, using a fixed buffer size of 1000, but received different amounts of concept supervision: 10% for CBMs, which is enough to learn the intended concepts in the offline setting, and _none_ for DeepProbLog. 

The _offline_ performance of CBM and DeepProbLog are excellent, achieving around 96% label accuracy and 99% concept accuracy, showing that both are capable of solving the learning task. In the _continual_ setting, however, the gap between the neural and NeSy models widens noticeably. The results are reported in Table 1. Despite DeepProbLog being harder to learn than CBMs (as shown by RESTART and NA¨IVE), all replay strategies – i.e., ER, DER, DER++, and COOL– perform much better when paired with DeepProbLog than CBMs: remarkably, label Class-IL sees gains close to 50% for DER, and similarly FWT for COOL. This highlights the benefits of knowledge, which apply despite DeepProbLog having access to _no concept supervision_ . The regularization strategies LWF and EWC are not informative, as they struggle to improve on the NA¨IVE and RESTART baselines both _with_ and _without_ knowledge.[6] 

Overall, Table 1 indicates that _when reasoning shortcuts are absent_ , knowledge facilitates identifying better concepts, and thus better predictions. By retaining these concepts, COOL manages to outperform all other competitors on both 

6 The only exception is LWF on CBM, which displayed pathological behavior, see Appendix E. 

CBMs and DeepProbLog. The runner-up, DER, keeps up only when knowledge is available and with a substantial margin in terms of FWT (45% vs. 83%). In contrast, even though CBMs can acquire good concepts, this does not always yield good predictions, chiefly because the top layer undergoes forgetting, cf. Section 3.1. Thanks to knowledge, DeepProbLog avoids this issue altogether. 

**Q2: COOL avoids reasoning shortcuts.** Next, we evaluate the impact of concept quality and rehearsal in MNAdd-Shortcut and CLE4EVR, which _are_ affected by reasoning shortcuts. Given the sub-par performance of CBMs, we focus on DeepProbLog from now on. Also, we restrict our attention to COOL and DER, the runner up in the previous experiment. Results for all other competitors are available in Appendix E. For MNAdd-Shortcut we set a buffer size of 1000 and 100 epochs per task, and to 250 examples and 50 epochs for CLE4EVR. 

The results in Figure 2 shows that, when no concept supervision is in place, the presence of shortcuts complicates retaining the correct concepts, as displayed by low values of Class-IL ( **C** ). This does not impact directly Class-IL ( **Y** ) in the case of CLE4EVR, but yields extremely low OOD generalization, around 10% for MNAdd-Shortcut and 25% for CLE4EVR. The effect on increasing concept supervision on DER is only seemingly positive, as label accuracy does improve in both data sets. However, Class-IL on concepts (about 20–50%) and OOD accuracy (10%–30%) are very poor, despite the supervision. What happens is precisely the issue depicted in Figure 1: the model acquires good concepts for one task, but – due to reasoning shortcuts and lack of concept rehearsal – these get corrupted when fit- 

7 

**Neuro-Symbolic Continual Learning** 

ting on the next task. Since COOL retains the high quality concepts identified via supervision, the latter leads to clear improvements in label accuracy, concept accuracy _and_ OOD accuracy for COOL. As a result, COOL improves on DER by about +30% and +60% in terms of Class-IL ( **C** ) and +40% and +60% in OOD, in the two data sets respectively. 

We stress that label-based strategies inevitably fall for reasoning shortcuts _even if concept supervision is provided_ . This is clearly shown by the concept confusion matrices reported in Figure 2 (right). Notice that, out of _all_ strategies, only COOL manages to prevent shortcuts. Further details are available in Appendix E. 

**Q3: COOL requires minimal concept supervision.** Figure 2 shows that COOL identifies high-quality concepts when given dense annotations for only 1% of the training set. This translates to about 30 examples per task in MNAdd-Shortcut, and to only 12 in CLE4EVR. Increasing concept supervision to 10% improves Class-IL ( **C** ) by 3% and shrinks its variance, but 1% is enough to substantially outperform DER in our tests. 

## **6. Related Work** 

**Neuro-symbolic integration.** NeSy encompasses a diverse family of methods integrating learning and reasoning (De Raedt et al., 2021). Here, we focus on approaches for encouraging neural networks to output structured predictions consistent with prior knowledge.The two main strategies introduce an additional loss penalizing inconsistent predictions (Xu et al., 2018; Fischer et al., 2019; Ahmed et al., 2022b) or a top reasoning layer (Manhaeve et al., 2018; Giunchiglia & Lukasiewicz, 2020; Hoernle et al., 2022; Ahmed et al., 2022a). Since the former cannot guarantee that the model outputs consistent predictions, we focus on the latter. In either case, end-to-end training requires to differentiate through the knowledge. One option is to soften the knowledge using fuzzy logic (Diligenti et al., 2012; Donadello et al., 2017), but doing so can introduce semantic and learning artifacts (Giannini et al., 2018; van Krieken et al., 2022a). An alternative is to cast reasoning in terms of probabilistic logics (De Raedt & Kimmig, 2015), which preserves semantics and allows for sound inference and learning. DeepProbLog is just an example of Nesy strategies (Manhaeve et al., 2021; Huang et al., 2021; Winters et al., 2022; Ahmed et al., 2022a; van Krieken et al., 2022b). All NeSy approaches are _offline_ and suffer from catastrophic forgetting, and existing continual strategies do not protect them from reasoning shortcuts, as shown in Section 5. Since these depend only on the latent nature of concepts, they affect probabilistic-logic and fuzzy logic architectures alike. COOL applies to all these, cf. Appendix A.4. 

**Continual Learning.** CL algorithms attempt to pre- 

serve model plasticity while mitigating catastrophic forgetting (Robins, 1995) using a variety of techniques (van de Ven et al., 2022; Qu et al., 2021). A first group of strategies, like Experience Replay (Riemer et al., 2019) and ER-ACE (Caccia et al., 2022), store and rehearse a limited amount of examples from previous tasks. Doing so ignores additional “dark knowledge” learned by the past model, so techniques like DER (Buzzega et al., 2020), DER++, and others (Rebuffi et al., 2017; Li & Hoiem, 2017; Castro et al., 2018; Hou et al., 2019), drop rehearsal in favor of distillation. COOL follows the same strategy. Popular alternatives include architectural approaches (Rusu et al., 2016), which freeze or add model parameters as needed, and regularization strategies (De Lange & Tuytelaars, 2021; Kirkpatrick et al., 2017; Aljundi et al., 2018; Zenke et al., 2017). These introduce extra penalties in the loss function to discourage changing parameters essential for discriminating classes, but can struggle with complex data (Aljundi et al., 2019). To the best of our knowledge, CL has only been tackled in flat prediction settings (e.g., classification), and existing strategies focus on preserving label accuracy only. The only work on forgetting in CBMs is (Marconato et al., 2022a), which however ignores knowledge altogether. 

**Reasoning shortcuts.** In machine learning, “shortcuts” refer to models that exploit _spurious_ correlations between inputs and annotations to achieve high training accuracy (Ross et al., 2017; Lapuschkin et al., 2019). Proposed solutions include dense annotations (Ross et al., 2017), out-of-domain data (Parascandolo et al., 2020), and interaction with annotators (Teso et al., 2022). Stammer et al. (2021) have investigated shortcuts in NeSy and proposed to fix them using knowledge, under the assumption that concepts are high-quality. We make no such assumption. Our work is the first to investigate _reasoning_ shortcuts that knowledge cannot always prevent and their preminence in NeSy-CL. 

## **7. Conclusion** 

We initiated the study of Neuro-Symbolic Continual Learning and showed that knowledge, although useful, can be insufficient to prevent acquiring reasoning shortcuts that compromise concept semantics and cross-task transfer. Our approach, COOL, acquires and preserves high-quality concepts, attaining better concepts and performance than existing CL strategies in three new NeSy-CL benchmarks. 

## **Acknowledgements** 

We acknowledge the support of the MUR PNRR project FAIR - Future AI Research (PE00000013) funded by the NextGenerationEU. The research of AP and ST was partially supported by TAILOR, a project funded by EU Horizon 2020 research and innovation program under GA No 

8 

**Neuro-Symbolic Continual Learning** 

952215. We acknowledge the CINECA award under the ISCRA initiative, for the availability of high performance computing resources and support. The research of SC was partially supported by Italian Ministerial grant PRIN 2020 “LEGO.AI: LEarning the Geometry of knOwledge in AI systems”, n. 2020TA3K9N. The research of EF was partially supported by the European Union’s Horizon 2020 research and innovation program DECIDER under Grant Agreement 965193. We acknowledge Angelo Porrello for his useful discussion with us. 

## **References** 

- Ahmed, K., Teso, S., Chang, K.-W., Van den Broeck, G., and Vergari, A. Semantic Probabilistic Layers for NeuroSymbolic Learning. In _NeurIPS_ , 2022a. 

- Ahmed, K., Wang, E., Chang, K.-W., and Van den Broeck, G. Neuro-symbolic entropy regularization. In _UAI_ , 2022b. 

- Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., and Tuytelaars, T. Memory aware synapses: Learning what (not) to forget. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , September 2018. 

- Aljundi, R., Lin, M., Goujaud, B., and Bengio, Y. Gradient based sample selection for online continual learning. _Advances in neural information processing systems_ , 32, 2019. 

- Blender Online Community. _Blender - a 3D modelling and rendering package_ . Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 

- Boschini, M., Bonicelli, L., Buzzega, P., Porrello, A., and Calderara, S. Class-incremental continual learning into the extended der-verse. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022. 

- Buzzega, P., Boschini, M., Porrello, A., Abati, D., and Calderara, S. Dark experience for general continual learning: a strong, simple baseline. _Advances in neural information processing systems_ , 33:15920–15930, 2020. 

- Caccia, L., Aljundi, R., Asadi, N., Tuytelaars, T., Pineau, J., and Belilovsky, E. New insights on reducing abrupt representation change in online continual learning. _ICLR_ , 2022. 

- Castro, F. M., Mar´ın-Jimenez, M. J., Guil, N., Schmid, C.,´ and Alahari, K. End-to-end incremental learning. In _Proceedings of the European conference on computer vision (ECCV)_ , pp. 233–248, 2018. 

- Chen, Z., Bei, Y., and Rudin, C. Concept whitening for interpretable image recognition. _Nature Machine Intelligence_ , 2020. 

- Choi, Y., Vergari, A., and Van den Broeck, G. Probabilistic circuits: A unifying framework for tractable probabilistic models. _UCLA_ , 2020. 

- Darwiche, A. and Marquis, P. A knowledge compilation map. _Journal of Artificial Intelligence Research_ , 17:229– 264, 2002. 

- De Lange, M. and Tuytelaars, T. Continual prototype evolution: Learning online from non-stationary data streams. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 8250–8259, 2021. 

- De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., and Tuytelaars, T. A continual learning survey: Defying forgetting in classification tasks. _PAMI_ , 2021. 

- De Raedt, L. and Kimmig, A. Probabilistic (logic) programming concepts. _Machine Learning_ , 2015. 

- De Raedt, L., Dumanciˇ c, S., Manhaeve, R., and Marra, G.´ From statistical relational to neural-symbolic artificial intelligence. In _Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence_ , pp. 4943–4950, 2021. 

- Diligenti, M., Gori, M., Maggini, M., and Rigutini, L. Bridging logic and kernel machines. _Machine learning_ , 86(1): 57–88, 2012. 

- Donadello, I., Serafini, L., and Garcez, A. D. Logic tensor networks for semantic image interpretation. In _IJCAI_ , 2017. 

- Fischer, M., Balunovic, M., Drachsler-Cohen, D., Gehr, T., Zhang, C., and Vechev, M. Dl2: Training and querying neural networks with logic. In _International Conference on Machine Learning_ , pp. 1931–1941. PMLR, 2019. 

- Giannini, F., Diligenti, M., Gori, M., and Maggini, M. On a convex logic fragment for learning and reasoning. _IEEE Transactions on Fuzzy Systems_ , 2018. 

- Giunchiglia, E. and Lukasiewicz, T. Coherent hierarchical multi-label classification networks. _NeurIPS_ , 2020. 

- Gruber, T. R. Toward principles for the design of ontologies used for knowledge sharing? _International journal of human-computer studies_ , 43(5-6):907–928, 1995. 

- Hoernle, N., Karampatsis, R. M., Belle, V., and Gal, K. Multiplexnet: Towards fully satisfied logical constraints in neural networks. In _AAAI_ , 2022. 

- Hou, S., Pan, X., Loy, C. C., Wang, Z., and Lin, D. Learning a unified classifier incrementally via rebalancing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 831–839, 2019. 

9 

**Neuro-Symbolic Continual Learning** 

- Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. _ArXiv_ , abs/1704.04861, 2017. 

- Huang, J., Li, Z., Chen, B., Samel, K., Naik, M., Song, L., and Si, X. Scallop: From probabilistic deductive databases to scalable differentiable reasoning. _NeurIPS_ , 2021. 

- Huszar, F.´ Note on the quadratic penalties in elastic weight consolidation. _Proceedings of the National Academy of Sciences_ , 115(11):E2496–E2497, 2018. 

- Johnson, J., Hariharan, B., Van Der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _CVPR_ , 2017. 

- Kambhampati, S., Sreedharan, S., Verma, M., Zha, Y., and Guan, L. Symbols as a Lingua Franca for Bridging Human-AI Chasm for Explainable and Advisable AI Systems. In _Proceedings of Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI)_ , 2022. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Bengio, Y. and LeCun, Y. (eds.), _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. 

- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. _Proceedings of the national academy of sciences_ , 114(13):3521–3526, 2017. 

- Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., and Liang, P. Concept bottleneck models. In _ICML_ , 2020. 

- Lapuschkin, S., Waldchen,¨ S., Binder, A., Montavon, G., Samek, W., and Muller, K.-R. Unmasking clever hans pre-¨ dictors and assessing what machines really learn. _Nature communications_ , 10(1):1–8, 2019. 

- LeCun, Y. The mnist database of handwritten digits. _http://yann. lecun. com/exdb/mnist/_ , 1998. 

- Li, Z. and Hoiem, D. Learning without forgetting. _IEEE transactions on pattern analysis and machine intelligence_ , 2017. 

- Li, Z. and Søgaard, A. Qlevr: A diagnostic dataset for quantificational language and elementary visual reasoning. In _Findings of NAACL_ , 2022. 

- Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Scholkopf,¨ B., and Bachem, O. Challenging common assumptions in the unsupervised learning of disentangled representations. In _ICML_ , 2019. 

- Locatello, F., Tschannen, M., Bauer, S., Ratsch,¨ G., Scholkopf,¨ B., and Bachem, O. Disentangling factors of variations using few labels. In _International Conference on Learning Representations_ , 2020. 

- Manhaeve, R., Dumancic, S., Kimmig, A., Demeester, T., and De Raedt, L. DeepProbLog: Neural Probabilistic Logic Programming. _NeurIPS_ , 2018. 

- Manhaeve, R., Marra, G., and De Raedt, L. Approximate inference for neural probabilistic logic programming. In _KR_ , 2021. 

- Marconato, E., Bontempo, G., Teso, S., Ficarra, E., Calderara, S., and Passerini, A. Catastrophic forgetting in continual concept bottleneck models. In _Image Analysis and Processing. ICIAP 2022 Workshops: ICIAP International Workshops, Lecce, Italy, May 23–27, 2022, Revised Selected Papers, Part II_ , pp. 539–547. Springer, 2022a. 

- Marconato, E., Passerini, A., and Teso, S. Glancenets: Interpretabile, leak-proof concept-based models. _NeurIPS_ , 2022b. 

- Misino, E., Marra, G., and Sansone, E. VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming. _NeurIPS_ , 2022. 

- Parascandolo, G., Neitz, A., ORVIETO, A., Gresele, L., and Scholkopf, B.¨ Learning explanations that are hard to vary. In _International Conference on Learning Representations_ , 2020. 

- Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., and Wermter, S. Continual lifelong learning with neural networks: A review. _Neural Networks_ , 113:54–71, 2019. 

- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- Qu, H., Rahmani, H., Xu, L., Williams, B., and Liu, J. Recent advances of continual learning in computer vision: An overview. _arXiv preprint arXiv:2109.11369_ , 2021. 

- Rebuffi, S.-A., Kolesnikov, A., Sperl, G., and Lampert, C. H. icarl: Incremental classifier and representation learning. In _Proceedings of the IEEE conference on Computer Vision and Pattern Recognition_ , pp. 2001–2010, 2017. 

10 

**Neuro-Symbolic Continual Learning** 

- Ren, S., He, K., Girshick, R., and Sun, J. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in neural information processing systems_ , 28, 2015. 

- Riemer, M., Cases, I., Ajemian, R., Liu, M., Rish, I., Tu, Y., and Tesauro, G. Learning to learn without forgetting by maximizing transfer and minimizing interference. 2019. 

- Robins, A. Catastrophic forgetting, rehearsal and pseudorehearsal. _Connection Science_ , 1995. 

- Ross, A. S., Hughes, M. C., and Doshi-Velez, F. Right for the right reasons: training differentiable models by constraining their explanations. In _Proceedings of the 26th International Joint Conference on Artificial Intelligence_ , pp. 2662–2670, 2017. 

- Rost, B., Liu, J., Nair, R., Wrzeszczynski, K. O., and Ofran, Y. Automatic prediction of protein function. _Cellular and Molecular Life Sciences CMLS_ , 60(12):2637–2650, 2003. 

   - Winters, T., Marra, G., Manhaeve, R., and De Raedt, L. DeepStochLog: Neural Stochastic Logic Programming. In _AAAI_ , 2022. 

   - Wu, Y., Chen, Y., Wang, L., Ye, Y., Liu, Z., Guo, Y., and Fu, Y. Large scale incremental learning. In _CVPR_ , 2019. 

   - Xu, J., Zhang, Z., Friedman, T., Liang, Y., and Broeck, G. A semantic loss function for deep learning with symbolic knowledge. In _ICML_ , 2018. 

   - Xu, Y., Yang, X., Gong, L., Lin, H.-C., Wu, T.-Y., Li, Y., and Vasconcelos, N. Explainable object-induced action decision for autonomous vehicles. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 9523–9532, 2020. 

   - Zenke, F., Poole, B., and Ganguli, S. Continual learning through synaptic intelligence, 2017. 

- Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Hadsell, R. Progressive neural networks. _arXiv preprint arXiv:1606.04671_ , 2016. 

- Stammer, W., Schramowski, P., and Kersting, K. Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting with their Explanations. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 3619–3629, 2021. 

- Teso, S., Alkan, O.,[¨] Stammer, W., and Daly, E. Leveraging explanations in interactive machine learning: An overview. _arXiv preprint arXiv:2207.14526_ , 2022. 

- van de Ven, G. M., Tuytelaars, T., and Tolias, A. S. Three types of incremental learning. _Nature Machine Intelligence_ , pp. 1–13, 2022. 

- van Krieken, E., Acar, E., and van Harmelen, F. Analyzing differentiable fuzzy logic operators. _Artificial Intelligence_ , 2022a. 

- van Krieken, E., Thanapalasingam, T., Tomczak, J. M., van Harmelen, F., and Teije, A. t. A-nesi: A scalable approximate method for probabilistic neurosymbolic inference. _arXiv preprint arXiv:2212.12393_ , 2022b. 

- Vergari, A., Choi, Y., Liu, A., Teso, S., and Van den Broeck, G. A compositional atlas of tractable circuit operations for probabilistic inference. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- Vitter, J. S. Random sampling with a reservoir. _ACM Transactions on Mathematical Software (TOMS)_ , 11(1): 37–57, 1985. 

11 

**Neuro-Symbolic Continual Learning** 

## **A. Proofs** 

## **A.1. Proof of Theorem 1** 

Taking Equation (5) as reference, the log-likelihood of _D_ can be rewritten as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-04.png)


Here, the inner product runs over all possible values of **c** . To see what the optima of this quantity look like, fix a single example ( **x** _,_ **y** ) _∈D_ and let _C_ **y** be the set of values **c** that satisfy the knowledge K[ **Y** _/_ **y** ] and _C_ **y** ¯ be those that violate the knowledge. The likelihood of ( **x** _,_ **y** ) amounts to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-06.png)


The inner product is maximized whenever _pθ_ ( **C** _|_ **x** ) allocates _all_ probability mass to values **c** that satisfy K[ **Y** _/_ **y** ], because the remaining ones do not contribute anything to the likelihood. Hence, in order to maximize Equation (8) it is sufficient that, for every ( **x** _,_ **y** ) _∈D_ , _pθ_ ( **C** _|_ **x** ) assigns zero probability to all concept configurations **c** that are ruled out by the knowledge K[ **Y** _/_ **y** ]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-08.png)


**Remarks:** This result essentially states that DeepProbLog’s reasoning layer has the same effect as the Semantic Loss (Xu et al., 2018) on the underlying neural network, and it is of independent interest. Notice that Theorem 1 also holds for non-uniform label distributions without any change to the proof. 

## **A.2. Proof of Theorem 2** 

We start by proving a general lemma: 

**Lemma 1.** _Consider tasks T_[1] _, . . . , T_[(] _[t]_[)] _and two parameter configurations φ, ψ ∈_ Θ _. If both models assign non-zero likelihood to all examples in D_[1:] _[t] , there exists a finite constant γ, depending only on the model architecture, knowledge and data, such that:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-13.png)


_Proof._ The left-hand side can be expanded to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-16.png)


Recall that, for any _a, b ∈_ [ _β, ∞_ ], it holds that _|_ log _a −_ log _b| ≤ β_[1] _[|][a][ −][b][|]_[.][For any][ (] **[x]** _[,]_ **[ y]**[)][ and task] _[ s][ ≤][t]_[, it holds that:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-18.png)


The first step follows by choosing _β_ := min( **x** _,_ **y** ) _∈D_ 1: _t_ min _θ∈{φ,ψ} pθ_ ( **y** _|_ **x** ; K[(] _[s]_[)] ) _>_ 0, the second one from Equation (5), and the last one from H¨older’s inequality. By Equation (4), the max norm of _u_ K( _s_ ) amounts to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0012-20.png)


**Neuro-Symbolic Continual Learning** 

where we chose _ζ_ = min **c** min _s≤t Z_ ( **c** ; K[(] _[s]_[)] ). Therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0013-02.png)


In order to prove Theorem 2, let _θ_[(] _[t][−]_[1)] be the parameters learned after observing _t −_ 1 tasks and _θ_ to be learned at the current iteration. Applying Lemma 1 with _φ_ = _θ_[(] _[t][−]_[1)] and _ψ_ = _θ_ to Equation (3) yields the desired result. 

**Remark:** In the worst case _ζ_ can be as small as 1, which occurs if in all tasks _s ≤ t_ the knowledge K[(] _[s]_[)] accepts a single concept configuration **c** for every example ( **x** _,_ **y** ); more commonly, _ζ_ is exponential in the number of concepts _k_ . Also, _β_ , which is the minimum likelihood attained by either _pφ_ or _pψ_ on the data sets _D_[(1)] _, . . . , D_[(] _[t]_[)] , is actively maximized during learning. 

## **A.3. What Are Correct Semantics?** 

Strictly speaking, the only concept distribution with the actual _correct semantics_ is the ground-truth distribution _p_ ( **C** _|_ **x** ). 

Our assumption is that the ground-truth concept distribution we are given is always consistent with the knowledge, in the sense that _p_ ( **C** _|_ **x** ) _|_ = K[(] _[t]_[)] [ **X** _/_ **x** _,_ **Y** _/_ **y** ] for every possible task _T_[(] _[t]_[)] = ( _D_[(] _[t]_[)] _,_ K[(] _[t]_[)] ) and example ( **x** _,_ **y** ) _∈D_[(] _[t]_[)] . In other words, we assume that the knowledge correctly reflects how the world works. Under this assumption, having the correct semantics is useful _in practice_ because, when paired with the knowledge, the ground-truth distribution by construction always yields correct labels in every possible task the learner can in principle receive. This is a form of _systematic generalization_ . 

Naturally, if the learned distribution _pθ_ ( **C** _|_ **x** ) matches the ground-truth distribution exactly, then it will also achieve systematic generalization. This condition, however, is very restrictive. Pragmatically, we can relax this requirement, and say that a distribution encodes the correct semantics if it is _indistinguishable from the ground-truth distribution in terms of what concepts it predicts_ . Formally, we say that a distribution _pθ_ ( **C** _|_ **x** ) is _semantically equivalent_ to the ground-truth distribution on data _D_ if it allows us to infer the same concept configuration for all data points, or formally: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0013-09.png)


where we used _≡_ to indicate set equivalence. This is quite intuitive: any distribution satisfying Equation (17) will yield the same concepts **c** as _p_ ( **C** _|_ **x** ) for every **x** , and therefore also the same MAP states **y** under any choice of knowledge K. 

The opposite is _not_ generally true: the knowledge K might have multiple possible solutions, in the sense that different choices of concepts **c** might yield the same label **y** . In this case, the label does not carry enough information to recover the ground-truth concepts argmax **c** _p_ ( **c** _|_ **x** ), and therefore also a concept distribution _pθ_ ( **C** _|_ **x** ) semantically equivalent to _p_ ( **C** _|_ **x** ). This is exactly what we mean by _reasoning shortcuts_ : concept distributions that achieve high performance on the observed task(s) but have no guarantee of systematically generalizing to future tasks, or more formally: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0013-12.png)


## **A.4. Reasoning Shortcuts in other NeSy Architectures** 

While Theorem 1 shows that reasoning shortcuts do affect DeepProbLog, which maximizes for label likelihood, we remark that they are a _general_ phenomenon. It is easy to see that reasoning shortcuts occur whenever the prior knowledge admits deducing the correct _label_ **y** from _concepts_ **c** that do not have the correct semantics, as this makes it is impossible for a model to distinguish between concepts with “correct” vs. “incorrect” semantics by maximizing accuracy alone. This impacts _offline_ NeSy prediction tasks and NeSy-CL problems alike; indeed, Theorem 1 makes no assumption on how the training set _D_ has been generated. 

Reasoning shortcuts are not specific to DeepProbLog. On the contrary, this situation can be triggered by a variety of other NeSy architectures, including but not limited to: 

- ( _i_ ) NeSy predictors that rely on a top reasoning layer to ensure predictions are consistent with prior knowledge, which are typically trained to maximize some surrogate of the label accuracy, including (Ahmed et al., 2022a; Giunchiglia & Lukasiewicz, 2020; Hoernle et al., 2022; Huang et al., 2021; Winters et al., 2022; van Krieken et al., 2022b). 

13 

**Neuro-Symbolic Continual Learning** 

- ( _ii_ ) NeSy predictors that rely on _relaxed_ reasoning layers obtained by softening the logical prior knowledge (Diligenti et al., 2012; Donadello et al., 2017), because this transformation usually preserves existing optima of the label accuracy. As such, it also preserves unintended optima – that is, reasoning shortcuts. 

- ( _iii_ ) Neural networks trained to maximize accuracy and consistency with prior knowledge using the Semantic Loss and similar techniques (Xu et al., 2018; Fischer et al., 2019; Ahmed et al., 2022b). In fact, Theorem 1 shows that DeepProbLog is affected by shortcuts precisely because, from the neural network’s perspective, its reasoning layer acts exactly like the Semantic Loss; see our remark in Appendix A. 

More generally, reasoning shortcuts impact NeSy tasks and architectures beyond these, at least as long as models are trained by optimizing loss functions that do not measure or correlate with concept quality. We leave a detailed analysis of specific cases to future work. 

## **B. Implementation Details** 

In this Section, we report useful details for the models and the metrics adopted in the evaluation. 

## **B.1. Hardware and Software Implementation** 

The code for the project was developed on top mammoth (Boschini et al., 2022), a well-known CL framework. We included the implementation of DeepProbLog for MNIST-Addition from VAEL (Misino et al., 2022). The generation of CLE4EVR was adapted from (Stammer et al., 2021). All experiments were implemented using Python 3 and Pytorch (Paszke et al., 2019) and run on a server with 128 CPUs, 1TiB RAM, and 8 A100 GPUs. 

## **B.2. Metrics** 

We adopted standard CL measures, namely task-incremental (Task-IL) and class-incremental (Class-IL) accuracy, applied here to both labels and concepts predictions, as well as forward transfer (FWT) and backward transfer (BWT) on the labels, see also (Buzzega et al., 2020). Below we write _T_ to indicate the last task. 

- **Class-IL** measures the average accuracy on the test sets of all tasks _T_ . In Table 1, we report Class-IL at the very last task _T_ , defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0014-11.png)


where _A_ **Y** ( _θt, s_ ) denotes the accuracy on the labels evaluated on the test set of task _s_ . Class-IL for concepts is analogous, but builds on the average accuracy over _all_ concepts. 

- **Task-IL** is the average accuracy over the test sets of all tasks up to _t_ , evaluated only on examples annotated with the classes or concepts appearing in task _t_ . The definition is identical to Equation (19) except that we mask the prediction of model so as to place mass only on the labels appearing in _D[s]_ , with _s ≤ t_ . 

- **FWT** evaluates the adaptability of the model at each time-step to the successive task. Formally, at each _t_ , FWT measures the average gain in accuracy between _θt_ and a random baseline _θ_ rand when predicting the labels of the task _t_ + 1. This can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0014-15.png)


where _θ_ rand denotes the initialized model with random weights. 

- **BWT** measures how much forgetting the model undergoes by looking at how much accuracy for each task _t_ is retained after the last task. Formally: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0014-18.png)


14 

**Neuro-Symbolic Continual Learning** 

For the sake of brevity, in the main paper we reported only Class-IL on labels and concepts, and FWT. Task-IL was omitted because it does not account for accuracy on past concepts that no longer occur in the last task, and BWT because it does not as informative as Class-IL. All results for these extra metrics on MNAdd-Seq are reported in Appendix E. 

## **B.3. Architectures & Models Details** 

MNIST-Addition: For both CBMs and DeepProbLog we adopted the same architecture for extracting concepts – henceforth, _encoder_ – and we implemented it as a standard convolutional neural network, with dropout set at 50% after each convolution module. We also inject a noise term _ϵ ∼N_ (0 _,_ 0 _._ 1) after the encoder to stabilize the overall training process. The complete structure is reported in Table 2. 

_Table 2._ CNN Encoder for MNIST-Addition 

||_Table 2._|NN Encoder forMNIST-Addition||
|---|---|---|---|
|INPUT SHAPE|LAYER TYPE|PARAMETERS|ACTIVATION|
|(28_,_28_,_1)|Convolution|depth=64, kernel=4, stride=2, padding=1|ReLU|
|(14_,_14_,_64)|Dropout|_p_= 0_._5||
|(14_,_14_,_64)|Convolution|depth=128, kernel=4, stride=2, padding=1|ReLU|
|(7_,_7_,_128)|Dropout|_p_= 0_._5||
|(7_,_7_,_128)|Convolution|depth=256, kernel=4, stride=2, padding=1|ReLU|
|(3_,_3_,_256)|Flatten|||
|(2304)|Linear|dim=10, bias = True||



In both CBMs and DeepProbLog, each input digit **x**[(] _[i]_[)] is predicted independently and mapped to a 10-dimensional bottleneck **z**[(] _[i]_[)] . Then, the two encodings are stacked together, obtaining the overall representation **z** = ( **z**[(1)] _,_ **z**[(2)] ). 

The classifier (top layer) of the CBM is designed to predict the sum the **z**[(1)] and **z**[(2)] . A simple linear layer, which is the standard choice in CBM (Koh et al., 2020), is insufficient to successfully address the task. Therefore, we implemented the classifier via a bi-linear operation on the encodings, i.e.,: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0015-08.png)


where _W[y]_ is a learnable class-specific 10 _×_ 10 tensor of real entries, and the softmax is over all classes _y ∈{_ 0 _, . . . ,_ 17 _}_ . 

Conversely, the _reasoning_ (top) layer of DeepProbLog is implemented as in Equation (5). Specifically, each **z**[(] _[i]_[)] encodes the logits of the probability _pθ_ ( _Ci|_ **x**[(] _[i]_[)] ) for the _i_ -th digit, which we convert into a categorical probability distribution through a softmax activation. 

CLE4EVR: In a first step, we used Faster-RCNN (Ren et al., 2015) to extract the bounding boxes associated to objects in all images. The bounding box predictor is a pretrained MobileNet (Howard et al., 2017) fine-tuned on training images from the first task only, using the ground-truth bounding boxes of CLE4EVR images. The bounding box predictor is kept frozen in all successive tasks. We discarded all examples **x** _i_ with less than 2 predicted bounding boxes. Concept supervision was transferred from the ground-truth bounding boxes to the predicted ones based on overlap. 

We scale each predicted bounding box to an image of size 28 _×_ 28 _×_ 3 which is then passed to the encoder, implemented once again using a CNN with dropout with _p_ = 50% after each convolution layer and normal noise _ϵ ∼N_ (0 _,_ 0 _._ 1) added to the final output. The architecture is the same as in Table 2, except that the input has depth 3 instead of 1 and that the bottleneck is 20-dimensional, with 10 dimensions allocated for the shape and 10 for the color of the input object, each with its own softmax to produce shape and color probability distributions. The DeepProbLog _reasoning_ layer is as in Equation (5). 

During inference, the prediction returns invalid ( _⊥_ ) whenever the number of predicted bounding boxes is less than 2. We counted invalid predictions as wrong predictions in all metrics evaluated in the test set. 

## **B.4. Hyper-parameter Selection** 

All continual strategies have been trained with the same number of epochs and buffer dimension. The actual values depend on the specific benchmark: 25 epochs per task and a buffer size of 1000 examples for MNAdd-Seq, 100 epochs and 1000 examples for MNAdd-Shortcut, and 50 epochs each task and 250 examples for CLE4EVR. In all experiments, we employed the Adam optimizer (Kingma & Ba, 2015) combined with exponential decay ( _γ_ = 0 _._ 95). The initial learning rate is restored at the beginning of a new task. 

15 

**Neuro-Symbolic Continual Learning** 

For each data set, we optimized the weight of the concept supervision _wc_ based on the Class-IL ( **Y** ) performance of ER using grid-search on the validation set (union of all tasks). Then, for each strategy, we selected the best learning rate and strategy-specific hyperparameters through a grid-search on the validation set, so as to optimize Class-IL ( **Y** ) on a single random seed. The learning rate was chosen from the range of [10 _[−]_[5] _,_ 10 _[−]_[2] ]. The exact values can be found in the source code. 

## **C. NeSy-CL Benchmarks** 

In this section, we provide a more detailed description of the benchmarks introduced in Section 5. 

## **C.1.** MNAdd-Seq 

We derived MNAdd-Seq from the MNIST-Addition data set of Manhaeve et al. (2018). Here, the knowledge encodes the following constraint: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0016-06.png)


for a total of 19 possible sums. MNAdd-Seq is both _label-incremental_ and _concept-incremental_ ; in each task only two possible sums appear, obtaining in total 9 tasks. Specifically: 

**Task** 1 **:** _Y ∈{_ 0 _,_ 1 _}_ and _C ∈{ , }_ ; 

**Task** 2 **:** _Y ∈{_ 2 _,_ 3 _}_ and _C ∈{ , , , }_ ; 

**Task** 3 **:** _Y ∈{_ 4 _,_ 5 _}_ and _C ∈{ , , , , , }_ ; 

**Task** 4 **:** _Y ∈{_ 6 _,_ 7 _}_ and _C ∈{ , , , , , , , }_ ; 

**Task** 5 **:** _Y ∈{_ 8 _,_ 9 _}_ and _C ∈{ , , , , , , , , , }_ ; 

**Task** 6 **:** _Y ∈{_ 10 _,_ 11 _}_ and _C ∈{ , , , , , , , , }_ ; 

**Task** 7 **:** _Y ∈{_ 12 _,_ 13 _}_ and _C ∈{ , , , , , , }_ ; 

**Task** 8 **:** _Y ∈{_ 14 _,_ 15 _}_ and _C ∈{ , , , , }_ ; 

## **Task** 9 **:** _Y ∈{_ 16 _,_ 17 _}_ and _C ∈{ , , }_ ; 

In total, the data set counts 42k training and 6k test examples. 

## **C.2.** MNAdd-Shortcut 

This benchmark is a case-study composed of two task, built considering the following constraints: 

- In the first task, we present only even digits and four possible sums: (i) + = 6, (ii) + = 10, (iii) + = 10, and (iv) + = 12. 

- In the second task, only odd numbers are considered, i.e., _C ∈{ , , , , }_ , and all their possible sums. 

The rationale behind this construction is that it makes it possible to satisfy the knowledge in both tasks by leveraging reasoning shortcuts, and specifically those described in Appendix D. 

The overall data set contains 13 _._ 8k training examples and 2k test data. We also collected an OOD test set containing examples not appearing in the training, validation and test sets, which comprise sums of odd and even digits, e.g., + = 1, and unseen combinations of even numbers, e.g., + = 16. 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-00.png)


**----- Start of picture text -----**<br>
Neuro-Symbolic Continual Learning<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-01.png)


**----- Start of picture text -----**<br>
TASK= 0 TASK = 1 TASK=2 TASK=3 TASK=4<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-16.png)


_Figure 3._ Examples images for each tasks in the CLE4EVR benchmark. 

## **C.3.** CLE4EVR 

Following Stammer et al. (2021), the CLE4EVR data set was generated using Blender (Blender Online Community, 2018), using additional objects from (Li & Søgaard, 2022) as well as three custom shapes. 

We generated different objects with 10 possible shapes and colors, and with 2 free variations on material _{_ `rubber` _,_ `metal` _}_ and size _{_ `small` _,_ `big` _}_ , that play no role in determining the label. Each image is of size 256 _×_ 430 _×_ 3, and contains two non-overlapping objects over a light-gray background. 

Table 3 reports what combinations of objects appear in each task. All tasks are composed of 1 _._ 1k training examples, 100 validation examples, and 500 test examples. Each task includes only two possible shapes (out of ten) and two colors (out of ten), without any overlap between tasks. An illustration is given in Figure 3. 

The knowledge K = K _[′] ∧_ K _[′′] ∧_ K _[′′′]_ encodes the following constraints: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0017-23.png)


with three output variables _Y_ 1 = `same` ~~`s`~~ `hape` , _Y_ 2 = `same` ~~`c`~~ `olor` and _Y_ 3 = `same` , giving rise to four mutually exclusive classes: 0 = different shape and color, 1 = same shape and different color, 2 = different shape and same color, 3 = same shape and same color. 

We also generated an additional OOD test set, comprising 300 images depicting _unseen combination_ of training objects, e.g., `redsquares` and `pinkdiamonds` (which occur in none of the tasks). All OOD examples all have label **Y** = (0 _,_ 0 _,_ 0). 

All generated images come with ground-truth bounding boxes annotated with the properties (i.e., concepts) of the objects they contain, as well as annotations for _Y_ 1, _Y_ 2, and _Y_ 3. The concept annotations are transferred to the bounding boxes predicted by Faster R-CNN during pre-processing, cf. Appendix B. 

_Table 3._ Task Organization in CLE4EVR 

||_Table 3._ Task Org|anization inCLE4EVR|
|---|---|---|
|TASK|COLORS|SHAPES|
|1|`red`,`gray`|`sphere`,`cube`|
|2|`green`,`blue`|`cylinder`,`tetrahedron`|
|3|`brown`,`purple`|`cone`,`triangular prism`|
|4|`yellow`,`cyan`|`pyramid`,`toroid`|
|5|`orange`,`pink`|`diamond`,`star prism`|



17 

**Neuro-Symbolic Continual Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0018-01.png)


**----- Start of picture text -----**<br>
OFFLINE @ 0% REPLAY METHODS @ 10%<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0018-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0018-03.png)


_Figure 4._ Confusion matrices of the learned concepts at the end of the very last task, for a single run. **Left** : The confusion matrix on concepts for OFFLINE without concept supervision. It shows that it is very likely to opt for a reasoning shortcut. **Right** : Confusion matrices on a single run for all replay-based methods. Here, COOL is pictured in **red** , while DER, DER++, and ER in shades of **blue** . Only COOL retains the correct semantics of the concepts, whereas the other replay methods are very likely to pick the shortcut of OFFLINE. 

## **D. Shortcut Solutions in** MNAdd-Shortcut **and** CLE4EVR 

In this section, we provide a more detailed account on the shortcut solutions for the continual scenarios introduced. 

## **D.1. Reasoning Shortcuts Due to Low-Level Correlations** 

Before proceeding, we observe that simply predicting multiple concepts jointly by a single neural network is sufficient to enable reasoning shortcuts. Intuitively, this happens because, since the network has access to all properties of all objects, it can automatically exploit correlations between them to satisfy the knowledge without the need for extracting any “proper” concepts. For instance, in MNIST-Addition the knowledge can simply group pairs of digits (both of which it has access to) into two single combinations of concepts that always yield the right labels. To see this, consider the following example: 

**Example 4.** _Consider a single_ MNIST-Addition _task consisting of digits summing to either_ 2 _or_ 3 _. By Theorem 1,_ Θ _[∗]_ (K _, D_ ) _contains φ with pφ_ ( **C** _|_ **x** ) _̸≡ p_ ( **C** _|_ **x** ) _, such that:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0018-10.png)


_This distribution fits the data perfectly and is consistent with the knowledge, and thus cannot be distinguished from the ground-truth distribution based on likelihood alone._ 

Notice that the two learned concepts ignore the value of the individual digits, and rather depend on the correlation between them. in MNIST-Addition, it is straightforward to avoid this situation _by construction_ by simply processing the two digits independently. The same can be done when processing objects in CLE4EVR. This partially motivates our choice of neural architecture, described in Appendix B. More precisely, in MNIST-Addition the adopted architecture factorizes the joint probability distribution on the concepts in: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0018-13.png)


While this is sufficient to guarantee independence among distinct objects, the prior knowledge can still admit reasoning shortcuts. Next, we describe the concrete reasoning shortcuts appearing in MNAdd-Shortcut and CLE4EVR. 

## **D.2. Shortcuts in** MNAdd-Shortcut 

In order to understand reasoning shortcuts in MNAdd-Shortcut, it is useful to view the sums as constraints on the possible values attributed to each concept _Cj_ . From this perspective, reasoning shortcuts occur whenever the combinations of digits present in the training data (which are five in each task) are insufficient to uniquely determine the four possible sums. 

18 

**Neuro-Symbolic Continual Learning** 

In particular, the problem of assigning the intended semantics of each learned concept can be expressed as a system of 4 linear equations with 5 variables, which in task 1 of MNAdd-Seq can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0019-02.png)


The first equation states that whatever values are assigned to the concepts that fire when a and a are present in the input **x** , have to satisfy the condition that their sum is 6 (in all examples in which they appear). It should be clear that the linear system is undetermined and infinitely many real solutions can be found for _ci_ , all of which except one do not capture the intended semantics of digits. 

One of these unintended solutions is very often picked by label-replay strategies. Specifically, the input can be easily confused for a 9, due to input similarity, which brings the model towards the assignment _c_ 4 = 9. This immediately yields the following unintentional mappings for all other digits: _c_ 0 = 5, _c_ 2 = 7, _c_ 6 = 1, and _c_ 8 = 3. This reasoning shortcut is often found when training offline on this task and also when training on both tasks of MNAdd-Shortcut, as done in our experiments, by ER, DER, and DER++, as shown in Figure 4. 

## **D.3. Shortcuts in** CLE4EVR 

Several shortcut exist in CLE4EVR. Recall that: 

- CLE4EVR is composed of 5 tasks, with four possible outcomes ( _different objects_ , _same shapes_ , _same colors_ , and _same objects_ ). 

- In each task only two possible shapes and colors are observed, and are never seen again in other tasks. 

In order to correctly classify the `same` ~~`c`~~ `olor` and `same` ~~`s`~~ `hape` labels, the model must acquire at least two distinct concepts for shape and two distinct concepts for colors in each task, but the knowledge provides little guidance as to _which_ shapes or colors need to be associated to the input objects. This leaves ample room for reasoning shortcuts. 

Let us focus on shapes only. Letting _S_ be the set of the 10 possible shapes _si_ , the model needs at least 10 _·_ (10 _−_ 1) _/_ 2 different negative examples of the knowledge to uniquely identify all possible shapes (up to permutation), one for each pair of mismatching shapes. Consider the map _π_ : _S →S_ mapping from ground-truth shape of the input object to the learned concept for shape. Ideally, we would like _π_ to be injective, so that no two distinct ground-truth shapes are mapped to the same learned shape. 

Consider a task with 4 possible shapes _s_ 1 = `cube` , _s_ 2 = `cone` , _s_ 3 = `cylinder` , and _s_ 4 = `toroid` . In order to guarantee injectivity, the data has to include enough combinations of shapes so that the map _π_ satisfies the following condition: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0019-12.png)


Notice that this map _is_ injective, in the sense that _si_ = _sj_ = _⇒ π_ ( _si_ ) _̸_ = _π_ ( _sj_ ) _, ∀i ̸_ = _j_ . All tasks in CLE4EVR, however, are designed to distinguishing between only _two_ possible shapes (and colors), hence the condition in Equation (28) is never satisfied. This is what allows for reasoning shortcuts. 

As a matter of fact, without concept supervision, all values for shapes and colors are equally likely to be picked up to solve the task. We observed this phenomenon in the case of 0% supervision reported in Figure 5. 

19 

**Neuro-Symbolic Continual Learning** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0020-01.png)


**----- Start of picture text -----**<br>
SHAPES COLORS LABELS<br>PREDICTED PREDICTED PREDICTED<br>RUTH<br>-T<br>ROUND<br>G<br>**----- End of picture text -----**<br>


_Figure 5._ Confusion matrices for COOL (in **red** ) and the other replay strategies (in shades of **blue** ) in CLE4EVR with no concept supervision, at the end of all tasks. All methods fails to attribute the correct semantics to the concepts and learn, instead, a shortcut which optimizes the Class-IL ( **Y** ). 

|NA¨IVE<br>ER<br>DER<br>DER++<br>COOL|MNAdd-Seq<br>(BUFFER SIZE= 1000)<br>TIME PER EPOCH<br>MEMORY OCCUPATION<br>0.238 s<br>-<br>0.385 s<br>6.280 Mb<br>0.475 s<br>6.348 Mb<br>0.667 s<br>6.356 Mb<br>0.476 s<br>6.360 Mb|CLE4EVR<br>(BUFFER SIZE= 250)|
|---|---|---|
|||TIME PER EPOCH<br>MEMORY OCCUPATION<br>0.335 s<br>-<br>0.344 s<br>1.570 Mb<br>0.412 s<br>1.587 Mb<br>0.482 s<br>1.589 Mb<br>0.439 s<br>1.590 Mb|



_Table 4._ **Time per epoch and memory occupation** for MNIST-Addition and CLE4EVR. 

## **E. Additional Results and Metrics** 

## **E.1. Time Overhead and Memory Occupation of COOL** 

We evaluate the time overhead and the memory requirements of COOL compared to other replay-based strategies in Table 4. All these strategies impose an additional time overhead due to the selection and replay of past examples. 

We measured the time required for completing a single epoch in the first task of MNIST-Addition and CLE4EVR. NA¨IVE provides the lower bound for the computation time, as it just fine-tunes the model on the new examples. ER in ¨ MNIST-Addition strategy requires almost _∼_ 0 _._ 1 s more than NAIVE for storing and replaying. In CLE4EVR, however, the gap is less evident between the two strategies. For MNAdd-Seq, the time per epoch of COOL amounts to _∼_ 0 _._ 476 s, which is comparable to that of DER _∼_ 0 _._ 475 s and lower than that of DER++ _∼_ 0 _._ 667 s. For CLE4EVR COOL requires 0 _._ 439 s per epoch, slightly increasing compared to DER _∼_ 0 _._ 412 s, but still being lower w.r.t. DER++ _∼_ 0 _._ 482 s. 

In terms of memory occupation, COOL requires slightly more memory than other replay strategies, due to the need of storing the _logits_ of the learned concepts. However, most of the overhead is due to storing the instances **x** themselves, which is the very same for all strategies and amounts to 6 _._ 272 Mb for MNIST-Addition and 1 _._ 568 Mb for CLE4EVR. 

## **E.2.** MNAdd-Seq 

We report in Table 5 results for all competing strategies and performance measures used (including those omitted in the main text), namely Class-IL and Task-IL on labels and concepts (denoted as **Y** and **C** , respectively), backward transfer (BWT), and forward transfer (FWT). 

These results confirm the ones reported in Section 5. Specifically, COOL attains higher performance w.r.t. all metrics in both CBM and DeepProbLog. They also show that COOL outperforms all other approaches in terms of BWT when paired with DeepProbLog, and is the runner-up with CBMs. 

The method with the best BWT on CBM is LWF, which however displays a pathological learning behavior, as made clear by the fact that it is the only method to have _negative_ forward transfer. The reason is that LWF struggles to learn the first few tasks properly, but performs reasonably on the latter ones. 

20 

**Neuro-Symbolic Continual Learning** 

|CBM @10%<br>DEEPPROBLOG@0%|STRATEGY|LABELS(**Y**)<br>CLASS-IL (_↑_)<br>TASK-IL (_↑_)|CONCEPTS(**C**)<br>CLASS-IL (_↑_)<br>TASK-IL (_↑_)||
|---|---|---|---|---|
|||||BTW (_↑_)<br>FWT (_↑_)|
||NA¨IVE<br>RESTART<br>LWF<br>EWC<br>ER<br>DER<br>DER++<br>COOL|11_._71_±_0_._8<br>27_._5_±_3_._9<br>10_._78_±_0_._1<br>32_._9_±_2_._1<br>18_._08_±_1_._8<br>55_._7_±_5_._3<br>11_._57_±_0_._6<br>38_._3_±_11_._2<br>13_._29_±_0_._4<br>32_._5_±_3_._2<br>18_._63_±_2_._5<br>50_._3_±_3_._4<br>18_._17_±_1_._6<br>48_._9_±_2_._4<br>**38****_._0****_±_ 1****_._9**<br>**78****_._4****_±_ 3****_._8**|36_._2_±_2_._6<br>50_._8_±_1_._7<br>29_._7_±_0_._1<br>43_._1_±_0_._6<br>63_._2_±_4_._4<br>78_._6_±_2_._5<br>37_._4_±_0_._6<br>52_._1_±_1_._7<br>43_._5_±_2_._0<br>67_._7_±_3_._3<br>53_._1_±_1_._7<br>73_._1_±_2_._0<br>54_._1_±_3_._0<br>73_._1_±_4_._5<br>**78****_._1****_±_ 2****_._5**<br>**90****_._4****_±_ 1****_._6**|_−_91_._1_±_1_._2<br>7_._5_±_0_._3<br>_−_98_._2_±_0_._1<br>7_._3_±_0_._2<br>**_−_18****_._5****_±_ 1****_._8**<br>_−_4_._7_±_1_._1<br>_−_90_._4_±_0_._8<br>7_._6_±_0_._4<br>_−_88_._0_±_0_._7<br>13_._4_±_1_._6<br>_−_88_._6_±_2_._9<br>15_._7_±_0_._9<br>_−_89_._2_±_1_._8<br>16_._6_±_1_._8<br>_−_68_._3_±_2_._1<br>**29****_._0****_±_ 4****_._8**|
||NA¨IVE<br>RESTART<br>LWF<br>EWC<br>ER<br>DER<br>DER++<br>COOL|6_._9_±_0_._2<br>7_._6_±_0_._2<br>9_._6_±_0_._3<br>22_._2_±_0_._8<br>6_._8_±_0_._5<br>11_._0_±_1_._6<br>6_._8_±_0_._4<br>7_._8_±_0_._4<br>44_._3_±_9_._7<br>81_._2_±_6_._5<br>68_._3_±_9_._4<br>94_._8_±_3_._2<br>62_._2_±_5_._4<br>93_._5_±_2_._1<br>**71****_._9****_±_ 2****_._9**<br>**96****_._6****_±_ 0****_._8**|6_._7_±_0_._4<br>16_._4_±_1_._0<br>0_._2_±_0_._1<br>11_._6_±_0_._5<br>10_._8_±_4_._6<br>21_._7_±_8_._3<br>7_._8_±_0_._6<br>18_._3_±_0_._6<br>62_._0_±_8_._6<br>82_._8_±_5_._1<br>81_._3_±_6_._9<br>93_._8_±_3_._0<br>77_._1_±_4_._2<br>92_._8_±_2_._9<br>**84****_._5****_±_ 1****_._9**<br>**95****_._4****_±_ 0****_._4**|_−_63_._1_±_0_._6<br>6_._2_±_0_._2<br>_−_69_._5_±_1_._9<br>6_._9_±_0_._8<br>_−_71_._2_±_5_._2<br>18_._3_±_0_._2<br>_−_62_._9_±_0_._4<br>6_._1_±_0_._3<br>_−_48_._3_±_8_._8<br>8_._2_±_4_._1<br>_−_30_._5_±_8_._5<br>44_._5_±_23_._7<br>_−_34_._8_±_5_._6<br>27_._1_±_5_._2<br>**_−_28****_._7****_±_ 3****_._2**<br>**83****_._2****_±_ 0****_._9**|



_Table 5._ **Additional results for** MNAdd-Seq **.** 

Surprisingly, this oddball behavior is sufficient for LWF to beat the baselines (NA¨IVE and RESTART) in terms of Class-IL **Y** (at around 18%), but not enough to outperform COOL, and also yields the aforementioned issue with FWT. We stress that this implies that LWF generalizes to forward tasks _worse than random_ . 

## **E.3.** MNAdd-Shortcut 

We report here in Table 6, all results obtained for all strategies in MNAdd-Shortcut. We performed the comparison adopting only DeepProbLog with increasing amount of concept supervision. 

|NA¨IVE<br>RESTART<br>EWC<br>LWF<br>ER<br>DER<br>DER++<br>COOL|SUPERVISION0%<br>CLASS-IL(**Y**)<br>CLASS-IL(**C**)<br>OOD (**Y**)<br>59.9_±_3.2<br>49.4_±_0.4<br>10.7_±_2.3<br>59.1_±_0.6<br>**49.7**_±_**0.1**<br>**12.9**_±_**0.4**<br>55.9_±_8.9<br>45.6_±_11.4<br>12.5_±_1.4<br>**62.2**_±_**1.8**<br>49.1_±_0.2<br>11.1_±_1.2<br>**45.0**_±_**24.7**<br>23.8_±_19.1<br>3.0_±_2.5<br>41.4_±_13.1<br>19.6_±_17.4<br>7.9_±_1.7<br>36.3_±_21.7<br>23.6_±_18.0<br>3.7_±_3.5<br>38.8_±_26.3<br>24.1_±_18.1<br>4.9_±_2.0|SUPERVISION1%<br>CLASS-IL(**Y**)<br>CLASS-IL(**C**)<br>OOD (**Y**)<br>57.5_±_0.5<br>49.2_±_0.1<br>12.9_±_0.4<br>58.1_±_1.6<br>49.3_±_0.1<br>12.9_±_0.5<br>59.2_±_1.5<br>49.5_±_0.1<br>12.6_±_0.8<br>55.8_±_1.4<br>48.7_±_0.1<br>12.5_±_0.7<br>70.0_±_7.4<br>48.9_±_0.3<br>7.6_±_1.5<br>76.2_±_1.9<br>49.3_±_0.1<br>7.3_±_0.4<br>**76.8**_±_**7.3**<br>49.0_±_0.6<br>5.9_±_0.5<br>67.9_±_2.0<br>**77.67**_±_**1.9**<br>**53.2**_±_**3.9**|SUPERVISION10%|
|---|---|---|---|
||||CLASS-IL(**Y**)<br>CLASS-IL(**C**)<br>OOD (**Y**)<br>59.9_±_0.9<br>49.4_±_0.1<br>12.3_±_0.5<br>59.1_±_0.9<br>49.6_±_0.1<br>13.2_±_0.6<br>59.7_±_0.8<br>49.6_±_0.1<br>12.4_±_0.3<br>58.0_±_1.2<br>49.2_±_0.1<br>12.6_±_0.6<br>77.6_±_1.8<br>49.3_±_0.1<br>6.9_±_0.5<br>77.5_±_3.6<br>49.5_±_0.1<br>6.8_±_0.9<br>**82.1**_±_**7.1**<br>49.4_±_0.2<br>5.0_±_1.6<br>70.7_±_2.1<br>**80.2**_±_**1.9**<br>**57.1**_±_**3.9**|



_Table 6._ MNAdd-Shortcut Additional results. 

With 0% concept supervision, all methods perform poorly, i.e., worse or comparably to NA¨IVE and RESTART in terms of Class-IL. Variance is also quite large for EWC, ER, DER, DER++, and COOL. The sole exception is LWF, which obtains higher performance and small variance in label classification. On the other hand, the results in OOD accuracy are all below 13%, indicating that no strategy can successfully identify high-quality concepts that transfer across tasks. 

The picture at 1% and 10% supervision is very similar: COOL outperforms all approaches in terms of concept quality and OOD accuracy by a large margin, while the other approaches pick up the reasoning shortcut, thus achieving high label accuracy only. 

## E.3.1. CLE4EVR 

The complete results for CLE4EVR, reported in Table 7, show the same trend as those on MNAdd-Shortcut. For completeness, we report the evolution across tasks of concept confusion matrices for all methods in Figure 6. The impact of the reasoning shortcut on DER, and its lack of impact on COOL, are clearly visible. 

21 

**Neuro-Symbolic Continual Learning** 

|NA¨IVE<br>RESTART<br>EWC<br>LWF<br>ER<br>DER<br>DER++<br>COOL|SUPERVISION0%<br>CLASS-IL(**Y**)<br>CLASS-IL(**C**)<br>OOD (**Y**)<br>44.9_±_0.7<br>9.2_±_1.2<br>25.0_±_0.9<br>39.1_±_0.9<br>10.5_±_1.4<br>11.1_±_3.7<br>41.5_±_7.1<br>8.3_±_3.2<br>15.7_±_8.4<br>47.1_±_0.8<br>6.6_±_2.2<br>27.2_±_2.8<br>85.3_±_0.3<br>10.2_±_5.5<br>26.3_±_3.5<br>77.7_±_1.4<br>7.7_±_2.8<br>**27.5**_±_**2.7**<br>**85.5**_±_**0.2**<br>11.1_±_4.4<br>23.2_±_1.8<br>70.7_±_5.7<br>8.5_±_4.5<br>25.6_±_2.3|SUPERVISION1%<br>CLASS-IL(**Y**)<br>CLASS-IL(**C**)<br>OOD (**Y**)<br>43.5_±_2.6<br>19.9_±_2.6<br>27.9_±_11.4<br>39.9_±_1.0<br>17.8_±_0.1<br>14.8_±_4.1<br>38.5_±_6.4<br>17.2_±_1.4<br>23.5_±_8.6<br>41.9_±_1.8<br>19.1_±_2.9<br>29.0_±_15.2<br>72.8_±_4.9<br>20.5_±_3.9<br>28.8_±_9.9<br>73.3_±_3.0<br>19.1_±_1.9<br>24.9_±_3.4<br>83.0_±_1.1<br>20.9_±_3.9<br>32.4_±_12.1<br>**83.2**_±_**0.5**<br>**85.9**_±_**2.9**<br>**91.1**_±_**1.5**|SUPERVISION10%|
|---|---|---|---|
||||CLASS-IL(**Y**)<br>CLASS-IL(**C**)<br>OOD (**Y**)<br>39.9_±_1.6<br>17.8_±_0.1<br>16.9_±_3.6<br>39.2_±_1.1<br>17.9_±_0.1<br>13.7_±_3.8<br>41.8_±_1.8<br>17.8_±_0.1<br>16.8_±_4.3<br>39.5_±_7.5<br>17.4_±_3.9<br>44.8_±_24.3<br>66.8_±_8.4<br>18.3_±_1.3<br>23.3_±_4.0<br>75.1_±_3.9<br>19.2_±_2.0<br>26.1_±_5.0<br>81.3_±_2.3<br>20.1_±_3.9<br>28.4_±_9.3<br>**85.2**_±_**0.3**<br>**87.9**_±_**0.1**<br>**91.9**_±_**0.2**|



_Table 7._ CLE4EVR Full results 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal_images/icml_2023_Neuro-Symbolic_Continual_Learning_Knowledge,_Reasoning_Shortcuts_and_Concept_Rehearsal.pdf-0022-03.png)


**----- Start of picture text -----**<br>
task= 1 task= 2 task= 3 task= 4 task= 5<br>shapes<br>colors<br>**----- End of picture text -----**<br>


_Figure 6._ Dynamics of confusion matrices for shapes and colors encodings on CLE4EVR with 10% concept supervision. COOL in **red** preserves the correct concepts, while DER (in **blue** ) suffers for shortcuts. Shape and color range in _{_ 0 _, . . . ,_ 9 _}_ . 

22 

