# **Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts** 

**Emanuele Marconato** DISI and DI University of Trento and University of Pisa Trento, Italy `emanuele.marconato@unitn.it` 

**Stefano Teso** CIMeC and DISI University of Trento Trento, Italy `stefano.teso@unitn.it` 

**Antonio Vergari Andrea Passerini** School of Informatics DISI University of Edinburgh University of Trento Edinburgh, UK Trento, Italy `avergari@exseed.ed.ac.uk andrea.passerini@unitn.it` 

## **Abstract** 

Neuro-Symbolic (NeSy) predictive models hold the promise of improved compliance with given constraints, systematic generalization, and interpretability, as they allow to infer labels that are consistent with some prior knowledge by reasoning over high-level concepts extracted from sub-symbolic inputs. It was recently shown that NeSy predictors are affected by _reasoning shortcuts_ : they can attain high accuracy but by leveraging concepts with _unintended semantics_ , thus coming short of their promised advantages. Yet, a systematic characterization of reasoning shortcuts and of potential mitigation strategies is missing. This work fills this gap by characterizing them as unintended optima of the learning objective and identifying four key conditions behind their occurrence. Based on this, we derive several natural mitigation strategies, and analyze their efficacy both theoretically and empirically. Our analysis shows reasoning shortcuts are difficult to deal with, casting doubts on the trustworthiness and interpretability of existing NeSy solutions. 

## **1 Introduction** 

Neuro-Symbolic (NeSy) AI aims at improving the _robustness_ and _trustworthiness_ of neural networks by integrating them with reasoning capabilities and prior knowledge [1, 2]. We focus on _NeSy predictors_ [3, 4], neural structured-output classifiers that infer one or more labels by reasoning over high-level _concepts_ extracted from sub-symbolic inputs, like images or text [5, 6, 7, 8, 9, 10, 11]. They leverage reasoning techniques to encourage – or even _guarantee_ – that their predictions comply with domain-specific regulations and inference rules. As such, they hold the promise of improved _systematic generalization_ , _modularity_ , and _interpretability_ , in that learned concepts can be readily reused in different NeSy tasks, as done for verification [12], and for explaining the model’s inference process to stakeholders [13, 14, 15]. On paper, this makes NeSy predictors ideal for high-stakes applications that require both transparency and fine-grained control over the model’s (in- and out-ofdistribution) behavior, such as medical diagnosis [16], robotics [17] and self-driving cars [18]. 

Much of the promise of these models relies on learned concepts being _high quality_ . The general consensus is that the prior knowledge constrains learned concepts to behave as expected [19] and issues with them are often tackled heuristically [20]. It was recently shown that, however, NeSy predictors can _attain high accuracy by leveraging concepts with unintended semantics_ [21]. Following 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0002-00.png)


Figure 1: **Reasoning shortcuts undermine trustworthiness** .chiodo fisso An autonomous vehicle has to decide whether to _Y_ = `stop` or _Y_ = `go` based on three binary concepts extracted from an image **x** , namely _C_ 1 = `red` light, _C_ 2 = `green` light and _C_ 3 = presence of `pedestrian` s (shown in **pink** ). **Left** : In Task 1, the prior knowledge K = ( `pedestrian` _∨_ `red` _⇒_ `stop` ) instructs the vehicle to stop whenever the light is red or there are pedestrians on the road. The model can perfectly classify an (even exhaustive) training set by acquiring a _reasoning shortcut that classifies pedestrians as red lights_ . **Right** : The learned concepts are then reused to guide an autonomous ambulance with the additional rule that in emergency situations red lights can be ignored, with potentially dire consequences. Our work identifies the causes of RSs (Section 4) and several mitigation strategies (Section 5). 

Marconato et al. [21], we refer to these as _reasoning shortcuts_ (RSs). RSs are problematic, as concepts encoding unintended semantics compromise generalization across NeSy tasks, as shown in Fig. 1, as well as interpretability and verification of NeSy systems [12]. Moreover, the only known mitigation strategies are based on heuristics [20, 11]. 

The issue is that RSs – and their root causes – are not well understood, making it difficult to design effective remedies. In this paper, we fill this gap. We introduce a formal definition of RSs and theoretically characterize their properties, highlighting how they are a general phenomenon affecting a variety of state-of-the-art NeSy predictors. Specifically, our results show that RSs can be shared across many NeSy architectures, and provide a way of _counting_ them for any given learning problem. They also show that RSs depend on _four key factors_ , namely the structure of the prior knowledge and of the data, the learning objective, and the architecture of the neural concept extractor. This enables us to identify several supervised and unsupervised mitigation strategies, which we systematically analyze both theoretically and empirically. Finally, we experimentally validate our findings by testing a number of representative NeSy predictors and mitigation strategies on four NeSy data sets. 

**Contributions.** Summarizing, we: ( _i_ ) Formalize RSs and identify four key root causes. ( _ii_ ) Show that RSs are a general issue impacting a variety of NeSy predictors. ( _iii_ ) Identify a number of mitigation strategies and analyze their effectiveness, or lack thereof. ( _iv_ ) Empirically show that RSs arise even when the data set is large and unbiased, and evaluate the efficacy of different mitigation strategies, highlighting the limits of unsupervised remedies and the lack of a widely applicable recipe. 

## **2 The Family of Neuro-Symbolic Predictors** 

**Notation.** Throughout, we indicate scalar constants _x_ in lower-case, random variables _X_ in upper case, and ordered sets of constants **x** and random variables **X** in bold typeface. Also, **x** _i_ : _j_ denotes the subset _{xi, . . . , xj}_ , [ _n_ ] the set _{_ 1 _, . . . , n}_ , and **x** _|_ = K indicates that **x** satisfies a logical formula K. 

**NeSy predictors.** A _NeSy predictor_ is a model that infers _n labels_ **Y** (taking values in _Y_ ) by reasoning over a set of _k_ discrete _concepts_ **C** (taking values in _C_ ) extracted from a sub-symbolic continous _input_ **X** (taking values in _X_ ). Reasoning can be implemented in different ways, but overall its role is to encourage the model’s predictions to comply with given _prior knowledge_ K, in the sense that predictions **y** that do not satisfy the knowledge are generally avoided. The prior knowledge K is assumed to be provided upfront and correct, as formalized in Section 3. Normally, only supervision on the labels **Y** is available for training, with the concepts **C** treated as latent variables. 

**Example 1.** _In_ `MNIST-Addition` _[7], given a pair of MNIST images [22], say_ **x** = ( _,_ ) _, the model has to infer the concepts_ **C** = ( _C_ 1 _, C_ 2) _encoding the digit classes, to predict their sum Y , in this case_ 8 _. Reasoning drives the model towards complying with constraint_ K = ( _Y_ = _C_ 1 + _C_ 2) _._ 

The concepts are modeled by a conditional distribution _pθ_ ( **C** _|_ **X** ) parameterized by _θ ∈_ Θ, typically implemented with a neural network. The predicted concepts can be viewed as “soft” or “neural” predicates with a truth value ranging in [0 _,_ 1]. As for the reasoning step, the most popular strategies 

2 

involve _penalizing_ the model for producing concepts and/or labels inconsistent with the knowledge at training time [8, 23, 24] or introducing a _reasoning layer_ that infers labels from the predicted concepts and also operates at inference time [7, 9, 25, 10]. In either case, end-to-end training requires to differentiate through the knowledge. Mainstream options include softening the knowledge using fuzzy logic [26, 6, 27, 11] and casting reasoning in terms of probabilistic logics [28, 7, 10]. 

To investigate the scope and impact of RSs, we consider three representative NeSy predictors. The first one is **DeepProbLog** (DPL) [7], which implements a sound probabilistic-logic reasoning layer on top of the neural predicates. DPL is a discriminative predictor of the form [21]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0003-02.png)


where the concept distribution is fully factorized and the label distribution is _uniform_[1] over all label-concept combinations compatible with the knowledge, that is, _u_ K( **y** _|_ **c** ) = 1 _{_ **c** _|_ = K[ **Y** _/_ **y** ] _} /Z_ ( **c** ; K) where the indicator 1 _{_ **c** _|_ = K[ **Y** _/_ **y** ] _} guarantees_ all labels inconsistent with K have zero probability, and _Z_ ( **c** ; K) =[�] **y**[1] _[{]_ **[c]** _[ |]_[=][ K][[] **[Y]** _[/]_ **[y]**[]] _[}]_[ is a normalizing constant.][Infer-] ence amounts to computing a most likely label argmax **y** _pθ_ ( **y** _|_ **x** ; K), while learning is carried out via maximum (log-)likelihood estimation, that is given a training set _D_ = _{_ ( **x** _,_ **y** ) _}_ , maximizing 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0003-04.png)


In general, it is intractable to evaluate Eq. (1) and solve inference exactly. DPL leverages knowledge compilation [29, 30] to make both steps practical. 

Our analysis on DPL can be carried over to other NeSy predictors implementing analogous reasoning layers [31, 32, 33, 10, 34]. Furthermore, we show that certain RSs affect also alternative NeSy approaches such as the **Semantic Loss** (SL) [8] and **Logic Tensor Networks** (LTNs) [6], two stateof-the-art penalty-based approaches. Both reward a neural network for predicting labels **y** consistent with the knowledge, but SL measures consistency in probabilistic terms, while LTNs use fuzzy logic to measure a fuzzy degree of knowledge satisfaction. See Appendix A for a full description. 

## **3 Reasoning Shortcuts as Unintended Optima** 

It was recently shown that NeSy predictors are vulnerable to _reasoning shortcuts_ (RSs), whereby the model attains high accuracy by leveraging concepts with _unintended semantics_ [21]. 

**Example 2.** _To build intuition, consider_ `MNIST-Addition` _and assume the model is trained on examples of only two sums:_ + = 1 _and_ + = 2 _. Note that there exist two distinct maps from images to concepts that perfectly classify such examples: one is the intended solution_ ( _�→_ 0 _, �→_ 1 _, �→_ 2) _, while the other is_ ( _�→_ 1 _, �→_ 0 _, �→_ 1) _. The latter is unintended._ 

**The ground-truth data generating process.** In order to properly define what a RS is, we have to first define what _non_ -shortcut solutions are. **S G** In line with work on causal representation learning [35, 36, 37], we do so by specifying the ground-truth data generation process _p[∗]_ ( **X** _,_ **Y** ; K). Specifically, we assume it takes the form illustrated in Fig. 2. In short, **X Y** we assume there exist _k_ unobserved ground-truth concepts **G** ( _e.g._ , in `MNIST-Addition` these are the digits 0 to 9) that determine _both_ the observations **X** (the MNIST images) and the labels **Y** (the sum). We also **C** allow for extra stylistic factors **S** , independent from **G** , that do influence the observed data ( _e.g._ , calligraphic style) but _not_ the labels. The groundFigure 2: The groundtruth concepts are discrete and range in _G_ = [ _m_ 1] _× · · · ×_ [ _mk_ ], whereas truth data generation prothe style **s** _∈_ R _[q]_ is continuous. The training and test examples ( **x** _,_ **y** ) are cess (in **black** ) and a then obtained by first sampling **g** and **s** and then **x** _∼ p[∗]_ ( **X** _|_ **g** _,_ **s** ) and NeSy predictor (in **blue y** _∼ p[∗]_ ( **Y** _|_ **g** ; K). Later on, we will use supp( **G** ) to denote the support of _p[∗]_ ( **G** ). We also assume that the ground-truth process is consistent with the prior knowledge K, in the sense that invalid examples are never generated: _p[∗]_ ( **y** _|_ **g** ; K) = 0 for all ( **g** _,_ **y** ) that violate K. 

Figure 2: The groundtruth data generation process (in **black** ) and a NeSy predictor (in **blue** ). 

**What is a reasoning shortcut?** By definition, a NeSy predictor _pθ_ ( **Y** _|_ **X** ; K) – shown in **blue** in Fig. 2 – acquires concepts with the correct semantics if it recovers the ground-truth concepts, _i.e._ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0003-13.png)


> 1In practice, the distribution of labels given concepts needs not be uniform [10]. 

3 

A model satisfying Eq. (3) easily generalizes to other NeSy prediction tasks that make use of the same ground-truth concepts **G** , as _pθ_ ( **C** _|_ **X** ) can be reused for solving the new task [21, 38]. It is also interpretable, in the sense that as long as stakeholders understand the factors **G** , they can also interpret concept-based explanations of the inference process that rely on **C** [39]. Naturally, there may exist many concept distributions that _violate_ Eq. (3) and that as such do _not_ capture the correct semantics. However, all those that achieve sub-par log-likelihoods can be ideally avoided simply by improving learning or supplying more examples. We define RSs as those concept distributions for which these strategies are not enough. 

**Definition 1.** _A reasoning shortcut is a distribution pθ_ ( **C** _|_ **X** ) _that achieves maximal log-likelihood on the training set but does not match the ground-truth concept distribution,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0004-02.png)


This makes RSs difficult to improve by regular means and also hard to spot based on predictions alone. Yet, since the concepts do not recover the correct semantics, RSs can compromise systematic generalization and interpretability. For instance, the shortcut concepts learned in Example 2 would fail completely at `MNIST-Addition` tasks involving digits other than 0, 1, and 2 and also at other arithmetic tasks, like multiplication, involving the same digits, as we show experimentally in Section 6. An examples of RSs in a high-stakes scenario is shown in Fig. 1. 

## **4 Properties of Reasoning Shortcuts** 

**Counting deterministic RSs.** We begin by assessing how many RSs exist in an idealized setting in which the ground-truth generative process is simple, we have access to the true risk, and the concepts **C** have been specified correctly, that is, _C_ = _G_ , and then proceed to work out what this entails. Specifically, we work with these assumptions: 

- **A1** . The distribution _p[∗]_ ( **X** _|_ **G** _,_ **S** ) is induced by a map _f_ : ( **g** _,_ **s** ) _�→_ **x** , _i.e._ , _p[∗]_ ( **X** _|_ **G** _,_ **S** ) = _δ{_ **X** = _f_ ( **g** _,_ **s** ) _}_ , where _f_ is _invertible_ , and _smooth_ over **s** . 

- **A2** . The distribution _p[∗]_ ( **Y** _|_ **G** ; K) is induced by a map _β_ K : **g** _�→_ **y** . This is what happens in `MNIST-Addition` , as there exists a unique value _y_ that is the sum of any two digits ( _g_ 1 _, g_ 2). 

Our analysis builds on a link between the NeSy predictor _pθ_ ( **Y** _|_ **X** ; K) and the concept extractor _pθ_ ( **C** _|_ **X** ), which depend on **X** , and their analogues _pθ_ ( **Y** _|_ **G** ; K) and _pθ_ ( **C** _|_ **G** ) that depend directly on the ground-truth concepts **G** . This link is formalized by the following lemma: 

**Lemma 1.** _It holds that: (i) The true risk of pθ can be upper bounded as follows:_ 

E( **x** _,_ **y** ) _∼p∗_ ( **X** _,_ **Y** ;K)[log _pθ_ ( **y** _|_ **x** ; K)] _≤_ E **g** _∼p_ ( **G** )� _−_ KL[ _p[∗]_ ( **Y** _|_ **g** ; K) _∥ pθ_ ( **Y** _|_ **g** ; K)] _−_ H[ _p[∗]_ ( **Y** _|_ **g** ; K)]� (5) 

_where_ KL _is the Kullback-Leibler divergence and_ H _is the Shannon entropy. Moreover, under_ _**A1** and_ _**A2** , pθ_ ( **Y** _|_ **X** ; K) _is an optimum of the LHS of Eq._ (5) _if and only if pθ_ ( **Y** _|_ **G** ; K) _is an optimum of the RHS. (ii) Under_ _**A1** , there exists a bijection between the deterministic concept distributions pθ_ ( **C** _|_ **X** ) _that are constant over the support of p_ ( **X** _|_ **g** ) _, for each_ **g** _∈_ supp( **G** ) _, and the deterministic distributions of the form pθ_ ( **C** _|_ **G** ) _._ 

All proofs can be found in Appendix B. Lemma 1 implies that the deterministic concept distributions _pθ_ ( **C** _|_ **X** ) of NeSy predictors _pθ_ ( **Y** _|_ **X** ; K) that maximize the LHS of Eq. (5), including those that are RSs, correspond one-to-one to the deterministic distributions _pθ_ ( **C** _|_ **G** ) yielding label distributions _pθ_ ( **Y** _|_ **G** ; K) that maximize the RHS of Eq. (5). Hence, we can count the number of _deterministic_ RSs by counting the deterministic distributions _pθ_ ( **C** _|_ **G** ): 

**Theorem 2.** _Let A be the set of mappings α_ : **g** _�→_ **c** _induced by all possible deterministic distributions pθ_ ( **C** _|_ **G** ) _, i.e., each pθ_ ( **C** _|_ **G** ) = 1 _{_ **C** = _α_ ( **G** ) _} for exactly one α ∈A. Under_ _**A1** and_ _**A2** , the number of deterministic optima pθ_ ( **C** _|_ **G** ) _of Eq._ (5) _is:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0004-14.png)


Intuitively, this sum counts the deterministic concept distributions _pθ_ ( **C** _|_ **X** ) – embodied here by the maps _α_ – that output concepts predicting a _correct_ label for each example in the training set. The 

4 

Table 1: **Impact of different mitigation strategies on the number of deterministic optima** : R is reconstruction, C supervision on **C** , MTL multi-task learning, and DIS disentanglement. All strategies reduce the number of _α_ ’s in Eq. (6), sometimes substantially, but require different amounts of effort to be put in place. Actual counts for our data sets are reported in Appendix C.2. 

|MITIGATION|REQUIRES|CONSTRAINT ON_α_|ASSUMPTIONS|RESULT|
|---|---|---|---|---|
|None|–|�<br>**g**_∈_supp(**G**)<br>�<br>(_β_K_◦α_)(**g**) =_β_K(**g**)<br>�|**A1**,**A2**|Theorem 2|
|MTL|Tasks|�<br>**g**_∈_supp(**G**)<br>�<br>_t∈_[_T_]<br>�<br>(_β_K(_t_) _◦α_)(**g**) =_β_K(_t_)(**g**)<br>�|**A1**,**A2**|Proposition 4|
|C|Sup. on**C**|�<br>**g**_∈S⊆_supp(**G**)<br>�<br>_i∈I_<br>�<br>_αi_(**g**) =_gi_<br>�|**A1**|Proposition 5|
|R|–|�<br>**g**_,_**g**_′∈_supp(**G**):**g**=**g**_′_<br>�<br>_α_(**g**) =_α_(**g**_′_)<br>�|**A1**,**A3**|Proposition 6|



ground-truth distribution _p[∗]_ ( **G** _|_ **X** ) is one such distribution, so the count is always at least one, but there may be more, and all of these are RSs. Eq. (6) gives us their exact number. This formalizes the intuition of Marconato et al. [21] that, as long as the prior knowledge K admits the correct label **y** to be inferred from more than one concept vector **c** , there is room for RSs. So far, we have assumed Eq. (2) is computed as in DPL. However, RSs are chiefly a property of the _prior knowledge_ , and as such also affect NeSy predictors employing different reasoning procedures or different relaxations of the knowledge. We show this formally in Appendix A. Deterministic RS are also important because – in certain cases – they define a basis for _all_ reasoning shortcuts: 

**Proposition 3.** _For probabilistic logic approaches (including DPL and SL): (i) All convex combinations of two or more deterministic optima pθ_ ( **C** _|_ **X** ) _of the likelihood are also (non-deterministic) optima. However, not all convex combinations can be expressed in DPL and SL. (ii) Under_ _**A1** and_ _**A2** , all optima of the likelihood can be expressed as a convex combination of deterministic optima. (iii) If_ _**A2** does not hold, there may exist non-deterministic optima that are not convex combinations of deterministic ones. These may be the only optima._ 

Combining Proposition 3 ( _i_ ) with Theorem 2 gives us a lower bound for the number of _nondeterministic_ RSs, in the sense that if there are at least two deterministic RS, then there exist infinitely many non-deterministic ones. An important consequence is that, if we can somehow control what deterministic RSs affect the model, then we may be able to implicitly lower the number of _non-deterministic_ RSs as well. However, Proposition 3 implies that there may exist _non-deterministic_ RSs that are unrelated to the deterministic ones and that as such cannot be controlled this way. 

## **5 Analysis of Mitigation Strategies** 

The key factors underlying the occurrence of deterministic RSs appear explicitly in Eq. (6). These are: ( _i_ ) the knowledge K, ( _ii_ ) the structure of supp( **G** ), ( _iii_ ) the objective function _L_ used for training (via Lemma 1), and ( _iv_ ) the architecture of the concept extractor _pθ_ ( **C** _|_ **X** ), embodied in the Theorem by _pθ_ ( **C** _|_ **G** ). This gives us a starting point for identifying possible mitigation strategies and analyzing their impact on the number of _deterministic_ RSs. Our main results are summarized in Table 1. 

## **5.1 Knowledge-based Mitigation** 

The _prior knowledge_ K is the main factor behind RSs and also a prime target for mitigation. The most direct way of eliminating unintended concepts is to edit K directly, for instance by eliciting additional constraints from a domain expert. However, depending on the application, this may not be feasible: experts may not be available, or it may be impossible to constrain K without also eliminating concepts with the intended semantics. 

A more practical alternative is to employ _Multi-Task Learning_ (MTL). The idea is to train a NeSy predictor over _T_ tasks sharing the same ground-truth concepts **G** but differing prior knowledge K[(] _[t]_[)] , for _t ∈_ [ _T_ ]. _E.g._ , one could learn a model to predict both the sum and product of MNIST digits, as in our experiments (Section 6). Intuitively, by constraining the concepts to work well across tasks, MTL leaves less room for unintended semantics. The following result confirms this intuition: 

**Proposition 4.** _Consider T NeSy prediction tasks with knowledge_ K[(] _[t]_[)] _, for t ∈_ [ _T_ ] _and data sets D_[(] _[t]_[)] _, all sharing the same p[∗]_ ( **G** ) _. Under_ _**A1** and_ _**A2** , any deterministic optimum pθ_ ( **C** _|_ **G** ) _of the_ MTL _loss (i.e., the average of per-task losses) is a deterministic optimum of a single task with prior_ 

5 

_knowledge_[�] _t_[K][(] _[t]_[)] _[.][The number of deterministic optima amounts to:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0006-01.png)


This means that, essentially, MTL behaves like a logical conjunction: any concept extractor _pθ_ ( **C** _|_ **G** ) incompatible with the knowledge of _any_ task _t_ is not optimal. This strategy can be very effective, and indeed it performs very well in our experiments, but it necessitates gathering or designing a _set_ of correlated learning tasks, which may be impractical in some situations. 

## **5.2 Data-based Mitigation** 

Another key factor is the support of _p[∗]_ ( **G** ): if the support is not full, the conjunction in Eq. (6) becomes looser, and the number of _α_ ’s satisfying it increases. This is what happens in `MNIST-EvenOdd` (Example 2): here, RSs arise precisely because the training set only includes a _subset_ of combinations of digits, leaving ample room for acquiring unintended concepts. 

We stress, however, that RSs can also occur if the data set is _exhaustive_ , as in the next example. **Example 3** ( `XOR` task) **.** _Consider a task with three binary ground-truth concepts_ **G** = ( _G_ 1 _, G_ 2 _, G_ 3) _in which the label Y is the parity of these bits, that is_ K = ( _Y_ = _G_ 1 _⊕ G_ 2 _⊕ G_ 3) _. Each label Y ∈{_ 0 _,_ 1 _} can be inferred from four possible concept vectors_ **g** _, meaning that knowing_ **y** _is not sufficient to identify the_ **g** _it was generated from. In this case, it is impossible to pin down the ground-truth distribution p[∗]_ ( **C** _|_ **G** ; K) _even if all possible combinations of inputs_ **x** _are observed._ 

One way of avoiding RSs is to explicitly guide the model towards satisfying the condition in Eq. (3) by supplying _supervision_ for a subset of concepts **C** _I ⊆_ **C** , with _I ⊆_ [ _k_ ], and then augmenting the log-likelihood with a cross-entropy loss over the concepts of the form[�] _i∈I_[log] _[ p][θ]_[(] _[C][i]_[=] _[g][i][|]_ **[x]**[)][.] Here, training examples ( **x** _,_ **g** _I ,_ **y** ) come with annotations for the concepts indexed by _I_ . The impact of this strategy on the number of deterministic RSs is given by the following result: 

**Proposition 5.** _Assume that concept supervision is available for all_ **g** _in S ⊆_ supp( **G** ) _. Under_ _**A1** , the number of deterministic optima pθ_ ( **C** _|_ **G** ) _minimizing the cross-entropy over the concepts is:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0006-08.png)


This strategy is very powerful: if _I_ = [ _k_ ], _S ≡_ supp( **G** ), and the support is complete, there exists only _one_ map _α_ that is consistent with the condition in Eq. (8) and it is the identity. Naturally, this comes at the cost of obtaining dense annotations for all examples, which is often impractical. 

## **5.3 Objective-based Mitigation** 

A natural alternative is to augment the log-likelihood with an _unsupervised_ penalty designed to improve concept quality. We focus on reconstruction penalties like those used in auto-encoders [40, 41, 42, 43, 44], which encourage the model to capture all information necessary to reconstruct the input **x** . To see why these might be useful, consider Example 2. Here, the model learns a RS mapping both and to the digit 1: this RS hinders reconstruction of the input images, and therefore could be avoided by introducing a reconstruction penalty. 

In order to implement this, we introduce additional latent variables **Z** that capture the style **S** of the input **X** and modify the concept extractor to output both **C** and **Z** , that is: _pθ_ ( **c** _,_ **z** _|_ **x** ) = _pθ_ ( **c** _|_ **x** ) _· pθ_ ( **z** _|_ **x** ). The auto-encoder reconstruction penalty is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0006-13.png)


where, _pψ_ ( **x** _|_ **c** _,_ **z** ) is the decoder network. We need to introduce an additional assumption **A3** : the encoder and the decoder separate content from style, that is, _pθ_ ( **C** _,_ **Z** _|_ **G** _,_ **S** ) := E **x** _∼p∗_ ( **x** _|_ **G** _,_ **S** ) _pθ_ ( **C** _,_ **Z** _|_ **x** ) factorizes as _pθ_ ( **C** _|_ **G** ) _pθ_ ( **Z** _|_ **S** ) and _pψ_ ( **G** _,_ **S** _|_ **C** _,_ **Z** ) := E **x** _∼pψ_ ( **x** _|_ **C** _,_ **Z** ) _p[∗]_ ( **G** _,_ **Z** _|_ **x** ) as _pψ_ ( **G** _|_ **C** ) _pψ_ ( **S** _|_ **Z** ). In this case, we have the following result: 

**Proposition 6.** _Under_ _**A1** and_ _**A3** , the number of deterministic distributions pθ_ ( **C** _|_ **G** ) _that minimize the reconstruction penality in Eq._ (9) _is:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0006-16.png)


(10) 

6 

||`XOR`<br>DPL<br>SL<br>LTN|`MNIST-Addition`<br>DPL<br>SL<br>LTN<br>96_._7%<br>82_._9%<br>100%<br>0%<br>0%<br>0%<br>Table 2: **Q1: Disentanglement (DIS) can be very**<br>**powerful** to lower the frequency of RSs on `XOR` and<br>`MNIST-Addition`data sets (the lower the better). Re-<br>sults are averaged over30_optimal_runs.|
|---|---|---|
|–|100%<br>100%<br>100%<br><br><br>||
|DIS|0%<br>0%<br>0%||



In words, this shows that indeed optimizing for reconstruction facilitates disambiguating between different concepts, _i.e._ , different ground-truth concepts cannot be mapped to the same concept. However, minimizing the reconstruction can be non-trivial in practice, especially for complex inputs. 

## **5.4 Architecture-based Mitigation** 

One last factor is the _architecture of the concept extractor pθ_ ( **C** _|_ **X** ), as it implicitly controls the number of candidate deterministic maps _A_ and therefore the sum in Theorem 2. If the architecture is unrestricted, _pθ_ ( **C** _|_ **X** ) can in principle map any ground-truth concept **g** that generated **x** to any concept **c** , thus the cardinality of _A_ increases exponentially with _k_ . 

A powerful strategy for reducing the size of _A_ is _disentanglement_ . A model is disentangled if and only if _pθ_ ( **C** _|_ **G** ) factorizes as[�] _j∈_ [ _k_ ] _[p][θ]_[(] _[C][j][|][ G][j]_[)][ [][45][,][ 36][].][In this case, the maps] _[ α]_[ also factorize] into per-concept maps _αj_ : [ _mj_ ] _→_ [ _mj_ ], dramatically reducing the cardinality of _A_ , as shown by our first experiment. In applications where the _k_ concepts are naturally independent from one another, _e.g._ , digits in `MNIST-Addition` , one can implement disentanglement by predicting each concept using the same neural network, although more general techniques exist [46, 47]. 

## **5.5 Other Heuristics based on Entropy Regularization** 

Besides these mitigation strategies, we investigate empirically the effect of the Shannon entropy _k_ loss, defined as 1 _− k_[1] � _i_ =1[H] _[m] i_[[] _[p][θ]_[(] _[c][i]_[)]][, which was shown to increase concept quality in DPL [][20][].] Here, _pθ_ ( **C** ) is the marginal distribution over the concepts, and H _mi_ is the normalized Shannon entropy over _mi_ possible values for the distribution. Notice that this term goes to zero only when each distribution _pθ_ ( _Ci_ ) is uniform, which may conflict with the real objective of the NeSy prediction (especially when only few concepts are observed). Other similar heuristics that are suited for reducing over-confidence in label predictions are based on label smoothing [48], energy-based models [49], annealing [50], and many others [51, 52, 53]. In principle, when applied at the concept level, they help reduce the over-confidence in concepts that is typical in deterministic RSs. While they could also be beneficial to mildly reduce the number of RSs, we take the Shannon entropy regularization as a representative of this family for our experiments. 

## **6 Case Studies** 

In this section, we evaluate the impact of RSs in synthetic and real-world NeSy prediction tasks and how the mitigation strategies discussed in Section 5 fare in practice. More details about the models and data are reported in Appendix C. The **code** is available at github.com/reasoning-shortcuts. 

**Q1: More data does not prevent RSs but disentanglement helps.** We start by evaluating the robustness of DPL, SL, and LTN to RSs in two settings where the data set is _exhaustive_ . In `XOR` (cf. Example 3), the goal is to predict the parity of three binary concepts **g** _∈{_ 0 _,_ 1 _}_[3] given prior knowledge K = ( _y_ = _g_ 1 _⊕ g_ 2 _⊕ g_ 3). The predictor receives the ground-truth concepts **g** as input and has to learn a distribution _pθ_ ( **C** _|_ **G** ). In `MNIST-Addition` (cf. Example 1) the goal is to correctly predict the sum of two MNIST digits, _e.g._ , **x** = ( _,_ ) and _y_ = 3. In both tasks, the training set contains examples of _all_ possible combinations of concepts, _i.e._ , supp( **G** ) = _G_ . 

Since the tasks are relatively simple, we can afford to study models that achieve near-optimal likelihood, for which Definition 1 approximately applies. To this end, for each NeSy architecture, we train several models using different seeds and stop as soon as we obtain 30 models with likelihood _≥_ 0 _._ 95. Then, we measure the percentage of models that have acquired a RS. We do the same also for models modified to ensure they are disentangled (DIS in the Table), see Appendix C for details. The results, reported in Table 2, clearly show that RSs affect _all_ methods if disentanglement is not in place. This confirms that, without implicit architectural biases, _optimizing for label accuracy alone is not_ 

7 

Table 3: **Q2: Impact of mitigation strategies.** We report the _F_ 1-score on the labels ( **Y** ) and concepts ( **C** ). All tested methods incorporate DIS and are averaged over 10 runs. **Top** : NeSy methods combined with R, C, H on `MNIST-EvenOdd` . **Bottom** : evaluation on single tasks _vs_ MTL on `MNIST-AddMul` . 

|`MNIST-EvenOdd`|DPL<br>_F_1 (**Y**)<br>_F_1 (**C**)|SL<br>_F_1 (**Y**)<br>_F_1(**C**)|LTN<br>_F_1 (**Y**)<br>_F_1(**C**)|
|---|---|---|---|
|–|85_._1_±_4_._6<br>0_._1_±_0_._1|99_._3_±_0_._2<br>0_._2_±_0_._1|98_._1_±_0_._2<br>0_._3_±_0_._1|
|R|79_._8_±_1_._0<br>0_._1_±_0_._0|99_._5_±_0_._2<br>0_._1_±_0_._0|76_._3_±_1_._1<br>0_._0_±_0_._0|
|H|98_._1_±_0_._1<br>0_._1_±_0_._1|99_._4_±_0_._1<br>0_._1_±_0_._0|81_._9_±_0_._5<br>53_._9_±_0_._7|
|C|84_._9_±_0_._1<br>0_._1_±_0_._1|99_._3_±_0_._4<br>21_._5_±_6_._2|98_._1_±_0_._2<br>0_._2_±_0_._1|
|R+ H|75_._4_±_0_._4<br>0_._2_±_0_._1|99_._6_±_0_._1<br>0_._1_±_0_._0|97_._9_±_2_._3<br>38_._1_±_16_._7|
|R+ C|84_._0_±_2_._2<br>1_._9_±_4_._4|99_._3_±_0_._2<br>61_._5_±_7_._8|98_._1_±_0_._2<br>0_._2_±_0_._1|
|H+ C|91_._9_±_3_._5<br>88_._0_±_6_._3|99_._4_±_0_._2<br>41_._5_±_8_._2|98_._2_±_0_._2<br>98_._6_±_0_._1|
|R+ H+ C|95_._4_±_0_._4<br>96_._2_±_0_._2|99_._5_±_0_._2<br>47_._2_±_9_._8|98_._1_±_0_._3<br>98_._5_±_0_._2|
|||||
|`MNIST-AddMul`|DPL<br>_F_1 (**Y**)<br>_F_1 (**C**)|SL<br>_F_1 (**Y**)<br>_F_1 (**C**)|LTN|
||||_F_1 (**Y**)<br>_F_1 (**C**)|
|ADD|68_._1_±_6_._7<br>0_._0_±_0_._0|99_._5_±_0_._2<br>0_._0_±_0_._1|67_._4_±_0_._1<br>0_._0_±_0_._0|
|MULT|100_._0_±_0_._0<br>37_._6_±_0_._2|100_._0_±_0_._0<br>76_._1_±_11_._7|98_._1_±_0_._5<br>78_._1_±_0_._4|
|MULTIOP|100_._0_±_0_._0<br>99_._8_±_0_._1|100_._0_±_0_._0<br>99_._8_±_0_._1|98_._3_±_0_._2<br>98_._3_±_0_._2|



_sufficient to rule out RSs, even when the data is exhaustive_ . However, when forcing disentanglement, the percentage of models affected by RSs drops to _zero_ for all data sets and methods, indicating that mitigation strategies that go beyond the standard learning setup – and specifically, disentanglement – can be extremely effective. 

**Q2: Disentanglement is not enough under selection bias.** Next, we look at two (non-exhaustive) data sets where RS occur due to _selection bias_ . Label and concept quality is measured using the _F_ 1- score on the macro average measured on the test set. From here onward, all models are disentangled by construction, and despite this are affected by RSs, as shown below. 

We start by evaluating `MNIST-EvenOdd` , a variant of `MNIST-Addition` (inspired by [21]) where only 16 possible pairs of digits out of 100 are given for training: 8 comprise only even digits and 8 only odd digits. As shown in [21], this setup allows to exchange the semantics of the even and odd digits while ensuring all sums are correct (because, _e.g._ , + = 12 = + ). As commonly done in NeSy, hyperparameters were chosen to optimize for label prediction performance on a validation set, cf. Appendix C. The impact of reconstruction (R), concept supervision (C), and Shannon entropy loss (H) on all architectures are reported in Table 3. Roughly speaking, a concept _F_ 1 below 95% typically indicates a RSs, as shown by the concept confusion matrices in Appendix C.5. The main take away is that _no strategy alone can effectively mitigate RSs for any of the methods_ . Additionally, for DPL and LTN, these also tend to interfere with label supervision, yielding degraded prediction performance. The SL is not affected by this, likely because it is the only method using a separate neural layer to predict the labels. Combining multiple strategies does improve concept quality on average, depending on the method. In particular, H + C and R + H + C help LTN identify good concepts, and similarly for DPL, although concept quality is slightly less stable. For the SL only concept supervision is relatively effective, but the other strategies are not, probably due to the extra flexibility granted by the top neural layer compared to DPL and LTN. 

Next, we evaluate the impact of multi-task learning on an arithmetic task, denoted `MNIST-AddMul` . Here, the model observes inputs **x** _∈{_ ( _,_ ) _,_ ( _,_ ) _,_ ( _,_ ) _}_ and has to predict both their sum _and_ their product, either separately (no MTL) or jointly (MTL). The results in Table 3 show that, as in Example 2, all methods are dramatically affected by RSs when MTL is not in place. DPL and LTN also yield sub-par _F_ 1-scores on the labels for the addition task. For SL and LTN, we also observe that, despite high concept _F_ 1 for the multiplication task, the digit is _never_ predicted correctly. This is clearly visible in the confusion matrices in Appendix C.5. However, solving addition and multiplication jointly via MTL ensures all methods acquire very high quality concepts. 

**Q3: Reasoning Shortcuts are pervasive in real-world tasks.** Finally, we look at RSs occurring in `BDD-OIA` [54], an autonomous vehicle prediction task. The goal is to predict multiple possible actions **Y** = ( `move_forward` _,_ `stop` _,_ `turn_left` _,_ `turn_right` ) from frames **x** of real driving scenes. Each scene is described by 21 binary concepts **C** and the knowledge K ensures the predictions are consistent with safety constraints ( _e.g._ , `stop` _⇒¬_ `move_forward` ) and guarantees concepts are predicted consistently with one another ( _e.g._ , `road_clear` _⇔¬_ `obstacle` ). Since this is a high-stakes task, we solve it with DPL, as it is the only approach out of the ones we consider that _guarantees_ hard compliance with the knowledge [7]. See Appendix C for the full experimental setup. 

Table 4 lists the results for DPL paired with different mitigation strategies. In order to get a sense of the model’s ability to acquire good concepts, we also include two baselines: `CBM-AUC` introduced by 

8 

Table 4: **Q3** . **Left** : Means and std. deviations over 10 runs for DPL paired with different mitigations. **Right** : Confusion matrices on the training set for DPL alone ( _top_ ) and paired with mitigation strategies ( _bottom_ ) for _{_ `move_forward,stop,turn_left,turn_right` _}_ concept vectors. 

|`CBM-AUC`|`BDD-OIA`|
|---|---|
||F1-mean (Y)<br>F1-mean(C)<br>70_._8_±_0_._1<br>62_._1_±_0_._1|
|C-only|64_._8_±_0_._2<br>60_._3_±_0_._1|
|DPL|71_._4_±_0_._1<br>39_._4_±_6_._2|
|DPL+H|**72**_._**1**_±_**0**_._**1**<br>48_._1_±_0_._3|
|DPL+C|68_._2_±_0_._2<br>60_._5_±_0_._1|
|DPL+C+H|68_._3_±_0_._3<br>**61**_._**7**_±_**0**_._**1**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0009-02.png)


Sawada and Nakamura [55] for this task, which learns jointly the concepts together with unsupervised ones, and processes both of them with a linear layer to obtain the labels; C-only, in which the concept extractor is trained with full concept supervision and frozen, and DPL is only used to perform inference. This avoids any interference between label and concept supervision. The _F_ 1-scores are computed over the test set and averaged over all 4 labels and all 21 concepts. On the right, we also report the concept confusion matrices (CMs) for the training set: each row corresponds to a ground-truth concept _vector_ **g** , and each column to a predicted concept _vector_ **c** . 

Overall, the results show that, unless supplied with concept supervision, DPL optimizes for label accuracy ( _F_ 1(Y)) by leveraging low quality concepts ( _F_ 1(C)). This occurs even when it is paired with a Shannon entropy penalty H. This is especially evident in the CMs, which show that, for certain labels, all ground-truth concepts **g** are mapped a single **c** , not unlike what happens in our first experiment for entangled models. Conversely, the concept quality of DPL substantially improves when concept supervision is available, at the cost of a slight degradation in prediction accuracy. The CMs back up this observation as the learned concepts tend to align much closer to the diagonal. Only for the `turn_left` label concept supervision fails to prevent collapse. This occurs because annotations for the corresponding concepts are poor: for a large set of examples, the concepts necessary to predict `turn_left` = 1 are annotated as negatives, complicating learning. In practice, this means that all variants of DPL predict those concepts as negative, hindering concept quality. 

## **7 Related Work** 

**Shortcuts in ML.** State-of-the-art ML predictors often achieve high performance by exploiting spurious correlations – or “shortcuts” – in the training data [56]. Well known examples include watermarks [57], background pixels [58, 59], and textual meta-data in X-ray scans [16]. Like RSs, regular shortcuts can compromise the classifier’s reliability and out-of-distribution generalization and are hard to identify based on accuracy alone. Proposed solutions include dense annotations [60], out-of-domain data [61], and interaction with annotators [62]. Shortcuts are often the result of confounding resulting from, _e.g._ , selective sampling. RSs may also arise due to confounding, as is the case in `MNIST-Addition` (cf. Example 1), however – as discussed in Section 5, data is not the only factor underlying RSs. For instance, in `XOR` (cf. Example 3) RSs arise despite exhaustive data. 

**Reasoning shortcuts.** The issue of RSs has so far been mostly neglected in the NeSy literature, and few remedies have been introduced but never theoretically motivated. Stammer et al. [63] have investigated shortcuts affecting NeSy architectures, but consider only _input-level_ shortcuts that occur even if concepts are high-quality and fix by injecting additional knowledge in the model. In contrast, we focus on RSs that impact the quality of learned concepts. Marconato et al. [21] introduce the concept of RSs in the context of NeSy but for continual learning, and proposes a combination of concept supervision and concept-level rehearsal to address them, without delving into a theoretical justification. Li et al. [11] propose a minimax objective that ensures the concepts learned by the model satisfy K. Like the entropy regularizer [20] we addressed in Section 5, this strategy ends up spreading probability across all concepts that satisfy the knowledge, including those that have unintended semantics. As such, it does not directly address RSs. 

**Neuro-symbolic integration.** While RSs affect a number of NeSy predictors, NeSy encompasses a heterogeneous family of architectures integrating learning and reasoning [1, 2]. We conjecture that 

9 

RSs do transfer to _all_ NeSy approaches that do not specifically address the factors we identified, but an in-depth analysis of RSs in NeSy is beyond the scope of this paper. 

**Relation to disentanglement.** Recovering the latent variables **G** has been the center of several works in representation learning [64, 35, 65]. Among these, achieving identifiability of the latent components has been studied in non-linear independent component analysis [66, 67, 68] and recently in Causal Representation Learning [46, 69, 70, 71]. In this respect, several overlaps exist with the natural mitigation strategies that we investigated: (i) multi-task learning has been shown to increase disentanglement [72] and provably leads to identifiability [73, 74], and (ii) latent variables supervision also largely increases the amount of disentanglement [75, 76, 47]. Our work is the witness that the intersection with _disentanglement_ literature is beneficial for learning the intended concepts in NeSy and, vice-versa, that knowledge-guided learning can be a new way of acquiring identifiable representations, by avoiding RSs. 

## **8 Conclusion** 

In this work, we provide the first in-depth analysis of RS affecting NeSy predictors. Our analysis highlights four key causes of RS and suggests several mitigation strategies, which we analyze both theoretically and empirically. Our experiments indicate that RSs do naturally appear in both synthetic and real-world NeSy prediction tasks, and that the effectiveness of mitigation strategies is model and task dependent, and that a general recipe for avoiding RSs is currently missing. 

We foresee that reasoning shortcuts extend beyond the current scope of NeSy predictors with known prior knowledge. This includes approaches that learn jointly both the concepts and the knowledge, like ROAP [77] and DSL [78], as well as fully neural models, like CBMs [79] and their variants [39, 80], which are also likely to be affected by RSs when concept supervision is limited. We plan to investigate further this direction in the near future. 

Ultimately, this work aims at jumpstarting research on the analysis and mitigation of RSs, with the hope of leading to more trustworthy and explainable NeSy architectures. 

**Broader impact.** Our work brings the subtle but critical issue of RSs to the spotlight, and highlights benefits and limitations of a variety of mitigation strategies, but it is otherwise fundamental research that has no direct societal impact. 

## **Acknowledgments and Disclosure of Funding** 

We thank the anonymous reviewers for their valuable feedback in improving the current manuscript. We acknowledge Emile von Krieken for proofreading our claims and Yoshihide Sawada for his help in deploying the models on `BDD-OIA` . We are also grateful to Pedro Zuidberg dos Martires for insightful discussions with us at the early stages of the work. We acknowledge the support of the PNRR project FAIR - Future AI Research (PE00000013), under the NRRP MUR program funded by the NextGenerationEU. The research of ST and AP was partially supported by TAILOR, a project funded by EU Horizon 2020 research and innovation programme under GA No 952215. 

## **References** 

- [1] Luc De Raedt, Sebastijan Dumanˇci´c, Robin Manhaeve, and Giuseppe Marra. From statistical relational to neural-symbolic artificial intelligence. In _Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence_ , pages 4943–4950, 2021. 

- [2] Artur d’Avila Garcez, Sebastian Bader, Howard Bowman, Luis C Lamb, Leo de Penning, BV Illuminoo, Hoifung Poon, and COPPE Gerson Zaverucha. Neural-symbolic learning and reasoning: A survey and interpretation. _Neuro-Symbolic Artificial Intelligence: The State of the Art_ , 342:1, 2022. 

- [3] Eleonora Giunchiglia, Mihaela Catalina Stoian, and Thomas Lukasiewicz. Deep learning with logical constraints. _arXiv preprint arXiv:2205.00523_ , 2022. 

10 

- [4] Tirtharaj Dash, Sharad Chitlangia, Aditya Ahuja, and Ashwin Srinivasan. A review of some techniques for inclusion of domain-knowledge into deep neural networks. _Scientific Reports_ , 12 (1):1–15, 2022. 

- [5] Michelangelo Diligenti, Marco Gori, and Claudio Sacca. Semantic-based regularization for learning and inference. _Artificial Intelligence_ , 2017. 

- [6] Ivan Donadello, Luciano Serafini, and Artur D’Avila Garcez. Logic tensor networks for semantic image interpretation. In _IJCAI_ , 2017. 

- [7] Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. DeepProbLog: Neural Probabilistic Logic Programming. _NeurIPS_ , 2018. 

- [8] Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy Broeck. A semantic loss function for deep learning with symbolic knowledge. In _ICML_ , 2018. 

- [9] Eleonora Giunchiglia and Thomas Lukasiewicz. Coherent hierarchical multi-label classification networks. _NeurIPS_ , 2020. 

- [10] Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, and Antonio Vergari. Semantic Probabilistic Layers for Neuro-Symbolic Learning. In _NeurIPS_ , 2022. 

- [11] Zenan Li, Zehua Liu, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, L Jian, et al. Learning with logical constraints but without shortcut satisfaction. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [12] Xuan Xie, Kristian Kersting, and Daniel Neider. Neuro-symbolic verification of deep neural networks. 2022. 

- [13] Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. _Nature Machine Intelligence_ , 1(5):206–215, 2019. 

- [14] Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: Deep learning for interpretable image recognition. _NeurIPS_ , 2019. 

- [15] Zhi Chen, Yijie Bei, and Cynthia Rudin. Concept whitening for interpretable image recognition. _Nature Machine Intelligence_ , 2020. 

- [16] Alex J DeGrave, Joseph D Janizek, and Su-In Lee. Ai for radiographic covid-19 detection selects shortcuts over signal. _Nature Machine Intelligence_ , pages 1–10, 2021. 

- [17] Elisa Maiettini, Giulia Pasquale, Vadim Tikhanoff, Lorenzo Rosasco, and Lorenzo Natale. A weakly supervised strategy for learning object detection on a humanoid robot. In _2019 IEEE-RAS 19th International Conference on Humanoid Robots (Humanoids)_ , pages 194–201. IEEE, 2019. 

- [18] Claudine Badue, Rânik Guidolini, Raphael Vivacqua Carneiro, Pedro Azevedo, Vinicius B Cardoso, Avelino Forechi, Luan Jesus, Rodrigo Berriel, Thiago M Paixao, Filipe Mutz, et al. Self-driving cars: A survey. _Expert Systems with Applications_ , 165:113816, 2021. 

- [19] Matt Fredrikson, Kaiji Lu, Saranya Vijayakumar, Somesh Jha, Vijay Ganesh, and Zifan Wang. Learning modulo theories. _arXiv preprint arXiv:2301.11435_ , 2023. 

- [20] Robin Manhaeve, Sebastijan Dumanˇci´c, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Neural probabilistic logic programming in deepproblog. _Artificial Intelligence_ , 298: 103504, 2021. 

- [21] Emanuele Marconato, Gianpaolo Bontempo, Elisa Ficarra, Simone Calderara, Andrea Passerini, and Stefano Teso. Neuro symbolic continual learning: Knowledge, reasoning shortcuts and concept rehearsal. In _ICML_ , 2023. 

- [22] Yann LeCun. The mnist database of handwritten digits. _http://yann. lecun. com/exdb/mnist/_ , 1998. 

11 

- [23] Marc Fischer, Mislav Balunovic, Dana Drachsler-Cohen, Timon Gehr, Ce Zhang, and Martin Vechev. Dl2: Training and querying neural networks with logic. In _International Conference on Machine Learning_ , pages 1931–1941. PMLR, 2019. 

- [24] Kareem Ahmed, Eric Wang, Kai-Wei Chang, and Guy Van den Broeck. Neuro-symbolic entropy regularization. In _UAI_ , 2022. 

- [25] Nick Hoernle, Rafael Michael Karampatsis, Vaishak Belle, and Kobi Gal. Multiplexnet: Towards fully satisfied logical constraints in neural networks. In _AAAI_ , 2022. 

- [26] Michelangelo Diligenti, Marco Gori, Marco Maggini, and Leonardo Rigutini. Bridging logic and kernel machines. _Machine learning_ , 86(1):57–88, 2012. 

- [27] Connor Pryor, Charles Dickens, Eriq Augustine, Alon Albalak, William Wang, and Lise Getoor. Neupsl: Neural probabilistic soft logic. _arXiv preprint arXiv:2205.14268_ , 2022. 

- [28] Luc De Raedt and Angelika Kimmig. Probabilistic (logic) programming concepts. _Machine Learning_ , 2015. 

- [29] Adnan Darwiche and Pierre Marquis. A knowledge compilation map. _Journal of Artificial Intelligence Research_ , 17:229–264, 2002. 

- [30] Antonio Vergari, YooJung Choi, Anji Liu, Stefano Teso, and Guy Van den Broeck. A compositional atlas of tractable circuit operations for probabilistic inference. _Advances in Neural Information Processing Systems_ , 34, 2021. 

- [31] Robin Manhaeve, Giuseppe Marra, and Luc De Raedt. Approximate inference for neural probabilistic logic programming. In _KR_ , 2021. 

- [32] Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, and Xujie Si. Scallop: From probabilistic deductive databases to scalable differentiable reasoning. _NeurIPS_ , 2021. 

- [33] Thomas Winters, Giuseppe Marra, Robin Manhaeve, and Luc De Raedt. DeepStochLog: Neural Stochastic Logic Programming. In _AAAI_ , 2022. 

- [34] Emile van Krieken, Thiviyan Thanapalasingam, Jakub M Tomczak, Frank van Harmelen, and Annette ten Teije. A-nesi: A scalable approximate method for probabilistic neurosymbolic inference. _arXiv preprint arXiv:2212.12393_ , 2022. 

- [35] Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. _Proceedings of the IEEE_ , 2021. 

- [36] Raphael Suter, Djordje Miladinovic, Bernhard Schölkopf, and Stefan Bauer. Robustly disentangled causal mechanisms: Validating deep representations for interventional robustness. In _International Conference on Machine Learning_ , pages 6056–6065. PMLR, 2019. 

- [37] Julius Von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. _Advances in neural information processing systems_ , 34: 16451–16467, 2021. 

- [38] Joaquin Quinonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence. _Dataset shift in machine learning_ . Mit Press, 2008. 

- [39] Emanuele Marconato, Andrea Passerini, and Stefano Teso. Glancenets: Interpretabile, leakproof concept-based models. _NeurIPS_ , 2022. 

- [40] Geoffrey E Hinton and Richard Zemel. Autoencoders, minimum description length and helmholtz free energy. _Advances in neural information processing systems_ , 6, 1993. 

- [41] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In _International conference on machine learning_ , 2014. 

12 

- [42] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In _International conference on machine learning_ , 2014. 

- [43] Partha Ghosh, Mehdi SM Sajjadi, Antonio Vergari, Michael J Black, and Bernhard Schölkopf. From variational to deterministic autoencoders. In _ICLR_ , 2020. 

- [44] Eleonora Misino, Giuseppe Marra, and Emanuele Sansone. VAEL: Bridging Variational Autoencoders and Probabilistic Logic Programming. _NeurIPS_ , 2022. 

- [45] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In _ICML_ , 2019. 

- [46] Francesco Locatello, Ben Poole, Gunnar Rätsch, Bernhard Schölkopf, Olivier Bachem, and Michael Tschannen. Weakly-supervised disentanglement without compromises. In _International Conference on Machine Learning_ , pages 6348–6359. PMLR, 2020. 

- [47] Rui Shu, Yining Chen, Abhishek Kumar, Stefano Ermon, and Ben Poole. Weakly supervised disentanglement with guarantees. In _ICLR_ , 2019. 

- [48] Rafael Müller, Simon Kornblith, and Geoffrey E Hinton. When does label smoothing help? _Advances in neural information processing systems_ , 32, 2019. 

- [49] Shuang Li, Yilun Du, Gido Martijn van de Ven, and Igor Mordatch. Energy-based models for continual learning. In _Energy Based Models Workshop-ICLR 2021_ , 2021. 

- [50] Qing Li, Siyuan Huang, Yining Hong, Yixin Chen, Ying Nian Wu, and Song-Chun Zhu. Closed loop neural-symbolic learning via integrating neural perception, grammar parsing, and symbolic reasoning. In _International Conference on Machine Learning_ , pages 5884–5894. PMLR, 2020. 

- [51] Hongxin Wei, Renchunzi Xie, Hao Cheng, Lei Feng, Bo An, and Yixuan Li. Mitigating neural network overconfidence with logit normalization. In _International Conference on Machine Learning_ , pages 23631–23644. PMLR, 2022. 

- [52] Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip Torr, and Puneet Dokania. Calibrating deep neural networks using focal loss. _Advances in Neural Information Processing Systems_ , 33:15288–15299, 2020. 

- [53] Luigi Carratino, Moustapha Cissé, Rodolphe Jenatton, and Jean-Philippe Vert. On mixup regularization. _The Journal of Machine Learning Research_ , 23(1):14632–14662, 2022. 

- [54] Yiran Xu, Xiaoyin Yang, Lihang Gong, Hsuan-Chu Lin, Tz-Ying Wu, Yunsheng Li, and Nuno Vasconcelos. Explainable object-induced action decision for autonomous vehicles. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , June 2020. 

- [55] Yoshihide Sawada and Keigo Nakamura. Concept bottleneck model with additional unsupervised concepts. _IEEE Access_ , 10:41758–41765, 2022. 

- [56] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. _Nature Machine Intelligence_ , 2(11):665–673, 2020. 

- [57] Sebastian Lapuschkin, Stephan Wäldchen, Alexander Binder, Grégoire Montavon, Wojciech Samek, and Klaus-Robert Müller. Unmasking clever hans predictors and assessing what machines really learn. _Nature communications_ , 10(1):1–8, 2019. 

- [58] Kai Yuanqing Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. Noise or signal: The role of image backgrounds in object recognition. In _ICLR_ , 2020. 

- [59] Patrick Schramowski, Wolfgang Stammer, Stefano Teso, Anna Brugger, Franziska Herbert, Xiaoting Shao, Hans-Georg Luigs, Anne-Katrin Mahlein, and Kristian Kersting. Making deep neural networks right for the right scientific reasons by interacting with their explanations. _Nature Machine Intelligence_ , 2(8):476–486, 2020. 

13 

- [60] Andrew Slavin Ross, Michael C Hughes, and Finale Doshi-Velez. Right for the right reasons: training differentiable models by constraining their explanations. In _Proceedings of the 26th International Joint Conference on Artificial Intelligence_ , pages 2662–2670, 2017. 

- [61] Giambattista Parascandolo, Alexander Neitz, ANTONIO ORVIETO, Luigi Gresele, and Bernhard Schölkopf. Learning explanations that are hard to vary. In _International Conference on Learning Representations_ , 2020. 

- [62] Stefano Teso, Öznur Alkan, Wolfang Stammer, and Elizabeth Daly. Leveraging explanations in interactive machine learning: An overview. _Frontiers in Artificial Intelligence_ , 2023. 

- [63] Wolfgang Stammer, Marius Memmel, Patrick Schramowski, and Kristian Kersting. Interactive disentanglement: Learning concepts by interacting with their prototype representations. _arXiv preprint arXiv:2112.02290_ , 2021. 

- [64] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. _IEEE transactions on pattern analysis and machine intelligence_ , 35(8): 1798–1828, 2013. 

- [65] Aapo Hyvärinen, Ilyes Khemakhem, and Hiroshi Morioka. Nonlinear independent component analysis for principled disentanglement in unsupervised deep learning. _Patterns_ , 4(10), 2023. 

- [66] Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ICA: A unifying framework. In _AISTATS_ , 2020. 

- [67] Simon Buchholz, Michel Besserve, and Bernhard Schölkopf. Function classes for identifiable nonlinear independent component analysis. _Advances in Neural Information Processing Systems_ , 35:16946–16961, 2022. 

- [68] Luigi Gresele, Julius Von Kügelgen, Vincent Stimper, Bernhard Schölkopf, and Michel Besserve. Independent mechanism analysis, a new concept? _Advances in neural information processing systems_ , 34:28233–28248, 2021. 

- [69] Kartik Ahuja, Divyat Mahajan, Yixin Wang, and Yoshua Bengio. Interventional causal representation learning. In _International conference on machine learning_ , pages 372–407. PMLR, 2023. 

- [70] Wendong Liang, Armin Keki´c, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele, and Bernhard Schölkopf. Causal component analysis. _arXiv preprint arXiv:2305.17225_ , 2023. 

- [71] Julius von Kügelgen, Michel Besserve, Wendong Liang, Luigi Gresele, Armin Keki´c, Elias Bareinboim, David M Blei, and Bernhard Schölkopf. Nonparametric identifiability of causal representations from unknown interventions. _arXiv preprint arXiv:2306.00542_ , 2023. 

- [72] Łukasz Maziarka, Aleksandra Nowak, Maciej Wołczyk, and Andrzej Bedychaj. On the relationship between disentanglement and multi-task learning. In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , pages 625–641. Springer, 2022. 

- [73] Sébastien Lachapelle, Tristan Deleu, Divyat Mahajan, Ioannis Mitliagkas, Yoshua Bengio, Simon Lacoste-Julien, and Quentin Bertrand. Synergies between disentanglement and sparsity: generalization and identifiability in multi-task learning. In _International Conference on Machine Learning_ , pages 18171–18206. PMLR, 2023. 

- [74] Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodolà, Stefano Soatto, Bernhard Schölkopf, and Francesco Locatello. Leveraging sparse and shared feature activations for disentangled representation learning. _arXiv preprint arXiv:2304.07939_ , 2023. 

- [75] Francesco Locatello, Michael Tschannen, Stefan Bauer, Gunnar Rätsch, Bernhard Schölkopf, and Olivier Bachem. Disentangling factors of variations using few labels. In _International Conference on Learning Representations_ , 2020. 

- [76] Weili Nie, Tero Karras, Animesh Garg, Shoubhik Debnath, Anjul Patney, Ankit B Patel, and Anima Anandkumar. Semi-supervised stylegan for disentanglement learning. In _Proceedings of the 37th International Conference on Machine Learning_ , pages 7360–7369, 2020. 

14 

- [77] Hao Tang and Kevin Ellis. From perception to programs: regularize, overparameterize, and amortize. In _International Conference on Machine Learning_ , pages 33616–33631. PMLR, 2023. 

- [78] Alessandro Daniele, Tommaso Campari, Sagar Malhotra, and Luciano Serafini. Deep symbolic learning: Discovering symbols and rules from perceptions. _arXiv preprint arXiv:2208.11561_ , 2022. 

- [79] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In _International Conference on Machine Learning_ , pages 5338–5348. PMLR, 2020. 

- [80] Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frederic Precioso, Stefano Melacci, Adrian Weller, et al. Concept embedding models. _arXiv preprint arXiv:2209.09056_ , 2022. 

- [81] Francesco Giannini, Michelangelo Diligenti, Marco Gori, and Marco Maggini. On a convex logic fragment for learning and reasoning. _IEEE Transactions on Fuzzy Systems_ , 2018. 

- [82] Thomas M Cover. _Elements of information theory_ . John Wiley & Sons, 1999. 

- [83] Kareem Ahmed, Kai-Wei Chang, and Guy Van den Broeck. A pseudo-semantic loss for deep generative models with logical constraints. In _Knowledge and Logical Reasoning in the Era of Data-driven Learning Workshop_ , July 2023. 

- [84] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- [85] Tommaso Carraro. LTNtorch: PyTorch implementation of Logic Tensor Networks, mar 2022. URL `https://doi.org/10.5281/zenodo.6394282` . 

- [86] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. URL `http://arxiv.org/abs/1412.6980` . 

- [87] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in neural information processing systems_ , 28, 2015. 

15 

## **A Other NeSy Predictors** 

In this appendix, we outline the NeSy prediction approaches used in our experiments and then show that they share deterministic RSs as DPL. 

The **semantic loss** (SL) [8] is a penalty term that encourages a neural network to place all probability mass on predictions that are consistent with prior knowledge K. In our setting, the SL is applied to a predictor _pθ_ ( **Y** _|_ **C** ) placed on top of a concept extractor _pθ_ ( **C** _|_ **X** ) and it can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0016-03.png)


Like DPL, the SL relies on knowledge compilation to efficiently implement Eq. (11). Importantly, if the distribution _pθ_ ( **c** _|_ **x** ) allocates mass _only_ to concepts **c** that satisfy the knowledge given **y** , these are optimal solutions, in the sense that the SL is exactly _zero_ . We will make use of this fact in Appendix A.1. 

During training, the SL is combined with any regular supervised loss _ℓ_ , for instance the cross-entropy, leading to the overall training loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0016-06.png)


with _µ >_ 0 a hyperparameter. During inference, the SL plays no role: the predicted label is obtained by simply taking the most likely configuration through a forward pass over the network. 

**Logic tensor networks** (LTNs) [6] are another state-of-the-art NeSy architecture that combines elements of reasoning- and penalty-based approaches. The core idea behind LTNs, and of all other NeSy predictors based of fuzzy logic [81], is to _relax_ the prior knowledge K into a real-valued function _T_ [K] _quantifying_ how close a prediction is to satisfying K. LTNs perform this transformation using _product real logic_ [6]. In our context, this function takes the form _T_ [K] : [0 _,_ 1] _[k] ×_ [0 _,_ 1] _[n] →_ [0 _,_ 1], and it takes as input the probabilities of the various concepts **C** and labels **Y** and outputs a degree of satisfaction. 

Crucially, fuzzy logics are designed such that, if all probability mass is allocated to configurations ( **c** _,_ **y** ) that _do_ satisfy the logic, then the degree of satisfaction is exactly 1, _i.e._ , maximal. We will leverage this fact in Appendix A.1. 

During training, LTNs guide the concept extractor _pθ_ ( **C** _|_ **X** ) towards predicting concepts that satisfy the prior knowledge K by penalizing it proportionally to how far away their predictions are from satisfying K given the ground-truth label **y** , that is, 1 _−T_ [K]( _p_ ( **C** _|_ **x** ) _,_ 1 _{_ **Y** = **y** _}_ ). During inference, LTNs first predict the most likely conceptsthrough the network, and then predict a label **y** ˆ that maximally satisfies the knowledge given ˆ **c** = argmax **c** _pθ_ ( **c** _|_ **x** ) using a forward pass ˆ **c** , again according to _T_ [K]. 

## **A.1 Deterministic Optima are Shared** 

The bulk of our theoretical analysis focuses on DPL because it offers a clear probabilistic framework, however in the following we show that deterministic optima do transfer to other NeSy predictors approaches. Specifically, under **A2** , DPL, SL, and LTN all admit the same deterministic optima (det-opts). 

To see this, let _C_ **y** be the set of concepts vectors **c** from which the label **y** can be inferred, that is, _C_ **y** = _{_ **c** _∈C_ : _p[∗]_ ( **y** _|_ **c** ) _>_ 0 _}_ . Now, take a deterministic RS _pθ_ ( **C** _|_ **X** ) for DPL, _i.e._ , a concept distribution that satisfies Definition 1 when the likelihood is computed as in Eq. (2). By determinism, this distribution can be equivalently written as a function mapping inputs **x** to concept vectors **c** ˆ( **x** ) _∈C_ , that is, _pθ_ ( **C** _|_ **X** = **x** ) = 1 _{_ **C** = ˆ **c** ( **x** ) _}_ . At the same time, by optimality and **A2** we have that, for every ( **x** _,_ **y** ) _∈D_ , the NeSy predictor _pθ_ ( **Y** _|_ **X** ; K) =[�] **c** _[u]_[K][(] **[Y]** _[|]_ **[ c]**[)] _[ ·][ p][θ]_[(] **[c]** _[ |]_ **[ X]**[)][ allocates] all probability mass to the correct label **y** . As a consequence, ˆ **c** ( **x** ) _∈C_ **y** , for every ( **x** _,_ **y** ) _∈D_ . 

Considerreasoning layer based on fuzzy logic,a NeSy predictor that first predictssuch that a predictionconcepts using **y** ˆ _pθ∈Y_ ( **C** _|_ is chosen so that it minimial **X** ) and then labels using a distance from satisfaction w.r.t. K, and the distance from satisfaction is also used as a training objective. Since **c** ˆ _∈C_ **y** for every ( **x** _,_ **y** ) _∈D_ , by definition of T-norms, the label with minimal distance from satisfaction will necessarily be **y** : all other labels cannot be inferred from the knowledge, 

16 

so they have larger distance from satisfaction. Hence, training loss will be minimal for this predictor as well, meaning that _pθ_ ( **C** _|_ **X** ) is a deterministic RS for it too. 

Next, consider a NeSy penalty-based predictor where predictions are obtained by first inferring concepts using MAP over the concept extractor, that is, ˆ **c** = argmax **c** _pθ_ ( **c** _|_ **X** ), and then performing a forward pass over a neural prediction layer _pθ_ ( **Y** _|_ **C** ). By construction, the penalty will be minimal whenever ( **y** _,_ ˆ **c** ( **x** )) _|_ = K. We already established that this is the case for all ( **x** _,_ **y** ) _∈D_ , meaning that _pθ_ ( **C** _|_ **X** ) is a deterministic RS for penalty-based predictors too. 

## **B Proofs** 

In the proofs, we suppress K from the notation for readability. 

## **B.1 Proof of Lemma 1: Upper Bound of the Log-Likelihood** 

**Proof plan.** The proof of the first point of our claim is split in three parts: 

1. We show that in expectation the log-likelihood in Eq. (2) is upper bounded by a term containing the KL divergence for _pθ_ ( **Y** _|_ **G** ). 

2. We prove that, under **A1** , any optimum of Eq. (2) minimizes also the KL. In this step, we make use of Information Theory [82] to connect the two. 

3. We show that assuming **A1** and **A2** the optima _pθ_ ( **Y** _|_ **G** ) for the KL are given only by optima _pθ_ ( **Y** _|_ **X** ) for Eq. (2). 

We proceed to prove the second point by leveraging the fact that _pθ_ ( **C** _|_ **G** ) is given by marginalizing _pθ_ ( **C** _|_ **X** ) over the generating distribution _p[∗]_ ( **X** _|_ **G** _,_ **S** ). 

**Point (** _**i**_ **).** (1) We upper bound the log-likelihood in Eq. (1) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0017-11.png)


In the second line, we introduced the distribution _p[∗]_ ( **x** _|_ **g** ) := E **s** _∼p_ ( **S** )[ _p[∗]_ ( **x** _|_ **g** _,_ **s** )], in the third line we applied Jensen’s inequality, and in the fifth line we added and subtracted _p[∗]_ ( **y** _|_ **g** ). This proves our claim. 

(2) We proceed showing that for a deterministic _p[∗]_ ( **X** _|_ **G** _,_ **S** ) every optimum _pθ_ ( **Y** _|_ **X** ) for the log-likelihood leads to an optimum _pθ_ ( **Y** _|_ **G** ) for the RHS of Eq. (29). We can rewrite the joint distribution as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0017-14.png)


17 

where _p[∗]_ ( **y** _|_ **x** ) = E **g** _∼p∗_ ( **g** _|_ **x** )[ _p[∗]_ ( **y** _|_ **g** )] and _p[∗]_ ( **G** _|_ **x** ) is the posterior underlying the data generative process. Hence, the log-likelihood in Eq. (1) can be equivalently written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-02.png)


In the first line, we used _p_ ( **X** _,_ **Y** ) = _p[∗]_ ( **Y** _|_ **X** ) _p_ ( **X** ), and then added and subtracted log _p[∗]_ ( **y** _|_ **x** ). By comparing Eq. (18) and Eq. (27), we obtain: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-04.png)


Now, consider a distribution _pθ_ ( **Y** _|_ **X** ) that attains maximal likelihood. Then, KL[ _p[∗]_ ( **y** _|_ **x** ) _∥pθ_ ( **y** _|_ **x** )] = 0, and we can rearrange the inequality in Eq. (28) to obtain: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-06.png)


Here, H[ **Y** _|_ **X** ] = E **x** _∼p_ ( **X** )[H[ _p[∗]_ ( **Y** _|_ **x** )] and H[ **Y** _|_ **G** ] = E **g** _∼p_ ( **G** )H[ _p[∗]_ ( **Y** _|_ **g** )] are conditional entropies. 

We want to show that, under **A1** , the right-hand side of Eq. (29) is in fact zero. As for the other conditional entropy term, recall that [82]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-09.png)


where I[ _·, ·_ ] is the mutual information. By the chain rule of the mutual information, cf. [82, Theorem 2.8.1], we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-11.png)


where I[ _·, · | ·_ ] is the conditional mutual information. The structure of our generative process in Fig. 1 implies that I[ **X** : **Y** _|_ **G** ] = 0, so Eq. (31) boils down to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-13.png)


By **A1** , we have that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-16.png)


Plugging this into Eq. (32) entails that I[ **X** : **Y** ] = I[ **Y** : **G** ], or equivalently that H[ **Y** : **X** ] = H[ **Y** : **G** ]. This means that the right-hand side of Eq. (29) is indeed zero, which entails that the KL is zero and that therefore _pθ_ ( **Y** _|_ **G** ) optimizes the right-hand side of Eq. (5). 

(3) We proceed showing that by assuming also **A2** whatever optimum _pθ_ ( **Y** _|_ **G** ) is identified by _pθ_ ( **Y** _|_ **X** ) that is also optimum. First, note that under **A2** _p[∗]_ ( **Y** _|_ **G** ) is deterministic, so we have H[ **Y** _|_ **G** ] = 0. With **A1** and **A2** , both _p[∗]_ ( **y** _|_ **g** ) and _p[∗]_ ( **y** _|_ **x** ) are deterministic and therefore Eq. (28) can be rewritten as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0018-19.png)


In particular, for each **x** the maximum of the log-likelihood is 0 and it is attained when the label probability is one. We make use of this observation in the next step. 

Next, we show that the bound in Eq. (28) is in fact _tight_ , in the sense that whenever _pθ_ ( **Y** _|_ **G** ) maximizes the left-hand side, _pθ_ ( **Y** _|_ **X** ) maximizes the right-hand side. We proceed by contradiction. Fix **g** and let _O_ be the set of those **s** for which _pθ_ ( **y** _|_ **x** = _f_ ( **g** _,_ **s** )) _<_ 1 and assume that it has 

18 

non-vanishing measure. Then, the posterior distribution is also strictly less than one: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-02.png)


Therefore, there cannot be any optimal solutions _pθ_ ( **Y** _|_ **G** ) that are given by non-optimal probabilities _pθ_ ( **Y** _|_ **X** ). This proves the claim. 

**Point (** _**ii**_ **).** We consider now which distributions _pθ_ ( **C** _|_ **G** ) correspond to a unique distribution _pθ_ ( **C** _|_ **X** ). First, we define as _P_ the set of candidate distributions _pφ_ ( **C** _|_ **X** ), with _φ ∈_ Θ, for which it holds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-05.png)


For all distributions of the form _pθ_ ( **C** _|_ **G** ) = 1 _{_ **C** = **c** _}_ , for **c** _∈C_ , _P_ restricts to a single element, i.e., _pφ_ ( **C** _|_ **x** ) = 1 _{_ **C** = **c** _}_ . We proceed by contradiction and consider a set _O_ **X** of non-vanishing measure such that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-09.png)


where we denoted _p_ ˜ _φ_ ( **C** _|_ **G** ) the normalized probability distribution given by integrating _pφ_ ( **C** _|_ **X** ) on _iff Oλ_ **X** =solely, and0 or _p_ ˜ _φ_ ( **C** _λ_ is the measure of _|_ **G** ) = 1 _{_ **C** = **c** _O}_ **X** , .whichNotice that the RHS ofcontradicts the claim. Eq. (44)Hence, is exactlyall probabilities 1 _{_ **C** = **c** _} pθ_ ( **C** _|_ **G** ) = 1 _{_ **C** = **c** _}_ are only given by probabilities _pθ_ ( **C** _|_ **x** ) = 1 _{_ **C** = **c** _}_ , for all **x** _∈X_ . This yields the claim. 

## **B.2 Proof of Theorem 2: Counting the Deterministic Optima** 

We want to count the number of deterministic optima of the log-likelihood in Eq. (1). Recall that, by Lemma 1, under **A1** and **A2** any optimum _pθ_ ( **Y** _|_ **X** ) of Eq. (2) yields an optimum _pθ_ ( **Y** _|_ **G** ) of the upper bound in Eq. (5). Following, by point ( _ii_ ) of Lemma 1 we have that deterministic optima are shared between _pθ_ ( **C** _|_ **X** ) and _pθ_ ( **C** _|_ **G** ). This means that we can equivalently count the number of deterministic optima _pθ_ ( **C** _|_ **G** ) for the upper bound in Eq. (5). We proceed to do exactly this. 

Let _A_ be the set of all possible maps _α_ : **g** _�→_ **c** , each inducing a candidate concept distribution _pθ_ ( **C** _|_ **G** ) = 1 _{_ **C** = _α_ ( **G** ) _}_ . The only _α_ ’s that achieve maximal likelihood are those that satisfy the knowledge for all **g** _∈_ supp( **G** ) for the learning problem, that is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-14.png)


_i.e._ , that it is indeed the case that the concepts output by _α_ ( **g** ) predict the ground-truth label _h_ K( **g** ). Notice that, only one of them is correct and coincides with the identity, _i.e._ , _α_ ( **g** ) = **g** . 

These _α_ are those that satisfy the knowledge on all examples, or equivalently the _conjunction_ of the knowledge applied to all examples, that is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0019-17.png)


19 

This means that only a subset of _A_ contains those maps consistent with the knowledge. The total number of these maps is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0020-01.png)


This yields the claim. 

## **B.3 Proof of Proposition 3: Link Between Deterministic and Non-deterministic Optima** 

**Point (** _**i**_ **)** . We begin by proving that, for DPL, any convex combination of optima of the likelihood Eq. (1) is itself an optimum. Fix any input **x** . First, _p[∗]_ ( **y** _|_ **x** ) is the optimal value of the log-likelihood in Eq. (2), according to Eq. (29) from point ( _i_ ) of Lemma 1. Now, let _p_[(1)] ( **C** _|_ **x** ) and _p_[(2)] ( **C** _|_ **x** ) be two concept distributions that both attain optimal likelihood, _i.e._ , for _i ∈{_ 1 _,_ 2 _}_ it holds that � **c** _[u]_[K][(] **[y]** _[ |]_ **[ c]**[)] _[ ·][ p]_[(] _[i]_[)][(] **[c]** _[ |]_ **[ x]**[) =] _[ p][∗]_[(] **[y]** _[ |]_ **[ x]**[)][.][The likelihood term of any convex combination of the two] optima is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0020-05.png)


Hence, the convex combination is also an optimum of the likelihood. Note that the very same reasoning applies to the Semantic Loss (Eq. (11)), again due to linearity of the expectation over **C** . **Remark 7.** _Since SL and DPL limit pθ_ ( **C** _|_ **X** ) _to be factorized as_[�] _i[p]_[(] _[C][i][|]_ **[X]**[)] _[,][some][convex] combinations that would be in principle solutions as per point_ _**(i)** cannot be expressed. Essentially, this translates into an additional constraint for the solutions of SL and DPL that is, given two factorized probabilities p_[(1)] ( **C** _|_ **x** ) _and p_[(2)] ( **C** _|_ **x** ) _, then:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0020-07.png)


_This applies only to models that integrate probabilistic logic by predicting the concepts independently, whereby relaxations of SL and DPL, like [10, 83], could naturally express arbitrary convex combinations of deterministic RSs._ 

**Remark 8.** _The above result does not hold for LTNs, in general. We show this by constructing a counter-example for the default choice of T-conorms [6]. That is, a convex combination of deterministic optima that is not itself an optimum. Recall that LTN uses product real logic to define a degree of knowledge satisfaction and consider the prior knowledge_ K = ( _C_ 1 _∨ C_ 2) _, where C_ 1 _and C_ 2 _are two distinct binary concepts. In LTN, the degree of satisfaction of_ K _is given by the T-conorm of the logical disjunction, which is defined as S_ ( _a, b_ ) = _a_ + _b − ab. Fix an input_ **x** _and consider two deterministic distributions p_[(1)] ( **C** _|_ **x** ) = 1 � **C** = (1 _,_ 0) _[⊤]_[�] _and p_[(2)] ( **C** _|_ **x** ) = 1 � **C** = (0 _,_ 1) _[⊤]_[�] _. It is clear that both distributions satisfy the prior knowledge and as such are optimal. Now take any convex combination p_ ( **C** ) = _λp_[(1)] ( **C** ) + (1 _− λ_ ) _p_[(2)] ( **C** ) _. Then, a_ = _λp_[(1)] ( _C_ 1 = 1) + (1 _− λ_ ) _p_[(2)] ( _C_ 1 = 1) = _λ and, for similar reasons, b_ = 1 _− λ. Then, it holds:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0020-10.png)


_where the equality holds iff λ ∈{_ 0 _,_ 1 _}._ 

**Point (** _**ii**_ **).** Under **A1** and **A2** , we can count all deterministic solutions _pθ_ ( **C** _|_ **G** ) via Theorem 2. Here, we show that these deterministic optima constitute a _complete_ basis for all optimal solutions of Eq. (29). From ( _i_ ), we have that any convex combination of the deterministic optima is also an optimum. We will show that these are the _only_ optimal solutions. 

Recall that the optimal solutions are all of the form _p[∗]_ ( **Y** = _β_ K( **g** ) _|_ **g** ) = 1, as a consequence of **A1** and **A2** from Lemma 1, for all **g** _∈_ supp( **G** ). Notice that any optimal solution _pθ_ ( **C** _|_ **g** ) must place 

20 

mass to those concepts that lead to the correct label. Formally, for every **g** and **y** = _β_ K( **g** ), it holds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0021-01.png)


where the equality holds _iff u_ K( **y** _|_ **c** ) = 1 for all _pθ_ ( **c** _|_ **x** ) _>_ 0. In other words, this means that each **c** _∼ pθ_ ( **C** _|_ **g** ) must be an optimal solution for the logic. This proves the claim. 

**Point (** _**iii**_ **)** . If **A2** does not hold, there can be optima solutions that are given as convex combinations of non-optimal deterministic probabilities. First, notice that according to Lemma 1, point ( **i** ), the optimal solutions for _pθ_ ( **Y** _|_ **X** ; K) minimize the KL term and are equivalent to _p[∗]_ ( **Y** _|_ **X** ). Then, consider for a given **x** an optimal solution _p_ ( **C** _|_ **x** ) = _λ_ 1 _{_ **C** = **c** 1 _}_ + (1 _− λ_ ) 1 _{_ **C** = **c** 2 _}_ that is a convex combination of two deterministic probabilities. From the convexity of the KL it holds: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0021-04.png)


where the equality holds _iff u_ K( **y** _|_ **c** 1; K) = _u_ K( **y** _|_ **c** 2; K) = _p[∗]_ ( **y** _|_ **x** ). This shows that _there can exist solutions that are convex combinations of non-optimal deterministic probabilities_ . We proceed to show how the space of optimal solutions is defined. On the converse, if _u_ K( **y** _|_ **c** 1; K) 1 _{_ **C** _−_ **c** 1 _}_ is a solution, also _u_ K( **y** _|_ **c** 2; K) 1 _{_ **C** _−_ **c** 2 _}_ must be a solution: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0021-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0021-07.png)


We now show that any optimum given by a generic _pφ_ ( **C** _|_ **x** ), with _φ ∈_ Θ, can be expressed as a convex combination of (1) an optimum that is a convex combination of deterministic optima and (2) an optimum that is a convex combination of deterministic, but non-optimal, probabilities: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0021-09.png)


where _C_ **y** = _{_ **c** _∈C_ : _u_ K = _p[∗]_ ( **y** _|_ **x** ) _}_ and _S_ **y** _[c]_[=] _[C][/][C]_ **[y]**[are][two][disjoint][sets.][In][the][second] line, we rewrote the summation on the two terms considering the two sets, whereas in the third lineandmust _p_ ¯( **c** _p_ we _|_ ¯be( **xc** )introduced:that _|_ leading **x** )also= _p_ to _p_ (¯ **c** ( **c** _λ_ optima, _| |_ = **xx** )) _/_[�] (1is **c** _−_ even _∈C_ an **y** _λ_ optimum, _[p]_ )when[(] . **[c]** Since _[|]_ **[x]** _p_[)] ˜([and] **c** eachby _|_ the _[p]_[˜] **x**[(] _p_ )˜ **[c]** (reducesprevious **c** _[|]_ **[x]** _|_[)] **x** )[=] leadto _[p]_ point.[(] **[c]** onlyto _[|]_ **[x]** an[)] Inthe _[/λ]_ optimumgeneral,[,] ground-truth[1] _[ −][λ]_[=] thereby[�] construction, **c** canelement, _∈S_ **y** ¯ be _[p]_[(] **[c]** many _[|] i.e._ **[x]** it[)] , _p_ ˜( **c** _|_ **x** ) = 1 � **C** _− f_ 1: _[−] k_[1][(] **[x]**[)] �. Notice that this must hold for all **x** _∼ p[∗]_ ( **X** ). This proves the claim. 

## **B.4 Proof of Proposition 4: Multi-task Learning** 

When considering multiple task _T_ , suppose that, for each **x** _∈_[�] _t[D][t]_[,][we][get] _[T]_[different][labels] **y**[(1)] _, . . . ,_ **y**[(] _[T]_[ )] , each given in accordance to knowledge K[(1)] _, . . . ,_ K[(] _[T]_[ )] , respectively. Similarly to Eq. (2), we consider the learning objective for all tasks with the joint log-likelihood term: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0021-13.png)


21 

Under **A1** and **A2** , Lemma 1 point ( _i_ ) holds for each task _t_ . In the following, we denote with _β_ K[(] _[t]_[)] the underlying maps for each K[(] _[t]_[)] , which by **A2** are deterministic. Hence, the learning objective becomes: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-01.png)


and, similarly to point ( _i_ ) of Lemma 1, we get the following upper-bound: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-03.png)


Notice that the optimal values for Eq. (69) are given by those distributions that are one for each term in the summation. Then, following from Theorem 2, we have that the deterministic maps _α_ ’s that optimize Eq. (69) must be consistent with each task _t_ and for each **g** _∈_ supp( **G** ) satisfy: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-05.png)


which, equivalently to Theorem 2, points to the condition that the maps _α_ must be consistent with solving all tasks _t_ . This exactly amounts to solving the conjunction of all knowledge K =[�] _t_[K][(] _[t]_[)][.] This yields the claim. 

## **B.5 Proof of Proposition 5: Concept Supervision** 

Let _p[∗]_ ( **G** _| S_ ) be the distribution of ground-truth concepts restricted to a subset _S ⊆_ supp( **G** ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-09.png)


and _p[∗]_ ( **X** _| S_ ) = E( **g** _,_ **s** ) _∼p∗_ ( **G** _|S_ ) _p∗_ ( **S** ) _p[∗]_ ( **X** _|_ **g** _,_ **s** ) be the corresponding restricted input distribution. Under **A1** , the expectation of the log-likelihood term for concept supervision in Section 5 is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-11.png)


where _p[∗]_ ( **X** _| S_ ) = E **g** _∼p∗_ ( **G** _|S_ )E **s** _∼p∗_ ( **S** ) _pθ_ ( **X** _|_ **g** _,_ **s** ). In the first line we made use of **A1** to write **x** = _f_ ( **g** _,_ **s** ), in the second line we used Jensen’s inequality, and then we introduced the marginal distribution _pθ_ ( **C** _|_ **G** ) = E **s** _∼p∗_ ( **s** )[ _pθ_ ( **C** _| f_ ( **G** _,_ **s** ))]. Recall that _I_ denotes the subset of the supervised ground-truth concepts and that the log-likelihood is evaluated only w.r.t. those concepts. 

Now notice that, given **A1** , the maximum for the LHS of Eq. (72) is zero, _i.e._ , coincides with maximum log-likelihood. Consistently the RHS of Eq. (72) is zero only when _pθ_ ( **C** _I |_ **g** ) places all mass on **g** _I_ . We proceed considering those maps _α_ : **g** _�→_ **c** which lead to deterministic, optimal solutions for the RHS. For these, it must hold that for each **g** _∈S_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-14.png)


When taken all together, we can estimate how many _α ∈A_ (cf. Appendix B.2) satisfy the above condition for all **g** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-16.png)


This yields the claim. 

## **B.6 Proof of Proposition 6: Reconstruction** 

Under **A1** , the reconstruction penalty can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0022-20.png)


22 

From **A3** , we have that the both encoder and decoder distributions are factorized: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0023-01.png)


This yields: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0023-03.png)


where in the first row we separated the two logarithms and removed the expectations on _pθ_ ( **Z** _|_ **s** ) and _pθ_ ( **C** _|_ **g** ) for the terms _pψ_ ( **g** _|_ **c** ) and _pψ_ ( **s** _|_ **z** ), respectively. In the third line, we discarded the term with the match of the reconstruction of the style, giving the lower bound. 

When _R_ goes to zero, the lower bound also goes to zero and the last term of Eq. (79) is minimized. This happens whenever, for each **c** _∼ pθ_ ( **c** _|_ **g** ), it holds that _pψ_ ( **g** _|_ **c** ) = 1. This condition also implies that if for two different **g** _[′]_ and **g** _[′′]_ there exist at least one **c** such that _pθ_ ( **c** _|_ **g** _[′]_ ) _·pθ_ ( **c** _|_ **g** _[′′]_ ) _>_ 0, then _pψ_ ( **g** _[′] |_ **c** ) and _pψ_ ( **g** _[′′] |_ **c** ) cannot both be optimal. 

We restrict now to a deterministic map _α_ : **g** _�→_ **c** for _pθ_ ( **C** _|_ **G** ) and describe the condition when optimal decoders are attained, _i.e._ , _pψ_ ( **g** _|_ **c** ) = 1 for some **c** . By the previous argument, an optimal map _α_ that leads to optimal decoders must always map ground-truth concepts **g** to different concepts **c** , that is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0023-07.png)


In particular, this condition must hold for all different **g** _∈_ supp( **G** ). The number of all solutions is then given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0023-09.png)


as claimed. 

## **C Experimental Details and Further Results** 

We report here all further details concerning the experiments in Section 6. 

## **C.1 Implementation** 

The code of the experiments builds on top of `nesy-cl` [21] and `CBM-AUC` [55]. All experiments are implemented with Python 3.8.16 and Pytorch [84] and run over one A100 GPU. The code for the experiments is available at github.com/reasoning-shortcuts. 

The implementation of DPL is taken from [21] and follows exactly Eq. (1), where to each world a label is assigned accordingly to the prior knowledge K. We implemented SL following the original paper [8], with the only difference that the prediction of the labels **Y** is done on top of the logits of _pθ_ ( **C** _|_ **x** ). This relaxes the conditional independence between labels and concepts while being more in line with the generative process we assumed. The implementation of LTN is adapted from `LTN-pytorch` [85] and is based on the satisfaction loss introduced in Appendix A. 

## **C.2 Data sets & Count of the Reasoning Shortcuts** 

Here, we illustrate how to count explicitly the number of deterministic RSs using the equations in Table 1, restricting ourselves to the case where no disentanglement is in place, for simplicity. Similarly to Section 4, we assume **A1** and **A2** and that supp( **G** ) = _C_ = _G_ , and count the total number of deterministic optima (or _det-opt_ s for short) under different mitigation strategies. In the following, _C_ **y** _⊆C_ = _G_ refers to the set of **c** _∈C_ or **g** _∈G_ that are mapped by K to label **y** _∈Y_ . Note that � **y** _[|C]_ **[y]** _[|]_[ =] _[ |G|]_[.] 

**Explicit count for the likelihood.** When the concept extractor _pθ_ ( **C** _|_ **X** ) is sufficiently expressive, the deterministic mappings _α_ are essentially arbitrary. Specifically, the set _A_ of these _α_ ’s includes all functions from _G_ to _C_ . These functions can be explicitly enumerated by counting how many ways there are to map each input vector **g** to an arbitrary vector **c** . 

23 

Here, we are interested in counting the number of _α_ ’s that attain optimal likelihood. Each such _α_ has to ensure that each **g** _∈G_ is mapped to a **c** _∈Cβ_ K( **g** ) that yields the correct label, or in short, _∀_ **g** _. α_ ( **g** ) _∈Cβ_ K( **g** ). This condition is satisfied if and only if, for every **y** in the data set, every **g** _∈C_ **y** is mapped to a **c** that is also in _C_ **y** , and there are exactly _|C_ **y** _|[|C]_ **[y]** _[|]_ ways to do this. This immediately shows that the overall number of _det-opt_ s _α_ is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0024-01.png)


This associates an explicit number to Theorem 2. 

**Explicit count for the reconstruction.** The effect of adding a reconstruction penalty is that now, in order to achieve optimal loss, _α_ has to map different **g** ’s to distinct concepts **c** ’s. Unless this is the case, it becomes impossible to reconstruct the ground-truth concepts from the learned ones, and therefore the input **x** generated from them. 

The resulting computation is the same as above, except that now we have to associate the different **g** ’s to different **c** ’s _without replacement_ . This yields the following count: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0024-05.png)


**Explicit count for concept supervision.** For the combination of logic and concept supervision, we suppose that for each equivalence class _C_ **y** , there are _ν_ **y** ground-truth concepts provided with supervision and that all concept dimensions _Ci_ receive this supervision. Let _S ⊂G_ be the set of supervised concepts, such that _|S|_ =[�] **y** _∈Y[ν]_ **[y]**[.][From above, this means that we are specifying a] total of _|S|_ ground-truth concepts. From Table 1, we have that the number of _det-opt_ s amounts to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0024-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0024-08.png)


**The disentangled case.** If the network is _disentangled_ , the enumeration procedure becomes substantially more complicated and cannot be written compactly in closed form. While the number of deterministic optima _α_ can still be computed exactly using model counting [29, 30], doing so is not necessary for the scope of our paper and therefore left to future work. 

## **C.2.1 Dataset:** `XOR` 

This dataset, introduced in Example 3, is a toy data set containing 3 bits **g** = ( _g_ 1 _, g_ 2 _, g_ 3), for a total of 8 possible combinations. The task consists in predicting the XOR operation among them, namely _y_ = ( _g_ 1 _⊕ g_ 2 _⊕ g_ 3). The dataset is exhaustive and has no validation and test set. The model performances are evaluated on the training set. 

**Reasoning shortcut.** For this dataset, we have _|C_ 0 _|_ = _|C_ 1 _|_ = 4. RSs arise depending on the structure of the underlying network. When the ground-truth concepts are processed all together without any mitigation strategy, we obtain that the number of _det-opt_ s amounts to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0024-13.png)


24 

The confusion matrices for all methods are reported in Appendix C.5. On the other hand, we show that in the _disentangled_ case, only two combinations suffice to identify the correct solution, _e.g._ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0025-01.png)


and here the only viable solution for the two is _α_ (0) = 0 and _α_ (1) = 1. This condition is met in all our experiments in Table 2. 

## **C.2.2 Dataset:** `MNIST-Addition` 

We consider the version introduced in [7], which consists of couples of digits, each ranging from 0 to 9, and the target consists in predicting the correct sum, _i.e._ , _y_ = _g_ 1 + _g_ 2. This data set contains all possible combinations, for a total of 100. The training set contains 42k data, the validation set 12k, and the test set 6k. 

**Reasoning shortcuts.** RSs arise only as a result of the joint prediction of both digits. Notice that the number of elements _|Cy|_ for each sum **y** can be evaluated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0025-06.png)


Therefore, the total number of _det-opt_ s amounts to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0025-08.png)


When providing _disentanglement_ , the number of RSs reduce to 0, as it sufficient to have the sums: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0025-10.png)


to uniquely identify the value of the digit _c_ . 

## **C.2.3 Dataset:** `MNIST-EvenOdd` 

This data set, proposed by [21], is a biased version of `MNIST-Addition` , where only some combinations of the digits appear. We consider here a more challenging scenario w.r.t. the proposed version, consisting of the sums: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0025-14.png)


Overall, the training set contains 6720 data, the validation set 1920, and the test set 960. 

**Reasoning shortcuts.** We describe the RSs that arise even when the architecture incorporates _disentanglement_ . We evaluate the possible RSs empirically noticing that the system of observed sums can be written as a linear system, as done by Marconato et al. [21]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0025-17.png)


Now, notice that we can find independent reasoning shortcuts for each of the two sides since they do not share any digits. For the LHS, we consider the sum _α_ (2) + _α_ (8) = 10 and notice that we can find at most 10 different attributions for having a correct sum. Notice that, some of them are not allowed as _α_ (8) = 0 _,_ 1 leads to inconsistent values for the fourth sum, and _α_ (8) = 3 leads to an inconsistent sum for the first equation. So in total, for the LHS, we obtain 7 possible solutions and, by symmetry, the same number also for the LHS. In total, the number of _det-opt_ s is equal to 7 _·_ 7. 

Experimentally, we consider limited concept supervision on _I_ = _{_ 4 _,_ 9 _}_ which should be sufficient, in principle, to disambiguate between even and odds digits. This happens because specifying _α_ (4) = 4 and _α_ (9) = 9 admits only the ground-truth concepts as the optimal solution. 

25 

## **C.2.4 Dataset:** `MNIST-AddMul` 

In this dataset, we consider fewer combinations of digits, explicitly: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0026-02.png)


and similarly for multiplication. This data set contains 1680 training examples, 480 for the validation, and 240 for the test set. 

**Reasoning shortcuts.** For the case of the addition task, we can have 2 possible solutions, which are: 

- _α_ (0) = 0, _α_ (1) = 1, _α_ (2) = 2, and _α_ (3) = 3; 

- _α_ (0) = 1, _α_ (1) = 0, _α_ (2) = 1, and _α_ (3) = 4. 

For multiplication, we have that since _α_ (1) _· α_ (3) = 3 it can be either _α_ (1) = 1 or _α_ (1) = 3. In both cases, it holds that _α_ (0) = 0 and _α_ (2) can be arbitrary. Hence, there are in total 2 _·_ 4 possible _det-opt_ s. In MTL, the reasoning shortcut for the addition does not hold since it leads to a sub-optimal solution for multiplication. 

## **C.2.5 Dataset:** `BDD-OIA` 

This data set contains frames of driving scene videos for autonomous predictions [54]. Each frame is annotated with 4 binary labels, indicating the possible actions, **Y** = ( `move_forward` _,_ `stop` _,_ `turn_left` _,_ `turn_right` ). Each scene is also annotated with 21 binary concepts **C** , underlying the _reasons_ for the possible actions, see Table 5. The training set contains 16k frames, with full label and concept supervision; the validation and the test set contain 2k and 4 _._ 5k annotated data, respectively. 

For designing the prior knowledge w.r.t. to the concepts in Table 5, we make use of the following rules for `move_forward` _/_ `stop` _predictions_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0026-11.png)


For `turn_left` , and similarly for `turn_right` , we use: 

 `can_turn` = `left_lane` _∨_ `left_green_light` _∨_ `left_follow cannot_turn` = `no_left_lane` _∨_ `left_obstacle` _∨_ `left_solid_line` (97)   `can_turn` _∧¬_ `cannot_turn` = _⇒_ `turn_left` 

Notice that, since the concepts are predicted together, as explained in Appendix C.4, we can count the number of RSs as follows: 

- For `move_forward` and `stop` , the labels are predicted with the constraints such that ( `move_forward` _,_ `stop` ) = (1 _,_ 1) has no support. Hence, we consider only the predictions (0 _,_ 0) _,_ (0 _,_ 1) _,_ (1 _,_ 0). Next, we identify: 

- _|C_ 0 _,_ 0 _|_ = 1, since it corresponds just to the case where no concepts are predicted. 

- _|C_ (1 _,_ 0) _|_ = 2[3] _−_ 1, which is the number of different concepts attribution for forward yielding a positive label; 

- _|C_ (0 _,_ 1) _|_ = 280 are the combination of the remaining concepts that yield the `stop` action, in agreement with the constraints. These were counted explicitly from the logic implementation. 

- Overall, the number of _det-opt_ s amount to 1 _·_ 7[7] _·_ 280[280] . 

- For `turn_left` and `turn_right` , we count the cardinality of positive and negative predictions of the two classes _|C_ 0 _|_ and _|C_ 1 _|_ : 

26 

**–** _|C_ 1 _|_ = 2[3] _−_ 1, that are the only concepts attributions for the positive label; **–** _|C_ 0 _|_ = 2[6] _−|C_ 1 _|_ , are all the remaining concept combinations. For left and right, separately, we obtain that possible optimal solutions amount to 7[7] _·_ 57[57] . 

Altogether, the count of _det-opt_ s for `BDD-OIA` goes as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0027-02.png)


Table 5: Concepts annotated in `BDD-OIA` . Table taken from [54] 

|**Action Category**|**Concepts**<br>**Count**|
|---|---|
|`move_forward`|`green_light`<br>7805<br>`follow`<br>3489<br>`road_clear`<br>4838|
|`stop`|`red_light`<br>5381<br>`traffic_sign`<br>1539<br>`car`<br>233<br>`person`<br>163<br>`rider`<br>5255<br>`other_obstacle`<br>455|
|`turn_left`|`left_lane`<br>154<br>`left_green_light`<br>885<br>`left_follow`<br>365|
||`no_left_lane`<br>150<br>`left_obstacle`<br>666<br>`letf_solid_line`<br>316|
|`turn_right`|`right_lane`<br>6081<br>`right_green_light`<br>4022<br>`right_follow`<br>2161|
||`no_right_lane`<br>4503<br>`right_obstacle`<br>4514<br>`right_solid_line`<br>3660|



## **C.3 Optimizer and Hyper-parameter Selection** 

The Adam optimizer [86] was employed for all experiments, with exponential decay of the learning rate amounting to _γ_ = 0 _._ 95, exception made for `BDD-OIA` , where we added a weight-decay with parameter _ω_ = 4 _·_ 10 _[−]_[5] and _γ_ was set to 0 _._ 1, to avoid over-fitting. 

The learning rate for all experiments was tuned by searching over the range 10 _[−]_[4] _÷_ 10 _[−]_[2] , with a total of 5 log steps. We found the SL penalty of 2 and 10 for `XOR` and `MNIST-AddMul` , respectively, to work well in our experiments. The strength of the single mitigation strategies ( _η_ ) for each method was chosen accordingly to a grid-search over _η ∈{_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _,_ 2 _,_ 5 _,_ 10 _}_ , varying the learning rate. The best hyper-parameters were selected, in the first step, based on the highest performances in _F_ 1-score for label accuracy on the validation set and, in the second step, on the lowest mitigation loss, accordingly to the value in the validation set. In particular, for SL we chose those runs that yielded the best trade-off between SL minimization and label prediction. For LTN, we found that concept supervision did interfere with the original training objective, for which the best weight for the mitigation strength was found at _γ_ = 10 _[−]_[2] . It was adopted for both LTN+C and LTN+R +C in **Q2** of Section 6. The hyper-parameters for the combined mitigation strategies were selected according to the aforementioned criterion, by only searching the best-combined mitigation strength, while keeping the individual best strengths fixed from the previous grid search. 

For `BDD-OIA` , we selected the learning rate ranging in the interval 10 _[−]_[4] _÷_ 10 _[−]_[2] upon selecting those runs with best _F_ 1-scores on labels in the validation set. The strength of the concept supervision and entropy regularization were varied in between _{_ 0 _._ 1 _,_ 1 _,_ 5 _}_ . 

All best hyper-parameters for our tests are reported in the code in the Supplementary Material. 

27 

## **C.4 Architectures** 

`XOR` : For this data set, we adopted two MLPs, one for the encoder _pθ_ ( **C** _|_ **G** ) and one for the decoder _pψ_ ( **G** _|_ **C** ), both with a hidden size of 3 and ReLU activations. _This architecture is used to empirically validate RSs without forcing disentanglement._ For the disentangled case, we considered a linear layer with weight _ω_ and bias _b_ . 

For SL only, we added an additional MLP, with a hidden size of 3 and tanh activations, implementing the map from the logits of **C** to **Y** . 

`MNIST-AddMul` : We report here the architectures that has been used for `MNIST-Addition` , `MNIST-Multiplication` , and `MNIST-EvenOdd` . For the joint prediction, _i.e._ , without _disentanglement_ , we used the encoder in Table 6. When considering _disentanglement_ , we processed each digit with the encoder in Table 7 and then stacked the two concepts together. For the reconstruction, we used the decoder in Table 8. For SL only, we added an MLP with a hidden size of 50, taking as input the logits of both concepts and processing them to the label. 

Table 6: Double digit encoder for `MNIST-Addition` 

|INPUT SHAPE|LAYER TYPE|PARAMETERS|ACTIVATION|
|---|---|---|---|
|(28_,_56_,_1)|Convolution|depth=32, kernel=4, stride=2, padding=1|ReLU|
|(32_,_14_,_28)|Dropout|_p_= 0_._5||
|(32_,_14_,_28)|Convolution|depth=64, kernel=4, stride=2, padding=1|ReLU|
|(64_,_7_,_14)|Dropout|_p_= 0_._5||
|(64_,_7_,_14)|Convolution|depth=128, kernel=4, stride=2, padding=1|ReLU|
|(128_,_3_,_7)|Flatten|||
|(2688)|Linear|dim=20, bias = True||



Table 7: Single digit Encoder for `MNIST-AddMul` 

|INPUT SHAPE|LAYER TYPE|PARAMETERS|ACTIVATION|
|---|---|---|---|
|(28_,_28_,_1)|Convolution|depth=64, kernel=4, stride=2, padding=1|ReLU|
|(14_,_14_,_64)|Dropout|_p_= 0_._5||
|(14_,_14_,_64)|Convolution|depth=128, kernel=4, stride=2, padding=1|ReLU|
|(7_,_7_,_128)|Dropout|_p_= 0_._5||
|(7_,_7_,_128)|Convolution|depth=256, kernel=4, stride=2, padding=1|ReLU|
|(3_,_3_,_256)|Flatten|||
|(2304)|Linear|dim=10, bias = True||



Table 8: Decoder for `MNIST-Addition` 

|INPUT SHAPE|LAYER TYPE|PARAMETERS|ACTIVATION|
|---|---|---|---|
|(40_,_1)|Unfatten|||
|(128_,_3_,_7)|ConvTranspose2d|depth=64, kernel=(5_,_4), stride=2, padding=1|ReLU|
|(64_,_7_,_14)|Dropout|_p_= 0_._5||
|(128_,_3_,_7)|ConvTranspose2d|depth=32, kernel=(4_,_4), stride=2, padding=1|ReLU|
|(32_,_14_,_28)|Dropout|_p_= 0_._5||
|(128_,_3_,_7)|ConvTranspose2d|depth=1, kernel=(4_,_4), stride=2, padding=1|Sigmoid|



`BDD-OIA` : Images of `BDD-OIA` are preprocessed following [55] with a Faster-RCNN [87] pre-trained on MS-COCO and fine-tuned on BDD-100k [54]. Successively, we adopted the pre-trained convolutional layer on [55] to extract linear features, with dimension 2048. These are the inputs for the NeSy model, which is implemented with a fully-connected NN, see Table 9. 

28 

Table 9: Fully connected layer for `BDD-OIA` 

|INPUT SHAPE|LAYER TYPE|PARAMETERS|ACTIVATION|
|---|---|---|---|
|(2048_,_1)|Linear + BatchNorm1d|dim=1024|Softplus|
|(1024)|Linear + BatchNorm1d|dim=512|Softplus|
|(512)|Linear + BatchNorm1d|dim=256|Softplus|
|(256)|Linear + BatchNorm1d|dim=128|Softplus|
|(128)|Linear + BatchNorm1d|dim=21|Softplus|



29 

## **C.5 Confusion Matrices** 

Following, we report the label and concept-level confusion matrices (CMs) for `XOR` , `MNIST-EvenOdd` , and `MNIST-Multiplication` . For all of them, we report those obtained in runs with maximal _F_ 1- score over the labels. 

## **C.6** `XOR` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0030-03.png)


**----- Start of picture text -----**<br>
NONE DIS<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0030-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0030-05.png)


Figure 3: **CMs for** `XOR` : ( _Left_ ) All NeSy models fail for RSs without any mitigation. ( _Right_ ) Providing DIS avoids all RSs. 

## **C.7** `MNIST-EvenOdd` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0030-08.png)


Figure 4: **NeSy models without mitigation strategies.** ( _Left_ ) DPL picks a RS that uses only 3 digits. ( _Middle_ ) SL optimizes for label predictions but does not always predict a correct configuration for the digits. ( _Right_ ) LTN also picks a RS using only 3 digits. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0030-10.png)


Figure 5: **NeSy models with R.** ( _Left_ ) DPL picks a sub-optimal RS. ( _Middle_ ) SL optimizes for label predictions but through a RS. ( _Right_ ) LTN also picks a sub-optimal RS. For all runs, we found that R interferes with the standard learning objective of DPL and LTN, respectively. 

30 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0031-00.png)


Figure 6: **NeSy models with C.** ( _Left_ ) DPL picks a sub-optimal RS and fails to predict correctly the digits 4 and 9. ( _Middle_ ) SL predicts correctly the 4’s and 9’s but does not avoid the RS. ( _Right_ ) LTN picks a RS that uses only 3 digits. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0031-02.png)


Figure 7: **NeSy models with H.** ( _Left_ ) DPL picks a RS that uses concepts more sparsely. ( _Middle_ ) SL picks a RS, irrespectively of the mitigation. ( _Right_ ) LTN tends to align to the diagonal but fails to predict correctly multiple digits. The performance, nonetheless, is sub-optimal. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0031-04.png)


Figure 8: **NeSy models with R and H.** ( _Left_ ) DPL picks a sub-optimal RS by using only three digits. ( _Middle_ ) SL picks a RS. ( _Right_ ) LTN learns correctly the odd digits but learns a RS for the even ones. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0031-06.png)


Figure 9: **NeSy models with R and C.** ( _Left_ ) DPL picks a sub-optimal RS by using only three digits. ( _Middle_ ) SL improves along the diagonal but fails to correctly encode four digits. ( _Right_ ) LTN picks a RS. 

31 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0032-00.png)


Figure 10: **NeSy models with C and H.** ( _Left_ ) DPL correctly retrieves almost all digits but fails with the digit 7. For this method, we also found runs completely recovering the diagonal. ( _Middle_ ) SL picks a sub-optimal RS. ( _Right_ ) LTN avoids the RS. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0032-02.png)


Figure 11: **NeSy models with R, C and H.** ( _Left_ ) DPLcorrectly identifies the underlying concepts. ( _Middle_ ) SL does not avoid RSs. ( _Right_ ) LTN also identifies the correct digits. 

32 

## **C.8** `MNIST-AddMul` 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0033-01.png)


Figure 12: **CM on the addition task.** All models, DPL ( _Left_ ), SL ( _Middle_ ), and LTN ( _Right_ ) fail for the RS introduced in Example 1, while failing also to predict the digit 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0033-03.png)


Figure 13: **CM on the product task.** ( _Left_ ) DPL picks a RS where it fails to capture the correct semantics of all digits, except 0, ( _Middle_ , _Right_ ) SL, and likewise LTN, acquires a RS where it never predicts correctly the digit 2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts_images/nips_2023_Not_All_Neuro-Symbolic_Concepts_Are_Created_Equal_Analysis_and_Mitigation_of_Reasoning_Shortcuts.pdf-0033-05.png)


Figure 14: **CM on MTL.** Once the predictors are trained for solving addition and multiplication _jointly_ through MTL, they all successfully acquire the concepts with the intended semantics: all confusion matrices are very close to being diagonal. 

33 

