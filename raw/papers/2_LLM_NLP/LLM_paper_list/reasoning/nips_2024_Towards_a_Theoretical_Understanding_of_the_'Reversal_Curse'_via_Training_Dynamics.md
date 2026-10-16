# **Towards a Theoretical Understanding of the ‘Reversal Curse’ via Training Dynamics** 

**Hanlin Zhu** _[∗]_ UC Berkeley `hanlinzhu@berkeley.edu` 

**Baihe Huang** _[∗]_ UC Berkeley `baihe_huang@berkeley.edu` 

**Shaolun Zhang Michael Jordan Jiantao Jiao** UC Berkeley UC Berkeley UC Berkeley `shaolun_zhang@berkeley.edu jordan@cs.berkeley.edu jiantao@berkeley.edu` 

**Yuandong Tian** Meta AI `yuandong@meta.com` 

**Stuart Russell** UC Berkeley `russell@cs.berkeley.edu` 

## **Abstract** 

Auto-regressive large language models (LLMs) show impressive capacities to solve many complex reasoning tasks while struggling with some simple logical reasoning tasks such as inverse search: when trained on “ _A → B_ ” (e.g., _Tom is the parent of John_ ), LLM fails to directly conclude “ _B ← A_ ” (e.g., _John is the child of Tom_ ) during inference even if the two sentences are semantically identical, which is known as the “reversal curse”. In this paper, we theoretically analyze the reversal curse via the training dynamics of (stochastic) gradient descent for two auto-regressive models: (1) a bilinear model that can be viewed as a simplification of a one-layer transformer; (2) one-layer transformers under certain assumptions. Our analysis reveals that for both models, the reversal curse is a consequence of the (effective) model weights _asymmetry_ , i.e., the increase of weights from a token _A_ to token _B_ during training does not necessarily cause the increase of the weights from _B_ to _A_ , which is caused by the training dynamics under certain choice of loss function and the optimization space of model parameters. Moreover, our analysis can be naturally applied to other logical reasoning tasks such as chain-of-thought (COT), which provides a new perspective different from previous work that focuses on expressivity. Finally, we conduct experiments to validate our theory on multi-layer transformers under different settings. Our code is available at `https://github.com/marlo-z/reversal_curse_analysis/` . 

## **1 Introductions** 

Large language models (LLMs) have shown great performance in solving complex reasoning tasks that require multiple reasoning steps through in-context learning (ICL), such as zero-shot learning [1, 2], few-shot learning [3, 4, 5], or via further fine-tuning [6, 7, 8]. However, without the above inferencetime techniques or model fine-tuning (probably combined with data manipulations), an auto-regressive LLM might struggle with simple logical reasoning tasks that require multiple reasoning steps learned during training separately [9], where the reversal curse [10] serves as a well-known example. 

*Equal contributions. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

The reversal curse refers to the phenomenon that an auto-regressive LLM that learns “ _A → B_ ” (e.g., _Tom is the parent of John_ ) during training fails to generalize to the reverse direction “ _B ← A_ ” (e.g., _John is the child of Tom_ ) even if the pair of relationship “ _→_ ” and “ _←_ ” are reverse to each other and the two sentences are semantically identical. Although some previous works propose different methods to mitigate the reversal curse, including reversing the training dataset [11, 12] and training on different objectives such as autoregressive blank infilling [13], these methods might negatively affect the model performance on other tasks since they either alter the dataset or the model architecture. Without dataset manipulation or changing the auto-regressive nature (causal structure) of the model, there are two other candidate solutions to mitigate the reversal curse. 

First, one might constrain the model parameters to satisfy a higher-level regularity for specific relationships. For example, a reversal-type regularity can be viewed as a pair of relationships ( _→_ , _←_ ) and two sets _A, B_ such that a model trained on “ _A → B_ ” will also increase its probability of “ _B ← A_ ” for all _A ∈A, B ∈B_ , which induces a subspace of model parameters that satisfy this regularity. If one can train the model within this subspace, then training on “ _A → B_ ” can, by definition, help to learn “ _B ← A_ ”. However, for a general LLM, it is extremely challenging to find the subspace and manually hard-code the constraint during optimization even for one pair of relationships, not to mention there are numerous relationships. Since it is intractable to manually hard-code the constraints to the model parameter, one can alternatively expect the model to learn the higher-level regularity by training samples under unconstrained optimization. However, this is also hard, according to our analysis, through the popular cross-entropy (CE) loss that aims to maximize the next token prediction probability for the models studied in our paper. 

Second, one can use a different loss function which is “symmetric”, rather than the popular CE loss. However, the “symmetric” loss might drive the model to learn meaningless sentences. For example, when trained on the sentence “John is tall”, a “symmetric” loss function might drive the model to learn “tall is John”, which is not what we expect. To prevent the model from the above undesired behavior, in practice, CE loss is still widely-used. 

Therefore, in this paper, we analyze the reversal curse via training dynamics of the widely-used unconstrained optimization for the CE loss. We summarize our main contributions as follows: 

- We theoretically analyze the reversal curse where training or test sequences have the form “ _A → B_ ” or “ _B ← A_ ” via training dynamics of (stochastic) gradient descent under two auto-regressive models: a bilinear model (Section 3) and one-layer transformers under certain assumptions similar to [14] (Section 4). The analysis of both models reveals that the widely-used unconstrained optimization for CE loss leads to model weights _asymmetry_ , i.e., the increase of (effective) weights (after reparameterization) from the token _A_ to token _B_[1] during training does not necessarily cause the increase of the weights from _B_ to _A_ , which further causes the reversal curse. Although the (effective) weights from _A_ to _B_ and from _B_ to _A_ might be related to some extent due to reparameterization, their correlation is weak and thus show asymmetry as empirically verified in Section 5. 

- The techniques we used to analyze the reversal curse can be applied to other logical reasoning tasks. In particular, we use the above framework to analyze chain-of-thought (COT) [4], and we show that a model trained on “ _A → B_ ” and “ _B → C_ ” separately struggles to directly conclude “ _A ; C_ ” without COT even if it is logically true (Section 4.2). Different from the previous work [15] that theoretically studies COT through the expressivity of transformers, our work provides a new perspective through training dynamics. 

- We also empirically validate our theoretical results on multi-layer transformers (Section 5). 

The _asymmetry_ of auto-regressive model weights caused by widely-used unconstrained optimization for CE loss indicates that auto-regressive LLMs might not automatically deduce certain types of conclusions using separate knowledge learned during training under current popular training paradigms: to make a model predicting token _B_ where the input token is _A_ , the model might need to see _B_ following _A_ in the same sequence during the training set. This also highlights the importance of ICL, data augmentation, or planning for LLMs with the current popular causal transformer-based structures to solve complex reasoning tasks. 

> 1The weights from _A_ to _B_ can be viewed as the logits of token _B_ when the input is _A_ . 

2 

## **1.1 Related works** 

**LLM Reasoning.** The strong performance of LLMs on reasoning tasks [3, 7, 16, 2, 17, 4, 18, 19] has prompted many studies on the reasoning capabilities of LLMs. [20] argues that transformers perform implicit Bayesian inference in ICL. [21] shows that transformers implement a specific type of circuits called “induction heads” that are key to the ICL abilities of LLMs. [22] proves that causal structures are encoded in transformer layers during the training dynamics. [23] identifies a backward chaining mechanism of transformers in deductive reasoning. Apart from in-context reasoning, LLMs still demonstrate limitations in other types of reasoning tasks [24, 25, 26]. 

**Reversal Curse.** [10] identifies the phenomenon of reversal curse. This drawback of LLMs is also demonstrated in [27]. [9] studies a similar phenomenonin which LLMs face difficulty in manipulating already learned knowledge. Several paper studies eliminating the reversal curse by extending causal attention to bidirectional attention [13], training on reversed samples [12], permuting semantic units [11], or introducing reverse logic data [28]. Given all the empirical works, theoretical analysis of the reversal curse phenomenon remains scarce. 

**Expressivity of LLMs.** There is a long line of works [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 21, 45] studying the behavior of LLMs through the expressivity of transformers. It has been shown that transformers can implement simple functions such as sparse linear functions, two-layer neural networks, and decision trees [42], gradient descent [37, 44, 46], automata [47], Turing machines [48], variational inference [49], and bandit algorithms [50]. Different from [15] that study COT via expressivity, we analyze reversal curse and COT via training dynamics. 

**Training dynamics of LLMs.** There are rich literatures in the optimization of attention layers [51, 52, 53, 54, 55, 56, 57]. [58, 59] study the dynamics of a single linear attention layer in in-context linear regression. [60] proves convergence of one-layer transformers in random feature regime. [61] shows the convergence of gradient descent on one-layer transformers in in-context linear regression with orthogonal data. [14] studies the convergence of one-layer transformers in a class of next-token prediction tasks. [62] studies training dynamics of multi-layer transformers. [22] studies gradient descent on a class of two-layer transformers in in-context learning tasks with latent causal structures. Our paper studies the reversal curse via training dynamics under both bilinear settings and one-layer transformers. For one-layer transformers, we use the same framework as [14] without the need for certain technical assumptions such as long input sequences, different learning rates for different parameters (except for Appendix C.3), or weak correlations that are required for [14]. Besides, we focus on the generalization ability of models for logical reasoning tasks while [14] mainly focus on optimization, and we identify the asymmetry and intransitivity properties of model weights, which are the core reasons for the failure of LLM for certain types of logical reasoning. Moreover, our analysis of the bilinear model only requires the embedding to be almost orthonormal, while [14] essentially assumed the embedding vectors to be fixed and one-hot. 

## **2 Preliminaries** 

**Basic notations.** For any integer _N >_ 0, we use [ _N_ ] to denote the set _{_ 1 _,_ 2 _, . . . , N }_ . Let R, N denote the set of real numbers and natural numbers, respectively. For real variables _x_ 1 _, . . . , xn_ , we use poly( _x_ 1 _, . . . , xn_ ) to denote the polynomial of _x_ 1 _, . . . , xn_ . We use _f_ ( _n_ ) ≲ _g_ ( _n_ ) if there exists a constant _C >_ 0 s.t. _f_ ( _n_ ) _≤ Cg_ ( _n_ ) _, ∀n_ ; we say _g_ ( _n_ ) ≳ _f_ ( _n_ ) if _f_ ( _n_ ) ≲ _g_ ( _n_ ). 

We use _**e** i_ to denote one-hot vectors where only the _i_ -th entry of _**e** i_ equals one and all other entries are zero. We use **1** to denote all-one vectors, **0** to denote zero vectors or zero matrices, and _I_ to denote the identity matrix. We will also add subscripts when we want to explicitly show the dimension, such as **0** _d_ , _Id_ for _d_ -dimensional zero vector and _d × d_ identity matrix. We use _⊗_ to denote tensor product of vectors or matrices and use _**x**[⊗]_[2] and _A[⊗]_[2] to denote _**x** ⊗_ _**x**_ and _A ⊗ A_ for vector _**x**_ and matrix _A_ . 

We use _N_ ( _**µ** ,_ Σ) (or adding subscripts such as _Nd_ ( _·, ·_ ) if we want to show dimensions explicitly) to denote the (multi-variate) Gaussian distribution with mean _**µ**_ and covariance Σ. Also, we use ∆( _X_ ) to denote the set of distributions over a set _X_ and use E[ _·_ ] to denote expectation. For any dataset _D_ = _{x_ 1 _, x_ 2 _, . . . , xn}_ where _xi ∈X_ and a function _f_ : _X →_ R, we define the empirical expectation over the dataset as E _D_ [ _f_ ] = _n_[1] � _ni_ =1 _[f]_[(] _[x][i]_[)][.][See additional notations in][ Appendix A][.] 

3 

|Entities|Forward|Backward|Direct|Indirect|Others|
|---|---|---|---|---|---|
|`A`,`B`,`C`,`A`_i_,`B`_i_,`C`_i_|_→_|_←_|_→_|_;_|`R`1,`R`2|



Table 1: Notations for tokens in Section 4. “ _→_ ” and “ _←_ ” denote forward and backward relationships for the reversal curse. “ _→_ ” and “ _;_ ” denote direct and indirect implication for COT. `R` 1 and `R` 2 are relationship tokens in Section 4.3. `A` , `B` , `C` , `A` _i_ , `B` _i_ , `C` _i_ denote tokens representing entities. 

**Auto-regressive models.** Define the vocabulary _V_ = [ _M_ ] for a positive integer _M >_ 0 which is the size of the vocabulary. Let _x_ = ( _x_ 1 _, x_ 2 _, . . . , xT_ ) be a sequence of tokens of length _T_ where each token _xt ∈V, ∀t ∈_ [ _T_ ]. See Table 1 for notations of different tokens used in Section 4. We study auto-regressive models _pθ_ ( _·|x_ ) _∈_ ∆( _V_ ) parameterized by _θ_ that take the sequence _x_ as input and predict the distribution of the next token _xT_ +1 _∈V_ . For both models that we study in this paper, the next token probability is modeled as the softmax applied to the logits _lθ_ ( _·|x_ ) _∈_ R _[M]_ of each token in the vocabulary, i.e., _pθ_ ( _y|x_ ) = � _v_ ex _∈V_ p([exp(] _lθ_ ( _y[l] |[θ] x_[(] )) _[v][|][x]_[))] _[,][∀][y][∈V]_[.][Also, each token] _[ v][∈V]_[has a] corresponding (fixed or learnable) embedding vector _**u** v ∈_ R _[d]_ . 

## **3 Bilinear Models** 

We start analyzing the reversal curse under bilinear models, which can be viewed as simplified one-layer transformers with input length one and decoder layer only. Also, in this section, we assume the embeddings of each token are fixed, so we directly use the embedding vector to represent a token. 

**Datasets.** Assume the vocabulary has size _m_ where each token _v_ 1 _, v_ 2 _, . . . , vm i.i.d.∼Nd_ (0 _d, d_[1] _[I][d]_[)][.] Let _V_ = _{v_ 1 _, . . . , vm}_ and let _X_ = _{x_ 1 _, . . . , xn}_ and _Y_ = _{y_ 1 _, . . . , yn}_ be disjoint random subsets of _V_ . Assume all training and test sequences have a length of two. For any 2 _≤ i ≤ n_ , the training dataset contains both sequence ( _xi, yi_ ) and ( _yi, xi_ ). In addition, the training set contains ( _x_ 1 _, y_ 1) while the test set only contain one example ( _y_ 1 _, x_ 1). During training, the model learns both ( _xi, yi_ ) and ( _yi, xi_ ) for _i ≥_ 2 to conclude that ( _xi, yi_ ) is equivalent to ( _yi, xi_ ). For example, _X_ is a set of names and _Y_ is a set of books. The sequence ( _xi, yi_ ) means “ _xi_ is the author of _yi_ ”, and the sentence ( _yi, xi_ ) means “ _yi_ is written by _xi_ ”. We test whether the model is able to infer an unseen sequence ( _y_ 1 _, x_ 1) given the training data which includes the other direction ( _x_ 1 _, y_ 1). 

**Bilinear model.** We consider a bilinear model parameterized by Θ _∈_ R _[d][×][d]_ of which the input contains single token _x ∈V_ . The logits of the next token _y ∈V_ is defined as _l_ Θ( _y|x_ ) = _x[⊤]_ Θ _y_ which is bilinear in _x_ and _y_ , and thus the next token probability is _p_ Θ( _y|x_ ) = exp( _l_ Θ( _y|x_ )) � _v∈V_[exp(] _[l]_[Θ][(] _[v][|][x]_[))][.][The][training][loss][for][the][bilinear][model][is][the][cross-entropy][loss] _[L]_[(Θ)][=] 1 2 _n−_ 1[(][�] _i[n]_ =1 _[−]_[log] _[ p]_[Θ][(] _[y][i][|][x][i]_[) +][ �] _[n] i_ =2 _[−]_[log] _[ p]_[Θ][(] _[x][i][|][y][i]_[))][ and the test loss (reversal loss) is] _[ L]_ `[rev]`[(Θ) =] _−_ log _p_ Θ( _x_ 1 _|y_ 1). We study the training dynamics of gradient flow _[d] dt_[Θ] _[t]_[=] _[ −∇L]_[(Θ] _[t]_[)][ with the initial-] ization Θ0 that can be either randomly sampled from _N_ ( **0** _[⊗]_[2] _, σ_[2] _I[⊗]_[2] ) or set as a pretrained parameter 1 2 satisfying 2 _m[<][p]_[Θ][0][(] _[y][i][|][x][i]_[)] _[, p]_[Θ][0][(] _[x][i][|][y][i]_[)] _[<] m_[for][all] _[i][∈]_[[] _[n]_[]][.][The][following][theorem][shows][a] separation during training dynamics. 

**Theorem 1** (Separation of training dynamics (informal statement of Theorem 5)) **.** _Fix any δ, ϵ ∈_ (0 _,_ 1) _. For small σ and d ≥_ poly( _n, m,_ 1 _/ϵ,_ log(1 _/δ_ )) _, with probability at least_ 1 _− δ, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0004-08.png)


Theorem 1 shows that the reversal loss is lower bounded by the training loss. Note that for large _d_ and small _ϵ_ close to 0, ( _L_ (Θ _t_ ) _/L_ (Θ0)) _[ϵ]_ is close to 1 and thus _L_ `[rev]` (Θ _t_ ) ≳ _L_ `[rev]` (Θ0) which implies that _p_ Θ( _x_ 1 _|y_ 1) remains small during training. We summarize the above argument in Theorem 2. 

**Theorem 2** (Lower bound of reversal loss (informal statement of Theorem 6) **.** _Fix arbitrary c >_ 0 _and C ≤_ log( _m/_ 2) _. Suppose σ is small and d ≥_ poly( _n, m,_ log(1 _/δ_ ) _,_ log _c,_ 1 _/_ log _C_ ) _. With probability at least_ 1 _− δ, it holds that L_ _`[rev]`_ (Θ _τ_ ) _≥ C, where τ denotes the first time such that L_ (Θ _t_ ) _≤ c._ 

4 

The proofs of Theorems 1 and 2 are deferred to Appendix B. Theorem 2 implies that for large _d_[2] , while the training loss can be trained to be arbitrarily small, the reversal loss remains large. In other words, the model fails to infer an unseen sequence ( _y_ 1 _, x_ 1) given the training data which includes the other direction ( _x_ 1 _, y_ 1). Furthermore, even if the model is fine-tuned on new data from a pre-trained parameter Θ0 that initially grasps the concept of reversal and satisfies 21 _m[<] p_ Θ0( _yi|xi_ ) _, p_ Θ0( _xi|yi_ ) _< m_[2][for new data, it fails to extend this understanding to new, unseen data.] 

A core reason that the reversal curse happens on the above bilinear model is that the parameter matrix Θ _t_ is asymmetric. Consequently, the logits _l_ Θ _t_ ( _y|x_ ) = _x[⊤]_ Θ _ty_ and _l_ Θ _t_ ( _x|y_ ) = _y[⊤]_ Θ _tx_ generally differ. Consider a special case where each _vi_ is a one-hot vector. Then _l_ Θ( _y|x_ ) = _x[⊤]_ Θ _y_ = Θ _ij_ and _l_ Θ( _x|y_ ) = _y[⊤]_ Θ _x_ = Θ _ji_ for _x_ = _**e** i, y_ = _**e** j_ . Training on ( _x, y_ ) can increase Θ _ij_ but not Θ _ji_ , which means the model does not automatically learn the reversal direction ( _y, x_ ) from ( _x, y_ ). In Section 4, we will show that for one-layer transformers, the reversal curse is mainly caused by the same reason, i.e., the asymmetry of the model weights. 

## **4 One-Layer Transformers** 

In Section 3, we analyzed the reversal curse under a bilinear model. In this section, we analyze the reversal curse for one-layer transformers in a similar setting to [14] via training dynamics. We also extend our analysis to chain-of-thought in Section 4.2. 

**Basic notations.** Let _V_ = [ _M_ ] be the vocabulary. For any token _x ∈_ [ _M_ ], we also use the corresponding one-hot vector _**x**_ = _**e** x ∈_ R _[M]_ to represent it. Let _U_ = [ _**u**_ 1 _,_ _**u**_ 2 _, . . . ,_ _**u** M_ ] _[⊤] ∈_ R _[M][×][d]_ be the embedding matrix, where _**u** x ∈_ R _[d]_ is the embedding of token _x ∈_ [ _M_ ]. Note that _U[⊤]_ _**x**_ = _**u** x_ . Consider the _i_ -th training sample in the dataset, _x_ [ _i_ ] = ( _x_ 1[ _i_ ] _, . . . , xT_ [ _i_ ][ _i_ ] _, xT_ [ _i_ ]+1[ _i_ ]), a sequence of tokens of length _T_ [ _i_ ] + 1. Here, _xT_ [ _i_ ]+1[ _i_ ] is the next token (or equivalently the label) to be predicted, _xT_ [ _i_ ][ _i_ ] is the query token, and ( _x_ 1[ _i_ ] _, . . . , xT_ [ _i_ ] _−_ 1[ _i_ ]) are contextual tokens. For token _xt_ [ _i_ ], we also use its one-hot vector _**x** t_ [ _i_ ] = _**e** xt_ [ _i_ ] _∈_ R _[M]_ to represent it. Define the contextual token matrix _X_ [ _i_ ] = [ _**x**_ 1[ _i_ ] _, . . . ,_ _**x** T_ [ _i_ ] _−_ 1[ _i_ ]] _[⊤] ∈_ R[(] _[T]_[ [] _[i]_[]] _[−]_[1)] _[×][M]_ . We omit all _i_ in notations when the context is clear. 

**One-layer transformer.** For a training sample _x_ = ( _x_ 1 _, . . . , xT , xT_ +1), its contextual token matrix _X_ = [ _**x**_ 1 _, . . . ,_ _**x** T −_ 1] _[⊤]_ and thus _XU_ = [ _**u** x_ 1 _, . . . ,_ _**u** xT −_ 1] _[⊤]_ contains the contextual token embeddings. We study one-layer transformers in the same setting as [14]. In particular, for an input˜ token sequence ( _x_ 1 _, . . . , xT_ ), after the one-layer self-attention, we can obtain _**u** T_ = _U[⊤]_ LN( _X[⊤]_ _**b** T_ ) where _btT_ = _T −_ exp(1 _**u**[⊤] xT[W][Q][W][ ⊤] K_ _**[u]**[x] t[/]_ ~~_√_~~ _d_ ) _**[b]**[T]_[=][[] _[b]_[1] _[T][ , . . . , b][T][ −]_[1] _[,T]_[ ]] _[⊤]_[contains][attention][scores] � _t[′]_ =1[exp(] _**[u]**[⊤] xT[W][Q][W][ ⊤] K_ _**[u]**[x] t′[/]_ ~~_√_~~ _d_ )[,] (after softmax) that query token _xT_ attend to each contextual token[3] , LN( _**x**_ ) = _**x** /∥_ _**x** ∥_ 2 is the _ℓ_ 2- normalization operator, and _WQ, WK ∈_ R _[d][×][d][k]_ are trainable query and key matrices respectively. The logit of _x ∈_ [ _M_ ] is then calculated by a decoder layer, i.e., _lθ_ ( _x|x_ 1 _, . . . , xT_ ) = _**u**[⊤] x[W][V]_ _**[u]**_[˜] _[T]_[, where] _[ θ]_ encodes all parameters in the transformer, and _WV ∈_ R _[d][×][d]_ can be viewed as a reparameterization of value and output matrices. Finally, the next token prediction probability is obtained by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0005-06.png)


We use the cross-entropy loss function to train the model over the whole training set _D_ train, i.e., max _U,WK ,WQ,WV J_ ≜ E _D_ train [log _pθ_ ( _xT_ +1 _|x_ 1 _, . . . , xT_ )]. 

**Reparameterization.** Similar to [14], we define _Z_ = _UWQWK[⊤][U][ ⊤][/] √d ∈_ R _[M][×][M]_ and _Y_ = _UWV[⊤][U][ ⊤][∈]_[R] _[M][×][M]_[and are interested in their dynamics after reperameterization.][Then, the attention] score (after softmax) and next token probability become 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0005-09.png)


> 2Empirically, _d_ only needs to be of the order of logarithm of the vocabulary size. See Appendices E.2.1 and E.2.2 for additional results. 

> 3Note that we assume the query token will not attend to itself as in [14]. 

5 

and the objective can be written as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0006-01.png)


Let _ηY_ , _ηZ_ be the learning rate of matrices _Y_ and _Z_ respectively. Then the gradient of _Y_ and _Z_ can be characterized by the following lemma: 

**Lemma 1** (Gradient of _Y_ and _Z_ for 1-layer transformer, Lemma 1 of [14]) **.** _The gradient of Y and Z w.r.t._ (2) _of batch size 1 and learning rate ηY and ηZ can be written as_ 

_Y_ ˙ = _ηY LN_ ( _X[⊤]_ _**b** T_ )( _**x** T_ +1 _−_ _**α**_ ) _[⊤] , Z_ ˙ = _ηZ_ _**x** T_ ( _**x** T_ +1 _−_ _**α**_ ) _[⊤] Y[⊤][P] X[ ⊥][⊤]_ _**b** T X[⊤]_ diag( _**b** T_ ) _X,_ (3) _∥X[⊤]_ _**b** T ∥_ 2 _where P_ _**v**[⊥]_ ≜ _I −_ _**vv**[⊤] /∥_ _**v** ∥_ 2[2] _[projects][any][vector][to][orthogonal][complement][of]_ _**[v]**[,]_ _**[α]**_[=] [ _α_ 1 _, α_ 2 _, . . . , αM_ ] _[⊤] ∈_ R _[M] with_ _**α**_ = exp � _Y[⊤] LN_ ( _X[⊤]_ _**b** T_ )� _/_ **1** _[⊤]_ exp � _Y[⊤] LN_ ( _X[⊤]_ _**b** T_ )� _._ 

_Proof. Y ,_[˙] _Z_[˙] can be obtained by direct calculation. One can refer to the proof of Lemma 1 of [14]. 

## **4.1 Main results for the reversal curse** 

In this section, we analyze the reversal curse where data points are three-token sentences “ `A` _→_ `B` ” or “ `B` _←_ `A` ”. For each sentence, `A` and `B` are two distinct tokens that represent two entities, and “ _→_ ” and “ _←_ ” are two special tokens representing a pair of relationships inverse to each other. 

**Datasets.** Let _N_ train _>_ 0, _N_ test[(1)] _[>]_[ 0][ and] _[ N]_ test[ (2)] _[>]_[ 0][ and denote] _[ N]_ total[=] _[ N]_ train[+] _[ N]_ test[ (1)][+] _[ N]_ test[ (2)][.][Let] `A` _i,_ `B` _i ∈V, ∀i ∈_ [ _N_ total] be 2 _N_ total distinct tokens representing distinct entities. Let _→, ←∈V_ be two additional different tokens that represent two inverse relationships. Specifically, we have `A` _i →_ `B` _i_ and `B` _i ←_ `A` _i_ for all _i ∈_ [ _N_ total]. For notation convenience, we define the following three index sets 

_I_ train = [ _N_ train] _, I_ test[(1)][= [] _[N]_ train[+] _[ N]_ test[ (1)][]] _[\I]_ train _[,] I_ test[(2)][= [] _[N]_ total[]] _[\]_[(] _[I]_ train _[∪I]_ test[(1)][)] _[.]_ The training set _D_ train consists of all `A` _i →_ `B` _i_ and `B` _i ←_ `A` _i_ for _i ∈I_ train. In addition, _D_ train contains `A` _i →_ `B` _i_ for _i ∈I_ test[(1)][and] `[ B]` _i[←]_ `[A]` _i_[for] _[ i][ ∈I]_ test[(2)][.][For convenience, we let] _[ N]_[=] _[ |D]_ train _[|]_[ to be the size of] the training set. The test set _D_ test consists of `B` _i ←_ `A` _i_ for _i ∈I_ test[(1)][and] `[ A]` _i[→]_ `[B]` _i_[for] _[ i][ ∈I]_ test[(2)][.][Under] our construction of the dataset, the LLM will learn the relationship between `A` _i_ and `B` _i_ for _i ∈I_ train in both directions to deduce that _→_ is reverse to _←_ , and learn the relationship between `A` _i_ and `B` _i_ for _i ∈I_ test[(1)] _[∪I]_ test[(2)][in one direction and will be tested for the other.] 

We use _pθ_ ( `A` _i|_ `B` _i ←_ ) and _pθ_ ( `B` _i|_ `A` _i →_ ) to more compactly represent _pθ_ ( _x_ 3 = `A` _i|x_ 1 = `B` _i, x_ 2 = _←_ ) and _pθ_ ( _x_ 3 = `B` _i|x_ 1 = `A` _i, x_ 2 = _→_ ), respectively. Our goal is to prove through the training dynamics of one-layer transformers that the test probability remains negligible during training. In particular, we are interested in _pθ_ ( `A` _i|_ `B` _i ←_ ) _, ∀i ∈I_ test[(1)][and] _[ p] θ_[(] `[B]` _i[|]_ `[A]` _i[→]_[)] _[,][ ∀][i][ ∈I]_ test[(2)][.] 

For convenience, we assume zero-initialization _Y_ (0) = **0** and _Z_ (0) = **0** . This is the same as [14] and is reasonable since empirically, _Y_ and _Z_ are usually initialized as inner products of _d_ -dimensional vectors with i.i.d Gaussian entries, and thus are almost zero (Lemma 8 in Appendix B). The following proposition shows the initial train/test probabilities are uniform over the vocabulary _V_ . 

**Proposition 4.1** (Initial probability under zero initializaion) **.** _Assume the transformer is under zero-initialization θ_ (0) = ( _Y_ (0) _, Z_ (0)) _with Y_ (0) = **0** _and Z_ (0) = **0** _. For any i ∈_ [ _Ntotal_ ] _, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0006-13.png)


The proof is deferred to Appendix C.1.1. Proposition 4.1 shows that initially, the probability of predicting any `B` (or `A` , respectively) given any `A` _→_ (or `B` _←_ , respectively) as input is uniform over the whole vocabulary. When _Y_ (0) and _Z_ (0) are not exactly **0** but close to **0** , the initial prediction will still be close to the uniform distribution, which is similar to Lemma 6. Next we analyze the dynamics of _pθ_ ( _t_ )( `B` _i|_ `A` _i →_ ) and _pθ_ ( _t_ )( `A` _i|_ `B` _i ←_ ). 

**Proposition 4.2** (Next token probability) **.** _For input sequence_ ( _x_ 1 _, x_ 2) _, the next token probability under parameters θ_ ( _t_ ) _is pθ_ ( _t_ )( _x|x_ 1 _, x_ 2) = exp ( _Y_ ( _t_ ) _x_ 1 _,x_ ) _/_[�] _x[′] ∈_ [ _M_ ][exp (] _[Y]_[ (] _[t]_[)] _[x]_ 1 _[,x][′]_[)] _[, where][ Y]_[ (] _[t]_[)] _[i,j] is the entry of the matrix Y_ ( _t_ ) _at row i and column j._ 

6 

The proof is also deferred to Appendix C.1.1. According to Proposition 4.2, the next token probability when the entity in the input is _x_ 1 is determined by the _x_ 1-th row of the matrix _Y_ ( _t_ ). Another nice property indicated by Proposition 4.2 is that we don’t need to keep track of the dynamics of _Z_ ( _t_ ), which could greatly simplify the analysis. The following lemma shows the dynamics of _Y_ ( _t_ ). **Lemma 2** (Dynamics of _Y_ ( _t_ )) **.** _Assume we run SGD with batch size 1_[4] _, and assume M ≫_ 100 _and M_[0] 1 _[.]_[99] _[≪][η][Y][<]_[ 1] _[.][Let][ t]_[ ≳] _[N] η_[ln] _Y[ M] and let Y_ ( _t_ ) _i denote the i-th row of Y_ ( _t_ ) _and Y_ ( _t_ ) _ij denote the_ ( _i, j_ ) _-th entry of Y_ ( _t_ ) _. Then for training sequence_ ( _x_ 1 _, x_ 2 _, x_ 3) _∈Dtrain at time t, we have_ 

_Y_ ( _t_ ) _x_ 1 _,x_ 3 ≳ ln ( _MηY t/N_ ) _, and Y_ ( _t_ ) _x_ 1 _,x_ ≲ _−_ ln ( _MηY t/N_ ) _/M, ∀x ̸_ = _x_ 3 _,_ 

_and for any test sequence_ ( _x_ 1 _, x_ 2 _, x_ 3) _∈Dtest, we have Y_ ( _t_ ) _x_ 1 _,x_ = 0 _, ∀x ∈_ [ _M_ ] _._ 

The proof of Lemma 2 is presented in Appendix C.1.2. Lemma 2 implies the asymmetry of the model weights _Y_ ( _t_ ): for two tokens _x_ 1 _, x_ 3, when _x_ 1 appears as a contextual token and _x_ 3 serves as the next token in the same training sequence, the model weights _Y_ ( _t_ ) _x_ 1 _,x_ 3 gets increased during training while _Y_ ( _t_ ) _x_ 3 _,x_ 1 will not get increased. Combining Proposition 4.2, we can obtain our main theorem for the reversal curse. 

**Theorem 3** (Reversal curse) **.** _Assume we run SGD with batch size 1, and assume M ≫_ 100 _and M_[0] 1 _[.]_[99] _[≪][η][Y][<]_[ 1] _[.][Let][ t]_[ ≳] _[N] η_[ln] _Y[ M] denote the time step which also satisfies_ ln _t_ ≳ ln( _NM/ηY_ ) _. For training sequence_ ( _x_ 1 _, x_ 2 _, x_ 3) _∈Dtrain at time t, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0007-05.png)


_for some constant c >_ 0 _, and for any test sequence_ ( _x_ 1 _, x_ 2 _, x_ 3) _∈Dtest that is not included in the training set Dtrain, we have pθ_ ( _t_ )( _x_ 3 _|x_ 1 _, x_ 2) _≤_ 1 _/M ._ 

Theorem 3 shows that although the direction presented in the training set can be learned nearly perfectly, the model’s next token prediction of the reverse direction is almost a random guess. The proof is deferred to Appendix C.1.3. We also empirically validate the above results for multi-layer transformers in Section 5. 

## **4.2 Chain-of-thought** 

In this section, we extend our analysis in Section 4.1 to study other logical relationships. In particular, we study chain-of-thought (COT) [4] and show its importance via training dynamics. COT encourages LLMs to output a series of intermediate reasoning steps to increase their performance. Consider the simplest example, where the model learns two facts that `A` _→_ `B` and `B` _→_ `C` , and we want to test whether the model is able to directly conclude that `A` _;_ `C` . COT indicates that if an LLM is only trained on `A` _→_ `B` and `B` _→_ `C` , it would be easier for the model to deduce `A` _;_ `C` during the inference time if the model can first output the intermediate steps `A` _→_ `B` and `B` _→_ `C` , instead of directly predicting the next token `C` given the input “ `A` _;_ ”. The failure of directly deducing `A` _;_ `C` is also empirically observed by [9]. 

Theoretically, [15] shows the importance of COT for some complex reasoning tasks through the lens of the expressivity of transformers. In this section, we show the importance of COT through a different angle, i.e., training dynamics. We show that for the above simplest two-step reasoning, without COT, the model is not able to directly predict `C` given the input “ `A` _;_ ” even if it learns `A` _→_ `B` and `B` _→_ `C` . 

**Theorem 4** (Importance of chain-of-thought, informal statement of Theorem 7) **.** _Under certain assumptions as stated in Theorem 7, for any_ _`A` i,_ _`B` i,_ _`C` i s.t._ _`A` i →_ _`B` i and_ _`B` i →_ _`C` i are in the training set but_ _`A` i ;_ _`C` i is not, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0007-12.png)


We defer the details of the dataset construction and proof to Appendix C.2. Theorem 4 shows that although the LLM learns `A` _i →_ `B` _i_ and `B` _i →_ `C` _i_ nearly perfectly, it cannot directly deduce `A` _i ;_ `C` _i_ . Analogous to the asymmetry of causal transformer weights as we discussed in Section 4.1, our analysis of COT reveals another property, i.e., intransitivity: training the weights associated with `A` to `B` and `B` to `C` does not necessarily increase the weights associated with `A` to `C` . 

> 4The lemma holds even if the batch size is larger than 1 and the analysis is essentially the same. 

7 

We also emphasize that the model fails to directly deduce `A` _i ;_ `C` _i_ when the two intermediate steps `A` _i →_ `B` _i_ and `B` _i →_ `C` _i_ are trained _separately_ . If the two steps are concatenated into a single training sequence, it is possible that the model learns `A` _i ;_ `C` _i_ directly [19]. 

## **4.3 Roles of the attention score matrix** 

During the analysis of Sections 4.1 and 4.2, we show that the reversal curse and the importance of COT are largely due to the asymmetry and intransitivity of causal transformer weights (in our case, the weight matrix _Y_ ( _t_ )). However, it seems that the dynamics of the attention score matrix _Z_ ( _t_ ) do not impact the model performance. Below, we briefly discuss the role of the attention score matrix _Z_ ( _t_ ). 

In (1), the attention score is used to calculate the weights _btT_ , where a contextual token _xt_ with a larger attention score attended by the query token _xT_ has a larger weight. Note that we use the same formulation as the previous work [14] where the query token will not attend to itself. Therefore, for a three-token training sequence, the weights _b_ 12 is always one since there is only one contextual token _x_ 1, no matter whether the value of the attention score is high or low. 

However, consider a slightly different setting, where the relationship is represented by two tokens. In that case, _x_ 1 = `A` _i, x_ 2 = `R` 1 _, x_ 3 = `R` 2 _, x_ 4 = `B` _i_ , and there are two contextual tokens `A` _i_ and `R` 1. The role of the attention score is then to select the important token, i.e., `A` _i_ , by putting more weights on it. Theorem 2 of [14] showed that under certain assumptions, the query token `R` 2 will attend more to “distinct tokens” `A` _i_ and less to the “common token” `R` 1. Therefore, the query token `R` 2 will eventually put all weights to `A` _i_ , and the remaining analysis remains the same as in Sections 4.1 and 4.2. See Appendix C.3 for a more rigorous analysis. 

## **5 Experiments** 

In this section, we conduct experiments to further validate our theoretical results in Section 4 on multi-layer transformers. We show experimental results of the reversal curse in this section and COT in Appendix D. Note that in Sections 3 and 4, we theoretically proved the reversal curse for both the bilinear model and one-layer transformer under certain assumptions. Now, we empirically show that the reversal curse still happens even for multi-layer transformers. In Appendix E.2.3, we also provide empirical results that the reversal curse does not happen in ICL settings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0008-07.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs<br>Train<br>70 Val<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>0 500 1000 1500 2000 2500 3000<br>Number of Epochs<br>Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 1: Experiment results of reversal curse under default configuration (see Table 3). The curves represent the (average) negative log probability of the model predicting the next token to be `B` _i_ when the input is “ `A` _i →_ ”, or to be `A` _i_ when the input is “ `B` _i ←_ ”. While the sentences in the training set can be learned nearly perfectly (as shown by the training curve where the next token probability converges to one), the model is not able to predict the correct next token in the validation set better than a uniformly random guess. Both curves are averaged over 10 random seeds. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Forward direction logits (Train) Reverse direction logits (Train)<br>80<br>60<br>40<br>20<br>Seen direction logits (Val) Unseen direction logits (Val)<br>0<br>20<br>40<br>60<br>**----- End of picture text -----**<br>


Figure 2: Visualization of the weights (logits) of the model with default configurations trained after 3000 epochs for the reversal curse experiment. For the top-left matrix, the _i_ -th row corresponds to an entity token `A` _i_ for a training pair, and the _i_ -th column corresponds to an entity token `B` _i_ for a training pair. The ( _i, j_ )-th entry represents the model weights from the token `A` _i_ to `B` _j_ , i.e., the logits of `B` _j_ when the input sequence consists of only `A` _i_ . Similarly, for the bottom-left matrix, the row corresponds to the input entity tokens of the seen direction (the direction included in the training set) of validation pairs, and the column corresponds to output entity tokens. The two matrices on the right are obtained by swapping row tokens and column tokens of their corresponding left matrices. Note that the diagonals of the bottom-right matrix are all close to zero, while the diagonals of other matrices all have large values. This implies that if a pair of tokens ( `A` _,_ `B` ) only appear in the training set in one direction, then the model weights associated with the other direction will hardly get trained. 

**Dataset construction.** Below, we describe how we generate our synthetic dataset for experiments on the reversal curse. We choose the vocabulary _V_ = _{_ 0 _,_ 1 _, . . . , N }_ for a specified _N >_ 0. We randomly sample two disjoint sets of entities _A, B ⊂V_ with _|A|_ = _|B|_ = _|V|/_ 4, and reserve two additional tokens for relationships _→_ and _←_ , respectively.[5] Next, we specify a bijection from _A_ to _B_ uniformly at random. For each `A` _i ∈A_ and its corresponding `B` _i ∈B_ , we can obtain a pair of sequence ( `A` _i →_ `B` _i_ , `B` _i ←_ `A` _i_ ). We split the set of all pairs into training pairs and validation pairs. For each training pair, both sequences will be included in the training set, while for the validation pair, we randomly select one sequence for the training set and the other for the validation set. Therefore, the model will learn both directions for the training pairs and only one direction for each validation pair while being tested in the unseen direction. 

**Model architectures.** We train multi-layer transformers based on GPT-2 architecture [63]. Figure 1 shows the results where the model has 24 layers, 12 attention heads per layer, uses absolute positional encoding, and we choose the vocabulary size of 800. The training set size is 340, and the validation set size is 60 (resulting from 140 training pairs and 60 validation pairs). We also conducted experiments with various model configurations and vocabulary sizes in Appendix E.2. Besides, all hyperparameters and different model configurations are presented in Appendix E.1. 

**Results.** Figure 1 shows that during the training, the next token probability for training data increases a lot while the next token probability for validation data remains unchanged or gets even smaller. This is consistent with our theoretical results of Theorem 3. 

According to our theoretical analysis, the reversal curse happens due to the asymmetry of model (reparameterized) weights (i.e., logits of a token given another token as input), and we also empirically validate the asymmetry for multi-layer transformers. Figure 2 shows the model weights from a token _x_ 1 to _x_ 3 is trained large for a training sequence ( _x_ 1 _, x_ 2 _, x_ 3) as represented by the diagonals of the 

> 5By default, each entity consists of one token. See multi-token experiments in Appendices E.2 and E.3 

9 

first three matrices, while the weights from a token _x_ 1 to _x_ 3 remains nearly zero for a validation sequence ( _x_ 1 _, x_ 2 _, x_ 3) as represented by the diagonals of the last matrix, which is consistent with Lemma 2. This implies that if a pair of tokens ( `A` _,_ `B` ) only appear in the training set in one direction, then the model weights associated with the other direction will hardly get trained. 

## **6 Conclusions** 

In this paper, we study the reversal curse theoretically via training dynamics of (1) a bilinear model, which is a simplification of the one-layer transformer; (2) one-layer transformers under certain technical assumptions similar to [14]. Our theoretical results suggest that a core reason the reversal curse happens in auto-regressive LLMs is the asymmetry of the model weights, and we apply our technique to prove the necessity of COT for one-layer transformers, which is mainly due to the intransitivity of model weights. The asymmetry and intransitivity of model weights caused by unconstrained optimization of CE loss indicate that an auto-regressive LLM might mainly focus on learning text sequences during training _separately_ instead of automatically deducing indirect conclusions under the current popular training paradigms. This highlights the importance of ICL, data augmentation, or planning for current auto-regressive LLMs to solve complex reasoning tasks. 

As for future directions, it would be interesting and important to study: (1) What is a unified way to characterize and study the reversal curse, COT, and other similar logical reasoning tasks? (2) Our paper mainly focuses on three-token sequences, where each entity or relationship is represented by a single token. While we empirically explored the setting where each entity might consist of multiple tokens and distinct entities might share a few tokens, it would be interesting to analyze the multiple-token setting theoretically. (3) We theoretically analyzed the bilinear model and one-layer transformer, and it would be an important future direction to extend the analysis to multi-layer transformers. 

## **Acknowledgements** 

This work was partially supported by a gift from Open Philanthropy to the Center for HumanCompatible AI (CHAI) at UC Berkeley and by NSF Grants IIS-1901252 and CCF-2211209. The work was done when HZ was a visiting researcher at Meta. 

## **References** 

- [1] Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the few-shot paradigm. In _Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems_ , pages 1–7, 2021. 

- [2] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877–1901, 2020. 

- [4] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [5] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- [6] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

10 

- [8] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. _Advances in Neural Information Processing Systems_ , 35:15476–15488, 2022. 

- [9] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.2, knowledge manipulation. _arXiv preprint arXiv:2309.14402_ , 2023. 

- [10] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". _arXiv preprint arXiv:2309.12288_ , 2023. 

- [11] Qingyan Guo, Rui Wang, Junliang Guo, Xu Tan, Jiang Bian, and Yujiu Yang. Mitigating reversal curse via semantic-aware permutation training. _arXiv preprint arXiv:2403.00758_ , 2024. 

- [12] Olga Golovneva, Zeyuan Allen-Zhu, Jason Weston, and Sainbayar Sukhbaatar. Reverse training to nurse the reversal curse. _arXiv preprint arXiv:2403.13799_ , 2024. 

- [13] Ang Lv, Kaiyi Zhang, Shufang Xie, Quan Tu, Yuhan Chen, Ji-Rong Wen, and Rui Yan. Are we falling in a middle-intelligence trap? an analysis and mitigation of the reversal curse. _arXiv preprint arXiv:2311.07468_ , 2023. 

- [14] Yuandong Tian, Yiping Wang, Beidi Chen, and Simon Du. Scan and snap: Understanding training dynamics and token composition in 1-layer transformer, 2023. 

- [15] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [16] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In _International Conference on Machine Learning_ , pages 9118–9147. PMLR, 2022. 

- [17] Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, and Yejin Choi. Maieutic prompting: Logically consistent reasoning with recursive explanations. _arXiv preprint arXiv:2205.11822_ , 2022. 

- [18] Simon Jerome Han, Keith J Ransom, Andrew Perfors, and Charles Kemp. Inductive reasoning in humans and large language models. _Cognitive Systems Research_ , 83:101155, 2024. 

- [19] Xinyi Wang, Alfonso Amayuelas, Kexun Zhang, Liangming Pan, Wenhu Chen, and William Yang Wang. Understanding the reasoning ability of language models from the perspective of reasoning paths aggregation. _arXiv preprint arXiv:2402.03268_ , 2024. 

- [20] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_ , 2021. 

- [21] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. _arXiv preprint arXiv:2209.11895_ , 2022. 

- [22] Eshaan Nichani, Alex Damian, and Jason D Lee. How transformers learn causal structure with gradient descent. _arXiv preprint arXiv:2402.14735_ , 2024. 

- [23] Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Christian Bartelt. A mechanistic analysis of a transformer trained on a symbolic multi-step reasoning task. _arXiv preprint arXiv:2402.11917_ , 2024. 

- [24] Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. Large language models still can’t plan (a benchmark for llms on planning and reasoning about change). _arXiv preprint arXiv:2206.10498_ , 2022. 

- [25] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. _ACM Transactions on Intelligent Systems and Technology_ , 2023. 

- [26] Dylan Zhang, Curt Tigges, Zory Zhang, Stella Biderman, Maxim Raginsky, and Talia Ringer. Transformer-based models are not yet perfect at learning to emulate structural recursion. _arXiv preprint arXiv:2401.12947_ , 2024. 

- [27] Chengwen Qi, Bowen Li, Binyuan Hui, Bailin Wang, Jinyang Li, Jinwang Wu, and Yuanjun Laili. An investigation of llms’ inefficacy in understanding converse relations. _arXiv preprint arXiv:2310.05163_ , 2023. 

11 

- [28] Ruilin Luo, Tianle Gu, Haoling Li, Junzhe Li, Zicheng Lin, Jiayi Li, and Yujiu Yang. Chain of history: Learning and forecasting with llms for temporal knowledge graph completion. _arXiv preprint arXiv:2401.06072_ , 2024. 

- [29] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? _arXiv preprint arXiv:1912.10077_ , 2019. 

- [30] Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to recognize formal languages. _arXiv preprint arXiv:2009.11264_ , 2020. 

- [31] Satwik Bhattamishra, Arkil Patel, and Navin Goyal. On the computational power of transformers and its implications in sequence modeling. _arXiv preprint arXiv:2006.09286_ , 2020. 

- [32] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. _arXiv preprint arXiv:1807.03819_ , 2018. 

- [33] Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing-complete. _Journal of Machine Learning Research_ , 22(75):1–35, 2021. 

- [34] Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in self-attention mechanisms. In _International Conference on Machine Learning_ , pages 5793–5831. PMLR, 2022. 

- [35] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 1:1, 2021. 

- [36] Valerii Likhosherstov, Krzysztof Choromanski, and Adrian Weller. On the expressive power of self-attention matrices. _arXiv preprint arXiv:2106.03764_ , 2021. 

- [37] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. _arXiv preprint arXiv:2211.15661_ , 2022. 

- [38] Haoyu Zhao, Abhishek Panigrahi, Rong Ge, and Sanjeev Arora. Do transformers parse while predicting the masked word? _arXiv preprint arXiv:2303.08117_ , 2023. 

- [39] Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. Self-attention networks can process bounded hierarchical languages. _arXiv preprint arXiv:2105.11115_ , 2021. 

- [40] Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. Exploring length generalization in large language models. _arXiv preprint arXiv:2207.04901_ , 2022. 

- [41] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: Sgd learns parities near the computational limit. _Advances in Neural Information Processing Systems_ , 35:21750–21764, 2022. 

- [42] Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. _Advances in Neural Information Processing Systems_ , 35:30583–30598, 2022. 

- [43] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. _arXiv preprint arXiv:2212.07677_ , 2022. 

- [44] Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. _arXiv preprint arXiv:2306.04637_ , 2023. 

- [45] Shuai Li, Zhao Song, Yu Xia, Tong Yu, and Tianyi Zhou. The closeness of in-context learning and weight shifting for softmax regression. _arXiv preprint arXiv:2304.13276_ , 2023. 

- [46] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International Conference on Machine Learning_ , pages 35151–35174. PMLR, 2023. 

- [47] Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. _arXiv preprint arXiv:2210.10749_ , 2022. 

12 

- [48] Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers. _Advances in Neural Information Processing Systems_ , 35:12071–12083, 2022. 

- [49] Song Mei and Yuchen Wu. Deep networks as denoising algorithms: Sample-efficient learning of diffusion models in high-dimensional graphical models. _arXiv preprint arXiv:2309.11420_ , 2023. 

- [50] Licong Lin, Yu Bai, and Song Mei. Transformers as decision makers: Provable in-context reinforcement learning via supervised pretraining. _arXiv preprint arXiv:2310.08566_ , 2023. 

- [51] Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. In _International Conference on Learning Representations_ , 2020. 

- [52] Jiri Hron, Yasaman Bahri, Jascha Sohl-Dickstein, and Roman Novak. Infinite attention: Nngp and ntk for deep attention networks. In _International Conference on Machine Learning_ , pages 4376–4386. PMLR, 2020. 

- [53] Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. _arXiv preprint arXiv:2203.03466_ , 2022. 

- [54] Enric Boix-Adsera, Etai Littwin, Emmanuel Abbe, Samy Bengio, and Joshua Susskind. Transformers learn through gradual rank increase. _arXiv preprint arXiv:2306.07042_ , 2023. 

- [55] Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer: A memory viewpoint. _arXiv preprint arXiv:2306.00802_ , 2023. 

- [56] Samy Jelassi, Michael Sander, and Yuanzhi Li. Vision transformers provably learn spatial structure. _Advances in Neural Information Processing Systems_ , 35:37822–37836, 2022. 

- [57] Charlie Snell, Ruiqi Zhong, Dan Klein, and Jacob Steinhardt. Approximating how single head attention learns. _arXiv preprint arXiv:2103.07601_ , 2021. 

- [58] Arvind Mahankali, Tatsunori B Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention. _arXiv preprint arXiv:2307.03576_ , 2023. 

- [59] Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. _arXiv preprint arXiv:2306.09927_ , 2023. 

- [60] Hengyu Fu, Tianyu Guo, Yu Bai, and Song Mei. What can a single attention layer learn? a study through the random features lens. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [61] Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. _arXiv preprint arXiv:2310.05249_ , 2023. 

- [62] Yuandong Tian, Yiping Wang, Zhenyu Zhang, Beidi Chen, and Simon Du. Joma: Demystifying multilayer transformers via joint dynamics of mlp and attention. _arXiv preprint arXiv:2310.00535_ , 2023. 

- [63] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-theart natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 38–45, Online, October 2020. Association for Computational Linguistics. URL `https://www.aclweb.org/anthology/ 2020.emnlp-demos.6` . 

- [64] Beatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. _Annals of Statistics_ , pages 1302–1338, 2000. 

- [65] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_ , 568:127063, 2024. 

13 

## **A Additional Notations** 

Let _δij_ = 1 for _i_ = _j_ and _δij_ = 0 for _i_ = _j_ . For a squared matrix _A ∈_ R _[d][×][d]_ , its trace is Tr( _A_ ) =[�] _[d] i_ =1 _[A][ii]_[.][For][two][matrices] _[A, B][∈]_[R] _[m][×][n]_[of][the][same][shape,][their][inner][product][is] defined as _⟨A, B⟩_ = Tr( _AB[⊤]_ ). For any matrix _A ∈_ R _[m][×][n]_ , its (Frobenius) norm is defined as _∥A∥_ = ~~�~~ _⟨A, A⟩_ . For any vector _**x**_ = ( _x_ 1 _, . . . , xd_ ) _[⊤] ∈_ R _[d]_ or a matrix _A ∈_ R _[m][×][n]_ , we define _n_ the zero-norm as _∥_ _**x** ∥_ 0=[�] _[d] i_ =1[1] _[{][x][i]_[=][0] _[}]_[ or] _[ ∥][A][∥]_[0][=][�] _[m] i_ =1 � _j_ =1[1] _[{][A][ij]_[=][0] _[}]_[ where][ 1] _[{·}]_[ is the] indicator function. 

## **B Missing Proofs of Section 3** 

**Theorem 5** (Separation of training dynamics, formal statement of Theorem 1) **.** _Fix arbitrary δ, ϵ ∈_ (0 _,_ 1) _. Let v_ 1 _, . . . , vm be independently sampled from Nd_ ( **0** _d, d_[1] _[I][d]_[)] _[.][Let][ x]_[1] _[, . . . , x][n][and][ y]_[1] _[, . . . , y][n] be sampled uniformly at random from {v_ 1 _, . . . , vm} without replacement. Define_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0014-04.png)


_Consider the gradient flow_ Θ _t_ : _t ≥_ 0 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0014-06.png)


_where_ Θ0 _∼N_ ( **0** _[⊗]_[2] _, σ_[2] _· I[⊗]_[2] ) _or_ Θ0 _satisfies_ 21 _m[<][p]_[Θ][0][(] _[y][i][|][x][i]_[)] _[, p]_[Θ][0][(] _[x][i][|][y][i]_[)] _[<] m_ 2 _[for all][ i][∈]_[[] _[n]_[]] _[.]_ 1 _Suppose σ ≤_ 100 ln(64 _m_[2] _/δ_ ) _[and] d ≥_[10][6] _[n]_[4] _[m]_[2][ lo][g][4][(][2] _[m]_[)][ lo][g(][64] _[m]_[2] _[n]_[2] _[/][δ]_[)] _. ϵ_[2] 

_With probability at least_ 1 _− δ, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0014-09.png)


_Proof._ Let _v_ = � 400 _n_[2] _m_[2] log( _d_ 64 _m_[2] _n_[2] _/δ_ ) . By Lemma 3 and Lemma 4, with probability at least 1 _− δ_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0014-11.png)


**Theorem 6** (Lower bound of reversal loss, formal statement of Theorem 2) **.** _Fix arbitrary c >_ 0 _and_ 1 _C ≤_ log( _m/_ 2) _. Under the setting of Theorem 5, suppose σ ≤_ 100 ln(64 _m_[2] _/δ_ ) _[and]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0014-13.png)


_With probability at least_ 1 _− δ,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0014-15.png)


_where τ denotes the first time such that L_ (Θ _t_ ) _≤ c._ 

14 

_Proof._ By continuity, _L_ (Θ _τ_ ) = _c_ . By Theorem 5, when 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-01.png)


with probability at least 1 _− δ_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-03.png)


Under this event, applying Lemma 6, we can obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-05.png)


log log( _Cm/_ 2) To ensure that the right hand side is _C_ , we set _ϵ_ = log log(2 _c m_ )[.][One may check that the definition of] _d_ satisfies Eq. (4). It follows that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-07.png)


## **B.1 Training dynamics** 

**Lemma 3** (Dynamics of the forward loss) **.** _Let x_ 1 _, . . . , xn, y_ 1 _, . . . , yn, L_ (Θ) _, and_ Θ _t_ ( _t ≥_ 0) _be defined as in Theorem 5. When σ ≤_ 100 ln(161 _n_[2] _/δ_ ) _[and][d][≥]_[1600] _[n]_[3] _[m]_[2][ log(8] _[m]_[2] _[n]_[2] _[/δ]_[)] _[,][with] probability at least_ 1 _− δ, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-10.png)


_Furthermore,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-12.png)


> _Proof._ For convenience, we assume _x_ 1 _, . . . , xn_ = _v_ 1 _, . . . , vn_ and _y_ 1 _, . . . , yn_ = _vn_ +1 _, . . . v_ 2 _n_ WLOG. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0015-14.png)


15 

Similarly, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0016-01.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0016-03.png)


Applying Lemma 7, with probability at least 1 _− δ/_ 2, for any _t ≥_ 0 we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0016-05.png)


and 

Furthermore, Lemma 6 implies that with probability at least 1 _− δ/_ 2 

The following arguments are based on the event that the above inequalities hold. 

16 

We first show that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0017-01.png)


We prove this by contradiction. Let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0017-03.png)


By Eq. (7), it is obvious that _τ >_ 0. Assume without loss of generality that _p_ Θ _τ_ ( _y_ 1 _|x_ 1) _≤_ 21 _m_[.] It follows that there exists _δ >_ 0 such that _p_ Θ _t_ ( _y_ 1 _|x_ 1) is a decreasing function in ( _τ − δ, τ_ ) and _p_ Θ _t_ ( _y_ 1 _|x_ 1) = min _{p_ Θ _t_ ( _xi|yi_ ) _, p_ Θ _t_ ( _yi|xi_ ) : _i ∈_ [ _n_ ] _}_ for any _t ∈_ ( _τ − δ, τ_ ). It follows that for _t ∈_ ( _τ − δ, τ_ ), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0017-05.png)


where the first inequality is from Eq. (6); the second inequality is due to[�] _j_ = _i_ + _n[|][α] i,j_[(] _[t]_[)] _[|]_[=] _[ α] i,i_[(] _[t]_[)] + _n_[=] 1 _− p_ Θ _t_ ( _yi|xi_ ) _,_[�] _j_ = _i[|][β] i_[(] + _[t]_[)] _n,j[|]_[=] _[β] i_[(] + _[t]_[)] _n,i_[=][1] _[ −][p]_[Θ] _t_[(] _[x][i][|][y][i]_[)][for][all] _[i][∈]_[[] _[n]_[]][;][the][third][inequality][is] because _p_ Θ _t_ ( _y_ 1 _|x_ 1) = min _{p_ Θ _t_ ( _xi|yi_ ) _, p_ Θ _t_ ( _yi|xi_ ) : _i ∈_ [ _n_ ] _}_ . However, _p_ Θ _t_ ( _y_ 1 _|x_ 1) is a decreasing function in ( _τ − δ, τ_ ), a contradiction. Therefore, we conclude that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0017-07.png)


Now we show 

By Eq. (5), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0017-10.png)


where the second inequality is due to[�] _j_ = _i_ + _n[|][α] i,j_[(] _[t]_[)] _[|]_[=] _[α] i,i_[(] _[t]_[)] + _n_[=][1] _[ −][p]_[Θ] _t_[(] _[y][i][|][x][i]_[)] _[,]_[ �] _j_ = _i[|][β] i_[(] + _[t]_[)] _n,j[|]_[=] _βi_[(] + _[t]_[)] _n,i_[=][1] _[ −][p]_[Θ] _t_[(] _[x][i][|][y][i]_[)][for][all] _[i][∈]_[[] _[n]_[]][;][the][third][inequality][applies][Cauchy-Schwarz][inequality;] 

17 

1 the last inequality uses the fact that _p_ Θ _t_ ( _xi|yi_ ) _, p_ Θ _t_ ( _yi|xi_ ) _>_ 2 _m_[for][any] _[t][≥]_[0] _[, i][∈]_[[] _[n]_[]][and][the] inequality 1 _− x ≥_ 2 log(1log _/ x_ (2 _m_ ))[for] _[ x][ ∈]_[(] 2[1] _m[,]_[ 1)][.] By Lemma 10, we conclude that _∀t ≥_ 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0018-01.png)


which completes the proof. 

**Lemma 4** (Dynamics of the reversal loss) **.** _Let x_ 1 _, . . . , xn, y_ 1 _, . . . , yn, L_ _`[rev]`_ (Θ) _, and_ Θ _t_ ( _t ≥_ 0) _be defined as in Theorem 5. When σ ≤_ 100 ln(161 _n_[2] _/δ_ ) _[and][ d][≥]_[400] _[n]_[2] _[m]_[2][ log(8] _[m]_[2] _[n]_[2] _[/δ]_[)] _[/ϵ]_[2] _[,][with] probability at least_ 1 _− δ,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0018-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0018-05.png)


18 

Furthermore, Lemma 6 implies that with probability at least 1 _− δ/_ 2, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-01.png)


The following arguments are based on the event that the above inequalities hold. 

By Eq. (8), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-04.png)


Notice that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-06.png)


where the first inequality uses[�] _j_ = _i_ + _n[|][α] i,j_[(] _[t]_[)] _[|]_[=] _[ α] i,i_[(] _[t]_[)] + _n_[= 1] _[ −][p]_[Θ] _t_[(] _[y][i][|][x][i]_[)] _[,]_[ �] _j_ = _i[|][β] i_[(] + _[t]_[)] _n,j[|]_[=] _[ β] i_[(] + _[t]_[)] _n,i_[=] 1 _− p_ Θ _t_ ( _xi|yi_ ) for all _i ∈_ [ _n_ ]; the third inequality uses the fact that 1 _− x ≤−_ log _x_ ; the last inequality applies Lemma 3. 

Similarly, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-09.png)


Combining, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-11.png)


By Lemma 10, we conclude that _∀t ≥_ 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-13.png)


This completes the proof. 

**Lemma 5** (Gradient of the loss function) **.** _Define_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0019-16.png)


_Then we have_ 

19 

_Proof._ We have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-01.png)


The statements follow immediately. 

## **B.2 Initialization** 

_i.i.d._ **Lemma 6** (Initial distributions are all close to uniform) **.** _Fix any δ ∈_ (0 _,_ 1) _. Let x_ 1 _, x_ 2 _, . . . , xn ∼ i.i.d. Nd_ ( **0** _d, d_[1] _[I][d]_[)] _[.][Let]_[ Θ] _[∈]_[R] _[d][×][d][, where each]_[ Θ] _[ij] ∼ N_ (0 _, σ_[2] ) _independent of x_ 1 _, . . . , xn. For any i, j ∈_ [ _n_ ] _, define_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-06.png)


_Proof._ Let _v_ = 0 _._ 2. By Lemma 8, with probability at least 1 _− δ_ , we have 

_|⟨xi, xj⟩− δij|≤ v, ∀i, j ∈_ [ _n_ ] _._ 

Conditioned on the above high-probability event, we can further obtain that for any _j ∈_ [ _n_ ] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-10.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-12.png)


It follows that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-14.png)


which completes the proof. 

## **B.3 Subspace embedding** 

**Lemma 7** ( _ℓ_ 1-subspace embedding of Gaussian second-order tensors) **.** _Let z_ 1 _, . . . , zn be independently sampled from Nd_ ( _**0** d, d_[1] _[I][d]_[)] _[.][Let][ I]_[1] _[,][ I]_[2] _[⊆]_[[] _[n]_[]] _[ ×]_[ [] _[n]_[]] _[ be two index sets.][Let][ I]_[0][=] _[I]_[1] _[ ∩I]_[2] _[.][If] d ≥_ 64 log(2 _n_[2] _/δ_ ) _/ϵ_[2] _, then with probability at least_ 1 _− δ,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0020-18.png)


20 

_holds for any αi,j, βi,j. Furthermore, if d ≥_ 64 _k_[2] log(2 _n_[2] _/δ_ ) _/ϵ_[2] _, then with probability at least_ 1 _− δ,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0021-01.png)


_Proof._ Using Lemma 8 and Cauchy-Schwarz inequality, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0021-03.png)


The statements follow directly from plugging in suitable values of _d_ . 

**Lemma 8** (Almost orthonormal) **.** _Let x_ 1 _, x_ 2 _, . . . , xn i.i.d.∼Nd_ ( **0** _d, d_[1] _[I][d]_[)] _[.][For any][ ϵ, δ][∈]_[(0] _[,]_[ 1)] _[, when] d ≥_ 16 log(2 _n_[2] _/δ_ ) _/ϵ_[2] _, it holds that with probability at least_ 1 _− δ,_ 

_|⟨xi, xj⟩− δij|≤ ϵ, ∀i, j ∈_ [ _n_ ] _._ 

_Proof._ Fix _i ̸_ = _j ∈_ [ _n_ ]. Notice 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0021-08.png)


Using Lemma 9, we have that with probability at least 1 _− δ/n_[2] , _|⟨xi, xj⟩|≤ ϵ._ 

Furthermore, fix _i ∈_ [ _n_ ], with probability at least 1 _− δ/n_[2] , _|⟨xi, xi⟩−_ 1 _|≤ ϵ._ 

The statement then follows from union bound over _i, j ∈_ [ _n_ ]. 

**Lemma 9** (Almost normal for a fixed vector) **.** _For a d-dimensional random vector x ∼Nd_ ( **0** _d, d_[1] _[I][d]_[)] _and any v ∈_ (0 _,_ 1 _/_ 2) _,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0021-13.png)


_In particular, when d ≥_ 16 log(1 _/_ (2 _δ_ )) _/v_[2] _, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0021-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0021-16.png)


The second inequality follows from simple arithmetics. 

21 

## **B.4 Useful results** 

**Lemma 10** (ODE bound) **.** _Let c_ 1 _, c_ 2 _, c_ 3 _>_ 0 _. Suppose the function f_ 1 _, f_ 2 : R+ _→_ R _satisfies f_ 1(0) _>_ 0 _, f_ 2(0) _>_ 0 _and_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0022-02.png)


_Then_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0022-04.png)


_Proof._ The conditions imply that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0022-06.png)


It follows that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0022-08.png)


Rearranging the above inequalities, one can obtain the desired results. 

**Lemma 11** ( _χ_[2] -concentration bound, Lemma 1 of [64]) **.** _Let g_ 1 _, . . . , gt be i.i.d. N_ (0 _,_ 1) _random variables. Then for any x ≥_ 0 _,_ 

_and_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0022-12.png)


## **C Missing Proofs of Section 4** 

In this section, we show missing proofs in Section 4. 

## **C.1 Proofs of Section 4.1** 

## **C.1.1 Proofs of Proposition 4.1 and Proposition 4.2** 

We first show the proofs of Proposition 4.1 and Proposition 4.2, respectively. 

_Proof of Proposition 4.1._ Actually, for any three tokens _x_ 1 _, x_ 2 _, x_ 3, it holds that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0022-19.png)


since _Y_ (0) = **0** . 

22 

_Proof of Proposition 4.2._ Note that the input sequence length _T_ = 2. By (1), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-01.png)


where _**b**_ 2 = [ _b_ 12] and _b_ 12 = 1. Also, _X_ = [ _**x**_ 1] _[⊤]_ is a one-hot row vector. Therefore, LN( _X[⊤]_ _**b**_ 2) = LN( _**x**_ 1 _b_ 12) = LN( _**x**_ 1) = _**x**_ 1, and thus 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-03.png)


where _Y_ ( _t_ ) _i,j_ is the entry of the matrix _Y_ ( _t_ ) at row _i_ and column _j_ . 

## **C.1.2 Proof of Lemma 2** 

_Proof of Lemma 2._ We first calculate the gradient of _Y_ when the current batch is a sequence ( _x_ 1 _, x_ 2 _, x_ 3). Note that the input sequence length _T_ = 2 and by the proof of Proposition 4.2, we have LN( _X[⊤]_ _**b** T_ ) = LN( _X[⊤]_ _**b**_ 2) = _**x**_ 1. By Lemma 1, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-07.png)


where _**α**_ = [ _α_ 1 _, α_ 2 _, . . . , αM_ ] _[⊤] ∈_ R _[M]_ with _**α**_ = exp � _Yx[⊤]_ 1 � _/_ **1** _[⊤]_ exp � _Yx[⊤]_ 1 �. Note that _**x**_ 1( _**x**_ 3 _−_ _**α**_ ) _[⊤]_ is a matrix with only _x_ 1-th row non-zero since _**x**_ 1 is one-hot. Therefore, the update of each row of _Y_ are independent and only _x_ 1-th row of _Y_ gets updated at the current time step. 

Now we consider any fixed _x_ 1 _∈V_ . Let _tx_ 1 _,i_ be the time step that the _x_ 1-th row of Y gets updated (i.e., the first token of the training data is _x_ 1 for the current batch) for the _i_ -th time and let _tx_ 1 _,_ 0 = 0 for notation convenience, then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-10.png)


For convenience, we denote _**y**_ ( _i_ ) = _Y_ ( _tx_ 1 _,i_ ) _[⊤] x_ 1[, and thus] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-12.png)


where _**y**_ (0) = **0** and _**α**_ ( _i −_ 1) = exp( _**y**_ ( _i −_ 1)) _/_ **1** _[⊤]_ exp( _**y**_ ( _i −_ 1)). Note that for a fixed _x_ 1, _x_ 3 is also fixed by our construction of the dataset. By Lemma 5 of [14], we can obtain that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-16.png)


with _h[∗]_ (0) = 0. Combining Lemma 7 and 9 in [14], we have 

Note that the update of each row of _Y_ are independent, the training set has size _N_ , and the batch size is 1, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-19.png)


where the training data at time step _t_ is ( _x_ 1 _, x_ 2 _, x_ 3). Combining (11), we can obtain that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-21.png)


and 

On the other hand, for any sequence ( _x_ 1 _, x_ 2 _, x_ 3) in the test set, since the _x_ 1-th row of _Y_ has never been updated, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0023-24.png)


23 

## **C.1.3 Proof of Theorem 3** 

_Proof._ We first consider training sequence ( _x_ 1 _, x_ 2 _, x_ 3) at time _t_ . By Proposition 4.2, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0024-02.png)


and by Lemma 2, we have 

for some constant _c >_ 0. Therefore, 

where the last inequality holds since 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0024-06.png)


Finally, for any sequence ( _x_ 1 _, x_ 2 _, x_ 3) _∈D_ test, since _Y_ ( _t_ ) _x_ 1 _,x_ = 0 _, ∀x ∈_ [ _M_ ] according to Lemma 2, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0024-08.png)


which completes the proof. 

## **C.2 Additional details and proof of Section 4.2** 

**Datasets.** Let _N_ train _>_ 0, _N_ test _>_ 0 be two positive integers and let _N_ total = _N_ train + _N_ test. Let `A` _i,_ `B` _i,_ `C` _i ∈V, ∀i ∈_ [ _N_ total] be 3 _N_ total distinct tokens. Let _→, ; ∈V_ = [ _M_ ] be two additional different tokens that represent “direct implication” and “indirect implication” respectively. Specifically, we have `A` _i →_ `B` _i_ , `B` _i →_ `C` _i_ and `A` _i ;_ `C` _i_ for all _i ∈_ [ _N_ total]. For notation convenience, we define the following two index sets 

_I_ train = _{_ 1 _,_ 2 _, . . . , N_ train _}, I_ test = _{N_ train + 1 _, . . . , N_ total _}._ 

The training set _D_ train consists of all `A` _i →_ `B` _i_ , `B` _i →_ `C` _i_ and `A` _i ;_ `C` _i_ for _i ∈I_ train. In addition, _D_ train contains `A` _i →_ `B` _i_ and `B` _i →_ `C` _i_ for _i ∈I_ test. For convenience, we let _N_ = _|D_ train _|_ to be the size of the training set. The test set _D_ test consists of `A` _i ;_ `C` _i_ for _i ∈I_ test. Under our construction of the dataset, the LLM will learn the relationship between `A` _i_ , `B` _i_ and `C` _i_ for _i ∈I_ train in both direct and indirect implication, and learn the relationship between `A` _i_ , `B` _i_ and `C` _i_ for _i ∈I_ test only in direct implication and will be tested for indirect implication. 

Similar to the reversal curse in Section 4.1, we aim to prove through the training dynamics of onelayer transformers that the test probability remains negligible during training. In particular, we are interested in 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0024-15.png)


We also use _pθ_ ( `B` _i|_ `A` _i →_ ), _pθ_ ( `C` _i|_ `B` _i →_ ) and _pθ_ ( `C` _i|_ `A` _i ;_ ) to more compactly represent _pθ_ ( _x_ 3 = `B` _i|x_ 1 = `A` _i, x_ 2 = _→_ ), _pθ_ ( _x_ 3 = `C` _i|x_ 1 = `B` _i, x_ 2 = _→_ ) and _pθ_ ( _x_ 3 = `C` _i|x_ 1 = `A` _i, x_ 2 = _;_ ), respectively. 

The following theorem shows the importance of the chain-of-thought method: 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-00.png)


_for some constant c >_ 0 _and_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-02.png)


_Proof._ Recall that by Proposition 4.2, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-04.png)


and by Lemma 2, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-06.png)


for some constant _c >_ 0. Therefore, using the same proof as Theorem 3, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-08.png)


Additionally, according to the proof of Lemma 2, _Y_ ( _t_ ) `A` _i,x_ has the same value across all _x_ = `B` _i_ , which implies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-10.png)


Similarly, applying Lemma 2 to _Y_ ( _t_ ) `B` _i_ , we can obtain that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-12.png)


## **C.3 Analysis of the reversal curse for the four-token sequences** 

In this section, we analyze the reversal curse where data points are four-token sentences “ `AR` 1 `R` 2 `B` ” or “ `BR` 1 `R` 2 `A` ”. For each sentence, `A` and `B` are two distinct tokens that represent two entities, and `R` 1, `R` 2 are two special tokens jointly representing a relationship that is inverse to itself (e.g., `R` 1 `R` 2 represents “is the friend of”, then `AR` 1 `R` 2 `B` means “ `A` is the friend of `B` ” and `BR` 1 `R` 2 `A` means “ `B` is the friend of `A` ” ). 

**Datasets.** Let _N_ train _>_ 0 and _N_ test _>_ 0 and denote _N_ total = _N_ train + _N_ test. Let `A` _i,_ `B` _i ∈ V, ∀i ∈_ [ _N_ total] be _K_ ≜ 2 _N_ total distinct tokens representing distinct entities. WLOG, we assume `A` _i,_ `B` _i ∈_ [ _K_ ] _, ∀i ∈_ [ _N_ total]. Let `R` 1 _,_ `R` 2 _∈V_ be two additional different tokens that jointly represent a relationship that is inverse to itself. Specifically, we have `A` _i_ `R` 1 `R` 2 `B` _i_ and `B` _i_ `R` 1 `R` 2 `A` _i_ for all _i ∈_ [ _N_ total]. For notation convenience, we define the following two index sets 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0025-16.png)


The training set _D_ train consists of all `A` _i_ `R` 1 `R` 2 `B` _i_ and `B` _i_ `R` 1 `R` 2 `A` _i_ for _i ∈I_ train. In addition, _D_ train contains `A` _i_ `R` 1 `R` 2 `B` _i_ for _i ∈I_ test. For convenience, we let _N_ = _|D_ train _|_ to be the size of the training set. The test set _D_ test consists of `B` _i_ `R` 1 `R` 2 `A` _i_ for _i ∈I_ test. Under our construction of the dataset, the LLM will learn the relationship between `A` _i_ and `B` _i_ for _i ∈I_ train in both directions to deduce that the relationship “ `R` 1 `R` 2” is reverse to itself, and learn the relationship between `A` _i_ and `B` _i_ for _i ∈I_ test in one direction and will be tested for the other. WLOG, we assume for each training sequence ( _x_ 1 _,_ `R` 1 _,_ `R` 2 _, x_ 4) _∈D_ train, _x_ 4 _∈_ [ _N_ ]. 

25 

Now we assume that the learning rate _ηY ≫ ηZ_ and therefore can treat _X[⊤]_ _**b** T_ as fixed when analyzing the dynamics of _Y_ . For any sequence ( _x_ 1 _, x_ 2 = `R` 1 _, x_ 3 = `R` 2 _, x_ 4 = _n_ ) in training or test dataset with _T_ = 3, we define _**f** n_ = LN( _X[⊤]_ _**b** T_ ). Then the gradient of _Y_ in (3) becomes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0026-01.png)


Note that for three-token sequences, _**f** n_ is a one-hot vector, and thus _Y_[˙] has only one non-zero row, and each row of _Y_ can be analyzed independently. For the four-token sequences, we use the same reparameterization strategy as in [14], where we denote _W_ = [ _**w**_ 1 _,_ _**w**_ 2 _, . . . ,_ _**w** K_ ] _[⊤]_ ≜ _F[⊤] Y ∈_ R _[K][×][M]_ with _F_ = [ _**f**_ 1 _, . . . ,_ _**f** K_ ] _∈_ R _[M][×][K]_ . 

Note that the parameter _W_ can be viewed as a combination of _Y_ and _Z_ where _Z_ is fixed. The following lemma shows the dynamics of _W_ , assuming we are performing gradient updates on _W_ instead of _Y_ . 

**Lemma 12** (Dynamics of _W_ ) **.** _Assume we perform gradient updates directly on W instead of Y with learning rate ηY and batch size 1, and assume M ≫_ 100 _and M_[0] 1 _[.]_[99] _[≪][η][Y][<]_[1] _[.][Let][ t]_[≳] _[N] η_[ln] _Y[ M] and let W_ ( _t_ ) _i denote the i-th row of W_ ( _t_ ) _and W_ ( _t_ ) _ij denote the_ ( _i, j_ ) _-th entry of W_ ( _t_ ) _. Then for training sequence_ ( _x_ 1 _,_ _`R`_ 1 _,_ _`R`_ 2 _, x_ 4) _∈Dtrain at time t, we have_ 

_W_ ( _t_ ) _x_ 4 _,x_ 4 ≳ ln ( _MηY t/N_ ) _, and W_ ( _t_ ) _x_ 4 _,x_ ≲ _−_ ln ( _MηY t/N_ ) _/M, ∀x ̸_ = _x_ 4 _, and for any test sequence_ ( _x_ 1 _,_ _`R`_ 1 _,_ _`R`_ 2 _, x_ 4) _∈Dtest, we have W_ ( _t_ ) _x_ 4 _,x_ = 0 _, ∀x ∈_ [ _M_ ] _._ 

> _Proof._ According to Lemma 3 of [14], for training sequence ( _x_ 1 _,_ `R` 1 _,_ `R` 2 _, x_ 4) _∈D_ train at time _t_ , only the _x_ 4-th row of _W_ will be updated and the gradient 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0026-07.png)


where _**α**_ ( _t_ ) = exp( _**w** x_ 4( _t_ )) _/_ ( **1** _[⊤]_ exp( _**w** x_ 4( _t_ ))). Therefore, the dynamics of _W_ is nearly identical to the dynamics of _Y_ in Lemma 2, and we can use the proof of Lemma 2 to conclude the results. 

With the dynamics of _W_ , we can obtain the following result: 

**Proposition C.1** (Reversal curse for the four-token sequences) **.** _Assume we perform gradient updates directly on W instead of Y with learning rate ηY and batch size 1, and assume M ≫_ 100 _and M_[0] 1 _[.]_[99] _[≪][η][Y][<]_[ 1] _[.][Let][ t]_[ ≳] _[N] η_[ln] _Y[ M] denote the time step which also satisfies_ ln _t_ ≳ ln( _NM/ηY_ ) _. For training sequence_ ( _x_ 1 _,_ _`R`_ 1 _,_ _`R`_ 2 _, x_ 4) _∈Dtrain at time t, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0026-11.png)


_for some constant c >_ 0 _, and for any test sequence_ ( _x_ 1 _,_ _`R`_ 1 _,_ _`R`_ 2 _, x_ 4) _∈Dtest that is not included in the training set Dtrain, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0026-13.png)


_Proof._ For any sequence ( _x_ 1 _, x_ 2 = `R` 1 _, x_ 3 = `R` 2 _, x_ 4) where _T_ = 3, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0026-15.png)


The above next token probability formulation is almost identical to Proposition 4.2 after replacing _Y_ with _W_ . Combining the dynamics of _W_ as shown in Lemma 12, we can use the proof of Theorem 3 to conclude the result. 

26 

Finally, we discuss the role of _Z_ in the four-token-sequence settings. Note that Proposition C.1 assumes gradient update on _W_ instead of _Y_ . While we are not able to perform gradient updates on _W_ directly, it is equivalent to modifying the gradient of _Y_ to be 

˙ _Y_ = _ηY_ ( _**f** n − FE[′]_ _**e** n_ )( _**e** n −_ _**α** n_ ) _[⊤]_ 

according to Lemma 3 of [14], where the next token _xT_ +1 = _n_ , _E[′]_ = ( _I_ + _E_ ) _[−]_[1] _− I_ , _E_ = _F[⊤] F − I_ , and _F_ = [ _**f**_ 1 _, . . . ,_ _**f** N_ ] _∈_ R _[M][×][N]_ which only contains the next token that appears in the training set. Compared to the original gradient of _Y_ 

˙ _Y_ = _ηY_ _**f** n_ ( _**e** n −_ _**α** n_ ) _[⊤] ,_ 

one can obtain that the modification of the gradient of _Y_ is small if _λ_ 1( _E_ ) is small. 

Note that _Eii_ = _∥_ _**f** i∥_ 2[2] _[−]_[1][=][0][ and] _[ E][ij]_[=] _**[f]**[ ⊤] i_ _**[f]**[j]_[.][Also, for any sequence][ (] _[x]_[1][=] _[n][′][, x]_[2][=] `[R]`[1] _[, x]_[3][=] `R` 2 _, x_ 4 = _n_ ) in training set, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0027-06.png)


where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0027-08.png)


Note that `R` 1 is a common token that appears in each training sentence, and _n[′]_ is a distinct token that only appears in one training sentence as a contextual token. By Theorem 2 of [14], under certain technical assumptions, _Z_[˙] `R` 2 _,n′ >_ 0 and _Z_[˙] `R` 1 _,n <_ 0 and one can expect _Z_ `R` 2 _,n′ − Z_ `R` 1 _,n_ to be sufficiently large after sufficient time of training. Therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0027-10.png)


with[˜] _b_ 23 close to 0. Consider a simple case where for each _n ∈_ [ _N_ ], _**f** n_ = _√_ 1 _− c_[2] _**e** n′_ + _c_ _**e**_ `R` 1 for _c_ sufficiently small. Then 

_Eij_ = _**f** i[⊤]_ _**[f]**[j]_[= (] ~~�~~ 1 _− c_[2] _**e** i′_ + _c_ _**e**_ `R` 1) _[⊤]_ (�1 _− c_[2] _**e** j′_ + _c_ _**e**_ `R` 1) = _c_[2] _._ Therefore, one can calculate that _λ_ 1( _E_ ) = _c_[2] ( _N −_ 1). When _c ≪_ ~~_√_~~ 1 _N_[, we have] _[ λ]_[1][(] _[E]_[)] _[≪]_[1][, and] thus the gradient update of _W_ and gradient update of _Y_ are almost the same. 

## **D Experiments for chain-of-thought** 

In this section, we conduct experiments for COT on multi-layer transformers to validate theoretical results in Section 4.2. 

**Dataset construction.** Similar to Section 5, we randomly sample three disjoint sets of entities _A, B, C ⊂V_ , and reverse two additional tokens for _→_ and _;_ , respectively. Next, we specify a bijection from _A_ to _B_ , and a bijection from _B_ to _C_ randomly. For each `A` _i ∈A_ and its corresponding `B` _i ∈B_ and `C` _i ∈C_ , we can obtain a triple of sequences ( `A` _i →_ `B` _i,_ `B` _i →_ `C` _i,_ `A` _i ;_ `C` _i_ ), and split the set of all triples into training triples and validation triples. All three sequences of a training triple will be added to the training set, while for a validation triple, we add `A` _i →_ `B` _i_ and `B` _i →_ `C` _i_ to the training set and add `A` _i ;_ `C` _i_ to the validation set. Therefore, the model will learn both direct and indirect implications for the training triples and only learn the direct implications for each validation triple while being tested on the indirect implication. 

**Results.** Figure 3 shows the experiment results for COT using the same model architecture and configurations as in Figure 1 (the training set size is 540, and the validation set size is 60 resulting from 140 training triples and 60 validation triples), which is consistent with Theorem 7. One can refer to Appendix E.3 for additional experiments with various model configurations and vocabulary sizes. We also empirically validate the intransitivity of model weights (i.e., logits) for multi-layer transformers in Figure 4, which shows that for a validation triple ( `A` _i,_ `B` _i,_ `C` _i_ ) of which only the direct implication “ `A` _i →_ `B` _i_ ” and “ `B` _i →_ `C` _i_ ” appears in the training set, although the weights from `A` _i_ to `B` _i_ and from `B` _i_ to `C` _i_ are trained large as indicated by the diagonals of the first two bottom matrices, the weights from `A` _i_ to `C` _i_ gets hardly trained as indicated by the diagonals of the last matrix. We also emphasize that another reason that COT is necessary is that all tokens `A` _i_ , `B` _i_ , and `C` _i_ are different tokens with randomly initialized embedding and thus irrelevant. When these tokens are relevant and show specific patterns, the validation loss can also get better. See more details in Appendix E.4. 

27 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0028-00.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs<br>Train<br>80 Val<br>60<br>40<br>20<br>0<br>0 500 1000 1500 2000 2500 3000<br>Number of Epochs<br>Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 3: Experiment results of COT under default configuration (see Table 3). The curves represent the (average) negative log probability of the model predicting the next token to be: (1) `B` _i_ given the input “ `A` _i →_ ”, (2) `C` _i_ given the input “ `B` _i →_ ”, or (3) `C` _i_ given the input “ `A` _i ;_ ”. Similar to the reversal curse experiment, while the sentences in the training set can be learned nearly perfectly, the model is not able to predict the correct next token in the validation set better than a uniformly random guess. Both curves are averaged over 10 random seeds. 

## **E Additional Experimental Results** 

In this section, we show additional experimental results for Section 5. 

## **E.1 Details of model architectures and hyperparameters** 

For both the reversal curse and COT experiments, we used the GPT2 model architecture [63][6] and trained the model with the AdamW optimizer for 3000 epochs of batch size 64. See Table 2 for a full list of hyperparameters. We also conducted experiments under various model configurations and vocabulary sizes to show that the results in Section 5 and appendix D are consistent under different settings. See Table 3 for a complete list of different configurations, where the default choices are boldened. For each curve in all figures, the results are averaged over 10 trials, and the error bar is calculated using standard deviation. We run each trial on an Nvidia A100 GPU and it typically takes 0.5-1.5 hours for each trial. 

|**Parameters**|**Values**|
|---|---|
|Learning Rate|0.01|
|Weight Decay_λ_|0.9|
|(_β_1_, β_2)|(0.9, 0.999)|
|Batch Size|64|
|Number of Epochs|3000|



Table 2: Full list of hyperparameters for AdamW optimizer and training. 

## **E.2 Additional experimental results for the reversal curse** 

In this section, we show additional experimental results for the reversal curse under different configurations, including different vocabulary sizes (Figure 5), different number of layers (Figure 6), 

> 6Apache License 2.0 

28 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0029-00.png)


**----- Start of picture text -----**<br>
Logits of B given A (Train) Logits of C given B (Train) Logits of C given A (Train)<br>100<br>75<br>50<br>25<br>0<br>Logits of B given A (Val) Logits of C given B (Val) Logits of C given A (Val)<br>25<br>50<br>75<br>**----- End of picture text -----**<br>


Figure 4: Visualization of the weights (logits) of the model with default configurations trained after 3000 epochs for COT experiment. The matrices are similar to Figure 2. The row tokens for the top matrices are `A` _i_ , `B` _i_ , `A` _i_ and column tokens are `B` _i_ , `C` _i_ , `C` _i_ for training triples respectively. Similarly, the bottom matrices correspond to validation triples. For validation triples ( `A` _i,_ `B` _i,_ `C` _i_ ), the weights from `A` _i_ to `C` _i_ get hardly trained as indicated by the diagonals of the last matrix. 

|**Parameters**|**Values**|
|---|---|
|Number of Layers<br>Number of Heads<br>Vocabulary Size<br>Entity Length<br>Positional Encoding Type<br>Token,Positional Embedding|12,**24**, 48<br>**12**<br>20, 50, 200,**800**, 2000<br>**1**, 2, 3<br>None,**Absolute**, Relative<br>**Learnable**,Frozen|



Table 3: The list of different configurations for experiments in Appendices E.2 and E.3. Default choices are boldened for each row. 

different positional encoding (Figure 7), different entity lengths (Figure 8) and whether token and positional embeddings are trainable or fixed (Figure 9). Our experimental results consistently show that the reversal curse happens under different settings. 

We provide additional experimental results that (1) the reversal curse still happens even if the embedding dimension is much smaller than the vocabulary size in Appendix E.2.1; (2) the embeddings of different tokens are nearly orthogonal; (3) the reversal curse does not happen under the in-context learning settings. 

## **E.2.1 The reversal curse under small embedding dimensions** 

Although for theoretical analysis in Section 3, we assumed the embedding dimension is polynomial in the vocabulary size, in practice, the embedding dimension only needs to be the order of logarithm of the vocabulary size. Figure 10 shows that for a much smaller embedding size, the reversal curse still happens. 

29 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0030-00.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs 60 Negative Log Probability over Epochs<br>60 Tra i n Train<br>50 Val 50 Val<br>40<br>40<br>30 30<br>20 20<br>10 10<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) Vocabulary size 20 (b) Vocabulary size 50<br>60 Negative Log Probability over Epochs Negative Log Probability over Epochs<br>80 Train<br>50 Val<br>40 60<br>30 Train<br>Val 40<br>20<br>20<br>10<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(c) Vocabulary size 200 (d) Vocabulary size 2000<br>Negative Log Probability Negative Log Probability<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 5: Results for reversal curse for different vocabulary sizes. All other configurations are set as default values as in Table 3. The training set sizes for the above four experiments are 9, 20, 85, 850 respectively, and the validation set sizes are 1, 4, 15, 150 respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0030-02.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>Train Train<br>80<br>Val 60 Val<br>60<br>40<br>40<br>20<br>20<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) 12 layers (b) 48 layers<br>Figure 6: Results for reversal curse for different numbers of layers of the transformer. All other<br>configurations are set as default values as in Table 3.<br>Negative Log Probability over Epochs Negative Log Probability over Epochs<br>70<br>60 Train Train<br>60<br>Val Val<br>50 50<br>40 40<br>30 30<br>20 20<br>10 10<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) No positional encoding (b) Relative positional encoding<br>Negative Log Probability Negative Log Probability<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 7: Results for reversal curse with no positional encoding or relative positional encoding. For relative positional encoding, we follow the Rotary Position Embedding (RoPE) method proposed by [65]. We use the implementation of this repo, MIT license. All other configurations are set as default values as in Table 3. 

30 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0031-00.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>80<br>50 Train Train<br>Train (word) Train (word)<br>40 60<br>Val Val<br>30 Val (word) Val (word)<br>40<br>20<br>20<br>10<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) Entity length 2 (b) Entity length 3<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 8: Results for reversal curse with different entity lengths. Each entity `A` _i_ or `B` _i_ consists of multiple tokens and different entities may have overlapped tokens. The “Train” curve represents the negative log probability of predicting the first token of the output entity, and “Train (word)” represents the negative log probability of predicting all tokens one by one of the output entity. All other configurations are set as default values as in Table 3. The training set sizes for the above two experiments are 680 and 250, respectively, and the validation set sizes are 120 and 50, respectively. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0031-02.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs<br>80 Train<br>Val<br>60<br>40<br>20<br>0<br>0 500 1000 1500 2000 2500 3000<br>Number of Epochs<br>Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 9: Results for reversal curse with fixed token embedding and fixed positional embedding. All other configurations are set as default values as in Table 3. 

## **E.2.2 Near orthogonal embeddings** 

Note that in Section 3, our analysis relies on the fact that embeddings of different tokens are nearly orthogonal, and in Section 4, the embeddings are effectively one-hot. In Figure 11, We show that in practice, even if the embedding dimension is much smaller than the vocabulary size, the near orthogonality condition still holds. 

## **E.2.3 The reversal curse does not happen in ICL settings** 

We also emphasize that the reversal curse does not happen in ICL settings, which means if “ `A` _→_ `B` ” is provided as part of the prompt, then the model is able to answer “ `B` _←_ `A` ”. Figure 12 shows preliminary results of ICL. All the sentences in the dataset have the format of “ `A` _i_ `RB` _j ⇔_ `B` _j_ `R` _[−]_[1] `A` _i_ ”, which is a seven-token sentence where `R` and `R` _[−]_[1] is a pair of relationships inverse to each other, and “ _⇔_ ” is another reserved token representing equivalence. There are ten different `B` _j_ and _n_ different `A` _i_ where _n_ = 100 for the left figure in Figure 12 and _n_ = 200 for the right figure. For each _Ai_ , we construct ten sentences using different _Bj_ , and we randomly chose three of them to be included in the validation set and seven other sentences in the training set. The result of Figure 12 shows that the reversal curse does not happen during the ICL setting. 

31 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0032-00.png)


**----- Start of picture text -----**<br>
Num Layers: 1, Embed Dim: 20 Num Layers: 1, Embed Dim: 768 Num Layers: 24, Embed Dim: 20<br>Train 100 Train 100 Train<br>100<br>Val Val Val<br>80 80<br>80<br>60 60<br>60<br>40 40 40<br>20 20 20<br>0 0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs Number of Epochs<br>Negative Log Probability Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 10: Results for reversal curse for different numbers of layers and different embedding dimensions. All other configurations are set as default values as in Table 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0032-02.png)


**----- Start of picture text -----**<br>
Num Layers: 1, Embed Dim: 20 Num Layers: 1, Embed Dim: 768 Num Layers: 24, Embed Dim: 20 1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>0.25<br>0.50<br>0.75<br>1.00<br>**----- End of picture text -----**<br>


Figure 11: A heat map of cosine similarity between token embeddings (all `A` _i_ and `B` _i_ ) after 3000 epochs training under different settings. The settings are the same as Figure 10. Under different numbers of layers or embedding dimensions, most of the non-diagonal entries are close to 0, which shows that the embeddings of different tokens are nearly orthogonal. 

## **E.3 Additional experimental results for chain-of-thought** 

In this section, we show additional experimental results for COT under different configurations, including different vocabulary sizes (Figure 13), different number of layers (Figure 14), different positional encoding (Figure 15), different entity lengths (Figure 16) and whether token and positional embeddings are trainable or fixed (Figure 17). Note that our experimental results consistently show the necessity of COT under different settings. 

## **E.4 Chain-of-thought with relevant tokens** 

In Appendix D, we briefly mentioned that the irrelevance of different entity tokens is one of the reasons that COT is necessary. Now, we show that if the entity tokens are correlated and show specific patterns, it is possible for a model to deduce indirect implications automatically. 

Instead of using single tokens `A` _i_ , `B` _i_ , `C` _i_ to represent each entity, now we use two tokens `Ai` , `Bi` , `Ci` to represent entities, where `A` , `B` and `C` are three common tokens shared by each triple, and token `i` are distinct for each triple. Figure 18 shows that for the above version of COT where tokens in the same chain are correlated, the model is able to “deduce” `Ai` _;_ `Ci` after training on `Ai` _→_ `Bi` , `Bi` _→_ `Ci` and other training samples. 

32 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0033-00.png)


**----- Start of picture text -----**<br>
Vocab size: 400 Vocab size: 800<br>14<br>Train Train<br>15.0<br>12 Va l Val<br>12.5<br>10<br>8 10.0<br>6 7.5<br>4 5.0<br>2 2.5<br>0 0.0<br>2<br>0 200 400 600 800 1000 0 200 400 600 800 1000<br>Number of Epochs Number of Epochs<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 12: Training and validation loss under in-context learning (ICL) settings. All sentences consist of seven tokens and have the form of “ `A` _i_ `RB` _j ⇔_ `B` _j_ `R` _[−]_[1] `A` _i_ ”. The loss is calculated on the last token. For the left figure, the training set size is 700, the validation set size is 300, and the vocabulary size is 400; for the right figure, the training set size is 1400, the validation set size is 600, and the vocabulary size is 800. All other configurations are set as default values as in Table 3. The result shows that the reversal curse does not happen in ICL settings, i.e., if “ `A` _i_ `RB` _j_ ” is provided as part of the prompt, then the model is able to recognize “ `B` _j_ `R` _[−]_[1] `A` _i_ ”. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0033-02.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>Train 80 Train<br>60 Val Val<br>60<br>40<br>40<br>20<br>20<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) Vocabulary size 20 (b) Vocabulary size 50<br>Negative Log Probability over Epochs Negative Log Probability over Epochs<br>60 Train<br>50 80 Val<br>40 60<br>Train<br>30 Val 40<br>20<br>10 20<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(c) Vocabulary size 200 (d) Vocabulary size 2000<br>Negative Log Probability Negative Log Probability<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 13: Results for COT for different vocabulary sizes. All other configurations are set as default values as in Table 3. The training set sizes for the above four experiments are 14, 32, 135, 1350 respectively, and the validation set sizes are 1, 4, 15, 150 respectively. 

33 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0034-00.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>Train 80 Tra i n<br>80<br>Val Val<br>60<br>60<br>40<br>40<br>20 20<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) 12 layers (b) 48 layers<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 14: Results for COT for different number of layers of the transformer. All other configurations are set as default values as in Table 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0034-02.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>Train Train<br>80<br>80 Val Val<br>60 60<br>40 40<br>20 20<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) No positional encoding (b) Relative positional encoding<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 15: Results for COT with no positional encoding or relative positional encoding. For relative positional encoding, we follow the Rotary Position Embedding (RoPE) method proposed by [65]. All other configurations are set as default values as in Table 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0034-04.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>Train Train<br>50<br>30 Train (word) Train (word)<br>Val 40 Val<br>20 Val (word) 30 Val (word)<br>10 20<br>10<br>0 0<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) Entity length 2 (b) Entity length 3<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 16: Results for COT with different entity lengths. The setting and curves are similar to Figure 8. All other configurations are set as default values as in Table 3. The training set sizes for the above two experiments are 1080 and 400, respectively, and the validation set sizes are 120 and 50. 

34 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0035-00.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs<br>Train<br>50<br>Val<br>40<br>30<br>20<br>10<br>0<br>0 500 1000 1500 2000 2500 3000<br>Number of Epochs<br>Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 17: Results for COT with fixed token embedding and fixed positional embedding. All other configurations are set as default values as in Table 3. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics_images/nips_2024_Towards_a_Theoretical_Understanding_of_the_'Reversal_Curse'_via_Training_Dynamics.pdf-0035-02.png)


**----- Start of picture text -----**<br>
Negative Log Probability over Epochs Negative Log Probability over Epochs<br>Train 20 Train<br>30 Val Val<br>15<br>20<br>10<br>10 5<br>0<br>0<br>5<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(a) Training set size 160 (b) Training set size 250<br>Negative Log Probability over Epochs Negative Log Probability over Epochs<br>20<br>Train Train<br>15 Val 15 Val<br>10 10<br>5 5<br>0 0<br>5 5<br>0 500 1000 1500 2000 2500 3000 0 500 1000 1500 2000 2500 3000<br>Number of Epochs Number of Epochs<br>(c) Training set size 400 (d) Training set size 1600<br>Negative Log Probability Negative Log Probability<br>Negative Log Probability Negative Log Probability<br>**----- End of picture text -----**<br>


Figure 18: Results for COT where each entity is represented by two tokens, i.e., `Ai` , `Bi` , or `Ci` . The validation set sizes are 50. The model is able to “deduce” unseen `Ai` _;_ `Ci` by learning underlying patterns. 

35 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: See abstract and Section 1. Guidelines: 

   - The answer NA means that the abstract and introduction do not include the claims made in the paper. 

   - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

   - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

   - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: See Sections 1 and 6. Guidelines: 

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

- The authors are encouraged to create a separate "Limitations" section in their paper. 

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

## 3. **Theory Assumptions and Proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [Yes] 

36 

Justification: See Sections 3 and 4 and appendices B and C. Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 

Answer: [Yes] 

Justification: See Section 5 and appendices D and E. Guidelines: 

- The answer NA means that the paper does not include experiments. 

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

## 5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

37 

## Answer: [Yes] 

Justification: Our code is available at `https://github.com/marlo-z/reversal_ curse_analysis/` . 

Guidelines: 

- The answer NA means that paper does not include experiments requiring code. 

- Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/ public/guides/CodeSubmissionPolicy` ) for more details. 

- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( `https: //nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. **Experimental Setting/Details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: See Section 5 and appendices D and E. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: See Appendix E.1. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

- The assumptions made should be given (e.g., Normally distributed errors). 

38 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments Compute Resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: See Appendix E.1. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS code of Ethics. 

Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: See Section 1 and Section 6. 

Guidelines: 

- The answer NA means that there is no societal impact of the work performed. 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

39 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: The paper poses no such risks. 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: See Appendix E. 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

40 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: Our code is available at `https://github.com/marlo-z/reversal_ curse_analysis/` . 

Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

## 15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

41 

