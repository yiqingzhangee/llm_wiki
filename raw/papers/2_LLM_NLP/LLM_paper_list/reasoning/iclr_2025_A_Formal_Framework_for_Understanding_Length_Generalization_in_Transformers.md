Preprint 

## A FORMAL FRAMEWORK FOR UNDERSTANDING LENGTH GENERALIZATION IN TRANSFORMERS 

**Xinting Huang**[1] _[∗]_ **Andy Yang**[2] _[∗]_ **Satwik Bhattamishra**[3] **Yash Sarrof**[1] **Andreas Krebs**[4] **Hattie Zhou**[5] **Preetum Nakkiran**[6] **Michael Hahn**[1] _[†]_ 1Saarland University 2University of Notre Dame 3University of Oxford 4University of T¨ubingen 5Mila, Universit´e de Montr´eal 6Apple 

## ABSTRACT 

A major challenge for transformers is generalizing to sequences longer than those observed during training. While previous works have empirically shown that transformers can either succeed or fail at length generalization depending on the task, theoretical understanding of this phenomenon remains limited. In this work, we introduce a rigorous theoretical framework to analyze length generalization in causal transformers with learnable absolute positional encodings. In particular, we characterize those functions that are identifiable in the limit from sufficiently long inputs with absolute positional encodings under an idealized inference scheme using a norm-based regularizer. This enables us to prove the possibility of length generalization for a rich family of problems. We experimentally validate the theory as a predictor of success and failure of length generalization across a range of algorithmic and formal language tasks. Our theory not only explains a broad set of empirical observations but also opens the way to provably predicting length generalization capabilities in transformers. 

## 1 INTRODUCTION 

A key problem in neural sequence modeling is generalization from shorter to longer sequences – _length generalization_ . A wide range of empirical research has found that transformers’ ability at length generalization is mixed, with success found on some problems and failure on others (e.g. Bhattamishra et al., 2020; Anil et al., 2022; Wang et al., 2024a; Kazemnejad et al., 2023; Zhou et al., 2024b; Awasthi and Gupta, 2023; Jelassi et al., 2023; 2024; Chang and Bisk, 2024). For instance, while transformer decoders can easily copy long strings (Bhattamishra et al., 2024), length generalization is substantially more brittle and depends on the absence of repetitions in the string (Zhou et al., 2024a; Jelassi et al., 2024). Similarly, while transformers can in theory simulate many finite-state automata in principle (Liu et al., 2023), their success of length generalization in practice varies widely across different automata (Liu et al., 2023; Bhattamishra et al., 2020). Theoretical understanding of these phenomena is largely lacking, making it difficult to anticipate on which problems transformers will succeed or fail to generalizing beyond the length of their training inputs. 

An important step towards theoretical understanding was made in the RASP-L Conjecture (Zhou et al., 2024a). This conjecture states that transformers show good length generalization exactly on those problems that have simple programs in RASP-L, a fragment of the RASP language (Weiss et al., 2021) with substantial restrictions on the ways in which positional information can be used. While Zhou et al. (2024a) provided empirical evidence in support of this idea, two important gaps remain: First, the RASP-L language has not been fully formalized and its expressiveness itself is not well understood; thus, it is largely open how to prove that a certain problem is indeed not representable in it. Second, while compelling empirical evidence supports a link between definability in RASP fragments and length generalization, no formal proof exists. 

We present a general theoretical framework analyzing length generalization as ultimate identifiability in the limit: When the input-output behavior of a function is observed at longer and longer input lengths, we ask under what conditions a learner can at some point converge on inferring the 

> _∗_ XH and AY are co-first authors. 

> _†_ Lead senior author. Contact: mhahn@lst.uni-saarland.de 

1 

Preprint 

ground-truth function. We answer this in the positive for a specific idealized learning strategy and a well-defined class of functions: whenever a function belongs to this class, transformers are guaranteed to length-generalize in an idealized setting. 

Our results apply to multilayer transformers, focusing on causal transformers with absolute positional encodings (APE) or without positional encodings (NoPE). A key technical challenge in analyzing length generalization for absolute positional encodings is the scaling of the transformer’s parameter count with the input length. To address this, we define a transformer-like limiting object, the _Limit Transformer_ , which encapsulates the computations of a sequence of transformers operating on longer and longer inputs into a single object. We then define an idealized inference procedure in which transformers are fitted to reproduce a target function on successively longer inputs while minimizing a specific norm-based regularizer. Our main theoretical result states that the inference procedure will ultimately lead to length generalization for sufficiently long training inputs, provided the ground-truth function is expressible by a single such limiting object across all input lengths: 

**Theorem 1** (Informal Version of Theorem 7) **.** _Let f be the target function expressible by a single Limit Transformer at all input lengths, subject to restrictions on the use of positional information. Choose transformers Tn (n_ = 1 _,_ 2 _,_ 3 _, . . . ) with context size n, where Tn reproduces the behavior of f up to length[n]_ 2 _[, while minimizing a norm-based regularizer.][Then, for large][ n][,][ T][n][will match the] output of the target function f up to length ≤ n._ 

We then show that the expressivity of Limit Transformers can be understood for many functions. We extend a recently introduced RASP variant ( **C-RASP** , Yang and Chiang, 2024) to provide lower bounds, showing that transformers will succeed at length generalization on various concrete problems under the inference procedure. Conversely, we employ communication complexity to obtain upper bounds on the class of functions for which length generalization is predicted. Experiments confirm the success of the theory at predicting empirical length generalization behavior across various algorithmic tasks and formal languages. Overall, our results formalize the RASP-L Conjecture and take a step toward a theoretical understanding of length generalization. 

## 2 MODEL OF TRANSFORMERS 

**Positional Encoding Scheme** We study two positional encoding schemes. One uses no positional encoding at all; we refer to this as NoPE (No Positional Encoding). The other one uses Absolute Positional Encodings (APE), with learned per-position embedding vectors _**p**_ 1 _, . . . ,_ _**p** N_ . We follow Zhou et al. (2024a) in requiring transformers to be able to perform a task at different offsets within a longer context. Whereas Zhou et al. (2024a) concatenated different examples of a task, we simply encode an input _x_ of length _|x|_ = _k ≤ N_ using positional encodings _**p**_ 1+ _o, . . . ,_ _**p** k_ + _o_ where _o_ is an offset such that _k_ + _o ≤ N_ , and require that the transformer correctly performs the task independently of the offset _o ≥_ 0. This mimics the computations in language models, where the same reasoning task can typically appear at different places in a long context. For simplicity, we treat positions outside of the input, including those preceding the offset, as empty. 

**Parameterization** We focus on transformers with causal masking; for simplicity, we will use the term “transformer” for these throughout. A transformer _T_ is parameterized by a finite alphabet Σ, a width _d ∈_ N, a token embedding matrix _**E** ∈_ R _[|]_[Σ] _[|×][d]_ , a context width _N_ ( _T_ ) _∈_ N _∪{_ + _∞}_ , positional encodings _{_ _**p** i ∈_ R _[d]_ : 1 _≤ i < N_ ( _T_ ) + 1 _}_ , a depth _L ∈_ N and head count _H ∈_ N, key, query, and value matrices _{_ _**K** l,h,_ _**Q** l,h,_ _**V** l,h ∈_ R _[d][×][d]_ : 1 _≤ l ≤ L,_ 1 _≤ h ≤ H}_ , MLP matrices and biases _{_ _**A** l ∈_ R _[d][×][d] ,_ _**B** l ∈_ R _[d][×][d]_ ; _**b** l ∈_ R _[d]_ : 1 _≤ l ≤ L}_ , and an unembedding matrix _**U** ∈_ R _[|]_[Σ] _[|×][d]_ . For matrices, we use _∥· ∥_ and _∥· ∥F_ to denote the spectral and Frobenius norm respectively. 

**Computation of Activations and Outputs** We assume a standard causal transformer, with a few technical points: We explicitly scale attention logits with the logarithm of the input length, omit layer norm, allow Heaviside activations in addition to ReLU activations, and assume that, while the transformer may overall compute at infinite precision, attention logits are operated over at fixed fractional precision. We next define all computations formally. Reserving a special SOS symbol not in Σ, written as “$”, we take the set of input strings to be S, the set of strings _x ∈_ Σ _[∗]_ where _x_ 1 = $ and $ does not occur in _x_ 2 _...|x|_ . We now define the computation of the transformer _T_ on an input _x ∈_ S where 1 _≤|x| < N_ ( _T_ ) + 1 (that is – _|x| ≤ N_ ( _T_ ) if _N_ ( _T_ ) _< ∞_ , and _|x|_ is any finite length 

2 

Preprint 

otherwise). If _L_ is the number of layers, then we write the output of layer _l_ = 1 _, . . . , L_ at position _i_ = 1 _, . . . , N_ ( _T_ ) as _**y** i_[(] _[l]_[)] _∈_ R _[d]_ . Let _o ≥_ 0 be any offset such that _|x|_ + _o < N_ ( _T_ ) + 1 – that is, _x_ still fits into the transformer’s context width if encoded at offset _o_ . Given this offset, we set 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0003-02.png)


where _xi ∈_ Σ is the input symbol at position _i_ . Attention logits, at query position _i_ and key position _j_ are computed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0003-04.png)


We assume standard softmax attention, but incorporate scaling with log _|x|_ following prior work finding it necessary to theoretically represent sparse functions and circumvent theoretical limitations of soft attention (Chiang and Cholak, 2022; Edelman et al., 2022): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0003-06.png)


After each attention block, the activations are passed through a one-layer MLP: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0003-08.png)


where we allow the activation function _ψl_ to be, in each coordinate, either ReLU or Heaviside (see Appendix D.1 for discussion of this). We omit layer norm, as it plays no important role in our results, but it can be accounted for. See Appendix D.3 for the role of layer norm. 

We assume an infinite-precision setup for the activations, with the restriction that attention logits (2) and the output of the exp( _·_ ) function are both rounded to _p_ fractional bits of precision before further processing. This is a mild restriction preventing tiny changes in attention patterns from potentially snowballing into large changes in the output due to infinite precision. See Appendix D.2. 

We conceive of a transformer _T_ as a map from strings _x ∈_ S ( _|x| ≤ N_ ( _T_ )) to vectors of nexttoken prediction logits, _T_ ( _x, o_ ) _∈_ R _[|][x][|×|]_[Σ] _[|]_ , where for _i_ = 1 _, . . . , |x|_ , _T_ ( _x_ ) _i_ = _**Uy** i_[(] _[L]_[)] for the unembedding matrix _**U** ∈_ R _[|]_[Σ] _[|×][d]_ , and _o_ is the offset. In line with the assumed setup, we focus on transformers whose input-output behavior is _invariant across offsets_ : _T_ ( _x, o_ ) = _T_ ( _x, o[′]_ ) for any 0 _≤ o, o[′] ≤ N_ ( _T_ ) _−|x|_ . Let _F_ (Σ) be the class of all maps _f_ mapping _x ∈_ S to _f_ ( _x_ ) _∈_ R _[|][x][|×|]_[Σ] _[|]_ . 

## 3 THEORETICAL FRAMEWORK 

## 3.1 LIMIT TRANSFORMERS 

Our theory addresses the setting of transformers with absolute positional encodings, where the width may grow with the input length. Importantly, we cannot view the ground-truth function as realized by a single transformer: Even if one assigned such a transformer an infinite number of positional encodings, it would still effectively only be able to distinguish between a bounded number of positions, because the width of positional encodings within a single transformer is bounded. Instead, we will derive a parameterization of transformers that allows us to convert sequences of transformers operating on longer and longer sequences to a single limiting transformer-like object. Our key technical idea is to reparameterize the transformer in terms of **product functions** , inner products of parameter vectors as mediated by parameter matrices, such as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0003-15.png)


and various others; see Appendix F.1 for the full formal definition. We first note that the transformer’s computations are uniquely specified by such products. The number of products as in (5) depends, among others, on _|_ Σ _|_ , _L_ , _H_ , _N_ ( _T_ ), but crucially not on _d_ . We will use this parameterization to translate sequences _T_ 1 _, T_ 2 _, T_ 3 _, . . ._ of transformers running on inputs of length 1 _,_ 2 _,_ 3 _, . . ._ to limiting transformer-like objects that are applicable at all input lengths, while keeping width _d_ bounded even if the widths of _Tn_ diverge to infinity. This limiting object, a **Limit Transformer** , differs from an ordinary transformer, as defined in Section 2, just in a few respects. Formally: 

3 

Preprint 

**Definition 2.** _A_ Limit Transformer _is a transformer T where:_ 

_1. N_ ( _T_ ) = + _∞_ 

_2. All parameters (including positional encodings_ _**p** i, and the output of ϕl,h) are expressed in p-bit precision, for some p ∈_ N 

_3. In deviation from ordinary transformers, attention logits on input length N are computed as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0004-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0004-06.png)


Intuitively, a Limit Transformer can use positional information in two ways: through boundedwidth and bounded-precision positional encodings _**p** i_ , and through potentially more complicated functions _ϕl,h_ . Given a transformer, we obtain a Limit Transformer by encoding products of the form _**p**[T] i_ _**[K]** l,h[T]_ _**[Q]**[l,h]_ _**[p]**[j]_[into the functions] _[ ϕ][l,h]_[.][All other product functions involving positional encodings] are expressed in terms of the positional encodings of the Limit Transformer. Our main result will link length generalization to expressibility by Limit Transformers satisfying specific properties: 

**Definition 3.** _A function f_ : N _×_ N _→_ R _is “translation-invariant” if f_ ( _i, j_ ) = _f_ ( _i_ + _τ, j_ + _τ_ ) _, ∀i ≤ j, ∀τ ≥_ 0 _, and “local” if there is τ such that f_ ( _i, j_ ) = 0 _when j > i_ + _τ . A Limit Transformer satisfies (1)_ PERIODIC _if_ _**p** i_ = _**p** i_ +∆ _for all i for some_ ∆ _>_ 0 _, and (2)_ LOCAL _if each ϕl,h is translation-invariant and local._ 

Given a set Θ _n_ of transformers, the parameterization in terms of inner products permits a translation from a transformer _T ∈_ Θ _n_ to a Limit Transformer, whose width is bounded in terms of _R_ ( _T_ ), and which further satisfies PERIODIC and LOCAL. This is formalized in Lemma 52 in the Appendix. 

## 3.2 DEFINITION OF INFERENCE PROCEDURE 

To define the inference procedure, we specify the following hypothesis class at each input length _n_ : **Definition 4** (Hypothesis Class) **.** _For each n_ = 1 _,_ 2 _,_ 3 _, . . . , define the hypothesis class_ Θ _n as the set of transformers T (as defined in Section 2) where (1) N_ ( _T_ ) = _n, (2) each parameter vector and matrix of T is represented at p bits of precision, for some p ∈_ N _, (3) each product function involving positional encodings is translation-invariant. That is, every product function involving exactly one positional encoding is constant across positions, and for every_ 1 _≤ i, j, i_ + ∆ _, j_ + ∆ _≤ n,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0004-12.png)


_whenever_ _**M**_ 1 _. . ._ _**M** k is a product of parameter matrices linking the input layer._[1] 

Note that the width _d_ of the transformers _T ∈_ Θ _n_ is unconstrained. The most interesting requirement here is the third one: We ask that, while the positional encodings _**p** i_ will typically vary with position, their contributions to the transformer’s computations are offset-independent. This is a stronger requirement than for the input-output behavior to be offset-independent: we ask for the transformer’s “algorithm” itself to be the same across offsets. This is a substantive condition, but we believe it to be a natural requirement in the context of length generalization. Our inference procedure will use a regularizer _R_ favoring simpler hypotheses. The following will be sufficient: 

**Definition 5** (Regularizer) **.** _Let T ∈_ Θ _n, thus N_ ( _T_ ) = _n. Define R_ ( _T_ ) _as the sum of (1) L_ + _H; (2) the precision p used in Definition 4; the precision p used for rounding logits and the output of_ exp( _·_ ) _(Section 2); (3)_ max _l,h rank_ ( _**V** l,h_ ) _; (4)_ max _l,h ∥_ _**K** l,h[T]_ _**[Q]**[l,h][∥][;]_[max] _[l,h][ ∥]_ _**[V]**[l,h][∥][;]_[max] _[l][ ∥]_ _**[A]**[l][∥][F][ ,] ∥_ _**B** l∥F ; ∥_ _**U** ∥; (5)_ max _i ∥_ _**p** i∥_ 2 _,_ max _σ ∥_ _**E** σ∥_ 2 _,_ max _l ∥_ _**b** l∥_ 2 _; (6) the term_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0004-16.png)


The idea of (8) is to discourage accidental attention between far-away positions that do not appear together during training, which could hamper length generalization. Due to translation invariance, 

> 1Such as _**K**_ 1 _T,h_ _**[Q]**_[1] _[,h]_[,] _**[ V]**_ 2 _[T] ,h_ _**[K]**_ 3 _[T] ,h[′]_ _**[Q]**_[3] _[,h][′]_ _**[V]**_[1] _[,h][′′]_[, and similar.][See Appendix F.2 for a formal definition.] 

4 

Preprint 

this term entails a bound on products for all pairs _**p** i,_ _**p** j_ ( _i ≤ j_ ) entering causal attention. While such a regularizer is not part of standard training, standard initialization tends to lead to bounded values for (8) when _d_ is large (Appendix G.1); it thus captures an implicit bias of standard initialization and training. Importantly, the width _d_ does not explicitly enter _R_ ; as a consequence, for any sufficiently large _C_ , the number of transformers _Tn ∈_ Θ _n_ with _R_ ( _Tn_ ) _≤ C_ is infinite, simply because _d_ is not constrained. Nonetheless, this regularizer will be sufficient for identification under our idealized inference procedure, which observes the input-output behavior of the target function _f_ on inputs of length _≤[n]_ 2[and selects a transformer] _[ T]_[with maximal context window] _[ n]_[,] _[ T][∈]_[Θ] _[n]_[that exactly fits] that input-output behavior while minimizing the regularizer _R_ ( _T_ ): 

**Definition 6** (Inference Procedure) **.** _Given a function f ∈F_ (Σ) _, the_ Inference Procedure _obtains a sequence of transformers T_ 1 _, T_ 2 _, · · · ∈_ Θ _n as follows. Define Un as the set of T ∈_ Θ _n matching the behavior of f on all inputs of length ≤[n]_ 2 _[.][Then choose][ T][n][∈][U][n][such that]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0005-03.png)


In (9), we do not simply ask for minimizing the regularizer, as the set of elements of _Un_ with _R_ ( _T_ ) smaller than a given value need not be finite and thus a minimum need not be attained by any _Tn_ . Importantly, we only ask _Tn_ to match the behavior of _f_ up to length _[n]_ 2[,][formalizing][the][idea] of training on shorter inputs and testing on longer ones; our identifiability guarantee will provide conditions under which _Tn_ will end up matching _f_ correctly up to length _n_ – representing length generalization. While we take the testing length to be twice the training length, there is nothing special about this; our analysis works whenever the training length diverges to infinity. 

## 3.3 MAIN RESULT: CONVERGENCE OF INFERENCE PROCEDURE 

Our main result asymptotically characterizes length generalization under the inference procedure from Definition 6. For functions representable by Limit Transformers satisfying LOCAL and PERIODIC, we guarantee that _any_ run of the Inference Procedure will ultimately achieve length generalization, so that transformers with context length _n_ chosen to fit the target function on inputs with length _≤[n]_ 2[will, when] _[ n]_[ is sufficiently large, also perform correctly at all lengths] _[ ≤][n]_[.][Formally,] 

**Theorem 7** (Guaranteed Length Generalization in the Limit) **.** _Let f ∈F_ (Σ) _. Then the following are equivalent:_ 

_1. f is expressible by a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _._ 

_2. (Guaranteed Length Generalization) Applying the Inference Procedure from Definition 6 someto f generates N_ 0 _such that, for alla sequence m > NT_ 1 _, T_ 2 _, . . ._ 0 _, Tmwith matches_ sup _n f_ =1 _on all inputs of any length,_ 2 _,_ 3 _,... R_ ( _Tn_ ) _< ∞, for which k ≤ therem. is_ 

The formal proof is in Appendix B.1. Intuitively, if _f_ is expressible by a Limit Transformer satisfying PERIODIC and LOCAL, then, even though the Inference Procedure produces infinitely many distinct _transformers T_ 1 _, T_ 2 _, . . ._ , these can only traverse a finite set of underlying _algorithms_ , each described by some Limit Transformer. PERIODIC and LOCAL ensure that the Limit Transformer’s parameter count effectively remains finite, as its position-related parameters can be fully specified in terms of _**p**_ 1 _, . . . ,_ _**p**_ ∆ _−_ 1 and _ϕ_ (1 _,_ 1) _, . . . , ϕ_ (1 _, τ_ ). The regularizer bounds width, depth, and precision of the Limit Transformers; this keeps the set of algorithms traversed finite. Each of these finitely many algorithms will either be ruled out at some input length _n_ or else match the behavior of _f_ at all input lengths. At some finite _N_ 0, only the latter type of algorithm remains; hence, transformers produced after this point will match the target function. The proof also entails a result on NoPE length generalization: Applying the inference procedure to _f_ while constraining _**p** i ≡_ 0 will lead to length generalization when _f_ is expressible by a Limit Transformer where all _**p** i_ and all _ϕl,h_ are zero (Corollary 18). While Theorem 7 guarantees length generalization from length _[n]_ 2[to] length _n_ for expressible problems, it does not rule out length generalization for inexpressible problems. Such a statement becomes possible if we allow arbitrary scaling of training vs. testing lengths (Appendix B.4). Besides length generalization guarantees, Limit Transformers are also useful in providing _expressiveness_ results for transformers with absolute positional encodings (Appendix C). 

5 

Preprint 

## 4 WHICH FUNCTIONS ARE IDENTIFIABLE? EXPRESSIVENESS OF LIMIT TRANSFORMERS AND **C-RASP** 

We have found that, if a target function _f_ is expressible by a Limit Transformer satisfying PERIODIC and LOCAL, then Theorem 7 indicates length generalization under our Inference Procedure. In order to understand the ramifications of this result, we now study what functions Limit Transformers can express – for these functions, Theorem 7 will then guarantee length generalization. 

## 4.1 SIMPLE EXAMPLE: INDUCTION HEAD 

We consider the task of predicting the next token in proportion to the frequency at which different tokens had previously followed tokens matching the current one: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0006-05.png)


We can construct a Limit Transformer with two layers and one head, wherein the first layer _ϕ_ ( _i, j_ ) = 1 if _i_ + 1 = _j_ and 0 else; each head copies the preceding symbol’s embedding. In the second layer, attention focuses on positions with the same symbol. The transformer outputs next-token predictions in proportion to bigram frequencies in the context, up to approximation error _O_ ( _n_[1][)][ (due] to logit scaling). Hence, Theorem 7 guarantees that the Inference Procedure will length-generalize on (10), providing a length generalization guarantee for an induction head circuit (Olsson et al., 2022). A special case of (10) occurs when each symbol occurs exactly once; here, such an induction head circuit suffices to copy a string (Zhou et al., 2024a), we thus obtain a length generalization guarantee for copying such strings (see Section 5). 

## 4.2 LENGTH GENERALIZATION FOR **C-RASP** 

We next present a large class of functions for which Theorem 7 guarantees length generalization. We extend the **C-RASP** formalism (Yang and Chiang, 2024) with positional information, and then show that any function defined by a **C-RASP** program is expressible by a Limit Transformer; hence, transformers will, by Theorem 7, length-generalize on those functions. We first define **C-RASP** : 

**Definition 8** ( **C-RASP** ) **.** _Let_ Σ _be an alphabet, let_ Φ _be a set of_ unary relations _ϕ_ : N _→{_ 0 _,_ 1 _}, and let_ Ψ _be a set of_ binary relations _ψ_ : N _×_ N _→{_ 0 _,_ 1 _}. A_ _**C-RASP**_ [Φ _,_ Ψ] _program P is defined as a sequence P_ 1 _, . . . , Pk of_ _**C-RASP** operations. There are two sorts of operations:_ 

|**_Boolean-Valued Operations_**<br>**_Initial_**<br>_P_(_i_) :=_Qσ_(_i_)<br>_for σ ∈_Σ<br>**_Boolean_**<br>_P_(_i_) :=_¬P_1(_i_)<br>_P_(_i_) :=_P_1(_i_)_∧P_2(_i_)<br>**_Constant_**<br>_P_(_i_) :=_⊤_<br>**_Positional_**<br>_P_(_i_) :=_ϕ_(_i_)<br>_for ϕ ∈_Φ<br>**_Comparison_**<br>_P_(_i_) :=_C_1(_i_)_≤C_2(_i_)|**_Count-Valued Operations_**|
|---|---|
||**_Counting_**<br>_C_(_i_) := **#**[_j ≤i, ψ_(_i, j_)] _P_(_j_)<br>_for ψ ∈_Ψ_∪{⊤}_|
||**_Conditional_**<br>_C_(_i_) :=_P_(_i_) ?_C_1(_i_)**_:_**_C_2(_i_)|
||**_Addition_**<br>_C_(_i_) :=_C_1(_i_) +_C_2(_i_)|
||**_Subtraction_**<br>_C_(_i_) :=_C_1(_i_)_−C_2(_i_)|
||**_Constant_**<br>_C_(_i_) := 1|
|||



A Counting operation returns the number of positions _j ≤ i_ where _P_ ( _j_ ) and _ψ_ ( _i, j_ ) hold. A conditional operation returns _C_ 1( _i_ ) if _P_ ( _i_ ), and _C_ 2( _i_ ) otherwise. We use the value of the _last_ Boolean-valued operation, at the last position of the string, to determine acceptance using a **C-RASP** program. That is, if the program is run on input _w_ with final operation _L_ , then we accept _w_ if and only if _L_ ( _|w|_ ) is true. **C-RASP** [periodic _,_ local] is the class of **C-RASP** programs where each _ϕ_ ( _i_ ) is periodic in _i_ , and each _ψ_ ( _i, j_ ) is translation-invariant and local (Definition 3). We also write **C-RASP** [periodic _,_ local] for the class of all languages accepted by some **C-RASP** [periodic _,_ local] program. As an example, we present a program recognizing _L_ = Σ _[∗] ab_ Σ _[∗]_ over Σ = _{a, b}_ : 

6 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0007-01.png)


**----- Start of picture text -----**<br>
C-RASP  program for  L  = Σ [∗] ab Σ [∗] over Σ =  {a, b}<br>Ca− ( i ) := #  [ j ≤ i, j =  i − 1] Qa ( i ) # of immediately preceding  a (1)<br>Pa− ( i ) :=  Ca− ( i )  ≥ 1 Position  i − 1 holds an  a (2)<br>Qab ( i ) :=  Qb ( i )  ∧ Pa− ( i ) A substring  ab  ends at position  i (3)<br>Cab ( i ) := #  [ j ≤ i ] Qab ( j ) # of substrings  ab (4)<br>L ( i ) :=  Cab ( i )  ≥ 1 At least one  ab  precedes position  i (5)<br>**----- End of picture text -----**<br>


Any **C-RASP** [periodic, local] program can be translated to a Limit Transformer with corresponding positional functions. We say a Limit Transformer _T accepts_ an input if the value in the last dimension in the last position of the output is greater than 0, and rejects otherwise. 

**Theorem 9.** _For every_ _**C-RASP**_ [Φ _,_ Ψ] _program P with local functions_ Ψ _and periodic functions_ Φ _there exists a Limit Transformer T∞ that satisfies_ PERIODIC _and_ LOCAL _such that for all w ∈_ Σ _[∗] , P accepts w iff T∞ accepts_ $ _w. If P uses no local or periodic relations, then T requires no functions ϕl,h or positional encodings_ _**p** i._ 

The proof is in Appendix B.6. As a consequence, the Inference Procedure will ultimately lengthgeneralize on inputs from a function _f_ expressible by a **C-RASP** [periodic _,_ local] program. If the **C-RASP** program requires no positional functions (i.e., it is in **C-RASP** [ _∅_ ]), then length generalization will succeed even with NoPE transformers. We establish that various functions are in **C-RASP** : **Theorem 10.** _Membership in the following languages is definable in_ _**C-RASP**_ [ _∅_ ] _: (1) MAJORITY, (2) DYCK-1, (3) a[n] b[n] c[n] ._ 

The proof is in Appendix C.1. By Theorem 9, these positive results translate into statements about length generalization under the Inference Procedure. For these tasks, length generalization even with NoPE is empirically already well-documented (Bhattamishra et al., 2020). Further, **C-RASP** [local] can implement versions of the Induction Head task from Section 4.1, see Appendix C.2.1–C.2.2. **C-RASP** also helps understand why transformers show varying abilities even on simple finite-state languages (Bhattamishra et al., 2020; Liu et al., 2023; 2024), a fact poorly understood theoretically. For instance, we have the following: 

**Lemma 11.** _Consider the alphabet_ Σ = _{a, b, e}._ 

_1. PARITY_ := _b[∗]_ ( _ab[∗] ab[∗]_ ) _[∗] ̸∈_ _**C-RASP**_ [ _periodic, local_ ] 

_2._ ( _aa_ ) _[∗] ∈_ _**C-RASP**_ [ _periodic, local_ ] _and_ ( _aa_ ) _[∗] ̸∈_ _**C-RASP**_ [ _∅_ ] 

_3._ Σ _[∗] be[∗] ̸∈_ _**C-RASP**_ [ _periodic, local_ ] 

_4. L ∈_ _**C-RASP**_ [ _∅_ ] _for piecewise testable L_ 

The proof builds on logics with majority quantifiers, whose expressiveness covers **C-RASP** (Appendix C.3). Notably, all of these languages are recognizable by simple finite-state automata which are expressible by transformers (Liu et al., 2023), but empirical length generalization behavior differs in line with **C-RASP** expressiveness (Section 5). PARITY (1) has long been found difficult for transformers (e.g. Hahn, 2020; Bhattamishra et al., 2020; Anil et al., 2022; Chiang and Cholak, 2022; Del´etang et al., 2023; Hahn and Rofin, 2024). Result (2) exemplifies the effect of different positional relations. The language Σ _[∗] be[∗]_ (3) is a simple model of FlipFlop (Liu et al., 2024), a language on which transformers empirically struggle to generalize perfectly despite its simplicity for recurrent models (Liu et al., 2024; Sarrof et al., 2024). The class (4) is useful for determining the expressibility of languages in **C-RASP** [ _∅_ ], as in Section E.1.2. 

## 4.3 LIMITATIONS: LOGARITHMIC COMMUNICATION COMPLEXITY 

Having shown that various functions are definable by Limit Transformers, we now provide a simple technique for showing that various functions are _not_ definable by Limit Transformers. Informally, any function satisfying the conditions in Theorem 7 has logarithmic communication complexity. Formally: 

**Theorem 12.** _Let T be a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _. On an input x ∈_ Σ[2] _[N] , assume Alice has access to x_ 1 _...N and Bob has access to xN_ +1 _..._ 2 _N . Then Alice can communicate C_ log _N bits to Bob, where C depends on T but not N , so that Bob can compute each activation in the second half,_ _**y** i_[(] _[l]_[)] _(N_ + 1 _≤ i ≤_ 2 _N )._ 

7 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Found CRASP[Periodic, Local] Program No CRASP[Periodic, Local] Program Found CRASP[] Program No CRASP[] Program<br>Binary  Binary Majority<br> Majority  Interleave Majority Sort<br>100 100 100 100<br>75 75 75 75 1 2 3 4 5 6<br>50 50 50 50<br>25 25 25 25<br>0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>Copy  Copy  7 8 9 10 11 12<br> Unique  Repeat Parity Addition<br>100 100 100 100<br>75 75 75 75<br>50 50 50 50<br>25 25 25 25 13 14 15 16 17<br>0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>**----- End of picture text -----**<br>


Figure 1: Experimental results (y-axis: accuracy), at lengths _≤_ 50 (Bin 1, training), [51 _,_ 100] (Bin 2), and [101 _,_ 150] (Bin 3, generalization), for APE (solid) and NoPE (dotted). Green lines indicate that we found a **C-RASP** program ( **C-RASP** [periodic _,_ local] for APE, **C-RASP** [ _∅_ ] for NoPE), red lines indicate that we proved nonexistence, or found no program. Random baselines are indicated in gray in (left), and very close to zero in (right). On the algorithmic problems (left), we replicate prior empirical findings; **C-RASP** expressiveness predicts observed length generalization. On the regular languages (right, with same _x_ and _y_ -axes as left, Table 2), length generalization tracks **C-RASP** expressiveness established in Lemma 11 ((1) = ( _aa_ ) _[∗]_ , (17) = Σ _[∗] be[∗]_ ) and other results (see Appendix E.1). **C-RASP** expressiveness performs much better than circuit complexity and standard notions of regular language complexity in predicting length generalization (Appendix, Figures 3–4). 

The proof is in Appendix B.3. In principle, computing activations in the second half of the input requires full knowledge of the first half of the input, because positions in the second half can freely attend to positions in the first half. In this situation, one would expect Bob to need full knowledge of _[N]_[claimed in the theorem.] 2[input symbols from Alice’s part, exponentially more than the] _[ C]_[ log] _[ N]_ This is indeed needed if _T_ performs the task of, say, checking if _x_ 1 _...,N_ and _xN_ +1 _..._ 2 _N_ are identical. However, if _T_ satisfies PERIODIC and LOCAL, attention must largely be determined by the presence of tokens and token sequences; when an attention head’s behavior is determined by positional information, it can only focus its attention on a local neighborhood or equally distribute it over a periodic pattern. Intuitively, in such cases, the set of possible queries and keys can be grouped into a finite partitioning, of size bounded independently of _N_ . It then suffices for Alice to communicate, for each possible group of keys, an aggregate of the value vectors at the positions where a matching key is computed. The proof (Appendix B.3) formalizes this. As a corollary: 

**Corollary 13.** _The following problems are not expressible by Limit Transformers satisfying_ PERIODIC _and_ LOCAL _: (1) copying arbitrary strings, (2) addition of n-digit numbers._ 

This is proven in Appendix B.3. As a consequence, any run of the Inference Procedure on these functions will output solutions _T_ 1 _, T_ 2 _, T_ 3 _, . . ._ for which the depth, number of heads, parameter norms or ranks, MLP dimensions, or precision _p_ , must increase with the input length _n_ ; indeed, length generalization is empirically challenging for these functions (Section 5). 

## 5 EXPERIMENTS 

We evaluate the expressiveness of Limit Transformers and **C-RASP** as a predictor of empirical length generalization of NoPE and APE transformers. Based on Theorems 7 and 9, we expect that APE transformers should length-generalize on problems with a **C-RASP** [periodic,local] program and that NoPE transformers will be successful in those cases where we found a **C-RASP** [ _∅_ ] program. We test this prediction on a suite of algorithmic problems and formal languages, largely taken from prior empirical work on length generalization (Bhattamishra et al., 2020; Zhou et al., 2024a), but evaluated within a uniform framework. 

8 

Preprint 

**Setup** For each task, the model is trained on inputs whose LEN is in the range [ _lmin,_ 50], where _lmin_ is the minimum length for this task. LEN is the length of the input in the algorithmic tasks (Appendix E.2), and the overall sequence length in the formal language tasks. The model is tested on 3 test sets, where LEN is in the range [ _lmin,_ 50], [51 _,_ 100], [101 _,_ 150]; these lengths are based on the source of the regular languages benchmark (Bhattamishra et al., 2020). We trained using a standard AdamW setup; see details in Appendix E.3. Hyperparameters are selected by searching in order of increasing complexity until we find a setting that performs well up to length _≤_ 100. We interpret results on lengths [101 _,_ 150] as a measure of length generalization. Each model has as many positional encodings as needed to encode the longest inputs (at least 150); each input is presented with a random offset in agreement with the theoretical setup. On algorithmic sequence-to-sequence tasks, we train with cross-entropy loss on the output. On formal languages, where next-symbol predictions are generally not deterministic, we instead train the model to predict the _set of legal next symbols_ , with each such set coded as an atomic symbol (as in Bhattamishra et al., 2020; Sarrof et al., 2024). At test time, predictions are considered correct on a sequence if and only if the output at every step is correct; the random baseline is thus very low on the formal language benchmark. We report accuracy, the fraction of test sequences where predictions are correct at every step. 

**Algorithmic Problems** We evaluate on 8 algorithmic problems, which largely overlap with Zhou et al. (2024a), but are tailored to those where **C-RASP** expressiveness can be clearly settled. Tasks are defined formally in Appendix E.2.1. A new problem here is BINARY MAJORITY INTERLEAVE, which interleaves multiple MAJORITY functions and can be solved by **C-RASP** using periodic functions. Length generalization behavior matches **C-RASP** expressiveness; **C-RASP** [ _∅_ ] expressiveness predicts the success of NoPE (see Figure 1). In agreement with prior empirical results (Zhou et al., 2024a; Jelassi et al., 2023), COPY is difficult in the presence of repetition and easy when it is avoided; these findings match **C-RASP** expressiveness (Corollary 13 and Section 4.1). 

**Formal Languages** We applied the experimental framework to 17 regular languages assembled by Bhattamishra et al. (2020), who evaluated length generalization in transformers and LSTMs. Whereas LSTMs perform strongly across the board, the behavior of transformers on these regular languages has so far eluded theoretical understanding. While it is known that transformers struggle with PARITY, it has remained unclear why they would struggle to length-generalize on some seemingly very simple languages. We found **C-RASP** [periodic,local] programs for 13 of the languages and proved nonexistence for the others (Appendix E.1.2). Length generalization succeeded in those cases where we had found a **C-RASP** [periodic,local] program (see Figure 1 right). In those cases where a **C-RASP** [ _∅_ ] program exists, generalization also succeeded with NoPE. Generalization failed for languages where no **C-RASP** program exists, such as Σ _[∗] be[∗]_ (#17 in Figure 1; Lemma 11). 

## 6 DISCUSSION 

Prior work has empirically found that transformers’ length generalization capabilities differ between tasks, but theoretical understanding has been lacking. We have introduced a formal framework analyzing length generalization in an idealized inference procedure. The framework explains what is common across the diverse tasks where prior research has empirically observed successful length generalization, in terms of expressiveness in two simple mathematical formalisms, Limit Transformers and **C-RASP** . We also proved that various problems, on which length generalization is less successful empirically, are not expressible in one or both of these formalisms. Beyond length generalization, the framework further sheds light on the expressiveness of APE transformers. Our results on length generalization study an idealized regularizer and assume perfect fitting of the training distribution. Making the guarantee from Theorem 7 more realistic by incorporating SGD training dynamics and subsampling of training data is an interesting problem for future research. 

Our results can be viewed as formalizing the RASP-L Conjecture (Zhou et al., 2024a). Both Limit Transformers and **C-RASP** [periodic,local] formalize intuitions underlying RASP-L in restricting how positional information can be used. An important advance over Zhou et al. (2024a) is that we settle the expressiveness of these formalisms for many problems, and are able to explicitly prove a variety of problems with poor empirical length generalization, such as copying with repeated strings, to be inexpressible by Limit Transformers. Our results provide a step towards rigorously confirming the idea that expressiveness in such restricted formalisms predicts length generalization. 

9 

Preprint 

**Expressiveness of Transformers** A substantial line of research has studied the in-principle expressiveness of transformers (Strobl et al., 2024). Transformers express a subset of the class **TC**[0] (Merrill and Sabharwal, 2023b; Strobl, 2023), but it is unknown if this inclusion is proper. _All_ problems considered in Section 5 are in **TC**[0] , but empirical length generalization behavior largely tracks **C-RASP** [periodic,local] expressiveness, which defines a proper subclass of **TC**[0] (Appendix C.3.1). While it remains open if the expressive power of transformers exhausts **TC**[0] , our results suggest a separation between **TC**[0] and those problems for which length generalization is possible with absolute positional encodings. In particular, our results suggest that the existence of APE transformers that perform a task across larger ranges of input lengths is linked to the expressiveness of Limit Transformers (Section C). It is an open question how far new, yet-to-be-discovered positional encoding schemes may increase the range of length generalization; empirical evidence indicates that NoPE and APE may be hard to beat by other general-purpose encodings (Kazemnejad et al., 2023). 

The proof of Theorem 12 is closely linked to previous communication-complexity bounds for transformer layers (Sanford et al., 2023; 2024; Peng et al., 2024; Bhattamishra et al., 2024), which importantly were shown only for individual layers, not multilayer transformers. Indeed, Bhattamishra et al. (2024) showed that such a logarithmic bound is not in general possible for arbitrary multilayer transformers. In contrast, our result applies even at multilayer models, which is enabled by the restrictions on the ways in which positional information can be used in a Limit Transformer. 

**Length Generalization of Transformers** Various studies have empirically evaluated length generalization in transformers. Our work is most closely related to Zhou et al. (2024a), discussed above. Bhattamishra et al. (2020) study length generalization on formal languages; we find that **CRASP** [periodic,local] expressiveness explains behavior on their benchmark well (Section 5). Anil et al. (2022) show that language models, finetuned on various reasoning problems, do not lengthgeneralize well. Wang et al. (2024a) evaluate length generalization of NoPE transformers on realworld tasks. Kazemnejad et al. (2023) explore length generalization across different positional encoding schemes, finding NoPE to perform surprisingly well. Zhou et al. (2024b) show that length generalization for addition improves with specific encoding schemes and input formats. Jelassi et al. (2024) show that transformers can succeed in length generalization on copying when inputs avoid _n_ -gram repetition. Chang and Bisk (2024) empirically find limitations in generalization in counting. In contrast to the rich landscape of empirical studies, theoretical understanding of length generalization has been limited. Most relevant, Ahuja and Mansouri (2024) study length generalization in simple neural architectures, including a one-layer transformer setup with linear (not softmax) attention. Our results, in contrast, apply to multi-layer softmax transformers and make statements about many concrete problems that have been studied empirically. Some other works (e.g. Hou et al., 2024; Xiao and Liu, 2023) provide length-generalizing constructions for certain problems but leave open whether learning would lead to such constructions. Wang et al. (2024b) show that GD training leads to length generalization on a specific token selection task. 

**Limitations** The main limitation of our results is that we study idealized asymptotic identification of a global minimum with perfect knowledge of behavior on the training distribution (cf. Sec. 6 and Q.4 in App. A for more discussion). Extending Theorem 7 to account for subsampling of the training data and learning dynamics is an important problem for future research. In particular, providing a practical upper bound on the threshold _N_ 0 at which length generalization is expected is an interesting problem. Our study focuses on absolute positional encodings; extending it to other positional encodings (e.g. Su et al., 2024; Press et al., 2021; Ruoss et al., 2023) is another important problem for future research. 

## 7 CONCLUSION 

We have introduced a theoretical framework that unifies a broad array of empirical findings about successes and failures of length generalization in transformers with absolute positional encodings. Our framework is based on the analysis of an idealized inference procedure, for which length generalization provably happens whenever the ground-truth function is expressible with only limited access to positional information. By providing upper and lower bounds on the expressiveness of these objects, we accurately predict the success and failure of length generalization across various algorithmic tasks and formal languages. 

10 

Preprint 

## CONTRIBUTIONS 

MH coordinated the project. MH and XH developed the conceptual framework of Theorem 7, with input from the other authors. XH and YS contributed Section 5. AY contributed Section 4.2 with input from MH and AK. MH, XH, AY, YS jointly developed the translation to Limit Transformers; MH worked out the formalization. SB contributed Proposition 54, Lemma 56, and provided conceptual input throughout the project. AK contributed to settling the **C-RASP** expressiveness of formal languages. PN and HZ provided conceptual and writing-level input over the course of the project. MH drafted the remaining portions of the paper and the proof of Theorem 7, including definitions and lemmas. 

## ACKNOWLEDGMENTS 

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – ProjectID 232722074 – SFB 1102. MH thanks Lena Strobl, Dana Angluin, David Chiang, Mark Rofin, Anthony Lin, and Georg Zetzsche for conversations on related topics. 

## REFERENCES 

- Ahuja, K. and Mansouri, A. (2024). On provable length and compositional generalization. _arXiv preprint arXiv:2402.04875_ . 

- Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., and Neyshabur, B. (2022). Exploring length generalization in large language models. _Advances in Neural Information Processing Systems_ , 35:38546–38556. 

- Awasthi, P. and Gupta, A. (2023). Improving length-generalization in transformers via task hinting. _CoRR_ , abs/2310.00726. 

- Barcelo, P., Kozachinskiy, A., Lin, A. W., and Podolskii, V. (2024). Logical languages accepted by transformer encoders with hard attention. In _The Twelfth International Conference on Learning Representations_ . 

- Barrington, D. A. M., Compton, K., Straubing, H., and Th´erien, D. (1992). Regular languages in NC1. _Journal of Computer and System Sciences_ , 44(3):478–499. 

- Behle, C., Krebs, A., and Mercer, M. (2007). Linear circuits, two-variable logic and weakly blocked monoids. In Kuˇcera, L. and Kuˇcera, A., editors, _Mathematical Foundations of Computer Science 2007_ , pages 147–158, Berlin, Heidelberg. Springer Berlin Heidelberg. 

- Behle, C., Krebs, A., and Reifferscheid, S. (2009). Regular languages definable by majority quantifiers with two variables. In Diekert, V. and Nowotka, D., editors, _Developments in Language Theory_ , pages 91–102, Berlin, Heidelberg. Springer Berlin Heidelberg. 

- Bhattamishra, S., Ahuja, K., and Goyal, N. (2020). On the ability and limitations of transformers to recognize formal languages. In Webber, B., Cohn, T., He, Y., and Liu, Y., editors, _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020_ , pages 7096–7116. Association for Computational Linguistics. 

- Bhattamishra, S., Hahn, M., Blunsom, P., and Kanade, V. (2024). Separations in the representational capabilities of transformers and recurrent architectures. _CoRR_ , abs/2406.09347. 

- Cadilhac, M. and Paperman, C. (2022). The regular languages of wire linear AC[0] . _Acta Informatica_ , 59(4):321–336. 

- Chang, Y. and Bisk, Y. (2024). Language models need inductive biases to count inductively. _CoRR_ , abs/2405.20131. 

- Chiang, D. and Cholak, P. (2022). Overcoming a theoretical limitation of self-attention. In Muresan, S., Nakov, P., and Villavicencio, A., editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022_ , pages 7654–7664. Association for Computational Linguistics. 

11 

Preprint 

- De la Higuera, C. (2010). _Grammatical inference: learning automata and grammars_ . Cambridge University Press. 

- Del´etang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., and Ortega, P. A. (2023). Neural networks and the chomsky hierarchy. 

- Edelman, B. L., Edelman, E., Goel, S., Malach, E., and Tsilivis, N. (2024). The evolution of statistical induction heads: In-context learning markov chains. _arXiv preprint arXiv:2402.11004_ . 

- Edelman, B. L., Goel, S., Kakade, S., and Zhang, C. (2022). Inductive biases and variable creation in self-attention mechanisms. In _International Conference on Machine Learning_ , pages 5793–5831. PMLR. 

- Hahn, M. (2020). Theoretical limitations of self-attention in neural sequence models. _Transactions of the Association for Computational Linguistics_ , 8:156–171. 

- Hahn, M. and Rofin, M. (2024). Why are sensitive functions hard for transformers? In _Proceedings of the 2024 Annual Conference of the Association for Computational Linguistics (ACL 2024)_ . arXiv Preprint 2402.09963. 

- Hao, Y., Angluin, D., and Frank, R. (2022). Formal language recognition by hard attention transformers: Perspectives from circuit complexity. _Transactions of the Association for Computational Linguistics_ , 10:800–810. 

- Hou, K., Brandfonbrener, D., Kakade, S. M., Jelassi, S., and Malach, E. (2024). Universal length generalization with turing programs. _CoRR_ , abs/2407.03310. 

- Jelassi, S., Brandfonbrener, D., Kakade, S. M., and Malach, E. (2024). Repeat after me: Transformers are better than state space models at copying. In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_ . OpenReview.net. 

- Jelassi, S., d’Ascoli, S., Domingo-Enrich, C., Wu, Y., Li, Y., and Charton, F. (2023). Length generalization in arithmetic transformers. _CoRR_ , abs/2306.15400. 

- Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., and Reddy, S. (2023). The impact of positional encoding on length generalization in transformers. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S., editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ . 

- Krebs, A. (2008). _Typed semigroups, majority logic, and threshold circuits_ . PhD thesis, Universit¨at T¨ubingen. 

- Lange, K.-J. (2004). Some results on majority quantifiers over words. In _Proceedings. 19th IEEE Annual Conference on Computational Complexity, 2004._ , pages 123–129. IEEE. 

- Liu, B., Ash, J., Goel, S., Krishnamurthy, A., and Zhang, C. (2024). Exposing attention glitches with flip-flop language modeling. _Advances in Neural Information Processing Systems_ , 36. 

- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., and Zhang, C. (2023). Transformers learn shortcuts to automata. In _The Eleventh International Conference on Learning Representations_ . 

- McNaughton, R. and Papert, S. A. (1971). _Counter-Free Automata (MIT research monograph no. 65)_ . The MIT Press. 

- Merrill, W. and Sabharwal, A. (2023a). The expressive power of transformers with chain of thought. In _NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning_ . 

- Merrill, W. and Sabharwal, A. (2023b). A logic for expressing log-precision transformers. In _Thirtyseventh Conference on Neural Information Processing Systems_ . 

- Merrill, W. and Sabharwal, A. (2023c). The parallelism tradeoff: Limitations of log-precision transformers. _Transactions of the Association for Computational Linguistics_ , 11:531–545. 

12 

Preprint 

- Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N. (2017). Exploring generalization in deep learning. _Advances in neural information processing systems_ , 30. 

- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al. (2022). In-context learning and induction heads. _arXiv preprint arXiv:2209.11895_ . 

- Peng, B., Narayanan, S., and Papadimitriou, C. (2024). On limitations of the transformer architecture. _arXiv preprint arXiv:2402.08164_ . 

- Press, O., Smith, N. A., and Lewis, M. (2021). Train short, test long: Attention with linear biases enables input length extrapolation. _arXiv preprint arXiv:2108.12409_ . 

- Ruoss, A., Del´etang, G., Genewein, T., Grau-Moya, J., Csord´as, R., Bennani, M., Legg, S., and Veness, J. (2023). Randomized positional encodings boost length generalization of transformers. _arXiv preprint_ . 

- Sanford, C., Hsu, D., and Telgarsky, M. (2024). One-layer transformers fail to solve the induction heads task. _arXiv preprint_ . 

- Sanford, C., Hsu, D. J., and Telgarsky, M. (2023). Representational strengths and limitations of transformers. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S., editors, _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ . 

- Sarrof, Y., Veitsman, Y., and Hahn, M. (2024). The expressive capacity of state space models: A formal language perspective. _CoRR_ , abs/2405.17394. 

- Sch¨utzenberger, M. P. (1965). On finite monoids having only trivial subgroups. _Inf. Control._ , 8(2):190–194. 

- Shazeer, N. (2020). GLU variants improve transformer. _arXiv preprint arXiv:2002.05202_ . 

- Strobl, L. (2023). Average-hard attention transformers are constant-depth uniform threshold circuits. _CoRR_ , abs/2308.03212. 

- Strobl, L., Merrill, W., Weiss, G., Chiang, D., and Angluin, D. (2024). What Formal Languages Can Transformers Express? A Survey. _Transactions of the Association for Computational Linguistics_ , 12:543–561. 

- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_ , 568:127063. 

- Tesson, P. and Th´erien, D. (2002). Diamonds are forever: The variety DA. In _Semigroups, algorithms, automata and languages_ , pages 475–499. World Scientific. 

- Tomita, M. (1982). Dynamic construction of finite-state automata from examples using hillclimbing. In _Proceedings of the Fourth Annual Conference of the Cognitive Science Society_ , pages 105–108. 

- Wang, J., Ji, T., Wu, Y., Yan, H., Gui, T., Zhang, Q., Huang, X., and Wang, X. (2024a). Length generalization of causal transformers without position encoding. In Ku, L., Martins, A., and Srikumar, V., editors, _Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024_ , pages 14024–14040. Association for Computational Linguistics. 

- Wang, Z., Wei, S., Hsu, D., and Lee, J. D. (2024b). Transformers provably learn sparse token selection while fully-connected nets cannot. In _Forty-first International Conference on Machine Learning_ . 

- Weiss, G., Goldberg, Y., and Yahav, E. (2021). Thinking like transformers. In _International Conference on Machine Learning_ , pages 11080–11090. PMLR. 

13 

Preprint 

- Xiao, C. and Liu, B. (2023). Conditions for length generalization in learning reasoning skills. _CoRR_ , abs/2311.16173. 

- Yang, A. and Chiang, D. (2024). Counting like transformers: Compiling temporal counting logic into softmax transformers. In _First Conference on Language Modeling_ . 

- Yang, A., Chiang, D., and Angluin, D. (2023). Masked hard-attention transformers recognize exactly the star-free languages. _arXiv Preprint_ . 

- Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J. M., Bengio, S., and Nakkiran, P. (2024a). What algorithms can transformers learn? A study in length generalization. In _The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024_ . OpenReview.net. 

- Zhou, Y., Alon, U., Chen, X., Wang, X., Agarwal, R., and Zhou, D. (2024b). Transformers can achieve length generalization but not robustly. _CoRR_ , abs/2402.09371. 

14 

## **Contents** 

|**1**|**Introduction**|**Introduction**||**1**|
|---|---|---|---|---|
|**2**|**Model of Transformers**|||**2**|
|**3**|**Theoretical Framework**|||**3**|
||3.1|Limit Transformers . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|3|
||3.2|Defnition of Inference Procedure|. . . . . . . . . . . . . . . . . . . . . . . . . .|4|
||3.3|Main Result: Convergence of Inference Procedure . . . . . . . . . . . . . . . . . .||5|
|**4**|**Which Functions are Identifable? Expressiveness of Limit Transformers and C-RASP**|||**6**|
||4.1|Simple Example: Induction Head .|. . . . . . . . . . . . . . . . . . . . . . . . . .|6|
||4.2|Length Generalization for**C-RASP**|. . . . . . . . . . . . . . . . . . . . . . . . .|6|
||4.3|Limitations: Logarithmic Communication Complexity<br>. . . . . . . . . . . . . . .||7|
|**5**|**Experiments**|||**8**|
|**6**|**Discussion**|||**9**|
|**7**|**Conclusion**|||**10**|
|**A **|**FAQ**|||**16**|
|**B**|**Proofs about Limit Transformers**|||**18**|
||B.1|Proof of Theorem 7 . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|18|
||B.2|Result for NoPE Transformers . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|21|
||B.3|Logarithmic Communication Complexity for Limit Transformer . . . . . . . . . .||21|
||B.4|Statement of Main Theorem for Arbitrary Training Lengths . . . . . . . . . . . . .||22|
||B.5|Corollary about Expressivity . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|23|
||B.6|From**C-RASP**to Limit Transformers . . . . . . . . . . . . . . . . . . . . . . . .||24|
|**C **|**Expressivity Proofs for C-RASP**|||**26**|
||C.1|**C-RASP**Constructions . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|||C.1.1<br>Majority<br>. . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|||C.1.2<br>Dyck-1 . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|||C.1.3<br>_anbncn_ . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|26|
|||C.1.4<br>Existential Quantifcation|. . . . . . . . . . . . . . . . . . . . . . . . . .|27|
|||C.1.5<br>Piecewise Testable Languages . . . . . . . . . . . . . . . . . . . . . . . .||27|
||C.2|**C-RASP**[periodic_,_local]Constructions . . . . . . . . . . . . . . . . . . . . . . . .||27|
|||C.2.1<br>Induction Head (Argmax Version) . . . . . . . . . . . . . . . . . . . . . .||27|
|||C.2.2<br>Induction Head (All possible next symbols) . . . . . . . . . . . . . . . . .||28|
|||C.2.3<br>(_aa_)_∗_. . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|29|
||C.3|Expressibility of Regular Languages in**C-RASP**[periodic,local] . . . . . . . . . .||29|
|||C.3.1<br>Link to Majority Logic . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|29|
|||C.3.2<br>Inexpressibility ofΣ_∗be∗_.|. . . . . . . . . . . . . . . . . . . . . . . . . .|32|
|||C.3.3<br>Inexpressibility of PARITY|. . . . . . . . . . . . . . . . . . . . . . . . .|34|
|**D **|**Discussion of Design Choices**|||**36**|
||D.1|MLP Activation Functions . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|36|
||D.2|Fixed Precision . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|37|
||D.3|Layer Norm . . . . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|37|
|**E**|**Additional Details for Experiments**|||**37**|
||E.1|Regular Languages from the Bhattamishra et al 2020 Benchmark . . . . . . . . . .||37|
|||E.1.1<br>Language Defnitions . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|37|
|||E.1.2<br>**C-RASP**Expressiveness .|. . . . . . . . . . . . . . . . . . . . . . . . . .|38|
||E.2|Algorithmic Tasks . . . . . . . . .|. . . . . . . . . . . . . . . . . . . . . . . . . .|41|
|||E.2.1<br>Task Defnitions for Algorithmic Problems<br>. . . . . . . . . . . . . . . . .||41|



15 

Preprint 

|||E.2.2<br>Limit Transformers and**C-RASP**expressiveness on algorithmic tasks . . .|43|
|---|---|---|---|
||E.3|Details of Experimental Setup<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|44|
|**F**|**Translating between Transformers and Limit Transformers**||**45**|
||F.1|Product Parameterization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|45|
||F.2|Formal Defnition of Hypothesis Class . . . . . . . . . . . . . . . . . . . . . . . .|49|
||F.3|From Limit Transformers to Transformers . . . . . . . . . . . . . . . . . . . . . .|49|
||F.4|From Transformers to Limit Transformers . . . . . . . . . . . . . . . . . . . . . .|54|
|||F.4.1<br>Proving Lemma 52 (I): Preliminaries<br>. . . . . . . . . . . . . . . . . . . .|55|
|||F.4.2<br>Proving Lemma 52 (II): Construction of_T∞_. . . . . . . . . . . . . . . . .|56|
|||F.4.3<br>Proving Lemma 52 (III): Proving Correctness . . . . . . . . . . . . . . . .|60|
|**G **|**Additional Supporting Results**||**68**|
||G.1|Regularizer at Initialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|68|
||G.2|Empirical Length Generalization of Positional Functions . . . . . . . . . . . . . .|70|
||G.3|Bound for Encodings Norm in Terms of Function Complexity<br>. . . . . . . . . . .|70|



## A FAQ 

## **(1)** _What is the point of introducing Limit Transformers?_ 

Limit Transformers are a mathematical formalism helping us prove a length generalization guarantee (Theorem 7) for a broad class of functions, not just one specific function. They thus serve as an object that can help us prove things about standard transformers. 

**(2)** _What is the relation between Limit Transformers and_ _**C-RASP** ? Why use two different formalisms?_ 

Limit Transformers are closely connected to standard transformers and provide a convenient formalism for formalizing a length generalization guarantee in our inference setup (Theorem 7); they also provide bounds on APE transformer expressiveness as a side result (Appendix C). **C-RASP** is a formalism based on the RASP language (Weiss et al., 2021), intended to provide a formal abstraction of the kinds of computations that transformers can perform in a human-readable format. Limit Transformers with PERIODIC and LOCAL express all the functions definable in **C-RASP** [periodic,local], though it is open if this inclusion is strict. We provide rigorous tools for understanding the expressiveness of both formalisms. For Limit Transformers, we prove a logarithmic communication complexity bound (Theorem 12). **C-RASP** brings additional use in understanding expressiveness from two angles. First, one can conveniently prove functions expressible by writing down programs, as we did in Section 4.2. Second, to prove negative results, we can bring to bear a set of deep results about logics using majority quantifiers (Krebs, 2008), which allow us to settle the expressiveness of many problems provably. Positive results translate into positive results about Limit Transformer expressiveness and hence, length generalization under our idealized learning setup. While it is open if problems not expressible in **C-RASP** cannot in principle show length generalization, experimental results suggest that such an implication might hold in many cases. 

**(3)** _Why are Limit Transformers needed – can’t one just consider transformers whose parameters have infinite precision and hence can accommodate infinitely many different positional encodings_ _**p** i?_ 

The key advantage of Limit Transformers is that they effectively have finite parameter counts whenever they satisfy LOCAL and PERIODIC, which is useful in establishing Theorem 7. In an ordinary transformer, due to fixed width, effectively distinguishing unboundedly many positions requires infinitely many parameters _**p** i_ . Even then, a function as simple as _ϕ_ ( _i, j_ ) = _δij_ cannot be exactly represented for infinitely many _i, j_ by a product _**p** i_ _**Q**[T]_ _**Kp** j_ at bounded width _d_ . 

**(4)** _Why is the idealized setup considered for the analysis, as opposed to more practical frameworks of learning?_ 

Proving guarantees in a more practical setting (SGD training dynamics, subsampling of training data) would, of course, be ideal. However, such guarantees have been notoriously difficult to establish for deep learning models (Neyshabur et al., 2017). Standard frameworks for learning, such as 

16 

Preprint 

PAC-learning, assume that the training and test distributions are the same, which precludes out-ofdistribution guarantees such as length generalization. Even within the PAC-learning framework, obtaining nontrivial guarantees for deep neural networks remains challenging without making strong assumptions. Instead of analyzing the learning and generalization of Transformers trained with gradient-based methods, our work aims to understand the length generalization properties of Transformers from an architectural perspective. A substantial body of work (cf. Section 6) has empirically investigated the length generalization properties of Transformers and found a complex array of empirical behavior, while theoretical understanding has been very limited. Hence, consolidating the theoretical relation between these empirical observations and the computational model of Transformers seems like an important direction. Our work provides a formal framework, based on an idealized model of learning, that separates the tasks on which Transformers succeed and those on which they fail to length-generalize. The learning model considered in our work is closely related to the “identification in the limit” setting, which has been widely studied for decades in the context of learning automata and grammars (De la Higuera, 2010). Our framework is successful in explaining a wide range of empirical observations (Figure 1). This is a substantial advance, as no prior theoretical framework has been able to explain the empirical patterns in Figure 1 to the extent that our framework can. We hope that further work can build on these insights to establish guarantees that reproduce this success while narrowing the gap between theoretical analysis and practical learning. 

**(5)** _Why does the length generalization condition in Theorem 7 ask for_ sup _n R_ ( _Tn_ ) _< ∞? Isn’t asking for length generalization sufficient?_ 

If sup _n R_ ( _Tn_ ) = _∞_ , a transformer _T_ minimizing _R_ ( _T_ ) while fitting behavior at some length will be unlikely to work at substantially longer lengths because performing the task correctly at longer and longer lengths requires unbounded increase in _R_ ( _T_ ). It might still happen that generalization from length _[n]_ 2[to][length] _[n]_[is][possible][in][certain][problems][not][expressible][by][Limit][Transformers.] However, this will depend on the problem and the specific scaling of test lengths relative to training lengths; for problems not satisfying the conditions in Theorem 7, length generalization will fail when the test length is made sufficiently longer than the training length, even as the training length diverges to infinity. We make this formal in Section B.4. 

## **(6)** _Given a task, how can one settle Limit Transformer and_ _**C-RASP** expressiveness?_ 

Showing that a task is definable by Limit Transformers or **C-RASP** simply requires providing an explicit construction, as we exemplify for various tasks (Section C.1). For showing that a task is not definable in these formalisms, we provide a battery of methods that allow us to provide an answer for many tasks: communication complexity (Theorem 12) applies to both formalisms; for showing non-definability in **C-RASP** , reduction to specific languages already proven not to be expressible (such as Parity and _Lbb_ , see Appendix E.1.2) is frequently useful. 

**(7)** _Why does the guarantee specifically apply to_ PERIODIC _and_ LOCAL _Limit Transformers? What is special about such positional relations?_ 

Local positional relations are important because, if a product function of the form _**p**[T] i_ _**[Q]**[T]_ _**[ Kp]**[j]_[,] where the rank of _**Q** ,_ _**K**_ is not constrained, takes nonzero values at unboundedly long distances _j −i_ , there is no general reason why the function should length-generalize. Independent initialization of the _**p** i_ ’s tends to lead to values close to zero for most of these products (Appendix G.1); our Inference Procedure incorporates this via the term (8). Given this, one expects a learned model to still exhibit small products at distances not present in the training distribution, and hence a failure of length generalization in the presence of nonlocal product functions. 

The situation is different for products involving _**V** l,h_ matrices, whose rank is penalized by _R_ ( _·_ ); these are able to represent not local, but periodic functions. In the finite-precision setup, a translationinvariant product function of the form _**p**[T] i_ _**[M]**_[1] _[ . . .]_ _**[ M]**[k]_ _**[p]**[j]_[must][be][periodic][in] _[j][−][i]_[whenever][one] of the matrices _**M**_ 1 _. . ._ _**M** k_ has bounded rank as the number of positions considered diverges to infinity, with period bounded in terms of the rank (Lemma 48). Hence, in a transformer _T ∈_ Θ _n_ , any product function involving one or more _**V** l,h_ matrices needs to be periodic with period bounded in terms of _R_ ( _T_ ). 

17 

Preprint 

## B PROOFS ABOUT LIMIT TRANSFORMERS 

## B.1 PROOF OF THEOREM 7 

We re-state and then prove Theorem 7: 

**Theorem 14** (Guaranteed Length Generalization in the Limit, restated from Theorem 7) **.** _Let f ∈ F_ (Σ) _. Then the following are equivalent:_ 

_1. f is expressible by a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _._ 

_2. (Guaranteed Length Generalization) Consider the inference procedure from Definition 6 applied to f with R, generating a sequence T_ 1 _, T_ 2 _, . . . . For_ any _such sequence, there is some N_ 0 _such that, for all m > N_ 0 _, Tm matches f on all inputs of any length k ≤ m, and_ sup _n_ =1 _,_ 2 _,_ 3 _,... R_ ( _Tn_ ) _< ∞._ 

**Remark 15.** _We note that a limit transformer T∞ representing f need not itself be offset-invariant. It is sufficient to have_ 

_T∞_ ( _x,_ 0) = _f_ ( _x_ ) (6) 

_Lemma 47 shows that such a function has a sequence of transformers Tn ∈_ Θ _n which are offsetinvariant, even without assuming T∞ to be offset-invariant._ 

**High-Level Proof Sketch** The key to the proof is a compactness property: Any sequence _T_ 1 _, T_ 2 _, . . ._ ( _Ti ∈_ Θ _i_ ) where sup _i R_ ( _Ti_ ) _< ∞_ has a subsequence of transformers whose behavior across inputs can be summarized into a single Limit Transformer. For 1 _⇒_ 2, given a sequence generated by the Inference Procedure, we show that _R_ stays bounded and use the compactness property to show that a subsequence exhibits behavior equivalent to _f_ . To show that, in fact, _all_ possible sequences _Tn_ generated by the Inference Procedure ultimately exhibit behavior equivalent to _f_ , when _n_ is large, we show that subsequences failing to length-generalize would exhibit increasing attention dot products between far-away positions as input length increases. However, due to the penalty on attention dot products in _R_ , any such sequence would, for large _n_ , need to have a higher value of _R_ than sequences avoiding such an increase. For 2 _⇒_ 1, we obtain the Limit Transformer from the compactness property applied to the sequence generated by the Inference Procedure. The penalty on attention dot products enforces that it satisfies LOCAL; the bounds on the MLP and value matrices enforce that the positional encodings in the Limit Transformer can be taken to be periodic. 

**Preliminaries and Formal Proof** We now proceed to the formal proof. We make crucial use of the two technical Lemmas 47 and 52, which provide translations between ordinary transformers and Limit Transformers. 

The following definition will be used: 

**Definition 16.** _If T ∈_ Θ _i, then define R−_ ( _T_ ) _to be R_ ( _T_ ) _minus the term in Eq. (8). That is,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0018-14.png)


The following lemma will be used for both directions of the main theorem: 

**Lemma 17.** _Let T_ 1 _, T_ 2 _, . . . , where Tn ∈_ Θ _n, be a sequence generated by the Inference Procedure based on the functional behavior of a function f ∈F, and such that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0018-17.png)


_Then f is expressible by a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _, and there is some N_ 0 _such that, for all m > N_ 0 _, Tm matches f on all inputs of length k ≤ m._ 

_Proof._ From the sequence _T_ 1 _, T_ 2 _, . . ._ generated by the Inference Procedure, we obtain, using Lemma 52, Limit Transformers _T_[˜] 1 _, T_[˜] 2 _, . . ._ such that sup _i R∞_ ( _T_[˜] _i_ ) _< ∞_ where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0018-20.png)


18 

Preprint 

and, in each _Tn_ , _T_[˜] _n_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-02.png)


Due to sup _i R∞_ ( _T_[˜] _i_ ) _< ∞_ , we know that, except for the functions _ϕl,h_ , only a finite number of Limit Transformer parameter settings will be traversed by _T_[˜] _i_ . Each function _ϕl,k_ is local; however, a priori, they might not be local for any single finite _τ_ across the different _T_[˜] _n_ . We will show that this is not possible, i.e., we will show that all _ϕl,k_ are local for a single finite _τ_ across the different _T_[˜] _n_ . This will occupy us for the remainder of the proof. 

First, we note that _R_ ( _Tn_ ) converges because inf _T ∈Un R_ ( _T_ ) is bounded and monotonically increasing in _n_ . For each _τ_ and each _n_ , we consider 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-05.png)


where the _ϕl,h_ function is taken from _T_[˜] _n_ when defining _Dn_ . 

Consider _R−_ ( _Tn_ ) (Equation 7). Let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-08.png)


and let _ν_ 1 _, ν_ 2 _, ν_ 3 _, . . ._ be such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-10.png)


Then, for some _D_ 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-12.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-14.png)


Indeed, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-16.png)


because[2] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-18.png)


Define, for each _τ ∈_ N, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-20.png)


As this function is monotonically increasing, and as _ϕl,h_ has bounded precision, there must be _τ∞_ such that _D∞_ ( _τ∞_ ) = lim _τ →∞ D∞_ ( _τ_ ). 

Now define a sequence _Tn[′]_[by selecting, for each] _[ n]_[, an] _[ i]_[(] _[n]_[)][ such that] _[ ν] i_ ( _n_ ) _[≥][n]_[ and] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-23.png)


(with equality when _νj ≥ τω_ ); then define _Tn[′]_[as][the][restriction][of] _[T][ν] i_ ( _n_ )[to][positions][up][to] _[n]_[.][As] _Tνi_ ( _n_ ) agrees with the behavior of _f_ up to length _νi_ 2( _n_ ) _≥[n]_ 2[,][we][also][find][that] _[T][ ′] n_[agrees][with][the] behavior of _f_ up to length _[n]_ 2[.][Then] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0019-25.png)


> 2 In general, if _an_ + _bn_ converges and _an, bn_ are bounded, then the limit lim( _an_ + _bn_ ) equals lim sup _an_ + lim inf _bn_ . For, assume lim sup _an_ + lim inf _bn >_ lim( _an_ + _bn_ ) (similar if _>_ is replaced by _<_ ). Then let _i_ ( _n_ ) be a subsequence such that _ai_ ( _n_ ) _→_ lim sup _an_ . Then lim( _an_ + _bn_ ) = lim( _ai_ ( _n_ ) + _bi_ ( _n_ )) = lim sup _an_ + lim _bi_ ( _n_ ) _≥_ lim sup _an_ + lim inf _bn >_ lim( _an_ + _bn_ ), contradiction. 

19 

Preprint 

Since _Tn_ was created by the Inference Procedure, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-03.png)


giving 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-05.png)


Hence, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-07.png)


and _D∞_ ( _τ∞_ ) = _D_ 0. Now assume there are infinitely many _n_ such that _ϕl,h_ is not _τ∞_ -local in _Tn_ , hence, infinitely many _n_ such that _Dn_ ( _n_ ) _≥ Dn_ ( _τ∞_ ) + 2 _[−]_[2] _[p]_ . Then: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-09.png)


This is a contradiction. 

We thus have shown that the functions _ϕl,k_ ( _T_[˜] _n_ ) must be local for a uniform _τ∞_ . We thus know that the sequence _T_[˜] _i_ only traverses a finite set of possible Limit Transformers. The set of traversed functions becomes stationary at some _i_ = _N_ 0; all of these must be functionally equivalent to _f_ . Hence, _Ti_ is functionally equivalent to _f_ at all lengths _≤ i_ as soon as _i_ exceeds some threshold _N_ 0. 

We now prove the theorem. 

_Proof of the Theorem._ Both directions are corollaries of Lemma 17. 

**2** _⇒_ **1:** This directly follows from Lemma 17. 

**1** _⇒_ **2:** By Lemma 47, for each _i_ = 1 _,_ 2 _,_ 3 _, . . ._ , there are _T_[�] _i ∈_ Θ _i_ such that _R_ := sup _i R_ ( _T_[�] _i_ ) _< ∞_ such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-18.png)


By LOCAL, 

and we conclude 

Lemma 17 now provides _N_ 0 _>_ 0 and a function _g_ such that for all _m > N_ 0, 

On the other hand, for any string _x ∈_ S, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-23.png)


Hence, _f ≡ g_ and for all _m > N_ 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0020-25.png)


20 

Preprint 

## B.2 RESULT FOR NOPE TRANSFORMERS 

**Corollary 18.** _For ease of the reader, we mark the differences to Theorem 7 in blue font._ 

_Let f ∈F_ (Σ) _. Then the following are equivalent:_ 

_1. f is expressible by a Limit Transformer satisfying where all_ _**p** i ≡_ 0 _, ϕl,h ≡_ 0 _._ 

_2. (Guaranteed Length Generalization) Consider the inference procedure from Definition 6 applied to f with R while constraining all_ _**p** i ≡_ 0 _, generating a sequence T_ 1 _, T_ 2 _, . . . . For_ any _such sequence, there is some N_ 0 _such that, for all m > N_ 0 _, Tm matches f on all inputs of any length k ≤ m, and_ sup _n_ =1 _,_ 2 _,_ 3 _,... R_ ( _Tn_ ) _< ∞._ 

_Proof._ Retracing the proof of Lemma 47 shows that, when translating a Limit Transformer to an ordinary transformer, the positional encodings can be taken to be zero when _**p** i ≡_ 0, _ϕl,h ≡_ 0 in the Limit Transformer. Retracing the proof of Lemma 52 shows that, when _**p** i ≡_ 0 in a transformer, the resulting Limit Transformer will have zero positional encodings and zero outputs for all _ϕl,h_ . The proof of Theorem 7 then applies equally to show Corollary 18. 

## B.3 LOGARITHMIC COMMUNICATION COMPLEXITY FOR LIMIT TRANSFORMER 

**Theorem 19** (Restated from Theorem 12) **.** _Let T be a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _. Assume that T operates in precision O_ (log _N_ ) _, i.e., attention weights are rounded to O_ (log _N_ ) _precision. On an input x ∈_ Σ[2] _[N] , assume Alice has access to x_ 1 _...N and Bob has access to xN_ +1 _..._ 2 _N . There is a communication protocol in which Alice and Bob exchange at most C_ log _N bits, where C depends on T but not N or x, and Bob can compute each activation in the second half,_ _**y** i_[(] _[l]_[)] _(N_ + 1 _≤ i ≤_ 2 _N ). Further, C is bounded linearly by R∞_ ( _T_ ) _._ 

_Proof._ First, note that all activations _**y** i_[(] _[l]_[)] are computed at log _N_ precision because parameters are _·_ at fixed precision and the output of exp( ) in the softmax attention computation is computed at fixed fractional precision. We first consider the attention logits, in the case where _j < N ≤ i_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0021-10.png)


where Round _p_ [ _. . ._ ] rounds each entry to the closest number with _p_ fractional bits. It is certainly sufficient to have access to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0021-12.png)


where _p[′]_ depends on _p_ and the largest singular value of _**K** l,h[T]_ _**[Q]**[l,h]_[, which is a finite constant.][We can] thus partition the positions _j_ = 1 _, . . . , N −_ 1 into a bounded number of sets, indexed by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0021-14.png)


2. max( _N − j, N − L_ ) where _L_ = max _{k_ : _ϕl,h_ (1 _, k_ ) _̸_ = 0 _}_ . 

Due to the finite precision rounding of logits and the locality of positional relations, we can maintain a finite set of keys and queries (though not values). This is fundamental to getting a logarithmic communication bound. 

We show the claim by induction over the layers. 

We can write 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0021-19.png)


21 

Preprint 

The residual stream is known to Bob by inductive hypothesis. We need to understand the term inside the sum. The green terms are fully known to Alice, and the blue ones are fully known to Bob by inductive hypothesis: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0022-02.png)


Alice can communicate the green terms for every set in the partitioning of the indices _j < N_ defined above. In fact, it is sufficient to communicate the number of relevant positions and the sum of the vectors _**V** l,h_ _**y** j_[(] _[l][−]_[1)] . 

**Corollary 20** (Restated from Corollary 13) **.** _The following problems are not expressible by Limit Transformers satisfying_ PERIODIC _and_ LOCAL _: (1) copying strings with repeated n-grams, (2) addition of n-digit numbers._ 

_Proof._ Formally, we define _copying_ as the task of, given a prefix $ _x_ #, autoregressively predicting _x_ . Copying _with repeated n-grams_ means that there is no restriction on the repetition of consecutive subspans of _x_ of any length; this is in contrast to copying tasks with restrictions on the repetition of n-grams (for some _n_ ) in _x_ (Jelassi et al., 2024; Zhou et al., 2024a), which we study separately (Appendix E.2). 

Formally, we define _addition_ as the task of, given a prefix $ _x_ + _y_ =, where _x, y_ are binary strings, to output the sum of the numbers denoted by _x, y_ in binary. 

The communication complexity lower bound for copying follows from a standard communication complexity lower bound for determining string equality. The bound follows for addition since the special case of adding 0 to a number amounts to copying. 

**Remark 21.** _Analogous bounds follow for various other algorithmic and formal language problems. For instance, the special case of multiplying with 1 amounts to copying; hence, such a bound holds for multiplication. For the unbounded-depth Dyck over two bracket types, we can consider a word of the form_ ( _i_ 1 _. . ._ ( _iN_ ) _jN . . ._ ) _j_ 1 _, which is in the Dyck language if and only if ik_ = _jk for all k, again allowing a reduction to the communication complexity lower bound for determining string equality._ 

## B.4 STATEMENT OF MAIN THEOREM FOR ARBITRARY TRAINING LENGTHS 

Our main theorem considers generalization from length _[n]_ 2[to length] _[ n]_[.][Here, we discuss an alterna-] tive version applying to arbitrary scaling of training vs testing lengths. In particular, in such a setup, we explicitly obtain _failure of length generalization for inexpressible functions_ , though potentially requiring testing on lengths more than twice the lengths used in training. We use the following definition: 

**Definition 22.** _A_ training length _is a function t_ : N _→_ N _satisfying_ lim _t→∞ t_ ( _n_ ) = + _∞ and t_ ( _n_ ) _≤ n for all n._ 

_If t_ ( _n_ ) _is a training length, then the t_ ( _n_ ) _-Inference Procedure determines Tn ∈_ Θ( _n_ ) _to match f at all inputs of lengths ≤ t_ ( _n_ ) _while minimizing R_ ( _Tn_ ) _up to n_[1] _[.]_ 

_The special case of t_ ( _n_ ) = _[n]_ 2 _[is the Inference Procedure from Definition 6.]_ 

We then state: 

**Theorem 23.** _Let f ∈F_ (Σ) _. The following are equivalent:_ 

_1. f is expressible by a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _._ 

22 

Preprint 

_2. Let t_ ( _n_ ) _be any training length. Then the t_ ( _n_ ) _-Inference Procedure will output solutions T_ 1 _, T_ 2 _, . . . such that, for some N_ 0 _, for all m > N_ 0 _, Tm matches f at all lengths ≤ m._ 

Intuitively, this says that, when selected to fit the behavior of _f_ on sufficiently long inputs of length _t_ ( _n_ ), the output of the Inference Procedure will generalize to unboundedly longer inputs of length _n_ , where _n_ can be arbitrarily larger than _t_ ( _n_ ). 

**Corollary 24.** _Assume f ∈F_ (Σ) _is not expressible by a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _. Then, for some training length t_ ( _n_ ) _, the t_ ( _n_ ) _-Inference Procedure outputs a sequence Tn where infinitely many Tn fail to match f at length n._ 

**Remark 25.** _There are two important differences compared to Theorem 7. First, the second condition refers to length generalization for all arbitrary training lengths t_ ( _n_ ) _, not specifically training length[n]_ 2 _[.][Second, the second condition does not ask for]_[ sup] _[i][ R]_[(] _[T][i]_[)] _[ <][ ∞][, but simply asks for][ T][n][ to] ultimately length generalize._ 

_Proof of Theorem 23._ 1 _⇒_ 2 The proof of Theorem 7 remains valid in this direction without any changes, as it does not specifically rely on the training lengths being half the overall context size. 

2 _⇒_ 1 We show the contrapositive. Assume _f_ is not expressible by a Limit Transformer satisfying PERIODIC and LOCAL. Then, using the same arguments as in the proof of Lemma 17[3] , any sequence _Tn ∈_ Θ _n_ that matches _f_ will have lim inf _n→∞ R_ ( _Tn_ ) = _∞_ ( _†_ ). Now consider _k ∈_ N; we will assign every _k_ a number _nk > k_ , starting with _n_ 0 = 0. For each _n > k_ , there is _T_[ˆ] _k,n ∈_ Θ _n_ that matches _f_ up to length _k_ while _Uk_ := sup _n R_ ( _T_[ˆ] _k,n_ ) _< ∞_ for every fixed _k_ . Now select _nk > nk−_ 1 such that no _T ∈_ Θ _nk_ with _R_ ( _T_ ) _≤ Uk_ + 1 matches _f_ at length _nk_ ; this is possible because of ( _†_ ). We thus obtain a sequence ( _k, nk_ ) _∈_ N _×_ N. By construction, there are infinitely many distinct different values _nk_ . Then define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0023-07.png)


Then _t_ ( _n_ ) is a training length. By definition, the _t_ ( _n_ )-Inference Procedure will, whenever _n_ is one of the _nk_ ’s, find a transformer _Tnk_ with _R_ ( _Tnk_ ) _≤ Uk_ + _n_[1] _k_[that fails to match] _[ f]_[at length] _[ n]_[ =] _[ n][k]_[.] 

## B.5 COROLLARY ABOUT EXPRESSIVITY 

We have introduced Limit Transformers as a formalism for distilling computations of transformers performing on longer and longer sequences into a single limiting object, helping understand length generalization. Here, we show that they also provide a simple lower bound for the expressiveness of causal transformers across input lengths: 

**Corollary 26.** _Let f ∈F_ (Σ) _. Assume f is expressible by a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _. Then at each input length N , there exists a transformer TN performing f on all inputs of length up to N such that:_ 

_1. The parameters of TN are expressed at p bit precision, with p independent of N_ 

_2. The number of heads and layers of TN is bounded independently of N ._ 

_3. The width d of TN is bounded as O_ ( _N_ ) _._ 

We note that an important aspect is that _TN_ performs correctly not just at length _N_ , but at _all lengths up to N_ . This distinguishes the result from constructions guaranteeing the existence of a transformer at a fixed length. For instance, Bhattamishra et al. (2024) provide a transformer for testing equality between length _N_ -strings (which could also be used for copying), but this construction uses specific positional encodings that depend on the input length. In contrast, the result here provides conditions under which a transformer can perform a task at all lengths up to a given bound; in this stronger 

> 3Assume there is a sequence _Tn ∈_ Θ _n_ that matches _f_ and has lim inf _n→∞ R_ ( _Tn_ ) _< ∞_ . Translating each element to a Limit Transformer leads to a sequence where, except perhaps for the functions _ϕl,h_ , only a finite number of settings will be traversed. Now, as in the proof of Lemma 17, one can use _D∞_ ( _τ_ ) to construct a sequence of Limit Transformers that are local for a single _τ_ . The important difference to Lemma 17 is that here we are not assuming the sequence ( _Tn_ ) _n_ to be constructed by the inference procedure, but we nonetheless obtain such a sequence. 

23 

Preprint 

setup, no APE transformer for copying with uniform complexity bounds as provided by Corollary 26 is known, and the problem is indeed not expressible by Limit Transformers satisfying PERIODIC and LOCAL (Corollary 13). In contrast, Corollary 26 provides APE constructions performing correctly _up to_ any given length for a wide class of problems including **C-RASP** [periodic,local]. 

Another important feature is that the construction provides a fixed precision for the parameters, as is the case in real-world implementations. We note that, if parameters are at fixed precision, it is generally not possible to find a single transformer across all input lengths in the APE setting; hence, it is unavoidable that the width of the transformers will need to increase as the input length increases. Importantly, many other aspects of the transformer’s complexity, such as the number of heads and layers, remain bounded. 

_Proof._ The statement is an immediate corollary of Lemma 47, which provides transformers _T_ 1 _, T_ 2 _, . . ._ with bounded _R_ ( _TN_ ), which by Definition 5 entails a uniform bound on precision, heads, and layers. The construction provided in the proof of Lemma 47 provides a width bounded as _O_ ( _N_ ). 

## B.6 FROM **C-RASP** TO LIMIT TRANSFORMERS 

The proofs are adaptations of the proofs from Yang and Chiang (2024). 

**Theorem 27** (Restated from Theorem 9) **.** _For every_ _**C-RASP**_ [Φ _,_ Ψ] _program P with local functions_ Ψ _and any periodic functions_ Φ _there exists a Limit Transformer T∞ that satisfies_ PERIODIC _and_ LOCAL _such that for all w ∈_ Σ _[∗] , P accepts w iff T∞ accepts_ $ _w. If P uses no local or periodic relations, then T requires no functions ϕl,h or positional encodings_ _**p** i._ 

**Remark 28.** _We note that the Limit Transformer T∞ provided by the proof of Theorem 27 emulates the_ _**C-RASP** program P at zero offset: That is, P accepts w iff a predetermined entry in the last output dimension of T∞_ ($ _w,_ 0) _is above some threshold. In principle, its computations may not be offset-invariant, i.e., for the constructed T∞, the output T∞_ ($ _w, o_ ) _may depend on o. Importantly, the proof of Theorem 7 does not require a Limit Transformer computing f to be offset-invariant, but just requires it to compute f when the offset is zero. This is because Lemma 47 ensures that, for any Limit Transformer T∞ satisfying_ LOCAL _and_ PERIODIC _, even if it is not offset-invariant, there are transformers Tn ∈_ Θ _n whose behavior matches T∞_ ( _·,_ 0) _._ 

_Proof of Theorem 27._ **C-RASP** has two sorts of operations, a Boolean sort and a Count sort. We will simulate each operation in the transformer by storing the Boolean values as _{_ 0 _,_ 1 _}_ , and storing the counts as _c_[That is, we say that a Limit Transformer] _[ T][∞]_[simulates a] **[ C-RASP]**[ program] _[ P]_[if for] _i_ +1[.] every operation _Pk_ of _P_ there is a dimension _dk_ in _T_ such that when _Pk_ ( _i_ ) when run on _w_ is true iff _T∞_ ($ _w_ ) _i_ +1 _,dk_ = 1 (and 0 otherwise) for Boolean operation and _Pk_ ( _i_ ) = _c_ iff _T∞_ ($ _w_ ) _i_ +1 _,k_ = _i_ +1 _c_ for count operations. 

The theorem will be shown by induction on the length of _P_ . As a clarifying note, we use 0-indexing everywhere in this proof. If _P_ is of length 0, we only have initial _Qσ_ ( _i_ ) vectors, which can be simulated by appropriately setting the word embedding. Otherwise, assume all programs of length _≤ k_ are simulated by some transformer, and we have cases for each type of operation _Pk_ +1( _i_ ) can be. All cases are identical to Yang and Chiang (2024) except for comparison, conditional, and counting. 

First, we must address the SOS token $. There exists a transformer layer that sets the entire vector to **0** in the initial position while leaving all other layers untouched. For instance, we may use a conditional operation, as described later in the proof. 

If _Pk_ +1( _i_ ) := _ϕ_ ( _i_ ), a periodic positional function in Φ, then it is simulated in _T∞_ by appropriately setting _**p** i_ in the positional encoding. 

If _Pk_ +1( _i_ ) := _P_ ( _i_ ) ? _C_ 1( _i_ ) **:** _C_ 2( _i_ ), we can implement the following function: for _P ∈{−_ 1 _,_ 1 _}_ and _V ∈_ [0 _,_ 1] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0024-13.png)


24 

Preprint 

This is achieved by _f_ ( _P, V_ ) = ReLU( _V − P_ ) _−_ ReLU( _−P_ ). Thus, the desired Conditional Output can be defined in a single FFN as _f_ ( _P, V_ 1) + _f_ ( _−P, V_ 2), where the first layer and ReLU compute each _f_ term and the second layer adds them together. 

If _Pk_ +1( _i_ ) := _C_ 1( _i_ ) _≤ C_ 2( _i_ ). By the inductive hypothesis _C_ 1( _i_ ) and _C_ 2( _i_ ) are stored in dimensions _d_ 1 and _d_ 2 as the value _[C] i_ +1[1][(] _[i]_[)][and] _[C] i_ +1[2][(] _[i]_[)][.][It suffices to check that] _[C] i_ +1[2][(] _[i]_[)][=] _[C] i_ +1[1][(] _[i]_[)] _[≥]_[0][.] 

To compute this, we use the Heaviside activation function, which we used in our model of MLPs as discussed in D.1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0025-04.png)


Thus, there exists an MLP which, letting _x_ 1 and _x_ 2 be the values in dimensions _d_ 1 and _d_ 2, computes (hs( _x_ 2 _− x_ 1) + 1) _/_ 2 in the dimension reserved for _Pk_ +1, which will be the Boolean value in _{_ 0 _,_ 1 _}_ corresponding to _C_ 1( _i_ ) _≤ C_ 2( _i_ ). 

If _C_ ( _i_ ) := **#** [ _j ≤ i_ ] _P_ ( _j_ ) (using _ψ_ ( _i, j_ ) = _⊤_ ), then the desired sum is computed using uniform attention since the boolean representation of _P_ ( _j_ ) is just 0 or 1. We enforced that _P_ (0) is false, so it does not contribute to the sum. This is described in more detail in Yang and Chiang (2024), though the case here is simpler. 

If _C_ ( _i_ ) := **#** [ _j ≤ i, ψ_ ( _i, j_ )] _P_ ( _j_ ), we can think of it as implementing **#** [ _j ≤ i_ ] _ψ_ ( _i, j_ ) _∧ P_ ( _j_ ). Suppose _ψ_ is a local function of the following form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0025-08.png)


Then _C_ ( _i_ ) will either be 1 or 0 depending if _P_ ( _i − ℓ_ ) is true or false. If we set the query and key matrices to 0 we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0025-10.png)


We assume the log is base 2, but the argument is similar for others. Then we can have attention compute 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0025-12.png)


If _P_ ( _i − ℓ_ ) and _¬P_ ( _j_ ) for _j_ = _i − ℓ_ , then we have a lower bound: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0025-14.png)


If _¬P_ ( _i − ℓ_ ) and _P_ ( _j_ ) for _j_ = _i − ℓ_ then we have an upper bound: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0025-16.png)


25 

Preprint 

1 Since _N_ ln 2 _≥ i_ , and we know that _P_ ( _i−ℓ_ ) _⇐⇒ ci,k ≥_ 2[1][, we can construct an MLP that computes] the correct value. It will output either _i_ +10[or] _i_ +11[, in the dimension reserved for] _[ P][k]_[+1][(] _[i]_[)][, for instance] by using a conditional operation that checks that the output of the attention layer _ci_ +1 _,k ≥_[1] 2[, which] was shown in an earlier case. 

## C EXPRESSIVITY PROOFS FOR **C-RASP** 

## C.1 **C-RASP** CONSTRUCTIONS 

## C.1.1 MAJORITY 

MAJORITY is the language of strings over Σ = _{_ 0 _,_ 1 _}_ with at least as many 1’s as 0’s. 

## MAJORITY 

_C_ 1( _i_ ) := **#** [ _j ≤ i_ ] _Q_ 1( _i_ ) (1) _C_ 0( _i_ ) := **#** [ _j ≤ i_ ] _Q_ 0( _i_ ) (2) _M_ ( _i_ ) := _C_ 1( _i_ ) _≥ C_ 0( _i_ ) (3) 

C.1.2 DYCK-1 

Dyck-1 is the language of strings over Σ = _{_ 0 _,_ 1 _}_ with at least as many 1’s as 0’s. 

## Dyck-1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0026-11.png)


**----- Start of picture text -----**<br>
C (( i ) := #  [ j ≤ i ] Q (( j ) The number of ( up to position  i (1)<br>C )( i ) := #  [ j ≤ i ] Q )( j ) The number of ) up to position  i (2)<br>V  ( i ) :=  C (( i )  < C )( i ) Violation : there are more ) than ( (3)<br>CV  ( i ) := #  [ j ≤ i ] V  ( j ) The number of  Violations (4)<br>M ( i ) :=  CV  ( i ) = 0 Matched : zero  Violations (5)<br>B ( i ) :=  C (( i ) =  C )( i ) Balanced : same number of ( and ) (6)<br>D ( i ) :=  M ( i )  ∧ B ( i ) String is  Matched  and  Balanced (7)<br>**----- End of picture text -----**<br>


C.1.3 _a[n] b[n] c[n]_ 

Let Σ = _{a, b, c}_ . This is another example of a counter language which **C-RASP** can express and which transformers have been observed to length generalize on (Bhattamishra et al., 2020). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0026-14.png)


**----- Start of picture text -----**<br>
a [n] b [n] c [n]<br>Ca ( i ) := #  [ j ≤ i ] Qa ( j ) (1)<br>Cb ( i ) := #  [ j ≤ i ] Qb ( j ) (2)<br>Cc ( i ) := #  [ j ≤ i ] Qc ( j ) (3)<br>A ( i ) :=  Cb ( i ) +  Cc ( i ) = 0 (4)<br>B ( i ) :=  Cc ( i ) = 0 (5)<br>CA ( i ) := #  [ j ≤ i ] Qa ( j )  ∧ A ( j ) (6)<br>CB ( i ) := #  [ j ≤ i ] Qb ( j )  ∧ B ( j ) (7)<br>Ga ( i ) :=  CA ( i ) =  Ca ( i ) (8)<br>Gb ( i ) :=  CB ( i ) =  Cb ( i ) (9)<br>Gabc ( i ) :=  Ca ( i ) =  Cb ( i ) =  Cc ( i ) (10)<br>L ( i ) :=  Ga ( i )  ∧ Gb ( i )  ∧ Gabc ( i ) (11)<br>**----- End of picture text -----**<br>


26 

Preprint 

## C.1.4 EXISTENTIAL QUANTIFICATION 

This is generally a useful primitive, so to save a little space we can add a macro for existential quantification towards the left in **C-RASP** . This is easily defined using counting: 

_P_ ( _i_ ) := _[←−] ∃ A_ ( _i_ ) _C_ ( _i_ ) := **#** [ _j ≤ i_ ] _A_ ( _j_ ) (1) _P_ ( _i_ ) := _C_ ( _i_ ) _≥_ 1 (2) 

And we abbreviate this using _P_ ( _i_ ) := _[←−] ∃ A_ ( _i_ ). We demonstrate its use below. 

## C.1.5 PIECEWISE TESTABLE LANGUAGES 

Piecewise testable languages are Boolean combinations of languages of the form Σ _[∗] a_ 1Σ _[∗] a_ 2Σ _[∗] . . ._ Σ _[∗] an_ Σ _[∗]_ . This allows us to check for the presence of noncontiguous substrings, which contrasts with the proof in C.3.2 that implies the presence of contiguous substrings cannot be expressed in **C-RASP** [ _∅_ ]. 

It suffices to show programs for languages of the form _L_ = Σ _[∗] a_ 1Σ _[∗] a_ 2Σ _[∗] . . ._ Σ _[∗] an_ Σ _[∗]_ , since Boolean combinations are recognizable using Boolean operations of **C-RASP** . For _L_ we have the following **C-RASP** program which has the final accepting operation _Ln_ : 

## Σ _[∗] a_ 1Σ _[∗] a_ 2Σ _[∗] . . ._ Σ _[∗] an_ Σ _[∗]_ 

_L_ 1( _i_ ) := _[←−] ∃ Qa_ 1 ( _i_ ) _a_ 1 occurred (1) _L_ 2( _i_ ) := _[←−] ∃ Qa_ 2 ( _i_ ) _∧ L_ 1( _i_ ) _a_ 2 occurred, preceded by _a_ 1 (2) ... (3) _Ln_ ( _i_ ) := _[←−] ∃ Qan_ ( _i_ ) _∧ Ln−_ 1( _i_ ) _an_ occurred, preceded by _an−_ 1, . . . , preceded by _a_ 1 (4) 

C.2 **C-RASP** [PERIODIC _,_ LOCAL] CONSTRUCTIONS 

C.2.1 INDUCTION HEAD (ARGMAX VERSION) 

As an example consider Σ = _{a, b, c}_ . Predicate _NEXTa_ ( _i_ ) is true iff the next token should be an _a_ . First we can define predecessor 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0027-13.png)


Then we can count bigram occurence by counting 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0027-15.png)


Then each _NEXTa_ ( _i_ ) predicate can be defined by checking the current symbol and finding the most frequently occuring bigram. 

27 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0028-01.png)


**----- Start of picture text -----**<br>
NEXTa ( i ) (Argmax) over Σ =  {a, b, c}<br>... (1)<br>MOREaa,ab ( i ) :=  Caa ( i )  ≥ Cab ( i ) (2)<br>MOREaa,ac ( i ) :=  Caa ( i )  ≥ Cac ( i ) (3)<br>NEXTa ( i ) :=  Qa ( i )  ∧ MOREaa,ab ( i )  ∧ MOREaa,ac ( i ) (4)<br>**----- End of picture text -----**<br>


This corresponds to testing, for the _f_ in Equation 10, for which _σ_ the entry _f_ ( _x_ 1 _. . . xN_ ) _N,σ_ is maximal. 

C.2.2 INDUCTION HEAD (ALL POSSIBLE NEXT SYMBOLS) 

Consider Σ. For _a ∈_ Σ, predicate _NEXTa_ ( _i_ ) is true iff the next token can possibly be an _a_ . As in Section C.2.1, first, we can define predecessor 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0028-05.png)


Then we can check for bigram occurrence by counting 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0028-07.png)


If a bigram _σa_ ever occurred previously in the string, nonzero probability is assigned to predicting _a_ when at symbol _σ_ . Then each _NEXTa_ ( _i_ ) predicate can be defined as follows 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0028-09.png)


**----- Start of picture text -----**<br>
NEXTa ( i ) (All Possible) over Σ =  {a, b, c}<br>... (1)<br>NEXTa ( i ) := � [ Qσ ( i )  ∧ EXISTSσa ( i )] (2)<br>σ∈ Σ<br>**----- End of picture text -----**<br>


where[�] _σ∈_ Σ[can be expressed using the Boolean operations] _[ ∧]_[and] _[ ¬]_[ as defined in Section 4.2.][This] corresponds to testing, for the _f_ in Equation 10, for which _σ_ we have _f_ ( _x_ 1 _. . . xN_ ) _N,σ >_ 0. 

**Generating based on this program** Consider an input prefix of the form # _x_ #, where # denotes a separator symbol. If we iteratively generate the next symbol _a_ by selecting _a ∈_ Σ where _NEXTa_ holds at the last position, we generate a string # _x_ # _y_ where all bigrams in # _y_ had already occurred in # _x_ , a simple version of the in-context Markov chains studied by (Edelman et al., 2024). 

**Special Case: Unique Copying** In the special case of an input prefix where each symbol occurs at most once in _x_ , the generation procedure defined above will copy _x_ , and (assume we stop at #) resulting in an overall string of the form # _x_ # _x_ #. This is essentially the RASP-L construction of unique copying noted by Zhou et al. (2024a). 

**Necessity of Positional Relations** Intuitively, an induction head circuit requires positional information; indeed, we observe length generalization in unique copying with APE but not with NoPE (Figure 1). Formally, we can prove as follows that the predicate _NEXTa_ defined above for each _a ∈_ Σ, while definable in **C-RASP** [local], is not definable in **C-RASP** [ _∅_ ]. Consider Σ = _{a, b}_ ; then the predicate _NEXTb_ can be used to define the (disjoint) union of the languages Σ _[∗] ab_ Σ _[∗] a_ , Σ _[∗] bb_ Σ _[∗] b_ . As the first one is definable in **C-RASP** [ _∅_ ][4] and the union is disjoint, the second would be definable if _NEXTb_ is. This contradicts the fact that Σ _[∗] bb_ Σ _[∗] ̸∈_ **C-RASP** [ _∅_ ], because 

> 4It is sufficient to check whether _a_ and _b_ both are present and whether one _b_ has a _a_ in its preceding context; as Σ = _{a, b}_ , this is equivalent to _ab_ being a substring 

28 

Preprint 

Σ _[∗] bb_ Σ _[∗] ̸∈ MAJ_[�] 2[ _<_ ] (Lemma 6.11 in Krebs (2008)) and the inclusion **C-RASP** [ _∅_ ] _⊆ MAJ_[�] 2[ _<_ ] (see Section C.3.1). 

C.2.3 ( _aa_ ) _[∗]_ 

The following function that checks the parity of a position mod 2 is a periodic function. 

_ϕ_ ( _i_ ) := _i ≡_ 0 mod 2 

So the following program recognizes ( _aa_ ) _[∗]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0029-06.png)


**----- Start of picture text -----**<br>
( aa ) [∗]<br>C¬a ( i ) := #  [ j ≤ i ] ¬Qa ( j ) (1)<br>A ( i ) :=  C¬a ( i ) = 0 (2)<br>D ( i ) :=  ϕ ( i )  ∧ A ( i ) (3)<br>**----- End of picture text -----**<br>


The Boolean value of the last operation in the last position of the string is the accepting value. This is true if the string is of even length and contains only _a_ ’s. Overall, we have constructed a program in **C-RASP** [periodic _,_ local]. 

- C.3 EXPRESSIBILITY OF REGULAR LANGUAGES IN **C-RASP** [PERIODIC,LOCAL] 

**Lemma 29** (Restated from Lemma 11) **.** _Consider the alphabet_ Σ = _{a, b, e}._ 

_1. PARITY_ := _b[∗]_ ( _ab[∗] ab[∗]_ ) _[∗] ̸∈_ _**C-RASP**_ [ _periodic, local_ ] 

_2._ ( _aa_ ) _[∗] ∈_ _**C-RASP**_ [ _periodic, local_ ] _and_ ( _aa_ ) _[∗] ̸∈_ _**C-RASP**_ [ _∅_ ] 

_3._ ( _a|b|e_ ) _[∗] be[∗] ̸∈_ _**C-RASP**_ [ _periodic, local_ ] 

_4. L ∈_ _**C-RASP**_ [ _∅_ ] _for piecewise testable L_ 

_Proof._ 1–3 are shown in Lemmas 36 (for 3.), 38 (for 2.), 41 (for 1.), and Appendix C.2.3 (for 2.). 4. is shown in Appendix C.1.5. 

## C.3.1 LINK TO MAJORITY LOGIC 

In understanding the expressiveness of **C-RASP** , we draw on an established body of work on logics using MAJORITY quantifiers. Merrill and Sabharwal (2023a); Strobl (2023) show that the expressiveness of transformers is upper-bounded by uniform **TC**[0] , which can be defined as the logic **FOM** [BIT]. This logic is defined in terms of MAJORITY quantifiers and various predicates. **CRASP** [periodic,local] can be viewed as a highly restricted fragment of this logic. Specifically, it is contained in _MAJ_[�] 2[ _<,_ +1 _, Mod_ ], which was studied by Krebs (2008); Behle et al. (2007; 2009); results about that logic help understand the expressiveness of **C-RASP** : 

**Definition 30.** _MAJ_[�] 2[ _<,_ +1 _, Mod_ ] _is the logic defined by the constructs_ 

_1. The construct_ 

� _Maj x ⟨ϕ_ 1 _, . . . , ϕc⟩_ 

_2. The predicates Qa_ ( _x_ ) _for a ∈_ Σ 

_3. Numerical predicates for q, j ∈_ N _: Modm,r_ ( _x_ ) _, Succ_ ( _y, x_ ) 

_4. Boolean connectives_ 

29 

Preprint 

_5. First-order quantifiers_[5] 

_such that only two variables (say, x and y) can appear within a formula. We define the semantics, when x ∈_ Σ _[∗] , by defining_ 

_1. for the majority quantifier:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0030-04.png)


_2. for the predicates:_ 

_w|x_ = _i |_ = _Qa_ ( _x_ ) _⇔ wi_ = _a w|x_ = _i |_ = _Modm,r_ ( _i_ ) _⇔ i ≡ r_ (mod _m_ ) _w|x_ = _i,y_ = _j |_ = _Succ_ ( _j, i_ ) _⇔ j_ + 1 = _i_ 

_Semantics of Boolean connectives and first-order quantifiers follow the standard definition._ 

_A language L ⊆_ Σ _[∗] is_ definable _in MAJ_[�] 2[ _<,_ +1 _, Mod_ ] _if there is a formula ϕ without free variables such that w ∈L if and only if w |_ = _ϕ._ 

The logic _MAJ_[�] 2[ _<_ ] results by omitting the numerical predicates defined under (3). 

It is straightforward to convert **C-RASP** programs into formulas of _MAJ_[�] 2[ _<,_ +1 _,_ MOD]. As we shall see later in Section C.3.3, the inclusion is strict because _PARITY_ is expressible even in � _MAJ_ 2[ _<_ ]. 

**Proposition 31.** _**C-RASP**_ [ _periodic, local_ ] _⊆ MAJ_[�] 2[ _<,_ +1 _, MOD_ ] 

We make two remarks about the corollaries of the result: 

**Remark 32.** _First, the proof, simply by omitting the positional relations, also yields a corresponding inclusion without positional relations:_ _**C-RASP**_ [ _∅_ ] _⊆ MAJ_[�] 2[ _<_ ] _._ 

_Second, the result implies that_ _**C-RASP**_ [ _periodic, local_ ] _defines a subclass of_ _**TC**_[0] _, in fact, all_ _**C-RASP**_ [ _periodic, local_ ] _programs translate into uniform_ _**TC**_[0] _circuits with_ a linear number of gates _by results in Krebs (2008, Theorem 4.33 and Figure 4.4). The inclusion is strict, e.g., PARITY has a linear-size_ _**TC**_[0] _circuit but is not definable by_ _**C-RASP**_ [ _periodic, local_ ] _, as we show below._ 

_Proof of Proposition 31._ First, every periodic positional function _ϕ_ ( _i_ ) is a Boolean function _Modm,r_ ( _i_ ) _⇐⇒ i_ = _r_ mod _m_ . For local functions _ψ_ ( _i, j_ ), it suffices to only consider functions of the form _ψ_ ( _i, j_ ) _⇐⇒ j_ = _i_ + _c_ for _c ∈_ Z. This is because the counting operation _C_ ( _i_ ) := **#** [ _j ≤ i, |i − j| ≤ c_ ] _P_ ( _j_ ) is equivalent to _C_[ˆ] ( _i_ ) where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0030-16.png)


And it is possible to further reduce this by distributing the disjunctions over many counting operations so that each one only contains a single disjunct as positional function. It helps that a predicate fst( _i_ ) is definable in _MAJ_[�] 2[ _<,_ +1 _,_ MOD] which is true iff _i_ = 0. For instance fst( _i_ ) := _Maj_[�] _j⟨j ≤ i, ⊤⟩_ 

For each Boolean **C-RASP** operation _P_ ( _i_ ), there exists a _MAJ_[�] 2[ _<,_ +1 _,_ MOD] formula _P_[ˆ] ( _i_ ) with one free variable that is equivalent. By induction, all cases are straightforward except for comparison operations. 

> 5These can be simulated by majority quantifiers with two variables by Proposition 5.5 in Krebs (2008), which is based on Corollary 3.3 in Lange (2004). Nonetheless, as the simulation is unobvious, they are useful for writing formulas in _MAJ_[�] 2[ _<,_ +1 _, MOD_ ]. 

30 

Preprint 

For comparison operations, we will first show a formula that is equivalent for all nonempty strings. Accounting for the empty string is easy, depending on the constants in the comparison. WLOG we are able to rewrite the formula (not in standard **C-RASP** notation) as the following, where _αk, β ∈_ Z 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-02.png)


We’ve grouped the uniform counting operations that have _ψ_ = _⊤_ together. Then using a case disjunction, we can rewrite it all the local counting operations as the following (using I[ _ϕ_ ] as notational convenience to turn _ϕ_ ( _j_ ) _∈{⊥, ⊤}_ to the corresponding value in _{_ 0 _,_ 1 _}_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-04.png)


We can see that for nonempty strings within each case, the additive constant _β_ can be reformulated as � _β_ +[�] _m≤M[α][m][τ][m]_ � _·_ **#** [ _j ≤ i_ ] fst( _j_ ), and we can just add it to the summation using _αk_ +1 = � _β_ +[�] _m≤M[α][m][τ][m]_ �. Then, it is possible to define formulas for each case disjunction using a series of existential quantifiers and _succ_ ( _j, i_ ). For instance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-06.png)


This means it now suffices to focus on the sum of counting terms and simulate that using a _Maj_[�] formula. For _k ≤_ ( _K_ + 1), if _αk >_ 0 consider the list of formulas 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-08.png)


Intuitively, the _Maj_[�] _j_ quantifier can only check if the total count is greater than half the possible positions, so to check if a count is _>_ 0 we need to pad the quantifier with a bunch of trivially true formulas to ensure the total count is at least half by default. And if _αk <_ 0 we use 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-10.png)


Let _L_ = _L_ 1 ++ _L_ 2 ++ _. . ._ ++ _LK_ +1 be the concatenation of all these lists, and let _φ_ 1 _, φ_ 2 _, . . . , φ|L|_ list out the formulas in _L_ . Then we claim the following formula will compute the correct value for nonempty strings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-12.png)


For empty strings, if � _β_ +[�] _m≤M[α][m][τ][m]_ � _>_ 0 then define _ϕ_ 0( _i_ ) := _¬Maj_[�] _j⟨⊤⟩∧⊤_ . Otherwise, use _ϕ_ 0( _i_ ) := _¬Maj_[�] _j⟨⊤⟩∧⊥_ . Then we can define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0031-14.png)


And verifying the correctness of this is straightforward. 

31 

Preprint 

## C.3.2 INEXPRESSIBILITY OF Σ _[∗] be[∗]_ 

Krebs (2008); Behle et al. (2007; 2009) used infinite groups to establish results about the expressiveness of _MAJ_[�] 2[ _<_ ]; by Corollary 31, these results entail results on **C-RASP** [periodic _,_ local]. In particular, Lemma 6.11 in Krebs (2008) shows that _Lbb ̸∈ MAJ_[�] 2[ _<_ ]; this result turns out to have profound consequences for **C-RASP** expressiveness. 

**Definition 33.** _Let_ Σ = _{a, b, e}. Define Lbb_ := Σ _[∗] be[∗] b_ Σ _[∗] ._ 

**Lemma 34.** _Let φ be a MAJ_[�] 2[ _<,_ +1 _, MOD_ ] _formula. There exists a morphism h_ ( _σ_ ) = _e[s] σe[s][−]_[1] _and a MAJ_[�] 2[ _<_ ] _formula ψ such that for every w ∈_ Σ _[∗] , h_ ( _w_ ) ⊨ _ϕ ⇐⇒ w_ ⊨ _ψ_ 

_Proof._ Let _M_ be the least common multiple of all occurring moduli in _φ_ . Let _C_ be the maximum nesting depth of _Succ_ ( _x, y_ ) predicates (which must be bounded by the quantifier depth of _φ_ ). Intuitively, we can think of _C_ as the largest number where a subformula _φ_ ( _x_ + _C_ ) occurs in _φ_ . Let _s_ = _MC_ , and define the morphism _h_ ( _σ_ ) = _e[s] σe[s][−]_[1] . Here we will use the notation _ϕ[c]_ ( _x_ ) that is true at position _x_ in _w_ whenever _ϕ_ is true at position _x_ + _c_ in _h_ ( _w_ ), for _c ∈_ [ _−s, s −_ 1]. 

We will show that for every formula _φ_ ( _x_ ) of _MAJ_[�] 2[ _<,_ +1 _, MOD_ ] with at most one free variable, we can define _φ[−][s]_ ( _x_ ) _, φ[−]_[(] _[s][−]_[1)] ( _x_ ) _. . . φ_[(] _[s][−]_[1)] ( _x_ ) such that for all _i ∈_ [0 _, |w|−_ 1] and _c ∈_ [ _−s, s −_ 1] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0032-07.png)


Intuitively, what this does is it takes every interval of [ _x − s, x_ + ( _s −_ 1)] around each position in _h_ ( _w_ ) and stores it vertically at that position in _w_ . We will induct on the complexity of _φ_ . If _φ_ ( _x_ ) is _Qe_ ( _x_ ), then _φ_[0] ( _x_ ) := _Qe_ ( _x_ ), and then _φ[c]_ ( _x_ ) = _⊤_ for every other _c_ = 0, since the morphism _h_ pads neutral symbols _e_ in _h_ ( _w_ ) between every symbol from _w_ . If _φ_ ( _x_ ) is _Qσ_ ( _x_ ) for _σ_ = _e_ , we have that _φ_[0] := _Qσ_ ( _x_ ) and _φ_[+] _[c]_ ( _x_ ) = _⊥_ for every other _c_ = 0. If _φ_ ( _x_ ) is _Modm,r_ ( _x_ ), the _φ[c]_ can also be “hardcoded” similarly, as every position in _h_ ( _w_ ) that has a symbol from _w_ is going to be = 0 mod _s_ . 

Boolean formulas are also straightforward. The only hard case is if we have a formula _φ_ ( _x_ ) = � _MAJ y⟨φ_ 1( _x, y_ ) _, . . . , φk_ ( _x, y_ ) _⟩_ . We can think of _φ_ specifying the constraint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0032-10.png)


The idea here is to rewrite _ψi_ ( _x, y_ ) in terms of its unary formulas (which we can apply the inductive hypothesis to) and then split _h_ ( _w_ ) into some intervals, upon which evaluating _φ_[+] _[c]_ ( _x_ ) will be simpler. First we can rewrite each _φi_ ( _x, y_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0032-12.png)


Where _Fi_ is a Boolean function, the _α_ are unary in _x_ , the _β_ are unary in _y_ , and the _χ_ ( _x, y_ ) are inequalities of _x_ and _y_ , possibly with +1’s, of the form _x ≤ y_ + 1, for example. To save space, we will abbreviate the above expression by grouping the _α, β, χ_ formulas together notationally _Fi_ ( _χi_ ( _x, y_ ) _, αi_ ( _x_ ) _, βi_ ( _y_ )). 

The _χ_ formulas are not unary, but we can “eliminate” the _χ_ terms by casework over intervals of the string. We will show this by example for a summation with only one # _y_ term. This argument works identically if we had many of # _y_ terms, but it would add notational clutter. So if we had a formula _φ_ ( _x_ ) = _MAJ_[�] _y⟨φ_ 1( _x, y_ ) _⟩_ we could think of it as in the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0032-15.png)


Then we can construct the formula _φ[c]_ ( _x_ ) for _c ∈_ [ _−s, s −_ 1] by using the following partition of intervals of the string. Let Ξ be the set of inequalities 

32 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0033-01.png)


And we define some notation. For _ξ ∈_ Ξ, let _χ[ξ]_ ( _x, y_ ) _∈{⊤, ⊥}_ evaluate _χ_ in the case _ξ_ holds. For instance if _χ_ ( _x, y_ ) is _y < x_ + 1, then _χ[y>x]_[+2] ( _x, y_ ) = _⊥_ . Since the intervals defined by _ξ ∈_ Ξ disjoint and cover the entirety of the string, every _χ_ can be evaluated in this manner. Then, we can essentially compute the sum in each interval, and only precision is needed in the interval [ _x − s, x_ + ( _s −_ 1)], so _φ_ ( _x_ ) is equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0033-03.png)


By the inductive hypothesis, all _α[c]_ and _β[c]_ are definable solely in terms of _MAJ_[�] 2[ _<_ ], so the entire formula is equivalent to a _MAJ_[�] _y_ formula that quantifies over all the bracketed formulas above, as well as equally many trivially true formulas, as described more clearly in Proposition C.3.1. As mentioned before, since this argument also applies to a summation of # _y_ terms, this completes the proof. Then for any _φ_ ( _x_ ) in _MAJ_[�] 2[ _<,_ +1 _, MOD_ ], after performing the above translation the resulting formula _φ_[0] ( _x_ ) is our desired formula in _MAJ_[�] 2[ _<_ ]. 

**Lemma 35.** _Lbb ̸∈ MAJ_[�] 2[ _<,_ +1 _, MOD_ ] 

_Proof._ Assume for sake of contradiction that _Lbb_ is definable by a formula _φ_ of _MAJ_[�] 2[ _< ,_ +1 _, MOD_ ]. Let _h_ and _ψ_ be as guaranteed by the above lemma. Then for _w ∈_ Σ _[∗]_ , _w ∈ Lbb ⇐⇒ h_ ( _w_ ) _∈ Lbb_ . This means _ψ_ defines _Lbb_ which contradicts Lemma 6.11 in Krebs (2008), which has shown that _Lbb ̸∈ MAJ_[�] 2[ _<_ ]. 

33 

Preprint 

**Lemma 36.** _For_ Σ = _{a, b, e}, it holds that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0034-02.png)


_Proof._ To get a contradiction, note that a **C-RASP** [periodic _,_ local] program Φ for Σ _[∗] be[∗]_ could be used to construct one for _Lbb_ , as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0034-04.png)


## C.3.3 INEXPRESSIBILITY OF PARITY 

First, let the _depth_ of a **C-RASP** operation be the maximum depth of nesting of counting operations in it. For instance if _C_ ( _i_ ) := **#** [ _j ≤ i_ ] _P_ ( _j_ ), the depth of the _C_ is depth of _P_ ( _i_ ) plus one. None of the other operations are greater than the depth of its dependencies. We will induct on program depth for the following proof: 

**Lemma 37.** _Let_ Σ = _{a}. For any_ _**C-RASP**_ [ _∅_ ] _program P there exists an n such that for all w where |w| ≥ n, either all such w are accepted by P or all are rejected_ 

_Proof._ If _P_ is depth 0, it is equivalent to either _Qa_ ( _i_ ) or _¬Qa_ ( _i_ ), which either rejects every string or accepts every string. 

Otherwise, let all **C-RASP** programs of depth _k_ give constant output for strings above length _n_ , and then consider a program _P_ of depth _k_ + 1. _P_ will be equivalent to a Boolean combination of linear constraints. We will see that each linear constraint becomes constant for strings above a certain length. Consider any linear constraints over _X_ many counts _Cx_ of depth _k_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0034-10.png)


For string of length _i ≥ n_ , this is equivalent to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0034-12.png)


Where I[ _Cx_ ( _n_ )] denotes the truth value of _Cx_ ( _n_ ) _∈{_ 0 _,_ 1 _}_ . Rearrange this to 

The sums _c_ 1 =[�] _x≤X[α][x]_[ (][I][[] _[C][x]_[(] _[n]_[)])][ and] _[ c]_[2][=][ �] _x≤X[α][x]_[ (] **[#]**[ [] _[j][≤][n]_[]] _[C][x]_[(] _[j]_[))][ are constants depend-] ing on the formula and _n_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0034-15.png)


Depending on if _c_ 1 _, c_ 2 are positive or negative, we either derive a lower bound _m_ after which the linear constraint _L_ ( _i_ ) is always true, or always false for _i ≥ m_ . Since any formula of depth _k_ + 1 is a Boolean combination of these linear constraints, we take the max of all the _m_ ’s from them, and any string larger than this will always be accepted or rejected by _P_ . 

34 

Preprint 

## **Lemma 38.** ( _aa_ ) _[∗] ̸∈_ _**C-RASP**_ [ _∅_ ] 

That is, no **C-RASP** [ _∅_ ] program can determine if a general string has an even length. The same proof applies to testing whether the string length is a multiple of any other fixed integer. 

_Proof._ Using the previous lemma, for every **C-RASP** [ _∅_ ] program there exists an _n_ such that the program accepts ( _aa_ ) _[n]_ iff it accepts ( _aa_ ) _[n] a_ . So no program can recognize ( _aa_ ) _[∗]_ . 

We will use this to show that no **C-RASP** [periodic _,_ local] program cannot recognize _PARITY_ , as the extra positional operations do not give sufficient expressive power. We start with an observation that simplifies the proof 

**Proposition 39.** _As syntactic sugar we allow j < i as a mask in_ _**C-RASP** counting operations._ 

_Proof._ Consider the counting operation _C_ ( _i_ ) := **#** [ _j ≤ i_ ] _P_ ( _j_ ). We can define the program 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0035-07.png)


And essentially, this operation will compute the count 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0035-09.png)


**Lemma 40.** _Let P be a_ _**C-RASP**_ [ _periodic, local_ ] _program over_ Σ = _{a, b}. There is some s >_ 0 _and a morphism h_ ( _a_ ) = _b[s] ab[s][−]_[1] _and a_ _**C-RASP**_ [ _∅_ ] _program P_[ˆ] _over_ Σ = _{a} such that for all w ∈ a[∗] , if P accepts h_ ( _w_ ) _iff P_[ˆ] _accepts w._ 

_Proof._ Choose _s_ to be the least multiple of all moduli occurring in _P_ that is also greater than all the _|c|_ in local functions _j_ = _i_ + _c_ . For every operation _P_ ( _i_ ) of _P_ , we will define _P_[ˆ] _[c]_ ( _i_ ) for _c ∈_ [ _−s, s−_ 1] such that _P_[ˆ] _[c]_ ( _i_ ) when run on _w_ is equivalent to _P_ ( _s_ + _i_ (2 _s_ ) + _c_ ) when run on _h_ ( _w_ ). 

If _P_ ( _i_ ) is _Qa_ ( _i_ ) or _Qb_ ( _i_ ), it is straightforward, as _P_[ˆ] _[c]_ ( _i_ ) is true iff _c_ = 0. Modular predicates are also capable of being “hardcoded”, as positions in _w_ are always 0 mod _s_ in _h_ ( _w_ ). All other kinds of operations are also straightforward using the inductive hypothesis. The only ones that need care are counting operations. First, consider a counting operation without positional functions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0035-13.png)


We can define each _C_[ˆ] _[c]_ ( _i_ ) using a program like the following. The idea is that the entire window of [ _j − s, j_ + ( _s −_ 1)] around each _j < i_ can be counted up completely, but around _i_ we only consider 

35 

Preprint 

the interval [ _−s, i_ + _c_ ]: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0036-02.png)


Otherwise, if we have a counting operation that involves a local positional function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0036-04.png)


Then the operation returns either the count 1 or 0 and we can just use 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0036-06.png)


Since _d_ will not exceed _±s_ , _A_[ˆ] _[d]_ exists. Using these constructions we can see that ( _b[s] ab[s][−]_[1] ) _[n]_ is accepted by operation _P_ ( _i_ ) in _P_ iff _a[n]_ is accepted by the constructed operation _P_[ˆ][(] _[s][−]_[1)] ( _i_ ). 

## **Lemma 41.** _PARITY ̸∈_ _**C-RASP**_ [ _periodic, local_ ] 

_Proof._ If such a program existed, it would be able to distinguish between ( _b[s] ab[s][−]_[1] )[2] _[n]_ and ( _b[s] ab[s][−]_[1] )[2] _[n]_[+1] for all _n_ (using the _s_ guaranteed by the previous lemma). However, this implies the existence of a **C-RASP** [ _∅_ ] program over Σ = _{a}_ that recognizes ( _aa_ ) _[∗]_ . This contradicts Lemma 38. 

## D DISCUSSION OF DESIGN CHOICES 

## D.1 MLP ACTIVATION FUNCTIONS 

Our analysis allows ReLU and the Heaviside function as activation functions in MLPs. ReLU is a standard choice in theoretical studies of neural networks and transformers (e.g. Bhattamishra et al., 2024; Sanford et al., 2023). Modern LLMs also use other functions such as SwiGLU (Shazeer, 2020), but universal approximation theorems guarantee that ReLU networks can approximate smooth functions on bounded domains well. While the choice of ReLU is not necessarily key to our results, it is important that the number of active units provides a meaningful upper bound on the complexity of the function expressed. Our results would continue to go through if _ϕ_ is an arbitrary activation function but operates at _p_ -bit precision. 

We also allow the Heaviside function as a second activation function. Heaviside allows exactly performing threshold computations at arbitrary input lengths, which is relevant to simulating **C-RASP** at arbitrary input lengths. This includes simple problems such as MAJORITY, on which transformers empirically do well. Real-world transformers generally do not include this function, though ReLU MLPs can approximate it arbitrarily closely. Theoretically modeling length generalization on such functions with only ReLU MLPs is an interesting possible extension of our theory. 

36 

Preprint 

## D.2 FIXED PRECISION 

As described in Section 2, we assume that attention logits and the exponentials inside softmax are rounded to _p_ fractional bits of precision before further processing. This allows us to cluster keys and queries into finite numbers of clusters, and compute all activations at logarithmic precision, used for proving the logarithmic communication complexity bound (Theorem 12). We note that logarithmic precision of the intermediate activations is also key to upper bounds of transformers in terms of **TC**[0] shown by Merrill and Sabharwal (2023b). 

We also assume that the parameters in transformers and Limit Transformers are expressed at fixed precision (Definitions 4 and 2), and penalize the precision _p_ used of representing the parameters as part of the regularizer used in our inference procedure (Definition 5). Indeed, penalizing unbounded precision of parameter values is necessary to enable full identification of a transformer algorithm from behavior at finite lengths. For any real number _α ≥_ 0, a one-layer transformer with real-valued parameters can express the function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0037-04.png)


For any two distinct _α, β_ , the functions _Fα_ and _Fβ_ are distinct on sufficiently long inputs (though, when _α_ and _β_ are close, very long inputs will be needed to distinguish them). Thus, there are _uncountably_ many distinct functions implemented by transformers; however, their distinction relies on infinite precision. In an infinite precision setup, one cannot hope to identify algorithms implemented by transformers from finite data, no matter the input length and the regularization applied to the model size. In contrast, when parameters are representable in finite precision (as in real computers), the number of distinct algorithms expressed by Limit Transformers is countable, and ultimate identification from long inputs is possible when the precision required for representing the parameters is penalized. 

## D.3 LAYER NORM 

Real-world transformers use Layer Norm or RMSNorm, whereby activations _**y** i_[(] _[l]_[)] are rescaled to have norm or standard deviation _√d_ . Layer norm can be incorporated into the translation to Limit Transformers (Lemma 52) by recording terms of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0037-08.png)


when _**v** ∈VOl_ 1 and _**w** ∈VOl_ 2 and _S_ 1 _, S_ 2 _∈P_ . We can record these products in further dimensions of _**y**_[�] _i_[(] _[l]_[)][, so that] _[ ∥]_ _**[y]** i_[(] _[l]_[)] _[∥]_ 2[2][is recoverable from] _**y**_[�] _i_[(] _[l]_[)][.][The simplest approach is then to modify the] definition of Limit Transformers by normalizing _**y**_[�] _i_[(] _[l]_[)] based on this recovered norm. 

## E ADDITIONAL DETAILS FOR EXPERIMENTS 

E.1 REGULAR LANGUAGES FROM THE BHATTAMISHRA ET AL 2020 BENCHMARK 

E.1.1 LANGUAGE DEFINITIONS 

Descriptions follow Bhattamishra et al. (2020). 

**Tomita Grammars.** Definitions are shown in Table 1. 

_**Dn**_ are defined on the alphabet Σ = _{a, b}_ by the recursion _Dn_ = ( _aDn−_ 1 _b_ ) _[∗]_ . 

**PARITY.** PARITY is _b[∗]_ ( _ab[∗] ab[∗]_ ) _[∗]_ . It is contained in the set of algorithmic tasks. 

**Others.** Other languages: ( _aa_ ) _[∗]_ , ( _aaaa_ ) _[∗]_ and ( _abab_ ) _[∗]_ (not star-free), _aa[∗] bb[∗] cc[∗] dd[∗] ee[∗]_ , _{ab}[∗] d{b, c}[∗]_ , and _{_ 0 _,_ 1 _,_ 2 _}[∗]_ 02 _[∗]_ (star-free). 

37 

Preprint 

|Grammar|Star-Free|Defnition|
|---|---|---|
|1<br>2<br>3<br>4<br>5<br>6<br>7|Yes<br>Yes<br>No<br>Yes<br>No<br>No<br>Yes|1*<br>(10)*<br>strings without odd-length strings of ones followed by odd-<br>length strings of zeros (i.e., no012_n_+102_m_+11substrings)<br>strings without any 000’s substrings<br>strings of even length with an even number of 1’s<br>strings where number of 0’s - number of 1’s is divisible by 3<br>0*1*0*1|



Table 1: Tomita Grammars (originally due to Tomita, 1982), following Bhattamishra et al. (2020). 

|#<br>Language|**C-RASP**expressiveness<br>[_∅_]<br>[periodic,local]|Star-Free?<br>Dot-Depth<br>**AC**0?|
|---|---|---|
|1<br>Tomita 1<br>2<br>Tomita 2<br>3<br>Tomita 3<br>4<br>Tomita 4<br>5<br>Tomita 5<br>6<br>Tomita 6<br>7<br>Tomita 7<br>8<br>_D_2<br>9<br>_D_3<br>10<br>_D_4<br>11<br>_D_12<br>–<br>PARITY<br>12<br>(_aa_)_∗_<br>13<br>(_aaaa_)_∗_<br>14<br>(_abab_)_∗_<br>15<br>_aa∗bb∗cc∗dd∗ee∗_<br>16<br>_{a, b}∗d{b, c}∗_<br>17<br>_{_0_,_1_,_2_}∗_02_∗_|yes<br>yes<br>yes<br>yes<br>no<br>no<br>no<br>yes<br>no<br>no<br>no<br>no<br>yes<br>yes<br>yes<br>yes<br>yes<br>yes<br>yes<br>yes<br>yes<br>yes<br>no<br>no<br>no<br>yes<br>no<br>yes<br>no<br>yes<br>yes<br>yes<br>yes<br>yes<br>no<br>no|yes<br>1<br>yes<br>yes<br>1<br>yes<br>no<br>–<br>yes<br>yes<br>1<br>yes<br>no<br>–<br>no<br>no<br>–<br>no<br>yes<br>1<br>yes<br>yes<br>2<br>yes<br>yes<br>3<br>yes<br>yes<br>4<br>yes<br>yes<br>12<br>yes<br>no<br>–<br>no<br>no<br>–<br>yes<br>no<br>–<br>yes<br>no<br>–<br>yes<br>yes<br>1<br>yes<br>yes<br>1<br>yes<br>yes<br>2<br>yes|



Table 2: The finite-state languages in the benchmark from Bhattamishra et al. (2020), with the numbering from Figure 1, **C-RASP** expressiveness properties, and three established notions of complexity (star-freeness, dot depth, and membership in the circuit complexity class **AC**[0] ). In the **C-RASP** columns, “yes” means we found a **C-RASP** program; “no” means we proved that no **C-RASP** program can exist. Note that _{_ 0 _,_ 1 _,_ 2 _}[∗]_ 02 _[∗]_ is equivalent to the language Σ _[∗] be[∗]_ from Lemma 11. Note also that we discuss PARITY in the algorithmic benchmark, as it is included in Zhou et al. (2024a). See discussion in Appendix E.1.2. See Figure 2 for a version of Figure 1 (right) with languages labeled. 

## E.1.2 **C-RASP** EXPRESSIVENESS 

All languages in the benchmark are in **TC**[0] , and all are expressible in principle by transformers (Liu et al., 2023). We were able to provably settle the **C-RASP** expressiveness for all languages, with results shown in Table 2. We compare **C-RASP** expressiveness with the standard notions of complexity of finite-state languages considered by Bhattamishra et al. (2020): whether languages are star-free, and (among the star-free ones) their dot-depth. While many star-free languages in the sample show length-generalization, star-freeness does not overall account for the observed behavior (Figure 4). Within the star-free languages, a standard complexity metric is dot-depth (Figure 2); this again does not accurately predict length-generalization: it succeeds for a language with dot depth 12 but fails for a language with dot depth 2. We also considered the circuit complexity of regular languages (Barrington et al., 1992). All regular languages included in the sample are in the class **TC**[0] ; most are also in **AC**[0] , a smaller class sometimes compared to transformers (Hao et al., 2022; Barcelo et al., 2024). Transformers show poor length generalization on the non- **AC**[0] regular lan- 

38 

Preprint 

guages[6] , but also fail on various languages that are in **AC**[0] . On algorithmic problems, transformers succeed on some non- **AC**[0] problems such as Majority (Table 3). Overall, **C-RASP** expressiveness is much more successful than previously considered notions of complexity in accounting for empirical length generalization behavior of transformers. 

**Proof Sketches for Membership Claims** We sketch proofs for all **C-RASP** expressiveness claims in Table 2. We first note that **C-RASP** [periodic _,_ local] is closed under the inverse images of morphisms where each symbol is mapped to a string of the same length. That is: 

**Lemma 42.** _**C-RASP**_ [ _periodic, local_ ] _is closed under the inverse images of morphisms where each symbol is mapped to a string of the same length. That is, if h_ : Σ1 _→_ Σ _[n]_ 2 _[(for][some] fixed n) is extended to a map_ Σ _[∗]_ 2 _[→]_[Σ] _[∗]_ 1 _[and][L][∈]_ _**[C-RASP]**_[[] _[periodic][,][ local]_[]] _[,][then][h][−]_[1][(] _[L]_[)] _[∈]_ _**C-RASP**_ [ _periodic, local_ ] _._ 

_Proof._ Let _h_ be such a morphism. We we take any **C-RASP** [periodic _,_ local] program operation _P_ ( _i_ ) that operates on _h_ ( _w_ ) into an equivalent program which operates on _w_ 

For _P_ ( _i_ ) = _Qa_ ( _i_ ), we can use _P_[ˆ][0] ( _i_ ) := _Qh_ ( _a_ )0( _i_ ) _, P_[ˆ][1] ( _i_ ) := _Qh_ ( _a_ )1( _i_ ), and so on. 

For _P_ ( _i_ ) = _Modm,r_ ( _i_ ), then for _c ≤ n_ we can define (relying on the fact that _m, n_ are fixed) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0039-07.png)


For _P_ ( _i_ ) = **#** [ _j ≤ i, ψ_ ( _i, j_ )] _V_ ( _j_ ), where _ψ_ ( _i, j_ ) = _⊤_ , then like before we can define _P_[ˆ] _[c]_ ( _i_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0039-09.png)


If _ψ_ ( _i, j_ ) := _j_ = _i − d_ , then it takes a little bit more care, as the single position _j_ upon which to check _V_ ( _j_ ) may occur far behind the current position. First, via a similar argument to Proposition C.3.3, it is possible to simulate the counting operation **#** [ _j ≤ i − k, ψ_ ( _i, j_ )] _V_ ( _j_ ) for any constant _c ∈_ N. We can do this by counting **#** [ _j ≤ i_ ] _j_ = _i − k[′] ∧ ψ_ ( _i, j_ ) _V_ ( _j_ ) for each _k[′] ≤ k_ , since _j_ = _i − k[′] ∧ ψ_ ( _i, j_ ) remains a local function, and then subtracting that from the original count over 

> 6By the results of Barrington et al. (1992), regular languages outside of **AC** 0 are all, informally speaking, at least as PARITY, and indeed they provably are not in **C-RASP** [periodic _,_ local]. 

39 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0040-01.png)


For _c > d_ , the position to check occurs beyond _i_ , so we use: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0040-03.png)


All other cases are straightforward, and it can be verified that these constructions are correct. 

**Tomita 1** _∈_ **C-RASP** [ _∅_ ] A **C-RASP** program can detect the presence of a symbol other than 1 and flag a violation. 

**Tomita 2** _∈_ **C-RASP** [ _∅_ ] A **C-RASP** program expresses: At each position, either the current symbol is a 0 and the count of ones and zeros is balanced, or the current symbol is a 1 and the count of ones is one more than the count of zeros. 

**Tomita 3** _̸∈_ **C-RASP** [ _∅_ ] For a given string of the form 010 _[K]_ , the outputs will converge as _K →∞_ by the same argument as for ( _aa_ ) _[∗]_ in Lemma 38. Hence, Tomita 3 _̸∈_ **C-RASP** [ _∅_ ]. 

**Tomita 3** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] Informally, the only way periodic and local predicates are likely to help is if the lengths of contiguous blocks of zeros and ones were bounded (local), or the parity of the lengths of 1 and 0 substrings were globally linked to the parity of the positions of transitions 10 and 01 (periodic), but neither is the case. Sketching a formal proof, assume a **C-RASP** [periodic _,_ local] program is given for Tomita 3. First, we eliminate the periodic predicates by labeling every symbol with the position modulo _s_ , where _s_ is a multiple of 2 and all moduli appearing in periodic functions; giving an extended alphabet 11 _, . . . ,_ 1 _s_ ; 01 _, . . . ,_ 0 _s_ . For sufficiently large _c_ that is a co-prime with _s_ , we can then also eliminate local functions by merging an adjacent block of length _c_ around every transition between ones and zeros into a single symbol Λ; indexed by the first symbol inside the block and whether the transition happens at the _floor_ ( _s/_ 2)-th (second part has even length) or _ceil_ ( _s/_ 2)-th (second part has odd length) position in the block. The resulting language, over an extended alphabet, is recognized by a **C-RASP** [ _∅_ ] program capable of determining whether a string of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0040-09.png)


contains a substring of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0040-11.png)


This is impossible for the same reasons that Σ _[∗] bbb_ Σ _[∗]_ (Tomita-4) is not in **C-RASP** [ _∅_ ].[7] 

> 7Formally, Theorems 6.10 and 6.12 in Krebs (2008) show that the regular languages in _MAJ_ � 2[ _<_ ] are contained in **DA** _∗_ **G** . The second component in this product can capture the first subscript of each Λ, but not 

40 

Preprint 

**Tomita 4** _̸∈_ **C-RASP** [ _∅_ ] By Lemma 6.11 in Krebs (2008) (discussed in Appendix C.3.2), Σ _[∗] bb_ Σ _[∗] ̸∈ MAJ_[�] 2[ _<_ ]. In analogy, Σ _[∗] bbb_ Σ _[∗] ̸∈ MAJ_[�] 2[ _<_ ]. 

**Tomita 4** _∈_ **C-RASP** [ **periodic** _,_ **local** ] A **C-RASP** program tests whether there is a position with a 0 where the preceding position also holds a 0 and the position preceding that also holds a 0. 

**Tomita 5** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] This language is the intersection of PARITY with the strings of even length. **C-RASP** [periodic _,_ local] inexpressiveness follows from the same arguments as for PARITY (Lemma 41). 

**Tomita 6** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] Consider first the language _L_ 3 where the number of 1’s is divisible by 3. This is not in **C-RASP** [periodic _,_ local] in analogy to PARITY. Now consider the length-preserving morphism _h_ (1) = 001, _h_ (0) = 000. Then _h_ ( _w_ ) _∈LT omita_ 6 _⇔ w ∈L_ 3. 

**Tomita 7** _∈_ **C-RASP** [ _∅_ ] Tomita 7 is equivalent to _{ϵ} ∪ a_[+] _∪ b_[+] _∪ a_[+] _b_[+] _∪ b_[+] _a_[+] _∪ a_[+] _b_[+] _a_[+] _∪ a_[+] _b_[+] _a_[+] _b_[+] . It can be shown this is equivalent to Σ _[∗] \_ Σ _[∗] b_ Σ _[∗] a_ Σ _[∗] b_ Σ _[∗] a_ Σ _[∗]_ , and this was constructed in section C.1.5. Interestingly, directly implementing this **C-RASP** [ _∅_ ] construction in a transformer appears to require at least four layers; this is in contrast to the **C-RASP** [periodic _,_ local] construction discussed next, where two layers are sufficient.[8] 

**Tomita 7** _∈_ **C-RASP** [ **periodic** _,_ **local** ] A **C-RASP** program can count the number of positions with the bigrams 01, 10 to detect a violation. 

_Dn ∈_ **C-RASP** [ _∅_ ] Similar to Tomita 2. 

**PARITY** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] See Lemma 41. 

( _aa_ ) _[∗] ̸∈_ **C-RASP** [ _∅_ ] See Lemma 38. 

( _aa_ ) _[∗] ∈_ **C-RASP** [ **periodic** _,_ **local** ] See Example C.2.3. 

( _aaaa_ ) _[∗] ̸∈_ **C-RASP** [ _∅_ ] _, ∈_ **C-RASP** [ **periodic** _,_ **local** ] Analogous to ( _aa_ ) _[∗]_ . 

( _abab_ ) _[∗] ̸∈_ **C-RASP** [ _∅_ ] _, ∈_ **C-RASP** [ **periodic** _,_ **local** ] Analogous to ( _aa_ ) _[∗]_ . 

_aa[∗] bb[∗] cc[∗] dd[∗] ee[∗] ∈_ **C-RASP** [ _∅_ ] A **C-RASP** program indicates, first, the presence of “a”, “b”, “c“, “d”, “e”, and, second, that every “a” is preceded only by “a”; every “b” is preceded only by “a” or “b”; every “c” is preceded only by “a”, “b”, “c”; and analogously for “d”, “e”. 

_{a, b}[∗] d{b, c}[∗] ∈_ **C-RASP** [ _∅_ ] There is a single _d_ ; _a_ can only appear before it; _c_ can only appear after it. 

_{_ 0 _,_ 1 _,_ 2 _}[∗]_ 02 _[∗] ̸∈_ **C-RASP** [ **periodic** _,_ **local** ] See Lemma 36. Note that this is equivalent to the language Σ _[∗] be[∗]_ over the alphabet Σ = _{a, b, e}_ . 

E.2 ALGORITHMIC TASKS 

E.2.1 TASK DEFINITIONS FOR ALGORITHMIC PROBLEMS 

The tasks are generally from Zhou et al. (2024a), except for Binary Majority Interleave. Here, we define each formally. 

> the second. Since the language Σ _[∗] aaa_ Σ _[∗] ∪_ Σ _[∗] bbb_ Σ _[∗]_ over Σ = _{a, b}_ is not in **DA** (shown, e.g., via Theorem 2c in Tesson and Th´erien (2002), which would entail syntactic congruence of ( _aabb_ ) _[ω] bb_ ( _aabb_ ) _[ω]_ and ( _aabb_ ) _[ω]_ ), the claimed **C-RASP** [ _∅_ ] program cannot exist. 

> 8The first layer collects bigrams, the second layer compares the count of each bigram to the count of SOS tokens (which is known to be one) to count the bigrams. 

41 

Preprint 

**Binary Majority.** The binary majority problem identifies the most frequent bit in a sequence of random bits. An example is SOS 0 1 ... 0 SEP 1 EOS The part 0 1 ... 0 is the sequence of random bits. We define LEN to be the length of this part. We constrain the sequences such that the number of 0s and 1s are always not equal. The model is trained with the language modeling loss on the part 1 EOS , in other words, it is only trained to predict the most frequent bit and EOS token. The minimum length _lmin_ of this task is 1. 

**Binary Majority Interleave.** The sequences in this problem are created by interleaving multiple binary majority (see above) inputs while avoiding repeating special tokens (e.g., SOS). We use 3 binary majority sequences to compose one sequence in this task. Formally speaking, given 3 binary sequences of the same length, _x_[1] 1 _[,][ · · ·][ x]_[1] _n_[,] _[x]_[2] 1 _[,][ · · ·][ x]_[2] _n_[,][and] _[ x]_[3] 1 _[,][ · · ·][ x]_[3] _n_[,][and their corresponding labels] (most frequent bits) _y_[1] , _y_[2] , _y_[3] , the interleaved input is SOS _x_[1] 1 _[, x]_ 1[2] _[, x]_ 1[3] _[, x]_ 2[1] _[, x]_ 2[2] _[, x]_ 2[3] _[,][ · · ·][, x]_[1] _n[, x] n_[2] _[, x] n_[3][.] SEP _y_[1] , _y_[2] , _y_[3] EOS. 

An example is SOS 1 0 1 1 0 0 ... SEP 1 0 0 EOS LEN in this problem refers to length between SOS and SEP (excluding). The model is trained with the language modeling loss on the part 1 0 0 EOS and _lmin_ = 3. 

**Majority.** This problem is similar to the binary majority problem except that the vocabulary is bigger. An example is SOS c b a b SEP b EOS , where c b a b is a sequence of random tokens, each of which is sampled independently from an alphabet of 26 symbols. The LEN is defined as the length of this part. We constrain the sequences such that there is always a unique answer. The model is trained with the language modeling loss on the part b EOS . _lmin_ = 1. 

**Sort.** In sort problem, the model outputs a sorted version of the given sequence. An example is SOS 14 23 6 9 SEP 6 9 14 23 EOS where 14 23 6 9 is a sequence of unique numbers. LEN in this problem refers to the length of this part. The model is trained with the language modeling loss on the part 6 9 14 23 EOS . In this problem, _lmin_ = 1. The total vocabulary size of tokens except for special tokens is equal to the maximum testing length, i.e., 150. 

**Copy (unique).** In this problem, the model outputs the same sequence as the given sequence, which consists of unique tokens. An example is SOS 14 23 6 9 SEP 14 23 6 9 EOS where the first 14 23 6 9 is a sequence of unique numbers. LEN in this problem refers to length this part. The model is trained with the language modeling loss on the second part 14 23 6 9 EOS . In this problem, _lmin_ = 1. The total vocabulary size of tokens except for special tokens is equal to the maximum testing length, i.e., 150. 

**Copy (repeat).** In this problem, the model outputs the same sequence as the given sequence, which can contain repeated tokens. An example is SOS b a b SEP b a b EOS where the first b a b is a sequence of random symbols. As in Zhou et al. (2024a), we use an alphabet of only 2 symbols. LEN in this problem refers to length this part. Each symbols is sampled independently and uniformly. The model is trained with the language modeling loss on the second part b a b EOS . In this problem, _lmin_ = 1. 

**Parity.** In the parity problem, the model recognizes whether the given sequence contains an even number of 1s and outputs a corresponding token. An example is SOS 1 0 0 1 0 SEP e EOS The bits before SEP and after SOSare a random sequence of bits. LEN in this problem refers to the length of this part. The token between SEP and EOS is the label, it can be either “e” or “o”, meaning even or odd number of 1s. The model is trained with the loss on the part e EOS . _lmin_ = 0 for this problem. The bits are randomly sampled in a way such that the number of 1s is distributed uniformly given a fixed LEN. 

**Addition.** In the addition problem, the model does binary addition. An example is SOS 1 0 1 + 1 0 = 1 1 1 EOS The two operands are sampled randomly. LEN in this problem refers to the total length of them, including “+” and “=”. The model is trained with the loss on the part 1 1 1 EOS . _lmin_ = 4 for this problem. Note that we do not pad zeros in the front of operands to make them of equal length. The length of the first operand is sampled uniformly in [1, LEN-2], and the remaining length is for the second operand. After determining the lengths, random bits are sampled uniformly. 

42 

Preprint 

||**C-RASP**[_∅_]<br>**C-RASP**[periodic_,_local]|**AC**0?|
|---|---|---|
|Binary Majority<br>Binary Majority Interleave<br>Majority<br>Sort<br>Copy (unique)<br>Copy (repeat)<br>Parity<br>Addition|yes<br>yes<br>none found<br>yes<br>yes<br>yes<br>yes<br>yes<br>no<br>yes<br>no<br>no<br>no<br>no<br>no<br>no|no<br>no<br>no<br>yes<br>yes<br>yes<br>no<br>yes|



Table 3: Expressiveness properties of algorithmic tasks as defined in Appendix E.2.1 and discussed in Appendix E.2.2. In the **C-RASP** columns, “yes” means we found a **C-RASP** program; “no” means we proved that no **C-RASP** program can exist; “none found” means that we found no program despite best efforts. All problems are expressible in **TC**[0] , the tightest known upper-bound on the expressiveness of transformers. We also show membership in the circuit complexity class **AC**[0] , a smaller class sometimes compared to transformers (Hao et al., 2022; Barcelo et al., 2024); it is not predictive of length generalization either here or in the regular languages benchmark (e.g., Addition is in **AC**[0] but Majority is not). 

E.2.2 LIMIT TRANSFORMERS AND **C-RASP** EXPRESSIVENESS ON ALGORITHMIC TASKS See Table 3. We provide proof sketches. 

**Binary Majority** _∈_ **C-RASP** [ _∅_ ] A single count operation is sufficient. 

**Binary Majority Interleave and C-RASP** [ _∅_ ] We did not find a **C-RASP** [ _∅_ ] program, though we do not have a rigorous proof of nonexistence. Note that, by Lemma 38, even the (seemingly easier) task of determining whether a given input is well-formed (input length is a multiple of the number of different majority sequences) cannot be solved by **C-RASP** [ _∅_ ]. 

**Binary Majority Interleave** _∈_ **C-RASP** [ **periodic** _,_ **local** ] Periodic functions can be used to separately implement each count operation. 

**Majority** _∈_ **C-RASP** [ _∅_ ] Similar to Binary Majority. 

**Copy (unique)** _∈_ **C-RASP** [ **periodic** _,_ **local** ] If character (or _n_ -gram) repetition is prevented, then the sequence length is bounded by the alphabet, so that the space of possible inputs becomes finite, seemingly precluding the asymptotic analysis done in Theorem 7. To overcome this (apparent) challenge, we consider two formalizations of this task as operating on unbounded-length inputs. 

First, as explained in Zhou et al. (2024a) (and relatedly by Jelassi et al. (2023)), the unique copying task can be realized with an induction head circuit (Section 4.1 and Appendix C.2.2). More specifically, each position first records whether SEP has already appeared. An induction circuit then predicts new tokens in proportion to how frequently they have previously followed appearances of the current token before SEP (Section 4.1). Copying without repetition is a special case where each token occurs at most once, so the output of _f_ in (10) is always 0 or 1. We show in Appendix C.2.2 that the induction head construction from Section 4.1 is expressible in **C-RASP** [periodic _,_ local] but not in **C-RASP** [ _∅_ ]. 

We also considered a second formalization of the task, in terms of _repeated_ copying, where, given an input such as SOS a c b SEP, the model repeatedly copies the string, always predicting the next character, leading to an unbounded sequence SOS a c b SEP a c b SEP a c b SEP.... This turns the copying task into a function _f ∈F_ (Σ) that operates on unboundedly long sequences, outputting next-token predictions at each position. 

**Copy (repeat)** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] One proof proceeds via communication complexity: By Corollary 13, copying of general strings is not expressible by Limit Transformers and hence not in **C-RASP** [periodic _,_ local]. While valid, this proof does not make transparent why length 

43 

Preprint 

generalization is much easier if repetition is avoided. A different approach, not using communication complexity and crucially using the _presence of repetition_ proceeds from the fact that over the alphabet Σ _⊇{a, b, e}[∗]_ , the language Σ _[∗] be[∗] b_ Σ _[∗] ̸∈_ **C-RASP** [periodic _,_ local] (Lemma 35), and uses it to deduce that, given an input of the form _vbe[k] bw_ # _vbe[k]_ ( _v, w ∈_ Σ _[∗]_ , _k_ large), no **C-RASP** [periodic,local] program can reliably determine whether a _b_ should follow. 

**Sort** _∈_ **C-RASP** [ _∅_ ] As explained in Zhou et al. (2024a), this can be realized by selecting the smallest number in the input that is larger than the last output symbol. This algorithm does not require local or periodic positional information. 

Note that, as in COPY (unique), the input length is bounded by the alphabet size in this case, but we can view it as a task defined with unbounded length by the same trick as for COPY (unique), whereby an initial sequence such as SOS a c b SEP is repeatedly sorted, leading to an unbounded sequence SOS a c b SEP a b c SEP a b c SEP.... 

**Parity** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] See Lemma 41. 

**Addition** _̸∈_ **C-RASP** [ **periodic** _,_ **local** ] Addition is at least as hard as copying because the special case of adding zero to a number amounts to copying (Corollary 13). 

## E.3 DETAILS OF EXPERIMENTAL SETUP 

As mentioned in the main paper, at train time, we add random offsets to position indices so that all position embeddings are trained. The offsets are sampled uniformly at random in the range [0 _, N −|x|_ ] (see Section 2). Like Zhou et al. (2024a), we sample independent training batches on the fly instead of using a finite-size training set. In contrast, each test set contains 2000 samples that are sampled at the beginning of each experiment. 

For the problems where we train models with language modeling loss, the length of inputs is sampled uniformly from minimum up to maximum length in the specified range. This is true for training data and all test sets. As mentioned before, we also use predictive modeling. At each step, the model outputs a label indicating the set of possible next characters, including EOS. The models are trained on a whole sequence of tokens. In decoder-only models, standard predictive modeling approaches are less straightforward. Therefore, we assess predictions by combining the input and output spaces. For every position in the sequence, we evaluate the predicted character by comparing the output space (where each embedding represents a subset of possible next tokens) against the expected value. 

We train decoder-only transformer from scratch, using implementations from Hugging Face Transformers[9] . We train models for maximum 30k steps with a batch size of 64. We stop training early once the model’s accuracy reaches 100% on the in-distribution test set (the one in range [ _lmin,_ 50]. The model is trained with a dropout rate of 0.0. We use AdamW, with a weight decay rate of 0.01. 

In preliminary experiments, we found that different model architectures, while achieving 100% accuracy on in-distribution data, may perform differently on out-of-distribution data. To draw a conclusion about how the model performs on a problem in general, we determine the hyperparameters as follows: We consider configurations of _{_ 1, 2, 4 _}_ layers, _{_ 1, 2, 4 _}_ heads and model dimension of _{_ 16, 64, 256 _}_ , and learning rate of _{_ 0.001, 0.0001 _}_ . We sweep all the configurations by iterating over every combination and choose the one that achieves the highest accuracy on [51 _,_ 100] among those configurations whose accuracy on [ _lmin,_ 50] is 100%. When there are multiple such options, e.g., their accuracy on [51 _,_ 100] is 100%, the one with the simplest architecture is selected (when estimating complexity, we assume the following priority: number of layers _>_ number of heads _>_ model dimension). The final hyperparameters we used for each task are shown in Table 6 and 7. 

When no configuration from the search space defined above can achieve accuracy of 100% on [ _lmin,_ 50], e.g., in the case of ADDITION, we use an extra configuration, where the number of layers is 12, number of heads is 12, model dimension is 768, learning rate is 1e-4 or 3e-5 (if 1e-4 

> 9https://huggingface.co/docs/transformers/en/model_doc/gpt2# transformers.GPT2LMHeadModel 

44 

Preprint 

|**Problem**|**Model Size**|**LR**|**Max Steps**|
|---|---|---|---|
|Tomita-1, 2|1 layer; 1 head; 16 dim|1e-3|30k|
|_D_2,_D_3,_D_4,_D_12|1 layer; 4 head; 128 dim|1e-4|30k|
|Tomita-4, 7|4 layer; 2 head; 64 dim|1e-3|30k|
|_{a, b}∗d{b, c}∗_,_aa∗bb∗cc∗dd∗ee∗_|6 layer ; 4 head; 64 dim|1e-4|30k|
|(_aa_)_∗_,(_aaaa_)_∗_,(_abab_)_∗_|6 layer; 4 head; 256 dim|1e-4|60k|
|Tomita-3, 5, 6|6 layer; 4 head; 256 dim|1e-4|60k|
|_{_0_,_1_,_2_}∗_02_∗_, Parity|6 layer; 4 head; 256 dim|1e-4|60k|



Table 4: Experimental Hyperparameters for testing NoPE on the Regular Languages 

|**Problem**|**Model Size**|**LR**|**Max Steps**|
|---|---|---|---|
|Tomita-1, 2|1 layer; 1 head; 16 dim|1e-3|30k|
|_D_2,_D_3,_D_4,_D_12|1 layer; 4 head; 128 dim|1e-4|30k|
|Tomita-4, 7|4 layer; 2 head; 128 dim|1e-3|30k|
|_{a, b}∗d{b, c}∗_,_aa∗bb∗cc∗dd∗ee∗_|4 layer ; 4 head; 64 dim|1e-4|30k|
|(_aa_)_∗_,(_aaaa_)_∗_,(_abab_)_∗_, Parity|4 layer; 4 head; 128 dim|1e-4|40k|
|_{_0_,_1_,_2_}∗_02_∗_, Tomita-3, 5, 6|6 layer; 4 head; 128 dim|1e-3|30k|



Table 5: Experimental Hyperparameters for testing APE on the Regular Languages. 

does not work), and a bigger maximum number of iterations, 60k, we also use the first 3k steps as warm-up steps. 

After we determine the hyperparameter configuration, we run the experiments with multiple random seeds and report the average accuracy of 5 successful runs (those runs where the model achieves 100% accuracy on in-distribution data). We do no select successful runs in cases where we use the biggest architecture (the 12-layer configuration), because we find in many cases the accuracy on in-distribution data stops at around 99%. 

The random baseline plotted for the algorithmic tasks (Figure 1, left) is computed using a 2-layer MLP with a token embedding layer; hence, the model predicts the next token solely based on the current token. It is trained with the same hyperparameters as transformers, the learning rate is 1e-3. 

## F TRANSLATING BETWEEN TRANSFORMERS AND LIMIT TRANSFORMERS 

Here, we formally introduce the product parameterization, formally state the hypothesis class, and state and prove the technical lemmas establishing the correspondence between ordinary transformers and limit transformers. 

## F.1 PRODUCT PARAMETERIZATION 

This parameterization is defined as follows: 

|**Problem**|**Model Size**|**LR**|**Max Steps**|
|---|---|---|---|
|Binary Majority|1 layer; 1 head; 16 dim|1e-3|30k|
|Binary Majority Interleave|2 layer; 4 head; 256 dim|1e-4|30k|
|Majority|1 layer; 2 head; 256 dim|1e-3|30k|
|Sort|1 layer; 2 head; 256 dim|1e-4|30k|
|Copy (unique)|2 layer; 1 head; 64 dim|1e-3|30k|
|Copy (repeat)|4 layer; 4 head; 256 dim|1e-3|30k|
|Parity|4 layer; 2 head; 256 dim|1e-4|30k|
|Addition|12 layer;12 head;768 dim|1e-4|60k(3k)|



Table 6: Experimental hyperparameters for testing APE on each problem. In the last column, numbers in parenthesis mean the warm-up steps, which is 0 when there is no number in parenthesis. 

45 

Preprint 

|**Problem**|**Model Size**|**LR**|**Max Steps**|
|---|---|---|---|
|Binary Majority|1 layer; 1 head; 16 dim|1e-3|30k|
|Binary Majority Interleave|12 layer; 12 head; 768 dim|1e-4|60k (3k)|
|Majority|1 layer; 1 head; 64 dim|1e-3|30k|
|Sort|1 layer; 1 head; 256 dim|1e-3|30k|
|Copy (unique)|4 layer; 4 head; 256 dim|1e-3|30k|
|Copy (repeat)|4 layer; 4 head; 256 dim|1e-3|30k|
|Parity|12 layer; 12 head; 768 dim|3e-5|60k (3k)|
|Addition|12 layer;12 head;768 dim|1e-4|60k(3k)|



Table 7: Experimental hyperparameters for testing NoPE on each problem. In the last column, numbers in parenthesis mean the warm-up steps, which is 0 when there is no number in parenthesis. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0046-03.png)


**----- Start of picture text -----**<br>
Found CRASP[Periodic, Local] Program No CRASP[Periodic, Local] Program Found CRASP[] Program No CRASP[] Program<br>(aa)* (aaaa)* (abab)* Tomita-3 Tomita-5 Tomita-6<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>D-2 D-3 D-4 D-12 Tomita 1 Tomita-2<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>Tomita-4 Tomita-7 aa*bb*cc*dd*ee* {a,b}*d{b,c}* {0,1,2}*02*<br>100 100 100 100 100<br>0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>**----- End of picture text -----**<br>


Figure 2: Detailed results for regular languages with language names, corresponding to the right part of Figure 1 but with individual languages labeled. 

**Definition 43** (Product Parameterization) **.** _For l_ = 1 _, . . . , L, set_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0046-06.png)


_Given a transformer T with N_ ( _T_ ) _< ∞, define:_ 

_where the matrix product over a set S ∈P_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0046-09.png)


46 

Preprint 

## **Algorithmic Problems** 

|B<br>0<br>50<br>100<br>B<br>0<br>50<br>100|Binary<br>Majority|Binary<br>Majority|Binary<br>Majority|Binary<br>Majority|Binary Majority<br>Interleave<br>In AC0|Binary Majority<br>Interleave<br>In AC0|Binary Majority<br>Interleave<br>In AC0|Majority<br>Not In AC0|Majority<br>Not In AC0|Majority<br>Not In AC0|Majority<br>Not In AC0|n 1<br>Bin 2<br>Bin 3<br>Sort<br>n 1<br>Bin 2<br>Bin 3<br>Addition|n 1<br>Bin 2<br>Bin 3<br>Sort<br>n 1<br>Bin 2<br>Bin 3<br>Addition|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||0<br>50<br>100|||0<br>50<br>100||||0<br>50<br>100|||
|||||||||||||||
|||||||||||||||
|||||||||||||||
||in 1<br>Bin 2<br>Bin 3<br>Bi<br>Copy<br>Unique||||n 1<br>Bin 2<br>Bin 3<br>B<br>Copy<br>Repeat|||in 1<br>Bin 2<br>Bin 3<br>Bi<br>Parity||||||
|||||0<br>50<br>100|||0<br>50<br>100||||0<br>50<br>100|||
|||||||||||||||
|||||||||||||||
|||||||||||||||
||in 1<br>Bin 2<br>Bin 3<br>Bi||||n 1<br>Bin 2<br>Bin 3<br>B|||in 1<br>Bin 2<br>Bin 3<br>Bi||||||



## **Regular Languages** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0047-04.png)


**----- Start of picture text -----**<br>
In AC0 Not In AC0<br>(aa)* (aaaa)* (abab)* Tomita-3 Tomita-5 Tomita-6<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>D-2 D-3 D-4 D-12 Tomita 1 Tomita-2<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>Tomita-4 Tomita-7 aa*bb*cc*dd*ee* {a,b}*d{b,c}* {0,1,2}*02*<br>100 100 100 100 100<br>0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>**----- End of picture text -----**<br>


Figure 3: Membership in the circuit complexity class **AC**[0] does not predict transformers’ length generalization on algorithmic problems (top) or regular languages (bottom). Prior work has often linked the expressiveness of transformers to circuit complexity (e.g. Hahn, 2020; Hao et al., 2022; Merrill and Sabharwal, 2023c; Strobl, 2023; Barcelo et al., 2024). All tasks included in our experiments are in the class **TC**[0] , the tightest known upper bound on transformers’ expressiveness. A well-known circuit complexity class within **TC**[0] is **AC**[0] , known to upper-bound the power of certain hard-attention models of transformers (Hao et al., 2022; Barcelo et al., 2024), which may raise hopes that it helps understand transformers’ practical abilities. However, membership in this class does not predict transformers’ length generalization behavior. On the algorithmic problems, there is no apparent correlation at all; majority-type problems, which the attention mechanism can easily implement, are not in **AC**[0] , but problems with super-logarithmic communication complexity such as copying and addition (Corollary 13) are contained. On the regular languages, **AC**[0] exactly covers the class **FO** [ _reg_ ]. This class can be proven to include all regular languages in **C-RASP** , but it also includes various languages that transformers length-generalize poorly on, such as Tomita-3. A natural subclass, obtained by restricting the size of **AC**[0] circuits to a linear number of wires, yields the class **FO** 2[ _Reg_ ] (Cadilhac and Paperman, 2022), which does not match transformers’ behavior well either, e.g. it includes _{_ 0 _,_ 1 _,_ 2 _}[∗]_ 02 _[∗]_ (bottom right, equals Σ _[∗] be[∗]_ from Lemma 11) but does not include D-12. Taken together, established circuit complexity classes do not account for Transformers’ length generalization behavior. Compare to **C-RASP** results in Figures 1 and 2. 

47 

Preprint 

## **(A) Star-Free vs Non-Star-Free Languages** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0048-02.png)


**----- Start of picture text -----**<br>
Star Free + APE Non Star Free + APE Star Free + NoPE Non Star Free + NoPE<br>(aa)* (aaaa)* (abab)* Tomita-3 Tomita-5 Tomita-6<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>D-2 D-3 D-4 D-12 Tomita 1 Tomita-2<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>Tomita-4 Tomita-7 aa*bb*cc*dd*ee* {a,b}*d{b,c}* {0,1,2}*02*<br>100 100 100 100 100<br>0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>**----- End of picture text -----**<br>


## **(B) Dot-Depth** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0048-04.png)


**----- Start of picture text -----**<br>
Dot Depth=1 Dot Depth=2 Dot Depth=3 Dot Depth=4 Dot Depth=12<br>(aa)* (aaaa)* (abab)* Tomita-3 Tomita-5 Tomita-6<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>D-2 D-3 D-4 D-12 Tomita 1 Tomita-2<br>100 100 100 100 100 100<br>0 0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>Tomita-4 Tomita-7 aa*bb*cc*dd*ee* {a,b}*d{b,c}* {0,1,2}*02*<br>100 100 100 100 100<br>0 0 0 0 0<br>Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3 Bin 1 Bin 2 Bin 3<br>**----- End of picture text -----**<br>


Figure 4: (1) Comparing length-generalization with a standard notion of the complexity of finitestate languages: Star-free languages (green) do not require modular counting (McNaughton and Papert, 1971), have simpler algebraic representations in terms of group-free monoids (Sch¨utzenberger, 1965), are easily represented by modern state-space models (Sarrof et al., 2024), and match the expressiveness of a formal model of hard attention Transformers (Yang et al., 2023). However, they do not consistently lead to length generalization in transformers, which on the other hand lengthgeneralize on some non-star-free languages such as ( _aa_ ) _[∗]_ . The expressiveness of **C-RASP** correctly accounts for the observed behavior. (2) Within the star-free languages, a standard complexity metric is dot-depth, with increased dot-depth indicating increased complexity (non-star-free languages are plotted in gray color). Dot-depth does not predict length generalization, which succeeds on some languages at dot depths 1 and 12 and fails at some languages at intermediate depth. See Figure 3 for further discussion regarding another existing notion of complexity, circuit complexity, also much less successful than **C-RASP** expressiveness at predicting length generalization. Compare to **CRASP** results in Figures 1 and 2. 

48 

Preprint 

_is computed in descending order of layers; with the S associated with the lowest layer at the right. For instance,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0049-02.png)


**Remark 44.** _Here, we exemplify the Product Parameterization (Definition 43)._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0049-04.png)


**Remark 45.** _For ease of notation, we have not restricted the layers from which different vector parameters are taken in the definition of α and β; hence, they will also include products that are not relevant to actual computations, such as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0049-06.png)


_where a vector of the form V_ 3 _,h′′′_ _**p** j cannot actually feed into the computation of queries in the first layer. This is simply for simplicity of notation; such products will not impact results, though one could explicitly exclude them if one wants to obtain tighter quantitative bounds on the parameter count of Limit Transformers in Lemma 52._ 

## F.2 FORMAL DEFINITION OF HYPOTHESIS CLASS 

**Definition 46** (Hypothesis Class, corresponds to Definition 4) **.** _Let p ∈_ N _be fixed. For each n_ = 1 _,_ 2 _,_ 3 _, . . . , define the hypothesis class_ Θ _n as the set of transformers T (as defined in Section 2) where_ 

_1. N_ ( _Tn_ ) = _n_ 

_2. each parameter vector and matrix of T is represented at p bits of precision_ 

_3. each product function (Definition 43) involving positional encodings is translationinvariant. That is, every product function involving exactly one positional encoding is constant across positions, and for every_ 1 _≤ i, j, i_ + ∆ _, j_ + ∆ _≤ n,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0049-13.png)


_for all l, h, S_ 1 _, S_ 2 _making these objects well-defined._ 

## F.3 FROM LIMIT TRANSFORMERS TO TRANSFORMERS 

**Lemma 47.** _Let T∞ be a Limit Transformer satisfying_ PERIODIC _and_ LOCAL _. Then there are transformers T_ 1 _, T_ 2 _, . . . (Ti ∈_ Θ _i) such that, for all i ∈_ N _, x ∈_ S _, o ∈{_ 0 _, . . . , i −|x|}:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0049-17.png)


_and_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0049-19.png)


_Proof._ We use hats to indicate the parameters of the constructed transformers. Fix _N ∈_ N; we construct _TN_ . 

49 

Preprint 

Let ∆ be the periodicity of _**p** i_ . The construction sets _L_[ˆ] = _L_ + 2 and _H_[ˆ] = max _{_ 1 _, H,_ ∆ _}_ . Each activation has _d_[ˆ] := _d_ + _N_ + 3∆+ 2 dimensions, which can be partitioned into six regions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0050-02.png)


Region I directly emulates the computations of the Limit Transformer. Region II carries absolute positional information, used for simulating the positional functions _ϕl,h_ . 

We define the token and positional encodings as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0050-05.png)


where _**e** i_ is the _i_ -th unit vector. Region I holds token information. Region II holds exact position information. Region III holds modular position information. Region V indicates whether the token is start-of-sequence (SOS) or not. Content will be written to Regions IV and VI by attention components. 

**Intuition** At first sight, a simple intuitive translation just uses Regions I and II, placing all parameters of _T∞_ into Region I, and taking one-hot vectors _**e** i_ for Region II to encode exact positional information, so that the functions _ϕl,h_ can be implemented by products _**p**_ ˆ _i_ _**K**_[ˆ] _T_ _**Q**_[ˆ] _**p**_ ˆ _j_ . One would thus use the simpler encoding _**E**_[˜] _σ_ := [ _**E** σ,_ 0] and _p_ ˜ _i_ := [ _**p** i,_ _**e** i_ ]. Such a translation would be able to reproduce the input-output behavior of _T∞_ . However, it would fall short in two ways: First, the positional encodings _**p** i_ can give rise to patterns whereby _**p**[T] i[K][T][ Q]_ _**[p]**[j]_[is periodic in] _[ j][ −][i]_[ when] _[ j][ −][i]_ is large, and thus bounded away from zero at unboundedly many distances _j − i_ , making (8) unbounded. We will avoid this by routing modular positional information through a value matrix _**V**_ 1 _,_ 1 before making it available to attention computations; intuitively, this is possible because the vectors _**p** i_ have a bounded-dimensional span. Modular positional information starts out in Region III of _**p**_ ˆ _i_ , and is copied by _**V**_ 1 _,_ 1 into Region IV; no _**K**[T]_ _**Q**_ matrix in the construction will directly address Region III. Second, transformers in Θ _N_ must satisfy the requirement that all product functions are translation-invariant; such a requirement need not be implemented by _T∞_ (e.g., MLPs could respond differently to different _**p** i_ ’s), and thus also not by the simple translation sketched. We overcome this by adding ∆ different attention heads, each assigned to some _k ∈{_ 0 _, . . . ,_ ∆ _−_ 1 _}_ , each of which primarily attends to the SOS symbol (based on Region V), with a stronger weight in the _k_ -th head falling on SOS if the distance between the query position and the SOS position is congruent to _k_ modulo ∆ (based on Region IV). These attention weights are written, via the value matrix, to Region VI. An MLP then compares each of these attention weights to the weights resulting from uniform attention, thereby determining the head giving rise to the highest attention weight, and places the matching encoding _**p** i_ into Region I. This construction maintains translation-invariance of all product functions; most importantly, it makes all _β_ functions involving positional encodings equal to zero: The MLP reads from Region VI, whose entries are linear combinations of entries from Region ˆ V, which is zero in all _pi_ . Crucially, the dependence of the positional encoding _**p** i_ written to Region ˆ I on the original positional information in _pi_ is mediated entirely through attention weights, which do not enter the definition of the product functions. Once these computations are completed, Region I matches the activations in _T∞_ at offset 0, and a direct simulation of _T∞_ can proceed based on Regions I and II. Taken together, we expand the intuitive simple construction to make rigorous the intuition that bounded-rank positional information can be utilized by transformers even under the constraints that (8) be bounded and that product functions be translation-invariant. 

**Layer 1: Copying Periodic Positional Information to Region IV** In the lowest layer, each position attends to itself and moves the periodic positional information from Region III to Region IV. 

50 

Preprint 

Formally: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-02.png)


for some Ω _>_ 0 to be chosen later. After attention (the MLP does nothing, _**A**_ 1 _,_ _**B**_ 1 _,_ _**b**_ 1 _≡_ 0), the output will be 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-04.png)


where _ϵ <_ 0 _._ 1 when Ω is sufficiently large. Region III will not be addressed by any further downstream matrices or vectors. The idea behind this operation is to ensure no _**K**_[ˆ] _[T]_ _**Q**_[ˆ] matrix will directly have to read from Region III – rather, any dependence of attention logits on modular positional information is mediated by the _**V**_ 1 _,_ 1 matrix. This allows us to keep (8) bounded even in the presence of such dependence, as we detail below. Note that this strategy importantly relies on ∆ _≤R∞_ ( _T∞_ ), so that _rank_ ( _**V**_ 1 _,_ 1) is bounded independently of N. Intuitively, modular positional information (unlike the full positional information encoded in Region II) can be routed through bounded-rank components, which enables keeping (8) bounded. 

**Layer 2: Determining Position Relative to SOS** We now add a second layer, in which we attend with ∆+ 1 different heads, where the _s_ -th head tests whether the distance to SOS is congruent to _s_ modulo ∆, and the ∆+ 1-st head attends uniformly. By determining which head attends most strongly to SOS, we can read out the relative position with an MLP without breaking the translation invariance of product functions. 

For _h_ = 1 _, . . . ,_ ∆, let _**Q**_ 2 _,h_ be such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-08.png)


Let _**K**_ 2 _,h_ ( _h_ = 1 _, . . . ,_ ∆) be the identity matrix restricted to Regions IV and V. Further, let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-10.png)


Then, for _h_ = 1 _, . . . ,_ ∆, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-12.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-14.png)


Intuitively, heads 1 _, . . . ,_ ∆ attend preferentially to positions at a given distance modulo ∆; head ∆+ 1 attends everywhere. Define _**V**_ 2 _,h_ for _h_ = 1 _, . . . ,_ ∆+ 1 by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0051-16.png)


51 

Preprint 

As the only vector parameter with an entry in Region V is the token embedding for the SOS token, the outcome of this attention block effectively writes the attention falling on the SOS token for each of the ∆+ 1 attention heads. We then use a Heaviside MLP with the number of hidden units bounded in ∆ to determine for which _s_ = 1 _, . . ._ ∆ it holds that entry _s_ in Region VI has a greater (as opposed to smaller) entry than entry ∆+ 1. The MLP, via the _B_ matrix, then writes _**p** s_ to Region I. A special case occurs at SOS, where Region V is 1 (it is 0 everywhere else); here, the ∆ MLP units described above are disabled (a 1 in Region V causes a large negative number to be added to their inputs) and the MLP instead writes _**p**_ 1 to Region I. Overall, ∆+ 1 MLP units with Heaviside activation are sufficient for this construction. Overall, after the MLP, the _i_ -th position in the string has _**p**_ (( _i−_ 1)%∆)+1 added to Region I. Overall, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0052-02.png)


where _o_ is the offset, and the second equality holds if the Limit Transformer is run at _o_ = 0. In layers 2 _, . . . , L_[ˆ] , only Regions I and II will receive any consideration. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0052-04.png)


where _i ≤ j_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0052-06.png)


to satisfy 

_**p**[T] i[W][l,h]_ _**[p]**[j]_[=] _[ ϕ][l,h]_[(] _[i, j]_[)] (23) Then _W_ is a sum of matrices each of which has the value _ϕl,h_ ( _i−j, j_ ) for some _i_ on an (off-)diagonal and zeros elsewhere; hence 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0052-09.png)


Overall, _∥K_[ˆ] _l[T]_ +1 _,h[Q]_[ˆ] _[l]_[+1] _[,h][∥]_[can be bounded, independently of] _[ N]_[, in terms of] _[ ∥]_ _**[K]** l,h[T]_ _**[Q]**[l,h][∥]_[and] _[ ∥][ϕ][l,h][∥]_[1][.] As _ϕl,h_ is local, this 1-norm is finite. 

We construct all other parameter matrices and vectors by placing the parameter from the Limit Transformer into the Region I and leaving all other regions zero. Now, by induction, the first _d_ dimensions of any activation will match those of the Limit Transformer, but shifted by two layers: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0052-12.png)


As the _U_ matrix only reads from Region I, the output will also be the same as for the Limit Transformer. 

For the excess heads, the matrices are just set to 0 – in the first or the higher layers – depending on whether _H_ or ∆ is larger than the other. 

**Bounding Norms and Ranks** At _l ≥_ 2, now the ranks of _**V** l,h_ and the norms of _**A** l,_ _**B** l, U_ and the _ℓ_ 2 norms of _e, b, c_ will be the same they were in the Limit Transformer. The increases from the first and second layer are bounded in terms of ∆ and hence _R∞_ ( _T∞_ ) 

52 

Preprint 

**Verifying Boundedness of (8)** By construction, _**p**_ ˆ _[T] i_ _**K**_[ˆ] 1 _[T] ,h_ _**[Q]**_[ˆ][1] _[,h]_[ ˆ] _**[p]**[j]_[=] _[ δ][ij][δ][h]_[1][ and] _**[p]**_[ˆ] _[T] i_ _**K**_[ˆ] 2 _[T] ,h_ _**[Q]**_[ˆ][2] _[,h]_[ ˆ] _**[p]**[j][≡]_ 0. For the higher layers, the boundedness follows because _T∞_ satisfies LOCAL. 

**Verifying Translation Invariance** We need to verify that all product functions are translationˆ invariant. Each _pi_ contains entries in Regions II and III. In the first layer, we have _α_ 1 _,_ 1 _,∅,∅,_ _**p** i,_ _**p** j_ = _δij_ , hence, these are translation-invariant. In the second layer, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0053-03.png)


These are all translation-invariant. All _α_ 2+ _l,h,∅,∅,_ _**p** i,_ _**p** j_ equal a function _ϕl,h_ ( _i, j_ ) and hence are translation invariant. No _**V**_ matrix ever reads from Region II. Overall, all _α_ products are translationinvariant. In higher layers, _**K**[T]_ _**Q**_ matrices read from Regions I and II. The positional encodings _**p**_ ˆ _i_ write to Region II but not – not even when mediated directly through value matrices – Region I, and the products are translation-invariant because the functions _ϕl,h_ are. 

Consider 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0053-06.png)


because the first layer MLP does nothing. Second, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0053-08.png)


because the second layer MLP only reads from Region VI, and none of _**p** i,_ _**V**_ 1 _**p** i_ contain any entries in Region VI. Also, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0053-10.png)


since none of _**p** i_ , _**V**_ 1 _**p** i_ , _**V**_ 2 _**V**_ 1 _**p** i,_ _**V**_ 2 _**p** i_ contain any entries in Region I, where _**U** σ_ has its entries. 

53 

Preprint 

Since _**V**_ 2 _,s_ all read only from Region V, whose entries have no connection to _p_ ˆ _i_ , we also have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-02.png)


Overall, all _β_ products involving _**p** i_ are translation-invariant. 

## F.4 FROM TRANSFORMERS TO LIMIT TRANSFORMERS 

We first establish various smaller lemmas. The first lemma informally says that, when _f_ ( _i, j_ ) := _**p**[T] i_ _**[Ap]**[j]_[is][translation-invariant][in][(] _[i, j]_[)][,][and] _**[A]**_[has][bounded][rank,][then] _[f]_[(] _[i, j]_[)][is][periodic.][This] lemma is key to the prominent role of the PERIODIC property in Theorem 7. 

**Lemma 48.** _Let p ∈_ N _, and let N ∈_ N _. Let_ _**p**_ 1 _, . . . ,_ _**p** N ∈_ R _[k] such that ∥_ _**p** i∥_ 2 _< C; let_ _**A** ∈_ R _[k][×][k] . Let f_ ( _i, j_ ) := _**p**[T] i_ _**[Ap]**[j][be translation invariant, in the sense that]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-07.png)


_Further assume that, for i ≤ j, f_ ( _i, j_ ) _can be expressed with p fractional bits._[10] _Define for n ≥_ 0 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-09.png)


_Then there is_ ∆ _∈_ N _upper-bounded in terms of_ rank( _**A**_ ) _, p, C, and ∥_ _**A** ∥ (but not N ) such that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-11.png)


_Proof._ Let _ρ_ := rank( _**A**_ ); it is _>_ 0 without loss of generality. We write the singular value decomposition of _**A**_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-13.png)


where Σ _∈_ R _[ρ][×][ρ]_ , _**U** ,_ _**V** ∈_ R _[ρ][×][d]_ , where _∥_ _**U** ∥, ∥_ _**V** ∥≤_ 1. Then we can write 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-15.png)


R[2 rank(] _[A]_[)] ; note that this increases the norms of these objects at most by a multiplicative constant while preserving the products _**p**[T] i_ _**[Ap]**[j]_[.] 

As _∥_ _**p** i∥_ 2 _< C_ for all _i_ , we find that, when _N_ is sufficiently large, for each _ϵ >_ 0, there are _**p** i,_ _**p** j_ ( _i < j_ ) such that: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-18.png)


Take ∆:= _j − i_ . The minimum required distance ∆ can be upper bounded by considering the _ϵ_ -packing number of _{v_ : _∥v∥≤ C}_ and applying the pigeonhole principle. Hence, overall, ∆ can be upper-bounded in terms of _ϵ_ , rank( _**A**_ ), and _C_ . Importantly, this bound is independent of _N_ . Hence, _∀k ∈{i_ + ∆ _, . . . , N }_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0054-20.png)


> 10In particular, this is satisfied if _**p** i_ , _**A**_ are each expressed at some fixed precision _q_ , where _p ≥_ 3 _q_ . 

54 

Preprint 

Equivalently, using the substitution _l_ := _k − i −_ ∆, we have, for any _l ∈{l, . . . , N −_ ∆ _}_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0055-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0055-03.png)


**Lemma 49.** _Assume each parameter in a transformer is represented at p bits of precision. Then each product function is exactly represented at_ (4 + 2 _L_ ) _p bits of precision._ 

_Proof._ Each product function consists at most of two vectors, a key and query matrix, and up to 2 _L_ value matrices. This results in a sum of numbers that each are a product of up to 4+2 _L_ numbers that each are individual parameters. As each number is represented at _p_ bits of precision, each product is represented at (4 + 2 _L_ ) _p_ bits of precision. 

It will be useful to define a complexity metric applicable to Limit Transformers: 

**Definition 50.** _For a Limit Transformer T , define R∞_ ( _T_ ) _as the sum of_ 

_1. L_ + _H_ + _d_ 

_2. the precision p used for expressing the parameters (Definition 2), and the precision p used ·_ 

_for rounding attention logits and the output of_ exp( ) _(Section 2)._ 

_3. the maximum ℓ[∞] norm of all parameter vectors and matrices (including positional encodings)_ 

_4. the minimal periodicity_ ∆ _of the positional encodings_[11] 

_∞ 5._ max _l,h_ � _i_ =1 _[|][ϕ][l,h]_[(1] _[,]_[ 1 +] _[ i]_[)] _[|]_[2] _[ (short:]_[max] _[l,h][ ∥][ϕ][l,h][∥]_[2] 2 _[).]_ 

**Proposition 51.** _Let A ∈_ R+ _. Let U be the set of Limit Transformers T such that R∞_ ( _T_ ) _≤ A. Then the set of parameter settings in U , other than the ϕl,h functions, is finite._ 

_Proof._ Immediate. 

We now state the key lemma translating ordinary transformers to Limit Transformers using the product parameterization: 

**Lemma 52.** _Let T ∈_ Θ _n, at p bits of precision. Let the alphabet_ Σ _be fixed._[12] _Then there is a Limit Transformer T∞ such that T ≡ T∞ at length ≤ n and_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0055-17.png)


_for some universal function F_ : R+ _→_ R+ _and_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0055-19.png)


_for the_ _**p** i,_ _**K** l,h,_ _**Q** l,h parameters of T ; for each l, h. In particular, T∞ satisfies_ PERIODIC _and each ϕl,h is translation-invariant._ 

We prove the lemma in the remainder of the section. 

F.4.1 PROVING LEMMA 52 (I): PRELIMINARIES 

We discuss various preliminaries, before presenting the construction, explaining its intuition, and explaining how its soundness is formally proven. 

> 11Formally, 1 plus the supremum of the set of ∆’s for which _pi ̸≡_ _**p** i_ +∆. 

> 12The bound on _R∞_ ( _T∞_ ) depends on it, but during the inference procedure, the alphabet is assumed fixed. 

55 

Preprint 

**Basic Idea** We will construct _T∞_ so that the entries in every parameter are 0, 1, or one of the product functions from Definition 43. This will automatically ensure that its parameters are represented at fixed precision _p_ bounded in terms of _R_ ( _T_ ), and with each entry bounded in terms of the spectral norms of parameter matrices and the _ℓ_ 2 norms of parameter vectors, hence, also bounded in terms of _R_ ( _T_ ). Importantly, we will use the definition of _R_ ( _T_ ) and the definition of Θ _n_ to restrict attention to a number of product functions that are bounded only in terms of _R_ ( _T_ ), independently of _n_ . 

**Bounding Active MLP Units** First, given _∥_ _**A** l∥F < ∞_ , the number of nonzero entries is bounded as _≤[∥]_ 2 _**[A]**[−][l]_[2] _[∥][p] F_[2][, which is bounded in terms of] _[ R]_[(] _[T]_[)][.][Similarly, the number of nonzero entries in] _**[ b]**[l]_[is] bounded as _≤_ 2 _∥[−] b∥_[2][2] 2 _[p]_[,][and][similarly][for] _**[B]**[l]_[.][Let] _[d][MLP][≤][d]_[be,][across] _[l]_[=][1] _[, . . . , L]_[the][maximum] of the maximum number of nonzero entries in _**b** l_ , and of the maximum number of nonzero rows in _**A** l_ . Without loss of generality, by reordering rows in _**A** l_ and columns in _**B** l_ , we may assume that, in each layer, these entries are in the first _dMLP_ dimensions. Then _dMLP_ is upper-bounded in terms of these (or _d_ , if the bounds exceed _d_ ); this is bounded in terms of _R_ ( _T_ ) and independent of _n_ . In each layer, only _≤ dMLP_ of the MLP units have nonzero input weights _**A** l_ , output weights _**B** l_ , or biases _**b** l_ . Removing product functions belonging to the inactive units, we set: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0056-03.png)


Then, the size of _VI_[�] is bounded in terms of _R_ ( _T_ ). 

**Periodicity of Bounded-Rank Positional Functions** By Lemma 48 all products _**p**[T] i[. . .]_ _**[ p]**[j]_[where] the intervening material has bounded rank are periodic in _j − i_ with period bounded in terms of the rank of the intervening material, and hence _R_ ( _T_ ). Let ∆ be the least common multiple of the periods obtained from Lemma 48 across the (finitely many) different products of the form _βl,h,S_ 1 _,S_ 2 _,_ _**p** i,_ _**p** j_ where _S_ 1 _∪S_ 2 = _∅_ . Define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0056-06.png)


Then, the size of _VO_[�] is bounded in terms of _R_ ( _T_ ). 

F.4.2 PROVING LEMMA 52 (II): CONSTRUCTION OF _T∞_ 

**Translation in Terms of Regions** We use hats (i.e., � _·_ ) to mark the parameters and activations of the Limit Transformer, distinguishing those from the parameters and activations of the original transformer _T_ . 

Each _d_ -dimensional parameter vector and activation (residual streams and attention outputs) is translated to a vector consisting of three regions, each having a fixed number of dimensions bounded in terms of _R_ ( _T_ ), That is, each vector parameter or activation _**v**_ (e.g., _**p** i_ , _**E** σ_ , _**y** i_[(] _[l]_[)][) is translated to a] parameter or activation _**v**_ � (e.g., _**p**_ � _i_ , _**E**_[�] _σ_ , _**y**_[�] _i_[(] _[l]_[)][) vector consisting of the following three regions:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0056-11.png)


56 

Preprint 

**Intuition of the Construction** The first region, denoted Γ _S,_ _**w**_ ( _**v**_ �), has one entry for every choice of _S ∈P_ ; _**w** ∈ VI_[�] _l_ 2. Intuitively, the entry Γ _S,_ _**w**_ ( _**v**_ �) describes the outcome of applying all value matrices in _S_ and then finally the vector _**w**[T]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0057-02.png)


(Recall Definition 43 for the notation[�] _S∈S[S]_[.)][The second and third regions each have one entry] for every choice of 1 _≤ l ≤ L, S_ 2 _⊆T_ 2; _T_ 1 _, T_ 2 _∈P,_ _**w**_ 1 _,_ _**w**_ 2 _∈ VO_[�] . These regions contain the information necessary for computing attention logits. Intuitively, _T_ 1, _T_ 2 describe the value matrices through which _**w**_ 1 and _**w**_ 2, respectively, pass before the computation of attention logits in layer _l_ . For parameter vectors _**w**_ 1 _,_ _**w**_ 2 (e.g., token embeddings or columns of a _**B** l_ matrix – positional encodings are somewhat special), we simply expect (note the duplicated arguments _T_ 1, _T_ 2 – these will be explained in the next paragraph): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0057-04.png)


Thus, Λ can be viewed as holding key parameters, whereas Ω can be viewed as holding query parameters, for the contribution that the pair of _**w**_ 1 _,_ _**w**_ 2 makes to attention logits in layer _l_ , after passing through the value matrices in _T_ 1, _T_ 2, respectively. As a convention, at the level of parameter vectors, the Λ component will hold the attention logit contribution (the RHS of this equation), whereas the Ω component will just hold zeros and ones. At the level of intermediate activations _**v**_ ( _**y** i_[(] _[l]_[)] or _**Y** i_[(] _[l]_[)] ), the situation is slightly more complex: here, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0057-06.png)


denotes the contribution to attention logits for head _h_ at layer _k_ arising from multiples of �� _S∈T_ 1 _[S]_ �� _**w**_ 1 in an activation _**y**_[�] _i_[(] _[k][−]_[1)] interacting with multiples of �� _S∈T_ 2 _[S]_ �� _**w**_ 2 in an activation _**y**_[�] _j_[(] _[k][−]_[1)] ; a similar idea applies to Ω _..._ . However, additional care is needed to ensure that only contributions from value matrices are counted that were actually passed through. The additional argument _S_ 1 serves as a “to-do-list”: it records which of the value matrices in _T_ 1 still have to be traversed; whenever an activation passes through a value matrix _**V** l,h′_ , the value matrix _**V**_[�] _l,h′_ of the Limit Transformer moves entries from Λ _S_ 1 _,T_ 1 _,T_ 2 _,k,h,_ _**w**_ 1 _,_ _**w**_ 2 to Λ _S_ 1 _−{_ _**V** l,h′ },T_ 1 _,T_ 2 _,k,h,_ _**w**_ 1 _,_ _**w**_ 2 – effectively removing itself from the “to-do-list”. The same princple applies to Ω, which maintains a to-do-list _S_ 2 for _T_ 2. In the end, only those components where the to-do-lists are empty (formally, _S_ 1 = _S_ 2 = _∅_ ) will enter attention logit computations of the Limit Transformer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0057-08.png)


where the sum runs over all _**v** ,_ _**w** ∈ VO_[�] , and _T_ 1 _, T_ 2 runs over all sets of value matrices from layers _≤ l_ . 

In the remainder of the proof, we present a detailed formal construction implementing this intuition. We first define, for each vector _**v** ∈{_ _**p** i,_ _**E** σ,_ _**b** l_ : _i, l, σ}_ its translation � _**v**_ ; throughout, we will define each of the three regions. 

**Vector Parameters** _**b** l ∈VO_ We take ( _**b**_[�] _l_ ) _s_ := ( _**b** l_ ) _s_ for _s_ = 1 _, . . . , dMLP_ . 

**Vector Parameters** _**E** σ ∈VO_ The first region provides products with other vectors appearing at higher layers (rows/columns of the _**A** l_ matrices and the unembedding matrix). The second and third 

57 

Preprint 

regions provide products leading up to keys and values. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0058-02.png)


� **Vector Parameters:** _**p** i_ For _**v**_ = _**p** i_ , the construction is analogous, however, we zero out the entries for Λ _∅,∅,l,h,_ _**p** i,_ _**p** j_ and Ω _∅,∅,l,h,_ _**p** i,_ _**p** j_ , as these will be taken care of by the _ϕl,h_ functions. Formally: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0058-04.png)


We need to establish that _T∞_ satisfies PERIODIC with the period ∆ given above. First, _βS,_ _**v** ,_ _**p** i_ is independent of _i_ by translation-invariance, thus trivially periodic in _i_ . Second, the Λ and Ω entries are periodic in _i_ with period ∆ by construction. 

**Matrix Parameters:** _**V**_[�] _l,h_ Each entry in the _**V**_[�] _l,h_ matrix is zero or one. We define it implicitly, in terms of its action on the three different regions: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0058-07.png)


**Matrix Parameters:** _**A**_[�] _l_ **,** _**B**_[�] _l_ Let _s ∈{_ 1 _, . . . , dMLP }_ . For the _s_ -th unit in the MLP at layer _l_ , we first define the _s_ -th row of _**A**_[�] _l_ by setting 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0058-09.png)


58 

Preprint 

and define the _s_ -th column of _**B**_[�] _l_ as follows – writing _X_[ˆ] _∈_ R for the _s_ -th hidden unit activation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-02.png)


This defines the _**A**_[�] _l_ and _**B**_[�] _l_ matrix parameters, and we can write, letting _ψl,s_ denote the activation function (ReLU or Heaviside) applying to the _s_ -th hidden MLP unit: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-04.png)


or equivalently 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-06.png)


matching the formulation of MLPs for our model of transformers (Equation 4). 

Note that the hidden dimension of the MLP in the Limit Transformer is now _dMLP_ , which will be smaller than _d_[�] . We thus pad the remaining rows/columns of _**A**_[�] _l,_ _**B**_[�] _l_ , and the remaining entries of _**b**_[�] _l_ with zeros. 

**A partial order on sets of value matrices** For _S, T ∈P_ , we write _T ≥l S_ to denote that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-10.png)


**Intuitively** , “ _T ≥l S_ ” says that “ _among the value matrices in T , the activation has already passed through all value matrices at layer l and below_ ”. For example: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-12.png)


� **Matrix Parameters:** _**K** l,h[T]_ _**[Q]**[l,h]_ We again define them implicitly in terms of regions; this can be realized using matrices _**K**_[�] _l,h[T][,]_ _**Q**_[�] _l,h_ where all entries are 0 or 1. Importantly, we sum only those entries where the “to-do-lists” _S_ 1 _, S_ 2 are empty, and the sets _T_ 1 _, T_ 2 only contain value matrices at layers _≤ l_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-14.png)


**Matrix Parameters:** _**U**_ The _**U**_ matrix is translated as follows: 

**Positional Function** Define for _l_ = 1 _, . . . , L_ and _h_ = 1 _, . . . , H_ , when 1 _≤ i ≤ j ≤ N_ ( _T_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0059-17.png)


As _T ∈_ Θ _n_ , _ϕl,h_ ( _i, j_ ) only depends on _j − i_ . 

59 

Preprint 

**Bounding** _R∞_ ( _T∞_ ) First, we showed above that _d_[�] is upper-bounded in _R_ ( _T_ ). Second, all parameters are represented at precision bounded in terms of _R_ ( _T_ ): those parameters that are taken from product functions have precision _≤_ 4 _Lp_ bits; those involving the SVDs of _K[T] Q_ matrices also have bounded precision. Third, the _ℓ[∞]_ norm of all parameter vectors is bounded in terms of _R_ ( _T_ ) by construction. Fourth, ∆ is bounded in terms of _R_ ( _T_ ) as discussed above. The boundedness of the fifth term is immediate. 

**Summary** We have constructed a Limit Transformer _T∞_ such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0060-03.png)


for some universal function _F_ : R+ _→_ R+ and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0060-05.png)


for theinvariant. _**p** i,_ We have also constructed _**K** l,h,_ _**Q** l,h_ parameters of _T_ _**p**_ � _i_ ; for eachwith period _l, h_ ∆. , so thatBy assumption on _T∞_ satisfies P _T_ , eachERIODIC _ϕl,h_ .is translation- 

## F.4.3 PROVING LEMMA 52 (III): PROVING CORRECTNESS 

In order to conclude Lemma 52, it remains to establish the correctness of the translation; that is, _T ≡ T∞_ at length _≤ N_ ( _T_ ). To do this, it suffices to show that both transformers provide the same next-token predictions for each _i_ = 1 _, . . . , N_ ( _T_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0060-09.png)


**Informally** , proving this requires showing that the attention logits and MLP activations in _T∞_ match tor parameters and how value matricesthose in _T_ ; the result then follows from the linearity of _**V**_[�] _l,h_ move information. Γ _∅,_ _**U** σ_ and the way **Formally** Γ, we prove the correctness _..._ is defined for the vecof the translation inductively, by showing the following equalities. Recall (from Definition 43) that, when _S ∈P_ is a set of value matrices, we write[�] _S∈S[S]_[for the product of these matrices, ordered] by layers, with the matrix associated with the lowest layer at the right. Then 

**Lemma 53** (Preservation of Products by Translation) **.** _For layer l ∈{_ 0 _,_ 1 _, . . . , L}, for any S, S_ 1 _, S_ 2 _∈P for any k > l, for any m ≥ l, and for any_ _**w** ∈ VI_[�] _, by induction over the layers l:_ 

60 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0061-01.png)


_and analogous statements with the post-MLP activations_ _**y** i_[(] _[l]_[)] _[,]_[ �] _**y** i_[(] _[l]_[)] _replaced by the pre-MLP activations_ _**Y** i_[(] _[l]_[)] _,_ _**Y**_[�] _i_[(] _[l]_[)] _._ 

From (A), we in particular obtain the correctness of the translation by noting that next-token predictions are replicated: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0061-04.png)


_Proof of Lemma 53._ The formal proof proceeds by induction over _l_ . It is conceptually straightforward, consisting of expanding definitions and taking care of the special treatment of positional encodings. We show it in considerable detail to build intuition. For the **inductive base** , at _l_ = 0, where _**y** i_[(0)] is a sum of a word embedding and a positional embedding, the claims are immediate from the definitions. For expository purposes, and for building intuition for the more complex inductive step, we show them in more detail. Starting from (for simplicity, we are taking the offset to be zero here): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0061-06.png)


61 

Preprint 

we first, for (A), write 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0062-02.png)


proving case (A) of the inductive base. Second, for (B) and (C), write using the linearity of Λ _...,_ Ω _..._ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0062-04.png)


The only way of satisfying _T ≥_ 0 _S_ is for _T_ to equal _S_ . After plugging in the definitions, the sums collapse due to the indicator terms in the definition of the token and positional encodings, and we obtain after simplifying: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0062-06.png)


By translation-invariance, the second and third term are independent of the positional encoding arguments. By our choice of ∆ at the beginning of the proof, the fourth term equals _αS_ 1 _,S_ 1 _,S_ 2 _,k,h,_ _**p** i,_ _**p** j ·_ 1 _S_ 1 _∪S_ 2= _∅_ , as this is periodic in ( _i, j_ ) with periodicity ∆. We can thus rewrite as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0062-08.png)


Applying the definition of _α..._ , the above equals 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0062-10.png)


This establishes cases (B) and (C) of the inductive base. The proof of cases (D) and (E) in the inductive base is analogous. 

For the **inductive step** , the **intuition** is that each activation _**y** i_[(] _[l]_[)] is a linear combination of vector parameters, with different sets of value matrices acting on those: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0062-13.png)


where the coefficients are determined by attention weights and the activations of MLP hidden units. Importantly, the attention weights and MLP activations turn out to be the same in the Limit Transformer as in the original transformer, provided we can prove that attention and MLPs are faithfully 

62 

Preprint 

simulated (which indeed follows from cases A and C of the inductive claim). Hence, the same decomposition is valid in the Limit Transformer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0063-02.png)


with the same _λ_ _**v** ,i,l,S_ coefficients as in the original transformer. Then, case (A) of the inductive claim follows intuitively by the calculation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0063-04.png)


�[��] � which is warranted provided that, when _**b** ∈VO_ and _**w** ∈VI_ , we have that _**w**[T] S∈S[S]_[�] � _**v**_[��] � equals _**w**[T] S∈S[S]_ � _**v**_ – this is ensured because of the way the vector parameters _**v**_ and the value matrices _**V**_[�] _l,h_ are defined. The same idea establishes cases (D–E). A similar, though somewhat more complex (due to the _bilinear_ nature of attention) calculation establishes cases (B–C). Formalizing this reasoning essentially amounts to inductively proving cases (A–E); it will not be necessary to keep track of an explicit decomposition using _λ..._ coefficients; rather, one can mechanically verify these conditions inductively by plugging in definitions and applying the inductive hypothesis. 

**Formally proving the inductive step** consists in mechanically expanding definitions and applying the inductive hypothesis. First, (C) applied to layer _l −_ 1 entails that the attention logits for attention heads operating in layer _l_ match those of the original transformer. We start by establishing the inductive step for the pre-MLP activations _**Y** i_[(] _[l]_[)] . Starting from: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0063-07.png)


we show the inductive step first for (A) in the case of the pre-MLP activation _**Y** i_[(] _[l]_[)] . For _S_ satisfying _∀l[′] ∈{_ 1 _, . . . , L}_ : [( _**V** l[′] ,h[′] ∈S_ ) _⇒ l[′] > l_ ] (61) 

we consider 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0063-10.png)


where ˜ _ai,j_ denotes attention weights. The claim here now follows from the inductive hypothesis for (A): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0063-12.png)


63 

Preprint 

where the last step used (60). Next, we consider (B) and (C). First, for (B), assuming _T_ 1 _∪T_ 2 = _∅_ , we first find using (60) and the linearity of Λ _..._ and Ω _..._ (portions changed marked in blue): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0064-02.png)


Now the definition of _**V**_[�] _l,h_ allows us to rewrite this as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0064-04.png)


64 

Preprint 

In order to directly apply the inductive hypothesis, we rearrange the summations: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0065-02.png)


Note that, as above, _S_ 1, _S_ 2 are fixed and the sums run over _T_ 1 _, T_ 2. We can rewrite the above as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0065-04.png)


We are now ready to apply the inductive hypothesis: Directly plugging the inductive hypothesis for (B) into the second through fourth terms gives us: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0065-06.png)


65 

Preprint 

We now distinguish two cases, for proving (B) and (C). The first one is that _S_ 1 = _S_ 2 = _∅_ . In this case, by case (C) of the inductive hypothesis: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0066-02.png)


Now, applying (60) again, we rearrange to sums to obtain the conclusion 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0066-04.png)


proving (upon rearranging) the inductive step for (C) in the case of _**Y** i_[(] _[l]_[)] . In the second case, _S_ 1 _∪ S_ 2 = _∅_ ; here, we use case (B) of the inductive hypothesis to instead rewrite as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0066-06.png)


This proves the inductive step for (B) in the case of _**Y** i_[(] _[l]_[)] . We next address the inductive step for (D) in the case of the pre-MLP activation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0066-08.png)


By unfolding _**Y**_[�] _j_[(] _[l]_[)] using (60) and using the linearity of Ω _S_ 2 _,T_ 1 _,T_ 2 _,k,h,_ _**v** ,_ _**w**_ , the claim follows directly from the inductive hypothesis for (D). The same reasoning applies to (E). Overall, we have proven the inductive step (A–E) for the pre-MLP activations _**Y** i_[(] _[l]_[)] . 

We now need to show that the inductive step also holds for the post-MLP activations. Recall that the MLP acts as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0066-11.png)


66 

Preprint 

The proof proceeds by expanding this equation and reducing the claim to the already-proven inductive step for pre-MLP activations (for handling the direct contribution from the pre-MLP activation), and cases (D) and (E) (for handling the contributions of the MLP units). First, we note that, for each _l, s_ , by the case (A) of the inductive hypothesis and by the definition of _**b**_[�] _l_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0067-02.png)


We will abbreviate this number as Ξ _l,s,i ∈_ R. We now prove the case (A) of the inductive step for the post-MLP activation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0067-04.png)


To prove cases (B) and (C) for the post-MLP activations _**y** i_[(] _[l]_[)][, we now consider] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0067-06.png)


67 

Preprint 

We apply the inductive step for the pre-MLP activation in cases (D), and (E) to rewrite the second and third term, and apply the definition of _**B**_[�] _l_ to rewrite the fourth term: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0068-02.png)


In the case where _S_ 1 = _S_ 2 = _∅_ , we obtain using case (C) of the inductive hypothesis for the pre-MLP activation: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0068-04.png)


Using (64) and the definition of Ξ _l,s,i_ , this rewrites to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0068-06.png)


from which case (B) of the inductive hypothesis follows by rearranging. If instead _S_ 1 _∪S_ 2 = _∅_ , the same reasoning leads to case (C) of the inductive hypothesis. Analogous reasoning establishes cases (D) and (E) for the post-MLP activations. Overall, we have established the inductive step for cases (A–E) for the post-MLP activations. 

## G ADDITIONAL SUPPORTING RESULTS 

## G.1 REGULARIZER AT INITIALIZATION 

Here, we provide evidence that the additional regularizer (8) is bounded independently of _N_ under plausible initializations of parameters. Recall 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0068-11.png)


Intuitively, and as formalized in Proposition 54, when independently initializing the positional encodings _**p** i_ , their inner products as mediated through _**K** l,h[T]_ _**[Q]**[l,h]_[ will tend to be small.][As long as the] width grows linearly with _N_ , the aggregate value of (8) will tend to be bounded independently of _N_ . Note that (8) only includes products involving position 1, which due to translation invariance for _T ∈_ Θ _n_ places a bound on all products. As standard training does not enforce translation invariance of the products _**p**[T] i_ _**[K]** l,h[T]_ _**[Q]**[l,h]_ _**[p]**[j]_[,][one][may][also][be][interested][in][a][variant][that][takes][all][pairs][of] 

68 

Preprint 

positions into account, to the extent that they can enter causal attention: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0069-02.png)


Here, the same conclusion holds. We describe it formally, at the example of the second variant, in Proposition 54. 

**Proposition 54.** _Assume d_ = Θ( _N_ ) _. Assume the entries of each_ _**p**_ 1 _, . . . ,_ _**p** N ∈_ R _[d] and_ _**K** l,h[T]_ _**[Q]**[l,h][∈]_ R _[d][×][d] (l_ = 1 _, . . . , L; h_ = 1 _, . . . , H) are initialized i.i.d. from N_ (0 _, d_[1][)] _[.][The number of layers][ L][ and] heads H are constant with respect to N . Then_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0069-05.png)


_Proof._ We begin by showing that the expectation of each term in the sum is _O_ (1 _/d_ ) and hence the sum is bounded by a constant. There are two cases for the expectation of terms: (i) The first is _i ̸_ = _j_ when the vectors _**p** i_ and _**p** j_ are independent and the second is _i_ = _j_ when _**p** i_ and _**p** j_ are dependent. 

For this section, let _**K** ,_ _**Q**_ denote the matrices _**K** l,h_ _**Q** l,h_ for any fixed _l, h_ . Let _**K** ij_ and _**Q** ij_ denote the entry of the corresponding matrices _i_ th column and _j_ th row. Let _A_ = _**K**[T]_ _**Q** ∈_ R _[d][×][d]_ . Note that the expectation of any entry of _A_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0069-08.png)


Further, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0069-10.png)


For products of two different entries _**A** i,j_ _**A** mn_ , note that E[ _**A** i,j_ _**A** m,n_ ] = _d_ E[[�] _[d] u_ =1 � _v_ =1 _**[K]**[i,u]_ _**[Q]**[j,u]_ _**[K]**[m,v]_ _**[Q]**[n,v]_[)]][ which is][ 0][ when] _[ i][ ̸]_[=] _[ m]_[ or] _[ j]_[=] _[ n]_[.] For each term _|_ _**p**[T] i[A]_ _**[p]**[j][|]_[2][, we have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0069-12.png)


since all terms of the form E[ _**p** i,u_ _**p** i,v_ _**A** u,v_ _**A** m,n_ _**p** j,v_ _**p** j,n_ ] are 0 due to independence of _**A**_ and _**p**_ . For _i ̸_ = _j_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0069-14.png)


69 

Preprint 

For _i_ = _j_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0070-02.png)


Since _d_ = Θ( _N_ ) and each term is less than _d_[3][, we have that the sum in Eq.68 is] _[ O]_[(1)][.] 

## G.2 EMPIRICAL LENGTH GENERALIZATION OF POSITIONAL FUNCTIONS 

Here, we show empirically that, when directly fitting parameters so that a product _**p** i_ _**K**[T]_ _**Qp** j_ reproduces some function _ϕ_ ( _·, ·_ ) at smaller distances, it will length generalize when these are local or periodic but under different conditions matching the role of local and periodic functions in our theory. Specifically, we show that they length-generalize well at large _d_ when they are LOCAL; whereas, when _d_ is smaller, length generalization works well when they are PERIODIC. Length generalization is poor on functions that are neither local nor periodic. 

**Experimental Setup** We randomly initialize 200 position embeddings _{_ _**p** i ∈_ R _[d]_ : 1 _≤ i <_ 201 _}_ , as well as query and key matrices, _**Q** ,_ _**K** ∈_ R _[d][×][d]_ . We experiment with _d_ = _{_ 32 _,_ 256 _}_ . We optimize the mean square error (MSE) between _**p**[T] i_ _**[K]**[T]_ _**[ Qp]**[j]_[and] _[ϕ]_[(] _[·][,][ ·]_[)][on][length][of][50,][and][test][on][length] _{_ 50 _,_ 100 _,_ 150 _}_ . Concretely, during training, we sample random offsets _o_ from [0, 150] and take the sequence of _**p**_ 1+ _o, . . . ,_ _**p**_ 50+ _o_ to compute the loss. When testing on length _n_ , we compute the average loss over all offsets in [0, 200-n]. We ignore the loss on those entries where _j < i_ to mimic causal masking. The _ϕ_ ( _·, ·_ ) we use in experiments (except for the one combined from two _ϕ_ ( _·, ·_ )) only takes two values, 0 when the condition is false and 2 log 50 when the condition is true. We thus use different conditions to describe different _ϕ_ ( _·, ·_ ). For example, we use _ϕ_ : j=i-c to denote the following function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0070-07.png)


where _c_ is a constant number. 

The embeddings and weight matrices are trained with Adam optimizer, using a batch size of 64, a learning rate of 1e-3, for 15k steps. Additionally, we add mean squared weights (i.e., squared Frobenius norm divided by the number of elements) to the loss to mimic the training regularizer, with a coefficient of 0.01. 

Results are in Figure 5 and 6. Note that in both figures, the y-axis uses a logarithmic scale above 1.0 and stays linear scale below 1.0. In the last column of 6, “combined” denotes functions that combine two functions as follows: _ϕ_ = _ϕ_ 1 + _ϕ_ 2, where _ϕ_ 1 : j=i-c and _ϕ_ 2 : (i-j)= _c_ 2 mod _c_ 1. 

## G.3 BOUND FOR ENCODINGS NORM IN TERMS OF FUNCTION COMPLEXITY 

Recall that our regularizer includes a penalty (8) on attention dot products. Here, we discuss a conjecture: 

**Conjecture 55.** _The term (8) can be removed from the regularizer while maintaining a (potentially weaker) length generalization guarantee for Limit Transformers._ 

To provide a heuristic argument for this, assume that for each upper bound N on the input length, we have a configuration of positional encodings and the matrix _**A**_ , such that the following property holds: For any indices _N ≥ j > i >_ 0, let 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0070-15.png)


where _F_ : N _→_ R is a function that maps to numbers representable in _p_ -bit precision. The function _F_ and the precision _p_ are chosen globally, across the different N’s. Boundedness of _R_ ( _Tn_ ) across 

70 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0071-01.png)


Figure 5: Appendix G.2: MSE loss in fitting (length = 50) and generalizing (higher lengths) functions _ϕ_ ( _·, ·_ ) with products _**p**[T] j_ _**[K]**[T]_ _**[ Qp]**[i]_[. We show local functions testing if] _[ j]_[=] _[ i][−][c]_[ (left), if] _[ j][> i][−][c]_ (center), periodic functions testing whether _i−j ≡ c_ 2 (mod _c_ 1) (right). We show results at small (top, _d_ = 32) and high (bottom, _d_ = 256) dimensionality. Local functions length-generalize well when dimensionality is high (bottom left, bottom center); generalization is more successful with functions concentrated on few pairs (bottom left is nonzero at only one value of _j −i_ ; bottom center is nonzero at _c_ different values of _j − i_ ). Periodic functions length-generalize well when dimensionality is low (top right). The results match the distinct roles played by local and periodic functions in our theoretical constructions: Periodic functions are mediated by bounded-rank products (Lemma 48), local functions are mediated by the products _**p**[T]_ _**K** l,h[T]_ _**[Qp]**_[.] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0071-03.png)


Figure 6: Appendix G.2: MSE loss in fitting (length = 50) and generalizing (higher lengths) functions _ϕ_ ( _·, ·_ ) with products _**p**[T] j_ _**[K]**[T]_ _**[ Qp]**[i]_[.][We show functions that are neither local nor periodic, which] test if _j < i − c_ (left) if _i − j_ is a prime number (center), and a function created by adding a local and a periodic function (right). We show results at small (top, _d_ = 32) and high (bottom, _d_ = 256) dimensionality. Compared with results in Figure 5, we can see that such functions, neither local nor periodic, length-generalize poorly. 

71 

Preprint 

_n_ entails _∥_ _**p** i∥_ 2 _, ∥_ _**A** ∥ < C_ , for _C_ a global constant. We also know that sup _x∈_ N _|F_ ( _x_ ) _| < ∞_ . We conjecture that one can use these assumptions, and Lemma 56, to prove that F cannot be “too complicated”. Specifically, we conjecture that _F_ will be **ultimately periodic** : when _x_ exceeds some threshold, _F_ ( _x_ + ∆) = _F_ ( _x_ ) for some period ∆. For, if _F_ is not ultimately periodic, we hope to construct a matrix _G_ whose nuclear norm can be made arbitrarily large, so large as to give a superconstant lower bound on _∥_ _**A** ∥_ – which is a contradiction. First note that Lemma 56 even holds if _**G**_ = _**Y**[T]_ _**AX**_ where _**X** ,_ _**Y**_ are two different matrices with _n_ unit-norm columns. That is, we can consider a matrix _**G** ij_ = _**p**[T] xi_ _**[Ap][y]** j_[where we conjecture that one can choose] _**[ x]**_[1] _[, ...,]_ _**[ x]**[n]_ and _**y**_ 1 _, ...,_ _**y** n_ to give an arbitrarily large lower bound on _∥_ _**A** ∥_ , under the assumption that _F_ is not ultimately periodic. Here, it is important that _F_ maps to fixed-precision output; otherwise, one could get functions that have irrational periods and thus are not periodic when restricted to N. 

**Lemma 56.** _Let x_ 1 _, . . . , xn ∈_ R _[d] be vectors with ∥xi∥_ 2 = 1 _, and let A ∈_ R _[d][×][d] arbitrary. Let G ∈_ R _[n][×][n] such that Gij_ = _x[T] i[Ax][j][.][Then]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0072-03.png)


_where ∥A∥ denotes the spectral norm._ 

_Proof._ First, note that for any matrix _B_ = _U_ Σ _V_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0072-06.png)


where _V[T] U[T]_ has each singular value bounded by 1 (in fact, it’s orthogonal). In other words, _∥B∥∗_ = _tr_ ( _BM_ ) where _M_ is an orthogonal matrix. 

For any two real-valued and possibly non-square matrices, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0072-09.png)


and by submultiplicativity, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0072-11.png)


Using Eq. 72 and the above two properties, it follows that, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers_images/iclr_2025_A_Formal_Framework_for_Understanding_Length_Generalization_in_Transformers.pdf-0072-13.png)


72 

