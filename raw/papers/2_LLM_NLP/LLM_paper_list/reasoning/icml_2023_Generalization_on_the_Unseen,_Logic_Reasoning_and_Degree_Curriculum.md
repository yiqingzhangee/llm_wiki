Journal of Machine Learning Research 25 (2024) 1-58 

Submitted 2/24; Published 11/24 

## **Generalization on the Unseen, Logic Reasoning and Degree Curriculum** 

## **Emmanuel Abbe** 

emmanuel.abbe@epfl.ch 

_EPFL, Apple, Lausanne, VD, Switzerland_ 

**Samy Bengio** _Apple, Cupertino, CA, USA_ 

**Aryo Lotfi** _EPFL, Lausanne, VD, Switzerland_ 

**Kevin Rizk** 

bengio@apple.com 

aryo.lotfi@epfl.ch 

kevin.rizk@epfl.ch 

_EPFL, Lausanne, VD, Switzerland_ 

**Editor:** Kilian Weinberger 

## **Abstract** 

This paper considers the learning of logical (Boolean) functions with a focus on the _generalization on the unseen (GOTU)_ setting, a strong case of out-of-distribution generalization. This is motivated by the fact that the rich combinatorial nature of data in certain reasoning tasks (e.g., arithmetic/logic) makes representative data sampling challenging, and learning successfully under GOTU gives a first vignette of an ‘extrapolating’ or ‘reasoning’ learner. We study how different network architectures trained by (S)GD perform under GOTU and provide both theoretical and experimental evidence that for sparse functions and a class of network models including instances of Transformers, random features models, and linear networks, a _min-degree-interpolator_ is learned on the unseen. More specifically, this means an interpolator of the training data that has minimal Fourier mass on the higher degree basis elements. These findings lead to two implications: (1) we provide an explanation to the _length generalization problem_ for Boolean functions (e.g., Anil et al. 2022); (2) we introduce a curriculum learning algorithm called _Degree-Curriculum_ that learns monomials more efficiently by incrementing supports. Finally, we discuss extensions to other models or non-sparse regimes where the min-degree bias may still occur or fade, as well as how it can be potentially corrected when undesirable. 

**Keywords:** reasoning, out-of-distribution generalization, implicit bias, length generalization, curriculum learning 

## **1 Introduction** 

Neural networks trained by stochastic gradient descent (SGD) have proved to be a powerful learning paradigm when there is enough representative data about the distribution to be learned, specifically in applications involving images or text where there is also a good understanding of the relevant architectures. 

> ©2024 Emmanuel Abbe, Samy Bengio, Aryo Lotfi, and Kevin Rizk. 

> License: CC-BY 4.0, see `https://creativecommons.org/licenses/by/4.0/` . Attribution requirements are provided at `http://jmlr.org/papers/v25/24-0220.html` . 

Abbe, Bengio, Lotfi, and Rizk 

There is now an increasing interest in tackling tasks involving more ‘reasoning’ components, which depart from classical perception tasks of images and texts. While such tasks remain vaguely defined, a list that we consider here under this class is given by: (1) arithmetic and algebra (Saxton et al., 2019; Lewkowycz et al., 2022), (2) synthetic tasks such as PVR (Zhang et al., 2021) and LEGO (Zhang et al., 2022), (3) visual reasoning such as CLEVR (Johnson et al., 2017), (4) physical reasoning such as Phyre (Bakhtin et al., 2019), (5) algorithmic data such as CLRS (Veliˇckovi´c et al., 2022) and reasoning on graphs (Mahdavi et al., 2023). 

One common trademark of these tasks is that the input space is usually of a discrete and combinatorial nature, and consequently, the data may not necessarily lay on a lowdimensional manifold that is well sampled. In various cases, the input space may even have a variable length. This combinatorial nature is already present in text, but it is further amplified in, say, arithmetic since most symbol combinations could a priori represent a valid input (in contrast to text). Further, the target function in such tasks may rely on a large composition of logical steps or mathematical operations that require to be jointly learned. Therefore, in such reasoning tasks, the setting with abundant representative data seems less prominent. This motivates us to focus on a strong out-of-distribution (OOD) generalization setting. 

For instance, when learning arithmetic or logic functions on a training set with a bounded length or a limited number of truth assignments, how would the neural network generalize on more general input assignments? (This is a case of length generalization.) When training a neural network to learn a Boolean formula, such as a voting scheme on data from a polarized cohort of voters, how does the network generalize to an unpolarized cohort? 

We thus consider the problem of learning functions with a holdout domain where part of the distribution support is barely/never seen at training, and with target functions that are Boolean to capture the discrete and combinatorial nature of various reasoning tasks (e.g., arithmetic, decision trees, logical circuits). Learning successfully under holdout gives a first vignette that the learner is operating with a certain amount of ‘reasoning’ or ‘extrapolation’ since memorization is voided on the unseen domain. 

## **1.1 Our Main Contributions** 

1. We lay down some basic principles of stronger generalization requirements that rely on the ‘generalization on the unseen (GOTU)’ performance metric, defined as a strong case of OOD generalization to measure ‘extrapolating’ or ‘reasoning’ properties of models on considered tasks. 

2. We study how standard neural network architectures trained by (S)GD perform on the GOTU metric, in particular, which solutions are learned on the unseen domain for such architectures: 

   - (i) we prove three theoretical results showing that for a class of network models including random features model (Theorem 11), deep diagonal linear networks (Theorem 15), and 2-layer fully connected linear neural networks (Theorem 18), a _min-degreeinterpolator (MD interpolator)_ is learned on the unseen; 

   - (ii) we show experimental results (Section 4) supporting that encoder-only Transformers tend to also have the _min-degree bias (MD bias)_ towards MD interpolators. 

2 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

The MD interpolator is defined as the interpolator of minimal _degree-profile_ , i.e., the Boolean function interpolating the training data and having a Fourier-Walsh transform whose energy concentrates on basis elements of lowest possible degree. Connections to algebraic geometry are given in Appendix C in order to characterize how MD interpolators can be constructed from the ‘vanishing ideal’ of the seen data. We also point out that very large learning rates or other architectures (such as meanfield networks) can exhibit leaky MD bias (i.e., assigning larger mass on higher-degree monomials); see Appendix B.2.[1] 

3. Using these, we obtain two additional results: (i) we provide a formal explanation (Theorem 22) to the ‘length generalization problem’ discussed in the work of Anil et al. (2022) (for the case of bounded weight vectors, also related to the work of Zhang et al., 2022); 

   - (ii) we turn the min-degree bias into an asset to accelerate learning by introducing a curriculum learning algorithm called ‘Degree-Curriculum’ (Algorithm 1), which successively increases the input complexity with respect to Hamming weights in order to incrementally learn the monomials support (see Section 6). 

4. Finally, we provide experimental results on the role of the sparsity condition (Section 7.1) and the role of the causal attention mask in Transformers (Section 7.2). We show that the min-degree bias may still be present or fade depending on these conditions. We also demonstrate how the min-degree bias can be mitigated when undesirable using symmetry-based regularizers in Section 9. 

This work is an extension of earlier work (Abbe et al., 2023a). In this extension, we have added the theoretical analysis for 2-layer fully connected linear neural networks (Theorem 18) along with experimental results (in Appendix B) for the general case conjectured in Conjecture 17. We also investigated relaxing the sparsity condition in Section 7.1 and using Transformers with causal attention masking in Section 7.2. We have also included a preliminary attempt to tame the min-degree bias when undesirable using a symmetry-based regularization term in Section 9. 

## **2 Generalization on the Unseen** 

The classical setting of statistical learning theory requires the control of three error pillars for the generalization of a learning model: (1) the approximation error (depending on the properties/richness of the model class), (2) the estimation error (depending on the properties/richness of the training set), (3) the optimization error (depending on the properties/richness of the training algorithm). 

In some of the recent deep learning applications for computer vision and natural language processing, the richness of the training set, the size of the model and its alignment with the data, as well as the computational power, make the three pillars well controlled. The recent success of large language models (LLM) and scaling laws are perfect examples of this phenomenon (Alabdulmohsin et al., 2022). 

> 1. Our code is available at `https://github.com/aryol/GOTU` 

3 

Abbe, Bengio, Lotfi, and Rizk 

As mentioned in the introduction, the type of data occurring in reasoning tasks is slightly different due to the richness and combinatorial nature of the data. To better cope with this challenge, we propose in this paper to depart from the classical generalization objectives described with the three pillars. We focus instead upfront on distribution shift and, more specifically, a strong case of OOD generalization where part of the distribution domain is almost/completely unseen at training but used at testing (in particular, prohibiting any memorization scheme). 

Of course, on the unseen domain, all bets are off for generalization: one cannot hope for an algorithm trained on a given data domain to perform well on a larger data domain without any incentive to do so. Yet various algorithms will have various implicit biases on the unseen and thus produce various solutions on the unseen. Understanding this ‘bias on the unseen’ for different network architectures and Boolean target functions is the objective of this paper. 

We start by redefining the generalization error when the train and test distribution are not necessarily the same. 

**Definition 1** _Let X_ 1 _, . . . , Xm be drawn i.i.d. under µ_ 1 _and labeled by a target function f , and let f_[˜] _be the function learned by a learning algorithm. The algorithm has_ ( _µ_ 1 _, µ_ 2 _, m, ϵ_ ) _- generalization (for loss ℓ) if_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0004-05.png)


_In other words, the algorithm is trained under distribution µ_ 1 _and tested under distribution µ_ 2 _, producing ϵ-test-loss with sample complexity m._ 

Now we focus on a special case of interest, a strong case of OOD generalization where we essentially see all the data on some part of the domain but miss another part. Naturally, we will next study a ‘soft version’ of this metric, where both in-distribution and out-ofdistribution generalization are considered, but this strong case is already rich and insightful. 

**Definition 2 (Generalization on the Unseen)** _Consider a given sample domain_ Ω _. During training, part of_ Ω _is not sampled, and we call this the unseen domain (or the holdout set) U. At testing, however, we sample from the full set_ Ω _. This represents a special case of the previous definition where µ_ 1 = _µ|_ Ω _\U and µ_ 2 = _µ|_ Ω _for some µ._ 

_We now further specify the setting: we assume that the training error is 0 on the training domain_ Ω _\ U, e.g., by seeing all the samples in_ Ω _\ U, and define the generalization on the unseen (GOTU) for an algorithm f_[˜] _and target function f as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0004-10.png)


_where ∼U U indicates uniform sampling from U. Notice we only sample on U at testing because we assumed zero training error and considered the whole_ Ω _\ U as the training set._ 

A few remarks are in order: 

- GOTU is a special case of OOD and distribution shift setting that is extremal in the sense that it completely gives access to part of the distribution domain and completely omits the complement. Since we consider rich enough models to interpolate the data, 

4 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

the ‘statistical’ and ‘approximation’ pillars of the learning problem are removed (there may still be randomness used by the learning algorithm, thus statistical analysis may still be relevant). The problem thus turns into a pure optimization problem where the central object of study is the implicit bias of the learning algorithm on the unseen. Note that this is not exactly the same implicit bias as studied in the setting of overparametrized models (Soudry et al., 2018; Gunasekar et al., 2017, 2018b; Arora et al., 2019; Razin and Cohen, 2020; Chizat and Bach, 2020; Moroshko et al., 2020) as here we have the distribution shift and investigate the behavior of the equivalence class of interpolators on the unseen _U_ . 

- In some experiments, we replace the ‘perfect’ training data on the seen domain with a ‘large’ sampling on the seen domain. We defined the GOTU in the extreme case to simplify the number of parameters to track and to allow for cleaner theorem statements, but there could also be a sampling rate on Ω _\U_ ; this is left for future research. Also, we assume a uniform prior here because this is a natural first case for arithmetic/logic tasks, but this could also be relaxed. 

- We will consider different subsets _U_ in the applications. We are sometimes interested in _U_ s for which the data invariances or equivariances could give hope to learn. This is further specified with the next definition. 

**Definition 3** _A function f_ : Ω _→_ R _is (1) G-invariant, or invariant under the group action G on_ Ω _, if f_ ( _gx_ ) = _f_ ( _x_ ) _for all g ∈ G, x ∈_ Ω _; (2) Gi,o-equivariant, or equivariant under Gi,o, if f_ ( _gix_ ) = _gof_ ( _x_ ) _for all_ ( _gi, go_ ) _∈ Gi,o and x ∈_ Ω _._[2] 

As stated earlier, we cannot expect algorithms to generalize on the unseen domain by themselves. However, we can hope that certain training algorithms will catch invariances/equivariances and thus extrapolate. For example, consider the parity function on _d_ bits _x_ 1 _, . . . , xd ∈{±_ 1 _}_ defined as _f_ ( _x_ 1 _, . . . , xd_ ) = _x_ 1 _x_ 2 _· · · xd_ . This function is permutationinvariant (group _G_ = _Sd_ ). In particular, if one uses a model favoring permutation symmetries, one may not have to see all inputs that are permutation equivalent. There has been a series of works designing layers/architectures that are equivalent under a prespecified family of actions (e.g., all permutations) (see Ravanbakhsh et al., 2017; Zaheer et al., 2017; Hartford et al., 2018). More recently, Zhou et al. (2021) propose a method to learn invariances in a multi-task setting using meta-learning. An example of an equivariant Boolean function would be the majority function on _{_ +1 _, −_ 1 _}[d]_ , _d_ odd, with the action of global bit flipping on the input and the output (since the majority is reversed if all the bits are flipped). Thus a holdout on vectors of dual-weight could again be handled by a model having such an equivariance. Note that we are also interested in cases where these equi/in-variances are not present in the target, to understand what solutions neural networks favor on the unseen. 

## **3 Results** 

We consider _f_ : Ω _→_ R with Ω= _{±_ 1 _}[d]_ . We use [ _d_ ] to denote _{_ 1 _,_ 2 _, . . . , d}_ . We introduce some preliminary material on Boolean functions in the next part and then state our results. 

> 2. See the works of Dummit and Foote (2004); Ravanbakhsh et al. (2017); Zhou et al. (2021) for more details on group actions and the definitions of invariance and equivariance. 

5 

Abbe, Bengio, Lotfi, and Rizk 

## **3.1 Preliminaries** 

**Fourier-Walsh Transform** Any function _f_ : _{±_ 1 _}[d] −→_ R can be expressed as _f_ ( _x_ ) = � _T ⊆_ [ _d_ ] _[f]_[ˆ][(] _[T]_[)] _[χ][T]_[ (] _[x]_[)][where] _[χ][T]_[ (] _[x]_[)][=][�] _i∈T[x][i]_[and] _[f]_[ˆ][(] _[T]_[)][=][E] _X∼U {±_ 1 _}[d]_[[] _[χ][T]_[ (] _[X]_[)] _[f]_[(] _[X]_[)]][are][the] monomials and the coefficients respectively (O’Donnell, 2014). For example, the majority function on 3 bits can be written as Maj( _x_ 1 _, x_ 2 _, x_ 3) =[1] 2[(] _[x]_[1][ +] _[ x]_[2][ +] _[ x]_[3] _[ −][x]_[1] _[x]_[2] _[x]_[3][).] 

**Unseen Domain and Vanishing Ideals** We now introduce the unseen domain _U_ . First, consider the canonical holdout (Abbe et al., 2022a), when a bit is frozen during training, e.g., _xi_ = 1 and _U_ = _{x ∈{±_ 1 _}[d]_ : _xi_ = _−_ 1 _}_ . In this case, one can see that any function of the form _f_ ( _x_ )+(1 _−xi_ )∆( _x_ ) (∆( _x_ ) is arbitrary) is an equivalent interpolator on the training data. For general unseen domain _U ⊆_ Ω= _{±_ 1 _}[d]_ , there exist polynomials _v_ 1( _x_ ) _, . . . , vk_ ( _x_ ) such that _x ∈_ Ω _\ U ⇐⇒ v_ 1( _x_ ) = _. . ._ = _vk_ ( _x_ ) = 0 (see Appendix C). Consequently, all solutions of the form _f_ ( _x_ ) + ∆1( _x_ ) _v_ 1( _x_ ) + _· · ·_ + ∆ _k_ ( _x_ ) _vk_ ( _x_ ) are equivalent at training. This is the quotient space of _f_ under the vanishing ideal defined by Ω _\ U_ . We refer to Appendix C for more details on this relation to algebraic geometry. 

We now define measures of complexity relevant to us. 

**Definition 4 (Degree)** _For a function f_ : _{±_ 1 _}[d] →_ R _, the degree_ deg( _f_ ) _refers to the maximum degree of the monomials present in the Fourier-Walsh transform of f . For example, the degree of_ Maj( _x_ 1 _, x_ 2 _, x_ 3) =[1] 2[(] _[x]_[1][ +] _[ x]_[2][ +] _[ x]_[3] _[ −][x]_[1] _[x]_[2] _[x]_[3][)] _[is][3.]_ 

**Definition 5 (Degree profile)** _For f_ : _{±_ 1 _}[d] →_ R _, we define the degree-profile of f ,_ DegP( _f_ ) _∈_ R _[d]_[+1] _such that_ DegP( _f_ ) _i_ =[�] _T ⊆_ [ _d_ ] _,|T |_ = _d_ +1 _−i[f]_[ˆ][(] _[T]_[)][2] _[for]_[1] _[≤][i][≤][d]_[ + 1] _[.][Fur-] thermore, we consider lexicographic ordering on these vectors, i.e.,_ DegP( _f_ ) _<_ DegP( _g_ ) _iff ∃i_ DegP( _f_ ) _i <_ DegP( _g_ ) _i and_ DegP( _f_ ) _j_ = DegP( _g_ ) _j_ 1 _≤ j < i. For instance, the degree-profile of_ Maj( _x_ 1 _, x_ 2 _, x_ 3) =[1] 2[(] _[x]_[1][ +] _[ x]_[2][ +] _[ x]_[3] _[ −][x]_[1] _[x]_[2] _[x]_[3][)] _[is]_[(1] _[/]_[4] _[,]_[ 0] _[,]_[ 3] _[/]_[4] _[,]_[ 0)] _[.]_ 

Intuitively, DegP( _f_ ) _i_ represents the Fourier mass on degree-( _d_ + 1 _− i_ ) monomials (for 1 _≤ i ≤ d_ + 1) and DegP( _f_ ) captures the spectrum of _f_ . Note that the degree-profile is a stronger notion than the degree, i.e., deg( _f_ ) _<_ deg( _g_ ) = _⇒_ DegP( _f_ ) _<_ DegP( _g_ ). 

**Definition 6 (Min-degree interpolators)** _Consider a target function f and unseen domain U. The set of interpolators is defined as F_ int( _f, U_ ) = _{g_ : _{±_ 1 _}[d] →_ R _| g_ ( _x_ ) = _f_ ( _x_ ) _, ∀x ∈U[c] }, where U[c]_ := Ω _\U is the seen domain. We call an interpolator a min-degree interpolator (MD interpolator) of_ ( _f, U_ ) _(or of {x, f_ ( _x_ ) _}x∈U c) if it is an element of F_ int( _f, U_ ) _that minimizes the degree-profile with respect to the lexicographic order. This means that no part of the Fourier-Walsh expansion of the interpolator could be replaced with a lower-degree alternative and still interpolate._ 

For example, consider the case of ‘canonical holdout’ where we always have _x_ 1 = 1 at training, i.e., _U_ = _{x ∈{±_ 1 _}[d]_ : _x_ 1 = _−_ 1 _}_ , and target function _x_ 1 _x_ 2 + _x_ 1 _x_ 3 _x_ 4. Here, both _x_ 1 _x_ 2 + _x_ 3 _x_ 4 and _x_ 2 + _x_ 3 _x_ 4 are of degree 2 but only _x_ 2 + _x_ 3 _x_ 4 is an MD interpolator since _x_ 1 _x_ 2 in the first function is replaceable with the lower-degree _x_ 2. Further, note that there may be multiple interpolators having minimal max-degree rather than degree-profile. For example, consider the unseen domain induced by _xi_ = _xj_ and target _f_ ( _x_ ) = _xi_ + _xj_ . Then 2 _xi_ and _xi_ + _xj_ are both interpolators with minimal max-degree, but only _xi_ + _xj_ is an 

6 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

interpolator with a minimal degree-profile. In fact, the MD interpolator is always unique (if _f_ 1 and _f_ 2 are interpolators with the same degree-profile, then _[f]_[1][+] 2 _[f]_[2] is an interpolator with a smaller degree-profile unless _f_ 1 = _f_ 2.) 

## **3.2 Main Theoretical Results** 

We show that certain models have a min-degree implicit bias on the unseen. We start by giving another example. 

## 3.2.1 Result Preview from an Example 

Consider trying to learn the majority target function on 3 voters _x_ 1 _, x_ 2 _, x_ 3 having the following data distribution: voters 1 and 2 never vote both negatively, i.e., ( _x_ 1 _, x_ 2) is never ( _−_ 1 _, −_ 1) in the training data. Now train a neural network to learn the target on such a training data distribution (with only 3 variables, one will quickly see all sequences satisfying the required condition; this is to simplify the example, in our results, we consider higher dimensional versions of such examples). Since we always have ( _x_ 1 _, x_ 2) _̸_ = ( _−_ 1 _, −_ 1), it must be the case that (1 _− x_ 1)(1 _− x_ 2) = 0 (this ensures that either _x_ 1 or _x_ 2 must be equal to 1). Thus, the functions _f_ ( _x_ ) or _f_ ( _x_ )+∆( _x_ )(1 _− x_ 1)(1 _− x_ 2) (for any arbitrary ∆) are equivalent on the training data. One can thus wonder which ∆function will a neural network trained by (S)GD converge to. There is no reason to expect that it will converge to ∆= 0; so can we characterize which ∆will occur? 

Our main results show that —(i) provably for random features model or diagonal/2-layer linear networks in the linear case (three architectures that we can analyze rigorously), and (ii) empirically for encoder-only Transformers — (S)GD will converge to a ∆that makes _f_ ( _x_ ) + ∆( _x_ )(1 _− x_ 1)(1 _− x_ 2) having the lowest ‘degree-profile’ (see Definition 5), which in the above majority example is obtained as follows: first expand the target in the basis of multivariate monomials, Maj( _x_ 1 _, x_ 2 _, x_ 3) = ( _x_ 1 + _x_ 2 + _x_ 3 _− x_ 1 _x_ 2 _x_ 3) _/_ 2, then find ∆( _x_ ) that makes ( _x_ 1 + _x_ 2 + _x_ 3 _− x_ 1 _x_ 2 _x_ 3) _/_ 2 + ∆( _x_ )(1 _− x_ 1)(1 _− x_ 2) having the least _ℓ_ 2 mass on the highest degree monomials, i.e., in this case, ∆( _x_ ) = _x_ 3 _/_ 2, giving ( _x_ 1 + _x_ 2 + _x_ 3 _− x_ 1 _x_ 2 _x_ 3) _/_ 2 + ∆( _x_ )(1 _− x_ 1)(1 _− x_ 2) = ( _x_ 1 + _x_ 2 + 2 _x_ 3 _− x_ 1 _x_ 3 _− x_ 2 _x_ 3) _/_ 2 which is degree 2 rather than 3 (see Figure 10 for numerical experiments). This paper describes what are the general mathematical concepts behind this specific example: (i) Fourier-Walsh Boolean analysis, (ii) the notion of vanishing ideal, and (iii) minimal degree-profile interpolators and the implicit bias of neural networks towards them. To begin with, we formalize the concept of unseen domains using the following definition. 

**Definition 7** _We consider a P -dimensional latent function h_ : _{±_ 1 _}[P] →_ R _embedded in ambient dimension d. More precisely, we consider learning f_ : _{±_ 1 _}[d] →_ R _such that f_ ( _x_ ) = _h_ ( _xi_ 1 _, . . . , xiP_ ) _. We further denote I_ = _{i_ 1 _, . . . , iP } and xI_ = ( _xi_ 1 _, . . . , xiP_ ) _. We also assume that some specific combinations of xI are not present in the training samples, i.e., xI ∈U/[∗] ⊂{±_ 1 _}[P] and define the unseen domain as U_ = _{x ∈{±_ 1 _}[d] | xI ∈U[∗] }._ 

Note that considering sparse functions enables us to define the unseen domain properly and differentiate between the unseen domain (where there are minimal structures) and unseen data (for example when there is uniform sampling). In the next section, we present our results on learning sparse Boolean functions with the random features model. 

7 

Abbe, Bengio, Lotfi, and Rizk 

## 3.2.2 Results for Random Features Model 

Our first result is for the random features (RF) model (Rahimi and Recht, 2007). The RF model was initially introduced to approximate kernels and enhance the time complexity of kernel methods (Rahimi and Recht, 2007). RF models can also be viewed as approximations of neural networks in the NTK regime (Jacot et al., 2018; Ghorbani et al., 2019; Mei and Montanari, 2022). In this paper, we take the latter view on them as well, with the following formulation. 

**Definition 8 (Random features model)** _Consider x ∈_ R _[d] as the input; we define random features model with N random features as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0008-04.png)


_where ai ∈_ R _are the trainable parameters, σ is the activation function, and wi, bi ∼ N_ (0 _, d_[1][)] _[⊗][d][ ⊗N]_[(0] _[,] d_[1][)] _[are][the][random][weights][and][biases.][We][use][ϕ][i]_[(] _[x]_[) :=] _[ σ]_[(] _[⟨][w][i][, x][⟩]_[+] _[ b][i]_[)] _[as] a shorthand notation for the i-th feature._ 

The following activation property strengthens the condition presented in the work of Abbe et al. (2022c). 

**Definition 9 (Strongly expressive)** _We call a continuous activation function σ_ : R _→_ R _strongly expressive up to P if (A1) σ satisfies upper bound_ E _g∼N_ (0 _,_ 2)[ _σ_ ( _g_ )[4] ] _< ∞; and (A2) ∀T ⊆_ [ _d_ ] _, |T | ≤ P_ E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _T_ )[2] ] = Ω _d_ ( _d[−|][T][|]_ ) _, where ϕ_[ˆ] _w,b_ ( _T_ ) := E _x_ [ _σ_ ( _⟨w, x⟩_ + _b_ ) _χT_ ( _x_ )] _is the Fourier coefficient of T in the random feature created by w, b._ 

As will be proven in Lemma 23, property (A1) implies E[ _ϕ_[ˆ] _w,b_ ( _T_ )[2] ] = _O_ ( _d[−|][T][|]_ ) for _|T |_ = _Od_ (1). Therefore, the second condition (A2) is ensuring that the model is able to strongly express degree _k ≤ P_ monomials. 

We note that _ϕ_[ˆ] _w,b_ ( _T_ )[2] has been studied in the work of Abbe et al. (2022c) as the initial alignment (INAL) between monomial _χT_ ( _x_ ) and _ϕw,b_ ( _x_ ). Indeed, based on Lemma A.2. of Abbe et al. (2022c), the following conditions give us a family of strongly expressive activation functions. 

**Lemma 10** _Any continuous polynomially-bounded function σ such that its first P coefficients in the Hermite expansion are non-zero is strongly expressive up to P ._ 

For example, polynomial activation functions such as (1 + _x_ ) _[k]_ are strongly expressive up to _k_ . 

**Theorem 11** _Let f_ : _{±_ 1 _}[d] →_ R _be a P_ = _Od_ (1) _-sparse function to be learned in the GOTU setting (Definition 7) by a random features model with parameters_ ( _N, σ, a, b, w_ ) _(Definition 8) with a strongly expressive activation function. As N diverges, the random features model can interpolate the training data with high probability. Furthermore, defining f_ RF _[d,N]_[(] _[U]_[)] _[to][be][the][interpolating][solution][minimizing][∥][a][∥]_[2] _[(i.e.,][the][solution][reached][by] gradient descent/flow starting from a_ = 0 _under ℓ_ 2 _loss), we have w.h.p._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0008-13.png)


8 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

_where_ MinDegInterp( _f, U_ ) _is the min-degree interpolator on the training data {x, f_ ( _x_ ) _}x∈U c and ϵd is a function on P variables that tends pointwise to 0 as d diverges. (We refer to the above as a ‘min-degree bias’ or ‘MD bias’.)_ 

_Proof Sketch._ In Lemma 23, we show that random features generated by a strongly expressive _σ_ have in general a decaying degree-profile with E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _T_ )[2] ] = Θ( _d[−|][T][|]_ ) for _|T | ≤ P_ . We then investigate the interpolators in the Fourier-Walsh basis and show that the minimality condition of _∥a∥_ 2 is equivalent to learning the minimal degree-profile interpolator since high-degree monomials are less expressed in the features and consequently larger _∥a∥_ ’s are required to capture them. The full proof relies on concentration results and Boolean Fourier analysis and is given in Appendix A. 

**Corollary 12** _For a random features model with an activation function that satisfies property (A1) in Definition 9 and with diverging number of neurons (N →∞), we have_ E _x_ [ _f_ RF( _x_ ; _a, w, b_ ) _χT_ ( _x_ )][2] = _Od_ ( _[∥] d[a][|][T][∥][ |]_[2][)] _[for][|][T][|]_[=] _[O][d]_[(1)] _[.][Put][simply,][the][coefficient][of][de-] gree k_ = _Od_ (1) _monomials is bounded by O_ ( _[∥][a][∥] k_[)] _[.]_ ~~_√_~~ _d_ 

This corollary is also proved in Appendix A. As a result, to learn a solution of degree _k_ = _Od_ (1), the squared norm of the model’s weights, _∥a∥_[2] , must be at least Ω( _d[k]_ ). 

**Remark 13 (Other activation functions)** _Note that Theorem 11 does not hold for any arbitrary activation function. For example, if the activation function is σ_ ( _z_ ) = _z_[2] _, one can easily see that_ E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _x_ )( _{i}_ )[2] ] _,_ E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _x_ )( _{i, j}_ )[2] ] _∈_ Θ _d_ ( _d[−]_[2] ) _, and hence degree1 monomials have no priority over degree-2 monomials. An important case is the ReLU activation. Results of Abbe et al. (2022c) show that for the ReLU activation and |T |_ = _Od_ (1) _, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0009-06.png)


_Consequently, the min-degree bias still exists, but in a weaker form. For further discussion and experiments on the ReLU activation refer to Appendix A._ 

In the experiments, we show that having the sparsity assumption may not be necessary in some cases, and the min-degree bias can be observed for small values of _d_ and _N_ as well. Furthermore, we show that the min-degree bias goes beyond the random features and NTK models; see Section 4. 

We next focus on linear neural networks where we will be able to analyze non-linear dynamics for gradient flow. 

## 3.2.3 Results for Linear Neural Networks 

In this section, we study the min-degree bias in linear neural networks. First, note that in the case of linear functions, replacing a degree-1 variable _xk_ with the degree-0 variable 1 is the only case of lower degree bias. In other words, we consider the case that unseen data is _U_ = _{x | xk_ = _−_ 1 _}_ (referred to canonical holdout in work of Abbe et al., 2022a). We conjecture that linear neural networks trained with gradient flow learn the min-degree interpolator with a leakage factor (coefficient of the frozen variable in the learned function) 

9 

Abbe, Bengio, Lotfi, and Rizk 

that vanishes as their initialization scale goes to 0. We prove this conjecture formally for diagonal linear neural networks and two-layer fully connected networks. Further, we discuss how the proof can be generalized and provide experiments to support the conjecture. 

We start with the simpler case of diagonal linear neural networks with bias. We define them as follows. 

**Definition 14 (Diagonal linear neural network with bias)** _We define a diagonal linear neural network (DLNN) with bias as an extension of diagonal neural networks, where there is only one parameter for bias at the last layer. I.e.,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0010-04.png)


_where θ, d, and L represent the model’s parameters, input dimension, and depth, respectively._ 

**Theorem 15** _Let f_ : _{±_ 1 _}[d] →_ R _be a linear function f_ ( _x_ 1 _, . . . , xd_ ) = _f_[ˆ] ( _∅_ )+[�] _[d] i_ =1 _[f]_[ˆ][(] _[{][i][}]_[)] _[x][i][.] Consider learning this function using gradient flow on a diagonal neural network (where depth L ≥_ 2 _) while the k-th component is frozen at training (the canonical holdout setting with U_ = _{x ∈{±_ 1 _}[d] | xk_ = _−_ 1 _}). For any ϵ >_ 0 _, there exists an αmax (increasing with L) such that if all the model’s parameters are initialized i.i.d. under the uniform distribution U_ ( _−α, α_ ) _for any_ 0 _< α ≤_ min _{α_ max _,_[1] 2 _[}][, then, with probability 1, the training loss converges] to 0, and the coefficient of the learned function f_ NN _on the high-degree monomial xk is less than ϵ, i.e., f_[ˆ] NN( _{k}_ ) _≤ ϵ._ 

_Proof Sketch._ We prove this theorem by analyzing the trajectory of gradient flow on the parameters. Primarily, we show the convergence of the model. Note that _f_[ˆ] NN( _{k}_ ) _≤ ϵ_ is equivalent to _xk_ being ignored by the neural network, i.e., the frozen variable _xk_ not contributing to the bias learned by the neural network. We pursue the proof in two steps. As the first step, we show there exists a time _Tϵ_ such that the bias is almost learned by the bias parameter and the rest of the parameters and the role of _xk_ = 1 are still small (note that this point is close to a saddle). For the second step, we show that the contribution of _xk_ = 1 to the bias will not change much throughout the training process. 

Next, we focus on the general case of fully connected linear neural networks. We parameterize them with the definition below. 

**Definition 16** _We define a fully connected linear neural network with depth L as follows_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0010-10.png)


_where bi ∈_ R _[d][i] , Wi ∈_ R _[d][i][−]_[1] _[×][d][i] are the weights and biases of the i-th layer. Note that d_ 0 = _d is the input dimension and dL_ = 1 _is the output dimension. Also, sometimes we slightly abuse the notation by referring to the last layer’s weight vector by WL_ = _wL._ 

10 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

Now again assume that _xk_ is frozen to 1 during training. Denote the first layer’s weights connected to _xk_ by _W_ 1 _,k_ . One can easily see that the weights connected to the frozen coordinate act exactly similar to the biases of the first layer and they follow the same dynamics. More precisely, _∇W_ 1 _,k L_ ( _t_ ) = _∇b_ 1 _L_ ( _t_ ), where _L_ ( _t_ ) is the loss function. As a result, they have the same updates in gradient descent or gradient flow. Particularly, the weights incident to _xk_ at time _t_ satisfy _W_ 1 _,k_ ( _t_ ) = _b_ 1( _t_ ) + ( _W_ 1 _,k_ (0) _− b_ 1(0)). As a result, to show that the weight of _xk_ in the function learned by the linear neural network is negligible, it is enough to show that the role of the first layer’s bias is negligible. Accordingly, we propose the more general conjecture below. We will also formalize this equivalence in Remark 20. 

**Conjecture 17** _Consider training a depth L ≥_ 2 _fully connected linear neural network defined in Definition 16 with the ℓ_ 2 _loss function_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0011-03.png)


_with gradient flow, i.e., θ_[˙] = _[dθ] dt_[=] _[−∇][θ][L]_[(] _[θ]_[)] _[.][Further,][assume][the][neural][network][is][ini-] tialized with Wi_ (0) = _αWi and bi_ (0) = _αbi for_ 1 _≤ i ≤ L where α is the initialization scale, and Wi,bi are initial directions independent from α. For technical reasons, we also allow the first layer’s bias to have a different speed, i.e., b_[˙] 1 = _−γ∇b_ 1 _L_ ( _θ_ ) _where γ >_ 0 _is a constant. We conjecture that there exists_ 0 _< ϵ_ = _oα_ (1) _such that if the model is initialized with scale α then the contribution of first layer’s bias would be smaller than ϵ, i.e., ∥b_ 1 _∥, |wL[T][W][ T] L−_ 1 _[· · ·][ W][ T]_ 2 _[b]_[1] _[|][≤][ϵ][.][Moreover,][∥][W][i][∥][F][remains][O][α]_[(1)] _[for][all][layers][during] training._ 

The intuition for this conjecture is that the updates for the last layer’s bias are of constant order at initialization, i.e., _b_[˙] _L_ = _θα_ (1). On the ˙other hand, the˙update of the rest of the parameters scale with _α_ at initialization, i.e., _Wi_ = _O_ ( _α_ ) and _bi_ = _O_ ( _α_ ) for _i_ = _L_ . As a result, the bias of the neural network is first learned by the bias of the last layer. We conjecture that this picture will not change much as training continues. We prove this formally for a two-layer linear neural network with appropriate initialization in Theorem 18. We also provide experimental evidence for this conjecture in Appendix B. 

**Theorem 18** _Consider Conjecture 17 for two-layer neural networks. We prove that for w[∗]_ = 0 _if the initialization satisfies ∥w_ 2 ~~_∥_~~[2] _> ∥W_ 1 _∥_[2] + _γ[−]_[1] _∥b_ 1 _∥_[2] _then the statement of the conjecture holds. I.e., we prove an upper bound for ∥b_ 1 _∥ and |w_ 2 _[T][b]_[1] _[|][which][vanishes][as][the] initialization scale α goes to zero while ∥W_ 1 _∥F , ∥w_ 2 _∥ remain Oα_ (1) _during training._ 

_Proof Sketch._ We prove this theorem by analyzing the trajectory of gradient flow on the parameters in three phases. In the first phase, we prove that the bias of the last layer learns the total bias of the target while all other parameters remain small as the updates for the bias of the last layer are of the constant order compared to the updates of other parameters which are of order _O_ ( _α_ ). In the second phase, we prove that _w_ 2 and _W_ 1 will reconstruct the weight term _w[∗]_ while _∥b_ 1 _∥_ remains small. At this point, both the loss function and _∥b_ 1 _∥_ are small. For the last phase, we prove that the parameters would not change much after this point. So the bias parameter and its contribution remain negligible. The full proof is presented in Appendix A. The condition _∥w_ 2 ~~_∥_~~[2] _> ∥W_ 1 _∥_[2] + _γ[−]_[1] _∥b_ 1 _∥_[2] on the initialization is 

11 

Abbe, Bengio, Lotfi, and Rizk 

only needed for the analysis of the trajectory in the second phase of our proof and is not uncommon (e.g., see a similar one in the results of Yun et al., 2021). 

One can easily see that phase 1 of the proof can be generalized to any neural network in Conjecture 17. Also, it is possible to generalize phase 3 to deeper neural networks (assuming that the neural network reaches the starting point of this phase). Phase 2, on the other hand, may require more technical work for deeper networks. Particularly, one may need to break this phase into several steps and analyze the trajectory in each of these steps. 

**Remark 19** _Note that even for a constant initialization scale, as depth L grows, there are L terms that reconstruct the bias. One could thus expect that the role of the first layer’s bias to decrease in this reconstruction as L grows. This is an independent phenomenon that we do not capture in Conjecture 17 where we have the vanishing initialization scale._ 

**Remark 20** _Again, consider the original problem in which xk_ = 1 _is frozen. We noted that the weights incident to xk, W_ 1 _,k work exactly in the same manner as the bias parameter of the first layer b_ 1 _. Indeed, one can define an equivalent bias parameter_[˜] _b_ 1 = _b_ 1 + _W_ 1 _,k and remove xk from the coordinates. The only caveat is that this parameter would have a speed of γ_ = 2 _for the updates (this is also the reason that we introduced γ in Conjecture 17). We can simply check that if Conjecture 17 is satisfied (e.g., as in Theorem 18), then the frozen bit will be ignored showing the min-degree bias. We prove this more formally in Appendix A as well._ 

**Remark 21** _Note that with the assumptions of Theorem 15 or Conjecture 17, the generalization error of the model becomes_[3] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0012-06.png)


_where_ Inf _k_ ( _f_ ) = _f_[ˆ] ( _{k}_ )[2] _is the Boolean influence of the k-th bit (O’Donnell, 2014). This confirms the empirical observations of Abbe et al. (2022a) on fully connected linear neural networks._ 

## **4 Experiments** 

In this section, we present our experimental results on the min-degree bias of neural networks.[4] We have used two architectures for our experiments in this part: a random features model (Definition 8) and an encoder-only Transformer (Vaswani et al., 2017). We show that both these architectures have a very strong min-degree bias and would learn min-degree interpolators. We will also consider a multi-layer perceptron (MLP) with 4 hidden layers and a 2-layer neural network with mean-field parametrization (Mei et al., 2018) in Section 7.3. We will show that these architectures also follow the min-degree bias although in a weaker way and possibly with a leakage. By doing this, we consider a spectrum of models covering lazy regimes, active/feature learning regimes, and models of practical interest. For the Transformer, we use an encoder-only architecture with bidirectional attention, absolute learnable positional embeddings, and a classification token. Also, _±_ 1 bits are first 

> 3. The factor 4 is removed if we consider the half-quadratic loss and GOTU on the full space. 

> 4. Code: `https://github.com/aryol/GOTU` 

12 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

encoded using an encoding layer and then passed to the Transformer; while for the rest of the architectures, binary vectors are directly used as the input. Note that the input in our tasks is always of fixed size and does not contain any causal structures. Also, the output is continuous and 1-dimensional. This makes encoder-only Transformers the most natural choice in the Transformers family. Nonetheless, we explore the properties of Transformers with causal attention in Section 7.2. 

For each experiment, we generate all binary sequences in _U[c]_ = _{±_ 1 _}[d] \ U_ for training.[5] We then train models under the _ℓ_ 2 loss. We employ Adam (Kingma and Ba, 2015) optimizer for the Transformer model and mini-batch SGD for the rest of the architectures. We also use moderate learning rates as learning rate can affect the results (refer to Appendix B.2). During training, we evaluate the coefficients of the function learned by the neural network using _f_[ˆ] NN( _T_ ) = E _x∼U {±_ 1 _}d_ [ _χT_ ( _x_ ) _f_ NN( _x_ )] to understand which interpolating solution has been learned by the model. Moreover, each experiment is repeated 10 times and averaged results are reported. For more information on the setup of experiments, hyperparameter sensitivity analysis, and additional experiments refer to Appendix B. 

Here, we consider the following 3 functions and unseen domains on input dimension 15. Dimension 15 is used as a large dimension where the training data can be generated explicitly but has otherwise no specific meaning (Appendix B provides other instances). The first function is an example of degree-2 where the unseen domain induces a degree-1 MD interpolator. The second example is the classic degree-2 parity or XOR function. The third example is such that the function is symmetric under cyclic permutations while its MD interpolator is not, in order to test whether certain models would favor symmetric interpolators. We consider other examples such as the majority function in Appendix B. Let: 

1. _f_ 1( _x_ ) = _x_ 0 _x_ 1 _−_ 1 _._ 25 _x_ 1 _x_ 2 + 1 _._ 5 _x_ 2 _x_ 0 and _U_ 1 = _{x_ 0 _x_ 1 _x_ 2 = _−_ 1 _}_ . In this case, we have _fx_ ˜10( _xx_ 1) == _xx_ 22 _−_ , _x_ 11 _.x_ 252 _x_ =0 + 1 _x_ 0, _._ 5and _x_ 1. _x_ 2 _x_ 0 = _x_ 1 at training, hence the MD interpolator is 

2. _ff_ ˜22(( _xx_ ) =) = _xx_ 10 + _x_ 1 _x_ 0and _−_ 1 _U_ for2 =the _{_ (seen _x_ 0 _, x_ 1domain.) = ( _−_ 1 _, −_ 1) _}_ . Note that the MD interpolator is 

3. _f_ 3( _x_ ) = _x_ 0 _x_ 1 _x_ 2 + _x_ 1 _x_ 2 _x_ 3 + _· · ·_ + _x_ 13 _x_ 14 _x_ 0 + _x_ 14 _x_ 0 _x_ 1 and unseen domain _U_ 3 = _{_ ( _x_ 0 _, x_ 1 _, x_ 2) = ( _−_ 1 _, −_ 1 _, −_ 1) _}_ . In this case, the MD interpolator is given by _f_[˜] 3( _x_ ) = ( _x_ 0 _x_ 1 + _x_ 1 _x_ 2 + _x_ 2 _x_ 0 _− x_ 0 _− x_ 1 _− x_ 2 + 1) + _x_ 1 _x_ 2 _x_ 3 + _· · ·_ + _x_ 13 _x_ 14 _x_ 0 + _x_ 14 _x_ 0 _x_ 1. 

We generally obtain that the encoder-only Transformer exhibits a strong MD bias similar to the random features model.[6] The solutions learned by the Transformer and the RF model for _f_ 1, _f_ 2, _f_ 3 are shown in Figure 1. It can be seen that these are very close to the MD interpolator in all cases. We will try learning the same examples with an MLP and a mean-field model in Section 7.3. 

> 5. In practice, one can generate a large enough number of samples so that the function is learned well on the training distribution. 

> 6. Note that the RF model in Figure 1 has a small leakage, simply caused by the ambient dimension being _d_ = 15 and not diverging as in Theorem 11. 

13 

Abbe, Bengio, Lotfi, and Rizk 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0014-01.png)


**----- Start of picture text -----**<br>
f1 - Transformer f2 - Transformer f3 - Transformer<br>1.5 1.00 1.00<br>1.0 0.75 0.75 1<br>0.50 0.50 x0<br>0.50.00.51.0 1 x x xxxxx2 1 00100xxxx1221x2 0.250.000.250.500.751.00 1 xx x 01 0x1 0.250.000.250.500.751.00 x xx xx xx 12 0 010 6x xx xx1 221 7xx 2 8<br>0 10 20 30 40 0 5 10 15 20 0 20 40 60 80<br>Epoch Epoch Epoch<br>f1 - RF f2 - RF f3 - RF<br>1.5<br>1.00<br>1.00<br>1.0 0.75 0.75 1<br>0.50 0.50 x0<br>0.50.00.51.0 1 x x xxxx2 1 0 0 10xxx 1 22 0.250.000.250.500.75 1 xx x 01 0x1 0.250.000.250.500.75 x xx xx xx 12 0 010 6x xx xx1 221 7xx 2 8<br>x0x1x2 1.00 1.00<br>0 10 20 30 40 0 10 20 30 40 0 10 20 30 40 50<br>Epoch Epoch Epoch<br>Coefficient of the monomial Coefficient of the monomial Coefficient of the monomial<br>Coefficient of the monomial Coefficient of the monomial Coefficient of the monomial<br>**----- End of picture text -----**<br>


Figure 1: Target functions _f_ 1 (left), _f_ 2 (middle), and _f_ 3 (right) learned by the encoder-only Transformer (top row) and the RF model (bottom row). Note that in all of the cases, the Transformer and the RF model learn a solution very close to the min-degree interpolator. More precisely, the coefficients of _x_ 0 _x_ 1 _, x_ 1 _x_ 2 _, x_ 2 _x_ 0 in the left plot ( _f_ 1), the coefficient of _x_ 0 _x_ 1 in the middle plot ( _f_ 2), and the coefficient of _x_ 0 _x_ 1 _x_ 2 in the right plot ( _f_ 3) are close to zero. 

## **5 Length Generalization** 

Several recent works on the reasoning of neural networks evaluate whether neural networks are able to generalize when the length of the problem is increased, and it is often found that neural networks struggle with length generalization (Zhang et al., 2022; Anil et al., 2022). For example, consider learning the parity problem parity( _x_ 1 _, . . . , xd_ ) = _x_ 1 _x_ 2 _· · · xd_ on _xi_ = _±_ 1. Two variants of this task can be considered: (1) the number of bits, _d_ , is increased during test, and (2) _d_ is the same during training and test; however, during training, only samples with a bounded number of _−_ 1’s are observed, i.e., the radius _r_ Hamming ball _Br_ := _{x ∈{±_ 1 _}[d] |_ # _−_ 1( _x_ ) _≤ r}_ (note that +1 is the identity element in this setting). Anil et al. (2022) show that both of these variants capture the notion and difficulty of length generalization.[7] Here, we focus on the latter variant which falls under our GOTU setting. 

**Theorem 22** _Consider a Boolean function f_ : _{±_ 1 _}[d] →_ R _. Then (i) there exists a unique function fr_ : _{±_ 1 _}[d] →_ R _such that ∀x ∈ Br, fr_ ( _x_ ) = _f_ ( _x_ ) _and_ deg( _fr_ ) _≤ r; (ii) when f is a parity function (monomial) of degree k ≤ d, the ℓ_ 2 _-test-loss of the MD interpolator is larger k−_ 1 2 _than_ � _r_ � _._ 

> 7. We train our model directly on the parity function; whereas Anil et al. (2022) use large language models and fine-tune parity tasks on them. In this sense, our approach is closer to Zhang et al. (2022) who also train models on their synthetic task from scratch. 

14 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0015-01.png)


**----- Start of picture text -----**<br>
1.75 B 15  (full space)<br>1.50 B10<br>B9<br>1.25 B8<br>B7<br>1.00 B6<br>0.75<br>0.50<br>0.25<br>0.00<br>0 2 4 6 8 10 12 14<br>Degree<br>Total norm of monomials of a certain degree<br>**----- End of picture text -----**<br>


Figure 2: Learning full parity function in dimension _d_ = 15 in the length generalization setting with inputs in _B_ 6 _, B_ 7 _, B_ 8 _, B_ 9 _, B_ 10 and _B_ 15 (full space) respectively, with an MLP (model details in Appendix B). X-axis: degree-profile component, Y-axis: degree-profile value, i.e.,[�] _T_ : _|T |_ = _x[f]_[ˆ][NN][(] _[T]_[)][2][.][As][the][length][of][training][samples][is][decreased,][the][coefficient] of the full parity gets smaller and the coefficients of low-degree monomials get larger. 

We defer the proof to Appendix A. Now consider learning the parity function _x_ 1 _x_ 2 _· · · xd_ where training samples have _r_ or less _−_ 1 coordinates, i.e., training samples belong to _Br_ . Using the previous theorem, there is a degree _r_ alternative to _x_ 1 _x_ 2 _· · · xd_ . Note that when such a low-degree alternative exists, assuming the min-degree bias, the model will learn this alternative instead of the full function of degree _d_ . This explains why in this case neural networks cannot generalize when the length is increased. We conduct an experiment to evaluate this, where we learn the full parity function on 15 bits using the MLP model trained on different lengths. Figure 2 shows that we learn more of lower degree terms and less of the full parity term as we train on shorter lengths. 

## **6 Curriculum Learning** 

The bias of neural networks towards min-degree solutions can also be used to boost the learning via a curriculum learning (Bengio et al., 2009) algorithm. We propose to train models by increasing the ‘complexity’ of training samples with respect to the input Hamming weight, i.e., _Br_ 1 _⊆ Br_ 2 _⊆ . . . ⊆ Brk_ where _Br_ is the Hamming ball of radius _r_ . Training a model on samples included in _Br_ with _r < d_ produces biased inputs compared to the uniform distribution. It has been shown that learning parities with GD on biased inputs is easier for various architectures (Malach et al., 2021; Daniely and Malach, 2020). In particular, the bias in the input distribution can be viewed as converting a monomial on non-centered inputs to a staircase on centered inputs as discussed in the work of Abbe et al. (2021). Moreover, Abbe et al. (2022b) show that the sample complexity for learning staircases is significantly reduced compared to that of monomials of matching degree. In particular, a layer-wise analysis shows that the hidden neurons in the first layer detect the support of a parity function under biased inputs, allowing for the fitting of the target function with the second layer if enough neuron diversity is available. One can thus attempt to bootstrap this approach and progressively climb the support (and degree) of the target function by 

15 

Abbe, Bengio, Lotfi, and Rizk 

training successively the network on increasing balls. We now develop this approach into a general curriculum algorithm in Algorithm 1. 

**Algorithm 1** Degree-Curriculum algorithm 

**Input:** Training samples _S_ = _{_ ( _xi, yi_ ) _}[m] i_ =1[;][Curriculum] _[B][r]_ 1 _[⊂][B][r]_ 2 _[⊂][. . .][⊂][B][r] k_[=] _[B][d]_[;] Loss threshold _ϵ_ **for** _i_ = 1 **to** _k_ **do** _Sri_ := _{_ ( _x, y_ ) _∈ S|x ∈ Bri}_ (samples in _Bri_ ) initialize train loss = 1 + _ϵ_ . **while** train loss _> ϵ_ **do** train model with SGD on _Sri_ update train loss **end while end for** 

Note that at the _i_ -th step of Algorithm 1, all the training samples belong to _Bri_ . Thus, for models obeying the MD bias on the unseen, the model learns the MD interpolator of degree at most _ri_ . Further, if the sampling set _S_ is such that _B_ ( _ri_ ) _∩ S_ contains enough degree _ri_ elements, the MD interpolator is of degree _ri_ — see Theorem 22. If one then takes _ri_ = _ri−_ 1 + 1, the new MD interpolator has monomials at step _i −_ 1 that are contained in those at step _i_ , as in the learning of a merged staircase functions (Abbe et al., 2022b) (and a lower leap function more generally if one takes a leap in the curriculum degrees). Thus, for a parity target, the Degree-Curriculum algorithm learns the support sets incrementally as for the implicit staircase function. 

We evaluate the Degree-Curriculum algorithm on learning full parity functions of degrees 16 and 30, i.e., _x_ 0 _x_ 1 _· · · x_ 15 and _x_ 0 _x_ 1 _· · · x_ 29 with an MLP. More precisely, for the same training set and hyperparameters, we once train the MLP with normal SGD and once with the proposed Degree-Curriculum algorithm. We choose curriculum _B_ 4 _, B_ 8 _, B_ 12 _, B_ 16 (leap 4 curriculum) for degree-16 parity and _B_ 1 _⊂ B_ 2 _⊂· · · ⊂ B_ 29 _⊂ B_ 30 (leap 1 curriculum) for degree-30 parity. We use loss threshold _ϵ_ = 0 _._ 001. The results are depicted in Figure 3. It can be seen that the Degree-Curriculum algorithm can reduce the sample complexity for learning parity functions. 

In Algorithm 1, it is assumed that the training set is given with the random access model. We can also consider a variant with the query access model, where at step _i_ , training samples are queried directly from _Bri_ (or some distribution). In the former case, the probability of a sample belonging to _Br_ is small for small values of _r_ (e.g., _r_ = _od_ ( _d_ )). We thus expect the Degree-Curriculum algorithm under the query access model to be more efficient in that regard. In a concurrent work, Cornacchia and Mossel (2023) have investigated the benefit of using a query model with a biased sample distribution before a denser distribution to learn parities. Particularly, an improvement in the number of GD iterations has been proved using 1-step gradient arguments. In addition, Abbe et al. (2023b) has pursued the approach from this paper and the paper of Cornacchia and Mossel (2023) and has shown a formal separation between learning with and without curriculum for parities on a common data distribution. More specifically, it has been shown that for a data distribution that is a mixture of dense (uniform) and sparse (e.g., similar to _B_ 1) inputs, one can use a two-step 

16 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0017-01.png)


**----- Start of picture text -----**<br>
dim = 16 dim = 30<br>10 [0]<br>10 [0]<br>10 1<br>10 1<br>10 2<br>Curriculum (leap 4) Curriculum (leap 1)<br>No Curriculum No Curriculum<br>3000 5000 7000 9000 11000 8000 10000 12000 14000 16000 18000 20000<br>Number of training samples Number of training samples<br>Test loss Test loss<br>**----- End of picture text -----**<br>


Figure 3: Generalization loss on the 16-parity (left) and 30-parity (right) targets for different numbers of samples with and without the Degree-Curriculum Algorithm. We note that the MLP model trained without curriculum was not able to learn the full parity function in dimension 30 for the given sample sizes (and even up to 10[5] samples), in contrast to the same model trained with the Degree-Curriculum. 

curriculum (first on the sparse samples and then on the whole distribution) and learn the parity using fewer optimization steps comparing to the unordered samples. 

Note that in the Boolean setting and for the parity functions, +1 is the identity element. Thus, the number of _−_ 1’s used in the Degree-Curriculum algorithm can also be viewed as the length of the inputs. Interestingly, some works in the natural processing domain have used the length of the sentences (possibly along with other properties) to design their curriculum strategy (Spitkovsky et al., 2010; Zaremba and Sutskever, 2014; Kocmi and Bojar, 2017; Platanios et al., 2019). Finally, we can naturally extend the Degree-Curriculum algorithm to non-Boolean settings using the same principle as above: 

_Build curriculum sets {B_[˜] _i} of ‘increased complexity’ in order to have a path of learned functions on support sets {S_[(] _[i]_[)] _} that are as tightly nested as possible (e.g., staircases or low-leap functions as in the work of Abbe et al., 2022b), with the target function at last_ . 

## **7 Min-Degree Bias Beyond the Previous Settings** 

In this section, we study min-degree bias beyond the previous settings. Particularly, we investigate the effects of lifting the sparsity condition, the effects of using causal attention masking in Transformers, and using other architectures, namely, MLPs and mean-field networks. 

## **7.1 Small Ambient Dimension** 

In Theorem 11, we showed that for sparse functions and unseen domains (see Definition 7) the solution of the random features model would converge to the min-degree interpolator as the ambient dimension and number of features diverge. In our experiments presented in Section 4 and Appendix B, we demonstrated that the min-degree bias is visible even for small values of the dimension. Particularly, for ( _f_ 3 _, U_ 3) = ( _x_ 0 _x_ 1 _x_ 2 + _· · ·_ + _x_ 13 _x_ 14 _x_ 0 + _x_ 14 _x_ 0 _x_ 1 _, {_ ( _x_ 0 _, x_ 1 _, x_ 2) = ( _−_ 1 _, −_ 1 _, −_ 1) _}_ ), we can observe the min-degree bias despite the 

17 

Abbe, Bengio, Lotfi, and Rizk 

function not being sparse (see Figures 1 and 5). Note that in this case, the degree and the size of the unseen domain are small in comparison to the ambient dimension. In this section, we show that the min-degree bias can be weak if the ambient dimension is small compared to the degree and size of the unseen domain. Here, we consider two examples: degree-2 parity with holdout of pattern ( _−_ 1 _, −_ 1), i.e., (parity2 _, U_ ) = ( _x_ 0 _x_ 1 _, {_ ( _x_ 0 _, x_ 1) = ( _−_ 1 _, −_ 1) _}_ ) and degree-4 parity with a frozen bit (parity4 _, U_ ) = ( _x_ 0 _x_ 1 _x_ 2 _x_ 3 _, {x_ 0 = _−_ 1 _}_ ). Note that given the unseen domains any interpolator of parity2 can be written as (1 _− α_ Leak)( _x_ 0 + _x_ 1 _−_ 1) + _α_ Leak _x_ 0 _x_ 1 where (1 _− α_ Leak) is the coefficient of the min-degree interpolator and _α_ Leak is the leakage coefficient. Similarly, any interpolator of parity4 is of the form (1 _− α_ Leak) _x_ 1 _x_ 2 _x_ 3 + _α_ Leak _x_ 0 _x_ 1 _x_ 2 _x_ 3 _x_ 4. In Figure 4, we trained different models on these functions embedded in varying ambient dimensions and computed the leakage coefficient _α_ Leak. It can be seen that if the ambient dimension _d_ is too small, the min-degree bias may become weak or disappear. In such cases, other architecture-specific implicit biases may become relevant. This is also related to the experiments conducted by Zhou et al. (2024) where it is shown that for a Boolean AND target with all variables being active (i.e., the ambient dimension is equal to the effective dimension), Transformers learn a function different than the min-degree interpolator which is conjectured by Zhou et al. (2024) to be the shortest RASP-L program (a type of program that encodes what Transformers tend to compute, for more about RASP see Weiss et al., 2021). However, these non-min-degree results seem to be rather ‘boundary cases’, i.e., when the ambient and effective dimensions are exactly the same or very close. As soon as the ambient dimension exceeds the effective one by a large enough margin, it appears that the min-degree bias dominates again, as seen in Figure 4. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0018-02.png)


**----- Start of picture text -----**<br>
degree-2 parity degree-4 parity<br>0.35<br>Transformer 0.8 Transformer<br>RF (ReLU) RF (Poly)<br>0.30 0.7<br>0.6<br>0.25<br>0.5<br>0.20<br>0.4<br>0.15 0.3<br>0.2<br>0.10<br>0.1<br>0.05 0.0<br>2 4 6 8 10 12 14 4 6 8 10 12 14<br>Dimension Dimension<br>Leak  Leak<br>**----- End of picture text -----**<br>


Figure 4: Learning (parity2 _, U_ ) = ( _x_ 0 _x_ 1 _, {_ ( _x_ 0 _, x_ 1) = ( _−_ 1 _, −_ 1) _}_ ) (left) and (parity4 _, U_ ) = ( _x_ 0 _x_ 1 _x_ 2 _x_ 3 _, {x_ 0 = _−_ 1 _}_ ) (right) embedded in different dimensions with different models. For parity2 (left) we can see that the min-degree bias is strong for the Transformer even for low-ambient dimensions. We can also see that for the RF model, the min-degree bias becomes stronger as the ambient dimension increases. For parity4 (right) we can see that the Transformer can almost recover the true function when the ambient and active dimensions match. As the ambient dimension grows slightly, we see that the coefficient of the higher degree term falls rapidly resulting in learning the MD interpolator. 

18 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

## **7.2 Transformer with Causal Attention** 

In our main experiments, we have used an encoder-only Transformer architecture with bidirectional attention and learnable absolute positional embeddings. First, we explain our reasoning for this choice. Note that in our settings, most of the coordinates are i.i.d. uniform _±_ 1 bits due to the restricted size of the coordinates in the unseen domain (see Definition 7). As a result, we do not have any locality structure a priori, and hence, the use of relative positional embeddings does not seem suitable for our tasks. Moreover, our output is a single continuous variable which makes any sort of auto-regressive training inapt as well. 

On the other hand, the recent work of Kazemnejad et al. (2024) has shown that in decoder-only Transformers with no positional embeddings, the causal attention masks make the recovery of positional information possible (in contrast to encoder-only architectures with bidirectional attention in which the removal of positional embeddings makes the architecture permutation invariant). Moreover, they have shown that decoders with no positional embedding may exhibit superior performance in some length generalization tasks compared to decoders with positional embeddings. Motivated by this, we tried modifying our encoder architecture by making the attentions causal (unidirectional) and removing the positional embeddings. We trained this variant in a supervised setting with _ℓ_ 2 loss similar to the original encoder-only architecture. Interestingly, we observed that this new architecture may lose the min-degree bias in some settings. 

Notice that our original architecture with positional embeddings and bidirectional attention is symmetric with respect to different coordinates. However, this is not true when we use causal attention. For example, the behavior of encoders (with no attention masking) trained on tasks ( _f, U_ ) = ( _x_ 0 _x_ 1 _, {x_ 0 = _−_ 1 _}_ ) and ( _f, U_ ) = ( _x_ 14 _x_ 15 _, {x_ 14 = _−_ 1 _}_ ) in dimension _d_ = 16 would be the same, while this is not necessarily true for Transformers with causal attentions. (In Table 1, we see that the behavior is indeed different.) In other words, with causal attention, the positions of latent bits (and unseen domain) matter. As an example, we try learning the parity of two bits embedded in ambient dimension _d_ = 16. We also freeze one of the bits to +1 during training (same task as examples above). If our function is _xixj_ with _xi_ = 1 during training, the interpolator would have the form (1 _− α_ Leak) _xj_ + _α_ Leak _xixj_ where the min-degree bias predicts that _α_ Leak would be small. In Table 1, we have tried different positions for the latent coordinate and the frozen bit and reported the learned solution averaged over 10 random seeds. Note that we still see the min-degree bias for most of the placements, while for some of the placements, the min-degree bias disappears. Notice that we can also keep the positional embeddings while making the attentions causal. In this case, we can observe the min-degree bias again (potentially still weaker than the encoder-only architecture with bidirectional attention) as seen in Table 1. As reported in Table 1, the behavior of the Transformer architecture with causal masking heavily depends on the positions of latent coordinates and possibly the position of the bits involved in the unseen domain which creates a large set of placements for each sparse function. As a result, understanding the implicit bias of the Transformer model with causal masking requires a new avenue of investigation which we leave for future work. 

In any case, as mentioned earlier, using a Transformer with causal attention may not be the natural model choice for learning Boolean/logic targets that do not have a causal structure in their input space. In fact, for Boolean inputs in which coordinates do not usu- 

19 

Abbe, Bengio, Lotfi, and Rizk 

|Target Fixed<br>bit|Causal mask without pos. emb.<br>_α_Leak _±_std<br>learned function<br>(averaged)|Causal mask with pos. emb.<br>_α_Leak _±_std<br>learned function<br>(averaged)|
|---|---|---|
|_x_0_x_1<br>_x_0<br>_x_0_x_1<br>_x_1<br><br>_x_14_x_15<br>_x_14<br>_x_14_x_15<br>_x_15<br>_x_2_x_8<br>_x_2<br>_x_2_x_8<br>_x_8<br>_x_7_x_13<br>_x_7<br>_x_7_x_13<br>_x_13|**0**_._**55**_±_**0**_._**04**<br>**0**_._**45**_x_1+**0**_._**55**_x_0_x_1<br>_−_0_._02_±_0_._02<br>1_._02_x_0_−_0_._02_x_0_x_1<br>0_._06_±_0_._05<br>0_._93_x_15+ 0_._06_x_14_x_15<br>**0**_._**47**_±_**0**_._**04**<br>**0**_._**53**_x_14+**0**_._**47**_x_14_x_15<br>0_._15_±_0_._05<br>0_._85_x_8+ 0_._15_x_2_x_8<br>0_._0_±_0_._02<br>1_._0_x_2+ 0_._0_x_2_x_8<br>0_._01_±_0_._02<br>0_._99_x_13+ 0_._01_x_7_x_13<br>0_._02_±_0_._01<br>0_._99_x_7+ 0_._02_x_7_x_13|0_._17_±_0_._02<br>0_._83_x_1+ 0_._17_x_0_x_1<br>0_._02_±_0_._02<br>0_._97_x_0+ 0_._02_x_0_x_1<br>0_._0_±_0_._01<br>0_._99_x_15+ 0_._0_x_14_x_15<br>0_._01_±_0_._02<br>1_._0_x_14+ 0_._01_x_14_x_15<br>0_._01_±_0_._02<br>0_._99_x_8+ 0_._01_x_2_x_8<br>0_._0_±_0_._02<br>1_._0_x_2+ 0_._0_x_2_x_8<br>0_._01_±_0_._02<br>0_._99_x_13+ 0_._01_x_7_x_13<br>0_._02_±_0_._02<br>1_._0_x_7+ 0_._02_x_7_x_13|



Table 1: Learning parity of two bits while one bit is frozen to one during training using Transformers with causal attention masking. Each row represents one particular combination for the position of the two bits and the frozen (fixed) bit. In columns, we report the average leakage coefficient ( _±_ standard deviation) and average learned function using 10 seeds for a Transformer with causal masking and no positional embedding and also for a Transformer with causal masking and positional embedding. For the Transformer without positional embeddings it can be seen that only two of the placements lead to the violation of the min-degree bias (in bold). For the Transformer with causal masking and positional embeddings, there is only placement that leads to a non-negligible leakage. For all other cases, the min-degree bias is still strongly present. Note that the leakage for encoder-only Transformer with bidirectional attention is negligible and independent of placement and thus not reported in this table. 

ally follow any causal relationship, encoder-only Transformers with bidirectional attention (which are symmetric with respect to different coordinates) seem to be the most reasonable choice. In this case, the min-degree bias dominates in the sparse regime. It is mostly intriguing from a theoretical viewpoint to see how the causal attention masking and the removal of positional embeddings affect the min-degree bias in some cases depending on the placement of the function. 

## **7.3 Other Architectures** 

In Section 4, we showed that encoder-only Transformers have a strong min-degree bias similar to the random features model. Now, we investigate two other architectures, namely, multi-layer perceptron (MLP) with 4 hidden layers and 2-layer neural network with meanfield parameterization (Mei et al., 2018). We show that these architectures also have the min-degree bias although in a weaker format. As a result, they learn leaky min-degree interpolators meaning that they partly capture the higher degree solution along with the min-degree solution. Particularly, we try learning examples ( _f_ 1 _, U_ 1) _,_ ( _f_ 2 _, U_ 2) _,_ ( _f_ 3 _, U_ 3) of Section 4 with the MLP and mean-field model. The results are depicted in Figure 5 showing that these models have leaky min-degree biases. Further, in Appendix B.2, we discuss the 

20 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

effect that large learning rates may increase the leakage of these models. For additional experiments refer to Appendix B. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0021-02.png)


**----- Start of picture text -----**<br>
f1 - MLP f2 - MLP f3 - MLP<br>1.5<br>1.00<br>1.00<br>1.0 0.75 0.75<br>0.50 0.50 1<br>0.5 1 0.25 1 0.25 x0<br>0.0 x x 2 1 0.00 xx01 0.00 xx12<br>x0 0.25 x0x1 0.25 x 0 x 1<br>0.51.0 xxx 0 10xxx 1 22 0.500.75 0.500.75 xxx 0 10xxx 2 21x2<br>x0x1x2 1.00 1.00 x 6 x 7 x 8<br>0 10 20 30 40 0 10 20 30 40 50 0 20 40 60 80 100<br>Epoch Epoch Epoch<br>f1 - Mean-Field f2 - Mean-Field f3 - Mean-Field<br>1.5<br>1.00<br>1.00<br>1.0 0.75 0.75 1<br>0.50 0.50 x0<br>0.50.00.51.0 1 x x xxxx2 1 0 0 10xxx 1 22 0.250.000.250.500.75 1 xx x 01 0x1 0.250.000.250.500.75 x xx xx xx 12 0 010 6x xx xx1 221 7xx 2 8<br>x0x1x2 1.00 1.00<br>0 10 20 30 40 0 10 20 30 40 50 0 20 40 60 80 100<br>Epoch Epoch Epoch<br>Coefficient of the monomial Coefficient of the monomial Coefficient of the monomial<br>Coefficient of the monomial Coefficient of the monomial Coefficient of the monomial<br>**----- End of picture text -----**<br>


Figure 5: Functions _f_ 1 (left), _f_ 2 (middle), and _f_ 3 (right) of Section 4 learned by the MLP (top row) and the mean-field model (bottom row). In all of these examples, the higher degree monomials (represented by the solid orange lines in the middle and left columns) are replaceable by the lower degree alternative (represented by the dashed lines). The MLP and mean-field models learn a leaky min-degree interpolator with the coefficient of the higher degree term possibly bounded away from 0. 

## **8 Related Literature** 

Given the deployment of machine learning models in the real world, out-of-distribution generalization is a critical aspect of machine learning that has been extensively studied both in theory (Ben-David et al., 2006; Mansour et al., 2009; Redko et al., 2020) and in practice (Gulrajani and Lopez-Paz, 2021; Miller et al., 2021; Wiles et al., 2022). Our work considers an extreme case of distribution shift in which part of the domain is entirely unseen during the training, and thus OOD generalization is only possible if the target function has special structures (e.g., being compositional or having in/equi-variances) and the model captures those structures. OOD generalization and the ability to extrapolate have also been used as proxies for measuring the reasoning capabilities of neural networks (Saxton et al., 2019; Zhang et al., 2021; Csord´as et al., 2021; Zhang et al., 2022) as these models are prone to memorization of training samples (Carlini et al., 2019; Feldman and Zhang, 2020; Kandpal et al., 2022; Carlini et al., 2023; Zhang et al., 2021) or learning undesirable shortcuts (Zhang et al., 2022). A special case is length generalization (Zaremba and Sutskever, 2014; Lake and Baroni, 2018; Hupkes et al., 2020; Zhang et al., 2022; Anil et al., 2022; Zhou et al., 2024), i.e., generalization to the input lengths beyond what is seen during the training. In 

21 

Abbe, Bengio, Lotfi, and Rizk 

this paper, we provided an explanation for the length generalization problem in the simple instance of parity functions (Anil et al., 2022). 

It has been shown that training with gradient descent imposes particular implicit regularization on the solutions found by the models such as sparsity (Moroshko et al., 2020), norm minimization (Bartlett et al., 2021), and margin maximization (in linear classification setting) (Soudry et al., 2018). This implicit regularization (or implicit bias) of neural networks trained with gradient-based algorithms has been used to explain the generalization of (often overparametrized) models (Bartlett et al., 2021). These results depend on the optimizer (Gunasekar et al., 2018a) and model (Gunasekar et al., 2018b) and are usually proven for simple models such as linear models (Soudry et al., 2018; Yun et al., 2021; Jacot et al., 2021) including diagonal linear neural networks (Gunasekar et al., 2018b; Moroshko et al., 2020) as studied in this paper. Our result for the random feature model builds upon the implicit bias toward solutions with minimum norm (Bartlett et al., 2021). Also related to us is the spectral bias (Xu et al., 2019; Rahaman et al., 2019) stating that neural networks, when learning a function in continuous settings, capture the lower frequency components faster (note that degree in Boolean functions plays a similar role to the frequency). In this paper, we develop a related insight in the Boolean setting by introducing the notion of degree-profile and showing the min-degree implicit bias for several models theoretically and empirically. On the drawbacks of such implicit biases, Shah et al. (2020) put forward the notion of extreme simplicity bias, showing that neural networks may ignore complex predictive features and solely rely on the simpler features. This simplicity bias may result in vulnerability to adversarial perturbations and poor OOD performance. In this paper, we also discuss how the min-degree bias can practically hinder length generalization. 

## **9 Conclusions and Future Directions** 

In this paper, we put forward the concept of generalization on the unseen (GOTU) and considered the learning of Boolean functions. We showed that various network architectures have a bias toward the min-degree interpolator, with theoretical results for the RF and diagonal/2-layer linear neural networks, and experimental results for encoder-only Transformers. We also found empirically that for large learning rates or for other models such as mean-field networks, a leaky version of the MD bias takes place. We also observed that using causal attention masking along with removing positional embeddings in Transformers or having very small ambient dimensions can tame the min-degree bias in some cases. 

We showed that the MD bias can be used in a curriculum learning algorithm where the training takes place on sets of increasing complexity. We also demonstrated that the MD bias can impede the learning of symmetric solutions and can make length generalization difficult. 

The min-degree bias is a form of Occam’s razor chosen by GD-trained neural nets, where the ‘simplicity’ is measured by the ‘degree-profile’. However, this might not be a desirable form of razor for various reasoning tasks. We believe that other forms promoting symmetries, compositionality, or more generally minimum description length (MDL) may often be more suitable. The next natural steps are thus to correct this min-degree bias. We propose here some directions to pursue: (1) architecture design promoting symmetries or compo- 

22 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

sitionality, (2) hyperparameter tuning (e.g., learning rates, scale), (3) data augmentation and multitasking, (4) MDL-like regularization at training. 

Lastly, we provide a demonstration of the last idea. For this example, consider learning task _f_ 3( _x_ ) = _x_ 0 _x_ 1 _x_ 2 + _x_ 1 _x_ 2 _x_ 3 + _· · ·_ + _x_ 14 _x_ 0 _x_ 1 and _U_ 3 = _{_ ( _x_ 0 _, x_ 1 _, x_ 2) = ( _−_ 1 _, −_ 1 _, −_ 1) _}_ . Note that this target has a cyclic symmetry but the min-degree interpolator is not invariant under any permutation (other than the identity). We add regularization term _L_ reg = E _π,x_ [( _f_ NN( _x_ ) _− f_ NN( _π_ ( _x_ )))[2] ], where _x_ is a random binary vector and _π_ is drawn uniformly from the set of all permutation ( _π_ ( _x_ ) refers to permuting indices of _x_ according to _π_ ), to the loss function. Note that this regularization term is 0 if and only if the neural network’s function is permutation invariant. We train the encoder-only Transformer with this regularizer. Particularly, we use 256 random samples and 1 random permutation in each iteration to estimate the regularization loss. In Figure 6, we show the coefficient of different monomials during training with this regularizer. It can be seen that the high-degree term _x_ 0 _x_ 1 _x_ 2 is mostly recovered by the end of training. In Figure 1, we saw that training Transformers on this task without regularizer would result in high-degree term _x_ 0 _x_ 1 _x_ 2 not being learned. Moreover, we can compare the generalization error of these two cases. With the regularization term, the Transformer achieves the generalization error 0 _._ 95 _±_ 0 _._ 13, while if no regularization is used the generalization error would be 8 _._ 14 _±_ 0 _._ 9 (we report average and standard deviation over 10 seeds). More general investigations of this approach, also on how to learn symmetry groups, are left to future work. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0023-03.png)


**----- Start of picture text -----**<br>
f3 - Transformer with symmetry regularizer<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>0.25<br>0.50<br>0.75 1 x2 x1x2<br>x0 x0x1 x0x1x2<br>1.00 x 1 x 0 x 2 x 6 x 7 x 8<br>0 20 40 60 80 100 120 140<br>Epoch<br>Coefficient of the monomial<br>**----- End of picture text -----**<br>


Figure 6: Function _f_ 3 learned by a Transformer with a regularization term. One can see that the high-degree monomial _x_ 0 _x_ 1 _x_ 2 (orange solid line) is mostly recovered compared to Figure 1 where it was completely lost. 

## **Acknowledgments and Disclosure of Funding** 

The authors of this work were funded by EPFL and Apple. We would like to thank anonymous reviewers and colleagues for their feedback on the earlier versions of this work. 

23 

Abbe, Bengio, Lotfi, and Rizk 

## **Appendix A. Proofs** 

## **A.1 Proofs for the Random Features Model** 

We start by proving a lemma showing that for strongly expressive activation functions each random feature is low-degree in the sense that the high-degree monomials have small coefficients in the Fourier-Walsh expansion of the random features. 

**Lemma 23 (Random features are low-degree)** _Consider random features generated by an activation function that is strongly expressive up to P_ = _Od_ (1) _, i.e., ϕw,b_ ( _x_ ) = _σ_ ( _⟨w, x⟩_ + _b_ ) _where wi, b ∼N_ (0 _, d_[1][)] _[are][the][random][weights][and][bias.][We][have][the][following] additional properties:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0024-05.png)


_A4._ E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _T_ ) _ϕ_[ˆ] _w,b_ ( _T[′]_ )] = 0 _for T_ = _T[′] ; and_ 

_A5._ E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _T_ )[2] ] = 0 _⇐⇒ ϕ_[ˆ] _w,b_ ( _T_ ) = 0 _∀ w, b,_ 

_where ϕ_[ˆ] _w,b_ ( _T_ ) _is the coefficient of monomial T in random feature ϕw,b_ ( _x_ ) _._ 

**Proof** For property (A3), consider all the subsets of [ _d_ ] = _{_ 1 _, . . . , d}_ with size _k ≤ P_ : _T_ 1 _, T_ 2 _, . . . , T d_[Due][to][the][symmetry,][we][have][E] _[w,b]_[[ˆ] _[ϕ]_[(] _[T]_[1][)][2][]][=] _[· · ·]_[=][E] _[w,b]_[[ˆ] _[ϕ]_[(] _[T]_[(] _[d]_[More-] ( _k_[).] _k_[))][2][].] over, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0024-10.png)


where in Equation 6 we used Parseval’s identity. By assumption (A1) on the function we know that E _g∼N_ (0 _,_ 2)[ _σ_ ( _g_ )[4] ] is finite. Thus, E _g∼N_ (0 _,_ 2)[ _σ_ ( _g_ )[2] ][2] is also finite and consequently E _g∼N_ (0 _, d_ +1 _d_[)][[] _[σ]_[(] _[g]_[)][2][]][2][can][be][upper][bounded][independently][of] _[d]_[,][which][proves][the][existence] part. Furthermore, E _w,b_ [ _ϕ_[ˆ] ( _Ti_ )[2] ] = _Od_ (� _kd_ � _−_ 1) = _Od_ ( _d−k_ ) _,_ where we used _k ≤ P_ = _Od_ (1). Now by property (A2), we can conclude that E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _T_ )[2] ] = Θ( _d[−|][T][|]_ ) for _|T | ≤ P_ . For property (A4), assuming _T_ = _T[′]_ take _i ∈ T_ ∆ _T[′]_ . Without loss of generality suppose _i ∈ T, i ∈/ T[′]_ . For weight vector _w_ , we flip the sign of the _i_ -th coordinate and denote the resulting vector by _w−i_ . Now note that E _x_ [ _σ_ ( _⟨w, x⟩_ + _b_ ) _χT_ ( _x_ )] = _−_ E _x_ [ _σ_ ( _⟨w−i, x⟩_ + _b_ ) _χT_ ( _x_ )] and E _x_ [ _σ_ ( _⟨w, x⟩_ + _b_ ) _χT ′_ ( _x_ )] = E _x_ [ _σ_ ( _⟨w−i, x⟩_ + _b_ ) _χT ′_ ( _x_ )]. Hence, _ϕ_[ˆ] _w,b_ ( _T_ ) _ϕ_[ˆ] _w,b_ ( _T[′]_ ) = _−ϕ_[ˆ] _w−i,b_ ( _T_ ) _ϕ_[ˆ] _w−i,b_ ( _T[′]_ ) and E _w,b_ [ _ϕ_[ˆ] _w,b_ ( _T_ ) _ϕ_[ˆ] _w,b_ ( _T[′]_ )] = 0. 

Note that the last property is a consequence of the continuity assumption on the activation function. 

Now we can prove Theorem 11. 

**Proof** _(Theorem 11)_ First, recall the set of all interpolating solutions on the training set _U[c]_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0024-15.png)


24 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

Note that a solution given by _a_ 1 _, . . . , aN_ is interpolating if and only if ~~_√_~~ 1 _N_ � _Ni_ =1 _[a][i][ϕ][i]_[(] _[x]_[)] _[∈] F_ int. 

Moreover, we study the features and solutions in the Fourier-Walsh basis. First, we index all possible monomials, i.e., _{T_ 1 _, T_ 2 _, . . . , T_ 2 _d}_ = 2 _[{]_[1] _[,]_[2] _[,]_[3] _[,...,d][}]_ and _χTi_ ( _x_ ) =[�] _j∈Ti[x][j]_[.][Further,] we define the coefficient of monomial _Tj_ in the _i_ -th feature as _ϕ_[ˆ] _i_ ( _Tj_ ) := E _x_ [ _ϕi_ ( _x_ ) _χTj_ ( _x_ )] and _F ∈_ R[2] _[d][×][N]_ as the matrix of features in the Fourier expansion, i.e., _Fi,j_ = ~~_√_~~ 1 _N[ϕ]_[ˆ] _[j]_[(] _[T][i]_[).][Using] this notation, _a_ corresponds to an interpolating solution if and only if 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0025-03.png)


where _g_ ˆ represents function _g_ in the Fourier-Walsh basis. Furthermore, note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0025-05.png)


Note that weights and biases of the features are sampled i.i.d., therefore, as _N →∞_ , ( _FF[T]_ ) _i,j_ behaves like _N_ (E _w_ [ _ϕ_[ˆ] _w_ ( _Ti_ ) _ϕ_[ˆ] _w_ ( _Tj_ )] _, N[−]_[1] Var _w_ [ _ϕ_[ˆ] _w_ ( _Ti_ ) _ϕ_[ˆ] _w_ ( _Tj_ )]) in distribution, due to the central limit theorem (CLT). More precisely, we need to invoke the law of large number to get concentration on the mean. In our cases, the variances are finite due to property (A1). More specifically, E _g∼N_ (0 _,_ 2)[ _σ_ ( _g_ )[4] ] is finite, and hence, E _g∼N_ (0 _, d_ +1 _d_[)][[] _[σ]_[(] _[g]_[)][4][]][is] finite. Moreover, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0025-07.png)


where we used Parseval’s identity from Equation (9) to Equation (10). We define Φ _∈_ R[2] _[d][×]_[2] _[d]_ as a shorthand notation as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0025-09.png)


where we have used properties (A3) and (A4). 

## A.1.1 Existence of Interpolating Solutions 

Now, we show that an interpolating solution exists with high probability. Particularly, take any interpolator _g_ that only depends on the latent variables _xi_ 1 _, . . . , xiP_ and we show that _g_ ˆ is in the image of _F_ w.h.p. and hence being an interpolating solution given Equation (7). Consider monomials such as _T_ for which _∀w, b ϕ_[ˆ] _w,b_ ( _T_ ) = 0. Due to properties (A2) and (A5), we know that such _T_ ’s satisfy deg( _T_ ) _> P_ , hence their corresponding rows are both zero in _F_ and in _g_ ˆ. We remove these rows from _F_ and _g_ ˆ and call the new ones _F_[˜] and _g_ ˆ[˜] . We also remove corresponding rows and columns from Φ and denote the new matrix by Φ.[˜] 

ˆ ˜ ˜ˆ ˆ Note _Fa_ = _g ⇐⇒ Fa_ = _g_ , therefore to prove that _g ∈_ Image( _F_ ) its enough to show that _F_[˜] is full row-rank, or equivalently, _F_[˜] _F_[˜] _[T]_ is full rank. Note that _F_[˜] _F_[˜] _[T]_ converges to Φ[˜] almost surely. Note that Φ[˜] is a diagonal matrix such that all elements on the diagonal are 

25 

Abbe, Bengio, Lotfi, and Rizk 

positive as all zero-entries of the diagonal are already removed by property (A5).˜Φ is full rank and _F_ ˜ _F_ ˜ _[T]_ becomes full rank almost surely as _N →∞_ . This concludesThereforethe proof of the existence of interpolators. 

## A.1.2 Learning the Min Degree-Profile Interpolating Solution 

Now, we investigate the interpolating solution found by the model. Note that we are interested in the interpolating solution with the minimum norm _∥a∥_ 2 (which is the solution found by GD starting from _a_ = 0). Consider an interpolating solution _g ∈F_ int _._ The ˆ ˆ interpolator _g_ is found by the model if and only if _Fa_ = _g_ , where _g_ is the Fourier expansion ˆ of _g_ written in the vector form. Moreover, note that the _a_ satisfying _Fa_ = _g_ with the minimum norm _∥a∥_ 2 is _a[∗] g_[=] _[ F][ †][g]_[ˆ][, where] _[ F][ †]_[ is the Moore-Penrose pseudo-inverse.][Therefore,] we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0026-04.png)


Now note that we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0026-06.png)


We know that _FF[T]_ almost surely converges to Φ, which is a diagonal matrix. Moreover, by property (A5), we know that the zero elements on the diagonal of Φ correspond to zero rows of _F_ , and hence zero entries of _g_ since _g ∈_ Im( _F_ ). Thus, we can say that ( _FF[T]_ ) _[†]_ and _∥F[†] g_ ˆ _∥_[2] converge to Φ _[†]_ and _g[T]_ Φ _[†] g_ as _N →∞_ w.h.p. Furthermore, since _g ∈_ Im( _F_ ), zero entries on diagonal Φ (or Φ _[†]_ ) correspond to zero entries of _g_ , thus, we also have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0026-08.png)


Also note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0026-10.png)


We now focus on interpolators minimizing the quantity introduced in Equation (15). First, note that these interpolators do not have any monomials having a variable other than latent variables _{xi_ 1 _, . . . , xip}_ , i.e., all of the learned monomials would be in 2 _[{][x][i]_[1] _[,...,x][ip][}]_ . To see this, consider an interpolating solution _g_ containing such monomials, _T_ 1 _, . . . , Tm ̸⊆ IP_ = _{i_ 1 _, . . . , iP }_ . For simplicity, we use the notation _x_ = ( _xIP , x_ [ _d_ ] _\IP_ ) to differentiate between latent variables and the rest of the bits. Now define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0026-12.png)


Note that _gI_ (( _xIP , x_ [ _d_ ] _\IP_ )) is independent of _x_ [ _d_ ] _\IP_ . Therefore _gI_ ( _x_ ) = _g_ ( _x_ ) for all the training samples. Moreover, note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0026-14.png)


26 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

which shows that _gI_ Φ _[†] gI < g_ Φ _[†] g_ unless _g_ = _gI_ . Note that if E _w,b_ [ _ϕ_[ˆ] ( _T_ )[2] ] = 0 for some ˆ _T_ , then _g_ ( _T_ ) = 0, since we are considering the solution learned by the RF model and _g_ ˆ _∈_ Im( _F_ ). In sum, the function learned by the RF model converges to an interpolator that only contains the latent coordinates, as _N →∞_ w.h.p. Note that E[ _ϕ_[ˆ] _w,b_ ( _T_ )[2] ] is the same for all _T_ of the same size due to symmetry, we denote this shared quantity by _ϕ_[ˆ] _|T |,d_ . Now, we revisit Equation (15), for the functions defined on latent coordinates _IP_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0027-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0027-03.png)


Note that since _σ_ is strongly expressive up to _P_ , we have _ϕ_[ˆ] _[−] k,d_[1][=][Θ(] _[d][k]_[).] Putting this along Equation (19) shows that the solution of the RF model converges to MinDegInterp + _ϵd_ almost surely as _N →∞_ , where _ϵd_ is a vanishing function (w.r.t. _d_ ) on the latent coordinates, which concludes the proof. 

Now we can easily prove Corollary 12. 

**Proof** _(Corollary 12)_ Consider the random features model _f_ RF( _x_ ; _a_ ) = ~~_√_~~ 1 _N_ � _Ni_ =1 _[a][i][ϕ][i]_[(] _[x]_[).] Similar to the proof of the main theorem, we can represent the function given by the random features model in the Fourier-Walsh basis as a vector of size 2 _[d]_ denoted by _f_[ˆ] RF. We consider the same _F ∈_ R[2] _[d][×][N]_ as above which represented the coefficients of monomials in individual features. As a result, _f_[ˆ] RF = _Fa_ . Note that property (A1) ensures that _F[T] F →_ Φ due to the central limit theorem as _N →∞_ . With the same arguments as the main proof, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0027-07.png)


Note that Φ _,_ Φ _[†]_ are both diagonal matrices. Now consider a monomial _χS_ of size _|S|_ = _Od_ (1). Assuming that this monomial corresponds with index _s_ of _f_[ˆ] RF, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0027-09.png)


which concludes the proof. Note that we used the fact that if Φ _s_ = Φ _[†] s_[= 0,][this][monomial] cannot be generated and _f_ RF _,s_ = 0. Also, we used the upper bound Φ _s_ = E _w,b_ [ _ϕ_[ˆ] ( _S_ )[2] ] = _O_ ( _d[−|][S][|]_ ) (part of property A3) which is a consequence of property (A1). 

## A.1.3 RF Model with ReLU Activation 

In this part, we study the random features model equipped with the ReLU activation function. Here, we mostly rely on the results of Abbe et al. (2022c). First, following proposition B.1 of Abbe et al. (2022c), we note that for every odd _k ≥_ 3, the coefficient of _k_ -th Hermite polynomial in the Hermite expansion of ReLU is zero. On the other hand, this coefficient is non-zero for _k_ = 1 and any even _k_ . Consequently, following Lemma A.2 

27 

Abbe, Bengio, Lotfi, and Rizk 

of Abbe et al. (2022c), for monomials _χT_ and _|T | ≤ P_ = _Od_ (1) we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0028-02.png)


where _ϕ_[ˆ] _w,b,_ ReLU( _T_ ) is the coefficient of monomial _T_ in random feature created by the weights and bias _w, b_ and the ReLU activation. Informally, Equation (20) indicates that odd monomials with degrees larger than one are not strongly expressed in the random features when ReLU is used as the activation function. Nonetheless, note that as in Lemma 23, we can still deduce that E _w,b_ [ _ϕ_[ˆ] _w,b,_ ReLU( _T_ )] = _O_ ( _d[−|][T][|]_ ) for _|T | ≤ P_ = _Od_ (1). This upper bound along with the lower bounds obtained in Equation (20) and the minimization problem of Equation (19) indicate that the random features model with ReLU activation would replace degree 2 or 2 _k_ + 1 monomials with lower degree monomials if possible. However, it might not replace degree 2 _k_ + 2 monomials with degree 2 _k_ + 1 monomials for _k ≥_ 1. We further illustrate this with an experiment. 

We consider learning _f_ ( _x_ 0 _, . . . , x_ 14) = _x_ 0 _x_ 1 _x_ 2 + _x_ 0 _x_ 3 _x_ 4 _x_ 5 under the unseen domain _U_ = _{x ∈{±_ 1 _}_[14] _|x_ 0 = _−_ 1 _}_ . Note that in this case, the min-degree interpolator is _x_ 1 _x_ 2 + _x_ 3 _x_ 4 _x_ 5. However, for the ReLU activation, we know that _x_ 3 _x_ 4 _x_ 5 would not necessarily be preferred to _x_ 0 _x_ 3 _x_ 4 _x_ 5 since the deg( _x_ 3 _x_ 4 _x_ 5) = 3 is odd. In Figure 7, we compare the solution learned by the RF model with ReLU and polynomial activation (here (1 + _x_ )[6] ). It can be seen that the polynomial activation learns the MD interpolator, whereas the RF with the ReLU activation function only learns the lower-degree monomial for the odd monomial and not for the even one. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0028-05.png)


**----- Start of picture text -----**<br>
RF - ReLU activation RF - Poly activation<br>1.00 1.00<br>0.75 0.75<br>x1x2 x1x2<br>0.50 xx03xx14xx25 0.50 xx03xx14xx25<br>x0x3x4x5 x0x3x4x5<br>0.25 0.25<br>0.00 0.00<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Epoch Epoch<br>Coefficient of the monomial Coefficient of the monomial<br>**----- End of picture text -----**<br>


Figure 7: Target function _f_ ( _x_ 0 _, . . . , x_ 14) = _x_ 0 _x_ 1 _x_ 2 + _x_ 0 _x_ 3 _x_ 4 _x_ 5 being learned by random features models under _U_ = _{x_ 0 = _−_ 1 _}_ . The RF model with strongly expressive activation (here (1 + _x_ )[6] ) learns the min-degree interpolator (right), while the min-degree bias of the RF model with ReLU activation depends on the degree of monomials being even or odd (left). More precisely, the RF model does not prefer degree 2 _k_ +1 monomial to degree 2 _k_ +2 monomial for _k ≥_ 1. Note that for the RF with ReLU activation (left), the coefficients of _x_ 3 _x_ 4 _x_ 5 and _x_ 0 _x_ 3 _x_ 4 _x_ 5 are equal and hence overlap. 

## **A.2 Proof for Diagonal Linear Neural Networks** 

Here, we present the proof of Theorem 15. **Proof** We denote parameters at time _t_ by _θ_ ( _t_ ). Also, we consider the training under half _ℓ_ 2 loss, to simply remove the 2 factor from gradients. Consider the (half) _ℓ_ 2 loss function 

28 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

for a training sample _x_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0029-02.png)


Moreover, we know every component of the training sample is sampled from Rad([1] 2[), except] the frozen bit which is set to _xk_ = 1. We denote this uniform distribution by _U−[d][−] k_[1][.][Given] this, the expected loss of the training set can be calculated as follows 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0029-04.png)


where we have used Parseval’s theorem (O’Donnell, 2014) to get the last equation. For ˆ ˆ simplicity, we define _B_ := _f_ ( _∅_ ) + _f_ ( _{k}_ ) and _B_ NN := _b_ +[�] _[L] l_ =1 _[w] k_[(] _[l]_[)] as the total bias of the target function and the neural network respectively. We know the gradient flow (GF) of the parameters of the neural network is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0029-06.png)


Therefore, using (23), we can derive the gradient flow for each of the parameters as below 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0029-08.png)


29 

Abbe, Bengio, Lotfi, and Rizk 

Using the above, we can derive the balancedness property of the neural network, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0030-02.png)


Therefore, _∀i_ ( _wi_[(] _[l]_[)][)][2] _[−]_[(] _[w] i_[(] _[l][′]_[)] )[2] is constant during training. Using this property, we can show that most of the model’s parameters are always bounded away from 0 during training. To see this, fix an index _i ∈_ [ _d_ ]. Let _ji[∗]_[= argmin] _j∈_ [ _L_ ] _[|][w] i_[(] _[j]_[)][(0)] _[|]_[.][Furthermore,][define] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0030-04.png)


Since the model parameters are initialized randomly using the uniform distribution, we can say that _ci >_ 0 with probability 1. Now, due to the balancedness property, we know that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0030-06.png)


Now we are able to show the convergence of the model. To begin with, note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0030-08.png)


Now, first, we consider an index _i ̸_ = _k_ . We have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0030-10.png)


Now using (31), we can say 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0030-12.png)


30 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

Therefore, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0031-02.png)


which shows 

2 in other words, _Ll_ =1 _[w] i_[(] _[l]_[)] _− f_[ˆ] ( _{i}_ ) goes to 0 exponentially fast in time, _t_ . Finally, we �� � make the same analysis for ( _B_ NN _− B_ )[2] . We have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0031-05.png)


The last equation shows that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0031-07.png)


i.e., ( _B_ NN( _t_ ) _− B_ )[2] converges to 0 exponentially fast in _t_ as well. Equations (23), (36), and (37) show that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0031-09.png)


where _c_ = 2 min(1 _,_ min( _{ci}i_ = _k_ ) _[L][−]_[1] ); hence, loss converges to zero exponentially fast in time (however, it is still initialization-dependent). 

As shown in (37), the bias of neural network, converges like ( _B_ NN( _t_ ) _− B_ )[2] _≤_ ( _B_ NN(0) _− B_ )[2] _e[−]_[2] _[t]_ . We denote _R_ := _|f_[ˆ] ( _∅_ ) + _f_[ˆ] ( _{k}_ ) _|_ + 1 _> |B_ NN(0) _− B|_ . Now notice that if _t ≥ Tϵ_ := log[8] _ϵ[R]_[,][then][we][have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0031-12.png)


We now show the growth of[�] _[L] l_ =1 _[w] k_[(] _[l]_[)] is comparatively slower, and therefore, it will not capture the bias fast enough and will remain small during the entire training process. More precisely, we first bound[�] _[L] l_ =1 _[w] k_[(] _[l]_[)] at the beginning of training ( _t ≤ Tϵ_ ). We define 

31 

Abbe, Bengio, Lotfi, and Rizk 

_m_ = argmax _l∈_ [ _L_ ] _|wk_[(] _[l]_[)][(0)] _[|]_[.][Again,][by][balancedness][property,][we][know][it][will][remain][the] largest during training, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-02.png)


Now note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-04.png)


where in the last line we used the fact that ( _B_ NN( _t_ ) _− B_ )[2] is decreasing. Now, we provide a bound for _|wk_[(] _[m]_[)] _|_ . First, we consider the case that _L_ = 2. In this case, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-06.png)


where we used Gronwall’s lemma in the last equation. It also shows 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-08.png)


Now, we consider the case that _L >_ 2. In this case, we also have (this could be considered as an extension of Gronwall’s lemma, note that _wk_[(] _[m]_[)] _>_ 0) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-10.png)


using the above we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-12.png)


hence, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-14.png)


Now we consider each of these bounds at _t_ = _Tϵ_ . First, for _L_ = 2, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0032-16.png)


32 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

which is upper bounded by 8 _[ϵ]_[if][(] _[w] k_[(] _[m]_[)] (0))[2] _≤ α_[2] _≤ αmax_[2][=] 8 _e_[2] _ϵ[RTϵ]_[.][Now,][we][consider][the] bound for deeper networks, _L >_ 2, at time _t_ = _Tϵ_ . We want to bound[�] _[L] l_ =1 _[w] k_[(] _[l]_[)][(] _[t]_[)][by] 8 _[ϵ]_[.] Using (48) this will happen if we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0033-02.png)


_L−_ 2 1 which will happen if _|wk_[(] _[m]_[)] (0) _| ≤ α_ max := (( _L −_ 2) _RTϵ_ + ([8] _ϵ_[)] _L_ ) 2 _−L_ . So we proved for small enough initializations, there exists a time, _Tϵ_ , where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0033-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0033-05.png)


We now show that this picture will not change much during the rest of the training process. To see this, note that _|B_ NN( _t_ ) _− B|_ is always decreasing over time and is continuous. Therefore, _B_ NN( _t_ ) _− B_ cannot change the sign (since changing the sign means that the variable had become equal to 0 at some time, which is contrary to the fact that its absolute value is decreasing). Considering equations (25) and (32) we can conclude that both _b_ ( _t_ ) _− B_ and _L_ � _l_ =1 _[w] k_[(] _[l]_[)][(] _[t]_[)][are][either][increasing][or][decreasing][during][the][whole][training.][First,][assume] both of them are increasing. For _t > Tϵ_ , we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0033-07.png)


The case for both functions being decreasing is also similar. This shows that _f_ NN( _{k}_ ) _< ϵ_ during the entire training. Now we can study GOTU loss for _t ≥ Tϵ_ using Parseval’s theorem 

33 

Abbe, Bengio, Lotfi, and Rizk 

as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0034-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0034-03.png)


which proves the theorem. Note that if we consider half _ℓ_ 2 loss for the entire population Ω the loss becomes _f_[ˆ] ( _{k}_ )[2] + _Ot_ ( _e[−][ct]_ ) + _Oϵ_ ( _ϵ_ ). 

**Remark 24 (Initialization of bias variable)** _Note that the analysis is independent of the initialization of the bias variable (as long as it satisfies a simple bound such as |b_ (0) _| ≤_ 1 2 _[).]_ 

**Remark 25 (Effect of depth)** _The current theorem proves that the low-degree solution is learned when the initialization scale is small enough. To see the effect of depth, we prove that α_ max _found in this proof is increasing by depth, L. In other words, if we have deeper networks, we can use larger initializations and still have the generalization error close to the Boolean influence._ 

_L−_ 2 1 **Proof** _(Remark 25)_ Consider _L ≥_ 3. We know that _α_ max := (( _L −_ 2) _RTϵ_ + ([8] _ϵ_[)] _L_ ) 2 _−L_ . For simplicity define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0034-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0034-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0034-10.png)


Now note that _α_ max = _g_ ( _L −_ 2). Therefore, we need to prove _g_ ( _x_ ) is increasing for _x ≥_ 1. To see this, note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0034-12.png)


34 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0035-01.png)


which holds because 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0035-03.png)


_x_ Therefore, exp( _[−]_[ln][(] _[xP] x_[+] _[Q] x_ +2 ) ) = _g_ ( _x_ ) is increasing. Finally, we have to compare _α_ max for depths 2 and 3. Note that for depth two _α_ max(2) =[�] 8 _ϵ[e][−][RT][ϵ]_[=] ~~�~~ _Q_ 1 _[e][−][P]_[while][for][depth] three, we have _α_ max(3) = _P_ +1 ~~_[√]_~~[3] _Q_[.][Therefore,][we][have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0035-05.png)


which gives the desired result. 

## **A.3 Proof for 2-Layer Fully Connected Linear Network** 

Here, we provide the proof for Theorem 18 and Remark 20. **Proof** ( _Theorem 18_ ) We recall that the loss function is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0035-09.png)


We define _rw_ := _W_ 1 _w_ 2 _− w[∗]_ and _rb_ := _w_ 2 _[T][b]_[1][ +] _[b]_[2] _[ −][b][∗]_[which in turn assess the reconstruction] of the weights and the bias. Using the gradient flow, each of the parameters follows the update rule presented below. (We often use the dot notation for derivatives with respect to time.) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0035-11.png)


First note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0035-13.png)


35 

Abbe, Bengio, Lotfi, and Rizk 

Thus, the loss function is decreasing and hence 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-02.png)


Further, we can check that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-04.png)


As a result, we get the following conservation rule 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-06.png)


taking traces from both sides, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-08.png)


Now we are ready to move to phase 1 of the training. 

_Phase 1._ There exists vanishing _α_ 1 = _oα_ (1) such that when we start training we reach a point such that _∥b_ 1 _∥, ∥W_ 1 _∥F , ∥w_ 2 _∥, |rb| ≤ α_ 1. In other words, the bias of the neural network has been learned and all parameters other than the last layer’s bias have stayed small. First, we bound the growth of _∥w_ 2 _∥, ∥W_ 1 _∥F , ∥b_ 1 _∥_ . Let _m_ = max _{∥w_ 2 _∥_[2] _, ∥W_ 1 _∥_[2] _F[,][ ∥][b]_[1] _[∥]_[2] _[}]_[.] Note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-11.png)


As a result, for _c_ 1 := max _{_ 4 _,_ 2 _γ}_ ~~�~~ _L_ (0), we have _[d][∥][w] dt_[2] _[∥]_[2] _,[d][∥][W] dt_[1] _[∥] F_[2] _,[d][∥][b] dt_[1] _[∥]_[2] _≤ c_ 1 _m_ . Thus 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-13.png)


where we used Gr¨onwall’s lemma to bound _m_ ( _t_ ). Also note that at the initialization, _m_ (0) = _O_ ( _α_[2] ). 

Now we focus on the reduction rate of _rb_ . Note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0036-16.png)


36 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

Define _β_ 1 = (2 _α_[2] max _{_ ~~_∥_~~ _w_ 2 _∥_[2] _, ∥W_ 2 _∥_[2] _F[,][ ∥] b_ 1 _∥_[2] _}L_ (0) _[c]_[1] )[1] _[/]_[(] _[c]_[1][+1)] and _t_ 1 to be the first moment such that _m_ = max _{∥w_ 2 _∥_[2] _, ∥W_ 1 _∥_[2] _F[,][ ∥][b]_[1] _[∥]_[2] _[}][≥][β]_[1][or] _[r] b_[2] _[≤][β]_[1][.][For][0] _[ ≤][t][≤][t]_[1][,][we][have][(for] _[α]_ small enough) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0037-02.png)


where we used Gr¨onwall’s lemma again. We can also conclude that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0037-04.png)


Now, given Equation (72), we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0037-06.png)


where we used the definition of _β_ 1. Therefore, _m_ ( _t_ 1) _≤[β]_ 2[1][,][and][hence,][it][is] _[r] b_[2][that][must] have become less than or equal to _β_ 1. Indeed, if we define _α_ 1 = _[√] β_ 1 = _oα_ (1) we can see that at _t_ 1 we reach to a point such that _∥b_ 1 _∥, ∥W_ 1 _∥F , ∥w_ 2 _∥, |rb| ≤ α_ 1. 

Now we can analyze phase 2 of the dynamics. 

_Phase 2._ Assuming that _∥b_ 1 _∥, ∥W_ 1 _∥F , ∥w_ 2 _∥, |rb| ≤ α_ 1 for vanishing _α_ 1, we can show that there exists vanishing _α_ 2 = _oα_ 1(1) such that we reach to a point satisfying _∥rw∥, |rb|, ∥b_ 1 _∥≤ α_ 2. In other words, during training, we reach to point that both the weights and the bias of the target is almost learned by the neural network and first layer’s bias _b_ 1 is still small. In this phase, we are going to use the assumption 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0037-10.png)


Let _t_ 2 be the first moment that _∥w_ 2( _t_ 2) _∥_[2] = _α_ 1 (note that _∥w_ 2( _t_ 1) _∥_[2] _≤ α_ 1[2][at the beginning of] this phase at _t_ 1). Note that if the dynamics never reach this point, then both _∥b_ 1 _∥, ∥w_ 2 _∥_ are bounded and there is nothing to prove. For _t_ 1 _≤ t ≤ t_ 2, we have that _∥W_ 1 _∥F , ∥w_ 2 _∥ ≤[√] α_ 1 and _∥b_ 1 _∥≤[√] γα_ 1. Thus, considering Equation (73), _rb_ also remains bounded by ~~�~~ _γL_ (0) _α_ 1. Note that (for small enough _α_ 1) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0037-12.png)


Also for _t_ 1 _≤ t ≤ t_ 2, _∥w[∗] ∥− α_ 1 _≤∥rw∥_ = _∥W_ 1 _w_ 2 _− w[∗] ∥≤∥w[∗] ∥_ + _α_ 1. Using this simple bound and _∥w_ 2( _t_ ) _∥_[2] _≥∥W_ 1( _t_ ) _∥_[2] _F_[+] _[ γ][−]_[1] _[∥][b]_[1][(] _[t]_[)] _[∥]_[2][,][we][have][(for][small][enough] _[α]_[1][)] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0037-14.png)


37 

Abbe, Bengio, Lotfi, and Rizk 

Consequently, assuming that _α_ 1 is small enough, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0038-02.png)


therefore, one can see that there exists a constant _c_ 2 _>_ 0 such that _∥rw_ ( _t_ 2) _∥≤∥w[∗] ∥− c_ 2 _α_ 1. From this point, we can analyze the learning dynamics concerning _w[∗]_ . Similar to the first phase, define _α_ 2 = (log log _α_[1] 1[)] _[−]_[1][and] _[t][′]_ 2[to][be][the][first][moment][such][that] max _{[√] γ,_ 1 _}_ max _{|rb|, ∥b_ 1 _∥} ≥ α_ 2 or _∥rw∥≤ α_ 2. For _t_ 2 _≤ t ≤ t[′]_ 2[and][small][enough] _[α]_[1][,][we] have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0038-04.png)


From the inequality above, we can conclude that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0038-06.png)


Now we bound _rb_ and _b_ 1. Define _m[′]_ = max _{rb_[2] _[,][ ∥][b]_[1] _[∥]_[2] _[}]_[.][Given][the][derivatives][for] _[r][b][, b]_[1] one can easily see that _d∥bdt_ 1 _∥_[2] _≤_ 2 _γm[′] ∥w_ 2 _∥_ and _drdtb_[2] _≤_ 2 _m[′] ∥w_ 2 _∥_ ~~�~~ _L_ (0). Therefore, for _η_ = max _{_ 2 ~~�~~ _L_ (0) _,_ 2 _γ}_ and _t_ 2 _≤ t ≤ t[′]_ 2[,][we][have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0038-08.png)


where we have used Gr¨onwall’s inequality again. Now we bound � _tt_ 2 _′_ 2 _[∥][w]_[2] _[∥][dt]_[.][Note][that] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0038-10.png)


38 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

where we used Cauchy’s inequality in the last line. Therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0039-02.png)


Recall that we had _α_ 2 = (log log _α_[1] 1[)] _[−]_[1][.][Therefore,][we][can][see][that] _[m][′]_[(] _[t]_[)] _[≪][α]_ 2[2][(note][that] _√α_ 1 exp( ~~�~~ log _α_[1] 1[) =] _[ o]_[(] _[α]_ 1[0] _[.]_[5+] _[ϵ]_ ) for any _ϵ >_ 0). Therefore, by _t[′]_ 2[values][of] _[|][r][b][|][,][ ∥][b]_[1] _[∥]_[have][not] grown to _α_ 2. So it must be _∥rw∥_ which has become less than or equal to _α_ 2. Thus, we proved the claim of the second phase that we reach a point such that _∥rw∥, |rb|, ∥b_ 1 _∥≤ α_ 2 for a vanishing _α_ 2 = _oα_ (1). Now, we are ready to move to the last phase of training. 

_Phase 3._ Considering that _∥rw∥, |rb|, ∥b_ 1 _∥≤ α_ 2 = _oα_ (1) (i.e., both the weight and the bias of the target is almost learned and bias of the first layer is still small), we show that the parameters would not move significantly after this point. 

First, we show that the loss function is having an exponential decay at this point. Note that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0039-06.png)


Now note that _L_ ( _t[′]_ 2[)] _[ ≤]_[2] _[α]_ 2[2][.][We also know that the loss is decreasing, thus for small enough] _α_ 2 we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0039-08.png)


As a result, we can conclude that for _c_ 3 = min _{_[1] 2 _[∥][w][∗][∥][,]_[ 1] _[}][ >]_[ 0,][we][have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0039-10.png)


where we used Gr¨onwall’s inequality again. Consequently, we can see that _∥rw∥, |rb|_ have exponential decay as well, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0039-12.png)


We also need to provide an upper bound for _∥w_ 2( _t[′]_ 2[)] _[∥]_[.][Note][that][we][have][proved][that] _w_ 2 _w_ 2 _[T][−][W][ T]_ 1 _[W]_[1] _[ −][γ][−]_[1] _[b]_[1] _[b][T]_ 1[=] _[ O]_[(] _[α]_[2][)][is][constant][during][training.][Thus][at] _[t][′]_ 2[,][we][have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0039-14.png)


Note that we know _∥rw∥_ = _∥W_ 1 _w_ 2 _− w[∗] ∥≤ α_ 2 and hence _W_ 1 _w_ 2 = _θ_ (1). Also ( _b[T]_ 1 _[w]_[2][)][2] _[≤] α_ 2[2] _[∥][w]_[2] _[∥]_[2][.][Therefore,][using][the][above][equation,][one][can][see][that] _[∥][w]_[2][(] _[t][′]_ 2[)] _[∥]_[=] _[O]_[(1)][(as][a] result _∥W_ 1( _t[′]_ 2[)] _[∥][F]_[=] _[ O]_[(1)][as][well),][i.e.,][there][exists] _[c]_[4] _[>]_[ 0][(not][depending][on] _[ α, α]_[1] _[, α]_[2][)][such] that for small enough _α_ 2 _∥W_ 1 _∥F , ∥w_ 2 _∥≤ c_ 4. (Note that during the entire training _∥rw∥_ and thus _∥W_ 1 _w_ 2 _∥_ is bounded. So we have _∥W_ 1 _∥F , ∥w_ 2 _∥_ = _Oα_ (1) during the first two phases as well.) To conclude, assume the contrary of phase 3, i.e., define _t_ 3 as the first moment that 

39 

Abbe, Bengio, Lotfi, and Rizk 

at least one of _∥W_ 1 _∥F ≥_ 2 _c_ 4, _∥w_ 2 _∥≥_ 2 _c_ 4, or _∥b_ 1 _∥≥[√] α_ 2 happens. For _t[′]_ 2 _[≤][t][≤][t]_[3][we][can] bound the change in all of the variables as follows (assuming that _α_ 2 is small enough) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0040-02.png)


where we used the exponential decay of ~~�~~ _L_ ( _t_ ) to derive the inequalities above. Note that the above inequalities show that none of _∥W_ 1 _∥F ≥_ 2 _c_ 4, _∥_ 22 _∥≥_ 2 _c_ 4, _∥b_ 1 _∥≥[√] α_ 2 can happen, proving our claim that the parameters would not move significantly. Particularly note that _∥b_ 1 _∥ <[√] α_ 2 and _|w_ 2 _[T][b]_[1] _[|][≤]_[2] _[c]_[4] _[√] α_ 2 where _α_ 2 = _oα_ (1) vanishes as _α →_ 0, showing that the proof of the theorem is now complete. 

Now we can also provide the proof for Remark 20. **Proof** _(Remark 20)_ Assume that we want to learn _f_ ( _x_ ) = _f_[ˆ] ( _∅_ )+[�] _[d] i_ =1 _[f]_[ˆ][(] _[{][i][}]_[)] _[x][i]_[ and suppose] that the _k_ -th bit is frozen. We know the other bits have a uniform and independent distribution. Therefore, we can use Parseval’s identity to write the training loss as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0040-05.png)


where the expectation on the LHS is uniform over all _x_ with frozen coordinate _k_ . Also, _W_ 1 _,k_ represents the first layer’s weights incident to the frozen coordinate and _W_ 1 represents _W_ 1 where _W_ 1 _,k_ is removed from the matrix. Similarly, _w[∗]_ represents _w[∗]_ when the _k_ -th coordinate is removed. Furthermore, define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0040-07.png)


One can easily check that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0040-09.png)


In other words, _W_ 1 _,k_ and _b_ 1 have the same gradient, and thus their difference stays the same over time. Now we define a new bias variable as[˜] _b_ 1 = _W_ 1 _,k_ + _b_ 1. We know that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0040-11.png)


Now by revisiting (77), we see that this is the loss function of the neural network resulting from removing the frozen coordinate and combining the bias variables as in Conjecture 17. The only difference is that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0040-13.png)


40 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

i.e., the update speed of this newly defined biased variable is _γ_ = 2. Therefore, if the conjecture is satisfied then we know that this bias _∥_[˜] _b_ 1 _∥_ and its contribution _∥wL[T][· · ·][ W][ T]_ 2[˜] _[b]_[1] _[∥]_ are bounded by _ϵ_ . Here, we can deduce the same about _W_ 1 _,k_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0041-02.png)


Therefore, _f_[ˆ] NN( _{k}_ ) = _wL[T][· · ·][ W][ T]_ 2 _[W]_[1] _[,k]_[is][also] _[O]_[(] _[ϵ]_[ +] _[ α]_[)][proving][the][remark.] 

## **A.4 Proof for the Length Generalization Theorem** 

**Proof** First, we prove the existence and uniqueness of such low-degree interpolators. Afterward, we consider it explicitly for parity functions. 

Note that we know there are no _r_ + 1 bits which are all equal to _−_ 1 in _Br_ . Therefore, for any _r_ + 1 indices, we have ( _xi_ 1 _−_ 1) _· · ·_ ( _xir_ +1 _−_ 1) = 0. Consequently, each _xi_ 1 _· · · xir_ +1 can be replaced by a degree _r_ polynomial. Now consider the Fourier-Walsh expansion of _f_ ( _x_ ). By applying the previous identity, one can replace all monomials in the Fourier-Walsh expansion of _f_ ( _x_ ) with degree _r_ (or less) alternatives, while the value of the function on _Br_ does not change. 

Now we prove the uniqueness. Consider all monomials _χT_ ( _x_ ) where _|T | ≤ r_ . There _d d d_ are in total �0� + �1� + _· · ·_ + � _r_ � = _|Br|_ of such monomials and consider all functions given by these monomials _fa_ ( _x_ ) =[�] _i[|][B]_ =1 _[r][|][a][i][χ][T] i_[(] _[x]_[).][Note][that][for][each] _[x][j][∈][B][r]_[1] _[≤][j][≤] |Br|_ , _fa_ ( _xj_ ) =[�] _[|] i[B]_ =1 _[r][|][a][i][χ][T] i_[(] _[x][j]_[).][In][other][words,] _[f][a]_[(] _[x][j]_[)][is][a][linear][combination][of] _[a][i]_[’s,][i.e.,] ( _fa_ ( _x_ 1) _, . . . , f_ ( _x|Br|_ )) _[T]_ = _M_ ( _a_ 1 _, . . . , a|Br|_ ) _[T]_ , where _Mi,j_ = _χTj_ ( _xi_ ). Now note that we have proven that any function can be written in this way, i.e., rank( _M_ ) = _|Br|_ showing that dim(ker( _M_ )) = 0 and hence the uniqueness. 

Now, we particularly study the case of monomials. Without loss of generality, consider degree _k > r_ monomial parity _k_ ( _x_ ) := _x_ 1 _x_ 2 _· · · xk_ . We claim that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0041-09.png)


is the the unique low-degree equivalent of parity _k_ on _Br_ , i.e., parity _k_ ( _x_ ) = _fr_ ( _x_ ) _∀x ∈ Br_ . To see this, take any _x ∈ Br_ . Define _s_ ( _x_ ) as the number of _−_ 1 bits in _x_ 1 _, · · · , xk_ , i.e., _s_ ( _x_ ) := _|{xi_ = _−_ 1 _|_ 1 _≤ i ≤ k}|_ . Note that 0 _≤ s_ ( _x_ ) _≤ k_ and parity _k_ ( _x_ ) = ( _−_ 1) _[s]_[(] _[x]_[)] . Furthermore, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0041-11.png)


41 

Abbe, Bengio, Lotfi, and Rizk 

Therefore, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0042-02.png)


where we used the fact that _s_ ( _x_ ) _≤ r_ . Now we consider the constant term (i.e., bias) of _fr_ ( _x_ ). Indeed notice that the constant in _fr_ ( _x_ ) is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0042-04.png)


It can easily be proven that the above constant is equal to ( _−_ 1) _[r]_[�] _[k][−] r_[1] � by induction on _r_ . Note that it is clear for _r_ = 1 and the induction step from _r_ to _r_ + 1 is given by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0042-06.png)


Therefore, by Parseval’s identity we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0042-08.png)


which proves the lower bound. Note that we ignored other Fourier-Walsh coefficients for the lower bound above. 

## **Appendix B. Experiment Details and Additional Experiments** 

In this section, we provide details on the architectures and experimenting procedure as well as presenting additional results. 

## **B.1 Experiment Details** 

## B.1.1 Architectures 

We use MLP, Transformer (Vaswani et al., 2017), mean-field (Mei et al., 2018) and random features model (Definition 8) for experiments. Here, we describe them in detail. 

- **MLP.** The MLP model is a fully connected network consisting of 4 hidden layers of sizes 512 _,_ 1024 _,_ 512 _,_ 64. The ReLU activation function is used for all layers except the output layer. Moreover, the standard initialization of PyTorch has been followed, i.e., the weights of each layer are initialized with _U_ ( ~~_√_~~ dim _−_ 1 in _[,]_ ~~_√_~~ dim1 in[)][where][dim][in][is][the] input dimension of the layer. 

42 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

- **Transformer.** We have employed the encoder part of Transformer networks which are widely used in computer vision (Dosovitskiy et al., 2021) and language modeling (Raffel et al., 2020). First, all binary _±_ 1 bits are encoded into a 256-dimensional vector using a shared embedding layer. Afterward, the embedded input is passed through 12 transformer layers. Finally, a linear layer is used to compute the output of the model. Moreover, the size of MLP hidden layers is set to 256, and 6 heads are used for the self-attention blocks. 

   - Also, note that we use bidirectional attention with learnable positional embeddings for our main experiments. Only for Section 7.2 we use causal attention masking. 

- **Mean-field.** We also use a two-layer neural network in the mean-field parametrization. More precisely, following Abbe et al. (2022b), _f_ MF( _x_ ) = _N_[1] � _Ni_ =1 _[a][i][σ]_[(] _[⟨][w][i][, x][⟩]_[+] _[b][i]_[),] where _ai ∼ U_ ( _−_ 1 _,_ 1) and _wi, bi ∼ U_ ( ~~_√_~~ _[−]_[1] _d[,]_ ~~_√_~~ 1 _d_[)] _[⊗][d][⊗][U]_[(] ~~_√_~~ _[−]_[1] _d[,]_ ~~_√_~~ 1 _d_[).][We][use][ReLU][as][the] activation function and set the number of neurons to _N_ = 2[16] . Note that with this formulation, one has to take large values for the learning rate, e.g., 100 or 1000. 

- **Random features model.** Following Definition 8, we use _f_ RF =[�] _[N] i_ =1 _[a][i][σ]_[(] _[⟨][w][i][, x][⟩]_[+] _bi_ ) as the parametrization of the RF model. Moreover, we initialize _ai_ = 0 and _wi, bi ∼N_ (0 _, d_[1][)] _[⊗][d][⊗N]_[(0] _[,] d_[1][)][where] _[d]_[is][the][input][dimension.][We][also][use] _[N]_[=][2][13] random features for our experiments. We have used the ReLU activation function for almost all of the experiments. We have only used polynomial activation (1 + _x_ )[6] for the experiment comparing RF models with the ReLU activation and polynomial activation (Figure 7). 

## B.1.2 Procedure 

The implementation of experiments has been done using the PyTorch framework (Paszke et al., 2019). Additionally, the experiments were executed on NVIDIA A100 GPUs and the experiments took around 90 GPU hours in total (excluding the selection of hyperparameters). Now we discuss the training procedures. 

**Length generalization and main experiments.** We first explain the experiments of the main experiment section and also experiments for the length generalization. For each function _f_ : _{±_ 1 _}[d] →_ R and unseen domain _U_ , we generate all binary vectors in _U[c]_ = _{±_ 1 _}[d] \ U_ for the training set. Consequently, we usually take small values of _d_ for the experiments. Our main motivation for doing so is to eliminate the randomness generated by the sampling of training examples and also to assume the in-distribution generalization. Nonetheless, we believe the min-degree bias still holds when training examples are sampled randomly as is illustrated in the experiments included in this appendix. 

We then train our models. For the Transformer, we have used Adam (Kingma and Ba, 2015) optimizer with batch size 256. For the RF models, we have used mini-batch SGD with a batch size of 256. Also, for the rest of the architectures, SGD with batch size 64 has been used. We did not observe any significant difference in the results of experiments by varying the batch sizes. We generally selected the learning rates per model (and task) by the stability of the training and the speed of convergence. We have included more details about the learning rate in Appendix B.2. We also set the number of training epochs large enough that the loss of models is always less than 10 _[−]_[2] . We also note that Transformers usually 

43 

Abbe, Bengio, Lotfi, and Rizk 

learn the target function in a few epochs, reaching a loss of the order of 10 _[−]_[4] . After that, the training becomes unstable in some instances. Indeed note that Transformers are usually trained with learning rate schedulers. However, we did not use any learning rate schedulers for simplicity and instead opted for early stopping to avoid unstable phases of training. Also for the results reported for causal attention masking in Table 1 and particularly for instance _x_ 7 _x_ 13 with _x_ 7 = 1 some of the seeds became unstable and did not converge. So for this particular instance, we reported the results for the first 10 seeds where the training loss converged. We note that even for the unstable seeds the min-degree bias was never violated. Note that our main objective is to demonstrate the min-degree bias of neural networks and not to optimize any performance metric. As a result, we did not focus on hyperparameter tuning in these experiments. Generally, hyperparameters used for our experiments are available in our code online: `https://github.com/aryol/GOTU` . 

Finally, we track the coefficient of different monomials, i.e., _f_[ˆ] NN( _T_ ) = E _x_ [ _χT_ ( _x_ ) _f_ NN( _x_ )] during the training. We have also repeated each experiment for 10 different seeds and reported the averages. Particularly, we have also drawn 95%-CI in Figures 3, 8, and 4 but we did not draw CI for other experiments to keep the plots more readable. 

**Curriculum learning experiments.** In contrast to other experiments, there is no unseen domain in these experiments. Also here we draw a fixed number of samples uniformly from _{±_ 1 _}[d]_ . We train the MLP model with the same training set, learning rate, and batch size, once with normal mini-batch SGD and once with Degree-Curriculum (Algorithm 1). Therefore, everything between the Degree-Curriculum algorithm and the normal training is the same. We use Adam optimizer for these experiments as we found it to be faster than plain SGD. Moreover, we selected the learning rate based on the results of the normal training and then used the same learning rate for the Degree-Curriculum algorithm to have a fair comparison. Finally, we compare the average generalization loss between the two algorithms. 

## **B.2 Sensitivity to Learning Rate** 

We noticed that the min-degree bias of some architectures such as MLPs depends on the learning rate. More precisely, we observed that smaller learning rates promote the mindegree bias and larger learning rates increase the leakage of the models. Here, we demonstrate the effect of the learning rate with an example. Consider learning _f_ 2( _x_ 0 _, . . . , x_ 14) = _x_ 0 _x_ 1 under unseen domain _U_ 2 = _{_ ( _x_ 0 _, x_ 1) = ( _−_ 1 _, −_ 1) _}_ . In this case, the min-degree interpolator is _x_ 0 + _x_ 1 _−_ 1. Nonetheless, any _α_ Leak( _x_ 0 _x_ 1)+(1 _− α_ Leak)( _x_ 0 + _x_ 1 _−_ 1) is also a valid interpolator where _α_ Leak shows the leakage of the interpolator. We tried learning _f_ 2 under _U_ 2 with an MLP and varied the learning rate; the results are depicted in Figure 8. It can be seen that larger learning rates cause higher leaks in the models. We note that training becomes more unstable with larger learning rates to the point that the model cannot be trained with learning rates larger than 0 _._ 2. Also notice that _α <_ 0 _._ 5 in all cases, hence, the min-degree alternatives are still dominant. In general, in our experiments, we tried to select moderate values for the learning rate to ensure that the optimization process is stable. Nonetheless, we never used learning rate below 10 _[−]_[5] for Adam and we usually set learning rate between 10 _[−]_[4] to 10 _[−]_[3] for SGD. Exact hyperparameters for different experiments are available in our code. 

44 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0045-01.png)


**----- Start of picture text -----**<br>
0.45<br>0.40<br>0.35<br>0.30<br>0.25<br>0.20<br>10 4 10 3 10 2 10 1<br>Learning rate<br>)1<br>x<br>0<br>x<br>Leak  (coefficient of<br>**----- End of picture text -----**<br>


Figure 8: Leakage of the interpolators learned by the MLP model trained with different learning rates. Larger learning rates weaken the min-degree bias and lead to higher leaks. 

## **B.3 Additional Experiments** 

Here, we complete the experiments presented in Section 4 and also provide an experiment in support of Conjecture 17. 

First, we try learning example ( _f_ 2 _, U_ 2) of Section 4 in a larger ambient dimension. More specifically, we use ambient dimension _d_ = 40 and consider learning _f_ 2( _x_ 0 _, x_ 1 _, . . . , x_ 39) = _x_ 0 _x_ 1 under unseen domain _{_ ( _x_ 0 _, x_ 1) = ( _−_ 1 _, −_ 1) _}_ . In this case, the MD interpolator is again _x_ 0 + _x_ 1 _−_ 1. For this experiment, we can not generate all 2[39] elements of _U[c]_ , thus, we only use 2[15] samples uniformly drawn from _U[c]_ . We also use the same number of samples for the estimation of Fourier-Walsh coefficients. The results are depicted in Figure 9. For the random features model, it can be seen that the leakage is reduced compared to Figure 1 where the ambient dimension is 15. On the other hand, the leakage of other models has remained the same, which shows that the sparsity and ambient dimension do not affect them. This is indeed consistent with our expectations as we know models such as the mean-field are able to perform feature-learning and learn the support of sparse Boolean functions (Abbe et al., 2022b). 

Further, we consider the majority function on 3 bits embedded in a 40-dimensional ambient space, i.e., _f_ 4( _x_ 0 _, x_ 1 _, . . . , x_ 39) = Maj( _x_ 0 _, x_ 1 _, x_ 2) =[1] 2[(] _[x]_[0][ +] _[ x]_[1][ +] _[ x]_[2] _[ −][x]_[0] _[x]_[1] _[x]_[2][)][under] the unseen domain _U_ 4 = _{x ∈{±_ 1 _}[d] |_ ( _x_ 0 _, x_ 1) = ( _−_ 1 _, −_ 1) _}_ . Note that in this case _x_ 0 _x_ 1 _x_ 2 can be replaced with _x_ 0 _x_ 2 + _x_ 1 _x_ 2 _− x_ 2 which leads to MD interpolator being equal to 1[Similar][to][the][previous][experiments,][we][trained][the][RF,] 2[(] _[x]_[0][ +] _[ x]_[1][ + 2] _[x]_[2] _[ −][x]_[0] _[x]_[2] _[ −][x]_[1] _[x]_[2][).] MLP, mean-field, and Transformer on this instance. For this example, we do not generate the whole _U[c]_ , and instead, we use 2[15] samples. This number is still large enough that gives the generalization on the seen domain. The results of this experiment are presented in Figure 10. Note that in this case, the original target function is more symmetric than the MD interpolator. Nonetheless, none of the models are able to recover the more symmetric function. 

Finally, we present an experiment on linear neural networks in support of Conjecture 17. Particularly, consider learning linear function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0045-08.png)


45 

Abbe, Bengio, Lotfi, and Rizk 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0046-01.png)


**----- Start of picture text -----**<br>
f2 (ambient dim = 40) - RF<br>1.00<br>0.75<br>0.50<br>0.25 1<br>x0<br>0.00<br>x1<br>0.25 x0x1<br>0.50<br>0.75<br>1.00<br>0 20 40 60 80 100<br>Epoch<br>f2 (ambient dim = 40) - MLP<br>1.00<br>0.75<br>0.50<br>0.25 1<br>x0<br>0.00<br>x1<br>0.25 x0x1<br>0.50<br>0.75<br>1.00<br>0 10 20 30 40 50<br>Epoch<br>Coefficient of the monomial<br>Coefficient of the monomial<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0046-02.png)


**----- Start of picture text -----**<br>
f2 (ambient dim = 40) - Transformer<br>1.00<br>0.75<br>0.50<br>0.25 1<br>x0<br>0.00<br>x1<br>0.25 x0x1<br>0.50<br>0.75<br>1.00<br>0 5 10 15 20<br>Epoch<br>f2 (ambient dim = 40) - Mean-Field<br>1.00<br>0.75<br>0.50<br>0.25 1<br>x0<br>0.00<br>x1<br>0.25 x0x1<br>0.50<br>0.75<br>1.00<br>0 10 20 30 40 50<br>Epoch<br>Coefficient of the monomial<br>Coefficient of the monomial<br>**----- End of picture text -----**<br>


Figure 9: _f_ 2( _x_ 0 _, x_ 1 _, . . . , x_ 39) = _x_ 0 _x_ 1 learned by the RF, Transformer, MLP, and mean-field models while training samples satisfy ( _x_ 0 _, x_ 1) _̸_ = ( _−_ 1 _, −_ 1). Consequently, _x_ 0 _x_ 1 (solid orange line) is replaceable by _x_ 0 + _x_ 1 _−_ 1 (dashed lines). The Transformer, MLP, and mean-field models learn leaky solutions and the leakage is very similar to Figures 1 and 5 where the ambient dimension is 15. In contrast, the leak of the RF model is decreased in comparison to Figure 1. 

with a 4-layer fully connected linear neural network such that the width of each layer is 256. We first initialize each layer with PyTorch’s default initialization (i.e., each layer’s weights are initialized with _U_ ( ~~_√_~~ dim _−_ 1 in _[,]_ ~~_√_~~ dim _−_ 1 in[)][where][dim][in][is][the][input][dimension][of][the] layer). Then, we multiply the weights of each layer with the initialization scale parameter _α_ to finish the initialization. Then we train the neural network on _f_ with the whole 2[13] possible samples using mini-batch SGD with batch size 256 and learning rate 10 _[−]_[4] . We stop the training when the training loss becomes less than 10 _[−]_[4] . At the end, we compute how much each layer’s bias is contributing to the bias learned by the neural network. More precisely, we compare _w_ 4 _[T][W][ T]_ 3 _[W][ T]_ 2 _[b]_[1][,] _[w]_ 4 _[T][W][ T]_ 3 _[b]_[2][,] _[w]_ 4 _[T][b]_[3][,][and] _[b]_[4][(respectively][the][contributing] bias from the first layer to the last layer). Particularly, we plot the absolute value of these contributing biases against the initialization scale in the log-log scale in Figure 11. It can be seen that as the initialization scale decreases, all of the bias becomes captured by the bias of the last layer and the contributions of other layers’ biases (including the first layer’s) go to zero supporting Conjecture 17. 

46 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0047-01.png)


**----- Start of picture text -----**<br>
f4 - RF f4 - Transformer<br>1.00 1.00<br>0.75 1 0.75 1<br>x0 x0<br>0.50 x 1 0.50 x 1<br>x2 x2<br>0.25 0.25<br>x0x1 x0x1<br>0.00 x 0 x 2 0.00 x 0 x 2<br>x1x2 x1x2<br>0.25 x0x1x2 0.25 x0x1x2<br>0.50 0.50<br>0 20 40 60 80 0 5 10 15 20 25 30<br>Epoch Epoch<br>f4 - MLP f4 - Mean-Field<br>1 1<br>1.00 1.00<br>x0 x0<br>0.75 x1 0.75 x1<br>x2 x2<br>0.50 x 0 x 1 0.50 x 0 x 1<br>x0x2 x0x2<br>0.25 0.25<br>x1x2 x1x2<br>0.00 x 0 x 1 x 2 0.00 x 0 x 1 x 2<br>0.25 0.25<br>0.50 0.50<br>0 10 20 30 40 0 10 20 30 40<br>Epoch Epoch<br>Coefficient of the monomial Coefficient of the monomial<br>Coefficient of the monomial Coefficient of the monomial<br>**----- End of picture text -----**<br>


Figure 10: _f_ 4( _x_ 0 _, x_ 1 _, . . . , x_ 39) = Maj( _x_ 0 _, x_ 1 _, x_ 2) =[1] 2[(] _[x]_[0][+] _[x]_[1][+] _[x]_[2] _[−][x]_[0] _[x]_[1] _[x]_[2][)][learned][by] the RF, Transformer, MLP, and mean-field models while samples satisfying ( _x_ 0 _, x_ 1) = ( _−_ 1 _, −_ 1) are excluded from training. In this case, _x_ 0 _x_ 1 _x_ 2 (orange solid line) is replaceable by _x_ 0 _x_ 2 + _x_ 1 _x_ 2 _− x_ 2. As expected, the RF learns the MD interpolator. The Transformer also learns the MD interpolator with a small leakage. On the other hand, the MLP and mean-field models have a more considerable leakage. 

## **Appendix C. Vanishing Ideals** 

In this section, we discuss the connection between unseen domains in Boolean settings and algebraic geometry and vanishing ideals. We refer interested readers to the work of Dummit and Foote (2004) for broader coverage of this topic. First, we recall some basic properties of rings and fields. A ring is a set with two binary operations, the addition + and the multiplication _∗_ where _∗_ may not have an inverse. A field is a ring such that all nonzero elements have an inverse. For example, Z with addition and multiplication is a ring but not a field. Whereas R and C are examples of fields. Here we will mostly work with polynomial rings with _d_ variables. Note that R[ _x_ 1 _, x_ 2 _, x_ 3 _, · · · , xd_ ] is of special interest to us since any Boolean function _f_ : _{±_ 1 _}[d] →_ R can be represented by a polynomial _p_ ( _x_ ) _∈_ R[ _x_ 1 _, x_ 2 _, x_ 3 _, · · · , xd_ ] thanks to its Fourier-Walsh expansion. Particularly, we focus on polynomial rings _R_ = K[ _x_ 1 _, x_ 2 _, · · · xd_ ] where K is a field. We start by recalling a few definitions. 

**Definition 26 (Ideal)** _Let R be a commutative ring. I ⊆ R is an ideal if_ 

47 

Abbe, Bengio, Lotfi, and Rizk 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0048-01.png)


**----- Start of picture text -----**<br>
Contribution of the biases of different layers based on init. scale<br>10 [0]<br>10 1<br>First layer<br>Second layer<br>Third layer<br>10 2 Fourth layer<br>10 1 10 [0]<br>Initialization scale<br>Contribution of the layer's bias<br>**----- End of picture text -----**<br>


Figure 11: Contribution of each layer’s bias to the bias learned by the neural network depending on the initialization scale. It can be seen that as the initialization scale decreases the bias of the neural network is dominantly learned by the last layer’s bias. Further, the contribution of biases of all other layers goes to 0. 

- ( _I,_ +) _is a group, and_ 

- _for all r ∈ R and i ∈ I we have ri ∈ I._ 

Having defined ideals, note that ideals can be generated from a _G ⊆ R_ . 

**Definition 27** _Consider a commutative ring R and let G ⊆ R. The ideal generated by G denoted by ⟨G⟩ is the smallest ideal that contains G. Particularly, if G_ = _{g_ 1 _, g_ 2 _, · · · , gn} is finite, we have_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0048-07.png)


For example, for _R_ = R[ _x_ 1 _, x_ 2], we have _⟨x_ 1 _−_ 1 _, x_ 1 _x_ 2 + 5 _⟩_ = _{p_ ( _x_ 1 _−_ 1) + _q_ ( _x_ 1 _x_ 2 + 5) _|p, q ∈_ R[ _x_ 1 _, x_ 2] _}_ . Another important notion is the notion of quotients, which is similar to the modulo operator. The following definition will make it more rigorous. 

**Definition 28 (Quotient)** _Let R be a commutative ring and I an ideal of R. Quotient R/I is defined as elements of the form r_ + _I with r ∈ R such that r_ + _I_ = _r[′]_ + _I if r − r[′] ∈ I. Furthermore, for any for r_ + _I, r[′]_ + _I ∈ R/I, addition_ + _and multiplication · for R/I are defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0048-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0048-11.png)


48 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

_Also, for r[′] ∈ R/I, any element r ∈ R satisfying r − r[′] ∈ I is called a representative of r[′] . R/I as defined above is indeed a ring._ 

Consider the following ideal _I_ Ω = _⟨x_[2] 1 _[−]_[1] _[, x]_[2] 2 _[−]_[1] _[,][ · · ·][, x]_[2] _d[−]_[1] _[⟩]_[of][R][[] _[x]_[1] _[, x]_[2] _[, x]_[3] _[,][ · · ·][, x][d]_[]] for Boolean functions. Note that for each binary bit _xi_ we have _x_[2] _i[−]_[1][=][0.][Therefore,] the Fourier-Walsh transform is a bijection between R[ _x_ 1 _, x_ 2 _, x_ 3 _, · · · , xd_ ] _/I_ Ω and the set of Boolean functions. 

Now we are ready to define vanishing ideals. Given a set of points _S ⊆_ K _[d]_ where K is a field, we are interested in the set of polynomials that are zero on _S_ . In the case of generalization on the unseen domain _U ⊆_ Ω, we are interested in the functions that vanish on _U[c]_ = Ω _\ U_ , as they are 0 on the training set and give a class of interpolators on _U[c]_ . 

**Definition 29 (Vanishing ideals)** _For a field_ K _and S ⊆_ K _[d] , vanishing ideal I_ ( _S_ ) _of S is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0049-05.png)


Note that _I_ Ω is indeed the vanishing ideal of Ω, i.e., _I_ (Ω) = _I_ Ω = _⟨x_[2] 1 _[−]_[1] _[, x]_ 2[2] _[−]_[1] _[,][ · · ·][, x]_[2] _d[−]_ 1 _⟩_ . Furthermore, for any _S ⊆{±_ 1 _}[d]_ , we have _I_ Ω _⊆ I_ ( _S_ ) and thus _I_ ( _S_ ) can be written as _I_ ( _S_ ) = _⟨v_ 1 _, v_ 2 _, . . . , vn⟩_ + _I_ Ω for some _n ∈_ N and Boolean functions _v_ 1 _, . . . , vn_ . For example, consider canonical holdout _U_ = _{x ∈{±_ 1 _}[d] |x_ 1 = _−_ 1 _}_ ; in this case we get _I_ ( _U[c]_ ) = _⟨x_ 1 _−_ 1 _⟩_ + _I_ Ω. We could also do an ‘inverse operation’: given an ideal or set of functions, find all the points which are zero under the elements of the ideal. 

**Definition 30** _For a field_ K _and G ⊆ R_ = K[ _x_ 1 _, . . . , xd_ ] _, and I_ = _⟨G⟩, we define V_ ( _G_ ) = _V_ ( _I_ ) _as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0049-08.png)


Therefore, operations _V_ and _I_ give us a way to transfer some algebraic properties to geometric properties. What we defined could be seen as part of the theory of classical algebraic geometry. In algebraic geometry, we are interested in the following type of sets _S_ : 

**Definition 31** _A set S ⊆_ K _[d] is called an affine variety, if there exists some ideal I such that V_ ( _I_ ) = _S._ 

In our case, all the _S_ are affine varieties as they are finite. For more details about algebraic geometry, please refer to the work of Cox et al. (2013). 

Now given an _S ⊆_ Ω, the following lemma gives us a recipe to find _I_ ( _S_ ). 

**Lemma 32** _For S and W two affine varieties, we have that I_ ( _S ∪ W_ ) = _I_ ( _S_ ) _∩ I_ ( _W_ ) _. Also, for x_ = ( _i_ 1 _, i_ 2 _, . . . , id_ ) _∈_ K _[d] , we have that I_ ( _x_ ) = _mx_ = _⟨x_ 1 _− i_ 1 _, x_ 2 _− i_ 2 _, . . . , xd − id⟩, where mx is a maximal ideal._ 

Since in our case _S_ is finite, one can apply this lemma a multitude of times to find _I_ ( _S_ ). Moreover, this ideal only vanishes on the elements of _S_ and not on any other element in Ω _\ S_ . 

49 

Abbe, Bengio, Lotfi, and Rizk 

**Example 1** _Suppose we work with d_ = 2 _, and we only allow the set V_ := _{_ ( _−_ 1 _, −_ 1) _,_ (1 _,_ 1) _}. We will have that I_ ( _V_ ) := _⟨x_ 1 + 1 _, x_ 2 + 1 _⟩∩⟨x_ 1 _−_ 1 _, x_ 2 _−_ 1 _⟩. By doing the calculations or using an algebra program (e.g., SageMath) we find that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0050-02.png)


So in general, for a certain _S ⊆_ Ω, we would like to express, _I_ ( _S_ ) as _⟨v_ 1 _, . . . , vn⟩_ + _I_ Ω for some desirable Boolean functions _v_ 1 _, v_ 2 _, v_ 3 _, . . . , vn_ . In fact, there are known algorithms that find a basis for an ideal (M¨oller and Buchberger, 1982). 

Before relating what we have defined to unseen domains, note that in our case the conditions only depend on a subset of the variables. Without loss of generality, suppose our conditions only depend on the first _k_ coordinates. Mathematically, that means _U_ = _Uk ×{−_ 1 _,_ 1 _}[d][−][k]_ , where _Uk ⊆{−_ 1 _,_ 1 _}[k]_ . Hence, we have _U[c]_ = _Uk[c][×{−]_[1] _[,]_[ 1] _[}][d][−][k]_[.][The following] lemma allows us to calculate _I_ ( _U[c]_ ) based on _I_ ( _Uk[c]_[).] 

**Lemma 33** _Suppose that U[c]_ = _Uk[c][×][{−]_[1] _[,]_[ 1] _[}][d][−][k][for][some][U] k[c][⊆{−]_[1] _[,]_[ 1] _[}][k][,][if][I]_[(] _[U] k[c]_[)][=] _⟨v_ 1 _, v_ 2 _, . . . , vn⟩_ + _⟨x_[2] 1 _[−]_[1] _[, . . . , x]_[2] _k[−]_[1] _[⟩][for][Boolean][functions][v]_[1] _[, v]_[2] _[, . . . , v][n][,][we][have] I_ ( _U[c]_ ) = _⟨v_ 1 _, v_ 2 _, . . . , vn⟩_ + _I_ Ω _._ 

Now having defined the vanishing ideals and quotients, we explain how they relate to our setting. In our setting, we are given _U ⊂_ Ω= _{−_ 1 _,_ 1 _}[d]_ representing the unseen domain, and a Boolean function _f_ that we wish to learn, which could be seen as an element of _R_ = R[ _x_ 1 _, . . . , xd_ ]. As we finish training, we will converge to a solution _f_ sol which is an interpolator of _f_ on _U[c]_ . This means that _f − f_ sol vanishes on _U[c]_ and so _f − f_ sol _∈ I_ ( _U[c]_ ). Hence, _f_ + _I_ ( _U[c]_ ) = _f_ sol+ _I_ ( _U[c]_ ), which means that _f_ sol is a representative of the class _f_ + _I_ ( _U[c]_ ) in the ring _R/I_ ( _U[c]_ ). Here we are interested in the minimum degree-profile interpolator, and our goal is to classify given a _f_ and _U_ , the minimum degree-profile representatives of _f_ + _I_ ( _U[c]_ ) in the ring _R/I_ ( _U[c]_ ). This gives us an overview of how our settings can be related to algebraic notions. 

## **C.1 Minimum Degree-Profile Interpolators** 

We are generally interested in finding the minimum degree-profile interpolators. One way to do this is as follows: given a Boolean function _f_ : _{−_ 1 _,_ 1 _}[d] →_ R which we suppose depends only on variables _x_ 1 _, . . . , xP_ for some integer _P_ and an unseen set _U ⊆_ Ω, we find Boolean functions _v_ 1 _, . . . , vn_ which only depend on the variables _x_ 1 _, . . . , xP_ such that _I_ (Ω _\ U_ ) = _⟨v_ 1 _, v_ 2 _, . . . , vn⟩_ + _I_ Ω. We know that minimum degree interpolator _f_ MDI is of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0050-09.png)


for some Boolean functions _g_ 1 _, . . . , gn_ . Now note that if we look at the equation above through the lens of Fourier-Walsh expansion, we realize that coefficients of _f_ MDI are linear combinations of Fourier coefficients of _g_ 1 _, . . . , gn_ . One can use this structure to minimize the elements of the degree-profile one by one since each element of the degree-profile is a quadratic expression in Fourier coefficients of _g_ 1 _, . . . , gn_ . Therefore, one can solve these second-degree optimization problems to calculate the unique MD interpolator. 

The process presented above is quite long, but there are some cases for which it is easier to find the minimum degree-profile interpolator. We will present some examples below. 

50 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

**Example 2 (Generalized canonical holdout)** _Given a point in {−_ 1 _,_ 1 _}[k] that is i_ = ( _i_ 1 _, . . . , ik_ ) _∈{−_ 1 _, −_ 1 _}[k] , for U_ = ( _{−_ 1 _,_ 1 _}[k] \{i}_ ) _×{−_ 1 _,_ 1 _}[d][−][k] and for any Boolean function f , the minimum degree-profile interpolator can be found as follows: we first notice that I_ (Ω _\U_ ) = _⟨x_ 1 _−i_ 1 _, . . . , xk −ik⟩_ + _I_ Ω _. And so given f , the minimum degree-profile interpolator corresponds to f_ MDI( _x_ 1 _, . . . , xk, xk_ +1 _, . . . , xd_ ) = _f_ ( _i_ 1 _, . . . , ik, xk_ +1 _, . . . , xd_ ) _._ 

**Example 3** _For U_ = _{_ ( _−_ 1 _, −_ 1) _,_ (1 _,_ 1) _} × {−_ 1 _,_ 1 _}[d][−]_[2] _and for any Boolean function f , the MD interpolator can be computed by noticing that I_ (Ω _\ U_ ) = _⟨x_ 1 + _x_ 2 _⟩_ + _I_ Ω _. Hence, given an f and in order to find the MD interpolator one should replace any x_ 1 _found by_ 2[1][(] _[x]_[1] _[−][x]_[2][)] _and all x_ 2 _by_[1] 2[(] _[x]_[2] _[ −][x]_[1][)] _[.]_ 

Here is another case where it is easy to find the MD interpolator. We further present a proof for why it is the MD interpolator in this case. 

**Lemma 34** _Let i_ = ( _i_ 1 _, . . . , ik_ ) _∈{−_ 1 _,_ 1 _}[k] be any point. For any Boolean function f and U_ = _i × {−_ 1 _,_ 1 _}[d][−][k] , we have that f has a minimum degree-profile interpolator given by replacing all x_ 1 _x_ 2 _· · · xk found by another polynomial g[′]_ ( _x_ 1 _, . . . , xk_ ) _which can be determined._ 

This is an expected result, we provide nonetheless a formal proof. **Proof** We have _I_ = _I_ (Ω _\ U_ ) = _⟨_ ( _x_ 1 + _i_ 1)( _x_ 2 + _i_ 2) _· · ·_ ( _xk_ + _ik_ ) _⟩_ + _I_ Ω. By expanding ( _x_ 1 + _i_ 1)( _x_ 2 + _i_ 2) _· · ·_ ( _xk_ + _ik_ ), we get an expression of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0051-06.png)


with _g_ ( _x_ 1 _, x_ 2 _, . . . , xk_ ) containing all the possible monomials consisting of _x_ 1 _, . . . , xk_ of degree strictly less than _k_ with coefficients being 1 or _−_ 1. Consider the polynomial _f_ MDI, by replacing all the _x_ 1 _x_ 2 _· · · xkp_ ( _xk_ +1 _, . . . , xd_ ) that appears in _f_ by _−g_ ( _x_ 1 _, x_ 2 _, . . . , xk_ ) _p_ ( _xk_ +1 _, . . . , xd_ ). We claim that _f_ MDI is the minimum degree-profile interpolator. In fact, suppose that this is not the case, so there exists a polynomial _q_ such that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0051-08.png)


is not equal to and has a lower degree-profile than _f_ MDI. For this to happen, we need at least one monomial of _f_ MDI to be (partly) replaced by the same degree or lower degree alternatives. Among all such monomials, we consider the highest degree one, _χM_ =[�] _i∈M[x][i]_[.] We assume that _M_ = _T ∪ R_ such that _T ⊆_ [ _k_ ] and _R ∩_ [ _k_ ] = _∅_ . Note that monomials that contained _x_ 1 _x_ 2 _· · · xk_ are already replaced, hence, _T_ = [ _k_ ]. We write _q_ in the form of _q_ ( _x_ ) = _s_ ( _x_ 1 _, . . . , xk_ ) _χR_ + _q[′]_ ( _x_ ) where _q[′]_ ( _x_ ) does not contain any monomial of the form _χRχT ′_ for _T[′] ⊆_ [ _k_ ]. Note that by our assumption _q_ ( _x_ )( _x_ 1 + _i_ 1) _· · ·_ ( _xk_ + _ik_ ), and thus, _s_ ( _x_ 1 _, . . . , xk_ ) _χR_ ( _x_ 1 + _i_ 1) _· · ·_ ( _xk_ + _ik_ ) must have generated _βχM_ , for some _β_ = 0. Note that _χM_ is the highest degree monomial (partly) replaced by _q_ . Thus, _χ_ [ _k_ ] _∪R_ , is not generated by _q_ , otherwise, the degree-profile would have been increased. In other words, _s_ ( _x_ 1 _, . . . , xk_ ) _χR_ ( _x_ 1 + _i_ 1) _· · ·_ ( _xk_ + _ik_ ) must have generated _βχM_ ( _β_ = 0) and not _χ_ [ _k_ ] _∪R_ . We now show that such a thing is impossible and reaches a contradiction. Assume that _s_ ( _x_ 1 _, . . . , xk_ ) =[�] _U ⊆_ [ _k_ ] _[α][U][χ][U]_[.][Notice][we][can][remove][the] _[χ][R]_[part][from][the][question.][Thus,] we can consider the equivalent statement that _s_ ( _x_ 1 _, . . . , xk_ )( _x_ 1 + _i_ 1) _· · ·_ ( _xk_ + _ik_ ) does not 

51 

Abbe, Bengio, Lotfi, and Rizk 

generate _x_ 1 _· · · xk_ while it generates _βχT_ . Now we compute the coefficients of _χT_ and _χ_ [ _k_ ] = _x_ 1 _· · · xk_ in _s_ ( _x_ 1 _, . . . , xk_ )( _x_ 1 + _i_ 1) _· · ·_ ( _xk_ + _ik_ ). We have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0052-02.png)


and the coefficient of _χT_ is[�] _U ⊆_ [ _k_ ] _[α][U]_ � _n∈_ [ _k_ ] _\_ ( _T_ ∆ _U_ ) _[i][n]_[=] _[ β]_[.][Using [] _[k]_[]] _[\]_[(] _[U]_[∆] _[T]_[) =] _[ U]_[∆([] _[k]_[]] _[\][T]_[),] we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum_images/icml_2023_Generalization_on_the_Unseen,_Logic_Reasoning_and_Degree_Curriculum.pdf-0052-04.png)


where we used the fact that _i_[2] _j_[=][1] _[∀][j]_[.][Hence,] _[β]_[=][0][which][is][a][contradiction,][showing] that it is not possible to reduce the degree-profile. 

## **References** 

Emmanuel Abbe, Enric Boix-Adsera, Matthew Brennan, Guy Bresler, and Dheeraj Nagaraj. The staircase property: How hierarchical structure can guide deep learning, NeurIPS, 2021. 

- Emmanuel Abbe, Samy Bengio, Elisabetta Cornacchia, Jon Kleinberg, Aryo Lotfi, Maithra Raghu, and Chiyuan Zhang. Learning to reason with neural networks: Generalization, unseen data and boolean measures. _Advances in Neural Information Processing Systems_ , 35:2709–2722, 2022a. 

- Emmanuel Abbe, Enric Boix-Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks, COLT, 2022b. 

- Emmanuel Abbe, Elisabetta Cornacchia, Jan Hazla, and Christopher Marquis. An initial alignment between neural network and target is needed for gradient descent to learn. In _International Conference on Machine Learning_ , pages 33–52. PMLR, 2022c. 

- Emmanuel Abbe, Samy Bengio, Aryo Lotfi, and Kevin Rizk. Generalization on the unseen, logic reasoning and degree curriculum. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 31–60. PMLR, 23–29 Jul 2023a. URL `https: //proceedings.mlr.press/v202/abbe23a.html` . 

52 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

- Emmanuel Abbe, Elisabetta Cornacchia, and Aryo Lotfi. Provable advantage of curriculum learning on parity targets with mixed inputs. _Advances in Neural Information Processing Systems_ , 36:24291–24321, 2023b. 

- Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws in language and vision. _Advances in Neural Information Processing Systems_ , 35: 22300–22312, 2022. 

- Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. Exploring length generalization in large language models. _Advances in Neural Information Processing Systems_ , 35:38546–38556, 2022. 

- Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. Phyre: A new benchmark for physical reasoning. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Peter L Bartlett, Andrea Montanari, and Alexander Rakhlin. Deep learning: a statistical viewpoint. _Acta numerica_ , 30:87–201, 2021. 

- Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira. Analysis of representations for domain adaptation. _Advances in neural information processing systems_ , 19, 2006. 

- Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In _Proceedings of the 26th Annual International Conference on Machine Learning_ , ICML ’09, page 41–48, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605585161. doi: 10.1145/1553374.1553380. URL `https: //doi.org/10.1145/1553374.1553380` . 

- Nicholas Carlini, Chang Liu, Ulfar[´] Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In _USENIX Security Symposium_ , volume 267, 2019. 

- Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL `https: //openreview.net/forum?id=TatRHT_1cK` . 

- Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In _Conference on Learning Theory_ , pages 1305– 1338. PMLR, 2020. 

- Elisabetta Cornacchia and Elchanan Mossel. A mathematical model for curriculum learning for parities. In _International Conference on Machine Learning_ , pages 6402–6423. PMLR, 2023. 

53 

Abbe, Bengio, Lotfi, and Rizk 

- David Cox, John Little, and Donal OShea. _Ideals, varieties, and algorithms: an introduction to computational algebraic geometry and commutative algebra_ . Springer Science & Business Media, 2013. 

- R´obert Csord´as, Kazuki Irie, and J¨urgen Schmidhuber. The devil is in the detail: Simple tricks improve systematic generalization of transformers. In _Proc. Conf. on Empirical Methods in Natural Language Processing (EMNLP)_ , Punta Cana, Dominican Republic, November 2021. 

- Amit Daniely and Eran Malach. Learning parities with neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, _Advances in Neural Information Processing Systems_ , volume 33, pages 20356–20365. Curran Associates, Inc., 2020. URL `https://proceedings.neurips.cc/paper/2020/file/ eaae5e04a259d09af85c108fe4d7dd0c-Paper.pdf` . 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/forum?id=YicbFdNTTy` . 

- David Steven Dummit and Richard M Foote. _Abstract algebra_ , volume 3. Wiley Hoboken, 2004. 

- Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. _Advances in Neural Information Processing Systems_ , 33:2881–2891, 2020. 

- Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Limitations of lazy training of two-layers neural network. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/ forum?id=lQdXeXDoWtI` . 

- Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit regularization in matrix factorization. _Advances in neural information processing systems_ , 30, 2017. 

- Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias in terms of optimization geometry. In _International Conference on Machine Learning_ , pages 1832–1841. PMLR, 2018a. 

- Suriya Gunasekar, Jason D Lee, Daniel Soudry, and Nati Srebro. Implicit bias of gradient descent on linear convolutional networks. _Advances in neural information processing systems_ , 31, 2018b. 

54 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

- Jason Hartford, Devon Graham, Kevin Leyton-Brown, and Siamak Ravanbakhsh. Deep models of interactions across sets. In _International Conference on Machine Learning_ , pages 1909–1918. PMLR, 2018. 

- Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. Compositionality decomposed: How do neural networks generalise? _Journal of Artificial Intelligence Research_ , 67:757–795, 2020. 

- Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. _Advances in neural information processing systems_ , 31, 2018. 

- Arthur Jacot, Fran¸cois Ged, Berfin S¸im¸sek, Cl´ement Hongler, and Franck Gabriel. Saddleto-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. _arXiv preprint arXiv:2106.15933_ , 2021. 

- Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2901–2910, 2017. 

- Nikhil Kandpal, Eric Wallace, and Colin Raffel. Deduplicating training data mitigates privacy risks in language models. In _International Conference on Machine Learning_ , pages 10697–10707. PMLR, 2022. 

- Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. 

- Tom Kocmi and Ondˇrej Bojar. Curriculum learning and minibatch bucketing in neural machine translation. In Ruslan Mitkov and Galia Angelova, editors, _Proceedings of the International Conference Recent Advances in Natural Language Processing, RANLP 2017_ , pages 379–386, Varna, Bulgaria, September 2017. INCOMA Ltd. doi: 10.26615/ 978-954-452-049-6 ~~0~~ 50. URL `https://doi.org/10.26615/978-954-452-049-6_050` . 

- Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In _International conference on machine learning_ , pages 2873–2882. PMLR, 2018. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _Advances in Neural Information Processing Systems_ , 35:3843–3857, 2022. 

55 

Abbe, Bengio, Lotfi, and Rizk 

- Sadegh Mahdavi, Kevin Swersky, Thomas Kipf, Milad Hashemi, Christos Thrampoulidis, and Renjie Liao. Towards better out-of-distribution generalization of neural algorithmic reasoning tasks. _Transactions on Machine Learning Research_ , 2023. ISSN 2835-8856. URL `https://openreview.net/forum?id=xkrtvHlp3P` . 

- Eran Malach, Pritish Kamath, Emmanuel Abbe, and Nathan Srebro. Quantifying the benefit of using differentiable learning over tangent kernels. In Marina Meila and Tong Zhang, editors, _Proceedings of the 38th International Conference on Machine Learning_ , volume 139 of _Proceedings of Machine Learning Research_ , pages 7379–7389. PMLR, 18–24 Jul 2021. URL `https://proceedings.mlr.press/v139/malach21a.html` . 

- Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation: Learning bounds and algorithms. In _Proceedings of The 22nd Annual Conference on Learning Theory (COLT 2009)_ , Montr´eal, Canada, 2009. URL `http://www.cs.nyu.edu/~mohri/ postscript/nadap.pdf` . 

- Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. _Communications on Pure and Applied Mathematics_ , 75(4):667–766, 2022. 

- Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. _Proceedings of the National Academy of Sciences_ , 115(33): E7665–E7671, 2018. 

- John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In _International Conference on Machine Learning_ , pages 7721–7735. PMLR, 2021. 

- H Michael M¨oller and Bruno Buchberger. The construction of multivariate polynomials with preassigned zeros. In _European Computer Algebra Conference_ , pages 24–31. Springer, 1982. 

- Edward Moroshko, Blake E Woodworth, Suriya Gunasekar, Jason D Lee, Nati Srebro, and Daniel Soudry. Implicit bias in deep linear classification: Initialization scale vs training accuracy. _Advances in neural information processing systems_ , 33:22182–22193, 2020. 

- Ryan O’Donnell. _Analysis of Boolean Functions_ . Cambridge University Press, 2014. doi: 10.1017/CBO9781139814782. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- Emmanouil Antonios Platanios, Otilia Stretcu, Graham Neubig, Barnabas Poczos, and Tom Mitchell. Competence-based curriculum learning for neural machine translation. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics:_ 

56 

Generalization on the Unseen, Logic Reasoning and Degree Curriculum 

_Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 1162–1172, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1119. URL `https://aclanthology.org/N19-1119` . 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ , 21(140): 1–67, 2020. 

- Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In _International Conference on Machine Learning_ , pages 5301–5310. PMLR, 2019. 

- Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. _Advances in neural information processing systems_ , 20, 2007. 

- Siamak Ravanbakhsh, Jeff Schneider, and Barnabas Poczos. Equivariance through parameter-sharing. In _International conference on machine learning_ , pages 2892–2901. PMLR, 2017. 

- Noam Razin and Nadav Cohen. Implicit regularization in deep learning may not be explainable by norms. _Advances in neural information processing systems_ , 33:21174–21187, 2020. 

- Ievgen Redko, Emilie Morvant, Amaury Habrard, Marc Sebban, and Youn`es Bennani. A survey on domain adaptation theory: learning bounds and theoretical guarantees. _arXiv preprint arXiv:2004.11829_ , 2020. 

- David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities of neural models. In _International Conference on Learning Representations_ , 2019. URL `https://openreview.net/forum?id=H1gR5iR5FX` . 

- Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, and Praneeth Netrapalli. The pitfalls of simplicity bias in neural networks. _Advances in Neural Information Processing Systems_ , 33:9573–9585, 2020. 

- Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. _Journal of Machine Learning Research_ , 19(70):1–57, 2018. 

- Valentin I Spitkovsky, Hiyan Alshawi, and Dan Jurafsky. From baby steps to leapfrog: How “less is more” in unsupervised dependency parsing. In _Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics_ , pages 751–759, 2010. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, �Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing systems_ , 30, 2017. 

57 

Abbe, Bengio, Lotfi, and Rizk 

- Petar Veliˇckovi´c, Adri`a Puigdom`enech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, and Charles Blundell. The clrs algorithmic reasoning benchmark. In _International Conference on Machine Learning_ , pages 22084– 22102. PMLR, 2022. 

- Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In _International Conference on Machine Learning_ , pages 11080–11090. PMLR, 2021. 

- Olivia Wiles, Sven Gowal, Florian Stimberg, Sylvestre-Alvise Rebuffi, Ira Ktena, Krishnamurthy Dj Dvijotham, and Ali Taylan Cemgil. A fine-grained analysis on distribution shift. In _International Conference on Learning Representations_ , 2022. URL `https://openreview.net/forum?id=Dl4LetuLdyK` . 

- Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency principle: Fourier analysis sheds light on deep neural networks, 2019. 

- Chulhee Yun, Shankar Krishnan, and Hossein Mobahi. A unifying view on implicit bias in training linear neural networks. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/forum?id=ZsZM-4iMQkH` . 

- Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. _Advances in neural information processing systems_ , 30, 2017. 

- Wojciech Zaremba and Ilya Sutskever. Learning to execute. _arXiv preprint arXiv:1410.4615_ , 2014. 

- Chiyuan Zhang, Maithra Raghu, Jon M. Kleinberg, and Samy Bengio. Pointer value retrieval: A new benchmark for understanding the limits of neural network generalization. _ArXiv_ , abs/2107.12580, 2021. 

- Yi Zhang, Arturs Backurs, S´ebastien Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner. Unveiling transformers with lego: a synthetic reasoning task. _arXiv preprint arXiv:2206.04301_ , 2022. 

- Allan Zhou, Tom Knowles, and Chelsea Finn. Meta-learning symmetries by reparameterization. In _International Conference on Learning Representations_ , 2021. URL `https://openreview.net/forum?id=-QxT4mJdijq` . 

- Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Joshua M. Susskind, Samy Bengio, and Preetum Nakkiran. What algorithms can transformers learn? a study in length generalization. In _The Twelfth International Conference on Learning Representations_ , 2024. URL `https://openreview.net/forum?id=AssIuHnmHX` . 

58 

