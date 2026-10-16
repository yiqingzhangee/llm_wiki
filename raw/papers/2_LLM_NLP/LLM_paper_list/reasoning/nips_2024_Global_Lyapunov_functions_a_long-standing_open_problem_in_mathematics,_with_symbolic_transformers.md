# **Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers** 

**Alberto Alfarano** _[∗]_ FAIR, Meta `albealfa@meta.com` 

**François Charton** _[∗]_ FAIR, Meta – Ecole des Ponts `fcharton@meta.com` 

**Amaury Hayat** _[∗]_ 

CERMICS, Ecole des Ponts - Institut Polytechnique de Paris 

```
amaury.hayat@enpc.fr
```

## **Abstract** 

Despite their spectacular progress, language models still struggle on complex reasoning tasks, such as advanced mathematics. We consider a long-standing open problem in mathematics: discovering a Lyapunov function that ensures the global stability of a dynamical system. This problem has no known general solution, and algorithmic solvers only exist for some small polynomial systems. We propose a new method for generating synthetic training samples from random solutions, and show that sequence-to-sequence transformers trained on such datasets perform better than algorithmic solvers and humans on polynomial systems, and can discover new Lyapunov functions for non-polynomial systems. 

## **1 Introduction** 

As large language models achieve human-level performance over a broad set of tasks [Bakhtin et al., 2022, Rozière et al., 2024, Zhou et al., 2023], their capability to _reason_ becomes a focus of discussion and research. There is no single definition of reasoning, and work in this area encompasses factuality, real world alignment, compositionality, the discovery and following of rules, &c. Still, mathematics are considered as one of the purest, and most demanding, forms of reasoning [Kant, 1787]. As such, solving research-level mathematical problems is a major milestone in demonstrating the reasoning capabilities of language models. Such an advance in AI would also transform mathematical practice. 

There is little research on applying language models to open problems of mathematics. Except a few papers on combinatorial optimization and graph theory [Romera-Paredes et al., 2024, Wagner, 2021], most prior works focus on problems with known solutions [Trinh et al., 2024, Lample et al., 2022, Polu and Sutskever, 2020, Charton et al., 2020]. We believe this lack of results is due to two main reasons. First, research problems may require specialized work by mathematicians [Buzzard et al., 2020] before they can be handed to language models. Second, most math transformers are trained on sets of problems and solutions which are hard to generate in the case of open problems, when no generic method for finding a solution is known. 

In this paper, we focus on a long-standing, yet easy to formalize, open problem in mathematics: discovering the Lyapunov functions that control the global stability of dynamical systems – the boundedness of their solutions when time goes to infinity with respect to an equilibrium or an orbit. A famous instance of this problem is the _three-body problem_ : the long-term stability of a system of three celestial bodies subjected to gravitation. The stability problem was studied by Newton, Lagrange and 

> _∗_ Equal contribution 

Preprint. Under review. 

Poincaré. Lyapunov discovered that stability is guaranteed if an entropy-like function for the system –the Lyapunov function– can be found. Unfortunately, no method is known for deriving Lyapunov functions in the general case, and Lyapunov functions are only known for a small number of systems. 

We propose a new technique for generating training data from randomly sampled Lyapunov functions. Sequence-to-sequence transformers trained on these datasets achieve near perfect accuracy (99%) on held-out test sets, and very high performance (73%) on out-of-distribution test sets. We show that higher accuracies (84%) can be achieved by enriching the training set with a small number (300) of easier examples that can be solved with existing algorithmic methods. These enriched models greatly outperform state-of-the-art techniques and human performance on a variety of benchmarks. 

Finally, we test the capability of our models to discover yet unknown Lyapunov functions on randomly generated systems. On polynomial systems, the only ones current methods can solve, our models find Lyapunov function for 10 _._ 1% or systems, vs 2 _._ 1% for state-of-the-art techniques. On non-polynomial systems, where no algorithm is known, our best models discover new Lyapunov functions for 12 _._ 7% of systems. Our research demonstrates that generative models can be used to solve research-level problems in mathematics, by providing mathematicians with guesses of possible solutions. The solutions proposed by the black-box model are explicit and their mathematical correctness can be verified. We believe this research is an AI-driven blueprint for solving open problems in mathematics. 

## **Related works** 

Most classical methods for finding Lyapunov rely on parameterized families of candidate solutions, and attempt to derive conditions on the parameters [Coron, 2007, Giesl, 2007]. Additional techniques such as backstepping or forwarding [Coron, 2007, Chap. 12] were introduced to leverage the specifics of particular systems. These techniques are limited to specific, or simple, systems. The global Lyapunov functions of polynomial systems that are sums of squares of polynomials of given degree can be found by computational-intensive algorithmic tools, such as SOSTOOLS [Prajna et al., 2002, 2005], which leverage the fact that the Lyapunov function belongs to a finite-dimensional space. 

Methods involving neural networks have been proposed in recent years [Chang et al., 2019, Grande et al., 2023, Edwards et al., 2024, Liu et al., 2024]. They train feed-forward networks to approximate Lyapunov functions of a given system, and use a Satisfiability Modulo Theories (SMT) solver as a verifier which proposes potential counter-examples. This approach, very different from ours, was shown to be successful for several well-studied high dimensional systems. However, it only finds local or semi-global Lyapunov functions (see Definition A.3). Since the Lyapunov functions that are found are implicit, it would be hard for mathematicians to check whether they are global Lyapunov functions or not. Semi-global Lyapunov functions are useful in many engineering fields such as robotics, where one wants a system to be robust to small perturbations. In other fields, like epidemics, being resilient to large perturbations is central, and global Lyapunov functions are required. 

Transformers trained on synthetic datasets have been proposed for many problems of mathematics, including arithmetic [Nogueira et al., 2021], linear algebra [Charton, 2022a], symbolic integration [Lample and Charton, 2019], symbolic regression [Biggio et al., 2021], Shortest Vector Problem [Wenger et al., 2022], Gröbner basis computation Kera et al. [2023] and theorem proving [Polu and Sutskever, 2020]. Charton et al. [2020] investigate a problem related to ours: the local stability of dynamical systems. Different architectures were used to solve hard problems in combinatorial optimisation [Romera-Paredes et al., 2024], and graph theory [Wagner, 2021]. 

## **2 System stability and Lyapunov functions** 

The stability of dynamical systems is a hard mathematical question, which intrigued many generations of mathematicians, from Newton and Lagrange in the 18th century, to Poincaré in the 20th in the context of the three-body problem. The main mathematical tool for assessing stability was proposed by Lyapunov, who showed in 1892 that a system is stable if a decreasing entropy-like function –the Lyapunov function– can be found [Khalil, 1992, Coron, 2007, Lyapunov, 1892]. Later, the existence of a Lyapunov function was shown to be a necessary condition for the stability of large classes of systems [Persidskii, 1937, Massera, 1949, Kellett, 2015]. Unfortunately, these very strong results provide no clue on how to find Lyapunov functions, or just proving their existence for a particular 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0003-00.png)


Figure 1: Dynamic of a stable system: trajectories may be complicated but as long as they start in the red ball they remain in the blue ball. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0003-02.png)


Figure 2: Two stable systems and associated Lyapunov functions discovered by our model. The second, a polynomial system with a non-polynomial Lyapunov function, was studied in Ahmadi et al. [2011a]. 

system. In fact, 130 years later, systematic derivations of global Lyapunov functions are only known in a few special cases, and their derivation in the general case remains a well-known open problem. In mathematical terms, we consider the dynamical system 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0003-05.png)


where _x ∈_ R _[n]_ , _f ∈ C_[1] (R _[n]_ ) and _x_ ˙ = _[dx] dt_[.][We want to know if the system has a stable equilibrium] around a point _x[∗]_ such that _f_ ( _x[∗]_ ) = 0. We assume, without loss of generality, that _x[∗]_ = 0. 

**Definition 2.1.** The system (1) is _stable_ when, for any _ε >_ 0, there exists _η >_ 0 such that, if _∥x_ (0) _∥ < η_ , the system (1) with initial condition _x_ (0) has a unique solution _x ∈ C_[1] ([0 _,_ + _∞_ )) and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0003-08.png)


In other words, a system is stable if a solution that begins close to the origin ( _∥x_ (0) _∥ < η_ ) stays close to the origin at all time ( _∥x_ ( _t_ ) _∥≤ ε_ ). Lyapunov proved that the stability is related to the existence of what is now called a Lyapunov function. 

**Definition 2.2.** The function _V ∈ C_[1] (R _[n] ,_ R+) is said to be a _(global) Lyapunov function_ for the system (1) if the following condition are satisfied 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0003-11.png)


**Theorem 2.3** (Lyapunov 1892) **.** _If the system_ (1) _has a Lyapunov function, then it is stable._ 

In fact, the existence of a Lyapunov function is more powerful and provides additional information. **Theorem 2.4** (LaSalle, 1961) **.** _If the system_ (1) _has a Lyapunov function V , then all the solutions of_ (1) _converge to the largest invariant set of {f_ ( _x_ ) _· ∇V_ ( _x_ ) = 0 _}._ 

In many cases this largest invariant set is reduced to _{x[∗]_ = 0 _}_ and the system is said _globally asymptotically stable_ (all solutions converge to the equilibrium, see Appendix A). 

Most dynamical systems are unstable.grow exponentially with time, and the solutions ofFor instance, the solutions of the simple system _x_ ˙ ( _t_ ) = 1 + _x_ ( _t_ )[2] ( _x ∈_ R) always blow up before _x_ ˙ ( _t_ ) = _x_ ( _t_ ) _t_ = _π_ . No Lyapunov functions can be found for these systems. 

On the other hand, stable systems can have an infinite number of Lyapunov functions. The system 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0003-17.png)


has _V_ ( _x_ ) = _a_ 0 _x_[2] 0[+] _[ a]_[1] _[x]_[2] 1[as a Lyapunov function for any choice of] _[ a]_[0] _[>]_[ 0][ and] _[ a]_[1] _[>]_[ 0][.] 

3 

In the general case, there is no systematic way of discovering a Lyapunov function, or even showing that one exist. Tools exist for small polynomial systems with special “sum of squares” (SOS) Lyapunov functions, but they need a lot of resources, do not always find a solution, and fail once the systems involve more than a few variables. 

We also consider a related, but easier, problem: finding nontrivial _V_ which are semi-definite positive, i.e. _V_ verifying _V_ ( _x_ ) _≥_ 0 instead of _V_ ( _x_ ) _>_ 0 in Equation (3). These functions, called _barrier functions_ , form “barriers” that divide R _[n]_ into two subspaces. A solution starting inside the barrier must remains in the same subspace, which is an invariant set of the system [Prajna et al., 2007, Xu et al., 2015]. For polynomial systems, barrier functions are slightly easier to find using SOS solvers. 

## **3 Experimental settings** 

In this work, we train sequence-to-sequence transformers [Vaswani et al., 2017] to predict a Lyapunov function for a given system, when it exists. We frame the problem as a translation task: problems and solutions are represented as sequences of symbolic tokens, and the model is trained from generated pairs of systems and Lyapunov functions to minimize the cross-entropy between the predicted sequence and the correct solution. We train transformers with 8 layers, 10 attention heads and an embedding dimension of 640 (ablation studies on different model sizes can be found in Appendix C), on batches of 16 examples, using the Adam optimizer [Kingma and Ba, 2014] with a learning rate of 10 _[−]_[4] , an initial linear warm-up phase of 10,000 optimization steps, and inverse square root scheduling. All experiments run on 8 V100 GPU with 32 GB of memory, for 3 or 4 epochs of 2 _._ 4 million examples per epoch. Training time is between 12 to 15 hours per GPU. 

**Tokenization.** Model inputs are systems of the form ( ˙ _xi_ = _fi_ ( _x_ 1 _, . . . , xn_ )) _i∈{_ 1 _,...,n}_ , represented by the _n_ functions _fi_ . Model outputs are single functions _V_ ( _x_ 1 _, . . . , xn_ ). As in Lample and Charton [2019], functions are represented as trees, with operators in their internal nodes, and variables or constants as their leaves. Trees are then enumerated in Polish (pre-order) notation to produce sequences of tokens that can be processed by the transformer. 

All operators and variables are tokenized as single symbols (e.g. ‘ `cos` ’ or ‘ _x_ 1’). Integer constants are tokenized as sequences of “digits” in base 1000 (e.g. 1024 as the sequence [ `+, 1, 24` ]), and real constants, in scientific notation, as pairs of two integers (mantissa and exponent, e.g. _−_ 3 _._ 14 as [ `-` _,_ 314 _,_ `10ˆ` _,_ `-` _,_ 2]). For instance: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0004-06.png)


enumerated as the sequences: [ _∗,_ cos _, ∗,_ 2 _._ 1 _, x_ 0 _,_ + _, x_ 1 _,_ 2] and [sin _,_ + _, ∗,_ 3 _, x_ 1 _,_ 2], and finally tokenized as [ `*` , `cos` , `*` , `21` , `10ˆ` , `-` , `1` , _x_ 0, `+` , _x_ 1, `2` , `SEP` , `sin` , `+` , `*` , `3` , _x_ 1, `2` ] (using `SEP` as a separator). 

**Evaluation.** Trained models are evaluated on sets of stable systems. Since systems have an infinite number of Lyapunov functions, we cannot check the model predictions by comparing them to the solutions from the test set, and need to use an external verifier. For polynomial systems, we verify that there exists a small positive polynomial _P_ such that _−∇V · f_ and _V − P_ are sum of squares (SOS) of polynomials (with _P_ = 0 for barrier functions), using a Python solver based on SumOfSquares [Yuan, 2024]. For non-polynomial systems, we also use a verifier based on `shgo` that checks (3) numerically. To further ensure correctness we also verify the symbolic solutions using Satisfiability Modulo Theories (SMT) solvers, relying on dReal [Gao et al., 2013] for verification through interval analysis. This guarantees that equations (3) hold, at least in a chosen ball around the origin. The performances of the two verifiers (numerical solver and SMT) are similar, a comparison is provided in Table 1. Both the SOS and SMT verifiers sometimes fail to return an answer. In that case, we classify the solution as wrong, even though it might have been correct. As a result, model accuracies may be underestimated. 

4 

Model predictions use beam search with early stopping, normalizing log-likelihood scores by their sequence length. We report results with beam size 1 (greedy decoding) and beam size 50. With beam size 50, we consider the model to be correct if one Lyapunov function is found among the 50 guesses. 

## **4 Data generation** 

Our models are trained and tested on large datasets of pairs of stable systems and associated Lyapunov functions. Sampling such stable systems raises two difficulties. First, most dynamical systems are unstable, and no general method exists for deciding whether a system is stable. Second, once a stable system is sampled, there is no general technique for finding a Lyapunov function, except in particular cases. In this paper, we rely on **Backward generation** [Lample and Charton, 2019], sampling solutions and generating associated problems, for the general case, and **forward generation** , sampling systems and calculating their solutions with a solver, for the tractable polynomial systems of small degree. 

## **4.1 Backward generation** 

Backward generation methods, sampling problems from their solutions, are only useful if the model can be prevented from learning to reverse the generation procedure, or from “reading” the solutions in the generated problems. For instance, when training a model to solve the hard problem of finding the roots of an integer polynomial [Charton, 2022b], one can easily generate a polynomial from its roots, i.e. from the roots 3 _,_ 5 and 7, generate the polynomial: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0005-05.png)


However, if the model is trained from factorized form of _P_ ( _X_ ), it will learn to read the roots in the problem, instead of computing them. On the other hand, the developed and simplified form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0005-07.png)


offers no clues. A second difficulty of backward generation is that sampling solutions instead of problems biases the training distribution. A model trained on backward-generated data may not perform well on a forward-generated test set. Finally, prior work Yehuda et al. [2020] observed that, for hard problems, backward generation methods sometimes focus on easier sub-problems (see, for instance, our comment below about choosing _f_ = _−∇V_ in step 2). 

We propose a procedure for generating a stable system _S_ from a random Lyapunov function _V_ . The rationale is the following. Since _V_ must be positive with a strict minimum in 0, and tend to infinity at infinity ((3)), we first generate _V_ = _V_ proper + _V_ cross where _V_ proper belongs to a class of functions with a guaranteed strict minimum in zero and _V_ cross to a larger class of non-negative functions, valued 0 at the origin, but with no guarantee of a strict minimum (step 1 and Appendix B). From _V_ , we need to generate _f_ so that the third condition of (3) is met. A naive solution would be _f_ = _−∇V_ since _f · ∇V ≤_ 0 would hold. But this would severely limit the systems we create, and turn the Lyapunov function discovery problem (find _V_ from _f_ ) into an easier integration problem (find _V_ from _−∇V_ ). Instead, starting from _f_ 0 = _−∇V_ , we apply the following transformations: 

- multiply each coordinate of _f_ 0 by random non-negative functions _h_[2] _i_[(step 4) and call it] _[f]_[˜][0][.] 

- • generate a random function _ϕ_ =[�] _[p] i_ =1 _[g][i]_[(] _[x]_[)] _[e][i]_[(] _[x]_[)][ (steps 2 and 3), where] _[ e][i]_[ are orthogonal] to _∇V_ ( _x_ ), and set _f_ = _φ_ + _f_[˜] 0. We have _ϕ · ∇V_ = 0 and ( _ϕ_ + _f_[˜] 0) _· ∇V ≤_ 0. 

These transformations guarantee that all conditions in (3) are met. On the other hand, they allow _f_ to span a very large set of systems, since any _f_ satisfying _∇V_ ( _x_ ) _· f_ ( _x_ ) _≤_ 0 can be written as the sum of a function collinear to _∇V_ ( _x_ ) and a function orthogonal to _∇V_ ( _x_ ). 

Specifically, the procedure can be summarized as follows (see Appendix B for more details). 

**Step 1** Generate a random function _V_ , satisfying _V_ ( _x_ ) _> V_ (0) _, ∀x ∈_ R _[n] \ {_ 0 _}_ , and _V_ ( _x_ ) _→_ + _∞_ when _∥x∥→_ + _∞_ . **Step 2** Compute the gradient _∇V_ ( _x_ ) and denote _Hx_ = _{z ∈_ R _[n] | z · ∇V_ ( _x_ ) = 0 _}_ the hyperplane[2] orthogonal to _∇V_ ( _x_ ), for any _x ∈_ R _[n]_ . 

> 2if _∇V_ ( _x_ ) = 0 this is the whole space instead, but this does not change the method. 

5 

|Timeout|SMT Solver<br>10 minutes<br>60 minutes|SOS Solver<br>10 minutes<br>60 minutes|
|---|---|---|
||||
|Correct Lyap function<br>Solver Timeouts<br>Incorrect Lyap function|82.6<br>94.1<br>17.4<br>5.9<br>0<br>0|89.6<br>95.3<br>10.4<br>4.7<br>0<br>0|



Table 1: **SMT and SOS timeout and error rates,** benchmarked on correct Lyapunov functions. 

**Step 3** Select 1 _≤ p ≤ n_ at random and sample _p_ vectors _{e[i]_ ( _x_ ) _}i∈{_ 1 _,...,p}_ from hyperplane _Hx_ . Generate _p_ real-valued functions ( _gi_ ) _i∈{_ 1 _,...,p}_ . **Step 4** Select 1 _< k_ 1 _≤ n_ at random, generate _k_ 1 random real-valued functions ( _hi_ ) _i∈{_ 1 _,...,k_ 1 _}_ , set _hi_ = 0 for _k_ 1 + 1 _≤ i ≤ n_ . **Step 5** Build the _n_ functions 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0006-03.png)


with _π_ a random permutation of _{_ 1 _, ..., n}_ . 

**Step 6** Simplify the functions _fi_ , obscuring patterns from the generative process. 

This method produces a stable system _S_ : _x_ ˙ = _f_ ( _x_ ), with _V_ as its Lyapunov function. The difficulty of inferring _V_ from _S_ hinges on a careful choice of the vectors _e[i]_ . For instance, if we naively select _e[i]_ as an orthonormal basis of _Hx_ , computed from _∇V_ ( _x_ ) by Gram-Schmidt orthogonalization, prefactors like 1 _/∥∇V_ ( _x_ ) _∥_ appear at step 3, and are unlikely to simplify away at step 6. This provides the model with a shortcut: reading _∥∇V_ ( _x_ ) _∥_ in _S_ , and using it to recover _∇V_ and then _V_ , not a trivial task, but an easier one than discovering Lyapunov functions. To counter this, we relax the orthonormality condition on _e[i]_ ( _x_ ), so that 1 _/∥∇V_ ( _x_ ) _∥_ never appears, yet keep the _e[i]_ ( _x_ ) simple enough for _∇V_ -specific patterns in[�] _i[g][i]_[(] _[x]_[)] _[e][i]_[(] _[x]_[)][ to simplify away at step 6.][We also want to ensure] that the _e[i]_ span all of _Hx_ , or the systems generated will not be diverse enough. 

In our experiments, we slightly modify this procedure, by running steps 2 to 6 five times for each Lyapunov function _V_ created at step 1. As a result, 5 systems are generated that share the same Lyapunov function (a discussion of this choice can be found in Appendix C.1). From a mathematical point of view, a Lyapunov function describes a hidden quantity in a system, and we believe that providing the model with several systems that share this hidden quantity should help it learn the parts of the system that contribute to this hidden quantity, and therefore learn a Lyapunov function. 

This procedure can be tuned to generate specific classes of systems. By choosing _V_ , _gi_ and _hi_ in particular classes, we can constrain the system functions _fi_ to be polynomials, polynomials of functions (e.g. trigonometric polynomials), or more general functions (see Appendix B.4 for more). 

The Lyapunov functions obtained here are correct by design. Nevertheless, we still performed an evaluation of the solutions both as a safeguard and to benchmark the failure and timeout rates of the SMT and SOS solvers on correct solutions, which we report in Table 1. 

## **4.2 Forward generation** 

Whereas the stability problem is unsolved in the general case, methods exist to calculate Lyapunov functions of polynomial systems, when they exist and can be written as a sum of squares of polynomials (see Section 1). These algorithms, of polynomial complexity, are very efficient for small systems, but their CPU and memory requirements explode as the size of the systems grows. We leverage them to generate forward datasets, as follows. 

**Step 1** Generate a polynomial system at random 

**Step 2** Use a routine to find a polynomial sum-of-squares (SOS) Lyapunov function. **Step 3** Keep the system if such function exists, restart from step 1 otherwise. 

This approach has several limitations. First, since most polynomial systems are not stable, and the computation of SOS Lyapunov function involves a complicated search [Prajna et al., 2005], it is slow and limited to small systems of polynomials with small degree. Second, because not all stable polynomial systems have polynomial SOS Lyapunov functions [Ahmadi et al., 2011a], it can only generate a subset of stable polynomial systems. 

6 

Finally, SOS routines process the constraints in Equation (3) by solving semi-definite programming (SDP) problems. This guarantees that _V_ is a sum-of-squares, hence we have _V_ ( _x_ ) _≥_ 0, but not necessarily _V_ ( _x_ ) _>_ 0, for _x_ = 0. As a result, these methods can only discover _barrier functions_ . State-of-the-art methods circumvent this by introducing the stronger constraint _V_ ( _x_ ) _≥_[�] _[n] i_ =1 _[ε][i][x] i_[2][,] with _εi_ small Prajna et al. [2002]. _V_ then has a unique minimum in _x_ = 0, which makes it a Lyapunov function, but this further restricts the class of polynomial systems that the method can solve. 

## **4.3 Datasets** 

We generate 2 backward and 2 forward datasets for training and evaluation purpose, and one smaller forward dataset for evaluation purposes (see Table 8 in Appendix B.6 for a list). 

**Backward datasets** Our main backward set, **BPoly** , features 1 million non-degenerate polynomial systems _S_ with integer coefficients, and 2 to 5 equations (in equal proportions). We also create **BNonPoly** , a dataset of 1 million non-degenerate non-polynomial systems with 2 to 5 equations. In this dataset, the coordinates of _f_ are polynomials of general functions, e.g. trigonometric polynomials, or functions such as 3 cos( _x_ 1) + 2 _x_ 1 _e[x]_[2] . For such general systems, no method for discovering a Lyapunov function is known. 

**Forward datasets** All 2 forward datasets are generated using a solver derived from the SumOfSquares package in Python, and implementing techniques similar to those used in SOSTOOLS (see Appendix B.5). All systems in these datasets are non-zero integer polynomials with 2 to 3 equations, and integer polynomial Lyapunov functions – the only systems these methods can solve. We create **FLyap** , a dataset of 100,000 systems having a non-homogeneous polynomial as a Lyapunov function. We also have a dataset focusing on barrier functions (see the end of section 4.2): **FBarr** features 300,000 systems having a non-homogeneous polynomial as a barrier function. The small size of these datasets is due to the computational cost of SOS methods, and the difficulty of discovering Lyapunov or barrier functions. 

To allow for comparison with SOSTOOL, the state-of-the-art method for discovering Lyapunov functions of polynomial systems, we also generated a test set of 1,500 polynomial systems with integer coefficients that SOSTOOLS can solve ( **FSOSTOOLS** ). 

## **5 Results** 

Our models trained on different datasets achieve near perfect accuracy on held-out test sets, and very high performances on out-of-distribution test sets, especially when enriching the training set with a small number of forward examples. They greatly outperform state-of-the-art techniques and also allow to discover Lyapunov functions for new systems. These results are detailed below. 

## **5.1 In and out-of-distribution accuracy** 

In this section, we present the performance of models trained on the 4 datasets. All models achieve high in-domain accuracy – when tested on held-out test sets from the datasets they were trained on (Table 2). On the forward datasets, barrier functions are predicted with more than 90% accuracy, and Lyapunov functions with more than 80%. On backward datasets, models trained on BPoly achieve close to 100% accuracy. We note that beam search, i.e. allowing several guesses at the solution, brings a significant increase in performance (7 to 10% with beam size 50, for the low-performing models). We use beam size 50 in all further experiments. 

|Accuracy<br>Backward datasets<br>bs=1<br>bs=50|Accuracy<br>Forward datasets<br>bs=1<br>bs=50|
|---|---|
|||
|BPoly (polynomial)<br>99<br>100<br>BNonPoly (non-poly)<br>77<br>87|FBarr (barrier)<br>93<br>98<br>FLyap (Lyapunov)<br>81<br>88|



Table 2: **In-domain accuracy of models** . Beam size (bs) 1 and 50. 

The litmus test for models trained on generated data is their ability to generalize out-of-distribution (OOD). Table 3 presents evaluations of backward models on forward-generated sets (and the other 

7 

way around). All backward models achieve high accuracy (73 to 75%) when tested on forwardgenerated random polynomial systems with a sum-of-squares Lyapunov functions (FLyap). The best performances are achieved by non-polynomial systems (BNonPoly), the most diverse training set. The lower accuracy of backward models on forward-generated sets of systems with barrier functions (FBarr) may be due to the fact that many barrier functions are not necessarily Lyapunov functions. On those test sets, backward models must cope with a different distribution and a (slightly) different task. Forward models, on the other hand, achieve low performance on backward test sets. This is possibly due to the small size of these training set. 

Overall, these results seem to confirm that backward-trained models are not learning to invert their generative procedure. If it were the case, their performance on the forward test sets would be close to zero. They also display good OOD accuracy. 

|Backward datasets<br>FLyap<br>FBarr|Forward datasets<br>BPoly|
|---|---|
|||
|BPoly (polynomial)<br>**73**<br>35<br>BNonPoly (non-poly)<br>**75**<br>24|FBarr (barrier)<br>15<br>FLyap (Lyapunov)<br>10|



Table 3: **Out-of-domain accuracy of models** . Beam size 50. Columns are the test sets. 

## **5.2 Enriching training distributions for improved performance** 

To improve the OOD performance of backward models, we add to their training set a tiny number of forward-generated examples, as in Jelassi et al. [2023]. Interestingly, this brings a significant increase in performance (Table 4). Adding 300 examples from FBarr to BPoly brings accuracy on FBarr from 35 to 89% (even though the proportion of forward examples in the training set is only 0 _._ 03%) and increases OOD accuracy on FLyap by more than 10 points. Adding examples from FLyap brings less improvement. 

These results indicate that the OOD performance of models trained on backward-generated data can be greatly improved by adding to the training set a small number of examples (tens or hundreds) that we know how to solve. Here, the additional examples solve a weaker but related problem: discovering barrier functions. The small number of examples needed to boost performance makes this technique especially cost-effective. 

|Forward<br>Examples added<br>datasets<br>(1M in training set)|FLyap<br>FBarr|
|---|---|
|No mixture<br>0<br>73<br>35||
|||
|FBarr<br>30<br>300<br>3,000<br>30,000|75<br>61<br>83<br>89<br>85<br>93<br>89<br>95|
|||
|FLyap<br>10<br>100<br>1,000<br>10,000|75<br>25<br>82<br>29<br>83<br>37<br>86<br>38|



Table 4: **Mixing backward data (BPoly) with a small number of forward examples** . Beam size 50. 

## **5.3 Comparing with state-of-the-art baselines** 

To provide a baseline for our models, we developed `findlyap` , a Python counterpart to the MATLAB Lyapunov function finder from SOSTOOLS (see Appendix B.5). We also introduce FSOSTOOLS, a test set of 1,500 polynomial systems with integer coefficients that SOSTOOLS can solve. We also tested AI-based tools (see Appendix E for the full list of parameters sweeps we used for each of these methods), such as Fossil 2 [Edwards et al., 2024], ANLC v2 [Grande et al., 2023] and LyzNet [Liu et al., 2024]. These methods achieve low accuracies on our test sets. This might be due to the fact that these tools are designed to solve a different problem: discovering local or semi-global Lyapunov function (and potentially finding a control function), while we target global Lyapunov functions. 

8 

|Test sets|SOSTOOL<br>`findlyap`|Existing AI methods<br>Fossil 2<br>ANLC<br>LyzNet|Models<br>PolyMixture<br>FBarr<br>FLyap<br>BPoly|
|---|---|---|---|
|||||
|FSOSTOOLS<br>FBarr<br>FLyap<br>BPoly|-<br>-<br>-<br>15|32<br>30<br>46<br>12<br>18<br>28<br>42<br>32<br>66<br>10<br>6<br>24|**84**<br>80<br>53<br>54<br>**89**<br>-<br>28<br>35<br>83<br>**93**<br>-<br>73<br>**99**<br>15<br>10<br>-|



Table 5: **Performance comparison on different test sets** . Beam size 50. PolyMixture is BPoly + 300 FBarr. 

Table 5 compares `findlyap` and AI-based tools to our models on all available test sets. A model trained on BPoly complemented with 500 systems from FBarr (PolyMixture) achieves 84% on FSOSTOOLS, confirming the high OOD accuracy of mixture models. On all generated test sets, PolyMixture achieves accuracies over 84% whereas `findlyap` achieves 15% on the backward generated test set. This demonstrates that, on polynomial systems, transformers trained from backward-generated data achieve very strong results compared to the previous state of the art. 

On average Transformer-based models are also much faster than SOS methods. When trying to solve a random polynomial system with 2 to 5 equations (as used in Section 5.4), `findlyap` takes an average of 935.2s (with a timeout of 2400s). For our models, inference and verification of one system takes 2.6s on average with greedy decoding, and 13.9s with beam size 50. 

## **5.4 Into the wild - discovering new mathematics** 

Our ultimate goal is to discover new Lyapunov functions. To test our models’ ability to do so, we generate three datasets of random systems: polynomials systems with 2 or 3 equations ( **Poly3** ), polynomial systems with 2 to 5 equations ( **Poly5** ), and non-polynomial systems with 2 or 3 equations ( **NonPoly** ). For each dataset, we generate 100,000 random systems and eliminate those that are trivially locally exponentially unstable in _x[∗]_ = 0, because the Jacobian of the system has an eigenvalue with strictly positive real part [Khalil, 1992]. We compare `findlyap` and AI based methods with two models trained on polynomial systems, FBarr, and PolyM(ixture) – a mixture of BPoly and 300 examples from FBarr– and one model trained on a mixture of BPoly, BNonPoly and 300 examples from FBarr (NonPolyM). 

Table 6 presents the percentage of correct solutions found by our models. On the polynomial datasets, our best model (PolyM) discover Lyapunov functions for 11 _._ 8 and 10 _._ 1% of the (degree 3 and degree 5) systems, ten times more than `findlyap` . For non-polynomial systems, Lyapunov functions are found for 12 _._ 7% of examples. These results demonstrate that language model trained from generated datasets of systems and Lyapunov function can indeed discover yet unknown Lyapunov functions and perform at a much higher level that state-of-the-art SOS solvers. 

|Sample<br>Test set<br>size|SOSTOOL<br>`findlyap`|Existing AI methods<br>Fossil 2<br>ANLC<br>LyzNet|Forward<br>FBarr|Backward models<br>PolyM<br>NonPolyM|
|---|---|---|---|---|
||||||
|Poly3<br>65,215<br>Poly5<br>60,412<br>NonPoly<br>19,746|1.1<br>0.7<br>-|0.9<br>0.6<br>4.3<br>0.3<br>0.2<br>2.1<br>1.0<br>0.6<br>3.5|11.7<br>8.0<br>-|**11.8**<br>11.2<br>**10.1**<br>9.9<br>-<br>**12.7**|



Table 6: **Discovering Lyapunov comparison for random systems** . Beam size 50. PolyM is BPoly + 300 FBarr. NonPolyM is BNonPoly + BPoly + 300 FBarr. 

## **5.5 Expert iteration** 

Given the performance on our model in Table 6, we can use the newly solved problems to further fine-tune the model. Specifically, we create a sample of verified model predictions for polynomial systems, **FIntoTheWild** , we add it to the original training sample and we continue training the model. 

We test different strategy to finetune the model and we report performance on forward benchmarks and “into the wild” in Table 7. 

- _n1_ : Add 20,600 samples from BPoly (20,000), FBarr (50), FLyap (50) and FIntoTheWild (500) to keep similar proportion used during pretraining 

9 

- _n2_ : Add 2,000 samples from FLyap (1,000) and FIntoTheWild (1,000) to improve on both forward benchmark and in the wild 

- _n3_ : Add 50 samples from FIntoTheWild to show that this indeed helps 

- _n4_ : Add 1,000 samples from FIntoTheWild 

- _n5_ : Add 2,000 samples from FIntoTheWild 

- _n6_ : Add 5,000 samples from FIntoTheWild to see if there are benefits to add more samples 

We also retrain a model ( _n7_ ) from scratch using a mixture of BPoly (1M), FBarr (500), FLyap (500) and FIntoTheWild (2,000). 

|Strategy|Forward benchmark<br>FBarr<br>FLyap|Regenerated IntoTheWild<br>Poly3<br>Poly5|
|---|---|---|
||||
|Baseline|93<br>84|11.7<br>9.6|
||||
|_n1_<br>_n2_<br>_n3_<br>_n4_<br>_n5_<br>_n6_<br>_n7_|94<br>84<br>90<br>85<br>92<br>84<br>92<br>84<br>89<br>79<br>85<br>72<br>93<br>81|10.3<br>9.6<br>12.2<br>11.3<br>12.4<br>10.1<br>**13.5**<br>**11.9**<br>13.5<br>11.9<br>13.5<br>11.9<br>12.1<br>10.0|



Table 7: **Expert iteration using IntoTheWild correct guesses** . The Poly3 and Poly5 test sets are regenerated, to prevent data contamination. 

We notice that the addition of 1,000 verified predictions to our training set of 1 million improves performance on the “into to wild” test sets by about 15%, while not affecting the other test sets ( _n4_ ). Adding more examples seems to be detrimental, as it decreases the performance on other benchmarks ( _n5_ and _n6_ ). We also notice that finetuning with mixed data from other distributions is not efficient ( _n1_ and _n2_ ) and a small contribution already help to get some improvements (result _n3_ ). Finally, it’s not efficient to pretrain the model from scratch using data from FIntoTheWild ( _n7_ ). 

## **6 Discussion** 

We have shown that models can be trained from generated datasets to solve a long-standing open problem in mathematics: discovering the Lyapunov functions of stable dynamical systems. For random polynomial systems, our best models can discover Lyapunov functions in five times more cases than state-of-the-art methods. They can also discover Lyapunov functions of non-polynomial systems, for which no algorithm is yet known, and were able to re-discover a non-polynomial Lyapunov function of a polynomial systems discovered by Ahmadi et al. [2011a] (Appendix F). 

The backward generation method introduced in section 4.1 is the key innovation in this paper. The main problem with such approaches is their tendency to generate training sets with very specific distributions, which prevent models from generalizing to general instances of the problem. Our models can generalize out of their training distributions (Table 3), and we can improve their performance by adding to their training set a tiny number of systems that we know how to solve (Table 5). 

While our models exceed the algorithmic state of the art, one might wonder **how they compare to human mathematicians** . To this effect, we proposed 75 problems from the FSOSTOOLS dataset (polynomial systems with 2 or 3 equations) as an examination for 25 first year Masters students in mathematics, following a course on the subject. Each student was given 3 systems chosen at random and had a total of 30 min. Their performance was 9.33%, significantly lower than our models (84%). 

Our work has a number of limitations. Because there is no known way to tell whether a random system is stable, we lack a good benchmark on non-polynomial systems. Also, all the systems studied in this paper are relatively small, at most 5 equations for polynomial systems and 3 for non-polynomial. We believe that scaling to larger models should help tackle larger, and more complex, systems. Finally, this work could be extended to take into account the domain of definition of non-polynomial systems. 

The broader implications of our work extend into two directions: the capability of transformers to reason, and the potential role of AI in scientific discovery. While large language models perform at human level on a broad set of tasks, they are still embarrassingly clumsy on many simple problems 

10 

of logic and reasoning, to the point that it was suggested that planning and high level reasoning may be an inherent limitation of auto-regressive transformer architectures. Our results suggest that transformers can indeed be trained to discover solutions to a hard problem of symbolic mathematics that humans solve through reasoning, and that this is enabled by a careful selection of training examples, instead of a change of architecture. We do not claim that the Transformer is reasoning but it may instead solve the problem by a kind of “super-intuition” that stems from a deep understanding of a mathematical problem. 

From a mathematical point of view, we propose a new, AI-based, procedure for finding Lyapunov functions, for a broader class of systems than were previously solvable using current mathematical theories. While this systematic procedure remains a black box, and the “thought process” of the transformer cannot be elucidated, the solutions are explicit and their mathematical correctness can be verified. This suggests that generative models can already be used to solve research-level problems in mathematics, by providing mathematicians with guesses of possible solutions. While a small minority of mathematicians is currently using deep-learning tools, we believe generative models have the potential to foster tremendous progress on a number of research subjects, and may eventually become a central component in the future landscape of mathematical practice. 

## **Acknowledgements** 

This work was performed in part using HPC resources from GENCI–IDRIS (Grant 2023AD011014527). The authors also acknowledge the Office of Advanced Research Computing (OARC) at Rutgers, The State University of New Jersey. The authors would also like to thank the Master students of the Mathematics and Computer Science Department of the Ecole des Ponts - IP Paris from the year 2023-2024 who attended the course Control of Dynamical Systems. 

## **References** 

- Amir Ali Ahmadi, Miroslav Krstic, and Pablo A. Parrilo. A globally asymptotically stable polynomial vector field with no polynomial lyapunov function. In _2011 50th IEEE Conference on Decision and Control and European Control Conference_ , pages 7579–7580, 2011a. doi: 10.1109/CDC. 2011.6161499. 

- Amir Ali Ahmadi, Miroslav Krstic, and Pablo A. Parrilo. A globally asymptotically stable polynomial vector field with no polynomial lyapunov function. In _2011 50th IEEE Conference on Decision and Control and European Control Conference_ , pages 7579–7580, 2011b. doi: 10.1109/CDC. 2011.6161499. 

- Martin S. Andersen, Joachim Dahl, and Lieven Vandenberghe. Cvxopt. `https://cvxopt.org/` , 2023. URL `https://cvxopt.org/` . Version 1.3.2; Free software package for convex optimization based on Python. Last updated August 9, 2023. 

- Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, Athul Paul Jacob, Mojtaba Komeili, Karthik Konath, Minae Kwon, Adam Lerer, Mike Lewis, Alexander H. Miller, Sasha Mitts, Adithya Renduchintala, Stephen Roller, Dirk Rowe, Weiyan Shi, Joe Spisak, Alexander Wei, David Wu, Hugh Zhang, and Markus Zijlstra. Human-level play in the game of <i>diplomacy</i> by combining language models with strategic reasoning. _Science_ , 378(6624):1067–1074, 2022. 

- Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurelien Lucchi, and Giambattista Parascandolo. Neural symbolic regression that scales, 2021. 

- Kevin Buzzard, Johan Commelin, and Patrick Massot. Formalising perfectoid spaces. In _Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs_ . ACM, jan 2020. doi: 10.1145/3372885.3373830. URL `https://doi.org/10.1145%2F3372885. 3373830` . 

- Ya-Chien Chang, Nima Roohi, and Sicun Gao. Neural Lyapunov Control. In _Advances in Neural Information Processing Systems_ , volume 32. Curran Associates, Inc., 2019. URL `https://proceedings.neurips.cc/paper/2019/file/ 2647c1dba23bc0e0f9cdf75339e120d2-Paper.pdf` . 

11 

- François Charton, Amaury Hayat, and Guillaume Lample. Learning advanced mathematical computations from examples. _arXiv preprint arXiv:2006.06462_ , 2020. 

François Charton. Linear algebra with transformers, 2022a. 

François Charton. Computing the roots of polynomials, 2022b. `https://f-charton.github.io/ polynomial-roots` . 

Jean-Michel Coron. _Control and nonlinearity_ . American Mathematical Soc., 2007. 

- Alec Edwards, Andrea Peruffo, and Alessandro Abate. Fossil 2.0: Formal certificate synthesis for the verification and control of dynamical models. In _Proceedings of the 27th ACM International Conference on Hybrid Systems: Computation and Control_ , HSCC ’24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400705229. doi: 10.1145/3641513.3651398. URL `https://doi.org/10.1145/3641513.3651398` . 

- Sicun Gao, Soonho Kong, and Edmund M. Clarke. dreal: An smt solver for nonlinear theories over the reals. In Maria Paola Bonacina, editor, _Automated Deduction – CADE-24_ , pages 208–214, Berlin, Heidelberg, 2013. Springer Berlin Heidelberg. 

- Peter Giesl. _Construction of global Lyapunov functions using radial basis functions_ , volume 1904. Springer, 2007. 

- Davide Grande, Andrea Peruffo, Enrico Anderlini, and Georgios Salavasidis. Augmented Neural Lyapunov Control. _IEEE Access_ , 11:67979–67986, 2023. doi: 10.1109/ACCESS.2023.3291349. 

- Samy Jelassi, Stéphane d’Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, and François Charton. Length generalization in arithmetic transformers. _arXiv preprint arXiv:2306.15400_ , 2023. 

Immanuel Kant. _Critique of pure reason_ . 1787. 

- Christopher M Kellett. Classical converse theorems in lyapunov’s second method. _Discrete & Continuous Dynamical Systems-Series B_ , 20(8), 2015. 

- Hiroshi Kera, Yuki Ishihara, Yuta Kambe, Tristan Vaccon, and Kazuhiro Yokoyama. Learning to compute gr _\_ " obner bases. _arXiv preprint arXiv:2311.12904_ , 2023. 

- Hassan K. Khalil. _Nonlinear systems_ . Macmillan Publishing Company, New York, 1992. ISBN 0-02-363541-X. 

- Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Guillaume Lample and François Charton. Deep learning for symbolic mathematics. _arXiv preprint arXiv:1912.01412_ , 2019. 

- Guillaume Lample, Marie-Anne Lachaux, Thibaut Lavril, Xavier Martinet, Amaury Hayat, Gabriel Ebner, Aurélien Rodriguez, and Timothée Lacroix. HyperTree Proof Search for Neural Theorem Proving. _Advances in neural information processing systems_ , 2022. 

- Jun Liu, Yiming Meng, Maxwell Fitzsimmons, and Ruikun Zhou. Lyznet: A lightweight python tool for learning and verifying neural lyapunov functions and regions of attraction, 2024. URL `https://arxiv.org/abs/2403.10013` . 

- Aleksandr Mikha˘ılovich Lyapunov. _The general problem of the stability of motion_ . PhD thesis, Kharkov University, 1892. 

- Jose Luis Massera. On liapounoff’s conditions of stability. _Annals of Mathematics_ , pages 705–721, 1949. 

- Rodrigo Frassetto Nogueira, Zhiying Jiang, and Jimmy Lin. Investigating the limitations of the transformers with simple arithmetic tasks. _CoRR_ , abs/2102.13019, 2021. URL `https://arxiv. org/abs/2102.13019` . 

12 

- KP Persidskii. On a theorem of liapunov. In _CR (Dokl.) Acad. Sci. URSS_ , volume 14, pages 541–543, 1937. 

- Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving, 2020. 

- Stephen Prajna, Antonis Papachristodoulou, and Pablo A Parrilo. Introducing sostools: A general purpose sum of squares programming solver. In _Proceedings of the 41st IEEE Conference on Decision and Control, 2002._ , volume 1, pages 741–746. IEEE, 2002. 

- Stephen Prajna, Antonis Papachristodoulou, Peter Seiler, and Pablo A Parrilo. Sostools and its control applications. _Positive polynomials in control_ , pages 273–292, 2005. 

- Stephen Prajna, Ali Jadbabaie, and George J Pappas. A framework for worst-case and stochastic safety verification using barrier certificates. _IEEE Transactions on Automatic Control_ , 52(8): 1415–1428, 2007. 

- Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. _Nature_ , 625(7995):468–475, 2024. 

- Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code, 2024. 

- Guillaume Sagnol and Maximilian Stahlberg. PICOS: A Python interface to conic optimization solvers. _Journal of Open Source Software_ , 7(70):3915, February 2022. ISSN 2475-9066. doi: 10.21105/joss.03915. 

- Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. _Nature_ , 625(7995):476–482, 2024. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in neural information processing systems_ , pages 5998–6008, 2017. 

- Adam Zsolt Wagner. Constructions in combinatorics via neural networks. _arXiv preprint arXiv:2104.14516_ , 2021. 

- Emily Wenger, Mingjie Chen, Francois Charton, and Kristin E Lauter. Salsa: Attacking lattice cryptography with transformers. _Advances in Neural Information Processing Systems_ , 35:34981– 34994, 2022. 

- Xiangru Xu, Paulo Tabuada, Jessy W Grizzle, and Aaron D Ames. Robustness of control barrier functions for safety critical control. _IFAC-PapersOnLine_ , 48(27):54–61, 2015. 

- Gal Yehuda, Moshe Gabel, and Assaf Schuster. It’s not what machines can learn, it’s what we cannot teach. _arXiv preprint arXiv:2002.09398_ , 2020. 

- Chenyang Yuan. Sumofsquares.py, 2024. URL `https://sums-of-squares. github.io/sos/index.html#python` . Software version 1.2.0; Available at https://github.com/yuanchenyang/SumOfSquares.py. 

- Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers, 2023. 

13 

## **Appendix** 

## **A Mathematical definitions** 

In this Appendix, we recall several mathematical definitions and theorems related to the Lyapunov function problem. We first introduce the notion of global asymptotic stability (GAS). 

**Definition A.1.** We say that the (equilibrium _x[∗]_ = 0 of the) system (1) is _globally asymptotically stable_ if it is stable and for any _x_ 0 _∈_ R _[n]_ there exists a unique solution _x ∈ C_[1] ([0 _,_ + _∞_ ); R _[n]_ ) to (1) which satisfies in addition 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0014-04.png)


This notion translates the fact that the equilibrium _x[∗]_ = 0 is robust even to large perturbations. This notion is related to the existence of a Lyapunov function thanks, for instance, to LaSalle Invariance Principle: 

**Theorem A.2** (LaSalle Invariance Principle (global)) **.** _Assume there exists a Lyapunov function for the system_ (1) _and let S be the largest subset of {∇V_ ( _x_ ) _· f_ ( _x_ ) = 0 _} that is invariant by the dynamics of_ (1) _. If S_ = _{_ 0 _}, then the system_ (1) _is globally asymptotically stable._ 

Note that if _∇V_ ( _x_ ) _· f_ ( _x_ ) _<_ 0 for any _x ̸_ = 0 then necessarily _S_ = _{_ 0 _}_ . Because finding a (global) Lyapunov function is a challenging mathematical problem, and still an open problem in general, weaker notions exists. 

**Definition A.3.** The function _V ∈ C_[1] (R _[n] ,_ R+) is said to be a _semi-global Lyapunov function_ for the system (1) if there exists _r >_ 0 such that the following condition are satisfied 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0014-09.png)


Finding a semi-global Lyapunov function is usually easier than finding a global Lyapunov function. A semi-global Lyapunov function is enough to show that the equilibrium _x[∗]_ = 0 is robust to small perturbations which, for several engineering applications, is enough. More specifically, 

**Definition A.4.** We say that the (equilibrium _x[∗]_ = 0 of the) system (1) is _locally asymptotically stable_ if it is stable and if there exists _r >_ 0 such that for any _∥x_ 0 _∥≤ r_ there exists a unique solution _x ∈ C_[1] ([0 _,_ + _∞_ ); R _[n]_ ) to (1) which satisfies in addition 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0014-12.png)


Similarly to global Lyapunov function, the existence of a semi-global Lyapunov function is useful to ensure local asymptotic stability 

**Theorem A.5** (LaSalle Invariance Principle (local)) **.** _Assume there exists a semi-global Lyapunov function V , and let S be the largest subset of {∇V_ ( _x_ ) _· f_ ( _x_ ) = 0 _} invariant by the dynamics of_ (1) _. If S_ = _{_ 0 _} then the system_ (1) _is locally asymptotically stable._ 

## **B Generation procedure** 

## **B.1 Function generation** 

To generate random functions we sample random trees with unary and binary internal nodes, and then randomly select operators for these nodes, and variables and integers for leaves (as in Lample and Charton [2019], Charton et al. [2020]). Our binary operators are the four operations and the power function. Unary operators are exp _,_ log _,_ sqrt _,_ sin _,_ cos _,_ tan. 

To generate polynomials, we randomly sample a given number of monomials, with integer or real coefficients. The number of monomials, range of the coefficients, and the powers and number of terms of each monomial, are randomly selected between bounds, provided as hyperparameters. 

14 

## **B.2 Backward generation** 

We build globally stable systems by first generating a Lyapunov function _V_ at random, and then building a dynamic system which has _V_ as a Lyapunov function. The procedure is: 

**Step 1a:** We generate _V_ as _V_ = _V_ cross + _V_ proper where _V_ proper belongs to a given class of positive definite function and _V_ cross belongs to a larger class, but of non-negative functions only with _V_ cross(0) = 0. More specifically, we generate 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-03.png)


with _m_ a random integer, and _pi_ random functions verifying _pi_ (0) = 0. The nature of functions _pi_ depends on the systems we want to generate (polynomial or not). Clearly _V_ cross( _x_ ) _≥_ 0 and _V_ cross(0) = 0. We similarly generate 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-05.png)


with _n_ a random integer, _βi_ random positive integers and _A_ = ( _αi,j_ )( _i,j_ ) _∈{_ 1 _,...,n}_ 2 a random positive definite matrix, with a given probability of being diagonal. As a consequence, _V_ proper is strictly minimal in _x_ = 0. When generating barrier functions, we can optionally set _V_ proper = 0. 

**Step 1b:** In this step, we increase the class of functions that can be sampled for _V_ cross and _V_ proper by several transformations: 

1. **Composition of** _V_ **proper** with probability _p_ 1 _,c_ , replace 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-09.png)


- with _I_ selected at random from a pre-defined set of _increasing-functions_ (Appendix B.4), 

- 2. **Product** _V_ **proper** with probability _p_ 1 _,m_ , replace 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-11.png)


with _g_ selected at random from a pre-defined set of _positive-functions_ (Appendix B.4), _h_ a sub-expression of _V_ proper, i.e. _h_ ( _x_ ) = � _q ασ_ ( _i_ ) _,σ_ ( _j_ ) _xβσσ_ ( _i_ () _i_ ) _[x] βσσ_ ( _j_ ( _j_ ))[,][for] _[q][≤][n]_[and] _[σ]_[a] _i_ =1 permutation of _{_ 1 _, . . . , n}_ . 

3. **Composition of** _V_ **cross** : for every _i ∈{_ 1 _, ..., m}_ , with probability _p_ 2, replace 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-14.png)


with _bi_ a real function that is bounded from below with a minimum (not necessarily unique) in _ξi_ and chosen at random from a pre-defined set of _bounded-functions_ (Appendix B.4). Recall that _pi_ are the functions appearing in _V_ cross. 

**Step 1c:** Gathering the functions _V_ proper and _V_ cross together, we define the Lyapunov function (candidate) _V_ ( _x_ ) = _V_ cross( _x_ ) + _V_ proper( _x_ ). Overall, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-17.png)


where _I_ is the identity with probability 1 _− p_ 1 _,c_ , _g_ is the constant function 1 with probability 1 _− p_ 1 _,m_ and _bk_ ( _x_ ) = _x_[2] with probability 1 _− p_ 2. Such a Lyapunov function satisfies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-19.png)


Indeed, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0015-21.png)


and _V_ ( _x_ ) _> bk_ ( _ξk_ ) for any _x ∈_ R _[n] \ {_ 0 _}_ , since _g_ is a positive function, _I_ is increasing and ( _αi,j_ ) _i,j∈{_ 1 _,...,n}_ is positive definite. 

15 

**Step 2:** In this step we create the random vectors orthogonal to _∇V_ that will be useful in the generation of the system _f_ (see Section 4.1). Taking advantage of the form of the condition (3), for any _x ∈_ R, denote 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0016-01.png)


the hyperplane orthogonal to _∇V_ ( _x_ ). Then, for a random _p ∈{_ 1 _, ..., n}_ , generate _p_ random real-valued functions ( _gi_ ) _i∈{_ 1 _,...,p}_ , and _p_ vectors _{e[i] }i∈{_ 1 _,...,p}_ from this hyperplane as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0016-03.png)


where _A_ = _∇V_ ( _x_ ) and _Aj_ refers to the _j − th_ component of the vector and _τ_ 1 and _τ_ 2 random functions from N _\{_ 0 _}_ into _{_ 1 _, ..., n}_ , such that _τ_ 1( _i_ ) _̸_ = _τ_ 2( _i_ ). This implies that for any _i ∈{_ 1 _, ..., n}_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0016-05.png)


Note that, so long _∇V_ ( _x_ ) _̸_ = 0, one can use this process to construct a generative family of _Hx_ , and the _e[i]_ span the whole _Hx_ . If _∇V_ ( _x_ ) = 0 then _Hx_ = R _[n]_ . 

**Step 3:** Generate at random _k_ 1 real-valued functions ( _hi_ ) _i∈{_ 1 _,...,k_ 1 _}_ , where 1 _≤ k_ 1 _≤ n_ is chosen at random. Set _hi_ = 0 for _k_ 1 _< i ≤ n_ . 

**Step 4:** Build the system 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0016-09.png)


with _π_ a random permutation of _{_ 1 _, ..., n}_ . 

Overall, the function _f_ satisfies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0016-12.png)


hence _V_ is a Lyapunov function of the system _x_ ˙ ( _t_ ) = _f_ ( _x_ ( _t_ )). 

**Step 5:** Expand and simplify the equations of _f_ (using Sympy), in order to eliminate obvious patterns due to the generation steps (that the model could recognize and leverage), eliminate duplicated systems in the training set, and limit the length of training sequences. All polynomial systems are expanded into normal form. 

## **B.3 Backward generation modes** 

**Polynomial generation** : we generate polynomial systems with sum-of-square Lyapunov functions to allow for easy comparison with existing methods such as SOSTOOLS Prajna et al. [2002, 2005]. In this case, all _Pi_ are polynomials with no zero-order term and _p_ 1 _,c_ = _p_ 1 _,m_ = _p_ 2 = 0. Also, _fi_ and _gi_ are polynomials (Appendix B.1). We generate _fi_ with a degree lower or equal to half the maximal degree of _gi_ and a maximal value of coefficients of the order of the square root of the maximal value of _gi_ . Since the _fi_ are squared in the final system, this allows _fi_[2][and] _[ g][i]_[ to have the same order, and] prevents the transformer from inferring unwanted additional information by looking at the higher degree monomial. 

**Generic generation** : _Pi_ is generated as _Pi_ ( _x_ ) = _Qi_ ( _x_ ) _− Qi_ (0), where _Qi_ ( _x_ ) is a random function generated as per Appendix B.1 and _fi_ and _gi_ are also generated as per Appendix B.1. Optionally the functions can be generated as polynomials of non-polynomial operators taken from a pre-defined set of _operators_ . 

**Other generation modes** : we have other generation modes corresponding to interesting particular cases: gradient flow systems, systems where the 2-norm (resp. a weighted 2-norm) is a Lyapunov function, etc. 

16 

## **B.4 Generation design parameters** 

Our generator allows us to generate generic stable systems and yet to have a large control on the distribution. For polynomials, for instance, we have a control on the maximal and average degree, number of monomials, power and number of variables of the monomials, coefficients, etc. We can also specify whether the coefficients are integers, floats, with which precision. Overall we have a total of 36 generation hyper-parameters that influence the distribution of the synthetic data created. The main generation design parameters are: 

- int_base: encoding base for integers 

- max_int: Maximum integer value 

- precision: Float numbers precision 

- prob_int: Probability of sampling integers vs variables (for non-polynomial expressions) 

- min_dim: minimal number of equations in the system 

- max_dim: maximal number of equations 

- max_degree: maximal degree of polynomial terms in a Lyapunov function 

- n_terms: maximal number of terms in polynomials for the Lyapunov function 

- nb_ops_proper: maximal number of operators in _Vproper_ (non polynomial generation) 

- nb_ops_lyap: maximal number of operators in _Vproper_ (non polynomial generation) 

- operators_lyap: list of operators to be considered (non polynomial generation) 

- polynomial_V: if true generated expressions are polynomials of (potentially non-polynomial) operators 

- pure_polynomial: generate polynomial systems only 

- cross_term: _Vcross_ = 0 if False. 

- max_nb_cross_term: bound on m in _Vcross_ 

- proba_diagonal: with this probability, the positive definite form of _Vproper_ is imposed to be diagonal 

- only_2_norm: if True, the Lyapunov function is the 2-norm. 

- strict: if True, generates a strict Lyapunov function (i.e. _∇V · f <_ 0) 

- proper: if set to false, _Vproper_ = 0 and _V_ is only a barrier function. 

- float_resolution_poly: float resolution of the polynomials generated by generate_bounded_polynomial. 

- generate_gradient_flow: When set to True, the backward generation only generates gradient flows systems. 

- gen_weight: exponential weight which bias the sampling of _k_ 1 and _p_ , the number of components of non-zero _hi_ and _gi_ . 

- max_order_pure_poly: maximal polynomial order of _hi_ 

- max_n_term_fwd: maximal number of terms in each equations in the fwd generation 

- SOS_checker: if True, uses a SOS verifier to evaluate the candidate Lyapunov function (if False uses the verifier based on `shgo` ) 

- SMT_checker: if True, uses an SMT verifier to evaluate the candidate Lyapunov function (if False uses the verifier based on `shgo` ) 

- multigen: number of different system generated per Lyapunov function. 

- increasing_func: the set of increasing functions used in the generation (see Step 1b). Default is _{_ exp _,_ ln(1 + _x_[2] ) _,[√]_ 1 + _x}_ . 

- positive_func: the set of positive functions used in the generation (see Step 1b). Default is _{_ exp _,_ 1 + cos( _x_ ) _,_ 1 + sin( _x_ ) _}_ . 

- bounded_func: the set of bounded functions used in the generation (see Step 1b). Default is _{_ cos _,_ sin _}_ . 

## **B.5 Forward SOS solver** 

SOSTOOLS is one of the most famous toolbox for sum-of-square optimization, in particular for finding SOS Lyapunov functions Prajna et al. [2002, 2005]. It is natively available in MATLAB and relies on an underlying SDP solver that can be chosen. In Python an analogous toolbox is the package SumOfSquares Yuan [2024] which relies on the same principle, however does not have specific functionalities for Lyapunov functions. As a consequence we implemented these functionalities in our codebase based on the MATLAB implementations in SOSTOOLS. We implemented a function `SOS_checker` , which takes in input a system of equations in sympy and a candidate Lyapunov function and checks SOS conditions on _V_ ( _x_ ) and _−∇V_ ( _x_ ) _· f_ ( _x_ ), and a function `findlyap` , 

17 

analogous to the `findlyap` function in SOSTOOLS, which takes a system of equations in sympy and either returns a function satisfying SOS conditions on _V_ ( _x_ ) and _−∇V_ ( _x_ ) _· f_ ( _x_ ), returns false if no such function exists, or returns none if it fails to provide an answer. SumOfSquares relies itself on picos Sagnol and Stahlberg [2022] and we use the default solver cvxopt Andersen et al. [2023]. 

## **B.6 List of datasets** 

|Dataset|Description<br>Size<br>Resources<br>(000)<br>(CPU.hours)|
|---|---|
|||
|BPoly<br>BNonPoly|Backward polynomial systems, non-zero<br>1,000<br>210<br>Backward non-polynomial systems, non-zero<br>1,000<br>220|
|||
|FBarr<br>FLyap<br>FSOSTOOLS|Forward, non-homogeneous polynomial barrier functions<br>300<br>9,670<br>Forward, homogeneous polynomial Lyapunov functions<br>100<br>4,620<br>Forward, SOSTOOLS solved systems<br>1.5|



Table 8: **Datasets generated.** Backward systems are degree 2 to 5, forward systems degree 2 to 3. All forward systems are polynomial. 

## **C Additional results** 

## **C.1 Impact of multigeneration** 

In the backward generation procedure, after sampling one random _V_ , it is possible to generate any number of different systems _fi_ such that _V_ is the Lyapunov function for each of the systems _fi_ . We call the maximal number of system generated per Lyapunov function the `multigen` parameter. The actual number of systems generated per Lyapunov function is chosen at random for each Lyapunov function between 1 and `multigen` . In Section 5 we reported results using `multigen` equal to 5. Here we report the in-domain and out-of-domain performance of the models trained on backward BPoly datasets of size 1 million varying the parameter `multigen` . 

||In-domain|OOD|
|---|---|---|
|Multigen|BPoly|FLyap|
|1|95|58|
|5|100|73|
|10|100|75|
|25|100|76|
|50|100|70|
|100|100|68|



Table 9: **In-domain and out-of-domain accuracy of models** . Beam size 50. 

Table 9 shows that generating a moderate amount of different systems with the same Lyapunov function actually improves the model capability to generalize out-of-domain. This suggests that the model is learning, at least partially, to separate the parts of the system which contribute to the Lyapunov function. Above a certain multigen threshold, model performances start to decline. This may be due to the low diversity present in the dataset, i.e. the limited number of different Lyapunov functions the model is trained on (the total number of systems in the training set remains constant so the total number of Lyapunov function decreases with the value of the parameter `multigen` ). 

## **C.2 Performance of smaller transformer models** 

In Section 5 we report results using a transformer with 8 encoder and decoder layers, 10 attention heads and an embedding dimension of 640. We also trained smaller models with 6 encoder and decoder layers, 8 attention heads and an embedding dimension of 512. Tables 10, 11 report the main results. Results are in line with what we showed in section 5 

18 

|In-domain<br>OOD<br>Backward datasets<br>FLyap|In-domain<br>OOD<br>Forward datasets<br>BPoly|
|---|---|
|||
|BPoly (polynomial)<br>100<br>70<br>BNonPoly (non-poly)<br>85<br>71|FBarr (barrier)<br>97<br>13<br>FLyap (Lyapunov)<br>86<br>11|



Table 10: **In-domain and out-of-domain accuracy of models** . Beam size 50. 

|Forward<br>Mixing<br>datasets<br>proportion|Into the wild<br>FBarr<br>FLyap<br>Poly3<br>Poly5|
|---|---|
|||
|No mixture<br>0%|31<br>70<br>2.3<br>1.6|
|FBarr<br>0.01%<br>60<br>72<br>2.5<br>1.7<br>0.1%<br>93<br>72<br>9.2<br>6.4||
|FLyap<br>0.01%<br>29<br>73<br>2.8<br>1.6<br>0.1%<br>19<br>76<br>2.9<br>1.7||



Table 11: **Performance of mixing backward data (BPoly) with a small number of forward examples on forward benchmark and “into the wild”** . Beam size 50. 

## **D Comparison of SOS, SMT and** `shgo` 

We compare our model performance when we employ them to discover new Lyapunov function. We report performances with dReal SMT and SOS verifiers for Poly and dReal SMT and `shgo` for NonPoly distributions, respectively. Table 12 shows that SMT results are slightly lower, because of timeouts (which we report in Table 13, but comparable. Note that the performances on polynomial systems were already theoretically guaranteed thanks to the former SOS verifier. 

|Model (by training distribution)|FBarr|BPolyMixture|NonPolyMixture|
|---|---|---|---|
|Poly3|10.5/11.7|11.1/11.8|10.6/11.2|
|Poly5|6.5/8.0|8.7/10.1|8.4/9.9|
|NonPoly|||8.3/12.7|



Table 12: **Results of SMT** with SOS and `shgo` verifiers for Poly and NonPoly systems, respectively. 

|Test sets|Into|the wild|
|---|---|---|
|Timeout|10 minutes|120 minutes|
|Correct Lyap function|87.3|92.2|
|SMT Timeouts|10.8|5.8|
|Incorrect Lyap function|1.9|2.0|



Table 13: **SMT timeout and error rates.** Most SMT failures are due to timeout. 

## **E AI method sweep** 

To report the AI-based tools results on the seven benchmarks (BPoly, BNonPoly, FLyap, FBarr, Poly3, Poly5, NonPoly) we did a hyperparameter sweep. To get the best hyperparameter setting, we sweep on FLyap and then fix these hyperparameters for the different datasets. In bold we show the chosen parameters, selected to maximize the correctness on FLyap, subject to the 20 minutes timeout. 

## **Lyznet [Liu et al., 2024]** 

- lr = [3 _·_ 10 _[−]_[5] , **10** _[−]_ **[4]** , 3 _·_ 10 _[−]_[4] ] 

- points = [ **100,000** , 300,000, 1,000,000] 

- layer width = [(2,20), **(3,6)** , (6,2)] 

- epoch = [1, 5, **25** ] 

- net type = [None, **Poly** ] 

19 

## **Fossil 2.0 [Edwards et al., 2024]** 

- iters = [10, **50** , 250] 

- activations = [( **x[2]** ), ( _x_[2] , _x_[2] ), (sigmoid), (sigmoid, sigmoid), (poly4), (poly4, poly4)] 

- hidden neurons = [6, **10** , 20] 

- data = [500, **1000** , 2000] 

- lr = [0.01, 0.03, **0.1** ] 

## **ANLC v2 [Grande et al., 2023]** 

- iters = [10, **50** , 250] 

- activations = [( **x[2]** , **x** , **x** ), ( _x_[2] , _x_[2] , _x_ ), ( _x_[2] , _x_[2] , _x_ , _x_ ), ( _x_[2] , _x_[2] , _x_[2] , _x_ )] 

- hidden neurons = [6, **10** , 20] 

- max data = [500, **1000** , 2000] 

- lr = [ **0.01** , 0.03, 0.1] 

## **F Some examples** 

To understand the model performance and compare against the SOSTOOL performance, we manually inspect some systems with 2 or 3 equations where the following conditions hold: (1) the Jacobian of the system has the maximum eigenvalue with real part equal to 0 (i.e. tools like the spectral mapping theorem cannot decide on the stability), (2) no weighted 2-norm functions can be a Lyapunov function, (3) `findlyap` times out after 4 hours. We show some examples below. 

## **F.1 A polynomial system with non polynomial solution** 

## **System** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0020-16.png)


It’s known that there is no polynomial Lyapunov function for this system [Ahmadi et al., 2011b]. Our poly models and `findlyap` failed, as expected. Nonetheless, one of our non-poly models with beam search of beam size 100 proposed _V_ ( _x_ ) = ln(1 + 5 _x_[2] 0[) +] _[ x]_ 1[2][similar to the one that was recently] found in [Ahmadi et al., 2011b]. 

It’s clear that _V_ (0) = 0 and _V_ ( _x_ ) _>_ 0 for all _x ̸_ = 0. Also 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0020-19.png)


as desired. 

## **F.2 A system that has no diagonal Lyapunov function** 

## **System** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0020-23.png)


**Model inference** : Our model recovers _V_ ( _x_ ) = 10 _x_[2] 0[+ 2] _[x]_[0] _[x]_[2] 1[+ 3] _[x]_[4] 1[+ 6] _[x]_[2] 1[.] 

Clearly _V_ (0) = 0 and _V_ ( _x_ ) = 9( _x_ 0)[2] + ( _x_ 0 + _x_[2] 1[)][2][+ 2(] _[x]_[2] 1[)][2][+ 6] _[x]_[2] 1 _[>]_[0][for][all] _[x]_[=][0][.][Also] _∇V_ ( _x_ ) _· f_ ( _x_ ) = _−x_[2] 1[(116] _[x]_[2] 1[+ 120)] _[ ≤]_[0][.] 

**Non existence of a Diagonal Lyapunov function** : Suppose for the sake of contradiction that there exists a function _V_ 1 which satisfies 3 and can be expressed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0020-27.png)


20 

Clearly _V_ 1(0) = 0. Given that _V_ 1( _x_ 0 _,_ 0) _>_ 0 for _x_ 0 = 0, it follows that _n_ is even and _an >_ 0. Also _n m_ we know that _∇V_ 1( _x_ ) _· f_ ( _x_ ) = 2 � _iaix[i]_ 0 _[−]_[1] _x_[2] 1 _[−]_[10] � _jbjx[j]_ 1 _[≤]_[0][ for all choices of][ (] _[x]_[0] _[, x]_[1][)][.][If] _i_ =1 _j_ =1 _n m_ we let _x_ 1 = 1 we obtain _∇V_ 1( _x_ ) _· f_ ( _x_ ) = 2 � _iaix_ 0 _[i][−]_[1] _−_ 10 � _jbj_ . This expression can be seen _i_ =1 _j_ =1 

as a polynomial _g_ ( _x_ 0) with real coefficients and odd degree _n −_ 1. The leading coefficient, 2 _nan_ , is positive because _an >_ 0 and _n ≥_ 1. This means that lim _x_ 0 _→_ + _∞[g]_[(] _[x]_[0][) = +] _[∞]_[, meaning that there exists] an _x_ 0 such that _g_ ( _x_ 0) _>_ 0. This contradicts 3. 

## **F.3 A system with 3 equations and a higher degree** 

## **System** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0021-04.png)


**Model inference** : Our model recovers different solutions. Here we show two of them 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers_images/nips_2024_Global_Lyapunov_functions_a_long-standing_open_problem_in_mathematics,_with_symbolic_transformers.pdf-0021-06.png)


We checked with SumOfSquares that _V_ 1 _>_ 0 _, V_ 2 _>_ 0 _, ∇V_ 1 _· f ≤_ 0 and _∇V_ 2 _· f ≤_ 0. 

## **F.4 Other examples** 

|System|Lyapunov function|
|---|---|
|||
|�<br>˙_x_0 =_−_5_x_3<br>0 _−_2_x_0_x_2<br>1<br>˙_x_1 =_−_9_x_4<br>0 + 3_x_3<br>0_x_1 _−_4_x_3<br>1|_V_(_x_) = 6_x_6<br>0 + 7_x_4<br>0 +_x_3<br>0 + 10_x_2<br>0 + 8_x_2<br>1|
|||
|�<br>˙_x_0 =_−x_5<br>0 _−_4_x_3<br>0 _−_9_x_0_x_4<br>1 + 3_x_0_x_3<br>1<br>˙_x_1 =_−_3_x_4<br>0_x_2<br>1 _−_10_x_3<br>0_x_1 + 3_x_0_x_2<br>1 _−_7_x_3<br>1|_V_(_x_) =_x_4<br>0 + 9_x_2<br>0 + 3_x_2<br>1|
|||
|<br><br><br><br><br>˙_x_0 =_−_3_x_3<br>0 + 3_x_0_x_2 _−_9_x_0<br>˙_x_1 =_−x_3<br>0 _−_5_x_1 + 5_x_2<br>2<br>˙_x_2 =_−_9_x_3<br>2|_V_(_x_) =_x_4<br>0 + 7_x_2<br>0_x_2<br>2 + 3_x_2<br>0 + 4_x_0_x_2<br>2 + 3_x_2<br>1 + 2_x_4<br>2 + 10_x_2<br>2|
|||
|<br><br><br><br><br>˙_x_0 =_−_8_x_0_x_2<br>1 _−_10_x_4<br>1<br>˙_x_1 =_−_8_x_3<br>1 + 3_x_2<br>1 _−_8_x_1<br>˙_x_2 =_−x_2|_V_(_x_) = 4_x_2<br>0 _−_2_x_0_x_2<br>1 + 6_x_4<br>1 + 4_x_2<br>1 +_x_2<br>2|



Table 14: **Some additional examples generated from our models** . 

21 

