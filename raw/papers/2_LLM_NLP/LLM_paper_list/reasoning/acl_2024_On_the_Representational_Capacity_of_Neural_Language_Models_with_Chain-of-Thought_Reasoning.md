# **On the Representational Capacity of Neural Language Models with Chain-of-Thought Reasoning** 

**Franz Nowak**[*] **Anej Svete**[*] **Alexandra Butoi Ryan Cotterell** {franz.nowak, anej.svete, alexandra.butoi, ryan.cotterell}@inf.ethz.ch 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0001-03.png)


## **Abstract** 

The performance of modern language models (LMs) has been improved by chain-of-thought (CoT) reasoning, i.e., the process of generating intermediate results that guide the model towards a final answer. A possible explanation for this improvement is that CoT reasoning extends an LM’s computational power, as RNNs and transformers with additional scratch space are known to be Turing complete. Comparing LMs to Turing machines, however, introduces a category error—Turing machines decide language membership, whereas LMs define _distributions_ over strings. To bridge this gap, we formalize CoT reasoning in a probabilistic setting. We present several results on the representational capacity of recurrent and transformer LMs with CoT reasoning, showing that they can represent the same family of distributions over strings as _probabilistic_ Turing machines. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0001-06.png)


https://github.com/rycolab/cot-lms 

## **1 Introduction** 

Motivated by how humans solve complex problems by noting down intermediary results, Wei et al. (2022b) introduced **chain-of-thought** (CoT) reasoning.[1] CoT reasoning helps language models (LMs) solve reasoning tasks by allowing them to store intermediary results in a scratch space that is _not_ part of the final output. The empirical success of CoT reasoning has made it an established component of modern neural LMs (Nye et al., 2021; Wei et al., 2022a,b; Suzgun et al., 2022; Kojima et al., 2023; Wies et al., 2023)—seemingly overnight. Such empirical success motivates a thorough understanding of the abilities of CoT-augmented LMs. While existing theoretical treatments have shed light on some aspects of this framework, we are still far from a concrete theoretical understanding of CoT (Feng et al., 2023). 

CoT reasoning can be interpreted as encouraging LMs to perform additional sequential computation 

> *Equal contribution. 

> 1 We use the more general term CoT _reasoning_ over the original term CoT _prompting_ as prompting is just one way to elicit CoT reasoning (Wang and Zhou, 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0001-13.png)


**----- Start of picture text -----**<br>
Probabilistic unrestricted languages<br>Unbounded precision Thm. 4.5<br>Transformers FGGGGGGGGGGGGGGGGGGGGGGGGGGB Probabilistic<br>Thm. 4.6<br>Unbounded precision EGGGGGGGGGGGGGGGGGGGGGGGGGGC Turing machines<br>Elman RNNs Thm. 4.4<br>Probabilistic regular languages<br>Fixed-precision GGGGGGGGGGGGGBThm. 4.2 Probabilistic<br>Elman RNNs FGGGGGGGGGGGGG<br>Thm. 4.1 Finite State<br>Fixed-precision DGGGGGGGGGGGGG Automata<br>Transformers Thm. 4.3<br>Neural architecture Corresponding model<br>with CoT reasoning of computation<br>**----- End of picture text -----**<br>


Figure 1: A schematic overview of our main results, showing which neural language models extended with CoT reasoning (left) correspond to which probabilistic models of computation (right). 

steps while storing the result of that computation. Naturally, this suggests that CoT reasoning should be formalized in a manner that exploits our understanding of well-known, stateful models of computation. Theory of computation provides a natural toolbox for this avenue of exploration. Indeed, the internal configurations of many neural language models have been previously linked to intermediate steps in neural LMs (Pérez et al., 2021; Merrill and Sabharwal, 2024; Feng et al., 2023). 

A **language model** is definitionally a distribution over strings from some fixed alphabet.[2] This definition is somewhat at odds with the way existing work has investigated the representational capacity of CoT-augmented LMs. Firstly, most results in this area concern unweighted language _recognition_ , which is different from the inherently probabilistic task of language modeling. Secondly, Turing completeness results such as Pérez et al.’s (2021) construction—a seminal result showing the Turing completeness of transformer LMs— usually require _additional symbols_ not present in the original alphabet to correctly simulate a Turing machine. Because the output alphabets of the transformer LM and the Turing machine it simulates fundamentally differ, it is difficult 

> 2An alphabet is a finite, non-empty set. 

to discuss their _equivalence_ . This motivates a more fine-grained treatment of the representational capacity of neural LMs in which: _(1)_ LMs are treated as _probabilistic_ models of computation, i.e., they assign weights to strings rather than deciding language membership, and _(2)_ language model equivalence is defined on the output string level, and the analysis takes into account the additional information required to be encoded in additional outputs to achieve the equivalence. 

We address the first point by analyzing CoTaugmented LMs using _probabilistic_ models of computation, which provide a convenient way of describing families of distributions over strings with standard models of sequential reasoning. To address the second point, we define a new type of relationship between language models, which we call **regular reducibility** . Intuitively, an LM is regularly reducible into another one if the strings generated by the latter are sufficiently simple transformations of those generated by the former. An instance of such a transformation is the deletion of the intermediary computation steps of CoT reasoning. Formalizing CoT reasoning in this framework, we find that it increases the computational power in both RNN and transformer LMs. Concretely, we find that CoT-augmented constant-precision RNN LMs are equivalent to probabilistic finitestate automata (PFSAs). This is in contrast to the constant-precision RNN LMs _without_ CoT reasoning, which are equivalent to _deterministic_ PFSAs (Svete and Cotterell, 2023). Additionally, we show how Turing-complete linear-precision RNN LMs can be thought of as performing CoT reasoning. Finally, we show that both linearly bounded precision RNN LMs and logarithmically bounded precision decoder-only transformer LMs with CoTreasoning can simulate any probabilistic Turing machine. Taken together, our results frame CoT reasoning in pure language modeling terms and describe its intuitive and formal connections to probabilistic models of computation. 

## **2 Preliminaries** 

In our analysis, we assume rational (rather than real) arithmetic for all definitions and computations. 

## **2.1 Language Models** 

A **formal language** _L_ is a subset of the Kleene closure Σ _[∗]_ of some alphabet Σ. We call an element 

_**y**_ of Σ _[∗]_ a **string** . We denote the empty string by _ε_ , and we assume that _ε ∈/_ Σ. A **discrete semimeasure** over Σ _[∗]_ is a function _µ_ : Σ _[∗] →_ [0 _,_ 1] such that � _**y** ∈_ Σ _[∗][µ]_[ (] _**[y]**_[)] _[≤]_[1][ (][Bauwens][,][ 2013][;][ Icard][,][ 2020][).] If the semimeasure of all strings sums to one, i.e., � _**y** ∈_ Σ _[∗][µ]_[ (] _**[y]**_[)][=][1][, then] _[ µ]_[ is called a] **[ probability] measure** .[3] Finally, for any alphabet Σ, we define the set Σ _ε_ = Σdef _∪{ε}_ . 

**Definition 2.1.** _A_ _**language model** (LM) p is a semimeasure over_ Σ _[∗] . If p is a probability measure, it is called a_ _**tight** language model._ 

**Definition 2.2.** _Two LMs p and q are_ _**weakly equivalent** if p_ ( _**y**_ ) = _q_ ( _**y**_ ) _for all_ _**y** ∈_ Σ _[∗] ._ 

Most modern LMs are autoregressive, i.e., they define _p_ ( _**y**_ ) through conditional distributions of the next symbol given the string produced so far and the probability of ending the string, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0002-10.png)


Here, EOS denotes the special end-of-string symbol, which specifies that the generation of a string has halted. The inclusion of EOS allows (but does not guarantee) that a language model _p_ , autoregressively, constitutes a probability measure over Σ _[∗]_ (Du et al., 2023); a model defined as in Eq. (1) may sum to _less than_ 1 in a pathological case. For any alphabet Σ, we define the set def Σ = Σ _∪{_ EOS _}_ . The conditional probability distributions are usually defined based on _vectorial_ representations of _**y** <t_ computed by a **language encoder** enc : Σ _[∗] →_ R _[D]_ (Chan et al., 2024). 

**Definition 2.3.** _A_ _**representation-based** LM is any LM that can be written as an autoregressive language model (Eq._ (1) _) where the conditional distributions over the next symbol yt ∈_ Σ _are given by_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0002-13.png)


_where_ enc : Σ _[∗] →_ R _[D] is a language encoder,_ **E** _∈_ R _[|]_ Σ _|×D is an output matrix, and_ **f** _is a projection function._[4] 

3This definition differs from Li and Vitányi’s (2008) who instead define semimeasures over _prefix strings_ . 

> 4A common choice for **f** is the softmax. Since our analyses use rational arithmetic, we instead opt for sparsemax (Martins and Astudillo, 2016). However, all of our results can be extended to the use of the more common softmax function through the use of log activations and the extended real numbers R _∪{−∞, ∞}_ (Svete et al., 2024). 

## **2.2 Regular Language Models** 

Probabilistic finite-state automata are a wellunderstood probabilistic computational model. 

**Definition 2.4.** _A_ _**probabilistic finite-state automaton** (PFSA) is a tuple_ (Σ _, Q, λ, ρ, δ_ ) _where_ Σ _is an alphabet, Q is a finite set of states, δ ⊆ Q ×_ Σ _×_ Q _≥_ 0 _×Q is a finite set of weighted transitions where_ 

_y/w we write transitions_ ( _q, y, w, q[′]_ ) _∈ δ as q −−→ q[′] , and λ, ρ_ : _Q →_ Q _≥_ 0 _are functions that assign each state its initial and final weight, respectively. Moreover, for all states q ∈ Q, δ, λ and ρ satisfy_ � _q∈Q[λ]_[ (] _[q]_[) = 1] _[, and]_ � _w_ + _ρ_ ( _q_ ) = 1 _. y/w q−−→q[′] ∈δ_ 

Next, we will define some basic concepts related to PFSAs. A PFSA _A_ = (Σ _, Q, λ, ρ, δ_ ) is **deterministic** if _|{q | λ_ ( _q_ ) _>_ 0 _}|_ = 1 and, for every _q ∈ Q, y ∈_ Σ, there is at most one _q[′] ∈ Q_ such _y/w_ that _q −−→ q[′] ∈ δ_ with _w >_ 0.[5] Any state _q_ where _λ_ ( _q_ ) _>_ 0 is called an **initial state** , and if _ρ_ ( _q_ ) _>_ 0, it is called a **final state** . A **path** _**π**_ of length _N_ is a sequence of subsequent transitions in _A_ , denoted as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0003-05.png)


def The **yield** of a path is **s** ( _**π**_ ) = _y_ 1 _· · · yN_ . The � **prefix weight** w of a path _**π**_ is the product of the transition and initial weights, whereas the **weight** of a path additionally has the final weight multiplied in. In symbols, this means 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0003-07.png)


def def with _w_ 0 = _λ_ ( _q_ 1) and _wN_ +1 = _ρ_ ( _qN_ +1). We write Π( _A_ ) for the set of all paths in _A_ and we write Π( _A,_ _**y**_ ) for the set of all paths in _A_ with yield _**y**_ . The sum of weights of all paths that yield a certain string _**y** ∈_ Σ _[∗]_ is called the **stringsum** , which we write as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0003-09.png)


The stringsum gives the probability of the string _**y**_ . This way, PFSAs induce a particularly well-understood family of LMs. 

> 5 In this paper, we do _not_ distinguish between a transition for a given symbol with weight _w_ = 0 and the absence of a transition for that symbol. That is, we assume there always _y/w_ exists a transition _q −−→ q[′] ∈ δ_ for any _q, q[′] ∈ Q_ and _y ∈_ Σ, albeit possibly with _w_ = 0. Such a choice turns out to be useful in our technical exposition. 

**Definition 2.5.** _An LM p is a_ _**regular** LM if there exists a PFSA A whose induced language model pA is weakly equivalent to p._ 

**PFSAs and non-determinism.** Although PFSAs share many properties with unweighted (booleanweighted) finite-state automata, one important difference relates to determinization. In the unweighted case, the class of deterministic and nondeterministic FSAs are equivalent, i.e., any nondeterministic FSA has an equivalent deterministic FSA that accepts the same language. This result, however, does not hold for PFSAs: There exist PFSAs that admit no deterministic equivalent (Mohri, 1997; Buchsbaum et al., 2000), meaning that non-deterministic PFSAs are strictly more expressive than deterministic ones. 

**Computing string probabilities under nondeterminism.** Autoregressive LMs (cf. Eq. (1)) and PFSAs fall under the larger framework of models that specify probability distributions _implicitly_ (Icard, 2020).[6] However, in autoregressive neural LMs with fixed precision, only one sequence of computational actions can yield a particular string, meaning that they can only model deterministic weighted regular languages (Svete and Cotterell, 2023). On the other hand, non-deterministic PFSAs compute string probabilities by _additively_ combining the probabilities of all paths that yield the same string, and hence lie outside the grasp of such neural models.[7] As we show later, CoT reasoning provides a principled way to overcome this limitation and allow fixed-precision neural models to simulate non-deterministic automata. 

## **2.3 Regular Functions** 

We now define a finite-state machine that, in addition to scanning, also _outputs_ strings. 

**Definition 2.6.** _A_ _**finite-state transducer** (FST) is a_ 6 _-tuple T_ = ( _Q,_ Σ _,_ Ξ _, I, F, δ_ ) _, where Q is a finite set of_ _**states** ,_ Σ _is an alphabet of_ _**input symbols** ,_ ∆ _is an alphabet of_ _**output symbols** , I, F ⊆ Q are sets of initial and final states, respectively, and δ ⊆ Q ×_ Σ _ε ×_ ∆ _ε × Q is a set of transitions._ 

Similar to PFSA transitions, we write FST transitions of the form ( _q, x, y, q[′]_ ) _∈ δ_ as _q −−→x_ : _y q[′]_ . 

6Other examples of such models include hidden Markov models, which are equivalent to PFSAs (Icard, 2020). 

7Note that RNN LMs with linearly bounded precision can simulate non-deterministic PFSAs in real-time by encoding a probability distribution over the PFSA’s current state in the hidden state of the RNN (Svete et al., 2024). 

Finally, we give the following definition which we will use to formalize CoT reasoning. 

**Definition 2.7.** _A_ _**regular relation** is a relation ϕ ⊆_ Σ _[∗] ×_ ∆ _[∗] that is representable by an FST. If ϕ is a (partial) function, it is called a_ _**regular function** ._ 

Like all relations, regular relations _ϕ ⊆_ Σ _[∗] ×_ ∆ _[∗]_ have a trivial inverse _ϕ[−]_[1] _⊆_ ∆ _[∗] ×_ Σ _[∗]_ . 

## **2.4 Turing Machines** 

We consider the following definition of a probabilistic Turing machine. 

**Definition 2.8.** _A_ _**probabilistic Turing machine** (PTM) is a two-tape machine with a working tape and an output tape specified by the_ 6 _-tuple M_ = ( _Q,_ Σ _,_ Γ _, δ, qι, qφ_ ) _, where Q is a finite set of states,_ Σ _is an output alphabet,_ Γ _is a tape alphabet including the blank symbol ⊔, qι, qφ ∈ Q are the initial and final states, and δ ⊆ Q ×_ Γ _×_ Σ _ε × {L, R} ×_ Q _≥_ 0 _× Q ×_ Γ _is a rationally weighted transition relation. L and R signify the PTM head moving left (L) or right (R) on the tape after a transition. We write transitions_ 

_y,d/w_ ( _q, γ, y, d, w, q[′] , γ[′]_ ) _∈ δ as_ ( _q, γ_ ) _−−−→_ ( _q[′] , γ[′]_ ) _. Moreover, we require that for any given q ∈ Q, γ ∈_ Γ _, the weights satisfy_[�] ( _q,γ_ ) _−−−→y,d/w_ ( _q[′] ,γ[′]_ ) _∈δ[w]_[= 1] _[.]_ 

A transition should be interpreted as follows: When _M_ in state _q_ , reads _γ_ on the working tape, writes _y_ on the output tape, writes _γ[′]_ on the working tape, move the head in direction _d_ on the working tape. Each computation step randomly selects a transition according to its weight _w_ .[8] 

This definition of Turing machines straightforwardly induces a semimeasure over strings (Nowak et al., 2023, Remark 2.2). It is sometimes easier to prove claims about Turing machines through another equivalent machine (Hopcroft et al., 2001). The probabilistic two-stack pushdown automaton is one example (Nowak et al., 2023). See App. C.3 for an overview. 

## **2.5 Recurrent Neural Language Models** 

Recurrent neural LMs are LMs whose conditional probabilities are given by an RNN.[9] 

> 8 For more details, see App. C.1. 

> 9Throughout this paper, we will focus on Elman RNNs (Elman, 1990) as they are the easiest to analyze and a special case of more common networks, e.g., those based on long short-term memory (LSTM; Hochreiter and Schmidhuber, 1997) and gated recurrent units (GRUs; Cho et al., 2014). 

**Definition 2.9.** _An_ _**Elman RNN** R_ = (Σ _, σ, D,_ **U** _,_ **V** _,_ **b** _,_ _**η**_ ) _is an RNN with the following hidden state recurrence:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0004-14.png)


_where_ **h** _t ∈_ Q _[D] is the state vector_[10] _at time step t,_ _**η** ∈_ Q _[D] is an initialization parameter, yt ∈_ Σ _is the input symbol at time step t,_ **r** : Σ _→_ Q _[R] is a symbol representation function,_ **U** _∈_ Q _[D][×][D] and_ **V** _∈_ Q _[D][×][R] are parameter matrices,_ **b** _∈_ Q _[D] is a bias vector, and_ _**α**_ : Q _[D] →_ Q _[D] is an element-wise, non-linear activation function._[11] 

Because **h** _t_ hides the symbols consumed by the Elman RNN, we also use the evocative notation **h** ( _**y**_ ) to denote the result of the application of Eq. (7b) over the string _**y**_ = _y_ 1 _· · · yt_ . The notation **h** ( _**y**_ ) makes it clear that an RNN LM implicitly defines a language encoder. 

**Definition 2.10.** _A representation-based LM is called an_ _**Elman LM** if its representation function is defined by the hidden state of an Elman RNN_ def enc ( _**y** <t_ ) = **h** ( _**y** <t_ ) _._ 

The most common choice for the projection function **f** is the softmax, whose limitation is that it implies the LM has full support, i.e., an Elman LM with a softmax projection assigns positive probability to all strings in Σ _[∗]_ . To construct non-fullsupport Elman LMs, we instead use the **sparsemax** (Martins and Astudillo, 2016): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0004-19.png)


In contrast to the softmax function, sparsemax is the **identity** on **∆** _[N][−]_[1] , i.e., we have sparsemax( **x** ) = **x** for **x** _∈_ **∆** _[N][−]_[1] . 

## **2.6 Neural Networks and Numerical Precision** 

All implementations of LMs on modern hardware require representations to be fixed-precision floating-point numbers or arbitrary-precision (rational) numbers. In the case of arbitrary precision, an important consideration in processing strings _**y** ∈_ Σ _[∗]_ is the number of bits required to store the representations and how the number of bits scales with the length of the string, _|_ _**y** |_ . This motivates the following definition of precision. 

10Throughout this paper all vectors are column vectors. 

11 Common examples of _**α**_ include the Heaviside function H( _x_ ) def= 1 _{x >_ 0 _}_ , the sigmoid function _σ_ ( _x_ ) def= 1+exp(1 _−x_ )[,] def and the ReLU ( _x_ ) = max (0 _, x_ ). 

**Definition 2.11.** _The_ _**precision** ψ_ ( _**y**_ ) _of a representation-based neural LM is the number of bits required to represent the entries of_ enc ( _**y**_ ) _:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0005-01.png)


_We say that a representation-based LM is of_ 

- _**constant precision** if ψ_ ( _**y**_ ) = _O_ (1) _, i.e., if ψ_ ( _**y**_ ) _≤ C for all_ _**y** ∈_ Σ _[∗] and some C ∈_ R _,_ 

- _**logarithmically bounded precision** if ψ_ ( _**y**_ ) = _O_ (log _|_ _**y** |_ ) _, i.e., if there exist T_ 0 _∈_ N _and C ∈_ R _such that for all_ _**y** ∈_ Σ _[∗] with |_ _**y** | ≥ T_ 0 _, ψ_ ( _**y**_ ) _≤ C_ log2 _|_ _**y** |,_ 

- _**linearly bounded precision** if ψ_ ( _**y**_ ) = _O_ ( _|_ _**y** |_ ) _, i.e., there exist T_ 0 _∈_ N _and C ∈_ R _such that for all_ _**y** ∈_ Σ _[∗] with |_ _**y** | ≥ T_ 0 _, ψ_ ( _**y**_ ) _≤ C|_ _**y** |, and_ 

- _**unbounded precision** if ψ_ ( _**y**_ ) _cannot be bounded by a function of |_ _**y** |._ 

The constructions in the existing literature range from constant to unbounded precision. The RNNs considered by Svete and Cotterell’s (2023) encoding of deterministic PFSAs, for example, results in an RNN that is of constant precision. Weiss et al. (2018), Merrill (2019) and Merrill et al. (2020) consider models with logarithmically bounded precision. In contrast, articles treating the Turing completeness of neural networks (Siegelmann and Sontag, 1992; Nowak et al., 2023) require unbounded precision to be able to represent the fact that a Turing machine may fail to halt. Naturally, our constructions of Turing complete LMs will also require unbounded precision. 

## **2.7 Transformer Language Models** 

Transformer LMs compute the conditional distributions _p_ ( _yt |_ _**y** <t_ ) by means of self-attention. Because transformer LMs necessitate the precise introduction of multiple sub-components, we postpone the full introduction to App. C.4 and only review the basics here. We borrow much of the notation and formalization from Svete and Cotterell (2024). 

A transformer is a composition of multiple transformer **layers** , each of which implements the **attention mechanism** . The attention mechanism works as follows. It takes a **query** vector **q** _∈_ R _[D]_ and two matrices: The matrix **K** _∈_ R _[N][×][D]_ of **keys** and the matrix **V** _∈_ R _[N][×][D]_ of **values** and computes a weighted average of the value vectors based on 

the compatibilities of the key vectors to the query vector, as scored by a scoring function _f_ . 

Attention weights are computed by normalizing the scores _f_ ( **q** _,_ **k** 1) _, . . . , f_ ( **q** _,_ **k** _t_ ). The choice of the normalization function has concrete implications on representational capacity (Hao et al., 2022; Svete and Cotterell, 2024). We focus on the **hard attention** projection function. 

**Definition 2.12.** _**Hard attention** is computed with the_ hardmax _projection function_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0005-14.png)


def _for d ∈_ [ _D_ ] _, where_ **x** _∈_ R _[D] and m_ = _|_ argmax ( **x** ) _| is the cardinality of the argmax set._ A transformer layer uses the attention mechanism followed by a position-wise MLP[12] to compute augmented representations **z** _t_ of the input representations **X** _t_ = � **x** _[⊤]_ 1[;] _[ · · ·]_[;] **[ x]** _[⊤] t_ � _∈_ R _[t][×][D]_ . The query **q** _t_ , the keys **K** _t_ , and values **V** _t_ are all transformations of the input representations **X** _t_ . Initially, **X** _t_ are computed by some static representation function of the symbols and their positions. Multiple transformer layers are stacked into a transformer, which computes the (deep) contextual representations of all symbols in the string. The contextual representation of the final symbol in the string is then used to define the representation function enc of a transformer LM. 

## **3 CoT Reasoning and Weak Equivalence** 

In this section, we argue that CoT reasoning can be seen as a way of comparing the input language of a formal automaton with the output language of an autoregressive LM. We then introduce a formalization of CoT reasoning that allows us to characterize the representational capabilities of CoT-augmented LMs in terms of their weak equivalence to wellstudied weighted automata. 

## **3.1 Equivalence and Augmented Alphabets** 

Autoregressive LMs generate strings by outputting individual symbols _y ∈_ Σ one at a time until EOS is generated. The resulting string _**y**_ is the output of the LM and the outputs generated in this manner implicitly define the distribution of the LM. Models studied in existing work on the representational capacity of neural LMs, however, 

12For more details, see App. C.4 

often do not (only) emit symbols from Σ. Rather, they also emit symbols from some _larger_ alphabet ∆ that includes symbols that encode additional information required to simulate a given formal model of computational. For example, the symbols generated by Pérez et al.’s (2021) transformer contain both the output symbols as well as (the changes to) the configuration of the Turing machine. 

It is not difficult to see how generating strings from an augmented alphabet can be seen as a form of CoT reasoning. The outputs _not_ intended to be a part of the final output can simply be seen as the result of the additional computational steps performed by the CoT-augmented LM. These outputs are later removed in post-processing and only the string with symbols from our target alphabet Σ remains. In this sense, Pérez et al.’s (2021) construction works with a form of CoT reasoning without explicitly mentioning it. This connection was made explicit in concurrent work by Merrill and Sabharwal (2024). 

Outputting additional information is not regarded as an issue when the task is to simulate (unweighted) Turing machine runs on a given _input string_ because what matters, in that case, is only whether the model accepts or rejects the input (Siegelmann and Sontag, 1992; Pérez et al., 2021, _inter alia_ ). However, when considering the LM induced by a PTM, we do care about the alphabet the distribution over strings is over. Thus, given an LM over Σ _[∗]_ , it follows that neural LMs outputting additional information cannot define the same distribution, i.e., they cannot be weakly equivalent. Nevertheless, the ability to output additional information while generating a string seems natural and useful; it is the heart of CoT reasoning. And, indeed, models that can only output symbols that are part of the final string are restricted to doing real-time computation (Weiss et al., 2018; Nowak et al., 2023; Svete et al., 2024). 

## **3.2 Regular Reducibility** 

We now formalize the notion of CoT reasoning through the following definition which allows a model to output additional information _not_ considered part of the final output: 

**Definition 3.1.** _An LM p over_ ∆ _is_ _**regularly reducible** to a LM q over_ Σ _if there exists a regular function ϕ_ : ∆ _[∗] →_ Σ _[∗] such that q ◦ ϕ is weakly equivalent to p._ 

We can use the function _ϕ_ to map the strings 

sampled from an LM, ones additionally encoding the intermediary steps of computation, into the final output, i.e., strings _**y** ∈_ Σ _[∗]_ . 

**Definition 3.2.** _We say an alphabet_ ∆ _satisfies the_ Σ _**-augmentation condition** for an alphabet_ Σ _if_ ∆ _⊆_ Σ _ε ×_ Γ _ε \ {_ ( _ε, ε_ ) _} for some alphabet_ Γ _._ 

This means if we care about strings from Σ _[∗]_ and we have an LM over the closure of a Σ- augmented alphabet ∆, then each symbol the LM outputs encodes is either an output symbol from Σ, an intermediate computation symbol from Γ, or both.[13] The computation symbols can then easily be removed by applying the per-symbol projecdef tion function _ϕ_ (( _y, γ_ )) = _y_ , lifted to a non-lengthincreasing homomorphism over strings _ϕ_ : ∆ _[∗] →_ Σ _[∗]_ to the LM’s output element-wise.[14] This function can be implemented by a simple single-state transducer (making it a regular function): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0006-10.png)


## **3.3 Properties of Regular Reducibility** 

To motivate our formalization, in this section, we show several properties of regular reducibility that will allow us to reason about the representational capacity of CoT-augmented LMs later in §3.4. Particularly, as we will see, there exists a strong connection between regular reducibility and the addition of _non-deterministic_ choices in the model. To exemplify this concretely, the next theorem shows that regular reducibility allows us to model non-deterministic PFSAs with deterministic ones, which, as discussed in §2.2, cannot be done with just deterministic PFSAs in general. 

**Theorem 3.1.** _Let A_ = (Σ _, Q, δ, λ, ρ_ ) _be a PFSA. Then, there exists a_ deterministic _PFSA A[′] over the alphabet_ Σ _× Q and with the state space_ Σ _× Q that is regularly reducible to A._ 

_Proof._ See App. E. ■ 

A similar connection exists for PPDAs (see App. E). Thm. 3.1 captures a crucial property of regular reducibility: It allows us to _simulate_ non-determinism with a deterministic device. As 

13While in practice the CoT and the final output come from the same alphabet, we use separate ones for ease of notation and analysis. This is without loss of generality; see App. D. 

> 14A non-length-increasing homomorphism is a function _h_ : ∆ _[∗] →_ Σ _[∗]_ where _h_ ( _ε_ ) = _ε_ , _h_ ( _x_ ) = _y_ for _x ∈_ ∆ and _y ∈_ Σ _ε_ , and that further satisfies _h_ ( _x_ 1 _x_ 2 _· · · xT_ ) = _h_ ( _x_ 1) _h_ ( _x_ 2) _· · · h_ ( _xT_ ). 

An LM _p_ generates strings from ∆ _[∗] ⊆_ (Σ _× Q_ ) _[∗]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0007-01.png)


Figure 2: An LM _p_ can generate strings from a stateaugmented alphabet ∆. The intermediate outputs _qi_ allow it to condition on the previous states of the computation. By _post-hoc_ removing the intermediate outputs with _ϕ_ , we obtain a CoT-augmented LM _p_ over Σ _[∗]_ that generates more human-readable outputs. 

exemplified in the cases of regular LMs, this can concretely increase the representational capacity of a model class, as in the case of Thm. 3.1. On the other hand, regular reducibility never _decreases_ the representational capacity of a model class, since _ϕ_ can always be set to the identity function. Due to the close connection between neural LMs and determinism (Svete and Cotterell, 2023), one could hope that a similar increase in capabilities could be achieved for neural LMs augmented with CoT reasoning as well. In §3, we show that this is indeed the case. 

## **3.4 Regular Reducibility and CoT Reasoning** 

After introducing the notion of regular reducibility and presenting some of its core properties, we now use it to define CoT-augmented LMs. 

**Definition 3.3.** _Given alphabets_ Σ _and_ ∆ _where_ ∆ _satisfies the_ Σ _-augmentation condition, an LM p over_ Σ _[∗] is called a_ _**CoT-augmented LM** induced by LM p over_ ∆ _[∗] if p is regularly reducible from p . That is, there exists a regular function ϕ_ : ∆ _[∗] →_ Σ _[∗] such that p_ =def _p ◦ ϕ−_ 1 _._ 

Def. 3.3 defines a CoT-augmented LM _p_ through the LM _p_ that generates strings from ∆ _[∗]_ and then applies the regular function _ϕ_ to the generated strings to obtain strings from Σ _[∗]_ . This allows _p_ to output additional information while still defining a probability distribution over strings from Σ _[∗]_ (see Fig. 2). The weight of a string _**y** ∈_ Σ _[∗]_ under _p_ is then computed as the sum over the weights of all strings in the preimage of _ϕ_ , analogously to Eq. (6): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0007-08.png)


This is the approach we take in §4.2.2. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0007-10.png)


**----- Start of picture text -----**<br>
q 0 q 1 qt− 1 qt<br>BOS y 1 · · · · · · · · · yt− 1 ? · · ·<br>p ( qt,yt|qt− 1)<br>BOS q 0 yq 11 · · · yqtt−− 22 yqtt−− 11 ? · · ·<br>**----- End of picture text -----**<br>


Figure 3: If _**y** ≤t_ uniquely determines a subpath in the PFSA, the current state _qt_ can be uniquely determined _without_ knowing the previous state (top). If, however, the substring _**y** ≤t_ can lead to multiple states, the relevant next-symbol distribution cannot be determined. Storing the _sampled_ states as part of the output fixes this by keeping track of the sampled PFSA subpath (bottom). 

## **4 CoT and Representational Capacity** 

We now connect the notions introduced in §3 to the representational capacity of neural LMs. We begin in §4.1 by showing how CoT reasoning endows autoregressive LMs with _non-determinism_ by using scratch space to keep track of the current branch of execution. We then use similar principles to emulate probabilistic Turing machines with Elman RNNs (§4.2.1) and transformers (§4.2.2). 

## **Neural LMs and formal models of computation.** 

Probabilistic models of computation are often presented as autoregressive _generators_ of strings that implicitly define probability (semi)measures (Icard, 2020). This interpretation is particularly important when addressing non-determinism, because it allows for multiple executions (for example, in the case of PFSAs, multiple different _paths_ ) generating the same string. Here, we treat neural LMs as autoregressive generators of strings and show that, by defining identical next-symbol distributions, they implicitly define the same semimeasures over strings as classical probabilistic models of computation. 

## **4.1 Neural LMs and Regular LMs** 

We first discuss the connection between CoTaugmented neural LMs and PFSAs. 

## **4.1.1 Recurrent Neural LMs** 

Minsky’s (1954) construction provided one of the first connections between a neural network and a formal computational model. It showed that RNNs can emulate (deterministic) FSAs, where the RNN accepts string by activating a particular 

neuron after reading the string. The relationship in the probabilistic case was explored by Svete and Cotterell (2023), who show the equivalence of pure Elman RNNs (without CoT reasoning) and _deterministic_ PFSAs.[15] This illustrates an important distinction between the deterministic and non-deterministic frameworks and thus a discrepancy between general PFSAs and RNN LMs. Intuitively, the discrepancy comes from the fact that there is no non-determinism in the recurrence of an RNN, which is thus unable to capture the possibly non-deterministic decisions of the PFSA. CoT reasoning endows an RNN with exactly this nondeterminism because it allows the RNN to sample and refer back to _trajectories_ rather than only symbols by annotating each of its outputs with the current state of the PFSA.[16] Upon reading the previously randomly generated state of the automaton captured in the output, it can follow the randomly sampled generating trajectories. 

The following two theorems show that RNN LMs with fixed precision and CoT reasoning are weakly equivalent to general PFSA. This requires showing the correspondence in both directions, i.e., that _(1)_ the distribution induced by any PFSA can be generated by a constant-precision RNN LM with CoT, and _(2)_ any CoT-augmented constant-precision RNN LM can be emulated by a PFSA. 

**Theorem 4.1.** _For any regular LM, there exists a weakly equivalent CoT-augmented constantprecision Elman RNN LM._ 

> _Proof intuition._ We show how, using the RNN’s recurrence and output sampling step, we can implement the transition function of any PFSA. The RNN starts by sampling an output symbol containing an initial state _q_ 0 according to the initial distribution _λ_ without emitting a language symbol. This output gets fed back into the RNN at the next time step, allowing it to _read_ the sampled state, and the next symbol–state pair is sampled according to the conditional distribution defined by the output state, as illustrated by Fig. 3. Finally, the states in the generated string are removed by a regular function, leaving only the language output. See App. F for the detailed proof. ■ 

> 15The relationship between RNN LMs and _non_ - deterministic PFSAs was explored by Svete et al. (2024), albeit with linearly bounded-precision RNNs. 

> 16Recall that non-determinism means that multiple possible transitions to different states can yield the same symbol (§2.2). 

**Theorem 4.2.** _For any constant-precision CoTaugmented Elman RNN LM, there exists a weakly equivalent PFSA._ 

■ 

_Proof._ See App. F. 

Thms. 4.1 and 4.2 establish the equivalence between CoT-augmented _constant-precision_ Elman RNN LMs and general regular LMs. This illuminates the added representational capacity awarded by CoT reasoning: Storing the current FSA state in the output string and removing it later allows the model to handle non-determinism which is not possible without the additional information. 

## **4.1.2 Transformer LMs** 

We now show an analogous claim to Thm. 4.1 for Transformer LMs with CoT. This is simply a stepping stone towards full probabilistic Turing completeness—a similar construction will then lead us to the full proof that transformer LMs with unbounded precision can simulate PTMs. 

**Theorem 4.3.** _For any regular LM, there exists a weakly equivalent CoT-augmented constantprecision transformer LM._ 

_Proof sketch._ We use the construction from Svete and Cotterell (2024) which shows how to encode _n_ -gram LMs in a transformer. Here, the alphabet of the transformer is augmented with the states of the PFSA, i.e., the symbol at each position _t_ contains not only an output symbol but also the current state of the PFSA at time _t_ . Thereby each input symbol contains all the information required to compute the next-symbol probabilities (it is effectively a unigram LM). Because the next symbol probabilities in a PFSA only depend on the current state, the transformer does not need to use attention at all and can rely solely on the output of its residual connections. See App. F for the detailed proof. ■ 

## **4.2 Neural LMs and PTMs** 

We now extend the results from the previous section from representing the simple regular LM to expressing all enumerable semimeasures over strings by emulating probabilistic Turing machines. For this, in the RNN case, we require unbounded precision and ReLU activation functions rather than fixed precision and Heaviside activations. 

## **4.2.1 Recurrent Neural LMs** 

First, note the following definition: 

**Definition 4.1.** _An autoregressive LM is called a_ _**real-time LM** if it never outputs empty symbols (ε)._ 

Nowak et al. (2023) show that RNN LMs with rationally valued activations that are not restricted to operating in real time are Turing complete and weakly equivalent to a subset of rationally weighted PTMs. Intuitively, not operating in real time gives an LM additional computation time while storing the results of its computations in the hidden states. This is a form of CoT reasoning since we could equivalently let the LM output additional symbols that do not count toward the final output.[17] The ability to _erase_ certain symbols, as done by our transducer _ϕ_ , helps make the setup of Nowak et al. (2023) realistic. Importantly, beyond giving the LM more computation time, CoT also allows the LM to model non-determinism in PTMs. 

**Theorem 4.5.** _For any PTM-induced LM, there exists a weakly equivalent unbounded-precision CoT-augmented Transformer LM._ 

_Proof intuition._ The proof follows Pérez et al. (2021) but is adapted to the probabilistic case. Again, the main idea is to augment the output alphabet with enough information about the current PTM configuration to reconstruct the probabilities of possible actions at each time step. This information and appropriate positional encodings are enough to recover the PTM’s current configuration and thus the next-action distribution, allowing us to construct a weakly equivalent transformer LM. A regular function then removes the additional information from the string. See App. F for details. ■ 

## **4.2.3 Weak Equivalence** 

**Theorem 4.4.** _For every LM induced by a nondeterministic probabilistic Turing machine, there exists a weakly equivalent CoT-augmented RNN LM with unbounded precision._ 

_Proof intuition._ The proof follows Nowak et al.’s (2023) probabilistic version of the proof by Siegelmann and Sontag (1992), but extended to account for CoT reasoning. For the ability to simulate _all_ PTMs rather than just a subset, we augment the output alphabet of the RNN with enough information about the current PTM configuration so that subsequent steps can uniquely identify the previous action taken. Concretely, the output alphabet ∆ contains information about the current state, the symbol written on the working tape of the PTM, and the head action performed. This additional information is then removed by our regular function at the end, yielding only the output of the simulated PTM generated according to its probability. A detailed proof is presented in App. F. ■ 

## **4.2.2 Transformer LMs** 

The representational capacity of transformer LMs has received a lot of attention in the last few years. Pérez et al. (2021) established that the encoder–decoder variant of the architecture is Turing complete. Since then, concurrent work has shown non-probabilistic Turing completeness of LM-oriented decoder-only variants (Merrill and Sabharwal, 2024; Feng et al., 2023). We extend the work to the _probabilistic_ case. 

17On the other hand, some CoT-augmented LMs do operate in real time but over an extended alphabet; see §4.1. 

Thms. 4.4 and 4.5 show that transformer and RNN LMs with CoT reasoning are at least as powerful as PTMs. Weak equivalence requires us to also prove the reverse of these two theorems, analogous to Thm. 4.2 for constant-precision Elman RNN LMs. 

**Theorem 4.6.** _For any rationally valued RNN LM and transformer LM with CoT reasoning, there exists a weakly equivalent PTM._ 

_Proof._ RNN LMs define enumerable semimeasures (Nowak et al., 2023, App. G), which can always be expressed by a PTM (Icard, 2020, Thm. 3). Following the same reasoning, transformer LMs define enumerable semimeasures as well; the probability of a string _**y**_ is defined as the sum probabilities of all runs of the transformer which result in the output _**y**_ , of which there are countably many. Finally, there is only a countable number of ways a regular function can transform any string, so RNN LMs and Transformer LMs with CoT still define enumerable semimeasures. ■ 

## **5 Conclusion** 

Many modern language models have been shown to perform better with CoT reasoning. Despite its empirical success, CoT reasoning has yet to be well understood formally. Recent theoretical work in this area has analyzed the Turing completeness of CoT-augmented LMs, failing to account for the inherently probabilistic nature of LMs. We fix this mismatch in the current literature by introducing a novel formalization of CoT reasoning. 

## **Limitations** 

Our constructions simulating PFSAs rely on fixedprecision arithmetic in line with the finite-memory nature of such automata. In contrast, the Turingcomplete models all rely on _unbounded_ precision with respect to the length of the string—either to be able to encode arbitrarily large stacks in the hidden state in the case of RNN LMs or to be able to encode positional information in the case of transformer LMs. This inevitably results from the unbounded number of computational steps per emitted symbol by a Turing machine and is distinctly different from the scaling with respect to the number of _computational steps_ . In the transformer case, the precision scales logarithmically with the number of steps (since we use the same positional encodings as Pérez et al. (2021)), which is standard for theoretical investigations of transformer models (Yao et al., 2021; Merrill et al., 2022; Merrill and Sabharwal, 2023). In the case of RNNs, the precision scales linearly with the computation sequence length, due to the encoding of a stack as a rational number, where each entry on the stack occupies a single digit. While in line with previous theoretical work, these assumptions are unrealistic in practice since neural LMs normally use fixed precision floating point arithmetic. 

We do not use layer normalization with transformers—this is done for simplicity—but note that layer normalization has been found to increase the representational capacity of transformers in some cases (Chiang and Cholak, 2022; Merrill and Sabharwal, 2024). 

Moreover, we only prove theoretically the equivalence between CoT-augmented LMs and formal models of computation. That is, we do not give a training algorithm to elicit the emergence of specific Turing complete or non-deterministic regular automata in neural LMs, and we make no claims about the training efficiency or even the feasibility of training neural LMs for this purpose. 

## **Ethics Statement** 

Since this work deals with purely theoretical properties of neural language models, we do not foresee any ethical issues arising from this work. 

## **References** 

- Steven Abney, David McAllester, and Fernando Pereira. 1999. Relating probabilistic grammars and automata. 

In _Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics_ , pages 542– 549, College Park, Maryland, USA. Association for Computational Linguistics. 

- Y. Bar-Hillel, M. Perles, and E. Shamir. 1961. On formal properties of simple phrase structure grammars. _Zeitschrift für Phonetik, Sprachwissenschaft und Kommunikationsforschung_ , 14:143–172. 

- Bruno Bauwens. 2013. Upper semicomputable sumtests for lower semicomputable semimeasures. _arXiv preprint arXiv:1312.1718_ . 

- Satwik Bhattamishra, Arkil Patel, and Navin Goyal. 2020. On the computational power of transformers and its implications in sequence modeling. In _Proceedings of the 24th Conference on Computational Natural Language Learning_ , pages 455–475, Online. Association for Computational Linguistics. 

- Adam L. Buchsbaum, Raffaele Giancarlo, and Jeffery R. Westbrook. 2000. On the determinization of weighted finite automata. _SIAM Journal on Computing_ , 30(5):1502–1531. 

- Robin Chan, Reda Boumasmoud, Anej Svete, Yuxin Ren, Qipeng Guo, Zhijing Jin, Shauli Ravfogel, Mrinmaya Sachan, Bernhard Schölkopf, Mennatallah ElAssady, and Ryan Cotterell. 2024. On affine homotopy between language encoders. _arXiv preprint arXiv:2406.02329_ . 

- David Chiang and Peter Cholak. 2022. Overcoming a theoretical limitation of self-attention. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 7654–7664, Dublin, Ireland. Association for Computational Linguistics. 

- Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014. On the properties of neural machine translation: Encoder–decoder approaches. In _Proceedings of SSST-8, Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation_ , pages 103–111, Doha, Qatar. Association for Computational Linguistics. 

- Stephen Chung and Hava Siegelmann. 2021. Turing completeness of bounded-precision recurrent neural networks. In _Advances in Neural Information Processing Systems_ , volume 34, pages 28431–28441. Curran Associates, Inc. 

- Li Du, Lucas Torroba Hennigen, Tiago Pimentel, Clara Meister, Jason Eisner, and Ryan Cotterell. 2023. A measure-theoretic characterization of tight language models. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 9744–9770, Toronto, Canada. Association for Computational Linguistics. 

- Jeffrey L. Elman. 1990. Finding structure in time. _Cognitive Science_ , 14(2):179–211. 

- Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. 2023. Towards revealing the mystery behind chain of thought: A theoretical perspective. _arXiv preprint arXiv:2305.15408_ . 

- Michael Hahn. 2020. Theoretical limitations of selfattention in neural sequence models. _Transactions of the Association for Computational Linguistics_ , 8:156– 171. 

- Yiding Hao, Dana Angluin, and Robert Frank. 2022. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. _Transactions of the Association for Computational Linguistics_ , 10:800–810. 

- Yiding Hao, William Merrill, Dana Angluin, Robert Frank, Noah Amsel, Andrew Benz, and Simon Mendelsohn. 2018. Context-free transductions with neural stacks. In _Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_ , pages 306–315, Brussels, Belgium. Association for Computational Linguistics. 

- John Hewitt, Michael Hahn, Surya Ganguli, Percy Liang, and Christopher D. Manning. 2020. RNNs can generate bounded hierarchical languages with optimal memory. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1978–2010, Online. Association for Computational Linguistics. 

- Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. _Neural Computation_ , 9(8):1735– 1780. 

- John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman. 2001. _Introduction to Automata Theory, Languages, and Computation_ , 3 edition. Pearson. 

- Thomas F. Icard. 2020. Calibrating generative models: The probabilistic Chomsky–Schützenberger hierarchy. _Journal of Mathematical Psychology_ , 95:102308. 

- S. C. Kleene. 1956. Representation of events in nerve nets and finite automata. In C. E. Shannon and J. McCarthy, editors, _Automata Studies. (AM-34), Volume 34_ , pages 3–42. Princeton University Press, Princeton. 

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2023. Large language models are zero-shot reasoners. _arXiv preprint arXiv:2205.11916_ . 

- Samuel A. Korsky and Robert C. Berwick. 2019. On the computational power of RNNs. _arXiv preprint arXiv:1906.06349_ . 

- Ming Li and Paul M.B. Vitányi. 2008. _An Introduction to Kolmogorov Complexity and Its Applications_ , 3 edition. Springer Publishing Company, Incorporated. 

- André F. T. Martins and Ramón F. Astudillo. 2016. From softmax to sparsemax: A sparse model of attention and multi-label classification. In _Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48_ , ICML’16, page 1614–1623. 

- Warren S. McCulloch and Walter Pitts. 1943. A logical calculus of the ideas immanent in nervous activity. _The bulletin of mathematical biophysics_ , 5(4):115– 133. 

- William Merrill. 2019. Sequential neural networks as automata. In _Proceedings of the Workshop on Deep Learning and Formal Languages: Building Bridges_ , pages 1–13, Florence. Association for Computational Linguistics. 

- William Merrill and Ashish Sabharwal. 2023. The parallelism tradeoff: Limitations of log-precision transformers. _Transactions of the Association for Computational Linguistics_ , 11:531–545. 

- William Merrill and Ashish Sabharwal. 2024. The expressive power of transformers with chain of thought. In _The Twelfth International Conference on Learning Representations_ . 

- William Merrill, Ashish Sabharwal, and Noah A. Smith. 2022. Saturated transformers are constant-depth threshold circuits. _Transactions of the Association for Computational Linguistics_ , 10:843–856. 

- William Merrill and Nikolaos Tsilivis. 2022. Extracting finite automata from RNNs using state merging. _arXiv preprint arXiv:2201.12451_ . 

- William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. 2020. A formal hierarchy of RNN architectures. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 443–459, Online. Association for Computational Linguistics. 

- Marvin Lee Minsky. 1954. _Neural Nets and the Brain Model Problem_ . Ph.D. thesis, Princeton University. 

- Mehryar Mohri. 1997. Finite-state transducers in language and speech processing. _Computational Linguistics_ , 23(2):269–311. 

- Mehryar Mohri. 2009. _Weighted Automata Algorithms_ , pages 213–254. Springer Berlin Heidelberg, Berlin, Heidelberg. 

- Mehryar Mohri, Fernando Pereira, and Michael Riley. 2008. _Speech Recognition with Weighted Finite-State Transducers_ , pages 559–584. Springer Berlin Heidelberg, Berlin, Heidelberg. 

- Franz Nowak, Anej Svete, Li Du, and Ryan Cotterell. 2023. On the representational capacity of recurrent neural language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 7011–7034, Singapore. Association for Computational Linguistics. 

- Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2021. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ . 

- C.H. Papadimitriou. 1994. _Computational Complexity_ . Theoretical computer science. Addison-Wesley. 

- Clemente Pasti, Andreas Opedal, Tiago Pimentel, Tim Vieira, Jason Eisner, and Ryan Cotterell. 2023. On the intersection of context-free and regular languages. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pages 737–749, Dubrovnik, Croatia. Association for Computational Linguistics. 

- Fernando C. N. Pereira and Michael D. Riley. 1997. Speech recognition by composition of weighted finite automata. In _Finite-State Language Processing_ . The MIT Press. 

- Jorge Pérez, Pablo Barceló, and Javier Marinkovic. 2021. Attention is turing-complete. _Journal of Machine Learning Research_ , 22(75):1–35. 

- Hava T. Siegelmann and Eduardo D. Sontag. 1992. On the computational power of neural nets. In _Proceedings of the Fifth Annual Workshop on Computational Learning Theory_ , COLT ’92, page 440–449, New York, NY, USA. Association for Computing Machinery. 

- Michael Sipser. 2013. _Introduction to the Theory of Computation_ , 3 edition. Cengage Learning. 

- Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2022. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. _arXiv preprint arXiv:2210.09261_ . 

- Anej Svete and Ryan Cotterell. 2023. Recurrent neural language models as probabilistic finite-state automata. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 8069–8086, Singapore. Association for Computational Linguistics. 

   - Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. 2023. Transformers learn in-context by gradient descent. In _Proceedings of the 40th International Conference on Machine Learning_ , ICML’23. JMLR.org. 

   - Xuezhi Wang and Denny Zhou. 2024. Chain-ofthought reasoning without prompting. _arXiv preprint arXiv:2402.10200_ . 

   - Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022a. Emergent abilities of large language models. _arXiv preprint arXiv:2206.07682_ . 

   - Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ , volume 35, pages 24824–24837. Curran Associates, Inc. 

   - Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. On the practical computational power of finite precision RNNs for language recognition. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 740–745, Melbourne, Australia. Association for Computational Linguistics. 

   - Noam Wies, Yoav Levine, and Amnon Shashua. 2023. Sub-task decomposition enables learning in sequence to sequence tasks. _arXiv preprint arXiv:2204.02892_ . 

   - Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. 2021. Self-attention networks can process bounded hierarchical languages. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 3770–3785, Online. Association for Computational Linguistics. 

- Anej Svete and Ryan Cotterell. 2024. Transformers can represent _n_ -gram language models. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics_ , Mexico City. Association for Computational Linguistics. 

- Anej Svete, Franz Nowak, Anisha Mohamed Sahabdeen, and Ryan Cotterell. 2024. Lower bounds on the expressivity of recurrent neural language models. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics_ , Mexico City. Association for Computational Linguistics. 

## **A Discussion** 

At a high level, our claims characterize neural architectures endowed with CoT reasoning in terms of well-understood _probabilistic_ models of computation, giving the framework of CoT reasoning a novel probabilistic perspective. This allows the reconciliation of theoretical results about neural LMs with the general probabilistic language modeling framework. In doing so, we put existing results about the representational capacity of LMs into perspective and show how they can be thought of as performing “CoT reasoning in disguise.” Perhaps surprisingly, the inclusion of CoT reasoning steps results in a natural inclusion of non-determinism in otherwise deterministic neural LMs, as exemplified by Thm. 4.1. Outputting the states as part of the generated string—whose generation is inherently non-deterministic— allows a neural LM to keep track of the trajectory of the current execution. This suggests that CoT reasoning might provide an interesting avenue for exploring the non-deterministic representational capacity of neural LMs. Note that this means CoT reasoning can endow neural LMs with higher expressivity since, e.g., non-deterministic PFSAs are strictly more expressive than deterministic ones. Similarly, our result that RNNs or transformer LMs with CoT reasoning can simulate PTMs rather than regular Turing machines is important as it allows _probabilistic_ computation using such neural LMs. This means one can sample from them multiple times to assess the certainty of string inclusion in a language. Furthermore, the problems LMs could solve efficiently can be described by different complexity classes such as BPP, ZPP, etc. instead of P.[18] 

## **B Related Work** 

**Turing completeness of RNNs.** Plenty of existing work has investigated the representational capacity of RNNs, both as recognizers as well as LMs (e.g., McCulloch and Pitts, 1943; Kleene, 1956; Siegelmann and Sontag, 1992; Hao et al., 2018; Korsky and Berwick, 2019; Merrill, 2019; Merrill et al., 2020; Hewitt et al., 2020; Chung and Siegelmann, 2021; Merrill et al., 2022; Merrill and Tsilivis, 2022; Svete and Cotterell, 2023; Nowak et al., 2023, _inter alia_ ). Most relevant to our work, Siegelmann and Sontag (1992); Chung and Siegelmann (2021) show how RNNs with unbounded precision and unbounded computation time can simulate Turing machines and discuss the implications. Nowak et al. (2023) extend this to the probabilistic setting, showing that RNN LMs can emulate certain PTMs, but require the LMs to be able to perform non-emitting steps. 

**Turing completeness of transformers.** Pérez et al. (2021) show the Turing completeness of hard attention encoder–decoder transformer by encoding the configuration of the Turing machine in the output of the transformer. Bhattamishra et al. (2020) provide a different perspective on Turing completeness of the architecture by showing that transformers can simulate RNNs. Turing completeness is in this sense a simple consequence of the Turing completeness of RNNs. Bhattamishra et al.’s (2020) construction, however, relies on emitting symbols from an (uncountably) infinite set (rational-valued vectors), in contradiction to the requirement of the LM working over a _finite_ alphabet. In this sense, they make use of the so-called _regression_ transformer setup, which does not lend itself well to the discrete language modeling (Von Oswald et al., 2023). In concurrent work, Merrill and Sabharwal (2024) adapt Pérez et al.’s (2021) result to the CoT setting, connecting Turing completeness of transformer decoders to CoT reasoning similar to our work. In contrast to our work, they make statements about the model’s ability to decide language membership (simulating a Turing machine that accepts or rejects an input) and not to represent probabilistic languages. In that sense, the transformer they construct is not a language model. Besides being probabilistic, our construction also enables the analysis of any autoregressive LM architecture, and we focus on RNN and transformer LMs. Feng et al. (2023) similarly show that CoT transformers can perform arithmetic expressions and dynamic programming by generating intermediate results, backing these claims up with experimental evidence, but stopping short of showing Turing completeness. Du et al. (2023) provide a related result, showing that transformer LMs with continuous transformation functions are _tight_ . This might at first glance contradict the results on Turing completeness, since the latter might require the model to run indefinitely, resulting in a non-tight model. However, 

> 18For a comparison of these complexity classes, see e.g. Papadimitriou (1994). 

the discrepancy is resolved by noting that the tightness result crucially relies on the use of the softmax normalization function over the reals R (without _±∞_ ) in Eq. (2). This, together with the encoding function of the transformer, results in a non-diminishing probability of generating EOS and thus in a tight model. Our use of the sparsemax function sidesteps this issue by allowing us to assign EOS probability 0. 

## **C Additional Preliminaries** 

In our construction, we assume that strings are prefixed with the beginning-of-string symbol BOS. This is to give a base case for recurrent definitions such as Def. 2.1. This is just for ease of notation in our proofs. Note that instead of _p_ ( _y |_ BOS), we could also equivalently write _p_ ( _y | ε_ ). 

## **C.1 Turing Machines** 

Under our definition, a probabilistic Turing machine has two tapes.[19] The first is the **processing tape** on which symbols from the tape alphabet can be read and written. The second is a write-only **output tape** for symbols of the output alphabet. In the beginning, the processing tape contains the designated _⊥_ symbol in the leftmost cell while all other cells contain the blank symbol _⊔_ . The output tape is empty at the beginning. Starting in the initial state _qι_ , at each time step _t_ , a transition is sampled out of all available transitions for the given state _q_ and the current working tape symbol _γ_ , and then applied. The sampling happens according to the transition probability _w_ (recall that the transition weights _w_ are non-negative 

_y,d/w_ and sum to 1 for each pair of state _q_ tape symbol _γ_ ). We write transitions as ( _q, γ_ ) _−−−→_ ( _q[′] , γ[′]_ ), where _q, q[′] ∈ Q, γ, γ[′] ∈_ Γ _, y ∈_ Σ _ε_ , _w ∈_ Q _≥_ 0, and _d ∈{L, R}_ , with the following interpretation. When the machine is in state _q_ and its head is reading _γ_ on the working tape, it moves to state _q[′]_ , writes _γ[′]_ to the working tape, writes _y_ to the output tape if _y ∈_ Σ or nothing if _y_ = _ε_ , and it moves the head on the working tape by one symbol in the direction _d_ , i.e., to the left ( _L_ ), or to the right ( _R_ ). As with any non-deterministic machine, the above definition naturally gives rise to the notion of a tree of possible computations (Sipser, 2013, p. 48). A **branch** _**π**_ of the computation tree of a PTM is a sequence of consecutive transitions 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0014-06.png)


We say that a branch is **accepting** if the branch reaches the final state _qφ_ .[20] The **yield** of an accepting computation branch[21] is the sequence of symbols _**y** ∈_ Σ _[∗]_ written on the output tape at that point in the computation, i.e., a concatenation of the (non- _ε_ ) symbols _y_ 1 _, . . . , yN_ , where _N_ = _|_ _**π** |_ is the number of transitions in the branch. The **weight** of an accepting branch is the product of the weights of its transitions, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0014-08.png)


We denote the branches that yield a given string _**y**_ by Π( _M,_ _**y**_ ). The sum of weights of all branches that yield a certain string _**y** ∈_ Σ _[∗]_ is the **stringsum** of that string, defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0014-10.png)


This definition gives rise to a semimeasure over strings whose sum over all possible strings is exactly the halting probability[22] of the PTM, i.e., the probability that starting from the initial state _qι_ , _M_ reaches a final state _qφ_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0014-12.png)


> 19Adding another tape does not increase the computational power of a Turing machine (Sipser, 2013, Ch. 3). 

> 20The final state has no outgoing transitions. 

> 21We only consider branches that end in a final state when discussing the yield and weight of branches. This means we only take into account finite instances of computation. 

> 22The halting probability is the probability that the execution of the PTM will end in a halting state after finitely many steps. 

## **C.2 Pushdown Automata** 

**Definition C.1.** _A_ _**probabilistic pushdown automaton** (PPDA) is a tuple_ ( _Q,_ Σ _,_ Γ _, δ,_ ( _qι,_ S) _,_ ( _qφ, ε_ )) _where Q is a finite set of states,_ Σ _is the input alphabet,_ Γ _is the stack alphabet, δ ⊆ Q ×_ Γ _×_ Σ _ε ×_ Q _≥_ 0 _× Q ×_ Γ _[∗] is a finite set of weighted transitions, and_ ( _qι,_ S) _and_ ( _qφ, ε_ ) _are called the initial and final configuration, respectively. Moreover, for all states q ∈ Q and stack symbols γ ∈_ Γ _, δ satisfies_ � _w_ = 1 _. y/w_ ( _q,γ_ ) _−−→_ ( _q[′] ,_ _**γ**_ ) _∈δ_ 

Our definition of PPDA is similar to that of Abney et al. (1999), but, unlike theirs, our machine must end its computation in the final configuration rather than just empty the stack. We write transitions _y/w_ ( _q, γ, y, w, q[′] ,_ _**γ**_ ) _∈ δ_ as ( _q, γ_ ) _−−→_ ( _q[′] ,_ _**γ**_ ) which represent a move with weight _w_ from state _q_ to _q[′]_ while scanning or outputting _y_ , popping the symbol _γ_ from the stack and pushing the sequence of symbols _**γ**_ . **Definition C.2.** _A_ _**configuration**_ ( _q,_ _**γ**_ ) _∈ Q ×_ Γ _[∗] is a pair containing the current state and the current contents of the stack._ 

A PPDA _P_ = ( _Q,_ Σ _,_ Γ _, δ,_ ( _qι,_ S) _,_ ( _qφ, ε_ )) is called **deterministic** if for every _q ∈ Q_ , _γ ∈_ Γ and _y/w y ∈_ Σ _ε_ , there is at most one transition ( _q, γ_ ) _−−→ q[′] ,_ _**γ**_ with _w >_ 0. Additionally, if there is a transition _y/w ε/w[′] q, γ −−→ q[′] ,_ _**γ**_ such that _y ∈_ Σ, then there is no transition _q, γ −−−→ q[′′] ,_ _**γ**[′]_ with _w[′] >_ 0. Intuitively, there exists at most one next move given any configuration in a deterministic PPDA. A **run** _**π**_ of a PPDA is a sequence of configurations and transitions, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0015-04.png)


where ( _qn,_ _**γ** n_ ) is the confguration reached by taking transition _τ n_ from configuration � _qn−_ 1 _,_ _**γ** n−_ 1� for any _n ∈_ [1 _, N_ ]. If ( _q_ 0 _,_ _**γ**_ 0) = ( _qι,_ S) and ( _qN ,_ _**γ** N_ ) = ( _qφ, ε_ ), we call _**π**_ **accepting** . The **yield** of an def accepting run is **s** ( _**π**_ ) = _y_ 1 _· · · yN_ , where _yn_ is the symbol scanned by _τ n_ . The **weight** of an accepting run, w ( _**π**_ ), is the product of the weight of its transitions, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0015-06.png)


where _wn_ is the weight of transition _τ n_ . We write Π ( _P,_ _**y**_ ) for the set of all accepting runs with yield _**y**_ . The sum of the weights of all accepting runs of some PPDA _P_ that yield a certain string _**y** ∈_ Σ _[∗]_ is called the **stringsum** and is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0015-08.png)


## **C.3 Two-stack Pushdown Automata** 

**Definition C.3.** _A_ _**two-stack pushdown automaton** (2PDA) is a machine specified by the_ 6 _-tuple P_ = ( _Q,_ Σ _,_ Γ _, δ, qι, qφ_ ) _, where Q is a finite set of states,_ Σ _is an alphabet of input symbols,_ Γ _is an alphabet of stack symbols, including the bottom-of-stack symbol ⊥, δ_ : _Q ×_ Γ _×_ Σ _ε ×_ Q _≥_ 0 _× Q ×_ Γ[4] _ε[is a finite set of] rationally-weighted transitions, and qι and qφ are the initial and the final state, respectively._ 

We use a definition similar to that of Nowak et al.’s (2023), which assumes without loss of generality that transitions are determined by the current state and the top symbol of the first stack only. See Nowak et al. (2023, Appendix B) for a proof. We write transitions as _q −−−−−−−−→γ,y,γ_ 1 _→γ_ 3 _/w q[′]_ , which represent a move _γ_ 2 _→γ_ 4 

with weight _w_ from state _q_ , with the top of the first stack _γ_ , to state _q[′]_ , while scanning or outputting _y_ , popping _γ_ 1 and _γ_ 2 from the first and second stack, respectively, and pushing _γ_ 3 and _γ_ 4 to the first and second stack, respectively. A 2PDA is called **probabilistic** if, for every state _q_ and top symbol _γ_ of the first stack, the weights of the transitions define a probability distribution, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0015-13.png)


|Symbol|Symbol|Symbol|Type|Meaning|||
|---|---|---|---|---|---|---|
|[_N_]|||_⊂_N|The set_{_1_, . . . , N}_for_N ∈_N.|||
|Σ_,_Σ<br>_,_||Σ|alphabet|Σis a set of symbols,Σ<br>def<br>= Σ_∪{_BOS_}_,|Σ|def<br>= Σ_∪{_EOS_}_|
|_y,y_<br>_,_|_y_||_∈_Σ|A symbol, element ofΣ_,_Σ<br>_,_or<br>Σ.|||
|**_y_**|||_∈_Σ_∗_|A string overΣ.|||
|**_y_**_i_<br>_j_<br>�_y_�|||_∈_Σ_∗_<br>_∈{_0_,_1_}|_Σ_|_|A substring of**_y_**, a string.<br>One-hot encoding of the symbol_y ∈_Σ.|||
|_D_|||_∈_N|Size of the contextual representations in the||transformer or RNN.|
|**∆**_N−_1|||_⊆_R_N_|The_N −_1-dimensional probability simplex.|||
|_f_|||R_D ×_R_D →_R|A scoring function.|||
|**f**|||R_N →_**∆**_N−_1|A normalization function.|||
|_Q_,_K_,_V_,_O_|||R_D →_R_D_|The query, key, value, and output functions.|||
|_F_|||R_D →_R_D_|The fnal transformer LM transformation function.|||
|enc|||Σ_∗→_R_D_|The string representation function.|||
|**r**Π|||Σ<br>_×_N_→_R_D_|The position-augmented representation function.|||
|_L_|||_∈_N|Number of layers.|||
|_H_|||_∈_N|Number of heads.|||
|_H_|||R_HD →_R_D_|The head combining function.|||
|(_·_;_· · ·_||;_·_)||Vertical concatenation operator of vectors||or matrices.|



Table 1: A summary of the notation used in the paper. 

As before, we define a **run** in the 2PDA as a sequence of consecutive transitions. A run is called **accepting** if it ends in a final configuration. The yield of an accepting run is the sequence of symbols _y ∈_ Σ that the 2PDA scans (or outputs) during the run. And the **weight** of an accepting run, w ( _**π**_ ), is the product of the weight of its transitions, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0016-03.png)


where _wn_ is the weight of transition _τ n_ . Finally, the **stringsum** of string _**y**_ is defined as the sum of the weights of all accepting runs that yield _**y**_ , i.e., Π ( _P,_ _**y**_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0016-05.png)


## **C.4 Transformer Language Models** 

Transformer LMs are LMs whose conditional distributions _p_ ( _yt |_ _**y** <t_ ) are computed by a _transformer_ . A transformer is a composition of multiple transformer _layers_ , each of which implements the _attention mechanism_ . We give definitions of these building blocks in what follows. Our formalization and notation closely follows Svete and Cotterell (2024). 

**Notation.** We use bold unitalicized letters such as **x** _∈_ R _[D]_ to denote real-valued vectors and italicized letters _xj ∈_ R for their entries. Capital bold letters such as **X** _∈_ R _[N][×][D]_ denote matrices. All vectors are _column_ vectors unless transposed. We define the vertical stacking operator ( _·_ ; _· · ·_ ; _·_ ), which denotes the vertical concatenation of the _D_ -dimensional _column_ vectors **x** 1 _, . . . ,_ **x** _N_ into a _ND_ -dimensional vector ( **x** 1; _· · ·_ ; **x** _N_ ) _∈_ R _[ND]_ and the concatenation of the _D_ -dimensional _row_ vectors **x** _[⊤]_ 1 _[, . . . ,]_ **[ x]** _[⊤] N_[into a] matrix **X** _∈_ R _[N][×][D]_ with _N_ rows and _D_ columns. Given the matrix **X** = � **x** _[⊤]_ 1[;] _[ · · ·]_[;] **[ x]** _[⊤] N_ �, we write **X** _n_ = � **x** _[⊤]_ 1[;] _[ · · ·]_[;] **[ x]** _[⊤] n_ � for the submatrix composed of the first _n_ rows. We call a function _f_ : R _[D] ×_ R _[D] →_ R whose purpose is to evaluate the compatibility of two vectors a **scoring function** . A **normalization function f** : R _[N] →_ **∆** _[N][−]_[1] maps vectors in R _[N]_ to _N_ probabilities. Here, **∆** _[N][−]_[1] =[def] **x** _∈_ [0 _,_ 1] _[N] |_[�] _[N] n_ =1 _[x][n]_[= 1] � � is the _N −_ 1-dimensional probability simplex. This notation is summarized in Tab. 1. 

**The Attention Mechanism.** The attention mechanism works as follows. It takes a **query** vector **q** _∈_ R _[D]_ and two matrices: The matrix **K** _∈_ R _[N][×][D]_ of **keys** and the matrix **V** _∈_ R _[N][×][D]_ of **values** and computes a weighted average of the value vectors based on the compatibilities of the key vectors to the query vector, as scored by a scoring function _f_ . A formal definition is given below. 

**Definition C.4** (Attention Mechanism) **.** _The_ _**attention mechanism**_ Att : R _[D] ×_ R _[N][×][D] ×_ R _[N][×][D] →_ R _[D] is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-02.png)


_is the vector of normalized scores between the query_ **q** _and the keys in_ **K** _, f is a scoring function and_ **f** _is a normalization function._ 

**Attention types.** Attention weights are computed by normalizing the scores _f_ ( **q** _,_ **k** 1) _, . . . , f_ ( **q** _,_ **k** _t_ ). The choice of the projection function **f** determines the type of attention and has concrete implications on representational capacity (Hao et al., 2022). We focus on the **hard attention** projection function. 

**Definition C.5.** _**Hard attention** is computed with the_ hardmax _projection function:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-06.png)


def _for d ∈_ [ _D_ ] _, where_ **x** _∈_ R _[D] and m_ = _|_ argmax ( **x** ) _| is the cardinality of the argmax set._ **Definition C.6.** _A_ _**multi-layer perceptron** (MLP) F_ : R _[D] →_ R _[D] is a function defined as the composition of elementary functions_ **f** 1 _, . . . ,_ **f** _L_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-08.png)


_where each function_ **f** _ℓ for ℓ ∈_ [ _L_ ] _is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-10.png)


_where_ **W** _ℓ ∈_ R _[D][×][D] is a square weight matrix specific to layer ℓ,_ **b** _ℓ ∈_ R _[D] is a bias vector, and_ _**β** is an element-wise non-linear activation function. The function_ **f** 1 _is called the_ _**input layer** , the function_ **f** _L is called the_ _**output layer** , and the function_ **f** _ℓ for ℓ_ = 2 _, . . . , L −_ 1 _are called_ _**hidden layers** ._[23] 

**The Transformer Architecture.** A transformer layer uses the attention mechanism to compute augmented representations **z** _t_ = Att ( **q** _t,_ **K** _t,_ **V** _t_ ) of the input representations **X** _t_ = ( **x** 1; _· · ·_ ; **x** _t_ ). The query **q** _t_ , the keys **K** _t_ , and values **V** _t_ are all transformations of the input representations **X** _t_ . **Definition C.7.** _Given query, key, value, and_ _**output** functions Q, K, V , O_ : R _[D] →_ R _[D] , a_ _**transformer layer** is a function L_ : R _[T][×][D] →_ R _[T][×][D] that computes_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-13.png)


_for t ∈_ [ _T_ ] _where_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0017-16.png)


> 23Note that we refer to MLPs by the number of hidden layers, e.g., a one-layer-MLP is an MLP with one _hidden_ layer. 

_Here, we define_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-02.png)


Note: _For simplicity, we do not include layer normalization._ 

The functions _Q, K, V_ are usually implemented as linear transformations and _O_ as an MLP. Some theoretical literature, however, also considers more general function classes, e.g., all smooth functions (Hahn, 2020). 

Without further modification, the transformations applied by the transformer layer are position-invariant, which necessitates the addition of explicit positional information. 

**Definition C.8.** _A symbol_ _**representation function** is a function_ **r** : Σ _→_ R _[D]_ **[r]** _and a_ _**positional encoding** is a function_ Π: N _→_ R _[D]_[Π] _. A_ _**position-augmented** representation function_ **r** Π : Σ _×_ N _→_ R _[D] (with D_ = _D_ **r** + _D_ Π _) is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-07.png)


**Definition C.9.** _A_ _**static encoding** R is a function R_ : Σ _[T] →_ R _[T][×][D] defined for any T ∈_ N _as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-10.png)


Multiple transformer layers are stacked into a transformer, which computes the (deep) contextual representations of all symbols in the string. 

**Definition C.10.** _For L ∈_ N _, an L-layer_ _**transformer** T is defined as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-13.png)


_where Lℓ for ℓ ∈_ [ _L_ ] _are transformer layers and R is a static encoding._ 

A transformer computes the contextual representations of the symbols _**y**_ = _y_ 1 _· · · yT_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-16.png)


If _R_ is clear from the context or arbitrary, we will omit it as an argument to _T_ and just write _T_ ( _**y**_ ). **Definition C.11.** _Given a transformer T , a final representation transformation function F_ : R _[D] →_ R _[D] , and a string_ _**y** ∈_ Σ _[∗] with |_ _**y** |_ = _T , we define the_ _**encoding function**_ enc _as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-19.png)


**Transformer Language Models.** So far, we have only defined how the transformer architecture can be used to compute the contextual representations of the symbols. To complete the definition, we define a transformer _language model_ as follows. 

**Definition C.12.** _A_ _**transformer LM** pT is the representation-based autoregressive LM with the representation function_ enc _from Eq._ (34) _. That is, pT defines the conditional probability distributions_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0018-22.png)


## **C.5 Weighted Finite-state Transducers** 

**Definition C.13.** _A (rational-)_ _**weighted finite-state transducer** (WFST) is the tuple_ (Σ _,_ Ξ _, Q, δ, λ, ρ_ ) _where_ Σ _and_ Ξ _are the input and output alphabets, respectively, Q is a finite set of states, δ ⊆ Q ×_ Σ _ε ×_ Ξ _ε ×_ Q _≥_ 0 _× Q is a finite set of weighted transitions where we write transitions_ ( _q, y, x, w, q[′]_ ) _∈ δ y_ : _x/w as q −−−→ q[′] , and λ, ρ_ : _Q →_ Q _≥_ 0 _are functions that assign each state its initial and final weight, respectively._ 

WFSTs are thus a special class of (weighted) finite-state automata that operate over two alphabets. Just like PFSAs are a generalization of unweighted FSAs, WFSAs represent a more general class of machines than unweighted FSTs, such as the ones used in the definition of regular reducibility (cf. Def. 3.1). The weighted versions will prove useful when talking about various aspects of equivalence with regular reducibility. As a special case of weighted finite-state automata, WFSTs compute weights of their inputs—in this case, weights of _pairs_ of strings (the inputs and their outputs): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0019-03.png)


Of interest are also marginal sums of the inputs, i.e., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0019-05.png)


## **C.5.1 Operations on WFSTs** 

We will use WFSTs as building blocks in the exposition of regular reducibility (cf. §3.2). In this subsection, we outline some operations on the computational models that will be particularly useful. 

**Composition.** The **composition** of two WFSTs results in a WFST that, similarly to function composition, maps the strings with the first WFST, passes them to the second one, and returns the (weight of the) output of the second transducer (Pereira and Riley, 1997). More formally, given the WFSTs _T_ 1 = (Σ _,_ Υ _, Q, δ, λ, ρ_ ) and _T_ 2 = (Υ _,_ Ξ _, Q, δ, λ, ρ_ ), their composition _T_ 2 _◦T_ 1 computes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0019-09.png)


Intuitively, Eq. (38) computes the weight of all the possible ways of mapping _**y**_ to _**x**_ by first mapping _**y**_ to some _**z** ∈_ Υ _[∗]_ and then mapping that _**z**_ into _**x**_ . The WFST ( _T_ 2 _◦T_ 1) can be computed from _T_ 1 and _T_ 2 in time _O_ (( _|Q_ 1 _|_ + _|δ_ 1 _|_ ) ( _|Q_ 2 _|_ + _|δ_ 2 _|_ )) (Mohri, 2009). 

**Projecting a WFST.** Any WFST can be **projected** onto an (input or output) weighted finite-state _automaton_ (WFSA)[24] that computes the cumulative weights of individual input or output strings (Mohri et al., 2008). Given a WFST _T_ = (Σ _,_ Ξ _, Q, δ, λ, ρ_ ), its input projection is the automaton _AI_ = (Σ _, Q, δI, λ, ρ_ ) that computes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0019-12.png)


The output projection automaton is defined analogously. While it might not be immediately obvious that Eq. (39) represents the computations of a WFSA, _AI_ can be easily constructed by ignoring the output labels on the transitions (and additively merging any transitions that become identical after the removal of the output labels). That is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0019-14.png)


24A weighted finite-state automaton is a generalization of a probabilistic finite-state automaton where the weights do not have to form probability distributions. We also extend the definition of WFSAs to allow _ε_ -transitions, i.e. with symbols from Σ _ε_ . Note that every such WFSA can be converted into a weakly equivalent WFSA without _ε_ -transitions. 

**Lifting a PFSA. Lifting** transforms a given PFSA into a trivial WFST that implements the identity function _**y** �→_ _**y**_ . Concretely, given the PFSA _A_ = (Σ _, Q, λ, ρ, δ_ ), its lifted WFST _T A_ = (Σ _,_ Σ _, Q, δT , λ, ρ_ ) defines the transitions 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0020-01.png)


It is easy to see that _T A_ computes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0020-03.png)


Lifting is useful when one wants to compose a PFSA with a WFST, as is the case when discussing regular reducibility. 

**Inverting a WFST.** The **inverted** WFST of some WFST _T_ = (Σ _,_ Ξ _, Q, δ, λ, ρ_ ) is the WSFT _T[−]_[1] = �Ξ _,_ Σ _, Q, δ[−]_[1] _, λ, ρ_ � that maps the outputs of _T_ to their inputs. _δ[−]_[1] is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0020-06.png)


i.e., it is composed of transitions from _δ_ with their input–output labels flipped. 

## **D Separation of Alphabets** 

In our analysis, we use a different alphabet for the chain of thought (Γ) than for the final output (Σ) to clarify the distinction between output that encodes automata configurations vs the output that constitutes the weighted output language of that automaton. However, we can also define a regular function for the case where Γ = Σ. For this, we choose a specific symbol in Σ, e.g., _X_ , whose first occurrence signifies the boundary between the chain of thought and the language output. Then, _ϕ_ can be defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0020-10.png)


where _**y** ∈_ Σ _[∗] \{X}_ , and _**z** ∈_ Σ _[∗]_ . This can be easily modeled by the following simple two-state transducer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0020-12.png)


**----- Start of picture text -----**<br>
X :  ε<br>0 1<br>y ∈ Σ \{X}  :  ε z ∈ Σ :  z<br>**----- End of picture text -----**<br>


Figure 4: A simple WFST _T_ implementing _ϕ_ . 

Note that in this case, the chain of thought has to be confined in its entirety as one string, with the entire final output coming afterward. This simply means that the simulated Turing machine first performs all the computations it needs on the working tape, and then writes the final string to the output tape in real time. The proof of Thm. 4.4 can easily be adapted to work in this setting by only tagging the _ε_ symbols in Γ, e.g. by symbols _a_ and _b_ and defining _ϕ_ to remove them until the occurrence of another symbol, e.g. _c_ . 

## **E Proofs: Regular Reducibility** 

## **E.1 Nondeterminism in PFSAs** 

**Theorem 3.1.** _Let A_ = (Σ _, Q, δ, λ, ρ_ ) _be a PFSA. Then, there exists a_ deterministic _PFSA A[′] over the alphabet_ Σ _× Q and with the state space_ Σ _× Q that is regularly reducible to A._ 

_Proof._ Given the PFSA _A_ = (Σ _, Q, λ, ρ, δ_ ), we want to show the existence of a _deterministic_ PFSA _A[′]_ over the alphabet Σ _× Q_ that is regularly reducible to _A_ . We construct the deterministic PFSA _A[′]_ = �Σ _× Q,_ Σ _× Q, δ[′] , λ[′] , ρ[′]_[�] as follows. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-01.png)


We first prove that _A[′]_ is probabilistic and deterministic. 

- **Probabilistic** : We compute 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-06.png)


and, for any ( _y, q_ ) _∈_ Σ _× Q_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-08.png)


- **Deterministic** : Let ( _y, q_ ) _∈_ Σ _× Q_ be a state of _A[′]_ and ( _y[′] , q[′]_ ) a symbol in its alphabet. By definition of _δ[′]_ , ( _y[′] , q[′]_ ) uniquely determines the target state, which is identical to the symbol—( _y[′] , q[′]_ ). 

We now show that _A[′]_ is indeed regularly reducible to _A_ . The alphabet Σ _× Q_ clearly satisfies the def Σ-augmentation condition. We define the regular function _ϕ_ ( _y, q_ ) = _y_ , an instance of the general function described in §3.2. We will show that _A[′]_ is weakly equivalent to _A ◦ ϕ_ , or, equivalently, that _A[′] ◦ ϕ[−]_[1] is weakly equivalent to _A_ . 

Since _A[′]_ is an _acceptor_ of strings in (Σ _× Q_ ) _[∗]_ , we first transform it into weighted finite-state _transducer T A_ implementing the mapping[25] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-12.png)


def ( _y,q_ ):( _y,q_ ) _/w_ ( _y,q_ ) _/w_ for _**x** ,_ _**x**[′] ∈_ ∆ _[∗]_ . This is a simple WFST with transitions _δT A′_ = _q −−−−−−−−→ q[′] | q −−−−→ q[′] ∈ δ[′]_ . � � _T A′_ can then be composed with the weighed version of the FST implementing _ϕ[−]_[1] , _T ϕ−_ 1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0021-14.png)


25See App. C.5 for a definition of weighted finite-state transducers and the operations on them. 

_T A′ ◦T ϕ−_ 1 then computes, for _**y** ∈_ Σ _[∗] ,_ _**x**[′] ∈_ ∆ _[∗]_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0022-01.png)


We then turn _T A′ ◦T ϕ−_ 1 into a PFSA in the standard way by projecting the WFST onto the transition input labels (Mohri et al., 2008). This results in a PFSA that assigns the string _**y** ∈_ Σ _[∗]_ the probability that equals the sum of all possible mappings of _**y**_ to any _**x**[′] ∈_ ∆ _[∗]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0022-03.png)


This shows that _A[′] ◦ ϕ[−]_[1] is weakly equivalent to _A_ , which finishes the proof. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0022-05.png)


## **E.2 Nondeterminism in PPDAs** 

**Theorem E.1.** _Let P_ = ( _Q,_ Σ _,_ Γ _, δ,_ ( _qι,_ S) _,_ ( _qφ, ε_ )) _be a PPDA. Then, there exists a_ deterministic _PPDA P[′] over the alphabet_ (Σ _ε_ ) _× Q ×_ Γ _with the state space_ (Σ _ε_ ) _× Q ×_ Γ _that is regularly reducible to P._ 

_Proof._ Given a PPDA _P_ = ( _Q,_ Σ _,_ Γ _, δ,_ ( _qι,_ S) _,_ ( _qφ, ε_ )), we prove the existence of a _deterministic_ PPDA _P[′]_ over the alphabet Σ _ε × Q ×_ Γ that is regularly reducible to _P_ . 

We construct the deterministic PPDA _P[′]_ = �Σ _ε × Q ×_ Γ _,_ Σ _ε × Q ×_ Γ _,_ Γ _, δ[′] ,_ ( _qι,_ S) _,_ ( _qφ, ε_ )� with the set of transitions 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0022-10.png)


- **Probabilistic:** For any ( _y, q, γ_ ) _∈_ Σ _ε × Q ×_ Γ and _γ[′] ∈_ Γ, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0022-12.png)


- **Deterministic:** Just like in the finite-state case, let ( _y, q, γ_ ) _∈_ Σ _ε × Q ×_ Γ be a state, _γ[′] ∈_ Γ a stack symbol and ( _y[′] , q[′] , γ[′]_ ) _∈_ Σ _ε × Q ×_ Γ an input symbol. Then ( _y[′] , q[′] , γ[′]_ ) uniquely determines the next configuration. 

We now show that _P[′]_ is regularly reducible to _P_ . More precisely, we define the regular function _ϕ_ ( _y, q, γ_ ) =def _y_ and show that _P ′ ◦ ϕ−_ 1 is weakly equivalent to _P_ . 

Let _G[′]_ be a CFG[26] that is equivalent to _P[′]_ (see Hopcroft et al., 2001, Section 6.3.2 for the construction). Using grammars, rather than pushdown automata, will simplify the proof as there are well-known constructions (e.g., Bar-Hillel et al. (1961), Pasti et al. (2023)) for composing CFGs and FSTs. We therefore prove that _G[′] ◦ ϕ[−]_[1] is weakly equivalent to _P_ . 

Recall that _ϕ[−]_[1] can be implemented as an FST _T ϕ−_ 1 defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0023-03.png)


Adapting Corollary 1 from Pasti et al. (2023) to the FST case, we get 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0023-05.png)


Then, just like in the regular case, we compute � _G[′] ◦T ϕ−_ 1� ( _**y**_ ) by summing over all possible _**x** ∈_ Σ _[∗]_ values: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0023-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0023-08.png)


This shows that _G[′] ◦T ϕ−_ 1 is weakly equivalent to _P_ , thus _P[′] ◦T ϕ−_ 1 is weakly equivalent to _P_ . This concludes the proof. ■ 

## **F Proofs: Representational Capacity of Neural LMs** 

## **F.1 Finite-state Language Models** 

**Theorem 4.1.** _For any regular LM, there exists a weakly equivalent CoT-augmented constant-precision Elman RNN LM._ 

_Proof._ Let _A_ = (Σ _, Q, λ, ρ, δ_ ) be a PFSA. We divide the definition of a weakly equivalent Elman LM _R_ = (Σ _× Q, D,_ **U** _,_ **V** _,_ **b** _,_ _**η**_ ) with the output matrix **E** into multiple steps.[27] 

**Note:** The paper assumes a BOS-padding of the input strings. Since CoT-augmented LMs work over a potentially larger alphabet, the padding symbol has to be changed accordingly. Particularly, the CoTaugmented RNN will work over the alphabet Σ _× Q_ . For simplicity, assume that any input string is padded by the symbol (BOS _, q_ 0) for some arbitrary (but fixed) state _q_ 0 _∈ Q_ . 

> 26 Similar to the notation used for PPDAs, we use _G_ ( _**y**_ ) to denote the weight of the string _**y**_ under the grammar _G_ . 

> 27Notice that _R_ is CoT-augmented by construction, as it works over the alphabet ∆ def= Σ _× Q_ . ∆ also satisfies the Σ-augmentation condition. 

**Hidden states.** Let _D_ = _|_ Σ ~~_ε_~~ _[||][Q][|]_[.][We will represent the input symbols with the one-hot representation] function � _·, ·_ � that computes � _y, q_ � _∈{_ 0 _,_ 1 _}[|]_[Σ] ~~_ε_~~ _[||][Q][|]_ :[28] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0024-01.png)


def while other entries of the vector are zero.[29] We can define _**η**_ = **0** _D_ . 

**Recurrence.** Conveniently, the CoT-augmented RNN will store the current PFSA state in its output “symbol”. Therefore, its next hidden state **h** _t_ will not, in fact, depend on the previous one ( **h** _t−_ 1); **h** _t−_ 1 will only be used to compute the next-symbol–state output distribution through Eq. (2). The RNN therefore only has to appropriately incorporate the input information. With this in mind, we define the following Elman RNN parameters **U** _,_ **V** , and **b** : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0024-04.png)


where **O** _D_ is a _D × D_ -dimensional matrix of zeros, **I** _D_ is the _D_ -dimensional identity matrix, and **0** _D_ is a _D_ -dimensional vector of zeros. Then, we define the output matrix **E** _∈_ R _[|]_[Σ] ~~_ε_~~ _[||][Q][|×][D]_ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0024-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0024-07.png)


**Computation of string probabilities.** We now show that _R_ with **E** defines the same conditional distributions as _A_ , meaning that it implicitly defines the same probability distribution over strings. As mentioned above, we assume that the input string is prefixed with the (BOS _, q_ 0) symbol. Let the previously generated output symbol _∈_ ∆ be � _y, q_ � (in the first step, the “generated” pair is (BOS _, q_ 0)). We get that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0024-09.png)


The one-hot encoding � _y, q_ � is then used to “index” the appropriate column of the output matrix **E** as **Eh** _t_ , which contains the probabilities of the _next_ symbol–state pairs defined by the PFSA, as per Eq. (59a): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0024-11.png)


Passing **Eh** _t_ through the normalization function, the next state and symbol are sampled and passed as input to the RNN, repeating the update step. Since the Elman RNN following these dynamics by constructions generates paths (sequences of states in the output symbols) with the same probabilities as the input automaton, it enumerates all accepting paths of any string _**y** ∈_ Σ _[∗]_ with the same probabilities as the original PFSA. Applying the transformation _ϕ_ to the produced outputs and summing over sequences that yield the same string will thus result in strings sampled from the PFSA. This finishes the proof. ■ 

**Theorem 4.2.** _For any constant-precision CoT-augmented Elman RNN LM, there exists a weakly equivalent PFSA._ 

> 28Throughout the paper, we index vectors and matrices directly with set elements rather than integer values, meaning each index corresponds to exactly one element from the given set. 

> 29If any of the two arguments are empty, its corresponding component is also zero. 

_Proof._ Let _p_ be a CoT Heaviside Elman RNN LM over the alphabet Σ with the regular function _ϕ_ : ∆ _[∗] →_ Σ _[∗]_ for an Σ-augmented alphabet ∆. Let _p_ be the underlying RNN LM over ∆ _[∗]_ , that is, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-01.png)


by Def. 3.3. We want to show the existence of a possibly non-deterministic PFSA _A_ that is weakly equivalent to _p_ . We write _p_ = _[∼] p[′]_ to mean _p_ is weakly equivalent to _p[′]_ (cf. Def. 2.2). 

By Svete and Cotterell (2023, Lem. 4.1), there exists a DPFSA _A_ = (∆ _, Q, δ, λ, ρ_ ) over the augmented alphabet ∆ weakly equivalent to _p_ . This means that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-04.png)


and thus 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-06.png)


Just like in the proof of Thm. 3.1, _A ◦ ϕ[−]_[1] can be computed as a composition of (lifted) WFSTs that is then projected onto the input component, resulting in a PFSA weakly equivalent to _p_ . This finishes the proof. ■ 

**Theorem 4.3.** _For any regular LM, there exists a weakly equivalent CoT-augmented constant-precision transformer LM._ 

Before we proceed to the full proof of Thm. 4.3, we first show the following simple but useful lemma. **Lemma F.1.** _Define the following linear transformation functions:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-11.png)


_for all_ **x** _∈_ R _[D] . A transformer layer L with the parameters V and O and residual connections implements the identity function irrespective of the parameters Q, K, and f :_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-13.png)


_for all_ **x** _∈_ R _[D] ._ 

_Proof._ By definition of a transformer layer (cf. Eqs. (27), (28a) and (28b)), _L_ computes 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-17.png)


This allows us to show Thm. 4.3. 

_Proof._ Let _A_ = (Σ _, Q, λ, ρ, δ_ ) be a PFSA. Like the CoT-augmented RNN in Thm. 4.2, the CoTdef augmented transformer LM will work over the alphabet ∆ = Σ ~~_ε_~~ _[×][ Q]_[.][It][will][start][by][generating][a] random initial state according to the initial-state distribution _λ_ upon reading the designated padding symbol (BOS _, q_ 0). Then, at step _t_ of generation, it will generate the next symbol–state pair by sampling from the next-transition distribution _p_ � _qt, yt[|][ q][t][−]_[1] � given the state stored in the previously generated output tuple. 

> [def] More formally, define BOS _[′]_ = (BOS _, q_ 0) for some arbitrary but fixed _q_ 0 _∈ Q_ . Further, define the static representation function 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0025-21.png)


as in Eq. (57). Let _L_ be the identity transformer layer from Lemma F.1 and _T_ the transformer with the single layer _L_ . Then, by Lemma F.1, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0026-01.png)


def for any _t_ . Defining the transformation _F_ ( **x** ) = **x** , it therefore holds that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0026-03.png)


In words, enc ( _**y** <t_ ) contains the current symbol–state pair of the PFSA. The CoT-augmented transformer LM can therefore sample the next symbol–state pair from the conditional distribution defined by the PFSA by setting the values of the output matrix **E** as in Thm. 4.1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0026-05.png)


Clearly, the identical conditional probabilities defined by the CoT-augmented transformer _T_ result in strings over Σ ~~_ε_~~ _[×][ Q]_[sampled][with][probabilities][equal][to][the][path][probabilities][from][the][PFSA,][as][in] Thm. 4.1. Applying the transformation _ϕ_ to the produced outputs will thus result in strings sampled from the PFSA, giving us a CoT-augmented transformer LM weakly equivalent to _A_ . Lastly, since all the representations in the model are position-invariant, the constructed transformer is of constant precision. ■ 

## **F.2 Probabilistic Turing Machine Language Models** 

**RNN LMs with CoT reasoning.** Next, we show that, for every PTM, its induced LM can be encoded in an unbounded precision CoT RNN LM. 

**Theorem 4.4.** _For every LM induced by a non-deterministic probabilistic Turing machine, there exists a weakly equivalent CoT-augmented RNN LM with unbounded precision._ 

_Proof._ We rely mainly on existing results by Nowak et al. (2023) but use the additional information afforded by the augmented output alphabet, resulting in a simpler construction and interpretation. By Nowak et al. (2023, Prop. 3.1 and Thm. 3.1), the LM families induced by PTMs and probabilistic 2PDA are weakly equivalent, so it suffices to show that a CoT RNN LM can encode any given 2PDA. 

We first reiterate their main result together with its condition. 

**Definition F.2.** _A 2PDA P is called_ Σ _**-deterministic** if, for any current state q any top symbol on its first stack γ and any output symbol from_ Σ _ε, there is at most one transition with non-zero weight._ 

**Theorem F.3.** _Nowak et al. (2023, Thm 3.2) Every_ Σ _-deterministic probabilistic 2PDA can be encoded in an RNN LM that can output empty tokens ε._ 

Nowak et al. (2023, Thm. 3.2) shows that an RNN LM with output alphabet Σ _ε_ can simulate any probabilistic 2PDA over Σ if it is Σ-deterministic. Let _P_ be an arbitrary probabilistic 2PDA, with _P_ = ( _Q,_ Σ _,_ Γ _, δ, qι, qφ_ ). Now define another 2PDA _P_ = ( _Q,_ ∆ _,_ Γ _, δ[′] , qι, qφ_ ), which generates outputs over the extended alphabet ∆ =def _Q ×_ Γ4 _ε[×]_[ Σ] _[ε]_[.][Define the transitions, add the following transitions to] _[ δ][′]_ in _P[′]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0026-15.png)


The constructed _P[′]_ satisfies the required Σ-determinism condition, meaning that we can construct a weakly equivalent RNN LM with _ε_ outputs. Finally, define the following regular function _ϕ_ : ∆ _ε →_ Σ _ε_ that removes all additional information post-hoc: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0026-17.png)


The RNN LM therefore directly simulates runs from the 2PDA _P[′]_ . Applying _ϕ_ to the outputs of the RNN LM results in strings in Σ _[∗]_ . Since the transformation groups all execution runs that output _**y** ∈_ Σ _[∗]_ and the resulting CoT-augmented LM _p_ (cf. Def. 3.3) sums over them, we are left with a weakly equivalent LM. ■ 

**Theorem 4.5.** _For any PTM-induced LM, there exists a weakly equivalent unbounded-precision CoTaugmented Transformer LM._ 

> _Proof._ Here, we adapt the construction by Pérez et al. (2021) to the decoder-only case with prompt length (denoted by _n_ in Pérez et al. (2021)) zero since we only care about _generating_ strings from the CoT-augmented transformer LM. Moreover, rather than implementing a deterministic transition function of a Turing machine (which is simulated by a particular layer of their transformer), we implement the probabilistic transition function of a rational-valued PTM in the _sampling_ step of the transformer LM. As in our previous constructions, this step endows the model with non-determinism. 

At a high level, we will construct a CoT-augmented transformer LM over Σ _[∗]_ that will output symbols def def from the augmented alphabet ∆ = _Q ×_ Γ _×_ Σ ~~_ε_~~ _[×][ A][ ×][ A]_[.][Here,] _[A]_ = _{−_ 1 _,_ 0 _,_ 1 _}_ , where we will for conciseness identify the action LEFT with _−_ 1 and the action RIGHT with 1. Note that we, like Pérez et al. (2021), _do not_ allow the action NOOP in our construction, but we include the value 0 in the action set as it will be useful for the starting conditions of the constructed transformer. As explained later, such an alphabet ∆ contains enough information for the transformer LM to be able to reconstruct the configuration of the PTM at every time step, and thus match its conditional probability distributions. 

**Notation.** Let _M_ = ( _Q,_ Σ _,_ Γ _, δM, qι, qφ_ ) be a QPTM and _p_ the LM over Σ _[∗]_ it induces. In the following, we denote with _qt ∈ Q_ the state of the PTM, with _vt ∈_ Γ the symbol written to the working tape, with _at_ the action performed, with _st_ the symbol read, and with _yt_ the symbol written to the output tape, all at time step _t_ . 

**High-level idea of the construction.** Before formally describing the components of the CoT-augmented transformer LM in separate lemmata, we give a high-level overview of the construction. The twolayer (single-head) transformer will, when computing enc _**y** <t_ for _**y** <t[∈]_[∆] _[∗]_[, perform the following] � � computations, very similar to those described by Pérez et al. (2021): 

1. **Input representations** ( _R_ ; Lemma F.5): The input representation function represents the input symbols _y ∈_ ∆ with a multi-hot encoding (one that contains an individual one-hot encoding of each of the components _qt, vt−_ 1 _, yt−_ 1 _, at−_ 1 _, at−_ 2) of the form[30][,][31] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0027-07.png)


for _t ∈_ N _≥_ 0. The representations also include additional components (the “empty” zero values in the vector, explained below) used for processing and the positional encoding of the time step, 1 _, t_ + 1 _, t_ +11 _[,]_ ( _t_ +1)1[2][.][This component is explained in more detail in Lemma][ F.5][.] 

2. **Layer 1** ( _L_ 1; Lemma F.6): The first layer uses the information about the actions performed at all previous time steps (contained in the input static representations of the CoT-augmented symbols) to 

> 30In the following, green color denotes the components that are added or computed by each of the described components. 

> 31Notice that the symbol _y ∈_ Σ is not part of the internal representation, since it is not required for the simulation of the PTM. It is only used to construct the final output string stored in the output tape. 

compute the locations of _M_ ’s head at each time step. This results in the internal representations of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0028-01.png)


for _t ∈_ N _≥_ 0, where _c_ ( _t_ ) denotes the position of _M_ ’s head at time _t_ . This component is identical to the _second_ layer of the transformer from Pérez et al. (2021) and is explained in more detail in Lemma F.6. 

3. **Layer 2** ( _L_ 2; Lemma F.7): Uses the _L_ 1-computed information about the head locations at each time step to (almost) compute _st_ —the symbol read by the head of the PTM at time step _t_ . This results in the internal representations of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0028-04.png)


for _t ∈_ N _≥_ 0, where _ℓ_ ( _t_ ) denotes the last time step when _M_ ’s head wrote to _c_ ( _t_ ). This component is analogous to the _third_ layer of the transformer from Pérez et al. (2021) and is explained in more detail in Lemma F.7. 

4. **Output function** ( _F_ ; Lemma F.8): The function _F_ uses the information computed by the two layers of the transformer to compute the one-hot encoding of the current configuration of the PTM. In particular, this includes _M_ ’s current state ( _qt_ ), the symbol read by _M_ ’s head ( _st_ ), and, for reasons that we explain shortly, the action performed by _M_ at the previous time step ( _at−_ 1). This results in the representation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0028-07.png)


which is used for sampling the next transition of the PTM. This component resembles the output function of Pérez et al. (2021) and is explained in more detail in Lemma F.8. 

5. **Sampling** (Lemma F.9): The representation � _qt, st, at−_ 1� is used to index the output matrix **E** that contains the conditional probabilities _p_ ( _· | qt, st_ ). This can be used to sample the next augmented symbol � _q[t]_[+1] _, vt, yt, at, at−_ 1� (here, the last component, _at−_ 1, equals the “input” to the sampling step). This component is explained in more detail in Lemma F.9. 

The idea of the construction—iteratively computing and storing the modifications to the PTM configuration in the generated string _**y**_ —is therefore identical to Pérez et al.’s (2021) one. In particular, the two components that perform the bulk of the simulation—Layer 1 and Layer 2—are identical to the components from Pérez et al. (2021). We describe and show the correctness of the components in the lemmata in the rest of the section. The correctness of the construction follows from the correctness of the components. 

**Weak equivalence.** To define a CoT-augmented LM with such a transformer, we can define a transducer that projects the outputs in � _Q ×_ Γ _×_ Σ _ε[×][ A][ ×][ A]_ � _∗_ onto Σ _∗_ in the standard way by retaining only the outputs in Σ ~~_ε_~~[.][This collapses all the executions of the transformer LM (which simulates the executions of] the PTM with the same probabilities, as per Lemma F.9) yielding the same string in Σ _[∗]_ , resulting in a CoT-augmented transformer LM weakly equivalent to _p_ . 

**Precision.** The constructed transformer computes and stores position-dependent values at several points during the computation. The precision required for the representation of these values grows logarithmically with the number of computational steps. However, since the number of computational steps performed by a PTM generating a string is potentially unbounded in the length of the string (Hopcroft et al., 2001, p. 339), the transformer’s precision is subsequently unbounded as well (Nowak et al., 2023). ■ 

We begin with a general lemma about the disjunction of one-hot encodings. **Lemma F.4.** _Let S_ 1 _, . . . , S n be finite sets and let_ � _s_ 1 _, . . . , sn_ � _∈{_ 0 _,_ 1 _}[|S]_[1] _[|···|S][n][|] denote the one-hot encoding of the tuple_ ( _s_ 1 _, . . . , sn_ ) _∈S_ 1 _× · · · × S n. Define the matrices_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-02.png)


_element-wise as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-04.png)


_Then, it holds that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-06.png)


_Proof._ Eq. (78) can equivalently be written as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-08.png)


Indexing the elements of **W** _[S][i]_ � _s_ 1 _, . . . , sn_ � directly with _s ∈S i_ , we then compute 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-10.png)


which is the definition of the elements of � _si_ �. The equality in Eq. (81d) follows from the fact that the summand is non-zero exactly when _s[′] j_[=] _[ s][j]_[for all] _[ j]_[=] _[ i]_[.] ■ 

**Lemma F.5** (Input representations) **.** _Define the following static representation function of the CoT-_ def _augmented symbols y_ = � _q, v, y, a[′] , a_ � _∈_ ∆ _:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-13.png)


_where_ � _q, v, y, a[′] , a_ � _∈{_ 0 _,_ 1 _}[|][Q][||]_[Γ] _[||]_ Σ ~~_ε_~~ _[||][A][||][A][|] denotes the one-hot encoding of the tuple_ � _q, v, y, a[′] , a_ � _,_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0029-15.png)


_for matrices_ **W** _[Q] ,_ **W**[Γ] _,_ **W** _[A] from Lemma F.4,_ **A** _[′] ,_ **A** _∈_ R[1] _[×][D] defined as_[32] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-01.png)


_and D_ = [def _|Q|_ + _|_ Γ _|_ + _|A|_ + 2] + 2 + [1 + _|_ Γ _|_ ] + 4 _. Then, it holds that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-03.png)


_Proof._ The first part (computing the one-hot encodings of _∼δ_ , _v_ , and _a[′]_ ) follows from the construction of the matrices **W** _[Q] ,_ **W**[Γ] , and **W** _[A]_ from Lemma F.4. The second part (computing the values _a[′]_ and _a_ with the matrices **A** _[′]_ and **A** ) follows from the definition of the matrices **A** and **A** _[′]_ : By construction, the values of _a[′]_ and _a_ will be copied from the one-hot encodings into the resulting entry of the vector. ■ 

We now describe the two layers of the transformer that compute the quantities required to be able to determine the current configuration of the PTM at each time step. Here, we heavily rely on the construction by Pérez et al. (2021). 

Let _M_ = ( _Q,_ Σ _,_ Γ _, δM, qι, qφ_ ) be a rationally-weighted PTM. Let _t ∈_ N and define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-07.png)


as the position of the head of the PTM at time _t_ . Further, define the set 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-09.png)


_L_ ( _t_ ) contains the time steps at which the PTM _M_ so far visited (and thus wrote to) the tape cell read by _M_ at time _t_ . Then, define _ℓ_ ( _t_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-11.png)


In words, _ℓ_ ( _t_ ) denotes the time step at which the PTM _M_ last visited (and thus wrote on) the tape cell read by _M_ at time _t_ if this cell was visited yet. Otherwise, _ℓ_ ( _t_ ) equals _t_ . 

Now, define the BOS symbol over the augmented alphabet Γ as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-14.png)


Let _qι, q_ 1 _, . . . , qT_ , _⊥, v_ 1 _, . . . , vT_ , BOS _, y_ 1 _, . . . , yT_ , and _a_ 0 _, a_ 1 _, . . . , aT_ be the sequences of states visited, symbols written on the processing tape, symbols written on the output tape, and actions performed by _M_ def in the first _T_ steps. Define _a−_ 1 = _a_ 0 = 0, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0030-16.png)


32 Recall that we can identify the actions of the PTM with the integers _−_ 1, 0, and 1. 

and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-01.png)


These will be the inputs to the transformer layers and thus,[33] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-03.png)


With this, we can describe and prove the correctness of the first layer of the transformer we are building. 

**Lemma F.6** (Layer 1) **.** _There exists a transformer layer L_ 1 _that, given the inputs_ **X**[0] _, computes the values tc_ +1( _t_ ) _[and][c]_[(] _t[t]_ +1 _[−]_[1][)] _for all t_ = 1 _, . . . , T . More precisely, denoting_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-06.png)


_it holds that_ **x**[1] _t[contains the entries containing the values] t[c]_ +1[(] _[t]_[)] _[and][c]_[(] _t[t]_ +1 _[−]_[1][)] _[:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-08.png)


> _Proof._ This follows from Pérez et al. (2021, Lemma 9), which we summarize here.[34] Concretely, _L_ 1 is implemented by a transformer layer with trivial (zero-valued) query and key transformations and a value transformation _V_ that copies the values of the actions _at−_ 1 and _at−_ 2 to the entry that will hold the values _tc_ +1( _t_ )[and] _[c]_[(] _t[t]_ +1 _[−]_[1][)][.][Since the current head location] _[ c]_[ (] _[t]_[)][ is simply the sum of those values (cf.][Eq. (][86][)),] attending to all previous positions results in Eq. (94). Formally, we define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-11.png)


Here, _n[′]_ and _n_ refer to the indices of the rows at which the values _t[c]_ +1[(] _[t]_[)][and] _[c]_[(] _t[t]_ +1 _[−]_[1][)] will be stored (that is, the rows below the values of _at−_ 1 and _at−_ 2). All other rows of **V**[1] are zero. Then, for _t ∈_ N _≥_ 0, it holds that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0031-13.png)


> 33Note that since the transformer over the augmented alphabet is deterministic, this representation is unique. 

> 34 More precisely, since their construction stores the actions _at_ and _at−_ 1 (the former is possible because the action is not sampled but deterministically computed based on the configuration of the PTM), their layer computes the values _t[c]_ +1[(] _[t]_[)][and] _t[c]_ +1[(] _[t]_[)][.] As we show later in Lemma F.7, this does not affect the correctness of the construction. 

1 for all _j ≤ t_ , resulting in **s** _t_ = hardmax ( **0** ) = _t_ +1 **[1]** _[t]_[, and] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0032-01.png)


This results in 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0032-03.png)


Furthermore, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0033-01.png)


which is what we needed to show. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0033-03.png)


**Lemma F.7** (Layer 2) **.** _There exists a transformer layer L_ 2 _that, given the outputs_ **X**[1] _of L_ 1 _from Lemma F.6, computes the values ℓ_ ( _t_ ) _and_ � _vℓ_ ( _t_ )� _for all t_ = 1 _, . . . , T . More precisely, denoting_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0033-05.png)


_it holds that_ **x**[2] _t[contains the entries containing the values][ ℓ]_[(] _[t]_[) + 1] _[ and]_[ �] _[v] ℓ_ ( _t_ )[�] _[:]_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0033-07.png)


_Proof._ This follows from Pérez et al. (2021, Lemma 10). The idea of the construction is for the selfattention mechanism at time _t_ to attend to exactly the entry from time step _ℓ_ ( _t_ ) + 1 (i.e., to compute the query and key vectors such that argmax ( **s** _t_ ) = _ℓ_ ( _t_ ) + 1) since that entry will contain the information about the symbol written at the time step before—at _ℓ_ ( _t_ ). Then, the values _ℓ_ ( _t_ ) and _vℓ_ ( _t_ ) are obtained by copying the corresponding values of the positional encoding and the written symbol from that time step. 

Formally, we define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0034-01.png)


Pérez et al. (2021, Lemma 10) show that, given the parameters above, the output of the scoring function _f_ is maximized at the entry _ℓ_ ( _t_ ) _∈{_ 0 _, . . . , t}_ .[35] Since **V**[2] copies the second value from the positional encoding of the symbol at time step _ℓ_ ( _t_ ), which is _ℓ_ ( _t_ )+1, this entry appears in the output of the attention mechanism. Furthermore, **V**[2] also copies the value � _vℓ_ ( _t_ )� from the same entry. This, together with the zero-valued function _O_ and residual connections, results in Eq. (102). ■ 

**Lemma F.8** (Correctness of the output function) **.** _There exists an MLP F that, given the outputs_ **X**[2] _of L_ 2 _, computes the one-hot encoding of the current configuration of the PTM. More concretely, it holds that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0034-04.png)


_Proof._ Here, we define a function _F_ similar to that of Pérez et al. (2021, Lemma 11), but with an additional layer that handles the addition of the _⊥_ symbol not handled by Pérez et al. (2021). The logic nonetheless remains the same: _F_ receives the output of _L_ 2 of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0034-06.png)


and _(1)_ copies the value of _qt_ , _(2)_ copies the value of _at−_ 1 _(3)_ compares the value of _ℓ_ ( _t_ ) to _t_ to determine whether _st_ = _⊔_ or _st_ = _vℓ_ ( _t_ ), _(4)_ compares the value of _t_ to 0 to determine whether _st_ = _⊥_ or _st_ = _vℓ_ ( _t_ ). 

> 35More precisely, their construction results in the maximum being at _ℓ_ ( _t_ + 1), since layer 1 in their construction, in contrast to ours, computes the values _[c]_[(] _t[t]_ +1[+][1][)] and _t[c]_ +1[(] _[t]_[)][, shifting all computations by one step.][This is the consequence of the aforementioned] difference between their deterministic and our probabilistic framework. 

Concretely, _F_ will take the form of a three-layer MLP 

_F_ ( **x** ) = ReLU(def **W** 5ReLU( **W** 4ReLU( **W** 3ReLU( **W** 2ReLU( **W** 1 **x** + **b** 1) + **b** 2) + **b** 3) + **b** 4) + **b** 5) (106) where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0035-02.png)


are the weights of the MLP and **b**[1] _, . . . ,_ **b**[5] are the corresponding biases. For conciseness, we will denote 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0035-04.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0035-06.png)


Then, we define 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0035-08.png)


(110) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0036-00.png)


**----- Start of picture text -----**<br>
1: C 1 C 1: C 2 C 2: C 3 C 3: C 4 C 4: C 5 C 5+1 C 5+2<br>� �� � � �� � � �� � � �� � � �� � � �� � � �� �<br>I |Q| 0 |Q|<br> I | Γ |   0 | Γ | <br>W [2] = I |A| I | Γ |  (112) b [2] = 00 ||A Γ ||  (113)<br> I | Γ |   0 | Γ | <br> 1 − 1  0<br> 1   0 <br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0036-01.png)


**----- Start of picture text -----**<br>
1: C 1 C 1: C 2 C 2: C 3 C 3: C 4 C 4: C 5 C 5+1 C 5+2<br>� �� � � �� � � �� � � �� � � �� � � �� � � �� �<br>I |Q| 0 |Q|<br>W [3] = I | Γ | I |A| − 1 | Γ | − 1 | Γ |  b [3] = 00 ||A Γ ||  (115)<br>I | Γ | 1 | Γ | − 1 | Γ |<br> I | Γ | 1 | Γ |   − 1 | Γ | <br>(114)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0036-02.png)


Lastly, we set **W**[5] and **b**[5] such that the the function **x** _�→_ ReLU� **W**[5] **x** + **b**[5][�] computes the one-hot encoding of the three input one-hot encodings � _qt_ � _,_ � _st_ � _,_ � _at−_ 1� by implementing the logic AND operations, as in Svete and Cotterell (2024, Lemma B.1). 

By construction, we get that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0037-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0037-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0038-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0038-01.png)


Since the three logical expressions in Eq. (121b) are complementary, it holds that the second component of **y**[4] holds either the value of _vt−_ 1, _⊔_ , or _⊥_ depending on the value of _ℓ_ ( _t_ ) and _t_ . This is exactly the symbol that will be read by the PTM at time step _t_ , i.e., _st_ : If _ℓ_ ( _t_ ) _< t_ (which means that cell _c_ ( _t_ ) has been visited before), then _st_ = _vt−_ 1; if _ℓ_ ( _t_ ) = _t_ and _t >_ 0 (which means that cell _c_ ( _t_ ) has not been visited before, and _t >_ 0), then _st_ = _⊔_ ; and if _t_ = 0 (meaning that the PTM just started executing and is still reading the initial symbol _⊥_ ), then _st_ = _⊥_ . By the construction of **W**[5] and **b**[5] , we get that **y**[5] = � _qt, st, at−_ 1�. ■ **Lemma F.9** (Correctness of the sampling step) **.** _Define the output matrix_ **E** _∈_ R _[|][Q][||]_[Γ] _[||]_ Σ _ε||A||A|×|Q||_ Γ _||A| as_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0038-03.png)


_for q, q[′] ∈ Q, v ∈_ Γ _, y ∈_ Σ _ε, a ∈ A, a[′] ∈{−_ 1 _,_ 1 _}, and s ∈_ Γ _. Then, it holds that_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0038-05.png)


_Proof._ Follows directly from Lemma F.8 and the construction of **E** :By Lemma F.8, we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0038-07.png)


By construction of **E** , we have that 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0039-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning_images/acl_2024_On_the_Representational_Capacity_of_Neural_Language_Models_with_Chain-of-Thought_Reasoning.pdf-0039-02.png)


