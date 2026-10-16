# **A Logic for Expressing Log-Precision Transformers** 

**William Merrill Ashish Sabharwal** New York University Allen Institute for AI `willm@nyu.edu ashishs@allenai.org` 

## **Abstract** 

One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed that finite-precision transformer classifiers can be equivalently expressed in a generalization of first-order logic. However, finite-precision transformers are a weak transformer variant because, as we show, a single head can only attend to a constant number of tokens and, in particular, cannot represent uniform attention. Since attending broadly is a core capability for transformers, we ask whether a minimally more expressive model that can attend universally can also be characterized in logic. To this end, we analyze transformers whose forward pass is computed in log _n_ precision on contexts of length _n_ . We prove any log-precision transformer classifier can be equivalently expressed as a first-order logic sentence that, in addition to standard universal and existential quantifiers, may also contain majority-vote quantifiers. This is the tightest known upper bound and first logical characterization of log-precision transformers. 

Any log-precision transformer can be re-expressed as a sentence in FO(M) logic, e.g.: 

M _i._ `a` ( _i_ ) _∧_ M _j._ `b` ( _j_ ) _∧¬∃k, ℓ._ ( `a` ( _k_ ) _∧_ `b` ( _ℓ_ ) _∧ ℓ< k_ ) _(m_ _`a` ’s followed by m_ _`b` ’s, i.e.,_ _`a`[m]_ _`b`[m] )_ 

## `aaaabbbb` ✓ `aaabbbbb` ✗ `baaaabbb` ✗ 

Figure 1: A first-order logic with majority (FO(M)) sentence for `a` _[m]_ `b` _[m]_ . In addition to standard _∀_ and _∃_ quantifiers over string indices, FO(M) allows _majority_ quantifiers (M) that take a majority-vote across indices. `a` ( _i_ ) indicates whether token _i_ is `a` (and analogously for `b` ). We prove FO(M) can express any function computed by a log-precision transformer. 

## **1 Introduction** 

The incredible success of deep learning models, especially very large language and vision transformers with hundreds of billions of parameters (Brown et al., 2020; Thoppilan et al., 2022), has come at the cost of increasingly limited understanding of how these models actually work and when they might fail. This raises many concerns, such as around their safe deployment, fairness, and accountability. Does the inner working of a transformer defy description in a simpler symbolic system that we can better understand? Or _can transformer computation be described using a familiar symbolic formalism?_ Understanding how to view the reasoning process of a transformer in terms of logic could potentially expand our ability to formally reason about their behavior over large domains of inputs. 

Chiang et al. (2023) provide a partial answer to this question, showing that any _finite-precision_ transformer classifier can be expressed as a sentence in a variant of first-order logic with counting 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 

quantifiers and modular arithmetic over input position indices. Specifically, counting quantifiers take the form _∃_[=] _[x] i_ : _ϕ_ ( _i_ ) where _x_ is a count variable and _i_ is a position index. They show that there exists a single sentence in this logic that computes the output of the transformer for any input string of any length. This is a powerful result because it shows that a simple logical formalism is fully sufficient to describe all the complexity of a massive finite-precision transformer. It also provides an upper bound on finite-precision transformers: any function that cannot be defined in first-order counting logic with modular indexing cannot be expressed by the transformer. 

However, Chiang et al.’s result is not fully general because it relies on the transformer precision being fixed with respect to the transformer’s context length. More generally, as we will demonstrate in Section 3, finite-precision transformers are a fundamentally weak variant of transformers: crucially, cannot express uniform attention patterns, which are a core algorithmic primitive of transformers (Weiss et al., 2018). In fact, we show that they can only attend to a constant number of input positions, which may be seen as a rather limited generalization of hard attention.[1] For example, Chiang et al. show that their logic for finite-precision transformers cannot recognize `a` _[m]_ `b` _[m]_ , whereas in practice, transformers can (Bhattamishra et al., 2020).[2] This motivates studying a formal model of transformers where precision grows with context length (which we formalize as _log-precision_ ), making it possible to capture uniform attention as well as other broad attention patterns. This is useful both for recognizing `a` _[m]_ `b` _[m]_ and more generally for reasoning globally over the input. 

We demonstrate that _log-precision_ transformer classifiers can also be expressed as sentences in a simple logic: _first-order logic with majority_ , or FO(M), over inputs strings (Barrington et al., 1990). In addition to standard existential and universal quantifiers, FO(M) has _majority_ quantifiers that return true iff at least half the propositions they quantify are true. It also allows comparing input positions (e.g., _ℓ< k_ in Figure 1) and accessing their individual bits. Our main result is as follows: 

**Theorem 1** (Informal version of Theorem 2) **.** _For any log-precision transformer T , there exists an_ FO(M) _sentence ϕ that computes the same function as T , i.e., ϕ_ ( _x_ ) = _T_ ( _x_ ) _for any input string x._ 

**Upper bound.** Theorem 2 shows transformers with more than finite precision can also be expressed in a simple extension of first-order logic, going beyond Chiang et al. (2023)’s result. On the other hand, FO(M) is a strict superset of Chiang et al.’s counting logic; it can simulate counting quantifiers (see Section 2.2) and allows non-modular position comparisons. Thus, handling a more general class of transformers powerful enough to express uniform attention slightly weakens the bound. 

Still, our result constitutes (to our knowledge) the tightest upper bound on log-precision transformers and the first defined in terms of logic, building on a line of complexity-theoretic work analyzing the power of transformers (Hahn, 2020; Merrill et al., 2022; Liu et al., 2023; Merrill & Sabharwal, 2023). In particular, FO(M) strengthens the upper bound of log-space-uniform TC[0] by Merrill & Sabharwal (2023). The refined bound adds to the limitations of transformers identified by Merrill & Sabharwal (2023): for example, it establishes unconditionally that log-precision transformers cannot compute boolean matrix permanents, and shows that, in a certain formal sense, integer division and matching parentheses are among the formally hardest problems that transformers can solve (see Section 4).[3] 

**Mechanistic interpretability.** Beyond providing an upper bound on the reasoning problems solvable by transformers, we believe Theorem 1 could guide the design of “transformer-complete” programming languages similar in spirit to RASP (Weiss et al., 2018). RASP is a declarative programming language designed to capture transformer computation, and Lindner et al. (2023) implement a compiler from RASP _into_ transformers. Unlike RASP, FO(M) can provably express any transformer (Theorem 1), which we believe justifies using it (or an equivalent but more user-friendly variant) as a target language for programs extracted _from_ transformers. 

Similar to a decision tree, an FO(M) sentence has the interpretable property that each sub-sentence corresponds to a constraint on input (see Figure 1). In contrast, the internal modules of a transformer or circuit do not satisfy this since they map between arbitrary latent spaces. We speculate this property 

> 1Hard attention is provably substantially weaker than general attention (Hao et al., 2022; Merrill et al., 2022). 

> 2Technically, the empirical results of Bhattamishra et al. (2020) are for `a` _m_ `b` _m_ `c` _m_ , a harder variant of `a` _m_ `b` _m_ . 

> 3To be clear, Theorem 1 is one-sided: every transformer can be expressed as an FO(M) sentence, but not necessarily the other way. Moreover, we believe that many FO(M) sentences _cannot_ be expressed by transformers. An exact logical characterization of transformers remains an open problem. 

2 

could facilitate interpreting models by translating them to FO(M), though a careful exploration of the algorithmic and HCI aspects of this idea lies outside the current paper’s theoretical scope. 

**Contributions.** Our results shed new light on how to view the computation inside transformers in terms of logic. Specifically, our main contributions are to prove the following: 

1. Fixed-precision transformers can only attend to a fixed number of tokens, and those with precision less than log log _n_ cannot uniformly attend over length- _n_ contexts (Proposition 1). 

2. Log-precision transformer classifiers can be expressed as sentences in FO(M) (Theorem 2). 

## **2 Preliminaries: Transformers and** FO(M) 

Let Σ be a finite alphabet. We denote by _[∗]_ the Kleene star operator, i.e., for a set _X_ , _X[∗]_ =[�] _[∞] n_ =0 _[X][n]_[.] We will view transformers and FO(M) sentences both as functions from Σ _[∗] →{_ 0 _,_ 1 _}_ , and show that any function a transformer computes can also be computed by an FO(M) sentence. 

## **2.1 Transformers** 

We view the transformer precision _p_ as a function of the context length _n_ , writing _p_ ( _n_ ) where appropriate. Let D _p_ be the datatype of _p_ -precision floats, i.e., tuples _⟨m, e⟩_ where _m, e_ are signed integers together taking _p_ bits. Using _|x|_ to mean the size of integer _x_ , a float represents the value _m ·_ 2 _[e][−|][m][|]_[+1] .[4] Following Appendix A of Merrill & Sabharwal (2023), we define _p_ -truncated addition (+ _,_[�] ), multiplication ( _·_ ), and division ( _/_ ) over D _p_ . We now define a _transformer encoder binary classifier_ over D _p_ , largely adopting Merrill & Sabharwal’s notation.[5] 

**Definition 1.** A _p_ -precision transformer _T_ with _h_ heads, _d_ layers, model dimension _m_ (divisible by _h_ ), and feedforward width _w_ is specified by: 

      - _[[m]] p_[whose form is defined in Appendix C.1;][6] 

   1. An embedding function _ϕ_ : Σ _×_ N _→_ D _[[m]]_ 

   2. For each 1 _≤ ℓ ≤ d_ and 1 _≤ k ≤ h_ , a head similarity function _s[ℓ] k_[:][ D] _[m] p[×]_[ D] _[m] p[→]_[D] _[p]_[whose] form is defined in Appendix C.2; 

   3. For each 1 _≤ ℓ ≤ d_ and 1 _≤ k ≤ h_ , a head value function _vk[ℓ]_[:][ D] _[m] p[→]_[D] _[m/h] p_ whose form is defined in Appendix C.2; 

   4. For each 1 _≤ ℓ ≤ d_ , an activation function _f[ℓ]_ : (D _[m/h] p_ ) _[h] ×_ D _[m] p[→]_[D] _[m] p_[whose][form][is] defined in Appendix C.3 and implicitly uses the feedforward dimension _w_ ; 

   5. An output classifier head _κ_ : D _[m] p[→{]_[0] _[,]_[ 1] _[}]_[ whose form is defined in Appendix C.4.] 

- **Definition 2.** We define the transformer computation and output as a function of an input _x ∈_ Σ _[n]_ . 

   1. Embeddings: For 1 _≤ i ≤ n_ , **h**[0] _i_[=] _[ ϕ]_[(] _[x][i][, i]_[)][.][6] 

   2. Self Attention: For 0 _≤ ℓ ≤ d −_ 1, (multihead) self-attention block _ℓ_ + 1 computes _h_ attention heads: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0003-17.png)


3. Activation Block: For 0 _≤ ℓ ≤ d −_ 1, activation block _ℓ_ + 1 aggregates the head outputs to produce **h** _[ℓ]_[+1] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0003-19.png)


4. Classifer Head: The network prediction on _x ∈_ Σ _[n]_ is _κ_ ( **h** _[d] n_[)][.] 

> 4 _⟨_ 101 _,_ 010 _⟩_ represents 1 _._ 012 _×_ 2102 . This is closer to the IEEE standard than the _m ·_ 2 _e_ semantics used in Merrill & Sabharwal (2023), letting us define the minimum representable float more realistically in Proposition 1. 5Increasing the classifier’s output space arity (e.g., a transformer that predicts the next token) or switching to causal attention of a decoder-only model would not change our results. However, our proof no longer goes through if the decoder can generate tokens that get added to the input at the next step (cf. Pérez et al., 2019). 

> 6 _ϕ_ , like _p_ , is actually a function of the context length _n_ , and Appendix C.1 enforces that _ϕ_ is computable in O(log _n_ ) time, as standard choices of positional embeddings would satisfy. 

3 

We say _T_ ( _x_ ) = _κ_ ( **h** _[d] |x|_[)][and] _[L][T]_[is][the][language][of] _[x][∈]_[Σ] _[∗]_[such][that] _[T]_[ (] _[x]_[)][=][1][.][We][refer][to] _ϕ, s[ℓ] k[, v] h[ℓ][, f][ ℓ]_[, and] _[ κ]_[ as the] **[ core functions]**[ in] _[ T]_[ , and to embeddings, self attention, activation, and the] classifier head as the **components** of _T_ . We write _θT_ for the concatenated vector of parameters for the functions _ϕ, s[ℓ] k[, v] h[ℓ][, f][ ℓ]_[, and] _[ κ]_[, for all][ 1] _[ ≤][ℓ][≤][d]_[ and][ 1] _[ ≤][k][≤][h]_[.] 

We define a **log-precision transformer** as one where _p_ is at most O(log _n_ ) and is a “simple” function, i.e., computable in O(log _n_ ) time. In our model, the weights _θT_ defining _T_ are fixed, but the precision _p_ used to compute the forward pass can depend on _n_ (see Footnote 13 for a generalization). 

## **2.2 First-Order Logic with Majority** 

As we will show, transformers can be translated into sentences in FO(M). But what do such sentences look like? Informally, FO(M) is first-order logic extended to also have majority (M) quantifiers. Following Barrington et al. (1990), our sense of FO(M) takes _strings_ in Σ _[∗]_ as input and returns 0 or 1 to define a formal language. In this setting, quantifiers range over _indices_ (positions) into the string. Predicates can be applied to the variables introduced by these quantifiers. 

**Definition 3** (FO(M) index) **.** Indices in FO(M) are integers denoting positions in the input string: 

1. The constant 1, representing the first token’s position. 

2. The constant _n_ , representing the last token’s position. 

3. Strings (e.g., _i, j, k_ ) representing variables ranging over positions 1 to _n_ . 

4. Any index built by applying addition or subtraction to other indices.[7] 

**Definition 4** (FO(M) formula) **.** Formulas in FO(M) are constructed as follows:[8] 

1. Let Σ be a finite alphabet. For each _σ ∈_ Σ and any index _i_ , _σ_ ( _i_ ), e.g., `a` ( _i_ ), is a formula that is true if the _i_ -th input token is _σ_ .[9] 

2. For any indices _i, j_ , the formula bit( _i, j_ ) returns the _j_ -th bit of the binary expansion of _i_ .[10] 

3. For two indices _i, j_ , _i_ = _j_ , _i ≤ j_ , and _i ≥ j_ are formulas with their conventional semantics. 

4. For two formulas _ϕ, ψ_ , _ϕ ∧ ψ_ and _ϕ ∨ ψ_ are formulas with their conventional semantics. 

5. For any formula _ϕ_ (which may refer to _i_ ), the following are valid formulas: 

   - (a) _∃i. ϕ_ means some value of _i_ in [1 _, n_ ] makes _ϕ_ true. 

   - (b) _∀i. ϕ_ means all values of _i_ in [1 _, n_ ] make _ϕ_ true. 

   - (c) M _i. ϕ_ means _≥ n/_ 2 values of _i_ in [1 _, n_ ] make _ϕ_ true. 

We use parentheses where necessary to disambiguate the order of operations. General formulas may contain free (i.e., unbound) variables: e.g., _∀i. i_ = _j_ . A _sentence_ is an FO(M) formula _ϕ_ with no free variables. Sentences represent functions from from Σ _[∗]_ to _{_ 0 _,_ 1 _}_ and thus define a formal language.[11] 

**Extensions.** Beyond Definition 4, FO(M) can express _counting_ and _threshold_ quantifiers in terms of majority quantifiers (Barrington et al., 1990). Given a formula _ϕ_ , a counting quantifier creates a new formula _∃[k] i_ : _ϕ_ that is true iff _ϕ_ is true across exactly _k_ values of _i_ . Threshold quantifiers _∃[≤][k]_ and _∃[≥][k]_ work similarly but check if _ϕ_ is true for at least or at most _k_ values of _i_ . In addition, we show in Appendix A that FO(M) can express _conditional majority_ quantifiers, which create a formula M _i_ : _ϕ_ [ _ψ_ ] that is true iff _ψ_ is true for at least half the values of _i_ that make _ϕ_ true. 

## **2.2.1 Examples** 

To illustrate the formalism, we provide example languages definable in FO(M) with Σ = _{_ `a` _,_ `b` _}_ . First, we show two languages that do not require majority quantifiers to express: 

**Example 1** (Bigram matching) **.** Strings containing the bigram `ab` : _∃i_ [ `a` ( _i_ ) _∧_ `b` ( _i_ + 1)] _._ **Example 2** (Skip-bigram matching) **.** Strings containing the long-distance pattern `a` _. . ._ `b` (cf. “induction heads” of Elhage et al. 2021): _∃i_ [ `b` ( _i_ ) _∧∃j_ [ _j ≤ i ∧_ `a` ( _j_ )]] _._ 

> 7Barrington et al. (1990) did not introduce this as a primitive, but it can be simulated using the _≤_ predicate. 

> 8We write parentheses to indicate the order of operations. 

> 9Barrington et al. (1990) define _Qb_ ( _i_ ) for _b ∈{_ `0` _,_ `1` _}_ . We generalize this to an arbitrary vocabulary Σ by assuming each token is one-hot-encoded: _σ_ ( _i_ ) = _Q_ `1` ( _|_ Σ _|i_ + _s_ ) where _s_ is the index of _σ_ in the vocabulary. 10This predicate is included in the logic for technical reasons; see Barrington et al. (1990). 

> 11One can also take multiple sub-sentences within _ϕ_ to be labeled as ordered outputs, thus allowing _ϕ_ to be a function from Σ _[∗]_ to _{_ 0 _,_ 1 _}[k]_ for some fixed constant _k_ . 

4 

In contrast, Example 3 is a simple example that requires majority quantifiers (Furst et al., 1984): **Example 3** (Majority) **.** Strings with more `b` ’s than `a` ’s: M _i_ [ `b` ( _i_ )] _._ 

Figure 1 showed how FO(M) can be used to recognize patterns like `a` _[m]_ `b` _[m]_ . A similar idea can be used to model parentheses matching (Barrington et al., 1990): 

**Example 4** (1-Dyck) **.** The well-balanced parentheses language (with `a` opening and `b` closing): 

_∀i._ ( _∃a, b._ (( _∃[a] j_ : `a` ( _j_ ) _∧ j ≤ i_ ) _∧_ ( _∃[b] j_ : `b` ( _j_ ) _∧ j ≤ i_ ) _∧ b ≤ a_ )) _∧_ M _i._ `a` ( _i_ ) _∧_ M _j._ `b` ( _j_ ) _._ 

**Example 5** (Integer Arithmetic) **.** Iterated addition (i.e., summing _n n_ -bit numbers), iterated multiplication, and division (Hesse, 2001) can all be expressed in FO(M). 

## **3 Finite Precision Transformers Cannot Attend Universally** 

Attention heads that spread attention weight uniformly across inputs have been observed in transformer LMs (Merrill et al., 2021) and make soft attention fundamentally more powerful than hard attention (Hao et al., 2022; Merrill et al., 2022). In particular, uniform attention is an important primitive that transformers can use to solve tasks involving counting (Bhattamishra et al., 2020; Chiang et al., 2023), taking majority votes (Merrill et al., 2022), and matching parentheses or sorting (Weiss et al., 2021). A transformer with sufficient precision can easily implement uniform attention by setting the keys and queries across all positions to be constant. However, attention heads with finite precision cannot represent uniform attention over long sequences as a consequence of the following: 

**Proposition 1.** _Let_ **a** _∈_ R _[n] s.t._[�] _[n] i_ =1 _[a][i]_[= 1] _[ and]_[ ˜] **[a]** _[ its nearest][ p][-precision float approximation.]_ 

_1. Then the number of nonzero entries of_ ˜ **a** _is upper bounded by its precision: specifically,_ ˜ **a** _has at most_ 2[2] _[p] nonzero entries._ 

_2. Moreover, if p <_ log log _n and_ **a** _is uniform (i.e., ai_ = 1 _/n), then_ ˜ **a** = _[⃗]_ 0 _._ 

_Proof._ The smallest positive value representable by a _p_ -precision float is 2 _[−]_[(] _[p][m][−]_[2+2] _[pe][−]_[1][)] which is bounded below by 2 _[−]_[2] _[p]_[+1] . Letting _k_ = 2[2] _[p]_ , it holds that 2 _[−]_[2] _[p]_[+1] = 2 _/k_ . So if ˜ _ai_ gets the minimum value, then _ai ≥_ 1 _/k_ . Since[�] _i[a][i]_[= 1][, there can be at most] _[ k]_[ indices satisfying this property.][This] implies there can be at most _k_ nonzero entries in ˜ **a** . If _n > k_ and **a** is uniform, 1 _/n_ is less than half of the minimum representable value of 2 _/k_ . Thus, ˜ **a** = _[⃗]_ 0. 

Proposition 1 says that fixed-precision transformers are artificially limited because they can only attend over bounded-length windows, making them similar to hard-attention transformers (Hao et al., 2022). Morever, they cannot compute uniform attention over contexts of length _n_ with less than log log _n_ precision. This explains why Chiang et al. (2023) prove finite-precision transformers provably cannot recognize `a` _[m]_ `b` _[m]_ , while in practice transformers have been shown to learn even its harder variant `a` _[m]_ `b` _[m]_ `c` _[m]_ even with long context lengths (Bhattamishra et al., 2020). In essence, their upper bound only applies in the asymptotic regime when _n >_ 2[2] _[p]_ . 

In contrast, transformers in practice have enough precision both to compute uniform attention and recognize `a` _[m]_ `b` _[m]_ on practical context lengths. More concretely, the bfloat16 representation allows uniform attention over 2[6+2][7] _≈_ 10[42] tokens and normal float16[12] allows 2[10+2][4] _≈_ 10[8] tokens, both well above the typical context window of transformers. This motivates a formal model of transformers with enough precision to compute uniform attention and recognize languages such as `a` _[m]_ `b` _[m]_ . 

## **4 Main Result: Expressing Log-Precision Transformers in** FO(M) 

By Proposition 1, precision must grow with the context length _n_ ( _p >_ log log _n_ ) for a transformer to compute uniform attention and other attention patterns with unbounded range, like practical transformers. In this paper, we analyze any transformer with up to O(log _n_ ) precision. We show that any function computable by log-precision transformers can be expressed in FO(M): 

> 12We account for the division of _p_ into _pm_ and _pe_ rather than treating them together. Our minimum value differs slightly from numpy but is on the same order of magnitude. Moving to float8 lowers the length upper bound for uniform attention to 2[3+2][3] _≈_ 2048, which suggests float8 LMs will have limited length generalization. 

5 

**Theorem 2.** _Let T be a log-precision transformer with a parameter vector θT fixed for all context lengths n._[13] _Then, there exists an_ FO(M) _sentence ϕ that computes the same function as T , i.e., ϕ_ ( _x_ ) = _T_ ( _x_ ) _for any input string x._ 

Theorem 2 is the tightest known upper bound for log-precision transformers and shows that it is still possible to characterize transformers in a simple variant of first-order logic even with log-precision and uniform attention. As alluded to earlier, Theorem 2 immediately implies that any problem complete for FO(M) (or a larger class) is also transformer-hard. Since integer division and Dyck language membership are known to be FO(M)-complete (Hesse, 2001; Aaronson et al., 2022), it follows, perhaps surprisingly, that the entire computation of any transformer on input _x_ can be reduced to a single integer division or a finite number of Dyck-language queries: 

**Corollary 2.1.** _Let T be a transformer satisfying Theorem 2. For any input x, there exist first-order definable integers a, b, and i (dependent on T and x) such that T_ ( _x_ ) _equals the i-th bit of ⌊a/b⌋. For any x, there also exist first-order definable strings w_ 1 _, . . . , wm such that T_ ( _x_ ) _is first-order definable in terms of the membership of the wi’s in k-Dyck._ 

## **5 Preliminaries for Proving Theorem 2** 

## **5.1 Computation Graphs** 

A _computation graph G_ over a datatype D _⊆{_ 0 _,_ 1 _}[∗]_ and a countable set of primitive functions F _⊆_ D _[∗] ×_ D is a directed acyclic graph where: 

1. Each node is labelled by a _node type_ : a function _f ∈_ F computed by this node. 

2. Each edge represents a value D flowing as output from one node into another node. We consider the edges flowing into node _j_ to have an order, i.e., be numbered. 

3. F contains the special symbol input, which designates _k_ nodes as input nodes. We refer to _k_ as the _arity_ and assume w.l.o.g. that nodes 0 _, . . . , k −_ 1 are inputs.[14] 

4. A single node is taken as the output node (w.l.o.g., the node with the largest index). 

A computation graph _G_ of arity _k_ parameterizes a function D _[k] →_ D in the standard way: the input nodes are assigned the input values, and the value of each node is computed (traversing the graph in a bottom-up topological order) as a function of the values of its children until the output node receives a value. The value of the output node is considered the output of the function. It is worth noting that computation graphs can only process inputs of bounded length. To process arbitrary-length inputs, we will need to generalize them to computation graph families (Section 5.2). 

For a computation graph _G_ , size( _G_ ) is the number of nodes, depth( _G_ ) is the length of the longest path from an input node to the output, and arity( _G, i_ ) is the number of inputs to node _i_ . 

**Threshold circuits.** A threshold circuit is a special case of a computation graph where D = _{_ 0 _,_ 1 _}_ and _F_ is the set of threshold functions of the form _θ≤_ ∆ and _θ≥_ ∆ over D _[∗]_ , defined as follows: _θ≤_ ∆( _x_ ) = 1 if[�] _σ∈x[σ][≤]_[∆][and][ 0][ otherwise;] _[ θ][≥]_[∆][(] _[x]_[)][ is defined analogously.][Typical AND, OR,] and NOT gates are a special case of threshold gates, as is an IDENTITY gate.[15] 

We allow nodes with the _k[′] ≥_ 1 largest indices to all be designated as (ordered) output nodes. A threshold circuit with arity _k_ and _k[′]_ output nodes will thus be a function from _{_ 0 _,_ 1 _}[k]_ to _{_ 0 _,_ 1 _}[k][′]_ . This will be convenient when simulating neural network components that output multiple bits. 

We will find it useful to consider threshold circuits as a kind of compilation target for computation graphs: in other words, we will be concerned with simulating computation graphs defined over more complex functions and data types into threshold circuits. 

## **5.2 Computation Graph Families** 

A computation graph family over D and F is a mapping from _n ∈_ N to a computation graph _Gn_ for processing inputs of size _n_ . Thus, _G_ defines a function from D _[∗] →_ D, where _G_ ( _x_ ) = _G|x|_ ( _x_ ). 

> 13Theorem 2 can also be extended to apply to log-precision transformers with _log-uniform weights_ , i.e., where _θT_ can grow in size and precision with _n_ (see Appendix B). 

> 14By convention in computer science, we let computation graph nodes be zero-indexed. 

> 15For more background on threshold circuits, see Merrill & Sabharwal (2023) and Merrill et al. (2022). 

6 

Intuitively, computation graph families are useful because they generalize computation graphs to define functions over _unbounded-length_ strings as inputs. 

**Size, depth, and arity.** For computation graph families, the size, depth, and arity become functions of the input length _n_ : size _G_ ( _n_ ) = size( _Gn_ ) _,_ depth _G_ ( _n_ ) = depth( _Gn_ ) _,_ arity _G_ ( _n, i_ ) = arity( _Gn, i_ ) _._ 

**Uniformity.** The infinite set _G_ can be alternatively represented by two functions: 

1. node _G_ ( _n, i_ ), which returns the type of node _i_ in _Gn_ if _i ≤_ size( _Gn_ ), and _∅_ otherwise. For example, if node _i_ computes the logical AND of its inputs, then node _G_ ( _n, i_ ) = _∧_ . 

2. edge _G_ ( _n, i, j_ ), which returns the argument index of _i_ into node _j_ if _Gn_ contains an edge _i → j_ and _−_ 1 otherwise. edge _G_ ( _n, i, j_ ) only needs to be defined over _i, j <_ size( _Gn_ ). For example, if _Gn_ contains a node _j_ with three incoming edges, the second of which comes from node _i_ , then edge _G_ ( _n, i, j_ ) = 1. 

A pair of algorithms implementing these two functions uniquely specifies a computation graph family, as it enables building the computation graph _Gn_ for any _n_ . Uniform computation graph families (generalizing uniform circuits; cf. Arora & Barak, 2009) are families where node _G_ and edge _G_ can be computed efficiently, i.e., under some constraints on space or time: 

**Definition 5** (Uniformity) **.** A computation graph family _G_ is _T_ ( _n_ )-uniform iff node _G_ ( _n, i_ ) and edge _G_ ( _n, i, j_ ) can be computed by a deterministic Turing machine in time _T_ ( _n_ ). We focus on _log-uniform_ computation graph families: i.e., where _T_ ( _n_ ) = O(log _n_ ).[16] 

**Threshold circuit families.** These are simply families of threshold circuits. We will be simulating computation graph families with threshold circuit families. Log-uniform TC[0] is the class of languages recognized by log-uniform constant-depth, poly-size threshold circuit families. See Merrill & Sabharwal (2023); Liu et al. (2023); Arora & Barak (2009) for more background on TC[0] and circuits. 

## **6 Proof of Theorem 2** 

The idea is to simulate a transformer with a log-uniform TC[0] circuit family. Since log-uniform TC[0] = FO(M), this would imply any transformer can be expressed in FO(M). First, we note that transformers are log-uniform computation graphs: 

**Lemma 1** (Proof in Appendix B.1) **.** _A transformer T is a log-uniform computation graph family where_ F _contains embedding, self-attention, feedforward, and output components._ 

Further, each core module of the transformer can be simulated by a log-uniform TC[0] circuit family: 

**Lemma 2** (Proof in Appendix B.2) **.** _Let T be a log-precision transformer with fixed parameters θT . Then each component in_ F _is computable in log-uniform_ TC[0] _._ 

Intuitively, we can now simulate a transformer in log-uniform TC[0] by just simulating each of its components with a threshold circuit and routing their inputs and outputs appropriately. However, we will need two more technical conditions to verify that this construction is indeed log-uniform: 

**Lemma 3** (Proof in Appendix B.3) **.** _Let T be a log-precision transformer with fixed parameters θT . There exists a function_ bsize( _n_ ) _that is a power of_ 2 _and computable in_ O(log _n_ ) _time s.t._ size _F_ ( _n_ ) _≤_ bsize( _n_ ) _for all F ∈_ F _._ 

**Lemma 4** (Proof in Appendix B.4) **.** _If F is a log-uniform_ TC[0] _family and_ size _F_ ( _n_ ) _≤_ bsize( _n_ ) _, there exists a log-uniform_ TC[0] _family F[′] s.t. F_ ( _x_ ) = _F[′]_ ( _x_ ) _for all x and_ size _F ′_ ( _n_ ) = bsize( _n_ ) _._ 

Combined, Lemmas 3 and 4 show that each _F ∈_ F is computable by a log-uniform TC[0] family with size bsize( _n_ ) that is a power of 2 and computable in time O(log _n_ ). We will show these conditions imply a transformer _T_ can be simulated by a TC[0] family _C_ (Theorem 3) and moreover that _C_ is log-uniform (Corollary 3.2). By the equivalence of log-uniform TC[0] and FO(M) (Barrington et al., 1990), we then conclude that any log-precision transformer can be expressed in FO(M). 

16Past work (Merrill & Sabharwal, 2023) analyzes transformers with a similarly named but weaker notion of uniformity, namely log- _space_ (rather than log- _time_ ) uniformity. 

7 

**Algorithm 1** node _C_ ( _n, i_ ) _Return the type of gate i in circuit Cn._ 

- 1: _F ←_ node _G_ ( _n,_ bnode( _n, i_ )) 

- 2: **if** _F_ = _∅_ **then** 3: **return** node _F_ ( _n, i −_ bstart( _n, i[′]_ )) 

- 4: **else return** _∅_ 

## **Algorithm 2** edge _C_ ( _n, i, j_ ) 

_If Cn contains an edge i → j, return the argument number of that edge. Otherwise, return −_ 1 _._ 

- 1: _i[′] ←_ bnode( _n, i_ ) 

- 2: _j[′] ←_ bnode( _n, j_ ) 

- 3: _si ←_ bstart( _n, i[′]_ ) 

- 4: _sj ←_ bstart( _n, j[′]_ ) 5: **if** _i[′]_ = _j[′]_ **then** 6: _F ←_ node _G_ ( _n, i[′]_ ) 7: **return** edge _F_ ( _n, i − si, j − sj_ ) 8: **else if** edge _G_ ( _n, i[′] , j[′]_ ) _≥_ 0 **then** 9: _bi ← i −_ ( _si_ + bsize( _n, i[′]_ ) _− p_ ( _n_ )) 

- 10: _bj ← j −_ ( _sj_ + _p_ ( _n_ ) _·_ edge _G_ ( _n, i[′] , j[′]_ )) 11: **if** _bi_ = _bj < p_ ( _n_ ) **then return** _j − sj_ 12: **else return** _−_ 1 13: **else return** _−_ 1 

## **6.1 Simulating Computation Graph Families with Circuit Families** 

We give algorithms that take a computation graph family and define a circuit family simulating it. Intuitively, the algorithms creates contiguous blocks of circuit gates simulating each node in the computation graph and route inputs and outputs between blocks appropriately. 

**Block mapping.** This algorithm depends on a _block mapping_ , which is an implementation of the following three functions: 

1. The _block node_ bnode( _n, i_ ): the index of the node that gate _i_ ’s block is simulating. 

2. The _block start_ bstart( _n, i[′]_ ): the smallest gate index in the block simulating node _i[′]_ . 

3. The _block size_ bsize( _n, i[′]_ ): the number of gates in the block simulating node _i[′]_ . 

Further, we enforce that a valid block mapping must satisfy that, for all _i_ , with _i[′]_ = bnode( _n, i_ ), 

bstart( _n, i[′]_ ) _≤ i <_ bstart( _n, i[′]_ ) + bsize( _n, i[′]_ ) _._ 

Let _G_ be a computation graph whose primitive functions are computable by log-uniform threshold circuits. We can identify each primitive function with a log-uniform threshold circuit family _F_ that computes it, where the first arity _F_ ( _n_ ) gates are IDENTITY gates reserved for taking input. For such a graph, node _G_ can be taken to return a symbol identifying a circuit family _F_ . In this case, our algorithm requires that, for all _i[′]_ , the block size of _i[′]_ must match the size of the circuit for the type of block _i[′]_ , i.e., bsize( _n, i[′]_ ) = sizenode _G_ ( _n,i′_ )( _n_ ). These properties let us meaningfully identify a graph node _i[′]_ with a block of nodes that will simulate it. This intuition enables us to develop Algorithms 1 and 2 for constructing a uniform threshold circuit family from a uniform computation graph family. 

**Theorem 3.** _Let G be a computation graph over a finite set of node types_ F _, where each F ∈_ F _is specified by a log-uniform circuit family. Let_ bnode _,_ bstart _, and_ bsize _be a valid block mapping in the sense above. Then Algorithms 1 and 2 define a circuit family C such that_ 

_1. C and G compute the same_ D _[∗] p[→]_[D] _[p][function (let the final][ p][ gates of each][ C][i][be its output).]_ 

_2._ depth _C_ ( _n_ ) _≤_ depth _G_ ( _n_ ) _·_ max _F_ depth _F_ ( _n_ ) _._ 

_3._ size _C_ ( _n_ ) _≤_ size _G_ ( _n_ ) _·_ max _F_ size _F_ ( _n_ ) _._ 

_Proof._ Assume w.l.o.g. that the gates of _C_ are topologically ordered. We show by induction over circuit gates _j_ (with _j[′]_ = bnode( _n, j_ )) that: 

1. For all _i[′] < j[′]_ , the last _p_ nodes of block _i[′]_ store the value of node _i[′]_ . 

2. For all _i_ such that bstart( _n, j[′]_ ) _≤ i ≤ j_ , gate _i_ of _C_ (as a function of the input nodes of _j[′]_ ) computes gate _i −_ bstart( _n, j[′]_ ) of node _G_ ( _n, j[′]_ ). 

Base case. We have two circuits with no gates, so the premises are trivially satisfied. 

Inductive case. Assume the premises hold up to _j_ . We will show they hold for _j_ + 1. Let _T_ = node _G_ ( _n, j[′]_ ). By Premise 1, we know that the last _p_ nodes of block _i[′]_ store the output of node _i[′]_ , for 

8 

_i[′] < j[′]_ . By Algorithm 2, for each _i[′]_ such that edge _G_ ( _n, i[′] , j[′]_ ) = _a_ with 0 _≤ k <_ arity _F_ ( _n_ ), gates _kp_ through _k_ ( _p_ + 1) _−_ 1 of block _j[′]_ will copy the final _p_ gates of block _i[′]_ . Thus, the first _k ·_ arity _F_ ( _n_ ) gates of block _j[′]_ store the inputs to node _j[′]_ . 

At this point, we use Premise 2 to conclude that the first _j −_ bstart( _n, j[′]_ ) gates of block _j[′]_ compute the same function as the first _j −_ bstart( _n, j[′]_ ) gates of _F_ with respect to this input. Thus, we just need to show that gate _j_ + 1 is also correct. Within Algorithm 2, we fall in case _i[′]_ = _j[′]_ , meaning that gate _j_ + 1 of block _j[′]_ gates the same inputs as gate _j_ + 1 of _F_ . By Algorithm 1, the type of gate _j_ + 1 in block _j[′]_ is the type of gate _j_ + 1 of _F_ . Thus, gate _j_ + 1 in block _j[′]_ computes the same function of the input gates as gate _j_ + 1 in _F_ . If _j_ + 1 = bsize( _n, j[′]_ ), we conclude that the final _p_ gates of block _j[′]_ store the output of node _j[′]_ . 

Let XC[0] denote any family of constant-depth, poly-size circuits, including AC[0] and TC[0] .[17] **Corollary 3.1.** _Let G be a constant-depth, poly-size computation graph family over a finite_ F _. If every node type in_ F _can be computed by_ XC[0] _circuits, the function computed by G is in_ XC[0] _._ 

Since a transformer has constant depth and polynomial size, Corollary 3.1 lets us easily recover prior results about hard-attention transformers (Hao et al., 2022; Hahn, 2020) and saturated attention transformers (Merrill et al., 2022) using a common framework. All one has to do is show that all individual node types in such transformers can be computed by AC[0] and TC[0] circuits, respectively. 

Corollary 3.1 established that Algorithms 1 and 2 construct a circuit family that simulates _G_ . With the right block mapping, _C_ will be log-uniform as long as _G_ and its node types are log-uniform. **Corollary 3.2.** _Let G be a log-uniform, constant-depth computation graph family over a finite_ F _, where each F ∈_ F _is specified by a log-uniform_ TC[0] _family with_ size _F_ ( _n_ ) = bsize( _n_ ) _that is a power of_ 2 _computable in_ O(log _n_ ) _time. Then G can be simulated by a log-uniform_ TC[0] _family C that obeys the size and depth properties of Theorem 3._ 

_Proof._ Let _C_ be the circuit family defined by Algorithms 1 and 2 given _G_ and the following block mapping: bnode( _n, i_ ) = _⌊i/_ bsize( _n_ ) _⌋,_ bstart( _n, i[′]_ ) = _i[′] ·_ bsize( _n_ ) _,_ bsize( _n, i[′]_ ) = bsize( _n_ ) _._ Since bsize( _n_ ) is a power of 2, bnode and bstart are reducible to left and right shifting over O(log _n_ )- bit integers, which can be implemented in O(log _n_ ) time. Thus, each block mapping function is computable in time O(log _n_ ). Since node _G_ and edge _G_ are just calling functions computable in time O(log _n_ ) with constant overhead, we conclude that _C_ , the circuit family they define, is log-uniform, and it is already known to simulate _G_ with constant depth and polynomial size by Theorem 3. 

## **7 Conclusion** 

We proved that any log-precision transformer classifier can be translated to an FO(M) sentence that computes the same function (on all inputs of any length). This result comes by first simulating a transformer with a highly uniform threshold circuit family, and then leveraging the established equivalence of log-uniform circuits and FO(M). Transformers and other neural nets are often discussed in contrast with symbolic models based on logical formalisms (Garnelo & Shanahan, 2019)—an immediate implication of our result is that it is possible to express the inner workings of transformers also in a simple logic, challenging the premise of a rigid division between symbolic and neural models. Our results also provide the tightest known upper bound on log-precision transformers. 

While it is striking that a full transformer can be translated to a sentence in a logic as simple as FO(M), we believe the bound is not tight. In particular, we conjecture that it is possible to simulate any transformer with an FO(M) sentence of quantifier depth of at most 2, which could be proven by establishing a hierarchy theorem describing the FO(M) quantifier depth needed to simulate a TC[0] family of a certain size. It would also be an interesting extension to translate real transformers to FO(M) sentences. In this sense, we believe our results provide a theoretical foundation to guide mechanistic interpretability work (cf. Weiss et al., 2021; Lindner et al., 2023). 

Our findings provide a novel view into transformer classifiers and their limits. It would be exciting for future research to extend our results to account for other common practical uses of transformers, such as for long-form generation, chain-of-thought reasoning, and in-context learning. 

> 17Formally, F just needs to contain _∧_ and _∨_ . 

9 

## **Acknowledgments** 

We thank Paul Beame, David Chiang, anonymous reviewers, and researchers at the Allen Institute for AI for feedback. Thanks to Noa Nabeshima for pointing out a minor notational inconsistency. WM was supported by an NSF graduate research fellowship and in part by NSF award 1922658. 

## **References** 

- Aaronson, S., Kuperberg, G., and Habryka, O. TC[0] : Constant depth threshold circuits, 2022. URL `https://complexityzoo.net/Complexity_Zoo:T#tc0` . 

- Arora, S. and Barak, B. _Computational Complexity: A Modern Approach_ . Cambridge University Press, 2009. 

- Barrington, D. A. M., Immerman, N., and Straubing, H. On uniformity within NC[1] . _Journal of Computer and System Sciences_ , 41(3):274–306, 1990. 

- Bhattamishra, S., Ahuja, K., and Goyal, N. On the ability and limitations of transformers to recognize formal languages. In _EMNLP_ , 2020. 

- Brent, R. P. and Zimmermann, P. _Modern computer arithmetic_ , volume 18. Cambridge University Press, 2010. 

- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In _NeurIPS_ , 2020. 

- Chiang, D., Cholak, P., and Pillay, A. Tighter bounds on the expressivity of transformer encoders. _ICML_ , 2023. 

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. https://transformer-circuits.pub/2021/framework/index.html. 

- Furst, M. L., Saxe, J. B., and Sipser, M. Parity, circuits, and the polynomial-time hierarchy. _Mathematical systems theory_ , 17:13–27, 1984. 

- Garnelo, M. and Shanahan, M. Reconciling deep learning with symbolic artificial intelligence: representing objects and relations. _Current Opinion in Behavioral Sciences_ , 29:17–23, 2019. ISSN 2352-1546. 

- Hahn, M. Theoretical limitations of self-attention in neural sequence models. _TACL_ , 8:156–171, 2020. 

- Hao, Y., Angluin, D., and Frank, R. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. _TACL_ , 10:800–810, 2022. 

- Hesse, W. Division is in uniform TC[0] . In _International Colloquium on Automata, Languages, and Programming_ , pp. 104–114. Springer, 2001. 

- Hunter, P., Bouyer, P., Markey, N., Ouaknine, J., and Worrell, J. Computing rational radical sums in uniform TC0. _Foundations of Software Technology and Theoretical Computer Science_ , 2010. 

- Lindner, D., Kramár, J., Rahtz, M., McGrath, T., and Mikulik, V. Tracr: Compiled transformers as a laboratory for interpretability. _arXiv_ , abs/2301.05062, 2023. 

- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., and Zhang, C. Transformers learn shortcuts to automata. In _ICLR_ , 2023. 

10 

Merrill, W. and Sabharwal, A. The parallelism tradeoff: Limitations of log-precision transformers. _TACL_ , 11:531–545, 2023. 

- Merrill, W., Ramanujan, V., Goldberg, Y., Schwartz, R., and Smith, N. A. Effects of parameter norm growth during transformer training: Inductive bias from gradient descent. In _EMNLP_ , 2021. 

- Merrill, W., Sabharwal, A., and Smith, N. A. Saturated transformers are constant-depth threshold circuits. _TACL_ , 10, 2022. 

- Pérez, J., Marinkovi´c, J., and Barceló, P. On the Turing completeness of modern neural network architectures. In _ICLR_ , 2019. 

- Thoppilan, R., Freitas, D. D., Hall, J., Shazeer, N. M., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y., Li, Y., Lee, H., Zheng, H., Ghafouri, A., Menegali, M., Huang, Y., Krikun, M., Lepikhin, D., Qin, J., Chen, D., Xu, Y., Chen, Z., Roberts, A., Bosma, M., Zhou, Y., Chang, C.-C., Krivokon, I. A., Rusch, W. J., Pickett, M., Meier-Hellstern, K. S., Morris, M. R., Doshi, T., Santos, R. D., Duke, T., Søraker, J. H., Zevenbergen, B., Prabhakaran, V., Díaz, M., Hutchinson, B., Olson, K., Molina, A., Hoffman-John, E., Lee, J., Aroyo, L., Rajakumar, R., Butryna, A., Lamm, M., Kuzmina, V. O., Fenton, J., Cohen, A., Bernstein, R., Kurzweil, R., Aguera-Arcas, B., Cui, C., Croak, M., Chi, E., and Le, Q. LaMDA: Language models for dialog applications. _ArXiv_ , abs/2201.08239, 2022. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In _NeurIPS_ , 2017. 

- Weiss, G., Goldberg, Y., and Yahav, E. On the practical computational power of finite precision RNNs for language recognition. In _ACL_ , 2018. 

- Weiss, G., Goldberg, Y., and Yahav, E. Thinking like transformers. _ICML_ , 2021. 

- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T.-Y. On layer normalization in the transformer architecture. In _ICML_ , 2020. 

11 

## **A Conditional Majority** 

Given formulas _ϕ, ψ_ , M _i_ : _ϕ. ψ_ is a sentence that is true iff _ψ_ is true for at least half the values of _i_ that make _ϕ_ true. 

**Proposition 2.** _For any two predicates ϕ_ ( _i_ ) _and ψ_ ( _i_ ) _,_ M _i_ : _ϕ_ ( _i_ ) _. ψ_ ( _i_ ) _can be expressed in_ FO(M) _._ 

_Proof._ M _i_ : _ϕ. ψ_ can be rewritten using a counting quantifier and a threshold quantifier: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0012-04.png)


The formula 2 _k[′]_ = _k_ can be defined using bit. We then use the fact that counting and threshold quantifiers can be expressed in terms of majority quantifiers (Barrington et al., 1990) to conclude that M _i_ : _ϕ. ψ_ can be expressed in FO(M). 

## **B Omitted Proofs** 

Table 1 summarizes the notation we use in the following proofs when describing computation graphs and circuit families. 

Table 1: Summary of common notation for computation graph and circuit families. 

|Graph|Circuit|Output Range|Description|
|---|---|---|---|
|_i′_|_i_|Z|index of node or gate|
|node_G_(_n, i′_)|node_C_(_n, i_)|F18|type of node or gate|
|edge_G_(_n, i′, j′_)|edge_C_(_n, i, j_)|Z|argument # of edge_i →j_|
|size_G_(_n_)|size_C_(_n_)|Z|# of nodes or gates|
|depth_G_(_n_)|depth_C_(_n_)|Z|longestpath length|
|||||
|bnode(_n, i_)||[0_,_size_G_(_n_)]|block containing_i_|
|bstart(_n, i′_)||[0_,_size_C_(_n_)]|frst gate in block_i′_|
|bsize(_n, i′_)||Z|size of block_i′_|



## **B.1 Transformers are Log-Uniform Computation Graph Families** 

We now justify that the computation graph family defining a transformer is log-uniform. To do this, we introduce a stronger notion of uniformity called _column uniformity_ that captures the highly regular structure of the transformer. 

Let node( _G, i_ ) be the _i_ -th node of computation graph _G_ . Let _a_ mod _b_ be the remainder when _a_ is divided by _b_ . 

**Definition 6** (Column uniformity) **.** A computation graph family _G_ is _T_ ( _n_ )-column-uniform iff there exists a computation graph _K_ (with fixed size w.r.t _n_ ) such that, for all _i, j_ such that 0 _≤ i, j <_ size _G_ ( _n_ ): 

1. node _G_ ( _n, i_ ) = node ( _K, i_ mod size( _K_ )). 

2. If _⌊i/_ size( _K_ ) _⌋_ = _⌊j/_ size( _K_ ) _⌋_ , then 

edge _G_ ( _n, i, j_ ) = edge ( _K, i_ mod size( _K_ ) _, j_ mod size( _K_ )) _._ 

Otherwise, edge _G_ ( _n, i, j_ ) can be computed by a deterministic Turing machine in time _T_ ( _n_ ). 

We define _log-column-uniform_ analogously to log-uniform: i.e., we let _T_ ( _n_ ) = O(log _n_ ). logcolumn-uniform implies log-uniform because our implementations of node _G_ and edge _G_ can store _K_ in a finite lookup table and compute the quotient and remainder of _i_ and _j_ by size( _K_ ) in O(log _n_ ) time using Lemma 12. The edges outside of _K_ are computable in O(log _n_ ) time by construction. 

**Lemma 1** (Proof in Appendix B.1) **.** _A transformer T is a log-uniform computation graph family where_ F _contains embedding, self-attention, feedforward, and output components._ 

12 

_Proof._ We show the stronger condition that any transformer _T_ is a log-column-uniform computation graph family, which implies it is log-uniform. 

We have the column _K_ by Definition 2: all that remains to show is that edge _GT_ can be computed in time O(log _n_ ) for edges outside the column. These edges route from the layer _ℓ_ output to the self-attention heads of layer _ℓ_ + 1. Following from the column structure, there exists _kℓ_ such that a node _i_ is an output vector of layer _ℓ_ iff _kℓ_ = _i_ mod size( _K_ ). In a finite lookup table, we can store _kℓ_ for each _ℓ_ + 1, and use this for self-attention routing. For an unmasked self-attention head _j_ , we compute: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0013-02.png)


For causally masked attention, we extend the first case to check that _⌊i/_ size( _K_ ) _⌋≤⌊j/_ size( _K_ ) _⌋_ . Either way, this logic can be implemented in time O(log _n_ ) via Lemma 12. Thus, we conclude that _GT_ is column-uniform. 

## **B.2 Transformer Components are Computable by Log-Uniform Threshold Circuits** 

**Lemma 2** (Proof in Appendix B.2) **.** _Let T be a log-precision transformer with fixed parameters θT . Then each component in_ F _is computable in log-uniform_ TC[0] _._ 

We prove a more general version of Lemma 2 that handles some cases with weights growing with _n_ . The weights _θT_ are just a special case of a computation graph (that do not depend on the input); we can thus apply our definition of log-uniform to them. Lemma 2 follows from a more general result with log-uniform _θT_ : 

**Lemma 5.** _Let T be a log-uniform transformer with log-uniform θT . Then each component in_ F _is computable in log-uniform_ TC[0] _._ 

_Proof._ In Appendix C, we show that log-uniform _θT_ implies: 

1. The embedding component is computable in log-uniform TC[0] (Lemma 6). 

2. The self attention mechanism is computable in log-uniform TC[0] (Lemma 7). 

3. The activation block is computable in log-uniform TC[0] (Lemma 8). 

4. The output classifier head is computable in log-uniform TC[0] (Lemma 9). 

We have shown that each _F ∈_ F is computable in log-uniform TC[0] . 

## **B.3 Transformer Component Size Has a Log-Time Upper Bound** 

**Lemma 3** (Proof in Appendix B.3) **.** _Let T be a log-precision transformer with fixed parameters θT . There exists a function_ bsize( _n_ ) _that is a power of_ 2 _and computable in_ O(log _n_ ) _time s.t._ size _F_ ( _n_ ) _≤_ bsize( _n_ ) _for all F ∈_ F _._ 

_Proof._ Let 2 _[b]_[(] _[n]_[)] be the least power of 2 at least as large as size _F_ ( _n_ ) for all _F_ . We observe that 2 _[b]_[(] _[n]_[)] is at most 2 _·_ max _F_ size _F_ ( _n_ ) for all _n_ . Because each _F_ has poly size, there is a fixed _k_ such that, for large enough _n_ ,[19] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0013-17.png)


Define _b[′]_ ( _n_ ) = _k⌈_ log _n⌉_ and bsize( _n_ ) = 2 _[b][′]_[(] _[n]_[)] . bsize( _n_ ) is both a power of 2 and an upper bound on 2 _[b]_[(] _[n]_[)] ; what remains to be shown is that it can be computed in time O(log _n_ ). We can first compute _⌈_ log _n⌉_ in time O(log _n_ ) by finding the greatest nonzero index of _n_ . Next, we can compute _b[′]_ ( _n_ ) = _k ·⌈_ log _n⌉_ in time O(log log _n_ ) since _k_ is fixed size and _⌈_ log _n⌉_ has size at most O(log log _n_ ) (Brent & Zimmermann, 2010). Finally, we compute bsize( _n_ ) = 2 _[b][′]_[(] _[n]_[)] by simply left-shifting 1 at most O(log _n_ ) times. 

> 19We can compute bsize( _n_ ) for small _n_ using finite lookup. 

13 

## **B.4 Circuit Families Can Be Padded to Log-Time Size Upper Bounds** 

Recall that the last _p_ bits of our circuits represent the circuit’s output (cf. Section 5.1). In Lemma 4, we consider _F_ ( _x_ ) = _F[′]_ ( _x_ ) if and only if the last _p_ bits of _F_ and _F[′]_ agree for all _x_ . 

**Lemma 4** (Proof in Appendix B.4) **.** _If F is a log-uniform_ TC[0] _family and_ size _F_ ( _n_ ) _≤_ bsize( _n_ ) _, there exists a log-uniform_ TC[0] _family F[′] s.t. F_ ( _x_ ) = _F[′]_ ( _x_ ) _for all x and_ size _F ′_ ( _n_ ) = bsize( _n_ ) _._ 

> _Proof._ The high level idea is that we can pad _F_ to a circuit _F[′]_ that has size bsize( _n_ ) and simply copies over the _p_ output bits of _F_ to its own last _p_ bits using identity gates. 

We first set node _F ′_ to copy over the existing circuit and append identity nodes. Let Id denote an identity node. Then node _F ′_ is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0014-05.png)


We see that the size of _F[′]_ will thus be of size bsize( _n_ ). 

Next, we extend edge _F ′_ ( _n, i, j_ ) to route the original output bits to the new output bits. Recall that an edge value of 0 means _i_ is the first argument of gate _j_ , and an edge value of _−_ 1 means there is no edge _i → j_ . Let _kj_ = _p_ ( _n_ ) _−_ (bsize( _n_ ) _− j_ ) be the index of node _j_ as an output gate in _F[′]_ . For example, _k_ = 0 for the first output bit. Now let output _F_ ( _n, i, k_ ) represent whether node _i_ is the _k_ -th output of _Fn_ . We can compute output _F_ ( _n, i, k_ ) in terms of node _F_ as follows: 

output _F_ ( _n, i, k_ ) _⇐⇒_ node _F_ ( _n, i_ + _p_ ( _n_ ) _− k −_ 1) _̸_ = _∅∧_ node _F_ ( _n, i_ + _p_ ( _n_ ) _− k_ ) = _∅._ 

Then edge _F ′_ is defined: 

edge _F_ ( _n, i, j_ ) if edge _F_ ( _n, i, j_ ) _̸_ = _−_ 1 edge _F ′_ ( _n, i, j_ ) = 0 if output _F_ ( _n, i, kj_ )   _−_ 1 otherwise _._ 

The first condition simply copies over the original edges. The second condition adds _p_ ( _n_ ) new edges (for the different values of _k_ ) that route the final _p_ ( _n_ ) nodes of _F_ to the final _p_ ( _n_ ) nodes of _F[′]_ , guaranteeing that the two circuits will compute the same function. 

Because both node _F ′_ and edge _F ′_ just rely on addition, conditional branching, and a finite number of calls to functions computable in time O(log _n_ ), they are both computable in time O(log _n_ ). 

## **C Transformer Column Components** 

In this section, we generally omit layer subscripts for clarity. We assume a pre-norm (Xiong et al., 2020) parameterization of the transformer for concreteness and because this is more standard in newer transformers. However, the results would also hold with the original post-norm (Vaswani et al., 2017). 

As mentioned in the main text, we view _θT_ as a concatenation of the parameters for the transformer functions. Thus, if _m_ and _w_ are computable in time O(log _n_ ) and _θT_ is log-uniform, it follows that the parameter vector for each _ϕ, s, v, f_ , and _κ_ is itself log-uniform because we can map indices in the smaller parameter vectors to indices in _θT_ in time O(log _n_ ). 

## **C.1 Transformer Embeddings** 

For each position 1 _≤ i ≤ n_ , the transformer embedding function represents token _σi ∈_ Σ and its position _i_ with a vector. Let **V** be an embedding matrix of size _|_ Σ _| × m_ where each row represents the embedding for some _σ_ . Let _f_ : N _→_ D _[m] p_[be computable in time][ O(log] _[ n]_[)][.][Then,] 

_ϕ_ ( _σi, i_ ) = **v** _σi_ + _f_ ( _i_ ) _._ 

**Lemma 6.** _If θT is log-uniform, then ϕ is computable in log-uniform_ TC[0] _._ 

14 

_Proof._ The embedding block can be expressed as a constant-size computation graph that constructs **V** , computes **v** _σi_ using an affine transformation, computes _f_ ( _i_ ), and then, finally, sums **v** _σi_ and _f_ ( _i_ ). The first step is computable by a log-uniform constant-depth, poly-size threshold circuit family since _θT_ is log-uniform. We can compute an affine transformation via a log-uniform constant-depth poly-size threshold circuit family via Lemma 10. _f_ ( _i_ ) can be directly computed by the Turing machine constructing the circuit by construction. The sum of the two terms can then be computed by a log-uniform constant-depth threshold circuit of size polynomial in _m_ , which is also polynomial in _n_ . Since we have a computation graph where all node types are computable by log-uniform, constant-depth, poly-size threshold circuit families, we conclude by Corollary 3.2 that _ϕ_ can also be computed by log-uniform, constant-depth, poly-size threshold circuit family. 

## **C.2 Self Attention** 

The two components of the self attention block are _s_ , the similarity function, and _v_ , the value function. Let **h** _i_ be the hidden state at the previous layer and **h**[¯] _i_ = lnorm( **h** _i_ ). Then, the similarity function first computes queries and keys, and then takes the scaled dot-product between them: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0015-03.png)


Then the value function is defined _v_ ( **h** _i_ ) = **W** _h_ **h**[¯] _i_ + **b** _h_ . We first show that the value function (and also the keys and queries by symmetry) is computable in log-uniform TC[0] : 

**Lemma 7.** _If θT is log-uniform, then the self-attention component is computable in log-uniform_ TC[0] _._ 

_Proof. v_ is a composition of constructing the parameters (in log-uniform TC[0] since _θT_ is loguniform), layer norm (in log-uniform TC[0] by Lemma 11), and an affine transformation (in loguniform TC[0] by Lemma 10). Thus, _v_ is computable in log-uniform TC[0] . 

Computing _s_ is a constant-depth computation graph. First, we compute **q** _i_ and **k** _i_ and then multiply them, and all of these steps are in log-uniform TC[0] . Next, we can compute _m_ and _h_ in time O(log _n_ ) and build a log-uniform TC[0] circuit that divides the product of the last step by � _m/h_ . Finally, we compute _p_ -precision exp, which can be expressed in log-uniform TC[0] as multiplication followed by left-shifting. Thus, by Corollary 3.2, _s_ can be computed in log-uniform TC[0] . 

_s_ and _v_ are log-uniform, so their size _p_ is at most poly( _n_ ). Computing self attention reduces to binary multiplication and division over D _p_ , and performing iterated addition (summation) over _n_ numbers in D _p_ . Binary multiplication, binary division (Hesse, 2001), and iterated addition (Merrill & Sabharwal, 2023) can all be computed in log-uniform TC[0] , i.e., by a log-uniform, constant-depth threshold circuit family of size at most poly( _p_ ) _≤_ poly( _n_ ). Thus, self attention can also be computed in log-uniform TC[0] . 

## **C.3 Activation Block** 

The activation function _f_ encapsulates the aggregation of the attention head outputs and the feedforward subnetwork of the transformer. _f_ takes as input attention head outputs **a** _i,_ 1 _, . . . ,_ **a** _i,h ∈_ D _[m/h] p_ and the previous layer value **h** _i_ . 

The first part of the activation block simulates the pooling part of the self-attention sublayer. The head outputs are first concatenated to form a vector **a** _i_ , which is then passed through an affine transformation ( **W** _o,_ **b** _o_ ) : D _[m] p[→]_[D] _[m] p_[followed by residual connections to form the sublayer output] **o** _i ∈_ D _[m] p_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0015-12.png)


The second part of the activation block first applies layer-norm and then simulates the feedforward subnetwork to compute the next layer vector **h** _[′] i_[.][Let] **[o]**[¯] _[i]_[=][lnorm(] **[o]** _[i]_[)][.][Let] _[σ]_[be][a][nonlinearity] computable in linear time on its input (in the most standard transformer, ReLU). Then, for affine 

15 

**h** _[′] i_[=] **[ W]**[2] _[σ]_[(] **[W]**[1] **[o]**[¯] _[i]_[+] **[ b]**[1][) +] **[ b]**[2][+] **[ o]** _[i][.]_ 

transformations ( **W** 1 _,_ **b** 1) : D _[m] p[→]_[D] _[w] p_[and][ (] **[W]**[2] _[,]_ **[ b]**[2][) :][ D] _[w] p[→]_[D] _[m] p_[, the feedforward subnetwork can] be defined: 

**Lemma 8.** _If θT is log-uniform, then f is computable in log-uniform_ TC[0] _._ 

_Proof._ The activation block can be expressed as a constant-size computation graph where the nodes construct affine transformation parameters, apply affine transformations, compute layer-norm, and compute elementwise nonlinearities. Since each of these nodes is computable by a log-uniform, constant-depth, poly-size threshold circuit family, the activation block is as well. 

## **C.4 Output Classifier Head** 

We assume the output from the transformer is computed as follows. First, **h**[¯] 1 = lnorm( **h** 1). Then, we use a parameter vector **w** _∈_ D _[m] p_[and bias term] _[ b]_[ to compute:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0016-06.png)


**Lemma 9.** _If θT is log-uniform, then κ is computable in log-uniform_ TC[0] _._ 

> _Proof._ We can express computing _κ_ as a composition of constructing the parameters **w** _, b_ and computing the affine transformation. Both parts of this composition are computable by a log-uniform, constant-depth, poly-size threshold circuit family, so computing _κ_ is as well. 

## **D Neural Net Building Blocks** 

In this section we analyze the uniformity of common neural net building blocks that are used within the various high-level transformer components. 

## **D.1 Affine Transformations** 

Affine transformations are a core part of neural networks used in various parts of the transformer. An affine transformation takes as input parameters ( **W** _,_ **b** ) : D _[a] p[→]_[D] _[b] p_[and a vector] **[ x]** _[ ∈]_[D] _[a] p_[and returns] **Wx** + **b** . 

**Lemma 10.** _For p_ = O(log _n_ ) _, any p-precision affine transformation where_ **W** _,_ **b** _are log-uniform is computable by a log-uniform, constant-size threshold circuit family of size polynomial in a and b._ 

_Proof._ We first use the uniformity of **W** _,_ **b** to construct them in O(log _n_ ) time. For the transformation **Wx** + **b** , first compute each **w** _i ⊙_ **x** in parallel, where _⊙_ represents elementwise multiplication. Since binary multiplication over polynomial-size numbers is in log-uniform TC[0] , this can be done in parallel with log-uniform TC[0] circuits. We then use _b_ log-uniform, constant-depth, poly-size threshold circuit families, each corresponding to an output index, that compute the sum over the _a_ entries of each **w** _i ⊙_ **x** . The affine transformation corresponds to the composition of these two steps, and is thus computable by a log-uniform TC[0] circuit family. 

## **D.2 Layer Norm** 

The layer norm is applied between sublayers in the transformer. Let _µ_ = (1 _/d_ )[�] _[d] i_ =1 _[x][i]_[.][The layer] norm **y** _∈_ D _[m] p_[of a vector] **[ x]** _[ ∈]_[D] _[m] p_[is computed, for scalars] _[ a, b][ ∈]_[D] _[p]_[,] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0016-17.png)


**Lemma 11.** _If a, b are log-uniform, the layer norm over a vector of size m can be computed by a log-uniform threshold circuit family of constant depth and size polynomial in m._ 

16 

_Proof._ First compute _m_ using summation over the constant term 1 from 1 to _m_ . This summation can be computed by a log-uniform constant-depth threshold circuit family of size polynomial in _m_ . Then compute the sum over **x** using a similar circuit, and divide them to get _µ_ , using the fact that integer division is in log-uniform TC[0] (Hesse, 2001). We can then compute **x** _− µ_ in log-uniform TC[0] . 

At this point, we can compute _∥_ **x** _− µ∥_ in log-uniform TC[0] (Hunter et al., 2010), then divide each **x** _− µ_ by the norm in log-uniform TC[0] , and then apply the final affine transformation in log-uniform TC[0] (Lemma 10). Thus, computing layer norm is in log-uniform TC[0] . 

## **E Arithmetic Complexity** 

**Lemma 12.** _Given an m-bit integer a and n-bit integer b, we can compute the quotient ⌊a/b⌋ and remainder a_ mod _b in time_ O( _mn_ ) _._ 

_Proof._ Let _D_ ( _m, n_ ) and _M_ ( _m, n_ ) denote, respectively, the time complexity of dividing and multiplying an _m_ -bit integer by an _n_ -bit integer. Brent & Zimmermann (2010) give the following fact: _D_ ( _m_ + _n, n_ ) _≤_ O( _M_ ( _m, n_ )). With the goal of analyzing _D_ ( _m, n_ ), we apply this as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers_images/nips_2023_A_Logic_for_Expressing_Log-Precision_Transformers.pdf-0017-05.png)


Applying Lemma 12 when _a_ has size O(log _n_ ) and _b_ has size O(1) says that we can do division in time O(log _n_ ). 

17 

