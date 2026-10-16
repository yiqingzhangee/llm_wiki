# **Abstract Visual Reasoning: An Algebraic Approach for Solving Raven’s Progressive Matrices** 

Jingyi Xu[1] _[∗]_ Tushar Vaidya[2] _[◦∗]_ Yufei Wu[2] _[◦∗]_ Saket Chandra[1] Zhangsheng Lai[3] _[◦]_ Kai Fong Ernest Chong[1] _[†]_ 1Singapore University of Technology and Design 

2Nanyang Technological University 3Singapore Polytechnic 

jingyi ~~x~~ u@mymail.sutd.edu.sg lai ~~z~~ hangsheng@sp.edu.sg 

tushar.vaidya@ntu.edu.sg yufei002@e.ntu.edu.sg _{_ saket ~~c~~ handra,ernest ~~c~~ hong _}_ @sutd.edu.sg 

## **Abstract** 

_We introduce algebraic machine reasoning, a new reasoning framework that is well-suited for abstract reasoning. Effectively, algebraic machine reasoning reduces the difficult process of novel problem-solving to routine algebraic computation. The fundamental algebraic objects of interest are the ideals of some suitably initialized polynomial ring. We shall explain how solving Raven’s Progressive Matrices (RPMs) can be realized as computational problems in algebra, which combine various well-known algebraic subroutines that include: Computing the Gr¨obner basis of an ideal, checking for ideal containment, etc. Crucially, the additional algebraic structure satisfied by ideals allows for more operations on ideals beyond set-theoretic operations._ 

_Our algebraic machine reasoning framework is not only able to select the correct answer from a given answer set, but also able to generate the correct answer with only the question matrix given. Experiments on the I-RAVEN dataset yield an overall_ 93 _._ 2% _accuracy, which significantly outperforms the current state-of-the-art accuracy of_ 77 _._ 0% _and exceeds human performance at_ 84 _._ 4% _accuracy._ 

## **1. Introduction** 

When we think of machine reasoning, nothing captures our imagination more than the possibility that machines would eventually surpass humans in intelligence tests and general reasoning tasks. Even for humans, to excel in IQ tests, such as the well-known Raven’s progressive matrices (RPMs) [8], is already a non-trivial feat. A typical RPM instance is composed of a question matrix and an answer set; see Fig. 1. A question matrix is a 3 _×_ 3 grid of panels 

> _∗_ Equal contributions. _†_ Corresponding author. 

> _◦_ This work was done when the author was previously at SUTD. Code: https://github.com/Xu-Jingyi/AlgebraicMR 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0001-13.png)


Figure 1. An example of RPM instance from the I-RAVEN dataset. The correct answer is marked with a red box. 

that satisfy certain hidden rules, where the first 8 panels are filled with geometric entities, and the 9-th panel is “missing”. The goal is to infer the correct answer for this last panel from among the 8 panels in the given answer set. 

The ability to solve RPMs is the quintessential display of what cognitive scientists call fluid intelligence. The word “fluid” alludes to the mental agility of discovering new relations and abstractions [56], especially for solving _novel_ problems not encountered before. Thus, it is not surprising that abstract reasoning on _novel_ problems is widely hailed as the hallmark of human intelligence [9]. 

Although there has been much recent progress in machine reasoning [29, 32, 59, 60, 62, 63, 73, 75, 83, 84], a common criticism [13, 49, 50] is that existing reasoning frameworks have focused on approaches involving extensive training, even when solving well-established reasoning tests such as RPMs. Perhaps most pertinently, as [13] argues, reasoning tasks such as RPMs should not need task- 

> This work is supported by the National Research Foundation, Singapore under its AI Singapore Program (AISG Award No: AISG-RP2019-015) and under its NRFF Program (NRFFAI1-2019-0005), and by Ministry of Education, Singapore, under its Tier 2 Research Fund (MOET2EP20221-0016). 

1 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0002-00.png)


Figure 2. An overview of our algebraic machine reasoning framework, organized into 2 stages. 

specific performance optimization. After all, if a machine optimizes performance by training on task-specific data, then that task cannot possibly be novel to the machine. 

To better emulate human reasoning, we propose what we call “algebraic machine reasoning”, a new reasoning framework that is well-suited for abstract reasoning. Our framework solves RPMs _without_ needing to optimize for performance on task-specific data, analogous to how a gifted child solves RPMs without needing practice on RPMs. Our key starting point is to _define_ concepts as ideals of some suitably initialized polynomial ring. These ideals are treated as the “actual objects of study” in algebraic machine reasoning, which do not require any numerical values to be assigned to them. We shall elucidate how the RPM task can be realized as a computational problem in algebra involving ideals. 

Our reasoning framework can be broadly divided into two stages: (1) algebraic representation, and (2) algebraic machine reasoning; see Fig. 2. In the first stage, we represent RPM panels as ideals, based on perceptual attribute values extracted from object detection models. In the second stage, we propose 4 invariance modules to extract patterns from the RPM question matrix. 

To summarize, our main contributions are as follows: 

- We reduce “solving the RPM task” to “solving a computational problem in algebra”. Specifically, we present how the discovery of abstract patterns can be realized very concretely as algebraic computations known as primary decompositions of ideals. 

- In our algebraic machine reasoning framework, we introduce 4 invariance modules for extracting patterns that are meaningful to humans. 

- Our framework is not only able to select the correct answer from a given answer set, but also able to generate answers _without needing any given answer set_ . 

- Experiments conducted on RAVEN and I-RAVEN datasets demonstrate that our reasoning framework significantly outperforms state-of-the-art methods. 

## **2. Related Work** 

**RPM solvers** . There has been much recent interest in solving RPMs with deep-learning-based methods [29, 43, 62, 84, 85, 88–91]. Most methods extract features from raw RPM images using nueral networks, and select answers by measuring panel similarities. Several works instead focus on generating correct answers without needing the answer set [55, 64]. To evaluate the reasoning capabilities of these methods, RPM-like datasets such as PGM [62] and RAVEN [83] have been proposed. Subsequently, I-RAVEN [26] and RAVEN-FAIR [5] are introduced to overcome a shortcut flaw in the answer set generation of RAVEN. 

**Algebraic methods in AI** . Using algebraic methods in AI is not new. Systems of polynomial equations are commonly seen in computer vision [57] and robotics [12], which are solved algebraically via Gr¨obner basis computations. In statistical learning theory, methods in algebraic geometry [78] and algebraic statistics [15] are used to study singularities in statistical models [40, 79, 80, 82], to analyze generalization error in hierarchical models [76, 77], to learn invariant subspaces of probability distributions [34,38], and to model Bayesian networks [16, 70]. A common theme in these works is to study suitably defined algebraic varieties. In deep learning, algebraic methods are used to study the expressivity of neural nets [11, 31, 45, 87]. In automated theorem proving, Gr¨obner basis computations are used in proof-checking [68]. Recently, a matrix formulation of first-order logic was applied to the RPM task [86], where relations are approximated by matrices and reasoning is framed as a bilevel optimization task to find best-fit matrix operators. As far as we know, methods from commutative algebra have not been used in machine reasoning. 

## **3. Proposed Algebraic Framework** 

In abstract reasoning, a key cognitive step is to “discover patterns from observations”, which can be formulated con- 

2 

cretely as “finding invariances in observations”. In this section, we describe how algebraic objects known as ideals are used to represent RPM instances, how patterns are extracted from such algebraic representations, and how RPMs can be solved, both for answer selection and answer generation, as computational problems in algebra. 

## **3.1. Preliminaries** 

Throughout, let _R_ = R[ _x_ 1 _, . . . , xn_ ] be the ring of polynomials in variables _x_ 1 _, . . . , xn_ , with real coefficients. In particular, _R_ is closed under addition and multiplication of polynomials, i.e., for any _a, b ∈ R_ , we have _a_ + _b, ab ∈ R_ . 

## **3.1.1 Algebraic definitions** 

**Ideals in polynomial rings** . A subset _I ⊆ R_ is called an _ideal_ if there exist polynomials _g_ 1 _, . . . , gk_ in _R_ such that 

_I_ = _{f_ 1 _g_ 1 + _· · ·_ + _fkgk|f_ 1 _, . . . , fk ∈ R}_ contains all polynomial combinations of _g_ 1 _, . . . , gk_ . We say that _G_ = _{g_ 1 _, . . . , gk}_ is a _generating set_ for _I_ , we call _g_ 1 _, . . . , gk generators_ , and we write either _I_ = _⟨g_ 1 _, . . . , gk⟩_ or _I_ = _⟨G⟩_ . Note that generating sets of ideals are not unique. If _I_ has a generating set consisting only of monomials, then we say that _I_ is a _monomial ideal_ . (Recall that a _monomial_ is a polynomial with a single term.) Given ideals _J_ 1 = _⟨g_ 1 _, . . . , gk⟩_ and _J_ 2 = _⟨h_ 1 _, . . . , hℓ⟩_ , there are three basic operations (sums, products, intersections): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0003-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0003-07.png)


_J_ 1 _∩ J_ 2 := _{r ∈ R_ : _r ∈ J_ 1 and _r ∈ J_ 2 _}._ 

Most algebraic computations involving ideals, especially “advanced” operations (e.g. primary decompositions), require computing their Gr¨obner bases as a key initial step. More generally, Gr¨obner basis computation forms the backbone of most algorithms in algebra; see Appendix A.2. **Primary decompositions** . In commutative algebra, primary decompositions of ideals are a far-reaching generalization of the idea of prime factorization for integers. Its importance to algebraists cannot be overstated. Informally, every ideal _J_ has a decomposition _J_ = _J_ 1 _∩· · · ∩ Js_ as an intersection of finitely many _primary_ ideals. This intersection is called a _primary decomposition_ of _J_ , and each _Jj_ is called a _primary component_ of the decomposition. In the special case when _J_ is a monomial ideal, there is an **unique** minimal primary decomposition with maximal monomial primary components [4]; We denote this unique set of primary components by pd( _J_ ). See Appendix A.3 for details. 

## **3.1.2 Concepts as monomial ideals** 

We define a _concept_ to be a monomial ideal of _R_ . In particular, the zero ideal _⟨_ 0 _⟩⊆ R_ is the concept “null”, and could be interpreted as “impossible” or “nothing”, while the ideal _⟨_ 1 _⟩_ = _R_ is the concept “conceivable”, and could be interpreted as “possible” or “everything”. Given a concept 

_J ⊆ R_ , a monomial in _J_ is called an _instance_ of the concept. For example, _x_ black _x_ square is an instance of _⟨x_ square _⟩_ (the concept “square”). For each _xi_ , we say _⟨xi⟩⊆ R_ is a _primitive_ concept, and _xi_ is a _primitive_ instance. 

**Theorem 3.1.** _There are infinitely many concepts in R, even though there are finitely many primitive concepts in R. Furthermore, if J ⊆ R is a concept, then the following hold:_ 

- _(i) J has infinitely many instances, unless J_ = _⟨_ 0 _⟩._ 

- _(ii) J has a unique minimal generating set consisting of finitely many instances, which we denote by_ mingen( _J_ ) _._ 

- _(iii) If J_ = _⟨_ 1 _⟩, then J has a unique set of associated concepts {P_ 1 _, . . . , Pk}, together with a unique minimal primary decomposition J_ = _J_ 1 _∩· · · ∩ Jk, such that each Ji is a concept contained in Pi, that is maximal among all possible primary components contained in Pi that are concepts._ 

See Appendix A.4 for a proof of Theorem 3.1 and for more details on why defining concepts as monomial ideals captures the expressiveness of concepts in human reasoning. 

## **3.2. Stage 1: Algebraic representation** 

We shall use the RPM instance depicted in Fig 1 as our running example, to show the entire algebraic reasoning process: (1) algebraic representation; and (2) algebraic machine reasoning. In this subsection, we focus on the first stage. Recall that every RPM instance is composed of 16 panels filled with geometric entities. For our running example, each entity can be described using 4 attributes: “color”, “size”, “type”, and “position”. We also need one additional attribute to represent the “number” of entities in the panel. 

## **3.2.1 Attribute concepts** 

In human cognition, certain semantically similar concepts are naturally grouped to form a more general concept. For example, concepts such as “red”, “green”, “blue”, “yellow”, etc., can be grouped to form a new concept that represents “color”. Intuitively, we can think of “color” as an attribute, and “red”, “green”, “blue”, “yellow” as attribute values. 

For our running example, the 5 attributes are represented by 5 concepts (monomial ideals). In general, all possible values for each attribute are encoded as generators for the concept representing that attribute. However, for ease of explanation, we shall consider only those attribute values that are involved in Fig. 1 to explain our example: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0003-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0003-24.png)


Let _L_ := _{_ num _,_ pos _,_ type _,_ color _,_ size _}_ be the set of attribute labels, and let _A_ all :=[�] _ℓ∈L[A][ℓ]_[.][Initialize][the][ring] 

3 

_R_ := R[ _A_ all] of all polynomials on the variables in _A_ all with real coefficients. For each _ℓ ∈L_ , let _Jℓ_ be the concept _⟨Aℓ⟩⊆ R_ . These concepts, which we call _attribute concepts_ , are task-specific. We assume humans tend to discover and organize complex patterns in terms of attributes. Thus for pattern extraction, we shall use the inductive bias that a concept representing a pattern is deemed meaningful if it is in some attribute concept. 

## **3.2.2 Representation of RPM panels** 

In order to encode the RPM images algebraically, we first need to train perception modules to extract attribute information directly from raw images. One possible approach for perception, as used in our experiments, is to train 4 RetinaNet models (each with a ResNet-50 backbone) separately for all 4 attributes except “number”, which can be directly inferred by counting the number of bounding boxes. 

After extracting attribute values for entities, we can represent each panel as a concept. For example, the top-left panel of the RPM in Fig. 1 can be encoded as the concept 

_J_ 1 _,_ 1 = _⟨x_ two _x_ left _x_ square _x_ black _x_ avg _, x_ two _x_ right _x_ triangle _x_ gray _x_ avg _⟩_ 

in the polynomial ring _R_ . Here, _J_ 1 _,_ 1 represents a panel with two entities, a black square of average size on the left, and a gray triangle of average size on the right. The indices in _J_ 1 _,_ 1 tell us that the panel is in row 1, column 1. Similarly, we can encode the remaining 7 panels of the question matrix as concepts _J_ 1 _,_ 2 _, J_ 1 _,_ 3 _, . . . , J_ 3 _,_ 2 and encode the 8 answer options as concepts _J_ ans1 _, . . . , J_ ans8. In general, every monomial generator of each concept describes an entity in the associated panel. 

The list of 8 concepts **J** = [ _J_ 1 _,_ 1 _, . . . , J_ 3 _,_ 2] shall be called a _concept matrix_ ; this represents the RPM question matrix with a missing 9-th panel. Let **J** _i_ := [ _Ji,_ 1 _, Ji,_ 2 _, Ji,_ 3] (for _i_ = 1 _,_ 2) represent the _i_ -th row in the question matrix. 

## **3.3. Stage 2: algebraic machine reasoning** 

Previously in Section 3.2, we have already encoded the question matrix in an RPM instance as a concept matrix **J** = [ _J_ 1 _,_ 1 _, . . . , J_ 3 _,_ 2]. In this subsection, we will introduce the reasoning process of our algebraic framework. 

Our goal of extracting patterns for a single row of **J** can be mathematically formulated as “finding invariance” across the concepts that represent the panels in this row. (The same process can be applied to columns.) This seemingly imprecise idea of “finding invariance” can be realized very concretely via the computation of primary decompositions. Ideally, we want to extract patterns that are meaningful to humans. Hence we have designed 4 invariance modules to mimic human cognition in pattern recognition. 

## **3.3.1 Prior knowledge** 

To use algebraic machine reasoning, we adopt: 

- Inductive bias of attribute concepts (see Section 3.2.1); 

- Useful binary operations on numerical values; 

- Functions that map concepts to concepts. 

There are numerous binary operations, such as + _, −, ×_ , _÷,_ min _,_ max, etc., that can be applied to numerical values extracted from concepts. For the RPM task, we use + _, −_ . 

In algebra, the study of functions between algebraic objects is a productive strategy for understanding the underlying algebraic structure. Analogously, we shall use maps on concepts to extract complex patterns. For the RPM task, we need to cyclically order the values in _Aℓ_ for each attribute _ℓ ∈L_ before we can extract sequential information. To encode the idea of “next”, we introduce the function _f_ next( _J|_ ∆) defined on concepts _J_ , where ∆ represents the step-size. Each variable _x ∈Aℓ_ that appears in a generator of _J_ is mapped to the ∆-th variable after _x_ , w.r.t. the cyclic order on _Aℓ_ . For example, _f_ next( _⟨x_ square _x_ gray _x_ avg _⟩|_ 1) = _⟨x_ pentagon _x_ dgray _x_ large _⟩_ , and _f_ next( _⟨x_ square _⟩| −_ 2) = _⟨x_ circle _⟩_ . 

## **3.3.2 Reasoning via primary decompositions** 

Given concepts _J_ 1 _, . . . , Jk_ that share a common “pattern”, how do we extract this pattern? Abstractly, a common pattern can be treated as a concept _K_ that contains all of these concepts _J_ 1 _, . . . , Jk_ . If there are several common patterns _K_ 1 _, . . . , Kr_ , then each concept _Ji_ can be “decomposed” as _Ji_ = _K_ 1 _∩· · ·∩ Kr ∩ Ki[′]_[for some ideal] _[ K] i[′]_[.][Thus, we have] the following algebraic problem: Given _J_ 1 _, . . . , Jk_ , compute their common components _K_ 1 _, . . . , Kr_ . 

Recall that a concept _J_ has a unique minimal primary decomposition, since concepts are monomial ideals. Thus, to extract the common patterns of concepts _J_ 1 _, . . . , Jk_ , we first have to compute pd( _J_ 1) _, . . . ,_ pd( _Jk_ ), then extract the common primary components. The intersection of (any subset of) these common components would yield a new concept, which can be interpreted as a common pattern of the concepts _J_ 1 _, . . . , Jk_ . As part of our inductive bias, we are only interested in those primary components that are contained in attribute concepts. See Appendix A.3 for further details. 

## **3.3.3 Proposed invariance modules** 

Our 4 proposed invariance modules are: (1) intra-invariance module, (2) inter-invariance module, (3) compositional invariance module, and (4) binary-operator invariance module. Intuitively, they check for 4 general types of invariances across a sequence of concepts _J_ 1 _, . . . , Jk_ (e.g. a row **J** _i_ = [ _Ji,_ 1 _, Ji,_ 2 _, Ji,_ 3] for the RPM task). Such invariances apply not just to the RPM task, but could be applied to other RPM-like tasks, e.g. based on different prior knowledge, different grid layouts, etc. Full computational details for our running example can be found in Appendix B.3. 

**1. Intra-invariance module** extracts patterns where the set of values for some attribute within concept _Ji_ remains invariant over all _i_ . First, we define _J_ + := _J_ 1 + _· · ·_ + _Jk_ and _J∩_ := _J_ 1 _∩· · · ∩ Jk_ ; see Section 3.1.1. Intuitively, _J_ + and 

4 

_J∩_ are concepts that capture information about the entire sequence _J_ 1 _, . . . , Jk_ in two different ways. Next, we compute the common primary components of _J_ + and _J∩_ that are contained in attribute concepts. Finally, we return the attributes associated to these common primary components: _P_ intra([ _J_ 1 _. . . Jk_ ]) := �attr _∈L | ∃I ∈_ pd( _J_ +) _∩_ pd( _J∩_ ) _, I ⊆⟨A_ attr _⟩_ �. **2. Inter-invariance module** extracts patterns arising from the set difference between pd( _J∩_ ) and pd( _J_ +). Thereafter, we check for the invariance of these extracted patterns across multiple sequences. The extracted set of patterns is: 

_I ⊆_ pd( _J∩_ ) _−_ pd( _J_ +) _, P_ inter([ _J_ 1 _, . . . , Jk_ ]) := (attr _, I_ ) _,_ attr _∈L, I ⊆⟨A_ attr _⟩∀I ∈I_ � ����� � where _I_ is a set of concepts, and “ _−_ ” refers to set difference. We omit pd( _J_ +) so that we do not overcount the patterns already extracted in the previous module. Informally, for each pair (attr _, I_ ), the concepts in _I_ can be interpreted as those “primary” concepts that correspond to at least one of _J_ 1 _, . . . , Jk_ , that do not correspond to all of _J_ 1 _, . . . , Jk_ , and that are contained in _⟨A_ attr _⟩_ . 

**3. Compositional invariance module** extracts patterns arising from invariant attribute values in the following new sequence of concepts: 

[ _J_ 1 _[′][, . . . , J] k[′]_[] = [] _[f][ k][−]_[1][(] _[J]_[1][)] _[, f][ k][−]_[2][(] _[J]_[2][)] _[, . . . , f]_[(] _[J][k][−]_[1][)] _[, J][k]_[]] _[,]_ where _f_ is some given function. Intuitively, for such patterns, there are some attributes whose values are invariant in [ _f_ ( _Ji_ ) _, Ji_ +1] for all _i_ = 1 _, . . . , k −_ 1. By checking the intersection of primary components of the concepts in the new sequence, the extracted set of patterns is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0005-04.png)


## **3.3.4 Extracting row-wise patterns** 

Given a concept matrix **J** = [ _J_ 1 _,_ 1 _, . . . , J_ 3 _,_ 2], how do we extract the patterns from its _i_ -th row? We first begin by extracting the common position values among all 8 panels: 

comPos( **J** ) := � _p ∈A_ pos _| ∃I ∈_[�] _J∈_ **J**[pd][(] _[J]_[)] _[, p][ ∈][I]_ � For each common position _p ∈_ comPos( **J** ), we generate two new concept matrices **J**[¯][(] _[p]_[)] and **J**[ˆ][(] _[p]_[)] , such that: 

   - Each concept _J_[¯] _i,j_[(] _[p]_[)][in] **[J]**[¯][(] _[p]_[)][is][generated][by][the][unique] generator in _Ji,j_ that is divisible by _p_ ; 

- Each concept _J_[ˆ] _i,j_[(] _[p]_[)][in] **[J]**[ˆ][(] _[p]_[)][is][generated][by][all][genera-] tors in _Ji,j_ that are not divisible by _p_ . 

- (Recall that generators of a concept are polynomials.) 

Informally, we are splitting each panel in the RPM image into 2 panels, one that contains only the entity in the 

common position _p_ , and the other that contains all remaining entities not in position _p_ . This step allows us to reason about rules that involve only a portion of the panels. 

Consequently, if comPos( **J** ) = _{p_ 1 _, . . . , pk}_ , then we can extend the single concept matrix into a list of concept matrices [ **J** _,_ **J**[¯][(] _[p]_[1][)] _,_ **J**[ˆ][(] _[p]_[1][)] _, . . . ,_ **J**[¯][(] _[p][k]_[)] _,_ **J**[ˆ][(] _[p][k]_[)] ]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0005-13.png)


whereextendedthelist,unioni.e.ranges **J** ˇ _∈_ over[ **J** _,_ ¯ **J** all[(] _[p]_ 1concept[)] _,_ ˆ **J**[(] _[p]_ 1[)] _, . . . ,_ matrices ¯ **J**[(] _[p] k_[)] **J**[ˇ] _,_ ˆ **J** in[(] _[p] k_ the[)] ]. Note that _Pi_[(all)] ( **J** ) can be regarded as all the patterns extracted from the _i_ -th row of the original concept matrix **J** . If instead **J** = [ _J_ 1 _,_ 1 _, . . . , J_ 3 _,_ 3] is a list containing 9 concepts, then we can define _P_ 3[(all)] ( **J** ) analogously. 

## **Algorithm 1** Answer selection. 

**Inputs:** Concept matrix **J** = [ _J_ 1 _,_ 1 _. . . J_ 3 _,_ 2], and associated answer set [ _J_ ans1 _, . . . , J_ ans8]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0005-17.png)


## **3.4. Solving RPMs** 

## **3.4.1 Answer selection** 

In Section 3.3.4, we described how row-wise patterns can be extracted using the 4 invariance modules. Thus, a natural approach for answer selection is to determine which answer option, when inserted in place of the missing panel, would maximize the number of patterns that are common to all three rows. Consequently, answer selection is reduced to a simple optimization problem; see Algorithm 1. 

## **3.4.2 Answer generation** 

Since our algebraic machine reasoning framework is able to extract common patterns that are meaningful to humans, hidden in the raw RPM images, it provides a new way to generate answers without needing a given answer set. This is similar to a gifted human who is able to solve the RPM task, by first recognizing the patterns in the first two rows, then inferring what the missing panel should be. Intuitively, we are applying “inverse” operations of the 4 invariance modules to generate the concept representing the missing panel; see Algorithm 2 for an overview. 

5 

Briefly speaking, for a given RPM concept matrix **J** , we first compute the common patterns among the first two rows via _P_ 1 _,_ 2( **J** ) := _P_ 1[(all)] ( **J** ) _∩P_ 2[(all)] ( **J** ); see (1). Each element in _P_ 1 _,_ 2( **J** ) is a pair ( _K,_ **J**[ˇ] ), where _K_ is a common pattern (for rows 1 and 2) specific to one attribute, and **J**[ˇ] is the corresponding concept matrix. (This represents the _difficult_ step of pattern discovery by a gifted human.) Then, we go through all common patterns to compute the attribute values for the missing 9th panel. (This represents a _routine_ consistency check of the discovered patterns; see Appendix B.2 for full algorithmic details, and B.3 for an example.) 

In general, when integrating all the attribute values for _J_ 3 _,_ 3 derived from the patterns in _P_ 1 _,_ 2( **J** ), it is possible that entities (i) have multiple possible values for a single attribute; or (ii) have missing attribute values. Case (i) occurs when there are multiple patterns extracted for a single attribute, while case (ii) occurs when there are no nonconflicting patterns for this attribute. For either case, we randomly select an attribute value from the possible values. 

|**Algorithm 2**Answergeneration.|**Algorithm 2**Answergeneration.||
|---|---|---|
|**Inputs:** Concept matrix**J**= [_J_1_,_1_. . . J_3_,_2].|||
|1:|**for**(_K,_ ˇ**J**)_∈P_(all)<br>1<br>(**J**)_∩P_(all)<br>2<br>(**J**)**do**|//_see_(1)|
|2:|**if** [ˇ_J_3_,_1_,_ ˇ_J_3_,_2]does not confict with pattern|_K_ **then**|
|3:|Compute attribute value for ˇ_J_3_,_3using pattern_K_.||
|4:|Collect all the above attribute values for_J_3_,_3.||
|5:|**while**∄unique value for some attribute of an entity**do**||
|6:|Randomly choose one valid attribute value.||
|7:|Generate ideal_J_3_,_3 _⊆R_.||
|8:|**return**_J_3_,_3and the corresponding image.||



## **4. Discussion** 

Algebraic machine reasoning provides a fundamentally new paradigm for machine reasoning beyond numerical computation. Abstract notions in reasoning tasks are encoded very concretely as ideals, which are computable algebraic objects. We treat ideals as “actual objects of study”, and we do not require numerical values to be assigned to them. This means our framework is capable of reasoning on more qualitative or abstract notions that do not naturally have associated numerical values. Novel problem-solving, such as the discovery of new abstract patterns from observations, is realized concretely as computations on ideals (e.g. computing the primary decompositions of ideals). In particular, we are _not_ solving a system of polynomial equations, in contrast to existing applications of algebra in AI (cf. Section 2). Variables (or primitive instances) are _not_ assigned values. We do _not_ evaluate polynomials at input values. 

Theory-wise, our proposed approach breaks new ground. We established a new connection between machine reasoning and commutative algebra, two areas that were completely unrelated previously. There is over a century’s worth 

of very deep results in commutative algebra that have not been tapped. Could algebraic methods be the key to tackling the long-standing fundamental questions in machine reasoning? It was only much more recently in 2014 that L´eon Bottou [6] suggested that humans should “build reasoning capabilities from the ground up”, and he speculated that the missing ingredient could be an algebraic approach. 

Why use ideals to represent concepts? Why not use sets? Why not use symbolic expressions, e.g. polynomials? Intuitively, we think of a concept as an “umbrella term” consisting of multiple (potentially infinitely many) instances of the concept. Treating concepts as merely sets of instances is inadequate in capturing the expressiveness of human reasoning. A set-theoretic representation system with finitely many “primitive sets” can only have finitely many possible sets in total. In contrast, we proved that we can construct infinitely many concepts from only _finitely many_ primitive concepts (Theorem 3.1). This agrees with our intuition that humans are able to express infinitely many concepts from only finitely many primitive concepts. The main reason is that the “richer” algebraic structure of ideals allows for significantly more operations on ideals, beyond set-theoretic operations. See Appendix A.4 for further discussion. 

Why is our algebraic method fundamentally different from logic-based methods, e.g. those based on logic programming? At the heart of logic-based reasoning is the idea that reasoning can be realized concretely as the resolution (or inverse resolution) of logical expressions. Inherent in this idea is the notion of _satisfiability_ ; cf. [28]. Intuitively, we have a logical expression, usually expressed in a canonical normal form, and we want to assign truth values (true or false) to literals in the logical expression, so that the entire expression is satisfied (i.e. truth value is “true”); see Appendix C.1 for more discussion. In fact, much of the exciting progress in automated theorem proving [2, 27, 36, 39, 81, 92] is based on logic-based reasoning. 

In contrast, algebraic machine reasoning builds upon computational algebra and computer algebra systems. At the heart of our algebraic approach is the idea that reasoning can be realized concretely as solving computational problems in algebra. Crucially, there is no notion of satisfiability. We do not assign truth values (or numerical values) to concepts in _R_ = k[ _x_ 1 _, . . . , xn_ ]. In particular, although primitive concepts _⟨x_ 1 _⟩, . . . , ⟨xn⟩_ in _R_ correspond to the variables _x_ 1 _, . . . , xn_ , we do not assign values to primitive concepts. Instead, ideals are treated as the “actual objects of study”, and we reduce “solving a reasoning task” to “solving non-numerical computational problems involving ideals”. Moreoever, our framework can discover new patterns beyond the actual rules of the RPM task; see Section 5.2. 

In the RPM task, we have attribute concepts representing “position”, “number”, “type”, “size”, and “color”; these are concepts that categorize the primitive instances according 

6 

||Method|Avg. Acc.|Center|2_×_2G|3_×_3G|O-IC|O-IG|L-R|U-D|
|---|---|---|---|---|---|---|---|---|---|
|1|LSTM [83]|18.9 / 13.1|26.2 / 13.2|16.7 / 14.1|15.1 / 13.7|21.9 / 12.2|21.1 / 13.0|14.6 / 12.8|16.5 / 12.4|
|2|WReN [62]|23.8 / 34.0|29.4 / 58.4|26.8 / 38.9|23.5 / 37.7|22.5 / 38.8|21.5 / 22.6|21.9 / 21.6|21.4 / 19.7|
|3|ResNet [83]|40.3 / 53.4|44.7 / 52.8|29.3 / 41.9|27.9 / 44.3|46.2 / 63.2|35.8 / 53.1|51.2 / 58.8|47.4 / 60.2|
|4|ResNet+DRT [83]|40.4 / 59.6|46.5 / 58.1|28.8 / 46.5|27.3 / 50.4|46.0 / 69.1|34.2 / 60.1|50.1 / 65.8|49.8 / 67.1|
|5|LEN [88]|41.4 / 72.9|56.4 / 80.2|31.7 / 57.5|29.7 / 62.1|52.1 / 84.4|31.7 / 71.5|44.2 / 73.5|44.2 / 81.2|
|6|CoPINet [84]|46.1 / 91.4|54.4 / 95.1|36.8 / 77.5|31.9 / 78.9|52.2 / 98.5|42.8 / 91.4|51.9 / 99.1|52.5 / 99.7|
|7|DCNet [91]|49.4 /**93.6**|57.8 / 97.8|34.1 / 81.7|35.5 / 86.7|57.0 /**99.0**|42.9 /**91.5**|58.5 /**99.8**|60.0 /**99.8**|
|8|NCD [89]|48.2 / 37.0|60.0 / 45.5|31.2 / 35.5|30.0 / 39.5|62.4 / 40.3|39.0 / 30.0|58.9 / 34.9|57.2 / 33.4|
|9|SRAN [26]|60.8 / -|78.2 / -|50.1 / -|42.4 / -|68.2 / -|46.3 / -|70.1 / -|70.3 / -|
|10|PrAE [85]|77.0 / 65.0|90.5 / 76.5|85.4 / 78.6|45.6 / 28.6|63.5 / 48.1|60.7 / 42.6|96.3 / 90.1|97.4 / 90.9|
|11|Our Method|**93.2**/ 92.9|**99.5**/**98.8**|**89.6**/**91.9**|**89.7**/**93.1**|**99.6**/ 98.2|**74.7**/ 70.1|**99.7**/ 99.2|**99.5**/ 99.1|
||Human [83]|- / 84.4|- / 95.5|- / 81.8|- / 79.6|- / 86.4|- / 81.8|- / 86.4|- / 81.8|



Table 1. Performance on I-RAVEN/RAVEN. We report mean accuracy, and the accuracies for all configurations: Center, 2x2Grid, 3x3Grid, Out-InCenter, Out-InGrid, Left-Right, and Up-Down. 

to their semantics, into what humans would call attributes. Intuitively, an attribute concept combines certain primitive concepts together in a manner that is “meaningful” to the task. For example, _⟨x_ white _, x_ gray _, x_ black _⟩_ is “more meaningful” than _⟨x_ white _, x_ circle _, x_ large _⟩_ as a “simpler” or “generalized” concept, since we would treat _x_ white _, x_ gray _, x_ black as instances of a single broader “color” concept. 

Notice that the primitive concepts correspond precisely to the prediction classes of our object detection models. Such prediction classes are already implicitly identified by the available data. Consequently, our method is limited by what our perception modules can perceive. For other tasks, e.g. where text data is available, entity extraction methods can be used to identify primitive concepts. Note also that our method requires prior knowledge, since there is no training step for the reasoning module. This limitation can be mitigated if we replace user-defined functions on concepts with trainable functions optimized via deep learning. 

In general, the identification of attribute concepts is taskspecific, and the resulting reasoning performance would depend heavily on these identified attribute concepts. Effectively, our choice of attribute concepts would determine the _inductive bias_ of our reasoning framework: As we decompose a concept _J_ into “simpler” concepts (i.e. primary components in pd( _J_ )), only those “simpler” concepts contained in attribute concepts are deemed “meaningful”. Concretely, let _J, J[′]_ ⊊ _R_ be concepts such that pd( _J_ ) = _{J_ 1 _, . . . , Jk}_ and pd( _J[′]_ ) = _{J_ 1 _[′][, . . . , J] ℓ[′][}]_[, i.e.] _[ J, J][′]_[ have minimal primary] decompositions _J_ = _J_ 1 _∩· · ·∩Jk_ and _J[′]_ = _J_ 1 _[′][∩· · ·∩][J] ℓ[′]_[, re-] spectively. We can examine their primary components and extract out those primary components (between the two primary decompositions) that are contained in some common attribute concept. For example, if _A_ is an attribute concept of _R_ such that _J_ 1 _⊆ A_ and _J_ 1 _[′][⊆][A]_[, then] _[ J]_[and] _[ J][′]_[share a] “common pattern”, represented by the attribute concept _A_ . 

## **5. Experiment results** 

To show the effectiveness of our framework, we conducted experiments on the RAVEN [83] and I-RAVEN datasets. In both datasets, RPMs are generated according to 7 configurations. We trained our perception modules on 4200 images from I-RAVEN [26] (600 from each configuration), and used them to predict attribute values of entities. The average accuracy of our perception modules is 96 _._ 24%. For both datasets, we tested on 2000 instances for each configuration. Overall, our reasoning framework is fast (7 hours for 14000 instances on a 16-core Gen11 Intel i7 CPU processor). See Appendix B for full experiment details. 

## **5.1. Comparison with other baselines** 

Table 1 compares the performance of our method with 10 other baseline methods. We use the accuracies on I- RAVEN reported in [26, 89] for methods 1-7, and the accuracies on RAVEN reported in [83, 89] for methods 1-5. All the other accuracies are obtained from the original papers. As a reference, we also include the human performance on the RAVEN dataset (i.e. _not_ I-RAVEN) as reported in [83]. 

## **5.2. Ambiguous instances and new patterns** 

Although our method outperforms all baselines, some instances have multiple answer options that are assigned equal top scores by our framework. Most of these cases occur due to the discovery of (i) “accidental” unintended rules (e.g. Fig. 3); or (ii) new patterns beyond the actual rules in the dataset (e.g. Fig. 4). Case (i) occurs because in the design of I-RAVEN, at most one rule is assigned to each attribute. 

Interestingly, case (ii) reveals that our framework is able to discover completely new patterns that are not originally designed as rules for I-RAVEN. In Fig. 4, the new pattern discovered is arguably very natural to humans. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0008-00.png)


Figure 3. An example of an ambiguous RPM instance. The given answer is option **g** . For I-RAVEN, the type sequence (“circle”, “hexagon”, “pentagon”) in the first two rows follows a Progression rule with consecutively decreasing type indices, so **g** could be a correct answer. (Remaining attribute values are determined by other patterns.) However, our framework assigns equal top scores to both options **d** and **g** , as a result of another inter-invariance pattern for type (the type set _{_ “circle”, “hexagon”, “pentagon” _}_ is invariant across the rows). Thus, option **d** could also be correct. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0008-02.png)


Figure 4. An example of an RPM instance with an unexpected new pattern. The given answer is option **h** . In each row, the number of entities in the first 2 panels sum up to the number of entities in the 3rd panel, so **h** could be correct. However, our framework assigns equal top scores to both options **b** and **h** , as a result of a new interinvariance pattern for number (informally, every panel has either 1 or 2 entities). Thus option **b** could also be correct. 

## **5.3. Evaluation of answer generation** 

Every RPM instance is assumed to have a single correct answer from the given answer set. However, there are multiple other possible images that are also acceptable as correct answers. For example, images modified from the given correct answer, via random perturbations of those attributes that are not involved in any of the rules (e.g. entity angles in the I-RAVEN dataset), are also correct. All these distinct correct answers (images) can be encoded algebraically as the same concept, based on prior knowledge of which raw perceptual attributes are relevant for the RPM task. Hence, to evaluate the answer generation process proposed in Section 3.4.2, we will directly evaluate the generated concepts. Let _J_ = _⟨e_ 1 _, . . . , ek⟩_ and _J[′]_ = _⟨e[′]_ 1 _[, . . . , e][′] ℓ[⟩]_[be concepts] 

representing the ground truth answer and our generated answer, respectively. Here, each _ei_ (or _e[′] i_[)][is][a][monomial][of] the form _x_[(] _i_[pos][)] _x_[(] _i_[type][)] _x_[(] _i_[color][)] _x_[(] _i_[size][)] , and represents an entity described by 4 attributes. Motivated by the well-known idea of Intersection over Union (IoU), we propose a new similarity measure between _J_ and _J[′]_ . In order to define analogous notions of “intersection” and “union”, we first pair _ei_ with _e[′] j_[if] _[x]_[(] _i_[pos][)] = _x[′] j_ (pos) (i.e. same “position” values). This pairing is well-defined, since the “position” values of the entities in any panel are uniquely determined. Hence we can group all entities in _J_ and _J[′]_ into 3 sets: _S_ 1 := _{_ ( _ei, e[′] j_[)] _[ |][ e][i][∈][J, e][′] j[∈][J][′][, x]_[(] _i_[pos][)] = _x[′] j_ (pos) _}_ ; _S_ 2 := _{ei ∈ J |_ ∄ _e[′] j[∈][J][′]_[such that][ (] _[e][i][, e][′] j_[)] _[ ∈][S]_[1] _[}]_[;] _S_ 3 := _{e[′] j[∈][J][′][|]_[ ∄] _[e][i][∈][J]_[such that][ (] _[e][i][, e][′] j_[)] _[ ∈][S]_[1] _[}][.]_ We can interpret _S_ 1 and _S_ 1 _∪ S_ 2 _∪ S_ 3 as analogous notions of the “intersection” and “union” of _J_ and _J[′]_ , respectively. Thus, we define our similarity measure as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0008-07.png)


where in (3), _a_ ranges over the 4 attributes in _{_ pos, type, color, size _}_ . Here, _φ_ ( _ei, e[′] j_[)][ is the similarity score between] _ei_ and _e[′] j_[, measured by the proportion of common variables.] 

The overall average similarity score of the generated answers is 67 _._ 7%. Note that within a panel, some attribute values such as “size”, “color” and “position”, may be totally random for 2x2Grid, 3x3Grid, Out-InGrid (e.g. as shown in Fig. 3). Hence, achieving high similarity scores for such cases would inherently require task-specific optimization and knowledge of how the data is generated. We assume neither. This could explain why our overall similarity score is lower than our answer selection accuracy. 

For examples of generated images, see Appendix B.5. 

## **6. Conclusion** 

Algebraic machine reasoning is a reasoning framework that is well-suited for abstract reasoning. In its current form, we have used primary decompositions as a key algebraic operation to discover abstract patterns in the RPM task, via the invariance modules that we have specially designed to mimic human reasoning. The idea that “discovering common patterns” can be realized concretely as “computing primary decompositions” is rather broad, and could potentially be applied to other inferential reasoning tasks. 

More generally, our algebraic approach opens up new possibilities of tapping into the vast literature of commutative algebra and computational algebra. There are numerous algebraic operations on ideals (ideal quotients, radicals, saturation, etc.) and algebraic invariants (depth, height, etc.) that have not been explored in machine reasoning (or even in AI). Can we use them to tackle other reasoning tasks? 

8 

## **Appendices for Abstract Visual Reasoning: An Algebraic Approach for Solving Raven’s Progressive Matrices** 

**Outline.** As part of supplementary materials for this paper, we provide further details, organized into the following three appendices: 

- Appendix A gives further algebraic details on algebraic machine reasoning. 

- Appendix B gives further experiment details for how we solved RPMs. 

- Appendix C gives further discussion on algebraic machine reasoning, including its potential societal impact and its differences from logic-based reasoning. 

## **A. Further Algebraic Details** 

This appendix serves as an in-depth elaboration of the algebraic ideas presented in Section 3. For definitions and relevant terminology, we have kept this appendix as selfcontained as possible. For proofs of “standard” algebraic results, we provide references to explicit proofs in textbooks wherever possible, and provide full proof details otherwise. For ease of reading, it would be helpful if the reader has at least some prior exposure to the basics of commutative algebra, e.g. at the level of undergraduate abstract algebra. For a gentle introduction to commutative algebra and computational algebra, we recommend [12]. For a more detailed treatment of the subject, see [17]. 

Throughout the appendix, let _R_ := k[ _x_ 1 _, . . . , xn_ ] be a polynomial ring on _n_ variables _x_ 1 _, . . . , xn_ , with coefficients in a field k. In this paper, we assumed for simplicity that k = R; in fact, k can be chosen to be any field. In particular, k need not be an algebraically closed field.[1] (Note that C is an algebraically closed field, while R is not.) This means that we allow k = R. We also allow k to be a field with nonzero characteristic (e.g. a finite field Z _/p_ Z for some prime number _p_ ). The specific choice of field k will be irrelevant for the details discussed in this appendix, since none of the algebraic computations required for reasoning involve numerical computations on the coefficients of the terms of polynomials. For concreteness, the reader may choose to assume henceforth that k = R, without any loss of generality. 

- 1When dealing with a system of polynomial equations in _n_ variables, 

- where the coefficients of the polynomial equations are treated as values in k, any solution to this system of polynomial equations is a point in k _[n]_ . It is frequently assumed that k is algebraically closed, so that the analysis of the solution space is easier. Similarly, when considering the algebraic variety associated to such a system of polynomial equations, the analysis of the algebraic variety is easier when the base field k is algebraically closed. In our paper, we do not deal with or solve systems of polynomial equations, and we will not work with algebraic varieties. Hence, we will not assume that k is algebraically closed. 

For our algebraic notation, we use “ _⟨g_ 1 _, . . . , gk⟩_ ” to denote an ideal generated by _g_ 1 _, . . . , gk_ , in contrast to some other authors who use the notation “( _g_ 1 _, . . . , gk_ )” instead to refer to ideals. For us, we reserve the use of round brackets to denote tuples, e.g. “( _g_ 1 _, . . . , gk_ )” is a _k_ -tuple. All other algebraic notation we use is consistent with the present-day notation used in the commutative algebra literature. 

- The rest of Appendix A is organized as follows: 

- Appendix A.1 gives further details and relevant results concerning concepts (i.e. monomial ideals), including algebraic operations on ideals, and with a focus on the special case of monomial ideals. 

- Appendix A.2 gives a detailed treatment of Gr¨obner basis theory, the key “workhorse” underlying most of the algorithms in computational algebra and commutative algebra. 

- Appendix A.3 gives more details on primary ideals and primary decompositions, as well as a discussion on _how primary decompositions relate to the inductive bias for algebraic machine reasoning_ . 

- Appendix A.4 completes the proof of Theorem 3.1 from Section 3.1.2, and gives details on _why defining concepts as monomial ideals captures the expressiveness of concepts in human reasoning._ 

## **A.1. Concepts, monomial ideals, ideal operations** 

Let N be the set of non-negative integers. For each _**α**_ = ( _α_ 1 _, . . . , αn_ ) _∈_ N _[n]_ , let **x** _**[α]**_ denote the monomial _x[α]_ 1[1] _[x][α]_ 2[2] _[. . . x] n[α][n]_ in _R_ . By definition, a _monomial_ in _R_ is a polynomial of the form **x** _**[α]**_ , i.e. the leading coefficient of a monomial is always 1, the multiplicative identity of the field k. The _degree_ of **x** _**[α]**_ , which we denote by deg( **x** _**[α]**_ ), is the sum _α_ 1 + _· · ·_ + _αn_ . For example, deg( _x_[2] 1 _[x]_[3] 2 _[x]_[7] 3[) = 12][.] 

## **A.1.1 Algebraic properties of concepts** 

Let _MR_ be the set of all monomials in _R_ . We begin with the observation that the polynomial ring _R_ is a k-vector space with basis _MR_ . This means that every polynomial _p_ in _R_ can be written as _p_ =[�] _m∈MR[a][m][ ·][ m]_[, where each] _[ a][m]_[is] a uniquely determined scalar in k, called the _coefficient_ of the monomial _m_ in _p_ . A monomial _m ∈MR_ is called a _monomial of p_ if its coefficient _am_ is non-zero. 

**Proposition A.1.** _[12, Chap. 2.4, Lemma 3] Let p be a polynomial of R, and let J be a monomial ideal of R. Then p is an element of J if and only if every monomial m of p is an element of J._ 

Although _MR_ is infinite (since it contains monomials of all degrees), it is a well-known algebraic fact that every 

9 

ideal _I_ of _R_ has a _finite_ generating set; this is famously known as Hilbert’s basis theorem (see [12, Chap. 2.5, Thm. 4] or [17, Thm. 1.2] for a proof). For monomial ideals, we have the following stronger result: 

**Proposition A.2.** _[48, Lemma 1.2] Every monomial ideal of R has a_ _**unique** minimal generating set consisting of finitely many monomial generators._ 

To make sense of Proposition A.2, note that a generating set _G_ for an ideal _J_ of _R_ is called _minimal_ if no strictly smaller subset _G[′]_ ⊊ _G_ generates _J_ . This means if we are given a monomial ideal _J_ of _R_ , and any subset _M ⊆MR_ of monomials that generates _J_ , i.e. _J_ = _⟨M⟩_ , then Proposition A.2 tells us that we can always find a minimal subset _M[′] ⊆M_ (with respect to set inclusion) that generates the same ideal _J_ , such that this minimal subset _M[′]_ is uniquely determined, independent of our choice of _M_ . This unique minimal generating set of monomials shall be denoted by mingen( _J_ ). 

Given any monomial ideal _J_ of _R_ , a monomial contained in mingen( _J_ ) is called a _minimal monomial generator_ of _J_ . Since a concept is defined to be a monomial ideal of _R_ , we can analogously define a _minimal instance_ of _J_ to be a minimal monomial generator of _J_ . Thus, a concept is generated by all of its minimal instances. 

Recall that there are three basic operations on ideals: sums, products, and intersections. Let _J_ 1 = _⟨g_ 1 _, . . . , gk⟩_ and _J_ 2 = _⟨h_ 1 _, . . . , hℓ⟩_ be any two ideals of _R_ . Then the sum _J_ 1 + _J_ 2, product _J_ 1 _J_ 2 and intersection _J_ 1 _∩ J_ 2 are 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0010-05.png)


In particular, _J_ 1 _∩ J_ 2 is an ideal, since for any polynomials _r_ 1 _, . . . , rs_ in _J_ 1 _∩ J_ 2, the polynomial combination _r_ 1 _p_ 1 + _· · ·_ + _rsqs_ (for any polynomials _p_ 1 _, . . . , ps_ in _R_ ) is by definition contained in both _J_ 1 and _J_ 2. 

A useful property of these basis operations on ideals is that monomial ideals remain as monomial ideals under such operations: 

**Proposition A.3.** _If J_ 1 _and J_ 2 _are concepts of R, then J_ 1 + _J_ 2 _, J_ 1 _∩ J_ 2 _and J_ 1 _J_ 2 _are also concepts of R. Furthermore, we always have J_ 1 _J_ 2 _⊆ J_ 1 _∩ J_ 2 _._ 

_Proof._ Clearly, _J_ 1+ _J_ 2 and _J_ 1 _J_ 2 are monomial ideals, since the generators specified in their definitions are monomials. Also, for each 1 _≤ i ≤ k_ , 1 _≤ j ≤ ℓ_ , the element _gihj_ is by definition contained in both _J_ 1 and _J_ 2, which means that every minimal monomial generator of _J_ 1 _J_ 2 is an element of _J_ 1 _∩ J_ 2, thus we get _J_ 1 _J_ 2 _⊆ J_ 1 _∩ J_ 2. 

Finally, we shall prove that _J_ 1 _∩ J_ 2 is a monomial ideal. Let _M[′]_ := _MR ∩ J_ 1 _∩ J_ 2, and define the monomial ideal _J[′]_ := _⟨M[′] ⟩⊆ R_ . By definition, _M[′] ⊆ J_ 1 _∩ J_ 2, hence _J[′] ⊆ J_ 1 _∩ J_ 2. To show that the converse _J[′] ⊇ J_ 1 _∩ J_ 2 holds, consider an arbitrary element _p ∈ J_ 1 _∩ J_ 2, and write _p_ =[�] _m∈MR[a][m][ ·][ m]_[, where each] _[ a][m][∈]_[k][ is the coefficient] of the monomial _m_ . Note that _p ∈ J_ 1 and _p ∈ J_ 2. Since _J_ 1 and _J_ 2 are both monomial ideals, Proposition A.1 implies that every monomial _m_ of _p_ (i.e. with coefficient _am_ = 0) is an element of _J_ 1 _∩ J_ 2, and so must also be contained in _M[′]_ = _M ∩ J_ 1 _∩ J_ 2. Thus, _p ∈⟨M[′] ⟩_ = _J[′]_ . 

## **A.1.2 A hierarchy of different kinds of concepts** 

Recall that the first stage of algebraic machine reasoning is algebraic representation. For the RPM task, we extracted attribute information from raw RPM images, and encoded each RPM panel algebraically as a concept. These concepts that we have used for our algebraic representation of RPM panels are particularly nice: Their minimal monomial generators are squarefree. Recall that a monomial **x** _**[α]**_ in _R_ (for some _**α**_ = ( _α_ 1 _, . . . , αn_ ) _∈_ N _[n]_ ) is called _squarefree_ if every exponent _αi_ is either 0 or 1, or equivalently, if **x** _**[α]** ̸∈⟨x_[2] 1 _[, . . . , x]_[2] _n[⟩]_[.][(Informally,][a monomial] _[ m]_ is squarefree if the variables appearing in _m_ do not have “higher powers”, i.e. none of the variables have exponents _≥_ 2.) Monomial ideals generated by squarefree monomials are called _squarefree monomial ideals_ , and such ideals are heavily studied in commutative algebra for their rich properties [48, 66]. Hence, we are motivated to define the following notion of “basic” concepts. 

**Definition A.4.** A concept _J_ in _R_ is called _basic_ if _J_ = _R_ , and if all of its minimal instances (i.e. minimal monomial generators) are squarefree. 

We shall later prove in Theorem A.13 that basic concepts have a particularly nice structure for extracting patterns. 

We shall also introduce the notion of “simple” concepts. Consider the ideal _J_ 1 := _⟨x_ black _x_ square _, x_ white _x_ circle _⟩_ , which is a basic concept that represents “black square or white circle”. Such a concept is contained in the intuitively simpler concept _J_ 2 := _⟨x_ black _, x_ white _⟩_ representing “black or white”. Note that _J_ 2 is generated by two primitive instances _x_ black and _x_ white. To capture this intuition that “black or white” is simpler than “black square or white circle”, we define a concept _J_ in _R_ to be _simple_ if _J_ is generated by a non-empty set of primitive instances in _R_ , i.e. a non-empty subset of _{x_ 1 _, . . . , xn}_ . Equivalently, a concept is _simple_ if it is the sum of primitive concepts. Notice that all simple concepts are basic, since a primitive instance is squarefree. As we shall see later in Appendix A.3.2, every basic concept can be “decomposed” into multiple simple concepts. 

10 

To summarize, we have the following hierarchy of different kinds of concepts: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0011-01.png)


## **A.2. Gr¨obner basis theory** 

In this subsection, we review basic terminology and several elementary results in Gr¨obner basis theory. For a quick introduction to what a Gr¨obner basis of an ideal is, see [69]. 

Why is Gr¨obner basis theory important? In essence, if we want to _compute_ the answer to a computational problem involving ideals, we typically need, as a key initial step, to compute the Gr¨obner bases of the input ideals. Subsequent computational steps would usually involve working directly with the computed Gr¨obner bases, rather than the originally given generating sets for the input ideals. To avoid the reader thinking that Gr¨obner basis computations are for “more advanced” operations on ideals, we highlight here that computing the intersections of ideals already relies on Gr¨obner basis computations: Given ideals _J_ 1 _, J_ 2 of _R_ with generating sets _G_ 1 _, G_ 2 respectively, it is already not trivial to compute a generating set for the intersection _J_ 1 _∩ J_ 2. 

## **A.2.1 What is a Gr¨obner basis?** 

Recall that _R_ is a k-vector space with basis _MR_ , where _MR_ denotes the set of monomials in _R_ . The starting point in Gr¨obner basis theory is the sorting of the set _MR_ . 

**Definition A.5.** A _monomial order_ on _R_ is a well-order on _MR_ , denoted by _<_ , such that for all monomials _m_ 1 _, m_ 2 _, m_ in _MR_ satisfying _m ̸_ = 1, the following condition holds: 

## If _m_ 1 _> m_ 2, then _mm_ 1 _> mm_ 2 _> m_ 2. 

Given a monomial order _<_ , and any polynomial _p_ in _R_ , we can always write _p_ = _c_ 1 _m_ 1 + _· · ·_ + _ckmk_ for some non-zero scalars _c_ 1 _, . . . , ck_ in k, and some monomials _m_ 1 _, . . . , mk_ in _MR_ that are sorted in descending order _m_ 1 _> · · · > mk_ with respect to the monomial order _<_ . These scalars and monomials are uniquely determined given _p_ and _<_ . Recall that the monomials _m_ 1 _, . . . , mk_ are precisely all the monomials _of p_ , and each scalar _ci_ is the coefficient of _mi_ in _p_ . The _initial monomial_ of _p_ (with respect to _<_ ), denoted by in _<_ ( _p_ ), is the largest monomial _m_ 1. By default, we define in _<_ (0) = 0. 

Suppose _A_ = _{p_ 1 _, . . . , pk}_ is a finite set of polynomials _k_ in _R_ . Write _p_ = �� _i_ =1 _[f][i][p][i]_ � + _q_ where _f_ 1 _, . . . , fk, q_ are polynomials in _R_ , such that in _<_ ( _p_ ) _≥_ in _<_ ( _fipi_ ) for all 1 _≤ i ≤ k_ , and such that none of the monomials of _q_ are divisible by any monomial contained in _{_ in _<_ ( _p_ 1) _, . . . ,_ in _<_ ( _pk_ ) _}_ . Such an expression always exists and is called a _standard expression_ of _p_ with respect to _A_ . Any such polynomial _q_ 

is called a _remainder_ of _p_ with respect to _A_ . If 0 is a remainder of _p_ with respect to _A_ , then we say that _p reduces to zero_ with respect to _A_ . Note that standard expressions and remainders of _p_ (with respect to _A_ ) are in general not unique. 

Let _f, g ∈ R_ be polynomials that are not both the zero polynomial. The _S-pair_ of _f_ and _g_ (with respect to _<_ ) is the polynomial 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0011-13.png)


where gcd(in _<_ ( _f_ ) _,_ in _<_ ( _g_ )) denotes the greatest common divisor of in _<_ ( _f_ ) and in _<_ ( _g_ ), and _cf_ (resp. _cg_ ) is the coefficient of in _<_ ( _f_ ) (resp. in _<_ ( _g_ )) in _f_ (resp. _g_ ). We use the convention that gcd( _f,_ 0) = _f_ for all non-zero polynomials _f ∈ R_ . By default, define _S_ (0 _,_ 0) = 0. It is straightforward to check that if gcd(in _<_ ( _f_ ) _,_ in _<_ ( _g_ )) = 1, then _S_ ( _f, g_ ) always reduces to zero with respect to _{f, g}_ . 

For any ideal _I_ of _R_ , the monomial ideal in _<_ ( _I_ ) := _⟨{_ in _<_ ( _f_ ) _|f ∈ I}⟩_ , generated by the initial monomials of all elements in _I_ , is called the _initial ideal_ of _I_ (with respect to _<_ ). A _Gr¨obner basis_ for _I_ (with respect to _<_ ) is a **finite** set _G_ of elements in _I_ , such that in _<_ ( _I_ ) is generated by the set _{_ in _<_ ( _f_ ) _|f ∈G}_ . Gr¨obner bases for _I_ (with respect to any monomial order) always exist, and every Gr¨obner basis for _I_ must generate _I_ . If _G_ is a Gr¨obner basis for _I_ , then the remainder of _p_ with respect to _G_ is unique. In particular, _p ∈ I_ if and only if _p_ reduces to zero with respect to _G_ . 

Consider any finite generating set _A_ for the ideal _I_ . _Buchberger’s criterion_ says that _A_ is a Gr¨obner basis for _I_ (with respect to some given monomial order _<_ ) if and only if _S_ ( _f, g_ ) reduces to zero for every _f, g ∈A_ . This criterion yields _Buchberger’s algorithm_ , which is an algorithm to compute a Gr¨obner basis _G_ for any ideal _I_ of _S_ , given a finite generating set _A_ for _I_ as input. Buchberger’s algorithm is described as follows: Starting with the input generating set _A_ , compute the remainders of the _S_ -pairs _S_ ( _f, g_ ) with respect to _A_ , for all generators _f, g ∈A_ ; if any such remainder _q_ is non-zero, then insert the polynomial _q_ into _A_ and repeat the process. This process must terminate after finitely many steps, and the final _A_ obtained after the process terminates is a Gr¨obner basis for _I_ , which contains the initially given generating set _A_ . Note that by definition, all _S_ -pairs of pairs of elements in the final Gr¨obner basis _A_ must reduce to zero with respect to _A_ . 

## **A.2.2 Computational subroutines in algebra** 

Computational problems in algebra can be solved using computer algebra systems. Well-known computer algebra systems (e.g. Maple, Magma, Mathematica, MATLAB; cf. [12, Appendix C]) are frequently used to solve systems of polynomial equations, while more specialized systems (e.g. 

11 

CoCoA [1], GAP [19], Macaulay2 [24], SageMath [72], Singular [14]) are more commonly used by algebraists to solve computational problems where ideals are the “actual objects of study”, i.e. not involving the assignment of numerical values to variables. 

Gr¨obner bases are always defined with respect to some given monomial order _<_ . In many computer algebra systems (e.g. CoCoA [1], Macaulay2 [24], SageMath [72], Singular [14]), the default monomial order for Gr¨obner basis computations is the graded reverse-lexicographic order, also known as the degree reverse-lexicographic order. 

**Definition A.6.** The _graded reverse-lexicographic order_ (grevlex order) on _R_ , which we denote by _<rℓ_ , is a monomial order on _R_ , such that for any distinct monomials **x** _**[α]**_ , **x** _**[β]**_ in _R_ , where _**α**_ = ( _α_ 1 _, . . . , αn_ ) and _**β**_ = ( _β_ 1 _, . . . , βn_ ), we have **x** _**[α]** >rℓ_ **x** _**[β]**_ if and only if either deg( **x** _**[α]**_ ) _>_ deg( **x** _**[β]**_ ); or deg( **x** _**[α]**_ ) = deg( **x** _**[β]**_ ), and _αi < βi_ for the largest 1 _≤ i ≤ n_ such that _αi_ = _βi_ . 

For example, if _n_ = 3 (i.e. _R_ = k[ _x_ 1 _, x_ 2 _, x_ 3]), then the degree 3 monomials in _R_ , arranged in the grevlex order, is given as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0012-04.png)


Notice that in our definition for the grevlex order, the degree 1 monomials (which are precisely the _n_ variables in _R_ ) are arranged as _x_ 1 _>rℓ x_ 2 _>rℓ> · · · >rℓ xn_ . Hence, by permuting these variables, there are _n_ ! distinct grevlex orders that can be defined on a polynomial ring with _n_ variables. In general, we would need to specify a linear order on the _n_ variables in _R_ , then define the grevlex order on top of the specified linear order for these degree 1 monomials. For this paper, we assume implicitly and without loss of generality that _x_ 1 _> x_ 2 _> · · · > xn_ for all monomial orders _<_ on _R_ . 

**Ideal membership.** Let _p ∈ R_ be a polynomial, let _I_ be an ideal of _R_ , and suppose that _G_ is a Gr¨obner basis for _I_ . How do we decide if _p_ is contained in _I_ ? We can compute the remainder of _p_ with respect to _G_ . This remainder is zero if and only if _p_ is contained in _I_ ; see [12, Chap. 2.8]. 

**Ideal containment.** Let _J_ 1 _, J_ 2 be ideals of _R_ , and suppose _G_ 1 _, G_ 2 are Gr¨obner bases for _J_ 1 _, J_ 2, respectively. How do we decide if _J_ 1 _⊆ J_ 2? Suppose _G_ 1 = _{g_ 1 _, . . . , gk}_ . Then _J_ 1 _⊆ J_ 2 if and only if _gi_ reduces to zero with respect to _G_ 2 for all 1 _≤ i ≤ k_ . 

**Intersection of ideals.** Let _J_ 1 _, J_ 2 be ideals of _R_ , and suppose that _G_ 1 = _{g_ 1 _, . . . , gk}, G_ 2 = _{h_ 1 _, . . . , hℓ}_ are Gr¨obner bases for _J_ 1 _, J_ 2, respectively. What is a possible generating set _G_ for _J_ 1 _∩J_ 2? We can compute _G_ algorithmically as follows: First, construct an “extended” polynomial ring _R[′]_ := k[ _x_ 1 _, . . . , xn, t_ ] that has an additional variable _t_ , so that _R_ is a subring of _R[′]_ , and consider any monomial order _<_ such that any monomial divisible by _t_ is larger 

than all monomials not divisible by _t_ . Construct a new ideal _K_ := _⟨tg_ 1 _, . . . , tgk,_ (1 _− t_ ) _h_ 1 _, . . . ,_ (1 _− t_ ) _hℓ⟩_ contained in the new ring _R[′]_ , compute a Gr¨obner basis _G[′]_ for _K_ with respect to the described monomial order _<_ , then compute the subset _G ⊆G[′]_ comprising all those elements _g[′] ∈G[′]_ such that in _<_ ( _g[′]_ ) is not divisible by _t_ . This final set _G_ is not just a generating set for _J_ 1 _∩ J_ 2; it is in fact a Gr¨obner basis for _J_ 1 _∩ J_ 2. For proof details on why this algorithm works, see [12, Chap. 4.3]. 

For a more detailed treatment of Gr¨obner basis theory, including its rich connections to algorithms in computational algebra and the numerous computational problems in commutative algebra, see [17, Chap. 15]. 

## **A.3. Primary ideals and primary decompositions** 

The notion of primary decompositions arises as a farreaching generalization of the idea of prime factorization in integers, and has deep implications in number theory, group theory, and algebraic geometry. For example, the prime factorization of integers, the classification of finite abelian groups, and the decomposition of algebraic varieties into their irreducible components, can be interpreted as three seemingly different decomposition theorems that are three special cases of the celebrated Lasker–Noether theorem; see [17, Thm. 3.10] for a statement of this theorem in its full generality. 

In our context, the Lasker–Noether theorem tells us that every ideal of _R_ can be decomposed into an intersection of finitely many “primary” ideals. Such a decomposition is called a “primary decomposition”, and there are several known algorithms for computing the primary decompositions of ideals in polynomial rings; see [18,21,65] for stateof-the-art algorithms. In our paper, the computation of primary decompositions of concepts (i.e. monomial ideals) is a key ingredient for our algebraic machine reasoning framework. 

## **A.3.1 What exactly is a primary decomposition?** 

To rigorously define primary decompositions and related notions, we first need to introduce more technical algebraic terminology. 

**Prime ideals and primary ideals** . Let _J_ be an ideal of _R_ . We say that _J_ is _proper_ if _J_ is not the entire polynomial ring _R_ , i.e. if _J_ = _⟨_ 1 _⟩_ . We say that _J_ is a _prime_ ideal if _J_ is proper ideal that satisfies the following condition: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0012-17.png)


A prime ideal _P_ of _R_ is called an _associated prime_ of _J_ , if there exists an element _r ∈ R_ satisfying _r ̸∈ J_ , such that _P_ = _{p ∈ R|pr ∈ J}_ . We say the ideal _J_ is _primary_ if _J_ has exactly one associated prime; if this unique associated 

12 

prime is _P_ , then we say that _J_ is a _P -primary_ ideal. There is another useful equivalent definition for primary ideals; cf. [17, Prop. 3.9]. An ideal _J_ of _R_ is _primary_ if it is a proper ideal that satisfies the following condition: 

If _x, y_ are elements of _R_ such that _xy ∈ J_ , then either _x ∈ J_ , or _y[k] ∈ J_ for some _k ∈_ Z[+] . 

**Primary decompositions and associated primes** . We now state (a special case of) the Lasker–Noether theorem: 

**Theorem A.7** ( [17, Thm. 3.10]) **.** _Every proper ideal J of R can be written as an intersection J_ = _J_ 1 _∩· · ·∩Jk for some finitely many primary ideals J_ 1 _, . . . , Jk of R. Furthermore, if each Ji is a Pi-primary ideal for a prime ideal Pi of R, then every associated prime of J must be one of the prime ideals P_ 1 _, . . . , Pk._ 

A _primary decomposition_ of _J ⊆ R_ is a decomposition of _J_ as an intersection of finitely many primary ideals of _R_ . Theorem A.7 says that primary decompositions for proper ideals of _R_ always exist. Note that in general, an ideal _J_ could have multiple primary decompositions. 

Consider a primary decomposition _J_ = _J_ 1 _∩· · · ∩ Jk_ of a proper ideal _J_ . The primary ideals _J_ 1 _, . . . , Jk_ are called the _primary components_ of this primary decomposition.[2] If each _Ji_ is a _Pi_ -primary ideal for some prime ideal _Pi_ , then we say that _Ji_ is a _Pi_ -primary component (of this primary decomposition). If the number _k_ of primary components in the primary decomposition _J_ = _J_ 1 _∩· · · ∩ Jk_ is the smallest among all possible primary decompositions of _J_ , then we say that _J_ = _J_ 1 _∩· · · ∩ Jk_ is a _minimal_ primary decomposition. In this case, the associated primes _P_ 1 _, . . . , Pk_ must all be distinct (see [17, Thm. 3.10c]). Note that minimal primary decompositions are in general not unique. 

The _radical_ of an ideal _J ⊆ R_ is the ideal defined by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0013-07.png)


The next proposition relates radicals to the associated primes of primary components in a primary decomposition. 

**Proposition A.8.** _Let J be a proper ideal of R, and let J_ = _J_ 1 _∩· · · ∩ Jk be a minimal primary decomposition of J. Then for each_ 1 _≤ i ≤ k, the radical[√] Ji is a prime ideal, and the primary component Ji is a[√] Jiprimary ideal. Furthermore,[√] J_ 1 _, . . . ,[√] Jk are all the distinct associated primes of J._ 

_Proof._ Let _J_ 1 _[′][, J]_ 2 _[′]_[be ideals of] _[ √] Ji_ such that _J_ 1 _[′][J]_ 2 _[′][⊆√] Ji_ . Let _j_ 1 _∈ J_ 1 _[′][, j]_[2] _[∈][J]_ 2 _[′]_[be][arbitrarily][chosen][non-zero][ele-] ments. By the definition of a radical, we have ( _j_ 1 _j_ 2) _[t] ∈ J_ 

2When dealing with primary components, it is implicitly assumed that any primary component is referenced with respect to some given primary decomposition. Since primary decompositions are not unique, merely mentioning that _Ji_ is a primary component of _J_ without having any context of the primary decomposition that _Ji_ is a primary component of, would be ambiguous. 

for some _t ∈_ Z[+] , so since _Ji_ is primary, we infer that either _j_ 1 _[t][∈][J][i]_[, or] _[j]_ 2 _[tk][∈][J][i]_[for some] _[ k][∈]_[Z][+][; this implies that ei-] ther _j_ 1 _∈[√] Ji_ or _j_ 2 _∈[√] Ji_ , i.e. _[√] Ji_ is a prime ideal. Thus, by [17, Cor. 2.12], _[√] Ji_ is the smallest prime ideal containing _Ji_ , hence [17, Prop. 3.9] implies that _[√] Ji_ is the unique associated prime of _Ji_ . Finally, [17, Thm. 3.9c] completes the proof. 

**Remark A.9.** A useful consequence of Proposition A.8 is that although an ideal _J_ of _R_ could have multiple different minimial primary decompositions _J_ = _J_ 1 _∩· · · ∩ Jk_ , the set _{[√] J_ 1 _, . . . ,[√] Jk}_ of prime ideals corresponding to any minimal primary decomposition is always the same; this set is called the _set of associated primes_ of _J_ . 

**Remark A.10.** In computer algebra systems, a primary decomposition is typically represented as a list of primary ideals. Although there are known algorithms for computing a minimal primary decomposition[3] of an arbitrary proper ideal of a polynomial ring (see, e.g., [67] for several implementations in Macaulay2), we caution the reader that the computation of primary decompositions may not be supported in some computer algebra systems. 

**Unique minimal primary decompositions** . In the special case that _J_ is a proper monomial ideal of _R_ , the minimal primary decompositions of _J_ are particularly nice: 

**Theorem A.11** ( [4]; cf. [17, Ex. 3.11]) **.** _Let J be a proper monomial ideal of R. Then there exists a minimal primary decomposition J_ = _J_ 1 _∩· · · ∩ Jk satisfying the following:_ 

- _Every primary component Ji is a monomial ideal._ 

- _Each primary component Ji is a Pi-primary ideal, where the prime ideal Pi_ = _[√] Ji is generated by a subset of {x_ 1 _, . . . , xn}._ 

- _Each primary component Ji is maximal among all possible monomial Pi-primary components._ 

_Furthermore, any minimal primary primary decomposition of J satisfying these properties is uniquely determined up to re-orderings of the primary components._ 

Succinctly, every proper monomial ideal _J_ of _R_ has a **unique** minimal primary decomposition with maximal monomial primary components, whose corresponding uniquely determined set _{J_ 1 _, . . . , Jk}_ of primary components is denoted by ( _J_ ). 

In our language of algebraic machine reasoning (see, in particular, Appendix A.1.2), Theorem A.11, combined with Proposition A.8, yields the following corollary. 

**Corollary A.12.** _Let J be a concept of R such that J_ = _⟨_ 1 _⟩, i.e. J is not the “conceivable” concept. Then there exist_ 

> 3Existing implementations of algorithms to compute primary decompositions always return minimal primary decompositions. For a fixed ideal _J_ , it is possible to compute more than one minimal primary decomposition, e.g. in Macaulay2; see [25, 67]. 

13 

_finitely many distinct simple concepts P_ 1 _, . . . , Pk of R that are uniquely determined by J, and there exists a primary decomposition J_ = _J_ 1 _∩· · · ∩ Jk of the concept J as an intersection of k concepts J_ 1 _, . . . , Jk, such that every radical √Ji is the simple concept Pi. Furthermore, any such decomposition for which Ji is maximal over all possible concepts with radical[√] Ji_ = _Pi, is uniquely determined._ 

**Primary decompositions for basic concepts** . As we have mentioned in Appendix A.1.2, we encoded RPM panels algebraically as basic concepts, i.e. whose minimal instances are squarefree. The following theorem tells us that basic concepts are particularly nice, because they have unique minimal primary decompositions whose primary components are simple concepts. 

**Theorem A.13.** _Let J be a basic concept of R. Then there exists a primary decomposition J_ = _J_ 1 _∩· · · ∩ Jk of the concept J as an intersection of finitely many distinct simple concepts J_ 1 _, . . . , Jk. Furthermore, such a decomposition is unique up to re-orderings of the simple concepts J_ 1 _, . . . , Jk._ 

_Proof._ Suppose mingen( _J_ ) = _{m_ 1 _, . . . , mk}_ . Since _J_ = _⟨_ 1 _⟩_ , none of the monomial generators _m_ 1 _, . . . , mk_ can be 1, so each of them has degree _≥_ 1. If all of _m_ 1 _, . . . , mk_ have degrees exactly 1, then _J_ is by definition a simple concept, and we are trivially done. Thus, we can assume henceforth that there is some minimal monomial generator _mj_ such that deg( _mj_ ) _≥_ 2. This means we can write _mj_ = _m[′] m[′′]_ for some monomials _m[′] , m[′′]_ in _MR_ , each with degree _≥_ 1. Since _J_ is a basic concept, all minimal monomial generators of _J_ must be squarefree, so _mj_ is a squarefree monomial, whichˆ implies that gcd( _m[′] , m[′′]_ ) = 1. Hence, by defining _G_ := mingen( _J_ ) _\{mj}_ , we can then decompose _J_ as an intersection 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0014-04.png)


where each of the two ideals _⟨G_[ˆ] _∪{m[′] }⟩_ and _⟨G_[ˆ] _∪{m[′′] }⟩_ in this decomposition is a squarefree monomial ideal. 

Consequently, for every squarefree monomial ideal _J[′]_ in any decomposition of _J_ as an intersection of squarefree monomial ideals, we can iteratively decompose _J[′]_ further as an intersection of two strictly larger squarefree monomial ideals whenever _J[′]_ contains a minimal monomial generator _m_ with degree _≥_ 2. By iteratively repeating this process, we would then get a decomposition _J_ = _J_ 1 _∩· · · ∩ Jk_ , where _J_ 1 _, . . . , Jk_ are squarefree monomial ideals whose minimal monomial generators all have degree 1. By definition, each such _Ji_ is precisely a simple concept, and _[√] Ji_ = _Ji_ , therefore our assertion follows from Corollary A.12. 

## **A.3.2 Inductive bias for algebraic machine reasoning** 

Theorem A.13 has an important consequence. It tells us that any basic concept _J_ of _R_ , no matter how “complicated” it 

may seem, can always be decomposed as an intersection of simple concepts. Simple concepts are easy to interpret, since by definition they are sums of primitive concepts. 

Among all possible simple concepts (i.e. whose generating sets range over all possible non-empty subsets of _{x_ 1 _, . . . , xn}_ , there are certain “distinguished” simple concepts that are important for reasoning. Such “distinguished” simple concepts are task-specific, and we call them _attribute concepts_ . Informally, an attribute concept is a concept representing a certain attribute of interest. 

In the RPM task, recall that we have attribute concepts representing “position”, “number”, “type”, “size”, and “color”; these are simple concepts that categorize the primitive instances according to their semantics, into what humans would call attributes. For example, the “type” attribute concept _J_ type = _⟨x_ triangle _, x_ square _, x_ pentagon _, x_ hexagon _, x_ circle _⟩_ is a simple concept generated by _x_ triangle _, x_ square _, x_ pentagon _, x_ hexagon _, x_ circle, which represent all possible instances of the atribute “type” in the RPM task. 

The minimal instances of an attribute concept can be interpreted as “attribute values”. This means that a simple concept contained in an attribute concept would be generated by “attribute values” of a single “attribute”, and hence would inherit the same semantics as the corresponding attribute concept. For example, the simple concept _J_ := _⟨x_ triangle _, x_ pentagon _, x_ circle _⟩_ is contained in _J_ type and inherits the semantics of the attribute “type”. Notice that _J_ type is a concept that is meaningful to humans (representing possible types of geometric entities), hence _J_ would also be a concept that is similarly meaningful to humans (representing some possible types of geometric entities). After all, the minimal instances of _J_ are all instances of the same attribute “type”. 

Note that the identification of such attribute concepts is task-specific, and the resulting reasoning performance (e.g. in terms of the extracted patterns) would depend heavily on these identified attribute concepts. In particular, our choice of which simple concepts are attribute concepts would determine the _inductive bias_ of our reasoning framework: After decomposing a basic concept into multiple simple concepts, only those simple concepts contained in attribute concepts are deemed “meaningful”, and we extract such simple concepts as concepts representing patterns that are meaningful to humans. Roughly speaking, when extracting common patterns with this inductive bias, our reasoning framework is “biased” towards only those “simpler” concepts that are contained in attribute concepts. 

What about primitive concepts? Typically, they are selfevident and easy to define. This is because the precise formulation of any reasoning task would first require an identification of (the semantics of) the possible data attribute values, which then naturally informs us what our primitive 

14 

concepts should be. For the RPM task, we had 4 perception modules for the 4 attributes “color”, “size”, “type”, “position”, so all the possible attribute values across these 4 attributes would naturally form primitive concepts. 

In general, for visual reasoning tasks, the possible object classes that can be predicted by perception models would naturally take the role of primitive concepts for our reasoning framework. 

## **A.4. Algebraic intuition for concepts** 

## **A.4.1 Proof of Theorem 3.1** 

For convenience to the reader, we re-state Theorem 3.1: 

**Theorem 3.1.** _There are infinitely many concepts in R, even though there are finitely many primitive concepts in R. Furthermore, if J ⊆ R is a concept, then the following hold:_ 

- _(i) J has infinitely many instances, unless J_ = _⟨_ 0 _⟩._ 

- _(ii) J has a unique minimal generating set consisting of finitely many instances, which we denote by_ mingen( _J_ ) _._ 

- _(iii) If J_ = _⟨_ 1 _⟩, then J has a unique set of associated concepts {P_ 1 _, . . . , Pk}, together with a unique minimal primary decomposition J_ = _J_ 1 _∩· · · ∩ Jk, such that each Ji is a concept contained in Pi, that is maximal among all possible primary components contained in Pi that are concepts._ 

_Proof._ Since _R_ = k[ _x_ 1 _, . . . , xn_ ], there are a total of _n_ primitive concepts in _R_ , i.e. _⟨x_ 1 _⟩, . . . , ⟨xn⟩_ . Since the set _MR_ of all monomials in _R_ is infinite, and since each _m ∈MR_ generates a concept _⟨m⟩_ , there are infinitely many concepts in _R_ . To show part (i), notice that if _J_ = _⟨_ 0 _⟩_ , then _J_ must contain at least one monomial _m ∈MR_ , thus by the definition of an ideal, _J_ contains _mx[k]_ 1[for all] _[ k][∈]_[N][ and so has] infinitely many instances. For part (ii), see Proposition A.2. Finally, part (iii) follows directly from Corollary A.12 and Proposition A.8. **A.4.2 Compatibility with human reasoning and human understanding of concepts** 

In cognitive science, _concepts_ are the most fundamental constructs for understanding human cognition [46, 53]. There are numerous competing theories [7,22,30,42,44,47, 58] on what “concepts” are (or should be). Even the seemingly innocuous question of whether the underlying principles (upon which these different theories are based on) are compatible or not, has been fiercely debated, and still remains unresolved today [71]. 

Faced with such circumstances as our backdrop, we do not seek to add to the debate by presenting a new theory on concepts (in the sense of constructs for human cognition). Rather, we seek to explain how our definition of concepts as monomial ideals adequately captures some of the richness of various aspects of concepts in human cognition. To us, 

our algebraic approach is grounded in the intuition that concepts are _computable_ , and they represent abstractions that derive _computable meanings_ from its relations to other concepts. 

**Compositionality.** Concepts are compositional [30, 35]. A concept could be composed of multiple “simpler” concepts. Distinct concepts could share a common concept. In philosophy and in linguistics, this compositional structure for concepts is widely accepted as an important aspect of the human experience in learning new concepts [23]. For the RPM task, the algebraic representation we used to encode RPM panels is compositional. We can compose a concept in _R_ by means of sums, products and intersections of concepts in _R_ . We can decompose a concept as an intersection of multiple concepts (via primary decompositions). As we have discussed in Section 4, and elaborated in Appendix A.3.2, the discovery of abstract patterns is based on primary decompositions; for this algebraic approach to work, we inherently need the “compositional” property of ideals; cf. Appendix C.1. 

**Essence of concepts.** In some theories of concepts [22, 30, 42], there is an imprecise notion of the “essence” of a concept. We could view this “essence” as the definition of the concept (e.g. consider the famous example that the concept “bachelor” is synonymous with “unmarried man”). However, more recent theories of concepts have espoused the belief that many concepts that are meaningful to humans (e.g. “knowledge”, “truth”) do not lend themselves easily to precise definitions in terms of other concepts [20]. This lack of definitional structure in some concepts has led to the idea that concepts can be defined in terms of its constituent _features_ . For example, “feathers” and “wings” can be viewed as features of the concept “bird”. 

Careful distinction has to be made between _defining features_ and _irrelevant features_ . For example, “singing and chirping” could be a feature of the concept “bird”, but it is not a defining feature: A bird does not need to be able to sing or chirp to be considered a bird. Only the defining features of the concept would be considered part of the “essence” of that concept. However, it remains a challenge as to how the defining features of concepts can be extracted in a principled manner; see [37]. 

Is there an analogous notion of “essence” for concepts defined as monomial ideals? By definition, we can construct a concept in _R_ by specifying its generating set; this is very much analogous to specifying features of the concept. Given a generating set _G_ for a concept _J_ in _R_ , we can compute the set mingen( _J_ ) of minimal monomial generators of _J_ . By the uniqueness of mingen( _J_ ) (see Proposition A.2), this means we can interpret the generators in mingen( _J_ ) as being analogous to the defining features of the concept. Generators in _G_ that are not in mingen( _J_ ) are redundant for generating _J_ , and could be interpreted as irrelevant features. 

15 

Consequently, we can view mingen( _J_ ) as the “essence” of concept _J_ : Effectively, mingen( _J_ ) is a “distinguished” finite set of instances that determines all the infinitely many possible instances in _J_ ; cf. Theorem 3.1 in Section 3.1.2 

In contrast, there is no analogous notion of mingen( _J_ ) in a logic-based or set-theoretic setting: For a given logical expression, there is _a priori_ no “distinguished” subset of the Boolean-type variables (appearing in the logical expression), or a subset of the clauses, that would represent the entire logical expression. For any given infinite set, if we do not have any further information about its elements, then there is no natural way to assign to it a “distinguished” finite subset that still represents the entire infinite set. 

**Concepts with partial definitions.** In the classical theory of concepts, a concept is characterized in terms of necessary and sufficient conditions for an instance to be a member of that concept [7]. A key criticism of this classical theory is that humans are still able to reason with concepts that have _partial definitions_ ; this has been the starting point of neoclassical theories of concepts since the 1980s [46]. 

In algebraic machine reasoning, our notion of concepts is well-equipped to handle partial definitions. Consider the attribute value “ _x_ square” from the RPM task. Note that the concept _⟨x_ square _⟩_ represents “square”, which can be defined mathematically as a geometric entity in a plane with four equal sides and four right angles. Yet, we do not require this full definition to solve the RPM task. All we need is a _partial_ definition based on the condition that a square has four sides; we do not explicitly use other necessary conditions such as “equals sides” and “equal angles”. The necessary condition “four sides” is encoded as follows: (i) the relation _f_ next( _⟨x_ triangle _⟩_ ) = _⟨x_ square _⟩_ ; (ii) the relation _f_ next( _⟨x_ square _⟩_ ) = _⟨x_ pentagon _⟩_ ; and implicitly, (iii) a perception model that distinguishes the object class associated to _x_ square from other object classes associated to the remaining variables in _A_ type. For example, even if we do not know the defining feature that a square should have four right angles, and allow a rhombus to be an instance of concept _⟨x_ square _⟩_ , we could still solve the RPM task, provided that interior angles are not a relevant feature for solving (which is the case for both the RAVEN and I-RAVEN datasets). 

At the heart of the versatility of our algebraic approach, especially in dealing with partial definitions, is the fact that we are free to impose any semantics we want on the primitive concepts. In particular, the primitive concepts in _R_ do not necessarily need to match the traditional notion of “primitive concepts”. In the classical theory of concepts (in cognitive science), primitive concepts are primarily based on human sensory perception [42]. In contrast, what we define as primitive concepts in _R_ is up to us, and would effectively depend on the specific reasoning task we wish to solve, as well as the perception models we have; cf. Appendix A.3.2. 

Furthermore, in contrast to logic-based approaches, we do not need to “assign” truth values to all primitive concepts. For example, to determine if an instance _m_ is contained in the concept _J_ = _⟨x_ white _x_ square _⟩_ , we do not need to know if the entity represented by _m_ is small or large. The “position” and “size” attribute values could be unknown. All we need for determining that _m ∈ J_ is to verify the _necessary_ conditions, that the entity represented by _m_ is white and is a square. We could still reason about all instances in _J_ , even if some of these instances have certain unknown attribute values. Consequently, our reasoning framework is able to deal with _partial information_ , and still make inferences about instances for which some of the truth values (true or false), on whether they are contained in each primitive concept, could be unknown. Intuitively, we do not have to assign truth values to all variables (representing statements about instance membership in primitive concepts), to be able to reason using algebraic machine reasoning. 

## **B. Further experiment details** 

In this section, we provide further implementation details of our algebraic machine reasoning framework and more experimental results, organized as follows: 

- Appendix B.1 gives further details on the algebraic representation stage, including the design of the attribute concepts and the performance of our object detection modules. 

- Appendix B.2 describes the details on how to “inversely” apply the 4 invariance modules to generate answers for RPM instances. 

- Appendix B.3 gives examples of full computational details of the 4 invariance modules and the answer generation process. 

- Appendix B.4 gives details on an ablation study of our algebraic machine reasoning framework to analyze the effectiveness of each invariance module in the answer selection task. 

- Appendix B.5 provides examples of generated answers for each RPM configuration in I-RAVEN [26]. 

## **B.1. Algebraic representation for RPM task** 

## **B.1.1 Semantics of algebraic objects** 

**Attribute concepts** . An instance of the RAVEN [83] (or I- RAVEN [26]) dataset is given as a pair of files: an NPZ file that stores the pixel values for a raw image, and an XML file that stores the given answer and information about the attribute values of entities in each of the 16 panels. As described in Section 3.2.1, we first need to define five disjoint sets of variables (representing the five attributes) to encode the RPM instances algebraically. In thi paper, we only considered those attribute values that are involved in our running example for ease of explanation. However, in order to 

16 

|Attribute value set|Variable labels|
|---|---|
|||
|_A_num :=_{x_0_, . . . , x_8_}_|_x_one_, x_two_, . . . , x_nine|
|||
|_A_pos :=_{x_9_, . . . , x_31_}_|_x_(0_._16_,_0_._16_,_0_._33)_, x_(0_._16_,_0_._5_,_0_._33)_, x_(0_._16_,_0_._83_,_0_._33)_, x_(0_._5_,_0_._16_,_0_._33)_, x_(0_._5_,_0_._5_,_0_._33)_,_<br>_x_(0_._5_,_0_._83_,_0_._33)_, x_(0_._83_,_0_._16_,_0_._33)_, x_(0_._83_,_0_._5_,_0_._33)_, x_(0_._83_,_0_._83_,_0_._33)_,_<br>_x_(0_._25_,_0_._25_,_0_._5)_, x_(0_._25_,_0_._75_,_0_._5)_, x_(0_._75_,_0_._25_,_0_._5)_, x_(0_._75_,_0_._75_,_0_._5)_,_<br>_x_(0_._42_,_0_._42_,_0_._15)_, x_(0_._42_,_0_._58_,_0_._15)_, x_(0_._58_,_0_._42_,_0_._15)_, x_(0_._58_,_0_._58_,_0_._15)_,_<br>_x_(0_._5_,_0_._25_,_0_._5)_, x_(0_._5_,_0_._75_,_0_._5)_, x_(0_._25_,_0_._5_,_0_._5)_, x_(0_._75_,_0_._5_,_0_._5)_, x_(0_._5_,_0_._5_,_1_._0)_, x_dummy|
|||
|_A_type :=_{x_32_, . . . , x_36_}_|_x_triangle_, x_square_, x_pentagon_, x_hexagon_, x_circle|
|||
|_A_color :=_{x_37_, . . . , x_46_}_|_x_#255_, x_#224_, x_#196_, x_#168_, x_#140_, x_#112_, x_#84_, x_#56_, x_#28_, x_#0|
|||
|_A_size :=_{x_47_, . . . , x_68_}_|_x_(0_._6_,_0_._15)_, x_(0_._7_,_0_._15)_, x_(0_._8_,_0_._15)_, x_(0_._9_,_0_._15)_,_<br>_x_(0_._4_,_0_._33)_, x_(0_._5_,_0_._33)_, x_(0_._6_,_0_._33)_, x_(0_._7_,_0_._33)_, x_(0_._8_,_0_._33)_, x_(0_._9_,_0_._33)_,_<br>_x_(0_._4_,_0_._5)_, x_(0_._5_,_0_._5)_, x_(0_._6_,_0_._5)_, x_(0_._7_,_0_._5)_, x_(0_._8_,_0_._5)_, x_(0_._9_,_0_._5)_,_<br>_x_(0_._4_,_1)_, x_(0_._5_,_1)_, x_(0_._6_,_1)_, x_(0_._7_,_1)_, x_(0_._8_,_1)_, x_(0_._9_,_1)_,_|



Table 2. Five sets of variables that correspond to the attribute values of five attributes. 

|Attribute|Sub-sequences|Remarks|
|---|---|---|
||||
|position|�_x_(0_._16_,_0_._16_,_0_._33)_, x_(0_._16_,_0_._5_,_0_._33)_, x_(0_._16_,_0_._83_,_0_._33)_, x_(0_._5_,_0_._16_,_0_._33)_, x_(0_._5_,_0_._5_,_0_._33)_,_<br>_x_(0_._5_,_0_._83_,_0_._33)_, x_(0_._83_,_0_._16_,_0_._33)_, x_(0_._83_,_0_._5_,_0_._33)_, x_(0_._83_,_0_._83_,_0_._33)<br>�<br>�_x_(0_._25_,_0_._25_,_0_._5)_, x_(0_._25_,_0_._75_,_0_._5)_, x_(0_._75_,_0_._25_,_0_._5)_, x_(0_._75_,_0_._75_,_0_._5)<br>�<br>�_x_(0_._42_,_0_._42_,_0_._15)_, x_(0_._42_,_0_._58_,_0_._15)_, x_(0_._58_,_0_._42_,_0_._15)_, x_(0_._58_,_0_._58_,_0_._15)<br>�<br>_x_(0_._5_,_0_._25_,_0_._5)_, x_(0_._5_,_0_._75_,_0_._5)_, x_(0_._25_,_0_._5_,_0_._5)_, x_(0_._75_,_0_._5_,_0_._5)_, x_(0_._5_,_0_._5_,_1_._0)_, x_dummy|<br><br><br><br><br><br><br><br><br><br><br>Splitting based on<br>width (the last entry)<br>_f_next(_⟨x⟩,_∆) =_⟨x_dummy_⟩_|
||||
|size|[_x_(0_._6_,_0_._15)_, x_(0_._7_,_0_._15)_, x_(0_._8_,_0_._15)_, x_(0_._9_,_0_._15)]<br>[_x_(0_._4_,_0_._33)_, x_(0_._5_,_0_._33)_, x_(0_._6_,_0_._33)_, x_(0_._7_,_0_._33)_, x_(0_._8_,_0_._33)_, x_(0_._9_,_0_._33)]<br>[_x_(0_._4_,_0_._5)_, x_(0_._5_,_0_._5)_, x_(0_._6_,_0_._5)_, x_(0_._7_,_0_._5)_, x_(0_._8_,_0_._5)_, x_(0_._9_,_0_._5)]<br>[_x_(0_._4_,_1)_, x_(0_._5_,_1)_, x_(0_._6_,_1)_, x_(0_._7_,_1)_, x_(0_._8_,_1)_, x_(0_._9_,_1)]|Splitting based on<br>width (the last entry)|



Table 3. The sub-sequences for attributes “position” and “size”, within which we cyclically order the variables to capture the sequential information of the attribute values. 

use the same algebraic representation set-up across all RPM instances, we shall define the sets of variables that are composed of all possible attribute values. 

In Table 2, we provide the list of all 69 variables (denoted by _x_ 0 _, . . . , x_ 68), grouped into our five attribute sets. We shall also assign labels to these 69 variables; these labels are character strings meant for easier human interpretation of the meaning of each variable. We shall use the indices and variable labels inter-changeably, e.g. _x_ 0 = _x_ one. Note that the last variable _x_ dummy in _A_ pos is introduced for the purpose of subsequent use (in conjunction with the function _f_ next) in the compositional invariance module, which will be explained later. For the rest of the primitive instances, we briefly describe below how they should be interpreted: 

- Instances in _A_ num represent the number of geometric 

entities in a given panel. 

- Instances in _A_ pos represent the positions of sub-panels corresponding to geometric entities, in the form of 3- tuples ( _x_ center _, y_ center _,_ width). (The position values provided in the XML files are in the form of 4-tuples, i.e. ( _x_ center _, y_ center _,_ width _,_ height). We omit the 4th entry since the width and height of each sub-panel are always equal.) 

- Instances in _A_ type represent the types (or “shapes”) of geometric entities. 

- Instances in _A_ color represent the grayscale pixel values of geometric entities, where _x_ #255 represents “white” and _x_ #0 represents “black”. 

- Instances in _A_ size represent the absolute sizes of subpanels, in the form of 2-tuples (relative size, width). 

17 

(The size values provided in the XML files are the relative size of entities (i.e. 0 _._ 4 _, . . . ,_ 0 _._ 9) with respect to the size of the sub-panel.) 

Let _L_ := _{_ num _,_ pos _,_ type _,_ color _,_ size _}_ denote the set of attribute labels. Then for each attribute label _ℓ ∈L_ , we can define the attribute concept _Jℓ_ = _⟨Aℓ⟩_ . 

**Function** _f_ **next in the compositional invariance module** . We have described in Section 3.3.1 that we cyclically order the values in _Aℓ_ for each attribute label _ℓ ∈L_ , and define a function _f_ next( _J|_ ∆) to encode the idea of “next”. Here ∆ represents the step size to map any variable _x ∈Aℓ_ that appears in a generator of concept _J_ to the ∆-th variable after _x_ ( _|_ ∆ _|_ -th variable before _x_ for negative ∆), w.r.t. to the cyclic order on _Aℓ_ . The cyclic order within each _Aℓ_ is used to guarantee that _f_ next( _⟨x⟩|_ ∆) always lies in the same attribute concept as _⟨x⟩_ . 

For attributes “number”, “type” and “color”, the variables can be naturally ordered following the sequential order as indicated in Table 2. For attributes “position” and “size”, we split the sequences of variables into multiple sub-sequences. Within each sub-sequence, we define a cyclic order that follows human intuition for the sequential information that is inherent in the semantics of the variable labels; see Table 3. For example, the “next” position of _x_ (0.75, 0.75, 0.5) (representing the bottom-right subpanel) should cycle back to _x_ (0.25, 0.25, 0.5) (representing the top-left subpanel), the “next” size of _x_ (0 _._ 9 _,_ 0 _._ 5) should cycle back to _x_ (0 _._ 4 _,_ 0 _._ 5), etc. Note that the last few variables _x_ (0 _._ 5 _,_ 0 _._ 25 _,_ 0 _._ 5) _, . . . , x_ (0 _._ 5 _,_ 0 _._ 5 _,_ 1 _._ 0) of “position” are not in any sub-sequence, as there is no obvious sequential information on these position variables. Hence we introduce a new variable _x_ dummy so that for any position variable _x ∈{x_ (0 _._ 5 _,_ 0 _._ 25 _,_ 0 _._ 5) _, . . . , x_ (0 _._ 5 _,_ 0 _._ 5 _,_ 1 _._ 0) _}_ , 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0018-04.png)


**Function** _g_ **in the binary-operator invariance module** . In Section 3.3.3, we introduced the binary-operator module to extract numerical patterns, based on a given real-valued function _g_ on concepts, and a given set Λ of binary operators. In our experiments on RAVEN and I-RAVEN, we only considered attributes “number”, “color” and “size” for this invariance module, as the remaining two attributes “type” and “position” do not have obvious numerical meanings. For attribute “number”, we define the function _g_ num that maps each concept _J_ to the number of generators in the unique minimal generating set of _J_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0018-06.png)


For attribute “color”, we define the function _g_ color to extract the variable index within the color value sequence in Table 2. Note that we only apply _g_ color to those concepts _J_ for 

|Class|GTs|Dets|Recall|AP|
|---|---|---|---|---|
|triangle|120978|120984|1.000|1.000|
|square|95967|95967|1.000|1.000|
|pentagon|121722|121722|1.000|1.000|
|hexagon|95017|95017|1.000|1.000|
|circle|118578|118578|1.000|1.000|



Table 4. Object detection results for attribute “type”, evaluated on the validation set of I-RAVEN. “GTs” refers to the number of ground truth objects, “Dets” refers to the number of detected objects, and “AP” refers to the average precision score. 

which the color variables are invariant across all generators in mingen( _J_ ). Similarly for attribute “size”, we define the function _g_ size to extract the variable index within the size value sub-sequences in Table 3. 

For example, consider the concept below: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0018-12.png)


We can compute 

- _g_ num( _J_ ) = _|_ mingen( _J_ ) _|_ = 2, which indicates that _J_ has two minimal monomial generators; 

- _g_ color( _J_ ) = 0, which indicates that the color variable (invariant across all generators in mingen( _J_ )) is the first element in the color value sequence; 

- _g_ size( _J_ ) = None, as the size variables are different across the two generators in mingen( _J_ ). 

## **B.1.2 Object detection modules** 

|Class|GTs|Dets|Recall|AP|
|---|---|---|---|---|
|#255|114701|114743|1.000|1.000|
|#224|48561|48570|1.000|1.000|
|#196|49626|49627|1.000|1.000|
|#168|50013|50016|1.000|1.000|
|#140|52634|52639|1.000|1.000|
|#112|51144|51144|1.000|1.000|
|#84|48390|48396|1.000|1.000|
|#56|48566|48568|1.000|1.000|
|#28|44464|44468|1.000|1.000|
|#0|44163|44167|1.000|1.000|



Table 5. Object detection results for attribute “color”, evaluated on the validation set of I-RAVEN. 

In this subsubsection, we explain how we trained the object detection modules to extract the attribute values from the raw RPM images. As mentioned in 3.2.2, we trained 4 standard RetinaNet models (each with a ResNet-50 backbone) separately for all attributes except attribute “number”, 

18 

|Class|GTs|Dets|Recall|AP|
|---|---|---|---|---|
|(0.16, 0.16, 0.33)|16646|16646|1.000|1.000|
|(0.16, 0.5, 0.33)|16803|16803|1.000|1.000|
|(0.16, 0.83, 0.33)|16531|16559|1.000|1.000|
|(0.5, 0.16, 0.33)|16652|16652|1.000|1.000|
|(0.5, 0.5, 0.33)|18455|18445|0.999|0.999|
|(0.5, 0.83, 0.33)|18354|18348|1.000|1.000|
|(0.83, 0.16, 0.33)|32000|32000|1.000|1.000|
|(0.83, 0.5, 0.33)|32000|32000|1.000|1.000|
|(0.83, 0.83, 0.33)|32000|32000|1.000|1.000|
|(0.25, 0.25, 0.5)|16761|16761|1.000|1.000|
|(0.25, 0.75, 0.5)|16851|16851|1.000|1.000|
|(0.75, 0.25, 0.5)|18420|18404|0.999|0.999|
|(0.75, 0.75, 0.5)|96000|96000|1.000|1.000|
|(0.42, 0.42, 0.15)|16753|16753|1.000|1.000|
|(0.42, 0.58, 0.15)|16611|16611|1.000|1.000|
|(0.58, 0.42, 0.15)|18128|18128|1.000|1.000|
|(0.58, 0.58, 0.15)|17992|17992|1.000|1.000|
|(0.5, 0.25, 0.5)|18081|18042|0.998|0.998|
|(0.5, 0.75, 0.5)|18135|18104|0.998|0.998|
|(0.25, 0.5, 0.5)|48648|48688|1.000|1.000|
|(0.75, 0.5, 0.5)|32000|32000|1.000|1.000|
|(0.5, 0.5, 1.0)|18441|18417|0.999|0.999|



Table 6. Object detection results for attribute “position”, evaluated on the validation set of I-RAVEN. 

which can be directly inferred by counting the number of predicted bounding boxes within a panel. 

The RAVEN/I-RAVEN dataset typically contains 70,000 RPM instances that are evenly distributed among 7 configurations. Within each configuration, the 10,000 RPM instances are randomly split into 6,000 training instances, 2,000 validation instances, and 2,000 test instances. For our experiments, we used only 10% of the training set, i.e. we trained our object detection modules on 4,200 RPM instances (600 from each configuration). Every RPM instance can be split into 16 grayscale images of size 160 _×_ 160, corresponding to 8 panels in the question matrix and 8 answer options. Each image has one or more geometric entities, which we shall detect. The ground truth labels and bounding box of each geometric entity can be obtained from the given XML files. 

We used the MMDetection package [10] for standard training, with their implementation of RetinaNet models with a ResNet-50 backbone. All hyper-parameters are set to the default values, with the following exceptions: (i) during training, the initial learning rate is 0.003 for all attributes; and (ii) during the inference stage, we use a confidence threshold of 0.99 for “type”, and 0.95 for the other 3 attributes. All 4 modules are trained over 12 epochs. The evaluation results on the validation set with 14,000 in- 

|Class|GTs|Dets|Recall|AP|
|---|---|---|---|---|
|(0.6, 0.15)|23552|19372|0.823|0.823|
|(0.7, 0.15)|14447|11715|0.811|0.811|
|(0.8, 0.15)|14811|14805|1.000|1.000|
|(0.9, 0.15)|20084|20084|1.000|1.000|
|(0.4, 0.33)|32152|32152|1.000|1.000|
|(0.5, 0.33)|30635|30635|1.000|1.000|
|(0.6, 0.33)|30070|31111|1.000|1.000|
|(0.7, 0.33)|29539|29539|1.000|1.000|
|(0.8, 0.33)|30528|30528|1.000|1.000|
|(0.9, 0.33)|29332|29361|1.000|1.000|
|(0.4, 0.5)|35748|35748|1.000|1.000|
|(0.5, 0.5)|34319|34319|1.000|1.000|
|(0.6, 0.5)|32247|32247|1.000|1.000|
|(0.7, 0.5)|33439|33439|1.000|1.000|
|(0.8, 0.5)|33002|33002|1.000|1.000|
|(0.9, 0.5)|32357|32365|1.000|1.000|
|(0.4, 1.0)|5455|5455|1.000|1.000|
|(0.5, 1.0)|5829|5829|1.000|1.000|
|(0.6, 1.0)|5336|5336|1.000|1.000|
|(0.7, 1.0)|26793|26793|1.000|1.000|
|(0.8, 1.0)|25638|25636|1.000|1.000|
|(0.9, 1.0)|26949|26949|1.000|1.000|



Table 7. Object detection results for attribute “size”, evaluated on the validation set of I-RAVEN. 

|Confguration|RAVEN|I-RAVEN|
|---|---|---|
|Center|32000|32000|
|2_×_2Grid|32000|32000|
|3_×_3Grid|31996|31986|
|O-IC|23290|23617|
|O-IG|31988|31987|
|L-R|31999|31998|
|U-D|32000|32000|
|Avg. proportion|96_._10%|96_._24%|



Table 8. The perceptual results of attribute values extracted via object detection modules, evaluated on RAVEN and I-RAVEN, in terms of the number of correctly detected panels (out of 2 _,_ 000 _×_ 16 panels) for each configuration and the overall proportions of correctly detected panels. 

stances (2,000 for each configuration) are shown in Tables 4-7. 

According to the tables above, the object detection module for attribute “type” has only 6 errors out of 552,262 objects, which significantly outperforms the object detection modules for the other 3 attributes. Hence the module for “type” is used as an “anchor” to combine the detection results obtained from all 4 modules. In particular, we first 

19 

obtain the list of detected objects, with a confidence score above 0.99, from the “type” detection module. The prediction result of each object can be reformulated as a pair (type label, bounding box). For every detected “type” object, we then obtain the lists of detection results from the other 3 detection modules with a confidence score above 0.8; within each list, we select the (label, bounding box) pair with the highest Intersection Over Union (IoU) score w.r.t. the anchor bounding box from “type” module. These 3 selected pairs are assumed to be related to the entity corresponding to the pair generated by the “type” module. Thereafter, we can collect 4 closely related (label, bounding box) pairs, each representing a detected object returned by one detection module, and we can combine the labels from these pairs as the attribute values extracted from a single entity. Finally, we applied the 4 object detection modules trained on I-RAVEN to both RAVEN and I-RAVEN datasets, to extract the attribute values from the raw RPM test images. The evaluation results are given in Table 8. 

## **B.2. Full algorithmic details for “inverse" invariance modules and answer generation** 

In this subsection, we provide the details on how to generate answers for RPM instances based on the common patterns extracted from the first two rows and the first two panels in the 3rd row. Informally, for every non-conflicting pattern pair ( _K,_ **J**[ˇ] ) _∈P_ 1 _,_ 2( **J** ) := _P_ 1[(][all][)] ( **J** ) _∩P_ 2[(][all][)] ( **J** ), we shall “inversely” apply the corresponding invariance module to the 3rd row **J**[ˇ] 3 := [ _J_[ˇ] 3 _,_ 1 _, J_[ˇ] 3 _,_ 2]. Recall the following: 

• _Pi_[(][all][)] ( **J** ) represents the set of pattern pairs extracted from the _i_ -th row of concept matrix **J** ; 

   - **J**[ˇ] represents a concept matrix from the extended list [ **J** _,_ **J**[¯][(] _[p]_[1][)] _,_ **J**[ˆ][(] _[p]_[1][)] _, . . . ,_ **J**[¯][(] _[p][k]_[)] _,_ **J**[ˆ][(] _[p][k]_[)] ]; 

- _K_ represents a common pattern specific to one attribute extracted from the first two rows of **J**[ˇ] . 

- We begin by introducing the algorithmic details for each “inverse” invariance module. 

## **B.2.1 “Inverse” intra-invariance module** 

**Algorithm 3** “Inverse” intra-invariance module. **Inputs:** A pattern _ℓ_ and the 3rd row **J**[ˇ] 3 := [ _J_[ˇ] 3 _,_ 1 _, J_[ˇ] 3 _,_ 2]. 1: _I_ out _←⟨_ 0 _⟩_ . 2: _I ←_ pd( _J_[ˇ] 3 _,_ 1 + _J_[ˇ] 3 _,_ 2) _∩_ pd( _J_[ˇ] 3 _,_ 1 _∩ J_[ˇ] 3 _,_ 2). 

|1: <br>2:|_I_out _←⟨_0_⟩_.<br> _I ←_pd(ˇ_J_3_,_1+ ˇ_J_3_,_2|
|---|---|
|3:|**for**_I ∈I_ **do**|
|4:|**if**_I ⊆⟨Aℓ⟩_**then**|
|5:|_I_out _←I_.|
|6:|**return**_I_out.|



Given a sequence of concepts _J_ 1 _, . . . , Jk_ , we have denoted the sum and intersection of these concepts by _J_ + := 

_J_ 1 + _· · ·_ + _Jk_ and _J∩_ := _J_ 1 _∩· · · ∩ Jk_ . Recall that this intra-invariance module extracts patterns where for some attribute, the corresponding set of attribute values for entities in _Ji_ remains invariant over all _i_ . The set of such extracted patterns is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0020-11.png)


Hence for a pattern pair ( _K,_ **J**[ˇ] ) extracted via this invariance module, the common pattern _K_ is by definition an attribute label _ℓ ∈L_ . The set of values for attribute _ℓ_ should remain invariant across _J_[ˇ] 3 _,_ 1, _J_[ˇ] 3 _,_ 2 and _J_[ˇ] 3 _,_ 3. See Algorithm 3 for the pseudo-code to compute the ideal generated by the corresponding attribute variable for _J_[ˇ] 3 _,_ 3. 

Note that Algorithm 3 can automatically filter out those patterns that conflict with the 3rd row **J**[ˇ] 3, without needing any manual checking. Specifically, the output of Algorithm 3 would be a zero ideal ( _I_ out = _⟨_ 0 _⟩_ ) if the input pattern _ℓ_ conflicts with the input 3rd row **J**[ˇ] 3. For the remaining 3 “inverse” invariance modules, we have the same mechanism to automatically filter the conflicting patterns, i.e. the presence of conflicting patterns would yield an output _I_ out = _⟨_ 0 _⟩_ . 

## **B.2.2 “Inverse” inter-invariance module** 

- **Algorithm 4** “Inverse” inter-invariance module. **Inputs:** A pattern ( _ℓ, I_ ) and the 3rd row **J**[ˇ] 3 := [ _J_[ˇ] 3 _,_ 1 _, J_[ˇ] 3 _,_ 2]. 1: _I_ out _←⟨_ 0 _⟩_ . 2: Compute _I[′]_ in (5). 3: **if** _I −I[′]_ = ∅ and _I ∩I[′]_ = ∅ **then** 4: _I_ out _←_[�] _I∈I−I[′][ I]_[.] 5: **return** _I_ out. 

Recall that the set of patterns extracted via inter-invariance module is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0020-17.png)


For a pattern pair ( _K,_ **J**[ˇ] ) extracted via this invariance module, the common pattern _K_ = ( _ℓ, I_ ) is a pair, whose first entry _ℓ_ is an attribute label, and whose second entry _I_ is a set of ideals contained in the attribute concept _⟨Aℓ⟩_ . Naturally, we shall infer the attribute values for _J_[ˇ] 3 _,_ 3 based on the set difference _I −I[′]_ , where _I[′]_ is defined by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0020-19.png)


Note that this _I[′]_ is defined in the manner that is analogous to our definition for _P_ inter (restricted to the 3rd row of **J**[ˇ] and to attribute _ℓ_ ). See Algorithm 4 for a summary. 

20 

## **B.2.3 “Inverse” compositional invariance module** 

**Algorithm 5** “Inverse” compositional invariance module. 

|**Inputs:** A pattern (_ℓ, f_next_|_∆=∆_∗_)|**Inputs:** A pattern (_ℓ, f_next_|_∆=∆_∗_)|**Inputs:** A pattern (_ℓ, f_next_|_∆=∆_∗_)|and the 3rd row ˇ**J**3 :=|and the 3rd row ˇ**J**3 :=|
|---|---|---|---|---|
|[ˇ_J_3_,_1_,_ ˇ_J_3_,_2].|||||
|1: _I_out _←⟨_0_⟩_.|||||
|2: _I ←_pd<br>�<br>_f_ 2<br>next( ˇ_J_3_,_1_|_∆_∗_)<br>�|_∩_pd|�|_f_next(ˇ_J_3_,_2_|_∆_∗_)<br>�|.|
|3: **for**_I ∈I_ **do**|||||
|4:<br>**if**_I ∈⟨Aℓ⟩_**then**|||||
|5:<br>_I_out _←I_.|||||
|6: **return**_I_out.|||||



Given a function _f_ that maps concepts to concepts, the compositional invariance module extracts the patterns arising from invariant attribute values in the new sequence of concepts: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0021-04.png)


The set of such patterns is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0021-06.png)


Each pattern _K_ = ( _ℓ, f_ ) can be interpreted as follows: The values of attribute _ℓ_ are invariant across the new sequence of concepts. In other words, for any pair of successive concepts [ _Ji, Ji_ +1], _i_ = 1 _, . . . , k −_ 1, in the original sequence, the values of attribute _ℓ_ should be invariant in [ _f_ ( _Ji_ ) _, Ji_ +1]. This property provides a natural way to compute the attribute values of a concept _Ji_ +1, given the attribute values of the previous concept _Ji_ in the sequence; see Algorithm 5. Note that for the RPM task, we introduce the _f_ next _|_ ∆ function for our compositional invariance module to extract the sequential information within a row of the concept matrix. 

## **B.2.4 “Inverse” binary-operator invariance module** 

|**Algorithm 6**“Inverse” binary-operator invariance module.|
|---|
|**Inputs:**<br>A pattern (<br>**_⊘_**_, gℓ,_Λ), and the 3rd row ˇ**J**3 :=|
|[ˇ_J_3_,_1_,_ ˇ_J_3_,_2].|
|1: _I_out _←⟨_0_⟩_.|
|2: _t ←gℓ_(ˇ_J_3_,_1)_⊘_1_gℓ_(ˇ_J_3_,_2).<br>//_expected gℓ_(ˇ_J_3_,_3)_value_|
|3: _x ←t_-th variable in the corresponding attribute se-|
|quence (or subsequence) for attribute_ℓ_.|
|4: _I_out _←⟨x⟩_.|
|5: **return**_I_out.|



Given a real-valued function _g_ on concepts and a set Λ of binary operators, the binary-operator invariance module extracts patterns arising from the sequence of numbers 

_g_ ( _J_ 1) _, . . . , g_ ( _Jk_ ), described as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0021-12.png)


In other words, we can represent each pattern as an equation, obtained by inserting binary operators chosen from Λ between the numbers in the sequence. 

For a new given sequence of concepts _J_ 1 _[′][, . . . , J] k[′] −_ 1[and] a pattern _K_ = ([ _⊘_ 1 _, . . . , ⊘k−_ 2] _, g,_ Λ), the value of _g_ ( _Jk[′]_[)] we expect from this pattern can be directly computed as _g_ ( _Jk[′]_[)][=] _[g]_[(] _[J]_ 1 _[′]_[)] _[ ⊘]_[1] _[· · · ⊘][k][−]_[2] _[g]_[(] _[J] k[′] −_ 1[)][.][Then][we][can][infer] the information about concept _Jk[′]_[based on the meaning of] function _g_ . 

As described in Section B.1, we introduced the function _g_ num( _J_ ) to extract the number of minimal monomial generators of _J_ (which also represents the number of entities in the corresponding panel), and introduced the functions _g_ color (resp. _g_ size) to extract the variable index within the “color” value sequence (resp. within the “size” value sub-sequence). Hence we can directly infer the corresponding attribute variables for _Jk[′]_[from the values of] _[ g][ℓ]_[(] _[J] k[′]_[)][, for] _ℓ ∈_ [“num” _,_ “color” _,_ “size”]. See Algorithm 6 for a summary. 

|**Algorithm 7**Detailed answergenerationprocess.|**Algorithm 7**Detailed answergenerationprocess.|**Algorithm 7**Detailed answergenerationprocess.|**Algorithm 7**Detailed answergenerationprocess.|
|---|---|---|---|
|**Inputs: A concept matrix J**= [_J_1_,_1_, . . . , J_3_,_2]**.**||||
|1:|Compute comPos(**J**) = [_p_1_, . . . , pk_].|||
|2:|_I_out _←⟨_0_⟩_.|||
|3:|**for**_pi ∈_[_p_1_, . . . , pk_]**do**|||
||_// initialize with fxed “number” and “position”_|||
|4:|_I_(_i_) _←⟨xk−_1_xpi⟩_.|||
|5:|**for**_K ∈P_(¯**J**(_pi_)<br>1<br>)_∩P_(¯**J**(_pi_)<br>2|)**do**||
|6:|_I∗←_Output ideal generated based||on one of the|
||Algorithms3-6, given inputs_K_ and ¯**J**(_pi_)<br>3||.|
|7:|**if**_I∗_=_⟨_0_⟩_**then**|||
|8:|_I_(_i_) _←I_(_i_) _∩I∗_|||
|9:|_m ←_generator randomly selected frommingen(_I_(_i_)).|||
|10:|_I_out _←I_out+_⟨m⟩_.|||
|11:|_G ←_∅|||
|12:|**for** _m ∈_mingen(_I_out)**do**|||
|13:|_m′ ←m_|||
|14:|**while**_∃ℓ∈L_s.t. _x_does not|divide_m ∀x ∈Aℓ_**do**||
|15:|_x′ ←_a variable randomly|selected from_Aℓ_.||
|16:|_m′ ←m′x′_.|||
|17:|_G ←G ∪{m′}_.|||
|18:|_I_out _←⟨G⟩_.|||
|19:|**return**_I_out.|||



## **B.2.5 Overall process of answer generation** 

In the RAVEN/I-RAVEN dataset, the values for some attribute within a panel may contain randomness by design. 

21 

|Methods|Avg. Acc.|Center|2_×_2G|3_×_3G|O-IC|O-IG|L-R|U-D|
|---|---|---|---|---|---|---|---|---|
|With all 4 invariance modules|93.2|99.5|89.6|89.7|99.6|74.7|99.7|99.5|
|W/o intra-invariance module|63.7|63.5|71.6|73.4|60.2|49.8|63.2|64.0|
|W/o inter-invariance module|59.0|64.3|56.8|59.2|59.2|43.5|64.5|65.8|
|W/o compositional invariance module|69.5|73.0|65.9|64.3|78.1|61.5|72.9|71.2|
|W/o binary operator invariance module|74.6|75.8|69.3|67.1|86.2|70.2|77.3|76.7|



Table 9. Detailed ablation study results of our algebraic machine reasoning framework, evaluated on the I-RAVEN dataset, in terms of the overall answer-selection accuracy and the individual accuracies for all seven RPM configurations. 

For example, there do not exist rules for both “number” and “position” to avoid conflicts, thus the entity positions in a panel from some configurations (such as 2 _×_ 2Grid, 3 _×_ 3Grid, and Out-InGrid) could be totally random once the number of entities is fixed. The values of other attributes could also be random; see Fig. 5 for example. It is difficult to extract patterns from those concept matrices with such random attribute values, and effectively impossible to generate an answer that matches the given correct answer. Hence, for any given RPM instance, we shall simplify the answer generation process by focusing on each single position. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0022-03.png)


Figure 5. An example of a RPM instance with random values for attributes “position”, “type” and “color”, within each panel. The correct answer is marked with a red box. 

Recall that for a concept matrix **J** representing a given RPM instance, we extract the common patterns from each concept matrix **J**[ˇ] in an extended list of concept matrices, i.e. [ **J** _,_ **J**[¯][(] _[p]_[1][)] _,_ **J**[ˆ][(] _[p]_[1][)] _, . . . ,_ **J**[¯][(] _[p][k]_[)] _,_ **J**[ˆ][(] _[p][k]_[)] ]; see Section 3.3.4. Here _p_ 1 _, . . . , pk_ are the common positions we extracted from the **J** 8 panels of the question matrix corresponding to¯[(] _[p] t_[)] _,_ ˆ **J**[(] _[p] t_[)] are defined as follows: **J** , and all 

- Each concept _J_[¯] _i,j_[(] _[p][t]_[)] in **J**[¯][(] _[p][t]_[)] is generated by the unique generator in _Ji,j_ that is divisible by _pt_ ; 

- Each concept _J_[ˆ] _i,j_[(] _[p][t]_[)] in **J**[ˆ][(] _[p][t]_[)] is generated by all generators in _Ji,j_ that are not divisible by _pt_ . 

In other words, for a common position _pt_ , each panel in the original question matrix is split into two panels, with one panel containing only the entity in position _pt_ (represented 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0022-09.png)


Figure 6. An example of RPM instance from the I-RAVEN dataset. The correct answer is marked with a red box. (This is the same as Fig. 1; we put it here for convenience.) 

by the corresponding concept in **J**[¯][(] _[p][t]_[)] ), and the other panel containing all the remaining entities (represented by the corresponding concept in **J**[ˆ][(] _[p][t]_[)] ). In the answer generation task, we shall consider only all **J**[¯][(] _[p][i]_[)] as discussed before. Intuitively, we can separately extract the patterns restricted to each single common position, across the panels in the original question matrix. Since the attributes “number” and “position” are inherently fixed within each **J**[¯][(] _[p][i]_[)] , we only need to extract patterns for the remaining 3 attributes. See Algorithm 7 for details. Note that in Line 6 of Algorithm 7, when generating the output ideal _I[∗]_ , the exact algorithm (from Algorithm 3-6) to apply to the inputs _K_ and **J**[¯][(] 3 _[p][i]_[)] is uniquely determined by the syntax of _K_ . 

## **B.3. Examples for invariance modules** 

Consider the RPM instance depicted in Fig. 6. In this subsection, we provide specific computational details on this RPM instance to show how the invariance modules are used to extract patterns, and generate the correct answer. 

**Example 1 (Intra-invariance module)** . Consider the first row in Fig. 6. By extracting attribute values for entities in each panel via the object detection modules, we can 

22 

## Next, we compute pd( _J_ +) _,_ pd( _J∩_ ) as follows: 

obtain the algebraic representation given as follows: 

pd( _J_ +) = _{⟨x_ two _⟩, ⟨x_ small _⟩, ⟨x_ left _, x_ right _⟩,_ 

## **J** 1 := [ _J_ 1 _,_ 1 _, J_ 1 _,_ 2 _, J_ 1 _,_ 3] _,_ 

- _⟨x_ white _, x_ gray _, x_ dgray _, x_ black _⟩,_ 

_J_ 1 _,_ 1 = _⟨x_ two _x_ left _x_ square _x_ black _x_ avg _, x_ two _x_ right _x_ triangle _x_ gray _x_ avg _⟩,_ 

_⟨x_ square _, x_ pentagon _, x_ hexagon _, x_ circle _⟩, ⟨x_ left _, x_ dgray _⟩, ⟨x_ left _, x_ pentagon _, x_ hexagon _, x_ circle _⟩, ⟨x_ right _, x_ square _, x_ pentagon _, x_ circle _⟩, ⟨x_ right _, x_ square _, x_ pentagon _, x_ black _⟩, ⟨x_ right _, x_ square _, x_ circle _, x_ white _⟩, ⟨x_ right _, x_ square _, x_ white _, x_ black _⟩, ⟨x_ right _, x_ pentagon _, x_ circle _, x_ gray _⟩, ⟨x_ right _, x_ pentagon _, x_ gray _, x_ black _⟩, ⟨x_ right _, x_ circle _, x_ white _, x_ gray _⟩, ⟨x_ right _, x_ white _, x_ gray _, x_ black _⟩, ⟨x_ square _, x_ pentagon _, x_ circle _, x_ dgray _⟩, ⟨x_ square _, x_ pentagon _, x_ dgray _, x_ black _⟩, ⟨x_ square _, x_ circle _, x_ white _, x_ dgray _⟩, ⟨x_ square _, x_ white _, x_ dgray _, x_ black _⟩, ⟨x_ pentagon _, x_ hexagon _, x_ circle _, x_ gray _⟩, ⟨x_ pentagon _, x_ circle _, x_ gray _, x_ dgray _⟩, ⟨x_ pentagon _, x_ gray _, x_ dgray _, x_ black _⟩, ⟨x_ circle _, x_ white _, x_ gray _, x_ dgray _⟩, },_ 

_J_ 1 _,_ 2 = _⟨x_ two _x_ left _x_ pentagon _x_ gray _x_ avg _, x_ two _x_ right _x_ square _x_ gray _x_ avg _⟩,_ 

_J_ 1 _,_ 3 = _⟨x_ two _x_ left _x_ circle _x_ white _x_ avg _, x_ two _x_ right _x_ pentagon _x_ gray _x_ avg _⟩._ 

## We first compute pd( _J_ +) and pd( _J∩_ ). 

- pd( _J_ +) = _{⟨x_ two _⟩, ⟨x_ avg _⟩, ⟨x_ left _, x_ right _⟩, ⟨x_ white _, x_ gray _, x_ black _⟩, ⟨x_ triangle _, x_ square _, x_ pentagon _, x_ circle _⟩, ⟨x_ left _, x_ gray _⟩,_ 

_⟨x_ square _, x_ circle _, x_ gray _⟩, ⟨x_ square _, x_ white _, x_ gray _⟩ ⟨x_ circle _, x_ gray _, x_ black _⟩, ⟨x_ left _, x_ triangle _, x_ square _, x_ pentagon _⟩ ⟨x_ right _, x_ square _, x_ pentagon _, x_ circle _⟩ ⟨x_ right _, x_ square _, x_ pentagon _, x_ white _⟩ ⟨x_ right _, x_ pentagon _, x_ circle _, x_ black _⟩ ⟨x_ right _, x_ pentagon _, x_ white _, x_ black _⟩ ⟨x_ triangle _, x_ square _, x_ pentagon _, x_ white _⟩},_ 

- pd( _J∩_ ) = _{⟨x_ two _⟩, ⟨x_ avg _⟩, ⟨x_ left _, x_ right _⟩, ⟨x_ gray _⟩, ⟨x_ triangle _, x_ square _⟩, ⟨x_ square _, x_ pentagon _⟩, ⟨x_ pentagon _, x_ circle _⟩, ⟨x_ left _, x_ triangle _⟩, ⟨x_ left _, x_ square _⟩, ⟨x_ left _, x_ pentagon _⟩, ⟨x_ right _, x_ square _⟩, ⟨x_ right _, x_ pentagon _⟩, ⟨x_ right _, x_ circle _⟩, ⟨x_ right _, x_ white _⟩, ⟨x_ right _, x_ black _⟩, ⟨x_ triangle _, x_ black _⟩, ⟨x_ pentagon _, x_ white _⟩}._ 

- pd( _J∩_ ) = _{⟨x_ two _⟩, ⟨x_ small _⟩, ⟨x_ left _, x_ right _⟩, ⟨x_ pentagon _⟩, ⟨x_ square _, x_ circle _⟩, ⟨x_ hexagon _, x_ circle _⟩, ⟨x_ white _, x_ dgray _⟩, ⟨x_ gray _, x_ dgray _⟩, ⟨x_ dgray _, x_ black _⟩, ⟨x_ left _, x_ hexagon _⟩, ⟨x_ left _, x_ circle _⟩, ⟨x_ left _, x_ dgray _⟩, ⟨x_ right _, x_ square _⟩, ⟨x_ right _, x_ circle _⟩, ⟨x_ right _, x_ white _⟩, ⟨x_ right _, x_ gray _⟩, ⟨x_ right _, x_ black _⟩, ⟨x_ square _, x_ dgray _⟩, ⟨x_ hexagon _, x_ black _⟩, ⟨x_ circle _, x_ gray _⟩, ⟨x_ circle _, x_ dgray _⟩}._ 

For those primary components that are contained in attribute concepts, we have highlighted them in blue. These are the concepts that are deemed “meaningful” for the RPM task. Note that most of the primary components are not contained in any attribute concept. 

Next, we compute the set intersection pd( _J_ +) _∩_ pd( _J∩_ ) = _{⟨x_ two _⟩, ⟨x_ avg _⟩, ⟨x_ left _, x_ right _⟩}_ , which can be interpreted to mean that all panels have two average-sized entities, one on the left side and one on the right side. 

Similarly, we have indicated in blue those primary components that are contained in attribute concepts. 

Recall that the output of the intra-invariance module is: 

Recall that the output of the inter-invariance module is: 

_P_ intra([ _J_ 1 _. . . Jk_ ]) := �attr _∈L | ∃I ∈_ pd( _J_ +) _∩_ pd( _J∩_ ) _, I ⊆⟨A_ attr _⟩_ �. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0023-20.png)


## Hence, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0023-22.png)


Hence _P_ inter( **J** 2) is the following set with two pairs: ���“color”“type” _,, {⟨ {⟨xx_ whitepentagon _, x_ dgray _⟩, ⟨x⟩_ square _, ⟨x_ gray _, x, x_ circledgray _⟩, ⟨⟩x, ⟨_ hexagon _x_ black _, x, x_ circledgray _⟩}⟩}_ �� _,_ �. 

**Example 2 (Inter-invariance module)** . Consider the second row **J** 2 in Fig. 6. Similarly, we first obtain the algebraic representation of **J** 2: 

The first pair encodes the information that the three extracted “color” concepts correspond to “white or dark gray”, “gray or dark gray”, and “black or dark gray”. The second pair encodes the information that the three extracted _,_ “type” concepts correspond to “pentagon”, “square or circle”, and “hexagon or circle”. 

**J** 2 := [ _J_ 2 _,_ 1 _, J_ 2 _,_ 2 _, J_ 2 _,_ 3] _,_ 

_J_ 2 _,_ 1 = _⟨x_ two _x_ left _x_ pentagon _x_ white _x_ small _, x_ two _x_ right _x_ pentagon _x_ dgray _x_ small _⟩,_ 

_J_ 2 _,_ 2 = _⟨x_ two _x_ left _x_ circle _x_ black _x_ small _, x_ two _x_ right _x_ hexagon _x_ dgray _x_ small _⟩, J_ 2 _,_ 3 = _⟨x_ two _x_ left _x_ square _x_ gray _x_ small _, x_ two _x_ right _x_ circle _x_ dgray _x_ small _⟩._ 

23 

**Example 3 (Compositional invariance module)** . Consider the right subpanels of the first row **J**[¯][(] 1[right][)] in Fig. 6, and ∆= 1. We can generate the new row: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0024-01.png)


Recall that the output of the compositional invariance module is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0024-03.png)


Hence, we have _P_ comp( **J**[¯][(] 1[right][)] ) = _{_ (“type” _, f_ next _|_ ∆=1) _}_ , which means that the “type” value (i.e. _x_ pentagon) remains invariant in the new row generated by _f_ next with step ∆= 1. 

**Example 4 (Binary-operator module)** . Recall that the output of the binary-operator module is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0024-06.png)


Let _g_ be the real-valued function on concepts given by _J �→ |_ mingen( _J_ ) _|_ . Note that for both rows **J** 1 and **J** 2 in Fig. 6, every panel _Ji,j_ has two entities, which can be computed as _g_ ( _Ji,j_ ) = 2. Hence, if Λ = _{_ + _, −}_ , then _P_ binary( **J** 1) and _P_ binary( **J** 2) are both empty. 

**Example 5 (Answer generation)** . For the example depicted in Fig. 6, we have comPos( **J** ) = _{x_ left _, x_ right _}_ . Note that[¯] **J** ( _x_ left) = ˆ **J** ( _x_ right). (The panel with only the left entity is identical to the panel without the right entity.) Hence we shall consider only the list of concept matrices [ **J** _,_[¯] **J** ( _x_ left) _,_ ¯ **J** ( _x_ right)]. 

By applying the invariance modules iteratively, we can then compute _P_ 1 _,_ 2( **J** ) := _P_ 1[(all)] ( **J** ) _∩P_ 2[(all)] ( **J** ), whose 13 elements are labeled as P1 _, . . . ,_ P13: 

P1 = �“num” _,_ **J** � P2 = �“num” _,_[¯] **J** ( _x_ left)[�] P3 = �“num” _,_[¯] **J** ( _x_ right)[�] P4 = �“pos” _,_ **J** � P5 = �“pos” _,_[¯] **J** ( _x_ left)[�] P6 = �“pos” _,_[¯] **J** ( _x_ right)[�] P7 = �(“type” _, {⟨x_ square _⟩, ⟨x_ pentagon _⟩, ⟨x_ circle _⟩}_ ) _,_[¯] **J** ( _x_ left)[�] P8 = �(“type” _, f_ next _|_ ∆=1) _,_[¯] **J** ( _x_ right)[�] P9 = �“color” _,_[¯] **J** ( _x_ right)[�] P10 = �(“color” _, {⟨x_ white _⟩, ⟨x_ gray _⟩, ⟨x_ black _⟩}_ ) _,_[¯] **J** ( _x_ left)[�] P11 = �“size” _,_ **J** � P12 = �“size” _,_[¯] **J** ( _x_ left)[�] P13 = �“size” _,_[¯] **J** ( _x_ right)[�] 

We can also generate the algebraic representations of the first two panels in the 3rd row **J** 3 := [ _J_ 3 _,_ 1 _, J_ 3 _,_ 2]: 

_J_ 3 _,_ 1 = _⟨x_ two _x_ left _x_ circle _x_ avg _x_ gray _, x_ two _x_ right _x_ pentagon _x_ gray _x_ large _⟩_ ; _J_ 3 _,_ 2 = _⟨x_ two _x_ left _x_ square _x_ avg _x_ white _, x_ two _x_ right _x_ hexagon _x_ gray _x_ large _⟩._ 

Among these 13 patterns, only P11 (the “size” values of entities are invariant across each row in **J** ) _conflicts_ with 

**J** 3, as the “size” values of the two entities in _J_ 3 _,_ 1 (or _J_ 3 _,_ 2) are different. Each remaining non-conflicting pattern gives information about the corresponding attribute value for _J_ 3 _,_ 3. P1 to P6: two entities in _J_ 3 _,_ 3, with one entity on the left side and the other entity on the right side; P7 _⇒ x_ pentagon for the left entity; P8 _⇒ x_ circle for the right entity; P9 _⇒ x_ gray for the right entity; P10 _⇒ x_ black for the left entity; P12 _⇒ x_ avg for the left entity; and P13 _⇒ x_ large for the right entity. Consequently, we can collect all the attribute values to form our generated answer concept 

_J_ 3 _[′] ,_ 3[=] _[ ⟨][x]_[two] _[x]_[left] _[x]_[pentagon] _[x]_[avg] _[x]_[black] _[, x]_[two] _[x]_[right] _[x]_[circle] _[x]_[large] _[x]_[gray] _[⟩]_[.] 

## **B.4. Ablation study** 

In order to study the effect of each invariance module, we conducted an ablation study and evaluated the performance of our algebraic machine reasoning framework on the answer selection task; see Table 9. 

For any given RPM instance, there could be a tie in the comPattern list (defined in Algorithm 1), i.e. there exist multiple entries in comPattern with the same highest score. Recall that the _i_ -th entry in comPattern represents the number of extracted common patterns across 3 rows, with the _i_ -th answer option inserted in the question matrix. Hence we report the weighted accuracies of our framework, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0024-19.png)


where _N_ denotes the total number of RPM test instances, and _ni_ represents the number of answer options with the same highest score for the _i_ -th test instance. 

As shown in Table 9, the inter-invariance module has the most significant contribution towards the final average accuracy. Interestingly, this very same module allows us to discover an unexpected new pattern as depicted in Fig. 4, which is a rather natural pattern that human could conceivably think of, but which is not one of the designed rules for I-RAVEN. 

## **B.5. Examples of generated answers** 

To further evaluate the performance of our algebraic machine reasoning framework on the answer generation task, we provide in Fig. 7 the comparison between the given correct answer images and our generated answer images for the I-RAVEN dataset. For each configuration, we choose the first 10 test files to show the comparison between the given answer images and the generated images. (In the I-RAVEN dataset, every 10 instances are split into 6 training instances, 2 validation instances, and 2 test instances. Hence every test instance has a filename that ends with “8 ~~t~~ est” or “9 ~~t~~ est”.) 

From Fig. 7, we can see that for the configurations Center, Out-InCenter, Left-Right, and 

24 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices_images/cvpr_2023_Abstract_Visual_Reasoning_An_Algebraic_Approach_for_Solving_Raven’s_Progressive_Matrices.pdf-0025-00.png)


Figure 7. A collection of the given answer images (top) and the corresponding generated images (bottom) for the first 10 I-RAVEN test instances in each configuration. For those test instances for which our reasoning framework fails to extract non-conflicting patterns, the output of Algorithm 7 would be _I_ out = _⟨_ 0 _⟩_ ; we have indicated such generated results as completely black panels. 

25 

Up-Down, our generated answers capture almost all salient image attribute features, except for the angles of entities (i.e. the orientation with respect to the panels) which is not involved in any of the rules of RAVEN/I-RAVEN. However, for the remaining configurations, our reasoning framework is only able to generate a part of the answer. This is because we simplified the answer generation process by separately extracting patterns restricted to every single common position, across the panels in the question matrix. Hence, for those configurations where the RPM panels always contain a fixed set of positions, our reasoning framework is able to effectively extract the patterns and generate the correct answer. On the other hand, for those configurations where the RPM panels may contain an arbitrary set of positions, our reasoning framework does not perform as well. 

## **C. Further Discussion** 

## **C.1. Why is algebraic machine reasoning different from logic-based reasoning?** 

Logic is the foundation of reasoning. Logic programming [3, 28, 41] is a logic-based programming paradigm that serves as the foundational computational framework for logic-based reasoning methods. At the heart of these methods in logic-based reasoning is the idea that reasoning can be realized very concretely as the resolution (or inverse resolution) of logical expressions. Inherent in this idea is the notion of _satisfiability_ ; cf. [28]. Intuitively, we have a logical expression, typically expressed in some canonical normal form, and we want to assign truth values (true or false) to literals in the logical expression, so that the entire logical expression is satisfied (i.e. with truth value “true”). In fact, much of today’s very exciting progress in automated theorem proving [2, 27, 36, 39, 54, 74, 81, 92] is based on logic-based reasoning. 

In contrast, algebraic machine reasoning builds upon computational algebra and computer algebra systems; cf. Appendix A.2.2. As described in Section 4, at the heart of our algebraic approach is the idea that reasoning can be realized very concretely as solving computational problems in algebra. Crucially, there is no notion of satisfiability. We do not assign truth values (or numerical values) to concepts in _R_ = k[ _x_ 1 _, . . . , xn_ ]. In particular, although primitive concepts _⟨x_ 1 _⟩, . . . , ⟨xn⟩_ in _R_ correspond to the variables _x_ 1 _, . . . , xn_ , we do not assign values to primitive concepts. Instead, as we have emphasized in Section 3, ideals are treated as the “actual objects of study”, and we reduce “solving a reasoning task” to “solving non-numerical computational problems involving ideals”. 

This contrast is perhaps even more evident when comparing algebraic machine reasoning to inductive logic programming [51, 52]. Both seek to solve reasoning tasks involving the discovery of new patterns, but they are funda- 

mentally different approaches. 

In inductive logic programming, we are given (as our starting point) background knowledge _B_ and examples _E_ , both in the form of logical formulas. For example, _B_ could be a conjunction of implications ( _p_ 1 _∧· · ·∧ pk → q_ ), and _E_ could be a formula in conjunctive normal form. The goal is to discover a hypothesis _H_ , which is a logical formula that must satisfy _B ∧ H |_ = _E_ (i.e. _B ∧ H_ entails _E_ ). Informally, we seek to discover a “good” hypothesis _H_ that is consistent with both background knowledge and examples. The key technical challenge is to find a hypothesis _H_ that is “good” in a precise computable sense, from among the numerous (potentially exponentially many) possible hypotheses. In essence, inductive logic programming tackles this technical challenge as a search-and-selection problem [33, 52, 61]: In the space of all candidate hypotheses, search for a subset of valid hypotheses; then within this subset, select a valid hypothesis that maximizes or minimizes a pre-defined objective function (e.g. select _H_ that minimizes Kolmogorov complexity). 

In contrast, algebraic machine reasoning is intrinsically not search-based. When solving the RPM task, the new patterns discovered are computed via algebraic computations. For the RPM answer generation task, we generate answers to RPM instances not by searching for valid answers in the (huge) space of all candidate answers, but by directly computing our answer. Pattern discovery becomes computing: We are computing various primary decompositions, and computing which of the primary components are contained in attribute concepts. For the RPM answer selection task, we solve it as a “compute-and-select” problem. For the RPM answer generation task, we compute attribute values from the extracted patterns wherever possible, and randomly select attribute values for those attributes of entities not involved in the extracted patterns. 

When solving RPMs, if we try to frame the reasoning process of our algebraic approach in a language similar to inductive logic programming, then have the following: Given an RPM instance represented as a concept matrix **J** , we extract patterns _P_ 1 _,_ 2( **J** ) from the first two rows of **J** ; here _P_ 1 _,_ 2( **J** ) takes the role of “background knowledge”. We then systematically check if the extracted patterns conflict with concepts _J_ 3 _,_ 1 _, J_ 3 _,_ 2 (representing the two panels in the 3rd row of the question matrix); the patterns that can be extracted from [ _J_ 3 _,_ 1 _, J_ 3 _,_ 2] take the role of “examples”. Finally, we wish to discover a suitable concept _J_ 3 _,_ 3 for the missing 9-th panel; here _J_ 3 _,_ 3 takes the role of “hypothesis”. 

However, a closer look would reveal several disparities in our attempted analogy. We seem to have analogous notions for “background knowledge”, “examples” and “hypothesis” from the set-up of inductive logic programming, yet what we call “background knowledge” (i.e. the extracted patterns _P_ 1 _,_ 2( **J** )) is not given prior knowledge, but 

26 

must be computed. Similarly, the “examples” are not given prior knowledge, but are extracted patterns from [ _J_ 3 _,_ 1 _, J_ 3 _,_ 2] that must be computed. To discover a “hypothesis”, we are not searching for it, but computing it. Effectively, the common missing ingredient for making this “analogy” proper is algebraic computations, which is the essence of algebraic machine reasoning. 

## **C.2. Potential societal impact of algebraic machine reasoning** 

Algebraic machine reasoning could potentially help to automate “easier” reasoning tasks currently performed by humans. In education, algebraic machine reasoning could help with the design of better intelligence tests. In finance, our framework could help in processing personal bank loan applications or detecting fraud, based on invariant features of such cases. A negative implication, however, could also be present. Our reasoning process is based on extracting patterns from a few examples (as in the RPM task) and then generalizing. If the cases of bank loan applications or fraud have already been unfairly associated with certain socioeconomic groups, then that inequity would be propagated even in our reasoning framework. 

Another potential downstream application of our framework is in medical diagnostics. Diagnostic decision making, based on a patient’s reported symptoms and past medical history, could be modeled as reasoning tasks. Specific symptoms and medical conditions could be encoded as concepts and a final medical diagnosis could potentially be computed. If properly implemented, an algebraic machine reasoning framework for medical diagnosis would significantly speed up the diagnostic decision-making process, as well as reduce the cost of medical diagnostics. Unfortunately, such an application, while bringing clear benefits, also comes with ethical concerns. If a wrong medical diagnosis is made, when a doctor uses algebraic machine reasoning to aid in diagnostic decision-making, who gets the blame? Will the creators of the reasoning framework be liable for legal action? 

Biasness could also be inherent in the _human_ encoding process of a machine reasoning framework. If concepts encoded reflect extant human bias, then reasoning output could be flawed, yet seemingly “reasonable”. For any future application of algebraic machine reasoning to legal or criminal cases, care must be taken in assessing and processing information (concepts) associated with each counter-party. Though fairness is outside the scope of our framework, our approach could highlight potential misuses. 

Lastly, since algebraic machine reasoning is able to surpass human performance on the RPM task (an intelligence test originally designed to evaluate geeral human intelligence and abstract reasoning), our work is effectively cracking open one of the biggest bastions of human cognition. 

This may inadvertently contribute towards the fear by the general public, of AI outcompeting humans. One way to mitigate this is through education and outreach to demystify algebraic machine reasoning. 

## **References** 

- [1] J. Abbott, A. M. Bigatti, and L. Robbiano. CoCoA: a system for doing Computations in Commutative Algebra. Available at http://cocoa.dima.unige.it. 12 

- [2] Ibrahim Abdelaziz, Maxwell Crouse, Bassem Makni, Vernon Austel, Cristina Cornelio, Shajith Ikbal, Pavan Kapanipathi, Ndivhuwo Makondo, Kavitha Srinivas, Michael Witbrock, et al. Learning to guide a saturation-based theorem prover. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022. 6, 26 

- [3] Krzysztof R Apt. Logic programming. _Handbook of Theoretical Computer Science, Volume B: Formal Models and Sematics (B)_ , 1990:493–574, 1990. 26 

- [4] Dave Bayer, Andr´e Galligo, and Mike Stillman. Gr¨obner bases and extension of scalars. In _Computational algebraic geometry and commutative algebra (Cortona, 1991)_ , Sympos. Math., XXXIV, pages 198–215. Cambridge Univ. Press, Cambridge, 1993. 3, 13 

- [5] Yaniv Benny, Niv Pekar, and Lior Wolf. Scale-localized abstract reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12557–12565, 2021. 2 

- [6] L´eon Bottou. From machine learning to machine reasoning. _Machine learning_ , 94(2):133–149, 2014. 6 

- [7] Susan Carey. The origin of concepts. _Journal of Cognition and Development_ , 1(1):37–41, 2000. 15, 16 

- [8] Patricia A Carpenter, Marcel A Just, and Peter Shell. What one intelligence test measures: a theoretical account of the processing in the Raven progressive matrices test. _Psychological review_ , 97(3):404, 1990. 1 

- [9] John B Carroll et al. _Human cognitive abilities: A survey of factor-analytic studies_ . Number 1. Cambridge University Press, 1993. 1 

- [10] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. _arXiv preprint arXiv:1906.07155_ , 2019. 19 

- [11] Kai Fong Ernest Chong. A closer look at the approximation capabilities of neural networks. In _International Conference on Learning Representations_ , 2020. 2 

- [12] David A Cox, John Little, and Donal O’Shea. _Ideals, varieties, and algorithms: an introduction to computational algebraic geometry and commutative algebra_ . Springer, 2015. 2, 9, 10, 11, 12 

- [13] Ian Davidson and Peter B. Walker. Towards fluid machine intelligence: Can we make a gifted AI? _Proceedings of the AAAI Conference on Artificial Intelligence_ , 33(01):9760– 9764, 2019. 1 

- [14] Wolfram Decker, Gert-Martin Greuel, Gerhard Pfister, and Hans Sch¨onemann. SINGULAR 4-3-0 — A computer alge- 

27 

bra system for polynomial computations. http://www. singular.uni-kl.de, 2022. 12 

- [15] Mathias Drton, Bernd Sturmfels, and Seth Sullivant. _Lectures on algebraic statistics_ , volume 39. Springer Science & Business Media, 2008. 2 

- [16] Eliana Duarte, Orlando Marigliano, and Bernd Sturmfels. Discrete statistical models with rational maximum likelihood estimator. _Bernoulli_ , 27(1):135–154, 2021. 2 

- [17] David Eisenbud. _Commutative algebra_ , volume 150 of _Graduate Texts in Mathematics_ . Springer-Verlag, New York, 1995. With a view toward algebraic geometry. 9, 10, 12, 13 

- [18] David Eisenbud, Craig Huneke, and Wolmer Vasconcelos. Direct methods for primary decomposition. _Invent. Math._ , 110(2):207–235, 1992. 12 

- [19] The GAP Group. _GAP – Groups, Algorithms, and Programming, Version 4.12.1_ , 2022. https://www.gap-system.org. 12 

- [20] Edmund Gettier. Is justified true belief knowledge? In _Arguing about knowledge_ , pages 14–15. Routledge, 2020. 15 

- [21] Patrizia Gianni, Barry Trager, and Gail Zacharias. Gr¨obner bases and primary decomposition of polynomial ideals. _J. Symbolic Comput._ , 6(2-3):149–167, 1988. Computational aspects of commutative algebra. 12 

- [22] Alison Gopnik and Henry M Wellman. The theory theory. In _An earlier version of this chapter was presented at the Society for Research in Child Development Meeting, 1991._ Cambridge University Press, 1994. 15 

- [23] Richard E. Grandy. Understanding and the principle of compositionality. _Philosophical Perspectives_ , 4:557–572, 1990. 15 

- [24] Daniel R. Grayson and Michael E. Stillman. Macaulay2, a software system for research in algebraic geometry. Available at http://www.math.uiuc.edu/Macaulay2/. 12 

- [25] Serkan Hos¸ten and Gregory G Smith. Monomial ideals. In _Computations in algebraic geometry with Macaulay 2_ , pages 73–100. Springer, 2002. 13 

- [26] Sheng Hu, Yuqing Ma, Xianglong Liu, Yanlu Wei, and Shihao Bai. Stratified rule-aware network for abstract visual reasoning. In _Proceedings of AAAI Conference on Artificial Intelligence (AAAI)_ , 2021. 2, 7, 16 

- [27] Geoffrey Irving, Christian Szegedy, Alexander A Alemi, Niklas E´en, Franc¸ois Chollet, and Josef Urban. Deepmathdeep sequence models for premise selection. In _Advances in Neural Information Processing Systems_ , pages 2235–2243, 2016. 6, 26 

- [28] Joxan Jaffar and J-L Lassez. Constraint logic programming. In _Proceedings of the 14th ACM SIGACT-SIGPLAN symposium on Principles of programming languages_ , pages 111– 119, 1987. 6, 26 

- [29] Marius Jahrens and Thomas Martinetz. Solving Raven’s progressive matrices with multi-layer relation networks. _arXiv preprint arXiv:2003.11608_ , 2020. 1, 2 

- [30] Hans Kamp and Barbara Partee. Prototype theory and compositionality. _Cognition_ , 57(2):129–191, 1995. 15 

- [31] Joa Kileel, Matthew Trager, and Joan Bruna. On the expressive power of deep polynomial neural networks. _Advances in neural information processing systems_ , 2019. 2 

- [32] Youngsung Kim, Jinwoo Shin, Eunho Yang, and Sung Ju Hwang. Few-shot visual reasoning with meta-analogical contrastive learning. _Neurips 2020_ , 2020. 1 

- [33] Tim Kimber, Krysia Broda, and Alessandra Russo. Induction on failure: Learning connected horn theories. In _International Conference on Logic Programming and Nonmonotonic Reasoning_ , pages 169–181. Springer, 2009. 26 

- [34] Franz J Kir´aly, Paul Von B¨unau, Frank C Meinecke, Duncan AJ Blythe, Klaus-Robert M¨uller, and Kenji Fukumizu. Algebraic geometric comparison of probability distributions. _Journal of Machine Learning Research_ , 13(3), 2012. 2 

- [35] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Ingredients of intelligence: From classic debates to an engineering roadmap. _Behavioral and Brain Sciences_ , 40, 2017. 15 

- [36] Guillaume Lample and Franc¸ois Charton. Deep learning for symbolic mathematics. In _International Conference on Learning Representations_ , 2020. 6, 26 

- [37] Barbara Landau. Will the real grandmother please stand up? the psychological reality of dual meaning representations. _Journal of Psycholinguistic Research_ , 11(1):47–62, 1982. 15 

- [38] Paul Larsen and Franz Kir´aly. Fano schemes of generic intersections and machine learning. _International Journal of Algebra and Computation_ , 24(07):923–933, 2014. 2 

- [39] Wenda Li, Lei Yu, Yuhuai Wu, and Lawrence C Paulson. Isarstep: a benchmark for high-level mathematical reasoning. In _International Conference on Learning Representations_ , 2021. 6, 26 

- [40] Shaowei Lin. _Algebraic methods for evaluating integrals in Bayesian statistics_ . PhD thesis, UC Berkeley, 2011. 2 

- [41] John W Lloyd. _Foundations of logic programming_ . Springer Science & Business Media, 2012. 26 

- [42] John Locke. An essay concerning human understanding, 1690. 1948. 15, 16 

- [43] Muyang Lyu, Ruixuan Liu, and Junyi Wang. Solving raven’s progressive matrices using rnn reasoning network. In _2022 7th International Conference on Computational Intelligence and Applications (ICCIA)_ , pages 32–37. IEEE, 2022. 2 

- [44] Barbara C Malt and Edward E Smith. Correlated properties in natural categories. _Journal of verbal learning and verbal behavior_ , 23(2):250–269, 1984. 15 

- [45] Petros Maragos, Vasileios Charisopoulos, and Emmanouil Theodosis. Tropical geometry and machine learning. _Proceedings of the IEEE_ , 2021. 2 

- [46] Eric Ed Margolis and Stephen Ed Laurence. _Concepts: Core Readings._ The MIT Press, 1999. 15, 16 

- [47] Douglas L Medin, Edward E Smith, et al. Concepts and concept formation. _Annual review of psychology_ , 35(1):113– 138, 1984. 15 

- [48] Ezra Miller and Bernd Sturmfels. _Combinatorial commutative algebra_ , volume 227 of _Graduate Texts in Mathematics_ . Springer-Verlag, New York, 2005. 10 

- [49] Melanie Mitchell. Abstraction and analogy-making in artificial intelligence. _arXiv preprint arXiv:2102.10717_ , 2021. To appear in Annals of the New York Academy of Sciences. 1 

- [50] Melanie Mitchell. Why AI is harder than we think. _arXiv preprint arXiv:2104.12871_ , 2021. 1 

28 

- [51] Stephen Muggleton. Inductive logic programming. _New generation computing_ , 8(4):295–318, 1991. 26 

- [52] Stephen Muggleton. Inverse entailment and progol. _New generation computing_ , 13(3):245–286, 1995. 26 

- [53] Gregory Murphy. _The big book of concepts_ . MIT press, 2004. 15 

- [54] Aditya Paliwal, Sarah Loos, Markus Rabe, Kshitij Bansal, and Christian Szegedy. Graph representations for higherorder logic and theorem proving. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 34(03):2967–2974, Apr. 2020. 26 

- [55] Niv Pekar, Yaniv Benny, and Lior Wolf. Generating correct answers for progressive matrices intelligence tests. _Advances in Neural Information Processing Systems_ , 33:7390–7400, 2020. 2 

- [56] Patrick Perret. Children’s inductive reasoning: Developmental and educational perspectives. _Journal of Cognitive Education and Psychology_ , 14(3):389–408, 2015. 1 

- [57] Sylvain Petitjean. Algebraic geometry and object representation in computer vision. In Martial Hebert, Jean Ponce, Terry Boult, and Ari Gross, editors, _Object Representation in Computer Vision_ , pages 155–165, Berlin, Heidelberg, 1995. Springer Berlin Heidelberg. 2 

- [58] Steven Pinker and Ray Jackendoff. The faculty of language: what’s special about it? _Cognition_ , 95(2):201–236, 2005. 15 

- [59] Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, and Jian Tang. RNNLogic: Learning logic rules for reasoning on knowledge graphs. In _International Conference on Learning Representations_ , 2021. 1 

- [60] Markus Norman Rabe, Dennis Lee, Kshitij Bansal, and Christian Szegedy. Mathematical reasoning via selfsupervised skip-tree training. In _International Conference on Learning Representations_ , 2021. 1 

- [61] Oliver Ray, Krysia Broda, and Alessandra Russo. Hybrid abductive inductive learning: A generalisation of progol. In _International Conference on Inductive Logic Programming_ , pages 311–328. Springer, 2003. 26 

- [62] Adam Santoro, Felix Hill, David Barrett, Ari Morcos, and Timothy Lillicrap. Measuring abstract reasoning in neural networks. In _International Conference on Machine Learning_ , pages 4477–4486, 2018. 1, 2, 7 

- [63] Snejana Shegheva and Ashok Goel. The structural affinity method for solving the Raven’s Progressive Matrices test for intelligence. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2018. 1 

- [64] Fan Shi, Bin Li, and Xiangyang Xue. Raven’s progressive matrices completion with latent gaussian process priors. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 9612–9620, 2021. 2 

- [65] Takeshi Shimoyama and Kazuhiro Yokoyama. Localization and primary decomposition of polynomial ideals. _J. Symbolic Comput._ , 22(3):247–277, 1996. 12 

- [66] Richard P. Stanley. _Combinatorics and commutative algebra_ , volume 41 of _Progress in Mathematics_ . Birkh¨auser Boston, Inc., Boston, MA, second edition, 1996. 10 

- [67] Mike Stillman, Carolyn Yackel, Justin Chen, and Mahrud Sayrafi. PrimaryDecomposition: primary decomposition and 

associated primes routines. Version 2.0. A _Macaulay2_ package available at https://github.com/Macaulay2/ M2/tree/master/M2/Macaulay2/packages. 13 

- [68] J¨urgen Stuber. Superposition theorem proving for commutative rings. In _Automated Deduction—A Basis for Applications_ , pages 31–55. Springer, 1998. 2 

- [69] Bernd Sturmfels. What is . . . a Gr¨obner basis? _Notices Amer. Math. Soc._ , 52(10):1199–1200, 2005. 11 

- [70] Seth Sullivant. Algebraic geometry of gaussian bayesian networks. _Advances in Applied Mathematics_ , 40(4):482–513, 2008. 2 

- [71] Henry Taylor and Peter Vickers. Conceptual fragmentation and the rise of eliminativism. _European Journal for Philosophy of Science_ , 7(1):17–40, 2017. 15 

- [72] The Sage Developers. _SageMath, the Sage Mathematics Software System (Version 9.6)_ , 2022. https://www.sagemath.org. 12 

- [73] Sjoerd Van Steenkiste, Francesco Locatello, J¨urgen Schmidhuber, and Olivier Bachem. Are disentangled representations helpful for abstract visual reasoning? In _Advances in Neural Information Processing Systems_ , pages 14245–14258, 2019. 1 

- [74] Mingzhe Wang and Jia Deng. Learning to prove theorems by learning to generate theorems. In _Advances in Neural Information Processing Systems_ , volume 33, pages 18146– 18157, 2020. 26 

- [75] Po-Wei Wang, Priya Donti, Bryan Wilder, and Zico Kolter. Satnet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver. In _International Conference on Machine Learning_ , pages 6545–6554. PMLR, 2019. 1 

- [76] Sumio Watanabe. Algebraic analysis for nonidentifiable learning machines. _Neural Comput._ , 13(4):899–933, apr 2001. 2 

- [77] Sumio Watanabe. Algebraic geometrical methods for hierarchical learning machines. _Neural Netw._ , 14(8):1049–1060, oct 2001. 2 

- [78] Sumio Watanabe. _Algebraic geometry and statistical learning theory_ . Number 25. Cambridge university press, 2009. 2 

- [79] Sumio Watanabe. A widely applicable bayesian information criterion. _Journal of Machine Learning Research_ , 14(3):867–897, 2013. 2 

- [80] Sumio Watanabe and Manfred Opper. Asymptotic equivalence of bayes cross validation and widely applicable information criterion in singular learning theory. _Journal of machine learning research_ , 11(12), 2010. 2 

- [81] Yuhuai Wu, Albert Jiang, Jimmy Ba, and Roger Baker Grosse. Int: An inequality benchmark for evaluating generalization in theorem proving. In _International Conference on Learning Representations_ , 2021. 6, 26 

- [82] Keisuke Yamazaki and Sumio Watanabe. Singularities in mixture models and upper bounds of stochastic complexity. _Neural Networks_ , 16(7):1029–1038, 2003. 2 

- [83] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and SongChun Zhu. RAVEN: A dataset for relational and analogical visual reasoning. In _Proceedings of the IEEE Conference_ 

29 

_on Computer Vision and Pattern Recognition_ , pages 5317– 5327, 2019. 1, 2, 7, 16 

- [84] Chi Zhang, Baoxiong Jia, Feng Gao, Yixin Zhu, Hongjing Lu, and Song-Chun Zhu. Learning perceptual inference by contrasting. In _Advances in Neural Information Processing Systems_ , pages 1075–1087, 2019. 1, 2, 7 

- [85] Chi Zhang, Baoxiong Jia, Song-Chun Zhu, and Yixin Zhu. Abstract spatial-temporal reasoning via probabilistic abduction and execution. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 2, 7 

- [86] Chi Zhang, Sirui Xie, Baoxiong Jia, Ying Nian Wu, SongChun Zhu, and Yixin Zhu. Learning algebraic representation for systematic generalization in abstract reasoning. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXIX_ , pages 692–709. Springer, 2022. 2 

- [87] Liwen Zhang, Gregory Naitzat, and Lek-Heng Lim. Tropical geometry of deep neural networks. In _International Conference on Machine Learning_ , pages 5824–5832. PMLR, 2018. 2 

- [88] Kecheng Zheng, Zheng-Jun Zha, and Wei Wei. Abstract reasoning with distracting features. In _Advances in Neural Information Processing Systems_ , volume 32, 2019. 2, 7 

- [89] Tao Zhuo, Qiang Huang, and Mohan Kankanhalli. Unsupervised abstract reasoning for raven’s problem matrices. _IEEE Transactions on Image Processing_ , 30:8332–8341, 2021. 2, 7 

- [90] Tao Zhuo and Mohan Kankanhalli. Solving raven’s progressive matrices with neural networks. _arXiv preprint arXiv:2002.01646_ , 2020. 2 

- [91] Tao Zhuo and Mohan Kankanhalli. Effective abstract reasoning with dual-contrast network. In _International Conference on Learning Representations_ , 2021. 2, 7 

- [92] Zsolt Zombori, Josef Urban, and Chad E Brown. Prolog technology reinforcement learning prover. In _International Joint Conference on Automated Reasoning_ , pages 489–507. Springer, 2020. 6, 26 

30 

