Published as a conference paper at ICLR 2025 

# FROM PROBABILITY TO COUNTERFACTUALS: THE INCREASING COMPLEXITY OF SATISFIABILITY IN PEARL’S CAUSAL HIERARCHY 

Julian Dörfler[∗] Saarland University, Germany jdoerfler@cs.uni-saarland.de 

Benito van der Zander[†] University of Lübeck, Germany b.vanderzander@uni-luebeck.de 

Markus Bläser[‡] Saarland University, Germany mblaeser@cs.uni-saarland.de 

Maciej Li´skiewicz[§] University of Lübeck, Germany maciej.liskiewicz@uni-luebeck.de 

## ABSTRACT 

The framework of Pearl’s Causal Hierarchy (PCH) formalizes three types of reasoning: probabilistic (i.e. purely observational), interventional, and counterfactual, that reflect the progressive sophistication of human thought regarding causation. We investigate the computational complexity aspects of reasoning in this framework focusing mainly on satisfiability problems expressed in probabilistic and causal languages across the PCH. That is, given a system of formulas in the standard probabilistic and causal languages, does there exist a model satisfying the formulas? 

Our main contribution is to prove the exact computational complexities showing that languages allowing addition and marginalization (via the summation operator) yield NP[PP] -, PSPACE-, and NEXP-complete satisfiability problems, depending on the level of the PCH. These are the first results to demonstrate a strictly increasing complexity across the PCH: from probabilistic to causal and counterfactual reasoning. On the other hand, in the case of full languages, i.e. allowing addition, marginalization, and multiplication, we show that the satisfiability for the counterfactual level remains the same as for the probabilistic and causal levels, solving an open problem in the field. 

## 1 INTRODUCTION 

The development of the modern causal theory in AI and empirical sciences has greatly benefited from an influential structured approach to inference about causal phenomena, which is based on a reasoning hierarchy named “Ladder of Causation”, also often referred to as the “Pearl’s Causal Hierarchy” (PCH) ((Shpitser & Pearl, 2008; Pearl, 2009; Bareinboim et al., 2022), see also (Pearl & Mackenzie, 2018) for a gentle introduction to the topic). This three-level framework formalizes various types of reasoning that reflect the progressive sophistication of human thought regarding causation. It arises from a collection of causal mechanisms that model the “ground truth” of unobserved nature formalized within a Structural Causal Model (SCM). These mechanisms are then combined with three patterns of reasoning concerning observed phenomena expressed at the corresponding layers of the hierarchy, known as probabilistic (also called associational in the AI literature), interventional, and counterfactual (for formal definitions of these concepts, see Sec. 2). 

A basic term at the probabilistic/associational layer is expressed as a common probability, such as[1] P(x, y). This may represent queries like “How likely does a patient have both diabetes (X = x) 

> ∗Contributing equally first authors. 

> †Contributing equally first authors. 

> ‡Contributing equally last authors. 

> §Contributing equally last authors. 

> 1In our paper, we consider random variables over discrete, finite domains. By an event we mean a propositional formula over atomic events of the form X = x, such as (X = x ∧ Y = y) or (X = x ∨ Y = y). 

1 

Published as a conference paper at ICLR 2025 

and high blood pressure (Y = y)?” From basic terms, we can build more complex terms by using additions (linear terms) or even arbitrary polynomials (polynomial terms). This can be combined with the use of unary summation operator, which allows to express marginalization in a compact way. Formulas at this layer consist of Boolean combinations of (in)equalities of basic, linear or, in the general case, polynomial terms. The interventional patterns extend the basic probability terms by allowing the use of Pearl’s do-operator (Pearl, 2009) which models an experiment like a Randomized Controlled Trial (Fisher, 1936). For instance, P([x]y) which[2] , in general differs from P(y|x), allows to ask hypothetical questions such as, e.g., “How likely is it that a patient’s headache will be cured (Y = y) if he or she takes aspirin (X = x)?”. An example formula at this layer is P([x]y) = �z[P][(][y][|][x, z][)][P][(][z][)][ which estimates the causal effect of the intervention][ do][(][X][=][ x][)][ (all patients take] aspirin) on outcome variable Y = y (headache cure). It illustrates the use of the prominent backdoor adjustment to eliminate the confounding effect of a factor represented by the variable Z (Pearl, 2009). The basic terms at the highest level of the hierarchy enable us to formulate queries related to counterfactual situations. For example, P([X=x]Y =y|(X=x[′] , Y =y[′] )) expresses the probability that, for instance, a patient who did not receive a vaccine (X = x[′] ) and died (Y = y[′] ) would have lived (Y = y) if he or she had been vaccinated (X = x). 

The computational complexity aspects of reasoning about uncertainty in this framework have been the subject of intensive studies in the past decades, especially in the case of probabilistic inference with the input probability distributions encoded by Bayesian networks (see, e.g., (Pearl, 1988; Cooper, 1990; Dagum & Luby, 1993; Roth, 1996; Park & Darwiche, 2004)). The main focus of our work is on the computational complexity of satisfiability problems and their validity counterparts which enable formulating precise assumptions on data and implications of causal explanations. 

The problems take as input a Boolean combination of (in)equalities of terms at the PCH-layer of interest with the task to decide if there exists a satisfying SCM for the input formula or if the formula is valid for all SCMs, respectively. For example, for binary random variables X and Y , the formula consisting of the single equality[�] x �y[P][((][X][=][x][)][ ∧][(][X][=][/ x][ ∨][Y][ =][y][)][ ∧][(][X][=][/ x][ ∨][Y][ =][/ y][)) = 0][ in the] language of the probabilistic layer is satisfied since there exists an SCM in which it is true (in fact, the formula holds in any SCM)[3] . An SCM for inputs at this layer can be identified with the standard joint probability distribution, in our case, with P (X=x, Y =y), for x, y ∈{0, 1}. 

The complexity of the studied satisfiability problems depends on the combination of two factors: (1) the PCH-layer to which the basic terms belong and (2) the operators which can be used to specify the (in)equalities of the input formula. The most basic operators are “+” and “·” (leading to get linear, resp., polynomial terms) and, meaningful in causality, the unary summation operator Σ used to express marginalization. Of interest is also conditioning, which will be discussed in our paper, as well. The main interest of our research is focused on the precise characterization of the computational complexity of satisfiability problems (and their validity counterparts) for languages of all PCH layers, combined with increasing the expressiveness of (in)equalities by enabling the use of more complex operators. 

Related Work to our Study. In their seminal paper, Fagin, Halpern, and Megiddo (1990) explore the language of the lowest probabilistic layer of PCH consisting of Boolean combinations of (in)equalities of basic and linear terms. Besides the complete axiomatization for the used logic, they show that the problem of deciding satisfiability is NP-complete indicating that the complexity is surprisingly no worse than that of propositional logic. The authors subsequently extend the language to include (in)equalities of polynomial terms, aiming to facilitate reasoning about conditional probabilities. While they establish the existence of a PSPACE algorithm for deciding whether such a formula is satisfiable, they leave the exact complexity open. Recently, Mossé, Ibeling, and Icard (2022) resolved this issue by demonstrating that deciding satisfiability is ∃R-complete, where ∃R represents the well-studied class defined as the closure of the Existential Theory of the Reals (ETR) under polynomial-time many-one reductions. Furthermore, for the higher, more expressive PCH layers Mossé et al. prove that for (in)equalities of polynomial terms both at the interventional and 

Moreover, by P(Y =y, X=x), etc., we mean, as usually, P(X=x ∧ Y =y). Finally by P(x, y) we abbreviate P(Y =y, X=x). 

> 2A common and popular notation for the post-interventional probability is P(Y =y|do(X=x)). In this paper, however, we use the notation P([X=x]Y =y) since it is more convenient for our analysis. 

> 3Interestingly, the instance can be seen as a result of reduction from the not-satisfiable Boolean formula a ∧ (a ∨ b) ∧ (a ∨ b). 

2 

Published as a conference paper at ICLR 2025 

the counterfactual layer the decision problems still remain ∃R-complete (we recall the definitions of the complexity classes in Sec. 2.2). 

The languages used in these studies, and also in other works as, e.g., (Nilsson, 1986; Georgakopoulos et al., 1988; Ibeling & Icard, 2020), are able to fully express probabilistic reasoning, resp., inferring interventional and counterfactual predictions. In particular, they allow one to express marginalization which is a common paradigm in this field. However, since the languages do not include the unary summation operator Σ, the abilities of expressing marginalization are relatively limited. Thus, for instance, to express the marginal distribution of a random variable Y over a subset of (binary) variables {Z1, . . . , Zm} ⊆{X1, . . . , Xn} as[�] z1,...,zm[P][(][y, z][1][, . . . , z][m][)][,][an][encoding] without summation requires an expansion P(y, Z1=0, . . . , Zm=0) + . . . + P(y, Z1=1, . . . , Zm=1) of exponential size in m. Consequently, to analyze the complexity aspects of the problems under study, languages allowing standard notation for encoding marginalization using the Σ operator are needed. In (van der Zander et al., 2023), the authors present a first systematic study in this setting. They introduce a new natural complexity class, named succ-∃R, which can be viewed as a succinct variant of ∃R, and show that the satisfiability for the (in)equalities of polynomial terms, both at the probabilistic and interventional layer, are complete for succ-∃R. They leave open the exact complexity for the counterfactual case. Moreover, the remaining variants (basic and linear terms) remain unexplored for all PCH layers. 

Our Contribution. The previous research establishes that, from a computational perspective, many problems for the interventional and counterfactual reasoning are not harder than for pure probabilistic reasoning. In our work, we show that the situation changes significantly if, to express marginalization, the common summation operator is used. Below we highlight our main contributions, partially summarized also in Table 1 which involve complexity classes related to each other as follows[4] : 

NP ⊆∃R, NP[PP] ⊆ PSPACE ⊆ NEXP ⊆ succ-∃R ⊆ EXPSPACE (1) 

• For combinations of (in)equalities of basic and linear terms, unlike previous results, the compact summation for marginalization increases the complexity, depending on the level of the PCH: from NP[PP] -, through PSPACE-, to NEXP-completeness. 

• The counterfactual satisfiability for (in)equalities of polynomial terms is succ-∃R-complete, which solves the open problem in (van der Zander et al., 2023). 

• Accordingly, the validity problems for the languages above are complete for the corresponding complement 

|Terms|L1 (prob.)|L2 (interv.)|L3 (count.)|
|---|---|---|---|
|basic<br>lin<br>poly||NP (a)<br>∃R (b)||
|basic & marg.<br>lin & marg.|NPPP (1)|PSPACE(2)|NEXP(3)|
|poly & marg.||succ-∃R (c,4)||



Table 1: Completeness results for the satisfiability problems (a) for L1 (Fagin et al., 1990), for L2 and L3 (Mossé et al., 2022), (b) (Mossé et al., 2022), (c) for L1 and L2 (van der Zander et al., 2023). Our results (1)-(4): Theorem 4, 8, 9, resp. Theorem 10. 

complexity classes. Interestingly, both satisfiability and validity for basic and linear languages with marginalization are PSPACE-complete at the interventional layer. 

Our results demonstrate, for the first time, a strictly increasing complexity of reasoning across the PCH – from probabilistic to causal to counterfactual reasoning – under the widely accepted assumption that the inclusions NP[PP] ⊆ PSPACE ⊆ NEXP in Eq. (1) above are proper. This relation, in the case of basic and linear languages with marginalization, aligns with the strength of their expressive power: From previous research we know that the probabilistic languages are less expressive than the causal languages, and the causal languages, in turn, are less expressive than the corresponding counterfactual languages (for more discussion on this, see Sec. 3). 

In addition, the impact of establishing exact completeness results for probabilistic, causal, and counterfactual reasoning, as stated in our work, lies in their implications for algorithmic approaches to solve these problems. Under widely accepted complexity assumptions like, e.g., NP ̸= PSPACE, the completeness of a problem highlights inherent limitations in applying algorithmic techniques, 

> 4The relationship between ∃R and NPPP is unknown. 

3 

Published as a conference paper at ICLR 2025 

such as dynamic programming, divide-and-conquer, SAT- or ILP-solvers, which are only effective for NP-complete problems. This, in turn, justifies the use of heuristics or algorithms of exponential worst-case complexity. Moreover, using the succ-∃R-completeness as a yardstick for measuring computational complexity of problems, we show that the complexity of counterfactual reasoning (for the most general queries) remains the same as for common probabilistic reasoning. This is quite a surprising result, as the difference between the expressive power of both settings is huge. 

Structure of the Paper. In Sec. 2, we provide the main concepts of causation and define formally problems considered in this work. We derive the complexity of satisfiability for basic and linear languages in Sec. 3 and for polynomial languages in Sec. 4. Due to space constraints, some proofs are omitted from the main text, and only proof outlines are provided. The complete proofs can be found in Sec. A in the appendix. Furthermore, Sec. B in the appendix, provides an example illustrating the three types of reasoning in the framework of PCH and in Sec. C we give formal definitions for syntax and semantics of the languages of the hierarchy. 

## 2 PRELIMINARIES 

In this section, we give definitions of the main concepts of the theory of causation, including the Structural Causal Model (SCM), provide syntax and semantics for the languages of the PCH, and discuss the complexity classes used in the paper. To help understand the formal definitions, we encourage the reader unfamiliar with the theory of causation to refer to Section B in the appendix, where we provide an example that, we hope, will make it easier to understand the formal definitions and the intuitions behind them. 

## 2.1 THE LANGUAGES OF CAUSAL HIERARCHY 

We give here an informal but reasonably precise description of the syntax and semantics of the languages studied in this paper. For formal definitions, see Section C in the appendix. 

We always consider discrete distributions and represent the values of the random variables as Val = {0, 1,..., c − 1}. We denote by X the set of variables used in a system and by capital letters X1, X2,..., we denote the individual variables. We assume that Val is fixed and of cardinality at least two, and that all variables Xi share the same domain Val. A value of Xi is often denoted by xi or a natural number. By an atomic event, we mean an event of the form X = x, where X is a random variable and x is a value in the domain of X. The language Eprop of propositional formulas δ over atomic events is defined as the closure of such events under the Boolean operators ∧ and ¬. The atomic intervention is either empty ⊥ or of the form X = x. An intervention formula is a conjunction of atomic interventions. The language of post-interventional events, denoted as Epost-int, consists the formulas of the form [α]δ where α is an intervention and δ is in Eprop. The language of counterfactual events, Ecounterfact, is the set Epost-int closed under ∧ and ¬. 

The PCH consists of languages on three layers each of which is based on terms of the form P(δi), with i = 1, 2, 3. For the observational (associational) language (Layer 1), we have δ1 ∈Eprop, for the interventional language (Layer 2), we have δ2 ∈Epost-int, and, for the counterfactual language (Layer 3), δ3 ∈Ecounterfact. The expressive power and computational complexity properties of the languages depend largely on the operations that are allowed to apply on terms P(δi). Allowing gradually more complex operators, we define the languages which are the subject of our studies. The terms for levels i = 1, 2, 3 are described as follows. The basic terms, denoted as Ti[base] , are probabilities P(δi) as, e.g., P(X1=x1 ∨ X2=x2) in T1[base] or P([X1=x1]X2=x2) in T2[base] . From basic terms, we build more complex linear terms Ti[lin] by using additions and polynomial terms Ti[poly] by using arbitrary polynomials. By Ti[base][⟨][Σ][⟩] , Ti[lin][⟨][Σ][⟩] , and Ti[poly][⟨][Σ][⟩] , we denote the corresponding sets of terms when including a unary marginalization operator of the form[�] x[t][for][a][term][t][.][In] the summation, we have a dummy variable x which ranges over all values 0, 1, . . ., c − 1. The summation[�] x[t][is][a][purely][syntactical][concept][which][represents][the][sum][t][[][0][/][x][]][+][ t][[][1][/][x][] +][ ...][+] t[[c][ −][1] /x], where by t[[v] /x], we mean the expression in which all occurrences of x are replaced with value v. For example, for Val = {0, 1}, the expression[�] x[P][(][Y][ =1][, X][=][x][)][ semantically represents] P(Y =1, X=0) + P(Y =1, X=1). 

Now, let Lab = {base, base⟨Σ⟩, lin, lin⟨Σ⟩, poly, poly⟨Σ⟩} denote the labels of all variants of languages. Then for each ∗∈ Lab and i = 1, 2, 3, we define the languages L[∗] i[of Boolean combinations] of inequalities in a standard way by the grammars: f ::= t ≤ t[′] | ¬f | f ∧ f where t, t[′] are terms in Ti[∗][.][Although the languages and their operations can appear rather restricted, all the usual elements] 

4 

Published as a conference paper at ICLR 2025 

of probabilistic and causal formulas can be encoded in a natural way (see Section C in the appendix for more discussion). 

To define the semantics, we use SCMs as in (Pearl, 2009, Sec. 3.2). An SCM is a tuple M = (F , P, U, X), with exogenous variables U and endogenous variables X = {X1, . . . , Xn}. F = {F1,..., Fn} consists of functions such that Fi calculates the value of variable Xi from the values (x, u) as Fi(pai, ui), where[5] , Pai ⊆ X and Ui ⊆ U. Pai are all endogenous variables that influence Xi and pai are their values[6] . P specifies a probability distribution of all exogenous variables U. Without loss of generality, we assume that the domains of exogenous variables are also discrete and finite. The functions Fi are deterministic, i.e., the value of every endogenous variable is uniquely determined given the values of the exogenous variables. Since the exogenous variables follow a probability distribution, this implies a probability distribution over the endogenous variables. 

For any basic intervention formula [Xi=xi] (which is our notation for Pearl’s do-operator do(Xi=xi)), we denote by FXi=xi the functions obtained from F by replacing Fi with the constant function Fi(pai, ui) := xi. We generalize this definition for any intervention α in a natural way and denote as Fα the resulting functions. For any ϕ ∈Eprop, we write F , u |= ϕ if ϕ is satisfied for the values of X calculated from the values u. For any intervention α, we write F , u |= [α]ϕ if Fα, u |= ϕ. And for all ψ, ψ1, ψ2 ∈Ecounterfact, we write (i) F , u |= ¬ψ if F , u ̸|= ψ and (ii) F , u |= ψ1 ∧ ψ2 if F , u |= ψ1 and F , u |= ψ2. Finally, for ψ ∈Ecounterfact, let SM(ψ) = {u | F , u |= ψ} be the set of values of U satisfying ψ. For some expression e, we define the value �e�M of the expression e given a model M, recursively in a natural way, starting with basic terms as follows �P(ψ)�M =[�] u∈SM(ψ)[P][(][u][)][and,][for][δ][∈E][prop][,] �P(ψ|δ)�M = �P(ψ ∧ δ)�M/�P(δ)�M, assuming that the expression is undefined if �P(δ)�M = 0. We will sometimes write PM(ψ) instead of �P(ψ)�M, for short. For two expressions e1 and e2, we define M |= e1 ≤ e2, iff, �e1�M ≤ �e2�M. The semantics for negation and conjunction are defined in the usual way, giving the semantics for M |= ϕ for any formula ϕ in L[∗] 3[[.]] 

3[[.]] 

## 2.2 SATISFIABILITY FOR PCH LANGUAGES AND RELEVANT COMPLEXITY CLASSES 

The (decision) satisfiability problems for languages of PCH, denoted by SAT[∗] Li[, with][ i][ = 1][,][ 2][,][ 3][ and] ∗∈ Lab, take as input a formula ϕ in L[∗] i[and ask whether there exists a model][ M][ such that][ M][ |][=][ ϕ][.] Analogously, the validity problems for L[∗] i[consists in deciding whether, for a given][ ϕ][,][ M][ |][=][ ϕ][ holds] for all models M. From the definitions, it is obvious that variants of the problems for the level i are at least as hard as their counterparts at a lower level. 

We note, that the satisfiability problem (and its complement, the validity problem) does not assume anything about SCMs, including their structure. However, our languages allow queries of the form ψ ⇒ ϕ, which enable us to verify satisfiability, resp., the validity, for the formula ϕ in SCMs which satisfy properties expressed by the formula ψ, whereby, e.g., ψ can encode a graph structure of the model. Thus, the formalism used in our work allows for the formulation of a wide range of queries. 

To measure the computational complexity of SAT[∗] Li[, a central role play the following, well-known] Boolean complexity classes NP, PSPACE, NEXP, and EXPSPACE (for formal definitions see, e.g., Arora & Barak (2009)). Recent research has shown that the precise complexity of several natural satisfiability problems can be expressed in terms of the classes over the real numbers ∃R and succ-∃R. For a comprehensive compendium on ∃R, see Schaefer et al. (2024). Recall, that the existential theory of the reals (ETR) is the set of true sentences of the form 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0005-09.png)


where ϕ is a quantifier-free Boolean formula over the basis {∨, ∧, ¬} and a signature consisting of the constants 0 and 1, the functional symbols + and ·, and the relational symbols <, ≤, and =. The sentence is interpreted over the real numbers in the standard way. The theory forms its own complexity class ∃R which is defined as the closure of ETR under polynomial time manyone reductions (Grigoriev & Vorobjov, 1988; Canny, 1988; Schaefer, 2009; Schaefer et al., 2024). A succinct variant of ETR, denoted as succ-ETR, and the corresponding class succ-∃R, have been introduced by van der Zander et al. (2023). succ-ETR is the set of all Boolean circuits C that encode 

> 5We consider recursive models, that is, we assume the endogenous variables are ordered such that variable Xi (i.e. function Fi) is not affected by any Xj with j > i. 

> 6SCMs are often represented as graphs, in which case the variables Pai can be represented as the parents of variable Xi. However, the definition using functions does not refer to any graphs. 

5 

Published as a conference paper at ICLR 2025 

a true sentence as in (2) as follows. Assume that C computes a function {0, 1}[N] →{0, 1}[M] . Then {0, 1}[N] represents the node set of the tree underlying ϕ and C(i) is an encoding of the description of node i, consisting of the label of i, its parent, and its two children. The variables in ϕ are x1, . . . , x2N . As in the case of ∃R, to succ-∃R belong all languages which are polynomial time many-one reducible to succ-ETR. 

For two computational problems A, B, we will write A ≤P B if A can be reduced to B in polynomial time, which means A is not harder to solve than B. A problem A is complete for a complexity class C, if A ∈C and, for every other problem B ∈C, it holds B ≤P A. By co-C, we denote the class of all problems A such that its complements A belong to C. 

## 3 THE INCREASING COMPLEXITY OF SATISFIABILITY IN PCH FOR LINEAR LANGUAGES WITH MARGINALIZATION 

The expressive power of the languages L[∗] i[, with][ ∗∈{][base][,][ lin][,][ poly][}][ and the layers][ i][=][1][,][ 2][,][ 3][ has] been the subject of intensive research. It is well known, see e.g. (Pearl, 2009; Bareinboim et al., 2022; Mossé et al., 2022; Suppes & Zanotti, 1981), that they form strict hierarchies along two dimensions: First, on each layer i, the languages L[base] i , L[lin] i[,][ and][ L][poly] i have increasing expressiveness; Second, for every ∗∈{base, lin, poly} 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0006-05.png)


where the proper inclusion means that the language L[∗] i[is less expressive than][ L][∗] i+1[.][Note that since] adding marginalization does not change the expressiveness of the language L[∗] i[, the strict inclusions] in (3) hold also for ∗∈{base⟨Σ⟩, lin⟨Σ⟩, poly⟨Σ⟩}. 

To prove such a proper inclusion, it suffices to show two SCMs that are indistinguishable in the less expressive language, but that can be distinguished by some formula in the more expressive language. E.g., the SCMs: M = (F , P, U, X) and M[′] = (F[′] , P, U, X), with binary variables U = {U1, U2}, X = {X1, X2}, probabilities P (Ui = 0) = 1/2, and mechanism F : Xi := Ui, resp. F[′] : X1 := U1U2 + (1 − U1)(1 − U2), X2 := U1 + X1(1 − U1)U2, have the same distributions PM(X1, X2) = PM[′] (X1, X2). Thus, M and M[′] are indistinguishable in any language of the probabilistic layer. On the other hand, after the intervention X1 = 1, we get PM([X1=1]X2=1) = 1/2 and PM′ ([X1=1]X2=1) = 3/4. Then, e.g., for the L[base] 2 formula ϕ : P([X1=1]X2=1) = P([X1=1]X2=0), we have M |= ϕ, but M[′] ̸|= ϕ which means that ϕ distinguishes M from M[′] . 

This section focuses on the basic and linear languages across the PCH (the case of polynomial languages will be discussed in Sec. 4 separately). As mentioned in the introduction, a comparison of these languages from the perspective of computational complexity reveals surprisingly different properties than the ones described above. For basic and linear languages disallowing marginalization, the satisfiability for the counterfactual level remains the same as for the probabilistic and causal levels: problems SAT[base] Li and SAT[lin] Li[for all][i][=][1][,][ 2][,][ 3][ are][ NP][-complete, i.e.,][as][hard as reasoning] about propositional logic formulas ((Fagin et al., 1990; Mossé et al., 2022), cf. also Table 1). In this section, we show that the situation changes drastically when marginalization is allowed: satisfiability problems SATL[base] i[⟨][Σ][⟩] and SATL[lin] i[⟨][Σ][⟩] became NP[PP] -, PSPACE-, resp. NEXP-complete depending on the level i. This demonstrates the first strictly increasing complexity of reasoning across the PCH, assuming the widely accepted assumption that the inclusions NP[PP] ⊆ PSPACE ⊆ NEXP are proper. 

## 3.1 THE PROBABILISTIC (OBSERVATIONAL) LEVEL 

Marginalization via the summation operator, combined with the language expressing events δ on a specific level of PCH, increases the complexity of reasoning to varying degrees, depending on the level. In the probabilistic case, it jumps from NP- to NP[PP] -completeness since the atomic terms P(δ) can contain Boolean formulas δ, which, combined with the summation operator, allows to count the number of all satisfying assignments of a Boolean formula by summing over all possible values for the random variables in the formula. Determining this count is the canonical PP-complete problem, so evaluating the equations given a model is PP-hard. From this, together with the need of finding a model, we will conclude in this section that SATL[base] 1[⟨][Σ][⟩] and SATL[lin] 1[⟨][Σ][⟩] are NP[PP] -complete. We start with a technical but useful fact that a sum in the probabilistic language can be partitioned into a sum over probabilities and a sum over purely logical terms. This generalizes the property shown by Fagin et al. (1990) in Lemma 2.3 for languages without the summation operator. 

6 

Published as a conference paper at ICLR 2025 

Fact 2. Let δ ∈ Eprop be a propositional formula over variables Xi1 , . . . , Xil . A sum �xi1[. . .][ �] xil[P][(][δ][)][is][equal][to][�] xˆ1[. . .][ �] xˆn[p][x][ˆ] 1[...][x][ˆ] n �xi1[. . .][ �] xil[δ][x][ˆ] 1[...][x][ˆ] n[(][x][i] 1[, . . . , x][i] l[)][where] the range of the sums is the entire domain, pxˆ1...xˆn is the probability of P(X1=ˆx1 ∧ . . . ∧ Xn=ˆxn) and δxˆ1...xˆn (xi1 , . . . , xil ) a function that returns 1 if the implication (X1=ˆx1 ∧ . . . ∧ Xn=ˆxn) → δ(xi1 , . . . , xil ) is a tautology and 0 otherwise. The canonical satisfiability problem SAT, where instances consist of Boolean formulas in propositional logic, plays a key role in a broad range of research fields, since all problems in NP, including many real-world tasks can be naturally reduced to it. As discussed earlier, SAT[base] L1[and][S][AT][lin] L1[do] not provide greater modeling power than SAT. Enabling the use of summation significantly changes this situation: below we demonstrate the expressiveness of SATL[base] 1[⟨][Σ][⟩] , showing that any problem in NP[PP] can be reduced to it in polynomial time. 

Lemma 3. SATL[base] 1[⟨][Σ][⟩] is NP[PP] -hard. 

Proof. The canonical NP[PP] -complete problem E-MajSat is deciding the satisfiability of a formula ψ : ∃x1 . . . xn : #y1 . . . ynφ ≥ 2[n][−][1] , i.e. deciding whether the Boolean formula φ has a majority of satisfying assignments to Boolean variables yi after choosing Boolean variables xi existentially (Littman et al., 1998). We will reduce this to SATL[base] 1[⟨][Σ][⟩] . Let there be 2n random variables X1, . . . , Xn, Y1, . . . , Yn associated with the Boolean variables. Assume w.l.o.g. that these random variables have domain {0, 1}. Let φ[′] be the formula φ after replacing Boolean variable xi with Xi = 0 and yi with Yi = yi. 

Consider the probabilistic inequality ϕ :[�] y1[. . .][ �] yn[P][(][φ][′][)][≥][2][n][−][1][whereby][ 2][n][−][1][is encoded] as[�] x1[. . .][ �] xn−1[P][(][⊤][)][.][The left hand side equals] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0007-05.png)


according to Fact 2. Since the last sum ranges over all values of yi, it counts the number of satisfying assignments to φ given xˆi. Writing this count as #y(φ(ˆx1 . . . ˆxn)), the expression becomes: ˆ ˆ ˆ ˆ �x1[. . .][ �] xn �y1[. . .][ �] yn[p][x][ˆ][1][...][x][ˆ][n][,][y][ˆ][1][...][y][ˆ][n][#][y][(][φ][(ˆ][x][1][ . . .][ ˆ][x][n][))][.] Since #y(φ(ˆx1 . . . ˆxn)) does not depend on yˆ, we can write the expression as �xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n][#][y][(][φ][(ˆ][x][1][ . . .][ ˆ][x][n][))][ whereby][ p][x][ˆ][1][...][x][ˆ][n][=][ �] yˆ1[. . .][ �] yˆn[p][x][ˆ][1][...][x][ˆ][n][,][y][ˆ][1][...][y][ˆ][n][.] If ψ is satisfiable, there is an assignment xˆ1, . . . , ˆxn with #y(φ(ˆx1 . . . ˆxn)) ≥ 2[n][−][1] . If we set pxˆ1...xˆn = 1 and every other probability pxˆ[′] 1[...][x][ˆ] n[′][= 0][,][ ϕ][ is satisfied.] If ϕ is satisfiable, let x[max] 1 , . . . , x[max] n be the assignment that maximizes #y(φ(x[max] 1 . . . x[max] n )). Then 2[n][−][1] ≤ �xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n][#][y][(][φ][(ˆ][x][1][ . . .][ ˆ][x][n][))] ≤ �xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n][#][y][(][φ][(][x] 1[max] . . . x[max] n )) = ��xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n] � #y(φ(x[max] 1 . . . x[max] n )) = #y(φ(x[max] 1 . . . x[max] n )). Hence ψ is satisfied for x[max] 1 . . . x[max] n . 

Theorem 4. For probabilistic reasoning, the satisfiability problems SATL[base] 1[⟨][Σ][⟩] and SATL[lin] 1[⟨][Σ][⟩] , for the basic and linear languages respectively, are NP[PP] -complete. 

Proof idea: Having Lemma 3, it remains to prove that the problem is in NP[PP] . To this aim, we show that any satisfiable instance has a solution of polynomial size: we rewrite the sums �xi1[. . .][ �] xil[P][(][δ][)][in][the][expressions][according][to][Fact][2][which][allows][to][encode][the][instance] as a system of m linear equations with unknown coefficients pxˆ1...xˆn , where m is bounded by the instance size. Such a system has a non-negative solution with at most m entries positive of polynomial size. Then, we guess non-deterministically such solutions and verify its correctness estimating the values[�] xi1[...][ �] xil[δ][x][ˆ][1][x][ˆ][n][(][x][i][1][, .., x][i] l[)][ using the][ PP][ oracle. A full proof, as for all further proof] ideas, can be found in the appendix. 

Remark 5. NP[PP] is the class describing the complexity of another, relevant primitive of the probabilistic reasoning which consists in finding the Maximum a Posteriori Hypothesis (MAP). To study its computational complexity, the corresponding decision problem is defined which asks if for a given Bayesian network B = (G, PB), where probability PB factorizes according to the structure of network G, a rational number τ , evidence e, and some subset of variables Q, there is an instantiation q to Q such that PB(q, e) =[�] y[P][B][(][q][,][ y][,][ e][)][>][τ][.][It][is][well][known][that][the][problem][is] NP[PP] -complete (Roth, 1996; Park & Darwiche, 2004). 

7 

Published as a conference paper at ICLR 2025 

Finally, we draw our attention to the impact of negation on the complexity of reasoning for such languages of the probabilistic layer. The hardness of SATL[lin] 1[⟨][Σ][⟩] depends on negations in the Boolean formulas δ of basic terms P(δ), which make it difficult to count all satisfying assignments. Without negations, marginalization just removes variables, e.g.[�] x[P][(][X][=][x][∧][Y][ =][y][)][ becomes][ P][(][Y][ =][y][)][. This] observation leads to the following: 

Proposition 6. SATL[base] 1[⟨][Σ][⟩] and SATL[lin] 1[⟨][Σ][⟩] are NP-complete if the primitives in Epost-int are restricted to not contain negations. 

## 3.2 THE INTERVENTIONAL (CAUSAL) LEVEL 

In causal formulas, one can perform interventions to set the value of a variable, which can recursively affect the value of all endogenous variables that depend on the intervened variable. This naturally corresponds to the choice of a variable value by an existential or universal quantifier in a Boolean formula, since in a Boolean formula with multiple quantifiers, the value chosen by each quantifier can depend on the values chosen by earlier quantifiers. Thus, an interventional equation can encode a quantified Boolean formula. This makes SATL[lin] 2[⟨][Σ][⟩] PSPACE-hard. As we will show below, it is even PSPACE-complete. 

Lemma 7. SATL[base] 2[⟨][Σ][⟩] is PSPACE hard. 

Proof. We reduce from the canonical PSPACE-complete problem QBF. Let Q1x1 Q2x2 · · · Qnxnψ be a quantified Boolean formula with arbitrary quantifiers Q1, . . . , Qn ∈{∃, ∀}. We introduce Boolean random variables X = {X1, . . . , Xn} to represent the values of the variables and denote by Y = {Y1, . . . Yk} ⊆ X the universally quantified variables. By Lemma 12, we can enforce an ordering X1 ≺ X2 ≺ . . . ≺ Xn of variables, i.e. variable Xi can only depend on variables Xj with j < i. Our only further constraint is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0008-07.png)


where ψ[′] is obtained from ψ be replacing positive literals xi by Xi = 1 and negative literals xi by Xi = 0. 

Suppose the constructed SATL[lin] 2[⟨][Σ][⟩] formula is satisfied by a model M. We show that Q1x1 Q2x2 · · · Qnxnψ is satisfiable. Each probability implicitly sums over all possible values u of the exogenous variables. Fix one such u with positive probability. Combined with X1 ≺ . . . ≺ Xn this implies all random variables now deterministically depend only on any of the previous variables. Equation (4) enforces P([y]ψ[′] ) = 1 for every choice of y and thus simulates the Y being universally quantified. As the existential variables xi, we then choose the value xi of Xi which can only depend on Xj with j < i. The formula ψ[′] and thus ψ is then satisfied due to P([y]ψ[′] ) = 1. 

On the other hand, suppose Q1x1 Q2x2 · · · Qnxnψ is satisfiable, We create a deterministic model M as follows: The value of existentially quantified variables Xi is then computed by the function Fi(x1, . . . , xi−1) defined as the existentially chosen value when the previous variables are set to x1, . . . , xi−1. The values of the universally quantified variables do not matter since we intervene on them before every occurrence. This satisfies the required order of variables and since ψ is satisfied we have P([y]ψ[′] ) = 1 for every choice of the universally quantified variables y, thus satisfying equation (4). 

Theorem 8. For causal reasoning, the satisfiability problems SATL[base] 2[⟨][Σ][⟩] and SATL[lin] 2[⟨][Σ][⟩] , for the basic and linear languages respectively, are PSPACE-complete. 

Proof idea: After Lemma 7, we only have to show that SATL[lin] 2[⟨][Σ][⟩] is in PSPACE. For the proof details, see the appendix. The basic idea is to evaluate interventions one at a time without storing the entire model because each primitive can only contain one intervention. For each sum, we only need to store the total probability of all its primitives, and increment this probability, if a new intervention satisfies some of the primitives. 

As with SATL[lin] 1[⟨][Σ][⟩] , there can only be polynomially many exogenous variable assignments u with non-zero probability pu, which are independent of each other and can be guessed nondeterministically. We can also guess a causal order of the endogenous variables, such that variables can only depend on the variables preceding them in the causal order. This causal order allows one to guess the variables affected by any intervention in a sound way. 

8 

Published as a conference paper at ICLR 2025 

We enumerate all exogenous variable assignments u, each having probability pu. Recursively, we can enumerate all possible interventions α, which yield the values x of the endogenous variables, which—given the exogenous variables—also have probability pu. Then we count for each sum how many of its primitives are satisfied by α and x, and increment the accumulated value of the sum by pu for each. 

After this enumeration, we know the numeric value of every sum, and can verify the resulting equation system. 

## 3.3 THE COUNTERFACTUAL LEVEL 

On the counterfactual level, one can perform multiple interventions, and thus compare different functions of the model to each other. Hence, the formulas can only be evaluated if the entire, exponential-sized model is known. Thus deciding the satisfiability requires exponential time and non-determinism to find the model, making SATL[lin] 3[⟨][Σ][⟩] NEXP-complete. 

Theorem 9. For counterfactual reasoning, the satisfiability problems SATL[base] 3[⟨][Σ][⟩] and SATL[lin] 3[⟨][Σ][⟩] , for the basic and linear languages respectively, are NEXP-complete. 

Proof idea: This NEXP-hardness follows from a reduction from the NEXP-complete problem of checking satisfiability of a Schönfinkel-Bernays sentence to the satisfiability of SATL[base] 3[⟨][Σ][⟩] . These are first-order logic formulas of the form ∃x∀yψ where ψ cannot contain any quantifiers or functions. The x and y are encoded as Boolean random variables, where the existentially quantified x are encoded into the existence of a satisfying SCM, while the universally quantified y are encoded by marginalization, i.e. a condition[�] y[P][([][y][]][ψ][′][)][=][2][n][for][some][formula][ψ][′][derived][from][ψ][and] where n denotes the number of variables y. The counterfactuals now allow us to ensure that the random variables Ri representing the relations within ψ deterministically depend on their respective inputs by comparing whether Ri changes between an intervention on all variables (except Ri) versus an intervention on only its dependencies. Marginalization finally allows us to combine these exponentially many checks for all possible values of all variables into a single equation. 

The containment in NEXP follows from expanding the formulas with sums to exponentially larger formulas without sums. 

## 4 THE COMPLEXITY OF SATISFIABILITY FOR POLYNOMIAL LANGUAGES WITH MARGINALIZATION 

Van der Zander et al. (2023) prove that SAT[poly] L1[⟨][Σ][⟩] and SAT[poly] L2[⟨][Σ][⟩] are complete for succ-∃R whenever the basic terms are allowed to also contain conditional probabilities. They, however, leave open the exact complexity status of SAT[poly] L3[⟨][Σ][⟩] . Here we show that the problem is in succ-∃R, with and without conditional probabilities, which proves its succ-∃R-completeness. 

Theorem 10. SAT[poly] L3[⟨][Σ][⟩] ∈ succ-∃R. This also holds true if we allow the basic terms to contain conditional probabilities. 

Proof idea: Bläser et al. (2024) introduced the NEXPreal machine model, where succ-∃R is precisely the set of all languages decidable by exponential-time non-deterministic real RAMs. Combined with the algorithm proving SAT[poly] L3[∈∃][R][ (however without subtractions or conditional proba-] bilities) from Mossé et al. (2022) and the classification ∃R = NPreal from Erickson et al. (2022), i.e. ∃R being the set of all languages decidable by polynomial-time non-deterministic real RAMs, we can expand the unary sums explicitly and then run the non-deterministic real RAM algorithm for the resulting SAT[poly] L3[instance.][Special care has to be taken in][dealing with subtractions or conditional] probabilities, here we use a trick by Tsaitin, the details of which can be found in Lemma 16 in the appendix. 

We note that Ibeling et al. (2024, Theorem 3) independently obtained a variant of the above result, also using the machine characterization of succ-∃R given by Bläser et al. (2024). 

Remark 11. It can be shown that the hardness proofs for SAT[poly] L1[⟨][Σ][⟩] , SAT[poly] L2[⟨][Σ][⟩] , and SAT[poly] L3[⟨][Σ][⟩] do not need conditional probabilities in the basic terms. 

9 

Published as a conference paper at ICLR 2025 

## 5 DISCUSSION 

This work studies the computational complexities of satisfiability problems for languages at all levels of the PCH. Our new completeness results nicely extend and complement the previous achievements by Fagin et al. (1990), Mossé et al. (2022), van der Zander et al. (2023), and Bläser et al. (2024). The main focus of our research was on languages allowing the use of marginalization which is expressed in the languages by a summation operator Σ over the domain of the random variables. This captures the standard notation commonly used in probabilistic and causal inference. 

A very interesting feature of the satisfiability problems for the full, polynomial languages is the following property. For both variants, with and without summation operators, while the expressive powers of the corresponding languages differ, the complexities of the corresponding satisfiability problems at all three levels of the PCH are the same. Interestingly, the same holds for linear languages without marginalization, too (cf. Table 1.). We find that the situation changes drastically in the case of linear languages allowing the summation operator Σ. One of our main results characterizes the complexities of SATL[lin] 1[⟨][Σ][⟩] , SATL[lin] 2[⟨][Σ][⟩] , and SATL[lin] 3[⟨][Σ][⟩] problems as NP[PP] , PSPACE, and NEXPcomplete, resp. The analogous completeness results hold for SATL[base] i[⟨][Σ][⟩] . 

Another interesting feature is that the completeness results for linear languages are expressed in terms of standard Boolean classes while the completeness of satisfiability for languages involving polynomials over the probabilities requires classes over the reals. 

## ACKNOWLEDGMENTS 

This work was supported by the Deutsche Forschungsgemeinschaft (DFG) grant 471183316 (ZA 1244/1-1). We thank Till Tantau for the key idea that made the proof of the NP[PP] -hardness of SATL[lin] 1[⟨][Σ][⟩] possible. 

10 

Published as a conference paper at ICLR 2025 

## REFERENCES 

- Antonis Achilleos. NEXP-completeness and universal hardness results for justification logic. In Proc. International Computer Science Symposium in Russia (CSR 2015), pp. 27–52. Springer, 2015. 

- Sanjeev Arora and Boaz Barak. Computational complexity: a modern approach. Cambridge University Press, 2009. 

- Elias Bareinboim, Juan D. Correa, Duligur Ibeling, and Thomas Icard. On Pearl’s Hierarchy and the Foundations of Causal Inference, pp. 507–556. Association for Computing Machinery, New York, NY, USA, 2022. 

- Markus Bläser, Julian Dörfler, Maciej Li´skiewicz, and Benito van der Zander. The Existential Theory of the Reals with Summation Operators. In Julián Mestre and Anthony Wirth (eds.), 35th International Symposium on Algorithms and Computation (ISAAC 2024), volume 322 of Leibniz International Proceedings in Informatics (LIPIcs), pp. 13:1–13:19, Dagstuhl, Germany, 2024. Schloss Dagstuhl – Leibniz-Zentrum für Informatik. ISBN 978-3-95977-354-6. doi:10.4230/LIPIcs.ISAAC.2024.13. URL https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ISAAC.2024.13. 

- John Canny. Some algebraic and geometric computations in PSPACE. In Proc. of the 20th ACM Symposium on Theory of Computing (STOC 1988), pp. 460–467. ACM, 1988. 

- Gregory F Cooper. The computational complexity of probabilistic inference using bayesian belief networks. Artificial intelligence, 42(2-3):393–405, 1990. 

- Paul Dagum and Michael Luby. Approximating probabilistic inference in Bayesian belief networks is NP-hard. Artificial Intelligence, 60(1):141–153, 1993. 

- Jeff Erickson, Ivor Van Der Hoog, and Tillmann Miltzow. Smoothing the gap between NP and ER. SIAM Journal on Computing, pp. FOCS20–102, 2022. 

- Ronald Fagin, Joseph Y Halpern, and Nimrod Megiddo. A logic for reasoning about probabilities. Information and Computation, 87(1-2):78–128, 1990. 

- Ronald Aylmer Fisher. Design of experiments. British Medical Journal, 1(3923):554, 1936. 

- George Georgakopoulos, Dimitris Kavvadias, and Christos H. Papadimitriou. Probabilistic satisfiability. Journal of Complexity, 4(1):1–11, 1988. 

- Dima Grigoriev and Nicolai Vorobjov. Solving systems of polynomial inequalities in subexponential time. J. Symb. Comput., 5(1/2):37–64, 1988. 

- Duligur Ibeling and Thomas Icard. Probabilistic reasoning across the causal hierarchy. In Proc. 34th AAAI Conference on Artificial Intelligence (AAAI 2020), pp. 10170–10177. AAAI Press, 2020. 

- Duligur Ibeling, Thomas Icard, and Milan Mossé. On probabilistic and causal reasoning with summation operators. Journal of Logic and Computation, pp. exae068, 2024. 

- Harry R. Lewis. Complexity results for classes of quantificational formulas. Journal of Computer and System Sciences, 21(3):317–353, 1980. 

- Michael L. Littman, Judy Goldsmith, and Martin Mundhenk. The computational complexity of probabilistic planning. Journal of Artificial Intelligence Research, 9:1–36, 1998. 

- Milan Mossé, Duligur Ibeling, and Thomas Icard. Is causal reasoning harder than probabilistic reasoning? The Review of Symbolic Logic, pp. 1–26, 2022. 

- Nils J. Nilsson. Probabilistic logic. Artificial Intelligence, 28(1):71–87, 1986. 

- James D Park and Adnan Darwiche. Complexity results and approximation strategies for MAP explanations. Journal of Artificial Intelligence Research, 21:101–133, 2004. 

11 

Published as a conference paper at ICLR 2025 

Judea Pearl. Probabilistic reasoning in intelligent systems: networks of plausible inference. Morgan Kaufmann, 1988. 

Judea Pearl. Causality. Cambridge University Press, 2009. ISBN 0-521-77362-8. 

- Judea Pearl and Dana Mackenzie. The book of why: the new science of cause and effect. Basic books, 2018. 

- Dan Roth. On the hardness of approximate reasoning. Artificial Intelligence, 82(1-2):273–302, 1996. 

- Marcus Schaefer. Complexity of some geometric and topological problems. In Proc. International Symposium on Graph Drawing (GD 2009), pp. 334–344. Springer, 2009. 

- Marcus Schaefer, Jean Cardinal, and Tillmann Miltzow. The existential theory of the reals as a complexity class: A compendium. arXiv preprint arXiv:2407.18006, 2024. 

- Ilya Shpitser and Judea Pearl. Complete identification methods for the causal hierarchy. Journal of Machine Learning Research, 9:1941–1979, 2008. 

- Patrick Suppes and Mario Zanotti. When are probabilistic explanations possible? Synthese, 48: 191–199, 1981. 

- Benito van der Zander, Markus Bläser, and Maciej Li´skiewicz. The hardness of reasoning about probabilities and causality. In Proc. Joint Conference on Artificial Intelligence (IJCAI 2023), 2023. 

- Junzhe Zhang, Jin Tian, and Elias Bareinboim. Partial counterfactual identification from observational and experimental data. In Proc. International Conference on Machine Learning (ICML 2022), pp. 26548–26558. PMLR, 2022. 

12 

Published as a conference paper at ICLR 2025 

## A TECHNICAL DETAILS AND PROOFS 

In this section we prove the completeness results in Table 1: the first result will follow from Lemma 13 and 3, Result (2) from Lemma 14 and 7, and (3) will follow from Lemma 15. We also show the proof of Proposition 6. 

## A.1 PROOFS OF SECTION 3.1 

We first prove the completeness results in Theorem 4, which will follow from Lemma 13 and 3. 

Proof of Fact 2.[�] xi1[. . .][ �] xil[P][(][δ][)][ is equivalent to] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0013-06.png)


which completes the proof. 

ˆ Since the second sum only depend on x1 . . . ˆxn and not on pxˆ1...xˆn, they can be calculated without knowing the probability distribution. Due to the dependency on xˆ1 . . . ˆxn the expression cannot be simplified further in general. However, if δ contains no constant events like Xij = 0 but only events Xij = xij depending on the summation variables xij , the sum always includes one iteration where the event occurs and c − 1 iterations where it does not. Thus the sum �xi1[. . .][ �] xil[δ][x][ˆ][1][...][x][ˆ][n][(][x][i][1][, . . . , x][i] l[)][ is constant and effectively counts the number of satisfying as-] signments to the formula δ. Since the sum[�] xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n][is][always][1][,][the][original][sum] �xi1[. . .][ �] xil[P][(][δ][)][ also counts the number of satisfying assignments.] 

Lemma 12. In L2 one can encode a causal ordering. 

Proof of Lemma 12. Given (in-)equalities in L2, we add a new variable C and add to each primitive the intervention [C = 0]. This does not change the satisfiability of (in-)equalities. 

Given a causal order Vi1 ≺ Vi2 ≺ . . ., we add c equations for each variable Vij , j > 1: P([C=1, Vij−1 =k]Vij =k) = 1 for k = 1, . . . , c. 

The equations ensure, that if one variable is changed, and C = 1 is set, the next variable in the causal ordering has the same value, thus fixing an order from the first to the last variable. 

Lemma 13. SATL[lin] 1[⟨][Σ][⟩] is in NP[PP] . 

Proof. First we need to show that satisfiable instances have solutions of polynomial size. We write each (sum of a) primitive in the arithmetic expressions as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0013-15.png)


according to Fact 2. 

ˆ The value of all pxˆ1...xˆn can be encoded as a vector ⃗p ∈ R[c][n] . For each x1, . . . , ˆxn, the sum �xi1[. . .][ �] xil[δ][x][ˆ][1][...][x][ˆ][n][(][x][i][1][, . . . , x][i] l[)][ is a constant integer][ ≤][c][l][.][Suppose there are][ m][ such sums or] primitives in the instance, whose values can be encoded as a matrix A ∈ R[m][×][c][n] . 

Then the value of every sum in the instance is given by the vector A⃗p ∈ R[m] . 

There exists a non-negative vector ⃗q ∈ Q[c][n] containing at most m non-zero entries with A⃗q = A⃗p (Lemma 2.5 in Fagin et al. (1990)). By including a constraint[�] xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n][=][1][when] constructing the matrix A, we can ensure that all values in ⃗q are valid probabilities. ⃗q is a polynomial sized solution and can be guessed non-deterministically. 

13 

Published as a conference paper at ICLR 2025 

For this solution, the sum[�] xˆ1[. . .][ �] xˆn[p][x][ˆ][1][...][x][ˆ][n][can then be evaluated (over all non-zero entries] pxˆ1...xˆn) in polynomial time. Each sum[�] xi1[. . .][ �] xil[δ][x][ˆ][1][...][x][ˆ][n][(][x][i][1][, . . . , x][i] l[)][ can be evaluated using] the PP oracle because a PP oracle is equivalent to a #P oracle which can count the number of satisfying assignments of δ. 

Finally, we give the proof of Proposition 6. 

Proof of Proposition 6. In the proof of Lemma 13, we require a PP-oracle to evaluate the sum �xi1[. . .][ �] xilˆ[δ][x][ˆ][1][...][x][ˆ][n][(][x][i][1][, . . . , x] ˆ[i] l[)][,][which][counts][the][number][of][assignments][x][i][1][, . . . , x][i] l[that] make (X1 = x1 ∧ . . . ∧ Xn = xn) → δ(xi1 , . . . , xil ) a tautology for given ˆxi. Without negations, there are also no disjunctions, so δ consists of a conjunction of terms that compare some variable to some constant, Xi = j, or toˆsome variable Xi = xij used in the summation. For any constant Xi = j, we check whether xi = j. If that is false, the number of satisfyingˆassignments to δ is zero. From any condition Xi = xij , we learnˆ thatˆ xij has to be equal to xi. If there is any contradiction, i.e., Xi = xij ∧ Xk = xij and xi = xk, the number of assignments is also zero. This determines the value of each xij occurring in δ and makes δ a tautology. 

It leaves the value of xij not occurring in δ undefined, but those do not affect δ, and can be chosen arbitrarily, so the number of assignments is just multiplied by the size of the domain c for each not-occurring variable. 

This can be evaluated in polynomial time, so the complexity is reduced to NP[P] = NP. 

The problems remain NP-hard since a Boolean formula can be encoded in the language L[base] 1 of combinations of Boolean inequalities. Each Boolean variable x is replaced by P (X) > 0 for a corresponding random variable X. For example, a 3-SAT instance like (xi1,1 ∨¬xi1,2 ∨ xi1,3 ) ∧ (0)satisfiability.xi2∧,1 (∨¬P (Xxii22,,21 )∨¬>x0i2,∨¬3 ) ∧P. . .(X can be encoded asi2,2 ) > 0 ∨¬P (X (Pi2,(3X) i1>,1 )0)> ∧0 ∨¬. . ., Pwhich(Xi1,2clearly) > 0 ∨hasP (theXi1,same3 ) > 

A.2 PROOFS OF SECTION 3.2 

Lemma 14. SATL[lin] 2[⟨][Σ][⟩] is in PSPACE. 

Proof. We need to show that Algorithm 1 is correct and in PSPACE. The basic idea of the algorithm is that rather than guessing a model and evaluating each sum with its interventions[7] , we enumerate all possible interventions (and resulting values) and increment each sum that includes the intervention. Thereby, rather than storing the functions and interventions, we only need to store and update the value of the sums. 

By definition, each sum[�] yi[P][([][α][i][]][δ][i][)][in][the][input][can][be][written][as][�] u[p][u] �yi:F ,u|=[αi]δi[1][,] where the second sum does not depend on pu. As in Lemma 13, one can write the probabilities as a single vector ⃗pu, each sum[�] yi:F ,u|=[αi]δi[1][ as row in a matrix][ A][, such that an entry of the row is][ 1] if F , u |= [αi]δi holds and 0 otherwise. Then one obtains the value of all sums as product A ⃗pu. A small model property[8] follows that there are only polynomial many, rational probabilities pu. These can be guessed non-deterministically[9] . 

> 7For example, a sum like[�] x[P][([][X][=][x][]][Y][ =][y][)][ performs][multiple interventions][on][X][,][which][is difficult to] evaluate. Sums containing only a single intervention could be evaluated trivially. 8Small model property means that whenever there is a satisfying probability distribution, then there is also one with only polynomially many positive elementary probabilities. Note that, e.g., the probability distribution on n binary random variables has 2[n] entries. Some of the previous results we refer to, e.g., Fagin et al. (1990); Ibeling & Icard (2020), are based on this small property. 

> 9The algorithm just guesses them without doing any rewriting of the equations. This paragraph just explains why this guessing is possible. 

14 

Published as a conference paper at ICLR 2025 

Input: SATL[lin] 2[⟨][Σ][⟩] instance Output: Is the instance satisfiable? 1 Guess small model probabilities pu; 2 Guess a causal order X1, . . . , Xn; 3 Rewrite each sum[�] yi[P][([][α][i][]][δ][i][)][ in the input as][ �] u[p][u] �yi:F ,u|=[αi]δi[1][;] 4 Initialize a counter ci to zero for each such sum 5 for pu > 0 do 6 Guess values x1, .., xn; 7 Simulate-Interventions(1, {}, x1, ..., xn) 8 end 

9 Replace the sums by ci and verify whether the (in-)equalities are satisfied; 

10 Function Simulate-Interventions(i, α, x1, . . . , xn) Input: Current variable Xi; set of interventions α; values x1, . . . , xn 11 if i > n then 12 for each sum counter cj do 13 for all possible values yj of the sum do 14 if αj (after inserting yi) is [{Xi = xi for i ∈ α}] and x1, .., xn satisfy δj (after inserting yi) then 15 increment cj by pu 16 end 17 end 18 end 19 else 20 Simulate-Interventions(i + 1, α, x1, ..., xn) ; 21 for value v do 22 Let x[′] 1[, ..., x][′] n[:=][ x][1][, ..., x][n][;] 23 x[′] i[:=][ v][;] 24 if v = xi then 25 guess new values x[′] i+1[, ..., x] n[′] 26 end 27 Simulate-Interventions(i + 1, α ∪{i}, x[′] 1[, ..., x] n[′][)][ ;] 28 end 29 end 

Algorithm 1: Solving SATL[lin] 2[⟨][Σ][⟩] 

Next, we combine the terms[�] u[p][u][of][all][sums][(implicitly).] For example, two sums �yi[P][([][α][i][]][δ][i][) +][ �] yj[P][([][α][j][]][δ][j][)][ can be rewritten as] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0015-06.png)


The algorithm performs this sum over u in line 5 and calculates the next sums in the subfunction. We can and will ignore the actual values u. Relevant is only that the value of the sums is multiplied by pu and that the functions F might change in each iteration. 

Recall that the functions F = (F1, . . . , Fn) determine the values of the endogenous variables, that is the value of variable Xi is given by xi = Fi(u, x1, . . . , xi−1). Thereby the functions (i.e. variables) have a causal order X1 ≺ . . . ≺ Xn, such that the value xi only depends on variables Xj with j < i. In reverse, this means that each intervention on a variable Xi can only change variables Xj with j > i. 

Rather than storing the functions F = (F1, . . . , Fn), the algorithm only stores the values xi = Fi(u, x1, . . . , xi−1), i.e. the values of the endogenous variables. The algorithm knows these values as well as the causal order, since they can be guessed non-deterministically. 

15 

Published as a conference paper at ICLR 2025 

The subfunction Simulate-Interventions then performs all possible interventions recursively, intervening first on variable X1, then X2, . . ., until Xn. The parameter i means an intervention on variable Xi, α is the set of all previous interventions, and x1, . . . , xn the current values. 

In line 20, it proceeds to the next variable, without changing the current variable Xi (simulating all possible interventions includes intervening on only a subset of variables). In line 21, it enumerates all values x[′] i[for variable][ X][i][.][If][ x][i][=][x][′] i[,][then the][intervention does nothing.][If][x][i][=][x][′] i[,][then][the] intervention might change all variables Xj with j > i (because the function Fj might depend on Xi and change its value). This is simulated by guessing the new values x[′] j[.][Thereby, we][get][the][new] values without considering the functions. 

In the last call, line 27, it has completed a set of interventions α. The function then searches every occurrence of the interventions α in the (implicitly expanded) input formula. That is, for each sum �yj:F ,u|=[αj]δj[1][, it counts how often][ α][j][=][ α][ occurs in the sum while the values][ x][i][satisfy][ δ][j][.][10] Since each intervention is enumerated only once, in the end, it obtains for all sums their exact value. It can then verify whether the values satisfy the (in-)equalities of the input. 

If a satisfying model exists, the algorithm confirms it, since it can guess the probabilities and the values of the functions. In reverse, if the algorithm returns true, a satisfying model can be constructed. The probabilities directly give a probability distribution P (u). The functions Fi can be constructed because, for each set of values u, x1, . . . , xi−1, only a single value for xi is guessed, which becomes the value of the function. 

Algorithm 1 runs in non-deterministic polynomial space and thus in PSPACE. 

A.3 PROOFS OF SECTION 3.3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0016-07.png)


Proof. We will reduce the satisfiability of a Schönfinkel-Bernays sentence to the satisfiability of SATL[base] 3[⟨][Σ][⟩] . The class of Schönfinkel–Bernays sentences (also called Effectively Propositional Logic, EPR) is a fragment of first-order logic formulas where satisfiability is decidable. Each sentence in the class is of the form ∃x∀yψ whereby ψ can contain logical operations ∧, ∨, ¬, variables x and y, equalities, and relations Ri(x, y) which depend on a set of variables, but ψ cannot contain any quantifier or functions. Determining whether a Schönfinkel-Bernays sentence is satisfiable is an NEXP-complete problem Lewis (1980) even if all variables are restricted to binary values Achilleos (2015). 

We will represent Boolean values as the value of the random variables, with 0 meaning FALSE and 1 meaning TRUE. We will assume that c = 2, so that all random variables are binary, i.e. Val = {0, 1}. 

In the proof, we will write (in)equalities between random variables as = and =. In the binary setting, X = Y is an abbreviation for (X = 0 ∧ Y = 0) ∨ (X = 1 ∧ Y = 1), and X = Y an abbreviation for ¬(X = Y ). To abbreviate interventions, we will write [w] for [W = w], [w] for interventions on multiple variables [W = w], and [v \ w] for interventions on all endogenous variables except W. 

We use random variables X = {X1, . . . Xn} and Y = {Y1, . . . Yn} for the quantified Boolean variables x, y in the sentence ∃x∀yψ. For each distinct k-ary relation Ri(z1, . . . , zk) in the formula, we define a random variable Ri and variables Zi[1][, . . . , Z] i[k][for the arguments. For the][ j][-th occurrence] of that relation Ri(t[1] ij[, . . . , t][k] ij[)][with][t][l] ij[∈{][x][1][, . . . , x][n][, y][1][, . . . , y][n][}][,][we][define][another][random] variable Ri[j][.] 

We use the following constraint to ensure that Ri only depends on its arguments: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0016-13.png)


Thereby[�] v[refers to summing over all values of all endogenous variables][11][ in the model and the] constraint says that an intervention on Zi[1][, . . . , Z] i[k][gives the same result for][ R][i][ as an intervention on] Zi[1][, . . . , Z] i[k][and the remaining variables, excluding][ R][i][.] 

> 10Rather than incrementing the counter cj by 1 and then multiplying the final result by pu, we increment it directly by pu. 

> 11All variables include variables Zi1[, . . . , Z] i[k][.] 

16 

Published as a conference paper at ICLR 2025 

We use the following constraint to ensure that Ri[j][only depends on its arguments:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0017-02.png)


and that Ri[j][and][ R][i][ have an equal value for equal arguments:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0017-04.png)


We add the following constraint for each Xi to ensure that the values of X are not affected by the values of Y: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0017-06.png)


Here the first sum sums over all values v of all endogenous variables V (including Xi and Y), and the second sums sums over values for variables Y. The intervention [v \ xi] intervenes on all variables except Xi and sets the values y to the values of the first sum. The intervention [v \ (xi, y), Y = y[′] ] intervenes on all variables except Xi and sets the values y to the values y[′] of the second sum. The constraint thus ensures that the value of Xi does not change when changing Y from y to y[′] . 

Let ψ[′] be obtained from ψ by replacing equality and relations on the Boolean values with the corresponding definitions for the random variables: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0017-09.png)


Suppose the SATL[base] 3[⟨][Σ][⟩] instance is satisfied by a model M. We need to show ∃x∀yψ is satisfiable. Each probability P(. . .) implicitly sums over all possible values u of the exogenous variables. The values x of the variables X might change together with the values u, however, any values x that are taken at least once can be used to satisfy ∃x∀yψ: If there was any x that would not satisfy ψ for all values of y, P([y]ψ[′] ) would be less than 1 for these values of x (determined by u) and y, and equation (9) would not be satisfied. For each relation Ri, we choose the values given by the random variable Ri. Each occurrence Ri[j][(][t][1] ij[, . . . , t][k] ij[)][has a][value that is][given by][ [][Z] i[1][=][t][1] ij[, . . . , Z] i[k][=][t][k] ij[]][R][i][in][the model][ M][.][Due to] equations (7) and equations (6), that is the same value as [Ti[1][=][t][1] ij[, . . . , T][ k] ij[=][t][k] ij[]][R] i[j][, which is the] value used in [y]ψ[′] . Since [y]ψ[′] is satisfied, so is ψ and ∃x∀yψ. 

Suppose ∃x∀yψ is satisfiable. We create a deterministic model M as follows: The value of random variables X is set to the values chosen by ∃x. The relation random variables Ri are functions depending on random variables Zi[1][, . . . , Z] i[k][that return the value of the relation][ R][i][(][z][1][, . . . , z][k][)][.][The re-] lation random variables Ri[j][on arguments][ T][ 1] ij[, . . . , T][ k] ij[return the value of the relation][ R][i][(][t][1] ij[, . . . , t][k] ij[)][.] This satisfies Equation 5 and 6 because the functions only depend on their arguments, and Equation 7 because the functions result from the same relation (so the functions are dependent, but causally independent, which yields a non-faithful model. But the equations do not test for faithfulness or dependences). All other random variables can be kept constant, which satisfies Equation 8 (despite being constant in the model, the causal interventions can still change their values). Finally, Equation 9 holds, because ψ is satisfied for all Y. 

The problem can be solved in NEXP because expanding all sums of a SATL[lin] 3[⟨][Σ][⟩] instance creates a SAT[lin] L3[instance of exponential size, which can be solved non-deterministically in a time polynomial] to the expanded size as shown by Mossé et al. (2022). 

A.4 PROOFS OF SECTION 4 

Mossé et al. (2022) already prove SAT[poly] L3[∈∃][R][ when the][ L] 3[poly] -formula is allowed to neither contain subtractions nor conditional probabilities. We slightly strengthen this result to allow both of them. Lemma 16. SAT[poly] L3[∈∃][R][.][This also holds true if we allow the basic terms to contain conditional] probabilities. 

17 

Published as a conference paper at ICLR 2025 

Proof. (Mossé et al., 2022) show that SAT[poly] L3[without subtraction or conditional probabilities is in] ∃R. Their algorithm is given in the form of a NP-reduction from SAT[poly] L3[to][ETR][and][using][the] closure of ∃R under NP-reductions. In particular given a L[poly] 3 -formula ϕ, they replace each event P(ǫ) by the sum[�] δ∈∆[+] :δ|=ǫ[P][(][δ][)][ where][ ∆][+][⊆E][counterfact][ is a subset of size at most][ |][ϕ][|][.][They add] the constraint[�] δ∈∆[+][ P][(][δ][)][=][1][ and then replace each of the][P][(][δ][)][ by a variable constrained to][be] between 0 and 1 to obtain a ETR-formula. Note that the final ETR-formula allows for subtraction, so ϕ is allowed to have subtractions as well. Remains to show how to deal with conditional probabilities. We define conditional probabilities P(δ|δ[′] ) to be undefined if P(δ[′] ) = 0 (this proof works similarly for other definitions). In ϕ replace P(δ|δ[′] ) by[P] P[(] ([δ,δ] δ[′] )[′][)][.][The][resulting][ETR-formula][then] contains some divisions. To remove some division[α] β[, we use Tsaitin’s trick and replace][α] β[by a fresh] variable z. We then add the constraints α = z · β and β = 0 to the formula. 

Now we are ready to show that SAT[poly] L3[⟨][Σ][⟩] can be solved in NEXP over the Reals. 

Proof of Theorem 10. By Lemma 16 we know that SAT[poly] L3[is][in][∃][R][.][Thus,][by][(Erickson et al.,] 2022), there exists an NPreal algorithm, call it A, which for a given L[poly] 3 -formula decides if is 

To solve the SAT[poly] L3[⟨][Σ][⟩] problem, a NEXPreal algorithm expands firstly all sums of a given instance and creates an equivalent SAT[poly] L3[instance][of][size][bounded][exponentially][in][the][size][of][the][initial] input formula. Next, the algorithm A is used to decide if the expanded instance is satisfiable. 

18 

Published as a conference paper at ICLR 2025 

## B LEVELS OF PEARL’S CAUSAL HIERARCHY: AN EXAMPLE 

To illustrate the main ideas behind the causality notions, we present in this section an example that, we hope, will make it easier to understand the formal definitions. In the example, we consider a (hypothetical) scenario involving three attributes represented by binary random variables: pneumonia modeled by Z = 1, drug treatment (e.g., with antibiotics) represented by X = 1, and recovery, with Y = 1 (and Y = 0 meaning mortality). Below we describe an SCM which models an unobserved true mechanism behind this setting and the canonical patterns of reasoning that can be expressed at appropriate layers of the hierarchy. 

Structural Causal Model An SCM is defined as a tuple (F , P, U, X) which is of unobserved nature from the perspective of a researcher who studies the scenario. The SCM models the ground truth for the distribution P (U) of the population and the mechanism F . In our example, the model assumes three independent binary random variables U = {U1, U2, U3}, with probabilities: P (U1=1) = 0.75, P (U2=1) = 0.8, P (U3=1) = 0.4, and specifies the mechanism F = {F1, F2, F3} for the evaluation of the three endogenous (observed) random variables 

U1 U2 U3 P (u) Z X Y 0 0 0 0.03 1 0 0 0 0 1 0.02 1 0 1 0 1 0 0.12 1 0 0 0 1 1 0.08 1 0 1 1 0 0 0.09 0 0 0 1 0 1 0.06 0 0 1 1 1 0 0.36 0 1 0 1 1 1 0.24 0 1 1 

Z, X, Y as follows: Z := F1(U1) = 1−U1; X := F2(Z, U2) = (1−Z)U2; Y := F3(X, U1, U3) = X(1 − U1)(1 − U3) + (1 − X)(1 − U1)U3 + U1U3. Thus, our model determines the distribution P (u), for u = (u1, u2, u3), and the values for the observed variables, as can be seen above. 

The unobserved random variable U1 models all circumstances that lead to pneumonia and Z is a function of U1 (which may be more complex in real scenarios). Getting a treatment depends on having pneumonia but also on other circumstances, like having similar symptoms due to other diseases, and this is modeled by U2. So X is a function of Z and U2. Finally, mortality depends on all circumstances that lead to pneumonia, getting the treatment, and on further circumstances like having other diseases, which are modeled by U3. So Y is a function of U1, X, and U3. We always assume that the dependency graph of the SCM is acyclic. This property is also called semiMarkovian. 

|always assume that the dependency graph of the SCM is acyclic. This prope<br>Markovian.|rty is also|calledsemi-|
|---|---|---|
|Layer 1Empirical sciences rely heavily on the use of observed data, which<br>are typically represented as probability distributions over observed (measur-|Z X Y<br>0 0 0|P(z, x, y)<br>0.09|
|able) variables. In our example, this is the distribution P over Z, X, and|0 0 1|0.06|
|Y. The remaining variablesU1, U2, U3, as well as the mechanismF, are of|0 1 0|0.36|
|unobserved nature. Thus, in our scenario, a researcher gets the probabilities|0 1 1|0.24|
|(shown to the right) P(z, x, y) = �<br>u δF,u(z, x, y)·P(u), where vectors|1 0 0|0.15|
|u= (u1, u2, u3)∈{0,1}3 and|1 0 1|0.10|



δF ,u(z, x, y) = 1 if F1(u1)=z, F2(z, u2)=x, and F3(x, u1, u2)=y; otherwise δF ,u(z, x, y) = 0. The relevant query in our scenario P (Y =1|X=1) can be evaluated as P (Y =1|X=1) = P (Y =1, X=1)/P (X=1) = 0.24/0.6 = 0.4 which says that the probability for recovery (Y =1) is only 40% given that the patient took the drug (X=1). On the other hand, the query for X=0 can be evaluated as P (Y =1|X=0) = P (Y =1, X=0)/P (X=0) = 0.16/0.4 = 0.4 which may lead to the (wrong, see the next layer) opinion that the drug is irrelevant to recovery. 

|Layer 2 Consider a randomized drug<br>trial in which each patient receives treat-<br>ment, denoted as do(X=1), regardless<br>of pneumonia (Z) and other conditions<br>(U2). We model this by performing a hy-<br>pothetical intervention in which we re-<br>place inF the mechanismF2(Z, U2)by<br>the constant function 1 and leaving the<br>remaining<br>U1 U2 U3<br>P(u)<br>Z X=1 Y<br>0<br>0<br>0<br>0.03<br>1<br>1<br>1<br>0<br>0<br>1<br>0.02<br>1<br>1<br>0<br>0<br>1<br>0<br>0.12<br>1<br>1<br>1<br>0<br>1<br>1<br>0.08<br>1<br>1<br>0<br>1<br>0<br>0<br>0.09<br>0<br>1<br>0<br>1<br>0<br>1<br>0.06<br>0<br>1<br>1<br>1<br>1<br>0<br>0.36<br>0<br>1<br>0<br>1<br>1<br>1<br>0.24<br>0<br>1<br>1<br>Z Y<br>P([X=1]z, y)<br>0 0<br>0.45<br>0 1<br>0.30<br>1 0<br>0.10<br>1 1<br>0.15|
|---|
|functions unchanged. If FX=1 = {F ′<br>1=F1, F′<br>2=1, F′<br>3=F3}denotes the new mechanism, then the|
|post-interventionaldistributionP([X=1]Z, Y)is specifed asP([X=1]z, y) = �<br>u δFX=1,u(z, y)·|
|P(u),whereδFX=1,u denotes functionδ as above, but for the new mechanismFX=1 (the distribu-|
|tion is shown on the right-hand side). A common and popular notation for the post-interventional|
|probability is P(Z, Y|do(X=1)). In this paper, we use the notation P([X=1]Z, Y) since it is|
|more convenient for analyses involving counterfactuals.<br>To determine the causal effect of the|



19 

Published as a conference paper at ICLR 2025 

drug on recovery, we compute, in an analogous way, the distribution P ([X=0]Z, Y ) after the intervention do(X=0), which means that all patients receive placebo. Then, comparing the value P ([X=1]Y =1) = 0.45 with P ([X=0]Y =1) = 0.40, we can conclude that P ([X=1]Y =1) − P ([X=0]Y =1) > 0. This can be interpreted as a positive (average) effect of the drug in the population which is in opposite to what has been inferred using the purely probabilistic reasoning of Layer 1. Note that it is not obvious how to compute the post-interventional distributions from the observed probability P (Z, X, Y ); Indeed, this is a challenging task in the field of causality. 

Layer 3 The key phenomena that can be modeled and analyzed at this level are counterfactual situations. Imagine, e.g., in our scenario there is a group of patients who did not receive the treatment and died (X=0, Y =0). One may ask, what would be the outcome Y had they been given the treatment (X=1). In particular, one can ask what is the probability of recovery if we had given the treatment to the patients of this group. Using the formalism of Layer 3, we can express this as a counterfactual query: P ([X=1]Y =1|X=0, Y =0) = P ([X=1](Y =1) ∧ (X=0, Y =0))/P (X=0, Y =0). Note that the event [X=1](Y =1) ∧ (X=0, Y =0) incorporates simultaneously two counterfactual mechanisms: FX=1 and F . This is the key difference to Layer 2, where we can only have one. We define the probability in this situation as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0020-03.png)


X=0, Y =0 is satisfied only for (U1, U2, U3) ∈{(0, 0, 0), (0, 1, 0), (1, 0, 0)} (first table), and of them only {(0, 0, 0), (0, 1, 0)} satisfies [X=1]Y =1 (third table). Thus, by marginalizing Z, we get P ([X=1]Y =1|X=0, Y =0) = 0.15/0.24 = 0.625 which may be interpreted that more than 62% of patients who did not receive treatment and died would have survived with treatment. Finally, we would like to note that, in general, the events of Layer 3 can be quite involved and incorporate simultaneously many counterfactual worlds. 

Graph Structure of an SCM Below we remind, how an SCM can be represented in the form of a Directed Acyclic Graph (DAG) and show such a DAG for the model discussed above. 

Let M = (F = {F1, . . . , Fn}, P, U, X = {X1, . . . , Xn}) be an SCM. We assume that the model is Markovian, i.e. that the exogenous arguments Ui, Uj of Fi, resp. Fj are independent whenever i = j. These exogenous arguments are not shown in the DAG. We note that a general model as discussed above, called semi-Markovian, which allows for the sharing of exogenous arguments and allows for arbitrary dependencies among the exogenous variables, can be reduced in a standard way to the Markovian model by introducing auxiliary “unobserved” variables. Thus, in our example, to get a Markovian model, we can assume, X = {X, Y, Z, U1}, where X, Y, Z remain observed variables but U1 is of unobserved nature. 

We define that a DAG G = (X, E) represents the graph structure of M if, for every Xj appearing as an argument of Fi, Xj → Xi is an edge in E. DAG G is called the causal diagram of the model M Pearl (2009); Bareinboim et al. (2022). The DAG for the discussed SCM therefore has the following form (here, as is usually done in Markovian models, variables U2 and U3 that only affect X, respectively Y , are omitted): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0020-08.png)


Moreover, the DAG of the intervention model discussed in subsection Layer 2, with functional mechanism FX=1 has the following form, meaning that all in-going edges to X are removed from the pre-interventional model: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0020-10.png)


20 

Published as a conference paper at ICLR 2025 

## C SYNTAX AND SEMANTICS OF THE LANGUAGES OF PCH: FORMAL DEFINITIONS 

We always consider discrete distributions in the probabilistic and causal languages studied in this paper. We represent the values of the random variables as Val = {0, 1,..., c − 1} and denote by X the set of random variables used in a system. By capital letters X1, X2,..., we denote the individual variables and assume, w.l.o.g., that they all share the same domain Val . A value of Xi is often denoted by xi or a natural number. In this section, we describe syntax and semantics of the languages starting with probabilistic ones and then we provide extensions to the causal systems. 

By an atomic event, we mean an event of the form X = x, where X is a random variable and x is a value in the domain of X. The language Eprop of propositional formulas over atomic events is defined as the closure of such events under the Boolean operators ∧ and ¬. To specify the syntax of interventional and counterfactual events we define the intervention and extend the syntax of Eprop to Epost-int and Ecounterfact, respectively, using the following grammars: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0021-04.png)


Note that since ⊤ means that no intervention has been applied, we can assume that Eprop ⊆Epost-int. 

The PCH consists of three languages L1, L2, L3, each of which is based on terms of the form P(δ). For the (observational or associational) language L1, we have δ ∈Eprop, for the (interventional) language L2, we have δ ∈Epost-int and for the (counterfactual) language L3, δ ∈Ecounterfact. The expressive power and computational complexity properties of the languages depend largely on the operations that we are allowed to apply to the basic terms. Allowing gradually more complex operators, we describe the languages which are the subject of our studies below. We start with the description of the languages Ti[∗][of terms, with][ i][ = 1][,][ 2][,][ 3][, using the following grammars][12] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0021-07.png)


where δ1 are formulas in Eprop, δ2 ∈Epost-int, δ3 ∈Ecounterfact. 

The probabilities of the form P(δi) are called primitives or basic terms. In the summation operator[�] x[, we have a dummy variable][ x][ which ranges over all values][ 0][,][ 1][, . . . , c][ −][1][.][The summation] �x[t][is][a][purely][syntactical][concept][which][represents][the][sum][t][[][0][/][x][]][+][t][[][1][/][x][]][+][ ...][+][t][[][c][ −][1][/][x][]][,] where by t[[v] /x], we mean the expression in which all occurrences of x are replaced with value v. For example, for Val = {0, 1}, the expression[�] x[P][(][Y][ =1][, X][=][x][)][semantically][represents] P(Y =1, X=0) + P(Y =1, X=1). We note that the dummy variable x is not a (random) variable in the usual sense and that its scope is defined in the standard way. 

In the table above, the terms in Ti[base] are just basic probabilities with the events given by the corresponding languages Eprop, Epost-int, or Ecounterfact. Next, we extend terms by being able to compute sums of probabilities and by adding the same term several times, we also allow for weighted sums with weights given in unary. Note that this is enough to state all our hardness results. All matching upper bounds also work when we allow for explicit weights given in binary. In the case of Ti[poly] , we are allowed to build polynomial terms in the primitives. On the right-hand side of the table, we have the same three kinds of terms, but to each of them, we add a marginalization operator as a building block. 

The polynomial calculus Ti[poly] was originally introduced by Fagin, Halpern, and Megiddo (Fagin et al., 1990) (for i = 1) to be able to express conditional probabilities by clearing denominators. While this works for Ti[poly] , this does not work in the case of Ti[poly][⟨][Σ][⟩] , since clearing denominators with exponential sums creates expressions that are too large. But we could introduce basic terms of the form P(δi|δ) with δ ∈Eprop explicitly. All our hardness proofs work without conditional probabilities but all our matching upper bounds are still true with explicit conditional probabilities. 

> 12In the given grammars we omit the brackets for readability, but we assume that they can be used in a standard way. 

21 

Published as a conference paper at ICLR 2025 

Expression as P(X=1)+ P(Y =2) · P(Y =3) is a valid term in T1[poly] and[�] z[P][([][X][=0](][Y][ =1][, Z][=][z][))] and[�] z[P][(([][X][=0]][Y][ =1)][, Z][=][z][)][ are valid terms in the language][ T] 3[poly][⟨][Σ][⟩] , for example. Now, let Lab = {base, base⟨Σ⟩, lin, lin⟨Σ⟩, poly, poly⟨Σ⟩} denote the labels of all variants of languages. Then for each ∗∈ Lab and i = 1, 2, 3 we define the languages L[∗] i[of Boolean combinations] of inequalities in a standard way: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy_images/iclr_2025_From_Probability_to_Counterfactuals_the_Increasing_Complexity_of_Satisfiability_in_Pearl's_Causal_Hierarchy.pdf-0022-02.png)


Although the language and its operations can appear rather restricted, all the usual elements of probabilistic and causal formulas can be encoded. Namely, equality is encoded as greater-or-equal in both directions, e.g. P(x) = P(y) means P(x) ≥ P(y) ∧ P(y) ≥ P(x). The number 0 can be encoded as an inconsistent probability, i.e., P(X=1 ∧ X=2). In a language allowing addition and multiplication, any positive integer can be easily encoded from the fact P(⊤) ≡ 1, e.g. 4 ≡ (1 + 1)(1 + 1) ≡ (P(⊤) + P(⊤))(P(⊤) + P(⊤)). If a language does not allow multiplication, one can show that the encoding is still possible. Note that these encodings barely change the size of the expressions, so allowing or disallowing these additional operators does not affect any complexity results involving these expressions. 

To define the semantics of the languages, we use a structural causal model (SCM) as in (Pearl, 2009, Sec. 3.2). An SCM is a tuple M = (F , P, U, X), such that V = U ∪ X is a set of variables partitioned into exogenous (unobserved) variables U = {U1, U2,...} and endogenous variables X. The tuple F = {F1,..., Fn} consists of functions such that function Fi calculates the value of variable Xi from the values (x, u) of other variables in V as Fi(pai, ui)[13] , where Pai ⊆ X and Ui ⊆ U. P specifies a probability distribution of all exogenous variables U. Since variables X depend deterministically on the exogenous variables via functions Fi, F and P obviously define the joint probability distribution of X. Throughout this paper, we assume that domains of endogenous variables X are discrete and finite. In this setting, exogenous variables U could take values in any domains, including infinite and continuous ones. A recent paper (Zhang et al., 2022) shows, however, that any SCM over discrete endogenous variables is equivalent for evaluating post-interventional probabilities to an SCM where all exogenous variables are discrete with finite domains. As a consequence, throughout this paper, we assume that domains of exogenous variables U are discrete and finite, too. 

For any basic Eint-formula Xi=xi (which, in our notation, means do(Xi=xi)), we denote by FXi=xi the function obtained from F by replacing Fi with the constant function Fi(v) := xi. We generalize this definition for any interventions specified by α ∈Eint in a natural way and denote as Fα the resulting functions. For any ϕ ∈Eprop, we write F , u |= ϕ if ϕ is satisfied for values of X calculated from the values u. For α ∈Eint, we write F , u |= [α]ϕ if Fα, u |= ϕ. And for all ψ, ψ1, ψ2 ∈Ecounterfact, we write (i) F , u |= ¬ψ if F , u ̸|= ψ and (ii) F , u |= ψ1 ∧ ψ2 if F , u |= ψ1 and F , u |= ψ2. Finally, for ψ ∈Ecounterfact, let SM = {u | F , u |= ψ}. We define �e�M, for some expression e, recursively in a natural way, starting with basic terms as follows �P(ψ)�M = �u∈SM(ψ)[P][(][u][)][and,][for][δ][∈E][prop][,][�][P][(][ψ][|][δ][)][�][M][=][�][P][(][ψ][∧][δ][)][�][M][/][�][P][(][δ][)][�][M][,][assuming][that][the] expression is undefined if �P(δ)�M = 0. For two expressions e1 and e2, we define M |= e1 ≤ e2, if and only if, �e1�M ≤ �e2�M. The semantics for negation and conjunction are defined in the usual way, giving the semantics for M |= ϕ for any formula ϕ in L[∗] 3[.] 

> 13We consider recursive models, that is, we assume the endogenous variables are ordered such that variable Xi (i.e. function Fi) is not affected by any Xj with j > i. 

22 

