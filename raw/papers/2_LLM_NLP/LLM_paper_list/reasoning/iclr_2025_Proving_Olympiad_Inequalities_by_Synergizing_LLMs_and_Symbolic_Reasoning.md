Published as a conference paper at ICLR 2025 

# PROVING OLYMPIAD INEQUALITIES BY SYNERGIZING LLMS AND SYMBOLIC REASONING 

**Zenan Li**[1] _[∗]_ **, Zhaoyu Li**[2] **,** _[∗]_ **Wen Tang**[3] **, Xian Zhang**[4] **, Yuan Yao**[1] **, Xujie Si**[2] **, Fan Yang**[4] **, Kaiyu Yang**[5] , _[†]_ **Xiaoxing Ma**[1] _[†]_ 

1 State Key Lab of Novel Software Technology, Nanjing University, China, 2 University of Toronto, 3 Peking University, 4 Microsoft Research, 5 Meta FAIR lizn@smail.nju.edu.cn, zhaoyu@cs.toronto.edu zhxian@microsoft.com, kaiyuy@meta.com, xxm@nju.edu.cn 

**Problem (Chen, 2014):** If _a, b, c_ are positive reals and _a_[2] + _b_[2] + _c_[2] = 1, then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0001-06.png)


Figure 1: We prove inequality problems in math Olympiads that involve a finite number of real variables, hypotheses, and one conclusion. Both the hypotheses and the conclusion consist of constants, variables, algebraic operations (e.g., addition, multiplication), and transcendental functions like _exp_ . 

## ABSTRACT 

Large language models (LLMs) can prove mathematical theorems formally by generating proof steps ( _a.k.a._ tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure 1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data. 

## 1 INTRODUCTION 

Automated theorem proving has been a long-standing goal in AI (Newell & Simon, 1956). Recent research explores leveraging large language models (LLMs) to generate formal proofs that can be verified in formal proof systems like Lean (de Moura et al., 2015), opening a new avenue to theorem proving (Li et al., 2024; Yang et al., 2024). This promising approach has already led to tools that assist human mathematicians (Song et al., 2024) and the first AI that achieves silver-medal performance in the International Mathematical Olympiad (IMO) (Google DeepMind, 2024). 

While LLM-based proof generation shows great promise across various mathematical domains, its performance is constrained by the scarcity of formal proof data. Furthermore, it remains an open 

> _∗_ Equal contribution. This work was partially done during Zenan’s internship at MSRA. 

> _†_ Equal advising. All experiments were conducted outside Meta’s compute infrastructure. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Current inequality problem New inequality problem Inequality sequence (Proof)<br>Given 𝑎, 𝑏, 𝑐 three reals, prove that Given 𝑎, 𝑏, 𝑐 three reals, prove that Given 𝑎, 𝑏, 𝑐 three reals, prove that<br>𝑎𝑏+ 𝑏𝑐+ 𝑐𝑎≤𝑎 [!] + 𝑏 [!] + 𝑐 [!] (2𝑎 [!] + 𝑏 [!] )(2𝑐 [!] + 𝑏 [!] ) ≤𝑎 [!] + 𝑏 [!] + 𝑐 [!] … … 𝑎𝑏+ 𝑏𝑐+ 𝑐𝑎≤𝑎 [!] + 𝑏 [!] + 𝑐 [!]<br>1. scale Cauchy_Schwarz_3vars<br>Tactic generation & pruning Goal filtering & ranking (2𝑎 [!] + 𝑏 [!] )(2𝑐 [!] + 𝑏 [!] ) ≤𝑎 [!] + 𝑏 [!] + 𝑐 [!]<br>(2𝑎 [!] + 𝑏 [!] )(2𝑐 [!] + 𝑏 [!] ) ≤𝑎 [!] + 𝑏 [!] + 𝑐 [!] ① (2𝑎 [!] + 𝑏 [!] )(2𝑐 [!] + 𝑏 [!] ) ≤𝑎 [!] + 𝑏 [!] + 𝑐 [!] 2. scale AM_GM_2vars<br>𝑎𝑏+ 𝑏𝑐+ 𝑐𝑎≤3 ! 𝑎 [!] 𝑏 [!] 𝑐 [!] ② 3𝑎𝑏+ 3𝑏𝑐+ 3𝑐𝑎≤ 𝑎+ 𝑏+ 𝑐 [!] 2𝑎 [!] + 𝑏 [!] + 2𝑐 [!] + 𝑏 [!]<br>≤𝑎 [!] + 𝑏 [!] + 𝑐 [!]<br>Symbolic-based scaling … …  … …  LLM ranking 2<br>𝑎 𝑏−𝑎+ 𝑏3𝑎𝑏+ 3𝑏𝑐+ 3𝑐𝑎≤𝑐−𝑏+ 𝑐 𝑎+ 𝑏+ 𝑐𝑎−𝑐≤0 [!] 𝑎𝑏𝑎𝑏 ++ 𝑏𝑐𝑏𝑐 ++ 𝑐𝑎𝑐𝑎 ≤≤ 3 𝑎+! 𝑎!𝑏𝑏!!𝑐−! 2𝑎𝑏 + 𝑐! C A … 3. llm_simplify 𝑎 [!] + 𝑏 [!] + 𝑐 [!]  ≤𝑎 [!] + 𝑏 [!] + 𝑐 [!]<br>LLM-generated rewriting … …  … …  Symbolic filtering AC BB QED!<br>∀∃⊨<br>x ∧y ∨z𝑎! = 1<br>**----- End of picture text -----**<br>


Figure 2: An overview of our neuro-symbolic inequality prover LIPS. By integrating both LLMs and symbolic methods in an iterative process of tactic generation and goal selection, it can generate human-readable and formally verifiable proofs in Lean for Olympiad-level inequality problems. 

problem whether LLMs can perform precise and complex symbolic manipulations (Hammond & Leake, 2023). To address these limitations, mechanical symbolic reasoning is still essential. Unlike LLMs, symbolic methods leverage domain-specific knowledge to achieve greater efficiency and generalization without relying on extensive training data (Wu, 2008; Heule et al., 2016). Integrating LLMs with symbolic methods presents a promising strategy for tactic generation and theorem proving. This raises a key question: _Which aspects of mathematical reasoning are best suited to LLMs, and which to symbolic methods?_ By exploring this question, we aim to combine the strengths of both approaches effectively. Since symbolic methods are inherently domain-specific, we focus on a concrete domain: inequalities, which offers a balance between feasibility and practicality. 

**Mathematical Inequalities.** Inequalities arise from various branches of mathematics (Hardy et al., 1952). Their proofs often rely on a relatively small set of fundamental techniques, yet these techniques can be applied in remarkably intricate and nuanced ways. In this paper, we further constrain our scope to elementary algebraic inequalities (Figure 1), which were prevalent in high-school mathematics competitions (Manfrino et al., 2010), e.g., Problem 2 in IMO 2000. Additionally, these inequalities are closely related to quantifier-free real arithmetic in satisfiability modulo theories (SMT) (Barrett & Tinelli, 2018), which have numerous practical applications in formal verification. 

Even this restricted class of inequalities poses significant challenges that surpass the capabilities of current symbolic or neural provers. Symbolic methods (Yang, 1999; Uray, 2020) partition the variable space (e.g., R[3] for three variables) into a finite number of cells, which are then exhaustively enumerated. These approaches suffer from combinatorial explosion and quickly become computationally infeasible for competition-level problems. Furthermore, due to their enumerative nature, these methods are black boxes that cannot produce human-readable proofs. On the other hand, neural approaches (Wu et al., 2021; Wei et al., 2024) fine-tune language models to generate formal proofs by predicting tactics or evaluating subgoals during proof search. However, due to data scarcity and difficulties in generalization, they frequently underperform on certain types of problems. 

**Our Approach.** We introduce LIPS (LLM-based inequality prover with symbolic reasoning), a neuro-symbolic framework that synergistically combines LLMs with domain-specific symbolic techniques, as illustrated in Figure 2. Specifically, we analyze common proving strategies used by humans in inequality proofs and categorize them into two types of tactics: _scaling_ and _rewriting_ . Scaling tactics apply existing lemmas (e.g., the Cauchy-Schwarz inequality) to scale a subterm in the current goal. The set of lemmas is finite, and each lemma can be applied only in a limited number of ways. Therefore, we can enumerate all possible scaling tactics using symbolic tools. However, not all scaling tactics are useful for the current goal; over-scaling may render the goal invalid. We filter out such invalid scaling by using symbolic tools such as SMT solvers to check for counterexamples of the resulting goals. Rewriting tactics, on the other hand, transform a term into an equivalent form (e.g., subtracting 2 _ab_ from both sides of the current goal). Any term can be rewritten in infinite ways, making exhaustive enumeration impossible. To address this, we use LLMs to generate rewriting tactics by designing a series of prompts for different rewriting formats. By leveraging the mathematical intuition embedded in LLMs, we implicitly prune the infinite tactic space, sampling the most promising equivalent transformations. 

2 

Published as a conference paper at ICLR 2025 

Scaling and rewriting the current goal leads to a set of subgoals with new inequalities to prove. Efficient proof search requires prioritizing the most promising subgoals for further exploration. To this end, we employ two strategies: _symbolic filtering_ and _neural ranking_ . In symbolic filtering, we use heuristics based on inequalities’ homogeneity and decoupling properties to filter out unpromising subgoals. In neural ranking, the remaining subgoals are fed into an LLM, which compares and ranks them using chain-of-thought prompting (Wei et al., 2022). After filtering and ranking, we end up with a small number of ranked subgoals. We then iteratively generate and apply new tactics, filter and rank the resulting subgoals, until the final goal becomes trivially provable, resulting in a proof that is both human-readable and formally verified by Lean. 

We evaluate LIPS on 161 challenging inequalities collected from three problem sets (Chen, 2014; Tung, 2012; Wei et al., 2024). The experimental results show that LIPS consistently achieves stateof-the-art performance and significantly outperforms existing neural and symbolic methods in both effectiveness and speed. Remarkably, out of 61 inequality problems sourced from various math Olympiad competitions, LIPS successfully proves 56 within 90 minutes, whereas the previous best approach proves only 41 problems and fails to generate human-readable proofs. 

## 2 INEQUALITY THEOREM PROVING 

In this section, we briefly review existing neural and symbolic methods for inequality proving. In a nutshell, symbolic methods complete proofs by exploring the problem space exhaustively, whereas neural methods use LLMs to sample proof steps from a vast tactic space. Each approach comes with distinct advantages and challenges. 

## 2.1 SYMBOLIC METHODS 

Consider a set of polynomial inequalities Φ _j ▷◁_ 0 _, j_ = 1 _, . . . , m_ , where _▷◁ ∈{≤, <, ≥, >,_ = _, ̸_ = _}_ , and each polynomial is defined over _n_ variables as Φ _j_ ( _x_ 1 _, . . . , xn_ ) :=[�] _[k] i_ =1 _[a][i][·][x][i]_ 1[1] _[x]_ 2 _[i]_[2] _[· · ·][ x] n[i][n]_[,] cylindrical algebraic decomposition (CAD) (Arnon et al., 1984; Caviness & Johnson, 2012; Kremer, 2020) divides the variable space R _[n]_ into multiple connected semi-algebraic sets, known as _cells_ . Within each cell, the sign of every polynomial remains constant (positive, negative, or zero). Figure 6(a) illustrates this process with an example of CAD-producing cells for two intersecting unit circles. By exploiting this sign invariance, we can determine the satisfiability of inequalities by enumerating and checking all the cells, rather than searching the entire infinite space R _[n]_ . 

CAD and its variants have been extensively utilized in modern SMT solvers (Jovanovi´c & de Moura, 2013; Kremer et al., 2022; Uncu et al., 2023). For inequality theorem proving, CAD transforms the problem into an enumeration task, exhaustively examining all cells to assess whether the inequality can be satisfied. While it is capable, CAD’s performance remains unsatisfactory for several reasons. Firstly, CAD’s heuristic strategies are primarily designed to efficiently find counterexamples rather than to optimize the proof search process. Secondly, CAD’s proving mechanism fails to generate explicit and interpretable reasoning paths, hindering both automatic verification by existing interactive theorem provers and human interpretation of the proofs. Additionally, CAD suffers from double exponential computational complexity relative to the number of variables _n_ (Davenport & Heintz, 1988), causing its efficiency to decrease significantly as the number of variables increases. Specifically, when dealing with nonlinear inequalities involving fractions or radical expressions, auxiliary variables are always introduced to eliminate these terms (e.g., converting _[√] x_ 1 + _[√] x_ 2 + _x_[2] 3[= 0][ into] _{x_ 4 + _x_ 5 + _x_[2] 3[= 0] _[, x]_[2] 4[=] _[ x]_[1] _[, x]_[2] 5[=] _[ x]_[2] _[, x]_[4] _[>]_[ 0] _[, x]_[5] _[>]_[ 0] _[}]_[).][Although this transformation successfully] rewrites the inequality into polynomial form, it drastically degrades the performance of CAD. 

## 2.2 NEURAL METHODS 

In contrast to symbolic methods based on CAD, some approaches leverage neural networks to predict tactics within an interactive theorem prover, generating human-like, step-by-step formal proofs. Specifically, these inequality proofs are typically structured in a top-down sequential manner, where each tactic either transforms the current goal into a new subgoal or directly completes the proof (see Figure 6(b)). Figure 6(c) provides an example proof of the inequality _ab_ + _bc_ + _ca ≤ a_[2] + _b_[2] + _c_[2] . 

Among existing work, INT (Wu et al., 2021) designs a theorem generator for elementary-level inequalities by randomly sampling axioms from a fixed set. It trains a Transformer (Vaswani et al., 

3 

Published as a conference paper at ICLR 2025 

2017) model to predict tactics and utilizes another value network with the Monte Carlo tree search to complete proofs. Similarly, AIPS (Wei et al., 2024) implements a synthetic generator that can produce IMO-level inequalities and trains a language model to score each inequality expression in a curriculum manner, performing a best-first search to solve these problems. However, their inequality generators have some restrictions – either limited in difficulty or constrained by specific forms like cyclic symmetry. Moreover, these approaches mainly rely on fine-tuning models on large-scale datasets, making them highly dependent on the quantity and diversity of the training data. 

## 3 TACTIC GENERATION AND PRUNING 

Compared to general theorem proving, the tactics used in inequality proofs are more well-structured: Common tactics can be categorized into two types, namely _scaling_ and _rewriting_ (Lee, 2004; Ikenaga, 2018). Scaling refines the given inequality using a known inequality lemma, such as the arithmetic and geometric means (AM-GM) inequality, while rewriting transforms a given inequality into an equivalent form, such as multiplying both sides by two. Scaling and rewriting have distinct characteristics. The number of lemmas that can be used in scaling is finite, whereas the number of equivalent transformations in rewriting can be infinite, which leads to different strategies for tactic generation. Furthermore, unlike rewriting, scaling does not preserve the provability of the inequality, leading to different strategies for pruning the resulting subgoals. 

Let’s consider the inequality problem in Figure 1 as a running example. To the best of our knowledge, the proof for this problem is not readily available online, and neither the most advanced LLMs (e.g., OpenAI o3-mini and DeepSeek-R1) nor existing CAD solvers can solve this inequality. 

## 3.1 SCALING TACTICS 

Given an inequality lemma (e.g., the special case of the AM-GM inequality _u_[2] + _v_[2] _≥_ 2 _uv_ ), we enumerate all possible ways to instantiate its arguments ( _u_ and _v_ ) so that it matches a part of the current proof goal. This is a classical problem called e-matching (De Moura & Bjørner, 2007; Moskal et al., 2008) and can be solved using existing symbolic tools (Willsey et al., 2021). However, it introduces a challenge due to the potentially large number of possible patterns. In our running example, there are a total of 162 possible pattern matches for the two-variable AM-GM inequality, 1 including cases like _{u_ := _a, v_ := _√_ 2 _}_ , _{u_ := 1 _, v_ := ~~_√_~~ _a_[2] +2 _[}]_[.] 

Since scaling tactics refine the inequality goal, they may produce potentially incorrect subgoals (i.e., unprovable statements). For instance, applying the AM-GM inequality with the pattern _{u_ := _a, v_ := _√_ 2 _}_ would transform the original inequality (1) into 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0004-08.png)


~~_√_~~ 3 However, this inequality does not hold when _a_ = _b_ = _c_ = 3[.][Specifically, out of the 162 possible] patterns for applying the AM-GM inequality in our running example, only six (i.e., _{u_ := _a, v_ := _b}_ and its symmetric or cyclical versions) yield correct deductions. Therefore, we propose using CAD to identify counterexamples in the new inequality goals and eliminate the scaling tactics that produce them. Moreover, since most scaling tactics are incorrect and thus induce counterexamples, CAD can efficiently detect and discard them using well-established heuristic strategies. 

To further enhance the efficacy and efficiency of scaling tactic pruning, we propose several additional methods to complement CAD in searching for counterexamples: 

**Quick Check via Test Cases.** When CAD identifies a counterexample for a scaling tactic, we store it as a “test case”. For any subsequent scaling tactic, we will perform a quick check using this test case before invoking CAD to determine if the counterexample invalidates the tactic. 

**Incorporating Numerical Optimization.** Since most inequality problems are differentiable, we also use gradient-based optimization as an effective alternative when CAD fails. Specifically, we rewrite the inequality _f_ ( _x_ ) _≤ g_ ( _x_ ) into min _x_ [ _g_ ( _x_ ) _− f_ ( _x_ )], and then integrate Newton’s method with simulated annealing to solve it (Fu & Su, 2016; Ma et al., 2019; Ni et al., 2023). 

4 

Published as a conference paper at ICLR 2025 

**Utilization of Prior Knowledge.** Additionally, prior knowledge can be leveraged in scaling tactic pruning when the specific form of an inequality problem is known. For example, the inequalities generated and evaluated by AIPS (Wei et al., 2024) are cyclically symmetric and sufficiently tight to achieve equality. Consequently, AIPS verifies the consistency of equality conditions across multiple tactics and discards subgoals that either violate these conditions or do not conform to the desired form. However, to accommodate a broader range of inequalities, our current framework does not incorporate such priors, even though this information could efficiently prune scaling tactics. 

## 3.2 REWRITING TACTICS 

After pruning, only a few scaling tactics are applicable. For example, we may choose to apply the two-variable AM-GM inequality (with pattern _{u_ := _a, v_ := _b}_ and its cyclical versions) and derive a new proof goal from the initial goal (1), formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0005-04.png)


**----- Start of picture text -----**<br>
1 1 1 1 1 1<br>a [2] + 2 [+] b [2] + 2 [+] c [2] + 2 [≤] 3 a [2] + 3 b [2] +  c [2] [+] 3 b [2] + 3 c [2] +  a [2] [+] 3 a [2] + 3 c [2] +  b [2] [.] (3)<br>**----- End of picture text -----**<br>


To solve the current proof goal, we need some equivalent transformations. For example, one may simplify the proof goal by replacing 3 _a_[2] +31 _b_[2] + _c_[2][with] 3 _−_ 12 _c_[2][using the assumption] _[ a]_[2][ +] _[ b]_[2][ +] _[ c]_[2][= 1][.] However, figuring out effective ways to transform the goal needs creativity and cannot be readily automated through brute-force methods. Consequently, relying solely on symbolic pattern matching tends to be highly ineffective. Moreover, the argument space for rewriting tactics is typically infinite. For example, the assumption _a_[2] + _b_[2] + _c_[2] = 1 can be inserted at almost any point within the inequality, significantly enlarging the space of possible rewriting tactics. 

To generate and prune the rewriting tactics, we propose directly prompting an LLM to generate candidates. Specifically, for different types of rewriting (e.g., rearrangement, simplification, denominator cancellation, etc.), we design tailored prompts that guide the LLM to transform the current proof goal into an appropriate form. The details of these prompts can be found in Appendix C. In this setting, tactic pruning is performed implicitly leveraging the algebraic intuition of LLMs, thereby effectively reducing the space of possible rewriting tactics. 

Besides neural-guided rewriting, we incorporate two additional symbolic rewriting tactics: the sumof-squares (Chen, 2013) and the tangent line (Li, 2005) tricks. The sum-of-squares trick attempts to transform the current expression into a summation of non-negative terms, e.g., proving 2 _x_[2] +2 _xy −_ 3 _x_ + _y_[2] + 4[9] _[≥]_[0][ by recognizing][ 2] _[x]_[2][ + 2] _[xy][ −]_[3] _[x]_[ +] _[ y]_[2][ +][9] 4[=][(] _[x]_[ +] _[ y]_[)][2][ + (] _[x][ −]_[3] 2[)][2][.][The tangent] line trick is a powerful variable substitution technique that is frequently used to tackle inequality problems in math competitions. 

## 4 GOAL FILTERING AND SELECTION 

Scaling and rewriting tactics yield a collection of new proof goals. By combining these newly generated goals with existing unexplored ones, we can derive a candidate set. The next step is to select the most promising goal from this candidate set for subsequent proving. In our running example, we can derive 16 new proof goals from the current goal (3). Below, we present three of these new goals, each derived by applying the two-variable AM-GM inequality, the three-variable Titu inequality, and simplifying using the given assumption _a_[2] + _b_[2] + _c_[2] = 1, respectively: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0005-10.png)


**----- Start of picture text -----**<br>
1 1 1 1 1 1<br>a [2] + 2 [+] b [2] + 2 [+] c [2] + 2 [≤] [2] � a [2] + 3 b [2] + 3 c [2] [·] b [2] + 3 a [2] + 3 c [2] [+] c [2] + 3 a [2] + 3 b [2] [;] (4)<br>1 1 1 9<br>a [2] + 2 [+] b [2] + 2 [+] c [2] + 2 [≤] 7 a [2] + 7 b [2] + 7 c [2] [;] (5)<br>1 1 1 1 1 1<br>a [2] + 2 [+] b [2] + 2 [+] c [2] + 2 [≤] 3  − 2 a [2] [+] 3  − 2 b [2] [+] 3  − 2 c [2] [.] (6)<br>**----- End of picture text -----**<br>


To select proof goals effectively, existing methods often train a language model as a value function to evaluate and rank each goal in the candidate set. However, limitations in data quantity and diversity can significantly degrade the performance of these fine-tuned models. For example, the largest 

5 

Published as a conference paper at ICLR 2025 

collected inequality dataset in Lean comprises only 46K samples (Ying et al., 2024), while the largest synthesized dataset of competition-level inequalities contains merely 191K cyclically symmetric instances (Wei et al., 2024). Consequently, we opt to directly employ an off-the-shelf LLM (e.g., GPT-4o) without fine-tuning. However, as more proof goals accumulate, the context length required for prompting an LLM for goal ranking becomes substantial, potentially leading to the issue of LLMs being “lost in the middle” (Liu et al., 2024). To address this, we propose dividing the goal selection pipeline into two stages: symbolic filtering and neural ranking. 

## 4.1 SYMBOLIC FILTERING 

First, we eliminate proof goals that are less promising based on carefully designed symbolic rules. Specifically, we prioritize two key properties: (1) _homogeneity_ , meaning both sides of the inequality have the same degree (e.g., _a_[2] + _b_[2] _≥_ 2 _ab_ ); and (2) _decoupling_ , which refers to whether the inequality contains mixed-variable terms (e.g., _abc_ is considered coupled). Both properties are reasonable criteria for prioritization. Regarding homogeneity, most substitution and transformation tactics preserve homogeneity. Hence, a homogeneous inequality allows for a broader range of tactics to be applied, thereby producing more valuable proof goals. For decoupling, an inequality with fewer coupled terms is not only clearer but also amenable to a greater variety of techniques, such as the sum-of-squares and tangent line tricks. 

To measure decoupling and homogeneity, we first approximate the proof goal by a polynomial inequality using Taylor expansion. We then compute the expectation of the number of variables in each term and the variance of the degree of each term, respectively. Formally, given a polynomial inequality expressed as Φ( _x_ 1 _, . . . , xn_ ) =[�] _[k] i_ =1 _[a][i][ ·][ x]_ 1 _[i]_[1] _[x]_ 2 _[i]_[2] _[· · ·][ x] n[i][n][≤]_[0][, the decoupling score (DC)] and homogeneity score (HM) are computed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0006-05.png)


where _di_ = _i_ 1 + _· · ·_ + _in_ is the total degree of _i_ -th term in the inequality. 

It is worth noting that the homogeneity score and the decoupling score are not always consistent. In our running example, the newly generated goals in (4), (5), and (6) achieve homogeneity scores of 0.56, 0.55, and 0.80, and decoupling scores of 0.44, 0.48, and 0.66, respectively. As a result, we normalize the scores into [0 _,_ 1] and then compute the average score to filter the candidates. 

## 4.2 NEURAL RANKING 

Symbolic rules are not universally effective. Hence, we use these rules solely to eliminate unpromising proof goals, leaving top- _k_ candidates, for final selection by an LLM. Unlike symbolic filtering, which requires explicit definitions of inequality metrics, we use the chain-of-thought prompting (Wei et al., 2022; Chu et al., 2023) to query an LLM to rank the proof goals based on their proving difficulty. The detailed prompt for the running example is shown below. 

## Prompt of neural ranking 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0006-11.png)


**----- Start of picture text -----**<br>
I am trying to prove the original inequality: “ If a, b, c are positive reals and a [2] +  b [2] +  c [2] = 1, then<br>a [2] 1+2 [+] b [2] 1+2 [+] c [2] 1+2 [≤] 3 a [2] +31 b [2] + c [2] [+] 3 b [2] +31 c [2] + a [2] [+] 3 a [2] +31 c [2] + b [2] [”, and transform it into the following]<br>inequalities.<br>1 1 1 1 1 1<br>(1) a [2] +2 [+] b [2] +2 [+] c [2] +2 [≤] [2] � a [2] +3 b [2] +3 c [2] [·] b [2] +3 a [2] +3 c [2] [+] c [2] +3 a [2] +3 b [2]<br>(2) a [2] 1+2 [+] b [2] 1+2 [+] c [2] 1+2 [≤] 7 a [2] +79 b [2] +7 c [2]<br>(3) a [2] 1+2 [+] b [2] 1+2 [+] c [2] 1+2 [≤] 3 − 12 a [2] [+] 3 − 12 b [2] [+] 3 − 12 c [2]<br>Your task is to rank the transformation results in a descent order. Note that<br>1. Please reason step by step;<br>2. More meaningful transformation, i.e., reduce the proving difficulty, should be ranked higher;<br>3. Put the index of selected inequality within  \\ boxed {{}} , e.g.,  \\ boxed {{ (1),(2),(3) }} .<br>**----- End of picture text -----**<br>


6 

Published as a conference paper at ICLR 2025 

Table 1: Proof success rates (%) on three datasets. LIPS consistently outperforms baselines. 

|Dataset<br># of Problems|Neural Provers|Symbolic Provers|Symbolic Provers|
|---|---|---|---|
|ChenNEQ<br>41<br>MO-INT<br>20<br>567NEQ<br>100|0.0<br>17.0<br>-<br>0.0<br>15.0<br>50.0<br>0.0<br>4.0<br>-|70.7<br>68.2|**95.1**<br>24.4_↑_|
|||60.0<br>60.0|**80.0**<br>20.0_↑_|
|||54.0<br>52.0|**68.0**<br>14.0_↑_|
|Total<br>161|0.0<br>8.6<br>-|59.0<br>57.1|**76.3**<br>17.3_↑_|



> _†_ We include the results reported by the AIPS paper since the code is not publicly available. 

> _‡_ CAD and MMA only output verification results, they cannot produce human-readable proofs. 

## 5 EXPERIMENTS 

In this section, we conduct a series of experiments to address the following three research questions: 

**RQ1: Efficacy** – Can LIPS prove more problems compared to existing methods? 

**RQ2: Efficiency** – Can LIPS obtain proofs in less time compared to existing methods? 

**RQ3: Scalability** – Can LIPS be improved by using more scaling lemmas or more powerful LLMs? 

## 5.1 EXPERIMENTAL SETUP 

**Datasets.** We evaluate LIPS on three datasets: ChenNEQ, MO-INT-20, and 567NEQ, respectively. ChenNEQ consists of 41 Olympiad-level inequalities collected by Chen (2014); MO-INT is a new competition-level inequality benchmark introduced in AIPS (Wei et al., 2024), featuring 20 problems sourced from IMO shortlists and various national mathematical Olympiads; 567NEQ consists of 567 hard inequalities created by Tung (2012) and we randomly selected 100 problems from the original problem set as the testbed for our framework. To formalize the problems in Lean 4, we directly translate the LaTeX source code into Lean using manually defined rules. 

**Baselines.** We compare LIPS with five baselines: DSP (Jiang et al., 2023b), CAD (Kremer, 2020), MMA (Wolfram Research), MCTS (Wu et al., 2021), and AIPS (Wei et al., 2024). DSP consists of two steps, natural language reasoning generation and proof autoformalization, and we instantiate the LLM used in each step GPT-4o. MCTS (Monte Carlo tree search) has been explored in previous studies (Wu et al., 2021) and serves as an alternative method for proof goal selection. AIPS is an inequality prover system based on SymPy, which has demonstrated the capability to prove competition-level inequalities. CAD integrates a series of CAD-based inequality solvers including Z3 (De Moura & Bjørner, 2008), CVC5 (Kremer et al., 2022), RC-CAD (Lemaire et al., 2005), and Bottema (Lu, 1998). MMA, referring to Mathematica, incorporates the CAD algorithm with other reduction strategies, providing a powerful algebraic system for inequality verification (Wolfram Research, 2020). Further implementation details for the baseline methods are provided in Appendix C. 

**Implementation.** The detailed processes of tactic generation, goal selection, as well as the overall framework are provided in Appendix B. To construct the tactics, we design a total of 96 scaling tactics and 16 rewriting tactics, each formalized in Lean 4. The corresponding premises and LLM prompts are summarized in Appendix C. For the counterexample search in scaling tactic pruning, we integrate four CAD-based solvers (Z3, CVC5, RC-CAD, and Bottema) and implement an optimizer based on SciPy (Virtanen et al., 2020). For the LLM involved in transformation tactic generation and proof goal ranking, we use GPT-4o (version Azure-0501). In symbolic filtering, we fix the size of the filtered goal set to 10, as it is the largest size that ensures GPT-4o’s efficacy. The code, together with the experimental data, is available at https://github.com/Lizn-zn/NeqLIPS. 

## 5.2 EXPERIMENTS 

**RQ1 : Efficacy.** We evaluate the proof success rates of LIPS and the five comparative methods across the three datasets. For each proving task, a time limit of 90 minutes is imposed, consistent with that of AIPS and the standard problem-solving time constraint in the IMO. The overall results are presented in Table 1. First, we observe that the neural methods (DSP and MCTS) cannot achieve satisfactory performance. An analysis of DSP’s results reveals that GPT-4o is unable to provide ac- 

7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
10090 8689 AIPS88 92 85LIPS8490 95 87 86 9292 8596 9494 9797 90 8994 8490 363024 22 Oracle 24LIPS 29 22 33<br>8070 78 81 78 75 66 74 18126 9 717 614 9 6 6 6 11 6 11 16 7 5 8 6 8 815 4 5 7 813 614<br>60 61 62 60 0<br>1 2 3 4 5 8 9 10 11 13 14 15 16 17 18 19 1 2 3 4 5 8 9 10 11 13 14 15 16 17 18 19<br>Index of successfully proved problems Index of successfully proved problems<br>Tactic pruning ratio (%) Number of iterations<br>**----- End of picture text -----**<br>


Figure 3: Tactic pruning ratio ( _↑_ , higher is better) and number of iterations ( _↓_ ). The results illustrate that the tactic pruning method of LIPS is very stable, and outperforms the existing method by 7.92% on average. Furthermore, the high efficiency of LIPS is derived from accurate goal selection, allowing a proof to be successfully constructed with a small number of goal selection iterations. 

curate natural language solutions or generate precise formal proofs in the Lean 4 language, resulting in a zero success rate. Alternatively, MCTS struggles to effectively identify the correct reasoning path among numerous proof goals, causing many proving attempts to terminate due to timeouts. We also evaluate the performance of the recent OpenAI o1-preview model on the MO-INT dataset. Through manual inspection of the generated natural language answers, we find that none of the problems are correctly solved. Examples of these neural methods are provided in Appendix E. 

Symbolic provers outperform neural provers, CAD and MMA achieve overall success rates of 59.0% and 57.1%, respectively. However, LIPS further surpasses symbolic provers by a significant margin of 14.0% to 24.4%. Notably, symbolic provers fail to produce any human-readable reasoning path. In contrast, the proofs generated by LIPS are not only accessible and human-readable, but also have been successfully verified by the Lean theorem prover. For reference, we include two examples (one showcasing a successful proof and the other demonstrating a failed attempt) in Appendix E. 

**RQ2 : Efficiency.** Since existing methods are built on different deduction engines, a direct comparison of their proving time could be unfair. Instead, we break down the efficiency evaluation into two aspects, i.e., the pruning ratio of scaling tactics and the number of iterations in goal selection. Given that AIPS uses the equality check as the scaling tactic pruning strategy, we compare this approach with the CAD-based strategy employed in LIPS. Figure 3 provides the result of each problem in the MO-INT dataset. LIPS outperforms the existing method in 13 out of 20 problems, and achieves an average improvement of 7.92%. 

Furthermore, we count the number of goal selection iterations for each successfully proved problem, and present the results in Figure 3. Due to the absence of a comparison method, we only include the oracle (i.e., optimal goal selection) as a reference. We can observe that LIPS performs no more than 33 search loops to successfully obtain a proof, and for 12 out of 16 problems, it exceeds the oracle by fewer than 10 steps. In addition, LIPS requires an average of 15.75 search loops, which is only 2.17 times that of the oracle (7.25), demonstrating that LIPS’s efficiency also stems from its high accuracy in generating proving paths. 

**RQ3 : Scability.** We conduct four experiments on the ChenNEQ dataset to explore the scalability of LIPS’s symbolic and neural components. For the symbolic part, we first examine how expanding the scaling tactics affects the performance of LIPS. To this end, we randomly select 7 sets of scaling tactics with varying sizes and plot the performance curve in Figure 4(a). The results show that the proof success rate consistently increases as more scaling tactics are included, suggesting potential benefits in further enlarging our scaling tactic library. The second experiment investigates the effect of different sizes of the filtered set. The corresponding performance curve is shown in Figure 4(b). We observe that the success rate remains robust (over 85%) with sizes from 8 to 16, but decreases significantly when the set is either too small or too large. 

For the neural part, we explore the performance of LIPS with different LLMs serving in rewriting tactic generation and neural ranking. We select three alternative LLMs, i.e., Mathstral 7B (Jiang et al., 2023a), LLaMA-3 8B (AI, 2024), and DeepSeek-chat V2.5 (DeepSeek-AI, 2024). We also include a baseline method as an ablative study. In rewriting tactic generation, the baseline uses SymPy simplify function instead of existing LLM-based rewriting tactics. In neural ranking, we directly use random selection as the baseline. The proving success rates are provided in Figure 5. The re- 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Baseline Mathstral LLaMA-3 Deepseek-chat GPT4-o<br>100 100<br>100 95.1 92.7 95.1<br>80 80 85.4 80.5 87.8<br>60 60 75 73.2 63.4<br>40 40 50 51.2<br>20 20<br>25 17.1<br>0 0<br>0 16 32 48 64 80 96 0 4 8 12 16 20 24 28 32<br>Various sizes of scaling tactics Various sizes of filtered set 0 Tactic Generation Neural Ranking<br>Various LLMs in rewriting tactic generation and neural ranking<br>Proving success rate (%) Proving success rate (%)<br>Proving success rate (%)<br>**----- End of picture text -----**<br>


Figure 4: Performance curves across different scaling tactics and filtered set sizes. Results indicate that: (1) Adding scaling tactics consistently improves the proving success rate; (2) The effectiveness of symbolic filtering and the size of the filtered set are critical factors. 

Figure 5: Performance of various LLMs as alternatives. Results illustrate that: (1) LLMs are crucial in rewriting tactic generation and neural ranking; (2) The more powerful LLM can achieve higher proving success rate. 

sults demonstrate that all three alternative LLMs exhibit strong mathematical intuition, achieving performance comparable to GPT-4o in neural ranking. However, there exists a small decline in performance for rewriting tactic generation, which may be due to differences in instruction following and mathematical reasoning capabilities. 

**Ablation study.** To showcase the strength of our neuro-symbolic paradigm, we present the results of removing neural or symbolic modules of LIPS in Appendix D. We also analyze the performance of symbolic solvers, offering guidance on optimal time limit settings in LIPS. 

**Case study.** Besides the running example (1), we provide two additional examples in Appendix F to illustrate that LIPS can discover new proofs, which are previously unavailable online. Moreover, we also present two examples in Appendix F to demonstrate that users can verify human-written proofs by comparing them with the reasoning paths generated by LIPS. 

## 6 RELATED WORK 

**Symbolic Tools for Mathematical Reasoning.** Symbolic tools are essential for performing exact computations and formal reasoning in mathematics. Interactive theorem provers such as Isabelle (Paulson, 1994), Coq (Coq, 1996), and Lean (de Moura et al., 2015) enable users to build verifiable proofs manually, ensuring correctness through rigorous formal logic. These systems have been instrumental in formalizing and verifying significant mathematical theorems, including the Four Color Theorem (Gonthier, 2008) and the Kepler Conjecture (Hales et al., 2017). Alternatively, some symbolic reasoning tools aim to solve mathematical problems without human intervention. Automated theorem provers like E (Schulz, 2002) and Vampire (Kov´acs & Voronkov, 2013) are designed to prove mathematical statements by systematically exploring possible proofs within a logical framework, particularly excelling in first-order logic. SMT solvers such as Z3 (De Moura & Bjørner, 2008) and CVC5 (Barbosa et al., 2022) determine the satisfiability of logical formulas with respect to background theories like arithmetic, bit-vectors, and arrays by integrating logical reasoning with theory-specific decision procedures. Computer algebra systems like Mathematica (Wolfram Research), Maple (Heck & Koepf, 1993), and SymPy (Meurer et al., 2017) manipulate mathematical expressions symbolically, supporting functionalities such as simplification, differentiation, integration, and equation solving. Despite their capabilities, these automated solvers struggle with competitive mathematical problems and often cannot generate human-readable reasoning steps. 

**Machine Learning for Formal Theorem Proving.** There is a longstanding tradition of leveraging machine learning techniques to automate theorem proving (Urban et al., 2008; Gauthier et al., 2017; Zhang et al., 2021; Piotrowski et al., 2023; Blaauwbroek et al., 2024). Recently, the emergence of LLMs has expanded the potential of these techniques, offering new opportunities for automating theorem proving (Li et al., 2024; Yang et al., 2024). A line of research (Polu & Sutskever, 2020; Wu et al., 2021; Han et al., 2022; First et al., 2023; Yang et al., 2023; Xin et al., 2024a; Lin et al., 2025) fine-tunes pre-trained language models on large-scale formal datasets to predict the tactics given a proof goal. Alternative approaches (Jiang et al., 2023b; Xin et al., 2024b; Zhao et al., 2024; Zheng et al., 2024; Thakur et al., 2024) integrate LLMs into structured prompt frameworks for formal theorem proving, leveraging information such as natural language proofs or feedback from 

9 

Published as a conference paper at ICLR 2025 

interactive theorem provers. Some methods (Polu & Sutskever, 2020; Lample et al., 2022; Polu et al., 2023; Wang et al., 2023; Wei et al., 2024) also train LLMs as value networks to evaluate each subgoal and guide the proof search to completion. However, these methods primarily rely on LLMs to prune the search space, and their performance heavily depends on the quality and diversity of the training data, which may limit their generalizability to novel or more complex mathematical problems. A notable exception and closely related work to ours is AlphaGeometry (Trinh et al., 2024), which uses a language model only to predict auxiliary construction rules, after which its symbolic solver automatically enumerates all inference rules to generate the proof in a specialized language used in GEX (Chou et al., 2000). Compared to their approach, our neuro-symbolic framework generates step-by-step proofs in general-purpose formal language Lean and allows for integrating arbitrary symbolic tools that do not require direct proof generation. Furthermore, our focus on inequality problems spans a much wider range of complex mathematical skills than plane geometry problems. Our techniques could serve as a solid foundation for broader areas of mathematical research such as information theory (Dembo et al., 1991), optimization (Nesterov, 2013), and deep learning (Roberts et al., 2022), making it a suitable pathway to more advanced mathematical problems. 

## 7 LIMITATIONS AND FUTURE WORK 

While LIPS has shown significant promise in generating formal proofs for Olympiad inequalities, several avenues remain open for enhancement and expansion. 

**Automating the Formalization of Tactics.** Our framework currently relies on a set of manually crafted tactics for scaling and rewriting inequalities, such as various forms of AM-GM inequality. This manual effort may impact scalability, given that the effectiveness of our approach is closely tied to the breadth of available tactics. Future work could focus on automating the discovery, formalization, and proof of new tactics to expand the tactic library. Developing methods for automatic tactic generation would reduce human effort and enhance the framework’s scalability and adaptability. 

**Enhancing the Reasoning Capabilities of LLMs.** We leverage the mathematical insights learned by LLMs in our framework, and there is potential to further improve their reasoning performance. One promising direction is to collect or generate additional formal inequality problems and their corresponding proofs to create a richer dataset for fine-tuning LLMs specifically for this task. Some existing techniques (Li et al., 2025) may be useful for generating diverse and high-quality problems to enhance the LLMs’ capabilities in handling inequalities, leading to better overall performance. 

**Broadening the Application Domain.** While our framework currently focuses on Olympiad-level elementary algebraic inequalities, extending it to more complex problems, such as concentration inequalities in machine learning theory, presents an exciting avenue for future research. This would involve improving the symbolic solver to handle inequality structures that consist of infinite variables and higher-order concepts like expectations or variances. Developing efficient algorithms and symbolic reasoning methods for these advanced mathematical constructs could significantly broaden the applicability of our neuro-symbolic paradigm. Extending our approach to other mathematical domains holds great potential and is a promising direction for future work. 

## 8 CONCLUSION 

In this paper, we introduce a neuro-symbolic framework for generating formal proofs that integrates the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods, specifically focusing on the domain of Olympiad inequalities. We categorize the tactics used in inequality proofs into two types: scaling and rewriting. Symbolic methods are employed to generate and filter scaling tactics by applying a set of lemmas through mechanical symbolic reasoning. LLMs are leveraged to generate rewriting tactics, implicitly pruning the infinite number of equivalent transformations to a manageable set. We further combine symbolic tools with LLMs to prune and rank subgoals, enhancing the efficiency of proof search. Experiments on challenging inequalities from three problem sets show that our neuro-symbolic inequality prover LIPS significantly outperforms both LLMs and symbolic methods, demonstrating the effectiveness of the neuro-symbolic integration and laying a solid foundation for its adoption in broader domains. 

10 

Published as a conference paper at ICLR 2025 

## ACKNOWLEDGMENT 

We appreciate anonymous reviewers for their valuable comments and engaging discussions. This work was partially conducted during Zenan’s and Wen’s internships at MSRA. It is supported by the National Natural Science Foundation of China (Grants #62025202) and the Frontier Technologies R&D Program of Jiangsu (BF2024059). Zhaoyu Li and Xujie Si were also supported, in part, by Individual Discovery Grants from the Natural Sciences and Engineering Research Council of Canada, and the Canada CIFAR AI Chair Program. Xian Zhang (zhxian@microsoft.com), Kaiyu Yang (kaiyuy@meta.com), and Xiaoxing Ma (xxm@nju.edu.cn) are the corresponding authors. 

## REFERENCES 

Meta AI. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/ blob/main/MODEL_CARD.md. 

- Dennis S Arnon, George E Collins, and Scott McCallum. Cylindrical algebraic decomposition I: The basic algorithm. _SIAM Journal on Computing_ , 1984. 

- Haniel Barbosa, Clark Barrett, Martin Brain, Gereon Kremer, Hanna Lachnitt, Makai Mann, Abdalrhman Mohamed, Mudathir Mohamed, Aina Niemetz, Andres N¨otzli, et al. cvc5: A versatile and industrial-strength SMT solver. In _International Conference on Tools and Algorithms for the Construction and Analysis of Systems_ , 2022. 

Clark Barrett and Cesare Tinelli. _Satisfiability modulo theories_ . 2018. 

- Lasse Blaauwbroek, Mirek Olˇs´ak, Jason Rute, Fidel Ivan Schaposnik Massolo, Jelle Piepenbrock, and Vasily Pestun. Graph2Tac: Online representation learning of formal math concepts. In _International Conference on Machine Learning (ICML)_ , 2024. 

- Bob F Caviness and Jeremy R Johnson. _Quantifier elimination and cylindrical algebraic decomposition_ . Springer Science & Business Media, 2012. 

Evan Chen. Supersums of square-weights (SOS), 2013. 

- Evan Chen. A brief introduction to olympiad inequalities. 2014. URL https://web. evanchen.cc/handouts/Ineq/en.pdf. 

- Shang-Ching Chou, Xiao-Shan Gao, and Jing-Zhong Zhang. A Deductive Database Approach to Automated Geometry Theorem Proving and Discovering. _Journal of Automated Reasoning_ , 2000. 

- Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. A survey of chain of thought reasoning: Advances, frontiers and future. _arXiv preprint arXiv:2309.15402_ , 2023. 

- Project Coq. The Coq proof assistant-reference manual. _INRIA Rocquencourt and ENS Lyon, version_ , 5, 1996. 

- James H Davenport and Joos Heintz. Real quantifier elimination is doubly exponential. _Journal of Symbolic Computation_ , 1988. 

- Leonardo De Moura and Nikolaj Bjørner. Efficient e-matching for SMT solvers. In _Automated Deduction–CADE-21: 21st International Conference on Automated Deduction Bremen, Germany, July 17-20, 2007 Proceedings 21_ , pp. 183–198. Springer, 2007. 

- Leonardo De Moura and Nikolaj Bjørner. Z3: An efficient SMT solver. In _International Conference on Tools and Algorithms for the Construction and Analysis of Systems_ , 2008. 

- Leonardo de Moura, Soonho Kong, Jeremy Avigad, Floris Van Doorn, and Jakob von Raumer. The Lean theorem prover (system description). In _International Conference on Automated Deduction (CADE)_ , 2015. 

- DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model, 2024. 

11 

Published as a conference paper at ICLR 2025 

- Amir Dembo, Thomas M Cover, and Joy A Thomas. Information theoretic inequalities. _IEEE Transactions on Information theory_ , 1991. 

- Emily First, Markus Rabe, Talia Ringer, and Yuriy Brun. Baldur: Whole-proof generation and repair with large language models. In _ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)_ , 2023. 

- Zhoulai Fu and Zhendong Su. XSat: A fast floating-point satisfiability solver. In _International Conference on Computer Aided Verification (CAV)_ , 2016. 

- Thibault Gauthier, Cezary Kaliszyk, and Josef Urban. Learning to reason with HOL4 tactics. In _International Conference on Logic for Programming, Artificial Intelligence and Reasoning_ , 2017. 

- Georges Gonthier. The Four Colour Theorem: Engineering of a Formal Proof. In _Proceedings of the Asian Symposium on Computer Mathematics_ , 2008. 

Google DeepMind. AI achieves silver-medal standard solving international mathematical olympiad problems. https://deepmind.google/discover/blog/ ai-solves-imo-problems-at-silver-medal-level/, 2024. 

- Thomas Hales, Mark Adams, Gertrud Bauer, Tat Dat Dang, John Harrison, Hoang Le Truong, Cezary Kaliszyk, Victor Magron, Sean McLaughlin, Tat Thang Nguyen, et al. A Formal Proof of the Kepler Conjecture. In _Forum of Mathematics, Pi_ , 2017. 

- Kristian J Hammond and David B Leake. Large language models need symbolic AI. In _International Workshop on Neural-Symbolic Learning and Reasoning,_ , 2023. 

- Jesse Michael Han, Jason Rute, Yuhuai Wu, Edward W Ayers, and Stanislas Polu. Proof Artifact CoTraining for Theorem Proving with Language Models. In _International Conference on Learning Representations (ICLR)_ , 2022. 

Godfrey Harold Hardy, John Edensor Littlewood, and George P´olya. _Inequalities_ . 1952. 

Andr´e Heck and Wolfram Koepf. _Introduction to Maple_ . 1993. 

- Marijn JH Heule, Oliver Kullmann, and Victor W Marek. Solving and verifying the boolean Pythagorean triples problem via cube-and-conquer. In _International Conference on Theory and Applications of Satisfiability Testing_ , 2016. 

- Bruce Ikenaga. Techniques in proving inequalities, Sep 2018. URL https://sites. millersville.edu/bikenaga/math-proof/inequalities/inequalities. html. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. _arXiv preprint arXiv:2310.06825_ , 2023a. 

- Albert Q Jiang, Sean Welleck, Jin Peng Zhou, Wenda Li, Jiacheng Liu, Mateja Jamnik, Timoth´ee Lacroix, Yuhuai Wu, and Guillaume Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In _International Conference on Learning Representations (ICLR)_ , 2023b. 

Mats Jirstrand. _Cylindrical algebraic decomposition-an introduction_ . Link¨oping University, 1995. 

- Dejan Jovanovi´c and Leonardo de Moura. Solving non-linear arithmetic. _ACM Communications in Computer Algebra_ , 2013. 

- Levente Kocsis and Csaba Szepesv´ari. Bandit based Monte-Carlo planning. In _European conference on machine learning_ , 2006. 

- Laura Kov´acs and Andrei Voronkov. First-order theorem proving and Vampire. In _International Conference on Computer Aided Verification (CAV)_ , 2013. 

- Gereon Kremer. _Cylindrical algebraic decomposition for nonlinear arithmetic problems_ . PhD thesis, Dissertation, RWTH Aachen University, 2020, 2020. 

12 

Published as a conference paper at ICLR 2025 

- Gereon Kremer, Andrew Reynolds, Clark Barrett, and Cesare Tinelli. Cooperating techniques for solving nonlinear real arithmetic in the cvc5 SMT solver (system description). In _International Joint Conference on Automated Reasoning_ , 2022. 

- Guillaume Lample, Timothee Lacroix, Marie-Anne Lachaux, Aurelien Rodriguez, Amaury Hayat, Thibaut Lavril, Gabriel Ebner, and Xavier Martinet. Hypertree Proof Search for Neural Theorem Proving. In _Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Hojoo Lee. Topics in inequalities-theorems and techniques. _Korea Institute for Advanced Study, Seoul_ , 2004. 

- Franc¸ois Lemaire, M Moreno Maza, and Yuzhen Xie. The regularchains library in Maple. _ACM SIGSAM Bulletin_ , 2005. 

- Kin-Yin Li. Using tangent lines to prove inequalities. _Mathematical Excalibur_ , 2006:1, 2005. 

- Zenan Li, Zhi Zhou, Yuan Yao, Xian Zhang, Yu-Feng Li, Chun Cao, Fan Yang, and Xiaoxing Ma. Neuro-symbolic data generation for math reasoning. _Advances in Neural Information Processing Systems_ , 37:23488–23515, 2025. 

- Zhaoyu Li, Jialiang Sun, Logan Murphy, Qidong Su, Zenan Li, Xian Zhang, Kaiyu Yang, and Xujie Si. A survey on deep learning for theorem proving. In _Conference on Language Modeling (COLM)_ , 2024. 

- Yong Lin, Shange Tang, Bohan Lyu, Jiayun Wu, Hongzhou Lin, Kaiyu Yang, Jia Li, Mengzhou Xia, Danqi Chen, Sanjeev Arora, et al. Goedel-prover: A frontier model for open-source automated theorem proving. _arXiv preprint arXiv:2502.07640_ , 2025. 

- Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_ , 2024. 

- Yang Lu. Practical automated reasoning on inequalities: Generic programs for inequality proving and discovering. _Asian Technology Conference in Mathematics_ , 1998. 

- Yi-An Ma, Yuansi Chen, Chi Jin, Nicolas Flammarion, and Michael I Jordan. Sampling can be faster than optimization. _Proceedings of the National Academy of Sciences (PNAS)_ , 2019. 

- Radmila Bulajich Manfrino, Jos´e Antonio G´omez Ortega, and Rogelio Valdez Delgado. _Inequalities: a mathematical olympiad approach_ . 2010. 

- Aaron Meurer, Christopher P Smith, Mateusz Paprocki, Ondˇrej Cert´ık, Sergey B Kirpichev, Matthew[ˇ] Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K Moore, Sartaj Singh, et al. SymPy: symbolic computing in python. _PeerJ Computer Science_ , 2017. 

- Michał Moskal, Jakub Łopusza´nski, and Joseph R Kiniry. E-matching for fun and profit. _Electronic Notes in Theoretical Computer Science_ , 198(2):19–35, 2008. 

Yurii Nesterov. _Introductory lectures on convex optimization: A basic course_ . 2013. 

- Allen Newell and Herbert Simon. The logic theory machine–a complex information processing system. _IRE Transactions on information theory_ , 1956. 

- Xinpeng Ni, Yulun Wu, and Bican Xia. Solving SMT over non-linear real arithmetic via numerical sampling and symbolic verification. In _International Symposium on Dependable Software Engineering: Theories, Tools, and Applications_ , 2023. 

Lawrence C Paulson. _Isabelle: A Generic Theorem Prover_ . 1994. 

- Bartosz Piotrowski, Ramon Fern´andez Mir, and Edward Ayers. Machine-learned premise selection for Lean. In _International Conference on Automated Reasoning with Analytic Tableaux and Related Methods_ , 2023. 

- Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. _arXiv preprint arXiv:2009.03393_ , 2020. 

13 

Published as a conference paper at ICLR 2025 

- Stanislas Polu, Jesse Michael Han, Kunhao Zheng, Mantas Baksys, Igor Babuschkin, and Ilya Sutskever. Formal Mathematics Statement Curriculum Learning. In _International Conference on Learning Representations (ICLR)_ , 2023. 

- Daniel A Roberts, Sho Yaida, and Boris Hanin. _The principles of deep learning theory_ . Cambridge University Press Cambridge, MA, USA, 2022. 

Stephan Schulz. E–a brainiac theorem prover. _AI Communications_ , 2002. 

- Peiyang Song, Kaiyu Yang, and Anima Anandkumar. Towards large language models as copilots for theorem proving in Lean. _arXiv preprint arXiv: Arxiv-2404.12534_ , 2024. 

- Amitayush Thakur, George Tsoukalas, Yeming Wen, Jimmy Xin, and Swarat Chaudhuri. An in-context learning agent for formal theorem-proving. In _Conference on Language Modeling (COLM)_ , 2024. 

- Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. _Nature_ , 2024. 

- Nguyen Duy Tung. 567 nice and hard inequality. https://phamtuankhai.wordpress. com/wp-content/uploads/2012/04/567-bat-dang-thuc-hay.pdf, April 2012. 

- Ali K Uncu, James H Davenport, and Matthew England. SMT-solving induction proofs of inequalities. _arXiv preprint arXiv:2307.16761_ , 2023. 

- Marcell J´anos Uray. On proving inequalities by cylindrical algebraic decomposition. In _Annales Universitatis Scientiarum Budapestinensis de Rolando Eotvos Nominatae. Sectio Computatorica_ , 2020. 

- Josef Urban, Geoff Sutcliffe, Petr Pudl´ak, and Jiˇr´ı Vyskoˇcil. MaLARea SG1—machine learner for automated reasoning with semantic guidance. In _International Joint Conference on Automated Reasoning_ , 2008. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Neural Information Processing Systems (NeurIPS)_ , 2017. 

- Pauli Virtanen, Ralf Gommers, Travis E Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. _Nature methods_ , 2020. 

- Haiming Wang, Ye Yuan, Zhengying Liu, Jianhao Shen, Yichun Yin, Jing Xiong, Enze Xie, Han Shi, Yujun Li, Lin Li, et al. DT-Solver: Automated theorem proving with dynamic-tree sampling guided by proof-level value function. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2023. 

- Chenrui Wei, Mengzhou Sun, and Wei Wang. Proving olympiad algebraic inequalities without human demonstrations. In _Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track_ , 2024. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Neural Information Processing Systems (NeurIPS)_ , 2022. 

- Max Willsey, Chandrakana Nandi, Yisu Remy Wang, Oliver Flatt, Zachary Tatlock, and Pavel Panchekha. egg: Fast and extensible equality saturation. _Proc. ACM Program. Lang._ , 5(POPL), January 2021. doi: 10.1145/3434304. URL https://doi.org/10.1145/3434304. 

Wolfram Research. Mathematica, version 14.1. URL https://www.wolfram.com/ mathematica. 

- Wolfram Research. Cylindrical Decomposition, 2020. URL https://reference.wolfram. com/language/ref/CylindricalDecomposition.html. 

14 

Published as a conference paper at ICLR 2025 

Wen-ts¨un Wu. On the decision problem and the mechanization of theorem-proving in elementary geometry. In _Selected Works Of Wen-Tsun Wu_ , pp. 117–138. World Scientific, 2008. 

- Yuhuai Wu, Albert Q Jiang, Jimmy Ba, and Roger Grosse. INT: An inequality benchmark for evaluating generalization in theorem proving. In _International Conference on Learning Representations (ICLR)_ , 2021. 

- Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and Xiaodan Liang. DeepSeek-Prover: Advancing theorem proving in llms through large-scale synthetic data. _arXiv preprint arXiv:2405.14333_ , 2024a. 

- Huajian Xin, Haiming Wang, Chuanyang Zheng, Lin Li, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, et al. LEGO-Prover: Neural theorem proving with growing libraries. In _International Conference on Learning Representations (ICLR)_ , 2024b. 

- Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. LeanDojo: Theorem proving with retrieval-augmented language models. In _Neural Information Processing Systems (NeurIPS)_ , 2023. 

- Kaiyu Yang, Gabriel Poesia, Jingxuan He, Wenda Li, Kristin Lauter, Swarat Chaudhuri, and Dawn Song. Formal mathematical reasoning: A new frontier in AI. _arXiv preprint arXiv:2412.16075_ , 2024. 

- Lu Yang. Recent advances in automated theorem proving on inequalities. _Journal of Computer Science and Technology_ , 1999. 

- Huaiyuan Ying, Zijian Wu, Yihan Geng, Jiayu Wang, Dahua Lin, and Kai Chen. Lean Workbook: A large-scale Lean problem set formalized from natural language math problems. _arXiv preprint arXiv:2406.03847_ , 2024. 

- Liao Zhang, Lasse Blaauwbroek, Bartosz Piotrowski, Cezary Kaliszyk, and Josef Urban. Online machine learning techniques for Coq: a comparison. In _International Conference on Intelligent Computer Mathematics_ , 2021. 

- Xueliang Zhao, Wenda Li, and Lingpeng Kong. Subgoal-Based Demonstration Learning for Formal Theorem Proving. In _International Conference on Machine Learning (ICML)_ , 2024. 

- Chuanyang Zheng, Haiming Wang, Enze Xie, Zhengying Liu, Jiankai Sun, Huajian Xin, Jianhao Shen, Zhenguo Li, and Yu Li. Lyra: Orchestrating dual correction in automated theorem proving. _Transactions on Machine Learning Research (TMLR)_ , 2024. 

15 

Published as a conference paper at ICLR 2025 

## A SYMBOLIC METHODS VS. NEURAL METHODS 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0016-02.png)


**----- Start of picture text -----**<br>
Initial goal (to be proved) 𝑎𝑏+ 𝑏𝑐+ 𝑐𝑎≤𝑎 [!]  + 𝑏 [!]  + 𝑐 [!]<br>Tactic1 Cauchy-Schwarz<br>Intermediate goals’ (2a [!] + 𝑏 [!] )(2𝑐 [!] + 𝑏 [!] ) ≤𝑎 [!]  + 𝑏 [!]  + 𝑐 [!]<br>Tactic2<br>AM-GM<br>Intermediate goals’’ (2𝑎 [!] + 𝑏 [!] + 2𝑐 [!] + 𝑏 [!] )/2 ≤𝑎 [!]  + 𝑏 [!]  + 𝑐 [!]<br>Tactic3 Simplification<br>Final goal (trivial) 𝑎 [!]  + 𝑏 [!]  + 𝑐 [!] ≤𝑎 [!]  + 𝑏 [!]  + 𝑐 [!]<br>(a) An example of CAD (b) Inequality proving process (c) An example of proof<br>**----- End of picture text -----**<br>


Figure 6: Figure (a) demonstrates how CAD is performed on two intersecting unit circles, deriving multiple sign-invariant cells (i.e., colored region). Figure (b) illustrates the process of inequality proving, which constructs a chain of proof goals by iteratively applying tactics; Figure (c) provides a corresponding instantiation of proving _ab_ + _bc_ + _ca ≤ a_[2] + _b_[2] + _c_[2] . 

Figure 6(a) illustrates how symbolic methods work in proving inequalities. Symbolic methods are based on the CAD algorithm, which divides the underlying space R _[n]_ into multiple connected semialgebraic sets. In each cell, the sign of every polynomial remains constant (positive, negative, or zero). Therefore, we just need to sample one point to determine the satisfiability of each cell, rather than scanning the whole R _[n]_ space. One can refer to Caviness & Johnson (2012); Arnon et al. (1984); Jirstrand (1995) for more details of CAD algorithm. 

Figure 6(b) and 6(c) illustrate a commonly used paradigm in inequality proving. In a nutshell, a theorem prover often starts with the proof goal, and then iteratively transforms it into a simpler form until the final version can be easily confirmed. This approach has two main advantages. First, it ensures that the resulting proof can be more easily formalized in formal languages such as Lean. Second, since the hypotheses involved in inequality proofs are often straightforward, scaling and rewriting the proof goal is typically more efficient. 

## B PSEUDO CODE OF LIPS 

Algorithm (3) outlines the overall process of LIPS proof generation. The detailed steps for tactic generation and pruning are provided in Algorithm 1, while the specifics for goal filtering and ranking are described in Algorithm 2. 

**Algorithm 1** Tactic generation and pruning of LIPS 

**Input:** A proof goal _g_ ; A lemma library of scaling tactics Φ and a prompt set of rewriting tactics Ψ; A language model _M_ . **Output:** Tactic set _T_ . 1: Initialize the tactic set _T_ = _{}_ 2: **for** _t_ in Φ **do** _▷ Scaling tactic generation_ 3: Obtain arguments of the tactic _t_ using pattern ~~m~~ atch on the goal Φ. 4: Check the tactic _t_ (with derived arguments) via the symbolic solvers. 5: **if** no counterexamples exist **then** _▷ Tactic pruning & Solver update_ 6: Add the tactic _t_ into the tactic set _T_ . 7: **else** 8: Update the symbolic solvers by including newly detected counterexamples. 9: **end if** 10: **end for** 11: **for** _t_ in Ψ **do** _▷ Rewriting tactic generation & pruning_ 12: Obtain arguments of the tactic _t_ by prompting the LLM. 13: Add the tactic _t_ (with derived arguments) into the tactic set _T_ . 14: **end for** 

16 

Published as a conference paper at ICLR 2025 

**Algorithm 2** Goal filtering and ranking of LIPS 

**Input:** A goal candidate set Ω. **Output:** A ranked set Ω _[′]_ . 

1: Initialize a new goal set Ω _[′]_ = _{}_ . 2: **for** _g_ in Ω **do** 3: Compute the homogeneity _α_ and decoupling _β_ for _g_ . 4: Define the average score of the goal _g_ by ( _α_ + _β_ ) _/_ 2. 5: **end for** 

6: Prompt the LLM to rank the first ten goals in Ω _[′]_ 

_▷ Symbolic goal filtering ▷ Neural goal ranking_ 

7: Obtain arguments of the tactic _t_ by prompting the LLM. 

- 8: Re-rank the goal set Ω _[′]_ according to LLM’s responses. 

**Algorithm 3** Overall proof generation process of LIPS 

**Input:** A formal inequality problem _g_ 0; A lemma library of scaling tactics Φ and a prompt set of rewriting tactics Ψ; A language model _M_ . **Output:** A formal proof or timeout. 

1: Initialize the candidate goal set Ω= _{g_ 0 _}_ . 

2: **for** _i_ = 1 _, . . . ,_ **do** 3: Select the first goal _g_ in _S_ for exploration. 4: Obtain the tactic set _T_ by applying Algorithm (1) on the current goal _g_ . 5: **for** _t_ in _T_ **do** _▷ Tactic application_ 6: Apply the tactic to the current _g_ in Lean, deriving a new goal _g[′]_ . 7: Add the new goal _g[′]_ into the the candidate goal set Ω. 8: **end for** 9: Update the goal set Ω by applying Algorithm (2). 10: **end for** 

## C ADDITIONAL DETAILS FOR EXPERIMENTS 

The experiments were conducted on four Linux servers equipped with 4 Intel(R) Xeon(R) Platinum 8280L CPU @ 2.80GHz. Each server ran Ubuntu 22.04 with GNU/Linux kernel 6.5.0-1015-azure. Each proving task was performed within a docker sandbox, utilizing 192 assigned CPU cores. 

**Neural provers.** For DSP, we directly use the official code, and adapt it to Lean language and GPT-4o (version Azure-05-01). MCTS is implemented based on classic upper confidence bounds applied to trees algorithm (Kocsis & Szepesv´ari, 2006). The value function is defined as _f_ ( _ϕ_ ) = _vϕ_ + _C_ ~~�~~ log( _Nϕ_ ) _/nϕ_ , where _nϕ_ is the number of the proof goal _ϕ_ is selected and explored, _Nϕ_ represents the number of _ϕ_ ’s parent represents selected and explored, and _C_ is a hyperparameter set to _C_ = _√_ 2. For the average reward _vϕ_ of the proof goal _ϕ_ , we use the same heuristic function as Wei et al. (2024), which calculates the maximum depth of the expression trees on both sides. 

**Symbolic provers.** For CAD, we utilize a portfolio including a suite of solvers, i.e., Z3, CVC5, RC-CAD, and Bottema. It will claim the problem is successfully proved if any one of four tools outputs unsat, and vice vica. Among four tools, Z3 and CVC5 are two popular SMT solvers; RCCAD refers to the CylindricalAlgebraicDecompose function in Maple 2024 RegularChain package;[1] Bottema is a CAD-based inequality prover developed by Yang (1999).[2] As to MMA, we integrate two commands, i.e., Reduce and FindInstance in Wolfram-Mathematica (version 13.0.1), and apply the same peripheral logic with CAD. 

**LIPS.** In our framework, the symbolic solver employed for pruning scaling tactics also consists of solvers Z3, CVC5, RC-CAD, and Bottema, complemented by a numerical optimizer grounded in SciPy (Virtanen et al., 2020). The time limit of searching counterexamples is set to 5 seconds. Scaling tactics encompasses a comprehensive array of inequality lemma, including AM-GM, AMHM, Cauchy-Schwarz, Power Mean, Chebyshev, Muirhead, Jensen, Titu, Schur, Holder inequalities, 

> 1 https://www.maplesoft.com/support/help/maple/view.aspx?path=RegularChains 

> 2 https://faculty.uestc.edu.cn/huangfangjian/en/article/167349/content/2378.htm 

17 

Published as a conference paper at ICLR 2025 

as well as a selection of valuable inequalities contributed by L´aszi´o Kozma.[3] To facilitate their application in Lean 4, we have developed multiple variations of each inequality, accounting for different numbers of variables and directions. Figure 7 illustrates our scaling tactics using twovariable AM-GM inequality. 

theorem NEQ_AM_GM_left_2vars (u v k l right : R) (hk : k _≥_ 0) (h : k * (u ˆ 2 + v ˆ 2) / 2 + l _≤_ right) : k * (u * v) + l _≤_ right := by suffices (u - v) ˆ 2 _≥_ 0 by nlinarith positivity 

theorem NEQ_AM_GM_right_2vars (u v k l left : R) (hk : k _≥_ 0) (h : left _≤_ 2 * k * (u * v) + l) : left _≤_ k * (u ˆ 2 + v ˆ 2) + l := by suffices (u - v) ˆ 2 _≥_ 0 by nlinarith positivity 

Figure 7: Two examples of AM-GM inequality encoded as scaling tactics 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0018-05.png)


**----- Start of picture text -----**<br>
Prompts of Rewriting Tactics Generation (Simplification)<br>### Task<br>Your task is to use the condition  { condition } , rearrange and rewrite the expression given by the user into<br>an absolutely different form.<br>### Notice<br>1. Please reason step by step<br>2. Only four operators, add, sub, multiply, and division, can be used, and should NOT introduce new<br>variables<br>3. Put the final results within  \\ boxed {{}} , e.g.,  \\ boxed {{ x + 1/y - z }}<br>### Response<br>User:<br>{ problem }<br>Assistant:<br>Prompts of Rewriting Tactics Generation (Others)<br>### Task<br>You should rewrite the inequality given by the user according to the rule  { rule }<br>### Notice<br>1. Please reason step by step<br>2. Follow the given example, and output the result for the given inequality<br>3. Put the final results within  \\ boxed {{}} , e.g.,  \\ boxed {{ x + 1/y - z }}<br>### Example<br>User:<br>{ example p roblem }<br>Assistant:<br>{ example a nswer }<br>### Response<br>User:<br>{ problem }<br>Assistant:<br>**----- End of picture text -----**<br>


Figure 8: Prompts of simplification and other operations used for generating rewriting tactics 

For rewriting tactics, we design 16 relevant operations, i.e., simplification w/o assumptions, simplification w/ assumptions, completing the square, variable substitution, expression expansion, expression rearrangement, expression multiplication, cancellation of denominators/numerators, cancellation of powers, extraction/cancellation of common factors, separation/reduction of fractions, sum-of-squares trick, and tangent line trick. Except for the sum-of-squares trick and tangent line trick implemented based on SymPy, we use two prompt templates for these operations, shown in Figure 8. For each operation, we repeatedly query the LLM three times to cover more rewriting tac- 

> 3 https://www.lkozma.net/inequalities_cheat_sheet/ineq.pdf 

18 

Published as a conference paper at ICLR 2025 

tics. We use a low-temperature setting of GPT-4o in rewriting tactic generation and neural ranking (T=0.1, top ~~p~~ =1.0, max ~~t~~ oken=2048). 

## D ABLATION STUDY 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0019-03.png)


**----- Start of picture text -----**<br>
20 300<br>CAD MMA<br>16 240<br>12 180<br>8 120<br>4 60<br>0 0 5 10 15 20 25 30 0 0 5 10 15 20 25 30<br>Solution number Solution number<br>CPU time in seconds CPU time in seconds<br>**----- End of picture text -----**<br>


Figure 9: CPU time vs. solution count curves of CAD and MMA. The results show that symbolic solvers can verify the inequality in less than five minutes. For other problems that cannot be verified quickly, allocating more time does not improve the results. In addition, it is important to note that symbolic solvers are unable to construct human-readable proofs for the problems. 

**Scaling tactics.** Figure 4(a) has shown how the performance on the ChenNEQ dataset (Y-axis) changes as we increase the number of lemmas used by scaling tactics (X-axis). When X = 0 (without scaling tactics), only 3 out of 41 theorems can be proved, whereas LIPS can prove 39 theorems. 

**Rewriting tactics.** We observed that the successfully proved 3 theorems without scaling tactics are achieved by two rewriting tactics, sum-of-squares and tangent line trick. Hence, we further investigate the ablation of these two tactics. We newly evaluated LIPS w/o sum-of-squares and tangent lines on ChenNEQ. Among the 39 out of 41 proved inequalities, none of them requires sum-of-squares, and 3 requires the tangent line trick. 

**Symbolic filtering.** Figure 4(b) has shown how the performance on the ChenNEQ dataset (Y-axis) changes as we increase the size of the filtered set (X-axis). A smaller X means more aggressive symbolic filtering. When X is large, e.g., 32, the performance drops under 40%, highlighting the importance of symbolic filtering. We did not test values of X beyond 32, as this consistently caused GPT-4o to hit the maximum token limit (set at 4096). 

**Neural ranking.** Figure 5 presents an ablation analysis where neural ranking is replaced with random selection. The results on ChenNEQ demonstrate a significant drop in LIPS performance, from 95.1% to 41.2%, underscoring the necessity of neural ranking. Additionally, we incorporate a new ablation analysis that examines the effect of replacing neural ranking with purely symbolic filtering. The success rate decreased to 43.9%, indicating that symbolic filtering cannot determine an accurate proof path. 

**Symbolic solvers.** To explore the performance of symbolic solvers, we use the ChenNEQ dataset and plot the CPU time vs. solution count curves on Figure 9. It can be observed that (1) symbolic solvers are very fast, they can verify the inequality in less than five minutes; (2) for other problems that cannot be verified quickly, allocating more time cannot improve the results. It also means that we only need to set a small time limit for symbolic solvers in LIPS. 

Table 2: LIPS proving success rate if different time budgets 

|Method|_<_10min|_<_30min|_<_50min|_<_70min|_<_90min|
|---|---|---|---|---|---|
|LIPS|5%|15%|35%|60%|80%|



As to the results of LIPS, we list the success rate vs. CPU time in Table 2. Additionally, we plan to further enhance the efficiency of LIPS in the next version, for example, by optimizing the SymPy pattern ~~m~~ atching function. 

19 

Published as a conference paper at ICLR 2025 

## E CASE STUDY 

## A Failed Case of DSP 

**Prompt:** Draft an informal solution similar to below. The informal solution will be used to sketch a formal Lean proof. Here are some examples: Example: 

Informal: 

(* ### Problem _{_ Problem-1 _}_ ### Solution _{_ Solution-1 _}_ *) Informal: 

(* ### Problem _{_ Problem-2 _}_ ### Solution _{_ Solution-2 _}_ *) Informal: 

(* ### Problem if _a, b, c_ are positive real numbers, then 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0020-08.png)


## **Informal solution:** 

### Solution 

We will use the AM-GM inequality which states that for non-negative real numbers _x_ and _y_ , the arithmetic mean is greater than or equal to the geometric mean: _[x]_[+] 2 _[y] ≥_ ~~_[√]_~~ _xy_ . First, let’s consider the terms on the left-hand side one by one and apply AM-GM to each term along with its corresponding term on the right-hand side. 

For the term _[a] b_[and its corresponding term] ~~�~~ _ab_[2][2] +1+1[, we apply AM-GM to] _[a] b_[and] _[a] b[·] a_[2] 1+1 = _[b]_[2] _a_[+1] : _b_[2] +1 _ab_[+] _[b]_[2] _a_[+1] _≥ a[b]_[2][ + 1] = _a_[2] + 1 2 ~~�~~ _b[·] a_ ~~�~~ _b_[2] + 1 _[.]_ Repeating this process for _c[b]_[with] ~~�~~ _cb_[2][2] +1+1[and] _a[c]_[with] ~~�~~ _ac_[2][2] +1+1[, we obtain:] _cb_[+] _[c]_[2] _b_[+1] _≥ b_[2] + 1 2 ~~�~~ _c_[2] + 1 _[,]_ and _ac_[+] _[a]_[2] _c_[+1] _≥ c_[2] + 1 2 ~~�~~ _a_[2] + 1 _[.]_ 

Adding these three inequalities together, we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0020-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0020-15.png)


Thus, we have shown that the original inequality holds for any positive real numbers _a, b, c_ .*) 

## **Prompt:** 

Translate the informal solution into a sketch of the formal Lean 4 proof. Add ‘suggest ~~t~~ actics’ in the sketch whenever possible. ‘suggest ~~t~~ actics‘ will be used to call the automated theorem prover. Here are some examples: 

Example: 

Informal: 

(* ### Problem _{_ informal ~~p~~ roblem _}_ ### Solution _{_ informal ~~s~~ olution _}_ 

20 

Published as a conference paper at ICLR 2025 

Formal: _{_ formal ~~p~~ roblem _}_ *) Informal: (* ### Problem _{_ informal ~~p~~ roblem _}_ ### Solution _{_ informal ~~s~~ olution _}_ Formal: 

## **Formal solution:** 

import Mathlib.Data.Real.Basic import Mathlib.Tactic.Linarith import Mathlib.Tactic.SuggestTactics 

theorem inequality_amgm (a b c : R) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : (a / b) + (b / c) + (c / a) _≥_ (sqrt ((aˆ2 + 1) / (bˆ2 + 1))) + (sqrt ((bˆ2 + 1) / (cˆ2 + 1))) + (sqrt ((cˆ2 + 1) / (aˆ2 + 1))) := by 

_-- Apply the Arithmetic Mean-Geometric Mean Inequality (AM-GM)_ have am_gm_ab := real.am_gm (a / b) ((bˆ2 + 1) / a), have am_gm_bc := real.am_gm (b / c) ((cˆ2 + 1) / b), have am_gm_ca := real.am_gm (c / a) ((aˆ2 + 1) / c), 

_-- The AM-GM inequality gives us the following:_ 

- _-- am_gm_ab : (a / b + (bˆ2 + 1) / a) / 2 ≥ sqrt ((a / b) * ((bˆ2 + 1) / a))_ 

- _-- am_gm_bc : (b / c + (cˆ2 + 1) / b) / 2 ≥ sqrt ((b / c) * ((cˆ2 + 1) / b))_ 

- _-- am_gm_ca : (c / a + (aˆ2 + 1) / c) / 2 ≥ sqrt ((c / a) * ((aˆ2 + 1) / c))_ 

_-- Now, we sum the three inequalities:_ linarith [am_gm_ab, am_gm_bc, am_gm_ca], 

_-- Simplify the right-hand side to match the terms of the original inequality_ 

_-- ‘suggest_tactics‘ could assist in automating algebraic simplifications here:_ suggest\_tactics, 

_-- Conclude the proof using positivity of additional terms -- Positivity of terms like (bˆ2 + 1) / a and others should be straightforward_ have pos_terms : (bˆ2 + 1) / a + (cˆ2 + 1) / b + (aˆ2 + 1) / c > 0, 

{ apply add_pos, _-- breaking it down into multiple positive parts_ all_goals { apply div_pos; linarith } }, 

_-- Dropping the additional positive terms from the left-hand side to strengthen the inequality_ 

linarith [pos_terms], 

### Explanation of the Steps: 

1) Initial Setup and Imports: We start by importing necessary modules from Mathlib, such as basic real number operations and the linarith tactic for linear arithmetic. 

2) Statement of the Theorem: The theorem inequality ~~a~~ mgm is stated for three positive real numbers _a_ , _b_ , and _c_ . The assumptions ha, hb, and hc enforce the positivity conditions. 3) ...... 4) ...... 

### Conclusion 

This sketch outlines the formal structure of the proof and highlights where automation can assist. 

21 

Published as a conference paper at ICLR 2025 

## A Failed Case of OpenAI o1-preview 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0022-02.png)


**----- Start of picture text -----**<br>
Prompt: Please solve the following problem: For  a >  0 , b >  0 , c >  0 , d >  0 such that  a · b  +  b · c  +<br>c · d  +  d · a  = 1, show that:<br>a [3] b [3] c [3] d [3]<br>b  +  c  +  d [+] c  +  d  +  a [+] d  +  a  +  b [+] a  +  b  +  c [≥] [1] 3<br>Informal solution:<br>To solve this problem, we aim to show that for positive real numbers  a , b , c , d  satisfying  ab  +  bc  +<br>cd  +  da  = 1, the inequality<br>a [3] b [3] c [3] d [3]<br>b  +  c  +  d [+] c  +  d  +  a [+] d  +  a  +  b [+] a  +  b  +  c [≥] [1] 3<br>holds true.<br>Step 1: Simplify the Inequality . First, note that<br>b  +  c  +  d  = ( a  +  b  +  c  +  d )  − a  =  S − a,<br>where  S =  a  +  b  +  c  +  d . Thus, the inequality can be rewritten as:<br>a [3]<br>� S − a [≥] 3 [1] [.]<br>cyclic<br>Step 2: Analyze the Function . Consider the function<br>x [3]<br>f ( x ) = for x ∈ (0 , S ) .<br>S − x<br>**----- End of picture text -----**<br>


We will investigate the convexity of this function to apply Jensen’s inequality. Compute the second derivative: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0022-04.png)


_Step 4: Relate S to the Given Condition_ . From the given condition _ab_ + _bc_ + _cd_ + _da_ = 1, and using the Cauchy-Schwarz inequality: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0022-06.png)


But since _ab_ + _bc_ + _cd_ + _da_ = 1, we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0022-08.png)


Also, note that 

Therefore: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning_images/iclr_2025_Proving_Olympiad_Inequalities_by_Synergizing_LLMs_and_Symbolic_Reasoning.pdf-0022-11.png)


22 

Published as a conference paper at ICLR 2025 

But this contradicts the desired inequality, suggesting a miscalculation. _Step 5: Test the Equality Case_ . Consider _a_ = _b_ = _c_ = _d_ =[1] 2[.][Then:] 1 _ab_ + _bc_ + _cd_ + _da_ = 4[1] = 1 � 2 _[·]_ 2 � Compute the left-hand side: 1 3 1 _a_[3] � 2 � 8 = 4 _·_ = 4 _·_[1][2][1] cyclic� _b_ + _c_ + _d_[= 4] _[ ·]_ 32 32 8 _[·]_ 3[=] 3 This shows that equality holds when _a_ = _b_ = _c_ = _d_ =[1] 2[.] _x_[3] _Step 6: Conclude the Proof_ Since the function _f_ ( _x_ ) = _S−x_[is convex, the minimum of the sum occurs] when all variables are equal, i.e., _a_ = _b_ = _c_ = _d_ . Therefore, the inequality holds true: _a_[3] _b_[3] _c_[3] _d_[3] _b_ + _c_ + _d_[+] _c_ + _d_ + _a_[+] _d_ + _a_ + _b_[+] _a_ + _b_ + _c[≥]_[1] 3 _[.]_ 

## A Failed Case of LIPS 

**Problem (USAMO 2004):** Let _a, b, c_ be positive reals. Prove that 

( _a_ + _b_ + _c_ )[3] _≤_ ( _a_[5] _− a_[2] + 3) _∗_ ( _b_[5] _− b_[2] + 3) _∗_ ( _c_[5] _− c_[2] + 3) 

## **Formal solution:** 

theorem Problem_3dP5 {a b c : R} (ha : a > 0) (hb : b > 0) (hc : c > 0) : (a + b + c) ˆ 3 _≤_ (a ˆ 5 - a ˆ 2 + 3) * (b ˆ 5 - b ˆ 2 + 3) * (c ˆ 5 - c ˆ 2 + 3) := by scale NEQ_Cauchy_Schwarz_left_sqrt_3vars (u1 := 1) (u2 := 1) (u3 := 1) (v1 := a) (v2 := b) (v3 := c) _. . ._ scale NEQ_weighted_AM_GM_left_3vars (u := a + b + c) (v := 3) (w := aˆ2 + bˆ2 + cˆ2) (i1 := 1) (i2 := 1/2) (i3 := 1/2) _. . ._ llm_frac_reduce _. . ._ llm_cancel_denom _. . ._ scale NEQ_Cauchy_Schwarz_sqrt_left_2vars (u1 := 1) (u2 := 1) (v1 := a) (v2 := a) _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ llm_frac_apart _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ llm_frac_apart _. . ._ llm_simplify _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ llm_simplify _. . ._ llm_factor _. . ._ = 4*sqrt (2)*sqrt (bˆ2 + cˆ2)*(3 + aˆ2 + bˆ2 + cˆ2 + 2*a + 2*b + 2*c)ˆ2 + (1 + a)ˆ2*(3 + aˆ2 + bˆ2 + cˆ2 + 2*a + 2*b + 2*c)ˆ2 - (64*(3 + aˆ5 - aˆ2)*(3 + bˆ5 - bˆ2)*(3 + cˆ5 - cˆ2)) FAIL(TIMEOUT)! 

23 

Published as a conference paper at ICLR 2025 

## A Successful Case of LIPS 

**Problem (Evan Chen):** If _a, b, c_ are positive reals and _a_[2] + _b_[2] + _c_[2] = 1, then 1 1 1 1 1 1 _a_[2] + 2[+] _b_[2] + 2[+] _c_[2] + 2 _[≤]_ 6 _ab_ + _c_[2][+] 6 _bc_ + _a_[2][+] 6 _ca_ + _b_[2] _[.]_ **Formal solution:** theorem Problem_2d4e1 {a b c : R} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a ˆ 2 + b ˆ 2 + c ˆ 2 = 1) : 1 / (a ˆ 2 + 2) + 1 / (b ˆ 2 + 2) + 1 / (c ˆ 2 + 2) _≤_ 1 / (6 * a * b + c ˆ 2) + 1 / (6 * b * c + a ˆ 2) + 1 / (6 * c * a + b ˆ 2) := by scale NEQ_AM_GM_div_square_right_2vars (u := b) (v := c) _. . ._ scale NEQ_AM_GM_div_square_right_2vars (u := b) (v := a) _. . ._ scale NEQ_AM_GM_div_square_right_2vars (u := a) (v := c) _. . ._ llm_simplify _. . ._ llm_simplify _. . ._ llm_rearrange _. . ._ sym_tangent_line _. . ._ llm_simplify _. . ._ = 27/49 - (27*aˆ2/49) - (27*bˆ2/49) - (27*cˆ2/49) try close SUCCESS! 

## F SOME MORE CASES OF LIPS 

## A Different Proof Generated by LIPS (1) 

**Problem:** Let _a, b, c_ be three positive reals. Prove that if _abc_ = 1, then 

_a_[2] + _b_[2] + _c_[2] _≥ a_ + _b_ + _c_ 

Source: https://web.evanchen.cc/handouts/Ineq/en.pdf 

## **Formal solution:** 

theorem Example_1d7 {a b c : R} (h : a * b * c = 1) : a + b + c _≤_ a ˆ 2 + b ˆ 2 + c ˆ 2 := by scale NEQ_AM_GM_left_square_2vars (u := 1) (v := a) _. . ._ scale NEQ_AM_GM_left_square_2vars (u := 1) (v := c) _. . ._ scale NEQ_AM_GM_left_square_2vars (u := 1) (v := b) _. . ._ llm_rearrange _. . ._ llm_simplify _. . ._ = 3/2 - aˆ2/2 - bˆ2/2 - cˆ2/2 llm_rearrange (left := 3/2) (right := aˆ2/2 + bˆ2/2 + cˆ2/2) scale NEQ_AM_GM_right_normal_3vars (u := aˆ2/2) (v := bˆ2/2) _. . ._ llm_simplify _. . ._ = (a*b*c)ˆ2 / 8 llm_simplify _. . ._ = 1 / 8 try close 

Note: At first glance, “AM-GM alone is hopeless here, because whenever we apply AMGM, the left and right hand sides of the inequality all have the same degree”. However, LIPS find a proof by using 1 _× a ≤[a]_[2] 2[+1] , 1 _× b ≤[b]_[2] 2[+1][,][ 1] _[ ×][ c][ ≤][c]_[2] 2[+1] . 

24 

Published as a conference paper at ICLR 2025 

## A Different Proof Generated by LIPS (2) 

**Problem (JMO 2012):** Let _a, b, c_ be positive reals. Prove that 

_a_[3] + 5 _b_[3] _[b]_[3][ + 5] _[c]_[3] _[c]_[3][ + 5] _[a]_[3] _≥_[3] 3 _a_ + _b_[+] 3 _b_ + _c_[+] 3 _c_ + _a_ 2[(] _[a]_[2][ +] _[ b]_[2][ +] _[ c]_[2][)] _[.]_ Source: https://artofproblemsolving.com/community/c5h476722p2669114 

## **Formal solution:** 

theorem Problem_3de6’’ {a b c : R} (ha : a > 0) (hb : b > 0) (hc : c > 0) : (3 / 2) * (a ˆ 2 + b ˆ 2 + c ˆ 2) _≤_ (a ˆ 3 + 5 * b ˆ 3) / (3 * a + b) + (b ˆ 3 + 5 * c ˆ 3) / (3 * b + c) + (c ˆ 3 + 5 * a ˆ 3) / (3 * c + a) := by llm_cancel_factor _. . ._ scale NEQ_Muirhead2_onestep_right_3vars (u := a) (v := b) _. . ._ scale NEQ_Muirhead1_left_3vars (u := c) (v := a) (w := b) _. . ._ scale NEQ_Muirhead1_left_3vars (u := a) (v := b) (w := c) _. . ._ scale NEQ_Muirhead2_onestep_left_3vars (u := a) (v := b) _. . ._ llm_cancel_factor _. . ._ 

scale NEQ_Muirhead2_onestep_right_3vars (u := a) (v := b) _. . ._ llm_simplify _. . ._ 

scale NEQ_Muirhead2_left_3vars (u := c) (v := b) (w := a) _. . ._ 

Note: A completely mechanical proof, achieved by iteratively applying Muirhead inequality six times after canceling the denominator 

## LIPS Checks Existing Answers (1) 

**Problem (Japan 1997):** Prove that ( _b_ + _c − a_ )[2][(] _[c]_[ +] _[ a][ −][b]_[)][2][(] _[a]_[ +] _[ b][ −][c]_[)][2] ( _b_ + _c_ )[2] + _a_[2][+] ( _c_ + _a_ )[2] + _b_[2][+] ( _a_ + _b_ )[2] + _c_[2] _[≥]_ 5[3] Source: https://artofproblemsolving.com/community/c6h146p537 **Informal solution No.1 (provided by AoPS community):** Solution: Let : _P_ = (( _bb_ ++ _cc_ ) _−_[2] + _a_ ) _a_[2][2][+] ([(] _c[c]_ +[+] _a[a]_ ) _[−]_[2] + _[b]_[)] _b_[2][2][+] ([(] _a[a]_ +[+] _b[b]_ ) _[−]_[2] + _[c]_[)] _c_[2][2] By Cauchy Swarchz inequality, we have ( _a_[2] + _b_[2] + _c_[2] + ( _a_ + _b_ )[2] + ( _b_ + _c_ )[2] + ( _c_ + _a_ )[2] )( _P_ ) _≥_ ( _a_ + _b_ + _c_ )[2] (3( _a_[2] + _b_[2] + _c_[2] ) + 2( _ab_ + _bc_ + _ca_ ))( _P_ ) _≥_ ( _a_ + _b_ + _c_ )[2] (3( _a_ + _b_ + _c_ )[2] _−_ 4( _ab_ + _bc_ + _ca_ ))( _P_ ) _≥_ ( _a_ + _b_ + _c_ )[2] Hence, ( _a_ + _b_ + _c_ )[2] _P ≥_ 3( _a_ + _b_ + _c_ )[2] _−_ 4( _ab_ + _bc_ + _ca_ )[2] but we know that ( _a_ + _b_ + _c_ )[2] _≥_ 3( _ab_ + _bc_ + _ca_ ), 3( _ab_ + _bc_ + _ca_ ) _P ≥_ 9( _ab_ + _bc_ + _ca_ ) _−_ 4( _ab_ + _bc_ + _ca_ )[=][3] 5 ... 

Note: the proof is correct if all variables are assumed to be positive. However, without this assumption, applying Cauchy-Schwarz is incorrect, and a counterexample is successfully found by our symbolic solver [ _c_ := 8 _._ 0 _, a_ := 1 _/_ 8 _, b_ := ( _−_ 1 _._ 0)]. This problem is proved by LIPS using llm ~~s~~ implification and tangent ~~l~~ ine trick tactics. 

25 

Published as a conference paper at ICLR 2025 

LIPS Checks Existing Answers (2) **Problem (Japan 1997):** Prove that ( _b_ + _c − a_ )[2][(] _[c]_[ +] _[ a][ −][b]_[)][2][(] _[a]_[ +] _[ b][ −][c]_[)][2] ( _b_ + _c_ )[2] + _a_[2][+] ( _c_ + _a_ )[2] + _b_[2][+] ( _a_ + _b_ )[2] + _c_[2] _[≥]_ 5[3] Source: https://artofproblemsolving.com/community/c6h146p537 **Informal solution No.2 (provided by AoPS community):** .... don’t use so much of mathematics on that simple problem.. :P :D here’s the solution using TITU’s lemma.. The lemmaif _x, y, a, b_ are reals and _x, y >_ 0 then _[a] x_[2][+] _[b] y_[2] _[≥]_ ([(] _x[a]_[+] + _[b] y_[2] ) simply applying the lemma on the lhs twice we get ( _a_ + _b_ + _c_ )[2] _LHS ≥_ 3( _a_[2] + _b_[2] + _c_[2] )+2( _ab_ + _bc_ + _ca_ ) by AM-GM ( _a_ + _b_ + _c_ )[2] _≥_ 3( _ab_ + _bc_ + _ca_ ) and 3( _a_[2] + _b_[2] + _c_[2] ) + 2( _ab_ + _bc_ + _ca_ ) _≥_ 5( _ab_ + _bc_ + _ca_ ) ... Note: it appears that the issue arises from the inequality being in the wrong direction. Actually, the problem is unprovable when applying the Titu’s lemma (a counterexample found by our symbolic solver: [ _b_ = 1 _._ 0 _, a_ = 1 _._ 0 _, c_ = 0 _._ 5]) 

26 

