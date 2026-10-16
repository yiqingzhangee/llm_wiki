## **Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning** 

**Xiaoqian Wu Yong-Lu Li** _[∗]_ **Jianhua Sun Cewu Lu** _[∗]_ Shanghai Jiao Tong University `{enlighten, yonglu_li, gothic, lucewu}@sjtu.edu.cn` 

## **Abstract** 

Human reasoning can be understood as a cooperation between the intuitive, associative “System-1” and the deliberative, logical “System-2”. For existing System1-like methods in visual activity understanding, it is crucial to integrate System-2 processing to improve explainability, generalization, and data efficiency. One possible path of activity reasoning is building a symbolic system composed of symbols and rules, where one rule connects multiple symbols, implying human knowledge and reasoning abilities. Previous methods have made progress, but are defective with limited symbols from handcraft and limited rules from visual-based annotations, failing to cover the complex patterns of activities and lacking compositional generalization. To overcome the defects, we propose a new symbolic system with two ideal important properties: broad-coverage symbols and rational rules. Collecting massive human knowledge via manual annotations is expensive to instantiate this symbolic system. Instead, we leverage the recent advancement of LLMs (Large Language Models) as an approximation of the two ideal properties, _i_ . _e_ ., Symbols from Large Language Models ( **Symbol-LLM** ). Then, given an image, visual contents from the images are extracted and checked as symbols and activity semantics are reasoned out based on rules via fuzzy logic calculation. Our method shows superiority in extensive activity understanding tasks. Code and data are available at `https://mvig-rhos.com/symbol_llm` . 

## **1 Introduction** 

Human reasoning can be understood as a cooperation between two cognitive systems: the intuitive, associative “System-1” and the deliberative, logical “System-2” [36]. An example of System-2 reasoning is illustrated in Fig. 1a. A feather and an iron ball are dropped at the same height in a vacuum, which one will land first? Human intelligence derives the answer from the Universal Gravitation Law, _i_ . _e_ ., reasoning in a physical symbolic system to avoid intuitive errors. As a critical component in building AI systems, visual activity understanding urgently needs System-2 reasoning. Previous works [37, 21, 41, 24] typically establish the mapping between visual inputs and natural language, where knowledge is embedded into some non-symbolic form such as the weights of language models [35]. Though effective, these System-1-like methods depend on large-scale visual data, thus suffering from diminishing marginal utility. Also, it fails in explainability and generalization due to black-box computing. Thus, it is important to integrate System-2 reasoning. 

With one glance at an image, we can effortlessly ground visual inputs to symbols/concepts and apply commonsense reasoning to imagine the world beyond the pixels, a process similar to the gravity case. Fig. 1b shows a possible path for System-2 visual reasoning for human activity: a symbolic system composed of symbols and rules. One rule “ `Hip seated in a boat` _∧_ `Hands control an engine` _∧_ `Hands hold onto the side of a boat` _|_ = `Human ride a boat` ” connects symbols 

> _∗_ Corresponding authors. 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
a)<br>feather iron ball System-2 Reasoning<br>Abstract 𝐹= 𝐺 𝑚𝑟0 [2] 𝑚 𝑎= 𝐺 𝑚0 𝑎1 = 𝑎2 Answer<br>Which will  𝐹= 𝑚𝑎 𝑟 [2] 𝑣1 = 𝑣2<br>land first?<br>b) Activity Reasoning<br>𝒎𝟏 :  Hip seated in a boat<br>Abstract<br>𝒎𝟐 :  Hands control  𝒄 :  Human  Activity<br>an engine ride a boat Semantic<br>𝒎𝟑 :  Hands hold onto<br>…… … the side of a boatRule: 𝒎𝟏 ∧ 𝒎𝟐 ∧ 𝒎𝟑 ⊨ 𝒄<br>**----- End of picture text -----**<br>


Figure 1: System-2 reasoning in **a)** solving physical questions and **b)** understanding human activities. They both involve a symbolic system implying human commonsense. 

( _e_ . _g_ ., “ `Hands control an engine` ” “ `Human ride a boat` ”) and implies human commonsense. Then, given an image, symbols are first extracted from visual inputs and then activity semantics are deduced based on rules. In this paper, we focus on this line of System-2 reasoning and zoom in from general visual understanding to human activities in various complex contexts, which is a challenging problem in visual understanding. 

Current activity reasoning methods are not well equipped with System-2 reasoning. For instance, HAKE [28, 26] adopts a similar path as Fig. 1b from the perspective of human body part states and makes progress toward activity reasoning, but is defective with _limited symbols from handcraft_ (93 primitives) and _limited rules_ from visual-based annotations (100 K+ images). For example, Fig. 1b shows several rules related to the activity “ `Human ride a boat` ”, _e_ . _g_ ., 

(R1, green) `Hip seated in a boat` _∧_ `Hand hold onto the side of the boat` _∧_ `Hand operate the steering wheel` _|_ = `Human ride a boat` 

(R2, orange) `A rope is attached to a boat` _∧_ `Hand hold the rope` _∧_ `Hand adjust a sail` _∧_ `Feet hold onto the rope` _∧_ `Leg stretched out` _|_ = `Human ride a boat` 

However, the rules summarized by [26] are “(R3) `Hip sit in` _∧_ `Hand hold` _|_ = `Human ride a boat` ” “(R4) `Hand hold` _∧_ `Feet close with` _|_ = `Human ride a boat` ” , which fail to cover the complex patterns of human activities. Thus, it lacks compositional generalization, _i_ . _e_ ., a generalized understanding of visual concepts, their relationships, and their novel combinations. 

In view of this problem, we propose a novel symbolic system that implies abundant human knowledge and organizes diverse reasonable rules as in Fig. 1b. To achieve this, the symbols should be with **broad semantic coverage** to express various activity contexts. Moreover, the rules should be **rational and unambiguous** . R4 is a failure case, where ‘ `Hand hold` ” and “ `Feet close with` ” do not necessarily lead to “ `Human ride a boat` ”, and “ `Human tie a boat` ”“ `Human exit a boat` ” are other possible conclusions. Ambiguous semantics should be corrected in rules to avoid confusion. Collecting massive human knowledge via manual annotation is expensive to instantiate this symbolic system. Our main insight is Symbols from Large Language Models ( **Symbol-LLM** ), _i_ . _e_ ., leveraging the recent advancement of Large Language Models (LLMs) to approximate the above two important properties. This is achieved by the _symbol-rule loop_ and _entailment check_ strategy. Then, the symbolic system can be used to reason with visual inputs, where visual contents from the images are _extracted and checked_ as symbols and activity semantics are reasoned out via _fuzzy logic calculation_ . 

In conclusion, our main contributions are: **1)** We point out that current System-2 reasoning [26] suffers from a defective symbolic system with hand-crafted symbols and limited, ambiguous rules. **2)** To overcome the defects, we propose a novel symbolic system with broad-coverage symbols and rational rules. Further, we propose Symbol-LLM to instantiate it and show how it helps visual reasoning. **3)** In extensive activity understanding tasks, our method shows superiority in explainability, generalization, and data efficiency. 

## **2 Related Work** 

**Activity Understanding** [29, 12] is still a difficult task compared with object detection [39] or pose estimation [11]. This is caused by its complex visual patterns and long-tail data distribution [43]. 

2 

There are mainly image-based [5, 19], video-based [18, 4] and skeleton-based [42, 8] methods. In this paper, we focus on image-based methods. 

**Vision-Language Models** (VLMs) have been intensively studied recently. Trained with web-scale image-text pairs, they can make zero-shot predictions on various recognition tasks [48] including human activities. Though effective, previous System-1-like methods [37, 21, 41, 24] need to be improved in explainability, generalization, and data efficiency. Thus, integrating System-2 reasoning, which is our main focus, is important. 

**Probabilistic Programming** [38, 13] unifies probabilistic modeling and traditional programming. We follow the principles of the standard first-order logic (FoL) in probabilistic programming, _e_ . _g_ ., logical connectives _∧_ , _∨_ . The standard FoL typically solves simplified relational tasks, where symbols are definitely True/False. However, when adapting FoL in vision reasoning tasks, the symbols are not known to be True/False. Instead, the models are required to predict the symbol probabilities (usually via a neural network). Thus, a hybrid “neuro-symbolic” method is adopted. 

**Neuro-Symbolic Reasoning** provides a possible way of combining System-1 and System-2 learning, where knowledge is represented in symbolic form and learning and reasoning are computed by a neural network. [15]. Recently, researchers utilize neural-symbolic reasoning in tasks including visual question answering [2, 20, 45, 35, 40], decision-making [7], motion programming [23], scene understanding [31]. Focused on activity reasoning, HAKE [28, 26] propose an intermediate space spanned by human body part states as primitives. To overcome its inherent defect in compositional generalization, we propose a novel symbolic system with broad semantics and rational rules. 

## **3 Method** 

## **3.1 Formulating the Symbolic System** 

## **3.1.1 Structure** 

We first analyze the structure of the symbolic system. As aforementioned, the activity symbolic system is composed of rules that connect symbols, which implies human knowledge and reasoning abilities. Formally, one rule involves several symbols, where one of them is the _conclusion_ and the others are _premises_ . Each rule takes the form _r_ : _m_ 1 _∧ m_ 2 _∧ . . . ∧ mnr |_ = _c_ , where _M_ = _{mi}[n] i_ =1 _[r]_ is the premise set and _c_ is the conclusion. Each conclusion corresponds to more than one rule with varied combinations of premise symbols because one activity typically has different visual patterns. Multiple rules for one conclusion are logically connected with _∨_ , _i_ . _e_ ., one conclusion holds if at least one rule is satisfied. A hyper-graph is a proper data structure to handle the complexity of the connection. Therefore, we define the structure of the activity symbolic system as a _B_ -Graph [14]. 

**Definition 1. (Directed Hypergraph [3])** _A hypergraph is a generalization of a graph in which an edge can join any number of vertices. A directed hypergraph is a pair_ ( _X , E_ ) _, where X is a set of vertices, and E is a set of pairs of subsets of X . Each of these pairs_ ( _D, C_ ) _∈E is a hyperedge; the vertex subset D is its domain, and C is its codomain._ 

**Definition 2. (** _B_ **-Graph [14])** _A B-graph is a type of directed hypergraph with only B-arcs. A B-arch is a type of a hyperedge that is directed to a single head vertex, and away from all its other vertices._ 

**Definition 3. (Activity Symbolic System)** _The activity symbolic system_ ( _S, R_ ) _is a B-Graph, where S is a set of vertices/symbols, and R is a set of pairs of subsets of S. Each of these pairs r_ = ( _M, C_ ) _∈R is a hyperedge/rule; the vertex subset M_ = _{mi}[n] i_ =1 _[r][is its domain/premises,] and C_ = _{c} is its codomain/conclusion. Equivalently, a rule takes the form r_ :[�] _m∈M[m][ |]_[=] _[ c][.] Since |C|_ = 1 _, r is a B-arch._ 

**Definition 4. (Decomposition of Activity Symbolic System)** _One conclusion c[∗] corresponds to one symbolic sub-system_ ( _Sc[∗] , Rc[∗]_ ) _, where Sc[∗] ⊆S, Rc[∗] ⊆R. ∀r_ = ( _M, C_ ) _∈Rc[∗] , C_ = _{c[∗] }. ∀s ∈Sc∗ \{c[∗] }, ∃ r_ = ( _M, C_ ) _∈Rc∗ , s ∈ M ._ 

Def. 3,4 is depicted in Fig. 2. Def. 3 is based on Def. 1,2 and analysis above. In applications, we typically judge a specific conclusion with other symbols/rules removed. It is achieved by decomposing the activity symbolic system (graph) into sub-systems (sub-graphs) in Def. 4. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
a) b) Semantic Coverage Approximation Symbol-Rule Loop<br>Rules:<br>Image  Language<br>∧ ⊨ Semantics Semantics head Queue tail<br>∧ ∧ ⊨<br>… ∧ ∧ ⊨ Symbol  Symbol  pop push<br>… … Semantics … … Semantics<br>… ∧ ∧ ⊨<br>… Rule Extension<br>Decomposition<br>c) Logical Entailment Approximation Entailment Check<br>Entailment   A rule 𝑟<br>√√√ score 𝑒= 𝜏(𝑟)<br>… … √√…× 𝑒ℎ In a picture… LLM<br>Estimate the<br>√√√√ probability …<br>… … …√√√√… 0 Entailment  score 𝑒<br>**----- End of picture text -----**<br>


Figure 2: Our activity symbolic system. **a)** Structure and decomposition of the symbolic system (Sec. 3.1.1). **b)** Semantic coverage (Sec. 3.1.2). It can be approximated based on LLMs’ knowledge and achieved via the symbol-rule loop (Sec. 3.2.1). **c)** Logical entailment (Sec. 3.1.2). It can be approximated based on an entailment scoring function and achieved by entailment check (Sec. 3.2.2). 

## **3.1.2 Two Ideal Properties** 

Existing methods [28, 26] can also be formulated under this structure. However, they fail to cover the complex patterns of activities and lack compositional generalization. To overcome the defects, we propose two important ideal properties: 1) symbols should be with broad _semantic coverage_ to express different conditions in the activity image database; 2) rules should satisfy _logical entailment_ [9] to add rationality and avoid ambiguity, _i_ . _e_ ., the premises set should lead to the conclusion without exception. They are formulated in Def. 5,6 and illustrated in Fig. 2b,c. 

**Definition 5. (Semantic Coverage of Activity Symbolic System)** _Given a very large-scale activity images database D_ = _{_ ( _I, AI , SI_ ) _} (I: image, AI : ground-truth activities happening in I, SI : ground-truth symbols happening in I), then ∀_ ( _I, AI , SI_ ) _∈D, ∀s ∈SI , s ∈S._ **Definition 6. (Logical Entailment of Activity Symbolic System)** _∀r_ = ( _M, C_ ) _∈R, M_ = _{mi}[n] i_ =1 _[r][, we have:] 1. ∀_ ( _I, AI , SI_ ) _∈D, if M ⊂SI , then c ∈ AI ;_ 

_2. ∀_ 1 _≤ i ≤ nr, ∃_ ( _I, AI , SI_ ) _∈D, M \{mi} ⊂SI , but c ∈/ AI ._ 

## **3.2 Instantiating the Symbolic System** 

To instantiate the proposed symbolic system, it is expensive to collect massive human knowledge via manual annotation as in HAKE [28, 26]. Instead, we use LLMs which have profoundly reshaped the acquisition of human knowledge and shown impressive language reasoning capabilities, _i_ . _e_ ., Symbol-LLM framework. In Sec. 3.2.1 and Sec. 3.2.2, we discuss the approximation of the two ideal properties with the help of LLMs and provide practical solutions, then a summarized pipeline is presented in Sec. 3.2.3. 

## **3.2.1 Solving Semantic Coverage** 

**Approximation.** In Def. 5, the target domain of semantic coverage is a very large-scale activity image database, which is too costly to implement. For instance, HAKE [28] collect symbols from 100 K+ image via manual annotations, which is a heavy workload but still very limited. Here, the target domain can be replaced with semantic coverage of language models since natural language is another carrier of human commonsense knowledge. LLMs profit from large-scale text pre-training and imply abundant text-based human knowledge, thus can handle complex visual activity patterns without extra visual data. The approximation of Def. 5 is: 

**Definition 7. (Approximation of Semantic Coverage of Activity Symbolic System)** _Given an LLM L and an activity set A, ∀ A ∈A, L implies a symbol set SA as premises of A, then ∀s ∈SA, s ∈S._ 

4 

**Symbol-Rule Loop.** In the implementation, generating rules connecting the symbols is non-trivial and challenging. Intuitively, we can ask LLMs “What is the rule of the activity”. However, the instruction is ambiguous for LLMs to generate satisfying answers. More information should be provided to generate rules. Also, it is costly to generate all symbols and exhaustively query their connections. Thus, we propose to adopt _symbol-rule loops_ to generate them alternately. Given a known symbol, we can add another symbol and extend it into a rule via prompting: 

(Rule Extension) 

```
Inapicture,IF[<knownsymbols>]AND[condition]THEN[<activity>].
[condition]isoneconcisephrase.Theformatis"<Theperson’s
hands/arms/hip/legs/feet><verb><object>".Whatis[condition]?
```

Then, the answer for “ `[condition]` ” is the extended symbol. Also, a candidate rule “ `<known symbols>` _∧_ `<extended symbol>` _|_ = `<activity>` ” is generated. Thus, we get a new extended symbol from this known symbol, and they are connected with rules. The new symbol can be repeatedly used as a known symbol to generate new rules. As is shown in Fig. 2b, a known symbol (the brown circle) is first “popped out” from a queue, and new symbols (the green circle) and rules (the pink edge) are generated by extending the known symbol. Then, the new symbol is “pushed in” to start the next loop. Thus, LLMs’ implicit knowledge is fully exploited to satisfy Def. 7. To get the initial symbols, we ask an LLM to describe hand-related states related to the activity. The prompts mentioned above are determined via a trial-and-error process by human experts on randomly sampled activities (detailed in supplementary). 

## **3.2.2 Solving Logical Entailment** 

**Approximation.** In Def. 6, it is costly to verify the logical entailment upon a very large-scale activity image database. Instead, we develop an entailment scoring function from an LLM based on its knowledge and language reasoning capability. As illustrated in Fig. 2c, as the rule extends from _m_ 1 _|_ = _c_ , _m_ 1 _∧ m_ 2 _|_ = _c_ to _m_ 1 _∧ m_ 2 _∧ m_ 3 _|_ = _c_ , new symbols are successively added, and the measured entailment score increases. With the scoring function, the approximation of Def. 6 is: 

**Definition 8. (Approximation of Logical Entailment of Activity Symbolic System)** _Given a function τ_ ( _·_ ) _to measure the entailment score of a rule, and an entailment threshold eh, then ∀r_ = ( _M, C_ ) _∈R, M_ = _{mi}[n] i_ =1 _[r][, we have:] 1. τ_ ([�] _m∈M[m][ |]_[=] _[ c]_[)] _[ ≥][e][h][;] 2. ∀_ 1 _≤ i ≤ nr, τ_ ([�] _m∈M \{mi}[m][ |]_[=] _[ c]_[)] _[ < e][h][.]_ 

**Entailment Check.** As is shown in Fig. 2c, to implement the scoring function based on an LLM, we rewrite the rule _r_ as a sentence and design the prompt as: 

## (Entailment Check) 

`In a picture, <symbol` 1 `>, <symbol` 2 `>` _. . ._ `<symbol` _nr_ `>. Estimate the probability that he is <activity> at the same time. Choose from: (a) 0.1, (b) 0.5, (c) 0.7, (d) 0.9, (e) 0.95, (f) unknown.` 

Then, the output answers can be used as the entailment score _e_ = _τ_ ( _r_ ), whose credibility depends on the knowledge from the LLM. For one rule, the answering text is sampled five times and their average scores are taken as the final result to add stability. In practice, we set _eh_ = 0 _._ 9. 

## **3.2.3 Summarized Pipeline** 

The pipeline of the symbolic system can be summarized as: given a target activity set _A_ , or equivalently, a conclusion set _C_ , we generate a sub-system ( _Sc, Rc_ ) for each _c ∈C_ (Def. 4). The symbolic system is merged as ( _S, R_ ) = ([�] _c∈C[S][c][,]_[ �] _c∈C[R][c]_[)][. Alg. 2 shows how to get the sub-system][ (] _[S][c][,][ R][c]_[)] for a given conclusion _c_ . Details are explained in Alg. 2 comment and the supplementary. 

## **3.3 Reasoning with Visual Inputs** 

With the activity symbol system constructed, we can utilize it to reason with visual inputs. As is shown in Fig. 3, visual information is extracted and checked based on the defined symbols and 

5 

**Algorithm 1** Instantiating the Symbolic System 

**Input:** conclusion _c_ , entailment threshold _eh_ **Output:** Symbolic Sub-System ( _Sc, Rc_ ) 1: _Sc_[0] _[←]_[Symbol_Initialization][ (] _[c]_[)] 2: _Sc[cand] .push_ ( _Sc_[0][)] _▷_ A queue _Sc[cand]_ stores symbols 3: _Sc ←{c}, Rc ←{}_ 4: **while** not _Sc[cand] .is_  empty_ () **do** 5: _mkno ←Sc[cand] .pop_ () _▷_ A known symbol _mkno_ is taken for rule extension 6: _M_ = _{mkno} ▷_ The premises set _M_ is set for the current rule 7: **while** Entailment_Check( _M_ ) _< eh_ **do** 8: _mnew ←_ Rule_Extension( _M, c_ ) 9: _M_ = _M ∪{mnew} ▷_ A new symbol _mnew_ is added 10: **end while** 11: _Sc ←Sc ∪ M ▷_ The symbol set _Sc_ is updated 12: _Rc ←Rc ∪{r_ :[�] _m∈M[m][ |]_[=] _[ c][}] ▷_ The rule set _Rc_ is updated 13: _Sc[cand] .push_ ( _M \Sc_ ) _▷M_ is added to _Sc[cand]_ with redundant symbols removed 14: **end while** 15: **return** ( _Sc, Rc_ ) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0006-02.png)


**----- Start of picture text -----**<br>
𝑚3(1), 𝑚2(2) 𝑚3(2) Symbol Extractor Symbol Checker Fuzzy-Logic Calculation<br>𝑚2(1), 𝑚1(2) 𝑚1(3) 𝑝= 0.9 𝑝= 0.5 𝑝= 0.9 𝑝= 0.7 ∧ ⋅,⋅⇒min(⋅,⋅)<br>Human ride a boat? 𝑚1(1) … 𝑚2(3) 𝑝= 0.8 𝑝= 0.8 ( ∧ ∨ ∧⋅,⋅⇒max(⋅,⋅)) ⋁ ( ∧ ∧ )<br>𝒎(𝟏)𝟏 :  Hands operate the steering wheel 𝑝= 0.7 𝑝= 0.3 𝑝= 0.5 𝑝= 0.3 ⋁ ( …… ) ……<br>𝒎(𝟏)𝟐 ,  𝒎𝟏(𝟐) :  Hip seated in a boat …… … 𝑝= 0.1 … 𝑝= 0.1 Activity Semantic<br>**----- End of picture text -----**<br>


Figure 3: Visual reasoning with the proposed activity symbol system. 

organized as a probability distribution on the hyper-graph. Then, the activity semantics can be reasoned out based on the rules and the probability of the symbols. We detail the key steps as follows. 

**Decomposition.** Given the input image and the target activity/conclusion _c_ , we first decompose the symbol system to exclude unrelated symbols and rules based on Def. 4. Thus, we get the rule set _Rc_ = _{r_[(] _[j]_[)] _}[N] j_ =1 _[c]_[, where a rule takes the form] _[ r]_[(] _[j]_[)][:] _[ m]_[(] 1 _[j]_[)] _∧ m_[(] 2 _[j]_[)] _∧ . . . m_[(] _n[j] r_[)] ( _j_ ) _[|]_[=] _[ c]_[.] 

**Extracting Visual Symbols.** Given an image, to extract visual content and express it in a symbolic form, we measure the probability of each defined symbol (Fig. 3). In [26], the visual extractor is a customized model trained with annotations, which is costly, especially for our symbolic system with broad-range symbols. Instead, we utilize a simplified visual extractor based on existing System-1-like VLMs. To determine the probability of a symbol, we convert it into a text question and query the answer from a VLM. For example, given a premise symbol _m_[(] _i[j]_[)] “ `Hip seated in a boat` ”, it is converted into a sentence “The person’s hip is seated in a boat. Yes/No?”. Used as a scoring function [1], the language model outputs the probability of answering “Yes”, “No” as _p_[(] _y,i[j]_[)] _[, p]_[(] _n,i[j]_[)][.][Then] the probability _p_[(] _i[j]_[)] of the symbol can be obtained via normalizing the Yes/No answer as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0006-07.png)


**Checking Visual Symbols.** To ensure the reliability of the symbols’ probabilities, we developed a checker module to filter out visual symbols with uncertain predictions. Each symbol is paraphrased into several variants, and the predictions of variants should be similar due to semantic consistency. Thus, the standard deviation of the predictions can be used to filter symbols with uncertainty. 

**Reasoning.** With the symbol extractor and checker, the probability _p_[(] _i[j]_[)] for each premise symbol _mi_[(] _[j]_[)] in the rule set _Rc_ is measured. Then, the probability _pc_ of the conclusion is reasoned based on premise symbols’ probabilities and the rules. Premise symbols within a rule are connected with _∧_ , while rules for a conclusion are connected with _∨_ . We utilize the fuzzy-logic calculation [46], where 

6 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0007-00.png)


**----- Start of picture text -----**<br>
Round1 Round2 Round3 17 15 Round4, 5<br>1 1 6 7 8 11 1 6 7 8 11 1 6 7 8<br>2 2 2 12 2 16 12<br>3 3 9 3 9 13 3 9 13<br>4 4 10 4 10 4 10<br>5 5 5 5 14<br>Round2 Rules: Round3 Rules: Round4 Rules: Round5 Rules:<br>1 → 6 → 7 → 8 6 → 1 → 7 → 8 : omitted 11 → 9 → 14;   14: early stop<br>2 → 1; 3 → 1 → 9 7 → 1 → 11 12 → 1 → 15; 15 → 1 → 17<br>4 → 10 → 6;  5 → 1 8 → 12; 9 → 13; 10: early stop 13 → 9 → 16;<br>**----- End of picture text -----**<br>


Figure 4: Detailed process for instantiating the symbolic sub-system of the activity “Human board an airplane”. As is shown in the left bottom box, there is some difference in graph presentation between this figure and Fig. 1-3: here the edge connects premises instead of connecting premise and conclusion. The numbers 1-17 correspond to the 17 symbols listed in Tab. 1. The “ _→_ ” in “1 _→_ 6 _→_ 7 _→_ 8” means that 6, 7, 8 is obtained via 1’s rule extension one after another. 

|6_→_7_→_8”|means that 6, 7, 8 is obtained via 1’s rule extension one after another.|means that 6, 7, 8 is obtained via 1’s rule extension one after another.|
|---|---|---|
||**Symbols**||
|**Round1**|(1)hold a boarding pass|(2) place luggage in overhead compartment|
||(3) adjust seatbelt|(4) wave goodbye to loved ones|
||(5) grip a luggage handle||
|**Round2**|(6)walk towards the boarding gate|(7)luggage visible beside him|
||(8) boarding pass is scanned by airport staff|(9) stand on the jet bridge|
||(10) luggage is loaded onto the plane||
|**Round3**|(11)reach for the airplane door handle|(12)stand in line with carry-on luggage|
||(13) hold the carry-on luggage||
|**Round4**|(14)open the airplane door|(15)move forward in the line|
||(16) move towards the airplane door||
|**Round5**|(17)airline staff checkingthe boarding pass||



Table 1: Generated symbols & rules for “human board an airplane”. Round _i_ : _i_ -th symbol-rule loop. 

the operator _x ∧ y, x ∨ y_ are replaced with _min_ ( _x, y_ ) _, max_ ( _x, y_ ) respectively. Thus, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0007-05.png)


The fuzzy-logic calculation is more explainable and lightweight than the logical modules in [26], whose parameters need extra training samples. Benefiting from the sound definition of our symbolic system, it has the potential to achieve _better_ performance even with simplified reasoning calculations. 

## **4 Experiment** 

## **4.1 Dataset and Metric** 

We conduct experiments on image-level activity understanding benchmarks with diverse tasks: **HICO** [5] (Human-Object Interaction (HOI) recognition), **Stanford40** [44] (action recognition), **HAKE** [28] **-Verb** (verb recognition), **HAKE** [28] **-PaSta** (conditional PaSta Q-A). HAKE [28]-Verb and HAKE [28]-PaSta is the newly constructed benchmark based on HAKE [28] data. We adopt mAP metric for HICO [5], HAKE [28]-Verb and Stanford40 [44], and top-1 accuracy metric for HAKE [28]-PaSta. For more details, please refer to the supplementary. 

## **4.2 Symbolic System Experiment** 

**An example.** The LLM used is OpenAI GPT3.5. To demonstrate the instantiation of our symbolic system, we take the activity “human board an airplane” as an example. Tab. 1 shows the symbols and rules generated, and Fig. 4 illustrates the detailed process. In the 1st round, 5 initial symbols are generated. Then in the 2nd round, they are extended into rules satisfying entailment. Later, the new symbols are used in the next round. Finally, 17 symbols and 12 rules are generated within 5 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0008-00.png)


**----- Start of picture text -----**<br>
a) 20 symbols count b) 1.0<br>rules count 0.8<br>15<br>0.6<br>Rule 0 Rule 5<br>10 0.4 Rule 1 Rule 6<br>Rule 2 Rule 7<br>5 0.2 Rule 3 Rule 8<br>Rule 4 Rule 9<br>0 0.0<br>1 2 3 4 5 1 2 3 4<br>Round #Premise Symbols in the Rule<br>Count<br>Entailment Score<br>**----- End of picture text -----**<br>


Figure 5: Statistics of the symbolic system. a) Accumulated symbols & rules count in each round. The data is from 50 randomly sampled activities. The average value and fluctuation range are shown. b) Increased entailment scores as the 10 randomly sampled rules extended with more premise symbols. 

rounds. In practice, at most 15 symbols are used for rule extension to avoid semantic redundancy. For example, in Fig. 4, symbols 16 and 17 are not used for rule extension, leading to a stop in the 5th round. From Fig. 4 and Tab. 1, we can find that diverse symbols are mined, including corner cases such as “adjust seatbelt” “reach for the airplane door handle”, verifying the effectiveness of the proposed symbolic system. Some special cases are shown as gray text. The rule “6 _→_ 1 _→_ 7 _→_ 8” is equivalent to the existing rule “1 _→_ 6 _→_ 7 _→_ 8”, thus omitted. Extension from the symbols 10 and 14 stops early because adding premises nevertheless leads to a drop in entailment scores. In practice, multiple rules are extended from a known symbol to increase diversity. 

**Entailment.** Fig. 5 shows statistics of the symbolic system. In Fig. 5a, we randomly sample 50 activities and show the accumulated symbols and rules count in each round, verifying the diversity of the generated symbols and rules. In Fig. 5b, we randomly sample 10 rules and show the entailment scores change as the rule extends with more premise symbols. For example, for rule 0: _m_ 1 _∧m_ 2 _∧m_ 3 _∧ m_ 4 _|_ = _c_ , the entailment score for _m_ 1 _|_ = _c, m_ 1 _∧m_ 2 _|_ = _c, m_ 1 _∧m_ 2 _∧m_ 3 _|_ = _c, m_ 1 _∧m_ 2 _∧m_ 3 _∧m_ 4 _|_ = _c_ are 0.28, 0.86, 0.87, and 0.93, respectively. The score is calculated by querying 5 times and averaged. We find a climb up in the entailment score , implying the effectiveness of the entailment check. Once the entailment score _e_ surpasses the threshold _eh_ = 0 _._ 9, it is regarded as a rule equipped with logical entailment and updated into the symbolic system. 

**Performance Evaluation.** To evaluate the symbolic system, we construct a SymAct (Symbol Activity) test set. It is a small subset of the HICO test set (120 images for 10 activity classes). For each image and each of its ground-truth activities, we list all of the symbols related to the activity in the symbolic system and annotate whether a symbol happens (0/1) in the image via human judgment. We find the proposed symbolic system has: **1) broader semantic coverage** than HAKE. For quantitative measurement, we count the happening symbols for each image-activity pair under different symbolic systems. The average number is 10.8 for ours and 1.8 for HAKE. **2) more rational rules** than HAKE. We evaluate confusion by counting different image-activity pairs which share the same symbols. For the HAKE symbolic system, the confusion problem is severe, with 261 confusion pairs over _C_ 120[2][= 7140][ pairs (accounting for 3.7%).][For our symbolic system, there are no confusion] pairs on the test set because of the presentation ability of symbols and entailment check. 

**Robustness.** We analyze the robustness of the generated symbolic system. We find it satisfies: **1) Convergence.** After a certain stopping criterion, a new round will bring few or even no new symbols to the sub-system. In practice, the stopping criterion is: At most 15 symbols are used for rule extension to avoid semantic redundancy. After that, new symbols are not used for rule extension, leading to a stop of the loop. **2) Low sensitivity to initial conditions.** The symbol-rule loop provides a reliable and stable symbolic sub-system over a range of initial conditions. **3) Low sensitivity to different prompts.** Experiment results are less sensitive to prompts with minor differences, _i_ . _e_ ., slightly paraphrasing the prompts. For more details, please refer to the supplementary. 

## **4.3 Visual Reasoning Experiment** 

**Implementation and Settings.** For visual reasoning, we use BLIP2 [24] ViT-g FlanT5-XL model to extract visual symbols. BLIP2 [24] is a visual-language pre-trained model with a frozen large language model, thus well equipped with question-answering abilities to effectively extract symbols. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0009-00.png)


**----- Start of picture text -----**<br>
||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|I|II|III|IV|Symbol probability|I|II|III|IV|
|𝒎𝟏|: talk with seller|0.92|0.87|0.07|0.16|
|𝒎𝟐|: reach for an orange|0.14|0.68|0.95|0.12|
|𝒎𝟑|: seller hand over orange|0.94|0.82|0.09|0.13|
|𝒎𝒂𝒙(𝒎𝒊𝒏𝒎𝒊𝒏𝒎𝒊𝒏0.57,0.67,0.680.92,0.940.92,0.14,|,|,|Rules for|𝑟3:𝑟𝑟 𝑚12::4 ∧𝑚𝑚11𝑚 ∧ ∧𝒄:5 buy orange𝑚 ∧𝑚23𝑚⊨⊨6|𝒄,𝒄,⊨|𝒄,|𝒎𝒎𝒎𝟒𝟓𝟔|: stand in front of fruit stand: : pick orange from a basketplace orange in a bag|0.570.670.68|0.850.900.83|0.040.060.03|0.770.450.74|
|𝒎𝒊𝒏|0.57,0.47,0.86|,|𝑟4:|𝑚4 ∧|𝑚7 ∧|𝑚8|⊨|𝒄,|𝒎𝟕|: hold a bag of oranges|0.47|0.88|0.05|0.75|
|𝒎𝒊𝒏|0.57,0.83,0.82|)|𝑟5:|𝑚4 ∧|𝑚9 ∧|𝑚10|⊨|𝒄|𝒎𝟖|: reach for a wallet|0.86|0.83|0.04|0.43|
|……|……|……|𝒎𝟗|: seller put the orange in bag|0.83|0.86|0.04|0.45|
|0.92|0.85|0.07|0.45|𝒎𝟏𝟎|: give money to seller|0.82|0.87|0.04|0.41|

**----- End of picture text -----**<br>


Figure 6: Visualization results. Symbols & activity predictions for “human buy orange” are shown. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0009-02.png)


**----- Start of picture text -----**<br>
||||||||
|---|---|---|---|---|---|---|
|HICO mAP|HAKE-verb mAP|Stanford-40|HAKE-PaSta|
|Method|
|fine-tuned|zero-shot|fine-tuned|zero-shot|zero-shot mAP|zero-shot Acc(%)|
|CLIP [37]|67.12|37.08|73.82|43.92|75.68|39.36|
|CLIP [37]+Reasoning|69.73|43.21|75.27|48.95|82.22|40.47|
|BLIP2 [24]|-|50.61|-|49.47|91.85|43.81|
|BLIP2 [24]+Reasoning|-|53.15|-|51.37|92.59|44.65|

**----- End of picture text -----**<br>


Table 2: Results on activity benchmarks. More HICO [5] baselines are listed in supplementary. 

As a _plug-and-play_ , the reasoning is compatible with existing System-1-like methods. We follow the setting of HAKE [26] to integrate the reasoning result with the prediction of System-1-like methods. We mainly choose 2 typical System-1-like methods: 1) CLIP [37]: a visual-language model pre-trained with large-scale image-text pairs and a contrastive objective, where we use its ViT-L/14 model with a resolution of 336 pixels; 2) BLIP2 [24]: with its image-to-text generation ability, it functions either as a System-1-like baseline, or as the symbol extractor for reasoning. To fully test the ability of the models, we adopt both “fine-tuned” and “zero-shot” settings, where the former model is fine-tuned on the training set of the dataset. 

**Results.** The results of visual reasoning are shown in Tab. 2. Comparing the two baselines, we find that with a frozen language model, BLIP2 [24] is more capable of understanding activity semantics and outperforms CLIP [37] in various zero-shot benchmarks. Under the fine-tuned setting, our method achieves **69.73** and **75.27** mAP respectively on HICO [5] and HAKE [28]-Verb, which outperforms the baseline CLIP [37] with **2.61** and **1.45** mAP respectively, verifying its effectiveness. Under the zero-shot setting, our method also outperforms existing System-1-like methods. The improvement compared with CLIP [37] is: **6.13** mAP (HICO [5]), **5.03** mAP (HAKE [26]-Verb), **6.54** mAP (Stanford40 [44]), **1.11** % acc (HAKE [26]-PaSta). The improvement compared with BLIP2 [24] is: **2.54** mAP (HICO [5]), **1.90** mAP (HAKE [26]-Verb), **0.74** mAP (Stanford40 [44]), **0.84** % acc (HAKE [26]-PaSta). In conclusion, our method brings performance improvement and shows significant zero-shot _generalization_ abilities in various activity understanding tasks. Notably, the improvement is achieved by exploiting knowledge instead of adding visual training samples. The exploited knowledge can be reused for future study. Thus, our method is superior in _data efficiency_ . Further, we visualize the results in Fig. 6. The rules and the logic calculation add _explainability_ to the inference process. Also, they help to correct false negative (I, II in Fig. 6) and false positive (III, IV) cases, thus improving compositional generalization. 

## **4.4 Ablation Study** 

We conduct ablation studies on HAKE [26]-Verb zero-shot setting and list the results in Tab. 3. 

**Reasoning.** We first verify the effectiveness of integrating System-2 reasoning. Results of CLIP [37] baseline can be either combined with reasoning results or trivially combined with other baseline results (CLIP [37]+BLIP2 [24]). We find that the former is superior (48.95 mAP) to the latter (44.76 mAP), though both outperform the baseline CLIP [37] (43.92 mAP). It verifies the necessity of System-2 reasoning other than trivially combining predictions from two models. 

9 

**Symbolic System.** We then verify the effects of the symbolic system on visual reasoning. We adopt the symbolic system in HAKE [26] and find a degradation from 48.95 to 44.98 mAP, as its defects in semantic coverage and rule rationality, and the shortage is amplified in the fuzzy logic calculation without learnable pa- **Method mAP** rameters. Next, we construct the symbolic system without logical CLIP [37]+Reasoning **48.95** entailment, where each rule has 3 premise symbols without checking CLIP [37]+BLIP2 [24]CLIP [37] 44.7643.92 its entailment score. It leads to a dropped performance because of rules from HAKE [26] 44.98 rules with errors or incomplete semantics. Then, the symbol-rule w/o entailment check 46.51 loop is removed, with only rules in the 1st and 2nd rounds updated w/o s80% rulesymbol-rule loop 47.5348.57 to the symbolic system. As the diversity of rules decreases, the model 50% rules 47.61 suffers from a decreased performance. The importance of diversity 20% rules 46.22 is further verified by the performance drop when only 80%, 50%, and CLIP [37] as extractorw/o checking symbols 46.0948.17 20% randomly sampled rules are used. 

Table 3: Ablation studies on zero-shot HAKE [26]-Verb. 

**Symbols Extractor/Checker.** Next, we replace the symbol extractor BLIP2[24] with CLIP [37] and find a performance fall with a weaker ability to extract visual information. Also, reasoning without a symbol checker suffers from degradation due to the negative effect of inaccurate symbol predictions. 

## **4.5 Bottleneck Analysis** 

Based on the SymAct test set, we make further analysis of the bottleneck of System-2 reasoning. We investigate the impact of symbol prediction and symbolic system separately. With perfect symbol predictions, we assume the probabilities of symbols are definitely known as 0/1 instead of _p ∈_ [0 _,_ 1]. The ground-truth symbols are annotated as explained in Sec. 4.2. With a perfect symbolic system, we assume that the generated rules cover all samples (image-activity pairs). The performance drop (100.00 _→_ 60.79 mAP, 83.11 _→_ 41.52 mAP) indicates the symbolic system **Symbol Prediction Symbolic System mAP** errors, _i_ . _e_ ., the generated rules have not covered all perfect perfect **100.00** samples (image-activity pairs). The drop (100.00 imperfectperfect imperfectperfect **83.1160.79** _→_ 83.11 mAP, 60.79 _→_ 41.52 mAP) indicates the imperfect imperfect **41.52** errors of symbol predictions, _i_ . _e_ ., symbol probabilities Table 4: System-1/2 analysis on SymAct cannot be predicted accurately. In the supplementary, test set. we also provide two failure cases that emerge from symbol prediction errors and symbolic system errors respectively. 

## **5 Conclusion and Discussion** 

In this work, we rethink the symbolic system in activity reasoning and propose a new one with broad-coverage symbols and rational rules. Thus, enhanced System-2 reasoning is integrated into System-1. We demonstrate how to instantiate it and how it helps to reason with visual inputs. Our method shows superiority in explainability, generalization, and efficiency in extensive experiments. 

**Computation Cost.** Symbol predictions will increase the computational cost as a trade-off for explainability and generalization. It can be eased by discovering the hierarchical and reusable nature of the symbols, which is detailed in the supplementary. 

**Broader Application.** The method could be more general and facilitate real-world applications. We choose activity understanding as a good and important initial test bed because it is a difficult task with complex visual patterns and a compositional nature. To verify the broader application, we provide some initial results on object recognition and VCR [47] tasks in the supplementary. 

**Limitation.** Rules generated from language models mostly boost System-1, but sometimes bring language bias, _e_ . _g_ ., the person only decides to buy an orange despite apples in the background. Further, the approximation is mainly based on pre-trained LLMs and can be improved with more elaborate designs, _e_ . _g_ ., human-in-the-loop, customized LLMs with higher-quality knowledge. Besides, System2 reasoning may be boosted if integrated into the System-1 training process. Also, instance-level activities and temporal encoding are beyond our scope since our main focus is on an overall framework to formulate and construct a novel symbolic system. We leave them to future work. 

10 

## **Acknowledgments and Disclosure of Funding** 

Some of the drawing materials in Fig. 1,2 are designed by Layerace / Freepik. 

This work is supported in part by the National Natural Science Foundation of China under Grants 62306175, National Key R&D Program of China (No.2021ZD0110704), Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), Shanghai Qi Zhi Institute, Shanghai Science and Technology Commission (21511101200). 

## **Appendix** 

## **A Notations** 

In Tab. 5, we conclude the notations in this work for clarity. 

|Notation|Notation|Defnition||||
|---|---|---|---|---|---|
|_r_||A rule. It takes the form_r_ :_m_1_∧m_2_∧. . . ∧mnr |_=_c_.||||
|_M_ =|_{mi}nr_<br>_i_=1|The premise symbols set of the rule_r_.||||
|_c_||The conclusion symbol of the rule_r_.||||
|_nr_||The size of the premise symbols set_M_.||||
|_∧_||Logic AND.||||
|_∨_||Logic OR.||||
|_|_=||Entailment.||||
|(_S, R_)||The activity symbolic system. _S_ is the symbol set, and_R_is the rule set.||||
|(_Sc, Rc_)||The activity symbolic sub-system given a conclusion_c_.||||
|_A\B_||The set difference of A and B.||||
|_D_||A very large-scale activity images database. _D_ =_{_(_I, A, SI_)_}_.||||
|_I_||An image.||||
|_AI_||The activities happening in an image_I_.||||
|_SI_||Ongoing symbols in an image_I_.||||
|_L_||An LLM.||||
|_A_||The activity set contains multiple activity classes. _A_=_{A}_||||
|_C_||The conclusion set contains multiple conclusions. _A_and_C_ is equivalent.||||
|_SA_||The premise symbols set for activity_A_. It is implied from an LLM.||||
|_e_||The entailment score of a rule.||||
|_eh_||The entailment score threshold to accept/reject a|rule. We set_eh_||= 0_._9.|
|_τ_(_·_)||A function to measure the entailment score_e_=_τ_(_r_)of||a rule_r_.||
|_r_(_j_)<br>_m_(_j_)<br>_i_||The_j_-th rule in_Rc_. It takes the form_r_(_j_) :_m_(_j_)<br>1<br>The_i_-th premise symbol in the_j_-th rule of_Rc_.|_∧m_(_j_)<br>2|_∧. . . m_(_j_)<br>_nr_(_j_) _|_=_c_.||
|_p_(_j_)<br>_y,i, p_(_j_)<br>_n,i_<br>_p_(_j_)<br>_i_||The output probability of answering “Yes"“No" for the symbol_m_(_j_)<br>_i_<br>from a VLM.<br>The probability of the symbol_m_(_j_)<br>_i_ .||||
|_Nquery,c_||The number of queries to generate the symbolic sub-system given the conclusion_c_.||||
|_Nent_||The number of sampling to calculate the entailment score. _Nent_|||= 5.|
|_|M_ (_j_)_|_||The size of the premise symbol set for a rule_r_(_j_).||||
|_{m_(_j_)<br>_i,k}_5<br>_k_=1||The paraphrased variants for the symbol_m_(_j_)<br>_i_ .||||
|_{p_(_j_)<br>_i,k}_5<br>_k_=1||The probability of the symbols_{m_(_j_)<br>_i,k}_5<br>_k_=1.||||
|_std_(_j_)<br>_i_||The standard deviation of_{p_(_j_)<br>_i,k}_5<br>_k_=1.||||
|_Ssys_1||The prediction of a System-1-like method.||||
|_Ssys_2||The prediction of a System-2-like method.||||
|_Sint_||The fnal integrated prediction.||||
|_α_1_, α_2||The re-scaling factors to normalize_Ssys_1_, Ssys_2.||||



Table 5: Notations and their definition of this work. 

**About the entailment notation.** We replace the notation “ _→_ ” in HAKE with “ _|_ =” for mathematical rigor. The notations “ _|_ =” (semantic entailment) and “ _→_ ” (logical/syntactic entailment) both describe the concept of one statement leading to another, but their emphasis is different. Semantic entailment 

11 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0012-00.png)


**----- Start of picture text -----**<br>
a) 𝑚𝑐(1)1,3, 𝑚𝑐(2)1,2, 𝑚𝑐(1)2,1 𝒄𝟐 𝒄𝟏 : H uman ride a boat b) Semantic Coverage<br>𝒄𝟐 : H uman fish Ima𝒟= {(𝐼, 𝐴, 𝑆ge Semantics𝐼)}<br>𝑚𝑐(1)1,2,𝑚 𝑚𝑐(1)𝑐(2)11,1,1 …𝒄𝟏 𝑚𝑐(2)…1,3 𝑚𝑐(3)1𝑚,1,𝑐(1) 𝑚2,3𝑐(1)2,2 𝑚the boat𝑚𝑐(1)𝑐(1)11,2,1, :   𝑚Hands control an engine𝑐(2)1,1 :  Hands hold onto the side of  … … Symbol Semantics(𝒮, ℛ)<br>Decomposition 𝑚𝑐(3)1,2 𝑚𝑐(1)1,3, 𝑚𝑐(2)1,2, 𝑚𝑐(1)2,1 :  Hip seated in a boat c)<br>… … … 𝑚𝑚𝑚𝑚𝑐𝑐(1)(3)𝑐(3)𝑐(2)2111,3,1,2,3 : , :  𝑚 Hands cast the line  :  Hands operate the steering wheel Legs stand on the boat deck𝑐(1)2,2 :  Hands hold the fishing rod 𝑚𝑚𝑚𝑚𝑚 Logical Entailment 12123 𝒓𝒓 [′] 𝑐𝑐 (𝐼(𝐼(𝐼(𝐼2131, 𝐴, 𝐴, 𝐴, 𝐴2131, 𝑆, 𝑆, 𝑆, 𝑆𝐼𝐼𝐼𝐼2131))))……√√√√√√√√√√√…√…√×<br>**----- End of picture text -----**<br>


Figure 7: Our activity symbolic system. a) The structure and decomposition. Its two important properties: b) semantic coverage and c) logical entailment. It details part of Fig. 2 in the main text. 

focuses on the reasoning relationship based on truth values or interpretations. When all interpretations that make all formulas in _A_ true also make _B_ true, we say that _A_ semantically entails _B_ . Syntactic entailment focuses on the ability to derive a conclusion from a set of premises based on formal reasoning rules. When _Q_ can be derived from the formulas in _P_ solely through reasoning rules, we say that _P_ syntactically entails _P_ . Semantic entailment focuses on truth tables or interpretations ( _i_ . _e_ ., under what circumstances or "worlds" a proposition is true), while syntactic entailment focuses on formal reasoning rules and proof processes. Visual reasoning focuses on interpretations instead of syntactic proof. Thus we replace the notation “ _→_ ” in HAKE with “ _|_ =”. 

## **B More Details of Symbolic System Formulation** 

## **B.1 Explanation of the Illustration** 

In the main text, Fig. 2 illustrates the overview of the proposed activity symbolic system. We give a more detailed explanation here. This section focuses on the part of Fig. 2 related to the symbolic system formulation. The other parts are left to Sec. C.1. 

Fig. 7a explains the structure and decomposition of our activity symbolic system. The premise symbols (gray circles), the conclusion symbols (red and green circles), and the rules (blue, red, brown, and green edges) are shown. Example meanings of symbols are also listed. When decomposing the activity symbolic system into sub-systems, unrelated symbols/rules are removed within each sub-system. Fig. 7b explains the semantic coverage property, _i_ . _e_ ., comparing the image semantics in the database _D_ and symbol semantics in the symbolic system ( _S, R_ ). Fig. 7c explains the logical entailment property. The rule _r_ : _m_ 1 _∧ m_ 2 _∧ m_ 3 _|_ = _c_ satisfies logical entailment, while _′ r_ : _m_ 1 _∧ m_ 2 _|_ = _c_ fails because in ( _I_ 2 _, A_ 2 _, SI_ 2), _{m_ 1 _, m_ 2 _} ⊂ SI_ 2 but _c ∈/ A_ 2. 

## **B.2 Asymmetry of Rule** 

In Sec. 3.1.1 in the main text, we analyze the structure of the symbolic system as a type of directed hyper-graph instead of the indirect hyper-graph. This is because of the asymmetry of the rule: within one rule, the premises symbol and the conclusion symbol cannot swap positions. That is, for a rule _′ r_ : _m_ 1 _∧ m_ 2 _∧ . . . ∧ mnr |_ = _c_ , , another rule _r_ : _m_ 1 _∧ . . . ∧ mi−_ 1 _∧ c ∧ mi_ +1 _. . . ∧ mnr |_ = _mi,_ (1 _≤ i ≤ nr_ ) is not necessarily satisfied. For example, 

_r_ : `Hip seated in a boat` _∧_ `Hand hold onto the side of the boat` _∧_ `Hand operate the steering wheel` _|_ = `Human ride a boat` _′ r_ : `Hip seated in a boat` _∧_ `Hand hold onto the side of the boat` _∧_ `Human ride a boat` _|_ = `Hand operate the steering wheel` 

_′_ The rule _r_ is not satisfied since there are other possible conclusions, _e_ . _g_ ., “ `Hand controls the engine` ”. Based on the asymmetry property of the rule, the activity symbolic system is defined 

12 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0013-00.png)


**----- Start of picture text -----**<br>
Semantic Coverage Approximation Symbol-Rule Loop<br>ℒ: {(𝐴, 𝑆𝐴)} Queue, 𝒮𝑐 [𝑐𝑎𝑛𝑑]<br>head tail<br>Language Semantics 𝑚𝑘𝑛𝑜<br>(𝒮, ℛ) pop push<br>… … Symbol Semantics<br>Rule Extension 𝑚𝑛𝑒𝑤<br>**----- End of picture text -----**<br>


Figure 8: Semantic coverage can be approximated based on LLMs’ knowledge and achieved via the symbol-rule loop. It details part of Fig. 2 in the main text. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0013-02.png)


**----- Start of picture text -----**<br>
Logical Entailment Approximation Entailment Check<br>Entailment   A rule 𝑟<br>score 𝑒= 𝜏(𝑟) 𝒓: 𝒎𝟏 ∧ 𝒎𝟐 ∧ 𝒎𝟑 ⊨ 𝒄<br>𝑒ℎ LLM<br>𝒓: 𝒎𝟏 ∧ 𝒎𝟐 ⊨ 𝒄 In a picture… Estimate the<br>probability …<br>𝒓: 𝒎𝟏 ⊨ 𝒄<br>0 Entailment  score 𝑒<br>**----- End of picture text -----**<br>


Figure 9: Logical entailment can be approximated based on an entailment scoring function and achieved by entailment check. It details part of Fig. 2 in the main text. 

based on a directed hyper-graph, with the edges pointing from the premise symbols to the conclusion symbol. 

## **B.3 Structure of the Previous Symbolic System** 

Existing methods [28, 26] can also be formulated under the structure in Def. 3 in the main text. Their symbolic system is a specific example of Def. 3, where the premise set _M_ for each rule _r_ is the subset of the hand-crafted activity primitives, _i_ . _e_ ., _M ⊂ M[∗]_ , _|M[∗] |_ = 76 and _M[∗]_ is a fixed set. 

## **C More Details of Symbolic System Instantiation** 

## **C.1 Explanation of the Illustration** 

We further explain the other part of Fig. 2 in the main text. As is shown in Fig. 8, semantic coverage can be approximated by comparing the symbol semantics in the symbolic system ( _S, R_ ) with the LLM semantics implying the premise symbols set _SA_ for an activity _A_ . In each symbolrule loop, a known symbol _mkno_ is taken from the queue _Sc[cand]_ , and the rule extension is applied with a new symbol _mnew_ added. As is shown in Fig. 9, the logical entailment is measured via an entailment scoring function _τ_ ( _·_ ). For the rule _r_ : _m_ 1 _∧ m_ 2 _∧ m_ 3 _|_ = _c_ , the entailment score for _m_ 1 _|_ = _c, m_ 1 _∧ m_ 2 _|_ = _c, m_ 1 _∧ m_ 2 _∧ m_ 3 _|_ = _c_ climbs up because more premise symbols are added. Once the entailment score _e_ surpasses the threshold _eh_ , it is regarded as a rule equipped with logical entailment and updated into the symbolic system. To implement the scoring function based on an LLM, we rewrite a rule _r_ as a sentence and design prompts. 

## **C.2 Summarized Pipeline** 

Alg. 2 in the main text shows how to get the sub-system ( _Sc, Rc_ ) for a given conclusion _c_ . We again show it in this supplementary and explain it for clarity. The target is to determine the rule set _Rc_ = _{r_[(] _[j]_[)] _}[N] j_ =1 _[c]_[, where a rule takes the form] _[ r]_[(] _[j]_[)][:] _[m]_[(] 1 _[j]_[)] _∧ m_[(] 2 _[j]_[)] _∧ . . . m_[(] _n[j] r_[)] ( _j_ ) _[|]_[=] _[c]_[, or equivalently,] _r_[(] _[j]_[)] :[�] _m∈M_[(] _[j]_[)] _[ m][ |]_[=] _[ c]_[.][Correspondingly, the symbol set is] _[ S][c]_[= (][�] _[N] j_ =1 _[c][M]_[ (] _[j]_[)][)] _[ ∪{][c][}]_[.] 

13 

**Algorithm 2** Instantiating the Symbolic System 

**Input:** conclusion _c_ , entailment threshold _eh_ **Output:** Symbolic Sub-System ( _Sc, Rc_ ) 1: _Sc_[0] _[←]_[Symbol_Initialization][ (] _[c]_[)] 2: _Sc[cand] .push_ ( _Sc_[0][)] _▷_ A queue _Sc[cand]_ stores symbols 3: _Sc ←{c}, Rc ←{}_ 4: **while** not _Sc[cand] .is_  empty_ () **do** 5: _mkno ←Sc[cand] .pop_ () _▷_ A known symbol _mkno_ is taken for rule extension 6: _M_ = _{mkno} ▷_ The premises set _M_ is set for the current rule 7: **while** Entailment_Check( _M_ ) _< eh_ **do** 8: _mnew ←_ Rule_Extension( _M, c_ ) 9: _M_ = _M ∪{mnew} ▷_ A new symbol _mnew_ is added 10: **end while** 11: _Sc ←Sc ∪ M ▷_ The symbol set _Sc_ is updated 12: _Rc ←Rc ∪{r_ :[�] _m∈M[m][ |]_[=] _[ c][}] ▷_ The rule set _Rc_ is updated 13: _Sc[cand] .push_ ( _M \Sc_ ) _▷M_ is added to _Sc[cand]_ with redundant symbols removed 14: **end while** 15: **return** ( _Sc, Rc_ ) 

First, the initial symbols _Sc_[0][are generated (L1 in Alg. 2).][For the symbol-rule loop (L4-14), a queue] _Sc[cand]_ stores candidate symbols, and each loop processes one symbol _mkno_ in _Sc[cand]_ . In each loop, a known symbol _mkno_ is taken as an element in the premise symbol set _M_ , and the rule extension is repeatedly applied with the new symbol _mnew_ added to _M_ until the entailment check is passed (L7-10). Thus, a rule _r_ :[�] _m∈M[m][ |]_[=] _[ c]_[ which satisfies logical entailment is obtained and added to] the rule set _Rc_ (L12). Meanwhile, the new symbols are added to the symbol set _Sc_ and the candidate symbol set _Sc[cand]_ , and the repeated symbols are removed. The symbol-rule loop continues until _Sc[cand]_ is emptied. Finally, we get the symbol set _Sc_ and rule set _Rc_ for the conclusion _c_ . 

## **C.3 Determining Prompts** 

The prompts are determined via a trial-and-error process by human experts on randomly sampled 10 activities. Human experts have read the image samples of these activities, and thus have expert knowledge about what symbols/rules are more suitable for visual reasoning. Each round in the trial-and-error process is as follows: 

- one expert writes a prompt 

- an LLM generates answers by the prompt 

- all experts judge whether the answer is suitable 

- all experts modify the prompt based on the discovered problems 

An example of the trial-and-error process of the prompts for symbol initialization is shown below. 

1. Round1: 

   - Prompt: In a picture, a person is boarding an airplane. What are useful visual features to distinguish his activity? 

   - Answer: Useful visual features...could include: 1. Luggage: Look for the presence of suitcases, backpacks, or other types of bags that suggest the person is preparing to travel... 

   - Comment: Object-centric, difficult to extract key information about human activities. 

2. Round2: 

   - Prompt: In a picture, a person is boarding an airplane. What is he doing? 

   - Answer: The person in the picture is boarding an airplane. Boarding refers to the process of entering the aircraft before the flight begins. It typically involves presenting a valid boarding pass, showing identification, and walking through the designated gate or bridge to reach the plane. 

14 

   - Comment: Verbose. 

3. Round3: 

   - Prompt: In a picture, a person is boarding an airplane. What is he doing? Answer with 5 concise, highly related phrases. The format is "<human> <verb> <object>". 

   - Answer: Person boarding airplane. 

   - Comment: The answer repeats the question. 

4. Round4: 

   - Prompt: In a picture, a person is boarding an airplane. What are his hands doing? Answer with 5 concise, highly related phrases. The format is "<hands> <verb> <object>". 

   - Answer: 1. Hands holding a boarding pass. 2. Hands placing luggage in overhead compartment. 3. Hands adjusting seatbelt. 4. Hands waving goodbye to loved ones. 5. Hands gripping a luggage handle. 

   - Comment: Satisfactory. 

## **C.4 An Example of Instantiation** 

We show the detailed process of generating the symbolic sub-system for the activity “ `human board an airplane` ”. It is an extension of the results in Fig. 4, Tab. 1 in the main text, and the method described in Sec. 3.2 in the main text. The LLM used here is OpenAI GPT3.5 via API, where the role is set as “You are helping me understand human activities in a picture.”. 

To generate initial premise symbols (L1 in Alg. 2), we prompt as: 

(Symbol Initialization) `In a picture, a person is boarding an airplane. What are his hands doing? Answer with 5 concise, highly related phrases. The format is "<hands> <verb> <object>". Output Format: 1. xxx 2. xxx 3. xxx 4. xxx 5. xxx` 

Then 5 initial symbols are generated from the answer, _e_ . _g_ ., “ `hold a boarding pass` ”, “ `place luggage in overhead compartment` ”. 

Then, we extend the symbol “ `hold a boarding pass` ” into a rule. To add a new premise symbol (L8 in Alg. 2), we prompt as: 

(Rule Extension) `In a picture, there is an airplane. IF [The person is holding a boarding pass.] AND [condition] THEN [The person is boarding the airplane.]. [condition] is one concise phrase. The format is "<The person’s hands/arms/hip/legs/feet> <verb> <object>". What is [condition]? Output Format: [condition] is: [xxx].` 

With an answer “[condition] is: The person is walking towards the boarding gate”, we get the new symbol “ `walk towards the boarding gate` ” and a candidate rule “ `hold a boarding pass` _∧_ `walk towards the boarding gate` _|_ = `Human board an airplane` ”. 

Then, an entailment check is applied for the candidate rule (L7 in Alg. 2). The prompt is: 

(Entailment Check) `In a picture, there is an airplane. The person is holding a boarding pass. The person is walking towards the boarding gate. Estimate the probability that he is boarding the airplane at the same time. Choose from: (a) 0.1, (b) 0.5, (c) 0.7, (d) 0.9, (e) 0.95, (f) unknown. Output Format: a/b/c/d/e/f.` 

We sample the answer 5 times and take the average entailment score. If the score _e < eh_ , we repeat the rule extension and entailment check. 

15 

(Rule Extension) 

```
Inapicture,thereisanairplane.IF[Thepersonisholding
aboardingpass.Thepersoniswalkingtowardstheboarding
gate.]AND[condition]THEN[Thepersonisboardingtheairplane.]
[condition]isoneconcisephrase.Theformatis"<Theperson’s
hands/arms/hip/legs/feet><verb><object>".Whatis[condition]?
OutputFormat:[condition]is:[xxx].
```

With a new symbol “ `luggage visible beside him` ”, the entailment score of the new rule is measured via prompting: 

(Entailment Check) 

```
Inapicture,thereisanairplane.Thepersonisholdingaboarding
pass.Thepersoniswalkingtowardstheboardinggate.Theluggageis
visiblebesidehim.Estimatetheprobabilitythatheisboardingthe
airplaneatthesametime.Choosefrom:(a)0.1,(b)0.5,(c)0.7,(d)
0.9,(e)0.95,(f)unknown.OutputFormat:a/b/c/d/e/f.
```

The rule extension and entailment check are repeated until the entailment score threshold is reached. Finally, we get a rule 

`hold a boarding pass` _∧_ `walk towards the boarding gate` _∧_ `luggage visible beside him` _∧_ `boarding pass is scanned by airport staff` _|_ = `Human board an airplane` 

The prompt differences are small modifications necessary to adapt different dataset/task settings. When an interacted object is known, it is listed in the rule extension prompt for emphasis ( _e_ . _g_ ., "there is an airplane"). For HAKE-PaSta, HOIs are known as conditions, and the symbol initialization is omitted because the known HOIs are initial premise symbols for rule extension. 

## **C.5 Computation Cost** 

Given a conclusion _c_ , the number of query _Nquery_ with a LLM is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0016-10.png)


where 1 refers to the first query to generate initial symbols, _Nent_ = 5 is the number of sampling to calculate the entailment score, and _|M_[(] _[j]_[)] _|_ is the size of the premise symbol set for a rule _r_[(] _[j]_[)] . 

For the example shown in Fig. 4 and Tab. 1 in the main text, we have _Nquery_ = 1 + 5 _∗_ (4 + 2 + 3 + 3 + 2 + 4 + 3 + 2 + 2 + 3 + 3 + 3 + 3 + 3 + 3) = 216, where the early stop symbols “10”“14” also need several queries to determine early stop. The exploited knowledge is reusable to facilitate future study. 

## **C.6 Comparing with HAKE** 

To compare the difference of the symbolic system between our method and HAKE [26], we list some of the rules for the activity “ `human buy an orange` ” in Tab. 6. We can see that our symbolic system is superior in semantic coverage and rule rationality. 

## **C.7 Performance Evaluation** 

To evaluate the symbolic system, we construct a SymAct (Symbol Activity) test set. It is a small subset of the HICO test set (120 images for 10 activity classes). The 10 activity classes are: "human board/wash an airplane", "human park/repair a bicycle", "human feed/race a horse", "human break/sign a baseball bat", and "human buy/wash an orange". These activities have relatively complex visual patterns and cover different interacted objects. For each image and each of its ground-truth activities, we list all of the symbols related to the activity in the symbolic system and annotate whether a symbol happens (0/1) in the image via human judgment. 

16 

We find the proposed symbolic system has: **1) broader semantic coverage** than HAKE. For quantitative measurement, we count the happening symbols for each image-activity pair under different symbolic systems. The average number is 10.8 for ours and 1.8 for HAKE. Though a rough estimation, the gap in symbol counts indicates that the symbol semantic of HAKE is limited. Furthermore, in our symbolic system, different activities have different symbols, further boosting semantic coverage. **2) more rational rules** than HAKE. We evaluate confusion by counting different image-activity pairs which share the same symbols. For the HAKE symbolic system, the confusion problem is severe, with 261 confusion pairs over _C_ 120[2][= 7140][ pairs (accounting for 3.7%).][For our] symbolic system, there are no confusion pairs on the test set because of the presentation ability of symbols and entailment check. 

|talk with seller|||_∧_|reach for an orange|-|-||_|_=buyan orange|
|---|---|---|---|---|---|---|---|---|
|talk with seller|||_∧_|seller hand over orange|-|-||_|_=buyan orange|
|stand in front of|fruit|stand|_∧_|place orange in a bag|_∧_|pick orange from a basket||_|_=buyan orange|
|stand in front of|fruit|stand|_∧_|hold a bagof oranges|_∧_|reach for a wallet||_|_=buyan orange|
|stand in front of|fruit|stand|_∧_|sellerput the orange in bag|_∧_|give moneyto seller||_|_=buyan orange|
|||||(a) Rules from our proposed|symbolic system.||||
|head: talk to|-|-||-<br>-||-|-|_|_=buyan orange|
|hand: reach for|_∧_|head: inspect<br>-<br>-||||-|-|_|_=buyan orange|
|hand: hold|_∧_|head: smell<br>-<br>-||||-|-|_|_=buyan orange|
|talk with seller|_∧_|reach for an orange<br>-<br>-||||-|-|_|_=buyan orange|
|hand: hold|_∧_|hand: reach for<br>_∧_<br>hand: squeeze||||_∧_|head: drink with|_|_=buyan orange|



(b) Rules from the open-source code of HAKE [26]. 

Table 6: Rules for the activity “ `human board an airplane` ” from the proposed symbolic system and HAKE [26]. 

## **C.8 Robustness** 

We analyze the robustness of the generated symbolic system as follows. 

**Convergence.** The symbol-rule loop should find a symbolic sub-system that "converges". In other words, after a certain stopping criterion, a new round will bring few or even no new symbols to the sub-system. In practice, the stopping criterion is: At most 15 symbols are used for rule extension to avoid semantic redundancy. After that, new symbols are not used for rule extension, leading to a stop of the loop. With the stopping criterion, the symbol-rule loop satisfies convergence. Take Tab.1 as an example: in rounds 1-5, the newly added symbols count is 5, 5, 3, 3, 1, which shows a convergence trend. Fig.5(a) in the main text shows a similar convergence trend. More generally, for the randomly sampled 50 activities, a new round brings an average of 1.9 symbols to the sub-system, verifying the convergence. 

**Sensitivity to initial conditions.** The symbol-rule loop should provide a reliable and stable symbolic sub-system over a range of initial conditions. The sensitivity to initial conditions mainly comes from the asymmetry of rule generation: with 4 as an initial symbol, a rule connecting 4&10&6 is generated, while the rule sometimes cannot be generated with 10 as an initial symbol. However, we find such sensitivity an infrequent case with multiple rules sampling. To verify this, we randomly sample 15 different initial symbol sets to generate the sub-system and then measure the average pairwise graph similarity. The average similarity on 10 randomly sampled activities is 0.91, verifying the low sensitivity to initial conditions. 

**Sensitivity to different prompts.** Experiment results are sensitive to prompts with major differences. Using the older version prompt during the trial-and-error process brings difficulty to correct and automated symbol generation. Experiment results are less sensitive to prompts with minor differences, _i_ . _e_ ., slightly paraphrasing the prompts. We conducted a brief experiment: We get another 2 paraphrased symbol init prompts and another 2 paraphrased rule extension prompts. We use these 3 _∗_ 3 = 9 prompts to generate 9 sub-systems and measure the average pairwise graph similarity. The average similarity (range: [-1,1]) on 10 randomly sampled activities is 0.83, verifying the relatively low sensitivity to different prompts. 

17 

## **D More Details of Visual Reasoning** 

## **D.1 Visual Symbol Extractor** 

The visual symbol extractor is based on a VLM BLIP2 [24] with its language model used as a scoring function [1]. A language model represents a distribution over potential completions _p_ ( _wk|w<k_ ), where _wk_ is a word that appears at a _k_ -th position in a text. While typical generation applications sample from this distribution, we can also use the model to score a candidate completion selected from a set of options [1]. With a symbol converted into a sentence, _e_ . _g_ ., “The person’s hip is seated in a boat. Yes/No?”, the language model outputs the probabilities of constrained responses “Yes”“No” as _py,i_[(] _[j]_[)] _[, p] n,i_[(] _[j]_[)][.][Then the probability] _[ p]_[(] _i[j]_[)] of the symbol can be obtained via normalizing the Yes/No answer with softmax function as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0018-03.png)


## **D.2 Visual Symbol Checker** 

To check symbols, one symbol _m_[(] _i[j]_[)] is paraphrased [6] into 5 variants _{m_[(] _i,k[j]_[)] _[}] k_[5] =1[.][The predictions of] variants should be similar due to semantic consistency. Thus, the standard deviation _std_[(] _i[j]_[)] of the predictions _{p_[(] _i,k[j]_[)] _[}] k_[5] =1[is calculated.][In practice, the symbols] _[ m]_[(] _i[j]_[)] with _std_[(] _i[j]_[)] _≥_ 0 _._ 05 are regarded as uncertain symbols and filtered out. The proportion of filtered symbols is around 5%. 

## **D.3 Dataset and Metric** 

We conduct experiments on image-level activity understanding benchmarks with diverse tasks. 

**HICO** [5] is a Human-Object Interaction (HOI) recognition benchmark, with 38,116 / 9,658 images in the train/test set. It defines 600 HOIs composed of 117 verbs and 80 COCO objects [30]. 

**Stanford40** [44] is an activity recognition dataset, with 9,532 images for 40 actions. Its train/test set contains 4,000/5,532 images. We omit its bounding box annotations and focus on the image-level recognition task. 

**HAKE** [28, 27] provides 118K+ images, which include 285K human instances, 250K interacted objects, and 724K HOI pairs with human body part states (PaSta) [28, 32]. To utilize the abundant activity samples, we follow the setting in [25] to split HAKE [28], with 22,156 images in the test set. Differently, we design different tasks to facilitate activity reasoning. 

**HAKE** [28, 27] **-Verb** is a verb recognition benchmark that defines 156 verbs related to human activity. To remove ambiguity, the verb prediction is conditioned upon ground-truth object labels, _i_ . _e_ ., excluding verbs contradictory to known objects. 

**HAKE** [28, 27] **-PaSta** is a multi-choice QA benchmark to predict PaSta with the ongoing HOIs known as conditions, dealing with finer-grained human activity. To maintain the sample balance of PaSta, we select a subset of 1,438 images from the original one [25] and match a QA pair for each image. 

For HICO [5] and HAKE [28, 27]-Verb, we use mAP for multi-label classification. For Stanford40 [44], we use mAP following the original setting. For HAKE [28]-PaSta, we use top-1 accuracy for multi-choice QA. 

For HICO [5], HAKE [28]-Verb and Stanford40 [44], symbols and rules are generated in the summarized pipeline in Sec. 3.2.3 in the main text and Sec. C.2 in supplementary. 

## **D.4 Results and Analysis** 

We further analyze the visual reasoning results in Tab. 2 in the main text. 

**More baselines.** We omit some HICO [5] baselines in the main text because of space limitations. We provide them here in Tab. 7. 

18 

|**Method**|**mAP**<br>**fne-tuned**<br>**zero-shot**|**mAP**<br>**fne-tuned**<br>**zero-shot**|
|---|---|---|
|R*CNN [17]|28.50|-|
|Girdhar_et al_. [16]|34.60|-|
|Mallya_et al_. [34]|36.10|-|
|Pairwise [10]|39.90|-|
|RelViT[33]|40.10|-|
|CLIP [37]|67.12|37.08|
|CLIP[37]+Reason|**69.73**|**43.21**|
|BLIP2 [24]|-|50.61|
|BLIP2[24]+Reason|-|**53.15**|



Table 7: Visual reasoning results on HICO [5]. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0019-02.png)


**----- Start of picture text -----**<br>
Symbol probability I II Symbol probability I II<br>𝒎𝟏 : hold a boarding pass  0.31 0.09 𝒎𝟏𝟎 : luggage is loaded onto the plane 0.73 0.77<br>𝒎 co 𝟐 m : place luggage in overhead  partment 0.86 0.76 𝒎 handle  𝟏𝟏 : reach for the airplane door  0.22 0.91<br>𝒎𝒎𝟑𝟒 : adjust seatbelt : wave goodbye to loved ones 0.470.22 0.970.09 𝒎 lugg 𝟏𝟐 a: stand in line with carry-on ge 0.11 0.04<br>𝒎𝟓 : grip a luggage handle 0.78 0.52 𝒎𝟏𝟑 : hold the carry-on luggage 0.30 0.64<br>𝒎𝟔 : walk towards the boarding gate  0.21 0.05 𝒎𝟏𝟒 : open the airplane door  0.34 0.91<br>𝒎𝟕 : luggage visible beside him 0.89 0.34 𝒎𝟏𝟓 : move forward in the line 0.40 0.28<br>𝒎 airport staff  𝟖 : boarding pass is scanned by  0.86 0.07 𝒎𝟏𝟔 : move towards the airplane door 0.10 0.72<br>𝒎𝟗 : stand on the jet bridge 0.54 0.44 𝒎 boardin 𝟏𝟕 : airline staff checking the g pass 0.78 0.13<br>I II<br>Rules for  𝒄:<br>board airplane<br>0.31 0.44<br>**----- End of picture text -----**<br>


Figure 11: Two failure cases for the activity "human board an airplane". The rules are the same as those in the main paper Tab.1. 

**Integrating System-1/2-like methods.** As a plug-and-play, the reasoning is compatible with existing System-1-like methods. We follow the setting of HAKE [26] to integrate the System-2 reasoning result _Ssys_ 2 with the prediction _Ssys_ 1 of System-1-like methods. The final prediction is _Sint_ = _α_ 1 _∗Ssys_ 1 + _α_ 2 _∗Ssys_ 2, where _α_ 1 _, α_ 2 is a re-scaling factor to normalize _Ssys_ 1 _, Ssys_ 2 and calculated from _Ssys_ 1 _, Ssys_ 2. 

**Performance gap between System-1-like methods.** We mainly choose 2 typical System-1-like methods: CLIP [37] and BLIP2 [24]. Comparing the two baselines, we find that with a frozen language model, BLIP2 [24] is more capable of understanding activity semantics and outperforms CLIP [37] in various zero-shot benchmarks. Furthermore, the performance boost from reasoning is more evident on CLIP [37] than on BLIP2 [24]. For example, on HICO [5], visual reasoning improves 6.13 mAP for CLIP [37] while 2.54 mAP for BLIP2 [24]. The performance gap between System-1-like methods can be partly explained by the model ensemble: the visual symbol extractor in reasoning is based on BLIP2 [24], thus applying reasoning to CLIP [37] model brings in more bonus. In ablation studies, we analyze the effects of the model ensemble (Sec. 4.4, the Reasoning section in the main text). We find that trivially combining with baseline results (CLIP [37]+BLIP2 [24]) (44.76 mAP) outperform the baseline CLIP [37] (43.92 mAP), but integrating reasoning is superior (48.95 mAP). It verifies the necessity of System-2 reasoning other than trivially combining predictions from two models. 

**Performance gap between benchmarks.** The performance improvement on HAKE [26]-PaSta is relatively smaller compared with other benchmarks. One possible explanation is the conclusion in HAKE [26]-PaSta is PaSta, which is more with more ambiguity compared with the activity class, _e_ . _g_ ., “hold something” vs. “ride a boat”. The ambiguity of multi-choice answers hinders reasoning, but it is caused by the inherent ambiguity of the PaSta definition instead of the proposed reasoning method. 

19 

However, despite the ambiguity, HAKE [26]-PaSta is still valuable in verifying the effectiveness of reasoning for finer-grained activity understanding tasks. 

**Failure Cases.** We provide two failure cases (false negative) for the activity "human board an airplane" in Fig. 11. In image I, the failure is caused by the prediction error of symbol probabilities. Some symbols are predicted as low probabilities by mistake, _e_ . _g_ ., "walk towards the boarding gate". Tiny human bounding boxes possibly cause the symbol prediction error. In image II, the failure emerges because the generated rules do not cover the situation in the image, _i_ . _e_ ., errors of the symbolic system. For example, the image contains the symbol "climb up the airplane edge" which is not in the symbolic system. 

## **D.5 Computation Cost** 

Symbol predictions will increase the computational cost as a trade-off for explainability and generalization. In this paper, we mainly focus on the effectiveness and implementation of our insight and do not preferentially consider efficiency. It can be eased by discovering the hierarchical and reusable nature of the symbols. We show some initial exploration below. 

**Hierarchical structure.** Not all symbols need to be predicted via pruning. Symbols can be organized as a tree with hierarchical prediction. For the Fig.6 example in the main text, _m_ 1... _m_ 10 can be organized as: 

- _ma_ : Interact with a seller: 

   - _m_ 1:Talk with seller 

   - _m_ 3: Seller hand over orange 

   - _m_ 10: Give money to seller 

- _mb_ : Interact with the oranges: 

   - _m_ 2: Reach for an orange 

- _mc_ : Interact with a container: 

   - _m_ 5: Place orange in a bag 

   - _m_ 6: Pick orange from a basket 

   - _m_ 7: Hold a bag of oranges 

   - _m_ 9: Seller put the orange in bag 

- _md_ : Payment Process: 

   - _m_ 4: Stand in front of fruit stand 

   - _m_ 8: Reach for a wallet 

The father symbols (e.g., _ma_ ) can be summarized based on the son symbols (e.g., _m_ 1) via simple semantic extraction ( _e_ . _g_ ., LLM prompt). On the symbol tree, the probability of a son symbol is no higher than its father symbol. When the probability of a father symbol is lower than a threshold (e.g., in image III, _ma_ probability less than 0.1), its son symbols ( _m_ 1, _m_ 3, _m_ 10) will be assigned the threshold directly without extra computation. Thus, the computational cost is saved via the hierarchical structure of symbols. For example, image III only needs 5 instead of 10 calculations: Interact with a seller(x), Interact with the oranges, Reach for an orange, Interact with a container(x), and Payment Process(x). The computation save will be more evident with the number of symbols scales. 

**Reusable symbols.** The total number of symbols to measure will increase with the number of potential activity classes, but not scale linearly. It is unavoidable to measure probabilities of more than one activity class for activity understanding tasks. This is because activity recognition is a multi-label classification rather than a multi-class classification: A person typically performs multiaction simultaneously, _e_ . _g_ ., standing while eating. Thus, the increase in symbol measurement comes from the inherent characteristics of activity tasks. However, the symbols are reused for different activity classes without repetitive computation. For example, "buy orange" and "visit store" shares the symbols: ( _ma_ ) Interact with a seller, ( _md_ ) Payment process, ( _m_ 1) Talk with the seller, ( _m_ 10) Give money to the seller, ( _m_ 8) Reach for a wallet. It is also illustrated in the main paper Fig.2(a): the red and green activities share some symbols (gray), _e_ . _g_ ., second from right, third from left. 

20 

**Quantitative comparison.** We provide a detailed analysis of a small case. There are 38 images known to contain oranges, where verbs (buy/cut/eat/hold/inspect/peel/pick/squeeze/wash) are to be predicted. We report the number of operations average per image on it. Without reasoning, activity classes are predicted directly, and the number of operations varies from images due to different ground-truth object labels which verb prediction conditions upon. The average number of operations is 9. With reasoning, the average number of operations is 71. However, with reusable symbols, the number is reduced to 31, and adding a hierarchical structure further reduces it to 23. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning_images/nips_2023_Symbol-LLM_Leverage_Language_Models_for_Symbolic_System_in_Visual_Human_Activity_Reasoning.pdf-0021-01.png)


## **What are person on the right doing?** 

a) person on the left is taking person on the right home. **b) person on the right are on a first date.** 

- c) person on the left is conducting a job interview with person on the right. 

d) They are picking something up. 

Figure 12: An example QA in VCR dataset. 

## **E Boarder Application** 

The method could be more general and facilitate real-world applications. We choose activity understanding as a good and important initial test bed because it is a difficult task with complex visual patterns and a compositional nature. The proposed symbolic system can be generalized to various tasks ( _e_ . _g_ ., classification, visual commonsense) with a similar formulation. Semantic coverage and logical entailment remain two important properties under a more general setting, and we can reason with visual inputs via a similar pipeline. We show two initial experiments below. 

**Object classification.** One example rule for an object ( _e_ . _g_ ., crocodile) is: long, slightly curved body _∧_ four short legs _∧_ ... _∧_ a long, muscular tail _→_ crocodile. We conduct experiments on CIFAR-100 [22] test set with a zero-shot setting. The performance is 77.50% accuracy for CLIP and 78.31% accuracy for CLIP+Reasoning. 

**Visual commonsense.** One example rule for the image and question in Fig. 12 is: person on the right shows affectionate gestures _∧_ person on the left engages in conversation _∧_ ... _∧_ The environment is a restaurant _→_ b. We conduct experiments on VCR [47] val set. BLIP2 is adopted as a baseline as it is a VQA-style task. VCR val set has 26534 image-question pairs, where we select a subset of 557 pairs for the test. The selected pairs are highly related to human activities (instead of role, mental, _.etc_ ) and only involve _one_ person’s activity. The performance is 46.32% accuracy for BLIP2 and 47.08% accuracy for BLIP2+Reasoning. 

We can see that the proposed pipeline can be applied to general visual tasks and facilitate reasoning. In this paper, we focus on method design instead of task generalization. We will analyze more general settings in future work. 

## **References** 

- [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, et al. Do as i can, not as i say: Grounding language in robotic affordances. _arXiv preprint arXiv:2204.01691_ , 2022. 

- [2] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In _CVPR_ , 2016. 

- [3] Claude Berge. _Hypergraphs: combinatorics of finite sets_ , volume 45. Elsevier, 1984. 

- [4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In _CVPR_ , 2017. 

- [5] Yu Wei Chao, Zhan Wang, Yugeng He, Jiaxuan Wang, and Jia Deng. Hico: A benchmark for recognizing human-object interactions in images. In _ICCV_ , 2015. 

- [6] Prithiviraj Damodaran. Parrot: Paraphrase generation for nlu., 2021. 

21 

- [7] Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou. Neural logic machines. _arXiv preprint arXiv:1904.11694_ , 2019. 

- [8] Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. In _CVPR_ , 2015. 

- [9] Richard Evans, David Saxton, David Amos, Pushmeet Kohli, and Edward Grefenstette. Can neural networks understand logical entailment? _arXiv preprint arXiv:1802.08535_ , 2018. 

- [10] Hao Shu Fang, Jinkun Cao, Yu Wing Tai, and Cewu Lu. Pairwise body-part attention for recognizing human-object interactions. In _ECCV_ , 2018. 

- [11] Hao-Shu Fang, Jiefeng Li, Hongyang Tang, Chao Xu, Haoyi Zhu, Yuliang Xiu, Yong-Lu Li, and Cewu Lu. Alphapose: Whole-body regional multi-person pose estimation and tracking in real-time. _TPAMI_ , 2023. 

- [12] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In _ICCV_ , 2019. 

- [13] Daan Fierens, Guy Van den Broeck, Joris Renkens, Dimitar Shterionov, Bernd Gutmann, Ingo Thon, Gerda Janssens, and Luc De Raedt. Inference and learning in probabilistic logic programs using weighted boolean formulas. _Theory and Practice of Logic Programming_ , 15(3):358–401, 2015. 

- [14] Giorgio Gallo, Giustino Longo, Stefano Pallottino, and Sang Nguyen. Directed hypergraphs and applications. _Discrete applied mathematics_ , 42(2-3):177–201, 1993. 

- [15] Artur d’Avila Garcez, Marco Gori, Luis C Lamb, Luciano Serafini, Michael Spranger, and Son N Tran. Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. _arXiv preprint arXiv:1905.06088_ , 2019. 

- [16] Rohit Girdhar and Deva Ramanan. Attentional pooling for action recognition. _NeurIPS_ , 2017. 

- [17] Georgia Gkioxari, Ross Girshick, and Jitendra Malik. Contextual action recognition with r* cnn. In _ICCV_ , 2015. 

- [18] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In _CVPR_ , 2018. 

- [19] Saurabh Gupta and Jitendra Malik. Visual semantic role labeling. _arXiv preprint arXiv:1505.04474_ , 2015. 

- [20] Drew Hudson and Christopher D Manning. Learning by abstraction: The neural state machine. _NeurIPS_ , 2019. 

- [21] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _ICML_ , 2021. 

- [22] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 

- [23] Sumith Kulal, Jiayuan Mao, Alex Aiken, and Jiajun Wu. Hierarchical motion understanding via motion programs. In _CVPR_ , 2021. 

- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 

- [25] Yong-Lu Li, Xinpeng Liu, Xiaoqian Wu, Xijie Huang, Liang Xu, and Cewu Lu. Transferable interactiveness knowledge for human-object interaction detection. _TPAMI_ , 2022. 

- [26] Yong-Lu Li, Xinpeng Liu, Xiaoqian Wu, Yizhuo Li, Zuoyu Qiu, Liang Xu, Yue Xu, Hao-Shu Fang, and Cewu Lu. Hake: a knowledge engine foundation for human activity understanding. _TPAMI_ , 2022. 

- [27] Yong-Lu Li, Liang Xu, Xinpeng Liu, Xijie Huang, Yue Xu, Mingyang Chen, Ze Ma, Shiyi Wang, Hao-Shu Fang, and Cewu Lu. Hake: Human activity knowledge engine. _arXiv preprint arXiv:1904.06539_ , 2019. 

- [28] Yong-Lu Li, Liang Xu, Xinpeng Liu, Xijie Huang, Yue Xu, Shiyi Wang, Hao-Shu Fang, Ze Ma, Mingyang Chen, and Cewu Lu. Pastanet: Toward human activity knowledge engine. In _CVPR_ , 2020. 

- [29] Ji Lin, Chuang Gan, and Song Han. Tsm: Temporal shift module for efficient video understanding. In _ICCV_ , 2019. 

22 

- [30] Tsung Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In _ECCV_ , 2014. 

- [31] Yunchao Liu and Zheng Wu. Learning to describe scenes with programs. In _ICLR_ , 2019. 

- [32] Cewu Lu, Hao Su, Yonglu Li, Yongyi Lu, Li Yi, Chi-Keung Tang, and Leonidas J Guibas. Beyond holistic object recognition: Enriching image understanding with part states. In _CVPR_ , 2018. 

- [33] Xiaojian Ma, Weili Nie, Zhiding Yu, Huaizu Jiang, Chaowei Xiao, Yuke Zhu, Song-Chun Zhu, and Anima Anandkumar. Relvit: Concept-guided vision transformer for visual relational reasoning. _arXiv preprint arXiv:2204.11167_ , 2022. 

- [34] Arun Mallya and Svetlana Lazebnik. Learning models for actions and person-object interactions with transfer to question answering. In _ECCV_ , 2016. 

- [35] Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. Krisp: Integrating implicit and symbolic knowledge for open-domain knowledge-based vqa. In _CVPR_ , 2021. 

- [36] Maxwell Nye, Michael Tessler, Josh Tenenbaum, and Brenden M Lake. Improving coherence and consistency in neural sequence models with dual-system, neuro-symbolic reasoning. _NeurIPS_ , 2021. 

- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- [38] Luc De Raedt, Kristian Kersting, Sriraam Natarajan, and David Poole. Statistical relational artificial intelligence: Logic, probability, and computation. _Synthesis lectures on artificial intelligence and machine learning_ , 10(2):1–189, 2016. 

- [39] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In _NeurIPS_ , 2015. 

- [40] Botian Shi, Lei Ji, Pan Lu, Zhendong Niu, and Nan Duan. Knowledge aware semantic concept expansion for image-text matching. In _IJCAI_ , volume 1, page 2, 2019. 

- [41] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In _CVPR_ , 2022. 

- [42] Raviteja Vemulapalli, Felipe Arrate, and Rama Chellappa. Human action recognition by representing 3d skeletons as points in a lie group. In _CVPR_ , 2014. 

- [43] Yue Xu, Yong-Lu Li, Jiefeng Li, and Cewu Lu. Constructing balance from imbalance for long-tailed image recognition. In _ECCV_ , 2022. 

- [44] Bangpeng Yao, Xiaoye Jiang, Aditya Khosla, Andy Lai Lin, Leonidas Guibas, and Li Fei-Fei. Human action recognition by learning bases of action attributes and parts. In _ICCV_ , 2011. 

- [45] Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neuralsymbolic vqa: Disentangling reasoning from vision and language understanding. _NeurIPS_ , 2018. 

- [46] L Zadeh. Fuzzy sets. _Inform Control_ , 8:338–353, 1965. 

- [47] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _CVPR_ , 2019. 

- [48] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. _arXiv preprint arXiv:2304.00685_ , 2023. 

23 

