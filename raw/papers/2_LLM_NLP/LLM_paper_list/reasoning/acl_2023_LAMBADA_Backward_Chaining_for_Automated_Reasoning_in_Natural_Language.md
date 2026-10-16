## **LAMBADA: Backward Chaining for Automated Reasoning in Natural Language** 

**Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, Deepak Ramachandran** Google Research 

{mehrankazemi, njkim, bhatiad, xxujasime, ramachandrand}@google.com 

## **Abstract** 

Remarkable progress has been made on automated reasoning with natural text, by using Language Models (LMs) and methods such as Chain-of-Thought and Selection-Inference. These techniques search for proofs in the forward direction from axioms to the conclusion, which suffers from a combinatorial explosion of the search space, and thus high failure rates for problems requiring longer chains of reasoning. The classical automated reasoning literature has shown that reasoning in the backward direction (i.e. from the intended conclusion to supporting axioms) is significantly more efficient at proof-finding. Importing this intuition into the LM setting, we develop a _Backward Chaining_ algorithm, called LAMBADA, that decomposes reasoning into four sub-modules. These sub-modules are simply implemented by few-shot prompted LM inference. We show that LAMBADA achieves sizable accuracy boosts over state-of-the-art forward reasoning methods on challenging logical reasoning datasets, particularly when deep and accurate proof chains are required. 

## **1 Introduction** 

Automated reasoning, the ability to draw valid conclusions from explicitly provided knowledge, has been a fundamental goal for AI since its early days (McCarthy, 1959; Hewitt, 1969). Furthermore, logical reasoning, especially reasoning with unstructured, natural text is an important building block for automated knowledge discovery and holds the key for future advances across various scientific domains. While in recent years tremendous progress has been made towards natural language understanding thanks to pretrained language models (LMs) (Brown et al., 2020; Chowdhery et al., 2022, _i.a.,_ ), the performance of these models for logical reasoning still lags behind (Rae et al., 2021; Creswell et al., 2023; Valmeekam et al., 2022) compared to the advancements in other areas such as reading comprehension and question-answering. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0001-08.png)


**----- Start of picture text -----**<br>
Facts: Eric is nice<br>1. Rough and cold that is what they<br>say about Blue Bob.<br>2. Eric, who is relatively young, is also Rule Selection<br>pretty big and tends to be cold. Rule6<br>3.4. Fred is green and cold too.For being so cold, it's good Harrycan remain nice. Fact CheckEric is big DecomposeEric is red Eric is youngCache<br>Rules: Rule Selection<br>1. Rough, cold people are blue. Rule3<br>2.3. Big, kind folks are green ones.If a person is big, rough, and cold, Eric is big Decompose Eric is cold<br>they are also red. Cache Eric is rough Cache<br>4. Most round and cold people are Rule Selection<br>often rough. Rule4 Rule5<br>5. Cold, young people are also certain<br>to be rough people. Decompose Decompose<br>6. An individual who is big, red andyoung is also a nice individual. Eric is round Eric is cold Eric is cold Eric is young<br>Goal: Rule Selection Pruned Fact Check Fact Check<br>● Eric is nice?<br>Label<br>● Proved<br>**----- End of picture text -----**<br>


Figure 1: The search trace of LAMBADA on an example from the ParaRules subset of ProofWriter (the _Sign Agreement_ and failed _Fact Check_ modules are omitted for brevity). 

While many problems benefit from LM scaling, scaling has been observed to provide limited benefit for solving complex reasoning problems. For example, Creswell et al. (2023) observed that for the Gopher family of LMs (Rae et al., 2021), the benefit of scaling for logic-based tasks is significantly worse than for other language tasks. Moreover, while finetuning initially seemed to enable logical reasoning in LMs (Clark et al., 2021; Tafjord et al., 2021), further exploration revealed that finetuned LMs mostly exploit spurious correlations (e.g., the correlation between the number of rules and the label) as opposed to learning to reason (Zhang et al., 2022b; Schlegel et al., 2022; Liu et al., 2023). Recently, prompting strategies such as Chain-ofThought (Wei et al., 2022) and Scratchpad (Nye et al., 2022) have contributed to improving performance of LMs on reasoning tasks, although they have been also shown to struggle with proof planning for more complex logical reasoning problems (Saparov and He, 2023). 

One solution to the aforementioned problems is to integrate the strength and reliability of classical AI models in logical reasoning with LMs (Garcez and Lamb, 2020; Marcus, 2020). In the literature, 

there are two major approaches to logical reasoning (Poole and Mackworth, 2010): 

1. _Forward Chaining (FC)_ where one starts from the facts and rules (“theory”), and iterates between making new inferences and adding them to the theory until the goal statement can be proved or disproved, 

2. _Backward Chaining (BC)_ where one starts from the goal and uses the rules to recursively decompose it into sub-goals until the sub-goals can be proved or disproved based on the theory. 

Previous approaches to reasoning with LMs mostly incorporate elements of FC into LMs (Tafjord et al., 2021; Creswell et al., 2023). FC requires selecting a subset of facts and rules from the entire set, which might be difficult for an LM as it requires a combinatorial search over a large space. Moreover, deciding when to halt and declare failure to prove is challenging in FC, as also noted by Creswell et al. (2023), sometimes requiring specialized modules trained on intermediate labels (Creswell and Shanahan, 2022). Indeed, the classical automated reasoning literature is heavily weighted towards BC or goal-directed strategies for proof-finding. 

In this paper, we show experimentally that BC is better suited for text-based deductive logical reasoning, as it does not require a combinatorial search for subset selection and there are more natural halting criteria for it. We develop a hybrid **LA** nguage **M** odel augmented **BA** ckwar **D** ch **A** ining technique (LAMBADA), where BC drives the high-level proof planning, and the LM performs the textual understanding and individual reasoning steps. We conduct experiments with challenging datasets for LM reasoning containing examples expressed in naturalistic text. The datasets contain proof chains of up to 5 hops in depth, and examples where the goal can neither be proved nor disproved from the provided theory. We show that LAMBADA achieves substantially higher deductive accuracy, and is considerably more likely to generate valid reasoning chains compared to other techniques which find correct conclusions with spurious proof traces, while also being more query efficient than other LMbased modular reasoning approaches. Our results strongly indicate that future work on reasoning with LMs should incorporate backward chaining or goal-directed planning strategies. 

## **2 Related Work** 

The deep learning based models that have been developed to solve text-based (logical) reasoning tasks can be categorized as follows (see Huang and Chang 2022 for a recent survey of the literature). 

**Pretraining on Relevant Tasks:** Pretraining an LM on corpora relevant to the target reasoning task can lead to improvements (Hendrycks et al., 2021; Shen et al., 2021). Pretraining is, however, costly especially for larger LMs. 

**Implicit Reasoning:** These approaches finetune LMs to produce the label directly given the input (Clark et al., 2021; Betz et al., 2021; Saeed et al., 2021; Han et al., 2022); reasoning is expected to happen implicitly in the parameters of the LM. It has been shown that finetuning LMs on logical reasoning tasks makes them learn spurious correlations (Zhang et al., 2022b; Schlegel et al., 2022), and is not robust to multi-hop reasoning (Kassner et al., 2020). Besides, finetuning large LMs is costly especially when the dataset is large, and may introduce distributional shocks to the model (Kazemi et al., 2023). In this paper, we focus on models that only take in-context examples as supervision. 

**Explicit Reasoning:** Generating the intermediate reasoning steps such as the chain of reasoning (Wei et al., 2022; Nye et al., 2022; Dalvi et al., 2021; Zelikman et al., 2022; Zhang et al., 2022a) has shown substantial improvement for many reasoning tasks (Suzgun et al., 2022). Such chains have been explored both in the forward and the backward directions, e.g., using multiple constrained LMs for logical reasoning (Zhang et al., 2022a). Gontier et al. (2020) investigated how transformer models perform when trained to perform forward or backward chaining, and drew conclusions about their internal reasoning strategies. We compare against a popular recent prompting strategy, namely Chain-of-Thought (CoT) (Wei et al., 2022), from this category. 

**Verifiers:** To improve CoT, some works train a verifier using chain-level labels. The verifier takes a reasoning chain produced by the model as input and judges the quality of the chain (Cobbe et al., 2021; Shen et al., 2021; Jhamtani and Clark, 2020; Zelikman et al., 2022). Using this verifier, one can then generate multiple reasoning chains (e.g., by running the algorithm multiple times with different decoding temperatures) and use the best chain according to the verifier. Since LAMBADA also 

generates proofs, verifiers are also applicable to our algorithm. In this paper, we assume not having access to chain-level labels, and leave experiments with verifiers as future work. 

**Length generalization:** A number of approaches specifically look into whether LMs can generalize from examples requiring shorter reasoning chains (shown to them either as demonstration or as finetuning data) to examples requiring longer chains (Anil et al., 2022; Tafjord et al., 2021). With our model, length generalization comes for free because the model learns the building blocks of solving the problem that are applied as many times as needed to solve the problem. 

**Modular Reasoning:** These approaches break the problem into smaller modules and use separate LMs to solve each module (Zhou et al., 2022; Khot et al., 2023; Sprague et al., 2022; Zhou et al., 2023; Dua et al., 2022; Wang et al., 2022; Schlag et al., 2023). LM-based approaches to logical reasoning typically makes use of a single LM module; for example, in Tafjord et al. (2021), a single LM module iteratively and exhaustively infers _all_ conclusions based on the facts and rules, and then the goal statement is compared against the final set of conclusions to confirm if it can be proved from the theory. Since exhaustively deriving all conclusions is computationally expensive, Creswell et al. (2023) consider a more scalable approach where the conclusions that are derived are informed by the goal; they iteratively apply two LLM modules one selecting a subset of the facts and rules informed by the goal and the other making new inferences based on the selected facts and rules and adding it back to the theory. In this paper, we compare against the second approach. 

**Natural Language Inference (NLI):** Logical reasoning can also be understood as identifying whether a logical entailment relation holds between two propositions (premise and hypothesis; the premise is the theory and the hypothesis is the statement to be proved). In this sense, NLI models are also relevant, although inferences under NLI typically adopt a more relaxed notion of entailment rather than purely logical (Dagan et al., 2013; Bowman et al., 2015; Williams et al., 2018). 

## **3 LAMBADA: Language Model Augmented Backward Chaining** 

We focus on performing automated reasoning over _facts_ , i.e., natural language assertions such as 

“Nice people are red”, that are coherent but not necessarily grounded in reality. A _rule_ is a natural language statement that is either of the form, or can be rewritten in the form, “If P then Q”; e.g., “Rough, cold people are blue” can be rewritten as “If a person is rough and cold, then they are blue”. P is called the _antecedent_ and Q is called the _consequent_ of the rule. A _theory C_ consists of facts _F_ = _{f_ 1 _, f_ 2 _, . . . , fn}_ and rules _R_ = _{r_ 1 _, r_ 2 _, . . . , rm}_ . We let _G_ represent a _goal_ that we would like to prove or disprove based on the theory. An example theory with fictional characters and rules is demonstrated in Figure 1. Based on the theory, one should prove or disprove the goal “Eric is nice”. 

## **3.1 Backward Chaining** 

Backward chaining (BC) is a strategy for reasoning that starts from the goal and recursively breaks the goal into sub-goals based on the rules that can be applied to it, until the sub-goals can be proved or disproved based on the facts or no more rules can be applied to break down the sub-goal further. 

Figure 1 shows an example of BC applied to a theory to prove a goal. Initially, BC verifies if the goal can be proved or disproved based on the facts (this step is omitted from the figure). Since none of the facts directly prove or disprove the goal, BC next selects a rule that can be applied to break down the goal into sub-goals. Whether or not a rule applies to a goal is determined by an operation called _unification_ in logic; Rule6 has the same consequent as the goal so the operation can be applied, but the other rules have different consequents and it cannot be applied. Using Rule6, the goal can be broken down into three sub-goals that should be proved for the goal to be proved. BC then makes recursive calls to prove each sub-goal. The algorithm continues until either a halting criterion is reached (e.g., reaching a certain depth in search), or a sub-goal can no longer be broken down (e.g., the left sub-tree under “Eric is rough”), or all sub-goals are proved (e.g., the right sub-tree under “Eric is rough”). 

The outcome of BC for a goal is either PROVED, DISPROVED, or UNKNOWN; e.g., its output for the goal in Figure 1 is PROVED, for “Fred is not green?” is DISPROVED (because it contradicts Fact3), and for “Fred is round?” is UNKNOWN (because the theory does not entail or contradict it). 

## **3.2 LM Modules in LAMBADA** 

To enable applying BC for text-based reasoning, we introduce four LM-based modules: _Fact Check_ , _Rule Selection_ , _Goal Decomposition_ , and _Sign Agreement_ , each implemented by showing relevant in-context demonstrations to a pretrained LM (see Appendix D.3 for details). We describe these modules and then proceed to the full algorithm. 

## **3.2.1 Fact Check** 

Given a set of facts _F_ from the theory and a goal _G_ , the _Fact Check_ module verifies if there exists a fact _f ∈F_ such that _f_ entails _G_ (in which case the goal is proved) or _f_ entails the negation of _G_ (in which case the goal is disproved). If no such fact can be found, then the truth of _G_ remains unknown. 

We implement _Fact Check_ with two submodules: the first sub-module selects a fact from the set of facts that is most relevant to the goal, and the second sub-module verifies if the goal can be proved or disproved based on that fact.[1] Since the first sub-module may fail to identify the best fact on the first try, if the truth of the goal remained unknown after one try, the selected fact can be removed and the sub-modules can be called again. This process can be repeated multiple times. In our experiments, we call the two sub-modules twice. 

## **3.2.2 Rule Selection** 

Given a set of rules _R_ from the theory and a goal _G_ , the _Rule Selection_ module identifies the rules _r ∈R_ such that the consequent of _r_ unifies with _G_ . These rules are then used for decomposing the goal into sub-goals. If no such rule can be identified, then the truth of _G_ remains unknown. 

As we did for _Fact Check_ , we implement _Rule Selection_ with two sub-modules: the first sub-module identifies the consequent of each rule (independent of the goal), and the second sub-module takes the rule consequents and the goal as input and identifies which one unifies with the goal. Note that due to the recursive nature of BC, the _Rule Selection_ module may be invoked multiple times during the proof of a goal. Since identifying the consequent of each rule is independent of the goal, this sub-module only needs to be called once. 

> 1Note that we select only one fact because the goals and sub-goals in the datasets we work with can be proved/disproved using single facts; The two modules can be adapted to selected multiple facts if this is not the case. 

## **Algorithm 1** LAMBADA 

|**Input:** Theory_C_ = (_F, R_), Goal_G_, Max-Depth|
|---|
|D|
|1: factCheckResult =_FactCheck_(_G_,_F_)|
|2: **if**factCheckResult =UNKNOWN**then**|
|3:<br>**return**factCheckResult|
|4: **if**D == 0**then**|
|5:<br>**return**UNKNOWN|
|6: _Rs_=_RuleSelection_(_G_,_R_)|
|7: **for**_r ∈_Rerank(_Rs_)**do**|
|8:<br>**G**=_GoalDecomposition_(_r, G_)|
|9:<br>**if**ProveSubgoals(_C_,**G**, D)**then**|
|10:<br>**if**_SignAgreement_(r,_G_)**then**|
|11:<br>**return**PROVED|
|12:<br>**else**|
|13:<br>**return**DISPROVED|
|14: **return**UNKNOWN|



## **3.2.3 Goal Decomposition** 

Given a rule _r_ and a goal _G_ such that the consequent of _r_ unifies with _G_ , the _Goal Decomposition_ module identifies the sub-goals that need to be proved in order for _G_ to be proved or disproved. The subgoals are identified based on the antecedent of _r_ . 

## **3.2.4 Sign Agreement** 

In the case where we succeed in proving the antecedent of _r_ , whether the goal is proved or disproved depends on whether the sign of the goal agrees or disagrees with the sign of the consequent of _r_ . For instance, in Figure 1, for the goal “Eric is nice.”, since the sign of the goal agrees with the sign of the consequent of Rule6 and the antecedent of the rule is proved, we conclude that the goal is proved. However, if Rule6 was “[...] is not going to be a nice individual.”, then the sign of the goal would disagree with the sign of the consequent and so we would conclude that the goal is disproved. This motivates the fourth module, _Sign Agreement_ , described below. 

Given a rule _r_ and a goal _G_ , the _Sign Agreement_ module verifies if the sign of the consequent of _r_ agrees or disagrees with the sign of the goal or not. 

## **3.3 The LAMBADA Algorithm** 

Algorithm 1 provides a high-level description of how the four LM modules described earlier can be integrated with BC to enable text-based logical reasoning (the function calls corresponding to LM modules are color-coded). 

LAMBADA can be understood as a depth-first 

## **Algorithm 2** ProveSubgoals 

**Input:** Theory _C_ = ( _F, R_ ), Sub-Goals **G** , MaxDepth D 

- 1: **for** _G_ in **G do** 

- 2: result = LAMBADA( _C_ , _G_ , D-1) 3: **if** result _̸_ = PROVED **then** 4: **return** False _# Assuming conjunction_ 5: **return** True 

search algorithm over the facts and the rules. It takes as input a theory _C_ = ( _F, R_ ), a goal _G_ , and a depth _D_ that defines a halting criterion for the algorithm based on the maximum allowed depth for the search. The search depth is a natural halting criterion corresponding to the maximum number of reasoning hops required for answering questions. 

Initially, the algorithm uses the _Fact Check_ module to check if _G_ can be proved or disproved using the facts. If this is the case, then the algorithm stops and returns the result (PROVED or DISPROVED). 

If _G_ cannot be proved or disproved, then the algorithm checks the depth _D_ : if _D_ = 0, then the algorithm stops and returns UNKNOWN indicating that _G_ could not be proved or disproved. Otherwise, the algorithm proceeds with applying rules. 

The _Rule Selection_ module is used to identify the rules _Rs_ from _R_ whose consequent unifies with _G_ . Once the set _Rs_ is identified, if LAMBADA can start with the rules that have a higher chance of succeeding at (dis)proving the goal, it can save computations and be less error-prone. Therefore, we include a _Rerank_ function in LAMBADA. Based on the intuition that shorter rules are likely to have fewer sub-goals (hence a higher chance of success), we start the search from shorter rules and proceed to longer rules if the shorter ones fail. We leave more sophisticated ranking strategies as future work. 

For each selected rule, the algorithm uses the _Goal Decomposition_ module to decompose _G_ into a set of sub-goals **G** that need to be proved and checks whether those sub-goals can be proved by making recursive calls to the algorithm (with reduced depth). If the sub-goals can be proved, then the algorithm uses the _Sign Agreement_ module to check whether the sign of the rule consequent agrees or disagrees with the sign of _G_ . If it does, then the algorithm returns PROVED and otherwise DISPROVED. If there is no rule for which the subgoals can be proved, then UNKNOWN is returned. 

During a proof, LAMBADA may be called multiple times with the same theory and goal; in Ap- 

pendix A we explain how cycles and redundant computations can be avoided using a cache. 

## **4 Experimental Setup** 

We describe our baselines and datasets here, and provide further implementation details in Appendix D. Unless stated otherwise, all experiments are based on the PaLM 540B model (Chowdhery et al., 2022). 

## **4.1 Baselines** 

We compare against the following two baselines. 

**Chain-of-Thought (CoT)** (Wei et al., 2022) is a popular neural approach based on demonstrating chains of inference to the LM within the in-context prompt. In addition to the few-shot demonstrations in <INPUT>/<LABEL> format in typical in-context learning settings, in CoT, an intermediate explanation for the label is also provided (<INPUT>/<EXPLANATION>/<LABEL>). In our work, the explanation corresponds to the proof. 

**Selection-Inference (SI)** (Creswell et al., 2023) is a strong modular reasoning approach based on forward chaining. SI contains two modules: (1) _selection_ , which, guided by the goal, selects a subset of the facts and rules from which new conclusions can be derived toward proving the goal, and (2) _inference_ , which takes the selected facts and rules and derives a new conclusion. The two modules are called iteratively, each time producing a single conclusion that is added back to the theory before the next iteration. The iterations continue until a halting criterion is met (a fixed number of steps in Creswell et al. 2023). 

## **4.2 Datasets** 

We experiment with challenging deductive logical reasoning datasets outlined below. 

**ProofWriter** (Tafjord et al., 2021) is a commonly used synthetic dataset for testing logical reasoning when facts and rules are expressed in naturalistic text. It contains two subsets: an open-world assumption (OWA) subset and a closed-world assumption (CWA) subset. In this paper, we use the OWA subset. Each example is a ( _theory, goal_ ) pair and the label is one of _{_ PROVED, DISPROVED, UNKNOWN _}_ where UNKNOWN indicates that the goal can neither be proved nor disproved. The dataset has five parts, each part requiring 0, _≤_ 1, _≤_ 2, _≤_ 3 and _≤_ 5 hops of reasoning, respectively. We report two sets of results on this dataset: (1) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0006-00.png)


**----- Start of picture text -----**<br>
1.0 1.0<br>0.98 0.99 1.0<br>0.8 0.770.97 0.9 0.87 0.82 0.72 0.8 0.98 0.92 0.93 0.77 0.91 0.8 0.87 0.78 0.81<br>0.60.4 Majority Class0.590.61 0.580.56 0.54 0.51 0.50.46 0.60.4 Majority Class0.69 0.65 0.59 0.54<br>CoT CoT<br>0.2 SI 0.2 SI<br>Lambada Lambada<br>0.0 0.0<br>Depth-0 Depth-1 Depth-2 Depth-3 Depth-5 Depth-0 Depth-1 Depth-2 Depth-3 Depth-5<br>(a) ProofWriter-PUD (b) ProofWriter-PD<br>1.0 0.8 0.86 0.94<br>0.97 0.98 0.99 0.96 0.8<br>0.9<br>0.8<br>0.76 0.7 0.6 0.6 0.6<br>0.6<br>0.4 0.52 0.45 0.4 0.4<br>Majority Class<br>0.2 CoTSI 0.2 0.2 0.28<br>Lambada<br>0.0 0.0 0.0<br>Depth-1 Depth-3 Depth-5 CoT Lambada CoT Lambada<br>(c) PrOntoQA (d) ParaRules (e) ProofWr. (d5)<br>Accuracy Accuracy<br>Accuracy Accuracy<br>Proof Accuracy<br>**----- End of picture text -----**<br>


Figure 2: Prediction accuracy results on (a) ProofWriter-PUD (b) ProofWriter-PD, (c) PrOntoQA, and (d) ParaRules datasets. (e) The proof accuracy of CoT and LAMBADA on ProofWriter (Depth-5) for a set of randomly sampled examples for which the models correctly predicted if the goal can be proved or disproved. 

with examples labeled UNKNOWN removed (for compatibility with previous work), and (2) with all three labels. Note that intermediate proof chains from ProofWriter are not used by our models in making predictions. For both cases, due to the cost of inference, we used the first 1000 examples in the test set. Hereafter, we refer to these two subsets as _ProofWriter-PD_ and _ProofWriter-PUD_ . 

**PrOntoQA** (Saparov and He, 2023) is a synthetic dataset created to analyze the capacity of LM-based approaches for logical reasoning. Compared to ProofWriter, PrOntoQA has lower natural language diversity and less l fact/rule variations (e.g., no conjunctions). However, the search traces typically contain multiple paths with only one of them leading to the proof, thus enabling testing the proof planning of different models. This dataset has multiple versions; we use the _fictional characters_ version, which is one of the hardest versions according to Saparov and He (2023). Similarly to ProofWriter, each version of PrOntoQA is divided into different parts depending on the depth of reasoning chains required (1, 3, and 5 hops). 

**ParaRules** (Tafjord et al., 2021) is a version of ProofWriter where the synthetically generated sentences in the theory are rewritten by crowdworkers to increase diversity and naturalness of the text. This lets us move beyond evaluating reasoning with templatic expressions, which is a key limitation of the other datasets. Each fact in ParaRules may be 

a combination of several sub-facts (see Fig. 1 for an example). The examples require proof depths of up to 5 and the label can be PROVED, DISPROVED, or UNKNOWN. We found some minor quality issues in ParaRules; we manually verified and fixed the first 500 examples of the test set (see Appendix D.2) and used this set for evaluation. 

## **5 Results** 

We now describe the results and compare LAMBADA and the baselines in detail. 

## **5.1 Label Prediction Accuracy** 

The results are reported in Figure 2, (a)–(d).[2] LAMBADA significantly outperforms the baselines, especially on ProofWriter-PUD which contains UNKNOWN labels (44% relative improvement compared to CoT and 56% compared to SI on Depth-5), the higher depths of PrOntoQA (37% relative improvement compared to CoT and 113% compared to SI on Depth-5), and the ParaRules dataset (43% relative improvement compared to CoT). These results overall show the merit of LAMBADA for logical reasoning. We highlight that the reasoning capacity of LAMBADA robustly generalizes to more naturalistic expressions, as demonstrated by the high accuracy on ParaRules, which is exactly 

> 2Due to the low performance of SI on ProofWriter and PrOntoQA and its high number of LM calls (see Figure 7), we only compared LAMBADA against CoT for ParaRules. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0007-00.png)


**----- Start of picture text -----**<br>
0.53<br>0.50 0.47<br>0.45<br>0.40<br>0.34<br>0.35<br>0.31 0.31<br>0.30<br>1st 2nd 3rd 4th 5th<br>k-th inference<br>Success Rate<br>**----- End of picture text -----**<br>


Figure 3: The success rate of the _k_ -th inference of SI on PrOntoQA (Depth-5) for different values of _k_ . As _k_ increases, the size of the input theory becomes larger and the success rate decreases. 

the desired outcome of combining the strengths of an LM and a symbolic reasoning algorithm. 

The results in Figure 2(a) reveal a shortcoming of the CoT approach in dealing with UNKNOWN labels. That is, unlike the examples for which the label is PROVED or DISPROVED, there is no natural chain of thought for the examples whose labels are UNKNOWN. Nevertheless, the performance of CoT is competitive for the ProofWriter-PD dataset, and the accuracy does not diminish substantially with increasing depth. We investigate the reason for this behaviour of CoT in the next section. 

## **5.2 Proof Accuracy** 

To understand the reason behind the high accuracy of CoT on higher depths of ProofWriter-PD, we randomly selected 50 examples from Depth-5 of the dataset where CoT predicted the label correctly, and manually verified if the proof chain is correct or not. For comparison, we also manually verified the proofs generated by LAMBADA following a similar procedure. The results are reported in Figure 2(e). 

While LAMBADA mostly produces correct chains, CoT produces correct chains only for 28% of the examples. We find that hallucination is the main source of error (48% of the examples; see Appendix B.2 for other prominent failure modes). The hallucinated facts and rules mostly resulted in shortcuts to the correct answer. This hints at the possibility of spurious correlations in ProofWriter-PD that can be exploited by CoT (see Appendix B.2, Figure 10 for examples). This result is consistent with previous work showing that when LMs are asked to solve logical reasoning end-to-end, they rely on spurious correlations (Zhang et al., 2022b). Note that for modular approaches like SI and LAMBADA, the intermediate modules are impervious to the spurious correlations between the input and the label and do not suffer from this issue. 

## **5.3 Forward vs. Backward Chaining** 

As previously explained, SI is based on forward chaining and its selection module requires a combinatorial search to find the right subset of facts and rules (see Appendix C), and the search space becomes progressively larger in each iteration of the algorithm as new inferences are added to the theory. To verify whether the increase in the search space makes forward chaining progressively harder, we measured the success rate of the _k_ -th inference of SI for different values of _k_ on Depth-5 of PrOntoQA (see Appendix B.3 for details). From the results in Figure 3, we can see that the success rate indeed decreases in the later inferences of the model, where the size of the input theory is larger and therefore a larger space needs to be searched to find the right combination of facts and rules. Note that none of the components in LAMBADA require selecting a _subset_ , hence no combinatorial search is required (see Appendix C for more details). 

SI also suffers from inferring redundant facts. Figure 4 reports the number of unique inferences from SI for the examples in ProofWriterPD (Depth-5) where SI incorrectly predicted UNKNOWN (i.e., examples where a proof exists but SI failed to find it). The result shows that SI inferences contained no redundant facts only 29% of the time; in 7% of the cases, all 5 inferred facts were identical, and in another 10%, only two unique inferences were made. This shows that SI, and maybe more generally forward-chaining approaches, suffer from redundant inference. 

SI also over-predicts DISPROVED in the binary case and UNKNOWN in the three-way classification case (see Appendix B.4), performing even worse than the majority class for Depth-5 of PrOntoQA which has more PROVED labels than DISPROVED. 

These results, together with Figure 2, show that backward chaining (which is the backbone of reasoning in LAMBADA) is a better choice compared to forward chaining (the backbone in SI). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0007-12.png)


**----- Start of picture text -----**<br>
20%<br>10%<br>7% 1 unique inference<br>34% 2 unique inferences<br>3 unique inferences<br>29% 4 unique inferences<br>5 unique inferences<br>**----- End of picture text -----**<br>


Figure 4: Number of unique inferences generated by SI for Depth-5 of ProofWriter-PUD when selection and inference modules are called five times. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0008-00.png)


**----- Start of picture text -----**<br>
0.8 1.0<br>0.25<br>0.8 0.20<br>0.6<br>0.15<br>0.6<br>0.4 0.10<br>0.4 0.05<br>0.2 0.00<br>Forward CoT 0.2 Forward CoT<br>Backward CoT Backward CoT<br>0.0 0.0<br>Depth-0 Depth-1 Depth-2 Depth-3 Depth-5 Depth-0 Depth-1 Depth-2 Depth-3 Depth-5<br>(a) (b) (c)<br>Forward CoTBackward CoT<br>Proof Accuracy<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 5: Prediction accuracy results on (a) ProofWriter-PUD and (b) ProofWriter-PD with forward and backward CoT. (c) compares the proof accuracy of forward and backward CoT on ProofWriter (Depth-5) for a set of randomly sampled examples for which the models correctly predicted the proof label. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0008-02.png)


**----- Start of picture text -----**<br>
1.0<br>0.94 0.99 0.94 0.98<br>0.8 0.81 0.93 0.97<br>0.6 0.75 0.68 0.75<br>0.4 0.45<br>0.35 0.31 0.33 0.32 PaLM 8B<br>0.2 PaLM 62B<br>PaLM 540B<br>0.0<br>Fact Check  Fact Check  Rule  Goal  Sign<br> 1 trial  2 trials  Selection  Decomposition Agreement<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 6: ProofWriter (val) performance of modules in LAMBADA in isolation, for different LM sizes. 

## **5.4 Does Backward CoT Suffice?** 

Our results may raise the question of whether it is enough to directly incorporate the steps of backward chaining into CoT prompts, or if modularity (as in LAMBADA) is also needed. To answer this question, we experiment with a backward version of CoT where the proofs are written in the backward direction from the goal to the premises. The label accuracies are presented in Figure 5(a)–(b) for ProofWriter-PUD and ProofWriter-PD, and their proof accuracy on ProofWriter-PD (Depth-5) in Figure 5(c). The label accuracy of forward and backward CoT are comparable, but forward CoT leads to better performance on PUD and backward CoT leads to better performance on PD. For proof accuracy, however, we see a clear difference between the two versions where backward CoT produces substantially lower quality proofs compared to forward chaining. This result is consistent with the observations of Gontier et al. (2020) for finetuned LMs. 

The above results show that a modular formulation (as in LAMBADA) is key to successful logical reasoning and simply providing CoT in the backward direction does not suffice. We note, however, 

that future work can use the traces of our model to finetune (smaller) language models (e.g., Zelikman et al. 2022), or use the traces as training data in future language models to improve their performance with CoT prompting. 

Taking the label and proof accuracy results together, there is also a potential that backward CoT models are more heavily relying on spurious correlations for the PD case where backward CoT outperformed CoT, as backward CoT achieves a similar label accuracy as forward CoT but with a much lower proof accuracy. 

## **5.5 Qualitative Analysis** 

In Figure 1, we show the search trace created by LAMBADA for an example from ParaRules, where the answer was predicted correctly. From the figure, one can see how backward chaining helps LAMBADA effectively search and create the reasoning chain and how the LM helps fact checking, rule selection, goal decomposition, and sign agreement checking. In Appendix B.1, we include an example that has a much larger search trace. 

## **5.6 Individual Module Analysis** 

To understand which components in LAMBADA are responsible for the failure cases, we computed the individual accuracy of the four modules described in Section 3. For this purpose, we created four datasets from the validation set of ProofWriter, each measuring only the performance of one module in isolation (see Appendix D.1 for details). 

Based on the results of the PaLM 540B model in Figure 6, _Rule Selection_ is the lowest performing module followed by _Goal Decomposition_ . It is possible that the _Rule Selection_ module (partially) fails for some examples but LAMBADA still arrives at 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0009-00.png)


**----- Start of picture text -----**<br>
219.34<br>SI<br>10 [2] Lambada 57.22 99.45<br>27.79 27.30<br>18.53<br>12.77<br>10.14<br>10 [1] 7.26<br>2.98<br>Depth-0 Depth-1 Depth-2 Depth-3 Depth-5<br>Dataset Depth<br>Avg. #Inferences<br>**----- End of picture text -----**<br>


Figure 7: Comparing LAMBADA and SI w.r.t. the average number of inference calls they make per example for different subsets of the ProofWriter-PUD dataset. 

the correct conclusion and proof (e.g., if in Figure 1 the third call to _Rule Selection_ only returned Rule5). For _Fact Check_ , when we allow the model to only select one fact, the accuracy is 0 _._ 94 but when we allow the model to select two facts, the accuracy is near perfect. The _Sign Agreement_ module also shows near-perfect accuracy. 

## **5.7 The Role of Scale** 

We repeat the experiment from Section 5.6 with PaLM 62B and 8B to examine the effect of LM scale on LAMBADA. According to the results in Figure 6, when we use PaLM 62B, the performance of the _Goal Decomposition_ and _Sign Agreement_ modules remain comparable, but the performance for the _Fact Check_ and _Rule Selection_ modules drop substantially. Unlike the first two modules, the second two rely on a one-to-many comparison between the goal and each of the facts/rules which may require a larger model capacity. Moreover, we observe that in PaLM 8B, the accuracy for all components drops significantly, in some cases becoming close to random prediction. 

We argue that the extent to which the higherlevel reasoning algorithm breaks the problem into sub-problems should be dependent on the scale and power of the base LMs. If smaller LMs are used, then one may need finer-grained problem decomposition (e.g., further decomposing the one-tomany comparisons in the selection module). And as LMs become larger and stronger in the future, one could rely on them to solve problems with a coarser-grained decomposition of the problem. 

## **5.8 Number of Inference Calls** 

Another advantage of LAMBADA is its efficiency compared to other approaches that require multiple LM inference calls per example such as SI. In Figure 7, we compare the average number of LM calls per example, for different depths of ProofWriter- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0009-08.png)


**----- Start of picture text -----**<br>
1.0<br>0.8<br>0.6<br>0.4<br>Original Test Set<br>0.2 Novel Token Test Set<br>Novel Template Test Set<br>0.0<br>Depth-0 Depth-1 Depth-2 Depth-3 Depth-5<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 8: The performance of LAMBADA on ProofWriter-PUD for the original, novel token, and novel template test sets. 

PUD. LAMBADA requires much fewer calls compared to SI, especially at higher depths: for Depth1, LAMBADA requires 3.8x fewer calls whereas for Depth-5 it requires 11.8x fewer calls. 

## **5.9 Lexical Robustness** 

To analyze the lexical sensitivity of LAMBADA, we modified the test set of ProofWriter-PUD by replacing various lexical items (names, adjectives, and verbs) with novel tokens and the rule templates with novel ones. We then compared the performance of LAMBADA on the original and the modified test sets using the same few-shot examples. The details of the modifications are in Appendix B.5. As can be seen in Figure 8, the performance of LAMBADA remains almost unchanged, demonstrating robustness to lexical and templatic variations. 

## **6 Conclusion and Future Directions** 

We developed LAMBADA, an algorithm for deductive logical reasoning with natural language that combines the capacity of LMs to handle naturalistic text input with the backward chaining algorithm for robust symbolic reasoning. We showed that LAMBADA achieves significant improvements over competitive approaches on challenging benchmarks, both in terms of label accuracy (predicting if a statement can be proved or disproved based on a theory) and proof accuracy. Importantly, this improvement was also observed in a dataset that expresses the theory in more naturalistic expressions, clearly illustrating the benefit of combining an LM with reasoning modules. We also demonstrated the query efficiency and lexical robustness of LAMBADA. Although in this paper we only experiment with formal reasoning problems and datasets, we believe our key insight on the efficacy of backward, goal-directed reasoning with LMs has broader implications and can be adapted to other NLP tasks where multi-step inference is required. 

## **Limitations** 

We identify some limitations and risks with our current work that can be addressed in future work. 

- The current work is mainly applicable to logical entailment problems, where one needs to solve a classification problem of whether a goal can be proved, disproved, or neither proved nor disproved based on a theory. Future work can extend LAMBADA to non-classification cases, e.g., where one needs to apply logical reasoning to answer questions such as “What color is Fiona?”. 

- The current work assumes all the rules are given as input and the rule set is small enough to be included in the prompt. Future work can extend LAMBADA to the cases where not all the rules are provided as input and part of the knowledge has to come from the LM itself, as well as the case where not all the rules can be included in the prompt due to the limitation in the prompt length. 

- The current work is limited to deductive reasoning with the _modus ponens_ rule; future work can expand the applicability of LAMBADA on datasets with other types of rules such as proof by contradiction, disjunction elimination, etc. 

- The calls made to the LM modules in LAMBADA are dependent on the value from the previous call. That is, we need to wait for the results from one call before we decide what the next call must be. Since making batch calls to the LMs is typically easier and faster, future work can find ways to implement LAMBADA with batch LM calls. 

- While we showed that LAMBADA is more efficient than SI in terms of the number of inference calls it makes to the LM, it still requires many more calls to the LM compared to approaches such as CoT, hence increasing the required computation and cost. 

## **References** 

- Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. 2022. Exploring length generalization in large language models. In _Advances in Neural Information Processing Systems_ , volume 35, pages 38546–38556. Curran Associates, Inc. 

- Gregor Betz, Christian Voigt, and Kyle Richardson. 2021. Critical thinking for language models. In _Proceedings of the 14th International Conference_ 

_on Computational Semantics (IWCS)_ , pages 63–75, Groningen, The Netherlands (online). Association for Computational Linguistics. 

- Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pages 632–642, Lisbon, Portugal. Association for Computational Linguistics. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Advances in Neural Information Processing Systems_ , volume 33, pages 1877–1901. Curran Associates, Inc. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. PaLM: Scaling language modeling with pathways. _arXiv:2204.02311_ . 

- Peter Clark, Oyvind Tafjord, and Kyle Richardson. 2021. Transformers as soft reasoners over language. In _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence_ , IJCAI’20. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. _arXiv:2110.14168_ . 

- Antonia Creswell and Murray Shanahan. 2022. Faithful reasoning using large language models. _arXiv:2208.14271_ . 

- Antonia Creswell, Murray Shanahan, and Irina Higgins. 2023. Selection-inference: Exploiting large language models for interpretable logical reasoning. In _The Eleventh International Conference on Learning Representations_ . 

- Ido Dagan, Dan Roth, Mark Sammons, and Fabio Massimo Zanzotto. 2013. Recognizing textual entailment: Models and applications. _Synthesis Lectures on Human Language Technologies_ , 6(4):1–220. 

- Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. 2021. Explaining answers with entailment trees. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 7358–7370, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Successive prompting for decomposing complex questions. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 1251–1265, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Artur d’Avila Garcez and Luis C Lamb. 2020. Neurosymbolic ai: the 3rd wave. _arXiv:2012.05876_ . 

- Nicolas Gontier, Koustuv Sinha, Siva Reddy, and Chris Pal. 2020. Measuring systematic generalization in neural proof generation with transformers. In _Advances in Neural Information Processing Systems_ , volume 33, pages 22231–22242. Curran Associates, Inc. 

- Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Luke Benson, Lucy Sun, Ekaterina Zubova, Yujie Qiao, Matthew Burtell, et al. 2022. FOLIO: Natural language reasoning with first-order logic. _arXiv:2209.00840_ . 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_ , volume 1. Curran. 

- Carl Hewitt. 1969. Planner: A language for proving theorems in robots. In _Proceedings of the 1st International Joint Conference on Artificial Intelligence_ , IJCAI’69, page 295–301, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc. 

- Jie Huang and Kevin Chen-Chuan Chang. 2022. Towards reasoning in large language models: A survey. _arXiv:2212.10403_ . 

- Harsh Jhamtani and Peter Clark. 2020. Learning to explain: Datasets and models for identifying valid reasoning chains in multihop question-answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 

pages 137–150, Online. Association for Computational Linguistics. 

- Nora Kassner, Benno Krojer, and Hinrich Schütze. 2020. Are pretrained language models symbolic reasoners over knowledge? In _Proceedings of the 24th Conference on Computational Natural Language Learning_ , pages 552–564, Online. Association for Computational Linguistics. 

- Mehran Kazemi, Sid Mittal, and Deepak Ramachandran. 2023. Understanding finetuning for factual knowledge extraction from language models. _arXiv:2301.11293_ . 

- Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2023. Decomposed prompting: A modular approach for solving complex tasks. In _The Eleventh International Conference on Learning Representations_ . 

- Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. 2023. Transformers learn shortcuts to automata. In _The Eleventh International Conference on Learning Representations_ . 

- Gary Marcus. 2020. The next decade in AI: four steps towards robust artificial intelligence. _arXiv:2002.06177_ . 

- John McCarthy. 1959. Programs with common sense. In _Proceedings of the Teddington Conference on the Mechanization of Thought Processes_ , pages 75–91, London. Her Majesty’s Stationary Office. 

- Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2022. Show your work: Scratchpads for intermediate computation with language models. In _Deep Learning for Code Workshop_ . 

- David L Poole and Alan K Mackworth. 2010. _Artificial Intelligence: foundations of computational agents_ . Cambridge University Press. 

- Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, 

   - Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. 2021. Scaling language models: Methods, analysis & insights from training Gopher. _arXiv:2112.11446_ . 

- Mohammed Saeed, Naser Ahmadi, Preslav Nakov, and Paolo Papotti. 2021. RuleBERT: Teaching soft rules to pre-trained language models. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 1460–1476, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Abulhair Saparov and He He. 2023. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In _The Eleventh International Conference on Learning Representations_ . 

- Imanol Schlag, Sainbayar Sukhbaatar, Asli Celikyilmaz, Wen-tau Yih, Jason Weston, Jürgen Schmidhuber, and Xian Li. 2023. Large language model programs. _arXiv preprint arXiv:2305.05364_ . 

- Viktor Schlegel, Kamen Pavlov, and Ian Pratt-Hartmann. 2022. Can transformers reason in fragments of natural language? In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 11184–11199, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Jianhao Shen, Yichun Yin, Lin Li, Lifeng Shang, Xin Jiang, Ming Zhang, and Qun Liu. 2021. Generate & rank: A multi-task framework for math word problems. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 2269–2279, Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Zayne Sprague, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. 2022. Natural language deduction with incomplete information. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 8230–8258, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. _arXiv:2210.09261_ . 

- Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 3621–3634, Online. Association for Computational Linguistics. 

- Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2022. Large language models still can’t plan (a benchmark for LLMs on planning and reasoning about change). In _NeurIPS 2022 Foundation Models for Decision Making Workshop_ . 

- Boshi Wang, Xiang Deng, and Huan Sun. 2022. Iteratively prompt pre-trained language models for chain of thought. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 2714–2730, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems_ , volume 35, pages 24824–24837. Curran Associates, Inc. 

- Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In _the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 1112–1122. Association for Computational Linguistics. 

- Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. STaR: Bootstrapping reasoning with reasoning. In _Advances in Neural Information Processing Systems_ , volume 35, pages 15476–15488. Curran Associates, Inc. 

- Hanlin Zhang, Ziyang Li, Jiani Huang, Mayur Naik, and Eric Xing. 2022a. Improved logical reasoning of language models via differentiable symbolic programming. In _First Workshop on Pre-training: Perspectives, Pitfalls, and Paths Forward at ICML 2022_ . 

- Honghua Zhang, Liunian Harold Li, Tao Meng, KaiWei Chang, and Guy Van den Broeck. 2022b. On the paradox of learning to reason from data. _arXiv:2205.11502_ . 

- Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In _The Eleventh International Conference on Learning Representations_ . 

- Hattie Zhou, Azade Nova, Hugo Larochelle, Aaron Courville, Behnam Neyshabur, and Hanie Sedghi. 2022. Teaching algorithmic reasoning via in-context learning. _arXiv:2211.09066_ . 

## **A Caching and Avoiding Loops for LAMBADA** 

Since LAMBADA is a recursive algorithm, during the proof of an example Algorithm 1 may be called 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0013-00.png)


**----- Start of picture text -----**<br>
Dave is not green.<br>Fact Check<br>Fail<br>Facts: Rule6 Rule3<br>1. Anne is cold. Decompose Rule Selection Decompose<br>2. Anne is kind. Dave is nice. Dave is kind. Dave is blue. Dave is cold.<br>3. Charlie is nice. Fact Check Stop: Other  Fact Check Already<br>4. Dave is white. Fail branch failed Fail Proved<br>5.6. Dave is young.Fiona is blue. Rule Selection Rule Selection<br>7. Fiona is white. Rule2 Rule5<br>Rules: Decompose Decompose<br>1. If Dave is green and Dave is white Dave is green. Dave is cold.<br>then Dave is blue. Fact Check Fact Check<br>2.3. If something is green then it is nice.If something is blue and cold then it Decompose Rule6 Rule SelectionFail Rule3 Decompose Rule SelectionFail<br>4. is green.If something is white and young Dave is nice. Dave is kind. Dave is blue. Dave is cold. DecomposeRule8<br>then it is kind.<br>5.6. If something is cold then it is blue.All nice, kind things are green. Stop: Cycle Detected branch failedStop: Other  Fact CheckFail branch failedStop: Other  Fact CheckDave is kind. Fact CheckDave is young.<br>7. All kind, cold things are white. Rule Selection Fail<br>8. If something is kind and young then Rule5 Rule Selection Already<br>it is cold. Decompose Rule4 Proved<br>Goal: Dave is cold.<br>● Dave is not green. Fact Check Decompose<br>Fail Dave is white. Dave is young.<br>Stop: Max  Fact Check Fact Check<br>Depth Reached<br>**----- End of picture text -----**<br>


Figure 9: The search trace of LAMBADA on an example from ProofWriter with depth=5 where the answer was predicted correctly. The sign agreement module has been omitted for brevity. The modules color-coded with blue represent the calls where the module retrieved the value from the cache instead of calling the LM. 

with the same goal multiple times. For instance, consider the goal “Eric is nice” for the theory in Figure 1. Applying Rule6 breaks the goal into three sub-goals. The first one is “Eric is big” which is proved using the _Fact Check_ module. For the second sub-goal, Rule3 is used to compose it into three sub-goals the first of which we have proved before. Since we have already proved this sub-goal, we can save a _Fact Check_ call if we cache previous results. 

Note that the result of a call to LAMBADA can be different depending on the input max depth. For example, the algorithm may return UNKNOWN when called for the theory and goal in Figure 1 with max depth 0, and return PROVED when called with max depth 3. Specifically, if we can prove/disprove a goal at depth _d_ , we can conclude that it can be proved/disproved at depths _≥ d_ as well and we can get the value from the cache. Moreover, if the algorithm returns UNKNOWN for a goal at depth _d_ , we can conclude that it will also return UNKNOWN at depths _< d_ . Therefore, if the algorithm is called for a theory and goal at depth _d_ , we also check other depths to see if we have the results for other depths that apply to this case. Besides having a cache for the entire algorithm that avoids redundant computations when the truth of a goal has been previously computed for a theory, each individual module can also have its own cache as it is possible that the module is called for the same theory and goal. We show one such example in Figure 9 (to be discussed 

## in Section B). 

LAMBADA may sometimes run into loops. For example, to prove a (sub-)goal “Fiona is round?”, after recursively identifying rules that unify with it and decomposing it into sub-goals, the algorithm may arrive at a point where it needs to prove the “Fiona is round?” sub-goal, which is equivalent to the initial goal. To avoid such loops, for each path in the proof trace, we keep track of the (sub-)goals that are to be proved and stop further exploring that branch of the search trace when a loop is identified. 

Note that in Algorithm 1, for clarity of the algorithm we did not include the caching and loop avoidance operations. Also note that caching and loop avoidance mainly help with reducing the number of inference calls. 

## **B Additional Results and Analyses** 

In this section, we provide some more in-depth qualitative and quantitative analysis of the results from our model and the baselines. 

## **B.1 Qualitative Analysis** 

In Figure 9, we provide the search trace of LAMBADA for an example in ProofWriter (Depth-5) for which LAMBADA correctly predicted that the goal is disproved based on the theory. We deliberately selected an example with a large search trace to demonstrate the various aspects of LAMBADA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0014-00.png)


**----- Start of picture text -----**<br>
Facts:  The bald eagle is green. The bald eagle is young. The bald eagle sees the dog. The bear likes the dog. The bear needs the cow. The<br>cow needs the dog. The cow sees the dog. The dog is blue. The dog is green. The dog is young. The dog needs the bear. The dog needs the<br>cow.<br>Rules: If someone sees the bald eagle and they are nice then the bald eagle needs the bear. If someone is nice and young then they need<br>the dog. If someone likes the cow and the cow needs the dog then the cow is kind. If someone is young and blue then they like the bear. If<br>someone is blue and they like the bear then the bear likes the cow. If someone is green and they need the bear then they need the dog. If<br>someone sees the bear then they are nice. If someone is kind then they see the bear.<br>Question:  The bear likes the cow.<br>CoT Inference:  Since the facts state that the bear likes the cow so the statement the bear likes the cow is true.<br>Facts:  [...] The bear sees the mouse. The lion chases the squirrel. The lion is blue. The mouse is big. [...]<br>Rules:  If someone likes the mouse and they are blue then they are red. If someone is blue then they see the mouse. If the lion sees the<br>squirrel and the lion is blue then the lion chases the mouse. If someone chases the bear then they see the squirrel. If someone sees the bear<br>then the bear likes the squirrel. If someone is young and they see the bear then they chase the mouse. If someone sees the mouse then they<br>chase the bear. If someone is blue and they chase the mouse then they are young.<br>Question:  The bear does not see the squirrel.<br>CoT Inference:  Since the facts state that the bear sees the mouse and the rules state that if someone sees the mouse then they chase the<br>bear so we can add to the facts that the bear chases the mouse. Since the facts state that the bear chases the mouse and the rules state that<br>if someone chases the mouse then they see the squirrel so we can infer that the statement the bear does not see the squirrel is false.<br>Facts:  Anne is round. Anne is young. Charlie is green. Charlie is round. Charlie is young. Erin is big. Erin is green. Erin is round. Erin is<br>young. Harry is nice. Harry is white.<br>Rules:  All young, white things are round. If something is nice and green then it is white. Cold things are round. Young, green things are nice.<br>If something is big and green then it is nice. White, nice things are young. All green things are cold. White, round things are green. All cold,<br>round things are big.<br>Question:  Charlie is not nice.<br>CoT Inference:  Since the facts state that Charlie is green and the rules state that if something is green and nice then it is white so we can<br>infer that Charlie is white. [...] so we can infer that Charlie is nice and the statement Charlie is not nice is false.<br>Facts:  Anne is blue. Anne is nice. Anne is quiet. Anne is rough. Anne is white. Bob is big. Charlie is rough. Erin is big. Erin is nice. Erin is<br>young.<br>Rules:  Quiet, nice things are white. If something is rough then it is quiet. If Bob is white then Bob is young. If Anne is big and Anne is blue<br>then Anne is rough. If Bob is rough and Bob is quiet then Bob is nice. Big things are rough. If Erin is nice and Erin is young then Erin is big.<br>Question:  Erin is not nice.<br>Inference:  Since the facts state that Erin is nice [...] Since the facts state that Erin is rough and the rules state that if something is rough then<br>it is quiet so we can infer that the statement Erin is not nice is false.<br>Hallucinated Fact<br>Hallucinated Rule<br>Conjunction Error<br>Wrong derivation<br>**----- End of picture text -----**<br>


Figure 10: Examples of wrong CoT proof chains from four different categories. The erroneous part is marked in red. 

LAMBADA starts by calling the _Fact Check_ module on the goal which fails to prove or disprove it. So _Rule Selection_ is called which identifies two rules that can be applied: Rule3 and Rule6. Since Rule6 is shorter, the reranker ranks it higher; LAMBADA starts with this rule and calls the _Goal Decomposition_ module which breaks the goal into two sub-goals: “Dave is nice.” and “Dave is kind.”. Starting with the first sub-goal, _Face Check_ fails on it so _Rule Selection_ is called which selects Rule2 and _Goal Decomposition_ decomposes the sub-goal into “Dave is green.”. 

Note that if the cycle checking was smart enough to understand that this sub-goal is the negation of the root goal, we could stop further searching this branch. However, we currently only do cycle matching for exact matches so the algorithm continues the search trace. 

_Fact Check_ fails again so _Rule Selection_ is called which selects Rule3 and Rule6 again, and since Rule6 is shorter the algorithm continues with that rule. _Goal Decomposition_ breaks the sub-goal into “Dave is nice.” and “Dave is kind.”. Considering the first sub-goal, the algorithm identifies a cycle and stops the search. The second sub-goal 

is also ignored as there is a conjunction between the sub-goals. 

The algorithm then continues with calling _Goal Decomposition_ for Rule3 which breaks the subgoal into “Dave is blue.” and “Dave is cold.”. Starting with the first sub-goal, since _Fact Check_ fails the algorithm calls _Rule Selection_ which selects Rule5 and _Goal Decomposition_ breaks the sub-goal into “Dave is cold.”. _Face Check_ fails on this sub-goal and since the maximum depth is reached, the algorithm stops expanding this branch. Moreover, the branch for “Dave is cold.” is no longer pursued because there was a conjunction between the sub-goals and one of them failed. 

Moving on to the right branch in Figure 9, the algorithm calls the _Goal Decomposition_ module for the goal and Rule3. Since we have previously computed it, the sub-goals “Dave is blue.” and “Dave is cold.” are returned from the cache. _Fact Check_ is called on “Dave is blue.” and since it has been computed before, the result (failure) is retrieved from the cache. The _Rule Selection_ module is called, where the result (Rule5) is again retrieved from the cache. _Goal Decomposition_ is then called and the sub-goal “Dave is cold.” is 

retrieved from the cache. _Fact Check_ fails again (retrieved from the cache), _Rule Selection_ selects Rule8 and _Goal Decomposition_ produces two subgoals: “Dave is kind.” and “Dave is young.”. For “Dave is kind.”, _Fact Check_ fails, _Rule Selection_ selects Rule4 and _Goal Decomposition_ produces two sub-goals: “Dave is white.” and “Dave is young.”. For both of these sub-goals, _Fact Check_ succeeds in proving them. The algorithm then also checks “Dave is young.” for the right branch, but since this sub-goal has already been proved, it just gets the result from the cache. The algorithm then checks “Dave is cold.” for the rightmost branch, but since this sub-goal has already been proved, it just gets the result from the cache. 

The model also calls the _Sign Agreement_ module for rules on the right branch (not shown in the Figure) and finds out that the sign of the rules and the sub-goals agree for all cases, except for the very first rule selected (Rule3) so it correctly concludes that the goal is disproved. 

## **B.2 Further Analysis of CoT** 

In Figure 2(e), we observed that CoT mostly produces wrong proof chains even when the predicted label is correct. Through manually analyzing 50 examples for which CoT predicted the correct label, we identified three dominant reasons for the chains being wrong: 1- hallucinating rules or facts, 2- not understanding conjunction, and 3- making invalid derivations. In Figure 10, we show failure examples from each category. Notice that, e.g., in the example with a hallucinated rule, CoT relies on a rule “if someone chases the mouse then they see the squirrel” which not only does not appear in the provided set of rules, but cannot even be derived with a combination of the rules. 

The high label accuracy of CoT and its low proof accuracy on ProofWriter-PD hint at the possibility of spurious biases that can be exploited by CoT. For example, we found that in 9 _._ 2% of the examples which require 1+ reasoning hops, the consequent of one of the rules in the theory is the same as the goal to be proved, and for 98 _._ 9% of these examples the label is PROVED. In several of these examples, CoT simply concluded that the goal can be proved in 0 hops based on a hallucinated fact. Moreover, the existence of the word “not” in the goal is highly predictive of the label: goals having “not” are mostly DISPROVED and goals not 

having “not” are mostly PROVED. The PUD case solves the latter issue to a large extent as the label for a good portion of the examples with or without “not” in UNKNOWN. The spurious correlations also explain the fluctuations in the CoT performance across different depths, as the performance depends on how much those correlations appear in the few-shot demonstrations. 

We reiterate that for SI and LAMBADA, such spurious correlations between the input and the label cannot be exploited because the intermediate modules are impervious to the correlations between the input and the label. 

## **B.3 Forward Chaining Becomes Progressively More Difficult** 

Algorithms such as SI that are based on forward chaining require a combinatorial search of the theory to find the right subset of facts and rules in each step of the reasoning. The search space becomes progressively larger as the algorithm makes new inferences and those inferences are added back to the theory. For example, if the initial size of the theory (i.e. the number of facts plus the number of rules) is _|C|_ , when making the _k_ -th inference the size of the theory is _|C|_ + _k −_ 1. 

Conceptually, as the model produces more inferences, the distance to the goal (in terms of the number of hops remaining between the goal and the facts) should reduce and so the later inferences should be more accurate. However, we hypothesize that the increase in the size of the theory (and hence the size of the search space) may result in lower success rates in the later inferences of the SI model. To verify this experimentally, we further analyzed the results of SI on depth-5 of PrOntoQA as follows. We extracted the subset of examples where the label was PROVED but SI failed to find a proof (these are examples where at least one of the inferences is not on the proof chain). Then, as a proxy for measuring the responsibility of the _k_ -th inference of the model for the failure, we measured the percentage of times the _k_ -th inference was on the proof chain (the proof chain for each test example is provided as part of the dataset). Notice that it is possible that, e.g., the first inference is not on the proof chain, but the rest of the inferences are. The results are reported in Figure 3 in the main text. The results show that the chance of producing inferences that are on the proof chain progressively decreases in the later inferences of the model where 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0016-00.png)


**----- Start of picture text -----**<br>
Chain Of Thought Selection Inference LAMBADA<br>proved 84 23 169 proved 34 0 242 proved 163 2 111<br>disproved 14 82 180 disproved 4 15 257 disproved 0 149 127<br>unknown 53 64 331 unknown 19 15 413 unknown 14 24 410<br>proved disproved unknown proved disproved unknown proved disproved unknown<br>Predicted label Predicted label Predicted label<br>(a) ProofWriter-PUD (Depth-5) (b) ProofWriter-PUD (Depth-5) (c) ProofWriter-PUD (Depth-5)<br>Chain Of Thought Selection Inference LAMBADA<br>proved 169 36 proved 6 209 proved 203 12<br>disproved 67 110 disproved 9 176 disproved 2 183<br>proved disproved proved disproved proved disproved<br>Predicted label Predicted label Predicted label<br>(d) PrOntoQA (Depth-5) (e) PrOntoQA (Depth-5) (f) PrOntoQA (Depth-5)<br>Chain Of Thought LAMBADA<br>proved 112 3 10 proved 222 8 20<br>disproved 8 95 22 disproved 6 224 20<br>unknown 87 72 91 unknown 40 44 416<br>proved disproved unknown proved disproved unknown<br>Predicted label Predicted label<br>(g) ParaRules (h) ParaRules<br>True label True label True label<br>True label True label True label<br>True label True label<br>**----- End of picture text -----**<br>


Figure 11: Confusion matrices. 

the size of the input theory (and hence the search space) is larger. 

## **B.4 Confusion Matrices** 

We reported the overall model accuracies in the main text. Here, we report finer-grained confusion matrices that help better understand the biases of the model. Figure 11 reports the confusion matrices for our datasets. According to the results, we observe that whenever LAMBADA predicts PROVED or DISPROVED, the prediction is mostly correct. The accuracy is slightly more on cases where the prediction is PROVED than DISPROVED. We believe this is because DISPROVED cases typically involve negation that makes the reasoning more complex. However, there are several examples 

for which the label is PROVED or DISPROVED, whereas the model predicts UNKNOWN. 

CoT and SI also show similar behaviour as LAMBADA on ProofWriter-PUD but with a larger bias toward prediction UNKNOWN. Moreover, SI shows a large tendency toward predicting DISPROVED for PrOntoQA. 

## **B.5 Lexical Sensitivity Analysis** 

To analyze the lexical sensitivity of LAMBADA, we created a new test for ProofWriter-PUD which contains tokens that do not appear in demonstration examples. Specifically, we manually created a pool of entity names, animal names, adjectives, and verbs (all of them previously not appearing in the ProofWriter dataset) and then made the follow- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0017-00.png)


**----- Start of picture text -----**<br>
1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>Original Test Set Original Test Set<br>0.2 Novel Token Test Set (v1) 0.2 Novel Template Test Set (v1)<br>Novel Token Test Set (v2) Novel Template Test Set (v2)<br>0.0 0.0<br>Depth-0 Depth-1 Depth-2 Depth-3 Depth-5 Depth-0 Depth-1 Depth-2 Depth-3 Depth-5<br>(a) (b)<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 12: The performance of LAMBADA on ProofWriter-PUD for (a) the original and the novel token test sets, (b) the original and the novel template test sets. The results show that LAMBADA is robust to lexical and template modifications. 

ing modifications for each example: 1- identified all entity names and mapped each entity name to a randomly selected name from the pool, 2- identified all animals and mapped each of them to a randomly selected animal from the pool, 3- identified all adjectives and mapped each of them to a randomly selected adjective from the pool, and 4- identified all verbs and mapped each of them (except the _to be_ verbs) to a randomly selected verb from the pool. As an example, dog may be mapped to bison in one example and to camel in another. Then, using the same few-shot examples as before, we tested the performance of LAMBADA on this modified test set and compared the results to the original test set. 

We also analyzed the sensitivity to the templates used for the rules. Toward this goal, we identified the templates used for the rules in the ProofWriter dataset and replaced each template with another template (previously not appearing in the ProofWriter dataset). For example, we changed the template “[X] things are [Y]” to “It is a truth that [X] things are always [Y] as well”. Then, using the same few-shot examples as before, we tested the performance of LAMBADA on this modified test set and compared the results to the original test set. 

We repeated the aforementioned experiments twice for each analysis each time using a different set of tokens/templates. The results in Figure 8 in the main text demonstrate the average accuracy across two runs. The results for individual runs are presented in Figure 12(a), (b) for the two analyses respectively. According to the results, while we observe some variations in the total accuracy (for some depths the performance goes slightly down and for some depths goes slightly up), the perfor- 

mance stays in the same ballpark, showing the robustness of LAMBADA. Moreover, comparing the results on the modified test set with those of the baselines reported in the main text, we observe that even on this modified test set, LAMBADA performs significantly better than the baselines tested on the original test set. 

## **C Combinatorial Search Issue in Forward Chaining** 

Consider a simple fictional theory with the following facts: 

[Anne is cold., Anne is nice and pink., Anne is kind., Anne is green., Anne is big and young., Anne is rough., Anne is round.] the following rules: 

[Cold, red people are white., Nice, blue people are white., Kind, green people are white., Cold, round people are white., Big, green people are white.] 

and the goal “Anne is white.”. An approach based on forward chaining requires selecting a subset of the facts and rules from the theory from which this goal can be proved. Specifically, it needs to select “Anne is cold.”, “Anne is round.”, and Cold, round people are white. from the theory. Such a selection requires a combinatorial search where different combinations of facts and rules should be tested to see which one can lead to proving the goal. An LM may fail to search this space effectively in a single inference call. 

SI uses an approximation to reduce the search space: it first makes an inference call to an LM to select one fact/rule, then it makes another inference call to select the next fact/rule based on the first one, and continues to make inference calls until a halting criterion is met. This approximation re- 

duces the search space from a combinatorial space to a linear space. Since the facts/rules are not selected jointly, however, the chances of selecting the wrong combinations of facts and rules increase because repairing a wrong first choice is not possible, and this leads to low performance as evidenced in our experimental results. 

With a backward chaining approach such as LAMBADA, on the other hand, no combinatorial search (or approximations to it) is required: the _Rule Selection_ module verifies each rule independently to see which one is applicable (i.e. a linear scan), the _Goal Decomposition_ module breaks goals into sub-goals based on each selected rule independently of the other selected rules, and the _Fact Check_ module verifies the existence of a fact that entails or contradicts the goal with a linear search over the facts. 

## **D Implementation Details** 

For our experiments, we used the PaLM 540B model (Chowdhery et al., 2022) for all the models (both LAMBADA and the baselines) served on a 4 _×_ 4 TPU v4 architecture. The decoding temperature was set to zero. For testing CoT on PrOntoQA, we used the same demonstration examples as the original work but slightly changed the wording by adding conjunctive words such as “Since” and “So” to make the chains have a better flow. The reason for this modification was that we found when working with PaLM, prompts that have a better flow result in better predictions. This can be viewed from Figure 13 where we compare the performance for the original prompts vs. the prompts with the conjunctive words added. It can be viewed that while the latter slightly underperforms on Depth-1 (where the reasoning flow is not as important), it substantially improves the results for higher depths (especially Depth-5). For ProofWriter, we wrote similar few-shot examples. 

For SI, we used the same demonstration examples as in the original work for ProofWriter; for PrOntoQA we wrote few-shot examples following a similar pattern to those for ProofWriter. For each dataset depth we used/wrote specific few-shot examples (e.g., when working with a subset of the data that has examples requiring at most _k_ hops of reasoning, our CoT demonstrations also require only _k_ hops of reasoning), except for ProofWriter Depth-5 where, following the original work, we used it for testing length-generalization and only 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language_images/acl_2023_LAMBADA_Backward_Chaining_for_Automated_Reasoning_in_Natural_Language.pdf-0018-05.png)


**----- Start of picture text -----**<br>
0.8<br>0.6<br>0.4<br>0.2 Original prompts<br>Prompts with better flow<br>0.0<br>Depth-1 Depth-3 Depth-5<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 13: CoT results on PrOntoQA with the original prompts vs. the prompts with conjunctive words added to make the sentences flow better. 

included examples with chains up to 3 hops. For running CoT on ProofWriter-PUD, we included extra few-shot examples where the label is UNKNOWN; the explanation for these examples is that the goal cannot be proved or disproved with a combination of the facts and the rules. For running SI on ProofWriter-PUD, after obtaining the inferences by running SI, we give the inferences and the goal to our _Fact Check_ module which decides if the goal can be proved, disproved, or neither. Since _ProofWriter-PD_ and _PrOntoQA_ are binary datasets but LAMBADA makes three-way predictions (PROVED, DISPROVED, and UNKNOWN), to test LAMBADA on these datasets, similar to SI we combine the UNKNOWN and DISPROVED predictions into one class. 

## **D.1 Datasets for Individual Module Evaluation** 

For creating datasets for measuring the performance of individual modules in LAMBADA, we proceeded as follows. For _Fact Check_ , we randomly selected 100 examples from the Depth-0 examples. We count a model prediction to be correct if it produces the same label as the one specified in the ProofWriter dataset. For _Rule Selection_ , we randomly selected 100 examples and manually enumerated every rule whose consequent unifies with the goal. A model prediction is considered correct if it predicts _all_ such rules correctly. For _Goal Decomposition_ , we randomly selected 100 rules and goals such that the consequent of the rule unifies with the goal and then manually wrote the subgoals. A model prediction is considered correct if it predicts _all_ the sub-goals correctly. For _Sign Agreement_ , we re-used the same examples from the _Goal Decomposition_ module and manually labeled them with respect to their sign agreement/disagreement. 

## **D.2 Quality Issues in ParaRules** 

We found the ParaRules dataset to has a high amount of variation in the text, in the facts, and in the rules thus making it a valuable benchmark for evaluating text-based logical reasoning. We also found a few quality issues in the ParaRules dataset that were introduced when annotators converted facts and rules into natural language form. Here, we describe some of the main issues that we found and fixed. 

- **Changing antecedents and consequents:** We found that in some cases where the rule was “X and Y imply Z”, the natural language version of the rule produced by annotators was written as if “X implies Y and Z” or “X implies Y or Z”. As an example, the rule “Cold, nice people are red.” was written in natural language form as “Some cold people can be nice at times,and red at at other times.”. For such cases, we modified the text to make the antecedents and consequent match the original rule. 

- **Introducing new antecedents:** In some cases, the annotator introduced new antecedents in the rule. For example, for a rule where the antecedents were “green”, “red” and “rough”, the annotator added another antecedent “naive” (“If someone is green and naive ...”). For such cases, we removed the extra antecedents. 

- **Turning general rules to specific ones:** In some cases, the natural language version of a general rule was written for only a specific entity. For example the rule “Rough, young, green people are very round.” was written as “Tom is a rough, young person to know ...”. We removed the specific entities and made the rule generally applicable. 

- **Introducing pronouns:** For some of the facts, we found that the annotator replaced the name of the entity with a pronoun. As an example, “Dave is ...” was annotated as “He is ...”. We replaced the pronouns with the original entity name in the theory. 

## **D.3 Prompts** 

We provide an overview of the prompts we used for each of the four components of our model for the ProofWriter dataset. 

The pseudo-code for the _Fact Check_ module is provided in Algorithm 3. For selecting a fact in _Fact Check_ , our prompt looks like the following: 

## **Algorithm 3** FactCheck 

**Input:** Facts _F_ , Goal _G_ , Number of trials _n_ 

- 1: **for** _n_ times do **do** 

- 2: f = FactSelection( _F_ , _G_ ) 3: result = FactVerifier(f, _G_ ) 4: **if** result _̸_ = UNKNOWN **then** 5: **return** result 6: _F_ = _F − f_ 

- 7: **return** UNKNOWN 

## **Algorithm 4** RuleSelection 

## **Input:** Rules _R_ , Goal _G_ 

- 1: **I** = RuleImplications( _R_ ) 

- 2: selected = SelectRules( **I** , _G_ ) 

- 3: **return** selected 

Example 1 Fact1: <FACT1> Fact2: <FACT2> ... Factn: <FACTn> Question: <QUESTION> Inference: For the question <QUESTION> the most relevant fact is Facti (<FACTi>). 

... Example K Fact1: <FACT> Fact2: <FACT> ... Factm: <FACT> 

Question: <QUESTION> Inference: 

For verifying if the goal/question can be derived from the selected fact, we use the following prompt: 

Example 1 Fact: <FACT> Question: <QUESTION> Inference: The fact <FACT> [X1] the question <QUESTION> so [X2]. 

... Example K Fact: <FACT> Question: <QUESTION> Inference: 

In the case where the goal can be proved from the fact, we replace [X1] with “is equivalent to” and [X2] with “so the answer is "yes"”. In the case where the goal can be disproved from the fact, we replace [X1] with “is the negation of” and [X2] with “so the answer is "no"”. And in the case where the goal can neither be proved 

nor disproved, we replace [X1] with “is neither equivalent nor the negation of” and [X2] with “so the question cannot be inferred from the fact”. 

The pseudo-code for the _Rule Selection_ module is provided in Algorithm 4. For finding the implication/consequent of the rules, we use the following prompt: 

Example 1 Rule1: <RULE1>, Rule2: <RULE2> ... Rulen: <RULEn> Inference: Rule1 implies [X1], _. . ._ , Rulen implies [Xn]. ... Example K Rule1: <RULE1>, Rule2: <RULE2> ... Rulem: <RULEm> Inference: 

[Xi]s depend on the consequent of each rule. For rules such as “Rough, nice people are red.” we write [Xi] as “(is; red)”, and for rules such as “If the cat chases the dog then the cat sees the dog.” we write [Xi] as “(cat; chase; dog)”. 

For rule selection based on the implications, we use the following prompt: 

Example 1 

Rule1 implies <IMLP1>, Rule2 implies <IMPL2>, ..., Rulen implies <IMPLn> Question: <QUESTION> Inference: The question is about <IMPLq>: Rule1 <IMPL1> [X1] <IMPLq>, _. . ._ , <IMPLn> [Xn] <IMPLq>. ... 

so the question breaks down to <SUBQ>*. 

... Example K Rule: <RULE> Question: <QUESTION> Inference: 

where <SUBJq> indicates the subject of the question, <PRM>* indicates the premises/antecedents in the rule (the * indicates that there might be multiple premises), and <SUBQ>* indicates the sub-goals. 

Finally, for sign agreement, we use the following prompt: 

Example 1 

Rule: <Rule> Question: <QUESTION> 

Inference: The rule implication <IMLPr> is [Xr], the question <IMPLq> is [Xq], so signs [Xd]. 

... Example K Rule: <RULE> Question: <QUESTION> Inference: 

where <IMLPr> shows the implication of the rule and <IMPLq> indicates the implication of the question. [Xr] and [Xq] are either “positive“ or “negated“ depending on the sign of the implication. [Xd] is either “agree“ or “disagree“ depending on whether the signs agree or not. 

Example K Rule1 implies <IMLP1>, Rule2 implies <IMPL2>, ..., Rulem implies <IMPLm> Question: <QUESTION> Inference: 

where each [X1] is either “is applicable to“ or “not applicable to“ depending on whether the rule can be applied or not. 

For goal decomposition, we use the following prompt: 

Example 1 Rule: <Rule> Question: <QUESTION> Inference: The question subject is <SUBJq> and the rule premises are <PRM>*, 

