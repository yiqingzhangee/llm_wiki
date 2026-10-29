# **LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models** 

**Mihir Parmar**[1] **Nisarg Patel**[1] **Neeraj Varshney**[1] **Mutsumi Nakamura**[1] **Man Luo**[1] **Santosh Mashetty**[1] **Arindam Mitra**[2] **Chitta Baral**[1] 

1Arizona State University 2Microsoft Research 

{mparmar3, nppatel7, chitta}@asu.edu 

## **Abstract** 

Recently developed large language models (LLMs) have been shown to perform remarkably well on a wide range of language understanding tasks. But, can they really “reason” over the natural language? This question has been receiving significant research attention and many reasoning skills such as commonsense, numerical, and qualitative have been studied. However, the crucial skill pertaining to ‘logical reasoning’ has remained underexplored. Existing work investigating this reasoning ability of LLMs has focused only on a couple of inference rules (such as modus ponens and modus tollens) of propositional and first-order logic. Addressing the above limitation, we comprehensively evaluate the logical reasoning ability of LLMs on 25 different reasoning patterns spanning over propositional, first-order, and non-monotonic logics. To enable systematic evaluation, we introduce _LogicBench_ , a natural language question-answering dataset focusing on the use of a single inference rule. We conduct detailed analysis with a range of LLMs such as GPT-4, ChatGPT, Gemini, Llama-2, and Mistral using chain-of-thought prompting. Experimental results show that existing LLMs do not fare well on _LogicBench_ ; especially, they struggle with instances involving complex reasoning and negations. Furthermore, they sometimes overlook contextual information necessary for reasoning to arrive at the correct conclusion. We believe that our work and findings facilitate future research for evaluating and enhancing the logical reasoning ability of LLMs[1] . 

## **1 Introduction** 

Large language models such as GPT-4, ChatGPT, Google Gemini, Llama-2 (Touvron et al., 2023), and Mistral (Jiang et al., 2023) have made remarkable progress in NLP research enabling machines 

> 1Data and code are available at https://github.com/ Mihir3009/LogicBench 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models_images/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models.pdf-0001-10.png)


**----- Start of picture text -----**<br>
Propositional Logic<br>Modus Tollens Commutation HypotheticalSyllogism DisjunctiveSyllogism ConstructiveDilemma<br>DestructiveDilemma BidirectionalDilemma Material Implication<br>First-Order Logic<br>Modus Tollens Modus Ponens HypotheticalSyllogism DisjunctiveSyllogism ConstructiveDilemma<br>Destructive Bidirectional Universal Existantial<br>Dilemma Dilemma Instantiation Instantiation<br>Non-monotonic Reasoning<br>Default Reasoning Default Reasoningwith IrrelevantInformation Default Reasoningwith SeveralDefaults with Open DomainDefault Reasoning Unknown Exp. 1Reasoning with<br>Reasoning with Reasoning with Reasoning with<br>Unknown Exp. 2 Unknown Exp. 3 Priorities<br>**----- End of picture text -----**<br>


Figure 1: Comprehensive representation of different inference rules and reasoning patterns covered by propositional, first-order, and non-monotonic logics. _Exp._ indicates Expectation 

to perform a variety of language tasks that were previously thought to be exclusive to humans (OpenAI, 2023; Brown et al., 2020; Zhao et al., 2023). However, the ability of these LLMs to reason “logically” over natural language text remains under-explored, even though logical reasoning is a fundamental aspect of intelligence and a crucial requirement for many practical applications, such as questionanswering systems (Khashabi, 2019) and conversational agents (Beygi et al., 2022). Although several datasets have been proposed (Clark et al., 2021; Tian et al., 2021; Joshi et al., 2020; Saeed et al., 2021) to evaluate the logical reasoning capabilities of LLMs, these datasets are limited in their scope by (1) not evaluating logical reasoning independently of other forms of reasoning such as LogiQA (Liu et al., 2021a) and ReClor (Yu et al., 2020); and (2) evaluating only a single type of logic and covering only few logical inference rules as done in FOLIO (Han et al., 2022) and ProntoQA (Saparov and He, 2023). Thus, our aim in this work is to address the lacuna of having a more comprehensive set of inference rules for evaluating the logical reasoning ability of LLMs. 

To this end, we introduce a systematically created question-answering dataset for the evaluation of logical reasoning ability using a single inference 

rule, called _LogicBench_ . Besides evaluating the logical reasoning ability of LLMs, by evaluating models on single inference rules, we can also gain insights into the frequency of text sequences corresponding to these rules in the pre-training data and their impact on model performance. As illustrated in Figure 1, _LogicBench_ includes a total of 25 reasoning patterns across ‘propositional, first-order, and non-monotonic’ logics. To the best of the authors’ knowledge, this is the first work to study non-monotonic reasoning, as well as various inference rules in propositional and first-order logics including hypothetical and disjunctive syllogism; and bidirectional, constructive, and destructive dilemmas in NLP domain. To evaluate LLMs using _LogicBench_ , we formulate two different tasks: (i) a Binary Question-Answering (BQA) task in which the context comprises logical statements and the models have to determine whether a conclusion given in the question is logically entailed by the context, and (ii) a Multiple-Choice Questions-Answering (MCQA) task where models must select the most appropriate logical conclusion from four distinct options, based on the provided context. The rationale behind having BQA and MCQA tasks is that it provides systematic standard metric-based evaluation (i.e., direct comparison of LLMs’ performance in terms of accuracy), which could be more challenging with open-ended question-answer formats. Examples instances of various reasoning patterns are presented in Table 4 and App. C. 

To construct _LogicBench_ , we use a three-stage procedure (refer to §3). In the first stage, we prompt GPT-3.5 to generate a variety of coherent natural language sentences having different ‘ontologies’ (i.e., a collection of concepts as car, person, and animals) and their corresponding negations (refer to §3.2.1). In the second stage, we generate _(context, question)_ pairs where the context represents a natural language narrative consisting of logical statements, and the question is formulated to exhibit the logical conclusion derived from the context. In the third stage, we generate task-specific data instances (i.e., _(context, question, answer)_ triplets). 

We conduct a comprehensive evaluation with a range of LLMs on _LogicBench_ including GPT-4, ChatGPT (GPT-3.5-Turbo), Gemini-Pro, Llama2-7B-Chat, and Mistral-7B-Instruct using chainof-thought (Wei et al., 2022). In particular, we measure the accuracy of LLMs predictions on both BQA and MCQA tasks. Our experiments result in several interesting findings such as LLMs often 

|Dataset|Logic Covered|Logic Covered|Logic Covered|Logic Covered|Logic Covered|
|---|---|---|---|---|---|
||PL|FL|NM|||
|Ruletaker<br>LogicNLI<br>ProofWriter<br>FOLIO<br>SimpleLogic<br>ProntoQA|✗|✓|✗|✗|Human-annotated|
||✗|✓|✗|✗|Semi-automated|
||✓|✓|✗|✗|✗|
||✗|✓|✗|✗|Human-annotated|
||✓|✗|✗|✗|✓|
||✗|✓|✗|✓|✓|
|**LogicBench**|✓|✓|✓|✓|✓|



Table 1: _LogicBench_ ’s comparison with current datasets 

struggle to reason over complex logical contexts that involve only a single inference rule and encounter difficulties with inference rules involving negations. Experimental results reveal that these models struggle with respect to many of the inference rules and patterns, suggesting significant room for improvement in their logical reasoning abilities. To further demonstrate the use of _LogicBench_ , we synthetically augment it and fine-tune T5-large. Our preliminary results (App. I) show that this improves the logical reasoning ability of existing models leading to performance improvement on other logic datasets, LogicNLI, and FOLIO ( _∼_ 2% on an average), and shows competitive performance on LogiQA and ReClor. 

## **2 Related Work** 

As LLMs continue to evolve rapidly, it becomes increasingly crucial to evaluate their diverse reasoning capabilities, as well as those of forthcoming LLMs. LogiQA (Liu et al., 2021a) and ReClor (Yu et al., 2020) have made notable contributions by compiling multichoice questions from standardized examinations that demand diverse forms of logical reasoning. In contrast to _LogicBench_ , these datasets involve mixed forms of reasoning and do not focus on assessing logical reasoning in isolation. 

A few past attempts have been made to evaluate only logical reasoning while excluding other forms of reasoning. For example, CLUTTER (Sinha et al., 2019) covers inductive reasoning, (Hahn et al., 2021) covers temporal logic, and Ruletaker (Clark et al., 2021) evaluates whether a transformerbased model emulates deductive reasoning over synthetically generated statements in a limited setting. LogicNLI (Tian et al., 2021) introduced a diagnostic benchmark for FOL reasoning, with the dataset constructed by automatically generating logic expressions and replacing the entity and attribute placeholders. 

Our proposed dataset is similar (in terms of task formulation) to ProofWriter (Tafjord et al., 2021), 

|**Names**|**Propositional Logic**|**Extension to a (restricted) First-order Logic**|
|---|---|---|
|MP|((_p →q_)_∧p_)_⊢q_|(_∀x_(_p_(_x_)_→q_(_x_))_∧p_(_a_))_⊢q_(_a_)|
|MT|((_p →q_)_∧¬q_)_⊢¬p_|(_∀x_(_p_(_x_)_→q_(_x_))_∧¬q_(_a_))_⊢¬p_(_a_)|
|HS|((_p →q_))_∧_(_q →r_))_⊢_(_p →r_)|(_∀x_((_p_(_x_)_→q_(_x_))_∧_(_q_(_x_)_→r_(_x_)))_⊢_(_p_(_a_)_→r_(_a_))|
|DS|((_p ∨q_)_∧¬p_)_⊢q_|(_∀x_(_p_(_x_)_∨q_(_x_))_∧¬p_(_a_))_⊢q_(_a_)|
|CD|((_p →q_)_∧_(_r →s_)_∧_(_p ∨r_))_⊢_(_q ∨s_)|(_∀x_((_p_(_x_)_→q_(_x_))_∧_(_r_(_x_)_→s_(_x_)))_∧_(_p_(_a_)_∨r_(_a_))) _⊢_(_q_(_a_)_∨s_(_a_))|
|DD|((_p →q_)_∧_(_r →s_)_∧_(_¬q ∨¬s_))_⊢_(_¬p ∨¬r_)|(_∀x_((_p_(_x_)_→q_(_x_))_∧_(_r_(_x_)_→s_(_x_)))_∧_(_¬q_(_a_)_∨¬s_(_a_))) _⊢_(_¬p_(_a_)_∨¬r_(_a_))|
|BD|((_p →q_)_∧_(_r →s_)_∧_(_p ∨¬s_))_⊢_(_q ∨¬r_)|(_∀x_((_p_(_x_)_→q_(_x_))_∧_(_r_(_x_)_→s_(_x_)))_∧_(_p_(_a_)_∨¬s_(_a_))) _⊢_(_q_(_a_)_∨¬r_(_a_))|
|CT|(_p ∨q_)_⊢_(_q ∨p_)|-|
|MI|(_p →q_)_⊢_(_¬p ∨q_)|-|
|EG|-|_P_ (_a_)_⇒∃xP_ (_x_)|
|UI|-|_∀x A ⇒A{x →a}_|



Table 2: Inference rules and (two) axioms that establish the relationship between premises and conclusions. MP: Modus Ponens, MT: Modus Tollens, HS: Hypothetical Syllogism, DS: Disjunctive Syllogism, CD: Constructive Dilemma, DD: Destructive Dilemma, BD: Bidirectional Dilemma, CT: Commutation, MI: Material Implication, EG: Existential Generalization, UI: Universal Instantiation 

FOLIO (Han et al., 2022), and ProntoQA (Saparov and He, 2023) which are QA datasets designed to test reasoning ability. ProofWriter provides multihop proofs for each example, while FOLIO gives diverse and complex logical expressions, however, it is only limited to FOL. ProntoQA (Saparov and He, 2023) provides explanation and reasoning steps but is limited to modus ponens in FOL. Nevertheless, several crucial attributes motivated us to create _LogicBench_ (see Table 1 for comparison). Additional datasets for evaluating logical reasoning also exist such as SimpleLogic (Zhang et al., 2023) provides a class of logical reasoning problems, TaxiNLI (Joshi et al., 2020) introduces logical taxonomy in NLI task and RuleBert (Saeed et al., 2021) covers only soft logical rules. In summary, _LogicBench_ evaluates logical reasoning in isolation and provides diverse inference rules and logic types compared to existing datasets. Extended related work is discussed in App. B. 

## **3 LogicBench** 

## **3.1 Logics Types** 

**Propositional Logic (PL)** Propositional logic employs a collection of statements or propositions (denoted as _P_ = _p_ 1 _, p_ 2 _, ..., pn_ , where _pi_ represents a proposition) and builds upon them using logical connectives such as ‘ _∧_ ’, ‘ _∨_ ’, ‘ _→_ ’, ‘ _↔_ ’, and ‘ _¬_ ’. Several inference rules for propositional logic have been defined using which given a set of premises, one can derive a sound conclusion. For example, let’s consider two propositions: _p_ 1, which states "It is raining," and _p_ 2, which states "It is cloudy." From these propositions, we can 

construct a context/knowledge base (KB) with two premises: (1) _p_ 1 _→ p_ 2 and (2) _p_ 1. With this KB, we can conclude _p_ 2. This inference rule is written as (( _p_ 1 _→ p_ 2) _∧ p_ 1) _⊢ p_ 2 and is known as ‘Modus Ponens’. In our study, we explore nine distinct inference rules of propositional logic, extensions of seven of them with one-variable and a universal quantifier, and two axioms of first-order logic as shown in Table 2. These inference rules provide a proper framework for deriving valid conclusions. 

**First-order Logic (FOL)** In this work, we consider a restricted set of logical axioms for FOL that utilize quantifiers, _∀_ (universal quantifier) and _∃_ (existential quantifier). The universal quantifier ( _∀_ ) denotes that a statement holds true for all instances within a specific category. In contrast, the existential quantifier ( _∃_ ) indicates that a statement is true for at least one instance within its scope. For instance, a simple extension of propositional ‘Modus Ponens’ is an inference rule where given the premises _∀_ ( _p_ ( _x_ ) _→ q_ ( _x_ )) and _p_ ( _a_ ), we conclude _q_ ( _a_ ) (e.g., given “All kings are greedy” and “Sam is a king”, we can conclude “Sam is greedy”). Here, we explore two axioms (EG and UI - in detail in App. C.3) and various inference rules that incorporate the quantifiers (shown in Table 2). 

**Non-monotonic (NM) Reasoning** In this work, we analyze a range of logical reasoning templates in NM logics involving “Default Reasoning,” “Reasoning about Unknown Expectations,” and “Reasoning about Priorities.” These templates are inspired by the compilation (Lifschitz, 1989) made in 1989 to evaluate the abilities of various non- 

|**Basic Default Reasoning**|**Default Reasoning with Irrelevant Information**|
|---|---|
|Context: Block A and block B are both heavy objects that are typically<br>found on the table. However, there is a possibility that block A might not<br>follow this usual convention. It is important to note this exception.<br>Conclusion: B is on the table.|Context: In a room flled with various objects, two heavy blocks, block A<br>and block B, stand out. Normally, heavy blocks like these are placed on the<br>table, but surprisingly, block A is not found on the table. On the other hand,<br>block B grabs attention with its vibrant red color.<br>Conclusion: B is on the table.|
|**Reasoning about Unknown Expectations**|**Reasoning about Priorities**|
|Context: In this situation, there are three heavy blocks: A, B, and C.<br>Typically, heavy blocks are found on the table. However, it is known that at<br>least one of the blocks, either A or B, is not currently on the table.<br>Conclusion: C is on the table.<br>Exactly one of A, B is not on the<br>table.|Context: John confdently states that the vehicle is situated in the driveway,<br>while Sara adamantly counters, asserting that it is not parked inside the<br>garage.<br>Conclusion: If John’s evidence is more reliable than Sara’s then the<br>car is parked in the driveway.|



Table 3: Illustrative examples of non-monotonic reasoning adapted from (Lifschitz, 1989). 

monotonic logics that were being developed at that time. Below Table 3 shows examples of NM reasoning. Additional examples are given in App. C.4. 

A key aspect of NM logics is to formalize notions such as “normally,” “typically,” and “usually” that are not directly formalizable using classical quantifiers in the first-order setting. The general rule “Heavy blocks are normally located on the table” does not imply that “All heavy blocks are always located on the table”. Rather, this rule allows for exceptions. Our work explores various NM reasoning patterns, as depicted in Figure 1, to delve deeper into the nuances of this type of reasoning. 

## **3.2 Data Creation** 

Our data creation procedure, illustrated in Figure 2, consists of three stages: 

**1. Sentence Generation:** Starting with a given prompt, we generate coherent sentences and their negations that incorporate different ontologies. 

**2. NL Conversion:** Pairs of _(context, question)_ are generated using pre-defined templates from which context is then converted to a natural language narrative using the prompt. 

**3. Task Instance Generation:** Task-specific _(context, question, answer)_ triplets are generated. BQA requires answers in the form of “yes” or “no”, whereas MCQA involves selecting one correct option from a set of four. We generate semantically preserving and inverting variations of these triplets to add more diversity for BQA. 

Examples of generated data corresponding to each logic type and reasoning patterns are presented in App. C. 

## **3.2.1 Sentence Generation** 

Here, the first step is to generate sentences with diverse _ontologies_ . An ontology represents a col- 

lection of concepts (e.g. car, person, animals, etc.) along with their corresponding associated properties. To generate these sentences, we prompt the GPT-3.5 model with instructions tailored for each inference rule (more details in App. A). 

An example of a prompt corresponding to the ‘Modus Tollens’ from PL is presented in App. A for better illustration. Note that our objective at this stage is not to generate logical sentences but rather to generate a diverse and coherent set of sentences that encompass various concepts. We also create a negation sentence corresponding to each generated sentence[2] . In this work, the scope of generating negations is simple (refer to App. C for examples), however, negations can be more complicated in the case of logic. These generated sentences will be combined with logical connectives in a later stage to form context and questions. 

## **3.2.2 NL Conversion** 

Here, the NL conversion is accomplished using two steps. First, we leverage the formal expressions of reasoning patterns to create templates that establish the desired NL formulation for each logical connective (i.e., templatized context). Second, we prompt GPT-3.5 to transform the templatized context into a story/narrative-based context, enhancing its naturalness. For instance, implication: “ _p → q_ ” is expressed as “If _p_ , then _q_ ”, conjunction: “ _p ∧ q_ ” as “ _p_ and _q_ .”, and disjunction: “ _p ∨ q_ ” as “At least one of the following is true: (1) _p_ and (2) _q_ . Note that we do not know which of (1) and (2) is true. It is possible that only (1) is true, or only (2) is true, or both are true.” since understanding the logical implication of ‘or’ when integrated into logical formulations posed challenges to both humans and 

2We use https://github.com/dmlls/negate to generate negated sentences 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models_images/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Stage 1: Sentence Generation Stage 2: NL Conversion Stage 3: Task Instance Generation<br>Prompt Modus Tollens: Task 1: Binary Question-Answer<br>((S1 → S2 )  ∧  ¬S2 )  ⊢  ¬S1<br>Context:  NL story comprising MT<br>Question:  Does this context imply  ¬ S1 ?<br>+ NegatorModule Selected Sentence Pairs <S1 ,¬S1 >, <S2 ,¬S2> Answer:  Yes<br>Context:  NL story comprising MT<br>Template-based<br>Converter Question:  Does this context imply  S1 ?<br>Answer:  No<br>Sentence Pairs<br><S1,  ¬ S1>, <S2,  ¬ S2>, .., <Sn,  ¬ Sn> Context: Question:  If  S Does this context imply  1 ,  then  S2 . We know that  ¬ S ¬ 1S ? 2 . Task 2: MCQA<br>Context:  NL story comprising MT<br>Question:  What would be the most<br>Example 1:  Prompt-based NL appropriate conclusion based on the<br>S:  John exercises regularly. Converter given context?<br>¬ S:  John does not exercise regularly.<br>Example 2:  Option 1:     Option 3: ¬ S1<br>¬ S:S:  Liam didn't finish his work early. Liam finished his work early. Question: Context:  Does this context implyNL story comprising MT ¬ S1 ? Option 2:     Option 4:<br>**----- End of picture text -----**<br>


Figure 2: Schematic representation of three-stage procedure for data creation. NL: Natural Language 

models. With these established formulations, we proceed to utilize the sentences generated in §3.2.1 to create the templatized context and questions corresponding to reasoning patterns. 

Then, the templatized context is converted into a narrative-based context using a prompt-based converter, enhancing its naturalness. The promptbased converter (essentially, prompting GPT-3.5) ensures that the context is no longer templatized yet follows the logical connection between sentences as mentioned in the logical rule (further details are presented in App. D). For instance, let’s consider the “Modus Tollens” from PL ((( _p → q_ ) _∧¬q_ ) _⊢¬p_ ), and the “Bidirectional Dilemma” from FOL ( _∀x_ (( _p_ ( _x_ ) _→ q_ ( _x_ )) _∧_ ( _r_ ( _x_ ) _→ s_ ( _x_ ))) _∧_ ( _p_ ( _a_ ) _∨¬s_ ( _a_ ))) _⊢_ ( _q_ ( _a_ ) _∨¬r_ ( _a_ ))). For these rules, Table 4 presents examples of logical templatized and narrative-based context, question, and task instances for both, BQA and MCQA. App. D showcases further examples corresponding to each inference rule and patterns from _LogicBench_ . 

## **3.2.3 Task Instance Generation** 

After generating the context and questions in §3.2.2, we generate _(context, question, answer)_ triplets for both tasks: (i) BQA, and (ii) MCQA. Here, narrative-based context is similar for both BQA and MCQA tasks, only the format of _(question, answer)_ pairs are different. 

**BQA** We generate semantically preserving and inverting variations of questions. Let’s consider the example of “Modus Tollens” from Table 4, having question as: “Does this imply that Liam didn’t 

finish his work early?” In this question, we observe one proposition: _s_ 1, representing the statement “Liam didn’t finish his work early,” can be used to create another question that did not follow the logical rule MT. We can create two possible tuples: _< ¬s_ 1 _, yes >, < s_ 1 _, no >_ . Each tuple has a question-answer combination using proposition _s_ 1. Moreover, we do not generate variations for the context since it offers no substantial diversity in the dataset. For question variations, we replace the variation of proposition _< s_ 1 _>_ in the original question with the corresponding tuples to add diversity to _LogicBench_ . The process allows us to create more variations of the question for BQA tasks, as illustrated in Figure 2 (Step 3 - Task 1). 

**MCQA** A prompt-based approach is used to create different incorrect options. For rule MT, as shown in Table 4, option 1 is a correct option that follows the rule MT logically ( _< ¬s_ 1 _>_ ), while the other three options are generated using prompting in a way that does not follow the rule being incorrect options. In addition to the options, the question is also replaced by a randomly selected question from the set of five questions. As seen in Figure 2 (Step 3 - Task 2), there is only one correct option out of four given options. More details related to incorrect option generation and instances examples for both tasks are in the App. C, and E.2. 

## **3.3 Statistics and Qualitative Analysis** 

**Statistics** We introduce two versions of our proposed dataset: _LogicBench(Eval)BQA_ and _LogicBench(Eval)MCQA_ . Statistics of both ver- 

|Generated Sentences in Stage 1|Context|Binary QA|MCQA|
|---|---|---|---|
|Inference rule: **MT**<br>p: Liam fnished his work early.<br>¬p: Liam did not fnish his work early.<br>q: he will order pizza for dinner.<br>¬q: he will not order pizza for dinner.|**Templatized Context:** If Liam fnishes his work early, then he will<br>order pizza for dinner. He won’t order pizza for dinner.<br>**NL Context:**<br>Liam knows that if he fnishes his work early<br>for the day, he will order pizza for dinner. However, on this particular<br>day, he decided against ordering pizza.|**Question 1:** Does this imply that Liam didn’t<br>fnish his work early? (Yes)<br>**Question 2:** Does this imply that Liam fnishes<br>his work early? (No)|**Question:** Based on the context, what conclusion would<br>be deemed most suitable?<br>**Option 1:** Liam didn’t fnish his work early. (Yes)<br>**Option 2:** Sarah had already ordered Chinese takeout.<br>(No)<br>**Option 3:** Rebecca fnished her work early. (No)<br>**Option 4:** Liam decided to order sushi instead. (No)|
|Inference rule: **BD**<br>p(x):<br>someone drinks lots of wa-<br>ter<br>q(x): they will feel hydrated<br>r(x): they eat too much sugar<br>s(x): they will experience a sugar crash<br>p(a): Jane drinks lots of water<br>¬p(a): Jane does not drink lots of water<br>q(a): she will feel hydrated<br>¬q(a): she will not feel hydrated<br>r(a): she eats too much sugar<br>¬r(a): she does not eat too much sugar<br>s(a): she will experience a sugar crash<br>¬s(a): she will not experience a sugar<br>crash|**Templatized Context:** If someone drinks lots of water, then they will<br>feel hydrated. If they eat too much sugar, then they will experience a<br>sugar crash. We know that at least one of the following is true (1)<br>Jane drinks lots of water and (2) she won’t experience a sugar crash.<br>Note that we do not know which ones of (1) and (2) are true. It might<br>be the case that only (1) is true, or only (2) is true or both are true.<br>**NL Context:**<br>If someone consumes a signifcant amount of<br>water, they will experience a state of hydration.<br>conversely, if<br>excessive amounts of sugar are ingested by them, a sugar crash will<br>ensue. it is known that at least one of the following statements is true:<br>either the Jane consumes ample water or she will not experience a<br>sugar crash. however, the actual veracity of either statement remains<br>ambiguous, as it could be the case that only the frst statement is true,<br>only the second statement is true, or both statements are true.|**Question 1:**Can we say at least one of the follow-<br>ing must always be true? (a) she will feel hydrated<br>and (b) she doesn’t eat too much sugar (Yes)<br>**Question 2:** Can we say at least one of the fol-<br>lowing must always be true? (a) she won’t feel<br>hydrated and (b) she eats too much sugar (No)<br>**Question 3:**Can we say at least one of the follow-<br>ing must always be true? (a) she will feel hydrated<br>and (b) she eats too much sugar (No)<br>**Question 4:** Can we say at least one of the fol-<br>lowing must always be true? (a) she won’t feel<br>hydrated and (b) she doesn’t eat too much sugar<br>(No)|**Question:** Taking into account the context provided,<br>what conclusion would be the most appropriate?<br>**Option 1:**<br>If Jane consumes ample water, she<br>will experience a sugar crash (No)<br>**Option 2:** John will feel hydrated or he won’t experi-<br>ence a sugar crash (No)<br>**Option 3:** Jane will feel hydrated or she doesn’t eat too<br>much sugar (Yes)<br>**Option 4:** Jane won’t feel hydrated or she will eat too<br>much sugar (No)|



Table 4: Illustrative examples of logical context and questions created using sentences that are generated in the first stage (§3.2.1) for both tasks, i.e., BQA and MCQA. 

sions are presented in Table 5. For _LogicBench(Eval)BQA_ , out of 1520, 520 samples are for ‘yes’ and 1000 samples are for ‘no’ labels. For _LogicBench(Eval)MCQA_ , there are 20 unique samples present for each rule, thus in total 500 unique samples. Furthermore, we synthetically augmented _LogicBench(Eval)_ for training purposes (i.e., _LogicBench(Aug)_ ) which consists of 150 unique data samples for each rule for BQA, resulting in a total of 12908 data samples including variations. 

|**Dataset**|**# of Instances**<br>**per Axiom**|**Total # of**<br>**Instances**|**Total # of Instances**<br>**(Including Variations)**|
|---|---|---|---|
|_LogicBench(Eval)BQA_<br>_LogicBench(Eval)MCQA_|20<br>20|500<br>500|1520<br>500|



Table 5: Statistics of _LogicBench(Eval)_ 

**Data Validation** Throughout the data generation phase of _LogicBench(Eval)_ , the authors conduct a review of the logical formations to ensure they follow the intended logical structure. We examine each narrative for any potential discrepancies, ensuring that they are logically sound and correctly represent the intended relationships between propositions. In addition to the logical formation, we also dedicated considerable effort to eliminating typos and validating the grammar. We also analyze the diversity in terms of different ontology and the logical nature of the _LogicBench(Eval)_ (presented in App. C.1). We mitigate errors encountered during the validation step (presented in App. F). 

## **4 Results and Analysis** 

## **4.1 Experimental Setup** 

**Task Formulation** For BQA, let us consider a set of data instances _Ir,L_ corresponding to the inference rule _r_ and logic type _L_ . In this set, _i[th]_ instance 

is represented as _Ir,L[i]_[=] _[{]_[(] _[c][i][, Q][i]_[)] _[}]_[ where] _[c][i]_[rep-] resents narrative context and _Qi_ = _{q_ 1 _, q_ 2 _, ..., qn}_ represents set of question and its variations corresponding to _i[th]_ instance. As discussed in §3, each context ( _c_ ) represents logical rules (e.g., All cats have fur. Tom is a cat.) and question ( _q_ ) represents the conclusion (e.g., Does Tom have fur?). To each context and question pair, i.e., _< c, q >_ , we assign a label from the set _Y_ = _{Y es, No}_ . We assign a label _Y es_ if the conclusion logically entails the context, otherwise, assign a label _No_ . To evaluate any LLMs on this setup, we provide _< p, c, q >_ as input to predict a label from _Y_ where _p_ is a natural language prompt. In the set _Ir,L_ for MCQA, _i[th]_ instance is represented as _Ir,L[i]_[=] _[ {]_[(] _[c][i][, q][i][, O][i]_[)] _[}]_ where _ci_ represents narrative context and _qi_ represents question and _Oi_ = _{o_ 1 _, o_ 2 _, o_ 3 _, o_ 4 _}_ represents four option choices. To each context and question pair, i.e., _< c, q >_ , we assign a label from the set _Y_ = _{o_ 1 _, o_ 2 _, o_ 3 _, o_ 4 _}_ . We assign a label _o_ 1 if the correct conclusion is presented in the first option, and likewise for other labels. To evaluate any LLMs on this setup, we provide _< p, c, q, o >_ as input to predict a label from _Y_ . 

**Experiments** We evaluate a range of prompting models including GPT-4, ChatGPT (GPT-3.5Turbo), Google Gemini-Pro, Llama-2-7B-Chat, and Mistral-7B-Instruct-v0.2. Each model is evaluated in a zero-shot setting where the chain-ofthought prompt is provided to the model without any in-context examples. This approach allows us to determine LLM’s inherent ability to do logical reasoning (based on pre-training), as we can not expect that various logical inference rules/patterns will always be made part of prompts. However, we 

|**Type**<br>**Rules**|**Type**<br>**Rules**|**Llama-2**|**Mistral**|**Gemini**|**ChatGPT**|**GPT-4**|
|---|---|---|---|---|---|---|
|||_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|
|PL|HS<br>DS<br>CD<br>DD<br>BD<br>MT<br>MI<br>CT|1000_._00<br>47_._350_._03<br>44_._810_._05<br>56_._820_._02<br>79_._940_._03<br>25_._470_._01<br>82_._220_._16<br>25_._220_._01<br>48_._890_._43<br>25_._130_._01<br>70_._540_._03<br>71_._460_._05<br>78_._570_._26<br>25_._340_._01<br>70_._000_._12<br>24_._980_._02|98_._630_._01<br>64_._240_._07<br>50_._770_._03<br>77_._660_._05<br>83_._370_._03<br>72_._300_._12<br>71_._260_._02<br>14_._160_._03<br>79_._680_._01<br>47_._360_._11<br>47_._960_._08<br>44_._960_._10<br>75_._360_._01<br>25_._510_._04<br>87_._380_._01<br>71_._990_._03|99_._310_._01<br>59_._600_._01<br>68_._210_._04<br>91_._650_._04<br>85_._560_._04<br>28_._150_._01<br>75_._480_._04<br>25_._220_._02<br>86_._810_._02<br>29_._510_._01<br>75_._990_._05<br>81_._470_._03<br>74_._690_._06<br>24_._840_._04<br>88_._310_._08<br>35_._110_._07|1000_._00<br>66_._820_._04<br>51_._260_._04<br>80_._710_._09<br>90_._750_._06<br>38_._630_._04<br>71_._280_._05<br>23_._470_._03<br>84_._910_._05<br>33_._470_._05<br>55_._630_._02<br>66_._110_._07<br>81_._600_._02<br>31_._790_._03<br>89_._880_._05<br>43_._330_._03|1000_._00<br>84_._750_._06<br>79_._960_._06<br>1000_._00<br>92_._850_._01<br>66_._120_._05<br>84_._510_._02<br>42_._400_._04<br>87_._860_._04<br>59_._060_._13<br>55_._280_._04<br>59_._050_._07<br>91_._840_._01<br>39_._720_._03<br>98_._590_._01<br>60_._710_._06|
||**Avg**|**71**_._**870**_._**13**<br>**37**_._**720**_._**02**|**74**_._**300**_._**03**<br>**52**_._**270**_._**07**|**81**_._**790**_._**04**<br>**46**_._**940**_._**03**|**78**_._**160**_._**04**<br>**48**_._**040**_._**05**|**86**_._**360**_._**03**<br>**63**_._**980**_._**05**|
|FOL<br>EG<br>UI<br>MP<br>HS<br>DS<br>CD<br>DD<br>BD<br>MT||1000_._00<br>71_._670_._03<br>1000_._00<br>1000_._00<br>85_._680_._06<br>97_._970_._04<br>96_._740_._03<br>96_._750_._03<br>98_._410_._03<br>1000_._00<br>80_._240_._05<br>56_._040_._02<br>85_._240_._04<br>85_._240_._04<br>90_._690_._08<br>90_._290_._04<br>85_._310_._02<br>96_._190_._03<br>90_._160_._002<br>91_._580_._03<br>97_._440_._04<br>73_._010_._04<br>96_._280_._03<br>95_._240_._05<br>98_._250_._03<br>96_._970_._05<br>1000_._00<br>91_._030_._04<br>94_._740_._00<br>1000_._00<br>1000_._00<br>36_._830_._01<br>98_._020_._00<br>65_._000_._05<br>97_._700_._00<br>52_._460_._04<br>95_._140_._00<br>46_._740_._04<br>89_._030_._01<br>61_._210_._01<br>58_._330_._14<br>52_._080_._04<br>61_._690_._03<br>92_._590_._13<br>75_._960_._07<br>85_._820_._07<br>72_._460_._10<br>94_._710_._05<br>63_._340_._04<br>1000_._00<br>1000_._00<br>26_._360_._01<br>82_._210_._02<br>82_._020_._09<br>94_._710_._02<br>33_._620_._02<br>87_._870_._06<br>39_._370_._06<br>92_._650_._01<br>74_._720_._05<br>54_._370_._19<br>23_._360_._01<br>73_._780_._01<br>6_._670_._11<br>82_._700_._02<br>28_._960_._01<br>72_._700_._08<br>26_._420_._11<br>85_._540_._04<br>54_._550_._08<br>92_._310_._13<br>25_._780_._01<br>75_._760_._01<br>44_._440_._10<br>82_._820_._08<br>28_._930_._05<br>85_._500_._03<br>38_._820_._02<br>84_._220_._02<br>66_._670_._10<br>77_._110_._05<br>88_._410_._09<br>69_._010_._06<br>91_._710_._01<br>71_._180_._10<br>89_._580_._04<br>62_._460_._03<br>92_._800_._06<br>64_._690_._04<br>1000_._00|||||
||**Avg**|**84**_._**420**_._**07**<br>**50**_._**390**_._**03**|**82**_._**440**_._**02**<br>**73**_._**660**_._**06**|**86**_._**630**_._**05**<br>**67**_._**170**_._**04**|**84**_._**240**_._**04**<br>**69**_._**200**_._**05**|**84**_._**750**_._**02**<br>**83**_._**190**_._**02**|
|NM<br>DRI<br>DRS<br>DRD<br>DRO<br>RE1<br>RE2<br>RE3<br>RAP||29_._060_._05<br>40_._980_._04<br>61_._510_._05<br>72_._220_._05<br>65_._450_._03<br>78_._500_._02<br>55_._770_._00<br>88_._890_._09<br>81_._290_._05<br>1000_._00<br>66_._560_._03<br>18_._400_._01<br>70_._130_._01<br>6_._200_._01<br>68_._050_._01<br>7_._200_._06<br>69_._060_._01<br>0_._000_._00<br>77_._590_._02<br>39_._320_._06<br>49_._210_._06<br>50_._470_._04<br>77_._640_._02<br>97_._780_._04<br>68_._100_._05<br>97_._430_._04<br>55_._110_._02<br>1000_._00<br>86_._960_._00<br>1000_._00<br>53_._250_._01<br>53_._600_._02<br>68_._300_._04<br>89_._560_._11<br>57_._850_._04<br>74_._030_._08<br>50_._850_._01<br>66_._670_._58<br>55_._560_._00<br>1000_._00<br>94_._070_._06<br>27_._550_._01<br>78_._790_._02<br>44_._390_._03<br>84_._040_._04<br>41_._980_._12<br>75_._820_._00<br>31_._740_._03<br>85_._440_._02<br>1000_._00<br>54_._170_._47<br>53_._620_._05<br>81_._310_._04<br>85_._670_._03<br>58_._480_._03<br>82_._220_._17<br>64_._670_._11<br>60_._680_._02<br>59_._420_._00<br>1000_._00<br>39_._050_._03<br>38_._050_._10<br>78_._610_._04<br>83_._590_._01<br>64_._670_._08<br>79_._680_._02<br>59_._050_._04<br>83_._570_._08<br>83_._120_._02<br>89_._080_._01<br>76_._820_._05<br>67_._220_._03<br>51_._730_._02<br>60_._480_._11<br>75_._580_._04<br>96_._670_._06<br>61_._440_._02<br>92_._160_._03<br>66_._070_._05<br>98_._720_._02|||||
||**Avg**|**57**_._**780**_._**09**<br>**43**_._**740**_._**04**|**71**_._**000**_._**03**<br>**67**_._**490**_._**05**|**67**_._**780**_._**04**<br>**69**_._**710**_._**07**|**61**_._**470**_._**03**<br>**65**_._**460**_._**10**|**74**_._**430**_._**02**<br>**84**_._**750**_._**02**|



Table 6: Evaluation of LLMs in terms of label-wise accuracy on _LogicBench(Eval)BQA_ , where _A_ ( _Y es_ ) and _A_ ( _No_ ) denote the accuracy for the _Y es_ and _No_ labels, respectively. DRI: Default Reasoning with Irrelevant Information, DRS: Default Reasoning with Several Defaults, DRD: Default Reasoning with a Disabled Default, DRO: Default Reasoning in an Open Domain, RE1: Reasoning about Unknown Expectations I, RE2: Reasoning about Unknown Expectations II, RE3: Reasoning about Unknown Expectations III, RAP: Reasoning about Priorities 

do evaluate these models in a few-shot setting, and present the results in App. G. 

**Metrics** Here, we evaluate performance in terms of accuracy for both tasks, BQA and MCQA. For the BQA, we measure accuracy corresponding to each label, i.e., _A_ ( _Y es_ ) and _A_ ( _No_ ). We evaluate each model on three different chain-of-thought prompts and report average results across these prompts. All prompts used for experiments are described in App. H. 

## **4.2 Main Results** 

Table 6 and Table 7 represent inference rule-wise performance corresponding to each LLMs for the BCQ and MCQA tasks, respectively. Specifically, Table 6 provides label-wise accuracy ( _A_ ( _Y es_ ) and _A_ ( _No_ )) for the BQA task, and Table 7 provides overall accuracy for the MCQA task. Both tables provide valuable insights into the performance of different models on various logic types and lead to several interesting findings. From Table 6, we 

can observe that ChatGPT achieves 48.04%, and GPT-4 shows a performance of 63.98% _A_ ( _Y es_ ) on average which indicates the challenge of classical logical reasoning (PL) even for larger LLMs such as ChatGPT and GPT-4. Furthermore, we can observe that models struggle more with inference rules of PL compared to FOL and NM. In addition, it is noticeable that each model performs relatively better on questions with a negative response (i.e., _No_ ) compared to questions with a positive response (i.e., _Y es_ ). This observation suggests that the models struggle to fully comprehend the logical relationship between the context and the conclusion (i.e., lower _A_ ( _Y es_ )). However, they demonstrate a relatively stronger understanding when the relationship is contradictory in nature (i.e., higher _A_ ( _No_ )). From Table 7, we can observe that larger models exhibit superior performance in selecting the correct choice to arrive at a logical conclusion. Interestingly, the performance of models decreases when the inference rules are longer or include nega- 

|**Type**|**Rules**|**Llama-2**|**Mistral**|**Gemini**|**ChatGPT**|**GPT-4**|
|---|---|---|---|---|---|---|
|PL|HS<br>DS<br>CD<br>DD<br>BD<br>MT<br>MI<br>CT|86_._670_._08<br>63_._330_._12<br>80_._000_._05<br>43_._330_._03<br>51_._670_._03<br>31_._670_._03<br>33_._330_._13<br>73_._330_._10|93_._330_._06<br>60_._000_._09<br>70_._000_._13<br>30_._000_._10<br>53_._330_._03<br>60_._000_._05<br>35_._000_._05<br>68_._330_._08|1000_._00<br>86_._670_._10<br>96_._670_._03<br>90_._000_._05<br>86_._670_._03<br>78_._330_._08<br>71_._670_._06<br>1000_._00|91_._670_._03<br>96_._670_._06<br>90_._000_._00<br>73_._330_._06<br>68_._330_._06<br>73_._330_._08<br>63_._330_._08<br>98_._330_._03|1000_._00<br>95_._000_._00<br>1000_._00<br>88_._330_._03<br>83_._330_._03<br>76_._670_._03<br>73_._330_._14<br>1000_._00|
||**Avg**|**57**_._**920**_._**07**|**58**_._**750**_._**07**|**88**_._**750**_._**04**|**81**_._**880**_._**05**|**89**_._**580**_._**03**|
|FOL|EI<br>UI<br>MP<br>HS<br>DS<br>CD<br>DD<br>BD<br>MT|80_._000_._00<br>63_._330_._03<br>85_._000_._09<br>61_._670_._06<br>43_._330_._06<br>75_._000_._05<br>36_._670_._06<br>35_._000_._05<br>41_._670_._03|85_._000_._05<br>75_._000_._05<br>98_._330_._03<br>70_._000_._05<br>36_._670_._03<br>61_._670_._06<br>46_._670_._03<br>43_._330_._06<br>66_._670_._03|95_._000_._00<br>98_._330_._03<br>1000_._00<br>81_._670_._03<br>70_._000_._05<br>93_._330_._03<br>85_._000_._05<br>78_._330_._06<br>81_._670_._06|93_._330_._03<br>91_._670_._03<br>1000_._00<br>73_._330_._06<br>78_._330_._10<br>80_._000_._05<br>71_._670_._06<br>66_._670_._13<br>86_._670_._06|1000_._00<br>98_._330_._03<br>1000_._00<br>76_._670_._03<br>95_._000_._05<br>91_._670_._03<br>93_._330_._03<br>91_._670_._06<br>86_._670_._03|
||**Avg**|**57**_._**960**_._**05**|**64**_._**810**_._**04**|**87**_._**040**_._**03**|**82**_._**410**_._**06**|**91**_._**510**_._**04**|
|NM|DRI<br>DRS<br>DRD<br>DRO<br>RE1<br>RE2<br>RE3<br>RAP|38_._330_._03<br>41_._670_._08<br>55_._000_._00<br>21_._670_._03<br>51_._670_._03<br>65_._000_._05<br>31_._670_._03<br>46_._670_._08|28_._330_._06<br>16_._670_._10<br>50_._000_._05<br>21_._670_._03<br>31_._670_._08<br>75_._000_._00<br>33_._330_._03<br>35_._000_._09|58_._330_._08<br>45_._000_._10<br>48_._330_._03<br>53_._330_._03<br>70_._000_._00<br>68_._330_._08<br>61_._670_._08<br>33_._330_._03|66_._670_._06<br>41_._670_._10<br>71_._670_._10<br>38_._330_._08<br>65_._000_._05<br>61_._670_._03<br>70_._000_._05<br>55_._000_._05|90_._000_._00<br>55_._000_._05<br>80_._000_._05<br>45_._000_._00<br>95_._000_._05<br>66_._670_._06<br>68_._330_._03<br>51_._670_._03|
||**Avg**|**43**_._**960**_._**04**|**36**_._**460**_._**05**|**54**_._**790**_._**05**|**58**_._**750**_._**07**|**68**_._**960**_._**03**|



Table 7: Evaluation of LLMs in terms of accuracy on _LogicBench(Eval)MCQA_ . 

tions. In contrast to Table 6, for MCQA, LLMs show superior performance for PL and FOL compared to NM. To further investigate these findings and provide a detailed analysis, we perform a thorough study of reasoning chains generated by LLMs and present our insights in the subsequent section. 

## **4.3 Analysis and Discussion** 

**Human Performance** We conduct a human evaluation on a subset of LogicBench(Eval) for both tasks, BQA and MCQA. Specifically, we selected 50 unique instances covering all 25 reasoning patterns from LogicBench(Eval). This selection resulted in total instances of 153 <context, question> pairs for BQA, and 50 <context, question, choices> pairs for MCQA. We hired three graduate student volunteers to provide the evaluations. The task instructions given to all three annotators closely resemble the prompts provided to models (App. H). Each instance pair is answered/annotated by three different annotators with 0.785 interannotator agreement (measured with raw/observed agreement) for BQA and 0.813 for MCQA. 

From the results (Table 8) for BQA, we see that humans achieve more than _∼_ 85% accuracy on various logic types on LogicBench(Eval) which indicates the capability of humans to comprehend single-step logical reasoning effectively. From Ta- 

|**Logic Type**|**BQA**|**BQA**|**MCQA**|
|---|---|---|---|
||**A(No)**|**A(Yes)**|**Accuracy**|
|**PL**<br>**FOL**<br>**NM**|85.42%<br>90.74%<br>72.22%|84.17%<br>91.18%<br>88.46%|100%<br>97.03%<br>62.05%|
|**Avg.**|82.79%|87.94%|86.36%|



Table 8: Human performance on three logic types averaged across three annotators for both tasks. 

ble 6, we observe that the average performance of all models is below human performance indicating room for improvement in their reasoning capabilities. From Table 7, we make similar observations for MCQA. However, we can see that the performance of NM for the MCQA task remains a challenge for both humans and LLMs. 

**Lower performance of LLMs on PL as compared to NM and FOL for BQA.** In the development of AI, NM logic was partly developed to formalize natural language constructs, such as “normally birds fly”, that were not formalizable in a straightforward manner using classical mathematical logics. Thus, while it was difficult for researchers to come up with non-monotonic logics and formalize non-monotonic reasoning, the fact that they were usually motivated by natural language examples, suggests that many of the nonmonotonic reasoning aspects are present in the NL text in the wild that is used in the pre-training of the ultra-large LLMs such as GPT4. While from human experience and complexity theory, FOL is harder than PL in general; in the LLM context, the crucial factor becomes what kind of logical sentences LLMs are pre-trained on. It seems that LLMs are pre-trained more on simple FOL sentences than on simple PL sentences (see Appendix I for further discussion). On the other hand, some PL features are perhaps less prevalent in human writing (on which LLMs are pre-trained) - such as Modes Tollens. Table 6 shows that GPT-4 achieves _∼_ 85% accuracy ( _A_ ( _Y es_ )) for simple inference rules such as HS(PL). However, GPT-4 performance dropped to _∼_ 59% _A_ ( _Y es_ ) for PL(MT). 

**Negations are hard to understand when embedded with logical rules.** Regarding PL and FOL, it is apparent that the models struggle more with the DD and MT inference rules. A closer look at Table 2 reveals that all of these rules include examples where the models need to draw conclusions based 

on negated premises. This indicates that the models encounter difficulties when negated premises are introduced. We also analyze the effect of negations on the reasoning chain (see App. I). 

**Longer inference rules are still challenging.** Table 6 indicates that the models face challenges when handling longer rules, such as BD, CD, and DD, both in PL and FOL. Hence, it can be concluded that these models struggle with longer logical dependencies in the premise, particularly when a higher number of propositions are present. In the case of NM reasoning, the models exhibit lower performance in DRS, indicating that a higher number of premises often leads to more frequent mistakes. 

## **LLMs sometimes overlook contextual informa-** 

**tion.** We investigate the LLMs’ logical reasoning ability in natural language, not in artificial logical formulations. Hence, we note that LLMs sometimes hallucinate information and overlook contextual information, leading to incorrect conclusions. To analyze this, we manually examine all the reasoning chains generated for instances sharing the same contexts in both BQA and MCQA tasks. We observe that, although this pattern is not dominant, it affects BQA more than MCQA. For a more indepth analysis, please refer to App. I. 

**Large models are better logical reasoners.** We analyze the results of both smaller (Llama-2-7B and Mistral-7B) and larger (ChatGPT, GPT-4, and Gemini) models. Table 6 and Table 7 show that larger models tend to exhibit higher performance across different types of logic. We further investigate an additional model with an intermediate size: Yi-34B-chat (Young et al., 2024) (results are presented in App. I). When compared to the Llama2-7B, the Yi-34B model (5x larger than Llama-7B) shows improvement in average performance across three logic types. Similarly, GPT-4 outperforms Yi-34B. This suggests that increasing the model size leads to substantial gains in performance, indicating the influence of larger model capacities on carrying out better logical reasoning. 

**Performance of BQA** _**vs.**_ **MCQA** From Table 6 and Table 7, we can see the overall performance of LLMs is higher on _PLMCQA_ compared to _PLBQA_ . Conversely, the performance is lower on _NMMCQA_ compared to _NMBQA_ . For PL, the performance gaps between the CT and DD inference rules primarily contributed to this trend, and 

DRO, RAP, and DRD for NM. We analyze the reasoning chains associated with these inference rules and presented our detailed observations in App. I. 

**Effect on other logic datasets** We trained the T5-large model on the _LogicBench(Aug)_ resulting in a model named LogicT5. Furthermore, we performed fine-tuning on four other logical reasoning datasets: LogiQA, Reclor, LogicNLI, and FOLIO. Further discussion is presented in App. I. 

## **5 Conclusions** 

In this work, we evaluated the logical reasoning ability of LLMs on 25 distinct inference rules and reasoning patterns covering PL, FOL, and NM logics. To this end, we introduced _LogicBench_ , a natural language question-answering dataset focusing on evaluating a single inference rule. We devised two tasks using _LogicBench_ : (i) BQA, and (ii) MCQA. We evaluated a range of LLMs including GPT-4, ChatGPT, Gemini-Pro, Llama-2, and Mistral on both tasks. Experimental results showed that LLMs do not perform well on _LogicBench_ , even though they require the application of only a single inference rule. Furthermore, we also augmented _LogicBench_ to _LogicBench(Aug)_ , which can be utilized for training purposes. Using _LogicBench(Aug)_ , we demonstrated that LLMs trained using it showcase an improved understanding of logical reasoning, resulting in a better performance on existing logic datasets. 

## **Limitations** 

While _LogicBench_ encompasses 25 distinct inference rules spanning three logic types (significantly more than any previous study) to comprehensively evaluate the logical reasoning capabilities of LLMs, it can be further extended by incorporating additional inference rules and logic types. However, with respect to first-order logic and logics with quantified variables, there can be an infinite number of such rules. In this study, we focused solely on evaluating model performance using a single inference rule; however, an interesting future direction can be enhancing the depth of reasoning complexity (i.e., multi-step reasoning) by incorporating combinations of inference rules to derive conclusions. We also note that this research is limited to the English language and can be extended to multilingual scenarios for evaluating the logical reasoning ability of LLMs. 

## **Ethics Statement** 

We have used AI assistants (Grammarly and ChatGPT) to address the grammatical errors and rephrase the sentences. 

## **Acknowledgement** 

We thank the anonymous reviewers for constructive suggestions, and the computer science graduate students of Arizona State University (ASU) who helped with the human annotations. We extend our gratitude to the Research Computing (RC) at ASU for providing computing resources for experiments. We acknowledge support by a 2023 Spring Amazon Research Award (ARA), and an award by Cisco via Silicon Valley Foundation. 

## **References** 

Jean-Philippe Bernardy and Stergios Chatzikyriakidis. 2020. Fracas: Temporal analysis. 

- Sajjad Beygi, Maryam Fazel-Zarandi, Alessandra Cervone, Prakash Krishnan, and Siddhartha Jonnalagadda. 2022. Logical reasoning for task oriented dialogue systems. In _Proceedings of the Fifth Workshop on e-Commerce and NLP (ECNLP 5)_ , pages 68–79, Dublin, Ireland. Association for Computational Linguistics. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In _Advances in Neural Information Processing Systems_ , volume 33, pages 1877–1901. Curran Associates, Inc. 

- Xiang Chen, Xin Xie, Ningyu Zhang, Jiahuan Yan, Shumin Deng, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. 2021. Adaprompt: Adaptive promptbased finetuning for relation extraction. _arXiv e- prints_ , pages arXiv–2104. 

- Peter Clark, Oyvind Tafjord, and Kyle Richardson. 2021. Transformers as soft reasoners over language. In _Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence_ , pages 3882–3890. 

- Leyang Cui, Yu Wu, Jian Liu, Sen Yang, and Yue Zhang. 2021. Template-based named entity recognition using BART. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 

1835–1845, Online. Association for Computational Linguistics. 

- Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Avia Efrat and Omer Levy. 2020. The turking test: Can language models understand instructions? _arXiv preprint arXiv:2010.11982_ . 

- Himanshu Gupta, Neeraj Varshney, Swaroop Mishra, Kuntal Kumar Pal, Saurabh Arjun Sawant, Kevin Scaria, Siddharth Goyal, and Chitta Baral. 2023. “john is 50 years old, can his son be 65?” evaluating NLP models’ understanding of feasibility. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pages 407–417, Dubrovnik, Croatia. Association for Computational Linguistics. 

- Tanmay Gupta, Amita Kamath, Aniruddha Kembhavi, and Derek Hoiem. 2021. Towards general purpose vision systems. _arXiv preprint arXiv:2104.00743_ . 

- Christopher Hahn, Frederik Schmitt, Jens U. Kreber, Markus Norman Rabe, and Bernd Finkbeiner. 2021. Teaching temporal logics to neural networks. In _International Conference on Learning Representations_ . 

- Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Luke Benson, Lucy Sun, Ekaterina Zubova, Yujie Qiao, Matthew Burtell, et al. 2022. FOLIO: Natural language reasoning with first-order logic. _arXiv preprint arXiv:2209.00840_ . 

- Peter Hase and Mohit Bansal. 2022. When can models learn from explanations? a formal framework for understanding the roles of explanation data. In _Proceedings of the First Workshop on Learning with Natural Language Supervision_ , pages 29–39, Dublin, Ireland. Association for Computational Linguistics. 

- Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. _arXiv preprint arXiv:2310.06825_ . 

- Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. 2020. How can we know what language models know? _Transactions of the Association for Computational Linguistics_ , 8:423–438. 

- Pratik Joshi, Somak Aditya, Aalok Sathe, and Monojit Choudhury. 2020. TaxiNLI: Taking a ride up the NLU hill. In _Proceedings of the 24th Conference on Computational Natural Language Learning_ , pages 41–55, Online. Association for Computational Linguistics. 

- Daniel Khashabi. 2019. _Reasoning-Driven QuestionAnswering for Natural Language Understanding_ . University of Pennsylvania. 

- Kirby Kuznia, Swaroop Mishra, Mihir Parmar, and Chitta Baral. 2022. Less is more: Summary of long instructions is better for program synthesis. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 4532– 4552, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Teven Le Scao and Alexander M Rush. 2021. How many data points is a prompt worth? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2627–2636. 

- Hector J. Levesque, Ernest Davis, and Leora Morgenstern. 2012. The Winograd Schema Challenge. In _Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning_ , KR’12, pages 552–561. AAAI Press, Rome, Italy. 

- Vladimir Lifschitz. 1989. Benchmark problems for formal nonmonotonic reasoning: Version 2.00. In _Non-Monotonic Reasoning: 2nd International Workshop Grassau, FRG, June 13–15, 1988 Proceedings 2_ , pages 202–219. Springer. 

- Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, et al. 2021. Few-shot learning with multilingual language models. _arXiv preprint arXiv:2112.10668_ . 

- Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2021a. Logiqa: a challenge dataset for machine reading comprehension with logical reasoning. In _Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence_ , pages 3622–3628. 

- Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2021b. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. _arXiv preprint arXiv:2107.13586_ . 

- Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics. 

- Man Luo, Sharad Saxena, Swaroop Mishra, Mihir Parmar, and Chitta Baral. 2022. Biotabqa: Instruction learning for biomedical table question answering. _arXiv preprint arXiv:2207.02419_ . 

- Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. 2014. The SICK (Sentences Involving Compositional Knowledge) dataset for relatedness and entailment. 

- R. Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. 

- Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2021. Metaicl: Learning to learn in context. _arXiv preprint arXiv:2110.15943_ . 

- Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. 2022a. Reframing instructional prompts to GPTk’s language. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 589–612, Dublin, Ireland. Association for Computational Linguistics. 

- Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2021. Cross-task generalization via natural language crowdsourcing instructions. _arXiv preprint arXiv:2104.08773_ . 

- Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. 2022b. NumGLUE: A suite of fundamental yet challenging mathematical reasoning tasks. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3505–3523, Dublin, Ireland. Association for Computational Linguistics. 

- Swaroop Mishra and Elnaz Nouri. 2023. HELP ME THINK: A simple prompting strategy for non-experts to create customized content with models. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 11834–11890, Toronto, Canada. Association for Computational Linguistics. 

- Mutsumi Nakamura, Santosh Mashetty, Mihir Parmar, Neeraj Varshney, and Chitta Baral. 2023. LogicAttack: Adversarial attacks for evaluating logical consistency of natural language inference. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pages 13322–13334, Singapore. Association for Computational Linguistics. 

## OpenAI. 2023. Gpt-4 technical report. 

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730–27744. 

- Mihir Parmar, Swaroop Mishra, Mirali Purohit, Man Luo, Murad Mohammad, and Chitta Baral. 2022. InBoXBART: Get instructions into biomedical multitask learning. In _Findings of the Association for Computational Linguistics: NAACL 2022_ , pages 112–128, Seattle, United States. Association for Computational Linguistics. 

- Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 2080–2094, Online. Association for Computational Linguistics. 

- Pruthvi Patel, Swaroop Mishra, Mihir Parmar, and Chitta Baral. 2022. Is a question decomposition unit all we need? In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 4553–4569, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Mohammed Saeed, Naser Ahmadi, Preslav Nakov, and Paolo Papotti. 2021. RuleBERT: Teaching soft rules to pre-trained language models. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 1460–1476, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. _Communications of the ACM_ , 64(9):99–106. 

- Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. 2021. Multitask prompted training enables zero-shot task generalization. _ICLR_ . 

- Abulhair Saparov and He He. 2023. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In _The Eleventh International Conference on Learning Representations_ . 

- Timo Schick and Hinrich Schütze. 2021. Exploiting cloze-questions for few-shot text classification and natural language inference. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pages 255–269, Online. Association for Computational Linguistics. 

- Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. 2019. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 4506–4515, Hong Kong, China. Association for Computational Linguistics. 

- Oyvind Tafjord, Peter Clark, Matt Gardner, Wen-tau Yih, and Ashish Sabharwal. 2019a. Quarel: A dataset and models for answering questions about qualitative relationships. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pages 7063– 7071. 

- Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 3621–3634, Online. Association for Computational Linguistics. 

- Oyvind Tafjord, Matt Gardner, Kevin Lin, and Peter Clark. 2019b. QuaRTz: An open-domain dataset of qualitative relationship questions. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 5941–5946, Hong Kong, China. Association for Computational Linguistics. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Jidong Tian, Yitian Li, Wenqing Chen, Liqiang Xiao, Hao He, and Yaohui Jin. 2021. Diagnosing the firstorder logical reasoning ability through LogicNLI. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 3738–3747, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Bertie Vidgen, Dong Nguyen, Helen Margetts, Patricia Rossini, and Rebekah Tromble. 2021. Cad: the contextual abuse dataset. 

- Liwen Wang, Rumei Li, Yang Yan, Yuanmeng Yan, Sirui Wang, Wei Wu, and Weiran Xu. 2022a. Instructionner: A multi-task instruction-based generative framework for few-shot ner. _arXiv preprint arXiv:2203.03903_ . 

- Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022b. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP 

tasks. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. _ICLR_ . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. _Advances in Neural Information Processing Systems_ , 35:24824–24837. 

- Orion Weller, Nicholas Lourie, Matt Gardner, and Matthew E. Peters. 2020. Learning from task descriptions. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1361–1375, Online. Association for Computational Linguistics. 

- Tongshuang Wu, Michael Terry, and Carrie Jun Cai. 2022. Ai chains: Transparent and controllable human-ai interaction by chaining large language model prompts. In _Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems_ , pages 1–22. 

   - Honghua Zhang, Liunian Harold Li, Tao Meng, Kai-Wei Chang, and Guy Van Den Broeck. 2023. On the paradox of learning to reason from data. In _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence_ , pages 3365–3373. 

   - Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. _arXiv preprint arXiv:2303.18223_ . 

   - Ruiqi Zhong, Kristy Lee, Zheng Zhang, and Dan Klein. 2021. Adapting language models for zero-shot learning by meta-tuning on dataset and prompt collections. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 2856–2878, Punta Cana, Dominican Republic. Association for Computational Linguistics. 

   - Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. 2019. “going on a vacation” takes longer than “going for a walk”: A study of temporal commonsense understanding. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 3363–3369, Hong Kong, China. Association for Computational Linguistics. 

- Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, Kentaro Inui, Satoshi Sekine, Lasha Abzianidze, and Johan Bos. 2019. Can neural networks understand monotonicity reasoning? In _Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_ , pages 31–40, Florence, Italy. Association for Computational Linguistics. 

- Qinyuan Ye and Xiang Ren. 2021. Learning to generate task-specific adapters from task description. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)_ , pages 646–653, Online. Association for Computational Linguistics. 

- Wenpeng Yin, Jamaal Hay, and Dan Roth. 2019. Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 3914–3923, Hong Kong, China. Association for Computational Linguistics. 

- Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. 2024. Yi: Open foundation models by 01. ai. _arXiv preprint arXiv:2403.04652_ . 

- Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2020. Reclor: A reading comprehension dataset requiring logical reasoning. In _International Conference on Learning Representations_ . 

## **A Example Prompt for Sentence Generation** 

Below is the general prompt structure prompted to generate data. The prompt schema, as depicted in Figure 3, comprise three crucial components: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models_images/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models.pdf-0014-02.png)


**----- Start of picture text -----**<br>
Definition<br>Examples<br>< sentences, context, question ><br>< sentences, context, question ><br>< sentences, context, question ><br>Instruction for Formatting<br>**----- End of picture text -----**<br>


Figure 3: Schematic representation of prompt. 

_**Definition**_ provides a detailed explanation of the task and offers a natural language representation of the reasoning pattern for which we are generating sentences. 

_**Examples**_ provide sample sentences that need to be generated. We also illustrate how these sentences will be utilized in later stages, emphasizing the importance of coherence and the inclusion of relevant ontological concepts. 

_**Format**_ We provide specific formatting instructions to guide the generation of sentences. 

Figure 4 illustrates an example prompt for the inference rule, namely, ‘modus tollens’ from propositional logic (PL). Modus tollens is formally represented as (( _p → q_ ) _∧¬q_ ) _⊢¬p_ , which can be understood in natural language as “If _p_ implies _q_ , and we know _¬q_ , then we can conclude _¬p_ .” In this prompt, the definition provides a comprehensive description of the inference rule in natural language. To encourage the generation of more relevant and coherent sentences, the prompt includes an examples section that demonstrates how the generated sentences will be utilized in a later stage. This serves, as an illustration, to guide GPT-3 in producing suitable outputs. In Figure 4, we present three examples involving sentences _p_ and _q_ , along with their respective contexts and questions. The prompt also includes instructions on how the generated sentences should be formatted. 

## **B Extended Related Work** 

As LLMs such as GPT-4, and Bard continue to evolve rapidly, it becomes increasingly crucial to 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models_images/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models.pdf-0014-10.png)


**----- Start of picture text -----**<br>
Definition of rule:<br>If p then q; not q, therefore not p<br>Instructions:  Understand the below examples to learn<br>the connection between sentences and how they have<br>been used. Use that understanding to generate<br>coherent sentences.<br>Examples:<br>p: Liam finished his work early.<br>q:  He will order pizza for dinner.<br>Context:  If Liam finished his work early, then he will order pizza<br>for dinner. He won't order pizza for dinner<br>Question:  Does this imply that Liam didn't finish his work early?<br>Answer:  yes<br>p: Lisa gets an A in the class.<br>q:  She will reward herself with a trip.<br>Context:  If Lisa gets an A in the class, then she will reward herself<br>with a trip. She won't reward herself with a trip<br>Question:  Does this imply that Lisa gets A in the class?<br>Answer:  no<br>p: Benjamin is going to the beach.<br>q:  He will go swimming.<br>Context:  If Benjamin is going to the beach, then he will go<br>swimming. He won't go swimming<br>Question:  Does this imply that Benjamin isn't going to the beach?<br>Answer:  yes<br>Instruction for Formatting<br>1. Generate only one pair of p, and q based on the above<br>understanding.<br>2. Generate each sentence in a new line.<br>3. Do not generate p, and q prefixes.<br>**----- End of picture text -----**<br>


Figure 4: Example prompt for _Modus Tollens_ inference rule from PL. 

evaluate their diverse language capabilities, as well as those of forthcoming LLMs. Recently, many datasets have been created that evaluate different language understanding skills such as pronoun resolution (Sakaguchi et al., 2021; Levesque et al., 2012), commonsense reasoning (Talmor et al., 2019), numerical reasoning (Dua et al., 2019; Patel et al., 2021; Mishra et al., 2022b), qualitative reasoning (Tafjord et al., 2019b,a), temporal reasoning (Zhou et al., 2019), and feasibility reasoning (Gupta et al., 2023). Now, we present the advancements in prompt and instruction tuning using LLMs. 

**Prompt Learning** The introduction of LLMs has significantly shifted the research trend in NLP to prompt-based learning methodologies (Liu et al., 2021b). Many studies have been conducted to investigate the efficacy of prompt-based learning in various applications including Text classification (Yin et al., 2019), Natural Language Inference 

(NLI) (Schick and Schütze, 2021), and Question Answering (QA) (Jiang et al., 2020), Information Extraction (IE) (Chen et al., 2021; Cui et al., 2021), to name a few. In a recent development, the T0 model employs prompts to achieve zero-shot generalization across various NLP tasks (Sanh et al., 2021). Le Scao and Rush (2021) suggested that the use of prompts could be as valuable as hundreds of data points on average. 

**Instruction Learning** Efrat and Levy (2020) was focused on whether existing LLMs understand instructions. The same work in the field of instruction by Hase and Bansal (2022); Ye and Ren (2021); Gupta et al. (2021); Zhong et al. (2021) has been proposed to show that models follow natural language instructions. In addition, Weller et al. (2020) developed a framework focusing on NLP systems that solve challenging new tasks based on their description. Mishra et al. (2021) have proposed natural language instructions for cross-task generalization of LLMs. Similarly, PromptSource (Sanh et al., 2021) and FLAN (Wei et al., 2021) were built for leveraging instructions and achieving zero-shot generalization on unseen tasks. Moreover, Parmar et al. (2022) shows the effectiveness of instructions in multi-task settings for the biomedical/clinical domain. Furthermore, Mishra et al. (2022a) discussed the impact of task instruction reframing. Min et al. (2021) introduced a framework to better understand in-context learning. Ouyang et al. (2022) proposed the InstructGPT model that is fine-tuned with human feedback to follow instructions. Wang et al. (2022a) has developed an instruction-based multitask framework for few-shot Named Entity Recognition (NER) tasks. In addition, many approaches have been proposed to improve model performance using instructions (Wu et al., 2022; Lin et al., 2021; Wang et al., 2022b; Luo et al., 2022; Kuznia et al., 2022; Patel et al., 2022; Mishra and Nouri, 2023). 

**Logic and NLI Datasets** FraCas (Bernardy and Chatzikyriakidis, 2020) offers a unique approach to temporal semantics by converting syntax trees into logical formulas tailored for inference, emphasizing temporal elements such as references, adverbs, aspectual classes, and progressives. The Monotonicity Entailment Dataset (MED) (Yanaka et al., 2019) dives deep into monotonicity reasoning within NLI, probing the synergy between lexical and syntactic structures and spotlighting inherent challenges in both upward and downward monotonic reasoning trajectories. The SICK 

(Marelli et al., 2014) dataset, with its foundation in 10,000 English sentence pairs, is designed to rigorously evaluate semantic relatedness and entailment, leveraging crowdsourced annotations for precision. HANS, or Heuristic Analysis for NLI Systems (McCoy et al., 2019), stands out by rigorously scrutinizing the dependability of NLI models, putting the spotlight on potential pitfalls tied to syntactic heuristics such as lexical overlap. Lastly, CAD (Vidgen et al., 2021) introduces a meticulously crafted dataset from Reddit entries, targeting the detection of online abuse. Nakamura et al. (2023) introduced LogicAttack, a method for performing adversarial attacks on NLI models using PL inference rules and proposed dataset of _∼_ 9 _k_ attack samples derived from the SNLI dataset. In contrast to these works, _LogicBench_ evaluates the logical reasoning capabilities of LLMs beyond the NLI task, focusing on the application of individual inference rules. 

## **C Examples of Data Instances** 

This section provides examples of _(context, question, answer)_ tuples corresponding to each inference rule and reasoning pattern. Additionally, it highlights the diverse range of question variations within the dataset associated with each inference rule and reasoning pattern. 

## **C.1 Word Cloud** 

Figure 5 provides a word cloud derived from the _LogicBench_ ( _Eval_ ). This word cloud highlights the logical nature and diversity of our evaluation dataset. Words such as ‘if’, ‘normally’, ‘usually’, and ‘then’ are prominently featured, suggesting their frequent use in the dataset, and suggesting the logical nature of the dataset. Moreover, we can also observe several words consisting of different ontologies such as ‘cat’, ‘car’, ‘garden’, and many more, suggesting diversity in the dataset. 

## **C.2 Propositional Logic (PL)** 

Here, we discuss examples of each inference rule present in the PL of the _LogicBench_ as shown in Table 9. Table 9 has context related to the inference rule and different variations of the question according to the rule. For instance, the first row of Table 9 shows the example for inference rule, _Hypothetical Syllogism (HS)_ , formally expressed as (( _p → q_ )) _∧_ ( _q → r_ )) _⊢_ ( _p → r_ ). The context represents the premise, i.e., (( _p → q_ )) _∧_ ( _q → r_ )), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models_images/acl_2024_LogicBench_Towards_Systematic_Evaluation_of_Logical_Reasoning_Ability_of_Large_Language_Models.pdf-0016-00.png)


Figure 5: Word cloud of context present in the LB(eval) 

and the first question (Q1) represents the conclusion, i.e., _p → r_ . Hence, Q1 is labeled as "Yes" since it supports the conclusion given the logical context. Furthermore, Q2 to Q4 represent different variations of the question by utilizing the variables ( _p, ¬p, r, ¬r_ ). For the HS, given the provided context, Q2 to Q4 contain the variations _¬p → r_ , _p →¬r_ , and _¬p →¬r_ , respectively, and are labeled as "No" since they do not support the conclusion. 

## **C.3 First-Order Logic (FOL)** 

Here, we discuss examples of each inference rule and two axioms (i.e., Existential Instantiation and Universal Instantiation) present in the FOL from the _LogicBench_ as shown in Table 10. _Existential Generalization (EG)_ , formally expressed as _P_ ( _a_ ) _⇒∃xP_ ( _x_ ) indicates that there is an element _a_ in the domain for which _P_ ( _a_ ) is true, then we know that _∃xP_ ( _x_ ) is true. _Universal Instantiation_ formally expressed as _∀x A ⇒ A{x �→ a}_ indicates that a statement holds true for all instances (x) within a specific category _A_ , hence it is also true for specific instance _a_ . 

Table 10 represents context related to the inference rule and variations of the question. The process of generating data instances for FOL follows a similar approach to that of PL. For example, the first row of Table 10 shows the example for axiom, _Existential Instantiation (EI)_ , formally expressed as _∃xP_ ( _x_ ) _⇒ P_ ( _a_ ). The context represents the initial premise _∃xP_ ( _x_ ) and the first question (Q1) represents the conclusion, i.e., _P_ ( _a_ ). Hence, Q1 is labeled as "Yes" since it supports the conclusion given the logical context. Furthermore, we generate the only variant of the question based on _¬P_ ( _a_ ) and labeled it as _No_ since it does not support the conclusion. 

## **C.4 Non-Monotonic (NM) Reasoning** 

Here, we discuss examples of each reasoning pattern present in the NM reasoning from the _LogicBench_ as shown in Table 11. Table 11 has context related to the reasoning pattern and different variants of the question. For example, the first row of Table 11 shows the example for _Default Reasoning with Irrelevant Information (DRI)_ . For this reasoning, based on the given context, there are also two possible variations of the question where one with a correct conclusion labeled as _Y es_ and another with an incorrect conclusion labeled as _No_ . 

## **D Examples of NL Conversion** 

This section illustrates the way natural language logical context and questions are created using the generated sentence in Stage 1 in addition to prompt-based templatized-context to narrative conversion. Table 13 shows examples of how context and question are generated from sentences corresponding to each inference rule for PL and FOL. Similarly, Table 14 shows examples of NM reasoning. From Table 13, we can see an example of sentence pairs ( _p, q_ ) and their corresponding negation pairs ( _¬p, ¬q_ ) for the ‘modus tollens’ inference rule from PL. These pairs are utilized to generate templatized logical context which is converted to narrative using a specific prompt[1D] designed for PL. In addition to the narrative, the question and its variations are also created using these sentence pairs. Similarly, in the second row, we have four generic rules with variable _x_ ( _p_ ( _x_ ) _, q_ ( _x_ ) _, r_ ( _x_ ) _, s_ ( _x_ )) and their specific cases (i.e., _x_ = _a_ ), along with their respective negative sentence pairs [( _p_ ( _a_ ) _, ¬p_ ( _a_ )), ( _q_ ( _a_ ) _, ¬q_ ( _a_ )), ( _r_ ( _a_ ) _, ¬r_ ( _a_ )), ( _s_ ( _a_ ) _, ¬s_ ( _a_ ))]. In the same manner, as PL, these examples demonstrate the generation of narrative and questions for the FOL inference rule called ‘Bidirectional Dilemma (BD)’, as shown in Table 13 using specific prompt[2D] designed for FOL. From Table 14, the first row presents an example of narrative and questions generated from a sentence pair for the ‘Default Reasoning with Irrelevant Information (DRI)’ from NM reasoning. In this specific instance, the generated sentences are ( _p, q, r, s, t_ ), and the negation is only required for the sentence _t_ . Therefore, there is a single negation pair ( _t, ¬t_ ), which is used to generate questions specific to the ‘DRI’. Same as PL and FOL, templatized context is also converted to narrative using a prompt[3D] created for NM. 

|Rule|Context|Question|
|---|---|---|
|HS|If Jim cleans his room, he will receive a reward and if he<br>receives a reward, he will use it to buy a new toy. So,<br>Jim decided to tidy up his room, hoping to earn a reward.<br>He diligently gathered his clothes, organized his toys, and<br>dusted every surface. After a few hours of hard work, Jim’s<br>room was spotless.|**Q1:** If Jim cleaned his room, does this imply that he will buy a<br>new toy? (Yes)<br>**Q2:** If Jim didn’t clean his room, does this entail that he won’t<br>buy a new toy? (No)<br>**Q3:** If Jim cleaned his room, does this imply that he won’t buy a<br>new toy? (No)<br>**Q4:** If Jim didn’t clean his room, does this imply that he will buy<br>a new toy? (No)|
|DS|Either Chloe is studying for her exams, or Mila is going<br>on vacation, or both scenarios are unfolding. It is unclear<br>which of the two options is true at this point. However, one<br>thing is certain - Chloe is not studying for her exams.|**Q1:** Does this entail that Mila is going on vacation? (Yes)<br>**Q2:** Does this mean that Mila isn’t going on vacation? (No)|
|CD|If I decide to go for a walk, I will be able to breathe in<br>some fresh air and revitalize myself. On the other hand, if I<br>choose to stay home, I will have the opportunity to enjoy a<br>movie. One thing is certain, either I go for a walk or I stay<br>home. It remains uncertain which of the two options I will<br>ultimately choose. It is entirely possible that I might opt<br>for the walk, or perhaps I will fnd myself drawn to staying<br>home, or even both possibilities might come to fruition.|Can we say at least one of the following must always be true?<br>**Q1:** (a) I will get some fresh air or (b) I will watch a movie (Yes)<br>**Q2:** (a) I won’t get some fresh air and (b) I will watch a movie<br>(No)<br>**Q3:** (a) I will get some fresh air and (b) I won’t watch a movie<br>(No)<br>**Q4:** (a) I won’t get some fresh air and (b) I won’t watch a movie<br>(No)|
|DD|If I decide to order takeout, it means I will save time. On<br>the other hand, if I choose to cook a meal, it means I will<br>save money. The interesting thing is that I am in a situation<br>where I won’t be able to save time or money. It is uncertain<br>whether I won’t save time or I won’t save money, or it could<br>even be both scenarios. The only thing that is clear is that<br>at least one of these possibilities is true.|Can we say at least one of the following must always be true?<br>**Q1:** (a) I don’t order takeout or (b) I don’t cook a meal (Yes)<br>**Q2:** (a) I order takeout and (b) I cook a meal (No)<br>**Q3:** (a) I don’t order takeout and (b) I cook a meal (No)<br>**Q4:** (a) I order takeout and (b) I don’t cook a meal (No)|
|BD|If it is sunny outside, then I will go for a walk. However, if<br>it rains, I will stay inside. Currently, it is uncertain whether<br>it is raining or not, but I do know that at least one of the<br>following is true: either it is raining or I will not go for a<br>walk. It is possible that only one of these statements is true,<br>or perhaps both are true.|Can we say at least one of the following must always be true?<br>**Q1:** (a) we will stay inside or (b) it is not sunny (Yes)<br>**Q2:** (a) we will not stay inside and (b) it is sunny (No)<br>**Q3:** (a) we will stay inside and (b) it is sunny (No)<br>**Q4:** (a) we will not stay inside and (b)it is not sunny (No)|
|MT|If Mason decides to leave his job, he will not receive any<br>salary. However, against all odds, Mason still receives his<br>salary. He fnds himself receiving his regular paycheck.|**Q1:** Does this infer that Mason didn’t leave his job? (Yes)<br>**Q2:** Does this infer that Mason left his job? (No)|
|MI|Rohan woke up in the morning and realized that he had<br>forgotten his lunch. Knowing that if he forgets his lunch, he<br>will not eat at school, he felt disappointed.|Based on context, can we say, at least one of the following must<br>always be true?<br>**Q1:** (a) Rohan didn’t forget his lunch and (b) he will not eat at<br>school (Yes)<br>**Q2:** (a) Rohan forgot his lunch and (b) he will eat at school (No)<br>**Q3:** (a) Rohan forgot his lunch and (b) he will not eat at school<br>(No)<br>**Q4:** (a) Rohan didn’t forget his lunch and (b) he will eat at school<br>(No)|
|CT|At least one of two things is true about Tom - he is either<br>an avid reader or he devours books of all genres. We are<br>unsure which one of these statements is true or if both are<br>true. It could be that only the frst statement is true, or only<br>the second statement is true, or even that both are true.|Can we say at least one of the following must always be true?<br>**Q1:** (a) he devours books of all genres or (b) Tom is an avid reader<br>(Yes)<br>**Q2:** (a) he doesn’t devour books of all genres and (b) Tom is an<br>avid reader (No)<br>**Q3:** (a) he devours books of all genres and (b) Tom isn’t an avid<br>reader (No)<br>**Q4:** (a) he doesn’t devour books of all genres and (b) Tom isn’t<br>an avid reader (No)|



Table 9: Examples of context and question-answer pairs for each rule of _Proportional logic_ from the LogicBench; HS: Hypothetical Syllogism, DS: Disjunctive Syllogism, CD: Constructive Dilemma, DD: Destructive Dilemma, BD: Bidirectional Dilemma, MT: Modus Tollens, MI: Material Implication, CT: Commutation 

|Rule|Context|Question|
|---|---|---|
|UI|All students are required to take an examination in order<br>to fulfll the requirements for their degree. Reema, being a<br>student, is also expected to fulfll the requirements.|**Q1:** Does Reema need to take an exam to complete her degree?<br>(Yes)<br>**Q2:**Does Reema need not to take an exam to complete her degree?<br>(No)|
|EG|The marathon race was won by James, who emerged as the<br>champion.|**Q1:** Does this imply that someone won the marathon race? (Yes)<br>**Q2:** Does this mean that no one won the marathon race? (No)|
|MP|If someone is extremely tired, then they will seek some rest<br>and relaxation. Today, Jack fnds himself utterly exhausted.|**Q1:** Does this entail that he will take rest? (Yes)<br>**Q2:** Does this entail that he won’t take rest? (No)|
|HS|If all the necessary supplies have been purchased by some-<br>one, then they can initiate the project. Once the project<br>is started by someone, they will complete it within the ex-<br>pected time-frame.|**Q1:** If Lily bought all the necessary supplies, does this mean that<br>she will fnish it on time? (Yes)<br>**Q2:** If Lily didn’t buy all the necessary supplies, does this imply<br>that she won’t fnish it on time? (No)<br>**Q3:** If Lily bought all the necessary supplies, does this entail that<br>she won’t fnish it on time? (No)<br>**Q4:** If Lily didn’t buy all the necessary supplies, does this imply<br>that she will fnish it on time? (No)|
|DS|It is known that one of the following options is true: some-<br>one goes to a museum or someone visits a park. The specifc<br>scenario could involve only the option to go to a museum<br>being true, or only the option to visit a park being true, or<br>both options being true. However, it is stated that Jill is<br>unable to go to a museum.|**Q1:** Does this imply that she can visit a park? (Yes)<br>**Q2:** Does this entail that she can’t visit a park? (No)|
|CD|If someone is painting a picture, then they will frame it.<br>similarly, the natural course of action for a writer would be<br>to publish their completed story. in this scenario, it is certain<br>that at least one of the following statements holds true: (1)<br>john is currently engrossed in painting a picture, or (2) john<br>is currently immersed in the act of writing a story. it should<br>be emphasized that we are unaware of which statement<br>specifcally applies, as there is a possibility that either (1)<br>alone is true, or (2) alone is true, or even that both (1) and<br>(2) are simultaneously true.|Can we say at least one of the following must always be true?<br>**Q1:** (a) he will frame it and (b) he will publish it. (Yes)<br>**Q2:** (a) he won’t frame it and (b) he will publish it. (No)<br>**Q3:** (a) he will frame it and (b) he won’t publish it. (No)<br>**Q4:** (a) he won’t frame it and (b) he won’t publish it. (No)|
|DD|If someone is taking care of their health, then they will be<br>ft. However, indulging in unhealthy habits can make indi-<br>viduals susceptible to various diseases. The truth is, we can<br>be certain about at least one of the following possibilities:<br>either Jenny won’t be ft and healthy, or she won’t be prone<br>to diseases. It is important to note that we are unaware of<br>which statement is accurate. It could be the case that only<br>the frst statement is true, only the second statement is true,<br>or both statements hold validity.|Can we say at least one of the following must always be true?<br>**Q1:** (a) Jenny doesn’t take care of her health and (b) she doesn’t<br>indulge in unhealthy habits (Yes)<br>**Q2:** (a) Jenny takes care of her health and (b) she indulges in<br>unhealthy habits (No)<br>**Q3:** (a) Jenny doesn’t take care of her health and (b) she indulges<br>in unhealthy habits (No)<br>**Q4:** (a) Jenny takes care of her health and (b) she doesn’t indulge<br>in unhealthy habits (No)|
|BD|If an individual consumes a signifcant amount of water,<br>they will experience a state of hydration. Conversely, if<br>excessive amounts of sugar are ingested, a sugar crash will<br>ensue. It is known that at least one of the following state-<br>ments is true: either the Jane consumes ample water or she<br>will not experience a sugar crash. However, the actual ve-<br>racity of either statement remains ambiguous, as it could be<br>the case that only the frst statement is true, only the second<br>statement is true, or both statements are true.|Can we say at least one of the following must always be true?<br>**Q1:** (a) she will feel hydrated and (b) she doesn’t eat too much<br>sugar (Yes)<br>**Q2:** (a) she won’t feel hydrated and (b) she eats too much sugar<br>(No)<br>**Q3:** (a) she will feel hydrated and (b) she eats too much sugar<br>(No)<br>**Q4:** (a) she won’t feel hydrated and (b) she doesn’t eat too much<br>sugar (No)|
|MT|If someone decides to go to the park, it is required that they<br>wear a mask. However, in this particular situation, John<br>does not wear a face covering.|**Q1:** Does this imply that John doesn’t visit the park? (Yes)<br>**Q2:** Does this entail that John visits the park? (No)|



Table 10: Examples of context and question-answer pairs for each rule of _First order logic_ from the LogicBench. 

|Rule|Context|Question|
|---|---|---|
|DRI|Once upon a time, in a land flled with animals, there were<br>two popular mammalian creatures, cats and dogs. Mammals<br>typically possessed a coat of fur, which kept them warm<br>and protected. However, cats were an exception to this rule,<br>as their bodies lacked fur. Nonetheless, both cats and dogs<br>were beloved by many for their unique traits. Dogs, known<br>for their loyalty, were particularly cherished by humans.|**Q1:** Does this imply that dogs have fur? (Yes)<br>**Q2:** Does this entail that dogs don’t have fur? (No)|
|DRS|John and Mary were expecting their frst child, flled with<br>the anticipation and excitement that all parents feel. Parents<br>are usually loving and supportive. Parents are normally<br>responsible. However, something seemed amiss in their<br>relationship. Mary, usually affectionate and caring, seemed<br>distant and uninvolved. On the other hand, John, known<br>for his responsible nature, started neglecting his duties and<br>became unreliable.|**Q1:** Does this imply that Mary is responsible and John is loving<br>and supportive? (Yes)<br>**Q2:** Does this entail that Mary isn’t responsible and John is loving<br>and supportive? (No)<br>**Q3:**Does this imply that Mary is responsible and John isn’t loving<br>and supportive? (No)<br>**Q4:** Does this entail that Mary isn’t responsible and John isn’t<br>loving and supportive? (No)|
|DRD|Jenny and Anna are known for their tall stature, which is<br>often associated with playing basketball. However, Anna<br>might be an exception to this norm.|**Q1:** Does this entail that Jenny plays basketball? (Yes)<br>**Q2:** Does this mean that Jenny doesn’t play basketball? (No)|
|DRO|In the bird kingdom, there are many different species that<br>possess unique characteristics. One such species is the<br>hummingbird, known for its ability to hover in mid-air and<br>its vibrant colors. While most birds engage in the annual<br>migration south for the winter, the hummingbird chooses<br>to stay put and brave the cold weather. This decision sets<br>the hummingbird apart from its fellow avian companions,<br>as it relies on its resilience and resourcefulness to survive<br>the harsh conditions.|**Q1:**Does this mean that all other birds than hummingbirds migrate<br>south for the winter? (Yes)<br>**Q2:** Does this mean that all other birds than hummingbirds don’t<br>migrate south for the winter? (No)|
|RE1|In a world where animals are often regarded as intelligent<br>creatures, there is a captivating tale that revolves around<br>cats, dogs, and horses. It is commonly believed that most<br>animals possess a level of intellect. However, there is an<br>intriguing twist to this belief as it is known that either cats<br>or dogs are not considered particularly intelligent. As the<br>story unfolds, we delve into the lives of these remarkable<br>creatures, their interactions, and the unique qualities that<br>each of them possesses.|**Q1:** Does this entail that horses are considered to be intelligent<br>creatures and exactly one of the cats or dogs is not considered<br>intelligent? (Yes)<br>**Q2:** Does this mean that horses aren’t considered to be intelligent<br>creatures and exactly one of cats or dogs is not considered intelli-<br>gent? (No)<br>**Q3:** Does this mean that horses are considered to be intelligent<br>creatures and exactly one of cats or dogs is considered intelligent?<br>(No)<br>**Q4:** Does this implies that horses aren’t considered to be intel-<br>ligent creatures and exactly one of cats or dogs is considered<br>intelligent? (No)|
|RE2|In the realm of cat communication, meowing serves as<br>a fundamental aspect of their vocal repertoire. However,<br>intriguingly enough, there exists a distinct species of cat<br>that deviates from this conventional norm. This peculiar<br>feline defes the expectations associated with its kind by<br>refraining from emitting any meows whatsoever.|**Q1:** Does this entail that exactly one species of cat doesn’t meow?<br>(Yes)<br>**Q2:** Does this imply that exactly one species of cat meows? (No)|
|RE3|In a world where cars were known for having four wheels,<br>it was considered a common fact that wheels typically came<br>equipped with spokes. However, amidst this widespread<br>understanding, there was an exception. At least one wheel<br>defed this norm and stood out from the rest by not having<br>any spokes at all.|**Q1:**Does this imply that cars have four wheels with spokes? (Yes)<br>**Q2:** Does this mean that cars don’t have four wheels with spokes?<br>(No)|
|RAP|In the midst of a heated argument, John adamantly claims<br>that sally was present at the store. However, Jane strongly<br>opposes John’s assertion, insisting that Sally was indeed<br>absent from the store.|**Q1:**If John’s evidence is more reliable than Jane’s, does this mean<br>that Sally was in the store? (Yes)<br>**Q2:**If John’s evidence is more reliable than Jane’s, does this mean<br>that Sally wasn’t in the store? (No)<br>**Q3:** If John’s evidence is less reliable than Jane’s, does this entail<br>that Sally was in the store? (No)<br>**Q4:** If John’s evidence is less reliable than Jane’s, does this imply<br>that Sally wasn’t in the store? (Yes)|



Table 11: Examples of context and question-answer pairs for each rule of _Non-monotonic logic_ from the LogicBench. 

Each prompt is shown below for PL, FOL, and NM templatized context to narrative conversion. 

## **1. Prompt for templatized to narrative conversion for PL(MT):** 

**Rule: Condition:** If p then q; **Situation:** not q 

**Context:** Templatized context Improve the context in human-like language and make story with rephrased sentences. 

**Instructions for generating a good story:** 

1. While generating a story, Use the rephrased sentences from the context of the story. 

2. Make sure to include sentences corresponding to the condition and situation from the rule in the story. 

3. Do not add any other extra information. 

4. To generate a story, DO NOT change the name of the main character from the context, if there’s any. If the main character within the context is ’I’, please generate a story with ’I’ as the main character. 

5. Generate only one paragraph with rephrased sentences. 

## **2. Prompt for templatized to narrative conversion for FOL:** 

**Context:** Templatized context Improve the context in human-like language and make a rephrase sentences around it. 

**Instructions for generating a good story:** 

1. Make sure to include only rephrased sentences in the story. 

2. Do not add any other extra information and do not reach any conclusion based on context. 

3. To generate a story, use only generic sentences without mentioning any main character. 

4. Extract a specific case sentence from the context, rephrase it, and append it at the end of the context only. 

5. Generate only one paragraph with rephrased sentences. 

6. DO NOT add any prefixes. 

## **3. Prompt for templatized to narrative conversion for NM:** 

**Context:** Templatized context 

- Improve the context in human-like language and rephrase sentences around it. 

## **Instructions for generating a good story:** 

1. Make sure to include only rephrased sentences in the story. 

2. Do not add any other extra information, but add some story-like context. 

3. To generate a story, DO NOT change the name of the main character from the context, if there’s any. Use the same names/words for main character. If the main character 

- within the context is ’I’, please generate a story with ’I’ as the main character. 

4. Generate only one paragraph with rephrased sentences. 

To ensure the quality of the narrative in the _LogicBench(Eval)_ for task 1 and task 2, we have created category-specific prompts to convert the templatized context to convert more human-like narrative. In total, we have created three different prompts each for PL[1D], FOL[2D], and NM[3D. The prompts are designed to ensure that the logical connection is established in the narrative. Each rule of PL, FOL, and NM has a unique logical progression that should be followed in the narrative is the motivation for us to go with three different instruction-based prompts. In the prompt, used for PL rules, the logical rule is also mentioned in two parts "Condition" and "Situation". Consider the example of logical rule MT - "If p then q; not q; therefore not p", the templatized context will have "If p then q; not q;" so condition will be the first part "If p then q" while the situation will be the second part "not q". In other terms, the specific conditions are the rules to be followed while the situation is the case present in the logical rule. In contrast, for FOL, we do not need to focus more on the specific condition rather we have to make sure a generalized case is present in the narrative with 1-2 specific sentences related to the rule. For NM, we do not have such a rule instead we have a logical connection between sentences and hence we go with the instruction-based prompting. 

## **E Task Instance Generation** 

This section discusses the task instance generation step, which is the last step in Fig. 2 in detail. The overall section is comprised of two subsections for BQA and MCQA. The subsection related to BQA 

discusses the variation generation step in detail with question generation while the MCQA subsection provides details of prompts used for incorrect option generation along with five pre-defined sets of questions. 

## **E.1 Task 1- BQA (variation generation)** 

As seen in Fig. 2, BQA has a narrative along with question, and answer pairs with variation in question and corresponding answer. As discussed in the main paper Data creation3.2, the narrative is created using the pre-defined rules for PL, FOL, and NM while a question is asked based on what can be entailed from the given context. For example, HS can be defined as "If p then q; if q then r; therefore, if p then r" and the narrative will have "If p then q; if q then r" and the question is asked as "Can we conclude if p then r?" having answer as "yes". Now, other variations in the question are asked in the following ways by negating the sentence (p and r) comb: Variation 1: "Can we conclude if _¬_ p then r?", Variation 2: "Can we conclude if p then _¬_ r?", Variation 3: "Can we conclude if _¬_ p then _¬_ r?". If there’s only one axiom (p) present in the question then there are only a 2 variations that can be made _¬_ asking about (p) and ( p). 

## **E.2 Task 2-MCQA (Question selection and incorrect option generation)** 

We have MCQA as task 2 in LogicBench. In this, we have one correct option from the four options, and three options are incorrectly generated using prompting. The question is a bit different from the BQA question formation as we have MCQA and have to identify which conclusion can be derived from the narrative. The question is randomly selected from the pre-defined set of questions. Here, the five different questions are as follows: 

1. What would be the most appropriate conclusion based on the given context? 

2. Considering the provided context, what conclusion would be deemed most suitable? 

3. In light of the context provided, what conclusion can be considered the most appropriate? 

4. Based on the context, what conclusion would be deemed most suitable? 

5. Taking into account the context provided, what conclusion would be most appropriate? 

The narrative for MCQA is the same as the BQA’s narrative and there’s no change. Comparing BQA methodology with MCQA, MCQA’s correct option is the question asked in the BQA which can be concluded from the logical rule present in the narrative. For example, HS can be defined as "If p then q; if q then r; therefore, if p then r" and the narrative will have "If p then q; if q then r" and the question is randomly chosen from the set of questions as mentioned above and the correct option from the different multiple choice will be "if p then r". Based on the information present, we have generated an incorrect option using prompting[4E.2]. Refer this link for more examples - https://anonymous.4open. science/r/LogicBench-EEBB. 

## **4. Prompt for incorrect option generation:** 

**Context:** Narrative **Question:** Randomly chosen question **Correct option:** Conclusion from narrative 

Create three more irrelevant, incorrect options to create multiple choice. Sometime use different persons name as well in creating incorrect options. **Instruction to generate options:** 1. Generate only three unique choices each in new line. 2. Do not use any prefix while generating. 

## **F Mitigation of Errors in** _**LogicBench**_ 

While validating _LogicBench(Eval)_ , we encountered errors within the synthetically generated narrative-based context. We mitigate these errors manually, categorizing them into two groups: (i) eliminating leakage of logical conclusions and (ii) ensuring the inclusion of intended logical premises. In the first category, we found _∼_ 15% of the narrative-based contexts (out of 500 total instances) were found to explicitly present the logical conclusion as a response to the question, bypassing the logical derivation process. This enables the model to extract the final logical conclusion from the context rather than derive it logically. In the second category, we found _∼_ 8% of the narrative-based contexts (out of 500 total instances) where narration lacked some necessary premise sentences crucial for reaching a logical conclusion. To address this issue, we manually incorporated those sentences to ensure the quality of the data. For the MCQA task, during the generation of three incor- 

rect options, we found instances where the model produces two semantically similar options, resulting in the creation of ambiguous choices in _∼_ 11% of the cases (out of a total of 500 instances). We manually mitigated all these errors from the data instances ensuring the high quality of our validation data. We believe that these two versions aim to accommodate different evaluations to explore the logical reasoning capabilities of LLMs. 

## **G Few-shot Experiments** 

This section discusses the performance of the different LLMs in a few-shot setting on the _LogicBench(Eval)BQA_ . Here, we only present a case study on the BQA task. For the fair comparison with Table 6, we analyze an average performance across _A_ ( _Y es_ ). Table 15 shows the performance for each inference rule and reasoning patterns achieved by Llama-2, Mistral, Gemini, ChatGPT, and GPT-4. 

As suggested in (Lu et al., 2022), prompting models are sensitive to in-context examples. Hence, we see mixed performance in Table 15 across all models. From Table 15, we can observe that incontext examples are helpful for Llama-2 since it consistently outperforms zero-shot baselines by large margins in terms of _A_ ( _Y es_ ). Llama-2 is remarkably good at following the in-context exemplars and mimicking the process to reach the correct conclusions. Thus, leveraging the in-context exemplars, Llama-2 achieves high accuracy in a few-shot setting. Specifically, Mistral consistently shows degraded performance for all logic types. However, ChatGPT improves performance on NM reasoning. Improved performance in NM reasoning demonstrates that the inclusion of in-context examples enhances the ability of these models to comprehend the nuanced meanings of logical terms such as “usually” and “typically”. In particular, we see that Gemini and GPT-4 improve performance on PL and FOL, respectively, but show competitive performance on NM. 

## **H Experimental Setup** 

## **H.1 Extended Discussion on Experiments** 

**Zero-shot setting** We evaluate GPT-4, and ChatGPT (GPT-3.5-Turbo) by utilizing their APIs provided by OpenAI[3] . We evaluate Google GeminiPro by utilizing its API provided by Google[4] The 

- 3https://platform.openai.com/docs/guides/gpt 

- 4https://ai.google.dev/ 

evaluation is conducted on the versions of GPT-4, ChatGPT, and Gemini released in January 2024. It’s important to note that these models are regularly updated, so when reproducing the results presented in Table 6 and Table 7 (main paper), there is a possibility of variations. For Llama-2 and Mistral, we utilize the 7B-Chat, and 7B-Instruct-v0.2 versions, respectively, from the huggingface model repository[5] . 

**Experiments on other logic datasets** In single and multi-task experiments on other logic datasets, we fine-tune the T5-large model for 10 epochs with a batch size of 16, 1024 maximum input length, an adaptive learning rate of 5 _e −_ 05, and an AdamW optimizer for each experiment. All experiments are performed using NVIDIA RTX A6000 GPUs. 

## **H.2 Prompts** 

All the experiments conducted in the zero-shot setting were performed using three distinct prompts. The reported results in Table 6 (main paper) represent the average performance across these prompts. All the prompts follow the common pattern which includes task description and formatting instructions. The following are the three different prompts utilized in the experiments: 

## **Prompt 1:** 

Given the context and question, think stepby-step logically to answer the question. Answer the question ONLY in ‘yes’ or ‘no’. Please use the below format: **Context** : [text with logical rules] **Question** : [question based on context] **Reasoning steps** : [generate step-by-step reasoning] **Answer** : Yes/No 

## **Prompt 2:** 

Let’s think step-by-step to answer the question given context. Answer the question ONLY in ‘yes’ or ‘no’. Please use the below format: **Context** : [text with logical rules] **Question** : [question based on context] **Reasoning steps** : [generate step-by-step reasoning] **Answer** : Yes/No 

5https://huggingface.co/models 

## **Prompt 3:** 

Given the context that contains rules of logical reasoning in natural language and question, perform step-by-step reasoning to answer the question. Based on context and reasoning steps answer the question ONLY in ‘yes’ or ‘no’. Please use the below format: 

**Context:** [text with logical rules] **Question:** [question based on context] **Reasoning steps:** [generate step-by-step reasoning] 

**Answer:** Yes/No 

For MCQA task, the reported results in Table 7 (main paper) represent the average performance across three prompts. All the prompts follow the common pattern which includes task description and formatting instructions. The following are the three different prompts utilized in the experiments: **Prompt 1:** 

Given the context that contains rules of logical reasoning in natural language, question, and options, perform step-by-step reasoning to answer the question. Answer the question ONLY in ’choice_1’ or ’choice_2’ or ’choice_3’ or ’choice_4’. DO NOT generate any text with it. Based on context and reasoning steps, choose the correct option. Please use the below format: 

**Context:** [text with logical rules] **Question:** [question based on context] **Options:** [four different choices] **Reasoning steps:** [generate step-by-step reasoning] 

**Answer:** choice_1 / choice_2 / choice_3 / choice_4 

## **Prompt 2:** 

Let’s think step-by-step to choose the correct option for the question given context. Answer the question ONLY in ’choice_1’ or ’choice_2’ or ’choice_3’ or ’choice_4’. DO NOT generate any text with it. Based on context and reasoning steps, choose the correct option. Please use the below format: 

**Context:** [text with logical rules] **Question:** [question based on context] **Options:** [four different choices] **Reasoning steps:** [generate step-by-step reasoning] 

**Answer:** choice_1 / choice_2 / choice_3 / choice_4 

## **Prompt 3:** 

Given the context, question, and options, think step-by-step logically to answer the question by choosing the correct option. Answer the question ONLY in ’choice_1’ or ’choice_2’ or ’choice_3’ or ’choice_4’. DO NOT generate any text with it. Based on context and reasoning steps, choose the correct option. 

Please use the below format: 

**Context:** [text with logical rules] **Question:** [question based on context] **Options:** [four different choices] **Reasoning steps:** [generate step-by-step reasoning] 

**Answer:** choice_1 / choice_2 / choice_3 / choice_4 

## **I Further Discussion on Results** 

**Effect on other logic datasets** Our experiments were carried out in two settings: single-task (finetuning and evaluation on one dataset) and multitask (fine-tuning on all four datasets combined, with separate evaluations for each dataset). App. H describes a detailed experimental setup. Table 12 represents the accuracy comparison between LogicT5 and baseline T5-large in both single-task and multi-task settings. 

The results indicate that training LLMs on _LogicBench(Aug)_ has a greater impact on logic datasets that primarily focus on logical reasoning, such as FOLIO and LogicNLI. Hence, we can observe that LogicT5 consistently outperforms the baseline for LogicT5 and FOLIO. However, LogiQA and ReClor encompass other forms of reasoning in addition to logical reasoning, hence, LogicT5 demonstrates competitive performance on them. 

**LLMs sometimes overlook contextual information** In this case study, we manually examined all reasoning chains for PL and NM generated by GPT-4. The example below illustrates the observed 

|Methods|Models|LogiQA|FOLIO|LogicNLI|ReClor|
|---|---|---|---|---|---|
|Single-Task<br>Multi-Task|T5-large<br>LogicT5<br>T5-large<br>LogicT5|16.8<br>**16.9**<br>**21.8**<br>19.7|69.6<br>**71.2**<br>83.8<br>**85.6**|82.3<br>**84.4**<br>68.2<br>**69.8**|35.4<br>**36.8**<br>**42.8**<br>40.0|



Table 12: Performance comparison between LogicT5 and baseline T5-large in terms of accuracy. 

pattern. Specifically, in step 5 of the reasoning chain, the model deviates from the provided context and introduces unrelated information, resulting in an incorrect prediction. Although these patterns are not prevalent, we found that _∼_ 1% of the total analyzed reasoning chains exhibited this pattern in PL and NM for BQA, and _∼_ 0 _._ 8% for MCQA. 

## **Rule:** Modus Tollens (PL) 

**Context:** James had always been forgetful, especially when it came to wearing a sweater. his mother had repeatedly told him, ’If he forgets to wear a sweater, then he will catch a cold.’ throughout the day, he went about his activities, oblivious to the dropping temperature. hours later, he returned home, but James didn’t catch a cold. **Question:** Does this infer that James didn’t forget to wear a sweater? 

## **Gold Answer:** Yes 

## **Reasoning steps:** 

‘if’ and ‘then’ in them”. From the results, we can observe that 13 of the 20 sentences in response to the prompt were FOL kind and only 7 were propositional kind. This shows that LLMs’ comprehend simple FOL sentences, thus, showing high overall accuracy on simpler FOL compared to PL. However, we interestingly observed that, as the depth increases, PL-based examples carrying a specific case/story become easier to understand compared to FOL-based generalized context. 

**Negations are hard to understand when embedded with logical rules.** Here, Table 16 presents reasoning chains corresponding to PL and FOL supporting the effect of negation on their predictions. 

**Performance of BQA** _**vs.**_ **MCQA** By analyzing the reasoning chains of CT and DD for PL, we observe that LLMs often employ the elimination method in MCQA. They prioritize options to select the most appropriate conclusion, thereby improving performance. Conversely, for NM, the models typically struggle to differentiate between two options and tend to hallucinate by assuming information, resulting in lower performance. We present reasoning chains corresponding to PL and NM supporting their behavior on both BQA and MCQA tasks at https://github.com/Mihir3009/LogicBench. 

**Results on Yi-34B** Table 17 provides results for both BQA and MCQA tasks on _LogicBench_ . 

1. James is known to be forgetful about wearing a sweater. 

2. His mother stated that if he forgets to wear a sweater, he will catch a cold. 

3. The temperature dropped, which would increase the likelihood of catching a cold if he were not wearing a sweater. 

4. Despite the dropping temperature, James did not catch a cold. 

5. The absence of a cold does not necessarily mean that James remembered to wear a sweater, as there could be other reasons why he did not catch a cold (e.g., he may have a strong immune system, spent little time outdoors, or was in a warm environment). 

6. Therefore, the fact that James did not catch a cold does not logically infer that he remembered to wear a sweater. Answer: No 

**PL** _**vs.**_ **FOL** To get an indication of this we gave GPT-4 prompt - “Give twenty statements that have 

|Rule|Generate Sentences in Step 1|NL logical expressions|
|---|---|---|
|MT|p: Liam fnished his work early.<br>_∼_p: Liam did not fnish his work early.<br>q: He will order pizza for dinner.<br>_∼_q: He will not order pizza for dinner.|**Context:** Liam had fnished his work early for the day, which meant that<br>he would typically have ordered pizza for dinner. However, on this partic-<br>ular day, he decided against ordering pizza and opted for something else instead.<br>**Question:** Does this imply that liam didn’t fnish his work early?|
|BD|p(x): someone drinks lots of water<br>q(x): they will feel hydrated<br>r(x): they eat too much sugar<br>s(x): they will experience a sugar crash<br>p(a): Jane drinks lots of water<br>_∼_p(a): Jane does not drink lots of water<br>q(a): she will feel hydrated<br>_∼_q(a): she will not feel hydrated<br>r(a): she eats too much sugar<br>_∼_r(a): she does not eat too much sugar<br>s(a): she will experience a sugar crash<br>_∼_s(a): she will not experience a sugar crash|**Context:** If an individual consumes a signifcant amount of water, they will<br>experience a state of hydration. Conversely, if excessive amounts of sugar<br>are ingested, a sugar crash will ensue. It is known that at least one of the<br>following statements is true: either the Jane consumes ample water or she will<br>not experience a sugar crash. However, the actual veracity of either statement<br>remains ambiguous, as it could be the case that only the frst statement is true,<br>only the second statement is true, or both statements are true.<br>**Question:**<br>Can we say at least one of the following must always be<br>true? (a) she will feel hydrated and (b) she doesn’t eat too much sugar|
|MP|p(x): someone is exhausted.<br>q(x): they will take a rest.<br>p(a): Jack is exhausted.<br>_∼_p(a): Jack is not exhausted.<br>q(a): he will take a rest.<br>_∼_q(a): he will not take a rest.|**Context:** If someone is extremely tired, then they will seek some rest and<br>relaxation. Today, Jack fnds himself utterly exhausted.<br>**Question:** Does this entail that he will take rest?|
|DS|p: Levi is not studying for his exams<br>_∼_p: Levi is studying for his exams<br>q: Maya is writing a book<br>_∼_q: Maya is not writing a book|**Context:** Either Levi is not studying for his exams or Maya is writing a book,<br>or maybe both. It was unclear which choice to make as he didn’t know if either<br>of the options was true. However, it turned out that Levi decided to prioritize<br>his exams and focus on studying. He knew that in order to succeed, he needed<br>to dedicate his time and energy to preparing for the upcoming tests.<br>**Question:** Does this mean that Maya is writing a book?|
|HS|p(x): someone buys all the necessary supplies.<br>q(x): they can start the project.<br>r(x): they will fnish it on time.<br>p(a): Lily bought all the necessary supplies.<br>_∼_p(a): Lily did not buy all the necessary sup-<br>plies.<br>q(a): She can start the project.<br>_∼_q(a): She can not start the project.<br>s(a): She will fnish it on time.<br>_∼_s(a): She will not fnish it on time.|**Context:** If all the necessary supplies have been purchased by someone, then<br>they can initiate the project. Once the project is started by someone, they will<br>complete it within the expected timeframe.<br>**Question:**<br>If lily bought all the necessary supplies, does this mean<br>that she will fnish it on time?|
|CD|p: Harry goes to the park.<br>_∼_p: Harry does not go to the park.<br>q: he will have a picnic with his family<br>_∼_q: he will not have a picnic with his family<br>r: he goes to the beach<br>_∼_r: he does not go to the beach<br>s: he will swim in the ocean<br>_∼_s: he will not swim in the ocean|**Context:** Harry had two options for his day off: going to the park or going to<br>the beach. If he decided to go to the park, it meant that he would have a lovely<br>picnic with his family. On the other hand, if he chose the beach, he would be<br>able to enjoy a refreshing swim in the ocean. The only thing that was certain<br>was that either Harry would go to the park or he would go to the beach. It was<br>unclear which option he would ultimately choose, as it could be that he would<br>go to the park alone, go to the beach alone, or even do both activities.<br>**Question:**<br>Can we say at least one of the following must always be<br>true? (a) he will have a picnic with his family and (b) he will swim in the ocean|
|DD|p: I order takeout<br>_∼_p: I did not order takeout<br>q: I will save time<br>_∼_q: I will not save time<br>r: I cook a meal<br>_∼_r: I did not cook a meal<br>s: I will save money<br>_∼_s: I will not save money|**Context:** If I decide to order takeout, it means I will save time. On the other<br>hand, if I choose to cook a meal, it means I will save money. The interesting<br>thing is that I am in a situation where I won’t be able to save time or money.<br>It is uncertain whether I won’t save time or I won’t save money, or it could<br>even be both scenarios. The only thing that is clear is that at least one of these<br>possibilities is true.<br>**Question:**<br>Can we say at least one of the following must always be<br>true? (a) i don’t order takeout and (b) i don’t cook a meal|
|CT|p: Tom is an avid reader<br>_∼_p: Tom is not an avid reader<br>q: he devours books of all genres<br>_∼_q: he does not devour books of all genres|**Context:** At least one of two things is true about Tom - he is either an avid<br>reader or he devours books of all genres. We are unsure which one of these<br>statements is true or if both are true. It could be that only the frst statement is<br>true, or only the second statement is true, or even that both are true.<br>**Question:**<br>Can we say at least one of the following must always be<br>true? (a) he devours books of all genres or (b) tom is an avid reader|
|MI|p: he is not eating healthy<br>_∼_p: he is eating healthy<br>q: he will not gain weight<br>_∼_q: he will gain weight|**Context:** Once upon a time, there was a guy named John who was very<br>conscious about his weight. He knew that if he didn’t eat healthy, he wouldn’t<br>be able to maintain his desired weight. John understood the condition that if he<br>didn’t eat healthy, he wouldn’t be able to gain weight.<br>**Question:** Based on context, can we say, at least one of the following<br>must always be true? (a) he is eating healthy and (b) he will not gain weight|
|EG|p(x): someone has coding skills<br>a: Sheila<br>p(a): Sheila is a profcient programmer|**Context:** Sheila possesses exceptional coding skills and is highly profcient<br>in programming. She has mastered various programming languages and is<br>known for her ability to write effcient and error-free code. Sheila’s expertise in<br>programming allows her to solve complex problems with ease. Her colleagues<br>often seek her assistance when facing technical challenges as she consistently<br>delivers high-quality solutions.<br>**Question:** does this mean that someone has coding skills?|
|UI|p(x): students need to take an exam to com-<br>plete their degree<br>a: Reema<br>p(a): Reema is a student.|**Context:** All students are required to take an examination in order to fulfll the<br>requirements for their degree. Reema, being a student, is also expected to fulfll<br>the requirements.<br>**Question:** Does reema need to take an exam to complete her degree?|



Table 13: Illustrative examples of NL logical context and questions created using sentences that are generated in §3.2.1 for rules covered in PL and FOL for BQA. 

|Rule|Generate Sentences in Step 1|NL logical expressions|
|---|---|---|
|DRI|p: Cats and dogs are mammals.<br>q: Mammals typically have fur.<br>r: Cats don’t have fur.<br>s: Dogs are loyal animals.<br>t: Dogs have fur.<br>_∼_t: Dogs don’t have fur.|**Context:** Once upon a time, in a land flled with animals, there<br>were two popular mammalian creatures, cats and dogs. Mammals<br>typically possessed a coat of fur, which kept them warm and<br>protected. However, cats were an exception to this rule, as their<br>bodies lacked fur. Nonetheless, both cats and dogs were beloved<br>by many for their unique traits. Dogs, known for their loyalty,<br>were particularly cherished by humans.<br>**Question:** Does this imply that dogs have fur?|
|DRS|p: John and Mary are parents.<br>q: Parents are usually loving and supportive.<br>r: Parents are normally responsible.<br>s: Mary isn’t loving and supportive.<br>t: John is not responsible.<br>u: Mary is responsible.<br>_∼_u: Mary isn’t responsible.<br>v: John is loving and supportive.<br>_∼_v: John isn’t loving and supportive.|**Context:** John and Mary were expecting their frst child, flled<br>with the anticipation and excitement that all parents feel. Parents<br>are usually loving and supportive. Parents are normally responsi-<br>ble. However, something seemed amiss in their relationship. Mary,<br>usually affectionate and caring, seemed distant and uninvolved.<br>On the other hand, John, known for his responsible nature, started<br>neglecting his duties and became unreliable.<br>**Question:**<br>Does this imply that Mary is responsible and<br>John is loving and supportive?|
|DRD|p: Jenny and Anna are tall.<br>q: Tall people usually play basketball.<br>r: Anna is possibly an exception to this rule.<br>s: Jenny plays basketball.<br>_∼_s: Jenny doesn’t play basketball.|**Context:** Jenny and Anna are known for their tall stature, which<br>is often associated with playing basketball. However, Anna might<br>be an exception to this norm.<br>**Question:** Does this entail that Jenny plays basketball?|
|DRO|p: Hummingbirds are birds.<br>q: Birds migrate south for the winter.<br>r: Hummingbirds do not migrate south for the<br>winter.<br>s: All other birds than hummingbirds migrate<br>south for the winter.<br>_∼_s: All other birds than hummingbirds don’t<br>migrate south for the winter.|**Context:** In the bird kingdom, there are many different species<br>that possess unique characteristics.<br>One such species is the<br>hummingbird, known for its ability to hover in mid-air and its<br>vibrant colors. While most birds engage in the annual migration<br>south for the winter, the hummingbird chooses to stay put and<br>brave the cold weather. This decision sets the hummingbird apart<br>from its fellow avian companions, as it relies on its resilience and<br>resourcefulness to survive the harsh conditions.<br>**Question:**<br>Does this mean that all other birds than hum-<br>mingbirds migrate south for the winter?|
|RE1|p: Cats, dogs, and horses are animals.<br>q: Animals are usually considered to be intel-<br>ligent creatures.<br>r: At least one of the cats or dogs is not con-<br>sidered intelligent.<br>s: Horses are considered to be intelligent crea-<br>tures.<br>_∼_s: Horses aren’t considered to be intelligent<br>creatures.<br>t: Exactly one of the cats or dogs is not con-<br>sidered intelligent.<br>_∼_t: Exactly one of the cats or dogs is consid-<br>ered intelligent.|**Context:**<br>In a world where animals are often regarded as<br>intelligent creatures, there is a captivating tale that revolves around<br>cats, dogs, and horses. It is commonly believed that most animals<br>possess a level of intellect. However, there is an intriguing twist to<br>this belief as it is known that either cats or dogs are not considered<br>particularly intelligent. As the story unfolds, we delve into the<br>lives of these remarkable creatures, their interactions, and the<br>unique qualities that each of them possesses.<br>**Question:**<br>Does this entail that horses are considered to<br>be intelligent creatures and exactly one of the cats or dogs is not<br>considered intelligent?|
|RE2|p: cats normally meow.<br>q: At least one species of cat doesn’t meow.<br>r: Exactly one species of cat doesn’t meow.<br>_∼_r: Exactly one species of cat meows.|**Context:** In the realm of cat communication, meowing serves<br>as a fundamental aspect of their vocal repertoire.<br>However,<br>intriguingly enough, there exists a distinct species of cat that<br>deviates from this conventional norm. This peculiar feline defes<br>the expectations associated with its kind by refraining from<br>emitting any meows whatsoever.<br>**Question:**<br>Does this entail that exactly one species of cat<br>doesn’t meow?|
|RE3|p: Cars have four wheels.<br>q: wheels normally have spokes.<br>r: at least one wheel does not have spokes.<br>s: Cars have four wheels with spokes.<br>_∼_s: Cars don’t have four wheels with spokes.|**Context:** In a world where cars were known for having four<br>wheels, it was considered a common fact that wheels typically<br>came equipped with spokes. However, amidst this widespread<br>understanding, there was an exception. At least one wheel defed<br>this norm and stood out from the rest by not having any spokes at<br>all.<br>**Question:**<br>Does this imply that cars have four wheels<br>with spokes?|
|RAP|p: John asserts that Sally was in the store.<br>q: Jane asserts that Sally was not in the store.<br>r: John’s evidence is more reliable than Jane’s.<br>_∼_r: John’s evidence is less reliable than Jane’s.<br>s: Sally was in the store.<br>_∼_s: Sally wasn’t in the store.|**Context:** In the midst of a heated argument, John adamantly<br>claims that Sally was present at the store. However, Jane strongly<br>opposes John’s assertion, insisting that Sally was indeed absent<br>from the store.<br>**Question:**<br>If John’s evidence is more reliable than Jane’s,<br>does this mean that Sally was in the store?|



Table 14: Illustrative examples of NL logical context and questions created using sentences that are generated in §3.2.1 for NM logic for BQA. 

|Type<br>Rule|Type<br>Rule|Llama-2|Mistral|Gemini|ChatGPT|GPT-4|
|---|---|---|---|---|---|---|
|||_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|_A_(_No_)<br>_A_(_Y es_)|
|PL|HS<br>DS<br>CD<br>DD<br>BD<br>MT<br>MI<br>CT|100.0<br>25.3<br>66.7<br>26.9<br>100.0<br>55.6<br>93.8<br>79.2<br>76.9<br>25.4<br>100.0<br>25.6<br>55.6<br>51.6<br>84.7<br>52.6|74.3<br>16.7<br>76.6<br>66.7<br>97.1<br>42.2<br>61.1<br>59.1<br>76.4<br>37.5<br>89.3<br>32.7<br>57.9<br>57.1<br>97.4<br>45.2|83.3<br>26.9<br>95.2<br>32.8<br>98.1<br>73.1<br>94.7<br>90.0<br>77.8<br>24.6<br>92.9<br>28.1<br>66.7<br>66.7<br>97.4<br>48.7|84.2<br>27.9<br>87.5<br>33.3<br>100.0<br>57.1<br>64.3<br>83.3<br>82.4<br>30.4<br>100.0<br>30.3<br>64.0<br>73.3<br>85.7<br>36.8|93.8<br>53.1<br>92.2<br>93.8<br>100.0<br>80.0<br>64.3<br>83.3<br>85.1<br>76.9<br>91.5<br>48.5<br>90.5<br>94.7<br>96.7<br>90.0|
||**Avg**|**84.7**<br>**42.8**|**78.8**<br>**44.7**|**88.3**<br>**48.9**|**83.5**<br>**46.6**|**89.2**<br>**77.5**|
|FOL<br>EG<br>UI<br>MP<br>HS<br>DS<br>CD<br>DD<br>BD<br>MT||90.9<br>27.5<br>76.8<br>36.4<br>88.9<br>27.6<br>77.4<br>29.6<br>90.4<br>53.6<br>75.0<br>25.8<br>79.7<br>83.3<br>88.0<br>31.5<br>85.7<br>33.3<br>92.1<br>88.2<br>100.0<br>69.0<br>100.0<br>64.5<br>95.2<br>100.0<br>100.0<br>100.0<br>100.0<br>95.2<br>100.0<br>100.0<br>100.0<br>83.3<br>100.0<br>100.0<br>100.0<br>100.0<br>100.0<br>100.0<br>100.0<br>51.3<br>97.3<br>44.2<br>87.5<br>65.0<br>100.0<br>50.0<br>92.9<br>66.7<br>90.5<br>94.7<br>61.9<br>63.2<br>81.0<br>84.2<br>81.0<br>84.2<br>74.1<br>100.0<br>80.0<br>25.3<br>74.7<br>0.0<br>88.0<br>30.9<br>78.3<br>29.4<br>85.1<br>76.9<br>72.7<br>58.6<br>77.8<br>58.1<br>81.8<br>94.1<br>79.2<br>93.8<br>90.5<br>94.7<br>85.7<br>69.2<br>57.7<br>64.3<br>82.4<br>73.9<br>86.4<br>94.4<br>66.7<br>100.0|||||
||**Avg**|**88.3**<br>**57.9**|**80.7**<br>**55.2**|**88.1**<br>**67.5**|**87.5**<br>**68.3**|**88.0**<br>**86.2**|
|NM<br>DRI<br>DRS<br>DRD<br>DRO<br>RE1<br>RE2<br>RE3<br>RAP||86.2<br>70.6<br>62.5<br>62.5<br>80.4<br>90.9<br>74.0<br>90.0<br>95.2<br>100.0<br>47.4<br>47.6<br>77.3<br>83.3<br>61.9<br>70.6<br>60.7<br>75.0<br>90.0<br>90.0<br>100.0<br>66.7<br>100.0<br>62.5<br>78.6<br>68.0<br>63.6<br>55.2<br>70.4<br>92.3<br>58.1<br>14.3<br>66.7<br>7.7<br>62.8<br>9.1<br>64.1<br>14.6<br>76.5<br>33.3<br>60.9<br>64.7<br>58.6<br>72.7<br>66.7<br>85.7<br>60.6<br>100.0<br>87.0<br>100.0<br>0.0<br>25.0<br>88.0<br>30.9<br>84.8<br>32.6<br>77.8<br>25.8<br>86.5<br>46.4<br>52.6<br>52.4<br>68.0<br>80.0<br>82.4<br>71.4<br>62.1<br>81.8<br>61.5<br>71.4<br>50.0<br>50.0<br>65.5<br>90.9<br>62.5<br>100.0<br>55.6<br>100.0<br>87.0<br>100.0|||||
||**Avg**|**56.9**<br>**48.9**|**73.3**<br>**61.3**|**72.5**<br>**66.0**|**64.8**<br>**67.8**|**81.8**<br>**79.2**|



Table 15: Performance of LLMs in few-shot setting in terms of label-wise accuracy on LogicBench(Eval), where _A_ ( _Y es_ ) and _A_ ( _No_ ) denote the accuracy for the _Y es_ and _No_ labels, respectively. 

|Rule|Context/Question|Actual/Pred|Reasoning Chains|
|---|---|---|---|
|**PL(MT)**|Context: if elijah has a new job, then he will get a<br>raise. however, he would not be receiving the raise<br>he had anticipated.<br>Question: does this infer that elijah doesn’t have a<br>new job?|Yes/No|1. The context provides a conditional statement: If Elijah has a<br>new job, then he will get a raise.<br>2. The context also states that Elijah would not be receiving the<br>raise he had anticipated.<br>3. The second statement does not negate the frst; it only states<br>that the raise is not as expected.<br>4. The lack of the anticipated raise does not necessarily mean<br>Elijah doesn’t have a new job; it could mean that the raise with the<br>new job is different from what he expected.<br>5. Therefore, the information provided does not allow us to infer<br>that Elijah doesn’t have a new job based solely on the raise situa-<br>tion.|
|**PL(MT)**|Context: james had always been forgetful, especially<br>when it came to wearing a sweater. his mother had<br>repeatedly told him, ’if he forgets to wear a sweater,<br>then he will catch a cold.’ throughout the day, he<br>went about his activities, oblivious to the dropping<br>temperature. hours later, he returned home, but james<br>didn’t catch a cold.<br>Question: does this infer that james didn’t forget to<br>wear a sweater?|Yes/No|1. The mother’s rule establishes a conditional relationship: If<br>James forgets to wear a sweater (condition), then he will catch a<br>cold (result).<br>2. It is stated that James did not catch a cold upon returning home.<br>3. According to the mother’s rule, if the result (catching a cold) did<br>not occur, we cannot conclusively determine whether the condition<br>(forgetting to wear a sweater) happened or not.<br>4. The absence of the result (catching a cold) does not necessarily<br>mean the condition (forgetting to wear a sweater) did not occur,<br>as there could be other factors at play that prevented James from<br>catching a cold.<br>5. Therefore, from the information given, we cannot infer with<br>certainty that James remembered to wear a sweater.|
|**FOL(DD)**<br>Context: if an individual operates a vehicle with cau-<br>tion, they will prevent any accidents from occurring.<br>however, if they decide to accelerate and exceed the<br>speed limit, they will be issued a fne. there is a cer-<br>tainty that at least one of the following statements is<br>true: either john will not be able to avoid an accident<br>or he will not face a penalty. it is unclear which of<br>these statements is accurate, as it could be only one<br>of them, or it is possible that both of them hold true.<br>Question: can we say at least one of the following<br>must always be true? (a) john doesn’t drive carefully<br>and (b) he doesn’t speed up<br>Yes/No<br>1. According to the context, if an individual operates a vehicle<br>with caution, they will prevent any accidents from occurring.<br>2. Also, if they decide to accelerate and exceed the speed limit,<br>they will be issued a fne.<br>3. It is stated that there is a certainty that at least one of the<br>following statements is true: either John will not be able to avoid<br>an accident or he will not face a penalty.<br>4. It is unclear which of these statements is accurate, as it could<br>be only one of them, or it is possible that both of them hold true.<br>5. The question asks if we can say at least one of the following<br>must always be true: (a) John doesn’t drive carefully and (b) he<br>doesn’t speed up.<br>6. Based on the given information, we cannot say that at least one<br>of the statements (a) and (b) must always be true.||||
|**FOL(DD)**|Context: if someone is taking care of thier health,<br>then they will be ft. however, indulging in unhealthy<br>habits can make individuals susceptible to various<br>diseases. the truth is, we can be certain about at least<br>one of the following possibilities: either jenny won’t<br>be ft and healthy, or she won’t be prone to diseases.<br>it is important to note that we are unaware of which<br>statement is accurate. it could be the case that only<br>the frst statement is true, only the second statement<br>is true, or both statements hold validity.<br>Question: can we say at least one of the following<br>must always be true? (a) jenny takes care of her<br>health and (b) she doesn’t indulge in unhealthy habits|No/Yes|1. According to the context, if someone takes care of their health,<br>they will be ft.<br>2. Indulging in unhealthy habits can make individuals susceptible<br>to various diseases.<br>3. The context states that we can be certain about at least one of<br>the following possibilities: either Jenny won’t be ft and healthy<br>or she won’t be prone to diseases.<br>4. It is important to note that we are unaware of which statement<br>is accurate.<br>5. The possibilities are:<br>a) Jenny takes care of her health and is ft.<br>b) Jenny doesn’t indulge in unhealthy habits and is not prone to<br>diseases.<br>c) Both statements hold validity.<br>6. Therefore, at least one of the following must always be true:<br>(a) Jenny takes care of her health and (b) she doesn’t indulge in<br>unhealthy habits.|



Table 16: Case study of reasoning chains for PL(MT) generated from ChatGPT, and FOL(DD) generated from GPT-4. 

|**Logic**<br>**Rule**|**Logic**<br>**Rule**|**BQA**|**MCQA**|
|---|---|---|---|
|||||
|**Type**||**A(No)**<br>**A(Yes)**|**Acc.**|
|||||
|PL|HS<br>DS<br>CD<br>DD<br>BD<br>MT<br>MI<br>CT|98_._810_._01<br>92_._130_._02<br>63_._610_._06<br>87_._250_._07<br>88_._880_._02<br>41_._040_._05<br>73_._690_._01<br>19_._530_._01<br>83_._620_._03<br>43_._380_._04<br>46_._880_._03<br>31_._550_._05<br>84_._490_._01<br>31_._570_._01<br>95_._480_._04<br>59_._150_._01|96_._670_._03<br>90_._000_._05<br>86_._670_._03<br>63_._330_._08<br>65_._000_._05<br>58_._330_._06<br>71_._670_._06<br>70_._000_._13|
|||||
|**Avg**||79_._430_._03<br>50_._70_._03<br>75_._210_._06||
|||||
|FOL|EG<br>UI<br>MP<br>HS<br>DS<br>CD<br>DD<br>BD<br>MT|100_._00_._0<br>98_._330_._03<br>90_._320_._00<br>94_._640_._00<br>84_._870_._02<br>100_._00_._0<br>96_._670_._02<br>94_._520_._00<br>77_._420_._03<br>90_._280_._09<br>90_._410_._01<br>40_._390_._02<br>72_._770_._02<br>22_._200_._03<br>79_._600_._03<br>30_._050_._06<br>50_._950_._02<br>56_._110_._21|88_._330_._03<br>88_._330_._08<br>93_._330_._08<br>76_._670_._03<br>55_._000_._05<br>81_._670_._08<br>50_._000_._20<br>60_._000_._05<br>68_._330_._19|
|||||
|**Avg**||82_._560_._02<br>69_._610_._05<br>73_._520_._09||
|||||
|NM|DRI<br>DRS<br>DRD<br>DRO<br>REI<br>REII<br>REIII<br>RAP|86_._100_._08<br>98_._250_._03<br>73_._790_._00<br>10_._370_._10<br>85_._760_._05<br>100_._00_._0<br>70_._370_._04<br>100_._00_._0<br>83_._800_._03<br>44_._310_._09<br>63_._270_._02<br>76_._030_._06<br>63_._890_._03<br>88_._330_._13<br>68_._150_._06<br>87_._120_._03|60_._000_._09<br>30_._000_._05<br>55_._000_._09<br>33_._330_._06<br>73_._330_._08<br>63_._330_._08<br>56_._670_._03<br>38_._330_._08|
|||||
||**Avg**|74_._390_._04<br>75_._550_._05|51_._250_._07|



Table 17: Evaluation of Yi-34B in terms of accuracy on _LogicBench(Eval)BQA_ and _LogicBench(Eval)MCQA_ . 

