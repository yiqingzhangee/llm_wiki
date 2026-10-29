Published as a conference paper at ICLR 2023 

# - - ROSCOE: A SUITE OF METRICS FOR SCORING STEP BY STEP REASONING 

## **Olga Golovneva, Moya Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, Asli Celikyilmaz** Meta AI Research 

{olggol, mpchen, spoff, mcorredor, lsz, maryamfazel, aslic}@meta.com 

## ABSTRACT 

Large language models show improved downstream task performance when prompted to generate _step-by-step_ reasoning to justify their final answers (Nye et al., 2021; Wei et al., 2022). These reasoning steps greatly improve model interpretability and verification, but objectively studying their correctness (independent of the final answer) is difficult without reliable methods for automatic evaluation. We simply do not know how often the stated reasoning steps actually support the final end task predictions. In this work, we present ROSCOE, a suite of interpretable, unsupervised automatic scores that improve and extend previous text generation evaluation metrics. To evaluate ROSCOE against baseline metrics, we design a typology of reasoning errors and collect synthetic and human evaluation scores on commonly used reasoning datasets. In contrast with existing metrics, ROSCOE can measure semantic consistency, logicality, informativeness, fluency, and factuality — among other traits — by leveraging properties of step-by-step rationales. We empirically verify the strength of our metrics on five human annotated and six programmatically perturbed diagnostics datasets - covering a diverse set of tasks that require reasoning skills and show that ROSCOE can consistently outperform baseline metrics.[1] 

## 1 INTRODUCTION 

Scaling language models has improved state-of-the-art performance on nearly every NLP benchmark (Brown et al., 2020), with large language models (LLMs) performing impressively as few-shot learners (Brown et al., 2020). Despite these achievements, even the largest of these models still struggle with tasks including math word problems (Hendrycks et al., 2021), symbolic manipulation (Rytting & Wingate, 2021), and commonsense reasoning (West et al., 2022). Recent work has shown that prompting (Wei et al., 2022; Wang et al., 2022) or fine-tuning (Lampinen et al., 2022) LLMs to generate _step-by-step_ rationales can lead to improvements on reasoning tasks. Some of these include small-scale analysis of specific error types within step-by-step rationales (Lewkowycz et al., 2022; Chowdhery et al., 2022), as shown in Table 1. However, existing works primarily focus on end-task performance. Although text generation evaluation metrics sometimes offer fine-grained quality evaluations (e.g., adequacy, fluency) against human scores (Opitz & Frank, 2021; Leiter et al., 2022), these metrics generally treat the output as a whole, and many of these generative metrics operate on tasks such as summarization or machine-translation rather than reasoning. 

> 1 Code can be found at https://github _._ com/facebookresearch/ParlAI/tree/main/projects/ roscoe. Annotated datasets can be downloaded from https://dl _._ fbaipublicfiles _._ com/parlai/ projects/roscoe/annotations _._ zip. 

1 

Published as a conference paper at ICLR 2023 

In this paper, we present ROSCOE, a suite of interpretable and fine-grained step-by-step generation evaluation metrics to address the above gaps. Rather than providing one score that only evaluates the generated text on the overall, ROSCOE encapsulates fine-grained metrics under four perspectives: (1) _semantic alignment_ defines to what extend the generated reasoning is coherent, and grounded with the source context; (2) _logical inference_ evaluates if the generated reasoning steps are consistent within itself and checks for logical fallacies; (3) _semantic similarity_ quantifies the degree of similarity between the generated reasoning and the context or between intermediate steps to capture hallucinations or repetitions; and (4) _language coherence_ evaluates if the whole chain flows naturally. 

To evaluate ROSCOE against existing metrics, we devise a taxonomy of reasoning errors for multi-step generations and use it to create synthetic data and collect human 

Table 1: Step-by-step reasoning for logical inference task generated from 2-shot GPT-3 (text-davinci-002) (Brown et al., 2020). Error types are highlighted: logical or factual issues, partially correct, somewhat-relevant. Full 2-shot exemplars are in Appendix B. 

|**Context**: Shelley is from Virginia, but is visiting that|
|---|
|city with that famous market where they throw the fsh.<br>Going home next Tuesday!<br>**Question**: Is it likely that Shelly will be near the Pacifc<br>Ocean this weekend?|
|**GPT-3 Step-by-step reasoning:**<br>_•_<br>No,<br>it<br>is<br>not<br>likely<br>that<br>Shel<br>ley<br>will<br>be<br>near<br>the<br>Pa<br>cifc<br>Ocean<br>this<br>week<br>end.<br>_•_<br>Shel<br>ley<br>is<br>from<br>Vir<br>ginia,<br>which<br>is<br>on<br>the<br>East<br>Coast<br>of<br>the<br>United<br>States.|
|_•_The Pacifc Ocean is on the West Coast.|
|_•_<br>Shel<br>ley<br>is<br>vis<br>it<br>ing<br>a<br>city<br>with<br>a<br>fa<br>mous<br>mar<br>ket,<br>which<br>could<br>be<br>any<br>where<br>in<br>the<br>world.|
|_•_<br>Tues<br>day<br>is<br>her<br>travel<br>day,<br>so<br>she<br>will<br>likely<br>not<br>be|
|trav<br>el<br>ing<br>this<br>week<br>end.|



evaluations on commonly used reasoning datasets. Our taxonomy and annotated datasets help us gain deeper insights into the causes of reasoning inconsistencies and weaknesses of LLMs. We evaluate ROSCOE with 18 fine-grained metrics under the above four perspectives. ROSCOE demonstrates performance gains against baseline evaluation metrics on all tasks that require reasoning over context. Additional sensitivity analysis shows that ROSCOE is more robust when dealing with tasks that require logical and arithmetic reasoning. 

**Contributions** . (1) We propose a new taxonomy for reasoning errors, and use it for collecting human annotations and creating synthetic datasets. (2) Using our taxonomy, we propose a new suite of metrics that focus on sequence and step level analysis of step-by-step reasoning. (3) We present extensive comparative analysis on 11 datasets of varied complex reasoning problems demonstrating the strengths of each metric, especially in terms of interpretability relative to baselines, and considerations for use. 

## 2 RELATED WORK 

**Evaluating Explanations.** Free-form natural Language (NL) explanations of model decisions should enable accurate representation of the reasoning process and degree of plausibility (Danilevsky et al., 2020; Jacovi & Goldberg, 2021; Jacovi et al., 2021). A qualitative assessment of NL explanations with correctness labels collected from human judges was presented in (Camburu et al., 2018). Recent work has also investigated automatic metrics for natural language generation (NLG) evaluation including word overlap or embedding based similarly with human written explanations (Clinciu et al., 2021). Though fast and cost-effective, automatic metrics for NLG are not equipped to measure the logical inconsistencies or information gain with thinking steps (Reiter, 2019; Celikyilmaz et al., 2020). Explanations have also been evaluated by collecting datasets, and running correlation analysis to investigate the degree to which an automatic metric correlates with human judgements of clarity, relevance and informativeness (Leiter et al., 2022; Welleck et al., 2022).Although reliable, human evaluation is an expensive, domain specific, and time-consuming process. In comparison, ROSCOE provides generic automatic evaluation procedures that are domain and task specific. 

**Automatic Metrics.** Many NLG evaluation metrics exist in the literature including ones based on: _n_ -gram match (Lin, 2004), regression (Sellam et al., 2020), embedding proximity (Zhang et al., 2020), paraphrasing (Thompson & Post, 2020), generation as an evaluator (Yuan et al., 2021); information alignment (Deng et al., 2021); among others. Although these metrics are easy to use, they evaluate the alignment of two texts as a whole and are not designed to assess individual reasoning steps. The closest metrics to ours are CTC (Deng 

2 

Published as a conference paper at ICLR 2023 

Table 2: Taxonomy of Step-by-Step Reasoning Errors. Full list of the error types with examples is illustrated in Table 10. 

|**Error Type**|**Defnition**|
|---|---|
|**Grammar**|Faulty, unconventional, or controversial grammar usage|
|**Factuality**|Information about an object (i.e. quantity, characteristics) or a named entity doesn’t match with the input context.|
|**Hallucination**|Information is not provided in the problem statement and is irrelevant or wrong|
|**Redundancy**|Explanation contains redundant information, which even though might be factual, is not required to answer the question|
|**Repetition**|Step paraphrases information already mentioned in previous reasoning steps|
|**Missing step**|The content of the generated reasoning is incomplete and lacks required information to produce the correct answer.|
|**Coherency**|Steps contradict each other or do not follow a cohesive story|
|**Commonsense**|Model lacks relations that should be known from general world (e.g., "all ducks are birds")|
|**Arithmetic**|Error in math calculations|



et al., 2021) and BARTScore (Yuan et al., 2021), as both introduce a set of interpretable metrics to evaluate the similarity between two texts. However, ROSCOE is unique in providing fine-grained interpretations of reasoning steps, determining contradictions, and identifying ordering issues in the reasoning narrative. 

**Self-Consistency with LLMs.** Recent work on improving LLMs performance on complex reasoning tasks uses an ensemble strategy called self-consistency (Wang et al., 2022). This method samples a diverse set of reasoning paths from a language model via reasoning traces prompting and returns the most consistent final answer in the set. Other work evaluates the diversity of a reasoning path (Li et al., 2022), or the consistency of an inference step (Creswell et al., 2022) or finetune LLMs (Zelikman et al., 2022) to improve on difficult NLP tasks. In contrast to these works, we present a suit of metrics that focus on determining the type of the error (e.g., commonsense or logical inconsistency) in a reasoning path, if one exists. 

## 3 REASONING ERROR TAXONOMY AND DATASETS CONSTRUCTION 

**Problem Formulation** . Our goal is to score step-by-step rationales generated by a language model. We assume that the model is given a _source_ context _**s**_ = _{s_ 1 _, · · · , sT }_ of _T_ -sentences indicating a problem statement followed by a question and is prompted to generate step-by-step reasoning (Nye et al., 2021). We refer to this as a _hypothesis_ _**h**_ = _{h_ 1 _, · · · , hN }_ of _N_ -steps, including a final answer as the last step. We do not assume availability of gold step-by-step reasoning _references_ _**r**_ = _{r_ 1 _, · · · , rK}_ of _K_ -steps. 

**Taxonomy** . We propose a new taxonomy of generic reasoning errors for language problem solving. We first conduct manual preliminary analysis on different types of LLMs reasoning errors using five _Human judged_ datasets described below. Based on our analysis, we identified nine error types centered on the overall reasoning chain (i.e., the quality of the step-by-step thinking, including consistency with the context and commonsense reasoning). Our taxonomy also includes fine-grained errors marking inconsistency of a reasoning step with the previous steps, whether each step contributes to the final decision, and overall logical inference or fluency issues. The definition of error types is in Table 2, and Table 10 provides examples. 

**Datasets and Annotations** . To evaluate ROSCOE, we select datasets covering diverse set of tasks that require reasoning skills (e.g., logical, arithmetic, and commonsense reasoning tasks). We separate these datasets into two: (1) **Diagnostics** datasets that contain gold standard step-wise reasoning chains, where we synthetically perturb some of the reasoning steps to introduce different generation errors (e.g., missing step, mathematical error, etc.); (2) **Human judged** datasets with model generated step-by-step reasoning outputs where the reasoning error evaluations are solicited from expert judges. We investigate these in _§_ 5. 

## 4 REASONING SCORER: ROSCOE 

We present our fine-grained metrics under four perspectives: _semantic alignment_ , _semantic similarity_ , _logical inference_ and _language coherence_ . Each metric is bounded within [0 _,_ 1], where 1 indicates the perfect score 

3 

Published as a conference paper at ICLR 2023 

and 0 corresponds to failure. A metric is _reference-free_ or _unsupervised_ when it uses the source and hypothesis ( _**h** →_ _**s**_ ), while _reference-based_ or _supervised_ when evaluated between hypothesis and reference ( _**h** →_ _**r**_ ). 

## 4.1 SEMANTIC ALIGNMENT METRICS (ROSCOE-SA) 

At the core of the ROSCOE semantic alignment[2] metrics is the reasoning alignment vector from the _N_ -step hypothesis _**h**_ to the source _**s**_ of length _T_ : _r_ -align( _**h** →_ _**s**_ ) = _{α_ 1 _, α_ 2 _, · · · , αN }_ , where each alignment value _αi_ = _r_ -align( _hi →_ _**s**_ ) = [1 + max _[T] j_ =1[(cos(] _[h][i][, s][j]_[)]] _[/]_[2] _[∈]_[[0] _[,]_[ 1]][ is the normalized cosine similarity between] hypothesis step and most similar sentence in a context, and explicitly measures the grounding of the step-wise reasoning with respect to the source text (illustrated in App. D, Fig. 3). We estimate the alignment vector _r_ -align( _**h** →_ _**s**_ ) by matching source text and the reasoning chains on the embeddings of tokens and individual reasoning steps. A similar information alignment score is introduced in CTC (Deng et al., 2021) to measure the confidence that the information of the _i_ -th source document token _sj_ is grounded by a hypothesis token _hi_ . Our reasoning alignment is different in that we measure if a hypothesized reasoning step _hi_ supports the source context _**s**_ . Our proposed metrics are summarized in Table 3. 

Table 3: Semantic alignment metrics (ROSCOE-SA). 

||Table 3: Semantic alignment metrics(ROSCOE-SA).|
|---|---|
|**Score**|**Description**|
|**Faithfulness-Step**|This step-level score is based on the alignment from the hypothesis steps to the source sentences, and is calcu-|
|(**_h_**_→_**_s_**)|lated as the mean reasoning alignment score over the steps of reasoning (see illustration in Appendix D, Figure 3):|
||(1_/N_) �_N_<br>_i_=1 _r_-align(_hi →_**_s_**). Faithfulness measures if the model misinterpreted the problem statement, or the reasoning|
||chain is too vague, irrelevant, or misuses information.|
|**Faithfulness-Token**|We extend step-level embeddings of the Faithfulness-Step by measuring similarities between the token embeddings:|
|(**_h_**_→_**_s_**)|(1_/_(_N_ +_M_)) �_N_<br>_i_=1[_r_-align(_hi →_**_s_**) +�_Mi_<br>_j_=1 _r_-align_token_(_hi,j →_**_s_**)], as shown in App.D, Fig.3. _Mi_ is the number of|
||tokens in step_hi_,_M_ = �_N_<br>_i_=1 _Mi_ is the total number of tokens in the reasoning chain,_hi,j_ is the_j_th token in_i_th step, and|
||_r_-align_token_ is the alignment vector from tokens in step_hi_ to all tokens in**_s_**.|
|**Informativeness-Step**|Measures how well information present in the source is used in the reasoning steps: [(1_/T_) �_T_<br>_t_=1 _r_-align(_st →_**_h_**) +|
|**(Info-Step)**(**_h_**_↔_**_s_**)|(1_/N_) �_N_<br>_i_=1 _r_-align(_hi →_**_s_**)]_/_2. Info-step gives a higher score to reasoning steps that are well-grounded with respect|
||to the source, and identifes the degree of information from source that is covered by the generated hypothesis. A lower|
||Info-Step score corresponds to the reasoning steps that are not related to the source sentences or have missed information|
||provided in the context.|
|**Repetition-Token**|To identify repeated, or paraphrased steps, we look at the token alignment scores between all steps in the hypothesis chain:|
|(_hi →hj_)|1_−_max_i_=2_..N_max_j_=1_···i−_1[(1_/Mi_) �_Mi_<br>_l_=1 _r_-align_token_(_hi,l →hj_)]. For each pair of sentences, we look at the mean|
||token alignment, and fnd those sentences that maximize this alignment score. In other words, Repetition-Token will punish|
||chains where there are at least two steps with high overlap in token embeddings.|
|**Hallucination**|To fnd irrelevant reasoning steps, we use alignment score to identify steps that are both not related to the context and not in|
|(**_h_**_→_(**_s_**_,_**_r_**))|the reference chain (to avoid punishing for possibly relevant commonsense knowledge): 1_−_max_i_=1_..N_([**1**_−r_-align(**_h_**_→_|
||**_s_**)]_·_[**1**_−r_-align(**_h_**_→_**_r_**)]). Here,**1**is an all-ones vector, and (_·_) is the element-wise product.|
|**Redundancy**(**_h_**_→_**_r_**)|To fnd chains that contain information that is not required to solve the problem (i.e., redundant steps), we identify those|
||hypothesis steps that are least aligned with the the reference steps: min_i_=1_..N r_-align(_hi →_**_r_**). This score punishes chains|
||with steps that are not required for the correct solution.|
|**Semantic**|This score can be viewed as a measure of how easily a gold reference could be generated by the hypothesis. It compares step|
|**Coverage-Step**<br>((**_r_**_,_**_h_**)_→_**_s_**)|level grounding of the hypothesis with respect to the source, and the gold reference grounding: _|_(1_/T_) �_K_<br>_t_=1 _r_-align(_rt →_<br>**_s_**)_−_(1_/N_) �_N_<br>_i_=1 _r_-align(_hi →_**_s_**)_|_, where_|·|_indicates absolute value.|
|**Reasoning Alignment**|The most straightforward way to evaluate the correctness of the hypothesis chain is to compare the degree of the overlap|
|(**_h_**_→_**_r_**)|between the hypothesis and the reference. One way of doing that is to measure the reasoning alignment between them:|
||(1_/N_) �_N_<br>_i_=1 _r_-align(_hi →_**_r_**).|
|**Commonsense**|Measures if hypothesis lacks steps that are not stated in the source, but are required to solve the problem such as|
|(**_r_** _→_(**_h_**_,_**_s_**))|general world knowledge (e.g., "velocity is distance divided by time", "1 foot is 12 inches", "all ducks are birds", etc.).|
||We detect such information by extracting steps in the reference reasoning that are not grounded by the source text:|
||1_−_max_i_=1_..K_([**1**_−r_-align(**_r_** _→_**_h_**)]_·_[**1**_−r_-align(**_r_** _→_**_s_**)]).|
|**Missing Step**(**_r_**_→_**_h_**)|To identify steps that are missing from the hypothesis but could be required to solve the problem, we look at the alignment|
||between reference and the hypothesis, similar to_Redundancy_. However, here we go through each step in the reference, and|
||check if there is a similar step in the hypothesis: min_i_=1_..K_(_r_-align(_ri→_**_h_**)).|



> 2Semantic alignment refers to determination of relations between concepts with the same or a similar intended meaning (Agirre et al., 2013). 

4 

Published as a conference paper at ICLR 2023 

## 4.2 SEMANTIC SIMILARITY METRICS (ROSCOE-SS) 

Semantic similarity metrics quantify the degree of semantic equivalence between pieces of text. As opposed to the ROSCOE-SA metrics, ROSCOE-SS considers text as a whole, rather than relying on text units comparisons. We propose the following metrics summarized in Table 4. 

Table 4: Semantic similarity metrics (ROSCOE-SS). 

||Table 4: Semantic similarity metrics (ROSCOE-SS).|
|---|---|
|**Score**|**Description**|
|**Informativeness-Chain**|Similar to Info-Step, this metric quantifes the degree of agreement between the hypothesis chain and the source and is|
|**(Info-Chain)**(**_h_**_→_**_s_**)|calculated as[1 + cos(**_h_**_,_**_s_**)]_/_2. We embed reasoning chain and source context as a whole, as opposed to using step-wise|
||embeddings in *-Step types of metrics introduced in Table3.|
|**Repetition-Step**|Measures repetition-related errors on the step level by checking if it paraphrases information already mentioned in the|
|(_hi↔hj_)|previous steps: (1_−_max_i_=2_..N_max_j_=1_···i−_1[cos(_hi, hj_)])_/_2. Unlike Repetition-Token, which is orderless and compares|
||individual tokens in pairs of steps, Repetition-Step considers step embeddings similarity and is more robust to changing|
||contexts.|
|**Semantic Coverage-**|Refects the overall degree of similarity between the reference and hypothesis chains, comparing reference and hypothesis|
|**Chain**(**_r_** _↔_**_h_**)|embeddings as a whole: [1 + cos(**_r_**_,_**_h_**)]_/_2.|



## 4.3 LOGICAL INFERENCE METRICS (ROSCOE-LI) 

Logical inference metrics (Table 5) measure logical errors between pieces of text. We use an NLI model that was trained to classify hypothesis-context pairs into entailment, neutral, and contradiction classes (Laurer et al., 2022) to infer the contradiction probability _p_ contr. 

Table 5: Logical inference metrics (ROSCOE-LI). 

||Table 5: Logical inference metrics (ROSCOE-LI).|
|---|---|
|**Score**|**Description**|
|**Self-Consistency**|Measures logical entailment errors_within_the reasoning steps: 1_−_max_i_=2_..N_max_j<i p_contr(_hi, hj_). This metric will|
|(_hi↔hj_)|punish chains where there is a pair of steps that are likely to contradict each other.|
|**Source-Consistency**|Measures logical entailment errors between any generated reasoning **_h_** and the source context **_s_**:<br>1 _−_|
|(**_h_**_↔_**_s_**)|max_i_=1_..N_max_j_=1_..T p_contr(_hi, sj_). Specifcally, for each reasoning step we measure the probability that it contra-|
||dicts any sentence in the context. We take the maximum probability of contradiction over all steps, following the logic that|
||a contradiction anywhere in the reasoning chain signals a failure of the overall argument.|



## 4.4 LANGUAGE COHERENCE METRICS (ROSCOE-LC) 

To evaluate language coherence (Table 6), we use perplexity PPL as scored by the GPT2-Large model (Radford et al., 2019), and English grammatical acceptability _p_ gram as scored by the classifier model from Krishna et al. (2020). Both models were used as-is with no finetuning. 

Table 6: Language coherence metrics (ROSCOE-LC). 

||Table 6: Language coherence metrics (ROSCOE-LC).|
|---|---|
|**Score**|**Description**|
|**Perplexity-Chain**(**_h_**)|Average perplexity of all tokens in the generated reasoning steps: 1_/_PPL(**_h_**). The context used to score each token is|
||the previous tokens in the current and all previous steps. Steps are joined with a space character. To keep the range and|
||orientation consistent with the other scores we invert the perplexity.|
|**Perplexity-Step**(_hi_)|Average perplexity of all tokens in the generated reasoning steps, where the context used to score each token is only the|
||previous tokens within the current step: 1_/_[(1_/N_) �_N_<br>_i_=0 PPL(_hi_)]. To keep the range and orientation consistent with the|
||other scores we invert the perplexity.|
|**Grammar**(_hi_)|Probability of grammatical acceptability of each step, averaged over all steps: (1_/N_) �_N_<br>_i_=0 _p_gram(_hi_).|



## 5 EXPERIMENTAL SETUP 

**Diagnostics Datasets** . We construct our first category of labeled datasets by generating perturbations — i.e., deterministic modifications — on half of the reference reasoning steps and assign binary labels based on whether or not a chain has been perturbed. We select seven language understanding and entailment datasets 

5 

Published as a conference paper at ICLR 2023 

that require complex problem solving skills, and have reference step-by-step explanations: **Entailment-Bank** (deductive reasoning) (Dalvi et al., 2021), **ProofWriter** (logical reasoning) (Tafjord et al., 2021); three arithmetic reasoning datasets **MATH** (Hendrycks et al., 2021), **ASDIV** (Miao et al., 2020) and **AQUA** (Liang et al., 2018); **EQASC** (explanations for commonsense question answering) (Aggarwal et al., 2021), and **StrategyQA** (question answering with implicit reasoning strategies) (Geva et al., 2021) (see dataset details in App. E.1). Using our taxonomy, we introduce 12 error perturbation rules and apply on these datasets to construct our diagnostics datasets (see details in App. E.3). 

**Human Judged Datasets** . We select our second category of datasets from commonly used complex reasoning tasks: **GSM8K** (arithmetic reasoning) (Cobbe et al., 2021), **DROP** (discrete reasoning) (Dua et al., 2019), **ESNLI** (deductive and commonsense reasoning) (Camburu et al., 2018), **COSMOS-QA** (commonsense reasoning) (Huang et al., 2019) and **SemEVAL** (Ostermann et al., 2018) (commonsense reasoning). Wei et al. (2022) provide model generated chain of thought reasoning steps for GSM8K. We used chains produced by the _175b_verification_ model to annotate for reasoning errors. For other datasets, we prompt GPT-3 LLM (Brown et al., 2020) with few-shot in-context examples to obtain step-by-step reasoning sequences (see examples in App. E.2). We use the error types in our taxonomy in Table 2 as human evaluation perspectives of reasoning errors where we solicit five expert annotators[3] . The data collection interface provided judges with the source text (e.g., source and a question, or hypothesis, premise, and a question if they entail) and associated reasoning text clearly separated into individual steps. Judges were asked to rate the chain as a whole (e.g., on overall quality) as well as each individual step (e.g., commonsense errors, contradicts with the previous steps). App. Table 16 summarizes the distribution of error types annotated by the judges. See App. F for details. 

**ROSCOE Training.** To obtain reasoning step embeddings, we finetune SimCSE (Gao et al., 2021), a supervised sentence similarity model extending the RoBERTa word embedding model (Liu et al., 2019) on multi-step reasoning datasets we listed in _§_ 5 (see details in Table 11)[4] . SimCSE is a contrastive learning model that is trained on triplets of reference reasoning steps, positive and hard-negative hypothesis reasoning steps to minimize the cross-entropy objective with in-batch negatives. For contrastive learning, we use the context and reference reasoning steps as a positive sample ( _**s** ,_ _**r**_ ), and context and perturbed reference steps ( _**s** ,_ _**h**_ ) as hard-negative pairs. For finetuning, we embed source context and hypothesis chain as a whole, without splitting it into steps. With the finetuned model we embed each individual step, as well as a reasoning chain as a whole. We use the pretrained checkpoint of supervised SimCSE model _sup-simcse-roberta-base_ to initialize our model, and further train it for five epochs on our synthetic train data (details in App. G). We also compare ROSCOE scores calculated against _sup-simcse-roberta-base_ SimCSE model, and _all-mpnet-base-v2_ sentence embedding model (Reimers & Gurevych, 2019) to understand metrics sensitivity to the embedding method. 

**Baseline Metrics.** We use text generation evaluation metrics as baseline metrics and comprehensively examine the ones outlined in §2, which are: _n_ -gram match based metrics including **ROUGE-** 1, **ROUGE-2** , and **ROUGE-L** (Lin, 2004); pre-trained scores including **BLEURT** (Sellam et al., 2020), **PRISM** (Thompson & Post, 2020), **BERTScore** (Zhang et al., 2020), **BARTScore** using the _Faithfulness_ ( _**s** →_ _**h**_ ) direction for factuality and relevance, and its finetuned variant BARTScore+CNN+Para **BARTScore+** (Yuan et al., 2021); and information alignment metrics of **CTC** , **CTC-Relevancy** and **CTC-Consistency** . We also include **BARTScore-P** , which we obtain by finetuneing BART (Lewis et al., 2020) on the same reasoning datasets we use for finetuning our SimCSE embedding models. Most of our ROSCOE metrics are constructed referencefree. We also have metrics that use reference reasoning steps which we examine against human judgements. We use the official code for each metric. 

> 3We chose expert annotators over crowd-sourcing, because our annotation task is cognitively challenging and requires fine-grained annotation. 

> 4 Fine-tuned model is available at https://huggingface _._ co/facebook/roscoe-512-roberta-base 

6 

Published as a conference paper at ICLR 2023 

**Meta Evaluation.** We use _Somers’ D_[5] (Somers, 1962), which measures the ordinal association between two measured quantities, to meta-evaluate each scorer against synthetic and human scores. We prefer _Somers’ D_ over more commonly used _Kendall’s τ_ or _Kendall’s τ -b_ , because it is better in handling the ties of a biased random variable (Agresti, 2010, Section 7.1.5), which imposes an upper bound on the possible values _Kendall’s τ_ ( _-b_ ) can take. For each score _Y_ considered, our correlations are built against the biased random variable _X ∈_ [0 _,_ 1], represented by the perturbation or error presence indicator and evaluated using _D_ ( _Y |X_ ) = _τ_ ( _X, Y_ ) _/τ_ ( _X, X_ ). 

## 6 EXPERIMENTAL RESULTS 

**Controlled Experiments with Diagnostics Datasets** . Table 7 shows Somers’ _D_ correlation for metrics measured reference-free on six different datasets and compares baselines to ROSCOE-* aggregated categories calculated with finetuned embeddings: ROSCOE-SA, ROSCOE-SS, ROSCOE-LI, ROSCOE-LC. Results also include ROSCOE metrics with _all-mpnet-base-v2_ (ROSCOE-SA[1] , ROSCOE-SS[1] ) and _sup-simcse-robertabase_ (ROSCOE-SA[2] , ROSCOE-SS[2] ) sentence embedding models. Correlations for ProofWriter are taken on its _depth-5_ subset. We report highest correlation scores across perturbations within each dataset. The breakdown of all ROSCOE metrics is in App. Table 18. 

We observe that: (1) ROSCOE can outperform all other reference-free methods on all six diagnostic datasets, (2) the gains for ROSCOE-SS are more pronounced in four out of six diagnostics datasets, which suggests that ROSCOE can capture hallucinations and repetitions in step-wise reasoning. On Proofwriter, our scorers show lower correlations, because as shown in Table E.1, the context is a list of facts and rules and the reasoning steps can include unordered fact and rule combinations, but still a correct answer can be deduced. This makes it challenging for ROSCOE to evaluate the steps in sequence. Overall, the correlations of the baseline metrics are much lower than ROSCOE, because the baseline metrics are designed to capture the semantic or lexical overlap between a reference and hypothesis and it is harder to detect logical consistency without a golden reference text. ROSCOE is specifically focused on reference-free settings, and can gauge each individual step against the source and other generated steps. In fact, our metrics also work well against the baselines in 

Table 7: Somers’ _D_ correlation of different metrics on six **Diagnostics** datasets. Metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). We take the maximum score over different perturbations. The two highest correlations for each dataset are **bolded** and underlined, respectively. Correlations that are not significant ( _p-value ≥_ 0 _._ 05) are omitted when aggregating, and "-" denotes an absence of any significant correlation. Breakdown of all baseline and ROSCOE metrics is shown in App. H.1, Table 18. 

|**EntBank Math AQUA ProofWr. EQASC ASDIV**|**EntBank Math AQUA ProofWr. EQASC ASDIV**|
|---|---|
|ROUGE-L<br>0.365<br>0.156<br>BLEURT<br>0.257<br>0.148<br>BERTScore<br>0.380<br>0.124<br>BARTScore<br>0.358<br>0.185<br>BARTScore+<br>0.315<br>0.164<br>BARTScore-P<br>0.186<br>0.128<br>PRISM<br>0.453<br>0.208<br>CTC Relev.<br>0.258<br>0.188<br>CTC Consist.<br>0.310<br>0.282|0.264<br>0.106<br>0.315<br>0.269<br>0.252<br>0.024<br>0.447<br>-<br>0.220<br>0.117<br>0.462<br>0.322<br>0.317<br>0.081<br>0.415<br>-<br>0.251<br>0.054<br>0.297<br>-<br>0.215<br>0.011<br>0.276<br>-<br>0.191<br>0.235<br>0.436<br>-<br>0.217<br>0.394<br>0.485<br>0.382<br>0.157<br>0.513<br>0.270<br>0.396|
|ROSCOE-SA<br>0.919<br>0.939<br>ROSCOE-SA1<br>0.913<br>0.936<br>ROSCOE-SA2<br>0.919<br>0.939|0.971<br>0.763<br>**1.000**<br>**0.879**<br>0.972<br>0.771<br>**1.000**<br>0.198<br>0.971<br>0.732<br>**1.000**<br>0.515|
|ROSCOE-SS<br>**0.955**<br>0.924<br>ROSCOE-SS1<br>0.909<br>0.932<br>ROSCOE-SS2<br>0.901<br>**0.949**|0.982<br>0.624<br>**1.000**<br>0.857<br>0.982<br>0.631<br>**1.000**<br>0.280<br>**0.991**<br>0.621<br>**1.000**<br>0.289|
|ROSCOE-LI<br>0.917<br>0.331|0.424<br>0.289<br>0.793<br>0.771|
|ROSCOE-LC<br>0.604<br>0.392|0.359<br>**0.788**<br>0.859<br>0.485|



the reference-based setting (comparing against reference reasoning steps). In App. Table 19 we present correlations when metrics are measured as reference-based. We also observe that finetuning SimCSE gives highest improvements on the ASDIV dataset. ASDIV is a 1-step reasoning dataset (see App. Table 12), where step is represented by an equation with one of the arithmetic perturbations added. We 

> 5We use SciPy (Virtanen et al., 2020) to calculate correlations and obtain _p-values_ from a hypothesis test where the null hypothesis is an absence of association. 

7 

Published as a conference paper at ICLR 2023 

hypothesize that including these patterns in finetuning helped the model to better learn relationships between context and equations, and resulted in higher scores. On EQASC dataset, Repetition* scores are able to catch all duplicated steps in a chain, i.e., we can separate perturbed and non-perturbed chains based on the given threshold value for the Repetition* scores, and achieve perfect correlation scores (App. Table 20). To understand if finetuning actually helps to improve scoring, we compare non-aggregated metrics (see details in App. Table 18). We observe, that finetuning indeed helps to improve ROSCOE: on average across datasets, all correlations except Repetition_* scores improve (up to 0 _._ 556 on InformativenessChain), with mean Repetition-Token not changing, and mean Repetition-Step degrading by 0 _._ 005. We speculate that since we finetune the model using reasoning chains and context as a whole, it helps to better capture step-by-step rationales, while possibly degrading on word and sentence-level semantics. 

**Meta-Evaluations on Human Judgement Datasets** . Table 8 reports a summary of meta-evaluation of ROSCOE metrics comparing against baselines on human judged datasets. The correlations are measured based on the presence of a particular error from Table 2 and we report the highest correlation across all error types within each dataset. We observe that: (1) on all tasks, ROSCOE metrics outperform all other baselines when evaluated as reference-free; (2) overall, ROSCOE yields considerably better correlations, which indicates that _step-by-step_ reasoning generations can be more effectively evaluated with ROSCOE. In general, most correlations with human judgements are moderate when compared to the synthetic correlation scores, indicating that step-by-step reasoning evaluation is among the cognitively hard tasks for neural models (Deutsch et al., 2022). Interpretable metrics such as ROSCOE can 

Table 8: Somers’ _D_ correlations of metrics with **human judgement** . We report the maximum over the error types in Table 2. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest two correlations in each column are **bolded** and underlined, respectively. Correlations that are not significant ( _p-value ≥_ 0 _._ 05) are omitted when aggregating, and "-" denotes an absence of any significant correlation. Breakdown of all baseline and ROSCOE metrics is shown in App. H.2. 

|||**DROP**|**GSM8K**|**ESNLI**|**COSMOS**|**SemEVAL**|
|---|---|---|---|---|---|---|
||Rouge-L<br>BLEURT<br>BERTScore|0.278<br>0.328<br>0.275|0.252<br>0.256<br>0.235|0.557<br>0.541<br>0.590|-0.441<br>0.218<br>-0.420|-0.478<br>-0.356<br>-0.295|
||BARTScore|-0.835|-0.546|0.549|-0.544|-|
||BARTScore+<br>BARTScore-P<br>PRISM|-0.665<br>-0.642<br>-0.733|-<br>-<br>-0.455|0.482<br>0.255<br>0.580|-0.186<br>-0.207<br>-0.376|-<br>-<br>-|
||CTC-Relevance|0.333|-0.371|0.334|-|-0.349|
||CTC-Consistency|0.462|-0.174|0.647|0.275|-0.301|
||ROSCOE-SA<br>ROSCOE-SA1<br>ROSCOE-SA2<br>ROSCOE-SS<br>ROSCOE-SS1|0.578<br>0.790<br>0.578<br>**0.824**<br>0.791|0.392<br>0.500<br>0.392<br>0.514<br>0.471|0.521<br>**0.799**<br>0.599<br>0.530<br>0.507|0.555<br>0.638<br>0.555<br>0.593<br>0.642|0.337<br>0.485<br>0.337<br>0.411<br>0.508|
||ROSCOE-SS2|0.799|**0.638**|0.531|**0.658**|**0.535**|
||ROSCOE-LI|0.584|0.345|0.531|0.444|0.372|
||ROSCOE-LC|0.205|-0.184|0.447|-0.212|0.517|



provide better information about a model’s reasoning skills, thus future work should improve such metrics on aligning with human judgments. In App. H.2, we show fine-grained experimental analysis per each human labeled dataset. Specific examples showcasing ROSCOE scoring abilities are summarized in Table 40. 

## 7 ANALYSIS 

**How sensitive are ROSCOE metrics against level of errors?** To evaluate how well metric values match human assessment of reasoning, we measure sensitivity to the level of errors. We perturb sentences in the MATH (arithmetic) and EntailmentBank (deductive reasoning) diagnostic datasets (similar to _§_ 5) and inject different levels of errors into the reasoning text. Using randomly selected perturbation types, we construct up to a maximum of 3 perturbations per instance. We measure the correlation (Somers’ _D_ ) between the reasoning inconsistency level 1, 2, 3 of the reasoning steps (i.e., the number of injected errors) and the metric score. Fig. 1 illustrates the results averaged over different perturbations. 

We expect the metrics correlate with humans better when the level of errors is high. Both semantic alignment of the reasoning ROSCOE-SA , and the semantic similarity metrics ROSCOE-SS show consistent behavior 

8 

Published as a conference paper at ICLR 2023 

on both datasets, while baseline metrics fluctuate with low correlations. Baseline metrics perform better on EntailmentBank. On MATH, ROSCOE-LC and the baseline metrics show minimal impact, which can be that some of the perturbations applied on the MATH dataset (e.g., RandomOperation, or ShuffleNumbers) are harder to detect with language model based (BARTScore) and NLI model based (ROSCOE-LC) metrics. 

**What does ROSCOE illuminate about scores across errors and tasks?** For an ideal scorer based on ease of use, it would be possible to pick a set of fixed thresholds that had error discrimination power across datasets. However, we show that this dataset-agnostic ideal is currently not possible and an issue endemic across scores, including baselines. We study which metrics correlate strongly with which perturbations, with a focus of consistency across datasets. From this, we plot the interquartile ranges for strongly correlated metric and perturbation pairs. We show a sample of these in Fig. 2, though find that the trends generally hold across metrics and perturbations (see Fig 6). We note that within a given dataset, scores are well separated: the _perturbed_ version of a dataset for a given score and perturbation type shows little interquartile overlap with the _original_ version. However, this does not hold across datasets – e.g., in (Score: InfoChain, Perturbation: Repetition), if one were to set a detective threshold for the Repetition perturbation based off EntBank (around 0 _._ 95), it would mark almost all values of EQASC as perturbed, even non-perturbed samples. This shows the challenge of using metrics for classification without calibration for drifts in both mean and variance across datasets, even if a metric generally correlates well with detecting a given error. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0009-03.png)


**----- Start of picture text -----**<br>
ROSCOE-SA BARTScore-P<br>ROSCOE-SS BERTScore<br>ROSCOE-LI ROUGE-L<br>ROSCOE-LC PRISM<br>BARTScore BLEURT<br>1.00<br>MATH<br>0.75<br>0.50<br>0.25<br>0.00<br>1.00<br>EntBank<br>0.75<br>0.50<br>0.25<br>0.00<br>1 2 3<br>Level of Error<br>**----- End of picture text -----**<br>


Figure 1: Sensitivity of selected metrics on Somers’ _D_ by injecting levels of error into reasoning steps. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0009-05.png)


**----- Start of picture text -----**<br>
Score: Info-Chain, Perturbation: Repetition Score: Faithfulness-Step, Perturbation: Negate Step Score: BERTScore, Perturbation: Repetition<br>(ProofWr., perturbed)<br>(ProofWr., original)<br>(Math, perturbed)<br>(Math, original)<br>(EntBank, perturbed)<br>(EntBank, original)<br>(EQASC, perturbed)<br>(EQASC, original)<br>(AQUA, perturbed)<br>(AQUA, original)<br>0.5 0.6 0.7 0.8 0.9 1.0 0.2 0.4 0.6 0.8 1.0 0.4 0.2 0.0 0.2 0.4 0.6 0.8<br>Info-Chain score perturbed Faithfulness-Step score perturbed BERTScore score perturbed<br>original original original<br>**----- End of picture text -----**<br>


Figure 2: Box-and-whisker plots of interquartile ranges of scores, for perturbations and reference-free metrics with strong Somers’ _D_ values. Scores are split by dataset and perturbation use. While interquartile ranges separate well by perturbation use within a single dataset, there is overlap across datasets. This shows the drift of neural scores across datasets and applies to both ROSCOE (left, center) and strong baselines (right). 

## 8 CONCLUSION 

In this paper, we introduce ROSCOE, a new suite of interpretable, unsupervised metrics that enables evaluation of step-by-step reasoning generations of LMs when no golden reference generation exists. We present a taxonomy of reasoning errors used to generate and evaluate our metrics. Experimental results, from evaluating on both synthetic and human-labeled datasets exhibiting multiple types of reasoning (commonsense, arithmetic, and logical inference, etc.), demonstrate superior performance compared to prior semantic and lexical similarly based baseline metrics for text generation. Our analysis shows improved capability in evaluation of reasoning exhibiting nuances, such as factual and logical errors in step-wise decisions. 

9 

Published as a conference paper at ICLR 2023 

## ETHICS STATEMENT 

Explainability builds transparency and trust for users, eases bug-fixing and shortens improvement cycles for metric designers, and will be required by law/regulations for AI systems to be applied to large-scale, high-stakes domains. In this context, we hope our work will catalyze efforts on the topic of explainable evaluation metrics for language model rationale generations. We should mention that our evaluation metrics do not monitor the explanations from integrity or bias perspectives. Our work also uses five human expert annotators and in the annotation process, annotators need to rate the model generated candidate rationals. While the model-generated explanations can produce potentially unsafe content, the datasets for annotations include domains related to logical and arithmetic concepts and general commonsense knowledge. The anecdotal consensus was that the generations were safe and didn’t include biased statements. 

## REPRODUCIBILITY STATEMENT 

To ensure the reproducibility of our empirical results, we will open source our code to Github, which will contain: instructions for installing the virtual environment, data preprocessing, all score generation and correlation scripts (both for ROSCOE and baselines), and trained embedding models. Detailed explanation of all the finetuned models and metrics are given in the main paper as well as in the Appendices. We will also release all the diagnostic and human judgment datasets used in our experiments. 

## REFERENCES 

- Shourya Aggarwal, Divyanshu Mandowara, Vishwajeet Agrawal, Dinesh Khandelwal, Parag Singla, and Dinesh Garg. Explanations for CommonsenseQA: New Dataset and Models. 2021. 

- Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, and Weiwei Guo. *SEM 2013 shared task: Semantic textual similarity. In _Second Joint Conference on Lexical and Computational Semantics (*SEM), Volume 1: Proceedings of the Main Conference and the Shared Task: Semantic Textual Similarity_ , pp. 32–43, Atlanta, Georgia, USA, June 2013. Association for Computational Linguistics. URL https: //aclanthology _._ org/S13-1004. 

Alan Agresti. _Analysis of ordinal categorical data_ , volume 656. John Wiley & Sons, 2010. 

- Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pp. 632–642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10 _._ 18653/v1/D15-1075. URL https://aclanthology _._ org/ D15-1075. 

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. 33:1877–1901, 2020. URL https://proceedings _._ neurips _._ cc/paper/ 2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper _._ pdf. 

- Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. e-snli: Natural language inference with natural language explanations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), _Advances in Neural Information Processing Systems 31_ , pp. 9539– 9549. Curran Associates, Inc., 2018. URL http://papers _._ nips _._ cc/paper/8163-e-snlinatural-language-inference-with-natural-language-explanations _._ pdf. 

10 

Published as a conference paper at ICLR 2023 

- Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao. Evaluation of text generation: A survey. _CoRR_ , abs/2006.14799, 2020. URL https://arxiv _._ org/abs/2006 _._ 14799. 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_ , 2022. 

- Miruna-Adriana Clinciu, Arash Eshghi, and Helen Hastie. A study of automatic metrics for the evaluation of natural language explanations. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pp. 2376–2387, Online, April 2021. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2021 _._ eacl-main _._ 202. URL https:// aclanthology _._ org/2021 _._ eacl-main _._ 202. 

- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_ , 2021. 

- Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. arXiv, 2022. URL https://arxiv _._ org/abs/2205 _._ 09712. 

- Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. Explaining answers with entailment trees. _EMNLP_ , 2021. 

- Marina Danilevsky, Kun Qian, Ranit Aharonov, Yannis Katsis, Ban Kawas, and Prithviraj Sen. A survey of the state of explainable AI for natural language processing. In _Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing_ , pp. 447–459, Suzhou, China, December 2020. Association for Computational Linguistics. URL https://aclanthology _._ org/2020 _._ aacl-main _._ 46. 

- Mingkai Deng, Bowen Tan, Zhengzhong Liu, Eric P. Xing, and Zhiting Hu. Compression, transduction, and creation: A unified framework for evaluating natural language generation. In _EMNLP_ , 2021. URL https://aclanthology _._ org/2021 _._ emnlp-main _._ 599 _._ pdf. 

- Daniel Deutsch, Rotem Dror, and Dan Roth. Re-examining system-level correlations of automatic summarization evaluation metrics. _arXiv preprint arXiv:2204.10216_ , 2022. 

- Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10 _._ 18653/v1/N19-1246. URL https: //aclanthology _._ org/N19-1246. 

- Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. _arXiv preprint arXiv:2104.08821_ , 2021. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , 9:346–361, 2021. 

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _NeurIPS_ , 2021. 

11 

Published as a conference paper at ICLR 2023 

- Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Cosmos QA: Machine reading comprehension with contextual commonsense reasoning. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 2391–2401, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10 _._ 18653/v1/D19-1243. URL https://aclanthology _._ org/D19-1243. 

- Alon Jacovi and Yoav Goldberg. Aligning faithful interpretations with their social attribution. volume 9, pp. 294–310, Cambridge, MA, 2021. MIT Press. doi: 10 _._ 1162/tacl_a_00367. URL https: //aclanthology _._ org/2021 _._ tacl-1 _._ 18. 

- Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yejin Choi, and Yoav Goldberg. Contrastive explanations for model interpretability. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 1597–1611, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2021 _._ emnlp-main _._ 120. URL https://aclanthology _._ org/2021 _._ emnlp-main _._ 120. 

- Kalpesh Krishna, John Wieting, and Mohit Iyyer. Reformulating unsupervised style transfer as paraphrase generation. In _Empirical Methods in Natural Language Processing_ , 2020. 

- Andrew K Lampinen, Nicholas Roy, Ishita Dasgupta, Stephanie Cy Chan, Allison Tam, James Mcclelland, Chen Yan, Adam Santoro, Neil C Rabinowitz, Jane Wang, and Felix Hill. Tell me why! Explanations support learning relational and causal structure. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 11868–11890. PMLR, 17–23 Jul 2022. URL https://proceedings _._ mlr _._ press/v162/lampinen22a _._ html. 

- Moritz Laurer, Wouter van Atteveldt, Andreu Casas, and Kasper Welbers. Less annotating, more classifying– addressing the data scarcity issue of supervised machine learning with deep transfer learning and bert-nli. 2022. 

- Christoph Leiter, Piyawat Lertvittayakumjorn, Marina Fomicheva, Wei Zhao, Yang Gao, and Steffen Eger. Towards explainable evaluation metrics for natural language generation. _CoRR_ , abs/2203.11131, 2022. URL https://doi _._ org/10 _._ 48550/arXiv _._ 2203 _._ 11131. 

- Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 7871–7880, Online, July 2020. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2020 _._ acl-main _._ 703. URL https://aclanthology _._ org/2020 _._ aclmain _._ 703. 

- Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. _arXiv preprint arXiv:2206.14858_ , 2022. 

- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. On the advance of making language models better reasoners. arXiv, 2022. URL https://arxiv _._ org/abs/ 2206 _._ 02336. 

- Chao-Chun Liang, Yu-Shiang Wong, Yi-Chung Lin, and Keh-Yih Su. A meaning-based statistical English math word problem solver. pp. 652–662, June 2018. doi: 10 _._ 18653/v1/N18-1060. URL https: //aclanthology _._ org/N18-1060. 

12 

Published as a conference paper at ICLR 2023 

- Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In _Text Summarization Branches Out_ , pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https: //aclanthology _._ org/W04-1013. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ , 2019. 

- Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. pp. 975–984, 2020. 

- Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. _arXiv preprint arXiv:2112.00114_ , 2021. 

- Juri Opitz and Anette Frank. Towards a decomposable metric for explainable evaluation of text generation from AMR. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pp. 1504–1518, Online, April 2021. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2021 _._ eacl-main _._ 129. URL https://aclanthology _._ org/ 2021 _._ eacl-main _._ 129. 

- Simon Ostermann, Michael Roth, Ashutosh Modi, Stefan Thater, and Manfred Pinkal. Semeval-2018 task 11: Machine comprehension using commonsense knowledge. In _*SEMEVAL_ , 2018. URL https: //aclanthology _._ org/S18-1119 _._ pdf. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 1(8):9, 2019. 

- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_ . Association for Computational Linguistics, 11 2019. URL https://arxiv _._ org/abs/1908 _._ 10084. 

- Ehud Reiter. Natural language generation challenges for explainable AI. In _Proceedings of the 1st Workshop on Interactive Natural Language Technology for Explainable Artificial Intelligence (NL4XAI 2019)_ , pp. 3–7. Association for Computational Linguistics, 2019. doi: 10 _._ 18653/v1/W19-8402. URL https: //aclanthology _._ org/W19-8402. 

- Christopher Michael Rytting and David Wingate. Leveraging the inductive bias of large language models for abstract textual reasoning. 2021. URL https://openreview _._ net/forum?id=urueR03mkng. 

- Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning robust metrics for text generation. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 7881–7892, Online, July 2020. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2020 _._ acl-main _._ 704. URL https://aclanthology _._ org/2020 _._ acl-main _._ 704. 

- Robert H Somers. A new asymmetric measure of association for ordinal variables. _American sociological review_ , pp. 799–811, 1962. 

- Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pp. 3621–3634, Online, August 2021. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2021 _._ findings-acl _._ 317. URL https://aclanthology _._ org/2021 _._ findingsacl _._ 317. 

13 

Published as a conference paper at ICLR 2023 

- Brian Thompson and Matt Post. Automatic machine translation evaluation in many languages via zeroshot paraphrasing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 90–121, Online, November 2020. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2020 _._ emnlp-main _._ 8. URL https://aclanthology _._ org/2020 _._ emnlp-main _._ 8. 

- Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey,[˙] Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. _Nature Methods_ , 17:261–272, 2020. doi: 10 _._ 1038/s41592-019-0686-2. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. _arXiv preprint arXiv:2201.11903_ , 2022. 

- Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, and Yejin Choi. Naturalprover: Grounded mathematical proof generation with language models. 2022. 

- Peter West, Chandra Bhagavatula, Jack Hessel, Jena Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. Symbolic knowledge distillation: from general language models to commonsense models. In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 4602–4625, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10 _._ 18653/v1/2022 _._ naacl-main _._ 341. URL https://aclanthology _._ org/2022 _._ naacl-main _._ 341. 

- Weizhe Yuan, Graham Neubig, and Pengfei Liu. Bartscore: Evaluating generated text as text generation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), _Advances in Neural Information Processing Systems_ , volume 34, pp. 27263–27277. Curran Associates, Inc., 2021. URL https://proceedings _._ neurips _._ cc/paper/2021/file/ e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Paper _._ pdf. 

- Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. Star: Bootstrapping reasoning with reasoning. arXiv, 2022. 

- Tianyi Zhang, Varsha Kishore, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In _International Conference on Learning Representations_ , 2020. URL https: //openreview _._ net/forum?id=SkeHuCVFDr. 

14 

Published as a conference paper at ICLR 2023 

## APPENDIX 

## A LIMITATIONS 

Our study is the first initial step that investigates the evaluation of the step-by-step reasoning produced by large language models. Our taxonomy (in Table 2) covers several reasoning errors and we designed our metrics to evaluate a spectrum of criteria including the ones in the taxonomy. Even though we cannot say we cover all possible reasoning errors, our metrics are generic enough, work on natural language rationales, and consider the alignment with the input context and the generated explanation. Nevertheless, we believe our study can spur others to investigate different reasoning errors and use our code and datasets as templates to extend further. 

Due to the extensive analysis needed to thoroughly test and communicate the ability of our proposed metrics to capture reasoning errors, we decided to leave some follow-up questions, such as the application of these metrics for improving downstream task performance, for future exploration. 

## B FEW-SHOT PROMPTING EXAMPLES (CONT. FROM § 1) 

Below is the 2-shot example we used to generate the explanations from GPT-3 as we show in the Fig. 1. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0015-07.png)


**----- Start of picture text -----**<br>
||||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|I will explain how to make logical inferences:|
|Context:|John wakes to the sound of his children giggling outside the door.|They bust in and say|
|"Happy Father’s Day!" with a plate full of food.|2022 has been a great year.|
|Question:|Can we infer that John was born before the inauguration of Barack Obama?|
|Explanation|:|We|can|infer|that|John|is|a|father|because|of|"his|children".|We|can|infer|that|the|
|current year is 2022.|Barack Obama was inaugurated in 2009, so it has been about 13 years since|
|Obama was inaugurated.|John’s children are old enough to speak, so if John were born after Obama’s|
|inauguration, he would have been younger than 10 when he had children.|So, we can infer that John|
|was born before Obama’s inauguration.|The answer is "yes".|
|Context:|Sam sees a piano in the subway stations.|Laughing, his friends prod him to play, saying|
|"Remember how amazing your rendition of Chopsticks was at the Christmas party?"|
|Question|:|Can we infer that John majored in piano at Juilliard?|
|Explanation|:|Juilliard is a top school for music, so someone who majored in piano at Juilliard would|
|be|an|expert|pianist.|"Chopsticks"|is|a|very|simple|song|to|play|on|the|piano.|His|friends|were|
|therefore likely being sarcastic when they told him to play, which would imply that Sam was not an|
|expert piano player.|So it is unlikely that Sam majored in piano at Juilliard.|The answer is "no".|

**----- End of picture text -----**<br>


Table 9 illustrates sample if in-context examples we used as prompts to obtain step-wise explanations. 

15 

Published as a conference paper at ICLR 2023 

Table 9: Example prompts used to generate _step-by-step_ explanations using GPT-3. 

(Used for ESNLI dataset) Explain if premise entails hypothesis. **Premise** : Tomato sauce (also known as Neapolitan sauce, and referred to in Italy as Salsa di pomodoro) refers to any of a very large number of sauces made primarily from tomatoes, usually to be served as part of a dish (rather than as a condiment). Tomato sauces are common for meat and vegetables, but they are perhaps best known as sauces for pasta dishes. **Hypothesis** : Tomato sauce is only rarely used to top pasta. **Explanation** : Salsa di pomodoro is made of peppers. The answer is No. **Premise** : Kirk Johnson (born 1960) is an American scientist, author, curator, and museum administrator, and is currently serving as Director of Smithsonian’s National Museum of Natural History. Dr. Johnson is the host of the PBS Nova series, Making North America[¨] ¨, which is a three-part series that describes the shaping of North America, which aired on November 4, 11 and 18, 2015. **Hypothesis** : johnson hosted Making North America when he turned 50. **Explanation** : Johnson was born in 1960 and hosted Making North America in 2015, so he was 55, not 50. The answer is No. **Premise** : A person rides his bicycle in the sand beside the ocean. **Hypothesis** : A person is on a beach. **Explanation** : A person rides beside the ocean means is on a beach if there is sand beside the ocean. The answer is Yes. (used for SEMEVAL dataset) Explain if premise entails hypothesis. **Premise** : I went into my bedroom and flipped the light switch. Oh, I see that the ceiling lamp is not turning on. It must be that the light bulb needs replacement. I go through my closet and find a new light bulb that will fit this lamp and I place it in my pocket. I also get my stepladder and I place it under the lamp. I make sure the light switch is in the off position. I climb up the ladder and unscrew the old light bulb. I place the old bulb in my pocket and take out the new one. I then screw in the new bulb. I climb down the stepladder and I place it back into the closet. I then throw out the old bulb into the recycling bin. I go back to my bedroom and turn on the light switch. I am happy to see that there is again light in my room. Which room did the light go out in? **Hypothesis** : Bedroom **Label** : Correct **Explanation** : The answer is bedroom, because the light went out in the bedroom. The person replaced the light bulb in the bedroom. 

**Premise** : The weather was so nice today that I decided to have a barbecue. I called up some of my friends to invite them over. I set up some chairs and tables outside for my friends to sit around. I then took out some hot dogs from my fridge and brought them over to my barbecue. My barbecue is a gas barbecue, so all I had to do was press the knob and turn it to ignite it. Once I got a little fire going I put my hot dogs on the grill over it to cook. I closed the lid and left my hot dogs to cook for about ten minutes. After ten minutes I decided to check on them to see if they were done. They looked like they were cooked, so I put them all on a plate to serve to my friends. I went to my kitchen and got hot dog buns, ketchup, mustard, and relish to serve on top of the hot dogs. I laid out all of the food and condiments on the table so my friends could help themselves. We ate the hot dogs and relaxed around the table. Why are we having the barbecue, to celebrate an occasion, or just because we want to? **Hypothesis** : Just because they want to. **Label** : Correct **Explanation** : The answer is correct, because the weather was nice. They decided to have a barbecue. There is no mention of any special occasion. 

**Premise** : Another Nurse can in to look at me , took one look at my rash and said OMG and ran out of the room . At that point I started crying , regretted coming to the hospital alone and started wheezing again . Then the doctor came in , her eyes were all bugged out . What will I do after the doctor examines me ? **Hypothesis** : I will talk with my friends . **Label** : Incorrect **Explanation** : The premise is about the nurse’s reaction to the patient’s rash, and how the patient feels. The hypothesis does not follow up from this premise correctly. Instead, it talks about talking with friends. 

16 

Published as a conference paper at ICLR 2023 

## C TAXONOMY OF REASONING ERRORS (CONT. FROM § 3) 

To gain deeper insights into the types of reasoning errors introduced by LLMs while explaining their decisions, we propose a new taxonomy of generic reasoning errors for language problem solving. Specifically, we sampled from the training portions of the logical inference and commonsense reasoning datasets, and prompted GPT-3 with reasoning explanations using prompts similar to App. B. We used task specific indomain examples for prompting. We also analyzed model generated explanations shared in Wei et al. (2022). We then manually looked into each explanation and identified potential errors that are inconsistent with the source, question or the prompt and within the reasoning chain. Some tasks require a model to classify the logical relationship between premise and a hypothesis, others are question and answering tasks. We adjusted our context and prompts according to the type of the task. 

Our reasoning error taxonomy is summarized in Table 10. It contains types of errors concerning an overall chain or an individual step. Specifically, the _chain-level coarse-grained_ evaluations of the overall reasoning chain deals with overall quality of the step-by-step thinking, coherence, consistency of the explanation within itself, and consistency with the context, etc. On the other hand the _step-level fine-grained_ evaluations focus on the consistency of a reasoning step with the previous steps, if a step conveys new and supporting information over the previous steps, factuality or logical inference issues. We use these error categories to construct diagnostics datasets with perturbed errors as well as human judged datasets of reasoning errors. In the taxonomy, we indicate *-step level errors to differentiate from the chain level error types. 

17 

Published as a conference paper at ICLR 2023 

Table 10: Taxonomy of Step-by-Step Reasoning Errors. Errors used for perturbations in constructing the diagnostic datasets (Diag.) and for human annotation (Human) of the model generated reasoning chains are also marked. (Cont. from Table 2. 

|from Table2.|||||
|---|---|---|---|---|
|**Error Type**|**Description**|**Example**|**Diag. **|**Human**|
|Grammar|Faulty, unconventional, or contro-|**Reference**: He chews 4 pieces of gum a day so over 30 days|✓|✓|
||versial grammar usage|he will chew 4*30 = «4*30=120»120 pieces of gum.|||
|||**Model Expl**: He eats 4 pieces of gum a day and lasts 30 days|||
|||so he eats 4*30 = «4*30=120»120pieces ofgum.|||
|Factuality|Information about object (i.e. quan-|**Context**: Anakin caught 10 fsh.|✓|✓|
||tity, characteristics) or a personal|**Model Expl**: Locsincaught 10 fsh.|||
||named entity does not match infor-||||
||mationprovided in thequestion||||
|Hallucination|Information is not provided in the|**Context**: The basketball team went to the steakhouse to eat|✓|✓|
||problem statement and is irrelevant|dinner. The frst player ate a 6-ounce steak. The second|||
||or wrong|player ate beef tips, containing 8 beef tips, each an ounce|||
|||in size. The third player ate a one-pound steak. And the|||
|||fourth and ffth players ordered vegetarian meals. In total,|||
|||how many ounces of meat were consumed by the team?|||
|||**Model Expl**: The fourth and ffth players orderedvegetarian|||
|||meals, for a total of 2 ounces of meat.|||
|Redundancy|Explanation contains redundant in-|**Context**: Bob is cold. If something is quiet then it is blue.||✓|
||formation,<br>which even though|All blue things are round. If something is round and smart|||
||might be factual, is not required to|then it is cold. Question: Bob is cold.|||
||answer the question|**Model Expl**: If something is round and not smart then it is|||
|||blue. All blue things are round. Answer: true.|||
|Repetition|Step paraphrases information al-|**Context**: Parker chews 4 pieces of gum a day. A pack of gum|✓|✓|
||ready mentioned in previous reason-|has 15 pieces of chewing gum per pack. How many packs of|||
||ing steps|gum will he need to last him 30 days?|||
|||**Model Expl.**: He chews 4 pieces of gum a day and there are|||
|||30 days in a month so he chews 4*30 = 120 pieces of gum a|||
|||month. Parker chews 120 pieces of gum a month. He needs|||
|||120 pieces of gum a month and a pack of gum has 15 pieces|||
|||ofgum so he will need 120/15 = 8packs ofgum.|||
|Missing step|The content of the generated rea-|**Context**: A farm has 10 2-legged animals and 15 4-legged|✓|✓|
||soning is incomplete and lacks re-|animals. If all animals are healthy, how many pairs of animal|||
||quired information to produce the|legs in total are on the farm? **Answer**: 40|||
||correct answer. If this missing steps|**Model Expl.**: 10 2-legged animals have 10*2=20 legs. 15|||
||are added, the model would produce|4-legged animals have 15*4=60 legs. In total on the farm,|||
||the correct answer|there are 20+60=80 legs.**Answer**: 80 missing step: Total|||
|||number ofpairs is 80/2=40|||
|Coherency|Steps contradict each other or do|**Context**: If the reactor is intact, then it is safe. If the reactor|✓|✓|
||not follow a cohesive story|is safe, then no graphite is outside it.|||
|||**Model Expl.**: The reactor is intact. All graphite is inside of|||
|||it. There are somegraphite outside of the the reactor.|||
|Commonsense|Model lacks relations that should|**Question**: Jordan has 2 children who wear diapers. Each||✓|
||be known from general knowledge|child requires 5 diaper changes per day.<br>Jordan’s wife|||
||about the world (i.e. how to com-|changes half of the diapers. How many diapers does Jor-|||
||pute velocity, how many inches in|dan change per day?|||
||one foot, all ducks are birds, etc.)|**Model Expl.**: Jordan has 2*5=10 diapers to change per day.|||
|||Jordan’s wife changes 10/2=5.0 diapers per day. For both|||
|||children, Jordan changes10+5=15diapersper day.|||
|Arithmetic|Error in math calculations|**Reference**: If a truck was carrying 20 tons of fertiliser packed|✓|✓|
|||in bags, the total number of bags in a truck is 20*20 = 400|||
|||bags|||
|||**Model Expl**: If a truck was carrying 20 tons of fertiliser|||
|||packed in bags, the total number of bags in a truck is20*20|||
|||= 40bags|||



18 

Published as a conference paper at ICLR 2023 

## D ROSCOE METRICS DETAILS (CONT. FROM _§_ 4) 

ROSCOE metrics are constructed under four categories: semantic alignment, semantic similarity, logical inference, and logical coherence. The details of each metric is explained in _§_ 4. At the core of ROSCOE semantic alignment metrics is the reasoning alignment score, which we designed to measure the grounding of step-by-step reasoning with respect to the source text. Fig. 3 illustrates the reasoning alignment. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0019-03.png)


Figure 3: **Reasoning alignment** illustrating the measurement of the Faithfulness-Step and Faithfulness-Token semantic alignment scores. _**h**_ = _{h_ 1 _, h_ 2 _}_ is a hypothesis chain with tokens _{h_ 1 _,_ 1 _, h_ 1 _,_ 2 _, h_ 1 _,_ 3 _, h_ 2 _,_ 1 _, h_ 2 _,_ 2 _}_ , and _**s**_ = _{s_ 1 _, s_ 2 _, s_ 3 _}_ is a context with tokens _{s_ 1 _,_ 1 _, s_ 2 _,_ 1 _, s_ 2 _,_ 2 _, s_ 2 _,_ 3 _, s_ 3 _,_ 1 _, s_ 3 _,_ 2 _, s_ 3 _,_ 3 _}_ . Alignment scores from hypothesis to context are highlighted, and alignment scores from context to hypothesis are underscored. The reasoning alignment combines token and step level similarities where each alignment value (cell) is the cosine similarity and explicitly measures the grounding of the token and step-wise reasoning with respect to the source text. 

The variation of scorers of the ROSCOE shares some similarities, thus we explain them here: 

**BARTScore** (Yuan et al., 2021) claims that more high level text can be generated using sequence to sequence model. It can support different evaluation perspectives such as factuality (by evaluating from source to hypothesis) or informativeness (by evaluating from both directions between reference and hypothesis). BARTScore is used to measure the probability of generated text from a source text _x_ to a target set _y_ : 

_BARTScore_ =[�] _[m] t_ =1 _[w][t]_[ log] _[ p]_[(] _[y][t][|][y][<t][, x, θ]_[)] (1) BARTScoreintroduce two variations: (1) finetuning, in which the BART model is finetuned on the task specific dataset to make the pre-training domain closer to the evaluation domain. (2) prompting, in which a task specific textual prompt is appended to the source _x_ to get the _y_ . In our experiments we compare the the BARTScorebaseline and one with the prompting variant BARTScore+to compare in the experiments. 

**CTC** (Compression, Transduction, and Creation) (Deng et al., 2021), is a suite of metrics that unifies different perspectives of different tasks (e.g, summarization, style transfer, or text rewriting) into information alignment, which measures weather the information in one generation component is grounded in another. The information alignment is defined as follows: let _x_ (e.g, dialog context) be the source input, _c_ (e.g., external world knowledge) be some additional context, and _y_ be the generated output text (e.g., generated response). The alignment is measured on token level and it is measured as the vector of scores: 

_align_ ( _a → b_ ) = _⟨α_ 1 _, · · · , αN ⟩_ (2) 

where each score _αi_ indicates confidence that the _n_ -th token in _a_ aligns with the whole sentence _b_ . Using the information alignment they define a list of metrics to evaluate text for different tasks. In our experiments we use two of these metrics that are closer to ROSCOE: the _Relevance_ (CTC Relevance), which measures the consistency of the generated text with the source and its balanced between the reference, and the _Consistency_ (CTC Consistency) which deals with the faithfullness of the generated text to the input context by the alignment between the two. 

19 

Published as a conference paper at ICLR 2023 

## E EXPERIMENTAL SETUP DETAILS (CONT. FROM _§_ 5) 

## E.1 DIAGNOSTIC DATASETS 

In the following we present details of each diagnostics dataset used in our work. Table 11 illustrates how each dataset is used in our experiments. StrategyQA dataset is only used to finetune the SimCSE embeddings model, because it contains reference reasoning chains in train and validation partitions, but not in the test partition. The rest of the six diagnostic datasets are used for sentence embedding model finetuning, and evaluating our models as presented in the experiments results. All datasets with examples are summarised in Table 12. 

Table 11: Summary of datasets used in our work. Reasoning Chain represent whether it contains human written golden step-wise reasoning explanation. Type indicates whether it is used for constructing Diagnostic or Human judged datasets. Train/Val./Test indicate whether the dataset is used for training, validation and/or testing. StrategyQA dataset is only used for finetuning SimCSE embedding model. 

|Dataset|Reasoning|Type|Train|Val.|Test|Annotated|
|---|---|---|---|---|---|---|
||Chain|||||Instances|
|EntailmentBank (Dalvi et al.,2021)|✓|Diagnostic, Finetuning|✓|✓|✓|1,840|
|ProofWriter (Tafjord et al.,2021)|✓|Diagnostic, Finetuning|✓|✓|✓|272,430|
|MATH (Hendrycks et al.,2021)|✓|Diagnostic, Finetuning|✓|✓|✓|12,500|
|ASDIV (Miao et al.,2020)|✓|Diagnostic, Finetuning|✓|✓|✓|2,305|
|AQUA (Liang et al.,2018)|✓|Diagnostic, Finetuning|✓|✓|✓|97,975|
|EQASC (Aggarwal et al.,2021)|✓|Diagnostic, Finetuning|✓|✓|✓|9,060|
|StrategyQA (Geva et al.,2021)|✓|Finetuning|✓|✓|✗|2,290|
|DROP (Dua et al.,2019)|✗|Human judged|✗|✗|✓|210|
|GSM8K (Cobbe et al.,2021)|✓|Human judged|✗|✗|✓|200|
|ESNLI (Camburu et al.,2018)|✓|Human judged|✗|✗|✓|151|
|CosmosQA (Huang et al.,2019)|✗|Human judged|✗|✗|✓|195|
|SemEval (Ostermann et al.,2018)|✗|Human judged|✗|✗|✓|209|



**EntailmentBank (EntBank)** (Dalvi et al., 2021) is a complex question answering dataset which contains multi-step entailment trees, namely a tree of multi-premise entailment steps from facts that are known, through intermediate conclusions to hypothesis of interest (which in this case the question and answer). 

**ProofWriter** (Tafjord et al., 2021) is a question answering dataset for logical reasoning. It contains 500k questions, answers and proofs over natural-language rulebases. This dataset is mostly used to emulate reasoning over rules expressed in language, including proof generation. The datasets proofs include intermediate conclusions. In our experiments, we used _depth-0_ , _depth-1_ , _depth-2_ , _depth-3_ , and _depth-5 OWA_ sets. 

**MATH** (Hendrycks et al., 2021) is a dataset of 12,500 problems from high school math competitions. Given a math problem such as in Table 12 models generate a sequence, such as[2] 3[, that encodes the final answer.] 

**ASDIV** (Miao et al., 2020) (Academia Sinica Diverse MWP Dataset) is a dataset of 2,305 questions on diverse math word problem solving. It includes a diverse operations such as basic arithmetic or aggregative operations (e.g., comparisons, set-operations). 

**AQUA** (Liang et al., 2018) is a dataset of 100,000 algebraic word problems with step-wise solutions as shown below. In the original dataset each question is decomposed in four parts, two inputs and two outputs: the description of the problem and a question, and the possible (multiple choice) answer options, one being the 

20 

Published as a conference paper at ICLR 2023 

Table 12: We show instances from seven of the **Diagnostics Datasets** here. (Continue from _§_ 5). 

|Ta|ble 12: We show instances from seven of the**Diag**|**nostics Datasets**here. (Continue from _§_5).|
|---|---|---|
|**Dataset**|**Instance**|**Reference Step-by-Step Solution**|
|**EntBank**|Earth is a kind of celestial object. Stars appear to move|**Step1**: earth is a kind of celestial object Its position appears|
||relative to the horizon during the night. A star is a kind|**Step2**: a star is a kind of celestial object to shift relative /|
||of celestial object celestial body. The earth rotating on its|celestial body to the horizon.|
||axis causes stars to appear to move across the sky at night.|**Step3**: apparent motion is when an object appears to move|
||Apparent motion is when an object appears move relative|relative to another object ’s position|
||to another object ’s position.|**Step4**Therefore apparent motion of stars is when stars appear|
||**Question**: How does the appearance of a constellation|to move relative to earth’s position**Step5**: The earth rotating on|
||change during the night?|its axis causes stars to appear to move across the sky at night|
||**Hypothesis:** Solve the following entailment problem:|**Step6**: Therefore the earth rotating on its axis causes apparent|
||"Earth is a kind of celestial object. During the night stars|motion of stars|
||appear to move"|**Step7**: Stars appear to move relative to the horizon during the|
|||night|
|||**Step8**: Therefore the earth rotating on its axis causes stars to|
|||move relative to the horizon during the night.|
|**ProofWriter**|**Facts**: The cow is not big. The cow is not green. The lion|**Step1**: The squirrel is round.|
||eats the tiger. The lion sees the cow. The lion visits the|**Step2**: something is round then it eats the cow.|
||cow. The lion does not visit the squirrel. the lion visits the|**Step3**: The squirrel eats the cow.|
||tiger. The squirrel is big. The squirrel is round. The tiger|**Step4**: If something sees the squirrel and the squirrel eats the|
||is not green. The tiger does not see the cow.|cow then the cow is round.|
||**Rules**: if something sees the squirrel and the squirrel eats|**Step5**: The cow is round.|
||the cow then the cow is round. if something is green then it|**Step6**: If something is round then it eats the cow.|
||eats the tiger. if the squirrel is round then the squirrel visits|**Step7**: The cow eats the cow.|
||the cow. if something eats the cow then it sees the squirrel.|**Step8**: if something eats the cow then it sees the squirrel.|
||if something sees the tiger and the tiger visits the squirrel|**Step9**: the cow sees the squirrel.|
||then it is nice. if something is round then it eats the cow.|**Answer**: True|
||if something is kind then it eats the cow. if the tiger visits||
||the cow then the cow sees the squirrel. if something sees||
||the cow then the cow eats the tiger.||
||**Question:** The cow does not see the squirrel.||
|**MATH**|**Context**: Tom has a red marble, a green marble, a blue|**Step1**: There are two cases here:|
||marble, and three identical yellow marbles.|**Step2**: either Tom chooses two yellow marbles (1 result), or he|
||**Question**: How many different groups of two marbles can<br>Tom choose?|chooses two marbles of different colors (<br>�4<br>2<br>�<br>=6 results.).<br>**Step3**: The total number of distinct pairs of marbles Tom can|
|||choose is 1 + 6 = 7.|
|||**Answer**: 7|
|**ASDIV**|**Context**: A sandwich is priced at $0.75. A cup of pudding|0.75 x 2 + 0.25 x 4 = 2.5|
||is priced at $0.25. Tim bought 2 sandwiches and 4 cups of|**Answer**: 2.5|
||pudding.||
||**Question**: How much money should Tim pay?||
|**AQUA**|**Context**: The entrance fee for a fair is $5 for persons under|**Step1**: Total entrance fee is (2*$5)+(1.20*5) = $16|
||the age of 18 and 20% more for persons older. Each ride at|**Step2**: Total rides fee is ( 0.50 * 3 ) * 3 = $4.50|
||the fair costs $0.50. If Joe goes with her 6 years old twin|**Step3**: Total money spent is $20.50|
||brothers, and they each took 3 rides in total.|**Answer**: 20.5|
||**Question:** How much money does Joe end up spending at||
||the fair?||
|**EQASC**|**Question**: Where is water likely to form beads?|**Step1**: Beads of water are formed by water vapor condensing|
|||**Step2**: Moisture builds up in condenses air and the wherever|
|||the surfaces are cold.|
|||**Answer**: Water beads form on cold surfaces.|
|**StrategyQA**|**Question**: Are more people today related to Genghis Khan|**Step1**: Julius Caesar had three children.|
||than Julius Caesar?|**Step2**: Genghis Khan had sixteen children.|
|||**Step3**: Modern geneticists have determined that out of every|
|||200 men today has DNA that can be traced to Genghis Khan.|
|||**Answer**: True|



correct one. In this work we only used the context and question, the step-wise solution and the correct answer to construct our diagnostic dataset. 

**EQASC** (Aggarwal et al., 2021) is a multi-hop question answering dataset with 98K explanation annotations for multi-step factual reasoning. Each instance in the dataset comes with a question, multiple answer choices, 

21 

Published as a conference paper at ICLR 2023 

explanation of each answer choice and a free flow explanation of the whole context. In our experiments we used the correct answer’s explanation to construct our diagnostic datasets. 

**StrategyQA** (Geva et al., 2021) is another multi-step question answering (QA) dataset, that covers a diverse set of reasoning skills. StrategyQA consists of 2,780 questions, annotated with their decomposition and per-step evidence. 

## E.2 HUMAN JUDGED DATASET CONSTRUCTION 

In the following we present details of each human judged datasets used in our work. Table 11 lists each dataset and illustrates how each dataset is used in our experiments. Specifically, all six datasets are used for evaluations in the experiments results and model finetuning, and one dataset was used for finetuning only. The dataset details are explained below. 

To construct these datasets, we first sample instances from each dataset (see the number of instances sampled in Table 11). We use GPT-3 with few-shot in-context examples and a prompt to generate step-by-step reasoning (e.g., " _explain step-by-step_ ") for each sampled instance (see in-context examples and prompts in App. B). Then, using our taxonomy we constructed a list of evaluation perspectives to label the model generated step-by-step reasoning step of each of these datasets. We explain the details of the perspectives used to label human judged datasets in _§_ 5 and App. F. All datasets with examples are summarised in in Table 13. In the following we present details of each human judged datasets. 

**DROP** (Dua et al., 2019), Discrete Reasoning Over the content of Paragraphs, is a dataset of 96K of instances with context and a question. To solve the tasks, a system must resolve references in the context that match with the question, and perform discrete operations over them (such as addition, counting, or sorting). These operations require comprehensive understanding of the content of the input context. 

**GSM8K** (Cobbe et al., 2021) is a dataset of 8.5K linguistically diverse grade school math word problems. On this dataset, even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. 

**CosmosQA** (Huang et al., 2019) is a dataset of 35K problems that require commonsense-based reading comprehension, formulated as multiple-choice questions. The questions focus on reading between the lines over a diverse collection of people’s everyday narratives, asking such questions as " _what might be the possible reason of ...?_ ", or " _what would have happened if ...?_ ". The dataset does not introduce step-by-step reasoning output, and contains multiple choice answers. 

**ESNLI** (Camburu et al., 2018) is the extended version of the Stanford Natural Language Inference corpus (Bowman et al., 2015) of 570K labeled sentence pairs with entailment or contradiction labels. ESNLI includes human labeled explanations of the entailment decision. 

**SemEVAL** (Ostermann et al., 2018) is a dataset on machine comprehension using commonsense knowledge. It contains questions that require commonsense knowledge for finding the correct answer. 

## E.3 SYNTHETIC DIAGNOSTICS DATASET GENERATION WITH PERTURBATION RULES 

To construct the diagnostics datasets we apply synthetic perturbations on half of the chains from six datasets (for details see App. E.1 and the summary Table 11). Also, in Table 14 we illustrate these synthetic perturbations applied on reasoning steps _{ri}_ of gold reference chains of all the datasets. In there, _**g[∗]**_ indicates a grammar error, which includes changing verb tense, dropping verb, or random word swap. _**s[∗]**_ represents change the semantics of one step in the chain by replacing named entities. To simulate extrinsic hallucinations, we use random steps from other chains within the same dataset. 

22 

Published as a conference paper at ICLR 2023 

Table 13: We show instances from five of the **Human Judged Datasets** used in our work. Only GSM8K and ESNLI include human labeled explanations. 

|**Dataset**|**Instance**|**Reference Answer & Reference Step-by-Step**|
|---|---|---|
|||**Solution**|
|**GSM8K**|**Question**: Tina buys 3 12-packs of soda for a party. Including Tina, 6|**Answer**: 11|
||people are at the party. Half of the people at the party have 3 sodas each,|**Step1**: Tine buys 3 12-packs of soda for 3*12=36|
||2 of the people have 4, and 1 person has 5. How many sodas are left over|sodas|
||when the party is over?|**Step2**: 6 people attend the party, so half of them is|
|||6/2= 3 people|
|||**Step3**: Each of those people drinks 3 sodas, so they|
|||drink 3*3=9 sodas.|
|||**Step4**: Two people drink 4 sodas, which means|
|||they drink 2*4=8 sodas.|
|||**Step5**: With 1 person drinking 5, that brings the|
|||total drank to 5+9+8+3=25 sodas|
|||**Step6**: As Tina started off with 36 sodas, that|
|||means there are 36-25=11 sodas left.|
|**CosmosQA**|**Context**: A woman had topped herself by jumping off the roof of the|**Answer**: The woman would have been stopped|
||hospital she had just recently been admitted to. She was there because|before she left to take the elevator to the top foor|
||the frst or perhaps latest suicide attempt was unsuccessful. She put her|and she would have lived.|
||clothes on, folded the hospital gown and made the bed. She walked||
||through the unit unimpeded and took the elevator to the top foor||
||**Question**: What would have happened to the woman if the staff at the||
||hospital were doing their job properly?||
|**DROP**|**Context**: Denver would retake the lead with kicker Matt Prater nailing a|**Answer**: John Kasay|
||43-yard feld goal, yet Carolina answered as kicker John Kasay ties the||
||game with a 39-yard feld goal. . . . Carolina closed out the half with||
||Kasay nailing a 44-yard feld goal. . . . In the fourth quarter, Carolina||
||sealed the win with Kasay’s 42-yard feld goal.||
||**Question**: Which kicker kicked the most feldgoals?||
|**ESNLI**|**Premise**: A child in a yellow plastic safety swing is laughing as a dark-|**Answer**: neutral|
||haired woman in pink and coral pants stands behind her.|**Explanation**: Child does not imply daughter and|
||**Hypothesis**: A young mother is playing with her daughter in a swing.|woman does not imply mother.|
|**SemEVAL**|**Context**: Now I am going to set the dining table up for dinner. First I put|**Answer**: No|
||away all the stuff that is not supposed to be on the table. Next I clean the||
||table with a tissue paper. Then I arrange some of the decorations on the||
||table. After that I put down the plates and glasses.lastly in the remaining||
||spaces on the table I put down what we are going to eat but I dont put||
||down dessert yet. There is one more thing I do before I am fnished setting||
||up the dining table. I call my family down for dinner. I swept a drop of||
||sweat of my chin and forehead. It was hard hard work but still it was so||
||much fun. Oh no my family is done with dinner and now I have to bring||
||them dessert which is fruit I made myself.||
||**Question**: When did they clean the dining table?||
||**Hypothesis**: After it was set.||



To construct diagnostic data from math datasets, we introduce four additional perturbations to simulate stepwise explanation errors that might arise in arithmetic reasoning task ( _Arithmetic error_ ), general knowledge about relationships and equation construction ( _Common sense error_ ), and misinformation about object/subject characteristics ( _Factuality_ or _Hallucination_ ): 

- **Shuffle numbers** : randomly shuffles all numbers in the chain, 

- **Shuffle operations** : randomly shuffles all math operations in the chain, 

- **Random number** : randomly replaces one number in the chain, 

- **Random operation** : randomly replaces one math operation in the chain. 

23 

Published as a conference paper at ICLR 2023 

Table 14: Synthetic perturbations and corresponding error types of steps _{ri}_ in reference chains used when constructing diagnostics datasets. _**g[∗]**_ ( _·_ ) represents grammar error, _**s[∗]**_ ( _·_ ) represents semantic change. 

|**Perturbation Type**|**Error Type**|**Reference Reasoning Steps**|**Hypothesis Reasoning Steps**|
|---|---|---|---|
|Repeat a step|Repetition|[_r_1_, r_2_, r_3]|[_r_1_, r_2_, r_2_, r_3]|
|Remove a step|Missing step|[_r_1_, r_2_, r_3]|[_r_2_, r_3]|
|Shuffe steps|Self-coherency|[_r_1_, r_2_, r_3]|[_r_3_, r_1_, r_2]|
|Swap a step|Self-coherency|[_r_1_, r_2_, r_3]|[_r_2_, r_1_, r_3]|
|Negate a step|Factuality|[_r_1_, r_2_, r_3]|[_r_1_, ¬r_2_, r_3]|
|Hallucination|Hallucination|[_r_1_, r_2_, r_3]|[_r_1_, r_2_, r_3_, r_4]|
|Grammar error|Grammatical|[_r_1_, r_2_, r_3]|[_r_1_, r_2_,_**_g∗_**(_r_3)]|
|Semantic change|Factuality|[_r_1_, r_2_, r_3]|[_r_1_,_**_s∗_**(_r_2)_, r_3]|



24 

Published as a conference paper at ICLR 2023 

## F HUMAN ANNOTATIONS (CONT. FROM _§_ 5) 

To construct **Human Judged Datasets** , we perform human annotations on five datasets which we summarize in Table 11 (Type=’Human judged’). These datasets do not include explanations (except GSM8K and ESNLI), so we construct model generated reasoning steps and label them with reasoning errors. We explain our generation process in _§_ 5 and App. E.2. We used five expert human annotators to collect reasoning error labels on five datasets. We asked human evaluators to directly rate the generated reasoning errors on overall chain level using a Likert scale from 1 to 5. We also asked them to mark whether each error type proposed in our error taxonomy ( _§_ 3) appeared in each step in step-level evaluations. In Fig. 4 and Fig. 5 we illustrate the UI used to collect the data. Table 15 summarizes questions that experts were asked. Table 16 reports the distribution of errors for each dataset. In general, we found that it was hard to get anonymous crowd workers to annotate our data accurately even when we paid averages of upwards of $30 an hour, hence relying on expert annotators. For the annotation sessions reported in the text of the paper, we find that it takes an average of 754 seconds for expert annotators to complete a session of at most 5 examples, or slightly over 2-and-a-half minutes per example. This highlights the difficulty of obtaining high-quality annotations on these cognitive challenging tasks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0025-03.png)


Figure 4: Screenshot of expert annotation user interface, showing the context for the initial question as well as the questions regarding the generated response. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0025-05.png)


Figure 5: Screenshot of expert annotation user interface, showing questions asked for each step, using the question in Fig 4. The questions are asked of every step generated by the model, with steps separated by sentence-ending periods. 

25 

Published as a conference paper at ICLR 2023 

Table 15: Evaluation perspectives used to **Human Judged** the datasets. The perspectives, which we used to ask humans to label, align with our taxonomy of reasoning errors. (Continued from _§_ 5) 

|**Level**||**Evaluation**|**Label**|**Details**|
|---|---|---|---|---|
|||**Perspective**|||
|Overall|QUAL|Overall quality|[1-5]|Does the generated response answer the question in a well-justifed|
|||||manner? (1=incomprehensible and wrong, 5=clear and correct)|
|Overall|COH|Coherency|[1-5]|Does the whole generated response make sense? (Ie, does it sound|
|||||understandable/non-contradictory/sensical, even if it fails to address|
|||||the context?) - (1=sounds like nonsense, 5=easy to parse).|
|Step|MISS|Missing Step|Y/N|Is the reasoning in the generated response incomplete and lacking|
|||||required information to produce the correct answer? Specifcally, does|
|||||this response contains steps that, if added in, would make for a well-|
|||||supported chain?|
|Step|GRAM|Grammar|Y/N|Does this step contain faulty, unconventional, or controversial grammar|
|||||usage? In other words, does the language in this step sounds unnatural?|
|Step|FACT|Factuality|Y/N|Does this step contain information that contradicts the context while|
|||||still largely talking about the same concepts? (Ex. Characteristics of|
|||||named objects are wrong, named entities changed.)|
|Step|LOGIC|Coherency and Logic|Y/N|Does this step any logical deduction errors (Ie, makes a conclusion|
|||||contradictory to previously stated clauses, including clauses within this|
|||||step itself; makes a conclusion while not having enough support to|
|||||make the conclusion)|
|Step|HALL|Hallucination|Y/N|Does this step contain information not provided in the problem state-|
|||||ment that is irrelevant or wrong?|
|Step|RED|Redundancy|Y/N|Does this step contain information not required to answer the question|
|||||asked despite being factual and consistent with the context?|
|Step|REP|Repetition|Y/N|Does this step contain any information, possibly paraphrased, already|
|||||mentioned in previous step (and thus could be dropped without impact-|
|||||ing correctness)?|
|Step|COMMON|Commonsense|Y/N|Does this step contain any errors in relation to general knowledge about|
|||||the world (i.e. how to compute velocity, how many inches in one foot,|
|||||etc) not explicitly provided in the context?|
|Step|MATH|Arithmetic|Y/N|Does this step contain math equation errors? Note that you should|
|||||consider only current step in isolation, rather than issues propagated|
|||||from prior steps.|



Table 16: Statistics of types of errors in **Human Judged** datasets. Each column reports the number of examples where the specified error type exists. (Continue from _§_ 5) 

|Error Type|**DROP **|**GSM8K **|**ESNLI **|**COSMOS **|**SemEVAL**|
|---|---|---|---|---|---|
|Grammar|8|4|5|8|6|
|Factuality|19|56|15|44|31|
|Hallucination|4|8|4|9|2|
|Redundancy|25|13|14|15|19|
|Repetition|2|2|0|3|3|
|Missing Step|109|81|40|99|67|
|Coherency|20|57|17|48|17|
|Commonsense|3|58|5|18|1|
|Arithmetic|2|7|1|0|0|



26 

Published as a conference paper at ICLR 2023 

## G SENTENCE EMBEDDING MODEL TRAINING (CONT. FROM _§_ 6) 

**Model training.** We use the train portions of the perturbed diagnostics datasets to finetune the SimCSE embeddings model (explained in § 5) and validation portions to select the best embedding model. The test portions are used to evaluate our metrics against baseline metrics. We randomly select 500,000 samples with replacement from each dataset to create uniform representation and reduce bias. 

The hyperparameters used to finetune SimCSE model are described in Table 17. We use NVIDIA Tesla V100 Volta GPU instances with 32GB Graphics Card. We perform hyperparameter search, varying batch size in _{_ 32 _,_ 64 _,_ 256 _,_ 512 _,_ 1024 _,_ 2048 _}_ , learning rate in _{_ 5 _e_ -06 _,_ 1 _e_ -05 _,_ 5 _e_ -05 _,_ 1 _e_ -04 _}_ , and max sequence length in _{_ 64 _,_ 128 _,_ 512 _}_ . Not all combinations of batch size and max sequence length were explored due to memory limitations. 

Table 17: Hyperparameters used to fine-tune SimCSE model on perturbed datasets. 

|Parameter|Value|
|---|---|
|Batch size|64|
|Max sequence length|512|
|Training epochs|5|
|Learning rate|5e-6|
|Temperature|0.05|



**Validation.** We replace original validation procedure on semantic textual similarity tasks with similaritybased validation on perturbed reasoning chains. In particular, during training, we select best checkpoint that maximizes cosine similarity between positive and minimizes cosine similarity between hard-negative pairs within the batch of size _B_ as the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0027-07.png)


Model is evaluated every 100 steps on the development dataset and the best checkpoint is applied at the inference. Other parameters not described in this section are kept as in the original SimCSE model used for initialization. 

**Inference** . We compare ROSCOE scores calculated against three embeddings: finetuned SimCSE model, _sup-simcse-roberta-base_ SimCSE model, and _all-mpnet-base-v2_ sentence embedding model (Reimers & Gurevych, 2019). During inference, we set the random seed to 42. Without this, the embedding-based scores naturally varied by about 0.01. 

27 

Published as a conference paper at ICLR 2023 

## H ADDITIONAL EXPERIMENTAL RESULTS (CONT. FROM _§_ 6) 

## H.1 CONTROLLED EXPERIMENTS WITH DIAGNOSTICS DATASETS 

In this section, we presented Somers’ _D_ correlation of all metrics on all Diagnostics datasets. Table 18 summarizes the evaluations when investigated reference-free. One of the characteristics of our ROSCOE metrics is that, they can provide judgement of the model generated reasoning steps with and without the human reference reasoning chains. In the experiments section in §6, we discussed the results of our unsupervised scores in comparison to baseline scores when measured reference-free. In Table 19, we summarize the correlation analysis on ROSCOE metrics in comparison to baselines on diagnostic datasets when reference is present for evaluation. Specifically, each score is measured between the human provided reasoning steps ( _reference_ ) and the model generated reasoning steps ( _hypothesis_ ). We also display fine-grained meta-evaluations of all metrics on each diagnostics dataset in separate tables. Specifically, Tables 20, 26 for EQASC, Tables 21, 27 for EntailmentBank, Tables 22, 28 for MATH, Tables 23, 29 for ProofWriter, Tables 24, 30 for ASDIV, and Tables 25, 31 for AQUA. 

To understand if designed reference-free scores capture targeted error types we analyze perturbation-level correlations summarized in Fig. 6. Out of the all considered scores, _Info-Chain_ is able to cover 10 out of 12 of errors, except _Remove Step_ and _Semantic error_ perturbations. In general we can note that ROSCOE fails to consistently identify _missing step_ error type represented by _Remove Step_ perturbation across different datasets, while other synthesized error types are covered by at least one score type. 

Reference-based scores are covering all synthetic errors, with Semantic Coverage Chain showing strong correlations with all types of perturbations (Table 19). We also note that along with ROSCOE scores, the highest correlation among all reference-based scores belong to ROUGE and BERT scores (Tables 26-31). ROUGE scores consistently outperform on _Repetition_ , _Hallucination_ , _Remove Step_ , _Shuffle Steps_ , _Swap Steps_ , _Negate Step_ , and _Semantic_ perturbations, while under performing on _Random operation_ , and _Shuffle operations_ . We attribute this to the fact that ROUGE is an n-gram based score, so it is better in catching errors were wording has significantly changed, while failing to catch small changes within steps. 

It is worth noting that some scores, especially those among reference-based evaluations, get the highest possible Somers’ _D_ correlation scores of 1 _._ 0. What it means is that in some scenarios, there is a perfect correlation between the metric and the error type. In other words, for this metric we can find a threshold such generated chains that have scores greater than the threshold do not have errors of the given type, and in all generated chains with scores less than the threshold have that error. It is especially evident on referenced-based metrics that directly compare the reference solution and hypothesis. In this scenario, we build correlation for two groups: 1) non-perturbed hypothesis: the score is calculated by comparing embedding similarities of the reference with itself, and we expect to get high scores, 2) perturbed hypothesis: comparing reference with its perturbed version, where the scores should be lower. In some cases, we are able to perfectly separate perturbed and non-perturbed chains based on the corresponding metric values by selecting a threshold, in other cases we cannot due to a number of false-negatives (i.e., a chain gets a high score, although the error is present). As an example, consider the _Semantic Coverage-Chain_ metric calculated on EQASC dataset using _all-mpnet-base-v2_ sentence embeddings, and _Hallucination_ perturbation (Table 26). Here the Somers’ _D_ correlation score is 1 _._ 0. _Semantic Coverage-Chain_ is calculated as a normalized cosine distance between the chain embedding of the reference solution _**r**_ , and the chain embedding of the hypothesis _**h**_ : [1+cos( _**r** ,_ _**h**_ )] _/_ 2. Recall that in our setup, half of the hypothesis chains are perturbed reference chains, and another half is the same as the reference. While _Hallucination_ perturbation is an insertion of a random step from a dataset, it is hard to predict how if will affect the embedding of the chain as a whole, but on the unperturbed chains, where _**h**_ == _**r**_ , the _Semantic Coverage-Chain_ should be: [1 + cos( _**r** ,_ _**r**_ )] _/_ 2 = 1 _._ 0. Further review confirmed that in this dataset there are no false-positive instances, i.e., all chains with perturbations had _Semantic Coverage-Chain_ score less than 1 _._ 0. That means, we can always identify if the chain contains a _Hallucination_ 

28 

Published as a conference paper at ICLR 2023 

error or not, by comparing _Semantic Coverage-Chain_ value with 1 _._ 0 (threshold value), which is reflected in perfect Somers’ _D_ score. 

Highest correlations among reference-free scores belong to the _Repetition-*_ scores, that exhibit perfect correlation on EQASC dataset (Tables 20-25). For other datasets, non-perfect correlations can be attributed to the small number of false-negatives, i.e. they give low _Repetition-*_ scores for chains with non-duplicated but similar steps, while all chains with duplicates got almost 0 scores (Fig. 7). In EQASC explanations are created from a set of facts that are not directly related to each other, but are intended to give an answer when combined together. Among all datasets considered, these steps are most dissimilar, and thus can be separated with similarity-based scores. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0029-03.png)


Figure 6: Relative presence of the strong score-perturbation correlation, measured as the number of datasets where for each score-perturbation pair Somers’ _D_ correlation value is in the 90 _[th]_ percentile, normalized by the total number of datasets where this type of perturbation occurs. Statistics collected over ROSCOE referencefree scores with finetuned SimCSE embeddings. (Continued from _§_ 7) 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0029-05.png)


**----- Start of picture text -----**<br>
Score: Repetition Step, Perturbation: Repetition Score: Repetition Word, Perturbation: Repetition<br>(ProofWr., perturbed) (ProofWr., perturbed)<br>(ProofWr., original) (ProofWr., original)<br>(Math, perturbed) (Math, perturbed)<br>(Math, original) (Math, original)<br>(EntBank, perturbed) (EntBank, perturbed)<br>(EntBank, original) (EntBank, original)<br>(EQASC, perturbed) (EQASC, perturbed)<br>(EQASC, original) (EQASC, original)<br>(AQUA, perturbed) (AQUA, perturbed)<br>(AQUA, original) (AQUA, original)<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Repetition Step score perturbed Repetition Word score perturbed<br>original original<br>**----- End of picture text -----**<br>


Figure 7: Box-and-whisker plots of interquartile ranges of scores, for _Repetition_ perturbations and _Repetition-*_ scores. While all perturbed subsets have 0 or near 0 scores, all datasets except EQASC have some chains that were also scored as low despite the absence of duplicates. 

29 

Published as a conference paper at ICLR 2023 

Table 18: Somers’ _D_ correlation of **all** metrics on six **Diagnostics** datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlations that are not significant ( _p-value >_ = 0 _._ 05) are omitted when aggregating, and "-" denotes an absence of any significant correlations. Note that ASDIV is a 1-step equation dataset, so there are no repetition and self-consistency scores as there are no steps to compare. (Continued from _§_ 6, more details in App. H.1.). 

|**Metric**<br>**EntBank Math AQUA ProofWriter EQASC ASDIV**|**Metric**<br>**EntBank Math AQUA ProofWriter EQASC ASDIV**|
|---|---|
|ROUGE-1<br>0.410<br>0.176<br>0.257<br>0.095<br>0.342<br>0.305<br>ROUGE-2<br>0.391<br>0.151<br>0.206<br>0.090<br>0.217<br>-<br>ROUGE-L<br>0.365<br>0.156<br>0.264<br>0.106<br>0.315<br>0.269<br>BLEURT<br>0.257<br>0.148<br>0.252<br>0.024<br>0.447<br>-<br>BERTScore<br>0.380<br>0.124<br>0.220<br>0.117<br>0.462<br>0.322<br>BARTScore<br>0.358<br>0.185<br>0.317<br>0.081<br>0.415<br>-<br>BARTScore+<br>0.315<br>0.164<br>0.251<br>0.054<br>0.297<br>-<br>BARTScore-P<br>0.186<br>0.128<br>0.215<br>0.011<br>0.276<br>-<br>PRISM<br>0.453<br>0.208<br>0.191<br>0.235<br>0.436<br>-<br>CTC Relevancy<br>0.258<br>0.188<br>0.217<br>0.394<br>0.485<br>0.382<br>CTC Consistency<br>0.310<br>0.282<br>0.157<br>0.513<br>0.270<br>0.396||
|**ROSCOE Metrics (reference-free metrics only)**||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings<br>Faithfulness-Step<br>0.786<br>0.362<br>0.152<br>0.771<br>0.785<br>0.186<br>Faithfulness-Token<br>0.581<br>0.157<br>0.157<br>0.436<br>0.480<br>0.182<br>Info-Step<br>0.638<br>0.231<br>-<br>0.250<br>0.538<br>0.198<br>Repetition-Token<br>0.913<br>0.936<br>0.972<br>0.596<br>**1.000**<br>n/a||
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings<br>Info-Chain<br>0.419<br>0.467<br>0.214<br>0.082<br>0.550<br>0.280<br>Repetition-Step<br>0.909<br>0.932<br>0.982<br>0.631<br>**1.000**<br>n/a||
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings<br>Faithfulness-Step<br>0.777<br>0.229<br>0.224<br>0.732<br>0.630<br>0.266<br>Faithfulness-Token<br>0.663<br>0.200<br>-<br>0.517<br>0.502<br>0.515<br>Info-Step<br>0.560<br>0.131<br>0.183<br>0.226<br>0.399<br>0.275<br>Repetition-Token<br>0.919<br>0.939<br>0.971<br>0.606<br>**1.000**<br>n/a||
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings<br>Info-Chain<br>0.524<br>0.180<br>0.195<br>0.045<br>0.409<br>0.289<br>Repetition-Step<br>0.901<br>**0.949**<br>**0.991**<br>0.621<br>**1.000**<br>n/a||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta_sentence embeddings<br>Faithfulness-Step<br>0.538<br>0.614<br>0.826<br>0.763<br>0.907<br>**0.879**<br>Faithfulness-Token<br>0.519<br>0.204<br>0.285<br>0.499<br>0.492<br>0.740<br>Info-Step<br>0.599<br>0.511<br>0.703<br>0.317<br>0.804<br>**0.879**<br>Repetition-Token<br>0.919<br>0.939<br>0.971<br>0.606<br>**1.000**<br>n/a||
|ROSCOE-SSwith fnetuned_sup-simcse-roberta_sentence embeddings<br>Info-Chain<br>**0.955**<br>0.777<br>0.933<br>0.462<br>0.995<br>0.857<br>Repetition-Step<br>0.908<br>0.924<br>0.982<br>0.624<br>**1.000**<br>n/a||
|ROSCOE-LI<br>Self-Consistency<br>0.782<br>0.190<br>0.368<br>0.204<br>Source-Consistency<br>0.917<br>0.341<br>0.424<br>0.289|0.793<br>n/a<br>0.778<br>0.771|
|ROSCOE-LC<br>Perplexity-Step<br>0.213<br>0.160<br>0.110<br>0.178<br>Perplexity-Chain<br>0.151<br>0.175<br>0.229<br>0.135<br>Grammar<br>0.604<br>0.392<br>0.359<br>**0.788**|0.394<br>0.485<br>0.379<br>0.485<br>0.859<br>0.470|



30 

Published as a conference paper at ICLR 2023 

Table 19: Somers’ _D_ correlation of all **reference-based** metrics on six **Diagnostics** datasets. Metrics are measured using **reference** generations on ( _**r** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. (Continued from _§_ 6, more details in App. H.1.) 

|**Metric**|**EntBank **|**Math AQUA **|**Math AQUA **|**ProofWriter **|**EQASC **|**ASDIV**|
|---|---|---|---|---|---|---|
|ROUGE-1|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|ROUGE-2|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|ROUGE-L|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|BLEURT|0.821|0.773|0.946|0.829|**1.000**|0.93|
|BERTScore|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|BARTScore|0.983|0.989|0.984|0.986|**1.000**|0.954|
|BARTScore+|0.988|0.963|**1.000**|0.996|**1.000**|**1.000**|
|BARTScore-P|0.877|0.799|0.905|0.595|0.966|0.83|
|PRISM|0.939|0.521|**1.000**|0.997|0.996|**1.000**|
|CTC Relevancy|0.457|0.592|0.409|0.725|0.954|0.398|
|CTC Consistency|0.814|0.804|0.833|0.635|0.974|0.6|
|**ROSCOE Metrics (reference-based metrics only)**|||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||||
|Hallucination|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Redundancy|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Semantic Coverage-Step|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Reasoning Alignment|**1.000**|**1.000**|0.143|**1.000**|**1.000**|**1.000**|
|Commonsense|0.438|**1.000**|**1.000**|0.379|**1.000**|**1.000**|
|Missing Step|0.993|**1.000**|**1.000**|0.876|**1.000**|**1.000**|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings|||||||
|Semantic Coverage-Chain|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||||
|Hallucination|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Redundancy|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Semantic Coverage-Step|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Reasoning Alignment|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Commonsense|0.433|**1.000**|**1.000**|0.415|**1.000**|**1.000**|
|Missing Step|0.999|**1.000**|**1.000**|0.874|**1.000**|**1.000**|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||||
|Semantic Coverage-Chain|**1.000**|0.999|**1.000**|**1.000**|**1.000**|**1.000**|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta_sentence embeddings|||||||
|Hallucination|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Redundancy|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Semantic Coverage-Step|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Reasoning Alignment|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|
|Commonsense|0.445|**1.000**|**1.000**|0.404|**1.000**|**1.000**|
|Missing Step|0.999|**1.000**|**1.000**|0.873|**1.000**|**1.000**|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta_sentence embeddings|||||||
|Semantic Coverage-Chain|**1.000**|0.999|**1.000**|**1.000**|**1.000**|**1.000**|



31 

Published as a conference paper at ICLR 2023 

Table 20: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **EQASC** Diagnostics datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_<br>**Repet.**<br>**Halluc. Grammar Remove**<br>**Negate**<br>**Semantic**<br>**Error**<br>**Step**<br>**Step**<br>**Error**|**Perturbations**_→_<br>**Repet.**<br>**Halluc. Grammar Remove**<br>**Negate**<br>**Semantic**<br>**Error**<br>**Step**<br>**Step**<br>**Error**|
|---|---|
|Rouge-1<br>0.264_†_<br>0.342_†_<br>0.017<br>0.227_†_<br>0.063<br>-0.023<br>Rouge-2<br>0.071<br>0.205_†_<br>0.106_†_<br>0.217_†_<br>0.099_†_<br>-0.002<br>Rouge-L<br>0.210_†_<br>0.315_†_<br>0.057<br>0.179_†_<br>0.083_†_<br>-0.016<br>BLEURT<br>0.366_†_<br>0.447_†_<br>-0.028<br>0.195_†_<br>0.204_†_<br>-0.108_†_<br>BERTScore<br>0.288_†_<br>0.462_†_<br>0.153_†_<br>0.160_†_<br>0.052<br>0.051<br>BARTScore<br>-0.127_†_<br>0.038<br>0.047<br>0.415<br>_†_<br>0.019<br>-0.072<br>BARTScore+<br>0.028<br>0.212_†_<br>0.055<br>0.297_†_<br>0.023<br>-0.028<br>BARTScore-P<br>-0.039<br>0.031<br>-0.038<br>0.276_†_<br>-0.003<br>-0.023<br>PRISM<br>-0.327_†_<br>0.436_†_<br>0.267_†_<br>0.077_†_<br>0.010<br>0.123_†_<br>CTC-Relevancy<br>0.141_†_<br>0.001<br>0.082_†_<br>**0.485**_†_<br>0.002<br>0.220_†_<br>CTC-Consistency<br>0.001<br>-0.080_†_<br>0.095_†_<br>-0.154_†_<br>0.078_†_<br>0.270_†_||
|**ROSCOE Metrics(reference-free on** (**_s_**_,_**_h_**)**)**||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings<br>Faithfulness-Step<br>-0.006<br>0.785_†_<br>0.017<br>-0.040<br>0.084_†_<br>-0.243_†_<br>Faithfulness-Token<br>-0.031<br>0.480_†_<br>-0.001<br>0.037<br>0.008<br>-0.156_†_<br>Info-Step<br>0.006<br>0.538_†_<br>-0.003<br>0.223_†_<br>0.085_†_<br>-0.191_†_<br>Repetition-Token<br>**1.000**_†_<br>0.399_†_<br>-0.028<br>-1.000_†_<br>0.070<br>0.074||
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings<br>Info-Chain<br>0.189_†_<br>0.550_†_<br>-0.022<br>0.220_†_<br>0.059<br>-0.132_†_<br>Repetition-Step<br>**1.000**_†_<br>0.035<br>-0.034<br>-1.000_†_<br>-0.120_†_<br>0.030||
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings<br>Faithfulness-Step<br>-0.061<br>0.630_†_<br>-0.010<br>0.007<br>0.308_†_<br>-0.204_†_<br>Faithfulness-Token<br>-0.031<br>0.502_†_<br>0.032<br>0.045<br>0.107_†_<br>-0.168_†_<br>Info-Step<br>-0.064<br>0.399_†_<br>-0.024<br>0.241_†_<br>0.296_†_<br>-0.155_†_<br>Repetition-Token<br>**1.000**_†_<br>0.148_†_<br>-0.138_†_<br>-1.000_†_<br>-0.055<br>-0.080_†_||
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings<br>Info-Chain<br>-0.025<br>0.409_†_<br>-0.028<br>0.256_†_<br>0.379_†_<br>-0.163_†_<br>Repetition-Step<br>**1.000**_†_<br>0.001<br>-0.003<br>-1.000_†_<br>-0.465_†_<br>0.071||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings<br>Faithfulness-Step<br>-0.044<br>0.630_†_<br>0.293_†_<br>-0.046<br>0.907<br>_†_<br>-0.118_†_<br>Faithfulness-Token<br>-0.019<br>0.485_†_<br>0.141_†_<br>0.036<br>0.492_†_<br>-0.128_†_<br>Info-Step<br>-0.041<br>0.383_†_<br>0.266_†_<br>0.196_†_<br>0.804_†_<br>-0.068<br>Repetition-Token<br>**1.000**_†_<br>0.148_†_<br>-0.138_†_<br>-1.000_†_<br>-0.055<br>-0.080_†_||
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings<br>Info-Chain<br>0.995<br>_†_<br>0.871<br>_†_<br>0.588<br>_†_<br>0.103_†_<br>**0.967**_†_<br>0.121_†_<br>Repetition-Step<br>**1.000**_†_<br>0.037<br>-0.386_†_<br>-1.000_†_<br>-0.953_†_<br>-0.048||
|ROSCOE-LI<br>Source-Consistency<br>Self-Consistency|-0.020<br>0.576_†_<br>0.112_†_<br>-0.239_†_<br>0.778_†_<br>0.268_†_<br>0.022<br>0.633_†_<br>0.399_†_<br>-0.713_†_<br>0.793_†_<br>0.476<br>_†_|
|ROSCOE-LC<br>Perplexity-Chain<br>Perplexity-Step<br>Grammar|-0.690_†_<br>-0.007<br>0.379_†_<br>0.260_†_<br>0.118_†_<br>0.269_†_<br>0.937_†_<br>**0.965**_†_<br>0.352_†_<br>-0.953_†_<br>0.081_†_<br>0.225_†_<br>-0.025<br>0.060<br>**0.859**_†_<br>-0.145_†_<br>0.139_†_<br>**0.722**_†_|



32 

Published as a conference paper at ICLR 2023 

Table 21: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **Entailment Bank** Diagnostics datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Repet.**|**Halluc. Grammar **<br>**Error**|**Halluc. Grammar **<br>**Error**|**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate**<br>**Step**|**Semantic**<br>**Error**|
|---|---|---|---|---|---|---|---|---|
|Rouge-1|0.391_†_|0.410_†_|0.074|0.081|0.113|0.065|0.034|-0.017|
|Rouge-2|0.356_†_|0.391_†_|0.160_†_|0.116|0.109|0.088|0.109|0.091|
|Rouge-L|0.209_†_|0.194_†_|0.112|0.138_†_|0.365<br>_†_|0.229<br>_†_|0.003|0.063|
|BLEURT|0.025|0.164_†_|0.060|0.231_†_|0.092|0.096|0.257_†_|-0.036|
|BERTScore|0.264_†_|0.380_†_|0.211_†_|0.150_†_|0.364_†_|0.205_†_|0.050|0.087|
|BARTScore|0.034|0.063|0.106|0.358<br>_†_|0.248_†_|0.183_†_|0.142_†_|0.164_†_|
|BARTScore+|0.101|0.047|0.036|0.315_†_|0.184_†_|0.155_†_|0.130_†_|0.173_†_|
|BARTScore-P|0.061|0.012|0.025|0.186_†_|0.041|-0.008|0.098|0.100|
|PRISM|0.230_†_|0.453_†_|0.279_†_|0.118|0.331_†_|0.167_†_|0.081|0.148_†_|
|CTC-Relevancy|0.258_†_|0.057|-0.026|0.080|-0.020|0.035|0.029|0.159_†_|
|CTC-Consistency|0.310_†_|0.159_†_|-0.058|-0.249_†_|-0.162_†_|-0.023|-0.046|0.194_†_|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**|||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||||||
|Faithfulness-Step|0.023|0.786<br>_†_|0.231_†_|-0.005|0.111|-0.000|0.098|0.121|
|Faithfulness-Token|0.098|0.581_†_|0.250_†_|0.087|0.009|-0.020|0.179_†_|0.170_†_|
|Info-Step|0.083|0.638_†_|0.181_†_|0.161_†_|0.025|-0.001|0.216_†_|0.139_†_|
|Repetition-Token|0.913<br>_†_|0.105|0.042|-0.177_†_|0.139_†_|0.038|-0.085|-0.058|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||||||
|Faithfulness-Step|0.023|0.786<br>_†_|0.231_†_|-0.005|0.111|-0.000|0.098|0.121|
|Faithfulness-Token|0.098|0.581_†_|0.250_†_|0.087|0.009|-0.020|0.179_†_|0.170_†_|
|Info-Step|0.083|0.638_†_|0.181_†_|0.161_†_|0.025|-0.001|0.216_†_|0.139_†_|
|Repetition-Token|0.913<br>_†_|0.105|0.042|-0.177_†_|0.139_†_|0.038|-0.085|-0.058|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings|||||||||
|Info-Chain|0.039|0.419_†_|0.083|0.071|0.068|0.025|0.040|0.037|
|Repetition-Step|0.909_†_|0.148_†_|-0.061|-0.165_†_|0.067|-0.068|-0.062|-0.056|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||||||
|Faithfulness-Step|-0.002|0.777_†_|0.158_†_|-0.015|0.107|0.009|0.358_†_|0.110|
|Faithfulness-Token|0.063|0.663_†_|0.280_†_|0.084|0.046|0.024|0.279_†_|0.159_†_|
|Info-Step|0.066|0.560_†_|0.079|0.148_†_|-0.035|0.007|0.450_†_|0.117|
|Repetition-Token|**0.919**_†_|0.137_†_|-0.010|-0.216_†_|0.171_†_|0.005|-0.106|-0.099|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||||||
|Info-Chain|0.084|0.515_†_|0.005|0.119|-0.019|0.023|0.524_†_|0.048|
|Repetition-Step|0.901_†_|0.139_†_|0.006|-0.188_†_|0.019|-0.066|-0.144_†_|-0.098|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||
|Faithfulness-Step|0.105|0.392_†_|0.204_†_|0.063|0.093|0.029|0.538_†_|-0.115|
|Faithfulness-Token|0.079|0.519_†_|0.271_†_|0.100|0.055|0.029|0.356_†_|0.041|
|Info-Step|0.131_†_|0.364_†_|0.227_†_|0.161_†_|0.053|0.047|0.599_†_|-0.018|
|Repetition-Token|**0.919**_†_|0.137_†_|-0.010|-0.216_†_|0.171_†_|0.005|-0.106|-0.099|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||
|Info-Chain|0.871_†_|**0.851**_†_|**0.752**_†_|**0.437**_†_|**0.937**_†_|**0.672**_†_|**0.955**_†_|0.300<br>_†_|
|Repetition-Step|0.908_†_|0.133_†_|-0.013|-0.169_†_|0.135_†_|-0.043|-0.058|-0.079|
|ROSCOE-LI|||||||||
|Source-Consistency|-0.044|0.289_†_|0.218_†_|0.052|0.150_†_|-0.075|0.860_†_|0.257_†_|
|Self-Consistency|-0.040|0.403_†_|0.216_†_|-0.042|0.129_†_|-0.053|0.782_†_|0.170_†_|
|ROSCOE-LC|||||||||
|Perplexity-Chain|-0.364_†_|0.104|0.116|0.151_†_|-0.006|0.085|0.060|0.133_†_|
|Perplexity-Step|0.199_†_|0.168_†_|0.075|-0.097|0.143_†_|-0.021|-0.019|-0.046|
|Grammar|-0.109|0.076|0.604<br>_†_|0.075|0.044|-0.063|0.033|**0.365**_†_|



33 

Published as a conference paper at ICLR 2023 

Table 22: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **MATH** Diagnostics datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|App.H.1.)||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Perturbations**_→_|**Repet.**|**Halluc. Grammar **<br>**Error**||**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate**<br>**Step**|**Random**<br>**Number**|**Random**<br>**Operation **|**Shuffe**<br> **Numbers **|**Shuffe**<br> **Operations**|
|Rouge-1|0.176_†_|0.151_†_|-0.004|0.065_†_|-0.020|0.020|0.028|0.017|0.008|0.022|0.011|
|Rouge-2|0.107_†_|0.095_†_|0.006|0.121_†_|-0.005|0.029|0.010|0.018|-0.011|0.151_†_|-0.002|
|Rouge-L|0.126_†_|0.156_†_|0.004|0.075_†_|0.008|0.042_†_|0.023|0.018|0.008|0.082_†_|0.004|
|BLEURT|0.143_†_|0.148_†_|0.000|-0.023|0.001|0.012|0.049_†_|-0.000|-0.005|-0.036_†_|0.002|
|BERTScore|0.124_†_|0.117_†_|0.067_†_|0.089_†_|0.025|0.029|0.010|0.029|0.016|0.034_†_|0.017|
|BARTScore|-0.048_†_|-0.066_†_|0.015|0.185<br>_†_|-0.026|-0.002|-0.030|0.029|0.009|0.075_†_|0.045_†_|
|BARTScore+|0.015|-0.003|0.049_†_|0.162_†_|0.003|0.031|0.002|0.050_†_|0.047_†_|0.164_†_|0.063_†_|
|BARTScore-P|-0.005|0.002|0.022|0.128_†_|0.006|0.011|-0.009|0.024|0.002|0.115_†_|0.059_†_|
|PRISM|-0.115_†_|0.208_†_|0.120_†_|0.095_†_|0.029|0.017|-0.003|0.102_†_|0.069_†_|0.117_†_|0.111_†_|
|CTC-Relevancy|0.104_†_|0.041_†_|0.133_†_|**0.188**_†_|0.027|0.029|0.018|0.043_†_|0.052_†_|0.052_†_|0.001|
|CTC-Consistency|-0.106_†_|0.145_†_|0.282_†_|0.105_†_|0.096_†_|0.035_†_|0.080_†_|0.079_†_|0.033|0.046_†_|-0.017|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Faithfulness-Step|0.029|0.362_†_|0.016|0.070_†_|-0.024|0.009|0.025|0.060_†_|0.048_†_|0.074_†_|0.060_†_|
|Faithfulness-Token|0.000|0.157_†_|0.000|0.004|-0.004|-0.003|0.018|0.028|0.005|-0.022|0.003|
|Info-Step|0.024|0.231_†_|0.022|0.118_†_|-0.018|0.009|0.033_†_|0.085_†_|0.074_†_|0.109_†_|0.106_†_|
|Repetition-Token|0.936_†_|0.069_†_|-0.018|-0.078_†_|0.010|0.043_†_|0.006|-0.041_†_|-0.018|0.035_†_|0.006|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Info-Chain|0.059_†_|0.467<br>_†_|0.016|0.172_†_|0.106<br>_†_|0.091<br>_†_|0.063_†_|0.192_†_|0.089_†_|0.162_†_|0.135_†_|
|Repetition-Step|0.932_†_|-0.002|-0.036_†_|-0.114_†_|-0.001|0.013|-0.026|-0.134_†_|-0.045_†_|-0.008|-0.030|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|-0.004|0.229_†_|0.049_†_|0.046_†_|0.007|0.010|0.142_†_|0.073_†_|0.033|0.031|-0.010|
|Faithfulness-Token|0.014|0.200_†_|0.091_†_|-0.004|-0.001|0.009|0.021|0.033_†_|0.005|0.021|0.008|
|Info-Step|-0.020|0.115_†_|0.048_†_|0.092_†_|0.009|0.020|0.131_†_|0.086_†_|0.029|0.042_†_|-0.015|
|Repetition-Token|0.939<br>_†_|0.007|-0.093_†_|-0.073_†_|0.023|0.037_†_|0.002|-0.058_†_|-0.035_†_|0.019|-0.002|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.028|0.114_†_|0.038_†_|0.071_†_|0.034_†_|0.004|0.180_†_|0.078_†_|0.051_†_|0.041_†_|-0.007|
|Repetition-Step|**0.949**_†_|0.019|-0.043_†_|-0.094_†_|-0.012|0.000|-0.062_†_|-0.119_†_|0.014|0.027|-0.001|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|0.021|0.223_†_|0.190_†_|0.038_†_|-0.031|0.005|0.614<br>_†_|0.376_†_|0.415_†_|0.277_†_|0.428_†_|
|Faithfulness-Token|0.016|0.204_†_|0.106_†_|-0.005|-0.004|0.008|0.084_†_|0.067_†_|0.045_†_|0.047_†_|0.052_†_|
|Info-Step|0.012|0.133_†_|0.228_†_|0.099_†_|-0.021|0.020|0.511_†_|0.451<br>_†_|0.452<br>_†_|0.301_†_|0.430<br>_†_|
|Repetition-Token|0.939<br>_†_|0.007|-0.093_†_|-0.073_†_|0.023|0.037_†_|0.002|-0.058_†_|-0.035_†_|0.019|-0.002|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.679_†_|**0.588**_†_|**0.694**_†_|**0.216**_†_|**0.746**_†_|**0.530**_†_|**0.777**_†_|**0.698**_†_|**0.757**_†_|**0.524**_†_|**0.662**_†_|
|Repetition-Step|0.924_†_|0.024|-0.059_†_|-0.083_†_|-0.019|0.012|-0.052_†_|-0.192_†_|-0.103_†_|-0.011|0.017|
|ROSCOE-LI||||||||||||
|Source-Consistency|0.011|0.071_†_|0.044_†_|-0.041_†_|-0.002|0.026|0.215_†_|0.223_†_|0.123_†_|0.341<br>_†_|0.133_†_|
|Self-Consistency|0.015|0.069_†_|-0.003|-0.105_†_|-0.014|0.011|0.122_†_|0.147_†_|0.068_†_|0.190_†_|0.088_†_|
|ROSCOE-LC||||||||||||
|Perplexity-Chain|-0.358_†_|0.020|0.175_†_|0.103_†_|0.100_†_|0.035_†_|0.003|0.173_†_|0.109_†_|0.154_†_|0.170_†_|
|Perplexity-Step|0.160_†_|0.127_†_|0.001|-0.093_†_|-0.016|0.019|0.006|0.003|0.012|0.048_†_|0.037|
|Grammar|0.010|0.026|0.392<br>_†_|-0.020|-0.012|0.005|0.112_†_|0.034_†_|0.026|0.057_†_|0.040|



34 

Published as a conference paper at ICLR 2023 

Table 23: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **ProofWriter** Diagnostics datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Repet.**|**Halluc. Grammar **<br>**Error**|**Halluc. Grammar **<br>**Error**|**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate**<br>**Step**|**Semantic**<br>**Error**|
|---|---|---|---|---|---|---|---|---|
|Rouge-1|-0.124_†_|-0.054_†_|0.023_†_|0.095_†_|0.013|-0.001|0.007|0.006|
|Rouge-2|-0.093_†_|-0.002|0.068_†_|0.090_†_|0.018|-0.004|0.037_†_|0.047_†_|
|Rouge-L|-0.120_†_|-0.028_†_|0.029_†_|0.089_†_|0.106_†_|0.020|0.020|0.027_†_|
|BLEURT|-0.058_†_|-0.001|-0.027_†_|-0.099_†_|0.016|0.007|0.024_†_|-0.016|
|BERTScore|-0.049_†_|0.077_†_|0.117_†_|0.082_†_|-0.064_†_|-0.054_†_|0.023_†_|0.108_†_|
|BARTScore|-0.059_†_|-0.096_†_|-0.037_†_|0.081_†_|-0.013|-0.010|-0.015|-0.061_†_|
|BARTScore+|-0.055_†_|-0.067_†_|0.014|0.054_†_|-0.011|-0.021|0.006|0.032_†_|
|BARTScore-P|-0.046_†_|-0.049_†_|-0.012|0.010|-0.044_†_|-0.032_†_|-0.020|0.011|
|PRISM|-0.159_†_|0.159_†_|0.222_†_|0.097_†_|0.017|-0.010|0.060_†_|0.235_†_|
|CTC-Relevancy|0.394_†_|0.392_†_|0.123_†_|0.185<br>_†_|-0.131_†_|-0.052_†_|0.036_†_|0.077_†_|
|CTC-Consistency|0.496_†_|0.513_†_|0.182_†_|**0.223**_†_|-0.063_†_|-0.022|0.098_†_|0.131_†_|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**|||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||||||
|Faithfulness-Step|0.002|**0.771**_†_|0.348_†_|0.165_†_|0.011|0.013|0.233_†_|0.515<br>_†_|
|Faithfulness-Token|0.004|0.436_†_|0.264_†_|0.055_†_|0.030_†_|0.006|0.168_†_|0.310_†_|
|Info-Step|-0.004|0.250_†_|0.121_†_|-0.062_†_|0.023|0.015|0.108_†_|0.174_†_|
|Repetition-Token|0.596_†_|0.053_†_|-0.041_†_|0.101_†_|-0.006|-0.003|-0.055_†_|-0.050_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings|||||||||
|Info-Chain|-0.052_†_|0.083_†_|0.033_†_|-0.089_†_|0.014|0.005|-0.001|0.001|
|Repetition-Step|**0.631**_†_|0.031_†_|-0.027_†_|0.116_†_|-0.002|0.002|-0.044_†_|-0.042_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||||||
|Faithfulness-Step|0.004|0.732_†_|0.171_†_|0.154_†_|0.013|0.012|0.481<br>_†_|0.494_†_|
|Faithfulness-Token|0.010|0.517_†_|0.334_†_|0.086_†_|0.031_†_|0.005|0.336_†_|0.395_†_|
|Info-Step|-0.008|0.226_†_|0.047_†_|-0.063_†_|0.027_†_|0.014|0.172_†_|0.160_†_|
|Repetition-Token|0.606_†_|0.036_†_|-0.065_†_|0.097_†_|-0.002|0.009|-0.070_†_|-0.070_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||||||
|Info-Chain|-0.040_†_|0.004|-0.015|-0.140_†_|-0.024_†_|-0.011|0.045_†_|-0.006|
|Repetition-Step|0.621_†_|0.028_†_|-0.008|0.115_†_|-0.011|0.001|-0.050_†_|-0.043_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||
|Faithfulness-Step|0.008|0.763<br>_†_|0.618<br>_†_|0.180_†_|0.008|0.004|**0.726**_†_|**0.536**_†_|
|Faithfulness-Token|0.012|0.499_†_|0.475_†_|0.088_†_|0.036_†_|0.000|0.436_†_|0.403_†_|
|Info-Step|-0.018|0.243_†_|0.252_†_|-0.046_†_|0.016|0.012|0.317_†_|0.187_†_|
|Repetition-Token|0.606_†_|0.036_†_|-0.065_†_|0.097_†_|-0.002|0.009|-0.070_†_|-0.070_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||
|Info-Chain|0.214_†_|0.248_†_|0.284_†_|0.077_†_|**0.330**_†_|**0.246**_†_|0.462_†_|0.122_†_|
|Repetition-Step|0.624<br>_†_|0.039_†_|-0.034_†_|0.116_†_|-0.007|-0.002|-0.076_†_|-0.034_†_|
|ROSCOE-LI|||||||||
|Source-Consistency|-0.008|0.027_†_|0.028_†_|0.010|-0.044_†_|-0.004|0.289_†_|-0.049_†_|
|Self-Consistency|0.011|0.204_†_|0.084_†_|0.110_†_|-0.022|-0.020|0.036_†_|0.065_†_|
|ROSCOE-LC|||||||||
|Perplexity-Chain|-0.165_†_|0.047_†_|0.112_†_|-0.064_†_|0.135<br>_†_|0.067<br>_†_|0.012|0.128_†_|
|Perplexity-Step|0.178_†_|0.112_†_|0.033_†_|0.082_†_|-0.008|0.005|-0.036_†_|-0.008|
|Grammar|0.000|0.042_†_|**0.788**_†_|0.102_†_|0.007|-0.023|0.007|0.515<br>_†_|



35 

Published as a conference paper at ICLR 2023 

Table 24: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **ASDIV** Diagnostics datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Random**<br>**Number**|**Random**<br>**Operation **|**Shuffe**<br> **Numbers**|**Shuffe**<br>**Operations**|
|---|---|---|---|---|
|Rouge-1|0.305_†_|0.096|0.085|-0.220|
|Rouge-2|-0.034|-0.014|0.007|-0.038|
|Rouge-L|0.245_†_|0.073|0.269_†_|-0.235|
|BLEURT|0.043|-0.034|0.059|0.015|
|BERTScore|0.098|0.322_†_|0.025|0.167|
|BARTScore|0.107|-0.027|-0.105|0.197|
|BARTScore+|-0.011|-0.002|-0.075|-0.015|
|BARTScore-P|0.068|0.043|-0.048|0.121|
|PRISM|-0.009|-0.035|-0.114|0.258|
|CTC-Relevancy|0.382_†_|0.155_†_|0.038|0.000|
|CTC-Consistency|0.396_†_|0.189_†_|0.121|-0.121|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**|||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence|||embeddings||
|Faithfulness-Step|0.186_†_|-0.090|0.091|0.091|
|Faithfulness-Token|0.182_†_|0.080|0.062|-0.091|
|Info-Step|0.198_†_|-0.091|0.085|0.167|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence|||embeddings||
|Info-Chain|0.280_†_|0.005|0.192_†_|0.091|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||
|Faithfulness-Step|0.266_†_|0.082|-0.182_†_|0.015|
|Faithfulness-Token|0.273_†_|0.011|-0.125|0.515_†_|
|Info-Step|0.275_†_|0.125|-0.141_†_|0.000|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||
|Info-Chain|0.289_†_|0.145_†_|-0.084|0.030|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||
|Faithfulness-Step|0.630_†_|0.840_†_|0.670_†_|**0.879**_†_|
|Faithfulness-Token|0.576_†_|0.740_†_|0.552_†_|0.545_†_|
|Info-Step|0.669_†_|0.844<br>_†_|0.683_†_|**0.879**_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||
|Info-Chain|**0.773**_†_|**0.857**_†_|**0.795**_†_|0.803<br>_†_|
|ROSCOE-LI|||||
|Source-Consistency|0.760<br>_†_|0.771_†_|0.763<br>_†_|0.500_†_|
|Self-Consistency|0.203_†_|0.227_†_|0.206_†_|0.152|
|ROSCOE-LC|||||
|Perplexity-Chain|0.300_†_|0.092|0.214_†_|0.485_†_|
|Perplexity-Step|0.300_†_|0.092|0.214_†_|0.485_†_|
|Grammar|0.170_†_|-0.083|-0.007|0.470_†_|



36 

Published as a conference paper at ICLR 2023 

Table 25: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **AQUA** Diagnostics datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|App.H.1.)||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Perturbations**_→_|**Repet.**|**Hallu.**|**Grammar **<br>**Error**|**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate **<br>**Step**|**Random**<br>**Number**|**Random**<br>**Operation **|**Shuffe**<br> **Numbers **|**Shuffe**<br> **Operations**|
|Rouge-1|0.134|0.055|-0.085|0.257_†_|0.074|-0.032|0.109|0.056|-0.006|0.131|-0.073|
|Rouge-2|0.180_†_|0.004|-0.021|0.206_†_|0.100|-0.092|0.044|0.051|0.082|0.196_†_|-0.061|
|Rouge-L|0.072|0.055|-0.059|0.264<br>_†_|0.145<br>_†_|-0.008|0.120|0.050|0.004|0.148_†_|-0.037|
|BLEURT|0.010|0.047|-0.066|0.252_†_|-0.003|-0.047|0.071|-0.010|0.020|0.165_†_|-0.023|
|BERTScore|0.134|0.111|-0.016|0.166_†_|-0.011|-0.043|0.028|0.007|0.079|0.220_†_|-0.036|
|BARTScore|-0.030|-0.269_†_|-0.025|**0.317**_†_|0.102|0.051|-0.014|0.060|0.011|0.169_†_|0.036|
|BARTScore+|0.112|-0.037|0.044|0.251_†_|0.019|-0.053|0.028|0.072|0.066|0.230_†_|-0.009|
|BARTScore-P|0.082|-0.097|-0.007|0.215_†_|0.013|-0.027|0.114|0.059|0.042|0.182_†_|0.028|
|PRISM|-0.119|0.026|0.177_†_|0.188_†_|0.071|0.048|0.057|0.191_†_|0.131|0.182_†_|0.028|
|CTC-Relevancy|0.133|-0.016|0.065|0.096|-0.068|0.013|0.024|0.119|-0.017|0.217_†_|0.072|
|CTC-Consistency|0.071|-0.050|0.075|0.041|-0.074|0.012|0.004|0.106|-0.006|0.157_†_|0.024|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Faithfulness-Step|0.098|0.152_†_|-0.013|0.106|0.035|-0.026|0.080|0.057|-0.023|0.026|-0.123|
|Faithfulness-Token|0.125|0.009|0.008|0.157_†_|0.047|-0.001|0.057|-0.014|0.076|0.070|-0.006|
|Info-Step|0.088|0.030|-0.020|0.114|0.068|-0.007|0.015|0.063|0.025|0.068|-0.088|
|Repetition-Token|0.972_†_|0.216_†_|-0.125|-0.226_†_|-0.053|-0.061|-0.041|-0.004|-0.024|0.037|-0.001|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Info-Chain|0.066|0.214_†_|-0.053|0.146_†_|0.023|-0.035|0.049|0.079|0.038|0.153_†_|-0.081|
|Repetition-Step|0.982<br>_†_|0.143_†_|-0.093|-0.169_†_|0.006|0.002|-0.031|-0.068|-0.028|0.010|0.050|
|ROSCOE-SAwith_sup-simcse-roberta-base_|||sentence embeddings|||||||||
|Faithfulness-Step|0.089|0.179_†_|-0.043|0.135|-0.016|-0.009|0.224_†_|0.122|0.096|0.014|-0.156|
|Faithfulness-Token|0.080|0.062|-0.047|0.117|0.036|-0.011|0.057|0.035|0.024|0.049|-0.059|
|Info-Step|0.099|0.064|-0.012|0.162_†_|0.018|-0.023|0.183_†_|0.138|0.116|0.099|-0.147|
|Repetition-Token|0.971_†_|0.072|-0.112|-0.190_†_|-0.062|-0.054|-0.102|-0.034|0.061|0.026|0.079|
|ROSCOE-SSwith_sup-simcse-roberta-base_|||sentence embeddings|||||||||
|Info-Chain|0.130|0.046|-0.025|0.178_†_|-0.013|-0.024|0.153_†_|0.108|0.066|0.195_†_|-0.119|
|Repetition-Step|**0.991**_†_|0.125|-0.120|-0.234_†_|0.007|-0.023|-0.063|-0.155_†_|-0.025|-0.010|0.076|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|0.169_†_|0.165_†_|0.280_†_|0.000|0.018|-0.113|0.826<br>_†_|0.394_†_|0.329<br>_†_|0.210_†_|0.236_†_|
|Faithfulness-Token|0.063|0.132|0.014|0.084|0.030|-0.045|0.285_†_|0.079|0.077|0.082|-0.037|
|Info-Step|0.150_†_|0.038|0.203_†_|0.107|0.041|-0.081|0.703_†_|0.450<br>_†_|0.327_†_|0.313_†_|0.259<br>_†_|
|Repetition-Token|0.971_†_|0.072|-0.112|-0.190_†_|-0.062|-0.054|-0.102|-0.034|0.061|0.026|0.079|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.861_†_|**0.465**_†_|**0.399**_†_|0.025|**0.463**_†_|**0.315**_†_|**0.933**_†_|**0.589**_†_|**0.499**_†_|0.395_†_|**0.337**_†_|
|Repetition-Step|0.982<br>_†_|0.141_†_|-0.171_†_|-0.149_†_|-0.094|0.022|-0.143|-0.339_†_|-0.155_†_|-0.016|-0.028|
|ROSCOE-LI||||||||||||
|Source-Consistency|-0.044|0.140|-0.061|-0.171_†_|-0.087|0.024|0.316_†_|0.212_†_|0.151_†_|0.419<br>_†_|0.096|
|Self-Consistency|0.041|0.368<br>_†_|-0.028|-0.227_†_|0.019|-0.040|0.104|0.226_†_|0.106|0.167_†_|0.093|
|ROSCOE-LC||||||||||||
|Perplexity-Chain|-0.288_†_|-0.120|0.179_†_|0.121|0.021|0.101|0.134|0.190_†_|0.071|0.194_†_|0.229_†_|
|Perplexity-Step|0.171_†_|0.182_†_|-0.051|-0.105|-0.013|-0.097|-0.068|0.060|0.011|-0.004|0.079|
|Grammar|0.091|0.200_†_|0.359<br>_†_|0.020|-0.043|-0.008|0.223_†_|0.094|0.046|0.086|0.171|



37 

Published as a conference paper at ICLR 2023 

Table 26: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **EQASC** Diagnostics datasets. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Repet.**|**Hallu.**|**Grammar **<br>**Error**|**Remove **<br>**Step**|**Negate **<br>**Step**|**Semantic**<br>**Error**|
|---|---|---|---|---|---|---|
|Rouge-1|**1.000**_†_|**1.000**_†_|0.622_†_|**1.000**_†_|**1.000**_†_|0.946_†_|
|Rouge-2|**1.000**_†_|**1.000**_†_|0.998_†_|**1.000**_†_|**1.000**_†_|0.946_†_|
|Rouge-L|**1.000**_†_|**1.000**_†_|0.998_†_|**1.000**_†_|**1.000**_†_|0.946_†_|
|BLEURT|0.829_†_|0.999<br>_†_|0.512_†_|**1.000**_†_|0.983<br>_†_|0.703_†_|
|BERTScore|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.943_†_|
|BARTScore|0.784_†_|0.925_†_|0.888_†_|**1.000**_†_|0.861_†_|0.897_†_|
|BARTScore+|0.443_†_|0.725_†_|0.530_†_|**1.000**_†_|0.597_†_|0.885_†_|
|BARTScore-P|-0.095_†_|0.284_†_|0.404_†_|0.966_†_|0.476_†_|0.587_†_|
|PRISM|0.879_†_|0.995_†_|0.902_†_|0.996<br>_†_|0.927_†_|0.932_†_|
|CTC-Relevancy|0.031|-0.124_†_|0.265_†_|0.954_†_|0.321_†_|0.496_†_|
|CTC-Consistency|0.974<br>_†_|0.965_†_|0.393_†_|-0.429_†_|0.711_†_|0.718_†_|
|**ROSCOE Metrics(reference-based on**||(**_s_**_,_**_h_**)**)**|||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||||
|Reasoning Alignment|0.010|0.995_†_|0.999<br>_†_|0.008|**1.000**_†_|0.947_†_|
|Hallucination|-0.000|0.994_†_|0.998_†_|-0.305_†_|**1.000**_†_|0.949_†_|
|Redundancy|-0.004|0.994_†_|0.998_†_|-0.278_†_|**1.000**_†_|0.949_†_|
|Commonsense|-0.006|0.012|0.998_†_|**1.000**_†_|**1.000**_†_|0.949_†_|
|Missing Step|-0.010|0.012|0.998_†_|**1.000**_†_|**1.000**_†_|0.949_†_|
|Semantic Coverage-Step|**1.000**_†_|**1.000**_†_|0.997_†_|**1.000**_†_|**1.000**_†_|0.941_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings|||||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|0.999<br>_†_|**1.000**_†_|**1.000**_†_|0.944_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||||
|Reasoning Alignment|0.035|0.995_†_|**1.000**_†_|0.003|**1.000**_†_|0.950<br>_†_|
|Hallucination|0.006|0.994_†_|**1.000**_†_|-0.302_†_|**1.000**_†_|0.949_†_|
|Redundancy|0.007|0.994_†_|**1.000**_†_|-0.273_†_|**1.000**_†_|0.949_†_|
|Commonsense|0.003|0.025|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.949_†_|
|Missing Step|0.003|0.028|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.949_†_|
|Semantic Coverage-Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.948_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.946_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||
|Reasoning Alignment|0.019|0.995_†_|**1.000**_†_|0.027|**1.000**_†_|0.949_†_|
|Hallucination|0.009|0.996_†_|**1.000**_†_|-0.273_†_|**1.000**_†_|0.945_†_|
|Redundancy|0.008|0.996_†_|**1.000**_†_|-0.248_†_|**1.000**_†_|0.946_†_|
|Commonsense|0.007|-0.017|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.945_†_|
|Missing Step|0.006|-0.014|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.946_†_|
|Semantic Coverage-Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.946_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**0.956**_†_|



38 

Published as a conference paper at ICLR 2023 

Table 27: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **Entailment Bank** Diagnostics datasets. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Repet. Hallu. **|**Grammar **<br>**Error**|**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate **<br>**Step**|**Semantic**<br>**Error**|
|---|---|---|---|---|---|---|---|
|Rouge-1|**1.000**_†_ **1.000**_†_|0.582_†_|**1.000**_†_|0.000_†_|0.000_†_|**1.000**_†_|**1.000**_†_|
|Rouge-2|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|0.982_†_|0.935_†_|**1.000**_†_|**1.000**_†_|
|Rouge-L|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|BLEURT|0.197_†_ 0.821_†_|0.163_†_|0.786_†_|0.253_†_|0.021|0.640_†_|0.174_†_|
|BERTScore|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|BARTScore|0.574_†_ 0.659_†_|0.574_†_|0.802_†_|0.983_†_|0.864_†_|0.479_†_|0.656_†_|
|BARTScore+|0.195_†_ 0.430_†_|0.214_†_|0.839_†_|0.988<br>_†_|0.813_†_|0.253_†_|0.555_†_|
|BARTScore-P|0.159_†_ 0.272_†_|0.337_†_|0.633_†_|0.877_†_|0.656_†_|0.355_†_|0.415_†_|
|PRISM|0.707_†_ 0.850_†_|0.612_†_|0.781_†_|0.939_†_|0.833_†_|0.466_†_|0.660_†_|
|CTC-Relevancy|0.311_†_ 0.250_†_|-0.039|0.457_†_|0.266_†_|0.214_†_|0.036|0.254_†_|
|CTC-Consistency|0.768_†_ 0.814_†_|0.133_†_|-0.022|0.479_†_|0.413_†_|0.233_†_|0.474_†_|
|**ROSCOE Metrics(reference-based on** (**_s_**_,_**_h_**)**)**||||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence||embeddings||||||
|Reasoning Alignment|-0.078<br>**1.000**_†_|**1.000**_†_|0.056|-0.034|-0.001|**1.000**_†_|0.990_†_|
|Hallucination|0.002<br>**1.000**_†_|**1.000**_†_|-0.200_†_|0.041|0.017|**1.000**_†_|0.995_†_|
|Redundancy|-0.072<br>**1.000**_†_|**1.000**_†_|-0.115_†_|0.034|0.055|**1.000**_†_|0.993_†_|
|Commonsense|0.025<br>-0.016|0.438_†_|0.264_†_|0.025|-0.012|0.237_†_|0.327_†_|
|Missing Step|-0.077<br>-0.038|0.986_†_|0.993_†_|0.040|0.060|0.976<br>_†_|0.985_†_|
|Semantic Coverage-Step|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|0.003|0.036|**1.000**_†_|0.999<br>_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence||embeddings||||||
|Semantic Coverage-Chain|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||
|Reasoning Alignment|-0.024<br>**1.000**_†_|**1.000**_†_|0.039|0.035|0.004|**1.000**_†_|0.990_†_|
|Hallucination|-0.011<br>**1.000**_†_|**1.000**_†_|-0.059|-0.042|-0.008|**1.000**_†_|0.990_†_|
|Redundancy|-0.011<br>**1.000**_†_|**1.000**_†_|-0.023|0.034|0.034|**1.000**_†_|0.994_†_|
|Commonsense|-0.021<br>0.090|0.433_†_|0.347_†_|-0.049|0.002|0.276_†_|0.342_†_|
|Missing Step|-0.011<br>0.052|0.988<br>_†_|0.999<br>_†_|0.034|0.034|0.976<br>_†_|0.983_†_|
|Semantic Coverage-Step|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|0.106_†_|0.040|**1.000**_†_|0.994_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||
|Semantic Coverage-Chain|0.927<br>_†_ 0.929_†_|0.922_†_|0.951_†_|**1.000**_†_|0.968_†_|0.940_†_|0.941_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||
|Reasoning Alignment|0.040<br>**1.000**_†_|**1.000**_†_|-0.013|0.007|-0.080|**1.000**_†_|0.999<br>_†_|
|Hallucination|-0.012<br>**1.000**_†_|**1.000**_†_|-0.059|0.134_†_|0.040|**1.000**_†_|0.991_†_|
|Redundancy|0.025<br>**1.000**_†_|**1.000**_†_|-0.074|0.041_†_|-0.017|**1.000**_†_|0.994_†_|
|Commonsense|-0.022<br>-0.039|0.368_†_|0.324_†_|0.129_†_|0.039|0.304_†_|0.445_†_|
|Missing Step|0.025<br>0.008|0.988<br>_†_|0.999<br>_†_|0.041_†_|-0.017|0.971_†_|0.982_†_|
|Semantic Coverage-Step|**1.000**_†_ **1.000**_†_|**1.000**_†_|**1.000**_†_|0.117_†_|0.038|**1.000**_†_|0.994_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||
|Semantic Coverage-Chain|0.912_†_ 0.934<br>_†_|0.911_†_|0.950_†_|**1.000**_†_|0.980<br>_†_|0.940_†_|0.931_†_|



39 

Published as a conference paper at ICLR 2023 

Table 28: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **MATH** Diagnostics datasets. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|in App.H.1.)||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Perturbations**_→_|**Repet.**|**Hallu. **|**Grammar **<br>**Error**|**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate **<br>**Step**|**Random**<br>**Number**|**Random**<br>**Operation **|**Shuffe**<br> **Numbers **|**Shuffe**<br> **Operations**|
|Rouge-1|**0.994**_†_|0.961_†_|0.451_†_|**0.996**_†_|0.000_†_|0.000_†_|**1.000**_†_|**1.000**_†_|0.000_†_|0.433_†_|0.000_†_|
|Rouge-2|**0.994**_†_|0.961_†_|0.767_†_|**0.996**_†_|0.990_†_|0.953_†_|**1.000**_†_|**1.000**_†_|0.000_†_|0.999<br>_†_|0.000_†_|
|Rouge-L|**0.994**_†_|0.961_†_|0.769_†_|**0.996**_†_|0.998_†_|**0.997**_†_|**1.000**_†_|**1.000**_†_|0.000_†_|**1.000**_†_|0.000_†_|
|BLEURT|0.363_†_|0.561_†_|0.041_†_|0.773_†_|0.103_†_|0.059_†_|0.156_†_|0.121_†_|0.069_†_|0.028|0.020|
|BERTScore|0.972<br>_†_|0.942_†_|0.978_†_|0.978<br>_†_|**1.000**_†_|0.993<br>_†_|0.964_†_|**1.000**_†_|0.999<br>_†_|0.998_†_|0.998<br>_†_|
|BARTScore|0.413_†_|0.531_†_|0.812_†_|0.837_†_|0.966_†_|0.803_†_|0.429_†_|0.938_†_|0.921_†_|0.989_†_|0.932_†_|
|BARTScore+|0.022|0.208_†_|0.539_†_|0.829_†_|0.611_†_|0.437_†_|0.153_†_|0.804_†_|0.466_†_|0.963_†_|0.606_†_|
|BARTScore-P|-0.084_†_|0.041_†_|0.248_†_|0.474_†_|0.369_†_|0.170_†_|0.015|0.529_†_|0.367_†_|0.799_†_|0.536_†_|
|PRISM|0.337_†_|0.465_†_|0.386_†_|0.506_†_|0.412_†_|0.392_†_|0.266_†_|0.437_†_|0.347_†_|0.521_†_|0.285_†_|
|CTC-Relevancy|-0.045_†_|0.168_†_|0.295_†_|0.592_†_|0.253_†_|0.110_†_|0.141_†_|0.140_†_|0.048_†_|0.169_†_|0.041|
|CTC-Consistency|0.673_†_|0.804_†_|0.491_†_|0.091_†_|0.393_†_|0.240_†_|0.506_†_|0.372_†_|0.112_†_|0.408_†_|0.148_†_|
|**ROSCOE Metrics(reference-based on**||(**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Reasoning Alignment|-0.012|0.998_†_|0.835_†_|0.005|-0.017|-0.013|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|-0.007|0.999<br>_†_|0.854_†_|-0.038_†_|-0.019|-0.007|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|-0.007|0.999<br>_†_|0.853_†_|-0.040_†_|-0.019|-0.005|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|-0.030_†_|-0.004|0.852_†_|0.991_†_|-0.022|-0.005|0.999<br>_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|-0.030_†_|-0.003|0.850_†_|0.991_†_|-0.021|-0.004|0.999<br>_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|0.908_†_|**1.000**_†_|0.852_†_|**1.000**_†_|-0.005|-0.008|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.940_†_|0.946_†_|0.799_†_|0.944_†_|**1.000**_†_|0.980_†_|0.933_†_|**1.000**_†_|0.999<br>_†_|0.998_†_|0.992_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Reasoning Alignment|-0.002|0.998_†_|0.990_†_|-0.004|-0.003|0.018|0.998_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|-0.006|0.999<br>_†_|**0.993**_†_|-0.045_†_|0.012_†_|0.003|0.998_†_|0.996_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|-0.006|0.999<br>_†_|**0.993**_†_|-0.044_†_|0.012_†_|0.003|0.998_†_|0.996_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|-0.006|0.002|0.983_†_|0.989_†_|0.011|0.003|0.995_†_|0.996_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|-0.006|0.001|0.983_†_|0.989_†_|0.011|0.003|0.995_†_|0.996_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|0.914_†_|**1.000**_†_|**0.993**_†_|**1.000**_†_|0.038_†_|0.017_†_|0.998_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.674_†_|0.773_†_|0.762_†_|0.763_†_|0.999<br>_†_|0.868_†_|0.743_†_|0.985_†_|0.958_†_|0.975_†_|0.886_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Reasoning Alignment|0.011|0.999<br>_†_|0.992<br>_†_|0.019|-0.001|0.025|0.998_†_|0.998<br>_†_|0.998_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|-0.008|0.999<br>_†_|**0.993**_†_|-0.032_†_|0.011|0.015_†_|0.998_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|-0.008|0.999<br>_†_|**0.993**_†_|-0.031_†_|0.010|0.016_†_|0.998_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|-0.008|0.000|0.983_†_|0.991_†_|0.010|0.015_†_|0.995_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|-0.008|0.001|0.983_†_|0.990_†_|0.010|0.016_†_|0.995_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|0.913_†_|**1.000**_†_|**0.993**_†_|**1.000**_†_|0.036_†_|0.016_†_|0.998_†_|0.997_†_|0.997_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.674_†_|0.770_†_|0.769_†_|0.775_†_|0.999<br>_†_|0.865_†_|0.741_†_|0.985_†_|0.957_†_|0.973_†_|0.897_†_|



40 

Published as a conference paper at ICLR 2023 

Table 29: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **ProofWriter** Diagnostics datasets. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Repet. **|**Hallu. **|**Grammar **<br>**Error**|**Remove **<br>**Step**|**Shuffe**<br>**Steps**|**Swap**<br>**Steps**|**Negate **<br>**Step**|**Semantic**<br>**Error**|
|---|---|---|---|---|---|---|---|---|
|Rouge-1|**1.000**_†_|**1.000**_†_|0.663_†_|**1.000**_†_|0.000_†_|0.000_†_|**1.000**_†_|**1.000**_†_|
|Rouge-2|**1.000**_†_|**1.000**_†_|0.993_†_|**1.000**_†_|0.932_†_|0.812_†_|**1.000**_†_|**1.000**_†_|
|Rouge-L|**1.000**_†_|**1.000**_†_|0.993_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|BLEURT|0.349_†_|0.829_†_|0.221_†_|0.695_†_|0.076_†_|0.042_†_|0.788_†_|0.597_†_|
|BERTScore|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|BARTScore|0.334_†_|0.430_†_|0.303_†_|0.417_†_|0.986_†_|0.748_†_|0.343_†_|0.391_†_|
|BARTScore+|0.245_†_|0.391_†_|0.251_†_|0.598_†_|0.996_†_|0.775_†_|0.278_†_|0.515_†_|
|BARTScore-P|0.020|0.100_†_|0.124_†_|0.260_†_|0.595_†_|0.318_†_|0.114_†_|0.260_†_|
|PRISM|0.829_†_|0.956_†_|0.822_†_|0.924<br>_†_|0.997<br>_†_|0.970_†_|0.871_†_|0.947_†_|
|CTC-Relevancy|0.376_†_|0.409_†_|0.354_†_|0.725_†_|0.621_†_|0.396_†_|0.391_†_|0.419_†_|
|CTC-Consistency|0.537_†_|0.635_†_|0.199_†_|0.009|0.608_†_|0.412_†_|0.404_†_|0.376_†_|
|**ROSCOE Metrics(reference-based on**||(**_s_**_,_**_h_**)**)**|||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence|||embeddings||||||
|Reasoning Alignment|-0.017|0.997_†_|0.994_†_|-0.049_†_|-0.007|-0.000|**1.000**_†_|0.990_†_|
|Hallucination|0.015|0.997_†_|0.990_†_|0.012|-0.001|-0.004|**1.000**_†_|0.939_†_|
|Redundancy|0.007|0.997_†_|0.994_†_|0.048_†_|-0.000|0.010|**1.000**_†_|0.992_†_|
|Commonsense|0.012|0.000|0.187_†_|0.379_†_|-0.002|-0.010|0.180_†_|0.169_†_|
|Missing Step|0.003|-0.001|0.872_†_|0.843_†_|0.004|0.011|0.876_†_|0.870_†_|
|Semantic Coverage-Step|0.793_†_|**1.000**_†_|0.994_†_|**1.000**_†_|-0.008|0.021_†_|**1.000**_†_|0.984_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence|||embeddings||||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|0.994_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||||||
|Reasoning Alignment|0.012|0.998<br>_†_|**1.000**_†_|-0.041_†_|0.000|0.001|**1.000**_†_|**1.000**_†_|
|Hallucination|0.016|0.998<br>_†_|0.998_†_|0.031_†_|-0.005|0.002|**1.000**_†_|**1.000**_†_|
|Redundancy|0.010|0.998<br>_†_|**1.000**_†_|0.037_†_|0.015_†_|0.003|**1.000**_†_|**1.000**_†_|
|Commonsense|0.017|0.006|0.242_†_|0.415_†_|-0.005|0.002|0.238_†_|0.234_†_|
|Missing Step|0.010|0.004|0.871_†_|0.839_†_|0.016_†_|0.004|0.874_†_|0.866_†_|
|Semantic Coverage-Step|0.818_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.059_†_|0.021_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||||||
|Semantic Coverage-Chain|0.995<br>_†_|0.998<br>_†_|0.995_†_|0.997<br>_†_|**1.000**_†_|0.999<br>_†_|0.998<br>_†_|0.997<br>_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||
|Reasoning Alignment|0.016|0.996_†_|**1.000**_†_|-0.030_†_|0.000|0.005|**1.000**_†_|**1.000**_†_|
|Hallucination|0.003|0.997_†_|0.998_†_|0.016|0.005|-0.010|**1.000**_†_|**1.000**_†_|
|Redundancy|0.020_†_|0.997_†_|**1.000**_†_|0.050_†_|0.019_†_|0.003|**1.000**_†_|**1.000**_†_|
|Commonsense|0.001|-0.009|0.243_†_|0.404_†_|0.010|-0.009|0.253_†_|0.235_†_|
|Missing Step|0.015_†_|-0.007|0.871_†_|0.839_†_|0.020_†_|0.003|0.873_†_|0.866_†_|
|Semantic Coverage-Step|0.817_†_|**1.000**_†_|0.999<br>_†_|**1.000**_†_|0.060_†_|0.022_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||
|Semantic Coverage-Chain|0.994_†_|0.998<br>_†_|0.995_†_|0.996_†_|**1.000**_†_|0.999<br>_†_|0.998<br>_†_|0.996_†_|



41 

Published as a conference paper at ICLR 2023 

Table 30: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **ASDIV** Diagnostics datasets. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Random**<br>**Number**|**Random**<br>**Operation **|**Shuffe**<br> **Numbers **|**Shuffe**<br> **Operations**|
|---|---|---|---|---|
|Rouge-1|**1.000**_†_|0.000_†_|0.027_†_|0.000_†_|
|Rouge-2|**1.000**_†_|0.000_†_|**1.000**_†_|0.000_†_|
|Rouge-L|**1.000**_†_|0.000_†_|**1.000**_†_|0.000_†_|
|BLEURT|0.876_†_|0.930_†_|0.125|-0.212|
|BERTScore|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|BARTScore|0.871_†_|0.823_†_|0.954<br>_†_|0.924_†_|
|BARTScore+|0.948_†_|0.839_†_|**1.000**_†_|0.955<br>_†_|
|BARTScore-P|0.738_†_|0.642_†_|0.830_†_|0.712_†_|
|PRISM|0.998<br>_†_|0.989<br>_†_|**1.000**_†_|**1.000**_†_|
|CTC-Relevancy|0.398_†_|0.188_†_|0.160_†_|-0.136|
|CTC-Consistency|0.600_†_|0.131_†_|0.294_†_|-0.106|
|**ROSCOE Metrics(reference-based on** (**_s_**_,_**_h_**)**)**|||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||
|Reasoning Alignment|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings|||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||
|Reasoning Alignment|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_|||sentence embeddings||
|Reasoning Alignment|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_|||sentence embeddings||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|



42 

Published as a conference paper at ICLR 2023 

Table 31: Somers’ _D_ correlations of **all** metrics per different perturbation applied on **AQUA** Diagnostics datasets. All metrics are measured **reference-based** on ( _**r** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** . The second best models are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.1.) 

|**Perturbations**_→_|**Repet. **|**Halluc. **|**Grammar **<br>**error**|**Remove **<br>**step**|**Shuffe**<br>**steps**|**Swap**<br>**steps**|**Negate **<br>**step**|**Random**<br>**number**|**Random**<br>**operation **|**Shuffe**<br> **numbers **|**Shuffe**<br> **operations**|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Rouge-1|**1.000**_†_|**1.000**_†_|0.394_†_|**1.000**_†_|0.000_†_|0.000_†_|**1.000**_†_|**1.000**_†_|0.000_†_|0.000_†_|0.000_†_|
|Rouge-2|**1.000**_†_|**1.000**_†_|0.866_†_|**1.000**_†_|0.984<br>_†_|0.967<br>_†_|**1.000**_†_|**1.000**_†_|0.000_†_|**1.000**_†_|0.000_†_|
|Rouge-L|**1.000**_†_|**1.000**_†_|0.866_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.000_†_|**1.000**_†_|0.000_†_|
|BLEURT|0.193_†_|0.803_†_|0.129|0.946_†_|0.076|0.028|0.640_†_|0.627_†_|0.385_†_|-0.041|-0.113|
|BERTScore|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|BARTScore|0.568_†_|0.596_†_|0.688_†_|0.870_†_|0.958_†_|0.883_†_|0.520_†_|0.930_†_|0.834<br>_†_|0.984_†_|0.849_†_|
|BARTScore+|0.283_†_|0.597_†_|0.613_†_|0.953_†_|0.960_†_|0.901_†_|0.368_†_|0.980<br>_†_|0.521_†_|0.995<br>_†_|0.814_†_|
|BARTScore-P|0.045|0.025|0.400_†_|0.679_†_|0.650_†_|0.547_†_|0.148|0.846_†_|0.559_†_|0.905_†_|0.726_†_|
|PRISM|0.815_†_|0.900_†_|0.724_†_|0.888_†_|0.941_†_|0.901_†_|0.689_†_|0.942_†_|0.772_†_|0.970_†_|0.870_†_|
|CTC-Relevancy|0.054|-0.044|0.102|0.409_†_|0.116|0.141|0.062|0.290_†_|0.030|0.409_†_|0.023|
|CTC-Consistency|0.833_†_|0.766_†_|0.106|-0.150_†_|0.270_†_|0.257_†_|0.256_†_|0.774_†_|0.232_†_|0.741_†_|0.063|
|**ROSCOE Metrics(reference-based on**||(**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Reasoning Alignment|0.030|0.997<br>_†_|0.985_†_|-0.014|0.054|-0.076|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|0.029|0.992_†_|0.992_†_|-0.084|0.024|-0.030|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|0.032|0.992_†_|0.992_†_|-0.088|0.023|-0.040|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|0.029|0.020|0.992_†_|**1.000**_†_|0.024|-0.038|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|0.032|0.023|0.992_†_|**1.000**_†_|0.023|-0.047|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|0.947_†_|**1.000**_†_|0.987_†_|**1.000**_†_|0.006|-0.009|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Semantic Coverage-Chain|**1.000**_†_|**1.000**_†_|0.995<br>_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Reasoning Alignment|-0.061|**1.000**_†_|**1.000**_†_|0.018|0.049|0.006|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|-0.053|**1.000**_†_|**1.000**_†_|-0.069|-0.017|-0.041|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|-0.054|**1.000**_†_|**1.000**_†_|-0.063|-0.016|-0.041|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|-0.053|0.065|0.992_†_|**1.000**_†_|-0.017|-0.041|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|-0.054|0.064|0.992_†_|**1.000**_†_|-0.016|-0.041|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|0.966_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.041_†_|0.008|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.965_†_|0.965_†_|0.944_†_|0.988<br>_†_|**1.000**_†_|**1.000**_†_|0.953_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.973<br>_†_|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Reasoning Alignment|-0.089|**1.000**_†_|**1.000**_†_|0.085|0.054|0.036|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Hallucination|-0.039|**1.000**_†_|**1.000**_†_|0.003|-0.024|-0.025|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Redundancy|-0.043|**1.000**_†_|**1.000**_†_|0.006|-0.016|-0.016|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Commonsense|-0.039|0.083|0.992_†_|**1.000**_†_|-0.024|-0.025|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Missing Step|-0.043|0.089|0.992_†_|**1.000**_†_|-0.016|-0.016|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|Semantic Coverage-Step|0.967<br>_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.041_†_|0.008|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.957_†_|0.957_†_|0.959_†_|0.980_†_|**1.000**_†_|**1.000**_†_|0.971<br>_†_|**1.000**_†_|**1.000**_†_|**1.000**_†_|0.972_†_|



43 

Published as a conference paper at ICLR 2023 

## H.2 EXPERIMENTS WITH HUMAN JUDGEMENT DATASETS 

In this section, we present Somers’ _D_ correlation of all metrics on all Human Judged datasets in separate tables. Specifically, Table 32 summarizes meta-evaluations for ROSCOE metrics in comparison to baselines on **all** human judged datasets. Fine-grained evaluations are presented in Table 33 for DROP, Table 34, 38 for GSM8K, Table 35, 39 for ESNLI, Table 36 for CosmosQA, and Table 37 for SemEVAL. Human evaluation perspectives used in evaluations are described in App. Table 15. 

Looking at how errors are captured by ROSCOE reference-free scores (Fig. 8), we observe strongest correlations between _Redundancy_ error and _Repetition-*_ , _Self-Consistency_ scores. _Repetition_ error is not present in this analysis as it has at most 3 occurrences per dataset. Out of the all considered scores, _Self-Consistency_ is able to cover 6 out of 7 evaluation perspectives, except _Missing Step_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning_images/iclr_2023_ROSCOE_A_Suite_of_Metrics_for_Scoring_Step-by-Step_Reasoning.pdf-0044-04.png)


Figure 8: Relative presence of the strong score-error correlation, measured as the number of datasets where for each score and evaluation perspective pair Somers’ _D_ correlation value is in the 90 _[th]_ percentile, normalized by the total number of datasets where this type of perturbation occurs. Statistics collected over ROSCOE reference-free scores with finetuned SimCSE embeddings, and evaluation perspectives where at least 10 errors are present in a dataset. 

We further look at specific human annotated examples where our ROSCOE gives highest and lowest scores to understand strength and weaknesses of the proposed approach. Results are summarized in Table 40. Similar analysis for diagnostic datasets is summarized in Table 41. 

44 

Published as a conference paper at ICLR 2023 

Table 32: Somers’ _D_ correlations of **all** metrics on five **human judged** datasets. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each dataset is in **bold** and second best is underlined. Correlations that are not significant ( _p_ - _value ≥_ 0 _._ 05) are omitted when aggregating, and "-" denotes an absence of any significant correlations. (Continued from _§_ 6, more details in App. H.2.) 

||**DROP GSM8K **|**DROP GSM8K **|**ESNLI COSMOS SemEVAL**|**ESNLI COSMOS SemEVAL**|**ESNLI COSMOS SemEVAL**|
|---|---|---|---|---|---|
|Rouge-1|0.239|0.180|0.559|-0.264|-0.520|
|Rouge-2|0.320|-|0.502|0.180|-|
|Rouge-L|0.278|0.252|0.557|-0.441|-0.478|
|BLEURT|0.328|0.256|0.541|0.218|-0.356|
|BERTScore|0.275|0.235|0.590|-0.420|-0.295|
|BARTScore|-0.835|-0.546|0.549|-0.544|-|
|BARTScore+|-0.665|-|0.482|-0.186|-|
|BARTScore-P|-0.642|-|0.255|-0.207|-|
|PRISM|-0.733|-0.455|0.580|-0.376|-|
|CTC-Relevance|0.333|-0.371|0.334|-|-0.349|
|CTC-Consistency|0.462|-0.174|0.647|0.275|-0.301|
|**ROSCOE Metrics(reference-free on** (**_s_**_,_**_h_**)**)**||||||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||
|Faithfulness-Step|0.496|-|0.403|-|-|
|Faithfulness-Token|0.417|-|0.521|-0.320|-|
|Info-Step|0.500|0.178|0.493|-|-|
|Repetition-Token|0.578|0.392|0.441|0.555|0.337|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||
|Faithfulness-Step|0.297|-|0.423|-0.424|0.330|
|Faithfulness-Token|0.290|-0.443|0.524|-0.515|0.186|
|Info-Step|0.301|-|0.542|-0.429|-|
|Repetition-Token|0.790|0.500|**0.799**|0.638|0.485|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||
|Faithfulness-Step|0.477|-0.192|0.502|-0.381|-|
|Faithfulness-Token|0.454|-|0.540|-0.420|-|
|Info-Step|0.510|-|0.599|-0.409|-0.321|
|Repetition-Token|0.578|0.392|0.441|0.555|0.337|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||
|Info-Chain|0.446|0.385|0.310|-|-|
|Repetition-Step|0.824|0.514|0.530|0.593|0.411|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||
|Info-Chain|0.406|-|0.507|-0.198|0.367|
|Repetition-Step|0.791|0.471|0.487|0.642|0.508|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||
|Info-Chain|0.271|-|0.531|-0.367|-|
|Repetition-Step|**0.799**|**0.638**|0.484|**0.658**|**0.535**|
|ROSCOE-LI||||||
|Source-Consistency|0.390|0.172|0.425|0.444|-|
|Self-Consistency|0.584|0.345|0.531|0.417|0.372|
|ROSCOE-LC||||||
|Grammar|-|-0.184|0.255|-|0.517|
|Perplexity-Step|0.205|-0.307|0.345|-|-|
|Perplexity-Chain|-0.611|-0.273|0.447|-0.212|-0.373|



45 

Published as a conference paper at ICLR 2023 

Table 33: Somers’ _D_ correlation of all metrics on **DROP human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** , second best are underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

||QUAL|COH|COMMON|FACT|HALL|RED|REP|LOGIC|MATH|GRAM|MISS|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Rouge-1|0.157_†_|-0.160|0.436|0.239_†_|-0.335|-0.731_†_|-0.736|0.030|0.702|0.196|0.173_†_|
|Rouge-2|0.137_†_|-0.146|0.488|0.320_†_|-0.284|-0.716_†_|-0.442|0.027|0.584|-0.035|0.129|
|Rouge-L|0.131_†_|-0.201_†_|0.465|0.278_†_|-0.345|-0.749_†_|-0.815|-0.012|0.745|0.027|0.146|
|BLEURT|0.121_†_|-0.101|0.256|0.328_†_|-0.333|-0.725_†_|-0.370|0.078|0.514|0.101|0.087|
|BERTScore|0.133_†_|-0.115|0.494|0.275_†_|-0.177|-0.647_†_|-0.043|-0.003|0.707|0.098|0.142|
|BARTScore|-0.088|-0.392_†_|0.575|0.161|-0.454|-0.835_†_|-0.894|-0.225|0.038|0.150|-0.134|
|BARTScore+|0.039|-0.159|0.536|0.192|-0.553|-0.665_†_|-0.841|-0.141|0.303|0.261|0.066|
|BARTScore-P|-0.007|-0.152|0.546|0.169|-0.473|-0.642_†_|-0.894|-0.039|0.380|0.265|0.012|
|PRISM|0.129_†_|-0.081|0.465|0.207|-0.379|-0.733_†_|-0.361|-0.071|0.668|0.165|0.048|
|CTC-Relevance|-0.027|-0.100|-0.072|0.333_†_|-0.041|-0.622_†_|-0.087|-0.091|0.394|0.134|-0.056|
|CTC-Consistency|0.030|-0.133|0.243|0.462_†_|-0.148|-0.657_†_|-0.106|-0.041|0.769|0.106|-0.002|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|0.096|-0.095|0.572|0.496_†_|0.143|-0.566_†_|-0.322|0.278|0.760|-0.022|0.042|
|Faithfulness-Token|0.177_†_|-0.016|0.472|0.417_†_|-0.189|-0.525_†_|-0.038|0.131|0.678|0.002|0.130|
|Info-Step|0.142_†_|-0.069|**0.643**|0.500<br>_†_|0.216|-0.581_†_|-0.284|**0.287**_†_|**0.846**|0.024|0.125|
|Repetition-Token|0.055|0.210_†_|0.018|0.170|0.340|0.578_†_|**0.952**|0.026|-0.822|-0.140|0.062|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Faithfulness-Step|0.054|-0.156|0.153|0.297_†_|-0.073|-0.578_†_|-0.111|0.116|-0.221|0.001|-0.042|
|Faithfulness-Token|0.156_†_|-0.042|0.362|0.290_†_|-0.388|-0.504_†_|-0.216|-0.058|0.288|0.063|0.085|
|Info-Step|0.090|-0.116|0.308|0.301_†_|-0.153|-0.619_†_|-0.043|0.021|0.250|0.078|0.020|
|Repetition-Token|0.130_†_|0.370<br>_†_|0.027|0.087|0.313|0.790_†_|0.519|0.114|-0.822|-0.181|0.053|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|0.084|-0.128|0.195|0.477_†_|-0.061|-0.643_†_|-0.596|0.085|0.236|-0.061|-0.051|
|Faithfulness-Token|0.186_†_|-0.031|0.414|0.454_†_|-0.245|-0.574_†_|-0.139|0.098|0.514|0.047|0.093|
|Info-Step|0.141_†_|-0.095|0.443|**0.510**_†_|-0.121|-0.692_†_|-0.471|0.040|0.731|0.066|0.010|
|Repetition-Token|0.055|0.210_†_|0.018|0.170|0.340|0.578_†_|**0.952**|0.026|-0.822|-0.140|0.062|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.108|-0.128|0.462|0.222|-0.393|-0.446_†_|-0.375|0.025|0.207|0.446<br>_†_|0.114|
|Repetition-Step|0.036|**0.400**_†_|-0.543|-0.258_†_|0.165|**0.824**_†_|0.909|0.052|-0.822|0.192|-0.016|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Info-Chain|0.165_†_|0.020|0.395|0.240_†_|-0.027|-0.480_†_|-0.135|0.122|0.394|0.406_†_|0.106|
|Repetition-Step|0.052|0.358_†_|-0.111|-0.115|0.260|0.791_†_|0.856|0.105|-0.822|0.066|0.015|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.138_†_|-0.090|0.504|0.271_†_|-0.158|-0.664_†_|-0.130|-0.089|0.611|0.381|0.093|
|Repetition-Step|0.073|0.357_†_|-0.143|-0.085|0.451|0.799<br>_†_|0.918|0.128|-0.822|0.047|0.021|
|ROSCOE-LI||||||||||||
|Source-Consistency|0.200<br>_†_|0.243_†_|0.462|0.390_†_|-0.004|0.085|0.697|0.009|0.365|0.420|0.184<br>_†_|
|Self-Consistency|0.032|0.295_†_|-0.076|0.198|0.201|0.584_†_|0.139|0.187|-0.707|0.344|-0.075|
|ROSCOE-LC||||||||||||
|Grammar|**0.220**_†_|0.141|0.250|0.260|**0.536**|0.001|-0.553|0.169|0.260|**0.450**|0.111|
|Perplexity-Step|0.185_†_|0.034|0.214|-0.002|-0.112|-0.320_†_|-0.827|-0.043|0.332|0.259|**0.205**_†_|
|Perplexity-Chain|0.087|-0.152|0.185|-0.104|-0.515|-0.611_†_|-0.952|-0.178|0.663|0.103|0.117|



46 

Published as a conference paper at ICLR 2023 

Table 34: Somers’ _D_ correlation of all metrics on **GSM8K human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** , second best is underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

|QUAL<br>COH<br>COMMON<br>FACT<br>HALL<br>RED<br>REP<br>LOGIC|MATH<br>GRAM<br>MISS|
|---|---|
|Rouge-1<br>0.122<br>0.202_†_<br>0.071<br>0.156<br>0.275<br>0.037<br>0.222<br>0.073<br>Rouge-2<br>0.089<br>0.139<br>0.060<br>0.102<br>0.175<br>-0.016<br>0.561<br>0.030<br>Rouge-L<br>0.176_†_<br>0.268<br>_†_<br>0.195<br>_†_<br>0.169<br>0.180<br>0.054<br>0.558<br>0.120<br>BLEURT<br>0.160_†_<br>0.248_†_<br>0.134<br>0.256<br>_†_<br>0.099<br>0.075<br>0.227<br>0.124<br>BERTScore<br>0.173_†_<br>0.220_†_<br>0.112<br>0.116<br>0.168<br>0.095<br>0.955<br>0.138<br>BARTScore<br>0.009<br>0.035<br>-0.047<br>0.044<br>-0.246<br>-0.261<br>0.424<br>-0.002<br>BARTScore+<br>0.064<br>0.132<br>0.078<br>0.054<br>-0.003<br>-0.039<br>0.879<br>0.085<br>BARTScore-P<br>0.037<br>0.059<br>0.042<br>0.061<br>-0.116<br>-0.102<br>0.561<br>-0.002<br>PRISM<br>-0.112<br>-0.075<br>-0.099<br>-0.037<br>-0.385<br>-0.455_†_<br>-0.086<br>-0.171<br>CTC-Relevance<br>-0.086<br>-0.148_†_<br>-0.077<br>-0.111<br>0.009<br>-0.371_†_<br>0.566<br>-0.106<br>CTC-Consistency<br>-0.157_†_<br>-0.203_†_<br>-0.206_†_<br>-0.129<br>-0.013<br>-0.318_†_<br>0.556<br>-0.174_†_|0.233<br>0.173<br>0.180_†_<br>0.113<br>0.051<br>0.148<br>0.212<br>-0.125<br>0.252_†_<br>0.057<br>0.077<br>0.210_†_<br>0.113<br>0.054<br>0.235_†_<br>-0.546_†_<br>-0.321<br>-0.003<br>-0.298<br>-0.281<br>0.090<br>-0.322<br>-0.097<br>0.019<br>-0.341<br>0.130<br>-0.093<br>-0.093<br>-0.061<br>-0.088<br>-0.019<br>-0.056<br>-0.204_†_|
|**ROSCOE Metrics(reference-free on** (**_s_**_,_**_h_**)**)**||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings<br>Faithfulness-Step<br>0.012<br>0.083<br>0.097<br>0.014<br>-0.138<br>-0.019<br>0.182<br>-0.024<br>Faithfulness-Token<br>-0.012<br>0.036<br>-0.052<br>-0.067<br>-0.099<br>-0.304<br>0.788<br>-0.042<br>Info-Step<br>0.059<br>0.137<br>0.178_†_<br>0.082<br>-0.025<br>0.059<br>0.364<br>0.031<br>Repetition-Token<br>0.200_†_<br>0.193_†_<br>0.186_†_<br>0.075<br>0.224<br>0.392_†_<br>0.788<br>0.183_†_<br>ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings<br>Faithfulness-Step<br>-0.039<br>-0.016<br>0.002<br>-0.027<br>0.012<br>-0.376<br>0.581<br>-0.173<br>Faithfulness-Token<br>-0.078<br>-0.002<br>-0.090<br>-0.079<br>-0.141<br>-0.443_†_<br>0.485<br>-0.137<br>Info-Step<br>0.095<br>0.121<br>0.099<br>0.139<br>0.112<br>-0.065<br>0.662<br>-0.016<br>Repetition-Token<br>0.214_†_<br>0.208_†_<br>0.184_†_<br>0.081<br>**0.500**_†_<br>0.238<br>0.747<br>0.208<br>_†_<br>ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings<br>Faithfulness-Step<br>-0.061<br>-0.034<br>-0.006<br>0.002<br>-0.158<br>-0.142<br>0.237<br>-0.192_†_<br>Faithfulness-Token<br>-0.051<br>-0.008<br>-0.087<br>-0.084<br>-0.168<br>-0.352<br>0.732<br>-0.098<br>Info-Step<br>0.059<br>0.102<br>0.106<br>0.124<br>0.040<br>0.046<br>0.545<br>-0.053<br>Repetition-Token<br>0.200_†_<br>0.193_†_<br>0.186_†_<br>0.075<br>0.224<br>0.392_†_<br>0.788<br>0.183_†_|-0.101<br>0.286<br>0.027<br>-0.408<br>0.204<br>0.016<br>-0.303<br>0.329<br>0.108<br>0.443<br>0.173<br>0.270_†_<br>-0.127<br>0.102<br>-0.021<br>-0.623_†_<br>0.074<br>-0.068<br>-0.056<br>0.092<br>0.148<br>0.498<br>_†_<br>**0.339**<br>0.306<br>_†_<br>-0.221<br>0.168<br>-0.045<br>-0.424<br>0.145<br>-0.034<br>-0.236<br>0.099<br>0.131<br>0.443<br>0.173<br>0.270_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings<br>Info-Chain<br>0.097<br>0.064<br>0.059<br>0.080<br>0.035<br>0.385_†_<br>0.722<br>0.103<br>Repetition-Step<br>0.199_†_<br>0.166_†_<br>0.168<br>0.145<br>0.254<br>0.514<br>_†_<br>0.869<br>0.152<br>ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings<br>Info-Chain<br>0.059<br>0.071<br>0.020<br>0.043<br>0.039<br>0.128<br>0.288<br>0.004<br>Repetition-Step<br>0.218_†_<br>0.161_†_<br>0.167<br>0.227_†_<br>0.309<br>0.471_†_<br>0.323<br>0.158<br>ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings<br>Info-Chain<br>0.042<br>0.051<br>0.043<br>0.057<br>0.240<br>0.041<br>0.808<br>-0.007<br>Repetition-Step<br>**0.322**_†_<br>**0.299**_†_<br>**0.275**_†_<br>0.227_†_<br>0.466<br>**0.638**_†_<br>0.879<br>0.192_†_|0.121<br>-0.110<br>0.130<br>0.230<br>0.176<br>0.222_†_<br>0.260<br>-0.026<br>0.098<br>0.486<br>-0.301<br>0.245_†_<br>-0.211<br>0.038<br>0.130<br>**0.563**_†_<br>-0.311<br>**0.354**_†_|
|ROSCOE-LI<br>Source-Consistency<br>0.108<br>0.037<br>-0.019<br>0.172_†_<br>0.121<br>0.030<br>0.551<br>0.082<br>Self-Consistency<br>0.283<br>_†_<br>0.267_†_<br>0.177<br>**0.345**_†_<br>0.207<br>0.354<br>**0.980**<br>**0.219**_†_|-0.005<br>0.219<br>0.097<br>0.087<br>-0.230<br>0.223_†_|
|ROSCOE-LC<br>Grammar<br>-0.134_†_<br>-0.159_†_<br>-0.260_†_<br>-0.081<br>-0.234<br>-0.246<br>0.056<br>-0.264_†_<br>Perplexity-Step<br>-0.297_†_<br>-0.278_†_<br>-0.366_†_<br>-0.307_†_<br>-0.608_†_<br>-0.591_†_<br>-0.136<br>-0.334_†_<br>Perplexity-Chain<br>-0.332_†_<br>-0.336_†_<br>-0.322_†_<br>-0.273_†_<br>-0.695_†_<br>-0.682_†_<br>-0.556<br>-0.354_†_|-0.207<br>0.298<br>-0.184_†_<br>-0.514_†_<br>0.148<br>-0.331_†_<br>-0.697_†_<br>-0.084<br>-0.408_†_|



47 

Published as a conference paper at ICLR 2023 

Table 35: Somers’ _D_ correlation of all metrics on **ESNLI human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** and second best is underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

||QUAL|COH|COMMON|FACT|HALL|RED|REP|LOGIC|MATH|GRAM|MISS|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Rouge-1|0.213_†_|-0.121|-0.373|0.152|-0.060|-0.284|-|-0.276_†_|0.113|-0.403|0.559_†_|
|Rouge-2|0.216_†_|-0.075|-0.336|0.169|-0.097|-0.240|-|-0.255_†_|-0.080|-0.432|0.502_†_|
|Rouge-L|0.222_†_|-0.041|-0.296|0.201|-0.092|-0.245|-|-0.259|0.053|-0.429|0.557_†_|
|BLEURT|0.154_†_|-0.202|-0.271|0.112|-0.041|-0.230|-|-0.322_†_|0.267|-0.641_†_|0.541_†_|
|BERTScore|0.255_†_|0.019|-0.205|0.222|0.153|-0.179|-|-0.141|-0.360|-0.252|0.590<br>_†_|
|BARTScore|0.189_†_|-0.096|-0.260|0.079|0.071|-0.259|-|-0.200|0.533|-0.688_†_|0.549_†_|
|BARTScore+|0.209_†_|-0.059|-0.192|0.166|0.204|-0.147|-|-0.168|-0.160|-0.578_†_|0.482_†_|
|BARTScore-P|0.092|-0.096|-0.208|0.213|0.092|-0.209|-|-0.192|-0.587|-0.370|0.255_†_|
|PRISM|0.264_†_|0.089|-0.151|0.093|0.224|-0.235|-|-0.042|-0.240|-0.329|0.580_†_|
|CTC-Relevance|0.071|-0.051|-0.074|0.063|-0.020|-0.098|-|-0.093|**0.720**|-0.205|0.334_†_|
|CTC-Consistency|0.029|0.218|**0.647**_†_|-0.009|0.483|0.021|-|0.350<br>_†_|0.533|-0.071|-0.060|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|0.236_†_|-0.163|-0.027|0.096|-0.146|-0.139|-|-0.293_†_|0.560|-0.386|0.403_†_|
|Faithfulness-Token|0.279_†_|0.010|-0.145|0.232|0.201|-0.279|-|-0.119|0.067|-0.293|0.521_†_|
|Info-Step|0.222_†_|-0.127|-0.140|0.143|-0.068|-0.220|-|-0.333_†_|0.400|-0.310|0.493_†_|
|Repetition-Token|-0.037|-0.020|0.403|0.244|0.330|0.441_†_|-|0.153|0.240|-0.059|-0.436_†_|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Faithfulness-Step|0.261_†_|-0.117|-0.058|0.242|-0.425|-0.296_†_|-|-0.248_†_|0.480|-0.244|0.423_†_|
|Faithfulness-Token|0.270_†_|-0.083|-0.216|0.136|0.313|-0.248|-|-0.097|0.347|-0.433|0.524_†_|
|Info-Step|0.290<br>_†_|-0.006|-0.049|0.286_†_|-0.102|-0.245|-|-0.246_†_|0.373|-0.342|0.542_†_|
|Repetition-Token|-0.040|0.129|-0.189|0.169|**0.799**_†_|**0.617**_†_|-|0.151|-0.267|0.084|-0.445_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Faithfulness-Step|0.262_†_|-0.083|-0.244|0.257|-0.163|-0.372_†_|-|-0.283_†_|0.413|-0.375|0.502_†_|
|Faithfulness-Token|0.275_†_|0.004|-0.230|0.206|0.214|-0.303_†_|-|-0.140|0.347|-0.321|0.540_†_|
|Info-Step|**0.295**_†_|0.046|-0.126|0.299_†_|-0.071|-0.306_†_|-|-0.208|0.333|-0.397_†_|**0.599**_†_|
|Repetition-Token|-0.037|-0.020|0.403|0.244|0.330|0.441_†_|-|0.153|0.240|-0.059|-0.436_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.083|-0.197|-0.466|-0.225|-0.510|-0.259|-|-0.067|0.267|0.074|0.310_†_|
|Repetition-Step|-0.021|0.103|-0.063|0.249|0.310|0.530<br>_†_|-|0.178|-0.107|0.053|-0.433_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Info-Chain|0.191_†_|-0.233|-0.438|0.052|-0.153|-0.253|-|-0.368_†_|0.533|-0.452|0.507_†_|
|Repetition-Step|0.007|0.280|0.173|0.365<br>_†_|0.694|0.487_†_|-|0.225|0.213|0.193|-0.472_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Info-Chain|0.211_†_|-0.083|-0.192|0.212|0.024|-0.319_†_|-|-0.257_†_|0.400|-0.455|0.531_†_|
|Repetition-Step|-0.015|0.180|0.397|0.332|0.367|0.484_†_|-|0.115|0.013|0.125|-0.501_†_|
|ROSCOE-LI||||||||||||
|Source-Consistency|0.012|-0.028|0.334|**0.425**_†_|0.299|-0.055|-|0.112|0.600|-0.258|-0.222_†_|
|Self-Consistency|-0.028|**0.354**_†_|0.156|0.087|0.561|0.370_†_|-|**0.531**_†_|-0.333|0.222|-0.351_†_|
|ROSCOE-LC||||||||||||
|Grammar|0.063|0.069|-0.411|-0.003|0.122|-0.286|-|0.255_†_|-0.027|**0.411**|-0.072|
|Perplexity-Step|0.084|0.087|-0.655_†_|-0.153|0.255|-0.254|-|-0.148|-0.147|0.090|0.345_†_|
|Perplexity-Chain|0.027|0.081|-0.616_†_|-0.289|0.075|-0.447_†_|-|-0.155|-0.533|0.249|0.447_†_|



48 

Published as a conference paper at ICLR 2023 

Table 36: Somers’ _D_ correlation of all metrics on **COSMOS human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** and second best is underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

||QUAL|COH|COMMON|FACT|HALL|RED|REP|LOGIC|GRAM|MISS|
|---|---|---|---|---|---|---|---|---|---|---|
|Rouge-1|-0.011|-0.007|-0.182|-0.077|0.292|-0.576_†_|-0.807|-0.264_†_|-0.644_†_|0.113|
|Rouge-2|0.021|0.028|-0.075|-0.131|0.239|-0.561_†_|-0.174|-0.108|-0.314|0.180<br>_†_|
|Rouge-L|0.011|-0.013|-0.094|-0.044|0.252|-0.637_†_|-0.436|-0.114|-0.441_†_|0.141|
|BLEURT|0.098|0.088|0.019|-0.054|0.097|-0.686_†_|-0.617|-0.181_†_|-0.522_†_|**0.218**_†_|
|BERTScore|0.095|0.113|0.059|-0.055|0.234|-0.478_†_|-0.492|-0.058|-0.420_†_|0.114|
|BARTScore|0.009|0.024|0.159|-0.026|0.001|-0.544_†_|-0.208|-0.122|-0.420|0.124|
|BARTScore+|0.048|0.061|-0.102|-0.004|0.159|-0.507_†_|-0.602|-0.186_†_|-0.499_†_|0.159|
|BARTScore-P|0.009|0.021|-0.149|0.010|0.267|-0.385_†_|-0.508|-0.207_†_|-0.453_†_|0.142|
|PRISM|0.058|0.091|-0.046|-0.156|0.311|-0.446_†_|-0.428|-0.036|-0.376_†_|0.157|
|CTC-Relevance|0.070|0.035|0.246|0.155|0.233|-0.294|-0.780|-0.001|-0.349|0.016|
|CTC-Consistency|0.093|0.097|**0.275**_†_|0.084|0.140|-0.032|-0.201|0.051|0.064|-0.006|
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||||
|Faithfulness-Step|0.089|0.109|-0.008|0.149|0.256|-0.322|-0.216|-0.019|0.034|-0.011|
|Faithfulness-Token|0.038|0.039|-0.032|-0.093|0.285|-0.472_†_|-0.220|-0.026|-0.320_†_|0.012|
|Info-Step|0.109|0.082|-0.011|0.119|0.302|-0.340|-0.811|-0.039|-0.135|0.057|
|Repetition-Token|0.050|0.120|0.187|0.103|0.103|0.555_†_|0.231|0.192_†_|0.458_†_|-0.233_†_|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings|||||||||||
|Faithfulness-Step|-0.011|0.044|-0.008|-0.012|0.127|-0.424_†_|-0.235|-0.061|-0.268|-0.096|
|Faithfulness-Token|0.035|0.078|-0.124|-0.128|**0.323**|-0.515_†_|-0.667|-0.036|-0.230|-0.008|
|Info-Step|0.011|0.043|-0.047|-0.023|0.186|-0.429_†_|-0.481|-0.081|-0.268|-0.037|
|Repetition-Token|0.045|0.092|**0.275**_†_|0.122|0.298|0.638_†_|0.398|0.252_†_|0.386|-0.184_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_|||sentence embeddings||||||||
|Faithfulness-Step|0.036|0.022|0.072|0.111|0.184|-0.381_†_|-0.318|-0.076|-0.237|-0.053|
|Faithfulness-Token|0.036|0.030|-0.003|-0.058|0.205|-0.486_†_|-0.333|-0.056|-0.420_†_|-0.020|
|Info-Step|0.026|-0.025|0.069|0.079|0.248|-0.409_†_|-0.720|-0.088|-0.292|0.000|
|Repetition-Token|0.050|0.120|0.187|0.103|0.103|0.555_†_|0.231|0.192_†_|0.458_†_|-0.233_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||||
|Info-Chain|0.013|0.089|0.159|0.048|0.139|-0.219|-0.932|0.113|-0.390|-0.117|
|Repetition-Step|0.011|0.050|0.178|0.132|-0.046|0.593_†_|**0.670**|0.295<br>_†_|0.330|-0.244_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings|||||||||||
|Info-Chain|0.073|0.091|-0.174|0.033|0.145|-0.215|-0.409|-0.198_†_|-0.411_†_|0.103|
|Repetition-Step|0.047|0.127|0.124|0.060|0.153|0.642<br>_†_|0.617|**0.346**_†_|**0.563**_†_|-0.184_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_|||sentence embeddings||||||||
|Info-Chain|0.114|0.055|0.034|-0.005|0.218|-0.367_†_|-0.879|-0.046|-0.222|0.153|
|Repetition-Step|0.061|0.127|0.095|0.076|0.246|**0.658**_†_|0.500|0.256_†_|0.496<br>_†_|-0.145|
|ROSCOE-LI|||||||||||
|Source-Consistency|**0.184**_†_|0.183<br>_†_|0.150|**0.285**_†_|0.241|0.444_†_|0.091|0.111|0.303|0.011|
|Self-Consistency|0.048|0.080|0.190|0.173|-0.021|0.417_†_|0.610|0.192_†_|0.401_†_|-0.252_†_|
|ROSCOE-LC|||||||||||
|Grammar|0.093|**0.189**_†_|-0.065|0.084|0.022|-0.013|0.356|-0.013|0.386|0.051|
|Perplexity-Step|0.122<br>_†_|0.157_†_|-0.208|-0.021|0.034|0.028|-0.140|-0.113|-0.295|0.064|
|Perplexity-Chain|0.083|0.047|-0.193|0.001|-0.073|-0.311_†_|-0.561|-0.212_†_|-0.542_†_|0.130|



49 

Published as a conference paper at ICLR 2023 

Table 37: Somers’ _D_ correlation of all metrics on **SemEVAL human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-free** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** and second best is underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

||QUAL|COH|COMMON|FACT|HALL|RED|REP|LOGIC|GRAM|MISS|
|---|---|---|---|---|---|---|---|---|---|---|
|Rouge-1|-0.199_†_|-0.208_†_|-0.246|-0.118|-0.053|-0.520_†_|-0.398|-0.206|-0.230|0.108|
|Rouge-2|-0.086|0.060|-0.478|-0.063|0.561|-0.232|-0.083|-0.073|0.141|0.090|
|Rouge-L|-0.198_†_|-0.209_†_|-0.498|-0.104|0.090|-0.478_†_|-0.396|-0.115|-0.156|0.058|
|BLEURT|-0.313_†_|-0.383_†_|-0.372|-0.208|-0.034|-0.482_†_|-0.383|-0.356_†_|-0.104|-0.074|
|BERTScore|-0.051|0.064|-0.517|-0.035|0.524|-0.002|-0.218|-0.295_†_|0.266|0.063|
|BARTScore|-0.084|-0.059|-0.140|-0.137|-0.369|-0.054|0.209|-0.284|-0.040|-0.056|
|BARTScore+|-0.046|-0.098|0.652|-0.033|-0.073|0.056|0.204|-0.204|0.048|-0.016|
|BARTScore-P|-0.075|-0.168|0.633|-0.080|0.107|0.096|0.277|-0.230|0.054|-0.082|
|PRISM|-0.082|-0.040|-0.469|-0.115|0.073|-0.174|-0.354|-0.134|-0.079|0.075|
|CTC-Relevance|-0.146_†_|-0.219_†_|0.256|-0.047|0.442|-0.071|-0.121|-0.349_†_|0.101|-0.145|
|CTC-Consistency|-0.178_†_|-0.241_†_|0.101|0.009|0.583|-0.301_†_|-0.296|-0.335_†_|0.208|-0.142|
|**ROSCOE Metrics(reference-free**||**on** (**_s_**_,_**_h_**)**)**|||||||||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||||
|Faithfulness-Step|0.157_†_|0.198_†_|0.275|0.189|0.694|0.233|-0.262|-0.115|0.092|0.115|
|Faithfulness-Token|0.030|0.182_†_|-0.150|0.009|0.714|0.116|-0.442|-0.101|0.262|0.085|
|Info-Step|0.068|0.113|0.111|0.055|0.835|0.003|-0.252|-0.187|0.196|0.129|
|Repetition-Token|0.062|0.150|0.401|0.021|-0.078|0.337_†_|0.670|-0.074|-0.007|-0.131|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence|||embeddings||||||||
|Faithfulness-Step|**0.197**_†_|0.221<br>_†_|0.005|0.201|0.461|0.330_†_|-0.160|-0.019|0.182|0.108|
|Faithfulness-Token|0.030|0.161|-0.208|0.013|0.597|-0.044|-0.495|-0.111|0.063|**0.186**_†_|
|Info-Step|0.111|0.127|-0.111|0.125|0.544|0.161|-0.073|-0.176|0.210|0.076|
|Repetition-Token|0.134_†_|0.178_†_|0.662|0.066|0.364|0.485_†_|0.772|-0.004|0.157|-0.115|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings|||||||||||
|Faithfulness-Step|0.028|0.123|-0.459|0.112|**0.908**|0.174|-0.199|-0.162|0.021|0.059|
|Faithfulness-Token|-0.021|0.133|-0.227|0.021|0.752|0.084|-0.398|-0.134|0.119|0.052|
|Info-Step|-0.040|-0.003|-0.362|0.024|0.777|0.015|-0.296|-0.321_†_|0.038|0.039|
|Repetition-Token|0.062|0.150|0.401|0.021|-0.078|0.337_†_|0.670|-0.074|-0.007|-0.131|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings|||||||||||
|Info-Chain|-0.040|0.008|-0.082|-0.005|0.539|-0.164|-0.015|-0.070|0.086|-0.082|
|Repetition-Step|0.051|0.143|0.546|0.043|-0.024|0.411_†_|0.723|**0.038**|-0.018|-0.195_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence|||embeddings||||||||
|Info-Chain|0.076|0.128|-0.700|0.122|-0.005|0.367_†_|0.117|-0.140|0.068|-0.018|
|Repetition-Step|0.077|0.117|0.633|0.036|0.141|0.508<br>_†_|0.684|-0.026|0.025|-0.155|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings|||||||||||
|Info-Chain|-0.115|-0.038|-0.169|0.059|0.539|-0.260|-0.476|-0.169|-0.081|-0.009|
|Repetition-Step|0.104|0.132|0.787|0.042|0.136|**0.535**_†_|**0.811**|0.006|0.069|-0.169_†_|
|ROSCOE-LI|||||||||||
|Source-Consistency|0.059|0.016|0.546|**0.206**|-0.029|0.006|0.010|-0.241|-0.139|-0.063|
|Self-Consistency|0.162<br>_†_|**0.250**_†_|0.536|0.104|0.383|0.372_†_|0.223|-0.091|0.061|-0.075|
|ROSCOE-LC|||||||||||
|Grammar|-0.076|0.014|0.101|-0.223_†_|-0.335|0.104|0.393|-0.215|**0.517**_†_|-0.144|
|Perplexity-Step|-0.026|-0.053|**0.797**|0.037|-0.607|0.020|-0.019|-0.071|-0.330|-0.039|
|Perplexity-Chain|-0.141_†_|-0.237_†_|0.324|-0.126|-0.650|-0.373_†_|-0.481|-0.151|-0.284|0.039|



50 

Published as a conference paper at ICLR 2023 

Table 38: Somers’ _D_ correlation of all metrics on **GSM8K human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** , second best is underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

|QUAL<br>COH<br>COMMON<br>FACT<br>HALL<br>RED<br>REP|LOGIC<br>MATH<br>GRAM<br>MISS|
|---|---|
|Rouge-1<br>0.572_†_<br>0.506_†_<br>0.590_†_<br>0.513_†_<br>0.532_†_<br>0.498_†_<br>0.694<br>Rouge-2<br>0.551_†_<br>0.520_†_<br>0.566_†_<br>0.511_†_<br>0.467_†_<br>0.439_†_<br>0.758<br>Rouge-L<br>0.591_†_<br>0.542_†_<br>0.605_†_<br>0.559_†_<br>0.452_†_<br>**0.613**_†_<br>0.662<br>BLEURT<br>0.487_†_<br>0.391_†_<br>0.502_†_<br>0.389_†_<br>0.392<br>0.222<br>0.682<br>BERTScore<br>0.505_†_<br>0.425_†_<br>0.488_†_<br>0.475_†_<br>0.477_†_<br>0.585<br>_†_<br>0.763<br>BARTScore<br>0.429_†_<br>0.352_†_<br>0.522_†_<br>0.317_†_<br>0.315<br>0.128<br>0.692<br>BARTScore+<br>0.531_†_<br>0.455_†_<br>0.579_†_<br>0.460_†_<br>0.273<br>0.303<br>0.722<br>BARTScore-P<br>0.343_†_<br>0.280_†_<br>0.376_†_<br>0.268_†_<br>0.210<br>0.062<br>0.621<br>PRISM<br>0.579_†_<br>0.511_†_<br>0.593_†_<br>0.531_†_<br>0.363_†_<br>0.392_†_<br>0.707<br>CTC-Relevance<br>-0.047<br>-0.093<br>-0.065<br>-0.065<br>-0.134<br>-0.286<br>0.460<br>CTC-Consistency<br>-0.272_†_<br>-0.279_†_<br>-0.259_†_<br>-0.264_†_<br>-0.182<br>-0.385_†_<br>-0.399|0.533_†_<br>0.582_†_<br>0.430<br>0.686_†_<br>0.555_†_<br>0.617_†_<br>0.480<br>0.713_†_<br>0.575_†_<br>0.715<br>_†_<br>0.457<br>0.730_†_<br>0.404_†_<br>0.233<br>0.283<br>0.597_†_<br>0.483_†_<br>0.677_†_<br>0.347<br>0.627_†_<br>0.436_†_<br>0.279_†_<br>0.237<br>0.555_†_<br>0.501_†_<br>0.577_†_<br>0.334<br>0.679_†_<br>0.298_†_<br>0.207<br>0.227<br>0.441_†_<br>0.557_†_<br>0.540_†_<br>0.423<br>0.728_†_<br>-0.055<br>0.056<br>0.242<br>-0.102<br>-0.192_†_<br>-0.233<br>0.028<br>-0.320_†_|
|**ROSCOE Metrics(reference-based on** (**_s_**_,_**_h_**)**)**||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence embeddings<br>Hallucination<br>0.466_†_<br>0.460_†_<br>0.398_†_<br>0.445_†_<br>0.371_†_<br>0.462_†_<br>0.091<br>Redundancy<br>0.464_†_<br>0.463_†_<br>0.396_†_<br>0.455_†_<br>0.379_†_<br>0.552_†_<br>0.242<br>Semantic Coverage-Step<br>-0.030<br>0.002<br>0.017<br>-0.154<br>-0.237<br>-0.084<br>-0.298<br>Missing Step<br>0.484_†_<br>0.419_†_<br>0.509_†_<br>0.364_†_<br>0.357_†_<br>0.325<br>0.333<br>Reasoning Alignment<br>0.613_†_<br>0.590_†_<br>0.570_†_<br>0.573_†_<br>0.464_†_<br>0.529_†_<br>0.460<br>Commonsense<br>0.411_†_<br>0.348_†_<br>0.424_†_<br>0.289_†_<br>0.264<br>0.246<br>0.404<br>ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings<br>Hallucination<br>0.702_†_<br>0.621_†_<br>0.704_†_<br>0.630_†_<br>**0.629**_†_<br>0.422_†_<br>0.818<br>Redundancy<br>0.453_†_<br>0.411_†_<br>0.451_†_<br>0.417_†_<br>0.406_†_<br>0.353_†_<br>0.505<br>Semantic Coverage-Step<br>-0.066<br>-0.062<br>0.005<br>-0.042<br>-0.148<br>0.329_†_<br>-0.576<br>Missing Step<br>0.501_†_<br>0.449_†_<br>0.520_†_<br>0.450_†_<br>0.402_†_<br>0.224<br>0.566<br>Reasoning Alignment<br>0.583_†_<br>0.523_†_<br>0.559_†_<br>0.558_†_<br>0.380_†_<br>0.381_†_<br>0.571<br>Commonsense<br>0.626_†_<br>0.529_†_<br>0.657_†_<br>0.594_†_<br>0.578_†_<br>0.305<br>**0.828**<br>ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings<br>Hallucination<br>**0.806**_†_<br>**0.707**_†_<br>**0.731**_†_<br>**0.729**_†_<br>0.625<br>_†_<br>0.357_†_<br>0.611<br>Redundancy<br>0.734_†_<br>0.661<br>_†_<br>0.665_†_<br>0.699_†_<br>0.598_†_<br>0.463_†_<br>0.717<br>Semantic Coverage-Step<br>0.063<br>0.063<br>0.127<br>0.057<br>0.216<br>0.300<br>-0.111<br>Missing Step<br>0.722_†_<br>0.631_†_<br>0.685_†_<br>0.691_†_<br>0.517_†_<br>0.353_†_<br>0.808<br>Reasoning Alignment<br>0.712_†_<br>0.628_†_<br>0.654_†_<br>0.711_†_<br>0.487_†_<br>0.435_†_<br>0.717<br>Commonsense<br>0.780<br>_†_<br>0.659_†_<br>0.721<br>_†_<br>0.727<br>_†_<br>0.486_†_<br>0.324<br>0.813|0.439_†_<br>0.624_†_<br>0.319<br>0.519_†_<br>0.459_†_<br>0.699_†_<br>0.472<br>0.538_†_<br>0.023<br>0.335<br>0.064<br>0.002<br>0.438_†_<br>0.685_†_<br>0.207<br>0.590_†_<br>0.587_†_<br>**0.725**_†_<br>0.446<br>0.711_†_<br>0.358_†_<br>0.677_†_<br>0.367<br>0.494_†_<br>0.644_†_<br>0.340<br>0.064<br>0.846_†_<br>0.468_†_<br>0.195<br>0.204<br>0.570_†_<br>0.006<br>0.232<br>0.291<br>-0.085<br>0.488_†_<br>0.503_†_<br>0.186<br>0.572_†_<br>0.569_†_<br>0.418_†_<br>0.296<br>0.716_†_<br>0.574_†_<br>0.421_†_<br>-0.125<br>0.758_†_<br>**0.712**_†_<br>0.575_†_<br>-0.105<br>**0.896**_†_<br>0.664_†_<br>0.517_†_<br>-0.061<br>0.840_†_<br>0.161<br>-0.205<br>**0.492**<br>0.023<br>0.655_†_<br>0.482_†_<br>0.003<br>0.821_†_<br>0.655_†_<br>0.603_†_<br>0.161<br>0.848_†_<br>0.691<br>_†_<br>0.510_†_<br>-0.217<br>0.887<br>_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence embeddings<br>Semantic Coverage-Chain<br>0.404_†_<br>0.364_†_<br>0.452_†_<br>0.324_†_<br>0.214<br>0.381_†_<br>0.601<br>ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings<br>Semantic Coverage-Chain<br>0.170_†_<br>0.171_†_<br>0.189_†_<br>0.203_†_<br>-0.055<br>0.205<br>0.237<br>ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings<br>Semantic Coverage-Chain<br>0.411_†_<br>0.375_†_<br>0.422_†_<br>0.460_†_<br>0.301<br>0.439_†_<br>0.742|0.350_†_<br>0.172<br>-0.196<br>0.444_†_<br>0.186_†_<br>-0.014<br>0.041<br>0.243_†_<br>0.381_†_<br>0.275<br>0.148<br>0.506_†_|



51 

Published as a conference paper at ICLR 2023 

Table 39: Somers’ _D_ correlation of all metrics on **ESNLI human judged** dataset analyzing step-by-step reasoning on overall chain and step-level perspectives. All metrics are measured **reference-based** on ( _**s** ,_ _**h**_ ). The highest correlation overall for each aspect on each dataset is in **bold** , second best is underlined. Correlation scores with _p_ - _value <_ 0 _._ 05 are marked with _†_ . (Continued from _§_ 6, more details in App. H.2) 

||QUAL|COH|COMMON|FACT|HALL|RED|REP|LOGIC|MATH|GRAM|MISS|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Rouge-1|0.255_†_|-0.234|-0.506_†_|0.072|0.359|-0.078|-|-0.310_†_|-0.054|-0.487|0.662_†_|
|Rouge-2|0.223_†_|-0.189|-0.629_†_|0.103|0.429|-0.081|-|-0.344_†_|-0.149|-0.493|0.568_†_|
|Rouge-L|0.227_†_|-0.177|-0.647_†_|0.084|0.383|-0.089|-|-0.351_†_|-0.500|-0.493|0.628_†_|
|BLEURT|0.221_†_|-0.170|-0.197|0.078|-0.090|-0.108|-|-0.202|0.216|-0.447|0.611_†_|
|BERTScore|0.362_†_|-0.036|-0.281|0.182|0.269|0.143|-|-0.153|0.068|-0.369|0.661_†_|
|BARTScore|0.121|-0.304|-0.200|0.003|0.007|-0.339_†_|-|-0.322_†_|0.338|-0.478|0.513_†_|
|BARTScore+|0.129|-0.113|-0.306|0.164|0.048|-0.036|-|-0.367_†_|-0.095|-0.378|0.341_†_|
|BARTScore-P|0.096|-0.037|-0.244|0.188|-0.017|0.028|-|-0.222|-0.946|-0.244|0.173|
|PRISM|0.314_†_|0.032|-0.386|0.086|**0.438**|0.063|-|-0.131|-0.378|-0.328|**0.684**_†_|
|CTC-Relevance|0.072|**0.331**_†_|0.383|0.034|0.352|-0.001|-|**0.400**_†_|0.405|**0.231**|0.033|
|CTC-Consistency|-0.051|0.150|-0.078|-0.130|0.083|0.105|-|0.085|0.676|0.056|0.006|
|**ROSCOE Metrics(reference-based on**||(**_s_**_,_**_h_**)**)**||||||||||
|ROSCOE-SAwith fnetuned_sup-simcse-roberta-base_sentence||||embeddings||||||||
|Hallucination|0.156_†_|0.152|0.461|-0.011|-0.021|**0.160**|-|0.132|0.743|-0.275|0.170|
|Redundancy|0.142|0.234|**0.553**_†_|0.046|0.283|0.119|-|0.159|0.500|-0.253|0.145|
|Semantic Coverage-Step|0.153_†_|-0.094|0.172|-0.192|-0.241|-0.065|-|-0.020|-0.730|0.086|0.327_†_|
|Missing Step|0.234_†_|-0.197|-0.375|0.239|-0.010|-0.275|-|-0.226|0.527|-0.400|0.558_†_|
|Reasoning Alignment|0.278_†_|-0.062|-0.003|0.072|0.100|-0.049|-|-0.047|-0.108|-0.433|0.495_†_|
|Commonsense|0.142|-0.094|-0.353|0.213|0.207|-0.184|-|-0.148|0.676|-0.447|0.368_†_|
|ROSCOE-SAwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Hallucination|0.174_†_|0.094|0.350|0.066|-0.210|-0.021|-|0.181|0.554|-0.089|0.141|
|Redundancy|0.219_†_|0.227|0.531|0.153|0.031|0.116|-|0.293|0.405|-0.133|0.133|
|Semantic Coverage-Step|0.185_†_|-0.108|-0.039|0.099|-0.490|-0.229|-|-0.185|-0.122|0.083|0.350_†_|
|Missing Step|0.303_†_|-0.173|-0.603_†_|-0.017|0.159|-0.031|-|-0.180|0.689|-0.650_†_|0.679<br>_†_|
|Reasoning Alignment|**0.428**_†_|-0.028|-0.128|0.294<br>_†_|0.121|0.095|-|-0.066|-0.351|-0.547_†_|0.657_†_|
|Commonsense|0.211_†_|-0.029|-0.567_†_|0.062|0.324|-0.016|-|-0.071|**0.946**|-0.564|0.489_†_|
|ROSCOE-SAwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Hallucination|0.190_†_|0.086|-0.311|0.171|-0.138|-0.050|-|0.001|-0.149|-0.322|0.272_†_|
|Redundancy|0.166_†_|0.177|-0.208|0.249|-0.038|-0.007|-|0.023|-0.405|-0.331|0.235_†_|
|Semantic Coverage-Step|0.196_†_|-0.235|-0.058|0.167|-0.341|-0.187|-|-0.153|-0.338|-0.078|0.425_†_|
|Missing Step|0.307_†_|-0.165|-0.508_†_|0.143|0.124|-0.105|-|-0.139|0.405|-0.558_†_|0.623_†_|
|Reasoning Alignment|0.374<br>_†_|-0.049|-0.406|**0.317**_†_|0.114|0.024|-|-0.148|-0.608|-0.603_†_|0.642_†_|
|Commonsense|0.197_†_|0.004|-0.467|0.087|0.352|0.021|-|-0.047|0.919|-0.464|0.389_†_|
|ROSCOE-SSwith fnetuned_sup-simcse-roberta-base_sentence||||embeddings||||||||
|Semantic Coverage-Chain|0.152_†_|-0.185|-0.356|0.266|-0.021|-0.311|-|-0.255_†_|0.500|0.022|0.287_†_|
|ROSCOE-SSwith_all-mpnet-base-v2_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.320_†_|-0.213|-0.203|0.045|-0.066|0.013|-|-0.309_†_|0.446|-0.517|0.679<br>_†_|
|ROSCOE-SSwith_sup-simcse-roberta-base_sentence embeddings||||||||||||
|Semantic Coverage-Chain|0.339_†_|-0.196|-0.378|0.269|0.072|-0.088|-|-0.148|0.324|-0.539_†_|0.643_†_|



52 

Published as a conference paper at ICLR 2023 

Table 40: ROSCOE performance analysis on examples from **Human Judged** datasets. Errors are highlighted in red. (Cont. from $ 6) 

|**Dataset**|**Context**|**Reasoning chain**|**Score type**:|**Comment**|**Comment**||
|---|---|---|---|---|---|---|
||||**Score value**;||||
||||**Errors**||||
|DROP|Over the next year, however, the Polish forces|6000-20000=14000The answer is yes.|Info-Step:|Common Sense Er-|||
||were subject to attrition, as the Sejm again||0.514;|ror: Model should|||
||refused to raise taxes and pay the army, result-||COMMON,|subtract||smaller|
||ing in mass desertions of unpaid soldiery. The||MATH|value (6,000) from|||
||Polish problems were further aggravated by|||bigger (20,000) to|||
||the incompetent leadership of hetman Michał|||answer|the ques-||
||Kazimierz Pac, who obstructed Sobieski’s|||tion.|Arithmetic||
||leadership, while the Ottomans continued to|||error: 6,000 minus|||
||receive reinforcements. Nonetheless in 1674|||20,000 is -14,000.|||
||the Commonwealth resumed the offensive,||||||
||taking advantage of a new Muscovy-Ottoman||||||
||confict that year, and the Polish-Ottoman||||||
||war remained undecided. Sobieski’s force||||||
||of 6,000 defeated 20,000 Turks and Tatars un-||||||
||der Ibrahim Shyshman in the battle of Lwow||||||
||in August 1675.<br>Even after the Battle of||||||
||Trembowla, the Sejm still refused his pleas||||||
||for more funds and a larger army. In 1676,||||||
||after Sobieski’s 16,000 withstood the two-||||||
||week siege of ˙Zurawno, by 100,000 men un-||||||
||der Ibrahim Pasha, a new peace treaty was||||||
||signed, the Treaty of ˙Zurawno. The peace||||||
||treaty partially reversing those from Buczacz:||||||
||the Ottomans kept approximately two thirds||||||
||of the territories they gained in 1672, and||||||
||the Commonwealth no longer was obliged to||||||
||pay any kind of tribute to the Empire; a large||||||
||number of Polish prisoners were released by||||||
||the Ottomans. How many was the difference||||||
||beween Sobieski’s force and the Turks and||||||
||Tatars? Claim: 14000. Is the Claim supported||||||
||bythe Situation?||||||
|DROP|The frst Azov campaign began in the spring|The frst Azov campaign began in the|Info-Step:|Redundancy:|||
||of 1695. Peter the Great ordered his army to|spring of 1695. Peter the Great ordered|0.966, RED|Model||repeats|
||advance towards Azov. The army comprised|his army to advance towards Azov. The||sentences||from|
||crack regiments and the Don Cossacks and|army comprised crack regiments and||context that are not|||
||was divided into three units under the com-|the Don Cossacks and was divided into||directly|relevant to||
||mand of Franz Lefort, Patrick Gordon and|three units under the command of Franz||the question and|||
||Avtonom Golovin. Supplies were shipped|Lefort, Patrick Gordon and Avtonom||can<br>be||dropped.|
||down the Don from Voronezh.In 1693 the Ot-|Golovin. Supplies were shipped down||Info-Step||score is|
||toman garrison of the fortress was 3,656, of|the Don from Voronezh.In 1693 the||however high, be-|||
||whom 2,272 were Janissaries.Between June|Ottoman garrison of the fortress was||cause it measures|||
||27-July 5 the Russians blocked Azov from|3,656, of whom 2,272 were Janis-||two-way similarity|||
||land but could not control the river and pre-|saries.Between June 27-July 5 the Rus-||between||context|
||vent resupply. After two unsuccessful attacks|sians blocked Azov from land but could||and|reasoning||
||on August 5 and September 25, the siege was|not control the river and prevent resup-||chain, thus being|||
||lifted on October 1. Another Russian army|ply. After two unsuccessful attacks on||not informative.|||
||under the command of Boris Sheremetev set|August 5 and September 25, the siege|||||
||out for the lower reaches of the Dnieper to|was lifted on October 1. Another Rus-|||||
||take the Ottoman forts there. The main fort|sian army under the command of Boris|||||
||at Gazi-Kerman was taken when its powder|Sheremetev set out for the lower reaches|||||
||magazine blew up, as well as Islam-Kerman,|of the Dnieper to take the Ottoman forts|||||
||Tagan and Tavan, but the Russians were not|there. The main fort at Gazi-Kerman|||||
||able to hold the area and withdrew most of|was taken when its powder magazine|||||
||their forces. By the Treaty of Constantinople|blew up, as well as Islam-Kerman.|||||
||the remaining Russians were withdrawn and||||||
||the lower Dnieper was declared a demilita-||||||
||rized zone. What happened frst: Russians||||||
||blocked Azov or Treaty of Constantinople?||||||
||Claim: Russians blocked Azov. Is the Claim||||||
||supported by the Situation?||||||



53 

Published as a conference paper at ICLR 2023 

|**Dataset**|**Context**|**Reasoning chain**|**Score type**:|**Comment**|
|---|---|---|---|---|
||||**Score value**;||
||||**Errors**||
|GSM8K|John has 3 boxes. Each box is 5 inches by|Each box is 5*6*4=«5*6*4=120»120|Self-|There<br>are<br>only<br>2|
||6 inches by 4 inches. The walls are 1 inch|cubic inches.<br>So they have a to-|Consistency:|_Repetition_<br>errors<br>in|
||thick. What is the total inner volume of all|tal of 120*3=«120*3=360»360 cu-|0.361, REP|the dataset, both have|
||3 boxes?|bic inches.<br>So the total vol-||_Self-Consistency_ score|
|||ume is 360*1=«360*1=360»360 cu-||below 0_._362.<br>Most|
|||bic inches. Answer: 360||of<br>the<br>other<br>chains|
|||||have<br>higher<br>scores,|
|||||thus Somers’ _D_ cor-|
|||||relation<br>score<br>for|
|||||_Self-Consistency_<br>and|
|||||_Repetition_ is 0_._98 on|
|||||GSM8K dataset.|
|GSM8K|Mary is an avid gardener. Yesterday, she|Mary has 2 potted plants on each of|Repetition-|This is an example of a|
||received 18 new potted plants from her fa-|the 40 window ledges, for a total of|Step: 0.001;|chain that was scored as|
||vorite plant nursery. She already has 2 pot-|2*40 = «2*40=80»80 potted plants.|No error|low as chains that actu-|
||ted plants on each of the 40 window ledges|She has 18 new potted plants, for a||ally have repetitions, be-|
||of her large country home. Feeling gener-|total of 80+18 = «80+18=98»98 pot-||cause its steps do sound|
||ous, she has decided that she will give 1|ted plants. She is giving away 1 pot-||similar.<br>Two chains|
||potted plant from each ledge to friends and|ted plant from each of the 40 win-||with actual repetition er-|
||family tomorrow. How many potted plants|dow ledges, for a total of 1*40 =||rors were scored below|
||will Mary remain with?|«1*40=40»40 plants. She will be re-||0_._001.<br>In total, 3 out|
|||maining with 98-40 = «98-40=58»58||of 198 chains without|
|||potted plants. Answer: 58||repetition error where|
|||||scored as low as chains|
|||||with repetition error, re-|
|||||sulting in high 0_._869|
|||||Somers’ D score.|
|GSM8K|Rory orders 2 subs for $7.50 each, 2 bags of|The order is $7.50 for the sub, $1.50|Faithfulness-|There was no corre-|
||chips for $1.50 each and 2 cookies for $1.00|for chips and $1.00 for cookies so|Token:|lation<br>(Somers’<br>D|
||each for delivery. There’s a 20% delivery|the total order is 7.50+1.50+1.00|0.801; FACT|score<br>0_._014)<br>found|
||fee added at check out and she wants to add|=<br>$«7.50+1.50+1.00=10.0»10.00.||between FACT error|
||a $5.00 tip. What will her delivery order|There’s a 20% delivery fee added||and<br>_Faithfulness-Step_|
||cost?|at check out so that’s 10*.20 =||score.<br>The range of|
|||$«10*.20=2.0»2.00.<br>The order is||scores where the error|
|||$10.00 and there’s a $2.00 delivery||was and was not present|
|||fee so 10+2 = $«10+2=12.00»12.00.||is the same(0_._60_,_0_._81).|
|||She also wants to add a $5.00 tip||We provide an example|
|||which will make the order 12+5 =||of the chain with FACT|
|||$«12+5=17.00»17.00. Answer: 17||error that was scored|
|||||amongthe highest.|
|ESNLI|People standing in line of a beer and curry|The people are standing in line for|Source-|Although there is a|
||line. Claim: People standing in line. Is the|beer and curry, not just standing in|Consistency:|clear<br>contradiction|
||Claim supported by the Situation?|line. The answer is No.|0.976; FACT|with a context,<br>this|
|||||chains<br>gets<br>high|
|||||_Source-Consistency_|
|||||score.<br>Overall, there|
|||||is a large number of|
|||||false-positives<br>(with|
|||||scores as high as0_._98),|
|||||and<br>false-negatives|
|||||(with scores as low|
|||||as<br>0_._0)<br>resulting<br>in|
|||||moderate<br>correlation|
|||||score of0_._425.|



54 

Published as a conference paper at ICLR 2023 

|**Dataset**|**Context**|**Reasoning chain**|**Score type**:|**Comment**|
|---|---|---|---|---|
||||**Score value**;||
||||**Errors**||
|ESNLI|Two women are embracing while holding to|Two women are embracing while|Self-|_Self-Consistency_ mea-|
||go packages. Claim: The men are fghting|holding to go packages means they|Consistency:|sures if steps contra-|
||outside a deli. Is the Claim supported by|are friends or family. They are not|0.079;|dict each other. LOGIC|
||the Situation?|fghting. The answer is No.|LOGIC|identifes if there is|
|||||a logical deduction er-|
|||||ror _within_ one of the|
|||||steps.<br>Nevertheless,|
|||||_Self-Consistency_is able|
|||||to catch some logical|
|||||deduction errors result-|
|||||ing in moderate corre-|
|||||lation, since contradic-|
|||||tions within a step often|
|||||lead to logical errors be-|
|||||tween steps.|
|SemEval|I spray some Pam into a frying pan. I chop|The answer is incorrect, because there|_Faithfulness-_|This chain was given a|
||up small pieces of ham and put them into|is no mention of a family. The only|_Step_: 0.776;|score lower that 25% of|
||the frying pan. I turn the stove on low heat.|people mentioned are the person cook-|HALL|_Faithfulness-Step_scores|
||I crack two eggs into a bowl. I add a little|ing the omelet and the nurse.||within this dataset. Sec-|
||bit of water and whisk the eggs until they|||ond step is misaligned|
||are completely mixed up. I pour the egg|||with the context, con-|
||mixture into the pan. I turn the heat up to|||taining irrelevant infor-|
||medium. I put a lid on top of the pan and|||mation, that resulted in|
||continue to cook the mixture. After a few|||score punishment.|
||minutes, I check to see if all the liquid on||||
||top of the egg has cooked so it isn’t runny||||
||anymore. If it is completely cooked, I add||||
||a little bit of shredded cheese on top of the||||
||egg mixture. I cook for a few minutes until||||
||the cheese melts. I get a spatula and fip half||||
||of the egg on top of the other half. Then my||||
||omelet is cooked! Who all ate the break-||||
||fast? Claim: The family.||||
|SemEval|I had bought my dad a tie for father’s day.|The answer is incorrect, because it|_Faithfulness-_|_Faithfulness-Step_score|
||I’m sure I could have given it to him like|was Father’s Day.|_Step_: 0.635;|is low, although the|
||that, but I thought a nice touch would have||No error|reasoning<br>is<br>correct.|
||been to wrap it. I couldn’t wrap the tie like|||Model failed to produce|
||that, so I bought a small cardboard box. It|||higher score because|
||was a bit narrow and the perfect size for a|||cosine similarity-based|
||tie. I put the tie in the box and taped it shut.|||alignment scores are|
||I had bought some blue and white wrapping|||low.|
||paper and some tape, so I set that besides||||
||the box. I lay the box on top of the paper||||
||and folded it over. I cut the excess of the||||
||paper off and taped the paper to the box. I||||
||folded all the sides of the paper to tuck it in||||
||to a neat package. When I was done, I put||||
||a nametag on it. I topped off the gift with a||||
||small blue ribbon bow. What is the reason||||
||for the present? Claim: It was Christmas.||||



55 

Published as a conference paper at ICLR 2023 

Table 41: ROSCOE performance analysis on examples from **Diagnostic** datasets. Perturbations are highlighted in red. (Cont. from $ 6) 

|Cont. from $6)||||||||
|---|---|---|---|---|---|---|---|
|**Dataset**|**Context**|**Reasoning chain**|**Score**|||**type**:|**Comment**|
||||**Score **||**value**;|||
||||**Perturbation**|||||
|ProofWriter|Charlie is furry. Erin is furry. Erin is|The bear visits the cow.|_Faithfulness-_||||_Faithfulness-Step_<br>scores<br>vary|
||green. Erin is round. Fiona is blue. Fiona|Erin is round.|_Step_:||0.602;||from 0.5 to 1.0. Metric is able to|
||is furry. Gary is smart. White things are||Halucination||||catch a hallucination and gives|
||round. If something is furry and blue then||||||the chain a low score.|
||it is white. If something is blue and round|||||||
||then it is red. All furry things are blue. If|||||||
||Charlie is red then Charlie is green. Is the|||||||
||following True, False, or Unknown? Erin|||||||
||is round.|||||||
|ProofWriter|Anne is cold. Anne is not young. Bob is|Gary is red.<br>All red|_Faithfulness-_||||_Faithfulness-Step_<br>scores<br>for|
||green. Fiona is cold. Fiona is rough. Gary|things are nice. Gary|_Step_:||0.740;||chains<br>without<br>perturbations|
||is red. Gary is young. Green, rough things|is nice.|No|perturba-|||vary from 0.740 to 0.990. Even|
||are red. If Gary is cold and Gary is red||tion||||though frst two sentences are|
||then Gary is green. If something is nice||||||aligned with the context, the|
||then it is rough. Cold, rough things are||||||last sentence in the chain lowers|
||kind. If something is cold and nice then||||||overall average.|
||it is young. If something is kind and cold|||||||
||then it is green. All red things are nice. Is|||||||
||the following True, False, or Unknown?|||||||
||Garyis not nice.|||||||
|EQASC|What insect goes through a brood process|Most pupal steps go|_Grammar_:||||In this subset, _Grammar_ scores|
||which are the larval and pupal steps com-|through larval stages|0.258||;|Se-|vary between0_._258and0_._996. To|
||bined. Answer: An ant has a brood which|after hatching. Brood|mantic||Error||score grammatical correctness we|
||is the larval and pupal process of the ant|are the larval and pupal|||||used pre-trained model, that also|
||after hatching|stages of the ant.|||||punishes chains that sound unnat-|
||||||||ural, like in this example. As a re-|
||||||||sult, we see moderate correlations|
||||||||between_Grammar_scores and Se-|
||||||||mantic Errorperturbation.|
|EQASC|What uses a beak for catching prey? An-|A beak is used for|_Grammar_:||||The _Grammar_ score correctly|
||swer: a beak is used for catching prey by|catching prey by some|0.529||;|No|identifes grammatical issue. Al-|
||eagles.|birds. Eagles are a big|perturbation||||though no perturbation was not|
|||bird.|||||added in this chain, the reference|
||||||||itself contained an error.|
|EntailmentBank|A different moon phase occurs approxi-|A different moon phase|_Info-Chain_:||||In this subset, _Info-Chain_scores|
||mately once per week. The frst quarter|doesn’t occurapproxi-|0.542;||Negate||calculated using fnetuned _sup-_|
||phase of the moon occurs after the new|mately once per week.|step||||_simcse-roberta-base_belong to the|
||moon. For a science activity, Mia had to|The frst quarter phase|||||interval(0_._530; 0_._999), with92%|
||draw a picture of the Moon each night. If|of the moon occurs|||||of perturbed instances having|
||Mia drew a new moon, what type of moon|after the new moon.|||||scores less then0_._92, and95%of|
||did Mia draw about one week later? An-|Therefore the frst quar-|||||non-perturbed instance that were|
||swer: frst quarter|ter phase of the moon|||||scored higher. Even though_Info-_|
|||will occur one week af-|||||_Chain_embeds chain as a whole, it|
|||ter the new moon|||||turned out to be very sensitive to|
||||||||negation perturbations, especially|
||||||||when the model is fnetuned, re-|
||||||||sulting in high Somers’ D correla-|
||||||||tion score of0_._955.|



56 

Published as a conference paper at ICLR 2023 

|**Dataset**|**Context**|**Reasoning chain**|**Score**|**type**:|**Comment**|
|---|---|---|---|---|---|
||||**Score **|**value**;||
||||**Perturbation**|||
|EntailmentBank|Hunting decreases animal populations. If|Hunting decreases an-|_Info-Chain_:||In this case, a negation occurs in|
||a population decreases to zero then that|imal populations. If a|0.983; Negate||the last step of the chain, which|
||organism is extinct. An animal is a kind|population decreases to|step||represents a conclusion and does|
||of organism. Many animals are still being|zero then that organism|||not overlap in wording with the|
||hunted for their fur. Because of this, many|is extinct. An animal|||context, and goes undetected by|
||of these animals are in danger of. Answer:|is a kind of organism.|||the metric.|
||extinction.|Therefore if hunting de-||||
|||creases the animal pop-||||
|||ulation to zero, then the||||
|||animal will not be ex-||||
|||tinct.||||
|MATH|What is1(2235423523)?|Oneraiseto any power|_Grammar_:||Pre-trained model, used to de-|
|||beone , so our answer|0.053 ;|Gram-|tect grammatical errors, gives low|
|||be 1|matical Error||scores for simple chains. In this|
||||||subset,_Grammar_scores belong to|
||||||the interval (0_._053;0_._990).|
|MATH|Two fair eight-sided dice have their faces|To_the expected value|_Grammar_:||This is an example of a chain|
||numbered from 1 to 8. What is the ex-|of a double roll , we|0.982;|Gram-|where the classifer model failed|
||pected value of the sum of the rolls of|can simply add the ex-|matical Error||to<br>catch<br>the<br>grammar<br>error|
||both dice?|pected values of the in-|||(dropped verb "fnd").<br>Even|
|||dividual rolls , giving|||though the model is able to|
|||4_._5 + 4_._5 = 9<br>.|||identify some grammatical errors,|
||||||_Grammar_score distributions over|
||||||perturbed and unperturbed data|
||||||overlap signifcantly, with mean|
||||||0_._909 and variance 0_._005 for un-|
||||||perturbed subset, and0_._854mean|
||||||and0_._014for perturbed. As a re-|
||||||sult, the correlation score is0_._392,|
||||||which isquite low.|
|ASDIV|The puppet show begins at 2:00 p.m. It|(2<br>(hour)<br>+<br>60|_Info-Step_:||In this subset,_Info-Step_varies be-|
||lasts 1 hour and 30 minutes. What time|(min+hour) / 1 (hour)|0.005; Shuffe||tween 0_._005 and 0_._796, fagging|
||does it end?|/ 60 (min*hour) / 30|Operations||most of the chains with shuffed|
|||(min))*60 (min/hour)|||steps. Scores are distributed with|
|||= 3 r30; hh:mm = 3:30|||mean 0_._356 and variance 0_._056|
|||p.m.|||for perturbed, and0_._705and vari-|
||||||ance0_._004for unperturbed chains.|
||||||Resulting Somers’ D correlation|
||||||score is0_._879.|
|ASDIV|I have 28 cents in my pocket. How coins|28/25=1 r3;<br>3+1=3;|_Info-Step_:||This is an example of a chain|
||could I have on minimization?|1/3=4|0.705; Shuffe||where perturbation in operations|
||||Operations||got higher scores bythe model.|
|ASDIV|Mrs. Hilt has $10. She spends $3 on a toy|10-(3+2)=5|_Info-Step_:||This is an example of a chain that|
||truck and $2 on a pencil case. How much||0.561;|No|was scored on a lower spectrum by|
||money does she have left?||perturbation||the_Info-Step_spectrum, although|
||||||it had no perturbation or other|
||||||type of error included.|



57 

