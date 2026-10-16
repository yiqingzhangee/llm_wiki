# **INSTRUCTRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales** 

**Zhepei Wei Wei-Lin Chen Yu Meng** Department of Computer Science University of Virginia 

```
{zhepei.wei,tuy8sy,yumeng5}@virginia.edu
```

## **Abstract** 

Retrieval-augmented generation (RAG) has shown promising potential to enhance the accuracy and factuality of language models (LMs). However, imperfect retrievers or noisy corpora can introduce misleading or even erroneous information to the retrieved contents, posing a significant challenge to the generation quality. Existing RAG methods typically address this challenge by directly predicting final answers despite potentially noisy inputs, resulting in an _implicit_ denoising process that is difficult to interpret and verify. On the other hand, the acquisition of explicit denoising supervision is often costly, involving significant human efforts. In this work, we propose INSTRUCTRAG, where LMs _explicitly_ learn the denoising process through self-synthesized rationales — First, we instruct the LM to explain how the ground-truth answer is derived from retrieved documents. Then, these rationales can be used either as demonstrations for in-context learning of explicit denoising or as supervised fine-tuning data to train the model. Compared to standard RAG approaches, INSTRUCTRAG requires no additional supervision, allows for easier verification of the predicted answers, and effectively improves generation accuracy. Experiments show INSTRUCTRAG consistently outperforms existing RAG methods in both training-free and trainable scenarios, achieving a relative improvement of 8.3% over the best baseline method on average across five knowledge-intensive benchmarks. Extensive analysis indicates that INSTRUCTRAG scales well with increased numbers of retrieved documents and consistently exhibits robust denoising ability even in out-of-domain datasets, demonstrating strong generalizability.[1] 

## **1 Introduction** 

While large language models (LMs) have demonstrated remarkable text generation abilities [6, 81, 82], they may occasionally produce factually incorrect contents [15, 25, 31, 79, 99, 110], particularly when the task at hand requires the most current information or out-of-domain knowledge not adequately represented in the pre-training corpus [35, 75, 106, 113]. This limitation significantly hinders the reliable deployment of LMs in high-stakes domains where factuality is crucial [60, 76, 93, 94]. 

In light of this, retrieval-augmented generation (RAG) [3, 23, 29, 42, 52] has been introduced to enhance the generation accuracy of LMs in knowledge-intensive tasks by leveraging the most upto-date information and specialized knowledge from external sources [40, 83, 102, 117]. However, the retrieved contents are typically mixed with irrelevant or even erroneous information due to the absence of perfect retrieval solutions [27, 39, 43, 44, 74] and the presence of noisy data in the retrieval corpus [28, 53, 105], posing a long-standing challenge to almost all RAG systems. Typically, vanilla RAG approaches address this issue _implicitly_ by training LMs to directly predict correct answers 

> 1 Our code is available at `https://github.com/weizhepei/InstructRAG` . 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales_images/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Vanilla RAG InstructRAG<br>Retrieved  Retrieved<br>documents Questionc q documents Questionc q<br>Learning Objective Learning Objective<br>Doc  d #1 1 p✓ ( a|q, d 1 , . . . , dK ) Doc  d #1 1 p✓ ( r|q, d 1 , . . . , dK )<br>Doc  d #2 2 Language Model Mθ Answerd a Doc  d #2 2 Language Model Mθ Rationale d r<br>· · · · · ·<br> Input:  <retrieved documents> Based on your knowledge and  Input:  <same as in vanilla RAG><br>Doc #N dK  the provided information, answer the question: <question> Doc #N dK  Output: The second document mentions… None of the other<br> Output: <answer>  documents are relevant… Therefore, the answer is <answer><br>**----- End of picture text -----**<br>


Figure 1: Comparison between vanilla RAG and our INSTRUCTRAG. In vanilla RAG, the model is tasked to directly predict answers given user queries and potentially noisy retrieved documents, without explicit denoising processes or explanations for how the answer is derived. In contrast, our proposed INSTRUCTRAG generates rationales that explicitly denoise the retrieved documents and justify the predicted answers, enhancing both the generation accuracy and trustworthiness. 

despite potentially noisy inputs. Such latent processes are not only difficult to interpret and verify but also vulnerable to higher noise ratios, especially when the number of retrieved documents is large [11, 13, 56, 91]. On the other hand, obtaining high-quality explicit denoising supervision often requires substantial human efforts, which is time-consuming and costly. 

In this work, we introduce a new RAG framework, INSTRUCTRAG, which enables the LM to _explicitly_ denoise retrieved information and justify its predicted final answers by generating denoising responses ( _i.e._ , _rationales_ ), as illustrated in Figure 1. Compared to vanilla RAG approaches, INSTRUCTRAG does not require any additional supervision, while enjoying improved generation accuracy and trustworthiness. Specifically, our method consists of two steps. First, given a set of question-answer pairs and potentially noisy retrieved documents, we prompt an instruction-tuned LM to synthesize denoising rationales that analyze the documents and articulate how they lead to the ground-truth answers (§ 2.2). Then, these synthetic rationales can be utilized as in-context learning examples or as supervised fine-tuning data, allowing the LM to explicitly learn to denoise retrieved contents (§ 2.3). The effectiveness of INSTRUCTRAG can be attributed to the strong instruction-following ability of LMs [34, 67, 88], a significant feature that still remains underexplored in the context of RAG. We show that such self-synthesized rationales not only provide high-quality explicit denoising supervision for in-domain RAG tasks, but also facilitate superior out-of-domain generalization. This finding underscores how instruction-tuned LMs can synthesize generalizable supervision to effectively overcome the inevitable noise in RAG. 

The main contributions of this work are as follows: (1) We propose INSTRUCTRAG, a simple yet effective RAG framework that allows LMs to explicitly denoise retrieved contents by generating rationales for better verifiability and trustworthiness. (2) INSTRUCTRAG is a self-synthesis method that does not require additional supervision compared to standard RAG methods, and can be seamlessly applied to both in-context learning and supervised fine-tuning settings. (3) INSTRUCTRAG consistently outperforms state-of-the-art RAG approaches, yielding a relative improvement of 8.3% on average compared to the best baseline method across five knowledge-intensive benchmarks. Extensive analysis and ablation studies further confirm the superiority of self-synthesized denoising rationales, and demonstrate INSTRUCTRAG’s robust denoising ability against increased noise ratios and strong task transferability in various training-free and trainable scenarios. 

## **2 Our Method: INSTRUCTRAG** 

In this section, we first introduce our problem setting (§ 2.1) and then present the proposed framework INSTRUCTRAG that enables LMs to explicitly denoise retrieved contents. As shown in Figure 2, our method consists of two steps. First, we prompt an instruction-tuned LM ( _i.e._ , rationale generator _Mϕ_ ) to synthesize rationales that provide denoising supervisions (§ 2.2). These rationales aim to explain how to derive the correct answer from potentially noisy retrieved documents for each training sample. Then, we guide the LM ( _i.e._ , rationale learner _Mθ_ ) to learn explicit denoising by leveraging these rationales as either in-context learning demonstrations or as supervised fine-tuning data (§ 2.3). 

As detailed in Algorithm 1, during the entire process, INSTRUCTRAG does not require any additional supervisions beyond standard RAG methods. By default, we instantiate both _Mϕ_ and _Mθ_ with the same off-the-shelf instruction-tuned model ( _i.e._ , meta-llama/Meta-Llama-3-8B-Instruct), making 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales_images/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Step 1: Rationale Generation for RAG Step 2: Explicit Denoising Learning<br>Denoising Instruction documentsRetrieved  In-Context Learning Supervised Learning<br>Doc  d 1 Language Model     M d  θ Language Model     M d θ<br>Questionc q Language Model     (Frozen) M d φ Doc d 2 (Frozen) (Trainable)<br>Answerd a · · · Prompt : <instruction> Prompt : <instruction><br>Doc  dK as demonstrations as training data<br>Rationale<br>The second document mentions…relevant… Therefore, the answer is <answer>None of the other documents are  Questionc q Rationale d r Doc  d 1 Doc  d 2 · · ·<br>**----- End of picture text -----**<br>


Figure 2: An overview of INSTRUCTRAG. In step one, given the question _q_ , retrieved documents _{d_ 1 _, · · · , dK}_ and ground-truth answer _a_ from the training set, we prompt an instruction-tuned LM ( _i.e._ , rationale generator _Mϕ_ ) to generate rationale _r_ that explains how the answer can be derived from the potentially noisy input. In step two, we utilize the synthesized rationales from the first step to guide the LM ( _i.e._ , rationale learner _Mθ_ ) to explicitly learn denoising of the retrieved documents, either through in-context learning or supervised learning. By default, we use the same model for both _Mϕ_ and _Mθ_ , but they can be instantiated with different models as well (see ablation study § 3.3). 

Table 1: Rationale generation prompt for the _i_ -th training sample. 

## Rationale Generation 

**Input:** Read the following documents relevant to the given question: { _qi_ } Document [1] (Title: _· · ·_ ): {contents of _d_[1] _i_[}] _· · ·_ Please identify documents that are useful to answer the given question: { _qi_ }, and explain how the contents lead to the answer: { _ai_ } {task-specific instruction} **Output:** {rationale _ri_ } 

INSTRUCTRAG a fully _self-synthesis_ method. We also experiment with different instantiations of _Mϕ_ and _Mθ_ and conduct ablation study in both training-free and trainable settings (§ 3.3). For simplicity, we use placeholders to represent omitted instructions in the prompts presented in this section, while the full list of complete prompt templates is provided in Appendix C. 

## **2.1 Problem Setting** 

We adopt the standard RAG setting where the LM _Mθ_ has access to annotated datasets of downstream tasks ( _e.g._ , question-answering task _T_ = _{⟨q, a⟩}_ ), and an external knowledge base with the offthe-shelf retriever _R_ for retrieval. Different from previous works [3, 105] which leverage additional supervisions from GPT-3 [6] or GPT-4 [1], we assume the model has strictly limited access to the above two information sources. Given a question _q_ , the retriever _R_ returns a set of potentially noisy documents _D_ = _{d_ 1 _, · · · , dK}_ from the external knowledge base. The model is then tasked to predict the correct answer _a_ to the given question _q_ based on _D_ and its own parametric knowledge, denoted as _pθ_ ( _a|q, D_ ). 

Our work focuses on investigating the noise robustness of LMs and developing efficient denoising techniques for RAG. Hence, we directly employ off-the-shelf retrievers instead of training our own, and prepend all retrieved documents to the question as input to the model, without any filtering or re-ranking. This setting is orthogonal to existing research efforts centered on optimizing the retriever or performing adaptive retrieval [3, 86, 102]. 

## **2.2 Rationale Generation via Instruction-Following** 

Recent studies [51, 63, 67] have made encouraging progress in aligning LMs with human preferences and intentions, enabling the synthesis of high-quality data that closely follows user instructions [98]. Inspired by these advances, we propose to leverage the LM’s strong instruction-following ability to generate explicit denoising responses ( _i.e._ , _rationales_ ) for RAG. As shown in Table 1, given a QA pair _⟨qi, ai⟩∈T_ and a set of retrieved documents _{d_[1] _i[,][ · · ·][, d][K] i[}]_[, we prompt an off-the-shelf LM] 

3 

## **Algorithm 1** INSTRUCTRAG 

|**Algorithm 1**INSTRUCTRAG|**Algorithm 1**INSTRUCTRAG|**Algorithm 1**INSTRUCTRAG|
|---|---|---|
|**Require:** Retriever_R_, Rationale Generator_Mϕ_, Rationale Learner_Mθ_, Training data_T_ =_{⟨q, a⟩}_|||
||`/* Training data generation */`||
|1:|**for**each_⟨q, a⟩∈T_ **do**||
|2:|Retrieve_D_ =_{d_1_, · · · , dK} ←R_(_q_)||
|3:|Synthesize denoising rationale_r ←Mϕ_(_q, a, D_)|_▷_Rationale Generation (§2.2)|
|4:|Augment training data_T →T_ + =_{⟨q, r⟩}_||
||`/* Two learning modes */`||
|5:|**if** `LearningMode`==`In-Context Learning`**then**|_▷_INSTRUCTRAG-ICL|
|6:|Sample ICL examples_E_ =_{⟨q, r⟩} ⊆T_ +||
|7:|_r ←Mθ_(_r|q, R_(_q_)_, E_)given inference query_q_|_▷_Detailed in Table10|
|8:|**else if** `LearningMode`==`Fine-Tuning`**then**|_▷_INSTRUCTRAG-FT|
|9:|Fine-tune_Mθ_ on_T_ + with retrieved documents_{⟨q, r, D⟩}_||
|10:|_r ←Mθ_(_r|q, R_(_q_))given inference query_q_|_▷_Detailed in Table11|
|11:|**return**_r_||



> _Mϕ_ (as the rationale generator) with denoising instructions to produce the corresponding rationale _ri_ that distinguishes useful documents from noisy ones and explains how the contexts lead to the ground-truth answer _ai_ . To ensure the synthetic rationales are aligned with the ground-truth answers, we use a simple substring match to assess their consistency. The consistency ratio on training samples with at least one relevant document containing the ground-truth answer is 98% on average across five benchmarks, supporting the reliability of synthetic rationales as a sanity check. This allows us to effectively augment the standard dataset _T_ = _{⟨q, a⟩} →T_[+] = _{⟨q, r⟩}_ with self-synthesized denoising rationales solely by instructing the LM, without any additional supervision. 

We also validate the necessity of using an LM-based generator ( _i.e._ , _Mϕ_ ) to create the rationales instead of employing simple heuristics — without the generator, rationales can be created in a template-based manner (Table 6), by roughly identifying relevant retrieved documents through simple substring-matching with the ground-truth answer. However, as demonstrated in our ablation study, this approach suffers from semantically inaccurate matching of relevant documents, leading to significant performance degradation. Another advantage of the LM-based generator is that it can produce high-quality rationales even _without referring to the ground-truth answer_ , which only results in a minor performance drop. More detailed analyses on rationale generation design can be found in our ablation study (§ 3.3). 

## **2.3 Learning Denoising Rationales in RAG** 

With the rationale-augmented dataset _T_[+] , it becomes possible to develop a rationale learner _Mθ_ that directly learns explicit denoising for RAG tasks with efficient learning strategies. Next, we introduce two simple yet effective learning methods in the _training-free_ and _trainable_ RAG settings, namely, INSTRUCTRAG-ICL and INSTRUCTRAG-FT. 

**INSTRUCTRAG-ICL** is a training-free instantiation of INSTRUCTRAG where the model learns denoising rationales via in-context learning (ICL). As shown in Table 10, given a test question _q_ and a set of retrieved documents _D_ = _{d_ 1 _, · · · , dK}_ , we first randomly sample _N_ demonstrations _⟨qi, ri⟩∈T_[+] from the rationale-augmented training dataset, and then prompt the model to follow the exemplars and generate rationale _r_ . To save memory and enhance inference efficiency, we only show exemplary questions and their corresponding rationales in such ICL demonstrations. 

**INSTRUCTRAG-FT** is a trainable instantiation of INSTRUCTRAG that learns denoising rationales via supervised fine-tuning (FT) with standard language modeling objective. As defined in Eq. (1), it maximizes the likelihood of rationale _r_ conditioned on question _q_ and retrieved documents _D_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales_images/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales.pdf-0004-08.png)


where _θ_ represents the model parameters. Both the training and inference of INSTRUCTRAG-FT share the same data format. As depicted in Table 11, it takes as input the retrieved documents followed by the question, and outputs the denoising rationale _r_ . 

4 

## **3 Experiments** 

## **3.1 Experimental Setting** 

Table 2: Dataset statistics and retrieval setting. 

|Dataset|Train|Test|Retriever|Top-_K_|Recall@_K_|
|---|---|---|---|---|---|
|PopQA|12,868|1,399|Contriever|5|68.7|
|TriviaQA|78,785|11,313|Contriever|5|73.5|
|Natural Questions|79,168|3,610|DPR|5|68.8|
|ASQA|4,353|948|GTR|5|82.2|
|2WikiMultiHopQA|167,454|12,576|BM25|10|40.7|



**RAG tasks and evaluation metrics.** We extensively validate the effectiveness of INSTRUCTRAG on five knowledge-intensive benchmarks, including PopQA [61], TriviaQA [38], Natural Questions [47], ASQA [78], and 2WikiMultiHopQA [24]. We use Wikipedia corpus as the retrieval source, and test our method with both sparse and dense off-the-shelf retrievers, including BM25 [71], DPR [39], GTR [65] and Contriver [27]. The retrieval quality is measured by Recall@ _K_ , indicating whether the retrieved _K_ documents contain the correct answer. Table 2 shows the detailed dataset statistics. Following standard evaluation settings [3], we adopt the official metric of correctness ( _str-em_ ), citation precision ( _pre_ ) and recall ( _rec_ ) for ASQA [18], and use _accuracy_ for the other tasks, which measures whether the ground-truth answers are included in the model generations [61, 73]. Additionally, we also adopt LLM-as-a-judge for further evaluation (§ 3.4), as the above standard metrics are subject to the limitations of pattern-matching, which cannot accurately handle semantic equivalence. 

**Baselines.** We compare our method with a wide range of RAG baselines under both training-free and trainable settings. Given that state-of-the-art LMs have incorporated a large amount of worldknowledge during the pre-training stage, we also report the performance of a non-retrieval baseline (namely, **vanilla zero-shot prompting** ) for reference. Specifically, the training-free RAG baselines includes: (1) **in-context retrieval-augmented language modeling (RALM)** [70], a prompting method that extends the non-retrieval baseline by presenting the model with retrieved documents; (2) **few-shot demonstration with instruction** , an ICL method using ground-truth question-answer pairs sampled from the training set as demonstration exemplars. 

The trainable RAG baselines include: (1) **vanilla supervised fine-tuning (SFT)** , a supervised method with the training objective of maximizing the data likelihood of ground-truth answer given potentially noisy input; (2) **RetRobust** [105], which fine-tunes the RAG model on a mixture of relevant and irrelevant contexts to make it robust to irrelevant contexts; (3) **Self-RAG** [3], a strong trainable baseline, focusing on adaptive retrieval controlled by special reflection tokens. Both RetRobust and Self-RAG were originally built on Llama-2 [82] with additional supervisions. For example, RetRobust augments the training data for multi-hop reasoning tasks ( _e.g._ , 2WikiMultiHopQA) by prompting GPT-3 to decompose the original query and generate intermediate subqueries, and Self-RAG requires GPT-4 to generate additional reflective tokens to augment training samples. 

For a fair comparison, we re-implement the two methods on Llama-27B and/or Llama-213B with augmented training data released by their authors, and report their performance as the higher one between the original scores and our reproduced results. As our method adopts instruction-tuned Llama-3 as the backbone model, we also train RetRobust and Self-RAG with Llama-3-Instruct8B and optimize their performance through extensive hyper-parameters search. More details on implementation, including training, inference, and prompt design are available in Appendix A and Appendix C. 

## **3.2 Main Result** 

Table 3 shows the overall experimental results, providing a comprehensive comparison between our INSTRUCTRAG and baseline methods in both training-free and trainable RAG settings. 

**Baselines without retrieval.** As shown in the first block, the basic instruction-tuned models (Llama-3Instruct8B and Llama-3-Instruct70B) already achieve notable performance across all five benchmarks, with the 70B model exhibiting a surprisingly competitive performance of 80.6% on the TriviaQA. This observation suggests that the required knowledge for these tasks mostly falls within the LM’s parametric knowledge, probably due to what is known as _data contamination_ ( _i.e._ , the presence of test data of downstream tasks in the pre-training data of LMs) [21, 30, 59]. 

5 

Table 3: Overall results of INSTRUCTRAG and baselines on five knowledge-intensive benchmarks in training-free and trainable RAG settings. We re-implement baselines and report their performance as the higher one between the original scores and our reproduced results. * indicates the results copied from [3] for reference. “–” indicates the results are not reported in the original paper or not applicable ( _e.g._ , some methods cannot produce citations). The best performance is highlighted in _**bold**_ **.** 

||PopQA|TriviaQA|NQ|MultiHopQA||ASQA||
|---|---|---|---|---|---|---|---|
|Method|(acc)|(acc)|(acc)|(acc)|(em)|(pre)|(rec)|
||_Baselines w/o Retrieval_|||||||
|**Vanilla Zero-shot Prompting**||||||||
|ChatGPT*|29.3|74.3|–|–|35.3|–|–|
|Llama-3-Instruct8B|22.8|69.4|46.6|45.6|30.6|–|–|
|Llama-3-Instruct70B|28.9|80.6|57.9|57.5|39.1|–|–|
||_RAG w/o Training_|||||||
|**In-Context RALM**[70]||||||||
|ChatGPT*|50.8|65.7|–|–|40.7|65.1|76.6|
|Llama-3-Instruct8B|62.3|71.4|56.8|43.4|40.0|62.1|66.4|
|Llama-3-Instruct70B|63.8|76.3|60.2|51.2|43.1|62.9|67.6|
|**Few-Shot Demo. w/ Instruction**||||||||
|Llama-3-Instruct8B|63.1|74.2|60.1|45.3|42.6|55.0|64.4|
|Llama-3-Instruct70B|63.9|79.1|62.9|53.9|45.4|49.3|57.1|
|**INSTRUCTRAG-ICL**||||||||
|Llama-3-Instruct8B|64.2|76.8|62.1|50.4|44.7|**70.9**|**74.1**|
|Llama-3-Instruct70B|**65.5**|**81.2**|**66.5**|**57.3**|**47.8**|69.1|71.2|
|||_RAG w/ Training_||||||
|**Vanilla Supervised Fine-tuning**||||||||
|Llama-3-Instruct8B|61.0|73.9|56.6|56.1|43.8|–|–|
|**Self-RAG**[3]||||||||
|Llama-27B|55.8|68.9|42.4|35.9|30.0|66.9|67.8|
|Llama-213B|56.3|70.4|46.4|36.0|31.4|**70.3**|**71.3**|
|Llama-3-Instruct8B|55.8|71.4|42.8|32.9|36.9|69.7|69.7|
|**RetRobust**[105]||||||||
|Llama-213B|–|–|39.6|51.5|–|–|–|
|Llama-3-Instruct8B|56.5|71.5|54.2|54.7|40.5|–|–|
|**INSTRUCTRAG-FT**||||||||
|Llama-3-Instruct8B|**66.2**|**78.5**|**65.7**|**57.2**|**47.6**|65.7|70.5|



**RAG without training.** The second block shows the comparison among training-free RAG methods. In-context RALM and few-shot demonstration with instruction methods generally achieve higher performance than the non-retrieval baseline, highlighting the importance of retrieval for knowledgeintensive tasks. Encouragingly, our INSTRUCTRAG-ICL consistently outperforms all training-free baselines across various metrics, confirming the effectiveness of self-synthesized denoising rationales. Moreover, the boost from 8B to 70B model also demonstrates that INSTRUCTRAG-ICL scales effectively with larger backbone models, validating the generalizability of our method. 

**RAG with training.** As present in the bottom block of Table 3, our INSTRUCTRAG-FT not only surpasses all non-retrieval and training-free baselines across all five benchmarks, but also significantly outperforms trainable RAG baselines on almost every metric. The only exception is in the ASQA task, where our method slightly underperforms Self-RAG in terms of citation ( _i.e._ , _pre_ and _rec_ ). This is because our work primarily focuses on explicit denoising for RAG to improve the correctness of generations, which is measured by _em_ . Despite not being explicitly optimized for citation metrics, our method still achieves competitive citation performance, significantly enhancing both generation accuracy and trustworthiness. Note that RetRobust achieves competitive performance on 2WikiMultiHopQA, which involves multi-hop reasoning. We attribute this to the additional training supervision provided by GPT-3, which enables the model to explicitly generate intermediate sub-queries and sub-answers. Another interesting finding is that Self-RAG consistently exhibits inferior performance compared to vanilla SFT, and even underperforms the training-free in-context RALM baseline across all benchmarks. We speculate the reason might be that these RAG tasks favor more domain-specific knowledge than general knowledge. However, it is challenging for Self-RAG to directly leverage in-domain features from existing training data as it requires GPT-4 to generate reflection tokens on these benchmarks, which is not available in our problem (§ 2.1). 

6 

Table 4: Ablation study on the impact of ground-truth answer, retrieved documents, and model size on rationale generation, and the use of demonstrations during model inference. The results of our default setting in INSTRUCTRAG are underlined. 

||Trainable RAG Setting|Trainable RAG Setting|Training-free RAG Setting|Training-free RAG Setting|
|---|---|---|---|---|
|Method|PopQA|ASQA|PopQA|ASQA|
||_Rationale Generation Design_||||
|with both|66.2|47.6|64.2|44.7|
|w/o ground-truth answer|65.2(_↓_1_._5%)|46.4(_↓_2_._5%)|64.0(_↓_0_._3%)|44.5(_↓_0_._4%)|
|w/o retrieved documents|64.5(_↓_2_._6%)|45.2(_↓_5_._0%)|64.1(_↓_0_._2%)|44.3(_↓_0_._9%)|
||_Model Size of Rationale Generator_||||
|rationale template (no generator)|59.6(_↓_10_._0%)|46.3(_↓_2_._7%)|60.0(_↓_6_._5%)|41.4(_↓_7_._4%)|
|Llama-3-Instruct (8B)|66.2|47.6|64.2|44.7|
|Llama-3-Instruct (70B)|67.0(_↑_1_._2%)|49.1(_↑_3_._2%)|64.8(_↑_0_._9%)|47.9(_↑_7_._1%)|
||_Inference Strategy Comparison_||||
|w/o demonstration|66.2|47.6|63.0(_↓_1_._9%)|43.1(_↓_3_._6%)|
|w/ demonstration|66.1(_↓_0_._2%)|44.7(_↓_6_._1%)|64.2|44.7|



## **3.3 Ablation Study** 

**Providing ground-truth answers and retrieved documents is important for rationale generation.** As depicted in the first block of Table 4, we ablate the rationale generation design from two aspects: (1) _w/o ground-truth answer_ , where the model has no access to the ground-truth answer during rational generation and must predict the answer and explain how it is derived solely based on retrieved documents; (2) _w/o retrieved documents_ , where the model is not provided with any retrieved documents during rational generation, and in this case, it has to explain the given answer based on its own knowledge. Although it is not surprising that our default design consistently outperforms the two ablations, it is encouraging to find that our method still works well even without access to the retrieved documents or ground-truth answers. This finding suggests the great potential of our INSTRUCTRAG to operate in a fully unsupervised manner, which we believe is an exciting direction for future work. 

**Larger rationale generator leads to better results.** The middle block shows how different sizes of rationale generators impact the performance of our method. It is evident that the template-based rationale generation method significantly underperforms our method, highlighting the necessity of rationale generator. This is because the template-based method relies on pattern matching to identify relevant documents containing the ground-truth answer, which only considers lexical similarity while ignoring semantic meaning. The neglect of semantics inevitably introduces noise in templategenerated rationales, making them less effective compared to rationales generated by LMs. Moreover, we also compare two variants of INSTRUCTRAG using Llama-3-Instruct8B and Llama-3-Instruct70B as rationale generators. The results show that the one with a 70B generator consistently outperforms its 8B counterpart in both training-free and trainable settings, indicating that the self-synthesized denoising rationales can provide better supervision when generated by stronger models. 

**Inference with demonstrations should only be applied to INSTRUCTRAG-ICL.** In the bottom block, we study the use of demonstrations during the model inference. While demonstrations play an important role for INSTRUCTRAG-ICL, we find that they actually hurt the performance of INSTRUCTRAG-FT. We attribute this to the fact that INSTRUCTRAG-FT is optimized to directly generate denoising rationales given potentially noisy input, without referring to any demonstrations. Therefore, providing in-context demonstrations for INSTRUCTRAG-FT is redundant and may compromise its capability due to the discrepancy between training and inference. 

## **3.4 Analysis** 

**INSTRUCTRAG-ICL consistently benefits from more demonstrations.** Figure 3a shows the demonstration sensitivity of INSTRUCTRAG-ICL and the few-shot demonstration with instruction baseline. It is interesting to find that the baseline method achieves its best performance with only one demonstration, and presenting more demonstrations actually harms its performance. In contrast, our method consistently improves with the increasing number of demonstrations, confirming the superiority of self-synthesized rationales over plain answers in terms of denoising. 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales_images/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales.pdf-0008-00.png)


**----- Start of picture text -----**<br>
InstructRAG-ICL InstructRAG-ICL InstructRAG-FT<br>Few-shot Demo. w/ Instruction 70 Few-shot Demo. w/ Instruction 60 70 Vanilla SFT 60<br>46<br>65 50 65 50<br>44<br>60 40 60 40<br>42 55 30 55 30<br>0 1 2 3 4 5 1 3 5 7 9 1 3 5 7 9<br># of demonstrations # of retrieved documents # of retrieved documents<br>(a) Training-free RAG setting. (b) Training-free RAG setting. (c) Trainable RAG setting.<br>(PopQA) (PopQA)<br>(ASQA)<br>EM Precision@K Precision@K<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 3: Impact of different number of demonstrations and retrieved documents. (a) Demonstration sensitivity study of INSTRUCTRAG-ICL. (b) Noise robustness study of INSTRUCTRAG-ICL. (c) Noise robustness study of INSTRUCTRAG-FT. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales_images/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales.pdf-0008-02.png)


**----- Start of picture text -----**<br>
50 Baseline InstructRAG Baseline InstructRAG 60 Baseline InstructRAG<br>66<br>40<br>64 40<br>30 62<br>20<br>20 60<br>ID OOD ID OOD ID OOD ID OOD ID OOD ID OOD<br>Training-Free Trainable Training-Free Trainable Training-Free Trainable<br>(a) Short-form to long-form QA. (b) Long-form to short-form QA. (c) Single-hop to multi-hop QA.<br>(ASQA) (PopQA) (MultiHop)<br>EM<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 4: Generalizing INSTRUCTRAG from source domain task to target domain task, where ID and OOD denote in-domain and out-of-domain settings. (a) PopQA (short-form QA task) as source domain and ASQA (long-form QA task) as target domain. (b) ASQA as source domain and PopQA as target domain. (c) PopQA (single-hop QA task) as source domain and 2WikiMultiHopQA (multi-hop QA task) as target domain. We adopt _few-shot demonstration with instruction_ and _vanilla supervised fine-tuning_ as the training-free and trainable baselines. 

**INSTRUCTRAG-ICL and INSTRUCTRAG-FT are robust to increased noise ratios.** Figure 3b and Figure 3c show the generation accuracy of INSTRUCTRAG-ICL and INSTRUCTRAG-FT and the corresponding retrieval precision under an increasing number of retrieved documents. While retrieving more documents provides richer external knowledge to the RAG model, it also introduces more noise and lowers the retrieval precision. As a result, both the training-free and trainable baselines show diminishing improvements or even degrade as the number of documents increases, reflecting their vulnerability to high noisy ratios. In contrast, our INSTRUCTRAG-ICL and INSTRUCTRAGFT are not negatively affected by this increased noise ratio but rather gain further improvement, demonstrating their robust denoising ability. 

**INSTRUCTRAG-ICL and INSTRUCTRAG-FT generalize well to unseen tasks.** Figure 4 demonstrates the generalization ability of our method in both training-free and trainable settings. For the in-domain (ID) method, it directly utilizes target domain demonstrations (in training-free settings) or is trained on the target domain task (in trainable settings). In contrast, the out-of-domain (OOD) method can only learn from demonstrations or training data in the source domain, and have no prior knowledge of the target domain. In this case, the model must leverage the knowledge learned from the source domain task to solve the unseen target domain task. The results show that our method ( _i.e._ , INSTRUCTRAG-ICL and INSTRUCTRAG-FT) consistently outperforms the baselines across various scenarios in both in-domain and out-of-domain settings, demonstrating strong task generalizability. One counter-intuitive finding is that in the scenario of generalizing from long-form to short-form QA task (Figure 4b), the training-free OOD method substantially outperforms its in-domain counterpart. We speculate that the training-free OOD method achieves better performance because it benefits from the demonstrations with long answers from the source domain (ASQA). The reason is that the questions in ASQA are ambiguous and can have multiple interpretations, and ground-truth long answers often address the questions from various perspectives, which can be regarded as a form of chain-of-thought demonstration. 

Furthermore, we also study the generalizability of INSTRUCTRAG to a non-QA knowledge-intensive task such as code generation. As presented in Table 5a, we directly apply INSTRUCTRAG-FT trained on the QA task (PopQA) to solve the unseen code generation task (HumanEval [12]), following the CodeRAG-Bench setup [87]. We evaluate the code generation performance using the standard pass@k metric and compare our method with the off-the-shelf Llama-3-8B-Instruct as the baseline. 

8 

Table 5: (a) Transfer from the QA task (PopQA) to the code generation task (HumanEval). Our method INSTRUCTRAG-FT is fine-tuned only on the QA task and is evaluated for solving the unseen code generation task. We compare it with off-the-shelf LLaMA-3-8B-Instruct in both non-retrieval and retrieval-augmented generation settings, and report the standard evaluation metrics pass@k [12]. (b) Evaluation with GPT-4o as the judge. Compared to pattern-matching based metrics, it allows the judge to consider semantic equivalence and is expected to yield a more fair evaluation. 

|Method<br>pass@1<br>pass@10<br>_Without Retrieval_<br>Llama-3-8B-Instruct<br>58.5<br>64.6<br>INSTRUCTRAG-FT<br>60.4<br>65.2<br>_With Retrieval_<br>Llama-3-8B-Instruct<br>59.8<br>69.5<br>INSTRUCTRAG-FT<br>64.6<br>71.3|Method<br>Pattern-based<br>LLM-based|
|---|---|
||_RAG w/o Training_<br>In-Context RALM<br>56.8<br>64.5<br>INSTRUCTRAG-ICL<br>62.1<br>67.6|
||_RAG w/ Training_<br>Vanilla SFT<br>56.6<br>65.1<br>INSTRUCTRAG-FT<br>65.7<br>69.7|



(a) Transfer from QA task to code generation task. (b) Evaluation with GPT-4o as the judge. It can be observed that our method consistently achieves better generalization performance in the unseen code generation task in both non-retrieval and RAG settings. This finding aligns with our observation that InstructRAG trained on QA tasks tends to generate more text-based comments that articulate the design of coding solutions compared to the off-the-shelf Llama-3-8B-Instruct, thereby leading to more accurate code generation. 

**Evaluation with LLM-as-a-judge.** Despite being standard evaluation metrics for question-answering, accuracy or exact match are known to be imperfect [13] as they mainly rely on pattern-matching to judge whether the predicted answer aligns with the ground-truth answer. Such metrics cannot handle cases where the predicted answer and ground-truth answer are synonyms ( _e.g._ , “Donald Trump” vs “Donald J. Trump” cannot be correctly recognized as a match), leading to biased evaluation results. 

Therefore, we adopt the LLM-as-a-judge [7, 115] approach to evaluate whether the model response aligns with the ground-truth answer using GPT-4o [66], which allows the judge to consider semantic equivalence and is expected to yield a more fair evaluation. As shown in Table 5b, we evaluate our method and baseline models on the open-domain Natural Questions benchmark in both training-free and trainable RAG settings. Compared to pattern-matching based metrics, LLM-as-a-judge generally leads to higher evaluation results, mostly due to its capability to accurately match semantically equivalent phrasings. Notably, our method consistently outperforms baselines under both pattern-matching based and LLM-based evaluation metrics, further validating the effectiveness of INSTRUCTRAG. 

## **4 Related Work** 

## **4.1 Retrieval-augmented Generation** 

Retrieval-augmented generation (RAG) is a widely adopted approach to enhance large language models (LLMs) with external knowledge [2, 4, 5, 19, 23, 42, 52, 70, 74], demonstrating promising potential to reduce hallucinations and enhance the generation accuracy of LLMs across various realworld applications [9, 37, 55, 58, 77, 80, 117]. Recently, a growing research effort has been devoted to enhancing RAG from various aspects, such as improving decoding efficiency [64, 36, 57], exploring long-context retrieval [96, 104], compressing prompts [32, 95], and addressing practical concerns such as adversarial retrieval [92, 116, 118] and privacy leakage [26, 108]. Despite their advantages, these RAG systems inevitably suffer from irrelevant information introduced by imperfect retrievers or noisy retrieval corpora. However, most existing works typically address this issue by improving the retrieval quality and reducing noise exposure to the model [22, 33, 72, 86, 100, 102, 109, 111]. Notable methods include adaptive retrieval [3, 35, 103] and query rewriting [8, 62]. In contrast, our work focuses on an orthogonal direction of developing explicit denoising methods for RAG, thereby enhancing the model’s noise robustness and generation accuracy, even in highly noisy contexts. 

## **4.2 Eliciting Reasoning in Large Language Models** 

Recent studies have extensively explored the reasoning capability of LMs, but typically not in the context of RAG where potentially noisy retrieved contents may mislead the reasoning if not 

9 

properly addressed. Chain-of-thought (CoT) prompting [89] is an effective method to elicit stepby-step reasoning from LMs by showing exemplars with detailed explanations ( _i.e._ , rationales [16, 49, 69, 107]) that lead to the final answer. However, such works often requires manually crafted demonstrations [84, 97], which is costly and requires extensive efforts and domain knowledge [114]. To mitigate this limitation, automatic chain-of-thought prompting (Auto-CoT) [112] is introduced to automatically select instances from the corpus coupled with zero-shot CoT [46], where the rationales are generated by LMs. Furthermore, it has been shown that CoT reasoning can be elicited even without explicit prompting, particularly for instruction-tuned LMs [85]. Another related work shows rationales generated by small models can help large models reason better [50]. Although rationalization has been extensively investigated in many NLP tasks [10, 20, 68, 90], none of them are designed for RAG, and how to leverage the instruction-following abilities of LMs for explicit denoising in the context of RAG is still underexplored. 

## **5 Conclusion** 

In this work, we presented INSTRUCTRAG, a simple retrieval-augmented generation (RAG) approach that explicitly denoises retrieved contents and produces accurate generations. By leveraging the strong instruction-following abilities of large language models, INSTRUCTRAG generates detailed rationales that articulate how the ground-truth answers can be derived from the retrieved documents. These synthetic rationales can serve as either in-context learning examples or supervised fine-tuning data, enabling the model to learn an explicit denoising process. Experiments on five knowledge-intensive benchmarks show INSTRUCTRAG consistently outperforms state-of-the-art RAG approaches with significant improvements in both training-free and trainable settings. Compared to the best baseline method, INSTRUCTRAG achieves an average improvement of 8.3% across all benchmarks, demonstrating its effectiveness in enhancing the noise robustness of retrieval-augmented generation. 

**Limitations.** In this work, we mainly conduct experiments on question answering-type tasks, and it remains unclear how our method may generalize to other scenarios ( _e.g._ , open-ended generation). Moreover, despite being the standard evaluation metrics, both accuracy and exact match are biased and cannot perfectly reflect the quality of the model’s generations. For instance, such metrics heavily rely on string matching, which assesses correctness at the lexical level rather than the semantic level, thereby failing to recognize different phrasings that convey identical meanings. The evaluation results also suffer from length bias, as longer generations tend to achieve higher accuracy. Exploring more advanced metrics like using LLMs as judges would better evaluate RAG model generations [102]. Another potential limitation is that our model might be subject to sample bias in the training data. Incorporating bias mitigation methods [17, 41, 101] would be helpful for further improving our work. 

**Future Work.** Future research directions include exploring more advanced techniques for generating high-quality rationales, such as incorporating domain-specific knowledge or leveraging multi-task learning to enable better generalization across various tasks. For instance, although the consistency ratio between synthetic rationales and ground-truth answers on training samples with at least one relevant document achieves 98%, the overall consistency ratio on all training samples is only 89%. This is because for some samples, none of the retrieved documents is relevant to the question, which significantly compromises the quality of the generated rationales. Therefore, it will be interesting to fully explore the potential of our method by incorporating additional designs such as a filtering mechanism, which we leave as future work. Additionally, investigating the scalability of our method to larger datasets and models could further demonstrate its generalizability and robustness. It will also be interesting to evaluate the model performance under long-context settings with a dynamic or extremely large number of retrieved documents. Finally, integrating our method with other advanced retrieval techniques, such as active retrieval, could potentially lead to even better performance on knowledge-intensive tasks. 

## **Acknowledgments** 

The authors would like to thank Xinyu Zhu from University of Virginia, Tianyu Gao and Zexuan Zhong from Princeton NLP group for their valuable feedback and discussions. 

10 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [2] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. Retrieval-based language models and applications. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts)_ , pages 41–46, 2023. 

- [3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- [4] Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and Wen-tau Yih. Reliable, adaptable, and attributable language models with retrieval. _arXiv preprint arXiv:2403.03187_ , 2024. 

- [5] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In _International conference on machine learning_ , pages 2206–2240. PMLR, 2022. 

- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Advances in neural information processing systems_ , 33:1877– 1901, 2020. 

- [7] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv preprint arXiv:2303.12712_ , 2023. 

- [8] Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu. RQ-RAG: Learning to refine queries for retrieval augmented generation. _arXiv preprint arXiv:2404.00610_ , 2024. 

- [9] Harrison Chase. LangChain, 2022. 

- [10] Howard Chen, Jacqueline He, Karthik Narasimhan, and Danqi Chen. Can rationalization improve robustness? In _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 3792–3805, 2022. 

- [11] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 17754–17762, 2024. 

- [12] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- [13] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. The power of noise: Redefining retrieval for rag systems. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 719–729, 2024. 

- [14] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. _arXiv preprint arXiv:2307.08691_ , 2023. 

- [15] Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. Chain-of-verification reduces hallucination in large language models. _arXiv preprint arXiv:2309.11495_ , 2023. 

11 

- [16] Yunlong Feng, Yang Xu, Libo Qin, Yasheng Wang, and Wanxiang Che. Improving language model reasoning with self-motivated learning. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)_ , pages 8840–8852, 2024. 

- [17] Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. Bias and fairness in large language models: A survey. _Computational Linguistics_ , pages 1–79, 2024. 

- [18] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 6465–6488, 2023. 

- [19] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. _arXiv preprint arXiv:2312.10997_ , 2023. 

- [20] Asish Ghoshal, Srinivasan Iyer, Bhargavi Paranjape, Kushal Lakhotia, Scott Wen-tau Yih, and Yashar Mehdad. Quaser: Question answering with scalable extractive rationalization. In _Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , pages 1208–1218, 2022. 

- [21] Shahriar Golchin and Mihai Surdeanu. Time travel in LLMs: Tracing data contamination in large language models. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- [22] Aman Gupta, Anup Shirgaonkar, Angels de Luis Balaguer, Bruno Silva, Daniel Holstein, Dawei Li, Jennifer Marsman, Leonardo O Nunes, Mahsa Rouzbahman, Morris Sharp, et al. RAG vs Fine-tuning: Pipelines, tradeoffs, and a case study on agriculture. _arXiv preprint arXiv:2401.08406_ , 2024. 

- [23] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In _International conference on machine learning_ , pages 3929–3938. PMLR, 2020. 

- [24] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott, Nuria Bel, and Chengqing Zong, editors, _Proceedings of the 28th International Conference on Computational Linguistics_ , pages 6609–6625, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. 

- [25] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. _arXiv preprint arXiv:2311.05232_ , 2023. 

- [26] Yangsibo Huang, Samyak Gupta, Zexuan Zhong, Kai Li, and Danqi Chen. Privacy implications of retrieval-based language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 14887–14902, 2023. 

- [27] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. _arXiv preprint arXiv:2112.09118_ , 2021. 

- [28] Gautier Izacard and Édouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pages 874–880, 2021. 

- [29] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. _Journal of Machine Learning Research_ , 24(251):1–43, 2023. 

12 

- [30] Alon Jacovi, Avi Caciularu, Omer Goldman, and Yoav Goldberg. Stop uploading test data in plain text: Practical strategies for mitigating data contamination by evaluation benchmarks. In _The 2023 Conference on Empirical Methods in Natural Language Processing_ , 2023. 

- [31] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Computing Surveys_ , 55(12):1–38, 2023. 

- [32] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. _arXiv preprint arXiv:2310.05736_ , 2023. 

- [33] Wenqi Jiang, Shuai Zhang, Boran Han, Jie Wang, Bernie Wang, and Tim Kraska. PipeRAG: Fast retrieval-augmented generation via algorithm-system co-design. _arXiv preprint arXiv:2403.05676_ , 2024. 

- [34] Zhengbao Jiang, Zhiqing Sun, Weijia Shi, Pedro Rodriguez, Chunting Zhou, Graham Neubig, Xi Victoria Lin, Wen-tau Yih, and Srinivasan Iyer. Instruction-tuned language models are better knowledge learners. _arXiv preprint arXiv:2402.12847_ , 2024. 

- [35] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 7969–7992, 2023. 

- [36] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin Jin. RAGCache: Efficient knowledge caching for retrieval-augmented generation. _arXiv preprint arXiv:2404.12457_ , 2024. 

- [37] Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. FlashRAG: A modular toolkit for efficient retrieval-augmented generation research. _arXiv preprint arXiv:2405.13576_ , 2024. 

- [38] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1601–1611, 2017. 

- [39] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 6769–6781, 2020. 

- [40] Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A Smith, Yejin Choi, Kentaro Inui, et al. RealTime QA: What’s the answer right now? _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [41] Noriaki Kawamae. Friendly conditional text generator. In _Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining_ , pages 420–428, 2023. 

- [42] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In _International Conference on Learning Representations_ , 2019. 

- [43] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp. _arXiv preprint arXiv:2212.14024_ , 2022. 

- [44] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. DSPy: Compiling declarative language model calls into self-improving pipelines. _arXiv preprint arXiv:2310.03714_ , 2023. 

13 

- [45] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- [46] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. _Advances in neural information processing systems_ , 35:22199–22213, 2022. 

- [47] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural Questions: A benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , 7:452–466, 2019. 

- [48] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles_ , pages 611–626, 2023. 

- [49] Andrew Lampinen, Ishita Dasgupta, Stephanie Chan, Kory Mathewson, Mh Tessler, Antonia Creswell, James McClelland, Jane Wang, and Felix Hill. Can language models learn from explanations in context? In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 537–563, 2022. 

- [50] Jooyoung Lee, Fan Yang, Thanh Tran, Qian Hu, Emre Barut, and Kai-Wei Chang. Can small language models help large language models reason better?: LM-guided chain-of-thought. In _Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)_ , pages 2835–2843, 2024. 

- [51] Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. _arXiv preprint arXiv:1811.07871_ , 2018. 

- [52] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrievalaugmented generation for knowledge-intensive NLP tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474, 2020. 

- [53] Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu, and Sanjiv Kumar. Large language models with controllable working memory. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 1774–1793, 2023. 

- [54] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. Pyserini: An easy-to-use Python toolkit to support replicable IR research with sparse and dense representations. _arXiv preprint arXiv:2102.10073_ , 2021. 

- [55] Jerry Liu. LlamaIndex, 2022. 

- [56] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_ , 12, 2024. 

- [57] Yuhan Liu, Hanchen Li, Kuntai Du, Jiayi Yao, Yihua Cheng, Yuyang Huang, Shan Lu, Michael Maire, Henry Hoffmann, Ari Holtzman, et al. CacheGen: Fast context loading for language model applications. _arXiv preprint arXiv:2310.07240_ , 2023. 

- [58] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey Svyatkovskiy. ReACC: A retrieval-augmented code completion framework. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6227–6240, 2022. 

- [59] Inbal Magar and Roy Schwartz. Data contamination: From memorization to exploitation. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 157–165, 2022. 

14 

- [60] Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D Manning, and Daniel E Ho. Hallucination-free? Assessing the reliability of leading AI legal research tools. _arXiv preprint arXiv:2405.20362_ , 2024. 

- [61] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 9802–9822, 2023. 

- [62] Shengyu Mao, Yong Jiang, Boli Chen, Xiao Li, Peng Wang, Xinyu Wang, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. RaFe: Ranking feedback improves query rewriting for RAG. _arXiv preprint arXiv:2405.14431_ , 2024. 

- [63] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. _arXiv preprint arXiv:2405.14734_ , 2024. 

- [64] Thomas Merth, Qichen Fu, Mohammad Rastegari, and Mahyar Najibi. Improving and accelerating retrieval-augmented generation with superposition prompting. In _Forty-first International Conference on Machine Learning_ , 2024. 

- [65] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, et al. Large dual encoders are generalizable retrievers. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 9844–9855, 2022. 

- [66] OpenAI. Hello GPT-4o. 2024. 

- [67] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_ , 35:27730–27744, 2022. 

- [68] Bhargavi Paranjape, Mandar Joshi, John Thickstun, Hannaneh Hajishirzi, and Luke Zettlemoyer. An information bottleneck approach for controlling conciseness in rationale extraction. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 1938–1952, 2020. 

- [69] Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. Explain yourself! leveraging language models for commonsense reasoning. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4932–4942, 2019. 

- [70] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. _Transactions of the Association for Computational Linguistics_ , 11:1316–1331, 2023. 

- [71] Stephen E Robertson and Steve Walker. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In _SIGIR’94: Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University_ , pages 232–241. Springer, 1994. 

- [72] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. _arXiv preprint arXiv:2401.18059_ , 2024. 

- [73] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. 

- [74] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. REPLUG: Retrieval-augmented black-box language models. _arXiv preprint arXiv:2301.12652_ , 2023. 

15 

- [75] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 3784–3803, 2021. 

- [76] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. _Nature_ , 620(7972):172–180, 2023. 

- [77] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. Improving the domain adaptation of retrieval augmented generation (RAG) models for open domain question answering. _Transactions of the Association for Computational Linguistics_ , 11:1–17, 2023. 

- [78] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. ASQA: Factoid questions meet long-form answers. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 8273–8288, 2022. 

- [79] Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, and Xin Luna Dong. Head-to-tail: How knowledgeable are large language models (LLM)? A.K.A will llms replace knowledge graphs? _arXiv preprint arXiv:2308.10168_ , 2023. 

- [80] Hanzhuo Tan, Qi Luo, Ling Jiang, Zizheng Zhan, Jing Li, Haotian Zhang, and Yuqun Zhang. Prompt-based code completion via multi-retrieval augmented generation. _arXiv preprint arXiv:2405.07530_ , 2024. 

- [81] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- [82] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- [83] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al. FreshLLMs: Refreshing large language models with search engine augmentation. _arXiv preprint arXiv:2310.03214_ , 2023. 

- [84] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ , 2022. 

- [85] Xuezhi Wang and Denny Zhou. Chain-of-thought reasoning without prompting. _arXiv preprint arXiv:2402.10200_ , 2024. 

- [86] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. RAT: Retrieval augmented thoughts elicit context-aware reasoning in long-horizon generation. _arXiv preprint arXiv:2403.05313_ , 2024. 

- [87] Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. Coderag-bench: Can retrieval augment code generation? _arXiv preprint arXiv:2406.14497_ , 2024. 

- [88] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In _International Conference on Learning Representations_ , 2021. 

- [89] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824–24837, 2022. 

- [90] Sarah Wiegreffe, Ana Marasovi´c, and Noah A Smith. Measuring association between labels and free-text rationales. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 10266–10284, 2021. 

16 

- [91] Kevin Wu, Eric Wu, and James Zou. How faithful are RAG models? quantifying the tug-of-war between RAG and LLMs’ internal prior. _arXiv preprint arXiv:2404.10198_ , 2024. 

- [92] Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, and Prateek Mittal. Certifiably robust RAG against retrieval corruption. _arXiv preprint arXiv:2405.15556_ , 2024. 

- [93] Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun. Lawformer: A pre-trained language model for Chinese legal long documents. _AI Open_ , 2:79–84, 2021. 

- [94] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrievalaugmented generation for medicine. _arXiv preprint arXiv:2402.13178_ , 2024. 

- [95] Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and selective augmentation. _arXiv preprint arXiv:2310.04408_ , 2023. 

- [96] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [97] Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. Search-in-thechain: Interactively enhancing large language models with search for knowledge-intensive tasks. In _Proceedings of the ACM on Web Conference 2024_ , pages 1362–1373, 2024. 

- [98] Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing, 2024. 

- [99] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models. _arXiv preprint arXiv:2401.11817_ , 2024. 

- [100] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation. _arXiv preprint arXiv:2401.15884_ , 2024. 

- [101] Ke Yang, Charles Yu, Yi R Fung, Manling Li, and Heng Ji. Adept: A debiasing prompt framework. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 10780–10788, 2023. 

- [102] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga, Anuj Kumar, Wen tau Yih, and Xin Luna Dong. CRAG – comprehensive RAG benchmark. _arXiv preprint arXiv:2406.04744_ , 2024. 

- [103] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [104] Howard Yen, Tianyu Gao, and Danqi Chen. Long-context language modeling with parallel context encoding. _arXiv preprint arXiv:2402.16617_ , 2024. 

- [105] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust to irrelevant context. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [106] Wenhao Yu, Zhihan Zhang, Zhenwen Liang, Meng Jiang, and Ashish Sabharwal. Improving language models via plug-and-play retrieval feedback. _arXiv preprint arXiv:2305.14002_ , 2023. 

- [107] Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D Goodman. Quiet-STaR: Language models can teach themselves to think before speaking. _arXiv preprint arXiv:2403.09629_ , 2024. 

- [108] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, et al. The good and the bad: Exploring privacy issues in retrievalaugmented generation (RAG). _arXiv preprint arXiv:2402.16893_ , 2024. 

17 

- [109] Jinghan Zhang, Xiting Wang, Weijieying Ren, Lu Jiang, Dongjie Wang, and Kunpeng Liu. RATT: A thought structure for coherent and correct LLM reasoning. _arXiv preprint arXiv:2406.02746_ , 2024. 

- [110] Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball. _arXiv preprint arXiv:2305.13534_ , 2023. 

- [111] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonzalez. RAFT: Adapting language model to domain specific RAG. _arXiv preprint arXiv:2403.10131_ , 2024. 

- [112] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [113] Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing. Verify-and-edit: A knowledge-enhanced chain-of-thought framework. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 5823–5840, 2023. 

- [114] Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, and Denny Zhou. Take a step back: Evoking reasoning via abstraction in large language models. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- [115] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [116] Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. Poisoning retrieval corpora by injecting adversarial passages. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 13764–13775, 2023. 

- [117] Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang, and Graham Neubig. DocPrompting: Generating code by retrieving the docs. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [118] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. PoisonedRAG: Knowledge poisoning attacks to retrieval-augmented generation of large language models. _arXiv preprint arXiv:2402.07867_ , 2024. 

18 

## **A Implementation Details** 

**Retrieval setup.** Following [3, 70], we use the Wikipedia dump from [39] as the external retrieval corpus for all five benchmarks studied in this work, where each document is a disjoint text block of up to 100 words extracted from a Wikipedia article. We compared all RAG methods under a diverse retrieval environment with various sparse and dense retrievers and number of retrieved documents. Specifically, we use Contriever-MS MARCO as the retriever for PopQA and TriviaQA, DPR for Natural Questions, GTR for ASQA, and BM25 for 2WikiMultiHopQA. By default, we retrieve the top 5 documents from the retrieval corpus for each query in all tasks except 2WikiMultiHopQA, where the top 10 documents are retrieved. We use the official weights for all dense retrievers and the implementation from Pyserini [54] for the sparse retriever BM25. 

**Training details.** Our models are trained on 4 Nvidia H100 GPUs with 80GB memory via fullparameter fine-tuning. We use fully sharded data parallelism (FSDP) for distributed training, along with FlashAttention [14] and bf16 mixed precision training enabled for computation efficiency. By default, all models are trained using the Adam optimizer [45] for 2 epochs, with a batch size of 128, a learning rate of 2.5e-5, and a cosine learning rate schedule with 3% warmup steps. For the trainable baseline vanilla SFT, we use a slightly different learning rate of 2e-5 based on our hyper-parameter search results. To fairly compare with Self-RAG and RetRobust, we re-implement them using Llama-3-Instruct-8B. We also optimize their performance through an extensive hyper-parameter search with learning rates in [8e-6, 1e-5, 2e-5] and training epochs in [1, 2, 3]. For Self-RAG, we use a learning rate of 1e-5 with a single training epoch. For RetRobust, we use a learning rate of 2e-5 with two training epochs. The only exception is the training for RetRobust on 2WikiMultiHopQA, where we train the model for 5 epochs on the augmented training set released by the original authors. The maximum token length for all models is fixed at 4096. 

**Inference details.** By default, the number of demonstrations used in INSTRUCTRAG-ICL and the baseline method few-shot demonstration with instruction is set to be 2. We use vLLM [48] to load models for memory-efficient inference and adopt the greedy decoding strategy for model generation. 

## **B Case Study** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales_images/iclr_2025_InstructRAG_Instructing_Retrieval-Augmented_Generation_via_Self-Synthesized_Rationales.pdf-0019-05.png)


**----- Start of picture text -----**<br>
Attention from Answer to Documents Attention from Answer to Documents<br>Doc 1 Doc 1<br>0 . 6 0 . 6<br>Doc 2 Doc 2<br>Doc 3 0 . 4 Doc 3 0 . 4<br>Doc 4 Doc 4<br>0 . 2 0 . 2<br>Doc 5 Doc 5<br>22 23 24 25 26 27 28 29 30 31 22 23 24 25 26 27 28 29 30 31<br>Last 10 Attention Layers Last 10 Attention Layers<br>(a) Vanilla SFT. (b) INSTRUCTRAG-FT.<br>**----- End of picture text -----**<br>


Figure 5: Visualization of model attention from answer to retrieved documents on a random sample from the ASQA task, where Doc 2 is the only relevant document that contains the correct answer. **Attention visualization.** To intuitively understand the denoising process of our INSTRUCTRAG, we visualize its attention from the answer to retrieved documents. As pointed out by a recent work [106], only attention distributions from deep layers can accurately reflect the LM’s retrieval behavior and focus on key information, while attention from shallow layers usually do not imply meaningful patterns. Therefore, we only plot the attention weights of the last 10 layers (Layer 22 to Layer 31). As presented in Figure 5, our model accurately identifies the only benign document from noisy input, showing a strong denoising signal compared to vanilla SFT. 

**Generation comparison.** Figure 6 compares the generated responses of in-context RALM, vanilla SFT, and our INSTRUCTRAG-FT for an actual question from the ASQA task. Among them, only our method can correctly answer this question while providing comprehensive denoising details. Specifically, it first identifies potentially relevant documents from noisy inputs, and then lays out the candidate information. More encouragingly, we find that INSTRUCTRAG-FT is able to refer to its own parametric knowledge when no relevant document is present in the context after denoising, demonstrating its superiority over existing RAG approaches. 

19 

_**Question**_ : Who played the admiral on Star Trek: Discovery? _**Answer**_ : Katrina Cornwell, portrayed by **Jayne Brook** , is a Starfleet admiral. _**Contexts**_ : Document 1 (Title: Star Trek: Insurrection):…Document 2 (Title: Star Trek: Discovery (season 1)):…Document 3 (Title: Star Trek):… Document 4 (Title: Brian Thompson):…Document 5 (Title: Star Trek: Enterprise):… _**In-Context RALM**_ According to Document 2, there is no mention of an Admiral character in Star Trek: Discovery. However, according to Document 3, Captain Gabriel Lorca (played by Jason Isaacs) is the commander of the USS Discovery. **Wrong Answer** _**Vanilla SFT**_ Star Trek: Discovery is an American television series created by Bryan Fuller and Alex Kurtzman for the streaming service CBS All Access. The series is set about a decade before the events of the original Star Trek series and  follows the crew of the USS Discovery as they attempt to stop a massive Klingon armada while uncovering the truth about the Red Angel and Section 31. The character of Admiral Cornwell is played by Jayne Brook. The character of Admiral Cornwell is a Starfleet officer who serves as the first officer of the USS Enterprise NCC-1701-D. The character of Admiral Cornwell is played by Elizabeth Maxwell. **Confusing Answer** 

## _**InstructRAG-FT**_ 

The documents that are useful to answer the question "Who played the admiral on Star Trek Discovery?" are: Document 1: Star Trek: Insurrection, Document 3: Star Trek, Document 5: Star Trek: Enterprise. These documents provide information about the casting of actors in Star Trek productions, which is relevant to the question. From Document 1, we learn that Anthony Zerbe played the role of Admiral Matthew Dougherty in Star Trek: Insurrection. From Document 3, Star Trek: Discovery is a prequel to the Star Trek franchise and premiered in 2017. From Document 5, we learn that Vaughn Armstrong played Admiral Maxwell Forest in Star Trek: Enterprise. However, none of these documents mention the specific admiral on Star Trek Discovery, which is Admiral Cornwell, played by Jayne Brook. The correct answer is not found in the provided documents, but it can be answered based on external knowledge. Jayne Brook played the role of Admiral Cornwell in Star Trek: Discovery. **Correct Answer** 

Figure 6: A case study of INSTRUCTRAG-FT compared with in-context RALM and vanilla SFT. The red texts denote irrelevant or inaccurate model generations, while the green texts denote contents relevant to the question. This study shows that our model can effectively identify relevant information from noisy input and leverage its own knowledge to correctly answer questions when required. 

## **C Prompt Templates** 

In this work, we instantiate the proposed INSTRUCTRAG with off-the-shelf instruction-tuned LMs ( _i.e._ , meta-llama/Meta-Llama-3-8B-Instruct and meta-llama/Meta-Llama-3-70B-Instruct), and apply the official Meta-Llama-3-Instruct chat template (marked in gray) in all prompts. 

**Rationale generation.** Below are the prompt templates for rationale generation used in all five tasks. Table 6 shows the rationale template used in the ablation study. For simplicity, we use the same prompt structure (Table 7) for all tasks with minor differences in task-specific instructions (Table 8). 

Table 6: Rationale template used in ablation study. 

## Rationale Template 

**Positive Template:** After reviewing the provided document, I found that only documents {documents} contain relevant information to answer the question. Based on my knowledge and the provided contents, the answer is: {answer}. **Negative Template:** After reviewing the provided document, I found that none of them contain relevant information to answer the question. Based on my knowledge and the provided contents, the answer is: {answer}. 

Table 7: Rationale generation prompt template. Rationale Generation **Input:** <|begin_of_text|><|start_header_id|>user<|end_header_id|> Read the following documents relevant to the given question: {question} Document [1] (Title: _· · ·_ ): {contents} _· · ·_ Please identify documents that are useful to answer the given question: “{question}”, and explain how the contents lead to the answer: {answer}. If none of the documents is aligned with the answer, in that case, you have to explain the answer only based on your own knowledge, without referring to the provided information. **{task-specific instruction}** <|eot_id|><|start_header_id|>assistant<|end_header_id|> **Output:** {rationale} 

20 

Table 8: Task-specific instruction used in rationale generation prompt. 

Task-specific Instruction for Rationale Generation 

**ASQA:** Note that the question may be ambiguous and have multiple correct answers. Make sure your response includes all correct answers and provides clear reasoning details followed by a concise conclusion. 

**PopQA:** Note that the question mainly asks about the object entity that holds a certain relationship with the given subject entity. There may be multiple correct answers. Make sure your response includes all correct answers and provides clear reasoning details followed by a concise conclusion. 

**TriviaQA / Natural Questions / 2WikiMultiHopQA:** Note that the question may be compositional and require intermediate analysis to deduce the final answer. Make sure your response is grounded and provides clear reasoning details followed by a concise conclusion. 

**Inference prompts** . Below we present the inference prompts for both training-free and trainable RAG methods used in this work, including in-context RALM (Table 9), few-shot demonstrations with instruction (Table 10), and vanilla supervised fine-tuning (Table 11). Note that for a fair comparison, the inference prompt for our INSTRUCTRAG-FT is exactly the same as vanilla SFT. Similarly, the inference prompt for INSTRUCTRAG-ICL shares the same inference instruction as the few-shot demonstrations with instruction. The only difference between the prompts of these two methods lies in the demonstrations where INSTRUCTRAG-ICL employs denoising question-rationale _⟨q, r⟩_ pairs, while few-shot demonstrations with instruction uses plain question-answer _⟨q, a⟩_ pairs. 

Table 9: Inference prompt for In-Context RALM. 

## In-Context RALM 

**Input:** <|begin_of_text|><|start_header_id|>user<|end_header_id|> Document [1] (Title: _· · ·_ ): {contents} 

_· · ·_ Based on your knowledge and the provided information, answer the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|> 

## **Output:** {answer} 

Table 10: Inference prompt for INSTRUCTRAG-ICL and few-shot demonstrations with instruction. 

## INSTRUCTRAG-ICL / Few-shot Demonstrations with instruction 

**Input:** <|begin_of_text|><|start_header_id|>user<|end_header_id|> Your task is to analyze the provided documents and answer the given question. Please generate a brief explanation of how the contents of these documents lead to your answer. If the provided information is not helpful to answer the question, you only need to respond based on your own knowledge, without referring to the documents. 

Below are some examples of how to answer the question: {example question _q_ 1} {example answer _a_ 1 / rationale _r_ 1} _· · ·_ Document [1] (Title: _· · ·_ ): {contents} _· · ·_ Based on your knowledge and the provided information, answer the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|> 

**Output:** {answer} 

Table 11: Inference prompt for INSTRUCTRAG-FT and vanilla supervised fine-tuning. 

## INSTRUCTRAG-FT / Vanilla Supervised Fine-tuning 

**Input:** <|begin_of_text|><|start_header_id|>user<|end_header_id|> Document [1] (Title: _· · ·_ ): {contents} _· · ·_ Based on your knowledge and the provided information, answer the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|> 

**Output:** {answer} 

21 

