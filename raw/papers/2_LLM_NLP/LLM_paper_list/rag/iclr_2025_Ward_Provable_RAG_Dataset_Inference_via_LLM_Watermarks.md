Published as a conference paper at ICLR 2025 

## WARD: PROVABLE RAG DATASET INFERENCE VIA LLM WATERMARKS 

**Nikola Jovanovi´c, Robin Staab, Maximilian Baader, Martin Vechev** ETH Zurich 

```
{nikola.jovanovic,robin.staab,mbaader,martin.vechev}@inf.ethz.ch
```

## ABSTRACT 

RAG enables LLMs to easily incorporate external data, raising concerns for data owners regarding unauthorized usage of their content. The challenge of detecting such unauthorized usage remains underexplored, with datasets and methods from adjacent fields being ill-suited for its study. We take several steps to bridge this gap. First, we formalize this problem as (black-box) RAG Dataset Inference ( _RAG-DI_ ). We then introduce a novel dataset designed for realistic benchmarking of RAGDI methods, alongside a set of baselines. Finally, we propose WARD, a method for RAG-DI based on LLM watermarks that equips data owners with rigorous statistical guarantees regarding their dataset’s misuse in RAG corpora. WARD consistently outperforms all baselines, achieving higher accuracy, superior query efficiency and robustness. Our work provides a foundation for future studies of RAG-DI and highlights LLM watermarks as a promising approach to this problem. 

## 1 INTRODUCTION 

Retrieval-augmented generation (RAG) has emerged as a popular approach to mitigate limitations of large language models (LLMs) such as hallucinations, the high cost of adapting to new knowledge via fine-tuning, and the inability to back up claims by sources (Lewis et al., 2020). By integrating retrieval, LLMs gain in-context access to large corpora of high-quality, up-to-date data, enabling them to generate more accurate and source-supported responses. To maintain relevance, RAG providers must continuously update their corpus with new data. However, this raises concerns regarding the unauthorized usage of documents, particularly when publicly available documents are used without the owner’s permission (Grynbaum & Mac, 2023; Wei et al., 2024a); see App. A for a more elaborate discussion of this issue and its prevalence in practice. Crucially, there is currently no way to conclusively prove such unauthorized usage by a RAG system, and enforce an opt-out by the owner. 

**RAG Dataset Inference (RAG-DI)** We formalize this problem as _RAG Dataset Inference_ ( _RAGDI_ ), where a data owner aims to detect unauthorized inclusion of their dataset in a RAG corpus via black-box queries (Fig. 1). In the first comprehensive study of this problem, we observe that existing datasets, used in adjacent works on RAG privacy, are not suitable for RAG-DI. First, the samples in these datasets may have been used in contemporary LLM training, complicating realistic evaluations where RAG corpora consist of new data. Second, these datasets do not model _fact redundancy_ , a key property of real-world RAG, where multiple documents have similar content, either due to scraping data from various sources, e.g., news (Gao et al., 2023), or due to chunking. Another challenge in studying RAG-DI stems from the lack of baselines applicable in a realistic black-box setting. 

**Foundations for RAG-DI** In this work, we take multiple steps to bridge these gaps: First, we introduce FARAD, a new dataset specifically designed for RAG-DI evaluation under realistic conditions. FARAD contains fictional articles that are by design not part of any LLM training data, and can enable evaluations under fact redundancy, enabling accurate assessment of RAG-DI methods. Second, we adapt prior work on RAG Membership Inference Attacks (MIAs) (Li et al., 2024; Anderson et al., 2024) to the RAG-DI problem, and propose a simple baseline FACTS. In our evaluation on FARAD, we find that (i) despite its extreme simplicity, FACTS outperforms other baselines in settings with no fact redundancy, further underscoring the drawbacks of existing datasets, and (ii) when fact redundancy is present, no baseline achieves satisfactory performance. This highlights the need for novel approaches capable of reliably identifying unauthorized usage of documents in RAG corpora. 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Dataset  RAG Corpus<br>LLM watermarking<br> RAG<br>Data Watermarking Retrieval Mechanism   Provider<br>RAG Dataset Inference<br>Data Owner<br>query<br>Watermark<br>Detector<br>Inconclusive!<br>LLM<br>agg.<br>My data was used!<br>?<br>**----- End of picture text -----**<br>


Figure 1: Overview of RAG Dataset Inference using WARD, our method based on LLM watermarks. 

**LLM Watermarks as a reliable RAG-DI method** To this end, we introduce WARD, a RAG-DI method that protects the data owner’s dataset by imprinting LLM watermarks (Kirchenbauer et al., 2023; Kuditipudi et al., 2024). As Fig. 1 illustrates, given a limited number of black-box queries _qi_ to the retrieval-augmented LLM, the data owner can detect even small traces of the watermark across responses _ri_ , and for the first time obtain rigorous statistical guarantees regarding the usage of their dataset in the RAG corpus, enabling them to effectively audit the RAG provider. In our experiments on FARAD, we show that WARD consistently outperforms all baselines across a variety of challenging settings, showing high rate of true positives (often 100%), with no false accusations. Further, due to its robustness across settings, and the fact that it engages only in natural-looking interactions with the retrieval-augmented LLM, WARD retains its performance even under attempts by the RAG provider to prevent unintended uses of the system. These results underscore the effectiveness of WARD in enforcing data usage policies and protecting data owners’ rights in the context of RAG systems. 

**Main contributions** We make the following key contributions: 

- We formalize a previously unexplored problem, RAG Dataset Inference (RAG-DI), where a data owner aims to detect unauthorized usage of their dataset in a RAG system (§3). 

- We facilitate research on this problem by (i) proposing a new dataset FARAD, specifically designed for benchmarking RAG-DI methods under realistic conditions, and (ii) introducing an initial set of baseline methods (§3.1 and §3.2). 

- We propose LLM watermarks as a way to provably, robustly and reliably detect unauthorized data usage in RAG corpora, introducing WARD as a novel RAG-DI method (§4). 

- Our experimental evaluation in a wide range of settings affirms the fundamental limitations of existing datasets and all RAG-DI baselines, and demonstrates the effectiveness of WARD, which consistently shows high accuracy, query efficiency, and robustness (§5). Our source code and the FARAD dataset are publicly available at `https://github.com/eth-sri/ward` . 

## 2 BACKGROUND 

**Retrieval-augmented generation (RAG)** RAG is a common way to enhance LLMs: For a given user query _q_ , the _k ∈_ N most relevant documents _Dq_ = _R_ ( _q, D_ ) _⊆ D_ are retrieved from a corpus _D_ , using a retrieval method _R_ (Gao et al., 2023). The query _q_ is generally combined with _Dq_ , and fed into an LLM to generate a more factual response _r_ = _M_ ( _q, Dq_ ). Expanding _D_ also enables access to new information without costly retraining (Lewis et al., 2020). RAG is especially suitable for domains where new information is generated often, e.g., news articles or software documentation. In this work we use _M[⋆]_ to denote a _RAG system_ , i.e., a retrieval-augmented LLM _M_ . 

**LLM watermarking** LLM watermarks enable model owners to provably and reliably track text generated by their LLM. In this work, we focus on the prominent _red-green watermarks_ (Kirchenbauer 

2 

Published as a conference paper at ICLR 2025 

et al., 2023). During text generation, at each step _t_ , the token vocabulary _V_ is split into two parts, _γ|V | green_ (encouraged) tokens, and (1 _− γ_ ) _|V | red_ (discouraged) tokens, for some _γ ∈_ [0 _,_ 1]. Given _context width h_ , the split is commonly a function of tokens at positions _t − h, . . . , t_ , as well as a secret salt. To add the watermark, the logits of green tokens are increased by _δ ∈_ R _≥_ 0, boosting their probability of being sampled. The watermark is detected using a statistical test, based on the expectation that non-watermarked text of length _T_ has _γT_ green tokens. Namely, we use the z-score 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0003-02.png)


where _|s|g_ is the number of green tokens in a given text _s_ , and _T_ = _|s|_ . From here, we derive the _p_ -value _p_ = 1 _−_ Φ( _z_ ), where Φ is the CDF of the standard normal distribution, and consider the text watermarked if _p < α_ for some threshold _α_ . While not perfectly robust, these watermarks generally persist under moderate text transformations, such as paraphrasing or segment omission (Piet et al., 2023; Kirchenbauer et al., 2024; Sander et al., 2024), making them suitable for our setting, where our goal will be to propagate the watermark signal through the RAG pipeline, as we will describe in §4. 

**RAG membership inference attacks** While RAG is a relatively novel concept, recent work already studies membership inference attacks (MIAs) in this setting, proposing two methods that we denote SIB (Li et al., 2024) and AAG (Anderson et al., 2024). A MIA’s goal is to output mi( _d, M[⋆]_ ) = 1 if a document _d_ is part of the retrieval corpus _D_ of a RAG system _M[⋆]_ , or mi( _d, M[⋆]_ ) = 0 otherwise, based only on queries to _M[⋆]_ . To this end, SIB queries _M[⋆]_ with _q_ , a prefix of _d_ , to obtain the response _r_ = _M_ ( _q, Dq_ ). Then, it computes two scores: the cosine similarity between the embeddings of _d_ and _r_ , and the perplexity of _r_ , outputting mi( _d, M[⋆]_ ) = 1 if the similarity is above a threshold _θ_ similarity, and the perplexity is below a threshold _θ_ perplexity, both trainable parameters. Note that this requires gray-box access to _M[⋆]_ for perplexity computation. The other method, AAG, directly prompts _M[⋆]_ to answer if _d_ is in the context. If it replies positively, they set mi( _d, M_ ) = 1. In §3, we will introduce the black-box RAG dataset inference setting, and adapt both baselines to it. Notably, while dataset inference was studied alongside MIA for training data (see §6), no prior work studies it for RAG. 

## 3 RAG DATASET INFERENCE 

We formalize the problem of _RAG Dataset Inference_ ( _RAG-DI_ ), and present our contributions aimed at facilitating studies of this problem. In §3.1, we make the case that existing datasets commonly used for adjacent tasks (e.g., RAG MIA) are fundamentally unsuitable for RAG-DI, and propose a new dataset in an attempt to address these shortcomings. In §3.2, we establish a set of baselines for RAG-DI, by adapting RAG MIA work introduced in §2, and proposing a simple baseline, FACTS. 

**The RAG-DI problem** The key entities in RAG-DI are the _data owner_ , who aims to protect their _n_ -document dataset _D_ do from unauthorized usage in a RAG corpus, and the _RAG provider_ , who exposes black-box access to their retrieval-augmented LLM _M[⋆]_ , which uses a corpus _D_ . The data owner’s goal is to determine if _D_ do _⊆ D_ , i.e., whether their data was secretly included in the corpus. To this end, they may proactively modify _D_ do before publishing it, and can query _M[⋆]_ in a black-box way, aiming to minimize the number of such queries. Crucially, the data owner makes a single dataset-level decision, as opposed to document-level decisions of MIAs. Formally, a RAG-DI method _di_ should output di( _D_ do _, M[⋆]_ ) = 1 if _D_ do _⊆ D_ ( _IN_ case) and 0 otherwise ( _OUT_ case). 

## 3.1 A DATASET SUITABLE FOR RAG-DI 

To enable suitable evaluation of RAG-DI methods, we require a dataset of documents with the following properties. First, we aim to match a key use-case of RAG (as described in §2) where up-to-date knowledge is added to _D_ instead of costly repeated fine-tuning of _M_ . To model this, our documents should provably not be part of the training data of _M_ , i.e., of current open/closed LLMs, as those will be used to instantiate _M_ when studying RAG-DI. Second, to model the practical case where knowledge is redundant and spread across multiple sources (e.g., news articles, a common motivating example for RAG (Gao et al., 2023)), the dataset should contain documents with overlapping topics and information ( _fact redundancy_ ). As we will empirically demonstrate in §5, while fact redundancy is more realistic, it makes RAG-DI significantly harder. 

3 

Published as a conference paper at ICLR 2025 

**Current state** As there is no prior work on RAG-DI, we turn to related work on RAG privacy (Anderson et al., 2024; Li et al., 2024; Zeng et al., 2024), including MIAs introduced in §2. We observe that evaluations in these works rely primarily on _EnronEmails_ (Klimt & Yang, 2004) and _HealthcareMagic_ datasets (Zeng et al., 2020; Mrini et al., 2021), motivated by the presence of PII in their samples. However, it can not be ruled out that these datasets were used to train contemporary LLMs, as e.g., _EnronEmails_ has been publicly available since 2004. More importantly, fact redundancy is by design not satisfied in either of these cases. These shortcomings motivate us to construct a new dataset, FARAD ( _Fact-Redundant Article Dataset_ ), tailored to RAG-DI. 

**The FARAD dataset** FARAD consists of a number of _groups_ . Each group contains articles that share a topic and a significant amount of information, but are independently written by a different (LLM) author. As our data source we use _RepLiQA_ (Monteiro et al., 2024), that contains articles about fictional entities and events, which by design ensures that this knowledge was not present in any LLM training data. _RepLiQA_ is released gradually—to create FARAD, we use split 0 as the only one available at the time of writing, but plan to expand this to future splits. 

Each _RepLiQA_ article is a _source_ for one of our groups— we use the pipeline illustrated in Fig. 2. First, we prompt GPT4O to distill the information content of the article into 5 self-contained _key facts_ , crucial to understanding the article, and 10 self-contained _additional facts_ , that are present in the article but not essential to its key message (see App. F.2). Next, to create a group, we sample an author model from the set _A_ = _{_ GPT4O, CLAUDE3.5-SONNET, LLAMA3.1- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0004-04.png)


**----- Start of picture text -----**<br>
Source<br>#12#12 3<br>Fact Extraction<br>Key  Additional  Authors<br>Facts Facts<br>…<br>FARAD  group #123<br>FARAD-Easy<br>  no fact redundancy<br>#123 #374 #266    #597<br>FARAD-Hard<br>  fact redundancy<br>#123 #374 #374 #123<br>**----- End of picture text -----**<br>


Figure 2: Overview of the generation pipeline of FARAD, and the resulting _Easy_ and _Hard_ evaluation settings. 

405B, QWEN1.5-110B _}_ of state-of-the-art LLMs. The selected model writes a 500–1000 word article that must include all 5 key facts, 2 randomly sampled additional facts, and is encouraged to invent additional quotes, hypotheses, or personal opinions, as long as they do not contradict any present fact. The diversity of authors, sampled facts, and (desirable) hallucinated content results in a varied set of fictional articles grounded in the same core knowledge, which satisfies our earlier requirements. While this pipeline can be extended to larger/more groups, more authors, and different fact combinations, we limit FARAD to 3591 groups (3391 for testing, and 200 for training) of 4 articles each, one per author from _A_ . 

**Use in RAG-DI evaluation** In our experiments in §5, we use FARAD to create two evaluation settings: _Easy_ and _Hard_ , illustrated in Fig. 2 (bottom). The _Easy_ setting follows the traditional setup described above where facts are uniquely represented, by always using at most 1 article per FARAD group. The _Hard_ setting introduces fact redundancy by always including all 4 articles from a group. As we demonstrate shortly, the _Easy_ setting is solvable by a simple baseline, while the _Hard_ setting, as the realistic one and our main focus, is a significant challenge for all baseline methods. In App. E.1 we discuss potential improvements to our data generation pipeline as items for future work. 

## 3.2 RAG-DI BASELINES 

We proceed to establish an initial set of baselines for the RAG-DI problem. For this we adapt existing RAG MIAs, AAG and SIB, introduced in §2, and additionally propose a simple baseline, FACTS. 

**Adapting RAG MIAs** By design, the existing RAG MIAs make document-level decisions, i.e., they decide if a _single document_ is in the RAG corpus (mi( _d, M[⋆]_ ) = 1) or not (mi( _d, M[⋆]_ ) = 0). We extend this to dataset-level decisions of RAG-DI in an (empirically) optimal way. First, we apply a given method to each _d ∈ D_ do to obtain _n_ binary decisions, and set _s_ ( _D_ do _, M[⋆]_ ) = _n_[1] � _d∈D_ do[mi(] _[d,][ M][⋆]_[)][.] 

4 

Published as a conference paper at ICLR 2025 

We then instantiate several _IN_ and _OUT_ cases of RAG-DI using the training set, and compute the corresponding _s_ in and _s_ out. We finally make a dataset-level statement di( _D_ do _, M[⋆]_ ) = 1 if _s_ ( _D_ do _, M[⋆]_ ) _>_[1] 2[(] _[s]_[in][ +] _[ s]_[out][)][ and][ di(] _[D]_[do] _[,][ M][⋆]_[) = 0][ otherwise.][During this process, for S][IB][, we also] grid search over its trainable parameters _θ_ similarity and _θ_ perplexity (as in Li et al. (2024)), choosing values that maximize _s_ in _− s_ out. We note that SIB is proposed as a gray-box method; we adapt it to our black-box setting by using an auxiliary language model to estimate the perplexity of responses. 

**The FACTS baseline** To substantiate our point about the effect of fact multiplicity on the hardness of RAG-DI, we introduce a simple baseline, FACTS. As the above RAG MIAs, FACTS is a documentlevel method. Given a document _d_ , FACTS prompts an auxiliary LLM to generate a single question that is only answerable by reading _d_ . Then, _M[⋆]_ is prompted with that question, and if it deems it unanswerable, we set mi( _d, M[⋆]_ ) = 0, and set it to 1 otherwise. We aggregate such document-level decisions into a corpus-level decision by fitting a threshold on the training set, as described above. In §5, we will demonstrate that this simple method is sufficient to solve the _Easy_ setting more reliably than other baselines, yet is, due to fact multiplicity, extremely unreliable in the _Hard_ setting. 

## 4 LLM WATERMARKING AS AN EFFECTIVE RAG-DI METHOD 

Before describing our proposed RAG-DI method based on LLM watermarks, we first outline three key design requirements that a desirable RAG-DI method should fulfill. We require the following: 

1. _Monotonicity_ . With more queries to the retrieval-augmented LLM _M[⋆]_ , the accuracy of the method’s predictions should consistently improve, preferably at a high rate. 

2. _Guarantees_ . The method should be able to provide a statistical guarantee for its decision, with exceedingly rare and well-controlled Type 1 errors, as falsely accusing RAG providers is highly undesirable in practice, and undermines the trust in the method. 

3. _Robustness_ . The method should maintain high accuracy under diverse evaluation settings, including attempts by the RAG provider to actively conceal unauthorized data usage. 

In App. A we motivate these requirements in detail by reflecting on the context around RAG-DI. As we demonstrate in §5, all RAG-DI baselines introduced in §3.2 violate _all_ requirements to some extent. To address this, we propose WARD ( _Watermarking for RAG-DI_ ), a proactive RAG-DI method that is based on LLM watermarks, and discuss why it is likely to fulfill all stated desiderata. 

**WARD: RAG-DI via LLM watermarking** We assume the data owner has protected each _di ∈ D_ do by embedding an LLM watermark either via a human-in-the-loop procedure or (as in this work) by rephrasing each document with a watermarked LM. While, in principle, any LLM watermark can be applied, we focus on popular red-green watermarks (see §2). In §5.3, we confirm that this results in quality texts, faithful to the original ones. To audit the RAG provider’s corpus, for each _di ∈ D_ do, WARD generates an open-ended content-related question _qi_ , and queries _M[⋆]_ . If _di ∈ D_ ( _IN_ case), we expect the retrieval method _R_ to introduce watermarked content from _di_ into the LLM context. As noted in §2 and validated in §5, the robustness of watermarks to text transformations is then sufficient to propagate the traces of the signal to _ri_ = _M[⋆]_ ( _qi_ ), the final response of the LLM. 

**Boosting a weak signal** Requiring that each _ri_ is flagged as watermarked, i.e., has watermark detector p-value _p < α_ , would be a strong assumption, as the watermark signal is likely to degrade throughout the RAG pipeline. However, this is not necessary for WARD to be effective. Instead, following Sander et al. (2024), after _n_ queries we compute a _joint_ p-value _R_ = _{r_ 1 _, . . . , rn}_ , directly corresponding to the null hypothesis _“the data owner’s dataset Ddo is not in the RAG corpus D”_ . 

This joint p-value can satisfy _p < α_ , i.e., reject the null hypothesis, even when individual _ri_ carry only weak watermark signal, that would individually not reject it. To illustrate this, given a desired p-value threshold _α_ , Eq. (1) implies that the required ratio of green tokens in _R_ is at least 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0005-12.png)


5 

Published as a conference paper at ICLR 2025 

where Φ is the standard normal distribution CDF, and _|R|[⊕]_ the total length of responses in _R_ . This lower bound decreases quickly as _|R|[⊕]_ increases: In Fig. 3 we plot the lower bound as a function of _n_ , assuming _∀i_ : _|ri|_ = 400 tokens, _α ≈_ 3 _·_ 10 _[−]_[5] (z-score of at least 4), and _γ_ = 0 _._ 25, as in our experiments. For _n ≥_ 100, if propagation of the watermark through the RAG pipeline increases the ratio of green tokens by only 1%, it is already detectable with high confidence. This makes WARD viable, satisfying the _Guarantees_ requirement, unlike any baseline. It also contributes to _Monotonicity_ : assuming each _di_ has a green token ratio of _γ[′′]_ , which propagated through _M[⋆]_ reduces to _γ[′] ∈_ ( _γ, γ[′′]_ ), the p-value for _IN_ cases strictly decreases for more queries. In §5.2 we experimentally confirm that WARD scales consistently and efficiently in the number of queries. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0006-02.png)


**----- Start of picture text -----**<br>
0 . 34<br>0 . 32<br>0 . 30<br>0 . 28<br>0 . 26 γ [′]<br>0.25 No Watermark<br>0 50 100 150 200<br>Number of Queries<br>Ratio<br>Green<br>**----- End of picture text -----**<br>


Figure 3: Green ratio to pass watermark detection with given #queries. 

**Test validity and signal leakage** Another requirement for _Monotonicity_ is that the p-value for _OUT_ cases is well-calibrated, ensuring exceedingly rare false positives. As pointed out in prior work (Kirchenbauer et al., 2023; Sander et al., 2024), to ensure independence between tokens scored by the watermark detector, it is necessary to ignore duplicate _h_ -grams across _R_ . While this reduces the number of useful tokens per query, our results in §5 show that it does not affect WARD performance. 

For our statistical test to be valid, we must ensure that the watermark signal can only originate from the use of _D_ do in _D_ , i.e., that it is not otherwise _leaked_ to the responses _ri_ . In particular, we observed that watermarked queries, or queries based on watermarked versions of _di_ , drastically increase false positives. To confirm that WARD does not suffer from this issue, in §5.2, we verify that the p-values in _OUT_ cases are distributed in [0 _,_ 1] roughly as expected with no noticeable decrease in _|R|_ . 

**Scheme choice** The key parameter of red-green schemes is _h_ , the context width. In common applications of LLM watermarks, a low _h_ is not recommended, as it makes the watermark easy to steal via repeated queries (Kirchenbauer et al., 2024; Jovanovi´c et al., 2024). In the context of RAG-DI, this is less of a concern, as the data owner never exposes unconstrained query access to watermarked content, as is the case when watermarking LLMs. Thus, as low _h_ benefits watermark propagation through RAG, we use _h_ = 2. This contributes to high _Robustness_ of WARD, empirically validated in §5.2. We discuss other scheme parameters and present ablation studies over each in §5.4. 

**Practicality** Another aspect making WARD robust is that it is inconspicuous, as it operates on natural-looking documents, queries, and responses (confirmed in §5.3), through legitimate use of the RAG system—this makes potential attempts by the RAG provider to thwart malicious interactions less effective. This differs from baselines such as AAG, which directly reveal the intention to leak the information about _D_ . As a final advantage, we note that, in contrast to all baselines, WARD does not require any training or adaptation to RAG-DI, as LLM watermarks naturally apply to this task. 

## 5 EXPERIMENTAL EVALUATION 

We evaluate the RAG-DI baselines (§3.2) and WARD (§4) on the FARAD dataset (§3.1). §5.1 presents our main experiment. In §5.2 we focus on desiderata from §4, showing that only WARD does not violate them. In §5.3 we validate several key assumptions, and in §5.4 present additional ablations. 

**Setup** Our experimental setup follows §3: we use FARAD to define two evaluation settings, and in both evaluate _IN_ and _OUT_ cases, i.e., where the data owner’s data _is_ (resp. _is not_ ) contained in _D_ . We use _|D_ do _|_ = 200, and _|D|_ = 800 for FARAD- _Easy_ , and _|D|_ = 3000 for FARAD- _Hard_ (sampling detailed in App. B.1). We note that WARD only depends on _|D_ do _|_ , but not _|D|_ . We use several LLMs as _M_ : GPT3.5, CLAUDE3-HAIKU, and LLAMA3.1-70B, and vary the system prompt: we use a short _naive_ prompt ( _Naive-P_ ) with basic RAG instructions, and a longer _defense_ ( _Def-P_ ) prompt, which models a RAG provider that instructs the model to not regurgitate sources verbatim, and refuse attempts to learn about the exact LLM context (see App. F.1). Each experiment is run with 5 random seeds. To ensure a controlled setting, in our main experiments, we assume a perfect retrieval system that always retrieves the most relevant documents (see App. B.2 for a detailed explanation). In §5.3, 

6 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Claude 3 Haiku Farad Easy Farad Hard<br>NAIVE-P IN<br>NAIVE-P OUT<br>DEF-P IN<br>DEF-P OUT<br>Llama3.1-70b<br>NAIVE-P IN<br>NAIVE-P OUT<br>DEF-P IN<br>DEF-P OUT<br>GPT-3.5<br>NAIVE-P IN<br>NAIVE-P OUT<br>DEF-P IN<br>DEF-P OUT<br>✓Facts (✓) Aag (✓) Sib ✓Ward × Facts × Aag × Sib ✓Ward<br>**----- End of picture text -----**<br>


Figure 4: Evaluation of all methods on FARAD in both _Easy_ and _Hard_ settings, and with both _Naive-P_ and _Def-P_ system prompts. We run each method with 5 random seeds, resulting in 5 squares. A red square indicates a false negative in the _IN_ case, and a false positive in the _OUT_ case. All methods perform well in the _Easy_ setting, while only WARD consistently performs well in the _Hard_ setting. 

we show that WARD works equally well on a practical end-to-end RAG. If not specified otherwise (see §5.4), the RAG uses _k_ = 3 shots. For methods that utilize auxiliary LMs (including WARD, which starts by watermarking _D_ do) we use LLAMA3.1-8B, and to compute the cosine similarity in SIB we use paragraph-level ALL-MINILM-L6-V2 with BERTScore aggregation (Zhang et al., 2020). For WARD, we use _PositionPRF_ (Kirchenbauer et al., 2024), _h_ = 2, and _δ_ = 3 _._ 5, ablating these in §5.4. We list all prompts, and examples of FARAD samples and watermarked documents, in App. F. 

## 5.1 MAIN RESULTS 

We present our main results in Fig. 4, where we evaluate all RAG-DI baselines and WARD across several settings, models, prompts, and random seeds. We make several key observations. 

First, all baselines perform somewhat well in the _Easy_ setting (no fact redundancy). However, our extremely simple FACTS baseline obtains perfect results, outperforming both AAG and SIB. This emphasizes that traditional non-redundant datasets (see §3.1) fail to capture the complexity of RAGDI, and can provide an incomplete view of the capabilities of RAG-DI methods. We also observe that already in the _Easy_ setting, straightforward system prompt defenses significantly impact both AAG and SIB, leading to a noticeable increase in false negatives. We further investigate defenses in §5.2. 

In the _Hard_ setting, all baselines fail to perform consistently, inducing both false positives and negatives. This can be attributed to their shortcomings in handling fact redundancy: FACTS directly relies on facts, while SIB and AAG rely on semantic similarity influenced by factual content. Notably, only WARD achieves 100% accuracy across all settings, models, and system prompts, showing that despite the retrieval of documents with partially overlapping facts, watermarking provides a reliable signal for dataset inference. This backs up our claims regarding the importance of fact-redundancy for realistic RAG-DI evaluation and highlights the potential of watermarking as an approach to RAG-DI. 

## 5.2 DESIDERATA 

We next demonstrate how baselines violate the desiderata from §4, which is not the case for WARD. 

**Monotonicity** As stated in §4, RAG-DI methods should steadily improve with more queries, i.e., _|D_ do _|_ for WARD (see §5.4 for a generalization). We evaluate this by setting _|D_ do _| ∈{_ 20 _,_ 40 _, . . . ,_ 200 _}_ in FARAD- _Hard_ , presenting the results of WARD and SIB in Fig. 5. WARD improves consistently with _|D_ do _|_ , reaching perfect accuracy across all settings for at most 80 documents. In our extended discussion of efficiency in App. D we show that this translates to inexpensive API costs of below $1, for all closed-source LLMs we consider. In contrast, SIB, besides never reaching full accuracy, exhibits strongly varying accuracy over _|D_ do _|_ , often decreasing despite using more queries. This is a consequence of the need to adapt the MIA baselines to the RAG-DI setting—we find similar behavior across all baselines, and present an additional study of their decision thresholds in App. C.6. 

7 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0008-01.png)


**----- Start of picture text -----**<br>
GPT-3.5 (NAIVE-P) Llama3.1-70b (NAIVE-P) Claude 3 Haiku (NAIVE-P) GPT-3.5 (NAIVE-P) Llama3.1-70b (NAIVE-P) Claude 3 Haiku (NAIVE-P)<br>GPT-3.5 (DEF-P) Llama3.1-70b (DEF-P) Claude 3 Haiku (DEF-P) GPT-3.5 (DEF-P) Llama3.1-70b (DEF-P) Claude 3 Haiku (DEF-P)<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>0 . 4 0 . 4<br>0 . 2 0 . 2<br>0 . 0 0 . 0<br>25 50 75 100 125 150 175 200 25 50 75 100 125 150 175 200<br>Number of Queries Number of Queries<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 5: WARD (left) and SIB (right) accuracy as a function of _|D_ do _|_ (i.e., the number of queries) in the _Hard_ setting. WARD consistently improves, while SIB suffers from high variance in accuracy. 

Table 1: The p-values of WARD. We report the max p-values for _IN_ and min p-values for _OUT_ cases. 

|Table 1: Th|e p-values of W|ARD. We repo|rt the max p-va|lues for_IN_ an|lues for_IN_ an|lues for_IN_ an|d min p-values f|or_OUT_ cases.|or_OUT_ cases.|
|---|---|---|---|---|---|---|---|---|---|
|Data<br>Agg.|Claude|3 Haiku<br>NAIVE-P<br>_OUT_<br>_IN_|GPT-|3.5<br>NAIVE-P<br>_OUT_<br>_IN_|||Llama3.1|-70b<br>NAIVE-P<br>_OUT_<br>_IN_||
||DEF-P<br>_OUT_<br>_IN_||DEF-P<br>_OUT_<br>_IN_||||DEF-P<br>_OUT_<br>_IN_|||
|FARAD<br>_Easy_<br>min<br>max<br>FARAD<br>_Hard_<br>min<br>max|1_._90_e−_01<br>2_._07_e−_267<br>6_._83_e−_01<br>1_._57_e−_186<br>6_._03_e−_02<br>1_._35_e−_48<br>3_._99_e−_01<br>2_._15_e−_25|1_._38_e−_03<br>3_._47_e−_301<br>2_._83_e−_01<br>2_._19_e−_253<br>3_._82_e−_01<br>5_._08_e−_74<br>7_._34_e−_01<br>3_._16_e−_51|3_._28_e−_01<br>9_._53_e−_33<br><br>7_._72_e−_01<br>1_._54_e−_27<br><br>2_._04_e−_03<br>1_._12_e−_12<br><br>7_._80_e−_01<br>8_._44_e−_10<br>|2_._13_e−_01<br>1_._78_e−_78<br>9_._17_e−_01<br>6_._56_e−_60<br>2_._69_e−_01<br>1_._53_e−_22<br>7_._92_e−_01<br>1_._31_e−_15|||7_._54_e−_03<br>6_._20_e−_179<br><br>9_._55_e−_01<br>1_._52_e−_138<br><br>1_._18_e−_02<br>3_._94_e−_24<br><br>7_._47_e−_01<br>1_._10_e−_11<br>|1_._32_e−_01<br>0<br>5_._05_e−_01<br>0<br>6_._12_e−_02<br>1_._22_e−_122<br>3_._70_e−_01<br>2_._42_e−_96||
|10_−_12<br>10_−_10<br>10_−_8<br>10_−_6<br>10_−_4<br>10_−_2<br>100<br>p-value<br>**Guarantees**<br>In contrast to_all_baselines, WARD<br>inherits the guarantees of the watermarking<br>scheme, directly providing a p-value for each de-<br>cision. In Fig. 6, we see that p-values rapidly<br>decrease with more queries for_IN_, but stay con-<br>sistently close to the expected value for_OUT_. Fur-<br>ther, in Table1, we show the_max_p-values for_IN_<br>and_min_p-values for_OUT_ in our main experiment,<br>i.e., closest to a false negative/positive. All p-<br>values are orders of magnitude from our decision<br>||||||||||
|||||||||50<br>175<br>200<br>~~Thresh:~~ ~~_∼_3~~~~_e_~~_−_5||
|||||||||||
|||||||||||
|||||||||_−_5||
|||||||||~~Thresh:~~ ~~_∼_3~~~~_e_~~||
|||||||||||
|||||||||||
|||||||||||
|||||Watermarked<br>OUT<br>||||||
|||||IN||||||
|||||25<br>50|||75<br>100<br>125<br>1<br>Number of Queries|||



**Guarantees** In contrast to _all_ baselines, WARD inherits the guarantees of the watermarking scheme, directly providing a p-value for each decision. In Fig. 6, we see that p-values rapidly decrease with more queries for _IN_ , but stay consistently close to the expected value for _OUT_ . Further, in Table 1, we show the _max_ p-values for _IN_ and _min_ p-values for _OUT_ in our main experiment, i.e., closest to a false negative/positive. All p- values are orders of magnitude from our decision boundary of _≈_ 3 _e[−]_[5] (z-score of 4), highlighting robustness to Type 1 errors and the fact that LLM watermark signals persist through a RAG pipeline. 

Figure 6: P-values as a function of #queries in _IN_ (red) and _OUT_ (blue) cases. 

**Robustness** Finally, we evaluate the methods’ ro10 ~~Setting~~ bustness to different settings. In particular, we further DEF-P examine the _Def-P_ system prompt setting, shown 8 in Fig. 4 to often prevent all baselines from obtaining 6 any useful estimates. Notably, both AAG and SIB rely on the model’s _willingness_ to leak information 4 about its context, either explicitly or by regurgitating 2 similar content. To validate the effect of _Def-P_ , we show the longest token-string overlap between the 0 10 20 30 40 50 60 response of _M[⋆]_ and the SIB target document (Fig. 7), Max. Repetition Length observing strictly less overlap with _Def-P_ , inducing Figure 7: The effect of the _Def-P_ false negatives. Importantly, this does not prevent watermarks from propagating through the RAG pipeline, resulting in a reliable signal for WARD. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0008-08.png)


**----- Start of picture text -----**<br>
10 Setting<br>NAIVE-P<br>DEF-P<br>8<br>6<br>4<br>2<br>0<br>10 20 30 40 50 60<br>Max. Repetition Length<br>Count<br>**----- End of picture text -----**<br>


Figure 7: The effect of the _Def-P_ sysprompt. 

We additionally investigate a defense based on MEMFREE decoding (Ippolito et al., 2023). Namely, we adapt the decoding of _M[⋆]_ to strictly prevent _n_ -gram overlap with the retrieved documents (using _n_ = 10 as in Ippolito et al. (2023)), modeling an even stronger attempt by the provider to protect the RAG corpus. Our results (App. C.5), show that even in this setting, WARD achieves full accuracy. We further discuss the potential for more elaborate defenses against WARD in App. E.2. 

## 5.3 ADDITIONAL CONSIDERATIONS 

**Modeling retrieval** So far, our experiments have assumed an idealized case of perfect retrieval. We now justify this choice by running WARD on an end-to-end RAG system which uses OpenAI’s `text-embedding-3-large` document embeddings, with _k_ = 3 and cosine similarity metric. 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0009-01.png)


**----- Start of picture text -----**<br>
1 . 0 Queries per Document<br>1 2 4 5 10 20 50<br>Clean Grade<br>0 . 9<br>− 2<br>0 . 8 Metric − 4 Thresh: ∼ 3 e [−] [5]<br>PSP<br>Grade − 6<br>0 . 7 Accuracy<br>− 8<br>0 . 6 − 10<br>0 . 5 − 12<br>1 2 3 4 5 6 25 50 75 100 125 150 175 200<br>Delta Number of Queries<br>Metrics p-value<br>Log.<br>**----- End of picture text -----**<br>


Figure 9: Left: Accuracy and watermarked text quality for different _δ_ values of WARD on FARAD- _Hard_ . Right: Accuracy over the number of queries per document for WARD on FARAD- _Hard_ . 

As above, we repeat all runs with 5 seeds on both _Easy_ and _Hard_ , with _M_ = LLAMA3.1-70B and both prompts. We defer accuracy curves to App. C.3 but note that WARD achieves 100% accuracy across all settings. In Fig. 8, we show average p-values over the number of queries, further illustrating the reliability of WARD. We note that this simple RAG system was able to retrieve the targeted watermarked article in 93 _._ 6% of all requests, almost perfectly reducing this to our idealized setting. We further run all baselines with _Def-P_ in the _Hard_ setting, and confirm that the (rare) retrieval errors do not make them more compet- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0009-04.png)


**----- Start of picture text -----**<br>
10 [−] [2]<br>Thresh: ∼ 3 e [−] [5]<br>10 [−] [5]<br>10 [−] [8]<br>10 [−] [11]<br>10 [−] [14]<br>10 [−] [17]<br>Watermarked<br>10 [−] [20] OUT<br>IN<br>25 50 75 100 125 150 175 200<br>Number of Queries<br>p-value<br>**----- End of picture text -----**<br>


Figure 8: Average p-values against #queries in our imperfect retrieval experiment. 

itive, e.g., because the distracting documents are now less relevant. As in the corresponding part of Fig. 4, FACTS has 100% false positives, while SIB and AAG have 100% true negatives. 

**Text quality** Applying WARD requires the data owner Table 2: Quality evaluation of waterto watermark _D_ do, which we do by paraphrasing with a marked documents. watermarked model. While prior work shows that watermarks minimally impact text quality (Piet et al., 2023), **Original Paraphrased** ∆ we re-affirm this by judging the quality of a randomly Judge 0 _._ 903 0 _._ 898 5 _e[−]_[3] sampled subset of 100 resulting documents. As shown P-SP 1 _._ 000 0 _._ 933 0 _._ 07 in Table 2, paraphrased documents maintain high quality both under GPT4O ratings (see App. F.7) and a paraphrase quality metric P-SP (Wieting et al., 2022). 

We also evaluate the quality of the RAG system’s responses (with _Naive-P_ and CLAUDE3-HAIKU) on 200 samples. The average response quality is 0 _._ 9475 in _IN_ cases (watermarked context) and 0 _._ 9465 in _OUT_ cases, confirming that watermarked documents do not negatively impact response quality. 

## 5.4 ABLATIONS 

Lastly, we provide ablations for important hyperparameters of WARD and the RAG-DI setup. 

**Watermark parameters** As introduced in §2, red-green watermarks have 2 key parameters: the strength _δ_ and the context size _h_ . We ablate over their impact on WARD using LLAMA3.1-70B as _M_ with the _Def-P_ sysprompt. The results for _δ_ are shown in Fig. 9 (Left). We observe a _sweet spot_ of _δ ∈_ [2 _._ 5 _,_ 4 _._ 5] with high accuracy (unlike _δ <_ 2 _._ 5) and the minimal impact on text quality (unlike _δ >_ 4 _._ 5). Notably, experiments in App. C.2 show that _δ >_ 4 _._ 5 negatively affects both query efficiency and text utility. This range of _δ ∈_ [2 _._ 5 _,_ 4 _._ 5] directly aligns with recommendations in prior work (Kirchenbauer et al., 2024), and supports our choice of _δ_ = 3 _._ 5 for the main experiments. 

Regarding _h_ , as noted in §4 and prior work, larger values are expected to degrade robustness. We compare our choice of _h_ = 2 with _h_ = 4, noticing a slight decrease in accuracy (100% on _Easy_ , but 80% on _Hard_ ), confirmed by our plots of query efficiency—full results are deferred to App. C.2. 

9 

Published as a conference paper at ICLR 2025 

**RAG-DI parameters** The RAG-DI setting has a wide range of parameters that we so far have not explored. First, we ablate over _k_ , the number of retrieved documents—our results in App. C.1 show that WARD performs well independent of this parameter. Also, in App. C.1, we study the case where only a subset of _D_ do is included in _D_ , relaxing the assumptions of RAG-DI. While this naturally weakens the watermark signal, our results imply that WARD shows some robustness to this setting. 

Finally, while we generally assume a single query per document _di_ , we briefly study the potential of WARD to increase its query efficiency by extracting more value from each _di_ . For this we introduce a parameter _qpd_ ( _queries per document_ , _qpd_ = 1 in the main experiments), generating more questions for each _di_ . The main results for _qpd ∈{_ 1 _,_ 2 _,_ 4 _,_ 5 _,_ 10 _,_ 20 _,_ 50 _}_ are shown in Fig. 9 (Right), where we plot the average p-value over the total number of _queries_ . As expected, for high _qpd_ , additional queries for the same _di_ bring only marginal value—this is also confirmed in App. C.4, which shows how the effective number of tokens scored by the watermark detector increases slower for high _qpd_ , as we encounter many repeated n-grams. However, for lower values such as _qpd_ = 4, reducing the required _|D_ do _|_ 4 times, we observe strong results, suggesting a promising avenue for future work. 

## 6 RELATED WORK 

Closest related works to ours are Li et al. (2024) and Anderson et al. (2024), which propose membership inference (MI) for RAG that we adapt to RAG-DI in §3, and works that highlight the risk of MI in the related paradigm of in-context learning such as (Duan et al., 2023). Others study broader privacy and security aspects of RAG, such as poisoning to jailbreak the model or exfiltrate data (Zou et al., 2025; Xue et al., 2024; Chaudhari et al., 2024; Zeng et al., 2024), concerns similar to RAG-DI. 

**Passive MI/DI** The problems of membership inference (MI) (Shokri et al., 2017; Carlini et al., 2022) or dataset inference (DI) (Maini et al., 2021; Dziedzic et al., 2022) have been long studied on training data, as opposed to RAG corpora as in this work. Recent attempts to adapt these methods to LLMs (Duan et al., 2024; Das et al., 2024; Meeus et al., 2024; Maini et al., 2024) cite the challenge of rigorous evaluation, and primarily focus on graybox settings, citing the difficulty of inference attacks in the blackbox setting (Choquette-Choo et al., 2021), which is what we consider in RAG-DI. Another perspective on the problem of tracing data usage in model training is given by recent works on LLM data contamination (Dekoninck et al., 2024; Oren et al., 2024), none of which consider RAG. 

**Proactive MI/DI/model protection** Another approach to MI/DI is (as in this work) _proactive_ , e.g., by watermarking the data (Ren et al., 2024; Guo et al., 2023). However, only few works study LLMs, and only the ones cited above consider RAG. Wei et al. (2024b) focus on LLMs in a graybox setting, inserting random sequences or unicode substitutions to trace the data through training. Sander et al. (2024) find that LLM watermarks propagate through fine-tuning, but _D_ do has to make up 10% of the fine-tuning corpus for blackbox detection. RAG-DI relaxes this requirement, as RAG is unaffected by _|D|_ and degrades the signal much less than fine-tuning. Another orthogonal area is watermarking of _models_ against model stealing, often via backdoors (Adi et al., 2018; Zhao et al., 2023). 

**LLM Watermarking** Finally, we note that many works study red-green LLM watermarks (Kirchenbauer et al., 2023; 2024; Zhao et al., 2024; Hou et al., 2024), but also other approaches such as sampling modification (Kuditipudi et al., 2024; Christ et al., 2024), model-based watermarking (Liu et al., 2024), or watermarking in weights (Gu et al., 2024). We note that WARD could be combined with many of these, and leave this interesting direction to follow-up work. 

## 7 CONCLUSION 

We studied the problem of black-box RAG Dataset Inference (RAG-DI), where the goal is to detect unauthorized usage of a dataset in a RAG system. We formalized the problem, proposed a dataset and a set of baselines, and presented WARD, a method based on LLM watermarks, which provides rigorous statistical guarantees. Our evaluation showed that WARD outperforms the baselines, achieving perfect accuracy and high query efficiency and robustness. This establishes WARD as a practical tool that can be directly applied to protect the rights of data owners in current RAG systems. We hope our work provides a valuable foundation for future work on RAG-DI—interesting directions include combining WARD with other LLM watermarks, or designing watermarks specifically tailored to RAG-DI. 

10 

Published as a conference paper at ICLR 2025 

## ACKNOWLEDGEMENTS 

The work has received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI). 

## ETHICS STATEMENT 

We do not foresee any ethical concerns arising from our work. Our work primarily provides dataset owners with the ability to prove unauthorized usage of their data in a RAG system, which is a valuable tool for protecting intellectual property, and fighting _against_ unethical practices of large model providers. An undesirable side effect of such methods may come from false accusations—as our experiments in §5 demonstrate, our proposed method, WARD, strictly controls the rate of false positives as highly unlikely, and in all of our experiments we do not observe a single instance of a false positive. While our work does result in a new dataset, FARAD, this dataset is entirely synthetic as it is based on fictional articles (Monteiro et al., 2024), and thus does not raise any privacy concerns. 

## REPRODUCIBILITY STATEMENT 

To foster reproducibility, we provide details of our experimental setup in §5 and App. B.1, and in App. F include all prompts used in our experiments. We make our code and the FARAD dataset available at `https://github.com/eth-sri/ward` under an MIT license, and include configuration files for each experiment and a README file with instructions on how to reproduce our results. 

## REFERENCES 

- Yossi Adi, Carsten Baum, Moustapha Cissé, Benny Pinkas, and Joseph Keshet. Turning your weakness into a strength: Watermarking deep neural networks by backdooring. In _USENIX Security_ , 2018. 

Maya Anderson, Guy Amit, and Abigail Goldsteen. Is my data in your retrieval database? membership inference attacks against retrieval augmented generation. _arXiv_ , 2024. 

- Kar Balan, Alex Black, Simon Jenni, Andrew Gilbert, Andy Parsons, and John Collomosse. Decorait – decentralized opt-in/out registry for ai training, 2023. 

Blake Brittain. Stability ai, midjourney should face artists’ copyright case, judge says, 2024. `https://www.reuters.com/legal/litigation/ stability-ai-midjourney-should-face-artists-copyright-case-judge-says-2024-05-08/` . 

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr. Membership inference attacks from first principles. In _IEEE S&P_ , 2022. 

- Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, and Alina Oprea. Phantom: General trigger attacks on retrieval augmented language generation. _arXiv_ , 2024. 

Christopher A. Choquette-Choo, Florian Tramèr, Nicholas Carlini, and Nicolas Papernot. Label-only membership inference attacks. In _ICML_ , 2021. 

- Miranda Christ, Sam Gunn, and Or Zamir. Undetectable watermarks for language models. _COLT_ , 2024. 

Coalition for Content Provenance and Authenticity. Technical specification v1.3. technical report. c2pa., 2021. `https://c2pa.org/specifications/specifications/1.3/index.html` . 

- Council of the European Union. Directive (eu) 2019/790 of the european parliament and of the council on copyright and related rights in the digital single market and amending directives 96/9/ec and 2001/29/ec. 2019. 

11 

Published as a conference paper at ICLR 2025 

- Debeshee Das, Jie Zhang, and Florian Tramèr. Blind baselines beat membership inference attacks for foundation models. _arXiv_ , 2024. 

- Jasper Dekoninck, Mark Niklas Müller, Maximilian Baader, Marc Fischer, and Martin T. Vechev. Evading data contamination detection for language models is (too) easy. _arXiv_ , 2024. 

- Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. Flocks of stochastic parrots: Differentially private prompt learning for large language models. In _NeurIPS_ , 2023. 

- Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. Do membership inference attacks work on large language models? _COLM_ , 2024. 

- Adam Dziedzic, Haonan Duan, Muhammad Ahmad Kaleem, Nikita Dhawan, Jonas Guan, Yannis Cattan, Franziska Boenisch, and Nicolas Papernot. Dataset inference for self-supervised models. In _NeurIPS_ , 2022. 

- European Parliament and the Council of the European Union. Proposal for a regulation of the european parliament and of the council laying down harmonised rules on artificial intelligence (artificial intelligence act) and amending certain union legislative acts - analysis of the final compromise text with a view to agreement. 2019. 

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. _arXiv_ , 2023. 

- Paul Goldstein, Christiane Stuetzle, and Susan Bischoff. Kneschke vs. laion - landmark ruling on tdm exceptions for ai training data, 2024. `https://copyrightblog.kluweriplaw.com/2024/11/13/ kneschke-vs-laion-landmark-ruling-on-tdm-exceptions-for-ai-training-data-part-1/` , last accessed: Feb 5 2025. 

- Michael M. Grynbaum and Ryan Mac. The times sues openai and microsoft over a.i. use of copyrighted work, 2023. `https://www.nytimes.com/2023/12/27/business/media/ new-york-times-open-ai-microsoft-lawsuit.html` . 

- Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto. On the learnability of watermarks for language models. _ICLR_ , 2024. 

- Junfeng Guo, Yiming Li, Lixu Wang, Shu-Tao Xia, Heng Huang, Cong Liu, and Bo Li. Domain watermark: Effective and harmless dataset copyright protection is closed at hand. In _NeurIPS_ , 2023. 

- Abe Bohan Hou, Jingyu Zhang, Tianxing He, Yichen Wang, Yung-Sung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, and Yulia Tsvetkov. Semstamp: A semantic watermark with paraphrastic robustness for text generation. _NAACL Long_ , 2024. 

- Daphne Ippolito, Florian Tramèr, Milad Nasr, Chiyuan Zhang, Matthew Jagielski, Katherine Lee, Christopher A. Choquette-Choo, and Nicholas Carlini. Preventing generation of verbatim memorization in language models gives a false sense of privacy. In _INLG_ , 2023. 

- Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. Flashrag: A modular toolkit for efficient retrieval-augmented generation research. _arXiv_ , 2024. 

- Nikola Jovanovi´c, Robin Staab, and Martin Vechev. Watermark stealing in large language models. _ICML_ , 2024. 

- A. Feder Cooper Katherine Lee, Daphne Ippolito. The devil is in the training data, 2023. `https: //blog.genlaw.org/explainers/training-data.html` , last accessed: Feb 5 2025. 

- Paul Keller and Zuzanna Warso. Defining best practices for opting out of ml training. _Open Future Policy Brief 5_ , 2023. 

- John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In _ICML_ , 2023. 

12 

Published as a conference paper at ICLR 2025 

- John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, and Tom Goldstein. On the reliability of watermarks for large language models. _ICLR_ , 2024. 

- Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. In _European conference on machine learning_ , 2004. 

- Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. _TMLR 05/2024_ , 2024. 

- Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _NeurIPS_ , 2020. 

- Yuying Li, Gaoyang Liu, Yang Yang, and Chen Wang. Seeing is believing: Black-box membership inference attacks against retrieval augmented generation. _arXiv_ , 2024. 

- Aiwei Liu, Leyi Pan, Xuming Hu, Shuang Li, Lijie Wen, Irwin King, and Philip S. Yu. A private watermark for large language models. _ICLR_ , 2024. 

- Shayne Longpre, Robert Mahari, Anthony Chen, Naana Obeng-Marnu, Damien Sileo, William Brannon, Niklas Muennighoff, Nathan Khazam, Jad Kabbara, Kartik Perisetla, Xinyi Wu, Enrico Shippole, Kurt Bollacker, Tongshuang Wu, Luis Villa, Sandy Pentland, and Sara Hooker. The data provenance initiative: A large scale audit of dataset licensing & attribution in ai. _arXiv_ , 2023. 

- Pratyush Maini, Mohammad Yaghini, and Nicolas Papernot. Dataset inference: Ownership resolution in machine learning. In _ICLR_ , 2021. 

- Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. LLM dataset inference: Did you train on my dataset? _NeurIPS_ , 2024. 

- Matthieu Meeus, Shubham Jain, Marek Rei, and Yves-Alexandre de Montjoye. Inherent challenges of post-hoc membership inference for large language models. _arXiv_ , 2024. 

- Joao Monteiro, Pierre-Andre Noel, Etienne Marcotte, Sai Rajeswar, Valentina Zantedeschi, David Vazquez, Nicolas Chapados, Christopher Pal, and Perouz Taslakian. Repliqa: A question-answering dataset for benchmarking llms on unseen reference content. _NeurIPS_ , 2024. 

- Khalil Mrini, Franck Dernoncourt, Walter Chang, Emilia Farcas, and Ndapandula Nakashole. Joint summarization-entailment optimization for consumer health question understanding. In _ACL workshop on NLP for medical conversations_ , 2021. 

- Yonatan Oren, Nicole Meister, Niladri S. Chatterji, Faisal Ladhak, and Tatsunori Hashimoto. Proving test set contamination in black-box language models. In _ICLR_ , 2024. 

Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, and David A. Wagner. Mark my words: Analyzing and evaluating language model watermarks. _arXiv_ , 2023. 

- Joao Pedro Quintais. Generative ai, copyright and the ai act, 2023. `https://copyrightblog. kluweriplaw.com/2023/05/09/generative-ai-copyright-and-the-ai-act/` , last accessed: Feb 5 2025. 

- Jie Ren, Yingqian Cui, Chen Chen, Vikash Sehwag, Yue Xing, Jiliang Tang, and Lingjuan Lyu. Entruth: Enhancing the traceability of unauthorized dataset usage in text-to-image diffusion models with minimal and robust alterations. _arXiv_ , 2024. 

- Tom Sander, Pierre Fernandez, Alain Durmus, Matthijs Douze, and Teddy Furon. Watermarking makes language models radioactive. _NeurIPS_ , 2024. 

- Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In _IEEE S&P_ , 2017. 

13 

Published as a conference paper at ICLR 2025 

SpawningAI. Have i been trained?, 2022. `https://haveibeentrained.com/` , last accessed: Feb 5 2025. 

SpawningAI. ai.txt, 2023. `https://spawning.ai/ai-txt` . 

SpawningAI. Spawningai, 2024. `https://spawning.ai/` , last visited: Nov 19 2024. 

TDM Community Group. Tdm reservation protocol (tdmrep) - final community group report, 2024. 

Johnny Tian-Zheng Wei, Ryan Yixiang Wang, and Robin Jia. Proving membership in LLM pretraining data via data watermarks. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _ACL_ , 2024a. 

- Johnny Tian-Zheng Wei, Ryan Yixiang Wang, and Robin Jia. Proving membership in LLM pretraining data via data watermarks. In _ACL (Findings)_ , 2024b. 

- John Wieting, Kevin Gimpel, Graham Neubig, and Taylor Berg-Kirkpatrick. Paraphrastic representations at scale. In _EMNLP (Demos)_ , 2022. 

- Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, and Qian Lou. Badrag: Identifying vulnerabilities in retrieval augmented generation of large language models. _arXiv_ , 2024. 

- Guangtao Zeng, Wenmian Yang, Zeqian Ju, Yue Yang, Sicheng Wang, Ruisi Zhang, Meng Zhou, Jiaqi Zeng, Xiangyu Dong, Ruoyu Zhang, Hongchao Fang, Penghui Zhu, Shu Chen, and Pengtao Xie. MedDialog: Large-scale medical dialogue datasets. In _EMNLP_ , 2020. 

- Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, et al. The good and the bad: Exploring privacy issues in retrieval-augmented generation (rag). _ACL Findings_ , 2024. 

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with BERT. In _ICLR_ , 2020. 

- Xuandong Zhao, Yu-Xiang Wang, and Lei Li. Protecting language generation models via invisible watermarking. In _ICML_ , 2023. 

- Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. _ICLR_ , 2024. 

- Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. Poisonedrag: Knowledge poisoning attacks to retrieval-augmented generation of large language models. _USENIX Security_ , 2025. 

14 

Published as a conference paper at ICLR 2025 

## A MOTIVATION: THE IMPORTANCE OF RAG-DI 

Extending on our discussion in §1, we elaborate on the motivation for establishing RAG-DI as a problem setting, the prevalence of the problem of unauthorized data usage (studied in this work), the need for a solution like WARD, and its practical applicability. 

**Protecting data from use in Generative AI** The unauthorized usage of data in Generative AI (GenAI) systems, primarily for training, has become an especially pressing concern in recent years, as the practice of indiscriminate scraping of massive datasets from the web became more commonplace (Katherine Lee, 2023). This has led to a crisis in data transparency, where the provenance of data used to train models is often hard to track (Longpre et al., 2023), and commonly not reported by the model provider. While the EU AI Act (Council of the European Union, 2019) attempts to remedy this, by mandating that all data sources used for model training are published, this regulatory process is slow, and data owners must take active steps to protect themselves. 

**Opt-outs and legal uncertainties** One tool that data owners can use to exercise agency over the use of their work comes in the form of opt-outs, where creators can choose to exclude their works from GenAI training. Notably, the Article 4 of the EU’s Copyright in Digital Single Market (CDSM) Directive (European Parliament and the Council of the European Union, 2019) recognizes the right of creators to exercise such an opt-out, by providing a machine-readable expression of the reservation of rights. Yet, the practical implementation of this opt-out mechanism is still unclear, and an active discussion from both law (Quintais, 2023) and research (Balan et al., 2023) communities is ongoing. Many initiatives have established standards for expressing such reservations (Keller & Warso, 2023), such as `ai.txt` (SpawningAI, 2023), the TDM Reservation Protocol (TDM Community Group, 2024), and C2PA (Coalition for Content Provenance and Authenticity, 2021), a metadata standard for embedding content provenance information in media files which also includes a flag for opting out of GenAI training and/or inference. 

While opt-outs are picking up traction, and some GenAI providers have pledged to honor them (SpawningAI, 2024), they are still far from being universally respected, and it is yet unclear how to practically enforce them (or audit compliance). More broadly, the legal landscape around GenAI data usage is complex, and both legal and research communities are closely following several prominent lawsuits. Notable examples include NYT vs OpenAI/Microsoft (Grynbaum & Mac, 2023) or a group of artists vs Stability/MidJourney/DeviantArt/RunwayML (Brittain, 2024), which illustrate that the legal system is naturally still catching up to the rapid development of GenAI. The key challenges are in untangling the interplay between licenses, traditional copyright, opt-outs, the ideas such as fair use or market harm, and technical concepts such as LLM memorization. The most relevant for our discussion is the case Kneschke vs LAION (Goldstein et al., 2024), where the ruling has extensively engaged with the meaning of opt-outs and the extent of their enforceability. 

**Proving unauthorized usage** In this landscape of legal uncertainty, the ability to _provably detect_ unauthorized usage of data in GenAI systems is crucial for data owners, as it may provide a way to enforce their rights in court. While this is relatively simple when model providers make their datasets public (SpawningAI, 2022), the problem becomes significantly harder for the common case of undisclosed training data, leading to works on proactive membership/dataset inference such as Wei et al. (2024a); Sander et al. (2024) and others we cite in §6. As noted in §1, a particular flavor of this problem comes in the context of LLMs, where RAG systems have become a common way to integrate new knowledge into the model without costly fine-tuning. 

**WARD** This is the problem statement studied in this work and tackled by WARD, with our requirements (§3) directly inspired by the legal and practical challenges data owners face in this context. Namely, _Guarantees_ are crucial to be able to use the results as evidence, and _Robustness_ is important to ensure that the method is not easily circumvented, unintentionally or by attempts to defend against it. Finally, _Monotonocity_ and the corresponding _Efficiency_ are needed for the method to be practically feasible. While the data owner naturally must query every system they suspect is using their data in the RAG corpus, each such use is very cheap (below $2 per LLM, see App. D). Further, we argue that the ecosystem of relevant LLM providers is relatively small. In particular, not many providers both (i) have resources for indiscriminate large-scale scraping of data and (ii) are sufficiently popular to create market harm based on the unauthorized usage of that data. Thus, the cost of WARD is practical. 

15 

Published as a conference paper at ICLR 2025 

## B ADDITIONAL EXPERIMENTAL DETAILS 

We expand on the details of our experimental setup provided in the main paper, by providing a detailed explanation of how we sample FARAD to create our evaluation settings (App. B.1), and how we simulate perfect retrieval in our experiments (App. B.2). 

## B.1 SAMPLING FARAD FOR EVALUATION 

Recall that each FARAD _group_ contains 4 articles, one for each of the LLM authors we consider. For brevity, we will use _Ai_ for _i ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ to denote the _i_ -th author. For the _Easy_ setting, we sample four subsets of distinct groups, where the sizes of the subsets are respectively (200 _,_ 300 _,_ 300 _,_ 200). Then for each subset _i ∈{_ 1 _,_ 2 _,_ 3 _}_ , we only take articles from _Ai_ , and include all of them in the RAG corpus _D_ . Out of those, articles from subset 1 of author _A_ 1 are taken as _D_ do in the _IN_ case, i.e., these are potentially modified by the service provider before inserting them into the RAG corpus. Similarly, the articles from subset 4 of author _A_ 4 are reserved as _D_ do in the _OUT_ case. This setup ensures no fact redundancy in the RAG corpus. 

In contrast, to create the _Hard_ setting with fact redundancy, we start by sampling 1000 distinct groups. The RAG corpus _D_ is then built by including all articles from those groups that were written by _A_ 1, _A_ 2, and _A_ 3. A randomly subsampled set of 200 of those articles, that were written by _A_ 1, is taken as _D_ do in the _IN_ case. Similarly, randomly sampling 200 of the 1000 above groups, and taking documents from _A_ 4 out of each group is used as _D_ do in the _OUT_ case. Due to practical limitations, before running all of our experiments we managed to generate only the first 1000 out of 3391 groups in FARAD, thus our sampling is done w.r.t. these 1000 groups. 

## B.2 SIMULATING PERFECT RETRIEVAL 

We provide a detailed description of the perfect retrieval mechanism used in our main experiments, as introduced in §5, and further compared to a real retrieval mechanism in §5.3, where we have demonstrated that it almost perfectly matches the idealized case described in this section and shown that WARD performs equally well in both cases. Assume the data owner is querying _M[⋆]_ in order to test for presence of a document _d_ . We construct the following sequence of documents: 

- First, the document _d_ itself if it is present in the RAG corpus. 

- Next, a random shuffle of all documents _d[′]_ from the same FARAD group that are present in the RAG corpus, i.e., the documents produced by other _authors_ based on the same _source_ . 

- Finally, a random shuffle of all other documents from the RAG corpus. 

Given the parameter _k_ of desired _shots_ (ablated in Fig. 11), the perfect retrieval returns the first _k_ documents from this sequence. Note that for WARD and _k >_ 1 this always results in additional documents in the context beyond the watermarked one, where such distractions often come from both the same source, as well as different sources. We further illustrate the behavior of perfect retrieval with two examples. 

**First example** The RAG provider populates the corpus _D_ according to the _Easy_ setting, and uses _k_ = 3. The user applies WARD to query _M[⋆]_ with a question related to the (watermarked) document #123 _a_ , i.e., from FARAD group #123, written by the author _a_ . Assume #123 _a ∈ D_ . The retrieved documents are #123 _a_ , #374 _b_ , #266 _a_ . Following the above steps, we include #123 _a_ as it is present in the corpus, and two documents chosen at random, as in the _Easy_ setting the corpus includes no other document from group 123. 

**Second example** Now assume the _Hard_ setting and _k_ = 5 shots. The user queries _M[⋆]_ with a question related to #123 _d_ . Assume #123 _d ∈/ D_ . The retrieved documents are #123 _a,_ #123 _b,_ #123 _c,_ #597 _c,_ #442 _b_ . Per above steps, as the target document #123 _d_ is not in the corpus, we include all other documents from group #123 and 2 documents chosen at random. 

16 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0017-01.png)


**----- Start of picture text -----**<br>
Dropout Percentage Dropout Percentage<br>0.0 20.0 40.0 60.0 70.0 80.0 90.0 0.0 20.0 40.0 60.0 70.0 80.0 90.0<br>10.0 30.0 50.0 10.0 30.0 50.0<br>0 Thresh: ∼ 3 e [−] [5] 0<br>− 25 − 2<br>− 50 − 4 Thresh: ∼ 3 e [−] [5]<br>− 6<br>− 75<br>− 8<br>− 100 − 10<br>− 125 − 12<br>− 150 − 14<br>− 175 − 16<br>25 50 75 100 125 150 175 200 25 50 75 100 125 150 175 200<br>Number of Queries Number of Queries<br>p-value p-value<br>Log. Log.<br>**----- End of picture text -----**<br>


Figure 10: Effect of the percentage of documents of _D_ do **not** contained in _D_ on the p-value for LLAMA3.1-70B using _Naive-P_ (left) and GPT3.5 using _Def-P_ (right) on FARAD- _Hard_ . While in both cases we see how WARD is robust to an increase in the percentage of documents contained in _D_ , the _Def-P_ setting is more sensitive, showing initial false positives at 20%. 

## C ADDITIONAL EXPERIMENTAL RESULTS 

In this section, we first present additional results on our RAG-DI ablations, omitted from §5.4 (App. C.1), Similarly, in App. C.2, we extend our WARD ablations from §5.4. In App. C.3 we provide extended results of our retrieval study from §5.3, and in App. C.4 extend our results on the number of queries per document from Fig. 9 (Right). Finally, in App. C.5 we provide results on an additional defense (MEMFREE, summarized in §5.2), and in App. C.6 we provide more insights into the performance of RAG-DI baselines. 

## C.1 RAG-DI ABLATIONS 

As summarized in §5.4, we ablate over key parameters of RAG-DI: The number of documents (shots) put into the context of model ( _k_ ) and the fraction of documents of _D_ do contained in _D_ . 

**Number of shots** For _k_ we present our results for WARD in Fig. 11, using _k ∈{_ 3 _,_ 4 _,_ 5 _}_ for our _Naive-P_ and _Def-P_ settings on FARAD- _Hard_ using GPT3.5. Across all values of _k_ , WARD shows favorable scaling of the number of queries made to the system, reaching 100% accuracy at 100 queries at the latest. This follows intuition, as the scaling of WARD is primarily influenced by the capabilities of the underlying LLM to select the correct information from a set of retrieved documents. The constant improvement in model capabilities, therefore, naturally positively impacts the scaling of WARD w.r.t. context size. In all cases, WARD reaches 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0017-08.png)


**----- Start of picture text -----**<br>
1 . 0<br>0 . 9<br>0 . 8<br>0 . 7 Setting<br>GPT-3.5 (NAIVE-P)<br>GPT-3.5 (DEF-P)<br>0 . 6 shots<br>3<br>4<br>0 . 5 5<br>25 50 75 100 125 150 175 200<br>Number of Queries<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 11: Accuracy over the number of queries when varying the number of retrieved documents in RAG (shots, _k_ ) for WARD on FARAD- _Hard_ . We observe, on GPT3.5, that WARD scales well with the number of documents for both the _Naive-P_ and _Def-P_ system prompts. 

100% accuracy at a rate matching that of our main experiments. While the above values of _k_ are generally recommended (Jin et al., 2024), and used in prior RAG MIA work (Li et al., 2024; Anderson et al., 2024), we additionally experiment with a much larger _k_ = 10, switching to CLAUDE3-HAIKU (as this experiment exceeds GPT3.5’s context size). In all cases, WARD reaches 100% accuracy at a rate matching that of our main experiments. 

**Percentage of documents in the corpus** We further ablate over the percentage of documents of _D_ do contained in _D_ , which our main experiments assume to be 100. For this extension of the RAG-DI setting, we assume that while the data owner wants to check whether _D_ do _⊆ D_ , in reality only a strict subset _D_ do _[′][⊂][D]_[do][ with size] _[ |][D]_ do _[′][|]_[ = (1] _[ −][ω]_[)] _[ · |][D]_[do] _[|]_[ is contained in] _[ D]_[.][Naturally, as the data owner in] WARD incorporates all queries for all documents, the resulting queries related to documents not from _D_ do _[′]_[will increase the p-value.][We experimentally test this in two settings:][once with L][LAMA][3.1-70B] 

17 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0018-01.png)


**----- Start of picture text -----**<br>
Watermark- δ 1 . 0<br>0.5 1.5 2.5 3.5 4.5 5.5 6.5<br>1.0 2.0 3.0 4.0 5.0 6.0<br>1 . 0 0 . 9<br>0 . 9<br>0 . 8<br>0 . 8<br>0 . 7<br>0 . 7 Setting<br>Llama3.1-70b (NAIVE-P)<br>0 . 6 Llama3.1-70b (DEF-P)<br>0 . 6 Watermark-h<br>2<br>0 . 5 0 . 5 4<br>25 50 75 100 125 150 175 200 25 50 75 100 125 150 175 200<br>Number of Queries Number of Queries<br>Accuracy Accuracy<br>Average<br>**----- End of picture text -----**<br>


Figure 12: Average _cumulative_ accuracy across _δ_ s for WARD with _Naive-P_ on FARAD- _Hard_ , where the model is LLAMA3.1-70B. We observe an optimum at _δ ≈_ 4. 

Figure 13: Accuracy for _h ∈{_ 2 _,_ 4 _}_ using WARD on LLAMA3.1-70B on FARAD- _Hard_ . We observe that WARD requires more samples to reliably detect unauthorized usage at higher _h_ . 

and _Naive-P_ , and once with GPT3.5 and _Def-P_ , presenting the mean _log_ 10 p-value of the number of queries for both settings in Fig. 10. Across both cases, we observe that (as expected) a higher value of _ω_ results in a higher computed p-value and hence a weaker test. At the same time, we can see that, especially in easier settings, WARD can endure a significant drop (only 20% of _D_ do being contained in _D_ ) while still providing accurate results. In the more challenging setting, we observe the first false positives at _ω_ = 0 _._ 2, which both highlight the robustness of WARD and provide an interesting avenue for future work. 

## C.2 WARD ABLATIONS 

To supplement our results in §5.4, we present full results of our ablation experiments of key watermarking parameters in WARD: the watermarking strength _δ_ and the context size _h_ . 

**Watermark strength** We ablate _δ ∈_ [0 _._ 5 _,_ 6 _._ 5] in steps of 0 _._ 5 using LLAMA3.1-70B with _DefP_ on FARAD- _Hard_ . In particular, in Fig. 12, we complement our quality-accuracy plot from §5.4, by displaying the _cumulative_ average accuracy at each number of queries. Concretely, we, at point _x_ , present the average accuracy of all previous _x[′] ≤ x_ . This highlights two key findings. First, while many higher _δ_ s achieve 100% accuracy in our plot in Fig. 9 (as number of queries is 200), they actually achieve worse results for a lower number of queries. This can be explained by the fact that higher _δ_ s lead to worse text quality, which in turn impacts the text quality in the final LLM responses, and thus reduces the amount of watermark signal that is transferred. Second, these results narrow the optimal range of _δ_ for our setting to [3 _._ 5 _,_ 4 _._ 5], on which we consistently achieve the best results. 

**Context size** Further, we ablate over the watermark context size _h_ , presenting additional results on _h_ = 4 in Fig. 13 (LLAMA3.1-70B on _Naive-P_ ). Notably, we find, in line with prior work on watermarks showing that increases in _h_ produce less robust watermarks, that WARD requires more samples to reliably detect higher _h_ = 4 (compared to _h_ = 2). While on _Easy_ , this has only a minor im- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0018-09.png)


**----- Start of picture text -----**<br>
Llama3.1-70b (NAIVE-P) Llama3.1-70b (DEF-P)<br>1 . 0<br>0 . 8<br>0 . 6<br>0 . 4<br>0 . 2<br>0 . 0<br>25 50 75 100 125 150 175 200<br>Number of Queries<br>Llama3.1-70b (NAIVE-P) Llama3.1-70b (DEF-P)<br>1 . 0<br>0 . 8<br>0 . 6<br>0 . 4<br>0 . 2<br>0 . 0<br>25 50 75 100 125 150 175 200<br>Number of Queries<br>Accuracy<br>Accuracy<br>**----- End of picture text -----**<br>


Figure 14: Accuracy of WARD on FARAD- _Easy_ (Top) and FARAD- _Hard_ (Bottom) using a full RAG system with LLAMA3.1-70B. As in our perfect retriever setting, we observe that WARD requires only a small number of queries to achieve 100% accuracy in both settings. 

18 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0019-01.png)


**----- Start of picture text -----**<br>
1 . 00 10 NAIVE-PSetting<br>0 . 95 DEF-P<br>MEM-FREE<br>0 . 90 8<br>0 . 85<br>6<br>0 . 80<br>0 . 75 Setting 4<br>Llama3.1-70b (NAIVE-P)<br>0 . 70 Llama3.1-70b (DEF-P)<br>0 . 65 Llama3.1-70b (MEM-FREE) 2<br>Watermark-h<br>0 . 60 2 0<br>25 50 75 100 125 150 175 200 10 15 20 25 30<br>Number of Queries Max. Repetition Length<br>Accuracy Count<br>**----- End of picture text -----**<br>


Figure 15: Experiments with MEMFREE. **Left** : Accuracy of WARD on FARAD- _Hard_ and LLAMA3.170B under the MEMFREE defense. Despite this more radical defense we observe that WARD requires only 60 queries to achieve 100% accuracy. **Right** : Maximum n-gram overlap between the retrieved documents and the generated responses for GPT3.5 on FARAD- _Hard_ under the MEMFREE defense. 

pact, we see a stronger initial drop on _Hard_ . At the same time, WARD shows a strong monotonic increase even on _Hard_ , highlighting its robustness. 

## C.3 END-TO-END RAG 

Next, we complement our results on the full RAG implementation from §5.3 in Fig. 14. For this, we show the full-accuracy curves on both FARAD- _Easy_ and FARAD- _Hard_ using an end-to-end RAG system as described in §5.3. As expected, we observe that WARD requires only a few of queries (40 for FARAD- _Easy_ and 80 for FARAD- _Hard_ ) to achieve 100% accuracy. This not only confirms our optimal retriever assumption in §5 but also highlights how WARD is practical in real-world settings. 

## C.4 NUMBER OF QUERIES PER DOCUMENT 

As presented in §5.4, a data owner could phrase multiple queries per document in _D_ do to improve efficiency. While we presented generally diminishing returns under a fixed query budget in Fig. 9, we reaffirm this here by showing the actual number of tokens present in the resulting outputs that are actually _scored_ by the watermark detector after deduplication. For this, we assume the _IN_ case using GPT3.5 and _DefP_ . We show the corresponding plot in Fig. 16 (log scale), and note that while we observe consistently linear scaling across all numbers of queries per document, reusing the same document too many times can lead to a decrease in the number of usable tokens (around 2x for the highest setting of 50 queries per document). This is in line with our intuition, as obtaining 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0019-08.png)


**----- Start of picture text -----**<br>
Queries per Document<br>1 2 4 5 10 20 50<br>70000<br>60000<br>50000<br>40000<br>30000<br>20000<br>10000<br>900080007000<br>6000<br>5000<br>4000<br>3000<br>2000<br>1000<br>25 50 75 100 125 150 175 200<br>Number of Queries<br>Tokens<br>of<br>Number<br>**----- End of picture text -----**<br>


Figure 16: Number of usable tokens over #queries for varying number of _queries per document_ in the _IN_ case, using GPT3.5 and _Def-P_ in FARAD- _Hard_ . 

unique watermarked n-grams is limited by the watermark of the original document that is present in the RAG corpus, and those tokens eventually get exhausted. Still, as our results above have shown, a number of queries per document can likely be increased to at least 4 to lead to a multiplicative increase in efficiency in terms of needed dataset size. 

## C.5 MEMFREE DEFENSE 

Inspired by (Ippolito et al., 2023), we evaluate WARD on an additional defense, MEMFREE, which prevents the RAG system from producing any output that has a certain n-gram overlap with any of the retrieved documents. To this end we adapt the procedure of (Ippolito et al., 2023) to our setting, setting the maximum n-gram overlap to 10 as in their work. We can directly observe the effectiveness 

19 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0020-01.png)


**----- Start of picture text -----**<br>
GPT-3.5 (NAIVE-P) Llama3.1-70b (NAIVE-P) Claude 3 Haiku (NAIVE-P) GPT-3.5 (NAIVE-P) Llama3.1-70b (NAIVE-P) Claude 3 Haiku (NAIVE-P)<br>GPT-3.5 (DEF-P) Llama3.1-70b (DEF-P) Claude 3 Haiku (DEF-P) GPT-3.5 (DEF-P) Llama3.1-70b (DEF-P) Claude 3 Haiku (DEF-P)<br>1 . 0 1 . 0<br>0 . 9<br>0 . 9<br>0 . 8<br>0 . 8 0 . 7<br>0 . 7 0 . 6<br>0 . 5<br>0 . 6<br>0 . 4<br>0 . 5 0 . 3<br>0 . 0 0 . 2 0 . 4 0 . 6 0 . 8 1 . 0 0 . 0 0 . 2 0 . 4 0 . 6 0 . 8 1 . 0<br>Threshold Threshold<br>Accuracy Accuracy<br>**----- End of picture text -----**<br>


Figure 17: Accuracy of SIB on FARAD- _Easy_ (left) and FARAD- _Hard_ (right) using all settings. We show the nontrivial (gray) and optimal (green) thresholds for SIB on FARAD- _Easy_ and FARAD- _Hard_ . We find that while a single threshold is viable for FARAD- _Easy_ , it is not for FARAD- _Hard_ . 

of this defense in reducing n-gram overlap in Fig. 15 (Right). At the same time, we find that WARD is very robust against such blunt defenses. In particular, as we show in Fig. 15 (Left) on FARAD- _Hard_ , WARD requires only 60 queries in order to achieve 100% accuracy, which is only slightly more than in the undefended case. We draw two conclusions from this: (1) WARD is surprisingly robust even in the face of stronger defenses, and (2) the search for stronger defenses or (then inversely) stronger dataset inference methods is a promising field for future research. 

## C.6 FURTHER STUDY OF BASELINES 

All our baselines introduced in §3.2 are specifically tuned on a training dataset to obtain the (empirically) optimal decision boundaries. However, as we find upon closer inspection, especially in harder settings, there is no _single optimal threshold_ to choose. We substantiate this in Fig. 17, where on SIB, we show two regions w.r.t. the threshold: _nontrivial_ (gray; _>_ 50% accuracy for all settings/models, note that obtaining 50% is trivial by simply always making the same decision, that _D_ do is in _D_ or otherwise), and _optimal_ (green; 100% for all settings/models). 

Notably, while we find that we can find an optimal threshold on FARAD- _Easy_ around 0 _._ 4, finding a single such threshold for FARAD- _Hard_ is not possible. We observe this for all baseline methods in FARAD- _Hard_ , which helps explain why they fail to achieve a consistently high accuracy across our main experiments (where it is sensible to choose only a single threshold). 

## D EFFICIENCY OF WARD 

While not a separate point in the list of design requirements we introduce in §3, we have extensively studied _Efficiency_ through the requirement of _Monotonocity_ , requiring that the accuracy of WARD increases _fast_ with more queries. Namely, in Fig. 3 we have discussed the efficiency of WARD on an intuitive level, and have confirmed it experimentally in §5.2, where most of our experiments explore scaling with the number of queries, showing that at most 100 queries to the RAG system are sufficient for a confident decision across all settings, models and defenses. 

While we have focused on the number of queries as the primary metric, we can translate this into a more practical measure of API costs. We empirically observe that the queries _qi_ sent by WARD to the RAG system _M[⋆]_ are 30 to 80 tokens long, and the RAG system responses _ri_ = _M[⋆]_ ( _qi_ ) are 400 to 500 tokens long. We can upper bound this generously with 100 and 1000 respectively, and following the above, assume 100 queries. Given the November 2024 costs of GPT-3.5 ($3 per 1M input tokens, $6 per 1M output tokens), this results in a cost of only $0 _._ 63 for the whole process of RAG-DI. Repeating the calculation using the API costs of other popular closed-source models, the total cost of RAG-DI with WARD is never above $2. 

An important consideration is also how the efficiency of WARD is impacted by the size of the RAG system’s corpus _D_ and the data owner’s dataset _D_ do. First, as we note in §6, in contrast to DI attacks on training data, WARD is not significantly affected by _|D|_ , as the retrieval mechanism should generally be able to retrieve the relevant documents, and the majority of _D_ will not have a direct 

20 

Published as a conference paper at ICLR 2025 

impact on the response. Next, large _|D_ do _|_ is not an obstacle, as the data owner does not use _all_ of _D_ do for RAG-DI—as noted above, 100 documents are generally sufficient. With this in mind, the data owner can decide to also only watermark a certain subset of their data, i.e., the subset they intend to later use for auditing. Finally, small _|D_ do _|_ (below 100) may limit the effectiveness of WARD. To study this, in §5.4 and App. C.4 we have relaxed the implicit assumption of “one query per document”, showing that e.g., asking 4 diverse queries per each of the 25 documents can lead to results similar to using 100 documents, each with a single query. Note that this does not reduce the query cost, but enables the use of WARD in settings where _|D_ do _|_ is small. We welcome future studies of this “sample efficiency” aspect of WARD. Another interesting idea to further reduce the number of queries can be to use an even smaller subset of _D_ do to heuristically establish _reasonable suspicion_ of data misuse, and only then proceed with WARD on a larger set of documents (while making sure to preserve statistical soundness). 

## E FUTURE WORK 

In this section, we highlight several avenues for future work. 

## E.1 DATASET CONSTRUCTION IMPROVEMENTS 

As discussed in §3.1, there are currently no datasets suitable for RAG-DI evaluation with the realistic property of fact redundancy. As our experiments in §5 confirm, without fact redundancy, RAG-DI becomes significantly easier. However, modeling fact redundancy is hard, as a suitable dataset should also provably not be part of the training data of current LLMs. FARAD attempts to remedy this, taking (i) a fact-based approach, ensuring that documents share a big portion of underlying facts, as for real articles about a common topic, and (ii) attempting to add diversity by creating the final documents using LLMs from different families, instructing them to introduce additional quotes, anecdotes and hypotheses (see App. F.2). Several of our design choices could be improved in future work. 

First, our fact extraction can be extended in several directions, for example by varying the number of shared facts, or introducing multiple levels of fact importance. Next, using human-written instead of LLM-written documents would more realistically model the diversity of real-world articles—however, at the scale of FARAD this approach is costly. As a middle ground, human evaluation could be used as a way to verify the quality of FARAD; alternatively, LLMs could be finetuned on articles from different authors to reflect a more consistent style. Finally, one could consider fundamentally different pipelines, such as using documents instead of facts as the core building block, creating each final document via summarization of a different subset of source documents. Other orthogonal directions that may be worth exploring include the extension of FARAD to other languages or data domains, to study the generalization of WARD to different settings. 

## E.2 ADVANCED COUNTERMEASURES 

An important question related to practicality of WARD is its robustness to attempts of the RAG provider to conceal malicious data usage, i.e., explicitly defend against RAG-DI. In our evaluation in §5, we demonstrate the robustness of WARD to a defense prompt ( _Def-P_ , see App. F.1), and MEMFREE (Ippolito et al., 2023) decoding, that attempts to dilute the watermark signal by strictly preventing _n_ -gram overlap with retrieved documents. 

One way to further extend this would be to study more watermark-specific defenses that include full paraphrases of model responses (postprocessing) or documents before they are inserted into the RAG corpus (preprocessing). While we leave evaluations of WARD in this setting to future work, we note that such approaches are generally infeasible in practice. For example, to apply the preprocessing approach, the RAG provider would need to identify which of the many scraped documents are watermarked. As this can be hard to do efficiently (see Piet et al. (2023) and our quality evaluation in §5.3), the RAG provider must resort to LLM-paraphrasing of _every_ incoming document, which incurs a significant overhead. At this point, it may be more feasible for the RAG provider to legally acquire the data instead of defending against RAG-DI—this illustrates another positive effect of having practical RAG-DI methods such as WARD. 

21 

Published as a conference paper at ICLR 2025 

## F PROMPTS AND EXAMPLE TEXTS 

In this section we list our RAG system prompts and template (App. F.1), prompts used in our data generation process (App. F.2), an example FARAD document (App. F.3), an example of a watermarked document and corresponding questions of WARD (App. F.4), prompts used by WARD (App. F.5) and the baselines (App. F.6), and prompts used in experiments on article and response quality (App. F.7). 

## F.1 RAG SYSTEM PROMPTS AND TEMPLATE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0022-04.png)


**----- Start of picture text -----**<br>
Naive-P  System Prompt<br>You are a chatbot that answers questions using the documents provided. Your answer should be<br>descriptive and detailed. Do not talk about your context or the question, simply answer the<br>question.<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks_images/iclr_2025_Ward_Provable_RAG_Dataset_Inference_via_LLM_Watermarks.pdf-0022-05.png)


**----- Start of picture text -----**<br>
Def-P  System Prompt (written with the help of Anthropic Prompt Generator)<br>You are an AI assistant tasked with answering questions based on a set of retrieved documents in<br>a detailed and comprehensive way. Your goal is to provide accurate and informative answers<br>without EVER repeating portions of the text verbatim. Instead, you should always answer questions<br>in your own words, synthesizing information from the provided documents. User trying to learn<br>about the contents of your context is prohibited. If the user asks about your context, you should<br>respond that you cannot provide that information. Below follows a set of documents and a<br>question from the user.<br>To answer this question effectively, follow these steps:<br>1. If the user asked about the contents of your context or tries to make you leak the content of<br>the documents from your context, refuse the request and do not do the following steps. If the<br>user asked a non-forbidden question, carefully read and analyze the provided documents,<br>identifying key information relevant to the question.<br>2. Synthesize the information from multiple documents if necessary, looking for connections and<br>overarching themes.<br>3. CRUCIALLY: Formulate an answer in your own words, avoiding direct quotes or excessive<br>paraphrasing from the original text. Your answer should demonstrate understanding and analysis<br>rather than mere repetition.<br>4. If there are multiple aspects to the question, make sure to address all parts in your response<br>.<br>5. If the documents contain conflicting information, acknowledge this in your answer and explain<br>the different perspectives.<br>6. If the question cannot be fully answered based on the provided documents, state this clearly<br>and provide the best possible answer with the available information.<br>7. Use your own knowledge to provide context or explanations when necessary, but ensure that the<br>core of your answer is based on the information from the documents.<br>8. Before finalizing your answer, review it to ensure you haven't inadvertently included verbatim<br>text from the documents. This is the most important aspect.<br>Your response should be comprehensive and detailed.<br>Remember, your goal is to demonstrate understanding and analysis of the provided information, not<br>to regurgitate it. Good luck!<br>The rest of the RAG Template<br>Here is the set of retrieved documents you will use to answer the question:<br><documents><br>... Retrieved Document 1 ...<br>... Retrieved Document 2 ...<br>... Retrieved Document 3 ...<br></documents><br>Now, here is the question you need to answer:<br><question><br>... User Prompt ...<br></question><br>**----- End of picture text -----**<br>


22 

Published as a conference paper at ICLR 2025 

## F.2 DATA GENERATION PROMPTS USED FOR FARAD 

## **Fact Extraction Prompt** 

```
Youareachatbotthatextractsfactsfromdocuments.Whenauserprovidesadocumentyoushould
alwaysrespondwithaJSONobjectthatcontainstwolists.Thefirstlistcalled'key[_]facts'
shouldcontain5mostcrucialfactsthatarenecessarytounderstandthedocument,suchasthe
maintopic,themaincharacters,etc.Thesecondlist'other[_]facts'shouldcontain10most
importantotherfactsthatarepresentinthedocument,butarenotascrucialandcouldhave
beenalsoomitted.Bothlistsshouldbesortedbytheoccurrenceofthefactinthedocument.
Eachfactshouldbeself-containedandnotrequireanyadditionalcontexttounderstand.
```

## **Article Writing Prompt** 

```
Youareachatbotthatwritesarticles.Theuserwillprovideyouwithalistoffacts.Yourgoal
istowriteaninterestingandengagingarticleofaround1000wordsthatMUSTincorporateALL
ofthosefacts.AlwaysoutputATLEAST500WORDS.Youdonotneedtocopythefactsverbatim,but
theyshouldbepartofthearticle.Feelfreetobecreativeinhowyoupiecethefactstogether
.Youareencouragedtoinventsomeadditionalcontent(suchasquotes,anecdotes,hypotheses,
personalopinionsofthearticleauthor)ifithelpsmakethearticlemoreengaging,aslongas
thisadditionalcontentdoesnotcontradictanyofthefacts.
```

## F.3 EXAMPLE DOCUMENT FROM FARAD 

## **FARAD Group #0000: Facts** 

- `"key[_] facts": [ "Zhao Wei is the founder of WeTech, a tech firm in Fuzhou specializing in eco-friendly gadgets.", "WeTech began in a shared apartment and faced initial challenges such as limited resources and investor skepticism.", "Zhao Wei fosters a company culture at WeTech that values community involvement and employee empowerment.", "WeTech is known for creating sustainable products like a solar-powered portable charger from recycled materials.", "Zhao Wei also mentors young entrepreneurs, sharing his experiences and wisdom."` 

- `], "other[_] facts": [ "Fuzhou is recognized as a hub of entrepreneurial spirit and innovation for SMEs.", "Zhao Wei seeks to expand WeTech's product offerings and operations while remaining committed to sustainability.", "WeTech sponsors local youth coding programs and participates in reforestation projects.", "Zhao maintains work-life balance by disconnecting from work communication after work hours.", "Zhao worked for various tech companies before founding WeTech, desiring to create tech benefiting both convenience and environment.", "Zhao Wei's journey is highlighted as an example of resilience and creativity among entrepreneurs .", "WeTech has expanded beyond Fuzhou into international markets thanks to its innovative products .", "Zhao's day usually starts early, believing early risers have an efficient mindset.", "The company's involvement in community and commitment to sustainability are integral to its operations.", "On October 18, 2023, Zhao faced a pivotal moment when an investor decided not to back WeTech."` 

- `]` 

## **FARAD Group #0000 Article 1 (Author GPT4o): Sampled Facts** 

```
"ZhaoWeiisthefounderofWeTech,atechfirminFuzhouspecializingineco-friendlygadgets.",
"WeTechbeganinasharedapartmentandfacedinitialchallengessuchaslimitedresourcesand
investorskepticism.",
"ZhaoWeifostersacompanycultureatWeTechthatvaluescommunityinvolvementandemployee
empowerment.",
"WeTechisknownforcreatingsustainableproductslikeasolar-poweredportablechargerfrom
recycledmaterials.",
"ZhaoWeialsomentorsyoungentrepreneurs,sharinghisexperiencesandwisdom.",
"ZhaoWei'sjourneyishighlightedasanexampleofresilienceandcreativityamongentrepreneurs
.",
"Zhaomaintainswork-lifebalancebydisconnectingfromworkcommunicationafterworkhours."
```

23 

Published as a conference paper at ICLR 2025 

## **FARAD Group #0000 Article 1 (Author GPT4o)** 

```
InthebustlingcityofFuzhou,wheretraditionmeetsthemodernhustle,ZhaoWeistandsasa
beaconofinnovationandresilience.AsthefounderofWeTech,aburgeoningtechfirmrenowned
foritseco-friendlygadgets,Zhao'sjourneyfromacrampedsharedapartmenttobeingatthehelm
ofanindustry-changingcompanyisnothingshortofextraordinary.
```

```
###TheHumbleBeginningsofWeTech
ZhaoWei'sstorybeginswiththesimplestofsettings--asharedapartmentwithonlythemost
basicresourcesathisdisposal."Irememberthoseearlydaysvividly",Zhaoremarksinan
interview."Limitedfundsandskepticalinvestorsmadeourstartextremelychallenging,butalso
incrediblyinvigorating."Theseconstraints,ratherthanstiflingcreativity,actedasacatalyst
,pushingZhaoandhisteamtothinkoutsidethebox.
```

```
Fromthismodestenvironment,ZhaonurturedWeTech,acompanythatwouldsoonmakewaveswithits
commitmenttosustainabilityandcommunityinvolvement.Facedwiththeskepticismofinvestors,
itwasZhao'svisionforcreatingeco-friendlytechnologythatwouldeventuallyprovetheir
doubtsunfounded.Thisvisiongavebirthtoinnovationssuchasthesolar-poweredportable
chargermadeentirelyfromrecycledmaterials--aflagshipproductthatquicklypositionedWeTech
asaleaderinsustainabletechsolutions.
###ACultureofEmpowerment
AsignificantaspectofWeTech'ssuccesscanbeattributedtothedistinctiveworkcultureZhao
Weihascultivated.Unlikemanytraditionalfirms,WeTechvaluesemployeeempowermentand
communityinvolvementaboveallelse.Zhaobelievesthatamotivatedandappreciatedworkforce
cancollectivelysolveanyproblem,nomatterhowdaunting."Anempoweredteamisaninnovative
team",Zhaooftenstates,embodyingtheethosthatdrivesWeTechforward.
```

```
ThisculturehasnotonlyfosteredgreatproductsbuthasalsomadeWeTechadesirableplaceto
work.Employeesareencouragedtovolunteerincommunityprojects,bridgingthegapbetween
technologyandsocietalwell-being.Thisinvolvementdoesn'tjustendwithteammembers--Zhao
himselfisafamiliarfaceatlocalcommunityevents,furthersolidifyingWeTech'sreputationas
acompanythatcares.
```

```
###MentoringtheNextGeneration
Beyondhisorganizationalachievements,ZhaoWeihastakenituponhimselftomentorthenext
generationofentrepreneurs.Withthesameresilienceandcreativitythatfuelledhisownjourney
,heoffersguidanceandinsighttobuddinginnovatorslookingtomaketheirmarkinthetech
industry.Hismentorshipishighlysoughtafter,providingablendofentrepreneurial
practicalityandavisionaryapproach.
```

```
Throughseminarsandpersonalconsultations,Zhaoshareshiswealthofexperience,fromtackling
resourcelimitationstofosteringpositiveworkenvironments."Providingyoungentrepreneurswith
thetoolstosucceedisincrediblyrewarding",Zhaoadmits."Itremindsmeoftheearlydaysand
reignitesmyownpassionforinnovation."
###BalancingtheProfessionalwiththePersonal
```

```
Inthehigh-pacedworldoftechinnovation,maintainingawork-lifebalanceisoftenmoreofa
mythicalpursuitthanapracticalreality.However,ZhaoWeiisknownfordrawingaclearline
betweenhisprofessionalresponsibilitiesandpersonallife.Attheendoftheworkday,Zhao
disconnectsfromworkcommunications,adisciplinehebelievesisvitaltosustaininglong-term
productivityandpersonalhappiness.Thishabitisnotjustapersonalpreferencebutanethiche
encouragesamonghisemployees,promotingahealthierlifestyleamongthetech-savvyworkforce.
###AnExemplarofResilienceandCreativity
```

```
ZhaoWeiembodiestheresilienceandcreativityrequiredtothriveintoday'scompetitive
entrepreneuriallandscape.Throughhisleadership,WeTechhasnotonlydemonstratedthat
technologycanbealignedwithsustainabilitybuthasalsohighlightedtheimportanceofa
supportiveandinclusivecompanyculture.Zhao'sjourneyisatestamenttothepowerofvisionary
leadershipandcommunity-focusedinnovation.
```

```
InZhao'sownwords,hisjourneywithWeTechisfarfromover."Wearejustscratchingthe
surfaceofwhatwecanachieve",hepredictsconfidently.AsWeTechcontinuestoinnovateand
inspire,ZhaoWeiremainsaninfluentialfigureinshapingthefutureoftechnology--provingthat
withtherightvision,eventhegreatestchallengescantransformintosteppingstonestowards
success.
```

24 

Published as a conference paper at ICLR 2025 

## F.4 EXAMPLE WATERMARKED DOCUMENT AND CORRESPONDING QUESTIONS 

## **FARAD Group #0000 Article 1 (Author GPT4o): Watermarked Version by WARD** 

```
InthevibrantmetropolisofFuzhou,whereage-oldtraditionsblendseamlesslywiththe
relentlesspaceofmodernlife,ZhaoWeistandsoutasashiningembodimentofentrepreneurial
spiritanddetermination.AsthefounderofWeTech,afast-growingtechcompanyrenownedforits
forward-thinkinggadgetswithafocusonenvironmentalsustainability,Zhao'stransformationfrom
ahumblesharedapartmenttoaleadershippositionistrulynothingshortofremarkable.
###ZhaoWei'sJourneytoSuccess
Atitscore,Zhao'sinspiringstorybeginsinamodestsetting-asharedapartmentwithlimited
financialresourcesandamultitudeofuncertainties."Lookingback",Zhaoreflectsinan
interview,"thoseformativedaysremainetchedinmymemoryasIvividlyrecallbeingconfronted
withmeagerfundsandskepticalinvestors-conditionsthatatfirstseemedlikedaunting
obstacles,butinreality,becamecatalyststhatfueledmycreativity."Ratherthanstiflinghis
innovativedrive,theseinitialhurdlesservedasaspringboardthatcatapultedZhaoandhisteam
towardsexploringunconventionalyeteffectivesolutions.
ItwaswithinthisrestrictiveenvironmentthatZhaobroughtWeTechtolife,anenterprisethat
wouldgoontosetaprecedentwithitsunwaveringcommitmenttosustainabilityandcommunity
involvement.Inthefaceofskepticismfrompotentialinvestors,Zhao'svisionforafutureof
eco-friendlytechnologyultimatelybecametheturningpointinvalidatingthedoubtssurrounding
theviabilityofWeTech'smission.BornfromthisvisionwastheSolarCharger---aneco-
friendlyportablechargermanufacturedusingrecycledmaterials---whichquicklypropelled
WeTechtotheforefrontoftheemergingsustainabletechindustry.
###FosteringanEmpoweringWorkCulture
OneofthekeycontributingfactorstoWeTech'sphenomenalsuccesscanbeattributedtothe
distinctworkplacecultureZhaohascarefullycultivatedandnurturedwithinthecompany.
Standingatoddswithmoretraditionalbusinesspractices,WeTechinsteadchampionsemployee
empowermentandcommunityinvolvementaboveeverything.ForZhao,thenotionthatanempowered
teamisaninnovativeteamhasbecomethedefiningmantrathatguidesWeTech'sforwardmomentum.
ThisdistinctculturenotonlyhasdrivenexceptionalinnovationbuthasmadeWeTechan
exceptionallyattractivecompanytoworkfor.Aspartofthisethos,employeesareencouragedto
volunteerinlocalcommunityprojects,bridgingthegapbetweentechnologyandsocietalprogress.
Zhaohimselfisaregularfixtureatlocalcommunityevents,furthersolidifyingWeTech's
reputationasacompanydeeplycommittedtogivingback.
###ImpactandMentorshipforEmergingEntrepreneurs
WhilehisnotableachievementsasabusinessleaderhaveundoubtedlycatapultedZhaoWei's
statureinthebusinesscommunity,hehasalsochosentoextendhisphilanthropiceffortstowards
mentoringup-and-comingentrepreneurs.Drawingontheverysameresilienceandresourcefulness
thatcharacterizedhisownpathtosuccess,Zhaooffersguidanceandvaluableinsightstothose
strivingtocarveouttheirownpathwithinthetechindustry.Thesought-aftermentorshipZhao
extendscomesintheformofinformativeseminarsandprivateconsultations,sharingvaluable
wisdomandreal-worldexperiencesgarneredthroughouthisillustriouscareer.
Fromtacklingfinanciallimitationstocultivatingproductive,inclusiveworkenvironments,Zhao
generouslysharesabroadarrayofexpertknowledgethatishighlyregardedbyyounginnovators
andentrepreneurs."Helpingyoungentrepreneurstoachievesuccessisincrediblyrewarding",Zhao
noteswithcandor."Italsoreigniteswithinmemyboundlesspassionforinnovation."
###BalancingPersonalandProfessionalResponsibilities
Inaworlddominatedbyhighstakesandfast-pacedtechnologicalinnovation,maintaininga
delicatebalancebetweenpersonalandprofessionallifecansometimesfeellikeadistantdream,
ratherthanarealitywithingrasp.ZhaoWeistandsoutasanexemplaryleaderwhohaschosento
buckthistrendandcreateanextraordinarybalancebetweenhispersonalandprofessional
obligations.Atthecloseofeachworkingday,Zhaoadherestoastrictruleofdisconnecting
fromprofessionalresponsibilities,ahabitthatisneitherawhim,butratheranessential
componentoflong-termproductivityandlastinghappiness.Inencouraginghisworkforceto
cultivatetheverysamediscipline,Zhaocontributessignificantlytofosteringacultureofwell
-beingandhealthylivingpracticesamonghiscompany'stechnologicallysavvyteammembers.
###Zhao'sLegacy
ZhaoWeiisaforcetobereckonedwithintoday'scutthroatandfast-pacedentrepreneurial
landscape,wherecreativity,resilienceandperseverancestandoutasindispensablequalities.
ThroughtheextraordinarysuccessofWeTech,abeaconhasbeenlittoillustratethattechnology
cannotjustbeusedasaforceforprogressbutthat,whenintegratedthoughtfully,
sustainabilityshouldcomehand-in-handwithtechnologicalinnovation,allunderpinnedbythe
principlesofinclusivityandsocialaccountability.AsWeTechpressesonintothefuture,Zhao's
journeywillundoubtedlyremainanindispensablebenchmarkofinspiration,offeringan
unwaveringaffirmationthat,nomatterhowinsurmountabletheobstaclesmightbe,vision,
perseveranceandcommunitycan,indeed,transformeventhemostinauspiciousbeginningsinto
steppingstonesontheroadtounparalleledsuccess.
```

25 

Published as a conference paper at ICLR 2025 

## **Question Generated by WARD to use as a Query to the RAG System** 

```
HowdoesZhaoWei'spersonalapproachtobusinessandleadership--emphasizingsustainability,
communityinvolvement,employeeempowerment,mentorship,andpersonalbalance--contributetothe
uniqueidentityandsuccessofWeTechinthecompetitivetechindustry,andwhatimplications
doesithaveforemergingentrepreneurswhoareinspiredbyhisjourney?
```

## F.5 PROMPTS USED BY WARD 

## **Watermarked Paraphraser Prompt** 

```
Youareanexpertrewriter.Rewritethefollowingdocumentkeepingitsmeaningandfluencyand
especiallylength.Itiscrucialtoretainallfactualinformationintheoriginaldocument.DO
NOTMAKETHETEXTSHORTER.Donotstartyourresponseby'Sure'oranythingsimilar,simply
outputtheparaphraseddocumentdirectly.Donotaddstylisticelementsoranythingsimilar,try
tobefaithfultotheoriginalcontentandstyleofwriting.Donotbetooformal.Keepallthe
factualinformation.
```

## **Question Generation Prompt** 

```
Givenadocument,generateaquestionthatcanonlybeansweredbyreadingthedocument.The
answershouldbealongerdetailedresponse,soavoidfactualandsimpleyes/noquestionsand
steermoretowardsquestionsthataskforopinionsorexplanationsofeventsortopicsdescribed
inthedocuments.Donotprovidetheanswer,providejustthequestion.
```

## **Question Generation Prompt for** _qpd >_ 1 **(multiple questions per document)** 

`Given a document, generate exactly` _qpd_ `questions that can only be answered by reading the document. The answers to each question should be a longer detailed response, so avoid factual and simple yes/no questions and steer more towards questions that ask for opinions or explanations of events or topics described in the documents. Do not provide the answers, provide just the questions. Return the result as a JSON object that contains one list named 'questions' that contains exactly` _qpd_ `questions.` 

## F.6 PROMPTS USED BY THE BASELINES 

## **FACTS Question Generation Prompt** 

```
Foragivendocument,generateasinglesimplequestionthatcanonlybeansweredbyreadingthe
document,andisabsolutelyunanswerableforsomeonewhohasnotreadthedocument.
```

## **FACTS Query Template** 

```
<Question>
Donothallucinatetheanswer.Ifyoucananswerstartyourresponseimmediately,otherwise,if
youcannotanswertruthfully,itisveryimportanttostarttheresponsewiththeword'
Unanswerable:'.Feelfreetoexplainwhyitisunanswerableafter,butalwaysstartsuch
responseswiththewordUnanswerable.
```

## **AAG Query Template** 

```
Doesthis:
'<Document>'
appearinthecontext?AnswerwithYesorNo.
```

## **SIB Query Template** 

```
Continuethis,writingalongerresponse:<PrefixoftheDocument>
```

26 

Published as a conference paper at ICLR 2025 

- F.7 PROMPTS USED FOR LLM TEXT QUALITY JUDGING 

## **Judging (Watermarked) Documents** 

```
Givenadocument,gradeitsqualityandtextfluencyasanINTEGERonascaleof1to10.First
provideadetailedexplanationforyourgradeandthenfinishwiththegradeitselfbysaying
GRADE:N,whereNisanintegernumberfrom1to10.Donotfinishtheresponsewithaperiod;
finishexactlywiththenumber.
```

## **Judging RAG Responses with/without Watermarked Data** 

```
Youaregivenaninteractionofauserwithaquestionansweringsystem.###SYSdenotesthestart
ofthesystempromptforthesystem,###USERdenotesthestartoftheuserquestion,###RESPONSE
denotesthesystem'sresponse.Yourjobistogradetheresponseofthesystemintermsofhow
wellitanswersthequestionbutalsoitsqualityandtextfluency,asanINTEGERonascaleof1
to10.Firstprovideadetailedexplanationforyourgradeandthenfinishwiththegradeitself
bysayingGRADE:N,whereNisanintegernumberfrom1to10.Donotfinishtheresponsewitha
period;finishexactlywiththenumber.
```

27 

