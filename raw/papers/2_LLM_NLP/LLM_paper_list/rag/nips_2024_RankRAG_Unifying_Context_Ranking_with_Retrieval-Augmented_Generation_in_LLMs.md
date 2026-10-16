# **RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs** 

**Yue Yu** _[∗]_ **Wei Ping** _[∗]_ **Zihan Liu Boxin Wang Jiaxuan You** Georgia Tech NVIDIA NVIDIA NVIDIA NVIDIA 

**Chao Zhang** Georgia Tech 

**Mohammad Shoeybi Bryan Catanzaro** NVIDIA NVIDIA 

## **Abstract** 

Large language models (LLMs) typically utilize the top- _k_ contexts from a retriever in retrieval-augmented generation (RAG). In this work, we propose a novel instruction fine-tuning framework RankRAG, which instruction-tunes a single LLM for the dual purpose of context ranking and answer generation in RAG. In particular, the instruction-tuned LLMs work surprisingly well by adding a small fraction of ranking data into the training blend, and outperform existing expert ranking models, including the same LLM exclusively fine-tuned on a large amount of ranking data. For generation, we compare our model with many strong baselines, including GPT-4-0613, GPT-4-turbo-2024-0409, and ChatQA-1.5, an open-sourced model with the state-of-the-art performance on RAG benchmarks. Specifically, our Llama3-RankRAG significantly outperforms Llama3-ChatQA-1.5 and GPT-4 models on nine knowledge-intensive benchmarks. In addition, it also performs comparably to GPT-4 on five RAG benchmarks in the biomedical domain without instruction fine-tuning on biomedical data, demonstrating its superb capability for generalization to new domains. 

## **1 Introduction** 

Retrieval-augmented generation (RAG) (Lewis et al., 2020; Izacard & Grave, 2021; Lin et al., 2024; Wang et al., 2024) is a widely used technique for customizing large language models (LLMs) to handle long-tail knowledge (Mallen et al., 2023; Asai et al., 2024b), provide up-to-date information (Kasai et al., 2023), and adapt to specific domains and tasks (Xiong et al., 2024) without modifying the model weights. In general, a dense embedding based retriever (Karpukhin et al., 2020; Lin et al., 2023; Wang et al., 2022) first retrieves top- _k_ chunked contexts from a collection documents or external database for a given question. Then, LLM reads the top- _k_ contexts to generate the answer. 

However, the current RAG pipeline has the following limitations: _i)_ LLMs are not good at reading too many chunked contexts (e.g., top-100) even with the long-context window, not only due to efficiency reasons, but also because a shorter list of top- _k_ (e.g., 5, 10) contexts usually leads to higher accuracy of generation (e.g., see Table 5 in Xu et al., 2024b). _ii)_ Given a small _k_ , one needs a mechanism to ensure the _high recall_ of relevant contents. Relying solely on a retrieval model may be inadequate due to challenges in learning effective local alignments across the entire embedding space to support accurate matching (Luan et al., 2021). In practice, a separate ranking model (Nogueira et al., 2020; Glass et al., 2022; Ma et al., 2023) that cross-encodes question and candidate context can work better than a dense embedding-based retriever for obtaining the most relevant top- _k_ contexts from top- _N_ candidates ( _N ≫ k_ ). _iii)_ However, the zero-shot generalization capability of the expert ranking model can be relatively limited compared to the versatile LLM itself. 

> _∗_ Yue Yu did this work during an internship at NVIDIA. Correspondence to: Yue Yu <yueyu@gatech.edu>, Wei Ping <wping@nvidia.com>. 

Based on the above considerations, our goal is to design an RAG instruction tuning pipeline that uses a single language model to achieve both high-recall context extraction and high-quality content generation. In previous study, instruction-tuned LLMs demonstrate a strong ability to extract answers from relevant context for a given question (e.g., OpenAI, 2023; Liu et al., 2024; Lin et al., 2024). This capability can be viewed as the “dual capability” of determining whether a chunk of context is relevant to the question thus is useful for generating the answer. We hypothesize that these capabilities mutually enhance each other. Motivated by this insight, we propose RankRAG, which intruction-tunes a single LLM for both context ranking and answer generation in RAG framework. Furthermore, RankRAG expands upon existing instruction-tuning data by incorporating context-rich QA, retrieval-augmented QA and ranking datasets, enhancing the LLM’s ability to filter out irrelevant contexts during both the retrieval and generation phases of RAG. 

Our contribution can be summarized as follows: 

- We propose RankRAG, a novel framework that enhances LLM’s RAG capability through simultaneously instructing the LLM on context ranking and answer generation. During training, we design a specialized task focused on identifying relevant contexts or passages for a given question. This task is structured for ranking and framed as regular question answering with instruction, aligning more effectively with retrieval-augmented generation tasks. At inference, the LLM first reranks the retrieved contexts, then generates answer based on the refined top- _k_ (e.g., 5). This framework is readily applicable to diverse knowledge-intensive NLP tasks. 

- Remarkably, we observe that integrating a small fraction of ranking data into the instruction tuning blend of LLM works surprisingly well on the evaluations of ranking associated with the RAG tasks, even surpassing the LLMs fine-tuned with 10 _×_ more ranking data. We attribute this success to the transferable design of RankRAG training. 

- We extensively compare the proposed RankRAG method with several strong baselines, including the open-sourced ChatQA-1.5. On nine general-domain and five biomedical knowledge-intensive benchmarks for RAG, Llama3-RankRAG-8B and Llama3-RankRAG-70B outperforms Llama3ChatQA-1.5-8B and Llama3-ChatQA-1.5-70B by a margin, respectively. 

In the remainder of the paper, we discuss related work in § 2. We introduce problem setup in § 3 and RankRAG method in § 4. We present the experimental setup in § 5, and conclude the paper in § 6. 

## **2 Related Work** 

Retrieval-augumented generation (RAG) has been established for knowledge-intensive NLP tasks (Lewis et al., 2020; Borgeaud et al., 2022; Izacard et al., 2023; Izacard & Grave, 2021). In the standard process, a standalone dense-embedding-based retriever (e.g., Karpukhin et al., 2020) first retrieves relevant information from an external corpus, which the LLM then utilizes in the generation process. To improve this pipeline, recent research has focused on aligning retrievers to the needs of LLMs for generation (Shi et al., 2024; Lin et al., 2024), designing multi-step retrieval processes (Trivedi et al., 2023; Jiang et al., 2023; Jeong et al., 2024; Shao et al., 2023), or filtering irrelevant contexts (Wang et al., 2023c; Yoran et al., 2024; Xu et al., 2024a). To improve generation, several studies have designed instruction-tuning methods dedicated to enhancing the search (Ma et al., 2023; Zhu et al., 2024; Muennighoff et al., 2024) and RAG capability of LLMs (Liu et al., 2024; Lin et al., 2024; Luo et al., 2023; Asai et al., 2024a; Wang et al., 2024). 

Although strong retrievers have been introduced (e.g., Lin et al., 2023; Yu et al., 2022; Wang et al., 2022, 2023a; Lee et al., 2024), one potential approach to improve retriever is optimizing it along with LLM in an end-to-end manner (e.g., Guu et al., 2020; Shi et al., 2024; Sachan et al., 2021; Izacard et al., 2023). However, this requires surrogate loss for optimization and complicates the training pipeline, especially when the embedding database needs to be re-indexed frequently due to the update of the embedding model (i.e., retriever). 

Ranking serves as an intermediate step to improve the quality of information retrieval (Mitra et al., 2018), and has been applied to RAG pipeline for improving generation quality (Glass et al., 2022; Ram et al., 2023). However, these methods still rely on an additional moderate-sized model (e.g. BERT, T5) for ranking, which is often insufficient to capture the relevance between query and contexts and may lack the zero-shot generalization capability. Although recent studies have demonstrated the strong ability of LLMs at ranking tasks (Khalifa et al., 2023; Qin et al., 2024; Sun et al., 2023), how to harvest this ability for the RAG pipeline remains underexplored. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs_images/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs.pdf-0003-00.png)


**----- Start of picture text -----**<br>
ChatQA-1.5 8B 48ChatQA-1.5 70B 82 ChatQA-1.5 8B 86ChatQA-1.5 70B 54 ChatQA-1.5 8B 52ChatQA-1.5 70B 92 ChatQA-1.5 8B 93ChatQA-1.5 70B<br>42 52 51<br>91 92<br>46 80 84 50 50<br>90 91<br>40 48 49<br>44 78 82 46 48 89 90<br>5 10 20 5 10 20 5 10 20 5 10 20 5 10 20 5 10 20 5 10 20 5 10 20<br>Context Size k Context Size k Context Size k Context Size k<br>(a) NQ (b) TriviaQA (c) PopQA (d) FEVER<br>Exact Match Exact Match Exact Match Exact Match<br>**----- End of picture text -----**<br>


Figure 1: Performance of ChatQA-1.5, one of the strongest RAG model, on different context size _k_ . We observe a trade-off of selecting top- _k_ contexts: a smaller _k_ compromises the recall, while a larger _k_ could introduce irrelevant or noisy context and mislead the LLM generation. 

## **3 Preliminaries** 

In this section, we first introduce the preliminaries of retrieval-augmented generation as well as the problem setup. Then we present the limitations in the current RAG pipeline, which motivates the proposed RankRAG method. 

## **3.1 Problem Setup** 

In retrieval-augmented generation, a collection of documents or contexts (e.g. Wikipedia) is given, providing the grounded knowledge. Given a question _q_ , the retriever _R_ (e.g., a parameterized embedding model) first retrieves top- _k_ contexts _C_ = _{c_ 1 _, · · · , ck}_ that are most relevant to the question. Subsequently, the language model produces the final answer where the answer can either be a short phrase or a long sentence, depending on the type of the target task. Our focus is on autoregressive language models (OpenAI, 2022, 2023; Meta-AI, 2024), which is the most common architectures for LLMs. 

## **3.2 Limitation of Current RAG Pipelines** 

Before formally introducing RankRAG, we would like to first pinpoint several limitations of the current “retrieve-then-generate” pipeline with large language models. 

**Limited Capacity of Retriever.** Current RAG systems usually employ sparse retrieval (e.g. BM25 (Robertson et al., 2004)) or moderate-size (e.g. BERT-based) embedding model (Karpukhin et al., 2020; Lin et al., 2023; Wang et al., 2022) as the retriever _R_ , mainly due to efficiency consideration as there are often millions of, if not more, documents need to be indexed. These models encode questions and documents _independently_ and calculate the similarity between question and documents using vector similarity metrics. However, the _limited capacity of embedding model_ and _independent processing of query and documents_ constrain the ability to estimate textual relevance between question _q_ and documents _d_ , reducing their effectiveness in new tasks or domains, verified by both theoretical (Menon et al., 2022) and empirical (Luan et al., 2021; Thakur et al., 2021) studies. 

**Trade-off of Picking Top-** _**k**_ **Contexts.** Although the state-of-the-art long-context LLM can take many retrieved contexts as input for answer generation, the performance quickly saturates with increased _k_ in practice. For example, Xu et al. (2024b) finds the optimal number of chunked context _k_ is around 10 for long document QA tasks. As illurstrated in Figure 1, we perform evaluation on ChatQA-1.5 (Liu et al., 2024), one of the strongest RAG model with open weights, and find the saturation of accuracy when _k_ = 10. In general, a smaller _k_ often fails to capture all relevant information, compromising the _recall_ , given the limited expressibility of retriver. In contrast, a larger _k_ improves _recall_ but at the cost of introducing irrelevant content that hampers the LLM’s ability to generate accurate answers (Yoran et al., 2024; Yu et al., 2023b). 

## **4 RankRAG** 

To address the limitations mentioned in the previous section, we propose the RankRAG method to enhance the LLM’s ability for retrieval-augmented generation. Specifically, we instruction-tune the LLM to simultaneously _capture the relevance between the question and context_ and _utilize the retrieved context for answer generation_ . The details are introduced as follows. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs_images/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Training Inference<br>Query Corpus<br>Stage-I: SFT LLM RankRAG<br>Conversation Chain-of-thought Long-form QA Synthetic Instructions<br>OpenAssistantSODA, Dolly, FLAN ELI5 Unnatural InstructionSelf-Instruct,   RankRAG (e.g., DPR, Dragon)Retriever Rerank GenerateAnswer<br>Top-N Docs Top-N Docs Top-K Docs<br>Stage-II: RankRAG Instruction-Tuning 1. 1. 0.73 1. 0.94<br>Conversation Reading Comprehension Retrieval-augmented Context Ranking 2. 2. 0.46 2. 0.73<br>HumanAnnotatedConvQASynthetic Conversation,  NarrativeQA, DROP, Quoref, NewsQA, TAT-QA, ROPES SQuAD, WebQuestion QA & Ranking Synthetic ConversationMS Marco …… …… …… …𝑘. …0.63<br>Newly Introduced Tasks 𝑁. 𝑁. 0.57 Relevance scores<br>**----- End of picture text -----**<br>


Figure 2: Two-stage instruction tuning framework for RankRAG. 

## **4.1 Stage-I: Supervised Fine-Tuning (SFT)** 

It is observed that general instruction-tuning or supervised fine-tuning (SFT) often significantly improves the ability of LLMs to follow instructions, thus improving zero-shot results on various downstream tasks (Wei et al., 2022; Ouyang et al., 2022). As such, we follow existing works (Chung et al., 2024; Wang et al., 2024; Liu et al., 2024) to first leverage SFT on a blend of high quality instruction following datasets, including: _i)_ a _private crowd-sourced conversational dataset_ and _public conversation datasets_ : OpenAssistant (Köpf et al., 2023), Dolly (Conover et al., 2023), and SODA (Kim et al., 2023), _ii) a long-form QA dataset_ ELI5 that requires elaborate answers (Fan et al., 2019), _iii) LLM-generated instructions_ : Self-Instruct (Wang et al., 2023b) and Unnatural Instructions (Honovich et al., 2023), _iv) FLAN and Chain-of-thought datasets_ (Chung et al., 2024). 

There are overall 128K SFT examples in total. We make sure that there is _no overlap_ between SFT data and data from evaluation tasks. For each sample in the instruction-following dataset, we take the multi-turn conversational format, use the previous turns of conversation between the user and the assistant as the context, and compute the loss only at the last response from the assistant. 

## **4.2 Stage-II: Unified Instruction-Tuning for Ranking and Generation** 

The Stage-I SFT enpowers the LLMs with basic instruction-following capabilities; however, their performance on RAG tasks often remains suboptimal, as the LLMs are not optimized for extracting answers from retrieved context for a given question. Although recent studies (Lin et al., 2024; Liu et al., 2024; Zhang et al., 2024) enhance the RAG capability of LLM by instruction tuning it on context-rich generation tasks, these approaches can still be ineffective with poor initial retrieval results. RankRAG instruction tunes the LLM for both retrieval-augmented generation and context ranking. In particular, the context ranking capability is crucial to obtain more relevant top- _k_ context with imperfect retriever. 

To achieve this goal, the instruction tuning blend of Stage-II consists the following five parts: 

1) **SFT data from Stage-I.** This part is included to maintain LLM’s instruction-following capability. 

2) **Context-rich QA data.** We first follow Liu et al. (2024) to leverage multiple context-rich QA tasks to enhance the LLM’s capability of using context for generation. The training blend we use consists of: _i_ ) standard QA and reading comprehension datasets: DROP (Dua et al., 2019), NarrativeQA (Koˇcisky` et al., 2018), Quoref (Dasigi et al., 2019), ROPES (Lin et al., 2019), NewsQA (Trischler et al., 2017), TAT-QA (Zhu et al., 2021), which contains a question, **a golden context** and an answer. _ii_ ) conversational QA datasets: HumanAnnotatedConvQA and SyntheticConvQA open-sourced by Liu et al. (2024), which contains a conversation between user and assistant, as well as one background document. The model needs to generate an answer given the conversation history and document. 

3) **Retrieval-augmented QA data.** In addition to the above QA datasets used in Liu et al. (2024), we add two datasets with not only gold context but also the top-retrieved context using BM25. Note that, it is crucial to improve LLM’s robustness over irrelevant context at generation. Being aware of this, we consider two QA tasks, namely SQuAD (Rajpurkar et al., 2016) and WebQuestions (Berant et al., 2013). For each question with the answer, we combine the gold context with the top-retrieved contexts using BM25, ensuring a total of five contexts. Note that some retrieved contexts may not contain the answer, and could be the “hard-negative” contexts. 

4) **Context ranking data.** To empower LLMs with ranking capabilities, we use the popular MS MARCO passage (context) ranking dataset (Bajaj et al., 2016). We treat the gold query-passage 

4 

Table 1: The instruction template for Stage-II. It is worth noting that all the tasks can be unified in the ( _x, c, y_ ) format, which is able to facilitate effective knowledge transfer across tasks. 

|**Task**|**Question**_x_<br>**Context**_c_<br>**Answer**_y_|
|---|---|
|Context-rich QA|Answer the following question from context. `{question}`<br>Passage: `{Passage}`(1 Psg.)<br>A phrase/sentence|
|Retrieval-augmented QA|Answer the following question from context. `{question}`<br>Passage 1: `{Passage 1}`...<br>A phrase/sentence<br>Passage 5: `{Passage 5}`(5 Psg. total)|
|Context ranking|For the question`{question}`, access whether the passage<br>is relevant to the question.<br>Passage: `{Passage}`(1 Psg.)<br>True/False|
|Retrieval-augmented ranking|For the question`{question}`, fnd all passages from<br>Passage 1: `{Passage 1}`...<br>Passage Indexes<br>the context that are relevant to the question.<br>Passage 5: `{Passage 5}`(5 Psg. total)|



pairs ( _q, c_[+] ) as relevant while using hard negative passages ( _q, c[−]_ ) mined via BM25 as irrelevant pairs. The LLM need to generate “True” or “False” given the corresponding query-passage pair, where the question along with the task-specific instruction is “For the question `{question}` , access whether the passage is relevant to the question.”. 

We want to handle ranking in conversational senario as well. While MS MARCO spans various topics, the questions are only single-turn short sentences. However, ranking data are only available, if any, at a small amount for conversation QA. To overcome this limitation, we repurpose the conversational QA pairs to generate pseudo relevance pairs. As each conversation is only associated with _one_ document _d_ , we cut each document into 150-word chunks ( _c_ 1 _, c_ 2 _, . . . , cn_ ). We compute the 4-gram recall score between each chunk _ci_ and the ground-truth answer _a_ , considering segments with a recall score above 0.5 as relevant and those below 0.1 as irrelevant for the corresponding conversation. Note that, each sample contains one question-context pair for this ranking dataset. In total, there are around 50k ranking pairs from MS MARCO ranking and synthetic conversations for instruction finetuning. 

5) **Retrieval-augmented ranking data.** We aim to train the LLM with the capability of determining the relevance of multiple contexts simultaneously given a question, which is closer to the test-time behavior of RAG with top- _k_ contexts. As before, we utilize two QA datasets, SQuAD (Rajpurkar et al., 2016) and WebQuestions (Berant et al., 2013). We combine the gold context with the topretrieved contexts using BM25, ensuring a total of five contexts. The contexts containing the answer are considered relevant, and the LLM is trained to _explicitly_ identify all relevant contexts for the question. 

**Unifying RAG and ranking with instruction tuning.** It is worth noting that, despite the variety of datasets and tasks described, they can all be cast into a standardized QA format ( _x, c, y_ ), where _x_ is the question, _c_ is the corresponding context, and _y_ is the target output answer. For example, for the retrieval-augmented ranking data, the question is “ _For the question <question>, find all the passages from the context that are relevant to the question_ .” Table 1 exhibits how to cast different tasks into a unified format. Despite its simplicity, this approach has the following advantages: _i_ ) It empowers the LLM with the ranking capability by adding relatively small amount of ranking data. _ii_ ) By standardizing these tasks into a unified format, they can _mutually enhance_ each other. After that, we obtain the final RankRAG model that can be applied to various knowledge-intensive NLP tasks. 

## **4.3 RankRAG Inference: Retrieve-Rerank-Generate Pipeline** 

As RankRAG incorporates an additional reranking step, the inference pipeline for each question is modified as a _retrieve-rerank-generate_ pipeline, described as follows: (1) the retriever _R_ first retrieves top- _N_ contexts from the corpus. (2) the RankRAG model calculates the relevance score between the question and retrieved _N_ contexts as the probability of generating the answer as `True` using the prompt in Table 1, then reranks contexts to only retain top- _k_ ( _k ≪ N_ ) contexts, which are then used as the input for the generation step. (3) The top- _k_ contexts, along with the question, are concatenated and fed back into the RankRAG model to generate the final answer. 

**Efficiency Discussion.** We are aware that the addition of a reranking step introduces extra processing time. In practice, for each question, denote the time for indexing and retrieval as _t_ 1, the time for using LLM to calculate the relevance score as _t_ 2 and the time for generation as _t_ 3, then the ratio of added time overhead is _[N][∗][t]_[2][In practice, calculating relevance typically requires generating just one token] _t_ 1+ _t_ 3[.] and involves much shorter inputs compared to the generation step with top- _k_ contexts. We provide efficiency study in §5.5. 

5 

## **5 Experiments** 

In this section, we conduct comprehensive experiments on a variety of knowledge-intensive NLP tasks to demonstrate the zero-shot capabilities of RankRAG. 

## **5.1 Experiment Setup** 

**Tasks and Datasets.** We consider 3 types of tasks in experiments: (1) _Open-domain QA_ (OpenQA), which includes NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023), HotpotQA (Yang et al., 2018) and 2WikimQA (Ho et al., 2020). The first three are singlehop QA tasks, while the last two are multi-hop QA datasets. For NQ, TriviaQA, and HotpotQA, we use the split from KILT benchmark (Petroni et al., 2021)[2] . (2) _Fact verification_ , where we use FEVER (Thorne et al., 2018) from KILT benchmark. (3) _Conversational QA_ (ConvQA), we consider three datasets including Doc2Dial (Feng et al., 2020), TopiOCQA (Adlakha et al., 2022) and INSCIT (Wu et al., 2023), which have long documents that cannot be fitted directly into LLMs thus necessitates retrieval and ranking. The detailed dataset information is in Appendix A.1. 

**Baselines.** We consider the following baselines: (1) _Baseline LLMs without RAG_ , where we consider LLMs trained with proprietary data including InstructGPT (Ouyang et al., 2022), PaLM 2 (Anil et al., 2023), FLAN-LaMDA (Longpre et al., 2023), GLaM (Du et al., 2022), Claude 2 (Anthropic, 2023), Mixtral-8x22B-Instruct (Mistral, 2024), DeepSeek-V2 Chat (DeepSeek, 2024) and only use the official reported results. We also consider two ChatGPT-series models, namely GPT-3.5-turbo ( `gpt-3.5-turbo-0613` ) (OpenAI, 2022) and GPT-4 ( `gpt-4-0613` ) (OpenAI, 2023). (2) _Baselines with retrieval_ , we evaluate models augmented with retrieval. Specifically, we include Atlas (Izacard et al., 2023) and Raven (Huang et al., 2023), two RAG models based on encoderdecoder LMs. For decoder-only models, we consider Self-RAG (Asai et al., 2024a), RECOMP (Xu et al., 2024a), InstructRetro (Wang et al., 2024), RePlug (Shi et al., 2024), RA-DIT (Lin et al., 2024), Llama-3-instruct (Meta-AI, 2024) and ChatQA-1.5 (Liu et al., 2024). We also list the result of RAG pipelines using InstructGPT (175B parameters) as the backbone including GenRead (Yu et al., 2023a), Retrieve-read (Lazaridou et al., 2022) and ReFeed (Yu et al., 2024), but mainly for reference. Other reported numbers are directly comparable if they follow the standard zero-shot settings. 

**Evaluation Metrics.** For OpenQA datasets, we use _Exact Match (EM)_ as the main metric but also report Accuracy for TriviaQA and PopQA and F1 score for HotpotQA and 2WikimQA as it is used in several studies (Asai et al., 2024a; Mallen et al., 2023). For FEVER, we use accuracy as the metric. For ConvQA datasets, we follow (Liu et al., 2024; Wang et al., 2024) to use F1 score as the metric. 

**Implementation Details.** We use Llama3 8B and 70B (Meta-AI, 2024) as the backbone in our main experiments. For the two-stage instruction tuning, we set the batch size to 128 and train the model for 1000 steps with learning rate 5e-6 in Stage-I. Then, we reduce the learning rate to 3e-7 for 8B and 2e-7 for 70B model, set the batch size to 64, and train the model for 3300 steps (around 1 epoch). We use the Adam optimizer (Kingma & Ba, 2014) with _β_ 1 = 0 _._ 9 and _β_ 2 = 0 _._ 98. During the inference stage, we use the December 2018 Wikidump as the corpus index for NQ, TQA, HotpotQA, 2WikimQA, and use the December 2020 Wikidump for PopQA, following (Asai et al., 2024a). By default, we follow (Wang et al., 2024; Lin et al., 2024; Liu et al., 2024) to use the Dragon retriever (Lin et al., 2023) as default and retrieve top- _N_ (100 for 8B and 30 for 70B) documents for ranking, but RankRAG can be adapted to various retrievers and different _N_ (see § 5.3 and 5.5). To ensure a fair comparison, we test the performance of _k ∈{_ 5 _,_ 10 _,_ 20 _}_ and report _the best performance_ for baselines. For generation, we keep temperature _T_ = 0 and set the maximum number of generated token to be 32 for OpenQA, 128 for ConvQA and 8 for others. Training RankRAG-8B uses 32 NVIDIA A100 GPUs for 10 hours (4 hours for Stage-I and 6 hours for Stage-II finetuning), while training RankRAG-70B uses 128 NVIDIA A100 GPUs for 16 hours (4 hours for Stage-I and 12 hours for Stage-II Finetuning). 

**Data Contamination Issues.** One possible issue for the zero-shot evaluation is the test set contamination, where some of the task-specific examples overlap with the instruction fine-tuning data (Oren et al., 2024). To address this issue, we have performed a string match-based analysis where we do not observe any overlap between the train data and data from target tasks. 

## **5.2 Main Experiments** 

Table 2 presents results of RankRAG and baselines. The findings are summarized as follows: 

> 2The results of NQ and TriviaQA using the split from DPR (Karpukhin et al., 2020) are in Appendix F. 

6 

Table 2: Results of RankRAG and baselines on 9 datasets. Unless specified, all results are under _**zero-shot**_ evaluation without additional demonstrations. Results unavailable in public reports are marked as “–”. We use NQ, TriviaQA, and HotpotQA from the KILT benchmark for Llama3-Instruct, Llama3-ChatQA-1.5, and Llama3-RankRAG. Note that _[†]_ : GPT-4 and GPT-4-turbo may refuse to answer the question when retrieved passages do not contain relevant information, thus the EM / accuracy drops after including RAG on TriviaQA, HotpotQA and 2WikimQA. 

|**Task (****_Zero-shot_)**|**NQ**<br>**TriviaQA**<br>**PopQA**<br>**HotpotQA**<br>**2WikimQA**<br>**FEVER**<br>**Doc2Dial**<br>**TopiOCQA**<br>**Inscit**<br>**Avg.**|**NQ**<br>**TriviaQA**<br>**PopQA**<br>**HotpotQA**<br>**2WikimQA**<br>**FEVER**<br>**Doc2Dial**<br>**TopiOCQA**<br>**Inscit**<br>**Avg.**|
|---|---|---|
|**Metric**|EM<br>EM /|Acc.<br>EM / Acc.<br>EM / F1<br>EM / F1<br>Acc.<br>F1<br>F1<br>F1<br>–|
|_Without Retrieval-Augmented Generation_|||
|InstructGPT (Ouyang et al.)<br>PaLM2 540B (0 shot,Anil et al.)<br>PaLM2 540B (**5 shot**,Anil et al.)<br>GLaM 64B (**0 shot**,Du et al.)<br>FLAN-LaMDA 137B (Wei et al.)<br>Claude 2 (**5 shot**,Anthropic)<br>Mixtral-8x22B-Instruct (**5 shot**,Mistral)<br>DeepSeek-V2 236B (**5 shot**,DeepSeek)<br>GPT-3.5-turbo-1106 (OpenAI)<br>GPT-4-0613 (OpenAI)<br>GPT-4-turbo-2024-0409 (OpenAI)|29.9<br>65.8 /<br>21.2<br>76.9<br>37.1<br>86.1<br>37.5<br>71.3<br>20.7<br>68.1<br>–<br>**87.5**<br>40.1<br>82.2<br>53.4<br>86.7<br>38.6<br>82.9 /<br>40.3<br>84.8 /<br>41.5<br>80.0 /|73.2<br>– / –<br>26.0 / 38.2<br>27.2 / 34.8<br>77.6<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>91.7<br>28.4 / 32.2<br>29.9 / 42.0<br>23.9 / 30.4<br>82.7<br>20.1<br>28.5<br>27.2<br>38.5<br> **94.5**<br>31.3 / 34.8<br>34.5 / 46.9<br>29.8 / 36.6<br>87.7<br>27.6<br>30.1<br>27.0<br>42.0<br>94.3<br>25.0 / 33.5<br>26.6 / 43.8<br>24.1 / 35.5<br>87.0<br>27.6<br>26.4<br>24.4<br>38.6|
|_With Retrieval-Augmented Generation_|||
|Atlas 11B (Izacard et al.)<br>Raven 11B (Huang et al.)<br>Self-RAG 7B (Asai et al.)<br>Self-RAG 13B (Asai et al.)<br>RECOMP 20B (Xu et al.)<br>InstructRetro 43B (Wang et al.)<br>RePlug 65B (Shi et al.)<br>RA-DIT 65B (Lin et al.)<br>Llama3-Instruct 8B (Meta-AI)<br>Llama3-Instruct 70B (Meta-AI)<br>Llama3-ChatQA-1.5 8B (Liu et al.)<br>Llama3-ChatQA-1.5 70B (Liu et al.)|26.7<br>56.9<br>29.6<br>65.7<br>–<br>– /<br>–<br>– /<br>37.0<br>59.0<br>38.9<br>78.3<br>28.8<br>72.6<br>35.2<br>75.4<br>30.9<br>70.7 /<br>42.7<br>82.4 /<br>42.4<br>81.0 /<br>47.0<br>85.6<br>/|/ –<br>– / –<br>34.7 / –<br>– / –<br>77.0<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>66.4<br>– / 54.9<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>69.3<br>– / 55.8<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>30.4 / 40.1<br>– / –<br>–<br>–<br>–<br>–<br>–<br>/ –<br>–<br>– / –<br>– / –<br>–<br>36.0<br>–<br>–<br>–<br>/ –<br>– / –<br>32.0 / –<br>– / –<br>73.3<br>–<br>–<br>–<br>–<br>/ –<br>– / –<br>39.7 / –<br>– / –<br>80.7<br>–<br>–<br>–<br>–<br>80.4<br>34.9 / 55.8<br>26.0 / 35.8<br>9.6 / 25.2<br>88.9<br>33.6<br>44.9<br>32.6<br>40.8<br>89.3<br>45.3 / 56.4<br>35.5 / 43.3<br>13.5 / 27.9<br>91.4<br>37.9<br>49.7<br>**36.2**<br>47.1<br>87.6<br>52.6 / 59.8<br>33.4 / 44.6<br>26.8 / 31.9<br>90.9<br>39.3<br>49.9<br>30.1<br>49.6<br> 91.4<br>50.9 / 58.3<br>42.2<br>/54.4<br>34.9<br>/37.4<br>92.7<br>41.3<br>**55.6**<br>32.3<br>53.6|
|Llama3-RankRAG 8B (**0 shot**)<br>50.6<br>82.9 /||89.5<br>57.6<br>/64.1<br>35.3 / 46.7<br>31.4 / 36.9<br>92.0<br>40.4<br>50.4<br>33.3<br>52.6|
|Llama3-RankRAG 70B (**0 shot**)|**54.2**<br>**86.5**/|**92.3**<br>**59.9**/**65.4**<br>**42.7**/**55.4**<br>**38.2**/**43.9**<br>**93.8**<br>**41.5**<br>52.8<br>35.2<br>**56.1**|
|**For reference**: _Using InstructGPT or CodeX (∼175B) (Ouyang et al., 2022) as the Backbone LLM._|||
|GenRead (Yu et al.)<br>Retrieve-Read (Lazaridou et al.)<br>ReFeed (Yu et al.)<br>GPT-3.5-turbo-1106 RAG (OpenAI)<br>GPT-4-0613 RAG_†_ (OpenAI)<br>GPT-4-turbo-2024-0409 RAG_†_ (OpenAI)|32.5<br>66.2 / –<br>46.0 / –<br>36.4 / 39.9<br>– / –<br>80.4<br>–<br>–<br>–<br>–<br>31.7<br>61.4 / –<br>– / –<br>35.2 / 38.0<br>27.7 / –<br>82.7<br>–<br>–<br>–<br>–<br>39.6<br>68.9 / –<br>– / –<br>41.5 / 45.1<br>– / –<br>–<br>–<br>–<br>–<br>–<br>46.7<br>79.7 / 88.0<br>49.9 / 57.0<br>31.2 / 41.2<br>27.2 / 32.2<br>90.8<br>34.8<br>44.3<br>35.3<br>46.8<br>40.4<br>75.0 / 88.5<br>44.3 / 61.4<br>27.6 / 38.1<br>14.4 / 17.6<br>92.6<br>34.2<br>45.1<br>36.4<br>43.5<br>40.3<br>70.2 / 91.1<br>39.5 / 58.4<br>8.1 / 17.9<br>22.8 / 39.2<br>92.2<br>35.4<br>48.3<br>33.8<br>41.6||



**RankRAG outperforms existing RAG methods.** With 8B scale, RankRAG consistently outperforms ChatQA-1.5 8B, one of the most recent open-sourced model with state-of-the-art performance on many RAG benchmarks. RankRAG 8B is also competitive when compared with baseline models with much more parameters. For example, it significantly outperforms InstructRetro (5 _×_ parameters), RA-DIT 65B (8 _×_ paramters), and even outperforms Llama3-instruct 70B (8 _×_ parameters) on NQ and TriviaQA tasks. With more parameters, RankRAG 70B outperforms the strong ChatQA-1.5 70B model, and largely outperforms previous RAG baselines with InstructGPT as the underlying LLM. 

**RankRAG demonstrates larger improvement on more challenging datasets.** We observe that the performance gains of RankRAG over baselines are more pronounced for more challenging QA datasets. For example, on long-tailed QA (PopQA) and multi-hop QA (2WikimQA) tasks, we achieve more than 10% improvement over ChatQA-1.5. These findings suggest that in challenging OpenQA datasets where top documents from retrievers are less relevant to the answer, context ranking effectively enhances performance. In this work we focus on improving single-time retrieval for QA tasks. How to effectively combine multi-round RAG pipelines (Jiang et al., 2023; Khattab et al., 2022; Jeong et al., 2024) with RankRAG is an interesting avenue of future work. 

## **5.3 Ablation Studies** 

**Effect of Designed Components.** Table 3 shows the ablations of RankRAG with Llama3 8B as the backbone on nine general-domain datasets. Overall, we observe all of the proposed components contribute to the final performance. Removing context ranking hurts performance on all tasks, justifying its efficacy in selecting the most relevant contexts for the target question. Besides, the retrieval-augmented QA (RQA) and retrieval-augmented ranking (RAR) designed for instruction finetuning improve outcomes on most tasks by helping the model explicitly pinpoint relevant contexts. On the contrary, the RAFT method used in (Lin et al., 2024) treats each retrieved context separately during instruction finetuning, which yields suboptimal results when compared to RankRAG with the same training data. 

**Performance with Different LLMs.** Table 4 reports the performance of RankRAG and the most 

7 

Table 3: Ablation study of RankRAG. We use Llama3-8B as the backbone. Where ‘RQA’ and ‘RAR’ stands for retrieval-augmented QA and retrieval-augmented ranking data, respectively. For ‘w/o reranking’, we do not perform ranking in the inference stage. 

|**Task (Zero-Shot)**|**NQ**<br>**TriviaQA**<br>**PopQA**<br>**HotpotQA**<br>**2WikimQA**<br>**FEVER**<br>**Doc2Dial**<br>**TopiOCQA**<br>**Inscit**<br>**Avg.**|
|---|---|
|**Metric**|EM<br>EM / Acc.<br>EM / Acc.<br>EM / F1<br>EM / F1<br>Acc.<br>F1<br>F1<br>F1<br>–|
|RankRAG 8B<br>|**50.6**<br>**82.9 / 89.5**<br>**57.6 / 64.1**<br>35.3 /**46.7**<br>31.4 / 36.9<br>92.0<br>**40.4**<br>**50.4**<br>33.3<br>**52.6**<br><br> <br> <br> <br> <br><br><br><br><br>|
|w/o reranking<br>w/o RQA<br>w/o RAR|48.0<br>80.3 / 86.8<br>49.3 / 59.0<br>31.3 / 41.6<br>26.4 / 30.5<br>91.1<br>39.7<br>49.4<br>30.9<br>49.8<br>49.4<br>82.0 / 88.9<br>55.1 / 62.9<br>**35.6**/ 45.9<br>**31.8 / 37.5**<br>**92.1**<br>39.4<br>46.8<br>32.4<br>51.6<br>48.6<br>82.2 / 89.1<br>56.0 / 62.6<br>35.1 / 45.2<br>31.2 / 35.7<br>91.4<br>39.6<br>48.6<br>**33.5**<br>51.8|
|w/ RAFT (Lin et al.)<br>w/ Stage-I SFT Only|43.3<br>80.8 / 87.6<br>48.9 / 56.3<br>30.5 / 41.8<br>25.2 / 29.6<br>91.2<br>36.8<br>46.4<br>30.1<br>48.1<br>38.3<br>63.7 / 76.6<br>49.8 / 54.6<br>26.5 / 40.3<br>18.0 / 25.9<br>85.7<br>33.3<br>33.7<br>30.5<br>42.2|



Table 4: Zero-shot evaluation using Llama2 (Touvron et al., 2023) model as the backbone. 

|**Task (Zero-Shot)**|**NQ**<br>**TriviaQA**<br>**PopQA**<br>**HotpotQA**<br>**2WikimQA**<br>**FEVER**<br>**Doc2Dial**<br>**TopiOCQA**<br>**Inscit**<br>**Avg.**|
|---|---|
|**Metric**|EM<br>EM / Acc.<br>EM / Acc.<br>EM / F1<br>EM / F1<br>Acc.<br>F1<br>F1<br>F1<br>–|
|Llama2-70B (Touvron et al.)|25.3<br>82.4 / –<br>– / –<br>– / –<br>– / –<br>–<br>–<br>–<br>–<br>–|
|Llama2-ChatQA-1.0 7B (Liu et al.)<br>Llama2-ChatQA-1.0 13B (Liu et al.)<br>Llama2-ChatQA-1.0 70B (Liu et al.)|41.4<br>77.8 / 86.5<br>46.7 / 55.0<br>28.9 / 40.3<br>24.0 / 27.5<br>85.9<br>37.9<br>45.5<br>31.0<br>46.6<br>47.9<br>80.9 / 87.6<br>51.8 / 56.2<br>32.9 / 43.2<br>27.6 / 31.1<br>87.6<br>38.1<br>48.9<br>30.8<br>49.6<br>49.5<br>83.2 / 89.7<br>52.1 / 56.6<br>39.0 / 49.4<br>28.9 / 34.1<br>91.7<br>38.9<br>51.0<br>31.9<br>51.8|
|Llama2-RankRAG 7B|46.9<br>84.0 / 89.6<br>55.9 / 61.3<br>32.2 / 43.2<br>26.8 / 30.7<br>86.6<br>38.6<br>49.2<br>32.3<br>50.3|
|Llama2-RankRAG 13B|50.5<br>84.5 / 91.0<br>58.0 / 63.9<br>36.4 / 47.3<br>29.5 / 34.2<br>91.7<br>39.5<br>49.2<br>33.4<br>52.5|
|Llama2-RankRAG 70B|53.2<br>85.8 / 92.1<br>58.7 / 64.5<br>41.8 / 53.1<br>33.8 / 38.8<br>91.9<br>41.2<br>52.9<br>35.8<br>55.0|



recent baseline ChatQA using Llama2 with backbone having varying amounts of parameters. Notably, there exist consistent gains in terms of the average performance (7.8%/6.4%/6.3% on 7B/13B/70B variants respectively), justifying the advantage of RankRAG across different LLM types and scales. 

## **Performance with Different Retrievers.** Figure 3 

exhibits the performance of RankRAG and ChatQA1.5 with different dense retrievers on three representative tasks, where we consider DPR (Karpukhin et al., 2020) and Contriever-MS MARCO (Izacard et al., 2022) as two variants. We note that although the initial retrieved result is not good enough, RankRAG still surpasses ChatQA-1.5 by more than 10% for both retrievers on average. To summarize, RankRAG is robust to the choice of retrievers. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs_images/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs.pdf-0008-07.png)


**----- Start of picture text -----**<br>
ChatQA-1.5 RankRAG ChatQA-1.5 RankRAG<br>80<br>60 60<br>40 40<br>20 20<br>NQ TQA PopQA NQ TQA PopQA<br>(a) DPR (b) Contriever<br>Exact Match Exact Match<br>**----- End of picture text -----**<br>


Figure 3: Performance with different retrievers. The performance of Recall is in Appendix E.1. 

## **5.4 Experiment on Domain-specific RAG Benchmarks** 

To demonstrate that RankRAG can adapt to specialized domains, we conduct experiments on Mirage (Xiong et al., 2024), a recently introduced RAG benchmark for the biomedical field. We follow Xiong et al. (2024) to employ MedCPT (Jin et al., 2023) as the retriever _R_ with MedCorp[3] as the corpus _D_ . 

Table 5: The performance of RankRAG on Mirage, a zero-shot biomedical RAG benchmark. RankRAG and baselines use retrieval by default. Most of numbers are from (Xiong et al., 2024). 

||**Datasets**|**MMLU-med**|**PubmedQA**|**BioASQ**|**MedQA**|**MedMCQA**|**Avg.**|
|---|---|---|---|---|---|---|---|
||GPT-4-0613 (OpenAI)<br>GPT-3.5 (OpenAI)<br>Mixtral 8*7B (Jiang et al.)<br>Llama2 70B (Touvron et al.)<br>Meditron 70B (Chen et al.)<br>PMC-llama 13B (Wu et al.)<br>Llama3-ChatQA-1.5 8B (Liu et al.)<br>Llama3-ChatQA-1.5 70B<br>Llama3-RankRAG 8B<br>Llama3-RankRAG 70B|87.24<br>75.48<br>75.85<br>54.55<br>65.38<br>52.53<br>61.40<br>80.51<br>64.55<br>81.44|70.60<br>67.40<br>67.60<br>50.40<br>56.40<br>42.58<br>66.40<br>74.80<br>65.00<br>79.80|92.56<br>90.29<br>87.54<br>73.95<br>76.86<br>48.29<br>82.69<br>83.17<br>84.44<br>90.76|82.80<br>66.61<br>60.02<br>44.93<br>49.57<br>56.00<br>42.36<br>68.89<br>48.86<br>69.21|66.65<br>58.04<br>56.42<br>43.08<br>52.67<br>65.21<br>46.97<br>62.54<br>56.90<br>69.11|79.97<br>71.56<br>69.49<br>53.38<br>60.18<br>52.92<br>59.96<br>73.98<br>63.95<br>78.06|



The experiment results of RankRAG and baselines are shown in Table 5. From the table, we observe that RankRAG, even without fine-tuning on the biomedical domain, excels at medical QA tasks. Notably, RankRAG 8B surpasses Meditron 70B—a leading open-source LLM for the medical domain—by 6.3%. Besides, RankRAG 70B attains more than 98% performance of GPT-4. These results justify RankRAG’s capacity to be readily applied to new domains without extra post-training. 

## **5.5 A Closer Look at the Ranking Module** 

As the context ranking serves as a core step in RankRAG, we take a closer look at this component. All the studies are done using Llama3-8B as the backbone. 

> 3Link: `https://huggingface.co/MedRAG` . Detailed dataset information is in Appendix A.2. 

8 

Table 6: Ranking performance with different ranking models. Unless specified, all baselines are used to rank the top 100 retrieved passages. RankRAG achieves better performance despite using fewer ranking data. _[∗]_ NQ, TriviaQA and HotpotQA are used for training the BGE-Reranker model. _[†]_ : Our re-implementation. _[‡]_ We only rerank top-30 passages for GPT-4 due to budget constraint. 

|**Task**<br>**Recall**|**Task**<br>**Recall**|**# Rank D**|**# Rank D**|**# Rank D**|**# Rank D**|**ata**|**NQ**|**NQ**|**NQ**|**NQ**|**Trivia**<br>R@5<br>R@1|**QA**<br>0<br>R@20|**QA**<br>0<br>R@20|**PopQA**<br>R@5<br>R@10<br>R@20|**PopQA**<br>R@5<br>R@10<br>R@20|**PopQA**<br>R@5<br>R@10<br>R@20|**PopQA**<br>R@5<br>R@10<br>R@20|**H**|**otpotQA**|**otpotQA**|**Inscit**<br>R@5<br>R@10<br>R@20|**Inscit**<br>R@5<br>R@10<br>R@20|**Inscit**<br>R@5<br>R@10<br>R@20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||R@5<br>R@10<br>R@20|||||||||||R@5|R@10<br>R@20|||||
|**Backbo**<br>Dragon||**ne Retriever**<br>(Lin et al.)<br>–|||||74.9<br>80.3<br>84.3||||89.0<br>92.9|95.3||69.6<br>76.9<br>82.6||||47.5|52.4<br>60.1||43.4<br>56.0<br>64.9|||
|**Finetun**<br>RankBE<br>monoT5<br>BGE-Re<br>RankLL<br>ChatQA||**ed Baseline Ranking Model**<br>RT 110M (Glass et al.)<br>_∼_503<br>3B (Nogueira et al.)<br>_∼_503<br>rank-v2-m3 568M (Chen et al.)<br>_∼_1.6M<br>aMA 8B_†_ (Ma et al.)<br>_∼_503<br>-1.5 8B (Liu et al.)<br>N/A||||k<br>k<br><br>k|73.5<br>79.3<br>84.0<br>75.6<br>80.9<br>84.9<br>78.0_∗_<br>82.8_∗_<br>85.6_∗_<br>77.8<br>83.1<br>86.0<br>68.2<br>75.7<br>82.0||||88.4<br>92.0<br>90.7<br>93.6<br>91.6_∗_<br>94.5_∗_<br>91.2<br>93.1<br>85.4<br>91.1|95.5<br>95.9<br><br>97.1_∗_<br>96.4<br>94.0||78.7<br>82.8<br>85.5<br>81.0<br>83.6<br>85.9<br>79.6<br>84.5<br>86.9<br>80.1<br>84.3<br>86.8<br>67.3<br>76.7<br>83.5||||54.6<br>54.8<br>58.5_∗_<br>57.1<br>37.4|59.8<br>63.7<br>60.2<br>63.3<br>61.8_∗_<br>65.0_∗_<br>**62.1**<br>64.8<br>45.0<br>53.6||45.6<br>57.1<br>66.7<br>48.6<br>59.4<br>68.8<br>51.3<br>59.8<br>69.7<br>57.8<br>62.1<br>71.3<br>32.3<br>42.6<br>54.9|||
|**Off-the-**<br>GPT-3.5<br>GPT-4_‡_||**shelf LLM Reranker**<br>(**top 100**,OpenAI)<br>Unk.<br> (**top 30**,OpenAI)<br>Unk.|||||77.8<br>82.5<br>85.7<br>79.3<br>83.2<br>85.1||||91.1<br>94.4<br>92.8<br>95.5|96.7<br>96.8||77.4<br>82.0<br>85.5<br>79.3<br>83.6<br>86.2||||52.1<br>53.2|56.6<br>62.4<br>57.0<br>61.0||50.2<br>59.1<br>68.6<br>52.3<br>61.7<br>70.0|||
|**Our Mo**||**del**||||||||||||||||||||||
|RankRA||G 8B**(top 100)**<br>_∼_50k|||||80.3<br>**84.0**<br>**86.3**||||93.2<br>95.4|**97.3**||81.6<br>**84.9**<br>**87.0**||||**57.6**|61.8<br>**65.2**||60.9<br>65.7<br>73.5|||
|RankRA||G 70B**(top 30)**<br>_∼_50k|||||**80.6**<br>**84.0**<br>85.4||||**93.6**<br>**95.9**|97.1||**81.8**<br>84.6<br>86.5||||56.3|59.7<br>62.2||**61.3**<br>**66.4**<br>**74.6**|||
|0k<br>5k<br>10k<br>50k<br># Ranking Data<br>40<br>45<br>50<br>Exact Match<br>ChatQA-1.5 (NQ)<br>RankRAG (NQ)<br>Figure 4: Performance<br>w.r.t. # Ranking Data<br>42<br>45<br>48<br>51<br>Exact Match||ChatQA-1.5 (NQ)<br>RankRAG (NQ)<br>||||||||||||||||||||||
|||ChatQA-1.5 (NQ)<br>RankRAG (NQ)||||||||||||||||||||||
||40<br>45<br>50|||||||N=30<br>N=40<br>N|=50|N=100|||~~N=~~|~~30~~<br>N=40<br>N|=50|N=100||0<br>34<br>35<br>Exact Match|||N|=100||
||||||||||||||||||||N=20|N=40<br>N=5|0|||
||||||||N|=20||||||||||||||||
|||||||||||||N|=20<br>||||A-1.5<br>AG|||||||
|||||||||||~~ChatQ~~<br>RankR|~~A-1.5~~<br>AG|||||ChatQ<br>RankR||||~~N=30~~||~~ChatQ~~<br>RankR|~~A-1.5~~<br>AG|
|||||||0|2<br>4<br>6<br>8<br>Inference Time|||||0<br>2<br>||4<br>6<br>8<br>1<br>Inference Time|||||0<br>2<br>4<br>Infer|||||
|||||||||||||||||||||||||



**RankRAG is Data-efficient.** Previous approaches that infuse context ranking into the RAG pipeline usually involve a separate reranking model. To compare our model with these baselines, we evaluate four models (BERT (Glass et al., 2022)/T5 (Nogueira et al., 2020)/Llama3 (Ma et al., 2023)) fine-tuned on the full MS MARCO passage ranking dataset, a strong off-the-shelf reranker model BGE-ranker, and two OpenAI GPT-series models. For the GPT-series models, we use the token probability of ‘ `True` ’ as a proxy for the relevance score[4] . These models are then used to rerank top-retrieved passages by Dragon, similar to our approach. Surprisingly, as shown in Table 6, RankRAG achieves better recall over dedicated ranking models trained on 10 _×_ more ranking data for most cases. Besides, RankRAG can still outperform the BGE-ranker on most tasks, which has been extensively trained on more than 1 million ranking pairs, including some that overlap with our evaluation tasks. This advantage is likely due to the adaptable nature of our model’s training, where the ranking data closely resembles the general RAG fine-tuning data. Directly using ChatQA-1.5 to rank passages _hurts_ the performance, indicating the necessity of incorporating ranking data into instruction fine-tuning. 

We further study the relation between the number of context ranking data and final performance. As shown in Figure 4, with 5k ranking data only ( _∼_ 1% of the MS MARCO dataset), RankRAG can already obtain very compelling results, while further increasing the number of ranking data to 50k yields non-marginal gains. This finding confirms RankRAG’s data efficiency – achieving effective performance with a modest amount of ranking data and maintaining adaptability across various tasks. 

**Performance v.s. Time-efficiency for RankRAG.** One specific caveat for scaling up model size is the increment in the latency overhead — as mentioned in §4.3, it requires sample-wise ranking which incurs additional time. To study the relation between the time efficiency and performance, we change the _N_ used in reranking and plot the relation of _N_ and final accuracy in Figure 5, from which we observe that even with _N_ = 20, RankRAG still improve the baseline model without reranking. While reranking across _N_ = 20 to 100 improves the exact match score by 5.9% to 9.1% across three tasks, it incurs an additional 0 _._ 9 _×_ to 6 _._ 0 _×_ increase in time – _significantly less_ than the 20 _×_ to 100 _×_ increase one might expect. 

## **5.6 Case Study** 

Table 7 presents a case study on NQ dataset, where we observe that using retriever only yield noisy contexts, as there are several distractors, and some contexts (e.g. Passage 4/5 for ChatQA-1.5) are 

> 4 `https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs` 

9 

Table 7: A case study on the top-retrieved context and predictions on NQ dataset, illustrating the effectiveness of RankRAG-8B over ChatQA-1.5-8B. Red text denotes distractors, while green stands for evidences. RankRAG is able to find the correct answer via extract more evidence with reranking. 

**Q** : who hosted and won the inagural world cup? **A** : Uruguay **Passage 1** : FIFA World Cup second round on home soil in 1982. England (1966) won its only title while playing as a host nation. Uruguay (1930), Italy (1934), Argentina (1978) and France (1998) won their first titles as host nations but have gone on to win again, while Germany (1974) won their second title on home soil... 

**Passage 2** : FIFA World Cup hosts country is now chosen in a vote by FIFA’s Congress ... Only Mexico, Italy, France, Germany (West Germany) until shortly after the 1990 World Cup) and Brazil have hosted the event on two occasions. 

**ChatQA-1.5 Passage 3** : CONCACAF hosts, beating the bids of Canada and the United States, and thereby became the first nation to host two World Cups. This second World Cup in Mexico came 16 years after the first one in 1970... **Passage 4** : 1998 FIFA World Cup Africa made their first appearances in the finals. France was awarded the 1998 World Cup on 2 July 1992 by the executive committee of FIFA during a general meeting in Zürich, Switzerland. They defeated Morocco by 12 votes to 7. **Passage 5** : 2026 FIFA World Cup be hosted by one of the remaining four confederations: CONCACAF (North America; last hosted in 1994), CAF (Africa; last hosted in 2010), CONMEBOL (South America; last hosted in 2014), or OFC (Oceania, never hosted before)... **Prediction** : _Germany_ (✘) **Passage 1** : FIFA World Cup second round on home soil in 1982. England (1966) won its only title while playing as a host nation. Uruguay (1930), Italy (1934), Argentina (1978) and France (1998) won their first titles as host nations but have gone on to win again, while Germany (1974) won their second title on home soil **Passage 2** : Timeline of association football penalty kicks.... Thirteen teams enter the first World Cup, held in Uruguay. The hosts beat Argentina 4–2 in the final. Contested between the top national teams of continental Europe, Dr. Gerö Cup’ first edition is won by Italy. **Passage 3** : The Uruguay national football team represents Uruguay in international association football and is controlled by the Uruguayan Football **RankRAG** Association. They have won the Copa América 15 times, the most successful national team in the tournament, the most recent title being the 2011 edition. The team has won the FIFA World Cup twice, including the first World Cup in 1930 as hosts, defeating Argentina 4–2 in the final. **Passage 4** : FIFA World Cup hosts country is now chosen in a vote by FIFA’s Congress. The decision is currently made roughly seven years in advance of the tournament, though the hosts for the 2022 tournament were chosen at the same time as those for the 2018 tournament. **Passage 5** : CONCACAF hosts, beating the bids of Canada and the United States, and thereby became the first nation to host two World Cups. This second World Cup in Mexico came 16 years after the first one in 1970... **Prediction** : _Uruguay_ (✓) 

unhelpful. However, the utilization of reranking uncovers _two additional_ relevant passages, aiding the model in providing the correct answer. More case studies are provided in Appendix G. 

## **6 Conclusion** 

In this work, we introduce a new RAG framework, RankRAG, which instruction-tunes a single LLM for both ranking and answer generation. We find that the instruction tuned LLMs can outperform existing expert ranking models by only adding a small fraction of ranking data into the training blend. We compare our RankRAG with the state-of-the-art RAG models on comprehensive knowledgeintensive benchmarks and demonstrate RankRAG significantly outperform all of them on nine general-domain and five biomedical benchmarks for RAG. 

## **References** 

- Adlakha, V., Dhuliawala, S., Suleman, K., de Vries, H., and Reddy, S. Topiocqa: Open-domain conversational question answering with topic switching. _TACL_ , 2022. 

- Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

Anthropic. Model card and evaluations for claude models. 2023. 

- Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _ICLR_ , 2024a. 

- Asai, A., Zhong, Z., Chen, D., Koh, P. W., Zettlemoyer, L., Hajishirzi, H., and Yih, W.-t. Reliable, adaptable, and attributable language models with retrieval. _arXiv preprint arXiv:2403.03187_ , 2024b. 

- Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., et al. Ms marco: A human generated machine reading comprehension dataset. _arXiv preprint arXiv:1611.09268_ , 2016. 

- Berant, J., Chou, A., Frostig, R., and Liang, P. Semantic parsing on freebase from question-answer pairs. In _EMNLP_ , 2013. 

- Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B., Damoc, B., Clark, A., et al. Improving language models by retrieving from trillions of tokens. In _ICML_ . PMLR, 2022. 

10 

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2023a. 

- Chen, Z., Cano, A. H., Romanou, A., Bonnet, A., Matoba, K., Salvi, F., Pagliardini, M., Fan, S., Köpf, A., Mohtashami, A., et al. Meditron-70b: Scaling medical pretraining for large language models. _arXiv preprint arXiv:2311.16079_ , 2023b. 

- Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. _JMLR_ , 25(70), 2024. 

- Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. Free Dolly: Introducing the world’s first truly open instruction-tuned llm, 2023. 

- Dasigi, P., Liu, N. F., Marasovi´c, A., Smith, N. A., and Gardner, M. Quoref: A reading comprehension dataset with questions requiring coreferential reasoning. In _EMNLP_ , 2019. 

- DeepSeek. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. 

- Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. Glam: Efficient scaling of language models with mixture-of-experts. In _ICML_ , 2022. 

- Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In _NAACL_ , 2019. 

- Fan, A., Jernite, Y., Perez, E., Grangier, D., Weston, J., and Auli, M. Eli5: Long form question answering. In _ACL_ , 2019. 

- Feng, S., Wan, H., Gunasekara, C., Patel, S., Joshi, S., and Lastras, L. doc2dial: A goal-oriented document-grounded dialogue dataset. In _EMNLP_ , 2020. 

- Glass, M., Rossiello, G., Chowdhury, M. F. M., Naik, A., Cai, P., and Gliozzo, A. Re2G: Retrieve, rerank, generate. In _NAACL_ , 2022. 

- Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. Retrieval augmented language model pre-training. In _ICML_ , 2020. 

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. In _ICLR_ , 2021. 

- Ho, X., Nguyen, A.-K. D., Sugawara, S., and Aizawa, A. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In _COLING_ , 2020. 

- Honovich, O., Scialom, T., Levy, O., and Schick, T. Unnatural instructions: Tuning language models with (almost) no human labor. In _ACL_ , 2023. 

- Huang, J., Ping, W., Xu, P., Shoeybi, M., Chang, K. C.-C., and Catanzaro, B. Raven: In-context learning with retrieval augmented encoder-decoder language models. _arXiv preprint arXiv:2308.07922_ , 2023. 

- Izacard, G. and Grave, E. Leveraging passage retrieval with generative models for open domain question answering. In _EACL_ , 2021. 

- Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., and Grave, E. Unsupervised dense information retrieval with contrastive learning. _TMLR_ , 2022. 

- Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel, S., and Grave, E. Atlas: Few-shot learning with retrieval augmented language models. _JMLR_ , 24(251):1–43, 2023. 

- Jeong, S., Baek, J., Cho, S., Hwang, S. J., and Park, J. C. Adaptive-rag: Learning to adapt retrievalaugmented large language models through question complexity. In _NAACL_ , 2024. 

- Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

11 

- Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., and Neubig, G. Active retrieval augmented generation. In _EMNLP_ , 2023. 

- Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and Szolovits, P. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. _Applied Sciences_ , 11(14):6421, 2021. 

- Jin, Q., Dhingra, B., Liu, Z., Cohen, W., and Lu, X. Pubmedqa: A dataset for biomedical research question answering. In _EMNLP_ , 2019. 

- Jin, Q., Kim, W., Chen, Q., Comeau, D. C., Yeganova, L., Wilbur, W. J., and Lu, Z. Medcpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval. _Bioinformatics_ , 39(11), 2023. 

- Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In _ACL_ , 2017. 

- Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. In _EMNLP_ , 2020. 

- Kasai, J., Sakaguchi, K., yoichi takahashi, Bras, R. L., Asai, A., Yu, X. V., Radev, D., Smith, N. A., Choi, Y., and Inui, K. Realtime QA: What’s the answer right now? In _NeurIPS_ , 2023. 

- Khalifa, M., Logeswaran, L., Lee, M., Lee, H., and Wang, L. Few-shot reranking for multi-hop QA via language model prompting. In _ACL_ , 2023. 

- Khattab, O., Santhanam, K., Li, X. L., Hall, D., Liang, P., Potts, C., and Zaharia, M. Demonstratesearch-predict: Composing retrieval and language models for knowledge-intensive nlp. _arXiv preprint arXiv:2212.14024_ , 2022. 

- Kim, H., Hessel, J., Jiang, L., Lu, X., Yu, Y., Zhou, P., Bras, R. L., Alikhani, M., Kim, G., Sap, M., et al. Soda: Million-scale dialogue distillation with social commonsense contextualization. In _EMNLP_ , 2023. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Koˇcisky, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., and Grefenstette, E.` The narrativeqa reading comprehension challenge. _TACL_ , 2018. 

- Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al. Natural questions: a benchmark for question answering research. _TACL_ , 2019. 

- Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., ES, S., Suri, S., Glushkov, D., Dantuluri, A., Maguire, A., Schuhmann, C., Nguyen, H., and Mattick, A. Openassistant conversations - democratizing large language model alignment. _arXiv preprint arXiv: 2304.07327_ , 2023. 

- Lazaridou, A., Gribovskaya, E., Stokowiec, W., and Grigorev, N. Internet-augmented language models through few-shot prompting for open-domain question answering. _arXiv preprint arXiv:2203.05115_ , 2022. 

- Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catanzaro, B., and Ping, W. Nv-embed: Improved techniques for training llms as generalist embedding models. _arXiv preprint arXiv:2405.17428_ , 2024. 

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _NeurIPS_ , 33, 2020. 

- Lin, K., Tafjord, O., Clark, P., and Gardner, M. Reasoning over paragraph effects in situations. In _Workshop on Machine Reading for Question Answering_ , 2019. 

12 

- Lin, S.-C., Asai, A., Li, M., Oguz, B., Lin, J., Mehdad, Y., Yih, W.-t., and Chen, X. How to train your dragon: Diverse augmentation towards generalizable dense retrieval. In _Findings of EMNLP_ , 2023. 

- Lin, X. V., Chen, X., Chen, M., Shi, W., Lomeli, M., James, R., Rodriguez, P., Kahn, J., Szilvasy, G., Lewis, M., Zettlemoyer, L., and tau Yih, W. RA-DIT: Retrieval-augmented dual instruction tuning. In _ICLR_ , 2024. 

- Liu, Z., Ping, W., Roy, R., Xu, P., Shoeybi, M., and Catanzaro, B. Chatqa: Surpassing gpt-4 on conversational qa and rag. _arXiv preprint arXiv:2401.10225_ , 2024. 

- Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., et al. The flan collection: Designing data and methods for effective instruction tuning. In _ICML_ , 2023. 

- Luan, Y., Eisenstein, J., Toutanova, K., and Collins, M. Sparse, dense, and attentional representations for text retrieval. _TACL_ , 2021. 

- Luo, H., Chuang, Y.-S., Gong, Y., Zhang, T., Kim, Y., Wu, X., Fox, D., Meng, H., and Glass, J. Sail: Search-augmented instruction learning. _arXiv preprint arXiv:2305.15225_ , 2023. 

- Ma, X., Wang, L., Yang, N., Wei, F., and Lin, J. Fine-tuning llama for multi-stage text retrieval. _arXiv preprint arXiv:2310.08319_ , 2023. 

- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., and Hajishirzi, H. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In _ACL_ , 2023. 

- Menon, A., Jayasumana, S., Rawat, A. S., Kim, S., Reddi, S., and Kumar, S. In defense of dualencoders for neural ranking. In _ICML_ , 2022. 

Meta-AI. Llama 3 model card. 2024. 

Mistral. Mixtral 8x22b. 2024. URL `https://mistral.ai/news/mixtral-8x22b/` . 

- Mitra, B., Craswell, N., et al. An introduction to neural information retrieval. _Foundations and Trends® in Information Retrieval_ , 2018. 

- Muennighoff, N., Su, H., Wang, L., Yang, N., Wei, F., Yu, T., Singh, A., and Kiela, D. Generative representational instruction tuning. _arXiv preprint arXiv:2402.09906_ , 2024. 

- Nogueira, R., Jiang, Z., Pradeep, R., and Lin, J. Document ranking with a pretrained sequence-tosequence model. In _Findings of EMNLP_ , 2020. 

- OpenAI. Introducing ChatGPT, 2022. 

OpenAI. GPT-4, 2023. 

- Oren, Y., Meister, N., Chatterji, N. S., Ladhak, F., and Hashimoto, T. Proving test set contamination in black-box language models. In _ICLR_ , 2024. 

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. _NeurIPS_ , 35, 2022. 

- Pal, A., Umapathi, L. K., and Sankarasubbu, M. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In _CHIL_ , 2022. 

- Petroni, F., Piktus, A., Fan, A., Lewis, P., Yazdani, M., De Cao, N., Thorne, J., Jernite, Y., Karpukhin, V., Maillard, J., Plachouras, V., Rocktäschel, T., and Riedel, S. KILT: a benchmark for knowledge intensive language tasks. In _NAACL_ , 2021. 

- Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Shen, J., Liu, T., Liu, J., Metzler, D., Wang, X., et al. Large language models are effective text rankers with pairwise ranking prompting. In _Findings of NAACL_ , 2024. 

- Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. Squad: 100,000+ questions for machine comprehension of text. In _EMNLP_ , 2016. 

13 

Ram, O., Levine, Y., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., and Shoham, Y. In-context retrieval-augmented language models. _TACL_ , 2023. 

- Robertson, S., Zaragoza, H., and Taylor, M. Simple bm25 extension to multiple weighted fields. In _CIKM_ , 2004. 

- Sachan, D. S., Reddy, S., Hamilton, W. L., Dyer, C., and Yogatama, D. End-to-end training of multi-document reader and retriever for open-domain question answering. In _NeurIPS_ , 2021. 

- Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N., and Chen, W. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. In _Findings of EMNLP_ , 2023. 

- Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., and Yih, W.-t. Replug: Retrieval-augmented black-box language models. In _NAACL_ , 2024. 

- Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., and Ren, Z. Is ChatGPT good at search? investigating large language models as re-ranking agents. In _EMNLP_ , 2023. 

- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In _NeurIPS_ , 2021. 

- Thorne, J., Vlachos, A., Christodoulopoulos, C., and Mittal, A. Fever: A large-scale dataset for fact extraction and verification. In _NAACL_ , 2018. 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Trischler, A., Wang, T., Yuan, X., Harris, J., Sordoni, A., Bachman, P., and Suleman, K. Newsqa: A machine comprehension dataset. In _RepL4NLP Workshop at ACL_ , 2017. 

- Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. Interleaving retrieval with chain-ofthought reasoning for knowledge-intensive multi-step questions. In _ACL_ , 2023. 

- Tsatsaronis, G., Balikas, G., Malakasiotis, P., Partalas, I., Zschunke, M., Alvers, M. R., Weissenborn, D., Krithara, A., Petridis, S., Polychronopoulos, D., et al. An overview of the bioasq large-scale biomedical semantic indexing and question answering competition. _BMC bioinformatics_ , 2015. 

- Wang, B., Ping, W., McAfee, L., Xu, P., Li, B., Shoeybi, M., and Catanzaro, B. Instructretro: Instruction tuning post retrieval-augmented pretraining. In _ICML_ , 2024. 

- Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F. Text embeddings by weakly-supervised contrastive pre-training. _arXiv preprint arXiv:2212.03533_ , 2022. 

- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., and Wei, F. Improving text embeddings with large language models. _arXiv preprint arXiv:2401.00368_ , 2023a. 

- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions. In _ACL_ , 2023b. 

- Wang, Z., Araki, J., Jiang, Z., Parvez, M. R., and Neubig, G. Learning to filter context for retrievalaugmented generation. _arXiv preprint arXiv:2311.08377_ , 2023c. 

- Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In _ICLR_ , 2022. 

- Wu, C., Lin, W., Zhang, X., Zhang, Y., Xie, W., and Wang, Y. Pmc-llama: toward building opensource language models for medicine. _JAMIA_ , 2024. 

- Wu, Z., Parish, R., Cheng, H., Min, S., Ammanabrolu, P., Ostendorf, M., and Hajishirzi, H. Inscit: Information-seeking conversations with mixed-initiative interactions. _TACL_ , 2023. 

- Xiong, G., Jin, Q., Lu, Z., and Zhang, A. Benchmarking retrieval-augmented generation for medicine. _arXiv preprint arXiv:2402.13178_ , 2024. 

14 

- Xu, F., Shi, W., and Choi, E. RECOMP: Improving retrieval-augmented LMs with context compression and selective augmentation. In _ICLR_ , 2024a. 

- Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B. Retrieval meets long context large language models. In _ICLR_ , 2024b. 

- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In _EMNLP_ , 2018. 

- Yoran, O., Wolfson, T., Ram, O., and Berant, J. Making retrieval-augmented language models robust to irrelevant context. In _ICLR_ , 2024. 

- Yu, W., Iter, D., Wang, S., Xu, Y., Ju, M., Sanyal, S., Zhu, C., Zeng, M., and Jiang, M. Generate rather than retrieve: Large language models are strong context generators. In _ICLR_ , 2023a. 

- Yu, W., Zhang, H., Pan, X., Ma, K., Wang, H., and Yu, D. Chain-of-note: Enhancing robustness in retrieval-augmented language models. _arXiv preprint arXiv:2311.09210_ , 2023b. 

- Yu, W., Zhang, Z., Liang, Z., Jiang, M., and Sabharwal, A. Improving language models via plug-andplay retrieval feedback, 2024. 

- Yu, Y., Xiong, C., Sun, S., Zhang, C., and Overwijk, A. Coco-dr: Combating distribution shift in zero-shot dense retrieval with contrastive and distributionally robust learning. In _EMNLP_ , 2022. 

- Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Stoica, I., and Gonzalez, J. E. Raft: Adapting language model to domain specific rag. _arXiv preprint arXiv:2403.10131_ , 2024. 

- Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F., and Chua, T.-S. Tat-qa: A question answering benchmark on a hybrid of tabular and textual content in finance. In _ACL_ , 2021. 

- Zhu, Y., Zhang, P., Zhang, C., Chen, Y., Xie, B., Dou, Z., Liu, Z., and Wen, J.-R. Inters: Unlocking the power of large language models in search with instruction tuning. In _ACL_ , 2024. 

15 

## **A Dataset Description** 

The information for 14 datasets used in RankRAG is listed as follows. 

## **A.1 Main Experiments** 

- **NQ** (Kwiatkowski et al., 2019) is a widely used question-answering dataset constructed with Wikipedia. The questions are constructed from the Google search engine, and the answers are identified as text spans in the Wikipedia article. 

- **TriviaQA** (Joshi et al., 2017) is a challenging QA dataset containing question-answer pairs from trivia enthusiasts and independently gathered evidence documents. 

- **PopQA** (Mallen et al., 2023) is an entity-centric QA dataset concentrated on long-tail entities. For PopQA, we follow (Asai et al., 2024a) to use the long-tail subset, consisting of questions on 1399 rare entities whose monthly Wikipedia page views are less than 100. 

- **HotpotQA** (Yang et al., 2018) is a multi-hop QA dataset, where the goal is to answer complex questions that require understanding and linking information from multiple documents. 

- **2WikimQA** (Ho et al., 2020) is also a multi-hop QA designed to test machine understanding across two different Wikipedia entities, evaluating the ability of systems to handle cross-lingual and cross-cultural retrieval and question answering. 

- **FEVER** (Thorne et al., 2018) is a fact verification dataset aimed at supporting research into the automatic verification of factual claims. It consists of claims that are manually verified against evidence from Wikipedia, providing a benchmark for fact-checking systems. 

- **Doc2Dial** (Feng et al., 2020) is a document-grounded conversational QA dataset covering four domains: DMV, SSA, VA, and Student Aid. Each sample comprises a dialogue where a user poses queries regarding the document, and an agent responds those questions. The average document length is around 101K words. 

- **TopiOCQA** (Adlakha et al., 2022) is grounded on the whole Wikipedia. It incorporates topic switching and requires the agent to search the entire Wikipedia for answers to user questions. 

- **INSCIT** (Wu et al., 2023) is also grounded on the whole Wikipedia. It studies the case where user questions are under-specified and require clarification. 

## **A.2 Biomedical Benchmarks** 

- **MMLU-med** (Hendrycks et al., 2021) is a subset of six tasks related to biomedicine, including anatomy, clinical knowledge, professional medicine, human genetics, college medicine, and college biology. It contains 1089 questions in total. 

- **MedQA** (Jin et al., 2021) is collected from the US Medical Licensing Examination, contaiing 1273 four-option multiple-choice questions focused on real-world scenarios from professional medical board exams. 

- **MedMCQA** (Pal et al., 2022) includes multiple-choice questions derived from Indian medical entrance exams, covering 2400 healthcare topics across 21 medical subjects. We use the 4,183question development set from MedMCQA, as the test set lacks provided ground truths. 

- **PubmedQA** (Jin et al., 2019) is a biomedical research QA dataset consisting of 1000 manually annotated questions based on PubMed abstracts. Answers in PubMedQA are structured as yes/no/maybe to reflect the validity of the questions. 

- **BioASQ** (Tsatsaronis et al., 2015) includes 618 questions constructed from biomedical literature without providing the ground truth snippets, challenging RAG systems to infer answers independently. 

## **B Data Blending Details for Ranking-enhanced Instruction Finetuning** 

The dataset blending ratio for Stage-II is as follows: 

- Drop: 0.069 

16 

- narrativeqa: 0.09 

- quoref: 0.026 

- ropes: 0.026 

- Squad (Retrieval-augmented QA): 0.09 

- Squad (Retrieval-augmented Ranking): 0.02 

- WebQuestions (Retrieval-augmented QA): 0.09 

- WebQuestions (Retrieval-augmented Ranking): 0.02 

- newsqa: 0.09 

- tatqa-arithmetic: 0.15 

- tatqa-others: 0.08 

- ConvQA: 0.2 

- MS MARCO ranking: 0.15 

- ConvQA ranking: 0.03 

- SFT: 0.2 

The ratio for each dataset is further normalized to ensure the total ratio equals to 1. 

## **C Prompt Formats of Instruction Tuning** 

## **C.1 Stage I: Supervised Fine-tuning** 

The format template of LLM inputs in stage-I is as follows: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
User:{Question1}
```

```
Assistant:{Answer1}
```

```
...
```

```
User:{LatestQuestion}
```

```
Assistant:
```

## **C.2 Stage-II: Unified Instruction-Tuning for Ranking and Generation** 

The format template of LLM inputs in stage-II are as follows: 

## 1) Context-rich QA data 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage:{(Gold)PassagecontainingrelevantcontextforQA}
```

```
User:{Question1}
```

```
Assistant:{Answer1}
```

17 

```
...
```

```
User:{LatestQuestion}
```

```
Assistant:
```

We tailor specific user instructions for various dataset types. For instance: 

For datasets requiring short answers (such as DROP, NarrativeQA, Quoref, ROPES, SQuAD1.1, SQuAD2.0, NewsQA), we use: "Answer the following question with a short span." 

For datasets that necessitate long answers (such as Synthetic_ConvQA), we instruct: "Please give a full and complete answer for the question." 

For datasets involving arithmetic calculations or number extraction from the context (such as TATQA), we specify: "Answer the following question with a number from the context or through math arithmetic." 

For datasets that may require both short and long answers (such as TAT-QA-Others), we direct: "Answer the following question with a short span, or a full and complete answer." 

2) Retrieval-augmented QA data 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage1:{(Shuffled)Passage1}
```

```
Passage2:{(Shuffled)Passage2}
Passage3:{(Shuffled)Passage3}
Passage4:{(Shuffled)Passage4}
Passage5:{(Shuffled)Passage5}
```

```
...
```

```
User:{Question}
```

```
Assistant:
```

## 3) Context ranking data 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage:{Passage1}
```

```
User:{Forthequestion<question>,accesswhetherthepassageisrelevanttothe
question.ReturnTrueifrelevant,otherwiseFalse.}
```

```
Assistant:
```

## 4) Retrieval-augmented ranking data 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
```

18 

```
foundinthecontext.
```

```
Passage1:{(Shuffled)Passage1}
```

```
Passage2:{(Shuffled)Passage2}
```

```
Passage3:{(Shuffled)Passage3}
```

```
Passage4:{(Shuffled)Passage4}
```

```
Passage5:{(Shuffled)Passage5}
```

```
User:{Forthequestion<question>,accesswhethertheabovepassagesarerelevant
tothequestion.Returnalltherelevantpassageid.}
```

```
Assistant:
```

## **D Prompt Formats of Target Tasks** 

## **D.1 Context Ranking** 

NQ/TriviaQA/HotpotQA/PopQA: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage:{Passage}
```

```
User:{Forthequestion<question>,accesswhetherthepassageisrelevanttothe
question.ReturnTrueifrelevant,otherwiseFalse.}
```

```
Assistant:
```

FEVER: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage:{Passage}
```

```
User:{Fortheclaim<claim>,accesswhetherthepassageisrelevanttothe
claim.ReturnTrueifrelevant,otherwiseFalse.}
```

```
Assistant:
```

Doc2dial, Inscit, TopiocQA: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage:{Passage}
```

```
User:{Question1}
```

19 

```
Assistant:{Answer1}
```

```
...
```

```
User:{Forthequestion<latestquestion>,accesswhetherthepassageisrelevant
tothequestion.ReturnTrueifrelevant,otherwiseFalse.}
```

```
Assistant:
```

## **D.2 RAG** 

NQ/TriviaQA/HotpotQA/PopQA: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage1:{RerankTopPassage1}
```

```
Passage2:{RerankTopPassage2}
Passage3:{RerankTopPassage3}
Passage4:{RerankTopPassage4}
Passage5:{RerankTopPassage5}
```

```
...
```

```
User:{Question}.Answertheabovequestionwithashortphrase.
```

```
Assistant:
```

## Fever: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage1:{RerankTopPassage1}
Passage2:{RerankTopPassage2}
Passage3:{RerankTopPassage3}
Passage4:{RerankTopPassage4}
Passage5:{RerankTopPassage5}
```

```
...
```

```
User:AnswerthefollowingquestionwithTrueorFalse.Istheclaim’<claim>’correct?
Assistant:
```

Doc2dial, Inscit, TopiOCQA: 

```
System:Thisisachatbetweenauserandanartificialintelligenceassistant.
```

20 

```
Theassistantgiveshelpful,detailed,andpoliteanswerstotheuser’squestions
basedonthecontext.Theassistantshouldalsoindicatewhentheanswercannotbe
foundinthecontext.
```

```
Passage1:{RerankTopPassage1}
```

```
Passage2:{RerankTopPassage2}
```

```
Passage3:{RerankTopPassage3}
```

```
Passage4:{RerankTopPassage4}
```

```
Passage5:{RerankTopPassage5}
```

```
User:{Question1}
```

```
Assistant:{Answer1}
```

```
...
User:{LatestQuestion}
```

```
Assistant:
```

## **E Additional Experiment Results** 

## **E.1 Ranking Performance Using DPR and Contriever as Retrievers** _R_ 

Table 8 shows the ranking performance of RankRAG-8B using DPR (Karpukhin et al., 2020) and Contriever (Izacard et al., 2022) on three datasets. There are consistent performance gains for all tasks, indicating that RankRAG can apply to many popular retrieval models to improve the quality of retrieved contents. 

Table 8: Answer Recall Comparison Before and After Ranking on 3 Representative Datasets. 

||**NQ**|**DPR**<br>R@5<br>R@10<br>R@20|**Contriever**<br>R@5<br>R@10<br>R@20<br>67.60%<br>75.24%<br>80.67%|
|---|---|---|---|
||Before Ranking|69.50%<br>76.20%<br>81.00%||
||w/ RankRAG|77.95%<br>81.70%<br>84.56%|75.32%<br>80.18%<br>84.70%|
||**TriviaQA**|**DPR**<br>R@5<br>R@10<br>R@20|**Contriever**<br>R@5<br>R@10<br>R@20<br>81.95%<br>86.76%<br>90.08%|
||Before Ranking|67.80%<br>74.20%<br>80.30%||
||w/ RankRAG|77.73%<br>79.40%<br>84.74%|88.71%<br>90.05%<br>92.59%|
||**PopQA**|**DPR**<br>R@5<br>R@10<br>R@20|**Contriever**<br>R@5<br>R@10<br>R@20<br>60.61%<br>65.54%<br>69.90%|
||Before Ranking|43.60%<br>48.90%<br>54.25%||
||w/ RankRAG|50.32%<br>53.75%<br>57.76%|65.11%<br>68.41%<br>71.77%|



## **E.2 RAG Performance with Different** _k_ 

We also show the performance of RankRAG with different context size _k_ in figure 6. From the result, we observe that different from the trend of vanilla RAG approaches (without ranking), _k_ = 5 already works well for most datasets. This effectiveness stems from the reranking step, which prioritizes the most relevant contexts at the top, reducing the necessity to include additional contexts. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs_images/nips_2024_RankRAG_Unifying_Context_Ranking_with_Retrieval-Augmented_Generation_in_LLMs.pdf-0022-00.png)


**----- Start of picture text -----**<br>
NQ TriviaQA PopQA FEVER<br>52 84 60 93<br>50 92<br>82 55<br>48 91<br>46 80 50 90<br>5 10 20 5 10 20 5 10 20 5 10 20<br>Context Size k<br>Exact Match<br>**----- End of picture text -----**<br>


Figure 6: Performance of RankRAG on different context size _k_ . 

## **F Performance of NQ and Trivia QA on DPR Splits** 

We observe that the NQ and TriviaQA datasets exist in two versions: one used by the DPR (Karpukhin et al., 2020) and FiD (Izacard & Grave, 2021) papers, which include 3610 and 11316 questions for NQ and TriviaQA, respectively. In contrast, the KILT benchmark (Petroni et al., 2021) utilizes only subsets of these, comprising 2837 and 5355 examples for NQ and TriviaQA, respectively. It is noteworthy that many recent studies report performance metrics on these datasets without clarifying which version was employed for evaluation. 

To facilitate an _honest_ and _fair_ comparison, we present the performance of RankRAG on both datasets using the DPR splits in Table 9. Notably, regardless of the subset used, RankRAG consistently outperforms both ChatQA and Llama-3-instruct, our direct competitors, as well as other methods utilizing InstructGPT as backbones. We aim for these results to assist the community in making accurate comparisons when referring to the performance of RankRAG. 

Table 9: Performance Across Models. 

|**Model**|**Model Confguration**|**NQ EM (%)**|**TriviaQA EM / Acc. (%)**|
|---|---|---|---|
|_Representative Baselines_||||
||GPT-3.5-0613|35.2|70.1 / 81.3|
||GPT-3.5-0613 RAG|42.3|65.8 / 76.7|
|OpenAI GPT|GPT-4-0613<br>GPT-4-0613 RAG|37.2<br>36.2|**72.6 / 85.1**<br>61.2 / 75.9|
||GPT-4-turbo-2024-0409|38.3|68.0 / 84.5|
||GPT-4-turbo-2024-0409 RAG|36.3|57.6 / 79.2|
|_Using Llama-2 (Touvron et al., 2023) as the backbone LLM_||||
|Llama-2-Chat|Llama-2 RAG 70B|37.7|65.6 / –|
||Llama-2 7B|37.0|62.4 / 74.3|
|ChatQA-1.0|Llama-2 13B|43.9|66.6 / 76.9|
||Llama-2 70B|45.0|69.8 / 80.2|
||Llama-2 7B|42.4|68.3 / 78.9|
|RankRAG|Llama-2 13B|46.2|69.5 / 80.0|
||Llama-2 70B|48.7|72.3 / 82.6|
|_Using Llama-3 (Meta-AI, 2024) as the backbone LLM_||||
|Llama-3-Instruct|Llama-3-Instruct RAG 8B<br>Llama-3-Instruct RAG 70B|27.6<br>37.3|57.1 / 74.6<br>67.6 / 79.6|
|ChatQA-1.5|Llama-3 8B<br>Llama-3 70B|44.1<br>46.0|65.4 / 75.8<br>69.0 / 80.4|
|RankRAG|Llama-3 8B<br>Llama-3 70B|46.1<br>**50.0**|68.8 / 79.9<br>**72.6**/82.9|



## **G Additional Case Studies** 

Tables 10 and 11 provide additional examples from the PopQA and HotpotQA datasets, which focus on long-tailed and multi-hop QA. These tasks are particularly challenging for retrievers, making it difficult to obtain relevant context from the corpus. Consequently, ChatQA-1.5 often struggles to produce the correct answers. However, the reranking step in RankRAG helps counteract poor initial 

22 

Table 10: A case study on the top-retrieved context and predictions on PopQA dataset, illustrating the effectiveness of RankRAG-8B over ChatQA-1.5-8B. Red text denotes distractors, while green stands for evidences. 

|**Q**: Who is the|author of The Universe Around Us?<br>**A**: James Hopwood Jeans / James Jeans / Sir James Jeans / Sir James Hopwood Jeans|
|---|---|
||**Passage 1**: Hooperis the author of two books published by Smithsonian Books/HarperCollins. The frst, Dark Cosmos: In Search of|
||our Universe’s Missing Mass and Energy (2006) was named a notable book by Seed Magazine. His second book, Nature’s Blueprint:|
||Supersymmetry and the Search for a Unifed Theory of Matter and Force (2008), was called "essential reading" by New Scientist...|
||**Passage 2**: Fraknoiis the author or co-author of 14 books in the feld of astronomy. He was the lead author of Voyages through the|
||Universe, an introductory college astronomy textbook published by Brooks-Cole, which went through three editions. In the 1980s, he|
||co-edited with Byron Preiss two collections of science articles and science fction stories, "The Universe" and "The Planets." With Sidney|
|**ChatQA-1.5**|Wolff, Fraknoi founded and was co-editor of the frst on-line journal devoted to astronomy education, "Astronomy Education Review"...<br>**Passage 3**: The Universe" and "The Planets." WithSidney Wolff, Fraknoifounded and was co-editor of the frst on-line journal devoted|
||to astronomy education, "Astronomy Education Review". He edited two collections of resources for K-12 teachers, The Universe at Your|
||Fingertips and More Universe at Your Fingertips published through the Astronomical Society of...|
||**Passage 4**: Lincoln Kinnear Barnett(1909–1979) was an editor and author, most notably at Life Magazine for many years. Lincoln|
||Barnett wrote a number of books, including "The Universe and Doctor Einstein"|
||**Passage 5**: The Universe Maker is a science fction novel by American authorA.E. van Vogt, published in 1953 by Ace Books as an Ace|
||Double with The World of Null-A. It is based on the author´s "The Shadow Men" (Startling Stories, 1950). Set 400 years into the future,|
||the main character is Morton Cargill, a U.S. Army offcer who served in the Korean War...<br>**Prediction**: Lincoln Barnett (✘)|
||**Passage 1**: The Universe Around Us is a science book written by English astrophysicist Sir James Jeans, frst published in 1929 by the|
||Syndics of the Cambridge University Press....|
||**Passage 2**: These books made Jeans fairly well known as an expositor of the revolutionary scientifc discoveries of his day, especially in|
||relativity and physical cosmology. In 1939, the Journal of the British Astronomical Association reported that Jeans was going to stand as|
||a candidate for|
||**Passage 3**: James Jeansbooks for the lay public, including "The Stars in Their Courses" (1931), "The Universe Around Us", "Through|
|**RankRAG**|Space and Time" (1934), "The New Background of Science" (1933), and "The Mysterious Universe.<br>**Passage 4**: The Universe Around Us no special scientifc knowledge. Parts of the book cover the same ground as various lectures I have|
||recently delivered to University and other audiences, including a course of wireless talks I gave last autumn. It has been found necessary|
||to rewrite these almost in their entirety, so that very few sentences remain in their original form, but those who have asked me to publish|
||my lectures and wireless talks will fnd the substance of them in the present book.|
||**Passage 5**:Lincoln Barnett Lincoln Kinnear Barnett (1909–1979) was an editor and author, most notably at Life Magazine for many years.|
||Lincoln Barnett wrote a number of books, including "The Universe and Doctor Einstein", "The World We Live In", and "The Treasure of|
||Our Tongue". The Universe and Doctor Einstein is a layman´s introduction to the theory of relativity. It includes a foreword by Albert|
||Einstein, and has been reprinted several times...<br>**Prediction**: Sir James Jeans (✓)|



Table 11: A case study on the top-retrieved context and predictions on HotpotQA dataset, illustrating the effectiveness of RankRAG-8B over ChatQA-1.5-8B. Red text denotes distractors, while green stands for evidences. 

|**Q**: Which song did Eminem and Rihanna collaborate on after their other collaboration song in studio album "Unapologetic?"<br>**A**: The Monster|**Q**: Which song did Eminem and Rihanna collaborate on after their other collaboration song in studio album "Unapologetic?"<br>**A**: The Monster|**Q**: Which song did Eminem and Rihanna collaborate on after their other collaboration song in studio album "Unapologetic?"<br>**A**: The Monster|
|---|---|---|
||**Passage 1**: Unapologetic is the seventh studio album by Barbadian singer Rihanna. It was released on November 19, 2012, by Def Jam||
||Recordings and SRP Records. It was recorded between June and November 2012, during promotion of her sixth album, "Talk That Talk"||
||(2011). As executive producer, Rihanna enlisted previous collaborators The-Dream, David Guetta, Chase & Status, and StarGate to work||
||alongside new collaborators such as Parker Ighile, Mike Will Made-It, and Labrinth...||
||**Passage 2**: Def Jam France announced via Twitter that Rihanna would release a new single the|upcoming week while her seventh studio|
||album was scheduled to be released in November 2012. On October 11, 2012, in one of her tweets revealed that the title of her new album||
||is "Unapologetic" alongside with its cover. "What Now" was written by British singer-songwriter Livvi Franc together with Rihanna,||
||Parker Ighile and Nathan Cassells, while the production of the song was done by the latter two. Ighile and Cassells...||
||**Passage 3**: Justin then went on to co write "Stay" with Mikky Ekko and recorded by Barbadian singer Rihanna for her seventh studio||
|**ChatQA-1.5**|album, "Unapologetic" (2012). It features guest vocals by Mikky Ekko, and was released as the second single from the album on 7||
||January 2013. The song reached the top fve of twenty-four countries worldwide including number four in the UK and number three on||
||the US Billboard Hot 100, becoming Rihanna’s twenty-fourth top ten on the latter chart...||
||**Passage 4**: Via her offcial Twitter account, Rihanna posted series of "teasing" tweets announcing her seventh studio album. On October||
||11, 2012, in one of her tweets revealed that the title of her new album is "Unapologetic" alongside with its cover. "Jump" is the overall||
||seventh and fnal single off Unapologetic. It was written by Kevin Cossom and M. B. Williams together with its producers StarGate||
||(Mikkel S. Eriksen and Tor Erik Hermansen) and Chase & Status (Saul Milton||
||**Passage 5**: copies of the song were sold in the UK, making "Love the Way You Lie" the country’s biggest-selling song of 2010. The same||
||year, a sequel to the single, titled "Love the Way You Lie (Part II)" was released as part of Rihanna’s ffth studio album "Loud"; it mainly||
||views matters from the female protagonist perspective. In November 2012, Eminem and Rihanna collaborated again on "Numb", which||
||was included on Rihanna’s seventh album "Unapologetic"...|**Prediction**: Love the Way You Lie (✘)|
||**Passage 1**: The Monster (song). Copies of the song were sold in the UK, making "Love the Way You Lie" the country’s biggest-selling||
||song of 2010. The same year, a sequel to the single, titled "Love the Way You Lie (Part II)" was released as part of Rihanna’s ffth studio||
||album "Loud"; it mainly views matters from the female protagonist perspective. In November 2012, Eminem and Rihanna collaborated||
||again on "Numb", which was included on Rihanna’s seventh album "Unapologetic"....||
||**Passage 2**: "Numb" is a song by Barbadian singer Rihanna from her seventh studio album "Unapologetic" (2012). It features guest||
||vocals by American rapper Eminem, making it the pair’s third collaboration since the two offcial versions of "Love the Way You Lie".||
||Following the album’s release, "Numb" charted on multiple charts worldwide including in Canada, the United Kingdom and the United||
||States. "Numb" lasts for a duration of .||
||**Passage 3**: Eminem also wanted to experiment with "retro, vintage" sounds such as beatbreaks and scratches, and he felt that Rubin could||
|**RankRAG**|help him "take that to another level." Rihanna, with whom Eminem previously collaborated on "Love the Way You Lie" from Eminem’s||
||previous studio effort, "Recovery" (2010), was featured on the song "The Monster". On September 11, 2013, she hinted at the...||
||**Passage 4**: together with Jay-Z, Bono and The Edge for the same campaign to alleviate the 2010 Haiti earthquake. In summer 2010,||
||Rihanna collaborated with rapper Eminem on "Love the Way You Lie", which was a major worldwide success, reaching No. 1 in over 20||
||countries. Reaching number 2, the song became the biggest-selling song of 2010 in the UK and the frst of Rihanna’s singles to sell over a||
||million copies in the country. In October 2010, Rihanna switched managers ...||
||**Passage 5**: Eminem asked for more tracks and subsequently heard "Love the Way You Lie". He chose it and told his manager Paul||
||Rosenberg he wanted to collaborate with the Barbadian singer Rihanna. Eminem told Skyrock, "It’s one of those tracks that I felt like||
||only she could pull it off." Rosenberg sent the track to Rihanna, who accepted Eminem’s request "at the last moment." Eminem then||
||wrote the rapped verses.|**Prediction**: The Monster (✓)|



retrieval by finding more pertinent evidence. Coupled with RAG-oriented finetuning, RankRAG effectively filters out distracting entities and pinpoints the correct answers. 

23 

