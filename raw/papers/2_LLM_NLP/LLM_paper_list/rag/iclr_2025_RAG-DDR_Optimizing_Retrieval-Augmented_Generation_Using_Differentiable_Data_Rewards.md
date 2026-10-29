Arxiv Preprint 

# RAG-DDR: OPTIMIZING RETRIEVAL-AUGMENTED - GENERATION USING DIFFERENTIABLE DATA RE WARDS 

**Xinze Li**[1] _[∗]_ **Sen Mei**[1] _[∗]_ **Zhenghao Liu**[1] _[†]_ **Yukun Yan**[2] _[†]_ **Shuo Wang**[2] **Shi Yu**[2] **Zheni Zeng**[2] **Hao Chen**[2] **Ge Yu**[1] **Zhiyuan Liu**[2] **Maosong Sun**[2] **Chenyan Xiong**[3] 

1Department of Computer Science and Technology, Northeastern University, China 

2Department of Computer Science and Technology, Institute for AI, Tsinghua University, China Beijing National Research Center for Information Science and Technology, China 

3Language Technologies Institute, Carnegie Mellon University, United States 

## ABSTRACT 

Retrieval-Augmented Generation (RAG) has proven its effectiveness in mitigating hallucinations in Large Language Models (LLMs) by retrieving knowledge from external resources. To adapt LLMs for RAG pipelines, current approaches use instruction tuning to optimize LLMs, improving their ability to utilize retrieved knowledge. This supervised fine-tuning (SFT) approach focuses on equipping LLMs to handle diverse RAG tasks using different instructions. However, it trains RAG modules to overfit training signals and overlooks the varying data preferences among agents within the RAG system. In this paper, we propose a Differentiable Data Rewards (DDR) method, which end-to-end trains RAG systems by aligning data preferences between different RAG modules. DDR works by collecting the rewards to optimize each agent with a rollout method. This method prompts agents to sample some potential responses as perturbations, evaluates the impact of these perturbations on the whole RAG system, and subsequently optimizes the agent to produce outputs that improve the performance of the RAG system. Our experiments on various knowledge-intensive tasks demonstrate that DDR significantly outperforms the SFT method, particularly for LLMs with smaller-scale parameters that depend more on the retrieved knowledge. Additionally, DDR exhibits a stronger capability to align the data preference between RAG modules. The DDR method makes generation module more effective in extracting key information from documents and mitigating conflicts between parametric memory and external knowledge. All codes are available at https://github.com/OpenMatch/RAG-DDR. 

## 1 INTRODUCTION 

Large Language Models (LLMs) have demonstrated impressive capabilities in language understanding, reasoning, and planning capabilities across a wide range of natural language processing (NLP) tasks (Achiam et al., 2023; Touvron et al., 2023; Hu et al., 2024). However, LLMs usually produce incorrect responses due to hallucination (Ji et al., 2023; Xu et al., 2024c). To alleviate the problem, existing studies employ Retrieval Augmented Generation (RAG) (Lewis et al., 2020; Shi et al., 2023; Peng et al., 2023) to enhance the capability of LLMs and help LLMs access long-tailed knowledge and up-to-date knowledge from different data sources (Trivedi et al., 2023; He et al., 2021; Cai et al., 2019; Parvez et al., 2021). However, the conflict between retrieved knowledge and parametric memory usually misleads LLMs, challenging the effectiveness of RAG system (Li et al., 2022; Chen et al., 2023; Asai et al., 2024b). 

> _∗_ indicates equal contribution. 

> _†_ indicates corresponding author. 

1 

Arxiv Preprint 

To ensure the effectiveness of RAG systems, existing research has focused on developing various agents to enhance the retrieval accuracy (Gao et al., 2024; Xu et al., 2023b; Jiang et al., 2023; Xu et al., 2024b). These approaches aim to refine retrieval results through query reformulation, reranking candidate documents, summarizing the retrieved documents or performing additional retrieval steps to find more relevant information (Yan et al., 2024; Trivedi et al., 2023; Asai et al., 2023; Yu et al., 2023a). To optimize the RAG system, the methods independently optimize different RAG modules by using the EM method (Singh et al., 2021; Sachan et al., 2021) or build the instruct tuning dataset for Supervised Fine-Tuning (SFT) these LLM based RAG modules (Lin et al., 2023; Asai et al., 2023). However, these SFT-based methods usually train LLMs to overfit the training signals and face the catastrophic forgetting problem (Luo et al., 2023b). 

Current research further aims to optimize these RAG modules for aligning their data preferences and primarily focuses on optimizing a two-agent framework, which consists of a retriever and a generator. Typically, these systems train only the retriever to supply more accurate documents to satisfy the data preference of the generator (Shi et al., 2023; Yu et al., 2023b). Aligning data preferences between the retriever and generator through refining retrieved knowledge is a straightforward approach to improving the effectiveness of RAG models. Nevertheless, the generator still faces the knowledge conflict, making LLMs not effectively utilize the retrieved knowledge during generation (Xie et al., 2024). Thus optimizing each RAG module by using the reward from the entire system during training, is essential for building a more tailored RAG system. 

This paper introduces a Differentiable Data Rewards (DDR) method for end-to-end optimizing agents in the RAG system using the DPO (Rafailov et al., 2024) method. DDR uses a rollout method (Kocsis & Szepesv´ari, 2006) to collect the reward from the overall system for each agent and optimizes the agent according to the reward. Specifically, we follow Asai et al. (2024a) and build a typical RAG system to evaluate the effectiveness of our DDR model. It consists of a knowledge refinement module for selecting retrieved documents and a generation module for producing responses based on queries and refined knowledge. Then we conduct the RAG-DDR model by optimizing the two-agent based RAG system using DDR. Throughout the optimization process, we use the reward from the entire RAG system and iteratively optimize both the generation and knowledge refinement modules to align data preferences across both agents. 

Our experiments on various Large Language Models (LLMs) demonstrate that Differentiable Data Rewards (DDR) outperforms all baseline models, achieving significant improvements over previous method (Lin et al., 2023) in a range of knowledge-intensive tasks. DDR can effectively retrofit LLMs for the RAG modeling and help generate higher-quality responses of an appropriate length. Our further analyses show that the effectiveness of our RAG-DDR model primarily derives from the generation module, which is optimized by the reward from the RAG system. Our DDR optimized generation module is more effective in capturing crucial information from retrieved documents and alleviating the knowledge conflict between external knowledge and parametric memory. Further analyses show that the effectiveness of DDR optimized RAG systems can be generalized even when additional noisy documents are incorporated during response generation. 

## 2 RELATED WORK 

Retrieval-Augmented Generation (RAG) is widely used in various real-world applications, such as open-domain QA (Trivedi et al., 2023), language modeling (He et al., 2021), dialogue (Cai et al., 2019), and code generation (Parvez et al., 2021). RAG models retrieve documents from external corpus (Karpukhin et al., 2020; Xiong et al., 2021) and then augment the LLM’s generation by incorporating documents as the context for generation (Ram et al., 2023) or aggregating the output probabilities from the encoding pass of each retrieved documents (Shi et al., 2023). They help LLMs alleviate hallucinations and generate more accurate and trustworthy responses (Jiang et al., 2023; Xu et al., 2023a; Luo et al., 2023a; Hu et al., 2023; Kandpal et al., 2023). However, retrieved documents inevitably incorporate noisy information, limiting the effectiveness of RAG systems in generating accurate responses (Xu et al., 2023b; 2024b; Longpre et al., 2021; Liu et al., 2024b). 

Some studies have demonstrated that the noise from retrieved documents can mislead LLMs, sometimes resulting in degraded performance even on some knowledge-intensive tasks (Foulds et al., 2024; Shuster et al., 2021; Xu et al., 2024b). Such a phenomenon primarily derives from the knowledge conflict between parametric knowledge of LLMs and external knowledge (Jin et al., 2024; 

2 

Arxiv Preprint 

Longpre et al., 2021; Chen et al., 2022; Xie et al., 2024; Wu et al., 2024). Xie et al. (2024) have demonstrated that LLMs are highly receptive to external evidence when external knowledge conflicts with the parametric memory (Xie et al., 2024). Thus, lots of RAG models focus on building modular RAG pipelines to improve the quality of retrieved documents (Gao et al., 2024). Most of them aim to conduct more accurate retrieval models by employing a retrieval evaluator to trigger different knowledge refinement actions (Yan et al., 2024), prompting LLMs to summarize the query-related knowledge from retrieved documents (Yu et al., 2023a) or training LLMs to learn how to retrieve and utilize knowledge on-demand by self-reflection (Asai et al., 2023). 

Optimizing RAG system is a crucial research direction to help generate more accurate responses. Previous work builds a RAG system based on pretrained language models and conducts an end-toend training method (Singh et al., 2021; Sachan et al., 2021). They regard retrieval decisions as latent variables and then iteratively optimize the retriever and generator to fit the golden answers. Recent research primarily focuses on optimizing LLMs for RAG. INFO-RAG (Xu et al., 2024a) focuses on enabling LLMs with the in-context denoising ability by designing an unsupervised pretraining method to teach LLMs to refine information from retrieved contexts. RA-DIT (Lin et al., 2023) builds a supervised training dataset and then optimizes the retriever and LLM by instruct tuning. However, these training methods focus on training LLMs to fit the training signals and face the issue of catastrophic forgetting during instruct tuning (Luo et al., 2023b). 

Reinforcement Learning (RL) algorithms (Schulman et al., 2017), such as Direct Preference Optimization (DPO) (Rafailov et al., 2024), are widely used to optimize LLMs for aligning with human preferences and enhancing the consistency of generated responses (Putta et al., 2024). Agent Q integrates MCTS and DPO to allow agents to learn from both successful and unsuccessful trajectories, thereby improving their performance in complex reasoning tasks (Putta et al., 2024). STEP-DPO further considers optimizing each inference step of a complex task as the fundamental unit for preference learning, which enhances the long-chain reasoning capabilities of LLMs (Lai et al., 2024). While these models primarily target the optimization of individual agents to improve response accuracy at each step, they do not focus on the effectiveness of data alignment within the multi-agent system. Instead of using SFT methods for RAG optimization (Lin et al., 2023), this paper focuses on using the DPO method to avoid overfitting the training signals and align data preferences across different agents, which is different from above RL-based optimization methods. 

## 3 RAG TRAINING WITH DIFFERENTIABLE DATA REWARDS 

This section introduces the Differentiable Data Rewards (DDR) method. We first introduce the DDR method, which optimizes the agent system by aligning the data preferences between agents (Sec. 3.1). Then we utilize knowledge refinement and generation modules to build the RAG pipeline and utilize the DDR method to optimize agents in this system (Sec. 3.2). 

## 3.1 DATA PREFERENCE LEARNING WITH DIFFERENTIABLE DATA REWARDS 

In a RAG system _V_ = _{V_ 1 _, . . . , Vt, . . . , VT }_ , agents exchange and communicate data. To optimize this system, we first forward-propagate data among agents and then evaluate the performance of the RAG system. Then we backward-propagate rewards to refine the data preferences of each agent. 

**Data Propagation.** During communication, the _t_ -th agent, _Vt_ , acts both as a sender and a receiver. Agent _Vt_ receives data from agent _Vt−_ 1 and simultaneously passes data to agent _Vt_ +1: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0003-09.png)


where _Vt −→yt Vt_ +1 denotes that the agent generates one response _yt_ of the maximum prediction probability and sends it to the agent _Vt_ +1. _x_ ⇝ and ⇝ _yT_ represent sending the input _x_ to the agent system _V_ and getting the output _yT_ from _V_ . The performance of the agent system _V_ can be evaluated by calculating the quality score _S_ ( _yT_ ) of the final output _VT_ of the agent system _V_ . 

**Differentiable Data Reward.** Unlike conventional supervised fine-tuning approaches (Lin et al., 2023), DDR tries to optimize the agent _Vt_ to align the data preference _Vt_ +1: _T_ , making the agent system produce a better response with higher evaluation score _S_ ( _yT_ ) of the whole RAG system. 

3 

Arxiv Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0004-01.png)


Figure 1: The Illustration of End-to-End Retrieval-Augmented Generation (RAG) Training with Our Differentiable Data Reward (DDR) Method. During training, we iteratively optimize the Generation module ( _V_ Gen) and Knowledge Refinement module ( _V_ KR). 

To optimize the agent _Vt_ , DDR aims to propagate the system reward to train the targeted agent. Specifically, we first instruct _Vt_ to sample multiple outputs ˜ _yt_ , which incorporate some perturbations intoSzepesv´ari,the agent2006).system.In detail,Thenweweregardcalculatethe agentsthe reward _Vt_ +1: _rT_ ( _x,_ as ˜ _y_ the _t_ ) withevaluationa rolloutmodel,processfeed(Kocsis _y_ ˜ _t_ to this& subsystem _Vt_ +1: _T_ and calculate the evaluation score _S_ ( _yT_ ) of the generated output _yT_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0004-04.png)


Finally, we maximize the probability of generating _y_ ˜ _t_[+][over] _[y]_[˜] _t[−]_[, where] _[y]_[˜] _t_[+][wins higher reward than] _y_ ˜ _t[−]_[(] _[r]_[(] _[x,]_[ ˜] _[y] t_[+][)] _[ > r]_[(] _[x,]_[ ˜] _[y] t[−]_[)][):] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0004-06.png)


where _σ_ is the Sigmoid function. The parameters of _Vt_ can be trained using the DPO (Rafailov et al., 2024) training loss, aiding the DDR in identifying optimization directions by contrastively learning from the positive ( _y_ ˜ _t_[+][) and negative (] _[y]_[˜] _t[−]_[) outputs:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0004-08.png)


where _β_ is a hyperparameter. _D_ is the dataset containing the input _x_ and its corresponding preference data pairs (˜ _yt_[+] _[,]_[ ˜] _[y] t[−]_[)][.] _[V] t_[ref] is the reference model, which is frozen during training. 

## 3.2 OPTIMIZING A SPECIFIC RAG SYSTEM THROUGH DDR 

Given a query _q_ and a set of retrieved documents _D_ = _{d_ 1 _, . . . , dn}_ , we build a RAG system by employing a knowledge refinement module ( _V_ KR) to filter unrelated documents and a generation module ( _V_ Gen) to produce a response. These two modules can be represented by a two-agent system: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0004-12.png)


_D_ where˜ . _V_ Gen _D_[˜] generates _⊆ D_ and _V_ theKRanswerproduces the select actions to filter out the noise documents inaccording to the query _q_ with filtered documents _D_ ˜ . As _D_ shownto buildin 

4 

Arxiv Preprint 

Figure 1, we iteratively tune different modules to conduct the RAG-DDR model, beginning with the generation module optimization and subsequently focusing on tuning the knowledge refinement module during training this RAG system. In the rest of this subsection, we will explain the details of how to optimize _V_ KR and _V_ Gen using DDR. 

**Knowledge Refinement Module.** We follow Asai et al. (2023) and build the knowledge refinement module _V_ KR to estimate the relevance of each document _di_ to the query _q_ for refinement. We feed both query _q_ and document _di_ to the _V_ KR and ask it to produce the action _y_ KR _[i][∈{]_[“YES”] _[,]_[ “NO”] _[}]_[,] which indicates whether _di_ is retained ( _y_ KR _[i]_[=][ “YES”) or discarded (] _[y]_ KR _[i]_[=][ “NO”):] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0005-03.png)


where _⊕_ denotes the concatenation operation, and InstructKR is a prompt designed for knowledge refinement. Then the refined document collection _D_[˜] = _{d_ 1 _, . . . , dk}_ is constructed, where _k <_ = _n_ . The document _di_ that leads the agent system to achieve the highest evaluation reward _r_ ( _x, y_ KR _[i]_[=] “YES”) is considered positive, while the document _dj_ that results in the lowest reward _r_ ( _x, y_ KR _[j]_[=] “YES”) is regarded as negative. _V_ KR is trained to maximize the probability _P_ ( _y_ KR _[i]_[=][“YES”] _[>] y_ KR _[i]_[=][“NO”] _[|][q, d][i]_[)][ for the][positive document.][As well as,][we maximize the probability] _[ P]_[(] _[y]_ KR _[j]_[=] “NO” _> y_ KR _[j]_[=][ “YES”] _[|][q, d][j]_[)][ to filter out irrelevant documents.] 

**Generation Module.** After knowledge refinement, the query _q_ and filtered documents _D_[˜] = _{d_ 1 _, . . . , dk}_ are fed to the generation module _V_ Gen. The response _y_ ˜Gen is sampled from _V_ Gen: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0005-06.png)


where InstructGen is a prompt for generating a tailored response. To reduce the misleading knowledge from the retrieved documents, we also sample responses using only the query as input: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0005-08.png)


The response that achieves the highest evaluation score _S_ (˜ _y_ Gen) is considered positive ( _y_ ˜Gen[+][), while] the lowest evaluation score is considered negative ( _y_ ˜Gen _[−]_[).][The generation module] _[ V]_[Gen][is optimized] to maximize the probability of generating the positive responsereward. By generating responses _y_ ˜Gen based on documents or _P_ the(˜ _y_ queryGen[+] _[>][y]_ alone,[˜] Gen _[−][|][q,]_ LLMs _[D]_[˜][)][ to win a higher] can learn to balance internal and external knowledge, alleviating the problems related to knowledge conflicts. 

## 4 EXPERIMENTAL METHODOLOGY 

This section first describes datasets, evaluation metrics, and baselines. Then we introduce the implementation details of our experiments. More experimental details are shown in Appendix A.1. 

**Dataset.** In our experiments, we follow RA-DIT (Lin et al., 2023) and use the instruction tuning datasets for training and evaluating RAG models. For all datasets and all baselines, we use bgelarge (Xiao et al., 2023) to retrieve documents from the MS MARCO 2.0 (Bajaj et al., 2016). 

During the training of DDR, we collect ten datasets covering two tasks, open-domain QA and reasoning. Specifically, we randomly sample 32,805 samples for the training set and 2,000 samples for the development set in our experiments. Following previous work (Lin et al., 2023; Xu et al., 2024a), we select the knowledge-intensive tasks for evaluation, including open-domain question answering, multi-hop question answering, slot filling, and dialogue tasks. The open-domain QA tasks consist of NQ (Kwiatkowski et al., 2019), MARCO QA (Bajaj et al., 2016) and TriviaQA (Joshi et al., 2017), which require models to retrieve factual knowledge to help answer the given question. For more complex tasks, such as multi-hop QA and dialogue, we use HotpotQA dataset (Yang et al., 2018) and Wikipedia of Wizard (WoW) (Dinan et al., 2019) for evaluation. Besides, we also employ T-REx (Elsahar et al., 2018) to measure one-hop fact look-up abilities of models. 

**Evaluation.** Following Xu et al. (2024a), we utilize Rouge-L and F1 as evaluation metrics for MARCO QA task and WoW task, respectively. For the rest tasks, we use Accuracy. 

**Baselines.** In our experiments, we compare DDR with five baseline models, including zero-shot models and supervised finetuning models. 

5 

Arxiv Preprint 

Table 1: Overall Performance of Different RAG Models. The **best** and second best results are highlighted. In our experiments, we employ Llama3-8B as the knowledge refinement module and utilize LLMs of varying scales (Llama3-8B and MiniCPM-2.4B) as the generation module. 

|**Method**|**Open-DomainQA**|**Multi-Hop QA**|**Slot Filling**|**Dialogue**|
|---|---|---|---|---|
||**NQ**<br>**TriviaQA**<br>**MARCOQA**|**HotpotQA**|**T-REx**|**WoW**|
|_MiniCPM-2.4B_|||||
|LLM w/o RAG<br>Vanilla RAG (2023)<br>REPLUG (2023)<br>RA-DIT (2023)<br>RAG-DDR (w/ 1-Round)<br>RAG-DDR(w/ 2-Round)|20.1<br>45.0<br>17.1<br>42.2<br>79.5<br>16.7<br>39.4<br>77.0<br>19.4<br>41.8<br>78.6<br>19.6<br>47.0<br>82.7<br>28.1<br>**47.6**<br>**83.6**<br>**29.8**|17.7<br>26.7<br>24.7<br>26.1<br>32.5<br>**33.2**|22.6<br>22.1<br>26.7<br>25.2<br>**32.6**<br>32.1|14.9<br>14.4<br>15.7<br>15.7<br>17.2<br>**18.2**|
|_Llama3-8B_|||||
|LLM w/o RAG<br>Vanilla RAG (2023)<br>REPLUG (2023)<br>RA-DIT (2023)<br>RAG-DDR (w/ 1-Round)<br>RAG-DDR(w/ 2-Round)|35.4<br>78.4<br>17.0<br>46.2<br>84.0<br>20.6<br>45.1<br>83.1<br>22.5<br>46.2<br>87.4<br>20.3<br>50.7<br>88.2<br>25.1<br>**52.1**<br>**89.6**<br>**27.3**|27.7<br>30.1<br>29.4<br>34.9<br>37.3<br>**39.0**|16.0<br>26.9<br>23.1<br>**41.7**<br>37.3<br>40.6|14.6<br>12.5<br>14.0<br>14.8<br>14.9<br>**16.8**|



We first treat the LLM as a black box and conduct three baselines, including LLM w/o RAG, Vanilla RAG and REPLUG. For the LLM w/o RAG model, we directly feed the query to the LLM and ask it to produce the answer according to its memorized knowledge. To implement the vanilla RAG model, we follow previous work (Lin et al., 2023), use retrieved passages as context and leverage the in-context learning method to conduct the RAG modeling. REPLUG (Shi et al., 2023) is also compared, which ensembles output probabilities from different passage channels. Besides, RADIT (Lin et al., 2023) is also compared in our experiments, which optimizes the RAG system using the instruct-tuning method. In our experiments, we reimplement REPLUG and RA-DIT baselines and do not finetune the retriever during our reproduction process, as the retriever we used has already been trained with massive supervised data and is sufficiently strong. 

**Implementation Details** . In our experiments, we employ Minicpm-2.4B-sft (Hu et al., 2024) and lama3-8B-Instruct (Touvron et al., 2023) as backbone models to construct the generation modules, and employ Llama3-8B-Instruct (Touvron et al., 2023) to build the knowledge refinement module. While training the RAG modules with DDR, we use automatic metrics such as Rouge-L and Accuracy to calculate the reward and set _β_ = 0 _._ 1. The learning rate is set to 5e-5, and each model is trained for one epoch. For the generation module, we feed 5 retrieved passages as external knowledge for augmenting the generation process. To optimize both knowledge refinement module and generation module, we use LoRA (Hu et al., 2022) for efficient training. 

## 5 EVALUATION RESULTS 

In this section, we first evaluate the performance of different RAG methods and then conduct ablation studies to show the effectiveness of different training strategies. Next, we examine the effectiveness of DDR training strategies on the generation module ( _V_ Gen) and explore how it balances internal and external knowledge through DDR. Finally, we present several case studies. 

## 5.1 OVERALL PERFORMANCE 

The performance of various RAG models is presented in Table 1. As shown in the evaluation results, RAG-DDR significantly outperforms these baseline models on all datasets. It achieves improvements of 7% compared to the Vanilla RAG model when using MiniCPM-2.4B and Llama3-8B to construct the generation module ( _V_ Gen). 

Compared with LLM w/o RAG, Vanilla RAG and REPLUG significantly enhance LLM performance on most knowledge-intensive tasks, indicating that external knowledge effectively improves the accuracy of generated responses. However, the performance of RAG models decreases on dialogue tasks, showing that LLMs can also be misled by the retrieved documents. Unlike these zero-shot methods, RA-DIT provides a more effective approach for guiding LLMs to filter out noise 

6 

Arxiv Preprint 

Table 2: Ablation Study. Both Vanilla RAG and RAG w/ _V_ KR are evaluated in a zero-shot setting without any fine-tuning. We then use DDR to optimize the knowledge refinement module ( _V_ KR), the generation module ( _V_ Gen), and both modules, resulting in three models: RAG-DDR (Only _V_ KR), RAG-DDR (Only _V_ Gen) and RAG-DDR (All). 

|RAG-DDR(Only_V_Gen)|and RAG-DDR(All).||||
|---|---|---|---|---|
|**Method**|**Open-DomainQA**|**Multi-Hop QA**|**Slot Filling**|**Dialogue**|
||**NQ**<br>**TriviaQA**<br>**MARCOQA**|**HotpotQA**|**T-REx**|**WoW**|
|_MiniCPM-2.4B_|||||
|Vanilla RAG<br>w/_V_KR<br>RAG-DDR (Only_V_KR)<br>RAG-DDR (Only_V_Gen)<br>RAG-DDR(All)|42.1<br>78.0<br>16.6<br>42.2<br>79.5<br>16.7<br>42.5<br>79.6<br>16.8<br>46.8<br>81.7<br>**28.3**<br>**47.0**<br>**82.7**<br>28.1|24.9<br>26.7<br>27.3<br>31.2<br>**32.5**|22.0<br>22.1<br>21.9<br>32.2<br>**32.6**|15.1<br>14.4<br>15.8<br>17.0<br>**17.2**|
|_Llama3-8B_|||||
|Vanilla RAG<br>w/_V_KR<br>RAG-DDR (Only_V_KR)<br>RAG-DDR (Only_V_Gen)<br>RAG-DDR(All)|45.4<br>83.2<br>20.8<br>46.2<br>84.0<br>20.6<br>46.8<br>84.7<br>20.7<br>50.2<br>87.8<br>**25.2**<br>**50.7**<br>**88.2**<br>25.1|28.5<br>30.1<br>30.7<br>36.9<br>**37.3**|26.6<br>26.9<br>28.5<br>36.2<br>**37.3**|12.3<br>12.5<br>12.5<br>14.8<br>**14.9**|



from retrieved content and identify accurate clues for answering questions. Nevertheless, RA-DIT still underperforms compared to Vanilla RAG on certain knowledge-intensive tasks, such as NQ and HotpotQA, showing that overfitting to golden answers is less effective for teaching LLMs to capture essential information for generating accurate responses. In contrast, RAG-DDR surpasses RA-DIT on almost all tasks, particularly with smaller LLMs (MiniCPM-2.4b), achieving a 5% improvement. This highlights the generalization capability of our DDR training method, enabling LLMs of varying scales to utilize external knowledge through in-context learning effectively. 

## 5.2 ABLATION STUDIES 

As shown in Table 2, we conduct ablation studies to explore the role of different RAG modules and evaluate different training strategies using DDR. 

This experiment compares five models, utilizing MiniCPM-2.4b and Llama3-8b to construct the generation module. The Vanilla RAG model relies solely on the generation module ( _V_ Gen) to produce answers based on the query and retrieved documents. RAG w/ _V_ KR adds an additional knowledge refinement module ( _V_ KR) to filter the retrieved documents and then feeds query and filtered documents to the generation module ( _V_ Gen). RAG-DDR (Only _V_ KR) indicates that we tune the RAG w/ _V_ KR model using DDR by only optimizing the knowledge refinement module ( _V_ KR). RAG-DDR (Only _V_ Gen) only optimizes the generation module ( _V_ Gen). RAG-DDR (All) optimizes both _V_ KR and _V_ Gen. 

Compared with the Vanilla RAG model, RAG w/ _V_ KR improves the RAG performance on almost all evaluation tasks, demonstrating the effectiveness of the knowledge refinement module in improving the accuracy of LLM responses. In contrast, RAG-DDR (Only _V_ Gen) shows greater improvements over RAG w/ _V_ KR than DDR (Only _V_ KR), indicating that the primary effectiveness of RAG-DDR comes from optimizing the generation module ( _V_ Gen) through DDR. When we begin with the RAGDDR (Only _V_ Gen) model and subsequently optimize the knowledge refinement module, the performance of RAG with _V_ KR improves. It shows that filtering noise from retrieved documents using feedback from the generation module is effective, which is also observed in previous work (Yu et al., 2023b; Izacard & Grave, 2020). However, the improvements from optimizing the knowledge refinement modules are limited, highlighting that enhancing the generation module’s ability to leverage external knowledge is more critical for the existing RAG system. 

## 5.3 CHARACTERISTICS OF THE GENERATION MODULE IN RAG-DDR 

In this experiment, we explore the characteristics of the generation module ( _V_ Gen) by employing various training strategies, including zero-shot (Vanilla RAG), the SFT method (RA-DIT), and DDR (RAG-DDR). As illustrated in Figure 2, we present the performance of _V_ Gen w/o RAG and _V_ Gen 

7 

Arxiv Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0008-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0008-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0008-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0008-04.png)


**----- Start of picture text -----**<br>
(a) V Gen w/o RAG. (b) V Gen w/ RAG. (c) Average Length of Responses<br>Generated by  V Gen.<br>**----- End of picture text -----**<br>


Figure 2: Characteristics of the Generation Module in RAG Optimized with Different Training Strategies. We use MiniCPM-2.4B to build the generation module ( _V_ Gen) and then train it using different strategies. The performance of _V_ Gen is shown in a zero-shot setting, along with the generation module optimized using the RA-DIT and DDR methods. 

Table 3: Experimental Results on Evaluating the Knowledge Usage Ability of the Generation Module ( _V_ Gen) of Different RAG Models. 

|**Method**|**Has-Answer**|**Miss-Answer**|**Internal Knowledge**|
|---|---|---|---|
||**NQ**<br>**HotpotQA**<br>**T-REx**|**NQ**<br>**HotpotQA**<br>**T-REx**|**NQ**<br>**HotpotQA**<br>**T-REx**|
|_MiniCPM-2.4B_||||
|LLM w/o RAG<br>Vanilla RAG<br>RA-DIT<br>RAG-DDR|27.6<br>26.7<br>36.6<br>59.1<br>51.7<br>36.8<br>58.3<br>47.6<br>41.7<br>**65.5**<br>**56.9**<br>**52.6**|**2.6**<br>**11.7**<br>4.1<br>1.1<br>9.1<br>2.5<br>1.7<br>10.8<br>4.2<br>2.4<br>10.8<br>**5.9**|100.0<br>100.0<br>100.0<br>71.1<br>70.1<br>60.9<br>76.9<br>73.7<br>73.4<br>**82.9**<br>**81.0**<br>**78.5**|
|_Llama3-8B_||||
|LLM w/o RAG<br>Vanilla RAG<br>RA-DIT<br>RAG-DDR|46.4<br>44.7<br>25.5<br>64.2<br>58.0<br>45.5<br>64.1<br>59.7<br>**65.3**<br>**69.5**<br>**64.3**<br>59.9|**6.7**<br>15.9<br>3.7<br>2.9<br>10.5<br>3.0<br>3.4<br>17.9<br>**10.8**<br>4.1<br>**18.0**<br>5.4|100.0<br>100.0<br>100.0<br>80.0<br>67.3<br>66.3<br>81.0<br>79.2<br>77.9<br>**88.4**<br>**82.0**<br>**79.2**|



w/ RAG. These experiments evaluate _V_ Gen’s ability to memorize knowledge and utilize external knowledge. Additionally, we report the average length of responses generated by _V_ Gen. 

As shown in Figure 2(a), we compare the performance of the generation module that relies solely on internal knowledge of parametric memory. Compared to the Vanilla RAG model, RA-DIT demonstrates a decline in performance on the NQ and HotpotQA tasks. Such a phenomenon reveals that the model loses previously acquired knowledge while learning new information during SFT (Luo et al., 2023b). In contrast, DDR not only outperforms the RA-DIT method but also achieves consistent improvements over the Vanilla RAG model across all evaluation tasks. This indicates that DDR can help learn more factual knowledge during training while also preventing the loss of previously memorized information through a reinforcement learning-based training approach. Then we feed retrieved documents to the generation module and show the generation performance in Figure 2(b). The evaluation results indicate that RA-DIT marginally outperforms the Vanilla RAG model, while RAG-DDR significantly improves generation accuracy by utilizing factual knowledge from the retrieved documents. Additional experiments showing the general capabilities of RAGDDR are presented in Appendix A.4. 

Finally, we show the average length of responses generated by _V_ Gen in Figure 2(c). Compared to the Vanilla RAG model, the average length of responses generated by RA-DIT decreases significantly, indicating that the SFT training method tends to cause LLMs to overfit the supervised data. On the contrary, RAG-DDR shows a more similar length distribution with Vanilla RAG model, enabling the model to generate responses of a more appropriate length. It demonstrates that training LLMs to learn data preferences from generated responses can help align the output format of RAG models more closely with that of the original LLMs. 

8 

Arxiv Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0009-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0009-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0009-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0009-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0009-05.png)


**----- Start of picture text -----**<br>
(a) NQ. (b) HotpotQA. (c) WoW. (d) T-REx.<br>**----- End of picture text -----**<br>


Figure 3: Effectiveness of Different RAG Models in Defending Noisy Information. We use MiniCPM-2.4B to build the generation module ( _V_ Gen). And then we retain one informative passage and randomly replace _n_ top-retrieved documents with noisy ones. 

## 5.4 EFFECTIVENESS OF RAG-DDR IN USING EXTERNAL KNOWLEDGE 

In this section, we investigate the capability of the generation module _V_ Gen in the RAG model to leverage external knowledge for response generation. We first evaluate the ability of _V_ Gen to balance the internal and external knowledge. Next, we evaluate the denoising ability by _V_ Gen feeding additional unrelated documents as the external knowledge. 

As shown Table 3, we first show the effectiveness of _V_ Gen in balancing internal and external knowledge during producing responses. We compare three training strategies: zero-shot (Vanilla RAG), SFT (RA-DIT), and DDR (RAG-DDR). Our experiment establishes three testing scenarios to evaluate the effectiveness of different RAG models by categorizing the evaluation data into three distinct scenarios: Has-Answer, Miss-Answer, and Internal Knowledge. The Has-Answer scenario indicates that the retrieved documents contain the golden answer, which can help the generation module to answer the question accurately. The Miss-Answer scenario indicates that the retrieved documents do not contain the golden answer and fail to provide sufficient support for LLMs to generate accurate responses. Lastly, the Internal Knowledge scenario further evaluates the ability of LLMs in dealing with the conflict between internal and external knowledge. The test cases of the Internal Knowledge scenario indicate that LLMs can generate accurate answers using only parametric memory, whereas RAG models produce incorrect responses. 

In the Has-Answer scenario, RAG-DDR and RA-DIT outperform the Vanilla RAG model on all datasets. This indicates that training the generation module enables it to capture essential knowledge facts, improving the accuracy of the generated responses. Compared with RA-DIT, RAG-DDR achieves consistent improvements over the Vanilla RAG model, which shows that DDR can better generalize the external knowledge usage ability to different knowledge-intensive tasks. In the Miss-Answer scenario, all RAG models perform significantly worse compared to the Has-Answer scenario, showing that the retrieved knowledge fails to provide sufficient information for generating accurate responses. However, RAG-DDR and RA-DIT mitigate the performance drop caused by incorporating retrieved documents by fine-tuning the generation module, illustrating their ability to effectively leverage internal knowledge and avoid being misled by irrelevant information. For the Internal Knowledge scenario, we evaluate the ability of RAG models in handling the conflict from external knowledge. The Vanilla RAG model decreases the generation accuracy more than 20% on all tasks, showing that the knowledge conflict can significantly affect the generation result. DDR exhibits strong effectiveness in mitigating the knowledge conflict in the Vanilla RAG model, resulting in a reduction of more than 10% in performance drop. This indicates that DDR effectively balances the utilization of internal and external knowledge, facilitating the robustness of the RAG model. 

In the second experiment, we extend the Has-Answer setting and further investigate the denoising effectiveness of _V_ Gen by adding different number of noise documents. As shown in Figure 3, we increase the noise by randomly replacing _n_ documents from the top-5 retrieved set with the last _n_ documents from the top-100 retrieved candidates, while ensuring that the ground truth document remains in the set. RA-DIT exhibits inconsistent performance and degrades in some RAG scenarios. In contrast, RAG-DDR consistently outperforms the Vanilla RAG model, maintaining a consistent improvement even as the number of noisy documents increases. It further confirms the effectiveness of our RAG-DDR approach in defencing noise. 

9 

Arxiv Preprint 

Table 4: Case Studies. These matching phrases are indicated in **Red** if the model answers correctly and in **Blue** if it answers incorrectly. We present the generated responses of different RAG models, which use MiniCPM-2.4B to build the generation module ( _V_ Gen). 

**Case 1:** When does Canada switch to daylight savings time 

**Document:** Daylight Saving Time Canada. Daylight Saving Time in Canada. In most of Canada Daylight Saving Time begins at 2:00 a.m. local time on **the second Sunday in March** . ... Canada Daylight Saving Time Start and End Dates 2020 March 8 November 1 2019 **March 10** November ... November 6 2010 **March 14** November 7 2009 March 08 November 1 ... 

**Vanilla LLMs:** Canada does not switch to or from daylight savings time. Instead, the clocks are automatically set forward by one hour during the spring. 

**Vanilla RAG: March 14 RA-DIT: March 8** 

**RAG-DDR:** Canada switches to Daylight Saving Time on **the second Sunday of March** and returns to Standard Time on the first Sunday of November. 

**Case 2:** The manager who recruited David Beckham managed Manchester United during what timeframe **Document:** Ferguson’s unprecedented achievements at Aberdeen led to managerial offers from some of the most prestigious clubs in Europe over the years, and he signed with Manchester in November **1986** . ... **Ferguson retired at the end of the 2012–13 Premier League season** but stayed on with Man U in a... 

## **Vanilla LLMs:** 1992–2003 **Vanilla RAG: 1986** –1993 **RA-DIT:** 1992 to **2013** 

**RAG-DDR:** The football manager who recruited David Beckham was Sir Alex Ferguson, and he managed Manchester United from **1986 to 2013** . 

**Case 3:** Following success at the 1979 election whose party conference speeech included the lines ’you turn if you want to, the lady’s not for turning’? 

**Document:** The lady’s not for turning - Wikipedia The lady’s not for turning From Wikipedia, the free encyclopedia Jump to navigation Jump to search 1980 **Margaret Thatcher** speech ... a phrase used by Margaret Thatcher, then Prime Minister, in her speech to **the Conservative Party Conference** ... 

**Vanilla LLMs:** The speaker in question is **Margaret Thatcher** , who was the leader of the Conservative Party and later became the Prime Minister of the United Kingdom. 

**Vanilla RAG: Conservative Party Conference RA-DIT: The Conservative Party RAG-DDR:** The 1979 Conservative Party conference speech by **Margaret Thatcher** included the lines ”you turn if you want to, the lady’s not for turning”. 

## 5.5 CASE STUDIES 

In Table 4, we present three randomly selected examples from the NQ, HotpotQA, and TrivialQA datasets to show the generated responses and evaluate the effectiveness of the RAG-DDR model. 

For the first case, the query asks about the “daylight savings time of Canada” and the retrieved documents contain detailed information about when the daylight savings time begins each year in Canada. However, since the exact date of daylight savings time changes annually, the most accurate answer is “the second Sunday in March”. RAG-DDR shows its effectiveness in accurately answering the question, while both Vanilla RAG and RAG-DIT are misled by noisy information such as “14th March” and “10th March”, leading to incorrect responses. This demonstrates that RAG-DDR model has the ability to distinguish the most accurate knowledge from ambiguous or misleading information in the retrieved documents. In the second case, the model must integrate multiple pieces of knowledge from the provided documents to answer the question. While the Vanilla RAG model and RAG-DIT only correctly answer half of the questions using partial knowledge, RAG-DDR successfully identifies the correct start time and end time. This indicates that RAG-DDR has a stronger capacity to integrate factual knowledge from different document segments. As shown in the third case, Vanilla LLM can answer the question correctly only depending on the parametric memory. Nevertheless, both Vanilla RAG and RA-DIT are misled by the confusing information from these retrieved documents and generate the response “the Conservative Party Conference”, which is entirely unrelated to the given question. In contrast, RAG-DDR accurately follows the intent of the question for generating the response, demonstrating the ability of RAG-DDR to mitigate the negative influence of external knowledge. 

10 

Arxiv Preprint 

## 6 CONCLUSION 

This paper proposes Differentiable Data Rewards (DDR), a method aimed at end-to-end optimizing the Retrieval-Augmented Generation (RAG) model using the DPO method. DDR optimizes each agent by collecting the reward in a rollout way and aligns data preferences among these communicative agents. We build a two-agent RAG system and optimize it using DDR to implement the RAG-DDR model. Our experiments demonstrate that DDR helps the generation module produce responses of an appropriate length and avoids overfitting the training signals during SFT. Our further analyses reveal that the DDR optimized generation model can better capture key information from retrieved documents and mitigate the conflict between external knowledge and parametric memory. 

## REFERENCES 

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _ArXiv preprint_ , 2023. URL https://arxiv.org/abs/2303.08774. 

Shourya Aggarwal, Divyanshu Mandowara, Vishwajeet Agrawal, Dinesh Khandelwal, Parag Singla, and Dinesh Garg. Explanations for CommonsenseQA: New Dataset and Models. In _Proceedings of ACL_ , pp. 3050–3065, 2021. URL https://aclanthology.org/2021.acl-long.238. 

- Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In _Proceedings of NAACL-HLT_ , pp. 2357–2367, 2019. URL https://aclanthology. org/N19-1245. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. _ArXiv preprint_ , 2023. URL https: //arxiv.org/abs/2310.11511. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations_ , 2024a. 

- Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi, and Wen-tau Yih. Reliable, adaptable, and attributable language models with retrieval. _ArXiv preprint_ , 2024b. URL https://arxiv.org/abs/2403.03187. 

- Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. Ms marco: A human generated machine reading comprehension dataset. _ArXiv preprint_ , 2016. URL https://arxiv.org/abs/ 1611.09268. 

- Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In _Proceedings of EMNLP_ , pp. 1533–1544, 2013. URL https: //aclanthology.org/D13-1160. 

- Deng Cai, Yan Wang, Wei Bi, Zhaopeng Tu, Xiaojiang Liu, Wai Lam, and Shuming Shi. Skeletonto-response: Dialogue generation guided by retrieval memory. In _Proceedings of NAACL-HLT_ , pp. 1219–1228, 2019. URL https://aclanthology.org/N19-1124. 

- Hung-Ting Chen, Michael Zhang, and Eunsol Choi. Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence. In _Proceedings of EMNLP_ , pp. 2292–2307, 2022. URL https://arxiv.org/pdf/2210.13701. 

- Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. _ArXiv preprint_ , 2023. URL https://arxiv.org/abs/2309. 01431. 

- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. _ArXiv preprint_ , 2018. URL https://arxiv.org/abs/1803.05457. 

11 

Arxiv Preprint 

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _ArXiv preprint_ , 2021. URL https://arxiv.org/abs/2110.14168. 

- Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. Wizard of wikipedia: Knowledge-powered conversational agents. In _Proceedings of ICLR_ , 2019. URL https://openreview.net/forum?id=r1l73iRqKm. 

- Hady Elsahar, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon Hare, Frederique Laforest, and Elena Simperl. T-REx: A large scale alignment of natural language with knowledge base triples. In _Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)_ , 2018. URL https://aclanthology.org/L18-1544. 

- Philip Feldman Foulds, R James, and Shimei Pan. Ragged edges: The double-edged sword of retrieval-augmented chatbots. _ArXiv preprint_ , 2024. URL https://arxiv.org/abs/2403. 01193. 

- Yunfan Gao, Yun Xiong, Meng Wang, and Haofen Wang. Modular rag: Transforming rag systems into lego-like reconfigurable frameworks. _ArXiv preprint_ , 2024. URL https://arxiv.org/abs/ 2407.21059. 

- Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of the Association for Computational Linguistics_ , pp. 346–361, 2021. URL https://arxiv.org/ pdf/2101.02235. 

- Junxian He, Graham Neubig, and Taylor Berg-Kirkpatrick. Efficient nearest neighbor language models. In _Proceedings of EMNLP_ , pp. 5703–5714, 2021. URL https://arxiv.org/pdf/ 2109.04212. 

- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. _ArXiv preprint_ , 2020. URL https://arxiv.org/abs/2009.03300. 

- Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao. Chatdb: Augmenting llms with databases as their symbolic memory. _ArXiv preprint_ , 2023. URL https: //arxiv.org/abs/2306.03901. 

- Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _Proceedings of ICLR_ , 2022. URL https://arxiv.org/pdf/2106.09685. 

- Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. _ArXiv preprint_ , 2024. URL https://arxiv.org/abs/2404. 06395. 

- Gautier Izacard and Edouard Grave. Distilling knowledge from reader to retriever for question answering. _ArXiv preprint_ , 2020. URL https://arxiv.org/abs/2012.04584. 

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Computing Surveys_ , 2023. URL https://arxiv.org/pdf/2202.03629. 

- Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In _Proceedings of EMNLP_ , pp. 7969–7992, 2023. URL https://aclanthology.org/2023.emnlp-main.495. pdf. 

- Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Li Qiuxia, and Jun Zhao. Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models. In _Proceedings of COLING_ , pp. 16867–16878, 2024. URL https: //aclanthology.org/2024.lrec-main.1466.pdf. 

12 

Arxiv Preprint 

- Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In _Proceedings of ACL_ , pp. 1601–1611, 2017. URL https://aclanthology.org/P17-1147. 

- Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to learn long-tail knowledge. In _Proceedings of ICML_ , pp. 15696–15707, 2023. URL https://proceedings.mlr.press/v202/kandpal23a/kandpal23a.pdf. 

- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In _Proceedings of EMNLP_ , pp. 6769–6781, 2020. URL https://aclanthology.org/2020. emnlp-main.550. 

- Levente Kocsis and Csaba Szepesv´ari. Bandit based monte-carlo planning. In _Proceedings of ECML_ , pp. 282–293, 2006. URL https://citeseerx.ist.psu.edu/document?repid=rep1&type= pdf&doi=6661e57237e4e8739b7a4946c4d3d4875376c068. 

- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. _Transactions of the Association for Computational Linguistics_ , pp. 452–466, 2019. URL https://aclanthology. org/Q19-1026. 

- Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Stepwise preference optimization for long-chain reasoning of llms. _ArXiv preprint_ , 2024. URL https: //arxiv.org/abs/2406.18629. 

- Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In _Proceedings of NeurIPS_ , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html. 

- Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. A survey on retrieval-augmented text generation. _ArXiv preprint_ , 2022. URL https://arxiv.org/abs/2202.01110. 

- Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. Ra-dit: Retrieval-augmented dual instruction tuning. _ArXiv preprint_ , 2023. URL https://arxiv.org/abs/2310.01352. 

- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In _Proceedings of ACL_ , pp. 158–167, 2017. URL https://aclanthology.org/P17-1015. 

- Hongwei Liu, Zilong Zheng, Yuxuan Qiao, Haodong Duan, Zhiwei Fei, Fengzhe Zhou, Wenwei Zhang, Songyang Zhang, Dahua Lin, and Kai Chen. Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark. _arXiv e-prints_ , pp. arXiv–2405, 2024a. URL https://arxiv.org/pdf/2405.12209. 

- Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_ , pp. 157–173, 2024b. URL https://arxiv.org/ pdf/2307.03172. 

- Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. Entity-based knowledge conflicts in question answering. _ArXiv preprint_ , 2021. URL https: //arxiv.org/abs/2109.05052. 

- Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. Sail: Search-augmented instruction learning. _ArXiv preprint_ , 2023a. URL https://arxiv.org/abs/2305.15225. 

13 

Arxiv Preprint 

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. _ArXiv preprint_ , 2023b. URL https://arxiv.org/abs/2308.08747. 

- Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Retrieval augmented code generation and summarization. _ArXiv preprint_ , 2021. URL https: //arxiv.org/abs/2108.11601. 

- Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al. Check your facts and try again: Improving large language models with external knowledge and automated feedback. _ArXiv preprint_ , 2023. URL https: //arxiv.org/abs/2302.12813. 

- Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents. _ArXiv preprint_ , 2024. URL https://arxiv.org/abs/2408.07199. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In _Proceedings of NeurIPS_ , 2024. URL https://arxiv.org/pdf/2305.18290. 

- Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. _Transactions of the Association for Computational Linguistics_ , pp. 1316–1331, 2023. URL https://arxiv.org/abs/2302. 00083. 

- Devendra Sachan, Mostofa Patwary, Mohammad Shoeybi, Neel Kant, Wei Ping, William L. Hamilton, and Bryan Catanzaro. End-to-end training of neural retrievers for open-domain question answering. In _Proceedings of ACL_ , pp. 6648–6662, 2021. URL https://aclanthology.org/ 2021.acl-long.519. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. _ArXiv preprint_ , 2017. URL https://arxiv.org/abs/1707.06347. 

- Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. _ArXiv preprint_ , 2023. URL https://arxiv.org/abs/2301.12652. 

- Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. In _Proceedings of EMNLP Findings_ , pp. 3784–3803, 2021. URL https://aclanthology.org/2021.findings-emnlp.320.pdf. 

- Devendra Singh, Siva Reddy, Will Hamilton, Chris Dyer, and Dani Yogatama. End-to-end training of multi-document reader and retriever for open-domain question answering. In _Proceedings of NeurIPS_ , pp. 25968–25981, 2021. URL https://arxiv.org/pdf/2106.05346. 

- Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging bigbench tasks and whether chain-of-thought can solve them. _ArXiv preprint_ , 2022. URL https: //arxiv.org/abs/2210.09261. 

- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In _Proceedings of NAACL-HLT_ , pp. 4149–4158, 2019. URL https://aclanthology.org/N19-1421. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _ArXiv preprint_ , 2023. URL https://arxiv.org/abs/ 2302.13971. 

- Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In _Proceedings of ACL_ , pp. 10014–10037, 2023. URL https://aclanthology.org/2023.acl-long. 557.pdf. 

14 

Arxiv Preprint 

Kevin Wu, Eric Wu, and James Zou. Clasheval: Quantifying the tug-of-war between an llm’s internal prior and external evidence. _ArXiv preprint_ , 2024. URL https://arxiv.org/pdf/2404.10198. 

- Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to advance general chinese embedding, 2023. URL https://arxiv.org/pdf/2309.07597. 

- Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In _Proceedings of ICML_ , 2024. URL https://arxiv.org/pdf/2305.13300. 

- Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In _Proceedings of ICLR_ , 2021. URL https://arxiv.org/pdf/2007.00808. 

- Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and selective augmentation. _ArXiv preprint_ , 2023a. URL https://arxiv.org/ abs/2310.04408. 

- Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-seng Chua. Search-in-the-chain: Towards the accurate, credible and traceable content generation for complex knowledge-intensive tasks. _ArXiv preprint_ , 2023b. URL https://arxiv.org/abs/2304.14732. 

- Shicheng Xu, Liang Pang, Mo Yu, Fandong Meng, Huawei Shen, Xueqi Cheng, and Jie Zhou. Unsupervised information refinement training of large language models for retrieval-augmented generation. _ArXiv preprint_ , 2024a. URL https://arxiv.org/abs/2402.18150. 

- Zhipeng Xu, Zhenghao Liu, Yibin Liu, Chenyan Xiong, Yukun Yan, Shuo Wang, Shi Yu, Zhiyuan Liu, and Ge Yu. Activerag: Revealing the treasures of knowledge via active learning. _ArXiv preprint_ , 2024b. URL https://arxiv.org/abs/2402.13547. 

Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models. _ArXiv preprint_ , 2024c. URL https://arxiv.org/abs/2401.11817. 

- Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation. _ArXiv preprint_ , 2024. URL https://arxiv.org/abs/2401.15884. 

- Yi Yang, Wen-tau Yih, and Christopher Meek. WikiQA: A challenge dataset for open-domain question answering. In _Proceedings of EMNLP_ , pp. 2013–2018, 2015. URL https:// aclanthology.org/D15-1237. 

- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In _Proceedings of EMNLP_ , pp. 2369–2380, 2018. URL https://aclanthology. org/D18-1259. 

- Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin Ma, Hongwei Wang, and Dong Yu. Chainof-note: Enhancing robustness in retrieval-augmented language models. _ArXiv preprint_ , 2023a. URL https://arxiv.org/abs/2311.09210. 

- Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. Augmentation-adapted retriever improves generalization of language models as generic plug-in. _ArXiv preprint_ , 2023b. URL https: //arxiv.org/abs/2305.17331. 

15 

Arxiv Preprint 

Table 5: Data Statistics. 

|||Table 5: Data Statistics.|||
|---|---|---|---|---|
|**Split**|**Task**|**Dataset**|**Metric**|**Total**|
|Training|Open-Domain QA|Commonsense QA (2019)<br>Math QA (2019)<br>Web Questions (2013)<br>Wiki QA (2015)<br>Yahoo! Answers QA<br>MARCOQA(2016)|Accuracy<br>Accuracy<br>Accuracy<br>Rouge-L<br>Rouge-L<br>Rouge-L|4,200<br>4,200<br>3,778<br>1,040<br>4,200<br>4,200|
||Reasoning|Algebra QA with Rationales (2017)<br>Explanations for CommonsenseQ (2021)<br>Grade School Math 8K (2021)<br>StrategyQA(2021)|Accuracy<br>Accuracy<br>Accuracy<br>Accuracy|2,727<br>4,200<br>4,200<br>2,060|
|Evaluation|Open-domain QA|Natural Questions (2019)<br>TriviaQA (2017)<br>MARCOQA(2016)|Accuracy<br>Accuracy<br>Rouge-L|2,837<br>5,359<br>3,000|
||Multi-Hop QA|HotpotQA(2018)|Accuracy|5,600|
||Slot Filling|T-REx(2018)|Accuracy|5,000|
||Dialogue|Wizard of Wikipedia(2019)|F1|3,000|



## A APPENDIX 

## A.1 ADDITIONAL EXPERIMENTAL DETAILS 

In this subsection, we first outline the process of constructing the training data. Then we show the prompt templates used in our experiments. In experiments, we keep the same training data, hyperparameters, and epochs to implement both RA-DIT and RAG-DDR for a fair comparison. 

**Data Preprocessing for DDR.** The quantity of our training and evaluation data, along with the corresponding evaluation metrics, are presented in Table 5. Then we describe the details of data preprocessing during training both knowledge refinement ( _V_ KR) and generation ( _V_ Gen) modules using our DDR method. 

Optimizing the knowledge refinement module is a rollout process, which individually feeds the top100 retrieved documents to the generation module ( _V_ Gen) with query to calculate the reward. Then, we apply the evaluation metrics shown in Table 5 to calculate the reward scores, identifying the documents that result in the highest scores as positive documents and those with the lowest scores as negative ones. Finally, we generate the triplet data _{_ query _,_ positive/negative documents _,_ “YES”/“NO” _}_ . To construct the dataset for training the generation module, we concatenate the refined documents with the query and then fed them into the generation module for sampling responses. We apply five different temperature settings (0.5, 0.6, 0.7, 0.8, 0.9) to sample responses and conduct the five-round sampling for each temperature. Afterward, we compute the reward score for each output using evaluation metrics to identify the positive and positive responses for each query. This process yields the triplet data _{_ query _,_ positive response _,_ negative response _}_ . 

**Prompt Templates.** For RA-DIT, we use the same instruction tuning template as Lin et al. (2023) and leverage top-5 retrieved documents as augmented knowledge during training. Then we describe the prompt templates used in RAG-DDR. As shown in Figure 4, we refer to the prompt designs of RA-DIT (Lin et al., 2023) and Self-RAG (Asai et al., 2023) to conduct tailored task prompts for different LLMs, helping LLMs generate better responses. In addition, we design separate prompts for LLMs w/ RAG and LLMs w/o RAG: _{_ Background: _{Documents}\_ n _{Instruction}}_ and _{Instruction}_ , where _Documents_ indicates the external documents provided for LLMs and _Instruction_ represents the task instructions. As illustrated in Figure 5, we design the prompts for the knowledge refinement tasks by referring to LangChain[1] , enabling LLMs to correctly generate “YES” or “NO” to retain or discard retrieved documents. 

> 1https://github.com/langchain-ai/langchain 

16 

Arxiv Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0017-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0017-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0017-03.png)


**----- Start of picture text -----**<br>
(a) MiniCPM-2.4B. (b) Llama3-8B.<br>**----- End of picture text -----**<br>


Figure 4: Prompts of Different Training and Evaluating Tasks. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0017-05.png)


Figure 5: Prompts Used for Knowledge Refinement. 

## A.2 ADDITIONAL ABLATION STUDIES ON RAG-DDR 

As shown in Table 6, we conduct additional ablation studies to explore the effectiveness of different training strategies: Independent Tuning, RAG-DDR ( _V_ KR First) and RAG-DDR ( _V_ Gen First). Independent Tuning indicates that the knowledge refinement module _V_ KR and generation module _V_ Gen are trained independently. RAG-DDR ( _V_ KR First) and RAG-DDR ( _V_ Gen First) are cascaded optimization models, indicating that we first train _V_ KR or _V_ Gen and subsequently optimize the other module by initializing the RAG model with the already optimized module. 

17 

Arxiv Preprint 

Table 6: Additional Ablation Study Results on RAG-DDR. 

|**Method**|**Open-DomainQA**|**Multi-Hop QA**|**Slot Filling**|**Dialogue**|
|---|---|---|---|---|
||**NQ**<br>**TriviaQA**<br>**MARCOQA**|**HotpotQA**|**T-REx**|**WoW**|
|_MiniCPM-2.4B_|||||
|Independent Tuning<br>RAG-DDR (_V_KR First)<br>RAG-DDR(_V_Gen First)|46.6<br>82.0<br>28.0<br>**47.4**<br>**83.2**<br>27.2<br>47.0<br>82.7<br>**28.1**|32.3<br>**33.3**<br>32.5|31.7<br>**35.0**<br>32.6|17.3<br>**17.3**<br>17.2|
|_Llama3-8B_|||||
|Independent Tuning<br>RAG-DDR (_V_KR First)<br>RAG-DDR(_V_Gen First)|49.9<br>88.3<br>25.1<br>50.4<br>**88.6**<br>**25.6**<br>**50.7**<br>88.2<br>25.1|37.4<br>36.9<br>**37.3**|37.0<br>36.7<br>**37.3**|14.9<br>14.9<br>**14.9**|




![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0018-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0018-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0018-05.png)


**----- Start of picture text -----**<br>
(a) The Accuracy of Knowledge Refinement. (b) The Accuracy of Generation Using Different Re-<br>fined Document Sets.<br>**----- End of picture text -----**<br>


Figure 6: Effectiveness of the Knowledge Refinement Module ( _V_ KR) Optimized Using Different Methods, including zero-shot (Vanilla RAG), SFT (RA-DIT), and DDR (RAG-DDR). 

Compared to Independent Tuning, the effectiveness of RAG-DDR is enhanced on all tasks. It shows that Independent Tuning results in misalignment of data preferences between the optimized modules, thereby impacting the overall performance of the RAG system. In contrast, the performance of RAGDDR ( _V_ KR First) and RAG-DDR ( _V_ Gen First) is indistinguishable across different datasets and _V_ Gen, demonstrating that the DDR method is robust to the training orders. 

## A.3 CHARACTERISTICS OF THE KNOWLEDGE REFINEMENT MODULE OF RAG-DDR 

As shown in Figure 6, we explore the characteristics of the knowledge refinement module ( _V_ KR) optimized using different methods, including Vanilla RAG, RA-DIT and RAG-DDR. 

For RA-DIT, we collect 30k pieces of data from the MARCO QA dataset to tune _V_ KR where each query has labeled positive and negative documents. We sample one positive document and one negative document for each query to construct the SFT dataset containing 60k pieces of data and fine-tune _V_ KR. The number of the SFT dataset is consistent with the amount of the RAG-DDR training dataset. The SFT dataset consists of triples in the form of _{_ query _,_ positive/negative documents _,_ “YES”/“NO” _}_ . 

As shown in Figure 6(a), we calculate the accuracy of top-5 documents by Vanilla RAG, RA-DIT, and RAG-DDR. The retrieval accuracy evaluates whether or not the retained top-5 documents contain the ground truth. If _V_ KR discards all the documents retrieved, the accuracy is 0. RAG-DDR outperforms Vanilla RAG and RA-DIT on all tasks. It indicates that the DDR method can make _V_ KR accurately retain documents that contain the necessary knowledge to answer the query. As shown in Figure 6(b), we use the refined documents by different knowledge refinement module ( _V_ KR) to augment the DDR trained generation module ( _V_ Gen). DDR-RAG also outperforms other models, indicating that DDR can help better align data preferences between _V_ Gen and _V_ KR. 

18 

Arxiv Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0019-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0019-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards_images/iclr_2025_RAG-DDR_Optimizing_Retrieval-Augmented_Generation_Using_Differentiable_Data_Rewards.pdf-0019-03.png)


**----- Start of picture text -----**<br>
(a) Llama3-8B. (b) MiniCPM-2.4B.<br>**----- End of picture text -----**<br>


Figure 7: The Ability of the Generation Module ( _V_ Gen) in RAG-DDR. 

## A.4 THE GENERAL LLM ABILITY OF DDR OPTIMIZED GENERATION MODULE 

In this experiment, we further explore the characteristics of the generation module ( _V_ Gen) optimized using different methods, zero-shot (LLM w/o RAG) and DDR (RAG-DDR). 

As shown in Figure 7, we compare the general ability of the generation module on several aspects: Mathematical (Liu et al., 2024a), Disciplinary Knowledge (Hendrycks et al., 2020), World Knowledge (Kwiatkowski et al., 2019), Logical Reasoning (Suzgun et al., 2022), and Common Sense Reasoning (Clark et al., 2018). These tasks are commonly used as benchmarks to assess the model’s inherent capabilities (Touvron et al., 2023; Hu et al., 2024). 

As shown in Figure 7(a), DDR enables Llama3-8B to maintain its strong language understanding and knowledge reasoning capabilities. The performance of MiniCPM-2.4B is shown in Figure 7(b). The evaluation results show that DDR significantly enhances the performance of the smaller parameter models, MiniCPM-2.4B, particularly in mathematical and common sense reasoning tasks. It illustrates that DDR not only preserves the original capabilities of LLMs but also offers some potential for enhancing their performance. 

19 

