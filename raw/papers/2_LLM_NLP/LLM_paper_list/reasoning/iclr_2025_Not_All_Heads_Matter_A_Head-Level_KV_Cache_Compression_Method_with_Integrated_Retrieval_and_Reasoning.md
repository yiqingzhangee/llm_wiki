Under review 

# NOT ALL HEADS MATTER: A HEAD-LEVEL KV CACHE COMPRESSION METHOD WITH INTEGRATED RETRIEVAL AND REASONING 

**Yu Fu,**[1] **Zefan Cai,**[2] **Abedelkadir Asi,**[3] **Wayne Xiong,**[3] **Yue Dong,**[1] **Wen Xiao**[3] 1University of California, Riverside, 2University of Wisconsin-Madison, 3Microsoft yfu093@ucr.edu, wxiao@microsoft.com 

## ABSTRACT 

Key-Value (KV) caching is a common technique to enhance the computational efficiency of Large Language Models (LLMs), but its memory overhead grows rapidly with input length. Prior work has shown that not all tokens are equally important for text generation, proposing layer-level KV cache compression to selectively retain key information. Recognizing the distinct roles of attention heads in generation, we propose HeadKV, a head-level KV cache compression method, and HeadKV-R2, which leverages a novel contextual reasoning ability estimation for compression. Our approach operates at the level of individual heads, estimating their importance for contextual QA tasks that require both retrieval and reasoning capabilities. Extensive experiments across diverse benchmarks (LongBench, LooGLE), model architectures (e.g., Llama-3-8B-Instruct, Mistral-7B-Instruct), and long-context abilities tests demonstrate that our head-level KV cache compression significantly outperforms strong baselines, particularly in low-resource settings (KV size = 64 & 128). Notably, our method retains just 1.5% of the KV cache while achieving 97% of the performance of the full KV cache on the contextual question answering benchmark.[1] 

## 1 INTRODUCTION 

Modern Large Language Models (LLMs) increasingly support extremely long inputs: GPT4 (Achiam et al., 2023), Llama-3 (Dubey et al., 2024), and Qwen-2 (Yang et al., 2024) handle up to 128K tokens, while Claude (Anthropic, 2024) supports up to 1 million tokens. These extended capacities improve performance on tasks like dialogue generation (Li et al., 2024a; Yi et al., 2024), question answering (Ho et al., 2020; Xu et al., 2023), and summarization (Xiao & Carenini, 2019; Koh et al., 2022). As input lengths increase, memory usage and latency grow significantly due to self-attention in transformers. To improve inference speed and efficiency, most LLM inference consists of two phases: prefilling for input processing and decoding for token generation, with key and value states from attention cached for reuse (KV cache). However, as input length increases, KV cache memory grows rapidly, posing significant challenges for storage and efficiency. 

To address this, KV cache compression methods (Xiao et al., 2024; Li et al., 2024d; Cai et al., 2024; Feng et al., 2024) have been proposed, typically using token eviction to optimize retention per layer or head during prefilling, reducing memory without impacting performance. However, none have explored varying KV cache size across individual heads. Inspired by prior observations (Voita et al., 2019; Wu et al., 2024; Zheng et al., 2024) that attention heads vary in importance for generation, we propose HeadKV, a **head-level KV cache compression method** that allocates KV cache budgets based on head importance distribution using a novel retrieval and reasoning importance estimation. Specifically, heads deemed more important are allocated larger KV cache budgets, while less important ones receive smaller allocations, optimizing memory usage without sacrificing performance. In addition to allocating KV cache across attention heads rather than layers, a key aspect of our approach is distributing cache budgets based on head importance measures. Prior work (Wu et al., 2024) proposed method to identify retrieval heads, using importance estimation to assess each head’s 

> 1https://github.com/FYYFU/HeadKV/ 

1 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
For Current Head Answer:  John's  ...<br>Question:  What is the favorite thing of Importance Score:  0.13/N +   ... Top-N Attention<br>the younger one between John and Mary? SUM<br>Rest Attention<br>Importance Score Estimation 0.08<br>0.05<br>told  us  so  much  he ... John is 12 years old. Mary is 13 years old. Mary's favorite thing is to... John's favorite thing is ... we did our best to ...<br>Reason Wrong  Copy Correct Copy<br>Context (N tokens) Context<br>Needle Sentences<br>Total Heads: Num of Layers * Num of Heads<br>Head-Level Allocation Local Windows Basic Budget Dynamic Budget<br>Head3<br>Head4<br>Head5<br>Prefilling Phrase After  Compression<br>Head3<br>Head4<br>Layers Head5<br>Importance Score Distribution<br>Heads<br>**----- End of picture text -----**<br>


Figure 1: Our proposed head-level KV cache compression method consists of two steps: (1) HeadLevel Importance Score Estimation (upper part): important heads that contribute to the **contextual reasoning ability** are identified using Needle-in-a-Haystack tests. (2) Head-Level KV Cache Allocation (lower part): KV cache budgets for each head during the prefilling phase are allocated based on the importance score distribution identified in the first step. 

role in retrieving relevant context. We integrate their measure with our head-level KV cache compression as a baseline, observing improved performance over layer-level cache compression. 

However, we argue that allocating larger KV cache budgets solely to retrieval heads is insufficient for tasks like contextual Question Answering (QA), which requires both retrieval and reasoning to handle long input contexts effectively. To address this, we propose **an importance-score estimation method that jointly evaluates each head’s retrieval and reasoning capabilities for KV cache allocation** . Specifically, as illustrated in the Importance Score Estimation section of Figure 1, we construct questions that require both the retrieval and reasoning abilities of the LLM. For instance, in the provided example, the model must first identify “who is younger between John and Mary” (John) by referencing the context of their ages, and then retrieve John’s favorite thing. We then estimate the importance score of each head based on the attention scores generated by the model while answering the question. Using the estimated importance scores, we allocate the KV cache budget for each individual head, meaning that heads demonstrating greater importance in retrieval and reasoning retain a larger portion of the KV cache, as shown in the Head-Level Allocation section of Figure 1. Within each head, we then retain only the most relevant KV cache entries, following the strategy proposed in Li et al. (2024d). 

We conduct experiments on various benchmarks requiring both retrieval and reasoning abilities, including QA tasks from LongBench (Bai et al., 2024) and LooGLE (Li et al., 2024b), using backbone models such as Llama-3-8B-Instruct (Dubey et al., 2024) and Mistral-7B-Instruct (Jiang et al., 2023). Our results demonstrate that the head-level KV cache compression method outperforms previous approaches on nearly all tasks. Furthermore, the allocation strategy based on our estimated importance scores—reflecting both retrieval and reasoning abilities—outperforms allocation strategies based on retrieval ability alone. In challenging scenarios like needle-in-a-Haystack and reasoning-in-a-Haystack tests, our methods effectively preserve the model’s retrieval and reasoning capabilities. Finally, experiments on memory and latency reveal that our approach significantly reduces memory usage and decoding latency, comparing with the original full KV. 

2 

Under review 

## 2 RELATED WORK 

## 2.1 ATTENTION HEADS 

Multi-Head Attention, a fundamental component of Transformer architectures (Vaswani et al., 2017), has been extensively analyzed to understand the roles of individual attention heads (Voita et al., 2019; Olsson et al., 2022; Jin et al., 2024; Wu et al., 2024; Zheng et al., 2024), often with goals such as pruning redundant heads(Shim et al., 2021; Kwon et al., 2022) or enhancing interpretability(Olsson et al., 2022; Jin et al., 2024; Zheng et al., 2024). For example, Voita et al. (2019) observed that only a small subset of heads plays a key role in machine translation, typically managing positional information, syntactic parsing, or focusing on rare words. Similarly, Olsson et al. (2022) identified ’induction heads’ that implement a copy-and-paste mechanism, enabling models to replicate previously encountered sequences. Additionally, Zheng et al. (2024) provided a thorough overview of recent efforts to characterize the diverse functions of attention heads, categorizing them into four types: Knowledge Recalling, In-Context Identification, Latent Reasoning, and Expression Preparation. A closely related study by Wu et al. (2024) discovered “retrieval heads” that play a crucial role in knowledge acquisition, using Needle-in-a-Haystack tests. These insights into head-level functionality serve as the foundation for our head-level KV cache compression methods, designed to jointly preserve both retrieval and reasoning capabilities during the compression process. 

## 2.2 KV CACHE COMPRESSION 

Improving computational efficiency for LLMs, particularly for extremely long inputs, has attracted considerable research interest (Shazeer, 2019; Chen et al., 2023; Dao, 2024; Zhou et al., 2024; Ye et al., 2024), including caching key and value vectors in Multi-Head Attention to improve generation efficiency (Xiao et al., 2024; Zhang et al., 2023; Li et al., 2024d; Cai et al., 2024). One challenge with full KV cache approaches is that the memory usage increases linearly with input length, leading to significant memory management challenges. Various KV cache compression techniques have been proposed to reduce memory usage and improve inference efficiency. For example, Xiao et al. (2024) addressed the ‘attention sink’ issue with StreamingLLM, retaining only the first _k_ tokens’ KV cache. Zhang et al. (2023) applied the Heavy Hitter Oracle strategy to select key cache entries, while Li et al. (2024d) used the attention scores of the last _α_ tokens to identify relevant entries. Cai et al. (2024) introduced PyramidKV, assigning smaller cache budgets to higher layers based on attention matrix patterns. 

While these methods have improved performance and efficiency, they largely depend on rigid layerlevel allocation strategies, which may not fully optimize KV cache allocation for downstream tasks. Feng et al. (2024) proposed dynamic head-level allocation using attention scores but still relied on layer-level budgeting. Similarly, Tang et al. (2024) employed Retrieval Heads distribution for headlevel allocation but retained a Full KV cache for key heads, limiting true head-level compression. In contrast, our approach allocates KV cache solely based on head-level importance scores, independent of layer constraints, resulting in more effective compression and outperforming baselines without increasing decoding latency or memory usage. 

## 3 METHOD 

This section presents our proposed head-level KV cache compression method, which consists of three key components: (1) identifying important heads and calculating head-level importance score distributions (3.1), (2) using these distributions to efficiently allocate KV cache budgets across heads (3.2), and (3) determining which Key and Value vectors to retain within each head (3.3). 

## 3.1 HEAD-LEVEL IMPORTANCE SCORE ESTIMATION 

Accurate budget allocation at the head level requires identifying which heads are most and least important for the given task. By leveraging this importance distribution, we can assign larger KV cache budgets to more critical heads and smaller budgets to less significant ones. To achieve this, we propose a novel importance-score estimation method, inspired by Wu et al. (2024), which allows us to effectively estimate the importance of each attention head for optimal KV cache allocation. 

3 

Under review 

|**Retrieval Example**|**Reasoning-Retrieval Example**|
|---|---|
|**Needle**:<br>The best thing to do in Beijing is to<br>take a walk in Chaoyang Park and have a cup of<br>Espresso in the evening.<br>**_q:_**What is the best thing to do in Beijing?|**_q_**: What is the favorite thing of the younger one<br>between John and Mary?<br>**Needle**:<br>John is 12 years old. Mary is 13 years old.<br>Mary's favorite thing is to take a walk in Chaoyang<br>Park and have a cup of Espresso in the evening.<br>John's  favorite thing is to play basketball at the local<br>gym and enjoy a smoothie afterward.|



Figure 2: Comparison of examples for head identification: Needle-in-a-Haystack test example from Wu et al. (2024) for identifying Retrieval Heads distribution (left), and our proposed Needle-in-aHaystack test example for identifying Retrieval-Reasoning Heads distribution (right). 

**Retrieval Heads** Wu et al. (2024) use the Needle-in-a-Haystack test[2] and custom retrieval examples, as shown in Figure 2, to estimate the importance score for each head. In these examples, a question _**q**_ that cannot be answered using the model’s parametric knowledge is paired with an answer _**k**_ (the “Needle”) inserted into a haystack _**c**_ at different positions _pi_ . The model is required to retrieve the exact answer _**k**_ from the combined input ( _**k** ,_ _**c**_ ). 

During each decoding step _t_ , a head _h_ earns a fraction of the importance score if (1) the token with the highest attention score arg max( _**a** h_ ) matches the generated token, and (2) the token is part of the inserted answer _**k**_ . The final importance score for each head _h_ is calculated accordingly: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0004-05.png)


where _N_ is the length of the inserted answer _**k**_ , and _**a** h_ is the attention score on the combined input from head _h_ . By using various settings of the inserted position _pi_ , they obtain the head-level Retrieval Heads distribution. 

Directly using this distribution poses two issues: (1) it focuses on the retrieval-and-paste mechanism, lacking consideration for the contextual and reasoning skills needed for complex questions; and (2) the distribution is too sparse for effective head-level KV cache allocation, with nearly 70% of heads receiving an importance score of zero due to the strict exact-match requirement (Wu et al., 2024). 

**Retrieval-Reasoning (R2) Heads** To address these issues, we propose a new importance score estimation method that accounts for both the retrieval and reasoning abilities of the heads, enabling a more accurate assessment of their significance. 

First, we construct retrieval-reasoning examples by adding explicit contextual reasoning steps to the retrieval examples from Wu et al. (2024), as shown in the Retrieval-Reasoning Example part of Figure 2. We further modify the inserted needle into three parts: _**k**_ = ( _**r** ,_ _**c**_[1] _,_ _**c**_[2] ), where _**r**_ is the reasoning step, and _**c**_[1] and _**c**_[2] are different answers to the refined question _**q**_ . The model must reason with _**r**_ to retrieve and generate the correct answer _**c**_[2] , avoiding the wrong answer _**c**_[1] . 

Secondly, we refine the estimation method by focusing on the entire correct answer _**c**_[2] (Correct Copy in Figure 1), since all tokens are relevant to the question _**q**_ . This approach aligns with Contextual QA, which requires both retrieval and reasoning abilities. By considering the full correct answer, the importance score for each head _h_ no longer depends solely on the token with the highest attention score. The importance score for head _h_ is calculated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0004-11.png)


> 2https://github.com/gkamradt/LLMTest ~~N~~ eedleInAHaystack 

4 

Under review 

where _ai ∈_ _**a** h_ represents the _i_ -th highest attention score from head _h_ , and top- _i_ ( _**a** h_ ) is the token with the _i_ -th highest score. We compute the importance score by considering the entire correct answer and increasing the number of tokens evaluated per head (Importance Score Estimation in Figure 1). Intuitively, heads with higher attention on the correct answer _**k**_ should receive higher importance scores. We further refine the score using attention weights, yielding a more accurate distribution, as shown in Eq. 2. 

## 3.2 HEAD-LEVEL KV CACHE ALLOCATION 

With the importance scores estimated for each head, we can identify the key heads and allocate the KV cache budget accordingly. In this section, we explain how to incorporate these distributions into head-level KV cache allocation. 

**Preliminary** In Multi-Head Attention, for each head _h_ in layer _l_ , the embedded input _**X**_ = _{x_ 1 _, x_ 2 _, . . . , xn} ∈_ R _[n][×][d]_ is mapped into different subspaces using the query _**W** Q[l]_[,][key] _**[W]**[ l] K_[,][and] value _**W** V[l][∈]_[R] _[d][×][d][h]_[matrices:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0005-05.png)


To optimize memory and enhance efficiency, KV cache compression methods (Xiao et al., 2024; Li et al., 2024d; Cai et al., 2024) are employed to discard unimportant KV cache entries while preserving performance. For each head _h_ , the compressed KV cache is reduced to _**K** h[l][∈]_[R] _[s][×][d][h]_[and] _**V** h[l][∈]_[R] _[s][×][d][h]_[, where] _[ s][ ≪][n]_[, resulting in a significant improvement to computational efficiency.] 

**Head-level Allocation** Previous works on KV cache compression during the prefill phase (Xiao et al., 2024; Li et al., 2024d; Cai et al., 2024; Feng et al., 2024) are limited to layer-level allocation, using either uniform or dynamic budgets per layer, but treating all heads within a layer equally. While Feng et al. (2024) incorporate head-level information, their approach still depends on layerlevel allocation as a prerequisite. 

Building on the head-level importance distributions, we propose a comprehensive KV cache allocation strategy. Each head _h_ is initially assigned a fixed KV cache size _b_ with an associated importance score _Sh_ . To allow dynamic allocation, we create a shared budget pool _B_ by extracting a portion of the budget from each head, leaving the remainder as the basic budget. This process is illustrated in the Head-Level Allocation section of Figure 1. The budget pool _B_ is then distributed among the heads in proportion to their importance scores _Sh_ . The final head-level KV cache allocation is as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0005-09.png)


where _b_ is the initial fix budget for each head, _β_ is a hyper-parameter to control the size of the dynamic budget pool, _L_ and _H_ is the numbers of layers and heads of current LLM respectively. 

The last _α_ instruct tokens are preserved before forming the dynamic budget pool _B_ to guide the selection process, as detailed in Section 3.3. The retained KV cache for each head includes: (1) the basic budget ( _b − β[b]_[), (2) the dynamic budget] _[ S][h][ ×][ B]_[, proportional to its importance score, and (3)] the last _α_ instruct tokens. 

## 3.3 KV CACHE SELECTION 

After determining the number of KV cache entries to retain using the above algorithm, we apply an attention-based selection strategy from prior works (Li et al., 2024d; Cai et al., 2024; Feng et al., 2024) to keep the most relevant entries. Specifically, the last _α_ instruction tokens (local windows) guide KV cache selection for each head. Attention scores from these local windows to the remaining tokens are aggregated through a pooling layer, with higher-scoring tokens considered more important and retained in the cache. 

5 

Under review 

## 4 EXPERIMENTS AND ANALYSIS 

This section outlines the experimental setup, including KV cache baselines and implementation details. We also conduct additional experiments that: (1) emphasize the importance of enhancing contextual reasoning in importance score estimation (4.3); (2) use the Needle-in-a-Haystack and Reasoning-in-a-Haystack tests to demonstrate how our head-level KV cache compression improves long-context retrieval and reasoning (4.4); and (3) provide a comprehensive comparison with previous methods, showing our approach delivers superior performance while maintaining computational efficiency (4.5). 

## 4.1 EXPERIMENT SETTINGS 

**Models and Datasets** We compare our head-level KV cache compression method against strong baselines using two open-source LLMs: Llama-3-8B-Instruct (Dubey et al., 2024) and Mistral-7BInstruct (Jiang et al., 2023). The evaluation is based on two benchmarks for long-context understanding: LongBench (Bai et al., 2024) and LooGLE (Li et al., 2024b). For LongBench, we use datasets from the Single-Doc QA and Multi-Doc QA categories to assess contextual reasoning. For LooGLE, we focus on the Long Dependency QA task, which includes four QA-related tasks. Dataset details are in Appendix Table 5. 

- **Baselines and Settings** We evaluate three strong KV cache compression methods as baselines, ensuring all retain the same number of KV cache entries for fair comparison: 1) **SnapKV** (Li et al., 2024d) uses the last _α_ tokens as local windows to guide KV cache selection. Attention scores from these windows to the remaining tokens are pooled to cluster information and guide the selection process. 

   - 2) **PyramidKV** (Cai et al., 2024) follows a pyramid attention pattern, allocating more KV cache to lower layers to retain key information, while reducing the budget for higher layers where information is already aggregated. 

   - 3) **Ada-KV** (Feng et al., 2024) dynamically allocates budgets to heads within each layer based on their concentration degrees, and can be combined with SnapKV or PyramidKV. AdaSnapKV is used as the baseline due to its superior performance over Ada-PyramidKV. 

Our proposed head-level KV cache compression method also requires a strategy to guide KV cache selection after allocating the budget. Therefore, we use the SnapKV method to guide the selection process for each head. We set the size of the local windows _α_ = 8 for both the baselines and our method. The hyper-parameter _β_ , which controls the size of the shared budget pool, was chosen from _{_ 1.005, 1.01, 1.1, 1.2, 1.5, 2, 5, 10 _}_ , and we report the best performance. 

## 4.2 MAIN RESULTS 

Table 1 lists the evaluation results for contextual tasks from the LongBench and LooGLE benchmarks. Our head-level KV cache compression method consistently outperforms strong baselines, especially with 64 and 128 KV cache configurations. In resource-constrained settings, precise KV cache allocation is crucial. Layer-level methods allocate a fixed cache size to all heads within a layer, making it difficult to retain essential information. Ada-SnapKV improves this by allowing dynamic allocation within layers, but still relies on fixed layer-level budgets. In contrast, our head-level strategy allocates dynamically across individual heads, retaining critical information by adjusting the budget based on each head’s importance. 

We perform head-level allocation using both the standard Retrieval Heads distribution (HeadKV-R) and our Retrieval-Reasoning Heads distribution (HeadKV-R2) for global KV cache allocation. This combination leads to superior performance across benchmarks. Notably, integrating the RetrievalReasoning Heads distribution significantly improves results over the standard Retrieval Heads distribution, highlighting the impact of our approach. Our Retrieval-Reasoning distribution even surpasses the Full-KV cache average (32.90), achieving 32.95 with a 1024 KV cache. Overall, both the head-level allocation strategy and Retrieval-Reasoning distribution are key to these performance gains. 

6 

Under review 

|Method|**Single-DocQA**<br>**Multi-DocQA**<br>**Avg.**<br>**NartvQA**<br>**Qasper**<br>**MF-en**<br>**HotpotQA**<br>**2WikiMQA**<br>**Musique**|**Long dependency QA**<br>**Avg.**<br>**Doc.QA**<br>**Info. Retrieval**<br>**Timeline**<br>**Computation**|
|---|---|---|
||Llama-3-8B-Instruct, KV Size = Fu|ll|
|FullKV|25.56<br>32.07<br>39.71<br>43.57<br>35.28<br>21.18<br>32.90|8.73<br>11.21<br>0.67<br>7.43<br>7.01|
||Llama-3-8B-Instruct, KV Size = 12|8|
|SnapKV<br>PyramidKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|22.11<br>15.79<br>31.01<br>41.12<br>29.20<br>19.35<br>26.43<br>22.01<br>17.05<br>31.52<br>39.27<br>28.99<br>18.34<br>26.20<br>22.99<br>19.95<br>34.22<br>42.97<br>30.82<br>20.15<br>28.52<br>**23.49**<br>25.39<br>38.15<br>42.45<br>32.84<br>19.95<br>30.38<br>21.80<br>**29.19**<br>**41.89**<br>**43.73**<br>**35.01**<br>**20.40**<br>**32.00**|8.36<br>9.46<br>**0.79**<br>6.56<br>6.29<br>8.89<br>9.63<br>0.61<br>6.72<br>6.46<br>9.07<br>10.3<br>0.54<br>6.59<br>6.63<br>8.87<br>10.35<br>0.78<br>**7.52**<br>6.88<br>**9.60**<br>**11.13**<br>0.67<br>7.22<br>**7.16**|
||Llama-3-8B-Instruct, KV Size = 10|24|
|SnapKV<br>PyramidKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|25.76<br>27.50<br>38.38<br>43.40<br>34.81<br>20.07<br>31.65<br>25.38<br>26.83<br>36.90<br>44.09<br>34.24<br>21.49<br>31.49<br>**25.79**<br>29.24<br>38.74<br>43.93<br>36.34<br>19.79<br>32.31<br>24.85<br>**30.94**<br>**39.82**<br>43.52<br>**36.58**<br>20.37<br>32.68<br>24.66<br>30.82<br>39.56<br>**43.97**<br>36.47<br>**22.24**<br>**32.95**|**9.61**<br>11.34<br>0.53<br>7.22<br>7.18<br>8.98<br>11.41<br>0.53<br>6.96<br>6.97<br>8.65<br>11.41<br>0.53<br>7.71<br>7.08<br>9.20<br>**11.67**<br>**0.55**<br>7.71<br>**7.28**<br>9.02<br>11.51<br>0.47<br>**7.85**<br>7.21|
||Mistral-7B-Instruct, KV Size = Ful|l|
|FullKV|26.63<br>32.99<br>49.34<br>42.77<br>27.35<br>18.78<br>32.98|12.17<br>15.52<br>0.49<br>10.03<br>9.55|
||Mistral-7B-Instruct, KV Size = 12|8|
|SnapKV<br>PyramidKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|21.47<br>21.95<br>45.24<br>33.88<br>21.83<br>15.53<br>26.65<br>21.76<br>21.98<br>43.72<br>32.76<br>22.73<br>15.59<br>26.42<br>21.57<br>24.59<br>46.70<br>35.74<br>25.57<br>14.37<br>28.09<br>23.97<br>**29.60**<br>48.40<br>39.66<br>26.31<br>**18.13**<br>31.01<br>**25.04**<br>27.95<br>**48.48**<br>**41.28**<br>**27.65**<br>18.05<br>**31.41**|10.86<br>12.24<br>0.57<br>8.81<br>8.12<br>10.64<br>11.9<br>0.47<br>8.69<br>7.93<br>11.14<br>12.37<br>0.45<br>9.57<br>8.38<br>11.43<br>13.04<br>0.53<br>**10.26**<br>8.82<br>**11.44**<br>**13.08**<br>**0.63**<br>10.20<br>**8.84**|
||Mistral-7B-Instruct, KV Size = 102|4|
|SnapKV<br>PyramidKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|25.38<br>30.22<br>49.29<br>41.84<br>26.60<br>18.08<br>31.90<br>24.28<br>30.05<br>49.17<br>40.49<br>26.43<br>18.80<br>31.54<br>24.82<br>31.49<br>48.80<br>41.18<br>27.38<br>18.22<br>31.98<br>**25.87**<br>31.44<br>49.55<br>**41.95**<br>27.09<br>**19.88**<br>32.63<br>25.64<br>**32.54**<br>**50.49**<br>41.80<br>**27.88**<br>18.89<br>**32.87**|11.69<br>13.89<br>0.52<br>10.54<br>9.16<br>11.77<br>14.51<br>0.51<br>10.19<br>9.25<br>11.96<br>13.82<br>**0.53**<br>9.92<br>9.06<br>**12.21**<br>14.17<br>0.50<br>**10.58**<br>9.37<br>11.94<br>**14.93**<br>0.50<br>10.49<br>**9.47**|



Table 1: Performance comparison on the LongBench and LooGLE benchmarks for Llama-3-8BInstruct and Mistral-7B-Instruct. Our head-level KV cache compression method outperforms all baselines, especially in low-resource settings (KV size = 128). It even exceeds the FullKV result (32.90) on Llama-3-8B-Instruct (KV size = 1024, 32.95), highlighting the benefits of incorporating contextual reasoning for head selection. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0007-03.png)


**----- Start of picture text -----**<br>
Llama-3-8B-Instruct Mistral-7B-Instruct<br>32 32<br>30<br>30<br>28 FullKV<br>28<br>SnapKV<br>26 PyramidKV<br>26 Ada-SnapKV<br>HeadKV-R<br>24<br>24 HeadKV-R2<br>200 400 600 800 1000 200 400 600 800 1000<br>KV size KV size<br>Avg. Score<br>**----- End of picture text -----**<br>


Figure 3: Results for different KV cache sizes (64, 128, 256, 512, 1024), showing average accuracy across six datasets from the LongBench benchmark with an average input length of 8,683 tokens. Notably, a KV cache size of 64 retains just 0.7% of the total tokens. 

In addition, we present the results for various retained KV cache sizes (64, 128, 256, 512, 1024) in Figure 3, , with detailed results available in Appendix Table 4. 

## 4.3 RETRIEVAL-REASONING HEADS 

As detailed in Section 3.1, we propose to improve the standard Retrieval Heads distribution by incorporating retrieval-reasoning examples and refining importance score estimation to better capture 

7 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Retrieval Heads Enhanced-Retrieval Heads Retrieval-Reasoning Heads<br>0.05<br>0.014<br>0.008<br>0.04 0.012<br>0.006 0.010<br>0.03<br>0.008<br>0.02 0.004 0.006<br>0.004<br>0.01 0.002<br>0.002<br>0.00 0.000<br>Layers Layers Layers<br>0 0 0<br>2 2 2<br>4 4 4<br>6 6 6<br>8 8 8<br>10 10 10<br>12 12 12<br>14 14 14<br>16 16 16<br>Heads 18 18 18<br>20 20 20<br>22 22 22<br>24 24 24<br>26 26 26<br>28 28 28<br>30 30 30<br>0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30<br>**----- End of picture text -----**<br>


Figure 4: Head visualization for Llama-3-8B-Instruct results. The Retrieval Heads distribution is sparse to effectively differentiate between heads, while our Retrieval-Reasoning Heads has denser distribution for such differentiation. See Appendix Figure7 for Mistral-7B-Instruct results. 

|Method|**Single-DocQA**<br>**NartvQA**<br>**Qasper**|**Single-DocQA**<br>**NartvQA**<br>**Qasper**|**MF-en**|**HotpotQA**|**Multi-DocQA**<br>**2WikiMQA**|**Musique**|**Avg.**|
|---|---|---|---|---|---|---|---|
|||Llama-3-8B-Instruct, KV Size=128||||||
|HeadKV-R|**23.49**|25.39|38.15|42.45|32.84|19.95|30.38|
|HeadKV-ER|23.33|25.86|40.28|43.25|33.23|20.28|31.04|
|HeadKV-R2|21.80|**29.19**|**41.89**|**43.73**|**35.01**|**20.40**|**32.00**|
|||Mistral-7B-Instruct, KV Size=128||||||
|HeadKV-R|23.97|**29.60**|48.40|39.66|26.31|**18.13**|31.01|
|HeadKV-ER|23.23|28.70|48.10|41.39|27.31|17.39|31.02|
|HeadKV-R2|**25.04**|27.95|**48.48**|**41.28**|**27.65**|18.05|**31.41**|



Table 2: Ablation study results on the LongBench benchmarks. HeadKV-R leverages the standard Retrieval Heads distribution. HeadKV-ER uses the retrieval examples from Wu et al. (2024) but with our proposed importance score estimation method. HeadKV-R2 leverages both our proposed importance score estimation method and the retrieval-reasoning examples. 

contextual reasoning and identify relevant heads. We also conduct an ablation study to evaluate the impact of these modifications. 

Table 2 presents the ablation study results, while Figure 4 provides visualizations for each distribution. Alongside the standard Retrieval Heads and Retrieval-Reasoning Heads distributions, we introduce the Enhanced-Retrieval Heads distribution, using retrieval examples with our modified importance score estimation method. Comparing Retrieval Heads (HeadKV-R) and Enhanced-Retrieval Heads (HeadKV-ER) reveals that focusing on the entire needle, rather than specific tokens, improves performance. Figure 4 shows that the Retrieval Heads distribution is sparse, while the EnhancedRetrieval and Retrieval-Reasoning distributions are much denser. The strict constraints on the Retrieval Heads distribution result in most heads receiving a score of zero, leading to a worse results when incorporating Retrieval Heads distributions. 

While the Enhanced-Retrieval Heads distribution improves performance slightly, it remains rely on retrieval examples and lacks full consideration of contextual reasoning. In contrast, the RetrievalReasoning Heads distribution, reflecting both retrieval and reasoning abilities, consistently outperforms other methods, underscoring the value of incorporating retrieval-reasoning examples. 

## 4.4 LONG-CONTEXT RETRIEVAL AND REASONING 

We conduct the Needle-in-a-Haystack test to assess the long-context retrieval ability of different KV cache compression methods. As illustrated in Figure 5, we set the KV size to 128 for all methods and keep the other hyperparameters consistent with previous experiments. Results from the Llama-3-8B-Instruct demonstrate that our head-level KV cache compression method effectively retain important information compared with other strong baselines, verifying the superior long-context retrieval ability of our proposed method. However, these tests only retrieve the inserted needle from the haystack and paste it into the generation, lacking an assessment of contextual reasoning ability 

8 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Llama-3-8B-Instrcut<br>FullKV  Average score: 100.00 SnapKV  Average score: 87.40<br>PyramidKV  Average score: 97.30 Ada-SnapKV  Average score: 95.30<br>HeadKV-R  Average score: 98.10 HeadKV-R2  Average score: 98.20<br>Context Length Context Length<br>**----- End of picture text -----**<br>


Figure 5: Neele-in-a-Haystack test results on Llama-3-8B-Instruct with KV cache = 128. We build our head-level KV cache method based on SnapKV and our proposed method significantly outperform all strong baselines. Moreover, our Retrieval-Reasoning Heads distribution maintains and improves long context retrieval ability. Results on Mistral-7B-Instruct can be found in Appendix Figure 8, which are consistent with results on Llama-3-8B-Instruct. 

|Method|0k<br>1k<br>2k<br>4k<br>8k<br>Avg.|0k<br>1k<br>2k<br>4k<br>8k<br>16k<br>32k<br>Avg.|
|---|---|---|
|FullKV|63.00<br>59.40<br>55.80<br>56.60<br>50.40<br>57.04|61.40<br>55.20<br>52.40<br>40.80<br>40.40<br>35.00<br>31.80<br>45.29|
|Llama-3-8B-Instruct, KV Size=128||Mistral-7B-Instruct, KV Size=128|
|SnapKV<br>PyramidKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|60.80<br>57.00<br>54.80<br>52.60<br>45.60<br>54.16<br>61.20<br>58.00<br>52.80<br>52.60<br>47.60<br>54.44<br>61.80<br>59.20<br>53.80<br>53.60<br>46.60<br>55.00<br>61.60<br>57.00<br>52.60<br>55.00<br>49.60<br>55.16<br>**62.60**<br>**60.40**<br>**55.00**<br>**55.80**<br>**50.40**<br>**56.84**|56.60<br>50.60<br>46.60<br>34.40<br>35.40<br>33.20<br>29.00<br>40.83<br>58.40<br>50.60<br>46.80<br>36.00<br>36.20<br>31.20<br>28.40<br>41.09<br>58.00<br>51.20<br>46.20<br>35.40<br>36.40<br>31.60<br>28.80<br>41.09<br>58.40<br>54.00<br>50.80<br>**38.00**<br>**37.40**<br>31.80<br>**30.20**<br>42.94<br>**60.20**<br>**54.20**<br>**51.20**<br>37.20<br>**37.40**<br>**32.60**<br>28.80<br>**43.09**|



Table 3: Reasoning-in-a-Haystack test results with KV cache = 128. The final results are averaged across QA1-QA5 tasks for each length. Unlike the Needle-in-a-Haystack test, this test inserts multiple needles into the haystack, requiring the model to reason through them to extract the correct answer. 

in long-context scenarios. This long-context contextual reasoning ability is crucial for many tasks, such as question answering (QA), summarization (Kuratov et al., 2024; Li et al., 2024c). Therefore, we conduct the Reasoning-in-a-Haystack test to evaluate the long-context reasoning ability of each KV cache compression method across different scenarios. 

We follow the setup from Kuratov et al. (2024) to conduct the Reasoning-in-a-Haystack test. This test enhances the bAbI benchmark (Weston et al., 2015), designed for reasoning evaluation, by using text from the PG19 dataset (Rae et al., 2020) as the haystack. Reasoning needles from bAbI are inserted into the haystack, and the model must retrieve and reason through them to generate the correct answer. We use the dataset from Kuratov et al. (2024), averaging results across QA1-QA5 tasks for evaluation. Examples of tasks are shown in Figure 9. 

As shown in Table 3, our head-level KV cache compression method significantly outperforms strong baselines, demonstrating its superior long-context reasoning. By incorporating retrieval-reasoning examples, our method achieves even better accuracy, particularly with the Llama-3-8B-Instruct model. Notably, combining head-level KV cache allocation with the standard Retrieval Heads distribution also yields improved results over other baselines. This is due to two factors: first, as shown in Figure 4, there is overlap between Retrieval and Retrieval-Reasoning Heads, indicating heads may serve multiple roles. Second, since the bAbI benchmark contains the answer within the inserted needle (see Figure 9), emphasizing retrieval alone helps our method locate the needle. 

9 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Decoding Latency Peak Memory Usage<br>FullKV<br>300 SnapKV 60<br>PyramidKV<br>250 Ada-SnapKV 50<br>HeadKV<br>200<br>40<br>150<br>100 30<br>50<br>20<br>0<br>0 1000 2000 3000 4000 4k 8k 16k 32k 64k 128k 256k<br>Generation Length Context Length<br>Decoding times (s) Peak Memory (GB)<br>**----- End of picture text -----**<br>


Figure 6: The Decoding Latency and Peak Memory Usage results. Our proposed method maintain the comparable computational efficiency with other KV cache compression baselines. 

## 4.5 MEMORY & LATENCY 

In this section, we evaluate the computational efficiency of our head-level KV cache compression method using the Mistral-7B-Instruct model with a maximum sequence length of 32K, and FlashAttention (Dao, 2024) as the default setting. 

To assess the decoding latency of each method, we use 32K-length data from the Reasoning-in-aHaystack test as input and set various generation lengths (1, 512, 1024, 2048, 4096) for comparison. As shown in the Decoding Latency of Figure 6, our proposed method achieves the same decoding latency as other KV cache compression methods while maintaining performance closest to the Full KV cache. Notably, the decoding time includes both the pre-filling time and the decoding time. Therefore, we can conclude that the pre-filling time for our method and other baselines (such as PyramidKV, Ada-SnapKV) is almost negligible, as shown at the starting point (generation length = 1) of Decoding Latency in Figure 6. 

In addition to decoding latency, KV cache compression methods also aim to reduce memory usage during the decoding phase. Therefore, we provide the Peak Memory Usage results, as shown in the Peak Memory Usage of Figure 6. All results for Peak Memory Usage are averaged over three trials. Our proposed methods achieve performance comparable to other KV cache compression baselines, significantly reducing memory usage compared to the Full KV cache. 

## 5 CONCLUSION AND FUTURE WORK 

In this paper, we introduce HeadKV-R2, a head-level KV cache compression method with two key components. We propose a strategy for allocating KV cache budgets across attention heads based on their importance scores and develop a novel approach to estimate these scores by constructing examples that account for both retrieval and reasoning abilities. By allocating larger KV cache budgets to more important heads and smaller budgets to less important ones, our approach efficiently retain important KV cache than layer-level KV cache compression methods. We thoroughly evaluate HeadKV across multiple benchmarks, models, and long-context ability tests, with comprehensive results demonstrating that our method achieves superior performance while maintaining computational efficiency. 

For future work, further exploration of various types of attention heads could offer valuable insights, such as those involved in in-context learning (Olsson et al., 2022) and truthfulness (Li et al., 2023). For example, placing greater emphasis on truthfulness heads could help mitigate issues like hallucination, thereby improving the factual accuracy of model outputs. Additionally, it would be worthwhile to investigate the development of a general task-specific score estimation algorithm. One potential direction is to leverage gradients from specific tasks to allocate KV cache budgets more effectively, enabling tailored compression to enhance model performance across diverse applications. 

10 

Under review 

## REFERENCES 

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Anthropic. The claude 3 model family: Opus, sonnet, haiku, March 2024. URL https:// www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/ Model_Card_Claude_3.pdf. Accessed: 2024-07-09. 

- Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 3119–3137, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.172. URL https://aclanthology.org/2024.acl-long.172. 

- Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Wen Xiao. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling, 2024. URL https://arxiv.org/abs/2406.02069. 

- Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling, 2023. URL https://arxiv.org/abs/2302.01318. 

- Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In _International Conference on Learning Representations (ICLR)_ , 2024. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S. Kevin Zhou. Ada-kv: Optimizing kv cache eviction by adaptive budget allocation for efficient llm inference, 2024. URL https://arxiv. org/abs/2407.11550. 

- Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott, Nuria Bel, and Chengqing Zong (eds.), _Proceedings of the 28th International Conference on Computational Linguistics_ , pp. 6609–6625, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.580. URL https: //aclanthology.org/2020.coling-main.580. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. URL https: //arxiv.org/abs/2310.06825. 

- Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models, 2024. URL https://arxiv.org/ abs/2402.18154. 

- Huan Yee Koh, Jiaxin Ju, Ming Liu, and Shirui Pan. An empirical survey on long document summarization: Datasets, models, and metrics. _ACM Computing Surveys_ , 55(8):1–35, December 2022. ISSN 1557-7341. doi: 10.1145/3545176. URL http://dx.doi.org/10.1145/ 3545176. 

- Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom Sorokin, and Mikhail Burtsev. Babilong: Testing the limits of llms with long context reasoning-in-a-haystack, 2024. URL https://arxiv.org/abs/2406.10149. 

11 

Under review 

- Woosuk Kwon, Sehoon Kim, Michael W. Mahoney, Joseph Hassoun, Kurt Keutzer, and Amir Gholami. A fast post-training pruning framework for transformers. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum?id=0GRBKLBjJE. 

- Jia-Nan Li, Quan Tu, Cunli Mao, Zhengtao Yu, Ji-Rong Wen, and Rui Yan. Streamingdialogue: Prolonged dialogue learning via long context compression with minimal losses, 2024a. URL https://arxiv.org/abs/2403.08312. 

- Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context language models understand long contexts?, 2024b. URL https://arxiv.org/abs/2311.04939. 

- Kenneth Li, Oam Patel, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum? id=aLLuYpn83y. 

- Mo Li, Songyang Zhang, Yunxin Liu, and Kai Chen. Needlebench: Can llms do retrieval and reasoning in 1 million context window?, 2024c. URL https://arxiv.org/abs/2407. 11963. 

- Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation, 2024d. URL https://arxiv.org/abs/2404.14469. 

- Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022. URL https://arxiv.org/ abs/2209.11895. 

- Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In _International Conference on Learning Representations_ , 2020. URL https://openreview.net/forum?id= SylKikSYDH. 

- Noam Shazeer. Fast transformer decoding: One write-head is all you need, 2019. URL https: //arxiv.org/abs/1911.02150. 

- Kyuhong Shim, Iksoo Choi, Wonyong Sung, and Jungwook Choi. Layer-wise pruning of transformer attention heads for efficient language modeling, 2021. URL https://arxiv.org/ abs/2110.03252. 

- Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Shikuan Hong, Yiwu Yao, and Gongyi Wang. Razorattention: Efficient kv cache compression through retrieval heads, 2024. URL https: //arxiv.org/abs/2407.15891. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 

- Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Anna Korhonen, David Traum, and Llu´ıs M`arquez (eds.), _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pp. 5797–5808, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1580. URL https://aclanthology. org/P19-1580. 

12 

Under review 

Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merri¨enboer, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks, 2015. URL https://arxiv.org/abs/1502.05698. 

- Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality, 2024. URL https://arxiv.org/abs/2404. 15574. 

- Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=NG7sS51zVF. 

- Wen Xiao and Giuseppe Carenini. Extractive summarization of long documents by combining global and local context. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 3011–3021, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1298. URL https://aclanthology.org/D19-1298. 

- Fangyuan Xu, Yixiao Song, Mohit Iyyer, and Eunsol Choi. A critical evaluation of evaluations for long-form question answering, 2023. URL https://arxiv.org/abs/2305.18201. 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 

- Lu Ye, Ze Tao, Yong Huang, and Yang Li. Chunkattention: Efficient self-attention with prefix-aware kv cache and two-phase partition, 2024. URL https://arxiv.org/abs/2402.15220. 

- Zihao Yi, Jiarui Ouyang, Yuwen Liu, Tianhao Liao, Zhe Xu, and Ying Shen. A survey on recent advances in llm-based multi-turn dialogue systems, 2024. URL https://arxiv.org/abs/ 2402.18013. 

- Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavyhitter oracle for efficient generative inference of large language models. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/ forum?id=RkRrPp7GKO. 

- Zifan Zheng, Yezhaohui Wang, Yuxin Huang, Shichao Song, Mingchuan Yang, Bo Tang, Feiyu Xiong, and Zhiyu Li. Attention heads of large language models: A survey, 2024. URL https: //arxiv.org/abs/2409.03752. 

- Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, and Yu Wang. A survey on efficient inference for large language models, 2024. URL https: //arxiv.org/abs/2404.14294. 

13 

Under review 

## A DETAILE RESULTS 

In Table 4, we show the detailed results of Figure 3 in the main paper. 

|Method|Single-DocQA<br>Multi-DocQA<br>Avg.<br>NartvQA<br>Qasper<br>MF-en<br>HotpotQA<br>2WikiMQA<br>Musique|Longdependency QA<br>Avg.<br>Doc.QA<br>Info. Retrieval<br>Timeline<br>Computation|
|---|---|---|
||Llama-3-8B-Instruct, KV Size=F|ull|
|FKV|25.56<br>32.07<br>39.71<br>43.57<br>35.28<br>21.18<br>32.90|8.73<br>11.21<br>0.67<br>7.43<br>7.01|
||Llama-3-8B-Instruct, KV Size=6|4|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|20.51<br>12.80<br>31.69<br>37.02<br>25.91<br>17.02<br>24.16<br>21.17<br>13.66<br>29.34<br>34.86<br>23.46<br>15.88<br>23.06<br>22.26<br>17.30<br>33.37<br>39.82<br>27.86<br>17.85<br>26.41<br>22.67<br>23.54<br>37.51<br>37.45<br>29.76<br>19.01<br>28.32<br>**23.21**<br>**25.33**<br>**38.71**<br>**40.64**<br>**31.33**<br>**19.35**<br>**29.76**|8.84<br>9.43<br>**0.66**<br>6.18<br>6.28<br>8.27<br>9.31<br>0.63<br>6.86<br>6.27<br>9.08<br>9.86<br>0.55<br>6.82<br>6.58<br>8.80<br>10.51<br>0.58<br>6.68<br>6.64<br>**9.46**<br>**10.66**<br>0.61<br>**6.92**<br>**6.91**|
||Llama-3-8B-Instruct, KV Size=1|28|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|22.11<br>15.79<br>31.01<br>41.12<br>29.20<br>19.35<br>26.43<br>22.01<br>17.05<br>31.52<br>39.27<br>28.99<br>18.34<br>26.20<br>22.99<br>19.95<br>34.22<br>42.97<br>30.82<br>20.15<br>28.52<br>**23.49**<br>25.39<br>38.15<br>42.45<br>32.84<br>19.95<br>30.38<br>21.80<br>**29.19**<br>**41.89**<br>**43.73**<br>**35.01**<br>**20.40**<br>**32.00**|8.36<br>9.46<br>**0.79**<br>6.56<br>6.29<br>8.89<br>9.63<br>0.61<br>6.72<br>6.46<br>9.07<br>10.3<br>0.54<br>6.59<br>6.63<br>8.87<br>10.35<br>0.78<br>**7.52**<br>6.88<br>**9.60**<br>**11.13**<br>0.67<br>7.22<br>**7.16**|
||Llama-3-8B-Instruct, KV Size=2|56|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|23.38<br>20.18<br>37.65<br>42.80<br>33.23<br>20.01<br>29.54<br>23.94<br>20.27<br>36.27<br>42.51<br>31.44<br>19.99<br>29.07<br>24.20<br>24.63<br>37.95<br>43.64<br>33.27<br>20.03<br>30.62<br>23.83<br>29.04<br>**39.90**<br>42.36<br>33.58<br>**20.57**<br>31.54<br>**24.68**<br>**30.49**<br>38.59<br>**44.32**<br>**36.41**<br>20.54<br>**32.51**|9.04<br>10.59<br>0.53<br>7.53<br>6.92<br>8.66<br>10.61<br>0.53<br>6.98<br>6.70<br>9.29<br>11.23<br>**0.62**<br>7.10<br>7.06<br>9.05<br>11.15<br>0.52<br>7.22<br>6.99<br>**9.47**<br>**11.56**<br>0.54<br>**7.65**<br>**7.31**|
||Llama-3-8B-Instruct, KV Size=5|12|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|25.47<br>23.75<br>38.64<br>43.66<br>33.98<br>19.83<br>30.89<br>24.69<br>23.65<br>35.10<br>43.25<br>31.16<br>20.06<br>29.65<br>**25.73**<br>25.44<br>37.84<br>43.78<br>33.95<br>19.83<br>31.09<br>23.84<br>29.21<br>**39.79**<br>44.41<br>36.09<br>20.59<br>32.32<br>24.75<br>**29.75**<br>38.03<br>**44.43**<br>**36.45**<br>**21.67**<br>**32.51**|9.00<br>11.07<br>0.63<br>7.34<br>7.01<br>8.90<br>10.62<br>**0.74**<br>7.57<br>6.96<br>9.22<br>11.18<br>0.53<br>7.42<br>7.09<br>9.13<br>**11.61**<br>0.56<br>7.12<br>7.11<br>**9.34**<br>11.26<br>0.56<br>**7.54**<br>**7.18**|
||Llama-3-8B-Instruct, KV Size=10|24|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|25.76<br>27.50<br>38.38<br>43.40<br>34.81<br>20.07<br>31.65<br>25.38<br>26.83<br>36.90<br>44.09<br>34.24<br>21.49<br>31.49<br>**25.79**<br>29.24<br>38.74<br>43.93<br>36.34<br>19.79<br>32.31<br>24.85<br>**30.94**<br>**39.82**<br>43.52<br>**36.58**<br>20.37<br>32.68<br>24.66<br>30.82<br>39.56<br>**43.97**<br>36.47<br>**22.24**<br>**32.95**|**9.61**<br>11.34<br>**0.53**<br>7.22<br>7.18<br>8.98<br>11.41<br>0.53<br>6.96<br>6.97<br>8.65<br>11.41<br>0.53<br>7.71<br>7.08<br>**9.20**<br>**11.67**<br>**0.55**<br>7.71<br>**7.28**<br>9.02<br>11.51<br>0.47<br>**7.85**<br>7.21|
||Mistral-7B-Instruct, KV Size=Fu|ll|
|FKV|26.63<br>32.99<br>49.34<br>42.77<br>27.35<br>18.78<br>32.98|12.17<br>15.52<br>0.49<br>10.03<br>9.55|
||Mistral-7B-Instruct, KV Size=6|4|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|19.95<br>18.63<br>38.16<br>31.24<br>21.39<br>13.81<br>23.86<br>20.91<br>19.61<br>38.05<br>32.18<br>22.87<br>15.26<br>24.81<br>22.70<br>21.28<br>42.39<br>34.35<br>22.40<br>14.85<br>26.33<br>**24.23**<br>25.22<br>46.02<br>38.82<br>26.05<br>**17.41**<br>29.63<br>21.77<br>**26.57**<br>**48.39**<br>**40.12**<br>**26.76**<br>16.21<br>**29.97**|10.41<br>11.49<br>0.46<br>9.38<br>7.94<br>10.64<br>11.69<br>0.56<br>9.06<br>7.99<br>10.56<br>11.50<br>0.45<br>8.72<br>7.81<br>10.94<br>13.14<br>**0.63**<br>9.11<br>8.46<br>**11.19**<br>**13.94**<br>0.48<br>**9.87**<br>**8.87**|
||Mistral-7B-Instruct, KV Size=12|8|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|21.47<br>21.95<br>45.24<br>33.88<br>21.83<br>15.53<br>26.65<br>21.76<br>21.98<br>43.72<br>32.76<br>22.73<br>15.59<br>26.42<br>21.57<br>24.59<br>46.70<br>35.74<br>25.57<br>14.37<br>28.09<br>23.97<br>**29.60**<br>48.40<br>39.66<br>26.31<br>**18.13**<br>31.01<br>**25.04**<br>27.95<br>**48.48**<br>**41.28**<br>**27.65**<br>18.05<br>**31.41**|10.86<br>12.24<br>0.57<br>8.81<br>8.12<br>10.64<br>11.9<br>0.47<br>8.69<br>7.93<br>11.14<br>12.37<br>0.45<br>9.57<br>8.38<br>11.43<br>13.04<br>0.53<br>**10.26**<br>8.82<br>**11.44**<br>**13.08**<br>**0.63**<br>10.20<br>**8.84**|
||Mistral-7B-Instruct, KV Size=25|6|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|22.26<br>24.94<br>48.30<br>36.76<br>25.16<br>14.93<br>28.72<br>21.42<br>25.36<br>47.94<br>38.75<br>25.82<br>15.30<br>29.10<br>23.81<br>27.04<br>48.56<br>38.32<br>25.34<br>16.42<br>29.91<br>**24.98**<br>29.31<br>49.01<br>41.36<br>**27.16**<br>17.34<br>31.53<br>24.94<br>**31.02**<br>**50.76**<br>**42.11**<br>26.14<br>**18.47**<br>**32.24**|11.07<br>12.39<br>0.53<br>9.13<br>8.28<br>11.57<br>12.35<br>0.56<br>9.51<br>8.50<br>11.67<br>13.57<br>0.52<br>10.53<br>9.07<br>11.94<br>13.30<br>**0.63**<br>**10.95**<br>**9.21**<br>**12.37**<br>**13.88**<br>0.48<br>9.86<br>9.15|
||Mistral-7B-Instruct, KV Size=51|2|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|24.18<br>28.87<br>48.74<br>38.84<br>25.48<br>15.04<br>30.19<br>23.07<br>28.97<br>48.37<br>39.54<br>25.63<br>16.59<br>30.36<br>24.22<br>29.92<br>48.96<br>40.08<br>25.55<br>17.45<br>31.03<br>24.97<br>30.94<br>49.45<br>42.25<br>26.34<br>18.54<br>32.08<br>**25.59**<br>**31.33**<br>**50.26**<br>**42.66**<br>**27.20**<br>**19.37**<br>**32.74**|11.96<br>13.47<br>0.52<br>10.50<br>9.11<br>11.34<br>13.32<br>**0.65**<br>10.81<br>9.03<br>12.12<br>14.53<br>0.52<br>10.57<br>**9.44**<br>**12.09**<br>13.88<br>0.62<br>**10.94**<br>9.38<br>11.62<br>**15.61**<br>0.50<br>9.97<br>9.43|
||Mistral-7B-Instruct, KV Size=10|24|
|SKV<br>PyramidKV<br>Ada-SKV<br>Ours. Copy<br>Ours. Mix|25.38<br>30.22<br>49.29<br>41.84<br>26.60<br>18.08<br>31.90<br>24.28<br>30.05<br>49.17<br>40.49<br>26.43<br>18.80<br>31.54<br>24.82<br>31.49<br>48.80<br>41.18<br>27.38<br>18.22<br>31.98<br>**25.87**<br>31.44<br>49.55<br>**41.95**<br>27.09<br>**19.88**<br>32.63<br>25.64<br>**32.54**<br>**50.49**<br>41.80<br>**27.88**<br>18.89<br>**32.87**|11.69<br>13.89<br>0.52<br>10.54<br>9.16<br>11.77<br>14.51<br>0.51<br>10.19<br>9.25<br>11.96<br>13.82<br>**0.53**<br>9.92<br>9.06<br>**12.21**<br>14.17<br>0.50<br>**10.58**<br>9.37<br>11.94<br>**14.93**<br>0.50<br>10.49<br>**9.47**|



Table 4: Details results for different KV cache (64, 128, 256, 512, 1024) on Llama-3-8B-Instruct and Mistral-7B-Instruct. 

14 

Under review 

## B DATASET DETAILS 

Table 5 provides details of the datasets used in our experiments. To evaluate the retrieval and contextual reasoning abilities of different KV cache compression methods, we use six question-answering datasets from LongBench (Bai et al., 2024) and four additional question-answering datasets from LooGLE (Li et al., 2024b) as benchmarks. For the LooGLE datasets, we randomly selected 100 examples to form the datasets used in this study. 

|Source|Label<br>Task<br>Task Type<br>Eval metric<br>Avg Len<br>Language<br>Nums|
|---|---|
|LongBench<br>LongBench<br>LongBench<br>LongBench<br>LongBench<br>LongBench<br>LooGLE<br>LooGLE<br>LooGLE<br>LooGLE|NrtvQA<br>NarrativeQA<br>Single-Doc. QA<br>F1<br>18,409<br>EN<br>200<br>Qasper<br>Qasper<br>Single-Doc. QA<br>F1<br>3,619<br>EN<br>200<br>MF-en<br>MultiFieldQA-en<br>Single-Doc. QA<br>F1<br>4,559<br>EN<br>150<br>HotpotQA<br>HotpotQA<br>Multi-Doc. QA<br>F1<br>9,151<br>EN<br>200<br>2WikiMQA<br>2WikiMultihopQA<br>Multi-Doc. QA<br>F1<br>4,887<br>EN<br>200<br>Musique<br>Musique<br>Multi-Doc. QA<br>F1<br>11,214<br>EN<br>200<br>Doc.QA<br>Comprehension&reasoning<br>Long Dependency QA<br>F1<br>15,498<br>EN<br>100<br>Info.Retrieval<br>Multiple information retrieval<br>Long Dependency QA<br>F1<br>14,808<br>EN<br>100<br>Timeline<br>Timeline reorder<br>Long Dependency QA<br>F1<br>15,425<br>EN<br>100<br>Computation<br>Computation<br>Long Dependency QA<br>F1<br>17,001<br>EN<br>100|



Table 5: Details of Datasets. 

## C LONGBENCH RESULTS 

We list the evaluation results on the remaining datasets from LongBench (Bai et al., 2024) in Table 6. Our head-level KV cache compression method also outperforms the other baselines, particularly on summarization and Code datasets across Llama-3-8B-Instruct and Mistral-7B-Instruct. 

|Method|Summarization<br>Few-Shot Learning<br>Synthetic<br>Code<br>Avg.<br>GovReport<br>QMSum<br>MultiNews<br>TREC<br>TriviaQA<br>SAMSum<br>PCount<br>PRe<br>LCC<br>RB-P|
|---|---|
||Llama-3-8B-Instruct, KV Size=Full|
|FullKV|28.71<br>23.26<br>26.64<br>73.5<br>90.48<br>42.33<br>4.80<br>69.25<br>59.29<br>54.05<br>47.23|
||Llama-3-8B-Instruct, KV Size=128|
|SnapKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|19.83<br>21.80<br>21.41<br>65.50<br>89.72<br>38.71<br>5.75<br>69.00<br>58.74<br>54.57<br>44.50<br>20.89<br>22.11<br>21.68<br>70.50<br>**90.82**<br>**39.20**<br>5.91<br>**69.50**<br>59.75<br>54.86<br>45.52<br>21.08<br>**22.35**<br>22.50<br>**71.50**<br>89.45<br>38.40<br>5.00<br>**69.50**<br>60.89<br>59.92<br>46.06<br>**21.76**<br>22.16<br>**23.94**<br>**71.50**<br>90.19<br>38.88<br>**6.60**<br>**69.50**<br>**61.08**<br>**60.21**<br>**46.58**|
||Mistral-7B-Instruct, KV Size=Full|
|FullKV|32.87<br>24.24<br>27.10<br>71.00<br>86.23<br>42.79<br>2.75<br>86.98<br>56.93<br>54.49<br>48.54|
||Mistral-7B-Instruct, KV Size=128|
|SnapKV<br>Ada-SnapKV<br>HeadKV-R<br>HeadKV-R2|20.76<br>22.72<br>21.38<br>67.00<br>85.06<br>40.22<br>3.51<br>65.06<br>52.20<br>47.01<br>42.49<br>21.13<br>22.76<br>22.25<br>68.50<br>**85.60**<br>41.05<br>3.33<br>62.54<br>52.88<br>49.25<br>42.93<br>22.19<br>22.86<br>22.57<br>69.50<br>85.46<br>**41.16**<br>3.56<br>**74.49**<br>54.60<br>50.89<br>44.73<br>**24.30**<br>**23.48**<br>**24.18**<br>**70.50**<br>85.54<br>40.72<br>**4.83**<br>72.63<br>**55.49**<br>**51.39**<br>**45.31**|



Table 6: Results on LongBench (KV cache=128). 

## D REASONING-IN-A-HAYSTACK DETAILS 

As shown in Figure 9, we provide examples for each task used in the Reasoning-in-a-Haystack test[3] . We use the dataset curated by Kuratov et al. (2024) for the Reason-in-a-Haystack test. Specifically, they use the “Input” listed in Figure 9 as the needle and split it into different sentences using dot. These sentences are then inserted into various positions within the haystack to form the final Reasonin-a-Haystack test examples. 

As shown in Figure 10, we provide detailed results across different context lengths and tasks. Our proposed head-level KV cache compression method achieves the best average score among all methods. Notably, on the Llama-3-8B-Instruct model, our method surpasses the Full KV cache on the 

> 3https://huggingface.co/datasets/RMT-team/babilong 

15 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0016-01.png)


**----- Start of picture text -----**<br>
Retrieval Heads 0.05 Enhanced-Retrieval Heads Retrieval-Reasoning Heads 0.016<br>0.012 0.014<br>0.04<br>0.010 0.012<br>0.03 0.008 0.010<br>0.008<br>0.006<br>0.02 0.006<br>0.004<br>0.004<br>0.01<br>0.002 0.002<br>0.00 0.000<br>Layers Layers Layers<br>0 0 0<br>2 2 2<br>4 4 4<br>6 6 6<br>8 8 8<br>10 10 10<br>12 12 12<br>14 14 14<br>16 16 16<br>Heads 18 18 18<br>20 20 20<br>22 22 22<br>24 24 24<br>26 26 26<br>28 28 28<br>30 30 30<br>0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30<br>**----- End of picture text -----**<br>


Figure 7: Head visualization results on Mistral-7B-Instruct. Our enhanced distribution can also cover the important retrieval heads. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0016-03.png)


**----- Start of picture text -----**<br>
Mistral-7B-Instruct<br>FullKV  Average score: 100.00 SnapKV  Average score: 80.10<br>PyramidKV  Average score: 91.60 Ada-SnapKV  Average score: 95.00<br>HeadKV-R  Average score: 94.20 HeadKV-R2  Average score: 96.00<br>Context Length Context Length<br>**----- End of picture text -----**<br>


Figure 8: Neele-in-a-Haystack test results on Mistral-7B-Instruct. The results are consist with those on Llama-3-8B-Instruct and our Retrieval-Reasoning Heads distribution (HeadKV-R2 still outperforms other strong baselines. 

QA3 task, the most challenging task among the five, as indicated in Kuratov et al. (2024). This suggests that our Retrieval-Reasoning Heads distributions are better at capturing semantically relevant and important content, leading to generating the correct answers. 

16 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0017-01.png)


**----- Start of picture text -----**<br>
Input : John travelled to the hallway. Mary journeyed to the<br>bathroom. Daniel went back to the bathroom. John moved to the<br>bedroom.<br>qa1<br>Question : Where is Mary?<br>Target : bathroom<br>Input : Daniel took the milk there. John journeyed to the garden.<br>Daniel went back to the hallway. Daniel journeyed to the bathroom.<br>Daniel dropped the milk. Daniel took the milk there. John grabbed<br>the apple there. Sandra journeyed to the kitchen. John went to the<br>qa2<br>hallway. Sandra went back to the garden.<br>Question :  Where is the apple?<br>Target : hallway<br>Input :  Mary picked up the apple. John went to the garden. Sandra<br>travelled to the office. Sandra took the milk. John went to the<br>bedroom. Sandra went to the kitchen. John journeyed to the office.<br>Mary left the apple. Mary travelled to the office. Sandra went to the<br>qa3<br>office. Daniel went to the hallway. Sandra discarded the milk.<br>Question :  Where was the milk before the office?<br>Target : kitchen<br>Input : The hallway is east of the bathroom. The bedroom is west<br>of the bathroom.<br>qa4<br>Question : What is the bathroom east of?<br>Target : bedroom<br>Input : Fred picked up the football there. Fred gave the football to<br>Jeff. Bill went back to the bathroom. Jeff grabbed the milk there.<br>Jeff gave the football to Fred. Fred handed the football to Jeff.<br>qa5 Jeff handed the football to Fred. Fred gave the football to Jeff.<br>Question :Who did Fred give the football to?<br>Target : Jeff<br>**----- End of picture text -----**<br>


Figure 9: Examples for each task used in Reasoning-in-a-Haystack test. 

17 

Under review 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning_images/iclr_2025_Not_All_Heads_Matter_A_Head-Level_KV_Cache_Compression_Method_with_Integrated_Retrieval_and_Reasoning.pdf-0018-01.png)


**----- Start of picture text -----**<br>
Llama-3-8B-Instruct Mistral-7B-Instruct<br>FullKV. Average Score: 57.04 100 FullKV. Average Score: 45.29 100<br>98 91 90 85 68 93 84 73 63 67 59 42<br>80 80<br>45 42 47 43 35 49 35 28 19 10 6 10<br>60 60<br>31 29 28 29 24 41 30 30 24 30 23 20<br>40 40<br>65 61 52 57 64 40 48 56 49 51 46 47<br>20 20<br>76 74 62 69 61 84 79 75 49 44 41 40<br>0 0<br>0k 1k 2k 4k 8k 0k 1k 2k 4k 8k 16k 32k<br>SnapKV. Average Score: 54.16 100 SnapKV. Average Score: 41.09 100<br>97 91 92 78 61 93 80 66 46 60 54 43<br>80 80<br>38 34 38 33 26 44 25 20 10 3 2 1<br>60 60<br>28 25 30 26 20 22 19 20 21 17 20 17<br>40 40<br>65 61 52 57 64 40 49 54 47 51 47 44<br>20 20<br>76 74 62 69 57 84 80 73 48 46 43 40<br>0 0<br>0k 1k 2k 4k 8k 0k 1k 2k 4k 8k 16k 32k<br>PyramidKV. Average Score: 54.44 100 PyramidKV. Average Score: 41.09 100<br>98 90 91 78 62 93 83 60 51 59 46 40<br>80 80<br>40 35 34 33 30 47 20 22 11 6 2 1<br>60 60<br>27 29 25 26 23 28 22 24 23 19 18 14<br>40 40<br>65 62 52 57 63 40 49 54 48 52 48 44<br>20 20<br>76 74 62 69 60 84 79 74 47 45 42 43<br>0 0<br>0k 1k 2k 4k 8k 0k 1k 2k 4k 8k 16k 32k<br>Ada-KV. Average Score: 55.00 100 Ada-SnapKV. Average Score: 42.94 100<br>98 92 90 78 63 93 80 65 49 58 52 45<br>80 80<br>37 41 44 39 28 44 24 21 14 6 4 5<br>60 60<br>33 28 21 25 19 28 24 19 20 22 13 12<br>40 40<br>65 61 52 57 64 40 49 54 46 52 47 44<br>20 20<br>76 74 62 69 59 85 79 72 48 44 42 38<br>0 0<br>0k 1k 2k 4k 8k 0k 1k 2k 4k 8k 16k 32k<br>Ours.Retrieval. Average Score: 55.16 100 Ours.Retrieval. Average Score: 42.94 100<br>98 91 91 77 63 92 80 65 56 60 54 45<br>80 80<br>39 35 35 41 32 47 30 28 14 9 5 6<br>60 60<br>30 24 23 31 27 29 33 34 27 25 15 16<br>40 40<br>65 61 52 57 65 40 48 54 45 49 45 43<br>20 20<br>76 74 62 69 61 84 79 73 48 44 40 41<br>0 0<br>0k 1k 2k 4k 8k 0k 1k 2k 4k 8k 16k 32k<br>Ours.R-Retrieval. Average Score: 56.84 100 Ours.R-Retrieval. Average Score: 43.09 100<br>97 93 92 84 67 93 85 74 53 62 57 43<br>80 80<br>39 38 36 39 33 49 33 23 16 7 6 4<br>60 60<br>36 36 33 30 28 35 28 32 25 26 17 13<br>40 40<br>65 61 52 57 67 40 46 55 44 49 43 42<br>20 20<br>76 74 62 69 57 84 79 72 48 43 40 42<br>0 0<br>0k 1k 2k 4k 8k 0k 1k 2k 4k 8k 16k 32k<br>Context Length Context Length<br>qa1 qa1<br>qa2 qa2<br>qa3 qa3<br>qa4 qa4<br>qa5 qa5<br>qa1 qa1<br>qa2 qa2<br>qa3 qa3<br>qa4 qa4<br>qa5 qa5<br>qa1 qa1<br>qa2 qa2<br>qa3 qa3<br>qa4 qa4<br>qa5 qa5<br>qa1 qa1<br>qa2 qa2<br>qa3 qa3<br>qa4 qa4<br>qa5 qa5<br>qa1 qa1<br>qa2 qa2<br>qa3 qa3<br>qa4 qa4<br>qa5 qa5<br>qa1 qa1<br>qa2 qa2<br>qa3 qa3<br>qa4 qa4<br>qa5 qa5<br>**----- End of picture text -----**<br>


Figure 10: Detail results for Reasoning-in-a-Haystack test on five datasets. 

18 

