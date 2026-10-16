# **Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection** 

**Yun Zhu**[1] **, Jia-Chen Gu**[3] **, Caitlin Sikora**[2] **, Ho Ko**[2] **, Yinxiao Liu**[1] **, Chu-Cheng Lin**[2] **, Lei Shu**[1] **, Liangchen Luo**[1] **, Lei Meng**[2] **, Bang Liu**[4] **, Jindong Chen**[1] 1Google DeepMind 

2Google 

3University of California, Los Angeles 

4Université de Montréal & Mila 

`{yunzhu,csikora,hoko,canoee,kitsing,leishu,luolc,leimeng,jdchen}@google.com gujc@ucla.edu` , `bang.liu@umontreal.ca` 

## **Abstract** 

Large language models (LLMs) augmented with retrieval exhibit robust performance and extensive versatility by incorporating external contexts. However, the input length grows linearly in the number of retrieved documents, causing a dramatic increase in latency. In this paper, we propose a novel paradigm named Sparse RAG, which seeks to cut computation costs through sparsity. Specifically, Sparse RAG encodes retrieved documents in parallel, which eliminates latency introduced by long-range attention of retrieved documents. Then, LLMs selectively decode the output by only attending to highly relevant caches auto-regressively, which are chosen via prompting LLMs with special control tokens. It is notable that Sparse RAG combines the assessment of each individual document and the generation of the response into a single process. The designed sparse mechanism in a RAG system can facilitate the reduction of the number of documents loaded during decoding for accelerating the inference of the RAG system. Additionally, filtering out undesirable contexts enhances the model’s focus on relevant context, inherently improving its generation quality. Evaluation results of two datasets show that Sparse RAG can strike an optimal balance between generation quality and computational efficiency, demonstrating its generalizability across both short- and long-form generation tasks. 

## **1 Introduction** 

Large language models (LLMs) have attracted increasing attention and exhibited impressive abilities to understand instructions and generate fluent outputs in natural language [4, 27, 35, 34]. Nevertheless, LLMs inevitably manifest hallucinations [13] due to their struggle with factual errors and inability to secure the accuracy of generated text solely by the parametric knowledge they encapsulate [39, 26]. Feeding the source of truth to LLMs in the format of retrieved context segments [29] alleviates this problem. The technique is widely known as Retrieval-Augmented Generation (RAG) [20, 21, 8]. 

Although the RAG framework is empirically shown to be effective, it can be expensive to scale up. This is because it requires prepending relevant documents retrieved from an external knowledge corpus to the queries [8]. As a result, the input length grows linearly in the number of documents, causing a dramatic increase in latency when using a standard Transformer whose the latency scales quadratically with the input length. Some prior works such as Fusion-in-Decoder (FiD) [12] and Parallel Context Windows (PCW) [28] have proposed to alleviate this issue. Yet these methods fail to strike an optimal balance between generation quality and computational efficiency. FiD was originally 

Preprint. Under review. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Accelerating_Inference_of_Retrieval-Augmented_Generation_via_Sparse_Context_Selection_images/iclr_2025_Accelerating_Inference_of_Retrieval-Augmented_Generation_via_Sparse_Context_Selection.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Retrieved Documents<br>Retrieval question d1 d2 … dN<br>question d1 d2 … dN [ASSESS] c1 Keep<br>LLM<br>Document question d1 d2 … dN [ASSESS] c2 Drop<br>Assessment … …<br>question d1 d2 … dN [ASSESS] cN Keep<br>Load key-value cache that<br>LLM<br>are assessed to keep<br>Generation Cq c1 c2 … cN [GENERATE] Answer<br>**----- End of picture text -----**<br>


Figure 1: An overview of Sparse RAG at inference. Each of the retrieved documents will be assessed by LLMs to decide whether to keep or drop by estimating a relevance score. Then, load the documents that are considered to keep for generation. 

designed for the encoder-decoder architecture, and thus is not compatible with currently prevalent decoder-only architectures without significant changes. While PCW can be applied to decoder-only LLMs, it only speeds up the model pre-filling and still incurs high latency since the whole context window cache is still being attended to when decoding each token. Moreover, the heavy reliance of generation on the retrieved knowledge raises significant concerns about the model’s behavior and performance in scenarios where retrieval may fail or return inaccurate results [33]. A typical approach for mitigating this issue is to rely on an external classifier to rank or filter the documents before prepending them to the input [36], but this process requires extra model calls which adds new complexity to inference. 

In light of the issues above, we propose a novel paradigm named Sparse RAG. It generally operationalizes by massive pre-filling where the key-value cache was generated through a single forward pass of the input tokens and selective decoding where the output is generated by attending to highly relevant tokens auto-regressively. Previous works where the length of the retrieved contexts during pre-filling are equal to that during decoding are called _dense_ -RAG in this paper. Different from that, Sparse RAG allows the pre-filled context to be significantly larger than the decoding context, where partial low-quality context is dynamically dropped according to their relevance to the input query. Furthermore, Sparse RAG combines the assessment of each individual context and the generation of the response into a single process. Special control tokens are used to prompt LLMs to assess each retrieved context for picking high-quality ones. Then only the key-value caches of those picked contexts are loaded into LLMs for decoding with another control tokens. 

The design of such a sparse mechanism in RAG system brings two unique advantages. Firstly, by reducing the number of key-value cache loads during the decoding process, LLMs can achieve lower latency as they are typically constrained by memory usage. Besides, filtering out undesirable contexts enhances the model’s focus on relevant context, inherently improving its generation quality. To demonstrate the effectiveness and efficiency of the proposed method, we evaluate on two datasets of PopQA [24], and Biography [25]. Experimental results show that Sparse RAG can achieve the similar or better performance while keep much better latency compared with standard dense-RAG or PCW-RAG, demonstrating its generalizability across both short- and long-form generation tasks. 

## **2 Related Work** 

**Retrieval-Augmented Generation** RAG is a family of techniques for generating output structures while using retrieved nearest-neighbor structures as a reference. It typically involves two stages: 

2 

Table 1: Comparisons with existing RAG-related works. 

|Approach|Corrective|No extra model|Prefll effciency|Decode effciency|
|---|---|---|---|---|
|RAG [19]|No|Yes|No|No|
|Corrective RAG [36]|Yes|No|No|No|
|PCW RAG [28]|No|Yes|Yes|No|
|Sparse RAG|Yes|Yes|Yes|Yes|



retrieval and generation. Retrieval finds most similar contexts based on BM25 or learned embedding, where the context could be represented as token embedding [16, 37], dense embedding [5] or raw text [9, 12, 19]. Once those contexts are retrieved, different architectures are leveraged to incorporate them into the model. Popular approaches includes concatenation [12, 19] and cross-attention [3, 18]. In recent years, the LLM architecture evolves towards a single decoder only model with significant larger sizes. To this end, concatenation of raw text [19] is becoming popular for being simple and practical. Many advanced approaches have been developed on top of it in recent years. Self-RAG [2] is proposed to selectively retrieve knowledge and introduce a critic model to decide whether to retrieve. Yoran et al. [38] designed an NLI model to identify the irrelevant context and improve robustness. SAIL [23] is tuned on instructions to insert retrieved documents before instructions. Toolformer [30] is pre-trained for calling APIs such as Wikipedia. Jiang et al. [14] actively anticipate future content and decide when and what to retrieve in long-form generation. CRAG [36] makes the attempt to explore and design corrective strategies for RAG to improve its robustness of generation. Specifically, an external T5 model is trained and used to determine the usefulness of the retrieved context. Generally, these approaches target on exploiting retrieval as a useful tool to augment generation or whether retrieval is necessary. 

**Efficiency in RAG** The efficiency of LLM inference is a widely explored research area, where different categories of approaches have been studied. Some works are towards architecture level acceleration such as efficient attention [31], Mixture of Expert [6], Transformer-alternative architectures [7], etc. While some other works are algorithm-level acceleration like quantization [22] or speculative decoding [17]. Nevertheless, most previous works are not RAG specific. Recently, RAG Cache [15] was proposed as a multilevel dynamic caching system tailored for RAG from the system perspective. Another stream of the work try to process each document individually so that no cross document attention computation is required, such as FiD [12] or PCW [28]. FiD encoded each retrieved passage independently from other passages and decoded by attending over the concatenation of the resulting representations of all the retrieved passages. PCW carved a long context into chunks (“windows”), restrict the attention mechanism to apply only within each window, and re-use the positional embeddings across the windows. 

Comparison with previous works that are the most relevant to our work is illustrated in Table 1. This work aims to strike an optimal balance between generation quality and computational efficiency. It is notable that the extra classifier in CRAG requires maintaining extra model with more complex serving infrastructure. When there is _N_ contexts retrieved. There is totally _N_ + 1 model runs. Our work also relies an classification process to refine the retrieved documents. However, our approach relies on the “internal” classification process that is aligned with the generation process. The total model run in our case is 1. 

## **3 Sparse RAG** 

Sparse RAG is designed for the decoder-only model architecture, which is the default case for most popular LLMs. Figure 1 presents an overview of Sparse RAG at inference, which designs the document assessment strategies to improve the robustness of generation. The key hypothesis of our approach is that the RAG task and per context assessment is a similar tasks and the model could do both tasks in one shot. We found simple and effective approaches to enable them. 

## **3.1 Training Process** 

Our work is based on the assumption that a certain amount of RAG training data is accessible, which allows us to effectively tailor and adapt existing LLMs to our specific needs. In the training phase, we integrate an additional Per Context Assessment (PCA) task into the training mixture. This PCA 

3 

task is included alongside the primary RAG task to enhance the model’s ability to assess and respond accurately within various contexts. By incorporating the PCA task, we aim to improve the overall performance and contextual understanding of the LLMs, ensuring they are more adept at handling a diverse range of scenarios presented during their use. 

**Data Augmentation with LLMs** For typical RAG data, one question-answer pair can be mapped to multiple retrieved contexts using either BM25 or existing standalone retriever. However, there are cases where no golden label regarding the quality of every retrieved context is available. To collect these missing labels, we leveraged two off-the-shelf LLMs [1, 34]–PALM2 and Gemini–to get the assessment for each context. We observe empirically that a second round of prompting for critique, especially using a different model from the initial round, ensures the best quality labels. We provide our prompts in Table 9 in the Appendix. We compare different model combinations for labeling to human ground truth labels in Section 4. 

**Multitasking Data Format** The LLM is trained on a mixture of two types of tasks, “Rate” assessment and answer generation. Specifically, we format the inputs and outputs of the two task types as 

- Assessment: {Question}{Context}{Control_Assessment}{Rate} 

- Generation: {Question}{Context_1}...{Context_N}{Control_Generation}{Answer} 

where Rate and Answer are targets for the generative tasks while everything before them are inputs. Control_Assessment and Control_Generation are fixed tokens to ensure the LLM can differentiate the two tasks. 

**Parallel Contexts** Since each context is rated independently in PCA task, we introduce this independence in the primary RAG task as well so two tasks could reuse KV cache during inference time. Thus for the generation task, we enforce no cross attention between different two contexts following the Parallel Context Windows [28] to process the input for training. Specifically, we modify two things in the standard LM training process. First, we change the attention masks to be block-wise, and restrict that context_i and context_j will not attend to each other. Question, Context_i and Control_Generation still follow the default attention mechanism, i.e. in the causal attention case, the latter could attend to the previous ones. Second, we adjust the position encoding to “mimic” the situation that Context_i is not visible to context_j that each context will have its own incremental positional ids. However, the position id of Control_Generation is designed to back to normal, where its position considers all the length of contexts. 

## **3.2 Inference Process** 

Given the question and retrieved contexts, Sparse RAG handles the assessment task and generation task in one single pass. 

**Per Context Assessment** Similar to the training process, when pre-filling the KV cache, each retrieved context is treated independently. Then these KV cache will be used to score the input, which is the concatenation of the token of Control_Assessment and the token of “Good”. The score is the probability of this tokens as the next tokens. 

**Generation** The generation uses a filtered KV cache, where only _K_ out of _N_ KV cache will finally be loaded. We use a simple threshold based filtering approach: we drop the context when its score is less than _sigma_ . Once those caches are loaded, Control_Generation will be used to prompt the model to generate the answers. 

4 

Table 2: Auto-rater comparison to ground truth. 

|Auto-labeling method|Auto-labeling method|Average F1|F1 Label 0|F1 Label 1|
|---|---|---|---|---|
|Rater model|Critic model||||
|PALM2 XL|n/a|0.729|0.765|0.694|
|PALM2 XL|PALM2 XL|0.781|0.820|0.741|
|Gemini Ultra|n/a|0.761|0.807|0.716|
|Gemini Ultra|Gemini Ultra|0.704|0.747|0.660|
|PALM2 XL|Gemini Ultra|0.728|0.776|0.680|
|Gemini Ultra|PALM2 XL|**0.821**|**0.861**|**0.781**|



## **4 Evaluation of Per Context Assessment** 

## **4.1 New Benchmark: Natural Question Per Context Assessment** 

We isolated a subset of 50 questions, each with 10 retrieved contexts, from the Natural Questions dataset. We assigned 3 raters to each question-context pair from a pool of 7 raters and provided the instructions in Section A.2. 

We aggregated responses for all 3 raters for each context, selecting the majority decision 0 or 1 for each context. We found that raters unanimously agreed on 351 out of 500 context, with 30% of the documents considered relevant. For questions where raters were not unanimously decided, a specialist rater was assigned to investigate more carefully and set the best label to correct mistakes of the other raters. This resulted in 6 additional documents considered relevant out of the entire dataset, boosting the portion of relevant documents to 31% and slightly increasing alignment with the auto-rater approaches (average F-score increase of 1.4% across auto-rater methods using these corrections as the ground truth). 

## **4.2 LLM rater comparisons** 

We tested several different LLM-based automatic labeling methods–different combinations of models and prompts–for creating training data for the classifier in Sparse RAG. We compared several of these auto-rater approaches by creating a ground-truth relevance dataset using human labeling. The auto-rater comparison using the revised human labels as the ground-truth is shown in Table 2. We find that combining two different models in two rounds–initial prompting and critique–provides the labels that are most closely aligned with the human labels. We hypothesize that the different representations learned by two different models are able to capture the most nuance in the input sequences, leading to better relevance judgements. We also observe that Gemini Ultra appears slightly less effective at critiquing model outputs than PALM2 XL. 

## **5 Evaluation of Sparse RAG** 

## **5.1 Benchmarks and Metrics** 

**PopQA** is a large-scale open-domain question answering (QA) dataset, consisting of 14k entitycentric QA pairs. Each question is created by converting a knowledge tuple retrieved from Wikidata using a template. We follows the setup from [36] and use Contriver [11] to retrieve the related contexts. Since PopQA is lacking the per-context assessment label, we adopted the “Gemini + PALM2” combination to get training labels. We split the dataset into training, validation and test set with 8:1:1 ratio. Since the answer is usually short, we report **Exact Match (EM)** and **F1** scores. 

**QMSum** [40] is a human-annotated benchmark for query-based multi-domain meeting summarization task, which consists of 1,808 query-summary pairs over 232 meetings in multiple domains. To adopt it in the RAG domain, we divide each conversation into different contexts where the average context contains 300 words, and filter the data point where the conversation containing ground truth label is split into multiple contexts. Note that this dataset has human labeled per-context assessment that we could leverage during training. Eventually we get 250 training data, 70 validation and 77 test set. The target of this data is longer and we report **RougeLSum** and **F1** scores. 

5 

## **5.2 Baselines** 

**RAG** We evaluated the standard RAG [20] where an LM generates output given the query prepended with all the top ranked documents using the same retriever as in our system. 

**PCW RAG** We applied the Parallel Context Windows [28] to the RAG process, where no cross attention is applied between documents. 

**Corrective RAG** It is the same with RAG except that an external T5-XXL classifier is trained using heuristic labels [36]. This classifier is used to process all the document and decide the rank. Note to facilitate the fair comparison, we did not adopt the "web search" feature of this paper. 

## **5.3 Experimental Configuration** 

The base LLMs used in the paper Gemini [34]. Although our approach could be applied to different training stage of the model, we apply LoRA tuning [10] to enforce alignment on top of the foundation LLMs due to its low resource requirement and wide usage. Note that the same LoRA tunning is applied to Sparse RAG and all baselines. In all our experiment, we apply LoRA in self-attention and use the default rank as 4. By default, we use the XXS size which could be running on-device. 

During training, we use 64 Tensor Processing Units (TPU) V3 chips for PopQA while use 128 Units for . The batch size is 64. We use the Adafactor optimizer [32] with a learning rate of 0 _._ 003. The training dropout rate is 0 _._ 05. We leverage the metrics of validation set to pick the best checkpoint. During inference, the temperature is set to 0 _._ 5. Unless specifically noted, we use sampling decoding with sample number 1 for our experiments. 

## **5.4 Inference Setup and Metrics** 

Evaluation of Sparse RAG was conducted on a Samsung S21 Ultra, utilizing the device’s CPU to assess real-world performance on a relatively mid-tier smartphone compared to the latest flagship models. Inference configuration consisted of fixed token lengths for queries, contexts and generated responses. This setup allows for evaluating the system’s efficiency and effectiveness under resource constraints typical of mobile devices, providing insights into its practical applicability for on-device question-answering tasks. 

Specifically, the overall inference process considers two stages. 

- Prefill stage: For baseline RAG Model, we measure the total time taken to process all input tokens (question and all contexts). For PCW RAG and Sparse RAG models: We take advantage of these models’ ability to cache question processing. We first measure the time to process the question alone. Then, we measure the time to process different contexts with the pre-processed kv-cached question. We use **Encoding Speed (ES)** (tokens per second (t/s)) to measure the efficiency of prefill stage. 

- Decoding stage: To assess decoding speed comprehensively, we generate output sequences using the same question but varying the amount of relevant context data considered, ranging from the top 1 most relevant document to the top _K_ . For each context size, we produce output sequences of different lengths. This systematic approach allows us to evaluate the impact of both context size and response length on decoding speed. We use **Decoding Speed (DS)** (tokens per second (t/s)) to measure the efficiency of decoding stage. 

## **5.5 Main Results** 

We report the results with both short form answer (PopQA) and long form answer (QMSum) in Table 3, where both quality and latency metrics are shared. “K” is the number of chosen memories. 

Notably, our proposed approach achieves the best quality while being the most efficient during inference compared to other “Dense RAG” approaches. It can be seen that Sparse RAG shares the same pre-filling efficiency with PCW, due to the parallel context encoding. However Sparse RAG can achieve significant better decoding efficiency than standard RAG and PCW RAG. In detail, out of 20 retrieved contexts, the Sparse RAG has average of 7 _._ 84 context for PopQA and 4 _._ 45 context 

6 

Table 3: Quality & Efficiency Tradeoff. For both short form and long form generation tasks, Sparse RAG achieves both higher quality and efficiency compared to “dense” RAG approaches. 

||ES|PopQA<br>EM<br>F1<br>K<br>DS|QMSum<br>F1<br>RougeLSum<br>K<br>DS|
|---|---|---|---|
|||||
|RAG [19]<br>PCW-RAG[28]<br>CRAG [36]<br>Sparse RAG|56.28<br>**147.58**<br>-<br>**147.58**|65.43<br>69.99<br>20<br>6.65<br>65.04<br>69.54<br>20<br>6.65<br>66.52<br>70.99<br>8.90<br>-<br>**67.17**<br>**71.16**<br>**7.84**<br>**12.28**|21.43<br>18.20<br>20<br>6.64<br>20.18<br>16.95<br>20<br>6.64<br>-<br>-<br>-<br>-<br>**23.96**<br>**20.10**<br>**4.45**<br>**16.05**|



for QMSum. This leads to almost **double** or even **triple** decoding speed. Meanwhile, Sparse RAG achieves higher quality metrics than the dense counterparts, which shows that Sparse RAG effectively filtered noisy and irrelevant contexts. 

On the other hand, it is interesting to compare Corrective-RAG(CRAG) with Sparse RAG on PopQA task since Corrective-RAG trained their own external classifier with T5 XXL based on PopQA data. Results suggested that our approach, albeit using an “in-place” classifier, is outperforming the CRAG with external classifier. Note the encoding and decoding speed of CRAG is not comparable because it involves multiple model running from the classifier. 

## **5.6 Analysis** 

## **5.6.1 Impact of Confidence Threshold** 

To show how the metrics changes with different thresholds, the performance of Sparse RAG corresponding to different thresholds is illustrated in Table 4. As the threshold gradually increases, the system will filter out more contexts, which would always reduce the latency during inference. In terms of generation quality, it can be seen that the performance of Sparse RAG was significantly improved as thresholds increased at the beginning, which shows the effectiveness of filtering out irrelevant contexts. Then, the performance was stable and dropped slightly. The reason might be that some truly relevant contexts are accidentally filtered out. 

Table 4: Sampling a few confidence threshold on PopQA and QMSum. Higher confidence threshold means filtering more memories. Efficiency wise, filtering more always helps; While in terms of quality. 

|y.|||
|---|---|---|
|Threshold|PopQA<br>EM<br>F1<br>K<br>DS|QMSum<br>F1<br>RougeLSum<br>K<br>DS|
||||
|0.05<br>0.1<br>0.15<br>0.2<br>0.25<br>0.3|66.95<br>70.97<br>9.75<br>10.61<br>66.84<br>70.66<br>8.72<br>11.70<br>**67.17**<br>**71.16**<br>7.84<br>12.28<br>66.77<br>70.54<br>7.13<br>12.88<br>65.75<br>69.64<br>6.56<br>13.00<br>63.86<br>68.2<br>**5.98**<br>**13.08**|22.85<br>19.49<br>7.92<br>11.92<br>23.78<br>19.98<br>6.68<br>12.89<br>23.43<br>19.66<br>5.77<br>13.01<br>23.2<br>19.79<br>5.05<br>14.54<br>**23.96**<br>**20.1**<br>4.45<br>16.05<br>23.84<br>19.99<br>**3.93**<br>**16.38**|



## **5.6.2 Silver Labels vs LLM Labels** 

In Corrective RAG, the T5 model was trained with silver labels that comes from the title matching [36]. We collect the same silver labels to replace the LLM labels and train the model with this new dataset. The comparison result is shown in Table 5. 

From these results, we can observe that the quality of the labels generated by LLMs is higher than that of the silver labels from Yan et al. [36]. We hypothesize that the superior quality of the LLM-generated labels is attributable to our methodology, which involved a two-round process of soliciting responses from two different LLMs. By engaging two distinct models, we likely enhanced the robustness and accuracy of the labels through a form of cross-validation, thereby mitigating potential biases or errors that might arise from relying on a single LLM. 

7 

Table 5: Comparing silver labels with LLM labels on PopQA. 

|Approach|EM|F1|K|DS|
|---|---|---|---|---|
|Sparse RAG|**67.17**|**71.16**|**7.84**|**12.28**|
|w/ silver label|66.97|71.05|8.26|11.99|



## **5.6.3 Ablation of Prefill Documents** 

One question one may have is that the better performance may rely on the “massive” prefilling but in practical situation, we can retrieve smaller amount of context which have better quality. To this end, we compare different numbers of the prefill document: 10 vs 20 for PopQA, and show the results in Table 6. Even in the setting lower prefilled case, the previous conclusion of Sparse RAG remains. It achieves better quality with higher speed of decoding. It makes sense that the gap between dense RAG and sparse RAG is relative smaller at 10 prefill documents. This is because the headroom is smaller as the top 10 documents naturally has less to filter and make it less “sparse”. Yet our approach still achieves best trade-off, showing the capability to generalize with different numbers of input documents. 

Table 6: Ablation on Different number of prefill documents for PopQA. 

|Approach|Prefll Documents|EM|F1|K|ES|DS|
|---|---|---|---|---|---|---|
|RAG|10|64.66|68.67|10|80.74|10.31|
|PCW RAG|10|63.9|68.58|10|**147.48**|10.31|
|Sparse RAG|10|**65.86**|**70.2**|**7.79**|**147.48**|**12.33**|
|RAG|20|65.43|69.99|20|56.28|6.65|
|PCW RAG|20|65.04|69.95|20|147.58|6.65|
|Sparse RAG|20|67.17|71.16|7.84|147.58|12.28|



## **5.6.4 Inference Efficiency Ablations** 

Focusing on the decoding stage, we delve into the performance differences when varying the number of retrieved memories (i.e., top-K documents) and the length of the generated response. This analysis provides valuable insights into the computational overhead associated with incorporating additional context and generating more extensive outputs, ultimately shedding light on the practical trade-offs for real-time question-answering on resource-constrained devices. 

As illustrated in Fig 2b, we observed that the baseline method requires over 50%(varies from 52% to over 55% depending on number of output tokens) more time to generate outputs of varying lengths compared to the Sparse RAG approach. This significant disparity in generation speed is primarily attributed to the increased computational burden incurred by the baseline method when considering a larger number of memories. As shown in Fig 2a, this heightened demand for computational resources results in a notable slowdown in terms of tokens per second (t/s). This observation underscores the efficiency advantages offered by Sparse RAG, especially in scenarios where a substantial amount of context needs to be considered during the decoding process. 

## **5.6.5 Ablation on Foundation Model Size** 

We applied Sparse RAG to different sizes of LLMs, specifically testing it on Gemini XS and Gemini XXS. The results of these experiments are presented in Table 7. The findings demonstrate that Sparse RAG is compatible with various foundation models, effectively adapting to different model sizes. Notably, with a reduced amount of decoding caches, Sparse RAG is capable of achieving the highest quality results. This indicates that Sparse RAG maintains its efficiency and effectiveness across different foundation models, making it a versatile approach for various LLM configurations. 

## **5.6.6 Using Golden Context Label during inference** 

Since QMSum provides golden per-context labels, we leverage these labels during inference to evaluate the upper bound performance under the condition of perfect per-context assessment. By 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Accelerating_Inference_of_Retrieval-Augmented_Generation_via_Sparse_Context_Selection_images/iclr_2025_Accelerating_Inference_of_Retrieval-Augmented_Generation_via_Sparse_Context_Selection.pdf-0009-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_Accelerating_Inference_of_Retrieval-Augmented_Generation_via_Sparse_Context_Selection_images/iclr_2025_Accelerating_Inference_of_Retrieval-Augmented_Generation_via_Sparse_Context_Selection.pdf-0009-01.png)


(a) Decoding speed with different number of memories (b) E2E Latency for decoding different number of tokens 

Figure 2: Inference Efficiency Comparison 

Table 7: Ablation on Different LLM model sizes. 

|Approach|Model Size|EM|F1|K|
|---|---|---|---|---|
|RAG|XS|66.52|70.87|20|
|PCW RAG|XS|65.75|70.37|20|
|Sparse RAG|XS|**68.26**|**72.26**|**6.27**|
|RAG|XXS|65.43|69.99|20|
|PCW RAG|XXS|65.04|69.95|20|
|Sparse RAG|XXS|67.17|71.16|7.84|



utilizing the highest quality labels available, we aim to determine the best possible outcomes that our model can achieve. The results of this experiment, which highlight the performance ceiling under ideal labeling conditions, are presented in Table 8. This approach allows us to understand the maximum potential of our model when supplied with optimal input data, thereby offering insights into its ultimate capabilities. 

Table 8: Trying golden labels on QMSum. 

|Approach|F1|RougeLSum|K|DS|
|---|---|---|---|---|
|Sparse RAG|23.96|20.1|4.45|16.05|
|+ golden label|**26.76**|**21.93**|**1.13**|**21.16**|



## **6 Conclusion** 

This paper presents Sparse RAG to address the challenges of increased input length and latency. Through a novel approach of massive pre-filling and selective decoding, Sparse-RAG efficiently manages the key-value cache of retrieved documents, allowing the LLMs to focus on highly relevant tokens. This selective attention mechanism not only reduces the computational burden during inference but also enhances the generation quality by filtering out irrelevant contexts. The evaluation on two datasets validates Sparse RAG’s ability to achieve a balanced trade-off between high-quality generation and computational efficiency, proving its versatility and effectiveness for both shortand long-form content generation tasks. This innovative paradigm showcases the potential for improving LLM performance in various applications by optimizing context management and inference processes. 

9 

## **Limitation and Future Work** 

Sparse RAG requires additional tuning on top of the LLMs, which is not a plug-and-use approach. Under cases where per-context assessment label is missing, the prompts used for LLM rating may need to be adjusted for different use cases. These adjustments ensure that the model can still function effectively despite the lack of specific context labels. Furthermore, future research will explore SparseRAG in multimodal contexts, investigating how Sparse-RAG can handle and integrate information from multiple types of data to improve its performance and applicability across diverse scenarios. 

## **7 Acknowledge** 

We would like to thanks Yu-hui Chen, Yuqi Li, Qifei Wang, Zonglin Li, Zhong Meng, Alex Chen, Jingbin Wang, Abhanshu Sharma, Alec Go, Jesper Andersen, David Racz, DeLesley Hutchins, Luke Sernau, David Munday, and Ewa Dominowska for their suggestions. 

10 

## **References** 

- [1] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, et al. 2023. PaLM 2 technical report. _CoRR_ , abs/2305.10403. 

- [2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. _CoRR_ , abs/2310.11511. 

- [3] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022. Improving language models by retrieving from trillions of tokens. In _International conference on machine learning_ , pages 2206–2240. PMLR. 

- [4] Tom B Brown, Benjamin Mann, Nick Ryder, et al. 2020. Language models are few-shot learners. In _Advances in neural information processing systems_ , pages 1877–1901. 

- [5] Michiel De Jong, Yury Zemlyanskiy, Nicholas FitzGerald, Fei Sha, and William Cohen. 2021. Mention memory: incorporating textual knowledge into transformers through entity mention attention. _arXiv preprint arXiv:2110.06176_ . 

- [6] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. _Journal of Machine Learning Research_ , 23(120):1–39. 

- [7] Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ . 

- [8] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Retrieval augmented language model pre-training. In _Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event_ , volume 119 of _Proceedings of Machine Learning Research_ , pages 3929–3938. PMLR. 

- [9] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In _International conference on machine learning_ , pages 3929–3938. PMLR. 

- [10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ . 

- [11] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. _Trans. Mach. Learn. Res._ , 2022. 

- [12] Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. _arXiv preprint arXiv:2007.01282_ . 

- [13] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. _ACM Comput. Surv._ , 55(12):248:1–248:38. 

- [14] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 7969–7992. Association for Computational Linguistics. 

- [15] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin Jin. 2024. Ragcache: Efficient knowledge caching for retrieval-augmented generation. _arXiv preprint arXiv:2404.12457_ . 

11 

- [16] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2019. Generalization through memorization: Nearest neighbor language models. _arXiv preprint arXiv:1911.00172_ . 

- [17] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In _International Conference on Machine Learning_ , pages 19274–19286. PMLR. 

- [18] Mike Lewis, Marjan Ghazvininejad, Gargi Ghosh, Armen Aghajanyan, Sida Wang, and Luke Zettlemoyer. 2020. Pre-training via paraphrasing. _Advances in Neural Information Processing Systems_ , 33:18470–18481. 

- [19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrievalaugmented generation for knowledge-intensive nlp tasks. _Advances in Neural Information Processing Systems_ , 33:9459–9474. 

- [20] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ . 

- [21] Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. 2022. A survey on retrievalaugmented text generation. _CoRR_ , abs/2202.01110. 

- [22] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. 2023. Awq: Activation-aware weight quantization for llm compression and acceleration. _arXiv preprint arXiv:2306.00978_ . 

- [23] Hongyin Luo, Tianhua Zhang, Yung-Sung Chuang, Yuan Gong, Yoon Kim, Xixin Wu, Helen Meng, and James R. Glass. 2023. Search augmented instruction learning. In _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pages 3717–3729. Association for Computational Linguistics. 

- [24] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 9802–9822. Association for Computational Linguistics. 

- [25] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_ , pages 12076–12100. Association for Computational Linguistics. 

- [26] Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon Shashua, and Yoav Shoham. 2023. Generating benchmarks for factuality evaluation of language models. _CoRR_ , abs/2307.06908. 

- [27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In _NeurIPS_ . 

- [28] Nir Ratner, Yoav Levine, Yonatan Belinkov, Ori Ram, Inbal Magar, Omri Abend, Ehud Karpas, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2022. Parallel context windows for large language models. _arXiv preprint arXiv:2212.10947_ . 

12 

- [29] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ . 

- [30] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. _CoRR_ , abs/2302.04761. 

- [31] Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need. _arXiv preprint arXiv:1911.02150_ . 

- [32] Noam Shazeer and Mitchell Stern. 2018. Adafactor: Adaptive learning rates with sublinear memory cost. In _International Conference on Machine Learning_ , pages 4596–4604. PMLR. 

- [33] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 31210–31227. PMLR. 

- [34] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ . 

- [35] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971. 

- [36] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective retrieval augmented generation. _arXiv preprint arXiv:2401.15884_ . 

- [37] Dani Yogatama, Cyprien de Masson d’Autume, and Lingpeng Kong. 2021. Adaptive semiparametric language models. _Transactions of the Association for Computational Linguistics_ , 9:362–373. 

- [38] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2023. Making retrieval-augmented language models robust to irrelevant context. _CoRR_ , abs/2310.01558. 

- [39] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song in the AI ocean: A survey on hallucination in large language models. _CoRR_ , abs/2309.01219. 

- [40] Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, et al. 2021. Qmsum: A new benchmark for query-based multi-domain meeting summarization. _arXiv preprint arXiv:2104.05938_ . 

13 

## **A Prompts and Instructions** 

## **A.1 Prompts Used for LLMs** 

We share the prompt used for calling LLMs to get per context assessment in Table 9. 

Table 9: The zero-shot prompts for LLM labeling and critique. 

Round 1 prompt You are now doing a reading comprehension task. It is important that you be as thorough, detail-oriented, and accurate as possible in your response. You are given a question, a set of accepted answers, a document and its title. The document does not necessarily contain the right answer to the question. You should read the title and the document and then check if they provide one of the correct answers to the question. If the title and document together contain the correct answer to the question, output a score of 1.0, otherwise output a score of 0.0. question: <question> accepted answers: <answers> title: <title> document: <document> output: Round 2 prompt Your job is to correct another model’s performance on a reading comprehension task. The model was given a question, a set of accepted answers, a document and its title. The document and title do not necessarily contain the right answer. The model was instructed to output a score of 1.0 if the document contains the answer, and a score of 0.0 otherwise. You will be given the same information as the other model along with its output. You should read the title and document and then check if they provide one of the correct answers to the question. Then check if you agree with the previous model’s output. If you agree, output the same score unchanged. If you disagree, output the corrected score. Your output should be as accurate as possible. question: <question> accepted answers: <answers> title: <title> document: <document> previous model’s score: <score> output: 

## **A.2 Rater Guidelines** 

We share the instructions provided to the human labelers in Table 10. 

Table 10: Instructions for raters creating ground-truth relevance dataset. 

Human Rater Instructions Please read the question, the answer and the context. Please answer if the context can help answer the question. If it can, select 1. Otherwise select 0. 1: good 0: bad Please use the answers as a hint. However, do not use "is the answer in the context?" as a heuristic for making the decision. 

## **B Dataset Analysis** 

During the human-labeling process, several raters flagged documents and questions that were difficult to label. In total, 23 out of 500 documents were flagged and 15 out of 50 questions were flagged. 

14 

We observed trends in questions and contexts shared in Table 11 that raised concerns about whether and how a human would be able to assess the relevance of the context. These concerns extend to expectations of how well LLMs would be able to do that as well. 

Table 11: Overview of trends, datasets, and examples with associated comments. 

|**Trend**|**Dataset**|**Example question**|**Comments**|
|---|---|---|---|
|Questions<br>with time-<br>dependent<br>answers|NQ|who is the president<br>of usa right now|Depends on when the question is asked.|
||NQ|who is the current<br>director of the us mint|Depends on when the question is asked.|
||NQ|when is the next dead-<br>pool movie being re-<br>leased|Depends on when the question is asked.|
||NQ|total number of death<br>row inmates in the us|Fluctuates over time.|
||PopQA|What is Prague the<br>capital of?|The borders in this region and the name of<br>the country have changed several times in<br>the 20th century.|
||PopQA|What is Dennis Rod-<br>man’s occupation?|The accepted answers are "actor, actress,<br>actors, actresses". He was an actor later in<br>his career, but he rose to prominence as a<br>professional basketball player.|
|Missing<br>synonyms,<br>abbreviations,<br>and aliases,<br>combined<br>with unclear<br>granularity|NQ|in<br>which<br>regions<br>are most of Africa<br>petroleum and natural<br>gas found|"Region" can refer to different levels of<br>granularity (e.g. Sub-Saharan Africa vs.<br>Ethiopia), but the only accepted answer is<br>"Nigeria".|
||NQ|what type of car is a<br>jeep|The accepted answers are only "off-road<br>vehicles", "light utility vehicles", "sport<br>utility vehicles", but "SUV" is clearly a<br>correct answer as well.|
|Non-<br>exhaustive list<br>of answers|NQ|cast of law & order<br>special victim unit|The accepted answers include 16 cast<br>members, but the show went on for 25<br>seasons with many cast changes and guest<br>stars not included in the list.|
|Oddly phrased<br>question|NQ|right to property ac-<br>cording to the consti-<br>tution of India is a|The only correct answer is "constitutional<br>right", but that is included in the question.<br>It’s not clear what type of answer would be<br>appropriate here.|
|Overly<br>specifc<br>answers<br>expected|NQ|where does the story<br>the great gatsby take<br>place|The only accepted answer here is "Long<br>Island of 1922", but the place is Long Island<br>and the question does not ask about when<br>the story is set.|
|Question<br>refers to an<br>entity with a<br>common<br>name without<br>disambiguation|PopQA|What<br>genre<br>is<br>Frances?|There is a musician and a flm called<br>"Frances", and both of those could arguably<br>have a genre associated with them.|
||PopQA|Who was the pro-<br>ducer of Hurt?|The question is referring to a song<br>performed by Christina Aguilera but there<br>are many other songs, movies, and other<br>entities that share the name and also have a<br>producer.|
||PopQA|What is the capital of<br>Cherokee County?|There are many different Cherokee Coun-<br>ties in different states in the USA.|



15 

We explored several ways of filtering our human-labeled subsample of Natural Questions to determine how they impacted context assessment F-scores overall and for each auto-rater. We provide two additional filtered versions of the human-labeled RAG relevance dataset as alternatives. See Table 12 for statistics on the filtered datasets and Table 13 for the auto-rater F-scores for each. Both statistical filtering approaches (e.g. removing contexts with non-unanimous labels) and targeted filtering approaches (e.g. removing questions or contexts flagged by human raters) lead to some improvement in F-scores for relevance labels. 

Table 12: Summary of Dataset Evaluations 

|**Dataset**|**# Questions**|**# Docs**|**Total % Relevant**|**Average # Docs per Question**|**Std Dev # Docs per Question**|**Average % relevant docs per question**|**Std Dev % relevant docs per question**|
|---|---|---|---|---|---|---|---|
|Original 3-rater labels|50|500|0.30|10|0|0.30|0.23|
|Revisited (released)|50|500|0.31|10|0|0.31|0.26|
|Filter non-unanimous (released)|50|351|0.23|7.02|2.59|0.27|0.29|
|Filter fagged docs|48|467|0.31|9.729|0.57|0.31|0.27|
|Filter fagged questions|35|350|0.29|10|0|0.29|0.26|
|Filter fagged docs and question (released)|34|330|0.29|9.7|0.58|0.29|0.27|
|Filter 4-rater split|50|444|0.28|8.88|1.49|0.29|0.27|



Table 13: Evaluation of Labeling Methods w/ Filtering 

|Dataset Filters|# Docs|Rater Model|Critic Model|Average F1|
|---|---|---|---|---|
|||PALM2 XL|n/a|0.729|
|||PALM2 XL|PALM2 XL|0.781|
|Specialized rater<br>corrections|500|Gemini Ultra<br>Gemini Ultra|n/a<br>Gemini Ultra|0.761<br>0.704|
|||PALM2 XL|Gemini Ultra|0.728|
|||Gemini Ultra|PALM2 XL 340B|**0.821**|
|||PALM2 XL|n/a/|0.741|
|||PALM2 XL|PALM2 XL|0.811|
|Filter non-unanimous<br>docs|351|Gemini Ultra<br>Gemini Ultra|n/a<br>Gemini Ultra|0.792<br>0.739|
|||PALM2 XL|Gemini Ultra|0.763|
|||Gemini Ultra|PALM2 XL|**0.856**|
|||PALM2 XL|n/a|0.750|
|||PALM2 XL|PALM2 XL|0.797|
|Filter fagged docs<br>and questions|330|Gemini Ultra<br>Gemini Ultra|n/a<br>Gemini Ultra|0.782<br>0.741|
|||PALM2 XL|Gemini Ultra|0.753|
|||Gemini Ultra|PALM2 XL|**0.833**|



16 

