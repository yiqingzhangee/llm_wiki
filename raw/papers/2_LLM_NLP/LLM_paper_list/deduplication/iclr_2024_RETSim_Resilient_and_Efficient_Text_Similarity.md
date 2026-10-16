Preprint 

## RETSIM: RESILIENT AND EFFICIENT TEXT SIMILARITY 

**Marina Zhang[1] , Owen Vallis[1] , Aysegul Bumin[*2] , Tanay Vakharia[1] , Elie Bursztein[1]** Google[1] University of Florida[2] 

## ABSTRACT 

This paper introduces RETSim (Resilient and Efficient Text Similarity), a lightweight, multilingual deep learning model trained to produce robust metric embeddings for near-duplicate text retrieval, clustering, and dataset deduplication tasks. We demonstrate that RETSim is significantly more robust and accurate than MinHash and neural text embeddings, achieving new state-of-the-art performance on dataset deduplication, adversarial text retrieval benchmarks, and spam clustering tasks. We also introduce the W4NT3D benchmark (Wiki-40B 4dversarial Near-T3xt Dataset) for evaluating multilingual, near-duplicate text retrieval capabilities under adversarial settings. RETSim and the W4NT3D benchmark are open-sourced under the MIT License at https://github.com/google/unisim. 

## 1 INTRODUCTION 

Robust near-duplicate text detection is an essential component of many tasks, including retrieving documents, detecting plagiarism (Sun et al., 2013) and blocking adversarial spam campaigns (Ahmed et al., 2022). Users have come to expect that systems can return accurate results despite their queries exhibiting a 20% to 30% typo rate (Hagen et al., 2017). Furthermore, efficiently deduplicating text datasets is critical to training state-of-the-art large language models (Lee et al., 2022; Kandpal et al., 2022). 

For more than two decades, MinHash-based (Broder et al., 1998) locality-sensitive hashing (LSH) has been the most prevalent algorithm used for near-duplicate detection due to its simplicity, robustness, and speed. For example, the vast majority of dataset deduplication efforts still rely on MinHash (Lee et al., 2022; Kocetkov et al., 2022). However, like all LSH-based techniques, MinHash is not without downsides; chief among them being that it is very parameter-sensitive and requires heavy tuning. Additionally, MinHash lacks resilience to typos due to its reliance on n-grams, leading to poor performance on noisy data and a vulnerability to hash-busting attacks (Issac et al., 2014). 

On the other hand, deep learning models are the dominant way to perform vector-based semantic text retrieval (Muennighoff et al., 2022), but so far, no neural embedding has been able to consistently outperform MinHash for robust near-duplicate detection (Silcock et al., 2022). This is mostly due to the focus on improving semantic capabilities, which leads models to be too large to run extremely quickly and the use of sub-word level tokenization, which is not resilient to typos and adversarial attacks (Morris et al., 2020; Bursztein et al., 2023). 

To fill this gap, we introduce RETSim (Resilient and Efficient Text Similarity), a lightweight, multilingual deep learning model trained specifically to produce robust neural embeddings specialized for near-duplicate detection. By combining the state-of-the-art RETVec text vectorizer, a modern transformer block (Hua et al., 2022), a large typo-augmented training corpus, and a metric learning training regime, RETSim is able to achieve new state-of-the-art performance on near-duplicate detection benchmarks (Section 4.2), dataset deduplication tasks (Sections 4.3 and 5.1), and spam clustering applications (Section 5.2). 

Furthermore, while datasets and benchmarks exist for corpus deduplication and near-duplicate text retrieval, none of these have focused on systematically evaluating near-duplicate retrieval performance under the presence of typos, word manipulations, and sentence or paragraph-level modifica- 

> *This work was done during the author’s internship at Google. 

1 

Preprint 

tions. To address this need, we additionally introduce the _W4NT3D_ benchmark (Wiki-40B 4dversarial Near-T3xt Dataset) which enables the evaluation of algorithms on adversarial near-duplicate text retrieval in a multilingual setting. We report the performance of RETSim, MinHash, and popular neural embeddings such as Universal Sentence Encoder (Cer et al., 2018) and LaBSE (Feng et al., 2022) on this new benchmark in Section 4.2, highlighting uneven performance across languages and types of adversarial manipulations. The RETSim model and the W4NT3D benchmark are open-sourced at https://github.com/google/unisim under the MIT License. 

## 2 RELATED WORK 

**Near-Duplicate Detection** Identifying noisy near-duplicate documents in a large corpus is a fundamental task with a wide range of applications, such as detecting plagiarism, finding reproduced content in literature or news articles (Gyawali et al., 2020; Silcock et al., 2022), and deduplicating training datasets for language models. Previous research has shown that duplicates in training datasets lead to inefficient training (Lee et al., 2022) and privacy concerns for large language models (LLMs), where models memorize and regenerate duplicated training sequences at a much higher frequency (Kandpal et al., 2022). 

Unlike semantic text similarity, the task of identifying textual near-duplicates has been predominated by non-neural, n-gram-based algorithms such as MinHash (Broder et al., 1998), which is the most widely used technique for deduplicating large training corpuses (Kocetkov et al., 2022; Lee et al., 2022). MinHash is a technique for estimating the Jaccard similarity between two sets. Algorithms such as MinHash or SimHash (Charikar, 2002) can be combined with locality-sensitive hashing (LSH) techniques for fast, approximate nearest neighbor search and data clustering. This allows them to scale and deduplicate corpuses containing terabytes of data such as C4 (Lee et al., 2022) and The Stack (Kocetkov et al., 2022). However, n-gram or shingling-based techniques typically require texts to be parsed into a standardized form (e.g. by lower-casing or stripping punctuation), which makes them susceptible to typos and adversarial attacks and pose a challenge when attempting to differentiate between dissimilar documents and near-duplicate documents with adversarial augmentations. 

**Semantic Text Similarity** The task of computing semantic similarity between text is closely related to near-duplicate detection. Semantic text similarity refers to the assessment of the semantic relatedness of two pieces of text based on their meaning rather than their syntactic structure, as in the case of near-duplicate detection. Recently, transformer-based language models such as Universal Sentence Encoder (Yang et al., 2019), LaBSE (Feng et al., 2022) and LLM-based embeddings (Anil et al., 2023) which embed text into high-dimensional embedding vectors have been successfully used to retrieve semantically-related documents using cosine similarity. Modern text retrieval systems combine these embeddings with an approximate nearest neighbor (ANN) search algorithm to efficiently retrieve documents matching user queries. 

However, language models have been shown to be vulnerable to adversarial attacks and naturallyoccurring typos (Alzantot et al., 2018; Gao et al., 2018; Morris et al., 2020). Furthermore, language models are typically very large and costly to run even with hardware acceleration, which makes them unsuited for large-scale dataset deduplication or identifying near-duplicates in the presence of typos or adversarial text manipulations. 

**Metric Learning** Metric learning aims to learn an embedding space where similar items have a small distance between their embeddings and dissimilar items are further away. Many state-ofthe-art embeddings use metric learning for unsupervised training or fine-tuning including SentenceBERT (Reimers & Gurevych, 2019) and E5 (Wang et al., 2022). 

RETVec is a resilient, multilingual embedding and text vectorizer trained to be robust against various forms of character-level typos and adversarial attacks. We extend the RETVec training regime to full text documents for RETSim. We use Multi-Similarity Loss (Wang et al., 2019) for pair-based metric learning, where typo-laden and near-duplicate versions of texts are trained to be closer in the embedding space, while other texts are pushed further away. Multi-Similarity Loss is based on a general weighting framework for pair-based losses and achieves state-of-the-art performance, outperforming alternatives such as Triplet Loss (Schroff et al., 2015). 

2 

Preprint 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity_images/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity.pdf-0003-01.png)


Figure 1: RETSim model architecture diagram. RETSim works on arbitrary length text by splitting texts into chunks of 512 characters during its vectorization phase and encodes them using the RETVec character vectorizer. The RETSim model then embeds each chunk of text into 256-dim partial embeddings and combines them to produce the global embedding. 

## 3 RETSIM 

## 3.1 ARCHITECTURE 

The RETSim model is composed of three main components (as depicted in Figure 1): 

**The character-level vectorizer** splits the input text into chunks of 512 characters, then uses the RETVec chararcter encoder (Bursztein et al., 2023) to encode each chunk, resulting in a batch of (512 _,_ 24) dense inputs. The RETVec character vectorizer encodes each Unicode character as a compact 24-bit binary representation based on its integer codepoint value. This allows the vectorizer to encode all valid Unicode characters and support all languages. Furthermore, the character-level vectorizer has been shown to be more resilient against typos and adversarial attacks. 

**A small transformer model** is used to compute 256-dimension embeddings for each chunk of the input text. RETSimPartial-Dup uses these embeddings directly to finding documents that have matching chunks of text. Architecturally, the model consists of two Gated Attention Unit (GAU) blocks (Hua et al., 2022), followed by a Generalized-Mean pooling layer (Radenovi´c et al., 2018), a dense projection layer which projects the embedding into 256 dimensions, and an L2 normalization layer. The model has only 536k parameters, which is more than two orders of magnitude smaller than other neural embeddings (Table 1). L2-normalization allows the embeddings to be compared using cosine similarity. We discuss the impact of key architecture design choices in Section 6. Hyperparameter details are provided in Appendix A.1.1, and additional ablations results in Appendix A.5. 

**An embedding averaging module** is then used to combine partial text embeddings into a full-text embedding which is used for global near-duplicate matching (RETSimNear-Dup). Averaging chunked embeddings to produce a global embedding is a standard technique used by many models (Cer et al., 2018) to support infinite length inputs in a cost-efficient manner. We experimented with other aggregation techniques to produce more accurate global embeddings, including training a deepaveraging network (Iyyer et al., 2015), but this did not improve performance and resulted in higher computation cost. RETSimNear-Dup and RETSimPartial-Dup are computed in a single forward pass which makes it computationally efficient. We output both types of embeddings as they have different applications: RETSimNear-Dup is better-suited for full-text matching and retrieval (Section 4.2), while RETSimPartial-Dup is used to find partial text matches where the near-duplicate content appears only in part of the document (Section 4.3). 

## 3.2 MODEL TRAINING 

**Dataset** We use the multilingual C4 dataset (mC4) for raw text data and following (Xue et al., 2020), we use a language sampling exponent of _α_ = 0 _._ 3 to balance sampling between low and high-resource languages. We only use text containing at least 16 characters, and we randomly select between 1 and 8 sentences (roughly 512 characters) for each text chunk. For each example in the 

3 

Preprint 

training dataset, we generate 5 pairs of augmented examples. We apply three levels of augmentation to each example text chunk (in this order): sentence-level, word-level, and character-level. For each level, we randomly select the augmentation to be applied from the following categories: insertion, deletion, substitution, and transposition. We randomly apply between 0 _−_ 25% sentence-level augmentation and up to 30% combined character and word-level augmentation. Empirically, we found that increasing the percentage of augmentation beyond this point causes RETSim’s performance to degrade. The full list of augmentations used can be found in Appendix A.2. 

**Training Procedure** We train RETSim using Multi-Similarity Loss (Wang et al., 2019) with _α_ = 4, _β_ = 40, _λ_ = 0 _._ 5, and _ϵ_ = 0 _._ 1. We hypertuned these parameters and the results are shown in Appendix A.5. We train for 1 million steps with batch size = 1024. The similarity loss trains the model to embed augmented versions of the same text closer in the embedding space, while dissimilar texts are pushed further apart. We use the LAMB optimizer (You et al., 2019) with a max learning rate of 0 _._ 001 and cosine decay. Detailed training hyperparameters are reported in Appendix A.1.2. 

## 4 EVALUATION 

|**Model/Algorithm**|**Type**<br>**Embed./Hash Size**<br>**# Model Parameters**|
|---|---|
|LaBSE<br>Multilingual USE<br>Multilingual E5-Base<br>PaLM 2 (Gecko)|Neural<br>768<br>471M<br>Neural<br>512<br>69M<br>Neural<br>768<br>278M<br>Neural<br>768<br>?|
|SimHash<br>MinHash|Hashing<br>_b_bits<br>N/A<br>Hashing<br>_n_hashes<br>N/A|
|RETSim|Neural<br>256<br>536k|



Table 1: Embedding models and hashing algorithms benchmarked in the paper. 

## 4.1 MODELS AND ALGORITHMS EVALUATED 

We benchmark RETSim against four multilingual semantic text embeddings as well as popular n- gram based algorithms primarily used in near-duplicate text detection (Table 1). Our baseline text embeddings include Multilingual Universal Sentence Encoder (Yang et al., 2019), LaBSE (Feng et al., 2022), Multilingual E5 (Wang et al., 2022), and PaLM 2 Gecko Embeddings (Anil et al., 2023). All text embeddings are L2-normalized and compared using cosine similarity. We use exact search to index and retrieve nearest neighbors from our vector index for the experiments in Section 4. 

For non-neural near-duplicate detection and clustering algorithms, we selected the two most popular algorithms: MinHash (Broder et al., 1998) and SimHash (Charikar, 2002). For MinHash, we use Datasketch’s MinHashLSH library. Following the most common practices in the literature (Silcock et al., 2022), we use 10 hash functions for MinHash unless otherwise specified. We use word-level n-grams where we select the best value out of _n_ = _{_ 2 _,_ 3 _,_ 4 _, ...,_ 10 _}_ . For SimHash, we use 64bit SimHash and conduct shingling at the character level, where the shingle size is selected from _n_ = _{_ 2 _,_ 3 _,_ 4 _, ...,_ 10 _}_ . For the near-duplicate detection benchmarks (NEWS-COPY and CORE NearDuplicates datasets), we tune the optimal deduplication threshold (e.g. based on cosine similarity for neural-based embeddings and Jaccard similarity for MinHash). Detailed hyperparameter settings for RETSim and baseline algorithms used in the evaluation can be found in Appendix A.3. 

## 4.2 W4NT3D: WIKI-40B 4DVERSARIAL NEAR-T3XT DATASET EVALUATION 

**Dataset Description** The vast majority of text retrieval benchmarks are focused on evaluating semantic performance. To the best of our knowledge, there is no multilingual benchmark for systematically measuring adversarial robustness for near-duplicate text retrieval. In an attempt to fill in the gap, we create and publish the W4NT3D benchmark (Wiki-40B 4dversarial Near-T3xt Dataset), which contains around 400k pairs of syntactically similar texts to evaluate near-duplicate text retrieval in the presence of various forms of text manipulations and typos. 

W4NT3D is based on the Wiki-40B dataset (Guo et al., 2020). The dataset is split into query examples and target examples, where query examples are synthetically-modified near-duplicate versions 

4 

Preprint 

of a target example (e.g. with typos). For each of the 41 language splits in Wiki-40B, we randomly select 10,000 texts. The length of the target string is uniformly selected from between 16 and 8192 characters, in order to test performance on short and long text. To construct the query text corresponding to a target text, we randomly apply up to 25% word and character augmentations, and up to 25% sentence and paragraph augmentations. For each augmentation, we uniformly select from the [insert, delete, substitute, and swap] operations. We use Recall@ _k_ with _k_ = 1 as the main metric, following the setup commonly found in semantic text retrieval benchmarks. 

|**Model/Algorithm**|**Arabic**<br>**Chinese**<br>**English**<br>**German**<br>**French**<br>**Spanish**<br>**Japanese**<br>**Korean**<br>**Russian**<br>**Thai**|**Avg (41 Langs)**|
|---|---|---|
|LaBSE<br>Multilingual USE<br>Multilingual E5-Base<br>PaLM 2 (Gecko)|0.915<br>0.917<br>0.944<br>0.931<br>0.930<br>0.888<br>0.931<br>0.949<br>0.918<br>0.882<br>0.915<br>**0.986**<br>0.958<br>0.942<br>0.938<br>0.903<br>**0.990**<br>0.984<br>0.910<br>0.888<br>0.936<br>0.980<br>0.959<br>0.944<br>0.948<br>0.896<br>0.979<br>0.986<br>0.911<br>0.921<br>0.497<br>0.623<br>0.961<br>0.932<br>0.934<br>0.911<br>0.578<br>0.701<br>0.851<br>0.571|0.921<br>0.912<br>0.937<br>0.823|
|SimHash<br>0.558<br>0.276<br>0.591<br>0.561<br>0.519<br>0.513<br>0.465<br>0.593<br>0.554<br>0.669<br>0.550<br>MinHash<br>0.633<br>0.172<br>0.591<br>0.558<br>0.556<br>0.575<br>0.223<br>0.814<br>0.523<br>0.416<br>0.538|||
|RETSimPartial-Dup<br>RETSimNear-Dup|0.928<br>0.946<br>0.954<br>0.949<br>0.947<br>0.938<br>0.963<br>0.971<br>0.946<br>0.941<br>**0.971**<br>0.971<br>**0.987**<br>**0.978**<br>**0.983**<br>**0.976**<br>0.986<br>**0.991**<br>**0.970**<br>**0.946**|0.949<br>**0.977**|



Table 2: Per-language retrieval performance for various embedding models and algorithms on the W4NT3D benchmark. Results on selected languages are reported alongside the average Recall@1 for all 41 languages. Full results for all languages are reported in Appendix A.4. 

**Multilingual Performance** Overall, RETSimNear-Dup achieves an average Recall@1 of 0.977 across all 41 languages on the W4NT3D benchmark (Table 2). RETSimPartial-Dup is second best with a Recall@1 of 0.949 and Multilingual E5, the best-performing baseline, is third with an average Recall@1 of 0.932. We expect that RETSimNear-Dup outperforms RETSimPartial-Dup because the W4NT3D benchmark requires an algorithm to not just find near-duplicates, but to find the most similar text. RETSimPartial-Dup optimizes for finding the most similar chunk of text in the corpus, which is not always the most similar text overall. Similarly, we hypothesize that MinHash and SimHash perform poorly on the W4NT3D benchmark due to their lack of ability to distinguish which is the most similar text among the near-duplicates detected, and embedding-based models and cosine similarity offer a more fine-grained measure of similarity. 

RETSimNear-Dup outperforms baseline algorithms on all languages except for Chinese and Japanese. For these languages, we theorize that semantic embeddings may have the slight edge in performance because their significantly larger model sizes (more than 100x larger than RETSim, as shown in Table 1) allow them to have a better representation on languages with large character sets. Furthermore, the sub-word level tokenizers used in the baseline embeddings often treat each character in Chinese or Japanese as individual tokens, which could offer higher resilience to typos. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity_images/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity.pdf-0005-06.png)


**----- Start of picture text -----**<br>
1.00 LaBSE<br>Multilingual USE<br>0.75 Multilingual E5-Base<br>PaLM 2 (Gecko)<br>0.50 SimHash<br>MinHash<br>0.25 RETSim (Partial-Dup)<br>RETSim (Near-Dup)<br>0.00<br>Paragraph Sentence Word Character<br>Augmentation Level<br>Recall@1<br>**----- End of picture text -----**<br>


Figure 2: Recall@1 performance on the W4NT3D benchmark, broken down by augmentation type. Results are averaged across all 41 language splits in W4NT3D. 

**Adversarial Resilience** Delving deeper into the impact of various types of text manipulation reveals that RETSimNear-Dup and RETSimPartial-Dup perform almost equally well regardless of the type of augmentation applied (Figure 2). Semantic text embeddings perform well on paragraph, sentence and word-level manipulations, but as expected, exhibit significantly weaker performance towards character-level typos. MinHash and SimHash struggle more with word-level augmentations than deep-learning based embeddings and collapse when character-level typos are introduced. We at- 

5 

Preprint 

tribute RETSim’s resilience to adversarial manipulations to the RETVec character encoder as well as using deep metric learning to train robust embeddings. 

Figure 4 reports the Recall@1 performance of the algorithms as the amount of augmentation increases. All algorithms perform perfectly when no augmentation is applied (exact matching), but as the percentage of augmentation increases, n-gram based approaches exhibit a steep drop in performance. Semantic text embeddings are able to sustain a larger degree of augmentation before their retrieval capabilities start to degrade (over 20%). RETSimNear-Dup is the most robust algorithm, with a noticeable drop in performance only after around 40% augmentation. This makes RETSim the most effective approach at clustering and deduplicating text under adversarial settings. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity_images/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity.pdf-0006-03.png)


**----- Start of picture text -----**<br>
1.0 1.0 LaBSE<br>Multilingual USE<br>0.8 Multilingual E5-Base<br>0.8 PaLM 2 (Gecko)<br>SimHash<br>0.6<br>0.6 MinHash<br>RETSim (Partial-Dup)<br>0.4 RETSim (Near-Dup)<br>0.4<br>0% 10% 20% 30% 40% 50%<br>Max Target Text Length<br>Text Augmentation Amount (%)<br>16 32 64 128 256 512 1024 2048 4096 8192<br>Recall@1<br>Recall@1<br>**----- End of picture text -----**<br>


Figure 3: Recall@1 performances on the W4NT3D benchmark (English only) for varying max target lengths. 

Figure 4: Recall@1 performances on the W4NT3D benchmark (English only) as the amount of augmentation applied to the query text increases. 

**Text Length Impact on Performance** Figure 3 reports the Recall@1 performance of RETSim and baseline algorithms as the length of the query and target text varies. We see that RETSimNear-Dup and RETSimPartial-Dup outperforms all other methods on short texts with fewer than 128 characters. As the text length increases beyond 512 characters, RETSimNear-Dup remains close to perfect while RETSimPartial-Dup’s performance degrades since it splits the text into multiple embeddings and finds the nearest matching _chunk_ of text. MinHash and SimHash also perform poorly on short text lengths and start to degrade on longer texts. For neural-based embeddings, we observe a slight drop in performance on longer texts for all models except RETSimNear-Dup and Multilingual USE, the only two embeddings that can handle arbitrary length inputs. 

## 4.3 REAL-WORLD NEAR-DUPLICATE DETECTION EVALUATION 

**Setup** We benchmark RETSim’s ability to identify near-duplicate content on real-world datasets from the literature. The NEWS-COPY Deduplication dataset (Silcock et al., 2022) contains 27,210 historical news articles with 122,876 positive duplicate pairs. The dataset consists of noisy nearduplicates due to factors like OCR errors, plagiarism, and news aggregation. We also evaluate the algorithms on the CORE Near-Duplicates dataset (Gyawali et al., 2020), which consists of 100k scholarly articles with 25k exact duplicates, 25k near-duplicates, and 50k non-duplicates. Nearduplicates in this dataset arise from article revisions, versioning and metadata differences, and human typos. A key difference between these two benchmarks and the W4NT3D benchmark is that these two benchmarks are focused on detecting and clustering near-duplicate text, rather than robust text retrieval based on syntactic similarity. For both benchmarks, we follow the experimental setup provided in the papers and report Adjusted Rand Index (ARI) for the NEWS-COPY dataset and report precision/recall/F1 scores on the CORE Near-Duplicates dataset. 

**Results** On the NEWS-COPY dataset, RETSimPartial-Dup outperforms all other approaches by a significant margin (4.8% ARI compared to our best MinHash result), as reported in Table 3. In the dataset, there are many near-duplicate pairs where one text is significantly longer than the other, so it is expected that RETSimPartial-Dup, which can find matching text chunks in documents, is more suited for the task and outperforms RETSimNear-Dup. Bucketing the near-duplicate detection rate of each algorithm by the length ratio between positive pairs (Figure 5), we observe that RETSimPartial-Dup outperforms MinHash regardless of the length ratio, but MinHash surpasses RETSimNear-Dup performance when one text is above roughly 1.5x the length of the other text in a near-duplicate pair. 

6 

Preprint 

|**Model/Algorithm**||**ARI**|
|---|---|---|
|Multilingual USE<br>Multilingual E5-Base<br>S-BERT*||0.730<br>0.742<br>0.700|
|SimHash<br>MinHash*<br>MinHash (Ours)||0.695<br>0.737<br>0.783|
||||
|RETSimPartial-Dup<br>RETSimNear-Dup||**0.831**<br>0.704|




![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity_images/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity.pdf-0007-02.png)


**----- Start of picture text -----**<br>
1.0<br>0.8<br>0.5<br>0.3<br>0.0<br>1 2 3 4 5 6 7 8 9 10<br>Length Ratio between Near-Duplicate Text Pair<br>RETSim (Partial-Dup) RETSim (Near-Dup) MinHash<br>Near-Duplicate Detection Rate<br>**----- End of picture text -----**<br>


Table 3: Performance comparison on the NEWS-COPY dataset. Adjusted Rand Index (ARI) values are reported. * denotes results from Silcock et al. (2022). 

Figure 5: Near-duplicate detection rate of RETSim vs MinHash for different length ratios of positive pairs. X-axis is the length of longer divided by shorter text, rounded to the nearest integer. 

Additionally, we noticed that the labels in the dataset were occasionally noisy, as a substantial portion of the RETSim false positives appear to be near-duplicates upon inspection (Appendix A.6). 

On the CORE Near-Duplicates dataset (Table 4), where documents (article title + abstract) are roughly the same size, RETSimPartial-Dup and RETSimNear-Dup performance is roughly equivalent. Both methods outperform the baselines in terms of macro F1 score and accuracy. We use MinHash + LSH with 256 hash functions for computational efficiency, as recommended by the datasketch library[1] for better accuracy than the default setting. Deduplication thresholds and detailed hyperparameter settings for the algorithms on both near-duplication datasets can be found in Appendix A.3. 

|**Model / Algorithm**|**Precision**<br>**Duplicates**<br>**Recall**<br>**Duplicates**<br>**Precision**<br>**Non-Duplicates**<br>**Recall**<br>**Non-Duplicates**<br>**Macro F1**<br>**Accuracy**|
|---|---|
|Exact Title Matching*|0.830<br>0.500<br>0.709<br>**0.992**<br>0.757<br>0.746|
|LaBSE<br>Multilingual USE<br>Multilingual E5-Base|0.937<br>0.923<br>0.930<br>0.943<br>0.933<br>0.919<br>0.917<br>0.907<br>0.918<br>0.927<br>0.917<br>0.909<br>0.931<br>0.908<br>0.919<br>0.939<br>0.924<br>0.920|
|MinHash + LSH|0.929<br>0.902<br>0.915<br>0.938<br>0.921<br>0.918|
|RETSimPartial-Dup<br>RETSimNear-Dup|**0.945**<br>**0.941**<br>**0.945**<br>0.949<br>**0.945**<br>**0.928**<br>0.928<br>0.937<br>0.942<br>0.934<br>0.935<br>0.926|



Table 4: Evaluation results on the CORE Near-Duplicates dataset. Precision/recall/macro F1 and accuracy numbers are reported. * denotes results from Gyawali et al. (2020). 

## 5 APPLICATIONS 

5.1 TRAINING DATASET DEDUPLICATION 

|**Model/Algorithm**|**% train examples**<br>**with dup in train**|**% valid examples**<br>**with dup in train**|
|---|---|---|
|MinHash + LSH|0.47%|0.46%|
|Exact Substring*|2.76%|0.52%|
|RETSimNear-Dup|3.17%|0.59%|
|RETSimPartial-Dup|12.77%|2.66%|



Table 5: Deduplication rate on Wiki-40B (English). * denotes results from Lee et al. (2022). 

**Setup** We evaluate RETSim’s ability to deduplicate text training datasets by deduplicating the English split of Wiki-40B (Guo et al., 2020). We conservatively set the cosine similarity deduplication threshold to 0.1 for RETSimNear-Dup and 0.15 for RETSimPartial-Dup to limit the amount of false positives, based on the optimal thresholds found in the evaluation (Appendix A.3). We use USearch’s default vector index for approximate nearest neighbor search (Vardanian, 2023). We compare 

> 1datasketch: Big Data Looks Small. https://github.com/ekzhu/datasketch. 

7 

Preprint 

|**Model/Algorithm**|**Accelerator**<br>**Batch Size**<br>**Embedding / Hashing**<br>**time (sec)**<br>**examples/sec**|
|---|---|
|MinHash + LSH<br>RETSim<br>RETSim<br>RETSim|CPU AMD 7950 32 cores<br>-<br>234<br>12544<br>Onnx CPU AMD 7950 32 cores<br>256<br>10839<br>270<br>TensorFlow GPU RTX 4090<br>4096<br>720<br>4062<br>TensorFlow GPU NVIDIA H100<br>16384<br>363<br>8069|



Table 6: Embedding/hashing speed of RETSim vs MinHash + LSH on the Wiki-40B dataset. 

against MinHash + LSH, where we set the number of hash functions to be 256 following Kocetkov et al. (2022) and use a Jaccard similarity threshold of 0.8 for deduplication (Lee et al., 2022). 

**Results** Overall, as reported in Table 5, RETSimNear-Dup finds slightly more duplicates in the Wiki40B training and validation splits. This is in-line with our deduplication results (Section 4.3) where RETSimNear-Dup outperforms other algorithms. On the other hand, RETSimPartial-Dup finds significantly more matches than the exact substring matching algorithm used in the previous study (Lee et al., 2022), showcasing the usefulness of performing both near-duplicate and partial-duplicate matching at once. This larger-than-expected number of partial matches indicate that machine learning practitioners should take extra care to deduplicate Wikipedia at the chunk level to avoid feeding duplicate text to their models. 

In terms of embedding speed (Table 6), RETSim is significantly slower than MinHash + LSH on CPU (46x slower), competitive when using a desktop GPU such as the RTX 4090 (3x slower) and almost on-par when using a high-end GPU like the NVIDIA H100 (1.5x slower). Our current code is written in Python and not fully optimized, so we expect this performance gap to significantly shrink as we optimize our implementation. Although RETSim is slower than MinHash, RETSim is significantly smaller and faster than other text embedding models, and closes the performance gap between neural and non-neural based methods for near-duplicate text detection and dataset deduplication. Both RETSimNear-Dup and RETSimPartial-Dup are returned at the same time so they have the same embedding speed. Indexing and retrieval times will depend on the vector index and search algorithm used. For longer documents, RETSimPartial-Dup will produce more embeddings than RETSimNear-Dup, so RETSimPartial-Dup offers a tradeoff between finer-grained matching versus indexing/retrieval speed, which will depend on the specific vector search algorithm and dataset used. 

## 5.2 IN THE WILD: SPAM EMAIL CLUSTERING 

In this section, we showcase RETSim’s real-world performance on clustering near-duplicate text which has been heavily manipulated by adversarial attacks by performing an evaluation on spam campaigns. Spam constitutes a strong proving ground for near-duplicate clustering algorithms as spammers employ adversarial augmentation techniques in an attempt to evade detection. Such augmentations typically include appending or prepending unrelated text, interleaving random words and different languages, intentionally introducing typos, abusing extended character sets such as emojis and homoglyphs, and more. These techniques are collectively referred to as hash-busting. 

**Setup** The dataset consists of 5,252 spam emails from 196 spam campaigns, donated by Gmail users who flagged them when they reached their inboxes. Each example contains the email subject concatenated with the message content. The emails were misclassified by a spam classifier due to their effective adversarial text manipulation techniques, which makes them a challenging test set for clustering evaluations. Some examples of hash-busting attacks and adversarial manipulations we observe include the use of homoglpyphs, uncommon Unicode character sets, invisible characters, and padding with random words from different languages. To get the ground truth campaign clusters, emails were manually reviewed and assigned to a specific spam campaign based on similarity by human reviewers. We use agglomerative clustering to cluster spam emails, and report homogeneity, completeness, V-Measure, and Adjusted Rand Index (ARI) metrics. 

**Results** Overall, we observed that RETSim is significantly better at clustering near-duplicates with adversarial manipulations, outperforming both SimHash and USE across all metrics considered (Table 7). In particular, we observed that RETSim outperforms USE by 4.6% on the V-Measure score which is our main metric. The results reported in this section are in-line with what we observe since we deployed RETSim as our main near-duplicate detection algorithm in December 2022. 

8 

Preprint 

|**Model / Algorithm**|**Homogeneity**<br>**Completeness**<br>**V-Measure**<br>**ARI**|
|---|---|
|USE<br>SimHash + LSH<br>RETSimNear-Dup|0.856<br>0.955<br>0.903<br>0.6<br>0.867<br>0.876<br>0.871<br>0.571<br>**0.937**<br>**0.963**<br>**0.949**<br>**0.747**|



Table 7: Performance on clustering adversarial spam campaigns in practice. 

## 6 ABLATION STUDIES 

**Setup** In this section, we summarize the key ablation studies we performed when designing RETSim. All the models used in this section are trained using the setup detailed in Appendix A.1.2, except we only train them for 100k steps to reduce computational costs. We evaluate RETSimNear-Dup’s performance for each model on a subset of the W4NT3D benchmark, where we randomly select 1000 examples from each of the 41 language splits and use Recall@1 as reported metric. 

|**Block Type**<br>**Recall@1**<br>RETVec MLP<br>0.975<br>ConvNeXt<br>0.978<br>BERT<br>0.973<br>T5<br>0.980<br>**_*GAU_**<br>**0.986**|**Chunk Size**<br>**Recall@1**<br>128<br>0.979<br>256<br>0.984<br>**_*512_**<br>**0.986**<br>1024<br>0.983<br>2048<br>0.978|**Embed. Dim**<br>**Recall@1**|
|---|---|---|
|||64<br>0.969<br>128<br>0.980<br>**_*256_**<br>**0.986**<br>512<br>0.986<br>768<br>0.986|



Table 8: RETSim ablation study results on architecture block type (left), text chunk size (middle), and embedding dimension (right). ***** _**Bold**_ denotes the value selected for the final RETSim model. 

**Results** Table 8 contains RETSim ablation study results on max text chunk size, architecture block type, and embedding size. The most important architectural decision was to decide the optimal text chunk size and finding the right balance between having the smallest size possible to maximize RETSimPartial-Dup efficiency while ensuring RETSimNear-Dup full-text embeddings can work effectively on full documents. We find that chunks of 512 characters offer the best performance. 

We also tested various model architectures and transformer blocks to find the best balance between efficiency and performance. We find that the more modern GAU block (Hua et al., 2022) outperforms the vanilla BERT transformer block (Devlin et al., 2019) and the T5 block (Xue et al., 2020). We also tried modern CNN architectures such as ConvNeXt (Liu et al., 2022) and the MLP architecture proposed in RETVec (Bursztein et al., 2023), but both were worse than GAU block performance. Last but not least, we found that increasing the embedding size past 256 dimensions does not yield any meaningful improvements for RETSimNear-Dup. Accordingly, we opted to use a 256-dimension embedding for space-efficiency and to maximize indexing and query speed. Additional ablation studies for other hyperparameters can be found in Appendix A.5. 

## 7 FUTURE WORK 

RETSim’s novel training regime, which combines metric learning and data augmentation, has many other potential applications that we plan to explore in future work. For example, it could be adapted or extended to train robust semantic embeddings or image similarity embeddings. Additionally, we expect that as general models become bigger and more expensive to run in the future, smaller, specialized models such as RETSim will emerge as an efficient alternative for a wide range of tasks. 

## 8 CONCLUSION 

In this paper, we introduced RETSim, a novel, multilingual text embedding which achieves stateof-the-art performance on near-duplicate text detection, dataset deduplication, and syntactic text similarity benchmarks. RETSim is significantly faster than previous neural-based text embeddings and more robust than n-gram based algorithms, which makes it suitable for large-scale text retrieval and dataset deduplication, especially in adversarial settings such as spam detection. Furthermore, we introduced the W4NT3D benchmark, the first multilingual dataset designed to measure the adversarial robustness of near-duplicate text detection algorithms. We open-source both RETSim and the W4NT3D benchmark under the MIT License. 

9 

Preprint 

## REFERENCES 

- Naeem Ahmed, Rashid Amin, Hamza Aldabbas, Deepika Koundal, Bader Alouffi, and Tariq Shah. Machine Learning Techniques for Spam Detection in Email and IoT Platforms: Analysis and Research Challenges. Security and Communication Networks, 2022:1–19, February 2022. ISSN 1939-0122, 1939-0114. doi: 10.1155/2022/1862888. URL https://www.hindawi.com/ journals/scn/2022/1862888/. 

- Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. Generating Natural Language Adversarial Examples, September 2018. URL http: //arxiv.org/abs/1804.07998. arXiv:1804.07998 [cs]. 

- Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Cl´ement Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark D´ıaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. PaLM 2 Technical Report, September 2023. URL http://arxiv.org/abs/2305.10403. arXiv:2305.10403 [cs]. 

- Andrei Z. Broder, Moses Charikar, Alan M. Frieze, and Michael Mitzenmacher. Min-wise independent permutations. In Proceedings of the thirtieth annual ACM symposium on Theory of computing, pp. 327–336, 1998. 

- Eli Bursztein, Marina Zhang, Owen vallis, Xinyu Jia, and Alexey Kurakin. RetVec: Resilient and Efficient Text Vectorizer. 2023. 

- Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, and Chris Tar. Universal sentence encoder. arXiv preprint arXiv:1803.11175, 2018. 

- Moses S. Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing, pp. 380–388, 2002. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, May 2019. URL http://arxiv. org/abs/1810.04805. arXiv:1810.04805 [cs]. 

- Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. Languageagnostic BERT Sentence Embedding, March 2022. URL http://arxiv.org/abs/2007. 01852. arXiv:2007.01852 [cs]. 

- Ji Gao, Jack Lanchantin, Mary Lou Soffa, and Yanjun Qi. Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers, May 2018. URL http://arxiv.org/abs/ 1801.04354. arXiv:1801.04354 [cs]. 

10 

Preprint 

- Mandy Guo, Zihang Dai, Denny Vrandeˇci´c, and Rami Al-Rfou. Wiki-40b: Multilingual language model dataset. In Proceedings of the 12th Language Resources and Evaluation Conference, pp. 2440–2452, 2020. 

- Bikash Gyawali, Lucas Anastasiou, and Petr Knoth. Deduplication of Scholarly Documents using Locality Sensitive Hashing and Word Embeddings. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pp. 901–910, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https://aclanthology. org/2020.lrec-1.113. 

- Matthias Hagen, Martin Potthast, Marcel Gohsen, Anja Rathgeber, and Benno Stein. A largescale query spelling correction corpus. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1261–1264, 2017. 

- Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc Le. Transformer quality in linear time. In International Conference on Machine Learning, pp. 9099–9117. PMLR, 2022. 

- B. Issac, R. Chiong, and S. M. Jacob. Analysis of phishing attacks and countermeasures, 2014. 

- Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, and Hal Daum´e Iii. Deep Unordered Composition Rivals Syntactic Methods for Text Classification. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1681–1691, Beijing, China, 2015. Association for Computational Linguistics. doi: 10.3115/v1/P15-1162. URL http://aclweb.org/anthology/P15-1162. 

- Nikhil Kandpal, Eric Wallace, and Colin Raffel. Deduplicating Training Data Mitigates Privacy Risks in Language Models. In Proceedings of the 39th International Conference on Machine Learning, pp. 10697–10707. PMLR, June 2022. URL https://proceedings. mlr.press/v162/kandpal22a.html. ISSN: 2640-3498. 

- Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Mu˜noz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The Stack: 3 TB of permissively licensed source code, November 2022. URL http://arxiv.org/abs/2211.15533. arXiv:2211.15533 [cs]. 

- Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris CallisonBurch, and Nicholas Carlini. Deduplicating Training Data Makes Language Models Better, March 2022. URL http://arxiv.org/abs/2107.06499. arXiv:2107.06499 [cs]. 

- Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11976–11986, 2022. 

- John X. Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP, October 2020. URL http://arxiv.org/abs/2005.05909. arXiv:2005.05909 [cs]. 

- Niklas Muennighoff, Nouamane Tazi, Lo¨ıc Magne, and Nils Reimers. MTEB: Massive Text Embedding Benchmark, October 2022. URL https://arxiv.org/abs/2210.07316v1. 

- Filip Radenovi´c, Giorgos Tolias, and Ondˇrej Chum. Fine-tuning CNN image retrieval with no human annotation. IEEE transactions on pattern analysis and machine intelligence, 41(7):1655– 1668, 2018. Publisher: IEEE. 

- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. arXiv preprint arXiv:1908.10084, 2019. 

- Florian Schroff, Dmitry Kalenichenko, and James Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 815–823, June 2015. doi: 10.1109/CVPR.2015.7298682. URL http: //arxiv.org/abs/1503.03832. arXiv:1503.03832 [cs]. 

11 

Preprint 

- Emily Silcock, Luca D’Amico-Wong, Jinglin Yang, and Melissa Dell. Noise-Robust De-Duplication at Scale, October 2022. URL http://arxiv.org/abs/2210.04261. arXiv:2210.04261 [cs]. 

- Yifang Sun, Jianbin Qin, and Wei Wang. Near Duplicate Text Detection Using Frequency-Biased Signatures. In David Hutchison, Takeo Kanade, Josef Kittler, Jon M. Kleinberg, Friedemann Mattern, John C. Mitchell, Moni Naor, Oscar Nierstrasz, C. Pandu Rangan, Bernhard Steffen, Madhu Sudan, Demetri Terzopoulos, Doug Tygar, Moshe Y. Vardi, Gerhard Weikum, Xuemin Lin, Yannis Manolopoulos, Divesh Srivastava, and Guangyan Huang (eds.), Web Information Systems Engineering – WISE 2013, volume 8180, pp. 277–291. Springer Berlin Heidelberg, Berlin, Heidelberg, 2013. ISBN 978-3-642-41229-5 978-3-642-41230-1. doi: 10.1007/978-3-642-41230-1 24. URL http://link.springer.com/10.1007/978-3-642-41230-1_24. Series Title: Lecture Notes in Computer Science. 

- Ash Vardanian. USearch by Unum Cloud, October 2023. URL https://github.com/ unum-cloud/usearch. 

- Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text Embeddings by Weakly-Supervised Contrastive Pre-training, December 2022. URL http://arxiv.org/abs/2212.03533. arXiv:2212.03533 [cs]. 

- Xun Wang, Xintong Han, Weilin Huang, Dengke Dong, and Matthew R. Scott. Multi-similarity loss with general pair weighting for deep metric learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5022–5030, 2019. 

- Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934, 2020. 

- Yinfei Yang, Daniel Cer, Amin Ahmad, Mandy Guo, Jax Law, Noah Constant, Gustavo Hernandez Abrego, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. Multilingual Universal Sentence Encoder for Semantic Retrieval, July 2019. URL http://arxiv.org/ abs/1907.04307. arXiv:1907.04307 [cs]. 

- Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962, 2019. 

12 

Preprint 

## A APPENDIX 

## A.1 RETSIM DETAILS 

## A.1.1 RETSIM MODEL HYPERPARAMETERS 

The full list of RETSim model hyperparameters can be found in Table 9. 

|**Hyperparameter**|**Value**|
|---|---|
|||
|Max input length (per chunk)<br>Block type<br># blocks<br>Hidden dim<br>Expansion rate<br>Activation function<br>Attention activation function<br>Absolute positional encoding<br>Relative positional encoding<br>Norm type<br>Pooling type<br>Dropout rate<br>Embedding dim<br># Parameters|512<br>GAU<br>2<br>256<br>1<br>Swish<br>relu2<br>ScaledSin<br>RoPE<br>ScaleNorm<br>GeM(_p_= 3)<br>0<br>256<br>536k|



Table 9: Detailed RETSim model hyperparameters. 

## A.1.2 RETSIM TRAINING HYPERPARAMETERS 

Table 10 details the hyperparameters settings for training configuration, loss, and optimizer used to train the RETSim model. 

|**Hyperparameter**|**Value**|
|---|---|
|||
|Batch size<br>Train steps<br>LAMB_ϵ_<br>LAMB_β_1<br>LAMB_β_2<br>Max learning rate<br>End learning rate<br>Learning rate decay<br>Weight decay|1024<br>1 million<br>1e-6<br>0.9<br>0.999<br>0.001<br>0<br>Cosine<br>0|



Table 10: RETSim detailed training hyperparameters. 

## A.2 TRAINING DATASET DETAILS 

Below, we provide the full list of augmentations used to generate augmented text for the RETSim training dataset, as described in Section 3.2. 

## SENTENCE-LEVEL AUGMENTATIONS 

- Deletion: 

   - Random sentence deletion 

   - Random sentence truncation 

- Insertion: 

13 

Preprint 

   - Random prefix sentence 

   - Random suffix sentence 

   - Random sentence insertion 

   - Repeat sentence 

- Substitution: 

   - Lowercase/uppercase sentence 

   - Random sentence substitution 

- Transposition: 

   - Neighboring Swap 

## WORD-LEVEL AUGMENTATIONS 

- Deletion: 

   - Random word deletion 

- Insertion: 

   - Random word insertion 

   - Random word insertion per language 

- Substitution: 

   - 3-gram frequency based word substitution 

   - Random word substitution 

   - Random word substitution per language 

   - Repeat word 

- Transposition: 

   - Neighboring Swap 

## CHARACTER-LEVEL AUGMENTATIONS 

- Deletion: 

   - Random character deletion 

- Substitution: 

   - Case substitution 

   - n-gram based substitution for _n_ = 3 _,_ 4 _,_ 5 

   - QWERTY keyboard typo substitution 

   - Homoglyphs substitution 

   - Random ASCII substitution 

   - Random character from language alphabet substitution 

   - Random punctuation substitution 

   - Random Unicode character substitution 

- Insertion: 

   - Character repetition 

   - n-grams based insertion for _n_ = 3 _,_ 4 _,_ 5 

   - Random character from language alphabet insertion 

   - Random punctuation insertion 

   - Random Unicode character insertion 

- Transposition: 

   - Neighboring swap 

## A.3 DETAILED EVALUATION HYPERPARAMETERS 

Figures 6 and 7 contain information on deduplication thresholds values and hyperparameter settings for each algorithm benchmarked on the NEWS-COPY and CORE deduplication datasets. 

14 

Preprint 

|**Model / Algorithm**|**Threshold Type**|**Threshold Value**|**Hyperparameters**|
|---|---|---|---|
|Multilingual USE|Cosine Similarity|0.96|-|
|Multilingual E5-Base|Cosine Similarity|0.88|-|
|SimHash|Hamming Distance|10|64 bits, 5-grams (character-level)|
|MinHash (Ours)|Jaccard Similarity|0.6|10 hash functions, 2-grams (word-level)|
|RETSimNear-Dup|Cosine Similarity|0.89|-|
|RETSimPartial-Dup|Cosine Similarity|0.84|-|



Figure 6: Hyperparameter settings for NEWS-COPY dataset evaluation in Section 4.3. 

|**Model / Algorithm**|**Threshold Type**|**Threshold Value**|**Hyperparameters**|
|---|---|---|---|
|LaBSE|Cosine Similarity|0.88|-|
|Multilingual USE|Cosine Similarity|0.97|-|
|Multilingual E5-Base|Cosine Similarity|0.87|-|
|SimHash + LSH|Hamming Distance|6|64 bits, 3-grams (character-level)|
|MinHash + LSH|Jaccard Similarity|0.5|256 hash functions, 3-grams (word-level)|
|RETSimNear-Dup|Cosine Similarity|0.86|-|
|RETSimPartial-Dup|Cosine Similarity|0.82|-|



Figure 7: Hyperparameter settings for CORE Near-Duplicates dataset evaluation in Section 4.3. 

## A.3.1 DEDUPLICATION THRESHOLD IMPACT 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity_images/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity.pdf-0015-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity_images/iclr_2024_RETSim_Resilient_and_Efficient_Text_Similarity.pdf-0015-07.png)


Figure 8: Precision/Recall/F1 scores for different cosine distance deduplication thresholds for RETSimNear-Dup (left) and RETSimPartial-Dup (right) on the NEWS-COPY dataset. 

## A.4 DETAILED W4NT3D BENCHMARK RESULTS 

Tables 11 and 12 show detailed performance results for RETSim and all baseline algorithms for every language split in the W4NT3D benchmark. 

15 

Preprint 

|**Model / Algorithm**<br>**ar**<br>**bg**<br>**ca**<br>**cs**<br>**da**<br>**de**<br>**el**<br>**en**<br>**es**<br>**et**<br>**fa**<br>**f**<br>**fr**<br>**he**<br>**hi**<br>**hr**<br>**hu**<br>**id**<br>**it**<br>**ja**<br>**ko**|LaBSE<br>0.915<br>0.915<br>0.897<br>0.912<br>0.938<br>0.931<br>0.918<br>0.944<br>0.888<br>0.923<br>0.912<br>0.926<br>0.930<br>0.937<br>0.927<br>0.912<br>0.898<br>0.915<br>0.929<br>0.931<br>0.949<br>Multilingual USE<br>0.915<br>0.909<br>0.883<br>0.900<br>0.927<br>0.942<br>0.874<br>0.958<br>0.903<br>0.930<br>0.870<br>0.913<br>0.938<br>0.841<br>0.648<br>0.889<br>0.885<br>0.928<br>0.937<br>0.990<br>0.984<br>Multilingual E5-Base<br>0.936<br>0.899<br>0.890<br>0.927<br>0.956<br>0.944<br>0.934<br>0.959<br>0.896<br>0.951<br>0.929<br>0.943<br>0.948<br>0.958<br>0.964<br>0.901<br>0.889<br>0.942<br>0.933<br>0.979<br>0.986<br>PaLM 2 (Gecko)<br>0.497<br>0.851<br>0.902<br>0.911<br>0.943<br>0.932<br>0.589<br>0.961<br>0.911<br>0.926<br>0.356<br>0.914<br>0.934<br>0.589<br>0.435<br>0.903<br>0.876<br>0.924<br>0.937<br>0.578<br>0.701<br>SimHash<br>0.558<br>0.510<br>0.485<br>0.579<br>0.592<br>0.561<br>0.557<br>0.591<br>0.496<br>0.665<br>0.568<br>0.632<br>0.519<br>0.651<br>0.563<br>0.557<br>0.530<br>0.533<br>0.513<br>0.465<br>0.593<br>MinHash<br>0.633<br>0.506<br>0.508<br>0.581<br>0.598<br>0.558<br>0.574<br>0.591<br>0.504<br>0.585<br>0.595<br>0.568<br>0.556<br>0.697<br>0.606<br>0.556<br>0.512<br>0.563<br>0.575<br>0.223<br>0.814<br>RETSimPartial-Dup<br>0.928<br>0.942<br>0.924<br>0.950<br>0.953<br>0.949<br>0.935<br>0.954<br>0.938<br>0.953<br>0.946<br>0.956<br>0.947<br>0.945<br>0.941<br>0.934<br>0.950<br>0.950<br>0.944<br>0.963<br>0.971<br>RETSimNear-Dup<br>0.971<br>0.976<br>0.968<br>0.973<br>0.988<br>0.978<br>0.974<br>0.987<br>0.976<br>0.985<br>0.971<br>0.981<br>0.983<br>0.989<br>0.975<br>0.962<br>0.962<br>0.976<br>0.982<br>0.986<br>0.991|Table 11: Full per-language Recall@1 performance for various embedding models and algorithms on the W4NT3D benchmark (part 1).|
|---|---|---|



|**Model / Algorithm**<br>**lt**<br>**lv**<br>**ms**<br>**nl**<br>**no**<br>**pl**<br>**pt**<br>**ro**<br>**ru**<br>**sk**<br>**sl**<br>**sr**<br>**sv**<br>**th**<br>**tl**<br>**tr**<br>**uk**<br>**vi**<br>**zh-cn**<br>**zh-tw**|LaBSE<br>0.919<br>0.922<br>0.919<br>0.931<br>0.928<br>0.928<br>0.944<br>0.909<br>0.918<br>0.922<br>0.931<br>0.930<br>0.906<br>0.882<br>0.947<br>0.930<br>0.899<br>0.932<br>0.917<br>0.918<br>Multilingual USE<br>0.902<br>0.919<br>0.932<br>0.936<br>0.921<br>0.931<br>0.952<br>0.869<br>0.910<br>0.901<br>0.908<br>0.906<br>0.899<br>0.888<br>0.949<br>0.940<br>0.893<br>0.910<br>0.986<br>0.985<br>Multilingual E5-Base<br>0.941<br>0.935<br>0.949<br>0.934<br>0.944<br>0.928<br>0.955<br>0.936<br>0.911<br>0.921<br>0.940<br>0.896<br>0.925<br>0.921<br>0.969<br>0.951<br>0.872<br>0.945<br>0.980<br>0.978<br>PaLM 2 (Gecko)<br>0.909<br>0.907<br>0.928<br>0.931<br>0.930<br>0.913<br>0.950<br>0.893<br>0.851<br>0.924<br>0.916<br>0.856<br>0.919<br>0.571<br>0.944<br>0.902<br>0.822<br>0.863<br>0.623<br>0.609<br>SimHash<br>0.609<br>0.624<br>0.533<br>0.527<br>0.577<br>0.586<br>0.548<br>0.514<br>0.554<br>0.575<br>0.580<br>0.553<br>0.552<br>0.669<br>0.507<br>0.606<br>0.517<br>0.609<br>0.276<br>0.315<br>MinHash<br>0.568<br>0.579<br>0.511<br>0.540<br>0.560<br>0.563<br>0.573<br>0.520<br>0.523<br>0.573<br>0.570<br>0.525<br>0.520<br>0.416<br>0.570<br>0.583<br>0.520<br>0.581<br>0.172<br>0.200<br>RETSimPartial-Dup<br>0.957<br>0.945<br>0.961<br>0.950<br>0.958<br>0.953<br>0.954<br>0.948<br>0.946<br>0.961<br>0.947<br>0.963<br>0.953<br>0.941<br>0.961<br>0.954<br>0.941<br>0.951<br>0.946<br>0.957<br>RETSimNear-Dup<br>0.980<br>0.983<br>0.980<br>0.979<br>0.981<br>0.979<br>0.985<br>0.977<br>0.970<br>0.971<br>0.977<br>0.979<br>0.969<br>0.946<br>0.989<br>0.978<br>0.957<br>0.985<br>0.971<br>0.968|Table 12: Full per-language Recall@1 performance for various embedding models and algorithms on the W4NT3D benchmark (part 2).|
|---|---|---|



16 

Preprint 

## A.5 ADDITIONAL ABLATION STUDIES 

This section includes ablation studies on additional hyperparameters for the RETSim model, including the loss function, pooling type, and model capacity. 

|_α_|_β_|_λ_|**Recall@1**|
|---|---|---|---|
|2|20|0.5|0.982|
|2|20|1|0.948|
|2|40|0.5|0.984|
|2|40|1|0.919|
|4|20|0.5|0.982|
|4|20|1|0.947|
|**4**|**40**|**0.5**|**0.986**|
|4|40|1|0.923|



Table 13: Ablation study on Multi-Similarity Loss hyperparameters for RETSim training. **Bold** indicates the hyperparameter setting selected for the final model. 

|**# Blocks**|**Hidden Dim**|**Recall@1**|
|---|---|---|
|2|64|0.965|
|2|128|0.980|
|**2**|**256**|**0.986**|
|2|512|0.986|
|3|64|0.962|
|3|128|0.980|
|3|256|0.984|
|3|512|0.987|
|4|64|0.966|
|4|128|0.980|
|4|256|0.985|
|4|512|0.986|



Table 14: Ablation study for RETSim model capacity and size (number of GAU blocks and hidden dimension for the blocks). **Bold** indicates the hyperparameter setting selected for the final model. 

|**Pooling Type**|**Recall@1**|
|---|---|
|Average Pooling|0.985|
|Max Pooling|0.983|
|**Generalized Mean Pooling**|**0.986**|



Table 15: Ablation study on pooling type for the RETSim model. **Bold** indicates the hyperparameter setting selected for the final model. 

A.6 SELECTED EXAMPLES FROM NEWS-COPY DATASET 

In this section, we randomly selected a set of false positives and false negatives for RETSim on the NEWS-COPY deduplication dataset to provide further insight into the results. 

17 

Preprint 

Text 1 chauffeur, a policeman and a passing journalist who tried to intervene. Beaton and the policeman were reported in serious condition. The 23-year-old princess and her husband of five months, Capt. Mark Phillips, were not hurt. But police experts said the holes left by one of the bullets fired into the car indicated it passed between them, missing them by inches. A police informant said it was believed 11 shots were fired by the assailant. Experts were studying two revolvers found at the scene. They said fi... 

By United Press tnfernational Ay SSAST OR BE FRE NG SG The federal government has proposed new methods of eoustructing federal buildings in a move to save ad- ditional energy and suggested ils elfort could be adapted to all new buildings, 

Text 2 

‘LONDON (AP) — Ian Ball, a 26-year- old unemployed Englishman, was brought into court today and charged with attempted murder during an at- tempt to kidnap Princess Anne from her car in the heart of London Wed- nesday night. Ball, lean-faced and bearded, stood stiffly in the dock at the Bow Street Magistrate’s court, handcuffed to two detectives. He spoke only once during his 60second appearance, saying iha London accent: “I want to apply for legal aid.” The court ordered him held for another hearing on Ma... Hy United Press International The federal government has Proposed new methods of constructing federal buildings in a move lo save addilional energy and suggested ils effort could be adapted to all new buildings, Arthur F, Sampson, General Services Administration ad- ministrater, said new features for such construction would include the collection of rain waler for cooling and irriga- tion, solar energy collectors and the covering of exterior walls with earth. “Whal we are saying is that these design criteri... 

Washington, Jan. 27. —(P)—Im- mediate — Washington, Jan. 27 1 AP).—Immiedlremoval of John F. J. Her- bert, as prohibition aie mmoval of John F. Herbert as pro- — hiadministrator for Montana and Idaho, was bition administrator for Montana and ‘Idaho de- manded in the senate today by Sen- ators was demanded m the Seuate to- ‘day by SenBorah, Idaho, and Whe´eeler, Montana, on ators Borah. idaho, and Waeeler, Montana. on the ground of charges placed before them by the ground of charges placed before them by department of justice investigators. Wheeler Depart- meat of Justice investigators. Wheeler accompanied his demand (Continued on Page accompanied his demand nith a declaration 2) that prohibition en- foreemen: had brukea down. He blamed the “politicians” and called upon the Law Enforcement Commussion to sum- mon members of the Republican Natona... By RAYMOND CLAPPEA (Dnited London, Jan. 24, W.P—The Amer- jean delePresa Stal Correspandoayy London, Jai, gation fo the naval cen- ference teday won its 38—(UP—-The Am ‘erlcnn delegation to demand for presentation of the case of auxilthe navat confer ence today won ls demand jary warships linsitation flrst at the next plefor pre- sentation: of the cnse of suxiliary trary session ‘Vhursday, The chief delegates, warships limitation first at tho noxt plenary meeting at Si, James Pelace, also decided that session Thuvaday, ‘Tho chlet delegates, mecthe plenary session would discuss the main tittg at St. James palace, also decided that tho confeyence questions in alphabetical order af plenary sesslon would discuss the Main conthe cauntries proposing. The American delference questions in alpha betical order of ihe egation won a second victory when it was countriea pro- posing. Press ta be Admitted decided to udmil certain representatives of the The American delegation woo a second vicpress at fie plenary... tory whe... 

Table 16: Example false negatives for RETSim on the NEWS-COPY dataset (pairs of texts not detected as near-duplicates by RETSim but labeled as near-duplicates in the original dataset). Examples are randomly selected and truncated at 512 characters for display. 

18 

Preprint 

## Text 1 

BOZEMAN, Mont. (AP) — Chet Huntley, whose resonant voice and rough-hewn face be- came familiar to millions on the nightly television news, died Wednesday in his mountain resort home. He was 62. He underwent surgery for lung cancer in January but had remained activesuntil recent weeks. He died at 2:20 a.m, according to his widow, Tippy Hunt.cy. Huntiey was teamed for 14 years with David Brinkley on NBC’s HuntleyBrinkley Re- port. He quit in 1970 and returned to his native Montana to develop the $20-millio... 

By THE ASSOCIATED PRESS Some Americans are paying up to 50 per cent more per month for electricity this year than they did last, an Associ- ated Press survey shows. Consumers are beginning to organize to fight the rate hikes. A spot check of monthly elec- tric bills this year and last showed that most increases have been about $1 or $2, gen- erally about 10 per cent, with the highest reported boost com- ing in Jacksonville, Fia., where the average tab went from $17.90 last year to $27.70 this year. Utility... 

BOZEMAN, Mont. (AP) — Vice President Gerald R. Ford says the world will miss the “‘unique abilities” of former television news anchorman Chet Huntley. Huntley, 62, died at his home Wednesday after a long bout with lung cancer. Family ‘spokesmen said a memorial service would be conducted for Huntley Sunday at the Big Sky of Montana’ resort and recreation area south of Bozeman. Huntley was chairman of the Big Sky board of directors. Another memorial service is scheduled Tuesday in the New York studios of the... 

WASHINGTON (AP) — The House has passed legislation raising the minimum wage from $1.60 an hour to $2 this year for most workers covered and to $2.30 for all by 1978. The bill, approved Wednesday 375 to 37, also would increase by 7 million to 56.5 million the number of workers covered by the minimum wage laws. The bill is a modified version of one President Nixon vetoed last year. However, he is expected to sign this one if it is finally approved after ad- justment with a similar Senate passed measure, altho... 

## Text 2 

BOZEMAN, Mont. (AP) - Chet Huntley, whose resonant voice and rough-hewn face became familiar to millions on the nightly television news, died Wednesday in his mountain resort home. He was 62. He underwent surgery for lung cancer in January but had remained active until recent weeks. He died at 2:20 a.m., according to his widow, Tippy Huntley. Huntley was teamed for 14 years with David Brinkley on NBC’s HuntleyBrinkley Report. He quit in 1970 and returned to his native Montana to develop the $20 million Bi... 

By Louise Cook Acenciaiod Prece Writer Same Americans are paying up io 20 per cent more per month far electricity this year ihan they did last, an -Associ- Press survey shows. onsumers are beginning to ze to fight the rate hikes, A spot check of monthly elec- tre hills this year and Jast showed that most increases ve been about $1 or $2, gen- erally about 10 per cent, with the highest reported boost coming in Jacksonville, Fla., where the average tab went from $17.90 last year to $27.70 this year... 

BOZEMAN, Mont. (AP) — Vice President Gerald R. Ford says the world will miss the “unique abilities” of former television news anchorman Chet Huntley. Huntley, 62, died at his home Wednesday after a long bout with lung cancer. Family spokesmen said a memorial service would be con- ducted for Huntley Sunday at the Big- Sky of Montana resort and recreation area south of Bozeman. Huntley was chair- man of the Big Sky board of directors. Another memorial service is scheduled Tuesday in the New York studios of... 

WASHINGTON (AP) — The House has passed legislation raising the minimum wage from $1.60 an hour to $2 this year for most workers covered and to $2.30 for all by 1978. The bill, approved Wednes- day 375 to 37, also would in- crease by 7 million to 56.5 million the number of workers cov- ered by the minimum wage laws. The bill is a modified version of one President Nixon vetoed last year. However, he is ex- ted to sign this one if it is inally approved after adjust- ment with a similar Senate- passed measu... 

Table 17: Example false positives for RETSim on the NEWS-COPY dataset (pairs of texts detected as near-duplicates by RETSim but not labeled as near-duplicates in the original dataset). Examples are randomly selected and truncated at 512 characters for display. 

19 

