Preprint. Work in progress. 

## - - VISRAG: VISION BASED RETRIEVAL AUGMENTED - GENERATION ON MULTI MODALITY DOCUMENTS 

**Shi Yu**[1] _[∗]_ **, Chaoyue Tang**[2] _[∗]_ **, Bokai Xu**[2] _[∗]_ **, Junbo Cui**[2] _[∗]_ **, Junhao Ran**[3] **, Yukun Yan**[1] _[†]_ **, Zhenghao Liu**[4] **, Shuo Wang**[1] **, Xu Han**[1] **, Zhiyuan Liu**[1] _[†]_ **, Maosong Sun**[1] 1Department of Computer Science and Technology, Tsinghua University 

2ModelBest Inc. 3Rice University 4Northeastern University yus21@mails.tsinghua.edu.cn 

## ABSTRACT 

Retrieval-augmented generation (RAG) is an effective technique that enables large language models (LLMs) to utilize external knowledge sources for generation. However, current RAG systems are solely based on text, rendering it impossible to utilize vision information like layout and images that play crucial roles in realworld multi-modality documents. In this paper, we introduce VisRAG, which tackles this issue by establishing a vision-language model (VLM)-based RAG pipeline. In this pipeline, instead of first parsing the document to obtain text, the document is directly embedded using a VLM as an image and then retrieved to enhance the generation of a VLM. Compared to traditional text-based RAG, VisRAG maximizes the retention and utilization of the data information in the original documents, eliminating the information loss introduced during the parsing process. We collect both open-source and synthetic data to train the retriever in VisRAG and explore a variety of generation methods. Experiments demonstrate that VisRAG outperforms traditional RAG in both the retrieval and generation stages, achieving a 25–39% end-to-end performance gain over traditional textbased RAG pipeline. Further analysis reveals that VisRAG is effective in utilizing training data and demonstrates strong generalization capability, positioning it as a promising solution for RAG on multi-modality documents. Our code and data are available at https://github.com/openbmb/visrag. 

## 1 INTRODUCTION 

Trained on massive data, large language models (LLMs) like GPT-4 (Achiam et al., 2023) have shown strong abilities in common NLP tasks using their parametric knowledge (Wei et al., 2022; Zhao et al., 2023). However, the issue of hallucination (Ji et al., 2023; Bang et al., 2023) and the challenge of updating the parametric knowledge limit their real-world application in specific domains. Retrieval-augmented generation (RAG) alleviates this problem by using a knowledge retriever, which has access to a custom outer knowledge base, to supply the LLM with the necessary information for generating outputs (Guu et al., 2020; Lewis et al., 2020; Yu et al., 2023). Opensource RAG frameworks like llamaindex (Liu, 2022) have been developed to facilitate the research and deployment of common RAG pipelines. 

Typical retrieval-augmented generation (RAG) pipelines are _text-based_ , operating on segmented texts as retrieval units (Yu et al., 2023; Asai et al., 2024; Yan et al., 2024), which we refer to as TextRAG. In real-world scenarios, knowledge is often presented in multi-modality documents such as textbooks and manuals which may have texts and figures intersected together. To acquire texts from such data sources, a _parsing_ stage is often employed, which typically involves a cascade of processes, including layout recognition, optical character recognition (OCR), and post-processing steps like text joining (Zhang et al., 2024). While effective in most scenarios, the parsing process inevitably introduces errors, which can negatively impact the retrieval and generation phases. Moreover, text-based RAG utilizes only textual information, overlooking potential information present in 

> _∗_ Equal contribution. 

> _†_ Corresponding authors. 

1 

Preprint. Work in progress. 

other modalities like images. Although research has been conducted on image retrieval and multimodal RAG, these approaches primarily focus on predefined scenarios wherein images and descriptive texts are properly extracted and paired (Wei et al., 2023; Sharifymoghaddam et al., 2024; Zhou et al., 2024), differing from real-world scenarios where texts and images (including figures) are often interleaved within a single document page. 

The recent development of vision-language models (VLMs) has introduced a promising approach to understanding complex visual cues in images and documents (OpenBMB, 2024b; Wang et al., 2024). By integrating a language model with a vision encoder, VLMs demonstrate superior abilities in applications such as describing pictures (Alayrac et al., 2022), explaining figures (Bavishi et al., 2023), and transcribing (printed and handwritten) text from document images (Laurenc¸on et al., 2024). Given the robust capabilities of VLMs in capturing multi-modal information present in images, an intriguing question arises: can the basic language model in the retrieval and generation components of TextRAG be substituted with a VLM, thus the parsing stage is bypassed and all the information of the document is preserved? 

In this paper, we present **Vis** ion-based **R** etrieval- **a** ugmented **G** eneration (VisRAG), to study the feasibility of building a pure-vision RAG pipeline using VLMs. VisRAG is built with a VLM-based retriever VisRAG-Ret and generator VisRAG-Gen. Inherited the bi-encoder of text-based dense retriever (Karpukhin et al., 2020), VisRAG-Ret maps the query and the document into an embedding space, but utilizing the document’s image directly instead of relying on extracted textual content. The embedding is obtained by applying weighted mean pooling on the final hidden states of the input text or vision tokens. After retrieving top- _k_ document images, VisRAG processes these images to generate the answer. While it is straightforward to use a VLM that supports multi-image input for generation, for VLMs that can only accept one single image, we propose page concatenation and weighted selection techniques to enable the handling of multiple documents. Throughout the process, VisRAG preserves all information in its original visual format, thereby preventing the potential information loss or distortion that might occur in traditional RAG pipelines. 

To evaluate VisRAG on real-world multi-modal documents, we construct datasets from open-source visual question answering (VQA) datasets and synthetic query-document pairs derived from webcrawled PDFs. In terms of retrieval, VisRAG-Ret exhibits superior performance in retrieving multimodal documents. It outperforms state-of-the-art text- and vision-centric retrievers and achieves better results than solely relying on its constituent vision encoder or language model under identical training conditions. For generation, VisRAG-Gen surpasses traditional text-based generators with open-source VLMs. With GPT-4o, capable of handling multiple images, VisRAG shows increasing performance gains with more retrieved documents, indicating the potential for improved multi-page reasoning in the future. As depicted in Figure 1, in a direct comparison of pipeline performances, VisRAG achieves a 39% relative improvement over TextRAG using MiniCPM-V 2.6 as the generator and a 25% relative improvement with GPT-4o as the generator, attributed to the cascade effect. Further analysis re- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0002-05.png)


**----- Start of picture text -----**<br>
MiniCPM-V Generator 50.91 50.02<br>50<br>GPT-4o Generator<br>45<br>40.03<br>40<br>36.59<br>35<br>30<br>25<br>20<br>15<br>10<br>TextRAG VisRAG<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: TextRAG vs. VisRAG on final generation accuracy. In TextRAG, parsed text serves as the basis for both retrieval and generation processes. In contrast, VisRAG leverages the original document image directly by using a VLM-based retriever and generator. Details can be found in Sec. 5.1. 

veals that VisRAG possesses better training data efficiency and generalization ability than baseline models, and demonstrates robustness across both text-centric and vision-centric documents. VisRAG shows great promise in replacing TextRAG as the next-generation standard for RAG pipelines. 

## 2 RELATED WORK 

**Retrieval-augmented Generation (RAG).** RAG enhances large language models (LLMs) by incorporating retrieved information from external knowledge bases, which assists in addressing 

2 

Preprint. Work in progress. 

knowledge-intensive tasks (Guu et al., 2020), reducing hallucinations (Semnani et al., 2023), and acquiring new knowledge (Vu et al., 2023). An RAG pipeline typically comprises a text-based retriever that fetches relevant information from the knowledge base given the user query, and an LLM-based generator that reads the query along with the retrieved information to generate an answer (Shi et al., 2024b; Yu et al., 2023). Prior research on RAG primarily focuses on: a) improving the retriever, which is typically a text encoder producing text embeddings, through generator feedback (Yu et al., 2023; Shi et al., 2024b); b) enhancing the generator via supervised fine-tuning (Lin et al., 2024; Xu et al., 2024a), in-context pre-training (Shi et al., 2024a), or advanced prompting (Xu et al., 2024c); and c) developing advanced RAG pipelines to handle long-form or multi-hop question answering (Jiang et al., 2023; Asai et al., 2024). However, research on RAG has predominantly targeted cleaned text corpora like Wikipedia from an academic standpoint. Building effective RAG pipelines for real-world, multi-modal documents remains a challenge. 

**Vision-language Models.** Recent advancements in vision-language models (VLMs) have greatly improved fine-grained multi-modal understanding. Since CLIP (Radford et al., 2021) pioneered contrastive visual-text alignment, models like Flamingo (Alayrac et al., 2022), LLaVA (Liu et al., 2023b), and BLIP (Li et al., 2022) have expanded LLMs to process visual inputs by connecting languages models with a CLIP-style vision encoder. Research has then shifted towards more advanced multi-task and multi-stage pre-training paradigms, enabling models to generalize across a wide range of vision-language tasks (Liu et al., 2024; Bai et al., 2023; Wang et al., 2023; Dai et al., 2023). This is followed by notable advancements in high-resolution visual understanding (Xu et al., 2024b; Bavishi et al., 2023; Lin et al., 2023) and OCR capabilities (Kim et al., 2022; Lee et al., 2023; Hong et al., 2024; Chen et al., 2024b). More recently, breakthroughs have been made in multi-image understanding (Li et al., 2024a; Wang et al., 2024). Recent open-source VLMs like the MiniCPM-V (Yao et al., 2024) and Qwen2-VL (Wang et al., 2024) series combine the merits of recent techniques, achieving state-of-the-art performance. Those features of VLMs provide a foundation for our vision-based RAG pipeline, which requires multi-modal document understanding. 

**Multi-modality Retrieval and RAG.** Multi-modal retrieval encompasses a wide range of tasks, such as retrieving a matching image given the text (Han et al., 2017), retrieving a text-image pair to answer a question (Chang et al., 2022), and retrieving texts that answer the given query about a provided image (Hu et al., 2023a; Luo et al., 2023), etc. Wei et al. (2023) propose UniIR, a universal multi-modal retrieval model capable of addressing the aforementioned multiple tasks. The retrieved information is then employed for incorporating knowledge (Hu et al., 2023b; Luo et al., 2021) or in-context learning (Tan et al., 2024; Liu et al., 2023a), with the aim of generating answers or images (Sharifymoghaddam et al., 2024). Prior research mentioned above is conducted on academic datasets, where texts and images are meticulously extracted from raw data and paired (e.g., images with their captions), to make it feasible to do separate encoding of data in different modalities. This hinders their applicability in real-world RAG scenarios, as real-world multi-modal documents are often presented in mixed modalities, and information may be distributed across various combinations of modalities. Concurrent works DSE (Ma et al., 2024) and ColPali (Faysse et al., 2024) address this issue by directly encoding the image of a document for retrieval. However, as these studies focus on retrieval, they lack a comprehensive comparison of their approaches with text-based retrieval in both in-domain and out-of-domain settings, and do not conduct an end-to-end RAG evaluation. 

## 3 METHODOLOGY 

In this section, we first recap the typical RAG pipeline (Sec. 3.1), then present our VisRAG framework (Sec. 3.2) and the construction of our training and evaluation data (Sec. 3.3). 

## 3.1 PRELIMINARY: RETRIEVAL-AUGMENTED GENERATION 

A typical retrieval-augmented generation (RAG) pipeline consists of a retriever and a generator, both built on large language models (LLMs)[1] . This pipeline operates on a knowledge corpus _D_ , which is processed into units for retrieval and generation, denoted as _D_ = _{d_ 1 _, . . . , dn}_ , where 

1In many cases, the retriever uses language models smaller than 1B parameters, which may not be considered “large”, but we use the term LLM for simplicity. 

3 

Preprint. Work in progress. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0004-01.png)


Figure 2: TextRAG (left) vs. VisRAG (right). Traditional text-based RAG (TextRAG) relies on parsed texts for retrieval and generation, losing visual information in multi-modal documents. Our vision-based RAG (VisRAG) employs a VLM-based retriever and generator to directly process the document page’s image, thereby preserving all information in the original page. 

_n_ is the number of retrieval units. Given a text query _q_ and the retrieval corpus _D_ , the retriever functions as _R_ : ( _q, D_ ) _→DR_ , taking _q_ and _D_ as inputs and producing a candidate set _DR ⊂D_ . To enable efficient search, the units in the knowledge corpus _D_ are pre-encoded into embeddings. During RAG pipeline inference, approximate nearest neighbor (ANN) search is applied to retrieve _DR_ , which serves as the knowledge source for generation. The generation process can be defined as a function _G_ : ( _q, DR_ ) _→ a_ , where _a_ represents the answer and _G_ denotes the LLM generator. This is achieved by prompting the LLM with the query and the retrieved units _DR_ to generate an answer. 

As shown in Figure 2 (left), traditional RAG frameworks (TextRAG) typically utilize text-based units for retrieval and generation. However, in real-world scenarios, data often appear in complex, multi-modal documents, requiring an additional parsing step to obtain text. In this paper, we propose to use the _page_ as the fundamental unit for retrieval and generation, which is directly processed by vision language models (VLMs) as an image without further processing during retrieval and generation. In subsequent sections, we use the terms “page” and “document” interchangeably. 

## 3.2 VISRAG: VISION-BASED RETRIEVAL-AUGMENTED GENERATION 

In this section, we present **Vis** ion-based **R** etrieval- **a** ugmented **G** eneration (VisRAG), as shown in Figure 2 (right). In contrast to traditional RAG frameworks which use text segments for both retrieval and generation, VisRAG leverages the image of the document to preserve all information. 

## 3.2.1 RETRIEVAL 

The first stage of VisRAG, VisRAG-Ret, aims to retrieve a set of pages from the corpus _D_ given _q_ . We follow the dual-encoder paradigm in text-based dense retrieval models (Karpukhin et al., 2020) but employ a VLM rather than an LLM to encode the query and page. Specifically, the query and page are encoded separately as text and image in the VLM, producing in a sequence of hidden states. To derive the final embedding, and given that we use generative VLMs with causual attention, we adopt the position-weighted mean pooling over the last-layer VLM hidden states (Muennighoff, 2022), giving higher weights to later tokens: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0004-09.png)


where **h** _i_ is the _i_ -th hidden state, _S_ is the sequence length, _wi_ = � _Sji_ =1 _[j]_[is the] _[ i]_[-th weight, and] **[ v]**[ is] the query or page embedding. The similarity score is calculated by the cosine similarity of the query 

4 

Preprint. Work in progress. 

and page embedding. VisRAG-Ret is optimized using the InfoNCE loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0005-02.png)


where _d_[+] , _D[−]_ are positive document and the negative document set of _q_ , respectively, _s_ ( _q, d_ ) is the similarity score between _q_ and _d_ , and _τ_ is the temperature. 

## 3.2.2 GENERATION 

The second stage of VisRAG, VisRAG-Gen, focuses on generating the answer according to the user query and retrieved pages using a VLM. We propose the following mechanisms to enable VisRAGGen to handle multiple retrieved pages in _DR_ for generation. The prompts used for generation is presented in Appendix E. 

**Page Concatenation.** A straightforward approach is to concatenate all pages in _DR_ into a single image to accommodate most VLMs that are trained to accept a single image. Formally, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0005-07.png)


where VLM-Single is a VLM that accepts a single image with text prompt and Concat is the image concatenation operation. In this paper, we experiment with horizontal concatenation. 

**Weighted Selection.** Another approach is to ask the VLM to generate an answer for every page from top- _k_ , and select a final one with the highest confidence (Lewis et al., 2020; Shi et al., 2024b). The final confidence is defined as the weighted generation probability of the answer: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0005-10.png)


where _P_ ( _a|d, q_ ) is calculated as the reciprocal of the perplexity of generating the answer _a_ conditioned on the single document _d_ , and _λ_ ( _d, q_ ) is the normalized retrieval score: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0005-12.png)


**VLMs Accepting Multiple Images.** Some recent VLMs like MiniCPM-V 2.6 (OpenBMB, 2024b) and Qwen-VL 2 (Wang et al., 2024) are designed and trained to accept multiple images as input to perform cross-image reasoning. This capability may be useful for the generation as the required information could be located on a single page from the retrieved document set _DR_ for single-hop questions or spread across multiple pages for multi-hop questions. Formally, we have 

_a ←−_ VLM-Multi( _q, {d|d ∈DR}_ ) _,_ (6) 

where VLM-Multi is the VLM that accepts multiple images with text prompt. 

## 3.3 DATA CONSTRUCTION 

To effectively build and evaluate RAG pipelines on multi-modal documents, we construct our datasets using a combination of visual question answering (VQA) datasets and synthetic data. The statistics of our constructed dataset are provided in Table 1. 

**Data Sources.** We collect question-document pairs from a series of VQA datasets, targeting different document types: MP-DocVQA (Tito et al., 2023) for industrial documents, ArXivQA (Li et al., 2024b), ChartQA (Masry et al., 2022), InfographicsVQA (Mathew et al., 2022), and PlotQA (Methani et al., 2020) for various figure types, and SlideVQA (Tanaka et al., 2023) for presentation slides. All datasets feature questions that can be answered using a single document (page), except for SlideVQA, which includes multi-hop questions requiring information from multiple pages. We follow the original datasets’ train-test splits, except for MP-DocVQA and InfographicsVQA, where the validation split serves as our evaluation set. Additionally, we enhance our training set by collecting openly available PDFs from online sources and generating queries using GPT-4o (OpenAI, 2024), with details presented in Appendix A.1. We assemble the retrieval corpus by gathering the positive document associated with each query from the training and evaluation sets. 

5 

Preprint. Work in progress. 

Table 1: Dataset statistics. We collect data from visual question answering (VQA) datasets for training and evaluation and synthetic additional query-document pairs for training. We apply filtering on VQA datasets to remove context-dependent queries that are not suitable for retrieval. 

|**Source**|**Document Type**|**% Filtered**|**Train**<br>**# Q-D Pairs**|**# Q**|**Evaluation**<br>**# D**<br>**# Pos. D per Q**|**Evaluation**<br>**# D**<br>**# Pos. D per Q**|
|---|---|---|---|---|---|---|
|ArXivQA (2024b)|Arxiv Figures|14%|25,856|8,640|8,066|1.00|
|ChartQA (2022)|Charts|41%|4,224|718|500|1.00|
|MP-DocVQA (2023)|Industrial Documents|69%|10,624|1,879|741|1.00|
|InfoVQA (2022)|Infographics|26%|17,664|2,046|459|1.00|
|PlotQA (2020)|Scientifc Plots|44%|56,192|11,307|9,593|1.00|
|SlideVQA (2023)|Slide Decks|22%|8,192|1,640|1,284|1.34|
|Synthetic|Various|-|239,358|-|-|-|



**Query Filtering.** Some queries extracted from VQA datasets are _context-dependent_ , which lack specificity to a certain entity. For instance, the response to “Where was the conference held?” varies based on the contextual document. Using such context-dependent queries in open retrieval tasks is ineffective because they lack strong document specificity. To address this, we implement an additional filtering stage to remove these context-dependent questions, where we prompt llama-3-8binstruct (AI@Meta, 2024) with human-annotated in-context samples to generate the classification label. Table 1 shows a substantial reduction in context-dependent questions across data sources. The details of filtering are presented in Appendix A.2. 

**Evaluation Metrics.** For retrieval, we evaluate the performance using MRR@10 and Recall@10. For generation, consistent with methods applied to the source datasets, we report the answer accuracy, employing a relaxed exact match metric which allows a 5% error margin for numeric responses (Masry et al., 2022; Methani et al., 2020). 

## 4 EXPERIMENTAL METHODOLOGY 

**Document Parsing.** To assess the performance of VisRAG in comparison to TextRAG, we employ specific text extraction methods. The first approach, referred to as “(OCR)” in subsequent text, is a pipeline that initially leverages PPOCR (Du et al., 2020) to identify text regions, then combines vertically aligned and horizontally proximate text boxes to reduce fragmentation. The second approach, termed “(Captioner)”, is an end-to-end model-based method. In this approach, we apply MiniCPM-V 2.0 (OpenBMB, 2024a; Yao et al., 2024), fine-tuned on paired (document image, extracted text) data, to directly parse text from the document image. Details of the parsing processes are presented in Appendix B. 

**Retrieval Experiments.** VisRAG-Ret is a document embedding model built on MiniCPM-V 2.0, a vision-language model that integrates SigLIP (Zhai et al., 2023) as the vision encoder and MiniCPM (Hu et al., 2024) as the language model. To ensure fair comparisons, we organize experiments into three settings: off-the-shelf, out-of-domain, and in-domain, as depicted below. 

- Off-the-shelf: We directly evaluate popular text and image retrieval models on extracted texts, including BM25 (OCR), a lexical model; bge-large-en-v1.5 (Xiao et al., 2023) (OCR) and NV-Embed-v2 (Lee et al., 2024) (OCR), state-of-the-art text embedding models with sizes 335M and 7.85B, respectively; and SigLIP, a CLIP-style (Radford et al., 2021) vision model serving as the encoder for MiniCPM-V series. 

- Out-of-domain: Models in this category are trained solely on synthetic data and evaluated on the VQA datasets, lacking in-domain supervision, in order to show the models’ generalization capabilities. These models include textual models MiniCPM (OCR), MiniCPM (Captioner), and vision model SigLIP. MiniCPM (OCR) and (Captioner) are MiniCPMbased text embedding models trained and evaluated on extracted text. 

- In-domain: Models in this category are trained on the blend of the VQA training data and synthetic data. We evaluate the same set of models as in the out-of-domain setting to show model performance when supervised labels are available. We also run ColPali (Faysse 

6 

Preprint. Work in progress. 

Table 2: Overall retrieval performance in MRR@10. The best retrieval performance in each group is marked in **bold** , and the second best performance is underlined. ColPali is trained on its synthetic data and datasets marked with _†_ . Corresponding Recall@10 performance can be found in Table 6. 

|**Model**|**# Para.**|**ArxivQA**|**ChartQA**|**DocVQA**|**InfoVQA**|**PlotQA**|**SlideVQA**|**Average**|
|---|---|---|---|---|---|---|---|---|
|(a)Off-the-shel||||f Models|||||
|BM25 (OCR)<br>bge-large (2023) (OCR)<br>NV-Embed-v2 (2024) (OCR)<br>SigLIP(2023)|n.a.<br>335M<br>7.85B<br>883M|32.30<br>27.63<br>**45.64**<br>10.93|41.64<br>39.92<br>**52.58**<br>32.95|71.92<br>50.21<br>**73.01**<br>25.66|63.99<br>67.01<br>**81.70**<br>38.77|35.32<br>30.95<br>**36.10**<br>12.90|84.57<br>79.94<br>**91.73**<br>52.50|54.96<br>49.28<br>**63.46**<br>28.95|
|(b)Out-of-domain: _Models Fine-tuned on Synthetic Data_|||||||||
|MiniCPM (OCR)<br>MiniCPM (Captioner)<br>SigLIP (2023)|2.72B<br>2.72B<br>883M|37.19<br>34.59<br>37.01|43.34<br>46.65<br>49.34|68.57<br>64.25<br>58.32|75.36<br>71.84<br>63.55|21.93<br>18.85<br>21.23|86.80<br>82.96<br>85.32|55.53<br>53.19<br>52.46|
|<br>VisRAG-Ret|3.43B|**59.03**|**51.18**|**73.28**|**81.17**|**28.70**|**90.38**|**63.96**|
|(c)In-domain: _Models Fine-tuned on Synthetic and In-domain data_|||||||||
|MiniCPM (OCR)<br>MiniCPM (Captioner)<br>SigLIP (2023)<br>ColPali (2024)|2.72B<br>2.72B<br>883M<br>2.92B|47.36<br>47.53<br>50.22<br>64.36_†_|54.43<br>54.28<br>**61.44**<br>56.46|74.13<br>68.87<br>66.01<br>**84.10**_†_|80.11<br>76.46<br>72.87<br>78.74_†_|**41.33**<br>35.93<br>40.15<br>30.37|90.49<br>85.45<br>89.51<br>**93.91**|64.64<br>61.42<br>63.37<br>67.99|
|VisRAG-Ret|3.43B|**67.00**|59.34|77.65|**84.05**|40.26|91.71|**70.00**|



et al., 2024) on our evaluation data. ColPali is a page embedding model that encodes a page into multiple vectors. Its training set includes its own synthetic data as well as ArxivQA, DocVQA, and InfographicsVQA (Faysse et al., 2024). 

We report VisRAG-Ret’s performance in both out-of-domain and in-domain settings. 

**Generation Experiments.** To evaluate generation performance, we fix the retrieval model to VisRAG-Ret and report the performance of various generation models and methods. For VisRAGGen, we compare the performance of the single-image VLM MiniCPM-V 2.0, which only accepts a single image, against the multi-image VLM MiniCPM-V 2.6 (OpenBMB, 2024b; Yao et al., 2024) and GPT-4o (OpenAI, 2024). MiniCPM-V 2.6 is an upgrade of MiniCPM-V 2.0, incorporating Qwen2-7B (Yang et al., 2024) as the language model and supporting multi-image input. We evaluate the performance of page concatenation and weighted selection on the single-image VLM. Additionally, we report the performance of text-based generation baselines, including MiniCPM (OCR) and GPT-4o (OCR), where only extracted texts are used for generation. For all experiments, we report results using the top-1, top-2, and top-3 retrieved documents, as well as an “Oracle” condition where the model is provided with only the positive document(s) to show the performance upper bound. 

**Implementation Details.** VisRAG-Ret is fine-tuned using in-batch negatives (Karpukhin et al., 2020) for one epoch with a batch size of 128 on 8 NVIDIA A100 80GB GPUs. The temperature parameter in Equation 2 is set to 0.02. Baseline retrievers are fine-tuned with the same hyperparameters, and textual baselines utilize extracted text data as document-side input. The generation part does not use any fine-tuning; we directly use off-the-shelf LLMs/VLMs for generation. 

## 5 EVALUATION RESULTS 

## 5.1 OVERALL PERFORMANCE 

**Retrieval Performance.** In this experiment, we compare VisRAG-Ret with (a) off-the-shelf models, and trained baselines in (b) out-of-domain setting where we only leverage synthetic data, and in (c) in-domain setting where we leverage both in-domain and synthetic training data. 

As shown in Table 2(a)(b), VisRAG-Ret, trained on out-of-domain data, outperforms all off-the-shelf baselines, including both text and vision models. It significantly outperforms both BM25 and bgelarge, and surpasses NV-Embed-v2, a state-of-the-art text retrieval model with 7.85B parameters. Note that bge-large and NV-Embed-v2 are trained on millions of query-doc pairs (Xiao et al., 2023; Lee et al., 2024), which are 10x more than our training data. Although bge-large outperforms BM25 on benchmarks like MTEB (Muennighoff et al., 2023), it fails on our datasets, indicating text-based embedding models trained on clean text struggle with texts parsed from real-world documents. 

7 

Preprint. Work in progress. 

Table 3: Overall generation performance in accuracy (%). Different generation models and methods utilize the same retriever, VisRAG. Performance relative to Oracle (using the ground-truth document(s) for generation) is colored in blue. 

|ment(s)forgeneration)|is colored|inblue.||||||
|---|---|---|---|---|---|---|---|
|**Model / Method**<br>**Input**|**ArxivQA**|**ChartQA**|**DocVQA**|**InfoVQA**|**PlotQA**|**SlideVQA**|**Average**|
|(a)TextRAG-Gen: _Text-based Generation_||||||||
|MiniCPM (OCR)<br>top-1<br>top-2<br>top-3<br>Oracle|42.87(96.0%)<br>42.57(95.4%)<br>42.63(95.5%)<br>44.64(100%)|24.79(83.6%)<br>25.07(84.5%)<br>24.65(83.1%)<br>29.67(100%)|27.99(79.1%)<br>28.21(79.7%)<br>25.49(72.0%)<br>35.39(100%)|16.86(88.5%)<br>16.32(85.6%)<br>15.30(80.3%)<br>19.06(100%)|21.97(97.8%)<br>22.18(98.8%)<br>21.52(95.8%)<br>22.46(100%)|25.43(91.2%)<br>26.83(96.3%)<br>27.38(98.2%)<br>27.87(100%)|26.65(89.3%)<br>26.86(90.0%)<br>26.16(87.7%)<br>29.85(100%)|
|GPT-4o (OCR)<br>top-1<br>top-2<br>top-3<br>Oracle|58.51(94.8%)<br>58.00(94.0%)<br>58.54(94.8%)<br>61.72(100%)|39.69(65.8%)<br>41.36(68.6%)<br>40.25(66.7%)<br>60.31(100%)|44.86(77.1%)<br>47.74(82.1%)<br>48.43(83.3%)<br>58.17(100%)|37.34(84.5%)<br>39.00(88.3%)<br>39.69(89.8%)<br>44.18(100%)|25.70(94.4%)<br>18.42(67.6%)<br>17.87(65.6%)<br>27.24(100%)|41.95(83.0%)<br>45.85(90.7%)<br>45.12(89.3%)<br>50.55(100%)|41.34(82.1%)<br>41.73(82.9%)<br>41.65(82.7%)<br>50.36(100%)|
|(b)VisRAG-Gen: _Single-image VLM(MiniCPM-V 2.0)_||||||||
|Page Concatenation<br>top-1<br>top-2<br>top-3<br>Oracle|56.82(94.5%)<br>56.22(93.5%)<br>55.49(92.3%)<br>60.10(100%)|28.97(76.8%)<br>25.63(67.9%)<br>25.35(67.2%)<br>37.74(100%)|36.83(76.1%)<br>27.83(57.5%)<br>20.81(43.0%)<br>48.38(100%)|24.29(85.7%)<br>18.96(66.9%)<br>16.42(57.9%)<br>28.35(100%)|25.95(84.1%)<br>25.73(83.4%)<br>24.37(79.0%)<br>30.84(100%)|31.10(89.7%)<br>29.75(85.8%)<br>28.17(81.2%)<br>34.68(100%)|34.44(84.8%)<br>30.96(76.3%)<br>28.62(70.5%)<br>40.60(100%)|
|Weighted Selection<br>top-1<br>top-2<br>top-3<br>Oracle|56.82(94.5%)<br>56.67(94.3%)<br>57.12(95.0%)<br>60.10(100%)|28.97(76.8%)<br>29.25(77.5%)<br>29.53(78.2%)<br>37.74(100%)|36.83(76.1%)<br>37.15(76.8%)<br>35.76(73.9%)<br>48.38(100%)|24.29(85.7%)<br>24.24(85.5%)<br>24.00(84.7%)<br>28.35(100%)|25.95(84.1%)<br>25.98(84.3%)<br>28.53(92.5%)<br>30.84(100%)|33.78(86.3%)<br>34.33(87.7%)<br>34.57(88.3%)<br>39.15(100%)|34.44(84.5%)<br>34.60(84.9%)<br>34.92(85.7%)<br>40.76(100%)|
|(c)VisRAG-Gen: _Multi-image VLM_||||||||
|MiniCPM-V 2.6<br>top-1<br>top-2<br>top-3<br>Oracle|65.30(90.4%)<br>65.14(90.1%)<br>65.45(90.6%)<br>72.27(100%)|41.50(63.1%)<br>41.23(62.7%)<br>42.76(65.0%)<br>65.74(100%)|61.36(74.8%)<br>65.57(80.0%)<br>66.05(80.5%)<br>82.01(100%)|46.63(83.9%)<br>46.29(83.3%)<br>45.45(81.8%)<br>55.57(100%)|44.22(74.3%)<br>35.24(59.2%)<br>37.67(63.3%)<br>59.51(100%)|46.46(81.8%)<br>48.60(85.5%)<br>49.45(87.0%)<br>56.83(100%)|50.91(77.9%)<br>50.34(77.1%)<br>51.14(78.3%)<br>65.32(100%)|
|GPT-4o<br>top-1<br>top-2<br>top-3<br>Oracle|62.94(93.1%)<br>62.19(92.0%)<br>62.26(92.1%)<br>67.58(100%)|43.45(66.0%)<br>43.59(66.2%)<br>46.10(70.0%)<br>65.88(100%)|59.50(75.9%)<br>63.54(81.0%)<br>64.93(82.8%)<br>78.45(100%)|54.01(83.0%)<br>56.99(87.5%)<br>56.45(86.7%)<br>65.10(100%)|29.74(75.8%)<br>26.42(67.4%)<br>27.26(69.5%)<br>39.22(100%)|50.49(80.5%)<br>56.04(89.3%)<br>56.34(89.8%)<br>62.74(100%)|50.02(79.2%)<br>51.46(81.5%)<br>52.22(82.7%)<br>63.16(100%)|



When trained with the same data setup, as demonstrated in Table 2(b)(c), VisRAG-Ret outperforms text models MiniCPM (OCR) & (Captioner) and the vision model SigLIP by a significant margin. The advantage is more pronounced in the out-of-domain setting, where VisRAG-Ret achieves 15% and 22% gains over MiniCPM (OCR) and SigLIP, respectively, compared to 8% and 10% in the in-domain setting. This indicates that VisRAG-Ret has better generalization capability compared to text- and vision-centric models. Notably, despite utilizing the same VLM MiniCPM-V 2.0 for parsing, MiniCPM (Captioner) performs worse than VisRAG-Ret, indicating that directly encoding with VLMs works better than using VLMs for parsing. This can be attributed to the inevitable information loss when multi-modality information is transcribed into text. 

Further analysis reveals that MiniCPM (OCR) and SigLIP perform differently across datasets: SigLIP excels in ArxivQA and ChartQA, while MiniCPM (OCR) significantly outperforms SigLIP in DocVQA and InfographicsVQA. This may be due to the different focuses of the two models: MiniCPM focuses on text, while SigLIP focuses on visual signals. VisRAG-Ret, built on top of MiniCPM-V 2.0, with a SigLIP encoder and a MiniCPM language model, combines the merits of both and performs well across all datasets, capturing more holistic information from a document. 

Compared to ColPali, a multi-vector document page embedding model, VisRAG-Ret not only maintains superior performance but also achieves much better memory efficiency. ColPali represents a page with 256KB of data distributed across 1030 128-dim vectors (Faysse et al., 2024), whereas VisRAG-Ret uses just 4.5KB in a single 2304-dimensional vector. This makes VisRAG-Ret more suitable for scaling to millions or billions of documents in real-world applications. 

**Generation Performance.** In this experiment, we apply a series of text- and vision-based generators and methods on top of the same retriever VisRAG-Ret to study their effectiveness in generating the answer given the query and retrieved documents. Table 3 shows the performance of (a) text-based generation (TextRAG-Gen), (b) generation using the VLM MiniCPM-V 2.0 which only accepts a single image as input, and (c) generation using VLMs which accept multiple images as input. 

When models are provided with only the ground-truth documents (“Oracle”), VisRAG-Gen models, which process the document image directly, significantly outperform RAG-Gen models, which rely solely on extracted text. For instance, MiniCPM-V 2.0 achieves 36% higher performance than MiniCPM (OCR) when using ground-truth documents. This underscores the importance of visual 

8 

Preprint. Work in progress. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0009-01.png)


**----- Start of picture text -----**<br>
22.1%<br>42.7%<br>73.4% 76.8%<br>51.3%<br>34.1%<br>26.6% 23.2%<br>(a) TextRAG with MiniCPM (OCR) as the retriever (b) VisRAG with VisRAG-Ret as the retriever and<br>and MiniCPM-V 2.6 (OCR) as the generator. MiniCPM-V 2.6 as the generator.<br>**----- End of picture text -----**<br>


Figure 3: Pipeline performance of (a) TextRAG and (b) VisRAG on InfographicsVQA. We visualize the portion of queries that have the positive document retrieved at the top-1 position (“Correct Retrieval”), and that are answered correctly given the top-1 retrieved document (“Correct Generation”). 

clues in extracting answers from documents and indicates that VisRAG-Gen has a higher performance upper bound than TextRAG-Gen. 

In practical scenarios where models receive the top-1 to 3 retrieved documents, which may include noise, VisRAG-Gen consistently outperforms TextRAG-Gen within the same model series. Specifically, for MiniCPM-V 2.0, capable of processing only a single image, the weighted selection approach demonstrates better performance than page concatenation when handling 2 or 3 retrieved documents. Simple concatenation may overwhelm the VLM with unnecessary information, while weighted selection filters answers based on multiple VLM outputs conditioned on individual documents, thus reducing the information burden. 

TextRAG pipelines usually benefit from an increased number of retrieved documents due to better information coverage (Zhu et al., 2024). However, while weighted selection enhances robustness in performance, there is no significant performance boost with a higher count of retrieved documents using this approach. Notably, only the most advanced VLMs, such as GPT-4o, which handle multiple images, show a marked performance increase as the number of retrieved documents rises. This suggests that reasoning over multiple images remains a challenging task for current VLMs. 

**End-to-end Performance.** In this experiment, we study the effectiveness of the VisRAG _pipeline_ , by comparing it with the TextRAG pipeline. We construct TextRAG using MiniCPM (OCR) and MiniCPM-V 2.6 (OCR) for retrieval and generation, respectively, and VisRAG using VisRAG-Ret for retrieval and MiniCPM-V 2.6 for generation. The performance on InfographicsVQA is visually represented in Figure 3. 

Notebly, VisRAG achieves a higher rate of accurately retrieving documents than TextRAG, and demonstrates a significantly improved rate of correct answer generation from accurately retrieved documents. The cumulative improvements in both retrieval and generation phases result in an overall accuracy increment from 22.1% to 42.7%. Across the six evaluation datasets, VisRAG shows a 39% relative accuracy increment on average, as illustrated in Figure 1. The case study of VisRAG and TextRAG is presented in Appendix F. 

## 5.2 TRAINING DATA EFFICIENCY 

As retrieval acts as the bottleneck in an RAG pipeline, it is crucial to have an effective retrieval component to maintain optimal performance. In this 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0009-10.png)


**----- Start of picture text -----**<br>
70<br>60<br>50<br>40<br>30<br>VisRAG-Ret<br>20 MiniCPM (OCR)<br>10 bge-large (OCR)<br>NV-Embed-v2 (OCR)<br>0<br>0 5.0e+04 1.0e+05 1.5e+05 2.0e+05<br># Train Q-D Pairs<br>Average MRR@10<br>**----- End of picture text -----**<br>


Figure 4: Average retrieval performance of VisRAG-Ret vs. MiniCPM (OCR) trained with different numbers of training examples. 

9 

Preprint. Work in progress. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0010-01.png)


**----- Start of picture text -----**<br>
Retrieval Retrieval & Generation<br>ArxivQA<br>InfographicsVQA<br>**----- End of picture text -----**<br>


Figure 5: Relative retrieval and generation performance of VisRAG, VisRAG (SigLIP), and TextRAG on different subsets of queries. The X-axes represent the query subsets where the lengths of the positive documents fall within specific percentile ranges. For comparative analysis, we set TextRAG’s performance to zero and show the performance differences of other models from TextRAG. 

experiment, we study the training data efficiency of VisRAG-Ret by evaluating the performance of VisRAG-Ret trained under different amounts of synthetic training data, i.e. in the out-of-domain setting. As shown in Figure 4, when only trained on 20k q-d pairs, VisRAG can surpass bge-large (OCR). After training on 150k pairs, it can further surpass NV-Embed-v2 (OCR), the SOTA 8Bsized text embedding model trained on millions of curated text pairs. This highlights VisRAG-Ret’s high training data efficiency and strong generalization capability, as all models are evaluated out-ofdomain. When compared with MiniCPM (OCR), which uses extracted text for training, VisRAG-Ret consistently achieves a performance gain of about 17% and exhibits a more stable training process. The results show VisRAG-Ret’s potential for further performance improvements by scaling up the training data. 

## 5.3 PERFORMANCE ON DIFFERENT DATA SUBSETS 

In this experiment, we assess the retrieval and generation performance of VisRAG and TextRAG defined in Figure 3, as well as VisRAG (SigLIP), which replaces the retriever in VisRAG with SigLIP. We report their performance across different data subsets by categorizing queries based on the lengths of their positive documents, measured by the number of tokens of the extracted text. Documents with a higher volume of extracted text may prioritize textual information over visual content. As illustrated in Figure 5, queries in ArxivQA and InfographicsVQA are divided into equal-sized bins according to the lengths of their relevant documents. For each bin, we calculate and plot the average performance differences between VisRAG and TextRAG, as well as between VisRAG (SigLIP) and TextRAG, to compare how each model performs relative to TextRAG. We observe that, in general, the relative performance of VisRAG and VisRAG (SigLIP) improves as the length of the relevant document decreases. This suggests that models with vision encoders can better understand documents that emphasize visual information. However, VisRAG (SigLIP) consistently underperforms VisRAG across all data subsets and, in some cases, even performs worse than TextRAG. In contrast, VisRAG consistently outperforms TextRAG, indicating that the underlying language model in VisRAG is crucial for better understanding the semantics conveyed through visual cues. 

10 

Preprint. Work in progress. 

## 6 CONCLUSION 

In this paper, we propose VisRAG, a novel retrieval-augmented generation (RAG) paradigm that utilizes vision-language models (VLMs) to facilitate retrieval and generation within an RAG pipeline, thereby eliminating the parsing stage required in traditional text-based RAG. Our empirical results demonstrate that VisRAG consistently outperforms text-based RAG on retrieval and generation while maintaining a simpler pipeline. We hope that VisRAG will inspire future RAG development to incorporate VLMs for handling multi-modal documents. 

## REFERENCES 

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/ llama3/blob/main/MODEL_CARD.md. 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In _Proceedings of NeurIPS_ , volume 35, pp. 23716–23736, 2022. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In _Proceedings of ICLR_ , 2024. 

- Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. 

- Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. In _Proceedings of AACL/IJCNLP 2023_ , pp. 675–718, 2023. 

- Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Tas¸ırlar. Introducing our multimodal models, 2023. URL https://www.adept. ai/blog/fuyu-8b. 

- Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In _Proceedings of CVPR_ , pp. 16495–16504, 2022. 

- Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. _arXiv preprint arXiv:2402.11684_ , 2024a. 

- Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. _arXiv preprint arXiv:2406.11317_ , 2024b. 

- Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. In _Proceedings of NeurIPS_ , 2023. 

- Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing Dang, and Haoshuang Wang. Pp-OCR: A Practical Ultra Lightweight OCR System. _arXiv_ , abs/2009.09941, 2020. 

- Manuel Faysse, Hugues Sibille, Tony Wu, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ , 2024. 

- Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented language model pre-training. In _Proceedings of ICML_ , pp. 3929–3938, 2020. 

11 

Preprint. Work in progress. 

Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, and Larry S Davis. Automatic spatially-aware fashion concept discovery. In _Proceedings of ICCV_ , pp. 1463–1471, 2017. 

- Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In _Proceedings of CVPR_ , pp. 14281–14290, 2024. 

- Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 12065–12075, 2023a. 

- Shengding Hu, Yuge Tu, Xu Han, Ganqu Cui, Chaoqun He, Weilin Zhao, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Xinrong Zhang, Zhen Leng Thai, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, dahai li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the Potential of Small Language Models with Scalable Training Strategies. In _First Conference on Language Modeling_ , volume abs/2404.06395, 2024. 

- Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid, David A Ross, and Alireza Fathi. Reveal: Retrieval-augmented visual-language pre-training with multi-source multimodal knowledge memory. In _Proceedings of CVPR_ , pp. 23369–23379, 2023b. 

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. _ACM Comput. Surv._ , (12):248:1–248:38, 2023. 

- Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In _Proceedings of EMNLP_ , pp. 7969–7992, 2023. 

- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In _Proceedings of EMNLP_ , pp. 6769–6781, 2020. 

- Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _Proceedings of ECCV_ , pp. 498–517. Springer, 2022. 

- Hugo Laurenc¸on, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? _arXiv preprint arXiv:2405.02246_ , 2024. 

- Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. _arXiv_ , abs/2405.17428, 2024. 

- Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _Proceedings of ICML_ , pp. 18893–18912, 2023. 

- Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In _Proceedings of NeurIPS_ , 2020. 

- Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_ , 2024a. 

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In _Proceedings of ICML_ , pp. 12888–12900, 2022. 

12 

Preprint. Work in progress. 

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models. In _Proceedings of ACL_ , pp. 14369–14387, 2024b. 

- Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. Ra-dit: Retrieval-augmented dual instruction tuning. In _Proceedings of ICLR_ , 2024. 

- Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. _arXiv preprint arXiv:2311.07575_ , 2023. 

- Bingshuai Liu, Chenyang Lyu, Zijun Min, Zhanyu Wang, Jinsong Su, and Longyue Wang. Retrievalaugmented multi-modal chain-of-thoughts reasoning for large language models. _arXiv preprint arXiv:2312.01714_ , 2023a. 

- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Proceedings of NeurIPS_ , volume 36, pp. 34892–34916, 2023b. 

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In _Proceedings of CVPR_ , pp. 26296–26306, 2024. 

Jerry Liu. LlamaIndex, 11 2022. URL https://github.com/jerryjliu/llama_index. 

- Man Luo, Yankai Zeng, Pratyay Banerjee, and Chitta Baral. Weakly-supervised visual-retrieverreader for knowledge-based question answering. In _Proceedings of EMNLP_ , pp. 6417–6431, 2021. 

- Man Luo, Zhiyuan Fang, Tejas Gokhale, Yezhou Yang, and Chitta Baral. End-to-end knowledge retrieval with multi-modal queries. In _Proceedings of ACL_ , pp. 8573–8589, 2023. 

- Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying multimodal retrieval via document screenshot embedding. _arXiv preprint arXiv:2406.11251_ , 2024. 

- Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. Chartqa: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In _Proceedings of ACL_ , pp. 2263–2279, 2022. 

- Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In _IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 2582–2591, 2022. 

- Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. Plotqa: Reasoning over scientific plots. In _The IEEE Winter Conference on Applications of Computer Vision (WACV)_ , March 2020. 

Niklas Muennighoff. Sgpt: Gpt sentence embeddings for semantic search. _arXiv preprint arXiv:2202.08904_ , 2022. 

- Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. In _Proceedings of EACL_ , pp. 2014–2037, 2023. 

OpenAI. Hello, gpt-4o — openai, 2024. URL https://openai.com/index/ hello-gpt-4o/. 

- OpenBMB. openbmb/minicpm-v-2, 2024a. URL https://huggingface.co/openbmb/ MiniCPM-V-2. 

- OpenBMB. openbmb/minicpm-v-2 ~~6~~ , 2024b. URL https://huggingface.co/openbmb/ MiniCPM-V-2_6. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _Proceedings of ICML_ , pp. 8748–8763, 2021. 

13 

Preprint. Work in progress. 

- Sina J Semnani, Violet Z Yao, Heidi C Zhang, and Monica S Lam. Wikichat: A few-shot llm-based chatbot grounded with wikipedia. _arXiv preprint arXiv:2305.14292_ , 2023. 

- Sahel Sharifymoghaddam, Shivani Upadhyay, Wenhu Chen, and Jimmy Lin. Unirag: Universal retrieval augmentation for multi-modal large language models. _arXiv preprint arXiv:2405.10311_ , 2024. 

- Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Xi Victoria Lin, Noah A Smith, Luke Zettlemoyer, Wen-tau Yih, and Mike Lewis. In-context pretraining: Language modeling beyond document boundaries. In _Proceedings of ICLR_ , 2024a. 

- Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In _Proceedings of NAACL_ , pp. 8364–8377, 2024b. 

- Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of CVPR_ , pp. 8317– 8326, 2019. 

- Cheng Tan, Jingxuan Wei, Linzhuang Sun, Zhangyang Gao, Siyuan Li, Bihui Yu, Ruifeng Guo, and Stan Z Li. Retrieval meets reasoning: Even high-school textbook knowledge benefits multimodal reasoning. _arXiv preprint arXiv:2405.20834_ , 2024. 

- Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A Dataset for Document Visual Question Answering on Multiple Images. In _Proceedings of AAAI_ , pp. 13636–13645, 2023. 

- Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi`ere, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ , 2024. 

- Rub`en Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for Multipage DocVQA. _Pattern Recognition_ , 144:109834, 2023. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ , 2023. 

- Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al. Freshllms: Refreshing large language models with search engine augmentation. _arXiv preprint arXiv:2310.03214_ , 2023. 

- Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024. 

- Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023. 

- Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. _arXiv preprint arXiv:2311.17136_ , 2023. 

- Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. _TMLR_ , 2022. 

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packaged resources to advance general chinese embedding. _arXiv preprint arXiv:2309.07597_ , 2023. 

14 

Preprint. Work in progress. 

- Peng Xu, Wei Ping, Xianchao Wu, Zihan Liu, Mohammad Shoeybi, and Bryan Catanzaro. Chatqa 2: Bridging the gap to proprietary llms in long context and rag capabilities. _arXiv preprint arXiv:2407.14482_ , 2024a. 

- Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and highresolution images. _arXiv preprint arXiv:2403.11703_ , 2024b. 

- Zhipeng Xu, Zhenghao Liu, Yibin Liu, Chenyan Xiong, Yukun Yan, Shuo Wang, Shi Yu, Zhiyuan Liu, and Ge Yu. Activerag: Revealing the treasures of knowledge via active learning. _arXiv preprint arXiv:2402.13547_ , 2024c. 

- Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation. _arXiv preprint arXiv:2401.15884_ , 2024. 

- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_ , 2024. 

- Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-V: A GPT-4v Level MLLM on Your Phone. _arXiv_ , abs/2408.01800, 2024. 

- Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. Augmentation-adapted retriever improves generalization of language models as generic plug-in. In _Proceedings of ACL_ , pp. 2421–2436, 2023. 

- Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid Loss for Language Image Pre-Training. In _Proceedings of ICCV_ , pp. 11941–11952, 2023. 

- Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, et al. Map-neo: Highly capable and transparent bilingual large language model series. _arXiv preprint arXiv:2405.19327_ , 2024. 

- Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. _ArXiv preprint_ , 2023. 

- Tianshuo Zhou, Sen Mei, Xinze Li, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu, Yu Gu, and Ge Yu. MARVEL: unlocking the multi-modal capability of dense retrieval via visual module plugin. In _Proceedings of ACL_ , pp. 14608–14624, 2024. 

- Kunlun Zhu, Yifan Luo, Dingling Xu, Ruobing Wang, Shi Yu, Shuo Wang, Yukun Yan, Zhenghao Liu, Xu Han, Zhiyuan Liu, et al. Rageval: Scenario specific rag evaluation dataset generation framework. _arXiv preprint arXiv:2408.01262_ , 2024. 

15 

Preprint. Work in progress. 

## A DATA CONSTRUCTION DETAILS 

## A.1 SYNTHETIC DATA 

Table 4: Statistics of crawled documents. We prompt GPT-4o to generate queries on these documents. 

|ments.||||
|---|---|---|---|
|**Name**|**Source**|**Description**|**# Pages**|
|Textbooks|https://openstax.org/|College-level textbooks including various subjects|10,000|
|ICML Papers|ICML 2023|ICML papers on various topics|5,000|
|NeurIPS Papers|NeurIPS 2023|NeurIPS papers on various topics|5,000|
|Manuallib|https://www.manualslib.com/|Manuals of various kinds of products|20,000|



To augment the training dataset of VisRAG, we gather additional documents from the web and utilize GPT-4o to generate queries based on these documents. The sources of the collected documents are listed in Table 4. The prompt employed is shown in Figure 6. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0016-06.png)


**----- Start of picture text -----**<br>
Hello, I have a super rich document library. Assume you are a curious but very ignorant<br>human. You often ask me questions (queries) to seek a precise document as a<br>reference for your question or request.<br>- Now, you have received another task:<br>- Here is a document image. This is a reference (target) that I provided from the<br>rich document library based on your query. Your task now is to imagine various<br>different angles of questions that I might ask.<br>### Your goal is to accurately find this document target as a potential reference<br>document candidate through queries in a very rich document library.<br>### The questions I ask might need references from the text, images, charts, or<br>implicit meanings in the document.<br>### Maximum number of query-answer pairs is 6.<br>Below is your output format:<br>‘‘‘json<br>{<br>"result":[<br>{<br>"answer": "",<br>"query" : ""<br>},<br>{<br>"answer": "",<br>"query" : ""<br>},<br>...<br>{<br>"answer": "",<br>"query" : ""<br>}<br>]<br>}<br>‘‘‘<br>{{ document }}<br>**----- End of picture text -----**<br>


Figure 6: Prompt for GPT-4o to generate queries, where _{{_ document _}}_ is the document page. 

## A.2 QUERY FILTERING 

As mentioned in Sec. 3.3, a significant portion of queries in VQA datasets are context-dependent that are unsuitable for retrieval. We prompt llama-3-8b-instruct (AI@Meta, 2024) to filter out such queries using the prompt in Figure 7, which includes human-annotated samples from DocVQA. 

## B DOCUMENT PARSING 

In this paper, we experiment with two categories of document parsing strategies: pipeline-based parsing and model-based parsing. 

16 

Preprint. Work in progress. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/rag/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents_images/iclr_2025_VisRAG_Vision-based_Retrieval-augmented_Generation_on_Multi-modality_Documents.pdf-0017-01.png)


**----- Start of picture text -----**<br>
I have some QA data here, and you can observe that the questions can be divided into<br>two categories:<br>The category #A: When you see this question alone without a given document, you are<br>sure to find a unique document in a corpus to provide a unique answer.<br>The category #B: When you see this question alone without a given document, you will<br>find hard to locate a document to give a deterministic answer for this question,<br>because you will find multiple candidate documents in a corpus, which may lead to<br>different answers for this question.<br>The number mentioned on the right of the leftside margin? #B<br>What is the date mentioned in the second table? #B<br>What is the full form of PUF? #A<br>What is the number at the bottom of the page, in bold? #B<br>Who presented the results on cabin air quality study in commercial aircraft? #A<br>What is the name of the corporation? #B<br>To whom this is addressed? #B<br>How many one-on-one interviews were completed during April 10th through the April 12th?<br>#A<br>What is the subject of the document/letter? #B<br>Who sent the letter? #B<br>Heading of the document? #B<br>What is the slope mentioned in the first table? #B<br>Where were the two magnesium containing papers made at? #A<br>what is the date in the letter? #B<br>What is the date mentioned in the letter? #B<br>Which part of Virginia is this letter sent from? #B<br>who were bothered by cigarette odors? #A<br>which cigarette would be better if offered on a thicker cigarette? #A<br>Cigarettes will be produced and submitted to O/C Panel for what purpose? #A<br>What is the heading of first table? #B<br>What is RIP-6 value for KOOL KS? #A<br>Which hetero-atoms does polar compounds contain? #A<br>One variable that has implicitly not been controlled? #B<br>Which corporation’s letterhead is this? #B<br>what is the contact person name mentioned in letter? #B<br>what is the date mentioned in this letter? #B<br>Another model of the 83mm with zero ventilation will be made at Semiworks within how<br>many weeks? #A<br>Hand sheets were made utilizing a 30% level of which component? #A<br>What is the source? #B<br>What is the heading of the document? #B<br>What is the subject? #B<br>Which test is used to evaluate ART menthol levels that has been shipped? #A<br>How much percent had not noticed any difference in the odor of VSSS? #A<br>What is the cigarette code of RIP-6(W/O Filter) 21/4SE? #A<br>What is the meeting date? #B<br>what is the subject of this letter? #B<br>what is the index for Retention of Franchise? #B<br>What is the heading of second table? #B<br>What is the full form of POVC? #A<br>what mm Marlboro Menthol were subjectively smoked by the Richmond Panel? #A<br>What sort of communication/letter is this? #B<br>According to the listed requirements, what must be the age group of female smokers? #A<br>How many one-on-one interviews were completed during April 10th through the April 12th?<br>#A<br>During the process of prototype production and ringtipping, some cigarettes were<br>observed to have burn holed in which paper? #A<br>How many distinct mechanisms appear to play a role in the breakup of a smoke column<br>into a multi-dimensional flowfield? #A<br>Where was the conference held? #B<br>Who is in cc in this letter? #B<br>Under BOLD, primary production of Blend #24- will be completed by which date? #A<br>{{ query }} #<br>**----- End of picture text -----**<br>


Figure 7: Prompt for llama3-8b-instruct to classify queries, where _{{_ query _}}_ is the query to be classified. Label B denotes context-dependent queries. 

17 

Preprint. Work in progress. 

## B.1 PIPELINE-BASED PARSING 

We consider the following document parsing pipelines: 

**Pytesseract.** Pytesseract is a Python wrapper for Google’s Tesseract OCR engine, offering a straightforward interface for text extraction from images. Unlike more complex methods, Pytesseract requires minimal pre-processing. By invoking the image ~~t~~ o ~~s~~ tring function, OCR is performed in a single step, directly returning the extracted text. Tesseract internally handles bounding boxes, confidence scores, and orientation correction. 

**PPOCR-based Methods.** PaddlePaddle OCR (PPOCR) (Du et al., 2020) is widely used for document text extraction, covering text detection, classification, and recognition. First, a text detection model identifies text regions and generates bounding boxes. These regions are then processed by a classification model to correct orientation issues like rotation or flipping. Next, a recognition model extracts the textual content from the corrected bounding boxes, returning recognized text with confidence scores. Only results with confidence scores above 0.6 are retained, and the bounding box coordinates, along with the recognized text, are stored for further processing. We apply the following strategies to obtain the final parsing result: 

- Adjacent Merging: To enhance text coherence, this policy combines adjacent text boxes based on vertical proximity (within 15 pixels) and horizontal alignment (within 100 pixels), reducing text fragmentation. This iterative merging process consolidates eligible text boxes into unified bounding boxes with concatenated text. Finally, the text from the remaining bounding boxes is combined with line breaks to produce the final result. 

- Layout Preserving: This policy maintains the original document structure by ordering text boxes based on their spatial positions. Spaces and line breaks are dynamically inserted to reflect horizontal and vertical gaps between text regions. This approach ensures that the extracted text mirrors the original document layout, preserving its formatting in the final result. 

We run the aforementioned pipelines on our dataset to obtain text-based training and evaluation data, and fine-tune a MiniCPM retriever to assess performance. The results are presented in Table 5. Methods based on PPOCR demonstrate significantly better performance compared to pytesseract, with adjacent merging and layout preserving yielding similar results. Consequently, we opt to use the adjacent merging policy for our “(OCR)” runs. 

Table 5: Overall retrieval performance of different document parsing pipelines. 

||**ArxivQA**|**ChartQA**|**DocVQA**|**InfoVQA**|**PlotQA**|**SlideVQA**|**Average**|
|---|---|---|---|---|---|---|---|
|(c)In-domain: _Models Fine-tuned on Synthetic and In-domain data_||||||||
|MiniCPM (Pytesseract)<br>MiniCPM (Adjacent Merging)<br>MiniCPM(Layout Preserving)|33.70<br>**47.36**<br>45.26|49.69<br>54.43<br>**55.78**|70.43<br>**74.13**<br>73.75|74.07<br>80.11<br>**80.22**|36.11<br>**41.33**<br>40.97|80.40<br>**90.49**<br>90.22|57.40<br>**64.64**<br>64.37|



## B.2 MODEL-BASED PARSING 

In addition to pipeline-based methods, we also employ a model-based parsing approach using MiniCPM-V 2.0 to directly transcribe document images into text. This method is referred to as “(Captioner)”. 

To train this model, we collect data from two sources: a) ALLaVA (Chen et al., 2024a) (image, caption) pairs, and b) VQA documents with descriptions generated by GPT-4V. We use the prompt in Figure 8 to instruct GPT-4V to generate detailed descriptions of documents from DocVQA, ChartQA, SlideVQA, InfographicsVQA, TextVQA (Singh et al., 2019), and ArxivQA. 

We train MiniCPM-V 2.0 with a batch size of 2048 and a learning rate of 5e-6 for 1 epoch. 

18 

Preprint. Work in progress. 

Based on the layout information, output the text in the image. Try not to modify the text, but you need to indicate the structure such as title, body text, subtitle, table, etc. Note: If there are charts or graphs, they should be described in detail. If you feel that there are more than 4000 words or most of the text in the image is unclear or most of the text contents in the image are not written in English, then directly return <none>. {{ document }} 

Figure 8: Prompt for GPT-4V to generate page description, where _{{_ document _}}_ is the document page. 

## C MODELS USED IN THIS PAPER 

**MiniCPM** (Hu et al., 2024) is a large language model (LLM) with 2.4 billion non-embedding parameters, demonstrating capabilities comparable to much larger models, such as Llama2-7B (Touvron et al., 2023) and Gemma-7B (Team et al., 2024). In this paper, we employ MiniCPM to construct the baseline text-based retriever (Table 2) and generator (Table 3). 

**SigLIP** (Zhai et al., 2023) is a CLIP-style multi-modal model designed to align text and vision representations. We utilize SigLIP-400m, released by Hugging Face[2] , which incorporates Flash Attention 2, increases maximum resolution to 980x980, and adopts the NaViT strategy to allow (a) variable resolution images and (b) aspect ratio preserved images. In this paper, SigLIP is used to develop the baseline vision-based retriever (Table 2). 

**MiniCPM-V 2.0** (OpenBMB, 2024a; Yao et al., 2024) is a vision-language model (VLM) with 2.8 billion non-embedding parameters, built upon SigLIP-400m and MiniCPM. It can process single images up to 1.8 million pixels (e.g., 1344x1344) at any aspect ratio. We use MiniCPM-V 2.0 to build VisRAG-Ret (Table 2) and VisRAG-Gen (Table 3(b)), as well as the document parsing model. 

**MiniCPM-V 2.6** (OpenBMB, 2024b; Yao et al., 2024) is an upgrade of MiniCPM-V 2.0 and MiniCPM-Llama3-V 2.5 (Yao et al., 2024). It is built upon SigLIP-400M and Qwen2-7B (Yang et al., 2024) with a total of 8.5B parameters, exihibiting a significant performance improvement over MiniCPM-Llama3-V 2.5 (Yao et al., 2024). Different from previous models, MiniCPM-V 2.6 can accept multiple images as the input and perform multi-modal in-context learning. It also demonstrates stronger OCR capabilities. We use MiniCPM-V 2.6 to build VisRAG-Gen (Table 3) and a text-based generation baseline MiniCPM-V 2.6 (OCR) (Figure 3, Figure 5). 

Note that, MiniCPM-Llama3-V 2.5 (Yao et al., 2024) is not used in this paper. 

**GPT-4o** (OpenAI, 2024) is OpenAI’s latest multi-modal model, capable of processing any combination of text, audio, image, and video inputs and generating outputs in text, audio, and image formats. We use GPT-4o to construct VisRAG-Gen (Table 3) and to synthesize training data. 

## D ADDITIONAL RESULTS 

Table 6 presents the retrieval performance in Recall@10. 

## E PROMPTS FOR GENERATION 

We present the prompts of VisRAG-Gen and TextRAG-Gen in Table 7. 

> 2https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit 

19 

Preprint. Work in progress. 

Table 6: Overall retrieval performance in Recall@10. 

|**Model**|**# Para.**|**ArxivQA**|**ChartQA**|**DocVQA**|**InfoVQA**|**PlotQA**|**SlideVQA**|**Average**|
|---|---|---|---|---|---|---|---|---|
|(a)|||Off-the-shelf Models||||||
|BM25 (OCR)<br>bge-large (2023) (OCR)<br>NV-Embed-v2 (2024) (OCR)<br>SigLIP(2023)|n.a.<br>335M<br>7.85B<br>883M|42.22<br>35.84<br>**56.11**<br>18.16|56.69<br>49.58<br>**65.32**<br>49.03|84.67<br>68.71<br>**89.52**<br>43.27|79.77<br>84.65<br>**93.89**<br>61.83|50.51<br>46.65<br>**52.42**<br>28.11|89.34<br>88.35<br>**96.54**<br>66.99|67.20<br>62.30<br>**75.63**<br>44.57|
|(b)Out-of-domain|||: _Models Fine-tuned on Synthetic Data_||||||
|MiniCPM (OCR)<br>MiniCPM (Captioner)<br>SigLIP (2023)|2.72B<br>2.72B<br>883M|47.06<br>47.27<br>49.51|56.69<br>60.86<br>63.65|86.06<br>81.32<br>78.61|90.13<br>88.91<br>82.16|39.52<br>37.11<br>41.36|93.74<br>91.25<br>93.03|68.87<br>67.79<br>68.05|
|<br>VisRAG-Ret|3.43B|**72.52**|**64.76**|**90.10**|**95.21**|**47.42**|**95.71**|**77.62**|
|(c)In-domain: _Models Fine-tuned on Synthetic and In-domain data_|||||||||
|MiniCPM (OCR)<br>MiniCPM (Captioner)<br>SigLIP (2023)<br>ColPali (2024)|2.72B<br>2.72B<br>883M<br>2.92B|59.21<br>60.81<br>67.12<br>76.46|69.22<br>67.83<br>**75.91**<br>70.19|89.36<br>85.68<br>84.62<br>**96.06**|92.72<br>91.25<br>90.81<br>93.65|61.94<br>58.42<br>**62.04**<br>48.62|95.74<br>93.50<br>95.00<br>96.97|78.03<br>76.25<br>79.25<br>80.33|
|<br>VisRAG-Ret|3.43B|**80.41**|72.98|92.97|**96.33**|61.47|**97.03**|**83.53**|



Table 7: Prompt templates for generation. “Others” refers to all VQA datasets except ArxivQA. 

||**TextRAG**|**VisRAG**|
|---|---|---|
|**ArxivQA**|Hint: _{{_parsed document(s)_}}_<br>Question: _{{_query_}}_<br>Options:<br>A._{{_Option 1_}}_<br>B._{{_Option 2_}}_<br>C._{{_Option 3_}}_<br>D._{{_Option 4_}}_<br>Answer directly with the letter of the correct option as<br>the frst character.|_{{_document(s)_}}_<br>Question: _{_query_}}_<br>Options:<br>A._{{_Option 1_}}_<br>B._{{_Option 2_}}_<br>C._{{_Option 3_}}_<br>D._{{_Option 4_}}_<br>Answer directly with the letter of the correct option as<br>the frst character.|
|**Others**|Image:_{{_parsed document(s)_}}_<br>Answer the question using a single word or phrase.<br>Question:_{{_query_}}_<br>Answer:|_{{_document(s)_}}_<br>Answer the question using a single word or phrase.<br>Question:_{{_query_}}_<br>Answer:|



## F CASE STUDY 

We show two cases in Table 8 and Table 9. In both instances, we compare VisRAG with TextRAG, maintaining the same setup as described in the “End-to-end Performance” paragraph in Sec. 5.1. 

In the first case from DocVQA, the user queries about “Club Jetty,” however, the term “Club Jetty” in the relevant document is not successfully extracted due to its decorative font. This leads to TextRAG failing to retrieve the document, while VisRAG successfully retrieves it. 

In the second case from InfographicsVQA, although both TextRAG and VisRAG successfully retrieve the document, TextRAG generates an incorrect response due to the loss of layout information, making it unclear which number (53% or 49%) pertains to Europe. VisRAG effectively utilizes the layout information and generates the correct answer. 

20 

Preprint. Work in progress. 

Table 8: Case study from DocVQA. In this case, VisRAG successfully retrieves the ground-truth document, while TextRAG fails, leading to VisRAG’s correct generation and TextRAG’s incorrect generation. 

**TextRAG VisRAG Query** On which day is Club Jetty closed? **Retrieved Top-1 Document** ✗ **Incorrect** ✓ **Correct** SMOKERS _←�_ EXPRESS _←�_ Express _←�_ Airlines _←�_ Yes that’s right. An Airline for _←�_ smokers is coming! But you _←�_ say, **Parsing ResultDocument** takingbers only.pectFullintificationV.I.P.great fun newAirlines(407)Certificatetherealincludes102City with every 7th andmost attentiveWithlife-time.interesting.ageIf you would like morerestaurantsshopping guidebers onlyairSSpaceplanutiveLasmenDecemberrangementscentlytion to ”Let’s Party”isRichardson,supportnewsmentarycountscountNoproblem.memberstheyRentaltheflight _←←_ freight _←_ screamingVegas,privileges.beverageoftocan’twho _��_ airportthe _←�_ andLoungescruise783-6124brainchildreconditionedCoast _←_ FAA), Smokers Express isonor older.25 South Atlantic Avenue _←_ serve _←�_ weyou _←_ newspaper _←_ car _←�_ process.only:Smokers _←_ helpapplications _�_ area _←�_ carryinglike _←_ ratesdoofoperations. _���_ andtoE _←_ a _�_ V Free Lotto ticket for each passengerpackagesmaya _�_ discounts _←_ t Real food for real people—SteaksMembershipcanairport _�←←_ lastwithserviceY Special party flights to Las Vegasthat,Free _←_ numberB _←_ Membership is restricted _�_ babiesleaseattractionsat _←←_ of _�_ to _�_ Take a look at _←�_ WithAtlanticincludesmokers airline where membership _��←_ call _←_ Smokersstaff of employee/owners _��_ seeflightcoupleyear.U Great contests and sweepstakes _�_ consultant,The _←_ Expressnon-stop _←←_ Awhat _�_ X _←_ headphones _�_ smokersofDC-9s. _←←←�_ Thoseor _�_ smoke.a _←�←�_ SmokersSmokersDiscountfor _←_ These _←_ ofN _�_ William _�_ the official Smokersfirst _←�_ nearinitiatinglittleup _��_ Orlando,City _←_ H _�_ (membersabout _�_ fromofinformation about SmokersFree _�_ inspecialTpersonnel11th flight freeExpressThe _←_ tois _�←_ write:Real50,000areSmokersserviceCocoatheCocoa _←←←_ pbitpositions _←��_ 29 _←←_ Smokers _←_ destinationInflightTheySome of the destinations _��_ Express _�←_ couponsclubWalts _�_ theinclub _←←←��_ ashtraysa _←�_ (Expresswhat members will receive:Bryantspeciallyperks _�_ Atlanta,the _��_ club, _�_ ExpressstyleF _←←_ Beach, _�_ membershipsmustBeachGoldABC’sofCocoa Beach, FL 32931beginning the FAAairportsisorganizedcostsExpressSmokersFree _�←�_ and _←_ FAA _←_ phoneluckfromTravelheadquarteredforNumbered _�_ providing _←�_ byfor _←�_ Chestnutbe _←←_ Travelare _←←_ to persons 18 years ofareainflightTraveland _←_ equipped _�_ whichofpractically _�_ Chicago,Airlines _←��_ andthisregulations?destination _��_ 18)SmokerZ The best trained,Space _�_ in the industry.service _←_ availablediscountGeorge _←_ traveling _←_ $25 _�_ FloridaExpress _←_ Smokers Expressmaps _←←←�←_ D _�_ your _�_ summer. _�_ areMagazine)R _←_ & Burgers _←_ you _���_ hasmovies _←_ service _�_ annually(formerlyFreetheand AtlanticCoast _←←_ W DiscountA _←_ M _←←�_ Card _←�_ making _�←_ Express”Mickey”does havestrong _←�_ Members _�_ for mem-everybusiness- _�_ subscrip-made _←_ home _��_ will _�←_ to _←_ Compli-club, _�_ Expressthey _←_ smokesandODallas, _�_ Qcharter _�←←←←��_ atmem-exec-Suite _�_ onCer-Dis-Dis-area _��_ find _��_ and _←←←←←←←←←_ Notheas-re-ar-ofintoGKisCa _���������_ eIceTheWaipouli,PLACEpick-up-LihueHALEKOASIAJOEIMPORTS822-99817:30p.m.aSTEAK-LOBSTER-MAHIMAHIandcuisineFreshandlovehavetheCLOSEWaipouli,worldandverynightandsmallBlanca....chops”HOMEEXCURSIONSSurfPlantationComplimentarydinner11Special introductoryfromRESTAURANTyouOPENFXPLOREKAUAIKAUAIWINDSURFING _←_ Sandwichesspecialtya.m. _←_ Cream _←_ island. _�←_ live _←_ weMarketfreeext.Cabaret _←_ p.m. _←←_ fromthethe _�_ Fruitand _←���←_ only) _←�←_ with _←��_ COFFEEKauaiofferClosed _�_ LATEfriendlyhours. _←_ hotel _←_ STYLE _�←←_ 7830 _�_ mostroaster _�_ in _�←←←_ Kapaa, _←_ ROADIandKauai3 _�_ Daquiris.Learntheentertainment _�_ of _��_ PAPERBACKPlace _←←←�_ own _���_ lots36Lihue, _←_ Waipouli,AfterKauai’s.MUUMUUS.5:30-10:30carea _←←�←_ and _�_ Over _←�_ thepickup _←←_ COCONUTOh Yes, we specialize _�_ complete28TUESDAYSBerryo Fresh Fruit _←←_ transportationPhone: _�←��_ of _←_ NawiliwiliMacadamia _�_ andone _��_ Kauai _←_ AtCOOKING” _←_ The _��_ TheRENTALShavehouseBestsellersLatitude 20/November 1978 _�←_ to _�_ dinnerOnfresh _←�_ otherKauai.5,000 _�_ fern822-3216Coconut245-9181& _←←�_ casualonlyof245.4970....after(We _�_ MarketLesson RateWindsurf _←_ KauaiMarket _←��_ thetheCallNUT _←←_ 822-4311theforin _←_ VELVET _�_ selectiongrottofrom _�←_ Dance _←��←_ booksentreeslateOPENmail _�_ our _←←_ banksSOUVENIRS _←_ Baydinnerlocalatmosphere.COMPLETEmostus _�←�←←←�←_ PlantationHUT _←←←��←_ Place-shop _�_ It’stime _�_ Place _←���_ mostforOASIS _�_ PLANTATION-EXPERIENCEISNut _←←_ DrinksMUSIC _�←���_ coffee tea nuts spices herbsgiftPhone: _←_ BandROASTINGGames _←←�_ in _←_ seafoodSUGAR _�_ (from _�_ CLUBIETTY7 _←←_ famous _�_ gourmetofcalledreservationsPAINTINGS.with _�←_ WAILUA _�_ of _�_ Waffle _�_ AMstock. _��_ CANTONESEresorts.(certifiedatpaks)uniqueNOW! _�_ the _←_ on your wayhours _←_ ofEASYyourHi, _←_ DINNER: _←_ &in Italian and French _←←_ Coconutto245-9290 _��_ M-S;Wailua _←�_ 39 _�_ paperbackthe _←_ andMENU-includingDISCO _←_ e Cold _�_ lounge _�_ Wailua _←_ bakedmyMILL484BrunchDinedining _�←←_ life245.3856We’reHawaiiana _��←�_ restaurants _←_ lunchHANDICRAFTSCasa _��_ FreshKauai’sOPENLEARNING _←_ Sun. _←_ Windsurfingname _�_ at _←_ Kuhio _�_ atinstruction) _←_ I _←_ T-SHIRTS& _�_ Plantation _←_ MARINA _�_ areauntil _←_ CENTERSNACKS _�_ 245-9181stuffedknow _←_ theRiverservice _��_ onatopen _�_ dailyBlanca,DanceRestaurant _�_ MARKETFruitDrinksFOOD _�_ books85:30-9:45 _←_ isCourtesy _←_ ortheEARLY-Hwy.SundaysCoconutfresh _←_ theAMHotels- _←��_ SunnyinKauaiyou’llNOW _�←_ every2989fromCasaporkonly _�_ inwee _←←←←←←←←←←←←←←←←←_ by-Wethe _�_ ontoatoa _�����������������_ Great tasting munchies for happy _←�_ hour. _←�_ American Smoker’s Journal _←�_ 38 WINTER ISSUE **Answer** Mondays ✗ **Incorrect** Tuesdays ✓ **Correct** 

21 

Preprint. Work in progress. 

Table 9: Case study from InfographicsVQA. In this case, both VisRAG and TextRAG successfully retrieve the correct document; however, only VisRAG effectively leverages the layout information, enabling accurate generation. In contrast, TextRAG suffers from information loss of the layout, resulting in incorrect responses. 

||**TextRAG**|**VisRAG**|
|---|---|---|
|**Query**|Whatpercent of account holders in Europe are usingLinkedIn for fnding job?||
|**Retrieved**<br>**Top-1 Document**|✓**Both Correct**||
|**Document**<br>**Parsing Result**|Social media_←�_job seeking trends_←�_Michael Page’s annual global survey of fnancial services<br>and banking_←�_employees was conducted in April 2014,more than 3,300 people participated_←�_<br>Linkedln_←�_Linkedin’s popularity continues to grow, though many job seekers don’t think of it<br>as part of_←�_their strategy.So hirers need to look to other sourcing channels too_←�_What pro-<br>portion of account holders_←�_use Linkedin for job seeking?_←�_93_←�_%_←�_30%_←�_of respon-<br>dents have_←�_anaccount-up_←�_10% from last year_←�_more women_←�_than men say_←�_they don’t<br>have_←�_an account_←�_53%_←�_In Europe_←�_49%_←�_In North America_←�_40%_←�_In the UK_←�_<br>Facebook_←�_Despite last year’s hype around Graph Search,Facebook hasn’t made any progress with<br>monetising_←�_its recruitment potential -jobseekers remain very negative about Facebook playing any<br>part_←�_13%_←�_said they’d be happy_←�_to see adverts_←�_92%_←�_said they would not be_←�_happy<br>to be contacted by_←�_a recruiter on Facebook_←�_1%_←�_Don’t bank on social media – Michael<br>Page brings you a broader range of talent, and jobs_←�_www.michaelpage.com.au/salarycentre_←�_of<br>respondents_←�_(who are job seekers) said they_←�_would use it to look for jobs_←�_MichaelPage_←�_<br>Financial Services_←�_Specialists in fnancial services recruitment_←�_www.michaelpage.com.au_←�_||
|**Answer**|49%✗**Incorrect**|53%✓**Correct**|



22 

