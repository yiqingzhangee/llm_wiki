# **A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions** 

**Jack Urbanek** _[∗][,][†]_ , **Florian Bordes** _[∗][,]_[1] _[,]_[2] _[,]_[3] , **Pietro Astolfi**[1] , **Mary Williamson**[1] , **Vasu Sharma**[1] , **Adriana Romero-Soriano**[1] _[,]_[2] _[,]_[4] _[,]_[5] 

> 1FAIR at Meta, 2Mila, 3Universite de Montreal, 4McGill University, 5Canada CIFAR AI chair 

> _∗_ Joint first author, _†_ Work done at Meta 

Curation methods for massive vision-language datasets trade off between dataset size and quality. However, even the highest quality of available curated captions are far too short to capture the rich visual detail in an image. To show the value of dense and highly-aligned image-text pairs, we collect the Densely Captioned Images (DCI) dataset, containing 7805 natural images human-annotated with maskaligned descriptions averaging above 1000 words each. With precise and reliable captions associated with specific parts of an image, we can evaluate vision-language models’ (VLMs) understanding of image content with a novel task that matches each caption with its corresponding subcrop. As current models are often limited to 77 text tokens, we also introduce a summarized version (sDCI) in which each caption length is limited. We show that modern techniques that make progress on standard benchmarks do not correspond with significant improvement on our sDCI based benchmark. Lastly, we finetune CLIP using sDCI and show significant improvements over the baseline despite a small training set. By releasing the first human annotated dense image captioning dataset, we hope to enable the development of new benchmarks or fine-tuning recipes for the next generation of VLMs to come. 

**Date:** June 18, 2024 **Correspondence:** fbordes@meta.com **Code:** https://github.com/facebookresearch/DCI 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0001-13.png)


## **1 Introduction** 

State-of-the-art vision-language models (VLMs) are often trained on large scale datasets such as LAION-400M (Schuhmann et al., 2021), YFCC100M (Thomee et al., 2016), or other undisclosed datasets crawled from the web. These datasets are formed by collecting images from the web and using alt-text (or other local text on the webpage) to create loose image-text pairs. These can then be filtered down trading off on quantity for quality (Sharma et al., 2018; Radenovic et al., 2023). Still, recent work has demonstrated that throwing these loose captions out entirely in favor of generated captions, with enhanced quality and density, can produce improved results (Doveh et al., 2023a). Other works (Li et al., 2023b,c; Xu et al., 2023a; Abbas et al., 2023) have demonstrated that it is possible to get CLIP-level performance using a vastly reduced compute, often by throwing away portions of the data resulting in more balance between image and text modalities. However, those approaches rely on automatic pipelines which do not generate reliable and long captions that can capture rich visual 

details in an image. From this it appears no existing dataset has high-quality image descriptions that are tightly-coupled enough with the image to train for or evaluate a deep alignment between the two domains. 

In the absence of high quality captions to evaluate VLMs, benchmarks such as ARO (Yuksekgonul et al., 2023) and VL-Checklist (Zhao et al., 2023) often complement image-caption pairs with hard negatives that are generated by slightly altering the initial (positive) description. Progress on these benchmarks has been rooted in training VLMs with negatives of similar construction to the tests (Yuksekgonul et al., 2023) rendering the methodologies ineffective on datasets such as Winoground (Thrush et al., 2022). Recent works (Lin et al., 2023) have called the evaluation capacity of many of these benchmarks into question, given how effective language-prior-based methods perform. More specifically, given the unlikeliness of the hard negative captions in these benchmarks, a good text encoder can achieve close to 100% accuracy without looking at the images. Moreover, Bordes et al. (2023) have shown that most improvements observed on ARO or VL-Checklist 

1 

do not translate on simple synthetic benchmarks for which the negative caption is as likely as the positive one. Since the use of VLMs is significantly increasing, it is crucial to make sure that we have a diverse suite of reliable benchmarks to asses their abilities. 

In this paper, we introduce the Densely Captioned Images dataset, a collection of 7805 images with dense and mask-aligned descriptions averaging above 1000 words each. One such example is provided in Figure 1, displaying just a subset of the collected text paired with their aligned masks. We demonstrate how to leverage this dataset to evaluate VLMs in two ways after summarizing captions to fit into CLIP’s 77 token limit, both with a negatives-based test as well as a novel matching task, referred to as _subcropcaption matching_ , that requires selecting appropriate captions for different regions of the same image. We evaluate existing baselines, and observe that no models perform well at both concurrently, and improved performance via negatives-based training comes at the cost of decreased performance on subcrop-caption matching. We also run some experiments using the summarized DCI as a fine-tuning dataset to evaluate the effectiveness of these captions for improving a model’s performance on other benchmarks, and compare the efficiency per-example to that from the automated annotation setup in DAC (Doveh et al., 2023a). 

To summarize, our contributions are: 

- We release the Densely Captioned Images (DCI) dataset, which contains dense and mask-aligned captions, alongside an LLM-summarized version (sDCI) containing captions under 77 tokens for use with current VLMs. 

- We provide a new benchmark for VLMs based on sDCI to evaluate fine-grained vision-language understanding, and show that no existing model can perform well at matching captions from within one image to corresponding sub-sections of that image. 

- We show that fine-tuning with high quality image-caption pairs is as good on ARO and VL-Checklist as fine-tuning on at least 10 _×_ the automatically annotated data, and that even without utilizing explicit negatives these pairs can improve performance on VL-C-Object from 81.17% to 88.37% . 

## **2 Related Works** 

The massive, loosely-labeled dataset approach that has enabled VLMs like CLIP (Radford et al., 2021) and powerful successors like BLIP2 (Li et al., 2023a), 

Flamingo (Alayrac et al., 2022), CM3leon (Yu et al., 2023), and many others, has been a clear forward step in vision-language modeling. Still recent benchmarks show that models trained in this manner display clear drawbacks in reasoning skills. Additional techniques have been proposed and adopted recently to close this gap, discussed below. 

_Vision-Language Datasets._ Over the last decade, there have been significant dataset collection efforts connecting images and text. Earlier works focused on curating datasets by leveraging human annotations, see _e.g._ , **COCO** (Chen et al., 2015), **Visual Genome** (Krishna et al., 2016), and **Flickr30k** (Young et al., 2014). The process resulted in high quality annotations, which were however oftentimes limited by the caption content – _i.e._ , relatively short phrases (5.1 to 10.3 words on average) grounded at image level or region level – and the data annotation scale (30k to 130k images). To increase scale, researchers gathered web-crawled data and introduced large scale datasets such as **YFCC100M** (Thomee et al., 2016), which contains 100M media objects. Yet, crawling the web oftentimes results in little correspondence between image and text pairs. To reduce noise between image and text pairs, efforts such as **SBU** (Ordonez et al., 2011) queried Flickr and filtered the noisy results, obtaining a _∼_ 1M images. Moreover, **Conceptual Captions** (CC) (Sharma et al., 2018) crawled a dataset of _∼_ 12M images and alt-text pairs, and included a protocol to filter noisy text-image pairs, resulting in 3M data points. Relaxing the filtering protocol allows to trade data quality for scale. Crawling alt-text also resulted in relatively short text descriptions with 10.3 words on average, which are most often grounded at image level. **Localized Narratives** (Pont-Tuset et al., 2020) was introduced as a dense visual grounding dataset leveraging a multi-modal annotation procedure, collecting _∼_ 850k text-image pairs with 36.5 words/caption on average. **RedCaps** (Desai et al., 2021) constituted another effort yielding large scale ( _∼_ 12M) web-curated data by exploring alternate data sources of high quality data instead of devising complex filtering strategies. **Wikipedia-based imagetext dataset** (WIT) (Srinivasan et al., 2021) extended dataset creation efforts by gathering a multilingual dataset of text-image-pairs consisting of _∼_ 11.5M images. **LAION-5B** (Schuhmann et al., 2022) further increased the web-crawling efforts by gathering a multilingual dataset of text-image pairs, and filtered the collected data with a pre-trained CLIP (Radford et al., 2021) model. Following, **LAION-CAT** (Radenovic et al., 2023) reduced noisy examples from LAION-5B by filtering for caption complexity, _i.e._ , captions that do not contain any action, and for 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0003-00.png)


**Figure 1** One example from the Densely Captioned Images dataset. Only part of the submask hierarchy is shown. 

3 

text spotting, _i.e._ , images that contain rendered text. **MetaCLIP** (Xu et al., 2023b) has also been released as an open dataset for reproducing CLIP. These very large scale datasets have been successfully used to advance the state-of-the-art of VLMs. 

_Vision-Language Evaluation Benchmarks._ Several recent advances in visual-language learning have focused on creating comprehensive benchmarks to evaluate model performance in more holistic ways. These benchmarks are instrumental in pushing the envelope of what VLM can understand and process, ensuring they move beyond superficial image-text matching towards genuine understanding of intricate relationships between visual and linguistic elements. In particular, **VL-CheckList** (Zhao et al., 2023) and **ARO** (Yuksekgonul et al., 2023) assess the VLM capabilities beyond average downstream task accuracy, by focusing on a model’s ability to understand objects, attributes, order or relations. ARO’s extensive scope, uncovers limitations in VLMs such as poor relational understanding and lack of order sensitivity. **Winoground** (Thrush et al., 2022) tests models for visio-linguistic compositional reasoning by asking VLM to match two images with two captions containing the same set of words but in different orders. This task requires models to discern the meaning conveyed by the order of words, reflecting different visual scenes. Current VLMs perform only marginally better than chance, highlighting a significant gap in compositional reasoning. **CREPE** (Compositional REPresentation Evaluation) (Ma et al., 2023) evaluates two aspects of compositionality: systematicity and productivity. Systematicity is measured by the model’s ability to represent seen versus unseen atoms and their compositions, while productivity gauges the model’s capacity to understand an unbounded set of increasingly complex expressions. Finally, **PUG** (Photorealistic Unreal Graphics) (Bordes et al., 2023) uses synthetic data to asses the compositional reasoning abilities of VLMs by progressively increasing the complexity of a given generated scene. One issue with these evaluation datasets is their frequent reliance on COCO, either directly as in ARO, or through Visual Genome as in VL-Checklist or CREPE. It is difficult to find an evaluation dataset of sufficient scale without COCO. 

_Vision-Language models._ Recent VLM advancements have built upon the foundational work of **CLIP** (Radford et al., 2021), which leveraged largescale image-text pairs to jointly pre-train an image encoder and a text encoder to predict which images are paired with which texts in a contrastive learning 

paradigm. **NegCLIP** build upon CLIP by leveraging negative captions when training. **BLIP** (Bootstrapping Language-Image Pre-training) (Li et al., 2022b) uses a new framework that bootstraps the captions from noisy web data for both understanding and generation tasks. Its successor **BLIP-2** (Li et al., 2023a) further streamlines the process by utilizing off-the-shelf frozen pre-trained image encoders and language models, bridging the modality gap with a lightweight querying mechanism. **Clip-rocket** (Fini et al., 2023) improves VLM baselines by showing that applying image and text augmentations makes up for most of the improvement attained by prior VLMs. **Flava** (Singh et al., 2022) proposes a foundation VLM model by combining existing VLMs objectives together with auxiliary in-modality losses for the text and vision encoders. **X-VLM** (Zeng et al., 2021) achieves success with a pretraining method matching sub-portions of the text to regions of the image at multiple granularities. These models introduces improvements over CLIP, focusing on efficiency, adaptability, and reducing the need for extensive labeled datasets, thereby pushing the boundaries of vision-language pre-training. The closest work to our approach is **DAC** (Densely Aligned Captions) (Doveh et al., 2023a), which improves with an automated LLM based pipeline the _caption quality_ and _density_ . By showing that DAC-enhanced CLIP models exhibit substantial gains on some benchmarks, this work underscores the critical role that caption quality and density play in the efficacy of VLMs. We build on this insight and explore how to further increase the caption quality and density by relying on human annotators, and analyze how that impacts downstream model performance. 

## **3 Dataset Construction** 

The Densely Captioned Images dataset, or **DCI** , consists of 7805 images from SA-1B (Kirillov et al., 2023), each with a complete description aiming to capture the full visual detail of what is present in the image. Much of the description is directly aligned to submasks of the image. An example is shown in Figure 1. In the top left we see the full image of a water pump, with an associated description. The italicized section is collected as a _standard caption_ , aiming to summarize the full image in about a sentence, similar to existing caption datasets. The remainder of that first description contains details about the relationship between visible entities in the image, as well as in-depth descriptions of regions that are not described as part of the submasks. All other text describing the image is associated with submasks of the image. Each submask has its own free-text label 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0005-00.png)


**Figure 2** Annotation view for writing description for masks of the image. The masked region appears highlighted for clarity. 

(not pictured) and description, and may also contain further submasks. Here for instance we see submasks for windows and balconies as being contained in the submask capturing three buildings in the background. 

## **3.1 Preparation** 

In order to collect the data, we first select images from a random privacy-mitigated subset of SA-1B. We then procedurally extract subregions of each image to annotate, as we found in initial trials that crowdsourcing both regions and descriptions concurrently overcomplicated the task and successful annotation rate. For this process, we turn to the Segment Anything Model (SAM) (Kirillov et al., 2023) and adapt their standard method to extract all masks from an image. 

For the extraction process, SAM usually relies on a grid of points across the entire image. In order to increase the possibility of selecting interesting regions worth annotating, we additionally apply a canny filter (Canny, 1986) and select random points within a radius from discovered edges. We then run SAM to detect all masks using both the grid and the near-edge points. Once the masks are returned, we establish a hierarchy of submasks by thresholding the number of overlapping pixels between two masks to determine if one is a submask of the other, or if the two masks should be joined. This helps reduce some of the noise introduced by the automatic masking process, and leaves us with a tree-like structure for the masks. Lastly, we remove any masks that are too small. We note that undergoing this process does not result in every detail of each image being selected as a candidate for annotation, and as such instances in the DCI dataset are not expected to have _complete_ submask-aligned coverage of all elements one could 

recognize in or discuss about an image. 

## **3.2 Collection Process** 

We use Mephisto (Urbanek and Ringshia, 2023) to host our task, pay crowdworkers to provide annotations on the dataset, and additionally run qualification steps. Workers that pass our qualifications are eligible to work on the main task which contains 3 stages: 

1. Workers are provided with the whole image, and asked to provide a short description of it. This is considered the _standard caption_ . 

2. Workers are provided with submasks of the image, one at a time starting with the leaves of the mask tree, displaying a SAM-selected region of the image as well as an indicator for where that region comes from. They are generally asked to provide a label and complete description for the pictured region, though are allowed to mark the region as ‘uninteresting’ and only provide a label, or ‘bad’ and provide nothing. These options allow us to focus worker time on useful annotations and help capture some of the noise of the automatic selection process. This is shown in Figure 2. For masks that contain submasks, workers are also provided with overlays that show the regions already annotated, and are asked to annotate in terms of what has already been written. 

3. After completing all the submasks, the worker is then shown the complete image again and asked to provide an overall description, paying attention to the relationship between previously annotated regions. 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0006-00.png)


**Figure 3** Example of a Llama2-generated summary and negative that comprise sDCI. Each image and submask have multiple summarizations and negatives. We also compare the caption quality between DAC (Doveh et al., 2023a) and DCI. In contrast to DCI that relies on human annotations, DAC used an automatic pipeline based on LLM for captioning. As we observe in this example, the DAC captions can suffer from hallucinations and miss important elements of the photo. In this work we argue that while improving automatic pipeline is an important research direction, for now the captions proposed are not reliable enough to be used to evaluate models and assess their abilities. 

An in-depth description of the filtering and quality assurance process can be found in Appendix B while the Datasheet (Gebru et al., 2021) is available in Appendix F. Complete annotation instructions, dataset download links as well as reproducible code are avaliable on our GitHub[1] . The DCI dataset is released under the CC-BY-NC license. 

## **3.3 Fitting DCI into 77 CLIP tokens** 

Ultimately, we collected an average of 1111 words (1279 CLIP tokens) per image, with a median of 941 words. This proves problematic for evaluating or fine-tuning CLIP-based VLMs given their maximum text token length of 77. Embedding pooling methods (Chen et al., 2018) to extend the effective input size for text modeling is an active research area (Chalkidis et al., 2022; Zhang et al., 2022), and current work suggests average-pooling embeddings over these longer descriptions would be ineffective. 

One possible approach would be to utilize the subsections of the image while providing the corresponding subcaption, in a manner akin to a multi-modal multicrop approach (Caron et al., 2021). Still, even when considering just the 91,424 submasks, the average to- 

> 1https://github.com/facebookresearch/DCI 

ken length is nearly 200 per caption. We instead use the longer context capabilities of Llama2 (Touvron et al., 2023) to summarize down the overall information in the image into CLIP-consumable portions. We generate multiple captions for each image and submask, using prompts that attempt to summarize down recursively until the result is in bounds. As this modification to the dataset is generated automatically, the summarizations may have introduced noise, and may not capture all of the detail in the full original captions. Summarizations also occasionally mix references or include context in a submask that isn’t the main focus. Still, the summaries are fairly high quality and more dense than those found in other datasets, especially when using more than one distinct summarization per image. We also prompt the LLM to generate negatives from these summaries, achieving a set of particularly hard negatives for CLIP to evaluate. We call this version of the dataset **summarized DCI (sDCI)** . Examples of full caption, _LLM summary_ and _LLM negative_ are included in Figure 3 and contrasted with DAC (Doveh et al., 2023a) data. More detail including the prompts used can be found in Appendix A. 

Ultimately, this fitting step produces a _lower bound_ on the level of vision-language understanding ‘res- 

6 

olution’ that the overall DCI dataset is capable of evaluating a model for. As newer models arise that are able to handle embedding much larger quantities of text content, it will be possible to make full use of DCI’s original annotated captions. 

## **3.4 Statistics** 

|Dataset|Imgs<br>Caps<br>Toks/Cap<br>Toks/Img|
|---|---|
|||
|DCI<br>DCI_sub_<br>sDCI<br>sDCI_sub_|7,805<br>7,805<br>1,282.09<br>**1,282.09**<br>96,007<br>96,007<br>199.33<br>199.33<br>7,805<br>87,268<br>49.21<br>536.00<br>96,007<br>714,630<br>36.60<br>263.01|
|||
|LN_COCO_<br>LN_COCO<_77|142,845<br>142,845<br>49.11<br>**49.11**<br>127,456<br>127,456<br>43.70<br>43.70|
|COCO|123,287<br>616,767<br>13.54<br>67.74|



**Table 1** Comparison of DCI dataset statistics to other datasets, focusing on average CLIP tokens per image or caption. Note the **26x** difference between DCI and the previous longest annotated dataset, Localized Narratives (LN). _sub_ denotes including submasks and their descriptions as examples, and sDCI refers to the LLMsummarized version of DCI that fits captions to 77 tokens (Sec. 3.3), while LN _COCO<_ 77 simply drops examples longer than 77 tokens ( _∼_ 10 _._ 8%). 

All-in-all the Densely Captioned Images dataset is far more dense than Localized Narratives on COCO images (Pont-Tuset et al., 2020) (later referred to as _LNCOCO_ ) and nearly 100 _×_ more dense than standard COCO captions (Chen et al., 2015). After reducing to CLIP-bounded summaries, it still contains more text density than both. Complete details can be found in Table 1. 

Here we see that the multiple-summarization method of sDCI produces fairly similar token per image values to the original dataset while keeping individual captions’ token lengths in bounds for CLIP. To get Localized Narratives into the 77 token bound, we simply drop longer examples. 

## **4 Evaluating VLMs with summarized DCI** 

we provide models with a crop around the selected masks, we call this _Subcrop-Caption Matching (SCM)_ , and we use a batch size of 8. We can run against our LLM-generated negatives as well. Given that LLM-summarization has provided us with multiple captions and negatives per image and submask, we supply the first unless noted otherwise. With this in mind, we construct 6 evaluations as follows: 

**[All SCM]:** Group each image with their subcrops, alongside one summarized caption per subcrop. Then use the model to find the most likely caption associated to each subcrop. This test measures the ability of the VLM to distinguish between the different parts that compose an image.[2] 

**[All Neg]:** Select one LLM summarized caption and the corresponding LLM-generated negative for each image and subcrop. Score a model on its ability to distinguish between the positive and negative. 

**[All Pick5-SCM]:** Use the same setup as **All SCM** , but rather than using only one caption per subcrop, we use 5 LLM generated captions per subcrop. We score a model as succeeding only when the worst-scoring positive caption scores higher than the best-scoring caption of any other image in the batch. This test evaluates if the representation space is structured such that captions belonging to a specific image are closest to the target image in the space. 

**[All Pick5-Neg]:** Use the same setup as **All Neg** , but rather than using one caption, we use 5 LLM summarized captions for each image and subcrop. If any of these captions score worse than the negative, the model fails the example. 

**[Base Neg]:** Using only the 7805 base images without subcrops, evaluate the model’s ability to distinguish between an LLM generated caption and its corresponding LLM-generated negative. Note, this is a strict subset of **All Neg** , though these captions are on the longer side on average and cover a different distribution. 

**[All Hard-Negs]:** Using the same setup as **All Neg** , but rather than using a single negative, use the negative across all LLM-generated negatives that CLIP scores highest. 

## **4.2 Results** 

## **4.1 Methodology** 

Using the 7805 images in the summarized Densely Captioned Images (sDCI) dataset, we construct a few different evaluations. As noted above, the ability to select multiple submasks from the same image and include them in the same batch allows us to create a CLIP-style test, wherein the model can evaluate a full batch of images and captions and score correctly which caption belongs to which image. As 

We compare in Table 2 the sDCI performances given by different state-of-the-art models: CLIP (Radford et al., 2021), NegCLIP (Yuksekgonul et al., 2023), BLIP (Li et al., 2022b), Flava (Singh et al., 2022) and 

> 2Since we used sDCI to fit current models token length, it is possible that some of the summaries remove the information that make possible to distinguish between the captions. Ideally this test should be performed on the non-summarized version once VLMs can handle 1000+ tokens. 

7 

|**Model**|**All**<br>**All Pick5**<br>**Base**<br>**All**<br>SCM<br>Neg<br>SCM<br>Neg<br>Neg<br>Hard Negs|
|---|---|
|CLIP Baseline (Radford et al., 2021)<br>40.06%<br>60.79%<br>11.21%<br>24.06%<br>67.56%<br>41.34%<br>NegCLIP (Yuksekgonul et al., 2023)<br>**43.35%**<br>56.00%<br>**13.22%**<br>4.82%<br>76.69%<br>50.84%<br>BLIP (Li et al., 2022b)<br>39.13%<br>54.02%<br>10.73%<br>5.51%<br>63.41%<br>53.23%<br>Flava (Singh et al., 2022)<br>38.08%<br>47.99%<br>8.01%<br>9.82%<br>11.6%<br>45.59%<br>X-VLM (Zeng et al., 2021)<br>38.45%<br>53.46%<br>10.96%<br>5.10%<br>44.29%<br>52.42%<br>DAC_LLM_(Doveh et al., 2023a)<br>37.45%<br>81.71%<br>8.13%<br>37.84%<br>**90.56%**<br>71.21%<br>DAC_SAM_ (Doveh et al., 2023a)<br>37.90%<br>**84.17%**<br>6.70%<br>**39.94%**<br>89.66%<br>**73.61%**||



**Table 2** sDCI test result: We compare existing baselines on our Subcrop-Caption Matching (SCM) and negatives tests. Additional results are available in Table 10 in the Appendix. We note our best model fine-tuned on sDCI from section 5 achieved 64.02% and 31.60% on a held-out test of **All SCM** and **All SCM Pick5** respectively, setting an upper bound for model performance. 

X-VLM (Zeng et al., 2021). Additional experiments on different architectures and pretraining datasets are available in Table 10 (see Appendix). The CLIP baseline starts at 40.12% on **All SCM** and 60.63% on **All Neg** . The only model to improve over CLIP on SCM tasks is NegCLIP, which follows the fact that the hard image negatives that NegCLIP is trained on provide the most similar task to what we test of any of these models. None of the models trained without an explicit CLIP-loss component outperform CLIP on SCM tasks, but DAC ultimately performs the worst. 

Performance on the Pick5 variations of each task follow the trends of the standard performance. Performance on **BaseNeg** for Flava point to a weakness in comparing longer text examples, given the significant drop from 47.99% to 11.6% that is not demonstrated in other models. 

## **5 Using summarized DCI as finetuning dataset** 

To evaluate the use and difficulty of the sDCI dataset for training, we fine-tune state-of-the-art models with it. In particular, we use a ViT/32B CLIP model in all of our experiments, requiring use of the CLIPbounded version of our dataset. We split sDCI into 7800 train, 100 validation, 112 test samples for this purpose. We use a training batch size of 32 and a learning rate of 5e-5 for all experiments, and run for up to 10 epochs. We train using both standard CLIP loss as well as an additional Negatives loss component, which follows the ‘text negative’ of NegCLIP (Yuksekgonul et al., 2023). Given the tiny size of our finetuning sets relative to the 400M pretraining images, we use LoRA (Hu et al., 2021) to reduce the trainable parameters. We train a model with and without negatives loss. 

Interestingly, models trained absent of CLIP (BLIP, Flava, X-VLM) experience a far less noticeable drop in performance between **All Neg** and **All Hard Negs** . This validates that sDCI’s CLIP-hard negatives are not simply a higher proportion of ‘impossible’ negatives, but rather capture some underlying trait about the negatives that CLIP models and their descendants all struggle with. 

None of the models presented perform well across all of the sDCI test set. Given each of the CLIP-style models have some kind of advantage on this test set due to being trained on some objective that sDCI directly evaluates, we expect that the BLIP, Flava, and X-VLM scores are somewhat representative for existing state-of-the-art models’ true performance on this test set. 

In order to make good use of the multiple summarized captions we have per image and submask, we randomly select one to be used in each individual epoch. We call this method _Pick1_ . We describe this method and other ablations we attempted in more detail in Appendix C. 

We follow the experimental setup of DAC (Doveh et al., 2023a) by evaluating our sDCI fine-tuned CLIP on the ARO and VL-Checklist benchmarks. We compare to DAC directly as it is the most similar work to ours in attempting to increase caption density. As noted in Figure 3, these automatically generated captions are generally noisy. As DAC is using 3M images for fine-tuning, we performed a small ablation on the number of DAC images to use for fine-tuning to be similar to our base image count (10,000 compared to our 8,012), or to our full mask count (100,000 compared to our 99,445). 

8 

|**Model**|**ARO**<br>VG-R<br>VG-A<br>COCO<br>FLICKR|**VL-Checklist**<br>Object<br>Attribute<br>Relation|
|---|---|---|
||||
|sDCI_P_1<br>sDCI_P_1_NL_0|76.23%<br>67.56%<br>88.58%<br>91.30%<br>57.34%<br>61.98%<br>39.36%<br>44.62%|80.71%<br>68.69%<br>70.12%<br>_88.37_%<br>70.42%<br>61.28%|
||||
|DAC_LLM_10_,_000<br>61.53%<br>63.89%<br>46.28_±_1.5%<br>59.41_±_1.9%<br>66.90%<br>57.4%<br>56.96%<br>DAC_LLM_100_,_000<br>61.0%<br>63.6%<br>48.2%<br>61.42%<br>66.87%<br>57.22%<br>57.18%<br>DAC_LLM_500_,_000<br>60.1%<br>63.8%<br>50.2%<br>61.6%<br>66.54%<br>57.39%<br>56.77%<br>DAC_LLM_3_,_000_,_000<br>**81.28**%<br>**73.91%**<br>**94.47%**<br>**95.68%**<br>87.30%<br>**77.27%**<br>86.41%<br>DAC_SAM_3_,_000_,_000<br>77.16%<br>70.5%<br>91.22%<br>93.88%<br>**88.50%**<br>75.83%<br>**89.75%**|||
|CLIP Baseline Radford et al. (2021)<br>59.98%<br>63.18%<br>47.9%<br>60.2%<br>81.17%<br>67.67%<br>61.95%<br>BLIP2 Li et al. (2023a)<br>41.16%<br>71.25%<br>13.57%<br>13.72%<br>84.14%<br>80.12%<br>70.72%<br>NegCLIP Yuksekgonul et al. (2023)<br>81%<br>71%<br>86%<br>91%<br>81.35%<br>72.24%<br>63.53%<br>SVLC Doveh et al. (2023b)<br>80.61%<br>73.03%<br>84.73%<br>91.7%<br>85%<br>71.97%<br>68.95%|||



**Table 3** sDCI fine-tuned CLIP performance against the ARO and VL-Checklist benchmark. We compare CLIP fine-tuned with sDCI against models fine-tuned using DAC captions. Since the DAC dataset contains 3M images whereas sDCI contains only 7805 images, we performed an ablation of the number of training images used in the DAC dataset. In this instance, DAC _LLM_ 10000 refer to fine-tuning CLIP using only 10,000 images from DAC. We plot the mean across 5 different seeds and display the standard deviation when it is above 1% accuracy. We observe that training on sDCI lead to significant improvement in comparison to DAC for a comparable number of examples. 

## **5.1 Results** 

## **6 Conclusion and Future Work** 

In Table 3, we show that, while the DCI Pick1 model trained with negatives loss (sDCI _P_ 1) does not reach the performance of DAC models trained on 3M images, it does improve over the CLIP baseline on most metrics[3] , and outperforms some baselines trained on more data. sDCI _P_ 1 does however outperform both sample-limited ablations of DAC, suggesting that a small number of highly aligned image to dense text pairs are more effective for training models than larger quantities of more loosely aligned or sparse data. Unsurprisingly, the version trained without negatives loss, sDCI _P_ 1 _NL_ 0, does not improve across most benchmarks, and even somewhat degrades when compared to the CLIP baseline.[4] Of note however is the significant bump in VL-Object, alongside some improvement to VL-Attribute. Improvements here suggest that the sDCI dataset successfully includes more object, and to a lesser degree attribute, information than the captions in the source dataset for CLIP. It does, however, point to a limitation of using the LLM summarizations and not incorporating mask information, as relational information is sometimes lost. 

> 3The decreased performance on VL-Object may be explained by our LLM-generated negatives not closely covering the test set negatives. 

> 4The degradation is likely due to the distribution shift and small sample size, given the training objective is the same as CLIP. 

We introduce the Densely Captioned Images dataset, and display clear use for it as a evaluation benchmark. We also show initial potential for using the dataset for fine-tuning. Given that in order to evaluate today’s models on DCI we had to reduce the size of the text to only 77 tokens, DCI should prove to be useful for a longer period of time as models that are able to consume and utilize larger amounts of text context become the norm. We envision that in those cases the full human annotated captions without length reduction would be provided. Today’s context size limitation also prevented us from fine-tuning existing models on the highly aligned text-image data within DCI, as existing models don’t have enough context size to handle the full text, but the dataset isn’t nearly large enough to pre-train a new set of models that could use the full text. It could be relevant to treat developing highly aligned text-image datasets in a similar manner to that used in machine translation for low-resource languages, which run into a similar issue with cost and difficulty to collect. This area of work has relied on automated methods such as bitext mining (Team et al., 2022) to bootstrap up from an initial set of expertly collected examples, which DCI may already provide the foundation for. Further, we haven’t attempted to incorporate the pixel-level masks that the dataset has in any of our experiments, instead opting to use crops around the masks to retain parity with our test set. This dataset is unique for both the extreme density and high degree of alignment present, and in this introductory work we’ve only scratched the surface of using this information to its fullest extent. 

9 

## **References** 

- Amro Kamal Mohamed Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, and Ari S. Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. In _ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models_ , 2023. https://openreview.net/forum? id=4vlGm9gv6c. 

- Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning, 2022. 

- Florian Bordes, Shashank Shekhar, Mark Ibrahim, Diane Bouchacourt, Pascal Vincent, and Ari S. Morcos. Pug: Photorealistic and semantically controllable synthetic data for representation learning. In _Advances in Neural Information Processing Systems_ , 2023. 

- John Canny. A computational approach to edge detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , PAMI-8(6):679–698, 1986. doi: 10.1109/ TPAMI.1986.4767851. 

- Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments, 2021. 

- Ilias Chalkidis, Xiang Dai, Manos Fergadiotis, Prodromos Malakasiotis, and Desmond Elliott. An exploration of hierarchical attention transformers for efficient long document classification, 2022. 

- Qian Chen, Zhen-Hua Ling, and Xiaodan Zhu. Enhancing sentence embedding with generalized pooling, 2018. 

- Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server, 2015. 

- Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: web-curated image-text data created by the people, for the people, 2021. 

- Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, and Leonid Karlinsky. Dense and aligned captions (dac) promote compositional reasoning in vl models, 2023a. 

- Sivan Doveh, Assaf Arbelle, Sivan Harary, Rameswar Panda, Roei Herzig, Eli Schwartz, Donghyun Kim, Raja Giryes, Rogerio Feris, Shimon Ullman, and Leonid 

Karlinsky. Teaching structured vision&language concepts to vision&language models, 2023b. 

- Enrico Fini, Pietro Astolfi, Adriana Romero-Soriano, Jakob Verbeek, and Michal Drozdzal. Improved baselines for vision-language pre-training. _Transactions on Machine Learning Research (TMLR)_ , 2023. 

- Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III au2, and Kate Crawford. Datasheets for datasets, 2021. 

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 

- Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything, 2023. 

- Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Fei-Fei Li. Visual genome: Connecting language and vision using crowdsourced dense image annotations, 2016. 

- Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng Gao. Elevater: A benchmark and toolkit for evaluating language-augmented visual models, 2022a. 

- Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. BLIP: bootstrapping language-image pre-training for unified vision-language understanding and generation. _CoRR_ , abs/2201.12086, 2022b. 

- Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023a. 

- Xianhang Li, Zeyu Wang, and Cihang Xie. An inverse scaling law for clip training, 2023b. 

- Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking, 2023c. 

- Zhiqiu Lin, Xinyue Chen, Deepak Pathak, Pengchuan Zhang, and Deva Ramanan. Visualgptscore: Visiolinguistic reasoning with multimodal generative pretraining scores, 2023. 

- Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, and Ranjay Krishna. Crepe: Can visionlanguage foundation models reason compositionally?, 2023. 

- Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million cap- 

10 

tioned photographs. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q. Weinberger, editors, _Advances in Neural Information Processing Systems_ , volume 24. Curran Associates, Inc., 2011. https: //proceedings.neurips.cc/paper_files/paper/2011/ file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf. 

- Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives, 2020. 

- Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan. Filtering, distillation, and hard negatives for visionlanguage pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6967–6977, 2023. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. _CoRR_ , abs/2103.00020, 2021. 

- Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs, 2021. 

- Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2556–2565, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1238. https://aclanthology.org/P18-1238. 

- Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. FLAVA: A foundational language and vision alignment model. In _CVPR_ , 2022. 

- Krishna Srinivasan, Karthik Raman, Jiecao Chen, Mike Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In _Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21)_ , 2021. https://arxiv.org/abs/2103.01913. 

- NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling human-centered machine translation, 2022. 

- Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. YFCC100m. _Communications of the ACM_ , 59(2):64–73, jan 2016. doi: 10.1145/2812802. https://doi.org/10.1145%2F2812802. 

- Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality, 2022. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. 

- Jack Urbanek and Pratik Ringshia. Mephisto: A framework for portable, reproducible, and iterative crowdsourcing, 2023. 

- Hu Xu, Saining Xie, Po-Yao Huang, Licheng Yu, Russell Howes, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Cit: Curation in training for effective vision-language data. _arXiv preprint arXiv:2301.02241_ , 2023a. 

- Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi 

11 

Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data, 2023b. 

- Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , 2:67–78, 2014. doi: 10.1162/ tacl_a_00166. https://aclanthology.org/Q14-1006. 

- Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang-Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel-Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, and Armen Aghajanyan. Scaling autoregressive multi-modal models: Pretraining and instruction tuning, 2023. 

- Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why visionlanguage models behave like bags-of-words, and what to do about it? In _International Conference on Learning Representations_ , 2023. https://openreview.net/ forum?id=KRLUvxh8uaX. 

- Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. _arXiv preprint arXiv:2111.08276_ , 2021. 

- Hang Zhang, Yeyun Gong, Yelong Shen, Weisheng Li, Jiancheng Lv, Nan Duan, and Weizhu Chen. Poolingformer: Long document modeling with pooling attention, 2022. 

- Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. Vl-checklist: Evaluating pre-trained vision-language models with objects, attributes and relations, 2023. 

12 

## **Appendix** 

## **A LLM Summarization and Negatives** 

In this section we discuss the prompting strategies we used on the LLaMA 70B-chat model to produce our summarized captions and negatives. 

## **A.1 Summarization** 

In order to generate multiple summaries for each image and subsection of the image, we used the following method. First, we attempted to create a first-pass summary of either the full image or of masks with the following prompts: 

**Full Image:** You are given a full-text description of an image. You should summarize it into about 65 words, being sure to include as much salient visual information as possible given the 65 word constraint, especially information from the start of the original desc- ription. The new description should apply for the original image. Respond with only the summary, in one line. 

**Submask:** You are given a description of part of an image. You should summarize it into a single line no longer than 65 words, being sure to include as much salient visual information as possible given the 65. Don’t include any details not present in the provided description. Respond with only the summary, in one line. 

This provided us with once caption per image and submask, which is used universally as the _first_ caption for each. We, however, wanted to have more options to capture potential details that may have been missed by the first summarization. To that end we repeated the following prompt until we had at least 6 captions per. 

**Multi-caption Prompt:** You are providing descriptions of an image. The goal is to create a list of summarized descriptions that all accurately describe the same image, but may pay attention to slightly different details. A complete description will be provided. Complete 5 entries in this list, each a few sentences long. Provide in the format:\n1. <description>\n2. ... 

## **A.2 Negatives** 

To generate negatives, we came up with three distinct prompts to produce negatives when given a summary. All of the negatives are generated from the _first_ summarized caption. Each attempted to 

raise different kinds of reasoning, either for simple basic edits, changes to structural information, or reconstructing a new sentence from the same bag of words. 

**basic:** You are given the description for an image. You should provide a mostly similar description, changing the original one slightly, but introducing enough significant differences such that the two descriptions could not possibly be for the same image. Keep the description length the same. Provide just the updated description. 

**layout:** You are given the description for an image. You should provide a mostly similar description, with one part of it changed in a way that visibly alters the structure, layout, or content of the image. The change should introduce enough difference such that the two descriptions could not be for the same image. Keep the description length the same. Provide just the updated description. 

**swaps:** You are given a description. Selecting ONLY from the same words, construct a random response with the same words in a completely new order. The new description should not remain accurate to the first. Try to use all of the words from the original description, and keep the length the same, but use the words nearly randomly such that the scene makes less sense. Try to pair nouns and adjectives differently than in the original. 

Ultimately, in initial ablations we found that the **swaps** prompt produced the best model performance, so we use these negatives as the default. 

## **A.3 Common Issues** 

One main issue with LLM-based summarization is that LLaMA2 was not particularly effective at producing answers within the specified word boundary. Thus, we additionally use prompts that attempt to get the model to shorten the output response so long as it is somewhat close in length. If after enough tries this didn’t work, we’d return to the original prompt. 

**Reduce:** You are given the description for a scene. Summarize and rephrase the description such that the new description is a few words shorter, but retains key information. 

Another issue is that the chat fine-tuning on the LLaMA2 model made it frequently preface the response with additional affirmative text, like “sure I can do that...”. Often this could be removed by simply removing the everything before the last line break. 

1 

## **A.4 Output Text Examples** 

Below we provide an example of a summarization for a base image (which includes details from all of the subimages during the prompting stage) as well as a summarization for one submask. We also share one example of each type of type of negative for both. 

## **A.4.1 Base Image** 

_Description:_ The upper floor of an ornate design building with the words “Dem Wahren, Schonen” visible on its top. There’s a light source with five light bulbs decorating each pillar, at least two marble statues and multiple arches between each pillar. An ornate building with at least two floors. There’s four small pillars with three black guard rails and four light stands on the pillars. There’s two marble statues on either end of the upper floor. The upper floor also has five arches with decorative symbols above them. There’s six visible tall supporting pillars on either side of the arches. On the top is the visible words: Dem Wahren Schoenon. On the bottom floor is three arches with a black light hanging from each of them. The inner arches and interiors are somewhat visible. In front of the building is a black streetlight. ...(submask text omitted)... 

_Selected Summaries (3/8): (1)_ An ornate building with four pillars, three arches, and a decorative facade. The building has a upper floor with five light bulbs, two marble statues, and multiple arches. The lower floor has three black guard rails and a streetlight. The image includes various decorative elements such as swirling branches, leaves, and faces. 

_(2)_ The image shows the upper floor of an ornate building with five pillars, each with a light source and decorative elements. The building has a grand entrance with multiple arches and marble statues. The words “Dem Wahren, Schonen” are visible on the top of the building. 

_(3)_ The image highlights the right side of the upper floor, focusing on the decorative elements on the pillars and the arches. The right most pillar has a thinner middle and wider bottom, and the top pillar decoration has swirly branches and a spread out decoration like a opened flower. The image also shows the inner pillars and the black guard rails. 

_basic negative:_ A grand building with four pillars, three arches, and a detailed facade showcasing mythical creatures and a clock. The building’s upper floor boasts five light bulbs, two marble statues, and several arches, while the lower floor features three black guard rails and a streetlight that illuminates 

the vibrant flowers and trees outside. 

_layout negative:_ A grand building with intricate details, including four pillars, three arches, and a decorative facade now covered in vines and moss, giving it a weathered and natural appearance. The upper floor features five light bulbs, two marble statues, and multiple arches, while the lower floor has three black guard rails and a streetlight. 

_swaps negative:_ A facade with five arches, three pillars, and four light bulbs. The building has a decorative upper floor with two marble statues, multiple black guard rails, and a streetlight. The lower floor has three arches and various decorative elements such as swirling branches, leaves, and faces. 

## **A.4.2 Submask** 

_Description: Front black street light_ - The black street light in front of the ornate building. It has a thin support beam holding it up with little lines going down it. There’s curved branches on the beams supporting the lights. Each light has the same design, with little pointy spires or points on its top tip. There’s about 6 visible lights on this street light. ...(submask text omitted)... 

_Selected Summaries (3/8): (1)_ A black street light stands in front of an ornate building, supported by a thin beam with curved branches. Six lights with pointy spires are visible, with the closest one featuring clear glass panes and a lantern-like shape. _(2)_ The front black street light is decorated with curved branches and has a lantern-like shape. The light closest to the image has a clear glass pane with ridges on the edges and a small decorative circle above it. 

_(3)_ This image highlights the unique design of a black street light, with a lantern-like shape and pointed spires on top. The light has a clear glass panel with ridges and a decorative circle, and is supported by a thin beam and curved branches. The image also shows the ornate building in the background. 

_basic negative:_ A white street light stands in front of a modern skyscraper, supported by a thick concrete pillar. Five lights with sleek, angular designs are visible, with the closest one featuring frosted glass panes and a cylindrical shape. 

_layout negative:_ A black street light stands in front of a modern skyscraper, supported by a thick concrete pillar. Six LED lights with sleek, angular designs are 

2 

visible, with the closest one featuring a frosted glass cover and a cylindrical shape. 

_swaps negative:_ A lantern-like shape stands in front of an ornate building, supported by a clear glass panes with curved branches. Six pointy spires are visible, with the closest one featuring a thin beam and black street light. 

## **B Crowdsourcing Methodology** 

Extending from the overall description for the main task in Section 3.2, we here provide an overview of the precursor qualification tasks, as well as some additional details of the preparation and packaging process for the dataset. The complete set of instructions, as well as code to be able to directly reproduce the collection for this dataset, is available on the project GitHub[5] . 

Note, we do not include the code for our initial version of the collection task, wherein workers were asked to both select the regions of the image that were worth annotating _and_ annotate them in the same pass, as these proved both incredibly time-consuming and often lower quality than using model-based mask generation and allowing workers to filter low-quality masks. 

## **B.1 Quality Assurance** 

High-quality datasets rely on getting solidlyperforming crowdworkers, which nowadays can prove to be an adversarial task initially. To remedy this situation, we set up a multi-stage process wherein workers could complete precursor tasks (for which they were compensated) that we could use to determine eligibility in the main task pool. 

Stage one of the task asked a few questions about a preset image, wherein we asked workers to provide a few sentences describing the image, then note a few things in the image they might include descriptions of if they were to need to write 1000 meaningful words about the image. After filtering out answers from bots, we manually reviewed answers for quality on the provided description, as well as having included any of a few details in the image that we felt were hard to notice on first pass in either the description or list of additional things they might describe. 

Workers with solid fluency in English and solid attention to detail were moved into stage two, wherein 

> 5https://github.com/facebookresearch/DCI/tree/main/ reproduction/crowdsourcing 

they had access to the full task and were eligible to complete it three times. In this stage we evaluated responses for a general understanding of the more complex task interface, and allowed workers who completed this stage or only made minor mistakes to the full tasks. Workers who did make minor mistakes were given feedback to help understanding. 

In the last stage, work was audited regularly from each worker to provide feedback about description quality, proper use of disqualifying bad masks, and overall tradeoff between time spent and work completed. We used this information to limit overcontribution from single individual, filter out a few workers that provided decreasing quality over time, and bonus workers who were slower than our target pace but providing exceptional quality. 

## **B.2 Instruction Details** 

Before entering the task, the worker was walked through an example task to familiarize them with the interface and the goal of the task. We additionally provided a few specific scenarios to anticipate common questions, like what to do when two masks are the same, or how to deal with writing for a mask that contains a mask that was already done, or what to do on images that were much simpler than the norm. A complete list of these instructions is available with our released code. 

## **B.3 Worker Metrics** 

Over the course of a month, we made the first qualification round eligible to a large cohort of workers on Mechanical Turk, and fielded over 800 submissions. Of these, roughly 250 workers made it through to the second task. 

In the second stage, we hand-reviewed around 600 tasks (up to 3 from each worker from the first stage), and ultimately moved a group of 120 workers to the full task. Of these, around 80 were regular contributors over the course of our collection. 

During the main task, we enacted controls to ensure that no worker provided more than 10% of the currently collected data at any time. We used automated metrics around words per image, words per minute, and unique words per image to act as an overview of worker quality, and hand-reviewed examples that were far outside of our expected bounds for these metrics. 

3 

## **C Extended Ablations** 

In order to evaluate ideal use techniques for the Densely Captioned Images dataset for the purpose of fine tuning, we run some ablation experiments and compare to performance on fine tuning on COCO and Localized Narratives. 

## **C.1 DCI Fine-tuning ablation methods** 

## **C.1.1 PickN Caption Training** 

In order to make use of all of the available captions for each image, we pick some number of captions from those available. For _N_ = 1, this just results in selecting one of the captions for each image randomly in each epoch. For _N >_ 1, we instead provide multiple captions during _both_ CLIP loss and negatives loss calculations. In these circumstances, loss is calculated between the _worst_ positive and the _best_ negative. As each image is supposed to have unique and high-quality captions across the whole set, the expectation here is that any caption for one image should score better than any caption for another. 

## **C.1.2 Image-based Batching** 

When training with _sDCI_ submasks, we have the opportunity to provide the model with exceptionally difficult negatives during the CLIP loss calculation, namely other submasks from the same image. We call this ablation _ImgGroup_ and run it for each _sDCI_ model to determine the impact that hard negatives like these have on CLIP training. 

and _LN_ runs, but we also include spacy-swap ablations for _DCI_ -trained models to compare with the LLM-generated negatives. 

## **C.2 Datasets and Training methods** 

Overall we use the same training parameters outlined in Section 5. We use the following five datasets in our ablations: 

1. _sDCIsub_ : All of the complete images and subimages with LLM generated captions (referred to above as _DCIsub<_ 77). 

2. _LN_ : All images and captions from the COCO subset of Localized Narrations with a CLIP token count under 77 (referred to above as _LNCOCO<_ 77). 

3. _LN_ 7805: The first 7805 images of _LN_ , used to be a same-size comparison to _DCI_ -trained models. 

4. _COCO_ : All images and captions from the COCO 2017 set. 

5. _COCO_ 7805 :The first 7805 images of _COCO_ , used to be a same-size comparison to _DCI_ - trained models. 

For each sweep we select the best model as determined by highest score on validation metrics from the same dataset used for training (for instance using _COCO_ valid for a model trained on _COCO_ train). 

For _sDCI_ , we split the test set into a training set of 7599, a valid set of 98, and a test set of 108. All metrics below are reported on the 108 image test set. 

## **C.1.3 Negatives Loss** 

## **C.3 Results** 

In order to evaluate the impact that over-weighing negatives during train time has on model performance, we launch one job with 9 times the negatives loss used in our standard experiments. This attempts to evaluate how easily the negative construction techniques in these standard benchmarks could be ‘gamed’. 

## **C.1.4 Negatives Selection** 

While we have LLM-generated negatives readily available for the _sDCI_ dataset (as described in Section A, we also wanted to compare to negatives for the _LN_ and _COCO_ datasets. For this we used a spacy-based swapping technique similar to NegCLIP (Yuksekgonul et al., 2023) or DAC (Doveh et al., 2023a), wherein noun phrases, verbs, and adjectives were randomly swapped inside of a given caption to create a negative. We used these spacy-swaps for all _COCO_ 

## **C.3.1 Aggregate DCI Ablations** 

A complete table of our sDCI ablations can be seen in Table 4, however it is more dense information than is likely useful in seeing overall trends. Instead, we aggregate over the different ablation methods. 

In Table 5, for ARO and VL-C we observe that generally constructing batches from the entire dataset rather than from masks in the same image is better for performance. 

We also see that using _Pick1_ is the most effective method for making use out of the LLMsummarizations of captions we’ve created, as it allows the model to see more of the related text (better than _first_ ) without potentially penalizing it for situations where two related overlapping captions are provided at the same time (which may be an issue with _Pick5_ ). 

4 

|Conditions<br>CaptsNLBatch|ARO<br>VG-R<br>VG-A<br>COCO FLICKR|VL-Checklist<br>Object AttributeRelation|sDCI All<br>SCM<br>Neg|All Pick5<br>SCM<br>Neg|Base<br>All<br>Neg<br>H-Neg|Base<br>All<br>Neg<br>H-Neg|
|---|---|---|---|---|---|---|
|First<br>9 Rand<br>First<br>9 Group<br>Pick1 9 Rand<br>Pick1 9 Group<br>Pick5 9 Rand<br>Pick5 9 Group|75.34%<br>62.72%<br>84.33%<br>87.86%|76.99%<br>67.77%<br>68.27%|51.71%<br>**97.20%**|8.69%<br>79.41%|**98.21%**|88.30%|
||74.82%<br>58.26%<br>85.05%<br>88.70%|69.44%<br>68.65%<br>68.93%|54.38%<br>96.79%|9.44%<br>79.69%|94.64%|**88.65%**|
||76.04%<br>65.51%<br>86.69%<br>89.72%|73.35%<br>65.04%<br>69.48%|44.12%<br>95.55%|12.59%<br>93.57%|93.75%|82.97%|
||76.41%<br>63.71%<br>82.89%<br>88.04%|72.68%<br>65.44%<br>63.33%|50.14%<br>95.62%|19.36%<br>94.05%|92.86%|84.06%|
||**78.61%**<br>65.65%<br>86.83%<br>**91.40%**|68.61%<br>66.10%<br>**71.82%**|39.33%<br>96.10%|7.46%<br>**94.94%**|**98.21%**|85.64%|
||75.65%<br>**68.92%**<br>85.45%<br>90.26%|70.17%<br>65.09%<br>63.80%|43.23%<br>96.17%|11.56%<br>94.32%|96.43%|83.79%|
|First<br>1 Rand<br>First<br>1 Group<br>Pick1 1 Rand<br>Pick1 1 Group<br>Pick5 1 Rand<br>Pick5 1 Group|64.00%<br>60.79%<br>83.93%<br>87.46%|76.73%<br>68.75%<br>63.53%|54.17%<br>96.17%|14.64%<br>75.92%|96.43%|85.91%|
||73.09%<br>62.60%<br>79.54%<br>84.84%|72.64%<br>69.37%<br>61.68%|61.76%<br>95.76%|19.63%<br>76.81%|97.32%|85.43%|
||76.23%<br>67.56%<br>**88.58%**<br>91.30%|80.71%<br>68.69%<br>70.12%|50.48%<br>94.39%|19.02%<br>88.85%|95.54%|81.40%|
||64.97%<br>65.05%<br>80.32%<br>87.62%|75.01%<br>66.70%<br>60.75%|58.14%<br>94.60%|26.81%<br>89.88%|96.43%|83.58%|
||72.56%<br>61.32%<br>78.00%<br>83.68%|77.64%<br>66.74%<br>66.65%|49.93%<br>94.19%|18.60%<br>86.87%|95.54%|80.78%|
||57.96%<br>61.49%<br>78.27%<br>84.06%|64.80%<br>63.05%<br>61.73%|51.44%<br>94.66%|19.97%<br>90.63%|93.75%|81.81%|
|First<br>0 Rand<br>First<br>0 Group<br>Pick1 0 Rand<br>Pick1 0 Group<br>Pick5 0 Rand<br>Pick5 0 Group|55.61%<br>55.83%<br>29.60%<br>39.22%|80.70%<br>67.71%<br>62.10%|57.25%<br>74.21%|22.23%<br>30.30%|80.36%|67.58%|
||49.91%<br>59.04%<br>41.68%<br>50.90%|73.11%<br>67.35%<br>60.03%|63.13%<br>75.44%|27.84%<br>33.11%|81.25%|69.49%|
||57.34%<br>61.98%<br>39.36%<br>44.62%|**88.37%**<br>**70.42%**<br>61.28%|56.77%<br>74.08%|23.73%<br>34.54%|83.93%|64.43%|
||50.88%<br>56.67%<br>46.62%<br>51.82%|76.83%<br>67.41%<br>58.98%|**64.02%**<br>71.55%|**31.60%**<br>35.16%|78.57%|66.28%|
||53.64%<br>62.39%<br>36.70%<br>43.78%|81.33%<br>69.79%<br>61.40%|56.77%<br>75.99%|22.16%<br>30.78%|82.14%|68.60%|
||46.89%<br>55.67%<br>32.05%<br>37.06%|75.24%<br>68.98%<br>63.35%|61.97%<br>77.84%|26.88%<br>32.35%|79.46%|73.05%|
|CLIP Baseline|59.98%<br>63.18%<br>47.9%<br>60.2%|81.17%<br>67.67%<br>61.95%|37.82%<br>60.12%|10.94%<br>23.19%|67.86%|39.95%|



**Table 4** Full _sDCI_ ablation analysis against all benchmarks. Cells are colored in comparison to the CLIP Baseline. 

|Ablation|ARO<br>VG-R<br>VG-A<br>COCO<br>FLICKR|VL-Checklist<br>Object<br>Attribute<br>Relation|
|---|---|---|
|Rand<br>ImgGroup|**67.71**%<br>**62.64**%<br>**68.22**%<br>73.23%<br>63.40%<br>61.27%<br>67.99%<br>**73.70**%|**77.71**%<br>**67.89**%<br>**66.07**%<br>72.21%<br>66.89%<br>62.51%|
|Neg Loss 9<br>Neg Loss 1<br>Neg Loss 0|**76.15**%<br>**64.13**%<br>**85.21**%<br>**89.33**%<br>68.14%<br>63.14%<br>81.44%<br>86.49%<br>52.38%<br>58.60%<br>37.67%<br>44.57%|71.87%<br>66.35%<br>**67.61**%<br>74.59%<br>67.22%<br>64.08%<br>**78.43**%<br>**68.61**%<br>61.19%|
|First<br>Pick1<br>Pick5|65.46%<br>59.87%<br>67.36%<br>73.16%<br>**66.98**%<br>**63.41**%<br>**70.74**%<br>**75.52**%<br>64.22%<br>62.57%<br>66.22%<br>71.71%|74.94%<br>**68.27**%<br>64.09%<br>**76.99**%<br>67.28%<br>63.99%<br>72.97%<br>66.63%<br>**64.79**%|



**Table 5** _sDCI_ Grouped ablation against standard benchmarks. Results from Table 4 are averaged across different ablations. 

5 

We do observe that using high negatives loss is very effective at gaining additional performance on most of these metrics, however this does not include VLObject and VL-Attribute, likely due to our method of constructing negatives not correlating very well with the types of negatives created in these tests. While this provides great scores, it mostly just provides evidence towards the aforementioned issues with these types of evaluations Lin et al. (2023). 

In Table 6, we observe a slightly different story. Here _ImgGroup_ appears to be the best method for selecting images, only performing worse on the Base Neg test. This is expected though, as the test set is constructed with sequential examples in a manner similar to the _ImgGroup_ ablation, which doesn’t have any effect on Base Neg given there are no submasks to deal with in that test. 

No individual setting for negatives loss performs best on all metrics, however it is unsurprising that negatives loss 0 results in the highest performance on Subcrop-Caption Matching tasks (given their similarity to the CLIP-style learning objective), and not as well as negatives-trained models on negatives tasks. 

The PickN ablations are somewhat surprising, as _first_ captions generally performed the best overall, and Pick1 outperformed training on Pick5 when testing on Pick5. 

## **C.3.2 DCI fine-tuning performance** 

In Table 7 we report our different ablations performance compared to the LN and COCO baselines, as well as CLIP and DAC as comparison points. We expect sDCI models to outperform all other baselines, given the test set is out-of-distribution for the other models. Still, there are some interesting observations available in this table. 

First, for sDCI, Localized Narrations, and COCO, only training on the smallest subset of images (7800) and without using masks, DCI ends up performing only as well as COCO for Subcrop-Caption matching, but both outperform Localized Narratives by a noticeable margin. This may point to data in Localized Narratives generally being less sample-efficient than baseline COCO captions. 

Second, moving to LLM-based captions increases performance on all negatives at the expense of performance on Subcrop-Caption masking. This would imply that the captions generated by LLMs may actually be overfitting to the test task to the detriment of performance on other metrics. 

Third, training directly on negatives as a method 

of improving models’ vision-language understanding universally decreases performance on SubcropCaption masking, a task that also definitely takes strong vision-language understanding. This is seen regardless of the training dataset used, or of any other sDCI ablation involved. 

## **C.3.3 Linear Transferability** 

We evaluate a subset of all of our models, trained on each of our candidate datasets, on the Elevater Li et al. (2022a) benchmark to determine linear transferability for our models. We expect some degradation given the relatively small size of our training datasets. We also evaluate them on zero-shot ImageNet specifically. 

Overall, in table 9 we observe a slight degradation in linear probe performance on their included datasets across all shots. In our ImageNet zero-shot evaluation reported in Table 8, we note that sDCI trained without negatives suffers much less degradation compared to with negatives, ending up with comparable performance to the DAC models. 

## **D Additional Selected Examples** 

We include a few additional examples from the DCI dataset, selected from a random subset of 20 instances to highlight certain elements of the dataset. In each we share a subset of the masks available per image. 

Figure 4 shows the level in-depth that the descriptions go to. Despite only being an image of shoes on some grass, DCI contains descriptions down to the details of the crossing pattern on the toecap or the specks of light colored materials in a clump of dirt on the ground. 

Figure 5 contains a fairly complex scene of various tiled stone buildings and light fixtures, however the description is able to identify a tree that is mostly obscured by a foreground building, as well as be indepth enough to describe the shape at the top of a lamppost in the image as a “tiny urn”. 

Figure 6 stands as another example of retaining useful and aligned information even when there’s a high amount of potential complexity in the image. While "An antique blue car in front of a row of trees" may be a standard caption for this image, we instead have details of the orientation of the car, details of what is visible in frame, and the resolution of the text goes all the way down to the small circular sticker on the rear passenger side door, or the screws on the reflector on the front bumper. 

6 

|Ablation|All<br>SCM<br>Neg|All Pick5<br>SCM<br>Neg|Base<br>Neg|All<br>Hard Negs|
|---|---|---|---|---|
|Rand<br>ImgGroup|51.17%<br>88.65%<br>**56.47**%<br>**88.71**%|16.57%<br>68.35%<br>**21.45**%<br>**69.56**%|**91.57**%<br>90.08%|78.40%<br>**79.57**%|
|NL 9<br>NL 1<br>NL 0|47.15%<br>96.24%<br>54.32%<br>**94.96**%<br>**59.99**%<br>74.85%|11.52%<br>**89.33**%<br>19.78%<br>84.83%<br>**25.74**%<br>32.71%|95.68%<br>**95.84**%<br>80.95%|**85.57**%<br>83.15%<br>68.24%|
|First<br>Pick1<br>Pick5|**57.07**%<br>**89.26**%<br>53.95%<br>87.63%<br>50.45%<br>89.16%|17.08%<br>62.54%<br>**22.19**%<br>**72.68**%<br>17.77%<br>71.65%|**91.37**%<br>90.18%<br>90.92%|**80.89**%<br>77.12%<br>78.95%|



**Table 6** _sDCI_ Grouped ablation against the 112 heldout sDCI test images. Results from Table 4 are averaged across different ablations. 

|Training Parameters<br>Dataset<br>Captions<br>Negatives<br>Batching|All<br>All Pick5<br>Base<br>All<br>SCM<br>Neg<br>SCM<br>Neg<br>Neg<br>Hard Negs|
|---|---|
|_sDCI_7805<br>First<br>LLM<br>Rand<br>_sDCI_<br>Pick1<br>LLM<br>ImgGroup<br>_sDCI_<br>Pick1<br>LLM<br>Rand<br>_sDCI_<br>First<br>LLM<br>ImgGroup<br>_sDCI_<br>First<br>LLM<br>Rand<br>_sDCI_<br>First<br>Spacy<br>Rand|38.30%<br>84.61%<br>9.10%<br>69.22%<br>92.86%<br>76.54%<br>58.14%<br>**94.60%**<br>26.81%<br>**89.88%**<br>96.43%<br>83.58%<br>50.48%<br>94.39%<br>19.02%<br>88.85%<br>95.54%<br>81.40%<br>61.76%<br>95.76%<br>19.63%<br>76.81%<br>**97.32%**<br>85.43%<br>54.17%<br>96.17%<br>14.64%<br>75.92%<br>96.43%<br>**85.91%**<br>55.13%<br>87.35%<br>19.08%<br>59.30%<br>89.29%<br>75.85%|
|_sDCI_<br>Pick1<br>None<br>ImgGroup<br>_sDCI_<br>Pick1<br>None<br>Rand<br>_sDCI_<br>First<br>None<br>ImgGroup<br>_sDCI_<br>First<br>None<br>Rand|**64.02**%<br>71.55%<br>**31.60%**<br>35.15%<br>78.57%<br>66.28%<br>56.77%<br>74.08%<br>23.73%<br>34.54%<br>83.93%<br>64.43%<br>63.13%<br>75.44%<br>27.84%<br>33.11%<br>81.25%<br>69.49%<br>57.25%<br>74.21%<br>22.23%<br>30.30%<br>80.36%<br>67.58%|
|_LN_<br>First<br>Spacy<br>Rand<br>_LN_7805<br>First<br>Spacy<br>Rand<br>_LN_<br>First<br>None<br>Rand<br>_COCO_<br>First<br>Spacy<br>Rand<br>_COCO_7805<br>First<br>Spacy<br>Rand<br>_COCO_<br>First<br>None<br>Rand|37.82%<br>76.95%<br>9.37%<br>46.31%<br>86.61%<br>63.20%<br>34.27%<br>75.58%<br>7.73%<br>37.82%<br>83.04%<br>61.63%<br>41.45%<br>58.82%<br>12.72%<br>21.75%<br>80.36%<br>53.42%<br>40.97%<br>79.21%<br>12.65%<br>52.74%<br>91.07%<br>64.71%<br>38.51%<br>79.75%<br>11.70%<br>55.27%<br>86.61%<br>64.16%<br>42.00%<br>61.35%<br>13.95%<br>21.41%<br>82.14%<br>52.60%|
|CLIP Baseline<br>_DACLLM_<br>_DACSAM_|37.82%<br>60.12%<br>10.94%<br>23.19%<br>67.86%<br>39.95%<br>36.87%<br>81.12%<br>8.00%<br>35.91%<br>86.61%<br>70.66%<br>36.46%<br>84.40%<br>6.91%<br>40.83%<br>89.29%<br>73.94%|



**Table 7** Dense Captions test results. We compare _DCI_ -trained models to models trained on Localized Narratives and COCO datasets, as well as to baselines. 

7 

|ImageNet 0-Shot<br>Model<br>Valid. Accuracy|ImageNet 0-Shot<br>Model<br>Valid. Accuracy|
|---|---|
|CLIP (baseline)<br>_sDCIP_1<br>_sDCIP_1_NL_0<br>DAC-LLM<br>DAC-SAM|60.96%<br>42.51%<br>51.44%<br>52.65%<br>53.43%|



**Table 8** ImageNet zero-shot. 

|Model|Elevater N-Shot<br>0<br>5<br>20<br>50<br>Full|
|---|---|
|_sDCIP_1<br>_sDCIP_1_NL_0|45.10% 60.78% 69.54% 72.91% 77.49%<br>51.29% 61.59% 70.80% 73.39% 77.92%|
|CLIP|55.59% 64.85% 71.90% 74.38% 78.96%|



**Table 9** Elevater scores for linear probe across 20 benchmark datasets 

Figure 7 displays a more active scene of two women cooking bread, however it still captures in-depth descriptions of everything contained in the image including flour spread on a table, ornamental details on a tablecloth, and additionally the placement of a bowl and a knife that were not captured in their own masks, but still were successfully annotated. 

## **E Evaluating additional baselines on sDCI** 

We evaluate a large quantity of available VLMs from RN50, ViT, roberta, convnext, and coca architectures on the DCI test set. Results can be seen in Table 10. 

Generally we observe that larger models perform better, but no model excels at all tasks. In contrast to what is observed for models trained with negatives, performance from these pretraining objectives is positively correlated between Subcrop-Caption Matching and detecting negatives. 

The most performant model on SCM averaged between the two tasks is coca_ViT-L-14 mscoco_finetuned_laion2b_s13b_b90k. The most performant model in average across negatives tasks is ViT-g-14 laion2b_s34b_b88k 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0021-00.png)


**Figure 4** One example from the Densely Captioned Images dataset, highlighting how in-depth descriptions are provided even for relatively simple scenes. 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0022-00.png)


**Figure 5** One example from the Densely Captioned Images dataset, highlighting how text is still highly aligned even with complex masks. 

10 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0023-00.png)


**Figure 6** One example from the Densely Captioned Images dataset, highlighting the high resolution of the text in alignment with the image, down to details of the stickers on car windows or screws on the reflectors. 

11 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions_images/cvpr_2024_A_Picture_is_Worth_More_Than_77_Text_Tokens_Evaluating_CLIP-Style_Models_on_Dense_Captions.pdf-0024-00.png)


**Figure 7** One example from the Densely Captioned Images dataset, displaying a scene with a complex interaction. The aligned description captures the action itself, alongside in-depth details like the clasp on the box on the wall, or the hand of one of the women pictured. 

12 

|Training Parameters<br>Arch<br>Dataset|All<br>All Pick5<br>Base<br>All<br>SCM<br>Neg<br>SCM<br>Neg<br>Neg<br>Hard Negs|
|---|---|
|RN50<br>yfcc15m<br>RN50<br>cc12m<br>RN50-quickgelu<br>openai<br>RN50-quickgelu<br>yfcc15m<br>RN50-quickgelu<br>cc12m<br>RN101<br>yfcc15m<br>RN101-quickgelu<br>openai<br>RN101-quickgelu<br>yfcc15m<br>ViT-B-32<br>openai<br>ViT-B-32<br>laion400m_e31<br>ViT-B-32<br>laion400m_e32<br>ViT-B-32<br>laion2b_e16<br>ViT-B-32<br>laion2b_s34b_b79k<br>ViT-B-32<br>datacomp_xl_s13b_b90k<br>ViT-B-32<br>datacomp_m_s128m_b4k<br>ViT-B-32<br>commonpool_m_clip_s128m_b4k<br>ViT-B-32<br>commonpool_m_laion_s128m_b4k<br>ViT-B-32<br>commonpool_m_image_s128m_b4k<br>ViT-B-32<br>commonpool_m_basic_s128m_b4k<br>ViT-B-32<br>datacomp_s_s13m_b4k<br>ViT-B-32<br>commonpool_s_clip_s13m_b4k<br>ViT-B-32<br>commonpool_s_laion_s13m_b4k<br>ViT-B-32<br>commonpool_s_image_s13m_b4k<br>ViT-B-32<br>commonpool_s_basic_s13m_b4k<br>ViT-B-32<br>commonpool_s_s13m_b4k<br>ViT-B-32-quickgelu<br>openai<br>ViT-B-32-quickgelu<br>laion400m_e31<br>ViT-B-32-quickgelu<br>laion400m_e32<br>ViT-B-32-quickgelu<br>metaclip_400m<br>ViT-B-32-quickgelu<br>metaclip_fullcc<br>ViT-B-16<br>openai<br>ViT-B-16<br>laion400m_e31<br>ViT-B-16<br>laion2b_s34b_b88k<br>ViT-B-16<br>datacomp_xl_s13b_b90k<br>ViT-B-16<br>datacomp_l_s1b_b8k<br>ViT-B-16<br>commonpool_l_clip_s1b_b8k<br>ViT-B-16<br>commonpool_l_laion_s1b_b8k<br>ViT-B-16<br>commonpool_l_image_s1b_b8k<br>ViT-B-16<br>commonpool_l_text_s1b_b8k<br>ViT-B-16<br>commonpool_l_basic_s1b_b8k<br>ViT-B-16<br>commonpool_l_s1b_b8k<br>ViT-B-16-quickgelu<br>metaclip_400m<br>ViT-L-14<br>openai<br>ViT-L-14<br>laion400m_e31<br>ViT-L-14<br>laion400m_e32<br>ViT-L-14<br>laion2b_s32b_b82k<br>ViT-L-14<br>datacomp_xl_s13b_b90k<br>ViT-L-14<br>commonpool_xl_clip_s13b_b90k<br>ViT-L-14<br>commonpool_xl_laion_s13b_b90k<br>ViT-L-14<br>commonpool_xl_s13b_b90k<br>ViT-L-14-quickgelu<br>metaclip_400m<br>ViT-L-14-quickgelu<br>metaclip_fullcc<br>ViT-L-14-quickgelu<br>dfn2b<br>ViT-H-14<br>laion2b_s32b_b79k<br>ViT-H-14-quickgelu<br>metaclip_fullcc<br>ViT-H-14-quickgelu<br>dfn5b<br>ViT-g-14<br>laion2b_s12b_b42k<br>ViT-g-14<br>laion2b_s34b_b88k<br>ViT-bigG-14<br>laion2b_s39b_b160k<br>roberta-ViT-B-32<br>laion2b_s12b_b32k<br>xlm-roberta-base-ViT-B-32<br>laion5b_s13b_b90k<br>xlm-roberta-large-ViT-H-14 frozen_laion5b_s13b_b90k|39.95% 55.85%<br>6.72% 19.30% 71.63%<br>55.35%<br>41.32% 48.79%<br>8.65% 19.69% 65.38%<br>49.72%<br>41.54% 62.20% 11.58% 23.51% 72.24%<br>54.00%<br>40.08% 55.89%<br>6.63% 19.06% 70.51%<br>55.14%<br>41.97% 47.70%<br>9.18% 17.98% 65.41%<br>48.63%<br>39.72% 55.85%<br>7.02% 19.73% 72.01%<br>54.88%<br>40.44% 62.70% 10.53% 19.48% 76.52%<br>56.22%<br>39.88% 55.51%<br>7.14% 19.31% 71.07%<br>54.47%<br>40.06% 60.79% 11.23% 24.12% 67.56%<br>41.34%<br>45.44% 62.96% 14.75% 22.29% 79.33%<br>57.27%<br>45.31% 62.95% 14.65% 22.07% 79.13%<br>57.27%<br>45.59% 60.13% 15.34% 20.23% 78.60%<br>56.26%<br>45.56% 60.12% 15.30% 20.58% 78.10%<br>56.28%<br>45.62% 59.92% 15.09% 20.24% 78.17%<br>56.19%<br>40.29% 58.53%<br>8.93% 19.93% 73.43%<br>55.70%<br>41.58% 58.35% 10.60% 19.63% 71.95%<br>54.76%<br>40.25% 59.00%<br>8.92% 20.45% 71.89%<br>55.81%<br>41.14% 58.33%<br>9.95% 20.28% 73.66%<br>55.54%<br>40.86% 58.32%<br>9.81% 19.54% 73.41%<br>55.24%<br>28.98% 55.47%<br>1.31% 18.44% 61.73%<br>54.16%<br>33.96% 59.22%<br>4.62% 21.46% 68.17%<br>56.34%<br>28.99% 55.50%<br>1.37% 18.45% 61.43%<br>54.27%<br>28.98% 55.47%<br>1.33% 18.31% 61.73%<br>54.16%<br>32.31% 57.03%<br>3.39% 19.01% 64.84%<br>54.89%<br>32.45% 58.97%<br>4.80% 22.63% 67.50%<br>55.64%<br>40.06% 60.79% 11.22% 24.18% 67.56%<br>41.34%<br>45.19% 61.19% 14.75% 21.76% 78.27%<br>55.67%<br>45.08% 61.14% 14.83% 21.88% 78.45%<br>55.60%<br>44.27% 60.73% 12.81% 20.28% 77.68%<br>57.23%<br>44.63% 59.75% 14.05% 20.61% 79.09%<br>56.34%<br>39.33% 61.40% 11.30% 26.02% 70.98%<br>51.96%<br>44.66% 61.46% 14.66% 22.60% 78.07%<br>56.09%<br>45.28% 60.81% 15.28% 20.60% 77.72%<br>57.44%<br>45.31% 59.96% 15.39% 20.42% 76.76%<br>56.48%<br>44.41% 59.45% 14.12% 20.32% 76.99%<br>55.91%<br>44.06% 60.71% 14.00% 20.73% 76.82%<br>55.88%<br>44.53% 60.45% 14.22% 20.40% 76.66%<br>56.24%<br>44.02% 58.72% 13.86% 19.56% 76.50%<br>55.23%<br>44.48% 60.21% 14.02% 20.67% 78.46%<br>56.91%<br>44.19% 59.99% 13.89% 19.79% 77.98%<br>56.47%<br>43.11% 58.31% 13.38% 19.10% 75.81%<br>54.20%<br>43.88% 61.31% 13.28% 21.47% 79.17%<br>57.95%<br>38.09% 59.76% 11.05% 24.82% 69.15%<br>52.42%<br>44.09% 63.61% 14.89% 21.92% 80.22%<br>57.90%<br>44.09% 63.43% 14.86% 21.65% 80.04%<br>57.78%<br>45.19% 58.82% 15.25% 19.82% 76.96%<br>56.73%<br>44.37% 63.19% 15.37% 22.43% 79.99%<br>59.45%<br>44.63% 61.28% 15.52% 21.37% 77.50%<br>58.05%<br>45.34% 62.77% 15.77% 21.50% 79.55%<br>59.30%<br>43.50% 62.09% 14.49% 21.65% 76.41%<br>58.67%<br>42.91% 61.60% 12.90% 21.43% 78.62%<br>59.11%<br>43.02% 61.92% 13.70% 21.93% 81.63%<br>59.22%<br>46.37% 60.27% 16.88% 20.65% 80.08%<br>58.98%<br>45.32% 63.44% 15.51% 21.35% 79.06%<br>60.24%<br>43.12% 62.44% 13.65% 21.72% 81.02%<br>59.99%<br>46.39% 60.99% 16.93% 21.21% 82.36%<br>60.19%<br>45.06% 61.44% 15.83% 21.22% 78.95%<br>58.44%<br>44.95% 63.72% 15.58% 22.15% 77.94%<br>60.40%<br>45.09% 62.14% 15.67% 20.48% 79.37%<br>60.44%<br>12.58% 47.85%<br>0.04% 14.40% 48.48%<br>47.37%<br>12.96% 50.62%<br>0.03% 16.57% 49.08%<br>50.49%<br>11.99% 51.06%<br>0.01% 17.67% 50.67%<br>50.59%|



**Table 10** Summarized Dense Captions test results on OpenCLIP models. We compare various baseline models on our Subcrop-Caption Matching (SCM) and negatives tests. 

13 

## **F DCI Dataset Datasheet** 

The following are our answers to the Datasheets for Datasets (Gebru et al., 2021) question list. 

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** Not to the author’s knowledge 

## **F.3 Collection Process** 

## **F.1 Motivation** 

**For what purpose was the dataset created?** To create an initial dataset of highly aligned text and image pairs that were not yet available, primarily for evaluating how well existing models can make use of all of the data. 

**Who created the dataset and on behalf of which entity?** Researchers on Meta’s FAIR research team created it on their own behalf. 

**Who funded the creation of the dataset?** Meta 

## **F.2 Composition** 

**What do the instances that comprise the dataset represent?** Images with text annotations 

**How many instances are there in total?** 7805 images with complete mask-aligned annotations. 

**Does the dataset contain all possible instances or is it a sample of instances from a larger set?** Images were sampled from a random subset of SA-1B’s (Kirillov et al., 2023) underlying image dataset. 

**What data does each instance consist of?** One image, a top-level caption and description, and then a list of submask-subcaption pairings covering a significant portion of the image’s content 

**Is there a label or target associated with each instance?** Just the text descriptions, no categorization is done. **Is any information missing from individual instances?** Not all of the image is covered in the submask-aligned captions, so the descriptions may still be considered incomplete. 

**Are relationships between individual instances made explicit?** There is no clear relationships between instances in the dataset. 

**Are there recommended data splits?** The dataset is intended **primarily** as a test set, however we also provide a finetuneing train/valid/test split for those wanting to use it for experiments. 

## **Are there any errors, sources of noise, or redundancies in** 

**the dataset?** Though attempts were made to keep the dataset high-quality, annotator error can be present through the dataset, modeling errors may cause some masks to have been omitted, and the LLM-based augmentation for scaling captions to CLIP length may introduce noise as well. 

**Is the dataset self-contained, or does it link to or otherwise rely on external resources?** Self-contained 

**Does the dataset contain data that might be considered confidential?** Not to the authors’ knowledge 

**How was the data associated with each instance acquired?** The images were selected from the SA-1B image dataset, and underwent a combination of automated and manual annotation 

**Whatmechanismsorprocedureswereusedtocollectthe data?** We use the Mephisto framework, as well as custom annotation interfaces, to collect the data. Complete details are available on the project’s GitHub. 

**If the dataset is a sample from a larger set, what was the sampling strategy?** Random selection from a single subset 

**Who was involved in the data collection process and how were they compensated?** Crowdworkers were paid well above minimum wage for their time spent. **Over what timeframe was the data collected?** Spring through Fall of 2023. 

**Were any ethical review processes conducted?** This collection process underwent internal review. 

## **F.4 Preprocessing/cleaning/labeling** 

**Was any preprocessing/cleaning/labeling of the data done?** The dataset was preprocessed using the Segment Anything Model in order to identify the regions of the image to be annotated. 

**Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data?** The raw data is unmodified by the extraction process. 

## **Is the software that was used to preprocess/clean/label** 

**the data available?** All of the software used to construct the dataset will be made available alongside the dataset release on the project’s GitHub. 

## **F.5 Uses** 

**Has the dataset been used for any tasks already?** The dataset is used for the Densely Captioned Images test set. 

**Are there tasks for which the dataset should not be used?** The dataset is intended as a test set. Any use outside of this is unplanned by the authors. 

## **F.6 Distribution** 

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?** The dataset will be made broadly available **How will the dataset will be distributed?** A download 

14 

script will be made available on the project GitHub, alongside a copy of the code used to collect and prepare the original dataset. 

**When will the dataset be distributed?** Upon release of the associated publication. 

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** The dataset will be released under CC-By-NC. 

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** Not to the authors’ knowledge. 

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** Not to the authors’ knowledge. 

## **F.7 Maintenance** 

**Who will be supporting/hosting/maintaining the dataset?** Meta’s FAIR team will host this dataset. **How can the owner/curator/manager of the dataset be contacted?** On the project’s GitHub page. **Is there an erratum?** Changes will be noted on the project’s github. 

**Will the dataset be updated?** The authors have no clear schedule to update or alter the dataset. 

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?** All known instances of people in the dataset have been face-blurred as per the SA-1B release, and no retention policy is known. 

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** They may do so from the project GitHub, following the terms included therein. 

15 

