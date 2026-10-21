# **FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication** 

Eric Slyman[1] _[,]_[2][*] Stefan Lee[1] Scott Cohen[2] Kushal Kafle[2] 1Department of EECS, Oregon State University 2Adobe 

## **Abstract** 

_Recent dataset deduplication techniques have demonstrated that content-aware dataset pruning can dramatically reduce the cost of training Vision-Language Pretrained (VLP) models without significant performance losses compared to training on the original dataset. These results have been based on pruning commonly used imagecaption datasets collected from the web – datasets that are known to harbor harmful social biases that may then be codified in trained models. In this work, we evaluate how deduplication affects the prevalence of these biases in the resulting trained models and introduce an easy-toimplement modification to the recent SemDeDup algorithm that can reduce the negative effects that we observe. When examining CLIP-style models trained on deduplicated variants of LAION-400M, we find our proposed FairDeDup algorithm consistently leads to improved fairness metrics over SemDeDup on the FairFace and FACET datasets while maintaining zero-shot performance on CLIP benchmarks._ 

## **1. Introduction** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0001-06.png)


**----- Start of picture text -----**<br>
Zero-Shot ImageNet  Zero-Shot FACET<br>(Accuracy@5  ↑ ) (Gender Disparity in Recall  ↓ )<br>89.9% 89.7% 89.7% .346<br>.303 .298<br>.104 .113 .109<br>LAION-400M SemDeDup FairDeDup LAION-400M SemDeDup FairDeDup<br>(100% Data) (50% Data) (50% Data) (100% Data) (50% Data) (50% Data)<br>Task Performance<br>Equality of Oppourtunity<br>Average Worst-ClassWorst-Group Average Worst-Class Average Worst-Class<br>**----- End of picture text -----**<br>


Figure 1. Training models on deduplicated data can yield similar results to the full-data setting on standard tasks like zero-shot ImageNet [16] classification ( **left** , higher is better _↑_ ). However, impacts on subgroup performance have not been studied. We discover cases such as gender disparity ( **right** , lower is better _↓_ ) where deduplication reinforces existing biases on FACET [26]. FairDeDup preserves performance while reducing bias from deduplication and, in some cases, w.r.t. the full-data setting. 

Recent Vision-Language Pretrained (VLP) models [55] that learn to align image and language encodings have demonstrated strong zero-shot performance on many standard perception tasks [12, 16, 71, 73]. Beyond these, VLP models have enabled complex downstream applications ranging from visually-aware chatbots [42, 44] and language-based image segmentation [37, 79] to instruction-guided robotics [62, 69] and semantic mapping of 3D scenes [35, 61]. The rapid adoption and widespread impact of these models is due in part to the incredibly broad range of content they can represent effectively – a scope far exceeding prior models trained on manually-curated, closed-world datasets [16, 43]. To acquire this capability, VLP models are trained on massive open-world datasets of image-caption pairs collected from the internet [58]. VLP models improve reliably 

> *Work conducted during Slyman’s 2023 summer internship at Adobe. 

with additional training data [13], driving the number of examples in these datasets into the billions [59]. This scale of uncurated data introduces at least two challenges – 1) training can be extremely costly, and 2) manual data curation to reduce undesirable social biases is economically prohibitive. In this work, we explore how dataset deduplication techniques developed to reduce training costs may exacerbate or ameliorate these biases in trained models. 

While larger pretraining datasets generally yield better model performance [13], the massive web-scraped datasets commonly used for training VLP models contain many identical samples (duplicates) or samples that capture nearly the same content under similar imaging conditions (semantic duplicates [1]). Several recently developed techniques for data pruning/deduplication have demonstrated that aggressive removal of these duplicates has limited im- 

1 

pact on the task performance of trained models [1, 48, 63]. For example, Abbas et al. [1] found that pruning LAION400M [58] by 50% resulted in trained models that achieved average performance within 0 _._ 5% of their full-data analogs across a range of common benchmark tasks – effectively cutting training time in half. 

However, these web-scale datasets contain a plethora of problematic social biases and harmful stereotypes [5, 6, 23]. These biases can often then be reflected in the behavior of models trained on these datasets [2, 27, 28, 68]. To better understand and reduce these potential harms, there is increased interest in analyzing the composition of these datasets and their downstream effects on trained models [5, 23, 77]. Deduplication techniques introduce another algorithmic step between dataset and model training that may systematically alter the data distribution – potentially amplifying, maintaining, or reducing the effect of dataset biases. Given that deduplication techniques will likely be widely deployed as cost-saving measures, understanding how their design affects the behavior of downstream models in terms of bias and fairness is a timely but unexamined question. 

To study this question, we investigate the fairness outcomes of CLIP-style [55] VLP models trained on the LAION-400M dataset [58] pruned with SemDeDup [1]. Replicating the results of Abbas et al. [1], we find task performance on CLIP Benchmark [12] is only marginally affected; however, evaluation on the fairness-focused FairFace [34] and FACET [26] datasets suggest deduplication results in mixed effects compared to the full-data setting. We observe increased disparities across gender, but both positive and negative changes for disparities across skin tone and age. Based on these findings, we propose FairDeDup – a fairness-aware data pruning algorithm that makes pruning decisions to improve representation of specified _sensitive concepts_ ( _e.g_ ., _gender_ , shown in Fig. 1). The implementation of FairDeDup is a simple modification to SemDeDup and specifying concepts can be done in natural language. Our large-scale experiments show that FairDeDup leads to improved fairness outcomes comapred to SemDeDup while maintaining comparable performance on standard zero-shot and retrieval-based performance benchmarks. To better understand the deduplication process, we run a smaller scale study deduplicating demographiclabeled data – finding that FairDeDup consistently retains more images depicting minority classes than SemDeDup. 

**Contributions.** We summarize our contributions below: 

- We conduct, to our knowledge, the first large-scale experiment evaluating the fairness outcomes of training largescale vision language models on pruned data – training CLIP-style models on full and deduplicated versions of the popular LAION-400M dataset then evaluating on standard fairness benchmarks for VLP models. 

- We find that models trained on SemDeDup [1] pruned data 

have varied effects on fairness outcomes from the full-data model; reinforcing some biases and mitigating others. 

- We introduce FairDeDup, a simple and efficient modification to SemDeDup that improves fairness outcomes while retaining task performance – improving fairness outcomes over SemDeDup in nearly all cases studied. 

## **2. Related Work** 

**Vision-Language Fairness.** Vision and language models have been shown to learn, reflect, and amplify problematic social biases. For example, vision systems have been shown to dehumanize minority groups by identifying them as animals [19] and degrade in task performance on intersectional combinations of gender and skin tone [9]. Likewise, language models are known to learn gendered associations of professions [8], increase sentiment-intensity along racial lines [38], and a myriad of other problems documented in [7, 65]. Vision-language models are not exempt from these problems [27, 47, 52] and can even reinforce them [64, 77]. 

Contemporary Vision-Language Pretrained models are frequently pretrained on massive but uncurated data scraped from the internet [11, 32, 40, 55]. While web-scale data is shown to improve performance, it also teaches models “ _misogyny, pornography, and malignant stereotypes_ ” [5]. VLP models demonstrate dehumanizing behavior with respect to racial subgroups in zero-shot text-image retrieval [2, 3], show bias related to gender [23, 27, 28, 68], age [23] and skin tone [23, 28, 68, 76] in image captioning, and also demonstrate biases relating to age, gender, skin tone, and ethnicity in text-image retrieval [23, 78]. These behaviors are attributed to the use of uncurated web-scale datasets in pretraining VLP models [5, 6, 23]. 

Mitigations for bias in VLP models typically include fairness-aware training [75] or post-hoc methods to disentangle useful concepts from sensitive attributes [3, 14, 60]. Unlike these methods, we seek to prevent bias from being reinforced in the dataset, rather than removing bias from the model itself. Though early vision-language fairness literature frequently calculates WEAT [10] and SEAT [49] embedding association measures extended for the multimodal setting [31, 57], these measures have been shown to be overly sensitive to small changes in model architecture and outputs [3]. As such, VLP model fairness is primarily evaluated on CelebA [45] and FairFace [34]. Recent datasets such as PHASE [23] and FACET [26] allow for the study of bias on “in the wild” data across diverse subgroups. 

**Dataset Pruning.** Several techniques exist for reducing the size of a dataset while preserving, or even improving, performance. We consider all techniques under this umbrella as _dataset pruning_ algorithms. Coreset selection chooses a weighted subset of training samples which closely estimate the full dataset’s gradient [25, 50] to perform data- 

2 

efficient training with little loss in performance. However, these methods do not scale well with dataset size and frequently require class labels [63]. The most similar work to ours among coreset selection algorithms is the recent D2 Pruning [48]. D2 Pruning utilizes graph based methods to select samples that are both _hard_ and _diverse_ across a data distribution. While promising, D2 Pruning does not evaluate any fairness outcomes and is only demonstrated to scale to DataComp Small (12.8M) [22], a low accuracy setting for VLP models peaking around 5% top-1 zero-shot ImageNet [16] accuracy. In comparison, base-sized CLIP-style models can range from 67-74% accuracy with web-scale data on the same task. We refer readers to [54] for a more thorough review of coreset selection algorithms. 

Large-scale deduplication typically attempts to find exact perceptual duplicates using techniques like perceptual hashing [20] or filtering [22] on image-text CLIP scores and target classes ( _e.g_ ., filtering to images close to ImageNet classes). Abbas et al. [1] introduces the concept of semantic duplicates, images with similar semantic meaning that are not perceptually the same image, alongside SemDeDup, a formalized version of the unsupervised deduplication algorithm from [63]. SemDeDup has been shown to be capable of significantly reducing dataset size with only marginal impact on performance. We choose to study SemDeDup due to the ubiquity of its underlying selection method among contemporary deduplication algorithms - cosine similarity between samples - and scalable nature. To our knowledge, we are the first to study the effect of data pruning on the fairness outcomes of VLP models and study the effects of fairness-aware pruning on their behavior. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0003-02.png)


**----- Start of picture text -----**<br>
1 Embed & Cluster Dataset<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0003-03.png)


**----- Start of picture text -----**<br>
CLIP<br>Web-Scale Feature K-Means<br>Data Extraction Clustering<br>2 Prune Semantic Duplicates Within Clusters<br>SemDeDup FairDeDup (ours)<br>Maximize distance to Maximize similarity to<br>cluster centroids underrepresented concepts<br>**----- End of picture text -----**<br>


Figure 2. The semantic deduplication pipeline following three clusters (�,�,�) with two subgroups (�,�). Connected shapes are duplicates. We **(1) embed** all images from the dataset with a pretrained model then partition with _k_ -means to enable efficient search during **(2) deduplication** . We make a simple modification to the maximum distance selection heuristic used by Abbas et al. [1] **(left)** to improve subgroup diversity by preserving samples which maximize similarity to poorly represented sensitive concepts according to user-specified concept prototypes ( **right** ). 

## **3. FairDeDup: Fair Semantic Deduplication** 

There frequently exists sensitive attributes in data for which it is desirable to obtain some notation of fairness [21]. For example, we may seek _demographic parity_ for _gender_ so that individuals do not receive differing treatment based on their gender identity. Such outcomes are usually based on social norms, organizational ethics, or even codified into discrimination law [4, 15, 29, 53]. Our goal is to improve post-deduplication fairness outcomes concerning these sensitive groups. To achieve this, we propose boosting the representation of underrepresented sensitive subgroups on the internet ( _e.g_ ., women of color) in the post-pruning dataset distribution. We allow for user-defined natural language _sensitive concepts_ , which captures these subgroups for consideration in the deduplication process, and leverage them to bias the selection of preserved samples towards those concepts which are currently underrepresented. 

## **3.1. Preliminaries: SemDeDup** 

We implement FairDeDup as a lightweight modification to the SemDeDup algorithm, which we describe here for com- 

pleteness. Abbas et al. [1] identify that pruning both exact perceptual duplicates ( _e.g_ ., copies of the same image) and those that carry redundant semantic information ( _e.g_ ., many photos of the same object from differing angles), denoted semantic duplicates, is helpful for improving the data efficiency of training large models. To achieve this, they propose SemDeDup [1], an extension of the unsupervised pruning metric from Sorscher et al. [63] to web-scale data. 

To identify duplicates, SemDeDup first leverages pretrained foundation models ( _e.g_ . CLIP [55]) to embed all images in the dataset into a semantically meaningful feature space. Na¨ıvely thresholding embedding similarity between all points to detect duplicates requires _O_ ( _n_[2] ) pairwise comparisons and is intractable for web-scale data like LAION-400M, which requires computing _≈_ 1 _._ 5 _×_ 10[17] cosine similarities. To mitigate this, the dataset is partitioned using an efficient _K_ -means algorithm under the assumption that pairwise similarity need only be calculated for approximately similar samples. SemDeDup then considers the resulting _O_ ( _n_[2] _/k_ ) pairwise similarities on an independent per 

3 

cluster basis. Within each cluster, they determine sets of samples within a 1 _−ϵ_ similarity threshold as duplicates and keep only the sample most distant from the cluster centroid. While this selection heuristic is motivated by the hardness hypothesis of Sorscher et al. [63], ablations show that the algorithm is robust to choosing even a random sample. 

## **3.2. FairDeDup** 

Due to the robustness of SemDeDup to the choice of selection heuristic on performance, we seek instead to replace the heuristic with one that can support our fairness motivation. We provide an overview following shared and unique steps of SemDeDup and FairDeDup in Fig. 2. 

**Sensitive Concept Prototypes.** Given a list of user-defined sensitive concepts _C_ that are desired to be represented in the pruned dataset, we denote the _concept prototype Pi_ for a sensitive concept _Ci∈C_ as the average text embedding of the set of captions generated from template strings ( _e.g_ ., “ _A photo of a {Ci}_ ”) capturing that concept. As is common for VLP models, we assume the embedding model supporting image clustering can also produce image-text alignment scores [32, 41, 55, 72, 74] and consider the case where alignment is determined as the cosine similarity between the representations produced by a vision Φ _I_ : _I →_ R _[d]_ and text Φ _T_ : _T →_ R _[d]_ encoder: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0004-04.png)


We measure how well an image aligns with a sensitive concept by measuring the image-text similarity between that image and the concept prototype _sim_ ( _I, Pi_ ). We choose concepts that both relate to commonly protected demographic subgroups of people and are annotated in common fairness datasets, such as ones based in race and gender. Additional details on the selection of sensitive concepts (Sec. 10) and a list of all concepts used (Sec. 12) are given in the appendix. While this work focuses on text-based prototypes, we note that our methodology trivially extends to image-based ones and beyond, as described in Sec. 6. 

**Sample Preservation Heuristic.** To determine which samples to prune, we consider duplicate _neighborhoods_ : the set of images within 1 _−ϵ_ similarity of a given point, and preserve only one sample from each neighborhood. For each cluster produced by _k_ -means, we track the running average similarity between preserved samples in that cluster and the sensitive concept prototypes. Until all samples are visited, we randomly select an unvisited sample, calculate the similarity between all samples in its neighborhood and the prototypes, and keep only the sample that maximizes similarity to the least similar running average prototype, marking all points in the neighborhood as visited. We preserve the sample with the highest average similarity across concept prototypes for the first neighborhood visited in a cluster. 

|1|# Input: prototypes, embeddings, eps|
|---|---|
|2|# Get similarity with concept prototypes|
|3|proto = embeddings @ prototypes.T|
|4||
|5|balance = AverageMeter(prototype.shape[0])|
|6|tovisit = torch.ones(embeddings.shape[0])|
|7|while tovist.any():|
|8|# Find an unvisited neighborhood|
|9|node = torch.where(tovisit)[0][0]|
|10|sims = embeddings[node] @ embeddings.T|
|11|neighbors = torch.where(sims > 1 - eps)[0]|
|12||
|13|# Maximize least represented concept|
|14|c = balance.get_min_concept()|
|15|point = proto[neigbors][:, c].argmax()|
|16|balance.update(point)|
|17||
|18|log_and_keep(point)|
|19|tovisit[neighbors] = 0|



Figure 3. PyTorch-style pseudo-code for FairDeDup selection given concept prototypes, within cluster embeddings, and an eps similarity threshold for determining neighborhoods. We omit the base case where the first sample selected within a cluster is the one with the highest average concept prototype similarity. 

We track running average similarity on a per cluster basis for two reasons: 1) to avoid a synchronous update step between workers processing clusters in parallel and 2) to prevent algorithmic “gaming” of the selection criteria by balancing concept representation on clusters which highly represent a concept due to some stereotyped notion. Given two clusters primarily composed of doctors and nurses, for example, per cluster processing prevents balancing underselection of feminine presenting doctors by overselecting feminine presenting nurses. We provide pseudo-code for the FairDeDup selection heuristic in Fig. 3. We visualize random samples after pruning a cluster manually identified to be primarily composed of people with FairDeDup and the SemDeDup maximum distance selection heuristic in Fig. 4, and show additional examples in the appendix (Sec. 11). 

## **4. Experiments** 

To assess the effect of deduplication on learned VLP models, we train CLIP-style models on variants of LAION400M [58] and evaluate their performance on both standard and fairness-oriented benchmarks for zero-shot classification and text-image retrieval. 

## **4.1. Models and Training** 

We train all models on LAION-400M [58] as a web-scale dataset representative of those typically used for largescale vision-language pretraining. LAION-400M contains image-text pairs extracted from Common Crawl[1] filtered to 

> 1https://commoncrawl.org/ 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0005-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0005-01.png)


**----- Start of picture text -----**<br>
(a) Maximum Distance Selection<br>(b) FairDeDup Selection<br>**----- End of picture text -----**<br>


Figure 4. A random sampling of preserved samples from a cluster primarily composed of medical professionals after deduplication. FairDeDup improves selection diversity featuring increased variability in age, skin tone, and gender presentation. 

have image-text CLIP similarity _≥_ 0 _._ 3 without significant further curating. This makes LAION-400M an ideal test case for our setting as it is sufficiently large to train VLP models, captures bias from the internet, and is expected to contain semantically redundant samples. At the time of our data collection, only 375M image-text pairs from LAION400M were still available for download. 

We train CLIP-ViT-Base/16 [55] models from the OpenClip [30] implementation with vision transformer [18] base (ViT-B-16) as the image encoder and text transformer [67] as the text encoder. We perform distributed training over 80-120 A100 GPUs depending on the model with a global batch size of 33,820 image-caption pairs for 16 epochs regardless of dataset size. We use the AdamW [46] optimizer with linear warm up and cosine annealed learning rate 

schedule peaking at 5 _×_ 10 _[−]_[4] . Additional hyperparameter details are provided in the appendix (Sec. 9). 

We evaluate CLIP training on three LAION-400M data settings for _performance_ and _fairness_ : 

**Baseline: LAION-400M.** We train a CLIP model on the full LAION-400M dataset for a total of 183k steps as a control by which to evaluate baseline performance and fairness. A good model in the deduplicated setting should perform similarly to this model on common benchmarks without negatively impacting subgroup disparity and skew. 

**SemDeDup LAION-400M.** For SemDeDup, we use a CLIP-ViT-Base/16 trained on WebImageText (WIT) [55] to produce image embedding which are partitioned into 50 _,_ 000 clusters using the FAISS [33] implementation of _k_ -means and set the _ϵ_ threshold for identifying duplicates within each cluster such that 50% of samples are pruned. 

**FairDeDup LAION-400M.** We leverage the same WIT trained CLIP model for FairDeDup as SemDeDup. We consider 110 sensitive concepts capturing intersectional combinations of age, gender, skin tone, race and ethnicity and represent them using embeddings of 330 corresponding captions (three each with minor syntactic variation). We use the average across captions of the same concept as our prototypes. We enumerate all sensitive concepts and templates used to generate the text prototypes in the appendix (Sec. 12). The selection step can be parallelized across CPUs up to the number of clusters produced by _k_ -means. We find that selection in this setting on a 32 CPU machine takes one hour on average and that the overall time is dominated by the shared GPU parallelizable embedding and clustering steps. 

## **4.2. Datasets and Metrics** 

We evaluate across three benchmarks to validate if models trained on deduplicated data are both _performant_ and _fair_ . 

**Zero-Shot Classification and Retrieval.** We evaluate the performance of each model across 41 common zeroshot classification and retrieval datasets from Clip Benchmark [12] such as ImageNet [16], Flickr30k [71], and VTAB [73]. A model trained on deduplicated data should perform at least as well as a model trained in the full-data setting on these benchmarks. 

**Fair Zero-Shot Classification.** The FACET [26] dataset contains expert reviewer annotations for 52 person related classes, gender presentation, skin tone, age, and other attributes, on a 32k image subset of Segement Anything 1 Billion (SA-1B) [36]. We perform zero-shot classification over the 52 person-classes by constructing a text prompt ( _e.g_ ., “ _A photo of a {class}_ ”) for each class and predicting the class used to construct the prompt with highest similarity to the image. Given a model _f_ , sensitive attribute label _l_ , person- 

5 

class _C_ , and set of of images _Il[C]_ which captures class _C_ featuring a person with label _l_ , we measure the average and worst-class disparity in recall between subgroups of sensitive attributes where disparity is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0006-01.png)


Conceptually, a large magnitude disparity indicates that a model better predicts positive instances of person-class _C_ for one of the two subgroups, while a disparity of zero indicates _equality of opportunity_ between subgroups. 

We evaluate subgroup disparity for average perceived gender expression by masculine _vs_ . feminine presentation, lighter (1-4MST[2] ) _vs_ . darker (6-10MST) skin tone, and middle _vs_ . younger and middle _vs_ . older age for all personclasses which have at least 25 samples in both subgroups. Gustafson et al. [26] consider only the 21k images capturing a single person in their disparity analysis for simplicity and alignment between tasks ( _e.g_ ., classification and visual grounding). To increase the sample size of our analysis, we consider each person in the dataset as a unique sample. We expand the bounding box for each person by 20% to capture context before extracting a square-padded image crop centered on the box, yielding 49,551 images. 

**Fair Image Retrieval.** FairFace [34] annotates a balanced dataset of 108k cropped faces from YFCC-100M [66] by seven racial groups with additional annotations for perceived gender and age. Similar to [3, 14, 60], we measure the degree to which the top- _k_ results of an image-text query differ over values of sensitive attributes in the 11k image validation set with respect to the desired proportion of those values with MaxSkew@1000 [24]. Given the top- _k_ images _τr[k]_[returned by image-text query] _[ r]_[, let the actual proportion] of images returned by the query for a particular value _ai∈A_ of sensitive attribute _A_ be _Pτr k[,r,a][i][∈]_[[0] _[,]_[ 1]][and][the][desired] proportion be _Pq,r,ai∈_ [0 _,_ 1], then the skew of value _ai_ is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0006-05.png)


One limitation of Skew@k is that it is defined only for a single value of a sensitive attribute. To give a more holistic view across all values that a sensitive attribute may take on, we report the most skewed _ai_ with MaxSkew@k: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0006-07.png)


Conceptually, MaxSkew indicates the “ _largest unfair advantage_ ” [24] provided to images with a particular value of the sensitive attribute for appearing in the the top- _k_ results of the query. We choose the desired proportion of images to be the same as the true distribution of those images 

> 2Monk Skin Tone scale [51] 

||**Full Data**<br>(100%)|**SemDeDup**<br>(50%)|**FairDeDup**<br>(50%)|
|---|---|---|---|
|IN1K_acc_@5|.899|.897(–.002)|.897(–.002)|
|INV2_acc_@5|.845|.841(–.004)|.837(–.008)|
|C10_acc_@5|.999|.998(–.001)|.999(–.000)|
|C100_acc_@5|.934|.934(–.000)|.939(+.005)|
|Flickr_R_@5|.873|.874(+.001)|.871(–.002)|
|COCO_R_@5|.633|.632(–.001)|.626(–.007)|



Table 1. Common zero-shot and text-image retrieval benchmarks for CLIP models on ImageNet1K [16], ImageNetV2 [56], CIFAR [39] (C10/C100), Flicker30k [71], and COCO Captions [43]. Higher ( _↑_ ) is better in all cases. The difference in performance from the full-data setting is shown in green (red) when improved (reduced). Both deduplication strategies yield models that preserve the performance of models trained on the full data. 

in the dataset. Under this condition, if the proportion of of _ai_ in the top- _k_ results is the same as its distribution in the dataset, MaxSkew obtains an optimal result of 0 and achieves _demographic parity._ Following [3], we report average MaxSkew@1000 across 240 (un)favorable captions orthogonal to images in the dataset ( _e.g_ ., “ _A photo of a {smart} person_ ”), matching test attributes and prompts for race ( _|A|_ =7), gender ( _|A|_ =2), and age ( _|A|_ =3). Similar to Seth et al. [60], we bin age into larger groups: _younger_ (0-19), _middle_ (20-49), and _older_ (50-70+) to reduce noise. 

We additionally report MinSkew@k, which captures the “ _worst disadvantage in representation_ ” for a subgroup, and the normalized discounted cumulative KL-divergence (NDKL), which captures the weighted average of Skew@k over all attribute values at varying settings of _k_ . Intuitively, MinSkew captures the severity of the most negatively biased subgroup juxtaposed against the most positively biased captured by MaxSkew, and NDKL is a summary statistic over configurations of Skew@k. We refer readers to Geyik et al. [24] for the formulation of MinSkew and NDKL. 

## **5. Results** 

**Deduplication Preserves Aggregate Performance.** In Tab. 1, we report Accuracy@5 for four common zeroshot image classification datasets: ImageNet1K [16], ImageNetV2 [56], CIFAR-10 and CIFAR-100 [39], and Recall@5 for two common image-text retrieval datasets: Flicker30k [71] and COCO Captions [43]. As expected, the performance drop from the full-data setting to deduplicated is marginal ( _≤_ 0 _._ 8%), indicating that performance is preserved after pruning 50% of the training data. The performance gap between the two deduplicated-data models is even smaller ( _≤_ 0 _._ 6%), and neither consistently performs more favorably across tasks. We refer readers to Sec. 13 in the appendix for results on additional datasets and metrics. 

6 

||||**Full**<br>(100%)|**SemDeDup**<br>(50%)|**FairDeDup**<br>(50%)|**Diff.**<br>(FDD-SDD)|
|---|---|---|---|---|---|---|
|**Gender**|Masc / Femm|Mean<br>Max<br>Gap|.104<br>.303<br>.199|.113(+ 9%)<br>.346(+14%)<br>.233(+17%)|**.109 (+ 5%)**<br>**.298 (– 2%)**<br>**.189 (– 5%)**|–.004<br>–.048<br>–.053|
|**Skin Tone**|Light / Dark|Mean<br>Max<br>Gap|.100<br>.354<br>.254|.112(+12%)<br>.342(– 3%)<br>.230(– 9%)|**.105 (+ 5%)**<br>**.320 (–10%)**<br>**.215 (–15%)**|–.007<br>–.022<br>–.015|
|**Age**|Mid / Yng|Mean<br>Max<br>Gap|.063<br>.268<br>.205|**.059 (– 6%)**<br>.230(–14%)<br>.171(–17%)|.075(+19%)<br>**.225 (–16%)**<br>**.150 (–27%)**|+.016<br>–.005<br>–.022|
|**Age**|Mid / Old|Mean<br>Max<br>Gap|.098<br>.252<br>.154|.096(– 2%)<br>.248(– 2%)<br>.152(– 1%)|**.087 (–11%)**<br>**.153 (–39%)**<br>**.066 (–57%)**|–.009<br>–.095<br>–.008|



Table 2. Absolute disparity (Eq. 2) in zero-shot classification performance on FACET [26] averaged across 52 person classes. Larger values indicate a greater performance gap between subgroups when predicting true positive samples of the same occupation. Lower ( _↓_ ) is better for all metrics. Best deduplicated model in **bold** . The percent change in fairness outcomes from the fulldata setting is shown in green (red) when improved (reduced). 

**SemDeDup Has Mixed Effects on Fairness.** We show the result of our zero-shot image classification evaluation on FACET [26] in Tab. 2, studying subgroups across gender, skin tone, and age-based sensitive attributes. We find that SemDeDup yields mixed impacts. SemDeDup reinforces average and worst-class disparity across gender subgroups, exacerbates average disparity in skin tone while mitigating the worst-class, and surprisingly aids in reducing average and worst-class disparity across age groups. 

In Tab. 3, we present the results of our text-image retrieval evaluation on FairFace [34], focusing on subgroups related to gender, race, and age. We again find that SemDeDup demonstrates mixed effects. SemDeDup reinforces gender skew across all metrics but mitigates skew towards the largest unfairly advantaged group (MaxSkew) while magnifying skew away from the worst disadvantaged group. 

**FairDeDup Improves Fairness Over SemDeDup.** FairDeDup improves fairness outcomes over SemDeDup on FACET by mitigating, rather than exasperating, worst-class gender disparity while improving disparity outcomes in all cases except for age between middle-aged and young subgroups. With respect to SemDeDup, FairDeDup reduces the average over groups for mean disparity by _._ 0001 ( _._ 0067 excluding Age Mid/Yng), worst-class by _._ 0425 and gap by _._ 0245. This result demonstrates that FairDeDup more closely achieves _equality of opportunity_ than SemDeDup. 

FairFace also shows evidence that FairDeDup improves fairness outcomes. While both methods increase gender 

|||**Full**<br>(100%)|**SemDeDup**<br>(50%)|**FairDeDup**<br>(50%)|**Diff.**<br>(FDD-SDD)|
|---|---|---|---|---|---|
|**Gender**|MinSkew<br>MaxSkew<br>NDKL|.159<br>.123<br>.010|.223(+40%)<br>.153(+24%)<br>.015(+50%)|**.182 (+14%)**<br>**.125 (+ 2%)**<br>**.012 (+20%)**|–.041<br>–.028<br>–.003|
|**Race**|MinSkew<br>MaxSkew|.545<br>.432|.583(+ 7%)<br>.401(– 7%)|**.513 (– 6%)**<br>**.372 (–14%)**|–.070<br>–.029|
||NDKL|.035|.034(– 3%)|**.030 (–14%)**|–.004|
||MinSkew|.618|.702(+14%)|**.647 (+ 5%)**|–.055|
|**Age**|MaxSkew|.241|**.224 (– 7%)**|.296(+23%)|+.072|
||NDKL|.023|**.022 (– 4%)**|.028(+22%)|+.006|



Table 3. Skew evaluation on FairFace [34] averaged over 240 textimage query templates. As MinSkew is a negative metric optimal at its upper bound of zero, we report its absolute value for readability so that lower ( _↓_ ) is better for all metrics. Best deduplicated model in **bold** . The percent change in fairness outcomes from the full-data setting is shown in green (red) when improved (reduced). 

skew, FairDeDup exhibits a milder skew across all summary metrics. For race, both methods mitigate the effects of the largest unfairly advantaged group (MaxSkew) compared to the baseline, while FairDeDup mitigates the magnitude of MaxSkew and reduces the skew against the worst disadvantaged class (MinSkew) compared to the baseline. Determining the best-performing method for age-based subgroups is inconclusive. Across gender and race groups, FairDeDup reduces MinSkew by _._ 0555, MaxSkew by _._ 0285 and NDKL by _._ 0035. This results demonstrates that FairDeDup better achieves _demographic parity_ than SemDeDup w.r.t. gender and race, even outperforming the full-data setting on race. 

## **6. Discussion** 

Below we discuss observations when pruning smaller-scale annotated data, potential FairDeDup variants for varied concept prototypes, and limitations of our approach. 

**Evaluation on Demographically Annotated Data.** In this paper, we have shown on large-scale real model training that FairDeDup achieves results on-par with SemDeDup on standard benchmarks, while demonstrating improved fairness outcomes. We believe that is the clearest signal about the applicability of FairDeDup in real-world usage. However, we would also like to directly demonstrate that FairDeDup does indeed select more diverse data representations compared to SemDeDup. To do so, we consider deduplicating the FACET [26] images described in Sec. 4. We perform _k_ -means clustering ( _k_ =50) on the images with ten different random seeds and apply both deduplication methods to each. In Tab. 4, we report the percent of the postpruning dataset labeled as non-majority classes for gender ( _feminine_ , _non-binary_ , _other_ ), skin tone ( _MST>_ 4, _other_ ), and age ( _younger_ , _older_ , _other_ ), averaged across the ten 

7 

||**Full Data**|**SemDeDup**|**FairDeDup**|
|---|---|---|---|
||(100%)|(50%)|(50%)|
|Gender|32.92%|31.91%|**32.29%**|
|Skin Tone|51.28%|50.46%|**51.06%**|
|Age|44.74%|43.62%|**44.06%**|



Table 4. Data mass allocated to minority classes in FACET [26] after deduplication averaged over ten random seeds. We consider minority classes by gender, skin tone, and age. The difference between means of SemDeDup and FairDeDup across trials is significant at _>_ 99 _._ 9% confidence ( _n_ =10) for all groups according to a paired t-test. In all cases, we observe that FairDeDup helps recover mass reallocated to the majority class by SemDeDup. 

trials. This analysis indicates that 1) SemDeDup does indeed reduce the frequency of the least well represented subgroups and 2) that FairDeDup mitigates this effect. The difference between means across trials of SemDeDup and FairDeDup is statistically significant at _≥_ 99 _._ 9% confidence ( _n_ =10) across all groups according to a paired t-test. 

**Variants and Applications of the FairDedup Algorithm.** In our experiments, we use text-based prototypes to guide FairDeDup towards balancing representation of sensitive concepts. However, the exact specification of these prototypes is flexible to other subjects ( _e.g_ ., non-person related) and modalities ( _e.g_ ., image-based concepts). FairDeDup can be trivially modified to consider any prototype for which the embedding model can output similarity to individual images, such as sets of semantically aligned images (e.g., based on image type, photographs, illustrations, infographics, etc), or a combination of image and text prototypes. Similarly, FairDedup can be used to boost underrepresented samples from arbitrary sets such as object entities [70], or other forms of semantic organization. 

## **6.1. Limitations** 

**Clustering Restrictions on Selection.** While clustering allows deduplication algorithms to scale to hundreds of millions of samples, it also limits the availability of lowerrepresented samples for balancing sensitive concept representation. Take, for example, a data subset capturing photos of dancers. If the clustering algorithm creates two “dancer” clusters, bifurcating across binary gender presentation, then FairDeDup will be unable to perform significant gender balancing due to the independent processing of each cluster. We note that the resulting balance will be based on a combination of the underlying number of “dancer” photos in the dataset and the rate of duplication within both groups. If the two clusters are approximately equal sized with equal frequency of semantic duplicates, the independent deduplication of both clusters is equivalent in representation to a joint deduplication with respect to the bifurcated attribute. 

We display demonstrative clusters in the appendix (Sec. 8). 

**Bias Transfer From the Embedding Model.** By deduplicating based on model embeddings, we subject the selection of samples to the biases of the embedding model. The majority of sensitive concepts we select are social constructs based in gender and race, and are not identifiable by anyone other than the photographed individual. We therefore expect sensitive concept representation to be based upon the predominate social norms they capture, rather than necessarily true identities of individuals. Nonetheless, we assert that a deduplication method which maintains the bias of the full-data setting is a favorable start to one that magnifies it. 

**Demographic Representation in Fairness Datasets.** Most contemporary fairness datasets lack annotations from the individuals they represent. Consequently, for nonstationary socially constructed attributes such as gender, race, and perceived _young/oldness_ , the captured data relies solely on annotators’ subjective understanding. Additionally, these datasets often limit gender representation to a binary perspective (occasionally including a small “other” category) [17], a necessary operationalization for scale that is not inclusive of bias characterization for diverse gender identities. We also note that fairness datasets cover a limited number of directions under which a model may express bias, excluding disability, national origin, and other sensitive attributes. Our analysis, therefore, only examines fairness outcomes with respect to contemporary and subjective evaluation of these limited available demographic attributes. 

## **7. Conclusion** 

In this paper, we study the fairness outcomes resulting from training large-scale vision-language models on semantically deduplicated web-scale data, using LAION-400M and SemDeDup as a representative dataset and deduplication algorithm pairing. We find that deduplication has consistently harmful effects on gender-based bias and mixed effects on skin tone/race- and age-based biases across zero-shot classification and text-image retrieval tasks. To improve fairness outcomes, we propose FairDeDup, a simple and efficient fairness-aware modification of the sample selection heuristic in SemDeDup which boosts the representation of userdefined sensitive concepts in the post-deduplication data distribution. Our experiments show that FairDeDup preserves the performance of the full-data setting on standard metrics for common image-text datasets, has more favorable fairness outcomes than SemDeDup across all cases for gender- and skin tone/race-based biases, and outperforms the baseline full-data setting in several instances. We hope for FairDeDup to provide a simple and tractable baseline for future work in fairness-aware deduplication. 

8 

## **References** 

- [1] Amro Abbas, Kushal Tirumala, D´aniel Simig, Surya Ganguli, and Ari S. Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. _arXiv preprint arXiv:2303.09540_ , 2023. 1, 2, 3, 6 

- [2] Sandhini Agarwal, Gretchen Krueger, Jack Clark, Alec Radford, Jong Wook Kim, and Miles Brundage. Evaluating clip: towards characterization of broader capabilities and downstream implications. _arXiv preprint arXiv:2108.02818_ , 2021. 2 

- [3] Hugo Berg, Siobhan Hall, Yash Bhalgat, Hannah Kirk, Aleksandar Shtedritski, and Max Bain. A prompt array keeps the bias away: Debiasing vision-language models with adversarial learning. In _Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing_ , pages 806–822, 2022. 2, 6 

- [4] Joseph R. Biden and The White House. Executive order on the safe, secure, and trustworthy development and use of artificial intelligence, 2023. Available: https : / / www . whitehouse . gov / briefing - room / presidential - actions / 2023 / 10 / 30 / executive - order - on - the - safe - secure - and-trustworthy-development-and-use-ofartificial-intelligence/. 3 

- [5] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornography, and malignant stereotypes. _arXiv:2110.01963 [cs]_ , 2021. 2 

- [6] Abeba Birhane, Vinay Prabhu, Sang Han, Vishnu Naresh Boddeti, and Alexandra Sasha Luccioni. Into the laions den: Investigating hate in multimodal datasets. _Neural Information Processing Systems_ , 2023. 2 

- [7] Su Lin Blodgett, Solon Barocas, Hal Daum´e III, and Hanna Wallach. Language (technology) is power: A critical survey of “bias” in NLP. In _Association for Computational Linguistics_ , pages 5454–5476, 2020. 2 

- [8] Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam Tauman Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In _Neural Information Processing Systems_ , 2016. 2 

- [9] Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In _ACM Conference on Fairness Acountability and Transparency_ , 2018. 2 

- [10] Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. Semantics derived automatically from language corpora contain human-like biases. _Science_ , 356(6334):183–186, 2017. 2 

- [11] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Alexander Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and 

Radu Soricut. Pali: A jointly-scaled multilingual languageimage model. In _International Conference on Learning Representations_ , 2023. 2 

- [12] Mehdi Cherti and Romain Beaumont. Clip benchmark, 2023. GitHub repository https://github.com/ LAION-AI/CLIP_benchmark. 1, 2, 5, 4 

- [13] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2818–2829, 2023. 1 

- [14] Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. Debiasing visionlanguage models via biased prompts. _arXiv preprint arXiv:2302.00070_ , 2023. 2, 6 

- [15] U.S. Equal Employment Oppourtunity Comission. Title vii of the civil rights act of 1964, 1964. Available: https:// www.eeoc.gov/statutes/title-vii-civilrights-act-1964. 3 

- [16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _Computer Vision and Pattern Recognition_ , pages 248–255, 2009. 1, 3, 5, 6, 4 

- [17] Hannah Devinney, Jenny Bj¨orklund, and Henrik Bj¨orklund. Theories of “gender” in nlp bias research. In _ACM Conference on Fairness, Accountability, and Transparency_ , pages 2083–2102, 2022. 8 

- [18] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _International Conference on Learning Representations_ , 2021. 5, 1 

- [19] Conor Dougherty. Google photos mistakenly labels black people ‘gorillas’. _New York Times_ , 2015. 2 

- [20] Ling Du, Anthony T.S. Ho, and Runmin Cong. Perceptual hashing for image authentication: A survey. _Signal Processing: Image Communication_ , page 115713, 2020. 3 

- [21] Jade S. Franklin, Karan Bhanot, Mohamed Ghalwash, Kristin P. Bennett, Jamie McCusker, and Deborah L. McGuinness. An ontology for fairness metrics. In _AAAI/ACM Conference on AI, Ethics, and Society_ , page 265–275, 2022. 3 

- [22] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In search of the next generation of multimodal datasets. _arXiv preprint arXiv:2304.14108_ , 2023. 3 

9 

- [23] Noa Garcia, Yusuke Hirota, Yankun Wu, and Yuta Nakashima. Uncurated image-text datasets: Shedding light on demographic bias. In _Computer Vision and Pattern Recognition_ , pages 6957–6966, 2023. 2 

- [24] Sahin Cem Geyik, Stuart Ambler, and Krishnaram Kenthapadi. Fairness-aware ranking in search & recommendation systems with application to linkedin talent search. In _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ , page 2221–2231, 2019. 6 

- [25] Chengcheng Guo, Bo Zhao, and Yasnbing Bai. Deepcore: A comprehensive library for coreset selection in deep learning. In _International Conference on Database and Expert Systems Applications_ , pages 181–195. Springer, 2022. 2 

- [26] Laura Gustafson, Chloe Rolland, Nikhila Ravi, Quentin Duval, Aaron Adcock, Cheng-Yang Fu, Melissa Hall, and Candace Ross. Facet: Fairness in computer vision evaluation benchmark. In _International Conference on Computer Vision_ , pages 20370–20382, 2023. 1, 2, 5, 6, 7, 8 

- [27] Lisa Anne Hendricks, Kaylee Burns, Kate Saenko, Trevor Darrell, and Anna Rohrbach. Women also snowboard: Overcoming bias in captioning models. In _Neural Information Processing Systems_ , 2016. 2 

- [28] Yusuke Hirota, Yuta Nakashima, and Noa Garcia. Quantifying societal bias amplification in image captioning. In _Computer Vision and Pattern Recognition_ , pages 13450–13459, 2022. 2 

- [29] IEEE. Ieee code of ethics, 2020. Available: https:// www.ieee.org/about/corporate/governance/ p7-8.html. 3 

- [30] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. Zendo https://doi.org/10.5281/ zenodo.5143773. 5 

- [31] Sepehr Janghorbani and Gerard De Melo. Multimodal bias: Introducing a framework for stereotypical bias assessment beyond gender and race in vision language models. _European Chapter of the Association for Computational Linguistics_ , 2023. 2 

- [32] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International conference on machine learning_ , pages 4904–4916, 2021. 2, 4 

- [33] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with GPUs. _IEEE Transactions on Big Data_ , 7(3):535–547, 2019. 5 

- [34] Kimmo K¨arkk¨ainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In _Winter Conference on Applications of Computer Vision_ , pages 1548–1558, 2021. 2, 6, 7 

- [35] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded 

   - radiance fields. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19729–19739, 2023. 1 

- [36] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. _arXiv preprint arXiv:2304.02643_ , 2023. 5 

- [37] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. _arXiv preprint arXiv:2304.02643_ , 2023. 1 

- [38] Svetlana Kiritchenko and Saif M. Mohammad. Examining gender and race bias in two hundred sentiment analysis systems. _Conference on Lexical and Computational Semantics_ , 2018. 2 

- [39] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images, 2009. 6 

- [40] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. _Neural Information Processing Systems_ , pages 9694–9705, 2021. 2 

- [41] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , pages 12888– 12900, 2022. 4 

- [42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 1 

- [43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European Conference on Computer Vision_ , pages 740–755, 2014. 1, 6, 4 

- [44] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv preprint arXiv:2310.03744_ , 2023. 1 

- [45] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In _Proceedings of International Conference on Computer Vision (ICCV)_ , 2015. 2 

- [46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2019. 5, 1 

- [47] Ryan Mac. Facebook apologizes after a.i. puts ‘primates’ label on video of black men. _New York Times_ , 2021. 2 

- [48] Adyasha Maharana, Prateek Yadav, and Mohit Bansal. D2 pruning: Message passing for balancing diversity & difficulty in data pruning. _arXiv preprint arXiv:2310.07931_ , 2023. 2, 3 

- [49] Chandler May, Alex Wang, Shikha Bordia, Samuel R Bowman, and Rachel Rudinger. On measuring social biases in sentence encoders. _North American Chapter of the Association for Computational Linguistics_ , 2019. 2 

10 

- [50] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In _International Conference on Machine Learning_ , pages 6950–6960, 2020. 2 

- [51] Ellis Monk. Monk skin tone scale, 2019. 6 

- [52] Safiya Noble. _Algorithms of Oppression: How Search Engines Reinforce Racism_ . New York University Press, Cambridge MA, 2018. 2 

- [53] Council of European Union. Directive Proposal (COM(2008)462), 2008. Available: https://eurlex.europa.eu/legal-content/en/TXT/?uri= CELEX%3A52008PC0426. 3 

- [54] Jeff M. Phillips. Coresets and sketches. _arXiv preprint arXiv:1601.00617_ , 2016. 3 

- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , 2021. 1, 2, 3, 4, 5 

- [56] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In _International Conference on Machine Learning_ , pages 5389–5400, 2019. 6 

- [57] Candace Ross, Boris Katz, and Andrei Barbu. Measuring social biases in grounded vision and language embeddings. In _North American Chapter of the Association for Computational Linguistics_ , pages 998–1008, 2021. 2 

- [58] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. _NeurIPS Workshop on Data Centric AI_ , 2021. 1, 2, 4 

- [59] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. _Neural Information Processing Systems_ , 35:25278–25294, 2022. 1 

- [60] Ashish Seth, Mayur Hemani, and Chirag Agarwal. Dear: Debiasing vision-language models with additive residuals. In _Computer Vision and Pattern Recognition_ , pages 6820– 6829, 2023. 2, 6 

- [61] Nur Muhammad Mahi Shafiullah, Chris Paxton, Lerrel Pinto, Soumith Chintala, and Arthur Szlam. Clip-fields: Weakly supervised semantic fields for robotic memory. _arXiv preprint arXiv:2210.05663_ , 2022. 1 

- [62] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In _Conference on Robot Learning_ , pages 894–906. PMLR, 2022. 1 

- [63] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. In _Neural Information Processing Systems_ , pages 19523–19536, 2022. 2, 3, 4 

- [64] Tejas Srinivasan and Yonatan Bisk. Worst of both worlds: Biases compound in pre-trained vision-and-language models. In _Proceedings of the 4th Workshop on Gender Bias in Natural Language Processing (GeBNLP)_ , pages 77–85, 2022. 2 

- [65] Tony Sun, Andrew Gaut, Shirlyn Tang, Yuxin Huang, Mai ElSherief, Jieyu Zhao, Diba Mirza, Elizabeth Belding, KaiWei Chang, and William Yang Wang. Mitigating gender bias in natural language processing: Literature review. In _Association for Computational Linguistics_ , pages 1630–1640, 2019. 2 

- [66] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. _Communications of the ACM_ , pages 64–73, 2016. 6 

- [67] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Neural Information Processing Systems_ , 30, 2017. 5, 1 

- [68] Angelina Wang, Solon Barocas, Kristen Laird, and Hanna Wallach. Measuring representational harms in image captioning. In _ACM Conference on Fairness, Accountability, and Transparency_ , pages 324–335, 2022. 2 

- [69] Ted Xiao, Harris Chan, Pierre Sermanet, Ayzaan Wahid, Anthony Brohan, Karol Hausman, Sergey Levine, and Jonathan Tompson. Skill acquisition by instruction augmentation on offline datasets. In _NeurIPS 2022 Foundation Models for Decision Making Workshop_ , 2022. 1 

- [70] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. _arXiv preprint arXiv:2309.16671_ , 2023. 8 

- [71] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , pages 67–78, 2014. 1, 5, 6, 4 

- [72] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. _Transactions on Machine Learning Research_ , 2022. 4 

- [73] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andr´e Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, Lucas Beyer, Olivier Bachem, Michael Tschannen, Marcin Michalski, Olivier Bousquet, Sylvain Gelly, and Neil Houlsby. The visual task adaptation benchmark. _arXiv preprint arXiv:1910.04867_ , 2019. 1, 5 

- [74] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In _Computer Vision and Pattern Recognition_ , pages 5579–5588, 2021. 4 

- [75] Yi Zhang, Junyang Wang, and Jitao Sang. Counterfactually measuring and eliminating social bias in vision-language pre-training models. In _ACM International Conference on Multimedia_ , pages 4996–5004, 2022. 2 

11 

- [76] Dora Zhao, Angelina Wang, and Olga Russakovsky. Understanding and evaluating racial biases in image captioning. In _International Conference on Computer Vision_ , pages 14830– 14840, 2021. 2 

- [77] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. In _Conference on Empirical Methods in Natural Language Processing_ , 2017. 2 

- [78] Kankan Zhou, Yibin LAI, and Jing Jiang. Vlstereoset: A study of stereotypical bias in pre-trained vision-language models. In _Association for Computational Linguistics_ , 2022. 2 

- [79] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. _arXiv preprint arXiv:2304.06718_ , 2023. 1 

12 

## **FairDeDup: Detecting and Mitigating Vision-Language Fairness Disparities in Semantic Dataset Deduplication** 

## Supplementary Material 

## **8. Bias Constrained Clusters** 

As described in Sec. 6, clustering may limit the availability of lower represented samples for biasing sensitive concept representation. We show several demonstrative clusters in Fig. 5 alongside descriptions of possible limitations. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0013-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0013-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0013-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0013-07.png)


## **9. Hyperparameters** 

|Parameter|Value|
|---|---|
|Model|CLIP [55]|
|Image Encoder|Vision Transformer Base/16 [18]|
|Text Encoder|Text Transformer [67]|
|Epochs|16|
|Batch Size|33,820|
|Learning Rate|5_×_10_−_4|
|LR Warmup|Linear / 2,000 Batches|
|LR Schedule|Cosine Annealing|
|Optimizer|AdamW [46]|
|_Decay_|0.2|
|_β_1|0.90|
|_β_2|0.98|
|_ϵ_|1_×_10_−_6|
|Precision|AMP BFloat16|




![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0013-10.png)


**----- Start of picture text -----**<br>
(a) Stock close-ups bifurcated across gender presentation and age.<br>**----- End of picture text -----**<br>


Table 5. Hyperparameters used to train all models. 

Figure 5. Sample Clusters where selecting for certain underrepresented concepts may be difficult due to them being split into an entirely different cluster. 

## **10. Choosing Sensitive Concepts** 

We derive sensitive concepts based on commonly protected groups in law. For example, Title VII of the US Civil Rights Act prohibits employment discrimination based on groups like _race_ and _religion_ . We take the intersection of commonly protected groups and those which are annotated in VL fairness datasets ( _e.g_ ., only _race_ from above) to inform our concept design. In this way, we are able to consider concepts relevant to real-world practice while retaining our ability to evaluate the effectiveness of our mitigation strategy in a lab setting. To represent intersectional identities, we specifically choose sensitive concepts which simultaneously capture many of these protected groups. 

1 

## **11. Deduplicated Cluster Examples** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0014-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0014-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/deduplication/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication_images/cvpr_2024_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_Deduplication.pdf-0014-03.png)


**----- Start of picture text -----**<br>
(a) Chef<br>(b) Doctor<br>(c) Firefighter<br>**----- End of picture text -----**<br>


Figure 6. Additional samples from person-related clusters after applying **SemDeDup (left)** and **FairDeDup (right)** . Samples are randomly selected from clusters manually identified to include people. We annotate each cluster with our identification of its semantic contents. 

2 

## **12. Sensitive Concept Prototypes** 

|1 person|56 young white woman|
|---|---|
|2 woman|57 young white man|
|3 man|58 young indian person|
|4 black person|59 young indian woman|
|5 black woman|60 young indian man|
|6 black man|61 young latino person|
|7 white person|62 young latino woman|
|8 white woman|63 young latino man|
|9 white man|64 young east asian person|
|10 indian person|65 young east asian woman|
|11 indian woman|66 young east asian man|
|12 indian man|67 young middle eastern person|
|13 latino person|68 young middle eastern woman|
|14 latino woman|69 young middle eastern man|
|15 latino man|70 young southeast asian person|
|16 east asian person|71 young southeast asian woman|
|17 east asian woman|72 young southeast asian man|
|18 east asian man|73 child|
|19 middle eastern person|74 black child|
|20 middle eastern woman|75 white child|
|21 middle eastern man|76 indian child|
|22 southeast asian person|77 latino child|
|23 southeast asian woman|78 east asian child|
|24 southeast asian man|79 middle eastern child|
|25 old person|80 southeast asian child|
|26 old woman|81 baby|
|27 old man|82 black baby|
|28 old black person|83 white baby|
|29 old black woman|84 indian baby|
|30 old black man|85 latino baby|
|31 old white person|86 east asian baby|
|32 old white woman|87 middle eastern baby|
|33 old white man|88 southeast asian baby|
|34 old indian person|89 boy|
|35 old indian woman|90 girl|
|36 old indian man|91 black boy|
|37 old latino person|92 black girl|
|38 old latino woman|93 white boy|
|39 old latino man|94 white girl|
|40 old east asian person|95 indian boy|
|41 old east asian woman|96 indian girl|
|42 old east asian man|97 latino boy|
|43 old middle eastern person|98 latino girl|
|44 old middle eastern woman|99 east asian boy|
|45 old middle eastern man|100 east asian girl|
|46 old southeast asian person|101 middle eastern boy|
|47 old southeast asian woman|102 middle eastern girl|
|48 old southeast asian man|103 southeast asian boy|
|49 young person|104 southeast asian girl|
|50 young woman|105 person with dark skin|
|51 young man|106 person with light skin|
|52 young black person|107 old person with dark skin|
|53 young black woman|108 old person with light skin|
|54 young black man|109 young person with dark skin|
|55 young white person|110 young person with light skin|
||(a) Sensitive Concepts|



1 A photo of a {concept} 2 This is a photo of a {concept} 3 A {concept} 

(b) Text Templates 

Figure 7. Sensitive concepts and templates used to generate text concept prototypes for FairDeDup in Sec. 3. 

3 

## **13. Additional Datasets and Metrics** 

|||**Full Data**|**SemDeDup**|**FairDeDup**|
|---|---|---|---|---|
|||(100%)|(50%)|(50%)|
||Acc@1|.2947|.3096|.2988|
|**-a**|Acc@5|.6247|.6359|.6271|
||MPCR|.3163|.3228|.3126|
||Acc@1|.5060|.5365|.5360|
|**-o**|Acc@5|.8235|.8420|.8385|
||MPCR|.5213|.5501|.5443|
||Acc@1|.7646|.7652|.7534|
|**-r**|Acc@5|.9249|.9245|.9223|
||MPCR|.7512|.7506|.7386|
||Acc@1|.6640|.6579|.6535|
|**1k**|Acc@5|.8993|.8974|.8970|
||MPCR|.6639|.6578|.6536|
|**sketch**|Acc@1<br>Acc@5<br>MPCR|.5142<br>.7803<br>.5144|.5036<br>.7774<br>.5040|.4989<br>.7724<br>.4993|
||Acc@1|.5821|.5776|.5747|
|**v2**|Acc@5|.8453|.8413|.8368|
||MPCR|.5822|.5774|.5747|



Table 6. Zero-shot classification performance on ImageNet [16] variants from CLIP Benchmark [12]. Mean Per Class Recall abbreviated as MPCR. Higher is better for all metrics. 

|||**Full Data**|**SemDeDup**|**FairDeDup**|
|---|---|---|---|---|
|||(100%)|(50%)|(50%)|
|**fickr30k**|Image R@5<br>Text R@5|.8728<br>.9640|.8736<br>.9670|.8714<br>.9620|
|**fickr8k**|Image R@5<br>Text R@5|.8570<br>.9450|.8592<br>.9400|.8574<br>.9300|
|**coco**|Image R@5<br>Text R@5|.6325<br>.7852|.6318<br>.7882|.6255<br>.7810|



Table 7. Image-text (Image R@5) and text-image (Text R@5) retrieval Recall@5 on Flickr [71] and COCO [43] from CLIP Benchmark [12]. Higher is better for all metrics. 

||||**Full Data**|**SemDeDup**|**FairDeDup**|
|---|---|---|---|---|---|
||||(100%)|(50%)|(50%)|
|||Acc@1|.8526|.8346|.8429|
|**cars**||Acc@5|.9923|.9909|.9922|
|||MPCR|.8541|.8344|.8429|
|**country**|211|Acc@1<br>Acc@5<br>MPCR|.1791<br>.3978<br>.1789|.1742<br>.3950<br>.1740|.1685<br>.3889<br>.1683|
|||Acc@1|.3976|.3969|.4809|
|**fer**|2013|Acc@5|.9358|.9115|.9413|
|||MPCR|.4057|.3702|.4152|
|||Acc@1|.1563|.1497|.1665|
|**fgvc**|aircraft|Acc@5|.4164|.4146|.4212|
|||MPCR|.1545|.1495|.1667|
|||Acc@1|.4159|.4101|.3383|
|**gtsrb**||Acc@5|.7352|.6814|.7062|
|||MPCR|.3884|.3806|.3571|
|**mnist**|mnist|Acc@1<br>Acc@5<br>MPCR|.3896<br>.8064<br>.3938|.5474<br>.8022<br>.5531|.5890<br>.9058<br>.5911|
|**objectnet**||Acc@1<br>Acc@5<br>MPCR|.4722<br>.7296<br>.4614|.4754<br>.7283<br>.4681|.4781<br>.7311<br>.4665|
|**render**|sst2|Acc@1<br>MPCR|.5371<br>.5368|.5157<br>.5155|.5041<br>.5036|
|||Acc@1|.9659|.9705|.9686|
|**stl10**||Acc@5|.9996|.9998|.9995|
|||MPCR|.9659|.9708|.9684|
|**sun397**||Acc@1<br>Acc@5<br>MPCR|.6856<br>.9344<br>.6730|.6786<br>.9351<br>.6613|.6838<br>.9363<br>.6662|
|**voc2007**||Acc@1<br>Acc@5<br>MPCR|.7432<br>.9498<br>.8197|.7617<br>.9518<br>.8297|.7584<br>.9515<br>.8286|



Table 8. Additional zero-shot classification results from CLIP Benchmark [12]. Mean Per Class Recall abbreivated as MPCR. Higher is better for all metrics. 

4 

|**Full Data**<br>(100%)<br>**SemDeDup**<br>(50%)<br>**FairDeDup**<br>(50%)<br>**caltech**<br>101<br>Acc@1<br>.8304<br>.8230<br>.8309<br>Acc@5<br>.9399<br>.9394<br>.9578<br>MPCR<br>.9193<br>.9035<br>.9061<br>**cifar**<br>10<br>Acc@1<br>.9198<br>.9255<br>.9203<br>Acc@5<br>.9986<br>.9984<br>.9992<br>MPCR<br>.9198<br>.9254<br>.9204<br>**cifar**<br>100<br>Acc@1<br>.7234<br>.7291<br>.7299<br>Acc@5<br>.9342<br>.9342<br>.9386<br>MPCR<br>.7231<br>.7289<br>.7303<br>**clevr**<br>obj dist<br>Acc@1<br>.2033<br>.2101<br>.2232<br>Acc@5<br>.9187<br>.9187<br>.9187<br>MPCR<br>.1686<br>.1633<br>.1673<br>**clevr**<br>count all<br>Acc@1<br>.1421<br>.2317<br>.1916<br>Acc@5<br>.6474<br>.8174<br>.6397<br>MPCR<br>.1397<br>.2264<br>.1873<br>**diabetic**<br>Acc@1<br>.0646<br>.3209<br>.0666<br>Acc@5<br>1.0000<br>1.0000<br>1.0000<br>MPCR<br>.2178<br>.1998<br>.2029<br>**dmlab**<br>Acc@1<br>.1949<br>.1692<br>.1960<br>Acc@5<br>.8430<br>.8236<br>.8270<br>MPCR<br>.1677<br>.1869<br>.1569<br>**dsprites**<br>label orient.<br>Acc@1<br>.0226<br>.0272<br>.0244<br>Acc@5<br>.1259<br>.1264<br>.1302<br>MPCR<br>.0231<br>.0276<br>.0249<br>**dsprites**<br>label xpos<br>Acc@1<br>.0305<br>.0315<br>.0305<br>Acc@5<br>.1600<br>.1568<br>.1625<br>MPCR<br>.0313<br>.0321<br>.0312<br>**dsprites**<br>label ypos<br>Acc@1<br>.0315<br>.0317<br>.0317<br>Acc@5<br>.1553<br>.1559<br>.1596<br>MPCR<br>.0311<br>.0312<br>.0312|**Full Data**<br>(100%)<br>**SemDeDup**<br>(50%)<br>**FairDeDup**<br>(50%)|
|---|---|
||**dtd**<br>Acc@1<br>.5330<br>.5021<br>.5074<br>Acc@5<br>.8293<br>.7883<br>.8202<br>MPCR<br>.5330<br>.5011<br>.5080|
||**eurosat**<br>Acc@1<br>.4904<br>.5220<br>.5246<br>Acc@5<br>.9335<br>.9289<br>.8987<br>MPCR<br>.5117<br>.5288<br>.5404|
||**fowers**<br>Acc@1<br>.6723<br>.6536<br>.6878<br>Acc@5<br>.8533<br>.8442<br>.8374<br>MPCR<br>.6657<br>.6384<br>.6509|
||**kitti**<br>dist.<br>Acc@1<br>.1589<br>.1505<br>.2363<br>MPCR<br>.2134<br>.1707<br>.2134|
||**pcam**<br>Acc@1<br>.5521<br>.4720<br>.5910<br>MPCR<br>.5521<br>.4719<br>.5909|
||**pets**<br>Acc@1<br>.8929<br>.8577<br>.8812<br>Acc@5<br>.9951<br>.9940<br>.9937<br>MPCR<br>.8917<br>.8570<br>.8802|
||**resisc**<br>45<br>Acc@1<br>.5694<br>.5892<br>.5763<br>Acc@5<br>.8889<br>.8938<br>.8914<br>MPCR<br>.5751<br>.5955<br>.5853|
||**smallnorb**<br>label azimuth<br>Acc@1<br>.0539<br>.0502<br>.0550<br>Acc@5<br>.2796<br>.2781<br>.2738<br>MPCR<br>.0547<br>.0509<br>.0560|
||**smallnorb**<br>label elevation<br>Acc@1<br>.0927<br>.1125<br>.1114<br>Acc@5<br>.5319<br>.5770<br>.5544<br>MPCR<br>.0922<br>.1116<br>.1106|
||**svhn**<br>Acc@1<br>.3973<br>.3657<br>.4008<br>Acc@5<br>.7782<br>.7694<br>.8107<br>MPCR<br>.3727<br>.3670<br>.3238|
||Table 10. Additional results for VTAB [73] extending Tab.9.|



Table 9. Zero-shot classification performance on Visual Task Adaptation Benchmark (VTAB) [73] datasets from CLIP Benchmark [12]. Mean Per Class Recall abbreviated as MPCR. Higher is better for all metrics. 

5 

## **14. Extended Pseudo-Code** 

- 1 def semdedup(embs, eps): 

2 # Sort by distance to the cluster centroid. 3 sort_by_dist_to_centroid(embs, desc=True) 4 5 # Compute the pairwise cosine similarity 6 pair_sims = embs @ embs.T 7 triu_sims = torch.triu(pair_sims, diagonal=1) 8 M = torch.max(triu_sim_matrix, dim=0)[0] 9 10 # Keep if the max similarity <= threshold 11 points = M <= 1 - epsilon 12 log_and_keep(points) 13 14 def fairdedup(embs, prototypes, eps): 15 # Get similarity with concept prototypes 16 proto = embs @ prototypes.T 17 18 balance = AverageMeter(prototype.shape[0]) 19 tovisit = torch.ones(embs.shape[0]) 20 while tovist.any(): 21 # Find an unvisited neighborhood 22 node = torch.where(tovisit)[0][0] 23 sims = embs[node] @ embs.T 24 neighbors = torch.where(sims > 1 - eps) 25 neighbors = neighbors[0] 26 27 # Maximize least represented concept 28 c = balance.get_min_concept() 29 point = proto[neigbors][:, c].argmax() 30 balance.update(point) 31 32 log_and_keep(point) 33 tovisit[neighbors] = 0 34 

- 35 # Input: embedding_model, dataset, eps, prototypes 

- 36 # Embed and cluster the dataset 

- 37 embeddings = embedding_model(dataset) 

- 38 per_cluster_embeddings = kmeans(embeddings) 

- 39 for cluster in per_cluster_embeddings: 

40 # Choose selection method 41 semdedup(cluster) 42 fairdedup(cluster) 

Figure 8. Extended PyTorch-style pseudo-code for SemDeDup and FairDeDup selection given concept prototypes, an embedding ~~m~~ odel, target dataset to deduplicate, and an eps similarity threshold for determining duplicates. SemDeDup pseudo-code modified from [1]. 

6 

