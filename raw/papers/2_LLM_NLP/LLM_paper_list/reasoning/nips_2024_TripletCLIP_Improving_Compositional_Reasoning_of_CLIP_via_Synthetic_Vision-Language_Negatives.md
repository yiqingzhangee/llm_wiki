# **TripletCLIP: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives** 

**Maitreya Patel** _[♢∗]_ **Abhiram Kusumba** _[♢†]_ **Sheng Cheng** _[♢†]_ **Changhoon Kim** _[♢]_ **Tejas Gokhale** _[♡]_ **Chitta Baral** _[♢]_ **Yezhou Yang** _[♢]_ 

> _♢_ Arizona State University _♡_ University of Maryland, Baltimore County tripletclip.github.io 

## **Abstract** 

Contrastive Language-Image Pretraining (CLIP) models maximize the mutual information between textual and visual modalities to learn representations. However, the lack of compositional diversity in contemporary image-text datasets limits the compositional reasoning ability of CLIP. We show that generating “hard” negative captions via in-context learning and synthesizing corresponding negative images with text-to-image generators offers a solution. We introduce a novel contrastive pre-training strategy that leverages these hard negative captions and images in an alternating fashion to train CLIP. We demonstrate that our method, named `TripletCLIP` , when applied to existing datasets such as CC3M and CC12M, enhances the compositional capabilities of CLIP, resulting in an absolute improvement of over 9% on the SugarCrepe benchmark on an equal computational budget, as well as improvements in zero-shot image classification and image retrieval. Our code, models, and data are available at: tripletclip.github.io. 

## **1 Introduction** 

Large-scale vision-language models, such as CLIP [41], have significantly advanced multi-modal learning by employing contrastive learning to acquire shared semantic representations from paired datasets. This approach has resulted in improved performance in vision-language tasks as well as zero-shot image classification [51] and segmentation [23, 62]. Beyond vision-language tasks, the individual components of these models, such as the vision encoder and the language encoder, are integral to several multimodal architectures and generative models such as multimodal large language models (MLLMs) [30, 26] and text-to-image (T2I) diffusion models [45, 37, 38]. Yet, compositional reasoning remains challenging and multimodal models continue to exhibit naïve “bag of words” behavior, frequently failing to distinguish between expressions like “bulb in the grass” and “grass in the bulb” [58, 53]. Addressing this challenge remains critical for enhancing vision-language models and their downstream applications. 

Contrastive learning of representations benefits from “hard negative samples” (i.e., points that are difficult to distinguish from an anchor point) [43]. However, at each optimization step for training CLIP, image-text pairs are _randomly_ sampled from the training dataset – this random sampling seldom exposes the model to highly similar negative pairs. We hypothesize that the limited compositional understanding of CLIP may stem from such issues in the optimization objective and sampling from training datasets. A straightforward solution could involve iteratively identifying hard negative pairs for each training iteration. However, due to the noisy captions and the scarcity of such pairs in existing datasets, prior work generates hard negative captions as a form of augmentation using rule-based strategies [58, 61]. For instance, given an image-text pair labeled “a brown horse”, an additional 

> _∗_ Corresponding author: maitreya.patel@asu.edu. _†_ indicates the equal contribution. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

negative caption “a blue horse” might be introduced. However in prior work, image data is not subjected to similar hard negative semantic augmentation during training; this is mainly because of the difficulty of making semantic perturbations at the pixel levels compared to sentence perturbation. While the text-only augmentation strategies have improved the models’ compositional understanding to a certain extent, it raises an intriguing question: _**could incorporating hard negative augmentation for both text and image modality further enhance the compositional reasoning capabilities of vision-language models?**_ 

Motivated by this, in this paper, we introduce a novel, simple, and yet highly effective strategy for integrating hard negative images as well as hard negative text to enhance the compositional understanding of vision-language models. Recent developments in text-to-image diffusion models have opened up possibilities for performing semantic perturbations within images [21]. Existing works have evaluated the impact of creating synthetic data for text-to-image generative models [3, 6]. However, it remains less explored how these generative models can benefit the CLIP-like models. To tackle this challenge, our approach leverages the in-context learning capabilities of LLMs to produce realistic, linguistically accurate negative captions [55]. We then employ a pre-trained textto-image diffusion model to create images corresponding to these captions, thereby enriching any given image-text dataset with valuable hard negatives that foster improved reasoning. This resulting `TripletData` comprises 13M image-text pairs to complement the CC3M and CC12M datasets [5]. 

We developed `TripletCLIP` , which incorporates hard negative image-text pairs effectively by using them to optimize a novel triplet contrastive loss function. Extensive experiments on the CC3M and CC12M datasets and various downstream tasks with an equal compute budget demonstrate that `TripletCLIP` significantly enhances compositional reasoning. Notably, `TripletCLIP` results in more than 9% and 6% absolute improvement on the SugarCrepe benchmark compared to LaCLIP [25] and NegCLIP [58], respectively. `TripletCLIP` also improves zero-shot classification and image-text retrieval performance with similar training-time concept diversity. An investigation into the effects of increasing training-time concept diversity revealed that baseline models consistently under-performed in compositional tasks despite an increase in integrated knowledge, while `TripletCLIP` demonstrated significant improvements. In summary, our **key contributions** are as follows: 

- We introduce a novel CLIP pre-training strategy that employs hard negative text and images in conjunction with triplet contrastive learning to enhance compositionality. 

- `TripletCLIP` consistently improves across downstream tasks, demonstrating the effectiveness of synthesizing hard negative image-text pairs. 

- Our extensive ablations on the choice of the loss function, modality-specific pre-training, the increase in concept diversity, and filtering high-quality `TripletData` provide deeper insights into the utility of hard negative image-text pairs for CLIP pre-training. 

- Ultimately, we present a promising avenue where synthetic contrastive datasets significantly improve reasoning capabilities, leading to the creation and release of the `TripletData` — a 13M contrastive image-text dataset. 

## **2 Related Work** 

**Vision-Language Models.** Recent advancements, including ALIGN [22] and CLIP [41], have gained significant interest due to their capability to learn transferable semantic representations across multiple modalities through contrastive learning. These models facilitate downstream tasks such as zero-shot classification [51], image-text retrieval [60, 2], visual grounding/reasoning [31], textto-image generation [52, 37, 38, 24], semantic segmentation [62, 23], and various evaluations [19, 47]. Subsequent research has sought to enhance various aspects of these models, including data efficiency [15], hierarchical representation learning [8], and the quantization of latent spaces for more stable pre-training [7]. LiT [59] employs a pre-trained frozen CLIP vision encoder to fine-tune a BERT-like text encoder [9], achieving notable improvements in zero-shot transfer performance. Similarly, BLIP-2 [27] combines contrastive pre-training with the next-token prediction for image captioning during training. However, these approaches generally presume the availability of highquality data. In contrast, `TripletCLIP` focuses on leveraging the proposed hard negative contrastive dataset and incorporating triplet contrastive pre-training for compositional data. This approach is orthogonal to prior works. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0003-00.png)


**----- Start of picture text -----**<br>
𝐿'()#$%& (Eq. 2) 𝐿*#$ (Eq. 4)<br>𝐿#$%& (Eq. 1) 𝐿'()#$%& (Eq. 2)<br>𝐿'()#$%& (Eq. 2)<br>Vision (𝐹!) Text (𝐹") Vision (𝐹!) Text (𝐹")<br>Vision (𝐹!) Text (𝐹") Vision (𝐹!)<br>𝑥 𝑦 𝑥 𝑦 𝑦′<br>𝑥 𝑦 𝑦′ 𝑥′<br>(a) CLIP (b) NegCLIP (c) TripletCLIP<br>**----- End of picture text -----**<br>


Figure 1: Comparison of training workflows of CLIP, NegCLIP, and `TripletCLIP` . ( _x, y_ ) represents the positive a image-text pair, and ( _x[′] , y[′]_ ) represents the corresponding negative image-text pair. 

**Data for Contrastive Pre-training.** The effectiveness of maximizing mutual information between modalities heavily relies on the quality of extensive, web-scraped datasets that ideally encompass all possible concepts and knowledge. For instance, despite its noise, the LAION dataset [48, 16], which includes more than 5 billion internet images paired with alt-text captions, is a primary resource. Studies show that over 1 billion data points are necessary to match the performance of the original CLIP model [16, 56]. Recent works like DataComp [16] and MetaCLIP [56] have focused on creating smaller, high-quality datasets by applying stringent filters and ensuring wordnet [33] synset-level concept diversity. Nevertheless, the inherently noisy nature of internet-scraped datasets can degrade model performance. Studies such as SynthCLIP [18] demonstrate that tripling the volume of fully synthetic data is required to equal the efficacy of real data. Other efforts like VeCLIP [25] and LaCLIP [14] enhance dataset quality by using generative language models to re-caption existing images, significantly boosting performance. 

**Compositionality for vision-language.** Despite the increased emphasis on data quality and modeling techniques, mastering compositionality remains a significant challenge for vision-language models. Benchmarks like ARO [58], VALSE [35], and CREPE [32] have been developed to assess models’ abilities to handle compositional data. SugarCrepe [20], in particular, offers a large-scale, systematic framework for such evaluations. Previous methods primarily focused on identifying hard negatives within existing datasets or generating synthetic negative captions [58, 61, 12, 11, 57, 49]. However, these rule-based generated captions are often unrealistic and linguistically flawed, leading to suboptimal model performance on complex datasets like SugarCrepe. A handful of works focus on finding negative images. [54] propose utilizing the video data. [44] focuses on object-centric image-editing to synthesize the negative images. [39] utilizes the simulation-based data negative data. 

Contrary to prior approaches that predominantly add unrealistic negative captions or very constrained negative images that are either very synthetic or object-focused, this work introduces `TripletCLIP` , which centers on generating naturally occurring _**hard negative image-text pairs**_ . We propose a novel triplet contrastive learning strategy that effectively utilizes these challenging data pairs. Additionally, while our method is distinct, integrating advancements that refine contrastive learning could potentially boost `TripletCLIP` ’s efficacy further. 

## **3 Method** 

This section begins with an overview of the contrastive learning algorithm used by CLIP and NegCLIP. We then describe the synthetic data generation pipeline for generating hard negatives using LLMs and T2I models and introduce triplet contrastive learning which forms the basis of `TripletCLIP` . A high-level comparison between prior work and `TripletCLIP` can be found in Figure 1. 

## **3.1 Preliminaries** 

The goal for self-supervised contrastive learning [13], when dealing with inputs from a single modality, is to use a feature extractor ( _F_ ) to encode inputs and their augmentations and minimize the InfoNCE loss [34] between the two encodings. CLIP is designed for multimodal settings (for example, vision and language inputs) – this entails using two encoders (one for each modality). 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0004-00.png)


**----- Start of picture text -----**<br>
“Heart trimmed  “Heart trimmed in  “A cartoon  “A cartoon  “man breaks free  “man gets<br>in white on top  red on bottom of  illustration of a  illustration of a  from a tackle  tackled during<br>of black frame.” silver frame.” stressed man taking cover  hiding from rainpeaceful man .” football game.during the  ” the basketball game.”<br>from gunfire.”<br>“A patterned  “A solid-colored  “small peony in  “large peony in  “A photo of a  “A photo of a<br>color schemewith a vintage background .” background with a futuristic color scheme.” the garden.” the greenhouse.” a stack of pins.map with pins or ” map with marbles or a stack of marbles.”<br>**----- End of picture text -----**<br>


Figure 2: Examples image-text pairs from `TripletData` . In each block, a positive pair from CC3M is on the left and corresponding negatives from `TripletData` are shown on the right. 

Let _X_ and _Y_ represent two modalities and _D_ = _{_ ( _xi, yi_ ) _}i_ =1 _,...,M_ , be the training dataset, where _xi ∈X_ and _yi ∈Y_ . The goal is to train two modality-specific encoders, _FX_ and _FY_ , by minimizing the InfoNCE loss between the normalized features extracted from the encoders. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0004-03.png)


where _⟨·⟩_ represents cosine similarity and _τ_ is the trainable temperature parameter. For simplicity, we do not show feature normalization in the InfoNCE loss. Similarly, we can define the _L[CL] Y→X_ training loss. The combined CLIP total training objective is given as, _LCLIP_ = _L[CL] X→Y_[+] _[ L][CL] Y→X_[.] By minimizing this training loss, both encoders learn representations that maximize the mutual information between two modalities. 

NegCLIP introduces synthetic augmentations to generate “hard” negative captions ( _yi[′][∈Y][′]_[)][by] performing semantic inverting perturbations to the reference captions ( _yi ∈Y_ ). Therefore, the single modality-specific hard negative augmentation-based training loss can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0004-06.png)


The total loss for NegCLIP for image modality ( _X_ ) and text modality ( _Y_ ) is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0004-08.png)


In Eq. 2, the negative samples are generated only for language modality as it is easy to make semanticlevel perturbations. Existing methods have not explored performing semantic perturbations in the image modality to create hard negatives. In this work, we demonstrate how hard negatives can be created in the image modality by leveraging the semantic language grounding and photorealism of text-to-image diffusion models. Our novel hard negative generation pipeline and refined training objective seeks to bridge the significant gap identified in literature. 

## **3.2** `TripletData` **: Image-text hard negative data augmentations** 

To generate high-quality hard negative image-text pairs, we follow a two-step procedure. The first stage is to generate hard negative captions from the ground truth positive caption. Second, to generate 

4 

images corresponding to the hard negative captions as negative images. The AltText captions from the existing web-scrapped datasets are very noisy, leading to the noisy and unreliable generation of hard negatives. Therefore, we build upon the existing work LaCLIP, which first rewrites the captions using LLM from the existing data that are linguistically accurate. Figure 2 illustrates several examples of positive and corresponding negative image-text pairs. 

**Generating hard negative captions.** Existing works perform random swapping, replacing, and adding actions between the nouns, attributes, and relations of the positive caption [58, 61]. This method results in nonsensical and grammatically incorrect artifacts, such as “a person riding on four slope,” which impedes the generation of negative images, ultimately leading to diminishing performance on harder benchmarks [20]. Therefore, we utilize the in-context learning ability of LLMs to generate negative captions. The choice of LLM is a trivial task as long as they provide hard negative captions. We find that Mistral-7B-Instruct-v0.2[2] performs reasonably better on our goal, and the output is easy to parse. We generate the negative captions in batches to speed up the generation process. Generating the 13M negative captions takes only 3 days on 8xRTX A6000. Instead of generating multiple hard negative captions, we find that a single high-quality hard negative caption is enough to improve the performance compared to the traditional NegCLIP style caption generation (see NegCLIP++ results in Table 4). We provide examples of various types of negative captions in the appendix. Specifically, we provide the following prompt to LLM: 

You are given the description of an image. You should provide a mostly similar description, changing the original one slightly but introducing enough significant differences such that the two descriptions could not possibly be for the same image. Keep the description length the same. Finally, only a few things (such as counting, objects, attributes, and relationships) can be modified to change the image structure significantly. Provide just the updated description. Examples: Input: A dog to the left of the cat. Output: A dog to the right of the cat. Input: A person wearing a red helmet drives a motorbike on a dirt road. Output: A person in a blue helmet rides a motorbike on a gravel path. Now, do the same for the following captions: Input: {} Output: 

**Generating hard negative images.** Typically, semantic perturbations within images require tools like image editing, which are resource-intensive and cannot be scaled. Remember that we want to provide additional ground truth references for the negative caption. Therefore, we propose to utilize the negative captions from the previous stage to generate the respective reference images directly for pre-training. As the previous stage generates negative captions that are linguistically correct, it becomes easier for image-generative models to synthesize the respective images precisely. We utilize pre-trained text-to-image diffusion models to generate the corresponding images. Specifically, we select SDXL-turbo [46] due to its relatively faster generation speed. After applying various inference time optimizations, we can generate 13M negative images within 2 days using 30 v100 GPUs. We provide various examples of the hard negative image-text pairs in the appendix. 

**Analyzing difficulty of the hard** `TripletData` **.** Let’s assume we have positive and negative image-text pairs from the `TripletData` , ( _xi, yi_ ) and ( _x[′] i[, y] i[′]_[)][, respectively.][If the data is] truly hard negative, existing pre-trained models should struggle to find the correct imagetext pairs (i.e., _cos_ ( _xi, yi_ ) _> cos_ ( _xi, yi[′]_[)][).][Fol-] lowing winoground, we measure the text-score, image-score, and group-score to evaluate the popular pretrained CLIP models. Table 1 shows that even CLIP models trained on billions of data struggle to get near human performance on `TripletData` , which is less difficult than winoground. Importantly, the goal of generating hard negative samples isn’t to add more diversity 

Table 1: Winoground-style evaluation of pretrained CLIP models on `TripletData` . 

|ViT-B/32<br>ViT-L/14<br>ViT-bigG||**Img Score **<br>40.29<br>44.84<br>42.94|**Text Score **<br>68.17<br>69.21<br>77.61|**Grp Score**<br>36.53<br>40.91<br>40.98|
|---|---|---|---|---|
|Siglip-so400m||44.24|71.27|26.10|
|**Humans (on Winoground)**<br>**88.50**|||**89.50**|**85.50**|
|Table 2: Wordnet synset analysis of captions from<br>CC3M and`TripletData`.|||||
||**CC3M**|**TripletData**<br>**(Negative Only)**|**TripletData **|**Intersection**|
|# unique|59094|59616|62741|**55969**|
|# total synsets|231M|215M|446M|-|



Table 2: Wordnet synset analysis of captions from CC3M and `TripletData` . 

in terms of unique concepts during the training but to add diversity in semantic meanings. Therefore, we measure the unique wordnet synsets in CC3M _vs._ `TripletData` . From Table 2, it can be observed that `TripletData` does not add any new concepts but uses existing concepts to provide negative 

> 2 `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2` 

5 

Table 3: **Importance of image-text hard negatives.** We measure the importance of various modalityspecific hard negatives on SugarCrepe, image-text retrieval, and ImageNet1k. We find that `TripletCLIP` results into the most optimal solution. **Bold** number indicates the best performance. 

|**Models**|**Negative Captions**|**Negative Images**|**SugarCrepe**|**Retrieval**|**ImageNet1k**|
|---|---|---|---|---|---|
|**LaCLIP**|_×_|_×_|54.09|8.19|3.79|
|**NegImage**|_×_|✓|56.28|9.20|4.48|
|**NegCLIP++**|✓|_×_|61.69|8.36|3.84|
|**TripletCLIP**|✓|✓|**63.49**|**16.42**|**7.31**|



samples that are semantically different. To summarize, `TripletData` contains the relatively hard negative image-text pairs that current models find difficult to differentiate. 

## **3.3 TripletCLIP** 

Prior works have demonstrated the value of hard negative captions for enhancing the compositionality of CLIP models via _LNegCLIP_ as the key training objective (Eq. 2) [58, 61]. However, it remains elusive if negative images alone can benefit or not. We conduct modality-specific ablations, reporting the average performance across the diverse set of benchmarks in Table 3 (we provide more details about experiments in Section 4). Our findings indicate that both “hard” negative captions and images individually boost performance when compared to LaCLIP. However, this initial empirical experiments to train the CLIP model on hard negative images (i.e., NegImage) by minimizing _LNegCLIP_ ( _Y, X , X[′]_ ) reveal that negative images alone cannot improve the compositionality significantly (see Table 3). We hypothesize that images contain low-level information, making it difficult to train the model using images as negative examples. Aligning with our initial motivation and building upon this crucial insight, we propose to utilize the negative images to regularize the effect of negative captions and to stabilize the pre-training. Therefore, to utilize these hard negative image-text pairs from the previous stage more effectively, we propose to focus on two triplets ( _X , Y, Y[′]_ ) and ( _X[′] , Y[′] , Y_ ), hence, the final triplet contrastive learning training objective is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0006-05.png)


Intuitively, the second term introduces the additional form of supervision that hard negative images are closer to the corresponding negative captions than positive captions. This allows the system to understand that if the positive image does not represent the negative caption “blue horse,” then what does this caption entail? Through this strategic alternation of hard negative image-text pairs for the `TripletCLIP` , we improve compositionality and image-text understanding of the vision-language model (see Table 3). We provide the pseudo-code in the appendix and the code in supplementary materials. This simple yet effective strategy elevates the training of the CLIP, offering a scalable framework to improve overall performance. 

## **4 Experiments & Results** 

## **4.1 Experiment Setup** 

**Pretraining Datasets.** We utilize the CC3M and CC12M datasets, which comprise 2.6M and 8.6M image-text pairs, respectively. Following the approach demonstrated by LaCLIP, we use LLMrewritten captions to replace noisy original captions. For NegCLIP, we introduce four negative captions per positive image-text pair, focusing on semantic inverting perturbations across four categories: attribute, relation, object, and action [61]. This generates approximately 10.4M and 34.4M text-only augmentations for CC3M and CC12M, respectively. To train the `TripletCLIP` , we create augmentations ( `TripletData` ) for both datasets to integrate hard negatives effectively. We produce one augmentation per image-text pair, adding 2.6M and 8.6M image-text augmented pairs for CC3M and CC12M, respectively. Finally, we perform all the ablations on the CC3M dataset. 

**Baselines.** We train LaCLIP, LaCLIP with real hard negatives (LaCLIP+HN), and NegCLIP from scratch to ensure consistency and fairness in our comparisons. As NegCLIP’s rule-based augmentations closely resemble some compositional benchmarks, so we introduce NegCLIP++ as an improved 

6 

Table 4: **Composition evaluations of the methods on SugarCrepe benchmark. Bold** number indicates the best performance and underlined number denotes the second-best performance. _†_ represents the results taken from SugarCrepe benchmark. 

|**Methods**|**Replace**<br>**Object**<br>**Attribute**<br>**Relation**|**Replace**<br>**Object**<br>**Attribute**<br>**Relation**|**Swap**<br>**Object**<br>**Attribute**|**Swap**<br>**Object**<br>**Attribute**|**Add**<br>**Overall**<br>**Object**<br>**Attribute**<br>**Avg.**|
|---|---|---|---|---|---|
|**CC3M**<br>**LaCLIP**<br>**LaCLIP + HN**<br>**NegCLIP**<br>**NegCLIP++**_(ours)_<br>**TripletCLIP** _(ours)_|59.44<br>53.17<br>51.42<br>63.44<br>55.96<br>50.71<br>62.71<br>58.12<br>54.48<br>64.77<br>66.12<br>**65.93**<br>**69.92**<br>**69.03**<br>64.72|||54.69<br>49.25<br>50.60<br>48.57<br>**56.33**<br>51.20<br>55.51<br>55.41<br>**56.33**<br>**57.96**|55.29<br>55.35<br>54.09<br>56.98<br>51.16<br>53.92<br>56.26<br>61.13<br>57.18<br>59.65<br>**64.45**<br>61.69<br>**62.61**<br>63.87<br>**63.49**|
|**Performance Gain****_w.r.t._ LaCLIP**|**10.48%**<br>**18.56%**<br>**13.30%**|||**1.64%**<br>**8.71%**|**7.32%**<br>**8.52%**<br>**9.40%**|
|**CC12M**<br>**LaCLIP**<br>**NegCLIP**<br>**NegCLIP++**_(ours)_<br>**TripletCLIP** _(ours)_||75.06<br>65.48<br>58.68<br>77.84<br>69.29<br>63.23<br>82.99<br>78.68<br>75.75<br>**83.66**<br>**81.22**<br>**79.02**||53.47<br>57.66<br>**66.53**<br>62.31<br>61.63<br>**65.47**<br>64.49<br>63.66|67.65<br>66.76<br>63.54<br>67.17<br>69.65<br>68.00<br>70.08<br>**76.01**<br>72.94<br>**73.67**<br>75.43<br>**74.45**|
|**Performance Gain****_w.r.t._ LaCLIP**||**8.60%**<br>**15.75%**<br>**20.34%**|**11.02%**<br>**6.00%**||**8.67%**<br>**7.35%**<br>**10.91%**|
|**DataComp**<br>`small:`**ViT-B/32**_†_ (13M)<br>`medium:`**ViT-B/32**_†_ (128M)<br>`large:`**ViT-B/16**_†_ (1B)<br>`xlarge:`**ViT-L/14**_†_ (13B)||56.90<br>56.85<br>51.99<br>77.00<br>69.54<br>57.68<br>**92.68**<br>79.82<br>63.94<br>**95.52**<br>**84.52**<br>69.99|50.81<br>50.00<br>57.72<br>57.06<br>56.10<br>57.66<br>**65.04**<br>**66.82**||53.93<br>60.55<br>54.43<br>66.73<br>64.88<br>64.37<br>**84.34**<br>**78.61**<br>73.31<br>**91.03**<br>**84.97**<br>**79.70**|



Table 5: **Zero-shot image-text retrieval and classification results. Bold** number indicates the best performance and underlined number denotes the second-best performance. 

|**Methods**|**Retrieval (R@5)**<br>**Image-to-Text**<br>**Text-to-Image**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**|**Retrieval (R@5)**<br>**Image-to-Text**<br>**Text-to-Image**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**|**Zero-shot Classifcation**<br>**VTAB**<br>**ImageNet1k**<br>**top-1**<br>**top-5**<br>**top-1**<br>**top-5**|**Zero-shot Classifcation**<br>**VTAB**<br>**ImageNet1k**<br>**top-1**<br>**top-5**<br>**top-1**<br>**top-5**|
|---|---|---|---|---|
||**Image-to-Text**<br>**MSCOCO**<br>**Flickr30k**||**VTAB**<br>**top-1**<br>**top-5**||
|CC3M<br>**LaCLIP**<br>**LaCLIP + HN**<br>**NegCLIP**<br>**NegCLIP++**_(ours)_<br>**TripletCLIP** _(ours)_|5.06<br>10.90<br>8.08<br>16.10<br>6.32<br>13.80<br>5.8<br>11.20<br>**10.38**<br>**22.00**|5.97<br>10.84<br>8.64<br>16.64<br>6.61<br>12.96<br>6.19<br>10.24<br>**11.28**<br>**22.00**|11.56<br>34.72<br>**12.31**<br>37.14<br>12.25<br>36.38<br>11.65<br>35.47<br>**12.31**<br>**41.45**|3.79<br>10.49<br>5.75<br>15.22<br>4.67<br>12.69<br>3.84<br>10.52<br>**7.32**<br>**18.34**|
|**Performance Gain**|**5.32%**<br>**11.1%**|**5.31%**<br>**11.16%**|**0.75%**<br>**6.73%**|**3.53%**<br>**7.85%**|
|CC12M<br>**LaCLIP**<br>**NegCLIP**<br>**NegCLIP++**_(ours)_<br>**TripletCLIP** _(ours)_|25.86<br>42.70<br>30.16<br>46.60<br>26.96<br>43.90<br>**33.00**<br>**55.90**|19.78<br>36.30<br>23.11<br>41.70<br>22.69<br>42.86<br>**28.50**<br>**52.38**|19.08<br>49.06<br>19.12<br>50.56<br>18.48<br>50.38<br>**20.81**<br>**53.40**|19.72<br>41.39<br>20.22<br>42.63<br>19.06<br>40.91<br>**23.31**<br>**47.33**|
|**Performance Gain**|**7.14%**<br>**13.2%**|**8.72%**<br>**16.08%**|**1.73%**<br>**4.34%**|**3.59%**<br>**5.94%**|



baseline. NegCLIP++ incorporates hard negative captions generated using LLM from `TripletData` , enhancing the language comprehension compared to standard NegCLIP. 

**Implementation Details.** Our experiments employ the ViT-B/32 [10] model architecture. To guarantee fair comparisons, we retrain all baseline models using identical hyperparameters. Since the overall training data for NegCLIP and `TripletData` is more than the baseline datasets, we align the number of iterations across all models to equalize the number of image-text pairs seen during training, similar to the strategy used in DataComp. The batch size is fixed to 1024 with the AdamW optimizer at a maximum learning rate of 0.0005, employing cosine decay. Training durations are set at approximately 100k iterations for CC3M and 200k iterations for CC12M. All models are trained on a single A100 (80GB) GPU using bf16 precision. The final training-related experiments and ablations will cost about 1200 A100 GPU hours. We leave the experiments on increasing the data and model size as future works for the community, as scaling further is not viable in the academic budget. 

**Downstream Datasets.** The primary objective of this study is to enhance the compositional capabilities of CLIP models. We mainly evaluate `TripletCLIP` and the baseline models using the challenging SugarCrepe composition benchmark, with additional performance assessments provided in the appendix for older benchmarks. Models are also tested on image-text retrieval tasks for broader evaluation using the Flickr30k [40] and MSCOCO [29] datasets. Zero-shot classification performance is assessed across approximately 18 different datasets. Evaluations adhere to the methodologies outlined in the CLIP-Benchmark[3] or the official benchmark implementations. 

> 3 `https://github.com/LAION-AI/CLIP_benchmark` 

7 

Table 6: **Ablation on filtering high-quality image-text pairs from** `TripletData` **.** We evaluate the `TripletCLIP` after applying the filters to ensure the quality similar to DataComp and compare the baselines on three benchmarks. We find that `TripletCLIP` results in the most optimal solution. **Bold** number indicates the best performance. _†_ represents that results are borrowed from DataComp. 

|**Models**|**Filtering Strategy**|**Data Size**|**Augmentations**|**SugarCrepe**|**Retrieval**|**ImageNet1k**|
|---|---|---|---|---|---|---|
||No fltering|12.8|-|55.61|6.49|2.7|
|**CLIP**_†_|CLIP Score|3.8|-|57.31|9.08|5.1|
||Image-based_∩_CLIP Score|1.4|-|54.75|5.63|3.9|
|**LaCLIP**|No fltering (CC3M)|2.6|-|54.09|8.19|3.79|
|**TripletCLIP**|No fltering (CC3M)|2.6|2.6|63.49|16.42|7.31|
|**TripletCLIP++**|CLIP Score (from CC12M)|1.4|1.4|**66.09**|**19.85**|**8.85**|



Table 7: **Finetuning-based composition evaluations of the methods on SugarCrepe benchmark. Bold** number indicates the best performance and underlined number denotes the second-best performance. 

|mance.||||
|---|---|---|---|
|**Methods**|**Replace**<br>**Object**<br>**Attribute**<br>**Relation**|**Swap**<br>**Object**<br>**Attribute**|**Add**<br>**Overall**<br>**Object**<br>**Attribute**<br>**Avg.**|
|**CLIP**<br>**CLIP (fnetuned)**<br>**NegCLIP**<br>**Baseline [44]**<br>**CoN-CLIP [50]**<br>**TSVLC (RB) [12]**<br>**TSVLC (LLM+RB) [12]**<br>**DAC[11]**|90.92<br>80.08<br>69.13<br>90.92<br>79.69<br>64.01<br>91.53<br>83.25<br>73.97<br>93.22<br>84.39<br>67.35<br>93.58<br>80.96<br>63.3<br>91.34<br>81.34<br>64.15<br>88.13<br>76.78<br>62.73<br>**94.43**<br>**89.48**<br>**84.35**|61.22<br>64.26<br>60.82<br>64.26<br>72.24<br>67.72<br>62.04<br>70.12<br>**87.29**<br>**79.62**<br>68.16<br>69.07<br>64.08<br>66.67<br>75.10<br>74.17|77.16<br>68.64<br>73.06<br>84.67<br>78.76<br>74.73<br>86.95<br>88.44<br>80.59<br>88.31<br>79.48<br>77.84<br>59.18<br>65.16<br>75.58<br>79.49<br>91.33<br>77.84<br>75.80<br>81.07<br>73.61<br>89.67<br>**97.69**<br>**86.41**|
|**TripletCLIP****_(ours)_**|**94.43**<br>85.53<br>80.94|69.80<br>69.82|**90.40**<br>86.27<br>82.46|



## **4.2 Compositional reasoning** 

We comprehensively analyze the compositional understanding of models on the SugarCrepe benchmark, as detailed in Table 4. Notably, `TripletCLIP` consistently outperforms all baseline models across all sub-categories of SugarCrepe on both the CC12M/CC3M training datasets. Specifically, `TripletCLIP` surpasses LaCLIP and NegCLIP by **10.91%/9.4%** and **6.45%/6.31%** on the CC12M and CC3M datasets, respectively. Our enhanced baseline, NegCLIP++, also shows improvement over standard NegCLIP, highlighting the benefits of LLM-generated negatives. Nevertheless, `TripletCLIP` further advances performance, underscoring the critical role of hard negative image-text pairs, not just text. Additional comparisons on older composition benchmarks (Valse [35], Cola [42], and Winoground [53]) in the appendix reveal `TripletCLIP` ’s consistent performance. Table 4 also contrasts `TripletCLIP` with models trained using the DataComp approach, which involves more parameters and training data, demonstrating that `TripletCLIP` achieves comparable performance to a ViT-B/16 model trained on 1 billion image-text pairs. 

## **4.3 Zero-shot evaluations** 

**Image-Text Retrieval.** In Table 5, we summarize the performance of models on text-to-image (T2I) and image-to-text (I2T) retrieval tasks on MSCOCO and Flickr30k datasets, where we report R@5 scores. Remarkably, `TripletCLIP` significantly outperforms baseline models by an average of **8%/10%** and **8%/12.5%** on I2T and T2I tasks, respectively, on the CC3M and CC12M datasets. Intriguingly, while LaCLIP+HN performs better than NegCLIP, `TripletCLIP` outstrips both. 

**Zero-shot Classification.** Table 5 also presents the average zero-shot classification performance on 18 standard datasets, including ImageNet1k. `TripletCLIP` consistently enhances top-1 accuracy by an average of **3%** and top-5 accuracy by **5-7%** compared to LaCLIP. Like the retrieval performance, LaCLIP+HN exceeds NegCLIP, yet `TripletCLIP` maintains the highest performance. Datasetspecific results are in the appendix. 

8 

Table 8: **Frozen encoder ablation.** LiT style fine-tuning ablations on SugarCrepe, image-text retrieval, and ImageNet1k. **Bold** number indicates the best performance. 

|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Models**<br>**Train Text**|**Train Vision**<br>**SugarCrepe**|**Train Vision**<br>**SugarCrepe**|**Train Vision**<br>**SugarCrepe**|**Train Vision**<br>**SugarCrepe**|**Train Vision**<br>**SugarCrepe**|**Train Vision**<br>**SugarCrepe**|**Train Vision**<br>**SugarCrepe**|**Retrieval**<br>**ImageNet1k**|**Retrieval**<br>**ImageNet1k**|**Retrieval**<br>**ImageNet1k**|**Retrieval**<br>**ImageNet1k**|**Retrieval**<br>**ImageNet1k**|**Retrieval**<br>**ImageNet1k**|**Retrieval**<br>**ImageNet1k**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**LaCLIP**<br>✓||||||||_×_<br>**0.6373**|||||||0.5345<br>31.21%|||||||
|**TripletCLIP** _(ours)_<br>✓||||||||_×_<br>0.6227|||||||**0.6817**<br>**34.25**%|||||||
|**LaCLIP**<br>_×_||||||||✓<br>0.5886|||||||0.1134<br>5.51%|||||||
|**TripletCLIP**_(ours)_<br>_×_||||||||✓<br>**0.6923**|||||||**0.2626**<br>**12.51**%|||||||
|||||||||||||||||||||||
|0.60<br>0.62<br>0.64<br>0.66<br>0.68|SugarCrepe|||||||Image-Text Retrieval|||||||ImageNet1k Performance|||||||
||||||||0.18<br>0.20<br>0.22<br>0.24<br>0.26<br>0.28<br>0.30<br>0.32|||||||10.00<br>12.00<br>14.00<br>16.00||||||||
|||||||||||||||||||||||
|||||||||||||||||||||||
|||||||||||||||||||||||
|||||||||||||||||||||||
|||||||||||||||||||||||
|||||||||||||||||||||||
|||||||||||||||||||||||
|||||||||||||||||||||LaCL<br>Tripl|IP<br>etCLIP|
|||||||||||||||||||||||
|||||||||||||||||||||||
||66000<br>67000<br>68000<br>69000<br>70000<br>Concept Coverage<br>|||||||66000<br>67000<br>68000<br>69000<br>70000<br>Concept Coverage|||||||66000<br>67000<br>68000<br>69000<br>70000<br>Concept Coverage|||||||



Figure 3: Average Results of LaCLIP and `TripletCLIP` for SugarCrepe Compositions, Image-Text Retrieval, and ImageNet1k over increasing concept diversity. 

## **4.4 Finetuning performance** 

In this paper, we focus on pretraining-based experiments as they allow greater flexibility in learning better representations. To complement this, we also performed additional fine-tuning experiments using hyperparameters similar to the baselines (without LoRA) and compared them against various publicly available baselines [44, 50, 12, 11]. As reported in Table 7, `TripletCLIP` improves compositionality and outperforms nearly all baselines. Furthermore, the observed drop in retrieval and zero-shot classification performance (Table 16) is attributed to limitations in the vision encoder, highlighting the challenges of existing pre-trained vision encoders in capturing semantic representations. This is further demonstrated in Table 8. 

## **4.5 Ablations** 

**Can a high-quality filtered dataset improve the performance?** Given that negative images in `TripletData` are generated using SDXL-turbo, these may not always be precise. Inspired by DataComp, we employ a pre-trained CLIP-L/14 to filter the image-text pairs, selecting the highest average similarity pairs (positive and negative) individually (i.e., score = ( _s_ ( _xi, yi_ ) + _s_ ( _x[′] i[, y] i[′]_[))] _[/]_[2][).][The top] 1.4M positive image-text pairs and their corresponding negatives from `TripletData` are selected. Table 6 details this comparison against DataComp pre-trained models. Remarkably, `TripletCLIP` already surpasses baselines without filtered data; however, with the filtered dataset, despite being trained on 50% smaller dataset, `TripletCLIP` ++ shows further performance improvements. This underlines the significant benefits of carefully selected `TripletData` in enhancing the performance. 

**Which modality-specific encoder plays the key role in improving compositionality?** To address this open question, we designed an ablation study similar to LiT, freezing either the pre-trained CLIP vision or text encoder while training the opposite modality-specific encoder from scratch. We observe the performance of LaCLIP and `TripletCLIP` on CC3M, as shown in Table 8. Freezing the vision model results in no performance gain on the SugarCrepe for `TripletCLIP` . However, significant improvements are noted when the vision encoder is actively trained, suggesting that the vision modality may be the bottleneck in compositionality. Notably, `TripletCLIP` outperforms LaCLIP in all settings, further demonstrating its robustness to different pre-training approaches. 

**Concept coverage analysis.** Improving performance on zero-shot transfer learning tasks such as retrieval involves two key components: adding more concept diversity during training and enhancing image-text alignment/compositionality. We create subsets of CC12M data with increasing concept diversity based on unique WordNet synsets. Specifically, we select 3M, 4M, 5M, and 6M subsets for 

9 

Table 9: TripletData as large-scale composition evaluation dataset after [53]. 

|**Methods**|**Text-Score**|**Image-Score**|**Group-Score**|
|---|---|---|---|
|**CLIP**|52.69|29.66|24.64|
|**NegCLIP**|54.84|30.42|25.82|
|**NegCLIP++**|36.50|30.67|20.11|
|**TripletCLIP****_(ours)_**|**92.25**|**66.82**|**64.30**|



training LaCLIP, while `TripletCLIP` training involves only half of these training data as positive pairs, and the rest are corresponding augmentations. Evaluations across SugarCrepe, retrieval tasks, and ImageNet1k (see Figure 3) indicate that `TripletCLIP` not only enhances SugarCrepe performance even at lower concept coverage levels but also significantly outperforms similar concept coverage in retrieval tasks, matching LaCLIP’s performance on zero-shot classification tasks that do not require compositionality at all. This bolsters our argument that incorporating hard negatives from both modalities markedly improves compositional understanding in CLIP, while baseline struggles to do so even with more concept diversity. 

**What if** `TripletData` **is used for large-scale compositional evaluations?** We evaluated the CC12M pre-trained models on a 50,000 random subset of the CC3M dataset using a Winoground-style approach [53]. As shown in Table 9, `TripletCLIP` significantly improves performance compared to the baselines. However, we partially attribute this improvement to spurious correlations learned from the data. At the same time, we note that the models have not fully converged, suggesting minimal risk of overfitting to these spurious correlations. 

## **5 Conclusion** 

In this work, we introduce `TripletCLIP` , a novel approach to enhancing compositional reasoning in vision-language models through the strategic incorporation of hard negative image-text pairs. Our comprehensive experiments across a suite of benchmarks demonstrate that `TripletCLIP` significantly outperforms existing methodologies such as LaCLIP and NegCLIP, achieving notable gains not only in compositionality but also in zero-shot classification and retrieval tasks as well. Further, our ablation studies highlight the critical role of modality-specific training and the careful curation of training data, underscoring the importance of both hard negative image and text components in the learning process. `TripletCLIP` ’s effectiveness with a smaller, refined dataset suggests a promising direction for future research—maximizing performance without the need for extensive data collection, thereby reducing computational costs and enhancing model efficiency. To this end, we provide an intriguing application of synthetic datasets via hard negative image-text pairs for vision-language tasks that could be easily extended to improve Multimodal Large Language Models and Text-to-Image generative models. 

**Limitations.** Due to constraints inherent in academic settings and limited computational resources, we were unable to scale `TripletCLIP` to handle hundreds of millions of image-text pairs or employ larger models within the scope of this study. Nevertheless, our results indicate a promising direction for future research within a consistent experimental framework, and we encourage subsequent work to explore scaling both the `TripletData` and `TripletCLIP` . Our experimental focus was primarily on the CLIP and LiT methodologies. With additional resources, however, extending our methodologies to more advanced contrastive learning techniques, such as SigLIP, would be feasible. In conclusion, our work introduces a compelling strategy for integrating open-ended hard negatives (both text and image) during the pre-training phase, providing a methodology and large-scale data that could benefit a variety of research domains. 

## **Acknowledgments** 

This work was supported by NSF RI grants #1750082, #2132724, and CPS grant #2038666. We thank the Research Computing (RC) at Arizona State University (ASU) for providing computing resources. The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers. 

10 

## **References** 

- [1] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15619–15629, 2023. 

- [2] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Composed image retrieval using contrastive learning and task-oriented clip-based features. _ACM Transactions on Multimedia Computing, Communications and Applications_ , 20(3):1–24, 2023. 

- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. _Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf_ , 2(3):8, 2023. 

- [4] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 9650–9660, 2021. 

- [5] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 3558–3568, June 2021. 

- [6] Agneet Chatterjee, Gabriela Ben Melech Stan, Estelle Aflalo, Sayak Paul, Dhruba Ghosh, Tejas Gokhale, Ludwig Schmidt, Hannaneh Hajishirzi, Vasudev Lal, Chitta Baral, et al. Getting it right: Improving spatial consistency in text-to-image models. In _European Conference on Computer Vision_ , pages 204–222. Springer, 2024. 

- [7] Yuxiao Chen, Jianbo Yuan, Yu Tian, Shijie Geng, Xinyu Li, Ding Zhou, Dimitris N Metaxas, and Hongxia Yang. Revisiting multimodal representation in contrastive learning: from patch and token embeddings to finite discrete tokens. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15095–15104, 2023. 

- [8] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Shanmukha Ramakrishna Vedantam. Hyperbolic image-text representations. In _International Conference on Machine Learning_ , pages 7694–7731. PMLR, 2023. 

- [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, 2019. 

- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- [11] Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola CascanteBonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, et al. Dense and aligned captions (dac) promote compositional reasoning in vl models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [12] Sivan Doveh, Assaf Arbelle, Sivan Harary, Eli Schwartz, Roei Herzig, Raja Giryes, Rogerio Feris, Rameswar Panda, Shimon Ullman, and Leonid Karlinsky. Teaching structured vision & language concepts to vision & language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 2657–2668, 2023. 

- [13] Linus Ericsson, Henry Gouk, Chen Change Loy, and Timothy M Hospedales. Self-supervised representation learning: Introduction, advances, and challenges. _IEEE Signal Processing Magazine_ , 39(3):42–62, 2022. 

- [14] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with language rewrites. _Advances in Neural Information Processing Systems_ , 36, 2024. 

11 

- [15] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and Vaishaal Shankar. Data filtering networks. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- [16] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [17] Shashank Goel, Hritik Bansal, Sumit Bhatia, Ryan Rossi, Vishwa Vinay, and Aditya Grover. Cyclip: Cyclic contrastive language-image pretraining. _Advances in Neural Information Processing Systems_ , 35:6704–6719, 2022. 

- [18] Hasan Abed Al Kader Hammoud, Hani Itani, Fabio Pizzati, Philip Torr, Adel Bibi, and Bernard Ghanem. Synthclip: Are we ready for a fully synthetic clip training? _arXiv preprint arXiv:2402.01832_ , 2024. 

- [19] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pages 7514–7528, 2021. 

- [20] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [21] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, and Liangliang Cao. Diffusion model-based image editing: A survey. _arXiv preprint arXiv:2402.17525_ , 2024. 

- [22] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International conference on machine learning_ , pages 4904–4916. PMLR, 2021. 

- [23] Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, and Humphrey Shi. Learning mask-aware clip representations for zero-shot segmentation. _Advances in Neural Information Processing Systems_ , 36:35631–35653, 2023. 

- [24] Changhoon Kim, Kyle Min, and Yezhou Yang. Race: Robust adversarial concept erasure for secure text-to-image diffusion model. _arXiv preprint arXiv:2405.16341_ , 2024. 

- [25] Zhengfeng Lai, Haotian Zhang, Wentao Wu, Haoping Bai, Aleksei Timofeev, Xianzhi Du, Zhe Gan, Jiulong Shan, Chen-Nee Chuah, Yinfei Yang, et al. From scarcity to efficiency: Improving clip training via visual-enriched captions. _arXiv preprint arXiv:2310.07699_ , 2023. 

- [26] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models?, 2024. 

- [27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pages 19730–19742. PMLR, 2023. 

- [28] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. _arXiv preprint arXiv:2308.03281_ , 2023. 

- [29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer, 2014. 

- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems_ , volume 36, pages 34892–34916. Curran Associates, Inc., 2023. 

12 

- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36, 2024. 

- [32] Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, and Ranjay Krishna. Crepe: Can vision-language foundation models reason compositionally? In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10910–10921, 2023. 

- [33] George A. Miller. Wordnet: a lexical database for english. _Commun. ACM_ , 38(11):39–41, nov 1995. 

- [34] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 

- [35] Letitia Parcalabescu, Michele Cafagna, Lilitta Muradjan, Anette Frank, Iacer Calixto, and Albert Gatt. Valse: A task-independent benchmark for vision and language models centered on linguistic phenomena. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8253–8280, 2022. 

- [36] Maitreya Patel, Tejas Gokhale, Chitta Baral, and Yezhou Yang. Conceptbed: Evaluating concept learning abilities of text-to-image diffusion models. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 14554–14562, 2024. 

- [37] Maitreya Patel, Changhoon Kim, Sheng Cheng, Chitta Baral, and Yezhou Yang. Eclipse: A resource-efficient text-to-image prior for image generations. _arXiv preprint arXiv:2312.04655_ , 2023. 

- [38] Maitreya Patel, Changhoon Kim, Sheng Cheng, Chitta Baral, and Yezhou Yang. Eclipse: A resource-efficient text-to-image prior for image generations. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9069–9078, 2024. 

- [39] Wujian Peng, Sicheng Xie, Zuyao You, Shiyi Lan, and Zuxuan Wu. Synthesize diagnose and optimize: Towards fine-grained vision-language understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 13279–13288, 2024. 

- [40] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _Proceedings of the IEEE international conference on computer vision_ , pages 2641–2649, 2015. 

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

- [42] Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan Plummer, Ranjay Krishna, and Kate Saenko. cola: A benchmark for compositional text-to-image retrieval. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [43] Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. Contrastive learning with hard negative samples. _arXiv preprint arXiv:2010.04592_ , 2020. 

- [44] Ugur Sahin, Hang Li, Qadeer Khan, Daniel Cremers, and Volker Tresp. Enhancing multimodal compositional reasoning of visual language models with generative negative mining. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 5563–5573, 2024. 

- [45] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. _arXiv preprint arXiv:2403.12015_ , 2024. 

- [46] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. _arXiv preprint arXiv:2311.17042_ , 2023. 

13 

- [47] Michael Saxon, Fatima Jahara, Mahsa Khoshnoodi, Yujie Lu, Aditya Sharma, and William Yang Wang. Who evaluates the evaluations? objectively scoring text-to-image prompt coherence metrics with t2iscorescore (ts2). _arXiv preprint arXiv:2404.04251_ , 2024. 

- [48] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. _Advances in Neural Information Processing Systems_ , 35:25278–25294, 2022. 

- [49] Harman Singh, Pengchuan Zhang, Qifan Wang, Mengjiao Wang, Wenhan Xiong, Jingfei Du, and Yu Chen. Coarse-to-fine contrastive learning in image-text-graph space for improved visionlanguage compositionality. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 869–893, 2023. 

- [50] Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa Singh, and Aparna Bharati. Learn" no" to say" yes" better: Improving vision-language models via negations. _arXiv preprint arXiv:2403.20312_ , 2024. 

- [51] Haoyu Song, Li Dong, Weinan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6088–6100, 2022. 

- [52] Ming Tao, Bing-Kun Bao, Hao Tang, and Changsheng Xu. Galip: Generative adversarial clips for text-to-image synthesis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14214–14223, 2023. 

- [53] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5238–5248, 2022. 

- [54] Tan Wang, Kevin Lin, Linjie Li, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Equivariant similarity for vision-language foundation models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 11998–12008, October 2023. 

- [55] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. In _2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022_ , 2022. 

- [56] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, ShangWen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. _arXiv preprint arXiv:2309.16671_ , 2023. 

- [57] Kaicheng Yang, Jiankang Deng, Xiang An, Jiawei Li, Ziyong Feng, Jia Guo, Jing Yang, and Tongliang Liu. Alip: Adaptive language-image pre-training with synthetic caption. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2922–2931, 2023. 

- [58] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [59] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18123–18133, 2022. 

- [60] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, and Ming-Wei Chang. Magiclens: Self-supervised image retrieval with open-ended instructions. In _Forty-first International Conference on Machine Learning_ , 2024. 

14 

- [61] Le Zhang, Rabiul Awal, and Aishwarya Agrawal. Contrasting intra-modal and ranking crossmodal hard negatives to enhance visio-linguistic fine-grained understanding. _arXiv preprint arXiv:2306.08832_ , 2023. 

- [62] Ziqin Zhou, Yinjie Lei, Bowen Zhang, Lingqiao Liu, and Yifan Liu. Zegclip: Towards adapting clip for zero-shot semantic segmentation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11175–11185, 2023. 

15 

## **A Broader Impact** 

In this study, we have demonstrated the potential of utilizing high-quality positive and negative pairs to enhance the compositional understanding of vision-language models like CLIP through the introduction of `TripletCLIP` . While our findings are specific to `TripletCLIP` , the underlying techniques hold promise for broader applications, including enhancing visual understanding in Multimodal Large Language Models (MLLMs) and Text-to-Image diffusion models. Although our approach yields significant performance improvements, it does require resources to generate largescale synthetic datasets. We encourage future research to explore the utility of this pre-training strategy within the latent space, which could reduce dependence on large generative models. Initiatives like JEPA [1] have already demonstrated the efficacy of focusing on latent space models, which suggests a promising avenue for reducing computational overhead. Importantly, our experiments reveal that significant enhancements in model performance are achievable even with substantially smaller data scales. This finding suggests that, when scaled appropriately, our methodology could substantially diminish resource dependencies and enhance the efficiency of pre-training processes for CLIP-like models. 

## **B Pseudocode of TripletCLIP** 

**def** forward ( b a t c h _ p o s i t i v e , b a t c h _ n e g a t i v e ) : _# get p o s i t i v e and n e g a t i v e image−t e x t p a i r s_ img_pos , t x t _ p o s = b a t c h _ p o s i t i v e img_neg , txt_neg = b a t c h _ n e g a t i v e 

_# compute p o s i t i v e image and t e x t r e p r e s e n t a t i o n s_ image_features_pos = c l i p . encode_image ( img_pos ) t e x t _ f e a t u r e s _ p o s = c l i p . encode_text ( t x t _ p o s ) _# compute n e g a t i v e image and t e x t r e p r e s e n t a t i o n s_ image_features_neg = c l i p . encode_image ( img_neg ) t e x t _ f e a t u r e s _ n e g = c l i p . encode_text ( txt_neg ) p o s i t i v e _ t x t = t o r c h . c a t ( [ t e x t _ f e a t u r e s _ p o s , t e x t _ f e a t u r e s _ n e g ] ) n e g a t i v e _ t x t = t o r c h . c a t ( [ t e x t _ f e a t u r e s _ n e g , t e x t _ f e a t u r e s _ p o s ] ) _# compute NegCLIP l o s s e s_ loss_1 = n e g c l i p _ l o s s ( image_features_pos , p o s i t i v e _ t x t ) loss_2 = n e g c l i p _ l o s s ( image_features_neg , n e g a t i v e _ t x t ) l o s s = loss_1 + loss_2 **return** l o s s 

## **C Hyperparameters** 

Table 10: Detailed pre-training hyper-parameters for CLIP training across various experiments and ablations. 

|**Hyperparameters**|**CC3M**|**CC12M**|**LiT**|**Concept Coverage Ablations**|
|---|---|---|---|---|
|**Batch size**|1024|1024|1024|1024|
|**Optimizer**|AdamW|AdamW|AdamW|AdamW|
|**Learning rate**|5_×_10_−_4|5_×_10_−_4|5_×_10_−_4|5_×_10_−_4|
|**Weight decay**|0.5|0.5|0.5|0.5|
|**Adam**_β_|(0.9, 0.999)|(0.9, 0.999)|(0.9, 0.999)|(0.9, 0.999)|
|**Adam**_ϵ_|1_×_10_−_8|1_×_10_−_8|1_×_10_−_8|1_×_10_−_8|
|**Total steps**|90,000|230,000|90,000|200,000|
|**Learning rate schedule**|cosine decay|cosine decay|cosine decay|cosine decay|



16 

Table 11: **Composition evaluations of the methods on various benchmarks. Bold** number indicates the best performance and underlined number denotes the second-best performance. 

|**Methods**<br>**Valse**|**Cola**<br>**Txt2Img**<br>**Img2Txt**<br>**Group**|**Winoground**<br>**SugarCrepe**<br>**Overall**<br>**Txt2Img**<br>**Img2Txt**<br>**Group**|
|---|---|---|
|**CC3M**<br>**LaCLIP**<br>43.19<br>**LaCLIP + HN**<br>47.91<br>**NegCLIP**<br>48.06<br>**NegCLIP++**_(ours)_<br>43.65<br>**TripletCLIP**_(ours)_<br>48.36<br>**TripletCLIP++** _(ours)_<br>**48.53**|28.10<br>13.33<br>10.48<br>20.95<br>5.54<br>2.38<br>21.42<br>14.76<br>8.57<br>29.05<br>13.33<br>7.14<br>**31.43**<br>13.33<br>9.52<br>**31.43**<br>**15.71**<br>**12.86**|**24.75**<br>9.25<br>6.00<br>54.09<br>23.65<br>23.25<br>4.25<br>3.00<br>53.92<br>20.15<br>21.00<br>8.50<br>5.25<br>57.18<br>23.09<br>22.00<br>5.50<br>3.25<br>61.69<br>23.20<br>24.25<br>6.25<br>4.25<br>63.49<br>25.11<br>22.25<br>**9.50**<br>**6.75**<br>**66.09**<br>**26.64**|
|**Performance Gain****_w.r.t._ LaCLIP**<br>**5.34%**|**3.33%**<br>**2.38%**<br>**2.38%**|**-2.50**<br>**0.25%**<br>**0.75%**<br>**12.00**<br>**2.99%**|
|**CC12M**<br>**LaCLIP**<br>55.69<br>**NegCLIP**<br>**58.59**<br>**NegCLIP++**_(ours)_<br>58.12<br>**TripletCLIP** _(ours)_<br>57.57|20.95<br>15.71<br>7.62<br>27.14<br>15.24<br>5.71<br>**33.33**<br>11.43<br>7.14<br>27.62<br>**19.53**<br>**11.43**|**26.25**<br>**8.00**<br>**6.25**<br>67.21<br>25.96<br>18.25<br>6.50<br>4.25<br>68.41<br>25.51<br>23.50<br>7.75<br>5.50<br>73.05<br>27.48<br>23.25<br>6.25<br>4.25<br>**74.55**<br>**28.06**|
|**Performance Gain****_w.r.t._ LaCLIP**<br>**1.88%**|**6.67%**<br>**3.82%**<br>**3.81%**|**-3.00**<br>**-1.75**<br>**-2.00**<br>**7.35**<br>**2.10%**|



Table 12: **Dataset-specific zero-shot classification results. Bold** number indicates the best performance and underlined number denotes the second-best performance. 

||**Models**|**fgvc aircraft**|**imagenet1k**|**caltech101**|**cifar100**|**clevr closest object distance**|**clevr count all**|**dmlab**|**dsprites label orientation**|**dsprites label x position**|**dtd**|**eurosat**|**fowers**|**kitti closest vehicle distance**|**pcam**|**pets**|**smallnorb label azimuth**|**smallnorb label elevation**|**svhn**|**Avg.**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||**LaCLIP**|0.84|3.79|24.65|8.04|15.86|11.26|**19.02**|**2.63**|3.14|5.53|8.56|3.43|26.72|48.16|3.11|5.75|10.86|6.68|11.56|
|**CC3M**|**LaCLIP+HN**<br>**NegCLIP**<br>**NegCLIP++**_(ours)_<br>**TripletCLIP**_(ours)_|1.23<br>1.02<br>1.02<br>**1.29**|5.75<br>4.67<br>3.84<br>7.32|33.20<br>29.44<br>20.61<br>37.05|8.84<br>8.17<br>6.44<br>**14.37**|17.18<br>**22.37**<br>22.03<br>15.55|11.18<br>12.97<br>10.01<br>13.49|16.60<br>18.81<br>18.91<br>15.37|2.52<br>2.57<br>2.60<br>2.62|3.10<br>3.19<br>3.21<br>2.98|6.65<br>6.81<br>4.63<br>8.78|10.02<br>7.39<br>11.70<br>22.74|4.29<br>3.77<br>3.59<br>5.97|18.85<br>**28.83**<br>23.91<br>16.74|51.93<br>41.72<br>49.34<br>53.49|4.44<br>4.17<br>4.74<br>5.81|4.92<br>4.84<br>5.43<br>5.34|11.12<br>11.37<br>10.02<br>11.51|9.78<br>8.32<br>7.70<br>**10.22**|12.31<br>12.25<br>11.65<br>12.31|
||**TripletCLIP++**_(ours)_|0.90|**8.85**|**38.54**|11.83|20.72|**13.79**|17.46|2.52|**3.58**|**8.99**|**24.43**|**8.20**|19.41|**56.60**|**19.38**|**5.76**|**12.46**|7.91|**15.62**|
|**CC12M**|**LaCLIP**<br>**NegCLIP**<br>**NegCLIP++**_(ours)_<br>**TripletCLIP**_(ours)_|**1.59**<br>1.56<br>1.38<br>1.29|19.72<br>20.22<br>19.06<br>**23.31**|62.98<br>63.45<br>64.66<br>**65.34**|23.62<br>26.02<br>24.86<br>**30.30**|11.99<br>**18.89**<br>15.96<br>16.26|11.38<br>15.80<br>14.87<br>**17.67**|**19.85**<br>16.20<br>15.62<br>16.58|2.91<br>2.45<br>**2.94**<br>2.80|**3.20**<br>3.13<br>3.07<br>3.18|13.03<br>14.26<br>13.62<br>**17.45**|**23.85**<br>17.43<br>14.33<br>23.26|13.43<br>13.55<br>14.30<br>**15.43**|24.05<br>22.78<br>16.60<br>**25.74**|50.38<br>50.22<br>49.98<br>**51.04**|**34.72**<br>30.85<br>31.92<br>33.25|**6.14**<br>5.37<br>5.42<br>5.58|11.04<br>12.04<br>12.08<br>**12.38**|9.54<br>10.04<br>12.13<br>**13.67**|19.08<br>19.12<br>18.48<br>**20.81**|



Table 10 provides a comprehensive overview of the pre-training hyperparameters employed across all baseline models and `TripletCLIP` . To ensure fair comparisons, we standardized the hyperparameters across all methodologies. Although larger batch sizes are typically associated with improved performance in contrastive learning, computational constraints necessitated fixing the batch size at 1024 for all experiments. To accommodate this batch size on a single A100 GPU, we employed bf16 precision. In terms of computational resources, experiments using the CC3M dataset required approximately 16 GPU hours, while those involving the CC12M dataset utilized up to 56 GPU hours per experiment. 

## **D Detailed Results** 

## **D.1 Compositional reasoning** 

Previously, we reported the results on SugarCrepe, the most challenging dataset, noted for its absence of language biases. However, evaluations were also conducted on other benchmarks, such as Valse, Cola, and Winoground. As indicated in Table 11, `TripletCLIP` achieves overall improvements of **2-3%** compared to LaCLIP and NegCLIP. The Valse benchmark, which contains text prompts that heavily favor the perturbations made for NegCLIP, shows a strong performance from NegCLIP, while NegCLIP++ encounters difficulties. Interestingly, `TripletCLIP` faces challenges in maintaining performance on Winoground, and baseline LaCLIP maintains the SOTA, which is counterintuitive to other benchmarks. Nonetheless, `TripletCLIP` still manages to outperform NegCLIP significantly. These results affirm that `TripletCLIP` sets a new standard for state-of-the-art compositional reasoning across diverse benchmarks. 

## **D.2 Dataset-specific zero-shot classification** 

Table 12 provides fine-grained results for the 18 zero-shot classification datasets. It can be observed that `TripletCLIP` consistently outperforms the baselines, achieving the best average results across these challenging datasets. Although the improvements are marginal, they are in line with expectations. 

17 

Table 13: **Composition evaluations of the methods on various benchmarks. Bold** number indicates the best performance and underlined number denotes the second-best performance. 

|**Methods**|**Text-to-Ima**|**ge Retrieval**<br>**Flickr30k**<br>**R@1**<br>**R@5**<br>**R@10**|**Image-to-T**|**ext Retrieval**<br>**Flickr30k**<br>**R@1**<br>**R@5**<br>**R@10**|
|---|---|---|---|---|
||**MSCOCO**<br>**R@1**<br>**R@5**<br>**R@10**||**MSCOCO**<br>**R@1**<br>**R@5**<br>**R@10**||
|**CC3M**<br>**LaCLIP**<br>**LaCLIP + HN**<br>**NegCLIP**<br>**NegCLIP***<br>**TripletCLIP**|1.56<br>5.06<br>8.90<br>2.60<br>8.08<br>13.02<br>1.74<br>6.32<br>10.44<br>1.50<br>5.80<br>9.70<br>**3.16**<br>**10.38**<br>**16.22**|3.70<br>10.90<br>16.00<br>6.30<br>16.10<br>22.40<br>4.90<br>13.80<br>19.60<br>4.40<br>11.20<br>16.30<br>**9.10**<br>**22.00**<br>**29.80**|1.73<br>5.97<br>9.50<br>2.66<br>8.64<br>13.38<br>1.95<br>6.61<br>10.62<br>1.85<br>6.19<br>9.82<br>**3.58**<br>**11.28**<br>**17.39**|3.54<br>10.84<br>16.02<br>5.98<br>16.64<br>23.82<br>4.76<br>12.96<br>18.50<br>3.50<br>10.24<br>15.20<br>**8.38**<br>**22.00**<br>**29.56**|
|**Performance Gain vs. CLIP**|**1.60%**<br>**5.32%**<br>**7.32%**|**5.40%**<br>**11.10%**<br>**13.80%**|**1.85%**<br>**5.31%**<br>**7.89%**|**4.84%**<br>**11.16%**<br>**13.54%**|
|**CC12M**<br>**LaCLIP**<br>**NegCLIP**<br>**NegCLIP***<br>**TripletCLIP**|10.50<br>25.86<br>35.60<br>12.32<br>30.16<br>41.44<br>10.94<br>26.96<br>36.64<br>**14.60**<br>**33.00**<br>**43.84**|21.30<br>42.70<br>54.60<br>24.70<br>46.60<br>58.20<br>18.70<br>43.90<br>55.90<br>**28.00**<br>**55.90**<br>**65.70**|7.21<br>19.78<br>28.37<br>8.56<br>23.11<br>32.60<br>8.80<br>22.69<br>32.13<br>**11.38**<br>**28.50**<br>**39.04**|15.06<br>36.30<br>47.00<br>18.66<br>41.70<br>53.42<br>18.24<br>42.86<br>53.78<br>**25.28**<br>**52.38**<br>**63.32**|
|**Performance Gain vs. CLIP**|**4.10%**<br>**7.14%**<br>**8.24%**|**6.70%**<br>**13.20%**<br>**11.10%**|**4.17%**<br>**8.72%**<br>**10.67%**|**10.22%**<br>**16.08%**<br>**16.32%**|



Table 14: **Ablation on choice pre-trained LLM.** We train NegCLIP++ (ViT-B/32) on negative captions generated from various LLMs and report SugarCrepe, Flickr30k Retrieval (R@5), and ImageNet-top5 performances. 

|**Models**|**SugarCrepe (avg.)**|**Retrieval (R@5)**|**ImageNet1k (top-5)**|
|---|---|---|---|
|**Gemma-2b-it**|56.00|12.60|**12.09%**|
|**Phi-3-mini-4k-instruct**|61.22|**13.02**|10.94%|
|**Mistral-7b-instruct-v0.2**|**61.69**|10.72|10.52%|



Table 15: **Ablation on CyCLIP with TripletLoss.** To evaluate the compatibility of TripletLoss with the CyCLIP, we train the ViT-B/32 models from search on CC3M with 512 batch size and report SugarCrepe, Flickr30k Retrieval (R@5), and ImageNet-top5 performances. 

|**Models**|**SugarCrepe (avg.)**|**Retrieval (R@5)**|**ImageNet1k (top-5)**|
|---|---|---|---|
|**LaCLIP**|55.11|12.80|12.58%|
|**TripletCLIP**|**65.71**|**24.62**|**19.95%**|
|**CyCLIP**|54.62|11.77|13.01%|
|**CyCLIP+TripletLoss**|**58.64**|**20.29**|**19.05%**|



As discussed in Figure 3, `TripletCLIP` does not introduce new concepts into the training data but focuses on augmentations that enhance representation without increasing concept diversity. These enhanced representations from `TripletCLIP` lead to average improvements of **1-3%** depending on the scenario. 

## **D.3 Detailed image-text retrieval performance** 

Table 13 presents detailed T2I and I2T retrieval results for the MSCOCO and Flickr30k datasets. We report results at different recall thresholds: R@1, R@5, and R@10. The data shows that `TripletCLIP` significantly outperforms the baselines across all recall rates. On average, `TripletCLIP` achieves a performance gain of **7-11%** over LaCLIP. Additionally, `TripletCLIP` improves performance by **3-6%** compared to previous state-of-the-art baselines. 

## **D.4 Additional ablations** 

**Choice of Pre-trained LLM.** We provide further details regarding the selection of LLMs for generating hard negative captions. NegCLIP++ was trained on 3 million generated negative captions for CC3M using three different LLMs, and the results are reported in Table 14. We find that Phi-3 achieves the best average performance, while Gemma-2b unexpectedly has a notable negative impact on compositionality. However, we opted to use Mistral-7b-instruct-v0.2, as Phi-3 was released after our experiments were completed, preventing its earlier evaluation. 

**Evaluating the Orthogonality of TripletLoss.** As discussed in Section 2, `TripletCLIP` can be integrated with various previously proposed methodologies. To evaluate this, we applied TripletLoss in conjunction with CyCLIP [17] and reported the results in Table 15. The most significant performance improvement is observed with the TripletLoss alone. However, TripletLoss also enhances the 

18 

Table 16: **Finetuning-based evaluations of the methods on Retrieval and ImageNet-1k benchmarks. Bold** number indicates the best performance and underlined number denotes the second-best performance. 

|**Methods**|**Retrieval**<br>**Ima**<br>**Text Retrieval (R@5)**<br>**Image Retrieval (R@5)**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**<br>**top-**|**Retrieval**<br>**Ima**<br>**Text Retrieval (R@5)**<br>**Image Retrieval (R@5)**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**<br>**top-**|**Retrieval**<br>**Ima**<br>**Text Retrieval (R@5)**<br>**Image Retrieval (R@5)**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**<br>**top-**|**Retrieval**<br>**Ima**<br>**Text Retrieval (R@5)**<br>**Image Retrieval (R@5)**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**<br>**top-**|**Retrieval**<br>**Ima**<br>**Text Retrieval (R@5)**<br>**Image Retrieval (R@5)**<br>**MSCOCO**<br>**Flickr30k**<br>**MSCOCO**<br>**Flickr30k**<br>**top-**|**geNet 1k**<br>**1**<br>**top-5**|
|---|---|---|---|---|---|---|
|**CLIP**<br>**CLIP (fnetuned)**<br>**NegCLIP**<br>**Baseline [44]**<br>**TSVLC (RB) [12]**<br>**TSVLC (LLM+RB) [12**<br>**DAC[11]**|**]**|**74.9**<br>68.9<br>66.00<br>81.4<br>71.70<br>71.82<br>54.5|94.60<br>88.40<br>88.60<br>**96.0**<br>93.00<br>92.50<br>79.60||55.92<br>83.38<br>**63.3**<br>53.50<br>81.10<br>49.9<br>53.41<br>81.12<br>48.8<br>**67.49**<br>**89.84**<br>61.4<br>62.01<br>87.12<br>58.8<br>62.24<br>87.46<br>59.7<br>63.51<br>87.84<br>51.0|**1**<br>**88.22**<br>5<br>79.16<br>5<br>78.34<br>0<br>88.10<br>1<br>85.97<br>7<br>87.02<br>2<br>81.22|
|**TripletCLIP****_(ours)_**||55.6|82.60||53.32<br>80.88<br>45.9|2<br>75.54|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0019-02.png)


**----- Start of picture text -----**<br>
0.40<br>DINOv2 GTE<br>0.30<br>CLIP 0.35 CLIP<br>0.25 0.30<br>0.20 0.25<br>0.20<br>0.15<br>0.15<br>0.10<br>0.10<br>0.05<br>0.05<br>0.00 0.00<br>0.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0<br>Cosine Similarity Cosine Similarity<br>Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 4: Positive _vs._ Negative modality-specific pair-based similarity distribution of pre-trained CLIP ViT-B/32 model _w.r.t._ the vision and text-only encoders. The left plot is the vision embedding similarities between positive and negative images. The right plot is the text embedding similarities between positive and negative captions. In the ideal scenario, the distribution should be skewed towards 0.0, which indicates that the model can correctly distinguish between the positive and negative data. 

performance over the base CyCLIP, demonstrating its adaptability and orthogonality with respect to other approaches. 

## **E Encoder Representation Distribution Analysis** 

Remember, this study aims to learn the representations that can distinguish between two data points that are very similar but semantically different. Firstly, we take LaCLIP and `TripletCLIP` models trained on CC12M. We also sampled 50000 positive+negative pairs from CC3M. Then, we measure the vision and text modality-specific cosine similarities between positive and negative pairs and plot the distribution (see Figure 5). It can be observed that vision representations from `TripletCLIP` are more skewed towards 0.0, suggesting that the vision encoder can distinguish between hard negative samples better than the baseline LaCLIP. However, in the case of the text modality, both methods perform similarly. This aligns with our findings from Table 8 that the vision encoder plays a crucial role in improving the compositionality, and to achieve this, our `TripletData` is necessary. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0020-00.png)


**----- Start of picture text -----**<br>
0.35 0.35 TripletCLIP (CC12M)<br>LaCLIP (CC12M)<br>0.30 0.30<br>0.25 0.25<br>0.20 0.20<br>0.15 0.15<br>0.10 0.10<br>0.05 0.05<br>0.00 0.00<br>0.0 0.2 0.4 0.6 0.8 0.2 0.4 0.6 0.8 1.0<br>Cosine Similarity Cosine Similarity<br>Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 5: Positive _vs._ Negative modality-specific pair-based similarity distribution of baseline LaCLIP and `TripletCLIP` . The left plot is the vision embedding similarities between positive and negative images. The right plot is the text embedding similarities between positive and negative captions. 

Table 17: **Question Generation.** Examples of LLM-generated existence-related questions from captions to evaluate the generated images. 

|captions to evaluate thegenerated images.||
|---|---|
|**Captions**|**Questions**|
||Is the entity a sheet?|
||Is the sheet made of fabric?|
|A smooth, fat sheet of woven fabric (which looks like woven silk) is shown in closeup.|Is the fabric woven?|
||Does the fabric look like silk?|
||Is the fabric fat and smooth?|
||Is there a sunset?|
||Is the sea calm?|
|The sunset over a calm sea.|Is the sunset over the sea?|
||Is the sunset happening during the day?|
||Is the sea rough?|
||Did the businesswoman fnish second?|
||Did the businesswoman descend the podium?|
|the businesswoman fnished second and descended the podium of the runners-up.|Are there runners-up?|
||Is the podium for the second place?|
||Is the businesswoman a runner?|
||Is there a kid in the public place?|
||Is the kid near a scooter?|
|Kid standing still near a scooter in a public place.|Is the scooter in a public place?|
||Is the kid standing still?|
||Is there a public place in the caption?|
||Is the local resident driving?|
||Is it happening in the fall?|
|A local resident drives in the fall on a scenic drive, wearing a red scarf.|Is there a scenic drive?|
||Is the local resident wearing a red scarf?|
||Is the local resident walking?|



## **F** `TripletData` **Analysis** 

This section provides qualitative examples of the `TripletData` and discusses various data analyses. 

**Difficulty of the data.** We further add one more analysis to investigate how difficult our dataset is. First, we take state-of-the-art language-only (GTE [28]) and vision-only (DINO [4]) embedding models and pretrained CLIP ViT-B/32. Later, we measure the modality-specific similarity between positive and negative vision and language data pairs. Figure 4 shows that the similarity distribution of the pretrained CLIP model between positive and negative text pairs follows the distribution of the text-only GTE model. Interestingly, vision distribution is drastically different. DINO can distinguish the positive and negative pairs correctly with high confidence. However, despite the visuals being so different, the pretrained CLIP model struggles to distinguish the different images. This further highlights that `TripletData` is indeed challenging for the vision-language models, even the ones trained on large-scale datasets. 

20 

**Evaluations of Generated Images.** Even though T2I diffusion models are widely evaluated on various tasks. We perform additional evaluations to measure how accurately generated images follow the text prompt. To do this, taking inspiration from [36], we first use LLM to generate binary "yes/no" style questions from the given caption. As shown in Table 17, we create five questions per hard negative caption. Later, we utilized the ViLT model to answer visual questions. Upon this investigation, we find that SDXL-turbo archives on an average of 76% accuracy. In other words, the T2I model can correctly generate an image that follows around 3/4th of the text. Additionally, we hypothesize that using an improved T2I model or image editing models to generate “hard” negative examples can further improve composition reasoning. 

**Qualitative Examples.** In Figure 6, we provide additional qualitative examples of the contrastive positive and “hard” negative pairs from the `TripletData` . Additionally, in Figure 7, we illustrated several examples where the T2I model could not precisely generate images corresponding to the caption. However, we may notice that in most cases, it maintains some of the important aspects. Because of this, despite not being 100% accurate all the time, it can help `TripletCLIP` improve performance across the evaluation benchmarks. 

## **Hard negative caption only examples:** 

1. **Raw Caption:** dog looking out from a window . **Language Rewrite:** A dog looking through the window at his owner. **Negative Caption (NegCLIP):** 

   - (a) A window looking through the dog at his owner. 

   - (b) A dog looking through the window at his dog. 

   - (c) A dog screams through the window at his owner. 

   - `TripletData` **Negative Caption:** A cat observing its owner from the window. 

2. **Raw Caption:** person attends the premiere of film 

   - **Language Rewrite:** A person attends the premiere of film **Negative Caption (NegCLIP):** 

   - (a) A premiere attends the person of film 

   - (b) A person attends the festival of film 

   - (c) A person watches the premiere of film 

   - `TripletData` **Negative Caption:** A person waits in line for film tickets. 

3. **Raw Caption:** white crocus spring flowers in the forest. 

   - **Language Rewrite:** white crocus flowers in the forest **Negative Caption (NegCLIP):** 

   - (a) white crocus flowers in the sky 

   - `TripletData` **Negative Caption:** red orchid flowers in the meadow. 

4. **Raw Caption:** flag with industry in the background 

   - **Language Rewrite:** A flag is holding in the background an industrial site. **Negative Caption (NegCLIP):** 

   - (a) A background is holding in the flag an industrial site. 

   - (b) A flag is holding in the background an earthquake site. 

   - (c) A drone is holding in the background an industrial site. 

   - (d) A flag is visible in the background an industrial site. 

   - `TripletData` **Negative Caption:** A flag is flapping in the foreground of a pastoral scene. 

5. **Raw Caption:** portrait of businessman with cardboard on his head carrying a briefcase and using an umbrella while standing by. **Language Rewrite:** A portrait of a businessman standing by with a briefcase and cardboard on his head carrying an umbrella while looking at a blue sky and parked cars on the street **Negative Caption (NegCLIP):** 

   - (a) A businessman of a portrait standing by with a briefcase and cardboard on his head carrying an umbrella while looking at a blue sky and parked cars on the street 

   - (b) A portrait of a businessman standing by with a briefcase and cardboard on his head carrying an umbrella while looking at a grey sky and parked cars on the street 

   - (c) A portrait of a businessman standing by with a briefcase and cardboard on his head carrying an umbrella while looking at a blue truck and parked cars on the street 

   - (d) A portrait of a businessman standing by with a briefcase and cardboard on his head carrying an umbrella while pointing at a blue sky and parked cars on the street 

21 

`TripletData` **Negative Caption:** A portrait of a businesswoman seated on a bench with a tote bag and newspaper on her lap holding an umbrella while looking at a red sunset over row houses. 

6. **Raw Caption:** 158834 is the portion of the bound train . **Language Rewrite:** Huge locomotives sit on the tracks in front of a building. **Negative Caption (NegCLIP):** 

   - (a) Huge tracks sit on the locomotives in front of a building. 

   - (b) Three locomotives sit on the tracks in front of a building. 

   - (c) Huge locomotives sit on the tracks in anticipation of a building. 

   - (d) Huge locomotives mounted on the tracks in front of a building. 

   - `TripletData` **Negative Caption:** Huge locomotives sit on the tracks in front of a bridge. 

7. **Raw Caption:** biological subfamily eating fish on a seaweed covered shore **Language Rewrite:** Two blue whales are eating salmon on a beach surrounded by seaweed **Negative Caption (NegCLIP):** 

   - (a) Two blue salmon are eating whales on a beach surrounded by seaweed 

   - (b) Two stranded whales are eating salmon on a beach surrounded by seaweed 

   - (c) Two blue whales are eating salmon on a farm surrounded by seaweed 

   - (d) Two blue whales are eating salmon on a beach covered by seaweed 

   - `TripletData` **Negative Caption:** Two blue whales are feeding on herring in a bay surrounded by kelp. 

8. **Raw Caption:** image of an original oil painting on canvas **Language Rewrite:** A young lady holding a painting to your face so you can see the detail of the painting 

## **Negative Caption (NegCLIP):** 

- (a) A young painting holding a lady to your face so you can see the detail of the lady 

- (b) A bearded lady holding a painting to your face so you can see the detail of the painting (c) A young lady holding a pencil to your face so you can see the detail of the painting (d) A young lady holding a painting to your face so you can enjoy the detail of the painting 

- `TripletData` **Negative Caption:** A young lady holding a painting away from her face to show its beauty to the audience. 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0023-00.png)


**----- Start of picture text -----**<br>
“photo of an interior of a  interior of a big “photo of an  “A bench by the sea is  “A bench by the lake is  thoroughbred “a  “a siamese cat, at six months<br>big white  red house with  shown on a shown on a  bengal cat, as a  old, has<br>house with<br>narrow windows  empty beach” sandy beach.” year old one, is  impressive<br>many windows that was the  that was the  a very powerful  strength for its<br>stage of the  animal” size.”<br>stage of the<br>filming of a<br>filming of a tv  movie.”<br>series.”<br>“A man's tattoo  “A woman's<br>covers much of  tattoo covers<br>his chest with  much of her  “add a  “add a vintage<br>the positive  back with the  modern finish  touch to your<br>and negative  positive and  to your look  appearance with<br>“lights, streets,  “lights,  space playing  negative space  with these  these charming<br>teeming,” squares,  an important  playing an  chic courts.” sneakers.”<br>part.” important<br>quiet,.”<br>part.”<br>“A simple  “A detailed  “Pigeons are  “Sparrows are  “A bedroom  “A bedroom<br>paper heart  paper flower  sitting in a  sitting on a lawn.” with a king bed  with a queen<br>pinned on a  pinned on a  park” and a bedside  bed and a<br>pink  white  table” chest of<br>background” background.” drawers.”<br>“A gold  “A silver  “The knife sits  “The knife sits on  “<PERSON>  “<PERSON><br>necklace” necklace with a  on a black  a white  sits for a  stands for a<br>different  background” background.” portrait by an  portrait by an<br>design.” artist” artist.”<br>**----- End of picture text -----**<br>


Figure 6: Qualitative examples of positive and hard negative image-text pairs from `TripletData` . In each block, left image-text pairs are positive images from CC3M, and right pairs are corresponding negatives from `TripletData` . 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives_images/nips_2024_TripletCLIP_Improving_Compositional_Reasoning_of_CLIP_via_Synthetic_Vision-Language_Negatives.pdf-0024-00.png)


**----- Start of picture text -----**<br>
“A small car at the side<br>“someone entered a<br>of the road where a<br>building looking cool in a  traditional horse-drawn<br>black leather jacket as<br>she walked.” carriage is driven by a<br>woman.”<br>“Standing next to a red  “A woman is using a ruler<br>ocean to measure the distance<br>, two children gaze<br>up at the half moon with  between two points<br>a promise to one day  outside of a bus.”<br>swim across the seas”<br>“A decorative Easter egg  “An hourglass in the<br>is balanced precariously<br>shape of a rectangular<br>on a fluffy bunny's<br>head.” pool.”<br>**----- End of picture text -----**<br>


Figure 7: Examples of T2I failures. 

24 

