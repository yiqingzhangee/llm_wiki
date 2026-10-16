# **Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation** 

**Yun Xing**[1] **Jian Kang**[1] **Aoran Xiao**[1] **Jiahao Nie**[1] **Ling Shao**[2] **Shijian Lu**[1] _[∗]_ 1 Nanyang Technological University 2 UCAS-Terminus AI Lab, UCAS, China 

## **Abstract** 

Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from language supervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from clear _semantic gaps_ between visual and textual modality: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such _semantic gap_ , we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a _concept archive_ that maintains potential visually-matched concepts with our proposed _vision-driven expansion_ and _text-to-vision-guided ranking_ . Relevant concepts can thus be identified via _cluster-guided sampling_ and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zeroshot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of bridging _semantic gap_ in pretraining data. Code is available at `https://github.com/xing0047/rewrite` . 

## **1 Introduction** 

Vision-Language Pre-training [34, 21, 1, 10], which aims to learn visual representations directly from natural language supervision, has endowed existing recognition systems with superior generality and open-vocabulary understanding capability. As a representative, CLIP [34] performs contrastive language-image pre-training on 400M web-crawled image-text pairs, whereby the learnt models may effortlessly transfer to a wide spectrum of classification tasks in a zero-shot manner. Motivated by the breakthrough, recent studies [43, 35, 44] extend the supervision paradigm to semantic segmentation, enabling spatial localization of textual queries in images and pixel grouping with solely supervision from image-text pairs. Distinct from conventional semantic segmentation, the language-supervised paradigm obviates the need for costly manual pixel-level annotation and enables million-level pretraining scale with much less effort. 

Despite the progresses [43, 35] in language-supervised semantic segmentation, the pre-training stage still suffers heavily from clear _semantic gap_ between visual and textual modality. In image-text pairs used for pre-training, it is ubiquitous that visual concepts appeared in images are missing in the corresponding textual captions. This happens largely because captions merely describe salient concepts that are worthy of mention [16, 25], while naturally forgo full semantic coverage of images 

> _∗_ Corresponding Author. 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
(a) Semantic Gap (c) Overview of Concept Curation<br>Image Concept Archive<br>Image Caption alpacas horse<br>alpacaspersongrasssky alpacas Another one of my male alpacas. vision-driven expansion persongrass llamacamelfield manlawnsky<br>Vision<br>(b) Semantic Bias Encoder Encoding<br>Retrieved Captions Group Token  text-to-vision-<br>guided ranking<br>image-to-text A gardener  Text Token<br>retrieval who is at<br>Image semantic bias Standing withllamas.work. EncoderText Supervision Signal  Baseline Ours Concept Archive (Ordered) alpacas<br>llama<br>Person  person<br>alpacasgrass llamagardener pictured while travelling. Prompted Texts alpacas grassmansky<br>person person … Prompt grassman cluster-guided sampling lawnfield<br>sky alpacas sky camelhorse<br>**----- End of picture text -----**<br>


Figure 1: Cross-modal _semantic gap_ is prevalent in web-crawled image-text pairs. As in (a), the caption text often captures certain salient visual concepts only in the paired image but misses many others (i.e., ‘ _person_ ’, ‘ _grass_ ’, and ‘ _sky_ ’) that are also useful in image-text modeling. Leveraging CLIP [34], more useful visual concepts could be captured via image-to-text retrieval, but the retrieved captions usually suffer from the _semantic bias_ as in (b) (i.e., ‘ _person_ ’ recovered but ‘ _grass_ ’ and ‘ _sky_ ’ still missing). Our proposed Concept Curation (CoCu) bridges the cross-modal _semantic gap_ effectively by _vision-driven expansion_ , _text-to-vision-guided ranking_ and _cluster-guided sampling_ while avoiding the negative effect by _semantic bias_ , as illustrated in (c). Best viewed in color. 

(Fig. 1 (a)). Under the presence of clear cross-modal _semantic gap_ in image-text pairs, the pre-training stage of language-supervised segmentation is found to be harder to converge, leading to inferior zero-shot performance on downstream tasks (more details are elaborated in Section 4.3). 

This work explores to bridge _semantic gaps_ in language-supervised semantic segmentation. For each image in the pre-training data, the goal is to recover the missing visual concepts in its paired caption for more comprehensive image-text modeling. With the rich vision-language correlations in off-the-shelf foundation models such as CLIP [34], a straight solution is to retrieve the missing concepts from the text captions of pre-training data. However, such retrieved captions suffer from the _semantic bias_ illustrated in Fig. 1 (b) (e.g., “person” recovered but “grass” and “sky” still missing). The root cause lies with the original text captions in foundation model pre-training, which only capture salient concepts appeared in the paired images. Hence, the retrieved captions and concepts still suffer from clear cross-modal _semantic gap_ . 

We propose Concept Curation (CoCu), a novel pipeline that side-steps the negative effect by _semantic bias_[2] while exploiting vision-language foundation models for semantic segmentation. CoCu consists of three sequential stages: 1) _vision-driven expansion_ that constructs _concept archive_ via crossimage retrieval; 2) _text-to-vision-guided ranking_ that scores the retrieved concepts according to their assigned relevancies; and 3) _cluster-guided sampling_ that exploits semantic diversity beyond the relevancy scores for concept ranking (Fig. 1 (c)). We perform pre-training from scratch on the segmentation backbone of [43] and evaluate zero-shot transfer over 8 widely adopted segmentation benchmarks. The experiments show that the proposed CoCu improves the baseline as well as the state-of-the-art consistently by large margins, indicating the necessity of closing the _semantic gap_ and the effectiveness of our designs in concept archiving and concept ranking. 

In summary, the contributions of this work are three-fold. _First_ , we identify the issue of _semantic gap_ in language-supervised semantic segmentation, and demonstrate the effectiveness of mining more relevant visual concepts for segmentation model pre-training. _Second_ , we design Concept Curation (CoCu), a novel pipeline that constructs _concept archives_ to expand and identify relevant 

2 to clarify, we refer to _semantic gap_ as a problem in web-crawled image-text pairs and _semantic bias_ as an issue in pre-trained vision-language model. 

2 

visual concepts from pre-training data, mitigating the _semantic gap_ between textual and visual modalities effectly. _Third_ , extensive experiments show that the proposed pipeline achieves superior zero-shot transfer performance and outperforms the state-of-the-art across 8 segmentation benchmarks consistently by large margins. 

## **2 Related Works** 

**Semantic Segmentation.** Partitioning an image into semantic regions, also known as semantic segmentation, has been widely studied due to myriad real-world applications such as video surveillance and autonomous driving. It has been explored along different directions, e.g., by designing different network architectures [29, 9, 11, 42], constructing benchmarks of diverse scenes and categories [12, 49, 5, 17], etc. However, collecting per-category dense annotations for supervised training is notoriously labor-intensive, which impedes the upscaling of semantic vocabulary greatly. Different from conventional semantic segmentation, segmentation from language supervision [43, 28, 7, 32, 35, 44] relieves the burden of mask annotation by leveraging image-text pairs available on the Internet. Beyond that, it can handle arbitrary new semantics thanks to the language supervision paradigm, making it feasible to learn generalizable segmentation models. 

**Vision-Language Pre-training.** Recently, Vision-Language Pre-training has become a predominant trend by learning visual representations from natural language supervision [34, 21, 24, 47, 1, 31, 10, **?** ]. By matching billion-scale image-text pairs via contrast, the learnt representations can be seamlessly transferred to various downstream classification tasks in a zero-shot manner. As a representative, CLIP [34] can match the performance of supervised baselines on ImageNet [13], meanwhile obtain competitive performance over plethora of downstream tasks without accessing any target data. The same learning paradigm has recently been explored for the task of semantic segmentation by hierarchical grouping [43], supervision mining [35, 44, 7, 32], etc. Nevertheless, state-of-the-art language-supervised segmentation is held back by the cross-modal _semantic gap_ between textual and visual pre-training data. Instead of relaxing the strict one-to-one correspondence in vanilla contrastive learning [35], we mitigate the _semantic gap_ by automated curation of relevant visual concepts through concept expanding and concept ranking. 

**Open-Vocabulary Semantic Segmentation.** Open-Vocabulary Semantic Segmentation has been studied extensively and most existing work can be broadly grouped into three categories. **Mix Supervision:** the first category follows a zero-shot manner [4, 23] which aims to segment new classes by learning from densely annotated seen classes in the pre-training data. Recently, several studies [18, 22, 46, 19, 26, 51, 45, 48] introduce language supervision to enhance the generality of the learnt zero-shot models. These approaches require no data annotation from new classes, but still rely on dense annotations from seen classes during the pre-training stage. **No Supervision** : the second category follows a training-free approach [50, 38, 37] which explores the segmentation potential of frozen vision-language models (VLMs) [50, 38, 37] to predict segmentation masks. However, most VLMs are trained with image-level supervision which restricts their capability on pixel/region-level predictions in semantic segmentation. **Language Supervision:** the third category follows a pure-language-supervision paradigm [43, 28, 32, 35, 7, 44] which aims to learn pixel grouping from solely image-text pairs. Our work follows the third approach. Different from existing studies, we identify the _semantic gap_ in pre-training image and text data and design concept curation that mitigates the _semantic gap_ with clearly improved semantic segmentation performance, more details to be described in the ensuing subsections. 

## **3 Methodology** 

With clear _semantic gaps_ between visual and textual concepts in pre-training data as illustrated in Fig. 1 (a), one naïve solution is to employ given image as query to retrieve related captions and derive missing textual concepts as described in Sec. 3.2. However, the naïve solution suffers from clear _semantic bias_ as most VLM-retrieved captions contain salient concepts only as illustrated in Fig. 1 (b). We thus further design _vision-guided expansion_ , _text-to-image-guided ranking_ and _cluster-guided sampling_ for better mitigation of the _semantic gap_ as presented in Fig. 1 (c) and Sec. 3.3. 

3 

## **3.1 Revisiting GroupViT** 

**Segmentation Architecture.** We use GroupViT [43] as the segmentation backbone for pre-training. Assume a batch of image-text pairs _{_ ( _x[I] , x[T]_ ) _}[B] i_ =1[, where] _[ x][I]_[amd] _[ x][T]_[denotes an image and its paired] caption, respectively. For the vision flow, a grouping-aware transformer _Fs[I]_[encodes image] _[ x][I]_[as] G segment tokens _zseg[I]_ = _{zseg[I] g , g_ = 1 _, ..., G} ∈_ R _[G][×][d]_ , where each segment token _zseg[I] g ∈_ R _[d]_ encodes an arbitrary-shaped region in image _x[I]_ . 

**Image-Caption Contrastive Loss.** To perform pre-training, the segment tokens _Zseg[I]_ are merged via average pooling, producing a global representation _zseg[I] ∈_ R _[d]_ that captures all the visual concepts appeared in image _x[I]_ . Meanwhile, the paired caption _x[T]_ is encoded to _z[T] ∈_ R _[d]_ by a text encoder _F[T]_ . The visual embedding _zseg[I]_ and textual embedding _z[T]_ are mapped to the same space by separate linear projectors. The segmentator is then learnt from language supervision by the standard contrastive objective InfoNCE [33], which is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0004-03.png)


where _τ_ is a learnable parameter initialized with 0.07 [34] and the _z[I] ·z[T]_ computes the cross-modal cosine similarity. 

**Multi-Label Loss** . Beyond learning segmentation from raw caption _x[T]_ , GroupViT [43] further introduces _L_ extra text labels _{x[T][l] , l_ = 1 _, ..., L}_ by prompting the extracted concepts _{cl, l_ = 1 _, ..., L}_ with handcrafted templates [34] (e.g., “a photo of a {concept}”). The _L_ text labels are fed to the same text encoder _F[T]_ to obtain textual representations of _{z[T][l] , l_ = 1 _, ..., L}_ . The language supervision by multi-label loss is thus defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0004-06.png)


The overall training objective of learning segmentation from language supervision in [43] is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0004-08.png)


**Discussion.** We use the exact same training objective as in GroupViT [43] to learn segmentation from language supervision. For each pre-training image _x[I]_ , the _multi-label loss_ enhances the contrastive learning [43] with _L_ extra positive pairs and _L_ ( _B −_ 1) extra negative pairs ( _B_ denotes batch size used in pre-training). However, we highlight that the simple concept prompting does not expand the textual concepts much. The cross-modal _semantic gap_ still exists and circulates in pre-training, which holds back the training convergence and degrades the zero-shot transfer. 

## **3.2 Naïve Solution** 

**Caption Curation.** The web-crawled image-text pairs are often noisy with imprecise and even irrelevant text descriptions [41]. In addition, many visual concepts (especially those inconspicuous in the background) in images are often missing in the corresponding text descriptions. Both factors lead to clear _semantic gaps_ between web-crawled images and texts. With super-rich image-text correlations in pre-trained VLMs such as CLIP [34], a straight solution, which we term by _caption curation_ , is to apply _x[I]_ as query to retrieve _L_ extra captions _{x[T][l] , l_ = 1 _, ..., L}_ from pre-training data. The _semantic gaps_ between visual and textual modality could thus be mitigated by identifying relevant concepts from the retrieved captions. 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Image-to-Image Retrieval<br>(within ALL DATA)<br>Concept Archive<br>Image-Text  CLIP Image<br>Pair Encoder alpaca I found you an  ! ;)  person.Camel the blue  under  sky . Personllama  hugs the . A white looking around. camel  is  … grassalpacascamel horsesky<br>field man<br>lawn<br>Another one of my  person llama<br>male  alpacas . Alpaca  resting  A  man  who  A  horse  grazes  Alpaca  in a  field<br>in the  grass. rides a  mule .  on the  lawn . during the day.<br>Text-to-Image Retrieval<br>Prompted Texts CLIP TextEncoder Score<br>a photo of  … camel person<br>person<br>Another one of  I found you an  Person  hugs the  A  man  who  grass<br>my male  alpacas . alpaca  ! ;)  person. llama . rides a  mule .  horse<br>a photo of  …<br>camel Concept<br>Camel  under  A white  camel  is  Alpaca  resting Another one of my<br>the blue  sky . looking around. in the  grass. male  alpacas . Concept Archive<br>(ordered)<br>alpacas<br>a photo of  … llama<br>grass<br>Person  hugs the  Another one of  Alpaca  resting  A  horse  grazes  person<br>llama . my male  alpacas . in the  grass. on the  lawn . grass<br>man<br>sky<br>a photo of  … lawn<br>horse<br>field<br>… … … A on the  horse  grazes  lawn . rides a A  manmule  who .  Personllama  hugs the . alpaca I found you an  ! ;)  person. camelhorse<br>…<br>…<br>**----- End of picture text -----**<br>


Figure 2: Illustration of _vision-driven expansion_ (above) and _text-to-image-guided ranking_ (below) in CoCu. To compensate for missing semantics, _vision-driven expansion_ establishes an archive of potential matched concepts through image-to-image retrieval, while _text-to-vision-guided ranking_ scores retrieved concepts based on assigned relevancy. The textual concepts can later be identified in pre-training by sampling. In the figure, images with a blue border □ are retrieved via expanded concepts (marked as blue) using their paired captions, while images with a red border □ represent images for curation (as anchor). Best viewed in color. 

**Semantic Bias.** Though _caption curation_ expands visual concepts effectively, retrieved captions often suffer from clear _semantic bias_ : VLMs tend to retrieve salient concepts but miss many inconspicuous ones that are also useful in image description. Consequently, visual concepts _C[I]_ = _{cl, l_ = 1 _, ..., M[I] }_ appeared in an image _x[I]_ are usually clearly more than textual concepts _C[T]_ = _{cl, l_ = 1 _, ..., M[T] }_ extracted from _{x[T] , x[T]_[1] _, ..., x[T][L] }_ (i.e., _M[I] > M[T]_ ). The root cause of the _semantic bias_ lies with the loose correlation between the visual and textual pre-training data of VLMs, where most captions just capture partial visual concepts appeared in the paired images [25]. The _semantic bias_ thus impedes convergence and effectiveness of language-supervised training without language supervision available for those non-described image regions. 

## **3.3 Concept Curation** 

To bridge semantic gaps in image-text pairs, we propose **Concept Curation (CoCu)** to rewrite caption semantics with the help of a pre-trained vision-language model. Consequently, **CoCu** finds more concept candidates that are aligned to images and compensate for missing semantics in captions. In pre-training, a multi-modal segmentor matches images and visual-enriched captions by contrastive objectives mentioned in Sec. 3.1, encoding better vision-language alignment in its representations. Details of CoCu are described as below. 

**Vision-driven Expansion.** For an image-text pair ( _x[I] , x[T]_ ), the goal of the vision-driven expansion is to build an archive of textual concepts _C[T]_ = _{cm, m_ = 1 _, ..., M }_ that are potentially matched with _x[I]_ as illustrated in Fig. 2. Instead of acquiring _C[T]_ via direct text retrieval as in _caption curation_ , we resort to cross-image retrieval to achieve the expansion. Concretely, _N_ image-text pairs _P_ = _{_ ( _x[I] i[,][x][T] i_[)] _[,][i]_[=][1] _[, ..., N][}]_[ are automatically selected from the pre-training data, where] _{x[I] i[,][i]_[=][1] _[, ..., N][}]_[are N][captioned images whose][visual features match][the best with][that of] _[x][I]_ (all encoded by CLIP). _C[T]_ can thus be derived by extracting textual concepts from the captions 

5 

_{x[T] i[,][i]_[ = 1] _[, ..., N][}]_[ of the] _[ N]_[best matched images.][Compared with the] _[ caption curation]_[ that retrieves] _L_ descriptions _{x[T] i[,][i]_[ = 1] _[, ..., L][}]_[,] _[ vision-driven expansion]_[ exploits all visual information in images] instead of biased caption texts (mostly describes salient visual concepts only), which helps restore more relevant textual concepts. In addition, it builds an extra image set _{x[I] i[,][i]_[ = 1] _[, ..., N][}]_[ that plays] a pivotal role in the upcoming stages. 

**Text-to-Vision-Guided Ranking.** For each textual concept _cm_ in the concept archive _C[T]_ , we assign a score _scm_ to represent its relevancy to image _x[I]_ . A naïve solution to _pm_ is to compute the cosine similarity between the visual representation of _x[I]_ and textual representation of _tm_ encoded by CLIP [34], which is simply defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0006-02.png)


where _tm_ is derived from visual concept _cm_ via prompt engineering [34, 43]. Note direct text retrieval could easily get biased towards salient visual concepts here, imposing low relevance scores for other text concepts in _concept archive_ . Beyond _f_ ( _x[I] , tm_ ), we also design a non-biased metric to capture the relevancy between image _x[I]_ and concept _cm_ . Specifically, for the _N_ retrieved image-text pairs _P_ , we first extract a subset _PG_ (blue box/text in Fig. 2 (below)), whose caption of each image-text pair contains the textual concept _cm_ . The non-biased metric is thus defined with _x[I]_ (red box) and image-text pairs _PG_ = _{_ ( _x[I] i[, x][T] i_[)] _[,][i]_[ = 1] _[, ..., N][ ′][}]_[ as follows:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0006-04.png)


The given term functions as follows: 1) lower relevancy between image _x[I]_ and irrelevant concepts (e.g., ‘horse’ in Fig. 2); 2) enhance relevancy between _x[I]_ and inconspicuous concepts (e.g., ‘grass’). Instead of computing relevancies of _x[I]_ to all textual concepts in a single run, we consider one _cm_ at a time and measure its relevancy by comparing _f_ ( _x[I] , tm_ ) with _{f_ ( _x[I] i[,][t][m]_[)] _[,][i]_[ = 1] _[, ..., N][ ′][}]_[.][The idea] behind this is simple: 1) comparing the responses of images _{x[I] , x[I]_ 1 _[, ..., x][I] N[′][}]_[ to the same textual] concept _cm_ ; 2) high relevancy is given if response of _x[I]_ to textual concept _cm_ is comparably high and vice versa. Take the visual concept ‘grass’ in Fig. 2 (below) as an example. The _tm_ (‘a photo of grass’) causes _x[I]_ to rank considerably higher than image captioned with _cm_ (‘grass’). In this case, we should be fairly confident to pass the concept _cm_ to _x[I]_ . In conclusion, the relevancy score is simply defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0006-06.png)


we perform ranking according to computed relevancies _{scm , m_ = 1 _, ..., M }_ , which represents chances identified by later sampling. 

**Cluster-guided Sampling.** The pre-training can thus be empowered by including the expanded and ranked textual concepts which as selected by sampling _L_ textual concepts according to their computed relevancies as in [43]. However, selection with relevancy alone is often short of semantic diversity in the selected text concepts. Instead of directly selecting _L_ concepts from the ranked archive _C[T]_ , we partition _C[T]_ into _L_ semantic clusters based on their textual representations and sample one textual concept from each semantic cluster. The _cluster-guided sampling_ has two clear benefits: 1) it includes more diverse semantics in each single training step; 2) it keeps good consistency with the expression of visual concepts, more details to be discussed in Sec. 4.4 and appendix. 

## **4 Experiments** 

## **4.1 Experimental Setup** 

**Training Detail** . We follow the prior study [43] and conduct pre-training on three publicly available image-text datasets: CC3M (C3) [36], CC12M (C12) [8], YFCC14M (Y14) [39]. For fair comparison, we use the same GroupViT [43] as the visual encoder, which is built upon ViT-S backbone [14, 40] and learnt from scratch. We set the global batch size for contrastive learning as 1,024 and use 4 Tesla V100 GPUs to carry out pre-training for all experiments. Consistent with [43], we set the initial learning rate to 0.0016. The pre-training undergoes 30 epochs, with a linear warmup for the first 2 epochs and a cosine schedule for the remaining epochs. _L_ is set to 3. In our ablations and discussions, we report the performance of models pre-trained on CC3M. 

6 

**Implementation of curation** .The curation pipeline utilizes clip-retrieval [3], a utility that enables efficient computation of CLIP embeddings and fast indexing for retrieval. We employ the CLIP ViT-B/16 [34] model for image/text inference and concept curation. For efficient semantic searching, we build indexing systems using autofaiss[3] . It is worth mentioning that alternative systems can also be used for implementing the curation process. 

**Evaluation** . We benchmark zero-shot transfer performance of CoCu on the validation splits of eight different datasets that cover a myriad of scenes and category sets, including Pascal VOC [15], Pascal Context [30], COCO [27], ImageNet-S-50, ImageNet-S-300 [17], COCO Stuff [5], Cityscapes [12], and ADE20K [49]. For the first five datasets, we follow [43] and evaluate foreground classes by thresholding the similarity between visual and textual embeddings. For other datasets, we evaluate both foreground and background classes. More details are given in the appendix. 

## **4.2 Comparison with the state-of-the-art** 

We first benchmark CoCu with state-of-the-art zero-shot methods [50, 43] and evaluate its effectiveness. Specifically, we follow prior work [43] and pre-train CoCu over the combination of C3, C12, and Y14 datasets. Tab. 1 reports zero-shot segmentation results. Besides GroupViT as the baseline method, we also compare the advanced MaskCLIP [50]), which directly leverages the frozen CLIP model for segmentation prediction without pre-training. In addition, for a comprehensive comparison, we list the performance of other advanced methods including 1) fully-supervised method [40] that provides Oracle’s performance, 2) self-supervised methods [20, 6] that pre-train models with unlabeled data and fine-tuning models over segmentation datasets. Detailed implementations of the comparing methods could be found in the appendix. 

As shown in Tab. 1, MaskCLIP achieves limited segmentation performance, primarily due to CLIP being trained with image-level supervision and thus falling short in precise pixel-level predictions. GroupViT achieves better performance than MaskCLIP, but still limited by insufficient supervision from language side in pre-training. On the contrary, our CoCu achieves the best segmentation performance over all eight benchmarks, surpassing GroupViT by large margins on average. This indicates the necessity of bridging _semantic gaps_ in language-supervised semantic segmentation and the effectiveness of our design. 

Table 1: **Performance of different zero-shot methods for semantic segmentation.** Abbreviations of benchmarks, from left to right: Pascal VOC [15], Pascal Context [30], Microsoft COCO [5], ImageNet-S [17], Cityscapes [12], and ADE20K [49]. BS denotes pre-training batch size, while LC represents local consistency [2] in mask prediction. _†_ denotes our re-implementation. CoCu consistently achieves the best performance across all benchmarks. 

|**Method**<br>**Pretrain Data**<br>**Supervision**<br>**LC**<br>**BS**<br>**Backbone**|**PVOC**<br>**PCON**<br>**COCO**<br>**IN50**<br>**IN300**<br>**CITY**<br>**ADE**<br>**STUF**|**AVG**|
|---|---|---|
|DeiT [40]<br>IN-1K<br>full<br>-<br>ViT-S|53.0<br>35.9<br>-<br>-<br>-<br>-<br>-<br>-|-|
|MoCo [20]<br>IN-1K<br>self<br>-<br>-<br>DINO [6]<br>IN-1K<br>self<br>-<br>-<br>MoCo [20]<br>C12,Y14<br>self<br>-<br>-<br>DINO [6]<br>C12,Y14<br>self<br>-<br>-|34.3<br>21.3<br>-<br>-<br>-<br>-<br>-<br>-<br>39.1<br>20.4<br>-<br>-<br>-<br>-<br>-<br>-<br>36.1<br>23.0<br>-<br>-<br>-<br>-<br>-<br>-<br>37.6<br>22.8<br>-<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-|
|MaskCLIP [50]<br>-<br>N.A.<br>✓<br>-<br>ResNet-50<br>MaskCLIP [50]<br>-<br>N.A.<br>✓<br>-<br>ViT-B/16|41.5<br>18.5<br>10.5<br>13.8<br>7.9<br>18.8<br>8.3<br>10.2<br>49.5<br>21.7<br>13.6<br>25.9<br>11.7<br>19.8<br>9.5<br>12.5|15.0<br>20.5|
|GroupViT [43]<br>C3,C12,Y14<br>text<br>4,096<br>ViT-S|**52.4**<br>22.3<br>**24.3**<br>44.3<br>23.5<br>15.8<br>10.4<br>13.0|25.7|
|GroupViT_†_ [43]<br>C3,C12,Y14<br>text<br>1,024<br>ViT-S<br>CoCu (ours)<br>C3,C12,Y14<br>text<br>1,024<br>ViT-S|43.8<br>19.3<br>19.6<br>37.8<br>17.2<br>17.2<br>10.4<br>13.6<br>49.7<br>22.8<br>22.0<br>46.7<br>24.7<br>21.9<br>12.0<br>14.9|22.4<br>26.8|
|GroupViT_†_ [43]<br>C3,C12,Y14<br>text<br>✓<br>1,024<br>ViT-S<br>CoCu (ours)<br>C3,C12,Y14<br>text<br>✓<br>1,024<br>ViT-S|45.4<br>19.9<br>20.3<br>39.2<br>17.7<br>17.6<br>10.6<br>13.9<br>51.4<br>**23.6**<br>22.7<br>**48.8**<br>**25.5**<br>**22.1**<br>**12.3**<br>**15.2**|23.1<br>**27.7**|



We further evaluate the robustness of CoCu with different pre-train data. Spcifically, we pre-train GroupViT and CoCu over CC3M and CC12M, respectively. We also sub-sample half of image-text pairs from CC12M (denoted as C12 _[∗]_ ) for pre-training. Tab. 2 shows the experimental results. We can observe consistent yet significant performance gains on eight benchmarks. The improvement by bridging _semantic gap_ is thus robust and not affected by pre-training size. 

> 3https://github.com/criteo/autofaiss.git 

7 

Table 2: **Zero-shot semantic segmentation performance with different pre-training data.** CoCu consistently outperforms the baseline method GroupViT across all benchmarks, demonstrating its effectiveness in bridging _semantic gaps_ and achieving significant improvements. 

|effectiveness in|bridging_semantic gaps_and achieving signifcant improvements.||
|---|---|---|
|**Method**<br>**Pretrain**<br>**Data**|**PVOC (21)**<br>**PCON (60)**<br>**COCO (81)**<br>**IN50 (51)**<br>**IN300 (301)**<br>**CITY (19)**<br>**ADE (150)**<br>**STUF (171)**|**AVG**|
|GroupViT_†_ [43]<br>C3<br>CoCu<br>C3|15_._5<br>10_._4<br>6_._5<br>10_._2<br>2_._9<br>8_._1<br>4_._4<br>7_._7<br>30_._6 (+15_._1%)<br>13_._9 (+3_._5%)<br>10_._8 (+4_._3%)<br>19_._3 (+9_._1%)<br>7_._3 (+4_._4%)<br>8_._2 (+0_._1%)<br>6_._1 (+1_._7%)<br>8_._5 (+0_._8%)|8_._2<br>13_._1 (+4_._9%)|
|GroupViT_†_ [43]<br>C12_∗_<br>CoCu<br>C12_∗_|32_._9<br>13_._3<br>12_._9<br>27_._9<br>12_._4<br>10_._7<br>5_._6<br>8_._6<br>34_._1 (+1_._4%)<br>16_._4 (+3_._1%)<br>17_._0 (+4_._1%)<br>33_._0 (+5_._1%)<br>17_._3 (+4_._9%)<br>11_._8 (+1_._1%)<br>8_._1 (+2_._5%)<br>9_._5 (+0_._9%)|15_._5<br>18_._4 (+2_._9%)|
|GroupViT_†_ [43]<br>C3,C12_∗_<br>CoCu<br>C3,C12_∗_|36_._5<br>15_._9<br>16_._2<br>33_._5<br>14_._0<br>12_._4<br>7_._0<br>10_._5<br>38_._1 (+1_._6%)<br>19_._2 (+3_._3%)<br>20_._1 (+3_._9%)<br>35_._8 (+2_._3%)<br>18_._9 (+4_._9%)<br>14_._7 (+2_._3%)<br>9_._5 (+2_._5%)<br>11_._4 (+0_._9%)|18_._2<br>21_._0 (+2_._8%)|
|GroupViT_†_ [43]<br>C12<br>CoCu<br>C12|37_._5<br>18_._0<br>18_._3<br>35_._7<br>16_._6<br>13_._5<br>9_._1<br>13_._1<br>40_._9 (+3_._4%)<br>21_._2 (+3_._2%)<br>20_._3 (+2_._0%)<br>40_._0 (+4_._3%)<br>19_._4 (+2_._8%)<br>15_._0 (+1_._5%)<br>11_._1 (+2_._0%)<br>13_._6 (+0_._5%)|20_._2<br>22_._7 (+2_._5%)|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0008-02.png)


**----- Start of picture text -----**<br>
(a) training convergence (b) segmentation results of  grass<br>loss<br>16 GroupViT epoch 4 epoch 9 epoch 14 epoch 19<br>14 CoCu (ours)<br>12<br>10<br>9<br>6<br>training<br>0 3 6 9 12 15 18 hours Caption:  a red fox drinking water.<br>GroupViT<br>CoCu (ours)<br>**----- End of picture text -----**<br>


Figure 3: **CoCu enhances training convergence.** (a) The training loss curves of GroupViT and CoCu demonstrate that CoCu significantly accelerates pre-training convergence. (b) CoCu achieves superior binary segmentation results (second row) compared to GroupViT (first row) for the concept of "grass," which is missing in the caption, using an example image captioned as "a red fox drinking water." Best viewed in color. 

## **4.3 CoCu helps convergence** 

**Loss Curve.** In Figure 3 (a), we compare the pre-training loss curves of GroupViT and our proposed method CoCu. We can see that CoCu exhibits a notably faster convergence rate, primarily attributed to the inclusion of curated semantic concepts for each image, resulting in more effective contrastive learning. Additionally, CoCu achieves a lower minimum loss by extracting significantly richer language concepts from image data. This enriches the training process by incorporating more identified image regions and ultimately learning representations that better align with the training data. 

**Qualitative Comparison** . We also present qualitative results that demonstrate the effectiveness of CoCu. In Figure 3 (b), we show binary segmentation results of GroupViT (first row) and CoCu (second row) over an example image with the caption "a red fox drinking water." Our focus is on the concept of "grass," which is missing in the caption. We compare the visual responses of models trained using these two methods at different checkpoints. Both methods improve progressively during training. However, GroupViT fails to correctly localize the region of "grass" due to the lack of direct supervision from the language side. In contrast, CoCu bridges the semantic gap by accurately capturing and localizing "grass," encoding it in representations during pre-training. Consequently, it achieves significantly better segmentation results under zero-shot context. 

Figure 4 displays the activation maps of GroupViT and CoCu for different concepts as text inputs that do not appear in the corresponding captions. These maps further demonstrate the superiority of CoCu in language-supervised learning. In all presented images, GroupViT incorrectly activates corresponding regions based on the given text inputs (e.g., activating the "sky" region with a text input of "person" for the first image). In contrast, CoCu enables the segmentor to have the highest activations on visually relevant regions indicated by the text. This suggests that segmentors derived from our method have a better capability to discriminate various visual concepts. More convergence results can be found in the appendix. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation_images/nips_2023_Rewrite_Caption_Semantics_Bridging_Semantic_Gaps_for_Language-Supervised_Semantic_Segmentation.pdf-0009-00.png)


**----- Start of picture text -----**<br>
GroupViT CoCu (Ours) GroupViT CoCu (Ours) GroupViT CoCu (Ours)<br>high<br>low<br>Text Input:  person Text Input:  milk Text Input:  plate<br>Caption:  Alpacas Kisses is a new business that brings alpacas to you Caption:  Baked powdered sugar donuts whole wheat. Caption:  Cooked pasta on wooden table.<br>**----- End of picture text -----**<br>


Figure 4: **Visualization of activation heatmaps.** GroupViT fails to activate on corresponding visual regions for concepts not represented in captions, while CoCu exhibits significantly better localization. High activation is shown as red, and low activation is displayed as blue. Best viewed in color. 

Table 3: **Ablation study of CoCu.** We conduct an ablation study on each designed modules. Zero-shot transfer performance on semantic segmentation results are reported, averaged across eight evaluation datasets. “Naïve ranking" refers to solely using cosine similarity between visual and textual representations (encoded by CLIP) as concept-to-image relevancy. “Naïve sampling" denotes selecting textual concepts based solely on relevancy before pre-training. 

|Model|Model|Expansion|Expansion|Ranking|Ranking|Sampling<br>Average<br>mIoU(%)|Sampling<br>Average<br>mIoU(%)|
|---|---|---|---|---|---|---|---|
|||_lang-driven_<br>_vision-driven_||_naïve text-to-vision-guided_||_naïve cluster-guided_||
|||||||||
|Baseline [43]<br>#1<br>#2<br>#3<br>#4 (_Full_CoCu)||✓<br>✓<br>✓<br>✓||✓<br>✓<br>✓<br>✓||✓<br>✓<br>✓<br>✓|8_._2<br>9_._9 (1_._7 _↑_)<br>10_._3 (2_._1 _↑_)<br>12_._4 (4_._2 _↑_)<br>13_._1 (4_._9 _↑_)|



## **4.4 Analysis** 

**Ablation Study.** We further assess the effectiveness of each module in CoCu, which includes _vision-driven expansion_ , _text-to-image-guided ranking_ and _cluster-guided sampling_ . Specifically, we pre-train five models with the combination of these modules or their alternative strategies, namely: 1) Baseline model of GroupViT, which is pre-trained without involving concept curation. 2) Model #1, which utilizes language-driven expansion, naïve ranking, and naïve sampling (Caption Curation in Sec. 3.2). 3) Model #2, which replaces language-driven expansion with vision-driven expansion on top of Model #1. 4) Model #3, which incorporates text-to-image-guided ranking on top of Model #2. And 4) the full CoCu Model #4, which combines vision-driven expansion, text-to-image-guided ranking, and cluster-guided sampling in pre-training. We report the average segmentation performance of these models across the eight datasets used previously (as shown in Table 1 and Table 2). Detailed illustrations of implementations are provided in appendix. 

As Tab. 3 shows, the simplest strategy of _language-driven_ in model #1 improves average mIoU by 1 _._ 7%, which comes from stronger vision-language correlation in pre-training data enhanced by direct text retrieval. Next, replacing direct text retrieval with _vision-driven expansion_ in model #2 brings an additional performance boost, highlighting its significance in capturing unbiased semantics. Furthermore, incorporating _text-to-vision-guided ranking_ in Model #3 brings another noticeable performance gain, underscoring the importance of measuring concept-to-image relevancy. Finally, we upgrade the sampling strategy from the naïve one that solely relies on relevancy to _cluster-guided sampling_ , and build model #4 with the full CoCu, which provides more diverse semantic information in each pre-training step, ultimately leading to the best zero-shot transfer performance for semantic segmentation. 

**Zero-Shot Classification** . In addition to its application in zero-shot segmentation, CoCu can also be used to improve zero-shot classification. Following the previous study [43], we evaluate CoCu and compare it with GroupViT on the ImageNet-1K dataset [13]. As shown in Table 4, CoCu exhibits significant performance gains over GroupViT, demonstrating its superiority in bridging semantic gaps across tasks and achieving improved zero-shot classification results. 

Table 4: **Zero-shot classification on ImageNet1K** . Acc@1 and Acc@5 denote top-1 and top-5 accuracy, respectively. 

|Method<br>Pre-training<br>data|Method<br>Pre-training<br>data|Zero-shot|Zero-shot|
|---|---|---|---|
|||Acc@1(%)<br>Acc@5(%)||
|GroupViT<br>C12<br>CoCu (ours)<br>C12||34.9<br>63.3<br>**38.4**(4_._5 _↑_)<br>**68.6**(5_._3 _↑_)||
|||||
|GroupViT<br>CoCu (ours)|C3,C12,Y14<br>C3,C12,Y14|36.8<br>**43.0**(6_._2 _↑_)|66.8<br>**73.7**(6_._9 _↑_)|



9 

## **5 Conclusion** 

In this paper, we identify the issue of _semantic gap_ in language-supervised semantic segmentation and explore how to bridge _semantic gaps_ effectively. To achieve this, we design Concept Curation, a novel pipeline that resolves the issue by three consecutive stages: _vision-driven expansion_ , _text-to-visionguided ranking_ and _cluster-guided sampling_ . Extensive experiments demonstrate the superiority of our method for boosting language-supervised semantic segmentation across a bundle of pre-training sets and evaluation benchmarks. Looking ahead, we hope to extend the idea of concept curation to other computer vision tasks, including object detection and instance segmentation. 

## **Acknowledgments and Disclosure of Funding** 

This project is funded by the Ministry of Education Singapore, under the Tier-2 project scheme with a project number MOE-T2EP20220-0003. 

## **References** 

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _arXiv preprint arXiv:2204.14198_ , 2022. 

- [2] Nikita Araslanov and Stefan Roth. Single-stage semantic segmentation from image labels. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4253–4262, 2020. 

- [3] Romain Beaumont. Clip retrieval: Easily compute clip embeddings and build a clip retrieval system with them. GitHub, 2022. 

- [4] Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick Pérez. Zero-shot semantic segmentation. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- [5] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1209–1218, 2018. 

- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 9650–9660, 2021. 

- [7] Junbum Cha, Jonghwan Mun, and Byungseok Roh. Learning to generate text-grounded mask for openworld semantic segmentation from only image-text pairs. _arXiv preprint arXiv:2212.00785_ , 2022. 

- [8] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 3558–3568, 2021. 

- [9] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. _IEEE transactions on pattern analysis and machine intelligence_ , 40(4):834–848, 2017. 

- [10] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual languageimage model. _arXiv preprint arXiv:2209.06794_ , 2022. 

- [11] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. _Advances in Neural Information Processing Systems_ , 34:17864–17875, 2021. 

- [12] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3213–3223, 2016. 

- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pages 248–255. Ieee, 2009. 

- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_ , 2020. 

- [15] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. _International journal of computer vision_ , 88:303–308, 2009. 

10 

- [16] Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh K Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C Platt, et al. From captions to visual concepts and back. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 1473–1482, 2015. 

- [17] Shanghua Gao, Zhong-Yu Li, Ming-Hsuan Yang, Ming-Ming Cheng, Junwei Han, and Philip Torr. Large-scale unsupervised semantic segmentation. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022. 

- [18] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Open-vocabulary image segmentation. _arXiv preprint arXiv:2112.12143_ , 2021. 

- [19] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVI_ , pages 540–557. Springer, 2022. 

- [20] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9729–9738, 2020. 

- [21] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , pages 4904–4916. PMLR, 2021. 

- [22] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation. _arXiv preprint arXiv:2201.03546_ , 2022. 

- [23] Peike Li, Yunchao Wei, and Yi Yang. Consistent structural relation learning for zero-shot segmentation. _Advances in Neural Information Processing Systems_ , 33:10317–10327, 2020. 

- [24] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. _arXiv preprint arXiv:2110.05208_ , 2021. 

- [25] Yehao Li, Yingwei Pan, Ting Yao, and Tao Mei. Comprehending and ordering semantics for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17990–17999, 2022. 

- [26] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. _arXiv preprint arXiv:2210.04150_ , 2022. 

- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pages 740–755. Springer, 2014. 

- [28] Quande Liu, Youpeng Wen, Jianhua Han, Chunjing Xu, Hang Xu, and Xiaodan Liang. Open-world semantic segmentation via contrasting and clustering vision-language embedding. In _Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XX_ , pages 275–292. Springer, 2022. 

- [29] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3431–3440, 2015. 

- [30] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 891–898, 2014. 

- [31] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets languageimage pre-training. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVI_ , pages 529–544. Springer, 2022. 

- [32] Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang, Ashish Shah, Philip HS Torr, and Ser-Nam Lim. Open vocabulary semantic segmentation with patch aligned contrastive learning. _arXiv preprint arXiv:2212.04994_ , 2022. 

- [33] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 

- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pages 8748–8763. PMLR, 2021. 

11 

- [35] Pengzhen Ren, Changlin Li, Hang Xu, Yi Zhu, Guangrun Wang, Jianzhuang Liu, Xiaojun Chang, and Xiaodan Liang. Viewco: Discovering text-supervised segmentation masks via multi-view semantic consistency. _arXiv preprint arXiv:2302.10307_ , 2023. 

- [36] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2556–2565, 2018. 

- [37] Gyungin Shin, Weidi Xie, and Samuel Albanie. Namedmask: Distilling segmenters from complementary foundation models. _arXiv preprint arXiv:2209.11228_ , 2022. 

- [38] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Retrieve and co-segment for zero-shot transfer. _arXiv preprint arXiv:2206.07045_ , 2022. 

- [39] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. _Communications of the ACM_ , 59(2):64–73, 2016. 

- [40] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In _International conference on machine learning_ , pages 10347–10357. PMLR, 2021. 

- [41] Bichen Wu, Ruizhe Cheng, Peizhao Zhang, Tianren Gao, Joseph E. Gonzalez, and Peter Vajda. Data efficient language-supervised zero-shot recognition with optimal transport distillation. In _International Conference on Learning Representations_ , 2022. 

- [42] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. _Advances in Neural Information Processing Systems_ , 34:12077–12090, 2021. 

- [43] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 18134–18144, 2022. 

- [44] Jilan Xu, Junlin Hou, Yuejie Zhang, Rui Feng, Yi Wang, Yu Qiao, and Weidi Xie. Learning open-vocabulary semantic segmentation models from natural language supervision. _arXiv preprint arXiv:2301.09121_ , 2023. 

- [45] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for openvocabulary semantic segmentation. _arXiv preprint arXiv:2302.12242_ , 2023. 

- [46] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIX_ , pages 736–753. Springer, 2022. 

- [47] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: fine-grained interactive language-image pre-training. _arXiv preprint arXiv:2111.07783_ , 2021. 

- [48] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianfeng Gao, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. _arXiv preprint arXiv:2303.08131_ , 2023. 

- [49] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 633–641, 2017. 

- [50] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In _Computer Vision– ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVIII_ , pages 696–712. Springer, 2022. 

- [51] Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al. Generalized decoding for pixel, image, and language. _arXiv preprint arXiv:2212.11270_ , 2022. 

12 

