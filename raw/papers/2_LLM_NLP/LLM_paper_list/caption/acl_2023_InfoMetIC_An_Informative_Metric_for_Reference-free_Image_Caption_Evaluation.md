# **InfoMetIC: An Informative Metric for Reference-free Image Caption Evaluation** 

**Anwen Hu**[1] **, Shizhe Chen**[2] **, LiangZhang**[1] **, Qin Jin**[1] 1School of Information, Renmin University of China 2INRIA 

{anwenhu,zhangliang00,qjin}@ruc.edu.cn shizhe.chen@inria.fr 

## **Abstract** 

Automatic image captioning evaluation is critical for benchmarking and promoting advances in image captioning research. Existing metrics only provide a single score to measure caption qualities, which are less explainable and informative. Instead, we humans can easily identify the problems of captions in details, e.g., which words are inaccurate and which salient objects are not described, and then rate the caption quality. To support such informative feedback, we propose an **Info** rmative **Met** ric for Reference-free **I** mage **C** aption evaluation (InfoMetIC). Given an image and a caption, InfoMetIC is able to report incorrect words and unmentioned image regions at finegrained level, and also provide a text precision score, a vision recall score and an overall quality score at coarse-grained level. The coarse-grained score of InfoMetIC achieves significantly better correlation with human judgements than existing metrics on multiple benchmarks. We also construct a token-level evaluation dataset and demonstrate the effectiveness of InfoMetIC in fine-grained evaluation. Our code and datasets are publicly available at https://github.com/HAWLYQ/ InfoMetIC. 

## **1 Introduction** 

Image captioning aims to automatically generate natural language sentences to describe image contents. Recently, there are significant breakthroughs in image captioning such as attentionbased model architectures (Anderson et al., 2018; Pan et al., 2020; Hu et al., 2020, 2021) and visionand-language pretraining (VLP) (Zhou et al., 2020; Xia et al., 2021; Li et al., 2022; Xu et al., 2021, 2023). However, as groundtruth image descriptions are extremely diverse and subjective, evaluating the image captioning performance remains a considerable challenge. 

The most widely used image captioning metrics such as METEOR (Banerjee and Lavie, 2005), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0001-09.png)


**----- Start of picture text -----**<br>
References:<br>1. A brown and white dog is running.<br>2. A dog runs on the snow with a package.<br>3. There is a hat buried in the snow.<br>Candidate 1(C1):  A dog on the snow.<br>Candidate 1(C2):  A dog with a hat is on the snow.<br>Candidate 2(C3):  A dog with a bag is on the snow.<br>METEOR SPICE CLIP-S Ours: InfoMetIC (w/o ref)<br>(w/ ref) (w/ ref) (w/o ref) Text Precision Vision Recall Overall<br>C1 0.27 0.31 0.78 dog ✔ snow 5.71 ✔ ✔ 0.62 ✔ ✘ 6.33<br>4.78 1.27<br>C2 0.37 0.40 0.76 dog ✔ hat ✘ snow ✔ ✔ ✔ ✘ 6.05<br>C3 0.37 0.27 0.84 dog ✔ bag 6.45 ✔ snow ✔ ✔ 1.69 ✔ ✔ 8.14<br>**----- End of picture text -----**<br>


Figure 1: Comparison of existing metrics and our informative metric (InfoMetIC). ‘w/ ref’ and ‘w/o ref’ mean using references or not. 

CIDEr (Vedantam et al., 2015a) and SPICE (Anderson et al., 2016) utilize human-written descriptions of images as references and measure similarities between generated captions and references for evaluation. Such reference-based approaches suffer from two major limitations. Firstly, these metrics mainly evaluate caption quality by n-gram overlaps which fail to measure genuine semantic similarities. Secondly, references require time-consuming annotations and thus there are only a few annotated captions (typically 5) for each image. The limited number of references cannot fully capture image contents, resulting in incorrect penalties when generated captions describe correct novel things that are not mentioned in the references. 

To alleviate the above limitations, recent works are more focusing on reference-free metrics, which directly use images instead of reference captions in evaluation. Benefited from the success of VLP on large-scale web data, UMIC (Lee et al., 2021) and CLIP-S (Hessel et al., 2021) leverage VLP models UNITER (Chen et al., 2020) and CLIP (Radford et al., 2021) respectively to calculate relevance scores between generated captions and images. Although they have achieved promising correlations with human judgments, they can only produce an overall score as quality measurement. We humans 

instead tend to evaluate captions considering two aspects: 1) whether the caption correctly describes the image content (named _text precision_ ); and 2) whether the image content is comprehensively described in the caption (named _vision recall_ ). For example, as shown Figure 1, we can easily tell the “hat” in the second candidate is incorrect, and some salient contents such as “the bag” are not mentioned, and thus form our final evaluation to the caption. 

For the purpose of providing explainable and detailed feedbacks, we propose a **Info** rmative **Met** ric for Reference-free **I** mage **C** aption evaluation (InfoMetIC). It is built on top of pretrained VLP models to measure fine-grained cross-modal similarities. InfoMetIC is able to point out incorrect semantic words in the caption and unmentioned regions in the image. Based on fine-grained evaluation, it derives text precision and vision recall scores to measure captioning accuracy and completeness respectively. We take a summation of the two scores to rate overall quality of the caption. 

Our contributions in this work are three-fold: 

- We propose a reference-free informative image captioning metric InfoMetIC. It can provide both coarse-grained scores and detailed token-level scores. 

- We automatically construct training examples based on annotations in image caption datasets and design coarse- and fine-grained tasks to train the evaluation model. 

- InfoMetIC achieves better correlation with human judgements on multiple benchmarks, as well as on our newly constructed fine-grained caption evaluation benchmark CapTokenEval. 

## **2 Related Work** 

**Reference-only caption evaluation.** This type of evaluation only employs human-written captions as references and measures text similarity as the evaluation score. Most widely used metrics such as BLEU-4 (Papineni et al., 2002), ROUGEL (Lin, 2004), METEOR (Banerjee and Lavie, 2005), CIDEr (Vedantam et al., 2015a) and SPICE (Anderson et al., 2016) all fall into this category. BLEU-4 calculates the precision of n-gram matches; ROUGE-L measures the recall of the longest common subsequence; METEOR utilizes wordnet-based synonym matching to relieve the shortage of exact word matching; CIDEr introduces tf-idf to re-weight the importance of different n-grams; SPICE converts 

captions into scene graphs for similarity comparison. One major limitation of the above metrics is that they cannot properly count synonym matches. To overcome this deficiency, BERT-S (Zhang et al., 2020) leverages learned embeddings from a pretrained language model BERT (Devlin et al., 2019) to better measure semantic similarities. BERT-S++ (Yi et al., 2020) further improves BERT-S by taking into account the variance of multiple references. **Reference+image caption evaluation.** As an image is worth a thousands of words, a limited number of references cannot fully cover image contents, making the reference-only caption evaluation less reliable. Therefore, some works combine both references and images to evaluate generated captions. REO (Jiang et al., 2019a) uses a pretrained image-text retrieval model SCAN (Lee et al., 2018) to extract image contextualized caption features for computing relevance, extraness and omission scores. TIGER (Jiang et al., 2019b) calculates grounding vectors for captions via SCAN to measure similarity, which represent how much captions are grounded in an image. ViLBERTScore (Lee et al., 2020) is similar to BERT-S except that it generates visually-grounded features for each caption token by ViLBERT (Lu et al., 2019). FAIEr (Wang et al., 2021) fuses scene graphs of the image and references as a union scene graph and compares it with the scene graph of generated captions. 

**Reference-free caption evaluation.** To alleviate the annotation burden of obtaining references, a few works propose to evaluate image captions without references. UMIC (Lee et al., 2021) fine-tunes a pretrained multimodal transformer UNITER (Chen et al., 2020) by contrastive learning to compute an image-text matching score. CLIP-S (Hessel et al., 2021) directly utilizes image-text similarity from CLIP (Radford et al., 2021) - an image-text matching model trained on large-scale open-domain data. CLIP-S has achieved state-of-the-art evaluation performance. However, these methods only provide single scores which are less informative to evaluate image captions. In this work, we aim to provide more fine-grained feedbacks, not only indicating the captioning quality from precision and recall aspects, but also pointing out detailed mistakes such as incorrect words and unmentioned regions. 

## **3 Method** 

We first introduce our model architecture in Sec 3.1 and then describe the training and inference ap- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0003-00.png)


**----- Start of picture text -----**<br>
Vision Recall Score Text Precision Score Fine-grained Scoring Module<br>ො𝑣𝑔 Ƹ𝑡𝑔 Text-filtered Vision Encoder<br>Fine-grained Scoring Module ො𝑣𝑡𝑔1 𝛼1𝑣<br>ො𝑣1 ො𝑣2 … ො𝑣𝑚 ො𝑣𝑛𝑢𝑙𝑙 Ƹ𝑡1 Ƹ𝑡2 … Ƹ𝑡𝑛 ො𝑣𝑚 𝛼𝑚𝑣 ො𝑣𝑔<br>Inter-modal Encoder ො𝑣𝑛𝑢𝑙𝑙 𝛼𝑛𝑢𝑙𝑙𝑣<br>𝑣𝑔 Vision Intra-modal Encoder 𝑣1 𝑣2 …… 𝑣𝑚 𝑣null 𝑡1 Text Intra-modal Encoder 𝑡2 … 𝑡𝑛 𝑡𝑔 𝑣Ƹ𝑡Ƹ𝑡𝑔12 𝛼𝛼1𝑡2𝑡 Ƹ𝑡𝑔<br>CLIP Vision Encoder CLIP Text Encoder<br>Ƹ𝑡𝑛 𝛼𝑛 [𝑡]<br>… [s]a[e] [s]man[e] … [s].[e] [s] a man riding on  Vision-filtered Text Encoder<br>the a motorcycle. [e]<br>…<br>…<br>**----- End of picture text -----**<br>


Figure 2: **Left:** the overall architecture of **Info** rmative **Met** ric for Reference-free **I** mage **C** aption evaluation (InfoMetIC). **Right:** the detailed structure of the _Fine-grained Scoring Module_ . 

proaches in Sec 3.2 and Sec 3.3 respectively. 

## **3.1 Model Architecture** 

Figure 2 illustrates the overall framework of our informative evaluation model, which consists of three modules: _Token-level Encoding_ , _Intra&Inter Modality Fusion_ and _Fine-grained Scoring_ . Given an image _I_ and a caption _C_ as inputs, the Tokenlevel Encoding module firstly generates a sequence of token-level features to represent the image and caption respectively. Then the Intra&Inter Modality Fusion module captures the intra- and intermodality relationships. Finally, the Fine-grained Scoring module produces token-level scores for each visual and textual token and derives vision recall, text precision, and overall scores based on the token-level scores. 

## **3.1.1 Token-level Encoding** 

VLP models have shown superior performance and generalization ability in many vision-and-language tasks (Chen et al., 2020). Therefore, we utilize a state-of-the-art VLP model CLIP to extract tokenlevel image and caption features. To be noted, our method can be adapted to different VLP models. **Image Token Features.** In order to obtain semantically meaningful image tokens, we use a pretrained object detector to detect region bounding boxes in image _I_ . We encode each cropped region via CLIP vision encoder to get fine-grained token-level features ( _v_ 1 _, ..., vm_ ), where _m_ is the number of detected regions. The whole image is encoded as a global vision feature _vg_ . We further utilize a zero vector to represent a vision null token _vnull_ , which aims to align with any texts irrelevant to the image. **Caption Token Features.** For a caption _C_ , CLIP 

text encoder can generate a global feature _tg_ to capture overall semantics of the whole sentence. Although it could also generate a sequence of text token features, these features can overuse the sentence context, which harms fine-grained evaluation. An illustration about the context overuse can be found in Appendix A. Therefore, we encode each token in _C_ separately as shown in Figure 2 to obtain independent token-level features ( _t_ 1 _, ..., tn_ ), where _n_ is the number of text tokens. 

## **3.1.2 Intra&Inter Modality Fusion** 

In order to learn intra-modal relationships, we utilize two multi-layer transformers (Vaswani et al., 2017) to encode image and text tokens separately. As spatial information is essential to infer relationships across image regions, we apply a linear layer to convert normalized bounding boxes as position features and add them to the initial image token features before fed into the intra-modal transformer. Likewise, we add learnable position features for the text tokens. For visual intra-modal encoding, we concatenate _vg_ with ( _v_ 1 _, · · · , vm, vnull_ ) to alleviate possible vision context loss in fine-grained image tokens due to imperfect detection. For textual intramodal encoding, we directly utilize ( _t_ 1 _, · · · , tm_ ) tokens as inputs. 

We concatenate the image and text token-level features after intra-modal encoding and utilize an inter-modal encoder to learn correlation between vision and text modalities. The inter-modal encoder is implemented as a multi-layer cross-modal transformer (Chen et al., 2020). We denote the output features for image tokens as _V_[ˆ] = (ˆ _v_ 1 _...,_ ˆ _vm,_ ˆ _vnull_ ), output features for text tokens as _T_[ˆ] = ( _t_[ˆ] 1 _, ..., t_[ˆ] _n_ ). 

## **3.1.3 Fine-grained Scoring** 

The Fine-grained Scoring module aims to predict which text tokens are incorrect and which image tokens are not mentioned. It consists of two crossmodal attention layers, namely Text-filterd Vision Encoder and Vision-filterd Text Encoder as shown in the right of Figure 2. To identify which image tokens are mentioned, we use global text feature _tg_ as query and token-level vision features _V_[ˆ] as key in the cross-modality attention layer to calculate visual token-level scores _α[v]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-03.png)


Similarly, to identify which text tokens are incorrect, we use global vision feature _vg_ as query and token-level text features _T_[ˆ] as key to calculate textual token-level scores _α[t]_ by another crossmodality attention layer. 

Based on token-level scores, we derive vision recall score and text precision scores to measure the comprehensiveness and accuracy of generated captions respectively. We take visual token-level scores _α[v]_ and token-level vision features _V_[ˆ] to obtain a text-conditioned vision feature ˆ _vg_ by weighed average as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-06.png)


The more image regions are mentioned in a caption, the closer its text-conditioned vision feature should be to the global vision feature _vg_ . Thus, we compute the vision recall score as the cosine similarity between _v_ ˆ _g_ and _vg_ , represented as _f[R]_ ( _I, C_ ) = cos(ˆ _vg, vg_ ) _/τ_ , where _τ_ is a learnable temperature parameter. Taking the untrained global vision feature _vg_ as the comparison object, our vision recall score implicitly considers the salience of visual information, as illustrated in Appendix B. In a similar way, we can obtain a vision-conditioned text feature _t_[ˆ] _g_ and compute a text precision score _f[P]_ ( _I, C_ ) = cos( _t_[ˆ] _g, tg_ ) _/τ_ . Our overall score is the summation of precision score and recall score: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-08.png)


## **3.2 Multi-task Learning** 

To learn fine-grained token-level predictions as well as coarse-grained text precision and vision recall scores, we propose multiple training tasks to jointly optimize our evaluation model. 

## **3.2.1 Coarse-grained Score Learning** 

Given an aligned image-caption pair ( _I, C_ ), we construct negative samples by pairing _I_ with other captions in the training batch or pairing _C_ with other images in the batch. Then, we calculate Noisy Contrastive Learning (NCE) loss _lr_ based on vision recall scores and _lp_ based on text precision scores. The NCE loss _lr_ is calculated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-15.png)


where _NI_ means a set of negative captions for image _I_ within the batch _B_ , _NC_ means negative images for caption _C_ . The NCE loss _lp_ is similar to Eq (5) but utilizes _f[P]_ ( _I, C_ ) scores in computation. 

**Hard Textual Negatives.** In the above coarsegrained score learning, negative captions for an image are randomly selected from the dataset and usually contains many irrelevant contents with the image. These textual negatives are not hard enough to learn a good vision recall score. Because the model could compute a high recall score for positive pairs by putting high weight to only one rather than all mentioned regions. To address this problem, we further design Hard Textual Negatives (HTN) during coarse-grained score learning. For multiple annotated captions of an image, we consider the one with more semantic words (nouns, verbs, adjectives and adverbs) should get higher vision recall score than the others. Therefore, we treat the other ones as hard textual negatives. The HTN loss _lr[h]_[is] calculated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0004-18.png)


where _C[h]_ is a hard textual negative for caption _C_ . 

## **3.2.2 Fine-grained Score Learning** 

To improve fine-grained evaluation, we design a sequence labeling task called Fine-grained Score learning. We automatically generate supervision signals to learn token-level predictions. For the text part, we prepare labels in a self-supervised manner. Given an image _I_ and its groundtruth caption _C_ , _′_ we generate a polluted caption _C_ by randomly 

replacing a semantic word with a frequent word of the same part-of-speech tag. The text sequence label _Y[t]_ for ( _I, C ′_ ) is constructed by setting the polluted word as 0 (incorrect) and other semantic words as 1 (correct). Non-semantic words such as adpositions, conjunctions are excluded in training. For the image part, we make use of existing phrase grounding annotations which align each phrase in a caption with its corresponding bounding boxes in the image. The vision sequence label _Y[v]_ for ( _I, C_ ) is constructed by setting all regions mentioned by the caption as 1 and otherwise 0. 

We use cross-entropy losses for both textual and visual fine-grained score learning tasks: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0005-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0005-03.png)


where _lt[token]_ and _lv[token]_ refer to the text-part and vision-part loss respectively, _α[t]_ and _α[v]_ are textual token-level scores and visual token-level scores in Eq (2), _n[s]_ is the number of semantic words. 

## **3.3 Inference** 

Given input pair ( _I, C_ ), we first compute tokenlevel scores _α[v]_ and _α[t]_ for fine-grained prediction with a threshold _β_ . Considering that a caption hardly contains more than 10 semantic words, we set _β_ as 0.1. For the text part, semantic tokens with a score greater than _β_ are judged as correct ones. For the image part, regions with a score greater than _β_ are identified as mentioned ones. 

Then we calculate the vision recall, text precision, and overall scores as in Eq (4). We denote our vision recall score _f[R]_ ( _I, C_ ) as InfoMetIC _[R]_ , text precision score _f[P]_ ( _I, C_ ) as InfoMetIC _[P]_ , and overall score _f[O]_ ( _I, C_ ) as InfoMetIC. Furthermore, we combine our overall score with the CLIP similarity: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0005-08.png)


where _τ[clip]_ is the temperature of CLIP. 

## **4 Experiment** 

## **4.1 Experimental Setting** 

**Training Datasets.** With the training splits of Flickr30k (Young et al., 2014) and MSCOCO (Lin et al., 2014) datasets, we construct 715,662 imagecaption pairs for general coarse-grained score learning, and 611,105 triplets with hard textual negatives. For fine-grained score leaning, we construct 

512,000 samples from MSOCO and Flick30k for the text part training and 178,689 samples from Flickr30k for the vision part training. 

**Implementation Details.** We use CLIP(ViT-B/32) for token-level encoding. The image regions are detected by the bottom-up model (Anderson et al., 2018). To remove redundant bounding boxes, we use k-means algorithm to generate 20 clusters among 100 detected regions and select one region per cluster. The details can be found in Appendix C. The maximum length for textual tokens is set as 32. In the intra&inter modality fusion, intra- and inter-modal encoders contain 4 and 2 transformer layers respectively. During training, the batch size is set as 32 and the initial learning rate is set as 1e-4. We iteratively train our model on multiple tasks for 32,000 iterations. The training ratio of coarse- and fine-grained tasks is 3:1. The training takes 5 hours on 4 V100 GPUs. 

## **4.2 Coarse-grained Score Evaluation** 

## **4.2.1 Evaluation Datasets** 

**Flickr8k-Expert** (Hodosh et al., 2013a) contains 5,644 pairs of images and machine-generated captions. Each pair is scored from 1 (irrelevant) to 4 (well related) by 3 expert annotators. 

**Flickr8k-CF** (Hodosh et al., 2013a) consists of 47,830 image-captions pairs. Each pair is judged "yes" or "no" by at least 3 annotators, where "yes" is for good captions. The final score of each pair is determined by the proportion of "yes". 

**Composite** (Aditya et al., 2018) contains 3,995 images from MSCOCO, Flickr30K and Flickr8k (Hodosh et al., 2013b). For each image, there are two machine-generated captions and one humanwritten caption. Every image-caption pair is scored from 1 (irrelevant) to 5 (perfectly related). 

**Pascal-50S** (Vedantam et al., 2015b) contains 4,000 triplets, each of which contains an image and two captions. Annotators are asked to judge which caption is better. According to caption types, Pascal-50S is evenly split into 4 subsets: ‘HC’ means two correct human-written captions; ‘HI’ means two human-written captions but one is wrong; ‘HM’ means one human-written caption and one machine-generated caption; ‘MM’ means two machine-generated captions. 

**THumB 1.0** (Kasai et al., 2021) contains 500 images from MSCOCO. Each image is paired with one human-written caption and four machinegenerated captions. For each image-caption pair, 

Table 1: Overall score comparison on Flickr8k-Expert (F-Ex), Flickr8k-CF (F-CF), Composite (Com) and Pascal50S. ‘w/ ref’ means using 4-5 ground-truth references. ‘w/o ref’ means using no reference. 

|**Type**|**Metric**|**F-Ex(**_τc_**)**|**F-CF(**_τb_**)**|**Com(**_τc_**)**|HC|**Pascal-50S (accuracy)**<br>HI<br>HM<br>MM|**Pascal-50S (accuracy)**<br>HI<br>HM<br>MM|**Pascal-50S (accuracy)**<br>HI<br>HM<br>MM|Mean|
|---|---|---|---|---|---|---|---|---|---|
||BLEU-4|30.8|16.9|30.6|52.5|90.4|63.0|42.3|55.8|
||ROUGE-L|32.3|19.9|32.4|55.0|95.3|93.1|58.7|75.5|
||METEOR|41.8|22.2|38.9|59.0|97.7|93.9|62.0|78.2|
||CIDEr|43.9|24.6|37.7|53.7|98.1|90.8|63.7|76.6|
||SPICE|44.9|24.4|40.3|56.9|96.3|87.1|66.4|76.7|
|w/ ref|BERT-S|39.2|22.8|30.1|54.4|96.1|94.3|56.4|75.3|
||BERT-S++|46.7|-|44.9|65.4|98.1|96.4|60.3|80.1|
||TIGEr|49.3|-|45.4|56.0|99.8|92.8|74.2|80.7|
||ViLBERTScore-F|50.1|-|52.4|49.9|99.6|93.1|75.8|79.6|
||FAIEr-4|52.6|35.4|57.7|59.7|**99.9**|92.7|73.4|81.4|
||RefCLIP-S|53.0|36.4|55.4|57.9|99.5|96.1|80.8|83.6|
||UMIC|46.8|-|56.1|66.1|99.8|98.1|76.2|85.1|
||FAIEr-r|50.1|32.4|50.5|-|-|-|-|-|
|w/o ref|CLIP-S<br>CLIP-S_tune_|51.5<br>54.3|34.4<br>36.6|53.8<br>57.3|60.4<br>61.0|99.4<br>99.5|**97.8**<br>95.9|77.1<br>**82.0**|83.7<br>84.6|
||InfoCLIP|32.6|23.5|15.3|37.3|87.3|58.9|72.9|64.1|
||InfoCLIP_tune_|37.7|27.7|24.6|37.3|92.5|62.7|74.7|66.8|
||InfoMetIC|54.2|36.3|59.2|69.0|99.8|94.0|78.3|85.3|
||InfoMetIC+|**55.5**|**36.6**|**59.3**|**69.9**|99.7|96.8|79.6|**86.5**|



there are a precision score measuring the accuracy of the caption, a recall score assessing how much of the salient information is covered, and a total score measuring the overall quality. 

## **4.2.2 Evaluation Metrics** 

We follow previous works (Hessel et al., 2021; Vedantam et al., 2015b; Kasai et al., 2021) to evaluate captioning metrics. We use kendall-c correlation ( _τc_ ) on Flickr8k-Expert, kendall-b correlation ( _τb_ ) on Flickr8k-CF, kendall-c correlation ( _τc_ ) on Composite, classification accuracy on Pascal-50s and Pearson correlation ( _ρ_ ) on THumB 1.0. 

## **4.2.3 Comparison with State of the Arts** 

We compare InfoMetIC with SOTA methods as well as three strong baselines: CLIP-S _tune_ , InfoCLIP and InfoCLIP _tune_ . CLIP-S _tune_ calculates an overall score as CLIP-S (Hessel et al., 2021) but is fine-tuned on MSCOCO and Flickr30k. InfoCLIP directly uses CLIP to perform fine-grained scoring like InfoMetIC but removes the Intra&Inter Modality Fusion and parameters in Fine-grained Scoring. InfoCLIP _tune_ is a fine-tuned version of InfoCLIP. More details can be found in the Appendix D. 

Table 1 shows the overall score comparison on Flickr8k-Expert, Flickr8k-CF, Composite and Pascal-50S. Our reference-free metric InfoMetIC achieves state-of-the-art correlation with human judgements on Composite and Pascal-5OS. It is on par with the strong baseline CLIP-S _tune_ on Flickr8k-Expert and Flickr8k-CF. To be noted, 

InfoMetIC performs much better than InfoCLIP, which proves the necessity of our model architecture upon CLIP backbones. After combined with CLIP similarity, InfoMetIC[+] further improves performances on all benchmarks. 

To separately evaluate the performance of our vision recall score InfoMetIC _[R]_ and text precision score InfoMetIC _[P]_ , we further conduct experiments on THumB 1.0 in Table 3. **First** , by comparing InfoMetIC _[P]_ and InfoMetIC _[R]_ , InfoMetIC _[R]_ achieves better correlation with human-labeled recall score and InfoMetIC _[P]_ achieves better correlation with human-labeled precision score. This indicates that our InfoMetIC _[R]_ and InfoMetIC _[P]_ indeed evaluates the recall of image contents and the precision of caption respectively. Besides, both InfoMetIC _[P]_ and InfoMetIC _[R]_ surpass the stateof-the-art reference-free metric CLIP-S on total score correlation. **Second** , our overall score InfoMetIC achieves significant boost on total score, which demonstrates that precision and recall are complementary in human’s final evaluation for captions. InfoMetIC[+] slightly improves the total score performance. **Third** , compared with the state-ofthe-art reference-based metric RefCLIP-S (Hessel et al., 2021), our InfoMetIC[+] achieves much better recall correlation but lower precision correlation with humans. This is because text-text semantic comparison is much easier than cross-modal semantic comparison, making the precision correlation of reference-based metrics higher. However, limited 

Table 2: Ablation Study on Flickr8k-Expert (F-Ex), Flickr8k-CF (F-CF), Composite (Com), Pascal-50S and THumB 1.0. HTN denotes using hard text negatives in coarse-grained score learning, and FS refers to the finegrained score leaning. ‘ _vg_ ’ means incorporating the vision global feature in the Intra&Inter Modality Fusion Module. 

|**Id**|**Architecture**<br>Intra Inter _vg_|**Architecture**<br>Intra Inter _vg_|**Architecture**<br>Intra Inter _vg_|**Training**<br>HTN FS|**Training**<br>HTN FS|**F-Ex**|**F-CF**|**Com**|HC|**Pascal-50S**<br>HI<br>HM<br>MM|**Pascal-50S**<br>HI<br>HM<br>MM|**Pascal-50S**<br>HI<br>HM<br>MM|Mean|**THumB w/o h**<br>P<br>R<br>Total|**THumB w/o h**<br>P<br>R<br>Total|**THumB w/o h**<br>P<br>R<br>Total|**THumB**<br>P<br>R|**THumB**<br>P<br>R|**w/ h**<br>Total|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|r1|✓||✓|||51.7|36.8|57.8|58.0|99.5|95.0|76.3|82.2|0.23|0.26|0.35|0.20|0.26|0.32|
|r2||✓|✓|||55.1|**37.1**|59.0|59.5|99.8|95.4|78.1|83.2|0.23|0.26|0.35|0.20|0.26|0.32|
|r3|✓|✓||||55.1|36.9|**59.4**|58.6|**99.9**|95.7|79.6|83.5|0.21|0.26|0.34|0.19|0.26|0.32|
|r4|✓|✓|✓|||**55.2**|36.9|59.3|58.0|99.7|96.1|**80.8**|83.7|0.22|0.26|0.35|0.20|0.26|0.33|
|r5|✓|✓|✓|✓||54.5|36.2|58.8|**69.3**|99.6|93.7|75.2|84.5|**0.23**|0.28|**0.37**|**0.22**|0.30|0.37|
|r6|✓|✓|✓||✓|**55.2**|37.0|59.3|60.2|99.7|96.8|79.6|84.1|0.22|0.26|0.34|0.20|0.26|0.32|
|r7|✓|✓|✓|✓|✓|54.2|36.3|59.2|69.0|99.8|94.0|78.3|**85.3**|0.22|**0.30**|**0.37**|0.21|**0.32**|**0.38**|



Table 3: Experiments on THumB 1.0. ‘w/o Human’ means discarding human annotated image-caption pairs. 

|pairs.||
|---|---|
|**Ref Metric**<br>**w/o Human**<br>P<br>R<br>Total|**w/ Human**<br>P<br>R<br>Total|
|||
|w/<br>BLEU<br>.21<br>.13<br>.25<br>ROUGE-L<br>.26<br>.17<br>.31<br>CIDEr<br>.27<br>.18<br>.33<br>SPICE<br>.26<br>.15<br>.30<br>BERT-S<br>.27<br>.18<br>.33<br>RefCLIP-S<br>**.34**<br>**.27**<br>**.44**|.15<br>.04<br>.13<br>.18<br>.07<br>.18<br>.21<br>.11<br>.23<br>.20<br>.09<br>.21<br>.20<br>.10<br>.21<br>**.31**<br>**.26**<br>**.41**|
|||
|w/o<br>InfoCLIP_R_<br>.05<br>.19<br>.17<br>InfoCLIP_P_<br>.11<br>-.22<br>-.08<br>InfoCLIP<br>.13<br>-.06<br>.04<br>InfoCLIP_tune_ .15<br>-.15<br>.00<br>CLIP-S<br>.18<br>.27<br>.32<br>CLIP-S_tune_<br>.15<br>.26<br>.29<br>InfoMetIC_R_<br>.18<br>.29<br>.34<br>InfoMetIC_P_<br>.23<br>.27<br>.36<br>InfoMetIC<br>.22<br>.30<br>.37<br>InfoMetIC+<br>**.22**<br>**.33**<br>**.39**|.05<br>.19<br>.17<br>.09<br>-.20<br>-.08<br>.11<br>.06<br>.03<br>.11<br>-.15<br>-.03<br>.17<br>.28<br>.32<br>.13<br>.26<br>.28<br>.19<br>.32<br>.36<br>.20<br>.27<br>.33<br>.21<br>.32<br>.38<br>**.21**<br>**.34**<br>**.39**|



Table 4: Cross-modal retrieval performances on Nocaps. 

|caps.||
|---|---|
|**Method**<br>**image to text**<br>R@1 R@5 R@10|**text to image**<br>R@1 R@5 R@10|
|||
|TIGER<br>63.8<br>87.0<br>92.4<br>CLIP-S<br>88.2<br>98.3<br>99.7<br>InfoMetIC<br>76.6<br>96.5<br>99.1<br>InfoMetIC+<br>**90.9**<br>**98.8**<br>**99.7**|22.5<br>66.5<br>81.9<br>67.5<br>91.5<br>95.8<br>71.6<br>94.4<br>97.7<br>**76.2**<br>**95.9**<br>**98.4**|



constructing hard negatives indeed helps model better evaluate the vision content recall. Adding fine-grained score learning task (r4 vs r6) is also beneficial to the performance of coarse-grained score, which performs better on Pascal-50S and is comparable on other datasets. When trained with all tasks together (r7), InfoMetIC further improves on Pascal-50S and THumB 1.0, and achieves stateof-the-art performance on all datasets. 

textual references cannot fully capture image contents, which is harmful for vision recall. **Finally** , InfoMetIC achieves much better performance than InfoCLIP, which shows the effectiveness of our proposed modules on top of CLIP. 

## **4.2.4 Ablation Study** 

We first validate the effectiveness of our model architecture. As shown in Table 2, removing Intramodal encoders (r2 vs r4) or Inter-modal encoder (r1 vs r4) results in performance drop on Flickr8kExpert, Composite and Pascal-50S. Besides, removing global vision feature _vg_ from Intra&Inter encoding (r3 vs r4) leads to slight performance drop on Flickr8k-Expert, Pascal-50S and THumB1.0. 

We then carry out ablation study to verify the effectiveness of our training strategy in Table 2. Our proposed hard textual negatives (r4 vs r5) achieves significant improvements on HC subset of Pascal50s and THumB 1.0 Recall. This shows that 

## **4.3 Generalization Ability** 

InfoMetIC are trained with image-captions of Flick30k and MSCOCO. To evaluate its generalization ability, we further conduct experiments on NoCaps (Agrawal et al., 2019), whose objects are greatly different from Flick30k and MSCOCO. Since there are no human-labeled scores for imagecaption pairs, we perform text-image cross-modal retrieval to validate the effectiveness of our metric. As shown in Table 4, InfoMetIC performs worse than CLIP-S on image-to-text retrieval but better on text-to-image retrieval. After combining with CLIP similarity, InfoMetIC[+] achieves the state-ofthe-art performance on both two retrieval tasks. It indicates our overall score can also perform well on instances with unseen objects. 

## **4.4 Fine-grained Score Evaluation** 

**Dataset.** To validate the token-level evaluation performance of InfoMetIC, we collect a fine- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0008-00.png)


**----- Start of picture text -----**<br>
Image Regions Evaluation:  Text Tokens Evaluation:  Image Regions Evaluation:  Text tokens Evaluation:<br>0.11 0.01 0.02 0 A   baseball  0.29  ✔ 0.21  player ✔ running 0.09  ✘ to 0 0.17 0.150.16 0.03 0.02 0.02 0 A   couple 0.06 ✘ of     0.0  0.49  bikes ✔ parked 0.18  ✔<br>0.12 a    base during   a    game  . 0.11 next   to   a    train .<br>0    0.24  ✔ 0.01     0 0.16  ✔ 0 0.01 0.01   0  0 0.26  ✔ 0<br>0.13 0.02<br>Mentioned Unmentioned (a) Mentioned Unmentioned (c)<br>Image Regions Evaluation:  Text Tokens Evaluation:  Image Regions Evaluation:  Text tokens Evaluation:<br>0.21 0.020.03 A  man riding a  skateboard 0.97 0.14 A  0.04 white ✘ 0.01 truck ✘ travels 0.05  ✘ down  0.14<br>0.12 0.01 0.01 up the  0.29  side   ✔ 0.16 of    a    ✔ 0.01   ramp0.22 ✔ . 0.00.0 0.0 0.14   0.04 a    residential ✘ 0.07 street ✘ past 0.05 ✘ some  0.14<br>0.07 0   0.15 ✔ 0.02 0   0.04 ✘ 0 trash cans  .<br>Mentioned Unmentioned Mentioned Unmentioned 0.01 ✘ 0.01 ✘ 0.15<br>(b) (d)<br>**----- End of picture text -----**<br>


Figure 3: Fine-grained evaluation examples using InfoMetIC. Semantic words in captions are underlined. Semantic words and image regions with a score lower than 0.1 are considered as incorrect words and unmentioned regions respectively. We show all mentioned regions but only the top unmentioned image regions for better visualization. 

grained caption evaluation benchmark called CapTokenEval[1] . CapTokenEval is built upon a subset of THumB 1.0. We select 700 image-caption pairs whose precision scores are not perfect (< 5.0). For the text part, annotators are asked to judge which words are irrelevant with the image. For the image part, we collect 20 bounding boxes and ask annotators to identify mentioned regions. More details about the annotation can be found in Appendix E. **Quantitative Results.** Given each image-caption pair, InfoMetIC produces sequence of prediction for both image regions and caption tokens. To quantify token-level evaluation performance, for the text part, we only calculate the accuracy of semantic tokens (nouns, verbs, adjectives and numbers). As shown in Table 5, without extra parameters, InfoCLIP achieves promising performance for finegrained visual evaluation but poor performance in the text part. Consistent with the result shown in Table 3 that InfoCLIP _[R]_ ourperforms InfoCLIP _[P]_ , it further shows the importance of context fusion for text precision evaluation. With multi-task learning, InfoMetIC achieves promising prediction accuracy on both vision and text sequence. Both hard textual negatives and fine-grained score learning task contribute to token-level evaluation performance. Notably, fine-grained score learning task greatly boosts the text-part accuracy. Coarse-grained contrastive learning for text precision score within a batch can result in the model only putting relatively higher weights on a few correct text tokens. Our fine-grained score learning task could effectively alleviate this lazy behavior by teaching the model to put high weights on all correct tokens. **Qualitative Results.** We show some qualitative results of token-level evaluation in Figure 3. 

1We will release codes and datasets upon acceptance. 

Table 5: Token-level evaluation on CapTokenEval. CS: coarse-grained score learning; HTN: adding hard textual negatives in CS; FS: fine-grained score leaning. 

|Method|CS|Training<br>HTN|FS|Accuracy<br>Vision Text|Accuracy<br>Vision Text|
|---|---|---|---|---|---|
|InfoCLIP|-|-|-|0.73|0.33|
|InfoCLIP_tune_|-|-|-|0.74|0.37|
||✓|_×_|_×_|0.74|0.36|
|Ours|✓<br>✓|✓<br>_×_|_×_<br>✓|0.75<br>0.75|0.37<br>0.79|
||✓|✓|✓|**0.75**|**0.80**|



Firstly, InfoMetIC is able to identify various mistakes made in captions, including wrong actions (e.g.“running” in case a), wrong objects (e.g.“ramp” in case b), and wrong modifiers (e.g.“couple” in case c). Secondly, InfoMetIC could report mentioned image regions (e.g. the “skateboard” region in case b) and unmentioned regions (e.g. the “building” region in case b). Especially, when the caption is totally irrelevant with the image, as shown in case d, InfoMetIC could not only judge the wrong semantic words but also inform that all image regions are not mentioned by putting a very high score to the vision null token. One limitation of current metric is that although we perform region filtering by clustering, we still find some similar regions as shown in Figure 3(c). Better ways to de-duplicate image regions could bring further improvement. 

## **5 Conclusion** 

To provide feedbacks on detailed mistakes of image captions, we propose a reference-free informative metric InfoMetIC based on a state-of-the-art visionlanguage model. InfoMetIC not only points out incorrect descriptions, but also tells which regions are not mentioned. Based on these fine-grained evaluation, InfoMetIC derives a text precision score, a 

vision recall score, and an overall score. We design both coarse- and fine-grained training tasks to optimize our metric. The overall score given by our metric achieves state-of-the-art correlation with human judgement on multiple benchmarks. We further build a token-level caption evaluation benchmark CapTokenEval to prove the effectiveness of our fine-grained evaluation. 

## **Limitations** 

This work focuses on informative image captioning evaluation, including an overall score, vision recall, text precision and token-level scores. The effectiveness of our metric is validated on standard image captioning benchmarks. InfoMetIC in this work may not perform well in other captioning tasks due to domain gap, but we contend that our general framework can be adapted to other domains such as text-aware image captioning. For example, for textaware image captioning which focuses more on scene texts in images, we could further encode text regions besides the existing object regions for better comparison with captions. In the future, we will comprehensively explore how to adapt our metric to other captioning tasks, such as text-aware image captioning and video captioning. 

_valuation@ACL_ , pages 65–72. Association for Computational Linguistics. 

- Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. Uniter: Universal image-text representation learning. In _European conference on computer vision_ , pages 104–120. Springer. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT (1)_ , pages 4171–4186. Association for Computational Linguistics. 

- Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A reference-free evaluation metric for image captioning. In _EMNLP (1)_ , pages 7514–7528. Association for Computational Linguistics. 

- Micah Hodosh, Peter Young, and Julia Hockenmaier. 2013a. Framing image description as a ranking task: Data, models and evaluation metrics. _J. Artif. Intell. Res._ , 47:853–899. 

- Micah Hodosh, Peter Young, and Julia Hockenmaier. 2013b. Framing image description as a ranking task: Data, models and evaluation metrics. _Journal of Artificial Intelligence Research_ , 47:853–899. 

- Anwen Hu, Shizhe Chen, and Qin Jin. 2020. Icecap: Information concentrated entity-aware image captioning. In _Proceedings of the 28th ACM International Conference on Multimedia_ , pages 4217–4225. 

## **References** 

- Somak Aditya, Yezhou Yang, Chitta Baral, Yiannis Aloimonos, and Cornelia Fermüller. 2018. Image understanding using vision and reasoning through scene description graph. _Comput. Vis. Image Underst._ , 173:33–45. 

- Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. 2019. nocaps: novel object captioning at scale. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 8948–8957. 

- Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2016. Spice: Semantic propositional image caption evaluation. In _European conference on computer vision_ , pages 382–398. Springer. 

- Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2018. Bottom-up and top-down attention for image captioning and visual question answering. In _CVPR_ , pages 6077–6086. Computer Vision Foundation / IEEE Computer Society. 

- Satanjeev Banerjee and Alon Lavie. 2005. METEOR: an automatic metric for MT evaluation with improved correlation with human judgments. In _IEE-_ 

- Anwen Hu, Shizhe Chen, and Qin Jin. 2021. Questioncontrolled text-aware image captioning. In _Proceedings of the 29th ACM International Conference on Multimedia_ , pages 3097–3105. 

- Ming Jiang, Junjie Hu, Qiuyuan Huang, Lei Zhang, Jana Diesner, and Jianfeng Gao. 2019a. Reorelevance, extraness, omission: A fine-grained evaluation for image captioning. _arXiv preprint arXiv:1909.02217_ . 

- Ming Jiang, Qiuyuan Huang, Lei Zhang, Xin Wang, Pengchuan Zhang, Zhe Gan, Jana Diesner, and Jianfeng Gao. 2019b. Tiger: text-to-image grounding for image caption evaluation. _arXiv preprint arXiv:1909.02050_ . 

- Jungo Kasai, Keisuke Sakaguchi, Lavinia Dunagan, Jacob Morrison, Ronan Le Bras, Yejin Choi, and Noah A Smith. 2021. Transparent human evaluation for image captioning. _arXiv preprint arXiv:2111.08940_ . 

- Hwanhee Lee, Seunghyun Yoon, Franck Dernoncourt, Trung Bui, and Kyomin Jung. 2021. UMIC: an unreferenced metric for image captioning via contrastive learning. In _ACL/IJCNLP (2)_ , pages 220– 226. Association for Computational Linguistics. 

- Hwanhee Lee, Seunghyun Yoon, Franck Dernoncourt, Doo Soon Kim, Trung Bui, and Kyomin Jung. 2020. Vilbertscore: Evaluating image caption using visionand-language bert. In _Proceedings of the First Workshop on Evaluation and Comparison of NLP Systems_ , pages 34–39. 

- Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. 2018. Stacked cross attention for image-text matching. In _Proceedings of the European Conference on Computer Vision (ECCV)_ , pages 201–216. 

- Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In _International Conference on Machine Learning_ , pages 12888–12900. PMLR. 

- Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81. 

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In _European conference on computer vision_ , pages 740–755. Springer. 

- Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In _NeurIPS_ , pages 13–23. 

- Yingwei Pan, Ting Yao, Yehao Li, and Tao Mei. 2020. X-linear attention networks for image captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10971– 10980. 

- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_ , pages 311–318. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In _ICML_ , volume 139 of _Proceedings of Machine Learning Research_ , pages 8748–8763. PMLR. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in neural information processing systems_ , pages 5998–6008. 

- Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015a. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4566–4575. 

- Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015b. Cider: Consensus-based image description evaluation. In _CVPR_ , pages 4566–4575. IEEE Computer Society. 

- Sijin Wang, Ziwei Yao, Ruiping Wang, Zhongqin Wu, and Xilin Chen. 2021. Faier: Fidelity and adequacy ensured image caption evaluation. In _CVPR_ , pages 14050–14059. Computer Vision Foundation / IEEE. 

- Qiaolin Xia, Haoyang Huang, Nan Duan, Dongdong Zhang, Lei Ji, Zhifang Sui, Edward Cui, Taroon Bharti, and Ming Zhou. 2021. Xgpt: Cross-modal generative pre-training for image captioning. In _CCF International Conference on Natural Language Processing and Chinese Computing_ , pages 786–797. Springer. 

- Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, and Fei Huang. 2021. E2e-vlp: End-to-end vision-language pre-training enhanced by visual learning. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 503–513. 

- Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. 2023. mplug-2: A modularized multi-modal foundation model across text, image and video. _arXiv preprint arXiv:2302.00402_ . 

- Yanzhi Yi, Hangyu Deng, and Jinglu Hu. 2020. Improving image captioning evaluation by considering inter references variance. In _ACL_ , pages 985–994. Association for Computational Linguistics. 

- Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. _Transactions of the Association for Computational Linguistics_ , 2:67–78. 

- Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Revisiting visual representations in vision-language models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5579–5588. 

- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with BERT. In _ICLR_ . OpenReview.net. 

- Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, and Jianfeng Gao. 2020. Unified vision-language pre-training for image captioning and VQA. In _AAAI_ , pages 13041–13049. AAAI Press. 

||Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|Caption: A man with a red helmet on a small moped on a dirt road.|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||**_Token_**|[s]|a|man|with|a|red|helmet|on|a|small|mo|ped|on|a|dirt|road|.|[e]|
||**_CLIP-S_ **|**18.6**|**9.96 **|**13.45 **|**17.08 **|**12.4 **|**17.95**|**16.02**|**19.3 **|**18.3 **|**17.72**|**21.2 **|**22.77 **|**19.38 **|**21.25**|**18.53**|**23.75 **|**26.1**|**32.62**|
|||||||||||||||||||||



Figure 4: An illustration about the context overuse during text encoding of CLIP. The CLIP-S of each token are calculated with global vision feature and token-level text feature got by original CLIP encoding way rather than individually encoding. 

## **A Context Overuse Issue** 

CLIP (Radford et al., 2021) is trained to well align global image representations and sentence representation. Thus it applies a triangle masking during text encoding and treats the representation of the last text token [e] as the sentence representation. Due to the training objective and text masking mechanism, the text context information is accumulated with the sequence order, which is unfavorable for text-part fine-grained evaluation. As shown in Figure 4, the third ‘a’ is a meaningless indefinite article but gets a higher relevance score than the correct noun ‘man’. 

||**_Caption_**|𝑰𝒏𝒇𝒐𝑴𝒆𝒕𝑰𝑪𝑹|
|---|---|---|
||A very large sheep is standing.|1.66|
||A very large sheep is standing<br>in thegrass.|3.80|
||A very large sheep is standing<br>underclouds.|2.60|



Figure 5: An illustration about the influence of object salience to our vision recall score InfoMetIC _[R]_ . 

## **B Salience of Visual Information** 

Our vision recall score is calculated by comparing the text-conditioned vision features (the CLIP’s global vision feature) rather than the sum or average of all regions features. CLIP is trained with massive image-caption pairs and achieves promising performance on multiple Vision-Language tasks. Thus it’s convincing that the global vision feature produced by CLIP could well represent the salient information in an image. As illustrated in Figure 5, both ‘cloud’ and ‘grass’ are objects in the image, but InfoMetIC gives the second caption higher vision recall score because ‘grass’ is more salient than ‘clouds’ in the image. 

Table 6: Performance of InfoMetIC with different cluster numbers on Flickr8k-Expert (F-Ex), Flickr8k-CF (F-CF), Composite (Com), Pascal-50S and THumB w/ Human. 

|Human.||||||
|---|---|---|---|---|---|
|cluster|F-Ex|F-CF|Com|Pascal50S|Thumb|
|10|54.2|36.1|58.3|84.8|0.36|
|20|54.2|36.3|59.2|85.3|0.38|
|30|54.4|36.3|59.5|85.2|0.36|
|40|54.7|36.2|59.2|85.3|0.39|
|50|54.8|36.3|59.5|85.3|0.37|



## **C Cluster Number Setting Details** 

Similar image regions can cause confusion during fine-grained evaluation. In this work, redundant regions are removed by K-means clustering algorithm. Concretely, with 100 bounding boxes given by the object detection model, we perform K-means to generate _N_ clusters. For each cluster, the region with highest confidence score given by the object detection model is maintained. The evaluation performance of InfoMetIC with different _N_ settings is shown in Table 6. With the cluster number ranging from 10 to 50, the overall evaluation performance of InfoMetIC shows minor difference on these benchmarks. Taking into account both performance and complexity, we finally set _N_ as 20. 

## **D Baseline Details** 

To verify the effectiveness of InfoMetIC, besides state-of-the-art caption metrics, we set extra three baselines CLIP-S _tune_ , InfoCLIP and InfoCLIP _tune_ . As shown in Figure 6(a), CLIP-S (Hessel et al., 2021) directly uses the global representations given by CLIP(Radford et al., 2021) to calculate a cosine similarity as the overall score. CLIP-S _tune_ follows the same calculation manner but uses a CLIP fine-tuned on MSCOCO and Flickr30k as the backbone. Previous metrics can’t do fine-grained 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0012-00.png)


**----- Start of picture text -----**<br>
Vision Recall Score Text Precision Score<br>Score ො𝑣𝑔 Ƹ𝑡𝑔<br>Fine-grained Scoring Module  （ parameter-free ）<br>CLIP Vision Encoder 𝑣𝑔 CLIP Text Encoder 𝑡𝑔 𝑣𝑔 … 𝑣𝑛𝑢𝑙𝑙 … 𝑡𝑔<br>CLIP Vision Encoder CLIP Text Encoder<br>[s] A man riding on<br>the a motorcycle. [e] [s] A man<br>… [s] A [e] [s] man [e] … [s] . [e] riding on the a<br>motorcycle. [e]<br>(a) CLIP-S<br>(b) InfoCLIP<br>**----- End of picture text -----**<br>


Figure 6: Overall architectures of baselines CLIP-S and InfoCLIP 

caption evaluation. Therefore, we set a fine-grained evaluation baseline InfoCLIP, as shown in Figure 6(b). InfoCLIP performs fine-grained scoring as InfoMetIC without Intra&Inter Modality Fusion and parameters in Fine-grained Scoring, e.g. _Wq[v]_ and _Wk[v]_[in][Eq][(][1][).][InfoCLIP] _[tune]_[means][using][a] 

## **E CapTokenEval Annotation Details** 

To quantify caption evaluation performance at token level, we collect a fine-grained caption evaluation benchmark called CapTokenEval. The details of our annotation are introduced in following subsections. 

## **E.1 Data Preparation** 

We prepare image-caption pairs for annotation based on the publicly released dataset THumB 1.0 (Kasai et al., 2021). THumB 1.0 collects 500 images from MSCOCO (Lin et al., 2014) and pairs each image with 4 captions generated by state-ofthe-art image captioning models, including UPDown (Anderson et al., 2018), Unified-VLP (Zhou et al., 2020), VinVL-base and VinVL-large (Zhang et al., 2021). There are a precision score, a recall score and a total score for each image-caption pair. To ensure that textual token-level evaluation in our benchmark is hard enough, we select imagecaption pairs whose precision score is not perfect (<5.0). We finally collect 700 image-captions pairs from ThumB 1.0. As the data used in our annotation all come from publicly released datasets, there are no ethic issues. 

For each image, we extract 100 bounding boxes with pre-trained object detection model BottomUp (Anderson et al., 2018). To filter similar image regions, we apply K-means clustering on these bounding boxes. We generate 20 clusters for each image and choose a bounding box with highest con- 

fidence score of object classification from each cluster. Thus, for each image-caption pair, we provide 20 image regions to annotators, who will choose which regions are mentioned by the caption. For the text part, we tokenize the caption with Spacy[2] . 

## **E.2 Annotation Platform** 

We build a platform to support the fine-grained annotation. Figure 7 presents the annotation interface on our platform, which consists of three major parts. The middle part contains an image-caption pair to be annotated. The left part is the textual token-level annotation area, which lists all tokens in the caption. The right part is the visual tokenlevel annotation area, which places 20 images with bounding boxes indicating different image regions. 

## **E.3 Annotation Instruction** 

Given an image-caption pair, we ask annotators to identify which tokens in the caption are incorrect and which regions are mentioned by the caption. Besides, we require that if the caption mentions an object without descriptions about details, the image regions of detailed components shouldn’t be classified as ‘Mentioned’. For example, for the caption ‘a group of people riding on the back of an elephant’, the image region of the elephant nose shouldn’t be judged as ‘Mentioned’. 

We invite 20 college students as annotators. They all have sufficient English proficiency to understand image captions in English. We provide a document to inform annotators the goal of our annotation and detailed instructions about the usage of the annotation platform. Each annotator is assigned 35 image-caption pairs for annotation. 

2https://spacy.io/usage 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation_images/acl_2023_InfoMetIC_An_Informative_Metric_for_Reference-free_Image_Caption_Evaluation.pdf-0013-00.png)


Figure 7: An overview of the annotation platform. 

