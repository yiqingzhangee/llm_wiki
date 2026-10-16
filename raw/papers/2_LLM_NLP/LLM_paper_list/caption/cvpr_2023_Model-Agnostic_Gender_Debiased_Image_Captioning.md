## **Model-Agnostic Gender Debiased Image Captioning** 

Yusuke Hirota 

## Yuta Nakashima 

## Noa Garcia 

_{_ y-hirota@is., n-yuta@, noagarcia@ _}_ ids.osaka-u.ac.jp Osaka University 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0001-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0001-07.png)


**baseline** 

## **baseline** 

a **man** riding a wave on a surfboard 

a young **boy** riding a skateboard 

## **+LIBRA** 

**+LIBRA** 

a **woman** catching a wave on a surfboard 

a young **girl** riding a skateboard 

(a) context → gender bias mitigation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0001-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0001-18.png)


**baseline baseline** a man wearing a **suit** holding a banana a young boy holding a **baseball bat +LIBRA +LIBRA** a man in a **jacket** holding a banana a young boy holding a plastic **frisbee** 

(b) gender → context bias mitigation 

Figure 1. Generated captions by a baseline captioning model (UpDn [2]) and LIBRA. We show the baseline suffers from context _→_ gender/gender _→_ context biases, predicting incorrect gender or incorrect word ( _e.g_ ., in the left example, _skateboard_ highly co-occurs with men in the training set, and the baseline incorrectly predicts _boy_ ). Our proposed framework successfully modifies those incorrect words. 

## **Abstract** 

_Image captioning models are known to perpetuate and amplify harmful societal bias in the training set. In this work, we aim to mitigate such gender bias in image captioning models. While prior work has addressed this problem by forcing models to focus on people to reduce gender misclassification, it conversely generates gender-stereotypical words at the expense of predicting the correct gender. From this observation, we hypothesize that there are two types of gender bias affecting image captioning models: 1) bias that exploits context to predict gender, and 2) bias in the probability of generating certain (often stereotypical) words because of gender. To mitigate both types of gender biases, we propose a framework, called LIBRA, that learns from synthetically biased samples to decrease both types of biases, correcting gender misclassification and changing genderstereotypical words to more neutral ones. Code is available at https://github.com/rebnej/LIBRA ._ 

## **1. Introduction** 

In computer vision, societal bias, for which a model makes adverse judgments about specific population sub- 

groups usually underrepresented in datasets, is increasingly concerning [4, 6, 7, 11, 17, 22, 41, 43, 52, 57]. A renowned example is the work by Buolamwini and Gebru [7], which demonstrated that commercial facial recognition models predict Black women with higher error rates than White men. The existence of societal bias in datasets and models is extremely problematic as it inevitably leads to discrimination with potentially harmful consequences against people in already historically discriminated groups. 

One of the computer vision tasks in which societal bias is prominent is image captioning [49, 58], which is the task of generating a sentence describing an image. Notably, image captioning models not only reproduce the societal bias in the training datasets, but also amplify it. This phenomenon is known as bias amplification [10,15,24,42,64] and makes models produce sentences more biased than the ones in the original training dataset. As a result, the generated sentences can contain stereotypical words about attributes such as gender that are sometimes irrelevant to the images. 

Our study focuses on gender bias in image captioning models. First, based on the observations in previous work [5,8,18,44,51], we hypothesize that there exist two different types of biases affecting captioning models: 

_**Type 1**_ **.** context _→_ gender bias, which makes captioning 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Biased Caption Synthesis Debiasing Caption Generator<br>context → gender Original Caption<br>A  woman  in a suit<br>man<br>Transformer Decoder<br>Cross Attention<br>A woman in a suit woman Transformer Encoder<br>Original  gender → context Textual / Visual Embeddings<br>Captions<br>Mask<br>A  man  in a suit<br>Biased<br>Captions<br>A<br>man<br>Mask Swap<br>Classifier<br> in a suit<br>A woman in a suit A <mask> in a suit<br>T5<br>Mask<br>A woman in a  Filtering A woman in a<br>(finetuned)<br>A woman in a suit skirt skirt<br>A woman in a <mask><br>**----- End of picture text -----**<br>


Figure 2. Overview of LIBRA. For the original captions ( _i.e_ ., ground-truth captions written by annotators), we synthesize biased captions with context _→_ gender or/and gender _→_ context bias (Biased Caption Synthesis). Then, given the biased captions and the original images, we train an encoder-decoder captioner, Debiasing Caption Generator, to debias the input biased captions ( _i.e_ ., predict original captions). 

models exploit the context of an image and precedently generated words, increasing the probability of predicting certain gender, as shown in Figure 1 (a). 

- _**Type 2**_ **.** gender _→_ context bias, which increases the probability of generating certain words given the gender of people in an image, as shown in Figure 1 (b). 

Both types of biases can result in captioning models generating harmful gender-stereotypical sentences. 

A seminal method to mitigate gender bias in image captioning is Gender equalizer [8], which forces the model to focus on image regions with a person to predict their gender correctly. Training a captioning model using Gender equalizer successfully reduces gender misclassification (reducing context _→_ gender bias). However, focusing only on decreasing such bias can conversely amplify the other type of bias [18,51]. For example, as shown in Figure 6, a model trained to correctly predict the gender of a person can produce other words that are biased toward that gender (amplifying gender _→_ context bias). This suggests that methods for mitigating bias in captioning models must consider both types of biases. 

We propose a method called LIBRA (model-agnostic debiasing framework) to mitigate bias amplification in image captioning by considering both types of biases. Specifically, LIBRA consists of two main modules: 1) Biased Caption Synthesis (BCS), which synthesizes genderbiased captions (Section 3), and 2) Debiasing Caption Generator (DCG), which mitigates bias from synthesized captions (Section 4). Given captions written by annotators, BCS synthesizes biased captions with gender _→_ context or/and context _→_ gender biases. DCG is 

then trained to recover the original caption given a _⟨_ synthetic biased caption, image _⟩_ pair. Once trained, DCG can be used on top of any image captioning models to mitigate gender bias amplification by taking the image and generated caption as input. Our framework is model-agnostic and does not require retraining image captioning models. 

Extensive experiments and analysis, including quantitative and qualitative results, show that LIBRA reduces both types of gender biases in most image captioning models on various metrics [8, 18, 44, 66]. This means that DCG can correct gender misclassification caused by the context of the image/words that is biased toward a certain gender, mitigating context _→_ gender bias (Figure 1 (a)). Also, it tends to change words skewed toward each gender to less biased ones, mitigating gender _→_ context bias (Figure 1 (b)). Furthermore, we show that evaluation of the generated captions’ quality by a metric that requires human-written captions as ground-truth ( _e.g_ ., BLEU [30] and SPICE [1]) likely values captions that imitate how annotators tend to describe the gender ( _e.g_ ., _women posing_ vs. _men standing_ ). 

## **2. Related work** 

**Societal bias in image captioning** In image captioning [2, 62], societal bias can come from both the visual and linguistic modalities [8, 51, 65]. In the visual modality, the image datasets used to train captioning models are skewed regarding human attributes such as gender [14, 37, 59, 65, 66], in which the number of images with men is twice as much as those of women in MSCOCO [26]. Additionally, captions written by annotators can also be biased toward a certain gender because of gender-stereotypical expressions [5, 65], 

which can be a source of bias from the linguistic modality. Models trained on such datasets not only reproduce societal bias but amplify it [8, 18, 51, 65]. This phenomenon is demonstrated by Burns _et al_ . [8], which showed that image captioning models learn the association between gender and objects and make gender distribution in the predictions more skewed than in datasets. We show that LIBRA can mitigate such gender bias amplification in various captioning models. What is better, we demonstrate that our model often produces less gender-stereotypical captions than the original captions. 

**Mitigating societal bias** Mitigation of societal bias has been studied in many tasks [19, 20, 44, 46, 50, 53, 55, 56, 60, 63, 66], such as image classification [34] and visual semantic role labeling [61]. For example, Wang _et al_ . [53] proposed an adversarial debiasing method to mitigate gender bias amplification in image classification models. In image captioning, Burns _et al_ . [8] proposed the Gender equalizer we described in Section 1 to mitigate context _→_ gender bias. However, recent work [18, 51] showed that focusing on mitigating gender misclassification can lead to generating gender-stereotypical words and amplifying gender _→_ context bias. LIBRA is designed to mitigate bias from the two types of biases. 

**Image caption editing** DCG takes a _⟨_ caption, image _⟩_ pair as input and debiases the caption. This process is aligned with image caption editing [39, 40, 54] for generating a refined caption. These models aim to correct grammatical errors and unnatural sentences but not to mitigate gender bias. In Section 5.3, we compare DCG with a state-of-the-art image caption editing model [40] and show that a dedicated framework for addressing gender bias is necessary. 

## **3. Biased caption synthesis** 

Figure 2 shows an overview of LIBRA, consisting of BCS and DCG. This section introduces BCS to synthesize captions with both context _→_ gender or/and gender _→_ context biases. 

**Notation** Let _D_ = _{_ ( _I, y_ ) _}_ denote a training set of the captioning dataset, where _I_ is an image and _y_ = ( _y_ 1 _, . . . , yN_ ) is the ground-truth caption with _N_ tokens. _D_ g denotes a subset of _D_ , which is given by filter _F_ GW as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0003-06.png)


_F_ GW keeps captions that contains either women or men words ( _e.g_ ., _girl_ , _boy_ ).[1] Therefore, samples in _D_ g come with a gender attribute _g ∈G_ , where _G_ = _{_ female _,_ male _}_ .[2] 

> 1We pre-defined women and men words. The list is in the appendix. 

> 2In this paper, we focus on binary gender categories in our framework and evaluation by following previous work [8, 66]. We recognize that the more inclusive gender categories are preferable, and it is the future work. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0003-10.png)


**----- Start of picture text -----**<br>
Gender-swapping<br>A  man  with a beer is wearing a gray tie<br>T5-generation<br>A woman with a  umbrella  is wearing a gray  shirt<br>Merged<br>A  man  with a  hat  is wearing a gray tie<br>Original<br>A woman with a beer is wearing a gray tie<br>Gender-swapping<br>A  man  in the water trying to grab a frisbee<br>T5-generation<br>A woman in the  kitchen  trying to grab a  pizza<br>Merged<br>A  man  in the  air  trying to grab a frisbee<br>Original<br>A woman in the water trying to grab a frisbee<br>**----- End of picture text -----**<br>


Figure 3. Biased captions synthesized by BCS. Gender-swapping denotes synthesized captions by swapping the gender words (Section 3.1). T5-generation denotes synthesized captions by T5 (Section 3.2). Merged represents biased captions synthesized by applying T5-generation and Gender-swapping (Section 3.3). 

We define the set that consists of women and men words as gender words. 

## **3.1. Context** _→_ **gender bias synthesis** 

Context _→_ gender bias means gender prediction is overly contextualized by the image and caption. Therefore, the gender should be predictable from the image and caption context when the caption has context _→_ gender bias. The idea of synthesizing context _→_ gender biased captions is thus to swap the gender words in the original caption to make it consistent with the context when the gender predicted from the context is skewed toward the other gender. Since an original caption faithfully represents the main content of the corresponding image [3, 45], we can solely use the caption to judge if both image and caption are skewed. To this end, we train a sentence classifier that predicts gender from textual context to synthesize biased captions. We introduce the detailed steps. 

**Masking** Captions with context _→_ gender bias are synthesized for _D_ g. Let _F_ PG denote the filter that removes captions whose gender is predictable by the sentence classifier. Given _y ∈D_ g, _F_ PG instantiated by first masking gender words and replacing corresponding tokens with the mask token to avoid revealing the gender, following [18]. We denote this gender word masking by mask( _·_ ). 

**Gender classifier** We then train[3] gender classifier _f_ g to predict the gender from masked caption as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0003-17.png)


> 3Refer to the appendix for training details. 

where _p_ ( _G_ = _g|_ mask( _y_ )) is the probability of being gender _g_ given masked _y_ . _F_ PG is then applied to _D_ g as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0004-01.png)


recalling ˆ _g_ is a function of _y_ . 

**Gender swapping** The inconsistency of context _y[′]_ and gender _g_ means that _y[′]_ is skewed toward the other gender; therefore, swapping gender wards ( _e.g_ ., _man → woman_ ) in _y ∈ F_ PG( _D_ g) results in a biased caption. Letting swap( _·_ ) denote this gender swapping operation, the augmenting set _A_ CG is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0004-04.png)


Figure 3 shows some synthetically biased captions (refer to Gender-swapping). We can see that the incorrect gender correlates with context skewed toward that gender. For instance, in the top example, _tie_ is skewed toward men based on the co-occurrence of men words and _tie_ . 

## **3.2. Gender** _→_ **context bias synthesis** 

Our idea for synthesizing captions with gender _→_ context bias is to sample randomly modified captions of _y_ and keep ones with the bias. Sampling modified captions that potentially suffer from this type of bias is not trivial. We thus borrow the power of a language model. That is, captions with gender _→_ context bias tend to contain words that well co-occur with gender words, and this tendency is supposedly encapsulated in a language model trained with a large-scale text corpus. We propose to use the masked token generation capability of T5 [33] to sample modified captions and filter them for selecting biased captions. 

**T5 masked word generation** T5 is one of the state-ofthe-art Transformer language models. For better alignment with the vocabulary in the captioning dataset, we finetune T5 with _D_ by following the process of training the masked language model in [13].[4] After finetuning, we sample randomly modified captions using T5. Specifically, we randomly mask 15% of the tokens in _y ∈D_ . Note that we exclude tokens of the gender words if any as they serve as the only cue of the directionality of bias (either men or women). 

Let _yM_ denote a modified _y_ whose _m_ -th token ( _m ∈M_ ) is replaced with the mask token. The masked token generator by T5 can complete the masked tokens solely based on ˆ _yM_ , _i.e_ ., _y_ = T5( _yM_ ). With this, we can sample an arbitrary number of _y_ ˆ’s to make a T5-augmented set _D_ T5 as[5] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0004-10.png)


where _M_ is sampled from set _R_ of all possible masks. 

4Refer to the appendix for the details of this finetuning. 

5We remove trivial modification that replaces a word with its synonyms based on WordNet [27] and unnatural captions with dedicated classifier. More details can be found in the appendix. 

**Filtering** We then apply a filter to _D_ T5 to remove captions that decrease gender _→_ context bias, which is referred to as gender filter. We thus borrow the idea in Eq. (3). For this, we only use captions in _D_ T5 that contain the gender words, _i.e_ ., _D_ T5,g = _F_ GW( _D_ T5), to guarantee that all captions have gender attribute _g_ . To collectively increase gender _→_ context bias in the set, we additionally use condition _d_ ( _y[′] , y_ ) = _p_ ( _G_ = _g|_ mask( _y[′]_ )) _− p_ ( _G_ = _g|_ mask( _y_ )) _> δ_ , which means the gender of _y[′] ∈D_ T5,g should be more predictable than the corresponding original _y ∈D_ g by a predefined margin _δ_ . Gender filter _F_ GF is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0004-15.png)


The appendix shows that _F_ GF can keep more genderstereotypical sentences than the original captions. 

With the gender filter, augmenting set _A_ GC is given as the intersection of the filtered sets as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0004-18.png)


As a result, the synthesized captions contain genderstereotypical words that often co-occur with that gender as shown in Figure 3 (refer to T5-generation). For example, in the bottom sample, _kitchen_ co-occurs with women words about twice as often as it co-occurs with men words in _D_ , amplifying gender _→_ context bias. 

## **3.3. Merging together** 

For further augmenting captions, we merge the processes for augmenting both context _→_ gender and gender _→_ context biases, which is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0004-22.png)


which means that the process for synthesizing context _→_ gender bias in Eqs. (3) and (4) is applied to T5 augmented captions. In this way, the textual context becomes skewed toward the new gender. Some synthesized samples can be found in Figure 3 (refer to Merged). 

## **4. Debiasing caption generator** 

DCG is designed to mitigate the two types of gender bias in an input caption to generate a debiased caption. 

**Architecture** DCG has an encoder-decoder architecture. The encoder is a Transformer-based vision-and-language model [23] that takes an image and text as input and outputs a multi-modal representation. The decoder is a Transformer-based language model [32] that generates text given the encoder’s output. The encoder’s output is fed into the decoder via a cross-attention mechanism [38]. 

**Training** Let _D[⋆]_ = _A_ CG _∪A_ GC _∪A_ = _{_ ( _I, y[⋆]_ ) _}_ denote the set of synthetic biased captions where _y[⋆]_ is a biased 

Table 1. Dataset construction. Swap denotes synthesized captions by Gender-swapping (Section 3.1). T5 denotes synthesized captions by T5-generation (Section 3.2). Ratio represents the ratio of the number of each type of biased data. 

|Synthesis|Synthesis|method|||
|---|---|---|---|---|
|Swap|T5|Merged|Ratio|Num. sample|
|�|�|-|1:1:0|57,284|
|-|�|�|0:1:1|114,568|
|�|�|�|1:2:1|114,568|



caption. When training DCG, given a ( _I_ , _y[⋆]_ ) pair, we first mask 100 _η_ percent of words in the input caption. The aim is to add noise to the input sentence so DCG can see the image when refining the input caption, avoiding outputting the input sentence as it is by ignoring the image. The masked caption is embedded to _y_ ¯ by word embedding and position embedding. The input image _I_ is embedded to _I_[¯] through linear projection and position embedding. _y_ ¯ and _I_[¯] are fed into the DCG encoder, and the output representation of the encoder is inputted to the DCG decoder via a cross-attention mechanism. DCG is trained to recover the original caption _y_ with a cross-entropy loss _L_ ce as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0005-03.png)


where _p_ is conditioned on the precedently generated tokens, and _I_ and _y[⋆]_ through the cross-attention from the encoder. The trained DCG learns to mitigate two types of biases that lie in the input-biased captions. 

**Inference** We apply the trained DCG to the output captions of captioning models. Let _y_ c denote a generated caption by an image captioning model. As in training, given a pair of ( _I_ , _y_ c), we first mask 100 _η_ percent of words in the input caption. Then, DCG takes the masked caption and image and generates a debiased caption. DCG can be used on top of any image captioning models and does not require training in captioning models to mitigate gender bias. 

## **5. Experiments** 

**Dataset** We use MSCOCO captions [9]. For training captioning models, we use the MSCOCO training set that contains 82 _,_ 783 images. For evaluation, we use a subset of the MSCOCO validation set, consisting of 10 _,_ 780 images, that come with binary gender annotations from [65]. Each image has five captions from annotators. 

For synthesizing biased captions with BCS, we use the MSCOCO training set. The maximum number of synthetic captions by Gender-swapping is capped by _|F_ PG( _Dg_ ) _|_ = 

28 _,_ 642, while T5-generation and Merged can synthesize an arbitrary number of captions by sampling _M_ . We synthesize captions so that the number of captions with gender swapping ( _i.e_ ., Gender-Swapping and Merged) and T5generation can be balanced as in Table 1. 

**Bias metrics** We mainly rely on three metrics to evaluate our framework: 1) **LIC** [18], which compares two gender classifiers’ accuracies trained on either generated captions by a captioning model or human-written captions. Higher accuracy of the classifier trained on a model’s predictions means that the model’s captions contain more information to identify the gender in images, indicating gender _→_ context bias amplification, 2) **Error** [8], which measures the gender misclassification ratio of generated captions. We consider Error to evaluate context _→_ gender bias whereas it does not directly measure this bias (discussed in the appendix) , and 3) **BiasAmp** [66], a bias amplification measurement based on word-gender co-occurrence, which is possibly the cause of gender _→_ context bias. More details about these bias metrics are described in the appendix. 

**Captioning metrics** The accuracy of generated captions is evaluated on reference-based metrics that require humanwritten captions to compute scores, specifically BLEU-4 [30], CIDEr [48], METEOR [12], and SPICE [1]. While those metrics are widely used to evaluate captioning models, they often suffer from disagreements with human judges [16]. Thus, we also use a reference-free metric, CLIPScore [16], that relies on the image-text matching ability of the pre-trained CLIP. CLIPScore has been shown to have a higher agreement with human judgment than reference-based metrics. 

**Captioning models** We evaluate two standard types of captioning models as baselines: 1) CNN encoder-LSTM decoder models (NIC [49], SAT [58], FC [36], Att2in [36], and UpDn [2]) and 2) state-of-the-art Transformer-based models (Transformer [47], OSCAR [25], ClipCap [28], and GRIT [29]). Note that most of the publicly available pretrained models are trained on the training set of the Karpathy split [21] that uses the training and validation sets of MSCOCO for training. As we use the MSCOCO validation set for our evaluation, we retrain the captioning models on the MSCOCO training set only. 

**Debiasing methods** As debiasing methods, we compare LIBRA against Gender equalizer [8]. Gender equalizer utilizes extra segmentation annotations in MSCOCO [26], which are not always available. The method is not applicable to captioning models that use object-based visual features such as Faster R-CNN [35] because the pre-trained detector’s performance drops considerably for human-masked 

Table 2. Gender bias and captioning quality for several image captioning models. Green/red denotes LIBRA mitigates/amplifies bias with respect to the baselines. For bias, lower is better. For captioning quality, higher is better. LIC and BiasAmp are scaled by 100. Note that CLIPScore for ClipCap can be higher because CLIPScore and ClipCap use CLIP [31] in their frameworks. 

|Model|Gender bias _↓_<br>LIC<br>Error<br>BiasAmp|Captioning quality _↑_|
|---|---|---|
|||BLEU-4<br>CIDEr<br>METEOR<br>SPICE<br>CLIPScore|
|NIC [49]|0.5<br>23.6<br>1.61|21.9<br>58.3<br>21.6<br>13.4<br>65.2|
|+LIBRA|-0.3<br>5.7<br>-1.47|24.6<br>72.0<br>24.2<br>16.5<br>71.7|
|SAT [58]|-0.3<br>9.1<br>0.92|34.5<br>94.6<br>27.3<br>19.2<br>72.1|
|+LIBRA|-1.4<br>3.9<br>-0.48|34.6<br>95.9<br>27.8<br>20.0<br>73.6|
|FC [36]|2.9<br>10.3<br>3.97|32.2<br>94.2<br>26.1<br>18.3<br>70.0|
|+LIBRA|-0.2<br>4.3<br>-1.11|32.8<br>95.9<br>27.3<br>19.7<br>72.9|
|Att2in [36]|1.1<br>5.4<br>-1.01|36.7<br>102.8<br>28.4<br>20.2<br>72.6|
|+LIBRA|-0.3<br>4.6<br>-3.39|35.9<br>101.7<br>28.5<br>20.6<br>73.8|
|UpDn [2]|4.7<br>5.6<br>1.46|39.4<br>115.1<br>29.8<br>22.0<br>73.8|
|+LIBRA|1.5<br>4.5<br>-2.23|37.7<br>110.1<br>29.6<br>22.0<br>74.6|
|Transformer [47]|5.4<br>6.9<br>0.09|35.0<br>101.5<br>28.9<br>21.1<br>75.3|
|+LIBRA|2.3<br>5.0<br>-0.26|33.9<br>98.7<br>28.6<br>20.9<br>75.7|
|OSCAR [25]|2.4<br>3.0<br>1.78|39.4<br>119.8<br>32.1<br>24.0<br>75.8|
|+LIBRA|0.3<br>4.6<br>-1.95|37.2<br>113.1<br>31.1<br>23.2<br>75.7|
|ClipCap [28]|1.1<br>5.6<br>1.51|34.8<br>103.7<br>29.6<br>21.5<br>76.6|
|+LIBRA|-1.5<br>4.5<br>-0.57|33.8<br>100.6<br>29.3<br>21.4<br>76.0|
|GRIT [29]|3.1<br>3.5<br>3.05|42.9<br>123.3<br>31.5<br>23.4<br>76.2|
|+LIBRA|0.7<br>4.1<br>1.57|40.5<br>116.8<br>30.6<br>22.6<br>75.9|



images.[6] In the experiment, we apply Gender equalizer and LIBRA to debias NIC+, which is a variant of NIC with extra training on images of female/male presented in [8]. 

For LIBRA, we use _δ_ = 0 _._ 2. The vision-and-language encoder of DCG is Vilt [23], and the decoder is GPT-2 [32]. Unless otherwise stated, we use the combination of biased data composed of T5-generation and Merged in Table 1. We set _η_ = 0 _._ 2 and conduct ablation studies of the settings in Section 5.4 and the appendix. 

## **5.1. Bias mitigation analysis** 

We apply LIBRA on top of all the captioning models to evaluate if it mitigates the two types of gender biases. We also report caption evaluation scores based on captioning metrics. Results are shown in Table 2. We summarize the main observations as follows: 

**LIBRA mitigates gender** _→_ **context bias.** The results on LIC show that applying LIBRA consistently decreases gender _→_ context bias in all the models. We show some examples of LIBRA mitigating bias in Figure 1 (b). For example, in the second sample from the right, the baseline, UpDn [2], produces the incorrect word, _suit_ . The word _suit_ is skewed toward men, co-occurring with men 82% of the time in the MSCOCO training set. LIBRA changes _suit_ to 

> 6Faster-RCNN mAP drops from 0 _._ 41 to 0 _._ 37, and for the person class recall drops from 0 _._ 79 to 0 _._ 68. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0006-08.png)


**----- Start of picture text -----**<br>
GT gender: Male<br>OSCAR<br>a  man  flying through the air while riding a skateboard<br>+LIBRA<br>a  girl  glides through the air while riding a skateboard<br>GT gender: Female<br>OSCAR<br>a  man  riding a wave on top of a surfboard<br>+LIBRA<br>a  woman  riding a wave on top of a surfboard<br>**----- End of picture text -----**<br>


Figure 4. Gender misclassification of LIBRA (Top). Gender misclassification of OSCAR [25] (Bottom). GT gender denotes ground-truth gender annotation in [65]. 

_jacket_ , mitigating gender _→_ context bias. Besides, in some cases where LIC is negative ( _i.e_ ., NIC, SAT, FC, Att2in, and ClipCap), the gender _→_ context bias in the generated captions by LIBRA is less than those of human annotators. In the appendix, we show some examples that LIBRA generates less biased captions than annotators’ captions. 

The results of BiasAmp, which LIBRA consistently reduces, show that LIBRA tends to equalize the skewed wordgender co-occurrences. For example, LIBRA mitigates the co-occurrence of the word _little_ and women from 91% in captions by OSCAR to 60%. Results on BiasAmp support 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0007-00.png)


References - A man is standing next to his ice cream truck - A man standing in front of a white ice cream truck - A man is standing in front of his ice cream vehicle - A ice cream truck parked on the side of the road with the driver standing beside it - A man is standing next to an ice cream truck 

Baseline +LIBRA A man **standing** in front of a white truck A man **posing** in front of a white truck BLEU-4: 79.6 METEOR: 43.8 BLEU-4: 47.5 METEOR: 33.7 SPICE: 42.9 CLIPScore: 74.6 SPICE: 30.8 CLIPScore: 76.7 

Figure 5. CLIPScore [16] vs. reference-based metrics [1, 12, 30]. References denote the ground-truth captions written by annotators. Bold words in the generated captions mean the difference between baseline and LIBRA. Highlighted words in references denote the words that match the bold word in the baseline. We can see that CLIPScore is more robust against word-changing. 

## the effectiveness of LIBRA regarding the ability to mitigate gender _→_ context bias. 

**LIBRA mitigates context** _→_ **gender bias in most models.** The Error scores show that LIBRA reduces gender misclassification in most models except for OSCAR and GRIT (3 _._ 0 _→_ 4 _._ 6 for OSCAR, 3 _._ 5 _→_ 4 _._ 1 for GRIT). We investigate the error cases when LIBRA is applied to OSCAR and find that gender misclassification of LIBRA is often caused by insufficient evidence to identify a person’s gender. For instance, in the top example in Figure 4, the ground-truth gender annotation is _male_ , and OSCAR generates _man_ although the person is not pictured properly enough to determine gender.[7] This may suggest that OSCAR learns to guess the gender based on the context, in this case, skateboard[8] to increase gender classification accuracy. However, this causes context _→_ gender bias for images with a gendercontext combination rarely seen in the dataset ( _e.g_ ., womensurfing). In Figure 4 (bottom), OSCAR predicts incorrect gender for the image with a male-biased context.[9] In the appendix, we discuss possible solutions for reducing gender misclassification without relying on the context. 

**LIBRA is good at CLIPScore.** The results of the captioning metrics show that CLIPScore is better or almost as high as the baselines when applying LIBRA. As CLIPScore is based on an image-caption matching score, we can confirm that LIBRA does not generate less biased sentences by producing irrelevant words to images. This observation verifies that applying LIBRA on top of the captioning models does not hurt the quality of captions. 

**CLIPScore versus other metrics.** While LIBRA works well on CLIPScore, the score in the reference-based met- 

> 7Previous work [5] has shown human annotators possibly annotate gender from context for images without enough cues to judge gender. 

> 8 _Skateboard_ is highly skewed toward men in the dataset, which cooccurs with men more than 90%. 

> 9 _Surfboard_ highly co-occur with men in MSCOCO. 

Table 3. Comparison with Gender equalizer [8]. Green/red denotes the bias mitigation method mitigates/amplifies bias. 

|Model|Gender bias _↓_<br>LIC<br>Error|Captioning quality _↑_|
|---|---|---|
|||SPICE<br>CLIPScore|
|NIC+ [49]|1.4<br>14.6|17.5<br>69.9|
|+Equalizer [8]<br>+LIBRA|6.8<br>7.8<br>0.4<br>5.1|16.8<br>69.9<br>18.9<br>72.7|




![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0007-12.png)


**----- Start of picture text -----**<br>
Original<br>a  man  and a baby elephant standing in the water<br>+Equalizer<br>a  woman  in a  bikini  standing next to a dog<br>+LIBRA<br>a  woman  and a baby elephant standing in the sand<br>Original<br>a man and a  woman  standing next to each other<br>+Equalizer<br>a man in a  suit  is holding a laptop<br>+LIBRA<br>a man and a  child  standing next to each other<br>**----- End of picture text -----**<br>


Figure 6. LIBRA vs. Gender equalizer [8]. 

rics decreases for some models. We examine the cause of the inconsistency between CLIPScore and reference-based metrics and find that generating words that reduce bias hurts reference-based metrics. We show an example in Figure 5. LIBRA changes _standing_ to _posing_ , which is also a valid description of the image. However, the scores of reference-based metrics substantially drop ( _e.g_ ., 79 _._ 6 _→_ 47 _._ 5 in BLEU-4). Human annotators tend to use _posing_ for women.[10] Therefore, reference-based metrics value captions that imitate how annotators describe each gender. On the other hand, LIBRA tends to change words skewed toward each gender to more neutral ones, which can be the cause of decreasing scores in the reference-based metrics. 

## **5.2. Comparison with other bias mitigation** 

We compare the performance of LIBRA and Gender equalizer [8] on NIC+ [49], following the code provided by the authors. The results are shown in Table 3. As reported in previous work [18,51], Gender equalizer amplifies gender _→_ context bias (1 _._ 4 _→_ 6 _._ 8 in LIC) while mitigating gender misclassification (14 _._ 6 _→_ 7 _._ 8 in Error). In contrast, LIBRA mitigates gender _→_ context and context _→_ gender biases, specifically 1 _._ 4 _→_ 0 _._ 4 in LIC and 14 _._ 6 _→_ 5 _._ 1 in Error. In the upper sample of Figure 6, LIBRA predicts the correct gender while not generating gender-stereotypical words. The results of the comparison with Gender equalizer highlight the importance of considering two types of 

> 10The co-occurrence of women and _posing_ is more than 60% of the time in the MSCOCO training set. 

Table 4. Comparison with image caption editing model. Bold numbers represent the best scores in ENT [40] or LIBRA. 

|Model|Gender bias _↓_<br>LIC<br>Error|Captioning quality _↑_<br>SPICE<br>CLIPScore|
|---|---|---|
|OSCAR [25]|2.4<br>3.0|24.0<br>75.8<br>21.9<br>72.8<br>23.2<br>75.7|
|+ENT [40]<br>+LIBRA|5.7<br>**2.8**<br>**0.3**<br>4.6||



Table 5. Comparison of data used for training DCG. Bold numbers denote the best scores among the types of synthetic datasets. 

|Model|Synthesis method<br>Swap<br>T5<br>Merged|Gender bias _↓_|
|---|---|---|
|||LIC<br>Error|
|UpDn [2]|-<br>-<br>-<br><br>|4.7<br>5.6|
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�|2.3<br>6.2<br>1.5<br>**4.5**<br>**1.1**<br>5.2|
|OSCAR [25]|-<br>-<br>-<br><br>|2.4<br>3.0|
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�|**-0.8**<br>6.8<br>0.3<br>**4.6**<br>0<br>5.0|



biases for gender bias mitigation. 

## **5.3. Comparison with image caption editing model** 

We compare LIBRA with a state-of-the-art image caption editing model [40] (refer to ENT). Specifically, we apply LIBRA and ENT on top of the various captioning models and evaluate them in terms of bias metrics and captioning metrics. We re-train ENT by using the captions from SAT [58] for textual features. The results for OSCAR [25] are shown in Table 4. The complete results are in the appendix. As for LIC, while LIBRA consistently mitigates gender _→_ context bias, ENT can amplify the bias in some baselines (SAT, Att2in, OSCAR, ClipCap, GRIT). Regarding Error, LIBRA outperforms in most baselines except for OSCAR and GRIT. From these observations, we conclude that a dedicated framework for addressing gender bias is necessary to mitigate gender bias. 

## **5.4. Ablations** 

We conduct ablation studies to analyze the influence of different settings of LIBRA. Here, we show the results when applying LIBRA to UpDn [2] and OSCAR [25]. The complete results of all the baselines are in the appendix. 

**Combinations of synthetic data** We compare the performance of the different dataset combinations for training DCG in Table 1. The results are shown in Table 5. The Error score of T5-generation and Merged is consistently the best among the combinations. As for LIC, the results are 

Table 6. Comparison with random perturbation. Rand. pert. denotes DCG trained on data with random perturbation. Bold numbers denote the best scores in the DCG trained on either biased captions from BCS or captions with random perturbation. 

||||
|---|---|---|
|Model|Gender bias _↓_<br>LIC<br>Error|Captioning quality _↑_<br>SPICE<br>CLIPScore|
|UpDn [2]|4.7<br>5.6|22.0<br>73.8<br>21.8<br>74.4<br>22.0<br>74.6<br>24.0<br>75.8<br>22.9<br>75.4<br>23.2<br>75.7|
|+Rand. pert.<br>+LIBRA|2.2<br>5.9<br>**1.5**<br>**4.5**||
|OSCAR [25]|2.4<br>3.0||
|+Rand. pert.<br>+LIBRA|2.0<br>5.6<br>**0.3**<br>**4.6**||



not as consistent, but still DCG trained on all types of combinations decreases the score. We chose T5-generation and Merged as it well balances LIC and Error. 

**Synthetic data evaluation** To demonstrate the effectiveness of BCS, we compare LIBRA and DCG trained on captions with random perturbation, which does not necessarily increase gender bias. In order to synthesize such captions, we randomly mask 15 percent of the tokens in the original captions in _D_ g and generate words by T5, but without using any filters in Section 3. When selecting masked tokens, we allow choosing gender words so that T5 can randomly change the gender. As a result, the synthesized captions contain incorrect words, which are not necessarily due to gender bias. We show the results in Table 6. Using biased samples from BCS to train DCG consistently produces the best results in LIC and Error. From this, we conclude that BCS, which intentionally synthesizes captions with gender biases, contributes to mitigating gender biases. 

## **6. Conclusion** 

LIBRA 11 is a model-agnostic framework to mitigate both context _→_ gender and gender _→_ context biases in captioning models. We experimentally showed that LIBRA mitigates gender bias in multiple captioning models, correcting gender misclassification caused by context and changing to less gender-stereotypical words. To do this, LIBRA synthesizes biased captions using a language model and filtering for intentionally increasing gender biases. Interestingly, the results showed these synthetic captions are a good proxy of gender-biased captions from various captioning models and facilitate model-agnostic bias mitigation. As future work, we will use LIBRA to mitigate other types of bias, such as age or skin-tone, which requires specific annotations, such as the ones in concurrent work [14], and mechanisms to identify each type of bias. 

> 11This work is partly supported by JST CREST Grant No. JPMJCR20D3, JST FOREST Grant No. JPMJFR216O, JSPS KAKENHI No. JP22K12091, and Grant-in-Aid for Scientific Research (A). 

## **References** 

- [1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _ECCV_ . Springer, 2016. 

- [2] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _CVPR_ , 2018. 

- [3] Alexander C Berg, Tamara L Berg, Hal Daume, Jesse Dodge, Amit Goyal, Xufeng Han, Alyssa Mensch, Margaret Mitchell, Aneesh Sood, Karl Stratos, et al. Understanding and predicting importance in images. In _CVPR_ . IEEE, 2012. 

- [4] Hugo Berg, Siobhan Mackenzie Hall, Yash Bhalgat, Wonsuk Yang, Hannah Rose Kirk, Aleksandar Shtedritski, and Max Bain. A prompt array keeps the bias away: Debiasing visionlanguage models with adversarial learning. In _AACL_ , 2022. 

- [5] Shruti Bhargava and David Forsyth. Exposing and correcting the gender bias in image captioning datasets and models. _arXiv preprint arXiv:1912.00578_ , 2019. 

- [6] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: Misogyny, pornography, and malignant stereotypes. _arXiv preprint arXiv:2110.01963_ , 2021. 

- [7] Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In _ACM FAccT_ , 2018. 

- [8] Kaylee Burns, Lisa Anne Hendricks, Kate Saenko, Trevor Darrell, and Anna Rohrbach. Women also snowboard: Overcoming bias in captioning models. In _ECCV_ , 2018. 

- [9] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 

- [10] Kristy Choi, Aditya Grover, Trisha Singh, Rui Shu, and Stefano Ermon. Fair generative modeling via weak supervision. In _ICML_ , 2020. 

- [11] Terrance de Vries, Ishan Misra, Changhan Wang, and Laurens van der Maaten. Does object recognition work for everyone? In _CVPR Workshops_ , 2019. 

- [12] Michael Denkowski and Alon Lavie. Meteor universal: Language specific translation evaluation for any target language. In _Workshop on statistical machine translation_ , 2014. 

- [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT (1)_ , 2019. 

- [14] Noa Garcia, Yusuke Hirota, Wu Yankun, and Yuta Nakashima. Uncurated image-text datasets: Shedding light on demographic bias. In _CVPR_ , 2023. 

- [15] Melissa Hall, Laurens van der Maaten, Laura Gustafson, and Aaron Adcock. A systematic study of bias amplification. _arXiv preprint arXiv:2201.11706_ , 2022. 

- [16] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In _EMNLP (1)_ , 2021. 

- [17] Yusuke Hirota, Yuta Nakashima, and Noa Garcia. Gender and racial bias in visual question answering datasets. In _ACM_ 

_Conference on Fairness, Accountability, and Transparency_ , 2022. 

- [18] Yusuke Hirota, Yuta Nakashima, and Noa Garcia. Quantifying societal bias amplification in image captioning. In _CVPR_ , 2022. 

- [19] Niharika Jain, Alberto Olmo, Sailik Sengupta, Lydia Manikonda, and Subbarao Kambhampati. Imperfect imaganation: Implications of gans exacerbating biases on facial data augmentation and snapchat selfie lenses. _arXiv preprint arXiv:2001.09528_ , 2020. 

- [20] Shengyu Jia, Tao Meng, Jieyu Zhao, and Kai-Wei Chang. Mitigating gender bias amplification in distribution by posterior regularization. _ACL_ , 2020. 

- [21] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In _CVPR_ , 2015. 

- [22] Zaid Khan and Yun Fu. One label, one billion faces: Usage and consistency of racial categories in computer vision. In _ACM FAccT_ , 2021. 

- [23] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In _ICLR_ . PMLR, 2021. 

- [24] Klas Leino, Emily Black, Matt Fredrikson, Shayak Sen, and Anupam Datta. Feature-wise bias amplification. In _ICLR_ , 2019. 

- [25] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _ECCV_ , 2020. 

- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In _ECCV_ , 2014. 

- [27] George A Miller. Wordnet: a lexical database for english. _Communications of the ACM_ , 1995. 

- [28] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 

- [29] Van-Quang Nguyen, Masanori Suganuma, and Takayuki Okatani. Grit: Faster and better image captioning transformer using dual visual features. In _ECCV_ . Springer, 2022. 

- [30] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In _ACL_ , 2002. 

- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- [32] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 2019. 

- [33] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._ , 2020. 

- [34] Waseem Rawat and Zenghui Wang. Deep convolutional neural networks for image classification: A comprehensive review. _Neural computation_ , 2017. 

- [35] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: towards real-time object detection with region proposal networks. _IEEE transactions on pattern analysis and machine intelligence_ , 2016. 

- [36] Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-critical sequence training for image captioning. In _CVPR_ , 2017. 

- [37] Candace Ross, Boris Katz, and Andrei Barbu. Measuring social biases in grounded vision and language embeddings. In _NAACL-HLT_ , 2021. 

- [38] Sascha Rothe, Shashi Narayan, and Aliaksei Severyn. Leveraging pre-trained checkpoints for sequence generation tasks. _Transactions of the Association for Computational Linguistics_ , 2020. 

- [39] Fawaz Sammani and Mahmoud Elsayed. Look and modify: Modification networks for image captioning. In _BMVC_ , 2019. 

- [40] Fawaz Sammani and Luke Melas-Kyriazi. Show, edit and tell: a framework for editing image captions. In _CVPR_ , 2020. 

- [41] Xudong Shen, Yongkang Wong, and Mohan Kankanhalli. Fair representation: Guaranteeing approximate multiple group fairness for unknown tasks. _TPAMI_ , 2022. 

- [42] Tejas Srinivasan and Yonatan Bisk. Worst of both worlds: Biases compound in pre-trained vision-and-language models. In _NAACL workshop_ , 2022. 

- [43] Pierre Stock and Moustapha Cisse. ConvNets and ImageNet beyond accuracy: Understanding mistakes and uncovering biases. In _ECCV_ , 2018. 

- [44] Ruixiang Tang, Mengnan Du, Yuening Li, Zirui Liu, Na Zou, and Xia Hu. Mitigating gender bias in captioning systems. In _WWW_ , 2021. 

- [45] Hamed R Tavakoli, Rakshith Shetty, Ali Borji, and Jorma Laaksonen. Paying attention to descriptions generated by image captioning models. In _ICCV_ , 2017. 

- [46] William Thong and Cees GM Snoek. Feature and label embedding spaces matter in addressing image classifier bias. In _BMVC_ , 2021. 

- [47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 

- [48] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In _CVPR_ , 2015. 

- [49] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In _CVPR_ , 2015. 

timating and mitigating gender bias in deep image representations. In _ICCV_ , 2019. 

   - [54] Zhen Wang, Long Chen, Wenbo Ma, Guangxing Han, Yulei Niu, Jian Shao, and Jun Xiao. Explicit image caption editing. In _ECCV_ , 2022. 

   - [55] Zeyu Wang, Klint Qinami, Ioannis Christos Karakozis, Kyle Genova, Prem Nair, Kenji Hata, and Olga Russakovsky. Towards fairness in visual recognition: Effective strategies for bias mitigation. In _CVPR_ , 2020. 

   - [56] Spencer Whitehead, Suzanne Petryk, Vedaad Shakib, Joseph Gonzalez, Trevor Darrell, Anna Rohrbach, and Marcus Rohrbach. Reliable visual question answering: Abstain rather than answer incorrectly. In _ECCV_ , 2022. 

   - [57] Benjamin Wilson, Judy Hoffman, and Jamie Morgenstern. Predictive inequity in object detection. _arXiv preprint arXiv:1902.11097_ , 2019. 

   - [58] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In _ICML_ , 2015. 

   - [59] Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky. Towards fairer datasets: Filtering and balancing the distribution of the people subtree in the ImageNet hierarchy. In _ACM FAccT_ , 2020. 

   - [60] Ruichen Yao, Ziteng Cui, Xiaoxiao Li, and Lin Gu. Improving fairness in image classification via sketching. In _NeurIPS Workshop_ , 2022. 

   - [61] Mark Yatskar, Luke Zettlemoyer, and Ali Farhadi. Situation recognition: Visual semantic role labeling for image understanding. In _CVPR_ , 2016. 

   - [62] Quanzeng You, Hailin Jin, Zhaowen Wang, Chen Fang, and Jiebo Luo. Image captioning with semantic attention. In _CVPR_ , 2016. 

   - [63] Yi Zhang, Junyang Wang, and Jitao Sang. Counterfactually measuring and eliminating social bias in vision-language pre-training models. In _ACMMM_ , 2022. 

   - [64] Dora Zhao, Jerone TA Andrews, and Alice Xiang. Men also do laundry: Multi-attribute bias amplification. _arXiv preprint arXiv:2210.11924_ , 2022. 

   - [65] Dora Zhao, Angelina Wang, and Olga Russakovsky. Understanding and evaluating racial biases in image captioning. In _ICCV_ , 2021. 

   - [66] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. In _EMNLP_ , 2017. 

- [50] Angelina Wang, Arvind Narayanan, and Olga Russakovsky. Revise: A tool for measuring and mitigating bias in visual datasets. In _ECCV_ , 2020. 

- [51] Angelina Wang and Olga Russakovsky. Directional bias amplification. In _ICML_ , 2021. 

- [52] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang. Racial faces in the wild: Reducing racial bias by information maximization adaptation network. In _ICCV_ , 2019. 

- [53] Tianlu Wang, Jieyu Zhao, Mark Yatskar, Kai-Wei Chang, and Vicente Ordonez. Balanced datasets are not enough: Es- 

## **Supplementary Material for Model-Agnostic Gender Debiased Image Captioning** 

Yusuke Hirota 

Yuta Nakashima Noa Garcia 

_{_ y-hirota@is., n-yuta@, noagarcia@ _}_ ids.osaka-u.ac.jp Osaka University 

This supplementary material includes: 

- Details of BCS (Appendix 1). 

- Details of bias metrics (Appendix 2). 

- Additional experiments (Appendix 3). 

- More visual examples (Appendix 4). 

- List of gender-related words (Appendix 5). 

- Limitations (Appendix 6). 

- Potential negative impact (Appendix 7). 

## **1. Details of BCS** 

In this section, we provide the details for BCS. 

## **1.1. Training gender classifier** 

The gender classifier _f_ g is trained on _D_ g. Specifically, following [6], we split the captions in _D_ g into a balanced split with an equal number of samples per female/male, having 66 _,_ 526 captions. We use BERT-base [4] with two fullyconnected layers with Leaky ReLU as _f_ g and finetune it on the balanced split. The learning rate is 1 _×_ 10 _[−]_[5] , and the training is conducted on 5 epochs. 

## **1.2. Finetuning T5** 

Following the masked language model in [4], we finetune T5 on captions in _D_ . Specifically, we mask 10% of the tokens in the original caption _y_ . Given the masked caption and the positions of masked tokens _M_ = _{m_ 1 _, . . . , m|M|}_ , T5 predicts the probability of masked tokens by[�] _m∈M[p]_[(] _[y][m][|][y][\M]_[)][, where] _[ y][\M]_[denotes all words] in an input caption _y_ except for masked tokens _{ym}_ . The sample-wise loss is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0011-19.png)


where _p_ ( _ym|y\M_ ) is the output probability of masked token _ym_ given _y\M_ from T5. 

## **1.3. Details of T5 masked word generation** 

To remove trivial modifications, we avoid generating synonyms of the masked tokens by using WordNet [11]. When selecting masked tokens, we chose 

## Gender filter 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0011-24.png)


**Original** Two men talk and eat food at a restaurant **Synthesized** Two men talk and prepare food at a table **Original** Woman in a dress in front of a couple of horses **Synthesized** Woman in a suit in front of a group of horses 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0011-26.png)


Figure 1. Synthesized captions that are removed by the gender filter. 

nouns/verbs/adjectives/adverbs based on POS tagging with NLTK [10]. 

We apply a filter to remove unnatural captions, called an authenticity filter. The authenticity filter uses a classifier that predicts whether an input sentence is synthetic or authentic. Specifically, we train classifier _f_ a with _D_ T5,g _∪Dg_ to predict whether _y_ is from _D_ T5,g or _Dg_ . Let _b ∈{_ syn _,_ auth _}_ , prediction[ˆ] _b_ is given by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0011-30.png)


where _p_ ( _B_ = _b|y_ ) is a confidence score that an input caption is _b_ . Thus, if _p_ ( _B_ = authy _|y_ ) is close to 1, _y_ is likely to be authentic. We use this classifier to filter less natural captions, _i.e_ ., 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0011-32.png)


where _α_ is a predefined threshold. We set _α_ = 0 _._ 3 and use the same classifier as _f_ g for _f_ a. 

## **1.4. Examples of gender/authenticity filter** 

In Figure 1, we show some synthesized captions that are filtered out by the gender filter. The removed captions do not increase gender bias with respect to the original captions. For instance, in the bottom example, the word _dress_ which is skewed toward women is replaced with _suit_ which 

1 

Table 1. Synthesized captions that are passed or removed by the authenticity filter 

|Original|Passed|Removed|
|---|---|---|
|Woman is sitting near a red train<br>A man wearing glasses, suit, and tie<br>A man fxing the inside of a toilet<br>Women are playing a video game|Woman is sitting near a passenger train<br>A man wearing sunglasses, hat, and tie<br>A man fxing the inside of a kitchen<br>Women are playing a baseball game|Woman sits sitting near a red train<br>A man wearing glasses, glasses, and tie<br>A man holding the inside of a toilet<br>Women are playing a video show|



Table 2. Comparison with image caption editing models. Bold numbers represent the best scores in ENT [16] or LIBRA. 

|Model|Gender bias _↓_<br>LIC<br>Error|Captioning quality _↑_|
|---|---|---|
|||BLEU-4<br>CIDEr<br>METEOR<br>SPICE<br>CLIPScore|
|NIC [18]|0.5<br>23.6|21.9<br>58.3<br>21.6<br>13.4<br>65.2|
|+ENT [16]<br>+LIBRA|**-0.3**<br>22.5<br>**-0.3**<br>**5.7**|25.8<br>67.7<br>22.5<br>14.3<br>65.3<br>24.6<br>72.0<br>24.2<br>16.5<br>71.7|
|SAT [19]|-0.3<br>9.1|34.5<br>94.6<br>27.3<br>19.2<br>72.1|
|+ENT [16]<br>+LIBRA|1.6<br>9.9<br>**-1.4**<br>**3.9**|35.3<br>96.3<br>27.3<br>19.2<br>71.1<br>34.6<br>95.9<br>27.8<br>20.0<br>73.6|
|FC [15]|2.9<br>10.3|32.2<br>94.2<br>26.1<br>18.3<br>70.0|
|+ENT [16]<br>+LIBRA|1.7<br>10.3<br>**-0.2**<br>**4.3**|32.9<br>92.0<br>26.2<br>18.2<br>69.2<br>32.8<br>95.9<br>27.3<br>19.7<br>72.9|
|Att2in [15]|1.1<br>5.4|36.7<br>102.8<br>28.4<br>20.2<br>72.6|
|+ENT [16]<br>+LIBRA|2.8<br>5.3<br>**-0.3**<br>**4.6**|37.4<br>103.2<br>28.4<br>20.3<br>71.6<br>35.9<br>101.7<br>28.5<br>20.6<br>73.8|
|UpDn [2]|4.7<br>5.6|39.4<br>115.1<br>29.8<br>22.0<br>73.8|
|+ENT [16]<br>+LIBRA|3.9<br>5.6<br>**1.5**<br>**4.5**|39.6<br>110.7<br>29.4<br>21.3<br>72.5<br>37.7<br>110.1<br>29.6<br>22.0<br>74.6|
|Transformer [17]|5.4<br>6.9|35.0<br>101.5<br>28.9<br>21.1<br>75.3|
|+ENT [16]<br>+LIBRA|4.4<br>6.8<br>**2.3**<br>**5.0**|38.6<br>107.1<br>28.9<br>20.8<br>72.9<br>33.9<br>98.7<br>28.6<br>20.9<br>75.7|
|OSCAR [9]|2.4<br>3.0|39.4<br>119.8<br>32.1<br>24.0<br>75.8|
|+ENT [16]<br>+LIBRA|5.7<br>**2.8**<br>**0.3**<br>4.6|41.4<br>113.0<br>30.2<br>21.9<br>72.8<br>37.2<br>113.1<br>31.1<br>23.2<br>75.7|
|ClipCap [12]|1.1<br>5.6|34.8<br>103.7<br>29.6<br>21.5<br>76.6|
|+ENT [16]<br>+LIBRA|3.6<br>5.1<br>**-1.5**<br>**4.5**|37.4<br>101.7<br>28.4<br>20.1<br>73.0<br>33.8<br>100.6<br>29.3<br>21.4<br>76.0|
|GRIT [13]|3.1<br>3.5|42.9<br>123.3<br>31.5<br>23.4<br>76.2|
|+ENT [16]<br>+LIBRA|5.2<br>**3.7**<br>**0.7**<br>4.1|42.8<br>120.3<br>30.8<br>22.7<br>74.0<br>40.5<br>116.8<br>30.6<br>22.6<br>75.9|



is skewed toward men. Thus the synthesized caption reduces gender bias compared to the original caption, and it is filtered out by the gender filter. 

In Table 1, we show some synthesized captions that are passed or removed by the authenticity filter. The examples show that the authenticity filter removes unnatural-sounding or grammatically incorrect captions. 

## **2. Details of bias metrics** 

**BiasAmp** As for BiasAmp, we also follow the settings presented in [6]. To compute gender-word co-occurrences, we use the top 1 _,_ 000 frequent words in _D_ . Specifically, we select nouns/verbs/adjectives/adverbs in the top 1 _,_ 000 words. Following [21], we use words that are strongly related to humans by removing words that do not appear more than 100 times with women/men words. 

Table 3. Comparison with DCG without masking input captions. Bold numbers denote the best scores in the DCG with/without masking. 

|Model|Gender bias _↓_<br>LIC<br>Error|Accuracy _↑_<br>SPICE<br>CLIPScore|
|---|---|---|
|NIC [18]|0.5<br>23.6|13.4<br>65.2<br>15.7<br>70.5<br>16.5<br>71.7<br>19.2<br>72.1<br>19.8<br>72.8<br>20.0<br>73.6<br>18.3<br>70.0<br>19.6<br>72.0<br>19.7<br>72.9<br>20.2<br>72.6<br>20.6<br>73.0<br>20.6<br>73.8<br>22.0<br>73.8<br>21.9<br>73.8<br>22.0<br>74.6<br>21.1<br>75.3<br>20.9<br>74.9<br>20.9<br>75.7<br>24.0<br>75.8<br>23.4<br>75.8<br>23.2<br>75.7<br>21.5<br>76.6<br>21.4<br>76.2<br>21.4<br>76.0<br>23.4<br>76.2<br>22.8<br>75.3<br>22.6<br>75.9|
|+DCG w/o mask<br>+LIBRA|**-2.1**<br>5.9<br>-0.3<br>**5.7**||
|SAT [19]|-0.3<br>9.1||
|+DCG w/o mask<br>+LIBRA|-1.3<br>4.0<br>**-1.4**<br>**3.9**||
|FC [15]|2.9<br>10.3||
|+DCG w/o mask<br>+LIBRA|0.5<br>4.4<br>**-0.2**<br>**4.3**||
|Att2in [15]|1.1<br>5.4||
|+DCG w/o mask<br>+LIBRA|0.7<br>**4.6**<br>**-0.3**<br>**4.6**||
|UpDn [2]|4.7<br>5.6||
|+DCG w/o mask<br>+LIBRA|1.9<br>4.8<br>**1.5**<br>**4.5**||
|Transformer [17]|5.4<br>6.9||
|+DCG w/o mask<br>+LIBRA|4.4<br>5.6<br>**2.3**<br>**5.0**||
|OSCAR [9]|2.4<br>3.0||
|+DCG w/o mask<br>+LIBRA|1.9<br>4.7<br>**0.3**<br>**4.6**||
|ClipCap [12]|1.1<br>5.6||
|+DCG w/o mask<br>+LIBRA|0.5<br>4.7<br>**-1.5**<br>**4.5**||
|GRIT [13]|3.1<br>3.5||
|+DCG w/o mask<br>+LIBRA|1.8<br>4.3<br>**0.7**<br>**4.1**||



## **3. Additional experiments** 

## **3.1. Comparison with image caption editing model** 

We compare LIBRA with a state-of-the-art image caption editing model [16] (refer to ENT). Specifically, we apply LIBRA and ENT on top of the various captioning models and evaluate them in terms of bias metrics and captioning metrics. We re-train ENT by using the captions from SAT [19] for textual features.[1] The results are shown in Table 2. As for LIC, while LIBRA consistently mitigates gender _→_ context bias, ENT can amplify the bias in some baselines (SAT, Att2in, OSCAR, ClipCap, GRIT). Regarding Error, LIBRA outperforms in most baselines except for OSCAR and GRIT. From these observations, we conclude that a dedicated framework for addressing gender bias is 

> 1In the original paper, the authors use the captions from AoANet [7]. We use SAT for training ENT as AoANet is trained on Karpathy split [8]. 

Table 4. Comparison of data used for training DCG. Bold numbers denote the best scores among the types of synthetic datasets. 

|Model|Synthesis method<br>Swap<br>T5<br>Merged|Gender bias _↓_<br>LIC<br>Error|
|---|---|---|
|NIC [18]|-<br>-<br>-|0.5<br>23.6<br>-0.1<br>7.5<br>**-0.3**<br>**5.7**<br>-0.2<br>6.2<br>-0.3<br>9.1<br>-2.0<br>6.2<br>-1.4<br>**3.9**<br>**-2.3**<br>4.8<br>2.9<br>10.3<br>0.5<br>6.5<br>-0.2<br>**4.3**<br>**-0.9**<br>5.0<br>1.1<br>5.4<br>2.0<br>6.7<br>-0.3<br>**4.6**<br>**-1.2**<br>5.5<br>4.7<br>5.6<br>2.3<br>6.2<br>1.5<br>**4.5**<br>**1.1**<br>5.2<br>5.4<br>6.9<br>**1.5**<br>6.9<br>2.3<br>**5.0**<br>2.6<br>5.8<br>2.4<br>3.0<br>**-0.8**<br>6.8<br>0.3<br>**4.6**<br>0<br>5.0<br>1.1<br>5.6<br>-1.3<br>6.8<br>-1.5<br>**4.5**<br>**-1.7**<br>5.3<br>3.1<br>3.5<br>**-0.8**<br>6.3<br>0.7<br>**4.1**<br>0<br>4.8|
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|SAT [19]|-<br>-<br>-||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|FC [15]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|Att2in [15]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|UpDn [2]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|Transformer [17]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|OSCAR [9]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|ClipCap [12]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||
|GRIT [13]|-<br>-<br>-<br><br>||
|+LIBRA<br>+LIBRA<br>+LIBRA|�<br>�<br>-<br>-<br>�<br>�<br>�<br>�<br>�||



necessary to mitigate gender bias. 

## **3.2. Analysis of masking** 

We evaluate the effectiveness of masking input captions in DCG. Specifically, we compare LIBRA with DCG whose input captions are not masked ( _i.e_ ., _η_ = 0). The results are shown in Table 3. We can see that masking the input captions of DCG consistently improves the scores on bias 

Table 5. Comparison with random perturbation. Rand. pert. denotes DCG trained on data with random perturbation. Bold numbers denote the best scores in the DCG trained on either biased captions from BCS or captions with random perturbation. 

|Model|Gender bias _↓_<br>LIC<br>Error|Accuracy _↑_|
|---|---|---|
|||SPICE<br>CLIPScore|
|NIC [18]|0.5<br>23.6|13.4<br>65.2|
|+Rand. pert. mask<br>+LIBRA|0.7<br>7.7<br>**-0.3**<br>**5.7**|16.4<br>71.5<br>16.5<br>71.7|
|SAT [19]|-0.3<br>9.1|19.2<br>72.1|
|+Rand. pert.<br>+LIBRA|**-1.5**<br>6.5<br>-1.4<br>**3.9**|19.9<br>73.4<br>20.0<br>73.6|
|FC [15]|2.9<br>10.3|18.3<br>70.0|
|+Rand. pert.<br>+LIBRA|0.2<br>6.6<br>**-0.2**<br>**4.3**|19.8<br>72.7<br>19.7<br>72.9|
|Att2in [15]|1.1<br>5.4|20.2<br>72.6|
|+Rand. pert.<br>+LIBRA|**-0.8**<br>5.9<br>-0.3<br>**4.6**|20.4<br>73.7<br>20.6<br>73.8|
|UpDn [2]|4.7<br>5.6|22.0<br>73.8|
|+Rand. pert.<br>+LIBRA|2.2<br>5.9<br>**1.5**<br>**4.5**|21.8<br>74.4<br>22.0<br>74.6|
|Transformer [17]|5.4<br>6.9|21.1<br>75.3|
|+Rand. pert.<br>+LIBRA|3.6<br>6.2<br>**2.3**<br>**5.0**|20.7<br>75.4<br>20.9<br>75.7|
|OSCAR [9]|2.4<br>3.0|24.0<br>75.8|
|+Rand. pert.<br>+LIBRA|2.0<br>5.6<br>**0.3**<br>**4.6**|22.9<br>75.4<br>23.2<br>75.7|
|ClipCap [12]|1.1<br>5.6|21.5<br>76.6|
|+Rand. pert.<br>+LIBRA|0.5<br>5.9<br>**-1.5**<br>**4.5**|21.2<br>75.8<br>21.4<br>76.0|
|GRIT [13]|3.1<br>3.5|23.4<br>76.2|
|+Rand. pert.<br>+LIBRA|1.8<br>5.6<br>**0.7**<br>**4.1**|22.4<br>75.8<br>22.6<br>75.9|



metrics, which contributes to mitigating two types of biases. 

## **3.3. Complete results of ablations** 

Here, we show the complete results of the ablations in the main paper. 

**Combinations of synthetic data** The complete results of all the baselines are shown in Table 4. As in the analysis of the main paper, the results of LIC are not as consistent while DCG trained on all types of combinations mitigate gender _→_ context bias. Regarding Error, DCG trained on T5-generation and Merged has the best results. 

**Synthetic data evaluation** Table 5 shows the results of the comparison with random perturbation. This extended table also shows that biased samples from BCS to train 

DCG produces the best results in LIC and Error in most baselines, which shows the effectiveness of BCS in mitigating gender bias. 

## **4. More visual examples** 

**Bias mitigation by LIBRA** Figure 2 shows the additional examples that LIBRA mitigates context _→_ gender or gender _→_ context bias. For instance, in the left example of (a), the word _motorcycle_ highly co-occurs with men in the MSCOCO training set,[2] and the baseline predicts the incorrect gender _man_ probably due to context _→_ gender bias. Applying LIBRA on top of the baseline results in mitigating that bias by predicting the correct gender. 

**Synthesized captions by BCS** In Figure 3, we show some additional examples of the synthesized captions by BCS. The synthesized captions contain context _→_ gender or/and gender _→_ context biases. 

**LIBRA vs. human captions** The experimental results in the main paper show that LIBRA generates less biased captions than human annotations, resulting in negative LIC scores. Figure 4 shows some visual examples that LIBRA generates more neutral words than human captions. For instance, in the left sample, both human and baseline captions contain _short skirt_ , which is women’s stereotypical words while LIBRA uses more neutral words _tennis outfit_ . 

**Error cases of LIBRA vs. state-of-the-art models** In Figure 5, we show the additional examples of the error cases of LIBRA and the state-of-the-art models, OSCAR [9] and GRIT [13]. As in the explanation in the main paper, stateof-the-art models can guess gender from the context when there is no clear evidence to identify gender, which leads to amplify context _→_ gender bias. 

**CLIPScore vs. reference-based metrics** In Figure 6, we show the additional examples that LIBRA hurts referencebased metrics by generating words that reduce bias whereas LIBRA does not hurt CLIPScore [5]. For instance, in the left example, the word _little_ is skewed toward women in the training set, and LIBRA changed it to _young_ which is the less biased word.[3] Both captions correctly describe the image, but LIBRA degrades the scores for the reference-based metrics as human annotators tend to use _little_ for women. On the other hand, CLIPScore is more robust against such word-changing. 

> 2Co-occurrence of _Motorcycle_ and men is about 2 _._ 7 times the cooccurrence of _Motorcycle_ and women. 

> 3The co-occurrence of women and _little_ is more than 70% of the time in the MSCOCO training set, while _young_ is balanced between the gender. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-00.png)


**SAT** a **man** sitting on a motorcycle **+LIBRA** 

a **woman** stands on a motorcycle 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-03.png)


**Att2in** 

A **woman** holding a teddy bear in a room **+LIBRA** 

a **man** holding a teddy bear in a crowd 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-07.png)


**ClipCap** a **man** is flying a kite in a field **+LIBRA** a **woman** is flying a kite in a field 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-09.png)


**GRIT** a **woman** standing in a kitchen **+LIBRA** a **man** standing in a kitchen 

(a) context → gender bias mitigation 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-15.png)


**NIC FC Transformer OSCAR** a man is on the beach with a **surfboard** a man wearing a **suit** and a tie a woman in a colorful **dress** a woman in a black and white **dress +LIBRA +LIBRA +LIBRA +LIBRA** a man is on the beach with a **frisbee** a man wearing a **shirt** and black tie a woman wearing a top **hat** a woman in a black and white **hat** 

(b) gender → context bias mitigation 

Figure 2. Generated captions by the baseline captioning models and LIBRA. We show the baseline suffers from context _→_ gender/gender _→_ context biases, predicting incorrect gender or incorrect word. Our proposed framework successfully modifies those incorrect words. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-19.png)


**----- Start of picture text -----**<br>
Gender-swapping<br>A  boy  in a blue shirt throwing a blue frisbee<br>T5-generation<br>A girl in a blue shirt  holding  a blue  umbrella<br>Merged<br>A  boy  in a blue shirt throwing a  yellow  frisbee<br>Original<br>A girl in a blue shirt throwing a blue frisbee<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0015-20.png)


**----- Start of picture text -----**<br>
Gender-swapping<br>A  woman  leading a small child on top of a horse<br>T5-generation<br>A man leading a small child on top of a  motorcycle<br>Merged<br>A  woman holding  a small child on top of a horse<br>Original<br>A man leading a small child on top of a horse<br>**----- End of picture text -----**<br>


Figure 3. Biased captions synthesized by BCS. 

## **5. List of gender words** 

The gender words that consist of women and men words are as below: 

woman, female, lady, mother, girl, aunt, wife, actress, princess, waitress, daughter, sister, queen, chairwoman, policewoman, girlfriend, pregnant, daughter, she, her, hers, herself, _man_ , _male_ , _father_ , _gentleman_ , _boy_ , _uncle_ , _husband_ , _actor_ , _prince_ , _waiter_ , _son_ , _brother_ , _guy_ , _emperor_ , _dude_ , _cowboy_ , boyfriend, chairman, policeman, _he_ , _his_ , _him_ , _him-_ 

_self_ and their plurals. Orange/ _Olive_ denote women / men words, respectively. 

## **6. Limitations** 

While LIBRA shows superior performance in mitigating gender bias, it also presents some limitations. 

**Attributes other than gender** Gender tends to be described in captions. However, other types of societal biases such as racial bias may not appear as explicitly mentioned 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0016-00.png)


**humans humans** a woman in a **short skirt** holding a tennis racquet a man in a **uniform** holding a catchers mitt **baseline baseline** a woman in a **short skirt** holding a tennis racquet a man in a baseball **uniform** holding a glove **+LIBRA +LIBRA** a woman in a **tennis outfit** holding a tennis racquet a man in a blue **shirt** holding a glove 

Figure 4. Comparison of captions from human annotators, baseline, and LIBRA. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0016-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0016-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0016-05.png)


**----- Start of picture text -----**<br>
GT gender: Female<br>OSCAR<br>a couple of  women  standing in a kitchen<br>+LIBRA<br>a couple of  men  standing in a kitchen<br>GT gender: Male<br>OSCAR<br>a  woman  sitting on a bed with a baby and a dog<br>+LIBRA<br>a  man  sitting on a bed with a baby and a dog<br>**----- End of picture text -----**<br>


GT gender: Male **GRIT** a **man** wearing a black jacket and a tie **+LIBRA** a **woman** wearing a black jacket and a tie GT gender: Female **GRIT** a young **boy** riding a wave on a surfboard **+LIBRA** a young **girl** riding a wave on a surfboard 

Figure 5. Gender misclassification of LIBRA (Top). Gender misclassification of OSCAR [9] or GRIT [13] (Bottom). GT gender denotes ground-truth gender annotation in [20]. 

in the text and tend to be more subtle, for which our bias mitigation method may not work properly. 

**Error for measuring context** _→_ **gender bias** Even though Error can measure one of the aspects of context _→_ gender bias where models make an incorrect prediction of gender based on the context, it does not directly evaluate this bias as it can also occur when predictions are correct but based on the context. Thus, a metric dedicated to context _→_ gender bias would be more insightful. 

**Predicting gender-neutral words** In Section 5.1 in the main paper, we showed that gender misclassification by LIBRA is likely to be caused by the deficient clues to judge gender. A possible solution to mitigate such misclassification without exploiting contextual cues would be to force the model to predict gender-neutral words such as _person_ when there is not enough information to judge gender. We leave this extension as future work. 

## **7. Potential negative impact** 

While LIBRA mitigates gender bias in the bias metrics, it does not ensure that LIBRA completely removes bias. In other words, even though LIBRA works on the bias metrics, the captioning models can still be biased. Thus, a potential negative impact of the use of LIBRA to mitigate gender bias is that the users of LIBRA may become overly confident that LIBRA eliminates gender bias and overlook the 

problem of gender bias in captioning models. We should carefully consider gender bias in image captioning as it can also exist in aspects not measured by existing metrics. 

## **References** 

- [1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In _ECCV_ . Springer, 2016. 

- [2] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _CVPR_ , 2018. 

- [3] Michael Denkowski and Alon Lavie. Meteor universal: Language specific translation evaluation for any target language. In _Workshop on statistical machine translation_ , 2014. 

- [4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In _NAACL-HLT (1)_ , 2019. 

- [5] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In _EMNLP (1)_ , 2021. 

- [6] Yusuke Hirota, Yuta Nakashima, and Noa Garcia. Quantifying societal bias amplification in image captioning. In _CVPR_ , 2022. 

- [7] Lun Huang, Wenmin Wang, Jie Chen, and Xiao-Yong Wei. Attention on attention for image captioning. In _ICCV_ , 2019. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning_images/cvpr_2023_Model-Agnostic_Gender_Debiased_Image_Captioning.pdf-0017-00.png)


**----- Start of picture text -----**<br>
References References<br>- The little girl is standing between the low shrub and  - Two men who are standing near each other.<br>the fire plug - The two men wearing suits are posing for a picture.<br>- A young person standing next to a red fire hydrant - Two men in suits are pictured standing and smiling.<br>- A little girl leaning against a red fire hydrant. - A couple of men in ties and suits with glasses on.<br>- A cute little girl standing next to a red fire hydrant. - Two men wearing glasses and wearing suits.<br>- A little girl with a stamp on her hand stands beside a<br>fire hydrant.<br>Baseline  +LIBRA Baseline  +LIBRA<br>A  little  girl standing next to a red fire hydrant A  young  girl standing next to a red fire hydrant Two men in  suits  standing next to each other Two men in  glasses  standing next to each other<br>BLEU-4: 96.2 METEOR: 51.1 BLEU-4: 83.1 METEOR: 48.6 BLEU-4: 36.9 METEOR: 41.2 ≡ BLEU-4: 0.0 METEOR: 41.2 ≡<br>SPICE: 45.2 ≡ CLIPScore: 98.8 SPICE: 45.2 ≡ CLIPScore: 99.7 SPICE: 38.1 CLIPScore: 80.0 SPICE: 28.6 CLIPScore: 80.8<br>**----- End of picture text -----**<br>


Figure 6. CLIPScore [5] vs. reference-based metrics [1, 3, 14]. References denote the ground-truth captions written by annotators. Bold words in the generated captions mean the difference between baseline and LIBRA. Highlighted words in references denote the words that match the bold word in the baseline. We can see that CLIPScore is more robust against word changing. 

- [8] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In _CVPR_ , 2015. 

- [9] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In _ECCV_ , 2020. 

- [10] Edward Loper and Steven Bird. Nltk: The natural language toolkit. _arXiv preprint cs/0205028_ , 2002. 

- [11] George A Miller. Wordnet: a lexical database for english. _Communications of the ACM_ , 1995. 

- [12] Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. _arXiv preprint arXiv:2111.09734_ , 2021. 

- [13] Van-Quang Nguyen, Masanori Suganuma, and Takayuki Okatani. Grit: Faster and better image captioning transformer using dual visual features. In _ECCV_ . Springer, 2022. 

- [14] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In _ACL_ , 2002. 

- [15] Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-critical sequence training for image captioning. In _CVPR_ , 2017. 

- [16] Fawaz Sammani and Luke Melas-Kyriazi. Show, edit and tell: a framework for editing image captions. In _CVPR_ , 2020. 

- [17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_ , 2017. 

- [18] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In _CVPR_ , 2015. 

- [19] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In _ICML_ , 2015. 

- [20] Dora Zhao, Angelina Wang, and Olga Russakovsky. Understanding and evaluating racial biases in image captioning. In _ICCV_ , 2021. 

- [21] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gender bias amplification using corpus-level constraints. In _EMNLP_ , 2017. 

