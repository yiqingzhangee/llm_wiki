Published as a conference paper at ICLR 2025 

## CAPTURED BY CAPTIONS: ON MEMORIZATION AND ITS MITIGATION IN CLIP MODELS 

**Wenhao Wang**[1] **, Adam Dziedzic**[1] **, Grace C. Kim**[2] **, Michael Backes**[1] **, Franziska Boenisch**[1] _[∗]_ 1CISPA, 2Georgia Institute of Technology 

## ABSTRACT 

Multi-modal models, such as CLIP, have demonstrated strong performance in aligning visual and textual representations, excelling in tasks like image retrieval and zero-shot classification. Despite this success, the mechanisms by which these models utilize training data, particularly the role of memorization, remain unclear. In uni-modal models, both supervised and self-supervised, memorization has been shown to be essential for generalization. However, it is not well understood how these findings would apply to CLIP, which incorporates elements from both supervised learning via captions that provide a supervisory signal similar to labels, and from self-supervised learning via the contrastive objective. To bridge this gap in understanding, we propose a formal definition of memorization in CLIP (CLIPMem) and use it to quantify memorization in CLIP models. Our results indicate that CLIP’s memorization behavior falls between the supervised and selfsupervised paradigms, with "mis-captioned" samples exhibiting highest levels of memorization. Additionally, we find that the text encoder contributes more to memorization than the image encoder, suggesting that mitigation strategies should focus on the text domain. Building on these insights, we propose multiple strategies to reduce memorization while at the same time improving utility—something that had not been shown before for traditional learning paradigms where reducing memorization typically results in utility decrease. 

## 1 INTRODUCTION 

Multi-modal models, such as CLIP (Radford et al., 2021), have demonstrated strong performance in representation learning. By aligning visual and textual representations, these models achieve state-ofthe-art results in tasks like image retrieval (Baldrati et al., 2022a;b), visual question answering (Pan et al., 2023; Song et al., 2022), and zero-shot classification (Radford et al., 2021; Ali & Khan, 2023; Wang et al., 2023; Zhang et al., 2022). Despite these successes, the mechanisms by which multi-modal models leverage their training data to achieve good generalization remain underexplored. 

In uni-modal setups, both supervised (Feldman, 2020; Feldman & Zhang, 2020) and selfsupervised (Wang et al., 2024b), machine learning models have shown that their ability to _memorize_ their training data is essential for generalization. It was indicated that, in supervised learning, memorization typically occurs for mislabeled samples, outliers (Bartlett et al., 2020; Feldman, 2020; Feldman & Zhang, 2020), or data points that were seen towards the end of training (Jagielski et al., 2022), while in self-supervised learning, high memorization is experienced particularly for atypical data points (Wang et al., 2024b). However, it is unclear how these findings extend to models like CLIP which entail elements from both supervised learning (through captions as supervisory signals) and self-supervised learning (through contrastive loss functions). 

Existing definitions of memorization offer limited applicability to CLIP and therefore cannot fully address the gap in understanding. The standard definition from supervised learning (Feldman, 2020) relies on one-dimensional labels and the model’s ability to produce confidence scores for these labels, whereas CLIP outputs high-dimensional representations. While the SSLMem metric (Wang et al., 2024b), developed for self-supervised vision models, could, in principle, be applied to CLIP’s vision encoder outputs, it neglects the text modality, which is a critical component of CLIP. Additionally, 

> _∗_ Correspondence to boenisch@cispa.de 

1 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0002-01.png)


**----- Start of picture text -----**<br>
An item of a vase with something  A group of people standing on top  A girl in pink sweater putting a blue  A yellow and blue fire hydrant<br>inside of it. of a field together. umbrella over a yellow fire hydrant. surrounded by leaves<br>two males and a female in a red top  a pole with some yellow lights in  Closeup of two street signs that read  A sign is displayed on a pole<br>holding some flowers front of a narrow building "Airport Pkwy" and "Karmill Ave." that says bump.<br>a parking lot with a bunch of cars  A sign with Oriental writing and the  A man in maroon shirt standing<br>car.A woman opening the trunk of her  in it words saying Hyatt on the Bund. next to a stainless steel refrigerator.<br>a sleeping toddler laying on a womans shoulder  A person is taken in this very picture. A pink Hello Kitty microwave on a store shelf. A blue street sign that reads "Thelonius Monk Circle."<br>An open point of view of a room with various things all around I am unable to see an image above A sign saying "Don't Honk, $350 Penalty" on a pole. A group of artistic surfboards are displayed in a tent.<br>(a) Most Memorized: CLIPMem  >  0.89 (b) Least Memorized: CLIPMem  ≈ 0.0<br>**----- End of picture text -----**<br>


Figure 1: **Examples of data with different levels of memorization.** Higher memorization scores indicate stronger memorization. We observe that atypical or distorted images, as well as those with incorrect or imprecise captions, experience higher memorization compared to standard samples and easy-to-label images with accurate captions. Results are obtained on OpenCLIP (Ilharco et al., 2021), with encoders based on the ViT-Base architecture trained on the COCO dataset. 

measuring memorization in only one modality, or treating the modalities separately, risks diluting the signal and under-reporting memorization. Our experimental results, as shown in Section 4.3, confirm this concern. Therefore, new definitions of memorization tailored to CLIP’s multi-modal nature are necessary. 

The only existing empirical work on quantifying memorization in CLIP models (Jayaraman et al., 2024) focuses on Déjà Vu memorization (Meehan et al., 2023), a specific type of memorization. The success of their method relies on the accuracy of the integrated object detection method and on the availability of an additional public dataset from the same distribution as CLIP’s training data, limiting practical applicability. To overcome this limitation, we propose _CLIPMem_ that measures memorization directly on CLIP’s output representations. Specifically, it compares the alignment— _i.e.,_ the similarity between representations—of a given image-text pair in a CLIP model trained with the pair, to the alignment in a CLIP model trained on the same data but without the pair. 

In our empirical study of memorization in CLIP using CLIPMem, we uncover several key findings. First, examples with incorrect or imprecise captions ("mis-captioned" examples) exhibit the highest levels of memorization, followed by atypical examples, as illustrated in Figure 1. Second, removing these samples from training yields significant improvements in CLIP’s generalization abilities. These findings are particularly noteworthy, given that state-of-the-art CLIP models are usually trained on large, uncurated datasets sourced from the internet with no guarantees regarding the correctness of the text-image pairs. Our results highlight that this practice not only exposes imprecise or incorrect data pairs to more memorization, often recognized as a cause for increased privacy leakage (Carlini et al., 2019; 2021; 2022; Song et al., 2017; Liu et al., 2021), but that it also negatively affects model performance. Furthermore, by disentangling CLIP’s two modalities, we are able to dissect how memorization manifests within each. Surprisingly, we find that memorization does not affect both modalities alike, with memorization occurring more in the text modality than in the vision modality. Building on these insights, we propose several strategies to reduce memorization while simultaneously improving generalization—a result that has not been observed in traditional supervised or self-supervised learning, where any reduction of memorization causes decreases in performance. Finally, at a deeper level, our analysis of the model internals, following Wang et al. (2024a), shows that CLIP’s memorization behavior sits between that of supervised and self-supervised learning. Specifically, neurons in early layers are responsible for groups of data points ( _e.g.,_ classes), similar to models trained using supervised learning, while neurons in later layers memorize individual data points, as seen in self-supervised learning. 

In summary, we make the following contributions: 

- We propose CLIPMem, a metric to measure memorization in multi-modal vision language models. 

2 

Published as a conference paper at ICLR 2025 

- Through extensive evaluation, we identify that "mis-captioned" and "atypical" data points experience the highest memorization, and that the text encoder is more responsible for memorization than the image encoder. 

- Based on our insights, we propose and evaluate multiple strategies to mitigate memorization in CLIP. We show that in CLIP, contrary to traditional supervised and self-supervised learning, a reduction of memorization does not need to imply a decrease in performance. 

## 2 BACKGROUND AND RELATED WORK 

**CLIP** . Contrastive Language-Image Pretraining (CLIP) (Radford et al., 2021) trains multi-modal encoders to map text-image pairs into a shared latent space with semantically equal representations. The core of CLIP is a two-encoder architecture with an image encoder _f_ img and a text encoder _f_ txt that are trained to maximize the similarity between the image and text features for correct text-image pairs, while minimizing the similarity for incorrect pairs. This is achieved using a contrastive loss function _L_ defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0003-05.png)


where sim( _·, ·_ ) is the cosine similarity, _τ_ is the temperature parameter, and _N_ is the batch size. This training makes CLIP versatile across various downstream tasks, including image classification, retrieval, captioning, and object recognition. There are different versions of CLIP. The popular Language augmented CLIP (LaCLIP) (Fan et al., 2023) augments original CLIP by introducing text augmentations during training in addition to the image augmentations (crops) performed in the original CLIP training to reduce overfitting. We study the impact of this practice on memorization and find it to be a suitable mitigation method. 

**Memorization.** Memorization refers to a model’s tendency to store specific details of individual training examples, rather than generalizing patterns across the dataset (Zhang et al., 2016; Arpit et al., 2017; Chatterjee, 2018; Feldman, 2020). This becomes problematic when models memorize sensitive data, as it has been shown to increase privacy risks (Carlini et al., 2019; 2021; 2022; Song et al., 2017). To date, memorization has been studied within _single modalities_ for supervised and self-supervised learning. In **supervised learning** , it has been shown that models tend to memorize _mislabeled_ (Feldman, 2020), _difficult_ , or _atypical_ examples (Arpit et al., 2017; Sadrtdinov et al., 2021), and that this memorization improves generalization, especially on long-tailed data (Feldman, 2020; Feldman & Zhang, 2020). Similar findings have been observed in **self-supervised learning** (SSL) in the vision domain (Wang et al., 2024b), where atypical samples experience high memorization, and a reduction of memorization in SSL encoders leads to decreased performance in various downstream tasks, such as classification, depth-estimation, and segmentation. A connection between memorization and generalization has also been observed in the language domain (Antoniades et al., 2024; Tirumala et al., 2022). In contrast to our work, these papers consider single-modality models. How those insights transfer to multi-modal models remains unclear. 

**Memorization in self-supervised learning.** Our CLIPMem builds on concepts from the SSLMem metric introduced by Wang et al. (2024b). This metric measures the memorization of an individual data point _x_ by an SSL encoder, based on the alignment of representations from augmented views of _x_ . Let _f_ : R _[n] →_ R _[d]_ be an SSL encoder trained using an SSL algorithm _A_ on an unlabeled dataset _S_ = _{xi}[m] i_ =1[.][The data augmentations are represented as][ Aug][(] _[x]_[) =] _[ {][a]_[(] _[x]_[)] _[|][a][ ∈]_[Aug] _[}]_[, where] _[ a]_[ is a] transformation function applied to the data point _x_ , mapping from R _[n] →_ R _[n]_ . The encoder’s output representation for a given data point _x_ is denoted as _f_ ( _x_ ). For a trained SSL encoder _f_ , the alignment loss for a data point _x_ is defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0003-09.png)


where _x[′] , x[′′]_ are augmented views of _x_ and _d_ ( _·, ·_ ) is a distance metric, typically the _ℓ_ 2 distance. SSLMem is then defined as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0003-11.png)


with _f_ being an SSL encoder trained with data point _x_ , and _g_ , an encoder trained without _x_ but otherwise on the same dataset. While this framework measures memorization using alignment loss 

3 

Published as a conference paper at ICLR 2025 

for single-modality encoders, this approach is unsuitable to leverage the signal over both modalities from multi-modal encoders like CLIP, as we also highlight empirically in Section 4.3. However, we can build on the main concepts from SSLMem to define a new metric that can evaluate memorization in CLIP, by considering both image and text representations, as we will detail in Section 3. 

**Memorization in CLIP** . Even though CLIP is a widely used vision-language encoder, there has been limited work on measuring memorization in CLIP. The only existing work (Jayaraman et al., 2024) applies the empirical Déjà Vu memorization framework from (Meehan et al., 2023) to CLIP. It measures memorization by computing the overlap between unique objects in potentially memorized images and their nearest neighbors—identified in the CLIP embedding space—from a public dataset. However, the reliance on external public data from the same distribution, along with the required accuracy of the object detection (which may not perform well for all samples, especially atypical ones (Kumar et al., 2023; Dhamija et al., 2020), limits the applicability of this approach. We further expand on this in Appendix A.1. In contrast, our CLIPMem operates directly on CLIP’s output representations and returns a joint score over both modalities. 

## 3 DEFINING MEMORIZATION OVER MULTI-MODAL ENCODERS 

## 3.1 PROBLEM SETUP 

Consider a single image-text pair ( _I, T_ ) from a dataset _S_ and two CLIP models: a model _f_ and a reference model _g_ , trained on dataset _S_ and _S[′]_ = _S \ {_ ( _I, T_ ) _}_ , respectively. We aim to quantify the memorization of ( _I, T_ ) in _f_ , trained on this data point, by leveraging model _g_ not trained on the data point but otherwise on the same data, in a _leave-one-out_ style of defining memorization (Feldman, 2020). We denote the image encoder in CLIP as _f_ img : Image _→_ R _[d]_ and the text encoder as _f_ txt : Text _→_ R _[d]_ . For the image-text pair ( _I_ , _T_ ), we denote with _f_ img( _I_ ) the output representation of _f_ ’s image encoder on image _I_ and with _f_ txt( _T_ ) the output representation of _f_ ’s text encoder on text _T_ . To evaluate the _alignment_ between the image and text representations, _i.e.,_ to quantify how similar the two representations are, we use cosine similarity _sim_ ( _f_ img( _I_ ) _, f_ txt( _T_ )), as defined in the original CLIP paper (Radford et al., 2021). 

## 3.2 ALIGNMENT WITH CONTRASTIVE OBJECTIVE 

During training, the contrastive objective in CLIP maximizes the cosine similarity for correct imagetext pairs while minimizing the cosine similarity for all the other _N −_ 1 incorrect pairs in any given training mini-batch with _N_ training samples. This means that for a given image _I_ and text _T_ , the training objective pulls _f_ img( _I_ ) and _f_ txt( _T_ ) closer together in the latent space, while pushing _f_ img( _I_ ) away from the representations of all other _N −_ 1 unrelated texts, and _f_ txt( _T_ ) away from all other images. Hence, the intuition is that the quality of alignment in _f_ , unlike in uni-modal self-supervised learning (Wang et al., 2024b), depends not only on the model’s ability to create well-aligned text and image representations for a given text-image pair, but also on its ability to create distant representations for the _N −_ 1 other representations. 

To formalize this intuition into a metric that quantifies the alignment of _f_ on the image-text pair ( _I, T_ ), we define _T_[�] _test_ as a set of _N −_ 1 randomly chosen testing samples that were not used in training _f_ or _g_ . Furthermore, when applicable, we denote random augmentations of the training data— _e.g.,_ text augmentations in versions like LaCLIP (Fan et al., 2023)—as _T[′] ∼_ Aug( _T_ ) for texts and _I[′] ∼_ Aug( _I_ ) for images. Then, we define the alignment score of _f_ on ( _I, T_ ) as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0004-09.png)


where high scores indicate a better alignment of _f_ on ( _I, T_ ). In case no text augmentations are applied, as in standard CLIP training, the first term is calculated only over _T_ . 

4 

Published as a conference paper at ICLR 2025 

## 3.3 DEFINING MEMORIZATION IN CLIP 

Given our definition of alignment scores, we can define our CLIPMem in a similar vein to the definition of memorization in supervised learning (Feldman, 2020), in the leave-one-out style. Given the image-text pair ( _I, T_ ) from dataset _S_ and two CLIP models, _f_ and _g_ , trained on dataset _S_ and _S[′]_ = _S \ {_ ( _I, T_ ) _}_ , respectively, we define CLIPMem as 

CLIPMem( _I, T_ ) = _A_ align( _f, I, T_ ) _−A_ align( _g, I, T_ ). (4) 

If a model _f_ has a significantly higher alignment score than model _g_ on ( _I, T_ ), this means that _f_ memorizes this data point. Note that taking the difference between _f_ and _g_ is crucial to get a solid estimate of memorization. This is because without "context", a high or low alignment score of _f_ does not express much information. The alignment of _f_ can be high without memorizing ( _I, T_ ), for example, if ( _I, T_ ) is a simple (but not memorized) training example. In this case, the reference model _g_ will also have a high score, such that the difference is again small. Thanks to this design of our CLIPMem, it will then correctly report low memorization. 

## 4 EMPIRICAL EVALUATION 

## 4.1 EXPERIMENTAL SETUP 

**Models and training.** We build our experiments on OpenCLIP (Cherti et al., 2023), an opensource Python version of Open-CLIP (Ilharco et al., 2021). The standard architecture used for the experiments builds on ViT-Base, but we also include experiments using ViT-Large. We train the model on the COCO dataset (Lin et al., 2014). Since COCO is much smaller than OpenCLIP’s standard training datasets, we reduce the training batch size to 128 and increase the epoch number from 32 to 100 to achieve similar performance. All other settings strictly follow OpenCLIP. For training DINO, as an example of an SSL vision encoder, we follow the default setting of Caron et al. (2021). The supervised model is trained as a multi-label classifier, also based on ViT-Base (with an additional fully connection layer) based on the first-level annotation captions in the COCO dataset. A full specification of our experimental setup is detailed in Appendix A.2. Additional experiments for measuring memorization on the BLIP (Li et al., 2022) model are presented in Appendix A.6. 

**Datasets.** We use COCO (Lin et al., 2014), CC3M (Sharma et al., 2018), and the YFCC100M (Thomee et al., 2016a) datasets to pre-train the OpenCLIP models. For the CC3M dataset, we randomly sample 75000 examples from the total of 2.91M data points. We evaluate the models by testing the linear probing accuracy on ImageNet (Deng et al., 2009) with an added classification layer trained on top of the output representations. We use the YFCC100M dataset to simulate an infinite data regime, _i.e.,_ using a single training run where no data point is repeated whereas we train iteratively using CC3M and COCO. 

**Measuring memorization.** We follow Wang et al. (2024b) to approximate our CLIPMem. Since training a separate pair of models for every data point whose memorization we aim to measure would be computationally intractable, we measure memorization of multiple data points at the same time. Therefore, we divide the original training set in four subsets: (1) _SS_ , data points that both model _f_ and _g_ were trained on, (2) _SC_ , data points used only for training _f_ , (3) _SI_ , data points used only for training _g_ , and (4) _SE_ , external "test" data points that none of the models was trained on. Note that _|SC|_ = _|SI |_ , such that _f_ and _g_ have the same number of training data points in total. For our experiments, following a similar approach to Wang et al. (2024b), we want to strike a balance when choosing the size of _SC_ . If the size is too large, then _f_ and _g_ might differ too much and not yield a strong memorization signal, but if it is too small, we would only have a memorization signal for too few data points. Concretely, for COCO and CC3M, we set _|SS|_ = 65000 and _|SC|_ = _|SI |_ = _|SE|_ = 5000. Memorization is reported as an average over all data points in _SC_ for model _f_ , or per individual data point in _SC_ . 

**Generating captions and images.** For generating additional captions for the training images, we rely on GPT-3.5-turbo. For each input image, we provide the representation produced by our trained OpenCLIP model and ask GPT to generate five new captions. Generated sample captions are presented in Figure 18. To generate additional images for the COCO dataset, we use Stable Diffusion v1.5 to generate five new images, one corresponding to each of the five per-image captions in the COCO dataset. Sample generated images are presented in Figure 17. 

5 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0006-01.png)


**----- Start of picture text -----**<br>
600<br>500 SC<br>400 SSIE<br>300 S S<br>200<br>Clean  SC Poisoned  SC<br>100<br>Clean Model 0.438 N/A<br>0 Poisoned Model 0.440 0.586<br>(b) Average CLIPMem scores.<br>CLIPMem<br>(a) Memorization scores across data subsets.<br>1.000.750.500.250.000.250.500.751.00<br>Frequency<br>**----- End of picture text -----**<br>


Figure 2: **Memorization with CLIPMem** . We train a CLIP model on COCO using standard image cropping and no text augmentations. (a) We present memorization scores according to CLIPMem per data subset. The significantly higher scores for _SC_ compared to _SS_ indicate that _f_ memorizes _SC_ . (b) We also study how inserting training samples with imprecise or incorrect captions ("miscaptioned") affects memorization. We refer to the model trained with correct captions as **Clean Model** , and the model trained with _SC_ containing 4500 standard canaries ( **Clean** ) and 500 mis-captioned ( **Mis-captioned** ) as **Poisoned Model** . We report CLIPMem over the different subsets of candidates. We observe that the mis-captioned samples experience a significantly higher memorization while the memorization of the clean data points is (almost) not affected. 

## 4.2 STUDYING MEMORIZATION USING CLIPMEM 

We first set out to analyze the general memorization in CLIP in order to identify which data points are memorized. To do this, we quantify CLIPMem over the different training subsets. Our results are presented in Figure 2a. In particular, we observe that CLIPMem for _SC_ , the data points only used to train model _f_ , is significantly higher than for _SS_ , the data points shared between the two models. Memorization for _SS_ is comparable to that for _SE_ , _i.e.,_ the external data not seen during training, indicating that _f_ does not memorize these samples. The data in _SI_ causes negative CLIPMem scores, indicating that this data is memorized by _g_ , not by _f_ . This is the expected behavior according to the definition of our metric. In Appendix A.4, we additionally highlight that memorization increases with model size, _i.e.,_ CLIP based on ViT-Large has a higher overall memorization with an average of 0 _._ 457 while CLIP based on ViT-Base only reaches 0 _._ 438 on average. 

Additionally, we analyze individual data points according to their reported CLIPMem. We give examples of highly memorized data points in CLIP in Figure 1 and more highly vs. little memorized samples in Figures 13,14,15, and 16 in Appendix A.9. Overall, the samples with high CLIPMem, _e.g.,_ in Figure 1 seem to be difficult examples and examples with imprecise or incorrect captions whereas the samples with low CLIPMem are simpler and (potentially consequently) more precisely captioned. In Appendix A.5, we show that these findings also hold when we operate in the _infinite data regime_ , _i.e.,_ when we perform only a single training run where no data point is repeated. 

Motivated by this insight and by observations from supervised learning where it was shown that models can memorize random labels (Zhang et al., 2016) and where mislabeled data experiences highest memorization (Feldman, 2020), we test if the same effect can also be observed in CLIP. Therefore, we "poison" our CLIP’s training data by randomly shuffling the captions among 500 of the 5000 candidate data points in _SC_ . Thereby, these 500 data points are "mis-captioned". We train a model based on this data and see that the mis-captioned examples experience significantly higher memorization (CLIPMem of 0.586) compared to the "clean" data points (CLIPMem of 0.440). Even though CLIP trains using a contrastive training objective, the memorization of clean data points is not significantly affected by training the model with the mis-captioned examples, as we can see by their CLIPMem that is 0.438 on the clean model and 0.440 on the poisoned model. 

## 4.3 MEASURING MEMORIZATION IN ONE MODALITY DOES NOT YIELD A STRONG SIGNAL 

To understand how important it is to take both modalities into account in our definition of CLIPMem, we set out to evaluate whether existing practical methods to measure memorization over uni-modal encoders (Wang et al., 2024b) yield a sufficiently strong memorization signal in CLIP. Therefore, we 

6 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0007-01.png)


**----- Start of picture text -----**<br>
600 700 600<br>500400 SSCS Avg. = -0.001 Avg. = 0.168 600500 S S CS  Avg.   Avg. = 0.209 = 0.013 500400 SSCS Avg. = 0.016 Avg. = 0.438<br>400<br>300 300 300<br>200 200 200<br>100 100 100<br>0 0 0<br>SSLMem SSLMem CLIPMem<br>(a) SSLMem (Img). (b) SSLMem (Text). (c) CLIPMem.<br>1.000.750.500.250.000.250.500.751.00 1 .00 0 .75 0 .50 0 .250.000.250.500.75 1.00 1.00 0.75 0.50 0.250.00 0.25 0.50 0.75 1.00<br>Frequency<br>**----- End of picture text -----**<br>


Figure 3: **Measuring memorization on individual modalities is not able to extract a strong signal.** (a)–(b) We measure SSLMem (Wang et al., 2024b) on the individual encoders of our CLIP model trained on COCO. (c) Our CLIPMem extracts a stronger memorization signal by using both modalities in CLIP jointly. 

apply their SSLMem to the individual encoder parts of CLIP. Since SSLMem relies on augmentations of the encoder input, we use image crops, like during CLIP training for the vision encoder, and the 5 COCO captions as augmentations for the text, like in (Fan et al., 2023). Our results in Figure 3 highlight that SSLMem and its naive adaptation to CLIP fail to yield a strong signal for memorization. In particular, there is a high overlap in scores between the non-memorized samples from _SS_ , and candidate examples for memorization _SC_ . Additionally, the highest reported memorization scores for _SC_ go up to around 0.65 (for SSLMem on the vision encoder) and 0.73 (for SSLMem on the text encoder). In contrast, our new CLIPMem is able to get a distinct signal for the candidates _SC_ with respect to _SS_ and reports a much higher memorization of 0.91. Thereby, our CLIPMem prevents under-reporting the actual memorization in CLIP. 

## 4.4 MEMORIZATION BETWEEN MODALITIES 

Our results in Figure 3 indicate that memorization is higher in CLIP’s text encoder than in the image encoder (the average SSLMem on _SC_ in the text encoder is 0 _._ 209 vs. 0 _._ 168 in the image encoder). To provide further insights into how memorization behaves between the modalities in CLIP, we first analyze the use of augmentations. We compare five cases: (1) no additional augmentations beyond the baseline (image cropping), (2) generating one image using a diffusion model for a given original caption, (3) generating five variations of each image using a diffusion model and randomly selecting one for each training iteration while keeping the caption fixed, (4) using the original image but randomly selecting one of the five COCO captions for each training iteration, and (5) randomly pairing each of the five generated images with one of the five COCO captions. 

As shown in Figure 14, there is quite a variability in the COCO captions for the same sample. Hence, some images might not fit well with the chosen training caption. This imprecise captioning can cause an increase in memorization. We observe that the effect is mitigated when using the 5 images with the 5 captions (5th case, see 

Table 1: Impact of augmentations. 

||Case|CLIPMem|Lin. Prob. Acc. (ImageNet)|
|---|---|---|---|
||1 Image, 1 Caption|0.438|63.11%_±_0.91%|
||1 Image (generated), 1 Caption<br>5 Images (generated), 1 Caption|0.428<br>0.424|63.97%_±_0.79%<br>64.60%_±_0.82%|
||1 Image, 5 Captions<br>5 Images (generated), 5 Captions|0.423<br>0.417|64.88%_±_0.83%<br>64.79%_±_0.99%|



Table 1). This phenomenon results most likely from the increased number of possible image-text pairs (25), such that individual incorrect or imprecise pairs are not seen so often during training. For the third case, _i.e.,_ row three in Table 1, we generate five images with a diffusion model based on all five captions per image from the COCO dataset. However, as we only use the first caption during training, this would introduce many mis-captioned images which significantly lowers performance and increases memorization. To avoid this problem, we removed 6000 mis-captioned samples. 

Our results in Table 1 highlight that augmenting text during training reduces memorization and increases performance more than augmenting images. However, applying augmentations of both text and images strikes the right balance between the reduction in memorization and the increase in performance. In fact, applying both augmentations reduces memorization most significantly. Overall, these results indicate that memorization in CLIP’s is tightly coupled to the captions assigned to the training images with imprecise captions having a destructive effect on CLIP performance and memorization. 

7 

Published as a conference paper at ICLR 2025 

## 4.5 RELATION TO CLIP MEMORIZATION TO (SELF-)SUPERVISED MEMORIZATION 

We further provide insights on whether CLIP’s memorization behavior is more alike to the one of supervised learning or SSL. This question is highly interesting since the captions in CLIP can be considered as a form of labels, like in supervised learning, whereas the contrastive training objective on the dataset resembles more SSL. We perform two experiments to gain a better understanding of the memorization behavior of CLIP with respect to supervised learning and SSL. 

First, we compare an SSL vision encoder pair _f_ and _g_ with the same architecture as CLIP’s vision encoder but trained from scratch on COCO using DINO, _i.e.,_ standard SSL training. We train _f_ and _g_ using the same candidates as the pair of CLIP models in our previous experiments. Then, we use the SSLMem metric from Wang et al. (2024b) to quantify memorization in the CLIP vision encoder and the SSL encoder, respectively. The CLIP vision encoder has a significantly lower SSLMem than the SSL encoder (0.209 vs. 0.279). Hence, CLIP vision encoders experience lower SSL memorization than SSL trained encoders. To further investigate the difference, we also report the overlap between the top 10% memorized samples between the two models, measured according to SSLMem. With an overlap of only 47 out of 500 (9.4%) samples, we find that CLIP memorizes significantly different samples than SSL encoders. Wang et al. (2024b) had performed a similar experiment on SSL vs. supervised learning and found that the two paradigms also lead to different samples being memorized. While this is, on the one hand, an effect of the different objective function, the difference between the memorized samples in CLIP and SSL is likely also closely connected to the additional captions that CLIP takes into account. While SSL-trained encoders can memorize atypical images, CLIP encoders can memorize typical images when they have an atypical, imprecise, or incorrect caption. 

Additionally, we compare the memorization behavior of CLIP against supervised and SSL-trained models on the neuron-level. Therefore, we train two additional ViTBase models on COCO using supervised training and SSL training with DINO. Then, we apply the UnitMem metric (Wang et al., 2024a) to measures how much individual neurons memorize individual samples from the training data. A high UnitMem suggests that neurons highly memorize individual data points instead of groups/classes of points. It had been shown that supervised learning causes neurons in lower layers to experience low UnitMem, _i.e.,_ being responsible for learning joint groups of data points, 

CLIP LP. Acc. = 63.11% 0.45 ~~SSL LP. Acc. = 61.45%~~ Supervised Avg. precision = 79.38% 0.40 BLIP LP. Acc. = 66.38% 0.35 0.30 0.25 1 2 3 4 5 6 7 8 9 101112 Layer 

Figure 4: **UnitMem metric: CLIP is between supervised and SSL models.** 

while neurons in later layers highly memorize individual data points. In contrast, for SSL, UnitMem was shown to remain relatively constant over layers with neurons in lower layers also being able to memorize individual data points. This difference was attributed to the different objective functions where supervised learning’s cross entropy loss pulls together data points from the same class, whereas SSL’s contrastive loss leads to individual data points being pushed away from each other (Wang et al., 2024a). Our results in Figure 4 highlight that CLIP, in terms of its memorization behavior, is between supervised learning and SSL. At the lower layers, it is much less selective than models trained with SSL, _i.e.,_ it focuses on groups of data points rather than memorizing individual data points, similar to supervised learning. Yet, in later layers, CLIP becomes more selective than SSL, _i.e.,_ it memorizes individual data points more in individual neurons, but still less than supervised learning which there has a very high average per-layer UnitMem. 

## 4.6 MITIGATING MEMORIZATION WHILE MAINTAINING GENERALIZATION 

The experiments from Table 1 suggest that using augmentations during training can improve generalization while also reducing memorization. This is an unexpected synergy since for both supervised learning (Feldman, 2020) and SSL (Wang et al., 2024b), generalization was shown to decline when memorization decreases. To further study the impact of mitigating memorization in CLIP on downstream generalization, we explore two orthogonal strategies for "augmenting" the text modality during CLIP training, first in the input space and second directly in the embedding space. Additionally, we analyze the effect of removing memorized samples from training. 

**Multiple captions.** We vary the number of captions used during training and report fine-grained insights into the resulting memorization and downstream performance in Figure 5a. Our results 

8 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0009-01.png)


**----- Start of picture text -----**<br>
0.4400 0.660 0.45 0.67<br>0.4375 0.655 0.66<br>0.4350 0.650 0.44 0.65<br>0.4325 0.645 0.64<br>0.43<br>0.4300 0.640 0.63<br>0.4275 0.635 0.42 0.62<br>0.4250 0.630 0.61<br>0.4225 0.625 0.41 0.60<br>0.4200 0.620<br>1 2 3 4 5<br># of Caption Used Noise Strength (Std.)<br>(a) Different numbers of captions. (b) Noising text embedding during training.<br>Figure 5: Mitigating memorization in CLIP improves downstream generalization. We<br>CLIP models with different "augmentations" in the textual domain. (a) We use multiple captions<br>for the same image during training. (b) We directly noise the text embeddings during the training<br>using Gaussian noise with a mean of 0 and different standard deviations (adding the Gaussian noise<br>(0 ,  0 . 15) gives us the sweet spot with the smallest memorization and highest performance). gives us the sweet spot with the smallest memorization and highest performance).<br>strategies successfully reduce memorization while improving performance.<br>0.67 Random Random<br>0.68<br>0.66 Clip Similarity Cli p  similarity<br>0.65 ClipMem Based 0.67 ClipMem Based<br>0.64<br>0.66<br>0.63<br>0.62 0.65<br>0.61<br>0.60 0.64<br># of Sample removed # of Sample removed<br>(a) CLIP trained on COCO. (b) CLIP trained on CC3M.<br>0 50 100 200 400 80016003200 0 50 100 200 400 80016003200<br>0 0.010.050.100.150.200.250.30<br>Accuracy<br>Accuracy<br>Memorization<br>Memorization<br>Linear probing Acc. Linear probing Acc.<br>**----- End of picture text -----**<br>


Figure 5: **Mitigating memorization in CLIP improves downstream generalization.** We train CLIP models with different "augmentations" in the textual domain. (a) We use multiple captions for the same image during training. (b) We directly noise the text embeddings during the training using Gaussian noise with a mean of 0 and different standard deviations (adding the Gaussian noise _N_ (0 _,_ 0 _._ 15) gives us the sweet spot with the smallest memorization and highest performance). gives us the sweet spot with the smallest memorization and highest performance). Both strategies successfully reduce memorization while improving performance. 

Figure 6: **Removing memorized samples according to CLIPMem has a stronger influence on the linear probing accuracy than removing random data points.** Removing the mislabeled samples based on CLIPMem improves the performance significantly, followed by a sharper drop when removing atypical samples. 

highlight the trend that the more captions are used during training, the lower memorization and the higher the linear probing accuracy. Our additional results in Table 3 highlight also that choosing all captions equally often is beneficial for utility while keeping memorization roughly the same. Since not in every dataset, multiple captions are available, we experiment with generating these captions with a language model. Our results in Table 7 where we train CLIP with captions generated by GPT3.5 show that the results both in terms of utility and memorization are extremely similar to the original captions, making this improved training strategy widely applicable. Our findings that modifying the text during training can reduce memorization align with the insights presented by Jayaraman et al. (2024). For datasets where only single captions are available, they proposed _text randomization_ , _i.e.,_ masking out a fraction of tokens during training as a mitigation for their Déjà Vu memorization. In contrast to our GPT3.5-generated captions, this masking, however, causes a drop in performance when mitigating memorization. We hypothesize that this is due to the higher distribution shift introduced by the masked tokens. 

**Noising the text embedding during training.** To overcome such shortcomings altogether and avoid any inherent distribution shifts, we propose to perform the "augmentations" directly in the embedding space. More precisely, we experiment with an approach where, during training, before calculating the cosine similarity between text and image embeddings for the contrastive loss, we add small amounts of Gaussian noise to the text embeddings. Our results in Figure 5b and Table 8 highlight that this strategy is highly effective in reducing memorization while improving downstream generalization. 

**Removing memorized samples.** Finally, we investigate the effect of removing memorized samples to understand how it impacts downstream performance. We perform an additional experiment where 

9 

Published as a conference paper at ICLR 2025 

we first train a CLIP model, then identify the highest memorized training data points, remove them, and retrain on the remaining data points only. We compare this method to two baselines where we either randomly remove samples or filter out the samples with the lowest CLIP similarity between the training data points’ two modalities. We showcase the effect on the downstream linear probing accuracy on ImageNet in Figure 6 with CLIP models trained on COCO and on the CCM3 dataset. For the COCO dataset, when removing up to 100 most memorized data points, we first observe a sharp increase in downstream performance in comparison to removing random samples. Then, the downstream performance starts dropping significantly more when removing memorized instead of random samples, until between 400 and 800 removed samples, the cutoff point is reached where model performance is worse when removing according to highest memorization instead of randomly. For the CC3M dataset, this cutoff occurs later, between 1600 and 3200 removed samples. While the CLIP similarity also manages to increase performance through removal, it is not as effective as CLIPMem, highlighting the value of considering memorization as a lens to identify noisy samples. This finding is significantly different than for supervised learning and SSL, where the removal of highly memorized samples _constantly_ harms performance more than the removal of random samples (Feldman, 2020; Wang et al., 2024b). We hypothesize that the effect observed in CLIP might result from the distinction between "mis-captioned" and atypical samples, where the former harm generalization while the latter help the model learn from smaller sub-populations (Feldman, 2020). We empirically support this hypothesis in Appendix A.3.1. The finding that CLIP generalization can be improved by identifying inaccurately captioned data points using our CLIPMem and removing them from training is of high practical impact, given that state-of-the-art CLIP models are usually trained on large, uncurated datasets sourced from the internet with no guarantees regarding the correctness of the text-image pairs. Overall, our results suggest that CLIPMem can help reduce memorization in CLIP while improving downstream generalization. 

## 5 CONCLUSION 

We presented CLIPMem, a formal measure to capture memorization in multi-modal models, such as CLIP. By not only quantifying memorization but also identifying _which_ data points are memorized and _why_ , we provide deeper insights into the underlying mechanisms of CLIP. Our findings highlight that memorization behavior of CLIP models falls between that of supervised and self-supervised models. In particular, CLIP highly memorizes data points with incorrect and imprecise captions, much like supervised models memorize mislabeled samples, but it also memorizes atypical examples. Furthermore, we find that memorization in CLIP happens mainly within the text encoder, which motivates instantiating mitigation strategies there. By doing so, we can not only _reduce memorization_ in CLIP but also _improve_ downstream generalization, a result that challenges the typical trade-offs seen in both supervised and self-supervised learning. 

## ACKNOWLEDGEMENTS 

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), Project number 550224287. 

## REFERENCES 

- Muhammad Ali and Salman Khan. Clip-decoder: Zeroshot multilabel classification using multimodal clip aligned representations. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 4675–4679, 2023. 

- Antonis Antoniades, Xinyi Wang, Yanai Elazar, Alfonso Amayuelas, Alon Albalak, Kexun Zhang, and William Yang Wang. Generalization vs. memorization: Tracing language models’ capabilities back to pretraining data. In _ICML 2024 Workshop on Foundation Models in the Wild_ , 2024. 

- Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. A closer look at memorization in deep networks. In _International conference on machine learning_ , pp. 233–242. PMLR, 2017. 

10 

Published as a conference paper at ICLR 2025 

- Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Conditioned and composed image retrieval combining and partially fine-tuning clip-based features. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 4959–4968, 2022a. 

- Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Effective conditioned and composed image retrieval combining clip-based features. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 21466–21474, 2022b. 

- Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. _Proceedings of the National Academy of Sciences_ , 117(48):30063–30070, 2020. 

- Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In _28th USENIX Security Symposium (USENIX Security 19)_ , pp. 267–284, 2019. 

- Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In _30th USENIX Security Symposium (USENIX Security 21)_ , pp. 2633–2650, 2021. 

- Nicholas Carlini, Matthew Jagielski, Chiyuan Zhang, Nicolas Papernot, Andreas Terzis, and Florian Tramer. The privacy onion effect: Memorization is relative. _Advances in Neural Information Processing Systems_ , 35:13263–13276, 2022. 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 9650–9660, 2021. 

- Satrajit Chatterjee. Learning and memorization. In _International conference on machine learning_ , pp. 755–763. PMLR, 2018. 

- Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 2818–2829, June 2023. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pp. 248–255. Ieee, 2009. 

- Akshay Dhamija, Manuel Gunther, Jonathan Ventura, and Terrance Boult. The overlooked elephant of object detection: Open set. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 1021–1030, 2020. 

- Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with language rewrites. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), _Advances in Neural Information Processing Systems_ , volume 36, pp. 35544–35575. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/ 2023/file/6fa4d985e7c434002fb6289ab9b2d654-Paper-Conference.pdf. 

- Vitaly Feldman. Does learning require memorization? a short tale about a long tail. In _Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing_ , pp. 954–959, 2020. 

- Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. _Advances in Neural Information Processing Systems_ , 33:2881–2891, 2020. 

- Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. URL https://doi.org/10.5281/ zenodo.5143773. 

11 

Published as a conference paper at ICLR 2025 

- Matthew Jagielski, Om Thakkar, Florian Tramer, Daphne Ippolito, Katherine Lee, Nicholas Carlini, Eric Wallace, Shuang Song, Abhradeep Guha Thakurta, Nicolas Papernot, et al. Measuring forgetting of memorized training examples. In _The Eleventh International Conference on Learning Representations_ , 2022. 

Bargav Jayaraman, Chuan Guo, and Kamalika Chaudhuri. Déjà vu memorization in vision-language models, 2024. URL https://arxiv.org/abs/2402.02103. 

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 

- Nishant Kumar, Siniša Šegvi´c, Abouzar Eslami, and Stefan Gumhold. Normalizing flow based feature synthesis for outlier-aware object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 5156–5165, 2023. 

- Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In _International conference on machine learning_ , pp. 12888–12900. PMLR, 2022. 

- Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll’a r, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. _CoRR_ , abs/1405.0312, 2014. URL http://arxiv.org/abs/ 1405.0312. 

- Hongbin Liu, Jinyuan Jia, Wenjie Qu, and Neil Zhenqiang Gong. Encodermi: Membership inference against pre-trained encoders in contrastive learning. In _Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security_ , pp. 2081–2095, 2021. 

- Casey Meehan, Florian Bordes, Pascal Vincent, Kamalika Chaudhuri, and Chuan Guo. Do ssl models have déjà vu? a case of unintended memorization in self-supervised learning. _arXiv e-prints_ , pp. arXiv–2304, 2023. 

- Junting Pan, Ziyi Lin, Yuying Ge, Xiatian Zhu, Renrui Zhang, Yi Wang, Yu Qiao, and Hongsheng Li. Retrieving-to-answer: Zero-shot video question answering with frozen large language models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 272–283, 2023. 

- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

- Ildus Sadrtdinov, Nadezhda Chirkova, and Ekaterina Lobacheva. On the memorization properties of contrastive learning. _arXiv preprint arXiv:2107.10143_ , 2021. 

- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of ACL_ , 2018. 

- Congzheng Song, Thomas Ristenpart, and Vitaly Shmatikov. Machine learning models that remember too much. In _Proceedings of the 2017 ACM SIGSAC Conference on computer and communications security_ , pp. 587–601, 2017. 

- Haoyu Song, Li Dong, Weinan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 6088–6100, 2022. 

- Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: the new data in multimedia research. _Commun. ACM_ , 59(2):64–73, January 2016a. ISSN 0001-0782. doi: 10.1145/2812802. URL https: //doi.org/10.1145/2812802. 

- Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. _Communications of the ACM_ , 59(2):64–73, 2016b. 

12 

Published as a conference paper at ICLR 2025 

- Kushal Tirumala, Aram Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. Memorization without overfitting: Analyzing the training dynamics of large language models. _Advances in Neural Information Processing Systems_ , 35:38274–38290, 2022. 

- Wenhao Wang, Adam Dziedzic, Michael Backes, and Franziska Boenisch. Localizing memorization in ssl vision encoders. _arXiv preprint arXiv:2409.19069_ , 2024a. 

- Wenhao Wang, Muhammad Ahmad Kaleem, Adam Dziedzic, Michael Backes, Nicolas Papernot, and Franziska Boenisch. Memorization in self-supervised learning improves downstream generalization. In _The Twelfth International Conference on Learning Representations (ICLR)_ , 2024b. 

- Zhengbo Wang, Jian Liang, Ran He, Nan Xu, Zilei Wang, and Tieniu Tan. Improving zero-shot generalization for clip with synthesized prompts. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 3032–3042, 2023. 

- Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In _International Conference on Learning Representations_ , 2016. 

- Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free adaption of clip for few-shot classification. In _European conference on computer vision_ , pp. 493–510. Springer, 2022. 

13 

Published as a conference paper at ICLR 2025 

## A APPENDIX 

## A.1 EXTENDED BACKGROUND 

**Déjà Vu Memorization in CLIP.** The Déjà Vu memorization framework (Jayaraman et al., 2024) is the only existing other work that attempts to quantify memorization in vision-language models. It uses the text embedding of a training image caption to retrieve relevant images from a public dataset of images. It then measures the fraction of ground-truth objects from the original image that are present in the retrieved images. If the training pair is memorized, retrieved images have a higher overlap in ground truth objects, beyond the simple correlation. While valuable, several aspects warrant further consideration for broader applicability of the framework. First, its focus on object-level memorization ignores non-object information like spatial relationships or visual patterns that can also influence memorization (Feldman, 2020; Wang et al., 2024b). To perform object retrieval, the framework also relies on object detection and annotation tools, which may introduce variability based on the accuracy and robustness of these tools. Additionally, the assumption that public datasets with similar distributions to the training data are readily available may not always hold, necessitating alternative approaches. Moreover, the framework does not analyze why certain images are memorized limiting detailed analysis. Finally, while Déjà Vu must address the challenge of distinguishing between memorization and spurious correlations, CLIPMem avoids this by directly assessing memorization on the output representations of the model. One notable difference between the results of our approach and Déjà Vu’s is that their findings show that their mitigation strategies can reduce memorization, but at the cost of decreased model utility. CLIPMem, in contrast, does not observe trade-offs between memorization and performance. 

## A.2 EXTENDED EXPERIMENTAL SETUP 

**General Setup.** All the experiments in the paper are done on a server with 4 A100 (80 GB) GPUs and a work station with one RTX 4090 GPU(24 GB). We detail the setup for our model training, both CLIP and SSL (relying on DINO) in Table 2. 

Table 2: **Experimental Setup.** We provide details on our setup for encoder training and evaluation. 

||Model Training<br>CLIP<br>DINO<br>Supervised ViT|Linear Probing|
|---|---|---|
|||CLIP<br>DINO<br>Supervised ViT|
|Training Epoch<br>Warm-up Epoch<br>Batch Size<br>Optimizer<br>Learning rate<br>Learning rate Schedule|100<br>300<br>100<br>5<br>30<br>5<br>128<br>1024<br>128<br>Adam<br>AdamW<br>Adam<br>1.2e-3<br>2e-3<br>1e-3<br>Cos. Decay<br>Cos. Decay<br>Cos. Decay|45<br>45<br>45<br>5<br>5<br>5<br>4096<br>4096<br>4096<br>LARS<br>LARS<br>LARS<br>1.6<br>1.6<br>1.6<br>Cos. Decay<br>Cos. Decay<br>Cos. Decay|



**Experimental Setup for SSLMem.** To experimentally evaluate memorization using the SSLMem framework (Wang et al., 2024b), the training dataset _S_ is split into four sets: _shared set_ ( _SS_ ) used for training both encoders _f_ and _g_ ; _candidate set_ ( _SC_ ) used only for training encoder _f_ ; _independent set_ ( _SI_ ) data used only for training encoder _g_ ; and an additional _extra set_ ( _SI_ ) from the test set not used for training either _f_ or _g_ . For training encoders, encoder _f_ is trained on _SS ∪ SC_ , while encoder _g_ is trained on _SS ∪ SI_ . The alignment losses _L_ align( _f, x_ ) and _L_ align( _g, x_ ) are computed for both encoders, and the memorization score _m_ ( _x_ ) for each data point is derived as the difference between these alignment losses, normalized to a range between _−_ 1 and 1. A score of 0 indicates no memorization, +1 indicates the strongest memorization by _f_ , and _−_ 1 indicates the strongest memorization by _g_ . 

**Normalization on CLIPMem.** For improved interpretability, we normalize our CLIPMem scores to a range of [ _−_ 1 _,_ 1]. A memorization score of 0 indicates no memorization, +1 indicates the strongest memorization on CLIP model f, and _−_ 1 indicates the strongest memorization on CLIP model g. We find the normalized CLIPMem score for a dataset using the following process: For each image-text pair ( _I, T_ ), we first calculate the CLIPMem score as the difference in alignment scores between two CLIP models _f_ and _g_ . Once CLIPMem scores are computed for all data points, we normalize them by dividing each score by the range, which is the difference between the maximum and minimum scores in the dataset. Finally, we report the normalized CLIPMem score for a dataset as the average of these normalized values. 

14 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0015-01.png)


**----- Start of picture text -----**<br>
Random<br>0.70<br>ClipMem Based<br>0.69<br>0.68<br>0.67<br>0.66<br>0.65<br>0.64<br>0.63<br>0 50 100 200 400 800 1600 3200<br># of Sample removed<br>Linear probing Acc.<br>**----- End of picture text -----**<br>


Figure 7: **Removing memorized samples.** We show the effect on downstream performance in terms of ImageNet linear probing accuracy and CLIPMem for a CLIP model trained on COCO using 5 text captions instead of 1, like done in Figure 6. We observe the same trend, with the difference that the peak is at roughly 500 removed samples rather than 100. This is likely due to the increase in captions (by factor 5) that causes increase in mis-captioned samples. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0015-03.png)


**----- Start of picture text -----**<br>
0.62<br>Random<br>Memorization Based<br>0.61<br>0.60<br>0.59<br>0.58<br>0 50 100 200 400 800 1600 3200<br># of Sample removed<br>Linear probing Acc.<br>**----- End of picture text -----**<br>


Figure 8: **Removing memorized samples in supervised learning.** We train a ViT-tiny on CIFAR10 (Krizhevsky et al., 2009) using supervised learning. We use our evaluation setup with _SC_ , _SS_ , _SI_ , and _SE_ to approximate the memorization metric from Feldman (2020). We use 5000 samples in _SC_ , but before training, we flip the labels of 200 samples. We calculate memorization over all samples in _SC_ and test the linear probing accuracy with ImageNet resized to 32*32 on the representations output before the original classification layer. 

## A.3 ADDITIONAL EXPERIMENTS 

## A.3.1 MEMORIZATION VS. GENERALIZATION IN CLIP 

**Extending evaluation.** In Figure 7, we perform the same experiment as in Figure 6, but on a CLIP model trained with 5 captions instead of 1. We observe the same trend, with the difference that the peak is at roughly 500 removed samples rather than 100. This is likely due to the increase in captions (by factor 5) that causes increase in mis-captioned samples. 

**Verifying the hypothesis on memorizing mis-captioned samples through supervised learning.** We repeat the same experiment in the supervised learning setup to understand where the increase and then decrease in linear probing accuracy stems from. To test our hypothesis that it stems from "mis-captioned" samples, we "poison" our supervised model by flipping the labels of 200 data points before training. Then, we approximate the memorization metric from Feldman (2020) in our setup and remove highly memorized vs. random data points. In the same vein as in Appendix A.3.1, we first observe an increase in linear probing accuracy when removing memorized samples (instead of random samples). The peak is at roughly 200 data points, _i.e.,_ the number of deliberately mislabeled samples. Until the cutoff point at roughly 3200 examples, linear probing accuracy is still higher when removing most memorized rather than random samples, which might suggest that there are other 

15 

Published as a conference paper at ICLR 2025 

Table 3: **Using different/multiple captions during training.** We evaluate CLIPMem how memorization on different data subsets and linear probing accuracy on ImageNet differ when using 1 caption (baseline), 5 COCO captions, one chosen at random at every round (random), and 5 COCO captions, but all chosen equally often, _i.e.,_ 20 out of 100 training epochs (balanced). We observe that increasing the number of captions reduces highest memorization. Yet, only when we balance the usage of caption, also model performance increases. 

||baseline|random|balanced|
|---|---|---|---|
|Avg. CLIPMem (Top 10 samples)|0.792|0.788|0.790|
|Avg. CLIPMem (Top 20%)|0.552|0.531|0.540|
|Linear Probing Acc.|63.11%_±_0.91%|62.44%_±_1.18%|64.88%_±_0.83%|



Table 4: **The CLIPMem and linear probing accuracy of model trained with original coco captions and captions generated by GPT3.5.** For ’Single Caption’, only one caption is used during training. For ’Five Caption’, all five caption are used equally during training (every caption trained for 20 epoch out of 100). The linear probing accuracy is tested on ImageNet 

||COCO|COCO|GPT3.5|GPT3.5|
|---|---|---|---|---|
||Single Caption|Five Caption|Single Caption|Five Caption|
|CLIPMem|0.438|0.423|0.430|0.411|
|LP. Acc.|63.11%_±_0.91%|64.88%_±_0.83%|63.09%_±_1.12%|64.47_±_0.72%|



outliers or inherently mislabeled samples whose removal improves model performance. After the cutoff, we observe the behavior as observed in prior work (Wang et al., 2024b; Feldman, 2020) that reducing memorization harms generalization more than reducing random data points from training. 

## A.3.2 THE EFFECT OF CAPTIONS 

In Table 3, we show that using more captions during training reduces memorization and that by using each caption at the same frequency over the training epochs, we can additionally improve model performance. Additionally, we show that captions generated by GPT3.5 have the same effect as the original COCO captions on memorization and linear probing accuracy in Table 4. 

## A.4 THE EFFECT OF MODEL SIZE 

In Table 5, we present how the model size affects the memorization level of CLIP models. Both models are trained using the same dataset and settings. We observe that with more parameters (larger model size), encoders have higher memorization capacity. This aligns with findings from previous research (Wang et al., 2024b; Feldman, 2020; Meehan et al., 2023). 

## A.5 VERIFICATION OF INFINITE DATA REGIMES 

To evaluate CLIPMem over infinite data regimes ( _i.e.,_ using a single training run where no data point is repeated), we use a subset _D_ (containing 7050000 samples) of YFCC100M dataset (Thomee et al., 2016b) to train another pair of ViT-Base models for only 1 epoch. Following our definition of CLIPMem, we further divide _D_ into _SS_ with 6950000 samples, _SC_ with 50000 samples, and _SI_ with 50000 samples. The reason we use 7M (6950000+50000) samples to train either model _f_ or model _g_ is to make sure the newly trained model has the same number of training samples as the model trained with K-epoch runs (70000 samples/epoch * 100 epoch). The results in Table 6 show that the model trained with infinite data regimes has higher linear probing accuracy on ImageNet as a downstream task and lower memorization scores, as measured by CLIPMem. This aligns with the fact that duplicated data points increase the memorization level and make the model over-fit, hence reducing the generalization (Wang et al., 2024b; Feldman, 2020). The results in Figure 9 show that the most memorized samples according to CLIPMem in the model trained with infinite data regimes are also samples with imprecise or incorrect captions. This aligns with our statements in Section 4.5. 

16 

Published as a conference paper at ICLR 2025 

Table 5: **CLIPMem and linear probing accuracy of models with different sizes.** The models are trained using identical settings and the same subset of the COCO dataset. Linear probing accuracy is tested on the ImageNet dataset as the downstream task. 

||Model<br>CLIPMem<br>Lin. Prob. Acc. (ImageNet)<br>ViT-base (Baseline in main paper)<br>0.438<br>63.11%_±_0.91%<br>ViT-large<br>0.457<br>67.04%_±_1.05%|
|---|---|
|Series taken during heritage surveys in 2007.<br>I have not had time to develop individual<br>captions for these, however happy to<br>provide some comment if required.<br>Cumbres and Toltec Scenic Railroad, historical<br>narrow gauge railroad between Chama, New<br>Mexico, and Antonito, Colorado. These photos<br>are from the station and railyard in Chama.<br>Day Four set here. Entire Japan collection here.<br>Natural Bridge & Rockshelter Wisconsin State<br>Natural Area #105 Natural Bridge State Park<br>Sauk County<br>Mountain Empire Filed Trip - Bristol, TN,<br>and environs Photo by Amy C. Evans, SFA<br>oral historian June 2009<br>www.southernfoodways.org<br>Photo By Terri Hodges<br>www.craftingthegalaxy.com Useage is<br>Creative Commons but credit is required<br>Mellwood Art Center in Louisville,<br>KY (1-27-06)<br>-- Lisa Brewster Sent from my Palm Pre<br>At Hellfest with After Forever<br>On the way to visit Western<br>Kentucky University (9/20/08)||



Figure 9: **Top 10 memorized samples according to CLIPMem in the model trained under infinite data regimes on YFCC100M.** The model is trained for one epoch, _i.e.,_ seeing each training data point exactly once. Even in this setup, the most memorized samples are still the ones with imprecise or incorrect captions. 

## A.6 EVALUATION ON BLIP 

To verify the effectiveness of CLIPMem over other similar multi-modal models, we train a BLIP model on COCO dataset following the same settings as the baseline model in the main paper. We present the results for CLIPMem on BLIP over all four data subsets in Figure 10, which is in agreement with the results of the BLIP model in Figure 2a. We also present the results for UnitMem in Figure 4, which is also very similar to the results of CLIP models 

## A.7 MEMORIZATION DISTRIBUTION DURING TRAINING 

We present the distributions of neurons with highest UnitMem during training in Figure 11. These results highly consistently indicate that in the early stages of training, neuronal memory occurs mainly in the lower layer of the clip model, while in the middle and later stages of training, neuronal memory is more concentrated in the later layer of the model. 

## A.8 HUMAN VS MACHINE GENERATED CAPTIONS 

For each image in the COCO dataset, we use GPT 3.5 (specifically, gpt-3.5-turbo) to generate 5 captions (from scratch). We use the following instruction in the OpenAI API: 

- def generate_description_for_image(image_caption, clip_features): prompt = f"Here is an image with the caption: ’{image_caption}’. " prompt += f"Based on this caption and the visual features represented by this embedding ’{clip_features}’, please generate a new detailed description." 

17 

Published as a conference paper at ICLR 2025 

Table 6: **Evaluation of CLIPMem under infinite data regimes,** _**i.e.,**_ **seeing every data point only once during training vs training with 100 epochs.** We observe that both setups reach comparable downstream accuracy and memorization. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0018-02.png)


**----- Start of picture text -----**<br>
Model CLIPMem Lin. Prob. Acc. (ImageNet)<br>ViT-Base (YFCC 7M, 1 epoch) 0.425 64.83%  ±  1.04%<br>ViT-Base (COCO 70K, 100 epochs) 0.438 63.11%  ±  0.91%<br>600<br>500 SC<br>SI<br>400 SE<br>300 S S<br>200<br>100<br>0<br>CLIPMem  for BLIP<br>1.000.75 0 .50 0 .250.000.250.500.751.00<br>Frequency<br>**----- End of picture text -----**<br>


Figure 10: **Memorization scores across data subsets on BLIP models** We train a BLIP model on COCO standard image cropping and no text augmentation. We present the results for CLIPMem over all 4 data subsets, which is in agreement with the results of the CLIP model in Figure 2a 

response = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[ {"role": "system", "content": "You are a helpful assistant that generates captions for images."}, {"role": "user", "content": prompt} ] ) return response[’choices’][0][’message’][’content’] 

We present the obtained captions in Figure 18. In Figure 12b, we analyze the pairwise cosine similarity in the original COCO and the GPT3.5 generated captions. We find that the GPT3.5 generated captions are slightly more uniform than the original COCO captions, reflecting in a higher pairwise cosine similarity. 

A.9 EXAMPLES FOR MEMORIZED SAMPLES 

18 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0019-01.png)


**----- Start of picture text -----**<br>
12<br>11 0.25<br>10<br>9 0.20<br>8<br>7 0.15<br>6<br>5 0.10<br>4<br>3 0.05<br>2<br>1 0.00<br>20 40 60 80 100<br>Epoch<br>(a)  Top 1% neurons<br>12<br>11 0.25<br>10<br>9 0.20<br>8<br>7 0.15<br>6<br>5 0.10<br>4<br>3 0.05<br>2<br>1 0.00<br>20 40 60 80 100<br>Epoch<br>(b)  Top 3% neurons<br>12<br>11<br>10 0.20<br>9<br>8 0.15<br>7<br>6<br>0.10<br>5<br>4<br>0.05<br>3<br>2<br>1 0.00<br>20 40 60 80 100<br>Epoch<br>(c)  Top 5% neurons<br>Layer<br>Percentage of total<br>Layer<br>Percentage of total<br>Layer<br>Percentage of total<br>**----- End of picture text -----**<br>


Figure 11: **Distribution of top 1%, 3%, and 5% neurons with highest UnitMem during training.** We train a CLIP model on COCO standard image cropping and no text augmentation following the settings of baseline model in main paper. We record the neurons with top 1%, 3%, and 5% of highest UnitMem during training (every epoch). 

19 

Published as a conference paper at ICLR 2025 

Table 7: **The machine generated captions provide similar performance to the original humangenerated captions.** We report the CLIPMem and linear probing accuracy of model trained with original COCO captions and captions generated by GPT 3.5. For the ’Single Caption’, only a single caption is used during training. For ’Five Captions’, all five captions are used equally during training (every caption trained for 20 epochs out of 100). The linear probing accuracy is tested on the ImageNet dataset as the downstream task. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0020-02.png)


**----- Start of picture text -----**<br>
COCO GPT 3.5<br>Single Caption Five Captions Single Caption Five Captions<br>CLIPMem 0.438 0.423 0.430 0.411<br>Linear Probing Accuracy (ImageNet) 63.11%  ±  0.91% 64.88%  ±  0.83% 63.09%  ±  1.12% 64.47  ±  0.72%<br>10000 10000<br>8000 8000<br>6000 6000<br>4000 4000<br>2000 2000<br>0 0<br>0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00<br>Cosine Similarity Cosine Similarity<br>(a)  COCO (Average: 0.798) (b)  GPT3.5 (Average: 0.851)<br>Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 12: **Pairwise cosine similarity of 5 captions from COCO and generated by GPT3.5.** 

|Noise|CLIPMem|Lin. Prob. Acc. (ImageNet)|
|---|---|---|
|None|0.438|63.11%_±_0.91%|
|_N_(0_._01)|0.435|63.36%_±_0.88%|
|_N_(0_._05)|0.428|64.02%_±_1.12%|
|_N_(0_._10)|0.421|64.95%_±_0.96%|
|**_N_ (0****_._15)**|**0****_._417**|**65****_._34**% **_±_ 0****_._84**%|
|_N_(0_._20)|0.422|64.83%_±_0.92%|
|_N_(0_._25)|0.436|63.28%_±_0.79%|
|_N_(0_._30)|0.447|61.50%_±_0.86%|
|_N_(0_._50)|0.491|57.04%_±_1.11%|
|_N_(0_._75)|0.501|52.28%_±_0.98%|
|_N_(1_._00)|0.504|51.92%_±_1.03%|



Table 8: **Noising text embedding during training.** We present the impact of adding noise to the text embedding during training for the ViT-base trained on COCO. 

20 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0021-01.png)


**----- Start of picture text -----**<br>
Caption 1: A flock of birds are flying through the sky. Caption 1: Laptop computer sitting on a desk with a black mouse on the table.<br>Caption 2: A large flock of birds flying high up in the air. Caption 2: A laptop sitting on a table near a mouse.<br>Caption 3: Many birds fly over houses while the sun is setting. Caption 3: A laptop sitting on a table with a mouse beside of it.<br>Caption 4: a flock of birds flying outside at sunset Caption 4: A laptop computer sitting next to a computer mouse.<br>Caption 5: A large flock of big birds flying away to roost Caption 5: The laptop and mouse sits upon the desk with one other item.<br>Caption 1: Several people are on the beach flying kites. Caption 1: Young children on snowy area of alpine region.<br>Caption 2: A group of people with kites on a beach. Caption 2: Two kids are on skis at the top of a slope.<br>Caption 3: A group of people fly large kites on a beach. Caption 3: Two people in ski gear with a mountain in the background.<br>Caption 4: A beech full of people with lots of parachutes overhead. Caption 4: Two people who are wearing snow skis on snow.<br>Caption 5: Several people are at the beach walking and flying kites. Caption 5: Two boys are skiing in snow in front of a mountain.<br>Caption 1: A dog sleeping in a dog bed on a floor. Caption 1: A close up of a wrist watch with focus on the date.<br>Caption 2: A DOG IS DRESS TO SLEEP IN THE DOG BED Caption 2: a watch saying it is friday the 13th<br>Caption 3: A dog is in a bed that matches his shirt. Caption 3: a close up of a clock and a date<br>Caption 4: A small dog in a basket that has a blanket that matches the  Caption 4: The object in the photo seems to be intriguing.<br>dogs clothing it is dressed in. Caption 5: A polished metal watch, seen close up, with the date Friday the 13th inset.<br>Caption 5: a dog is getting ready to lay down in his dog bed<br>Caption 1: Two vases that match in color sitting side by side. Caption 1: A street sign that says C have you paid?<br>Caption 2: A pair of very nice urns sitting on the floor. Caption 2: A street scene with focus on a sign on a pole.<br>Caption 3: a face with some kind of plant growing out of it. Caption 3: A sign says Have you paid and has a C on it.<br>Caption 4: Separate vases, one is empty, while the other has green flowers.  Caption 4: A street sign that is asking if you have paid.<br>Caption 5: Blue and black glazed jars with one holding a plant. Caption 5: A parking sign on the sidewalk next to a line of parked cars.<br>Caption 1: A large object floating in a lake next to a shore. Caption 1: A brown bear investigating something on top of a rock.<br>Caption 2: An elephant hangs out in a body of water. Caption 2: a big brown bear drinks water from a lake<br>Caption 3: A elephant walking a deep water to the other side Caption 3: A bear that is looking at some water.<br>Caption 4: The elephant wades through very deep, calm waters. Caption 4: A large brown bear peering into the water at the edge of a stream.<br>Caption 5: A large black object is in the middle of some water. Caption 5: A brown bear is looking into the water.<br>**----- End of picture text -----**<br>


Figure 13: **The 10 samples with lowest CLIPMem in the CLIP model trained with all 5 captions.** We can see that these samples contain clear concepts and precise captions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0021-03.png)


**----- Start of picture text -----**<br>
Caption 1: The man is on the dirt bike practicing his tricks. Caption 1: A bunch of stuffed animals in a room.<br>Caption 2: The image shows the stages of a biker jumping a ramp.  Caption 2: Teddy bears and hats for sale in a flea market booth.<br>Caption 3: A time lapse photo of a motorcycle stunt. Caption 3: there are many stuffed animals that are hanging at the top<br>Caption 4: A man riding a motorcycle over a ramp under a blue sky. Caption 4: Several teddy bears of different colors are being displayed.<br>Caption 5: There are many different people riding their motorcycles. Caption 5: We are looking at a display of prizes at a carnival game.<br>Caption 1: The wall of a bathroom is being torn down. Caption 1: A skateboarder looking down at his feet while skating.<br>Caption 2: A view of a house with the walls being demolished.  Caption 2: A skateboarder is moving swiftly down the street.<br>Caption 3: a partially demolished wall inside of a run down house. Caption 3: A person in a  yellow shirt, jeans and sneakers on a skateboard<br>Caption 4: A toilet in a room with a wall knocked out. Caption 4: There is a person riding a skateboard<br>Caption 5: A demolished wall with a view inside a bathroom with a toilet. Caption 5: First person view of a person riding a skate board.<br>Caption 1: A collection of assorted  farm animal photos  Caption 1: A car driving down a street in the pouring rain at night.<br>Caption 2: Farm pictures of ponies, chickens, sheep and goats. Caption 2: It is snowing as the cars drive down the street.<br>Caption 3: A composite of photos of a farm, including chickens, horses, and sheep. Caption 3: Night time city scene with now coming down at<br>Caption 4: Farm animals, horses, sheep, and chickens, pose for the camera. a traffic intersection.<br>Caption 5: Numerous images of farm animals including horses sheep  Caption 4: An intersection on a snowy night with several street lamps<br>goats chickens and a person at a petting zoo Caption 5: a city scene of a streetlights and cars in snow<br>Caption 1: A car driving down a street in the pouring rain at night. Caption 1: A setup for ubuntu the operating system, with a man at a desk.<br>Caption 2: It is snowing as the cars drive down the street. Caption 2: A basement office houses tech support for Ubuntu.<br>Caption 3: Night time city scene with now coming down at a traffic intersection. Caption 3: Man working on a laptop at a wooden table.<br>Caption 4: An intersection on a snowy night with several street lamps Caption 4: A man sitting at a table in a building.<br>Caption 5: a city scene of a streetlights and cars in snow Caption 5: a man sits at a desk next to some computers<br>Caption 1: a man sitting on a wooden bench in a field full of snow Caption 1: A sculpture made up of several traffic lights.<br>Caption 2: A man taking a nap on a park bench on a sunny day. Caption 2: A pole with many traffic lights on a street.<br>Caption 3: A man sitting on top of a park bench. Caption 3: A traffic light monstrosity sitting in the middle of a road.<br>Caption 4: a bench with a man that is sitting on it Caption 4: A traffic light tree sculpture in the middle of an intersection<br>Caption 5: A man rests on a bench in a snowy park Caption 5: a pole with many different traffic lights on a city street<br>**----- End of picture text -----**<br>


Figure 14: **The 10 samples with highest CLIPMem in the CLIP model trained with all 5 captions.** We can see that these samples contain atypical, difficult samples with imprecise or incorrect captions. 

21 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-05.png)


A girl in pink sweater putting a blue umbrella over a yellow fire hydrant. Closeup of two street signs that read "Airport Pkwy" and "Karmill Ave." A sign with Oriental writing and the words saying Hyatt on the Bund. A pink Hello Kitty microwave on a store shelf. A sign saying "Don't Honk, $350 Penalty" on a pole. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-11.png)


A yellow and blue fire hydrant surrounded by leaves A sign is displayed on a pole that says bump. A man in maroon shirt standing next to a stainless steel refrigerator. A blue street sign that reads "Thelonius Monk Circle." 

A group of artistic surfboards are displayed in a tent. 

Figure 15: **The 10 samples with lowest CLIPMem in the CLIP model trained with 1 caption.** We can see that these samples contain clear concepts and precise captions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-17.png)


**----- Start of picture text -----**<br>
An item of a vase with something<br>inside of it.<br>two males and a female in a red top<br>holding some flowers<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-19.png)


A group of people standing on top of a field together. 

a pole with some yellow lights in front of a narrow building 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0022-24.png)


a parking lot with a bunch of cars A woman opening the trunk of her in it car. A person is taken in this very a sleeping toddler laying on a womans shoulder picture. An open point of view of a room I am unable to see an image with various things all around above 

A woman opening the trunk of her car. 

Figure 16: **The 10 samples with highest CLIPMem in the CLIP model trained with 1 caption.** We can see that these samples contain atypical, difficult samples with imprecise or incorrect captions. 

22 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0023-01.png)


**----- Start of picture text -----**<br>
COCO Image COCO Caption Stable Diffusion Image<br>A group of people that are touching a animal.<br>A group of people walking through and petting a sheep.<br>A woman is petting a sheep that has been sheered.<br>A woman petting a sheep inside a pen at a fair.<br>A sheep in a stall being pet by a woman.<br>A man standing next to palm trees next to a bunch of<br>surfboards.<br>A SURFING BOARD STAND WITH A PERSON STANDING<br>NEAR BY.<br>A man standing in front of many surfboards at the beach.<br>A palm tree lined beach contains a large surf board rack,<br>and people milling about.<br>A group of people standing on the sands at the beach<br>A tree filled with lots of lemons and green leaves.<br>large yellow fresh fruit tree wet from rain<br>Yellow apples hanging from a leafy tree.<br>A close up of several oranges in a tree.<br>Several round, yellow fruits growing on a tree.<br>**----- End of picture text -----**<br>


Figure 17: **Samples of images generated by Stable Diffusion.** We present the generated images based on the COCO captions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0023-03.png)


**----- Start of picture text -----**<br>
COCO Captions GPT3.5 Captions<br>Three parrots sitting on a wooden branch next to each other. three parrots sitting on a branch<br>Three brightly-colored parrots are interacting in a tree. three blue and yellow parrots sitting on a branch<br>Three very colorful parrots perched on different sticks. three blue and yellow parrots sitting on a tree branch<br>Three brightly colored birds on top of a cluster of branches.  a trio of parrots sitting on a tree branch<br>Three parrots sitting on branches and grooming their feathers.  a trio of parrots sitting on a branch<br>A dim bathroom with a light over the sink a bathroom with a sink and mirror in it<br>A bathroom is dimly lit with a single bulb. a small bathroom with a sink and mirror<br>A dimly light bathroom with dark walls.  a bathroom with a sink and a mirror<br>A bathroom with a sink, mirror, and over head light. a small bathroom with a sink and a mirror<br>A bathroom sink with a light hanging above.  a bathroom with a sink and a mirror in it<br>The man is putting his bag on his shoulder as he walks away from the car. a man standing next to a car talking on a cell<br>A man in a peach colored shirt holding his luggage.  a man standing in front of a car talking on a cell<br>A man carrying a large bag through a parking lot. a man standing next to a car talking on a cell phone<br>a man with a hand bag talk on a cell phone  a man standing next to a parked car talking on a cell<br>A talking on a cell phone, with a bag on his shoulder. a man standing next to a car talking on the phone<br>The two giraffes stand facing away from each other next to the brown wall. two giraffes standing next to each other<br>Two giraffes walking away from each other against a fence. two giraffes standing in front of a wooden fence<br>two giraffes walking opposite of each other  A wood fence is behind two giraffes<br>A pair of two giraffes stand behind a line. two giraffes standing next to a wooden fence<br>Two giraffes walking in opposite directions near a wall. two giraffes standing in front of a fence<br>this is two cops riding on back of horses two police officers on horses<br>Two uniformed personnel are riding horses in a crowd two police officers riding horses<br>a couple of people are riding on the back of horses two police officers sitting on horses<br>A man and woman that are sitting on horses. two police officers on horsebacks<br>Two police officers riding through a crowd on top of horses. a couple of police officers on horses<br>**----- End of picture text -----**<br>


Figure 18: **Sample captions generated by GPT3.5.** We present the generated captions and the original image and captions from COCO. 

23 

Published as a conference paper at ICLR 2025 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models_images/iclr_2025_Captured_by_Captions_On_Memorization_and_its_Mitigation_in_CLIP_Models.pdf-0024-01.png)


**----- Start of picture text -----**<br>
600 600<br>SC Avg. = 0.168 SC Avg. = 0.213<br>500 500<br>SS Avg. = -0.001 SS Avg. = -0.008<br>400 400<br>300 300<br>200 200<br>100 100<br>0 0<br>1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00 1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00<br>SSLMem SSLMem<br>(a) SSLMem Image Encoder (1 caption) (b) SSLMem Image Encoder (5 captions)<br>700 700<br>SC Avg. = 0.209 SC Avg. = 0.250<br>600 600<br>SS Avg. = 0.013 SS Avg. = 0.002<br>500 500<br>400 400<br>300 300<br>200 200<br>100 100<br>0 0<br>1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00 1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00<br>SSLMem SSLMem<br>(c) SSLMem Text Encoder (1 caption) (d) SSLMem Text Encoder (5 captions)<br>600 600<br>SSLMemImage Avg. = 0.168<br>500 SC Avg. = 0.438 500 SSLMemText Avg. = 0.209<br>SS Avg. = 0.001 SSLMemSUM Avg. = 0.298<br>400 400 ClipMem Avg. = 0.438<br>300 300<br>200 200<br>100 100<br>0 0<br>1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00 1.00 0.75 0.50 0.25 0.00 0.25 0.50 0.75 1.00<br>CLIPMem SSLMem / CLIPMem<br>(e) CLIPMem (f) SSLMem, Naive Sum of SSLMem and CLIPMem<br>Frequency Frequency<br>Frequency Frequency<br>Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 19: **Evaluation of SSLMem and CLIPMem on a CLIP model trained on COCO.** Extended version of Figure 3 where we also include SSLMem calculated on encoders trained with 5 captions instead of 1. The trends in both cases are the same. SSLMem for the CLIP Models trained with the 5 captions is slightly higher since SSLMem uses the captions as augmentations for the calculation of the memorization. Overall, our CLIPMem reports the strongest memorization signal for CLIP. 

24 

