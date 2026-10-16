# **GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

**Ling Li**[1] **Yu Ye**[2] **Bingchuan Jiang**[3] **Wei Zeng**[1 4] 

## **Abstract** 

This work tackles the problem of geo-localization with a new paradigm using a large visionlanguage model (LVLM) augmented with human inference knowledge. A primary challenge here is the scarcity of data for training the LVLM - existing street-view datasets often contain numerous low-quality images lacking visual clues, and lack any reasoning inference. To address the dataquality issue, we devise a CLIP-based network to quantify the degree of street-view images being locatable, leading to the creation of a new dataset comprising highly locatable street views. To enhance reasoning inference, we integrate external knowledge obtained from real geo-localization games, tapping into valuable human inference capabilities. The data are utilized to train _GeoReasoner_ , which undergoes fine-tuning through dedicated reasoning and location-tuning stages. Qualitative and quantitative evaluations illustrate that _GeoReasoner_ outperforms counterpart LVLMs by more than 25% at country-level and 38% at citylevel geo-localization tasks, and surpasses StreetCLIP performance while requiring fewer training resources. The data and code are available at https://github.com/lingli1996/GeoReasoner. 

## **1. Introduction** 

Street-view geo-localization seeks to predict geographical locations for the given street-view images. The significance of street-view geo-localization is evident in a variety of applications, spanning social studies (Ye et al., 2019b), urban planning (Shen et al., 2018), and navigation (Chalvatzaras 

> 1The Hong Kong University of Science and Technology (Guangzhou)[2] Tongji University[3] Strategic Support Force Information Engineering University[4] The Hong Kong University of Science and Technology. Correspondence to: Bingchuan Jiang _<_ jbc021@163.com _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0001-09.png)


**----- Start of picture text -----**<br>
Retrieval-based approaches LVLM-based approaches<br>[ lat, lon ] 0.96 External Knowledge<br>ImageInput [ [ lat, lonlat, lon ] ] 0.610.24 [ lat, lon ] [ Outputlat, lon ] TrainingPhase LOC. 0.830.37 LVLM<br>QNT.<br>0.19<br>Classification-based approaches<br>TrainingPhase Geo CellsImages C1C3...... C1C2 C2C4...... ............ ............ Model InferencePhase ImageInput LVLM LocationAnswerGeo<br>InferencePhase ImageInput Model ClassificationResult QuestionInput Reasons<br>**----- End of picture text -----**<br>


_Figure 1._ Different paradigms in existing and the proposed geolocalization approaches: retrieval-based (left-top), classificationbased (left-bottom), and our LVLM-based (right). 

et al., 2022). As shown in Figure 1 (left), existing frameworks for street-view geo-localization can be mainly divided into two categories: _retrieval-based_ and _classificationbased_ . _Retrieval-based_ approaches entail identifying the most similar image within a geo-tagged image gallery and returning the corresponding geographical location (Zhu et al., 2022; Lin et al., 2022; Zhang et al., 2023b). However, the methods rely on the diversity and comprehensiveness of the geo-tagged image gallery, which can be challenging to curate. Alternatively, _classification-based_ approaches partition the Earth’s surface into distinct regions and assign the input image to a specific region (Clark et al., 2023; Pramanick et al., 2022; Muller-Budack et al.¨ , 2018; Seo et al., 2018; Weyand et al., 2016). While these methods leverage shared visual features within a single region, they may neglect valuable semantic information ( _e.g._ , signboard texts) crucial for geo-localization. More importantly, these classification methods often operate as black-box models, lacking reasoning capabilities for users to interpret. 

Achieving street view-based geo-localization with reasoning capability poses a considerable challenge. This study introduces a new paradigm that facilitates geo-localization with reasoning capability for street-view images, as depicted in Figure 1(right). The paradigm leverages an LVLM for its excellent capability in handling multi-modal visual and textual inputs and incorporates external knowledge learned from various online games for the reasoning procedure. Specifically, we introduce the concept of _locatability_ as a metric 

1 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

to quantify the degree of locatability in street-view images. On this basis, we devise a CLIP-based visual-text pairing network to match large-scale Google Street View (GSV) images with 3K finely reasoned text-image pairs from online games, to tackle the challenge of the absence of a highquality street-view dataset. The process filters through over 70K GSV images with geo-tags, all of which exhibit a high degree of locatability. 

Next, we construct an LVLM model, named _GeoReasoner_ , to overcome the difficulty of integrating reasoning capability in geo-localization. The training procedures of _GeoReasoner_ are divided into two folds: reasoning tuning and location tuning. In the first stage, we utilize the 3K reasoned text-image pairs encapsulating human inference knowledge, to fine-tune a well-trained LVLM model with LoRA (Hu et al., 2022) for reasoning adaptation. In the second stage, we leverage the curated 70K high-locatability GSV images dataset, to further fine-tune the LVLM model with another LoRA stacked on the first one for location tuning. We assess _GeoReasoner_ in terms of accuracy for both country-level ( _i.e._ , predicting the country in which a street view is located) and city-level ( _i.e._ , predicting the city in which a street view is located) geo-localization. The results demonstrate that _GeoReasoner_ outperforms the other counterparts by more than _25%_ at the country-level geo-localization and _38%_ at the city-level geo-localization with reasoning on our test dataset. Notably, _GeoReasoner_ performs slightly better than StreetCILP (Haas et al., 2023), which was trained on a substantially larger dataset of 1.1 million geo-tagged street-view images. We also evaluate _GeoReasoner_ against state-ofthe-art models for geo-localization using open benchmark datasets. The results show that _GeoReasoner_ achieves comparable performance with only 10k Flickr images used for training. The main contributions of our work are: 

- We present a new paradigm that leverages an LVLM and external knowledge of human inference for geolocalization with reasoning from street-view images. 

- We introduce the concept of _locatability_ and devise a CLIP-based network to quantify the degree of locatability in street-view images. 

- We propose _GeoReasoner_ , an LVLM that outperforms existing geo-localization models and provides detailed reasoning for the inferred results. 

## **2. Related work** 

## **2.1. Street Views** 

Street views, as the realm of physical environments routinely accessed and engaged with in daily life, bear significant relevance to human perception (Ye et al., 2019b) and urban design (Shen et al., 2018). Analyses of street views 

contribute to decision-making support (Ye et al., 2019a), improved understanding of urban social and economic structures (Bai et al., 2023b), and traffic asset monitoring and maintenance (Campbell et al., 2019; Li et al., 2021). This study places an emphasis on geo-localization based on street views. Specifically, drawing motivation from Zhang et al. (2018), we delineate the distribution of scene elements to quantify the degree of _locatability_ in street views. Highly locatable street-view images are curated to train an LVLM that surpasses existing geo-localization models. 

## **2.2. Image-based Geo-localization** 

Geo-localization entails determining spatial coordinates on the Earth’s surface, with broad applications in practical scenarios, including tracking individual trajectories (Cheng et al., 2022) and positioning autonomous vehicles (Chalvatzaras et al., 2022). This study focuses on image-based geo-localization, utilizing image data as input. Research on image-based geo-localization can be primarily classified into two approaches: retrieval-based (Zhu et al., 2022; Lin et al., 2022; Zhang et al., 2023b) and classificationbased (Clark et al., 2023; Pramanick et al., 2022; Muller-¨ Budack et al., 2018; Seo et al., 2018; Weyand et al., 2016). 

The retrieval-based approach involves the sequential matching of a single image with a gallery of overhead views, each labeled with geographical coordinates, and identifying the result with the highest matching as the location. However, the utilization of this method is limited due to its requirement for additional reference datasets. The classificationbased approach, exemplified by Weyand et al. (2016), involves subdividing the Earth’s surface into thousands of geographical cells and predicting the geographical unit to which an image belongs. The prediction effectiveness can be boosted with a dataset comprising millions of street views, whilst the granularity is influenced by the number of subdivided geographical cells. As such, many studies have been devoted to learning to corresponding multi-level features at different granularity (Vo et al., 2017), or multi-pair features for different tasks (Muller-Budack et al.¨ , 2018; Pramanick et al., 2022; Vivanco Cepeda et al., 2024). 

We approach image-based geo-localization with a novel paradigm. Specifically, we integrate semantic visual concepts that offer locatable features (Luo et al., 2022; Theiner et al., 2022), and incorporate human reasoning knowledge learned from geo-localization games using an LVLM. 

## **2.3. Vision-Language Models** 

The emergence of Large Language Models (LLMs) has significantly impacted various tasks related to natural language processing. These models exhibit remarkable performance in tasks such as text generation (Zhang et al., 2023a) and text-based question answering (Shao et al., 2023), owing to 

2 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0003-01.png)


**----- Start of picture text -----**<br>
External Knowledge<br>Text<br>Encoder<br>Green signs with white lettering provide highway information in South Korea. T1 T2 T3 Tm<br>Street-view Images I1 I1 T1 I1 T2 I1 T3 I1 Tm<br>I2 I2 T1 I2 T2 I2 T3 I2 Tm<br>EncoderImage I3 I3 T1 I3 T2 I3 T3 I3 Tm<br>Locatability<br>Metric<br>In InT1 InT2 InT3 InTm<br>**----- End of picture text -----**<br>


_Figure 2._ The locatability quantization network devises a CLIPbased visual-text pairing approach to predict the locatability metric. 

their robust and versatile capabilities. As a result, research attention has shifted towards exploring prompt engineering techniques to enhance the performance of LLMs in downstream tasks (Wei et al., 2022; Yao et al., 2024; Dai et al., 2023; Xu et al., 2023; Ying et al., 2024). 

Large vision-language models (LVLMs) integrate visual encoders with LLMs, exhibiting remarkable effectiveness in visual question-answering tasks (Liu et al., 2024; Bai et al., 2023a; Rao et al., 2023). This study harnesses the capabilities of LVLMs to address geo-localization of street views. However, the optimal utilization of LVLMs remains a challenging issue, particularly due to the absence of high-quality training data and a lack of reasoning capabilities. We overcome these challenges through an innovative paradigm and the thoughtful design of model architecture, contributing to a more effective utilization of LVLMs in this domain. 

## **3. GeoReasoner** 

This section outlines our approach to addressing two challenges: 1) the absence of a high-quality street-view geolocalization dataset (discussed in Sect. 3.1), and 2) the difficulty of integrating reasoning in geo-localization (discussed in Sect. 3.2), when constructing _GeoReasoner_ . 

## **3.1. Locatability-Enhanced Data Curation** 

Throughout the development of this work, we observed variations in the degree of locatability among different street views. For example, the images featuring textual signboards or prominent landmarks ( _e.g._ , Eiffel Tower) are easily locatable, whilst those captured in a tunnel or obscured by a wall tend to be less locatable. Refer to Figure 4 for further illustration. Simply merging all these street-view images to train an LVLM is not an optimal approach, as the inclusion of poor-quality data can adversely affect the training efficiency of updating an LVLM (Radford et al., 2021). To this end, we introduce _locatability_ , a metric that quantifies the level of locatability of street-view images. We then devise a CLIP-based visual-text pairing network to produce the desired locatability metric for an input street-view image, 

as shown in Figure 2. The network naturally incorporates data from two perspectives: 

- **Street-View Images.** We collected street-view images from the Google Street View[1] (GSV). To enrich the diversity of the dataset, we first selected the top global cities according to the Globalization and World Cities Study Group and Network (GaWC) ranking. Next, we utilized the global OpenStreetMap[2] (OSM) geographic database to obtain the vector data of the road network in these urban areas. The road network was passed to ArcPy, a Python site package of ArcGIS, to automatically extract sampling points at 4000-meter intervals and generate a CSV table containing information about these sampling points. Subsequently, we employed the GSV API interface to compile a comprehensive dataset. This dataset encompassed street-view images captured from four distinct directions - front, back, left, and right - of each sampling point. Considering the impact of data sparsity and image similarity, we randomly selected two of the four views from each data point, denoted as [ **I** _x,_ **I** _y_ ], where _x ∈_ ( _left, right_ ), _y ∈_ ( _front, back_ ). The process has yielded a total of over 130k street-view images with geo-tags collected from 72 cities in 48 countries. 

- **Textual Clues.** Textual clues often serve a pivotal role in delineating the geographical locations of streetview images. Two prominent games, GeoGuessr[3] and Tuxun[4] , which focus on geo-localization through street views, offer a potential solution to this gap. Their communities have collaboratively curated a well-organized collection of textual clues, used for pinpointing geographical locations across various countries and cities. These clues, maintained by both players and administrators, provide valuable domain knowledge that aids in identifying and evaluating key geographical features in street views. While such datasets now exist (Luo et al., 2022), there are no readily available image-text data pairs specifically tailored for LVLM training. To bridge this gap, we gathered image-text pairs for geo-localization from these two open-source communities. Subsequently, we utilized the BERTbased Named Entity Recognition (NER) (Kenton & Toutanova, 2019) model to clean and filter text that lacked specific geographical location information. In this way, we collected a total of over 3K textual clues that encapsulate rich geo-localization information. For instance, _“houses in central Chile are more likely to have terracotta tiled roofs”_ . Each clue is paired with a corresponding street-view image. 

- 1https://www.google.com/streetview 

- 2https://www.openstreetmap.org 

- 3https://www.geoguessr.com 

- 4https://tuxun.fun 

3 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0004-01.png)


**----- Start of picture text -----**<br>
STAGE 1 Reasoning Tuning STAGE 2 Location Tuning<br>Q : Q :<br>Vision Encoder Which country is the  Vision Encoder Which country<br>image located? and city is the<br>VL Adapter Game  Explain the reason. VL Adapter GSV  image located?<br>images images<br>LoRA 1 Pre-trained LLM { ' 'reasonscountry':': 'India’'Indian signboard’ } LoRA 1 LoRA 2 Pre-trained LLM { ' country'city':': 'Switzerland’ 'Zurich’ }<br>Street Image Answer<br>{  'country': 'South Korea',<br>Vision Encoder VL Adapter    'reasons   'city': ': 'Seoul', ['Address bar at the top<br>shows the address is in<br>Question LoRA 1 LoRA 2 Seoul', 'The picture is a street<br>According to the content of the image, please think step by step and deduce in  view', 'The writing on the<br>which country and city the image is most-likely located and give the most  Pre-trained LLM picture is in       Korean']<br>important reason.Output in JSON format, e.g. {'country': '', 'city': '' ,'reasons':''}.  }<br>Training<br>Input<br>Inference<br>**----- End of picture text -----**<br>


_Figure 3._ The architecture of _GeoReasoner_ consists of three modules: _Vision Encoder_ , _VL Adapter_ and _Pre-trained LLM_ . The model undergoes a two-fold supervised fine-tuning process: reasoning tuning and location tuning, to enable geo-localization with reasoning. 

With the GSV images and textual clues, our subsequent task is to filter GSV images with a high degree of locatability, for the purpose of training an LVLM. To achieve this, we design a CLIP-based visual-text pairing network. As depicted in Figure 2, the GSV images undergo processing by an image encoder that deduces the image attributes. 

Here, we first use MaskFormer (Cheng et al., 2021) to predict segmentation masks for various classes in GSV images, such as buildings, sky, and vehicles. We then compute an _n_ -length vector **I** _seg_ , which quantifies the area ratio of each mask class, where _n_ represents the number of classes. Subsequently, we utilize Sentence-BERT (Reimers & Gurevych, 2019) to measure the similarity between textual clues and semantic segmentation labels, yielding an _m × n_ matrix _M_ , where _m_ is the number of textual clues. After that, we normalize _M_ using min-max normalization, and set values lower than the threshold to zero, resulting in another _m × n_ matrix _M_[ˆ] . We reduce _M_[ˆ] to an _n_ -length vector by calculating the mean across its rows, and then normalize it to obtain **w** _loc_ . This vector represents the importance of each semantic segmentation label for geo-localization. With the segmentation mask ratio **I** _seg_ and the corresponding weight **w** _loc_ , the locatability metric of a GSV image is computed through the multiplication and accumulation of the respective values, as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0004-05.png)


where **I** _seg_ ( _k_ ) denotes pixel ratio of the _k_ -th class in the segmentation mask **I** _seg_ . 

A higher _locatability_ value indicates a higher degree of visual clues exhibited in a GSV image for geo-localization, while a lower value suggests the opposite. Empirically, we selected a threshold value of 0.4 for filtering locatable GSV images. This resulted in over 70k highly locatable images with geo-tags passing to the next stage for training an LVLM. 

## **3.2. Geo-localization with Reasoning** 

While many models ( _e.g._ , Clark et al. (2023); Pramanick et al. (2022); Muller-Budack et al.¨ (2018); Seo et al. (2018); Weyand et al. (2016)) exist for image-based geo-localization, these models typically predict locations without providing the inference process. This introduces several limitations: First, the models operate as black boxes without providing insights, making it challenging for users to interpret. This obstacle impedes further refinement of the geo-localization model. More importantly, studies have demonstrated that integrating the reasoning process can enhance the capabilities of LLMs (Qiao et al., 2023). Therefore, our objective is to construct an LVLM for image-based geo-localization with reasoning capability. 

**Model Architecture.** Figure 3 illustrates the architecture of the proposed model _GeoReasoner_ , which is based on QwenVL (Bai et al., 2023a). _GeoReasoner_ consists of three modules: _Vision Encoder_ , _Vision-Language (VL) Adapter_ and _Pre-trained LLM_ . Specifically, the _Vision Encoder_ module employs the Vision Transformer (ViT) (Dosovitskiy et al., 2021) architecture. The input street-view images are resized to a specific resolution and then divided into a set of image patches. To refine image patches into sequential representations compatible with an LLM, the _VL Adapter_ is introduced. In the VL Adapter, the sequence of visual features is initially condensed to a fixed length to address efficiency challenges posed by the substantial number of visual feature sequences. Subsequently, the processed visual features are integrated with the LLM using cross-attention mechanisms. Following this, the compressed visual feature sequence and text sequence are passed to the _Pre-trained LLM_ module, which functions as a decoder for generating the answer. 

**Supervised Fine-tuning.** The overall model undergoes a staged pre-training process that is divided into two folds: reasoning tuning and location tuning. In the first stage, our objective is to enhance the model’s reasoning capability 

4 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-01.png)


**----- Start of picture text -----**<br>
loc. = 0.70 loc. = 0.70 loc. = 0.71<br>loc. = 0.08 loc. = 0.14 loc. = 0.19<br>**----- End of picture text -----**<br>


_Figure 4._ Locatability examples. Top row: the street views are highly locatable by signboards, architectural styles, and landmarks. Bottom row: no visual clues for locating the street views. 

by utilizing textual clues paired with street-view images collected from geo-localization games. The input streetview image & question, and the output answer are formatted as prompts in the following manner: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-11.png)


Here, we can only provide reasoning at the country level due to the granularity exhibited in the image-text pairs. Nevertheless, this reasoning procedure is sufficient to facilitate the second stage of location tuning. Next, we integrate the prior knowledge of country information with highly locatable GSV images with geo-tags to infer the fine-grained citylevel location information. We utilize a similar prompt format as in the first stage but without a reasoning requirement. Both stages are fine-tuned from the pre-trained Qwen-VL with LoRA, which contributes to the overall performance improvement of Qwen-VL in both the reasoning and location tuning stages, allowing the model to better capture complex relationships within the image-text pairs. 

## **4. Experiments** 

We conduct a series of experiments to evaluate the effectiveness of the locatability-enhanced geo-localization dataset (Sect. 4.1) and the model _GeoReasoner_ for geo-localization with reasoning (Sect. 4.2). 

## **4.1. Experiments on Locatability-Enhanced Dataset** 

## 4.1.1. QUALITATIVE COMPARISON 

Figure 4 presents examples of the predicted locatability degrees of different street-view images by our locatability quantization network. The top row showcases street views 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-18.png)


_Figure 5._ The relationship between building proportion and the degree of locatability in street views. The locatability metric peaks when the building proportion is approximately 0.2. 

distinguished by prominent localizable attributes. The left image features the Korean language on a signboard, the middle image captures the distinctive _Art Nouveau_ architectural style commonly found in Switzerland, and the right image shows an art & design museum in India. In contrast, street views in the bottom row display lower locatability degrees. The left image resembles a tunnel, lacking additional discernible information for accurate localization. Similarly, the middle image is occluded by a wall, and the right image faces common vegetation that is available worldwide. 

For the proposed locatability metric in Equation 1, we also evaluated the relationship between building proportion and the degree of locatability of street views. The results are shown in Figure 5. The locatability metric slightly increases as the building proportion ranges from 0 to 0.2, but decreases as the building proportion continues to increase. The results indicate that buildings are not the sole determinant of locatability. As the proportion of buildings increases, the street-view images transition from panoramic to closeup views, leading to reduced information availability and consequently diminishing the degree of locatability. 

The qualitative analysis indicates the effectiveness of the locatability quantization network in predicting locatability degrees of street-view images. Furthermore, the prediction aligns with human inference knowledge harvested from real geo-localization games, providing the ground truths for fine- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0005-23.png)


_Figure 6._ Quantitative comparison of country- and city-level geolocalization accuracy by different models trained on mixed datasets with varying proportions of high locatable GSV images. 

5 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0006-01.png)


**----- Start of picture text -----**<br>
Input Question<br>According to the content of the image, please think step by step and deduce in which country and city the image is most-likely located and give the most important reason.Output in JSON format, e.g. {'country': '', 'city': '' ,'reasons':''}.<br>Ground Truth 'country': 'Singapore' 'city': 'Singapore'<br>LLaVA Qwen-VL<br>'country': 'United States' 'city': 'Miami'<br>'reason': 'The presence of palm trees and a blue car parked on<br>the side of the road are indicative of a warm climate, which is<br>commonly associated with Miami, Florida. Additionally, the image<br>shows a busy city street with people walking and cars driving, which is consistent with the urban environment of Miami.' 'country': ' Singapore ' 'city': ' Singapore GeoReasoner '<br>GPT-4V 'country': ' Singapore ' 'city': ' Singapore ' Singapore's public transportation system.', 'The small size of the 'reason': ['The  ComfortDelGro taxi is a recognizable symbol of<br>plate that is consistent with Singapore's format, and the architec-ture of the buildings suggests a modern, urban environment typical of Singapore. Additionally, the 'reason': 'The image features  tropical treestaxi  has ', the car has a license  COMFORT ' written  island nation and the need to accommodate a large population have led to the development of high-density residential areas such as Tiong Bahru.', 'The city-state has a tropical climate with high humidity and rainfall, which has influenced the development<br>on it, which is a taxi company operating in Singapore.' of urban green spaces such as parks and gardens.']<br>Ground Truth 'country': 'United States' 'city': 'Las Vegas'<br>LLaVA Qwen-VL<br>'country': ' United States ' 'city': 'New York'<br>'reason': 'The presence of the New York sign and the tall buildings<br>in the background, along with the busy street filled with cars, all<br>indicate that the image is most likely taken in New York City.'<br>GPT-4V<br>'reason': 'The presence of the replica Eiffel Tower and the New 'country': ' United States ' 'city': ' Las Vegas ' 'country': ' United States ' 'city': ' Las Vegas GeoReasoner '<br>the York- Las Vegas StripNew York Hotel  .' and  Casino , which are iconic landmarks on  Vegas Strip 'reason': ['Large neon lit buildings and billboards line the .'] Las<br>Ground Truth 'country': 'China' 'city': 'Lhasa'<br>LLaVA Qwen-VL<br>'country': ' China ' 'city': 'Beijing'<br>'reason': 'The presence of Chinese characters on the buildings<br>and signs, as well as the traditional Chinese clothing worn by the<br>people in the image, suggest that the location is in China, specifi-<br>cally in Beijing.'<br>GPT-4V GeoReasoner<br>'reason': 'The architectural style and the  traditional attire  of the  'reason': ['The picture shows a Tibetan woman in  traditional<br>person in the foreground are indicative of Tibetan culture. The  clothing , which is a distinctive feature of Tibet.', ' The flag of China<br>signs on the buildings are in  both Tibetan script and Chinese ,  can be seen in the picture, indicating that the picture was taken in<br>which is common in Tibet, China. The urban setting with the  China.', 'The picture shows a busy street in Lhasa, which is the<br>specific style of buildings and signage suggests that is taken in  capital city of Tibet.']<br>Lhasa, the capital city of the Tibet Autonomous Region in China.'<br>'country': 'Malaysia' 'city': 'Kuala Lumpur'<br>'country': ' United States ' 'city': ' Las Vegas '<br>'country': ' China ' 'city': ' Lhasa '<br>'country': ' China ' 'city': ' Lhasa ' 'country': ' China ' 'city': ' Lhasa '<br>'reason': 'There are tall buildings,  a ComfortDelGro taxi , and<br>people walking on the sidewalk.'<br>'reason': 'The neon lights and busy streets are iconic to Las Vegas,<br>Nevada.'<br>'reason': 'The main square of Lhasa is the most famous square in<br>Lhasa, and it is also the symbol of Lhasa.'<br>Input Image<br>Input Image<br>Input Image<br>**----- End of picture text -----**<br>


_Figure 7._ Examples of LVLM-based approaches in geo-localization with reasoning. Prediction results matching the ground truth are highlighted in green, while reasons offering valid information are marked in blue. 

## tuning the reasoning component in _GeoReasoner_ . 

## 4.1.2. QUANTITATIVE COMPARISON 

We conducted quantitative experiments to investigate the importance of using high-locatability GSV images in training the location component in _GeoReasoner_ . Various datasets were prepared, featuring different proportions of high-locatability GSV images, ranging from 0% (only lowlocatability GSV images) to 100% (only high-locatability GSV images). To ensure fairness, each experimental group retained consistent 10K GSV images, with only the proportion of high-locatability images varying. Subsequently, models were trained for each dataset, and their accuracy in country- and city-level geo-localization was evaluated on a randomly sampled set of 1K GSV images. 

The experimental results are presented in Figure 6. Overall, the results reveal that as the proportion of high-locatability GSV images in the training dataset increases, the performance of the fine-tuned location component improves in both country- and city-level geo-localization. Specifically, the country- and city-level geo-localization accuracy increases from 0.63 & 0.47 for 0% high-locatability GSV images, to 0.72 & 0.51 for 100% high-locatability GSV 

images. Notably, the experiments only utilize 10K GSV images instead of all the curated 70K high-locatability GSV images due to training complexity. Nevertheless, the results demonstrate that high-locatability GSV images offer more meaningful insights and less extraneous noise, making them highly valuable in the geo-localization task. 

## **4.2. Experiments on Geo-localization with Reasoning** 

## 4.2.1. QUALITATIVE COMPARISON WITH SOTA 

To assess the efficacy of _GeoReasoner_ in terms of geolocalization with reasoning, we conduct a qualitative comparison with state-of-the-art LVLM-based approaches, including LLaVA (Liu et al., 2024), Qwen-VL (Qwen7B) (Bai et al., 2023a), and GPT-4V (Achiam et al., 2023). In the experimental phase, we presented the same input street-view images, reasoning process, and result formats to these models. Specifically, a consistent prompt is used, as below: 

_According to the content of the image, please think step by step and deduce in which country and city the image is most likely located and offer possible explanations. Output in JSON format, e.g., {‘country’: ‘’, ‘city’: ‘’, ‘reasons’:‘’}_ . 

6 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

_Table 1._ Comparison of Precision, Recall and F1 scores in country-level and city-level geo-localization. * represents the model trained on high-locatability GSV images. 

|Model|Accuracy_↑_|Country<br>Recall_↑_|F1_↑_|Accuracy_↑_|City<br>Recall_↑_|F1_↑_|
|---|---|---|---|---|---|---|
|StreetCLIP (Haas et al.,2023)|0.7943|1.00|0.8854|0.7457|1.00|0.8543|
|LLaVA (Liu et al.,2024)|0.4029|1.00|0.5744|0.2400|1.00|0.3871|
|Qwen-VL (Qwen-7B) (Bai et al.,2023a)|0.5829|0.95|0.7225|0.3743|0.89|0.5270|
|GPT-4V (Achiam et al.,2023)|0.8917|0.34|0.4923|0.5083|0.31|0.3851|
|ViT* (Dosovitskiy et al.,2021)|0.7100|1.00|0.8304|0.6762|1.00|0.8068|
|GeoReasoner*|0.8237|1.00|**0.9033**|0.7521|1.00|**0.8585**|



Figure 7 illustrates the inference results of counterpart models and _GeoReasoner_ on three diverse street views from different countries and cities—namely, Singapore-Singapore (top), United States-Las Vegas (middle) and China-Lhasa (bottom). Overall, _GeoReasoner_ not only outperforms existing models in the accuracy of country or city-level predictions but also provides coherent explanations with insightful reasoning for the inference results. 

In Figure 7 (top), _GeoReasoner_ identifies the word ‘COMFORT’ on the taxi in the image. Drawing from prior knowledge, _‘the ComfortDelGro taxi is a distinctive symbol of Singapore’s public transportation system’_ in the text-image pairs, the model deduces that the area is likely to be in _Singapore_ . GPT-4V predicts the same geo-location with accurate reasoning, yet the other two models fail, either due to not recognizing the taxi by LLaVA or making an incorrect inference about the city by Qwen-VL. 

Figure 7 (middle) presents a scene of the Las Vegas Strip. A conspicuous ‘NEW YORK’ sign is prominently visible in the upper-left corner of the image. This sign causes the reasoning error in the task performed by LLaVA. Although Qwen-VL generates accurate predictions of _Las Vegas-United States_ , the most essential factor, _i.e._ , ‘Las Vegas Strip’, is not considered in the reasoning process. In contrast, both _GeoReasoner_ and GPT-4V provide the correct geo-location along with accurate inference. 

Based on the depiction of Chinese characters and traditional clothing in Figure 7 (right), all models make accurate predictions regarding the country, identifying it as _China_ . However, LLaVA makes an incorrect prediction of the city, specifying _Beijing_ . In contrast, the other models successfully predict the city as _Lhasa_ , providing sensible and justifiable reasons for their inferences. 

obtaining corresponding and relevant answers is not guaranteed at all times. Therefore, we included _Recall_ rate to measure the proportion of effective answers within the large language models. When calculating the _Accuracy_ rate, only the accuracy of these effective answers is taken into account. We additionally compute _F1_ values, taking into consideration both _Accuracy_ and _Recall_ metrics. 

Table 1 presents the prediction results by the counterparts and _GeoReasoner_ . Overall, _GeoReasoner_ outperforms all the counterparts, particularly those LVLM-based approaches. Taking the best performed Qwen-VL for example, _GeoReasoner_ outperforms it 25.02% on country-level geolocalization and 38.61% on city-level geo-localization, in terms of F1 value. Surprisingly, the recall performance of GPT-4V for the geo-localization task was notably low. Most of the responses were mainly: _‘I’m sorry, I can’t provide assistance with that request.’_ or _‘I’m sorry, but I am unable to provide the exact location, such as the country and city, for the image you have provided. My capabilities do not include analyzing specific details to determine the geographical location of the image content.’_ 

We speculate that GPT-4V has undergone extensive measures to ensure the model’s security and privacy, which may contribute to its reluctance or denial of recognition in the task of geo-localization. 

In comparison to StreetCLIP that is specialized in geolocalization, _GeoReasoner_ demonstrates only a slight superiority. Nevertheless, it’s important to note that StreetCLIP was trained on a significantly larger dataset of over 1.1 million street-view images, while our _GeoReasoner_ was trained with only 70K street views. For ViT trained on the same data, _GeoReasoner_ still exhibits superior geolocation capabilities. Moreover, _GeoReasoner_ offers reasoning capability, providing added value for various downstream tasks. 

## 4.2.2. QUANTITATIVE COMPARISON WITH SOTA 

We further conduct quantitative experiments to compare with counterparts LVLMs. In addition, we choose StreetCLIP (Haas et al., 2023) as the state-of-the-art classificationbased approach and omit retrieval-based approaches relying on a geo-tagged image gallery that is not available. It is important to clarify that, for the LVLM-based approaches, 

## 4.2.3. ABLATION EXPERIMENTS 

To assess the contributions of the location tuning and reasoning tuning components in _GeoReasoner_ , we design several ablation experiments using the Qwen-VL (Bai et al., 2023a) pre-trained model as the baseline. Next, we integrated the Qwen-VL pre-trained model with LoRA1 ( _GeoReasoner_ 

7 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

_Table 2._ Results of the ablation experiments using baseline Qwen-VL (Qwen-7B), GeoReasoner w/o location tuning, GeoReasoner w/o reasoning tuning, and the full GeoReasoner models. 

||Training|Training|||Performance|Performance|||
|---|---|---|---|---|---|---|---|---|
|Model|Reasoning|Location|Accuracy_↑_|Country<br>Recall_↑_|F1_↑_|Accuracy_↑_|City<br>Recall_↑_|F1_↑_|
|Qwen-VL (Qwen-7B)|-|-|0.5829|0.95|0.7225|0.3743|0.89|0.5270|
|GeoReasoner w/o location tuning|✓|_×_|0.6971|1.00|0.8215|0.4114|0.99|0.5813|
|GeoReasoner w/o reasoning tuning|_×_|✓|0.7803|1.00|0.8766|0.7029|1.00|0.8255|
|GeoReasoner|✓|✓|**0.8237**|1.00|**0.9033**|**0.7521**|1.00|**0.8584**|



_Table 3._ Comparison results on Im2GPS dataset. The top five rows are derived from the results reported in the paper, while the last four rows are from retesting on the filtered Im2GPS dataset, which includes only highly locatable data. 

|Model|Dataset<br>Train|w/ Filter<br>Test|Street<br>1km|City<br>25km|Country<br>750km|
|---|---|---|---|---|---|
|PlaNet|_×_|_×_|0.08|0.25|0.54|
|CPlaNet|_×_|_×_|0.17|0.37|0.62|
|ISNs|_×_|_×_|0.17|0.43|0.67|
|Translocator|_×_|_×_|0.20|0.48|0.76|
|GeoDecoder|_×_|_×_|0.22|0.50|0.80|
|ISNs|_×_|✓|0.25|0.43|0.78|
|GeoCLIP|_×_|✓|0.25|0.49|0.87|
|GeoReasoner|_×_|✓|0.10|0.41|0.82|
|GeoReasoner|✓|✓|0.13|0.44|0.86|



_Table 4._ Comparison results on Im2GPS3k dataset. The top six rows are derived from the results reported in the paper, while the last four rows are from retesting on the filtered Im2GPS3k dataset, which includes only highly locatable data. 

|Model|Dataset<br>Train|w/ Filter<br>Test|Street<br>1km|City<br>25km|Country<br>750km|
|---|---|---|---|---|---|
|PlaNet|_×_|_×_|0.09|0.25|0.48|
|CPlaNet|_×_|_×_|0.10|0.27|0.49|
|ISNs|_×_|_×_|0.11|0.28|0.50|
|Translocator|_×_|_×_|0.12|0.31|0.59|
|GeoDecoder|_×_|_×_|0.13|0.34|0.61|
|GeoCLIP|_×_|_×_|0.14|0.34|0.70|
|ISNs|_×_|✓|0.10|0.29|0.59|
|GeoCLIP|_×_|✓|0.12|0.38|0.83|
|GeoReasoner|_×_|✓|0.09|0.35|0.74|
|GeoReasoner|✓|✓|0.10|0.38|0.83|



without location tuning) and LoRA2 ( _GeoReasoner_ without reasoning tuning). The last experiment involved the full _GeoReasoner_ model, including both the location tuning and reasoning tuning components. The same prompts were utilized for all these models, as in the previous experiments. 

Table 2 presents the quantitative results in terms of _accuracy_ , _recall_ , and _F1_ . Overall, the results indicate that both the location tuning and reasoning tuning components improve the model performance. Specifically, the location tuning component is essential for geo-localization, as _GeoReasoner_ w/o reasoning tuning (row 3) achieves much higher accuracy than _GeoReasoner_ w/o location tuning (row 2), especially for fine-grained city-level prediction. This result further strengthens the evidence that high-locatability GSV images are essential for geo-localization. The reasoning tuning component also plays a significant role in the performance improvement, as evidenced by the superior performance of the full _GeoReasoner_ (row 4). 

## 4.2.4. GENERALIZABILITY EVALUATION 

To further assess the generalizability of _Georeasoner_ in geo-localization, we conduct additional testing on open Flickr image datasets of Im2GPS (Hays & Efros, 2008) and Im2GPS3k (Vo et al., 2017). Here we use only 10k Flickr images for fine-tuning _Georeasoner_ . Since _Georeasoner_ predicts city names rather than GPS coordinates, we first convert the predicted city names generated by _Georeasoner_ 

into the GPS coordinates of their respective city centers, then measure the distance between these predicted coordinates and ground-truth locations. 

Table 3 and Table 4 present the performance comparison of _Georeasoner_ with PlaNet (Weyand et al., 2016), CPlaNet (Seo et al., 2018), ISNs (Muller-Budack¨ et al., 2018), Translocator (Pramanick et al., 2022), GeoDecoder (Clark et al., 2023), and GeoCLIP (Vivanco Cepeda et al., 2024) on Im2GPS and Im2GPS3k datasets, respectively. The results demonstrate that fine-tuning _GeoReasoner_ using highly locatable images significantly improves prediction accuracy for street, city, and country levels (row 8 vs. row 9 in Table 3, and row 9 vs. row 10 in Table 4). Remarkably, despite being fine-tuned solely on a smaller number of Flickr images, _GeoReasoner_ achieves results comparable to ISNs and GeoCLIP trained on millions of Flickr images, particularly in terms of city- and countrylevel accuracy. Besides, _GeoReasoner_ trained on the filtered, highly locatable Flickr images also show improvements in the city- and country-level geo-localization, demonstrating the generalizability of our proposed _locatability_ module. 

## **5. Discussion** 

**The significance of high-locatability street-view images.** We observe a significant performance improvement when _GeoReasoner_ is trained upon high-locatability street-view 

8 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

images. Such images often contain explicit visual clues such as stylized architecture, traffic signs, and landmarks, providing the model with richer contextual information. Therefore, increasing the quality of the training dataset enhances the model’s geo-localization performance. Additionally, the quantity of high-locatability images is vital, as the model trained with 70K images (as in Sect. 4.2.2) achieves significantly higher accuracy than the one trained with 10K images (Sect. 4.1.2). In balancing the quality and quantity of the training dataset, we empirically applied a threshold of 0.4 to differentiate between highly and less localizable street views. Setting the threshold too high ( _e.g._ , 0.7) can lead to a notable decrease in the amount of high-locatability images, whilst a lower threshold ( _e.g._ , 0.1) may bring in introduce low-quality images. 

**The necessity of reasoning process.** The introduction of the reasoning component successfully elevated _GeoReasoner_ ’s performance in the geo-localization task. This signifies that LVLM can adeptly capture intricate relationships among image features, location clues, and geo-locations in the training process. Implemented an innovative solution to empower the reasoning capability within _GeoReasoner_ by leveraging human inference knowledge extracted from geolocalization games. Despite the relatively small dataset, a noticeable improvement in performance has been achieved. In the future, we plan to expand the reasoning dataset by diversifying the influencing clues. For instance, the current textual clues are absent of landscape information, which could provide invaluable insights for geo-localization. We will collaborate with domain experts such as urban planners and geographers to address these limitations. 

**Failure cases.** _GeoReasoner_ comprehends architectural style as a pivotal factor in geo-localization. However, the model can be misled by the learned significance of architectural style. Figure 8 presents a street view of the Eiffel Tower in Paris, France (left), and replicas of the Eiffel Tower in New York, USA (middle) and in Hangzhou, China (right). _GeoReasoner_ fails to distinguish between them, predicting all instances as located in Paris, France. This misclassification is not unique to _GeoReasoner_ but also extends to other LVLMs like GPT-4V. Consequently, it underscores the necessity for LVLM-based methods to delve deeper into knowledge for more sophisticated geo-localization capabilities. Once again, it is imperative to collaborate with domain experts and enhance the visual clues and reasoning procedure comprehensively to tackle this issue. 

## **6. Conclusion** 

In this paper, we present a new paradigm that integrates a large vision-language (LVLM) model with human inference knowledge for street view geo-localization with reasoning. We introduce the concept of _locatability_ and devise a CLIP- 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0009-06.png)


**----- Start of picture text -----**<br>
GT: Paris, France GT: New York, USA GT: Hangzhou, China<br>Predicted:Paris, France Predicted: Paris, France Predicted:Paris, France<br>**----- End of picture text -----**<br>


_Figure 8._ GeoReasoner fails to distinguish the Eiffel Tower and its replicas in New York, USA, and Hangzhou, China. 

based network to quantify the degree of locatability in streetview images, facilitating the selection of high-quality data. We design an LVLM-based model named _GeoReasoner_ , which harnesses external knowledge of human inference from real geo-localization games and curated high-quality data to enhance the performance of geo-localization tasks with reasoning capabilities. The model undergoes two-stage fine-tuning, namely _reasoning tuning_ and _location tuning_ . The _reasoning tuning_ stage aims to acquire potential linkage between coarse-grained geographical locations ( _i.e._ , country) and the associated positioning reasons. In _location tuning_ stage, we employ the curated high-quality data to further refine the model in fine-grained geo-localization ( _i.e._ , city) learning. Extensive experiments prove that _GeoReasoner_ outperforms previous models qualitatively and quantitatively. 

## **Acknowledgements** 

We would like to thank Yao Zhou and Wenqi Shao for their insightful discussions and Ziyao Gao for her assistance in drawing the figures in this paper. We also extend our gratitude to the anonymous reviewers for their valuable comments. This work is partially supported by the National Natural Science Foundation of China (62172398, 42171456, 52078343). 

## **Impact Statement** 

_GeoReasoner_ advances image-based geo-localization technologies that are pivotal for many applications such as autonomous navigation. The pipeline of constructing the dataset featuring high-locatability street views proves highly beneficial across multiple scenarios, such as urban studies, culture studies, and digital humanities, all of which are increasingly reliant on the analysis of high-quality street-view data. 

The proposed paradigm represents the fusion of LVLM with human inference knowledge, which has implications for the advancement of artificial intelligence (AI) that is more 

9 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

aligned with human cognition. The synergy can lead to the creation of AI that is not only more effective in complex inference tasks but also more understandable and relatable to human users. As AI becomes more pervasive in daily life, the importance of designing systems that are both transparent and capable of complex reasoning cannot be overstated. 

## **References** 

- Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. GPT-4 Technical Report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-VL: A Versatile VisionLanguage Model for Understanding, Localization, Text Reading, and Beyond. _arXiv preprint arXiv:2308.12966_ , 2023a. 

- Bai, Y., Shang, C., Li, Y., Shen, L., Jin, S., and Shen, Q. Transport Object Detection in Street View Imagery Using Decomposed Convolutional Neural Networks. _Mathematics_ , 11(18):3839, 2023b. 

- Campbell, A., Both, A., and Sun, Q. C. Detecting and mapping traffic signs from Google Street View images using deep learning and GIS. _Computers, Environment and Urban Systems_ , 77:101350, 2019. 

- Chalvatzaras, A., Pratikakis, I., and Amanatiadis, A. A. A Survey on Map-Based Localization Techniques for Autonomous Vehicles. _IEEE Transactions on Intelligent Vehicles_ , 8(2):1574–1596, 2022. 

- Cheng, B., Schwing, A., and Kirillov, A. Per-Pixel Classification is Not All You Need for Semantic Segmentation. _Advances in Neural Information Processing Systems_ , 34: 17864–17875, 2021. 

- Cheng, W., Wen, R., Huang, H., Miao, W., and Wang, C. OPTDP: Towards optimal personalized trajectory differential privacy for trajectory data publishing. _Neurocomputing_ , 472:201–211, 2022. 

- Clark, B., Kerrigan, A., Kulkarni, P. P., Cepeda, V. V., and Shah, M. Where We Are and What We’re Looking At: Query Based Worldwide Image Geo-Localization Using Hierarchies and Scenes. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 23182–23190, 2023. 

- Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei, F. Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pp. 4005–4019, 2023. 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In _Proceedings of International Conference on Learning Representations_ , 2021. 

- Haas, L., Alberti, S., and Skreta, M. Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization. _arXiv preprint arXiv:2302.00275_ , 2023. 

- Hays, J. and Efros, A. A. IM2GPS: estimating geographic information from a single image. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 1–8. IEEE, 2008. 

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-Rank Adaptation of Large Language Models. In _Proceedings of International Conference on Learning Representations_ , 2022. 

- Kenton, J. D. M.-W. C. and Toutanova, L. K. BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 4171–4186, 2019. 

- Li, Y., Wu, C., Li, L., Liu, Y., and Zhu, J. Caption Generation From Road Images for Traffic Scene Modeling. _IEEE Transactions on Intelligent Transportation Systems_ , 23(7):7805–7816, 2021. 

- Lin, J., Zheng, Z., Zhong, Z., Luo, Z., Li, S., Yang, Y., and Sebe, N. Joint Representation Learning and Keypoint Detection for Cross-View Geo-Localization. _IEEE Transactions on Image Processing_ , 31:3780–3792, 2022. 

- Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual Instruction Tuning. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Luo, G., Biamby, G., Darrell, T., Fried, D., and Rohrbach, A. G3: Geolocation via guidebook grounding. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pp. 5841–5853, 2022. 

- Muller-Budack, E., Pustu-Iren, K., and Ewerth, R.¨ Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification. In _Proceedings of the European Conference on Computer Vision_ , pp. 563–579, 2018. 

- Pramanick, S., Nowara, E. M., Gleason, J., Castillo, C. D., and Chellappa, R. Where in the World is this Image? Transformer-based Geo-localization in the Wild. In _Proceedings of the European Conference on Computer Vision_ , pp. 196–215, 2022. 

10 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

- Qiao, S., Ou, Y., Zhang, N., Chen, X., Yao, Y., Deng, S., Tan, C., Huang, F., and Chen, H. Reasoning with Language Model Prompting: A Survey. In _Proceedings of the Annual Meeting of the Association for Computational Linguistics_ , pp. 5368–5393, 2023. 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning Transferable Visual Models From Natural Language Supervision. In _Proceedings of International Conference on Machine Learning_ , pp. 8748–8763. PMLR, 2021. 

- Rao, J., Shan, Z., Liu, L., Zhou, Y., and Yang, Y. Retrievalbased Knowledge Augmented Vision Language Pretraining. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 5399–5409, 2023. 

- Reimers, N. and Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing_ , pp. 3980–3990, 2019. 

- Seo, P. H., Weyand, T., Sim, J., and Han, B. CPlaNet: Enhancing Image Geolocalization by Combinatorial Partitioning of Maps. In _Proceedings of the European Conference on Computer Vision_ , pp. 536–551, 2018. 

- Shao, Z., Yu, Z., Wang, M., and Yu, J. Prompting Large Language Models with Answer Heuristics for Knowledgebased Visual Question Answering. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 14974–14983, 2023. 

- Shen, Q., Zeng, W., Ye, Y., Mueller Arisona, S., Schubiger, S., Burkhard, R., and Qu, H. StreetVizor: Visual Exploration of Human-Scale Urban Forms Based on Street Views. _IEEE Transactions on Visualization and Computer Graphics_ , 24(1):1004–1013, 2018. 

- Theiner, J., Muller-Budack, E., and Ewerth, R.¨ Interpretable Semantic Photo Geolocation. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 750–760, 2022. 

- Vivanco Cepeda, V., Nayak, G. K., and Shah, M. GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Vo, N., Jacobs, N., and Hays, J. Revisiting IM2GPS in the Deep Learning Era. In _Proceedings of the IEEE International Conference on Computer Vision_ , pp. 2621– 2630, 2017. 

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. _Advances in Neural Information Processing Systems_ , 35: 24824–24837, 2022. 

- Weyand, T., Kostrikov, I., and Philbin, J. PlaNet - Photo Geolocation with Convolutional Neural Networks. In _Proceedings of the European Conference on Computer Vision_ , pp. 37–55, 2016. 

- Xu, P., Shao, W., Zhang, K., Gao, P., Liu, S., Lei, M., Meng, F., Huang, S., Qiao, Y., and Luo, P. LVLM-eHub: A Comprehensive Evaluation Benchmark for Large VisionLanguage Models. _arXiv preprint arXiv:2306.09265_ , 2023. 

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Ye, Y., Richards, D., Lu, Y., Song, X., Zhuang, Y., Zeng, W., and Zhong, T. Measuring daily accessed street greenery: A human-scale approach for informing better urban planning practices. _Landscape and Urban Planning_ , 191: 103434, 2019a. 

- Ye, Y., Zeng, W., Shen, Q., Zhang, X., and Lu, Y. The visual quality of streets: A human-centred continuous measurement based on machine learning algorithms and street view images. _Environment and Planning B: Urban Analytics and City Science_ , 46(8):1439–1457, 2019b. 

- Ying, K., Meng, F., Wang, J., Li, Z., Lin, H., Yang, Y., Zhang, H., Zhang, W., Lin, Y., Liu, S., et al. MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI. _arXiv preprint arXiv:2404.16006_ , 2024. 

- Zhang, F., Zhang, D., Liu, Y., and Lin, H. Representing place locales using scene elements. _Computers, Environment and Urban Systems_ , 71:153–164, 2018. 

- Zhang, H., Song, H., Li, S., Zhou, M., and Song, D. A Survey of Controllable Text Generation Using Transformerbased Pre-trained Language Models. _ACM Computing Surveys_ , 56(3):1–37, 2023a. 

- Zhang, X., Li, X., Sultani, W., Zhou, Y., and Wshah, S. Cross-View Geo-Localization via Learning Disentangled Geometric Layout Correspondence. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pp. 3480–3488, 2023b. 

- Zhu, S., Shah, M., and Chen, C. TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 1162–1171, 2022. 

11 

**GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model** 

## **A. Implementation Details** 

Table 5 and Table 6 present the hyper-parameter settings and training details for the models. We conducted training and testing on Nvidia A800 (80G), with CUDA 12.1, PyTorch 2.0.0, and Transformers 4.33.0. 

_Table 5._ The hyper-parameter settings of the proposed _GeoReasoner_ . 

|Hyper Params|Value|
|---|---|
|Learning Rate|1e-5|
|Total Batch Size|64|
|Weight Decay|0.1|
|Warmup Ratio|0.01|
|Optimizer|AdamW|
|Adam Beta1|0.9|
|Adam Beta2|0.95|
|LR Scheduler|cosine|
|Model Max Length|2048|



_Table 6._ The training details of the proposed _GeoReasoner_ . 

|||Training Speed|Inference Latency|Num of<br>Base Model|Params<br>LoRA|Flops|
|---|---|---|---|---|---|---|
|LoRA1|(reason)|0.41 sample/s|1.560s|9.6B|112.19M|71.9B|
|LoRA2|(location)|0.63 sample/s|0.894s|9.6B|112.19M|71.9B|



## **B. Additional Qualitative Results** 

Additionally, we present the results of the _GeoReasoner_ on alternative street-view images, depicted in Figure 9. Each street view image is annotated with the ground truth geographic location, along with the inference results from _GeoReasoner_ . It can provide geographical predictions accompanied by reasonable explanations. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model_images/icml_2024_GeoReasoner_Geo-localization_with_Reasoning_in_Street_Views_using_a_Large_Vision-Language_Model.pdf-0012-09.png)


**----- Start of picture text -----**<br>
Ground Truth Vienna, Austria GeoReasoner Ground Truth Boston, USA GeoReasoner<br>'country': 'Austria', 'city': 'Vienna'   'country': 'United States', 'city': 'Boston'<br>'reasons': ['The picture shows the Messe  'reasons': ['Massachusetts Institute of<br>Wien in Vienna, Austria', 'The flagpoles in  Technology is located in Boston,<br>front of the building carry the flags of Austria  Massachusetts']<br>and Vienna', 'The building has a glass facade<br>and is surrounded by trees and shrubs']<br>Ground Truth Tokyo, Japan GeoReasoner Ground Truth Doha, Qatar GeoReasoner<br>'country': 'Japan', 'city': 'Tokyo'   'country': 'Qatar', 'city': 'Doha'<br>'reasons': ['Mount Fuji is visible in the  'reasons': ['The picture shows the Souq<br>background', 'There are many street signs  Waqif, the old market in Doha, Qatar.']<br>and traffic lights', 'The street is lined with<br>shops and restaurants']<br>Ground Truth Vancouver, Canada GeoReasoner Ground Truth Fuzhou, China GeoReasoner<br>'country': 'Canada', 'city': 'Vancouver',<br>'reasons': ['Gastown is the oldest  'country': 'China', 'city': 'Fuzhou'<br>neighborhood in Vancouver, known for its  'reasons': ['The picture shows the Tulou of<br>steam clock and historic architecture.', 'The  the Hakka in Yongding, Fujian province', 'The<br>steam clock is a popular tourist attraction  sky is pink, which indicates that the time is<br>and a symbol of Gastown.', 'The  sunset', 'The lanterns are red, which is a<br>neighborhood is known for its trendy bars  traditional custom in Chinese festivals']<br>and restaurants.']<br>**----- End of picture text -----**<br>


_Figure 9._ Additional qualitative results from the proposed _GeoReasoner_ . 

12 

