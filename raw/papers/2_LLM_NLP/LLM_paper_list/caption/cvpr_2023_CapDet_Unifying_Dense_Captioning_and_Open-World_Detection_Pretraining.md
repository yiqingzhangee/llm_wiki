# **CapDet: Unifying Dense Captioning and Open-World Detection Pretraining** 

Yanxin Long[1][*] Youpeng Wen[1][*] Jianhua Han[2] Hang Xu[2] Pengzhen Ren[1] Wei Zhang[2] Shen Zhao[1][†] Xiaodan Liang[1] _[,]_[3][†] 

1Shenzhen Campus of Sun Yat-sen University 2Huawei Noah’s Ark Lab 3MBZUAI longyx9@mail2.sysu.edu.cn, wenyoupeng0@outlook.com, hanjianhua4@huawei.com, chromexbjxh@gmail.com, renpzh@mail.sysu.edu.cn, wz.zhang@huawei.com, z-s-06@163.com, xdliang328@gmail.com 

## **Abstract** 

_Benefiting from large-scale vision-language pre-training on image-text pairs, open-world detection methods have shown superior generalization ability under the zero-shot or few-shot detection settings. However, a pre-defined category space is still required during the inference stage of existing methods and only the objects belonging to that space will be predicted. To introduce a “real” open-world detector, in this paper, we propose a novel method named CapDet to either predict under a given category list or directly generate the category of predicted bounding boxes. Specifically, we unify the open-world detection and dense caption tasks into a single yet effective framework by introducing an additional dense captioning head to generate the region-grounded captions. Besides, adding the captioning task will in turn benefit the generalization of detection performance since the captioning dataset covers more concepts. Experiment results show that by unifying the dense caption task, our CapDet has obtained significant performance improvements (e.g., +2.1% mAP on LVIS rare classes) over the baseline method on LVIS (1203 classes). Besides, our CapDet also achieves state-of-the-art performance on dense captioning tasks, e.g., 15.44% mAP on VG V1.2 and 13.98% on the VG-COCO dataset._ 

## **1. Introduction** 

Most state-of-the-art object detection methods [37, 38, 55] benefit from a large number of densely annotated detection datasets ( _e.g._ , COCO [30], Object365 [40], LVIS [14]). However, this closed-world setting results in the model only being able to predict categories that appear in the training set. Considering the ubiquity of new concepts in real-world scenes, it is very challenging to locate and identify these new visual concepts. This predictive ability of new concepts in open-world scenarios has very important research value 

> *Equal contribution. 

> †Corresponding authors. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0001-10.png)


**----- Start of picture text -----**<br>
sofa suitcase<br>unknown remote<br>bottle<br>lamp<br>unknown<br>table<br>(a) OWD (b) OVD<br>a white wall outlet<br>dense captioning<br>head<br>a black lamp<br>alignment<br>region embeddings bottle<br>text encoder<br>pre-defined<br>category list remote,table, bottle, suitcase, sofa,…<br>(C) CapDet (ours)<br>**----- End of picture text -----**<br>


Figure 1. Comparison of the different model predictions under OWD, OVD, and our setting. (a) OWD methods [16, 20, 53] are not able to describe the detailed category of the detected unknown objects and (b) the performance of OVD methods [9, 14, 45] usually depends on the pre-defined category list during the inference. (c) With the unification of two pipelines of dense captioning and open-world detection pre-training, our CapDet can either predict under a given category list or directly generate the description of predicted bounding boxes. 

in real-world applications such as object search [32, 34], instance registration [50], and human-object interaction modeling [12]. 

Currently, the open world scenario mainly includes two tasks: _open world object detection_ [20] (OWD) and _openvocabulary object detection_ [49] (OVD). Although the paradigms of OWD and OVD tasks are closer to the real world, the former cannot describe the specific concept of the detected unknown objects and requires a pre-defined category list during the inference. Specifically, as shown 

in Figure 1, previous OWD methods [16, 20, 53] would recognize new concepts not in the predefined category space as “unknown”. Further, another line of task OVD requires the model to learn a limited base class and generalize to novel classes. Compared to the _zero-shot object detection_ (ZSD) proposed by [36], OVD allows the model to use external knowledge, _e.g._ , knowledge distillation from a largescale vision-language pre-trained model [9, 14], imagecaption pairs [49], image classification data [54], grounding data [28, 45, 51]. With the external knowledge, OVD methods show a superior generalization capacity to detect the novel classes within a given category space. However, as shown in Figure 1, when given an incomplete category list, OVD can only predict the concepts that appear in the given category list, otherwise, there will be recognition errors, ( _i.e._ , as illustrated in Figure 1 (b), the OVD methods prone to predict the “wall socket” as “remote”, since the latter is in the category list but not the former). 

Thus, under the OVD setting, we mainly face the following two challenges: ( _i_ ) it is difficult to define a complete list of categories; ( _ii_ ) low response values on rare categories often lead to recognition errors. This is mainly because we cannot exhaustively enumerate new objects in the real world, and secondly, it is difficult to collect enough samples for rare classes. However, the fact that rare objects in the real world, even some new objects that are unknown to humans, such as UFOs, do not prevent people from using natural language to describe it as “a flying vehicle that looks like a Frisbee”. 

Therefore, based on the above observations, in this paper, we consider a new setting that is closer to the open world and real scenes, _i.e._ , we expect the model to both detect and recognize concepts in a given category list, and to generate corresponding natural language descriptions for new concepts or rare categories of objects. Early dense captioning methods [11,19] can locate salient regions in images and generate the region-grounded captions with natural language. Inspired by this, to address the challenges faced in the OVD setting, we propose to unify the two pipelines of dense captioning and open-world detection pre-training into one training framework, called **CapDet** . It empowers the model with the ability to both accurately detect and recognize common object categories and generate dense captions for unknown and rare categories by unifying the two training tasks. 

Specifically, our CapDet constructs a unified data format for the dense captioning data and detection data. With the data unification, CapDet further adopts a unified pretraining paradigm including open-world object detection and dense captioning pre-training. For open-world detection pretraining, we treat the detection task as a semantic alignment task and adopt a dual encoder structure as [45] to locate and predict the given concepts list. The concepts 

list contains category names in detection data and regiongrounded captions in dense captioning data. For dense captioning pretraining, CapDet proposes a dense captioning head to take the predicted proposals as input to generate the region-grounded captions. Due to the rich visual concepts in the dense captioning data , the integration of dense captioning tasks will in turn benefit the generalization of detection performance. 

Our experiments show that the integration of few dense captioning data brings in large improvement in the object detection datasets LVIS, _e.g._ , +2.7% mAP on LVIS. The unification of dense captioning and detection pre-training gains an additional 2.3% increment on LVIS and 2.1% increment on LVIS rare classes. Besides, our model also achieves state-of-the-art performance on dense captioning tasks. Note that our method is the first to unify dense captioning and open-world detection pretraining. To summarize, our contributions are three folds: 

- We propose a novel open-vocabulary object detection framework CapDet, which cannot only detect and recognize concepts in a given category list but also generate corresponding natural language descriptions for new concept objects. 

- We propose to unify the two pipelines of dense captioning and open-world detection pre-training into one training framework. Both two pre-training tasks are beneficial to each other. 

- Experiments show that by unified dense captioning task and detection task, our CapDet gains significant performance improvements on the open-vocabulary object detection task ( _e.g._ , +3.3% mAP on LVIS rare classes). Furthermore, our CapDet also achieves stateof-the-art performance on the dense captioning tasks, _e.g._ , 15.44% mAP on Visual Genome (VG) V1.2 and 13.98% mAP on VG-COCO. 

## **2. Related Work** 

**Vision-Language Pre-training.** Vision-Language Pretraining [7, 18, 35] as a scheme in the domains of natural language processing [1, 6] and computer vision [8] obtains continual attention currently. And it exhibits strong performance and generalization ability on various downstream vision and cross-modal tasks. Among them, CLIP [35] and ALIGN [18] as dual-stream methods utilize large-scale image-text pairs on the Internet by cross-modal contrastive learning to get excellent zero-shot classification ability. Single-stream methods [22,27] unify visual and textual embeddings in a single transformer-based model, which can perform text generation tasks such as image caption and VQA. Some mixed architectures [26, 43] combine singlestream and dual-stream to explore a unified way of visionlanguage understanding and generation. However, these methods take low-resolution images as input and serve the 

task of classification and retrieval. Those vision-language pre-training approaches can not be applied to pure computer vision task directly, _i.e._ , object detection task. 

**Open World Object Detection / Open-Vocabulary Object Detection.** Object detection is a core computer vision task, which aims at localizing objects using a bounding box and classifying them. The mature detection approaches which show great performance on supervised data include one-stage detectors ( _i.e._ , YOLO [37], ATSS [52]) having a relatively high detection efficiency and two-stage detectors ( _i.e._ , Faster R-CNN [38], Mask R-CNN [17]) having good detection accuracy. However, how to generalize these methods to rare classes and novel concepts in the real world is a challenge. Currently, several object detection approaches for such open-world scenes have attracted extensive attention from academia and industry. These methods are divided into two tasks which are called open-world object detection and open-vocabulary object detection respectively depending on whether to detect the class of unknown classes. 

For the OWD task, Zhao et al. [53] proposed a proposal advisor to assist in identifying unknown proposals without supervision and a class-specific expelling classifier to filter out confusing predictions. For the OVD task, GLIP [28] converts the detection data into grounding format and proposes a fusion module to learn semantic vision information in grounding data. K-Lite [42] reconstructs the input format of the data in GLIP from sequential to parallel and uses nouns hierarchy and definition to format text sequence. DetCLIP [45] unifies detection, grounding, and image-text pair data in a paralleled formulation and constructs a concept dictionary to augment the text data, which strikes a balance between performance and efficiency. Differing from all these works, our CapDet can generate an open-set caption of each region proposal to cover situations where the semantics of new object instances are not in the given category list. 

**Dense Captioning.** Dense captioning aims at generating detailed descriptions for local regions, which usually needs to locate visual regions with semantic information and generate captions for these regions. J. Johnson et al. [19] utilized a fully convolutional localization network to locate regions of interest (RoIs) and then describe them. Afterward, many methods [29, 47] based on Faster-RCNN [38] and LSTM [13] are proposed to do dense captioning. X. Li et al. [29] arrange RoI features as a sequence and put them into LSTM with the guidance of the region features to form the complementary object context features. This method also needs ground truth bounding boxes auxiliary tests to achieve good results. But limited by the forget gate mechanism of LSTM, the inputted sequence cannot be too long. Then, the transformer-based method TDC [41] is proposed 

to tackle the long sequence forgotten problem. Instead, our CapDet proposes a transformer-based caption head to generate a caption using a single-stage detector ATSS while simultaneously achieving open-world detection. 

## **3. Method** 

The overview of our proposed CapDet is shown in Figure 2. To construct a detector to either predict under a given category list or directly generate the concepts of predicted bounding boxes, we incorporate detection data and dense caption data together. In this section, we will present a unified data format for the detection data and dense caption data in Section 3.1, the model architecture and pre-training objectives for open-world object detection pre-training in Section 3.2 and dense captioning in Section 3.3. 

## **3.1. Unified Formulation** 

We defined a unified triplet-wise data format ( _x, {_ **b** _i}[N] i_ =1 _[, y] i[N]_ =1[)] for each sample from different sources. Specifically, _x ∈_ R[3] _[×][h][×][w]_ is the input image, _{_ **b** _i|_ **b** _i ∈_ R[4] _}[N] i_ =1[denotes][the][bounding][boxes] coordinates for each region of the image, and the _yi[N]_ =1 represents the concepts of the corresponding boxes. _N_ denotes the number of regions. A concept _yi_ formatted as a sentence contains the category and textual description of the corresponding region. In detection data, a concept _y_ consists of the category name and the corresponding definition from the concept dictionary [45], while _yi_ represents the region-grounded caption in dense caption data. For example, for an image _x_ in detection data, _yi_ can be: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0003-09.png)


For an image _x_ from dense captioning data, _yi_ can be: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0003-11.png)


With the triplet, we can learn a unified image-text alignment objective on the detection data and the dense captioning data. The unified formulation also ensures the joint training of open-world object detection pre-training and dense captioning. 

## **3.2. Open-World Object Detection Pre-training** 

Based on the unified formulation of detection data and dense captioning data, we regard the captions of regions in dense captioning data as a kind of category and utilize two different sources of data for the open-world object detection pre-training. Compared with the limited class list of detection data, dense caption data contains richer concepts and more semantic information than class names of individual regions. On the other hand, localization and recognition are two essential tasks of object detection. Traditional object 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Alignment Loss<br>cat1 1 0 0 0 1 0 cap1<br>Language Modeling Loss<br>+ EOS cat2 0 0 1 1 0 0 cap2<br>… cat3 0 1 0 0 0 1 cap3 Feed Forward<br>Cross-Modal<br>Attention<br>Feed Forward<br>Self Attention<br>Regression Loss<br>Self Attention Centerness Loss Image Encoder Φ� Dense Captioning<br>Text Encoder Φ� � Head Φ�<br>+ EOSEOS BOS +<br>… ��<br>category concept ��<br>definitions:<br>categories:  cat 1. a common domesticated dog. dense captions:<br>cat 1. cat 2. dogbackpack + cat 2. back or shoulder.a bag carried by a strap on your  category  cap 1.cap 2. the umpire of a baseball game the catcher is crouched down<br>cat 3. bicycle cat 3. a wheeled vehicle has two wheels  concepts list cap 3. mitt on the player’s hand<br>… and is moved by foot pedals. …<br>…<br>**----- End of picture text -----**<br>


Figure 2. The overall architecture of CapDet. The training paradigm of CapDet contains open-world object detection pre-training and dense captioning. In detection, CapDet contains a dual vision-language encoder. The image encoder generates region embeddings from detection and dense captioning data. The regression loss and centerness loss are introduced to regress the locations. The text encoder takes the category concepts as input to generate the embeddings from the [EOS] token. Then we treat the detection task as a matching task and adopt an alignment loss for the category embeddings and region embeddings. In dense captioning, an additional dense captioning head is proposed to take the region embeddings as input and generate the textual captions for corresponding regions with natural language. 

detection always focuses on the salient objects in the image. While the dense captioning data contains lots of annotations which are just parts of an object, _e.g._ , _an ear of an elephant_ , it is not suitable to learn those annotations for the localization task. Therefore, we only calculate the localization loss on detection data. 

As shown in Figure 2, CapDet predicts the regions and treats the recognition task as a region-category matching task. For efficient learning on the matching task, we adopt the negative sampling proposed by [45] to provide negative concepts to enlarge the concept space in a batch. Specifically, for each iteration, we randomly sample a negative concept set and add to the positive concept set (N samples) in a batch to obtain the final concept set _yi[M]_ =1[, where M rep-] resents the sum of the number of positive and negative samples. Finally, we format the triplet to ( _x, {_ **b** _i}[N] i_ =1 _[, y] i[M]_ =1[)][.] CapDet contains a dual vision-language encoder and takes the triplet ( _x, {_ **b** _i}[N] i_ =1 _[, y] i[M]_ =1[)][ as input.][The image en-] coder Φ _v_ is an object detector that can predict the bounding boxes of regions from the input image _x_ and output the region features _O ∈_ R _[K][×][D]_ . The text encoder Φ _l_ takes the concept set _yi[M]_ =1[as][input][and][obtains][the][text][embeddings] _W ∈_ R _[M][×][D]_ from the special token [EOS] concatenated with the text input. _K, D_ denotes the number of predicted 

regions and region feature dimensions. The alignment score matrix _S ∈_ R _[K][×][M]_ of regions and texts is calculated by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0004-05.png)


where _T_ denotes the transpose operation. A ground-truth alignment matrix _G ∈{_ 0 _,_ 1 _}[K][×][M]_ is constructed to indicate the matching relation of regions and concepts. The alignment loss _Lalign_ is calculated by the predicted alignment scores of regions _S_ and the ground-truth alignment matrix _G_ . Following [28, 45], we adopt the ATSS [52] detector as an image encoder, and _Lalign_ is typically a sigmoid focal loss. As a one-stage detector, the localization loss contains centeredness loss _Lcen_ and bounding box regression loss _Lreg_ . The training objective of detection pre-training can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0004-07.png)


where _α_ and _β_ denote the weights for the centerness loss _Lcen_ and box regression loss _Lreg_ , respectively. The _Lcen_ is the sigmoid loss and the _Lreg_ is the GIOU loss [39]. 

|MODEL|BACKBONE<br>PRE-TRAINDATA|IMAGESNUMBER<br>LVIS<br>AP<br>AP_r_ / AP_c_ / AP_f_|
|---|---|---|
||||
|MASK-RCNN [17]<br>ATSS [52]<br>ATSS [52]|SWIN-T<br>LVIS<br>SWIN-T<br>LVIS<br>SWIN-L<br>LVIS|0.1M<br>34.1<br>19.1 / 34.0 / 37.0<br>0.1M<br>33.6<br>19.7 / 32.4 / 37.2<br>0.1M<br>43.9<br>30.6 / 43.7 / 46.3|
||||
|MDETR [21]<br>GLIP-T(A) [28]<br>GLIP-T(C) [28]<br>GLIP-T [28]<br>K-LITE[42]<br>K-LITE[42]<br>GLIPV2-T [51]|RN101<br>GOLDG+<br>SWIN-T+DH+F<br>O365<br>SWIN-T+DH+F<br>O365,GOLDG<br>SWIN-T+DH+F<br>O365,GOLDG,CAP4M<br>SWIN-T<br>O365<br>SWIN-T<br>O365,GOLDG<br>SWIN-T+DH+F<br>O365,GOLDG,CAP4M|0.77M<br>24.2<br>20.9 / 24.3 / 24.2<br>0.66M<br>18.5<br>14.2 / 13.9 / 23.4<br>1.43M<br>24.9<br>17.7 / 19.5 / 31.0<br>5.43M<br>26.0<br>20.8 / 21.4 / 31.0<br>0.66M<br>21.3<br>14.8 / 18.6 / 24.8<br>1.43M<br>26.1<br>17.2 / 24.6 / 29.0<br>5.43M<br>29.0<br>-<br>/<br>-<br>/<br>-|
||||
|DETCLIP-T(A) [45]<br>SWIN-T<br>O365<br>0.66M<br>28.8<br>26.0 / 28.0 / 30.0<br>DETCLIP-T(B) [45]<br>SWIN-T<br>O365, GOLDG<br>1.43M<br>34.4<br>26.9 / 33.9 / 36.3<br>DETCLIP-T(C)* [45]<br>SWIN-T<br>O365, VG<br>0.73M<br>31.5<br>27.5 / 30.6 / 33.0|||
|CAPDET(OURS)|SWIN-T<br>O365, VG|**0.73M**<br>**33.8**<br>**29.6 / 32.8 / 35.5**|



Table 1. Zero-shot performance on LVIS [15] MiniVal5k datasets. AP _r_ / AP _c_ / AP _f_ indicate the AP values for rare, common, and frequent categories, respectively. “DH” and “F” in GLIP [28] baselines stand for the dynamic head [4] and cross-modal fusion, respectively. Baselines with * are implemented with our code base. GoldG+ denotes the GoldG plus the COCO [30] caption dataset. 

## **3.3. Dense Captioning** 

The open-world object detection pre-training ensures CapDet gains the capacity to detect under given an arbitrary category list. However, when the given category list is not complete enough to cover the potential classes on a new domain data, the detector will perform worse on the categories which are not in the given list. Considering such limitation, we propose a dense captioning head Φ _C_ to generate semantically rich concepts with natural language for the predicted proposals. In the dense captioning task, the model receives an image and produces a set of regions and the corresponding captions. The dense captioning head is a cross-modal decoder that takes the c predicted regions features _O_ generated by the image encoder as input. The captioning ( _i.e._ , language modeling ) loss is calculated by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0005-04.png)


where _yit_ means the t token in caption _yi_ corresponding to region feature _Oi_ , and _yi_ ( _τ<t_ ) means tokens before t in caption _yi_ . The overall pre-training loss can be written as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0005-06.png)


where _wd, wc_ denote the weighting factor of _Ldet_ and _Lcap_ . 

To minimize the gap in the type of bounding boxes between the detection data and dense captioning data, we propose a simple way to transform our detector as a classagnostic detector and only select the top _k_ regions based on the centeredness scores to adapt to the dense captioning 

task. We can fine-tune our CapDet on the dense captioning data to achieve better performance. Specifically, we propose ”object” as the foreground concept and ”background” as the background concept. The text encoder Φ _l_ outputs the concept embeddings _W[′] ∈_ R[2] _[×][D]_ . Then the alignment scores _S[′] ∈_ R _[K][×]_[2] is calculated by Eqn. 1. The captioning head takes the top _k_ most confident proposal embeddings based on centeredness scores as input to predict the regiongrounded captions. 

## **4. Experiment** 

**Implementation Details.** For the image encoder, we adopt the Swin-T backbone proposed in SwinTransformer [31] which is pre-trained on ImageNet-1K [5]. We use 12 layers 8 heads transformer as our text encoder and load a base model checkpoint released by FILIP [46], in order to make a fair comparison with DetCLIP [45]. The structure of the dense captioning head is consistent with that in the text encoder but trained from scratch for a fair comparison. We employ AdamW [23] optimizer and set the batch size to 32. The learning rate is set to 1 _._ 4 _×_ 10 _[−]_[4] for the parameters of the image encoder and detection head, and 1 _._ 4 _×_ 10 _[−]_[5] for the text encoder and dense captioning head. When fine-tuning the VG dataset to do the dense captioning task, we set the learning rate to 1 _._ 4 _×_ 10 _[−]_[4] . Without otherwise specified, all models are trained with 12 epochs and the learning rate is decayed with a factor of 0.1 at the 8-th and the 11-th epoch. The context token 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0006-00.png)


Figure 3. Qualitative visualizations between GLIP-T, DetCLIP-T(C) and CapDet. From top to down, the three rows of images show the LVIS zero-shot detection results of GLIP-T, DetCLIP-T(C), and CapDet respectively. All models are pre-trained on O365 and VG. 

length for input text is set to 20. We set the number of input captions to 150, and the number of the region features N is determined by the feature map. The loss weight factor _wc_ and _wd_ are both set to 1.0. We build our model on MMDetection [2] code base. 

**Dataset.** Our CapDet is trained with two types of data, including detection data and caption data. Following DetCLIP [45], we use Object365 [40] (it will be abbreviated as O365 in the following paper) as detection data, and sample 0.66M data from O365 v2 for training. Following GLIP [28] and DetCLIP [45], LVIS [15] MiniVal5k (defined in [21]) which has 5000 images is used for detection evaluation. Moreover, we remove the training samples contained in the LVIS dataset for fair zero-transfer evaluation. For dense captioning data, we mainly conduct our experiments on VG [24] V1.2 and VG-COCO (defined in [41]). Following [41], we allocate 77398 images for training and 5000 images for validation and testing on VG. As demonstrated in [24], the ground-truth bounding boxes of VG are much denser than the other object detection datasets, _i.e._ , the average number of per sample in MS COCO [30] is only 7.1 _vs._ 35.4 in VG. Then an intersection of VG V1.2 and MS COCO is proposed by [41] and is denoted as VG-COCO, which has 38080 images for training, 2489 for validation, and 2476 for testing. 

**Benchmark Settings.** We mainly evaluate our method on open-vocabulary object detection and dense captioning task. For open-vocabulary object detection, we evaluate the direct domain transfer on LVIS [15] which contains 1203 categories. Following [28, 45], we metric the zero-shot 

detection performance by the Average Precision (AP) on a 5k subset. The annotations of LVIS data are split into three folds, _i.e._ , rare, common, and frequency, based on the number of categories. Since there is almost no overlap between the rare classes and the classes of training dataset Objects365 [40], the AP of the rare classes shows a valuable zero-shot detection performance. For the dense captioning task, we follow the setting of [19] to evaluate the VG and VG-COCO. The evaluation metric we adopt is the mean Average Precision proposed by [19] which is calculated across a range of thresholds for both localization and language accuracy, _i.e._ , the intersection over union (IOU) thresholds .3, .4, .5, .6, .7 are used for localization and the METEOR score’ thresholds 0, .05, .1, .15, .2, .25 is adopted for evaluating the language generation. 

## **4.1. Open-world Detection Results** 

Table 1 shows our zero-shot object detection performance on LVIS. We mainly train our CapDet with the backbone Swin-T [31] on the detection data Objects365 [40] and dense captioning data (VG [24]). Since DetCLIP does not report the performance on O365 and VG, we train DetCLIP on the two datasets under the same settings and denote it as DetCLIP-T(C) for a fair comparison. Comparing the 11th row and 12th row, our CapDet outperforms DetCLIPT(C) on the same data scale and backbone with an extra simple caption head. Moreover, our model’s zero-shot performance even surpasses the fully-supervised model with the same backbone by a large margin on rare classes, _i.e._ , CapDet outperforms ATSS by 9.9%. 

**Qualitative Visualizations** Figure. 3 illustrates the detection results on LVIS [24] dataset from GLIP-T, DetCLIPT(C), and CapDet. All three models are trained on O365 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0007-00.png)


**----- Start of picture text -----**<br>
JIVC<br>CapDet<br> w/o ft<br>CapDet<br>  w/ ft<br>**----- End of picture text -----**<br>


Figure 4. Qualitative visualizations between JIVC and CapDet. “w/o ft” means do caption without finetune, while “w/ ft” means with finetune. 

and VG, and details are given in Section 4.3. Given a category list, the rare classes are detected more precisely by our CapDet, _e.g._ , “kitchen table” in the first column, “horse buggy” in the third column, and “fishbowl” in the sixth column that our model CapDet detects correctly but the other two not. 

## **4.2. Dense Captioning Results** 

Due to the target bounding boxes in dense captioning data containing lots of local structures of objects and being much denser than the bounding boxes in object detection data, we do not regress the bounding box in the pre-training stage. The previous works directly train on the dense captioning data and generate captions on the top _k_ proposals ranking by a confidence score. When fine-tuning our model on the VG dataset for the dense captioning tasks, we transform our CapDet into a class-agnostic detector. Specifically, we propose “object” as the foreground concept and “background” as the background concept for computing alignment scores. The scores are used as proposal confidences to predict the region-grounded captions. 

Table 2 and Table 3 show CapDet significantly outperforms the latest work TDC [41] by 2.5% on mAP on VG and TDC+ROCSU [41] by 2.08%, respectively. It is worth noticing that, even against given the ground-truth bounding boxes with the previous method COCG [29] denoted 

|Method|mAP(%)|
|---|---|
|FCLN [19]<br>5.16<br>JIVC [44]<br>9.96<br>ImgG [29]<br>9.68<br>COCD [29]<br>9.75<br>COCG [29]<br>10.39<br>CAG-Net [47]<br>10.51<br>TDC [41]<br>11.90||
|CapDet (Ours)|**15.44**|



Table 2. Comparison of mAP (%) performance on dense captioning benchmark on the VG V1.2 dataset. 

as COCOG&GT, our CapDet still gains a 43.80% mAP increase and achieves state-of-the-art. One important reason is that the excellent detection performance of our model assists the localization ability of dense captioning tasks. **Qualitative Visualizations.** Figure 4 shows a qualitative visualization comparison between JIVC [44] and our CapDet. The three image rows from top to bottom are the visualization of JIVC, CapDet without fine-tuning, and CapDet with finetuning. In the second row, CapDet can locate more objects than JIVC, owing to our model’s superior localization performance. After finetuning, CapDet can further describe a region rather than a single object such as 

|Method|mAP(%)|
|---|---|
|||
|FCLN [19]<br>4.23<br>JIVC [44]<br>7.85<br>Max Pooling [29]<br>7.86<br>ImgG [29]<br>7.81<br>COCD [29]<br>7.92<br>COCG [29]<br>8.90<br>COCG-LocSiz [29]<br>8.76<br>COCG&GT [29]<br>9.79<br>TDC+ROCSU [41]<br>11.9||
|CapDet (Ours)|**13.98**|



Table 3. Comparison of mAP (%) performance on the dense captioning benchmark on the VG-COCO Dataset. 

|MODEL|DC HEAD|LVIS<br>AP<br>AP_r_ / AP_c_ / AP_f_|LVIS<br>AP<br>AP_r_ / AP_c_ / AP_f_|
|---|---|---|---|
|GLIP-T<br>GLIP-T<br>DETCLIP-T<br>DETCLIP-T|✗<br>✓<br>✗<br>✓|30.4<br>33.1<br>31.5<br>33.8|22.5 / 29.0 / 33.0<br>27.0 / 32.1 / 35.0<br>27.5 / 30.6 / 33.0<br>29.6 / 32.8 / 35.5|



Table 4. Ablations on integrating our dense captioning head into different baselines. 

|different baselines.||||
|---|---|---|---|
|Pre-training Data|Fine-tune|DCap mAP(%)|Box mAP(%)|
|VG<br>O365,VG<br>VG<br>O365,VG|✗<br>✗<br>✓<br>✓|12.86<br>4.72<br>13.83<br>15.44|27.65<br>9.65<br>28.58<br>30.61|



Table 5. Ablations on incorporating data from different sources. “DCap” stands for the dense caption mAP. 

“two women in a kitchen” in the 5-th column. 

## **4.3. Ablation Studies** 

## **4.3.1 Ablations for Unified Pre-training** 

**Effect on different baselines.** Table 4 investigates the advantages of dense captioning heads on different baselines. We integrate our dense captioning head with GLIP-T or DetCLIP-T. The GLIP-T is implemented with parallel text encoding without external knowledge following the setting as ablations in [42] on our code base. All the results are pre-trained on Objects365 and VG. The results show that our dense captioning head is able to boost the generalization and model-agnostic. 

**Effect of dense captioning data.** Table 1 shows the efficiency of incorporating dense captioning data. Specifically, only 0.07M data added, the DetCLIP-T(C) gains +2.7% overall AP and +1.5% _APr_ on LVIS compared to DetCLIP- 

T(A). The performance of DetCLP-T(A) on rare categories also outperforms DetCLIP-T(C) train on Objects365 and GOLDG, while the data size is 1.43M _vs._ 0.73M. 

## **4.3.2 Ablations for dense captioning** 

We investigate the impact of training policy and data from different sources on the dense captioning task. As shown in row1 in Table 5, our CapDet still achieves a significant performance which is directly trained on VG outperforms the previous task ( _i.e._ , TDC [41] in Table 2). Row2 is our CapDet and is pre-trained on Objects365 and VG, while only the bounding box in the Objects365 is regressed, and then transformed on a dense captioning task. Since the type of bounding boxes in dense captioning is different from the detection data, the result of the direct transforming to dense captioning is worse. However, we’ve proved that our model still keeps the dense captioning capacity on the salient objects in Figure 4. The results in row3 and row4 indicate that pre-training on the detection data Objects365 is also beneficial to the dense captioning task. 

## **5. Limitations** 

These are a few issues that we need to improve in the future: (1) Although our unification training paradigm works well on open-vocabulary object detection and dense captioning task, the training of dense captioning generation costs lots of time. (2) In addition, existing dense captioning data is high-cost to collect. We will research how to collect large-scale dense captioning data by auto annotation and get better performance with the scaled-up data. 

## **6. Conclusion** 

In this paper, we propose a novel open-world object detection method named CapDet. Our CapDet is more practical in the open world and real scenes. Specifically, CapDet introduces a unification training framework including openworld object detection pre-training and dense captioning. The unification enables our CapDet to localize and recognize concepts in an arbitrary given category list or directly generate textual captions for predicted new concept objects. Experiments show that the design of unification is both beneficial to open-world object detection tasks and dense captioning tasks. In the future, our CapDet can be easily injected into any open world and real scenes tasks. The unification framework can also be integrated into any other OWD/OVD methods to generate semantic-rich concepts for unknown/novel objects. 

**Acknowledgements** We gratefully acknowledge the support of MindSpore[1] , CANN (Compute Architecture for Neural Networks) and Ascend AI Processor used for this research. 

1https://www.mindspore.cn/ 

## **Appendix for CapDet: Unifying Dense Captioning and Open-World Detection Pretraining** 

## **Appendix A. Detailed Experimental Settings** 

The detailed architecture parameters for different modules of CapDet are shown in Table 6. For the learning rate scheduler, we assign a base learning rate and then linearly warm it up to the peak learning rate according to the effective total batch size by a square root strategy, _lrpeak_ = _lrbase ×_ ~~�~~ batchsize _/_ 16, _e.g._ , we set image encoder base learning rate to 1 _×_ 10 _[−]_[4] and it automatically scales to 1 _._ 4 _×_ 10 _[−]_[4] . The training hyperparameters used for CapDet are shown in Table 7. 

|Image Encoder|Value|||
|---|---|---|---|
|backbone|swin-t|Hyperparameter|Value(%)|
|neck<br>input resolution|fpn<br>1333_×_800|Image encoder lr<br>Text encoder lr|1_._4_×_10_−_4<br>1_._4_×_10_−_5|
|Text Encoder|Value|Crossmodal decoder lr|1_._4_×_10_−_5|
|width<br>heads<br>layers|512<br>8<br>12|Learning policy<br>warmup ratio<br>warmup iters<br>batchsize|CosineAnnealing<br>0.0001<br>1000<br>32|
|Cross-Modal Decoder|Value|weight decay|0.05|
|width|512|_wc_|1|
|heads|12|_wd_|1|
|layers|12|||



Table 6. Detailed architecture parameters for different module. 

Table 7. The training hyperparameters used for CapDet. 

## **Appendix B. Fine-tuning Results on LVIS** 

We provide the fine-tuning results on LVIS in Table 8 below. We observe that CapDet outperforms the baseline DetCLIP with 1.2% AP on average and 6.5% AP on rare classes. Besides, though pre-trained with fewer data and tasks, CapDet shows a competitive performance compared with the GLIPv2. 

|MODEL|BACKBONE<br>PRE-TRAINDATA|IMAGESNUMBER<br>LVIS<br>AP<br>AP_r_ / AP_c_ / AP_f_|
|---|---|---|
||||
|DETCLIP-T(C)* [45]<br>GLIPV2-T [28]|SWIN-T<br>O365, VG<br>SWIN-T+DH+F<br>O365, GOLDG, CAP4M|0.73M<br>45.6<br>33.6 / 45.8 / 47.5<br>5.43M<br>50.6<br>-<br>/<br>-<br>/<br>-|
|CAPDET(OURS)|SWIN-T<br>**O365, VG**|**0.73M**<br>**47.2**<br>**40.1**/ **46.9** /**48.7**|



Table 8. Fine-tuning performance on LVIS [15] MiniVal5k datasets. AP _r_ /AP _c_ /AP _f_ indicate the AP values for rare, common, and frequent categories. ‘DH’ and ‘F’ in GLIP [28] baselines stand for the dynamic head [4] and cross-modal fusion. 

## **Appendix C. Open-World Detection Results on LVIS Full Validation Set** 

Table 9 reports our zero-shot object detection performance on LVIS [15] full validation set. Following [28, 45], we take the class names with additional manually designed prompts as input of text encoder. Comparing the 5th row and 6th row, our CapDet still outperforms DetCLIP-T(C) on the same data scale and backbone with an extra simple caption head. The zeroshot performance surpasses the previous methods with the same backbone by a large margin on rare classes, _e.g._ , CapDet trained on fewer data outperforms GLIP-T [28] by 10.8% on AP _r_ . 

## **Appendix D. Analysis of the Improvements on OVD** 

We attribute the improvements on OVD to the reason that the incorporation of captioning head brings more generalizability for the region features, which in turn helps the learning of OVD task. Specifically, the dense captioning task is essentially a sequential classification task with a large enough class space ( _i.e._ , word tokens), while alignment task is a single-step 

|MODEL|BACKBONE<br>PRE-TRAINDATA<br>IMAGESNUMBER|LVIS VALFULL<br>AP<br>AP_r_ / AP_c_ / AP_f_|
|---|---|---|
|GLIP-T(A) [28]<br>GLIP-T [28]|SWIN-T+DH+F<br>O365<br>0.66M<br>SWIN-T+DH+F<br>O365,GOLDG,CAP4M<br>5.43M|12.3<br>6.00 / 8.00 / 19.4<br>17.2<br>10.1 / 12.5 / 25.2|
|DETCLIP-T(A) [45]<br>SWIN-T<br>O365<br>0.66M<br>22.1<br>18.4 / 20.1 / 26.0<br>DETCLIP-T(C) [45]<br>SWIN-T<br>O365, VG<br>0.73M<br>23.5<br>18.4 / 21.6 / 27.9|||
|CAPDET(OURS)|SWIN-T<br>O365, VG<br>0.73M|**26.1**<br>**20.9**/**24.4**/ **30.2**|



Table 9. Zero-shot transfer performance on LVIS [15] full validation dataset. AP _r_ /AP _c_ /AP _f_ indicates the AP values for rare, common, and frequent categories. ‘DH’ and ‘F’ in GLIP [28] baselines stand for the dynamic head [4] and cross-modal fusion. 

classification task with a limited class space. Therefore, training with dense captioning tasks will bring the region feature into a more proper location in feature space rather than simply pulling them together via only detection task. As shown in Table 10, we further conduct the experiments to demonstrate the effectiveness of pre-training under captioning. By comparing the row 2 and 5, we observe that even with only dense captioning data (VG data), pre-training with the dense captioning paradigm also brings a significant improvement. 

|MODEL|PRE-TRAINDATA|LVIS<br>AP<br>AP_r_ / AP_c_ / AP_f_|
|---|---|---|
||||
|DETCLIP-T [45]|O365<br>VG<br>O365, VG|28.8<br>26.0 / 28.0 / 30.0<br>10.3<br>8.6 / 10.1 / 10.8<br>31.5<br>27.5 / 30.6 / 33.0|
||||
|CAPDET|O365<br>VG<br>O365, VG|28.5<br>25.2 / 27.5 / 29.9<br>11.4<br>10.2 / 11.1 / 11.8<br>**33.8**<br>**29.6** /**32.8** /**35.5**|



Table 10. Zero-shot performance on LVIS [15] MiniVal5k datasets. AP _r_ / AP _c_ / AP _f_ indicate the AP values for rare, common, and frequent categories, respectively. “DH” and “F” in GLIP [28] baselines stand for the dynamic head [4] and cross-modal fusion, respectively. 

## **Appendix E. ‘Real’ Open-world Object Detection Deployment Strategy** 

In this paper, the detection and dense captioning task are illustrated separately for better understanding and comparison with other methods, since no benchmark has considered combining these two tasks. For the practical deployment, we propose a simple two-stage ensemble way to stay true to the motivation. Specifically, in the first stage, we execute detection on images among the pre-defined categories list and treat the proposals with maximum alignment scores among all classes less than a threshold as ‘unknown’ objects. Then in the second stage, we generate the captions for the ‘unknown’ objects. To demonstrate the effectiveness of the proposed strategies, We conduct detection on the images with 80 categories of COCO and regenerate captions for the ‘unknown’ objects. As shown in the Figure 5 , our proposed strategy expands the semantic space of the limited categories list and shows reasonable results. 

## **Appendix F. More Ablation Studies** 

**Ablations on Pre-trained Language Model** Table 11 reports the effect of different tokenizers and pre-trained language models loaded for text encoder. We ablate two kinds of pre-trained language models and corresponding tokenizers for our text encoder. For dense captioning head, we construct the same decoder as BLIP [26] decoder and keep the tokenizer the same as the text encoder. The results indicate the FILIP [46] encoder with byte pair encoding performs a better generalization, since it is pre-trained on a larger scale of data, _i.e._ , 300M in FILIP [46] _vs._ 128M in BLIP [26]. 

**Ablations on the Weighting Factor of Dense Captioning Loss** We study the effect of weights of detection loss and dense captioning loss during pre-training. We set the weighting factor of detection loss _wd_ to 1.0. Table 12 provides the ablations of the weighting factor of dense captioning loss _wc_ . We choose _wc_ = 1 for CapDet, since the result of overall AP is the best. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0011-00.png)


Figure 5. Deployment results. 

|Pre-trained Model<br>Tokenizer<br>Vocab Size|DC Head|LVIS<br>AP<br>AP_r_ / AP_c_/ AP_f_|
|---|---|---|
|BLIP [26]<br>WordPiece<br>30524<br>✗<br>30.4<br>26.7 / 29.4 / 32.0<br>✓<br>32.4<br>27.4 / 31.8 / 33.9|||
|FILIP [46]<br>BPE<br>49408<br>✗<br>31.5<br>27.5 / 30.6 / 33.0<br>✓<br>33.8<br>29.6 / 32.8 / 35.5|||



Table 11. Effect of different tokenizers and language models. ‘DC Head’ and ‘BPE’ stand for the integration of Dense Captioning Head and Byte Pair Encoding. 

|_wc_|LVIS<br>AP<br>AP_r_ / AP_c_/ AP_f_|
|---|---|
|||
|0.5<br>1.0<br>1.5|33.6<br>31.0 / 32.8 / 34.9<br>**33.8**<br>29.6 / 32.8 / 35.5<br>33.5<br>32.0 / 32.1 / 35.0|



Table 12. Effect of weighting factor of dense captioning loss. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0011-06.png)


Figure 6. Qualitative visualizations on LVIS. 

## **Appendix G. More Qualitative Results** 

**Open-World Detection Results** Figure 6 illustrates more detection results on LVIS [15] dataset from our CapDet. We highlight the detected rare classes’s text in red. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0012-00.png)


Figure 7. Qualitative visualizations on VG. 

**Dense Captioning Results** Figure 7 shows more captioning results on VisualGenome [24] dataset. Our model CapDet locates not only “object” such as “bicycle” but also “region” such as “a shadow on the ground”. We also explored the zeroshot generalization ability of CapDet. We directly use our model to do the zero-shot dense captioning task without finetuning on serveral datasets, which include SBU [33], LVIS [15], Open Image [25], BDD100K [48], Pascal VOC [10] and COCO [3]. As shown in Figure 8, CapDet can accurately locate objects and generate corresponding region-grounded captions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-01.png)


**----- Start of picture text -----**<br>
SBU<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-03.png)


**----- Start of picture text -----**<br>
BDD100k<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-06.png)


**----- Start of picture text -----**<br>
LVIS<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-07.png)


**----- Start of picture text -----**<br>
Open Image<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/caption/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining_images/cvpr_2023_CapDet_Unifying_Dense_Captioning_and_Open-World_Detection_Pretraining.pdf-0013-10.png)


**----- Start of picture text -----**<br>
Pascal VOC COCO<br>**----- End of picture text -----**<br>


Figure 8. Qualitative visualizations on several datasets. 

## **References** 

- [1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. _CoRR_ , abs/2005.14165, 2020. 2 

- [2] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, and Dahua Lin. Mmdetection: Open mmlab detection toolbox and benchmark. _ArXiv_ , abs/1906.07155, 2019. 6 

- [3] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. _arXiv preprint arXiv:1504.00325_ , 2015. 12 

- [4] Xiyang Dai, Yinpeng Chen, Bin Xiao, Dongdong Chen, Mengchen Liu, Lu Yuan, and Lei Zhang. Dynamic head: Unifying object detection heads with attentions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 7373–7382, 2021. 5, 9, 10 

- [5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. _2009 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 248–255, 2009. 5 

- [6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. _CoRR_ , abs/1810.04805, 2018. 2 

- [7] Xiaoyi Dong, Yinglin Zheng, Jianmin Bao, Ting Zhang, Dongdong Chen, Hao Yang, Ming Zeng, Weiming Zhang, Lu Yuan, Dong Chen, et al. Maskclip: Masked self-distillation advances contrastive language-image pretraining. _arXiv preprint arXiv:2208.12262_ , 2022. 2 

- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. _CoRR_ , abs/2010.11929, 2020. 2 

- [9] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14084–14093, 2022. 1, 2 

- [10] Mark Everingham, SM Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. _International journal of computer vision_ , 111(1):98–136, 2015. 12 

- [11] Yiqi Gao, Xinglin Hou, Yuanmeng Zhang, Tiezheng Ge, Yuning Jiang, and Peng Wang. Caponimage: Context-driven densecaptioning on image. _arXiv preprint arXiv:2204.12974_ , 2022. 2 

- [12] Georgia Gkioxari, Ross Girshick, Piotr Doll´ar, and Kaiming He. Detecting and recognizing human-object interactions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 8359–8367, 2018. 1 

- [13] Alex Graves. Long short-term memory. _Supervised sequence labelling with recurrent neural networks_ , pages 37–45, 2012. 3 

- [14] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. _arXiv preprint arXiv:2104.13921_ , 2021. 1, 2 

- [15] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 5356–5364, 2019. 5, 6, 9, 10, 11, 12 

- [16] Akshita Gupta, Sanath Narayan, KJ Joseph, Salman Khan, Fahad Shahbaz Khan, and Mubarak Shah. Ow-detr: Open-world detection transformer. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 9235–9244, 2022. 1, 2 

- [17] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In _Proceedings of the IEEE international conference on computer vision_ , pages 2961–2969, 2017. 3, 5 

- [18] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In _International Conference on Machine Learning_ , pages 4904–4916. PMLR, 2021. 2 

- [19] Justin Johnson, Andrej Karpathy, and Li Fei-Fei. Densecap: Fully convolutional localization networks for dense captioning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 4565–4574, 2016. 2, 3, 6, 7, 8 

- [20] KJ Joseph, Salman Khan, Fahad Shahbaz Khan, and Vineeth N Balasubramanian. Towards open world object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5830–5840, 2021. 1, 2 

- [21] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1780–1790, 2021. 5, 6 

- [22] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In _International Conference on Machine Learning_ , pages 5583–5594. PMLR, 2021. 2 

- [23] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 5 

- [24] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123(1):32–73, 2017. 6, 12 

- [25] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4. _International Journal of Computer Vision_ , 128(7):1956–1981, 2020. 12 

- [26] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. _arXiv preprint arXiv:2201.12086_ , 2022. 2, 10, 11 

- [27] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. _arXiv preprint arXiv:1908.03557_ , 2019. 2 

- [28] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10965–10975, 2022. 2, 3, 4, 5, 6, 9, 10 

- [29] Xiangyang Li, Shuqiang Jiang, and Jungong Han. Learning object context for dense captioning. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 33, pages 8650–8657, 2019. 3, 7, 8 

- [30] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European conference on computer vision_ , pages 740–755. Springer, 2014. 1, 5, 6 

- [31] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. _2021 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 9992–10002, 2021. 5, 6 

- [32] Jingjing Meng, Junsong Yuan, Jiong Yang, Gang Wang, and Yap-Peng Tan. Object instance search in videos via spatio-temporal trajectory discovery. _IEEE Transactions on Multimedia_ , 18(1):116–127, 2015. 1 

- [33] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. _Advances in neural information processing systems_ , 24, 2011. 12 

- [34] James Philbin, Ondrej Chum, Michael Isard, Josef Sivic, and Andrew Zisserman. Object retrieval with large vocabularies and fast spatial matching. In _2007 IEEE conference on computer vision and pattern recognition_ , pages 1–8. IEEE, 2007. 1 

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning_ , pages 8748–8763. PMLR, 2021. 2 

- [36] Shafin Rahman, Salman Khan, and Fatih Porikli. Zero-shot object detection: Learning to simultaneously recognize and localize novel concepts. In _Asian Conference on Computer Vision_ , pages 547–563. Springer, 2018. 2 

- [37] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 779–788, 2016. 1, 3 

- [38] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in neural information processing systems_ , 28, 2015. 1, 3 

- [39] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 658–666, 2019. 4 

- [40] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 8430– 8439, 2019. 1, 6 

- [41] Zhuang Shao, Jungong Han, Demetris Marnerides, and Kurt Debattista. Region-object relation-aware dense captioning via transformer. _IEEE Transactions on Neural Networks and Learning Systems_ , 2022. 3, 6, 7, 8 

- [42] Sheng Shen, Chunyuan Li, Xiaowei Hu, Yujia Xie, Jianwei Yang, Pengchuan Zhang, Anna Rohrbach, Zhe Gan, Lijuan Wang, Lu Yuan, et al. K-lite: Learning transferable visual models with external knowledge. _arXiv preprint arXiv:2204.09222_ , 2022. 3, 5, 8 

- [43] Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei. Vlmo: Unified vision-language pre-training with mixture-of-modality-experts. _arXiv preprint arXiv:2111.02358_ , 2021. 2 

- [44] Linjie Yang, Kevin Tang, Jianchao Yang, and Li-Jia Li. Dense captioning with joint inference and visual context. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2193–2202, 2017. 7, 8 

- [45] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu. Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection. _arXiv preprint arXiv:2209.09407_ , 2022. 1, 2, 3, 4, 5, 6, 9, 10 

- [46] Lewei Yao, Runhu Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. _ArXiv_ , abs/2111.07783, 2022. 5, 10, 11 

- [47] Guojun Yin, Lu Sheng, Bin Liu, Nenghai Yu, Xiaogang Wang, and Jing Shao. Context and attribute grounded dense captioning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6241–6250, 2019. 3, 7 

- [48] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 2636–2645, 2020. 12 

- [49] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection using captions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14393–14402, 2021. 1, 2 

- [50] Haoyang Zhang and Xuming He. Deep free-form deformation network for object-mask registration. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 4251–4259, 2017. 1 

- [51] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. _arXiv preprint arXiv:2206.05836_ , 2022. 2, 5 

- [52] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 9759–9768, 2020. 3, 4, 5 

- [53] Xiaowei Zhao, Xianglong Liu, Yifan Shen, Yuqing Ma, Yixuan Qiao, and Duorui Wang. Revisiting open world object detection. _arXiv preprint arXiv:2201.00471_ , 2022. 1, 2, 3 

- [54] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Kr¨ahenb¨uhl, and Ishan Misra. Detecting twenty-thousand classes using imagelevel supervision. In _European Conference on Computer Vision_ , pages 350–368. Springer, 2022. 2 

- [55] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. _arXiv preprint arXiv:2010.04159_ , 2020. 1 

