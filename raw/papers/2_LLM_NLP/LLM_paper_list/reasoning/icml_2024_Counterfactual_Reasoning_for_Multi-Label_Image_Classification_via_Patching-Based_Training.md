# **Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

**Ming-Kun Xie**[* 1] **Jia-Hao Xiao**[* 1] **Pei Peng**[1] **Gang Niu**[2] **Masashi Sugiyama**[2 3] **Sheng-Jun Huang**[1] 

## **Abstract** 

## **1. Introduction** 

The key to multi-label image classification (MLC) is to improve model performance by leveraging label correlations. Unfortunately, it has been shown that overemphasizing co-occurrence relationships can cause the overfitting issue of the model, ultimately leading to performance degradation. In this paper, we provide a causal inference framework to show that the correlative features caused by the target object and its co-occurring objects can be regarded as a mediator, which has both positive and negative impacts on model predictions. On the positive side, the mediator enhances the recognition performance of the model by capturing co-occurrence relationships; on the negative side, it has the harmful causal effect that causes the model to make an incorrect prediction for the target object, even when only co-occurring objects are present in an image. To address this problem, we propose a counterfactual reasoning method to measure the total direct effect, achieved by enhancing the direct effect caused only by the target object. Due to the unknown location of the target object, we propose patching-based training and inference to accomplish this goal, which divides an image into multiple patches and identifies the pivot patch that contains the target object. Experimental results on multiple benchmark datasets with diverse configurations validate that the proposed method can achieve stateof-the-art performance. The implementation is available at https://github.com/xiemk/ MLC-PAT. 

> *Equal contribution 1Nanjing University of Aeronautics and Astronautics, Nanjing, China 2RIKEN Center for Advanced Intelligence Project, Tokyo, Japan[3] The University of Tokyo, Tokyo, Japan. Correspondence to: Sheng-Jun Huang _<_ huangsj@nuaa.edu.cn _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s). 

In _single-label_ supervised learning, each instance is assigned with a single label to describe its semantics, while many realistic scenarios exhibit a _multi-labeled_ nature, where each instance is associated with multiple labels (Zhang & Zhou, 2013; Liu et al., 2021b). For example, an image captured on the beach may simultaneously contain objects such as _sea_ , _person_ , and _palm tree_ . The goal of MLC is to develop a classification model based on examples with multiple semantics, enabling it to predict all relevant labels for unseen instances. 

In comparison to single-label learning, tackling the MLC tasks proves significantly more challenging, mainly due to the exponentially expanded output space. Given the prevalent co-occurrence of objects in the real world, many methods have emerged to address the MLC problems by exploiting label correlations. In the context of deep learning, the pioneering work (Wang et al., 2016) employed recurrent neural networks (RNNs) to capture high-order label correlations. Subsequent methods focused on integrating cooccurrence relationships into training, achieved through the design of specialized architectures, _e.g._ , graph convolutional networks (GCNs) (Chen et al., 2019b), or the development of innovative training schemes, _e.g._ , label mask training (Lanchantin et al., 2021). 

While these methods have improved the practical performance of MLC, an excessive emphasis on co-occurrence has been shown to misguide model training, leading to a degradation in overall performance (Xu et al., 2022). To address this problem, the recent work (Xu et al., 2022) decomposed the original task into co-occurrence and dis-occurrence subtasks. This enabled the former to capture joint patterns, while the latter to focus on class-specific features. Another study (Liu et al., 2023) proposed to treat co-occurring objects as confounders, recognizing the potential for incorrect causalities between objects and predictions. To mitigate this issue, it performed an attention-based intervention to eliminate the harmful confounding effect. 

In this paper, we conduct a systematic study to investigate the impact of label co-occurrence in MLC. We first reveal that when using the common binary cross entropy (BCE) or 

1 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

its variant, the asymmetric loss (ASL) (Ridnik et al., 2021b) loss, which does not explicitly consider co-occurrence relationships, the model inherently captures correlative patterns among objects co-occurring within a single image. Moreover, we provide a theoretical explanation of this phenomenon from a causal perspective. We find that the correlative features derived from the target object and its cooccurring objects can be regarded as a mediator, exerting both positive and negative influences on model predictions. On the positive side, it enhances the ability of the model to identify co-occurring objects by exploiting correlative patterns; on the negative side, it causes the model to overfit to correlative features, resulting in inaccurate predictions of non-existent objects due to the misguidance by their cooccurring ones. 

To address this problem, our main idea is to mitigate the negative impact of the mediated effect, _i.e._ , the indirect effect when the target object is masked yet the co-occurrence information remains activated due to the presence of co-occurring objects. This motivates us to access the total direct effect between the target object and the model prediction, with the goal of disentangling the positive effect from the negative side in a _counterfactual_ world. We prove that this goal can be achieved by strengthening the direct causal effect caused solely by the target object. Since the location of the target object is unknown, we propose patching-based inference to divide the original image into multiple patches and identify which patch contains the target object. Without the need for additional training, the method can be integrated into existing methods, enhancing their performance during inference in a flexible manner. Moreover, through the integration of patching into the training process, we introduce patching-based training to mitigate the feature distribution shift between the training and inference phases, resulting in further performance improvement. Experimental results on multiple benchmark datasets with diverse configurations verify that our method can achieve state-of-the-art performance. 

formation for MLC. The representative methods adopted a GCN (Chen et al., 2019b) or developed a label masking strategy (Lanchantin et al., 2021) to explore label correlations during training. Recent works study how to improve multi-label classification performance with limited supervision, including semi-supervised multi-label learning (Xie et al., 2023; Wang et al., 2020a), multi-label learning with partial labels (Durand et al., 2019; Xie et al., 2022), and multi-label learning with noisy labels (Xie & Huang, 2022; Li et al., 2022). 

With the intensive consideration of correlations among variables, Peal’s causal graphs (Pearl, 2009; 2001) and causal inference methods are widely used in machine learning. It provides solutions for achieving goals, _e.g._ , improving model robustness by pursuing causal effects. Interventionbased and counterfactual-inference-based approaches were primarily utilized to achieve these goals. The intervention is mainly reflected in the control of confounding variables. For example, VC R-CNN (Wang et al., 2020b) applied causal intervention to unsupervised regional feature learning, aiming to introduce a kind of “visual common sense” into model design. CONTA (Zhang et al., 2020) attributed the cause of ambiguous pseudo-mask boundaries to the confusion context, and then removed the confusion bias via the backdoor criterion for weakly supervised semantic segmentation. CaaM (Wang et al., 2021), as a novel causal attention module, self-annotated confounders in an unsupervised method and decoupled confounders and mediators via adversarial training. On the other hand, counterfactual-based techniques have often been applied to control or analyze causal relationships for tasks of interest. For example, while momentum may inadvertently bias tail predictions toward head predictions when training on long-tailed data, it also enhances representation learning and head predictions through an induced mediator (Tang et al., 2020). This paradoxical effect of momentum could be effectively disentangled by optimizing the total direct effect (TDE). 

## **3. Preliminaries** 

## **2. Related Work** 

With the rapid development of deep models, MLC has recently attracted increasing attention. Existing solutions to the MLC problems can be roughly divided into three groups. The first kind of methods improved the traditional BCE loss to address the inherent positive-negative imbalance problem in the MLC tasks (Ridnik et al., 2021b). The second kind focused on identifying regions of interest associated with semantic labels by utilizing the global average pooling (GAP) (Verelst et al., 2023) strategy or the attention technique (Lanchantin et al., 2021; Liu et al., 2021a; Ridnik et al., 2023b). The last kind aimed at modeling label correlations, which have been regarded as fundamental in- 

In this section, we first provide necessary notations; and then we derive our results on the influence of co-occurrence for MLC. 

## **3.1. Notations** 

Let _**i** ∈I_ be an instance and _**y** ⊆Y_ be its corresponding labels, where _I_ = R _[d]_ is the instance space and _Y_ = _{_ 0 _,_ 1 _}[q]_ is the label space with _q_ possible class labels. Here, _yk_ = 1 indicates that the _k_ -th label is relevant to the instance; otherwise, _yk_ = 0. Let _D_ = _{_ ( _**i** j,_ _**y** j_ ) _}[n] j_ =1[be][the][given] training set with _n_ examples. Our goal is to train a deep neural network (DNN) _f_ ( _·_ ) based on the training set _D_ that can output accurate predictions for unseen instances. 

2 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0003-01.png)


**----- Start of picture text -----**<br>
0.5 InT<br>0.8<br>0.4 DeT<br>0.6<br>0.3<br>0.4 0.2<br>0.2 0.1<br>0.0 0.0<br>(a) Conditional True Positive Ratio (b) Conditional False Positive Ratio<br>car;parking meterhandbag;backpackhandbag;suitcasebottle;refrigeratorknife;forkbowl;applebowl;orangebowl;broccolibowl;carrotbowl;toasterbroccoli;carrotchair;vasecouch;remotedining table;forkdining table;broccolidining table;cakemouse;keyboardremote;couchrefrigerator;microwavesink;toothbrushbook;couchbook;tvbook;mouse car;parking meterhandbag;backpackhandbag;suitcasebottle;refrigeratorknife;forkbowl;applebowl;orangebowl;broccolibowl;carrotbowl;toasterbroccoli;carrotchair;vasecouch;remotedining table;forkdining table;broccolidining table;cakemouse;keyboardremote;couchrefrigerator;microwavesink;toothbrushbook;couchbook;tvbook;mouse<br>Conditional TPR Conditional FPR<br>**----- End of picture text -----**<br>


_Figure 1._ Comparison results between InT (each class has a visual backbone) and DeT (all classes share a visual backbone) in terms of conditional TPR (a) and FPR (b) for given class pairs on MS-COCO. DeT achieves greater conditional TPR and FPR than InT, meaning the co-occurrence relationships can be encoded by feature representations, exerting both positive and negative effects on model predictions. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0003-03.png)


**----- Start of picture text -----**<br>
X :  target object<br>Z<br>Z :  co-occurring object<br>O O : correlative features<br>Y : model prediction<br>X Y<br>: causal link<br>**----- End of picture text -----**<br>


_Figure 2._ An illustration of the causal graph. The dashed line indicates these two objects co-occurring in an image. 

We use _f_ ( _**i**_ ) to denote the predicted probabilities over all classes and _fk_ ( _**i**_ ) to denote the predicted probability of the _k_ -th class for instance _**i**_ . We use [ _q_ ] to denote the integer set _{_ 1 _, · · · , q}_ . 

## **3.2. Co-Occurrence: A Double-Edged Sword for MLC** 

We first provide a causal perspective to reveal that cooccurrence has positive and negative effects on MLC. As shown in Figure 2, we construct a causal graph to systematically study how co-occurrence relationships affect predictions. Without loss of generality, we assume that there are four variables in the causal graph: image content consisting of target object _X_ and its co-occurring object _Z_ , correlative features _O_ , and model prediction _Y_ . Note that i) we do not know the specific locations of _X_ and _Z_ in the image; ii) the number of co-occurring objects can be more than one. In the causal graph, each causal link indicates causalities between two nodes, _e.g._ , _X → Y_ means that the prediction _Y_ is generated based on the features of _X_ . 

To disclose the practical rationality of our causal graph, let us first take a brief look at MLC in the context of deep learning. To train a DNN, the most commonly used MLC loss function is the BCE loss or its improved variant ASL loss (Ridnik et al., 2021b), which decomposes the original task 

into multiple binary classification problems. For simplicity, we provide the definition of the BCE loss as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0003-10.png)


Although the loss function does not explicitly consider label correlations, our interesting finding is that the model would inherently capture co-occurrence relationships via feature representation learning. We perform experiments on the benchmark dataset MS-COCO (Lin et al., 2014) to show this interesting finding. Specifically, we use two strategies to train the models, dependent training (DeT), which jointly trains a DNN for all classes together, _i.e._ , the visual feature extractor is shared among all classes, and independent training (InT), which trains a DNN for each individual class, _i.e._ , each class has its own feature extractor. Figure 1(a) shows conditional true positive ratio (TPR) for a specific class given the presence of its co-occurring class, _e.g._ , _{_ car;parking meter _}_ represents the proportion of images containing both objects car and parking meter, in which the model predicts the presence of the object car. From the figure, it can be observed that DeT achieves better performance than InT on given class pairs. This phenomenon goes against the intuition that InT should have outperformed DeT, since it trains an individual model for each class using all training examples, owning stronger learning capacity. The only difference between DeT and InT lies in the fact that the former utilizes a model to fit all relevant labels within an image, allowing for automatic capture of co-occurrence information. This results in better model performance on co-occurring classes. These results disclose that even without explicitly integrating co-occurrence relationships, DeT can capture correlative patterns, thereby enhancing the model’s recognition ability. 

This phenomenon can be explained by the link ( _X, Z_ ) _→ O_ , signifying that the target object and its co-occurring 

3 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

object jointly contribute to the correlative features _O_ . Unlike InT, which derives the prediction _Y_ solely through the path _X → Y_ , DeT has an additional path ( _X, Z_ ) _→ O → Y_ . This allows the model to enhance its performance on co-occurring objects by leveraging correlative patterns, as shown in Figure 1(a). 

However, due to the powerful capacity of DNNs, they often tend to overfit to co-occurrence relationships. This can lead to incorrect predictions, indicating the presence of a particular object even when the image contains only its cooccurrences. Figure 1(b) provides an empirical validation, showing the conditional false positive ratio (FPR) with respect to the same class pairs in Figure 1(a). From the figure, we can see that DeT achieves significantly larger conditional FPR than InT, which discloses that DeT suffers from severe overfitting to correlative features in the classes where it can effectively capture co-occurrence information. 

This phenomenon can also be explained using our causal graph. Imagine a scenario where the object _X_ is not present in an image, but there is a co-occurring object _Z_ . During the inference process, the model can still predict _Y_ for _X_ through the path _Z → O → Y_ . The negative impact of co-occurrence relationships causes the model to make mistakes by predicting false positive labels. In one word, cooccurrence resembles a double-edged sword for MLC. On the positive side, it enhances the model’s recognition ability by capturing co-occurrence information; on the negative side, it induces overfitting to correlative features, resulting in incorrect predictions. 

## **4. The Proposed Method** 

To mitigate the negative side of mediated effect, a commonly used strategy is total direct effect (TDE) (Pearl, 2001; VanderWeele, 2013), designed to strengthen the direct causal effect along _X → Y_ . In the context of MLC, for a given image _**i**_ , we define TDE _k_ ( _**i**_ ) as the probability that the target object _**x**_ in the image belongs to class _k_ . 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0004-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0004-07.png)


where _X_ = _**x**_ 0 represents the operation of masking the features of target object _**x**_ . The subscript _**o**_ represents that the mediator _O_ takes the effect of the correlative features _**o**_ on the prediction. The goal of TDE is to preserve the positive side via the mediation path while mitigating the negative effect caused by the mediator. Specifically, the first term in Eq.(2) represents the prediction by incorporating the co-occurrence information _**o**_ caused by _**x**_ and _**z**_ ; while the second term denotes the prediction when the target object _**x**_ is masked (denoted by _**x**_ 0) but the co-occurrence information is still _**o**_ caused by the co-occurring object _**z**_ . It is noteworthy that the second term actually measures the bias 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0004-09.png)


**----- Start of picture text -----**<br>
Z 𝒵<br>𝒵0<br>O O<br>X Y X Y<br>**----- End of picture text -----**<br>


_Figure 3._ An illustration of TDE inference for MLC. 

of overfitting to the co-occurrence relationship between _X_ and _Z_ , _i.e._ , the model still predicts target label _Y[k]_ as positive in the presence of the co-occurring object _Z_ , despite the absence of the target object _X_ . The subtraction operator exactly removes the negative influence of the mediated effect. 

During inference, it is straightforward to obtain the first probability of TDE by training a model with Eq. (1). Unfortunately, due to the unknown location of the target object, it is hard to obtain the second probability by masking the features of _**x**_ . Considering that the subtraction operator often introduces the issue of negative values, we transform the second probability into its complementary form, _i.e._ , the probability for the target object _**x**_ when masking its co-occurring object _**z**_ . Then, using this complementary probability, as illustrated in Figure 3, we rewrite TDE as the following equivalent form[1] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0004-13.png)


where _Z_ = _**z**_ 0 represents the operation of masking the features of the co-occurring object _**z**_ , and _λ_ is a datasetdependent parameter. The intuition behind this transformation is that it transfers the mitigation of the negative mediated effect caused only by the co-occurring objects _**z**_ into the enhancement of the direct causal effect caused only by the target object _**x**_ . Compared to InT, TDE leverages the co-occurrence information by considering the mediated effect along ( _X, Y_ ) _→ O → Y_ (the first term in Eq. (3)), significantly enhancing model performance; compared to DeT, TDE captures discriminative patterns by enhancing the direct effect along _X → Y_ (the second term in Eq. (3)), which prevents the model from overfitting to correlative features. 

**Patching-Based Inference** To obtain the predicted probability _P_ ( _Y_ _**o**[k]_[= 1] _[|][X]_[=] _**[ x]**[, Z]_[=] _**[ z]**_[0][)][, we propose a patching-] based inference (PAT-I) method. The main idea involves partitioning the original image into multiple patches and subsequently determining which patch contains the target object. Then, the prediction for the identified patch can be considered as an approximation of the prediction for the 

1The details can be found in Appendix A. 

4 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

target object. Specifically, during inference, for each image _**i**_ , we crop it into _m_ equally-sized patches _{_ _**u** j}[m] j_ =1[.][By] feeding the original image and the patches into the trained model _f_ ( _·_ ), we obtain the predictions _**p**_ = [ _p_ 1 _, ..., pq_ ] and _**q** j_ = [ _qj_ 1 _, ..., qjq_ ], where _pk_ = _fk_ ( _**i**_ ) and _qjk_ = _fk_ ( _**u** j_ ) denote the predicted logits of the class _k_ for the image _**i**_ and patch _**u** j_ respectively. 

Given the probability that the target object may be present in multiple patches, rather than selecting a particular patch, we assign a weight to each patch based on its predictions 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0005-03.png)


where _τ_ is a temperature parameter that controls the degree of emphasis on a particular patch. Then, the predicted logits for different objects can be obtained by aggregating the _m_ logits _**q** j_ in a spatially-weighted manner, _**q**_ = � _**w** j ·_ _**q** j_ . _j_ =1 

Finally, according to Eq. (3), we obtain the model prediction for _**x**_ of class _k_ as TDE _k_ ( _**i**_ ) = _δ_ ( _pk_ + _λ · qk_ ), where _δ_ ( _·_ ) is the _Sigmoid_ function. In our experiments, we observed that setting _λ_ = 1 achieves favorable performance. The reason for performing above operations on logits instead of probabilities is that we found applying the _Sigmoid_ function to logits to convert them into probabilities compresses the distinctive information, which in turn hinders the effectiveness of our method. Generally, PAT-I has two main advantages: i) simplicity - it can be incorporated directly into the inference process to enhance model performance without any additional training; ii) flexibility - it can be seamlessly integrated into any trained model. 

**Patching-Based Training** Although PAT-I can improve model performance when integrated into the inference process, it often encounters the challenge of feature distribution shift between the original images and patches. This is because the model is trained only on original images that exhibit a significant distribution shift with patches. This may cause the model to produce unconfident predictions, resulting in suboptimal performance. To address this problem, we propose patching-based training (PAT-T) to utilize both images and patches for model training, which aims to alleviate the distribution shift between the training and inference phases. 

Specifically, we decompose the trained model _f_ ( _·_ ) into two components _f_ ( _·_ ) = _h ◦ g_ ( _·_ ), where _g_ ( _·_ ) is a backbone and _h_ ( _·_ ) is a classification head. During training, for each image _**i**_ , we obtain the predictions _**p**_ and _**q** j_ in the same way. In practice, to further improve the model performance, we make two modifications. Firstly, considering that the classification head is sensitive to two different classification tasks on images and patches, we adopt two classifier heads _hϕ_ and _hψ_ to make predictions on images and patches indepen- 

dently, _i.e._ , _**p**_ = _hϕ ◦g_ ( _**i**_ ) and _**q** j_ = _hψ ◦g_ ( _**u** j_ ), where _ϕ_ and _ψ_ denote two sets of parameters. Secondly, we also use another head _hθ_ to obtain the predicted logits _**q** j[w]_[=] _[ h][θ][◦][g]_[(] _**[u]**[j]_[)][,] which are used to estimate weights _**w** j_ according to Eq. (4). By decoupling the dependence between weight estimation and logits prediction, it enhances model performance by avoiding the accumulation of prediction errors on a single head. 

Given the predictions _**p**_ and _**q**_ on the image and patches, we define the classification loss of PAT-T as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0005-10.png)


where _ℓ_ ( _·, ·_ ) can be BCE or ASL loss. In our experiments, we use ASL loss because it often achieves better performance than BCE loss. 

## **5. Experiments** 

## **5.1. Experimental Settings** 

**Dataset** To evaluate the performance of the proposed method, we conduct experiments on three benchmark datasets, including MS-COCO 2014[2] (Lin et al., 2014), Pascal VOC 2007[3] (Everingham et al., 2010), and Visual Genome[4] (Krishna et al., 2017). MS-COCO contains 82,081 training images and 40,137 validation images for 80 classes, with an average of 2.9 labels per image. VOC 2007 consists of the _train-val_ set with 5,011 images and _test_ set with 4,952 images. Following previous works (Ridnik et al., 2021b; Liu et al., 2021a), we use _train-val_ set for training and _test_ set for testing. Visual Genome is a dataset that contains 108,249 images and covers 80,138 categories. Similar to the previous work (Chen et al., 2022), considering that most categories have very few examples, we conduct preprocessing on this dataset. The main steps include: i) merge categories: combine categories that share the same meaning; ii) filter by image count: exclude categories with fewer than 500 images. Finally, we obtain a dataset named VG-256 that contains 256 classes and 108,249 images. We randomly divide 70% for training and 30% for testing. 

**Evaluation Metric** Following previous works (Chen et al., 2019b; Ridnik et al., 2021b), we utilize the average precision (AP) for individual categories and the mean average precision (mAP) across all categories as our evaluation metrics. To provide a comprehensive assessment of model performance, we additionally showcase overall precision (OP), recall (OR), F1-measure (OF1), as well as per-category precision (CP), recall (CR), and F1-measure (CF1) for detailed comparisons. 

2https://cocodataset.org 

3http://host.robots.ox.ac.uk/pascal/VOC/ 

- 4https://homes.cs.washington.edu/˜ranjay/ 

- visualgenome/index.html 

5 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

_Table 1._ Comparison results on MS-COCO between our method and state-of-the-art methods. The backbones denoted with 21k and OI are pretrained on the ImageNet-21k and Open Image datasets, respectively. The best performance is highlighted in bold. All metrics are in %. 

|Method|Backbone|Resolution|mAP|CP|CR|CF1|OP|OR|OF1|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|ResNet-101 (He et al.,2016)<br>ML-GCN (Chen et al.,2019b)<br>KSSNet (Wang et al.,2020c)<br>MS-CMA (You et al.,2020)<br>MCAR (Gao & Zhou,2021)<br>CPSD (Xu et al.,2022)<br>IDA (Liu et al.,2023)|ResNet101<br>ResNet101<br>ResNet101<br>ResNet101<br>ResNet101<br>ResNet101<br>ResNet101|224_×_224<br>448_×_448<br>448_×_448<br>448_×_448<br>448_×_448<br>448_×_448<br>448_×_448|78.3<br>83.0<br>83.7<br>83.8<br>83.8<br>83.1<br>84.8|80.2<br>85.1<br>84.6<br>82.9<br>85.0<br>83.5<br>-|66.7<br>72.0<br>73.2<br>74.4<br>72.1<br>73.6<br>-|72.8<br>78.0<br>77.2<br>78.4<br>78.0<br>78.2<br>78.7|83.9<br>85.8<br>87.8<br>84.4<br>88.0<br>84.8<br>-|70.8<br>75.4<br>76.2<br>77.9<br>73.9<br>77.3<br>-|76.8<br>80.3<br>81.5<br>81.0<br>80.3<br>80.9<br>80.9|
|||||||||||
|PAT-T|ResNet101|448_×_448|85.0|87.6|74.4|80.5|88.4|77.9|82.8|
|||||||||||
|ASL (Ridnik et al.,2021b)<br>CPSD (Xu et al.,2022)<br>TResNetL (Ridnik et al.,2021a)|TResNetL<br>TResNetL<br>TResNetL (21k)|448_×_448<br>448_×_448<br>448_×_448|86.6<br>87.3<br>88.4|87.2<br>85.5<br>-|76.4<br>78.9<br>-|81.4<br>82.1<br>-|88.2<br>85.7<br>-|79.2<br>81.5<br>-|81.8<br>83.7<br>-|
|||||||||||
|PAT-T<br>PAT-T|TResNetL<br>TResNetL (21k)|448_×_448<br>448_×_448|88.7<br>90.6|89.0<br>90.6|78.8<br>81.0|83.6<br>85.5|89.3<br>90.9|81.3<br>82.9|85.1<br>86.7|
|||||||||||
|Q2L-TResL (Liu et al.,2021a)<br>Q2L-TResL|TResNetL<br>TResNet (21k)|448_×_448<br>448_×_448|87.3<br>89.2|87.6<br>86.3|76.5<br>81.4|81.6<br>83.8|88.4<br>86.5|78.5<br>83.3|83.1<br>84.9|
|PAT-T + Q2L-TResL<br>TResNetL<br>448_×_448<br>89.5<br>89.5<br>79.3<br>84.1<br>90.0<br>81.0<br>85.3<br>PAT-T + Q2L-TResL<br>TResNetL (21k)<br>448_×_448<br>91.0<br>89.1<br>82.7<br>85.8<br>89.8<br>83.9<br>86.8||||||||||
|||||||||||
|ML-Dec (Ridnik et al.,2023a)<br>Q2L-TResL<br>Q2L-SwinL<br>Q2L-CvT|TResNetL (OI)<br>TResNetL (OI)<br>Swin-L(21k)<br>CvT-w24(21k)|448_×_448<br>448_×_448<br>384_×_384<br>448_×_448|90.0<br>90.1<br>90.5<br>91.3|-<br>83.6<br>89.4<br>88.8|-<br>84.2<br>81.7<br>83.2|-<br>83.9<br>85.4<br>85.9|-<br>85.0<br>89.8<br>89.2|-<br>85.9<br>83.2<br>84.6|-<br>85.4<br>86.4<br>86.8|
|||||||||||
|PAT-T + ML-Dec<br>PAT-T + Q2L-TResL|TResNetL (OI)<br>TResNetL (OI)|448_×_448<br>448_×_448|91.4<br>**91.7**|90.1<br>89.7|82.9<br>83.4|**86.4**<br>**86.4**|90.5<br>90.3|84.8<br>84.9|**87.6**<br>87.5|



**Implementation Details** Following previous works (Ridnik et al., 2021b; Liu et al., 2021a), we choose ResNet101 and TResNetL (Ridnik et al., 2021c) pre-trained on ImageNet (Deng et al., 2009) or Open Image (Papadopoulos et al., 2017) as the backbone for our method. We employ Adam (Kingma & Ba, 2014) optimizer and one-cycle policy scheduler to train the model with maximal learning rate of 0.0001. Furthermore, we perform exponential moving average (EMA) (Tarvainen & Valpola, 2017) for the model parameters with a decay of 0.9997. We use the ASL loss as the base loss function since it shows superiority to BCE loss. We perform all experiments on GeForce RTX 3090 GPUs. The random seed is set to 1 for all experiments. 

## **6. Comparison with State-of-the-art Methods** 

than the comparing methods with the same configuration in terms of mAP, CF1 and OF1, which are the most important metrics, as other metrics are sensitive to the threshold. In particular, under the same configuration, _i.e._ , the same pre-trained back and resolution, PAT-T improves upon the performance of ASL by 2.1%, Q2L by 2.2%, ML-Decoder by 1.4%, respectively. Moreover, the optimal mAP score (91.7%) achieved by PAT-T is better than the current stateof-the-art mAP score (91.3%) achieved by Q2L-CvT. The backbone TResNet (54.7M) used by our method has a significantly smaller number of model parameters compared to CvT-w24 (277M) used by Q2L-CvT. These results convincingly show that PAT can achieve state-of-the-art performance based on a far smaller backbone. 

## **6.2. Performance on VOC 2007** 

## **6.1. Performance on MS-COCO** 

Table 1 reports the comparison results between our method PAT-T and state-of-the-art methods on MS-COCO. Due to the page limit, the results of PAT-I can be found in Appendix B. We can see that our method achieves better performance 

Table 2 reports the comparison results on VOC 2007. To make a fair comparison, following the previous works (Ridnik et al., 2021b; Liu et al., 2021a), we use TResNet pretrained on ImageNet-1K as the backbone. The results based 

6 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

_Table 2._ Comparison results on VOC 2007 between our method and state-of-the-art methods in terms of AP (%) and mAP (%). All results are reported at resolution 448 _×_ 448 except for the ADD-GCN and SSGRL, whose resolutions are 576 _×_ 576. The best performance is highlighted in bold. 

|Methods|aero bike bird boat bottle bus car cat chair cow table dog horse mbikepersonplant sheepsofa train tv|mAP|
|---|---|---|
|CNN-RNN<br>VGG+SVM<br>Fev+Lv<br>HCP<br>RDAL<br>RARL<br>SSGRL<br>MCAR<br>ASL(TResL)<br>ADD-GCN<br>Q2L-TResL|96.7 83.1 94.2 92.8 61.2 82.1 89.1 94.2 64.2 83.6 70.0 92.4 91.7<br>84.2<br>93.7<br>59.8 93.2 75.3**99.7**78.6<br>98.9 95.0 96.8 95.4 69.7 90.4 93.5 96.0 74.2 86.6 87.8 96.0 96.3<br>93.1<br>97.2<br>70.0 92.1 80.3 98.1 87.0<br>97.9 97.0 96.6 94.6 73.6 93.9 96.5 95.5 73.7 90.3 82.8 95.4 97.7<br>95.9<br>98.6<br>77.6 88.7 78.0 98.3 89.0<br>98.6 97.1 98.0 95.6 75.3 94.7 95.8 97.3 73.1 90.2 80.0 97.3 96.1<br>94.9<br>96.3<br>78.3 94.7 76.2 97.9 91.5<br>98.6 97.4 96.3 96.2 75.2 92.4 96.5 97.1 76.5 92.0 87.7 96.8 97.5<br>93.8<br>98.5<br>81.6 93.7 82.8 98.6 89.3<br>98.6 97.1 97.1 95.5 75.6 92.8 96.8 97.3 78.3 92.2 87.6 96.9 96.5<br>93.6<br>98.5<br>81.6 93.1 83.2 98.5 89.3<br>99.7 98.4 98.0 97.6 85.7 96.2 98.2 98.8 82.0 98.1 89.7 98.8 98.7<br>97.0<br>99.0<br>86.9 98.1 85.8 99.0 93.7<br>99.7**99.0**98.5 98.2 85.4 96.9 97.4 98.9 83.7 95.5 88.8 99.1 98.2<br>95.1<br>99.1<br>84.8 97.1 87.8 98.3 94.8<br>99.9 98.4 98.9 98.7 86.8 98.2 98.7 98.5 83.1 98.3 89.5 98.8 99.2<br>98.6<br>**99.3**<br>89.5 **99.4** 86.8 99.6 95.2<br>99.8**99.0**98.4**99.0** 86.7 98.1 98.5 98.3 **85.8** 98.3 88.9 98.8 99.0<br>97.4<br>99.2<br>88.3 98.7 **90.7**99.5**97.0**<br>99.9 98.9 99.0 98.4 87.7 98.6**98.8 99.1** 84.5 98.3 89.2 99.2 99.2<br>**99.2**<br>**99.3**<br>**90.2** 98.8 88.3 99.5 95.5|84.0<br>89.7<br>90.6<br>90.9<br>91.9<br>92.0<br>95.0<br>94.8<br>95.8<br>96.0<br>96.1|
|PAT-T + TResL|**100** 98.8**99.3**98.6 **88.8 99.0 98.8 99.1** 84.5 **98.9 90.1 99.5 99.4** 98.6<br>**99.3**<br>89.2 **99.4** 86.3**99.7**95.8|**96.2**|



_Table 3._ Comparison results on VG-256 between our method and state-of-the-art methods. All results are reported at resolution 448 _×_ 448. The best performance for each backbone is highlighted in bold. 

|Method|Backbone|mAP|CP|CR|CF1|OP|OR|OF1|
|---|---|---|---|---|---|---|---|---|
||||||||||
|ResNet101<br>PAT-I<br>PAT-T|ResNet101<br>ResNet101<br>ResNet101|47.7<br>48.5<br>**49.9**|51.1<br>53.9<br>52.0|48.2<br>46.3<br>51.7|49.6<br>49.8<br>**51.9**|63.3<br>64.0<br>63.5|64.0<br>64.8<br>66.0|63.6<br>64.4<br>**64.7**|
||||||||||
|TResNetL<br>PAT-I<br>PAT-T|TResNetL (21k)<br>TResNetL (21k)<br>TResNetL (21k)|53.9<br>54.8<br>**56.2**|57.2<br>58.7<br>58.0|53.4<br>52.5<br>56.0|55.2<br>55.4<br>**57.0**|66.7<br>66.1<br>67.1|67.7<br>69.1<br>69.3|67.2<br>67.6<br>**68.2**|
||||||||||
|Q2L-TResL<br>PAT-I+Q2L-TResL<br>PAT-T+Q2L-TResL|TResNetL (OI)<br>TResNetL (OI)<br>TResNetL (OI)|57.7<br>58.0<br>**59.5**|58.7<br>58.8<br>59.6|58.6<br>58.3<br>59.9|58.7<br>58.6<br>**59.7**|67.4<br>66.1<br>67.9|70.6<br>72.0<br>71.7|69.0<br>68.9<br>**69.8**|



on a stronger backbone can be found in Appendix B. Our method achieves the best performance when compared to state-of-the-art methods. Moreover, our method achieves the optimal performance in the majority of classes. It seems that our method does not yield as significant improvement as it did on MS-COCO. Except for the small number of images and relatively high performance achieved on VOC 2007, one possible reason may be that the co-occurrence relationships in the dataset are not as strong as in MS-COCO, as each example has only 1.5 labels on average. This makes models less prone to overfitting to label correlations. 

## **6.3. Performance on VG-256** 

Table 3 reports the comparison results between our methods PAT-I, PAT-T and the comparing methods on VG-256. We run all methods in the resolution of 448 _×_ 448. From the table, we can see that PAT-I can improve the model performance without any additional training cost. It enhances the positive impact while alleviating the negative impact of the mediated effect during the inference phase. Furthermore, by incorporating this idea into training, PAT-T achieves sig- 

nificant performance improvements and obtains the best performance. These results strongly verify the effectiveness of the proposed method. 

## **6.4. Ablation Studies** 

To demonstrate the effectiveness of our method in preventing models from overfitting to co-occurrence relationships, Figure 5 respectively illustrates the conditional FPR and conditional TPR of label pairs with a co-occurrence probability greater than 0.2. Baseline (ASL) and our PAT-T are trained with the same configuration (the backbone is TResNetL). In order to reduce the apparent fluctuations in the curve, we sort the label pairs according to the performance of Baseline. From the figure, we can see that the curve of PAT-T is generally below ASL, yielding that PAT-T can prevent the model from overfitting to co-occurrence, thereby making fewer mistakes in predicting false positive labels. Moreover, PATT also outperforms Baseline in terms of conditional TPR, indicating that PAT-T enhances the recognition capacity of the model significantly. 

7 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0008-01.png)


**----- Start of picture text -----**<br>
Traffic light Fire hydrant Bus<br>Baseline: 0.41 Baseline: 0.43 Baseline: 0.47<br>PAT-T: 0.85 PAT-T: 0.89 PAT-T: 0.93<br>Ori: 0.42, Pat: 0.98 Ori: 0.39, Pat: 0.99 Ori: 0.56, Pat: 0.99<br>0.07 0.88 0.02 0.01 0.74 0.01<br>0.01 0.03 0.95 0.02 0.23 0.01<br>Train Baseball bat Frisbee<br>Baseline: 0.91 Baseline: 0.82 Baseline: 0.78<br>PAT-T: 0.29 PAT-T: 0.41 PAT-T: 0.34<br>Ori: 0.58, Pat: 0.10 Ori: 0.71, Pat: 0.16 Ori: 0.71, Pat: 0.09<br>0.07 0.83 0.07 0.05 0.88 0.03<br>0.03 0.07 0.27 0.61 0.07 0.02<br>**----- End of picture text -----**<br>


_Figure 4._ Visualization of model predictions on MS-COCO. Ori denotes the probability predicted by PAT-T model on the original image; Pat denotes the probability predicted by PAT-T model on patches. The weights of each patch are reported in the four-grid. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0008-03.png)


**----- Start of picture text -----**<br>
0.5<br>0.4 Baseline<br>0.3 PAT-T<br>0.2<br>0.1<br>0.0<br>0 50 100 150 200<br>Sorted Pair Indices<br>1.0<br>0.9<br>0.8<br>0.7<br>0.6<br>0 50 100 150 200<br>Sorted Pair Indices<br>Conditional FPR<br>Conditional TPR<br>**----- End of picture text -----**<br>


_Figure 5._ Conditional FPR and conditional TPR of the label pairs with co-occurrence probabilities larger than 0.2 on MS-COCO. The pair indices are sorted according to the performance of Baseline. PAT-T outperforms Baseline in around 76% and 77% percentage of label pairs in terms of CFPR and CTPR. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0008-05.png)


**----- Start of picture text -----**<br>
0<br>5<br>0 10 20 30 40 50 60 70 80<br>Class Indices<br>(a) Original Prediction  p  vs Patch Prediction  q<br>4<br>2<br>0<br>0 10 20 30 40 50 60 70 80<br>Class Indices<br>(b) Optimum between ( p ,  q ) vs TDE Prediction  p  +  λ ·  q<br>MAP Increameant (%)<br>MAP Increameant (%)<br>**----- End of picture text -----**<br>


_Figure 6._ Corresponding to three output logits, original logits _**p**_ on an image, patch logits _**q**_ on patches, and TDE logits _**p**_ + _λ ·_ _**q**_ , we illustrate the per-class average precision increments between performance of each two logits to show the effectiveness of TDE. 

## **6.5. Step-Wise Evaluation** 

We perform a step-wise evaluation on MS-COCO to have a better understanding of how TDE keeps the positive side while removing the negative side of the mediated effect. Figure 6 (a) shows there is no significant difference between the performance of original logits _**p**_ and patch logits _**q**_ . In some classes, the former performs better, while in other classes, the latter exhibits superior performance. This means that introducing the patching technique does not lead to an improvement in model performance. Figure 6 (b) illustrates the improvements in performance of the proposed TDE logits on the basis of the optimum between the original logits _**p**_ and patch logits _**q**_ . We can see TDE maintains positive effects in almost every class, achieving the retention of positive causal effect and the removal of negative one. 

## **6.6. Case Studies** 

To disclose the mechanism behind PAT-T, as shown in Figure 4, we visualize some cases of predictions on MS-COCO. 

The first row reports the predictions of indistinguishable objects, _e.g._ , obscured, tiny, or ambiguous objects, when their co-occurring counterparts are present in the image. For such objects, our PAT-T gives the probabilities that are significantly higher than Baseline. This is because by focusing on a single patch instead of a whole image, our method is able to identify these objects (which can be validated by the probabilities on patches denoted by _Pat_ and weights in the four-grid), thereby enhancing the direct causal effect caused by target objects. The second row reports the predictions of objects that are absent, yet their co-occurring counterparts are present in the image. We can see Baseline suffers from overfitting to the correlative features caused by co-occuring objects. For example, in the second image, Baseline predicts a relatively high probability for _baseball bat_ due to the presence of its co-occurring object _baseball glove_ . While PAT-T predicts a smaller probability (0.41), since it does not find the target object in any patch, which can be validated by the predicted probabilities on patches (0.16). 

8 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

_Table 4._ Comparison results on MS-COCO between patch logits _**q**_ and TDE logits _**p**_ + _λ ·_ _**q**_ . 

|TDE logits**_p_**+_λ ·_**_q_**.|||
|---|---|---|
|Backbone|Patch Logits|TDE Logits|
||||
|TResNetL<br>TResNetL (21k)<br>TResNetL (OI)|87.1<br>88.8<br>89.5|88.7<br>89.2<br>90.8|
||||
|Q2L-TResL<br>Q2L-TResL (21k)<br>Q2L-TResL (OI)|87.9<br>89.1<br>90.1|89.5<br>91.0<br>91.7|



## **6.7. Study on Influence of Image Scale** 

In the PAT method, we found that training a backbone with images and patches results in a significant decline in performance. This is due to their inconsistent sizes, _i.e._ , the patches are only a quarter of the size of the image, severely disrupting feature representation learning. To deal with this problem, we first resize the patches to the size of the original image before inputting them into the model. Note that this operation does not incorporate additional information, because we first crop the resized image into patches and then resize them again. This operation only affects the scale of the targets. We perform experiments to verify that the main reason for the performance improvement is not due to the increase in object scale. Specifically, we train a model based on only resized patches using the loss function _ℓ_ ( _δ_ ( _**q**_ ) _,_ _**y**_ ) in Eq. (5). Table 4 reports mAP scores of patch logits and TDE logits. From the table, we can see that when the representation capability of the backbone is relatively weak, patch logits performs slightly better than the image logits (baseline results in Table 1 with corresponding backbones). As the representation capability of the backbone increases, this advantage disappears, and there is no significant difference in performance between patch logits and image logits. While our TDE logits still outperform patch logits with a large margin. These results convincingly validate that the effectiveness of our method stems from decoupling co-occurrence relationships using TDE. 

## **7. Conclusion** 

The paper studies the problem of multi-label learning, which aims to improve model performance by leveraging label correlations. Previous studies demonstrate that the powerful fitting capacity of DNNs often leads to overfitting to cooccurrence relationships. We establish a causal reasoning framework that treats the correlative features caused by the target object and its co-occurring ones as a mediator, exhibiting both positive and negative impacts on model predictions. To keep the positive side while removing the negative impact, we propose to enhance the direct causal 

effect caused by the target object when masking its cooccurring objects. In implementation, we propose patchingbased training to identify the pivot patch that contains the target object. Experimental results on multiple benchmark datasets validate the superiority of our method to state-ofthe-art methods. 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## **Acknowledgements** 

Sheng-Jun Huang was supported by Natural Science Foundation of Jiangsu Province of China (BK20222012, BK20211517), the National Key R&D Program of China (2020AAA0107000), and NSFC (62222605). Masashi Sugiyama was supported by JST CREST Grant Number JPMJCR18A2. 

## **References** 

- Chen, T., Wang, Z., Li, G., and Lin, L. Recurrent attentional reinforcement learning for multi-label image recognition. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

- Chen, T., Xu, M., Hui, X., Wu, H., and Lin, L. Learning semantic-specific graph representation for multi-label image recognition. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pp. 522–531, 2019a. 

- Chen, T., Pu, T., Wu, H., Xie, Y., and Lin, L. Structured semantic transfer for multi-label recognition with partial labels. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 36, pp. 339–346, 2022. 

- Chen, Z.-M., Wei, X.-S., Wang, P., and Guo, Y. Multi-label image recognition with graph convolutional networks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 5177–5186, 2019b. 

- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pp. 248–255. Ieee, 2009. 

- Durand, T., Mehrasa, N., and Mori, G. Learning a deep convnet for multi-label classification with partial labels. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 647–657, 2019. 

9 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

- Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman, A. The pascal visual object classes (voc) challenge. _International journal of computer vision_ , 88: 303–338, 2010. 

- Gao, B.-B. and Zhou, H.-Y. Learning to discover multiclass attentional regions for multi-label image recognition. _IEEE Transactions on Image Processing_ , 30:5920–5932, 2021. 

- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_ , 123:32–73, 2017. 

- Lanchantin, J., Wang, T., Ordonez, V., and Qi, Y. General multi-label image classification with transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16478–16488, 2021. 

- Li, S., Xia, X., Zhang, H., Zhan, Y., Ge, S., and Liu, T. Estimating noise transition matrix with label correlations for noisy multi-label learning. _Advances in Neural Information Processing Systems_ , 35:24184–24198, 2022. 

- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L.´ Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_ , pp. 740– 755. Springer, 2014. 

- Liu, R., Huang, J., Li, T. H., and Li, G. Causality compensated attention for contextual biased visual recognition. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- Liu, S., Zhang, L., Yang, X., Su, H., and Zhu, J. Query2label: A simple transformer way to multi-label classification. _arXiv preprint arXiv:2107.10834_ , 2021a. 

- Liu, W., Wang, H., Shen, X., and Tsang, I. W. The emerging trends of multi-label learning. _IEEE transactions on pattern analysis and machine intelligence_ , 44(11):7955– 7974, 2021b. 

- Papadopoulos, D. P., Uijlings, J. R., Keller, F., and Ferrari, V. Extreme clicking for efficient object annotation. In _Proceedings of the IEEE international conference on computer vision_ , pp. 4930–4939, 2017. 

- Pearl, J. Direct and indirect effects. In _Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence_ , pp. 411–420, 2001. 

- Pearl, J. Causal inference in statistics: An overview. _Statistics Surveys_ , 3:96 – 146, 2009. 

- Ridnik, T., Ben-Baruch, E., Noy, A., and Zelnik-Manor, L. Imagenet-21k pretraining for the masses. In _Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)_ , 2021a. 

- Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., and Zelnik-Manor, L. Asymmetric loss for multi-label classification. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 82–91, 2021b. 

- Ridnik, T., Lawen, H., Noy, A., Ben Baruch, E., Sharir, G., and Friedman, I. Tresnet: High performance gpudedicated architecture. In _proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pp. 1400–1409, 2021c. 

- Ridnik, T., Sharir, G., Ben-Cohen, A., Ben-Baruch, E., and Noy, A. Ml-decoder: Scalable and versatile classification head. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 32–41, January 2023a. 

- Ridnik, T., Sharir, G., Ben-Cohen, A., Ben-Baruch, E., and Noy, A. Ml-decoder: Scalable and versatile classification head. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pp. 32–41, 2023b. 

- Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ , 2014. 

- Tang, K., Huang, J., and Zhang, H. Long-tailed classification by keeping the good and removing the bad momentum causal effect. _Advances in Neural Information Processing Systems_ , 33:1513–1524, 2020. 

- Tarvainen, A. and Valpola, H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. _Advances in neural information processing systems_ , 30, 2017. 

- VanderWeele, T. J. A three-way decomposition of a total effect into direct, indirect, and interactive effects. _Epidemiology (Cambridge, Mass.)_ , 24(2):224, 2013. 

- Verelst, T., Rubenstein, P. K., Eichner, M., Tuytelaars, T., and Berman, M. Spatial consistency loss for training multi-label classifiers from single-label annotations. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 3879–3889, January 2023. 

10 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

- Wang, J., Yang, Y., Mao, J., Huang, Z., Huang, C., and Xu, W. Cnn-rnn: A unified framework for multi-label image classification. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 2285–2294, 2016. 

- Wang, L., Liu, Y., Qin, C., Sun, G., and Fu, Y. Dual relation semi-supervised multi-label learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 6227–6234, 2020a. 

- Wang, T., Huang, J., Zhang, H., and Sun, Q. Visual commonsense r-cnn. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 10760–10770, 2020b. 

- Wang, T., Zhou, C., Sun, Q., and Zhang, H. Causal attention for unbiased visual recognition. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 3091–3100, 2021. 

- Wang, Y., He, D., Li, F., Long, X., Zhou, Z., Ma, J., and Wen, S. Multi-label classification with label graph superimposing. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 34, pp. 12265–12272, 2020c. 

- Wang, Z., Chen, T., Li, G., Xu, R., and Lin, L. Multi-label image recognition by recurrently discovering attentional regions. In _Proceedings of the IEEE international conference on computer vision_ , pp. 464–472, 2017. 

_Neural Information Processing Systems_ , volume 36, pp. 25731–25747, 2023. 

   - Xu, J., Huang, S., Zhou, F., Huangfu, L., Zeng, D., and Liu, B. Boosting multi-label image classification with complementary parallel self-distillation. _arXiv preprint arXiv:2205.10986_ , 2022. 

   - Yang, H., Tianyi Zhou, J., Zhang, Y., Gao, B.-B., Wu, J., and Cai, J. Exploit bounding box annotations for multilabel object recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 280–288, 2016. 

   - You, R., Guo, Z., Cui, L., Long, X., Bao, Y., and Wen, S. Cross-modality attention with semantic graph embedding for multi-label classification. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pp. 12709–12716, 2020. 

   - Zhang, D., Zhang, H., Tang, J., Hua, X.-S., and Sun, Q. Causal intervention for weakly-supervised semantic segmentation. _Advances in Neural Information Processing Systems_ , 33:655–666, 2020. 

   - Zhang, M.-L. and Zhou, Z.-H. A review on multi-label learning algorithms. _IEEE transactions on knowledge and data engineering_ , 26(8):1819–1837, 2013. 

- Wei, Y., Xia, W., Lin, M., Huang, J., Ni, B., Dong, J., Zhao, Y., and Yan, S. Hcp: A flexible cnn framework for multi-label image classification. _IEEE transactions on pattern analysis and machine intelligence_ , 38(9):1901– 1907, 2015. 

- Wu, T., Huang, Q., Liu, Z., Wang, Y., and Lin, D. Distribution-balanced loss for multi-label classification in long-tailed datasets. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16_ , pp. 162–178. Springer, 2020. 

- Xie, M.-K. and Huang, S.-J. Ccmn: A general framework for learning with class-conditional multi-label noise. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 45(1):154–166, 2022. 

- Xie, M.-K., Xiao, J., and Huang, S.-J. Label-aware global consistency for multi-label learning with single positive labels. _Advances in Neural Information Processing Systems_ , 35:18430–18441, 2022. 

- Xie, M.-K., Xiao, J., Liu, H.-Z., Niu, G., Sugiyama, M., and Huang, S.-J. Class-distribution-aware pseudo-labeling for semi-supervised multi-label learning. In _Advances in_ 

11 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

_Table 5._ Comparison results on VOC 2007 between our method and state-of-the-art methods in terms of AP (%) and mAP (%). All results are reported at resolution 448 _×_ 448 except for the ADD-GCN and SSGRL, whose resolutions are 576 _×_ 576. The best performance is highlighted in bold. The asterisk (*) indicates the results reproduced using the released codes, as they were not provided in the original papers. 

|Methods|aero bike bird boat bottle bus car cat chair cow table dog horse mbikepersonplant sheepsofa train tv|mAP|
|---|---|---|
|CNN-RNN<br>VGG+SVM<br>Fev+Lv<br>HCP<br>RDAL<br>RARL<br>SSGRL<br>MCAR<br>ASL(TResL)<br>ADD-GCN<br>Q2L-TResL|96.7 83.1 94.2 92.8 61.2 82.1 89.1 94.2 64.2 83.6 70.0 92.4 91.7<br>84.2<br>93.7<br>59.8 93.2 75.3 99.7 78.6<br>98.9 95.0 96.8 95.4 69.7 90.4 93.5 96.0 74.2 86.6 87.8 96.0 96.3<br>93.1<br>97.2<br>70.0 92.1 80.3 98.1 87.0<br>97.9 97.0 96.6 94.6 73.6 93.9 96.5 95.5 73.7 90.3 82.8 95.4 97.7<br>95.9<br>98.6<br>77.6 88.7 78.0 98.3 89.0<br>98.6 97.1 98.0 95.6 75.3 94.7 95.8 97.3 73.1 90.2 80.0 97.3 96.1<br>94.9<br>96.3<br>78.3 94.7 76.2 97.9 91.5<br>98.6 97.4 96.3 96.2 75.2 92.4 96.5 97.1 76.5 92.0 87.7 96.8 97.5<br>93.8<br>98.5<br>81.6 93.7 82.8 98.6 89.3<br>98.6 97.1 97.1 95.5 75.6 92.8 96.8 97.3 78.3 92.2 87.6 96.9 96.5<br>93.6<br>98.5<br>81.6 93.1 83.2 98.5 89.3<br>99.7 98.4 98.0 97.6 85.7 96.2 98.2 98.8 82.0 98.1 89.7 98.8 98.7<br>97.0<br>99.0<br>86.9 98.1 85.8 99.0 93.7<br>99.7 99.0 98.5 98.2 85.4 96.9 97.4 98.9 83.7 95.5 88.8 99.1 98.2<br>95.1<br>99.1<br>84.8 97.1 87.8 98.3 94.8<br>99.9 98.4 98.9 98.7 86.8 98.2 98.7 98.5 83.1 98.3 89.5 98.8 99.2<br>98.6<br>99.3<br>89.5 99.4 86.8 99.6 95.2<br>99.8 99.0 98.4 99.0 86.7 98.1 98.5 98.3 85.8 98.3 88.9 98.8 99.0<br>97.4<br>99.2<br>88.3 98.7 **90.7**99.5**97.0**<br>99.9 98.9 99.0 98.4 87.7 98.6 98.8 99.1 84.5 98.3 89.2 99.2 99.2<br>99.2<br>99.3<br>90.2 98.8 88.3 99.5 95.5|84.0<br>89.7<br>90.6<br>90.9<br>91.9<br>92.0<br>95.0<br>94.8<br>95.8<br>96.0<br>96.1|
|PAT-T + TResL|**100** 98.8 99.3 98.6 88.8 **99.0**98.8 99.1 84.5 98.9 90.1 99.5 **99.4**<br>98.6<br>99.3<br>89.2 99.4 86.3 99.7 95.8|96.2|
|TResL* (21k)<br>Q2L-TResL*(21k)|**100** 99.1**99.6**98.6 89.5 97.9 99.1**99.4** 85.7 **100** 90.3 99.5 99.2<br>99.5<br>99.3<br>90.9 **99.9** 88.9 99.8 96.3<br>**100** 99.2 99.5 98.8 88.7 98.9 98.9**99.4 87.3 100** 89.8 99.3 98.0<br>**99.6**<br>**99.5**<br>91.1 **99.9** 90.6**99.9**96.3|96.6<br>96.7|
|PAT-T + TResL(21k)|**100 99.5**99.5**99.4 90.1** 98.4**99.2**99.3 86.8 99.9 **90.5 99.7** 99.3<br>99.3<br>99.4<br>**91.6 99.9** 89.8 99.8 96.8|**96.9**|



_Table 6._ Comparison results on MS-COCO between our method and state-of-the-art methods. The backbones denoted with 21k and OI are pretrained on the ImageNet-21k and Open Image datasets, respectively. All metrics are in %. 

|Method|Backbone|mAP|CP|CR|CF1|OP|OR|OF1|
|---|---|---|---|---|---|---|---|---|
|ASL<br>PAT-I<br>PAT-T|TResNetL<br>TResNetL<br>TResNetL|86.6<br>87.6<br>88.7|80.0<br>87.9<br>89.0|80.6<br>76.5<br>78.8|80.3<br>81.8<br>83.6|81.9<br>89.3<br>89.3|82.8<br>79.1<br>81.3|82.3<br>83.9<br>85.1|
|TResNetL<br>PAT-I<br>PAT-T|TResNetL (21k)<br>TResNetL (21k)<br>TResNetL(21k)|88.4<br>89.5<br>90.4|-<br>87.3<br>90.6|-<br>80.6<br>81.0|-<br>83.8<br>85.5|-<br>88.5<br>90.9|-<br>82.6<br>82.9|-<br>85.5<br>86.7|
|Q2L-TResL<br>PAT-I + Q2L-TResL<br>PAT-T + Q2L-TResL|TResNetL (21k)<br>TResNetL (21k)<br>TResNetL (21k)|89.2<br>89.8<br>91.0|86.3<br>85.7<br>89.1|81.4<br>81.1<br>82.7|83.8<br>83.3<br>85.8|86.5<br>87.4<br>89.8|83.3<br>82.4<br>83.9|84.9<br>84.8<br>86.8|
||||||||||
|ML-Decoder<br>PAT-I + ML-Decoder<br>PAT-T + ML-Decoder|TResNetL (OI)<br>TResNetL (OI)<br>TResNetL(OI)|90.0<br>90.5<br>91.4|-<br>86.0<br>90.1|-<br>82.9<br>82.9|-<br>84.4<br>86.4|-<br>87.5<br>90.5|-<br>84.4<br>84.8|-<br>86.0<br>87.6|



## **A. Derivation of Eq.** (3) 

Given an image containing the target object and its co-occurring object _{X_ = _**x** , Z_ = _**z** }_ , _{X_ = _**x** , Z_ = _**z**_ 0 _}_ and _{X_ = _**x**_ 0 _, Z_ = _**z** }_ form a partition, which are mutually exclusive and collectively exhaustive. According to the equation of total probability, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0012-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0012-08.png)


By dividing both sides by _P_ ( _X_ = _**x**_ 0 _, Z_ = _**z**_ ), we obtain 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0012-10.png)


_P_ ( _X_ = _**x** ,Z_ = _**z**_ )[(] _[X]_[=] _**[x]**[,][Z]_[=] _**[z]**_[0][)] where _α_ = _P_ ( _X_ = _**x**_ 0 _,Z_ = _**z**_ )[and] _[ β]_[=] _P[P]_ ( _X_ = _**x**_ 0 _,Z_ = _**z**_ )[are data-dependent parameters.] 

12 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

**Algorithm 1** Pseudocode of PAT-I in a PyTorch-like style 

# f: model # t: temperature # lam: tde parameter for [x; u ~~1~~ ; u ~~2~~ ; u ~~3~~ ; u ~~4~~ ], y in loader: # image logits and patch logits NxCx5 [l ~~x~~ ; l ~~u 1~~ ; l ~~u 2~~ ; l ~~u 3~~ ; l ~~u 4~~ ] = f([x; u ~~1~~ ; u ~~2~~ ; u ~~3~~ ; u ~~4~~ ]) # patch weights NxCx4 [w ~~1~~ ; w ~~2~~ ; w ~~3~~ ; w ~~4~~ ]=softmax([l ~~u 1~~ ;l ~~u 2~~ ;l ~~u 3~~ ;l ~~u 4~~ ]/t, dim=-1) # tde logits l ~~t~~ de = l ~~x~~ + lam*(w ~~1~~ *l ~~u 1~~ + w ~~2~~ *l ~~u 2~~ + w ~~3~~ *l ~~u 3~~ + w ~~4~~ *l ~~u 4~~ ).sum(dim=-1) # mAP score calculation mAP ~~s~~ core = mAP function(l ~~t~~ de, y) 

**Algorithm 2** Pseudocode of PAT-T in a PyTorch-like style 

# f: model # t: temperature for [x; u ~~1~~ ; u ~~2~~ ; u ~~3~~ ; u ~~4~~ ], y in loader: # image logits and patch logits NxCx5 [l ~~x~~ ; l ~~u 1~~ ; l ~~u 2~~ ; l ~~u 3~~ ; l ~~u 4~~ ] = f([x; u ~~1~~ ; u ~~2~~ ; u ~~3~~ ; u ~~4~~ ]) # patch weights NxCx4 [w ~~1~~ ; w ~~2~~ ; w ~~3~~ ; w ~~4~~ ]=softmax([l ~~u 1~~ ;l ~~u 2~~ ;l ~~u 3~~ ;l ~~u 4~~ ]/t, dim=-1) # weighted logits l ~~w~~ = (w 1*l ~~u 1~~ + w ~~2~~ *l ~~u 2~~ + w ~~3~~ *l ~~u 3~~ + w ~~4~~ *l ~~u 4~~ ).sum(dim=-1) # loss, Eq.(5) loss = AsymmetricLoss(l ~~x~~ , y) + AsymmetricLoss(l ~~w~~ , y) # model update loss.backward() 

Substituting the result of equation Eq. (7) into Eq. (2), we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training_images/icml_2024_Counterfactual_Reasoning_for_Multi-Label_Image_Classification_via_Patching-Based_Training.pdf-0013-06.png)


By dividing both sides by 1 _− α_ for each example, we obtain TDE _k_ ( _**i**_ ) = _P_ ( _Y_ _**o**[k]_[= 1] _[|][X]_[=] _**[ x]**[, Z]_[=] _**[ z]**_[) +] _[ λP]_[(] _[Y]_ _**o**[k]_[= 1] _[|][X]_[=] _**x** , Z_ = _**z**_ 0), where _λ_ = _β/_ (1 _− α_ ). We neglect the denominator (1 _− α_ ) for TDE _k_ ( _**i**_ ) since it does not affect the final result. 

## **B. Detailed Experimental Settings and Additional Results** 

On VOC 2007 dataset, our method is compared with the following representative algorithms: CNN-RNN (Wang et al., 2016), VGG+SVM (Simonyan & Zisserman, 2014), Fev+Lv (Yang et al., 2016), HCP (Wei et al., 2015), RDAL (Wang et al., 2017), RARL (Chen et al., 2018), SSGRL (Chen et al., 2019a), and MCAR (Gao & Zhou, 2021). 

Table 5 reports additional results based on TResNet (21k) on VOC 2007. We can see that our method outperforms the state-of-the-art methods. Table 6 reports the additional results of PAT-I on MS-COCO. As a comparison, we also report the results of baseline methods and PAT-T. From the table, we can see that PAT-I can achieve better performance than baseline methods, which verifies that the issue of overfitting to co-occurrence relationships can be addressed by incorporating counterfactual inference via the patching-based strategy. It is noteworthy that PAT-I does not even require training the model, which means that we can flexibly apply it to any trained model. 

13 

**Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training** 

## **C. Pseudocode of PAT-I and PAT-T** 

Algorithm 1 and Algorithm 2 provide the pseudocode of PAT-I and PAT-T in a Pytorch-like style, respectively. We can see that both two algorithms are simple to implement and act like plug-and-play plugins, which bring clear benefits. 

14 

