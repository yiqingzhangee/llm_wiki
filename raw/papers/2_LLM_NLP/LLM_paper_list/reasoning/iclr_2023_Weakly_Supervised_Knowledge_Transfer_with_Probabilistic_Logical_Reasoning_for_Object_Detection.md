Published as a conference paper at ICLR 2023 

WEAKLY SUPERVISED KNOWLEDGE TRANSFER WITH PROBABILISTIC LOGICAL REASONING FOR OBJECT DETECTION 

**Martijn Oldenhof** ESAT - STADIUS KU Leuven, Belgium martijn.oldenhof@kuleuven.be 

**Adam Arany** ESAT - STADIUS KU Leuven, Belgium adam.arany@esat.kuleuven.be 

**Yves Moreau** ESAT - STADIUS KU Leuven, Belgium yves.moreau@esat.kuleuven.be 

**Edward De Brouwer** ESAT - STADIUS KU Leuven, Belgium edward.debrouwer@gmail.com 

## ABSTRACT 

Training object detection models usually requires instance-level annotations, such as the positions and labels of all objects present in each image. Such supervision is unfortunately not always available and, more often, only image-level information is provided, also known as weak supervision. Recent works have addressed this limitation by leveraging knowledge from a richly annotated domain. However, the scope of weak supervision supported by these approaches has been very restrictive, preventing them to use all available information. In this work, we propose ProbKT, a framework based on probabilistic logical reasoning that allows to train object detection models with arbitrary types of weak supervision. We empirically show on different datasets that using all available information is beneficial as our ProbKT leads to significant improvement on target domain and better generalization compared to existing baselines. We also showcase the ability of our approach to handle complex logic statements as supervision signal. Our code is available at https://github.com/molden/ProbKT 

## 1 INTRODUCTION 

Object detection is a fundamental ability of numerous high-level machine learning pipelines such as autonomous driving [4; 16], augmented reality [42] or image retrieval [17]. However, training state-of-the-art object detection models generally requires detailed image annotations such as the boxcoordinates location and the labels of each object present in each image. If several large benchmark datasets with detailed annotations are available [26; 15], providing such detailed annotation on new specific datasets comes with a significant cost that is often not affordable for many applications. 

More frequently, datasets come with only limited annotation, also referred to as _weak supervision_ . This has sparked research in weakly-supervised object detection approaches [25; 6; 40], using techniques such as multiple instance learning [40] or variations of class activation maps [3]. However, these approaches have been shown to significantly underperform their fully-supervised counterparts in terms of robustness and accurate localization of the objects [39]. 

An appealing and intuitive approach to improve the performance of weakly supervised object detection is to perform transfer learning from an existing object detection model pre-trained on a fully annotated dataset [14; 46; 43]. This approach, also referred to as transfer learning or domain adaptation, consists in leveraging transferable knowledge from the pre-trained model (such as bounding boxes prediction capabilities) to the new weakly supervised domain. This transfer has been embodied in different ways in the literature. Examples include a simple fine-tuning of the classifier of bounding box proposals of the pre-trained model [43], or an iterative relabeling of the weakly supervised dataset for retraining a new full objects detection model on the re-labeled data [46]. 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0002-01.png)


Figure 1: **ProbKT** : Weakly supervised knowledge transfer with probabilistic logical reasoning. (Left) A model can be trained on the source domain using full supervision (labels, positions) but only on a limited set of shapes (cylinders and spheres). (Middle) The pre-trained model does not recognize the cubes from the target domain correctly. (Right) The model can adapt to the target domain after applying ProbKT and can recognize the cubes. 

However, existing approaches are very restrictive in the type of weak supervision they are able to harness. Indeed, some do not support new object classes in the new domain [20], others can only use a label indicating the presence of an object class [46]. However, in practice, the supervision on the new domain can come in very different forms. For instance, the count of each object class can be given, such as in atom detection from molecule images where only chemical formula might be given. Or, when many objects are present on an image, a range can be provided instead of an exact class counts ( _e.g._ "there are _at least_ 4 cats on this image"). Crucially, this variety of potential supervisory signals on the target domain cannot be fully utilized by existing domain adaption approaches. 

To address this limitation, we introduce ProbKT, a novel framework that allows to generalize knowledge transfer in object detection to arbitrary types of weak supervision using neural probabilistic logical reasoning [27]. This paradigm allows to connect probabilistic outputs of neural networks with logical rules and to infer the resulting probability of particular queries. One can then evaluate the probability of a query such as "the image contains at least two animals" and differentiate through the probabilistic engine to train the underlying neural network. Our approach allows for arbitrarily complex logical statements and therefore supports weak supervision like class counts or ranges, among other. To our knowledge, this is the first approach to allow for such versatility in utilizing the available information on the new domain. 

To assess the capabilities of this framework, we provide extensive empirical analysis of multiple object detection datasets. Our approach also supports any type of objects detection backbone architecture. We thus use two popular backbone architectures, DETR [7] and RCNN [34] and evaluate their performance in terms of accuracy, convergence as well of generalization on out-of-distribution data. Our experiments show that, due to its ability to use the complete supervisory signal, our approach outperforms previous works in a wide range of setups. 

_Key contributions_ : (1) We propose a novel knowledge transfer framework for object detection relying on probabilistic programming that uniquely allows using arbitrary types of weak supervision on the target domain. (2) We make our approach amenable to different levels of computational capabilities by proposing different approximations of ProbKT. (3) We provide an extensive experimental setup to study the capabilities of our framework for knowledge transfer and out-of-distribution generalization. 

## 2 RELATED WORKS 

A comparative summary of related works is given in Table 1. We distinguish three main categories: (1) pure weakly supervised object detection methods (WSOD) that do not leverage a richly annotated source domain, (2) unsupervised object detection methods with knowledge transfer (DA or domain adaptation methods) that do not use supervision on the target domain and (3) weakly supervised 

2 

Published as a conference paper at ICLR 2023 

object detection methods with knowledge transfer (WSOD w/transfer) that are restrictive in the type of supported weak supervision. To our knowledge, our work is the first to allow for arbitrary supervision on the target domain (and supporting new classes in the target domain) while also leveraging knowledge from richly annotated domains. ProbKT supports arbitrary weak supervision thanks to the inherited expressiveness of Prolog [41] which is based on a subset of first-order predicate logic, Horn clauses and is Turing-complete. 

**Weakly supervised object detection (WSOD)** This class of method allows training object detection models with only weak supervision. One can thus train these approaches directly on the target domain. However, they do not allow to leverage potentially available richly annotated datasets, which has been shown to lead to worse performance [39]. Different flavors of WSOD architectures have been proposed relying on a variety of implementations such as multiple instance learning (MIL)-based [25; 40] or class activation (CAM) based [47; 3]. In contrast to WSOD methods, our approach is designed to exploit existing richly annotated datasets and thus provides increased performance on the target domain. For a comprehensive review of WSOD methods we refer the reader to Shao et al. [39]. 

**Domain adaptation methods (DA)** In contrast to WSOD methods, domain adaptation methods do rely on fully supervised source domain dataset. However, they do not assume any supervision on the target domain and are therefore not equipped to exploit such signal when available [37; 8; 22; 48]. 

**WSOD with knowledge transfer** Our approach belongs to the class of weakly supervised object detection models with knowledge transfer. These methods aim to transfer knowledge from a source domain, where full supervision is available, to a target domain where only weak labels are available. Existing work in this class of models only allows for limited type of supervision of the target domain. Most architectures only support a label indicating the presence or absence of a class of object in the image[14; 46; 43]. Inoue et al. [20] allows for class counts as weak supervision but unfortunately does not allow for new classes in the target domain. In contrast, ProbKT natively allows for class counts and new classes as well as other types of weak supervision. 

**Neural probabilistic logical reasoning** Probabilistic logical reasoning combines logic and probability theory. Favored for its high-level reasoning abilities, it was introduced as an alternative way to deep learning in the quest for artificial intelligence [10]. Statistical artificial intelligence [32; 23] and probabilistic logic programming [11] are examples of areas relying on these premises. In a unification effort, researchers have proposed hybrid architectures, embedding both deep learning and logical reasoning components [38; 35]. Our work builds upon the recent advances in the field, where combinations of deep learning, logical, and _probabilistic_ approaches were introduced [27], allowing high-level reasoning with uncertainty using differentiable neural network architectures. 

|Method|Type|Annotated source dom.|Weak supervision|New classes|Implementation|
|---|---|---|---|---|---|
|Li et al. [25]|WSOD||presence/absence||MIL-based|
|Bilen and Vedaldi [6]|WSOD||presence/absence||spatial pyramid pooling layer|
|Song et al. [40]|WSOD||presence/absence||MIL based|
|Zhou et al. [47]|WSOD||mix||CAM-based|
|Bae et al. [3]|WSOD||mix||CAM based|
|Kundu et al. [24]|DA||one-shot||Class-Incremental DA|
|Saito et al. [37]|DA||||Strong-Weak Distribution Alignment|
|Chen et al. [8]|DA||||Adversarial training|
|Kim et al. [22]|DA||||Adversarial training and Domain Diversifcation|
|Zhu et al. [48]|DA||||selective region adaptation framework|
|Deselaers et al. [14]|WSOD w/transfer||presence/absence||CRF-based, iteratively|
|Zhong et al. [46]|WSOD w/transfer||presence/absence||MIL based, iteratively|
|Uijlings et al. [43]|WSOD w/transfer||presence/absence||MIL based, non iteratively|
|Inoue et al. [20]|WSOD w/transfer||class counts||DA + pseudolabeling, iteratively|
|ProbKT (ours)|WSOD w/transfer||**arbitrary**||Probabilistic logical reasoning, iteratively|



Table 1: Summary table of related works with weakly supervised object detection(WSOD), Domain Adaptation(DA) and weakly supervised knowledge transfer methods (WSOD w/ transfer). 

## 3 METHODOLOGY 

## 3.1 PROBLEM STATEMENT 

We consider the problem of weakly supervised knowledge transfer for object detection. Using a model trained on a richly annotated source domain, we aim at improving its performance on a less richly annotated target domain. 

3 

Published as a conference paper at ICLR 2023 

Let _Ds_ = _{_ ( _Is[i][, b][i] s[, y] s[i]_[) :] _[ i]_[ = 1] _[, ..., N][s]_[)] _[}]_[ be a dataset issued from the source domain and consisting] of _Ns_ images _Is_ along with their annotations. We write _b[i] s[∈]_[R] _[n][i][×]_[4][and] _[ y] s[i][∈{]_[1] _[, .., K][s][}][n][i]_[for the] box coordinates and class labels of objects in image _Is[i]_[.] _[n][i]_[is the number of objects present in image] _Is[i]_[and] _[ K][s]_[is the total number of object classes in the source domain.][This represents the typical] dataset required to train classical fully-supervised object detection architectures. The target dataset _Dt_ = _{_ ( _It[i][, q] t[i]_[) :] _[ i]_[ = 1] _[, ..., N][t]_[)] _[}]_[ contains] _[ N][t]_[image from the target domain along with image-level] annotations _qt[i]_[.][These annotations are logical statements about the content of the image in terms of] object classes and their location. Examples include the presence of different classes in each image ( _i.e._ the classical assumption in weakly supervised object detection) but also extends to the _counts_ of classes or a complex combination of counts of objects attributes ( _e.g._ "two red objects, and at least two bicycles"). What is more, the logical statements _qt[i]_[can include classes not already present in the] source domain. This type of logical annotation is then strictly broader than the restrictive supervision usually assumed. 

Based on the availability of a source dataset and a target dataset as described above, our goal is then to harness the available detailed information from the source domain to perform accurate object detection on the target domain. A graphical illustration of this process is given in Figure 1. 

## 3.2 BACKGROUND 

## 3.2.1 OBJECT DETECTION 

Object detection aims at predicting the location and labels of objects in images. One then wishes to learn a parametric function _fθ_ : _I →{B ×_ R _[K] }_[Z] with _fθ_ ( _I_ ) = _{_ ([ˆ] _b,_ ˆ _py_ ) _}[n]_[ˆ] = _{_ ([ˆ] _bi,_ ˆ _py,i_ ) : _i_ = 1 _, ...,_ ˆ _n}_ such that the distance between predicted and true boxes and labels, _d_ ( _{_ ([ˆ] _b,_ ˆ _py_ ) _}[n]_[ˆ] _, {_ ( _b, y_ ) _}[n]_ ), is minimum. Objects detection architecture would usually output box features proposals _{hi_ : _i_ = ˆ 1 _, ...,_ ˆ _n}_ conditioned on which they would predict the probability vector of class labels _py,i_ = _gp_ ( _hi_ ) and the box location predictions[ˆ] _bi_ = _gb_ ( _hi_ ) using shared parametric functions _gp_ ( _·_ ) and _gb_ ( _·_ ). For an object _n_ , we write the predicted probability of the object belonging to class _k_ as _p_ ˆ _[k] y,n_[.] 

## 3.2.2 PROBABILISTIC LOGICAL REASONING 

Probabilistic logical reasoning uses knowledge representation relying on probabilities that allow encoding uncertainty in knowledge. Such a knowledge is encoded in a probabilistic logical program _P_ as a set of _N_ probabilistic facts _U_ = _{U_ 1 _, ..., UN }_ and _M_ logical rules _F_ = _{f_ 1 _, ...fM }_ connecting them. A simple example of probabilistic fact is "Alice and Bob will each pass their exam with probability 0.5" and an example of logical rule is "if both Alice and Bob pass their exam, they will host a party". Combining probabilistic facts and logical rules, one can then construct complex probabilistic knowledge representation, that can also be depicted as probabilistic graphical models. 

Probabilistic logical programming allows to perform inference by computing the probability of a particular statement or query. For instance, one could query the probability that "Alice and Bob will host a party". This query is executed by summing over the probabilities of occurrence of the different _worlds w_ = _{u_ 1 _, ..., uN }_ ( _i.e._ individual realization of the set of probabilistic facts) that are compatible with the query _q_ . The probability of a query _q_ in a program _P_ can then be inferred as _PP_ ( _q_ ) =[�] _w[P]_[(] _[w]_[)] _[ ·]_[ I][[] _[F]_[(] _[w]_[)] _[≡][q]_[]][, where] _[ F]_[(] _[w]_[)] _[≡][q]_[stands for the fact that propagation of the] realization _w_ across the knowledge graph, according to the logical rules _F_ leads to _q_ being true. 

Remarkably, recent advances in probabilistic programming have lead to _learnable_ probabilistic facts [27]. In particular, the probability of a fact can be generated by a neural network with learnable weights. Such a learnable probabilistic fact is then referred to as a neural predicate _U[θ]_ , where we make the dependence on the weights _θ_ explicit. One can then train these weights to minimize a loss that depend on the probability of a query _q_ : _θ_[ˆ] = arg min _θ L_ ( _P_ ( _q | θ_ )). 

Our approach builds upon this ability to learn neural predicates and uses DeepProbLog [27] as the probabilistic reasoning backbone. DeepProbLog is a neural probabilistic logic programming language that allows to conveniently perform inference and differentiation with neural predicates. We refer the reader to the excellent introduction of Manhaeve et al. [28] for further details about this framework. 

4 

Published as a conference paper at ICLR 2023 

## 3.3 PROBKT: WEAKLY SUPERVISED KNOWLEDGE TRANSFER WITH PROBABILISTIC LOGICAL REASONING 

A graphical description of our approach is presented in Figure 2. Our framework starts from a pre-trained object detection model _fθ_ on the source domain. The backbone of this model is extracted and inserted into a new object detection model _fθ[∗]_[with new target box position predictors and box] label classifiers. This new model is then used to predict box proposals along with the corresponding box features on target domain images _It_ . These box features are then fed to a new target box position predictor and box label classifier. The predictions of this classifier are considered neural predicates and are given to a probabilistic logical module. This module evaluates the probability of queries _qt_ , the loss, and the corresponding gradient that can be backpropagated to the classifier and the backbone. As we want to maximize the probability of the queries being true, we use the following loss function: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0005-03.png)


In theory, the backbone can be trained end to end with this procedure. Our experiments showed that only updating the box features classifiers resulted in more stability as also shown in previous works [46]. We then adopt here the same iterative relabeling strategy, as described next. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0005-05.png)


**----- Start of picture text -----**<br>
Probabilistic Reasoning<br>: Labels probabilities<br>: Box positions<br>: Box features<br>: Probabilistic query<br>Object Detection Backbone<br>model<br>Image<br>**----- End of picture text -----**<br>


Figure 2: **ProbKT** . The pre-trained object detection backbone outputs the box features _h_ for the detected objects. Box classifiers (red) and box position predictors (blue) then predict corresponding label predictions _p_ ˆ _y_ and box position predictions[ˆ] _b_ that are fed to the probabilistic reasoning layer. This layer computes the probability of the query along with the gradients with respect to _p_ ˆ _y_ and[ˆ] _b_ that can be backpropagated through the entire network. 

## 3.3.1 ITERATIVE RELABELING 

The approach described above allows to fine-tune our model _fθ[∗]_[to the target domain.][To further] improve the performance, we propose an iterative relabeling strategy that consists in multiple steps : fine-tuning, re-labeling and re-training. A similar has also been proposed by Zhong et al. [46]. 

**Fine-tuning** . This step corresponds to training ProbKT on the weakly supervised labels, by minimizing the loss of Equation 1. 

**Re-labeling** . Once ProbKT has been trained, we can use its predictions to annotate images in the target domain. In practice, we only relabel images for which the model predictions comply with the available query labels in order to avoid too noisy labels. 

**Re-training** . The re-labeled target domain can be used to re-train the object detection backbone of ProbKTin a fully-supervised fashion. 

5 

Published as a conference paper at ICLR 2023 

This procedure can be repeated multiple times to improve the quality of the relabeling and the quantity of relabelled in the target domain dataset. A graphical representation of the relabeling pipeline is presented in Figure 3. 

Pre-training Fine-tuning Re-labeling Re-training 

Figure 3: Iterative relabeling. A full cycle is composed of a fine-tuning, a re-labeling and a re-training step. After one cycle, the fine-tuning step and/or re-labeling step can be iteratively repeated. 

## 3.3.2 COMPUTATIONAL COMPLEXITY AND APPROXIMATIONS 

The computational complexity of inference in probabilistic programming depends on the specific query _q_ and several approximations have been proposed for improving the computation time [44]. We propose two approaches for reducing the computational cost adapted to object detection: (1) filtering the data samples before applying ProbKT (see Appendix Section C.1) or (2) when the supervision consists of the class labels counts, considering only the most probable world (ProbKT*) instead of all possible worlds. 

## 3.3.3 PROBKT*: THE MOST PROBABLE WORLD AND CONNECTION TO HUNGARIAN MATCHING 

The probabilistic inference step requires a smart aggregation of all worlds compatible with the query _q_ . Yet, in certain cases, one can reduce the computational cost by only considering the most probable world. Indeed, consider the case when the query consists of the list of different class labels in the images. For a number of boxes ˆ _n_ proposed by the objects detection model, the query can be written ˆ as the set of labels _q_ = _{y[i]_ : _i_ = 1 _, ...,_ ˆ _n}_ . If we further write _p[k] y,n_[as the probability of the label of] box _n_ belonging to class _k_ given by the model (as introduced in Section 3.2.1), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0006-08.png)


where _σj_ corresponds to the _j[th]_ permutation of the query vector _q_ . To avoid the computation of each possible world contribution, one can only use the configuration with the largest contribution to _PP_ ( _q_ ) and discard the other ones. 

This possible world corresponds to the permutation _σ[∗]_ that satisfies: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0006-11.png)


Remarkably, this corresponds to the solution of the best alignment using the Hungarian matching ˆ algorithm with cost _c_ ( _n_ ) = (1 _− p[σ] y,n[j]_[(] _[n]_[)] ), as used, among others, in DETR [7]. Thus, when the query is is the set of class labels, the most plausible world can thus be inferred with the Hungarian matching algorithm. In Appendix C.2, we also show that the gradient of ProbKT can be interpreted as a probability weighted extension of the gradient resulting from the Hungarian matching. 

## 4 EXPERIMENTS 

## 4.1 DATASETS 

We evaluate our approach on three different datasets: (1) a CLEVR-mini dataset, (2) a Molecules dataset with images of chemical compounds, and (3) an MNIST-based object detection dataset. For each dataset, three subsets, corresponding to different domains, are used: (1) a source domain, (2) a target domain, and (3) an out-of-distribution domain (OOD). The source domain is the richly annotated domain that was used to pre-train the object detection model. The target domain is 

6 

Published as a conference paper at ICLR 2023 

the domain of interest but with image-level annotations only. Lastly, the OOD domain contains images from a different distribution than the source and target domains and is used to study the generalizability of the models. Source and target domains are split into 5 folds of train and validation sets and an independent test set. We focused our experiments on the small sample regime (range 1k-2k numbers of samples) both for the source as the target domain. More details on each dataset can be found in Appendix B. 

## 4.2 MODELS 

In the experiments, we apply our method ProbKT on two different pre-trained object detection backbone models: (1) DETR [7] and (2) FasterRCNN [34]. Both are pre-trained on the COCO dataset [26]. We also evaluate an Hungarian-algorithm approximation (ProbKT*) of our method when the weak supervision allows it. For sake of conciseness, we omit the results of ProbKT* here but they can be found in Appendix D. The details of the training procedures, as well as the hyper-parameters used for the different models and the different datasets are summarized in Table 4 in Appendix A. 

## 4.2.1 BASELINE MODELS 

As shown in Section 2, all available approaches for weakly supervised object detection are very restrictive in terms of the supervision signal they support. Our main comparison partner is the state of the art _WSOD-transfer_ method [46]. 

Additionally, we compare our approach against a Resnet50 [18] backbone pre-trained on ImageNet [12]. Fine-tuning is performed by adding an extra multitask regression layer that is trained to predict the individual counts of the objects in the image as in Xue et al. [45]. This architecture naturally relies only on label counts in the target images for fine-tuning. We then predict box predictions using class activation maps as in Bae et al. [3] to compare its performance on object localization. We call this approach _Resnet50-CAM_ . 

When the supervision signal allows it, we also compare with a DETR model trained end-to-end jointly on target and source domains, masking the box costs in the matching cost of the Hungarian algorithm for image-level annotated samples. We call this approach _DETR-joint_ . 

|Model|Data Domain|CLEVR count acc.|CLEVR mAP (mAP@IoU=0.5)|Mol. count. acc|Mol. mAP (mAP@IoU=0.5)|
|---|---|---|---|---|---|
|Resnet50-CAM|target domain|0_._97_±_0_._005|0_._036_±_0_._014(0_._200_±_0_._071)|**0****_._978****_±_ 0****_._004**|0_._0_±_0_._0(0_±_0)|
|Resnet50-CAM|OOD|0_._831_±_0_._016|0_._029_±_0_._010(0_._153_±_0_._044)|0_._0_±_0_._0|n/a*|
|Resnet50-CAM|source domain|0_._993_±_0_._003|0_._035_±_0_._019(0_._178_±_0_._084)|0_._828_±_0_._021|0_._0_±_0_._0(0_±_0)|
|WSOD-transfer|target domain|0_._944_±_0_._004|0_._844_±_0_._005(0_._988_±_0_._001)|0_._001_±_0_._0|0_._018_±_0_._004(0_._061_±_0_._011)|
|WSOD-transfer|OOD|0_._73_±_0_._011|0_._79_±_0_._005(0_._969_±_0_._001)|0_._003_±_0_._002|n/a*|
|WSOD-transfer|source domain|0_._989_±_0_._001|0_._926_±_0_._001(0_._995_±_0_._0)|0_._0_±_0_._0|0_._021_±_0_._003(0_._069_±_0_._009)|
|DETR-joint|target domain|0_._159_±_0_._133|0_._579_±_0_._012(0_._684_±_0_._019)|0_._357_±_0_._196|0_._197_±_0_._055(0_._481_±_0_._071)|
|DETR-joint|OOD|0_._084_±_0_._039|0_._534_±_0_._012(0_._66_±_0_._012)|0_._024_±_0_._021|n/a*|
|DETR-joint|source dom.|0_._923_±_0_._049|0_._908_±_0_._017(0_._992_±_0_._001)|0_._232_±_0_._127|0_._23_±_0_._063(0_._565_±_0_._08)|
|RCNN (pre-trained)|target domain|0_._0_±_0_._0|0_._586_±_0_._014(0_._598_±_0_._013)|0_._592_±_0_._007|**0****_._568****_±_ 0****_._005**(0_._785_±_0_._004)|
|RCNN (pre-trained)|OOD|0_._0_±_0_._0|0_._582_±_0_._012(0_._603_±_0_._011)|0_._348_±_0_._036|n/a*|
|RCNN (pre-trained)|source domain|0_._988_±_0_._002|**0****_._984****_±_ 0****_._01**(0_._996_±_0_._0)|0_._948_±_0_._004|**0****_._737****_±_ 0****_._005**(**0****_._979****_±_ 0****_._0**)|
|DETR (pre-trained)|target domain|0_._0_±_0_._0|0_._498_±_0_._019(0_._533_±_0_._024)|0_._464_±_0_._033|0_._314_±_0_._006(0_._542_±_0_._006)|
|DETR (pre-trained)|OOD|0_._0_±_0_._0|0_._477_±_0_._013(0_._531_±_0_._021)|0_._002_±_0_._001|n/a*|
|DETR (pre-trained)|source domain|0_._97_±_0_._009|0_._945_±_0_._009(0_._992_±_0_._001)|0_._581_±_0_._022|0_._409_±_0_._005(0_._722_±_0_._004)|
|ProbKT (DETR)|target domain|0_._946_±_0_._014|0_._803_±_0_._011(0_._989_±_0_._006)|0_._508_±_0_._027|0_._204_±_0_._02(0_._507_±_0_._014)|
|ProbKT (DETR)|OOD|0_._726_±_0_._035|0_._715_±_0_._006(0_._974_±_0_._006)|0_._004_±_0_._003|n/a*|
|ProbKT (DETR)|source domain|0_._987_±_0_._003|0_._948_±_0_._005(0_._995_±_0_._001)|0_._549_±_0_._026|0_._38_±_0_._013(0_._713_±_0_._006)|
|ProbKT (RCNN)|target domain|**0****_._975****_±_ 0****_._003**|**0****_._856****_±_ 0****_._039**(0_._993_±_0_._001)|0_._942_±_0_._009|0_._289_±_0_._041(**0****_._829****_±_ 0****_._054**)|
|ProbKT (RCNN)|OOD|0_._89_±_0_._022|**0****_._833****_±_ 0****_._042**(**0****_._991****_±_ 0****_._001**)|**0****_._603****_±_ 0****_._037**|n/a*|
|ProbKT (RCNN)|source domain|**0****_._995****_±_ 0****_._002**|0_._941_±_0_._041(0_._998_±_0_._001)|**0****_._96****_±_ 0****_._002**|0_._666_±_0_._005(0_._978_±_0_._002)|



Table 2: Results of the experiments for the datasets: CLEVR-mini and Molecules. Reported test accuracies over the 5 folds. Best method is in bold for each metric and data distribution. *: OOD test set of Molecules dataset has no bounding box labels. 

## 4.3 EVALUATION METRICS 

We evaluate the performance of the models on the different datasets based on two criteria : the count accuracy and the objects localization performance. The count accuracy measures the ratio of correct images where all individual counts of (all detected) objects are correct. To evaluate how well the 

7 

Published as a conference paper at ICLR 2023 

model is performing in localizing the different objects in the image we report the mean average precision (mAP) performance, a widely used metric for evaluating object detection models. 

## 4.4 WEAKLY SUPERVISED KNOWLEDGE TRANSFER WITH CLASS COUNTS 

We first investigate the performance of ProbKT when the weakly supervision consists of class counts only. The query _q_ for each image then consists of the number of objects from each class in the image. We evaluate the models on the CLEVR-mini and Molecules datasets. For the Molecules dataset, the query for an image containing 6 carbon atoms (C), 6 oxygen atoms (O) and 12 hydrogen atoms (H) would result in the following query: _q_ = ([ _C, O, H_ ] _,_ [6 _,_ 6 _,_ 12]). These weak labels in the case of the Molecules dataset are widely and easily available in the form of the chemical formula of the molecule on the image ( _e.g C_ 6 _H_ 12 _O_ 6). The recognition of atomic level entities on images of molecules is a challenge in the field of Optical Chemical Structure Recognition (OCSR) [9; 33; 29; 19]. For the CLEVR-mini dataset, the query for an example image containing 2 spheres, 1 cylinder and 3 cubes would be _q_ = ([Cube _,_ Cylinder _,_ Sphere] _,_ [3 _,_ 1 _,_ 2]). Formal descriptions of the queries for each task are presented in Appendix E. 

Results of the experiments are summarized in Table 2. We observe on both datasets that ProbKT is able to transfer knowledge from the source domain to the target domain and improve count accuracy on the target domain and in most cases also on the source domain. The count accuracy increases on both the target domain and on OOD, suggesting better generalization performance. This is in contrast with Resnet50-CAM which performs well on the target domain of the Molecules dataset but fails on OOD. We also note a significant improvement in object localization (mAP) for ProbKT on the CLEVR-mini dataset. However, fine-tuning seems detrimental for mAP on the Molecules dataset. This can be explained by the very small bounding boxes in the Molecules dataset. We therefore also report the mAP@IoU=0.5 where we observe some increase in performance after fine-tuning. Lastly, we observe that our approach outperforms WSOD-transfer on all metrics for both datasets. WSOD-transfer performs well on CLEVR-mini but fails for the Molecules dataset. This can be explained by the fact that this method only supports class indicators (whether a class is present in the image), which is particularly detrimental in molecules images containing a lot of objects. 

## 4.5 OTHER TYPES OF WEAK SUPERVISION 

## 4.5.1 CLASS RANGES 

The annotation of images is a tedious task, which limits the availability of fully annotated datasets. When the number of objects on an image is large, counting the exact number of objects of a particular class becomes too time-consuming. A typical annotation in this case consists oof class _ranges_ where, instead of exact class counts, an interval is given for the count. For example an image from the CLEVR-mini dataset with more than 4 cubes, exactly 4 cylinders and less than 4 spheres would result in the following query: _q_ = ([cube _,_ cylinder _,_ sphere] _,_ �[4 _, ∞_ [ _,_ [4 _,_ 5[ _,_ [0 _,_ 4[�). We evaluate this experimental setup and report results in Table 3. We observe that ProbKT performs significantly better than WSOD-transfer on count accuracy, which still uses only presence/absence labels. We note that Resnet50-CAM is unable to use this type of supervision and is thus reported as _n/a_ . 

|Model|Data Domain|MNIST count acc.|MNIST sum acc.|MNIST mAP (mAP@IoU=0.5)|CLEVR* count acc.|CLEVR* mAP (mAP@IoU=0.5)|
|---|---|---|---|---|---|---|
|Resnet50-CAM|target domain|0_._044_±_0_._041|0_._506_±_0_._063|0_._003_±_0_._003(0_._014_±_0_._011)|n/a|n/a|
|Resnet50-CAM|OOD|0_._01_±_0_._009|0_._015_±_0_._004|0_._003_±_0_._002(0_._011_±_0_._007)|n/a|n/a|
|Resnet50-CAM|source domain|0_._127_±_0_._132|0_._649_±_0_._108|0_._005_±_0_._004(0_._028_±_0_._018)|n/a|n/a|
|WSOD-transfer|target domain|n/a|n/a|n/a|0_._944_±_0_._004|**0****_._844****_±_ 0****_._005**(0_._988_±_0_._001)|
|WSOD-transfer|OOD|n/a|n/a|n/a|0_._73_±_0_._011|0_._79_±_0_._005(0_._969_±_0_._001)|
|WSOD-transfer|source domain|n/a|n/a|n/a|0_._989_±_0_._001|0_._926_±_0_._001(0_._995_±_0_._0)|
|RCNN (pre-trained)|target domain|0_._292_±_0_._005|0_._298_±_0_._005|0_._632_±_0_._014(0_._685_±_0_._002)|0_._0_±_0_._0|0_._586_±_0_._014(0_._598_±_0_._013)|
|RCNN (pre-trained)|OOD|0_._205_±_0_._004|0_._212_±_0_._004|0_._631_±_0_._013(0_._683_±_0_._002)|0_._0_±_0_._0|0_._582_±_0_._012(0_._603_±_0_._011)|
|RCNN (pre-trained)|source domain|0_._961_±_0_._008|0_._961_±_0_._008|**0**_._**917**_±_**0**_._**021**(0_._988_±_0_._002)|0_._988_±_0_._002|**0****_._984****_±_ 0****_._01**(0_._996_±_0_._0)|
|ProbKT (RCNN)|target domain|**0**_._**902**_±_**0**_._**005**|**0**_._**903**_±_**0**_._**005**|**0**_._**786**_±_**0**_._**021**(0_._974_±_0_._001)|**0****_._971****_±_ 0****_._006**|0_._838_±_0_._034(**0****_._993****_±_ 0****_._001**)|
|ProbKT (RCNN)|OOD|**0**_._**863**_±_**0**_._**008**|**0**_._**865**_±_**0**_._**008**|**0**_._**778**_±_**0**_._**021**(0_._97_±_0_._001)|**0****_._884****_±_ 0****_._01**|**0****_._812****_±_ 0****_._036**(**0****_._991****_±_ 0****_._001**)|
|ProbKT (RCNN)|source domain|**0**_._**967**_±_**0**_._**004**|**0**_._**967**_±_**0**_._**004**|0_._873_±_0_._016(0_._989_±_0_._001)|**0****_._994****_±_ 0****_._001**|0_._922_±_0_._035(**0****_._998****_±_ 0****_._001**)|



Table 3: Results of the experiments on the MNIST object detection dataset and on CLEVR* dataset (*CLEVR uses ranges of class counts as labels instead of exact class counts). Reported test accuracies over the 5 folds. Best method is in bold for each metric and data distribution. 

8 

Published as a conference paper at ICLR 2023 

## 4.5.2 COMPLEX QUERIES 

More complex types of weak supervision than the ones considered above are also possible. To illustrate the capabilities of our approach, we build an MNIST object detection dataset where images show multiple digits as objects. Examples images are available in Appendix B. The weak supervision is here the sum of all digits in the image : _q_ = SUM(digits). Our ProbKT can seamlessly integrate this type of supervision as shown in Table 3. As all other baselines are unable process this type of supervision, we compare against a pre-trained RCNN and a variation of Resnet50-CAM where we add an extra neural network layer that sums the individual counts to give the resulting sum. We report count accuracy, mAP and sum accuracy. The sum accuracy measures the ratio of correct images where the predicted sum (instead of the label of the digits) is correct. Details about the results on extra experiments with DETR as backbone using complex types of weak supervision can be found in Appendix D. 

## 4.6 ABLATION STUDIES 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0009-04.png)


**----- Start of picture text -----**<br>
CLEVR Dataset Performance 1.0 Mol. Dataset Performance 1.0 MNIST Dataset Performance<br>1.0<br>0.8 0.8<br>0.8<br>0.6 0.6 0.6<br>0.4 RCNN (ProbKT) 0.4 RCNN (ProbKT) 0.4<br>0.2 RCNN (ProbKT*) 0.2 RCNN (ProbKT*) 0.2<br>DETR (ProbKT) DETR (ProbKT) RCNN (ProbKT)<br>0.0 DETR (ProbKT*) 0.0 DETR (ProbKT*) 0.0 DETR (ProbKT)<br>0 1 2 3 4 0 1 2 3 4 0 1 2 3 4<br>Iteration Iteration Iteration<br>(a) CLEVR iterative relabeling (b) Molecules iterative relabeling (c) MNIST iterative relabeling<br>Count Val. Accuracy Count Val. Accuracy Sum Val. Accuracy<br>**----- End of picture text -----**<br>


Figure 4: Iterative relabeling performance for the different datasets. Iteration 0: pretrained on source domain. Iteration 1: fine-tuned. Iteration 2: re-labeled and re-trained. Iteration 3: relabeled and re-trained. Iteration 4: relabeled and re-trained. 

**Iterative relabeling** . In Figure 4, we plot the evolution of the performance on the test sets after multiple rounds of fine-tuning and re-labeling, as detailed in Section 3.3.1. The final performance reported in the results tables is selected based on best relabeling iteration on the validation dataset. We observe that iterative relabeling after fine-tuning can improve performance significantly. Nevertheless, the benefit of iterative relabeling is less pronounced for DETR on the Molecules dataset. We impute 

## **Object detection backbone** 

Our method can seamlessly accommodate different object detection backbones. In Table 2, we present the results for our method with a DETR[7] and a FasterRCNN[34] backbone. We observe that FasterRCNN is typically performing better. In particular, the DETR backbone performs poorly on the Molecules dataset. This could be due to the small objects in the Molecules dataset. Indeed, Carion et al. [7] recommend to use DETR-DC5 or DETR-DC5-R101 for small objects instead. 

## 5 CONCLUSIONS AND DISCUSSION 

Objects detection models are a key component of machine learning deployment in the real world. However, training such models usually requires large amounts of richly annotated images that are often prohibitive for many applications. In this work, we proposed a novel approach to train object detection models by leveraging richly annotated datasets from other domains and allowing arbitrary types of weak supervision on the target domain. Our architecture relies on a probabilistic logical programming engine that efficiently blends the power of symbolic reasoning and deep learning architecture. As such, our model also inherits the current limitations from the probabilistic reasoning implementations, such as higher computational complexity. We proposed several approaches to speed-up the inference process significantly and our work will directly benefit from further advances in this field. Lastly, the versatility of probabilistic programming could help support other related tasks in the future, such as image to graph translation. 

9 

Published as a conference paper at ICLR 2023 

**Reproducibility Statement** Details for reproducing all experiments shown in this work are available in Appendix E. More details on the datasets used in the experiments can be found in Appendix B. 

## ACKNOWLEDGMENTS 

AA, MO and YM are funded by (1) Research Council KU Leuven: Symbiosis 4 (C14/22/125), Symbiosis3 (C14/18/092); (2) Federated cloud-based Artificial Intelligence-driven platform for liquid biopsy analyses (C3/20/100); (3) CELSA - Active Learning (CELSA/21/019); (4) European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956832; (5) Flemish Government (FWO: SBO (S003422N), Elixir Belgium (I002819N), SB and Postdoctoral grants: S003422N, 1SB2721N, 1S98819N, 12Y5623N) and (6) VLAIO PM: Augmenting Therapeutic Effectiveness through Novel Analytics (HBC.2019.2528); (7) YM, AA, EDB, and MO are affiliated to Leuven.AI and received funding from the Flemish Government (AI Research Program). EDB is funded by a FWO-SB grant (S98819N). Computational resources and services used in this work were partly provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation - Flanders (FWO) and the Flemish Government – department EWI. 

## REFERENCES 

- [1] Mnist object detection dataset. URL https://github.com/hukkelas/ MNIST-ObjectDetection. accessed on 01.02.2022. 

- [2] Rdkit: Open-source cheminformatics. URL https://www.rdkit.org. accessed on 01.02.2022. 

- [3] Wonho Bae, Junhyug Noh, and Gunhee Kim. Rethinking class activation mapping for weakly supervised object localization. In _European Conference on Computer Vision_ , pages 618–634. Springer, 2020. 

- [4] Aseem Behl, Omid Hosseini Jafari, Siva Karthik Mustikovela, Hassan Abu Alhaija, Carsten Rother, and Andreas Geiger. Bounding boxes, segmentations and object coordinates: How important is recognition for 3d scene flow estimation in autonomous driving scenarios? In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 2574–2583, 2017. 

- [5] Lukas Biewald. Experiment tracking with weights and biases, 2020. URL https://www. wandb.com/. Software available from wandb.com. 

- [6] Hakan Bilen and Andrea Vedaldi. Weakly supervised deep detection networks. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2846–2854, 2016. 

- [7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _European conference on computer vision_ , pages 213–229. Springer, 2020. 

- [8] Yuhua Chen, Wen Li, Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Domain adaptive faster r-cnn for object detection in the wild. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 3339–3348, 2018. 

- [9] Djork-Arné Clevert, Tuan Le, Robin Winter, and Floriane Montanari. Img2mol–accurate smiles recognition from molecular graphical depictions. _Chemical science_ , 12(42):14174–14181, 2021. 

- [10] Luc De Raedt and Kristian Kersting. Probabilistic logic learning. _ACM SIGKDD Explorations Newsletter_ , 5(1):31–48, 2003. 

- [11] Luc De Raedt and Angelika Kimmig. Probabilistic (logic) programming concepts. _Machine Learning_ , 100(1):5–47, 2015. 

- [12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pages 248–255. Ieee, 2009. 

10 

Published as a conference paper at ICLR 2023 

- [13] Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. _IEEE signal processing magazine_ , 29(6):141–142, 2012. 

- [14] Thomas Deselaers, Bogdan Alexe, and Vittorio Ferrari. Weakly supervised localization and learning with generic knowledge. _International journal of computer vision_ , 100(3):275–293, 2012. 

- [15] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. _International Journal of Computer Vision_ , 88(2):303–338, June 2010. 

- [16] Eleonora Giunchiglia, Mihaela C˘at˘alina Stoian, Salman Khan, Fabio Cuzzolin, and Thomas Lukasiewicz. Road-r: The autonomous driving dataset with logical requirements. _arXiv preprint arXiv:2210.01597_ , 2022. 

- [17] Ibtihaal M Hameed, Sadiq H Abdulhussain, and Basheera M Mahmmod. Content-based image retrieval: A review of recent trends. _Cogent Engineering_ , 8(1):1927469, 2021. 

- [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 770–778, 2016. 

- [19] Rodrigo Hormazabal, Changyoung Park, Soonyoung Lee, Sehui Han, Yeonsik Jo, Jaewan Lee, Ahra Jo, Seung Hwan Kim, Jaegul Choo, Moontae Lee, et al. Cede: A collection of expertcurated datasets with atom-level entity annotations for optical chemical structure recognition. 

- [20] Naoto Inoue, Ryosuke Furuta, Toshihiko Yamasaki, and Kiyoharu Aizawa. Cross-domain weakly-supervised object detection through progressive domain adaptation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5001–5009, 2018. 

- [21] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2901–2910, 2017. 

- [22] Taekyung Kim, Minki Jeong, Seunghyeon Kim, Seokeon Choi, and Changick Kim. Diversify and match: A domain adaptive representation learning paradigm for object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 12456–12465, 2019. 

- [23] Daphne Koller, Nir Friedman, Sašo Džeroski, Charles Sutton, Andrew McCallum, Avi Pfeffer, Pieter Abbeel, Ming-Fai Wong, Chris Meek, Jennifer Neville, et al. _Introduction to statistical relational learning_ . MIT press, 2007. 

- [24] Jogendra Nath Kundu, Rahul Mysore Venkatesh, Naveen Venkat, Ambareesh Revanur, and R Venkatesh Babu. Class-incremental domain adaptation. In _European Conference on Computer Vision_ , pages 53–69. Springer, 2020. 

- [25] Dong Li, Jia-Bin Huang, Yali Li, Shengjin Wang, and Ming-Hsuan Yang. Weakly supervised object localization with progressive domain adaptation. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 3512–3520, 2016. 

- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European conference on computer vision_ , pages 740–755. Springer, 2014. 

- [27] Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Deepproblog: Neural probabilistic logic programming. _Advances in Neural Information Processing Systems_ , 31, 2018. 

- [28] Robin Manhaeve, Sebastijan Dumanˇci´c, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Neural probabilistic logic programming in deepproblog. _Artificial Intelligence_ , 298: 103504, 2021. 

11 

Published as a conference paper at ICLR 2023 

- [29] Martijn Oldenhof, Adam Arany, Yves Moreau, and Jaak Simm. Chemgrapher: optical graph recognition of chemical compounds by deep learning. _Journal of chemical information and modeling_ , 60(10):4506–4517, 2020. 

- [30] Martijn Oldenhof, Adam Arany, Yves Moreau, and Jaak Simm. Self-labeling of fully mediating representations by graph alignment. In _Benelux Conference on Artificial Intelligence_ , pages 46–65. Springer, 2021. 

- [31] Martijn Oldenhof, Ádám Arany, Yves Moreau, and Edward De Brouwer. Updating object detection models with probabilistic programming. 2022. ICML workshop - UpML. 

- [32] Luc De Raedt, Kristian Kersting, Sriraam Natarajan, and David Poole. Statistical relational artificial intelligence: Logic, probability, and computation. _Synthesis lectures on artificial intelligence and machine learning_ , 10(2):1–189, 2016. 

- [33] Kohulan Rajan, Achim Zielesny, and Christoph Steinbeck. Decimer: towards deep learning for chemical image recognition. _Journal of Cheminformatics_ , 12(1):1–9, 2020. 

- [34] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. _Advances in neural information processing systems_ , 28, 2015. 

- [35] Tim Rocktäschel and Sebastian Riedel. End-to-end differentiable proving. _Advances in neural information processing systems_ , 30, 2017. 

- [36] Noureddin M Sadawi, Alan P Sexton, and Volker Sorge. Chemical structure recognition: a rule-based approach. In _Document Recognition and Retrieval XIX_ , volume 8297, page 82970E. International Society for Optics and Photonics, 2012. 

- [37] Kuniaki Saito, Yoshitaka Ushiku, Tatsuya Harada, and Kate Saenko. Strong-weak distribution alignment for adaptive object detection. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6956–6965, 2019. 

- [38] Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning. _Advances in neural information processing systems_ , 30, 2017. 

- [39] Feifei Shao, Long Chen, Jian Shao, Wei Ji, Shaoning Xiao, Lu Ye, Yueting Zhuang, and Jun Xiao. Deep learning for weakly-supervised object detection and localization: A survey. _Neurocomputing_ , 2022. 

- [40] Hyun Oh Song, Ross Girshick, Stefanie Jegelka, Julien Mairal, Zaid Harchaoui, and Trevor Darrell. On learning to localize objects with minimal supervision. In _International Conference on Machine Learning_ , pages 1611–1619. PMLR, 2014. 

- [41] Leon Sterling and Ehud Y Shapiro. _The art of Prolog: advanced programming techniques_ . MIT press, 1994. 

- [42] Matteo Tomei, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Art2real: Unfolding the reality of artworks via semantically-aware image-to-image translation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5849–5859, 2019. 

- [43] Jasper Uijlings, Stefan Popov, and Vittorio Ferrari. Revisiting knowledge transfer for training object class detectors. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1101–1110, 2018. 

- [44] Thomas Winters, Giuseppe Marra, Robin Manhaeve, and Luc De Raedt. Deepstochlog: Neural stochastic logic programming. _arXiv preprint arXiv:2106.12574_ , 2021. 

- [45] Yao Xue, Nilanjan Ray, Judith Hugh, and Gilbert Bigras. Cell counting by regression using convolutional neural network. In _European Conference on Computer Vision_ , pages 274–290. Springer, 2016. 

12 

Published as a conference paper at ICLR 2023 

- [46] Yuanyi Zhong, Jianfeng Wang, Jian Peng, and Lei Zhang. Boosting weakly supervised object detection with progressive knowledge transfer. In _European conference on computer vision_ , pages 615–631. Springer, 2020. 

- [47] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2921–2929, 2016. 

- [48] Xinge Zhu, Jiangmiao Pang, Ceyuan Yang, Jianping Shi, and Dahua Lin. Adapting object detectors via selective cross-domain alignment. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 687–696, 2019. 

13 

Published as a conference paper at ICLR 2023 

## A TRAINING DETAILS 

For the hyper-parameters the idea was to stay as close as possible to the defaults of the pre-trained standard models although some lightweight tuning was done. In Table 4 a summary is given for the hyper-parameters used for the different models. 

|Model|dataset|epochs|lr|lr_step_size|lr-gamma|momentum|batch size|weight decay|optimizer|
|---|---|---|---|---|---|---|---|---|---|
|DETR pre-train (retrain)|CLEVR|max 100|0.0001|7 (7-8)|0.1||8|0.0001|AdamW|
|DETR pre-train (retrain)|Mols.|max 100|0.0001|20 (20)|0.1||8|0.0001|AdamW|
|DETR pre-train (retrain)|MNIST|max 100|0.0001|15-20 (20)|0.1||8|0.0001|AdamW|
|RCNN pre-train (retrain)|all datasets|max 30|0.005|5 (5)|0.1|0.9|1|0.0005|SGD|
|RCNN Finetune|all datasets|max 20|0.001||||16||Adam|
|DETR Finetune|CLEVR/Mols|max 20|0.001||||16||Adam|
|DETR Finetune|MNIST|max 20|0.01||||16||Adam|
|DETR Finetune*|CLEVR/Mols|max 100|0.002|20|0.1||8|0.0001|AdamW|
|RCNN Finetune*|CLEVR|max 20|0.001||||15||Adam|
|RCNN Finetune*|Mols|max 20|0.00001||||15||Adam|
|DETR masked box loss|CLEVR/Mols|max 100|0.0001|7|0.1||8|0.0001|AdamW|
|Resnet50-CAM models|all datasets|max 500|0.001||||32||Adam|



Table 4: Overview of hyperparameters for the different models, most hyperparamaters are left default from standard models. Tuning was mostly done on learning rate and learning rate scheduling. For every fold/dataset the best epoch/lr/lr_step_size model is used based on validation data. 

## B DATASETS 

We evaluate our approach on three different datasets: (1) a CLEVR-mini dataset, (2) a Molecules data set with images of chemical compounds, and (3) an MNIST-based object detection dataset. For each dataset, three subsets, corresponding to different domains, are used: (1) a source domain, (2) a target domain, and (3) an out-of-distribution domain (OOD). Source and target domains are split into 5 folds of train and validation sets and an independent test set. Sizes of the different splits per dataset are summarized in Table 5. 

14 

Published as a conference paper at ICLR 2023 

|Dataset|Type|Split|Size (number of samples)|
|---|---|---|---|
|MNIST object detection|Source|train|700|
|MNIST object detection|Source|validation|300|
|MNIST object detection|Source|test|1000|
|MNIST object detection|Target|train|700|
|MNIST object detection|Target|validation|300|
|MNIST object detection|Target|test|1000|
|MNIST object detection|OOD|test|1000|
|Molecules|Source|train|1400|
|Molecules|Source|validation|600|
|Molecules|Source|test|1000|
|Molecules|Target|train|1400|
|Molecules|Target|validation|600|
|Molecules|Target|test|1000|
|Molecules|OOD|test|1000|



Table 5: Dataset sizes for the different splits. For train and validations splits 5 folds are used. 

## B.0.1 CLEVR-MINI DATASET 

The CLEVR-mini dataset for our experiments is a selection of samples from the CLEVR dataset [21]. The different types available in the CLEVR dataset are combinations of shapes (cube, sphere, and cylinder), materials (metal and rubber), and sizes (large and small). Colors are ignored as the images are first converted to grayscale before feeding them to the models. For the richly annotated source domain, we randomly select images with only sphere or cylinder-shaped objects (no cubes) and with a maximum of four objects per image and a minimum of three objects. For the weakly annotated target domain we experiment with two type of annotations. Firstly we experiment when we have the class counts of objects in the image available. Secondly, instead of the exact counts of classes in the image the annotations only specify if there is exactly one object class in the image or multiple. The advantage of this kind of labeling is that the annotator does not need to count the objects and instead just make a distinction of only one object class in image or multiple. The images in the target domain can contain all combinations of object types (including cube-shaped objects) and allow a minimum of five objects per image and a maximum of six objects per image. For the OOD dataset we also select images with all possible combinations of object types, always with 10 objects per image. Some example images from the CLEVR-mini dataset can be found in Figure 1. 

## B.0.2 MOLECULES DATASET 

The Molecules dataset contains images depicting chemical compounds. For the richly annotated source domain, a procedure similar as described in Oldenhof et al. [29, 30] was executed using an RDKit [2] fork for generating the bounding box labels for the individual atoms present in the images. In the source domain, we allow the following atom types: carbon (C), hydrogen (H), oxygen (O), and nitrogen (N). In the weakly annotated target domain, we only have the counts of the atoms present which translates to the chemical formula of the molecule in the image ( _e.g C_ 6 _H_ 12 _O_ 6). The same classes from the source domain (C, H, O, and N) are also present in the target domain as well as an extra atom type: sulfur (S). The OOD test dataset consists of 1000 images from the external UoB dataset [36] where chemical compounds containing only the atom types present in the target domain (C, H, O, N, and S). Some example images from the Molecules dataset are visualized in Figure 5. 

## B.0.3 MNIST OBJECT DETECTION DATASET 

The MNIST object detection dataset is generated [1] using the original MNIST dataset [13]. Each image consists of three MNIST digits randomly positioned in the image. The MNIST object detection dataset allows experimenting with a more arbitrary type of weak supervision. Each object in this dataset represents a digit that can be aggregated. This allows to label an image with only the _sum_ of all digits in the image instead of the class counts of the objects. For the richly annotated source domain digits 7, 8, and 9 are left out. The weakly annotated target domain has all possible digit classes (0-9). The labels of the target domain only contain the sum of all digits. For the OOD test dataset, images are used that contain maximum of four MNIST digits, instead of three digits as in the other domains. Some example images from the MNIST object detection dataset are visualized in Figure 6. 

15 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0016-01.png)


Figure 5: Weakly supervised knowledge transfer with probabilistic logical reasoning (ProbKT). On the left we have source domain where a model can be trained using bounding box information(labels,positions) but only on a limited set of atom types (C,H, O, N). In the middle we can see that the pre-trained model is not able to recognize the sulfur (S) from target domain correctly. On the right we see that the model is able to adapt to target domain after probabilistic reasoning using weak labels (e.g. counts of objects on image) and is able to recognize the sulfur (S). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0016-03.png)


Figure 6: Weakly supervised knowledge transfer with probabilistic logical reasoning (ProbKT). On the left we have source domain where a model can be trained using bounding box information(labels,positions) but only on a limited set of digits (0, 1, 2, 3, 4, 5, 6). In the middle we can see that the pre-trained model is not able to recognize the digit eight (8) from target domain correctly. On the right we see that the model is able to adapt to target domain after probabilistic reasoning using weak labels (e.g. sum of digits on image) and is able to recognize the digit eight (8). 

## C PROBKT AND PROBKT* SUPPLEMENTARY DETAILS 

## C.1 FILTERING SAMPLES 

The computation complexity of inference in the probabilistic programming module grows with the number of possible worlds. In turn, the number of possible worlds grows with the number of probabilistic facts ˆ _n_ . 

One avenue to reduce the computational cost of the inference step is then to artificially reduce the ˆ numberinference query.of probabilisticWe compute the filtered set of probabilistic factsfacts in each image. Let _{py,n_ : _n_ = 1 _p_ ¯ _, ...,y,n_ ˆ _n_ by setting _}_ and _q_ the corresponding 

16 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0017-01.png)


Theprobability ofHowever, we also have to update the inference queryparameter 1 _δ_ or _∈_ 0[0 effectively discards the probabilistic fact _,_ 1] is a threshold at which we _q_ to reflect this filtration.consider the _p_ ¯ _y,n_ probabilisticfrom the inference procedure.We writefact as _q_ ¯ the filteredcertain. A query _q_ . 

**Example** To illustrate this filtration strategy let’s consider an MNIST image with 3 digits in the image : _{_ 3 _,_ 4 _,_ 7 _}_ . The query _q_ corresponds to the class labels in the images. That is _q_ = _{_ 3 _,_ 4 _,_ 7 _}_ . The object detection backbones outputs 3 box features with corresponding probabilities _{p_ ˆ _y,_ 0 _,_ ˆ _py,_ 1 _,_ ˆ _py,_ 2 _, }_ . Nowcompute the filtered querylet _e.g. p_ ˆ[3] _y,_ 1[=][0] _[.]_[99][.] _q_ ¯[We] = _{_[can] 4 _,_ 7 _}_[filter] .[out] _[p]_[ˆ] _[y,]_[1][(] _[i.e]_[the][prediction][for][a][digit][3][is][certain),][and] 

**Remark** Equation 2 suggests a filtering based on the output probabilities only. However, one can also use information about the query for the filtration. For instance, one would only filter out a probabilistic fact if it is consistent with the query _q_ . In the example above, it would be wiser not to filter out _e.g. p_ ˆ[9] _y,_ 1[=][0] _[.]_[99][ as no images are supposedly present in the image.][One should then] ideally propagate this probabilistic fact to the inference module such as to update the weights of the backbone and learn from this error. 

## C.2 GRADIENT OF THE LIKELIHOOD 

The ProbKT likelihood has the following form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0017-07.png)


where _α_ is a "possible world" matrix of indicator variables: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0017-09.png)


and _Eq_ is the set of all possible _α_ worlds compatible with the logical annotation _q_ . 

**Lemma 1.** _The gradient of the likelihood has the following form:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0017-12.png)


_where the weight has the form:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0017-14.png)


In case of the Hungarian matching the most probable possible word is selected, which corresponds to setting the conditional probability _P_ ( _E|Oi_ = _j_ ) to 1 if object _i_ is paired with label _j_ and 0 otherwise. The ProbKT gradient can be interpreted as a probability weighted extension of the gradient resulting from the Hungarian matching. 

## D FULL RESULTS 

In Table 6, we present the full results for the MNIST experiment. We report the count accuracy ( _i.e._ . correct identification of the digits in the image), sum accuracy ( _i.e._ correct estimation of the sum of 

17 

Published as a conference paper at ICLR 2023 

|Model|Type|mnist count acc.|mnist sum acc.|mnist mAP (mAP@IoU=0.5)|
|---|---|---|---|---|
|Resnet50-CAM (baseline)|In-distribution|0_._044_±_0_._041|0_._506_±_0_._063|0_._003_±_0_._003(0_._014_±_0_._011)|
|Resnet50-CAM (baseline)|OOD|0_._01_±_0_._009|0_._015_±_0_._004|0_._003_±_0_._002(0_._011_±_0_._007)|
|Resnet50-CAM (baseline)|Source Domain|0_._127_±_0_._132|0_._649_±_0_._108|0_._005_±_0_._004(0_._028_±_0_._018)|
|DETR (Pre-trained)|In-distribution|0_._26_±_0_._012|0_._262_±_0_._01|0_._518_±_0_._014(0_._637_±_0_._017)|
|DETR (Pre-trained)|OOD|0_._173_±_0_._01|0_._177_±_0_._009|0_._51_±_0_._012(0_._632_±_0_._015)|
|DETR (Pre-trained)|Source Domain|0_._859_±_0_._031|0_._86_±_0_._031|0_._781_±_0_._009(0_._957_±_0_._008)|
|DETR (ProbKT)|In-distribution|0_._662_±_0_._064|0_._664_±_0_._065|0_._615_±_0_._025(0_._856_±_0_._037)|
|DETR (ProbKT)|OOD|0_._532_±_0_._083|0_._533_±_0_._082|0_._591_±_0_._03(0_._845_±_0_._038)|
|DETR (ProbKT)|source domain|0_._878_±_0_._023|0_._879_±_0_._023|0_._737_±_0_._014(0_._952_±_0_._009)|
|RCNN (Pre-trained)|In-distribution|0_._292_±_0_._005|0_._298_±_0_._005|0_._632_±_0_._014(0_._685_±_0_._002)|
|RCNN (Pre-trained)|OOD|0_._205_±_0_._004|0_._212_±_0_._004|0_._631_±_0_._013(0_._683_±_0_._002)|
|RCNN (Pre-trained)|source domain|0_._961_±_0_._008|0_._961_±_0_._008|0_._917_±_0_._021(0_._988_±_0_._002)|
|RCNN (ProbKT)|In-distribution|0_._902_±_0_._005|0_._903_±_0_._005|0_._786_±_0_._021(0_._974_±_0_._001)|
|RCNN (ProbKT)|OOD|0_._863_±_0_._008|0_._865_±_0_._008|0_._778_±_0_._021(0_._97_±_0_._001)|
|RCNN (ProbKT)|source domain|0_._967_±_0_._004|0_._967_±_0_._004|0_._873_±_0_._016(0_._989_±_0_._001)|



Table 6: Results of the SUM experiments on the MNIST object detection dataset. Reported test accuracies over the 5 folds. 

the digits in the image) and the mean average precision (mAP) ( _i.e._ a common object detection metric that reflects the ability to predict the positions and labels of the objects). We observe that the Resnet baseline performs poorly, lacking the necessary logic to process this dataset. We used both DETR and RCNN as object detection backbones in our experiments, showing high test accuracies when fine-tuned with our approach. As the results suggest, RCNN backbones lead to better performance than the DETR backbone. 

|Model|Data Domain|CLEVR count acc.|CLEVR mAP (mAP@IoU=0.5)|Mol. count. acc|Mol. mAP (mAP@IoU=0.5)|
|---|---|---|---|---|---|
|Resnet50-CAM|target domain|0_._97_±_0_._005|0_._036_±_0_._014(0_._200_±_0_._071)|**0****_._978****_±_ 0****_._004**|0_._0_±_0_._0(0_±_0)|
|Resnet50-CAM|OOD|0_._831_±_0_._016|0_._029_±_0_._010(0_._153_±_0_._044)|0_._0_±_0_._0|n/a1|
|Resnet50-CAM|source domain|0_._993_±_0_._003|0_._035_±_0_._019(0_._178_±_0_._084)|0_._828_±_0_._021|0_._0_±_0_._0(0_±_0)|
|WSOD-transfer|target domain|0_._944_±_0_._004|0_._844_±_0_._005(0_._988_±_0_._001)|0_._001_±_0_._0|0_._018_±_0_._004(0_._061_±_0_._011)|
|WSOD-transfer|OOD|0_._73_±_0_._011|0_._79_±_0_._005(0_._969_±_0_._001)|0_._003_±_0_._002|n/a1|
|WSOD-transfer|source domain|0_._989_±_0_._001|0_._926_±_0_._001(0_._995_±_0_._0)|0_._0_±_0_._0|0_._021_±_0_._003(0_._069_±_0_._009)|
|DETR-joint|target domain|0_._159_±_0_._133|0_._579_±_0_._012(0_._684_±_0_._019)|0_._357_±_0_._196|0_._197_±_0_._055(0_._481_±_0_._071)|
|DETR-joint|OOD|0_._084_±_0_._039|0_._534_±_0_._012(0_._66_±_0_._012)|0_._024_±_0_._021|n/a1|
|DETR-joint|source dom.|0_._923_±_0_._049|0_._908_±_0_._017(0_._992_±_0_._001)|0_._232_±_0_._127|0_._23_±_0_._063(0_._565_±_0_._08)|
|DETR (Pre-trained)|target domain|0_._0_±_0_._0|0_._498_±_0_._019(0_._533_±_0_._024)|0_._464_±_0_._033|0_._314_±_0_._006(0_._542_±_0_._006)|
|DETR (Pre-trained)|OOD|0_._0_±_0_._0|0_._477_±_0_._013(0_._531_±_0_._021)|0_._002_±_0_._001|n/a1|
|DETR (Pre-trained)|source domain|0_._97_±_0_._009|0_._945_±_0_._009(0_._992_±_0_._001)|0_._581_±_0_._022|0_._409_±_0_._005(0_._722_±_0_._004)|
|ProbKT*(DETR)|target domain|0_._949_±_0_._005|0_._728_±_0_._014(0_._99_±_0_._003)|0_._589_±_0_._042|0_._373_±_0_._02(0_._669_±_0_._045)|
|ProbKT*(DETR)|OOD|0_._741_±_0_._038|0_._606_±_0_._017(0_._977_±_0_._004)|0_._008_±_0_._008|n/a1|
|ProbKT*(DETR)|source domain|0_._985_±_0_._004|0_._937_±_0_._006(0_._995_±_0_._001)|0_._275_±_0_._066|0_._371_±_0_._021(0_._649_±_0_._041)|
|ProbKT(DETR)|target domain|0_._946_±_0_._014|0_._803_±_0_._011(0_._989_±_0_._006)|0_._508_±_0_._027|0_._204_±_0_._02(0_._507_±_0_._014)|
|ProbKT(DETR)|OOD|0_._726_±_0_._035|0_._715_±_0_._006(0_._974_±_0_._006)|0_._004_±_0_._003|n/a1|
|ProbKT(DETR)|source domain|0_._987_±_0_._003|0_._948_±_0_._005(0_._995_±_0_._001)|0_._549_±_0_._026|0_._38_±_0_._013(0_._713_±_0_._006)|
|RCNN (pre-trained)|target domain|0_._0_±_0_._0|0_._586_±_0_._014(0_._598_±_0_._013)|0_._592_±_0_._007|**0****_._568****_±_ 0****_._005**(0_._785_±_0_._004)|
|RCNN (pre-trained)|OOD|0_._0_±_0_._0|0_._582_±_0_._012(0_._603_±_0_._011)|0_._348_±_0_._036|n/a1|
|RCNN (pre-trained)|source domain|0_._988_±_0_._002|**0****_._984****_±_ 0****_._01**(0_._996_±_0_._0)|0_._948_±_0_._004|**0****_._737****_±_ 0****_._005**(**0****_._979****_±_ 0****_._0**)|
|ProbKT*(RCNN)|target domain|0_._974_±_0_._004|0_._855_±_0_._025(**0****_._994****_±_ 0****_._001**)|0_._945_±_0_._006|0_._24_±_0_._042(0_._788_±_0_._073)|
|ProbKT*(RCNN)|OOD|**0****_._901****_±_ 0****_._017**|0_._827_±_0_._022(**0****_._991****_±_ 0****_._001**)|0_._592_±_0_._032|n/a1|
|ProbKT*(RCNN)|source domain|0_._993_±_0_._002|0_._95_±_0_._021(**0****_._998****_±_ 0****_._0**)|0_._96_±_0_._003|0_._655_±_0_._01(0_._974_±_0_._004)|
|ProbKT(RCNN)|target domain|**0****_._975****_±_ 0****_._003**|**0****_._856****_±_ 0****_._039**(0_._993_±_0_._001)|0_._942_±_0_._009|0_._289_±_0_._041(**0****_._829****_±_ 0****_._054**)|
|ProbKT(RCNN)|OOD|0_._89_±_0_._022|**0****_._833****_±_ 0****_._042**(**0****_._991****_±_ 0****_._001**)|**0****_._603****_±_ 0****_._037**|n/a1|
|ProbKT(RCNN)|source domain|**0****_._995****_±_ 0****_._002**|0_._941_±_0_._041(0_._998_±_0_._001)|**0****_._96****_±_ 0****_._002**|0_._666_±_0_._005(0_._978_±_0_._002)|



Table 7: Results of the experiments for the datasets: CLEVR-mini and Molecules. Reported test accuracies over the 5 folds. Best method is in bold for each metric and data distribution. 

1OOD test set of Molecules dataset has no bounding box labels. 

18 

Published as a conference paper at ICLR 2023 

## E SOURCE CODE AND DATASETS 

The source code and basic instructions are available on https://github.com/molden/ ProbKT. The source code integrates features from the Weights & Biases (WandB) platform [5]. Basic features are supported without the need for an account on WandB but to make full use of all features we recommend to create an account. 

Datasets can be downloaded here: 

- CLEVR-mini dataset https://figshare.com/s/db012765e5a38e14ef9c 

- Molecules dataset https://figshare.com/s/3dc3508d39bf4cff8c7f 

- MNIST object detection dataset https://figshare.com/s/ c760de026f000524db5a 

## **ProbLog script used in the ProbKT Probabilistic logical reasoning framework for counting of objects on an image (as on CLEVR-mini dataset):** 

:- use_module(library(lists)). nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9,10,11]) :: digit(X,Y). count([],X,0). count([X|T],X,Y):- count(T,X,Z), Y is 1+Z. count([X1|T],X,Z):- X1\=X,count(T,X,Z). countall(List,X,C) :- sort(List,List1), member(X,List1), count(List,X,C). roll([],L,L). roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y). countpart(List,[],[]). countpart(List,[H|T],[F|L]):- countall(List,H,F), countpart(List,T,L). count_objects(X,L,C) :- roll(X,[],Result), countpart(Result,L,C). 

The query _q_ in the case of class counts would be count_objects(X,L,C). For example an image _X_ with 1 small metal cube and 3 large rubber cylinders would result in the following query: count_objects(X,[small_metal_cube,large_rubber_cylinder],[1,3]). 

**ProbLog script used in the ProbKT Probabilistic logical reasoning framework for aggregating the digits on an image:** 

:- use_module(library(lists)). nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y). sum([],0). sum([X|T],Y):- sum(T,Z), Y is X+Z. roll([],L,L). roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y). sum_digits(X,Y) :- roll(X,[],Result), sum(Result,Y). 

The query _q_ in the case of sum of digits would be sum_digits(X,Y). For example an image _X_ with as sum of digits 12 would result in the following query: sum_digits(X,12). 

**ProbLog script used in the ProbKT Probabilistic logical reasoning framework for taking into account non-exact counts on images** 

19 

Published as a conference paper at ICLR 2023 

:- use_module(library(lists)). nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9,10,11]) :: digit(X,Y). count([],X,0). count([X|T],X,Y):- count(T,X,Z), Y is 1+Z. count([X1|T],X,Z):- X1\=X,count(T,X,Z). countall(List,X,C,A) :- A=0, sort(List,List1), member(X,List1), count(List,X,C). countall(List,X,C,A) :- A=1, sort(List,List1), member(X,List1), count(List,X,R), R>C. countall(List,X,C,A) :- A=-1, sort(List,List1), member(X,List1), count(List,X,R), R<C. 

roll([],L,L). roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y). 

countpart(List,[],[],[]). countpart(List,[H|T],[F|L],[A|B]):- countall(List,H,F,A), countpart(List,T,L,B). 

range_count_objects(X,L,C,S) :- roll(X,[],Result), countpart(Result,L,C,S). 

The query _q_ in the case of non exact counts of objects would be range_count_objects(X,L,C,S). For example an image _X_ with exactly one metal small cube and multiple rubber large spheres would result in the following query: range_count_objects(X,[s_metal_cube,l_rubber_sphere],[1,1],[0,1]). 

## E.1 INFERENCE EXAMPLE FOR MNIST DATASET 

To illustrate the inference process let us follow the evaluation of the clause sum([x1, x2],8), what can result from query sum_digits(X,8) in case of two visible digit in the image X. 

This clause is true if and only if _X_ 1 + _X_ 2 = 8. 

In case of MNIST digits ( _{_ 0 _,_ 1 _, . . . ,_ 9 _}_ ) enumerating the possible worlds would give the following set: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0020-10.png)


After summing the probability of all possible worlds we get: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0020-12.png)


where _p_ 1 and _p_ 2 are the distribution of random variable _X_ 1 and _X_ 2 respectively. 

Or in a general form: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection_images/iclr_2023_Weakly_Supervised_Knowledge_Transfer_with_Probabilistic_Logical_Reasoning_for_Object_Detection.pdf-0020-15.png)


20 

Published as a conference paper at ICLR 2023 

As expected the distribution of the sum is the convolution of the distributions of the two terms. This observation trivially generalizes to more than two terms. The cost function corresponding to the maximum likelihood estimation is the negative log-likelihood _−_ log( _pY_ ( _Y_ )). 

21 

