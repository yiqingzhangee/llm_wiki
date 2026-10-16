Published as a conference paper at ICLR 2023 

## VISUAL RECOGNITION WITH DEEP NEAREST CENTROIDS 

**Wenguan Wang**[1] _[∗†]_ **, Cheng Han**[2] **, Tianfei Zhou** _[∗]_[3] _[∗]_ **& Dongfang Liu**[2] _[†]_ CCAI, Zhejiang University[1] , Rochester Institute of Technology[2] & ETH Zurich[3] 

## ABSTRACT 

We devise deep nearest centroids (DNC), a conceptually elegant yet surprisingly effective network for large-scale visual recognition, by revisiting Nearest Centroids, one of the most classic and simple classifiers. Current deep models learn the classifier in a _fully parametric_ manner, ignoring the latent data structure and lacking explainability. DNC instead conducts nonparametric, case-based reasoning; it utilizes sub-centroidsoftrainingsamplestodescribeclassdistributionsandclearlyexplains the classification as the proximity of test data to the class sub-centroids inthe feature space. Due to the distance-based nature, the network output dimensionality is flexible, and all the learnable parameters are _only_ for data embedding. That means all the knowledge learnt for ImageNet classification can be _completely_ transferred for pixel recognition learning, under the “pre-training and fine-tuning” paradigm. Apart from its nested simplicity and intuitive decision-making mechanism, DNC can even possess _ad-hoc_ explainability when the sub-centroids are selected as actual training images that humans can view and inspect. Compared with parametric counterparts,DNCperformsbetteronimageclassification(CIFAR-10,CIFAR-100, ImageNet) and greatly boosts pixel recognition (ADE20K, Cityscapes) with improved transparency, using various backbone network architectures (ResNet, Swin) andsegmentationmodels(FCN,DeepLabV3, Swin). Our code is available at DNC. 

## 1 INTRODUCTION 

Deep learning models, from convolutional networks ( _e_ . _g_ ., VGG [1], ResNet [2]) to Transformer-based architectures ( _e_ . _g_ ., Swin [3]), push forward the state-of-the-art on visual recognition. With these advancements, parametric softmax classifiers, which learn a set of parameters, _i_ . _e_ ., weight vector, and bias term, for each class, have become the _de facto_ regime in the area (Fig. 1(b)). However, due to the parametricnature,they suffer from several limitations: **First** , they lack simplicity and explainability. The parameters in the classification layer are abstract and detached from the physical nature of the problem being modelled [4]. Thus these classifiers are hard to naturally lend to an explanation that humansareabletoprocess[5]. **Second** ,linearclassifiersaretypicallytrainedtooptimizeclassification accuracy only, paying less attention to modeling the latent data structure. For each class, only one single weight vector is learned in a fully parametric manner. Thus they essentially assume _unimodality_ for each class [6, 7], less tolerant of intra-class variation. **Third** , as each class has its own set of parameters, deep parametric classifiers require the output space with a fixed dimensionality (equal to the number of classes) [8]. As a result, their transferability is limited; when using ImageNet-trained classifiers to initialize segmentation networks ( _i_ . _e_ ., pixel classifiers), the last classification layer, whose parameters are valuable knowledge learnt from the image classification task, has to be thrown away. 

In light of the foregoing discussions, we are motivated to present deep nearest centroids (DNC), a powerful, nonparametric classification network (Fig. 1(d)). Nearest Centroids, which has historical roots dating back to the dawn of artificial intelligence [9–14], is arguably the simplest classifier. Nearest Centroids operates on an intuitive principle: given a data sample, it is directly classified to the class oftrainingexampleswhosemean(centroid)isclosesttoit.Apartfromitsinternal transparency, Nearest Centroids is a classical form of exemplar-based reasoning [5, 11], which is fundamental to our most effective strategies for tactical decision-making[15] (Fig. 1(c)). Numerous past studies[16–18] have shown that humans learn to solve new problems by using past solutions of similar problems. Despite itsconceptual simplicity, empiricalevidenceincognitivescience,andeverpopularity [19–22],Nearest 

> _∗_ Equal contribution 

> _†_ Corresponding author 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0002-01.png)


**----- Start of picture text -----**<br>
parametric classifier Cat...it looks<br>extractionfeature  ([ 1.3,  0.3,  class weight , -2.8],  bias -1.3) score 0.2 like              I ve  Training dataset<br>([-4.7,  9.4,  ,  0.6],  0.2) 0.1 ever seen, but  close<br>not like  far<br>feature<br>([ 2.5, -3.2,  ,  2.1],  0.9) 0.6 space<br>Dog? Pig?  Cat...it looks like<br>... or Cat? It should be Cat, because...umm... ... I ve ever seen, but not like<br>(a)  DNN (b) Human (c)      DNC (d)<br>**----- End of picture text -----**<br>


Figure 1: (b) Prevalent visual recognition models , built upon parametric softmax classifiers, have a few limitations, such as their non-transparent decision-making process. (c) Humans can use past cases as models when solving new problems[16, 18] ( _e_ . _g_ ., comparing withafewfamiliar/exemplar animals for categorization). (d) DNC makes classification based on the similarity of to class subcentroids (representative training examples) in the feature space. The class sub-centroids are vital for capturing underlying data structure, enhancing interpretability, and boosting recognition. 

Centroids and its utility in large datasets with high-dimensional input spaces are widely unknown or ignored by current community. Inheriting the intuitive power of Nearest Centroids, our DNC is able to serve as a _strong yet interpretable backbone_ for large-scale visual recognition; it is fully aware of the aforementioned limitations of parametric counterparts while shows even better performance. 

Specifically, DNC summarizes each class into a set of sub-centroids (sub-cluster centers) by clustering of training data inside the same class, and assigns each test sample to the class with the nearest subcentroid.DNCisessentiallyan _experience_ -/ _distance_ -basedclassifier–itmerelyreliesontheproximity of test query to local means of training data (“quintessential” past observations) in the deep feature space.Assuch,DNClearnsvisualrecognitionbydirectlyoptimizingtherepresentation,insteadofdeep parametric models needing an extra softmax classification layer after feature extraction. For training, DNC alternates between two steps: **i)** _class-wise clustering_ for automatically discovering class subcentroids, and **ii)** _classification prediction_ for supervised representation learning, through retrieving the nearest sub-centroids. However, since the feature space evolves continually during training, computing the sub-centroids is expensive – it requires a pass over the full training dataset after each batch update and limits DNC’s scalability. To solve this, we use a Sinkhorn Iteration[23] based clustering algorithm [24] for fast cluster assignment. We further adopt momentum update with an external memory for estimating online the sub-centroids (whose amount is more than 1K on ImageNet [25]) with small-batch size ( _e_ . _g_ ., 256). Consequently, DNC can be efficiently trained by simultaneously conducting clustering and stochastic optimization on large datasets with small batches, only slowing the training speed slightly ( _e_ . _g_ ., _∼_ 5% on ImageNet). 

DNC enjoys a few attractive qualities: **First** , improved _**simplicity**_ and _**transparency**_ . The intuitive workingmechanismandstatisticalmeaningofclasssub-centroidsmake DNC elegant and easy to understand. **Second** , automated discovery of _**underlying data structure**_ . By within-class deterministic clustering, the latent distribution of each class is automatically mined and fully captured as a set of representative local means. In contrast, parametric classifiers learn one single weight vector per class, intolerant of rich intra-class variations. **Third** , direct supervision of _**representation learning**_ . DNC achieves classification by comparing data samples and class sub-centroids on the feature space. With such distance-based nature, DNC blends unsupervised sub-pattern mining (class-wise clustering) and supervised representation learning (nonparametric classification) in a synergy: local significant patterns are automatically mined to facilitate classification decision-making; the supervisory signal from classification directly optimizes the representation, which in turn boosts meaningful clustering. **Forth** , better _**transferability**_ . DNC learns by _only_ optimizing the feature representation, thus the output dimensionality no longer needs to be as many as the classes. With this algorithmic merit, all the useful knowledge (parameters) learnt from a source task ( _e_ . _g_ ., ImageNet [25] classification) are stored in the representation space, and can be completely transferred to target tasks ( _e_ . _g_ ., Cityscapes [26] segmentation). **Fifth** , _**ad-hocexplainability**_ . If further restricting the class sub-centroids to be samples (images) of the training set, DNC can explain its prediction based on _IF· · ·Then_ rules and allow users to intuitively view the class representatives, and appreciate the similarity of test data to the representative images (detailed in §3&4.3). Such _ad-hoc_ explainability [27] is valuable in safety-sensitive scenarios, and differs DNC from most existing network interpretation techniques [28–30] that only investigate _post-hoc_ explanations and thus fail to elucidate precisely how a model works [31, 32]. 

DNC is an intuitive yet general classification framework; it is compatible with different visual recognition network architectures and tasks.Weexperimentallyshow: **In§4.1** ,withResNet[2]andSwin[3] 

2 

Published as a conference paper at ICLR 2023 

network architectures, DNC outperforms parametric counterparts on image classification, _i_ . _e_ ., **0.230.24** % top-1 accuracyonCIFAR-10[33]and **0.24-0.32** %onImageNet[25],by _trainingfromscratch_ . **In §4.2** , when using our ImageNet-pretrained, nonparametric versions of ResNet and Swin as backbones, our pixel-wise DNC classifier greatly improves the segmentation performance of FCN [34], DeepLabV3 [35], and UperNet [36], on ADE20K [37] ( **1.6-2.5** % mIoU) and Cityscapes [26] ( **1.11.9** % mIoU). These results verify DNC’s strong transferability and high versatility. **In §4.3** , after constraining class sub-centroids as training images of ImageNet, DNC becomes more interpretable, with only **0.12** % sacrifice in top-1 accuracy (but is still **0.17** % better than the parametric counterpart). 

These results are particularly impressive, considering the nonparametric and transparent nature of DNC. We feel this work brings fundamental insights into related fields. 

2 RELATED WORK 

**Distance-/Prototype-based Classifiers.** Among the numerous classification algorithms ( _e_ . _g_ ., logistic regression [38], Na¨ıve Bayes [39], random forest [40], support vector machines [41], and deep neural networks (DNNs) [42]), distance-based methods are particularly remarkable, due to their intuitive working mechanism. Distance-based classifiers are nonparametric and exemplar-driven, relying on similarities between samples and internally stored exemplars/prototypes. Thus they conduct case-based reasoning that humans use naturally in problem-solving, making them appealing and interpretable [16, 43]. _k_ -Nearest Neighbors ( _k_ -NN) [9, 10] is a form of distance-based classifiers; it uses _all_ training data as exemplars [44, 45]. Towards network implementation of _k_ -NN [46– 48], Wu _et al_ . [49] made notable progress; their _k_ -NN network outperforms parametric softmax based ResNet[2] and the learnt representation works well in few-shot settings. However, _k_ -NN classifiers (including the deep learning analogues) cost huge storage space and pose heavy computation burden ( _e_ . _g_ ., persistently retaining the training dataset and making full-dataset retrieval for each query) [50, 51], and the nearest neighbors may not be good class representatives [43]. Nearest Centroids [11–14] is another famous distance-based classifier yet has neither of the deficiencies of _k_ -NN [20, 43]. Nearest Centroids selects representative class centers, instead of all the training data, asexemplars.Guerriero _et al_ .[52] also investigate the idea of bringing Nearest Centroids into DNNs. However, they simply abstract each class into one single class mean, failing to capture complex class-wise distributions and showing weak results even in small datasets [33]. 

The idea of distance-based classification also stimulates the emergence of _prototypical networks_ , which mainly focus on few-shot [53, 54] and zero-shot [55, 56] learning. However, they often associate to each class only one representation (prototype) [57] and their prototypes are usually flexible parameters [51, 53, 56] or defined prior to training [8, 55]. In DNC, a prototype (sub-centroid) is either a generalization of a number of observations or intuitively a typical training visual example. Via clustering based sub-class mining, DNC addresses two key properties of prototypical exemplars: _sparsity_ and _expressivity_ [58, 59]. In this way, the representation can be learnt to capture the underlying class structure, hence facilitating large-scale visual recognition while preserving transparency. 

**Neural Network Interpretability.** As the _black-box_ nature limits the adoption of DNNs in decisioncritical tasks, there has been a recent surge of interest in DNNs’ interpretability. However, most interpretation techniques only produce posteriori explanations for already-trained DNNs, typically by analysis of reverse-engineer importance values [28–30, 60–65] and sensitivities of inputs [66–69]. As many literature outlined, _post-hoc_ explanations are problematic and misleading[32, 43, 70, 71];they cannotexplainwhat _actually_ makesaDNNarriveatitsdecisions[72].Topursue _ad-hoc_ explainability, some attempts have been initiated to develop explainable DNNs, by deploying more interpretable machineries into black-box DNNs [73–75] or regularizing representation with certain properties ( _e_ . _g_ ., sparsity [76], decomposability [77], monotonicity [78]) that can enhance interpretability. 

DNC intrinsically relies on class sub-centroid retrieving. The theoretical simplicity makes it easy to understand; when anchoring the sub-centroids to available observations, DNC can derive intuitive explanations based on the similarities of test samples to representative observations. It simultaneously conducts representation learning and case-based reasoning, making it self-explainable without post-hoc analysis[27]. DNC relates to _concept_ -based explainable networks[4, 5, 72–74, 79–81] that refer to human-friendly concepts/prototypes during decision making. These methods, however, necessitate nontrivial architectural modification and usually resort to pre-trained models, not to mention serving as backbone networks. In sharp contrast, DNC only brings minimal architectural change to parametric classifier based DNNs and yields remarkable performance on ImageNet [25] with training from scratch and _ad-hoc_ explainability. It provides solid empirical evidence, for the first time as far as we know, for the power of case-based reasoning in large-scale visual recognition. 

3 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0004-01.png)


**----- Start of picture text -----**<br>
supervised repre-  sub-centroid based<br>Training dataset feature space sentation learning      (    ) L 7 classification p 1 p 2 x feature space x<br>p 1 p 2 f<br>f p 1 p 2<br>y<br>sub-centroids<br>unsupervised sub-<br>class-wise clustering  pattern discovery (    )5<br>**----- End of picture text -----**<br>


Figure 2: Withadistance-/case-basedclassificationscheme,DNCcombinesunsupervised sub-pattern discovery and supervised representation learning in a synergy. 

## 3 DEEP NEAREST CENTROIDS (DNC) 

**Problem Statement.** Consider the standard visual recognition setting. Let _X_ be the visual space ( _e_ . _g_ ., image space for recognition, pixel space for segmentation), and _Y_ = _{_ 1 _,· · ·, C}_ the set of semantic classes. Given a training dataset _{_ ( _xn, yn_ ) _∈X×Y}[N] n_ =1[, the goal is to use the] _[ N]_[ training examples to fit] a _model_ (or hypothesis) _h_ : _X �→Y_ that accurately predicts the semantic classes for new visual samples. 

**Parametric SoftmaxClassifier.** Current commonpractice is toimplement _h_ as DNNs and decompose it as _h_ = _l ◦f_ . Here _f_ : _X �→F_ is a _feature extractor_ ( _e_ . _g_ ., convolution based or Transformer-like networks) that maps an input sample _x ∈X_ into a _d_ -dimensional representation space _F ∈_ R _[d]_ , _i_ . _e_ ., _**x**_ = _f_ ( _x_ ) _∈F_ ; and _l_ : _F �→Y_ is a _parametric classifier_ ( _e_ . _g_ ., the last fully-connected layer in recognition or last 1 _×_ 1 convolution layer in segmentation) that takes _**x**_ as input and produces class prediction ˆ _y_ = _l_ ( _**x**_ ) _∈Y_ . Concretely, _l_ assigns a query _x ∈X_ to the class _y_ ˆ _∈Y_ according to: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0004-06.png)


where _s[c] ∈_ R indicates the unnormalized prediction score ( _i_ . _e_ ., _logit_ ) for class _c_ , _**w**[c] ∈_ R _[d]_ and _b[c] ∈_ R are learnable parameters – class weight and bias term for _c_ . Parameters of _l_ and _f_ are learnt by minimizing the softmax cross-entropy loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0004-08.png)


Though highly successful, the use of the parametric classifier _l_ has drawbacks as well: **i)** The weight matrix _**W**_ =( _**w**_[1] _,· · ·,_ _**w**[C]_ ) _∈_ R _[d][×][C]_ andbiasvector _**b**_ =( _b_[1] _,· · ·, b[C]_ ) _∈_ R _[d]_ in _l_ are learnable parameters, which cannot provide any information about what makes the model _h_ reach its decisions. **ii)** _l_ makes theloss _L_ onlydependontherelativerelationamonglogits, _i_ . _e_ ., _{s[c] }c_ ,andcannotdirectlysuperviseon the representation _**x**_ [82, 83]. **iii)** _**W**_ and _**b**_ are learnt as flexible parameters, lacking explicit modeling of the underlying data structure. **iv)** The final output dimensionality is constrained to be the number of classes, _i_ . _e_ ., _C_ . During transfer learning, as different visual recognition tasks typically have distinct semanticlabelspaces(withdifferentnumberofclasses),theclassifierfromapretrainedmodelhastobe abandoned, even though the learnt parameters _**W**_ and _**b**_ are valuable knowledge from the source task. 

The question naturally arises: might there be a simple way to address these limitations of current _de facto_ , parametric classifier based visual recognition regime? Here we show that this is indeed possible, even with better performance. 

**DNC Classifier.** Our DNC (Fig. 2) is built upon the intuitive idea of Nearest Centroids, _i_ . _e_ ., assign a sample _x_ to the class _y_ ˆ _∈Y_ with the closest class center: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0004-12.png)


are defaulted towhere _⟨·, ·⟩_ is a distance measure, given as: _ℓ_ 2-normalized from now on. _⟨_ _**x**_ ¯ _**u**[c] ,_ _**v**_ is the mean vector of class _⟩_ = _−_ _**u**[⊤]_ _**v** /∥_ _**u** ∥∥_ _**v** ∥_ . For simplicity, all the features _c_ , _x[c] n_[is a training sample] of _c_ , _i_ . _e_ ., _yn[c]_[=] _[ c]_[, and] _[ N][ c]_[ is the number of training samples in] _[ c]_[. As such, the feature-to-class mapping] _F �→Y_ is achieved in a _nonparametric_ manner and _understandable_ from user’s view, in contrast to the parametric classifier _l_ that learns “non-transparent” parameters for each class. It makes more sense if multiple sub-centroids(local means) per class are used, which is in particular true for challenging visual recognition where complex intra-class variations cannot be simply described by the simple assumption of unimodality of data of each class.When representing each class _c_ as _K_ sub-centroids, denoted by _{_ _**p**[c] k[∈]_[R] _[d][}][K] k_ =1[, the] _[ C]_[-way classification for sample] _[ x]_[ takes place as a] _[ winner-takes-all]_[ rule:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0004-14.png)


Clearly, estimating class sub-centroids needs clustering of training samples within each class. As class 

4 

Published as a conference paper at ICLR 2023 

sub-centroids are sub-cluster centers in the latent feature space _F_ , they are locally significant visual patterns and can comprehensively represent class-level characteristics. DNC can be intuitively understood as selecting and storing prototypical exemplars for each class, and finding classification evidence for a previously unseen sample by retrieving the most similar exemplar. This also aligns with the _prototype theory_ in psychology [17, 84, 85]: prototypes are a typical form of cognitive organisation of real world objects. DNC thus emulates the case-based reasoning process that we humans are accustomed to[27]. For instance, when ornithologists classify a bird, they will compare it with those typical exemplars from known bird species to decide which species the bird belongs to [43]. 

**Sub-centroid Estimation.** To find informative sub-centroids that best represent classes, we perform deterministic clustering within each class on the representation space _F_ . More specifically, for each class _c_ , we cluster all the representations _{_ _**x**[c] n[∈]_[R] _[d][}][N] n_ =1 _[ c]_[into] _[ K]_[clusters][whose][centers][are][used][as] the sub-centroids of _c_ , _i_ . _e_ ., _{_ _**p**[c] k[∈]_[R] _[d][}][K] k_ =1[. Let] _**[ X]**[c]_[=[] _**[x]**_ 1 _[c][,][ · · ·][,]_ _**[ x]**[c] N[c]_[]] _[∈]_[R] _[d][×][N][c]_[and] _**[ P]**[ c]_[=[] _**[p]**_ 1 _[c][,][ · · ·][,]_ _**[ p]**[c] K_[]] _[∈]_ R _[d][×][K]_ denotethefeatureandsub-centroidmatrixes,respectively.Thedeterministicclustering, _i_ . _e_ ., the mapping from _**X**[c]_ to _**P**[c]_ , can be denoted as _**Q**[c]_ = [ _**q**_ 1 _[c][,][ · · ·][,]_ _**[ q]** N[c][c]_[]] _[ ∈{]_[0] _[,]_[ 1] _[}][K][×][N][ c]_[,][where] _[ n]_[-] _[th]_[ column] _**q** n[c][∈{]_[0] _[,]_[ 1] _[}][K]_[ is an one-hot assignment vector of] _[ n]_[-] _[th]_[ sample] _[ x][c] n[w.r.t]_[ the] _[ K]_[clusters.] _**[Q]**[c]_[is desired to] maximize the similarity between _**X**[c]_ and _**P**[c]_ , leading to the following binary integer program (BIP): 

max �( _**Q**[c]_ ) _[⊤]_ ( _**P**[c]_ ) _[⊤]_ _**X**[c]_[�] _, Q[c]_ = _{_ _**Q**[c] ∈{_ 0 _,_ 1 _}[K][×][N][c] |_ ( _**Q**[c]_ ) _[⊤]_ **1** _K_ = **1** _N[c] },_ (5) _**Q**[c] ∈Q[c]_[Tr] where **1** _K_ is a _K_ -dimensional all-ones vector. As in [24], we relax _Q[c]_ to be a _transportation polytope_ [23]: _Q[′][c]_ = _{_ _**Q**[c] ∈_ R _[K]_ + _[×][N][ c] |_ ( _**Q**[c]_ ) _[⊤]_ **1** _K_ = **1** _N c ,_ _**Q**[c]_ **1** _N c_ = _[N] K[ c]_ **[1]** _[K][}]_[,casting BIP(][5][) into an] _[optimal trans-] port_ problem.In _Q[′][c]_ ,besidesthe _one-hot assignment_ constraint ( _i_ . _e_ ., ( _**Q**[c]_ ) _[⊤]_ **1** _K_ = **1** _N c_ ),an _equipartition_ constraint ( _i_ . _e_ ., _**Q**[c]_ **1** _N c_ = _[N] K[c]_ **[1]** _[K]_[)][is added to inspire] _[ N][ c]_[ samples to be evenly assigned to] _[ K]_[ clusters.] This can efficiently avoid degeneracy, _i_ . _e_ ., mapping all the data to the same cluster. Then the solution can be given by a fast version [23] of Sinkhorn-Knopp algorithm [86], in a form of a normalized exponential matrix: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0005-04.png)


where the exponentiation is performed element-wise, _**α** ∈_ R _[K]_ and _**β** ∈_ R _[N][c]_ are two renormalization vectors, which can be computed using a small number of matrix multiplications via SinkhornKnopp Iteration [23], and _ε_ = 0 _._ 05 trades off convergence speed with closeness to the original transport problem. In short, by mapping data samples into a few clusters under the constraints _Q[′][c]_ , we pursue _sparsity_ and _expressivity_ [58, 59], making class sub-centroids representative of the dataset. 

**Training of DNC = Supervised Representation Learning+Automatic Sub-class Pattern Mining.** Ideally, according to class-wise cluster assignments _{_ _**Q**[c] }[C] c_ =1[, we can get totally] _[ CK]_[ sub-centroids] _{_ _**p**[c] k[}][C,K] c,k_ =1[,] _[ i]_[.] _[e]_[., mean feature vectors of the training data in the] _[ CK]_[clusters.][Then the training target] becomes as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0005-07.png)


Comparing(2)and(7),astheclasssub-centroids _{_ _**p**[c] k[}][c,k]_[are derived solely from data representations,] DNC learns visual recognition by directly optimizing the representation _f_ , instead of the parametric classifier _l_ . Moreover, with such a nonparametric, distance-based scheme, DNC builds a closer link to metric learning [87–93]; DNC can even be viewed as learning a metric function _f_ to compare data samples _{xn}n_ , under the guidance of the corresponding semantic labels _{yn}n_ . 

During training, DNC alternates two steps iteratively: **i)** class-wise clustering (5) for automatic subcentroid discovery, and **ii)** sub-centroid based classification for supervised representation learning (7). Through clustering, DNC probes underlying data distribution of each class, and produces informative sub-centroids by aggregating statistics from data clusters. This automatic sub-class discovery process also enjoys a similar spirit of recent clustering based unsupervised representation learning [24, 94– 101]. However, it works in a class-wise manner, since the class label is given. In this way, DNC optimizes the representation by adjusting the arrangement between sub-centroids and data samples. The enhanced representation in turn helps to find more informativesub-centroids,benefitingclassification eventually. As such, DNC conducts _unsupervised sub-class pattern discovery_ during _supervised representation learning_ , distinguishing it from most (if not all) of current visual recognition models. 

Since the latent representation _f_ evolves continually during training, class sub-centroids should be synchronized, which requires performing class-wise clustering over all training data after each batch 

5 

Published as a conference paper at ICLR 2023 

update. This is highly expensive on large datasets, even though Sinkhorn-Knopp iteration[23] based clustering (6) is highly efficient. To circumvent the expensive, offline sub-centroid estimation, we adopt _momentum update_ and _online_ clustering. Specifically, at each training iteration, we conduct class-wise clustering on the current batch and update each sub-centroid as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0006-02.png)


where _µ ∈_ [0 _,_ 1] is a momentum coefficient, and ¯ _**x**[c] k[∈]_[R] _[d]_[ is the mean feature vector of data assigned to] ( _c, k_ )-cluster in current batch. As such, the sub-centroids can be up-to-date with the change of parameters. Though effective enough in most cases, batch-wise clustering could not extend to a large number of classes, _e_ . _g_ ., when training on ImageNet[25] with 1K classes using batch size 256, not all the classes/clusters are present in a batch. To solve this, we store features from several prior batches in a memory, and do clustering on both the memory and current batch. DNC can be trained by gradient _∼_ backpropagation in small-batch setting, with negligible lagging ( 5% training delay on ImageNet). 

**Versatility.** DNC is a general framework; it can be effortless integrated into any parametric classifier based DNNs, with minimal architecture change, _i_ . _e_ ., removing the parametric softmax layer. However, DNC changes the classification decision-making mode, reforms the training regime, and makes the reasoning process more transparent, without slowing the inference speed. DNC can be appliedtovariousvisualrecognitiontasks,includingimageclassification(§4.1)andsegmentation(§4.2). 

**Transferability.** As a nonparametric scheme, DNC can handle an arbitrary number of classes with fixed output dimensionality ( _d_ ); all the knowledge learnt on a source task ( _e_ . _g_ ., ImageNet classification with 1K classes) are stored as a constant amount of parameters in _f_ , and thus can be _completely_ transferred for a new task ( _e_ . _g_ ., Cityscapes[26] segmentation with 19 classes), under the“pre-trainingand fine-tuning” paradigm. In a similar setting, the parametric counterpart has to discard 2M parameters during transfer learning ( _d_ =2048 when using ResNet101 [2]). See §4.2 for related experiments. 

**Ad-hoc Explainability.** DNC is a transparent classifier that has a built-in case-based reasoning process, as the sub-centroids are summarized from real observations and actually used during classification. So far we only discussed the case where the sub-centroids are considered as average mean feature vectors of a few training samples with similar patterns. When restricting the sub-centroids to be elements of the training set ( _i_ . _e_ ., representative training images), DNC naturally comes with human-understandable explanations for each prediction, and the explanations are loyal to the internal decision mode and not created post-hoc. Studies regarding _ad-hoc_ explainability are given in §4.3. 

## 4 EXPERIMENT 

## 4.1 EXPERIMENTS ON IMAGE CLASSIFICATION 

**Dataset.** The evaluation for image classification is carried out on CIFAR-10[33] and ImageNet[25]. 

**Network Architecture.** For completeness, we craft DNC on popular CNN-based ResNet50/100 [2] and recent Transformer-based Swin-Small/-Base [3]. Note that, we only remove the last linear classification layer, and the final output dimensionality of DNC is as many as the last layer feature of the parametric counterpart, _i_ . _e_ ., 2,048 for ResNet50/100, 768 for Swin-Small, and 1,024 for Swin-Base. 

**Training.** We use mmclassification[1] as codebase and follow the _default_ training settings. For CIFAR-10, we train ResNet for 200 epochs, with batch size 128. ThememorysizeforDNCmodelsis setas100batches.ForImageNet,wetrain100and300epochs with batch size 16 for ResNet and Swin, respectively. The initial learning rates of ResNet and Swin are set as 0 _._ 1 and 0 _._ 0005, scheduled by a step policy and polynomial annealing policy, respectively. Limited by our GPU capacity,thememory sizesaresetas 1 _,_ 000 and 500 batchesforDNCversionsofResNetandSwin,respectively.Otherhyperparametersareempiricallysetas: _K_ =4 and _µ_ =0 _._ 999. Table 1: **Classification top-1 accuracy top-1 accuracy accuracy** on Models are trained _from scratch_ on eight V100 GPUs. 

Table 1: **Classification top-1 accuracy top-1 accuracy accuracy** on CIFAR-10[33] test.#Params:thenumberof learnable parameters (same for other tables). 

**Results on CIFAR-10.** Table 1 compares DNC with the parametric counterpart, based on the most representative CNN network architecture, _i_ . _e_ ., ResNet. As seen,DNCobtainsbetterperformance:DNCis **0.23** % higheronResNet50,and **0.24** %higheronResNet101, usingfewerlearnableparameters.Withtheexactsame 

|Method<br>Backbone|#Params<br>top-1|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|23.52M<br>95.55%<br>23.50M<br>**95.78**%|
|ResNet [2] ResNet101<br>**DNC**-ResNet|42.51M<br>95.58%<br>42.49M<br>**95.82**%|



backbone architectures and training settings, one can safely attribute the performance gain to DNC. 

> 1https://github.com/open-mmlab/mmclassification 

6 

Published as a conference paper at ICLR 2023 

**Results on ImageNet.** Table 2 illustrates again our compelling results over different vision network architectures. In terms of top-1 acc., our DNC exceeds the parametric classifier by **0.29** % and **0.28** %onResNet50andResNet101,respectively. DNC also obtains promising results with Transformer architecture: **83.26** % _vs_ 83.02% on Swin-S, **83.68** % _vs_ 83.36% on Swin-B. These results are impressive, considering the transparent, case-based reasoning nature of DNC. One may notice another nonparametric, _k_ -NN classi- 

Table 2: **Classification top-1 and top-5 accuracy** on ImageNet [25] val. 

|Method<br>Backbone|#Params<br>top-1<br>top-5|
|---|---|
|ResNet [2] ResNet50<br>**DNC**-ResNet|25.56M 76.20% 93.01%<br>23.51M **76.49**% **93.08**%|
|ResNet [2] ResNet101<br>**DNC**-ResNet|44.55M 77.52% 93.06%<br>42.50M **77.80**% **93.85**%|
|Swin [3]<br>Swin-S<br>**DNC**-Swin|49.61M 83.02% 96.29%<br>48.84M **83.26**% **96.40**%|
|Swin [3]<br>Swin-B<br>**DNC**-Swin|87.77M 83.36% 96.44%<br>86.75M **83.68**% **97.02**%|



fier [49] reports 76.57% top-1 acc.,basedonResNet50.However,[49]istrainedwith 130 epochs; in this setting, DNC gains 76.64%. Moreover, as mentioned in §2, [49] poses huge storage demand, _i_ . _e_ ., retaining the whole ImageNet training set ( _i_ . _e_ ., 1.2M images) to perform the _k_ -NN decision rule, and suffers from very low efficiency, caused by extensive comparisons between each test image and all the training images. These limitations prevent the adoption of [49] in real application scenarios. In contrast,DNConlyreliesona small set of class representatives ( _i_ . _e_ .,four sub-centroids per class) for classification decision-making and causes no extra computation budget during network deployment. 

## 4.2 EXPERIMENTS ON SEMANTIC SEGMENTATION 

**Dataset.** TheevaluationforsemanticsegmentationiscarriedoutonADE20K[37]andCityscapes[26]. 

**Segmentation Network Architecture.** For comprehensive evaluation, we approach DNC on three famous segmentation models ( _i_ . _e_ ., FCN [34], DeepLabV3[35], UperNet [36]), using two backbone architectures ( _i_ . _e_ ., ResNet101[2] and Swin-B[3]).Forthesegmentationmodels,theonlyarchitec- tural modification is the removalofthe“segmentationhead”( _i_ . _e_ ., 1 _×_ 1 convbased,pixel-wiseclassi- fication layer).Forthebackbonenetworks,werespectivelyadoptparametricclassifier based and our nonparametric, DNC based versions, which are both trained on ImageNet [25] and reported in Table 2, for initialization. Thus for each segmentation model, we derive four variants from the different combinations of parametric and DNC versions of the backbone and segmentation network architectures. 

**Training.** We adopt mmsegmentation[2] as the codebase, and follow the _default_ training settings. We train FCN and DeepLabV3 with ResNet101 using SGD optimizer with an initial learning rate 0 _._ 1, and UperNet with Swin-B using AdamW with an initial learning rate 6e-5. For all the models, the learning rate is scheduled following a polynomial annealing policy. As common practices [102, 103], wetrainthemodelsonADE20K train withcropsize 512 _×_ 512 andbatchsize 16; on Cityscapes train with crop size 769 _×_ 769 and batch size 8. All the models are trained for 160K iterations on both datasets. Standard data augmentation techniques are used, including scale and color jittering, flipping, and cropping. The hyper-parameters of DNC are by default set as: _K_ =10 and _µ_ =0 _._ 999. 

**Performance on Segmentation.** As summarized in Table 3, our DNC segmentation models, no matter whether using DNC based backbones, obtain better performance over parametric competitors, _i_ . _e_ ., FCN + ResNet101, DeepLabV3 + ResNet101, UperNet + Swin-B,acrossdifferent segmentation and backbone network architectures. Taking as an example the famous FCN, the original version gains 39.91% and 75.6% mIoU on ADE20K and Cityscapes, respectively. By comparison, with the same backbone – ResNet101, DNC-FCN boosts the scores to 41.1% 

Table 3: **Segmentation mIoU score** on ADE20K [37] val and Cityscapes [26] val (top-1 acc. on ImageNet [25] val of backbones are also reported for reference). 

|Method|ImageNet<br>Backbone|ADE20K<br>Cityscapes<br>#Params|
|---|---|---|
||top-1acc.|mIoU<br>mIoU|
|FCN [34]<br>**DNC**-FCN|ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%<br>ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%|68.6M<br>39.9%<br>75.6%<br>68.6M 40.4%_↑_0.5 76.3%_↑_0.7<br>68.5M 41.1%_↑_1.2 76.7%_↑_1.1<br>68.5M **42.3**%_↑_**2.4 77.5**%_↑_**1.9**|
|DeepLabV3 [35]<br>**DNC**-DeepLabV3|ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%<br>ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%|62.7M<br>44.1%<br>78.1%<br>62.7M 44.6%_↑_0.5 78.7%_↑_0.6<br>62.6M 45.0%_↑_0.9 79.1%_↑_1.0<br>62.6M **45.7**%_↑_**1.6 79.8**%_↑_**1.7**|
|UperNet [36]<br>**DNC**-UperNet|Swin-B [3]<br>83.36%<br>**DNC**-Swin-B<br>83.68%<br>Swin-B [3]<br>83.36%<br>**DNC**-Swin-B<br>83.68%|90.6M<br>48.0%<br>79.8%<br>90.6M 48.4%_↑_0.4 80.1%_↑_0.3<br>90.5M 48.6%_↑_0.6 80.5%_↑_0.7<br>90.5M **50.5**%_↑_**2.5 80.9**%_↑_**1.1**|



and 76.7%. When turning to DNC pre-trained backbone – DNC-ResNet101, our DNC-FCN outperforms again its parametric counterpart – FCN, _i_ . _e_ ., 42.3% _vs_ 40.4% on ADE20K, 77.5% _vs_ 76.3% on Cityscapes. Similar trends can be also observed for DeepLabV3 and UperNet. 

> 2https://github.com/open-mmlab/mmsegmentation 

7 

Published as a conference paper at ICLR 2023 

**Analysis on Transferability.** One appealing feature of DNC is its strong transferability, as DNC learns classification by directly comparing data samples in the feature space. The results in Table 3 also evidence this point. For example, DeepLabV3 – even a parametric classifier based segmentation model – can achieve large performance gains, _i_ . _e_ ., 44.6% _vs_ 44.1% on ADE20K, 78.7% _vs_ 78.1% on Cityscapes, after using DNC-ResNet101 for fine-tuning. When it comes to DNC-DeepLabV3, better performance can be achieved, after replacing ResNet101 with DNC-ResNet101, _i_ . _e_ ., 45.7% _vs_ 45.0% on ADE20K, 79.8% _vs_ 79.1% on Cityscapes. We speculate this is because, when the segmentation and backbone networks are both built upon DNC, the model only needs to adapt the original representation space to the target task, without learning any extra new parameters. Also, these impressive results reflect the innovative opportunity of applying our DNC for more downstream visual recognition tasks, either as a new classification network architecture or a strong and transferable backbone. 

4.3 STUDY OF AD-HOC EXPLAINABILITY 

So far, we have empirically showed that, by inheriting the intuitive power of Nearest Centroids and the strong representation learning ability of DNNs, DNC can serve as a transparent yet powerful tool for visual recognition tasks with improved transferability. When anchoring the class sub-centroids to real observations ( _i_ . _e_ ., actual images from the training dataset), instead of selecting them as cluster centers ( _i_ . _e_ ., mean features of a set of training images), one may expect that DNC will gain enhanced _ad-hoc_ explainability. Next we will show this is indeed possible, only at negligible performance cost. 

**Experimental Setup.** We still train DNC version of ResNet50 on ImageNet train for 100 epochs. In the first 90 epochs, the model is trained in a standard manner, _i_ . _e_ ., computing the class sub-centroids as cluster centers. Then we anchor each sub-centroid to its closest training image, based on the cosine similarity of features. In the final 10 epochs, the sub-centroids are only updated as the features of their anchored training 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0008-05.png)


**----- Start of picture text -----**<br>
IF  ([ I ,        ]>[ I ,        ]  AND AND [ I,         ]>[ I,         ])<br>OR  ([ I ,        ]>[ I ,        ]  AND AND [ I,         ]>[ I,         ])<br>OR ([ I ,        ]>[ I ,        ]  AND AND [ I,         ]>[ I,         ])<br>OR ([ I ,        ]>[ I ,        ]  AND AND [ I,         ]>[ I,         ])  THEN  ( goose )<br>goose<br>desk<br>convertible<br>castle<br>computer<br>**----- End of picture text -----**<br>


images. Besides this, all the other training settings are as normal. In this way, we can get a more interpretable DNC-ResNet50. 

**Interpretable Class Subcentroids.** The top of Fig.3 plots our discovered subcentroid images for four 

Figure 3: _Top:_ sub-centroid images. _Bottom:_ rule created for “goose”. 

ImageNet classes. These representative images are diverse in appearance, viewpoints, illuminations, scales, _etc._ , characterizing their corresponding classes and allowing humans to view and understand. 

Table 4: **Classification top-1 and top-5 accuracy** onImageNet[25]25]] val,usingclustercenter _vs_ resembling real observation as class subcentroids,, basedonDNC-ResNet50architecture. Sub-centroid Architecture top-1p-1-1 top-5 _cluster center_ 76.49% 93.08% **DNC** -ResNet50 _real observation_ 76.37% 93.04% - ResNet50[2][2]2]] 76.20% 93.01% 

**Performance with Improved Interpretability.** We Table 4: **Classification top-1** then report the score of our DNC-ResNet50 based **curacy** onImageNet[25]25]] on the interpretable class representatives on Imageter _vs_ Net val. As shown in Table 4, enforcing the class centroids,, sub-centroids as real training images only brings Sub-centroid Architecture top-1p-1-1 marginal performance degradation ( _e_ . _g_ ., 76.49% _→ cluster center_ **DNC** -ResNet50 76.49% 76.37% top-1 acc.), while coming with better in- _real observation_ 76.37% terpretability. More impressively, _**our explainable**_ - ResNet50[2][2]2]] 76.20% _**DNC-ResNet50 even outperforms the vanilla black-box ResNet50**_ , _e_ . _g_ ., 76.37% _vs_ 76.20%. 

**Explain Inner Decision-Making Mode based on** _**IF** · · ·_ _**Then**_ **Rules.** With the simple Nearest Centroids mechanism, we can use the representative images to form a set of _IF · · ·Then_ rules [4], so as to intuitively interpret the inner decision-making mode of DNC for human users. In particular, let _I_[ˆ] denote a sub-centroid image for class _c_ , _I_[ˇ] 1: _T_ representative images for all the other classes, and _I_ a query image. One linguistic logical _IF · · ·Then_ rule can be generated for _I_[ˆ] : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0008-13.png)


where [ _·, ·_ ] stands for similarity, given by DNC. The final rule for class _c_ is created by combining all the rules of _K_ sub-centroid images _I_[ˆ] 1: _K_ of class _c_ (see Fig. 3 bottom): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0008-15.png)


8 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0009-01.png)


**----- Start of picture text -----**<br>
18.0% 0.00005% 0.000006% 48.1% 1.1% 0.0008%<br>80.9% 50.7%<br>DNC s prediction:  toucan DNC s prediction:  white wolf<br>(a) Groundtruth:  toucan (b) Groundtruth:  kuvasz<br>Pyrenees<br>vulture wolf Great  ice bear<br>toucan hornbill magpie white kuvasz<br>**----- End of picture text -----**<br>


Figure 4: DNC can provide (dis)similarity-based interpretation. For the two test samples, we only plot the normalized similarities for their corresponding closest sub-centroids from top-4 scoring classes. Table 5: A set of ablative experiments on ImageNet [25] val and ADE20K [37] val. 

|Memory|ImageNet|||ImageNet|ADE20K|
|---|---|---|---|---|---|
|(#_batch_)|top-1<br>top-5||_µ_|top-1<br>top-5|mIoU|
|0<br>700<br>800<br>900<br>**1000**|77.49%<br>93.09%<br>77.58%<br>93.16%<br>77.64%<br>93.35%<br>77.75%<br>93.67%<br>**77.80**%<br>**93.85**%||0<br>0_._9<br>0_._99<br>**0.999**<br>0_._9999|73.82%<br>93.02%<br>76.41%<br>93.07%<br>77.33%<br>93.51%<br>**77.80**%<br>**93.85**%<br>77.31%<br>93.48%|42.7%<br>43.6%<br>44.0%<br>**44.3**%<br>44.2%|



(a) Number of sub-centroids 

( _K_ ) for each class (b) Memory size (c) Momentum coefficient ( _µ_ ) **Interpret Prediction Based on (Dis)similarity to Sub-centroid Images.** Based on the interpretable class representatives, DNC can explain its predictions by letting users view and verify its computed (dis)similarity between query and class sub-centroid images. As shown in Fig. 4(a), an observation is correctly classified, as DNC thinks it looks (more) like a particular exemplar of “ _toucan_ ”. However, inFig. 4(b),DNCstrugglestoassigntheobservationtotwoexemplarsfrom“ _whitewolf_ ”and “ _kuvasz_ ” respectively, and makes a wrong decisionfinally.ThoughusersareunclearhowDNCmapsanimageto feature, they can easily understand the decision-making mode [43] ( _e_ . _g_ ., why is one class predicted over another), and verify the calculated (dis)similarity – the evidence for classification decision. 

## 4.4 DIAGNOSTIC EXPERIMENT 

To perform extensive ablation experiments, we train ResNet101 classification and DeepLabV3 segmentation models for 100 epochs and 80K iterations, on ImageNet and ADE20K, respectively. **Class Sub-centroids.** Table 5a studies the impact of the number of class sub-centroids ( _K_ ) for each class. When _K_ = 1, each class is represented by its centroid – the average feature vector of all the training samples of the class (Eq. 3), without clustering. The corresponding baseline obtains 77 _._ 31% top-1 acc. and 43 _._ 2% mIoU for classification and segmentation, respectively. For classification, increasing _K_ from 1 to 4 leads to better performance ( _i_ . _e_ ., 77 _._ 31% _→_ 77 _._ 80%). This supports our hypothesis that one single class weight/center is far from enough to capture the underlying data distribution and proves the efficacy of our clustering based sub-class pattern mining. We stop using _K >_ 4 as the required memory exceeds the computational limit of our hardware. We find similar trends on segmentation; using more sub-centroids ( _K_ : 1 _→_ 10)brings noticeable performance boost: 43.2% _→_ 44.3%.However,increasing _K_ above10provides marginalorevennegativegain.Thismaybebecause over-clustering finds some insignificant patterns, which are trivial or harmful for decision-making. **External Memory.** We next study the influence of the external memory, only used in image classification. As shown in Table 5b, DNC gradually improves the performance as the increase of the memory size. It reaches 77 _._ 80% top-1 acc. at size 1000. However, the results are still not reaching the performance saturating point, but rather the upper limit of our hardware’s computational budget. **Momentum Update.** Last, we ablate the effect of the momentum coefficient _µ_ (Eq. 8) that controls the speed of sub-centroid online updating. From Table 5c we find the behaviors of _µ_ are consistent in both tasks. In particular, DNC performs well with larger coefficients ( _i_ . _e_ ., _µ ∈_ [0 _._ 999 _,_ 0 _._ 9999]), signifying the importance of slow updating. The performance degrades when _µ ∈_ [0 _._ 9 _,_ 0 _._ 99], and encounters a large drop when _µ_ =0 ( _i_ . _e_ ., only using batch sub-centroids as approximations). 

## 5 CONCLUSION 

We present deep nearest centroids (DNC), building upon the classic idea of classifying data samples according to nonparametric class representatives. Compared to classic parametric models, DNC has merits in: **i)** systemic simplicity by bringing the intuitive Nearest Centroids mechanism to DNNs; **ii)** automated discovery of latent data structure using within-class clustering; **iii)** direct supervision of representation learning, boosted by unsupervised sub-pattern mining; **iv)** improved transferability that lossless transfers learnable knowledge across tasks; and **v)** _ad-hoc_ explainability by anchoring class exemplars with real observations. Experiments confirm the efficacy and enhanced interpretability. 

9 

Published as a conference paper at ICLR 2023 

## ACKNOWLEDGEMENT 

This research was supported by the National Science Foundation under Grant No. 2242243. 

## REFERENCES 

- [1] Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: ICLR (2015) 1 

- [2] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016) 1, 2, 3, 6, 7, 8, 18, 19, 20, 21, 22, 23 

- [3] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021) 1, 2, 6, 7, 18, 19, 20, 28, 29, 30 

- [4] Angelov, P., Soares, E.: Towards explainable deep neural networks (xdnn). Neural Networks **130** , 185–194 (2020) 1, 3, 8 

- [5] Li, O., Liu, H., Chen, C., Rudin, C.: Deep learning for case-based reasoning through prototypes: A neural network that explains its predictions. In: AAAI (2018) 1, 3 

- [6] Hayashi, H., Uchida, S.: A discriminative gaussian mixture model with sparsity. In: ICLR (2021) 1 

- [7] Zhou, T., Wang, W., Konukoglu, E., Van Gool, L.: Rethinking semantic segmentation: A prototype view. In: CVPR (2022) 1 

- [8] Mettes, P., van der Pol, E., Snoek, C.: Hyperspherical prototype networks. In: NeurIPS (2019) 1, 3 

- [9] Fix, E., Hodges Jr, J.L.: Discriminatory analysis - nonparametric discrimination: Small sample performance. Tech. rep., California Univ Berkeley (1952) 1, 3 

- [10] Cover, T., Hart, P.: Nearest neighbor pattern classification. IEEE TIT **13** (1), 21–27 (1967) 3 

- [11] Kolodner, J.L.: An introduction to case-based reasoning. Artificial Intelligence Review **6** (1), 3–34 (1992) 1, 3 

- [12] Kohonen, T.: Learning vector quantization. In: Self-Organizing Maps, pp. 175–189 (1995) 

- [13] Chaudhuri, B.: A new definition of neighborhood of a point in multi-dimensional space. Pattern Recognition Letters **17** (1), 11–17 (1996) 

- [14] Webb, A.R.: Statistical pattern recognition. John Wiley & Sons (2003) 1, 3 

- [15] Kim, B., Rudin, C., Shah, J.A.: The bayesian case model: A generative approach for casebased reasoning and prototype classification. In: NeurIPS (2014) 1 

- [16] Newell, A., Simon, H.A., et al.: Human problem solving, vol. 104 (1972) 1, 2, 3 

- [17] Rosch, E.H.: Natural categories. Cognitive Psychology **4** (3), 328–350 (1973) 5 

- [18] Aamodt, A., Plaza, E.: Case-based reasoning: Foundational issues, methodological variations, and system approaches. AI Communications **7** (1), 39–59 (1994) 1, 2 

- [19] Slade, S.: Case-based reasoning: A research paradigm. AI magazine **12** (1), 42–42 (1991) 1 

- [20] S´anchez, J.S., Pla, F., Ferri, F.J.: On the use of neighbourhood-based non-parametric classifiers. Pattern Recognition Letters **18** (11-13), 1179–1186 (1997) 3 

- [21] Gou, J., Yi, Z., Du, L., Xiong, T.: A local mean-based k-nearest centroid neighbor classifier. The Computer Journal **55** (9), 1058–1071 (2012) 

- [22] Mensink, T., Verbeek, J., Perronnin, F., Csurka, G.: Distance-based image classification: Generalizing to new classes at near-zero cost. IEEE TPAMI **35** (11), 2624–2637 (2013) 1 

- [23] Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. In: NeurIPS (2013) 2, 5, 6, 22, 24, 28 

- [24] Asano, Y.M., Rupprecht, C., Vedaldi, A.: Self-labelling via simultaneous clustering and representation learning. In: ICLR (2020) 2, 5, 24, 28 

10 

Published as a conference paper at ICLR 2023 

- [25] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M.S., Berg, A.C., Li, F.: Imagenet large scale visual recognition challenge. IJCV **115** (3), 211–252 (2015) 2, 3, 6, 7, 8, 9, 18, 19, 20, 21, 23, 24, 25, 26 

- [26] Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: CVPR (2016) 2, 3, 6, 7, 20, 23, 24, 28, 30 

- [27] Biehl, M., Hammer, B., Villmann, T.: Prototype-based models in machine learning. Wiley Interdisciplinary Reviews: Cognitive Science **7** (2), 92–111 (2016) 2, 3, 5 

- [28] Simonyan, K., Vedaldi, A., Zisserman, A.: Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034 (2013) 2, 3 

- [29] Zeiler, M.D., Fergus, R.: Visualizing and understanding convolutional networks. In: ECCV (2014) 

- [30] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Learning deep features for discriminative localization. In: CVPR (2016) 2, 3 

- [31] Lipton, Z.C.: The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery. Queue **16** (3), 31–57 (2018) 2 

- [32] Rudin, C.: Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence **1** (5), 206–215 (2019) 2, 3 

- [33] Krizhevsky, A., Hinton, G.: Learning multiple layers of features from tiny images. Technical report, University of Toronto (2009) 3, 6, 18, 19, 20, 21, 22, 23 

- [34] Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: CVPR (2015) 3, 7, 19, 20 

- [35] Chen, L.C., Papandreou, G., Schroff, F., Adam, H.: Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 (2017) 3, 7, 19, 20, 23 

- [36] Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: ECCV (2018) 3, 7, 19, 20, 28, 29, 30 

- [37] Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. In: CVPR (2017) 3, 7, 9, 20, 28, 29 

- [38] Friedman, J.H.: The elements of statistical learning: Data mining, inference, and prediction. Springer (2009) 3 

- [39] Hand, D.J., Yu, K.: Idiot’s bayes—not so stupid after all? International Statistical Review **69** (3), 385–398 (2001) 3 

- [40] Breiman, L.: Random forests. Machine Learning **45** (1), 5–32 (2001) 3 

- [41] Cortes, C., Vapnik, V.: Support-vector networks. Machine Learning **20** (3), 273–297 (1995) 3 

- [42] LeCun, Y., Bengio, Y., Hinton, G.: Deep learning. nature **521** (7553), 436–444 (2015) 3 

- [43] Rudin, C., Chen, C., Chen, Z., Huang, H., Semenova, L., Zhong, C.: Interpretable machine learning: Fundamental principles and 10 grand challenges. Statistics Surveys **16** , 1–85 (2022) 3, 5, 9 

- [44] Boiman, O., Shechtman, E., Irani, M.: In defense of nearest-neighbor based image classification. In: CVPR (2008) 3 

- [45] Guillaumin, M., Mensink, T., Verbeek, J., Schmid, C.: Tagprop: Discriminative metric learning in nearest neighbor models for image auto-annotation. In: ICCV (2009) 3 

- [46] Lippmann, R.P.: Pattern classification using neural networks. IEEE Communications Magazine **27** (11), 47–50 (1989) 3 

- [47] Salakhutdinov, R., Hinton, G.: Learning a nonlinear embedding by preserving class neighbourhood structure. In: Artificial Intelligence and Statistics. pp. 412–419 (2007) 

- [48] Papernot, N., McDaniel, P.: Deep k-nearest neighbors: Towards confident, interpretable and robust deep learning. arXiv preprint arXiv:1803.04765 (2018) 3 

- [49] Wu, Z., Efros, A.A., Yu, S.X.: Improving generalization via scalable neighborhood component analysis. In: ECCV (2018) 3, 7, 23 

11 

Published as a conference paper at ICLR 2023 

- [50] Garcia, S., Derrac, J., Cano, J., Herrera, F.: Prototype selection for nearest neighbor classification: Taxonomy and empirical study. IEEE TPAMI **34** (3), 417–435 (2012) 3 

- [51] Yang, H.M., Zhang, X.Y., Yin, F., Liu, C.L.: Robust classification with convolutional prototype learning. In: CVPR (2018) 3 

- [52] Guerriero, S., Caputo, B., Mensink, T.: Deepncm: Deep nearest class mean classifiers. In: ICLR workshop (2018) 3, 23, 25 

- [53] Snell, J., Swersky, K., Zemel, R.S.: Prototypical networks for few-shot learning. In: NeurIPS (2017) 3 

- [54] Dong, N., Xing, E.P.: Few-shot semantic segmentation with prototype learning. In: BMVC (2018) 3 

- [55] Jetley, S., Romera-Paredes, B., Jayasumana, S., Torr, P.: Prototypical priors: From improving classification to zero-shot learning. arXiv preprint arXiv:1512.01192 (2015) 3 

- [56] Xu, W., Xian, Y., Wang, J., Schiele, B., Akata, Z.: Attribute prototype network for zero-shot learning. In: NeurIPS (2020) 3 

- [57] Rebuffi, S.A., Kolesnikov, A., Sperl, G., Lampert, C.H.: icarl: Incremental classifier and representation learning. In: CVPR (2017) 3 

- [58] Quattoni, A., Collins, M., Darrell, T.: Transfer learning for image classification with sparse prototype representations. In: CVPR (2008) 3, 5, 23 

- [59] Biehl, M., Hammer, B., Schneider, P., Villmann, T.: Metric learning for prototype-based classification. In: Innovations in Neural Information Paradigms and Applications, pp. 183– 199 (2009) 3, 5, 23 

- [60] Erhan, D., Bengio, Y., Courville, A., Vincent, P.: Visualizing higher-layer features of a deep network. University of Montreal **1341** (3), 1 (2009) 3 

- [61] Mahendran, A., Vedaldi, A.: Understanding deep image representations by inverting them. In: CVPR (2015) 

- [62] Yosinski, J., Clune, J., Nguyen, A., Fuchs, T., Lipson, H.: Understanding neural networks through deep visualization. arXiv preprint arXiv:1506.06579 (2015) 

- [63] Bach, S., Binder, A., Montavon, G., Klauschen, F., M¨uller, K.R., Samek, W.: On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one **10** (7), e0130140 (2015) 

- [64] Shrikumar, A., Greenside, P., Kundaje, A.: Learning important features through propagating activation differences. In: ICML (2017) 

- [65] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-cam: Visual explanations from deep networks via gradient-based localization. In: ICCV (2017) 3 

- [66] Ribeiro, M.T., Singh, S., Guestrin, C.: ”why should i trust you?” explaining the predictions of any classifier. In: KDD (2016) 3 

- [67] Zintgraf, L.M., Cohen, T.S., Adel, T., Welling, M.: Visualizing deep neural network decisions: Prediction difference analysis. In: ICLR (2017) 

- [68] Koh, P.W., Liang, P.: Understanding black-box predictions via influence functions. In: ICML (2017) 

- [69] Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: NeurIPS (2017) 3 

- [70] Laugel, T., Lesot, M.J., Marsala, C., Renard, X., Detyniecki, M.: The dangers of post-hoc interpretability: Unjustified counterfactual explanations. In: IJCAI (2019) 3 

- [71] Arrieta, A.B., D´ıaz-Rodr´ıguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garc´ıa, S., Gil-L´opez, S., Molina, D., Benjamins, R., et al.: Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai. Information Fusion **58** , 82–115 (2020) 3 

- [72] Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., Su, J.K.: This looks like that: deep learning for interpretable image recognition. In: NeurIPS (2019) 3 

12 

Published as a conference paper at ICLR 2023 

- [73] Alvarez Melis, D., Jaakkola, T.: Towards robust interpretability with self-explaining neural networks. In: NeurIPS (2018) 3 

- [74] Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al.: Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In: ICML (2018) 3 

- [75] Wang, T.: Gaining free or low-cost interpretability with interpretable partial substitute (2019) 3 

- [76] Subramanian, A., Pruthi, D., Jhamtani, H., Berg-Kirkpatrick, T., Hovy, E.: Spine: Sparse interpretable neural embeddings. In: AAAI (2018) 3 

- [77] Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., Abbeel, P.: Infogan: Interpretable representation learning by information maximizing generative adversarial nets. In: NeurIPS (2016) 3 

- [78] You, S., Ding, D., Canini, K., Pfeifer, J., Gupta, M.: Deep lattice networks and partial monotonic functions. In: NeurIPS (2017) 3 

- [79] Ming, Y., Xu, P., Qu, H., Ren, L.: Interpretable and steerable sequence learning via prototypes. In: KDD (2019) 3 

- [80] Arık, S.O., Pfister, T.: Protoattend: Attention-based prototypical learning. Journal of Machine Learning Research **21** , 1–35 (2020) 

- [81] Nauta, M., van Bree, R., Seifert, C.: Neural prototype trees for interpretable fine-grained image recognition. In: CVPR (2021) 3 

- [82] Pang, T., Xu, K., Dong, Y., Du, C., Chen, N., Zhu, J.: Rethinking softmax cross-entropy loss for adversarial robustness. In: ICLR (2020) 4 

- [83] Zhang, X., Zhao, R., Qiao, Y., Li, H.: Rbf-softmax: Learning deep representative prototypes with radial basis function softmax. In: ECCV (2020) 4 

- [84] Rosch, E.: Cognitive representations of semantic categories. Journal of Experimental Psychology: General **104** (3), 192 (1975) 5 

- [85] Knowlton, B.J., Squire, L.R.: The learning of categories: Parallel brain systems for item memory and category knowledge. Science **262** (5140), 1747–1749 (1993) 5 

- [86] Knight, P.A.: The sinkhorn–knopp algorithm: convergence and applications. SIAM Journal on Matrix Analysis and Applications **30** (1), 261–275 (2008) 5 

- [87] Hadsell, R., Chopra, S., LeCun, Y.: Dimensionality reduction by learning an invariant mapping. In: CVPR (2006) 5, 28 

- [88] Sohn, K.: Improved deep metric learning with multi-class n-pair loss objective. In: NeurIPS (2016) 28 

- [89] Kaya, M., Bilge, H.S¸.: Deep metric learning: A survey. Symmetry **11** (9), 1066 (2019) 28 

- [90] Boudiaf, M., Rony, J., Ziko, I.M., Granger, E., Pedersoli, M., Piantanida, P., Ayed, I.B.: A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses. In: ECCV (2020) 

- [91] Sun, Y., Cheng, C., Zhang, Y., Zhang, C., Zheng, L., Wang, Z., Wei, Y.: Circle loss: A unified perspective of pair similarity optimization. In: CVPR (2020) 28 

- [92] Suzuki, R., Fujita, S., Sakai, T.: Arc loss: Softmax with additive angular margin for answer retrieval. In: Asia Information Retrieval Symposium. pp. 34–40 (2019) 

- [93] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. In: NeurIPS (2020) 5, 28 

- [94] Bautista, M.A., Sanakoyeu, A., Tikhoncheva, E., Ommer, B.: Cliquecnn: Deep unsupervised exemplar learning. In: NeurIPS (2016) 5, 28 

- [95] Xie, J., Girshick, R., Farhadi, A.: Unsupervised deep embedding for clustering analysis. In: ICML (2016) 28 

- [96] Caron, M., Bojanowski, P., Joulin, A., Douze, M.: Deep clustering for unsupervised learning of visual features. In: ECCV (2018) 28 

13 

Published as a conference paper at ICLR 2023 

- [97] Yan, X., Misra, I., Gupta, A., Ghadiyaram, D., Mahajan, D.: Clusterfit: Improving generalization of visual representations. In: CVPR (2020) 28 

- [98] Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised learning of visual features by contrasting cluster assignments. In: NeurIPS (2020) 28 

- [99] Li, J., Zhou, P., Xiong, C., Hoi, S.C.: Prototypical contrastive learning of unsupervised representations. In: ICLR (2020) 28 

- [100] Van Gansbeke, W., Vandenhende, S., Georgoulis, S., Proesmans, M., Van Gool, L.: Scan: Learning to classify images without labels. In: ECCV (2020) 28 

- [101] Yaling Tao, Kentaro Takagi, K.N.: Clustering-friendly representation learning via instance discrimination and feature decorrelation. In: ICLR (2021) 5, 28 

- [102] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer: Simple and efficient design for semantic segmentation with transformers. In: NeurIPS (2021) 7 

- [103] Cheng, B., Schwing, A.G., Kirillov, A.: Per-pixel classification is not all you need for semantic segmentation. In: NeurIPS (2021) 7 

- [104] Caesar, H., Uijlings, J., Ferrari, V.: Coco-stuff: Thing and stuff classes in context. In: CVPR (2018) 16, 19 

- [105] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: CVPR (2018) 18 

- [106] Huh, M., Agrawal, P., Efros, A.A.: What makes imagenet good for transfer learning? arXiv preprint arXiv:1608.08614 (2016) 20, 21 

- [107] Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset (2011) 21 

- [108] Wang, X., Gao, J., Long, M., Wang, J.: Self-tuning for data-efficient deep learning. In: ICML (2021) 21 

- [109] Oh Song, H., Xiang, Y., Jegelka, S., Savarese, S.: Deep metric learning via lifted structured feature embedding. In: CVPR (2016) 

- [110] Plested, J., Gedeon, T.: Deep transfer learning for image classification: a survey. arXiv preprint arXiv:2205.09904 (2022) 21 

- [111] Recht, B., Roelofs, R., Schmidt, L., Shankar, V.: Do imagenet classifiers generalize to imagenet? (2019) 22 

- [112] Wang, W., Zhou, T., Yu, F., Dai, J., Konukoglu, E., Van Gool, L.: Exploring cross-image pixel contrast for semantic segmentation. In: ICCV (2021) 23, 28 

- [113] Short, R., Fukunaga, K.: The optimal distance measure for nearest neighbor classification. IEEE Transactions on Information Theory **27** (5) (1981) 28 

- [114] Hastie, T., Tibshirani, R.: Discriminant adaptive nearest neighbor classification and regression. In: NeurIPS (1995) 28 

- [115] Bromley, J., Guyon, I., LeCun, Y., S¨ackinger, E., Shah, R.: Signature verification using a” siamese” time delay neural network. NeurIPS (1993) 28 

- [116] Schroff, F., Kalenichenko, D., Philbin, J.: Facenet: A unified embedding for face recognition and clustering. In: CVPR (2015) 28 

- [117] Chen, W., Chen, X., Zhang, J., Huang, K.: Beyond triplet loss: a deep quadruplet network for person re-identification. In: CVPR (2017) 28 

- [118] Wu, C.Y., Manmatha, R., Smola, A.J., Krahenbuhl, P.: Sampling matters in deep embedding learning. In: ICCV (2017) 28 

- [119] Taigman, Y., Yang, M., Ranzato, M., Wolf, L.: Deepface: Closing the gap to human-level performance in face verification. In: CVPR (2014) 28 

- [120] Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., Liu, W.: Cosface: Large margin cosine loss for deep face recognition. In: CVPR (2018) 

- [121] Cao, Z., Liu, D., Wang, Q., Chen, Y.: Towards unbiased label distribution learning for facial pose estimation using anisotropic spherical gaussian. In: ECCV (2022) 

14 

Published as a conference paper at ICLR 2023 

- [122] Cao, Z., Chu, Z., Liu, D., Chen, Y.: A vector-based representation to enhance head pose estimation. In: WACV (2021) 28 

- [123] Xiao, T., Li, S., Wang, B., Lin, L., Wang, X.: Joint detection and identification feature learning for person search. In: CVPR (2017) 28 

- [124] Gutmann, M., Hyv¨arinen, A.: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In: AISTATS (2010) 28 

- [125] Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018) 28 

- [126] Hjelm, R.D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., Bengio, Y.: Learning deep representations by mutual information estimation and maximization. In: ICLR (2019) 

- [127] Chen, X., Fan, H., Girshick, R., He, K.: Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297 (2020) 

- [128] Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: ICML (2020) 

- [129] He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum contrast for unsupervised visual representation learning. In: CVPR (2020) 28 

- [130] Biehl, M., Hammer, B., Villmann, T.: Distance measures for prototype based classification. In: International Workshop on Brain-Inspired Computing (2013) 28 

- [131] Ji, X., Henriques, J.F., Vedaldi, A.: Invariant information clustering for unsupervised image classification and segmentation. In: ICCV (2019) 28 

- [132] Zhan, X., Xie, J., Liu, Z., Ong, Y.S., Loy, C.C.: Online deep clustering for unsupervised representation learning. In: CVPR (2020) 28 

- [133] Yang, B., Fu, X., Sidiropoulos, N.D., Hong, M.: Towards k-means-friendly spaces: Simultaneous deep learning and clustering (2017) 28 

- [134] Wu, J., Long, K., Wang, F., Qian, C., Li, C., Lin, Z., Zha, H.: Deep comprehensive correlation mining for image clustering. In: ICCV (2019) 28 

- [135] Tschannen, M., Djolonga, J., Rubenstein, P.K., Gelly, S., Lucic, M.: On mutual information maximization for representation learning. arXiv preprint arXiv:1907.13625 (2019) 28 

- [136] Wu, Z., Xiong, Y., Yu, S.X., Lin, D.: Unsupervised feature learning via non-parametric instance discrimination. In: CVPR (2018) 28 

- [137] Tao, Y., Takagi, K., Nakata, K.: Clustering-friendly representation learning via instance discrimination and feature decorrelation. arXiv preprint arXiv:2106.00131 (2021) 

- [138] Tsai, T.W., Li, C., Zhu, J.: Mice: Mixture of contrastive experts for unsupervised image clustering. In: ICLR (2020) 28 

- [139] Liu, D., Cui, Y., Chen, Y., Zhang, J., Fan, B.: Video object detection for autonomous driving: Motion-aid feature calibration. Neurocomputing **409** (2020) 29 

- [140] Wang, Y., Liu, D., Jeon, H., Chu, Z., Matson, E.T.: End-to-end learning approach for autonomous driving: A convolutional neural network model. In: ICAART (2019) 

- [141] Liu, D., Cui, Y., Guo, X., Ding, W., Yang, B., Chen, Y.: Visual localization for autonomous driving: Mapping the accurate location in the city maze. In: ICPR (2021) 29 

- [142] Ronen, M., Finder, S.E., Freifeld, O.: Deepdpm: Deep clustering with an unknown number of clusters. In: CVPR (2022) 30 

- [143] Cheng, Z., Liang, J., Choi, H., Tao, G., Cao, Z., Liu, D., Zhang, X.: Physical attack on monocular depth estimation with optimal adversarial patches. In: ECCV (2022) 30 

- [144] Cheng, Z., Liang, J., Tao, G., Liu, D., Zhang, X.: Adversarial training of self-supervised monocular depth estimation against physical-world attacks. In: ICLR (2023) 30 

15 

Published as a conference paper at ICLR 2023 

## **SUMMARY OF THE APPENDIX** 

This appendix contains additional details for the ICLR 2023 submission, titled _Visual Recognition with Deep Nearest Centroids”_ . The appendix is organized as follows: 

- §A provides the pseudo code of DNC. 

- §B introduces more quantitative results on image classification. 

- §C gathers additional semantic segmentation results on COCO-Stuff [104] dataset. 

- §D presents the corresponding error bars on Tables 1, 2, and 3. 

- §E investigates the potential of DNC in sub-categories discovery. 

- §F reports the transferability of DNC towards other image classification task. 

- §G evaluates the performance of DNC on ImageNetv2 test sets. 

- §H compares the performance of using _k_ -means and Sinkhorn-Knopp clustering algorithms. 

- §I compares DNC with different distance (learning) based classifiers. 

- §J reports additional diagnostic experiments for further investigations on _K_ , update ratio of external memory, ImageNet capacity, feature size, and temperature parameter _ε_ in (6). 

- §K offers more detailed discussions regarding the GPU memory cost. 

- §L depicts more visual examples regarding the _ad-hoc_ explainability of DNC. 

- §M plots qualitative semantic segmentation results. 

- §N gives additional review of representative literature on metric-/distance-learning and clustering-based unsupervised representation learning. 

- §O discusses our limitations, societal impact, and directions of our future work. 

16 

Published as a conference paper at ICLR 2023 

## A PSEUDO CODE OF DNC AND CODE RELEASE 

The pseudo-code of DNC is given in Algorithm 1. To guarantee reproducibility, our code is available at https://github.com/ChengHan111/DNC. 

**Algorithm 1** Pseudo-code of DNC in a PyTorch-like style. 

# P: non-parametric sub-centroids (C x K x D) # X: feature embeddings (N x D) # C: number of classes # K: number of sub-centroids for each class # R: sinhorn-knopp iteration number # mu: momentum coefficient (Eq.8) # epsilon: hyper-parameter in the Sinkhorn-Knopp algorithm (Eq.6) def DNC(X, label) #== Model Prediction and Training Loss (Eq.7) ==# # image-to-centroid assignment (N x K x C, Eq.5) L = torch.einsum(’nd,ckd->nkc’, X, P) output = torch.amax(L, dim=1) loss = CrossEntropyLoss(output, label) #======= Sub-centroid Estimation =======# for c in range(C) init_L = L[...,c] Q = online_clustering(init_L) # assignments and embeddings for images in class c m_c = L[label == c] x_c = X[label == c, ...] # find images that are assigned to each sub-centroid # and correctly classified m_c_tile = repeat(m_c, tile=K) m_q = Q * m_c_tile # find images with label c that are correctly classified x_c_tile = repeat(m_c, tile=x_c.shape[-1]) x_c_q = x_c * x_c_tile f = torch.mm(m_q.transpose(), x_c_q) # num assignments for each sub-centroid of class c n = torch.sum(m_q, dim=0) # momentum update (Eq.8) if torch.sum(n) > 0: P_c = mu * P[c, n != 0, :] + (1-mu) * f[n != 0, :] P[c, n != 0, :] = P_c return loss def online_clustering(L, iters=3, epsilon=0.05) Q = torch.exp(L / epsilon) Q /= torch.sum(Q) for _ in range(R): # row normalization Q /= torch.sum(Q, dim=1, keepdim=True) Q /= K # column normalization Q /= torch.sum(L, dim=0, keepdim=True) Q /= N # make sure the sum of each column to be 1 Q *= N return one_hot(Q) 

17 

Published as a conference paper at ICLR 2023 

## B MORE EXPERIMENTS ON IMAGE CLASSIFICATION 

**Additional Results on CIFAR-10 [33].** CIFAR-10 dataset contains 60K (50K/10K for train/ test) 32 _×_ 32 colored images of 10 classes. Table 6 reports additional comparison results on CIFAR-10, based on ResNet-18 [2] network architecture. As seen, in terms of top-1 accuracy, our DNC is **0.94** % higher than the parametric counterpart, under the same training setting. 

Table 6: **Classification top-1 accuracy** on CIFAR10 [33] test. #Params: the number of learnable parameters (same for other tables). See §B for more details. 

|Method|Backbone||#Params|top-1|
|---|---|---|---|---|
|ResNet [2]<br>**DNC**-ResNet|ResNet18||11.17M<br>11.16M|93.55%<br>**94.49**%|



**Additional Results on CIFAR-100 [33].** Table 7 reports comparison results on CIFAR-100, based onResNet50 and ResNet101 network architectures[2]. CIFAR-100 dataset has 100 classes with 500 training images and 100 testing images per class. We can find that, DNC obtains consistently better performance, compared to the classic parametric counterpart. Specifically, DNC is **0.10** % higher on ResNet50, and **0.16** % higher on ResNet101, in terms of top-1 accuracy. 

Table 7: **Classification top-1 accuracy** on CIFAR100 [33] test, using lightweight backbone network architectures : MobileNet-V2 [105], and Swin-T [3]. See §B for more details. 

|Method<br>Backbone|#Params<br>top-1|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|23.71M<br>79.81%<br>23.50M<br>**79.91**%|
|ResNet [2] ResNet101<br>**DNC**-ResNet|42.70M<br>79.83%<br>42.49M<br>**79.99**%|



**Additional Results on ImageNet [25] using Lightweight Backbone Network Architectures.** Table 8 reports performance on ImageNet [25], using two lightweight backbone architectures: MobileNet-V2 [105], and Swin-T [3]. As can be seen, DNC, again, attributes decent performance. In particular, our DNC is **0.28** % and **0.30** % higher on MobileNet-V2 and Swin-Tiny, respectively. It is worth noticing that DNC can efficiently reduce the number of learnable parameters when the parametric classifier occupies a massive proportion in the original lightweight classification networks. Taking MobileNet-V2 as an example: our DNC reduces the number of learnable parameters from 3.50M to **2.22** M. 

Table 8: **Classification top-1 accuracy** onImageNet[25] val. See §B for more details. 

|Method<br>Backbone|#Params<br>top-1|
|---|---|
|MobileNet [105] MobileNet-V2<br>**DNC**-MobileNet|3.50M<br>71.76%<br>**2.22M**<br>**72.04**%|
|Swin [3]<br>Swin-T<br>**DNC**-Swin|28.29M<br>81.08%<br>**27.52M**<br>**81.38**%|



18 

Published as a conference paper at ICLR 2023 

## C MORE EXPERIMENTS ON SEMANTIC SEGMENTATION 

For through evaluation, we conduct extra experiments on COCO-Stuff [104], a famous semantic segmentation dataset. COCO-Stuff contains 9K/1K images for train/test of 80 object classes and 91 stuff classes. Similar to §4.2, we approach DNC on FCN[34],DeepLabV3[35],and UperNet[36], using two backbone architectures, _i_ . _e_ ., ResNet101 [2] and Swin-B [3]. All models are obtained by following the standard training settings of COCO-Stuff train, _i_ . _e_ ., crop size 512 _×_ 512, batch size 16, and 40K iterations. From Table 9 we can draw similar conclusions: in comparison with the parametric counterpart, DNC produces more precise segments and yields improved transferability. 

Table 9: **Segmentation mIoU score** on COCO-Stuff [104]. See §C for more details. 

|Method|Backbone|COCO-Stufftest|
|---|---|---|
|||mIoU|
|FCN [34]<br>**DNC**-FCN|ResNet101 [2]<br>**DNC**-ResNet101<br>ResNet101 [2]<br>**DNC**-ResNet101|32.63%<br>32.89%_↑_**0.26**<br>33.04%_↑_**0.41**<br>**33.49**%_↑_**0.86**|
|DeepLabV3 [35]<br>**DNC**-DeepLabV3|ResNet101 [2]<br>**DNC**-ResNet101<br>ResNet101 [2]<br>**DNC**-ResNet101|36.01%<br>36.28%_↑_**0.27**<br>36.51%_↑_**0.50**<br>**36.79**%_↑_**0.78**|
|UperNet [36]<br>**DNC**-UperNet|Swin-B [3]<br>**DNC**-Swin-B<br>Swin-B [3]<br>**DNC**-Swin-B|42.77%<br>42.84%_↑_**0.07**<br>43.13%_↑_**0.36**<br>**43.29**%_↑_**0.52**|



## D ERROR BARS 

In this section, we report standard deviation error bars on Tables 10, 11, and 12 for our main experiments regarding image classification (§4.1) and semantic segmentation (§4.2). The results are obtained by training the algorithm three times, with different initialization seeds. 

Table 10: **Classification top-1 accuracy** on CIFAR-10 [33] test with error bars. See §D for more details. 

|Method<br>Backbone|#Params<br>top-1|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|23.52M<br>95.55_±_(0.14)%<br>23.50M<br>**95.78**_±_ (0.12)%|
|ResNet [2] ResNet101<br>**DNC**-ResNet|42.51M<br>95.58_±_(0.13)%<br>42.49M<br>**95.82**_±_ (0.13)%|



Table 11: **Classification top-1 and top-5 accuracy** on ImageNet [25] val with error bars. See §D for more details. 

|Method<br>Backbone|#Params<br>top-1<br>top-5|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|25.56M<br>76.20_±_(0.10)%<br>93.01%<br>23.51M<br>**76.49**_±_ (0.09)%<br>**93.08**%|
|ResNet [2]<br>ResNet101<br>**DNC**-ResNet|44.55M<br>77.52_±_(0.11)%<br>93.06%<br>42.50M<br>**77.80**_±_ (0.10)%<br>**93.85**%|
|Swin [3]<br>Swin-S<br>**DNC**-Swin|49.61M<br>83.02_±_(0.14)%<br>96.29%<br>48.84M<br>**83.26**_±_ (0.13)%<br>**96.40**%|
|Swin [3]<br>Swin-B<br>**DNC**-Swin|87.77M<br>83.36_±_(0.12)%<br>96.44%<br>86.75M<br>**83.68**_±_ (0.12)%<br>**97.02**%|



19 

Published as a conference paper at ICLR 2023 

Table 12: **Segmentation mIoU score** on ADE20K [37] val and Cityscapes [26] val with error bars. See §D for more details. 

|Method|ImageNet<br>Backbone|ADE20K<br>Cityscapes<br>#Params|
|---|---|---|
||top-1acc.|mIoU<br>mIoU|
|FCN [34]<br>**DNC**-FCN|ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%<br>ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%|68.6M<br>39.9_±_(0.11)%<br>75.6_±_(0.13)%<br>68.6M<br>40.4_±_(0.11)%<br>76.3_±_(0.12)%<br>68.5M<br>41.1_±_(0.10)%<br>76.7_±_(0.11)%<br>68.5M<br>**42.3**_±_ (0.10)%<br>**77.5**_±_ (0.11)%|
|DeepLabV3[35]<br>**DNC**-DeepLabV3|ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%<br>ResNet101 [2]<br>77.52%<br>**DNC**-ResNet101<br>77.80%|62.7M<br>44.1_±_(0.13)%<br>78.1_±_(0.12)%<br>62.7M<br>44.6_±_(0.13)%<br>78.7_±_(0.13)%<br>62.6M<br>45.0_±_(0.10)%<br>79.1_±_(0.13)%<br>62.6M<br>**45.7**_±_ (0.09)%<br>**79.8**_±_ (0.12)%|
|UperNet [36]<br>**DNC**-UperNet|Swin-B [3]<br>83.36%<br>**DNC**-Swin-B<br>83.68%<br>Swin-B [3]<br>83.36%<br>**DNC**-Swin-B<br>83.68%|90.6M<br>48.0_±_(0.11)%<br>79.8_±_(0.13)%<br>90.6M<br>48.4_±_(0.10) %<br>80.1_±_(0.12)%<br>90.5M<br>48.6_±_(0.09) %<br>80.5_±_(0.10)%<br>90.5M<br>**50.5**_±_ (0.09)%<br>**80.9**_±_ (0.10)%|



## E EXPERIMENTS ON SUB-CATEGORIES DISCOVERY 

Through unsupervised, within-class clustering, our DNC represents each class as a set of automatically discovered class sub-centroids ( _i_ . _e_ ., cluster center). This allows DNC to better describe the underlying, multimodal data structure and robusty depict for rich intra-class variance. In other words, our DNC can effectively capture sub-class patterns, which is conducive to algorithmic performance. Such a capacity of sub-patter mining is also considered crucial for good transferable features – representations learnt on coarse classes are capable of fine-grained recognition [106]. 

In order to quantify the ability of DNC for automatic sub-category discovery, we follow the experimental setup posed by [106] – learning the feature embedding using coarse-grained object labels, and evaluating the learned feature using fine-grained object labels. This evaluation strategy allows us to assess the feature learning performance regarding how well the deep model can discover variations within each category. A conjecture is that deep networks that perform well on this test have an exceptional capacity to identify and mine sub-class patterns during training, which the proposed DNC seeks to rigorously establish. 

In particular, the network is first trained on coarse-grained labels with the baseline parametric softmax and with our non-parametric DNC using the same network architecture. After training on coarse classes, we use the top-1 nearest neighbor accuracy in the final feature space to measure the accuracy of identifying fine-grained classes. The classification performance evaluated in such setting is referred as **induction accuracy** as in [106]. Next we provide our experimental results on CIFAR100 [33] and ImageNet [25], respectively. 

**Performance of Sub-category Discovery on CIFAR100.** CIFAR100 includes both fine-grained annotations in 100 classes and coarse-grained annotations in 20 classes. We examine sub-category discovery by transferring representation learned from 20 classes to 100 classes. As shown in Table 13, DNC consistently outperforms the parametric counterpart: DNC increases 0.12%, in terms of the standard top-1 accuracy, on both ResNet50 and ResNet101 architectures. Nevertheless, when transferred to CIFAR100 ( _i_ . _e_ ., 100 classes) using _k_ -NN, a significant loss occurs on the baseline: 53.22% and 54.31% top-1 acc. on ResNet50 and ResNet101, respectively. Our features, on the other hand, provide 14.24% and 13.82% improvement over the baseline, achieving 67.46% and 68.13% top-1 acc. on 100 classes on ResNet50 and ResNet101, respectively. In addition, in comparison the parametric model, our approach results in only a smaller drop in transfer performance, _i_ . _e_ ., **-18.87** _vs_ -32.99 on ResNet50, and **-18.47** _vs_ -32.17 on ResNet101. 

20 

Published as a conference paper at ICLR 2023 

Table 13: **Top-1 induction accuracy** on CIFAR-100 [33] test using CIFAR-20 pre-trained models. Numbers reported with _k_ - nearest neighbor classifiers. See §E for more details. 

|Method<br>Backbone|20 classes<br>100 classes|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|86.21%<br>53.22%<br>**86.33**%<br>**67.46**%|
|ResNet [2] ResNet101<br>**DNC**-ResNet|86.48%<br>54.31%<br>**86.60**%<br>**68.13**%|



**Performance of Sub-category Discovery on ImageNet.** Table 14 provides experimental results of sub-category discovery on ImageNet val. As in [106], 127 coarse ImageNet categories are obtained by top-down clustering of 1K ImageNet categories on WordNet tree. Training on the 127 coarse classes, DNC improves the performance of baseline by 0.10% and 0.03%, achieving 84.39% and 85.91% on ResNet50 and ResNet101, respectively. When transferring to the 1K ImageNet classes using _k_ -NN, our features provide huge improvements, _i_ . _e_ ., 8.98% and 9.29%, over the baseline. 

Table 14: **Top-1 induction accuracy** on ImageNet [25] val using ImageNet-127 pre-trained models. Numbers reported with _k_ -nearest neighbor classifiers. See §E for more details. 

|Method<br>Backbone|127 classes<br>1000 classes|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|84.29%<br>43.23%<br>84.39%<br>**52.21**%|
|ResNet [2]<br>ResNet101<br>**DNC**-ResNet|85.88%<br>45.31%<br>85.91%<br>**54.60**%|



The promising transfer results on CIFAR100 and ImageNet serve as strong evidence to suggest that our DNC is capable of automatically discovering meaningful sub-class patterns – latent visual structures that are not explicitly presented in the supervisory signal, and hence handle intra-class variance and boost visual recognition. 

## F TRANSFERABILITY TOWARDS OTHER IMAGE CLASSIFICATION TASK 

In addition to conducting the coarse-to-fine transfer learning experiment (§E), we further evaluate the transfer learning performance by applying ImageNet-trained weight to Caltech-UCSD Birds200-2011 (CUB-200-2011) dataset [107], following [108–110]. 

Specifically, CUB-200-2011 dataset comprises 11,788 bird photos arranged into 200 categories, with 5,994 for training and 5,794 for testing. All the models use ResNet50 architecture [2], and are trained for 100 epochs. SGD optimizer is adopted, where the learning rate is initialized as 0.01 and organized following a polynomial annealing policy. Standard data augmentation techniques are used, including flipping, cropping and normalizing. Experimental results are reported in Table 15. As seen, DNC is + **0.73** % and + **0.39** % higher in Top-1 and Top-5 acc., respectively. This clearly verifies that DNC owns better transferability. 

Table 15: **Classification top-1 and top-5 accuracy** on CUB-2002011 test [107]. See §F for more details. 

|Model(ImageNet-trained)<br>Backbone|top-1<br>top-5|
|---|---|
|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|84.48%<br>96.31%<br>**85.21**%<br>**96.70**%|



21 

Published as a conference paper at ICLR 2023 

For parametric softmax classifier, both the feature network and softmax layer are fully learnable parameters. That means, for a new task, the softmax classifier has to both finetune the feature network and train a new softmax layer. However, for DNC, the feature network is the only learnable part. The class centers are not freely learnable parameters; they are directly computed from training data on the feature space. For a new task, DNC just needs to fine-tune the only learnable part – the feature network; the class centers are still directly computed from the training data according to the clustering assignments, without end-to-end training. Hence DNC owns better transferability during network fine-tuning. 

## G EVALUATION ON IMAGENETV2 TEST SETS 

We evaluate DNC-ResNet50 on ImageNetv2 [111] test sets, _i_ . _e_ ., “Matched Frequency”, “Threshold0.7” and “Top Images”. Each test set contains 10 images for each ImageNet class, collected from MTurk. In particular, each MTurk worker is assigned with a certain classes. Then each worker is asked to select images belonging to his/her target class, from several candidate images sampled from a large image pool as well as ImageNet validation set. The output is a _selection frequency_ for each image, _i_ . _e_ ., the fraction of MTurk workers selected the image in a task for its target class. Then three test sets are developed according to different principles defined on the selection frequency. For “Matched Frequency”, [111] first approximated the selection frequency distribution for each class using those “re-annotated” ImageNet validation images. According to these class-specific distributions, ten test images are sampled from the candidate pool for each class. For “Threshold0.7”, [111] sampled ten images from each class with selection frequency at least 0.7. For “Top Images”, [111] selected the ten images with the highest selection frequency for each class. The results on these three test sets are shown in Table 16. As seen, our DNC exceeds the parametric softmax based ResNet50 by + **0.52-0.89** % top-1 and + **0.26-0.47** % top-5 acc.. 

Table 16: **Classification top-1 and top-5 accuracy** on ImageNetv2 test sets [111]. See §G for more details. 

|testset|Method<br>Backbone|top-1<br>top-5|
|---|---|---|
|MatchedFrequency|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|63.30%<br>84.70%<br>**63.96**%<br>**85.17**%|
|Threshold0.7|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|72.70%<br>92.00%<br>**73.59**%<br>**92.26**%|
|TopImages|ResNet [2]<br>ResNet50<br>**DNC**-ResNet|78.10%<br>94.70%<br>**78.62**%<br>**94.96**%|



## H SINKHORN-KNOPP _vs k_ -MEANS CLUSTERING 

To further probe the impact of Sinkhorn-Knopp based clustering [23], we further report the performance of DNC by using the classic k-means clustering algorithm, on CIFAR-10 and CIFAR100 datasets [33]. From Table 17 We can find that DNC with Sinkhorn-Knopp performs much better and is much more training-efficient. 

Table 17: **Classification top-1 and training time** on CIFAR-10 and CIFAR100 [33] test sets. See §H for more details. 

|Dataset|Method<br>Backbone|top-1<br>Trainingtime(hours)|
|---|---|---|
|CIFAR10 [33]|**DNC**-_k_-means<br>ResNet50<br>**DNC**-Sinkhorn|93.88%<br>4.5<br>**95.78**%<br>**2.5**|
|CIFAR100 [33]|**DNC**-_k_-means<br>ResNet50<br>**DNC**-Sinkhorn|77.86%<br>16.3<br>**79.91**%<br>**3.7**|



22 

Published as a conference paper at ICLR 2023 

## I COMPARISON WITH DISTANCE (LEARNING) BASED CLASSIFIERS 

We next compare our DNC with two metric based image classifiers [49, 52] and one distance learning based segmentation model [112]. 

We first compare DNC with DeepNCM [52]. DeepNCM conducts similarity-based classification using class means. As DeepNCM only describes its training procedures for CIFAR-10 and CIFAR100 [33], we make the comparison on these two datasets to ensure fairness. From Table 18 we can find that, DNC significantly outperforms DeepNCM by + **2.11** % on CIFAR-10 and + **7.16** % on CIFAR-100, respectively. 

DeepNCM simply abstracts each class into one single class mean, failing to capture complex withinclass data distribution. In contrast, DNC considers _K_ sub-centers per class. Note that this is not just increasing the number of class representatives. This requires accurately discovering the underlying data structure. DNC therefore jointly conducts automated online clustering (for mining sub-class patterns) and supervised representation learning (for cluster center based classification). Finding meaningful class representatives is extremely challenging and crucial for Nearest Centroids. As the experiment in §H and Table 17 revealed, simply adopting classic _k_ -means causes huge performance drop and significant training speed delay. Moreover, DeepNCM computes class mean on each batch, which makes poor approximation of the real class center. In contrast, DNC adopts the external memory for more accurate data densities modeling – DNC makes clustering over the large memory of numerous training samples, instead of the relatively small batch. Thus DNC better captures complex within-class variants and addresses two key properties of prototypical exemplars: sparsity and expressivity [58, 59], and eventually gains much more promising results. 

Table 18: **Classification top-1** on CIFAR-10 and CIFAR100 [33] test. See §I for more details. 

|Dataset|Method<br>Backbone|top-1|
|---|---|---|
|CIFAR-10|DeepNCM [52]<br>ResNet50<br>**DNC**|93.67%<br>**95.78**%|
|CIFAR-100|DeepNCM [52]<br>ResNet50<br>**DNC**|72.75%<br>**79.91**%|



We also compare DNC with DeepNCA [49]. DeepNCA is a deep _k_ -NN classifier, which conducts similarity-based classification based on top- _k_ nearest training samples. As [49] only reports results on ImageNet [25], we make the comparison on ImageNet to ensure fairness. Note that 130 training epochs are used, as in [49]. From Table 19 we can observe that, DNC outperforms DeepNCA. As we discussed in §2 and §4.1, DeepNCA poses huge storage demand, _i_ . _e_ ., retaining the whole ImageNet training set ( _i_ . _e_ ., 1.2M images) to perform the _k_ -NN decision rule, and suffers from very low efficiency, caused by extensive comparisons between each test sample and ALL the training images. These limitations prevent the adoption of [49] in real application scenarios. In contrast, DNC only relies on a small set of class representatives ( _i_ . _e_ ., four sub-centroids per class) for decision-making and causes no extra computation budget during network deployment. 

Table 19: **Classification top-1** on ImageNet [25] val. See §I for more details. 

|Method<br>Backbone|top-1|
|---|---|
|DeepNCA [49]<br>ResNet50<br>**DNC**|76.57%<br>**76.64**%|



Finally, we compare DNC with ContrastiveSeg [112] on Cityscapes [26] val, using ResNet101 [2] backbone and DeepLabV3 [35] segmentation architecture. ContrastiveSeg applies contrastive learning to better shape the feature embedding space, so as to improve semantic segmentation performance. However, ContrastiveSeg still relies on the softmax classifier; it is essentially a distance 

23 

Published as a conference paper at ICLR 2023 

learning boosted parametric classifier. From Table 19 we can observe that, DNC greatly surpasses ContrastiveSeg by 0.6% mIoU. 

Table 20: **Segmentation mIoU score** on Cityscapes [26] val. See §I for more details. 

|Method<br>Backbone|mIoU|
|---|---|
|ContrastiveSeg-DeepLabV3<br>ResNet101<br>**DNC**-DeepLabV3|79.1%<br>**79.8%**|



These three experiments solidly demonstrate our effectiveness on both image classification and segmentation tasks, even compared with other distance (learning) based counterparts. 

## J ADDITIONAL DIAGNOSTIC EXPERIMENT 

**Output Dimensionality.** As stated in §4, the final output dimensionality of our DNC is set as many as the one of the last layer of the parametric counterpart, for the sake of fair comparison. However, owing to the distance-/similarity-based nature, DNC has the flexibility to handle any output dimensionality. In Table 21, we further study the influence of the output dimensionality of DNC. As seen, when setting the final output dimensionality as 1280, we can achieve **76.61%** top-1 acc., which is higher than the initial 2048 dimension configuration, _i_ . _e_ ., 76.49%. We attribute the reason for the better balance between memory capacity and feature dimensionality with the limitation of hardware computational budget – when reducing the final output dimensionality, the expressibility of the final feature is weakened but more image features can be stored in the external memory for more accurate sub-centroid estimation. 

Table 21: **Ablative experiments regarding the final output dimensionality** on ImageNet [25] val. See §J for more details. 

|Output|ImageNet|
|---|---|
|dimensionality|top-1<br>top-5|
|640<br>1024<br>**1280**<br>2048|76.23%<br>92.83%<br>76.28%<br>92.90%<br>**76.61**%<br>**93.12**%<br>76.49%<br>93.08%|



**Temperature Parameter** _ε_ **in (6).** Parameter _ε_ in (6) trades off convergence speed with closeness to the original transport problem [23, 24]. In Table 22, we further study the impact of _ε_ on ImageNet [25] val. 

Table 22: **Ablative experiments regarding temperature parameter** _ε_ **in (6)** on ImageNet [25] val. See §J for more details. 

||ImageNet|
|---|---|
|_ϵ_|top-1<br>top-5|
|0.01<br>**0.05**<br>0.1|76.34%<br>92.97%<br>**76.49**%<br>**93.08**%<br>76.40%<br>93.02%|



**Number of Centroids** _K_ **.** As shown in Table 23, we set _K_ with different values based on the number of training samples of each class. Specifically, ImageNet contains between 732 and 1300 training images (#images) per class. Then, _K_ = 1 is assigned to the class having between 732 and 874 training samples, _K_ = 2 to the class having between 875 and 1016 samples, _K_ = 3 to the class 

24 

Published as a conference paper at ICLR 2023 

having between 1017 and 1158 samples, and _K_ = 4 to the class having between 1159 and 1300 samples. We can find that we gain slightly better performance, +0.06% higher in top-1 acc. when compared with fixing _K_ = 4 for all the classes. 

Table 23: **Ablative experiments with varying** _K_ **on different classes** on ImageNet [25] val. See §J for more details. 

|_K_|ImageNet|
|---|---|
|range|top-1<br>top-5|
|unique value 4<br>varyingbetween 1 and 4|76.49%<br>93.08%<br>**76.55**%<br>**93.10**%|



## K MEMORY COST 

In our experiment, we only adopt external memory for ImageNet classification. Below we provide more discussion regarding this point. Semantic segmentation is a pixel-wise classification task, where each training image provides numerous pixel samples for each class. For ImageNet classification, however, each training image is only assigned to one single class. Moreover, ImageNet has 1K classes, while in general semantic segmentation only has dozens of classes. Therefore, for a training mini-batch of, for example, 256 images, every class in segmentation usually has many training pixel samples in each mini-batch; this allows us to use a large _K_ for clustering. However, under the same setting, for each mini-batch, there must have many ImageNet classes that do not have corresponding image samples – we have 1000 classes but each mini-batch only has 256 training images. This is why we need to build an external memory during ImageNet classification. This is also why applying Nearest Centroids for batch-wise ImageNet classification training is extremely challenging; [52] cannot handle ImageNet classification, as it only computes class means in a batch-wise manner. 

Table 24a and 24b provide statistics of GPU memory cost (per GPU usage) with respect to the number of class centroids _K_ and external memory size respectively. The statistics are gathered during the training of DNC-ResNet50 on ImageNet, using eight V100 GPUs. In our experiment, we set the size of the external memory as 256,000 image samples ( _i_ . _e_ ., 1000 batches) and _K_ = 4 for DNC-ResNet50 ( _i_ . _e_ ., a total of 4000 class sub-centroids on ImageNet). More specifically, for a memory with #batch=1000, it stores 8 #gpu _×_ 32 #batch size _×_ 1000 #batch = 256000 #image examples. In comparison, for segmentation, when we set _K_ = 10 ( _i_ . _e_ ., a total of 1500 class subcentroids for ADE20K), we have 65,536 pixel training samples in each mini-batch of 16 training images, without using memory. 

Table 24: **Statistics of GPU memory cost** with respect to the number of class centroids _K_ and external memory size, where 1000 batches = 256000 image samples. See §K for more details. 

|_K_ (Fixed memory size:|GPU memory||memory size|GPU memory|
|---|---|---|---|---|
|1000 batches)|cost(GBper GPU)||(Fixed_K_ = 4)|cost(GBper GPU)|
|1<br>2<br>3<br>4|16.07<br>19.04<br>22.03<br>26.31||400 batches<br>600 batches<br>800 batches<br>1000 batches|11.88<br>16.35<br>21.65<br>26.35|



(a) GPU memory cost _w.r.t_ number of centroids _K_ (b) GPU memory cost _w.r.t_ external memory size L ADDITIONAL STUDY OF AD-HOC EXPLAINABILITY 

**Interpretable Class Sub-centroids.** In Fig. 5, we show more examples of sub-centroid images for eight ImageNet classes. These representative images are automatically discovered by DNC, and can be intuitively viewed by users. As seen, the class sub-centroid images are able to capture diverse characteristics of their classes, in the aspects of appearance, viewpoints, scales, illuminations, _etc_ . 

**Interpret Prediction Based on (Dis)similarity to Sub-centroid Images.** Fig. 6 provides more results regarding the (dis)similarity-based interpretation of DNC prediction. As seen, based on the similarity of test images to the class sub-centroid images, users can clearly understand the decision making mode and make verification. DNC’s compelling explainability enables it to establish trustworthiness with humans and empowers its potential in high stake applications. 

25 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0026-01.png)


Figure 5: Sub-centroid images for eight randomly chosen classes from ImageNet [25]. See §L for more details. 

26 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0027-01.png)


**----- Start of picture text -----**<br>
…<br>17.4% 0.005% 0.0002%<br>81.9%<br>DNC's prediction:  siamese cat<br>Groundtruth:  siamese cat<br>…<br>6.8% 0.4% 0.03%<br>91.3%<br>DNC's prediction:   station wagon<br>Groundtruth:   station wagon<br>…<br>13.0% 5.3% 0.6%<br>81.2%<br>DNC's prediction:  grey fox<br>Groundtruth:  grey fox<br>…<br>47.2% 1.3% 0.0002%<br>51.0%<br>DNC's prediction: flamingo<br>Groundtruth:  spoonbill<br>…<br>48.2% 0.06% 0.03%<br>51.3%<br>DNC's prediction:  Ox<br>Groundtruth:  water buffalo<br>…<br>19.7% 6.9% 2.3%<br>69.8%<br>DNC's prediction:   tape player<br>Groundtruth:  cassette player<br>cat<br>siamese cat Persian cat cougar Egyptian<br>grille race car minivan<br>station wagon<br>grey fox kit fox red fox dhole<br>flamingo spoonbill pelican black stork<br>Ox<br>water  buffalo bison Indian  elephant<br>cassette  player cassette<br>tape player CD player<br>**----- End of picture text -----**<br>


Figure 6: More examples on DNC interpreting its predictions based on its computed similarity to class sub-centroid images. For each test image, we plot the normalized similarities for the corresponding closest sub-centroids from the top-4 scoring classes. See §L for more details. 

27 

Published as a conference paper at ICLR 2023 

## M QUALITATIVE RESULTS ON SEMANTIC SEGMENTATION 

Fig. 7 and Fig. 8 illustrate a few representative visual examples of semantic segmentation results on ADE20K [37] and Cityscapes [26], respectively. In comparison with the parametric counterpart, our approach produces more precise segments in different challenging scenes (for example, where objects with drastic photometric or geometric appearances). For instance, UperNet [36]-Swin-B [3] confuses on neighbouring objects with similar colors ( _e.g._ , desk and chair) and leaves a large false-negative regions (see the first image of Fig. 7); it also has difficulties in segmenting small scale objects ( _e.g._ , motorbike semantic parsing). Among these examples, DNC consistently demonstrates supreme performance. Essentially, we argue that the proposed DNC has a stronger ability to supervises the pixel embedding space via anchoring sub-centroids directly, leading to a better predictions on segmenting such hard cases. 

## N ADDITIONAL LITERATURE REVIEW 

This section gives additional review of representative literature on metric-/distance-learning and clustering-based unsupervised representation learning. 

**Metric Learning.** The goal of metric learning (also a.k.a distance learning) is to learn a distance metric/embedding which brings together similar samples and pushes away dissimilar ones. Metric learning has a long history, dating back to some early work for more than few decades ago [113, 114]. In particular, diverse metric learning objective functions, such as contrastive loss [87, 93, 115], triplet loss [116], quadruplet loss [117], and _n_ -pair loss [88], were proposed to measure similarity in the feature space for representation learning, and showed significant benefit in a wide range of applications, such as image retrieval [118], face recognition [116, 119–122], and person re-identification [123], to name a few representative ones. Recently, metric learning gained astonishing success in learning transferable deep representations from massive unlabeled data [89]. A family of _instancebased_ approaches used the contrastive loss [124, 125] to explicitly compare pairs of image representations [125–129]. Another group of methods adopted a _clustering-based_ strategy; they learn unsupervised representations by discriminating between groups of images without expensive pairwise comparison between image instances [24, 94–101]. More recently, there are some efforts that revisit the idea of metric learning in supervised learning setting [91–93, 112]. 

As distance-/similarity-based classifiers rely on the similarity between samples and class representatives for classification, the fields of metric learning and distance-based classification are naturally related and the selection of a proper distance measure impacts the success of distance-based classifiers [130]. Historically, metric learning and class center discovery are two critical research topics in the field of distance-based classification. As a nonparametric, distance-based classifier, DNC can be viewed as a learnable metric function, which is trained to compare data samples under the guidance of the corresponding semantic labels. Although current distance learning based algorithms also optimize the feature space by comparing data samples, they need parametric softmax for classification. Their trained models are still black-box parametric classifiers without any interpretability. In sharp contrast, DNC directly assigns an observation to the class of the closet centroids, without using parametric softmax. Moreover, its distance-based classification decisionmaking mode allows DNC to effortless adopt existing metric learning techniques (and the way of its current training can be already viewed as performing metric learning). 

**Clustering-based Self-supervised Representation Learning.** There is a recent trend to bind selfsupervised representation learning with clustering. Basically, clustering-based self-supervised representation learning is more efficient for large-scale training data and more tolerant of the similarity (semantic structure) among data samples, compared with the instance-level counterpart. More specifically, early approaches [24, 94, 96, 97, 100, 131, 132] learn representations of image samples and cluster assignments in an _alternative_ manner, _i_ . _e_ ., group features into clusters to derive pseudo supervisory signal and subsequently employ it for supervising representation learning. In very recent, numerous efforts have been devoted to _simultaneous_ clustering and representation learning based on, _e_ . _g_ ., data reconstruction [95, 133], mutual information maximization [131, 134, 135], or contrastive instance discrimination [98, 99, 101, 136–138]. 

Our work is also related to these clustering based unsupervised representation learning methods, especially the ones [24, 98] resorting to the fast Sinkhorn-Knopp algorithm [23] for robust clus- 

28 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-05.png)


**----- Start of picture text -----**<br>
Ours<br>Swin-B<br>Swin-B<br>Ours<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0029-19.png)


Figure 7: **Qualitative semantic segmentation results** of UperNet [36]-Swin-B [3] and DNC on ADE20K [37] val. Red and green bounding boxes represent the same zoom-in area on UperNet [36]-Swin-B [3] and DNC, respectively. See §M for more details. 

tering. They aim to _learn transferable representation_ from massive _unlabeled_ data. Although also involving a similar clustering procedure for automatic sub-pattern mining, DNC targets at building a strong _similarity-based classification network_ in the standard _supervised learning_ setting. In DNC, the automatically discovered class sub-centroids are informative class representatives, which explicitly capture latent data structure of each class, and serve as classification evidence with clear physical meaning. The whole training procedure is a hybrid of class-wise online clustering (for unsupervised sub-class discovery) and sub-centroid based classification (for supervised representation learning). This well addresses the nature of Nearest Centroids and brings novel insights into the visual recognition task itself. 

## O LIMITATION AND FUTURE WORK 

**Limitation.** _O_ �( _[n] ϵ_[3][2][)][which] One[would] limitation[reduce][the] of our[training] approach[efficiency.] is that[Though] the Sinkhorn-Knopp[in][practice,][we] algorithm[find][3][sinkhorn] runs in[loops] time per training iteration is sufficient enough for model representation, bringing a minor computational overhead ( _i.e._ , _∼_ 5% training delay on ImageNet). This also indicates possible directions for our future research. 

**Social Impact.** This work introduces DNC possessing the nature of nested simplicity, intuitive decision-making mechanism and even _ad-hoc_ explainability. On positive side, the approach advances model accuracy and is valuable in safety-sensitive applications by showing the advanced robustness on sub-categories discovery, _e.g._ , quality analytics, autonomous driving [139–141], _etc._ . For potential negative social impact, our DNC struggles in handling out-of-distribution data, which 

29 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-05.png)


**----- Start of picture text -----**<br>
Ours<br>Swin-B<br>Ours<br>Swin-B<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids_images/iclr_2023_Visual_Recognition_with_Deep_Nearest_Centroids.pdf-0030-19.png)


Figure 8: **Qualitative semantic segmentation results** of UperNet [36]-Swin-B [3] and DNC on Cityscapes [26] val. Red and green bounding boxes represent the same zoom-in area on UperNet [36]-Swin-B [3] and DNC, respectively. See §M for more details. is a common limitation of all the discriminative classifiers. Hence its utility in open-world scenarios should be further examined. 

**Future Work.** Despite DNC’s systemic simplicity and efficacy, it also comes with new challenges and unveils some intriguing questions. For example, incorporating more powerful, time-efficient online clustering algorithms into DNC might improve training speed and test accuracy. Also, the number of class centroids _K_ currently is set to a fixed value for all classes, which may not be optimal given that intra-class variability varies across classes. Our experiments in §J and Table 23 also suggest that simply varying _K_ with the number of training samples of the class can boost performance. Thus adopting the clustering algorithms that do not require a predefined and fixed number of clusters [142] may allow DNC to automatically determine _K_ for different classes, which eventually benefit performance. In addition, instead of only considering first-order statistics, DNC could be enhanced by second-order statistics, which contain more useful information, but must contend with the computational overhead they impose. Another essential future direction deserving of further investigation is the in-depth analysis of the intrinsic properties of DNC, such as its robustness against perturbation, adversarial attack [143, 144], and out-of-distribution data, with the comparison of the softmax based counterpart. This endeavor would help us to better understand the nature of parametric and nonparametric classifiers and reveals directions for further improvement. Furthermore, we will explore the possibility of unifying close-set and open-world visual recognition within our framework. Finally, considering the similarity-/distance-based nature of DNC, the incorporation of metric learning based training objectives is also another promising direction for further boosting the performance. Given the vast number of technique breakthroughs in recent years, we expect a flurry of innovation towards these promising directions. Overall, we believe the results presented in this paper warrant further exploration. 

30 

