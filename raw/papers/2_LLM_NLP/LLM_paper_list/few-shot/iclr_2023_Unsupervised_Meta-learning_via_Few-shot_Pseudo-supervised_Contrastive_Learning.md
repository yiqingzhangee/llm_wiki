Published as a conference paper at ICLR 2023 

## - UNSUPERVISED META-LEARNING VIA FEW SHOT - PSEUDO SUPERVISED CONTRASTIVE LEARNING 

**Huiwon Jang** `[A]` _[∗]_ **Hankook Lee** `[B]` _[∗†]_ **Jinwoo Shin** `[A] A` Korea Advanced Institute of Science and Technology (KAIST) `B` LG AI Research _{_ `huiwoen0516` _,_ `jinwoos` _}_ @ `kaist` _._ `ac` _._ `kr hankook` _._ `lee` @ `lgresearch` _._ `ai` 

## ABSTRACT 

_Unsupervised meta-learning_ aims to learn generalizable knowledge across a distribution of tasks constructed from unlabeled data. Here, the main challenge is how to construct diverse tasks for meta-learning without label information; recent works have proposed to create, e.g., pseudo-labeling via pretrained representations or creating synthetic samples via generative models. However, such a task construction strategy is fundamentally limited due to heavy reliance on the immutable pseudo-labels during meta-learning and the quality of the representations or the generated samples. To overcome the limitations, we propose a simple yet effective unsupervised meta-learning framework, coined _**P** seudo-_ _**s** upervised_ _**Co** ntrast (_ _**PsCo** )_ , for few-shot classification. We are inspired by the recent self-supervised learning literature; PsCo utilizes a _momentum network_ and a _queue_ of previous batches to improve pseudo-labeling and construct diverse tasks in a progressive manner. Our extensive experiments demonstrate that PsCo outperforms existing unsupervised meta-learning methods under various in-domain and cross-domain few-shot classification benchmarks. We also validate that PsCo is easily scalable to a large-scale benchmark, while recent prior-art meta-schemes are not. 

## 1 INTRODUCTION 

_Learning to learn_ (Thrun & Pratt, 1998), also known as _meta-learning_ , aims to learn general knowledge about how to solve unseen, yet relevant tasks from prior experiences solving diverse tasks. In recent years, the concept of meta-learning has found various applications, e.g., few-shot classification (Snell et al., 2017; Finn et al., 2017), reinforcement learning (Duan et al., 2017; Houthooft et al., 2018; Alet et al., 2020), hyperparameter optimization (Franceschi et al., 2018), and so on. Among them, _few-shot classification_ is arguably the most popular one, whose goal is to learn some knowledge to classify test samples of unseen classes during (meta-)training with few labeled samples. The common approach is to construct a distribution of few-shot classification (i.e., _N_ -way _K_ -shot) tasks and optimize a model to generalize across tasks (sampled from the distribution) so that it can rapidly adapt to new tasks. This approach has shown remarkable performance in various few-shot classification tasks but suffers from limited scalability as the task construction phase typically requires a large number of human-annotated labels. 

To mitigate the issue, there have been several recent attempts to apply meta-learning to _unlabeled data_ , i.e., _unsupervised meta-learning (UML)_ (Hsu et al., 2019; Khodadadeh et al., 2019; 2021; Lee et al., 2021; Kong et al., 2021). To perform meta-learning without labels, the authors have suggested various ways to construct synthetic tasks. For example, pioneering works (Hsu et al., 2019; Khodadadeh et al., 2019) assigned pseudo-labels via data augmentations or clustering based on pretrained representations. In contrast, recent approaches (Khodadadeh et al., 2021; Lee et al., 2021; Kong et al., 2021) utilized generative models to generate synthetic (in-class) samples or learn unknown labels via categorical latent variables. They have achieved moderate performance in few-shot learning benchmarks, but are fundamentally limited as: (a) the pseudo-labeling strategies are fixed during meta-learning and impossible to correct mislabeled samples; (b) the generative approaches heavily rely on the quality of generated samples and are cumbersome to scale into large-scale setups. 

> _∗_ Equal contributions 

> _†_ Work done at KAIST 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning_images/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning.pdf-0002-01.png)


**----- Start of picture text -----**<br>
Backbone  Projector  Predictor<br>PsCo<br>EMA<br>N-way K-shot<br>Stop-grad<br>Queue<br>Matching<br>Support Samples<br>**----- End of picture text -----**<br>


Figure 1: An overview of the proposed **P** seudo- **s** upervised **Co** ntrast (PsCo). PsCo constructs an _N_ - way _K_ -shot few-shot classification task using the current mini-batch _{_ **x** _i}_ and the queue of previous mini-batches; and then, it learns the task via contrastive learning. Here, **A** is a label assignment matrix found by the Sinkhorn-Knopp algorithm (Cuturi, 2013), _A_ is a pre-defined augmentation distribution, _f_ is a backbone feature extractor, _g_ and _h_ are projection and prediction MLPs, respectively, and _ϕ_ is an exponential moving average (EMA) of the model parameter _θ_ . 

To overcome the limitations of the existing UML approaches, in this paper, we ask whether one can (a) progressively improve a pseudo-labeling strategy during meta-learning, and (b) construct more diverse tasks without generative models. We draw inspiration from recent advances in _selfsupervised learning_ literature (He et al., 2020; Khosla et al., 2020), which has shown remarkable success in representation learning without labeled data. In particular, we utilize (a) a _momentum network_ to improve pseudo-labeling progressively via temporal ensemble; and (b) a _momentum queue_ to construct diverse tasks using previous mini-batches in an online manner. 

Formally, we propose **P** seudo- **s** upervised **Co** ntrast (PsCo), a novel and effective unsupervised metalearning framework, for few-shot classification. Our key idea is to construct few-shot classification tasks using the current and previous mini-batches based on the momentum network and the momentum queue. Specifically, given a random mini-batch of _N_ unlabeled samples, we treat them as _N_ queries (i.e., test samples) of different _N_ labels, and then select _K_ shots (i.e., training samples) for each label from the queue of previous mini-batches based on representations extracted by the momentum network. To further improve the selection procedure, we utilize top- _K_ sampling after applying a matching algorithm, Sinkhorn-Knopp (Cuturi, 2013). Finally, we optimize our model via supervised contrastive learning (Khosla et al., 2020) for solving the _N_ -way _K_ -shot task. Remark that our few-shot task construction relies on not only the current mini-batch but also the momentum network and the queue of previous mini-batches. Therefore, our task construction (i.e., pseudo-labeling) strategy (a) is progressively improved during meta-learning with the momentum network, and (b) constructs diverse tasks since the shots can be selected from the entire dataset. Our framework is illustrated in Figure 1. 

Throughout extensive experiments, we demonstrate the effectiveness of the proposed framework, PsCo, under various few-shot classification benchmarks. First, PsCo achieves state-of-the-art performance under both Omniglot (Lake et al., 2011) and miniImageNet (Ravi & Larochelle, 2017) few-shot benchmarks; its performance is even competitive with supervised meta-learning methods. Next, PsCo also shows superiority under cross-domain few-shot learning scenarios. Finally, we demonstrate that PsCo is scalable to a large-scale benchmark, ImageNet (Deng et al., 2009). 

We summarize our contributions as follows: 

- We propose PsCo, an effective unsupervised meta-learning (UML) framework for few-shot classification, which constructs diverse few-shot pseudo-tasks without labels utilizing the momentum network and the queue of previous batches in a progressive manner. 

- We achieve state-of-the-art performance on few-shot classification benchmarks, Omniglot (Lake et al., 2011) and miniImageNet (Ravi & Larochelle, 2017). For example, PsCo outperforms the prior art of UML, Meta-SVEBM (Kong et al., 2021), by 5% accuracy gain (58.03 _→_ 63.26), for 5-way 5-shot tasks of miniImageNet (see Table 1). 

2 

Published as a conference paper at ICLR 2023 

- We show that PsCo achieves comparable performance with supervised meta-learning methods in various few-shot classification benchmarks. For example, PsCo achieves 44.01% accuracy for 5-way 5-shot tasks of an unseen domain, Cars (Krause et al., 2013), while supervised MAML (Finn et al., 2017) does 41.17% (see Table 2). 

- We validate PsCo is also applicable to a large-scale dataset: e.g., we improve PsCo by 5.78% accuracy gain (47.67 _→_ 53.45) for 5-way 5-shot tasks of Cars using large-scale unlabeled data, ImageNet (Deng et al., 2009) (see Table 3). 

- 2 PRELIMINARIES 

- 2.1 PROBLEM STATEMENT: UNSUPERVISED FEW-SHOT LEARNING 

The problem of interest in this paper is _unsupervised few-shot learning_ , one of the popular unsupervised meta-learning applications. This aims to learn generalizable knowledge without human annotations for quickly adapting to unseen but relevant few-shot tasks. Following the meta-learning literature, we refer to the learning phase as meta-training and the adaptation phase as meta-test. 

Formally, we are only able to utilize an unlabeled dataset _D_ `meta` ~~`t`~~ `rain`[:=] _[{]_ **[x]** _i[}]_[ during meta-training] our model. At the meta-test phase, we transfer the model to new few-shot tasks _{Ti} ∼D_ `meta` ~~`t`~~ `est` where each task _Ti_ aims to classify query samples _{_ **x** _q}_ among _N_ labels using support (i.e., training) samples _S_ = _{_ ( **x** _s,_ **y** _s_ ) _}[NK] s_ =1[.][We][here][assume][the][task] _[T][i]_[consists][of] _[K]_[support][samples][for][each] label **y** _∈{_ 1 _, . . . , N }_ , which is referred to as _N_ -way _K_ -shot classification. Note that _D_ `meta` ~~`t`~~ `rain` and _D_ `meta` ~~`t`~~ `est`[can][come][from][the][same][domain][(i.e.,][the][standard][in-domain][setting)][or][different] domains (i.e., cross-domain) as suggested by Chen et al. (2019). 

## 2.2 CONTRASTIVE LEARNING 

Contrastive learning (Oord et al., 2018; Chen et al., 2020a; He et al., 2020; Khosla et al., 2020) aims to learn meaningful representations by maximizing the similarity between similar (i.e., positive) samples, and minimizing the similarity between dissimilar (i.e., negative) samples on the representation space. We first describe a general form of contrastive learning objectives based on the temperature-normalized cross entropy (Chen et al., 2020a; He et al., 2020) and its variant for multiple positives (Khosla et al., 2020) as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning_images/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning.pdf-0003-09.png)


where _{_ **q** _i}_ and _{_ **k** _j}_ are _ℓ_ 2-normalized query and key representations, respectively, **A** _∈{_ 0 _,_ 1 _}[NM]_ represents whether **q** _i_ and **k** _j_ are positive (A _i,j_ = 1) or negative (A _i,j_ = 0), and _τ_ is a hyperparameter for temperature scaling. 

Based on the recent observations in the self-supervised learning literature, we also describe a general scheme to construct the query and key representations using data augmentations and a momentum network. Formally, given a random mini-batch _{_ **x** _i}_ , the representations can be obtained as follows: 

**q** _i_ = Normalize( _hθ ◦ gθ ◦ fθ_ ( _ti,_ 1( **x** _i_ ))) _,_ **k** _i_ = Normalize( _gϕ ◦ fϕ_ ( _ti,_ 2( **x** _i_ ))) _,_ (2) where Normalize( _·_ ) is _ℓ_ 2 normalization, _ti,_ 1 _∼A_ 1 and _ti,_ 2 _∼A_ 2 are random data augmentations, _f_ is a backbone feature extractor like ResNet (He et al., 2016), _g_ and _h_ are projection and prediction MLPs,[1] respectively, and _ϕ_ is an exponential moving average (i.e., momentum) of the model parameter _θ_ .[2] Since a large number of negative samples plays a crucial role in contrastive learning, one can re-use the key representations of previous mini-batches by maintaining a queue (He et al., 2020). 

Note that the above forms (1) and (2) can be formulated as various contrastive learning frameworks. For example, SimCLR (Chen et al., 2020a) is a special case of no momentum _ϕ_ and no predictor _h_ . In addition, self-supervised contrastive learning methods (Chen et al., 2020a; He et al., 2020) often assume that **k** _i_ is only the positive key of **q** _i_ , i.e., A _i,j_ = 1 if and only if _i_ = _j_ , while supervised contrastive learning (Khosla et al., 2020) directly uses labels for **A** . 

> 1The prediction MLPs have been utilized in the recent SSL literature (Grill et al., 2020; Chen et al., 2021). 

> 2 _ϕ_ is updated by _ϕ ← mϕ_ + (1 _− m_ ) _θ_ for each training iteration where _m_ is a momentum hyperparameter. 

3 

Published as a conference paper at ICLR 2023 

## **Algorithm 1 P** seudo- **s** upervised **Co** ntrast (PsCo): PyTorch-like Pseudocode 

|#|f,<br>g,<br>h:<br>backbone,<br>projector,|f,<br>g,<br>h:<br>backbone,<br>projector,|f,<br>g,<br>h:<br>backbone,<br>projector,|and|and|and|predictor|predictor|predictor|predictor|||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|#|{f,g}_ema:<br>momentum<br>backbone,|||and|||projector||||||||
|#|queue:<br>momentum<br>queue<br>(Mxd)||||||||||||||
|#|mm:<br>matrix<br>multiplication,<br>mul:||||element-wise||||||multiplication||||
|def||PsCo(x):|||||||#|x:<br>a<br>mini-batch<br>of<br>N<br>samples|||||
|||x1,<br>x2<br>=<br>aug1(x),<br>aug2(x)|||||||#|two<br>augmented<br>views<br>of<br>x|||||
|||q<br>=<br>h(g(f(x1)))|||||||#|(Nxd)|N<br>query<br>representations||||
|||z<br>=<br>g_ema(f_ema(x2))|||||||#|(Nxd)|N<br>query<br>momentum<br>representations||||
|||sim<br>=<br>mm(z,<br>queue.T)|||||||#|(NxM)|similarity<br>matrix||||
|||A_tilde<br>=<br>sinkhorn(sim)|||||||#|(NxM)|soft<br>pseudo-label<br>assignment|||matrix|
|||s,<br>A<br>=<br>select_topK(queue,|A_tilde)||||||#|(NKxd)<br>s:<br>support<br>momentum||representations|||
||||||||||#|(NxNK)<br>A:<br>pseudo-label<br>assignment|||matrix||
|||logits<br>=<br>mm(q,<br>s.T)<br>/<br>temperature|||||||||||||
|||loss<br>=<br>logits.logsumexp(dim=1)||||-||mul(logits,<br>A).sum(dim=1)<br>/<br>K|||||||
|||return<br>loss.mean()|||||||||||||



## 3 METHOD: PSEUDO-SUPERVISED CONTRASTIVE META-LEARNING 

In this section, we introduce _**P** seudo-_ _**s** upervised_ _**Co** ntrast_ ( _PsCo_ ), a novel and effective framework for unsupervised few-shot learning. Our key idea is to construct few-shot classification pseudotasks using the current and previous mini-batches with the momentum network and the momentum queue. We then employ supervised contrastive learning (Khosla et al., 2020) for learning the pseudotasks. The detailed implementations of our task construction, meta-training objective, and meta-test scheme for unsupervised few-shot learning are described in Section 3.1, 3.2, and 3.3, respectively. Our framework is illustrated in Figure 1 and its pseudo-code is provided in Algorithm 1. Note that we use the same notations described in Section 2 for consistency. 

## 3.1 ONLINE PSEUDO-TASK CONSTRUCTION 

We here describe how to construct a few-shot pseudo-task using unlabeled data _D_ `meta` ~~`t`~~ `rain`[=] _[ {]_ **[x]** _i[}]_[.] To this end, we maintain a queue of previous mini-batches. Then, we treat the previous and current mini-batch samples as training (i.e., shots) and test (i.e., queries) samples for our few-shot pseudotask. Formally, let _B_ := _{_ **x** _i}[N] i_ =1[be the current mini-batch randomly sampled from] _[ D]_ `[meta]` ~~`t`~~ `rain`[, and] _Q_ := _{_ **x** ˜ _j}[M] j_ =1[be the queue of previous mini-batch samples.][Now, we treat] _[ B]_[=] _[ {]_ **[x]** _[i][}][N] i_ =1[as queries] of _N_ different pseudo-labels and find _K_ (appropriate) shots for each pseudo-label from the queue _Q_ . Remark that this approach to utilize the previous mini-batches encourages us to construct more diverse tasks. 

To find the shots efficiently, we utilize the momentum network and the momentum queue described in Section 2.2. For the current mini-batch samples, we compute the momentum query representations with data augmentations _ti,_ 2 _∼A_ 2, i.e., **z** _i_ := Normalize( _gϕ ◦ fϕ_ ( _ti,_ 2( **x** _i_ ))). Following Heinsteadet al.of(2020raw data), weinstorethe queueonly the _Q_ **z** momentum, i.e., _Q_ **z** :=representations _{_ **z** ˜ _j}[M] j_ =1[.][Remark] of the[that] previous[the][use] mini-batch[of][the][momentum] samples network is not only for efficiency but also for improving our task construction strategy because the momentum network is consistent and progressively improved during training. Following He et al. (2020), we randomly initialize the queue _Q_ **z** at the beginning of training. 

Now, the remaining question is as follows: How to find _K_ appropriate shots from the queue _Q_ for each pseudo-label using the momentum representations? Before introducing our algorithm, we first discuss two requirements for constructing semantically meaningful few-shot tasks: (i) shots and queries of the same label should be semantically similar, and (ii) all shots should be different. Based on these requirements, we formulate our assignment problem as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning_images/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning.pdf-0004-09.png)


Obtaining the exact optimal solution to the above assignment problem for each training iteration might be too expensive for our purpose (Ramshaw & Tarjan, 2012). Instead, we use an approximate algorithm: we first apply a fast version (Cuturi, 2013) of the Sinkhorn-Knopp algorithm to solve the 

4 

Published as a conference paper at ICLR 2023 

following problem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning_images/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning.pdf-0005-02.png)


which is an entropy-regularized optimal transport problem (Cuturi, 2013). Its optimal solution **A**[˜] _[∗]_ can be obtained efficiently and can be considered as a soft assignment matrix between the current mini-batch _{_ **z** _i}[N] i_ =1[and the queue] _[ Q]_ **[z]**[=] _[{]_ **[z]**[˜] _[j][}][M] j_ =1[.][Hence,][we select top-] _[K]_[elements for each row] of the assignment matrix **A**[˜] _[∗]_ and finally construct an _N_ -way _K_ -shot pseudo-task consisting of (a) query samples _B_ = _{_ **x** _i}[N] i_ =1[,][(b)][the][support][representations] _[S]_ **[z]**[:=] _[{]_ **[z]**[˜] _[s][}][NK] s_ =1[,][and][(c)][the][pseudo-] label assignment matrix **A** _∈{_ 0 _,_ 1 _}[N][×][NK]_ . Note that Figure 1 shows an example of a 5-way 2-shot task. We empirically observe that our task construction strategy satisfies the above requirements (i) and (ii) (see Section 4.3). 

## 3.2 META-TRAINING: SUPERVISED CONTRASTIVE LEARNING WITH PSEUDO TASKS 

We now describe our meta-learning objective _L_ `PsCo` for learning our few-shot pseudo-tasks. We here use our model _θ_ to obtain query representations: **q** _i_ := Normalize( _hθ ◦ gθ ◦ fθ_ ( _ti,_ 1( **x** _i_ ))) where _ti,_ 1 _∼A_ 1 is a random data augmentation for each _i_ . Then, our objective _L_ `PsCo` is defined as follows: 

_L_ `PsCo` := _L_ `Contrast` ( _{_ **q** _i}[N] i_ =1 _[,][ S]_ **[z]** _[,]_ **[ A]**[;] _[ τ]_ `[PsCo]`[)] _[,]_ (5) where _S_ **z** := _{_ **z** ˜ _s}[NK] s_ =1[is][the][support][representations][and] **[A]** _[∈{]_[0] _[,]_[ 1] _[}][N][×][NK]_[is][the][pseudo-label] assignment matrix, which are constructed by our task construction strategy described in Section 3.1. 

Since our framework PsCo uses the same architectural components as a self-supervised learning framework, MoCo (He et al., 2020), the MoCo objective _L_ `MoCo` can be incorporated into our PsCo without additional computation costs. Note that the MoCo objective can be written as follows: 

_L_ `MoCo` := _L_ `Contrast` ( _{_ **q** _i}[N] i_ =1 _[,][ {]_ **[z]** _[i][}][N] i_ =1 _[∪Q]_ **[z]** _[,]_ **[ A]** `[MoCo]`[;] _[ τ]_ `[MoCo]`[)] _[,]_ (6) where ( **A** `MoCo` ) _i,j_ = 1 if and only if _i_ = _j_ , and **z** _i_ := Normalize( _gϕ ◦ fϕ_ ( _ti,_ 2( **x** _i_ ))) as described in Section 3.1. We optimize our model _θ_ via all the objectives, i.e., _L_ `total` := _L_ `PsCo` + _L_ `MoCo` . Remark again that _ϕ_ is updated by exponential moving average (EMA), i.e., _ϕ ← mϕ_ + (1 _− m_ ) _θ_ . 

**Weak augmentation for momentum representations.** To successfully find the pseudo-label assignment matrix **A** , we apply weak augmentations for the momentum representations (i.e., _A_ 2 is weaker than _A_ 1) as Zheng et al. (2021) did. This reduces the noise in the representations and consequently enhances the performance of our PsCo as **A** becomes more accurate (see Section 4.3). 

## 3.3 META-TEST 

At the meta-test stage, we have an _N_ -way _K_ -shot task _T_ consisting of query samples _{_ **x** _q}_ and support samples _S_ = _{_ ( **x** _s,_ **y** _s_ ) _}[NK] s_ =1[.][3][We here discard the momentum network] _[ ϕ]_[ and use only the] online network _θ_ . To predict labels, we first compute the query representation **q** _q_ := Normalize( _hθ ◦ gθ◦fθ_ ( **x** _q_ )) and the support representations **z** _s_ := Normalize ( _gθ ◦ fθ_ ( **x** _s_ ))). Then we predict a label by the following classification rule: _y_ ˆ := arg max _y_ **q** _[⊤] q_ **[c]** _[y]_[where] **[ c]** _[y]_[:=][ Normalize][(][�] _s_ **[1]** _[y] s_[=] _[y][ ·]_ **[z]** _[s]_[)][ is] the prototype vector. This is inspired by our _L_ PsCo, which can be interpreted as minimizing distance from the mean (i.e., prototype) of the shot representations.[4] 

**Further adaptation for cross-domain few-shot classification.** Under cross-domain few-shot classification scenarios, the model _θ_ should further adapt to the meta-test domain due to the dissimilarity from meta-training. We here suggest an efficient adaptation scheme using only a few labeled samples. Our idea is to consider the support samples as queries. To be specific, we compute the query representation **q** _s_ := Normalize( _hθ ◦gθ ◦fθ_ ( **x** _s_ )) for each support sample **x** _s_ , and construct the label assignment matrix **A** _[′]_ as A _[′] s,s[′]_[=][1][ if and only if] _[ y][s]_[=] _[y][s][′]_[.][Then we simply optimize only] _[ g][θ]_[and] _hθ_ via contrastive learning, i.e., _L_ `Contrast` ( _{_ **q** _s}, {_ **z** _s},_ **A** _[′]_ ; _τ_ `PsCo` ), for few iterations. We empirically observe that this adaptation scheme is effective under cross-domain settings (see Section 4.3). 

> 3Note that _N_ and _K_ for meta-training and meta-test could be different. We use a large _N_ (e.g., _N_ = 256) during meta-training to fully utilize computational resources like standard deep learning, and a small _N_ (e.g., _N_ = 5) during meta-test following the meta-learning literature. 

> 4 _L_ PsCo = _− N_ 1 � _i τ_ PsCo1 **[q]** _i[⊤]_ � _K_ 1 � _j_[A] _[i,j]_ **[z]** _[j]_ � + term not depending on **A** . 

5 

Published as a conference paper at ICLR 2023 

Table 1: Few-shot classification accuracy (%) on Omniglot and miniImageNet benchmarks. We report the average accuracy over 2000 few-shot tasks for PsCo and self-supervised learning methods. Other reported numbers borrow from Khodadadeh et al. (2021); Kong et al. (2021). **Bold** entries indicate the best for each task configuration, among unsupervised and self-supervised methods. 

|**Method**|**Omniglot (way, shot)**<br>**(5,1)**<br>**(5,5)**<br>**(20,1)**<br>**(20,5)**|**miniImageNet (way, shot)**<br>**(5,1)**<br>**(5,5)**<br>**(5,20)**<br>**(5,50)**|
|---|---|---|
|_Training from Scratch_|52.50<br>74.78<br>24.91<br>47.62|27.59<br>38.48<br>51.53<br>59.63|
||_Unsupervised meta-learning_||
|CACTUs-MAML<br>CACTUs-ProtoNets<br>UMTRA<br>LASIUM-MAML<br>LASIUM-ProtoNets<br>Meta-GMVAE<br>Meta-SVEBM<br>**PsCo (Ours)**|68.84<br>87.78<br>48.09<br>73.36<br>39.90<br>53.97<br>63.84<br>69.64<br>68.12<br>83.58<br>47.75<br>66.27<br>39.18<br>53.36<br>61.54<br>63.55<br>83.80<br>95.43<br>74.25<br>92.12<br>39.93<br>50.73<br>61.11<br>67.15<br>83.26<br>95.29<br>-<br>-<br>40.19<br>54.56<br>65.17<br>69.13<br>80.15<br>91.10<br>-<br>-<br>40.05<br>52.53<br>61.09<br>64.89<br>94.92<br>97.09<br>82.21<br>90.61<br>42.82<br>55.73<br>63.14<br>68.26<br>91.85<br>97.21<br>79.66<br>92.21<br>43.38<br>58.03<br>67.07<br>72.28<br>**96.37**<br>**99.13**<br>**89.64**<br>**97.07**<br>**46.70**<br>**63.26**<br>**72.22**<br>**73.50**||
||_Self-supervised learning_||
|SimCLR<br>MoCo v2<br>SwAV|92.13<br>97.06<br>80.95<br>91.60<br>43.35<br>52.50<br>61.83<br>64.85<br>92.66<br>97.38<br>82.13<br>92.35<br>41.92<br>50.94<br>60.23<br>63.45<br>93.13<br>97.32<br>82.63<br>92.12<br>43.24<br>52.41<br>61.36<br>64.52||
||_Supervised meta-learning_||
|MAML<br>ProtoNets|94.46<br>98.83<br>84.60<br>96.29<br>46.81<br>62.13<br>71.03<br>75.54<br>98.35<br>99.58<br>95.31<br>98.81<br>46.56<br>62.29<br>70.05<br>72.04||



## 4 EXPERIMENTS 

In this section, we demonstrate the effectiveness of the proposed framework under standard fewshot learning benchmarks (Section 4.1) and cross-domain few-shot learning benchmarks (Section 4.2). We provide ablation studies regarding PsCo in Section 4.3. Following Lee et al. (2021), we mainly use Conv4 and Conv5 architectures for Omniglot (Lake et al., 2011) and miniImageNet (Ravi & Larochelle, 2017), respectively, for the backbone feature extractor _fθ_ . For the number of shots during meta-learning, we use _K_ = 1 for Omniglot and _K_ = 4 for miniImageNet (see Table 6 for the sensitivity of _K_ ). Other details are fully described in Appendix A. We omit the confidence intervals in this section for clarity, and the full results with them are provided in Appendix F. 

## 4.1 STANDARD FEW-SHOT BENCHMARKS 

**Setup.** We here evaluate PsCo on the standard few-shot benchmarks of unsupervised meta-learning: Omniglot (Lake et al., 2011) and miniImageNet (Ravi & Larochelle, 2017). We compare PsCo’s performance with unsupervised meta-learning methods (Hsu et al., 2019; Khodadadeh et al., 2019; 2021; Lee et al., 2021; Kong et al., 2021), self-supervised learning methods (Chen et al., 2020a;b; Caron et al., 2020), and supervised meta-learning methods (Finn et al., 2017; Snell et al., 2017) on the benchmarks. The details of the benchmarks and the baselines are described in Appendix D. 

**Few-shot classification results.** Table 1 shows the results of the few-shot classification with various (way, shot) tasks of Omniglot and miniImageNet. PsCo achieves state-of-the-art performance on both Omniglot and miniImageNet benchmarks under the unsupervised setting. For example, we obtain 5% accuracy gain (67.07 _→_ 72.22) on miniImageNet 5-way 20-shot tasks. Moreover, the performance is even competitive with supervised meta-learning methods, ProtoNets (Snell et al., 2017), and MAML (Finn et al., 2017) as well. 

## 4.2 CROSS-DOMAIN FEW-SHOT BENCHMARKS 

**Setup.** We evaluate PsCo on cross-domain few-shot classification benchmarks following Oh et al. (2022). To be specific, we use (a) benchmark of large-similarity with ImageNet: CUB (Wah et al., 2011), Cars (Krause et al., 2013), Places (Zhou et al., 2018), and Plantae (Horn et al., 2018); (b) benchmarks of small-similarity with ImageNet: CropDiseases (Mohanty et al., 2016), EuroSAT (Helber et al., 2019), ISIC (Codella et al., 2018), and ChestX (Wang et al., 2017). As baselines, we 

6 

Published as a conference paper at ICLR 2023 

Table 2: Few-shot classification accuracy (%) on cross-domain few-shot classification benchmarks. We transfer Conv5 trained on miniImageNet to each benchmark. We report the average accuracy over 2000 few-shot tasks for all methods, except Meta-SVEBM as it is evaluated over 200 tasks due to the long evaluation time. **Bold** entries indicate the best for each task configuration, among unsupervised and self-supervised methods. 

|ed and self-supervised methods.|ed and self-supervised methods.|ed and self-supervised methods.|ed and self-supervised methods.|ed and self-supervised methods.|
|---|---|---|---|---|
|(a) Cross-domain few-shot benchmarks similar to miniImageNet.|||||
|**Method**|**CUB**<br>**(5, 5)**<br>**(5, 20)**|**Cars**<br>**(5, 5)**<br>**(5, 20)**|**Places**<br>**(5, 5)**<br>**(5, 20)**|**Plantae**<br>**(5, 5)**<br>**(5, 20)**|
||_Unsupervised meta-learning_||||
|Meta-GMVAE<br>Meta-SVEBM<br>**PsCo (Ours)**|47.48<br>54.08<br>31.39<br>38.36<br>57.70<br>65.08<br>45.50<br>54.61<br>34.27<br>46.23<br>51.27<br>61.09<br>**57.38**<br>**68.58**<br>**44.01**<br>**57.50**<br>**63.60**<br>**73.95**|||38.27<br>45.02<br>38.12<br>46.22<br>**52.72**<br>**64.53**|
||_Self-supervised learning_||||
|SimCLR<br>MoCo v2<br>SwAV|52.11<br>61.89<br>37.40<br>50.05<br>60.10<br>69.93<br>53.23<br>62.81<br>38.65<br>51.77<br>59.09<br>69.08<br>51.58<br>61.38<br>36.85<br>50.03<br>59.57<br>69.70|||43.42<br>54.92<br>43.97<br>55.45<br>42.68<br>54.03|
||_Supervised meta-learning_||||
|MAML<br>ProtoNets|56.57<br>64.17<br>41.17<br>48.82<br>60.05<br>67.54<br>56.74<br>65.03<br>38.98<br>47.98<br>59.39<br>67.77|||47.33<br>54.86<br>45.89<br>54.29|
|(b) Cross-domain few-shot benchmarks dissimilar to miniImageNet.|||||
|**Method**|**CropDiseases**<br>**(5, 5)**<br>**(5, 20)**|**EuroSAT**<br>**(5, 5)**<br>**(5, 20)**|**ISIC**<br>**(5, 5)**<br>**(5, 20)**|**ChestX**<br>**(5, 5)**<br>**(5, 20)**|
||_Unsupervised meta-learning_||||
|Meta-GMVAE<br>Meta-SVEBM<br>**PsCo (Ours)**|73.56<br>81.22<br>73.83<br>80.11<br>33.48<br>39.48<br>71.82<br>83.13<br>70.83<br>80.21<br>38.85<br>48.43<br>**88.24**<br>**94.95**<br>**81.08**<br>**87.65**<br>**44.00**<br>**54.59**|||23.23<br>26.26<br>**26.26**<br>28.91<br>24.78<br>27.69|
||_Self-supervised learning_||||
|SimCLR<br>MoCo v2<br>SwAV|79.90<br>88.73<br>79.14<br>85.05<br>42.83<br>51.35<br>80.96<br>89.85<br>79.94<br>86.16<br>43.43<br>52.14<br>80.15<br>89.24<br>79.31<br>85.62<br>43.21<br>51.99|||25.14<br>**29.21**<br>25.24<br>29.19<br>24.99<br>28.57|
||_Supervised meta-learning_||||
|MAML<br>ProtoNets|77.76<br>83.24<br>71.48<br>76.70<br>47.34<br>55.09<br>76.01<br>83.64<br>64.91<br>70.88<br>40.62<br>48.38|||22.61<br>24.25<br>23.15<br>25.72|



test the previous state-of-the-art unsupervised meta-learning (Lee et al., 2021; Kong et al., 2021), self-supervised learning (Chen et al., 2020a;b; Caron et al., 2020), and supervised meta-learning (Finn et al., 2017; Snell et al., 2017). We here use our adaptation scheme (Section 3.3) with 50 iterations. The details of the benchmarks and implementations are described in Appendix E. 

**Small-scale cross-domain few-shot classification results.** We here evaluate various Conv5 models meta-trained on miniImageNet as used in Section 4.1. Table 2 shows that PsCo outperforms all the baselines across all the benchmarks, except ChestX, which is too different from the distribution of miniImageNet (Oh et al., 2022). Somewhat interestingly, PsCo competitive with supervised learning under these benchmarks, e.g., PsCo achieves 88% accuracy on CropDiseases 5-way 5-shot tasks, whereas MAML gets 77%. This implies that our unsupervised method, PsCo, generalizes on more diverse tasks than supervised learning, which is specialized to in-domain tasks. 

**Large-scale cross-domain few-shot classification results.** We also validate that our meta-learning framework is applicable to the large-scale benchmark, ImageNet (Deng et al., 2009). Remark that the recent unsupervised meta-learning methods (Lee et al., 2021; Kong et al., 2021; Khodadadeh et al., 2021) rely on generative models, so they are not easily applicable to such a large-scale benchmark. For example, we observe that PsCo is 2.7 times faster than the best baseline, Meta-SVEBM (Kong et al., 2021), even though Meta-SVEBM uses low-dimensional representations instead of full images during training. Hence, we compare PsCo with (a) self-supervised methods, MoCo v2 (Chen et al., 2020b) and BYOL (Grill et al., 2020), and (b) the publicly-available supervised learning baseline. We here use the ResNet-50 (He et al., 2016) architecture. The training details are described in Appendix E.4 and we also provide ResNet-18 results in Appendix F. 

7 

Published as a conference paper at ICLR 2023 

Table 3: 5-way 5-shot classification accuracy (%) on cross-domain few-shot benchmarks. We transfer ImageNet-trained ResNet-50 models to each benchmark. We report the average accuracy over 600 few-shot tasks. 

|few-shot tasks.|||||||||
|---|---|---|---|---|---|---|---|---|
|**Method**|**CUB**|**Cars**|**Places**|**Plantae**|**CropDiseases**|**EuroSAT**|**ISIC**|**ChestX**|
|MoCo v2|64.16|47.67|81.39|61.36|82.89|76.96|38.26|**24.28**|
|**+PsCo (Ours)**|**76.63**|**53.45**|**83.87**|**69.17**|**89.85**|**83.99**|**41.64**|23.60|
|BYOL|67.45|45.74|75.43|56.86|80.82|77.70|37.27|24.15|
|**+PsCo (Ours)**|**82.13**|**56.19**|**83.80**|**71.14**|**92.92**|**85.33**|**42.90**|**26.05**|
|Supervised|89.13|75.15|84.41|72.91|90.96|85.64|43.34|25.35|



Table 4: Component ablation studies on Omniglot. 

|Momentum|Predictor|Sinkhorn|Top-K sampling|_L_`MoCo`|(5, 1)|(5, 5)|(20, 1)|(20, 5)|
|---|---|---|---|---|---|---|---|---|
|✓|✓|✓|✓|✓|**96.37**|**99.13**|**89.64**|**97.07**|
|✗|✓|✓|✓|✓|90.32|96.78|76.17|90.41|
|✓|✗|✓|✓|✓|90.21|96.86|76.15|90.53|
|✓|✓|✗|✓|✓|95.81|98.94|88.25|96.57|
|✓|✓|✓|✗|✓|94.95|98.81|86.32|96.05|
|✓|✓|✓|✓|✗|93.16|97.40|81.03|91.33|



Table 3 shows that (i) PsCo consistently improves both MoCo and BYOL under this setup (e.g., 67% _→_ 82% in CUB), and (ii) PsCo benefits from the large-scale dataset as we obtain a huge amount of performance gain on the benchmarks of large-similarity with ImageNet: CUB, Cars, Places, and Plantae. Consequently, we achieve comparable performance with the supervised learning baseline, except Cars, which shows that our PsCo is applicable to large-scale unlabeled datasets. 

## 4.3 ABLATION STUDY 

**Component analysis.** In Table 4, we demonstrate the necessity of each component in PsCo by removing the components one by one: momentum encoder _ϕ_ , prediction head _h_ , Sinkhorn-Knopp algorithm, top- _K_ sampling for sampling support samples, and the MoCo objective, _L_ `MoCo` (6). We found that the momentum network _ϕ_ and the prediction head _h_ are critical architectural components in our framework like recent self-supervised learning frameworks (Grill et al., 2020; Chen et al., 2021). In addition, Table 4 shows that training with only our objective, _L_ `PsCo` (5), achieves meaningful performance, but incorporating it into MoCo is more beneficial. To further validate that our task construction is progressively improved during meta-learning, we evaluate whether a query and a corresponding support sample have the same true label. Figure 2a shows that our task construction is progressively improved, i.e., the task requirement (i) described in Section 3.1 satisfies. 

Table 4 also verifies the contribution of the Sinkhorn-Knopp algorithm and Top- _K_ sampling for the performance of PsCo. We further analyze the effect of the Sinkhorn-Knopp algorithm by measuring the overlap ratio of selected supports between different pseudo-labels. As shown in Figure 2b, there are almost zero overlaps when using the Sinkhorn-Knopp algorithm, which means the constructed task is a valid few-shot task, satisfying the task requirement (ii) described in Section 3.1. 

**Adaptation effect on cross-domain.** To validate the effect of our adaptation scheme (Section 3.3), we evaluate the few-shot classification accuracy during the adaptation process on miniImageNet (i.e., in-domain) and CropDiseases (i.e., cross-domain) benchmarks. As shown in Figure 2d, we found that the adaptation scheme is more useful in cross-domain benchmarks than in-domain ones. Based on these results, we apply the scheme to only the cross-domain scenarios. We also found that our adaptation does not cause over-fitting since we only optimize the projection and prediction heads _gθ_ and _hθ_ . The results for the adaptation effect on the whole benchmarks are represented in Appendix C. 

**Augmentations.** We here confirm that weak augmentation for the momentum network (i.e., _A_ 2) is more effective than strong augmentation unlike other self-supervised learning literature (Chen et al., 2020a; He et al., 2020). We denote the standard augmentation consisting of both geometric and color transformations by _Strong_ , and a weaker augmentation consisting of only geometric transformations as _Weak_ (see details in Appendix A). As shown in Table 5, utilizing the weak augmentation for _A_ 2 is much more beneficial since it helps to find an accurate pseudo-label assignment matrix **A** . 

8 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning_images/iclr_2023_Unsupervised_Meta-learning_via_Few-shot_Pseudo-supervised_Contrastive_Learning.pdf-0009-01.png)


**----- Start of picture text -----**<br>
correctly sampled shots overlap between sampled shots miniImageNet   miniImageNet miniImageNet   CropDiseases<br>70 6 75 100 5shot<br>20shot<br>60 4 70 95<br>w/ sinkhorn 5shot<br>50 w/out sinkhorn 65 20shot 90<br>2<br>40 Ours 60 85<br>MoCo v2<br>300 100 200 300 400 00 100 200 300 400 550 20 40 60 80 100 800 20 40 60 80 100<br>train epochs train epochs adaptation iterations adaptation iterations<br>(a) Pseudo-label quality (b) Shot overlap ratio (c) In-domain adaptation (d) Cross-domain adaptation<br>accuracy (%) accuracy (%) accuracy (%)<br>overlap ratio (%)<br>**----- End of picture text -----**<br>


Figure 2: (a) Pseudo-label quality, measuring the agreement between pseudo-labels and true labels, (b) Shot overlap ratio, measuring whether the shots for each pseudo-label are disjoint, during meta-training. (c,d) Performance while adaptation on in-domain (miniImageNet) and cross-domain (CropDiseases) benchmarks, respectively. We obtain these results from 100 random batches. 

Table 5: The ablation study with varying augmentation choices for _A_ 1 and _A_ 2 on miniImageNet. 

Table 6: The ablation study with varying _K_ on miniImageNet. 

|_A_1<br>_A_2<br>(5, 1)<br>(5, 5)<br>(5, 20)<br>(5, 50)<br>Strong<br>Strong<br>44.54<br>60.04<br>68.61<br>71.20<br>Strong<br>Weak<br>**46.70**<br>**63.26**<br>**72.22**<br>**73.50**<br>Weak<br>Strong<br>43.56<br>58.77<br>67.21<br>69.46<br>Weak<br>Weak<br>40.68<br>55.05<br>63.32<br>65.82|_K_<br>(5, 1)<br>(5, 5)<br>(5, 20)<br>(5, 50)|
|---|---|
||1<br>45.88<br>61.84<br>70.25<br>72.76<br>4<br>**46.70**<br>**63.26**<br>**72.22**<br>**73.50**<br>16<br>46.31<br>62.76<br>70.91<br>73.43<br>64<br>46.60<br>62.50<br>70.82<br>73.22|



**Training** _K_ **.** We also look at the effect of the training _K_ , i.e. number of shots sampled online. We conduct the experiment with _K ∈{_ 1 _,_ 4 _,_ 16 _,_ 64 _}_ . We observe that PsCo performs consistently well regardless of the choice of _K_ as shown in Table 6. The proper _K_ is suggested to obtain the best-performing models, e.g., _K_ = 4 for miniImageNet and _K_ = 1 for Omniglot are the best. 

## 5 RELATED WORKS 

**Unsupervised meta-learning.** Unsupervised meta-learning (Hsu et al., 2019; Khodadadeh et al., 2019; Lee et al., 2021; Kong et al., 2021; Khodadadeh et al., 2021) links meta-learning and unsupervised learning by constructing synthetic tasks and extracting the meaningful information from unlabeled data. For example, CACTUs (Hsu et al., 2019) cluster the data on the pretrained representations at the beginning of meta-learning to assign pseudo-labels. Instead of pseudo-labeling, UMTRA (Khodadadeh et al., 2019) and LASIUM (Khodadadeh et al., 2021) generate synthetic samples using data augmentations or pretrained generative networks like BigBiGAN (Donahue & Simonyan, 2019). Meta-GMVAE (Lee et al., 2021) and Meta-SVEBM (Kong et al., 2021) represent unknown labels via categorical latent variables using variational autoencoders (Kingma & Welling, 2014) and energy-based models (Teh et al., 2003), respectively. In this paper, we suggest a novel online pseudo-labeling strategy to construct diverse tasks without help from any pretrained network or generative model. As a result, our method is easily applicable to large-scale datasets. 

**Self-supervised learning.** Self-supervised learning (SSL) (Doersch et al., 2015) has shown remarkable success for unsupervised representation learning across various domains, including vision (He et al., 2020; Chen et al., 2020a), speech (Oord et al., 2018), and reinforcement learning (Laskin et al., 2020). Among SSL objectives, contrastive learning (Oord et al., 2018; Chen et al., 2020a; He et al., 2020) is arguably most popular for learning meaningful representations. In addition, recent advances have been made with the development of various architectural components: e.g., Siamese networks (Doersch et al., 2015), momentum networks (He et al., 2020), and asymmetric architectures (Grill et al., 2020; Chen & He, 2021). In this paper, we utilize the SSL components to construct diverse few-shot tasks in an unsupervised manner. 

## 6 CONCLUSION 

Although unsupervised meta-learning (UML) and self-supervised learning (SSL) share the same purpose of learning generalizable knowledge to unseen tasks by utilizing unlabeled data, there still exists a gap between UML and SSL literature. In this paper, we bridge the gap as we tailor various SSL components to UML, especially for few-shot classification, and we achieve superior performance under various few-shot classification scenarios. We believe our research could bring many future research directions in both the UML and SSL communities. 

9 

Published as a conference paper at ICLR 2023 

## ETHICS STATEMENT 

Unsupervised learning, especially self-supervised learning, often requires a large number of training samples, a huge model, and a high computational cost for training the model on large-scale data to obtain meaningful representations because of the absence of human annotations. Furthermore, finetuning the model for solving a new task is also time-consuming and memory-inefficient. Hence, it could raise environmental issues such as carbon generation, which could bring an abnormal climate and accelerate global warming. In that sense, meta-learning should be considered as a solution since its purpose is to learn generalizable knowledge that can be quickly adapted to unseen tasks. In particular, unsupervised meta-learning, which benefits from both meta-learning and unsupervised learning, would be an important research direction. We believe that our work could be a useful step toward learning _easily-_ generalizable knowledge from unlabeled data. 

## REPRODUCIBILITY STATEMENT 

We provide all the details to reproduce our experimental results in Appendix A, D, and E. The code is available at https://github.com/alinlab/PsCo. In our experiments, we mainly use NVIDIA GTX3090 GPUs. 

## ACKNOWLEDGMENTS AND DISCLOSURE OF FUNDING 

This work was mainly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST); No.2022-0-00713, Meta-learning applicable to real-world problems; No.2022-0-00959, Few-shot Learning of Causal Inference in Vision and Language for Decision Making). 

## REFERENCES 

Ferran Alet, Martin F. Schneider, Tomas Lozano-Perez, and Leslie Pack Kaelbling. Meta-learning curiosity algorithms. In _International Conference on Learning Representations_ , 2020. URL https://openreview.net/forum?id=BygdyxHFDS. 

- Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. 

- Ting Chen, Simon Kornblith, Mohammad Norouzi 0002, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In _Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event_ , volume 119 of _Proceedings of Machine Learning Research_ , pp. 1597–1607. PMLR, 2020a. 

- Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. A closer look at few-shot classification. In _7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019_ . OpenReview.net, 2019. 

- Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 15750–15758, 2021. 

- Xinlei Chen, Haoqi Fan, Ross B. Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. _CoRR_ , abs/2003.04297, 2020b. 

- Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9640–9649, 2021. 

- Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin K. Mishra, Harald Kittler, and Allan Halpern. 

10 

Published as a conference paper at ICLR 2023 

Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). In _15th IEEE International Symposium on Biomedical Imaging, ISBI 2018, Washington, DC, USA, April 4-7, 2018_ , pp. 168–172. IEEE, 2018. 

- Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In _Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States_ , pp. 2292–2300, 2013. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li 0002. Imagenet: A large-scale hierarchical image database. In _2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA_ , pp. 248–255. IEEE, 2009. 

- Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In _Proceedings of the IEEE international conference on computer vision_ , pp. 1422–1430, 2015. 

- Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. In _Advances in neural information processing systems_ , 2019. 

- Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. RL[2] : Fast reinforcement learning via slow reinforcement learning, 2017. URL https://openreview. net/forum?id=HkLXCE9lx. 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International conference on machine learning_ , pp. 1126–1135. PMLR, 2017. 

- Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In _International Conference on Machine Learning_ , pp. 1568–1577. PMLR, 2018. 

- Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in neural information processing systems_ , 33:21271–21284, 2020. 

- Yunhui Guo, Noel Codella, Leonid Karlinsky, James V. Codella, John R. Smith, Kate Saenko, Tajana Rosing, and Rog´erio Feris. A broader study of cross-domain few-shot learning. In _Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXVII_ , volume 12372 of _Lecture Notes in Computer Science_ , pp. 124–141. Springer, 2020. 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 770–778, 2016. 

- Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020_ , pp. 9726–9735. IEEE, 2020. 

- Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. _IEEE J Sel. Topics in Appl. Earth Observ. and Remote Sensing_ , 12(7):2217–2226, 2019. 

- Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alexander Shepard, Hartwig Adam, Pietro Perona, and Serge J. Belongie. The inaturalist species classification and detection dataset. In _2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018_ , pp. 8769–8778. IEEE Computer Society, 2018. 

11 

Published as a conference paper at ICLR 2023 

- Rein Houthooft, Yuhua Chen, Phillip Isola, Bradly Stadie, Filip Wolski, OpenAI Jonathan Ho, and Pieter Abbeel. Evolved policy gradients. In _Advances in Neural Information Processing Systems_ , 2018. 

- Kyle Hsu, Sergey Levine, and Chelsea Finn. Unsupervised learning via meta-learning. In _7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019_ . OpenReview.net, 2019. 

- Siavash Khodadadeh, Ladislau B¨ol¨oni, and Mubarak Shah. Unsupervised meta-learning for few-shot image classification. In _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada_ , pp. 10132–10142, 2019. 

- Siavash Khodadadeh, Sharare Zehtabian, Saeed Vahidian, Weijia Wang, Bill Lin, and Ladislau B¨olo¨oi. Unsupervised meta-learning through latent-space interpolation in generative models. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021. 

- Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. 

- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_ , 2015. URL http://arxiv.org/abs/1412.6980. 

- Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In _2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings_ , 2014. 

- Deqian Kong, Bo Pang, and Ying Nian Wu. Unsupervised meta-learning via latent space energybased model of symbol vector coupling. In _Fifth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems_ , 2021. 

- Jonathan Krause, Michael Stark 0003, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In _2013 IEEE International Conference on Computer Vision Workshops, ICCV Workshops 2013, Sydney, Australia, December 1-8, 2013_ , pp. 554–561. IEEE Computer Society, 2013. 

- Brenden M. Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua B. Tenenbaum. One shot learning of simple visual concepts. In _Proceedings of the 33th Annual Meeting of the Cognitive Science Society, CogSci 2011, Boston, Massachusetts, USA, July 20-23, 2011_ . cognitivesciencesociety.org, 2011. 

- Michael Laskin, Aravind Srinivas, and Pieter Abbeel. CURL: Contrastive unsupervised representations for reinforcement learning. In _International Conference on Machine Learning_ , 2020. 

- Dong-Bok Lee, Dongchan Min, Seanie Lee, and Sung Ju Hwang. Meta-gmvae: Mixture of gaussian vae for unsupervised meta-learning. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021. 

- Sharada P Mohanty, David P Hughes, and Marcel Salath´e. Using deep learning for image-based plant disease detection. _Frontiers in plant science_ , 7:1419, 2016. 

- Jaehoon Oh, Sungnyun Kim, Namgyu Ho, Jin-Hwa Kim, Hwanjun Song, and Se-Young Yun. Understanding cross-domain few-shot learning: An experimental study. _CoRR_ , abs/2202.01339, 2022. 

- Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_ , 2018. 

12 

Published as a conference paper at ICLR 2023 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zachary Devito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada_ , pp. 8024–8035, 2019. 

- Lyle Ramshaw and Robert E Tarjan. On minimum-cost assignments in unbalanced bipartite graphs. _HP Labs, Palo Alto, CA, USA, Tech. Rep. HPL-2012-40R1_ , 20, 2012. 

- Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In _5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings_ . OpenReview.net, 2017. 

- Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, 4-9 December 2017, Long Beach, CA, USA_ , pp. 4080–4090, 2017. 

- Yee Whye Teh, Max Welling, Simon Osindero, and Geoffrey E. Hinton. Energy-based models for sparse overcomplete representations. _Journal of Machine Learning Research_ , 4:1235–1260, 2003. 

Sebastian Thrun and Lorien Pratt. _Learning to learn_ . Springer, 1998. 

- Hung-Yu Tseng, Hsin-Ying Lee, Jia-Bin Huang, and Ming-Hsuan Yang 0001. Cross-domain fewshot classification via learned feature-wise transformation. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. 

- C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. Caltech-ucsd birds 200. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 

- Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In _2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017_ , pp. 3462–3471. IEEE Computer Society, 2017. 

- Mingkai Zheng, Shan You, Fei Wang 0032, Chen Qian 0006, Changshui Zhang, Xiaogang Wang 0001, and Chang Xu 0002. Ressl: Relational self-supervised learning with weak augmentation. In _Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_ , pp. 2543–2555, 2021. 

- Bolei Zhou, Agata[`] Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba 0001. Places: A 10 million image database for scene recognition. _IEEE Trans. Pattern Anal. Mach. Intell._ , 40(6): 1452–1464, 2018. 

13 

Published as a conference paper at ICLR 2023 

## A IMPLEMENTATION DETAILS 

We train our models via stochastic gradient descent (SGD) with a batch size of _N_ = 256 for 400 epochs. Following Chen et al. (2020b); Chen & He (2021), we use an initial learning rate of 0.03 with the cosine learning schedule, _τ_ `MoCo` = 0 _._ 2, and a weight decay of 5 _×_ 10 _[−]_[4] . We use a queue size of _M_ = 16384 since Omniglot (Lake et al., 2011) and miniImageNet (Ravi & Larochelle, 2017) has roughly 100k meta-training samples. Following Lee et al. (2021), we use Conv4 and Conv5 for Omniglot and miniImageNet, respectively, for the backbone feature extractor _fθ_ . We describe the detailed architectures in Table 7. For projection and prediction MLPs, _gθ_ and _hθ_ , we use 2-layer MLPs with a hidden size of 2048 and an output dimension of 128. For the hyperparameters of PsCo, we use _τ_ `PsCo` = 1 and a momentum parameter of _m_ = 0 _._ 99 (see Appendix B for the hyperparameter sensitivity). For the number of shots during meta-learning, we use _K_ = 1 for Omniglot and _K_ = 4 for miniImageNet (see Table 6 for the sensitivity of _K_ ). We use the last-epoch model for evaluation without any guidance from the meta-validation dataset. 

Table 7: Pytorch-like architecture descriptions for standard few-shot benchmarks 

|Backbone|Layer descriptions|Output shape|
|---|---|---|
|Conv4|[Conv2d(3_×_3, 64 filter), BatchNorm2d, ReLU, MaxPool2d(2_×_2)]_×_4|64_×_2_×_2|
|Conv5|[Conv2d(3_×_3, 64 filter), BatchNorm2d, ReLU, MaxPool2d(2_×_2)]_×_5|64_×_2_×_2|



**Augmentations.** We describe the augmentations for Omniglot and miniImagenet in Table 8. For Omniglot, because it is difficult to apply many augmentations to gray-scale images, we use the same rule for weak and strong augmentations. For miniImageNet, we use only geometric transformations for the weak augmentation following Zheng et al. (2021). 

Table 8: Pytorch-like augmentation descriptions for Omniglot and miniImageNet 

|Dataset|Augmentation<br>Descriptions|
|---|---|
|Omniglot|Strong, Weak<br>RandomResizeCrop(28, scale=(0.2, 1))<br>RandomHorizontalFlip()|
|miniImagenet|Strong<br>RandomResizedCrop(84, scale=(0.2, 1))<br>RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1)<br>RandomGrayScale(p=0.2)<br>RandomHorizontalFlip()<br>Weak<br>RandomResizedCrop(84, scale=(0.2, 1))<br>RandomHorizontalFlip()|



**Training procedures.** To ensure the performance of PsCo and self-supervised learning models, we use three independently-trained models with random seeds and report the average performance of them. 

## B ANALYSIS ON HYPERPARAMETER SENSITIVITY 

For the small-scale experiments, we use a momentum of _m_ = 0 _._ 99 and a temperature of _τ_ `PsCo` = 1. We here provide more ablation experiments with varying the hyperparameters _m_ and _τ_ `PsCo` . Table 9 and 10 show the sensitivity of hyperparameters on the miniImageNet dataset. We observe that PsCo achieves good performance even for non-optimal hyperparameters. 

Table 9: Sensitivity of momentum _m_ on miniImageNet (way, shot). 

Table 10: Sensitivity of temperature _τ_ `PsCo` on miniImageNet (way, shot). 

|_m_<br>**(5, 1)**<br>**(5, 5)**<br>**(5, 20)**<br>**(5, 50)**<br>0.9<br>46.49<br>62.18<br>70.21<br>72.77<br>0.99<br>**46.70**<br>**63.26**<br>**72.22**<br>**73.50**<br>0.999<br>45.96<br>61.53<br>69.66<br>72.04|_τ_`PsCo`<br>**(5, 1)**<br>**(5, 5)**<br>**(5, 20)**<br>**(5, 50)**|
|---|---|
||0.2<br>46.43<br>62.29<br>70.04<br>72.22<br>0.5<br>46.32<br>62.63<br>70.50<br>73.15<br>1.0<br>**46.70**<br>**63.26**<br>**72.22**<br>**73.50**|



14 

Published as a conference paper at ICLR 2023 

## C EFFECT OF ADAPTATION 

We measure the performance with and without our adaptation scheme on various domains using miniImageNet-pretrained PsCo. Table 11 shows that our adaptation scheme enhances the way to adapt to each domain. In particular, the adaptation scheme is highly suggested for cross-domain few-shot classification scenarios. 

Table 11: Before and after adaptation of PsCo in few-shot classification. 

|**Adaptation**|**miniImageNet**|**CUB**|**Cars**|**Places**|**Plantae**|**CropDiseases**|**EuroSAT**|**ISIC**|**ChestX**|
|---|---|---|---|---|---|---|---|---|---|
|||||_5-way_|_5-shot_|||||
|✗|63.26|55.15|42.27|62.98|48.31|79.75|74.73|41.18|24.54|
|✓|**63.30**|**57.38**|**44.01**|**63.60**|**52.72**|**88.24**|**81.08**|**44.00**|**24.78**|
|||||_5-way 20-shot_||||||
|✗|72.22|62.35|51.02|70.85|55.91|84.72|78.96|48.53|27.60|
|✓|**73.00**|**68.58**|**57.50**|**73.95**|**64.53**|**94.95**|**87.65**|**54.59**|**27.69**|



## D SETUP FOR STANDARD FEW-SHOT BENCHMARKS 

We here describe details of benchmarks and baselines in Section D.1 and D.2, respectively, for the standard few-shot classification experiments (Section 4.1). 

## D.1 DATASETS 

**Omniglot** (Lake et al., 2011) is a 28 _×_ 28 gray-scale dataset of 1623 characters with 20 samples each. We follow the setup of unsupervised meta-learning approaches (Hsu et al., 2019). We split the dataset into 120, 100, and 323 classes for meta-training, meta-validation, and meta-test respectively. In addition, the 0, 90, 180, and 270 degrees rotated views for each class become the different categories. Thus, we have a total of 6492, 400, and 1292 classes for meta-training, meta-validation, and meta-test respectively. 

**MiniImageNet** (Ravi & Larochelle, 2017) is an 84 _×_ 84 resized subset of ILSVRC-2012 (Deng et al., 2009) with 600 samples each. We split the dataset into 64, 16, and 20 classes for metatraining, meta-validation, and meta-test respectively as introduced in Ravi & Larochelle (2017). 

## D.2 BASELINES 

We compare our performance with unsupervised meta-learning, self-supervised learning, and supervised meta-learning methods. To be specific, (a) for the unsupervised meta-learning, we use CACTUs (Hsu et al., 2019) of the best options (ACAI clustering for Omniglot and DeepCluster for miniImageNet), UMTRA (Khodadadeh et al., 2019), LASIUM (Laskin et al., 2020) of the best options (LASIUM-RO-GAN for Omniglot and LASIUM-N-GAN for miniImageNet), Meta-GMVAE (Lee et al., 2021), Meta-SVEBM (Kong et al., 2021); (b) for the self-supervised learning methods, we use SimCLR (Chen et al., 2020a), MoCo v2 (Chen et al., 2020b), and SwAV (Caron et al., 2020); (c) for the supervised meta-learning, we use the results of MAML (Finn et al., 2017) and ProtoNets (Snell et al., 2017) reported in (Hsu et al., 2019). 

For training self-supervised learning methods in our experimental setups, we use the same architecture and hyperparameters. For the hyperparameter of temperature scaling, we use the value provided in each paper: _τ_ `SimCLR` = 0 _._ 5 for SimCLR, _τ_ `MoCo` = 0 _._ 2 for MoCo v2, and _τ_ `SwAV` = 0 _._ 1 for SwAV. For evaluation, we use K-Nearest Neightobrs (K-NN) for self-supervised learning methods since their classification rules are not specified. 

15 

Published as a conference paper at ICLR 2023 

## E SETUP FOR CROSS-DOMAIN FEW-SHOT BENCHMARKS 

We now describe the setup for cross-domain few-shot benchmarks, including detailed information on datasets, baseline experiments, implementational details, and the setup for large-scale experiments. 

## E.1 DATASETS 

For the cross-domain few-shot benchmarks, we use eight different datasets. We describe the dataset information in Table 12. We use the dataset split described in Tseng et al. (2020) for the benchmark of high-similarity and we use the dataset split described in Guo et al. (2020) for the benchmark of low-similarity. Because we do not perform the meta-training procedure using the datasets of crossdomain benchmarks, we only utilize the meta-test splits on these datasets. We use the 84 _×_ 84 resized samples for evaluation on small-scale experiments. 

Table 12: Information of datasets for cross-domain few-shot benchmarks. 

|ImageNet similarity|Datset|# of classes|# of samples|
|---|---|---|---|
||CUB (Wah et al.,2011)|200|11,788|
|High|Cars (Krause et al.,2013)<br>Places (Zhou et al.,2018)|196<br>365|16,185<br>1,800,000|
||Plantae (Horn et al.,2018)|5089|675,170|
||CropDiseases (Mohanty et al.,2016)|38|43,456|
|Low|EuroSAT (Helber et al.,2019)<br>ISIC (Codella et al.,2018)|10<br>7|27,000<br>10,015|
||ChestX (Wang et al.,2017)|7|25,848|



## E.2 BASELINES 

We compare our performance with (a) previous in-domain state-of-the-art methods of unsupervised meta-learning, self-supervised learning models, and supervised meta-learning models. 

**Unsupervised meta-learning models.** We use previous in-domain state-of-the-art methods of unsupervised meta-learning models, Meta-GMVAE(Lee et al., 2021) and Meta-SVEBM (Kong et al., 2021). We use the miniImageNet pretrained parameters that the paper provided, and follow the meta-test procedure of each model to evaluate the performance. 

**Self-supervised learning models.** We use SimCLR (Chen et al., 2020a), MoCo v2 (Chen et al., 2020b), and SwAV (Caron et al., 2020) of miniImageNet pretrained parameters as our baselines. Because self-supervised learning models are pretrained on miniImageNet, we additionally fine-tune the models with a linear classifier to let the models adapt to each domain. Following the setting provided in Guo et al. (2020); Oh et al. (2022), we detach the head of the models _gθ_ and attach the linear classifier _cψ_ to the model. We freeze the base network _fθ_ while fine-tuning and only _cψ_ is learned. We fine-tune the models via SGD with an initial learning rate of 0.01, a momentum of 0.9, weight decay of 0.001, and a batch size of _N_ = 4 for 100 epochs. 

**Supervised meta-learning models.** We use MAML (Finn et al., 2017) and ProtoNets (Snell et al., 2017) of Conv5 architectures of miniImageNet pretrained. Following the procedure of Snell et al. (2017), we train the models via Adam (Kingma & Ba, 2015) with a learning rate of 0.001 and cut the learning rate in half for every training of 2000 episodes. We train them for 60K episodes and use the model of the best validation accuracy. We train them through a 5-way 5-shot, and the rest of the hyperparameters are referenced in their respective papers. We observe that their performances are similar to the performance described in Table 1. 

## E.3 EVALUATION DETAILS 

To evaluate our method, we apply our adaptation scheme. Following Section 3.3, we freeze the base network _fθ_ . We train only projection head _gθ_ and prediction head _hθ_ via SGD with an initial learning rate of 0.01, a momentum of 0.9, and weight decay of 0.001 as self-supervised learning models are fine-tuned. We only apply 50 iterations of our adaptation scheme when reporting performance. 

16 

Published as a conference paper at ICLR 2023 

## E.4 LARGE-SCALE SETUP 

Here, we describe the setup for large-scale experiments. For evaluating, we use the same protocol with the small-scale experiments, except the scale of images is 224 _×_ 224. 

**Augmentations.** For large-scale experiments, we use 224 _×_ 224-scaled data. Thus, we use similar yet slightly different augmentation schemes with small-scale experiments. Following the strong augmentation used in Chen et al. (2020b;a), we additionally apply GaussianBlur as a random augmentation. We use the same configuration for weak augmentation. For evaluation, we resize the images into 256 _×_ 256 and then apply the CenterCrop to make 224 _×_ 224 images by following Guo et al. (2020). 

**ImageNet pretraining.** We pretrain MoCo v2 (Chen et al., 2020b), BYOL (Grill et al., 2020), and our PsCo of ResNet-18/50 (He et al., 2016) via SGD with a batch size of _N_ = 256 for 200 epochs. Following (Chen et al., 2020b; Chen & He, 2021), we use an initial learning rate of 0.03 with the cosine learning schedule, _τ_ MoCo = 0 _._ 2 and a weight decay of 0.0001. We use a queue size of _M_ = 65536 and momentum of _m_ = 0 _._ 999. For the parameters of PsCo, we use _τ_ PsCo = 0 _._ 2 and _K_ = 16 as the queue is 4 times bigger. For supervised pretraining, we use the the model checkpoint officially provided by torchvision (Paszke et al., 2019). 

17 

Published as a conference paper at ICLR 2023 

## F EXPERIMENTAL RESULTS WITH 95% CONFIDENCE INTERVAL 

We here provide the experimental results of Table 1, 2, and 3 with 95% confidence intervals in Table 13, 14, and 15, respectively. 

Table 13: Few-shot classification accuracy (%) on Omniglot and miniImageNet with a 95% confidence interval over 2000 few-shot tasks. 

|**Method**|**Omniglot (way, shot)**<br>**(5, 1)**<br>**(5, 5)**<br>**(20, 1)**<br>**(20, 5)**|**miniImageNet (way, shot)**<br>**(5, 1)**<br>**(5, 5)**<br>**(5, 20)**<br>**(5, 50)**|
|---|---|---|
|SimCLR<br>MoCo v2<br>SwAV|92.13_±_0.30<br>97.06_±_0.13<br>80.95_±_0.21<br>91.60_±_0.12<br>92.66_±_0.28<br>97.38_±_0.12<br>82,13_±_0.21<br>92.34_±_0.11<br>93.13_±_0.27<br>97.32_±_0.13<br>82.63_±_0.21<br>92.12_±_0.12|43.35_±_0.42<br>52.50_±_0.39<br>61.83_±_0.35<br>64.85_±_0.32<br>41.92_±_0.41<br>50.94_±_0.38<br>60.23_±_0.35<br>63.45_±_0.33<br>43.24_±_0.42<br>52.41_±_0.39<br>61.36_±_0.35<br>64.52_±_0.33|
|PsCo (ours)|**96.37**_±_0.20<br>**99.13**_±_0.07<br>**89.60**_±_0.17<br>**97.07**_±_0.07|**46.70**_±_0.42<br>**63.26**_±_0.37<br>**72.22**_±_0.32<br>**73.50**_±_0.29|



Table 14: Few-shot classification accuracy (%) on cross-domain few-shot classification benchmarks of Conv5 pretrained on miniImageNet with a 95% confidence interval over 2000 few-shot tasks. 

||(a) Cross-domain few-shot benchmarks similar to miniImageNet.|(a) Cross-domain few-shot benchmarks similar to miniImageNet.|(a) Cross-domain few-shot benchmarks similar to miniImageNet.|(a) Cross-domain few-shot benchmarks similar to miniImageNet.|
|---|---|---|---|---|
|**Method**|**CUB**<br>**(5, 5)**<br>**(5, 20)**|**Cars**<br>**(5, 5)**<br>**(5, 20)**|**Places**<br>**(5, 5)**<br>**(5, 20)**|**Plantae**<br>**(5, 5)**<br>**(5, 20)**|
|Meta-GMVAE<br>Meta-SVEBM|47.48_±_0.47<br>54.08_±_0.45<br>45.50_±_0.83<br>54.61_±_0.91|31.39_±_0.34<br>38.36_±_0.35<br>34.27_±_0.79<br>46.23_±_0.87|57.70_±_0.47<br>65.08_±_0.38<br>51.27_±_0.82<br>61.09_±_0.85|38.27_±_0.40<br>45.02_±_0.37<br>38.12_±_0.86<br>46.22_±_0.85|
|SimCLR<br>MoCo v2<br>SwAV|52.11_±_0.45<br>61.89_±_0.45<br>53.23_±_0.45<br>62.81_±_0.45<br>51.58_±_0.45<br>61.38_±_0.46|37.40_±_0.35<br>50.05_±_0.39<br>38.65_±_0.35<br>51.77_±_0.39<br>36.85_±_0.33<br>50.03_±_0.38|60.10_±_0.40<br>69.93_±_0.35<br>59.09_±_0.40<br>69.08_±_0.36<br>59.57_±_0.40<br>69.70_±_0.36|43.42_±_0.37<br>54.92_±_0.36<br>43.97_±_0.37<br>55.45_±_0.36<br>42.68_±_0.37<br>54.03_±_0.36|
|PsCo (ours)|**57.38**_±_0.44<br>**68.58**_±_0.41|**44.01**_±_0.39<br>**57.50**_±_0.40|**63.60**_±_0.41<br>**73.95**_±_0.36|**52.72**_±_0.39<br>**64.53**_±_0.36|
|_MAML_<br>_ProtoNets_|56.57_±_0.43<br>64.17_±_0.40<br>56.74_±_0.43<br>65.03_±_0.41|41.17_±_0.40<br>48.82_±_0.40<br>38.98_±_0.37<br>47.98_±_0.38|60.05_±_0.42<br>67.54_±_0.37<br>59.39_±_0.40<br>67.77_±_0.36|47.33_±_0.41<br>54.86_±_0.38<br>45.89_±_0.40<br>54.29_±_0.38|
||(b) Cross-domain|few-shot benchmarks dissimilar to miniImageNet.|||
|**Method**|**CropDiseases**<br>**(5, 5)**<br>**(5, 20)**|**EuroSAT**<br>**(5, 5)**<br>**(5, 20)**|**ISIC**<br>**(5, 5)**<br>**(5, 20)**|**ChestX**<br>**(5, 5)**<br>**(5, 20)**|
|Meta-GMVAE<br>Meta-SVEBM|73.56_±_0.53<br>81.22_±_0.39<br>71.82_±_1.03<br>83.13_±_0.78|73.83_±_0.42<br>80.11_±_0.35<br>70.83_±_0.83<br>80.21_±_0.73|33.48_±_0.30<br>39.48_±_0.28<br>38.85_±_0.76<br>48.43_±_0.81|23.23_±_0.23<br>26.26_±_0.24<br>**26.26**_±_0.65<br>28.91_±_0.69|
|SimCLR<br>MoCo v2<br>SwAV|79.90_±_0.39<br>88.73_±_0.28<br>80.96_±_0.37<br>89.85_±_0.27<br>80.15_±_0.39<br>89.24_±_0.28|79.14_±_0.38<br>85.05_±_0.32<br>79.94_±_0.37<br>86.16_±_0.31<br>79.31_±_0.39<br>85.62_±_0.31|42.83_±_0.29<br>51.35_±_0.27<br>43.43_±_0.30<br>52.14_±_0.27<br>43.21_±_0.30<br>51.99_±_0.27|25.14_±_0.23<br>**29.21**_±_0.24<br>25.24_±_0.23<br>29.19_±_0.24<br>24.99_±_0.23<br>28.57_±_0.24|
|PsCo (ours)|**88.24**_±_0.31<br>**94.95**_±_0.18|**81.08**_±_0.35<br>**87.65**_±_0.28|**44.00**_±_0.30<br>**54.59**_±_0.29|24.78_±_0.23<br>27.69_±_0.23|
|_MAML_<br>_ProtoNets_|77.76_±_0.39<br>83.24_±_0.34<br>76.01_±_0.40<br>83.64_±_0.33|71.48_±_0.38<br>76.70_±_0.33<br>64.91_±_0.38<br>70.88_±_0.33|47.34_±_0.37<br>55.09_±_0.34<br>40.62_±_0.31<br>48.38_±_0.29|22.61_±_0.22<br>24.25_±_0.22<br>23.15_±_0.22<br>25.72_±_0.23|



Table 15: Few-shot classification accuracy (%) on cross-domain few-shot classification benchmarks of pretrained ResNet-18/50 on ImageNet with a 95% confidence interval (5-way 5-shot). 

|**Methods**|**CUB**|**Cars**|**Places**|**Plantae**|**CropDiseases**|**EuroSAT**|**ISIC**|**ChestX**|
|---|---|---|---|---|---|---|---|---|
|_ResNet-18 pretrained_|||||||||
|MoCo v2|61.88_±_0.96|46.42_±_0.73|79.11_±_0.68|56.24_±_0.72|81.48_±_0.74|75.98_±_0.73|38.21_±_0.53|24.34_±_0.36|
|**+PsCo (Ours)**|**70.08**_±_0.87|**50.73**_±_0.76|**79.74**_±_0.64|**61.55**_±_0.76|**87.91**_±_0.57|**79.92**_±_0.64|**40.61**_±_0.52|**25.03**_±_0.42|
|_ResNet-50 pretrained_|||||||||
|MoCo v2|64.16_±_0.91|47.67_±_0.75|81.39_±_0.64|61.36_±_0.79|82.89_±_0.77|76.96_±_0.68|38.26_±_0.56|**24.28**_±_0.39|
|**+PsCo (Ours)**|**76.63**_±_0.84|**53.45**_±_0.76|**83.87**_±_0.58|**69.17**_±_0.70|**89.85**_±_0.78|**83.99**_±_0.52|**41.64**_±_0.55|23.60_±_0.36|
|BYOL|67.45_±_0.88|45.74_±_0.76|75.43_±_0.79|56.86_±_0.84|80.82_±_0.86|77.70_±_0.71|37.27_±_0.56|24.15_±_0.36|
|**+PsCo (Ours)**|**82.13**_±_0.70|**56.19**_±_0.76|**83.80**_±_0.62|**71.14**_±_0.71|**92.92**_±_0.44|**85.33**_±_0.54|**42.90**_±_0.55|**26.05**_±_0.46|
|Supervised|89.13_±_0.55|75.15_±_0.75|84.41_±_0.61|72.91_±_0.73|90.96_±_0.48|85.64_±_0.52|43.34_±_0.57|25.35_±_0.41|



18 

