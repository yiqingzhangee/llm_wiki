Published as a conference paper at ICLR 2023 

## - STUNT: FEW SHOT TABULAR LEARNING WITH - SELF GENERATED TASKS FROM UNLABELED TABLES 

**Jaehyun Nam**[1] **Jihoon Tack**[1] **Kyungmin Lee**[1] **Hankook Lee**[2] _[∗]_ **Jinwoo Shin**[1] 1Korea Advanced Institute of Science and Technology (KAIST) 2LG AI Research _{_ `jaehyun` _._ `nam` _,_ `jihoontack` _,_ `kyungmnlee` _,_ `jinwoos` _}_ @ `kaist` _._ `ac` _._ `kr hankook` _._ `lee` @ `lgresearch` _._ `ai` 

## ABSTRACT 

Learning with few labeled tabular samples is often an essential requirement for industrial machine learning applications as varieties of tabular data suffer from high annotation costs or have difficulties in collecting new samples for novel tasks. Despite the utter importance, such a problem is quite under-explored in the field of tabular learning, and existing few-shot learning schemes from other domains are not straightforward to apply, mainly due to the heterogeneous characteristics of tabular data. In this paper, we propose a simple yet effective framework for few-shot semi-supervised tabular learning, coined _Self-generated Tasks from UNlabeled Tables (STUNT)_ . Our key idea is to self-generate diverse few-shot tasks by treating randomly chosen columns as a target label. We then employ a meta-learning scheme to learn generalizable knowledge with the constructed tasks. Moreover, we introduce an unsupervised validation scheme for hyperparameter search (and early stopping) by generating a pseudo-validation set using STUNT from unlabeled data. Our experimental results demonstrate that our simple framework brings significant performance gain under various tabular few-shot learning benchmarks, compared to prior semi- and self-supervised baselines. Code is available at https://github.com/jaehyun513/STUNT. 

## 1 INTRODUCTION 

Learning with few labeled samples is often an essential ingredient of machine learning applications for practical deployment. However, while various few-shot learning schemes have been actively developed over several domains, including images (Chen et al., 2019) and languages (Min et al., 2022), such research has been under-explored in the tabular domain despite its practical importance in industries (Guo et al., 2017; Zhang et al., 2020; Ulmer et al., 2020). In particular, few-shot tabular learning is a crucial application as varieties of tabular datasets (i) suffer from high labeling costs, e.g., the credit risk in financial datasets (Clements et al., 2020), and (ii) even show difficulties in collecting new samples for novel tasks, e.g., a patient with a rare or new disease (Peplow, 2016) such as an early infected patient of COVID-19 (Zhou et al., 2020). 

To tackle such limited label issues, a common consensus across various domains is to utilize unlabeled datasets for learning a generalizable and transferable representation, e.g., images (Chen et al., 2020a) and languages (Radford et al., 2019). Especially, prior works have shown that representations learned with self-supervised learning are notably effective when fine-tuned or jointly learned with few labeled samples (Tian et al., 2020; Perez et al., 2021; Lee et al., 2021b; Lee & Shin, 2022). However, contrary to the conventional belief, we find this may not hold for tabular domains. For instance, recent state-of-the-art self-supervised tabular learning methods (Yoon et al., 2020; Ucar et al., 2021) do not bring meaningful performance gains over even a simple k-nearest neighbor (kNN) classifier for few-shot tabular learning in our experiments (see Table 1 for more details). We hypothesize that this is because the gap between trained self-supervised tasks and the applied few-shot task is large due to the heterogeneous characteristics of tabular data. 

Instead, we ask whether one can utilize the power of meta-learning to reduce the gap via fast adaption to unseen few-shot tasks; meta-learning is indeed one of the most effective few-shot learning 

> *Work done at KAIST. 

1 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_STUNT_Few-shot_Tabular_Learning_with_Self-generated_Tasks_from_Unlabeled_Tables_images/iclr_2023_STUNT_Few-shot_Tabular_Learning_with_Self-generated_Tasks_from_Unlabeled_Tables.pdf-0002-01.png)


**----- Start of picture text -----**<br>
x x ˜ y ˜<br>Select random columns Run k-means clustering Self-generated task<br>**----- End of picture text -----**<br>


Figure 1: An overview of the proposed _Self-generated Tasks from UNlabeled Tables (STUNT)_ : we generate the task label by running a k-means clustering over the randomly selected column features of the table, then perturb the selected columns to prevent from generating a trivial task. 

strategies across domains (Finn et al., 2017; Gu et al., 2018; Xie et al., 2018). We draw inspiration from the recent success in unsupervised meta-learning literature, which meta-learns over the self-generated tasks from unlabeled data to train an effective few-shot learner (Khodadadeh et al., 2019; Lee et al., 2021a). It turns out that such an approach is quite a promising direction for fewshot tabular learning: a recent unsupervised meta-learning scheme (Hsu et al., 2018) outperforms the self-supervised tabular learning methods in few-shot tabular classification in our experiments (see Table 1). In this paper, we suggest to further exploit the benefits of unsupervised meta-learning into few-shot tabular learning by generating more diverse and effective tasks compared to the prior works using the distinct characteristic of the tabular dataset’s column feature. 

**Contribution.** We propose a simple yet effective framework for few-shot semi-supervised tabular learning, coined _Self-generated Tasks from UNlabeled Tables_ (STUNT); see the overview in Figure 1. Our key idea is to generate a diverse set of tasks from the unlabeled tabular data by treating the table’s column feature as a useful target, e.g., the ‘blood sugar’ value can be used as a substituted label for ‘diabetes’. Specifically, we generate pseudo-labels of the given unlabeled input by running a k-means clustering on randomly chosen subsets of columns. Moreover, to prevent generating a trivial task (as the task label can be directly inferred by the input columns), we randomly replace the chosen column features with a random value sampled from the columns’ respective empirical marginal distributions. We then apply a meta-learning scheme, i.e., Prototypical Network (Snell et al., 2017), to learn generalizable knowledge with the self-generated tasks. 

We also find that the major difficulty of the proposed meta-learning with unlabeled tabular datasets is the absence of a labeled validation set; the training is quite sensitive to hyperparameter selection or even suffers from overfitting. To this end, we propose an unsupervised validation scheme by utilizing STUNT to the unlabeled set. We find that the proposed technique is highly effective for hyperparameter searching (and early stopping), where the accuracy of the pseudo-validation set and the test set show a high correlation. 

We verify the effectiveness of STUNT through extensive evaluations on various datasets in the OpenML-CC18 benchmark (Vanschoren et al., 2014; Bischl et al., 2021). Overall, our experimental results demonstrate that STUNT consistently and significantly outperforms the prior methods, including unsupervised meta-learning (Hsu et al., 2018), semi- and self-supervised learning schemes (Tarvainen & Valpola, 2017; Yoon et al., 2020; Ucar et al., 2021) under few-shot semisupervised learning scenarios. In particular, our method improves the average test accuracy from 59.89% _→_ 63.88% for 1-shot and from 72.19% _→_ 74.77% for 5-shot, compared to the best baseline. Furthermore, we show that STUNT is effective for multi-task learning scenarios where it can adapt to new tasks without retraining the network. 

## 2 RELATED WORK 

**Learning with few labeled samples.** To learn an effective representation with few labeled samples, prior works suggest leveraging the unlabeled samples. Such works can be roughly categorized as (i) semi-supervised (Kim et al., 2020; Assran et al., 2021) and (ii) self-supervised (Chen et al., 2020a;b) 

2 

Published as a conference paper at ICLR 2023 

approaches. For semi-supervised learning approaches, a common way is to produce a pseudo-label for each unlabeled data by using the model’s prediction and then train the model with the corresponding pseudo-label (Lee, 2013). More advanced schemes utilize the momentum network (Laine & Aila, 2017; Tarvainen & Valpola, 2017) and consistency regularization with data augmentations (Berthelot et al., 2019; Sohn et al., 2020) to generate a better pseudo-label. On the other hand, selfsupervised learning schemes aim to pre-train the representation by using domain-specific inductive biases (e.g., the spatial relationship of image augmentations), then fine-tune or adapt with a few labeled samples (Tian et al., 2020; Perez et al., 2021). In particular, previous studies have shown the effectiveness of self-supervised learning in the transductive setting compared to the few-shot methods (Chen et al., 2021). While recent works on both approaches heavily rely on augmentation schemes, it is unclear how to extend such methods to the tabular domain due to the heterogeneous characteristics of tabular datasets, i.e., there is no clear consensus on which augmentation is globally useful for the tabular dataset. Instead, we train the unlabeled dataset with an unsupervised meta-learning framework that does not rely on the effect of augmentation. 

**Learning with unlabeled tabular data.** Various attempts have been made to train a generalizable representation for tabular datasets using unlabeled samples. Yoon et al. (2020) is the first to target self-supervised learning on the tabular dataset by corrupting random features and then predicting the corrupted location (i.e., row and columns). As a follow-up, Bahri et al. (2022) directly adopts contrastive learning frameworks (Chen et al., 2020a) by corrupting randomly selected features to create positive views, and Ucar et al. (2021) shows that using effective three pretext task losses (i.e., reconstruction loss, contrastive loss, and distance loss) can achieve the state-of-the-art performance on the linear evaluation task (training a linear classifier with a labeled dataset upon the learned representation). However, while prior works have shown their effectiveness mainly on linear evaluation, we find that these methods may not be effective for few-shot learning scenarios (see Table 1). In this regard, we suggest utilizing the power of meta-learning for training an effective few-shot learner by proposing an unsupervised meta-learning framework for tabular data. Moreover, while some works have proposed a self-supervised learning framework (Somepalli et al., 2021; Majmundar et al., 2022) based on Transformer architectures (Vaswani et al., 2017), we believe a new approach is needed for few-shot learning (given the observations of prior self-supervised learning works in Table 1) and introduce an architecture-agnostic method that can be used for a wide range of applications. 

**Unsupervised meta-learning.** Meta-learning, i.e., learning to learn by extracting common knowledge over a task distribution, has emerged as a popular paradigm for enabling systems to adapt to new tasks in a sample-efficient way (Vinyals et al., 2016; Finn et al., 2017; Snell et al., 2017). Recently, several works have suggested unsupervised meta-learning schemes that self-generate a labeled task from the unlabeled dataset, as various few-shot learning applications suffer from high annotation costs. To self-generate the tasks, CACTUs (Hsu et al., 2018) run a clustering algorithm on a representation trained with self-supervised learning, Ye et al. (2022) and UMTRA (Khodadadeh et al., 2019) assumes the augmented sample as the same pseudo-class (Ye et al. (2022) also introduces effective strategies for unsupervised meta-learning: sufficient episodic sampling, hard mixed supports, and task specific projection head), and Meta-GMVAE (Lee et al., 2021a) utilize a variational autoencoder (VAE; Kingma & Welling (2014)) for clustering. However, despite the effectiveness of prior works on image datasets, we find that applying each method to the tabular domain is highly non-trivial; they assume high-quality self-supervised representation, data augmentations, or sophisticated generative models like VAE which are quite cumbersome to have in the tabular domain, e.g., see the work by Xu et al. (2019). Nevertheless, we have tried the prior methods for tabular learning using various techniques, e.g., several augmentations such as noise and permutation, but only CACTUs has shown its effectiveness; we find that Meta-GMVAE underperforms the baseline with the lowest performance that we consider. In this paper, we propose a new unsupervised meta-learning that is specialized for tabular datasets. 

## 3 STUNT: SELF-GENERATED TASKS FROM UNLABELED TABLES 

In this section, we develop an effective few-shot tabular learning framework that utilizes the power of meta-learning in an unsupervised manner. In a nutshell, our framework meta-learns over the self-generated tasks from the unlabeled tabular dataset, then adapt the network to classify the fewshot labeled dataset. We first briefly describe our problem setup (Section 3.1), and then the core component, coined _Self-generated tasks from UNlabeled Tables (STUNT)_ , which generates effective 

3 

Published as a conference paper at ICLR 2023 

and diverse tasks from the unlabeled tabular data (Section 3.2). Moreover, we introduce a pseudovalidation scheme with STUNT, where one can tune hyperparameters (and apply early stopping) even without a labeled validation set on few-shot scenarios (Section 3.3). 

## 3.1 PROBLEM SETUP: FEW-SHOT SEMI-SUPERVISED LEARNING 

We first describe the problem setup of our interest, the few-shot semi-supervised learning for classification. Formally, our goal is to train a neural network classifier _fθ_ : _X →Y_ parameterized by _θ_ where _X ⊆_ R _[d]_ and _Y_ = _{_ 0 _,_ 1 _}[C]_ are input and label spaces with _C_ classes, respectively, and assume that we have a labeled dataset _Dl_ = _{_ **x** _l,i,_ **y** _l,i}[N] i_ =1 _[l][⊆X][×][Y]_[and][an][unlabeled][dataset] _Du_ = _{_ **x** _u,i}[N] i_ =1 _[u][⊆X]_[for training the classifier] _[ f][θ]_[. Note that all the data points are sampled from a] certain data-generating distribution _p_ ( **x** _,_ **y** ) in an _i.i.d._ manner. We also assume that the cardinality of the given labeled set is very small, e.g., one sample per class, while we have a sufficient amount of the unlabeled dataset, i.e., _Nu ≫ Nl_ . 

## 3.2 UNSUPERVISED META-LEARNING WITH STUNT 

We now describe the core algorithm we propose, STUNT. To obtain a good classifier _fθ_ under the proposed setup, we suggest using an unsupervised meta-learning method, which (i) self-generates diverse tasks _{T_ 1 _, T_ 2 _, . . .}_ from the unlabeled dataset _Du_ where each _Ti_ contains few samples with pseudo-labels; (ii) meta-learns _fθ_ to generalize across the tasks; and, (iii) adapts the classifier _fθ_ using the labeled dataset _Dl_ . Algorithm 1 in Appendix D provides the detailed training process. 

**Task generation from unlabeled tables.** Our key idea is to generate a diverse set of tasks from the unlabeled data by treating a column feature of tabular data as a useful pseudo-label. Intuitively speaking, as any label type can be considered as a tabular column due to the heterogeneous characteristic of the tabular data (i.e., each column has a distinct feature), one can also rethink any column feature as a task label. In particular, since there exist some columns that have a high correlation with the original label, the new task constructed with such a column feature is highly similar to the original classi- 

|**BMI**<br>**Age**<br>43.1<br>30.5<br>33<br>53<br>Positive<br>Positive<br>28.1<br>21<br>Negative<br>…<br>…<br>…<br>**Diabetes**<br>137<br>197<br>89<br>…<br>**Blood Sugar**<br>**x**<br>**y**|**BMI**<br>**Age**<br>43.1<br>30.5<br>33<br>53<br>Positive<br>Positive<br>28.1<br>21<br>Negative<br>…<br>…<br>…<br>**Diabetes**<br>137<br>197<br>89<br>…<br>**Blood Sugar**<br>**x**<br>**y**|**BMI**<br>**Age**<br>43.1<br>30.5<br>33<br>53<br>Positive<br>Positive<br>28.1<br>21<br>Negative<br>…<br>…<br>…<br>**Diabetes**<br>137<br>197<br>89<br>…<br>**Blood Sugar**<br>**x**<br>**y**|**BMI**<br>**Age**<br>43.1<br>30.5<br>33<br>53<br>Positive<br>Positive<br>28.1<br>21<br>Negative<br>…<br>…<br>…<br>**Diabetes**<br>137<br>197<br>89<br>…<br>**Blood Sugar**<br>**x**<br>**y**|
|---|---|---|---|
|89|28.1|21|Negative|



Figure 2: Example data from the diabetes dataset (Bischl et al., 2021). The red column indicates the original target, and the green column indicates the possible alternative label. 

fication task, e.g., the original task of predicting ‘Diabetes’ through ‘BMI’ and ‘Age’ is similar to a new task that predicts ‘Blood Sugar’ by ‘BMI’ and ‘Age’ (see Figure 2). With this intuition, we generate pseudo-labels by running a k-means clustering over the randomly selected columns to improve the diversity and the possibility of sampling highly correlated columns (with the original label). 

Formally, to generate a single task _T_ `STUNT` , we sample the masking ratio _p_ from the uniform distribution _U_ ( _r_ 1 _, r_ 2), where _r_ 1 and _r_ 2 are hyperparameters with 0 _< r_ 1 _< r_ 2 _<_ 1, and generate a random binary mask **m** := [ _m_ 1 _, . . . , md_ ] _[⊤] ∈{_ 0 _,_ 1 _}[d]_ where[�] _i[m][i]_[=] _[ ⌊][dp][⌋]_[and] _[ ⌊·⌋]_[is the floor function, i.e.,] the greatest integer not exceeding the input. Then, for a given unlabeled data _Du_ , we select columns with the mask index with the value of one, i.e., `sq` ( **x** _⊙_ **m** ) _∈_ R _[⌊][dp][⌋]_ where _⊙_ is the element-wise product andzero. Based on the selected columns, we run a k-means clustering to generate the task label `sq` ( **x** _⊙_ **m** ) is a squeezing operation that removes the elements with the mask value of **y** ˜ _u,i_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_STUNT_Few-shot_Tabular_Learning_with_Self-generated_Tasks_from_Unlabeled_Tables_images/iclr_2023_STUNT_Few-shot_Tabular_Learning_with_Self-generated_Tasks_from_Unlabeled_Tables.pdf-0004-11.png)


where _k_ is the number of centroids, **1** _k ∈_ R _[k]_ is a vector of ones, and **C** is the centroid matrix. 

Since the task label **y** ˜ _u_ is generated from the data itself, the classifier can easily infer the label from bythe given clean data **x** ˜ _u_ := **m** _⊙_ **x** ˆ _u_ + (1 **x** _u_ . To prevent such an issue, we suggest perturbing the selected column features _−_ **m** ) _⊙_ **x** _u_ where each column feature element of **x** ˆ _u_ is sampled from the empirical marginal distribution of each column feature. Finally, the generated task from STUNT is defined as follows: _T_ `STUNT` := _{_ **x** ˜ _u,i,_ ˜ **y** _u,i}[N] i_ =1 _[u]_[.] 

4 

Published as a conference paper at ICLR 2023 

**Meta-learning with STUNT.** Based on the generated task, we suggest to meta-learn the network by utilizing Prototypical Network (ProtoNet; Snell et al. (2017)): performs a non-parametric classifier on top of the network’s embedding space. Specifically, ProtoNet learns this embedding space in which classification can be performed by computing distances to prototype vectors of each class, i.e., the average embedding vector of the class samples. The reason why we consider ProtoNet as a meta-learning framework is three-fold. First, the number of classes can be different for training and inference on ProtoNet, allows us to search the effective centroid number _k_ rather than fixing it to the class size _C_ . Second, the method is model- and data-agnostic, where it can be used for the tabular domain without any modification. Finally, despite the simplicity, ProtoNet is known to outperform advanced meta-learning schemes under various datasets (Ye et al., 2020; Tack et al., 2022). 

For a given task _T_ `STUNT` , we sample two disjoint sets from _T_ `STUNT` , i.e., _S_ and _Q_ , which are used for constructing the classifier, and training the constructed classifier, respectively. Concretely, we construct the ProtoNet classifier _fθ_ over the parameterized embedding _zθ_ : _X →_ R _[D]_ by using the prototype vectors of each pseudo-class **p** _c_ ˜ := _|S_ 1 _c_ ˜ _|_ �(˜ **x** _u,_ **y** ˜ _u_ ) _∈Sc_ ˜ _[z][θ]_[(˜] **[x]** _[u]_[)][ where] _[ S][c]_[˜][contains samples] with pseudo-class ˜ _c_ in _S_ : 

_fθ_ ( _y_ = _c_ ˜ _|_ **x** ; _S_ ) = exp( _−∥zθ_ ( **x** ) _−_ **p** _c_ ˜ _∥_ 2) (2) � _c_ ˜ _[′]_[exp(] _[−∥][z][θ]_[(] **[x]**[)] _[ −]_ **[p]** _c_ ˜ _[′][∥]_[2][)] _[.]_ We then compute the cross-entropy loss _L_ `CE` on the conducted classifier _fθ_ with set _Q_ , i.e., _L_ `meta` ( _θ, Q_ ) :=[�] (˜ **x** _u,_ **y** ˜ _u_ ) _∈Q[L]_ `[CE]` � _fθ_ (˜ **x** _u_ ; _S_ ) _,_ ˜ **y** _u_ �, where we train the network to minimize the metalearning loss _L_ `meta` over the diverse set of tasks _{T_ `STUNT` _,_ 1 _, T_ `STUNT` _,_ 2 _, . . . }_ . 

**Adapting with labeled samples.** After meta-learning the parameter _θ_ with self-generated tasks, we use the labeled set _Dl_ to construct the classifier for the few-shot classification by using ProtoNet, i.e., _fθ_ ( _·_ ; _Dl_ ) where the each prototype vector **p** _c_ is computed with samples of the label _c_ in _Dl_ . 

## 3.3 PSEUDO-VALIDATION WITH STUNT 

We find that the difficulty of the proposed unsupervised learning is the absence of a validation set for selecting the hyperparameters and early stopping the training. To tackle this issue, we introduce an unsupervised validation scheme where we generate a pseudo-validation set by running STUNT on the unlabeled set. Here, rather than sampling the columns for generating the cluster, we use all column features to remove the randomness throughout the validation and further use the clean tabular input contrary to the perturbed sample as in the original STUNT. 

Formally, we sample a certain portion of the unlabeled set _Du_ `[val]` _⊂Du_ , then generate the task label **y** _u_ `[val]` by running a k-means clustering over clean samples **x** `[val]` _u ∈Du_ where _k_ = _C_ , i.e., Eq. (1) with **m** = **1** _d_ . Then, for a given validation task _T_ `STUNT[val]`[=] _[ {]_ **[x]** `[val]` _u,i[,]_ **[ y]** _u,i_ `[val]` _[}][i]_[, we sample two disjoint sets,] _S_ `[val]` and _Q_ `[val]` , to evaluate the pseudo-validation performance of the ProtoNet classifier _fθ_ ( _·_ ; _S_ `[val]` ) by using Eq. (2), i.e., predicting pseudo-class of _Q_ `[val]` using prototype vectors made from _S_ `[val]` . 

## 4 EXPERIMENTS 

In this section, we validate the effectiveness of our method on few-shot tabular learning scenarios under various tabular datasets from the OpenML-CC18 benchmark (Bischl et al., 2021). Our results exhibit that STUNT consistently and significantly outperforms other methods, including unsupervised, semi- and self-supervised methods (Section 4.1). We further demonstrate that our method is even effective for few-shot multi-task learning (Section 4.2). Finally, we perform an ablation study to verify the effect of the proposed pseudo-validation scheme of our approach (Section 4.3). 

**Common setup.** For all the datasets, 80% of the data is used for training (unlabeled except for few-shot labeled samples) and 20% for testing, except for the income dataset, since split training and test data are provided. For STUNT, we use 20% of training data for pseudo-validation. We one-hot encode categorical features following the preprocessing of SubTab (Ucar et al., 2021) then apply normalization by subtracting the mean and dividing by the standard deviation for the income dataset and min-max scaling for other datasets, respectively. All baselines and STUNT are trained for 10K steps, while we follow the original training setting for CACTUs (Hsu et al., 2018). For all methods, we train a 2-layer multi-layer perceptron (MLP) with a hidden dimension of 1024. We provide additional information in the Appendix A. 

5 

Published as a conference paper at ICLR 2023 

Table 1: Few-shot test accuracy (%) on 8 datasets from the OpenML-CC18 benchmark (Bischl et al., 2021). We report the mean test accuracy over 100 different seeds. Checkmark ✓ indicates the use of 100 additional labeled samples for validation (Val.), including hyperparameter searching and early stopping. The bold denotes the highest mean score. 

|Type<br>Method|Val.|income<br>cmc<br>karhunen<br>optdigit<br>diabetes<br>semeion<br>pixel<br>dna|Avg.|
|---|---|---|---|
|# shot=1||||
|CatBoost<br>MLP<br>LR<br>Sup.<br>kNN|✓<br>✓<br>✓<br>-|57.00<br>34.60<br>55.67<br>61.32<br>60.02<br>43.21<br>59.16<br>41.35<br>60.52<br>35.06<br>48.67<br>61.02<br>57.25<br>40.88<br>55.62<br>44.39<br>59.64<br>35.08<br>55.05<br>65.19<br>57.61<br>42.90<br>59.71<br>44.28<br>61.22<br>34.99<br>54.42<br>65.58<br>58.56<br>44.35<br>61.48<br>42.67|52.06<br>50.43<br>52.43<br>52.82|
|Mean Teacher<br>ICT<br>Pseudo-Label<br>MPL<br>Semi-sup.<br>VIME-Semi|✓<br>✓<br>✓<br>✓<br>✓|60.63<br>35.58<br>54.57<br>66.10<br>58.05<br>43.56<br>61.02<br>46.58<br>61.83<br>36.53<br>58.37<br>69.12<br>58.08<br>43.48<br>60.88<br>46.55<br>60.52<br>34.97<br>49.44<br>61.50<br>57.03<br>41.42<br>56.12<br>44.26<br>60.85<br>35.13<br>47.66<br>61.52<br>57.39<br>41.82<br>56.01<br>44.22<br>56.40<br>32.97<br>57.40<br>66.85<br>58.16<br>40.43<br>52.86<br>39.18|53.26<br>54.36<br>50.66<br>50.58<br>50.53|
|SubTab + Fine-tune<br>SubTab + LR<br>SubTab + kNN<br>VIME + Fine-tune<br>VIME + LR<br>Self-sup.<br>VIME + kNN|✓<br>✓<br>-<br>✓<br>✓<br>-|59.74<br>35.65<br>41.11<br>49.88<br>59.35<br>30.49<br>42.23<br>40.86<br>61.88<br>35.68<br>50.32<br>67.05<br>58.06<br>40.27<br>60.40<br>45.68<br>61.58<br>35.87<br>48.74<br>66.05<br>59.22<br>39.99<br>61.30<br>44.16<br>60.50<br>34.98<br>47.50<br>61.31<br>57.23<br>41.09<br>53.79<br>44.30<br>61.99<br>35.30<br>59.62<br>70.52<br>56.95<br>47.20<br>64.17<br>51.36<br>62.16<br>35.55<br>58.56<br>69.31<br>58.35<br>46.99<br>64.62<br>50.29|44.91<br>52.42<br>52.36<br>50.09<br>55.89<br>55.78|
|UMTRA<br>SES<br>CACTUs<br>Unsup.-Meta.<br>**STUNT (Ours)**|-<br>-<br>-<br>-|57.23<br>35.46<br>49.05<br>49.87<br>57.64<br>26.33<br>34.26<br>25.13<br>56.39<br>34.59<br>49.19<br>56.30<br>59.97<br>33.73<br>49.19<br>39.56<br>**64.02**<br>36.10<br>65.59<br>71.98<br>58.92<br>48.96<br>67.61<br>65.93<br>63.52<br>**37.10**<br>**71.20**<br>**76.94**<br>**61.08**<br>**55.91**<br>**79.05**<br>**66.20**|41.87<br>47.37<br>59.89<br>**63.88**|
|# shot=5||||
|CatBoost<br>MLP<br>LR<br>Sup.<br>kNN|✓<br>✓<br>✓<br>-|64.51<br>39.75<br>82.38<br>84.05<br>65.75<br>68.69<br>84.49<br>63.46<br>66.25<br>37.40<br>77.56<br>83.30<br>64.32<br>66.25<br>81.97<br>59.73<br>66.53<br>37.15<br>81.02<br>86.22<br>64.19<br>67.87<br>85.02<br>58.88<br>70.49<br>38.56<br>79.98<br>84.89<br>67.32<br>68.33<br>84.02<br>61.45|69.14<br>67.10<br>68.36<br>69.38|
|Mean Teacher<br>ICT<br>Pseudo-Label<br>MPL<br>Semi-sup.<br>VIME-Semi|✓<br>✓<br>✓<br>✓<br>✓|67.05<br>37.73<br>81.08<br>86.66<br>65.45<br>69.67<br>85.24<br>61.47<br>70.13<br>38.09<br>84.58<br>87.01<br>65.47<br>70.26<br>86.12<br>63.37<br>66.26<br>37.49<br>78.60<br>83.71<br>64.46<br>67.49<br>82.94<br>60.06<br>67.61<br>37.47<br>77.85<br>83.70<br>64.51<br>67.08<br>82.39<br>59.65<br>65.13<br>37.32<br>80.53<br>87.13<br>65.39<br>64.80<br>82.83<br>52.08|69.29<br>70.63<br>67.63<br>67.53<br>66.90|
|SubTab + Fine-tune<br>SubTab + LR<br>SubTab + kNN<br>VIME + Fine-tune<br>VIME + LR<br>Self-sup.<br>VIME + kNN|✓<br>✓<br>-<br>✓<br>✓<br>-|66.01<br>37.60<br>67.80<br>75.40<br>66.69<br>56.46<br>75.34<br>55.62<br>70.12<br>37.67<br>73.25<br>86.07<br>64.92<br>61.34<br>82.14<br>58.90<br>71.91<br>39.51<br>69.56<br>83.60<br>68.79<br>59.87<br>80.13<br>61.57<br>65.97<br>37.25<br>77.82<br>83.13<br>64.40<br>63.63<br>81.01<br>59.58<br>67.80<br>37.51<br>82.87<br>87.42<br>64.29<br>71.53<br>86.79<br>69.62<br>72.16<br>39.28<br>79.15<br>83.86<br>66.94<br>68.45<br>84.07<br>71.09|62.62<br>66.80<br>66.87<br>66.60<br>70.98<br>70.63|
|UMTRA<br>SES<br>CACTUs<br>Unsup.-Meta.<br>**STUNT (Ours)**|-<br>-<br>-<br>-|65.78<br>38.05<br>67.28<br>73.29<br>64.41<br>35.90<br>51.32<br>25.08<br>68.27<br>39.04<br>74.80<br>78.46<br>66.61<br>52.74<br>74.80<br>52.25<br>72.03<br>38.81<br>82.20<br>85.92<br>66.79<br>65.00<br>85.25<br>**81.52**<br>**72.69**<br>**40.40**<br>**85.45**<br>**88.42**<br>**69.88**<br>**73.02**<br>**89.08**<br>79.18|52.64<br>63.37<br>72.19<br>**74.77**|



## 4.1 FEW-SHOT CLASSIFICATION 

**Dataset.** We select 8 datasets from the OpenML-CC18 benchmark (Bischl et al., 2021; Asuncion & Newman, 2007). The income (Kohavi et al., 1996) and cmc dataset consists of both categorical and numerical features. The mfeat-karhunen (karhunen), optdigits, diabetes, semeion, mfeat-pixel (pixel) contain only numerical features. The dna dataset consists of only categorical features. We demonstrate the performance of STUNT on all types, as described in Appendix G. For dataset selection, we consider the following attributes: (i) whether the dataset consists of both categorical and numerical features, (ii) consists only of numerical or categorical features, (iii) type of task (i.e., binary classification or multi-way classification task). We validate that STUNT is generally applicable to arbitrary tabular data by performing experiments across datasets with the above properties. 

**Baselines.** To validate our method, we compare the performance with four types of baselines: (i) supervised, (ii) semi-supervised, (iii) self-supervised, and (iv) unsupervised meta-learning methods. First, we compare with supervised learning methods such as CatBoost (Prokhorenkova et al., 2018), 2-layer MLP, k-nearest neighbors (kNN), and logistic regression (LR). kNN denotes the nearest neighbor classifier according to the prototype of the input data. Second, we compare our method to semi-supervised learning methods such as Mean Teacher (MT; Tarvainen & Valpola (2017)), Interpolation Consistency Training (ICT; Verma et al. (2019)), Pseudo-Label (PL; Lee (2013)), Meta Pseudo-Label (MPL; Pham et al. (2021)). We also have considered PAWS (Assran et al., 2021), 

6 

Published as a conference paper at ICLR 2023 

Table 2: 10-shot test accuracy (%) on 8 datasets from the OpenML-CC18 benchmark (Bischl et al., 2021). We report the mean test accuracy over 20 different seeds for each dataset. The bold indicates results within 1% from the highest mean score. 

|Method_\_Dataset|income<br>cmc<br>karhunen<br>optdigit<br>diabetes<br>semeion<br>pixel<br>dna|Avg.|
|---|---|---|
|kNN<br>ICT<br>VIME + LR<br>CACTUs|**74.27**<br>41.07<br>85.63<br>87.44<br>71.32<br>74.64<br>87.52<br>71.15<br>71.56<br>38.00<br>**88.25**<br>**90.84**<br>67.63<br>74.67<br>**89.13**<br>69.55<br>69.17<br>37.92<br>86.63<br>89.63<br>66.56<br>**77.66**<br>88.71<br>74.73<br>**73.63**<br>**42.14**<br>85.48<br>87.92<br>70.75<br>68.22<br>87.21<br>**84.40**|74.13<br>73.70<br>73.88<br>74.97|
|**STUNT (Ours)**|**74.08**<br>**42.01**<br>86.95<br>**89.91**<br>**72.82**<br>74.74<br>**89.90**<br>80.96|**76.42**|



the state-of-the-art semi-supervised learning method in the image domain, but observe that it is not effective for tabular datasets: we conjecture that the performance highly deviates from the choice of augmentation where tabular augmentation is not very effective compared to image augmentations. Third, we consider the recent state-of-the-art self-supervised method for tabular data, SubTab (Ucar et al., 2021) and VIME (Yoon et al., 2020) are pre-trained, then performance is evaluated with a few-shot labeled samples using fine-tuning, logistic regression, and kNN. Finally, we include CACTUs (Hsu et al., 2018), UMTRA (Khodadadeh et al., 2019), and SES (Ye et al., 2022) (along with semi-normalized similarity) as an unsupervised meta-learning baseline. Even though it is not clear how to design the augmentation strategy when applying UMTRA and SES to tabular data, we use marginal distribution masking, which are simple augmentation strategies used in SubTab. We provide additional results in Appendix I. We exclude Meta-GMVAE (Lee et al., 2021a) since in our experiments, it lags behind the baseline with the lowest performance that we consider. This is because training a variational auto-encoder (VAE; Kingma & Welling (2014)) for tabular datasets is highly non-trivial (Xu et al., 2019). 

**Few-shot classification.** For the few-shot classification, we evaluate the performance when one and five labeled samples are available per class, respectively. We find that some baselines, such as CatBoost and ICT, require a validation set as they are highly sensitive to hyperparameters.Therefore, we perform a hyperparameter search and early stopping with 100 additional labeled samples for all baseline except for kNNs and unsupervised meta-learning methods. We note that using more labeled samples for validation than training is indeed unrealistic. On the other hand, we use the proposed pseudo-validation scheme for hyperparameter searching and early stopping of STUNT. One surprising observation is that CatBoost even lags behind kNN despite careful hyperparameter search. This implies that gradient boosting decision tree algorithms may fail in few-shot learning, while they are one of the most competitive models in fully-supervised settings (Shwartz-Ziv & Armon, 2022). In addition, semi-supervised learning methods achieve relatively low scores, which means that in tabular domain, pseudo-label quality goes down when the number of labeled samples is extremely small. Also, unlike the results of the image domain, UMTRA and SES perform worse than CACTUs. We believe that the failures of them are mainly due to the absence of effective augmentation strategies for tabular data. 

As shown in Table 1, STUNT significantly improves the few-shot tabular classification performance even without using a labeled validation set. For instance, STUNT outperforms CACTUs from 67.61% _→_ 79.05% in the 1-shot classification of the pixel dataset. In particular, STUNT achieves the highest score in 7 of 8 datasets in both 1-shot and 5-shot classification problems, performing about 4% and 2% better than CACTUs in 1-shot and 5-shot cases, respectively. This is because STUNT is a _tabular-specific_ unsupervised learning method that generates myriad meta-training tasks than CACTUs because we randomly select a subset of the column features for every training iteration. 

**Low-shot classification.** We also validate our method when more labels are available, i.e., 10-shot. For baselines, we choose kNN, ICT, VIME + LR, and CACTUs because they show the best performance among supervised, semi-supervised, self-supervised, and unsupervised meta-learning methods in 1-shot and 5-shot classifications, respectively. Since a sufficient number of labeled training samples are available, we use the 2-shot sample from the 10-shot training sample for validation if the baseline requires hyperparameter search and early stop. In contrast, STUNT still does not use the labeled validation set, i.e., we utilize the proposed pseudo-validation scheme. As shown in Table 2, STUNT achieves the best score on average accuracy even under the low-shot classification setup. 

While STUNT outperforms the baselines in the few- and low-shot learning setups, we find ensembles of decision trees or other semi-supervised learning methods, e.g., CatBoost or ICT, achieve a better 

7 

Published as a conference paper at ICLR 2023 

Table 3: Few-shot multi-task test accuracy (%) on the emotions dataset (Vanschoren et al., 2014), consists of 6 binary classification tasks. We report the mean test accuracy over 100 different seeds for each task. The bold indicates the highest mean score. 

|Method_\_Task|amazed-surprised<br>happy-please<br>relaxing-calm<br>quiet-still<br>sad-lonely<br>angry-aggressive|Avg.|
|---|---|---|
|# shot=1|||
|kNN<br>SubTab + kNN<br>VIME + kNN<br>CACTUs|59.04<br>47.14<br>55.77<br>66.86<br>55.96<br>59.47<br>**63.32**<br>48.88<br>56.46<br>62.56<br>54.34<br>57.99<br>60.07<br>49.51<br>55.62<br>64.74<br>53.95<br>60.29<br>61.58<br>50.67<br>55.63<br>63.18<br>55.10<br>59.39|57.37<br>57.26<br>57.36<br>57.59|
|**STUNT (Ours)**|62.71<br>**51.63**<br>**59.28**<br>**69.34**<br>**56.38**<br>**63.43**|**60.46**|
|# shot=5|||
|kNN<br>SubTab + kNN<br>VIME + kNN<br>CACTUs|70.71<br>53.48<br>66.34<br>81.03<br>68.51<br>68.07<br>**74.41**<br>52.23<br>64.90<br>72.70<br>62.32<br>63.30<br>70.71<br>53.10<br>66.24<br>79.54<br>66.34<br>67.76<br>71.41<br>53.64<br>65.18<br>77.57<br>64.15<br>66.57|68.02<br>64.98<br>67.28<br>66.42|
|**STUNT (Ours)**|72.38<br>**55.09**<br>**67.39**<br>**83.10**<br>**68.61**<br>**70.10**|**69.45**|



performance when more labeled samples are provided, e.g., 50-shot. Although the few-shot learning scenario is of our primary interest and our current method is specialized for the purpose, we think further improving our method under many-shot regimes would be an interesting future direction. 

## 4.2 MULTI-TASK LEARNING 

In this section, we introduce another application of STUNT, the few-shot multi-task learning. As STUNT learns to generalize across various self-generated tasks, it can instantly be adapted to multiple tasks at test-time without further training the network. Formally, we consider **x** _l_ and **x** _u_ are sampled from a same marginal distribution _p_ ( **x** ), where the label space _Y_ differs. 

**Dataset.** We use the emotions dataset from OpenML (Vanschoren et al., 2014), consisting of a variety of audio data with multiple binary labels and 72 numerical features. In particular, the emotion dataset aims to classify the multiple properties: amazed-surprised, happy-please, relaxing-calm, quiet-still, sad-lonely, or angry-aggressive. Because it is multi-labeled, data can have multiple attributes simultaneously, such as amazed-surprised and relaxing-calm audio. 

**Baselines.** We compare STUNT with four baselines: kNN, SubTab + kNN, VIME + kNN, and CACTUs. All four baselines are chosen because they can adapt to multiple tasks with only one training procedure. On the other hand, methods such as ICT are excluded because they need a training procedure for each task. For example, in the case of the emotions dataset, six models are required. 

**Multi-task learning.** As shown from Table 3, STUNT outperforms 5 out of 6 tasks in both 1- shot and 5-shot multi-task adaptation. In particular, STUNT outperforms the best baseline from 57.59% _→_ 60.46% in 1-shot multi-task, and 68.02% _→_ 69.45% in 5-shot. This is because STUNT generates a wide variety of meta-training tasks based on the fact that there are myriad ways to randomly select subsets of column features. In addition, it makes sense to treat column features as alternate labels, especially in tabular data, because each column feature has a different meaning. Considering the presence of critical real-world few-shot multi-task scenarios, such as patients with more than one disease, we conclude that STUNT is a promising way to mitigate these problems. 

## 4.3 EFFECTIVENESS OF PSEUDO-VALIDATION 

In this section, we perform further analysis of the proposed pseudo-validation with STUNT. For the analysis, we use four datasets from the OpenML-CC18 benchmark: two datasets containing both categorical and numerical features (i.e., income, cmc) and two datasets with only numerical features (i.e., semeion, pixel). We validate the model by constructing a number of 1-shot meta-validation tasks, i.e., _|S_ `[val]` _|_ = _C_ , with an unlabeled validation set for all experiments in this section. 

**Hyperparameter search.** To validate that the proposed pseudo-validation scheme is useful for hyperparameter search, we show the correlation between the pseudo-validation accuracy and the test accuracy (achieved from the early stop point by the highest pseudo-validation accuracy). As shown in Figure 3, pseudo-validation accuracy and test accuracy have a positive correlation, which means 

8 

Published as a conference paper at ICLR 2023 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/iclr_2023_STUNT_Few-shot_Tabular_Learning_with_Self-generated_Tasks_from_Unlabeled_Tables_images/iclr_2023_STUNT_Few-shot_Tabular_Learning_with_Self-generated_Tasks_from_Unlabeled_Tables.pdf-0009-01.png)


**----- Start of picture text -----**<br>
income cmc<br>37.2<br>64<br>36.9<br>63<br>36.6<br>62<br>36.3<br>61<br>79.0 79.5 80.0 80.5 81.0 81.5 82.0 82.5 83.0 73 74 75 76 77 78 79 80<br>Pseudo-validation acc (%) Pseudo-validation acc (%)<br>semeion pixel<br>56.4<br>78<br>55.2<br>75<br>54<br>72<br>52.8<br>69<br>61.5 62.0 62.5 63.0 63.5 64.0 64.5 65.0 75.0 76.5 78.0 79.5 81.0 82.5 84.0<br>Pseudo-validation acc (%) Pseudo-validation acc (%)<br>1-shot test acc (%) 1-shot test acc (%)<br>1-shot test acc (%) 1-shot test acc (%)<br>**----- End of picture text -----**<br>


Figure 3: Correlation between the pseudo-validation accuracy (%) and the 1-shot test accuracy (%). The 1-shot test accuracy is achieved from the early stopped point by the highest pseudo-validation accuracy. Blue dots represent models trained with different hyperparameters. Red lines are the result of linear regression result of pseudo-validation accuracy and 1-shot test accuracy. 

Table 4: Early stopping performance with the pseudo-validation set. We report 1-shot and 5-shot test accuracy (%) of fully trained (Last) and early stopped models (Early). We report the mean test accuracy of 100 different seeds. The bold indicates the highest mean score. 

|Problem|income<br><br>Last<br>Early|cmc|semeion|pixel<br>Last<br>Early|
|---|---|---|---|---|
|||Last<br>Early|Last<br>Early||
|1-shot<br>5-shot|61.58<br>**63.52**<br>70.84<br>**72.69**|36.94<br>**37.10**<br>**40.43**<br>40.40|51.94<br>**55.91**<br>71.55<br>**73.02**|74.92<br>**79.05**<br>87.60<br>**89.08**|



that the higher the best pseudo-validation accuracy, the higher the test accuracy. Therefore, we use the pseudo-validation technique to search the hyperparameters of STUNT. Specifically, for the income, semeion, and pixel datasets, we find hyperparameters in eight combinations of hyperparameters. For the cmc dataset, we find hyperparameters in four combinations (indicated by blue dots in Figure 3). Although the best validation score may not guarantee the optimal hyperparameters, our method still gives reasonable hyperparameters. Additional information are reported in Appendix E. 

**Early stopping.** As shown in Table 4, our pseudo-validation method is also useful for relaxing the overfitting issue. For example, on the pixel dataset, evaluating with the early stop model achieves 4.13% better accuracy than evaluating with the model after 10K training steps. Sometimes it is better to evaluate the model after a full training step, such as the cmc dataset, but our method still provides a reasonable early stopping rule when we see that the performance of the early stopped model by the highest pseudo-validation result only performs about 0.03% lower on the cmc dataset. In addition, the optimal required training steps are not known and often vary widely across datasets, especially in the tabular domain. For example, Levin et al. (2022) uses different fine-tuning epochs for different training setups (e.g., if 4 downstream training samples are available, use 30 fine-tuning epochs, and if 20 samples are available, use 60 fine-tuning epochs). However, since we use the pseudo-validation approach for early stopping, all we have to do is train the model for enough training steps (e.g., 10K training steps in our case) and use the model that achieves the best pseudo-validation score. 

## 5 CONCLUSION 

In this paper, we tackle the few-shot tabular learning problem, which is an under-explored but important research question. To this end, we propose STUNT, a simple yet effective framework that meta-learns over the self-generated tasks from unlabeled tables. Our key idea is to treat randomly selected columns as target labels to generate diverse few-shot tasks. The effectiveness of STUNT is validated by various few-shot classification tasks on different types of tabular datasets, and we also show that the representations extracted by STUNT apply well in multi-task scenarios. We hope that our work will guide new interesting directions in tabular learning field in the future. 

9 

Published as a conference paper at ICLR 2023 

## ETHICS STATEMENT 

Since tabular data sometimes consists of privacy-sensitive features, e.g., social security number, one should always use the data carefully. However, STUNT is able to be well trained even with the encrypted data since the key idea is to use the tabular’s unique property that each column has distinct meanings. Therefore, even though privacy issues may occur for tabular learning, STUNT has the potential to be generally used along with privacy-preserving techniques, such as homomorphic encryption (Cheon et al., 2017). Unsupervised meta-learning with STUNT on encrypted features is an interesting future direction. 

## REPRODUCIBILITY STATEMENT 

We provide code for reproduction in the supplementary material and describe the implementation details in Appendix B. 

## ACKNOWLEDGMENTS AND DISCLOSURE OF FUNDING 

We would like to thank Sangwoo Mo, Jaehyung Kim, Sihyun Yu, Subin Kim, and anonymous reviewers for their helpful feedbacks and discussions. This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST); No.2022-0-00959, Few-shot Learning of Casual Inference in Vision and Language for Decision Making) and Samsung Electronics Co., Ltd (IO201211-08107-01). 

## REFERENCES 

- Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, and Michael Rabbat. Semi-supervised learning of visual features by non-parametrically predicting view assignments with support samples. In _IEEE International Conference on Computer Vision_ , 2021. 

Arthur Asuncion and David Newman. Uci machine learning repository, 2007. 

Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler. Scarf: Self-supervised contrastive learning using random feature corruption. In _International Conference on Learning Representations_ , 2022. 

- David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. MixMatch: A holistic approach to semi-supervised learning. In _Advances in Neural Information Processing Systems_ , 2019. 

- Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael G. Mantovani, Jan N. van Rijn, and Joaquin Vanschoren. Openml benchmarking suites. In _Advances in Neural Information Processing Systems Track on Datasets and Benchmarks_ , 2021. 

- Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In _International Conference on Machine Learning_ , 2020a. 

- Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big selfsupervised models are strong semi-supervised learners. In _Advances in Neural Information Processing Systems_ , 2020b. 

- Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. A closer look at few-shot classification. In _International Conference on Learning Representations_ , 2019. 

- Zitian Chen, Subhransu Maji, and Erik Learned-Miller. Shot in the dark: Few-shot learning with no base-class labels. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2668–2677, 2021. 

10 

Published as a conference paper at ICLR 2023 

- Jung Hee Cheon, Andrey Kim, Miran Kim, and Yongsoo Song. Homomorphic encryption for arithmetic of approximate numbers. In _International Conference on the Theory and Application of Cryptology and Information Security_ , 2017. 

- Jillian M. Clements, Di Xu, Nooshin Yousefi, and Dmitry Efimov. Sequential deep learning for credit risk monitoring with tabular financial data. _arXiv preprint arXiv:2012.15330_ , 2020. 

- Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In _International Conference on Machine Learning_ , 2017. 

- Jiatao Gu, Yong Wang, Yun Chen, Kyunghyun Cho, and Victor OK Li. Meta-learning for lowresource neural machine translation. In _Conference on Empirical Methods in Natural Language Processing_ , 2018. 

- Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: A factorizationmachine based neural network for ctr prediction. In _International Joint Conferences on Artificial Intelligence_ , 2017. 

- Kyle Hsu, Sergey Levine, and Chelsea Finn. Unsupervised learning via meta-learning. In _International Conference on Learning Representations_ , 2018. 

- Siavash Khodadadeh, Ladislau B¨ol¨oni, and Mubarak Shah. Unsupervised meta-learning for fewshot image classification. In _Advances in Neural Information Processing Systems_ , 2019. 

- Jaehyung Kim, Youngbum Hur, Sejun Park, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin. Distribution aligning refinery of pseudo-label for imbalanced semi-supervised learning. In _Advances in Neural Information Processing Systems_ , 2020. 

- Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International Conference on Learning Representations_ , 2014. 

- Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In _International Conference on Learning Representations_ , 2014. 

- Ron Kohavi et al. Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid. In _ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , 1996. 

- Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. In _International Conference on Learning Representations_ , 2017. 

- Dong Bok Lee, Dongchan Min, Seanie Lee, and Sung Ju Hwang. Meta-GMVAE: Mixture of gaussian VAE for unsupervised meta-learning. In _International Conference on Learning Representations_ , 2021a. 

- Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. _ICML Workshop : Challenges in Representation Learning (WREPL)_ , 2013. 

- Hankook Lee, Kibok Lee, Kimin Lee, Honglak Lee, and Jinwoo Shin. Improving transferability of representations via augmentation-aware self-supervision. In _Advances in Neural Information Processing Systems_ , 2021b. 

- Kyungmin Lee and Jinwoo Shin. R´enyicl: Contrastive representation learning with skew r´enyi divergence. In _Advances in Neural Information Processing Systems_ , 2022. 

- Roman Levin, Valeriia Cherepanova, Avi Schwarzschild, Arpit Bansal, C Bayan Bruss, Tom Goldstein, Andrew Gordon Wilson, and Micah Goldblum. Transfer learning with deep tabular models. _arXiv preprint arXiv:2206.15306_ , 2022. 

- Kushal Majmundar, Sachin Goyal, Praneeth Netrapalli, and Prateek Jain. Met: Masked encoding for tabular data. _arXiv preprint arXiv:2206.08564_ , 2022. 

- Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language model prompting for few-shot text classification. In _Annual Conference of the Association for Computational Linguistics_ , 2022. 

11 

Published as a conference paper at ICLR 2023 

Mark Peplow. The 100,000 genomes project. _British Medical Journal_ , 2016. 

- Ethan Perez, Douwe Kiela, and Kyunghyun Cho. True few-shot learning with language models. In _Advances in Neural Information Processing Systems_ , 2021. 

- Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, and Quoc V. Le. Meta pseudo labels. In _IEEE Conference on Computer Vision and Pattern Recognition_ , 2021. 

- Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. Catboost: unbiased boosting with categorical features. In _Advances in Neural Information Processing Systems_ , 2018. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_ , 2019. 

Ravid Shwartz-Ziv and Amitai Armon. Tabular data: Deep learning is not all you need. _Information Fusion_ , 2022. 

- Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In _Advances in Neural Information Processing Systems_ , 2017. 

- Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In _Advances in Neural Information Processing Systems_ , 2020. 

- Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C Bayan Bruss, and Tom Goldstein. Saint: Improved neural networks for tabular data via row attention and contrastive pre-training. _arXiv preprint arXiv:2106.01342_ , 2021. 

- Jihoon Tack, Jongjin Park, Hankook Lee, Jaeho Lee, and Jinwoo Shin. Meta-learning with selfimproving momentum target. In _Advances in Neural Information Processing Systems_ , 2022. 

- Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In _Advances in Neural Information Processing Systems_ , 2017. 

- Yonglong Tian, Yue Wang, Dilip Krishnan, Joshua B. Tenenbaum, and Phillip Isola. Rethinking few-shot image classification: a good embedding is all you need? In _European Conference on Computer Vision_ , 2020. 

- Talip Ucar, Ehsan Hajiramezanali, and Lindsay Edwards. Subtab: Subsetting features of tabular data for self-supervised representation learning. In _Advances in Neural Information Processing Systems_ , 2021. 

- Dennis Ulmer, Lotta Meijerink, and Giovanni Cin`a. Trust issues: Uncertainty estimation does not enable reliable ood detection on medical tabular data. In _Machine Learning for Health_ , 2020. 

- Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo. OpenML: networked science in machine learning. _ACM SIGKDD Explorations Newsletter_ , 2014. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_ , 2017. 

- Vikas Verma, Kenji Kawaguchi, Alex Lamb, Juho Kannala, Yoshua Bengio, and David Lopez-Paz. Interpolation consistency training for semi-supervised learning. In _International Joint Conferences on Artificial Intelligence_ , 2019. 

- Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In _Advances in Neural Information Processing Systems_ , 2016. 

- Annie Xie, Avi Singh, Sergey Levine, and Chelsea Finn. Few-shot goal inference for visuomotor learning and planning. In _Conference on Robot Learning_ , 2018. 

12 

Published as a conference paper at ICLR 2023 

Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Modeling tabular data using conditional GAN. In _Advances in Neural Information Processing Systems_ , 2019. 

- Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, and Fei Sha. Few-shot learning via embedding adaptation with set-to-set functions. In _CVPR_ , 2020. 

- Han-Jia Ye, Lu Han, and De-Chuan Zhan. Revisiting unsupervised meta-learning via the characteristics of few-shot tasks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2022. 

- Jinsung Yoon, Yao Zhang, James Jordon, and Mihaela van der Schaar. Vime: Extending the success of self- and semi-supervised learning to tabular domain. In _Advances in Neural Information Processing Systems_ , 2020. 

- Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. Deep learning based recommender system. _Association for Computing Machinery (ACM) Computing Surveys_ , 2020. 

- Peng Zhou, Xing-Lou Yang, Xian-Guang Wang, Ben Hu, Lei Zhang, Wei Zhang, Hao-Rui Si, Yan Zhu, Bei Li, Chao-Lin Huang, Hui-Dong Chen, Jing Chen, Yun Luo, Hua Guo, Ren-Di Jiang, Mei-Qin Liu, Ying Chen, Xu-Rui Shen, Xi Wang, Xiao-Shuang Zheng, Kai Zhao, Quan-Jiao Chen, Fei Deng, Lin-Lin Liu, Bing Yan, Fa-Xian Zhan, Yan-Yi Wang, Geng-Fu Xiao, and ZhengLi Shi. A pneumonia outbreak associated with a new coronavirus of probable bat origin. _Nature_ , 2020. 

13 

Published as a conference paper at ICLR 2023 

## A BASELINE DETAILS 

In this section, we provide brief explanations of the considered baselines and the hyperparameter search space of the baselines. In common, we use Adam optimizer (Kingma & Ba, 2014) with learning rate 1 _e −_ 3, and weight decay 1 _e −_ 4. All baselines are trained with batch size 100 for all experiments. 

**CatBoost.** CatBoost (Prokhorenkova et al., 2018) is one of gradient boosted decision tree algorithms. CatBoost consecutively builds decision trees in a way that reduces loss compared to previous trees. We search hyperparameters as shown in Table 5. 

Table 5: Hyperparameter search space of Catboost (Prokhorenkova et al., 2018). 

|Hyperparameter|Search space|
|---|---|
|||
|iterations<br>max depth<br>learning rate<br>bagging temperature<br>l2 leaf reg<br>leaf estimation iterations|_{_10, 100, 200, 500, 1000_}_<br>_{_4, 6, 8, 10_}_<br>_{_0.001, 0.01, 0.1, 0.03_}_<br>_{_0.6, 0.8, 1.0_}_<br>_{_1, 3, 5, 7_}_<br>_{_1, 2, 4, 8_}_|



**Mean Teacher (MT).** MT (Tarvainen & Valpola, 2017) is semi-supervised learning method which uses the consistency loss between the teacher output and student output. The teacher model weights are updated as an exponential moving average of the student weights. We use the decay rate as 0 _._ 999 for exponential moving average. We search for weight of consistency loss in _{_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 20 _,_ 50 _,_ 100 _}_ . 

**Interpolation Consistency Training (ICT).** ICT (Verma et al., 2019) is a semi-supervised learning method uses MT framework while the student parameters are updated to encourage the consistency between the output of mixed samples and the mixed output of the samples. We use the decay rate as 0 _._ 999, and search for the weight of consistency loss in _{_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 20 _,_ 50 _,_ 100 _}_ . We find the _β_ for Beta distribution in _{_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 5 _,_ 1 _}_ . 

**Meta Pseudo-Label (MPL).** MPL (Pham et al., 2021) is a semi-supervised learning method which utilizes the performance of the student on the labeled dataset to inform the teacher to generate better pseudo-labels. In particular, the student model learns from pseudo-labeled data given from the teacher model. We use the decay rate as 0 _._ 999, and search for the weight of the unsupervised loss in _{_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 20 _,_ 50 _,_ 100 _}_ . 

**VIME.** VIME (Yoon et al., 2020) is a self-supervised learning method which extracts useful representation by corrupting random features and then predicting the corrupted location. For VIME pre-training, we follow the best hyperparameters suggested from original paper. Using VIME representations, we perform k-nearest neighbor classify, logistic regression, and fine-tuning. Early stop is done for logistic regression and fine-tuning. 

**SubTab.** SubTab (Ucar et al., 2021) is a self-supervised learning method using effective three pretext task losses (i.e., reconstruction loss, contrastive loss, and distance loss). For SubTab pre-trianing, we follow the best hyperparameters suggested from original paper. Using SubTab representations, we perform k-nearest neighbor classify, logistic regression, and fine-tuning. Early stop is done for logistic regression and fine-tuning. 

**CACTUs.** CACTUs (Hsu et al., 2018) is an unsupervised meta-learning method which runs a clustering algorithm on a representation trained with self-supervised learning in order to self-generate the tasks. We follow the hyperparameters suggested in the original paper. 

For rest of the baselines, i.e., 2-layer multi-layer perceptron, logistic regression, pseudo-label (Lee, 2013), we use labeled validation set (i.e., additional 100 samples for 1-shot, 5-shot learning, and 2-shot samples for 10-shot learning) only for early stopping. 

14 

Published as a conference paper at ICLR 2023 

## B EXPERIMENTAL DETAILS 

In this section, we provide hyperparameters of STUNT in Table 6 for each dataset found through the proposed pseudo-validation scheme. Shot indicates the number of sample per pseudo-class in _S_ , and query indicates the number of sample per pseudo-class in _Q_ . Way indicates the number of centroids in Eq. (1). 

Table 6: Hyperparameters of STUNT 

||income|cmc|karhunen|optdigit|diabetes|semeion|pixel|dna|emotions|
|---|---|---|---|---|---|---|---|---|---|
|# shot|1|1|1|1|1|1|1|1|1|
|# query|15|5|15|15|15|15|15|15|5|
|# way|10|3|20|20|5|20|10|10|16|



Except for shot, query, and way, we use full batch when self-generating STUNT tasks, and then use meta-training task batch size 4, _r_ 1 = 0 _._ 2, _r_ 2 = 0 _._ 5 and Adam optimizer with learning rate 1 _e −_ 3. 

## C DATASET DETAILS 

In this section, we provide brief explanations of the considered datasets from the OpenML-CC18 benchmark (Vanschoren et al., 2014; Bischl et al., 2021). 

**Income.** The task of the income (Kohavi et al., 1996; Bischl et al., 2021) dataset is to classify whether a person makes less than 50K a year or more than 50K a year. 

**Cmc.** Cmc (Asuncion & Newman, 2007; Bischl et al., 2021) is an abbreviation for Contraceptive Method Choice. Literally, the target task is to predict the contraceptive method choice (i.e., no use, long-term or short-term). 

**Mfeat-karhunen (karhunen), mfeat-pixel (pixel).** The karhunen and pixel (Asuncion & Newman, 2007; Bischl et al., 2021) datasets describe features of handwritten numbers. In particular, the karhunen dataset aims to find the correlation between 64 features obtained through the KarhunenLoeve Transform and the 10 handwritten numbers drawn from the Dutch utility maps. On the other hand, the pixel dataset consists of 240 features by averaging 2 _×_ 3 windows. 

**Optdigit.** The optdigit (Asuncion & Newman, 2007; Bischl et al., 2021) is the dataset that describes the optical recognition of handwritten digits. 

**Diabetes.** Literally, the diabetes (Asuncion & Newman, 2007; Bischl et al., 2021) dataset aims to predict whether the patient is tested positive for diabetes or not. In particular, the dataset features are composed of 8 numerical features, including diastolic blood pressure and body mass index. 

**Semeion.** Semeion (Asuncion & Newman, 2007; Bischl et al., 2021) dataset is drawn by scanning and documenting handwritten digits from around 80 people. 

**Dna.** The task of the dna (Bischl et al., 2021) dataset is to classify the boundaries between exons and introns with 180 indicator binary variables. 

15 

Published as a conference paper at ICLR 2023 

## D ALGORITHM 

**Algorithm 1** STUNT: Self-generated Tasks from UNlabeled Tables 

**Require:** Unlabeled dataset _Du_ = _{_ **x** _u,i}[N] i_ =1 _[u]_[, Labeled dataset] _[ D][l]_[=] _[ {]_ **[x]** _[l,i][,]_ **[ y]** _[l,i][}][N] i_ =1 _[l]_[,] task batch size _Mt_ , learning rate _β_ , mask ratio hyperparameters _r_ 1 _, r_ 2. 1: Initialize _θ_ using the standard initialization scheme. 2: // Step 1: Unsupervised meta-learning with STUNT 3: **while** not done **do** 4: **for** _j_ = 1 to _Mt_ **do** 5: Sample mask ratio _p ∼ U_ ( _r_ 1 _, r_ 2). 6: **m** = [ _m_ 1 _, . . . , md_ ] _[⊤] ∈{_ 0 _,_ 1 _}[d]_ s.t.[�] _i[m][i]_[=] _[ ⌊][dp][⌋]_[.] 7: Run a k-means clustering: Eq. (1) with **x** _u ∈Du_ and **m** to generate the task label **y** ˜ _u_ . ˜ 8: _T_ `STUNT` _,j_ = _{_ **x** _u,i,_ ˜ **y** _u,i}[N] i_ =1 _[u]_[where] **[x]**[˜] _[u,i]_[:=] **[ m]** _[ ⊙]_ **[x]**[ˆ] _[u,i]_[ + (1] _[ −]_ **[m]**[)] _[ ⊙]_ **[x]** _[u,i]_[.] 9: Sample two disjoint sets _Sj_ and _Qj_ from a given task _T_ `STUNT` _,j_ . 10: _L_ `meta` ( _θ, Qj_ ) =[�] (˜ **x** _u,_ **y** ˜ _u_ ) _∈Qj[L]_ `[CE]` � _fθ_ (˜ **x** _u_ ; _Sj_ ) _,_ ˜ **y** _u_ �. 11: **end for** _β Mt_ 12: _θ ← θ − Mt[· ∇][θ]_ � _j_ =1 _[L]_ `[meta]`[(] _[θ,][ Q][j]_[)][.] 13: **end while** 

14: // Step 2: Adapt classifier with the labeled dataset 

15: Conduct a ProtoNet classifier _fθ_ ( _·_ ; _Dl_ ) using _Dl_ . 

## E HYPERPARAMETER DETAILS OF ABLATION STUDY 

As shown in Table 3, this section provides the search space of hyperparameters of ablation study in Section 4.3. Shot indicates the number of sample per pseudo-class in _S_ , and query indicates the number of sample per pseudo-class in _Q_ . Way indicates the number of centroids in Eq. (1). 

Table 7: Hyperparameter search space of datasets used in Section 4.3 

|Dataset<br>Hyperparameter|Search space|
|---|---|
|||
|income<br>(shot, query)<br>way|_{_(1, 5), (1, 15), (5, 10), (5, 20)_}_<br>_{_5, 10_}_|
|||
|cmc<br>(shot, query)<br>way|_{_(1, 5), (1, 15), (5, 10), (5, 20)_}_<br>_{_3_}_|
|||
|semeion<br>(shot, query)<br>way|_{_(1, 5), (1, 15), (5, 10), (5, 20)_}_<br>_{_10, 20_}_|
|||
|pixel<br>(shot, query)<br>way|_{_(1, 5), (1, 15), (5, 10), (5, 20)_}_<br>_{_10, 20_}_|



16 

Published as a conference paper at ICLR 2023 

## F FEW-SHOT TABULAR REGRESSION RESULTS 

Table 8: Few-shot regression tasks on 5 datasets from the OpenML (Vanschoren et al., 2014). We report the mean squared error over 100 different seeds for each dataset. The bold indicates the lowest mean error. 

|Input|news<br>abalone<br>cholesterol<br>sarcos<br>boston|
|---|---|
|# shot=5||
|||
|Raw<br>VIME<br>CACTUs|2.74E-04<br>1.75E-02<br>1.37E-02<br>**1.05E-02**<br>3.65E-02<br>2.69E-04<br>1.70E-02<br>1.37E-02<br>1.06E-02<br>**3.53E-02**<br>2.75E-04<br>1.72E-02<br>1.46E-02<br>1.06E-02<br>3.76E-02|
|||
|**STUNT (Ours)**|**2.68E-04**<br>**1.66E-02**<br>**1.35E-02**<br>1.06E-02<br>3.70E-02|
|# shot=10||
|||
|Raw<br>VIME<br>CACTUs|**2.53E-04**<br>1.49E-02<br>1.13E-02<br>9.21E-03<br>2.88E-02<br>**2.53E-04**<br>1.49E-02<br>1.13E-02<br>9.24E-03<br>**2.78E-02**<br>2.54E-04<br>1.51E-02<br>1.21E-02<br>**9.16E-03**<br>2.94E-02|
|**STUNT (Ours)**<br>**2.53E-04**<br>**1.46E-02**<br>**1.12E-02**<br>**9.16E-03**<br>2.90E-02||



We evaluate the capability of STUNT in few-shot regression tasks by replacing the ProtoNet classifier with a kNN regressor at the adaptation stage (i.e., after unsupervised meta-learning with STUNT). 

We consider 5 tabular regression datasets in OpenML (Vanschoren et al., 2014), where we preprocess the input and target features with min-max scaling. For comparison, we evaluate the performance of the kNN regressor on VIME (Yoon et al., 2020) and CACTUs (Hsu et al., 2018) representations. Also, we report the performance of naive kNN regressor on the raw input. We use _k_ = 5 and _k_ = 10 for 5-shot and 10-shot experiments, respectively, where _k_ is the number of nearest neighbors. 

In the Table 8, we report the average of mean-squared-errors (MSEs) over 100 different seeds of each method and dataset. The results indicate that STUNT is a competitive approach in a few-shot tabular regression task. However, the performance gap is often vacuous or marginal compared to the few-shot classification tasks. We believe that this is because STUNT meta-train networks with classification tasks, thus, can be more easily adapted to classification test-tasks. Therefore, extending STUNT by self-generating target-regression tasks with distinct column features could be effective in few-shot regression tasks, which we leave for future works. 

## G DATASET DESCRIPTION 

Table 9: Dataset description. We select 8 tabular datasets from the OpenML-CC18 benchmark (Bischl et al., 2021) for extensive evaluation. The selected dataset consists of (i) both numerical and categorical features, (ii) only numerical features, and (iii) only categorical features. 

|Property_\_Dataset|income<br>cmc<br>karhunen<br>optdigits<br>diabetes<br>semeion<br>pixel<br>dna|
|---|---|
|||
|# Columns<br># Numerical<br># Categorical<br># Classes|14<br>9<br>64<br>64<br>8<br>256<br>240<br>180<br>6<br>2<br>64<br>64<br>8<br>256<br>240<br>0<br>8<br>7<br>0<br>0<br>0<br>0<br>0<br>180<br>2<br>3<br>10<br>10<br>2<br>10<br>10<br>3|



17 

Published as a conference paper at ICLR 2023 

## H EFFECTIVENESS OF THE MARGINAL DISTRIBUTION MASKING 

Table 10: Few-shot test accuracy (%) on 4 datasets from the OpenML-CC18 benchmark (Bischl et al., 2021) according to the masking type. We report the mean test accuracy over 100 different seeds. The bold denotes the highest mean score. 

|Masking type|income|cmc|semeion|pixel|Avg.|
|---|---|---|---|---|---|
||# shot|=1||||
|No-masking|59.18|36.23|54.04|75.99|56.36|
|Zero-masking|61.88|35.47|54.44|77.49|57.32|
|Gaussian noise|60.34|36.49|55.45|78.39|57.67|
|**Marginal distribution**|**63.52**|**37.10**|**55.91**|**79.05**|**58.90**|
||# shot|=5||||
|No-masking|70.94|38.76|72.12|86.38|67.05|
|Zero-masking|71.25|40.36|71.46|87.62|67.67|
|Gaussian noise|69.91|40.04|**73.64**|87.93|67.88|
|**Marginal distribution**|**72.69**|**40.40**|73.02|**89.08**|**68.80**|



We compare marginal distribution masking with widely used masking strategies in the tabular domain; the zero-masking (i.e., replace the masked column feature with a zero value) and Gaussian noise (i.e., add gaussian noise to the masked column feature used in SubTab (Ucar et al., 2021)). For comparisons, we also use 4 datasets from the OpenML-CC18 benchmark (Bischl et al., 2021) that are used for experiments in Section 4.3. As shown in the Table 10, all masking strategies show meaningful improvement over the no-masking case, where our marginal distribution masking shows the best result. 

Note that marginal distribution masking is a popular masking (and augmentation) scheme in many tabular models, such as SubTab (Ucar et al., 2021), VIME (Yoon et al., 2020), and SCARF (Bahri et al., 2022). On the other hand, zero-masking and Gaussian noise may have higher chances of generating unrealistic data points. For example, zero-masking makes the data too sparse, and adding Gaussian noise to one-hot encoded categorical features is unrealistic. 

18 

Published as a conference paper at ICLR 2023 

## I COMPARISON WITH AUGMENTATION-BASED UNSUPERVISED META-LEARNING SCHEMES 

Table 11: Few-shot test accuracy (%) on 8 datasets from the OpenML-CC18 benchmark (Bischl et al., 2021). We report the mean test accuracy over 100 different seeds. The bold denotes the highest mean score. 

|mean score.|||
|---|---|---|
|Method|income<br>cmc<br>karhunen<br>optdigit<br>diabetes<br>semeion<br>pixel<br>dna|Avg.|
|# shot = 1|||
|UMTRA + Gaussian noise<br>UMTRA + Marginal distribution masking<br>SES + Gaussian noise<br>SES + Marginal distribution masking<br>CACTUs|60.15<br>34.37<br>47.80<br>38.85<br>58.38<br>25.00<br>32.77<br>23.25<br>57.23<br>35.46<br>49.05<br>49.87<br>57.64<br>26.33<br>34.26<br>25.13<br>58.85<br>34.98<br>38.95<br>57.63<br>59.45<br>36.38<br>40.99<br>35.80<br>56.39<br>34.59<br>49.19<br>56.30<br>59.97<br>33.73<br>49.19<br>39.56<br>**64.02**<br>36.10<br>65.59<br>71.98<br>58.92<br>48.96<br>67.61<br>65.93|40.07<br>41.87<br>45.38<br>47.37<br>59.89|
|**STUNT (Ours)**|63.52<br>**37.10**<br>**71.20**<br>**76.94**<br>**61.08**<br>**55.91**<br>**79.05**<br>**66.20**|**63.88**|
|# shot = 5|||
|UMTRA + Gaussian noise<br>UMTRA + Marginal distribution masking<br>SES + Gaussian noise<br>SES + Marginal distribution masking<br>CACTUs|64.90<br>36.59<br>68.06<br>58.91<br>64.27<br>32.48<br>50.14<br>23.20<br>65.78<br>38.05<br>67.28<br>73.29<br>64.41<br>35.90<br>51.32<br>25.08<br>64.28<br>38.70<br>60.50<br>77.55<br>67.32<br>56.70<br>57.96<br>40.39<br>68.27<br>39.04<br>74.80<br>78.46<br>66.61<br>52.74<br>74.80<br>52.25<br>72.03<br>38.81<br>82.20<br>85.92<br>66.79<br>65.00<br>85.25<br>**81.52**|49.82<br>52.64<br>57.93<br>63.37<br>72.19|
|**STUNT (Ours)**|**72.69**<br>**40.40**<br>**85.45**<br>**88.42**<br>**69.88**<br>**73.02**<br>**89.08**<br>79.18|**74.77**|



We evaluate UMTRA (Khodadadeh et al., 2019) and SES (Ye et al., 2022) (also utilizing SNS proposed by Ye et al. (2022)) on few-shot tabular learning tasks, where we use augmentation strategies used in SubTab (Ucar et al., 2021) (i.e., Gaussian noise and marginal distribution masking). Here, we tried our best to improve the performance of SES and UMTRA (e.g., tune variance of Gaussian noise). However, unlike the image domain, they performed worse than CACTUs (Hsu et al., 2018), as shown in Table 11. We believe that the failures of SES and UMTRA are mainly due to the absence of effective augmentation strategies for tabular data, and developing them will be an interesting future direction. 

19 

