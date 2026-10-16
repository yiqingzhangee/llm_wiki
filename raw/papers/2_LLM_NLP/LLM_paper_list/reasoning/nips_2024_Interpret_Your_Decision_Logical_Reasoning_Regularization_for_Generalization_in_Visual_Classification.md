# **Interpret Your Decision: Logical Reasoning Regularization for Generalization in Visual Classification** 

**Zhaorui Tan**[1] _[,]_[2] **, Xi Yang**[1] _[∗]_ **, Qiufeng Wang**[1] **, Anh Nguyen**[2] **, Kaizhu Huang**[3] _[∗]_ 

1 Xi’an-Jiaotong Liverpool University 

2 University of Liverpool 

3Duke Kunshan University 

## **Abstract** 

Vision models excel in image classification but struggle to generalize to unseen data, such as classifying images from unseen domains or discovering novel categories. In this paper, we explore the relationship between logical reasoning and deep learning generalization in visual classification. A logical regularization termed L-Reg is derived which bridges a logical analysis framework to image classification. Our work reveals that L-Reg reduces the complexity of the model in terms of the feature distribution and classifier weights. Specifically, we unveil the interpretability brought by L-Reg, as it enables the model to extract the salient features, such as faces to persons, for classification. Theoretical analysis and experiments demonstrate that L-Reg enhances generalization across various scenarios, including multi-domain generalization and generalized category discovery. In complex real-world scenarios where images span unknown classes and unseen domains, L-Reg consistently improves generalization, highlighting its practical efficacy. 

## **1 Introduction** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0001-09.png)


**----- Start of picture text -----**<br>
Without L-Reg<br>With L-Reg<br>Seen domain Unseen domain<br>**----- End of picture text -----**<br>


One critical challenge in visual classification models is their ability to generalize effectively to unseen samples or unknown classes. For instance, a model trained on real images of various animals should ideally classify animal sketches accurately (referred to as multi-domain generalization classi- **With L-Reg** fication [20, 35, 34, 23, 25, 37, 50]) or Figure 1: GradCAM [45]45]] visualizations for the unknown discover novel categories not present class ‘person’ across seen and unseen domains of the GMDG in the training set (referred to as generalized category discovery [54, 16]). baseline with _L_ 2 regularization that is trained without and with L-Reg, respectively. Both experiments share the same These problems are prevalent in realhyper-parameters, except the latter uses the L-Reg. world scenarios, where training datatarget pairs are usually insufficient, and labeling is time-consuming so that not every data is paired with a label. Meanwhile, test data is likely to contain shifts in both data and targets, making it essential to propose methods that generalize to border scenarios. 

Figure 1: GradCAM [45]45]] visualizations for the unknown class ‘person’ across seen and unseen domains of the GMDG baseline with _L_ 2 regularization that is trained without and with L-Reg, respectively. Both experiments share the same hyper-parameters, except the latter uses the L-Reg. 

Regularization terms, such as _L_ 2 regularization leading to weight decay, are commonly employed during training to improve a model’s generalization capabilities. However, the _L_ 2 regularization 

> _∗_ Corresponding authors. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

is _parametric-based_ rather than _sample-based_ , which may lead to ambiguous interpretability [58]. As illustrated in Fig. 1, the model trained solely with _L_ 2 regularization exhibits low interpretability. Other regularization terms [57–59] attempt to improve the interpretability of deep learning models for sequential signals rather than vision, whereas [39] proposes a regularization term to enhance interpretability for robustness in visual classification models rather than generalization. Drawing inspiration from logical reasoning has shown promise for better generalization and interpretability in various tasks. Current work unveils the effectiveness of logical reasoning in generalization tasks, such as boosting performance in length generalization [1, 3, 2, 60] and abstract symbol relational reasoning [10, 36] (e.g., mathematical solving and psychological tests). Several efforts, such as [6], explore the explicit entropy-based logical explanations of neural networks for image classification, confirming the presence and interpretability of logical reasoning within visual tasks. Yet, there are limited studies tackling the generalization of visual classification tasks through the lens of logical reasoning. 

This paper studies two pivotal questions corresponding to the above: _1) How does logical reasoning relate to visual tasks such as image classification? 2) How can we derive a logical reasoning-based regularization term to benefit generalization?_ To achieve these, we correlate the image classification procedure in computer vision with the framework of logic studies [4], positing that training an image classifier involves learning a _good general_ logical relationship between images and labels via an encoder. This good general logic is attained when the semantics generated by the encoder and classifier can be combined to form atomic formulas. Our exploration leads to the introduction of a sample-based Logical regularization term named L-Reg. We reveal that L-Reg efficiently reduces the _complexity_ of the model from two aspects: 1) L-Reg leads to a balanced feature distribution in the semantic space; 2) L-Reg reduces the number of weights with extreme values in the classifier. 

Intuitively, the complexity reduction achieved by L-Reg stems from its ability to filter out redundant features or semantics, focusing instead on the minimal yet sufficient semantics for classification - defined as semantic support in Definition 3.2, where the interpretability also emerges. This filtering feature benefits the generalization when there is a domain shift in data where the domain-dependent features are ignored for classification. Moreover, it further promotes generalization when unlabeled data from the unknown classes is present. If such data lacks the semantic support associated with known classes, it is then classified as belonging to an unknown class, and its corresponding semantic supports are extracted. These capabilities equip L-Reg with explicit interpretability. As Fig. 1 shows, with L-Reg, the model can identify the unknown class ‘person’, and pinpoint faces which are the crucial features for classifying this category. In contrast, the model trained solely with _L_ 2 (without L-Reg) focuses on the ambiguous features for classification. 

Rigorous theoretical analysis and experimental results validate that L-Reg yields better generalization across diverse scenarios. Specifically, L-Reg facilitates better performance under the aforementioned multi-domain generalization and generalized category discovery tasks, whose settings are presented in Fig. 2 (a)(b). Furthermore, to evaluate L-Reg’s robustness, we introduce a more complex real-world scenario, as shown in Fig. 2 (c), where unlabeled images may not only belong to unknown classes, but also originate from unseen domains. Even in this challenging context, L-Reg is still able to consistently demonstrate notable improvements in generalization, underscoring its practical utility and effectiveness. Our code is available at `https://github.com/zhaorui-tan/L-Reg_NeurIPS24` . 

## **2 Preliminaries and generalization settings for visual classification** 

Consider paired ( _X, Y_ ) _∼_ ( _X , Y_ ), ( _Xs, Ys_ ) _∼_ **(a). Multigeneralization-domain category discovery(b). Generalized generalization (c). Multi-+domain Generalized** ( _Xs, Ys_ ), and ( _Xu, Yu_ ) _∼_ ( _Xu, Yu_ ) denote all sets ��| �� ��| �� �� �� **category discovery** ��| �� ��| �� of inputs and labels, seen paired subsets ofand unseen paired subsets of ( _X, Y_ ), respectively. ( _X, Y_ ), ��| �� �� | �� �� �� ��u�| �| ��� ����| �| ��� Note that _Xu, Yu_ may be accessible for the model **Data-shift** �� �� separately, but their pairing relationships are not ac- **Sets avaliable for training Target-shift** ����| �| ��� ���� | � | ��� cessible. Let _D_ denote the possible domains, with **Sets unavaliable for training Sets for evaluation** _Ds, Du ⊂ D_ representing the seen and unseen do- **All-shift** mains. In classification tasks, an encoding function Figure 2: Diagrams of different generaliza- _g_ ( _x_ ) _→ Z ∈_ R _[M]_ is commonly introduced to map _X_ tion settings in visual classification tasks. into the latent feature set _Z_ , where each latent feature has _M_ dimensions. A predictor _h_ ( _Z_ ) _→ Y_[ˆ] _∈_ R _[K]_ maps _Z_ to predictions _Y_[ˆ] , where _K_ denotes 

Figure 2: Diagrams of different generalization settings in visual classification tasks. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0003-00.png)


**----- Start of picture text -----**<br>
All classes Known  classes Unknown  classes<br>Without L-Reg<br>With L-Reg<br>Without L-Reg With L-Reg<br>(a). Heatmap of classifer’s weights (b). Distribution of values of classifer’s weights under classes<br>Known<br>Unknown<br>**----- End of picture text -----**<br>


Figure 3: Visualizations of classifiers’ weights form models trained using GMDG on PACS dataset without and with L-Reg under mDG+GCD setting, respectively. Both experiments share the same hyper-parameters using Regnety-16g backbone, except the latter uses additional L-Reg. 

the number of classes and the dimensions of predictions. _P_ ( _·_ ) and _H_ ( _·_ ) symbolize probability and entropy, respectively. This paper discusses two typical cases for generalization in image classification tasks: (1) _Data-shift generalization: Xs_ and _Xu_ have distribution shifts, such as multi-domain generalization (mDG); and (2) _Target-shift generalization: Ys_ and _Yu_ have distribution shifts, which stands for tasks like generalized category discovery (GCD). We additionally explore a challenging scenario called _All-shift generalization:_ both _Xs_ and _Xu_ , _Ys_ and _Yu_ have distribution shifts, which is a combination of mDG and GCD tasks (mDG + GCD). The following lists the detailed settings for generalization. Please refer to Fig. 2 for brief diagrams. 

**Data-shift generalization: Problem setting for mDG.** Illustrated in Fig. 2 (a), mDG [9] intends to generalize well to unseen domains having the objective of min _H_ ( _Xs, Ys | Ds_ ) and expecting the model to be generalized to _Xu_ when predicting _Yu_ from the unseen domain _Du_ . In such cases, _Yu_ is fully accessible to the model since _Ys_ and _Yu_ share the same domain: _Ys_ = _Yu_ but there are shifts in _X_ where _Xs_ = _Xu_ . 

**Target-shift generalization: Problem setting for GCD.** GCD [54] (Fig. 2 (b)) aims to discover possible unseen labels among unlabeled datasets _Xu_ . The challenge is that the samples in _Xu_ may belong to known classes or unknown classes: _Ys_ = _Yu_ and probably _Ys ∩Yu_ = _∅_ . The model should be able to distinguish the samples from the known classes and cluster the samples for unknown classes simultaneously. Note that _Xu_ is used for model training, but the relationship between _Xu_ and _Yu_ is unseen for the model. In summary, shifts exist between _Ys_ and _Yu_ but not between _Xs_ and _Xu_ . 

**All-shift generalization: Problem setting for mDG + GCD.** To explore the generalization problem further, we introduce a setting that is the combination of mDG and GCD as shown in Fig. 2 (c). Specifically, the model is trained on the labeled pairs ( _Xs, Ys_ ) and unlabeled set _Xu_ from the seen domains _Ds_ ; _Xu_ may belong to known and unknown classes. Furthermore, the model is tested on _Xu_ from the unseen domain _Du_ , where _Xu_ may also come from the known and unknown classes. In this setting, the model is expected to 1) classify samples to the seen classes and discover the unseen classes among unlabeled samples from seen domains and 2) generalize this ability to the samples from the unseen domain. In this scenario, _Xs_ and _Xu_ have shifts, and so do _Ys_ and _Yu_ . 

For all aforementioned generalization settings, the objective can be summarized as minimizing the _generalization loss_ : 

**Definition 2.1** (Generalization loss) **.** Let the target model _f[∗]_ : _f[∗]_ ( _X, Y_ ) : _X → Y_ , can generalize across both seen and unseen sets _X, Y_ . Denote its trainable _f_ , which is only trained on the seen sets. The generalization loss for the unseen sets is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0003-08.png)


## **3 Logical regularization for generalization in image classification** 

Under the problem settings defined in Section 2, we introduce Logic regularization (L-Reg) targeting the objective: 

min _h,g_[E] _[z][i][∈][z,z][∈][Z]_[[] _[H]_[( ˆ] _[Y][ |][z][i][, D]_[)]] _[−]_[E] _[z][∈][Z]_[[] _[H]_[( ˆ] _[Y][ |][Z, D]_[)]] _[,]_ (2) 

3 

where _Y_[ˆ] _∈_ R _[K]_ = _h ◦ g_ ( _X_ ) is the prediction set. The corresponding Logic regularization loss (L-Reg) is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0004-01.png)


where _σj,i_ ( _Y_[ˆ] _[T] Z_ ) denotes the value at the _i, j_ position of _softmax_ ( _Y_[ˆ] _[T] Z_ ) and the soft-max function is applied at the last dimension. By incorporating other existing methods’ losses denoted by _Lmain_ , the overall loss is formulated as: 

_Lall_ = _Lmain_ + _αLL−Reg,_ (4) 

with a weight _α_ applied to balance two losses. As depicted in Fig. 1, L-Reg plays a pivotal role in extracting crucial features for image classification, thus enhancing generalization capabilities. This beneficial outcome can be attributed to two primary factors: 

**Reducing classifier complexity:** L-Reg streamlines the complexity of the classifier itself, as depicted in Fig. 3 (a). Notably, the heat map of the model with L-Reg displays fewer extremely valued weights, evidenced by the diminished presence of intense blue and red colors. This reduction implies that the classifier focuses on leveraging semantically rich and relevant features for decision-making (classification), sidelining the less relevant ones. Additionally, Fig. 3 (b) reveals a reduction in the number of semantic features used to classify each class. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0004-06.png)


**----- Start of picture text -----**<br>
Distance between values with most counts of known and unknown classes<br>Without L-Reg With L-Reg<br>**----- End of picture text -----**<br>


**Balancing feature complexity:** L-Reg results in a more balanced distribution of features compared to the baseline, as illustrated in Fig. 4. This balanced distribution suggests the elimination of certain extracted semantics characterized by dominant frequencies across all samples. Semantics that occur frequently across samples often lack decisiveness for classification. Hence, reducing their prominence contributes to more expressive feature space and less complex feature distributions. Coupled with the reduced classifier complexity, a simplified classifier achieved through L-Reg facilitates improved generalization across various settings. Specifically, the top row also on indicates the distance between the feature distributions of Reg the known and unknown classes, which is enlarged; thus, they are more dividable, leading to classification improvements. 

Figure 4: Visualizations of latent features form models trained using GMDG on PACS dataset without and with L- Reg under mDD+GCD setting using RegNetY-16G backbone, respectively. 

We present a logical-based theoretical analysis in Section 3.1 and provide the derivation details of L-Reg in Section 3.2. In addition, we discuss the efficacy of L-Reg under various generalization settings in Section 4. Furthermore, L-Reg serves as a plug-and-play loss function that is compatible with most existing frameworks. We conduct experiments applying L-Reg to various established approaches across different generalization settings, as outlined in Section 5. 

## **3.1 Logical framework for visual classification** 

This part provides the connections between logical reasoning and visual classification tasks. We would like to remind readers of the framework for studying logics and link it with our practical scenarios. 

**Definition 3.1.** Following [4], a logic _L_ is defined as a five-tuple in the form: 

_L_ = _⟨FL, ML, |_ = _L, mngL, ⊢L⟩ ,_ (5) 

where 1) _FL_ denotes the set of formulas formed by images and labels ( _X, Y_ ); 2) _ML_ represents different domains _D_ of _X_ ; 3) _|_ = _L_ is a binary relation relating the truth of whether the formulas are true or false, which has _|_ = _L⊆ ML × FL_ ; 4) _mngL_ : _FL × ML −→_ Sets defines the meaning of _X_ as determined by classifiers, where Sets indicate the class of all sets. (5) _⊢L_ symbolizes the provability relation of _L_ , evaluating formulas formed by _mngL_ is true or false in one possible world, such as the estimation criteria. More details of _L_ can be seen in Appendix B. 

4 

For clarity, we specify _L_ ( _Xs,Ys_ ) = � _F_ ( _Xs,Ys_ ) _, D, |_ =( _Xs,Ys_ ) _, h, ⊢_ ( _h_ ( _X_ ) _,Y_ )� as the logic formed on the given _X, Y_ sets. With the goal for logic to generalize across a broader scenario and provide extrapolation across all possible formulas in _L_ , a good general logic _L[∗]_ should be derived from _L_ through the feature extractor _g_ : 

_L[∗]_ = � _F_ ( _g_ ( _Xs_ ) _,Ys_ ) _, D, |_ =( _g_ ( _Xs_ ) _,Ys_ ) _, h, ⊢_ ( _h◦g_ ( _X_ ) _,Y_ )� _, s.t., ⊢_ ( _h◦g_ ( _X_ ) _,Y_ )= _|_ =( _g_ ( _Xs_ ) _,Ys_ ) _._ (6) Importantly, as a good general logic, _F_ ( _g_ ( _Xs_ ) _,Ys_ ) and _h_ in _L[∗]_ should form the _atomic formulas_ , i.e., the tuple of terms with a predicate: _h ◦ g_ ( _x_ ) belongs/not belongs to class _y_ in domain _d → Ture/False,_ where _x, y, d ∈ X, Y, D_ , which makes that _⊢_ ( _h◦g_ ( _Xu_ ) _,Yu_ )= _|_ =( _g_ ( _Xs_ ) _,Ys_ ) still holds. We simply denote one atomic formula in the form of _h_ ( _g_ ( _x_ ) _, y, d_ ) mapping to binary values. Additionally, _⊢_ ( _h◦g_ ( _X_ ) _,Y_ )= _|_ =( _g_ ( _Xs_ ) _,Ys_ ) in Eq. (6) can be safely omitted in the rest of the paper. Please see more details about the conditions of the good general logic in Appendix B. 

An additional tool is necessary to convert the logic problem into a continuous form, enabling the application of machine learning algorithms. The conditional entropy-based method enables a logically sound derivation of knowledge from the provided dataset with constraints [43]. Specifically, the probabilistic inference process adheres to a probabilistic version of Modus Ponens: _A → B, A ⊢ B_ (if _A_ then _B_ ; not _A_ therefore not _B_ ). It is important to note that the logical propositions in probabilistic Modus Ponens are uncertain, with the conditional probability replacing the material implication _A → B_ . This framework allows us to interpret logical deduction through the lens of entropy. Therefore, for Eq. (6) which implies 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0005-03.png)


finding _h ◦ g_ through optimization is equivalent to 

max _h,g_[E][(] _[x,y]_[)] _[∈]_[(] _[X,Y]_[ )] _[,d][∈][D][P]_[(] _[y][|][g]_[(] _[x]_[)] _[, d]_[)] _[ −R ⇐⇒]_[min] _h,g_[E][(] _[x,y]_[)] _[∈]_[(] _[X,Y]_[ )] _[,d][∈][D][H]_[(] _[y][|][g]_[(] _[x]_[)] _[, d]_[) +] _[ R][,]_ (8) 

where _R_ denotes any other possible regularization. 

As the logical framework for image classification takes shape, it becomes evident that the unresolved question of identifying an appropriate function _g_ to generate suitable atomic formulas emerges as a critical factor in ensuring the effectiveness of the overarching logic _L[∗]_ . This paper proposes L-Reg as the regularization to ensure _F_ ( _g_ ( _Xs_ ) _,Ys_ ) are formed by atomic formulas in Section 3.2. 

## **3.2 Constructing atomic formulas using L-Reg** 

In this part, we show the derivation details of L-Reg the aims to ensure the formation of suitable atomic formulas, as depicted in Eq. (6). As highlighted in [1], current algorithms may induce implicit biases towards unseen data, resulting in varied solutions for such data. However, expecting an algorithm to generalize effectively to unseen data domains without appropriate incentivization, such as specifically designed regularization, is unreasonable. Therefore, we aim to enhance the generalization capability of models by employing a logic-based regularization approach. To this end, we introduce the concept of _semantic support_ for image classification. 

**Definition 3.2** (Semantic support) **.** We denote _z_ = _g_ ( _x_ ), where _z ∈ Z_ , as a set of compositions of these semantics: _z_ := _{z[i] }[M] i_ =1[, where] _[ M]_[is the number of dimensions or semantics.][Notably, not] all semantics in _z_ may be useful for deduction or inference. We define the subset _γ_ of _z_ , extracted from the sample _x ∼X_ , as the semantic support of _x_ if _γ_ is sufficient for deducing the relationship between _x_ and a _y ∼Y_ . 

For instance, if the subset _{z_[1] _, z_[2] _} ⊆ z_ is sufficient for accurate inference, the values of other semantics _{z[i] }[M] i_ =3[will not impact the inference process.][When] _[ {][z]_[1] _[, z]_[2] _[}]_[ constitutes the minimal] combination of semantics required for inference, it is termed the semantic support. We denote Γ as the set of semantic supports of _X_ for deducing each individual class. 

**Derivation of L-Reg.** Regarding Eq. (6), if the semantic supports and their relationship with _Y_ form atomic formulas, Eq. (6) holds as a good general logic, and the generalization would be improved. Thus, we aim to learn the latent features _Z_ , which contain sufficient semantic supports for the deduction of _Y_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0005-13.png)


Specifically, _g_ ( _·_ ) should meet the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0005-15.png)


5 

i.e., the semantic support set for each class should be distinct. The multiple-class classification task has that _∀_ Γ _, |_ Γ _| ≤ M_ . Under the constraints demonstrated in Eq. (9) and Eq. (10), we need to achieve the following through optimization: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0006-01.png)


> where Γ[¯] denotes the negation of Γ, i.e., the set of semantics which does not include semantic support. 

Intuitively, Eq. (11) regularizes that the model should be able to judge whether a sample belongs to a class by using a minimal set of semantic supports; simultaneously, the semantic support sets are also implicitly disentangled for each class, not only for maintaining rich and useful semantics but also for enhancing the independence of deduction of each class. The actual collection of Γ appears to be intractable during optimization. Hence, we resort to deriving its bounds. Regarding Eq. (11), its former term can be elaborated as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0006-04.png)


where _zi_ is minimal semantics form _z_ , and E _[M] i_ =1 _[H]_[(] _[Y][ |][g]_[(] _[z][i]_[)] _[, D]_[)][is][the][upper-bound][for] min _h,g H_ ( _Y |g_ (Γ) _, D_ ). Therefore, minimizing E _[M] i_ =1 _[H]_[(] _[Y][ |][g]_[(] _[z][i]_[)] _[, D]_[)][is][equivalent][to][minimizing] _H_ ( _Y |g_ (Γ) _, D_ ). Meanwhile, for the latter in Eq. (11), we have: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0006-06.png)


where _H_ ( _Y |h_ ( _z_ )) _, D_ ) is the lower-bound for max _h,g H_ ( _Y |g_ (Γ)[¯] _, D_ ). Combining the aforementioned bounds, we have the L-Reg objective as Eq. (2). 

**Interpretability of semantic supports roots in forming atomic formulas.** The atomic formula _A[y]_ is of the form _h_ ( _g_ ( _x_ ) _, y, d_ ). Our aim is to find the good (most) general _A[y][∗] ∈A[y]_ for _y_ class from which the interpretability of L-Reg is derived. Consider _A[y]_ 1 _[,][ A][y]_ 2 _[∈A][y]_[, if] _[ A][y]_ 1[is more general than] _A[y]_ 2[, there will be a substitution] _[ ψ]_[ such that] _[ A][y]_ 1 _[ψ]_[=] _[ A][y]_ 2[[][52][].] _[A][y][∗]_[should meet] _[ A][y][∗][ψ]_[=] _[ A][y] i[∈A][y]_[,] which infers that _γ[y] ψ_ = _z[y]_ (cf. Eq. (9)) for predication of _y_ where _γ[y]_ is the semantic support. Note here that the form of _A[y]_ is constructed for _y ∈ Y_ , i.e., predicate whether the sample belongs to the _y_ class. Considering multiple classes _yi, yj ∈ Y, i ̸_ = _j_ , it has _A[y][i][∗]_ = _A[y][j][∗]_ thus _γ[y][i]_ = _γ[y][j]_ (cf. Eq. (10)), which constrains that different minimal semantic supports should be used for predicting different classes. The interpretability of L-Reg is based on _A[y][∗]_ , compelling the model to use distinct minimal semantic supports for each class. These minimal semantic supports can be interpreted as the most critical features for efficient prediction. For example, as shown in Fig. 1, the model with L-Reg has learned the facial features of the person class (see more examples in Appendix Figs. 7 to 12), forming the (informal) atomic formula _h_ (has a human face _,_ is person _, d ∈ D_ ) _→_ True. Similarly, it also leads to _h_ (not has a human face _,_ is person _, d ∈ D_ ) _→_ False. 

## **4 L-Reg under different generalization settings** 

**L-Reg under data-shift generalization.** The task mDG endeavors to facilitate a model’s ability to generalize to unseen domains by fostering invariance across seen domains [50]. In the context of mDG, the term _|D| ≥_ 2 in Eq. (8) typically denotes multiple domains. Traditionally, existing methods focus on minimizing domain gaps, leading to remarkable results [25, 50]. However, it is noteworthy that even when the domain gap is effectively minimized, and _|D|_ = 1 for the latent features can be considered, L-Reg still demonstrates its efficacy in promoting the generalization of _Xu_ from _Du_ . 

**Proposition 4.1** (Effectiveness of L-Reg in enhancing data-shift generalization.) **.** _Assume the gap across all domains is well minimized. Let f[∗] denote the target model that generalizes to the data Xu from the unseen domain with the lowest complexity. For a model f_ ( _[R] Xs,Ys_ ) _[, f]_[(] _[X] s[,Y] s_[)] _[trained under the] data-shift generalization setting (i.e.,_ ( _Xs, Ys_ ) _is accessible and Ys_ = _Yu). We have:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0006-12.png)


Please see proof details in Proposition C.1. To illustrate Proposition 4.1, consider the following intuitive example: In the seen domains, all cats are either black or white, while all dogs are brown. Now, imagine encountering a sample labeled ‘a brown cat’ from an unseen domain. Without the application of L-Reg, the model might erroneously classify it as a dog. However, with L-Reg in 

6 

Table 1: MDG results: Comparison between the proposed and previous non-ensemble and ensemble mDG methods. The best results for each group are highlighted in **bold** . Improvement and degradation in our approach from GMDG are highlighted in red. 

|Test domain<br>**PACS**<br>**VLCS**<br>**OffceHome**<br>**TerraIncognita**<br>**DomainNet**|**Avg.**|
|---|---|
|MMD [33]<br>84.7±0.5<br>77.5±0.9<br>66.3±0.1<br>42.2±1.6<br>23.4±9.5<br>Mixstyle [62]<br>85.2±0.3<br>77.9±0.5<br>60.4±0.3<br>44.0±0.7<br>34.0±0.1<br>GroupDRO [44]<br>84.4±0.8<br>76.7±0.6<br>66.0±0.7<br>43.2±1.1<br>33.3±0.2<br>IRM [5]<br>83.5±0.8<br>78.5±0.5<br>64.3±2.2<br>47.6±0.8<br>33.9±2.8<br>ARM [61]<br>85.1±0.4<br>77.6±0.3<br>64.8±0.3<br>45.5±0.3<br>35.5±0.2<br>VREx [30]<br>84.9±0.6<br>78.3±0.2<br>66.4±0.6<br>46.4±0.6<br>33.6±2.9<br>CDANN [35]<br>82.6±0.9<br>77.5±0.1<br>65.8±1.3<br>45.8±1.6<br>38.3±0.3<br>DANN [20]<br>83.6±0.4<br>78.6±0.4<br>65.9±0.6<br>46.7±0.5<br>38.3±0.1<br>RSC [24]<br>85.2±0.9<br>77.1±0.5<br>65.5±0.9<br>46.6±1.0<br>38.9±0.5<br>MTL [8]<br>84.6±0.5<br>77.2±0.4<br>66.4±0.5<br>45.6±1.2<br>40.6±0.1<br>MLDG [31]<br>84.9±1.0<br>77.2±0.4<br>66.8±0.6<br>47.7±0.9<br>41.2±0.1<br>Fish [46]<br>85.5±0.3<br>77.8±0.3<br>68.6±0.4<br>45.1±1.3<br>42.7±0.2<br>ERM [53]<br>84.2±0.1<br>77.3±0.1<br>67.6±0.2<br>47.8±0.6<br>44.0±0.1<br>SagNet [40]<br>86.3±0.2<br>77.8±0.5<br>68.1±0.1<br>48.6±1.0<br>40.3±0.1<br>SelfReg [26]<br>85.6±0.4<br>77.8±0.9<br>67.9±0.7<br>47.0±0.3<br>42.8±0.0<br>CORAL [48]<br>86.2±0.3<br>78.8±0.6<br>68.7±0.3<br>47.6±1.0<br>41.5±0.1<br>mDSDI [12]<br>86.2±0.2<br>79.0±0.3<br>69.2±0.4<br>48.1±1.4<br>42.8±0.1|58.8<br>60.3<br>60.7<br>61.6<br>61.7<br>61.9<br>62.0<br>62.6<br>62.7<br>62.9<br>63.6<br>63.9<br>64.2<br>64.2<br>64.2<br>64.5<br>65.1|
|Use RegNetY-16GF [47] as oracle model.||
|MIRO [25] (ECCV23)<br>97.4±0.2<br>79.9±0.6<br>80.4±0.2<br>58.9±1.3<br>53.8±0.1<br>GMDG [50] (CVPR24)<br>97.3±0.1<br>82.4±0.6<br>80.8±0.6<br>60.7±1.8<br>54.6±0.1|74.1<br>75.1|
|**GMDG + L-Reg**<br>**97.4**±0.20_._1_↑_<br>**82.4**±0.00_._1_↑_<br>**80.9**±0.50_._1_↑_<br>**62.9**±0.92_._2_↑_<br>**55.3**±0.00_._8_↑_|**75.8**0_._7_↑_|



place, the model is compelled to rely on minimal semantics for classification. This means filtering out irrelevant features such as color terms, thus enabling more accurate deductions. 

**L-Reg under target-shift generalization.** We demonstrate how L-Reg enhances generalized discovery in scenarios where only a subset of classes ( _Ys_ ) is available for training, and there may exist an overlap between the unseen classes ( _Yu_ ) and the seen classes ( _Ys_ ), denoted as _Yu ∩Ys_ = _∅_ . We define _Yu/Ys_ as the novel classes not included in _Ys_ , and _Yu ∼ Ys_ as the seen classes for _Xu_ classification, where _|D|_ = 1. Building upon Proposition 4.1, L-Reg further enhances GCG by improving the generalization performance on _Yu_ . 

**Proposition 4.2** (L-Reg improves target-shift generalization) **.** _When |D|_ = 1 _, L-Reg promotes generalization performance on Yu under the target-shift scenario._ 

_Proof._ When _|D|_ = 1, since all _Y_ belongs to a close set, minimizing _−H_ ( _Ys|g_ (Γ)[¯] _, D_ ) is equivalent to the following: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0007-06.png)


where _Y_[¯] _s_ is the negation of _Ys_ , i.e., _Yu/Ys_ . In this situation, if one sample does not contain sufficient semantic support to be classified under _Ys_ , it otherwise will be assigned under _Y/Ys_ , promoting performance for both _Yu/Ys_ and _Yu ∼ Ys_ . Therefore, the generalization performance on the unseen classes will be improved by L-Reg. 

**L-Reg under all-shift generalization.** When the domain gap is sufficiently minimized and _|D|_ = 1 can be considered, the combination of Proposition 4.1 and Proposition 4.2 demonstrates that L-Reg enhances generalization performance on both novel classes ( _Yu/Ys_ ) and seen classes ( _Yu ∼ Ys_ ) for _Xu_ from other domains. Our experiments validate that L-Reg, when applied in scenarios with well-minimized domain gaps, consistently improves generalization across all shifts. 

## **5 Experiments** 

To validate L-Reg, three groups of experiments under the three kinds of settings are conducted. Notably, all baselines we used already incorporate the _L_ 2 regulation in the form of weight decay. We also compare other commonly used regularization terms, such as independence or sparsity regularization on _Z_ . More results in Appendix Findicate that our L-Reg also surpasses them. 

## **5.1 Experiments on mDG** 

**Experimental settings.** We operate on the DomainBed suite [21] and leverage standard leaveone-out cross-validation as the evaluation protocol. We test L-Reg with GMDG [50] on 5 realworld benchmark datasets: PACS [32], VLCS [18], OfficeHome [55], TerraIncognita [7], and 

7 

DomainNet [42]. Following MIRO [25] and GMDG [50], the RegNetY-16GF backbone with SWAG pre-training [47]) is used. Specifically, we train the backbone using GMDG with L-Reg. Accuracy is adopted as the evaluation metric, and the results of the averages from three trials of each experiment, with standard deviations, are presented. See Supplementary H for more experimental details. 

**Results.** The experimental results presented in Table 1 demonstrate the efficacy of L-Reg in improving the performance of GMDG across all datasets in mDG classification tasks. Notably, more substantial improvements are observed when the GMDG baseline achieves relatively low accuracy. These observed enhancements provide empirical support for Proposition 4.1. Please see using L-Reg with basic ERM in Appendix E. For detailed insights into each domain within each dataset, please refer to Appendix H.1. 

## **5.2 Experiments on GCD** 

**Experimental settings.** We validate our approach through training PIM additionally with L-Reg. Six image datasets are adopted to validate the feasibility of our proposed RPIM compared to other competitors, including three generic object recognition datasets, CIFAR10 [29], CIFAR100 [29] and ImageNet-100 [17]; two fine-grained datasets CUB [56] and Stanford Cars [28]; and the longtail dataset Herbarium19 [49]. Following prior works [54, 16], we use the proposed accuracy metric from [54] of all classes, known classes, and unknown classes for evaluation. Please see a detailed description of the experimental setup in Appendix H.2. 

**Results.** The average results across all datasets for utilizing L-Reg with PIM are presented in Table 2, while detailed dataset-specific information is available in Appendix H Table 17. The results highlight that L-Reg consistently increases the accuracy of all unknown classes across all datasets, thus confirming the validity of Proposition 4.2. However, it is notable that L-Reg may marginally compromise the performance of known classes, as it reduces the size of semantic support for deducing _Y_ , thereby reducing the information available for known classification. Nevertheless, this compromise is deemed acceptable given the significant improvements observed for the unknown classes. 

Table 2: GCD results: Average results across all datasets of PIM with L-Reg. Improvements and degradation are highlighted in red and blue, respectively. 

|Average|All<br>Known<br>Unknown|
|---|---|
|K-means [38]<br>RankStats+ [22] (TPAMI-21)<br>UNO+ [19] (ICCV-21)<br>ORCA [13] (ICLR-22)<br>ORCA - ViTB16<br>GCD [54] (CVPR-22)<br>RIM [27] (NeurIPS-10)<br>TIM [11] (NeurIPS-20)|44.7<br>46.0<br>43.9<br>38.6<br>54.6<br>25.6<br>51.2<br>74.5<br>36.7<br>46.3<br>51.3<br>41.2<br>56.7<br>65.6<br>49.9<br>60.4<br>71.8<br>52.9<br>62.0<br>72.5<br>55.4<br>62.7<br>72.6<br>56.4|
|PIM [16] (ICCV-23)|67.4<br>**79.3**<br>59.9|
|**PIM + L-Reg**|**68.8**1_._4_↑_<br>79.00_._3_↓_<br>**62.7**2_._8_↑_|



Table 3: MDG+GCD results: Averaged accuracy scores for all, known and unknown classes across all five datasets. Improvements and degradation are highlighted in red and blue respectively. 

|Method<br>Domaingap|All<br>Known<br>Unknown|
|---|---|
|ERM<br>Not<br>**+L-Reg**<br>minimized|44.69<br>59.33<br>23.54<br>45.50<br>61.43<br>21.63|
|Imp.|0.81<br>2.09<br>-1.91|
|PIM<br>Not<br>**+L-Reg**<br>minimized|46.95<br>60.35<br>26.90<br>47.27<br>60.83<br>26.34|
|Imp.|0.32<br>0.48<br>-0.57|
|MIRO<br>Not suffciently<br>**+L-Reg**<br>minimized|49.67<br>68.86<br>25.79<br>52.11<br>71.26<br>26.49|
|Imp.|2.44<br>2.39<br>0.71|
|GMDG<br>Suffciently<br>**+L-Reg**<br>minimized|47.94<br>68.75<br>20.68<br>51.94<br>69.87<br>27.68|
|Imp.|4.00<br>1.12<br>7.01|



## **5.3 Experiments on mDG + GCD** 

**Experimental settings.** We utilize datasets designed for mDG tasks to conduct mDG + GCD experiments. During the training stage, only samples from seen domains are available, with half of the classes masked as unknown, and only their unlabeled data are utilized. Notably, even though all the unlabeled data originates from unknown classes during training, this prior knowledge is not assumed or constrained, aligning the setting with GCD. Similar to mDG, we adopt the leave-one-out cross-validation method. This entails testing each domain in each dataset as the unseen domain. The performance is tested on unseen domains by employing GCD metrics. To validate L-Reg’s efficacy comprehensively, we re-implement four methods under the mDG + GCD setting, testing them both with and without L-Reg. The four methods include ERM, PIM, MIRO, and GMDG. ERM serves as the baseline approach without additional regularization, while PIM maximizes information without minimizing domain gaps. MIRO and GMDG focus on minimizing domain gaps, with GMDG offering a comprehensive approach in this regard. It is worth noting that PIM has been re-implemented. For further experimental details, please refer to Appendix H.3. 

**Results.** The averaged results across all unseen domains of all datasets are summarized in Table 3. For a detailed breakdown of results for each domain in each dataset, please refer to Appendix H.3. As discussed in Proposition 4.1 and Proposition 4.2, a noticeable trend is observed wherein, as the 

8 

domain gap is gradually minimized, the improvements for unknown classes increase, with the best results achieved using GMDG with L-Reg. Known class **Unknown class** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0009-01.png)


**----- Start of picture text -----**<br>
Known class Unknown class<br>Without L-Reg With L-Reg Without L-Reg With L-Reg<br>Photo<br>cartoon<br>Seen domains<br>Sketch<br>Art painting<br>Useen domain<br>**----- End of picture text -----**<br>


## **L-Reg forms atomic formulas and improves in-** 

**terpretability.** Furthermore, Fig. 5 provides visual insights into the behavior of models trained with L- Reg. Evidently, these models tend to focus on minimal semantics sufficient for class distinctions. For the known classes, the efficacy of L-Reg can be intuitively understood as extracting the minimal semantic supports for a given class label. For instance, the presence of a guitar’s fingerboard, even in unseen domains, helps classify a sample as belonging to the guitar category, whose informal forms can be denoted as _h_ (has fingerboard _,_ is guitar _, d ∈ D_ ) _→_ True and _h_ (not has fingerboard _,_ is guitar _, d ∈ D_ ) _→_ False. For all known classes, samples with these minimal semantic supports are recognized accordingly. In contrast, if a sample lacks these minimal supports for any known class, it is very likely categorized as an unknown class. This behavior stems from Paper Eq.10 which ensures _A[y][i][∗]_ = _A[y][j][∗]_ through constraining _γ[y][i]_ = _γ[y][j]_ . L-Reg further enhances the model’s ability to identify minimal supports for unknown classes 

Figure 5: GradCAM visualizations of GMDG trained without and with L-Reg. The seen, unseen domains and known, unknown classes are denoted. 

by filtering out co-covariant features associated with other classes and thus generalizing to unseen domains. Therefore, the very interpretable features for unknown classes from unseen domains can be extracted using L-Reg. Fig. 5 (right side) demonstrates that the model with L-Reg can even extract facial features for the unknown person class and can generalize this to the unseen domain. Similarly, here we obtain (informal) atomic formulas as _h_ (has a face _,_ is person _, d ∈ D_ ) _→_ True, _h_ (not has a face _,_ is person _, d ∈ D_ ) _→_ False. 

However, as shown in Row 3, significant domain shifts, such as those between the sketch domain and other domains, pose challenges. Specifically, the differences between the stick-figure style of sketches of persons and figures from other domains can hinder the model’s ability to cluster sketches with other domains’ figures when the class label is unknown. Thus, under this circumstance, the model may fail to extract meaningful features from those sketches. We acknowledge this limitation and will explore solutions in future work. 

Table 4: Averaged results of applying L-Reg to different layers across domains in PACS. 

**L-Reg should be applied to features from deep layers.** One crucial precondition highlighted in the theoretical analysis is that L-Reg operates effectively with a representation _Z_ , where each dimension represents independent semantics. The semantic features usually come from the deeper layers of the model 

||All<br>Known<br>Unkown|
|---|---|
|GMDG|58.33<br>91.46<br>10.18|
|L-Reg: Deep layer|**67.82**<br>**91.86**<br>31.33|
|L-Reg: Earlier and the deep layers|58.97<br>80.73<br>**35.05**|



architecture [51]. However, Table 4 shows that applying L-Reg to features from earlier layers, which may not necessarily represent semantics, leads to a degradation in performance for known classes, albeit improving performance for unknown classes. This phenomenon arises due to the potential interdependence among features from earlier layers, resulting in penalization that may hinder the capture of semantic supports essential for known classes. To ensure generalization improvements without significant compromise to the performance of known classes, we advocate for applying L-Reg specifically to features extracted from deeper layers, such as the bottleneck layer. These suggest that the compromised results observed in Table 2 could be attributed to the less depth of the model structure, which fails to provide the expected semantic features. 

## **5.4 Apply L-Reg to congestion prediction for circuit design.** 

**Experimental settings.** We also test L-Reg in Congestion prediction on the CircuitNet [15] dataset by using CircuitFormer [63] backbone. The congestion prediction is for circuit design and benefits from logical reasoning-based approaches. All parameters, except for L-Reg, remain consistent with CircuitFormer, and we follow its metrics. 

9 

**Results.** Table 5 shows the results of predicTable 5: tion results on the CircuitNet dataset. We also include the results of Gpdl with UNet++ and CircuitFormer for better comparison. Notably, the improvements brought by CircuitFormer with L- Reg across all metrics, especially for the pearson **CircuitFormer + L-Re** metric can be observed. The consistent improvement with L-Reg across all metrics indicates L-Reg’s feasibility. 

Table 5: **Results of Congestion prediction:** Congestion prediction is proposed for circuit design. 

||pearson<br>spearman<br>kendall|
|---|---|
|Gpdl with UNet++<br>CircuitFormer (SOTA)|0.6085<br>0.5202<br>0.3855<br>0.6374<br>0.5282<br>0.3935|
|**CircuitFormer + L-Reg (Ours)**|**0.6553**<br>**0.5289**<br>**0.3944**|



## **6 Related work** 

**Logical reasoning for deep learning.** Current studies focus on length generalization or symbolic reasoning in the logic-based scope. For length generalization, [1] proposes the generalization to the unseen setting, theoretically verifying that commonly used models can generalize to the unseen and degree curriculum promotes the generalization ability of the transformer, followed by [3, 2, 60]. Another branch is to improve the logical reasoning ability for abstract symbols, such as learning the logical-based temples and expecting the model to generalize to unseen samples [10, 36]. These studies are closely related to languages, such as generating longer answering sequences or solving mathematical problems in large language models, lacking explicit connections to visual tasks. [6] delves into the logical explanations in image classification by explicitly extracting logical relationships. While this logical-based approach sheds light on the interpretability of image classification models, its specific benefits for visual generalization remain relatively unexplored. 

**Multi-domain generalization.** Current approaches for mDG in image classification focus on learning invariant representation across domains. Previous approaches like DANN [20] minimize feature divergences between source domains. CDANN [35], CIDG [34], and MDA [23] consider conditions for learning conditionally invariant features. MIRO [25] and GMDG [50] take advantage of pretrained models to improve generalization. Specifically, in comparison to MIRO, GMDG proposes a general entropy-based learning objective for mDG and sufficiently minimizes the domain gaps, yielding better generalization results. 

**Generalized category discovery.** Generalized category discovery, pioneered by [54], addresses unlabeled samples with both known and unknown classes. Furthermore, PIM [16] integrates InfoMax into generalized category discovery, effectively handling imbalanced datasets and surpassing GCD on both short- and long-tailed datasets. 

## **7 Conclusion** 

This paper presents L-Reg, a logical regularization approach tailored for image classification tasks using logic analysis frameworks. L-Reg yields better generalization across different settings by fostering balanced feature distributions and streamlining the classification model’s complexity. Rigorous theoretical analyses and empirical validations underscore its efficacy, as L-reg consistently improves generalization performance with different frameworks under various scenarios. 

**Limitation.** L-Reg narrows the extent of semantic supports, potentially diminishing the amount of information available for classification and leading to certain trade-offs in the performance of seen datasets. This effect is evidenced by the slight decline in the accuracy of known classes when L-Reg is applied, as shown in Table 2. A similar phenomenon is observed in Fig. 5, where the model fails to recognize a person in the sketch domain lacking facial features. Analysis from Table 4 suggests that these compromises may result from improper _Z_ . Future work should focus on mitigating potential compromises on seen datasets by exploring strategies for better capturing _Z_ through improved model architecture design. We offer more experimental results of possible solutions to this limitation in Appendix G, such as further constraining the independence of each dimension in _Z_ . Those results may suggest a direction for future work. 

## **Acknowledgments** 

The work was partially supported by the following: National Natural Science Foundation of China under No. 92370119, No. 62376113, No. 62206225, and No. 62276258. 

10 

## **References** 

- [1] Emmanuel Abbe, Samy Bengio, Aryo Lotfi, and Kevin Rizk. Generalization on the unseen, logic reasoning and degree curriculum. In _International Conference on Machine Learning_ , pages 31–60. PMLR, 2023. 

- [2] Emmanuel Abbe, Elisabetta Cornacchia, and Aryo Lotfi. Provable advantage of curriculum learning on parity targets with mixed inputs. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [3] Kartik Ahuja and Amin Mansouri. On provable length and compositional generalization. _arXiv preprint arXiv:2402.04875_ , 2024. 

- [4] Hajnal Andréka, István Németi, and Ildikó Sain. Universal algebraic logic. _Studies in Logic, Springer, due to_ , 2017. 

- [5] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. _arXiv preprint arXiv:1907.02893_ , 2019. 

- [6] Pietro Barbiero, Gabriele Ciravegna, Francesco Giannini, Pietro Lió, Marco Gori, and Stefano Melacci. Entropy-based logic explanations of neural networks. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pages 6046–6054, 2022. 

- [7] Sara Beery, Grant Van Horn, and Pietro Perona. Recognition in terra incognita. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 456–473, 2018. 

- [8] Gilles Blanchard, Aniket Anand Deshmukh, Ürun Dogan, Gyemin Lee, and Clayton Scott. Domain generalization by marginal transfer learning. _The Journal of Machine Learning Research_ , 22(1):46–100, 2021. 

- [9] Gilles Blanchard, Gyemin Lee, and Clayton Scott. Generalizing from several related classification tasks to a new unlabeled sample. _Advances in neural information processing systems_ , 24, 2011. 

- [10] Enric Boix-Adsera, Omid Saremi, Emmanuel Abbe, Samy Bengio, Etai Littwin, and Joshua Susskind. When can transformers reason with abstract symbols? _arXiv preprint arXiv:2310.09753_ , 2023. 

- [11] Malik Boudiaf, Imtiaz Ziko, Jérôme Rony, José Dolz, Pablo Piantanida, and Ismail Ben Ayed. Information maximization for few-shot learning. _Advances in Neural Information Processing Systems_ , 33:2445–2457, 2020. 

- [12] Manh-Ha Bui, Toan Tran, Anh Tran, and Dinh Phung. Exploiting domain-specific features to enhance domain generalization. _Advances in Neural Information Processing Systems_ , 34:21189–21201, 2021. 

- [13] Kaidi Cao, Maria Brbic, and Jure Leskovec. Open-world semi-supervised learning. In _International Conference on Learning Representations_ , 2022. 

- [14] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 9650–9660, 2021. 

- [15] Zhuomin Chai, Yuxiang Zhao, Wei Liu, Yibo Lin, Runsheng Wang, and Ru Huang. Circuitnet: An open-source dataset for machine learning in vlsi cad applications with improved domain-specific evaluation metric and learning strategies. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , 42(12):5034–5047, 2023. 

- [16] Florent Chiaroni, Jose Dolz, Ziko Imtiaz Masud, Amar Mitiche, and Ismail Ben Ayed. Parametric information maximization for generalized category discovery. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 1729–1739, 2023. 

- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_ , pages 248–255. Ieee, 2009. 

- [18] Chen Fang, Ye Xu, and Daniel N Rockmore. Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 1657–1664, 2013. 

- [19] Enrico Fini, Enver Sangineto, Stéphane Lathuilière, Zhun Zhong, Moin Nabi, and Elisa Ricci. A unified objective for novel class discovery. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9284–9292, 2021. 

11 

- [20] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. _The journal of machine learning research_ , 17(1):2096–2030, 2016. 

- [21] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. _arXiv preprint arXiv:2007.01434_ , 2020. 

- [22] Kai Han, Sylvestre-Alvise Rebuffi, Sebastien Ehrhardt, Andrea Vedaldi, and Andrew Zisserman. Autonovel: Automatically discovering and learning novel visual categories. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2021. 

- [23] Shoubo Hu, Kun Zhang, Zhitang Chen, and Laiwan Chan. Domain generalization via multidomain discriminant analysis. In _Uncertainty in Artificial Intelligence_ , pages 292–302. PMLR, 2020. 

- [24] Zeyi Huang, Haohan Wang, Eric P Xing, and Dong Huang. Self-challenging improves cross-domain generalization. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16_ , pages 124–140. Springer, 2020. 

- [25] Cha Junbum, Lee Kyungjae, Park Sungrae, and Chun Sanghyuk. Domain generalization by mutualinformation regularization with pre-trained models. _European Conference on Computer Vision (ECCV)_ , 2022. 

- [26] Daehee Kim, Youngjun Yoo, Seunghyun Park, Jinkyu Kim, and Jaekoo Lee. Selfreg: Self-supervised contrastive regularization for domain generalization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9619–9628, 2021. 

- [27] Andreas Krause, Pietro Perona, and Ryan Gomes. Discriminative clustering by regularized information maximization. _Advances in neural information processing systems_ , 23, 2010. 

- [28] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In _Proceedings of the IEEE international conference on computer vision workshops_ , pages 554–561, 2013. 

- [29] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 

- [30] David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (rex). In _International Conference on Machine Learning_ , pages 5815–5826. PMLR, 2021. 

- [31] Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy Hospedales. Learning to generalize: Meta-learning for domain generalization. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

- [32] Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Deeper, broader and artier domain generalization. In _Proceedings of the IEEE international conference on computer vision_ , pages 5542–5550, 2017. 

- [33] Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. Domain generalization with adversarial feature learning. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5400–5409, 2018. 

- [34] Ya Li, Mingming Gong, Xinmei Tian, Tongliang Liu, and Dacheng Tao. Domain generalization via conditional invariant representations. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

- [35] Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and Dacheng Tao. Deep domain generalization via conditional invariant adversarial networks. In _Proceedings of the European conference on computer vision (ECCV)_ , pages 624–639, 2018. 

- [36] Zenan Li, Yunpeng Huang, Zhaoyu Li, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, and Jian Lu. Neuro-symbolic learning yielding logical constraints. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- [37] Ziyue Li, Kan Ren, Xinyang Jiang, Yifei Shen, Haipeng Zhang, and Dongsheng Li. Simple: Specialized model-sample matching for domain generalization. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- [38] J MacQueen. Classification and analysis of multivariate observations. In _Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability_ , pages 281–297, 1967. 

12 

- [39] Ofir Moshe, Gil Fidel, Ron Bitton, and Asaf Shabtai. Improving interpretability via regularization of neural activation sensitivity. _arXiv preprint arXiv:2211.08686_ , 2022. 

- [40] Hyeonseob Nam, HyunJae Lee, Jongchan Park, Wonjun Yoon, and Donggeun Yoo. Reducing domain gap by reducing style bias. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8690–8699, 2021. 

- [41] Konstantinos Panagiotis Panousis, Dino Ienco, and Diego Marcos. Sparse linear concept discovery models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 2767–2771, 2023. 

- [42] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 1406–1415, 2019. 

- [43] Wilhelm Rödder. Conditional logic and the principle of entropy. _Artificial Intelligence_ , 117(1):83–106, 2000. 

- [44] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. _arXiv preprint arXiv:1911.08731_ , 2019. 

- [45] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In _Proceedings of the IEEE international conference on computer vision_ , pages 618–626, 2017. 

- [46] Yuge Shi, Jeffrey Seely, Philip HS Torr, N Siddharth, Awni Hannun, Nicolas Usunier, and Gabriel Synnaeve. Gradient matching for domain generalization. _arXiv preprint arXiv:2104.09937_ , 2021. 

- [47] Mannat Singh, Laura Gustafson, Aaron Adcock, Vinicius de Freitas Reis, Bugra Gedik, Raj Prateek Kosaraju, Dhruv Mahajan, Ross Girshick, Piotr Dollár, and Laurens Van Der Maaten. Revisiting weakly supervised pre-training of visual perception models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 804–814, 2022. 

- [48] Baochen Sun and Kate Saenko. Deep coral: Correlation alignment for deep domain adaptation. In _Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14_ , pages 443–450. Springer, 2016. 

- [49] Kiat Chuan Tan, Yulong Liu, Barbara Ambrose, Melissa Tulig, and Serge Belongie. The herbarium challenge 2019 dataset. _arXiv preprint arXiv:1906.05372_ , 2019. 

- [50] Zhaorui Tan, Xi Yang, and Kaizhu Huang. Rethinking multi-domain generalization with a general learning objective. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 23512–23522, June 2024. 

- [51] Zhaorui Tan, Xi Yang, and Kaizhu Huang. Semantic-aware data augmentation for text-to-image synthesis. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 5098–5107, 2024. 

- [52] Irene Tsapara and György Turán. Learning atomic formulas with prescribed properties. In _Proceedings of the eleventh annual conference on Computational learning theory_ , pages 166–174, 1998. 

- [53] Vladimir N. Vapnik. _Statistical Learning Theory_ . Wiley-Interscience, 1998. 

- [54] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Generalized category discovery. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7492–7501, 2022. 

- [55] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 5018–5027, 2017. 

- [56] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 

- [57] Chunyang Wu, Mark JF Gales, Anton Ragni, Penny Karanasou, and Khe Chai Sim. Improving interpretability and regularization in deep learning. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 26(2):256–265, 2017. 

- [58] Mike Wu, Michael Hughes, Sonali Parbhoo, Maurizio Zazzi, Volker Roth, and Finale Doshi-Velez. Beyond sparsity: Tree regularization of deep models for interpretability. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 32, 2018. 

13 

- [59] Mike Wu, Sonali Parbhoo, Michael Hughes, Ryan Kindle, Leo Celi, Maurizio Zazzi, Volker Roth, and Finale Doshi-Velez. Regional tree regularization for interpretability in deep neural networks. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 34, pages 6413–6421, 2020. 

- [60] Changnan Xiao and Bing Liu. A theory for length generalization in learning to reason. _arXiv preprint arXiv:2404.00560_ , 2024. 

- [61] Marvin Zhang, Henrik Marklund, Nikita Dhawan, Abhishek Gupta, Sergey Levine, and Chelsea Finn. Adaptive risk minimization: Learning to adapt to domain shift. _Advances in Neural Information Processing Systems_ , 34:23664–23678, 2021. 

- [62] Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Domain generalization with mixstyle. _arXiv preprint arXiv:2104.02008_ , 2021. 

- [63] Jialv Zou, Xinggang Wang, Jiahao Guo, Wenyu Liu, Qian Zhang, and Chang Huang. Circuit as set of points. _Advances in Neural Information Processing Systems_ , 36, 2024. 

14 

## **A Broader impact** 

Our regularization term based on logic for image classification offers significant potential beyond academia. By integrating logical constraints, our approach enhances model robustness, interpretability, and ethical alignment. This translates into improved performance on real-world tasks such as disease diagnosis in healthcare and mitigating biases in decision-making systems. Our work fosters interdisciplinary collaboration and contributes to the responsible deployment of AI technologies, ultimately benefiting society through enhanced efficiency, fairness, and transparency in machine learning applications. 

## **B Details of the logical framework for visual classification task** 

We provide more details of the connections between logical reasoning and visual classification tasks. **Definition B.1.** Following [4], a logic _L_ is a five-tuple defined in the form: 

_L_ = _⟨FL, ML, |_ = _L, mngL, ⊢L⟩ ,_ (16) 

where 

- _FL_ is a set of all formulas of _L_ . _FL_ arbitrarily refers to any collections that can be ‘expressed’ by language _L_ . Therefore, _FL_ could be not only a collection of languages but also images and labels ( _X, Y_ ) for computer vision cases. 

- _ML_ is a class called the class of all models (or possible worlds) of _L_ ; intuitively, this can be considered as different domains _D_ of _X_ . 

- _|_ = _L_ is a binary relation, _|_ = _L⊆ ML × FL_ , called the validity relation of _L_ . For example, in the known set, the ground truth label of the image is given as truth, which is the validity relation. 

- _mngL_ : _FL × ML −→_ Sets where Sets is the class of all sets. _mngL_ is a function with domain _FL × ML_ , called the meaning function of _L_ : Intuitively, _mngL_ extracts the meaning of the expressions can be understood as the classifiers. 

- _⊢L_ represents the provability relation of _L_ , telling us which formulas are ‘true’ in which possible world and usually is definable from _mngL_ , such as the estimation criteria in the machine learning system. 

Accordingly and still following [4], a good general logic is defined as: 

**Definition B.2** (General logic) **.** : A general logic is a class: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0015-13.png)


where _Sig_ is a class of sets; _L[P]_ = � _FL[P][, M] L[ P][,][ |]_[=] _[P] L[, mng] L[P][,][ ⊢][P] L[,]_ � is a compositional logic in the sense of Definition B.1 for _P ∈ Sig_ , and for any sets _P, Q ∈ Sig_ satisfies the following conditions: 

1. _P_ is the set of atomic formulas of _L[P]_ . 

2. _Cn_ ( _L[P]_ ) = _Cn_ ( _L[Q]_ ) := _Cn_ ( _L[∗]_ ) where _Cn_ ( _·_ ) is called the set of logical connectives of the given logic (these are operation symbols with finite or infinite ranks). 

3. Any bijection _f_ : _P_ ß _Q_ that extends to a bijection between the tautological formula algebras of _L[P]_ and _L[Q]_ induces an isomorphism between _L[P]_ and _L[Q]_ . 

4. If _P ⊆ Q_ , then _L[P]_ is a sublogic of _L[Q]_ . 

5. For any _P ∈ Sig_ and set _H_ , there is a _P[′] ∈ Sig_ such that _P[′]_ is disjoint from _H_ and _L[P][ ′]_ is an isomorphic copy of _L[P]_ . 

6. The union of a system _Pi, i ∈ I_ of pairwise disjoint sets _Pi_ from _Sig_ belongs to _Sig_ , whenever _I_ is not empty. Let Fr( _·_ ) denotes free algebra, Alg _m_ ( _L_ ) represents _{mng_ M(F) : M _∈ M }_ where F denotes the term algebra. Further, the tautological congruence of the logic belonging to the disjoint union _P_ is generated in Fr �Alg _m_ � _L[P]_[ �] _, P_ � as a 

15 

congruence by the union of the tautological congruence relations of the logics belonging to _Pi, i ∈ I_ . 

7. _Sig_ contains at least one non-empty set. 

Our L-Reg aims to regularize the semantics extracted by _g_ and the classifier to satisfy condition 1. 

_⊢_ ( _h◦g_ ( _X_ ) _,Y_ )= _|_ =( _g_ ( _Xs_ ) _,Ys_ ) **in Eq. (6) can be safely omitted in the rest of the paper.** Consider the logic formed on _X, Y_ : _L_ ( _Xs,Ys_ ) = � _F_ ( _Xs,Ys_ ) _, D, |_ =( _Xs,Ys_ ) _, h, ⊢_ ( _h_ ( _X_ ) _,Y_ )�. Assume we want def to study the logic of _⊢_ which can be defined in the form of _L⊢_ := _⟨FXs,Ys , D⊢, h⊢, |_ = _⊢⟩_ , where _D⊢, h⊢, |_ = _⊢_ are pseudo-components associated with _⊢_ . Particularly, _D⊢_ is a subset of all possible def world/domains from _F_ ( _Xs,Ys_ ): _D⊢_ = _{T ⊆ F_ ( _Xs,Ys_ ) : _T_ is closed under _⊢_ ( _h_ ( _X_ ) _,Y_ ) _}_ . For any def _T ∈ D⊢_ and _a ∈ F_ ( _Xs,Ys_ ), it has _h⊢_ ( _a, T_ ) = _{b ∈ F_ : _T ⊢_ ( _a ↔ b_ ) _}_ . Further, _|_ = _⊢_ in _T ∈ D⊢_ is def defined as _T |_ = _⊢ a ⇔ a ∈ T_ . [4] points out that the following condition is almost always satisfied: (Cond) _∀a, b ∈ F⊢, d ∈ D⊢_ , we have ( _h⊢_ ( _a, d_ ) = _h⊢_ ( _b, d_ )) and _d |_ = _⊢ a ⇒ d |_ = _⊢ b_ . Therefore, the semantical consequence relation induced by _|_ = _⊢_ coincides with the original syntactical _⊢_ ( _h◦g_ ( _X_ ) _,Y_ ) while Cond holds. Due to that _D⊢ ⊆ D_ , it infers that _|_ =( _g_ ( _Xs_ ) _,Ys_ ) coincides with _|_ = _⊢_ . Therefore, _⊢_ ( _h◦g_ ( _X_ ) _,Y_ )= _|_ =( _g_ ( _Xs_ ) _,Ys_ ) can be safely omitted in the rest of the paper. 

## **C Details of proofs** 

**Proposition C.1** (L-Reg reduces the complexity of the model, promoting data-shift generalization performance.) **.** _Assume the domain gap is well minimized. Consider a f[∗] is the target model that generalizes to the unseen with the lowest complexity. There are f_ ( _[R] Xs,Ys_ ) _[, f]_[(] _[X] s[,Y] s_[)] _[trained under the] setting of data-shift generalization (i.e.,_ ( _Xs, Ys_ ) _is accessible and Ys_ = _Yu), it has that:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0016-06.png)


_Proof._ We assume the loss is achieved for the tractable form by minimizing the mean squared error. In that case, we have _f_ ( _[∗] Xs,Ys_ )[for the given training set as:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0016-08.png)


In comparison to _f[∗]_ , _f_ ( _Xs,Ys_ ) for the given seen sets is as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0016-10.png)


and _f_ ( _[R] Xs,Ys_ )[is][derived][from] _[f]_[(] _[X] s[,Y] s_[)][,][where] _[Z][s]_[is][constrained][additionally][by][L-Reg][and][the] constrained _Zs_ is denoted as _Zs[R]_[:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0016-12.png)


For simplification, we denote ( _Z[T] Z_ ) _[−]_[1] _Z[T]_ , ( _Zs[T][Z][s]_[)] _[−]_[1] _[Z] s[T]_[, and][ (] _[Z] s[R][T] Zs_ ) _[−]_[1] _Zs[R][T]_ as _N[∗]_ , _N_ , and _N[R]_ , respectively. 

**The form of** _N_ **.** For multi-domain generalization, the model is tested on the unseen domain, referring that _Xu_ contains some unseen semantics besides the seen: _Zs ∼Zs, Zu ∼Zu, Zs_ = _Zu, Zs ∩Zu_ = _∅_ . Considering each dimension of _Z_ represents a specific semantics, we denote Γ as the dimensions of _Z_ that contain the seen semantics support in _Xs_ and Γ[¯] for the unseen, we can decompose _N_ as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0016-15.png)


**The form of** _N[∗]_ **.** Assume Γ already contains semantic support for deducting _Y_ ; thus, Γ[¯] would not affect the deduction of _Y_ . In such case, it has that Γ _[T]_[ ¯] Γ = **0** and Γ[¯] _[T]_ Γ = **0** and Γ[¯] _[T]_[ ¯] Γ = **1** where **0** _,_ **1** 

16 

denote zero matrix and identity matrix: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0017-01.png)


where we also expect _h_ (Γ) =[¯] **0** so that _zu_ does not influence the deduction. We now have _N[∗]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0017-03.png)


Note that for _N_ in _f_ ( _Xs,Ys_ ), Γ _[T]_[ ¯] Γ and Γ[¯] _[T]_ Γ are not constrained. Please refer to Lemma C.2. Furthermore, _h_ (Γ)[¯] is also not constrained. 

**The form of** _N[R]_ **.** Now we discuss the trainable _N[R]_ obtained with the application of L-Reg. The form of _N[R]_ is similar to _N[∗]_ . However, Eq. (11) indicates that L-Reg minimizes _||_ Γ _[T]_[ ¯] Γ _||_ 2 and _||_ Γ[¯] _[T]_ Γ _||_ 2 through _−H_ ( _Y |g_ (Γ))[¯] _, D_ ) and also minimizing _||h_ (Γ)[¯] _||_ 2: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0017-06.png)


**Compare** _GL_ ( _f_ ( _[R] Xs,Ys_ ) _[, f][ ∗][, X][u]_[)] **[ with]** _[ GL]_[(] _[f]_[(] _[X] s[,Y] s_[)] _[, f][ ∗][, X][u]_[)] **[.]**[By comparing the forms of] _[ N][ R][,][ N]_ and _N[∗]_ , it is obvious that _||N[R] −N[∗] ||_ 2 _≤ ||N −N[∗] ||_ 2. Therefore, we have that: _GL_ ( _f_ ( _[R] Xs,Ys_ ) _[, f][ ∗][, X][u]_[)] _[ ≤][GL]_[(] _[f]_[(] _[X] s[,Y] s_[)] _[, f][ ∗][, X][u]_[)][.] 

**Lemma C.2** (Minimizing _H_ ( _Y |g_ ( _X_ ) _, D_ ) + _R_ solely may cause generalization degradation) **.** _Minimizing H_ ( _Y |g_ ( _X_ ) _, D_ ) + _R solely without L-Reg may conflict with_ max _h,g H_ ( _Y |g_ (Γ)[¯] _, D_ ) _, causing invalid semantics for decision process and degrading the generalization._ 

_Proof._ We have the following relationship for _H_ ( _Y |g_ ( _z_ ) _, D_ ): 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0017-10.png)


Since the independence between _{zi}[M] i_ =1[is unconstrained,] _[ H]_[(] _[Y, g]_[(Γ)] _[|][g]_[(¯Γ)] _[, D]_[)][ may cause that] _[ Y]_ can be deducted from Γ[¯] . Therefore, Γ _[T]_[ ¯] Γ and Γ[¯] _[T]_ Γ are not constrained even when the domain gap is minimized where _|D|_ = 1, causing the sub-optimal generalization. 

## **D One toy example** 

We present a simplified informal illustrative example to compare the efficacy of our proposed L-Reg against conventional L1 and L2 regularization methods. As depicted in Fig. 6, the ground truth (GT) image represents the underlying data, generated according to _f[∗]_ ( _x_ 1 _, x_ 2) = sin(2 _πx_ 1) _·_ sin(2 _πx_ 2), where _x_ 1 and _x_ 2 denote the horizontal and vertical coordinates respectively, and the pixel color corresponds to the value of _f[∗]_ ( _x_ 1 _, x_ 2). The training domain is delineated by the black box, while the testing domain encompasses the area outside of this boundary. 

For our experiments, we use a 6-linear-layer size-110 ReLU model network. Mean squared error serves as the loss function. 

Our experimental results reveal that L-Reg enhances the model’s ability to extrapolate beyond the training domain. Notably, our proposed L-Reg demonstrates superior extrapolative capabilities compared to traditional _L_ 1 and _L_ 2 regularization methods. This observation highlights the efficacy of L-Reg in fostering improved generalization. 

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0018-00.png)


**----- Start of picture text -----**<br>
GT Base model  + �1 Regularization  + �2 Regularization  + L-Reg<br>**----- End of picture text -----**<br>


Figure 6: Prediction visualizations of MLP with different regularization terms. 

## **E Apply L-Reg to ERM Baseline for mDG** 

To further validate L-Reg’s efficacy for mDG, we use ERM as the baseline on the TerraIncognita dataset. For a fair comparison, all experiments share the same hyperparameter settings and use the Regnety-16gf backbone. Original ERM results are also included alongside our reproduced results. The results in Table 8 reveal that ERM with L-Reg significantly improves mDG performance (from 49.9% to 52.9%). 

## **F Compare L-Reg with more regularization terms** 

We also compare L-Reg with other regularization terms: The Ortho-Reg - the orthogonality regularization that constrains the independence of each dimension of the semantic feature _z_ ; and Sparsity - implemented as Bernoulli Sample of the latent features from the sparse linear concept discovery models [41] on our used PIM backbone. To investigate this fairly, we re-implemented the Bernoulli Sample of the latent features from the Sparse Linear Concept Discovery Models [41] on the same PIM backbone that we used, to achieve the sparsity. Table 6&Table 7 demonstrate that L-Reg outperforms Ortho-Reg and Sparsity. 

Especially, while a common sparse concept model may be able to achieve _γ[y] ψ_ = _z[y]_ by filtering irrelevant features through the sparsity, it may not ensure _γ[y][i]_ = _γ[y][j]_ , which is crucial for disentangling features used for predicting different classes. This limitation can potentially lead to degradation in generalization performance for common sparse concept models. 6&7 indicate that while L-Reg consistently achieves overall improvement, the sparse concept-based approach does not consistently improve generalization, validating the aforementioned difference. 

## **G Limitation of L-Reg and possible solutions** 

As analyzed and discussed in the paper, L-Reg is based on the precondition that each dimension of the latent features represents an independent semantic. 

We hypothesize this is due to the fact that our L-Reg is derived based on the precondition that _z[i] , z[j] ∈ z, I_ = _j_ is independent of each other. This condition holds for most deep-layer features but may not apply to shallow layers. Thus, applying L-Reg to the semantic features from the deep layers may improve the performance for unknown classes without negatively impacting known classes. 

Derived from this hypothesis, another possible solution is further regularizing the independence, which may lead to further improvements. To validate this hypothesis, we test L-Reg by reinforcing independence with Ortho-Reg. MDG results in Table 8 and GCD results in Table 6&Table 7 show that combining L-Reg with Ortho-Reg leads to further improvements, whereas Ortho-Reg alone may not guarantee improvements. These findings support our hypothesis and suggest that L-Reg, particularly 

18 

Table 6: **Results of GCD:** Averaged results across all datasets of PIM with different regularization applied to the latent features: Sparsity: achieved through Bernoulli Sample; Ortho-Reg: orthogonality regularization. +L-Reg outperforms other regularization terms when they are applied solely; +LReg+Ortho-Reg achieves the best performance and alleviates the performance degradation of unknown classes, validating our hypothesis in the paper that the improper _Z_ may result in compromises and constraining the independence of each _z[i] ∈ z, z ∈ Z_ may be helpful. 

|dence of each_zi ∈z, _|_z ∈Z_ maybe helpful.|
|---|---|
||Avg<br>All<br>Known<br>Unknown|
|PIM|67.4<br>79.3<br>59.9|
|+Sparsity<br>Improvements|66.6<br>77.3<br>60.0<br>-0.7<br>-2.0<br>0.1|
|+Ortho-Reg<br>Improvements|68.4<br>79.2<br>61.9<br>1.0<br>-0.1<br>2.0|
|**+L-Reg**|68.8<br>79.0<br>62.7|
|Improvements|1.4<br>-0.3<br>2.8|
|**+L-Reg+Ortho-Reg**|**69.3**<br>**79.6**<br>**63.4**|
|Improvements|**2.0**<br>**0.3**<br>**3.5**|



Table 7: **Results of GCD:** Detailed results across all datasets of PIM with different regularization applied to the latent features: Sparsity: achieved through Bernoulli Sample; Ortho-Reg: orthogonality regularization. 

||**CUB**<br>All<br>Known<br>Unknown|**Stanford Cars**<br>All<br>Known<br>Unknown|**Herbarium19**<br>All<br>Known<br>Unknown|
|---|---|---|---|
|PIM|62.7<br>75.7<br>56.2|43.1<br>66.9<br>31.6|42.3<br>56.1<br>34.8|
|PIM + Sparsity<br>Improvements|60.1<br>72.7<br>53.8<br>-2.6<br>-3.0<br>-2.4|40.4<br>61.7<br>30.1<br>-2.7<br>-5.2<br>-1.5|42.0<br>53.7<br>35.8<br>-0.3<br>-2.4<br>1.0|
|PIM + Ortho-Reg<br>Improvements|64.9<br>76.7<br>58.9<br>2.2<br>1.0<br>2.7|44.3<br>65.6<br>34.1<br>1.2<br>-1.3<br>2.5|42.9<br>57.2<br>35.1<br>0.6<br>1.1<br>0.3|
|**PIM + L-Reg**|65.3<br>76.0<br>60.0|44.8<br>66.0<br>34.6|**43.7**<br>55.8<br>**37.2**|
|Improvements|2.6<br>0.3<br>3.8|1.7<br>-0.9<br>3.0|1.4<br>-0.3<br>2.4|
|**PIM + L-Reg + Ortho-Reg**|**66.8**<br>**77.3**<br>**61.6**|**45.8**<br>**67.3**<br>**35.5**|43.3<br>57.5<br>35.6|
|Improvements|4.1<br>1.6<br>5.4|2.7<br>0.4<br>3.9|1.0<br>1.4<br>0.8|
||**CIFAR10**<br>All<br>Known<br>Unknown|**CIFAR100**<br>All<br>Known<br>Unknown|**ImageNet-100**<br>All<br>Known<br>Unknown|
|PIM|94.7<br>97.4<br>93.3|78.3<br>84.2<br>66.5|83.1<br>**95.3**<br>77.0|
|PIM + Sparsity<br>Improvements|94.2<br>97.4<br>92.6<br>-0.5<br>0.0<br>-0.7|79.7<br>**84.6**<br>69.7<br>1.4<br>0.4<br>3.2|83.4<br>93.7<br>78.2<br>0.3<br>-1.6<br>1.2|
|PIM + Ortho-Reg<br>Improvements|95.1<br>97.4<br>93.9<br>0.4<br>0.0<br>0.6|80.2<br>**84.6**<br>71.4<br>1.9<br>0.4<br>4.9|83.0<br>93.4<br>77.7<br>-0.1<br>-1.9<br>0.7|
|**PIM + L-Reg**|94.8<br>**97.6**<br>93.4|80.8<br>**84.6**<br>73.2|83.4<br>94.0<br>78.0|
|Improvements|0.1<br>0.2<br>0.1|2.5<br>0.4<br>6.7|0.3<br>-1.3<br>1.0|
|**PIM + L-Reg + Ortho-Reg**|**95.1**<br>**97.6**<br>**93.9**|**81.2**<br>84.2<br>**75.0**|**83.7**<br>93.6<br>**78.7**|
|Improvements|0.4<br>0.2<br>0.6|2.9<br>0.0<br>8.5|0.6<br>-1.7<br>1.7|



when applied to deep layers or in conjunction with Ortho-Reg, is beneficial. This suggests a direction for future work. 

## **H More experimental details and results** 

All experiments can be conducted on one NVIDIA GeForce RTX 3090 GPU. 

## **H.1 Multi-domain generalization** 

**Competitors.** We listed results from previous important work in the mDG field for better validation. They are: MMD [33], Mixstyle [62], GroupDRO [44], IRM [5], ARM [61], VREx [30], CDANN [35], DANN [20], RSC [24], MTL [8], MLDG [31], Fish [46], ERM [53], SagNet [40], SelfReg [26], CORAL [48], mDSDI [12], MIRO [25], and GMDG [50]. Among them, GMDG is treated as our baseline since it sufficiently minimizes the domain gaps. 

**Datasets.** We use PACS (4 domains, 9,991 samples, 7 classes) [32], VLCS (4 domains, 10 _,_ 729 samples, 5 classes) [18], OfficeHome (4 domains, 15,588 samples, 65 classes) [55], TerraIncognita 

19 

Table 8: **Results of mDG:** Results of using ERM as the baseline. We use the ERM method as the baseline to test L-Reg’s efficacy. Ortho-Reg: orthogonality regularization. This table includes results: (1) The improved performance of L-Reg on ERM baseline. (2) Comparison between L-Reg and the Ortho-Reg on ERM baseline. (3) Using L-Reg and Ortho-Reg together yields further promotion, validating our ‘improper _z_ ’ hypothesis in the Paper limitation part. The used dataset is TerraIncognita. All experiments share the same hyperparameters except the added regularization term. Each group of experiments is run with seeds [0,1,2], and the averaged results for each domain and additionally with the standard deviation (Std) are reported. 

|**TerraIncognita**|**Location 100**<br>**Location 38**<br>**Location 43**<br>**Location 46**<br>**Avg ± Std.**|
|---|---|
|ERM<br>ERM Reproduced<br>+Ortho-Reg|54.3<br>42.5<br>55.6<br>38.8<br>47.8<br>50.6<br>49.7<br>58<br>41.2<br>49.9±3.6<br>50.7<br>52.6<br>60.5<br>42.7<br>51.6±2.5|
|**+L-Reg**|52.7<br>51.7<br>61.3<br>45.8<br>52.9±4.2|
|**+L-Reg+Ortho-Reg**|61.5<br>48.6<br>60.3<br>44<br>53.6±0.5|



Table 9: Parameters for mDG task 

|Table 9: Parameters|for mDG|task|
|---|---|---|
|Use RegNetY-16GF|lr mult|_α_|
|TerraIncognita|2.5|1e-3|
|OffceHome|0.1|1e-3|
|VLCS|0.1|1e-4|
|PACS|0.1|5e-4|
|DomainNet|5.0|1e-3|



(TerraIncognita, 4 domains, 24 _,_ 778 samples, 10 classes) [7], and DomainNet (6 domains, 586,575 samples, 345 classes) [42]. 

**Training details.** We use GMDG [50] as our baseline. Especially, we use all loss terms proposed in GMDG as _Lmain_ . The training procedure is the same as MIRO [25] and GMDG. We use seeds 0 _,_ 1 _,_ 2 for all three trails training. 

**Parameters.** We adhere to the parameters proposed by GMDG, particularly focusing on its recommended loss terms. Furthermore, we provide a detailed listing of the hyper-parameters pertaining to L-Reg, along with the tuned ‘lr mult’, as outlined in Table 9, to facilitate the reproducibility of our results. 

**Evaluation metric.** The models undergo training on known domains and subsequent testing on unseen domains. For each trial, a distinct domain within the datasets is designated as the unseen domain. The evaluation metric reports the prediction accuracy achieved on these unseen domains. The aggregated results across all unseen domains within the datasets provide a comprehensive assessment of the algorithm’s performance in domain generalization for the given datasets. 

**More results.** Results of each domain for each dataset are presented in Tables 10 to 14. 

## **H.2 Generalized category discovery** 

**Competitors.** We compare our proposed method with existing generalized category discovery methods: GCD [54], and PIM [16]. In particular, PIM based on information maximization is the current state-of-the-art (SOTA) generalized category discovery method. Additionally, the traditional machine learning method, k-means [38]; three novel category discovery methods: RankStats+ [22], UNO+ [19], ORCA [13]; and several information maximization methods: RIM [27], and TIM [11] are adapted for generalized category discovery as competitors. The results of the modified novel category discovery methods are reported in [54], and the modified information maximization methods are reported in [16]. 

**Usage details of datasets for GCD.** Following the protocols of GCD and PIM [54, 16], the initial training set of each dataset is divided into labeled and unlabeled subsets; samples from half of the classes are assigned as unlabeled, and their labels are not used for training. Specifically, half of the image samples from known classes are allocated to the labeled subset, while the remaining half are assigned to the unlabeled subset. Additionally, the unlabeled subset includes all image samples from 

20 

the novel classes in the original dataset. As a result, the unlabeled subset consists of instances from _K_ different classes. The detailed statistics of datasets are listed in Table 15. 

**Training details.** Consistent with PIM, we utilize latent features extracted by the feature encoder DINO (VIT-B/16) [14] that is pre-trained on ImageNet [17] through self-supervised learning. The losses proposed in PIM are treated as _Lmain_ . The original PIM freezes the feature extractor during the training, directly using the pre-saved extracted features as the model input. For a fair comparison, we only added one linear layer as _g_ on the extracted features, which is the minimal modification. 

_L_ 2 **(weight decay) value searching.** For a more fair comparison, we conduct weight decay value searching to ensure that the weight of _L_ 2 is the best. To address this, we devised a methodology for weight decay searching involving the construction of smaller labeled and unlabeled subsets derived solely from the labeled data. To conduct parameter searching, we split the labeled samples to construct a ’smaller’ sub-labeled and sub-unlabeled set. Specifically, we take 50% of the samples from known classes as sub-unlabeled samples from unknown classes. Additionally, we take 25% of the samples from the remaining 50% of known classes as sub-unlabeled samples from known classes. The remaining samples are treated as sub-labeled samples. Hyper-parameters are then searched on these sub-labeled and sub-unlabeled sets. 

**Parameters of L-Reg.** The hyper-parameters of L-Reg values are shown in Table 16. 

**Evaluation metric.** Following prior works [54, 16], we use the proposed accuracy metric from [54] of all classes, known classes, and unknown classes for evaluation. 

**More results.** The results for each dataset are presented in Table 17. It is evident that L-Reg yields enhanced performance across half of the datasets for both known and unknown classes. On the remaining datasets, while L-Reg may slightly compromise the performance of known classes, it demonstrates significant improvements in the unknown classes, resulting in an overall enhancement in the performance across all classes. 

**More ablation results.** Due to the introduction of tuned weight decay and the additional _g_ component, we have conducted ablation studies to assess their impact. The results are summarized in Table 18. It is observed that the baseline model utilizing the tuned weight decays performs slightly better than the original weight decay settings. Notably, the tuned weight decays contribute to improvements in unknown classes while often leading to slight decreases in known classes across most datasets. Inclusion of the proposed extra component _g_ results in marginal improvements in both known and unknown classes compared to the tuned baseline. Our proposed L-Reg demonstrates significant improvements specifically in the unknown classes, thereby corroborating Proposition 4.2. However, as discussed in the main paper, it is acknowledged that L-Reg may entail compromises in the performance of known classes. 

## **H.3 Combination of multi-domain generalization and generalized category discovery** 

**Datasets.** We leverage the datasets utilized in mDG tasks to construct the mDG+GCD datasets. Specifically, during the seen domains of training, labels from approximately half of the classes are masked. For instance, in the PCAS dataset comprising 7 classes, classes labeled within the range [0 _,_ 1 _,_ 2 _,_ 3] are retained, while classes in [4 _,_ 5 _,_ 6] are masked. It is noteworthy that data categorized as unknown classes in our setup are from unknown classes. However, we acknowledge that this prior is not explicitly known. **To align with the GCD setting, we operate under the assumption that the unlabeled set may potentially include samples from known classes.** Consequently, we refrain from constraining the model by mandating that unlabeled data be classified solely as unknown classes. This adjustment introduces a more challenging generalization scenario. 

**Training details.** For all experiments, the implementation directly adds L-Reg to their previously proposed loss sets. The models are trained with the aforementioned labeled and unlabeled sets from the seen domains and tested on the samples from the unseen domain. 

**Parameters.** We include all the parameters for reproducing our experiments in the code. Please refer to the code for details. 

**Evaluation metric.** We use the same metric from the GCD task for the mDG+GCD task. Similarly, the metrics include the accuracy for all, known and unknown classes. 

21 

**More results.** The averaged results of each dataset are exhibited in Table 19, while the detailed results of each dataset are presented in Tables 20 to 24. 

## **H.4 More GradCAM visualizations** 

We provide more visualized examples of L-Reg. Examples of known classes can be seen in Figs. 7 to 10 and unknown classes in Figs. 11 and 12. Compromises in known sets, as discussed in the limitations, can be seen in Figs. 8 and 12. 

22 

Table 10: MDG experiments on TerraIncognita: More results of full GMDG+L-Reg for each category. 

|**TerraIncognita**|**Location 100**<br>**Location 38**<br>**Location 43**<br>**Location 46**|**Avg.**|
|---|---|---|
|ERM [21]<br>MIRO [25] (use ResNet-50)<br>GMDG [50] (use ResNet-50)|54.3<br>42.5<br>55.6<br>38.8<br>-<br>-<br>-<br>-<br>59.8±1.0<br>45.3±1.7<br>57.1±1.8<br>38.2±5|47.8<br>50.4<br>50.1±1.2|
|MIRO [25] (use RegNetY-16GF)<br>GMDG [50] (use RegNetY-16GF)|-<br>-<br>-<br>-<br>73.3±3.3<br>54.7±1.4<br>67.1±0.3<br>48.6±6.5|58.9±1.3<br>60.7±1.8|
|GMDG +**L-Reg**(use RegNetY-16GF)|**73.9**±0.8<br>**57.1**±2.3<br>**67.9**±1.1<br>**52.7**±4.0|**62.9**±0.9|



Table 11: MDG experiments on OfficeHome: More results of full GMDG+L-Reg for each category. 

|**OffceHome**|**art**<br>**clipart**<br>**product**<br>**real**<br>**Avg.**|**art**<br>**clipart**<br>**product**<br>**real**<br>**Avg.**|
|---|---|---|
||||
|ERM [21]<br>MIRO [25] (use ResNet-50)<br>GMDG [50] (use ResNet-50)|63.1<br>51.9<br>77.2<br>78.1<br>-<br>-<br>-<br>-<br>68.9±0.3<br>56.2±1.7<br>79.9±0.6<br>82.0±0.4|67.6<br>70.5±0.4<br>70.7±0.2|
||||
|MIRO [25] (use RegNetY-16GF)<br>GMDG [50] (use RegNetY-16GF)|-<br>-<br>-<br>-<br>**79.7**±1.6<br>67.7±1.8<br>87.8±0.8<br>87.9±0.7|80.4±0.2<br>80.8±0.6|
|GMDG +**L-Reg**(use RegNetY-16GF)|78.4±0.3<br>**69.3**±0.7<br>**87.9**±0.6<br>**88.0**±0.8|**80.9**±0.5|



Table 12: MDG experiments on VLCS: More results of full GMDG+L-Reg for each category. 

|**VLCS**|**caltech101**<br>**labelme**<br>**sun09**<br>**voc2007**|**Avg.**|
|---|---|---|
||||
|ERM [21]<br>MIRO [25] (use ResNet-50)<br>GMDG [50] (use ResNet-50)|97.7<br>64.3<br>73.4<br>74.6<br>-<br>-<br>-<br>-<br>98.3±0.4<br>65.9±1<br>73.4±0.8<br>79.3±1.3|77.3<br>79.0±0.0<br>79.2±0.3|
||||
|MIRO [25] (use RegNetY-16GF)<br>GMDG [50] (use RegNetY-16GF)|-<br>-<br>-<br>-<br>97.9±1.3<br>66.8±2.1<br>**80.8**±1<br>**83.9**±1.8|79.9±0.6<br>82.4±0.6|
|GMDG +**L-Reg**(use RegNetY-16GF)|**98.6**±0.1<br>**67.1**±0.1<br>80.7±0.7<br>83.0±0.8|**82.4**±0.0|



Table 13: MDG experiments on PACS: More results of full GMDG+L-Reg for each category. 

|**PACS**|**art_painting**<br>**cartoon**<br>**photo**<br>**sketch**|**Avg.**|
|---|---|---|
|ERM [21]<br>MIRO [25] (use ResNet-50)<br>GMDG [50] (use ResNet-50)|84.7<br>80.8<br>97.2<br>79.3<br>-<br>-<br>-<br>-<br>84.7±1.0<br>81.7±2.4<br>97.5±0.4<br>80.5±1.8|84.2<br>85.4±0.4<br>85.6±0.3|
|MIRO [25] (use RegNetY-16GF)<br>-<br>-<br>-<br>-<br>97.4±0.2<br>GMDG [50] (use RegNetY-16GF)<br>97.5±1.0<br>97.0±0.2<br>**99.4**±0.2<br>95.2±0.4<br>97.3±0.1|||
|GMDG +**L-Reg**(use RegNetY-16GF)|**97.6**±0.8<br>**97.1**±0.3<br>99.3±0.2<br>**95.3**±0.9|**97.4**±0.2|



Table 14: MDG experiments on DomainNet: More results of full GMDG+L-Reg for each category. 

|**DomainNet**|**clipart**<br>**info**<br>**painting**<br>**quickdraw**<br>**real**<br>**sketch**|**Avg.**|
|---|---|---|
|ERM [21]<br>MIRO [25] (use ResNet-50)<br>**GMDG**(use ResNet-50)|50.1<br>63.0<br>21.2<br>63.7<br>13.9<br>52.9<br>-<br>-<br>-<br>-<br>-<br>-<br>63.4±0.3<br>22.4±0.4<br>51.4±0.4<br>13.4±0.8<br>64.4±0.3<br>52.4±0.4|44.0<br>44.3±0.2<br>44.6±0.1|
|MIRO [25] (use RegNetY-16GF)<br>GMDG (use RegNetY-16GF)|-<br>-<br>-<br>-<br>-<br>-<br>74.0±0.3<br>39.5±1.5<br>61.5±0.3<br>**16.3**±1.2<br>73.9±1.5<br>62.8±2.4|53.8±0.1<br>54.6±0.1|
|GMDG +**L-Reg**(use RegNetY-16GF)|**74.1**±0.1<br>**42.6**±1.0<br>**62.3**±2.9<br>12.7±0.9<br>**75.9**±0.8<br>**64.6**±0.2|**55.4**±0.0|



23 

Table 15: Statistics of datasets. 

||CUB|Standford Cars|Herbarium19|CIFAR10|CIFAR100|ImageNet-100|
|---|---|---|---|---|---|---|
|Known classes|100|98|341|5|80|50|
|Seen data|1.5K|2.0K|8.9K|12.5K|20K|31.9K|
|All classes|200|196|683|10|100|100|
|Unseen data|4.5K|6.1K|25.4K|37.5K|30K|95.3K|



Table 16: Tuned weight decay values for each dataset. 

||CUB<br>Standford Cars<br>Herbarium19<br>CIFAR10<br>CIFAR100<br>ImageNet-100|
|---|---|
|Tuned weighted decay<br>_α_of L-Reg|0.02/2<br>0.02/2<br>0.02/2<br>0.05/2<br>0.005/2<br>0.005/2<br>0.1<br>0.001<br>0.2<br>0.01<br>0.0025<br>0.01|



Table 17: GCD results: Accuracy scores across fine-grained and generic PIM datasets with our L-Reg and other competitors. The best results of each group are highlighted in **bold** . Improvement and degradation in our approach from PIM are highlighted in red and blue, respectively. 

||**CUB**|**Stanford Cars**|**Herbarium19**|
|---|---|---|---|
|Approach|All<br>Known<br>Unknown|All<br>Known<br>Unknown|All<br>Known<br>Unknown|
|K-means<br>RankStats+ [22] (TPAMI-21)<br>UNO+ [19] (ICCV-21)<br>ORCA [13] (ICLR-22)<br>ORCA [13] - ViTB16<br>GCD[54] (CVPR-22)|34.3<br>38.9<br>32.1<br>33.3<br>51.6<br>24.2<br>35.1<br>49.0<br>28.1<br>27.5<br>20.1<br>31.1<br>38.0<br>45.6<br>31.8<br>**51.3**<br>**56.6**<br>**48.7**|12.8<br>10.6<br>13.8<br>28.3<br>61.8<br>12.1<br>35.5<br>**70.5**<br>18.6<br>15.9<br>17.1<br>15.3<br>33.8<br>52.5<br>25.1<br>**39.0**<br>57.6<br>**29.9**|12.9<br>12.9<br>12.8<br>27.9<br>55.8<br>12.8<br>28.3<br>**53.7**<br>14.7<br>22.9<br>25.9<br>21.3<br>25.0<br>30.6<br>19.8<br>**35.4**<br>51.0<br>**27.0**|
|InfoMax based methods||||
|RIM [27] (NeurIPS-10)<br>TIM[11] (NeurIPS-20)|52.3<br>51.8<br>52.5<br>53.4<br>51.8<br>54.2|38.9<br>57.3<br>30.1<br>39.3<br>56.8<br>30.8|40.1<br>**57.6**<br>30.7<br>40.1<br>57.4<br>30.7|
|PIM [16] (ICCV-23)|62.7<br>**75.7**<br>56.2|43.1<br>**66.9**<br>31.6|42.3<br>56.1<br>34.8|
|**PIM + L-Reg (Ours)**|**65.3**2_._6_↑_<br>**76.0** 0_._3_↑_<br>**60.0**3_._8_↑_|**44.8**1_._7_↑_<br>66.01_._4_↓_<br>**34.6**3_._0_↑_|**43.7**2_._4_↑_<br>55.80_._3_↓_<br>**37.2**1_._6_↑_|
|**CIFAR10**<br>**CIFAR100**<br>**ImageNet-100**||||
|Approach|All<br>Known<br>Unknown|All<br>Known<br>Unknown|All<br>Known<br>Unknown|
|K-means<br>RankStats+ [22] (TPAMI-21)<br>UNO+ [19] (ICCV-21)<br>ORCA [13] (ICLR-22)<br>ORCA [13] - ViTB16<br>GCD[54] (CVPR-22)|83.6<br>85.7<br>82.5<br>46.8<br>19.2<br>60.5<br>68.6<br>**98.3**<br>53.8<br>88.9<br>88.2<br>89.2<br>**97.1**<br>96.2<br>**97.6**<br>91.5<br>97.9<br>88.2|52.0<br>52.2<br>50.8<br>58.2<br>**77.6**<br>19.3<br>69.5<br>80.6<br>47.2<br>55.1<br>65.5<br>34.4<br>69.6<br>76.4<br>56.1<br>**70.8**<br>**77.6**<br>**57.0**|72.7<br>75.5<br>71.3<br>37.1<br>61.6<br>24.8<br>70.3<br>**95.0**<br>57.9<br>67.6<br>90.9<br>56.0<br>**76.5**<br>92.2<br>**68.9**<br>74.1<br>89.8<br>66.3|
|InfoMax based methods||||
|RIM [27] (NeurIPS-10)<br>TIM[11] (NeurIPS-20)|92.4<br>**98.1**<br>89.5<br>93.1<br>98.0<br>90.6|73.8<br>78.9<br>63.4<br>73.4<br>78.3<br>63.4|74.4<br>91.2<br>66.0<br>76.7<br>93.1<br>68.4|
|PIM [16] (ICCV-23)|94.7<br>97.4<br>93.3|78.3<br>84.2<br>66.5|83.1<br>**95.3**<br>77.0|
|**PIM + L-Reg(Ours)**|**94.8**0_._1_↑_<br>**97.6** 0_._2_↑_<br>**93.4**0_._1_↑_|**80.8**2_._5_↑_<br>**84.6**0_._2_↑_<br>**73.2**6_._7_↑_|**83.4**0_._3_↑_<br>94.01_._3_↓_<br>**78.0**1_._0_↑_|



Table 18: GCD results: Accuracy scores across fine-grained and generic datasets of each setting. The best results are highlighted in **bold** . To eliminate the impact of hyper-parameters on performance, we also present the results of PIM with tuned hyper-parameters (termed baseline tuned). _Lmain_ denotes the losses used in PIM. _g_ denotes the transformation applied to the input features. 

|**CUB**|**Stanford Cars**|**Herbarium19**|
|---|---|---|
|ID<br>Settings<br>All<br>Known<br>Unknown|All<br>Known<br>Unknown|All<br>Known<br>Unknown|
|1<br>Baseline(_Lmain_)<br>62.7<br>75.7<br>56.2|43.1<br>66.9<br>31.6|42.3<br>56.1<br>34.8|
|2<br>Baseline tuned (_Lmain_)<br>64.8<br>75.1<br>59.6<br>6<br>_Lmain_+_g_<br>64.9<br>**76.7**<br>58.9|42.6<br>59.3<br>34.6<br>44.7<br>65.8<br>34.6|43.1<br>57.6<br>35.4<br>43.0<br>**57.4**<br>35.2|
|9<br>**Ours** (_Lmain_+_h_+_LL−Reg_)<br>**65.3**<br>76.0<br>**60.0**|**44.8**<br>**66.0**<br>**34.6**|**43.7**<br>55.8<br>**37.2**|
|**CIFAR10**<br>**CIFAR100**<br>**ImageNet-100**|||
|ID<br>Settings<br>All<br>Known<br>Unknown|All<br>Known<br>Unknown|All<br>Known<br>Unknown|
|1<br>Baseline(_Lmain_)<br>94.7<br>97.4<br>93.3|78.3<br>84.2<br>66.5|83.1<br>**95.3**<br>77.0|
|2<br>Baseline tuned (_Lmain_)<br>95.0<br>96.1<br>94.4<br>6<br>_Lmain_+_g_<br>94.7<br>97.5<br>93.3|80.3<br>84.6<br>71.8<br>80.8<br>84.6<br>73.1|83.5<br>95.0<br>77.7<br>83.1<br>95.0<br>77.1|
|9<br>**Ours**(_Lmain_+_g_+_LL−Reg_)<br>**94.8**<br>**97.6**<br>**93.4**|**80.8**<br>**84.6**<br>**73.2**|**83.4**<br>94.0<br>**78.0**|



24 

Table 19: MDG+GCD results: accuracy scores of each dataset. Improvements are highlighted in red. 

|Method<br>|Domain gap|**PACS**<br>All<br>Known<br>Unknown|**HomeOffce**<br>All<br>Known<br>Unknown|**VLCS**<br>All<br>Known<br>Unknown|**TerraIncognita**<br>All<br>Known<br>Unknown|**DomainNet**<br>All<br>Known<br>Unknown|
|---|---|---|---|---|---|---|
|ERM<br><br>**+L-Reg**<br>|Not<br>minimized|57.26<br>77.77<br>22.33<br>55.86<br>77.69<br>19.06|44.80<br>74.67<br>8.50<br>43.56<br>71.78<br>9.68|61.51<br>82.89<br>34.88<br>61.49<br>81.33<br>36.65|37.34<br>20.46<br>45.15<br>40.73<br>29.27<br>35.56|22.56<br>40.89<br>6.85<br>25.86<br>47.07<br>7.19|
|Improvements||-1.40<br>-0.08<br>-3.27|-1.24<br>-2.89<br>1.18|-0.02<br>-1.55<br>1.77|3.38<br>8.81<br>-9.58|3.31<br>6.18<br>0.34|
|PIM<br><br>**+L-Reg**<br>|Not<br>minimized|56.35<br>71.06<br>27.43<br>58.47<br>76.49<br>26.22|43.42<br>72.44<br>8.13<br>44.20<br>71.75<br>10.85|63.19<br>80.34<br>40.24<br>59.29<br>77.96<br>36.81|47.75<br>35.31<br>50.85<br>49.74<br>34.08<br>50.01|24.03<br>42.59<br>7.86<br>24.66<br>43.86<br>7.78|
|Improvements||2.12<br>5.43<br>-1.21|0.78<br>-0.70<br>2.72|-3.90<br>-2.38<br>-3.43|2.00<br>-1.23<br>-0.84|0.63<br>1.27<br>-0.08|
|MIRO<br><br>**+L-Reg**|Minimized|56.83<br>85.62<br>24.85<br>68.44<br>97.77<br>25.64|48.28<br>80.61<br>9.03<br>53.59<br>79.50<br>22.21|61.53<br>82.72<br>35.03<br>62.07<br>83.18<br>35.21|50.22<br>39.92<br>49.45<br>44.85<br>40.87<br>38.42|31.49<br>55.44<br>10.57<br>31.58<br>54.97<br>10.98|
|Improvements||11.61<br>12.14<br>0.79|5.31<br>-1.11<br>13.18|0.54<br>0.46<br>0.18|-5.37<br>0.95<br>-11.03|0.10<br>-0.47<br>0.41|
|GMDG<br><br>**+L-Reg**<br>|Suffciently<br>minimized|58.33<br>91.46<br>10.18<br>67.82<br>91.86<br>31.33|48.85<br>81.41<br>9.22<br>51.96<br>79.74<br>18.15|61.36<br>83.31<br>33.75<br>62.32<br>82.77<br>36.09|40.02<br>32.38<br>40.07<br>45.86<br>39.77<br>41.55|31.15<br>55.17<br>10.18<br>31.75<br>55.18<br>11.30|
|Improvements||9.50<br>0.40<br>21.15|3.11<br>-1.68<br>8.92|0.97<br>-0.54<br>2.34|5.83<br>7.39<br>1.49|0.60<br>0.01<br>1.13|



Table 20: MDG+GCD results: accuracy scores of each domain in PACS dataset. 

|PACS<br>Method|**Avg**<br>All<br>Known<br>Unknown|**art_painting**<br>All<br>Known<br>Unknown|**cartoon**<br>All<br>Known<br>Unknown|**photo**<br>All<br>Known<br>Unknown|**sketch**<br>All<br>Known<br>Unknown|
|---|---|---|---|---|---|
|ERM<br>with our reg<br>Improvements|57.26<br>77.77<br>22.33<br>55.86<br>77.69<br>19.06<br>-1.40<br>-0.08<br>-3.27|47.77<br>90.00<br>0.00<br>45.33<br>85.40<br>0.00<br>-2.44<br>-4.60<br>0.00|56.08<br>83.49<br>20.47<br>50.91<br>90.09<br>0.00<br>-5.17<br>6.60<br>-20.47|59.13<br>47.35<br>68.85<br>63.70<br>48.51<br>76.23<br>4.57<br>1.16<br>7.38|66.06<br>90.23<br>0.00<br>63.52<br>86.75<br>0.00<br>-2.54<br>-3.48<br>0.00|
|PIM<br>with our reg<br>Improvements|56.35<br>71.06<br>27.43<br>58.47<br>76.49<br>26.22<br>2.12<br>5.43<br>-1.21|46.80<br>55.17<br>37.32<br>46.74<br>88.05<br>0.00<br>-0.06<br>32.87<br>-37.32|50.37<br>89.15<br>0.00<br>56.50<br>78.77<br>27.57<br>6.13<br>-10.38<br>27.57|62.05<br>49.50<br>72.40<br>64.30<br>48.51<br>77.32<br>2.25<br>-0.99<br>4.92|66.19<br>90.40<br>0.00<br>66.35<br>90.62<br>0.00<br>0.16<br>0.22<br>0.00|
|MIRO<br>with our reg<br>Improvements|56.83<br>85.62<br>24.85<br>68.44<br>97.77<br>25.64<br>11.61<br>12.14<br>0.79|51.86<br>97.70<br>0.00<br>68.46<br>97.82<br>35.24<br>16.60<br>0.11<br>35.24|56.45<br>99.91<br>0.00<br>61.51<br>98.02<br>14.09<br>5.06<br>-1.89<br>14.09|48.35<br>75.17<br>26.23<br>72.60<br>98.84<br>50.96<br>24.25<br>23.68<br>24.73|70.64<br>69.72<br>73.16<br>71.18<br>96.39<br>2.26<br>0.54<br>26.67<br>-70.90|
|GMDG<br>with our reg<br>Improvements|58.33<br>91.46<br>10.18<br>67.82<br>91.86<br>31.33<br>9.50<br>0.40<br>21.15|51.92<br>97.82<br>0.00<br>79.26<br>98.05<br>58.00<br>27.33<br>0.23<br>58.00|54.80<br>96.98<br>0.00<br>68.18<br>99.25<br>27.82<br>13.38<br>2.26<br>27.82|56.14<br>74.83<br>40.71<br>52.40<br>74.50<br>34.15<br>-3.74<br>-0.33<br>-6.56|70.45<br>96.22<br>0.00<br>71.47<br>95.66<br>5.34<br>1.02<br>-0.56<br>5.34|



Table 21: MDG+GCD results: accuracy scores of each domain in HomeOffice dataset. 

|HomeOffce<br>Method|**Avg**<br>All<br>Known<br>Unknown|**Art**<br>All<br>Known<br>Unknown|**Clipart**<br>All<br>Known<br>Unknown|**Product**<br>All<br>Known<br>Unknown|**Real World**<br>All<br>Known<br>Unknown|
|---|---|---|---|---|---|
|ERM<br>with our reg<br>Improvements|44.80<br>74.67<br>8.50<br>43.56<br>71.78<br>9.68<br>-1.24<br>-2.89<br>1.18|45.26<br>72.68<br>3.26<br>41.30<br>62.72<br>8.47<br>-3.96<br>-9.96<br>5.22|37.94<br>64.48<br>10.19<br>35.91<br>60.78<br>9.90<br>-2.03<br>-3.70<br>-0.29|46.71<br>78.74<br>9.87<br>48.34<br>78.95<br>13.14<br>1.63<br>0.21<br>3.27|49.28<br>82.80<br>10.68<br>48.68<br>84.67<br>7.22<br>-0.60<br>1.88<br>-3.46|
|PIM<br>with our reg<br>Improvements|43.42<br>72.44<br>8.13<br>44.20<br>71.75<br>10.85<br>0.78<br>-0.70<br>2.72|42.53<br>68.09<br>3.39<br>44.64<br>68.85<br>7.56<br>2.11<br>0.77<br>4.17|35.77<br>56.75<br>13.83<br>35.48<br>60.90<br>8.90<br>-0.29<br>4.15<br>-4.92|47.27<br>77.58<br>12.41<br>47.49<br>76.32<br>14.35<br>0.23<br>-1.26<br>1.94|48.11<br>87.35<br>2.90<br>49.17<br>80.92<br>12.59<br>1.06<br>-6.43<br>9.69|
|MIRO<br>with our reg<br>Improvements|48.28<br>80.61<br>9.03<br>53.59<br>79.50<br>22.21<br>5.31<br>-1.11<br>13.18|50.57<br>79.57<br>6.13<br>54.02<br>77.87<br>17.47<br>3.45<br>-1.70<br>11.34|39.55<br>67.23<br>10.60<br>43.87<br>70.98<br>15.52<br>4.32<br>3.75<br>4.92|51.35<br>86.16<br>11.32<br>59.94<br>83.95<br>32.32<br>8.59<br>-2.21<br>21.00|51.66<br>89.50<br>8.09<br>56.54<br>85.21<br>23.52<br>4.88<br>-4.29<br>15.43|
|GMDG<br>with our reg<br>Improvements|48.85<br>81.41<br>9.22<br>51.96<br>79.74<br>18.15<br>3.11<br>-1.68<br>8.92|51.60<br>81.96<br>5.08<br>52.83<br>79.15<br>12.52<br>1.24<br>-2.81<br>7.43|40.89<br>69.30<br>11.19<br>43.59<br>69.02<br>16.99<br>2.69<br>-0.28<br>5.80|51.15<br>87.53<br>9.32<br>56.31<br>83.11<br>25.48<br>5.15<br>-4.42<br>16.16|51.75<br>86.87<br>11.30<br>55.11<br>87.67<br>17.59<br>3.36<br>0.80<br>6.30|



Table 22: MDG+GCD results: accuracy scores of each domain in VLCS dataset. 

|VLCS<br>Method|**Avg**<br>All<br>Known<br>Unknown|**Caltech101**<br>All<br>Known<br>Unknown|**LabelMe**<br>All<br>Known<br>Unknown|**SUN09**<br>All<br>Known<br>Unknown|**VOC2007**<br>All<br>Known<br>Unknown|
|---|---|---|---|---|---|
|ERM<br>with our reg<br>Improvements|61.51<br>82.89<br>34.88<br>61.49<br>81.33<br>36.65<br>-0.02<br>-1.55<br>1.77|82.07<br>74.87<br>85.85<br>76.59<br>75.13<br>77.36<br>-5.48<br>0.26<br>-8.49|50.54<br>92.01<br>4.85<br>50.64<br>91.02<br>6.13<br>0.09<br>-0.99<br>1.29|62.07<br>95.15<br>11.38<br>60.70<br>92.83<br>11.48<br>-1.37<br>-2.33<br>0.10|51.35<br>69.51<br>37.45<br>58.02<br>66.35<br>51.63<br>6.66<br>-3.16<br>14.18|
|PIM<br>with our reg<br>Improvements|63.19<br>80.34<br>40.24<br>59.29<br>77.96<br>36.81<br>-3.90<br>-2.38<br>-3.43|80.39<br>72.05<br>84.77<br>72.61<br>73.33<br>72.24<br>-7.77<br>1.28<br>-12.53|53.84<br>91.74<br>12.07<br>53.98<br>90.75<br>13.45<br>0.14<br>-0.99<br>1.38|62.22<br>94.21<br>13.21<br>56.85<br>83.64<br>15.81<br>-5.37<br>-10.57<br>2.60|56.31<br>63.36<br>50.92<br>53.72<br>64.13<br>45.75<br>-2.59<br>0.77<br>-5.16|
|MIRO<br>with our reg<br>Improvements|61.53<br>82.72<br>35.03<br>62.07<br>83.18<br>35.21<br>0.54<br>0.46<br>0.18|82.77<br>74.10<br>87.33<br>82.51<br>74.62<br>86.66<br>-0.27<br>0.51<br>-0.67|51.81<br>91.83<br>7.72<br>49.51<br>94.43<br>0.00<br>-2.31<br>2.60<br>-7.72|62.22<br>95.59<br>11.09<br>60.97<br>94.65<br>9.35<br>-1.26<br>-0.94<br>-1.74|49.32<br>69.34<br>33.99<br>55.31<br>69.00<br>44.84<br>6.00<br>-0.34<br>10.85|
|GMDG<br>with our reg<br>Improvements|61.36<br>83.31<br>33.75<br>62.32<br>82.77<br>36.09<br>0.97<br>-0.54<br>2.34|82.51<br>74.87<br>86.52<br>84.54<br>74.62<br>89.76<br>2.03<br>-0.26<br>3.23|49.93<br>95.24<br>0.00<br>49.98<br>92.01<br>3.66<br>0.05<br>-3.23<br>3.66|59.86<br>93.96<br>7.62<br>61.39<br>95.03<br>9.84<br>1.52<br>1.07<br>2.22|53.13<br>69.17<br>40.85<br>53.39<br>69.43<br>41.11<br>0.26<br>0.26<br>0.26|



Table 23: MDG+GCD results: accuracy scores of each domain in TerraIncognita dataset. 

|TerraIncognita<br>Method|**Avg**<br>All<br>Known<br>Unknown|**Local 100**<br>All<br>Known<br>Unknown|**Local 38**<br>All<br>Known<br>Unknown|**Local 43**<br>All<br>Known<br>Unknown|**Local 46**<br>All<br>Known<br>Unknown|
|---|---|---|---|---|---|
|ERM<br>with our reg<br>Improvements|37.34<br>20.46<br>45.15<br>40.73<br>29.27<br>35.56<br>3.38<br>8.81<br>-9.58|46.51<br>1.25<br>57.07<br>52.94<br>0.14<br>65.27<br>6.43<br>-1.11<br>8.20|39.88<br>28.22<br>44.91<br>39.26<br>33.76<br>41.64<br>-0.62<br>5.53<br>-3.27|29.41<br>24.65<br>40.25<br>40.24<br>57.45<br>1.03<br>10.83<br>32.80<br>-39.22|33.59<br>27.70<br>38.36<br>30.47<br>25.71<br>34.32<br>-3.12<br>-1.99<br>-4.04|
|PIM<br>with our reg<br>Improvements|47.75<br>35.31<br>50.85<br>49.74<br>34.08<br>50.01<br>2.00<br>-1.23<br>-0.84|50.20<br>28.97<br>55.15<br>53.94<br>34.12<br>58.57<br>3.74<br>5.15<br>3.41|56.22<br>19.71<br>71.99<br>59.87<br>19.58<br>77.26<br>3.65<br>-0.13<br>5.28|46.69<br>47.94<br>43.86<br>47.07<br>62.75<br>11.35<br>0.38<br>14.82<br>-32.51|37.88<br>44.64<br>32.40<br>38.09<br>19.88<br>52.87<br>0.21<br>-24.76<br>20.47|
|MIRO<br>with our reg<br>Improvements|50.22<br>39.92<br>49.45<br>44.85<br>40.87<br>38.42<br>-5.37<br>0.95<br>-11.03|52.23<br>51.25<br>52.46<br>56.26<br>22.42<br>64.16<br>4.03<br>-28.83<br>11.71|55.54<br>14.73<br>73.16<br>31.27<br>23.75<br>34.52<br>-24.26<br>9.03<br>-38.64|48.93<br>62.89<br>17.13<br>54.03<br>65.11<br>28.79<br>5.10<br>2.22<br>11.66|44.19<br>30.79<br>55.06<br>37.84<br>52.18<br>26.20<br>-6.35<br>21.39<br>-28.86|
|GMDG<br>with our reg<br>Improvements|40.02<br>32.38<br>40.07<br>45.86<br>39.77<br>41.55<br>5.83<br>7.39<br>1.49|36.70<br>35.65<br>36.94<br>51.89<br>38.16<br>55.09<br>15.19<br>2.51<br>18.15|36.69<br>20.86<br>43.53<br>41.30<br>22.05<br>49.61<br>4.61<br>1.19<br>6.08|49.46<br>61.76<br>21.47<br>50.47<br>65.29<br>16.72<br>1.01<br>3.53<br>-4.75|37.24<br>11.24<br>58.33<br>39.77<br>33.59<br>44.79<br>2.53<br>22.34<br>-13.54|



25 

Table 24: MDG+GCD results: accuracy scores of each domain in DomainNet dataset. 

|DomainNet<br>Method|**Avg**<br>All<br>Known<br>Unknown|**clipart**<br>All<br>Known<br>Unknown|**info**<br>All<br>Known<br>Unknown|**painting**<br>All<br>Known<br>Unknown|
|---|---|---|---|---|
|ERM<br>with our reg<br>Improvements|22.56<br>40.89<br>6.85<br>25.86<br>47.07<br>7.19<br>3.31<br>6.18<br>0.34|31.04<br>58.32<br>7.15<br>32.03<br>58.43<br>8.91<br>0.99<br>0.11<br>1.76|17.94<br>34.71<br>6.85<br>18.17<br>34.31<br>7.50<br>0.23<br>-0.41<br>0.65|30.59<br>51.82<br>9.34<br>31.93<br>52.58<br>11.24<br>1.33<br>0.76<br>1.90|
|PIM<br>with our reg<br>Improvements|24.03<br>42.59<br>7.86<br>24.66<br>43.86<br>7.78<br>0.63<br>1.27<br>-0.08|32.01<br>57.38<br>9.80<br>31.91<br>57.76<br>9.26<br>-0.11<br>0.38<br>-0.54|18.80<br>33.56<br>9.03<br>16.99<br>30.77<br>7.89<br>-1.80<br>-2.79<br>-1.15|22.22<br>36.62<br>7.80<br>28.17<br>45.94<br>10.37<br>5.95<br>9.32<br>2.57|
|MIRO<br>with our reg<br>Improvements|31.49<br>55.44<br>10.57<br>31.58<br>54.97<br>10.98<br>0.10<br>-0.47<br>0.41|40.13<br>67.55<br>16.11<br>40.61<br>66.72<br>17.75<br>0.49<br>-0.83<br>1.64|25.84<br>48.53<br>10.84<br>25.58<br>45.83<br>12.19<br>-0.26<br>-2.70<br>1.35|37.89<br>62.45<br>13.29<br>36.74<br>62.29<br>11.15<br>-1.15<br>-0.16<br>-2.14|
|GMDG<br>with our reg<br>Improvements|31.15<br>55.17<br>10.18<br>31.75<br>55.18<br>11.30<br>0.60<br>0.01<br>1.13|40.38<br>70.69<br>13.84<br>40.91<br>68.17<br>17.05<br>0.53<br>-2.52<br>3.21|24.96<br>46.50<br>10.72<br>26.60<br>49.11<br>11.71<br>1.63<br>2.61<br>0.99|36.29<br>59.80<br>12.75<br>36.82<br>60.76<br>12.85<br>0.53<br>0.96<br>0.10|
|DomainNet<br>Method|**Avg**<br>All<br>Known<br>Unknown|**quickdraw**<br>All<br>Known<br>Unknown|**real**<br>All<br>Known<br>Unknown|**sketch**<br>All<br>Known<br>Unknown|
|ERM<br>with our reg<br>Improvements|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|8.88<br>12.83<br>4.91<br>9.04<br>14.73<br>3.31<br>0.16<br>1.91<br>-1.59|17.88<br>31.20<br>4.10<br>34.34<br>63.94<br>3.69<br>16.45<br>32.74<br>-0.41|29.01<br>56.45<br>8.76<br>29.68<br>58.41<br>8.49<br>0.67<br>1.96<br>-0.27|
|PIM<br>with our reg<br>Improvements|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|9.92<br>14.73<br>5.09<br>9.94<br>15.11<br>4.74<br>0.02<br>0.38<br>-0.35|29.09<br>53.88<br>3.42<br>30.26<br>56.13<br>3.47<br>1.17<br>2.25<br>0.05|32.12<br>59.35<br>12.03<br>30.68<br>57.43<br>10.95<br>-1.44<br>-1.93<br>-1.08|
|MIRO<br>with our reg<br>Improvements|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|8.06<br>12.12<br>3.98<br>9.36<br>15.73<br>2.95<br>1.30<br>3.61<br>-1.03|42.19<br>75.49<br>7.72<br>42.00<br>74.36<br>8.50<br>-0.20<br>-1.14<br>0.78|34.83<br>66.51<br>11.46<br>35.23<br>64.89<br>13.34<br>0.40<br>-1.62<br>1.88|
|GMDG<br>with our reg<br>Improvements|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|7.43<br>11.83<br>3.01<br>9.11<br>13.51<br>4.70<br>1.68<br>1.67<br>1.69|42.84<br>75.27<br>9.27<br>42.63<br>74.42<br>9.72<br>-0.21<br>-0.84<br>0.45|35.01<br>66.95<br>11.46<br>34.44<br>65.13<br>11.80<br>-0.58<br>-1.81<br>0.34|



26 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0027-00.png)


**----- Start of picture text -----**<br>
Unseen domain: Art painting<br>Seen domain: Photo<br>Seen domain: Sketch<br>Seen domain: Cartoon<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>**----- End of picture text -----**<br>


Figure 7: GradCAM visualizations: Baseline is GMDG. The used dataset is PACS. The model is trained under uDG+GCD setting with and without L-Reg, respectively. It can be seen that for the **known** class ‘dog,’ the model trained with L-Reg extracts the area around the nose area for classification across all seen and unseen domains. 

27 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0028-00.png)


**----- Start of picture text -----**<br>
Unseen domain: Art painting<br>Seen domain: Photo<br>Seen domain: Sketch<br>Seen domain: Cartoon<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>**----- End of picture text -----**<br>


Figure 8: GradCAM visualizations: Baseline is GMDG. The used dataset is PACS. The model is trained under uDG+GCD setting with and without L-Reg, respectively. It can be seen that for the **known** class ‘elephant,’ the model trained with L-Reg extracts the shape of long noses, teeth, and big ears for classification across all seen and unseen domains. The compromise of the known sets can be seen in the sketch domain, where those features are not significant. 

28 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0029-00.png)


**----- Start of picture text -----**<br>
Unseen domain: Art painting<br>Seen domain: Photo<br>Seen domain: Sketch<br>Seen domain: Cartoon<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>**----- End of picture text -----**<br>


Figure 9: GradCAM visualizations: Baseline is GMDG. The used dataset is PACS. The model is trained under uDG+GCD setting with and without L-Reg, respectively. It can be seen that for the **known** class ‘giraffe,’ the model trained with L-Reg extracts the feature of the long necks for classifying across all seen and unseen domains. 

29 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0030-00.png)


**----- Start of picture text -----**<br>
Unseen domain: Art painting<br>Seen domain: Photo<br>Seen domain: Sketch<br>Seen domain: Cartoon<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>**----- End of picture text -----**<br>


Figure 10: GradCAM visualizations: Baseline is GMDG. The used dataset is PACS. The model is trained under uDG+GCD setting with and without L-Reg, respectively. It can be seen that for the **known** class ‘guitar,’ the model trained with L-Reg extracts the features of the necks and the strings of the guitar for classification across all seen and unseen domains. 

30 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0031-00.png)


**----- Start of picture text -----**<br>
Unseen domain: Art painting<br>Seen domain: Photo<br>Seen domain: Sketch<br>Seen domain: Cartoon<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>**----- End of picture text -----**<br>


Figure 11: GradCAM visualizations: Baseline is GMDG. The used dataset is PACS. The model is trained under uDG+GCD setting with and without L-Reg, respectively. It can be seen that for the **unknown** class ‘horse,’ the model trained with L-Reg extracts the features of the overall outline shapes of horses for classification across all seen and unseen domains. 

31 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification_images/nips_2024_Interpret_Your_Decision_Logical_Reasoning_Regularization_for_Generalization_in_Visual_Classification.pdf-0032-00.png)


**----- Start of picture text -----**<br>
Unseen domain: Art painting<br>Seen domain: Photo<br>Seen domain: Sketch<br>Seen domain: Cartoon<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>L-Reg<br>Without<br>With  L-Reg<br>**----- End of picture text -----**<br>


Figure 12: GradCAM visualizations: Baseline is GMDG. The used dataset is PACS. The model is trained under uDG+GCD setting with and without L-Reg, respectively. It can be seen that for the **unknown** class ‘person,’ the model trained with L-Reg extracts the features of human faces for classification across all seen and unseen domains. The compromise of the known sets can be seen in the sketch domain, where those faces are not drawn. 

32 

