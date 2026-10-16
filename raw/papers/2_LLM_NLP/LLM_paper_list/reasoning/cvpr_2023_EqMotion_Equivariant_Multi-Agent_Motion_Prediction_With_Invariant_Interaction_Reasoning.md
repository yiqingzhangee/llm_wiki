# **EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning** 

Chenxin Xu[1,2] , Robby T. Tan[2] , Yuhong Tan[1] , Siheng Chen[1,3][*] , Yu Guang Wang[1] , Xinchao Wang[2] , Yanfeng Wang[3,1] 

1Shanghai Jiao Tong University, 2National University of Singapore, 3Shanghai AI Laboratory 

{xcxwakaka,tyheeeer,sihengc,yuguang.wang,wangyanfeng}@sjtu.edu.cn, {robby.tan,xinchao}@nus.edu.sg 

## **Abstract** 

_Learning to predict agent motions with relationship reasoning is important for many applications. In motion prediction tasks, maintaining motion equivariance under Euclidean geometric transformations and invariance of agent interaction is a critical and fundamental principle. However, such equivariance and invariance properties are overlooked by most existing methods. To fill this gap, we propose EqMotion, an efficient equivariant motion prediction model with invariant interaction reasoning. To achieve motion equivariance, we propose an equivariant geometric feature learning module to learn a Euclidean transformable feature through dedicated designs of equivariant operations. To reason agent’s interactions, we propose an invariant interaction reasoning module to achieve a more stable interaction modeling. To further promote more comprehensive motion features, we propose an invariant pattern feature learning module to learn an invariant pattern feature, which cooperates with the equivariant geometric feature to enhance network expressiveness. We conduct experiments for the proposed model on four distinct scenarios: particle dynamics, molecule dynamics, human skeleton motion prediction and pedestrian trajectory prediction. Experimental results show that our method is not only generally applicable, but also achieves state-of-the-art prediction performances on all the four tasks, improving by_ 24 _._ 0 _/_ 30 _._ 1 _/_ 8 _._ 6 _/_ 9 _._ 2% _. Code is available at https://github.com/MediaBrainSJTU/EqMotion._ 

## **1. Introduction** 

Motion prediction aims to predict future trajectories of multiple interacting agents given their historical observations. It is widely studied in many applications like physics [3, 28], molecule dynamics [7], autonomous driving [35] and humanrobot interaction [38,68]. In the task of motion prediction, an 

*Corresponding author. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0001-10.png)


**----- Start of picture text -----**<br>
Euclidean transformation Euclidean transformation<br>Past motion Spring Past motion Single bond<br>Future motion Stick Interaction Future motion  Double bond Interaction<br>(equivariant) (invariant) (equivariant) (invariant)<br>(a) Particle dynamics (b) Molecule dynamics<br>Euclidean transformation Euclidean transformation<br>Past motion Bone connection Past motion Meeting<br>Future motion Interaction Future motion  Interaction<br>(equivariant) (invariant) (equivariant) (invariant)<br>(c) Human skeleton motion (d) Pedestrian trajectories<br>**----- End of picture text -----**<br>


Figure 1. Motion equivariance and interaction invariance under the Euclidean geometric transformation is a fundamental principle for a prediction model, but this principle is often overlooked by previous works. In this work, we propose EqMotion to fill this gap. 

often-overlooked yet fundamental principle is that a prediction model is required to be equivariant under the Euclidean geometric transformation (including translation, rotation and reflection), and at the same time maintain the interaction relationships invariant. Motion equivariance here means that if an input motion is transformed under a Euclidean transformation, the output motion must be equally transformed under the same transformation. Interaction invariance means that the way agents interact remains unchanged under the input’s transformation. Figure 1 shows real-world examples of motion equivariance and interaction invariance. 

Employing this principle in a network design brings at least two benefits. First, the network will be robust to arbitrary Euclidean transformations. Second, the network will have the capability of being generalizable over rotations and translations of the data. This capability makes the network more compact, reducing the network’s learning burden and contributing to a more accurate prediction. 

Despite the motion equivariance property being important 

and fundamental, it is often neglected and not guaranteed by most existing motion prediction methods. The main reason is that these methods transform the input motion sequence directly into abstract feature vectors, where the geometric transformations are not traceable, causing the geometric relationships between agents to be irretrievable. Random augmentation will ease the equivariance problem, but it is still unable to guarantee the equivariance property. [30] uses non-parametric pre and post coordinate processing to achieve equivariance, but its parametric network structures do not satisfy equivariance. Some methods propose equivariant parametric network structures utilizing the higher-order representations of spherical harmonics [14, 61] or proposing an equivariant message passing [58], but they focus on the state-to-state prediction. This means that they use only one historical timestamp to predict one future timestamp. Consequently, these methods have limitations on utilizing motion’s temporal information and modeling interaction relationships since a single-state observation is insufficient for both interaction modeling and temporal dependency modeling. 

In this paper, we propose EqMotion, the first motion prediction model that is theoretically equivariant to the input motion under Euclidean geometric transformations based on the parametric network. The proposed EqMotion has three novel designs: equivariant geometric feature learning, invariant pattern feature learning and invariant interaction reasoning. To ensure motion equivariance, we propose an equivariant geometric feature learning module to learn a Euclidean transformable geometric feature through dedicated designs of equivariant operations. The geometric feature preserves motion attributes that are relevant to Euclidean transformations. To promote more comprehensive representation power, we introduce an invariant pattern feature learning module to complement the network with motion attributes that are independent of Euclidean transformations. The pattern features, cooperated into the geometric features, provide expressive motion representations by exploiting motions’ spatial-temporal dependencies. 

To further infer the interactions during motion prediction, we propose an invariant interaction reasoning module, which ensures that the captured interaction relationships are invariant to the input motion under Euclidean transformations. The module infers an invariant interaction graph by utilizing invariant factors in motions. The edge weights in the interaction graph categorize agents’ interactions into different types, leading to better interaction representation. 

We conduct extensive experiments on four different scenarios to evaluate our method’s effectiveness: particle dynamics, molecule dynamics, 3D human skeleton motion and pedestrian trajectories. Comparing to many task-specific motion prediction methods, our method is generally applicable and achieves state-of-the-art performance in all these tasks by reducing the prediction error by 24.0/30.1/8.6/9.2% 

respectively. We also present that EqMotion is lightweight, and has a model size less than 30% of many other models’ sizes. We show that EqMotion using only 5 % data can achieve a comparable performance with other methods that take full data. As a summary, here are our contributions: 

_•_ We propose EqMotion, the first motion prediction model that theoretically ensures sequence-to-sequence motion equivariance based on the parametric network. With equivariance, EqMotion promotes more generalization ability of motion feature learning, leading to more robust and accurate prediction. 

_•_ We propose a novel invariant interaction reasoning module, in which the captured interactions between agents are invariant to the input motion under Euclidean geometric transformations. With this, EqMotion achieves more generalization ability and stability in the interaction reasoning. 

_•_ We conduct experiments on four types of scenarios and find that EqMotion is applicable to all these different tasks, and importantly outperforms existing state-of-the-art methods on all the tasks. 

## **2. Related Work** 

**Equivariant Networks.** Equivariance first draws high attention on the 2D image domain. Since CNN structure is sensitive to rotations, researchers start to explore rotationequivariant designs like oriented convolutional filters [8, 49], log-polar transform [11], circular harmonics [67] or steerable filters [66]. Meanwhile, GNN architectures [73–75] exploring symmetries on both rotation and translation have been emerged. Specifically, [57, 62] achieves partial symmetries by promoting translation equivariance. [14, 61] builds filters using spherical harmonics allowing transformations between high-order representations, achieving the rotation and translation equivariance. [12, 24] construct a Lie convolution to parameterize transformations into Lie algebra form. [10] proposes a series of equivariant layers for point cloud networks. [27] propose geometric vector perceptions for protein structure learning. Recently, EGNN [58] proposes a simple equivariant message passing form without using computationally expensive high-order representations. [23] further extends it by considering geometrical constraints. However, most existing methods are only applicable to state prediction, limiting models from exploiting sequence information. [30] uses pre and post coordinate processing to achieve motion equivariance but its network structure does not satisfy equivariance. In this work, we propose an equivariant model based on the parametric network which is generally applicable to motion prediction tasks and achieves a more precise prediction. 

**Motion Prediction.** Motion prediction has wide application scenarios. [3, 53, 56] proposes graph neural networks for learning to simulate complex physical systems. [17, 28, 37, 71] both explicitly infer the interactions rela- 

tionships and perform prediction in physical systems. For human or vehicle trajectory prediction, social forces [21, 51], Markov [29, 65] process, and RNNs [1, 52, 63] are first methods to employ. Multi-model prediction methods are further proposed like using generator-discriminator structures [20, 22], multi-head output [43, 59], conditional variational autoencoders [32, 45, 55, 69, 71, 72, 76], memory mechanisms [48, 70], Gaussian mixture distribution prediction [17, 37]. The HD map information is specifically considered in the autonomous driving scenario [5, 6, 15, 22, 42, 43, 77]. For 3D human skeleton motion prediction, early methods are based on the state prediction [33, 60]. Later RNN-based models considering the sequential motion states are proposed [13, 26, 50, 64]. [19, 36] use spatial graph convolutions to directly regress the whole sequences. Besides, some methods [4,40,46,47] specifically exploit the correlations between body joints. Multi-scale graphs are built by [9, 39, 41] to capture different body-level dependencies. In this work, we propose a generally applicable motion prediction network which promotes Euclidean equivariance, a fundamental property but neglected by previous methods, to have a more robust and accurate prediction. 

## **3. Background and Problem Formulation** 

## **3.1. Motion Prediction** 

Here we introduce the general problem formulation of motion prediction, which aims to generate future motions given the historical observations. Mathematically, consider _M_ agents in a multi-agent system. Let **X** _i_ = [ **x**[1] _i[,]_ **[ x]**[2] _i[,][ · · ·]_ **[ x]** _[T] i_[p][]] _[ ∈]_ R _[T]_[p] _[×][n]_ and **Y** _i_ = [ **y** _i_[1] _[,]_ **[ y]** _i_[2] _[,][ · · ·]_ **[ y]** _i[T]_[f][]] _[∈]_[R] _[T]_[f] _[×][n]_[be][the] _[i]_[th] agent’s past and future motion, where _T_ p and _T_ f are the past and future timestamps and _n_ is the dimension of the system space. _n_ usually equals to 2 or 3 (corresponding to 2D or 3D case). The _t_ th timestamp locations **x** _[t] i_[and] **[ y]** _i[t]_[are] _[ n]_[-] dimension vectors. The whole system’s past and future motion is represented as X = [ **X** 1 _,_ **X** 2 _, · · · ,_ **X** _M_ ] _∈_ R _[M][×][T]_[p] _[×][n]_ and Y = [ **Y** 1 _,_ **Y** 2 _, · · · ,_ **Y** _M_ ] _∈_ R _[M][×][T]_[f] _[×][n]_ . We aim to propose a prediction network _F_ pred( _·_ ) so that the predicted future motions Y[�] = _F_ pred(X) are as close to the ground-truth future motions Y as possible. 

## **3.2. Equivariance and Invariance** 

The Euclidean geometric transformation has three basic forms: translation, rotation and reflection. The translation is modeled by a translation vector and rotation (or reflection) is modeled by an orthogonal rotation (or reflection) matrix. Let **t** _∈_ R _[n]_ be a translation vector and **R** _∈_ SO( _n_ ) be a _n × n_ rotation matrix, we have the following definitions for equivariance and invariance: 

**Definition 1** _Let_ X _be an input, F_ ( _·_ ) _be an operation and_ Z = _F_ (X) _be the corresponding output. The operation F_ ( _·_ ) _is called equivariant under Euclidean transformation if_ 

Z **R** + **t** = _F_ (X **R** + **t** ) _∀_ **R** _∈_ SO( _n_ ) _, ∀_ **t** _∈_ R _[n] ._ 

**Definition 2** _Let_ X _be an input, F_ ( _·_ ) _be an operation and_ Z = _F_ (X) _be the corresponding output. The operation F_ ( _·_ ) _is called invariant under Euclidean transformation if_ 

Z = _F_ (X **R** + **t** ) _∀_ **R** _∈_ SO( _n_ ) _, ∀_ **t** _∈_ R _[n] ._ Here, when a motion prediction network _F_ pred( _·_ ) satisfies Definition 1, _F_ pred( _·_ ) is said motion equivariant. When an interaction reasoning model _F_ reason( _·_ ) satisfies Definition 2, _F_ reason( _·_ ) is considered interaction invariant. We also say the output Z is equivariant/invariant (to the input motion under Euclidean transformation) if Definition 1/2 is satisfied. 

In the following section, we will introduce our motion prediction network along with geometric features that are equivariant under Euclidean transformations, and pattern features along with the interaction reasoning module are invariant under Euclidean transformations. 

## **4. Methodology** 

In this section, we present EqMotion, a motion prediction network which is equivariant under Euclidean geometric transformations. The whole network architecture is shown in Figure 2. The core of EqMotion is to successively learn equivariant geometric features and invariant pattern features by mutual cooperation in designed equivariant/invariant operations, which not only provide expressive motion and interaction representations, but also preserve equivariant/invariant properties. For agents’ motions X _∈_ R _[M][×][T]_[p] _[×][n]_ , the overall procedure of the proposed EqMotion is formulated as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0003-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0003-14.png)


Step (1a) uses an initialization layer _F_ IL( _·_ ) to obtain initial geometric features G[(0)] and pattern features **H**[(0)] . For interaction relationship unavailable cases, Step (1b) uses an invariant interaction reasoning module _F_ IRM( _·_ ) to infer an interaction graph _{_ **c** _ij}_ whose edge weight **c** _ij_ is the interaction category between agent _i_ and _j_ . Step (1c) uses the _ℓ_ th equivariant geometric feature learning layer _F_ EGFL[(] _[ℓ]_[)][(] _[·]_[)][ to] learn the ( _ℓ_ +1)th geometric feature G[(] _[ℓ]_[+1)] . Step (1d) uses the _ℓ_ th invariant pattern feature learning layer _F_ IPFL[(] _[ℓ]_[)][(] _[·]_[)][ to] learn the ( _ℓ_ +1)th pattern feature **H**[(] _[ℓ]_[+1)] . Step (1c) and Step (1d) will repeat _L_ times. Step (1e) uses an equivariant output layer _F_ EOL( _·_ ) to obtain the final prediction Y[�] . 

Note that i) to incorporate the geometric feature’s equivariance, we need to design equivariant operations for the initialization layer _F_ IL( _·_ ) and the equivariant geometric feature learning layer _F_ EGFL[(] _[ℓ]_[)][(] _[·]_[)][; ii) to introduce the pattern feature’s] invariance, we need to design invariant operations for the initialization layer _F_ IL( _·_ ) and the invariant pattern feature 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Invariant<br>ReasoningModule ···<br>Interaction graph<br>Feature  Equivariant Geometric  Equivariant Geometric  × L times··· Equivariant<br>Initialization Feature Learning Feature Learning Output Layer<br>Geometric  Geometric<br>features features<br>Invariant  Invariant<br>Past motion  Pattern Pattern ··· Predicted motion<br>Pattern  Feature learning Feature Learning Pattern<br>features features<br>**----- End of picture text -----**<br>


Figure 2. EqMotion architecture. In EqMotion, we first use a feature initialization layer to initialize geometric features and pattern features. We then successively update the geometric features and the pattern features by the equivariant geometric feature learning and invariant pattern feature learning layers, obtaining expressive feature representation. We further propose an invariant reasoning module to infer an interaction graph used in equivariant geometric feature learning. Finally, we use an equivariant output layer to obtain the final prediction. 

learning layer _F_ IPFL[(] _[ℓ]_[)][(] _[·]_[)][; and iii) the interaction graph cate-] gorizes agent’s spatial interaction into different categories for better interaction representing. The interaction graph is invariant due to the reasoning module _F_ IRM( _·_ ) design. In subsequent sections, we elaborate the details of each step. 

## **4.1. Feature Initialization** 

The feature initialization layer aims to obtain initial geometric features and pattern features while equipping them with different functionality. The initial geometric feature is denoted as G[(0)] = [ **G**[(0)] 1 _[,][ · · ·][,]_ **[ G]**[(0)] _M_[]] _[∈]_[R] _[M][×][C][×][n]_ whose _i_ th agent’s geometric feature consists of _C_ geometric coordinates. The initial pattern feature is denoted as **H**[(0)] = [ **h**[(0)] 1 _[,][ · · ·][,]_ **[ h]**[(0)] _M_[]] _[ ∈]_[R] _[M][×][D]_[ whose] _[ i]_[th agent’s pattern] feature is a _D_ -dimensional vector. Given the past motions X whose the _i_ th agent’s motion is **X** _i ∈_ R _[T]_[p] _[×][n]_ , we obtain two initial features of the _i_ th agent: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0004-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0004-06.png)


where _ϕ_ init_g( **X** ) = **W** init_g **X** _,_ **W** init_g _∈_ R _[C][×][T][p]_ is a linear function, X is the mean coordinate of all agents all past timestamps. _△_ is the difference operator to obtain the velocity **V** _i ∈_ R _[T]_[p] _[×][n]_ . _ρi ∈_ R _[T]_[p] is the velocity magnitude sequence and _θi ∈_ R _[T]_[p] is the velocity angle sequence. The superscript ( _·_ ) _[t]_ denotes the _t_ th element, _|| · ||_ 2 is vector 2- norm, angle( _·_ ) is the function calculating the angle between two vectors and _ϕ_ init_h( _·_ ) is an embedding function implemented by MLP or LSTM. [ _·_ ; _·_ ] represents concatenation. 

The geometric feature preserves both equivariant property and motion attributes that are sensitive to Euclidean geometric transforms since we linearly combine the equivariant locations. The pattern feature remains invariant and is sensitive to motion attributes independent of Euclidean geometric transforms. 

## **4.2. Invariant Reasoning Module** 

For most scenarios, the interaction relationship is implicit and unavailable. Thus, we propose an invariant reasoning module for inferring the interaction category between agents. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0004-11.png)


where _|| · ||_ 2 _,_ col is a column-wise _ℓ_ 2-distance, _ϕ_ rm( _·_ ), _ϕ_ rh( _·_ ), _ϕ_ rc( _·_ ) are learnable functions implemented by MLPs. sm( _·_ ) is the softmax function and _τ_ is the temperature to control the smoothness of the categorical distribution. The interaction edge weights _{_ **c** _ij}_ will be end-to-end learned with the whole prediction network. 

## **4.3. Equivariant Geometric Feature Learning** 

The equivariant geometric feature learning process aims to find more representative agents’ geometric features by exploiting their spatial and temporal dependencies, while maintaining the equivariance. During the process, we perform i) equivariant inner-agent attention to exploit the temporal dependencies; ii) equivariant inter-agent aggregation to model spatial interactions; and iii) equivariant non-linear function to further enhance the representation ability. 

**Equivariant inner-agent attention** To exploit the temporal dependency, we perform an attention mechanism on the coordinate dimension of the geometric feature since the coordinate dimension originates from the temporal dimension. Given the _i_ th agent’s geometric feature **G**[(] _i[ℓ]_[)] _∈_ R _[C][×][n]_ and pattern feature **h**[(] _i[ℓ]_[)][, we have] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0004-16.png)


coordinates summing up over all agents and all coordinates. 

The above operation will bring the following benefits: i) The learned attention weight is invariant and the learned dependencies between different timestamps of the motion will not be disturbed by irrelevant Euclidean transformations; and ii) The learned geometric feature is equivariant because of the coordinate-wise linear multiplication. 

**Equivariant inter-agent aggregation** The equivariant inter-agent aggregation aims to model spatial interactions between agents. The key idea is to use the reasoned or provided interaction category to learn aggregation weights, and then use the weights to aggregate neighboring agents’ geometric features. The aggregation operation reads 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0005-03.png)


where **e**[(] _ij[ℓ]_[)] _[∈]_[R] _[C]_[is the learned aggregation weights between] agent _i_ and _j_ , _|| · ||_ 2 _,_ col is a column-wise _ℓ_ 2-distance, _·_ is dot product and _Ni_ is the _i_ th agent’s neighboring set. For the _k_ th interaction category, we assign a function _ϕ_[(] _e,k[ℓ]_[)][(] _[·]_[)][ that is] implemented by MLP to model how the interaction works. 

The social influence from agent _j_ to agent _i_ is modeled by the coordinates’ difference between the two agents. The intuition behind this design is that the mutual force between two objects is always in the direction of the line they formed. 

Note that for most scenarios, we do not have an explicit definition of interacted "neighbors", thus we use a fullyconnected graph structure, which means every agent’s neighbors include all other agents. For some special cases with massive nodes, like point clouds, we construct a local neighboring set by choosing neighbors within a distance threshold. 

**Equivariant non-linear function** According to Eq. (2) and Eq. (3), the operation of inner-agent attention and interagent aggregation are linear combinations with agents’ coordinates. It is well known that non-linear operation is the key to improving neural networks’ expressivity. Therefore, we propose an equivariant non-linear function to enhance the representation ability of our prediction network while preserving equivariance. The key idea is proposing a criterion with the invariance property for splitting different conditions, and for each condition design an equivariant equation. Mathematically, the non-linear function is 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0005-08.png)


where **Q**[(] _i[ℓ]_[)] and **K**[(] _i[ℓ]_[)] is the learned query coordinates and key coordinates, **W** K[(] _[ℓ]_[)] _[∈]_[R] _[C][×][C]_[are learnable matrices for] ~~(~~ _ℓ_ ) queries and keys. G is the mean coordinates over all agents and all coordinates. **q**[(] _i,c[ℓ]_[)][,] **[ k]**[(] _i,c[ℓ]_[)][is the] _[ c]_[th coordinate of] 

**Q**[(] _i[ℓ]_[)] and **K**[(] _i[ℓ]_[)][.] _[⟨·][,][ ·⟩]_[is the vector inner product.] 

For every geometric coordinate **g** _i,c_[(] _[ℓ]_[)][, we learn a query co-] ordinate **q**[(] _i,c[ℓ]_[)][and a key coordinate] **[ k]**[(] _i,c[ℓ]_[)][.][We set the criterion] as the inner product of the query coordinate and the key coordinate. If the inner product is positive, we directly take the value of the query coordinate as output; otherwise, we clip the query coordinate vector by moving out its parallel components with the key coordinate vector. Finally, we obtain the _i_ th agent’s geometric features of the next layer **G**[(] _i[l]_[+1)] by gathering all the coordinates **g** _i,c_[(] _[l]_[+1)] . The non-linear function is equivariant, since the criterion is invariant and the two equations under two conditions are equivariant. **4.4. Invariant Pattern Feature Learning** The invariant pattern feature learning aims to obtain a more representative agent pattern feature through interacting with neighbors. Here we learn the agent’s pattern feature with an invariant message passing mechanism. Specially, we add the relative geometric feature difference into the edge modeling in the message passing to complement the information of relationships between agent absolute locations, which cannot be obtained by solely using the pattern features. Given the _i_ th agent’s pattern feature **h**[(] _i[ℓ]_[)] _∈_ R _[D]_ and geometric feature **G**[(] _i[ℓ]_[)] _∈_ R _[C][×][n]_ , the next layer’s pattern feature **h**[(] _i[l]_[+1)] is obtained by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0005-12.png)


Here **m**[(] _ij[ℓ]_[)][is][the][edge][features][and] **[p]**[(] _i[ℓ]_[)] is the aggregated neighboring features in the message passing. _|| · ||_ 2 _,_ col is a column-wise _ℓ_ 2-distance and [ _·_ ; _·_ ] denotes concatenation. Functions _ϕm_[(] _[ℓ]_[)][(] _[·]_[)][ and] _[ ϕ]_[(] _h[ℓ]_[)][(] _[·]_[)][ are implemented by MLPs.] 

## **4.5. Equivariant Output Layer** 

After total _L_ feature learning layers, we obtain the final _i_ th agent’s geometric feature **G**[(] _i[L]_[)] and pattern feature **h**[(] _i[L]_[)] . We use the geometric feature for the final prediction by a linear operation for equivariance: for the _i_ th agent: **Y** � _i_ = **W** out( **G**[(] _i[ℓ]_[)] _−_ G ~~(~~ _ℓ_ )) + G ~~(~~ _ℓ_ ) _,_ 

where **W** out _∈_ R _[T]_[f] _[×][D]_ is a learnable weight matrix. Finally, we gather all the agent prediction **Y**[�] _i_ to have the final predicted motions Y[�] . 

## **4.6. Theoretical Analysis** 

In this section, we analyze our network’s equivariance property and the interaction reasoning module’s invariance property, as stated in the following theorem. Let G[(] _[ℓ]_[)] _∈_ R _[M][×][C][×][n]_ be all agents’ geometric features, **H**[(] _[ℓ]_[)] _∈_ R _[M][×][D]_ be all agents’ pattern features at the _ℓ_ th layer, and _{_ **c** _ij}_ be the set of all the interaction categorical vectors. 

**Theorem 1** _For arbitrary translation vector_ **t** _∈_ R _[n] and rotation (or reflection) matrix_ **R** _∈_ SO( _n_ ) _, the modules with equivariance and invariance in our network satisfy:_ 

_1. For the initialization layer F_ IL( _·_ ) _, the initial geometric feature is equivariant and the initial pattern feature is invariant:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0006-02.png)


_2. The reasoning module F_ IRM( _·_ ) _along with reasoned interaction categorical vectors {_ **c** _ij} is invariant: {_ **c** _ij}_ = _F_ IRM(G[(0)] **R** + **t** _,_ **H**[(0)] ) _._ 

_3. The ℓth geometric feature learning layer F_ EGFL[(] _[ℓ]_[)][(] _[·]_[)] _[ is] equivariant:_ 

G[(] _[l]_[+1)] **R** + **t** = _F_ EGFL[(] _[ℓ]_[)][(][G][(] _[ℓ]_[)] **[R]**[ +] **[ t]** _[,]_ **[ H]**[(] _[ℓ]_[)] _[,][ {]_ **[c]** _[ij][}]_[)] _[.]_ 

_4. The ℓth pattern feature learning layer F_ IPFL[(] _[ℓ]_[)][(] _[·]_[)] _[is] invariant:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0006-07.png)


_5. The output layer F_ EOL( _·_ ) _is equivariant:_ � Y **R** + **t** = _F_ EOL(G[(] _[L]_[)] **R** + **t** ) _._ 

See the detailed proof in Appendix. Theorem 1 presents the equivariance/invariance properties with the Euclidean transformation for each operation in the proposed network. Based on Theorem 1, by combining all network operations together, we can show that the whole network is equivariant: 

**Corollary 1** _For arbitrary translation vector_ **t** _∈_ R _[n] and rotation matrix_ **R** _∈_ SO( _n_ ) _, our whole network EqMotion F_ pred( _·_ ) _satisfies:_ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0006-11.png)


## **5. Experiment** 

In this section, we validate our method on four different scenarios: particle dynamics, molecule dynamics, 3D human skeleton motion, and pedestrian trajectories. The detailed dataset description, implementation details, and additional experiment results are elaborated in the Appendix. 

**Metric** We use: i) Average Displacement Error (ADE) and Final Displacement Error (FDE). ADE/FDE is the _ℓ_ 2 distance of the predicted whole motion/endpoint to the ground truth of the whole motion/endpoint; ii) Mean Per Joint Position Error (MPJPE). It records average _ℓ_ 2 distance between predicted joints and target ones at each future timestamp. 

## **5.1. Scenario 1: Particle Dynamic Prediction** 

We use the particle _N_ -body simulation [28] in a 3D space similar to [14, 58]. In the Springs simulation, particles are randomly connected by a spring. In the Charged simulation, particles are randomly charged or uncharged. 

**Validation on interaction reasoning** Since we can get the ground-truth interaction through simulation, we evaluate the ability of interaction reasoning as a category recognition task. To evaluate the robustness of the reasoning result with 

Table 1. Interaction recognition accuracy and consistency (mean ± std in % in 5 independent runs) on the physical simulation. 

|**Model**|**Springs**<br>Accuracy<br>Consistency|**Charged**<br>Accuracy<br>Consistency|
|---|---|---|
|Corr.(path) [28]<br>Corr.(LSTM) [28]<br>EGNN [58]<br>NRI [28]<br>dNRI [17]|58.1_±_0.0<br>99.8_±_0.1<br>53.5_±_0.5<br>92.4_±_2.1<br>61.0_±_1.3<br>**100.0**_±_0.0<br>93.0_±_1.1<br>93.7_±_1.2<br>93.3_±_2.0<br>89.6_±_2.0|57.5_±_0.1<br>87.9_±_0.1<br>57.2_±_0.4<br>91.7_±_1.1<br>58.2_±_1.4<br>**100.0**_±_0.0<br>70.0_±_0.6<br>88.5_±_1.3<br>70.4_±_1.7<br>83.6_±_1.8|
|**Ours**<br>|**97.6**_±_1.1<br>**100.0**_±_0.0<br>  <br>|**80.9**_±_3.4<br>**100.0**_±_0.0<br>  <br>|
|Supervised|98.7_±_0.2<br>100.0_±_0.0|97.4_±_0.2<br>100.0_±_0.0|




![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0006-20.png)


Figure 3. Comparison of displacement error across different timestamps on the physical simulation. 

the Euclidean transformation, we introduce recognition "consistency", which is the ratio of the same recognition result under 20 random Euclidean transformations. The "Supervised" represents an upper bound that uses the ground-truth category to train the model. Table 1 reports the comparison of reasoning results on both springs and charged simulations, for recognizing whether there is a spring connection or electrostatic force. We see that i) our method achieves significant improvement on the recognition accuracy and has more robust reasoning results due to the invariance of our reasoning module; and ii) our method achieves a close performance to the upper bound, reflecting that our method is capable of reasoning a robust and accurate interaction category. 

**Validation on future prediction** We conduct the experiment to evaluate the prediction performance on the charged settings. Figure 3 compares the displacement error at all future timestamps of different methods. We see that our method (red line) achieves state-of-the-art prediction performance, reflecting our model’s efficiency in future prediction. 

## **5.2. Scenario 2: Molecule Dynamic Prediction** 

We adopt the MD17 [7] dataset which contains the motions of different molecules generated via a molecular dynamics simulation environment. The goal is to predict the motions of every atom of the molecule. We randomly pick four kinds of molecules: Aspirin, Benzene, Ethanol and Malonaldehyde and learn a prediction model for each molecule. 

Table 4 presents the comparison of different motion prediction methods. We achieve state-of-the-art prediction performance on all four molecules. The ADE/FDE across four molecules is decreased by 34.2%/30.1% on average. 

Table 2. Comparisons of short-term skeleton motion prediction on 11 representative actions and average results across all actions on H3.6M. 

|Motion|Walking|Eating|Smoking|Discussion<br>Directions<br>Phoning|Discussion<br>Directions<br>Phoning|Discussion<br>Directions<br>Phoning|
|---|---|---|---|---|---|---|
|millisecond|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|
|Res-sup. (CVPR’17)<br>Traj-GCN (ICCV’19)<br>DMGNN (CVPR’20)<br>MSRGCN (ICCV’21)<br>PGBIG (CVPR’22)<br>SPGSN (ECCV’22)<br>EqMotion(Ours)|29.4 50.8<br>76.0<br>81.5<br>12.3 23.0<br>39.8<br>46.1<br>17.3 30.7<br>54.6<br>65.2<br>12.2 22.7<br>38.6<br>45.2<br>10.2 19.8<br>34.5<br>40.3<br>10.1 19.4<br>34.8<br>41.5<br>**9.0**<br>**17.5**<br>**32.6**<br>**39.2**|16.8<br>30.6<br>56.9<br>68.7<br>8.4<br>16.9<br>33.2<br>40.7<br>11.0<br>21.4<br>36.2<br>43.9<br>8.4<br>17.1<br>33.0<br>40.4<br>7.0<br>15.1<br>30.6<br>38.1<br>7.1<br>14.9<br>30.5<br>37.9<br>**6.3**<br>**13.6**<br>**28.9**<br>**36.5**|23.0 42.6<br>70.1<br>82.7<br>7.9<br>16.2<br>31.9<br>38.9<br>9.0<br>17.6<br>32.1<br>40.3<br>8.0<br>16.3<br>31.3<br>38.2<br>6.6<br>14.1<br>28.2<br>34.7<br>6.7<br>13.8<br>28.0<br>34.6<br>**5.5**<br>**11.3**<br>**23.0**<br>**29.3**|32.9<br>61.2<br>90.9<br>96.2<br>12.5<br>27.4<br>58.5<br>71.7<br>17.3<br>34.8<br>61.0<br>69.8<br>12.0<br>26.8<br>57.1<br>69.7<br>10.0<br>23.8<br>53.6<br>66.7<br>10.4<br>23.8<br>53.6<br>67.1<br>**8.2**<br>**18.9**<br>**42.1**<br>**53.9**|35.4<br>57.3<br>76.3<br>87.7<br>9.0<br>19.9<br>43.4<br>53.7<br>13.1<br>24.6<br>64.7<br>81.9<br>8.6<br>19.7<br>43.3<br>53.8<br>7.2<br>17.6<br>40.9<br>51.5<br>7.4<br>17.2<br>39.8<br>50.3<br>**6.3**<br>**15.8**<br>**38.9**<br>**50.1**|38.0<br>69.3 115.0<br>126.7<br>10.2<br>21.0<br>42.5<br>52.3<br>12.5<br>25.8<br>48.1<br>58.3<br>10.1<br>20.7<br>41.5<br>51.3<br>8.3<br>18.3<br>38.7<br>48.4<br>8.7<br>18.3<br>38.7<br>48.5<br>**7.4**<br>**16.7**<br>**36.9**<br>**47.0**|
|Motion|Posing|Sitting|SittingDown|Waiting|WalkingTogether|Average|
|millisecond|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|
|Res-sup. (CVPR’17)<br>Traj-GCN (ICCV’19)<br>DMGNN (CVPR’20)<br>MSRGCN (ICCV’21)<br>PGBIG (CVPR’22)<br>SPGSN (ECCV’22)<br>EqMotion(Ours)|36.1 69.1 130.5<br>157.1<br>13.7 29.9<br>66.6<br>84.1<br>15.3 29.3<br>71.5<br>96.7<br>12.8 29.4<br>67.0<br>85.0<br>10.7 25.7<br>60.0<br>76.6<br>10.7 25.3<br>59.9<br>76.5<br>**8.2**<br>**18.9**<br>**43.4**<br>**57.5**|42.6<br>81.4<br>134.7 151.8<br>10.6<br>21.9<br>46.3<br>57.9<br>11.9<br>25.1<br>44.6<br>50.2<br>10.5<br>22.0<br>46.3<br>57.8<br>8.8<br>19.2<br>42.4<br>53.8<br>9.3<br>19.4<br>42.3<br>53.6<br>**8.1**<br>**18.0**<br>**41.2**<br>**52.9**|47.3 86.0<br>145.8<br>168.9<br>16.1 31.1<br>61.5<br>75.5<br>15.0 32.9<br>77.1<br>93.0<br>16.1 31.6<br>62.5<br>76.8<br>13.9 27.9<br>57.4<br>71.5<br>14.2 27.7<br>56.8<br>**70.7**<br>**13.0 26.5**<br>**56.2**<br>**70.7**|30.6<br>57.8<br>106.2<br>121.5<br>11.4<br>24.0<br>50.1<br>61.5<br>12.2<br>24.2<br>59.6<br>77.5<br>10.7<br>23.1<br>48.3<br>59.2<br>8.9<br>20.1<br>43.6<br>54.3<br>9.2<br>19.8<br>43.1<br>54.1<br>**7.6**<br>**17.4**<br>**39.9**<br>**51.1**|26.8<br>50.1<br>80.2<br>92.2<br>10.5<br>21.0<br>38.5<br>45.2<br>14.3<br>26.7<br>50.1<br>63.2<br>10.6<br>20.9<br>37.4<br>43.9<br>8.7<br>18.6<br>34.4<br>41.0<br>8.9<br>18.2<br>33.8<br>40.9<br>**7.8**<br>**16.1**<br>**30.6**<br>**37.1**|34.7<br>62.0 101.1<br>115.5<br>12.7<br>26.1<br>52.3<br>63.5<br>17.0<br>33.6<br>65.9<br>79.7<br>12.1<br>25.6<br>51.6<br>62.9<br>10.3<br>22.7<br>47.4<br>58.5<br>10.4<br>22.3<br>47.1<br>58.3<br>**9.1**<br>**20.1**<br>**43.7**<br>**55.0**|



Table 3. Comparisons of long-term skeleton motion prediction on 8 representative actions and average results across all actions on H3.6M. 

|Motion|Walking|Eating|Smoking|Discussion|Greeting|Phoning|Posing|WalkingTogether|Average|
|---|---|---|---|---|---|---|---|---|---|
|millisecond|560ms 1000ms|560ms 1000ms|560ms 1000ms|560ms 1000ms|560ms 1000ms|560ms 1000ms|560ms 1000ms|560ms<br>1000ms|560ms 1000ms|
|Res-Sup. [50]<br>Traj-GCN [47]<br>DMGNN [41]<br>MSRGCN [9]<br>PGBIG [44]<br>SPGSN [40]<br>EqMotion(Ours)|81.7<br>100.7<br>54.1<br>59.8<br>71.4<br>85.8<br>52.7<br>63.0<br>48.1<br>56.4<br>46.9<br>53.6<br>**43.4**<br>**52.8**|79.9<br>100.2<br>53.4<br>77.8<br>58.1<br>86.7<br>52.5<br>77.1<br>51.1<br>76.0<br>49.8<br>73.4<br>**48.4**<br>**73.0**|94.8<br>137.4<br>50.7<br>72.6<br>50.9<br>72.2<br>49.5<br>71.6<br>46.5<br>69.5<br>46.7<br>68.6<br>**41.0**<br>**63.4**|121.3<br>161.7<br>91.6<br>121.5<br>81.9<br>138.3<br>88.6<br>117.6<br>87.1<br>118.2<br>89.7<br>118.6<br>**75.3**<br>**105.6**|156.3<br>184.3<br>115.4<br>148.8<br>144.5<br>170.5<br>116.3<br>147.2<br>110.2<br>143.5<br>111.0<br>143.2<br>**108.7**<br>**142.0**|143.9<br>186.8<br>69.2<br>103.1<br>71.3<br>108.4<br>68.3<br>104.4<br>65.9<br>102.7<br><br>|165.4<br>236.8<br>114.5<br>173.0<br>125.5<br>188.2<br>116.3<br>174.3<br>106.1<br>164.8<br><br>|173.6<br>202.3<br>55.0<br>65.6<br>70.5<br>86.9<br>52.9<br>65.9<br>51.9<br>64.3<br><br>|129.2<br>165.0<br>81.6<br>114.3<br>93.6<br>127.6<br>81.1<br>114.2<br>76.9<br>110.3<br><br>|
|||||||66.7<br>102.5<br>**64.7**<br>**101.0**|110.3<br>165.4<br>**84.9**<br>**139.4**|49.8<br>60.9<br>**44.5**<br>**56.0**|77.4<br>109.6<br>**73.4**<br>**106.9**|
|Table 4. Prediction ADE/FDE (_×_10_−_2) on the MD17 dataset.<br>Aspirin<br>Benzene<br>Ethanol<br>Malonaldehyde<br>Radial Field [31] 17.98/26.20 7.73/12.47 8.10/10.61<br>16.53/25.10<br>TFN [61]<br>15.02/21.35 7.55/12.30 8.05/10.57<br>15.21/24.32<br>SE(3)-Trans [14] 15.70/22.39 7.62/12.50 8.05/10.86<br>15.44/24.47<br>EGNN [58]<br>14.61/20.65 7.50/12.16 8.01/10.22<br>15.21/24.00<br>LSTM<br>17.59/24.79<br>6.06/9.46<br>7.73/9.88<br>15.14/22.90<br>S-LSTM [1]<br>13.12/18.14<br>3.06/3.52<br>7.23/9.85<br>11.93/18.43<br> <br><br><br><br><br>Traj-GCN<br>SPGSN<br>EqMotion<br>(Ours)<br>G.T.<br>40<br>200<br>360<br>520<br>680<br>840<br>1000<br>Time (ms)<br>0||||||||||
|Aspirin<br>Benzene<br>Ethanol<br>Malonaldehyde||||||||||
|Radial Field [31] 17.98/26.20 7.73/12.47 8.10/10.61<br>16.53/25.10<br>TFN [61]<br>15.02/21.35 7.55/12.30 8.05/10.57<br>15.21/24.32<br>SE(3)-Trans [14] 15.70/22.39 7.62/12.50 8.05/10.86<br>15.44/24.47<br>EGNN [58]<br>14.61/20.65 7.50/12.16 8.01/10.22<br>15.21/24.00<br>LSTM<br>17.59/24.79<br>6.06/9.46<br>7.73/9.88<br>15.14/22.90<br>S-LSTM [1]<br>13.12/18.14<br>3.06/3.52<br>7.23/9.85<br>11.93/18.43<br> <br><br><br><br>||||||||||
|NRI [28]<br>12.60/18.50<br>1.89/2.58<br>6.69/8.78<br>12.79/19.86<br>NMMP [22]<br>10.41/14.67<br>2.21/3.33<br>6.17/7.86<br>9.50/14.89<br>GroupNet [69]<br>10.62/14.00<br>2.02/2.95<br>6.00/7.88<br>7.99/12.49<br>EqMotion(Ours)<br>**5.95**/**8.38**<br>**1.18**/**1.73**<br>**5.05**/**7.02**<br>**5.85**/**9.02**||||||||||



||Aspirin|Benzene|Ethanol|Malonaldehyde|
|---|---|---|---|---|
|Radial Field [31]|17.98/26.20|7.73/12.47|8.10/10.61|16.53/25.10|
|TFN [61]|15.02/21.35|7.55/12.30|8.05/10.57|15.21/24.32|
|SE(3)-Trans [14]|15.70/22.39|7.62/12.50|8.05/10.86|15.44/24.47|
|EGNN [58]<br>LSTM|14.61/20.65 <br>17.59/24.79|7.50/12.16 <br>6.06/9.46|8.01/10.22<br>7.73/9.88|15.21/24.00<br>15.14/22.90|
|S-LSTM [1]<br>NRI [28]<br>NMMP [22]<br>GroupNet [69]|13.12/18.14<br>12.60/18.50<br>10.41/14.67<br>10.62/14.00|3.06/3.52<br>1.89/2.58<br>2.21/3.33<br>2.02/2.95|7.23/9.85<br>6.69/8.78<br>6.17/7.86<br>6.00/7.88|11.93/18.43<br>12.79/19.86<br>9.50/14.89<br>7.99/12.49|
|EqMotion(Ours)|**5.95**/**8.38**|**1.18**/**1.73**|**5.05**/**7.02**|**5.85**/**9.02**|



Figure 4. Qualitative comparison on the action ‘Sitting Down’ of H3.6M. EqMotion completes the action more accurately. 

Motion outperforms many previous methods that are taskspecific, reflecting the effectiveness of EqMotion. 

## **5.3. Scenario 3: Human Skeleton Motion Prediction** 

We conduct experiments on the Human 3.6M (H3.6M) dataset [25] for 3D human skeleton motion prediction. H3.6M contains 7 subjects performing 15 actions. Following the standard paradigm [41, 50], we train the models on 6 subjects and tested on the specific clips of the 5th subject. 

## **5.4. Scenario 4: Pedestrian Trajectory Prediction** 

We conduct experiments on pedestrian trajectory prediction using the ETH-UCY dataset [34, 54], which contains 5 subsets, ETH, HOTEL, UNIV, ZARA1, and ZARA2. Following the standard setting [1, 20, 76], we use 3.2 seconds (8 timestamps) to predict the 4.8 seconds (12 timestamps). 

**Short-term motion prediction** Short-term motion prediction aims to predict future poses within 400 milliseconds. We compare EqMotion with six state-of-the-art methods. Table 2 presents the comparison of prediction MPJPEs. See the overall result in the Appendix. We see that i) EqMotion obtains superior performance at most timestamps across all the actions; and ii) compared to the baselines, EqMotion achieves much lower MPJPEs by 8.6% on average. 

Here we apply our EqMotion to two prediction modes: deterministic and multi-prediction. Deterministic means the model only outputs a single prediction for each input motion. Multi-prediction means the model has 20 predictions for each input motion. Under multi-prediction, ADE and FDE will be calculated by the best-performed predicted motion. To adapt to multi-prediction, we slightly modify EqMotion to repeat the last feature updating layer and the output layer 20 times in parallel to have a multi-head prediction, see the details in the Appendix. Table 5 compares our method with sixteen baselines. We observe that i) EqMotion achieves state-of-the-art performance with the lowest average ADE and FDE, outperforming many baseline methods that are specifically designed for this task. EqMotion reduces the FDE by 10.4% and 7.9% under deterministic and multiprediction settings; and ii) EqMotion achieves the best or the 

**Long-term motion prediction** Long-term motion prediction aims to predict the poses over 400 milliseconds. Table 3 presents the prediction MPJPEs of various methods. We see that EqMotion achieves more effective prediction on most actions and has lower MPJPEs by 3.8 % on average. 

**Visualization results** Figure 4 visualizes and compares the predictions on H3.6M. Baselines fail to predict future movements. EqMotion completes the action more precisely. 

The quantitative and visualization results reveal that Eq- 

Table 5. Prediction performance on the ETH-UCY dataset. The **bold** /underline font denotes the best/second best result. 

||Performance (ADE/FDE)|Performance (ADE/FDE)|
|---|---|---|
||||
|Deterministic|ETH<br>Hotel<br>Univ<br>Zara1<br>Zara2|Average|
||||
|S-LSTM [1]<br>SGAN-ind [20]<br>Traj++ [55]<br>TransF [16]<br>MemoNet [70]<br>EqMotion(Ours)|1.09/2.35 0.79/1.76 0.67/1.40 0.47/1.00 0.56/1.17<br>1.13/2.21 1.01/2.18 0.60/1.28 0.42/0.91 0.52/1.11<br>1.02/2.00 0.33/0.62 0.53/1.19 0.44/0.99 0.32/0.73<br>1.03/2.10 0.36/0.71 0.53/1.32 0.44/1.00 0.34/0.76<br>1.00/2.08 0.35/0.67 0.55/1.19 0.46/1.00 0.37/0.82<br>**0.96**/**1.92 0.30**/**0.58 0.50**/**1.10 0.39**/**0.86 0.30**/**0.68**|0.72/1.54<br>0.74/1.54<br>0.53/1.11<br>0.54/1.17<br>0.55/1.15<br>**0.49**/**1.03**|
||||
|Multi-prediction|ETH<br>Hotel<br>Univ<br>Zara1<br>Zara2|Average|
||||
|SGAN [20]<br>NMMP [22]<br>Traj++ [55]<br>PECNet [45]<br>Agentformer [76]<br>GroupNet [69]<br>MID [18]<br>GP-Graph [2]<br>EqMotion(Ours)|0.87/1.62 0.67/1.37 0.76/0.52 0.35/0.68 0.42/0.84<br>0.61/1.08 0.33/0.63 0.52/1.11 0.32/0.66 0.43/0.85<br>0.61/1.02 0.19/0.28 0.30/0.54 0.24/0.42 0.18/0.31<br>0.54/0.87 0.18/0.24 0.35/0.60 0.22/0.39 0.17/0.30<br>0.45/0.75 0.14/0.22 0.25/0.45<br>0.18<br>/**0.30** 0.14<br>/0.24<br>0.46/0.73 0.15/0.25 0.26/0.49 0.21/0.39 0.17/0.33<br>**0.39**/0.66 0.13<br>/0.22 **0.22**/0.45<br>**0.17**/**0.30 0.13**/0.27<br>0.43/0.63<br>0.18/0.30 0.24/0.42 **0.17**/0.31<br>0.15/0.29<br>0.40<br>/**0.61 0.12**/**0.18** 0.23<br>/**0.43** 0.18<br>/0.32 **0.13**/**0.23**|0.61/1.21<br>0.41/0.82<br>0.30/0.51<br>0.29/0.48<br>0.23<br>/0.39<br>0.25/0.44<br>**0.21**/0.38<br>0.23/0.39<br>**0.21**/**0.35**|
|Table 6. Ablation study on key modules of EqMotion on H3.6M.<br>EGFL IPFL IRM<br>80ms 160ms 320ms 400ms Average<br>12.9<br>31.9<br>68.2<br>82.4<br>48.9<br>✓<br>10.1<br>22.6<br>48.7<br>60.7<br>35.5<br>✓<br>✓<br>9.2<br>20.8<br>45.4<br>57.0<br>33.1<br>✓<br>✓<br>✓<br>**9.1**<br>**20.1**<br>**43.7**<br>**55.0**<br>**32.0**|||



Table 7. Ablation study on operations in the equivariant geometric feature learning on H3.6M dataset. 

|Ablation|80ms<br>160ms<br>320ms<br>400ms<br>Average|
|---|---|
|||
|w/o Inner att<br>w/o Inter agg<br>w/o Non-linear<br>**EqMotion**|9.2<br>20.5<br>44.3<br>55.7<br>32.4<br>9.7<br>22.0<br>47.2<br>58.9<br>34.5<br>9.4<br>21.3<br>46.7<br>58.6<br>34.0<br>**9.1**<br>**20.1**<br>**43.7**<br>**55.0**<br>**32.0**|



second best ADE/FDE result on most subsets, reflecting its effectiveness on the pedestrian trajectory prediction. 

## **5.5. Ablation Studies** 

**Effect of network modules** We explore the effect of three proposed key modules in EqMotion on the H3.6M dataset, including the equivariant geometric feature learning (EGFL), the invariant pattern feature learning (IPFL) and the invariant reasoning module (IRM). Table 6 presents the experimental result. It is observed that i) the proposed three key modules all contribute to an accurate prediction; and ii) the equivariant geometric feature learning module is most important since learning a comprehensive equivariant geometric feature directly for prediction is the most important. 

**Effect of equivariant operations** We explore the effect of three proposed operations in the equivariant feature learning module in EqMotion, including the inner-agent attention (Inner att), inter-agent aggregation (Inter agg) and non-linear function (Non-linear). Table 7 presents the results. We see that the proposed three key operations all contribute to promoting an accurate prediction. 

**Different amounts of training data** Figure 5 presents the comparison of model performance under different amounts of training data on H3.6M. We see that i) our 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0008-09.png)


Figure 5. Comparison of model performance on different amounts of data in short-term prediction on H3.6M dataset. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0008-11.png)


**----- Start of picture text -----**<br>
(CVPR’20)<br>(ICCV’19) (ICCV’21)<br>(ECCV’20)<br>(ECCV’22)<br>**----- End of picture text -----**<br>


Figure 6. Comparison of model size and MPJPE in short-term prediction on H3.6M dataset. The target means the ideal model. 

method achieves the best prediction performance under all training data ratios; and ii) our method even outperforms some full-data using baselines by only using 5% of training data since the equivariant design promotes the network generalization ability under Euclidean transformations. 

**Model size** Figure 6 compares EqMotion to existing methods in terms of the model size and prediction results in short-term prediction on H3.6M. We can observe that EqMotion has the smallest model size (less than 30% of other models’ sizes) with the lowest MPJPE thanks to the equivariant design that compacts the model free from generalizing over rotations and translations of the data. 

## **6. Conclusion** 

In this work, we present EqMotion, a motion prediction network that is theoretically equivariant under Euclidean transformations. EqMotion includes three novel designs: the equivariant geometric feature learning, the invariant pattern feature learning and the invariant reasoning module. We evaluate our method on four different scenarios and our method achieves state-of-the-art prediction performance. 

**Acknowledgements.** This project is partially supported by National Natural Science Foundation of China under Grant 62171276, the Science and Technology Commission of Shanghai Municipal under Grant 21511100900 and 22DZ2229005, and the National Research Foundation Singapore under its AI Singapore Programme (Award Number: AISG2-RP-2021-023). Robby T. Tan’s work is supported by MOE AcRF Tier, A-0009455-01-00. 

## **References** 

- [1] Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, and Silvio Savarese. Social lstm: Human trajectory prediction in crowded spaces. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 961–971, 2016. 3, 7, 8, 14 

- [2] Inhwan Bae, Jin-Hwi Park, and Hae-Gon Jeon. Learning pedestrian group representations for multi-modal trajectory prediction. In _European Conference on Computer Vision_ , pages 270–289. Springer, 2022. 8 

- [3] Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, et al. Interaction networks for learning about objects, relations and physics. _Advances in neural information processing systems_ , 29, 2016. 1, 2 

- [4] Yujun Cai, Lin Huang, Yiwei Wang, Tat-Jen Cham, Jianfei Cai, Junsong Yuan, Jun Liu, Xu Yang, Yiheng Zhu, Xiaohui Shen, et al. Learning progressive joint propagation for human motion prediction. In _European Conference on Computer Vision_ , pages 226–242. Springer, 2020. 3 

- [5] Sergio Casas, Wenjie Luo, and Raquel Urtasun. Intentnet: Learning to predict intention from raw sensor data. In _Conference on Robot Learning_ , pages 947–956. PMLR, 2018. 3 

- [6] Yuning Chai, Benjamin Sapp, Mayank Bansal, and Dragomir Anguelov. Multipath: Multiple probabilistic anchor trajectory hypotheses for behavior prediction. _arXiv preprint arXiv:1910.05449_ , 2019. 3 

- [7] Stefan Chmiela, Alexandre Tkatchenko, Huziel E Sauceda, Igor Poltavsky, Kristof T Schütt, and Klaus-Robert Müller. Machine learning of accurate energy-conserving molecular force fields. _Science advances_ , 3(5):e1603015, 2017. 1, 6, 14 

- [8] Taco Cohen and Max Welling. Group equivariant convolutional networks. In _International conference on machine learning_ , pages 2990–2999. PMLR, 2016. 2 

- [9] Lingwei Dang, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. Msr-gcn: Multi-scale residual graph convolution networks for human motion prediction. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 11467–11476, 2021. 3, 7 

- [10] Congyue Deng, Or Litany, Yueqi Duan, Adrien Poulenard, Andrea Tagliasacchi, and Leonidas J Guibas. Vector neurons: A general framework for so (3)-equivariant networks. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 12200–12209, 2021. 2 

- [11] Carlos Esteves, Christine Allen-Blanchette, Xiaowei Zhou, and Kostas Daniilidis. Polar transformer networks. _arXiv preprint arXiv:1709.01889_ , 2017. 2 

- [12] Marc Finzi, Samuel Stanton, Pavel Izmailov, and Andrew Gordon Wilson. Generalizing convolutional neural networks for equivariance to lie groups on arbitrary continuous data. In _International Conference on Machine Learning_ , pages 3165– 3176. PMLR, 2020. 2 

- [13] Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. Recurrent network models for human dynamics. In _Proceedings of the IEEE international conference on computer vision_ , pages 4346–4354, 2015. 3 

- [14] Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling. Se (3)-transformers: 3d roto-translation equivariant attention networks. _Advances in Neural Information Processing Systems_ , 33:1970–1981, 2020. 2, 6, 7, 13 

- [15] Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, and Cordelia Schmid. Vectornet: Encoding hd maps and agent dynamics from vectorized representation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 11525– 11533, 2020. 3 

- [16] Francesco Giuliari, Irtiza Hasan, Marco Cristani, and Fabio Galasso. Transformer networks for trajectory forecasting. In _2020 25th International Conference on Pattern Recognition (ICPR)_ , pages 10335–10342. IEEE, 2021. 8 

- [17] Colin Graber and Alexander G Schwing. Dynamic neural relational inference. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 8513–8522, 2020. 2, 3, 6 

- [18] Tianpei Gu, Guangyi Chen, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, and Jiwen Lu. Stochastic trajectory prediction via motion indeterminacy diffusion. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 17113–17122, 2022. 8 

- [19] Xiao Guo and Jongmoo Choi. Human motion prediction via learning local structure representations and temporal dependencies. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 33, pages 2580–2587, 2019. 3 

- [20] Agrim Gupta, Justin Johnson, Li Fei-Fei, Silvio Savarese, and Alexandre Alahi. Social gan: Socially acceptable trajectories with generative adversarial networks. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 2255–2264, 2018. 3, 7, 8, 14 

- [21] Dirk Helbing and Peter Molnar. Social force model for pedestrian dynamics. _Physical review E_ , 51(5):4282, 1995. 3 

- [22] Yue Hu, Siheng Chen, Ya Zhang, and Xiao Gu. Collaborative motion prediction via neural motion message passing. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6319–6328, 2020. 3, 7, 8 

- [23] Wenbing Huang, Jiaqi Han, Yu Rong, Tingyang Xu, Fuchun Sun, and Junzhou Huang. Equivariant graph mechanics networks with constraints. _arXiv preprint arXiv:2203.06442_ , 2022. 2 

- [24] Michael J Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, and Hyunjik Kim. Lietransformer: Equivariant self-attention for lie groups. In _International Conference on Machine Learning_ , pages 4533–4543. PMLR, 2021. 2 

- [25] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. _IEEE transactions on pattern analysis and machine intelligence_ , 36(7):1325–1339, 2013. 7, 14 

- [26] Ashesh Jain, Amir R Zamir, Silvio Savarese, and Ashutosh Saxena. Structural-rnn: Deep learning on spatio-temporal graphs. In _Proceedings of the ieee conference on computer vision and pattern recognition_ , pages 5308–5317, 2016. 3 

- [27] Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael JL Townshend, and Ron Dror. Learning from protein structure with geometric vector perceptrons. _arXiv preprint arXiv:2009.01411_ , 2020. 2 

- [28] Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting systems. In _International Conference on Machine Learning_ , pages 2688–2697. PMLR, 2018. 1, 2, 6, 7, 13 

- [29] Kris M Kitani, Brian D Ziebart, James Andrew Bagnell, and Martial Hebert. Activity forecasting. In _European conference on computer vision_ , pages 201–214. Springer, 2012. 3 

- [30] Miltiadis Kofinas, Naveen Nagaraja, and Efstratios Gavves. Roto-translated local coordinate frames for interacting dynamical systems. _Advances in Neural Information Processing Systems_ , 34:6417–6429, 2021. 2 

- [31] Jonas Köhler, Leon Klein, and Frank Noé. Equivariant flows: sampling configurations for multi-body systems with symmetric energies. _arXiv preprint arXiv:1910.00753_ , 2019. 7 

- [32] Namhoon Lee, Wongun Choi, Paul Vernaza, Christopher B Choy, Philip HS Torr, and Manmohan Chandraker. Desire: Distant future prediction in dynamic scenes with interacting agents. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 336–345, 2017. 3 

- [33] Andreas M Lehrmann, Peter V Gehler, and Sebastian Nowozin. Efficient nonlinear markov models for human motion. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1314–1321, 2014. 3 

- [34] Alon Lerner, Yiorgos Chrysanthou, and Dani Lischinski. Crowds by example. In _Computer graphics forum_ , volume 26, pages 655–664. Wiley Online Library, 2007. 7, 14 

- [35] Jesse Levinson, Jake Askeland, Jan Becker, Jennifer Dolson, David Held, Soeren Kammel, J Zico Kolter, Dirk Langer, Oliver Pink, Vaughan Pratt, et al. Towards fully autonomous driving: Systems and algorithms. In _2011 IEEE intelligent vehicles symposium (IV)_ , pages 163–168. IEEE, 2011. 1 

- [36] Chen Li, Zhen Zhang, Wee Sun Lee, and Gim Hee Lee. Convolutional sequence to sequence model for human dynamics. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 5226–5234, 2018. 3 

- [37] Jiachen Li, Fan Yang, Masayoshi Tomizuka, and Chiho Choi. Evolvegraph: Multi-agent trajectory prediction with dynamic relational reasoning. _Proceedings of the Neural Information Processing Systems (NeurIPS)_ , 2020. 2, 3 

- [38] Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, and Qi Tian. Actional-structural graph convolutional networks for skeleton-based action recognition. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 3595–3603, 2019. 1 

- [39] Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, and Qi Tian. Symbiotic graph neural networks for 3d skeletonbased human action recognition and motion prediction. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 44(6):3316–3333, 2021. 3 

- [40] Maosen Li, Siheng Chen, Zijing Zhang, Lingxi Xie, Qi Tian, and Ya Zhang. Skeleton-parted graph scattering networks for 3d human motion prediction. In _European Conference on Computer Vision_ , 2022. 3, 7 

- [41] Maosen Li, Siheng Chen, Yangheng Zhao, Ya Zhang, Yanfeng Wang, and Qi Tian. Dynamic multiscale graph neural networks for 3d skeleton based human motion prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 214–223, 2020. 3, 7, 14 

- [42] Junwei Liang, Lu Jiang, Kevin Murphy, Ting Yu, and Alexander Hauptmann. The garden of forking paths: Towards multifuture trajectory prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 10508–10518, 2020. 3 

- [43] Ming Liang, Bin Yang, Rui Hu, Yun Chen, Renjie Liao, Song Feng, and Raquel Urtasun. Learning lane graph representations for motion forecasting. In _European Conference on Computer Vision_ , pages 541–556. Springer, 2020. 3 

- [44] Tiezheng Ma, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. Progressively generating better initial guesses towards next stages for high-quality human motion prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6437–6446, 2022. 7 

- [45] Karttikeya Mangalam, Harshayu Girase, Shreyas Agarwal, Kuan-Hui Lee, Ehsan Adeli, Jitendra Malik, and Adrien Gaidon. It is not the journey but the destination: Endpoint conditioned trajectory prediction. In _European Conference on Computer Vision_ , pages 759–776. Springer, 2020. 3, 8 

- [46] Wei Mao, Miaomiao Liu, and Mathieu Salzmann. History repeats itself: Human motion prediction via motion attention. In _European Conference on Computer Vision_ , pages 474–489. Springer, 2020. 3 

- [47] Wei Mao, Miaomiao Liu, Mathieu Salzmann, and Hongdong Li. Learning trajectory dependencies for human motion prediction. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9489–9497, 2019. 3, 7 

- [48] Francesco Marchetti, Federico Becattini, Lorenzo Seidenari, and Alberto Del Bimbo. Mantra: Memory augmented networks for multiple trajectory prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 7143–7152, 2020. 3 

- [49] Diego Marcos, Michele Volpi, Nikos Komodakis, and Devis Tuia. Rotation equivariant vector field networks. In _Proceedings of the IEEE International Conference on Computer Vision_ , pages 5048–5057, 2017. 2 

- [50] Julieta Martinez, Michael J Black, and Javier Romero. On human motion prediction using recurrent neural networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2891–2900, 2017. 3, 7, 14 

- [51] Ramin Mehran, Alexis Oyama, and Mubarak Shah. Abnormal crowd behavior detection using social force model. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 935–942. IEEE, 2009. 3 

- [52] Jeremy Morton, Tim A Wheeler, and Mykel J Kochenderfer. Analysis of recurrent neural networks for probabilistic modeling of driver behavior. _IEEE Transactions on Intelligent Transportation Systems_ , 18(5):1289–1298, 2016. 3 

- [53] Damian Mrowca, Chengxu Zhuang, Elias Wang, Nick Haber, Li F Fei-Fei, Josh Tenenbaum, and Daniel L Yamins. Flexible neural representation for physics prediction. _Advances in neural information processing systems_ , 31, 2018. 2 

- [54] Stefano Pellegrini, Andreas Ess, Konrad Schindler, and Luc Van Gool. You’ll never walk alone: Modeling social behavior for multi-target tracking. In _2009 IEEE 12th International Conference on Computer Vision_ , pages 261–268. IEEE, 2009. 7, 14 

- [55] Tim Salzmann, Boris Ivanovic, Punarjay Chakravarty, and Marco Pavone. Trajectron++: Multi-agent generative trajectory forecasting with heterogeneous data for control. 2020. 3, 8 

- [56] Alvaro Sanchez-Gonzalez, Victor Bapst, Kyle Cranmer, and Peter Battaglia. Hamiltonian graph networks with ode integrators. _arXiv preprint arXiv:1909.12790_ , 2019. 2 

- [57] Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and Peter Battaglia. Learning to simulate complex physics with graph networks. In _International Conference on Machine Learning_ , pages 8459–8468. PMLR, 2020. 2 

- [58] Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In _International conference on machine learning_ , pages 9323–9332. PMLR, 2021. 2, 6, 7, 13 

- [59] Bohan Tang, Yiqi Zhong, Ulrich Neumann, Gang Wang, Ya Zhang, and Siheng Chen. Collaborative uncertainty in multiagent trajectory forecasting. _Advances in Neural Information Processing Systems_ , 34, 2021. 3 

- [60] Graham W Taylor and Geoffrey E Hinton. Factored conditional restricted boltzmann machines for modeling motion style. In _Proceedings of the 26th annual international conference on machine learning_ , pages 1025–1032, 2009. 3 

- [61] Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. _arXiv preprint arXiv:1802.08219_ , 2018. 2, 7 

- [62] Benjamin Ummenhofer, Lukas Prantl, Nils Thuerey, and Vladlen Koltun. Lagrangian fluid simulation with continuous convolutions. In _International Conference on Learning Representations_ , 2019. 2 

- [63] Anirudh Vemula, Katharina Muelling, and Jean Oh. Social attention: Modeling attention in human crowds. In _2018 IEEE international Conference on Robotics and Automation (ICRA)_ , pages 4601–4607. IEEE, 2018. 3 

- [64] Jacob Walker, Kenneth Marino, Abhinav Gupta, and Martial Hebert. The pose knows: Video forecasting by generating pose futures. In _Proceedings of the IEEE international conference on computer vision_ , pages 3332–3341, 2017. 3 

   - [68] Chenxin Xu, Siheng Chen, Maosen Li, and Ya Zhang. Invariant teacher and equivariant student for unsupervised 3d human pose estimation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 3013–3021, 2021. 1 

   - [69] Chenxin Xu, Maosen Li, Zhenyang Ni, Ya Zhang, and Siheng Chen. Groupnet: Multiscale hypergraph neural networks for trajectory prediction with relational reasoning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6498–6507, 2022. 3, 7, 8 

   - [70] Chenxin Xu, Weibo Mao, Wenjun Zhang, and Siheng Chen. Remember intentions: Retrospective-memory-based trajectory prediction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 6488– 6497, 2022. 3, 8 

   - [71] Chenxin Xu, Yuxi Wei, Bohan Tang, Sheng Yin, Ya Zhang, and Siheng Chen. Dynamic-group-aware networks for multiagent trajectory prediction with relational reasoning. _arXiv preprint arXiv:2206.13114_ , 2022. 2, 3 

   - [72] Xingyi Yang, Jingwen Ye, and Xinchao Wang. Factorizing knowledge in neural networks. In _European Conference on Computer Vision_ , 2022. 3 

   - [73] Xingyi Yang, Daquan Zhou, Songhua Liu, Jingwen Ye, and Xinchao Wang. Deep model reassembly. In _Conference on Neural Information Processing Systems_ , 2022. 2 

   - [74] Yiding Yang, Zunlei Feng, Mingli Song, and Xinchao Wang. Factorizable graph convolutional networks. In _Conference on Neural Information Processing Systems_ , 2020. 2 

   - [75] Yiding Yang, Jiayan Qiu, Mingli Song, Dacheng Tao, and Xinchao Wang. Distilling knowledge from graph convolutional networks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2020. 2 

   - [76] Ye Yuan, Xinshuo Weng, Yanglan Ou, and Kris M Kitani. Agentformer: Agent-aware transformers for socio-temporal multi-agent forecasting. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 9813– 9823, 2021. 3, 7, 8, 14 

   - [77] Yiqi Zhong, Zhenyang Ni, Siheng Chen, and Ulrich Neumann. Aware of the history: Trajectory forecasting with the local behavior data. In _Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXII_ , pages 393–409. Springer, 2022. 3 

- [65] Jack M Wang, David J Fleet, and Aaron Hertzmann. Gaussian process dynamical models for human motion. _IEEE transactions on pattern analysis and machine intelligence_ , 30(2):283–298, 2007. 3 

- [66] Maurice Weiler, Fred A Hamprecht, and Martin Storath. Learning steerable filters for rotation equivariant cnns. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 849–858, 2018. 2 

- [67] Daniel E Worrall, Stephan J Garbin, Daniyar Turmukhambetov, and Gabriel J Brostow. Harmonic networks: Deep translation and rotation equivariance. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 5028–5037, 2017. 2 

## **A. Theoretical Proofs** 

In this section, we prove Theorem 1 in our paper which shows EqMotion’s equivariance property and the interaction reasoning module’s invariance property. Note that, here we treat all the vectors to be row vectors since we multiply the rotation matrix by right. 

1. For the initialization layer _F_ IL( _·_ ), the initial geometric feature is equivariant and the initial pattern feature is invariant: 

G[(0)] **R** + **t** _,_ **H**[(0)] = _F_ IL(X **R** + **t** ) _._ 

**Proof:** For the _i_ th agent, we show its initial geometric feature is equivariant to the input motion under Euclidean transformation. When transforming the past motion, we have 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-05.png)


Thus we show the initial geometric feature is equivariant to the input motion under Euclidean transformation. We also show its initial pattern feature is invariant to the input motion under Euclidean transformation. When transforming the past motion, we have, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-08.png)


Thus we show the initial pattern feature is invariant to the input motion under Euclidean transformation. 

2. The reasoning module _F_ IRM( _·_ ) along with reasoned interaction categorical vectors _{_ **c** _ij}_ is invariant: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-11.png)


**Proof:** We first show the column-wise _ℓ_ 2-distance of geometric feature _||_ **G**[(0)] _i −_ **G**[(0)] _j[||]_[2] _[,]_[col][ is invariant since for the] _c_ th column ( _c_ = 1 _, · · · , C_ ), we have _||_ **g** _i,c_[(0)] **[R]**[+] **[t]** _[−]_[(] **[g]** _j,c_[(0)] **[R]**[+] **t** ) _||_ 2 = _||_ **g** _i,c_[(0)] **[R]** _[ −]_ **[g]** _j,c_[(0)] **[R]** _[||]_[2][=] _[||]_ **[g]** _i,c_[(0)] _[−]_ **[g]** _j,c_[(0)] _[||]_[2][.][Since][the] initial pattern feature is invariant, thus we have the edge feature **m** _[′] ij_[, the aggregated edge feature] **[ p]** _i[′]_[and the updated] node feature **h** _[′] i_[all to be invariant.][Finally, we have the in-] teraction categorical **c** _ij_ vector being invariant since **h** _[′] i_[and] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-13.png)


G[(] _[l]_[+1)] **R** + **t** = _F_ EGFL[(] _[ℓ]_[)][(][G][(] _[ℓ]_[)] **[R]**[ +] **[ t]** _[,]_ **[ H]**[(] _[ℓ]_[)] _[,][ {]_ **[c]** _[ij][}]_[)] _[.]_ 

**Proof:** We show the result by indicating the inner-agent attention, inter-agent aggregation and non-linear function all to be equivariant. We first show the inner-agent attention is equivariant. When transforming the input geometric feature, for every _i_ th agent ( _i_ = 1 _,_ 2 _, · · · , M_ ), 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-17.png)


Thus the inner-agent attention is equivariant. We then show the inter-agent aggregation is equivariant. When transforming the input geometric feature, we show the column-wise _ℓ_ 2-distance of geometric feature _||_ **G**[(] _i[ℓ]_[)] _−_ **G**[(] _j[ℓ]_[)] _[||]_[2] _[,]_[col][is][in-] variant since for the _c_ th column ( _c_ = 1 _, · · · , C_ ), we have _||_ **g** _i,c_[(] _[ℓ]_[)] **[R]**[+] **[t]** _[−]_[(] **[g]** _j,c_[(] _[ℓ]_[)] **[R]**[+] **[t]**[)] _[||]_[2][=] _[||]_ **[g]** _i,c_[(] _[ℓ]_[)] **[R]** _[−]_ **[g]** _j,c_[(] _[ℓ]_[)] **[R]** _[||]_[2][=] _||_ **g** _i,c_[(] _[ℓ]_[)] _[−]_ **[g]** _j,c_[(] _[ℓ]_[)] _[||]_[2][.][Thus the learned aggregation weights] **[ e]**[(] _ij[ℓ]_[)] is invariant. We have the inter-agent aggregation’s equivariance, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-20.png)


Thus the inner-agent attention is equivariant. We then show the non-linear function is equivariant. When transforming the input geometric feature, the inner product of the query _⊤_ coordinate and the key coordinate _⟨_ **q**[(] _i,c[ℓ]_[)] _[,]_ **[ k]**[(] _i,c[ℓ]_[)] _[⟩]_[=] **[ q]**[(] _i,c[ℓ]_[)] **[k]**[(] _i,c[ℓ]_[)] is invariant for every channel _c_ = 1 _,_ 2 _, · · · , C_ since 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0012-23.png)


and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-02.png)


Since the criterion is invariant and two equations under two conditions are both equivariant, the non-linear function is equivariant. Finally, combining the equivariance of inneragent attention, inter-agent aggregation and nonlinear function, we show the equivariance of the geometric feature learning layer. 

4. The _ℓ_ th pattern feature learning layer _F_ IPFL[(] _[ℓ]_[)][(] _[·]_[)][ is in-] variant: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-05.png)


**Proof:** Similar with the invariance of reasoning module, we first have the column-wise _ℓ_ 2-distance of geometric feature _||_ **G**[(] _i[ℓ]_[)] _−_ **G**[(] _j[ℓ]_[)] _[||]_[2] _[,]_[col][ is invariant.][Thus we have the variable] in the message passing **m**[(] _ij[ℓ]_[)][,] **[ p]**[(] _i[ℓ]_[)] all invariant. Finally the next layer’s pattern feature **h**[(] _[l]_[+1)] is invariant. 

5. The output layer _F_ EOL( _·_ ) is equivariant: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-08.png)


**Proof:** When transforming the input geometric feature, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-10.png)


Thus the output layer is equivariant. 

## **B. Optional Operations** 

## **B.1. DCT Processing** 

To have a compact representation of input motion data, here we apply an optional discrete cosine transform (DCT) along the time axis to convert the input motion into the frequency domain. Mathematically, for the input motion **X** _i_ of agent _i_ , we transform it by **X** _i ←_ **W** DCT( **X** _i −_ X) where **W** DCT _∈_ R _[T]_[p] _[×][T]_[p] is the DCT coefficients matrix. Correspondingly, we transform the predicted motion by an 

inverse DCT (iDCT) operation: **Y**[�] _i ←_ **W** iDCT **Y**[�] _i_ + X and **W** iDCT _∈_ R _[T]_[f] _[×][T]_[f] is the iDCT coefficients matrix. The remove-and-add operation about the mean location X is to ensure the translation equivariance. Since the DCT process is equivariant, adding this process will maintain whole network’s equivariance. 

## **B.2. Adding Velocity Information** 

We also introduce an optional operation to directly add the velocity information into the geometric feature by 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-18.png)


where _ρi_ is the velocity magnitude sequence and function _· ϕρ_ ( ) is implemented by MLP. Since the velocity magnitude sequence is invariant, thus the operation is equivariant. This operation is placed before the nonlinear function. 

## **C. Modification for Multi-prediction** 

To make EqMotion perform multiple predictions in pedestrian trajectory prediction, we slightly modify the network by using multiple prediction heads in parallel. Each prediction head consists of a feature learning layer and an output layer. Assuming the _i_ th output produced by the _i_ th prediction head is Y[�] _i_ , we use a minimum _ℓ_ 2 prediction loss formulated by, 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_images/cvpr_2023_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning.pdf-0013-22.png)


Through the loss, the optimal prediction will be optimized. 

## **D. Experiment Details** 

## **D.1. Dataset Description** 

## **D.1.1 Particle Dynamics** 

We use the particle N-body simulation environment [28] in a 3-dimensional space similar to [14, 58]. The system contains 5 interacted particles. In the reasoning task, in the Springs simulation, particles will be randomly connected by a spring with a probability of 0.5. The particles connected by springs interact via forces given by Hooke’s law. In the Charged simulation, particles will be randomly charged or uncharged. The charged particles will repel or attract others via Coulomb forces. The probability of positive charged, uncharged and negative charged is 0.25, 0.5, and 0.25. We predicted the future motion of 20 timestamps given the historical observations of 20 timestamps. We use a downsampling rate of 100. We use 5k, 2k and 2k samples for training, validating and testing, respectively. In the prediction task, the setting is similar except the probability of positive charged, uncharged and negative charged is 0.5, 0, and 0.5. 

## **D.1.2 Molecule Dynamics** 

We adopt the MD17 [7] dataset which contains the motions of different molecules generated via a molecular dynamics simulation environment. The goal is to predict the motions of every atom of the molecule. We randomly pick four kinds of molecules: Aspirin, Benzene, Ethanol and Malonaldehyde. We learn a prediction model for each molecule. We predicted the future motion of 10 timestamps given the observation of 10 timestamps. The raw data is a long sequence and we sample the trajectory with a sampling rate of 20 and a sampling gap of 400. We randomly pick 5k, 2k and 2k samples for training, validating and testing. 

Table 8. Effect of different numbers of learning layers on H3.6M. 

|Layers|80ms<br>160ms<br>320ms<br>400ms<br>Average|
|---|---|
|1<br>9.5<br>21.4<br>46.7<br>58.3<br>34.0<br>2<br>9.3<br>20.7<br>45.4<br>56.5<br>33.0<br>3<br>**9.1**<br>20.3<br>44.3<br>55.7<br>32.4<br>**4**<br>**9.1**<br>**20.1**<br>**43.7**<br>**55.0**<br>**32.0**<br>5<br>**9.1**<br>20.2<br>43.9<br>55.2<br>32.1||



for 80 epochs. For long-term motion prediction, we set the number of coordinates in the geometric feature _C_ as 96 and the dimension of the pattern feature _D_ as 64. The predefined category number _L_ is 4. We set the batch size to 100 and use an initial learning rate of 5e-4 with a decay rate of 0.8 for every 2 epochs. The model is trained for 100 epochs. 

## **D.1.3 3D Human Skeleton Motion** 

Human 3.6M (H3.6M) dataset [25] contains 7 subjects performing 15 classes of actions, and each subject has 22 body joints. All sequences are downsampled by two along time. Following previous paradigms [41, 50], the models are trained on the segmented clips in the 6 subjects and tested on the clips in the 5th subject. 

**Pedestrian Trajectories** We set the number of coordinates in the geometric feature _C_ as 64 and the dimension of the pattern feature _D_ as 64. The predefined category number _L_ is 4. We set the batch size to 100 and use an initial learning rate of 8e-4/5e-4/1e-3/5e-4/1e-3 with a decay rate of 0.8/0.8/0.95/0.8/0.9 for every 2/2/2/2/2 epochs on eth/hotel/univ/zara1/zara2 subsets, respectively. The model is trained for 50 epochs. 

## **D.1.4 Pedestrian Trajectories** 

ETH-UCY dataset [34, 54], contains 5 subsets, ETH, HOTEL, UNIV, ZARA1, and ZARA2. In the dataset, pedestrian trajectories are captured at 2.5Hz in multi-agent social scenarios. Following the standard setting [1, 20, 76], we use 3.2 seconds (8 timestamps) to predict the 4.8 seconds (12 timestamps). We use the leave-one-out approach, training on 4 sets and testing on the remaining set. 

## **D.2. Implementation Details** 

In all the experiments, we set the number of feature learning layers _L_ to 4. We use the Adam optimizer to train the model on a single NVIDIA RTX-3090 GPU. All the MLPs have 2 layers with a ReLU activation function. 

**Particle Dynamics** We set the number of coordinates in the geometric feature _C_ as 64 and the dimension of the pattern feature _D_ as 64. The predefined category number _L_ is 2. We set the batch size to 50 and use a learning rate of 5e-4. The model is trained for 200 epochs. 

## **E. Further Experiment Results** 

**Different numbers of layers** Table 8 shows the effect of different numbers of feature learning layers _L_ on the H3.6M dataset. We find that i) initially increasing _L_ leads to better performance as a more comprehensive geometric feature and pattern feature will be learned; and ii) when the number of layers is sufficient, the performance tends to be stable. 

## **F. Limitation and Future Work** 

This work focuses on a generally applicable motion prediction method. In the future, we plan to expand the method by adding specific designs for different tasks to further improve the model performance. We also expect the method can use more types of data to assist prediction, such as images and videos that contain map information. 

**Molecule Dynamics** We set the number of coordinates in the geometric feature _C_ as 64 and the dimension of the pattern feature _D_ as 64. The predefined category number _L_ is 2. We set the batch size to 50 and use a learning rate of 5e-4. The model is trained for 300 epochs. 

**Human Skeleton Motion** For short-term motion prediction, we set the number of coordinates in the geometric feature _C_ as 72 and the dimension of the pattern feature _D_ as 64. The predefined category number _L_ is 4. We set the batch size to 100 and use a learning rate of 5e-4. The model is trained 

Table 9. Comparisons of short-term prediction on Human3.6M. Results at 80ms, 160ms, 320ms, 400ms in the future are shown. 

|Motion|Walking|Eating|Smoking|Discussion|
|---|---|---|---|---|
|millisecond|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|
|Res-sup.<br>Traj-GCN<br>DMGNN<br>MSRGCN<br>PGBIG<br>SPGSN<br>EqMotion(Ours)|29.4<br>50.8<br>76.0<br>81.5<br>12.3<br>23.0<br>39.8<br>46.1<br>17.3<br>30.7<br>54.6<br>65.2<br>12.2<br>22.7<br>38.6<br>45.2<br>10.2<br>19.8<br>34.5<br>40.3<br>10.1<br>19.4<br>34.8<br>41.5<br>**9.0**<br>**17.5**<br>**32.6**<br>**39.2**|16.8<br>30.6<br>56.9<br>68.7<br>8.4<br>16.9<br>33.2<br>40.7<br>11.0<br>21.4<br>36.2<br>43.9<br>8.4<br>17.1<br>33.0<br>40.4<br>7.0<br>15.1<br>30.6<br>38.1<br>7.1<br>14.9<br>30.5<br>37.9<br>**6.3**<br>**13.6**<br>**28.9**<br>**36.5**|23.0<br>42.6<br>70.1<br>82.7<br>7.9<br>16.2<br>31.9<br>38.9<br>9.0<br>17.6<br>32.1<br>40.3<br>8.0<br>16.3<br>31.3<br>38.2<br>6.6<br>14.1<br>28.2<br>34.7<br>6.7<br>13.8<br>28.0<br>34.6<br>**5.5**<br>**11.3**<br>**23.0**<br>**29.3**|32.9<br>61.2<br>90.9<br>96.2<br>12.5<br>27.4<br>58.5<br>71.7<br>17.3<br>34.8<br>61.0<br>69.8<br>12.0<br>26.8<br>57.1<br>69.7<br>10.0<br>23.8<br>53.6<br>66.7<br>10.4<br>23.8<br>53.6<br>67.1<br>**8.2**<br>**18.8**<br>**42.1**<br>**53.9**|
|Motion|Directions|Greeting|Phoning|Posing|
|millisecond|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|
|Res-sup.<br>Traj-GCN<br>DMGNN<br>MSRGCN<br>PGBIG<br>SPGSN<br>EqMotion(Ours)|35.4<br>57.3<br>76.3<br>87.7<br>9.0<br>19.9<br>43.4<br>53.7<br>13.1<br>24.6<br>64.7<br>81.9<br>8.6<br>19.7<br>43.3<br>53.8<br>7.2<br>17.6<br>40.9<br>51.5<br>7.4<br>17.2<br>39.8<br>50.3<br>**6.3**<br>**15.8**<br>**38.9**<br>**50.1**|34.5<br>63.4<br>124.6<br>142.5<br>18.7<br>38.7<br>77.7<br>93.4<br>23.3<br>50.3<br>107.3<br>132.1<br>16.5<br>37.0<br>77.3<br>93.4<br>15.2<br>34.1<br>71.6<br>87.1<br>14.6<br>32.6<br>70.6<br>86.4<br>**12.7**<br>**30.1**<br>**68.3**<br>**85.2**|38.0<br>69.3<br>115.0<br>126.7<br>10.2<br>21.0<br>42.5<br>52.3<br>12.5<br>25.8<br>48.1<br>58.3<br>10.1<br>20.7<br>41.5<br>51.3<br>8.3<br>18.3<br>38.7<br>48.4<br>8.7<br>18.3<br>38.7<br>48.5<br>**7.4**<br>**16.7**<br>**36.9**<br>**47.0**|36.1<br>69.1<br>130.5<br>157.1<br>13.7<br>29.9<br>66.6<br>84.1<br>15.3<br>29.3<br>71.5<br>96.7<br>12.8<br>29.4<br>67.0<br>85.0<br>10.7<br>25.7<br>60.0<br>76.6<br>10.7<br>25.3<br>59.9<br>76.5<br>**8.2**<br>**18.9**<br>**43.4**<br>**57.5**|
|Motion|Purchases|Sitting|Sittingdown|Takingphoto|
|millisecond|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|
|Res-sup.<br>Traj-GCN<br>DMGNN<br>MSRGCN<br>PGBIG<br>SPGSN<br>EqMotion(Ours)|36.3<br>60.3<br>86.5<br>95.9<br>15.6<br>32.8<br>65.7<br>79.3<br>21.4<br>38.7<br>75.7<br>92.7<br>14.8<br>32.4<br>66.1<br>79.6<br>12.5<br>28.7<br>**60.1**<br>**73.3**<br>12.8<br>28.6<br>61.0<br>74.4<br>**11.2**<br>**26.8**<br>60.5<br>75.2|42.6<br>81.4<br>134.7<br>151.8<br>10.6<br>21.9<br>46.3<br>57.9<br>11.9<br>25.1<br>44.6<br>50.2<br>10.5<br>22.0<br>46.3<br>57.8<br>8.8<br>19.2<br>42.4<br>53.8<br>9.3<br>19.4<br>42.3<br>53.6<br>**8.1**<br>**18.0**<br>**41.2**<br>**52.9**|47.3<br>86.0<br>145.8<br>168.9<br>16.1<br>31.1<br>61.5<br>75.5<br>15.0<br>32.9<br>77.1<br>93.0<br>16.1<br>31.6<br>62.5<br>76.8<br>13.9<br>27.9<br>57.4<br>71.5<br>14.2<br>27.7<br>56.8<br>**70.7**<br>**13.0**<br>**26.5**<br>**56.2**<br>**70.7**|26.1<br>47.6<br>81.4<br>94.7<br>9.9<br>20.9<br>45.0<br>56.6<br>13.6<br>29.0<br>46.0<br>58.8<br>9.9<br>21.0<br>44.6<br>56.3<br>8.4<br>18.9<br>42.0<br>53.3<br>8.8<br>18.9<br>**41.5**<br>**52.7**<br>**7.9**<br>**17.7**<br>40.9<br>52.8|
|Motion|Waiting|WalkingDog|WalkingTogether|Average|
|millisecond|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|80<br>160<br>320<br>400|
|Res-sup.<br>Traj-GCN<br>DMGNN<br>MSRGCN<br>PGBIG<br>SPGSN<br>EqMotion(Ours)|30.6<br>57.8<br>106.2<br>121.5<br>11.4<br>24.0<br>50.1<br>61.5<br>12.2<br>24.2<br>59.6<br>77.5<br>10.7<br>23.1<br>48.3<br>59.2<br>8.9<br>20.1<br>43.6<br>54.3<br>9.2<br>19.8<br>43.1<br>54.1<br>**7.6**<br>**17.4**<br>**39.9**<br>**51.1**|64.2<br>102.1<br>141.1<br>164.4<br>23.4<br>46.2<br>83.5<br>96.0<br>47.1<br>93.3<br>160.1<br>171.2<br>20.7<br>42.9<br>80.4<br>93.3<br>18.8<br>39.3<br>73.7<br>86.4<br>17.8<br>37.2<br>**71.7**<br>**84.9**<br>**16.6**<br>**36.4**<br>72.5<br>86.2|26.8<br>50.1<br>80.2<br>92.2<br>10.5<br>21.0<br>38.5<br>45.2<br>14.3<br>26.7<br>50.1<br>63.2<br>10.6<br>20.9<br>37.4<br>43.9<br>8.7<br>18.6<br>34.4<br>41.0<br>8.9<br>18.2<br>33.8<br>40.9<br>**7.8**<br>**16.1**<br>**30.6**<br>**37.1**|34.7<br>62.0<br>101.1<br>115.5<br>12.7<br>26.1<br>52.3<br>63.5<br>17.0<br>33.6<br>65.9<br>79.7<br>12.1<br>25.6<br>51.6<br>62.9<br>10.3<br>22.7<br>47.4<br>58.5<br>10.4<br>22.3<br>47.1<br>58.3<br>**9.1**<br>**20.1**<br>**43.7**<br>**55.0**|



Table 10. Comparisons of long-term prediction on Human3.6M. Results at 560ms and 1000ms in the future are shown. 

|Motion|Walking|Eating|Smoking|Discussion|Directions|Greeting|Phoning|Posing|
|---|---|---|---|---|---|---|---|---|
|millisecond|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|
|Res-Sup.<br>Traj-GCN<br>DMGNN<br>MSRGCN<br>PGBIG<br>SPGSN<br>EqMotion(Ours)|81.7<br>100.7<br>54.1<br>59.8<br>71.4<br>85.8<br>52.7<br>63.0<br>48.1<br>56.4<br>46.9<br>53.6<br>**43.4**<br>**52.8**|79.9<br>100.2<br>53.4<br>77.8<br>58.1<br>86.7<br>52.5<br>77.1<br>51.1<br>76.0<br>49.8<br>73.4<br>**48.4**<br>**73.0**|94.8<br>137.4<br>50.7<br>72.6<br>50.9<br>72.2<br>49.5<br>71.6<br>46.5<br>69.5<br>46.7<br>68.6<br>**41.0**<br>**63.4**|121.3<br>161.7<br>91.6<br>121.5<br>81.9<br>138.3<br>88.6<br>117.6<br>87.1<br>118.2<br>89.7<br>118.6<br>**75.3**<br>**105.6**|110.1<br>152.5<br>71.0<br>101.8<br>102.1<br>135.8<br>71.2<br>100.6<br>**69.3**<br>**100.4**<br>70.1<br>100.5<br>70.4<br>101.3|156.3<br>184.3<br>115.4<br>148.8<br>144.5<br>170.5<br>116.3<br>147.2<br>110.2<br>143.5<br>111.0<br>143.2<br>**108.7**<br>**142.0**|143.9<br>186.8<br>69.2<br>103.1<br>71.3<br>108.4<br>68.3<br>104.4<br>65.9<br>102.7<br>66.7<br>102.5<br>**64.7**<br>**101.0**|165.7<br>236.8<br>114.5<br>173.0<br>125.5<br>188.2<br>116.3<br>174.3<br>106.1<br>164.8<br>110.3<br>165.4<br>**84.9**<br>**139.4**|
|Motion|Purchases|Sitting|SittingDown|TakingPhoto|Waiting|WalkingDog|WalkingTogether|Average|
|millisecond|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|560ms<br>1000ms|
|Res-Sup.<br>Traj-GCN<br>DMGNN<br>MSRGCN<br>PGBIG<br>SPGSN<br>EqMotion(Ours)|119.4<br>176.9<br>102.0<br>143.5<br>104.9<br>146.1<br>101.6<br>139.2<br>95.3<br>**133.3**<br>96.5<br>133.9<br>**93.5**<br>134.5|166.2<br>185.2<br>78.3<br>119.7<br>75.5<br>115.4<br>78.2<br>120.0<br>**74.4**<br>**116.1**<br>75.0<br>116.2<br>74.7<br>116.6|197.1<br>223.6<br>100.0<br>150.2<br>118.0<br>174.1<br>102.8<br>155.5<br>**96.7**<br>**147.8**<br>98.9<br>149.9<br>98.1<br>149.9|107.0<br>162.4<br>77.4<br>119.8<br>78.4<br>123.7<br>77.9<br>121.9<br>**74.3**<br>118.6<br>75.6<br>**118.2**<br>76.7<br>122.0|126.7<br>153.2<br>79.4<br>108.1<br>85.5<br>113.7<br>76.3<br>106.3<br>**72.2**<br>**103.4**<br>73.5<br>103.6<br>71.4<br>104.6|173.6<br>202.3<br>111.9<br>148.9<br>183.2<br>210.2<br>111.9<br>148.2<br>104.7<br>139.8<br>**102.4**<br>**138.0**<br>104.8<br>141.2|94.5<br>110.5<br>55.0<br>65.6<br>70.5<br>86.9<br>52.9<br>65.9<br>51.9<br>64.3<br>49.8<br>60.9<br>**44.5**<br>**56.0**|129.2<br>165.0<br>81.6<br>114.3<br>93.6<br>127.6<br>81.1<br>114.2<br>76.9<br>110.3<br>77.4<br>109.6<br>**73.4**<br>**106.9**|



