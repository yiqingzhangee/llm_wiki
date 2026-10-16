## **Bridge the Points: Graph-based Few-shot Segment Anything Semantically** 

**Anqi Zhang[1] , Guangyu Gao[1] , Jianbo Jiao** _[∗]_ **[2] , Chi Harold Liu[1] , and Yunchao Wei[3]** 

1School of Computer Science, Beijing Institute of Technology 

2The MIx group, School of Computer Science, University of Birmingham 

3WEI Lab, Institute of Information Science, Beijing Jiaotong University `andy_zaq@outlook.com` 

## **Abstract** 

The recent advancements in large-scale pre-training techniques have significantly enhanced the capabilities of vision foundation models, notably the Segment Anything Model (SAM), which can generate precise masks based on point and box prompts. Recent studies extend SAM to Few-shot Semantic Segmentation (FSS), focusing on prompt generation for SAM-based automatic semantic segmentation. However, these methods struggle with selecting suitable prompts, require specific hyperparameter settings for different scenarios, and experience prolonged one-shot inference time due to the overuse of SAM, resulting in low efficiency and limited automation ability. To address these issues, we propose a simple yet effective approach based on graph analysis. In particular, a Positive-Negative Alignment module dynamically selects the point prompts for generating masks, especially uncovering the potential of the background context as the negative reference. Another subsequent Point-Mask Clustering module aligns the granularity of masks and selected points as a directed graph, based on mask coverage over points. These points are then aggregated by decomposing the weakly connected components of the directed graph in an efficient manner, constructing distinct natural clusters. Finally, the positive and overshooting gating, benefiting from graph-based granularity alignment, aggregate high-confident masks and filter out the false-positive masks for final prediction, without relying on additional hyperparameters and redundant mask generation. Extensive experimental analysis across tasks including the standard FSS, One-shot Part Segmentation, and Cross Domain FSS validate the effectiveness and efficiency of the proposed approach, surpassing state-of-the-art generalist models with a mIoU of 58.7% on COCO-20[i] and 35.2% on LVIS-92[i] . The project page of this work is: https://andyzaq.github.io/GF-SAM/. 

## **1 Introduction** 

Previous semantic segmentation methods [1–8], which rely on the pixel-level classification, often struggle with generalization and overfitting due to limited labeled data. In addition, recent approaches, such as MaskFormer [9], have shifted the paradigm to mask-based classification, offering a more flexible approach to improving the segmentation performance by exploiting the consistency and completeness of generated class-agnostic masks. The Segment Anything Model (SAM) [10] further marks a significant advancement by utilizing extensive pre-training on huge-scale dataset SA-1B to achieve more robust, class-agnostic segmentation capabilities. SAM excels in producing precise masks across various domains using simple prompts such as points, boxes, and coarse masks. While the boundaries of these masks can closely align with object boundaries, the lack of semantic 

> _∗_ Corresponding Author. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Pascal 5 [i]<br>70<br>72.1<br>60 52.7 945M58.7 617M58.0 55.5 53.7 COCO 20 [i] 58.7 47.1iSAID 5 [i]<br>945M Ours 399M 180M<br>50 Matcher FSS 1000 88.0 48.7 ISIC<br>40<br>30 23.5 23.0 LVIS 92 [i] 35.2 49.5Deepglobe<br>PerSAM-F PerSAM<br>641M 641M<br>20 36.3 44.5<br>PACO Part PASCAL Part<br>10 [2] 10 [3] Ours<br>Inference Speed (10 3fps) Previous state-of-the-art generalist modelsPrevious state-of-the-art specialist models<br>i20 mIoU<br>COCO<br>**----- End of picture text -----**<br>


(a) Performance-efficiency comparison of FSS models. The (b) Comparison with previous generalist and numbers inside the points represent the numbers of parameters. specialist models on various FSS datasets. 

Figure 1: Performance comparisons of our approach against previous state-of-the-art methods regarding efficiency and generalized capabilities in Few-shot Semantic Segmentation. Figure 1(a) illustrates our approach’s superior performance in efficiency and effectiveness across various model sizes. Figure 1(b) demonstrates the generalizability of our approach across different domains. 

understanding and the requirement for manual prompts prevent SAM from being used in automatic semantic segmentation applications. 

Recent studies have attempted to automate this process in the Few-shot Semantic Segmentation (FSS), using a few reference images and a fine-grained external backbone network ( _e.g._ , DINOv2 [11]) to guide SAM in segmenting target semantic objects. However, these methods face two main challenges: achieving suitable points for precise and full coverage of the target object, and handling the ambiguity of SAM-generated masks, from partial to complete coverage. Specifically, they either utilize the most similar candidate point prompts for iterative mask generation and refinement [12], or build a restrictively selected set of point prompts for heuristically weighted mask merging based on manually designed metrics [13], outperforming both previous specialist methods [14–19] and generalist methods without SAM [20, 21]. However, these methods overlooked the underlying relationships between points (derived from fine-grained features) and masks (generated by SAM in a coarse-grained manner). This oversight led to low efficiency (as indicated in Fig. 1(a)) and limited automation capabilities. Alignment between these two types of granularity could uncover the potential of simple decision-making on masks, which can eliminate redundant refinement and manual hyperparameter selection for complicated metrics. 

In this paper, we explicitly explore the relationship between point prompts and corresponding masks from SAM, and present a simple yet effective parameter-free framework with only one-time mask generation to segment anything semantically, in a graph-based few-shot manner. We first introduce a Positive-Negative Alignment (PNA) module to dynamically select point prompts using foreground and background references from reference images. Unlike existing methods, our approach combines different granularity by constructing a directed graph according to mask coverage over points. Then, we perform connectivity analysis on the constructed graph to obtain several weakly connected components as automatic clustering of point prompts, which bridges points and masks as well as fine-grained and coarse-grained features. To mitigate the inevitable introduction of false positives in the PNA module, we further leverage weakly connected component clusters and limited semantic information in selected points, to more accurately filter and merge masks that mismatch in different granularities. In particular, our proposed method involves two post-gating based on weakly connected clusters: the positive gating retains masks capturing a greater proportion of potential target areas, while the overshooting gating screens out outlier points near object boundary, with coverage self-consistency consideration. 

Extensive experimental analysis on Few-shot Semantic Segmentation demonstrates both the efficiency and effectiveness of our approach, as shown in Fig. 1(b). We first conduct the experiments on generalized FSS datasets, including Pascal-5[i] [22], COCO-20[i] [23], FSS-1000 [24] and LVIS-92[i] [13]. Our approach surpasses existing state-of-the-art approaches on these datasets, with 5 _._ 8% and 2 _._ 2% 

2 

of improvement respectively on more challenging COCO-20[i] and LVIS-92[i] . As for the challenging One-shot Part Segmentation, our approach still exceeds previous methods with 1 _._ 6% of mIoU on both PACO-Part and PASCAL-Part. Furthermore, to demonstrate the ability of our approach across different domains, we perform an evaluation on several specific datasets, including Deepglobe [25], ISIC [26], and iSAID-5[i] [27]. The proposed approach establishes new state-of-the-art performance on mIoU with 49 _._ 5% on Deepglobe, 48 _._ 7% on ISIC, and 47 _._ 3% on iSAID-5[i] . 

Overall, our contributions are summarized as follows: 

- We present, to our knowledge, the first graph-based approach for SAM-based few-shot semantic segmentation, modeling the relationship of SAM-generated masks in an automatic clustering manner. 

- We propose a positive-negative alignment module and a post-gating strategy based on the weakly connected graph components, enabling a hyperparameter-free pipeline. 

- Extensive experimental comparisons and analysis across several datasets over various settings show the effectiveness and efficiency of the proposed method. 

## **2 Related Work** 

**Few-shot semantic segmentation.** Few-shot Semantic Segmentation (FSS) [22] aims to segment the target object using only a limited number of annotated reference samples for guidance. Previous FSS methods are mainly categorized into two types, namely the methods based on prototype matching [28– 34] and methods based on pixel-wise matching [35–40]. The methods based on prototype matching, _e.g._ PFENet [31], BAM [41], SSP [42], use the Mask Average Pooling operation from SGOne [43] to generate a prototype as a global representation of the reference features, and compare the target features with the prototypes. The methods based on pixel-wise matching compute the correlation of all pixels between target and reference features. Then different methods address the correlations through distinct mechanisms, such as 4D Convolution ( _e.g._ , HSNet [14]) and Transformer ( _e.g._ , HDMNet [44], AMFormer [45]). Although these specialist models perform significantly on specific tasks, they are prone to overfitting the training samples and often struggle to adapt to domain shifts. 

**SAM-based semantic segmentation.** Recently, Segment Anything Model (SAM) [10] has shown remarkable zero-shot class-agnostic segmentation capabilities using prompts like points, boxes, and coarse masks. However, the coarse-grained feature representation of SAM limits its effectiveness for fine-grained semantic segmentation tasks. Several approaches have been proposed to extend SAM for semantic segmentation. For example, Semantic-SAM [46] jointly train the model on SA-1B and other semantic aware segmentation datasets to enhance granularity. OV-SAM [47] combines SAM and CLIP [48] for open-vocabulary semantic segmentation. Moreover, some methods introduce SAM into FSS tasks. PerSAM and PerSAM-F [12] leverage SAM for personalized segmentation with one-shot guidance. Matcher [13] uses a SAM-based training-free structure, achieving impressive performance in both FSS and One-shot Part Segmentation. VRP-SAM [49] trains an external Visual Reference Prompt Encoder to automatically generate prompts from reference images using points, scribble, box, or masks. However, previous training-free methods struggled to balance performance and efficiency, often relying on excessive external manual hyperparameters. 

## **3 Preliminaries** 

Few-shot Semantic Segmentation (FSS) aims to segment target objects in an image with a few annotated reference images. Consider a scenario where each group of samples contains a target image _x[t]_ and a reference image set _R_ = _{x[r] k[, y] k[r][}][K] k_ =1[with][the][size][of] _[H][×][ W]_[,][where] _[x][r] k_[and] _[y] k[r]_[mean] the _kth_ reference image and its corresponding mask. Focusing on the 1-shot case, where _K_ = 1, it begins with a feature extraction backbone network _fB_ ( _·_ ), which encodes both _x[t]_ and _x[r]_ into semantic features _F[t]_ and _F[r]_ in R _[hw][×][c]_ , where _h_ and _w_ denote the height and width of the feature maps, and _c_ is the feature dimension. Subsequent few-shot processes utilize these feature maps to generate a predicted segmentation ˜ _y ∈_ R _[H][×][W]_ for _x[t]_ . This prediction is then compared to the Ground Truth (GT) _y[t]_ for evaluation. 

The Segment Anything Model (SAM) is a generalized foundation segmentation model adept at generating precise masks based on varied prompts of points, boxes, and coarse masks. Built around 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Reference Image  𝑥 [�]<br>Reference Feature  𝐹 [�]<br>𝑦 [�] 𝑆����<br>Backbone CorrelationPixel-wise Squeeze Alignment &Comparison<br>𝐶<br>𝑆�����<br>Target Feature  𝐹 [�] 𝑦 [�̅] Positive-Negative Alignment<br>Graph of Coverage  𝐺<br>Target Image  𝑥 [�] on Point Prompts ●<br>Positive<br>Gating<br>● Reserved Points Weakly<br>● Removed Points Connected  Selected<br>Components Point Prompts ●<br>Overshooting<br>Prediction Gating Clusters Mask<br>●●●●●●●●● Masks  from  Generator<br>Point Prompts ●<br>Post Gating Point-Mask Clustering<br>**----- End of picture text -----**<br>


Figure 2: Overview of our approach, where the Positive-Negative Alignment module recognizes the correlation between target features and reference features for point selection, the Point-Mask Clustering module efficiently clusters the points based on the coverage of corresponding masks, and Post-Gating filters out the false-positive masks for generating final prediction. 

a core architecture that includes an image encoder, a prompt encoder, and a mask decoder, SAM effectively processes input images _x[t]_ and prompts _P_ to produce detailed segmentation masks _y_ ˆ. These masks accurately delineate specific objects or regions within the images, based on the guidance provided by the prompts. 

## **4 Method** 

Diverging from traditional methods, we use a directed graph to exploit the natural relationships between points and their corresponding masks, representing fine-grained and coarse-grained features, respectively. As shown in Fig. 2, our approach mainly comprises the Positive-Negative Alignment (PNA) module, Point-Mask Clustering (PMC) module, and Post-Gating strategy. The PNA module leverages semantic features from the backbone network to sort pixel-wise correlations into similarity maps, enabling precise point selection. The PMC module then clusters masks based on these selected points, while Post-Gating strategy refines the selection, enhancing the accuracy and reliability of the final prediction. 

## **4.1 Positive-Negative Alignment for Point Selection** 

The PNA module efficiently selects point prompts to balance the number of points and coverage of target objects. Using the semantic features _F[r]_ and _F[t]_ from the reference and target images respectively (with _e.g._ , DINOv2 [11]), we get the pixel-wise correlation matrix _C ∈_ R _[hw][×][hw]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0004-07.png)


where _C_ ( _i, j_ ) represents the similarity between the _i_ -th pixel of target features _F[t]_ ( _i_ ) and the _j_ -th pixel of reference features _F[r]_ ( _j_ ). 

To minimize hyperparameter reliance, we leverage background features typically overlooked in FSS, indicated by the negative mask _y[r]_[˜] = _¬y[r]_ of the reference image. According to _y[r]_ and _y[r]_[˜] , we divide _C_ into _C_[+] and _C[−]_ in R _[hw][×][hw]_ for foreground and background features, respectively. We then introduce two positive similarity maps in mean and max aspects respectively: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0004-10.png)


4 

where _I_ resizes _y[r]_ to the same resolution as _F[r]_ and then flatten it into a vector, _max_ ( _·_ ) finds the maximum value in the _i_ -th row of _C_[+] . The mean positive similarity map _Smean_[+] _[∈]_[R] _[hw]_[captures] global similarity towards the reference object but may blur distinct internal features, reducing accuracy for complex objects. In contrast, the max positive similarity map _Smax_[+] _[∈]_[R] _[hw]_[focuses on the most] similar regions, enhancing recall but also increasing noise. To maintain distinctiveness while reducing noise, we introduce the mixture similarity map _Smix_[+][=] _[ S] mean_[+] _[⊙][S] max_[+][using the Hadamard product.] This method boosts target region distinctiveness by merging the strengths of both maps, while diminishing noise through the more stable global similarity. 

To select prompt points, we also use the mean negative similarity map _Smean[−]_[,][which][reflects] background similarity, noting that similar objects typically share higher background similarity values. We then align _Smix_[+][and] _[ S] mean[−]_[by min-max normalization] _[ M]_[ to get:] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0005-02.png)


where _Sp ∈_ R _[hw]_ is the filtered map for point selection, and **1** _{·}_ is 1 if the condition is true and 0 otherwise. Although we minimize false negatives, noise points remain. To select suitable points from _Sp_ without hyperparameters, we define the sum of elements in _Sp_ as the number _N_ of points to be selected. We then pick the _N_ highest-value points from _Sp_ as the point prompt set _**P**_ = _{Pl}[N] l_ =1[.] 

## **4.2 Point-Mask Clustering with Graph Connectivity** 

We utilize point prompts from _**P**_ to generate masks with SAM. Each point _Pl_ in _**P**_ corresponds to a unique mask _y_ ˆ _l ∈_ R _[H][×][W]_ . As our point selection strategy prioritizes the coverage of objects, false-negative masks are unavoidable. Moreover, mask coverage can vary significantly within the same region, ranging from partial to full object coverage. This necessitates understanding the internal relationships among coarse-grained masks and points from fine-grained feature comparison to ensure those covering the same target are accurately gathered. 

To address this, we design the Point-Mask Clustering (PMC) module, which clusters points and their corresponding masks based on mask coverage over points. Following the principles of efficiency and automation, the PMC module is based on a directed graph _G_ = ( _V, E_ ) with the vertex _vl_ in _V_ representing point _Pl_ and its corresponding mask _y_ ˆ _l_ . Edges in _E_ are established based on mask coverage over other points; an edge _el,m_ exists if mask ˆ _yl_ covers points _Pm_ (with _m ̸_ = _l_ ). Specifically, we do not establish edges for masks covering their corresponding points to avoid creating loops. 

The graph _G_ is a directed simple graph, allowing us to cluster vertices by identifying weakly connected components. This clustering process is hyperparameter-free, ensuring that every pair of vertices _u, v ∈ V_ within the same component has a directed path between them. Each weakly connected component encompasses a set of points _P_[ˆ] _p_ (with _P_ = _{P_[ˆ] _p}_ ) that are all covered by the union of their masks in _M_[ˆ] _p_ , where _p_ indexes the clusters. 

The advanced SAM plays a crucial role in maintaining the precision of the generated masks. The precision of high-quality masks typically ensures non-overlapping between masks and prompting points of adjacent regions, especially those of different categories. This is the precondition for the efficacy of our PMC module, as even slight errors could significantly impact the clustering accuracy. 

## **4.3 Post-Gating on Weakly Connected Components** 

Our PNA module, while efficient in selecting points, inadvertently includes false positives, as detailed in Sec. 4.1. To mitigate this, we further develop two gating strategies targeting distinct types of false positives based on clusters formed from weakly connected components. 

**Positive gating.** Despite the method in Sec. 4.1 diminishing the noise points outside the target region, there are still a few remaining noise points. These issues may have minimal impact on traditional segmentation methods, but under the SAM framework, masks derived from these noise points can significantly degrade accuracy. Moreover, some clusters of masks may extend beyond their intended target regions due to inaccuracies in SAM-generated masks or because the targeted object is part of a larger entity. Thus, we propose a Positive Gating strategy to address these issues. 

This strategy prioritizes mask effectiveness by assessing whether a mask contains more positive than negative pixels, thereby facilitating a specialized designed mask growth for final prediction. The 

5 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0006-00.png)


**----- Start of picture text -----**<br>
Point Prompt Clusters from  Points Indicated by their  ●Points Satisfying Self-Consistency ●Both Self-Consistency and Positive Gating<br>Weakly Connected Components Most Similar Clusters (Outer Ring) ●Points Failing Self-Consistency ●Either Self-Consistency or Positive Gating<br>●●●●●●●●● ●●●●●●●●● ●Neither Self-Consistency nor Positive Gating<br>**----- End of picture text -----**<br>


Figure 3: Illustration of the Overshooting Gating strategy. The outer ring of points in the second image indicates the most similar cluster of corresponding points, _i.e._ , points with different outside and inside colors do not satisfy the self-consistency. 

focus of mask growth is to enhance coverage of the target area rather than multiple objects, while minimizing background inclusion. Firstly, this method utilizes a parameter-free gating mechanism that discriminates between pixel polarities, based on the positive and negative similarity maps, _Smean_[+] and _Smean[−]_[, as described in Sec.][ 4.1][.][To achieve this, we utilize] _[ S] mean_[+][and] _[ S] mean[−]_[, along with the] median of _Smean_[+][(] _[i.e.]_[,][the][midpoint][between][the][maximum][and][minimum][values][of] _[S] mean_[+][),][to] constructs the polarity map _R_[¯] as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0006-03.png)


Then, using the polarity map _R_[¯] , we calculate the number of positive pixels of the _l[th]_ mask as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0006-05.png)


where _I_ resizes and flattens _y_ ˆ _l_ to the feature map dimensions. Subsequently, for each cluster _M_[ˆ] _p_ of weakly connected components, we sort the masks according to the ratio of positive pixel numbers to their areas. The indices of these sorted masks are denoted by _Q_ . We then initialize a blank pseudo mask _y_ ¨ _p ∈_ R _[H][×][W]_ and a set of positive points _P_[+] . Following this, we apply a Mask Growth algorithm as outlined in Sec. A.1 and Alg. 1 for maintaining positive masks. This algorithm iteratively evaluates whether the region of _y_ ˆ _q_ outside the pseudo mask _y_ ¨ _p_ is positive, updates _y_ ¨ _p_ with the identified positive mask, and adds its corresponding point into _P_[+] . 

_·_ **Overshooting gating.** The fine-grained semantic features from _f_ ( ) are reliable for locating target objects, yet the point coverage of the target areas varies, leading to both under-coverage and overcoverage. SAM effectively addresses under-coverage; however, over-coverage, which extends beyond target boundaries, often produces false-positive masks. These overshooting points, while semantically similar to the target areas in _F[t]_ , typically derive masks that cover areas outside the target, resulting in a mismatch of representations between the granularity of points and masks. Thus, these points cannot be clustered with points inside the target areas. 

Hence, we devise an overshooting gating strategy with consideration of self-consistency to eliminate overshooting points and their associated masks. As shown in Fig. 3, We assess the similarity between the features of each point _Pl_ and the union mask _y_ ˆ _p ∈_ R _[H][×][W]_ from each mask cluster _M_[ˆ] _p_ . The similarity computation for estimating **s** elf- **c** onsistency is performed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0006-09.png)


where _Sim_ ( _·, ·_ ) refers to the correlation calculation mentioned in Eq. 1. We introduce an external function _dist_ ( _·, ·_ ) to measure the distance in _F[t]_ between each point _Pl_ and the nearest selected point in _P_[ˆ] _p_ . This measure helps confine comparison to neighboring clusters, minimizing interference from other instances. We then identify the cluster most similar to the points and retain those in the point set _P[sc]_ that are more similar to their respective clusters. 

6 

Table 1: Performance on Few-shot Semantic Segmentation datasets of Pascal-5[i] , COCO-20[i] , FSS1000, and LVIS-92[i] . Gray means the in-domain trained results. The best results are shown in **bold** . 

|Methods|Pascal-5i<br>1-shot<br>5-shot|COCO-20i<br>1-shot<br>5-shot|FSS-1000<br>1-shot<br>5-shot|LVIS-92i<br>1-shot<br>5-shot|
|---|---|---|---|---|
||||||
|_specialist model_<br>HSNet [14][CVPR21]<br>VAT [50][ECCV22]<br>HDMNet [44][CVPR23]<br>AMFormer [45][NeurIPS23]|66.2<br>70.4<br>67.9<br>72.0<br>69.4<br>71.8<br>70.7<br>73.6|41.2<br>49.5<br>41.3<br>47.9<br>50.0<br>56.0<br>51.0<br>57.3|86.5<br>88.5<br>90.3<br>90.8<br>-<br>-<br>-<br>-|17.4<br>22.9<br>18.5<br>22.7<br>-<br>-<br>-<br>-|
||||||
|_generalist model_<br>PerSAM [12][ICLR24]<br>PerSAM-F [12][ICLR24]<br>Matcher [13][ICLR24]<br>VRP-SAM [49][CVPR24]<br>Ours|43.1<br>-<br>48.5<br>-<br>68.1<br>74.0<br>71.9<br>-<br>**72.1**<br>**82.6**|23.0<br>-<br>23.5<br>-<br>52.7<br>60.7<br>53.9<br>-<br>**58.7**<br>**66.8**|71.2<br>-<br>75.6<br>-<br>87.0<br>**89.6**<br>-<br>-<br>**88.0**<br>88.9|11.5<br>-<br>12.3<br>-<br>33.0<br>40.0<br>-<br>-<br>**35.2**<br>**44.2**|



Table 2: Performance on One-shot Part Segmentation datasets and Cross Domain Few-shot Semantic Segmentation datasets. The best results are shown in **bold** . 

|Methods|One-shot Part Seg.<br>PASCAL-Part<br>PACO-Part<br>1-shot<br>1-shot|One-shot Part Seg.<br>PASCAL-Part<br>PACO-Part<br>1-shot<br>1-shot|Cross Domain FSS<br>Deepglobe<br>ISIC<br>iSAID-5i<br>1-shot<br>5-shot<br>1-shot<br>5-shot<br>1-shot<br>5-shot|Cross Domain FSS<br>Deepglobe<br>ISIC<br>iSAID-5i<br>1-shot<br>5-shot<br>1-shot<br>5-shot<br>1-shot<br>5-shot|Cross Domain FSS<br>Deepglobe<br>ISIC<br>iSAID-5i<br>1-shot<br>5-shot<br>1-shot<br>5-shot<br>1-shot<br>5-shot|
|---|---|---|---|---|---|
|_specialist model_<br>HSNet [14][CVPR21]<br>DRA [51][CVPR24]<br>FRINet [52][TGRS23]|32.4<br>-<br>-|22.6<br>-<br>-|29.7<br>35.1<br>41.3<br>50.1<br>-<br>-|31.2<br>35.1<br>40.8<br>48.9<br>-<br>-|34.1<br>40.4<br>-<br>-<br>42.6<br>44.5|
|_generalist model_<br>PerSAM [12][ICLR24]<br>PerSAM-F [12][ICLR24]<br>Matcher [13][ICLR24]<br>Ours|32.5<br>32.9<br>42.9<br>**44.5**|22.5<br>22.7<br>34.7<br>**36.3**|31.4<br>-<br>35.0<br>-<br>48.1<br>50.9<br>**49.5**<br>**57.7**|23.9<br>-<br>23.6<br>-<br>38.6<br>35.0<br>**48.7**<br>**55.2**|19.2<br>-<br>20.3<br>-<br>33.3<br>34.3<br>**47.1**<br>**52.4**|



**Mask Merging.** Finally, we obtain two distinct sets of points, namely _P_[+] and _P[sc]_ . We then union the masks corresponding to points that are common to both _P_[+] and _P[sc]_ . The merged masks form ˜ the final prediction, denoted as _y_ . 

## **5 Experimental Results** 

## **5.1 Datasets** 

To illustrate the Few-shot Semantic Segmentation ability and generalization capacity, we conduct three types of sub-tasks, _i.e._ standard Few-shot Semantic Segmentation, One-shot Part Segmentation, and Cross Domain Few-shot Semantic Segmentation. The datasets for these tasks are as follows: 

**Pascal-5[i]** , **COCO-20[i]** , **FSS-1000** , and **LVIS-92[i]** are standard FSS datasets. Pascal-5[i] [22] is based on the Pascal VOC 2012 [53] and SDS [54]. The 20 classes are separated into 4 folds of 5 classes. COCO20[i] [23] is an 80-class dataset from MSCOCO [55], which has 4 folds with each fold containing 20 classes. FSS-1000 [24] contains 1000 classes. The training, validation, and testing folds contain 520, 240, and 240 classes, respectively. LVIS-92[i] [13] is more challenging for evaluating generalist models, which select 920 classes with more than 2 images and divide these classes into 10 folds. 

**PASCAL-Part** and **PACO-Part** [13] are One-shot Part Segmentation datasets. PASCAL-Part [56, 57] contains 56 different object parts in 4 superclasses. PACO-Part is built based on the PACO dataset [58], which has 456 object part classes. The 303 classes with at least 2 samples in PACO-Part are divided into four folds following Matcher [13]. 

**Deepglobe** , **ISIC2018** , and **iSAID-5[i]** are Cross Domain FSS datasets. The Deepglobe [25] contains satellite images of geographic categories including urban, agriculture, rangeland, forest, water, and 

7 

Table 3: Ablation study of Point Selection. 

|_S_+<br>_mean_|_S_+<br>_max_|_S−_<br>_mean_|Top_N_|mIoU|
|---|---|---|---|---|
|✓||✓|✓|53.1|
||✓|✓|✓|54.1|
|✓|✓|✓||56.4|
|✓|✓||✓|51.5|
|✓|✓|✓|✓|**58.7**|



Table 4: Ablation study of PMC and Post-Gating. 

||||
|---|---|---|
|PG<br>Strong<br>Weak|OG<br>Strong<br>Weak|COCO-20i<br>LVIS-92i|
|44.0<br>24.2<br>✓<br>57.1<br>34.3<br>✓<br>57.1<br>33.9<br>✓<br>✓<br>56.7<br>**35.2**<br>✓<br>✓<br>**58.7**<br>**35.2**|||
|k-means++||57.5<br>34.0|



Table 5: Ablation study of positive gating on each cluster. M.G. represents the Mask Growth algorithm. 

|Strategies<br>M.G.|COCO-20i<br>PASCAL-Part|
|---|---|
|||
|Sum<br>✓|55.3<br>39.1<br>58.6<br>44.3|
|||
|Num<br>✓|57.1<br>42.2<br>**58.7**<br>**44.5**|



Table 6: Ablation on the strategies of SelfConsistency measurement. 

|nsistencymeasurement.|nsistencymeasurement.|
|---|---|
|Strategies<br>mIoU<br>∆||
|None<br>Point Sim.<br>MAP Sim.<br>Mean Sim. W/o dist<br>Mean Sim. (Ours)|57.1<br>0.0<br>56.7<br>-0.4<br>57.7<br>+0.6<br>49.1<br>-8.0<br>**58.7**<br>**+1.6**|



barren. The ISIC2018 [26] is a skin lesion analysis dataset with three classes. The iSAID-5[i] [27] evenly split 3 folds for 15 classes based on the remote sensing dataset iSAID [59]. 

## **5.2 Implementation Details** 

Following the settings of PerSAM [12] and Matcher [13] for a fair comparison, we use DINOv2 [11] with a ViT-L/14 [60] as our feature extraction backbone, and SAM [10] with ViT-H as the mask generator. The input image sizes are set to 518 _×_ 518 for DINOv2 and 1024 _×_ 1024 for SAM following Matcher [13]. Except for the default hyperparameters of SAM and DINOv2, our approach **does not have any external hyperparameter** . We apply the mean Intersection over Union (mIoU) metric for evaluating the performance. All experiments are conducted on a single NVIDIA RTX2080Ti. 

## **5.3 Comparison with State-Of-The-Arts** 

**Comparison on the standard FSS datasets.** We compared our approach with other state-of-the-art specialist and generalist models. As shown in Tab. 1, our approach achieves 72.1% mIoU on the Pascal-5[i] dataset and 58.7% mIoU on COCO-20[i] dataset, which surpasses all previous specialist and generalist state-of-the-art models. Our approach reaches 35.2% mIoU in the more challenging dataset of LVIS-92[i] , with 2.2% of improvement compared to the previous training-free method Matcher. The performance remains competitive on the FSS-1000 compared with specialist models. The 5-shot result of Pascal-5[i] , COCO-20[i] , and LVIS-92[i] further extends the lead, which shows that our approach can effectively handle the few-shot scenario. 

**Comparison on the One-shot Part Segmentation datasets.** The One-shot Part Segmentation tasks evaluate the ability to fetch the target part from the whole object. The results in Tab. 2 show that our approach achieves the mIoU of 44.5% and 36.3% on both datasets of PASCAL-Part and PACO-Part, respectively. Our approach outperforms the state-of-the-art generalist model Matcher with 1.6% on both datasets. Given that Matcher employs **specific hyperparameters** to enhance part segmentation, our superior performance demonstrates the adaptability of our approach across both object and part segmentation contexts. 

**Comparison on the Cross Domain FSS datasets.** The Cross Domain FSS tasks validate the performance on different domains. Our approach achieves state-of-the-art performance in datasets of Deepglobe, ISIC, and iSAID-5[i] among other specialist domain models and generalist models. Especially within the context of the skin lesion analysis dataset ISIC and remote sensing dataset iSAID-5[i] , our approach outperforms Matcher by margins of 10.1% and 13.8% respectively. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Ref. Image  Tar. Image  Matcher  Baseline  B+PG  B+PG+OG(Ours)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0009-24.png)


Figure 4: Qualitative analysis of Matcher, Baseline, B+PG, B+PG+OG. B, PG, and OG respectively represent Baseline, Positive Gating, and Overshooting Gating. Masks in ref. image are shown in blue. 

## **5.4 Ablation Study** 

**Point selection.** We evaluate the impact of various similarity maps and the parameter-free selection of top N points on performance, as detailed in Sec. 4.1. As shown in Tab. 3, using either _Smean_[+][or] _Smax_[+][alone leads to a performance drop of up to][ 5] _[.]_[6%][ compared to using both.][This decline is due] to the inherent limitations of _Smean_[+][and] _[ S] max_[+][discussed in Sec][ 4.1][.][Additionally, the evaluation] confirms that picking the top-N points based on similarity, which is parameter-free and requires no additional settings, simplifies the process and increases accuracy by 2 _._ 1%. 

**Clustering method.** We compare our PMC module using weakly connected components with the PMC module using strong connected components, which provides finer clustering results. According to our experiment results in Tab. 4, the clusters from weakly connected components provide better performance on COCO-20[i] and LVIS-92[i] for both gating, as these clusters of masks have ideal coverage of the objects. Simply filtering the masks without clustering-based gating can only achieve 44.0% mIoU on COCO-20[i] and 24.2% mIoU on LVIS-92[i] , which is significantly lower than the performance achieved with clustering-based gating. Furthermore, our dynamic hyperparameter-free clustering method outperforms the k-means++ with 1.2% on both datasets. Note that k of k-means++ is set to 10 following Matcher [13]. 

**Positive** ˆ **gating.** Our approach compares the number of positive points and negative points in _S_[+] (Num) to judge whether the mask is positive. We conduct experiments for the strategy of comparing the sum of positive and negative values (Sum). The results in Tab. 5 demonstrate the Num strategy yields better performance, as comparing the number of pixels mitigates the influence of a few excessively high similarity values. Furthermore, the utilization of the Mask Growth algorithm improves both FSS and Part Segmentation performance by carefully retaining the positive regions. However, it weakens the improvement of Num due to their similar effects. 

**Overshooting gating.** Our Overshooting gating aims to filter out the overshooting points closely neighboring to target regions, thus having a less remarkable improvement of 1.6% compared to Positive Gating, as shown in Tab. 6. This performance still surpasses the mean similarity of com- 

9 

paring points with regions of clustered points (Point Sim.) or the prototypes from Masked Average Pooling [43] with union masks (MAP Sim.). More importantly, the distance function avoids the gating from 9.6% of performance decline. It ensures each cluster only affects its neighboring points. 

## **5.5 Qualitative Analysis** 

Here we present the qualitative results of Matcher, Baseline (1[st] row in Tab. 4), Baseline+PG (2[nd] row in Tab. 4) and our approach in Fig. 4. The bipartite matching of Matcher has a negative influence when the areas of the target object in reference and target images have significant differences, as shown in the 1[st] and 3[rd] rows. The positive gating with clustering filters out the noise masks in the 3[rd] row, while the Overshooting Gating further removes the masks belonging to overshooting points in the 2[nd] and 4[th] rows. More qualitative analyses please refer to the appendix. 

## **6 Conclusion** 

In this paper, we proposed an efficient, training-free SAM-based FSS approach that requires no external hyperparameters. As an automatic SAM-based semantic segmentation pipeline, our approach balanced candidate points and object coverage in the Positive-Negative Alignment (PNA) module, then used SAM-generated masks in the Point-Mask Clustering (PMC) module to enhance Post Gating. Extensive experiments validated the superior performance of our approach, advancing semantic segmentation without extensive parameter tuning or training. 

## **Acknowledgments and Disclosure of Funding** 

This work was supported by the National Natural Science Foundation of China under No. 62472033, No. U23A20314, and No. 61972036. J. Jiao is supported by the Royal Society Short Industry Fellowship (SIF\R1\231009) and the Amazon Research Award. 

## **References** 

- [1] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 3431–3440, 2015. 

- [2] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Semantic image segmentation with deep convolutional nets and fully connected crfs,” _arXiv preprint arXiv:1412.7062_ , 2014. 

- [3] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, “Rethinking atrous convolution for semantic image segmentation,” _arXiv preprint arXiv:1706.05587_ , 2017. 

- [4] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” _IEEE Transactions on Pattern Recognition and Machine Intelligence_ , vol. 40, no. 4, pp. 834–848, 2017. 

- [5] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” _Advances in neural information processing systems_ , vol. 34, pp. 12077–12090, 2021. 

- [6] Z. Zhang, G. Gao, Z. Fang, J. Jiao, and Y. Wei, “Mining unseen classes via regional objectness: A simple baseline for incremental segmentation,” _Advances in neural information processing systems_ , vol. 35, pp. 24340–24353, 2022. 

- [7] Z. Zhang, G. Gao, J. Jiao, C. H. Liu, and Y. Wei, “Coinseg: Contrast inter-and intra-class representations for incremental segmentation,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 843–853, 2023. 

- [8] A. Zhang and G. Gao, “Background adaptation with residual modeling for exemplar-free class-incremental semantic segmentation,” _Proc. European Conference on Computer Vision_ , 2024. 

10 

- [9] B. Cheng, A. Schwing, and A. Kirillov, “Per-pixel classification is not all you need for semantic segmentation,” _Advances in neural information processing systems_ , vol. 34, pp. 17864–17875, 2021. 

- [10] A. Kirillov, E. Mintun, and et al., “Segment anything,” in _Proc. IEEE International Conference on Computer Vision_ , pp. 4015–4026, 2023. 

- [11] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, _et al._ , “Dinov2: Learning robust visual features without supervision,” _arXiv preprint arXiv:2304.07193_ , 2023. 

- [12] R. Zhang, Z. Jiang, Z. Guo, S. Yan, J. Pan, H. Dong, Y. Qiao, P. Gao, and H. Li, “Personalize segment anything model with one shot,” in _Proc. International Conference on Learning Representations_ , 2024. 

- [13] Y. Liu, M. Zhu, H. Li, H. Chen, X. Wang, and C. Shen, “Matcher: Segment anything with one shot using all-purpose feature matching,” in _Proc. International Conference on Learning Representations_ , 2024. 

- [14] J. Min, D. Kang, and M. Cho, “Hypercorrelation squeeze for few-shot segmentation,” in _Proc. IEEE International Conference on Computer Vision_ , pp. 6941–6952, 2021. 

- [15] Q. Xu, W. Zhao, G. Lin, and C. Long, “Self-calibrated cross attention network for few-shot segmentation,” in _Proc. IEEE International Conference on Computer Vision_ , pp. 655–665, 2023. 

- [16] Y. Sun, Q. Chen, X. He, J. Wang, H. Feng, J. Han, E. Ding, J. Cheng, Z. Li, and J. Wang, “Singular value fine-tuning: Few-shot segmentation requires few-parameters fine-tuning,” _Advances in neural information processing systems_ , vol. 35, pp. 37484–37496, 2022. 

- [17] Y. Liu, N. Liu, Q. Cao, X. Yao, J. Han, and L. Shao, “Learning non-target knowledge for few-shot semantic segmentation,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 11573–11582, 2022. 

- [18] Z. Fang, G. Gao, Z. Zhang, and A. Zhang, “Hierarchical context-agnostic network with contrastive feature diversity for one-shot semantic segmentation,” _Journal of Visual Communication and Image Representation_ , vol. 90, p. 103754, 2023. 

- [19] G. Gao, Z. Fang, C. Han, Y. Wei, C. H. Liu, and S. Yan, “Drnet: Double recalibration network for few-shot semantic segmentation,” _IEEE Transactions on Image Processing_ , vol. 31, pp. 6733– 6746, 2022. 

- [20] X. Wang, X. Zhang, Y. Cao, W. Wang, C. Shen, and T. Huang, “Seggpt: Towards segmenting everything in context,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 1130–1140, 2023. 

- [21] X. Wang, W. Wang, Y. Cao, C. Shen, and T. Huang, “Images speak in images: A generalist painter for in-context visual learning,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 6830–6839, 2023. 

- [22] A. Shaban, S. Bansal, Z. Liu, I. Essa, and B. Boots, “One-shot learning for semantic segmentation,” in _Proc. British Machine Vision Conference_ , pp. 167.1–167.13, 2017. 

- [23] K. Nguyen and S. Todorovic, “Feature weighting and boosting for few-shot segmentation,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 622–631, 2019. 

- [24] X. Li, T. Wei, Y. P. Chen, Y.-W. Tai, and C.-K. Tang, “Fss-1000: A 1000-class dataset for few-shot segmentation,” in _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 2869–2878, 2020. 

- [25] I. Demir, K. Koperski, D. Lindenbaum, G. Pang, J. Huang, S. Basu, F. Hughes, D. Tuia, and R. Raskar, “Deepglobe 2018: A challenge to parse the earth through satellite images,” in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops_ , pp. 172–181, 2018. 

11 

- [26] N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, _et al._ , “Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic),” _arXiv preprint arXiv:1902.03368_ , 2019. 

- [27] X. Yao, Q. Cao, X. Feng, G. Cheng, and J. Han, “Scale-aware detailed matching for few-shot aerial image semantic segmentation,” _IEEE Transactions on Geoscience and Remote Sensing_ , vol. 60, pp. 1–11, 2021. 

- [28] C. Zhang, G. Lin, F. Liu, R. Yao, and C. Shen, “Canet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 5217–5226, 2019. 

- [29] K. Wang, J. H. Liew, Y. Zou, D. Zhou, and J. Feng, “Panet: Few-shot image semantic segmentation with prototype alignment,” in _Proc. IEEE International Conference on Computer Vision_ , pp. 9197–9206, 2019. 

- [30] G. Li, V. Jampani, L. Sevilla-Lara, D. Sun, J. Kim, and J. Kim, “Adaptive prototype learning and allocation for few-shot segmentation,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 8334–8343, 2021. 

- [31] Z. Tian, H. Zhao, M. Shu, Z. Yang, R. Li, and J. Jia, “Prior guided feature enrichment network for few-shot segmentation,” _IEEE Transactions on Pattern Recognition and Machine Intelligence_ , vol. 44, no. 2, pp. 1050–1065, 2020. 

- [32] M. Boudiaf, H. Kervadec, Z. I. Masud, P. Piantanida, I. Ben Ayed, and J. Dolz, “Few-shot segmentation without meta-learning: A good transductive inference is all you need?,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 13979–13988, 2021. 

- [33] A. Okazawa, “Interclass prototype relation for few-shot segmentation,” in _Proc. European Conference on Computer Vision_ , pp. 362–378, 2022. 

- [34] J.-W. Zhang, Y. Sun, Y. Yang, and W. Chen, “Feature-proxy transformer for few-shot segmentation,” in _Advances in neural information processing systems_ , pp. 6575–6588, 2022. 

- [35] X. Shi, D. Wei, Y. Zhang, D. Lu, M. Ning, J. Chen, K. Ma, and Y. Zheng, “Dense cross-queryand-support attention weighted mask aggregation for few-shot segmentation,” in _Proc. European Conference on Computer Vision_ , pp. 151–168, 2022. 

- [36] H. Liu, P. Peng, T. Chen, Q. Wang, Y. Yao, and X.-S. Hua, “Fecanet: Boosting few-shot semantic segmentation with feature-enhanced context-aware network,” vol. 25, pp. 8580–8592, 2023. 

- [37] Y. Wang, R. Sun, and T. Zhang, “Rethinking the correlation in few-shot segmentation: A buoys view,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 7183–7192, 2023. 

- [38] Y. Liu, N. Liu, X. Yao, and J. Han, “Intermediate prototype mining transformer for few-shot semantic segmentation,” vol. 35, pp. 38020–38031, 2022. 

- [39] S. Jiao, G. Zhang, S. Navasardyan, L. Chen, Y. Zhao, Y. Wei, and H. Shi, “Mask matching transformer for few-shot segmentation,” _Advances in neural information processing systems_ , vol. 35, pp. 823–836, 2022. 

- [40] G. Zhang, G. Kang, Y. Yang, and Y. Wei, “Few-shot segmentation via cycle-consistent transformer,” _Advances in neural information processing systems_ , vol. 34, pp. 21984–21996, 2021. 

- [41] C. Lang, G. Cheng, B. Tu, and J. Han, “Learning what not to segment: A new perspective on few-shot segmentation,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 8057–8067, 2022. 

- [42] Q. Fan, W. Pei, Y.-W. Tai, and C.-K. Tang, “Self-support few-shot semantic segmentation,” in _Proc. European Conference on Computer Vision_ , pp. 701–719, 2022. 

12 

- [43] X. Zhang, Y. Wei, Y. Yang, and T. S. Huang, “Sg-one: Similarity guidance network for one-shot semantic segmentation,” _IEEE Transactions on Cybernetics_ , vol. 50, no. 9, pp. 3855–3865, 2020. 

- [44] B. Peng, Z. Tian, X. Wu, C. Wang, S. Liu, J. Su, and J. Jia, “Hierarchical dense correlation distillation for few-shot segmentation,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 23641–23651, 2023. 

- [45] Y. Wang, N. Luo, and T. Zhang, “Focus on query: Adversarial mining transformer for few-shot segmentation,” _Advances in neural information processing systems_ , vol. 36, pp. 31524–31542, 2023. 

- [46] F. Li, H. Zhang, P. Sun, X. Zou, S. Liu, J. Yang, C. Li, L. Zhang, and J. Gao, “Semantic-sam: Segment and recognize anything at any granularity,” _arXiv preprint arXiv:2307.04767_ , 2023. 

- [47] H. Yuan, X. Li, C. Zhou, Y. Li, K. Chen, and C. C. Loy, “Open-vocabulary sam: Segment and recognize twenty-thousand classes interactively,” _arXiv preprint_ , 2024. 

- [48] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, _et al._ , “Learning transferable visual models from natural language supervision,” in _International conference on machine learning_ , pp. 8748–8763, PMLR, 2021. 

- [49] Y. Sun, J. Chen, S. Zhang, X. Zhang, Q. Chen, G. Zhang, E. Ding, J. Wang, and Z. Li, “Vrp-sam: Sam with visual reference prompt,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition_ , 2024. 

- [50] S. Hong, S. Cho, J. Nam, S. Lin, and S. Kim, “Cost aggregation with 4d convolutional swin transformer for few-shot segmentation,” in _Proc. European Conference on Computer Vision_ , pp. 108–126, 2022. 

- [51] J. Su, Q. Fan, G. Lu, F. Chen, and W. Pei, “Domain-rectifying adapter for cross-domain few-shot segmentation,” 2024. 

- [52] Q. Cao, Y. Chen, C. Ma, and X. Yang, “Few-shot rotation-invariant aerial image semantic segmentation,” _IEEE Transactions on Geoscience and Remote Sensing_ , vol. 62, pp. 1–13, 2023. 

- [53] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” _International Journal on Computer Vision_ , vol. 88, pp. 303–338, 2010. 

- [54] B. Hariharan, P. Arbeláez, L. Bourdev, S. Maji, and J. Malik, “Semantic contours from inverse detectors,” in _Proc. IEEE International Conference on Computer Vision_ , pp. 991–998, 2011. 

- [55] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in _Proc. European Conference on Computer Vision_ , pp. 740–755, 2014. 

- [56] X. Chen, R. Mottaghi, X. Liu, S. Fidler, R. Urtasun, and A. Yuille, “Detect what you can: Detecting and representing objects using holistic models and body parts,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 1971–1978, 2014. 

- [57] K. Morabia, J. Arora, and T. Vijaykumar, “Attention-based joint detection of object and semantic part,” _arXiv preprint arXiv:2007.02419_ , 2020. 

- [58] V. Ramanathan, A. Kalia, V. Petrovic, Y. Wen, B. Zheng, B. Guo, R. Wang, A. Marquez, R. Kovvuri, A. Kadian, _et al._ , “Paco: Parts and attributes of common objects,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 7141–7151, 2023. 

- [59] S. Waqas Zamir, A. Arora, A. Gupta, S. Khan, G. Sun, F. Shahbaz Khan, F. Zhu, L. Shao, G.-S. Xia, and X. Bai, “isaid: A large-scale dataset for instance segmentation in aerial images,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops_ , pp. 28–37, 2019. 

- [60] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, _et al._ , “An image is worth 16x16 words: Transformers for image recognition at scale,” _Proc. International Conference on Learning Representations_ , 2020. 

13 

## **A Appendix / supplemental material** 

## **A.1 More Details for Mask Growth Algorithm** 

We mention the Mask Growth algorithm in Sec. 4.3. The Mask Growth algorithm is designed for witheach cluster of masksan empty setmask _y_ ¨ _y_ ˆ _pq_ are removed.based on the sorted sequence of indices _P_[+] and a blank pseudo mask _M_ We compute the positive value[ˆ] _weak,p_ . The details of the algorithm are shown in Alg. _y_ ¨ _p_ . Then, we start an iterative process and get the current _Q_ . The parts of the current mask _s_[+] _q_[of the remaining parts.] 1. We first initialize[If] _y[ s]_ ˆ _q_[+] _q_ overlapping[is positive,] the mask _y_ ˆ _q_ is updated into the _y_ ¨ _p_ and its corresponding point _Pq_ is added into _P_[+] . As soon as the iterative process finishes, the set of positive points _P_[+] is established. 

## **Algorithm 1** Mask Growth for each cluster 

**Input:** _M_ ˆ _p,_ ¨ _yp, Q_ , _P_[+] **for** _n_ = 1 to _|Q|_ **do** _q ← Q_ ( _n_ ) _y_ ˆˆ _q ←_ ˆ _M_[ˆ] _p_ ( _q_ ) ¨ _yq_ = _yq_ & _∼ yp s_[+] _q[←]_[�] _[hw] i_ =1 _[S]_[ˆ][+][(] _[i]_[)] _[ ⊙I]_[(ˆ] _[y][q]_[)(] _[i]_[)] **if** _s_[+] _q[>]_[ 0] **[ then]** Add¨ _P_ ˆ _q_ to _P_ ¨[+] _. yp_ = _yq ∨ yp_ **end if end for Output:** _P_[+] 

## **A.2 Limitations** 

Our approach has impressive performance on Few-shot Semantic Segmentation tasks. However, due to the resolution of features _F[t]_ from DINOv2 not aligning with the required resolution for prompting the SAM, we directly map the coordinates of points in _F[t]_ to coordinates for prompting. This results in coordinate bias for small objects, as the gap between neighboring points can reach approximately 28 pixels. Our future work will focus on locating small objects. 

## **A.3 Societal Impacts** 

As a completely automatic SAM-based few-shot semantic segmentation approach without external hyperparameters, our method is capable of handling various scenarios of semantic segmentation, as demonstrated by our extensive experiments. The efficiency and generalizability of our method ensure a wide range of applications. Furthermore, since our training-free method is constructed upon the widely used open-source foundation models, we have not identified the negative societal impact to date. 

## **A.4 Details of Current SAM-based FSS Methods** 

Our approach aims to address several issues present in previous SAM-based FSS methods to achieve an automatic SAM-based model. These issues include the requirement of excessive external hyperparameters, overusing the mask generator of SAM, prolonged inference times, etc. The Tab. 7 shows the difference between our approach and previous SAM-based FSS methods. Fig. 5 shows the difference in using SAM as the mask generator between our approach and previous methods. Fig. 5(a) presents the iterative refinement of PerSAM, which involves generating masks from SAM 3 times. Fig. 5(b) exhibits that Matcher introduces an external Automatic Mask Generator, which automatically prompts for generating all mask proposals in the image. Our approach in Fig. 5(c) only utilizes the standard Mask Generator of SAM and generates the masks with our prompts only once. 

14 

Table 7: Details of the current SAM-based FSS methods. 

|Methods|PerSAM<br>PerSAM-F<br>Matcher<br>VRP-SAM<br>Ours|
|---|---|
|||
|Training-free<br>External-hyperparameters-free<br>Once mask generation<br>Inference speed (s/img)<br>COCO-20i mIoU|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>1.43<br>16.5<br>12.7<br>N/A<br>1.88<br>23.0<br>23.5<br>52.7<br>53.9<br>58.7|




![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0015-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0015-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0015-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0015-05.png)


**----- Start of picture text -----**<br>
(a) (b) (c)<br>**----- End of picture text -----**<br>


Figure 5: Comparison of the pipeline between the previous methods and our approach. (a) PerSAM [12] iteratively uses the Mask Generator to refine the mask. (b) Matcher [13] introduced an external Automatic Mask Generator [10] with automatic prompting to excessively generate masks from the whole image. (c) The effectiveness of the PMC module and Post-Gating ensures that our approach uses Mask Generator with our prompts only once. 

## **A.5 Discussion of SAM** 

## **A.5.1 Features from ViT Encoder of SAM** 

Previous state-of-the-art generalist FSS methods [] use DINOv2 or ResNet-50, instead of the default ViT encoder of SAM, for fine-grained features. We visualize the representative samples of Pascal-5[i] in Fig. 6. The 3[rd] column of maps represents the self-similarity of the _FSAM[t]_[.][We introduce the][ 3] _[ ×]_[ 3] average pooling for _FSAM[t]_[followed by computing the cosine similarity between the pooled features] and _FSAM[t]_[.][The maps illustrate that] _[ F][ t] SAM_[can accurately identify the regions of objects within the] image, where the features within each object region are nearly identical, while features between neighboring different objects are distinct. 

Although the characteristics of _FSAM[t]_[ensure the generation of high-quality masks, the coarse-grained] features are not suitable for locating the objects, as shown in the 4[th] column. The similarity between _FSAM[t]_[and] _[F][ r] SAM_[cannot][effectively][distinguish][the][target][object][well][compared][to] _[S] mean_[+][from] DINOv2. Therefore, we follow the previous methods using DINOv2 for fine-grained features. 

## **A.5.2 Masks analysis for Point-Mask Clustering** 

Our Point-Mask Clustering module introduces a parameter-free clustering method by constructing a graph of coverage. The effectiveness of the method primarily relies on the high-quality masks, whose boundaries mostly align with the object boundaries. We roughly analyze the coverage of masks generated from the points in the ground truth foreground region using 4000 samples from Pascal-5[i] . In particular,distributions, including the distribution of IoU betweenwe get the union of masks from the foreground points as _y_ ˆ _fore_ and union of masks from background _y_ ˆ _fore_ , and visualize three points ˆ _yback_ in Fig. 7, the ratio between the number of background points and all points covered by the _y_ ˆ _fore_ in Fig. 8, the number of background points covered by _y_ ˆ _fore_ in Fig. 9. The distribution charts demonstrate that most of the samples have acceptable coverage on the error points for Point-Mask Clustering. Given the limited precision of ground truth annotations, the analysis is for reference only. The effectiveness of our Point-Mask Clustering is validated in our ablation study in Tab. 4. 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-00.png)


**----- Start of picture text -----**<br>
Reference Image Target Image Self Similarity Cross Similarity 𝑆𝒎𝒆𝒂𝒏�<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-15.png)


Figure 6: Analysis of the features from default ViT encoder of SAM. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0016-17.png)


**----- Start of picture text -----**<br>
100<br>700<br>80<br>600<br>500<br>60<br>400<br>300 40<br>200<br>20<br>100<br>0 0<br>0 20 40 60 80 100<br>Error IoU (%)<br>Number of Images<br>Cumulative Percentage (%)<br>**----- End of picture text -----**<br>


Figure 7: The distribution of IoU between masks from foreground points and from background points. 

## **A.6 Additional Experiment Results.** 

## **A.6.1 Performance of Different Foundation Model Sizes** 

Tab. 8 shows the experiment results of our approach with different scales of SAM and DINOv2. Compared to the previous training-free method Matcher, our approach still achieves a better performance with SAM-Large and DINOv2-Base. The fair comparison with SAM-Huge and DINOv2-Large further demonstrates the effectiveness of our approach. 

## **A.6.2 Detailed Results of Evaluation Datasets** 

We present the detailed results on different Few-shot Semantic Segmentation datasets, including Pascal-5[i] in Tab. 9, COCO-20[i] in Tab. 10, LVIS-92[i] in Tab. 11, PASCAL-Part and PACO-Part in 

16 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0017-00.png)


**----- Start of picture text -----**<br>
100<br>800 80<br>600 60<br>400 40<br>200 20<br>0 0<br>0 20 40 60 80 100<br>Error Rate (%)<br>Number of Images<br>Cumulative Percentage (%)<br>**----- End of picture text -----**<br>


Figure 8: The distribution of the ratio between the number of background points and all points covered by the masks from foreground points. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0017-02.png)


**----- Start of picture text -----**<br>
100<br>700<br>600<br>80<br>500<br>60<br>400<br>300 40<br>200<br>20<br>100<br>0 0<br>0 20 40 60 80<br>Number of Error Points<br>Number of Images<br>Cumulative Percentage (%)<br>**----- End of picture text -----**<br>


Figure 9: The distribution of the number of background points covered by the masks from foreground points. 

Tab. 12, iSAID-5[i] in Tab. 13. The results show that our approach has remarkable performance in each fold of the datasets, demonstrating its generalized effectiveness in various scenarios. 

## **A.6.3 Multiple Random Seeds Experiment** 

Previous state-of-the-art methods, including both specialist methods and generalist methods, typically do not conduct multiple random seed experiments to evaluate the robustness. To demonstrate the robustness of our approach, we randomly set 5 different random seeds and conducted the experiments on the datasets that were not fully evaluated in the standard evaluation. As shown in Fig. 10, despite variations in random seeds, our approach consistently exhibits better performance compared to previous methods that were not evaluated with random seeds. 

## **A.7 Additional Ablation Study** 

## **A.7.1 Ablation Study of Pivots for Positive Gating** 

We apply both _smid_ and _Smean[−]_[as the pivots for Positive Gating in Sec.][ 4.3][, aiming to leverage both] the pivots from the _Smean_[+][itself and the negative similarity.][As shown in Tab.][ 14][, combining these] two pivots for Positive Gating yields a significant improvement compared to using only one pivot. Moreover, the combination method of _×_ shows a 0.3% mIoU enhancement compared to +. 

17 

Table 8: Evaluation of our approach with different sizes of SAM and DINOv2. 

|Methods|SAM<br>DINOv2|Params.|COCO-20i<br>FSS-1000<br>LVIS-92i|
|---|---|---|---|
|||||
|Matcher|huge<br>large|945M|52.7<br>87.0<br>31.4|
|||||
|Ours|base<br>base<br>large<br>base<br>large<br>large<br>huge<br>large|180M<br>399M<br>617M<br>945M|53.7<br>85.6<br>31.1<br>55.5<br>87.5<br>31.7<br>58.0<br>87.8<br>35.1<br>58.7<br>88.0<br>35.2|



Table 9: Detail results of Pascal-5[i] . 

|Methods|Pascal-5i 1-shot<br>fold0<br>fold1<br>fold2<br>fold3<br>mean|Pascal-5i 5-shot<br>fold0<br>fold1<br>fold2<br>fold3<br>mean|
|---|---|---|
||||
|AMFormer<br>Matcher<br>Ours|71.3<br>76.7<br>70.7<br>63.9<br>70.7<br>67.7<br>70.7<br>66.9<br>67.0<br>68.1<br>71.1<br>75.7<br>69.2<br>73.3<br>72.1|74.4<br>78.5<br>74.3<br>67.2<br>73.6<br>71.4<br>77.5<br>74.1<br>72.8<br>74.0<br>81.5<br>86.3<br>79.7<br>82.9<br>82.6|



Table 10: Detail results of COCO-20[i] . 

||Table 10: Detail results of|COCO-20i.|
|---|---|---|
|Methods|COCO-20i 1-shot<br>fold0<br>fold1<br>fold2<br>fold3<br>mean|COCO-20i 5-shot<br>fold0<br>fold1<br>fold2<br>fold3<br>mean|
||||
|AMFormer<br>Matcher<br>Ours|44.9<br>55.8<br>52.7<br>50.6<br>51.0<br>52.7<br>53.5<br>52.6<br>52.1<br>52.7<br>56.6<br>61.4<br>59.6<br>57.1<br>58.7|52.0<br>61.9<br>57.4<br>57.9<br>57.3<br>60.1<br>62.7<br>60.9<br>59.2<br>60.7<br>67.1<br>69.4<br>66.0<br>64.8<br>66.8|



Table 11: Detail results of LVIS-92[i] . 

||Table 11: Detail results of LVIS-92i.|
|---|---|
|Methods|LVIS-92i 1-shot<br>fold0<br>fold1<br>fold2<br>fold3<br>fold4<br>fold5<br>fold6<br>fold7<br>fold8<br>fold9<br>mean|
|||
|Matcher<br>Ours|31.4<br>30.9<br>33.7<br>38.1<br>30.5<br>32.5<br>35.9<br>34.2<br>33.0<br>29.7<br>31.4<br>30.9<br>37.9<br>37.1<br>39.6<br>31.2<br>36.4<br>39.1<br>35.7<br>32.3<br>31.5<br>35.2|
|||
|Methods|LVIS-92i 5-shot<br>fold0<br>fold1<br>fold2<br>fold3<br>fold4<br>fold5<br>fold6<br>fold7<br>fold8<br>fold9<br>mean|
|||
|Matcher<br>Ours|37.0<br>36.6<br>47.3<br>39.1<br>37.1<br>41.8<br>42.7<br>37.7<br>37.9<br>43.3<br>40.0<br>42.1<br>38.4<br>50.0<br>42.5<br>42.0<br>46.5<br>46.4<br>41.5<br>43.7<br>48.4<br>44.2|



## Table 12: Detail results of PASCAL-Part and PACO-Part. 

|Methods|PASCAL-Part<br>animals<br>indoor<br>person<br>vehicles<br>mean|PACO-Part<br>fold0<br>fold1<br>fold2<br>fold3<br>mean|
|---|---|---|
||||
|HSNet<br>Matcher<br>Ours|21.2<br>53.0<br>20.2<br>35.1<br>32.4<br>37.1<br>56.3<br>32.4<br>45.7<br>42.9<br>33.2<br>59.6<br>35.2<br>50.1<br>44.5|20.8<br>21.3<br>25.5<br>22.6<br>22.6<br>32.7<br>35.6<br>36.5<br>34.1<br>34.7<br>33.4<br>34.9<br>39.7<br>37.0<br>36.3|



Table 13: Detail results of iSAID-5[i] . 

|Methods|iSAID-5i 1-shot<br>fold0<br>fold1<br>fold2<br>mean|iSAID-5i 5-shot<br>fold0<br>fold1<br>fold2<br>mean|
|---|---|---|
||||
|FRINet<br>Matcher<br>Ours|46.5<br>36.9<br>43.9<br>42.6<br>37.3<br>23.8<br>38.8<br>33.3<br>53.4<br>36.8<br>51.2<br>47.1|48.9<br>38.1<br>46.5<br>44.5<br>38.3<br>24.0<br>40.6<br>34.3<br>59.3<br>39.9<br>58.0<br>52.4|



18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0019-00.png)


**----- Start of picture text -----**<br>
Ours<br>70 Ours with 5 random seeds<br>Previous state-of-the-art generalist models<br>60 Previous state-of-the-art specialist models<br>50<br>40<br>30<br>20<br>10<br>0<br>COCO-20 [i] Pascal-5 [i] LVIS-92 [i] PASCAL-Part PACO-Part Deepglobe ISIC<br>Datasets<br>mIoU<br>**----- End of picture text -----**<br>


Figure 10: Results of our approach in multiple random seeds experiment. The bars in the chart represent the result under the previous standard evaluation. The error bar depicts the boundaries of our performance. 

Table 14: Ablation study of pivots and combination operations in Positive Gating. 

|Pivots<br>_smid_<br>_S−_<br>_mean_|Combination|COCO-20i|
|---|---|---|
||||
|✓<br>✓<br>✓<br>✓<br>✓<br>✓|+<br>_×_|44.5<br>51.0<br>56.8<br>57.1|



Table 15: Ablation study of different strategies for Positive Gating of the masks. 

|Strategies|COCO-20i<br>LVIS-92i<br>PASCAL-Part<br>PACO-Part|
|---|---|
|||
|Union<br>Mask Growth|59.4<br>36.1<br>40.1<br>31.6<br>58.7<br>35.2<br>44.5<br>36.3|



## **A.7.2 Ablation Study of Other Strategies for Positive Gating** 

In Sec. 4.3 and Sec. A.1, we introduce the Mask Growth algorithm as our strategy for judging whether the mask is positive. We compare the strategy to separately judging each mask in Tab. 5 and judging the union mask of each cluster in Tab. 15. While simply judging the union mask shows better performance on COCO-20[i] and LVIS-92[i] that require complete coverage of objects, its performance on One-shot Part Segmentation has a significant decline. Considering the generalizability of our approach, we select the Mask Growth algorithm as our strategy. 

## **A.8 Additional Qualitative Analysis** 

We conduct additional qualitative analysis to better present the result of our approach. Fig. 11 further compare the Matcher, Baseline, B+PJ, and B+PJ+OJ (Ours) following Sec. 5.5. Fig. 12 illustrate the intermediate contents in the Post Gating. Moreover, we provide additional visualization results of standard FSS in Fig. 13, One-shot Part Segmentation in Fig. 14, and Cross Domain FSS in Fig. 15. These qualitative results demonstrate the effectiveness of our approach. Notably, some of the results are even better than the corresponding annotations. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-00.png)


**----- Start of picture text -----**<br>
Ref. Image  Tar. Image  Matcher  Baseline  B+PG  B+PG+OG(Ours)<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-25.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-26.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-27.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-28.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-29.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-30.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-31.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-32.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-33.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-34.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-35.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-36.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-37.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-38.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-39.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-40.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-41.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-42.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-43.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-44.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-45.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-46.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-47.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0020-48.png)


Figure 11: More Qualitative results on COCO-20[i] for comparison among Matcher, Baseline, B+PG, B+PG+OG. 

20 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-00.png)


**----- Start of picture text -----**<br>
Ref. Image<br>Tar. Image<br>� 𝒎𝒆𝒂𝒏<br>𝑆<br>� 𝒎𝒆𝒂𝒏<br>𝑆<br>Clusters<br>PG<br>OG<br>Ours<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-25.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-26.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-27.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-28.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-29.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-30.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-31.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-32.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-33.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-34.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-35.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-36.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-37.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-38.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-39.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-40.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-41.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-42.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-43.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-44.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-45.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-46.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-47.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0021-48.png)


Figure 12: Qualitative analysis of the contents in gating. Different colors of points in the images in column “Clusters" represent different clusters. The green points in images in columns “PG" and “OG" denote the points satisfying the gating criteria, while the red points denote those not satisfying. 

21 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-00.png)


**----- Start of picture text -----**<br>
Pascal-5 [i] FSS-1000  LVIS-92 [ i]<br>Ref. Image Tar. Image Result Ref. Image Tar. Image Result Ref. Image Tar. Image Result<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-25.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-26.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-27.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-28.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-29.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-30.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-31.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-32.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-33.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-34.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-35.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-36.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-37.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-38.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-39.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-40.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-41.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-42.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-43.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-44.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-45.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-46.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-47.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-48.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-49.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-50.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-51.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-52.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-53.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-54.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-55.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-56.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-57.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-58.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-59.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-60.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-61.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-62.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-63.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-64.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-65.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-66.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-67.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-68.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-69.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-70.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-71.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0022-72.png)


Figure 13: Qualitative analysis of the results on Pascal-5[i] , FSS-1000, and LVIS-92[i] . 

22 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-00.png)


**----- Start of picture text -----**<br>
PASCAL-Part  PACO-Part<br>Ref. Image  Tar. Image  Result  Ref. Image  Tar. Image  Result<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-25.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-26.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-27.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-28.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-29.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-30.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-31.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-32.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-33.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-34.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-35.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-36.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-37.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-38.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-39.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-40.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-41.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-42.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-43.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-44.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-45.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-46.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-47.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0023-48.png)


Figure 14: Qualitative analysis of the results on PASCAL-Part and PACO-Part. 

23 


![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-00.png)


**----- Start of picture text -----**<br>
Deepglobe  iSAID-5 [i]<br>Ref. Image  Tar. Image  Result  Ref. Image  Tar. Image  Result<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-14.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-15.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-16.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-17.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-18.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-19.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-20.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-21.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-22.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-23.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-24.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-25.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-26.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-27.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-28.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-29.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-30.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-31.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-32.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-33.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-34.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-35.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-36.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-37.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-38.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-39.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-40.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-41.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-42.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-43.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-44.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-45.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-46.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-47.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/few-shot/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically_images/nips_2024_Bridge_the_Points_Graph-based_Few-shot_Segment_Anything_Semantically.pdf-0024-48.png)


Figure 15: Qualitative analysis of the results on Deepglobe and iSAID-5[i] . 

24 

## **NeurIPS Paper Checklist** 

## 1. **Claims** 

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: Our main claims made in the abstract and the last two paragraphs of the introduction accurately reflect the paper’s contributions and scope. Guidelines: 

   - The answer NA means that the abstract and introduction do not include the claims made in the paper. 

   - The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

   - The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

   - It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

2. **Limitations** 

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] 

Justification: We discuss the limitations in the appendix. 

Guidelines: 

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

- The authors are encouraged to create a separate "Limitations" section in their paper. 

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

## 3. **Theory Assumptions and Proofs** 

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [NA] 

25 

Justification: We do not introduce theory assumptions and proofs. Guidelines: 

- The answer NA means that the paper does not include theoretical results. 

- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

- All assumptions should be clearly stated or referenced in the statement of any theorems. 

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

- Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. **Experimental Result Reproducibility** 

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] 

Justification: Our paper fully discloses all the information needed to reproduce the main experimental results of the paper. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

## 5. **Open access to data and code** 

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

26 

## Answer: [Yes] 

Justification: Our code and instructions are included in the supplementary material. The data we use for the experiments are all from open-access datasets. Guidelines: 

- The answer NA means that paper does not include experiments requiring code. 

- Please see the NeurIPS code and data submission guidelines ( `https://nips.cc/ public/guides/CodeSubmissionPolicy` ) for more details. 

- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( `https: //nips.cc/public/guides/CodeSubmissionPolicy` ) for more details. 

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. **Experimental Setting/Details** 

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: Our paper specifies all the test details for our training-free approach in the section of experiments. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

- The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. **Experiment Statistical Significance** 

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [Yes] 

Justification: We set 5 random seeds to evaluate the large datasets and evaluate all samples of the small datasets in the appendix. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

27 

- The assumptions made should be given (e.g., Normally distributed errors). 

- It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. **Experiments Compute Resources** 

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: Computer resources are described in Implementation Details of the experiment section. 

Guidelines: 

- The answer NA means that the paper does not include experiments. 

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. **Code Of Ethics** 

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics `https://neurips.cc/public/EthicsGuidelines` ? 

Answer: [Yes] 

Justification: Our research conforms with the NeurIPS Code of Ethics in every respect. Guidelines: 

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. **Broader Impacts** 

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: We discuss both potential positive societal impacts and negative societal impacts of our work in the appendix. 

Guidelines: 

- 

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

28 

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. **Safeguards** 

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: Our paper has no such risks. 

Guidelines: 

- The answer NA means that the paper poses no such risks. 

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. **Licenses for existing assets** 

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: The existing assets used in our paper, i.e., SAM and DINOv2, are released on GitHub under Apache License 2.0. 

Guidelines: 

- The answer NA means that the paper does not use existing assets. 

- The authors should cite the original paper that produced the code package or dataset. 

- The authors should state which version of the asset is used and, if possible, include a URL. 

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, `paperswithcode.com/datasets` has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

29 

- If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. **New Assets** 

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: Our new assets introduced in the paper are well documented in the supplementary material. 

Guidelines: 

- The answer NA means that the paper does not release new assets. 

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

- The paper should discuss whether and how consent was obtained from people whose asset is used. 

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. **Crowdsourcing and Research with Human Subjects** 

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: There is no crowdsourcing and research with human subjects in our paper. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

## 15. **Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects** 

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: Our paper does not involve crowdsourcing or research with human subjects. Guidelines: 

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

30 

