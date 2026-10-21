## **Geometrically-driven Aggregation for Zero-shot 3D Point Cloud Understanding** 

Guofeng Mei Luigi Riz Yiming Wang Fabio Poiesi Bruno Kessler Foundation, Via Sommarive, 18, 38123 Trento, Italy _{_ gmei,luriz, ywang, poiesi _}_ @fbk.eu 

## **Abstract** 

_Zero-shot 3D point cloud understanding can be achieved via 2D Vision-Language Models (VLMs). Existing strategies directly map VLM representations from 2D pixels of rendered or captured views to 3D points, overlooking the inherent and expressible point cloud geometric structure. Geometrically similar or close regions can be exploited for bolstering point cloud understanding as they are likely to share semantic information. To this end, we introduce the first training-free aggregation technique that leverages the point cloud’s 3D geometric structure to improve the quality of the transferred VLM representations. Our approach operates iteratively, performing local-to-global aggregation based on geometric and semantic point-level reasoning. We benchmark our approach on three downstream tasks, including classification, part segmentation, and semantic segmentation, with a variety of datasets representing both synthetic/real-world, and indoor/outdoor scenarios. Our approach achieves new state-of-the-art results in all benchmarks. We will release the source code publicly._ 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0001-05.png)


**----- Start of picture text -----**<br>
geometric<br>feature<br>extraction<br>geometrically<br>image driven<br>encoder aggregation<br>semantic feature extraction<br>prompt text  classification part segmentation semantic segmentation<br>encoder<br>downstream tasks<br>**----- End of picture text -----**<br>


Figure 1. Given a set of dense per-pixel VLM representations (e.g. CLIP [21]) extracted from different viewpoint images of a point cloud, our approach is the first geometrically-driven aggregation technique to effectively transfer these image representations to 3D points. We use geometric features (e.g. FPFH [22]) extracted from the point cloud to denoise the VLM representations through an iterative process. This process begins by aggregating information locally and then extends to operate globally, thereby facilitating improvements across a variety of downstream tasks. 

## **1. Introduction** 

Zero-shot point cloud understanding aims to develop deep learning models capable of performing recognition tasks, such as shape classification or segmentation, on data unobserved at training time [8, 17, 35]. Especially, visionlanguage models (VLMs), such as CLIP [21], trained on internet-scale image-text pairs, can be effectively employed for zero-shot visual recognition tasks[11]. However, training a 3D deep model using CLIP’s dataset size and procedure is currently infeasible since point cloud-text pair data of the same scale as that used for CLIP is not yet available. 

Transferring VLM representations from images to point clouds has proven to be the most effective strategy so far for achieving zero-shot point cloud understanding [8, 29, 33, 35]. For example, PointCLIP [33] pioneers a training-free approach to directly transfer and combine per-pixel VLM representations of multiple rendered views of a point cloud for zero-shot classification and part segmentation. Its suc- 

cessor PointCLIPv2 [35] enhances 3D understanding by refining view rendering quality. ConceptFusion [8] fuses perobject VLM representations into per-pixel representations, transferring them to 3D points without requiring training. Both PointCLIPv2 and ConceptFusion have been rigorously tested in downstream tasks involving 3D scene understanding. However, 3D points remain isolated when transferring features from 2D domain to 3D domain, lacking any bridge for intra-information flow. Additionally, current fusion operations often disregard the spatial geometric structures and global context of the point clouds. We argue that the geometric structure of a point cloud contains valuable information that could be used to enhance the quality of the transferred VLM representations. In practical terms, we can rely on the key assumption: VLM representations should exhibit local smoothness and global consistency when their geometric structures are similar. To the best of our knowl- 

edge, no existing training-free method has harnessed the geometric information when transferring VLM representations from pixels to 3D points. 

In this paper, we introduce the first geometrically-driven training-free approach to aggregate point-level VLM representations for zero-shot point cloud understanding, which we name GeoZe. Unlike previous methods that employ naive pooling operations to transfer and aggregate VLM representations from images to 3D points [8, 17, 35], GeoZe harnesses both local and global structural information to enable geometric consistency of VLM representations across the point cloud. GeoZe leverages superpoints to aggregate local information from neighboring points and facilitates a global exchange among superpoints with similar geometric structures, promoting accuracy and computational efficiency for downstream tasks. We utilize geometric representations (3D descriptors [22]) to identify the structures sharing similar VLM representations. Our approach employs a weighted linear combination for aggregation, where the weights are calculated by jointly considering point cloud VLM representations, geometric representations, and coordinates. A critical aspect of our method is maintaining the original alignment of visual representations in the representation space to ensure compatibility with their corresponding language representations. To achieve this, we introduce the concept of VLM representation anchors. These anchors serve to correct potential offsets that may arise during the aggregation process, thereby preserving the integrity of the original representations. We evaluate GeoZe on three distinct zero-shot downstream tasks (shape classification, part segmentation, and semantic segmentation) across five diverse datasets (ModelNet40 [28], ObjectScanNN [24], ShapeNetPart [30], ScanNet [5], and nuScenes [1]). We use various types of 3D data as benchmarks, including synthetic/real-world datasets, indoor/outdoor settings, and LiDAR/RGBD acquisition sensors. GeoZe consistently outperforms the considered baseline methods by a significant margin in a total of nine experiments. In summary, our contributions are as follows: 

- We introduce GeoZe, the first geometrically-driven aggregation approach for zero-shot point cloud understanding, leveraging VLMs. 

- We show GeoZe’s versatility, being training-free and adaptable to various architectures for different downstream tasks. 

- We establish new state-of-the-art benchmarks in zero-shot downstream tasks, including object classification, part segmentation, and semantic segmentation. 

## **2. Related works** 

**Zero-shot PCD understanding.** The advent of large-scale VLMs has prompted a surge in zero-shot point cloud understanding [3, 4], including shape classification [6, 7, 29, 33], 

and dense semantic segmentation [8, 17, 32, 35]. Zeroshot point cloud understanding can be performed by training a 3D encoder tasked with aligning 3D representations with VLM representations projected from images [6, 7, 17, 29, 32, 35]. For example, ULIP [29] trains a 3D encoder to align 3D features within the same latent space as the multi-view visual features extracted by CLIP’s visual encoder and the textual features processed by CLIP’s text encoder. CG3D [6] also trains a 3D encoder and introduces prompt tuning for the visual encoder. CLIP2Point [7] retrains CLIP’s original image encoder with multi-view depth maps using a contrastive approach. CLIP[2] [32] dissects 3D scenes into text-image-point cloud proxies and applies cross-modal pre-training to generate point-level VLM representations. OpenScene [17] distills multi-view pixel-level VLM representations into a 3D encoder to learn point-level representations. Zero-shot point cloud understanding can also be achieved in a training-free manner [8, 33, 35]. PointCLIP [33] renders multi-view depth maps from point clouds and aggregates view-level zero-shot predictions for 2D-to3D knowledge transfer. PointCLIPv2 [35] incorporates large language models for 3D-specific text prompts and introduces dense depth rendering with a novel 2D-3D mapping, enabling dense point-level representation and, consequently, part segmentation tasks. ConceptFusion [8] seeks dense point-level representation for semantic scene understanding by exploiting off-the-shelf image segmenters to combine local crops and global representations from images to extract pixel-level features. All the aforementioned methods do not take geometrical attributes into account during the semantic feature aggregation process. In contrast, GeoZe incorporates geometric information during the VLM representation aggregation phase on the point cloud to enhance zero-shot performance across various downstream tasks. GeoZe does not require any learnable parameters and can be seamlessly integrated into any training-free method. 

**Point representation aggregation.** Most methods learn to encode the geometry of point clouds via neural networks and supervised learning. PointNet [19] uses max pooling for aggregation, while PointNet++ [20] hierarchically merges representations at multiple scales. PointCNN [12] hierarchically aggregates local representations using convolution operators on a transformed feature space. RSCNN [13] introduces ad-hoc convolutions that consider local neighborhood relations and geometric structure for aggregation. SPG [10] models the input point clouds as hierarchical super-points graphs, in which geometrical feature aggregation is performed with graph analysis. DGCNN [25] learns a graph for each point cloud and applies convolution operations on its edges. Point Transformer [34] introduces a transformer block to aggregate features via self-attention. Such operation can be computationally expensive, works like Fast Point Transformer [16] and 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0003-00.png)


**----- Start of picture text -----**<br>
anchor<br>generation<br>superpoint  ge neration extractionpatch  agglocal  rega tio n aggreglobal  ga tion superpoint-to-p oin t  projsemantic  ec tion<br>aggregation<br>local aggregation g loba l a ggr eg ation superpoint-to-point<br>¯ a ggr eg atio n<br>p c ¯<br>p ¯ j p ¯ j ¯ p ¯ d p ¯ a p c<br>p a p ¯ b p ¯ b<br>**----- End of picture text -----**<br>


Figure 2. Overview of the GeoZe framework. GeoZe first cluster point cloud _**P**_ into superpoints _**P**_[¯] along with their associated geometric representation _**G**_[¯] , VLM representation _**F**_[¯] , and anchors _**C**_ . For each superpoint _**p**_ ¯ _j_ , we identify its _k_ NN within the point cloud to form a patch _**P**[j]_ with their features _**G**[j]_ and _**F**[j]_ . For each patch we perform a local feature aggregation to refine the VLM representations _**F**_ . The superpoints then undergo a process of global aggregation. A global-to-local aggregation process is applied to update the per-point features. Lastly, we employ the VLM feature anchors to further refine per-point features, which are then ready to be utilized for downstream tasks. 

Point TransformerV2 [27] mitigate this drawback. Instead of learning semantic representations directly from 3D geometry, one can exploit multi-view rendering to learn 3D representation with additional visual inputs [2, 15, 26]. Recent zero-shot point cloud understanding methods transfer VLM representation to 3D either via learning a 3D encoder with backbones from PointNet++ [20] or PointMLP [14], or by mapping dense multi-view visual representation onto the 3D point cloud without training [8, 35]. OpenScene [17] projects multi-view pixel-level VLM representations on the 3D points, and multiple VLM representations for a point are fused via average pooling. ConceptFusion [8] employs SLAM techniques to incrementally integrate pixel-level VLM representations into the 3D space with a momentumbased update scheme. Also PointCLIPv2 [35] uses average pooling to fuse multi-view pixel-level VLM representations. The aforementioned aggregation methods either consider 3D geometry employing learning-based methods or simply aggregate VLM representations without taking into account local and global geometrical structure. Differently, GeoZe can effectively aggregate VLM representation in a geometrically-driven manner without training. 

## **3. Our approach** 

## **3.1. Overview** 

We begin by extracting dense pixel-level VLM representations, e.g. with CLIP [21], from each viewpoint image of a point cloud. The image can be rendered [35] or be the original one used to reconstruct the point cloud [17]. These representations are then projected onto the 3D points. We compute point-level geometric representations, such as Fast Point Feature Histograms (FPFH) [22], to guide the aggregation of our VLM representations. By utilizing point cloud 

coordinates and geometric representations, we generate superpoints along with their ass ~~ocia~~ ted geo ~~metric an~~ d VLM representations. For each ~~supe~~ rpoint, ~~we iden~~ tify its _k_ NN within the point cloud to form ~~a p~~ atch. F ~~or each pat~~ ch we perform a local feature aggregation to denoise the VLM representations. Then, superpoints undergo a process of global aggregation. We apply Mean Shift clustering on point-level VLM representations to establish VLM representation anchors. A global-to-local aggregation process is applied to update the point representations, based on the coordinate distance and VLM representations similarity between points and superpoints. Lastly, we employ the VLM representation anchors to further refine point-level representations, which become then ready to be utilized for downstream tasks. The diagram of GeoZe is shown in Fig. 2. 

## **3.2. Problem formulation** 

Let _**P**_ = _{_ _**p** i ∈_ R[3] _}[N] i_ =1[be][a][point][cloud][composed][of] _[N]_ 3D points _**p** i_ . For each _**p** i_ we compute and associate a VLM representation _**f** i ∈_ R _[b]_ , where _b_ is the dimension of _**f** i_ . Let _**F**_ = _{_ _**f** i ∈_ R _[b] }[N] i_ =1[be the set of VLM representations as-] sociated to each 3D point. For each point _**p** i_ , we compute and associate a geometric representation _**g** i ∈_ R _[d]_ , where _d_ is the dimension of _**g** i_ . Let _**G**_ = _{_ _**g** i ∈_ R _[d] }[N] i_ =1[be][the][set] of geometric representations associated to each 3D point. GeoZe transforms _**F**_ in geometrically-coherent VLM representations by combining the information from _**P**_ , _**F**_ , and _**G**_ based on local and global information through an iterative aggregation process. _**F**_ can be used for downstream tasks, such as classification and segmentation. 

## **3.3. VLM representation anchors and superpoints** 

The cardinality of point clouds is typically large, hence we compute superpoints and perform computations on them. A 

superpoint is a cluster of points featuring homogeneous geometric and VLM representations within a neighborhood of points. Information can be propagated back to individual points, resulting in dense point estimations. Intervening on VLM representations may lead to representations that diverge from the original feature space. Hence, we calculate a set of VLM representation anchors from the superpoints, which act as a reference for determining the final point-level representations. Steps for this computation are: i) initialize a set of point seeds uniformly distributed over the point cloud; ii) compute superpoints based on both geometric and visual information; iii) compute anchors based on aggregates of points that exhibit similar VLM representations. **Seed initialization.** We compute seeds through Farthest Point Sampling (FPS) [20]. Specifically, we apply FPS on _**P**_ ¯ _**P**_ = to _{_ _**p**_ ¯sample _j ∈_ R _N_[¯][3] _}_ seeds _Nj_[¯] =1[be] based[the][set] on point[of][seeds.] coordinates.The corre-Let sponding _**F**_ ¯ = _{_ ¯ _**f** j_ VLM _∈_ R _[b] }_ representations _Nj_[¯] =1[, and] _**[G]**_[¯][=] _[ {]_ _**[g]**_[¯] and _[j][∈]_ geometric[R] _[d][}] Nj_[¯] =1[, respectively.] features are **Superpoint computation.** Points proximate in the representation space are not necessarily proximate in the coordinate space. We aim to produce superpoints that uniformly partition the entire point cloud; thus, we formulate this as an optimal transport problem [18]. Specifically, we compute superpoints iteratively by searching for centroids in the coordinate, geometric, and VLM representation spaces. This is achieved by initializing and updating their respective seeds, _**P**_[¯] , _**G**_[¯] , and _**F**_[¯] . Firstly, for a given _j_ -th seed, we select a set of _K_ 1 nearest neighbors in the coordinate space with respect to _**p**_ ¯ _j_ , and compute the average similarity _**µ**_ ¯ = _{_ _**µ**_ ¯ _j}_ with _**µ**_ ¯ _j_ = 2 _K_ 1 1 � _Kk_ =11 �1 _._ 0 + _∥_ _**f**_ **[¯]** _**f** j[k]_ **¯** _j[k][∥∥]_ _**f**_ **[¯]** _**f** j_ **[¯]** _j ∥_ �. Then, we solve the following optimization problem: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0004-01.png)


where _**γ** ij_ is the probability of affiliation of point _i_ to superpoint _j_ , and _**γ**_ = _{_ _**γ** ij}[N] ij N_[¯] is the set of these probabilities. Eq. (1) can be solved by using the Sinkhorn algorithm [18]. We update the elements of _**P**_[¯] , _**G**_[¯] and _**F**_[¯] as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0004-03.png)


This formulation prevents superpoints that are connected in the feature space from being grouped into the same categories if they are disjoint in 3D space. We iterate _**µ**_ ¯ and Eq. (2) Γ times to converge to a stable solution. 

**VLM representation anchors.** We model anchors as centroids of clusters characterized by homogeneous VLM representations and employ Mean-Shift for iterative cluster determination [9]. We use Mean-Shift over other clustering 

methodologies, such as KMeans or Mixture Models, due to its ability to adaptively vary the number of clusters based on a kernel bandwidth and to be density aware. This approach can be particularly effective because the clusters we seek are those characterized by high-density points with distinctive representations, allowing each anchor centroid to potentially represent a semantic category. Clusters with lower densities can instead be more likely to be representative of noisy points. 

We introduce an optimized version of the Mean Shift clustering algorithm [31], specifically designed to address the high computational costs. This is implemented by initializing the anchors with the original representations of the seeds, _**C**_[0] = _{_ _**C**_[0] _j_[=] _**[f]**_[¯] _[j][}] Nj_[¯] =1[.][Mean-Shift is iterative and the] _t_ -th iteration is computed as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0004-08.png)


where _δ_ is the bandwidth that corresponds to the average similarity from each point to its 16-th nearest neighbor within the VLM representation space, and _∥· ∥_ is the _L_ 2 norm. Mean-shift iterations persist until convergence, which in our experiments typically occurs around 40 iterations. Centroids are then computed using non-maximum suppression: starting with the point of maximum density, all proximates with a similarity greater than the threshold _β_ are excluded, with the process iteratively repeated. Points are subsequently allocated to segments, according to proximity to their closest cluster center. Finally, we select _L_ centroids, which serve as the label prototypes _**C**_ = _{_ _**C** l}[L] l_ =1 for subsequent analyses. 

## **3.4. Geometrically-informed aggregation** 

Given the superpoints and VLM representation anchors, we perform an aggregation of VLM representations at point level. We aggregate VLM representations locally within the neighborhood of each superpoint and subsequently conduct a global aggregation among the superpoints. This information is aggregated from superpoints to individual points, assigning a VLM representation to each point. 

**Local aggregation.** Assuming that VLM representations should exhibit local smoothness, we perform local aggregation through a linear combination of VLM representations in the neighborhood of each superpoint. Specifically, for the _j_ -th superpoint, we select a set of _K_ 2 nearest-neighbor points defined as _**P**[j]_ for the point coordinates, _**F**[j]_ for the point-level VLM representations, and _**G**[j]_ for the point-level geometric representations. We compute the linear combination weights as the solution to an optimal transport problem using the Sinkhorn algorithm [18]. We apply the Sinkhorn algorithm based on the similarity of points in terms of their 

geometric and VLM representations. This approach allows us to regulate the contributions from neighboring points, mitigating situations where a few highly similar points disproportionately influence the linear combination. We define the geometric similarity as _**S** g[j]_[=] _**[G]**[j]_ _**[G]**[j][⊤][/] √d_ , and the VLM representation similarity as _**S** v[j]_[=] _**[F]**[j]_ _**[F]**[j][⊤][/] √b_ . Therefore, we compute the linear combination weights as **W** _[j] g_[=] `[ SH]` � _**S** g[j]_ � _,_ **W** _[j] v_[=] `[ SH]` � _**S** v[j]_ � _,_ where `SH` is the Sinkhorn function with 5 iterations. We combine, normalize, and use these weights to aggregate the VLM representations 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0005-01.png)


where `SM` is the Softmax operation. Then, we use the _**F**[j]_ refined¯ local features to update the superpoint features by _**F**_ = _**γ**_ ¯ _**F**_ . _**γ**_ is same as in Eq. (1). Once all the superpoints are processed, the next step is the global aggregation. **Global aggregation.** Geometrically similar regions may belong to similar objects, hence their VLM representations should be similar too. In order to aggregate VLM representations of objects with similar geometric features we expand our search at global level. As for the local aggregation, we use the Sinkhorn algorithm based on the similarity of geometric and VLM representations. Let _**S**_[¯] _[g]_ be the geometric similarity computed among all superpoints, where each element is computed as _**S**_[¯] _ij[g]_[=] _**[g]**_[¯] _[i]_ _**[g]**_[¯] _j[⊤][/] √d_ . Similarly, let _**S**[v]_ be the VLM representation similarity, where each element is computed as _**S**_[¯] _ij[v]_[=] _**[f]**_[¯] _[i]_ _**[f]**_[¯] _[ ⊤] j[/] √b_ . The weights _**W**_[¯] for the global aggregation are expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0005-03.png)


Then,12 � _**F**_ **¯** +the _**W**_ ¯superpoint _**F**_ **[¯]** �. level features are updated by _**F**_ **[¯]** = **Superpoint-to-point aggregation.** We aggregate the updated superpoints’ VLM representations together with their neighboring points. We use the distance between the point and each superpoint to weight the contribution of the VLM representation to aggregate. Specifically, given _**P**_ a¯ queryto querypointpoint _**p** i_ is _∈_ decided _**P**_ , the byweighttheirofcoordinatesuperpointdistance _**p**_ ¯ _j ∈_ _**S** ij[c]_[=1] _[.]_[0] _[−]_[tanh][ (] _[∥]_ _**[p]**[i][−]_ _**[p]**_[¯] _[j][∥]_[2][)][and][feature][similarity] _**[S]** ij[v]_[=] _**f** i_ _**f**_[¯] _j[⊤][/] √b_ . We opted for the tanh kernel due to its ability to sharpen the distance measure, thereby ensuring that points are primarily influenced by their closest super- _N_[¯] points in the coordinate space. Let _**S**[c]_ = _{_ _**S** ij[c][}][N,] i,j_ =1 _,_ 1[and] _N_[¯] _**S**[v]_ = _{_ _**S** ij[v][}][N,] i,j_ =1 _,_ 1[.][Then, the weight] _**[ W]**_[is calculated by] 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0005-05.png)


The point-level feature is updated via _**F**_ = 2[1] � _**F**_ + **W** _**F**_[¯] � to produce our final aggregated representations. **Anchor projection.** Noisy representations may be incorporated during aggregation, thus causing offsets within the 

VLM representation space. Therefore, we project VLM representations onto our VLM representation anchors as _**f** i_ = arg max _**c** j ∈_ _**C**_ � _**f** i[⊤]_ _**[C]**_ � This makes the aggregated VLM representations to be more inclined to resemble their corresponding semantic representations. 

## **4. Results** 

## **4.1. Experimental setup** 

GeoZe is implemented using PyTorch and Open3D libraries. Experiments are executed on an NVIDIA A40 48GB GPU. We use FPFH [22] as geometric features. Additional setting details are in the Supplementary Material. **Prompting.** We use the text prompting techniques proposed in the comparison methods. Specifically, for PointCLIPv2 [35] we use the prompts obtained via “GPT Prompting” [35]. For OpenScene [17] and ConceptFusion [8] we use the “prompt engineering” proposed in [17]. 

## **4.2. Shape Classification** 

**Setting.** We evaluate GeoZe on four datasets for shape classification tasks. Specifically, we use the synthetic dataset ModelNet40 [28], and three variants of the real-world dataset ObjectScanNN [24]. ModelNet40 [28] comprises 12,311 point clouds, with 2,468 of these designated for testing, uniformly sampled from CAD mesh surfaces across 40 categories. We report the results on the test set. ScanObjectNN [24] comprises 15 categories and 2,902 point clouds divided in: OBJ-ONLY (ground-truth segmented objects extracted from the scene beforehand, i.e., objects without background), OBJ-BG (objects attached with background data), and S-PB-T50-RS (objects perturbed with random rotation and scaling). We report the results computed on the test set that comprises 580 point clouds. GeoZe processes the depth maps extracted with PointCLIPv2 [35]. Unlike PointCLIPv2 that performs average pooling on the global representations of each depth map, we transform depth maps in point clouds, apply GeoZe, and then perform max pooling on the new VLM representations. 

**Results.** Tab. 1 reports comparative results in terms of classification accuracy. GeoZe outperforms PointCLIPv2 on all datasets. Specifically, GeoZe achieves 70 _._ 17% on ModelNet40, outperforming PointCLIPv2 by +5 _._ 95. Moreover, GeoZe achieves 59 _._ 34%, 46 _._ 00%, and 39 _._ 87% on the three variations of ScanObjectNN. The margin with respect to PointCLIPv2 is significant, i.e., +18 _._ 03, +8 _._ 13, and +4 _._ 37 accuracy, respectively. In Fig. 3, we report a t-SNE comparison between the class representations extracted with PointCLIPv2 and GeoZe. We quantify t-SNE clusters with clustering metrics: Silhouette Coefficient, Inter-cluster Distance, and Intra-cluster Distance. The metrics underscore the efficacy of GeoZe in more effectively separating point features from diverse categories. 

Table 1. Zero-shot 3D classification results on ModelNet40 [23], and ScanObjectNN [24] datasets in terms of percentage accuracy. GeoZe outperforms PointCLIPv2 [35] on all datasets. Keys: MN40: ModelNet40, SOO: S-OBJ-ONLY, SOB: S-OBJBG, SPB: S-PB-T50-RS. repo.: reported from the paper. repr.: reproduced by us. 

|ced by us.|||
|---|---|---|
|Method||MN40<br>SOO<br>SOB<br>SPB|
|PointNetv2 (repo.)<br>PointNetv2 (repr.)<br>GeoZe||64.2<br>50.1<br>41.2<br>35.4<br>63.3<br>41.3<br>37.9<br>35.5<br>**70.2**<br>**59.3**<br>**46.0**<br>**39.9**|
|∆Accuracy||+6.8<br>+18.0<br>+8.1<br>+4.4|




![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0006-02.png)


**----- Start of picture text -----**<br>
SC: 0.02 SC: 0.11<br> Intra: 282.92  Intra: 257.99<br>Inter: 20.98 Inter: 22.17<br>(a) PointCLIPv2 [35] (b) GeoZe<br>**----- End of picture text -----**<br>


Figure 3. T-SNE embeddings of (a) PointCLIPv2 [35] and (b) GeoZe on ModelNet40. GeoZe produces better separated and grouped clusters for different categories, as evidenced by the superior silhouette coefficient (SC) and greater inter-cluster distance (inter), alongside a smaller intra-cluster distance (intra). 

## **4.3. Part Segmentation** 

**Setting.** We evaluate GeoZe on ShapeNetPart [30] for the part segmentation task. We compare GeoZe against PointCLIPv2 [35]. ShapeNetPart includes 2,874 different objects, divided in 16 categories, and containing a total of 50 different part labels. These annotations are at point level. Following the evaluation procedure in PointCLIPv2 [35], we randomly sample 2,048 points from each point cloud. **Results.** Tab. 2 reports the results in terms of the mean intersection of union (mIoU). For PointCLIPv2, we report the results presented in the original paper and the ones we reproduced by executing their released source code. GeoZe outperforms PointCLIPv2 by +5 _._ 6, achieving 57 _._ 4 mIoU. GeoZe improves the part segmentation quality for 14 out of 16 objects types, with very large margin in the case of _chair_ (+14 _._ 6), _earphone_ (+9 _._ 8), _laptop_ (+14 _._ 4) and _mug_ (+12 _._ 2). Fig. 4 reports qualitative part segmentation examples obtained with PointCLIPv2 and GeoZe. GeoZe can effectively denoise features, thus producing more homogeneous segmented parts. There are cases, like the aeroplane, that are difficult to improve due to the high level of noise. 

## **4.4. Semantic Scene Segmentation** 

**Setting.** We evaluate GeoZe on ScanNet [5] and nuScenes [1] for the semantic scene segmentation task. ScanNet is an RGBD dataset including 2.5M views of 1,513 

indoor scenes, annotated with point-level semantic labels and comprising 20 object classes. We evaluate methods on the original validation set. nuScenes is a LiDAR dataset of 400K outdoor scene point clouds with semantic annotations for 16 different classes. We apply GeoZe to the 3D point-level features extracted with different 2D feature extractors, specifically those used in OpenScene [17], based on OpenSeg and LSeg, and that used in ConceptFusion [8]. **Results.** In Tab. 3 we report the original results of OpenScene [17] and the ones reproduced by us with the _“2D Fusion”_ -version of their approach in the indoor dataset ScanNet. _2D Fusion_ is the only training-free version of OpenScene. For ConceptFusion [8], we reproduced the results with the original code[1] . GeoZe outperforms OpenScene (OpenSeg), OpenScene (LSeg), and ConceptFusion by +4 _._ 2, +2 _._ 8, and +3 _._ 1 mIoU on average, respectively. Specifically, GeoZe outperforms OpenScene (OpenSeg) on 17 out 20 classes of ScanNet, with significant gains on _picture_ (+8.6), _toilet_ (+15.1), and _bathtub_ (+20.1) classes. GeoZe outperforms OpenScene (LSeg) on all the 20 classes, in particular for _sink_ (+9.1), and _toilet_ (+9.6) classes. GeoZe outperforms ConceptFusion on 19 out of 20 classes, with a significant margin in the case of _bathtub_ (+18.5), _picture_ (+6.0), and _sink_ (+8.4) classes. 

Tab. 4 reports the results in the outdoor dataset nuScenes. Similarly to Tab. 3, we reproduce the results of the _2D Fusion_ -version of OpenScene [17] using the OpenSeg feature extractor as it is the only one available. GeoZe outperforms OpenScene (OpenSeg) by 2.0 mIoU on average, and on 11 out of the 16 classes, with a improvement on _bus_ by 8.1 mIoU, _truck_ by 6.3 mIoU, and _drivable surface_ by 5.2 mIoU. Fig. 5 reports examples of qualitative results. GeoZe produces more homogeneous segmented regions and reduces noise, particularly at the point cloud boundaries. 

## **4.5. Ablation study** 

We use the part segmentation task on ShapeNetPart for the ablation study of GeoZe. The Supplementary Material contains additional ablation studies. 

**Effectiveness of aggregation levels.** We assess the contributions of the local, global, and superpoint-to-point aggregations, and superpoint computation and anchor projection steps: for convenience, we name these steps as L, G, SP, SC and A. Since GeoZe first clusters a point cloud into superpoints and then propagates them back to its original resolution, we adopt the propagation strategy outlined in [20] as our baseline, in short B. Fig. 6 reports the results of our ablation study. B scores 54.0 mIoU. Superpoint computation (SC) and superpoint-to-point aggregation (SP) improve the baseline by 0 _._ 7 and 1 _._ 0 mIoU, respectively. The combination SC+SP results in an improvement of 1 _._ 7 mIoU. 

1These results are different from those reported in the original paper [8] because authors only evaluated a subset of the classes and scenes. 

Table 2. Zero-shot part segmentation results on ShapeNetPart [30] in terms of mean Intersection over Union (mIoU). GeoZe outperforms PointCLIPv2’s [35] on 14 out of 16 classes. Bold denotes best performance. Key: repo.: reported from the paper. repr.: reproduced by us. 

||mIoU|Airpl.<br>Bag<br>Cap<br>Car<br>Chair<br>Earph.<br>Guitar<br>Knife<br>Lamp<br>Laptop<br>Motor.<br>Mug<br>Pistol<br>Rocket<br>Skate<br>Table|
|---|---|---|
|PointCLIPv2 (repo.)<br>PointCLIPv2 (repr.)<br>GeoZe (ours)|49.5<br>51.8<br>**57.4**|33.5<br>60.4<br>52.8<br>-<br>51.5<br>56.5<br>71.5<br>66.7<br>-<br>61.6<br>-<br>48.0<br>-<br>**49.6**<br>43.9<br>61.1<br>33.5<br>60.4<br>52.9<br>27.2<br>51.5<br>56.5<br>71.5<br>76.7<br>44.7<br>61.5<br>**31.5**<br>48.0<br>46.1<br>**49.6**<br>43.9<br>61.1<br>**33.6**<br>**69.8**<br>**59.3**<br>**34.7**<br>**66.1**<br>**66.3**<br>**72.1**<br>**78.2**<br>**44.9**<br>**75.9**<br>27.6<br>**60.2**<br>**48.6**<br>48.4<br>**46.0**<br>**63.8**|
|∆IoU|+5.6|+0.1<br>+9.4<br>+6.4<br>+7.5<br>+14.6<br>+9.8<br>+0.6<br>+1.5<br>+0.2<br>+14.4<br>-3.9<br>+12.2<br>+2.5<br>-2.3<br>+2.1<br>+2.7|




![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0007-02.png)


**----- Start of picture text -----**<br>
GT<br>PointCLIPv2<br>GeoZe<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0007-03.png)


Figure 4. Zero-shot part segmentation results on ShapeNetPart [30]. (top row) ground-truth annotations, (middle row) PointCLIPv2 [35], and (bottom row) GeoZe. Parts segmented by GeoZe are more homogeneous than those segmented by PointCLIPv2. 

Table 3. Zero-shot semantic segmentation results on ScanNet [5] in terms of mean Intersection over Union (mIoU). GeoZe outperforms the comparison methods (OpenScene [17] and ConceptFusion [8]) on all datasets. Bold font indicates best performance. 

||mIoU|Wall<br>Floor<br>Cabin.<br>Bed<br>Chair<br>Sofa<br>Table<br>Door<br>Wind.<br>Books.<br>Pict.<br>Count.<br>Desk<br>Curt.<br>Refri.<br>Sh. C.<br>Toil.<br>Sink<br>Batht.<br>Oth. F.|
|---|---|---|
|OpenScene [17] (OpenSeg)|||
|Reported<br>Reproduced<br>GeoZe (ours)|41.4<br>43.6<br>**47.8**|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**58.4 72.3**<br>33.5<br>66.2 45.4 48.6 31.7<br>**40.1**<br>50.3<br>59.8<br>23.7<br>31.9<br>25.9 63.3<br>26.2<br>40.1<br>59.8 19.9<br>69.0<br>6.1<br>**58.4** 72.0<br>**34.1**<br>**68.6 45.6 50.2 32.6**<br>39.9<br>**51.4**<br>**62.2**<br>**32.3**<br>**38.2**<br>**27.5 66.8**<br>**30.7**<br>**48.0**<br>**74.9 26.1**<br>**89.1**<br>**7.0**|
|∆IoU|+4.2|0.0<br>-0.3<br>+0.6<br>+2.4<br>+0.2<br>+1.6<br>+0.9<br>-0.2<br>+1.1<br>+2.4<br>+8.6<br>+6.3<br>+1.6<br>+3.5<br>+4.5<br>+7.9<br>+15.1 +6.2<br>+20.1<br>+0.9|
|OpenScene [17] (LSeg)|||
|Reported<br>Reproduced<br>GeoZe (ours)|50.0<br>51.9<br>**54.7**|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>72.1 81.9<br>45.0<br>74.0 60.1 55.3 49.4<br>52.4<br>53.9<br>68.6<br>27.4<br>37.2<br>42.1 56.8<br>50.9<br>0.0<br>68.7 47.9<br>62.8<br>31.0<br>**72.7 82.7**<br>**45.9**<br>**76.0 62.0 56.9 51.2**<br>**53.3**<br>**55.4**<br>**70.3**<br>**31.9**<br>**41.5**<br>**44.0 58.6**<br>**56.0**<br>**1.3**<br>**78.3 57.0**<br>**68.0**<br>**31.7**|
|∆IoU|+2.8|+0.6<br>+0.8<br>+0.9<br>+2.0<br>+1.9<br>+1.6<br>+1.8<br>+0.9<br>+1.5<br>+1.7<br>+4.5<br>+4.3<br>+1.9<br>+1.8<br>+5.1<br>+1.3<br>+9.6<br>+9.1<br>+5.2<br>+0.7|
|ConceptFusion [8]|||
|Reproduced<br>GeoZe (ours)|23.4<br>**26.5**|**17.9** 38.6<br>22.4<br>44.7 31.8 33.8 24.9<br>36.5<br>20.1<br>36.7<br>8.4<br>2.6<br>9.8<br>30.7<br>13.5<br>19.1<br>13.8 17.3<br>39.8<br>5.8<br>14.5 **39.7**<br>**23.3**<br>**46.4 32.8 35.1 25.4**<br>**38.4**<br>**23.4**<br>**42.3**<br>**14.4**<br>**3.0**<br>**10.8 34.4**<br>**16.6**<br>**22.7**<br>**17.2 25.7**<br>**58.3**<br>**6.5**|
|∆IoU|+3.1|-3.4<br>+1.1<br>+0.9<br>+1.7<br>+0.9<br>+1.3<br>+0.5<br>+1.9<br>+3.3<br>+5.6<br>+6.0<br>+0.4<br>+1.0<br>+3.7<br>+3.1<br>+3.6<br>+3.4<br>+8.4<br>+18.5<br>+0.7|



Table 4. Semantic segmentation results on nuScenes [1] in terms of mean Intersection over Union (mIoU), using OpenSeg feature extraction. GeoZe outperforms the OpenScene on 11 out of 16 classes. Bold font indicates best performance. Key: repr.: reproduced by us. 

||mIoU|Barr.<br>Bicyc.<br>Bus<br>Car<br>Const.<br>Motor.<br>Pers.<br>Traf. c.<br>Trail.<br>Truck<br>Dr. Sur.<br>Oth. Fl.<br>Sidew.<br>Terr.<br>Manm.<br>Veget.|
|---|---|---|
|OpenScene (repr.)<br>GeoZe (ours)|20.6<br>**22.6**|1.9<br>**17.1**<br>31.1<br>28.6<br>14.8<br>**20.9**<br>**18.5**<br>5.1<br>13.9<br>19.6<br>34.7<br>0.3<br>**15.2**<br>23.3<br>**39.5**<br>45.8<br>**2.0**<br>16.3<br>**39.2**<br>**35.8**<br>**15.2**<br>18.7<br>12.9<br>**7.1**<br>**15.7**<br>**25.9**<br>**49.9**<br>**1.3**<br>14.9<br>**27.0**<br>33.2<br>**47.3**|
|∆IoU|+2.0|+0.1<br>-0.8<br>+8.1<br>+7.2<br>+0.4<br>-2.2<br>-5.6<br>+2.0<br>+1.8<br>+6.3<br>+15.2<br>+1.0<br>-0.3<br>+3.7<br>-6.3<br>+1.5|




![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0008-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0008-01.png)


**----- Start of picture text -----**<br>
GT<br>OpenScene<br>GeoZe<br>**----- End of picture text -----**<br>


Figure 5. Zero-shot semantic segmentation results on ScanNet [5] using OpenSeg feature extraction (Tab. 3). (top row) ground-truth annotations, (middle row) OpenScene (OpenSeg), and (bottom row) GeoZe. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0008-03.png)


**----- Start of picture text -----**<br>
57<br>56<br>55<br>54<br>mIoU<br>B SC SP SC+SP SC+L+SP SC+G+SP SC+A+SP SC+L+G+SP Full<br>**----- End of picture text -----**<br>


Figure 6. Ablation study on GeoZe’s modules, tested on the part segmentation task. Keys: B: baseline. SC: superpoint computation. SP: superpoint-to-point aggregation. G: global aggregation. L: local aggregation. A: anchor projection. Full: the whole model. 

Local aggregation (SC+L+SP) leads to an improvement of 0 _._ 5 mIoU over SC+SP, underlining the importance of local smoothing. Global aggregation (SC+G+SP) improves 0 _._ 8 mIoU over SC+SP, indicating the positive effect of relying on global semantic context. Anchor projection (SC+A+SP) improves 0 _._ 8 mIoU over SC+SP, while the combination of SC+L+G+SP results in 57 _._ 1 mIoU. The complete GeoZe reaches 57 _._ 4 mIoU. 

**Impact of feature types.** Tab. 5 reports our analysis of the importance of the different types of features in the aggregation process. Our experiments indicate that incorporating geometric, semantic and coordinate information significantly enhances the effectiveness of local aggregation. Similarly, global aggregation based on geometric and semantic features allows for a great increase in performance. It improves the segmentation accuracy by nearly +1.2 mIoU (55.7% vs. 56.9%). 

## **4.6. Computational analysis** 

We compare the processing time of GeoZe against the aggregation approaches of the evaluated methods and report the average time increment per point cloud. On ShapeNetPart, time measured for PointCLIPv2 is 8.73ms, and GeoZe 

Table 5. Ablation study evaluating the influence of geometric information on part segmentation within the ShapeNetPart dataset. Keys: Coors: coordinate, CLIP: VLM representation, GA: global aggregation, LA: local aggregation. 

||Feature|Feature Combinations|
|---|---|---|
|LA<br>CLIP [35]<br>✓<br>✓<br>✓<br>✓<br>✓<br>Coors<br>✓<br>✓<br>✓<br>FPFH [22]<br>✓<br>✓<br>✓|||
|GA|CLIP [35]<br>FPFH [22]|✓<br>✓<br>✓<br>✓<br>✓<br>✓|
||mIoU|55.7<br>56.2<br>56.2<br>56.4<br>55.7<br>56.5<br>56.9<br>57.4|



is 45.01ms. Compared to OpenScene, GeoZe requires an additional 51.34ms on the ScanNet, and it necessitates an extra 62.43ms on the nuScenes. 

## **5. Conclusions** 

We presented the first training-free aggregation technique that leverages the point cloud’s 3D geometric structure to improve the quality of the transferred VLM representations. GeoZe is designed to iteratively aggregate representations at local level first and global level then. VLM representations are extracted from a CLIP model [21], while geometric representations are extracted from the FPFH algorithm [22]. We carried out an extensive evaluation on three downstream tasks: classification, part segmentation, and semantic segmentation. We reported the results on both synthetic/realworld, and indoor/outdoor datasets, and showed that GeoZe achieves new state-of-the-art results in all benchmarks. 

**Limitations.** GeoZe depends on the quality of VLM representations, it may not accurately handle features of objects that constitute a small portion of the 3D scene, and it increases the inference time compared to a naive aggregation. 

## **References** 

- [1] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In _CVPR_ , 2020. 2, 6, 7 

- [2] Xia Chen, Jianren Wang, David Held, and Martial Hebert. Panonet3d: Combining semantic and geometric understanding for lidar point cloud detection. In _3DV_ , 2020. 3 

- [3] Ali Cheraghian, Shafin Rahman, Dylan Campbell, and Lars Petersson. Mitigating the hubness problem for zero-shot learning of 3d objects. In _BMVC_ , 2019. 2 

- [4] Ali Cheraghian, Shafin Rahman, Townim F Chowdhury, Dylan Campbell, and Lars Petersson. Zero-shot learning on 3d point cloud objects and beyond. _IJCV_ , 2022. 2 

- [5] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In _CVPR_ , 2017. 2, 6, 7, 8 

- [6] Deepti Hegde, Jeya Maria Jose Valanarasu, and Vishal Patel. Clip goes 3d: Leveraging prompt tuning for language grounded 3d recognition. In _ICCV_ , 2023. 2 

- [7] Tianyu Huang, Bowen Dong, Yunhan Yang, Xiaoshui Huang, Rynson WH Lau, Wanli Ouyang, and Wangmeng Zuo. Clip2point: Transfer clip to point cloud classification with image-depth pre-training. In _ICCV_ , 2023. 2 

- [8] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. ConceptFusion: Open-set Multimodal 3D Mapping. In _RSS_ , 2023. 1, 2, 3, 5, 6, 7 

- [9] Shu Kong and Charless C Fowlkes. Recurrent pixel embedding for instance grouping. In _CVPR_ , 2018. 4 

- [10] Loic Landrieu and Martin Simonovsky. Large-scale point cloud semantic segmentation with superpoint graphs. In _CVPR_ , 2018. 2 

- [11] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_ , 2023. 1 

- [12] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. _NeurIPS_ , 2018. 2 

- [13] Yongcheng Liu, Bin Fan, Shiming Xiang, and Chunhong Pan. Relation-shape convolutional neural network for point cloud analysis. In _CVPR_ , 2019. 2 

- [14] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. In _ICLR_ , 2022. 3 

- [15] Seyed Saber Mohammadi, Yiming Wang, and Alessio Del Bue. Pointview-gcn: 3d shape classification with multiview point clouds. In _ICIP_ , 2021. 3 

- [16] Chunghyun Park, Yoonwoo Jeong, Minsu Cho, and Jaesik Park. Fast point transformer. In _CVPR_ , 2022. 2 

- [17] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In _CVPR_ , 2023. 1, 2, 3, 5, 6, 7 

- [18] Gabriel Peyr´e, Marco Cuturi, et al. Computational optimal transport: With applications to data science. _Foundations and Trends® in Machine Learning_ , 2019. 4 

- [19] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In _CVPR_ , 2017. 2 

- [20] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. _NeurIPS_ , 2017. 2, 3, 4, 6 

- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 1, 3, 8 

- [22] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast point feature histograms (fpfh) for 3d registration. In _ICRA_ , 2009. 1, 2, 3, 5, 8 

- [23] Abhishek Sharma, Oliver Grau, and Mario Fritz. Vconvdae: Deep volumetric shape learning without object labels. In _ECCV_ , 2016. 6 

- [24] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In _ICCV_ , 2019. 2, 5, 6 

- [25] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. _ACM TOG_ , 2019. 2 

- [26] Xin Wei, Ruixuan Yu, and Jian Sun. View-gcn: View-based graph convolutional network for 3d shape analysis. In _CVPR_ , 2020. 3 

- [27] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling. _NeurIPS_ , 2022. 3 

- [28] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In _CVPR_ , 2015. 2, 5 

- [29] Le Xue, Mingfei Gao, Chen Xing, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In _CVPR_ , 2023. 1, 2 

- [30] Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. A scalable active framework for region annotation in 3d shape collections. _ACM TOG_ , 2016. 2, 6, 7 

- [31] Zehao Yu, Jia Zheng, Dongze Lian, Zihan Zhou, and Shenghua Gao. Single-image piece-wise planar 3d reconstruction via associative embedding. In _CVPR_ , pages 1029– 1037, 2019. 4 

- [32] Yihan Zeng, Chenhan Jiang, Jiageng Mao, Jianhua Han, Chaoqiang Ye, Qingqiu Huang, Dit-Yan Yeung, Zhen Yang, Xiaodan Liang, and Hang Xu. Clip2: Contrastive languageimage-point pretraining from real-world point cloud data. In _CVPR_ , 2023. 2 

- [33] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In _CVPR_ , 2022. 1, 2 

- [34] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In _ICCV_ , 2021. 2 

- [35] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. PointCLIPv2: Prompting CLIP and GPT for Powerful 3D Openworld Learning. In _ICCV_ , 2023. 1, 2, 3, 5, 6, 7, 8 

# **Geometrically-driven Aggregation for Zero-shot 3D Point Cloud Understanding Supplementary Materials** 

Guofeng Mei Luigi Riz Yiming Wang Fabio Poiesi Bruno Kessler Foundation, Via Sommarive, 18, 38123 Trento, Italy 

_{_ gmei,luriz, ywang, poiesi _}_ @fbk.eu 

In this supplementary file, we provide additional implementation details in Sec. A and further experimental analyses in Sec. B. 

## **A. Implementation Details** 

## **A.1. VLM representation extraction** 

For classification tasks on both the ModelNet40 [8] and ObjectScanNN [9] datasets, we adhere to the method used in PointCLIPv2 [12], sampling 1,024 points from each point cloud. These points are then projected into depth images from 10 different views for the extraction of VLM representations. As in PointCLIPv2 [12], we use the ViTB/16 [10] model as the visual encoder within the CLIP framework, which comprises 12 layers of multi-head selfattention (MHSA). We extract the VLM representations corresponding to image patches during the attention process at the final MHSA layer. These VLM representations are then subjected to bilinear interpolation, a technique we utilize to upscale the VLM representations to the original size of the image, specifically to 224 _×_ 224. Given that a single view projection captures only a partial point cloud, we utilize multi-view back projection to ensure thorough predictions for all points in the cloud. For points observable from multiple views, we perform linear interpolation of the VLM representations, fine-tuning this process based on the varying weights of the different views. These VLM representations are then fed into our GeoZe to obtain enhanced point-level representations. Subsequently, max-pooling is applied to generate the global features. Additionally, we extract global features using the ViT-B/16 model from each view, consistent with PointCLIPv2. 

For the part segmentation task on the ShapeNetPart [11] dataset, we sample 2,048 points per point cloud. These points are then projected onto depth images from 10 different viewpoints for VLM representation extraction using the ViT-B/16 model. After extraction, these VLM representations undergo bilinear interpolation for upsampling to their original image size. Similar to the classification process, we apply multi-view back projection to the point cloud us- 

Table 1. Hyper-parameters configurations for different datasets. 

|Dataset|Network<br>Γ<br>¯_N_<br>_K_1<br>_K_2|
|---|---|
|ModelNet40<br>ObjectScanNN<br>ShapeNetPart|PointCLIPv2[12]<br>16<br>256<br>32<br>24<br>PointCLIPv2[12]<br>16<br>256<br>32<br>24<br>PointCLIPv2[12]<br>16<br>256<br>32<br>24|
|ScanNet|OpenSeg [3]<br>8<br>3000<br>48<br>32<br>LSeg [5]<br>8<br>3000<br>48<br>32<br>ConceptFusion[4]<br>8<br>3000<br>48<br>32|
|nuScenes|LSeg [5]<br>8<br>2400<br>48<br>32|



ing the VLM representations. These VLM representations are subsequently processed through our GeoZe to achieve enhanced point-level representations. 

In assessing the semantic segmentation performance on the ScanNet [2] and nuScenes [1] datasets, we utilize the VLM representations that are provided by OpenScene [6]. To ensure consistency and facilitate a fair comparison, we adhere to the standard voxel size of 0.02m as in OpenScene. These VLM representations are then processed using our GeoZe, which facilitates the enhancement of point-level representations. 

## **A.2. Parameters** 

Tab. 1 presents the dataset-specific hyperparameters including the number of iterations (Γ), the number of superpoints ( _N_[¯] ), the number of points ( _K_ 1) used for computing similarity, and the number of neighboring points ( _K_ 2) for local aggregation. Specifically, increasing the number of superpoints from _[N]_ 8[to][a][maximum][of] _[N]_[¯] _[≤][N]_ 4[can][slightly][im-] prove results, as denoted by _[N]_ 8 _[≤][N]_[¯] _[≤][N]_ 4[.] _[ N]_[is the number] points of a point cloud. Our experiments suggest that optimal accuracy is achieved when the number of superpoints is maintained between one-eighth and one-third of the original point cloud size. Balancing both accuracy and time complexity, the parameters we adopted in our experiments are as listed in Tab. 1. 

1 

## **A.3. Geometric representation (FPFH) extraction** 

In our experiments, we compute FPFH [7] features for all points. To improve the computation time, we first downsample _M_ reference points. Then, we sample _K_ 3 neighboring points for each point from the _M_ reference points within a radius _r_ 1 for estimating the normals and _K_ 4 neighboring points for each point from _M_ reference points within a radius _r_ 2 for estimating FHFH. The details are reported in Tab. 2. 

Table 2. Hyper-parameters configurations for FPFH computation on different datasets. 

|Dataset|¯<br>_M_<br>_K_3<br>_K_4<br>_r_1<br>_r_2|
|---|---|
|ModelNet40<br>ObjectScanNN<br>ShapeNetPart|512<br>32<br>100<br>0.04<br>0.08<br>512<br>32<br>100<br>0.04<br>0.08<br>512<br>32<br>100<br>0.04<br>0.08|
|ScanNet|4800<br>32<br>100<br>0.05<br>0.10|
|nuScenes|5200<br>32<br>100<br>0.05<br>0.10|



## **B. Additional Experimental Analyses** 

## **B.1. VLM representation anchors** 

VLM representation anchor is designed to find important VLM representations that are more suitable for semantic alignment. In our effort to cover a broad spectrum of categories and improve the anchors’ general applicability, we conducte experiments using VLM representations derived from 32 point clouds. This idea is inspired by the concept of a memory bank for constrastive learning. We also produce geometric representation anchors, employing the same weight parameters as those used for calculating the VLM representation anchors. The purpose of these geometric representation anchors is to aid the anchor projection process, thereby mitigating issues of semantic misalignment. This is achieved by searching for the closest VLM representation anchor for each point, taking into account both the similarities of VLM representations with VLM representation anchors and geometric representations with geometric representation anchors. Fig. 1 displays the results of zero-shot semantic segmentation on ScanNet [2], using the OpenSeg as the feature extractor. This figure clearly shows the improved performance in zero-shot semantic segmentation achieved by incorporating VLM representation anchors (presented in the bottom row), especially when compared to the method that does not utilize VLM representation anchors, as seen in the third row of the figure. 

## **B.2. Classification visualization** 

In Fig. 2, we report a t-SNE comparison between the class representations extracted with PointCLIPv2 and GeoZe on ScanObjectNN [9] (S-PB-T50-RS). We quantify t-SNE 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0012-09.png)


**----- Start of picture text -----**<br>
GT<br>OpenScene<br>GeoZe/wo<br>GeoZe<br>**----- End of picture text -----**<br>


Figure 1. Zero-shot semantic segmentation results on ScanNet using OpenSeg feature extraction. (top row) ground-truth annotations, (second row) OpenScene (OpenSeg) [6], (third row) GeoZe/wo (OpenSeg) without using VLM representation anchors, and (bottom row) GeoZe (OpenSeg). ‘wo’ indicates the absence of VLM representation anchors. 

clusters with three clustering metrics: Silhouette Coefficient, Inter-cluster Distance, and Intra-cluster Distance. From these metrics, we can confirm the efficacy of GeoZe in better separating point features of diverse categories compared to PointCLIPv2. 

Fig. 3 compares the same global features using traditional clustering metrics for a more detailed assessment of the differences between GeoZe and PointCLIPv2. For a comprehensive analysis of intra- and inter-cluster statistics, we consider six extrinsic clustering measures, that explicitly compare classification predictions with ground-truth annotations. The Adjusted Rand Index (ARI) evaluates the similarity of cluster assignments through pairwise comparisons. The Adjusted Mutual Information (AMI) assesses the agreement of cluster assignments. Homogeneity (H) gauges the proportion of instances from a single class in a cluster, akin to Precision. Completeness (C) measures the proportion of a given class’s instances assigned to the same cluster, similar to Recall. The V-measure (V) quantifies clustering correctness using conditional entropy analysis. The Fowlkes-Mallows score (FM) evaluates clustering accuracy through the geometric mean of pairwise Precision and Recall. Higher scores in all these metrics indicate better performance. In Fig. 3, the histogram values are normalized, with the maximum value for each score set to 1. GeoZe outperforms PointCLIPv2 in all metrics. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0013-00.png)


**----- Start of picture text -----**<br>
SC: 0.00 SC: 0.01<br> Intra: 94.51  Intra: 91.91<br>Inter: 5.72 Inter: 7.35<br>(a) PointCLIPv2 [12] (b) GeoZe<br>**----- End of picture text -----**<br>


Figure 2. T-SNE embeddings of (a) PointCLIPv2 [12] and (b) GeoZe on ScanObjectNN [9] (S-PB-T50-RS). GeoZe produces better separated and grouped clusters for different categories, as evidenced by the superior silhouette coefficient (SC) and greater inter-cluster distance (inter), alongside a smaller intra-cluster distance (intra). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0013-02.png)


**----- Start of picture text -----**<br>
0.6<br>GeoZe<br>0.5 PointCLIPv2<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>ARI AMI H C V FM<br>Scores<br>**----- End of picture text -----**<br>


Figure 3. Comparison of clustering metrics (the higher the better) on ModelNet40 [8]. Metrics key: ARI: adjusted rand index, AMI: adjusted mutual information, H: homogeneity score, C: completeness score, V: V-measure, FM: Fowlkes-Mallows score. 

## **B.3. Segmentation results on nuScenes** 

Fig. 4 showcases a range of qualitative results obtained using our method, GeoZe, on the nuScenes outdoor dataset [1]. The effectiveness of GeoZe is underscored by its ability to produce more semantically coherent segmented regions, a property primarily attributed to our novel clustering technique. Additionally, the synergy of local and global aggregation techniques bolsters the VLM representation, making it more geometrically aware. Another reason is the integration of geometric representation assignment, which plays a pivotal role in reducing semantic misalignment during anchor projection. The positive effects of our method are particularly visible at the boundaries of point clouds, where GeoZe substantially lowers noise levels, outperforming the compared method OpenScene. This noise reduction is especially effective owing to the distinct geometric structures commonly found at these boundary zones. 

## **B.4. VLM representation guided clustering** 

In this section, we demonstrate the enhancement in semantic clustering performance achieved by integrating geometric information into VLM representation. We use the clustering scores to compute each cluster’s prototypical representation. Specifically, these prototypes are weighted averages of VLM representations, based on the clustering scores. Subsequently, each point is assigned to the prototype of its corresponding cluster. PCA projection is then applied to visualize the clustering results. Fig. 5 showcases some qualitative clustering results on ShapeNetPart [11] using different point-level coordinates (Coord.) and representations. The top row demonstrates results using only coordinates for clustering, while the second row combines the coordinates with VLM representations. The third row integrates coordinates and geometric representations (FPFH), offering more meaningful partitioning compared to using just coordinates or coordinates with VLM representations. The bottom row features GeoZe, which considers both coordinates and geometric representations (FPFH) for clustering, guided by VLM representation similarity, achieving the best clustering results (same parts tend to share colors). 

## **References** 

- [1] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In _CVPR_ , 2020. 1, 3, 4 

- [2] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0014-00.png)


**----- Start of picture text -----**<br>
GT OpenScene GeoZe<br>**----- End of picture text -----**<br>


Figure 4. Zero-shot semantic segmentation results on nuScenes [1] using OpenSeg as feature extractor. (left column) ground-truth annotations, (middle column) OpenScene (OpenSeg) [6], and (right column) GeoZe. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0015-00.png)


**----- Start of picture text -----**<br>
Coord.<br>Coord.+VLM<br>Coord.+FPFH<br>GeoZe<br>**----- End of picture text -----**<br>



![](raw/papers/2_LLM_NLP/LLM_paper_list/zero-shot/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding_images/cvpr_2024_Geometrically-driven_Aggregation_for_Zero-shot_3D_Point_Cloud_Understanding.pdf-0015-01.png)


Figure 5. Visualization of clustering results on ShapeNetPart [11] using various sources: (top row) coordinates only; (second row) coordinates with VLM representations; (third row) coordinates and geometric representations (FPFH); (fourth row) coordinates and geometric representations (FPFH), guided by VLM representation similarity. Coord. represents Coordinates. 

Richly-annotated 3d reconstructions of indoor scenes. In _CVPR_ , 2017. 1, 2 

- [3] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In _ECCV_ , pages 540–557. Springer, 2022. 1 

- [4] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. ConceptFusion: Open-set Multimodal 3D Mapping. In _RSS_ , 2023. 1 

- [5] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In _ICLR_ , 2022. 1 

- [6] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In _CVPR_ , 2023. 1, 2, 4 

Thanh Nguyen, and Sai-Kit Yeung. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In _ICCV_ , 2019. 1, 2, 3 

   - [10] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _NeurIPS_ , 2017. 1 

   - [11] Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. A scalable active framework for region annotation in 3d shape collections. _ACM TOG_ , 2016. 1, 3, 5 

   - [12] Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. PointCLIPv2: Prompting CLIP and GPT for Powerful 3D Openworld Learning. In _ICCV_ , 2023. 1, 3 

- [7] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast point feature histograms (fpfh) for 3d registration. In _ICRA_ , 2009. 2 

- [8] Abhishek Sharma, Oliver Grau, and Mario Fritz. Vconvdae: Deep volumetric shape learning without object labels. In _ECCV_ , 2016. 1, 3 

- [9] Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, 

