# **XMask3D: Cross-modal Mask Reasoning for Open Vocabulary 3D Semantic Segmentation** 

**Ziyi Wang** _[∗]_ **Yanbo Wang** _[∗]_ **Xumin Yu Jie Zhou Jiwen Lu** _[†]_ Department of Automation, Tsinghua University, China `{wziyi22, wyb23, yuxm20}@mails.tsinghua.edu.cn;` 

```
{lujiwen,jzhou}@tsinghua.edu.cn
```

## **Abstract** 

Existing methodologies in open vocabulary 3D semantic segmentation primarily concentrate on establishing a unified feature space encompassing 3D, 2D, and textual modalities. Nevertheless, traditional techniques such as global feature alignment or vision-language model distillation tend to impose only approximate correspondence, struggling notably with delineating fine-grained segmentation boundaries. To address this gap, we propose a more meticulous mask-level alignment between 3D features and the 2D-text embedding space through a cross-modal mask reasoning framework, **XMask3D** . In our approach, we developed a mask generator based on the denoising UNet from a pre-trained diffusion model, leveraging its capability for precise textual control over dense pixel representations and enhancing the open-world adaptability of the generated masks. We further integrate 3D global features as implicit conditions into the pre-trained 2D denoising UNet, enabling the generation of segmentation masks with additional 3D geometry awareness. Subsequently, the generated 2D masks are employed to align mask-level 3D representations with the vision-language feature space, thereby augmenting the open vocabulary capability of 3D geometry embeddings. Finally, we fuse complementary 2D and 3D mask features, resulting in competitive performance across multiple benchmarks for 3D open vocabulary semantic segmentation. Code is available at `https://github.com/wangzy22/XMask3D` . 

## **1 Introduction** 

As the integration of vision and language in deep learning continues to expand, text descriptions are increasingly utilized in visual generation [37, 50, 39, 35, 26, 40] and perception [36, 22, 23, 13, 45, 12] tasks. This integration enhances the adaptability of models in real-world applications and improves user experiences in customized artificial intelligence systems. Open vocabulary 3D semantic segmentation exemplifies a perception task that is trained on _base_ categories and demands robust extrapolation capabilities to discriminate fine-grained geometry in _novel_ categories that are invisible during training. The base and novel classes are only linked by the shared open vocabulary within the language space. However, constructing a shared 3D-text space while maintaining precise, modality-specific representation remains a significant challenge. Successfully addressing this issue would advance open vocabulary 3D semantic segmentation, facilitating virtual reality interactions and manipulations, and thereby contributing to the development of user-friendly robotics and autonomous driving technologies. 

Existing approaches for open vocabulary 3D perception predominantly aim to bridge the gap between 3D and text representations by using the 2D modality as an intermediary. One line of research [34, 

> _∗_ Equal contribution. †Corresponding author. 

38th Conference on Neural Information Processing Systems (NeurIPS 2024). 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0002-00.png)


**----- Start of picture text -----**<br>
3D Model with  Global  3D-2D-Text Alignment<br>Cross-Modal<br>3D Geometry  Fusion Block<br>Extractor<br>2D-to-3D  3D Result Fused Result<br>3D-to-2D Mask  Mask Loss<br>Novel Category<br>Condition<br>Toilet Desk<br>2D Open Mask<br>Generator Base Category<br>Wall Bathtub<br>2D Result Ground Truth<br>**----- End of picture text -----**<br>


Figure 1: **The overall framework of XMask3D.** The 3D model with only _coarse_ 3D-2D-text alignment struggles to segment novel categories with accurate boundaries. We propose to incorporate a 2D open mask generator conditioned on global 3D geometry features to create geometry-aware segmentation masks of novel categories. Then we apply fine-grained mask-level regularization on 3D features, thereby enhancing the dense open vocabulary capability of the 3D model. The cross-modal fusion block leverages the strengths of both branches to achieve optimal results. 

11, 47, 16] suggests aligning 3D embeddings with vision-language embedding spaces through global, patch-wise, or point-wise contrastive loss, while another line [49, 28, 44] investigates distilling open vocabulary knowledge from foundational vision-language models into 3D models. However, global feature alignment and model distillation techniques tend to overlook fine-grained 3D geometric details and produce coarse results. Conversely, point-wise contrastive learning is prone to noise and outliers. Although patch-wise feature alignment offers a compromise, a single patch in 2D may correspond to multiple unrelated and discontinuous regions in 3D, which can be misleading and ineffective. 

In this paper, we propose a more precise and consistent mask-level alignment between 3D features and the 2D-text embedding space, achieved through our proposed cross-modal mask reasoning method. The proposed XMask3D model comprises a 3D branch, a 2D branch, and a fusion block. The 3D branch, adaptable as any point cloud segmentation model, excels in capturing geometric features but struggles with novel category extrapolation. Conversely, the 2D branch serves as a mask generator, which predicts masks with embeddings aligned to the vision-language feature space but lacks 3D spatial perception capabilities. The fusion block allows these two branches to complement each other. Specifically, we propose utilizing the denoising UNet of a pre-trained text-to-image diffusion model with advanced vision-language modeling capabilities as the 2D mask generator. Diffusion model’s exceptional control over text-driven image generation demonstrates strong potential for creating fine-grained segmentation masks of novel categories. To promote thorough interaction between the two modality branches, we propose three mask-level techniques: (1) **3D-to-2D Mask Generation.** We condition the denoising mask generator on global point cloud features, producing geometry-aware masks that are better suited for transferring to the 3D modality. (2) **2D-to-3D Mask Regularization.** We apply mask-level regularization on 3D features to align with the vision-language embedding space. This enhances the open vocabulary capability of 3D features for novel categories while preserving fine-grained geometric information. (3) **3D-2D Mask Feature Fusion.** We merge mask features from both modalities in the fusion block, enhancing the synergy between 2D and 3D features. 

We conduct extensive experiments on multiple benchmarks of different datasets, including ScanNet20 [9], ScanNet200 [38], and S3DIS [1] datasets, to evaluate the effectiveness of our proposed method. XMask3D demonstrates competitive performance across all benchmarks. Additionally, we perform thorough ablation studies and provide intuitive visualizations to showcase the contribution of each proposed mask-level technique. In conclusion, the contributions of this paper can be summarized as follows: 

- We propose a novel XMask3D framework that, for the first time, leverages the denoising UNet of a generative text-to-image diffusion model for open vocabulary 3D perception. 

- We introduce 3D-to-2D mask generation, 2D-to-3D mask regularization, and 3D-2D mask feature fusion techniques to enhance meticulous mask-level 3D-2D-text feature alignment and strengthen cross-modal feature synergy. 

- We demonstrate the effectiveness of XMask3D on multiple benchmarks of various datasets and show outstanding performance. Ablation studies further convince the contribution of each proposed mask-level technique. 

2 

## **2 Related Work** 

## **2.1 2D Open Vocabulary Segmentation** 

Open vocabulary perception is a recently emerged research problem that focuses on enabling perception models to recognize novel categories that are invisible during supervised training, relying solely on a shared language vocabulary with the base categories. The key to addressing this problem lies in the Vision-Language model, which creates a shared embedding space for images and texts. Based on the types of Vision-Language models, previous literature on 2D open vocabulary segmentation can be broadly divided into two approaches: utilizing Vision-Language perception models like CLIP [36] or leveraging Vision-Language generation models like the diffusion model [18, 37]. 

Although traditional Vision-Language perception models like CLIP are primarily designed for classification, open vocabulary segmentation can also be viewed as a dense classification task. Consequently, several studies have proposed aligning dense image features with 2D-text embeddings, a concept pioneered by LSeg [23]. Successor models have introduced various techniques to enhance the feature alignment process, including attention-based combinations [30, 12], mask embedding decoupling [10], side network injection [46, 24], and cross-modal aggregation [6]. 

Generative-based methods utilize Vision-Language generation models, such as diffusion models, to produce segmentation masks that can be extrapolated to open vocabulary categories. Since the diffusion model can generate semantically meaningful images based on text conditions, its intermediate features effectively represent the vision-language embedding space. ODISE [45] first proposed using the intermediate features from the denoising UNet of a pre-trained diffusion model as input to a mask generator for segmentation. Other works [25, 21, 41] leverage the strong generative capabilities of the diffusion model to create prototypes or augmented image-mask pairs, thereby enhancing the open vocabulary capacity of the segmentation model from a data perspective. 

## **2.2 3D Open Vocabulary Segmentation** 

In 3D vision, Semantic Abstraction [15] opens up the avenue to leverage Vision-Language models for open-world 3D scene understanding. Subsequent studies have primarily developed two types of methods to address 3D open vocabulary segmentation: feature alignment and model distillation. 

The principle of feature alignment methods is to explicitly pull 3D representations towards the visionlanguage embedding space, using the 2D modality as a mediator to establish 3D-text relationships. OpenScene [34] employs a cosine similarity loss between point cloud features and image CLIP features, integrating them for open vocabulary perception. PLA [11] introduces hierarchical 3D caption pairs to progressively align scene-level, view-level, and entity-level features with the CLIP feature space in a coarse-to-fine manner. Its successor, RegionPLC [47], further introduces regionlevel captions with sliding windows and object bounding boxes, while CLIP-FO3D [49] similarly divides super-pixels for finer feature alignment. UniM-OV3D [16] utilizes a pre-trained point-text model, PointBind [14], to enforce uni-modality representation learning of point clouds, images, depth maps, and texts. OV3D [20] proposes leveraging foundation models to establish point-entityText associations through pixel, thereby enhancing open-vocabulary recognition within the 3D domain. 

Model distillation methods typically involve selecting a foundational Vision-Language model and transferring its knowledge to a 3D network using paired point clouds and image data. Seal [28] introduces spatial contrastive learning and temporal consistency regularization to distill vision foundation models for point cloud sequence segmentation in an open vocabulary setting. 3DOVS [27] aims to distill CLIP [36] and DINO [2, 33] into a neural radiance field [32] using novel alignment losses for 3D perception. Xiao et al. [44] propose object-level and voxel-level distillation losses for fine-grained 3D open vocabulary panoptic segmentation. 

## **3 Approach** 

## **3.1 Overview** 

The detailed architecture of XMask3D is depicted in Figure 2. It consists of three components: a 3D geometry extraction branch, a 2D mask generation branch, and a 3D-2D feature fusion module. The 3D geometry extraction branch is an encoder-decoder segmentation network, which can be 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0004-00.png)


**----- Start of picture text -----**<br>
Point Cloud Encoder Decoder Fusion Block<br>Mask Loss<br>Implicit 3D Captioner<br>CLIP Image Encoder CLIP Text Encoder<br>Condition<br>Category Labels Results<br>Image Mask Generator<br>Mask Feat.<br>Mask Contour<br>3D Feat.<br>Fused Feat. CLIP I/T Feat.<br>Diffusion UNet<br>**----- End of picture text -----**<br>


Figure 2: **The detailed architecture of XMask3D.** We introduce an auxiliary 2D branch, which utilizes global point cloud features as conditional input to generate open vocabulary masks. The contour of the mask is utilized for regularization at the mask level on 3D features, and the embeddings of the mask are fused with the 3D features to enhance cross-modal complementarity. 

implemented using any off-the-shelf architectures such as sparse convolution networks [7, 8] or Transformer-based networks [51, 43, 42]. These frameworks are specifically designed to extract geometric features _F_ 3d _∈_ R _[N][×][C]_ from 3D point clouds, where _N_ and _C_ represent the numbers of points and feature dimensions, respectively. However, _F_ 3d is discriminative only on base categories trained with supervision, relying on a pre-defined classifier implemented as a fully connected layer or a multi-layer perceptron. Therefore, it’s essential for _F_ 3d to be aligned with the vision-language feature space to enable unbounded open vocabulary segmentation, which is performed via perpoint similarity comparisons between _F_ 3d and CLIP [36] text embeddings of given category names: _F_ text = _E_ CLIPT ( **C** name) _∈_ R _[L][×][C]_ , where _L_ represents the number of categories, _E_ CLIPT denotes the CLIP text encoder, and **C** name stands for the category name such as _Table, Chair_ . 

Previous literature on feature alignment techniques varies from global or point-level contrastive learning to vision-language model distillation. However, none of these methods simultaneously achieves fine-grained precision and robustness to outliers. To address this limitation, we introduce a cross-modal mask reasoning method that performs mask-level feature alignment. Specifically, we propose a 2D mask generation branch in conjunction with the 3D branch to generate 2D masks with open vocabulary capability and use these masks to regularize _F_ 3d. Detailed information on mask generation and mask regularization can be found in Section 3.2 and Section 3.3, respectively. 

From the 2D mask generator, we obtain the 2D mask embeddings _G_ 2d _∈_ R _[M][×][C]_ and binary mask maps _M_ 2d _∈_ R _[H][×][W][ ×][M]_ , where _M_ represents the number of candidate masks, and _H_ and _W_ denote the height and width of the input images. Using the camera intrinsic matrix _K ∈_ R[3] _[×]_[3] and the view projection matrix _V ∈_ R[4] _[×]_[4] , we can establish associations between pixels in image _I_ and surface points in the corresponding point cloud _P_ . This allows us to derive back-projected 3D binary masks _M_ 3d _∈_ R _[N][ ′][×][M]_ from _M_ 2d. Here, _N[′]_ signifies the number of points that correspond to image pixels, following the relationship _N[′] < N, N[′] < H × W_ . Subsequently, we derive a pseudo mask feature _F_ 2d = _M_ 3d _· G_ 2d _, F_ 2d _∈_ R _[N][′][×][C]_ , with the subscript 2d indicating its origin from the 2D branch. 

Having obtained _F_ 3d from the 3D geometry extraction branch and _F_ 2d from the 2D mask generation branch, we implement a 3D-2D fusion block to combine these cross-modal and complementary features, resulting in _F_ fuse _∈_ R _[N][×][C]_ . Detailed information on this process can be found in Section 3.4. The final open vocabulary 3D semantic segmentation output _O ∈_ R _[N][×][L]_ is then calculated via: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0004-06.png)


Detailed training objectives can be found in Section 3.5. 

## **3.2 3D-to-2D Mask Generation** 

**Design Insights.** An optimal 2D branch is expected to exhibit robust open vocabulary capabilities, enabling it to predict accurate masks for novel categories. To this end, we employ the denoising UNet of the renowned text-to-image diffusion model [37] to extract features from the well-established text-2D embedding space, followed by a mask generator to convert features into segmentation masks. 

4 

We favor the generative diffusion model over the commonly adopted CLIP [36] model for two primary reasons. Indicated by Prompt-to-Prompt [17], the cross-attention maps from the intermediate layers of the diffusion model exhibit a high correlation with text concepts. Consequently, a welltrained diffusion model constructs a superior vision-language feature space, which can be effectively leveraged for open vocabulary perception. Moreover, the diffusion model provides precise text control over dense pixel generation, demonstrating a higher potential for generating fine-grained segmentation masks compared to the CLIP model, which relies on coarse global feature alignment. 

**Preliminary: Open Vocabulary Mask Generation with diffusion.** The inference process of a text-to-image diffusion model [37] begins with a random Gaussian noise and a conditional text embedding, generating high-quality images through iterative denoising steps. ODISE [45] was the first to propose the use of the denoising UNet of a pre-trained diffusion model for open vocabulary _2D_ segmentation. Given an input image _x_ , a noisy image _xt_ is first sampled at time step _t_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0005-02.png)


where _α_ ¯ _t_ =[�] _[t] k_ =1 _[α][k]_[, and] _[ α]_[1] _[, . . . , α][T]_[are pre-defined noise schedule.][Then the diffusion model’s] visual representation _f_ can be computed via the denoising step: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0005-04.png)


where the denoising UNet is the building block of the diffusion model, MLP stands for multi-layer perceptron, and _V_ is a frozen CLIP image encoder to encode _x_ into the vision-language embedding space. _V_ ( _x_ ) is the implicit caption embedding which serves as the text condition input to the diffusion model. Subsequently, the mask generator, implemented with Mask2Former [4, 3], uses the feature _f_ to produce _M_ class-agnostic binary masks _M_ 2d and their corresponding mask embedding features _F_ mask. Since _f_ is highly representative of the vision-language feature space, the model is inherently capable of generating open vocabulary segmentation masks and embeddings. The experimental results of ODISE strongly confirm this hypothesis. 

**Geometry-aware Mask Generation.** In XMask3D, the 2D mask generation branch is implemented by a variant of the ODISE model. We propose an _Implicit 3D Captioner_ that takes the global 3D feature _f_ 3d _∈_ R[1] _[×][C][g]_ from the 3D encoder as input, and predicts the implicit condition embedding to be injected into the diffusion model. Then Equation 3 can be replaced by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0005-07.png)


where _E_ represents the encoder of the point cloud segmentation model, and _Cg_ denotes the feature dimension of the global point cloud feature. The rationale behind this design is twofold. First, since MLP _◦ f_ 3d serves as the text condition for the pre-trained denoising UNet with frozen weights, the training objective of the 2D mask generation branch implicitly pushes _f_ 3d closer to the text-2D feature space. If _f_ 3 _d_ does not align with this space, the pre-trained denoising UNet will not recognize the condition, resulting in a high loss of the 2D branch. Through gradient descent, the point cloud encoder gradually distills some vision-language knowledge from the pre-trained and frozen denoising UNet. Second, _f_ 3d encapsulates rich 3D geometric information that the 2D branch lacks due to occlusion and dimensional compression issues inherent in images. Using _f_ 3d as the condition for the 2D branch encourages the model to produce geometry-aware mask outlines and embeddings, facilitating the back-projection of 2D masks into 3D space. The effectiveness of the proposed geometry-aware mask generation will be validated through quantitative ablation comparisons in Section 4.3. 

## **3.3 2D-to-3D Mask Regularization** 

Although some vision-language knowledge from the diffusion model is distilled to the point cloud encoder _E_ via the proposed Implicit 3D Captioner, the 3D feature _f_ 3d still deviates from the 2D-text embedding space. This is because there is no constraint on the point cloud decoder _D_ , and the encoder distillation via gradient descent is inherently weak. Consequently, it is crucial to introduce contrastive regularization in the training pipeline to explicitly align 3D features with the shared 2D-text embedding space. Existing contrastive learning methods [34, 11] between 3D features and 2D-text features typically explore global, patch-wise, or point-wise relations. However, global contrastive learning is too coarse, and point-wise feature alignment is prone to noise. While patchwise contrastive learning is more fine-grained and robust, it still lacks semantic clarity. A patch in a 2D image may correspond to multiple irrelevant and discontinuous regions in a 3D point cloud due to depth compression, resulting in ambiguous and less representative local features in 3D. 

5 

To this end, we propose an explicit 2D-to-3D mask regularization term for fine-grained and consistent feature space alignment between the 3D and 2D-text modalities. Specifically, we extract 3D mask embeddings _G_ 3d _∈_ R _[M][×][C]_ from 3D features _F_ 3d using the back-projected 3D binary mask _M_ 3d: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0006-01.png)


where _G[i]_ 3d _[∈]_[R][1] _[×][C]_[represents the] _[ i]_[th][ mask embedding, and avgpool signifies the average pooling] operation. _F_[˜] 3d _[i]_[is sampled from] _[ F]_[3] _[d]_[ at indices where] _[ M][i]_ 3d[= 1][, with] _[ M][i]_ 3d[being the] _[ i]_[th][ binary mask.] 

Given the 2D binary mask _M_ 2d and the input image _I_ , we can also derive a ground truth mask CLIP feature _G_ CLIP via a pre-trained CLIP model [36]. For detailed information on obtaining the ground truth _G_ CLIP, please refer to MaskCLIP [12] or Section A.1. Subsequently, the 2D-to-3D regularization term can be computed using a classical cosine contrastive loss: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0006-04.png)


As each mask region ideally corresponds to a distinct category, the pooled mask embedding achieves semantic consistency and representativeness. Consequently, contrastive learning at the mask level offers finer granularity than global contrast, greater robustness than point-wise contrast, and clearer distinction than patch-wise contrast. Through our proposed 2D-to-3D mask regularization, the 3D features are explicitly aligned with the 2D-text feature space, enhancing the performance of the 3D branch in open vocabulary segmentation. This progress is further substantiated by the ablation studies outlined in Section 4.3. 

## **3.4 3D-2D Mask Feature Fusion** 

The 3D-2D mask feature fusion block is devised to merge 3D features _F_ 3d _∈_ R _[N][×][C]_ with the pseudo mask feature _F_ 2d _∈_ R _[N][ ′][×][C]_ derived from the 2D branch. It is noteworthy that each element in _F_ 3d possesses unique and distinguishing embeddings, whereas elements in _F_ 2d pertaining to the same mask share identical mask embeddings. Consequently, _F_ 3d offers detailed geometric structural information, while _F_ 2d provides semantic features with robust open vocabulary capabilities. Our approach combines features from these two modalities to leverage their complementary insights, resulting in _F_ fuse which excels in both precise geometry delineation and expansive semantic extrapolation. Concretely, given that _N[′] < N_ , we selectively merge _F_ 2d and _F_ 3d solely on the _N[′]_ points where correspondences exist: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0006-08.png)


where cat represents concatenation. Ablation studies in Section 4.3 and visualization results in Section 4.2 will demonstrate that _F_ fuse effectively integrates the strengths of both _F_ 3d and _F_ 2d. 

## **3.5 Training Objectives** 

In XMask3D, our training strategy encompasses supervised segmentation loss ( _L_ seg) computed from 3D ( _L_[3d] seg[),][2D][(] _[L]_[2d] seg[),][and][fusion][(] _[L]_[fuse] seg[)][modalities.][We][employ][Cross][Entropy][loss][for][3D][and] fusion segments, and for 2D, we adopt multi-head losses including Cross Entropy, Dice, and Focal Loss, following ODISE [45] and Mask2Former [4, 3] guidelines. Additionally, we follow PLA [11] to introduce a binary head and view-level contrastive loss. The binary head is optimized with Binary Cross Entropy loss ( _L_ bi) to differentiate between base and novel categories. The view-level contrastive loss ( _L_ view) is calculated between the view global feature and text embedding of the view image caption, weighted by respective coefficients ( _ω_ view[3d][,] _[ ω]_ view[2d][,] _[ ω]_ view[fuse][).][Detailed information about] the binary head and the view-level regularization can be found in the PLA paper or in Section A.3 and A.2. In conclusion, the overall training objective can be adjusted by: 

where _ω_ seg _, ω_ mask _, ω_ view[3d] _[, ω]_ view[2d] _[, ω]_ view[fuse] _[, ω]_[bi] _[,]_[ are loss weight hyperparameters.] 

_L_ = _ω_ seg _L_ seg + _ω_ mask _L_ mask + _L_ view + _ω_ bi _L_ bi (8) 

6 

Table 1: **Results for open-vocabulary 3D semantic segmentation on ScanNet dataset.** We evaluate the performance with hIoU, base and novel mIoU on five benchmarks with different category splits. 

|||||Scannet|Scannet|Scannet||||||ScanNet200|ScanNet200|||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Method|B15/N4|||B12/N7|||B10/N9|||B170/N30|||B150/N50|||
||hIoU|Base|Novel|hIoU|Base|Novel|hIoU|Base|Novel|hIoU|Base|Novel|hIoU|Base|Novel|
|LSeg-3D [23]|0.0|64.4|0.0|0.9|55.7|0.1|1.8|68.4|0.9|1.5|21.1|0.8|3.0|20.6|1.6|
|3DGenZ [31]|20.6|56.0|12.6|19.8|35.5|13.3|12.0|63.6|6.6|2.6|15.8|1.4|3.3|14.1|1.9|
|3DTZSL [5]|10.5|36.7|6.1|3.8|36.6|2.0|7.8|55.5|4.2|0.9|4.0|0.5|0.7|3.8|0.4|
|PLA [11]|65.3|68.3|62.4|55.3|69.5|45.9|53.1|76.2|40.8|11.4|20.9|7.8|10.1|20.9|6.6|
|OpenScene [34]|65.7|68.8|62.8|56.8|61.5|51.7|54.3|71.8|43.6|14.2|22.5|10.4|15.2|23.5|11.2|
|OV3D [20]|72.4|70.2|74.7|68.5|74.1|63.7|64.8|77.6|55.6|–|–|–|–|–|–|
|XMask3D|70.0|69.8|70.2|61.7|70.2|55.1|55.7|76.5|43.8|18.0|27.8|13.3|15.5|24.4|11.4|



## **4 Experiments** 

## **4.1 Experiment Settings** 

**Datasets.** In accordance with prior literature, our research conducts experimentation on two prominent indoor scene datasets: ScanNet [9] and S3DIS [1]. ScanNet, a foundational dataset in this domain, comprises 1201 scenes allocated for training and 312 scenes designated for validation. Each scene within ScanNet furnishes point cloud data, multi-view images, and corresponding camera pose matrices. Similarly, S3DIS offers analogous data modalities, encompassing 271 rooms across six distinct indoor environments. Conforming to established conventions, we reserve Area 5 of S3DIS for validation purposes, ensuring consistency with prior methodologies. 

**Category Partition.** In alignment with previous research, we exclude the _otherfurniture_ class and partition the remaining classes into three benchmarks: B15/N4, B12/N7, and B10/N9. Here, _B15_ signifies the 15 fundamental categories that remain visible and supervised during the training process, while _N4_ denotes the presence of 4 novel categories introduced during evaluation. For the ScanNet variant featuring 200 classes [38], we adopt a similar approach, dividing the dataset into B170/N30 and B150/N50 benchmarks, each representing a distinct configuration of base and novel categories. Similarly, in the case of S3DIS, comprising 13 classes, we disregard the _clutter_ class and organize the dataset into B8/N4 and B6/N6 benchmarks. Detailed information can be found in Section A.4. 

**Metrics.** Following PLA [11], we present the mean Intersection over Union (mIoU) scores separately for both base and novel categories to assess open vocabulary segmentation performance. Additionally, to provide a comprehensive evaluation of the segmentation capability, we report the harmonic mean IoU (hIoU) derived from the mIoU scores of base and novel categories. This holistic metric offers insights into the overall segmentation efficacy across the dataset. 

**Implementation Details.** Our implementation incorporates MinkUNet [7] as the 3D branch and ODISE [45] as the 2D branch within the architecture. For the vision-language model, we opt for CLIPL [36]. The training regimen for the XMask3D model involves utilizing the AdamW optimizer [29] with a Cosine learning rate scheduler. We train the model for 150 epochs on 4 NVIDIA A800 GPUs, employing a batch size of 64. Notably, we introduce mask-level regularization to the training pipeline after the initial 50 epochs. This decision is motivated by the observation that the quality of mask prediction at the onset of training may be suboptimal, making the mask-level contrastive loss ineffective and potentially misleading. Detailed information regarding hyperparameter selections is provided in Section A.4. 

## **4.2 Main Results** 

**Quantitative Comparisons.** From Table 1 and Table 2, our proposed XMask3D outperforms previous methods across most benchmarks, irrespective of the novel category proportion or dataset sources. The performance indicates that XMask3D is a robust and generalizable method for open vocabulary 3D semantic segmentation. Notably, we compare XMask3D with our baseline method, PLA, on novel category performance. On the ScanNet dataset, XMask3D demonstrates improvements ranging from 7.4% to 20.0% over PLA. On the ScanNet200 dataset, XMask3D surpasses PLA by an impressive 

7 

Table 2: **Open-vocabulary 3D semantic seg-** Table 3: **Ablations for XMask3D pipeline design** . **mentation results on S3DIS dataset** . We report We conduct experiments on the B12/N7 benchmark. hIoU, base mIoU and novel mIoU metrics. Best (a) Ablation for implicit condition of the diffusion model. open-vocabulary results are highlighted in **bold** . 

||(a) Ablation for implicit condition of the diffusion model.|(a) Ablation for implicit condition of the diffusion model.|(a) Ablation for implicit condition of the diffusion model.|
|---|---|---|---|
||Cond Base Novel bed chair table BKS pic sink BT<br>Text 69.5<br>52.7 72.9 60.6 36.7 70.0 14.3 44.6 70.3<br>2D<br>69.7<br>53.6 70.6 63.3 40.9 68.4 12.4 51.6 67.8<br>3D<br>**70.2**<br>**55.1** 72.5 62.7 37.3 70.6 18.6 51.2 73.0|||
||(b) Ablation for mask regularization and fusion block.|||
||_L_mask|Base<br>2D<br>3D Fuse|Novel<br> 2D<br>3D<br>(∆3D) Fuse (∆Fuse)|
||✗<br>40.1 63.9<br>70.0 30.9 14.0<br>53.5<br>✓<br>40.6 64.3<br>70.2 30.8 25.7 (+11.7)<br>55.1 (+1.6)|||
|XMask3D<br>**46.8 63.1**<br>**37.2**<br>**44.9** 52.8<br>**39.1**||||



70.5% and 72.7%. On the S3DIS dataset, XMask3D shows boosts of 51.8% and 33.0% over PLA. Among these benchmarks, XMask3D achieves the highest improvements on the long-tail ScanNet200 dataset, primarily due to the introduction of the denoising UNet from the pre-trained diffusion model, which constructs a comprehensive text-2D embedding space with unlimited text descriptions. 

It is noteworthy that the OpenScene results are derived from the implementation of UniM-OV3D [16], but we do not compare XMask3D with UniM-OV3D in the tables. This is because UniM-OV3D employs PointBIND [14] and CLIP2Point [19] with an already aligned 3D-text embedding space, whereas XMask3D only integrates the commonly used 2D-text space and introduces techniques to pull 3D features from any point cloud models towards the shared embedding space. Another outstanding concurrent method is OV3D [20], which primarily focuses on EntityText extraction and PointEntityText association, while XMask3D concentrates on enhancing mask-level interaction between 2D and 3D modalities. Therefore, the contributions of OV3D and XMask3D are orthogonal and could complement each other in future explorations. Additionally, OV3D only provides experiment results on the ScanNet20 benchmarks, while its effectiveness on the ScanNet200 dataset that is more challenging and on the S3DIS dataset with different data distribution remains unexplored. 

**Visualization Results.** In Figure 3, we present a comparative visualization of novel categories between XMask3D and previous methods [11, 34]. XMask3D demonstrates superior accuracy in category predictions, produces finer segmentation boundaries, and generates more cohesive mask regions. Notably, the missegmented region on the bookshelf in the second row of XMask3D is classified as picture, which appears reasonable given the corresponding part in the view image. 

## **4.3 Ablation Studies** 

In this section, we comprehensively discuss the design choices of XMask3D through extensive ablation studies on the ScanNet B12/N7 benchmark. The results are presented in Table 3. 

**Mask Generation Condition.** In Section 3.2, we introduce an Implicit 3D Captioner to convert global 3D features into implicit condition embeddings for the diffusion model. In Table 3a, we compare this implicit 3D condition with the vanilla text condition and implicit 2D condition. The text condition is generated by a ViT-GPT2 [48] captioning model and encoded via the frozen CLIP text encoder. The 2D condition is generated by the frozen CLIP image encoder and a learnable MLP, following the design in ODISE [45] (Equation 3). Our proposed implicit 3D condition outperforms the others in novel category segmentation, demonstrating that integrating the 3D global feature with the diffusion model produces the most compatible open vocabulary masks with the 3D branch. 

**Mask Regularization.** In Section 3.3, we propose a fine-grained mask regularization term to align 3D features with the 2D-text embedding space. In the first line of Table 3b, we remove the fine-grained mask-level loss _L_ mask from the training pipeline. Besides the final results from the fusion block, we also report the intermediate results from the 2D and 3D branches. The inclusion of the mask loss results in a significant improvement of 11.7 in 3D performance on novel categories, demonstrating that our fine-grained mask-level regularization effectively brings 3D features closer to the 2D-text embedding space. Additionally, the performance gain on the fused output is 1.5, highlighting the positive impact of this regularization from 3D to fusion features. 

8 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0009-00.png)


**----- Start of picture text -----**<br>
Table<br>Chair<br>Sofa<br>Bookshelf<br>Cabinet<br>Picture<br>Chair<br>Bed<br>Table<br>Bed<br>Table<br>Desk<br>View Image Ground Truth PLA OpenScene XMask3D (Ours)<br>**----- End of picture text -----**<br>


Figure 3: **Visualization Comparisons between XMask3D and Previous Methods.** We compare XMask3D with PLA [11] and OpenScene [34] on the novel categories table, bookshelf, chair and bed. The regions corresponding to the novel categories are highlighted in red boxes. 

We also analyze the effects of mask regularization through visualizations in Figure 4. Comparing the first and second rows, the 3D segmentation results without the mask loss show inconsistency in the local region for the novel category chair, with most points misclassified as bookshelf. When the mask-level regularization is added, more points are correctly classified, resulting in a clearer segmentation mask. The visualizations of the fused outputs also demonstrate consistent enhancement in mask regularization within the final column. This is evident from the rectification of misclassified points previously labeled as table on the armrest of the right chair in the last column. 

**Modality Fusion.** When comparing the 2D, 3D, and fused metrics within the same line in Table 3b, we empirically find that the 2D branch performs relatively better on novel category segmentation, while the 3D branch excels at base category segmentation. This quantitative observation aligns with our design intention: to exploit geometric knowledge via the 3D branch and to enhance the model’s open vocabulary capability via the 2D branch. More importantly, the fused output outperforms both the 2D and 3D intermediate results on both base and novel splits. These results strongly support the effectiveness of our fusion design in merging the complementary knowledge from both modalities. 

We also present visualization evidence regarding modal complementarity and fusion effects in Figure 4. In the first group, the 2D branch exhibits unsatisfactory results of the base category wall around the whiteboard and behind the right chair, whereas effectively segmenting the novel category chair with high quality despite minor artifacts. Conversely, the 3D branch produces an unsatisfactory mask for the chair but excels in segmenting the wall based on geometric information. The fusion block leverages the strengths of both branches, mitigates their weaknesses, and yields satisfactory outcomes. Moreover, the fusion block accurately delineates regions such as the left foot of the chair on the right, where both the 2D and 3D branches falter, highlighting its potential for integrating cross-modality knowledge. The second group displays similar results that convince the effectiveness of the proposed mask loss and cross-modality fusion design. 

## **4.4 Limitations** 

Due to resource constraints, we only evaluate the performance of XMask3D on semantic segmentation in this study. However, the XMask3D pipeline has the potential to be extended to the instance and panoptic perception by replacing the 3D backbone with an instance or panoptic segmentation model. 

9 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0010-00.png)


**----- Start of picture text -----**<br>
Novel<br>Chair<br>without<br>Mask  Table<br>Loss<br>Bookshelf<br>Picture<br>with  Base<br>Mask  Wall<br>Loss<br>Floor<br>Desk<br>Ground Truth 2D Branch Output 3D Branch Output Fusion Block Output<br>Novel<br>Chair<br>without<br>Mask  Table<br>Loss<br>Bookshelf<br>Base<br>Wall<br>with<br>Mask  Floor<br>Loss<br>Desk<br>Curtain<br>Ground Truth 2D Branch Output 3D Branch Output Fusion Block Output<br>**----- End of picture text -----**<br>


Figure 4: **Visualization Results of Ablations.** The first and second groups show results from ScanNet B15/N4 and B12/N7 benchmark, respectively. In each group, the first and second rows display segmentation results without and with the proposed mask regularization. The last three columns compare the outputs from the intermediate 2D and 3D branches with the final fusion block. 

This adaptability arises from the auxiliary 2D branch, which is mask-based and capable of generating instance-level masks given appropriate annotations. Furthermore, since the XMask3D pipeline is agnostic to the 3D model used, it would be interesting to conduct experiments with more advanced point cloud models to compare performance. We hope we can exploit the strength of XMask3D with high-standard 3D backbones and a wider range of dense perception tasks in future research. Computational cost is another limitation of XMask3D, as we implement the Denoising UNet which has numerous parameters and relatively slow latency. In our future work, we plan to address this limitation by replacing the 2D branch with a more lightweight 2D open vocabulary mask generator. 

## **5 Conclusion** 

In this paper, we present XMask3D, designed for open vocabulary 3D semantic segmentation. We propose the integration of the denoising UNet of a pre-trained diffusion model to produce geometryaware segmentation masks conditioned on learnable implicit 3D embeddings. These binary 2D masks filter mask-level embeddings of 3D representations and apply mask regularization to enhance the open vocabulary capacity of 3D features. By fusing the 2D mask embeddings with fine-grained 3D features, we leverage the complementary knowledge from both modalities, achieving competitive performance across various benchmarks and datasets. Ablation studies and visualization comparisons further validate the effectiveness of the proposed cross-modal mask reasoning method. 

## **Acknowledgments and Disclosure of Funding** 

This work was supported in part by the National Natural Science Foundation of China under Grant 623B2063, Grant 62321005, Grant 62336004, and Grant 62125603. 

10 

## **A Additional Implementation Details** 

## **A.1 Mask-level Regularization** 

Mask-level regularization facilitates local and finely-grained alignment between features extracted from the 3D branch of XMask3D and the 2D-text embedding space. The ground truth for mask-level loss is computed using MaskCLIP [12], utilizing the predicted segmentation mask _M_ 2d _∈_ R _[M][×][H][×][W]_ and the view image _I ∈_ R[3] _[×][H][×][W]_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0011-03.png)


where _M_ represents the number of masks, and _H_ and _W_ denote the height and width of the image, respectively. _V_ refers to the pre-trained CLIP [36] image encoder. 

Specifically, the view image _I_ is encoded into image tokens _T_ I _∈_ R _[N][×][C]_ with the pre-trained CLIP image model, where _N_ is the number of image tokens and _C_ is the dimension of the CLIP embeddings. The class token _T_ C _∈_ R[1] _[×][C]_ is duplicated _M_ times as the mask class tokens _T_ M _∈_ R _[M][×][C]_ . Then _T_ I _, T_ C _, T_ M are concatenated together to perform masked attention with frozen weights from the pre-trained CLIP image model. The attention mask is designed as 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0011-06.png)


where T _m×n_ is an _m × n_ True matrix, F _m×n_ is an _m × n_ False matrix and P is defined as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0011-08.png)


where True means that this position is masked out i.e. not allowed to attend and False otherwise. Then the updated mask class tokens _T_ M _[′]_[from the masked attention layers can be regarded as the CLIP] embedding of the masked regions, serving as the ground truth _G_ CLIP for mask-level regularization. 

## **A.2 View-level Regularization** 

The view-level regularization facilitates coarse and high-level alignment between features extracted from the 3D branch and the 2D-text embedding space. To elaborate, upon receiving an image _I_ , we initially generate its text caption using a pre-trained captioning model, ViT-GPT2 [48]. Subsequently, we employ the pre-trained CLIP [36] text encoder to encode the text caption into the 2D-text embedding space, yielding _f_ view _[T]_[, which serves as the ground truth for view-level regularization.] 

We perform average pooling operation on dense 3D point cloud features, 2D image features and fused features to obtain their global embeddings _f_ view[3d] _[, f]_ view[ 2d] _[, f]_ view[ fuse][.][Then we implement contrastive loss] between global features and the ground truth text embeddings: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0011-13.png)


where _m_ = 3d _,_ 2d _,_ fuse denotes different modalities. 

## **A.3 Binary Head** 

Following PLA [11], we include a binary head to predict whether the points belong to base or novel categories. We implement a small 3D model as the binary head with minimum computation cost. The prediction _s[b]_ is utilized to modulate the over-confident semantic score _s_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0011-17.png)


where _sB_ is the semantic score computed solely on base classes with novel class scores set to zero. Similarly, _sN_ is computed only for novel classes, setting base class scores to zero. 

11 

Table 4: **Category Partitions.** We follow PLA to split ScanNet and S3DIS into several benchmarks. (a) ScanNet dataset. 

|Partition<br>B15/N4|Base Categories<br>wall, foor, cabinet, bed, chair, table, door, window, picture,<br>counter, curtain, refrigerator, showercurtain, sink, bathtub|Base Categories<br>wall, foor, cabinet, bed, chair, table, door, window, picture,<br>counter, curtain, refrigerator, showercurtain, sink, bathtub|Novel Categories<br>sofa, bookshelf, desk, toilet|Novel Categories<br>sofa, bookshelf, desk, toilet|
|---|---|---|---|---|
|B12/N7|wall, foor, cabinet, sofa, door, window, counter, desk,<br>curtain, refrigerator, showercurtain, toilet||bed, chair, table, bookshelf, picture, sink, bathtub||
|B10/N9|wall, foor, cabinet, bed, chair, sofa, table, door, window,<br>showercurtain, curtain||bookshelf, picture, counter, desk, refrigerator,<br>toilet, sink, bathtub||
|||(b) S3DIS dataset.|||
|Partition<br>B8/N4<br>B6/N6||Base Categories||Novel Categories<br>window, table, sofa, bookcase<br>foor, window, door, table, sofa, board|



Table 5: **Per-Class Results Comparison with PLA.** We compare the per-class open vocabulary segmentation results with PLA. Novel Classes are marked in blue . 

(a) ScanNet dataset. _Shower c._ is short for shower curtain. 

|Partition<br>B15/N4<br>B12/N7<br>B10/N9|Method|<br> <br>|Base No|vel|wall<br>foor<br>cabinet<br>bed|wall<br>foor<br>cabinet<br>bed|chair<br>sofa<br>table<br>door<br>window<br>bookshelf<br>picture<br>counter<br>desk<br>curtain<br>fridge<br>shower c.<br>toilet<br>sink<br>bathtub<br>1 87.9<br>75.9 72.2 61.9 62.1<br>69.5 30.9 60.1<br>46.5 70.7 50.5 66.1<br>56.8 59.0 81.7<br>8 86.2<br>68.4 74.0 62.1 60.8<br>74.4 29.8 65.3<br>52.1 73.2 57.5 58.9<br>86.0 66.5 83.8|
|---|---|---|---|---|---|---|---|
||PLA<br>XMask3D||68.3<br>62<br> 69.8<br>70|.4|84.6 9<br>84.2 9|5.0 64.9 81.<br>4.7 69.6 80.||
|||||.2||||
||PLA<br>XMask3D||69.5<br>45<br> 70.2<br>55|.9|84.7 9<br>83.3 9|5.1 65.3<br>57.<br>4.6 68.6<br>72.|8<br>44.2 75.9<br>34.5 62.5 62.3<br>62.1<br>20.5 57.8 61.4 72.4 47.9 64.9 85.9<br>28.4<br>69.6<br>5<br>62.7 76.0<br>37.3 62.6 58.0<br>70.6<br>18.6 64.8 59.6 71.5 59.1 60.0 83.9<br>51.2<br>73.0|
|||||.1||||
||PLA<br>XMask3D||76.2<br>40<br> 76.5<br>43|.8|83.8 9<br>83.8 9|5.2 64.3 80.<br>4.7 67.3 82.|9 88.0 78.5 73.2 60.6 61.5<br>68.6<br>17.7<br>23.4<br>51.3 70.6<br>25.7<br>38.2<br>51.3<br>27.3<br>61.7<br>6 89.1 78.8 72.9 61.2 62.7<br>75.2<br>17.7<br>45.9<br>54.5 71.9<br>28.2<br>22.9<br>59.6<br>42.3<br>47.6|
|||||.8||||
||||||||(b) S3DIS dataset.<br>ceiling<br>foor<br>wall<br>beam<br>column<br>window<br>door<br>table<br>chair<br>sofa<br>bookcase<br>board<br>93.9 97.8 82.9 0.0 17.2<br>15.6 53.7<br>35.8 86.3<br>05.3<br>37.3 43.3<br>86.4 88.3 81.4 0.0 31.4<br>31.1 61.4<br>35.8 75.4<br>17.6<br>64.5 80.7<br>93.7<br>79.1 80.1 0.1 28.5<br>24.1<br>08.4<br>37.6 87.0 54.0<br>24.0<br>06.9<br>86.4<br>47.4 80.9 0.2 23.7<br>33.7<br>30.2<br>14.7 74.2 51.6<br>47.4<br>60.8|
||||Partition<br>B8/N4<br>B6/N6|M|ethod<br>LA<br>ask3D <br>LA<br>ask3D|Base Novel<br>59.0<br>24.5<br> 63.1<br>37.2<br>55.5<br>29.4<br> 52.8<br>39.1||
|||||P<br>XM||||
|||||P<br>XM||||



## **A.4 Training and Inference Settings** 

**Training.** The supervised segmentation loss _L[m]_ seg[is calculated via the per-point classification Cross] Entropy Loss on _N_ points: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0012-07.png)


where _m_ = 3d _,_ 2D _,_ fuse denotes different modalities, _yi_ is the ground truth for base categories, _F_[¯] _m_ is the normalized feature, and _τ_ is a learnable temperature parameter. 

**Inference.** We follow ODISE [45] to combine the predicted logits **p** with the prediction from a text-image discriminative model to enhance the open vocabulary classification capacity of the model. Specifically, we leverage the mask-level regularization ground truth feature _G_ CLIP from Section A.1 to modulate the segmentation logits: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0012-10.png)


where _G_[¯] CLIP is the normalized feature of _G_ CLIP, _λ ∈_ [0 _,_ 1] is the fixed balancing factor. 

**Hyper-parameters.** For all benchmarks, we set the same _ω_ seg = 4 _, ω_ view[3d][= 1] _[, ω]_ view[2d][= 4] _[, ω]_ view[fuse][=] 1 _._ 5 as the hyper-parameter choices. The _ω_ mask and _ω_ bi are set differently across benchmarks. We 

12 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-00.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-01.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-02.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-03.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-12.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-13.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-14.png)


**----- Start of picture text -----**<br>
View Image Ground Truth PLA OpenScene XMask3D  XMask3D  XMask3D<br>(Fused) (2D Branch) (3D Branch)<br>**----- End of picture text -----**<br>


Figure 5: **Complete Visualization Comparisons.** We show comprehensive comparison between XMask3D fused/2D branch/3D branch outputs and previous methods (PLA [11]/OpenScene [34]). The figure corresponds to Figure 3 in the main paper. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0013-16.png)


**----- Start of picture text -----**<br>
Novel<br>Toilet<br>without<br>Mask  Desk<br>Loss<br>Bookshelf<br>Base<br>Wall<br>with<br>Mask  Floor<br>Loss<br>Bathtub<br>Ground Truth 2D Branch Output 3D Branch Output Fusion Block Output<br>**----- End of picture text -----**<br>


Figure 6: **Visualization Results of Ablations.** The first and second rows display segmentation results without and with the proposed mask regularization. The last three columns compare the outputs from the intermediate 2D and 3D branches with the final fusion block. 

set _ω_ mask = 0 _._ 5 _/_ 0 _._ 5 _/_ 1 _/_ 2 _/_ 2 _/_ 1 _/_ 1 _, ω_ bi = 16 _/_ 12 _/_ 8 _/_ 48 _/_ 32 _/_ 20 _/_ 15 for ScanNet B15/N4, B12/N7, B10/N9, ScanNet200 B170/N30, B150/N50, S3DIS B8/N4, B6/N6 benchmarks, respectively. 

**Category Partitions.** We follow PLA [11] to partition the ScanNet [9] and S3DIS [1] datasets into several different benchmarks. Here we list the detailed partition principle in Table 4 for reference. We do not list splits for the ScanNet200 dataset since there are too many categories. 

## **B Additional Experimental Results** 

## **B.1 Per-class Results Comparison** 

We contrast per-class segmentation mean Intersection over Union (mIoU) with PLA [11] across the ScanNet and S3DIS datasets in Table 5. XMask3D significantly outperforms PLA, particularly in novel categories, thereby showcasing the effectiveness of our proposed cross-modal mask reasoning approach on open vocabulary 3D semantic segmentation. 

## **B.2 Visualization Comparison** 

In Figure 5, we show 2D and 3D branch outputs from XMask3D in addition to Figure 3 in the main paper. The 2D branch misclassifies the novel categories, while the 3D branch produces discontinuous segmentation masks. Neither of them outperforms PLA or OpenScene. However, when their features are fused together, the output becomes superior, demonstrating the complementary knowledge across different modalities. 

13 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation_images/nips_2024_XMask3D_Cross-modal_Mask_Reasoning_for_Open_Vocabulary_3D_Semantic_Segmentation.pdf-0014-00.png)


**----- Start of picture text -----**<br>
Shower<br>Curtain<br>Curtain<br>Bathtub<br>Picture<br>Wall<br>Door<br>Sink<br>Counter<br>Cabinet<br>Input Image Ground Truth 2D Branch Output 3D Branch Output Fused Output<br>**----- End of picture text -----**<br>


Figure 7: **Failure cases of XMask3D.** We focus on shower curtain, picture, and sink novel categories in each line, respectively. 

## **B.3 Illustrations of Ablations** 

We present supplementary illustrations of ablation studies pertaining to mask regularization and cross-modality fusion in Figure 6, which corresponds to the sample in Figure 1 from the ScanNet B15/N4 benchmark. 

Comparing the first and second rows of the third column reveals that the proposed mask regularization loss yields significant improvements in the segmentation of novel categories by the 3D branch. Further comparison of the last three columns in the second row demonstrates that the output from the fusion block effectively leverages the strengths of both modality branches. Specifically, while the 2D branch excels at segmenting novel categories, it tends to produce discontinuous masks for base categories with larger regions (e.g., wall). Conversely, the 3D branch may perform relatively poorly in novel category segmentation but provides consistent geometric knowledge as a complementary aspect. Consequently, the fused output achieves superior performance in open vocabulary segmentation. 

## **C Failure Cases Analysis** 

In Figure 7, we display some failure cases of XMask3D. The first sample shows a bathtub with a shower curtain in the bathroom. However, the 2D/3D branch and the fused output of XMask3D all misclassify the shower curtain as curtain. This may be because these two categories are similar in object shape and texture, with the only difference being the surrounding environment. Since XMask3D only takes a corner of the room as input instead of the entire scene, the global environmental information is insufficient for making the correct category prediction. 

The second sample shows a large area of picture on the wall. The 2D, 3D branches and the fused output of XMask3D all misclassified it as wall, due to their similar geometry. In most cases, a picture is a small region on the wall, and this picture, as large as the wall, is a typical corner case. This failure case may reveal XMask3D’s over-reliance on geometric knowledge and lesser consideration of texture information when encountering out-of-distribution samples. 

The third sample shows a sink on a counter. Due to the occlusion problem, the sink point cloud is incomplete, negatively affecting the prediction of segmentation boundaries between the sink and the counter. This occurs because they are geometrically similar when the sinking-down part of the sink is missing. 

14 

## **References** 

- [1] Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In _CVPR_ , 2016. 

- [2] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _ICCV_ , 2021. 

- [3] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In _CVPR_ , 2022. 

- [4] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. _NeurIPS_ , 2021. 

- [5] Ali Cheraghian, Shafin Rahman, Dylan Campbell, and Lars Petersson. Transductive zero-shot learning for 3d point cloud classification. In _WACV_ , 2020. 

- [6] Seokju Cho, Heeseong Shin, Sunghwan Hong, Seungjun An, Seungjun Lee, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. _arXiv preprint arXiv:2303.11797_ , 2023. 

- [7] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In _CVPR_ , 2019. 

- [8] Spconv Contributors. Spconv: Spatially sparse convolution library. `https://github.com/ traveller59/spconv` , 2022. 

- [9] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In _CVPR_ , 2017. 

- [10] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In _CVPR_ , pages 11583–11592, 2022. 

- [11] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla: Language-driven open-vocabulary 3d scene understanding. In _CVPR_ , 2023. 

- [12] Zheng Ding, Jieke Wang, and Zhuowen Tu. Open-vocabulary universal image segmentation with maskclip. In _ICML_ , 2023. 

- [13] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In _ECCV_ , 2022. 

- [14] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. _arXiv preprint arXiv:2309.00615_ , 2023. 

- [15] Huy Ha and Shuran Song. Semantic abstraction: Open-world 3d scene understanding from 2d vision-language models. _arXiv preprint arXiv:2207.11514_ , 2022. 

- [16] Qingdong He, Jinlong Peng, Zhengkai Jiang, Kai Wu, Xiaozhong Ji, Jiangning Zhang, Yabiao Wang, Chengjie Wang, Mingang Chen, and Yunsheng Wu. Unim-ov3d: Uni-modality openvocabulary 3d scene understanding with fine-grained feature representation. _arXiv preprint arXiv:2401.11395_ , 2024. 

- [17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. _arXiv preprint arXiv:2208.01626_ , 2022. 

- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _NeurIPS_ , 33, 2020. 

- [19] Tianyu Huang, Bowen Dong, Yunhan Yang, Xiaoshui Huang, Rynson WH Lau, Wanli Ouyang, and Wangmeng Zuo. Clip2point: Transfer clip to point cloud classification with image-depth pre-training. In _ICCV_ , 2023. 

- [20] Li Jiang, Shaoshuai Shi, and Bernt Schiele. Open-vocabulary 3d semantic segmentation with foundation models. In _CVPR_ , pages 21284–21294, 2024. 

- [21] Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian Rupprecht. Diffusion models for zero-shot open-vocabulary segmentation. _arXiv preprint arXiv:2306.09316_ , 2023. 

15 

- [22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _ICCV_ , 2023. 

- [23] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Languagedriven semantic segmentation. _arXiv preprint arXiv:2201.03546_ , 2022. 

- [24] Jingyao Li, Pengguang Chen, Shengju Qian, and Jiaya Jia. Tagclip: Improving discrimination ability of open-vocabulary semantic segmentation. _arXiv preprint arXiv:2304.07547_ , 2023. 

- [25] Ziyi Li, Qinye Zhou, Xiaoyun Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Guiding text-to-image diffusion model towards grounded generation. In _ICCV_ , 2023. 

- [26] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In _CVPR_ , 2023. 

- [27] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, and Shijian Lu. 3d open-vocabulary segmentation with foundation models. _arXiv preprint arXiv:2305.14093_ , 2023. 

- [28] Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. Segment any point cloud sequences by distilling vision foundation models. _NeurIPS_ , 36, 2023. 

- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- [30] Chaofan Ma, Yuhuan Yang, Yanfeng Wang, Ya Zhang, and Weidi Xie. Open-vocabulary semantic segmentation with frozen vision-language models. _arXiv preprint arXiv:2210.15138_ , 2022. 

- [31] Björn Michele, Alexandre Boulch, Gilles Puy, Maxime Bucher, and Renaud Marlet. Generative zero-shot learning for semantic segmentation of 3d point clouds. In _3DV_ , 2021. 

- [32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. _Communications of the ACM_ , 2021. 

- [33] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 

- [34] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In _CVPR_ , 2023. 

- [35] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In _ICLR_ , 2022. 

- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_ , 2021. 

- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , 2022. 

- [38] David Rozenberszki, Or Litany, and Angela Dai. Language-grounded indoor 3d semantic segmentation in the wild. In _ECCV_ , 2022. 

- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. _NeurIPS_ , 35, 2022. 

- [40] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In _ICLR_ , 2023. 

- [41] Weijia Wu, Yuzhong Zhao, Mike Zheng Shou, Hong Zhou, and Chunhua Shen. Diffumask: Synthesizing images with pixel-level annotations for semantic segmentation using diffusion models. In _ICCV_ , 2023. 

- [42] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler, faster, stronger. _arXiv preprint arXiv:2312.10035_ , 2023. 

16 

- [43] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling. _NeurIPS_ , 2022. 

- [44] Zihao Xiao, Longlong Jing, Shangxuan Wu, Alex Zihao Zhu, Jingwei Ji, Chiyu Max Jiang, WeiChih Hung, Thomas Funkhouser, Weicheng Kuo, Anelia Angelova, et al. 3d open-vocabulary panoptic segmentation with 2d-3d vision-language distillation. _arXiv preprint arXiv:2401.02402_ , 2024. 

- [45] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In _CVPR_ , 2023. 

- [46] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In _CVPR_ , 2023. 

- [47] Jihan Yang, Runyu Ding, Weipeng Deng, Zhe Wang, and Xiaojuan Qi. Regionplc: Regional point-language contrastive learning for open-world 3d scene understanding. _arXiv preprint arXiv:2304.00962_ , 2023. 

- [48] ydshieh. Vit-gpt2 image captioning. `https://huggingface.co/nlpconnect/ vit-gpt2-image-captioning` . 

- [49] Junbo Zhang, Runpei Dong, and Kaisheng Ma. Clip-fo3d: Learning free open-world 3d scene representations from 2d dense clip. In _ICCV_ , 2023. 

- [50] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In _ICCV_ , 2023. 

- [51] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In _ICCV_ , 2021. 

17 

