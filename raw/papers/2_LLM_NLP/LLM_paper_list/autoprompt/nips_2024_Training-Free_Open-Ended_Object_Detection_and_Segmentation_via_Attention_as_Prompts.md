# **Training-Free Open-Ended Object Detection and Segmentation via Attention as Prompts** 

**Zhiwei Lin Yongtao Wang** _[∗]_ **Zhi Tang** Wangxuan Institute of Computer Technology, Peking University, China `{zwlin, wyt, tangzhi}@pku.edu.cn` 

## **Abstract** 

Existing perception models achieve great success by learning from large amounts of labeled data, but they still struggle with open-world scenarios. To alleviate this issue, researchers introduce open-set perception tasks to detect or segment unseen objects in the training set. However, these models require predefined object categories as inputs during inference, which are not available in real-world scenarios. Recently, researchers pose a new and more practical problem, _i.e._ , open-ended object detection, which discovers unseen objects without any object categories as inputs. In this paper, we present VL-SAM, a training-free framework that combines the generalized object recognition model ( _i.e.,_ Vision-Language Model) with the generalized object localization model ( _i.e.,_ Segment-Anything Model), to address the open-ended object detection and segmentation task. Without additional training, we connect these two generalized models with attention maps as the prompts. Specifically, we design an attention map generation module by employing head aggregation and a regularized attention flow to aggregate and propagate attention maps across all heads and layers in VLM, yielding high-quality attention maps. Then, we iteratively sample positive and negative points from the attention maps with a prompt generation module and send the sampled points to SAM to segment corresponding objects. Experimental results on the long-tail instance segmentation dataset (LVIS) show that our method surpasses the previous open-ended method on the object detection task and can provide additional instance segmentation masks. Besides, VL-SAM achieves favorable performance on the corner case object detection dataset (CODA), demonstrating the effectiveness of VL-SAM in real-world applications. Moreover, VL-SAM exhibits good model generalization that can incorporate various VLMs and SAMs. 

## **1 Introduction** 

Deep learning has achieved remarkable success in perception tasks, with autonomous driving as a typical practical application. Existing deep learning based perception models rely on extensive labeled training data to learn to recognize and locate objects. However, training data cannot cover all types of objects in real-world scenarios. When faced with out-of-distribution objects, existing perception models may fail to recognize and locate objects, which can lead to severe safety issues [24]. 

Many open-world perception methods [15, 48] are proposed to address this issue. Open-world perception tries to give precise results in dynamic and unpredictable environments, which contain novel objects and involve scene domain shifting. Current open-world perception methods can be roughly divided into two categories: _open-set_ and _open-ended_ . Open-set methods [52, 43, 6] often calculate the similarity between image regions and category names with a pretrained CLIP [35] model. Thus, they require predefined object categories as inputs for the CLIP text encoder during 

> _∗_ Corresponding author 

Preprint. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0002-00.png)


**----- Start of picture text -----**<br>
Image Input<br>Image Feature<br>SAM   SAM<br>Encoder Decoder Output Result<br>Attention Map<br>Prompt Prompt<br>Generation Encoder<br>Iterative Refinement<br>Attention Map<br>Generation<br>Answer:  On the road, there are vehicles (including a white<br>Question prompt: objects on the road.''''List all Vision-Language Model pedestrian crossing, and SUV in the foreground), traffic lights, road markings, a a pothole < continue generation ... > .  : Frozen<br>.<br>**----- End of picture text -----**<br>


Figure 1: **Illustration of VL-SAM.** Without additional training, we connect the vision-language and segment-anything models with attention maps as the intermediate prompts. 

inference. However, in many real-world application scenarios, we do not have the exact predefined object categories. For instance, in autonomous driving, self-driving vehicles may meet unexpected objects, including various rare animals. Besides, some objects cannot be presented by a simple category name, such as a human in an animal costume, which may look like an animal but is actually a human. Some methods use generic obstacle detection to handle unknown objects. However, many things do not have a significant 3D shape, like pits or grains on the ground. Thus, open-set methods cannot handle all situations. In contrast, open-ended methods [26, 48] are more general and practical since they can predict the object categories and locations themselves. 

In a separate line of research, large vision-language models (VLMs) [29, 23, 55] show a strong generalized ability to recognize objects, _e.g._ , it can recognize rare objects for corner cases in autonomous driving scenarios [44]. However, VLM’s localization ability is less accurate than that of specific perception models [51], sometimes missing objects or giving wrong localization results. On the other hand, as a pure vision model, segment-anything model (SAM) [20] exhibits good generalized segmentation capabilities for images from many different domains. However, SAM is unable to provide categories for segmented objects [49] and may yield numerous irrelevant segmentation results. 

In this paper, we propose to combine the existing generalized object recognition model, _i.e._ , VLM, with the generalized object localization model, _i.e._ , SAM, to address the open-ended object detection and segmentation task. We present VL-SAM, a training-free framework that connects two generalized models with attention maps as the intermediate prompts, as illustrated in Figure 1. Specifically, we utilize the attention maps generated by VLM when describing the whole driving scene to prompt the segmentation of SAM. Firstly, given the generated token of VLM, we use the token as the query to obtain the attention maps from all layers and heads of VLM. Then, in the attention map generation module, we introduce the head aggregation and attention flow mechanism to aggregate and propagate global attention maps through all heads and layers. Besides, to alleviate the collapse problem caused by causal masks when propagating with attention flow, we adopt a regularization term to constrain the attention flow propagation process. After that, to better guide SAM to segment with the attention maps, we present a prompt generation module by grouping and sampling positive and negative points as the point prompts for SAM. Furthermore, to reduce the number of missing objects, we further use the segmentation results from SAM to sample positive and negative points from attention maps iteratively until convergence. 

The main contributions of this work are summarized as follows: 

- We present VL-SAM, a training-free open-ended object detection and segmentation framework that connects the generalized object recognition model and the generalized object localization model with attention maps as the prompts. 

- We introduce a head aggregation and regularized attention flow mechanism to aggregate and propagate attention maps with the causal masks through all heads and layers. 

- We propose an iterative refinement pipeline with a positive and negative point sampling strategy for attention maps. 

2 

- VL-SAM outperforms the _open-ended_ method GenerateU and obtains competitive results compared with existing _open-set_ methods on the long-tail instance segmentation dataset LVIS [14]. In autonomous driving applications, VL-SAM achieves favorable corner case object detection performance on the CODA [24]. 

## **2 Related work** 

## **2.1 Vision Language Model** 

Large language models (LLMs), including GPT-3 [3], GLM [11], and LLaMA [40], have shown human-like dialogue and reasoning skills. However, the limitation of LLM’s ability to process and understand visual data restricts its application to more real scenarios. To overcome this, a cuttingedge Vision-Language Model (VLM) is introduced to open up new vistas for application. Recently, BLIP-2 [23] proposes Q-Former to connect and fuse image and text embeddings with three alignment pretrain losses. LLaMA-Adapter [53, 12], LLaVA [29], and MiniGPT [55] introduce an adapter or projection layer to align the embedding space from image and text. CogVLM [41] presents visual expert modules to transform the image features to align with text features in different transformer heads. SPHINX [28] utilizes several mixing techniques for multiple visual tasks. Furthermore, CogAgent [17] and LLaVA-Phi [57] view VLM as an agent or assistant to complete various tasks. 

Existing VLMs, especially GPT-4V [2], exhibit strong generalization capability for understanding and reasoning new or rare situations, _e.g.,_ it can deal with corner cases for autonomous driving [44]. However, the localization ability of VLMs is weaker than specific perception models, like SAM. 

In this paper, we equip VLM with generalized segmentation models, _i.e._ , SAM, to address the localization limitation of VLM for open-ended object detection and segmentation. We achieve this by connecting two models with attention maps as the prompts without additional training. 

## **2.2 Open-World Object Detection and Segmentation** 

With the advent of the CLIP models [35], open-world classification, object detection, and instance segmentation have made great progress at the same time. Open-world methods try to discover and recognize unseen objects in the training set during inference. Current open-world methods can be roughly classified into two types: _open-set_ [37] and _open-ended_ [26]. Open-set methods require redefined object categories, including seen objects and unseen objects in the training set, as inputs during inference. By contrast, open-ended methods can locate seen and unseen objects and generate their names simultaneously, as the current VLM does. In real-world applications, the exact categories may remain unknown for the perception models. For instance, in autonomous driving, self-driving vehicles often encounter unknown objects on the road, including overturned cars and construction vehicles with various shapes. Thus, the open-ended problem is more general and practical. 

**Open-Set Methods.** With the powerful text-image embedding matching with CLIP, current openset object detection methods mainly use a proposal network to obtain foreground object bounding boxes and embeddings, and then use CLIP as the open-set classification module to predict their categories. More recently, GLIP [25] proposes to use phrase grounding to pre-train open-world object detectors. GroundingDINO [30] presents cross-modality fusions to introduce text information to the image encoder for object grounding. SWORD [45] designs a novel contrastive method to learn the discrimination between foreground and background for instance segmentation. YOLO-World [7] introduces a prompt-then-detect paradigm for real-time open-world object detection. However, the above methods require predefined object categories as inputs for the text encoder. 

**Open-Ended Methods.** GenerateU [26] first proposes the open-ended problem. Concurrently, DetCLIPv3 [48] introduces a similar concept with open-ended. They present a generative framework with language models to generate object categories and bounding boxes at the same time. To achieve better generalization capabilities, they construct a large dataset with bounding box and caption pairs and finetune the whole network on the constructed dataset. 

In contrast, we propose a training-free open-ended framework, VL-SAM, that combines generalized recognition and segmentation models. VL-SAM can generate object categories with the generalized recognized model and then localize objects with the generalized segmentation models. 

3 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0004-00.png)


**----- Start of picture text -----**<br>
VL-SAM Framework<br>VLM<br>Question prompt: VLM Transformer Decoder : Frozen<br>''List all objects on the road.'' Tokenizer Query Query Query<br>Answer:  On the road, there are vehicles<br>(including a white SUV in the foreground), traffic<br>Image Key Key Key lights, road markings, a pedestrian crossing, and  a<br>Encoder Value Value Value pothole < continue generation ... > .<br>.<br>Attention Map<br>Attention Map Generation<br>Head Attention<br>Aggregation Flow<br>Multi-layer Multi-head Attention<br>Point Sampling<br>Iterative Refinement<br>Output<br>Result<br>SAM   SAM   Prompt<br>Encoder Decoder Encoder : Positive Point<br>SAM : Negative Point<br>Transformer Layer Transformer Layer Transformer Layer<br>**----- End of picture text -----**<br>


Figure 2: **An overview of VL-SAM framework.** We first use VLM to describe the input image and generate all possible objects’ names. Then, for each object name, we obtain the corresponding attention map with the attention map generation module. Finally, we sample point prompts from the attention map and send them to SAM to predict detection and segmentation results. 

## **3 Method** 

As shown 2, we provide an overview of our proposed framework. We use VLM and SAM as the generalized object recognition model and object localization model, respectively. Given an image input, we first use VLM to describe the scene and list all possible objects in the image. Then, for each object, we use the attention generation module with head aggregation and attention flow to obtain the high-quality attention map from VLM. Finally, we generate point prompts from the attention map and send them to SAM to get the location prediction iteratively. 

## **3.1 Preliminary** 

**Segment Anything Model.** SAM is a prompt-based segmentation model with excellent data generation capability. It consists of three components: an image encoder, a mask decoder, and a prompt encoder. SAM takes an image and a set of prompts, including points, a box, and a mask, as the inputs. To segment objects with the prompts, SAM first extracts image features with the image encoder. Concurrently, the set of prompts is sent to the prompt encoder to transform into the prompt tokens. Then, the image features, prompt tokens, and mask tokens interact in the mask decoder with the two-way transformers. Finally, the mask tokens are transformed into multi-scale segmentation masks by multiplying mask tokens with the image features following MaskDINO [22]. 

**Auto-Regressive Based Vision-Language Model.** Current Auto-Regressive based VLMs have yielded surprising performance in various vision-language tasks. The mainstream framework of current VLMs comprises four parts, _i.e._ , an image encoder, a text tokenizer, projection layers, and a language decoder. Given an image and text as inputs, VLMs extract image tokens and text tokens with the image encoder and text tokenizer, respectively. Then, the image tokens are aligned with text tokens with projection layers. After that, the tokens from two modals are concatenated and sent to the language decoder to generate text outputs. The language decoder adopts the next-token prediction paradigm that the probability of the current generated token _xt_ depends on all previous tokens ( _x_ 1 _, x_ 2 _, ..., xt−_ 1). 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-00.png)


**----- Start of picture text -----**<br>
Head Aggregation<br>Attention Head Weights<br>MeanMax<br>**----- End of picture text -----**<br>


Figure 3: **Head aggregation.** We aggregate information from all attention heads with head weights. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-02.png)


Figure 4: **Attention flow.** We propagate attention from the first layer to last layer with attention flow. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-04.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-05.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-06.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-07.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-08.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-09.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-10.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-11.png)



![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-12.png)


**Question:** ''List all objects on the road.'' **Answer:** On the road, there are vehicles ( **a red truck** and a blue bus), road signs, a pedestrian crossing, a white barrier, and a few other smaller objects that are not clearly identifiable from the image. 

**Question:** ''List all objects on the road.'' **Answer:** On the road, there are vehicles (including **a white SUV** in the foreground), traffic lights, road markings, a pedestrian crossing, and a pothole filled with asphalt patching material. 

**Question:** ''List all objects on the road.'' **Answer:** The objects on the road include a white bus, a yellow truck, **traffic cones** , and a person in camouflage attire. 

Figure 5: **Illustration of attention collapse.** For each column, from left to right, we show image inputs, attention flow (collapse), regularized attention flow, and generated answers from VLM. 

## **3.2 Attention Map Generation** 

The main idea of VL-SAM is to use attention maps of objects as the prompts for SAM to segment. Thus, how to generate a high-quality attention map for an object is critical. To achieve this, we introduce attention flow to aggregate and propagate attention maps through all transformer heads and layers in VLM. 

Specifically, given an image input, we ask VLM to give all possible objects in the image. During this process, we cache all queries and keys from VLM. Then, we multiply queries and keys with causal masks and SoftMax normalization to obtain similarity matrix _S ∈ N × N × H × L_ , where _N_ is the length of queries and keys, _H_ is the number of transformer heads, and _L_ denotes the number of transformer layers. _Si,j[h,l]_[represents the similarity between query] _[ i]_[ and key] _[ j]_[in the head] _[ h]_[,] layer _l_ . After that, we aggregate information from all transformer heads with mean-max attention head weights, as shown in Figure 3. In particular, we choose the maximum similarity weights of matrix _S_ in dimension _j_ and average them in dimension _i_ to obtain the attention head weights _W ∈_ 1 _×_ 1 _× H × L_ : 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-20.png)


Obviously, the attention head weight indicates the importance of each head in each layer. Then, we pointwise multiply attention head weight _W_ with similarity matrix _S_ and average all heads as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0005-22.png)


After aggregating all information from all heads, we present attention flow to further aggregate attention from all layers, as illustrated in Figure 4. Concretely, we use the attention rollout method [1] 

5 

to compute the attentions from layer _l −_ 1 to layer _l_ as follows: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/autoprompt/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts_images/nips_2024_Training-Free_Open-Ended_Object_Detection_and_Segmentation_via_Attention_as_Prompts.pdf-0006-01.png)


where _I_ is the identity matrix. After the attention rollout, we only need the attention map from the last layer. To obtain the image attention map of the generated token, we select the corresponding line and columns from _S_[¯] _[′][L]_ . 

However, since VLM uses causal masks for auto-regressive generation, simply adopting the attention rollout method causes attention collapse, as shown in Figure 5. Fortunately, we find a simple regularization term that can alleviate this problem efficiently. Specifically, for each column, assuming the unmasked length is _L_ 0, we multiply each value in this column with 1 _−_ ( _L_ 0 _−_ 1) _/L_ . With this regularization term, the attention value in the top left corner will be constrained. 

## **3.3 SAM Prompt Generation** 

The attention map generated in Section 3.2 has some unstable false positive peaks. To filter these false positive areas, we first use a threshold to filter weak activated areas and find the maximum connectivity area as the positive area [5]. The remaining area serves as a negative area. After that, we sample a positive point from the positive area with the maximum activated value and a negative point from the negative area with the weakest activated value. The positive and negative points serve as the point prompt pair for SAM. 

## **3.4 Iterative Refinement** 

The segmentation results from the SAM decoder may include rough edges and background noises. We adopt two iterative strategies to further refine the segmentation results. In the first iterative strategy, we follow the cascaded post-refinement in PerSAM [54] to take the initial segmentation masks generated with the positive and negative pairs as the additional prompt input for the SAM decoder. For the second iterative strategy, we use the segmentation masks in the first iterative strategy to mask the attention map _S_[¯] _[′]_ . Then, we iteratively generate positive and negative pairs with Prompt Generation in Section 3.3 from the masked attention map and send them to the SAM decoder. Finally, we aggregate the results with NMS [13]. 

## **3.5 Multi-scale Ensemble** 

Due to the low-resolution image input of the image encoder in VLM, VLM may fail to recognize small objects. For instance, it may generate an answer: ‘ _On the road, there are vehicles (a red truck and a blue bus), road signs, a pedestrian crossing, a white barrier, and_ _**a few other smaller objects that are not clearly identifiable from the image.**_ ’. To alleviate this issue, we follow SPHINX [28] to split an image ( _H × W_ ) into four sub-images ( _H/_ 2 _× W/_ 2) from the four corners and send each sub-image to VL-SAM independently. Finally, we ensemble the output of VL-SAM for four sub-images and the whole image. 

## **3.6 Question-prompt Ensemble** 

The output of VLM is sensitive to the input prompt. To obtain a more comprehensive description of the input image, we ask VLM to generate ten question prompts for scene description with the sentence:‘ _If we want you to list all possible objects in the given image, what questions should we ask? Please give 10 questions you prefer._ ’ Then, we use the generated question prompts for VL-SAM to segment objects and ensemble the outputs of all question prompts. 

## **4 Experiments** 

## **4.1 Implementation Details** 

We chose CogVLM-17B [41] with EVA2-CLIP-E [39] and Vicuna-7B-v1.5 [8] as the vision-language model. CogVLM-17B divides an image with 490 _×_ 490 into 35 _×_ 35 patches. We set the temperature 

6 

Table 1: **Comparison of object detection and segmentation results on LVIS minival.** ‘OpenEnded’ denotes that we do not have exact object categories during inference [26]. We report _fixed_ AP [9] for rare objects. * denotes using the external data. 

|Method|Type|Training|LVIS<br>box AP_rare_<br>mask AP_rare_|
|---|---|---|---|
|||||
|Mask R-CNN [16]<br>Deformable DETR [56]|Close-Set|✓<br>✓|26.3<br>25.1<br>24.2<br>-|
|||||
|GLIP [25]<br>GroundingDINO [30]<br>DetCLIP [47]<br>YOLOWorld [7]<br>OWLv2_∗_[32]|Open-Set|✓<br>✓<br>✓<br>✓<br>✓|20.8<br>-<br>27.4<br>-<br>26.9<br>-<br>27.1<br>-<br>39.0<br>-|
|||||
|GenerateU [26]<br>VL-SAM (Ours)|Open-Ended|✓<br>_×_|20.0<br>-<br>23.4<br>22.7|



to 0.8 and top-p for nucleus sampling to 0.1 for CogVLM-17B. For the generated localization model, we use SAM with ViT-Huge [10]. 

We evaluate VL-SAM in a _training-free zero-shot_ manner for all datasets. To obtain object categories from the generated sentence of VLM, we follow Tag2Text [18] to parse tags from the given sentence. To evaluate the open-ended performance on datasets with predefined object category names, we follow GenerateU [26] to adopt CLIP [35] text encoder and map the generated object categories to predefined categories in datasets for evaluation. Specifically, we use the text prompt ‘a {object category}’ for CLIP text encoder to calculate the similarity between generated object categories and predefined categories for mapping. All models are inferred on an 80G A800 machine. 

## **4.2 Main Results** 

**LVIS Dataset.** We evaluate VL-SAM on the LVIS dataset [14], which has a long tail of categories and annotations for over 1000 object categories. Following previous works [26, 7], we mainly evaluate VL-SAM on LVIS minival and report the fixed AP [9] for rare objects. 

As shown in Table 1, we list the performance for three types of perception methods, _i.e.,_ close-set, open-set [15], and open-ended. The different between open-set and open-ended is that open-set requires exact prior knowledge of object categories as inputs, while open-ended can generate them during inference in a zero-shot manner [26]. In a real scenario, we often do not have predefined object categories for a scene. Thus, open-ended methods are more general and practical. As we can see, VL-SAM outperforms GenerateU by 3.4 _APrare_ . Notably, VL-SAM is a training-free framework and can simultaneously obtain boxes and segmentation masks. In contrast, GenerateU needs to fine-tune both the image encoder and language model on VG [21] and GRIT [33] datasets, requiring significant training costs, and can only predict bounding boxes. Besides, VL-SAM achieves competitive detection and segmentation performance compared to open-set detection methods and close-set segmentation methods, respectively. 

**CODA Dataset.** To further demonstrate the effectiveness of the proposed method in the real-world application, we present the results of VL-SAM on corner case object detection dataset CODA for autonomous driving in Table 2. Specifically, as we can see, RPN only achieves 10.6 mAR, indicating that current open-set detectors relying on object proposals have difficulty dealing with corner cases. For more recent open-set detectors, they achieve higher mAR with CLIP as the object category predictor. For the open-ended method, LLaVA-Grounding ensembles VLM and grounding models into one model and achieves better performance than open-set methods. However, aggregating VLM and grounding models to one model requires joint training of two models, introducing additional training costs. By contrast, VL-SAM is a training-free framework and obtains significant performance improvement over LLaVA-Grounding from 18.4 mAR to 40.1 mAR. 

In addition, we evaluate the performance upper bound of the current SAM. We utilize ground-truth boxes as the box prompt for SAM decoder to segment objects. We can observe that, in this setting, SAM achieves 54.1 mAR and 94.1 AR50 since SAM has its limitations on segmentation tasks. 

7 

Table 2: **Comparison of object detection results on CODA.** We chose the best performance for _[∗]_ results from CODA. _[†]_ denotes few-shot object detectors in the one-shot setting. ‘Oracle’ represents utilizing ground-truth boxes as the box prompt for SAM. 

|Method|Type|VLM|Training|CODA<br>mAR<br>AR50<br>AR75|
|---|---|---|---|---|
||||||
|RetinaNet_∗_[27]<br>Faster R-CNN_∗_[36]<br>Cascade R-CNN_∗_[4]<br>Deformable DETR_∗_[56]<br>Sparse R-CNN_∗_[38]<br>Cascade Swin_∗_[31]<br>RPN_∗_[36]|Close-Set|_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_<br>_×_|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|12.8<br>23.2<br>11.9<br>10.7<br>19.2<br>10.2<br>10.4<br>18.5<br>9.7<br>9.0<br>22.2<br>5.6<br>10.1<br>19.6<br>9.0<br>9.9<br>17.2<br>9.7<br>10.6<br>20.0<br>10.2|
||||||
|ORE_∗_[19]<br>FsDet_†_ [42]<br>DeFRCN_†_ [34]<br>GroundingDINO [30]<br>YOLOWorld [7]|Open-Set|_×_<br>_×_<br>_×_<br>✓<br>✓|✓<br>✓<br>✓<br>✓<br>✓|8.3<br>16.4<br>7.4<br>4.2<br>7.7<br>4.0<br>4.5<br>8.9<br>4.2<br>12.6<br>21.7<br>13.3<br>16.1<br>26.2<br>19.6|
||||||
|LLaVA-Grounding [51]<br>VL-SAM (Ours)|Open-Ended|✓<br>✓|✓<br>_×_|18.4<br>30.5<br>22.0<br>40.1<br>90.1<br>50.5|
||||||
|GT+SAM (Oracle)|_−_|_−_|_−_|54.1<br>94.1<br>64.9|



Table 3: **Ablation of main components.** ‘Attn’ is short for ‘attention’. Each component improves the detection performance consistently. 

|Naive Attn|Attn Generation<br>Prompt Generation<br>Iterative Refne|Multi-scale<br>Question ensemble|mAR|
|---|---|---|---|
|✓<br>✓||✓<br>✓|2.2<br>5.0|
||✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|✓<br>✓<br>✓|10.1<br>12.3<br>14.1<br>27.3<br>40.1|



It sometimes over- or under-segments an object and cannot obtain perfect segmentation results. Nevertheless, VL-SAM achieves 74.1% mAR performance of this upper bound, demonstrating the effectiveness of the proposed framework. Overall, VL-SAM achieves favorable performance on the CODA dataset. 

## **4.3 Ablation Study** 

**Main Components.** As shown in Table 3, we conduct ablation studies on CODA to analyze the effectiveness of each component of VL-SAM. For the baseline Naive Attention method, we use the attention map from the last layers and average all attention heads. We can see that the Naive Attention baseline obtains unsatisfactory results even with multi-scale and question ensemble techniques. With the proposed attention generation module, we improve the baseline by 7.9 mAR. Adding points pairs with prompt generation brings 2.2 mAR improvement. Besides, refining the segmentation maps with the iterative refinement module improves the detection performance from 12.3 mAR to 14.1 mAR. Furthermore, ensembling with multi-scale image input and question prompt obtains 13.2 mAR and 12.8 mAR, respectively. Though multi-scale and question prompt ensembles greatly improve performance, these two ensemble techniques do not show effectiveness without the proposed components. In summary, the results show the effectiveness of each component proposed in VL-SAM. 

**Attention Generation.** To obtain high-quality attention maps from VLM, we introduce head weights to fuse transformer heads and a regularization term for attention flow. As shown in Table 4, simply using attention flow [1] almost fails to recognize objects for SAM due to the attention collapse caused by causal masks (Figure 5). With the regularization term, the attention flow mechanism shows its 

8 

Table 4: **Ablation of attention generation.** We can obtain high-quality attention maps with the proposed modules. 

|Naive Attention Map|Attention Flow<br>No Regularization<br>Regularization|Head Weight|mAR|
|---|---|---|---|
|||||
|✓|||2.2|
|||||
||✓<br>✓<br>✓|✓|0.1<br>8.5<br>10.1|



Table 5: **Ablation of model generalization.** VL-SAM can adopt various vision-language models and segmentation models. 

|Vision-Language Model|Segmentation Model|mAR|
|---|---|---|
||||
|CogVLM|SAM|40.1|
||||
|MiniGPT-4<br>LLaVA|SAM<br>SAM|34.7<br>37.2|
|CogVLM<br>MobileSAM<br>29.2|||



superiority over naive attention by improving 6.3 mAR. Moreover, fusing with head weights leads to a 1.6 mAR improvement. 

**Model Generalization.** To demonstrate the model generalization ability of the VL-SAM framework, we adapt two additional popular VLMs, MiniGPT-4 [55] and LLaVA [29] to replace CogVLM and use MobileSAM [50] to replace SAM. In Table 5, we present the results of using these models in the VL-SAM framework. Empirical results show that replacing CogVLM with MiniGPT-4 or LLaVA may reduce the object localization performance in corner cases as CogVLM shows more powerful multimodal chat and reasoning ability than MiniGPT-4 and LLaVA. This indicates that our VL-SAM framework can benefit from more powerful VLMs. Besides, replacing SAM with a more lightweight but less accurate MobileSAM also leads to performance drops. Nevertheless, all these results outperform previous methods (18.4 mAR) in Table 2. This evidences that our framework can generalize to multiple vision-language and segmentation models. 

## **5 Limitations** 

Since we combine VLM and SAM to address the open-ended object detection and segmentation task, VL-SAM inherits the defects of VLM and SAM. The first defect is the hallucination problem in VLM. VL-SAM also suffers from hallucinations, generating wrong object tokens and attention maps. The second defect is the low inference speed of VL-SAM. However, these defects can be fixed in the future. For example, there are many more efficient SAM variant models, including EfficientSAM [46] and MobileSAM [50]. Our framework can benefit from these new models since we can easily replace CogVLM and SAM in VL-SAM with these more efficient and highly accurate models. 

## **6 Conclusion** 

In this paper, we introduce VL-SAM, a framework that cascades VLM and SAM with the attention map to address the open-ended object detection and segmentation task. Without additional training, we adopt attention maps generated by VLM as the prompts for SAM to segment objects. We introduce the attention flow mechanism to aggregate high-quality attention maps. Besides, we present an iterative refinement pipeline with positive and negative points pair sampling strategy to acquire more accurate segmentation masks. Experimental results on the long-tail generic instance segmentation dataset LVIS show that VL-SAM beats the open-ended method GenerateU and achieves competitive performance compared with close-set and open-set methods. Moreover, VL-SAM achieves favorable results on the corner case object detection dataset CODA. 

9 

**Broader Impacts Statement.** This paper studies utilizing VLM and SAM for open-ended object detection and segmentation. We do not see potential privacy-related issues. This study may inspire future research on open-ended perception and potential corner case object detection applications in autonomous driving. However, the proposed model’s performance is not yet up to the level of practical application and may pose safety threats when applied directly in practice. 

10 

## **References** 

- [1] Samira Abnar and Willem H. Zuidema. Quantifying attention flow in transformers. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020. 

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _Neural Information Processing Systems (NeurIPS)_ , 2020. 

- [4] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2018. 

- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _IEEE International Conference on Computer Vision (ICCV)_ , 2021. 

- [6] Xi Chen, Shuang Li, Ser-Nam Lim, Antonio Torralba, and Hengshuang Zhao. Open-vocabulary panoptic segmentation with embedding modulation. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 

- [7] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-vocabulary object detection. _arXiv preprint arXiv:2401.17270_ , 2024. 

- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. _See https://vicuna. lmsys. org (accessed 14 April 2023)_ , 2023. 

- [9] Achal Dave, Piotr Dollár, Deva Ramanan, Alexander Kirillov, and Ross Girshick. Evaluating largevocabulary object detectors: The devil is in the details. _arXiv preprint arXiv:2102.01066_ , 2021. 

- [10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_ , 2020. 

- [11] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. GLM: general language model pretraining with autoregressive blank infilling. In _Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022. 

- [12] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. _arXiv preprint arXiv:2304.15010_ , 2023. 

- [13] Ross Girshick. Fast r-cnn. In _IEEE International Conference on Computer Vision (ICCV)_ , 2015. 

- [14] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2019. 

- [15] Akshita Gupta, Sanath Narayan, KJ Joseph, Salman Khan, Fahad Shahbaz Khan, and Mubarak Shah. Owdetr: Open-world detection transformer. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 

- [16] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In _IEEE International Conference on Computer Vision (ICCV)_ , 2017. 

- [17] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. _arXiv preprint arXiv:2312.08914_ , 2023. 

- [18] Xinyu Huang, Youcai Zhang, Jinyu Ma, Weiwei Tian, Rui Feng, Yuejie Zhang, Yaqian Li, Yandong Guo, and Lei Zhang. Tag2text: Guiding vision-language model via image tagging. _arXiv preprint arXiv:2303.05657_ , 2023. 

- [19] KJ Joseph, Salman Khan, Fahad Shahbaz Khan, and Vineeth N Balasubramanian. Towards open world object detection. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2021. 

- [20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _IEEE International Conference on Computer Vision (ICCV)_ , 2023. 

- [21] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International Journal on Computer Vision (IJCV)_ , 2017. 

11 

- [22] Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M Ni, and Heung-Yeung Shum. Mask dino: Towards a unified transformer-based framework for object detection and segmentation. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 

- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International Conference on Machine Learning (ICML)_ , 2023. 

- [24] Kaican Li, Kai Chen, Haoyu Wang, Lanqing Hong, Chaoqiang Ye, Jianhua Han, Yukuai Chen, Wei Zhang, Chunjing Xu, Dit-Yan Yeung, et al. Coda: A real-world road corner case dataset for object detection in autonomous driving. In _European Conference on Computer Vision (ECCV)_ , 2022. 

- [25] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022. 

- [26] Chuang Lin, Yi Jiang, Lizhen Qu, Zehuan Yuan, and Jianfei Cai. Generative region-language pretraining for open-ended object detection. _arXiv preprint arXiv:2403.10191_ , 2024. 

- [27] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In _IEEE International Conference on Computer Vision (ICCV)_ , 2017. 

- [28] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. _arXiv preprint arXiv:2311.07575_ , 2023. 

- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Neural Information Processing Systems (NeurIPS)_ , 2023. 

- [30] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. _arXiv preprint arXiv:2303.05499_ , 2023. 

- [31] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _ICCV_ , 2021. 

- [32] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. _Neural Information Processing Systems (NeurIPS)_ , 2023. 

- [33] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_ , 2023. 

- [34] Peter Pinggera, Sebastian Ramos, Stefan Gehrig, Uwe Franke, Carsten Rother, and Rudolf Mester. Lost and found: detecting small road hazards for self-driving vehicles. In _International Conference on Intelligent Robots and Systems (IROS)_ , 2016. 

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International Conference on Machine Learning (ICML)_ , 2021. 

- [36] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. _Neural Information Processing Systems (NeurIPS)_ , 2015. 

- [37] Walter J Scheirer, Anderson de Rezende Rocha, Archana Sapkota, and Terrance E Boult. Toward open set recognition. _IEEE Transactions on Pattern Recognition and Machine Intelligence (PAMI)_ , 2012. 

- [38] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 14454–14463, 2021. 

- [39] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. _arXiv preprint arXiv:2303.15389_ , 2023. 

- [40] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- [41] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_ , 2023. 

- [42] Xin Wang, Thomas E. Huang, Joseph Gonzalez, Trevor Darrell, and Fisher Yu. Frustratingly simple few-shot object detection. In _International Conference on Machine Learning (ICML)_ , 2020. 

- [43] Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, and Shengjin Wang. Detecting everything in the open world: Towards universal object detection. In _IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2023. 

12 

- [44] Licheng Wen, Xuemeng Yang, Daocheng Fu, Xiaofeng Wang, Pinlong Cai, Xin Li, Tao Ma, Yingxuan Li, Linran Xu, Dengke Shang, et al. On the road with gpt-4v (ision): Early explorations of visual-language model on autonomous driving. _arXiv preprint arXiv:2311.05332_ , 2023. 

- [45] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. Exploring transformers for open-world instance segmentation. In _IEEE International Conference on Computer Vision (ICCV)_ , 2023. 

- [46] Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, et al. Efficientsam: Leveraged masked image pretraining for efficient segment anything. _arXiv preprint arXiv:2312.00863_ , 2023. 

- [47] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu. Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection. _Neural Information Processing Systems (NeurIPS)_ , 2022. 

- [48] Lewei Yao, Renjie Pi, Jianhua Han, Xiaodan Liang, Hang Xu, Wei Zhang, Zhenguo Li, and Dan Xu. Detclipv3: Towards versatile generative open-vocabulary object detection. _arXiv preprint arXiv:2404.09216_ , 2024. 

- [49] Haobo Yuan, Xiangtai Li, Chong Zhou, Yining Li, Kai Chen, and Chen Change Loy. Open-vocabulary sam: Segment and recognize twenty-thousand classes interactively. _arXiv preprint arXiv:2401.02955_ , 2024. 

- [50] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. _arXiv preprint arXiv:2306.14289_ , 2023. 

- [51] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, et al. Llava-grounding: Grounded visual chat with large multimodal models. _arXiv preprint arXiv:2312.02949_ , 2023. 

- [52] Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. Glipv2: Unifying localization and vision-language understanding. _Neural Information Processing Systems (NeurIPS)_ , 2022. 

- [53] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _arXiv preprint arXiv:2303.16199_ , 2023. 

- [54] Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Xianzheng Ma, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot. _arXiv preprint arXiv:2305.03048_ , 2023. 

- [55] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_ , 2023. 

- [56] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: deformable transformers for end-to-end object detection. In _International Conference on Learning Representations (ICLR)_ , 2021. 

- [57] Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. Llava-phi: Efficient multi-modal assistant with small language model. _arXiv preprint arXiv:2401.02330_ , 2024. 

13 

