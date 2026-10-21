# **SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors** 

**Chenyang Ma Kai Lu Ta-Ying Cheng Niki Trigoni Andrew Markham** University of Oxford 

```
chenyang.ma@cs.ox.ac.uk
```

## **Abstract** 

Current state-of-the-art spatial reasoning-enhanced VLMs are trained to excel at spatial visual question answering (VQA). However, we believe that higher-level 3D-aware tasks, such as articulating dynamic scene changes and motion planning, require a fundamental and explicit 3D understanding beyond current spatial VQA datasets. In this work, we present **SpatialPIN** , a framework designed to enhance the **spatial** reasoning capabilities of VLMs through **p** rompting and **in** teracting with priors from multiple 3D foundation models in a zero-shot, training-free manner. Extensive experiments demonstrate that our spatial reasoning-imbued VLM performs well on various forms of spatial VQA and can extend to help in various downstream robotics tasks such as pick and stack and trajectory planning. 

## **1 Introduction** 

Equipping vision-language models (VLMs) the capacities of spatial reasoning unlocks exciting applications, such as general-purpose reward annotation [52], robotic data generation [61], and grounding 3D object affordances [26, 38]. However, the spatial reasoning capabilities of VLMs on fine-grained spatial understanding tasks are somewhat limited. Current state-of-the-art (SOTA) spatial reasoning-enhanced VLM [12] is mostly tested on spatial visual question answering (VQA), such as determining objects’ relative positions and orientations; experiments on higher-level tasks, such as scene comparisons and trajectory planning, which require more nuanced comprehension, are underexplored. 

Many works enhance the spatial reasoning capabilities of VLMs by training/fine-tuning them on standard spatial VQA datasets [12]. As a result, VLMs primarily learn surface-level associations between image-text-data triplets. Given the scarcity and difficulty of obtaining spatially rich embodied data or high-quality human annotations for 3D-aware queries, we hypothesize that these VLMs may not generalize to questions outside their dataset distribution or adapt to more challenging tasks that require an advanced level of spatial understanding. 

Recent studies [73, 7, 65, 69] in image space understanding show that VLMs, equipped with internetscale language knowledge, and multimodal foundation models capture complementary knowledge that can be combined to conduct new tasks spanning both modalities without additional training. Given the recent advancements in 3D foundation models [4, 41, 29], this work explores whether there exists an alternative approach to enhance VLMs with higher-level spatial-awareness by incorporating 3D priors from these models. 

To this end, we propose **SpatialPIN** , a framework that utilizes progressive prompting and interactions between VLMs and 2D/3D foundation models as “free lunch” to enhance spatial reasoning capabilities in a zero-shot, training-free manner. By using these foundation models to decompose, comprehend, and reconstruct an explicit 3D representation, SpatialPIN grasps the core understanding of the 3D 

space presented by the 2D image. This allows generalizations to various 3D-aware tasks, from VQAs to 3D trajectory planning. 

We provide an extensive empirical study combining multiple off-the-shelf and handcrafted datasets, ranging from fundamental spatial questions regarding relative positions and orientations to providing fine-grained 3D information on objects’ locations, sizes, inclinations, and dynamic changes, and plan for robotics tasks with full 3D trajectories. Results show that this straightforward approach significantly outperforms SOTA VLMs trained from extensive spatial VQAs (see SpatialPIN examples in Figure 1), consolidating our belief that a truly 3D-aware VLM can actually be imbued by simply injecting explicit, fundamental knowledge of the 3D scene. With the entire framework being fully modularized, each component can be easily replaced with the latest improvements within its specific domain. 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0002-02.png)


**----- Start of picture text -----**<br>
User You are a robotic grasper. The<br>task is to grasp the carrot.<br>Determine your goal location and<br>closing distance in centimeters to<br>ensure a firm grip on the carrot.<br>To ensure a firm grip, the grasper should target the center of the carrot at coordinates  SpatialPIN<br>(0.7399, 0.5185, 0.1503). This midpoint takes into account the carrot's width of 5.44 cm.<br>The grasper should close to approximately 5.3 cm, just less than the carrot's width, for an<br>optimal grasp.<br>User The task is to pick up<br>the plastic bottle and<br>place it inside the<br>trash can. Plan for it.<br>First, align the plastic bottle with the trash can by rotating the bottle approximately 23.1  SpatialPIN<br>degrees around its local z-axis. This adjustment aligns the bottle's x-axis (0.9254, -<br>0.3781, 0.0246) with the trash can's x-axis (0.9982, 0.0398, -0.0447). Next, move the<br>bottle by [0, 0, 5] cm relative to the trash can's 3D center to place it securely inside the<br>trash can. Using RRT*, the planned 3D trajectory is ...”<br>1 2 3 4<br>**----- End of picture text -----**<br>


Figure 1: We present **SpatialPIN** , a framework to enhance the **spatial** reasoning capabilities of VLMs through **p** rompting and **in** teracting with 3D priors in a zero-shot, training-free manner. 

In summary, our main contributions are threefold: 

- We investigate the problem of equipping VLMs with 3D reasoning capabilities without fine-tuning on large spatial VQA datasets. 

- We propose SpatialPIN, a modular plug-and-play framework that progressively enhances VLM’s 3D reasoning capabilities by prompting and interacting with 3D foundational models. 

- We show that SpatialPIN unlocks 3D-aware applications including spatial VQA and both classic and novel robotics tasks, supported by extensive experiments. 

## **2 Related Work** 

**VLM Grounding** With the recent birth of powerful LLMs and VLMs [8, 40, 2], the task of VLM grounding, or combining generative language models with real-world data to adapt to specific cases, has gained significant popularity. Several recent works focused on fine-tuning these LLMs for a wide range of downstream applications, such as interactive decision making [34], multi-task agents [62], or even tasks in interactive environments [67, 11]. A close work to ours is Socratic Model [73], a framework of combining multiple foundation models to unleash LLMs in downstream tasks. However, this work still focuses on tasks in 2D pixel space understanding of images. There remains many challenges in the 3D world to combine information for full scene understanding, which we hope to tackle in our paper. 

**VLM Spatial Understanding** Many VLMs encompass the ability of image-space reasoning and understanding [13, 33, 40]. There are even efforts in incorporating these understandings into image space manipulations and editting [7, 65]. However, the current ability of VLMs to fully understand a 3D scene and the potential interactions within this scene is still rather limited. Several works build from this foundation and establish datasets to help with spatial reasoning/understanding [30, 39, 45]. Recently, SpatialVLM [12] proposed fine-tuning a VLM on 3D-VQA datasets to enhance the precision of VLMs on 3D understanding tasks. Nevertheless, using a 3D-VQA dataset only provides a partial picture to the complete 3D understanding of an image, and could lead to suboptimal performances under out-of-distribution tasks. In this work, we hope to introduce holistic 3D information from multiple 3D foundation models via prompting and interactions as a way to enhance VLMs with a comprehensive 3D understanding given RGB inputs. 

## **3 Method** 

Given an RGB image _I ∈_ R _[H][×][W][ ×]_[3] of a scene with _K_ unknown objects and a spatial task _Q_ , our goal is to inspire VLMs with spatial reasoning capabilities and solve _Q_ with fine-grained 3D understanding. 

2 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0003-00.png)


**----- Start of picture text -----**<br>
A) Understanding 2D Scene B) Understanding 3D Scene<br>Input RGB Image & Spatial Tasks Solving with Spatial Reasoning<br>Inpainted objects & background Perspective fields & metric depth  Reconstructed 3D scene 1) Spatial VQA<br>A: The distance between the laptop<br>and the camera is 29.16 cm.<br>Language-GuidedSegmentation  Scene Size Estimation Camera, Depth &  Zero-Shot 3D Scene Reconstruction 2) Planning with External ToolA: To stack the can on the bowl, the<br>3D trajectory is ...<br>Scene size  Object spatial<br>information information<br>Q: What is the distance between the laptop and the camera?Q: How do I stack the can on the  VLM Understanding Objects by Constraining VLM Coarse 3D Understanding perspectiveScene  VLM Fine-Grained 3D Understanding<br>bowl?<br>“Object 0: laptop of color rose  “Visual cues reasoning ... “w.r.t. your info, I understand:<br>gold, texture metallic at location left-center. Spatial data reasoning ...Conclusion: horizontal.” Obj 0 spatial context ...Obj 0 size ...<br>...” Obj 0 - Closest per direction ...<br>...”<br>**----- End of picture text -----**<br>


Figure 2: **SpatialPIN** . Our plug-and-play framework is fully modularized and designed for zero-shot deployment. Each module can be easily replaced with the latest updates. Exact prompts for VLMs are in Appendix. 

To prevent the models from overfitting to the standard problems from spatial VQA datasets [12], we hope to derive a method that utilizes fundamental 3D foundation models to provide explicit scene understandings, then leverage the generalization capabilities of VLMs to tackle unforeseen tasks—all within a zero-shot, training-free manner. 

Our modular pipeline, SpatialPIN, enhances VLMs’ spatial understanding of an image through progressive interactions with the scene decomposition, comprehension, and reconstruction processes with prompting. For image scene understanding (Sec. 3.1), we use VLM to describe objects by appearance and 2D location, complemented by language-guided segmentation and repainting models to obtain occlusion-free object masks. Elevating 2D understanding to coarse 3D (Sec. 3.2), we use metric depth estimation and perspective fields to estimate the 3D scene size and conduct perspective canonicalization with VLM. For fine-grained 3D understanding (Sec. 3.3), we partially reconstruct the 3D scene, with the full 3D representation of foreground objects and the background as a plane. With the reconstructed 3D scene, we summarize spatial information and prompt it to the VLM for various downstream tasks. 

## **3.1 2D Image Scene Understanding** 

**Prompting: Objects Understanding by Constraining** We start with querying VLM to identify and understand objects given _I_ . We explicitly ask VLM to describe the objects by precise color, texture, and 2D spatial locations. This step is vital for two reasons: 1) enhance VLM’s understanding of the objects, 2) differentiate between items of similar or identical categories and appearances. 

As a concrete example, given the left image of Fig. 2, VLM outputs: `“` 

```
object0:laptopofcolorrosegold,texturemetallicatlocationleft-center.
object1:cameraofcolorblack,texturesmoothatlocationcenter-right....”
```

**2D Representations Refinement** The concise descriptions of identified objects are used as input text prompts for a language-guided segmentation model, enabling the acquisition of _K_ segmentation masks _{Mk[occ] }[K] k_ =1[, with each mask corresponding to a unique object.] 

However, an object _i ∈_ [1 _, K_ ] may be occluded by other object(s), leading to an incomplete mask _Mi[occ]_ , which may be burdensome when we elevate the image to a 3D representation in the later stage. To resolve this, we create an inpainting mask, _Mi[inp]_ , for each object, in which all objects except the one itself are removed and replaced with white pixels. The inpainted masks are again fed to the language-guided segmentation model along with input text prompts such that occlusion-free object masks, _{Mk[of][}] k[K]_ =1[, are obtained.][This two-step segmentation process for object] _[ i]_[ is formulated as:] 

_Mi[occ]_ = seg( _I, τi_ ) _, Mi[of]_ = seg(inpaint( _Mi[inp]_ ) _, τi_ ) _,_ (1) 

where seg denotes the language-guided segmentation and _τi_ denotes the description of object _i_ . In practice, to cleanly remove objects without residual fragments for inpainting, we apply dilation to and expand the white areas. Inpainted background _Ibg_ is acquired by removing and replacing all objects with dilated white pixels. 

3 

## **3.2 Coarse 3D Scene Understanding** 

**Scene Size Estimation** Using the estimated metric depth [29] and estimated camera intrinsic matrix by finding field of view (FOV) through perspective fields [29], we backproject to determine the dimensions of the 3D spatial scene. 

**Prompting: Perspective Canonicalization** 3D information without any knowledge regarding the camera perspectives lead to ambiguities [12]. Consider a question “What is the orientation of the bowl relative to the laptop?” with the input scene in Fig. 2, but taken from a top-down perspective. VLMs may output `“downward to the left”` , but the correct answer should be "front-left" because humans perceive orientation from a horizontal angle. To address this, we provide the VLM with _I_ , estimated scene size, and maximum and minimum dimensions, allowing it to reason about the camera shot angle (horizontal/top-down/bottom-up). Scene size information helps differentiate shot angles by providing clues about spatial layout and object proportions. For instance, if the depth variation is small, the VLM can infer a top-down or bottom-up angle along with visual cues. 

As a concrete example, given the left image of Fig. 2, VLM outputs: 

```
“Visualcuesreasoning:Objectsareviewedfromtheside,indicatingthecameraispositionedhorizontally
withaslightelevation.
```

```
Spatialdatareasoning:Thedepthvariessignificantlyfrom57.50cmto115.00cm,indicatingthecamera
capturesthesceneacrossdifferentdistances,supportingahorizontalperspective.
Conclusion:horizontal.”
```

## **3.3 Fine-Grained 3D Scene Understanding** 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0004-07.png)


**----- Start of picture text -----**<br>
camera<br>(a) Side View (b) Active Camera View (c) Original Input Image<br>**----- End of picture text -----**<br>


We partially reconstruct the 3D scene with full representation of foreground objects while simplifying the inpainted background as a plane, as shown in Fig. 3(a). We summarize spatial information from the reconstructed scene and prompt it to the VLM. Please see our Appendix for implementation details about reconstruction. 

Figure 3: Our method of **partial 3D scene reconstruction (a).** The reconstructed scene (b) and the input image (c) show high alignment. 

**Scene Initialization** Given the occlusion-free object masks, _{Mk[of][}] k[K]_ =1[, we use single-view 3D reconstruction model [][41][] to acquire object 3D] models, _{Ok}[K] k_ =1[, with canonical poses determined during reconstruction.][Pinhole camera is set] at the origin, looking at positive depth-axis. With the estimated background plane size (Sec. 3.2), we move the background plane, _Obg_ (visually identical to _Ibg_ ), along the depth-axis to fit precisely within the camera. 

**Scene Reconstruction** To resolve the imprecision of backprojection, our goal is to position object 3D models into the reconstructed 3D scene without visual discrepancies and ensure accurate depth. Instead of using naive backprojection, for an object _i ∈_ [1 _, K_ ], we perform raycasting from object 3D center _t[c] i_[on the camera plane to object 3D center] _[ t][bg] i_[on the background plane with metric depth] _di_ . The 3D coordinate _ti_ of object _i_ is: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0004-12.png)


The rotation _Ri_ of the 6D pose of object _i_ , _Pi_ = [ _Ri | ti_ ] is explained previously. After integrating all 3D object models into the 3D scene, we refine each object’s scale to accurately reflect depth variations by rendering binary masks and evaluate the length of their contour lines relative to their occlusion-free masks, through the lens of the pinhole camera, _tc_ . 

We determine the principal axes (x-axis, y-axis, and z-axis) of each object using the minimal oriented bounding box (OBB), which is essential for unlock novel applications. 

**Prompting: Objects and Spatial Context Understanding** The reconstructed 3D scene from _I_ with accurate object poses and scales is denoted as _V_ 0. As the final step of progressive prompting, we feed VLM the fine-grained 3D information derived from _V_ 0, grounding on the canonicalized perspective (Sec. 3.2). For example, with the input image in Fig. 2 and a horizontal camera shot angle, depth corresponds to the positive y-axis (similarly, in a top-down/bottom-up view, depth is 

4 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0005-00.png)


**----- Start of picture text -----**<br>
User SpatialPIN (Ours) SpatialVLM<br>Is the laptop further to camera compared  The laptop is 75.0 cm from the camera,  Yes, the laptop is further away from the<br>to the mug? while the mug is 54.7 cm from the camera,  camera compared to the mug.<br>making the laptop further.<br>Measure the height difference between two  The height difference between bowl 1  The two white bowls on the table have<br>white bowls. (12.0 cm) and bowl 2 (9.4 cm) is  different heights, with one being taller<br>IaOR-VQA approximately 2.6 cm. than the other.<br>Along which axis (W, D, H) is the box  The tilt along the z-axis (Height) is the  The box is more tilted along the<br>more tilted? largest (about 0.798), indicating the box  horizontal axis, as it is leaning on the<br>is more tilted along the Height (H) axis. table and appears to be at a slight angle.<br>What is the relative angle between the  Using the dot product formula, the  The box is positioned at a diagonal angle<br>box and the bottle? relative angle between the box and the  to the bottle on the table.<br>IaAD-VQA bottle is approximately 47.41 degrees.<br>Which objects rotate across the images? By object principle x-axes, both the  The knife and the cutting board rotate<br>knife and the milk rotate around their  across the image, while the box of apples<br>vertical axes, presenting a different  remains stationary.<br>face.<br>How far does the milk box move? By observing the change in the 3D center,  The milk box moves from the counter to<br>the milk box moves a distance of  the stove, indicating that it is being<br>IrSD-VQA approximately 22.26 cm.  used for cooking or heating.<br>**----- End of picture text -----**<br>


Figure 4: **Qualitative examples of spatial VQA.** SpatialPIN outputs answers with fine-grained 3D reasoning. _Zoom in for better view._ 

the negative/positive z-axis) in a right-handed coordinate system. The width and height axes can be determined accordingly, aligning each axis’s orientation with human perception. 

We feed VLM a paragraph describing the objects’ poses, sizes, and principal axes in physical units, alongside their spatial relationships. To augment VLM’s understanding, we also feed _V_ 0 with visualized object axes (see Fig. 2B). Visualizing 3D spatial information is pivotal in improving VLMs’ understanding of 3D spatial contexts derived from 2D images, validated by 3DAxiesPrompts [37]. Yet, we want to **emphasize** that we do not feed hardcoded information, such as objects’ relative distances and inclinations, to VLMs. Instead, we aim for the summarized 3D information to enhance VLMs’ general spatial understanding. 

As a concrete example for the left image on Fig. 2: 

```
“Obj1spatialcontext:3Dcenter:[7.0,100.0,9.0]cm;X-axis(right):[0.9529,-0.2456,0.1779];
Y-axis(back):[-0.3528,0.8746,0.3327];Z-axis(up):[-0.1761,-0.3285,0.9279]
Obj1size:13.54cmx9.37cmx9.50cm(WxDxH)
Obj1closestperdirection:left:Obj0;right:Obj2...”
```

## **3.4 Combining External Tools for Downstream Tasks** 

By partially reconstructing the 3D scene with visual alignments, our framework enables VLMs to use tools like rapidly-exploring random tree star (RRT*) [31] to generate accurate, collision-free paths based on task specifications (more details in Appendix). This capability unlocks novel and interesting applications when combined with task-specific prompting techniques, shown in Experiments (Sec.4). 

## **4 Experiments** 

We conduct experiments to answer the following questions: 1) Does our framework enhance the general spatial reasoning capabilities of VLMs, and how well does it perform? 2) What novel applications does our framework unlock for VLMs, and how well do we perform in these applications? 3) How effective is each module in our framework? 

Since we evaluate our approach on a wide range of tasks to test VLMs’ higher-level spatial awareness, some tasks are novel and lack existing/open source datasets. Therefore, for all our experiments, we use a combination of 4 existing datasets and 2 hand-crafted datasets. 

**Implementations** The language-guided segmentation model is Language Segment-Anything [44] and the repainting model is LaMa [55]. We use One-2-3-45++ [41] for single-view 3D reconstruction, perspective fields [29] for camera intrinsic estimation, and ZoeDepth [4] for depth estimation. For partial 3D scene reconstruction, we use Blender [17] as the 3D software. All inference is run on 1 NVIDIA A10 GPU with 24GB RAM. 

## **4.1 Spatial Visual Question Answering** 

We experiment on the basic form of spatial VQA introduced by SpatialVLM (IaOR-VQA), and two new forms introducted by us (IaAD- & IrSD-VQA). For IaOR-VQA, please check SpatialVLM [12] for details. For IaAD- & IrSD-VQA, please see our Appendix. 

5 

**Intra-Image Object Relations VQA (IaOR-VQA)** As the basic form of spatial VQA, it involves spatial reasoning about object relative orientations and sizes. This is divided into qualitative (e.g., “is [A] in front of [B]”, “is [A] smaller than [B]”) and quantitative (e.g., “how far apart are [A] and [B]”, “measure the width of [A]”) questions. 

We follow the evaluation method of SpatialVLM [12]. Since SpatialVLM did not release their evaluation dataset, we reproduce one using RGBD images from NOCS [57] (object dataset), RT-1 [6], and BridgeData V2 [56] (robotics manipulation datasets). We sample 13, 20, and 20 distinct scenes from each. We generate QA pairs using the SpatialVLM data generation pipeline [51], followed by manual refinement. We check correctness for qualitative questions and calculate distances for quantitative questions. We annotate 300 qualitative and 200 quantitative spatial VQA pairs (SpatialVLM has 331 and 215 for each). 

**Intra-Image Angular Discrepancies VQA (IaAD-VQA** ) We propose a new form of Spatial VQA that needs spatial reasoning about objects’ inclinations. It includes qualitative (e.g., “is [A] tilted”, “is [A] more tilted than [B]”) and quantitative questions (e.g., “how many degrees is [A] tilted vertically”, “measure the angle between [A] and [B]”). 

Since this form of Spatial VQA involves out-of-plane rotations, YCBInEOAT [64] (object tracking dataset) is a suitable choice. We sample 30 scenes from it and annotate 50 questions each for qualitative and quantitative spatial VQA pairs. 

**Inter-Image Spatial Dynamics VQA (IrSD-VQA)** We further propose a more challenging form of Spatial VQA. Given two images with multiple objects, the objects in the second image may move, rotate, incline, or the image may have a change in camera angle. The VLM needs to reason about these changes. Example qualitative questions include “does [A] move, rotate, or incline”, “does [A] incline along the y-axis” while quantitative questions include “how far does [A] move”, “how many degrees does [A] rotate horizontally”. 

As it is difficult to find a dataset that meets these requirements, we craft our own. We capture 20 image pairs using an iPhone 12 Pro Max, with each image containing 1 _−_ 5 objects, and annotate 50 questions each for qualitative and quantitative spatial VQA pairs. 

**Results** The results in Tables 1 and 2 on qualitative and quantitative IaOR-VQA demonstrate that providing various VLMs fine-grained 3D information enhances their spatial reasoning capacities by a large margin. Surprisingly, VLMs with math and geometry reasoning capacities (e.g., GPT-4V, GPT-4o) show substantial improvements with this information. 

Table 1: **Qualitative IaOR-VQA.** We exclude comparisons to PaLI [14], PaLM-E [20], and PaLM 2-E [3] as they are not open source, and include experiments with GPT-4o [1] in addition to GPT-4V [47], LLaVA-1.5 [40], and InstructBLIP [18]. We use the HF version of SpatialVLM [51]. 

||**GPT-4V**<br>w/o ours<br>w ours|**GPT-4o**<br>w/o ours<br>w ours|**LLaVA-1.5**<br>w/o ours<br>w ours|**InstructBLIP**<br>**SpatialVLM**<br>w/o ours<br>w ours|
|---|---|---|---|---|
|Accuracy%|70.7<br>86.3|69.0<br>**87.3**|70.0<br>83.0|62.3<br>79.3<br>76.7|



Table 2: **Quantitative IaOR-VQA.** SpatialVLM measures the accuracy by the percentage of answers that fall within 0.5x to 2.0x of the ground truth value. We also evaluate within narrower ranges of 0.75x to 1.33x and 0.9x to 1.11x. “Output number” means VLMs produce number in the response instead of vague descriptions. 

||**GPT-4V**<br>w/o ours<br>w ours|**GPT-4o**<br>w/o ours<br>w ours|**LLaVA-1.5**<br>w/o ours<br>w ours|**InstructBLIP**<br>**SpatialVLM**<br>w/o ours<br>w ours|
|---|---|---|---|---|
|Output numbers%<br>In range [50, 200]%<br>In range [75, 133]%<br>In range [90, 111]%|0.8<br>98.5<br>0.0<br>73.5<br>0.0<br>69.5<br>0.0<br>54.5|31.5<br>**99.5**<br>14.0<br>**74.5**<br>8.5<br>**70.5**<br>3.0<br>**55.0**|23.5<br>97.0<br>16.5<br>43.0<br>6.0<br>29.0<br>2.0<br>14.5|28.5<br>98.5<br>91.0<br>8.0<br>31.5<br>33.5<br>2.5<br>22.0<br>20.5<br>0.0<br>11.5<br>7.5|



The results in Tables 3 and 4 demonstrate the effectiveness of our approach on both qualitative and quantitative IaOR-VQA and IrSD-VQA tasks. Notably, the performance on quantitative IaOR-VQA is suboptimal compared to quantitative IrSD-VQA, despite the latter being more challenging. We 

6 

Table 3: **Qualitative IaAD-VQA** & **IrSD-VQA.** Since we test SpatialPIN on one VLM backbone for our proposed spatial VQA, for fair comparison, we should use SpatialVLM backbone (PaLM 2-E [3]). However, since it is not open source, we use GPT-4o as our backbone, as it shows the most improvement with our framework. 

||**IaAD-VQA**<br>GPT-4o<br>GPT-4o + ours<br>SpatialVLM|**IaAD-VQA**<br>GPT-4o<br>GPT-4o + ours<br>SpatialVLM|**IrSD-VQA**<br>GPT-4o<br>GPT-4o + ours<br>SpatialVLM|**IrSD-VQA**<br>GPT-4o<br>GPT-4o + ours<br>SpatialVLM|
|---|---|---|---|---|
|Accuracy%|68|**84**<br>62||64<br>**82**<br>54|
|||Table 4: **Quantitative IaAD-VQA**||&**IrSD-VQA.**|
|||**IaAD-VQA**<br>GPT-4o<br>GPT-4o + ours<br>SpatialVLM||**IrSD-VQA**<br>GPT-4o<br>GPT-4o + ours<br>SpatialVLM|
|Output numbers%<br>In range [50, 200]%<br>In range [75, 133]%<br>In range [90, 111]%||38<br>**100**<br>66<br>8<br>**64**<br>12<br>2<br>**42**<br>6<br>0<br>**30**<br>2||30<br>**100**<br>78<br>10<br>**68**<br>26<br>4<br>**54**<br>12<br>2<br>**38**<br>4|



believe this is because, for quantitative IrSD-VQA, the VLM sometimes confuses the camera and world coordinate frames, comparing the object’s principal axes with the world axes to reason about changes in angles. 

Fig. 4 presents qualitative examples on all forms of spatial VLM. 

## **4.2 Robotics Pick and Stack** 

Pick and stack is a classic robotics task. Given a robot’s egocentric observation of a scene with multiple objects and a task description, our pipeline uses traditional planning to solve the problem. This task demands advanced spatial reasoning, as the model must comprehend 3D locations, sizes, and physical properties of the objects (i.e., how much to grasp and how high to drop? Is the object deformable or articulated so the robotic grasper needs to grasp more firmly?). For instance, grasping and stacking a soft toy bear on a cube is significantly different from stacking a solid apple on a mug. The model reasons about grasping and stacking policies, directly outputting 3D trajectories for the robot’s end effector using traditional path planning algorithm as external tool. 

**Set-Up** We set up the pick-and-stack problem in the ManiSkill [22] simulator, applying real-world physics properties. Rigid and articulated objects are chosen from the YCB dataset [10] and are randomly allocated on the table within the robotic arm’s reach, with observations from different perspectives. We create 50 scenes. Since robot observations from simulated scenes suffer from sim2real gap and consider that most real-world robots have depth sensors, we use ground truth camera matrix and depth. 

We compare our method to the following baselines: 1) direct 3D information output from our framework without GPT-4o [1] reasoning about physics and object properties and 2) SpatialVLM with our RRT* trajectory generation module. 

**Results** Table 5 shows the results, with a qualitative example demonstrated in Fig. 5. The results indicate that using precise 3D information from our framework significantly improves the success rate, and incorporating VLM reasoning further enhances performance. 

## **4.3 Discovering and Planning for Robotics Tasks from a Single Image** 

We present a novel task that requires advanced spatial reasoning capacities of VLMs. Given a single RGB image of any scene comprising unknown environments and objects, the VLM discovers potential tasks and plans their execution with full 3D trajectories, with the **motivation** that it can be used for robot learning in future research. To solve this complex task and visualize the execution using our framework, we introduce: 1) a task proposal approach using VLM, 2) a novel axes-constrained 3D planning approach that enables spatial reasoning-imbued VLM to plan the object motion based on the proposed tasks by specifying waypoints. Please see Appendix for the pipeline and details. 

**Dataset** We create a diverse evaluation dataset by combining self-captured photos (38) using an iPhone 12 Pro Max and scenes (13) from NOCS [57]. Our dataset covers diverse scenes ( _e.g._ , office, 

7 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0008-00.png)


**----- Start of picture text -----**<br>
“Banana size: 19.8 cm x 6.9 cm x 3.66 cm (WxDxH). Cracker box size: 9.21 cm x 4.51 cm x 17.6 cm (WxDxH). Task: grasp<br>and stack the banana on the cracker box. Determine the grasper's closing distance and the optimal drop height in cm,<br>User using the image and spatial information.”<br>1 2 3 4<br>“The robot should grasp the banana at its thinnest side with a width of approximately 3.5 cm (just less than the<br>banana's 3.66 cm) for a firm grip. It should then lift the banana to 22.26 cm (cracker box height of 17.6 cm plus<br>banana height of 3.66 cm plus a 1 cm margin) and drop it from 22.16 cm to stack it securely on the cracker box. Using<br>SpatialPIN RRT*, the planned 3D trajectory is ...”<br>“The knife is used to slice the orange into halves or wedges on the cutting board.”<br>Input Planned Task Trajectories<br>**----- End of picture text -----**<br>


Figure 5: **Qualitative examples of pick and stack (top) and task trajectory planning (bottom).** SpatialPIN successfully outputs picking and stacking policies using spatial reasoning and plans 3D trajectories with geometric awareness to align with task descriptions. 

Table 5: **Pick and stack.** We classify the success rates into: 1) successfully picked, 2) successfully picked and contacted the target object but slipped/collided, and 3) successfully picked and stacked. 

||**GPT-4o + ours**|**Our Direct 3D Info**|**SpatialVLM + RRT***|
|---|---|---|---|
|Picked%|**44**|28|16|
|Picked&contacted%|8|12|6|
|Picked&stacked%|**36**|16|10|



kitchen, bathroom), and features a rich diversity of object categories (116) and quantities (185), with each image containing 1 _−_ 7 objects and 1 _−_ 3 tasks proposed for each object (278 tasks/planned trajectories in total). The dataset’s diversity is further enhanced by the variety of perspectives ( _e.g._ , frontal, top-down, side views). This deliberate choice of diverse angles, both in our own image capturing process and through the random extraction of frames from NOCS, aims to simulate a realistic and challenging array of scenes for evaluation. See Appendix for statistics and visuals. 

**Qualitative Demonstration** We present a qualitative example in Fig. 5. Additional examples in Appendix shows our framework’s capability to produce diverse and accurate task trajectories spanning various scenes and tasks. 

**Human Evaluation: User Study** We rely on human preference evaluation as one of our quantitative metrics. We ask 25 users to rate 5 translation and 5 rotation task executions in terms of task description alignment. For these complex context-dependent manipulation tasks, we instead ask users to judge 10 executions relative to human action, and to encapsulate their perception of the action in our with a single sentence. These sentence description will be used to test human understanding of our planned trajectories (please see Appendix). Note that our user study size is similar to those representative works such as ControlNet [74] and Prompt-to-Prompt [24]. Results in Table. 6. 

Table 6: **User study.** Ratings (scale 1 _−_ 5) are averaged. 

||Rotation<br>Translation<br>Manipulation|**Rating** <br>4.58<br>4.43<br>4.29|_↑_|
|---|---|---|---|



**Machine Understanding** We assess the interpretability of our generated task executions from a machine’s perspective using SOTA video understanding model, Video-LLaVA-7B [35]. We use two approaches: binary classification and descriptive generation. For classification, we feed the model with the task descriptions generated by VLM and ask question (is the video doing _. . ._ ?). In generation, we prompt Video-LLaVA-7B to articulate its interpretation of our task executions. To quantify the correspondence between the model’s perception and the tasks, we use OpenCLIP cosine similarity score [15]. 

8 

Table 8: **Ablation study.** For quantitative IaOR-VQA, the accuracy is measured by the answers that fall within 0.75x to 1.33x of the ground truth value. 

||**Overall Design**<br>ShAPO<br>SAM-6D + 3D models<br>SpatialVLM|**2D Understanding**<br>w/o objects|**3D Understanding**<br>**Ours**<br>w/o coarse<br>w/o fne-grained<br>w/o both|
|---|---|---|---|
|Qualitative<br>Quantitative|36.7<br>48.0<br>81.3<br>29.5<br>37.0<br>62.5|68.3<br>54.5|76.0<br>63.3<br>61.7<br>**87.3**<br>64.5<br>50.5<br>58.0<br>**70.5**|



However, we find that even SOTA video understanding model shows limited performance. To assess false positive rate in classification, we deliberately misalign the sequence of generated task executions with their corresponding task descriptions, expecting a theoretical accuracy of 0%. Contrary to expectations, Video-LLaVA-7B reports a false positive rate of 36.3%. To adjust for this anomaly, we subtract this rate from the model’s raw accuracy for correctly aligned video-task pairs. This method, while unconventional, provides a more fair and reasonable evaluation of machine video understanding, underscoring the current challenges faced by video understanding models in accurately interpreting complex video content. Results in Table. 7. 

||Table 7:|Results for|
|---|---|---|
||machine understanding||
||(classifcation and gen-<br>eration) on 278 task ex-<br>ecutions.||
|||**Machine**|
||Raw Acc_↑_|0.974|
||Fal-Pos Rate|_↓_<br>0.363|
||True Acc_↑_|0.611|
||OpenCLIP_↑_|0.636|



## **4.4 Ablation Study** 

We evaluate the effectiveness of each module in our framework on IaOR-VQA by 1) seeking alternative designs of the overall pipeline and 2) removing each component in our ablations. 

**Overall Design** To demonstrate our framework’s generalization across a wide range of objects, We replace our 2D + 3D pipeline with: 1) SOTA mesh-free single image object pose and size estimation model, ShAPO [27], 2) SOTA mesh-based single image object pose and size estimation model, SAM-6D [36], and feeds it with the object 3D model reconstructed by One-2-3-45++ [41], and 3) the data generation backbone of SpatialVLM [12]. Since models 1) and 2) do not provide language annotations for their outputs, we first summarize the numerical outputs using our approach in Sec. 3.3. Then, GPT-4V identifies QA pairs. 

**Removing 2D Understanding Module** In this case, the VLM no longer examines the objects through prompting, and only the object name is input into the language-guided segmentation model. **Removing 3D Understanding Modules** This means there is no scene size estimation, and the VLM does not conduct perspective canonicalization. During 3D scene reconstruction, we assume the image plane width to be 1 meter. 

To validate the fine-grained 3D scene understanding module, we replace object mask raycasting with backprojection using the object’s 2D center and remove the object scale calibration. 

To demonstrate the overall effectiveness of our 3D understanding modules, we simply backproject the input image with the estimated metric depth. 

**Results** Table 8 demonstrates the effectiveness of each module in our framework. The results also highlight the limitations of using off-the-shelf SOTA mesh-free and mesh-based single-image object pose and size estimation methods as our backbone. These methods are not language-driven and may struggle to generalize to novel objects in diverse input scenes. 

## **5 Discussion and Conclusion** 

We present **SpatialPIN** , a framework designed to enhance the **spatial** reasoning capabilities of VLMs through **p** rompting and **in** teracting with 3D priors in a zero-shot, training-free manner. We see our work as a step towards equipping VLMs with more generalized spatial reasoning capacities, demonstrated through applications in various forms of spatial VQA and both traditional and novel robotics tasks. 

**Limitations** Readers may be curious about the inference speed of our framework. The bottleneck is the 3D object reconstruction process and the API call to closed-source VLMs ( _∼_ 20 seconds per image). However, we want to highlight that this process runs only once per image, and the speed is expected to improve with future versions of 3D foundation models. 

9 

## **References** 

- [1] Open AI. Hello gpt-4o. `https://openai.com/index/hello-gpt-4o/` , 2024. 

- [2] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_ , 2023. 

- [3] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, and et al. Palm 2 technical report. _arXiv preprint arXiv:2305.10403_ , 2023. 

- [4] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. _arXiv preprint arXiv:2302.12288_ , 2023. 

- [5] Reiner Birkl, Diana Wofk, and Matthias Müller. Midas v3.1 - A model zoo for robust monocular relative depth estimation. _arXiv preprint arXiv:2307.14460_ , 2023. 

- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, and et al. RT-1: robotics transformer for real-world control at scale. In _Robotics: Science and Systems_ , 2023. 

- [7] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In _Conference on Computer Vision and Pattern Recognition_ , pages 18392–18402. IEEE, 2023. 

- [8] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In _Advances in Neural Information Processing Systems_ , 2020. 

- [9] Dingding Cai, Janne Heikkilä, and Esa Rahtu. OVE6D: object viewpoint encoding for depthbased 6d object pose estimation. In _Conference on Computer Vision and Pattern Recognition_ , pages 6793–6803. IEEE, 2022. 

- [10] Berk Çalli, Arjun Singh, Aaron Walsman, Siddhartha S. Srinivasa, Pieter Abbeel, and Aaron M. Dollar. The YCB object and model set: Towards common benchmarks for manipulation research. In _International Conference on Advanced Robotics_ , pages 510–517. IEEE, 2015. 

- [11] Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and PierreYves Oudeyer. Grounding large language models in interactive environments with online reinforcement learning. In _International Conference on Machine Learning_ , pages 3676–3713, 2023. 

- [12] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. _arXiv preprint arXiv:2401.12168_ , 2024. 

- [13] Ting Chen, Saurabh Saxena, Lala Li, David J. Fleet, and Geoffrey E. Hinton. Pix2seq: A language modeling framework for object detection. In _International Conference on Learning Representations_ , 2022. 

- [14] Xi Chen, Xiao Wang, Soravit Changpinyo, A. J. Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V. Thapliyal, James Bradbury, and Weicheng Kuo. Pali: A jointly-scaled multilingual languageimage model. In _International Conference on Learning Representations_ , 2023. 

10 

- [15] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In _Conference on Computer Vision and Pattern Recognition_ , pages 2818–2829. IEEE, 2023. 

- [16] Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. DIML/CVL RGBD dataset: 2m RGB-D images of natural indoor and outdoor scenes. _arXiv preprint arXiv:2110.11590_ , 2021. 

- [17] Blender Online Community. _Blender - a 3D modelling and rendering package_ . Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 

- [18] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In _Advances in Neural Information Processing Systems_ , 2023. 

- [19] Murtaza Dalal, Tarun Chiruvolu, Devendra Singh Chaplot, and Ruslan Salakhutdinov. Plan-seqlearn: Language model guided RL for solving long horizon robotics tasks. In _CoRL Workshop on Learning Effective Abstractions for Planning_ , 2023. 

- [20] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In _International Conference on Machine Learning_ , pages 8469–8488, 2023. 

- [21] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The KITTI dataset. _The International Journal of Robotics Research_ , pages 1231–1237, 2013. 

- [22] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In _International Conference on Learning Representations_ , 2023. 

- [23] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In _Conference on Learning Representations_ , 2023. 

- [24] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross-attention control. In _International Conference on Learning Representations_ , 2023. 

- [25] Ming-Kuei Hu. Visual pattern recognition by moment invariants. _IRE Transactions on Information Theory_ , pages 179–187, 1962. 

- [26] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. _arXiv preprint arXiv:2403.08248_ , 2024. 

- [27] Muhammad Zubair Irshad, Sergey Zakharov, Rares Ambrus, Thomas Kollar, Zsolt Kira, and Adrien Gaidon. Shapo: Implicit representations for multi-object shape, appearance, and pose optimization. In _European Conference on Computer Vision_ , pages 275–292, 2022. 

- [28] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. _Robotics and Automation Letters_ , pages 3019–3026, 2020. 

- [29] Linyi Jin, Jianming Zhang, Yannick Hold-Geoffroy, Oliver Wang, Kevin Blackburn-Matzen, Matthew Sticha, and David F. Fouhey. Perspective fields for single image camera calibration. In _Conference on Computer Vision and Pattern Recognition_ , pages 17307–17316. IEEE, 2023. 

- [30] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In _Conference on Computer Vision and Pattern Recognition_ , pages 1988–1997. IEEE, 2017. 

11 

- [31] Sertac Karaman and Emilio Frazzoli. Sampling-based algorithms for optimal motion planning. _The International Journal of Robotics Research_ , pages 846–894, 2011. 

- [32] Kourosh Khoshelham and Sander Oude Elberink. Accuracy and resolution of kinect depth data for indoor mapping applications. _Sensors_ , pages 1437–1454, 2012. 

- [33] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. _arXiv preprint arXiv:2304.02643_ , 2023. 

- [34] Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, Jacob Andreas, Igor Mordatch, Antonio Torralba, and Yuke Zhu. Pre-trained language models for interactive decision-making. In _Advances in Neural Information Processing Systems_ , 2022. 

- [35] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. _arXiv preprint arXiv:2311.10122_ , 2023. 

- [36] Jiehong Lin, Lihua Liu, Dekun Lu, and Kui Jia. SAM-6D: segment anything model meets zero-shot 6d object pose estimation. _arXiv preprint arXiv:2311.15707_ , 2023. 

- [37] Dingning Liu, Xiaomeng Dong, Renrui Zhang, Xu Luo, Peng Gao, Xiaoshui Huang, Yongshun Gong, and Zhihui Wang. 3daxiesprompts: Unleashing the 3d spatial task capabilities of GPT-4V. _arXiv preprint arXiv:312.09738_ , 2023. 

- [38] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. MOKA: open-vocabulary robotic manipulation through mark-based visual prompting. _arXiv preprint arXiv:2403.03174_ , 2024. 

- [39] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. _Transactions of the Association for Computational Linguistics_ , 2023. 

- [40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Advances in Neural Information Processing Systems_ , 2023. 

- [41] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. _arXiv preprint arXiv:2311.07885_ , 2023. 

- [42] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6d: Generalizable model-free 6-dof object pose estimation from RGB images. In _European Conference on Computer Vision_ , pages 298–315, 2022. 

- [43] Chenyang Ma, Xinchi Qiu, Daniel Beutel, and Nicholas Lane. Gradient-less federated gradient boosting tree with learnable learning rates. In _Proceedings of the 3rd Workshop on Machine Learning and Systems_ , pages 56–63, 2023. 

- [44] Luca Medeiros. Language segment-anything. `https://github.com/luca-medeiros/ lang-segment-anything` , 2023. 

- [45] Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, and Tanmay Gupta. OBJECT 3DIT: Language-guided 3d-aware image editing. In _Advances in Neural Information Processing Systems_ , 2023. 

- [46] Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Vincent Lepetit, and Tomas Hodan. CNOS: A strong baseline for cad-based novel object segmentation. In _International Conference on Computer Vision Workshops_ , pages 2126–2132. IEEE, 2023. 

- [47] OpenAI. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- [48] Xinchi Qiu, Heng Hen, Wanru Zhao, Pedro Porto Buarque de Gusmao, and Nicholas Donald Lane. Efficient vertical federated learning with secure aggregation. In _Federated Learning Systems (FLSys) Workshop@ MLSys 2023_ , 2023. 

- [49] Xinchi Qiu, Heng Pan, Wanru Zhao, Chenyang Ma, Pedro PB Gusmao, and Nicholas D Lane. vfedsec: Efficient secure aggregation for vertical federated learning via secure layer. _arXiv preprint arXiv:2305.16794_ , 2023. 

- [50] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , pages 1623–1637, 2022. 

12 

- [51] remyxai. Vqasynth, 2024. GitHub repository. 

- [52] Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, and David Lindner. Visionlanguage models are zero-shot reward models for reinforcement learning. _arXiv preprint arXiv:2310.12921_ , 2023. 

- [53] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGBD images. In _European Conference on Computer Vision_ , pages 746–760, 2012. 

- [54] Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, C. Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. BEHAVIOR: benchmark for everyday household activities in virtual, interactive, and ecological environments. In _Conference on Robot Learning_ , pages 477–490, 2021. 

- [55] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In _Winter Conference on Applications of Computer Vision_ , pages 3172–3182. IEEE, 2022. 

- [56] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In _Conference on Robot Learning (CoRL)_ , 2023. 

- [57] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J. Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In _Conference on Computer Vision and Pattern Recognition_ , pages 2642–2651. IEEE, 2019. 

- [58] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. IRS: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In _International Conference on Multimedia and Expo_ , pages 1–6. IEEE, 2021. 

- [59] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian A. Scherer. Tartanair: A dataset to push the limits of visual SLAM. In _International Conference on Intelligent Robots and Systems_ , pages 4909–4916. IEEE, 2020. 

- [60] Yen-Jen Wang, Bike Zhang, Jianyu Chen, and Koushil Sreenath. Prompt a robot to walk with large language models. _arXiv preprint arXiv:2309.09969_ , 2023. 

- [61] Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. _arXiv preprint arXiv:2311.01455_ , 2023. 

- [62] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with llms enables open-world multi-task agents. In _Advances in Neural Information Processing Systems_ , 2023. 

- [63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems_ , 2022. 

- [64] Bowen Wen, Chaitanya Mitash, Baozhang Ren, and Kostas E. Bekris. se(3)-tracknet: Datadriven 6d pose tracking by calibrating image residuals in synthetic domains. In _International Conference on Intelligent Robots and Systems_ , pages 10367–10373. IEEE, 2020. 

- [65] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llm-controlled diffusion models. _arXiv preprint arXiv:2311.16090_ , 2023. 

- [66] Ke Xian, Jianming Zhang, Oliver Wang, Long Mai, Zhe Lin, and Zhiguo Cao. Structure-guided ranking loss for single image depth prediction. In _Conference on Computer Vision and Pattern Recognition_ , pages 608–617. IEEE, 2020. 

- [67] Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, and Zhiting Hu. Language models meet world models: Embodied experiences enhance language models. In _Advances in Neural Information Processing Systems_ , 2023. 

13 

- [68] Fengyu Yang and Chenyang Ma. Sparse and complete latent organization for geospatial semantic segmentation. In _Conference on Computer Vision and Pattern Recognition_ , pages 1809–1818, 2022. 

- [69] Fengyu Yang, Chenyang Ma, Jiacheng Zhang, Jing Zhu, Wenzhen Yuan, and Andrew Owens. Touch and go: Learning from human-collected vision and touch. _Advances in Neural Information Processing Systems_ , pages 8081–8103, 2022. 

- [70] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. _arXiv preprint arXiv:2401.10891_ , 2024. 

- [71] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In _Conference on Computer Vision and Pattern Recognition_ , pages 1787–1796. IEEE, 2020. 

- [72] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In _Conference on Robot Learning_ , pages 1094–1100, 2019. 

- [73] Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S. Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, and Pete Florence. Socratic models: Composing zero-shot multimodal reasoning with language. In _International Conference on Learning Representations_ , 2023. 

- [74] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In _International Conference on Computer Vision_ . IEEE, 2023. 

- [75] Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. Texygen: A benchmarking platform for text generation models. In _SIGIR_ , pages 1097–1100. ACM, 2018. 

14 

## **Appendix for** _**SpatialPIN**_ 

## **A Overview** 

This Appendix includes: 1) more technical details about our partial 3D scene reconstruction, 2) additional details, templates, and visualizations of our proposed two forms of SpatialVQA, 3) implementation details on our proposed application: discovering and planning for robotics tasks from a single image (task proposal, axes-constrained motion planning through waypoints, and trajectory generation and smoothing), 4) more experiments on our proposed application (dataset statistics, additional qualitative demonstrations, human understanding, and task diversity), and 5) prompt details for VLMs. 

## **B Partial 3D Scene Reconstruction Details** 

**Metric Depth Estimation** Because a significant portion of depth estimation model [70, 50, 4, 5, 68] is trained on depth datasets with depth data determined by sensors [21, 53] and stereo matching [71, 16, 58, 59, 66], we assume that the predicted normalized depth is the perpendicular distance to the camera plane, instead of a straight line from the object to the camera lens [32]. 

**Camera Intrinsic Estimation** Given an RGB image, With the estimated vertical field of view (FOV), _θv_ , through perspective fields [29], the camera focal length _f_ can be found by: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0015-06.png)


where _H_ is the image height in pixels. 

**Object 6D Pose Estimation** Single-view 3D reconstruction model reconstructs mesh _Oi_ of object _i_ at the 3D origin, in the coordinate frame set by the input mask _Mi[of]_[, and captures] _[ O][i]_[ by a pinhole] camera. This camera, with 6D pose _Pc_ = [ _Rc | t[origin] c_ ], captures _Oi_ ’s canonical pose within the image. Thus, we can restore all objects’ canonical poses across all images by identifying _Pc_ . 

We use One-2-3-45++ [41], which provides the camera pose. For models without this information, we develop an efficient method to determine the pose by comparing object masks with rendered 3D model templates, inspired from matching-based 6D pose estimation works [36, 46, 9, 42]. 

We generate a set of object templates, denoted as _{Tj[i][}][J] j_ =1[, each rendered from the object’s 3D model] _Oi_ . These templates are created by positioning the camera at various locations on an icosphere surrounding the object in _SE_ (3) space, which simulates a spherical coverage around the object to capture its geometry from all angles uniformly. For each template _Tj[i]_[, we compute a matching score] against the occlusion-free object mask _Mi[of]_[.] We propose a simple yet effective score matching method. We draw a bounding rectangle around the segmented object inside _Mi[of]_ and across all _{Tj[i][}][J] j_ =1[, and crop the bounding rectangle.][We then] calculate the shape similarity between the contour line of cropped _Mi[of]_ and that of each cropped _Tj[i]_[using Hu moments [][25][].][Additionally, we crop and resize the bounding rectangle to the same] dimension, and evaluate the similarity based on the pixel area of the cropped and resized masks. Our score matching method can be formulated as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0015-11.png)


This dual approach allows for a comprehensive comparison that incorporates both the geometric configuration and the scale of the object representations. The best-matched template can be found by arg min _[J] j_ =1 _[L]_[(] _[M][ of] i[, T] j[ i]_[)][.] 

15 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0016-00.png)


**----- Start of picture text -----**<br>
Intra-Image Object Relations VQA<br>Intra-Image Angular Discrepancies VQA<br>Inter-Image Spatial Dynamics VQA<br>**----- End of picture text -----**<br>


Figure 6: **Example input images of all forms of spatial VQA.** 

**Object Scale Calibration** After integrating all 3D object models _{Ok}[K] k_ =1[into][the][3D][scene,] each with a pose _{Pk_ = [ _Rk | tk_ ] _}[K] k_ =1[and an initial scale] _[ {][S] k[init] }[K] k_ =1[set by the single-view 3D] reconstruction model, we refine their scales to accurately reflect depth variations ( _e.g._ , moving an apple from close to the camera to a distant corner reduces its apparent size). Through the lens of the pinhole camera with pose _Pc_ , we render binary masks for each object and evaluate the length of their contour lines relative to their occlusion-free masks. The adjusted, final scale of object _i_ can be expressed as: 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0016-03.png)


where _Mi[rend]_ is the rendered mask of object _i_ . 

## **C Additional Experiments and Details** 

## **C.1 Spatial Visual Question Answering** 

**Intra-Image Angular Discrepancies VQA (IaAD-VQA)** For annotation, since YCBInEOAT [64] offers ground truth object 6D poses, we first determine the table/ground plane using the principal axes of objects resting on it (if present). Then, we calculate the angles between the principal axes of different objects to annotate a list of qualitative and quantitative QA pairs. We provide a subset of the question template below. 

Qualitative questions: 

```
Is[A]tilted?
Is[A]tiltedtotheleft?
Is[A]inclinedtotheback?
Is[A]moretiltedthan[B]?
Is[A]moretiltedthan[B]totheback?
Is[A]moreinclinedthan[B]totheright?
Is[A]leaningtowards[B]vertically?
Is[A]straighterthan[B]?
Alongwhichaxis(W,D,H)is[A]moretilted?
Whichobject(s)arenotupright?
Howmanyobject(s)arenotupright?
```

Quantitative questions: 

```
Whatistheinclinationangleof[A]alongtheverticalaxis?
Howmanydegreesis[A]tiltedhorizontally?
Calculatetheangleoftiltfor[A]towardsback.
```

16 

```
Measurethetiltof[A]relativetothefront.
Whatistherelativeanglebetween[A]and[B]?
Measuretheanglebetween[A]and[B].
Whatistheanglebetweenthehorizontalaxisof[A]and[B]?
Howmuchis[A]inclinedalongthedepthaxiscomparedto[B]?
Measuretheinclinationdifferencealongtheverticalaxisfor[A]and[B].
Measuretheangulardeviationof[A]and[B]alongtheverticalaxis.
Determinetheangulardifferencebetweenthedepthaxesof[A]and[B].
Determinethetiltdifferencebetween[A]and[B]alongthehorizontalaxis.
Comparetheanglesoftiltfor[A]and[B]alongtheverticalaxis.
```

**Inter-Image Spatial Dynamics VQA (IrSD-VQA)** For annotation, we manually measure the changes in objects’ locations and angles between two photos. To measure the change in camera shot angle, we record the change in the angle of the tripod to which the iPhone 12 Pro Max is attached. We provide a subset of the question template below. 

Qualitative questions: 

```
Does[A]move?
Does[A]rotate?
Does[A]rotateclockwise?
Does[A]incline?
Does[A]movetotheright?
Does[A]movecloserto[B]?
Does[A]becomemoreupright?
Does[A]inclinemoretotheback?
Doestheanglebetween[A]and[B]becomesmaller?
Alongwhichdirectiondoes[A]move?
Alongwhichaxis(W,D,H)does[A]rotate?
Whichobject(s)move?
Howmanyobject(s)rotate?
Howmanyobject(s)becomemoretiltedtotheback?
Doesthecamerashotanglechange?
Alongwhichaxis(W,D,H)doesthecamerashotanglechange?
```

Quantitative questions: 

```
Howfardoes[A]movevertically?
Howfardoes[A]movehorizontally?
Howfardoes[A]movetowardstheback?
Howmanydegreesdoes[A]rotateclockwise?
Howmanydegreesdoes[A]rotatecounterclockwise?
Whatisthetotaldistance[A]movesfromitsoriginalposition?
Calculatetheangleofinclinationof[A]inthesecondimage.
Measurethetiltof[A]relativetothefirstimage.
Whatisthechangeinheightof[A]fromthefirsttothesecondimage?
Howmuchdoesthedistancebetween[A]and[B]change?
Howmuchdoes[A]inclinetowardstheleftcomparedtothefirstimage?
Whatistheangulardisplacementof[A]towardstheright?
Measuretherotationangleof[A]aboutitsownaxis.
Whatisthenewdistancebetween[A]and[B]inthesecondimage?
Calculatethedifferenceintheinclinationangleof[A]betweenthetwoimages.
Determinethechangeinangleof[A]relativetothegroundplane.
Whatistherelativemovementof[A]withrespectto[B]?
Measuretheangulardeviationof[A]and[B]alongtheverticalaxis.
Byhowmanydegreeshasthecamerashotanglechanged?
Alongwhichaxis/axeshasthecamerashotanglechanged?
Howdoes[A]appeartomoveifwedonotaccountforthecamerashotanglechange?
Whatistheperceivedchangeinorientationof[A]duetothecameraanglechange?
```

17 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0018-00.png)


**----- Start of picture text -----**<br>
A) Understanding 2D Scene B) Understanding 3D Scene C) Planning & Executing Tasks<br>Inpainted objects & background Object 3D models Reconstructed 3D scene Generated, smoothed trajectory<br>Language-Guided Single-View 3D Object  Zero-Shot 3D Scene  Trajectory Generation<br>Segmentation  Reconstruction Reconstruction & Smoothing<br>All spatial  Sparse<br>information waypoints<br>Input RGB Image VLM Understanding Objects by Constraining VLM Task Proposal VLM Coarse-Fine 3D Understanding VLM Motion Planning<br>“Object 0: laptop of color rose  “Task name: Can to Bowl Transfer “w.r.t. your info, I understand: “Task Name: Can to Bowl Transfer<br>gold, texture metallic at location left-center....” Description: Pick up the can and pour its contents into the bowl....” Obj 0 spatial context ...Obj 0 size ... Obj 0 - Closest per direction ......” 1. Move Manipulating Obj [3] to [6, 0, 7] cm relative to Target Obj [4]'s local [x, y, z] axes....” Generated Task Execution<br>**----- End of picture text -----**<br>


Figure 7: **Pipeline for discovering and planning for robotics tasks from a single image.** It incorporate the task proposal and motion planning modules based on SpatialPIN. 

```
Howdoesthepositionof[A]changerelativetothecameraangledifference?
Whatistheactualmovedistanceof[A]whenaccountingforthecamerashotanglechange?
```

**Dataset Examples** We illustrate example input images of all forms of spatial VQA in Fig. 6. 

## **C.2 Discovering and Planning for Robotics Tasks from a Single Image** 

Given a single RGB image of a scene with unknown environments and objects, the VLM identifies potential tasks and plans their execution using full 3D trajectories, complete with visualization. Figure 7 shows the pipeline, which incorporates task proposal and axes-constrained motion planning modules into SpatialPIN’s pipeline from our main paper (Fig. 2). 

**Task Proposal** We query VLM to propose meaningful, diverse tasks, each with a one-sentence task description. Instead of directly querying VLM for task proposal, we employ a hybrid approach that integrates role-play and object-based initialization. In the role-play scenario, we prompt VLM to envision itself as a robotic/human hand working in the scene to perform household tasks. For the object-based initialization, we guide VLM to sequentially focus on each identified object within the scene. When the scene contains more than one identified objects, VLM is instructed to suggest two tasks emphasizing interactions between the manipulating object and any of the detected objects, and an additional task focused solely on the manipulating object. If only one object is detected, VLM is directed to propose a task involving just that object. This strategy guarantees a broad spectrum of task suggestions, ensuring comprehensive object engagement. 

To further tailor the task proposals, we impose specific constraints, directing VLM to consider the practical affordances of objects while encouraging creative assumptions ( _e.g._ , a bowl’s capacity to hold water) and potential interactions ( _e.g._ , transferring water from a cup into a bowl). Additionally, we delineate clear boundaries by excluding tasks that entail the disassembly of objects, functionality tests, or the involvement of imaginary objects, thereby focusing on feasible and meaningful tasks. 

As a concrete example, given the image on the left of Fig. 2, with the manipulating object to be the red can, VLM will propose the following tasks: 

```
“Taskname:CantoBowlTransfer
Description:Pickupthecanandpouritscontentsintothebowl.
Taskname:CanRelocation
Description:Pickupthecanandplaceitinsidethebowl.
Taskname:CanRotation
Description:Rotatethecan90degreesonitsverticalaxis....”
```

**Axes-Constrained Motion Planning through Waypoint** We introduce a novel method to guide VLM to conduct motion planning within a 3D scene based on a proposed task by planning motion waypoints along the manipulating object’s principal axes. More specifically, we define four types of manipulations that VLM can use: 

18 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0019-00.png)


**----- Start of picture text -----**<br>
Diverse scenes<br>Rich diversity of object categories and quantities per scene<br>High variety of perspectives<br>**----- End of picture text -----**<br>


Figure 8: **Dataset statistics.** Our dataset presents 51 scenes—13 from NOCS and 38 captured from varied perspectives—featuring a wide range of object categories, quantities, and a diverse set of tasks and planned trajectories. 

_Rotation type 1:_ Axial rotation. The object rotates around its principle axes. 

_Rotation type 2:_ Rotation relative to the target object. 

- pitch: Tilt similar to pouring water, around a horizontal axis formed by the cross product of the connecting directional vector and the target’s vertical axis. 

- yaw: Horizontal rotation, like a camera panning, around a vertical axis formed by the cross product of the connecting directional vector and the pitch axis. 

- roll: Rotation like drilling a surface, around the connecting directional vector. 

_Translation type 1:_ Defines the goal relative to the target object’s principle axes, with translation values for its [ _x, y, z_ ] axes in centimeters. [0 _,_ 0 _,_ 0] cm indicates the goal is the center of the target object. 

_Translation type 2:_ Sets the goal relative to a directional vector between two reference objects, specifying how far (in cm) object 1 should move towards or away from object 2 along this vector. 

Since VLM inherently lacks the capability to provide 3D coordinates and low-level actions directly [60, 19], our method offers a practical workaround by translating natural language instructions into precise motion waypoints. This approach significantly enhances VLM’s utility in spatial reasoning and manipulation tasks without requiring direct 3D coordinate generation capabilities. Also, the four types of manipulations we defined are both simple and comprehensive, covering a broad spectrum of manipulation tasks. 

As a concrete example, given the image on the left of Fig. 7, with `“Task name: Can to Bowl Transfer”` , VLM will plan as follows: 

```
“TaskName:CantoBowlTransfer
Manipulatingobjidx:3
Interactingobjidx:4
```

```
1.MoveManipulatingObj[3]to[6,0,7]cmrelativetoTargetObj[4]’slocal[x,y,z]axes.
```

```
2.rotate_wref:RotateManipulatingObj[3]relativetoTargetObj[4]around[pitch]axisby[75]
degrees.”
```

In practice, the quality of motion planning by VLMs can be enhanced using various prompting techniques. One such technique is chain-of-thought (CoT) [63], where another LLM guides the VLM to plan each axes-constrained sparse waypoint step by step. 

**Dataset Statistics** See Fig. 8 for statistics and visuals. We standardize all image dimensions by resizing all to 640 _×_ 480. 

**More Qualitative Demonstration** We present more qualitative examples spanning various scenes and tasks, as shown in Fig. 9. 

19 


![](raw/papers/2_LLM_NLP/LLM_paper_list/reasoning/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors_images/nips_2024_SpatialPIN_Enhancing_Spatial_Reasoning_Capabilities_of_Vision-Language_Models_through_Prompting_and_Interacting_3D_Priors.pdf-0020-00.png)


**----- Start of picture text -----**<br>
“Move the red eraser to the left of the blue eraser and erase letters.”<br>“Lift the milk container and deposit it into the trash bin.”<br>“The cup is repositioned under the coffee dispenser to prepare for coffee pouring.”<br>“Cover the pot with the pot lid.”<br>“Insert the toilet brush into the toilet bowl and scrub for cleaning.”<br>Input Planned Task Trajectories<br>**----- End of picture text -----**<br>


Figure 9: **More qualitative examples.** With diverse input scenes and proposed tasks, our framework produces 3D trajectories with geometric awareness that aligns with the task descriptions. _Zoom in for better view._ 

Table 10: **Comparison of task diversity.** We sample 106 proposed tasks for fair comparison with RoboGen and previous RL benchmarks. 

||**Ours**|RoboGen [61]|Behavior-100 [54]|RLbench [28]|MetaWorld [72]|Maniskill2 [23]|
|---|---|---|---|---|---|---|
|Number of Tasks|106|106|100|106|50|20|
|Self-BLEU_↓_|**0.269**|0.284|0.299|0.317|0.322|0.674|
|Embedding Similarity_↓_|**0.154**|0.165|0.210|0.200|0.263|0.194|



## **Experiment on Human Understanding** 

We aim to understand human perception of our generated task executions by asking participants to provide a one-sentence description. We then evaluate the alignment between these descriptions and the ground truth task descriptions proposed by VLM, using OpenCLIP [15, 43]. Table. 9 reveals a high degree of alignment. Intriguingly, it appears humans understand our task executions more accurately than machines do (Table. 7). We hypothesize this discrepancy stems from the limitations of current video understanding models, whereas humans draw on their prior experiences for a deeper comprehension. 

Table 9: Results for machine understanding (generation) on 278 task executions and human understanding, where 25 users write descriptions for 10 tasks. 

||OpenCLIP|_↑_|**Machine**<br>0.636|**Human**<br>0.823|
|---|---|---|---|---|



**Experiment on Task Diversity** We evaluate the diversity of the proposed tasks in terms of semantic meaning using Self-BLEU and the embedding similarity [75, 48, 49] following RoboGen [61], where lower scores mean higher diversity. We also compare with previous reinforcement learning (RL) benchmarks. From Table. 10, ours method generates most diverse tasks as it is open to all scenes with no constraint. 

20 

## **C.3 Trajectory Generation Implementation Details** 

**Trajectory Generation** With the waypoints planned by VLM, we generate the manipulating object’s trajectory using path planning algorithm, specifically rapidly-exploring random tree star (RRT*) [31]. To generate accurate collision-free path, we perform K-means clustering on the point clouds of object 3D model with a high number of clusters, segmenting the object mesh into discrete voxels and treating each voxel as an obstacle. Then, to accurately consider the manipulating object’s dimensions, we grow the size of each voxel by its dimensions. 

**Handling VLM Planning Discrepancies** The waypoints generated by VLM are typically accurate and practical. Nonetheless, there are instances where the waypoints suggested by VLM lead to collisions as determined by the RRT* planner. This discrepancy is less about VLM’s misunderstanding of the objects’ sizes and their spatial relationship and more about the precision level of the waypoints, which may not match the exacting standards of the RRT* planner’s outcomes. To resolve this, we implement Gaussian sampling around the initially planned waypoints whenever a collision is detected. The sampling strategy is guided by a predefined set of geometric rules. In our 3D coordinate system, positive x-axis [1, 0, 0] points right, positive y-axis [0, 1, 0] is away from viewer, positive z-axis [0, 0, 1] is up. For translation type 1, we denote the goal pose relative to the target object’s principle axes as [dx, dy, dz]. For translation type 2, we denote the distance that object 1 moves towards object 2 as dD. The set of geometric rules are as follows: 

if type 1 & [dx, dy, dz] = [0, 0, 0]: sample along [x, y, z] axes elif type 1 & dx = 0 & dy = 0 & dz != 0: sample along [z] axis, zsampled _·_ dz > 1 elif type 1 & dz = 0: sample along [x, y] axes elif type 1 & dz != 0: sample along [x, y, z] axes, zsampled _·_ dz > 1 

if type 2: sample along the connecting directional vector, Dsampled _·_ dD > 1 

**Trajectory Smoothing** Finally, to ensure our trajectory is natural and smooth, we linearly interpolate rotation and interpolate translation using cubic spline. 

## **D Prompt Details** 

We show exact prompts for VLMs for our proposed application: discovering and planning for robotics tasks from a single image. 

## **Understanding Objects by Constraining Prompt.** 

```
Input:
```

```
RGBimage(640,480)=(width,height)withmultipleobjects.
```

```
Yourtaskistoidentifyandobjectsbyprecisecolor,texture,and2Dspatiallocations(inwords).
Donotusevaguephraselikemulti-colored.
```

```
Pleasewriteinthefollowingformat.Donotoutputanythingelse:
Objectidx(actualinteger,startfrom0):xofcolory,texturezatlocationw.
```

## **Task Proposal Prompt.** 

```
Input:
```

`1. RGB image (640, 480) = (width, height) with multiple objects.` 

```
2.Detectedobjectswithindex.
```

```
Youareasinglerobothandworkinginthisimagescenetoperformsimplehouseholdtasks.Tasksmust
bediscoveredfromtheimage.Considerobjects’affordancesandfeelfreetomakeassumptions(e.g.,a
bowlcancontainwater)andinteractionswithotherobjects(e.g.,pouringwaterfromacupintoabowl).
```

```
Tasktypes:
```

`1. Interaction between the manipulating object and one of the detected objects (involve translation, or` 

21 

```
translation+rotation).
```

```
2.Rotatemanipulatingobject(involverotation).
```

```
Strictlyfollowconstraints:
```

`1. Exclude tasks involving disassembly of objects.` 

`2. Exclude tasks involving cleaning or functionality testing.` 

`3. Exclude tasks involving imaginary objects.` 

`4. Manipulating object moves; interacting object static.` 

`5. Assume all objects are rigid, without joints or moveable parts (i.e., cannot deform, disassemble, transform). This applies even to objects that are typically articulated (e.g., laptop).` 

```
Propose3tasks(2interaction,1rotation)formanipulatingObject5.Writeinthefollowingformat.
Donotoutputanythingelse:
```

```
TaskName:xxx
```

```
Manipulatingobjidx:5
Interactingobjidx:obj_idx(actualinteger,ormanipulatingobjidx)
Description:basicdescriptions.
```

## **Coarse 3D Understanding Prompt.** 

```
Inputs:
```

`1. RGB image (640, 480) = (width, height) with multiple objects` 

`2. Detected objects with index.` 

`3. Image scene size.` 

`4. Maximum and minimum width, depth, and height.` 

```
Yourtaskistoidentifythecamerashotangle(horizontal,top-down,bottom-up).Reasonwithrespectto
thevisualcues,theimagescenesize,andmaximumsandminimumsalongeachdimension.Choosehorizontal
ifnotseverelyangled.
```

```
Pleasewriteinthefollowingformat.Beconcise.Donotoutputanythingelse:
Visualcuesreasoning:...
Spatialdatareasoning:...
Conclusion:horizontal/top-down/bottom-up.
```

## **Image and Spatial Context Understanding Prompt.** 

```
Inputs:
```

`1. RGB image (640, 480) = (width, height) with multiple objects and their visualized local axes (x red, y green, z blue).` 

`2. Detected objects with index.` 

`3. For each detected object, its 3D center, local xyz-axes, size, and spatial relationship relative to other objects.` 

```
The3DcoordinatesystemoftheimageisincentimetersandfollowsBlender.Positivex-axis[1,0,
0]right,positivey-axis[0,1,0]awayfromviewer,positivez-axis[0,0,1]up.Positiverotationis
counter-clockwisearoundallaxes.
```

```
Yourtaskistolearnthespatialcontext.Donotoutput.
```

## **Motion Planning Prompt.** 

```
Inputs:
```

`1. RGB image (640, 480) = (width, height) with multiple objects and their visualized local axes (x red, y green, z blue).` 

`2. Detected objects with index.` 

`3. Simple household tasks and descriptions to be performed by a single robot hand.` 

22 

```
Yourgoalistoplanfine-grainedmotionsforthemanipulatingobjecttocompletethetasksusingfour
manipulations,explainedasfollows:
```

```
Rotation:
```

```
rotate_self:Axialrotation.Theobjectrotatesarounditslocal[x/y/z]axisby[degrees].
rotate_wref:Rotationrelativetothetargetobject:
```

```
-pitch:Tiltsimilartopouringwater,aroundahorizontalaxisformedbythecrossproductofthe
connectingdirectionalvectorandthetarget’sz-axis.
```

```
-yaw:Horizontalrotation,likeacamerapanning,aroundaverticalaxisformedbythecrossproductof
theconnectingdirectionalvectorandthepitchaxis.
```

```
-roll:Rotationlikeadrillenteringasurface,aroundtheconnectingdirectionalvector.
Thedegreescanbespecifiedintwoways:
```

```
-Exact[degrees].Positivevaluesrotatethemanipulatingobjecttowardsthetargetobject.
-Fixed_towards/fixed_back.’fixed_towards’orientstheobjecttowardsthetarget,mimickingactions
likepouring(pitch),facing(yaw),ordrillinginto(yaw+roll)thetarget.’fixed_back’reversesthis
alignment.
```

```
Translation:
```

```
translate_tar_obj:Definesthegoalrelativetothetargetobject’slocalaxes,withtranslationvalues
forits[local_x,local_y,local_z]axesincentimeters.[0,0,0]cmindicatesthegoalisthecenterof
thetargetobject.
```

```
translate_direc_axis:Setsthegoalrelativetoadirectionalvectorbetweentworeferenceobjects,
specifyinghowfar(incm)object1shouldmovetowardsorawayfromobject2alongthisvector(positive
closer,negativeaway).Objectindicesmustdiffer,andifonereferenceobjectisthemanipulating
object,itscurrentlocationisused.
```

```
Strictlyfollowcaveats:
```

```
1.Applyrotate_wrefthoughtfullyandsequentiallyarounddifferentaxesasneeded.
```

```
2.Usetheprovidedspatialinformationandimageeffectivelyforunderstandingandplanningwithinthe
3Dscene.
```

```
3.Combinecommonphysicalunderstandingwiththescene’sspatialdetails(likerelativepositionsand
sizesofobjects)forstrategicplanning.
```

```
4.Rememberthatobjects’localaxes’positivedirectionsmightrequireusingnegativevaluesinrotation
andtranslationforauthenticmotionplanning.
```

```
Planasbelow.Fillinobj_idxbasedonthetasks.
rotate_self:RotateManipulatingObject[obj_idx]arounditslocalaxis[x/y/z]by[degrees].
rotate_wref:RotateManipulatingObject[obj_idx]relativetoTargetObject[target_obj_idx]around
[pitch/yaw/roll]axisby[degrees/fixed_towards/fixed_back].
translate_tar_obj:MoveManipulatingObject[obj_idx]to[a,b,c]cmrelativetoTargetObject
[target_obj_idx]’slocal[x,y,z]axes.
translate_direc_axis:MoveManipulatingObject[obj_idx][a]cmalongthedirectionalvectorfrom
ReferenceObject[ref_obj_1_idx]toReferenceObject[ref_obj_2_idx].
Herearesomefullexamples.Pleasewriteinthefollowingformat.Donotoutputanythingelse:
TaskCategory:Bearrotation
Description:Rotatethetoybear90degreesonitsverticalaxis.
MotionPlanning:
Manipulatingobjidx:bear_idx(actualinteger)
Interactingobjidx:bear_idx(actualinteger)
1.rotate_self:RotateManipulatingObject[bear_idx]arounditslocalaxis[z]by[90]degrees.
TaskName:Cupcontenttransfer
Description:Pickupthemugandpouritscontentsintothebowl.
MotionPlanning:
Manipulatingobjidx:cup_idx(actualinteger)
Interactingobjidx:bowl_idx(actualinteger)
1.translate_tar_obj:MoveManipulatingObject[cup_idx]to[5,-7,5]cmrelativetoTargetObject
[bowl_idx]’slocal[x,y,z]axes.
```

23 

```
2.rotate_wref:RotateManipulatingObject[obj_idx]relativetoTargetObject[bowl_obj_idx]around
[pitch]axisby[fixed_towards].
```

```
TaskName:Screwdriverpenetration
Description:Useascrewdrivertopenetrateanavocado.
MotionPlanning:
Manipulatingobjidx:screw_idx(actualinteger)
Interactingobjidx:avocado_idx(actualinteger)
1.translate_tar_obj:MoveManipulatingObject[screw_idx]to[-5,-5,0]cmrelativetoTargetObject
[avocado_idx]’slocal[x,y,z]axes.
2.rotate_wref:RotateManipulatingObject[screw_idx]relativetoTargetObject[avocado_idx]around
[yaw]axisby[fixed_towards].
```

```
3.rotate_wref:RotateManipulatingObject[screw_idx]relativetoTargetObject[avocado_idx]around
[roll]axisby[360]degrees.
```

24 

